#!/usr/bin/env python3
"""ctxproxy — Context-aware proxy for Ollama Cloud."""

import asyncio, json, logging, os, time, hashlib, sqlite3, argparse
from pathlib import Path
import aiohttp
from aiohttp import web

DEFAULT_PORT = 11438
DEFAULT_UPSTREAM = "https://ollama.com/v1"
CACHE_DB = Path("/tmp/ctxproxy_cache.db")
CONTEXT_LIMIT = 505_000
WARN_THRESHOLD = 0.80
CRIT_THRESHOLD = 0.95
COMPRESS_AT_PCT = 0.70
PROTECT_LAST_N = 10



import re


def compress_messages(messages, estimated, limit, protect_last_n=10):
    """Adaptive sliding window compression — keeps system + last N + similar messages."""
    if not messages or estimated < limit * 0.70:
        return messages, "none", estimated
    system_msgs = [m for m in messages if m.get("role") == "system"]
    non_system = [m for m in messages if m.get("role") != "system"]
    if len(non_system) <= protect_last_n + 2:
        return messages, "none", estimated
    tail = non_system[-protect_last_n:]
    last_user = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user = m.get("content", "")
            if isinstance(last_user, str):
                break
    similar = set()
    if last_user:
        last_words = set(re.findall(r'\w{4,}', last_user.lower()))
        for i, m in enumerate(non_system[:-protect_last_n]):
            content = m.get("content", "")
            if isinstance(content, str):
                words = set(re.findall(r'\w{4,}', content.lower()))
                if len(last_words & words) >= 2:
                    similar.add(i)
    compressed = list(system_msgs)
    for i in sorted(similar):
        if i < len(non_system) - protect_last_n:
            compressed.append(non_system[i])
    removed = len(non_system) - len(tail) - len(similar)
    if removed > 3:
        compressed.append({
            "role": "system",
            "content": f"[CONTEXT COMPRESSION: {removed} earlier messages removed. Only last {protect_last_n} messages and relevant ones preserved.]"
        })
    compressed.extend(tail)
    new_est = estimate_tokens(compressed)
    return compressed, "sliding_window", new_est


def estimate_tokens(messages, tools=None):
    """Rough token estimation for structured content."""
    total = 0
    for msg in (messages or []):
        content = msg.get("content", "")
        if isinstance(content, str):
            specials = sum(1 for c in content if c in "{}[]()<>\"'=;:/\\")
            ratio = 2.5 if specials > len(content) * 0.3 else 4.0
            total += int(len(content) / ratio)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    total += int(len(block["text"]) / 4)
        total += 4
    if tools:
        total += len(tools) * 15
    total += 20
    return total


CONTEXT_LIMIT = 505_000

logging.basicConfig(level=logging.INFO, format="[ctxproxy] %(levelname)s %(message)s")
log = logging.getLogger("ctxproxy")


class ResponseCache:
    """SQLite-backed exact-match response cache with TTL."""
    def __init__(self, db_path):
        self.db_path = db_path
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE IF NOT EXISTS response_cache (key TEXT PRIMARY KEY, response TEXT NOT NULL, created_at REAL NOT NULL, ttl REAL NOT NULL DEFAULT 300)")
        conn.execute("CREATE TABLE IF NOT EXISTS token_usage (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp REAL NOT NULL, model TEXT NOT NULL, prompt_tokens INTEGER DEFAULT 0, completion_tokens INTEGER DEFAULT 0, total_tokens INTEGER DEFAULT 0, estimated_tokens INTEGER DEFAULT 0)")
        conn.execute("CREATE TABLE IF NOT EXISTS cache_stats (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp REAL NOT NULL, event TEXT NOT NULL, detail TEXT DEFAULT '')")
        conn.commit()
        conn.close()

    def _make_key(self, model, messages, tools):
        raw = json.dumps({"model": model, "messages": messages, "tools": tools}, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, model, messages, tools):
        key = self._make_key(model, messages, tools)
        conn = sqlite3.connect(str(self.cache.db_path))
        row = conn.execute("SELECT response, created_at, ttl FROM response_cache WHERE key = ?", (key,)).fetchone()
        conn.close()
        if row:
            resp_json, created_at, ttl = row
            if time.time() - created_at < ttl:
                log.info("CACHE HIT key=%s...", key[:12])
                return json.loads(resp_json)
        return None

    def set(self, model, messages, tools, response, ttl=120):
        key = self._make_key(model, messages, tools)
        conn = sqlite3.connect(str(self.cache.db_path))
        conn.execute("INSERT OR REPLACE INTO response_cache VALUES (?, ?, ?, ?)", (key, json.dumps(response), time.time(), ttl))
        conn.commit()
        conn.close()
        log.info("CACHE SET key=%s...", key[:12])

    def log_usage(self, model, prompt_tokens, completion_tokens, estimated):
        conn = sqlite3.connect(str(self.cache.db_path))
        conn.execute("INSERT INTO token_usage (timestamp, model, prompt_tokens, completion_tokens, total_tokens, estimated_tokens) VALUES (?, ?, ?, ?, ?, ?)",
                     (time.time(), model, prompt_tokens, completion_tokens, prompt_tokens + completion_tokens, estimated))
        conn.commit()
        conn.close()

    def log_event(self, event, detail=""):
        conn = sqlite3.connect(str(self.cache.db_path))
        conn.execute("INSERT INTO cache_stats (timestamp, event, detail) VALUES (?, ?, ?)", (time.time(), event, detail))
        conn.commit()
        conn.close()

class CtxProxy:
    def __init__(self, upstream, port):
        self.upstream = upstream.rstrip("/")
        self.port = port
        self.app = web.Application()
        self.cache = ResponseCache(CACHE_DB)
        self.app.router.add_post("/v1/chat/completions", self.handle_chat)
        self.app.router.add_get("/health", self.handle_health)
        self.app.router.add_get("/stats", self.handle_stats)

    def _get_key(self):
        return os.environ.get("OLLAMA_API_KEY", "")

    async def handle_health(self, request):
        return web.json_response({"status": "ok", "cache_db": str(CACHE_DB)})

    async def handle_stats(self, request):
        conn = sqlite3.connect(str(self.cache.db_path))
        usage = conn.execute("SELECT COUNT(*), COALESCE(SUM(total_tokens),0), COALESCE(SUM(estimated_tokens),0) FROM token_usage").fetchone()
        cache_count = conn.execute("SELECT COUNT(*) FROM response_cache").fetchone()[0]
        recent = conn.execute("SELECT event, detail, timestamp FROM cache_stats ORDER BY id DESC LIMIT 10").fetchall()
        conn.close()
        return web.json_response({
            "calls": usage[0], "total_tokens": usage[1], "estimated_tokens": usage[2],
            "cache_entries": cache_count,
            "recent_events": [{"event": e[0], "detail": e[1], "time": e[2]} for e in recent],
        })

    async def handle_chat(self, request):
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        messages = body.get('messages', [])
        tools = body.get('tools')
        estimated = estimate_tokens(messages, tools)
        log.info("REQUEST model=%s messages=%d estimated=%d tokens",
                 body.get('model','?'), len(body.get('messages',[])), estimated)

        compressed = False
        if estimated > CONTEXT_LIMIT * COMPRESS_AT_PCT:
            comp_msgs, strategy, new_est = compress_messages(messages, estimated, CONTEXT_LIMIT)
            if strategy != "none":
                log.info("COMPRESSED: %d->%d tokens (%s)", estimated, new_est, strategy)
                messages = comp_msgs
                estimated = new_est
                compressed = True
                body["messages"] = messages

        if estimated > CONTEXT_LIMIT:
            pct = estimated / CONTEXT_LIMIT * 100
            detail = f"estimated={estimated} limit={CONTEXT_LIMIT} ({pct:.0f}%)"
            log.warning("CONTEXT OVERFLOW: %s", detail)
            self.cache.log_event("context_overflow", detail)
            return web.json_response(
                {"error": f"Bad Request (ref: ctxproxy-{hashlib.sha256(detail.encode()).hexdigest()[:12]})"},
                status=400,
            )

        cached = self.cache.get(body.get('model','?'), messages, tools)
        if cached:
            return web.json_response(cached)

        key = self._get_key()
        if not key:
            return web.json_response({"error": "Unauthorized"}, status=401)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.upstream}/chat/completions",
                json=body,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
                timeout=aiohttp.ClientTimeout(total=300),
            ) as resp:
                data = await resp.json()
                if resp.status == 200:
                    usage = data.get("usage", {})
                    pt = usage.get("prompt_tokens", 0)
                    ct = usage.get("completion_tokens", 0)
                    self.cache.log_usage(body.get('model','?'), pt, ct, estimated)
                    if pt > CONTEXT_LIMIT * WARN_THRESHOLD:
                        pct = pt / CONTEXT_LIMIT * 100
                        log.warning("HIGH CONTEXT: %d tokens (%.0f%%)", pt, pct)
                        self.cache.log_event("high_context", f"{pt}/{CONTEXT_LIMIT} ({pct:.0f}%)")
                    self.cache.set(body.get('model','?'), messages, tools, data)
                return web.json_response(data, status=resp.status)

    async def start(self):
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", self.port)
        await site.start()
        log.info("ctxproxy on :%d → %s", self.port, self.upstream)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--upstream", type=str, default=DEFAULT_UPSTREAM)
    args = parser.parse_args()
    proxy = CtxProxy(args.upstream, args.port)
    async def _start():
        await proxy.start()
        while True:
            await asyncio.sleep(3600)
    try:
        asyncio.run(_start())
    except KeyboardInterrupt:
        log.info("Shutdown")

if __name__ == "__main__":
    main()
