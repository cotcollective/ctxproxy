#!/usr/bin/env python3
"""
ctxproxy — Context-aware proxy for Ollama Cloud.
Sits between any OpenAI-compatible client and Ollama Cloud, adding:
  - Response caching (exact match, TTL)
  - Token usage monitoring + pre-400 alert
  - Adaptive sliding window compression
  - Content-addressed prefix caching
  - Streaming support
"""

import asyncio, json, logging, os, time, hashlib, sqlite3, argparse, re
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

logging.basicConfig(level=logging.INFO, format="[ctxproxy] %(levelname)s %(message)s")
log = logging.getLogger("ctxproxy")


# ── Token estimation ────────────────────────────────────────────────────
def estimate_tokens(messages, tools=None):
    """Rough token estimation — better than len//4 for structured content."""
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


# ── Adaptive compression ────────────────────────────────────────────────
def compress_messages(messages, estimated, limit, protect_last_n=10):
    """Adaptive sliding window — keeps system + last N + semantically similar messages."""
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


# ── Prefix helpers ──────────────────────────────────────────────────────
def hash_prefix(model, prefix_messages):
    raw = json.dumps({"model": model, "prefix": prefix_messages}, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


# ── SQLite cache ─────────────────────────────────────────────────────────
class ResponseCache:
    """SQLite-backed cache for responses, token tracking, prefix KV, and compression logs."""

    def __init__(self, db_path):
        self.db_path = db_path
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE IF NOT EXISTS response_cache (key TEXT PRIMARY KEY, response TEXT NOT NULL, created_at REAL NOT NULL, ttl REAL NOT NULL DEFAULT 300)")
        conn.execute("CREATE TABLE IF NOT EXISTS token_usage (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp REAL NOT NULL, model TEXT NOT NULL, prompt_tokens INTEGER DEFAULT 0, completion_tokens INTEGER DEFAULT 0, total_tokens INTEGER DEFAULT 0, estimated_tokens INTEGER DEFAULT 0)")
        conn.execute("CREATE TABLE IF NOT EXISTS cache_stats (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp REAL NOT NULL, event TEXT NOT NULL, detail TEXT DEFAULT '')")
        conn.execute("CREATE TABLE IF NOT EXISTS prefix_cache (key TEXT PRIMARY KEY, kv_state TEXT NOT NULL, model TEXT NOT NULL, created_at REAL NOT NULL, ttl REAL NOT NULL DEFAULT 3600, access_count INTEGER DEFAULT 0, last_access REAL DEFAULT 0)")
        conn.execute("CREATE TABLE IF NOT EXISTS compression_log (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp REAL NOT NULL, model TEXT NOT NULL, before_tokens INTEGER NOT NULL, after_tokens INTEGER NOT NULL, messages_before INTEGER NOT NULL, messages_after INTEGER NOT NULL, strategy TEXT NOT NULL)")
        conn.commit()
        conn.close()

    def _make_key(self, model, messages, tools):
        raw = json.dumps({"model": model, "messages": messages, "tools": tools}, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, model, messages, tools):
        key = self._make_key(model, messages, tools)
        conn = sqlite3.connect(str(self.db_path))
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
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("INSERT OR REPLACE INTO response_cache VALUES (?, ?, ?, ?)", (key, json.dumps(response), time.time(), ttl))
        conn.commit()
        conn.close()
        log.info("CACHE SET key=%s...", key[:12])

    def log_usage(self, model, prompt_tokens, completion_tokens, estimated):
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("INSERT INTO token_usage (timestamp, model, prompt_tokens, completion_tokens, total_tokens, estimated_tokens) VALUES (?, ?, ?, ?, ?, ?)",
                     (time.time(), model, prompt_tokens, completion_tokens, prompt_tokens + completion_tokens, estimated))
        conn.commit()
        conn.close()

    def log_event(self, event, detail=""):
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("INSERT INTO cache_stats (timestamp, event, detail) VALUES (?, ?, ?)", (time.time(), event, detail))
        conn.commit()
        conn.close()

    def get_prefix(self, model, prefix_hash):
        conn = sqlite3.connect(str(self.db_path))
        row = conn.execute("SELECT kv_state, created_at, ttl FROM prefix_cache WHERE key = ? AND model = ?", (prefix_hash, model)).fetchone()
        if row:
            kv_state, created_at, ttl = row
            if time.time() - created_at < ttl:
                conn.execute("UPDATE prefix_cache SET access_count = access_count + 1, last_access = ? WHERE key = ?", (time.time(), prefix_hash))
                conn.commit()
                conn.close()
                log.info("PREFIX HIT %s...", prefix_hash[:12])
                return kv_state
        conn.close()
        return None

    def set_prefix(self, model, prefix_hash, kv_state):
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("INSERT OR REPLACE INTO prefix_cache VALUES (?, ?, ?, ?, ?, 0, ?)",
                     (prefix_hash, kv_state, model, time.time(), 3600, time.time()))
        conn.commit()
        conn.close()
        log.info("PREFIX SET %s...", prefix_hash[:12])

    def log_compression(self, model, before, after, msgs_before, msgs_after, strategy):
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("INSERT INTO compression_log VALUES (?, ?, ?, ?, ?, ?, ?)",
                     (time.time(), model, before, after, msgs_before, msgs_after, strategy))
        conn.commit()
        conn.close()


# ── Proxy server ────────────────────────────────────────────────────────
class CtxProxy:
    def __init__(self, upstream, port):
        self.upstream = upstream.rstrip("/")
        self.port = port
        self.cache = ResponseCache(CACHE_DB)
        self.app = web.Application()
        self.app.router.add_post("/v1/chat/completions", self.handle_chat)
        self.app.router.add_get("/health", self.handle_health)
        self.app.router.add_get("/stats", self.handle_stats)

    def _get_key(self):
        key = os.environ.get("OLLAMA_API_KEY", "")
        if key:
            return key
        kf = "/tmp/ollama_key.txt"
        if os.path.exists(kf):
            with open(kf) as f:
                return f.read().strip()
        return ""

    async def _forward(self, body, key):
        """Forward request upstream. Returns (status, json_data, stream_resp, session)."""
        is_stream = body.get("stream", False)
        session = aiohttp.ClientSession()
        resp = await session.post(
            f"{self.upstream}/chat/completions",
            json=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
            timeout=aiohttp.ClientTimeout(total=300),
        )
        if is_stream:
            return resp.status, None, resp, session
        data = await resp.json()
        await session.close()
        return resp.status, data, None, None

    async def handle_health(self, request):
        return web.json_response({"status": "ok", "cache_db": str(CACHE_DB)})

    async def handle_stats(self, request):
        conn = sqlite3.connect(str(self.cache.db_path))
        usage = conn.execute("SELECT COUNT(*), COALESCE(SUM(total_tokens),0), COALESCE(SUM(estimated_tokens),0) FROM token_usage").fetchone()
        cache_count = conn.execute("SELECT COUNT(*) FROM response_cache").fetchone()[0]
        prefix_count = conn.execute("SELECT COUNT(*) FROM prefix_cache").fetchone()[0]
        comp_count = conn.execute("SELECT COUNT(*) FROM compression_log").fetchone()[0]
        recent = conn.execute("SELECT event, detail, timestamp FROM cache_stats ORDER BY id DESC LIMIT 10").fetchall()
        conn.close()
        return web.json_response({
            "calls": usage[0], "total_tokens": usage[1], "estimated_tokens": usage[2],
            "cache_entries": cache_count, "prefix_entries": prefix_count,
            "compressions": comp_count,
            "recent_events": [{"event": e[0], "detail": e[1], "time": e[2]} for e in recent],
        })

    async def handle_chat(self, request):
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        model = body.get("model", "unknown")
        messages = body.get("messages", [])
        tools = body.get("tools")
        is_stream = body.get("stream", False)

        estimated = estimate_tokens(messages, tools)
        log.info("REQUEST model=%s messages=%d estimated=%d tokens stream=%s",
                 model, len(messages), estimated, is_stream)

        # Adaptive compression
        compressed = False
        if estimated > CONTEXT_LIMIT * COMPRESS_AT_PCT and len(messages) > 3:
            comp_msgs, strategy, new_est = compress_messages(messages, estimated, CONTEXT_LIMIT)
            if strategy != "none":
                log.info("COMPRESSED: %d->%d tokens (%s)", estimated, new_est, strategy)
                self.cache.log_compression(model, estimated, new_est, len(messages), len(comp_msgs), strategy)
                messages = comp_msgs
                estimated = new_est
                compressed = True
                body["messages"] = messages

        # Hard limit check
        if estimated > CONTEXT_LIMIT:
            pct = estimated / CONTEXT_LIMIT * 100
            detail = f"estimated={estimated} limit={CONTEXT_LIMIT} ({pct:.0f}%)"
            log.warning("CONTEXT OVERFLOW: %s", detail)
            self.cache.log_event("context_overflow", detail)
            return web.json_response(
                {"error": f"Bad Request (ref: ctxproxy-{hashlib.sha256(detail.encode()).hexdigest()[:12]})"},
                status=400,
            )

        # Prefix caching (observational)
        if not is_stream and len(messages) > 2:
            prefix_msgs = [m for m in messages if m.get("role") == "system"]
            if prefix_msgs and estimate_tokens(prefix_msgs) > 500:
                ph = hash_prefix(model, prefix_msgs)
                if self.cache.get_prefix(model, ph):
                    self.cache.log_event("prefix_hit", f"model={model}")
                else:
                    self.cache.set_prefix(model, ph, json.dumps(prefix_msgs))

        # Response cache (non-streaming only)
        if not is_stream:
            cached = self.cache.get(model, messages, tools)
            if cached:
                self.cache.log_event("cache_hit", f"model={model}")
                return web.json_response(cached)

        # Forward
        key = self._get_key()
        if not key:
            return web.json_response({"error": "Unauthorized"}, status=401)

        status, data, stream_resp, stream_session = await self._forward(body, key)

        # Handle streaming
        if stream_resp is not None:
            response = web.StreamResponse(
                status=status,
                headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache", "Connection": "keep-alive"},
            )
            await response.prepare(request)
            pt, ct = 0, 0
            try:
                async for line in stream_resp.content:
                    if line:
                        decoded = line.decode("utf-8", errors="replace")
                        await response.write(line)
                        if decoded.startswith("data: ") and "usage" in decoded:
                            try:
                                u = json.loads(decoded[6:]).get("usage", {})
                                if u:
                                    pt, ct = u.get("prompt_tokens", pt), u.get("completion_tokens", ct)
                            except json.JSONDecodeError:
                                pass
            except Exception as e:
                log.error("Stream error: %s", e)
            finally:
                await stream_resp.release()
                await stream_session.close()
            if pt or ct:
                self.cache.log_usage(model, pt, ct, estimated)
            return response

        # Handle non-streaming
        if status == 200 and isinstance(data, dict):
            usage = data.get("usage", {})
            pt = usage.get("prompt_tokens", 0)
            ct = usage.get("completion_tokens", 0)
            self.cache.log_usage(model, pt, ct, estimated)
            if pt > CONTEXT_LIMIT * WARN_THRESHOLD:
                pct = pt / CONTEXT_LIMIT * 100
                log.warning("HIGH CONTEXT: %d tokens (%.0f%%)", pt, pct)
                self.cache.log_event("high_context", f"{pt}/{CONTEXT_LIMIT} ({pct:.0f}%)")
            self.cache.set(model, messages, tools, data)
        elif status != 200:
            self.cache.log_event("upstream_error", f"{status}: {str(data)[:200]}")

        return web.json_response(data, status=status)

    async def start(self):
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", self.port)
        await site.start()
        log.info("ctxproxy on :%d → %s", self.port, self.upstream)
        log.info("  Context limit: %d | Compression at %.0f%% | Prefix cache: ON", CONTEXT_LIMIT, COMPRESS_AT_PCT * 100)


def main():
    parser = argparse.ArgumentParser(description="ctxproxy — Context-aware proxy for Ollama Cloud")
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
