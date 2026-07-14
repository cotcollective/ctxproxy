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

logging.basicConfig(level=logging.INFO, format="[ctxproxy] %(levelname)s %(message)s")
log = logging.getLogger("ctxproxy")

class CtxProxy:
    def __init__(self, upstream, port):
        self.upstream = upstream.rstrip("/")
        self.port = port
        self.app = web.Application()
        self.app.router.add_post("/v1/chat/completions", self.handle_chat)
        self.app.router.add_get("/health", self.handle_health)

    def _get_key(self):
        return os.environ.get("OLLAMA_API_KEY", "")

    async def handle_health(self, request):
        return web.json_response({"status": "ok"})

    async def handle_chat(self, request):
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"error": "Invalid JSON"}, status=400)

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
