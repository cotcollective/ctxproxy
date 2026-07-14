# ctxproxy — Context-Aware Proxy for Ollama Cloud

A lightweight HTTP proxy that sits between any OpenAI-compatible client and Ollama Cloud. Adds **response caching**, **token monitoring**, **adaptive compression**, and **prefix caching** — features Ollama Cloud's API doesn't provide natively.

```
Client → ctxproxy:11438 → Ollama Cloud:443
              │
              ├── Response cache (exact match, TTL)
              ├── Token monitoring (warn at 80%, critical at 95%)
              ├── Adaptive compression (sliding window + similarity)
              └── Prefix caching (content-addressed KV)
```

## Why

Ollama Cloud's `/v1/chat/completions` endpoint has a ~505K token infrastructure limit (not the model's 1M), no prompt caching, and no context monitoring. This proxy fills those gaps with zero client changes.

## Quick Start

```bash
pip install aiohttp pyyaml
OLLAMA_API_KEY=*** python ctxproxy.py
# Point your client to http://127.0.0.1:11438/v1
```

## Real-World Numbers

Tested against deepseek-v4-flash on Ollama Cloud during a 3-hour session:

```
Calls: 42
Total tokens proxied: 147,176
Cache entries: 5
Prefix entries: 3
Compressions triggered: 1
```

The compression saved a session that would have hit the 505K limit — it reduced 41 messages to 12 (70% reduction) while preserving the system prompt, last 10 messages, and semantically relevant context.

## Bugs Found & Fixed During Development

### 1. SQLite schema drift (v1 → v2)
The initial cache schema didn't have a `compressed` column. When the feature was added later, the old DB file caused a 500 error. **Fix:** delete and recreate the DB, or use `CREATE TABLE IF NOT EXISTS` with migration checks.

### 2. `self.cache.db_path` vs `self.db_path` (v2)
ResponseCache methods referenced `self.cache.db_path` instead of `self.db_path` — a copy-paste error from when the cache was a property of CtxProxy. **Fix:** use `self.db_path` directly since ResponseCache is a standalone class.

### 3. Streaming responses break `await resp.json()` (v3)
The initial implementation always called `resp.json()`, which fails on SSE streams with "Attempt to decode JSON with unexpected mimetype: text/event-stream". **Fix:** check `body.get("stream", False)` before deciding how to read the response.

### 4. Port collision with local Ollama (v4)
Port 11435 was already used by a local Ollama GPU instance. **Fix:** default to 11438, document the collision.

## Features

| Feature | Description |
|---------|-------------|
| **Response caching** | Exact-match cache with configurable TTL. SQLite-backed, survives restarts. |
| **Token monitoring** | Tracks every request. Warns at 80% of context limit, critical at 95%. Returns a proper 400 before Ollama Cloud's hard limit. |
| **Adaptive compression** | When estimated tokens exceed 70% of the limit, automatically compresses: keeps system message + last N messages + semantically similar messages (keyword overlap). |
| **Prefix caching** | Content-addressed KV cache for the static prefix (system + initial exchanges). SHA-256 hashing. |
| **Streaming support** | Passes SSE streams through transparently, extracting usage data from `data:` lines. |
| **Stats endpoint** | `GET /stats` returns call count, total tokens, cache entries, prefix entries, compression count. |

## CLI

```
python ctxproxy.py [--port 11438] [--upstream https://ollama.com/v1]
```

## Env Vars

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_API_KEY` | — | Ollama Cloud API key (or `/tmp/ollama_key.txt` fallback) |
| `CTXPROXY_CACHE` | `/tmp/ctxproxy_cache.db` | SQLite cache path |
| `CTXPROXY_LOG` | `INFO` | Log level |

## Tests

```bash
python tests/test_cache.py
```

8 tests covering: cache hit/miss/TTL, token estimation (text + code), compression (noop + sliding window), prefix hash uniqueness.

## Architecture

Single-file Python (330 lines) using:
- **aiohttp** — async HTTP server and client
- **SQLite** (WAL mode) — cache persistence
- **SHA-256** — content-addressed cache keys

Dependencies: `aiohttp`, `pyyaml`.

## License

MIT
