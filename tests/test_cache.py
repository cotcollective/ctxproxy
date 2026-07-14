"""Tests for ctxproxy — response cache, token estimation, compression, prefix hashing."""
import sys, os, json, time, hashlib, sqlite3, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ctxproxy import ResponseCache, estimate_tokens, compress_messages

CACHE_DB = tempfile.mktemp(suffix=".db")
os.environ["CTXPROXY_CACHE"] = CACHE_DB

def test_cache_hit():
    cache = ResponseCache(CACHE_DB)
    model, msgs = "test-model", [{"role": "user", "content": "hi"}]
    resp = {"id": "test", "choices": [{"message": {"content": "hello"}}]}
    cache.set(model, msgs, None, resp)
    cached = cache.get(model, msgs, None)
    assert cached is not None, "Should return a hit"
    assert cached["choices"][0]["message"]["content"] == "hello"
    print("  PASS test_cache_hit")

def test_cache_miss():
    cache = ResponseCache(CACHE_DB)
    cached = cache.get("test-model", [{"role": "user", "content": "different"}], None)
    assert cached is None, "Should miss for different content"
    print("  PASS test_cache_miss")

def test_cache_ttl():
    cache = ResponseCache(CACHE_DB)
    model, msgs = "test-model", [{"role": "user", "content": "ttl-test"}]
    cache.set(model, msgs, None, {"id": "ttl"}, ttl=0)
    assert cache.get(model, msgs, None) is None, "Should expire with ttl=0"
    print("  PASS test_cache_ttl")

def test_estimate_tokens():
    est = estimate_tokens([{"role": "user", "content": "hello world"}])
    assert 0 < est < 100, f"Expected <100, got {est}"
    print(f"  PASS test_estimate_tokens ({est})")

def test_estimate_tokens_code():
    est = estimate_tokens([{"role": "user", "content": 'def f(): return {"a": 1}'}])
    assert est > 0, f"Expected >0, got {est}"
    print(f"  PASS test_estimate_tokens_code ({est})")

def test_compress_noop():
    msgs = [{"role": "system", "content": "test"}, {"role": "user", "content": "hi"}]
    c, s, _ = compress_messages(msgs, 10, 1000)
    assert s == "none", f"Expected no compression, got {s}"
    assert len(c) == len(msgs)
    print("  PASS test_compress_noop")

def test_compress_sliding_window():
    msgs = [{"role": "system", "content": "test"}]
    for i in range(20):
        msgs.append({"role": "user", "content": f"message {i}"})
        msgs.append({"role": "assistant", "content": f"response {i}"})
    est = estimate_tokens(msgs)
    c, s, _ = compress_messages(msgs, est, est * 0.5)
    assert s != "none", f"Expected compression, got {s}"
    assert len(c) < len(msgs), f"Expected fewer messages ({len(c)} vs {len(msgs)})"
    print(f"  PASS test_compress_sliding_window ({len(msgs)}->{len(c)})")

def test_prefix_hash():
    def _hash(model, prefix):
        raw = json.dumps({"model": model, "prefix": prefix}, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()
    h1 = _hash("model-a", [{"role": "system", "content": "hello"}])
    h2 = _hash("model-b", [{"role": "system", "content": "hello"}])
    h3 = _hash("model-a", [{"role": "system", "content": "world"}])
    assert h1 != h2, "Different models should differ"
    assert h1 != h3, "Different prefixes should differ"
    print("  PASS test_prefix_hash")

def test_sse_stream_parsing():
    """Simulate SSE stream parsing — the bug that broke resp.json()."""
    # Simulate what the proxy receives from Ollama Cloud
    chunks = [
        b'data: {"choices":[{"delta":{"content":"hello"}}],"usage":{"prompt_tokens":10,"completion_tokens":5}}\n\n',
        b'data: {"choices":[{"delta":{"content":" world"}}]}\n\n',
        b'data: [DONE]\n\n',
    ]
    full = b"".join(chunks)
    lines = full.decode("utf-8", errors="replace").split("\n")

    pt, ct = 0, 0
    content = ""
    for line in lines:
        if line.startswith("data: ") and line != "data: [DONE]":
            try:
                data = json.loads(line[6:])
                delta = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                content += delta
                usage = data.get("usage", {})
                if usage:
                    pt = usage.get("prompt_tokens", pt)
                    ct = usage.get("completion_tokens", ct)
            except json.JSONDecodeError:
                pass

    assert content == "hello world", f"Expected 'hello world', got {content!r}"
    assert pt == 10, f"Expected prompt_tokens=10, got {pt}"
    assert ct == 5, f"Expected completion_tokens=5, got {ct}"
    print(f"  PASS test_sse_stream_parsing (content={content!r}, pt={pt}, ct={ct})")

if __name__ == "__main__":
    tests = [test_cache_hit, test_cache_miss, test_cache_ttl,
             test_estimate_tokens, test_estimate_tokens_code,
             test_compress_noop, test_compress_sliding_window,
             test_prefix_hash, test_sse_stream_parsing]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
    sys.exit(0 if passed == len(tests) else 1)
