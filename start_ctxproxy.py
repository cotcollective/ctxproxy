#!/usr/bin/env python3
"""Starter that reads OLLAMA_API_KEY from ~/.hermes/.env or /tmp/ollama_key.txt."""
import os, sys
for src in [os.path.expanduser("~/.hermes/.env"), "/tmp/ollama_key.txt"]:
    if os.path.exists(src):
        with open(src) as f:
            for line in f:
                line = line.strip()
                if "OLLAMA_API_KEY" in line and "=" in line:
                    os.environ["OLLAMA_API_KEY"] = line.split("=", 1)[1].strip().strip("\"'")
                    break
        if os.environ.get("OLLAMA_API_KEY"):
            break
sys.path.insert(0, os.path.dirname(__file__))
import ctxproxy
ctxproxy.main()
