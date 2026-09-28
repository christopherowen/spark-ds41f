#!/usr/bin/env python3
"""Time the first requests after a fresh boot (usage: first_request.py BASE_URL)."""
import json, sys, time, urllib.request
base = sys.argv[1] if len(sys.argv) > 1 else "http://10.0.1.71:8000"
model = json.load(urllib.request.urlopen(base + "/v1/models"))["data"][0]["id"]
for i in range(4):
    body = {"model": model, "messages": [{"role": "user", "content": f"Write a haiku about rivers ({i})."}],
            "max_tokens": 48, "min_tokens": 48, "ignore_eos": True, "temperature": 0, "stream": True,
            "chat_template_kwargs": {"thinking": False}}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.perf_counter(); first = None
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if line.startswith("data:") and '"content"' in line and first is None:
                first = time.perf_counter() - t
    total = time.perf_counter() - t
    print(f"request {i}: first token {first * 1000:.0f} ms, total {total * 1000:.0f} ms", flush=True)
