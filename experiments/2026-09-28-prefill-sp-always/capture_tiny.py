#!/usr/bin/env python3
"""Record a few profiled cold prefills of the bench's smallest real-text size.

usage: capture_tiny.py BASE_URL   (a *-profile config must be serving)

Each prompt matches the prefill suite's 64 size (about 73 tokens), with its own
cache salt, one output token. Client-side time to first token is printed.
"""
import importlib.machinery
import importlib.util
import json
import sys
import time
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
loader = importlib.machinery.SourceFileLoader("spark3", str(ROOT / "bin/spark3"))
spec = importlib.util.spec_from_loader("spark3", loader)
spark3 = importlib.util.module_from_spec(spec)
loader.exec_module(spark3)
BASE = sys.argv[1].rstrip("/")


def post(path, body=None):
    request = urllib.request.Request(BASE + path, data=json.dumps(body or {}).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=1800) as response:
        return response.read()


def tiny(model, seed):
    return {"model": model, "prompt": spark3.source_text(int(64 * 0.97), seed) + "\n\nReply with the word ok.",
            "max_tokens": 1, "temperature": 0, "cache_salt": uuid.uuid4().hex}


with urllib.request.urlopen(BASE + "/v1/models", timeout=60) as response:
    model = json.load(response)["data"][0]["id"]
for seed in range(100, 106):
    post("/v1/completions", tiny(model, seed))
post("/start_profile")
for seed in range(200, 206):
    started = time.perf_counter()
    usage = json.loads(post("/v1/completions", tiny(model, seed)))["usage"]
    print(f"tiny prefill {1000 * (time.perf_counter() - started):.1f} ms, {usage['prompt_tokens']} tokens", flush=True)
post("/stop_profile")
