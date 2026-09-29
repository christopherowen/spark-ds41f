#!/usr/bin/env python3
"""Time, then profile, one 4,096-token prefill chunk at several context depths.

usage: capture_depth.py BASE_URL [--profile]   (a DeepSeek V4.1 arm must be serving)

One real-text document is tokenized once. For each round and depth D, a request
with a fresh cache salt prefills the document's first D - 4096 tokens, and a
second request with the same salt sends the first D tokens: it reuses the
cached prefix and prefills exactly one 4,096-token chunk (the scheduler's
chunk budget) at context depth D. Every warm request of a round runs before
its measured requests, so the profiler window of the last round (--profile)
holds only the measured chunks, in depth order. Server-side prefill time comes
from vllm:request_prefill_time_seconds, so the frontend's handling of a long
token list does not count.
"""
import importlib.machinery
import importlib.util
import json
import re
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
PROFILE = "--profile" in sys.argv[2:]
CHUNK = 4096
DEPTHS = (8192, 65536, 131072, 200704)
ROUNDS = 3


def post(path, body=None):
    request = urllib.request.Request(BASE + path, data=json.dumps(body or {}).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=3600) as response:
        return json.loads(response.read() or b"null")


def prefill_seconds_sum():
    with urllib.request.urlopen(BASE + "/metrics", timeout=60) as response:
        text = response.read().decode()
    match = re.search(r"^vllm:request_prefill_time_seconds_sum\{[^}]*\} (\S+)$", text, re.M)
    return float(match.group(1))


def complete(model, ids, salt):
    return post("/v1/completions", {"model": model, "prompt": ids, "max_tokens": 1,
                                    "temperature": 0, "cache_salt": salt})


with urllib.request.urlopen(BASE + "/v1/models", timeout=60) as response:
    model = json.load(response)["data"][0]["id"]
tokens = post("/tokenize", {"model": model, "prompt": spark3.source_text(215_000, 29)})["tokens"]
assert len(tokens) >= DEPTHS[-1], f"document has only {len(tokens)} tokens"
print(f"document {len(tokens)} tokens", flush=True)
for round_ in range(ROUNDS):
    profiled = PROFILE and round_ == ROUNDS - 1
    salts = {depth: uuid.uuid4().hex for depth in DEPTHS}
    for depth in DEPTHS:
        complete(model, tokens[: depth - CHUNK], salts[depth])
    if profiled:
        post("/start_profile")
    for depth in DEPTHS:
        before = prefill_seconds_sum()
        started = time.perf_counter()
        usage = complete(model, tokens[:depth], salts[depth])["usage"]
        client = time.perf_counter() - started
        time.sleep(1)  # the histogram is observed as the output loop finishes the request
        server = prefill_seconds_sum() - before
        print(json.dumps({"round": round_, "profiled": profiled, "depth": depth,
                          "prompt_tokens": usage["prompt_tokens"],
                          "prefill_ms": round(1000 * server, 1),
                          "client_ms": round(1000 * client, 1)}), flush=True)
    if profiled:
        post("/stop_profile")
