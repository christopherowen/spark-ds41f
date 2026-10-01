#!/usr/bin/env python3
"""Latency under mixed traffic: decoding streams, long prompts and short requests on a fixed schedule.

usage: mixed_latency.py BASE_URL [--rounds R]   (prints one JSON line per round and a summary)

Per round: four streams with distinct prompts decode 384 tokens each from t=0; a cold
~4000-token real-text prompt (16 tokens out) arrives at 1.0 s and another at 4.0 s; short
requests (32 tokens out) arrive at 2.0, 2.5, 3.0 and 5.0 s. Measured: the short requests' time
to first token, the long prompts' time to first token, the decoding streams' gaps between
streamed chunks (p50, p95, p99: prefill steps show up as long gaps), and their throughput.
Temperature 0, thinking off, fresh cache salts.
"""
import glob
import json
import os
import statistics
import sys
import threading
import time
import urllib.request
import uuid

BASE = sys.argv[1].rstrip("/")
ROUNDS = int(sys.argv[sys.argv.index("--rounds") + 1]) if "--rounds" in sys.argv else 3
HERE = os.path.dirname(os.path.abspath(__file__))
STREAMS = [
    "Explain how a hash map handles collisions, with an example in Python.",
    "Describe a walk through a spice market in Marrakech in the morning.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Compare TCP and UDP for real-time multiplayer games and when to use each.",
]
SHORT = ["What is the capital of Australia?", "Name three prime numbers above fifty.",
         "Give one synonym for quick.", "How many legs does a spider have?"]
source = ""
for path in sorted(glob.glob(os.path.join(HERE, "*.py"))):
    source += open(path).read() + "\n"
    if len(source) > 40000:
        break
LONG = [source[:12400], source[12400:24800]]
with urllib.request.urlopen(BASE + "/v1/models", timeout=60) as response:
    MODEL = json.load(response)["data"][0]["id"]


def stream(prompt, tokens, out, ignore_eos=False):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": tokens,
            "temperature": 0, "stream": True, "ignore_eos": ignore_eos, "cache_salt": uuid.uuid4().hex,
            "chat_template_kwargs": {"thinking": False}}
    request = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    out["start"] = time.monotonic()
    out["chunks"] = []
    with urllib.request.urlopen(request, timeout=900) as response:
        for raw in response:
            line = raw.decode().strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                chunk = json.loads(line[6:])
                if chunk.get("choices") and chunk["choices"][0].get("delta", {}).get("content"):
                    out["chunks"].append(time.monotonic())


def pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))] if values else float("nan")


summary = []
for r in range(ROUNDS):
    plan = [(0.0, p, 384, "stream", True) for p in STREAMS]
    plan += [(1.0, LONG[0], 16, "long", False), (4.0, LONG[1], 16, "long", False)]
    plan += [(t, SHORT[i], 32, "short", False) for i, t in enumerate((2.0, 2.5, 3.0, 5.0))]
    results, threads = [], []
    t0 = time.monotonic()
    for at, prompt, tokens, kind, ignore in sorted(plan, key=lambda p: p[0]):
        delay = at - (time.monotonic() - t0)
        if delay > 0:
            time.sleep(delay)
        out = {"kind": kind}
        results.append(out)
        thread = threading.Thread(target=stream, args=(prompt, tokens, out, ignore))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    gaps, rates = [], []
    for o in results:
        if o["kind"] == "stream" and len(o["chunks"]) > 1:
            gaps += [1000 * (b - a) for a, b in zip(o["chunks"], o["chunks"][1:])]
            rates.append(len(o["chunks"]) / (o["chunks"][-1] - o["chunks"][0]))
    ttft = {k: [1000 * (o["chunks"][0] - o["start"]) for o in results if o["kind"] == k and o["chunks"]]
            for k in ("short", "long")}
    row = {"round": r, "short_ttft_ms_median": round(statistics.median(ttft["short"]), 1),
           "short_ttft_ms_max": round(max(ttft["short"]), 1),
           "long_ttft_ms_mean": round(statistics.mean(ttft["long"]), 1),
           "stream_gap_ms_p50": round(pct(gaps, 0.5), 1), "stream_gap_ms_p95": round(pct(gaps, 0.95), 1),
           "stream_gap_ms_p99": round(pct(gaps, 0.99), 1),
           "stream_chunks_per_s": round(statistics.mean(rates), 2)}
    summary.append(row)
    print(json.dumps(row), flush=True)
    time.sleep(2)
keys = [k for k in summary[0] if k != "round"]
print(json.dumps({"summary": "mixed traffic latency", "rounds": ROUNDS,
                  **{k: round(statistics.mean(s[k] for s in summary), 1) for k in keys}}), flush=True)
