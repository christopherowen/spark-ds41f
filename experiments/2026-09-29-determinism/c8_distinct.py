#!/usr/bin/env python3
"""Eight concurrent streams with eight different prompts: decode throughput without shared routing.

usage: c8_distinct.py BASE_URL [--samples N] [--tokens T]   (prints one JSON line per sample and a summary)

The spark3 bench's eight-stream decode sends eight copies of one prompt; with
batch invariance the copies write the same text, route to the same experts
and decode faster for that reason alone. Here each stream has its own prompt
(fresh cache salts), thinking off, temperature 0. Per sample: generated
tokens over the window from the first stream's first token to the last
stream's last token, the per-stream decode rate, and time to first token.
"""
import json
import statistics
import sys
import threading
import time
import urllib.request
import uuid

BASE = sys.argv[1].rstrip("/")


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


SAMPLES, TOKENS = int(arg("--samples", "3")), int(arg("--tokens", "256"))
PROMPTS = [
    "Explain how a hash map handles collisions, with an example in Python.",
    "Describe a walk through a spice market in Marrakech in the morning.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Compare TCP and UDP for real-time multiplayer games and when to use each.",
    "Give a detailed recipe for lentil soup with timings and substitutions.",
    "Summarize the causes and consequences of the printing press in Europe.",
    "Explain the difference between supervised and unsupervised learning to a student.",
    "Plan a three-day hiking trip in the Alps with packing advice.",
]
with urllib.request.urlopen(BASE + "/v1/models", timeout=60) as response:
    MODEL = json.load(response)["data"][0]["id"]


def stream(prompt, out):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": TOKENS,
            "temperature": 0, "stream": True, "ignore_eos": True, "cache_salt": uuid.uuid4().hex,
            "chat_template_kwargs": {"thinking": False}, "stream_options": {"include_usage": True}}
    request = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    start = time.monotonic()
    first = last = None
    tokens = 0
    with urllib.request.urlopen(request, timeout=900) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                tokens = chunk["usage"]["completion_tokens"]
            if chunk.get("choices") and chunk["choices"][0].get("delta", {}).get("content"):
                now = time.monotonic()
                first = first or now
                last = now
    out.update(start=start, first=first, last=last, tokens=tokens)


rates = []
for sample in range(SAMPLES):
    results = [{} for _ in PROMPTS]
    threads = [threading.Thread(target=stream, args=(p, r)) for p, r in zip(PROMPTS, results)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    window = max(r["last"] for r in results) - min(r["first"] for r in results)
    total = sum(r["tokens"] for r in results)
    per_stream = [(r["tokens"] - 1) / (r["last"] - r["first"]) for r in results if r["last"] > r["first"]]
    rate = total / window
    rates.append(rate)
    print(json.dumps({"sample": sample, "tps": round(rate, 2), "tokens": total,
                      "per_stream_tps": round(statistics.mean(per_stream), 2),
                      "ttft_s": round(statistics.mean(r["first"] - r["start"] for r in results), 3)}), flush=True)
mean = statistics.mean(rates)
half = 1.96 * statistics.stdev(rates) / len(rates) ** 0.5 if len(rates) > 1 else 0.0
print(json.dumps({"summary": "c8 distinct prompts", "tps_mean": round(mean, 2),
                  "ci95_pct": round(100 * half / mean, 2), "samples": len(rates)}), flush=True)
