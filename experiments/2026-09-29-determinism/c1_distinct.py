#!/usr/bin/env python3
"""Single-stream decode over twenty-four distinct prompts, one at a time: speed averaged over texts.

usage: c1_distinct.py BASE_URL [--tokens T] [--rounds R] [--prompts N]   (prints one JSON line per request and a summary)

A batch-invariant arm writes one text per prompt, so its draft acceptance, and with it its
tokens per second, is a property of that text; one or two prompts say little about speed.
Each prompt (fresh cache salt, thinking off, temperature 0, T tokens, ignore_eos) streams alone;
per request: decode tokens per second from the first to the last token. Summary: mean, the 95%
interval over requests, and the mean of the per-prompt medians when R > 1. --prompts N takes N of the
prompts at an even stride (every other one for 12), keeping the mix of prose, code and JSON.
"""
import json
import statistics
import sys
import time
import urllib.request
import uuid

BASE = sys.argv[1].rstrip("/")


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


TOKENS, ROUNDS = int(arg("--tokens", "256")), int(arg("--rounds", "1"))
PROMPTS = [
    "Explain how a hash map handles collisions, with an example in Python.",
    "Describe a walk through a spice market in Marrakech in the morning.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Compare TCP and UDP for real-time multiplayer games and when to use each.",
    "Give a detailed recipe for lentil soup with timings and substitutions.",
    "Summarize the causes and consequences of the printing press in Europe.",
    "Explain the difference between supervised and unsupervised learning to a student.",
    "Plan a three-day hiking trip in the Alps with packing advice.",
    "Return a JSON object describing five fictional cities with name, population and founding year.",
    "Write a SQL query that finds the top three customers by revenue per region, and explain it.",
    "List the planets of the solar system with one interesting fact about each.",
    "Draft a polite email asking a landlord to fix a leaking kitchen tap.",
    "Explain how public-key cryptography lets two strangers agree on a secret.",
    "Describe the life cycle of a star from nebula to remnant.",
    "Write a Python function that merges overlapping intervals, with tests.",
    "Give advice for a first-time manager running their first one-on-one meetings.",
    "Summarize the plot of a heist film you invent, in five acts.",
    "Explain why the sky is blue and sunsets are red.",
    "Return a JSON array of six books with title, author, year and genre.",
    "Compare electric and petrol cars on cost, emissions and maintenance.",
    "Write a limerick about a cat who learns to code, then explain the meter.",
    "Explain how a bill becomes a law in a parliamentary system.",
    "Describe how to train for a first half marathon over twelve weeks.",
    "Explain gradient descent with a worked numerical example.",
]
COUNT = int(arg("--prompts", str(len(PROMPTS))))
PROMPTS = PROMPTS[:: max(1, len(PROMPTS) // COUNT)][:COUNT]
with urllib.request.urlopen(BASE + "/v1/models", timeout=60) as response:
    MODEL = json.load(response)["data"][0]["id"]


def one(prompt):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": TOKENS,
            "temperature": 0, "stream": True, "ignore_eos": True, "cache_salt": uuid.uuid4().hex,
            "chat_template_kwargs": {"thinking": False}, "stream_options": {"include_usage": True}}
    request = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
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
    return (tokens - 1) / (last - first)


rates = {i: [] for i in range(len(PROMPTS))}
for r in range(ROUNDS):
    for i, prompt in enumerate(PROMPTS):
        tps = one(prompt)
        rates[i].append(tps)
        print(json.dumps({"round": r, "prompt": i, "tps": round(tps, 2)}), flush=True)
all_rates = [x for v in rates.values() for x in v]
mean = statistics.mean(all_rates)
half = 1.96 * statistics.stdev(all_rates) / len(all_rates) ** 0.5
print(json.dumps({"summary": "c1 distinct prompts", "tps_mean": round(mean, 2), "ci95_pct": round(100 * half / mean, 2),
                  "requests": len(all_rates),
                  "per_prompt_median_mean": round(statistics.mean(statistics.median(v) for v in rates.values()), 2)}),
      flush=True)
