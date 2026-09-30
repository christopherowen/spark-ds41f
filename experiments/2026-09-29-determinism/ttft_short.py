#!/usr/bin/env python3
"""Latency of short prefills, the only batches the GEMV lookup changes.

usage: ttft_short.py BASE_URL [--reps 7]

The lookup change only moves eager batches whose row count is not a capture
size (at most 48 rows): short prompts' prefills. One request at a time, a
fresh cache salt each (no prefix reuse), thinking off, max_tokens 1: the
request time is the prefill plus one sampling step. Prompts of 1-40 words
give about 10-50 prompt tokens. Prints, per prompt, the prompt tokens and the
median and minimum request time over the repeats, and one JSON line per
prompt.
"""
import json
import statistics
import sys
import time
import urllib.request
import uuid

BASE = sys.argv[1].rstrip("/")
REPS = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 7
WORDS = ("amber basil cedar delta ember fjord grove harbor indigo juniper kestrel lantern meadow "
         "nectar orchid pebble quartz raven saffron thistle umber velvet willow xenon yarrow zephyr "
         "anchor birch cobalt dune eagle fern garnet heron iris jade kelp lotus maple nutmeg").split()
with urllib.request.urlopen(BASE + "/v1/models", timeout=60) as response:
    MODEL = json.load(response)["data"][0]["id"]


def request(prompt):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 1,
            "temperature": 0, "cache_salt": uuid.uuid4().hex, "chat_template_kwargs": {"thinking": False}}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as response:
        data = json.load(response)
    return time.perf_counter() - start, data["usage"]["prompt_tokens"]


request("warm up")
for words in (1, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40):
    prompt = "Repeat: " + " ".join(WORDS[:words])
    times, tokens = [], None
    for _ in range(REPS):
        elapsed, tokens = request(prompt)
        times.append(1000 * elapsed)
    print(json.dumps({"words": words, "prompt_tokens": tokens, "median_ms": round(statistics.median(times), 2),
                      "min_ms": round(min(times), 2), "reps": REPS}), flush=True)
