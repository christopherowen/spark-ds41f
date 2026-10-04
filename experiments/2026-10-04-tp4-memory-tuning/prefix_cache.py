#!/usr/bin/env python3
"""Prefix-cache behaviour at long context: footprint, hits, divergence and capacity.

Sequential greedy requests of real text (the bench's standard-library source
text), each about --tokens long. For every request the server's own counters
give the prompt tokens it served from the prefix cache
(vllm:prompt_tokens_cached), and vllm:kv_cache_usage_perc is sampled twice a
second while the request runs, so the peak shows the running footprint.

  cold       a new prompt P
  repeat     P again: everything up to the last partial block should hit
  diverge    P's first 75% with a different tail: the hit is the last retained
             sliding-window checkpoint at or before the divergence
  continue   P, its answer and a follow-up question: the hit resumes at the end
             of the previous turn
  capacity   --fill distinct prompts, then P again: a hit means the cache kept P

With --capacity N it instead loads N distinct prompts and recalls them newest
first until one misses, so a small KV pool shows how many prompts the cache
holds. A recall hit adds no blocks, so recalling never evicts an older prompt.

Writes one JSON line per request.
"""

import argparse
import importlib.machinery
import importlib.util
import json
import re
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
loader = importlib.machinery.SourceFileLoader("spark3_bench", str(ROOT / "bin/spark3"))
spec = importlib.util.spec_from_loader(loader.name, loader)
spark3 = importlib.util.module_from_spec(spec)
loader.exec_module(spark3)

QUESTION = "\n\n# Question: in one sentence, what does the last function above do?\n# Answer:"


def metric(url, name):
    """The sum of every sample of one metric on the server's /metrics page."""
    with urllib.request.urlopen(f"{url}/metrics", timeout=30) as response:
        text = response.read().decode()
    pattern = re.compile(rf"^{re.escape(name)}(?:{{[^}}]*}})? ([0-9.eE+-]+)$", re.M)
    return sum(float(value) for value in pattern.findall(text))


def complete(url, prompt, max_tokens):
    payload = {"model": "deepseek-v4.1-flash", "prompt": prompt, "max_tokens": max_tokens, "temperature": 0}
    req = urllib.request.Request(f"{url}/v1/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as response:
        body = json.loads(response.read())
    return body["choices"][0]["text"], body["usage"]["prompt_tokens"]


def measured(url, step, prompt, max_tokens=1):
    cached_before = metric(url, "vllm:prompt_tokens_cached_total")
    peak, done = [0.0], threading.Event()

    def sample():
        while not done.wait(0.5):
            try:
                peak[0] = max(peak[0], metric(url, "vllm:kv_cache_usage_perc"))
            except OSError:
                pass

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    started = time.monotonic()
    try:
        text, prompt_tokens = complete(url, prompt, max_tokens)
    finally:
        done.set()
        sampler.join()
    seconds = time.monotonic() - started
    time.sleep(1)  # the API server records a request's counters as its output is processed
    cached = int(metric(url, "vllm:prompt_tokens_cached_total") - cached_before)
    row = {"step": step, "prompt_tokens": prompt_tokens, "cached_tokens": cached,
           "computed_tokens": prompt_tokens - cached, "seconds": round(seconds, 1),
           "peak_kv_usage": round(peak[0], 4)}
    print(json.dumps(row), flush=True)
    return text, row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("--tokens", type=int, default=200_000, help="approximate prompt length")
    parser.add_argument("--fill", type=int, default=10, help="distinct prompts sent before P returns")
    parser.add_argument("--capacity", type=int, default=0,
                        help="instead: send this many distinct prompts, then re-send them newest first "
                             "until one misses; the hits are how many the cache held")
    args = parser.parse_args()
    if args.capacity:
        prompts = [spark3.source_text(args.tokens, 9500 + index) + QUESTION for index in range(args.capacity)]
        for index, prompt in enumerate(prompts):
            measured(args.url, f"load-{index + 1}", prompt)
        held = 0
        for index in reversed(range(len(prompts))):
            _, row = measured(args.url, f"recall-{index + 1}", prompts[index])
            if row["cached_tokens"] < row["prompt_tokens"] // 2:
                break
            held += 1
        print(json.dumps({"step": "held", "prompts": held, "of": len(prompts)}), flush=True)
        return
    body = spark3.source_text(args.tokens, 9100)
    prompt = body + QUESTION
    answer, _ = measured(args.url, "cold", prompt, max_tokens=32)
    measured(args.url, "repeat", prompt)
    cut = body.rfind("\n", 0, int(len(body) * 0.75)) + 1
    tail = spark3.source_text(args.tokens // 4, 9200)
    measured(args.url, "diverge", body[:cut] + tail + QUESTION)
    measured(args.url, "continue", prompt + answer + "\n# Follow-up: and what does it return?\n# Answer:")
    for index in range(args.fill):
        measured(args.url, f"fill-{index + 1}", spark3.source_text(args.tokens, 9300 + index) + QUESTION)
    measured(args.url, "capacity", prompt)


if __name__ == "__main__":
    main()
