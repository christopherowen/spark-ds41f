#!/usr/bin/env python3
"""Head-of-line latency: short requests that arrive while long prompts prefill.

usage: hol_latency.py BASE_URL [--rounds R] [--long-tokens N]   (one JSON line per request, a summary)

Per round:
- behind-one: a cold ~N-token prompt (default 65536, 16 tokens out) starts at t=0; short requests
  (32 tokens out) arrive at 2, 4, 6 and 8 s. With one prefill lane a short request waits for
  the long prompt's whole prefill; with two it can share a step.
- four-long: four cold ~N/2-token prompts at once (16 tokens out): first-come-first-served
  prefill gives the first an early first token and the last a late one; fair sharing gives
  all four a late one.
Measured: time to first token. Temperature 0, thinking off, fresh cache salts; the long
prompts are distinct slices of this repository's source text, trimmed with /tokenize.
"""
import json
import statistics
import sys
import threading
import time
import urllib.request
import uuid
from pathlib import Path

BASE = sys.argv[1].rstrip("/")


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


ROUNDS, LONG = int(arg("--rounds", "2")), int(arg("--long-tokens", "65536"))
ROOT = Path(__file__).resolve().parents[2]
SHORT = ["What is the capital of Australia?", "Name three prime numbers above fifty.",
         "Give one synonym for quick.", "How many legs does a spider have?"]
with urllib.request.urlopen(BASE + "/v1/models", timeout=60) as response:
    MODEL = json.load(response)["data"][0]["id"]


def post(path, body):
    request = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=1800) as response:
        return json.load(response)


def source_text():
    text = []
    for pattern in ("bin/spark3", "scripts/*.py", "docs/*.md", "experiments/*/*.py", "experiments/*/README.md"):
        for path in sorted(ROOT.glob(pattern)):
            text.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n\n".join(text)


def trimmed(text, tokens):
    """A prefix of TEXT close to TOKENS prompt tokens (chat template included)."""
    low, high = 0, len(text)
    for _ in range(12):
        middle = (low + high) // 2
        count = post("/tokenize", {"model": MODEL,
                                   "messages": [{"role": "user", "content": text[:middle]}]})["count"]
        if count < tokens:
            low = middle
        else:
            high = middle
    return text[:low]


def stream(prompt, tokens, out):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": tokens,
            "temperature": 0, "stream": True, "cache_salt": uuid.uuid4().hex,
            "chat_template_kwargs": {"thinking": False}}
    request = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    out["start"] = time.monotonic()
    with urllib.request.urlopen(request, timeout=1800) as response:
        for raw in response:
            line = raw.decode().strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                chunk = json.loads(line[6:])
                if chunk.get("choices") and chunk["choices"][0].get("delta", {}).get("content"):
                    out.setdefault("first", time.monotonic())
    out["end"] = time.monotonic()


def launch(schedule):
    """schedule: [(start seconds, prompt, tokens, name)] -> {name: timings}."""
    results, threads, t0 = {}, [], time.monotonic()
    for start, prompt, tokens, name in schedule:
        delay = start - (time.monotonic() - t0)
        if delay > 0:
            time.sleep(delay)
        results[name] = {}
        thread = threading.Thread(target=stream, args=(prompt, tokens, results[name]))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    return {name: round(r["first"] - r["start"], 3) for name, r in results.items() if "first" in r}


text = source_text()
step = len(text) // 6
longs = [trimmed(text[i * step:] + text[:i * step], LONG if i == 0 else LONG // 2) for i in range(5)]
summary = {"short": [], "long": [], "four_mean": [], "four_last": []}
for r in range(ROUNDS):
    ttft = launch([(0, longs[0], 16, "long")] +
                  [(2 + 2 * i, SHORT[i], 32, f"short{i}") for i in range(4)])
    print(json.dumps({"round": r, "scenario": "behind-one", "ttft_s": ttft}), flush=True)
    summary["long"].append(ttft["long"])
    summary["short"] += [v for k, v in ttft.items() if k.startswith("short")]
    ttft = launch([(0, longs[1 + i], 16, f"long{i}") for i in range(4)])
    print(json.dumps({"round": r, "scenario": "four-long", "ttft_s": ttft}), flush=True)
    summary["four_mean"].append(statistics.mean(ttft.values()))
    summary["four_last"].append(max(ttft.values()))
print(json.dumps({"summary": "head-of-line latency", "rounds": ROUNDS, "long_tokens": LONG,
                  "short_ttft_s_median": round(statistics.median(summary["short"]), 3),
                  "short_ttft_s_max": round(max(summary["short"]), 3),
                  "long_ttft_s_mean": round(statistics.mean(summary["long"]), 3),
                  "four_long_ttft_s_mean": round(statistics.mean(summary["four_mean"]), 3),
                  "four_long_ttft_s_last": round(statistics.mean(summary["four_last"]), 3)}), flush=True)
