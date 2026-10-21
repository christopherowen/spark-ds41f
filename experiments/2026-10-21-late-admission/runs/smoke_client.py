#!/usr/bin/env python3
"""Smoke client for the dummy-weight server of smoke.sh.

usage: smoke_client.py URL OUT.json

- greedy outputs of eight prompts, one at a time (compared between servers);
- four closed-loop streams for 30 s over distinct prompts with every chunk's arrival kept,
  decomposed by analyze_trace.py (the steps each request waited that did not carry it).
"""
import hashlib
import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_trace import analyze, summary_line  # noqa: E402

URL, OUT = sys.argv[1].rstrip("/"), sys.argv[2]
PROMPTS = [f"Item {i}: describe the history of the number {i * 37 % 101} in a few sentences."
           for i in range(64)]


def post(body):
    req = urllib.request.Request(URL + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=300)


def stream(prompt):
    body = {"model": "tiny", "prompt": prompt, "max_tokens": 64, "min_tokens": 64, "ignore_eos": True,
            "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True, "continuous_usage_stats": True}}
    start, chunks, prompt_tokens = time.monotonic(), [], None
    with post(body) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            usage = json.loads(line[5:]).get("usage") or {}
            prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
            chunks.append([time.monotonic(), usage.get("completion_tokens", 0), 0])
    return {"start": start, "end": time.monotonic(), "prompt_tokens": prompt_tokens, "chunks": chunks}


def metrics():
    with urllib.request.urlopen(URL + "/metrics", timeout=10) as response:
        text = response.read().decode()
    out = {}
    for line in text.splitlines():
        if line.startswith("vllm:") and "_bucket" not in line:
            name = line.split("{")[0].split(" ")[0]
            try:
                out[name] = out.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
            except ValueError:
                pass
    return out


greedy = []
for prompt in PROMPTS[:8]:
    with post({"model": "tiny", "prompt": prompt, "max_tokens": 32, "temperature": 0}) as resp:
        greedy.append(json.load(resp)["choices"][0]["text"])

lock, nxt, results = threading.Lock(), [8], []
t0 = time.monotonic()
before = metrics()


def worker(stream_id):
    while time.monotonic() - t0 < 30:
        with lock:
            i = nxt[0]
            nxt[0] += 1
        r = stream(PROMPTS[i % len(PROMPTS)])
        r.update(phase="c4", stream=stream_id, index=i, key=f"p{i:03d}")
        with lock:
            results.append(r)


threads = [threading.Thread(target=worker, args=(s,)) for s in range(4)]
for t in threads:
    t.start()
for t in threads:
    t.join()
report = {"requests": results,
          "phases": {"c4": {"start": t0, "metrics_before": before, "metrics_after": metrics()}}}
out = analyze(report)
summary = summary_line(out)["c4"]
json.dump({"greedy": greedy, "c4": summary}, open(OUT, "w"), indent=1)
digest = hashlib.sha256(json.dumps(greedy).encode()).hexdigest()[:8]
print(json.dumps({"greedy_sha256": digest, "c4": summary}))
