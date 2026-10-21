#!/usr/bin/env python3
"""Traced short-prompt load for the time-to-first-token decomposition.

usage: ttft_trace.py [URL] --output OUT.json [--repo PATH] [--workloads FILE] [--analyze]

Default prompts are servebench's 256 (thinking off, greedy, a fresh cache salt and exactly
256 output tokens per request, as servebench's closed loop sends them). Phases:
- isolated: twelve prompts sent one at a time to an idle engine, the server's counters
  read around each, so every isolated request has its own server-side first-token
  latency, queue time and prefill time;
- c4 for 120 s and c8 for 60 s: closed loops over distinct prompts.

With --workloads FILE (a JSON object with `set-a` and `set-b`, each request bodies by
key), the bodies are sent as given (thinking off): isolated, then set-a cycled on four
streams for 30 s and set-b in order on four streams for 90 s.

Every streamed chunk's arrival and cumulative completion tokens are kept (vLLM's
continuous usage), so engine step ends can be rebuilt from the streams' chunks
(analyze_trace.py). The server's queue, prefill and first-token histograms are read
before and after each phase. Stdlib only; refuses to start on a busy server. With
--analyze it prints analyze_trace.py's decomposition and a last JSON line with a
"summary" key (the lab's table reads it), so it runs as a lab extra (URL first).
"""

import argparse
import json
import random
import sys
import threading
import time
import urllib.request
import uuid
from pathlib import Path

HISTOGRAMS = (
    "vllm:time_to_first_token_seconds",
    "vllm:request_queue_time_seconds",
    "vllm:request_prefill_time_seconds",
    "vllm:inter_token_latency_seconds",
    "vllm:iteration_tokens_total",
    "vllm:e2e_request_latency_seconds",
)
COUNTERS = (
    "vllm:request_success_total",
    "vllm:generation_tokens_total",
    "vllm:prompt_tokens_total",
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
)
MAX_TOKENS = 256


def metrics(base):
    """Sums and counts of the histograms, their buckets, and the counters (labels summed)."""
    with urllib.request.urlopen(base + "/metrics", timeout=10) as response:
        text = response.read().decode()
    out = {"t": time.monotonic()}
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        try:
            value = float(line.rsplit(" ", 1)[1])
        except ValueError:
            continue
        for h in HISTOGRAMS:
            if name in (h + "_sum", h + "_count"):
                out[name] = out.get(name, 0.0) + value
            elif name == h + "_bucket":
                le = line.split('le="', 1)[1].split('"', 1)[0]
                key = f"{h}_bucket:{le}"
                out[key] = out.get(key, 0.0) + value
        if name in COUNTERS:
            out[name] = out.get(name, 0.0) + value
    return out


def busy(base):
    m = metrics(base)
    return m.get("vllm:num_requests_running", 0) + m.get("vllm:num_requests_waiting", 0)


def wait_idle(base, settle=0.5, limit=120):
    deadline = time.monotonic() + limit
    while busy(base):
        if time.monotonic() > deadline:
            raise RuntimeError("server did not become idle")
        time.sleep(0.05)
    time.sleep(settle)


def stream_one(url, model, body):
    req = dict(body)
    req["model"] = model
    req["stream"] = True
    req["stream_options"] = {"include_usage": True, "continuous_usage_stats": True}
    if "cache_salt" in req:
        req["cache_salt"] = uuid.uuid4().hex
    kwargs = dict(req.get("chat_template_kwargs") or {})
    kwargs["thinking"] = False
    kwargs["enable_thinking"] = False
    req["chat_template_kwargs"] = kwargs
    data = json.dumps(req).encode()
    r = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    chunks = []
    prompt_tokens = None
    completion = 0
    t0 = time.monotonic()
    with urllib.request.urlopen(r, timeout=600) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            now = time.monotonic()
            chunk = json.loads(payload)
            piece = ""
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                piece += delta.get("content") or delta.get("reasoning_content") or ""
            usage = chunk.get("usage")
            if usage:
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                completion = usage.get("completion_tokens", completion)
            chunks.append([now, completion, len(piece)])
    return {"start": t0, "end": time.monotonic(), "prompt_tokens": prompt_tokens,
            "completion_tokens": completion, "chunks": chunks}


def closed_loop(url, model, items, streams, seconds, cycle, phase):
    lock = threading.Lock()
    nxt = [0]
    results = []
    t0 = time.monotonic()

    def take():
        with lock:
            i = nxt[0]
            if not cycle and i >= len(items):
                return None
            nxt[0] += 1
            return i, items[i % len(items)]

    def worker(stream):
        while time.monotonic() - t0 < seconds:
            got = take()
            if got is None:
                return
            i, (key, body) = got
            r = stream_one(url, model, body)
            r.update(phase=phase, stream=stream, index=i, key=key)
            with lock:
                results.append(r)

    threads = [threading.Thread(target=worker, args=(s,)) for s in range(streams)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def default_items(repo):
    sys.path.insert(0, str(repo))
    from servebench.prompts import default_prompts

    prompts = default_prompts()
    random.Random(0).shuffle(prompts)
    return [(p["id"], {"messages": [{"role": "user", "content": p["text"]}],
                       "max_tokens": MAX_TOKENS, "temperature": 0, "cache_salt": "",
                       "ignore_eos": True, "min_tokens": MAX_TOKENS}) for p in prompts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base", nargs="?", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--workloads", help="external set-a/set-b request bodies")
    ap.add_argument("--output", required=True)
    ap.add_argument("--isolated", type=int, default=12)
    ap.add_argument("--analyze", action="store_true", help="print the decomposition and a summary line")
    a = ap.parse_args()
    base = a.base.rstrip("/")
    url = base + "/v1/chat/completions"
    if a.workloads:
        w = json.load(open(a.workloads))
        set_a, set_b = list(w["set-a"].items()), list(w["set-b"].items())
        warm, isolated_items = set_a, (set_a + set_b)[:a.isolated]
        phases = (("steady", set_a, 4, 30, True), ("sustained", set_b, 4, 90, False))
    else:
        items = default_items(a.repo)
        warm, isolated_items = items[252:], items[240:240 + a.isolated]
        phases = (("c4", items[:120], 4, 120, True), ("c8", items[120:240], 8, 60, True))
    if busy(base):
        raise SystemExit("server busy: refusing to measure")
    report = {"workloads": a.workloads or "servebench default", "requests": [], "phases": {}}

    threads = [threading.Thread(target=stream_one, args=(url, a.model, body)) for _, body in warm]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for i, (key, body) in enumerate(isolated_items):
        wait_idle(base)
        before = metrics(base)
        r = stream_one(url, a.model, body)
        wait_idle(base, settle=0.2)
        r.update(phase="isolated", stream=0, index=i, key=key, metrics_before=before,
                 metrics_after=metrics(base))
        report["requests"].append(r)
    print("isolated", len(isolated_items), flush=True)

    for phase, phase_items, streams, seconds, cycle in phases:
        wait_idle(base)
        before = metrics(base)
        start = time.monotonic()
        results = closed_loop(url, a.model, phase_items, streams, seconds, cycle, phase)
        wait_idle(base, settle=0.2)
        report["phases"][phase] = {"start": start, "streams": streams, "seconds": seconds,
                                   "metrics_before": before, "metrics_after": metrics(base),
                                   "requests": len(results)}
        report["requests"].extend(results)
        print(phase, len(results), flush=True)

    json.dump(report, open(a.output, "w"))
    if a.analyze:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import analyze_trace

        out = analyze_trace.analyze(report)
        analyze_trace.render(report, out)
        print(json.dumps({"summary": analyze_trace.summary_line(out), "output": a.output}), flush=True)


if __name__ == "__main__":
    main()
