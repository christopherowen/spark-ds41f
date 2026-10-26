#!/usr/bin/env python3
"""Where a window's first tokens went, from two /metrics scrapes (before and after it).

usage: ttft_split.py BEFORE.txt AFTER.txt [--client-ttft-ms MS] [--step-ms MS] [--json OUT]

vLLM already keeps these histograms, so reading them costs serving nothing (the same
endpoint servebench scrapes at every window boundary). Over the requests that finished
between the scrapes it reports each histogram's mean and bucket quantiles, and splits the
mean time to first token into:

- queue: arrival in the engine core to first scheduled (request_queue_time_seconds);
- engine: first scheduled to first token (request_prefill_time_seconds), which holds the
  rest of the step running when the request was scheduled, the step decided ahead under
  async scheduling, the prompt's own chunks, Engram's first-chunk wait and the first token's
  held step (sink rejection emits it after the next step verifies it);
- frontend: the rest of the server's TTFT (time_to_first_token_seconds less the two),
  tokenization, the API server's queues and output processing.

With --client-ttft-ms (servebench's mean TTFT for the same window) the remainder is
delivery: HTTP, SSE and the client. With --step-ms (the window's mean step) the engine part
is also given in steps. The engine part cannot be split further from counters; the lab's
traced replay (experiments/2026-10-20-ttft-gap/ttft_trace.py) does that.
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

HISTOGRAMS = {
    "ttft": "vllm:time_to_first_token_seconds",
    "queue": "vllm:request_queue_time_seconds",
    "prefill": "vllm:request_prefill_time_seconds",
    "decode": "vllm:request_decode_time_seconds",
    "inference": "vllm:request_inference_time_seconds",
    "e2e": "vllm:e2e_request_latency_seconds",
    "itl": "vllm:inter_token_latency_seconds",
}
LINE = re.compile(r'^(?P<name>[^{\s]+)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>\S+)')
LE = re.compile(r'(?:^|,)\s*le="([^"]*)"')


def histograms(text: str) -> dict[str, dict]:
    """Every histogram's buckets (summed over label sets), sum and count."""
    out: dict[str, dict] = {}
    for line in text.splitlines():
        match = LINE.match(line)
        if not match or line.startswith("#"):
            continue
        name, labels, value = match["name"], match["labels"] or "", match["value"]
        try:
            number = float(value)
        except ValueError:
            continue
        if math.isnan(number):
            continue
        for suffix in ("_bucket", "_sum", "_count"):
            if name.endswith(suffix):
                base = name[: -len(suffix)]
                entry = out.setdefault(base, {"buckets": {}, "sum": 0.0, "count": 0.0})
                if suffix == "_bucket":
                    le = LE.search(labels)
                    if le:
                        bound = math.inf if le.group(1) == "+Inf" else float(le.group(1))
                        entry["buckets"][bound] = entry["buckets"].get(bound, 0.0) + number
                else:
                    entry[suffix[1:]] += number
    return out


def delta(before: dict, after: dict) -> dict:
    out = {}
    for name, a in after.items():
        b = before.get(name, {"buckets": {}, "sum": 0.0, "count": 0.0})
        out[name] = {"buckets": {k: v - b["buckets"].get(k, 0.0) for k, v in a["buckets"].items()},
                     "sum": a["sum"] - b["sum"], "count": a["count"] - b["count"]}
    return out


def quantile(hist: dict, q: float) -> tuple[float, float] | None:
    """The bucket holding the q-th observation, as (lower, upper] bounds. vLLM's buckets are
    too wide for interpolation inside one to mean anything at these latencies."""
    bounds = sorted(hist["buckets"])
    if not bounds or not hist["count"]:
        return None
    rank = q * hist["count"]
    previous = 0.0
    for bound in bounds:
        if hist["buckets"][bound] >= rank:
            return previous, bound
        previous = bound
    return previous, math.inf


def summary(window: dict) -> dict:
    out = {}
    for key, name in HISTOGRAMS.items():
        hist = window.get(name)
        if not hist or not hist["count"]:
            continue
        out[key] = {"n": hist["count"], "mean_ms": 1000 * hist["sum"] / hist["count"],
                    **{f"p{int(100 * q)}_bucket_ms": ([1000 * b for b in v] if (v := quantile(hist, q)) else None)
                       for q in (0.5, 0.9, 0.99)}}
    if {"ttft", "queue", "prefill"} <= set(out):
        ttft, queue, engine = out["ttft"]["mean_ms"], out["queue"]["mean_ms"], out["prefill"]["mean_ms"]
        out["split_ms"] = {"queue": queue, "engine": engine, "frontend": ttft - queue - engine, "server_ttft": ttft}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--client-ttft-ms", type=float)
    parser.add_argument("--step-ms", type=float)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    window = delta(histograms(args.before.read_text()), histograms(args.after.read_text()))
    out = summary(window)
    print(f"{'histogram':10s} {'n':>7} {'mean ms':>9}  {'p50 in (ms]':>14} {'p90 in (ms]':>14} {'p99 in (ms]':>14}")
    for key in HISTOGRAMS:
        if key in out:
            r = out[key]
            cells = " ".join(f"{f'{v[0]:.0f}-{v[1]:.0f}':>14}" if (v := r[k]) else f"{'-':>14}"
                             for k in ("p50_bucket_ms", "p90_bucket_ms", "p99_bucket_ms"))
            print(f"{key:10s} {r['n']:7.0f} {r['mean_ms']:9.1f}  {cells}")
    split = out.get("split_ms")
    if split:
        if args.client_ttft_ms is not None:
            split["delivery"] = args.client_ttft_ms - split["server_ttft"]
        print("\nmean time to first token:")
        for key in ("frontend", "queue", "engine", "delivery"):
            if key in split:
                share = 100 * split[key] / (args.client_ttft_ms or split["server_ttft"])
                steps = f", {split[key] / args.step_ms:.2f} steps" if args.step_ms and key == "engine" else ""
                print(f"  {key:9s} {split[key]:8.1f} ms  {share:5.1f}%{steps}")
    if args.json:
        args.json.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    sys.exit(main())
