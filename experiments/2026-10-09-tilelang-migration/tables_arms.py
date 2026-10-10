#!/usr/bin/env python3
"""One run's performance receipts for several arms, each against the first (r5o).

usage: tables_arms.py RUN ARM [ARM...]   (from the deployment checkout; RUN names
       results/private/bench/RUN-<arm>/ and results/private/determinism/RUN/)

One-stream decode (prose, JSON: tokens/s with its 95% interval, GPU step time, accepted drafts
per draft event, distinct outputs; and twelve distinct prompts one at a time), cold prefill (tokens/s and TTFT), eight distinct concurrent
prompts, short-prompt request time (median of the per-length medians) and mixed-traffic latency
(short and long TTFT, decoding streams' chunk gaps). Changes are against the first arm; "over"
marks a regression beyond the 3% budget that its interval does not cover.
"""
import json
import os
import statistics
import sys

RUN, ARMS = sys.argv[1], sys.argv[2:]
B = f"results/private/bench/{RUN}-{{}}/bench.json"
D = f"results/private/determinism/{RUN}/{{}}-{{}}.jsonl"
BUDGET = 3.0


def lines(kind, arm):
    path = D.format(kind, arm)
    return [json.loads(x) for x in open(path) if x.strip().startswith("{")] if os.path.exists(path) else []


def change(value, base, higher_better=True, ci=0.0):
    if base is None or value is None or base == 0:
        return ""
    pct = 100 * (value - base) / base
    worse = -pct if higher_better else pct
    flag = "  over" if worse > BUDGET + ci else ""
    return f" {pct:+5.1f}%{flag}"


suites = {a: json.load(open(B.format(a)))["suites"] for a in ARMS if os.path.exists(B.format(a))}
for point in ("prose-c1", "json-nothink-c1"):
    print(point)
    base_tps = base_step = None
    for a in ARMS:
        p = suites.get(a, {}).get("decode", {}).get("points", {}).get(point)
        if not p:
            continue
        tps, ci = p["tps"]["mean"], p["tps"]["ci95_pct"]
        step = p["step_ms"]["mean"] if p["step_ms"].get("n") else None
        print(f"  {a:10s} tps {tps:7.2f} ±{ci:4.1f}%{change(tps, base_tps, True, ci)}   step "
              f"{step:6.2f} ms{change(step, base_step, False)}   accepted/draft {p['accepted_per_draft']['mean']:.3f}"
              f"   distinct outputs {p.get('distinct_outputs', '?')}")
        if base_tps is None:
            base_tps, base_step = tps, step
print("single stream, distinct prompts one at a time (tok/s ±95%)")
b = None
for a in ARMS:
    s1 = [x for x in lines("c1-distinct", a) if "summary" in x]
    if s1:
        print(f"  {a:10s} {s1[0]['tps_mean']:7.2f} ±{s1[0]['ci95_pct']:.1f}%{change(s1[0]['tps_mean'], b, True, s1[0]['ci95_pct'])}")
        b = b or s1[0]["tps_mean"]
print("cold prefill: tokens/s (TTFT s)")
base = {}
for a in ARMS:
    pts = suites.get(a, {}).get("prefill", {}).get("points", {})
    cells = []
    for size in sorted(pts, key=lambda s: int(s)):
        p = pts[size]
        tps = p["prefill_tps"]["mean"]
        cells.append(f"{size}: {tps:.0f} ({p['ttft_s']['mean']:.2f}){change(tps, base.get(size), True, p['prefill_tps']['ci95_pct'])}")
        base.setdefault(size, tps)
    print(f"  {a:10s} " + "; ".join(cells))
for streams, words in ((8, "eight"), (16, "sixteen")):
    print(f"{words} distinct concurrent prompts (tok/s ±95%)")
    b = None
    for a in ARMS:
        s = [x for x in lines(f"c{streams}-distinct", a) if "summary" in x]
        if s:
            print(f"  {a:10s} {s[0]['tps_mean']:7.2f} ±{s[0]['ci95_pct']:.1f}%"
                  f"{change(s[0]['tps_mean'], b, True, s[0]['ci95_pct'])}")
            b = b or s[0]["tps_mean"]
print("short prompts: request time, median over lengths (ms)")
b = None
for a in ARMS:
    rows = lines("ttft", a)
    vals = [x.get("median_ms") for x in rows if x.get("median_ms") is not None]
    if vals:
        m = statistics.median(vals)
        print(f"  {a:10s} {m:7.1f}{change(m, b, False)}")
        b = b or m
print("mixed traffic (means over rounds)")
b = {}
for a in ARMS:
    s = [x for x in lines("mixed", a) if "summary" in x]
    if s:
        s = s[0]
        keys = ("short_ttft_ms_median", "long_ttft_ms_mean", "stream_gap_ms_p50", "stream_gap_ms_p95",
                "stream_gap_ms_p99", "stream_chunks_per_s")
        cells = [f"{k.replace('_ms', '').replace('stream_', '')} {s[k]}{change(s[k], b.get(k), k == 'stream_chunks_per_s')}"
                 for k in keys]
        print(f"  {a:10s} " + "; ".join(cells))
        for k in keys:
            b.setdefault(k, s[k])
print("temperature 0: requests differing from each prompt alone (tokens / logprobs only)")
for a in ARMS:
    s = [x for x in lines("determinism", a) if "summary" in x]
    if s:
        s = s[0]
        cells = [f"{name} {r['token_diffs']}/{r['logprob_diffs']}" for name, r in s["scenarios"].items()]
        verdict = "identical" if s["identical"] else "DIFFERS"
        print(f"  {a:10s} {verdict}{'' if s['logprobs'] else ' (no logprobs)'}: " + "; ".join(cells))
