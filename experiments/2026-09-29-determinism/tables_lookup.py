#!/usr/bin/env python3
"""Summarize run30.sh's performance receipts.  usage: tables_lookup.py   (from the deployment checkout)

Per arm (one boot each, one pinned cost table): decode throughput with its 95%
interval, GPU step time, verified and accepted drafts per draft event for
prose and JSON answers at one and eight streams; then the short-prompt
request time (prefill plus one step) per prompt length, median of seven.
"""
import json
import os

ARMS = (("r5o", "r5o"), ("lookup", "r5o + GEMV lookup"), ("lookup-variant", "r5o + lookup + MoE variant"),
        ("detm", "deterministic MoE"), ("detm-lookup-variant", "deterministic + lookup + variant"))
B = "results/private/bench/lookup-{}/bench.json"
T = "results/private/determinism/lookup/ttft-{}.jsonl"
for point in ("prose-c1", "json-nothink-c1", "prose-c8", "json-nothink-c8"):
    print(point)
    base = None
    for key, label in ARMS:
        if not os.path.exists(B.format(key)):
            print(f"  {label:34s} missing")
            continue
        p = json.load(open(B.format(key)))["suites"]["decode"]["points"][point]
        tps = p["tps"]["mean"]
        step = p["step_ms"]["mean"] if p["step_ms"].get("n") else float("nan")
        change = "" if base is None else f"  {100 * (tps - base) / base:+5.1f}% vs r5o"
        base = tps if key == "r5o" else base
        print(f"  {label:34s} tps {tps:7.2f} ±{p['tps']['ci95_pct']:4.1f}%  step {step:6.2f} ms  "
              f"verified/draft {p['verified_per_draft']['mean']:.3f}  "
              f"accepted/draft {p['accepted_per_draft']['mean']:.3f}{change}")
print("short prompts: request time, prefill plus one step (median ms; prompt tokens)")
rows = {}
for key, label in ARMS:
    if not os.path.exists(T.format(key)):
        continue
    for line in open(T.format(key)):
        r = json.loads(line)
        rows.setdefault(r["prompt_tokens"], {})[key] = r["median_ms"]
header = "  tokens " + " ".join(f"{k:>20s}" for k, _ in ARMS)
print(header)
for tokens in sorted(rows):
    print(f"  {tokens:6d} " + " ".join(f"{rows[tokens].get(k, float('nan')):20.1f}" for k, _ in ARMS))
