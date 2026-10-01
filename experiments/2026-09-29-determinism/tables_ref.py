#!/usr/bin/env python3
"""Summarize run50.sh's performance receipts: r5o against the reference configuration.

usage: tables_ref.py   (from the deployment checkout)

Decode (prose and JSON answers, one and eight identical streams, three samples:
tokens per second with its 95% interval, GPU step time, verified and accepted
drafts per draft event), eight distinct concurrent prompts (c8_distinct.py,
256 tokens each), cold prefill of real text (tokens per second and time to
first token), and short-prompt request time (prefill plus one step, median).
"""
import json
import os

ARMS = (("r5o", "r5o"), ("ref", "reference configuration"))
B = "results/private/bench/ref-{}/bench.json"
D = "results/private/determinism/ref/{}-{}.jsonl"
suites = {k: json.load(open(B.format(k)))["suites"] for k, _ in ARMS if os.path.exists(B.format(k))}
for point in ("prose-c1", "json-nothink-c1", "prose-c8", "json-nothink-c8"):
    print(point)
    base = None
    for key, label in ARMS:
        if key not in suites:
            continue
        p = suites[key]["decode"]["points"][point]
        tps = p["tps"]["mean"]
        step = p["step_ms"]["mean"] if p["step_ms"].get("n") else float("nan")
        change = "" if base is None else f"  {100 * (tps - base) / base:+5.1f}% vs r5o"
        base = tps if key == "r5o" else base
        print(f"  {label:26s} tps {tps:7.2f} ±{p['tps']['ci95_pct']:4.1f}%  step {step:6.2f} ms  "
              f"verified/draft {p['verified_per_draft']['mean']:.3f}  accepted/draft "
              f"{p['accepted_per_draft']['mean']:.3f}  distinct outputs {p.get('distinct_outputs')}{change}")
print("eight distinct concurrent prompts (tokens/s over the window, ±95%)")
base = None
for key, label in ARMS:
    path = D.format("c8-distinct", key)
    if not os.path.exists(path):
        continue
    rows = [json.loads(line) for line in open(path)]
    summary = next(r for r in rows if "summary" in r)
    per_stream = [r["per_stream_tps"] for r in rows if "per_stream_tps" in r]
    change = "" if base is None else f"  {100 * (summary['tps_mean'] - base) / base:+5.1f}% vs r5o"
    base = summary["tps_mean"] if key == "r5o" else base
    print(f"  {label:26s} tps {summary['tps_mean']:7.2f} ±{summary['ci95_pct']:4.1f}%  per stream "
          f"{sum(per_stream) / len(per_stream):6.2f}{change}")
print("cold prefill, real text: prefill tokens/s ±95% (TTFT s)")
sizes = sorted({int(s) for v in suites.values() for s in v.get("prefill", {}).get("points", {})})
base = {}
for key, label in ARMS:
    if key not in suites:
        continue
    cells = []
    for s in sizes:
        p = suites[key]["prefill"]["points"][str(s)]
        tps, ttft = p["prefill_tps"]["mean"], p["ttft_s"]["mean"]
        change = "" if key == "r5o" else f" {100 * (tps - base[s]) / base[s]:+.1f}%"
        base.setdefault(s, tps)
        cells.append(f"{s}: {tps:.0f} ±{p['prefill_tps']['ci95_pct']:.1f}% ({ttft:.2f}){change}")
    print(f"  {label:26s} " + "; ".join(cells))
print("short prompts: request time, prefill plus one step (median ms)")
rows = {}
for key, _ in ARMS:
    path = D.format("ttft", key)
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            rows.setdefault(r["prompt_tokens"], {})[key] = r["median_ms"]
for tokens in sorted(rows):
    print(f"  {tokens:5d} tokens: " + "  ".join(f"{k} {v:.1f}" for k, v in rows[tokens].items()))
