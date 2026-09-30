#!/usr/bin/env python3
"""Summarize run42.sh's performance receipts.  usage: tables_rs.py   (from the deployment checkout)

Per arm (one boot each, one pinned cost table): decode throughput with its 95%
interval, GPU step time, verified and accepted drafts per draft event (prose
and JSON answers, one and eight streams, three samples); cold prefill of real
text (Python standard-library source) at 1024-65536 tokens: prefill tokens per
second and time to first token (three repeats).
"""
import json
import os

ARMS = (("r5o", "r5o"), ("detm-bi", "det. + all fixes + BI mode"),
        ("detm-bi-rs", "det. + all + BI + 0031"))
B = "results/private/bench/rs-{}/bench.json"
suites = {key: json.load(open(B.format(key)))["suites"] for key, _ in ARMS if os.path.exists(B.format(key))}
for point in ("prose-c1", "json-nothink-c1", "prose-c8", "json-nothink-c8"):
    print(point)
    base = None
    for key, label in ARMS:
        if key not in suites:
            print(f"  {label:30s} missing")
            continue
        p = suites[key]["decode"]["points"][point]
        tps = p["tps"]["mean"]
        step = p["step_ms"]["mean"] if p["step_ms"].get("n") else float("nan")
        change = "" if base is None else f"  {100 * (tps - base) / base:+5.1f}% vs r5o"
        base = tps if key == "r5o" else base
        print(f"  {label:30s} tps {tps:7.2f} ±{p['tps']['ci95_pct']:4.1f}%  step {step:6.2f} ms  "
              f"verified/draft {p['verified_per_draft']['mean']:.3f}  "
              f"accepted/draft {p['accepted_per_draft']['mean']:.3f}{change}")
print("cold prefill, real text: prefill tokens/s ±95% (TTFT s)")
sizes = sorted({int(s) for v in suites.values() for s in v.get("prefill", {}).get("points", {})})
print("  " + " " * 30 + "".join(f"{s:>24d}" for s in sizes))
base = {}
for key, label in ARMS:
    if key not in suites or "prefill" not in suites[key]:
        continue
    cells = []
    for s in sizes:
        p = suites[key]["prefill"]["points"].get(str(s))
        if p is None:
            cells.append(f"{'-':>24s}")
            continue
        tps, ttft = p["prefill_tps"]["mean"], p["ttft_s"]["mean"]
        change = "" if key == "r5o" else f" {100 * (tps - base[s]) / base[s]:+.1f}%"
        base.setdefault(s, tps)
        cells.append(f"{tps:8.0f} ±{p['prefill_tps']['ci95_pct']:3.1f}% ({ttft:.2f}){change}".rjust(24))
    print(f"  {label:30s}" + "".join(cells))
