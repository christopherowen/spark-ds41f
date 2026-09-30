#!/usr/bin/env python3
"""Tabulate run21's pinned decode benches.  usage: pin_table.py (from the deployment checkout)"""
import json

R = "results/private/bench/r5n-pin-{}/bench.json"
for point in ("prose-c1", "json-nothink-c1", "prose-c8", "json-nothink-c8"):
    print(point)
    for a in ("r5n-1", "detm-1", "r5n-2", "detm-2"):
        p = json.load(open(R.format(a)))["suites"]["decode"]["points"][point]
        step = p["step_ms"]["mean"] if p["step_ms"].get("n") else float("nan")
        print(f"  {a:7s} tps {p['tps']['mean']:7.2f} ±{p['tps']['ci95_pct']:4.1f}%  step {step:6.2f}  "
              f"verified/draft {p['verified_per_draft']['mean']:.3f}  accepted/draft "
              f"{p['accepted_per_draft']['mean']:.3f}  distinct {p['distinct_outputs']}")
