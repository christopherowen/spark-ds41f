#!/usr/bin/env python3
"""Summarize screen.sh's receipts.  usage: tables.py   (from the deployment checkout)

Prefill: per depth, each arm's rounds, mean, round-to-round spread and the
overlay's change. Decode: per point and boot, throughput, step time, verified
and accepted drafts per draft event.
"""
import json

out = "results/private/r5o"
rows = {}
for arm in ("r5n", "overlay"):
    for line in open(f"{out}/depth-{arm}.jsonl"):
        line = line.strip()
        if line.startswith("{") and '"depth"' in line:
            r = json.loads(line)
            rows.setdefault(r["depth"], {}).setdefault(arm, []).append(r["prefill_ms"])
print("prefill, 4,096-token chunk (ms)")
for depth in sorted(rows):
    o, b = rows[depth]["overlay"], rows[depth]["r5n"]
    mo, mb = sum(o) / len(o), sum(b) / len(b)
    spread = lambda v: 100 * (max(v) - min(v)) / min(v)
    print(f"  {depth:>7}: r5n {mb:7.1f} (spread {spread(b):.2f}%)  overlay {mo:7.1f} (spread {spread(o):.2f}%)  "
          f"change {100 * (mo - mb) / mb:+.2f}%")
R = "results/private/bench/r5o-{}/bench.json"
for point in ("prose-c1", "json-nothink-c1", "prose-c8", "json-nothink-c8"):
    print(point)
    for a in ("r5n-1", "overlay-1", "r5n-2", "overlay-2"):
        p = json.load(open(R.format(a)))["suites"]["decode"]["points"][point]
        step = p["step_ms"]["mean"] if p["step_ms"].get("n") else float("nan")
        print(f"  {a:9s} tps {p['tps']['mean']:7.2f} ±{p['tps']['ci95_pct']:4.1f}%  step {step:6.2f}  "
              f"verified/draft {p['verified_per_draft']['mean']:.3f}  accepted/draft {p['accepted_per_draft']['mean']:.3f}")
