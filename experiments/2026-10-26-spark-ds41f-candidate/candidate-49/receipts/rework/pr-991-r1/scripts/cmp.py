#!/usr/bin/env python3
"""Cell table for the PR #991 rework.

base = the 1107 arm (bench ``sw``) of the candidate-49 own-mark receipts, two boots;
cand = the ``hw`` arm of two new boots. A cell is worse beyond the interval when the
faster candidate boot is slower than the slower base boot, a gain when the slower
candidate boot beats the faster base boot. ``paired`` is each new boot's own
in-run hw-vs-sw delta (the bench's ``delta_pct``).

usage: cmp.py <receipt boot1> <receipt boot2> <new boot1> <new boot2> [--graph g1 g2] [--json out]
"""

import json
import sys

args = sys.argv[1:]
out_json = None
if "--json" in args:
    i = args.index("--json")
    out_json = args[i + 1]
    del args[i : i + 2]
graph = None
if "--graph" in args:
    i = args.index("--graph")
    graph = [json.load(open(p)) for p in args[i + 1 : i + 3]]
    del args[i : i + 3]
r1, r2, n1, n2 = (json.load(open(p)) for p in args[:4])
rows = []
tally = {"cells": 0, "gain_beyond_interval": 0, "worse_beyond_interval": 0, "within_interval": 0}
bit = True
for bench, recs in r1["benchmarks"].items():
    for i, rec in enumerate(recs):
        base = [r["benchmarks"][bench][i]["sw"]["median_us"] for r in (r1, r2)]
        prev = [r["benchmarks"][bench][i]["hw"]["median_us"] for r in (r1, r2)]
        cand = [n["benchmarks"][bench][i]["hw"]["median_us"] for n in (n1, n2)]
        nsw = [n["benchmarks"][bench][i]["sw"]["median_us"] for n in (n1, n2)]
        assert all(n["benchmarks"][bench][i]["M"] == rec["M"] for n in (n1, n2))
        ex = all(n["benchmarks"][bench][i]["bit_exact"] for n in (n1, n2))
        bit = bit and ex
        worse = min(cand) > max(base)
        gain = max(cand) < min(base)
        tally["cells"] += 1
        tally["worse_beyond_interval" if worse else "gain_beyond_interval" if gain else "within_interval"] += 1
        row = {
            "bench": bench,
            "M": rec["M"],
            "base_1107_receipt_boots_us": base,
            "pr991_receipt_boots_us": prev,
            "cand_boots_us": cand,
            "speedup_of_means_vs_1107": round(sum(base) / sum(cand), 3),
            "verdict": "worse" if worse else "gain" if gain else "within",
            "new_boots_sw_arm_us": nsw,
            "new_boots_paired_delta_pct": [n["benchmarks"][bench][i]["delta_pct"] for n in (n1, n2)],
            "bit_exact": ex,
        }
        if graph:
            gs = [g["benchmarks"][bench][i]["graph"] for g in graph]
            row["graph_us_per_launch"] = {
                "sw": [g["sw"]["median_us"] for g in gs],
                "hw": [g["hw"]["median_us"] for g in gs],
            }
        rows.append(row)

print(f"{'bench':40s} {'M':>5s} {'1107 rcpt':>15s} {'#991 rcpt':>15s} {'cand':>15s} {'x':>6s} {'v':6s} {'paired %':>14s}"
      + ("  graph sw->hw" if graph else ""))
for r in rows:
    line = (f"{r['bench']:40s} {r['M']:5d} {'/'.join('%.3f' % x for x in r['base_1107_receipt_boots_us']):>15s} "
            f"{'/'.join('%.3f' % x for x in r['pr991_receipt_boots_us']):>15s} "
            f"{'/'.join('%.3f' % x for x in r['cand_boots_us']):>15s} {r['speedup_of_means_vs_1107']:6.3f} "
            f"{r['verdict']:6s} {'/'.join('%+.2f' % x for x in r['new_boots_paired_delta_pct']):>14s}")
    if graph:
        g = r["graph_us_per_launch"]
        line += "  " + "/".join("%.3f" % x for x in g["sw"]) + " -> " + "/".join("%.3f" % x for x in g["hw"])
    print(line)
print(json.dumps(tally), "bit_exact_all:", bit)
if out_json:
    json.dump({"tally": tally, "bit_exact_all": bit, "cells": rows}, open(out_json, "w"), indent=1)
