"""Summarize ffnbench runs: per (shape, mode, M), each arm's two boot medians and its
increment over the baseline arm, judged by the boots' interval.

usage: summarize.py OUTDIR SESSION BASE ARM [ARM...]   (reads OUTDIR/SESSION-ARM-bK.json)
gain   : both boots of ARM below both boots of BASE (max(ARM) < min(BASE))
slower : both boots of ARM above both boots of BASE (min(ARM) > max(BASE))
within : otherwise
Also checks every graph digest against BASE's (bit-exactness) and prints the schedule.
"""

import json
import sys

out, session, base, *arms = sys.argv[1:]


def load(arm, boot):
    with open(f"{out}/{session}-{arm}-b{boot}.json") as f:
        return json.load(f)


data = {a: [load(a, 1), load(a, 2)] for a in [base, *arms]}
summary = {"session": session, "base": base, "arms": arms, "cells": [], "bit_exact": {}, "schedule": {}}
shapes = list(data[base][0]["shapes"])
ms = data[base][0]["ms"]
for a in [base, *arms]:
    summary["schedule"][a] = data[a][0]["shapes"][shapes[0]]["schedule"]
    ok = True
    for boot in (0, 1):
        for sh in shapes:
            for m in ms:
                g0 = data[base][0]["shapes"][sh]["graphs"][str(m)]
                g = data[a][boot]["shapes"][sh]["graphs"][str(m)]
                ref = g0["eager_digest"]
                vals = [g["eager_digest"], *g["graph_digest"].values(), *g.get("graph_digest_after_timing", {}).values()]
                if any(v != ref for v in vals):
                    ok = False
                    print("DIGEST MISMATCH", a, boot + 1, sh, m, vals, ref)
    summary["bit_exact"][a] = ok

print(f"base = {base}; cells: median us per layer, boot1/boot2; delta = arm - base (mean of boots); verdict by boot interval")
for sh in shapes:
    for mode in ("chain", "cold"):
        print(f"\n{sh} {mode}")
        hdr = f"{'M':>3} {base + ' b1/b2':>20}"
        for a in arms:
            hdr += f" | {a + ' b1/b2':>20} {'delta':>7} {'':6}"
        print(hdr)
        for m in ms:
            key = f"{mode}/{m}"
            bv = [data[base][b]["shapes"][sh]["timing"][key]["median_us"] for b in (0, 1)]
            line = f"{m:>3} {bv[0]:9.2f}/{bv[1]:9.2f}"
            for a in arms:
                av = [data[a][b]["shapes"][sh]["timing"][key]["median_us"] for b in (0, 1)]
                d = sum(av) / 2 - sum(bv) / 2
                if max(av) < min(bv):
                    v = "gain"
                elif min(av) > max(bv):
                    v = "slower"
                else:
                    v = "within"
                line += f" | {av[0]:9.2f}/{av[1]:9.2f} {d:+7.2f} {v:6}"
                summary["cells"].append({
                    "shape": sh, "mode": mode, "M": m, "arm": a,
                    "base_us": [round(x, 3) for x in bv], "arm_us": [round(x, 3) for x in av],
                    "delta_us": round(d, 3), "verdict": v,
                    "base_aa_us": round(bv[1] - bv[0], 3), "arm_aa_us": round(av[1] - av[0], 3),
                })
            print(line)

print("\nbit-exact vs base (every eager and graph digest, both boots):", summary["bit_exact"])
for a, s in summary["schedule"].items():
    print("schedule", a, "M=1:", s["1"], "| M=48:", s["48"])
for a in arms:
    cells = [c for c in summary["cells"] if c["arm"] == a]
    n = {v: sum(c["verdict"] == v for c in cells) for v in ("gain", "within", "slower")}
    print(f"{a}: {n} of {len(cells)} cells")
with open(f"{out}/{session}-summary.json", "w") as f:
    json.dump(summary, f, indent=1)
