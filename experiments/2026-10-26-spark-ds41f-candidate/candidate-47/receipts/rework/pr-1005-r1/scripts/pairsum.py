"""Summarize pairbench boots: per (shape, mode, M, variant) the paired window deltas
(variant - base, same window), their median and a 95% bootstrap interval of the
median, per boot. gain: both boots' intervals below 0; slower: both above 0.

usage: pairsum.py OUT LABEL_B1 LABEL_B2
"""

import json
import random
import statistics
import sys

out, l1, l2 = sys.argv[1:]
runs = [json.load(open(f"{out}/{l}.json")) for l in (l1, l2)]
variants = runs[0]["variants"]
ms = runs[0]["ms"]
rng = random.Random(7)


def ci(d):
    meds = []
    for _ in range(2000):
        s = [d[rng.randrange(len(d))] for _ in d]
        meds.append(statistics.median(s))
    meds.sort()
    return meds[50], meds[1949]


summary = {"labels": [l1, l2], "cells": [], "bit_exact": {}}
for v in variants:
    ok = True
    for r in runs:
        for sh, s in r["shapes"].items():
            b = s["digests"]["base"]
            for k, x in s["digests"][v].items():
                if x != b[k.replace("/after", "")] and x != b.get(k):
                    ok = False
    summary["bit_exact"][v] = ok

for sh in runs[0]["shapes"]:
    for mode in ("chain", "cold"):
        print(f"\n{sh} {mode}: base median us (b1/b2); per variant: median paired delta b1/b2 [95% CI b1] [CI b2] verdict")
        for M in ms:
            bt = [r["shapes"][sh]["timing"][f"base/{mode}/{M}"]["median_us"] for r in runs]
            line = f"M={M:2d} {bt[0]:8.2f}/{bt[1]:8.2f}"
            for v in variants[1:]:
                meds, cis = [], []
                for r in runs:
                    t = r["shapes"][sh]["timing"]
                    d = [x - y for x, y in zip(t[f"{v}/{mode}/{M}"]["windows_us"], t[f"base/{mode}/{M}"]["windows_us"])]
                    meds.append(statistics.median(d))
                    cis.append(ci(d))
                if all(c[1] < 0 for c in cis):
                    verdict = "gain"
                elif all(c[0] > 0 for c in cis):
                    verdict = "slower"
                else:
                    verdict = "-"
                line += f" | {v} {meds[0]:+5.2f}/{meds[1]:+5.2f} {verdict:6s}"
                summary["cells"].append({
                    "shape": sh, "mode": mode, "M": M, "variant": v, "base_median_us": [round(x, 3) for x in bt],
                    "paired_delta_median_us": [round(x, 3) for x in meds],
                    "paired_delta_ci95_us": [[round(c[0], 3), round(c[1], 3)] for c in cis], "verdict": verdict,
                })
            print(line)
print("\nbit-exact vs base:", summary["bit_exact"])
for v in variants[1:]:
    cells = [c for c in summary["cells"] if c["variant"] == v]
    print(v, {k: sum(c["verdict"] == k for c in cells) for k in ("gain", "-", "slower")}, "of", len(cells))
json.dump(summary, open(f"{out}/s2-pair-summary.json", "w"), indent=1)
