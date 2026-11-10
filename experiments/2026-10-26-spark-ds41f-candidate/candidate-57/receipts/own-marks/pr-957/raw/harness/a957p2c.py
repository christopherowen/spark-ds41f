"""#957 part 2 confirmation: 4+4 boots, ABBA ABBA (base 1, cand 1, cand 2, base 2, base 3,
cand 3, cand 4, base 4).

usage: python3 a957p2c.py RAWDIR > OUT.json

Per cell, by CUDA-graph replay (the median per call of each boot): the interval rule over all
4 + 4 boots (faster when the slowest candidate boot beats the fastest base boot, slower when the
fastest candidate boot is slower than the slowest base boot, else same), the mean difference,
and the four adjacent pairs' differences in run order: (b1, c1), (c2, b2), (b3, c3), (c4, b4).
"""
import json
import statistics
import sys
from pathlib import Path

raw = Path(sys.argv[1])
boot = lambda arm, k: json.loads((raw / f"mlac-{arm}-boot{k}.json").read_text())  # noqa: E731
base = [boot("base", k) for k in (1, 2, 3, 4)]
cand = [boot("c57p2", k) for k in (1, 2, 3, 4)]
pairs = [(0, 0), (1, 1), (2, 2), (3, 3)]  # base k with cand k: (b1,c1), (b2,c2), (b3,c3), (b4,c4)


def verdict(b, c):
    if max(c) < min(b):
        return "faster"
    if min(c) > max(b):
        return "slower"
    return "same"


out = {"order": "base1, c57p2 1, c57p2 2, base2, base3, c57p2 3, c57p2 4, base4", "cells": {}}
for key in base[0]["cells"]:
    for metric, get in (("graph_hot_us", lambda d: d["graph_hot"]["median_us"]),
                        ("kernel_us", lambda d: d["kernel"]["per_call_us"])):
        b = [get(x["cells"][key]) for x in base]
        c = [get(x["cells"][key]) for x in cand]
        cell = out["cells"].setdefault(key, {"route": base[0]["cells"][key]["route"]})
        cell[metric] = {
            "base": b, "cand": c,
            "mean_delta": round(statistics.mean(c) - statistics.mean(b), 3),
            "mean_delta_pct": round(100 * (statistics.mean(c) - statistics.mean(b)) / statistics.mean(b), 2),
            "pair_deltas": [round(c[j] - b[i], 3) for i, j in pairs],
            "pairs_lower": sum(c[j] < b[i] for i, j in pairs),
            "verdict": verdict(b, c),
        }
    cell["outputs_equal_all_boots"] = len({x["cells"][key]["out_hash"] for x in base + cand}) == 1
    cell["route_equal"] = all(x["cells"][key]["route"] == cell["route"] for x in base + cand)
print(json.dumps(out, indent=1))
