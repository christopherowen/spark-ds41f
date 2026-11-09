#!/usr/bin/env python3
"""Summaries for result.json: graph-mode and CUPTI two-boot tables, and the eager-cell view.

usage: summarize.py <json dir> <cmp.json> <out.json>
"""

import json
import sys
from pathlib import Path

d, cmpf, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
g = [json.loads((d / f"graph-r1-boot{b}.json").read_text()) for b in (1, 2)]
c = [json.loads((d / f"cupti-r1-boot{b}.json").read_text()) for b in (1, 2)]
cm = json.load(open(cmpf))

graph = []
for bench, rows in g[0]["benchmarks"].items():
    for i, r in enumerate(rows):
        sw = [x["benchmarks"][bench][i]["graph"]["sw"]["median_us"] for x in g]
        hw = [x["benchmarks"][bench][i]["graph"]["hw"]["median_us"] for x in g]
        eager_sw = [x["benchmarks"][bench][i]["sw"]["median_us"] for x in g]
        eager_hw = [x["benchmarks"][bench][i]["hw"]["median_us"] for x in g]
        graph.append({
            "bench": bench, "M": r["M"],
            "graph_us_per_launch_1107_boots": sw, "graph_us_per_launch_r1_boots": hw,
            "graph_speedup_of_means": round(sum(sw) / sum(hw), 3),
            "graph_r1_slower_beyond_interval": min(hw) > max(sw),
            "graph_r1_faster_beyond_interval": max(hw) < min(sw),
            "companion_eager_us_1107_boots": eager_sw, "companion_eager_us_r1_boots": eager_hw,
        })

cupti = {"gemm": [], "paged": []}
for kind in cupti:
    for i, r in enumerate(c[0][kind]):
        row = {"M": r["M"]}
        for arm in [k for k in r if k not in ("M", "block_M")]:
            row[arm] = [x[kind][i][arm]["median_us"] for x in c]
        cupti[kind].append(row)

res = {
    "graph_tally": {
        "cells": len(graph),
        "r1_faster_beyond_interval": sum(x["graph_r1_faster_beyond_interval"] for x in graph),
        "r1_slower_beyond_interval": sum(x["graph_r1_slower_beyond_interval"] for x in graph),
    },
    "graph_cells": graph,
    "cupti_kernel_median_us_boots": cupti,
}
Path(out).write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps(res["graph_tally"]))
for x in graph:
    print(f"{x['bench']:40s} {x['M']:5d} graph {'/'.join('%.3f' % v for v in x['graph_us_per_launch_1107_boots'])} -> "
          f"{'/'.join('%.3f' % v for v in x['graph_us_per_launch_r1_boots'])} x{x['graph_speedup_of_means']}"
          f"{' SLOWER' if x['graph_r1_slower_beyond_interval'] else ''}")
for kind, rows in cupti.items():
    for r in rows:
        print(kind, r)
