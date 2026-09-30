#!/usr/bin/env python3
"""Routed-MoE and top-k sum launches by grid in a trace.  usage: moe_grids.py TRACE.json.gz"""
import gzip
import json
import statistics
import sys
from collections import defaultdict

with gzip.open(sys.argv[1], "rt") as handle:
    events = json.load(handle)["traceEvents"]
groups = defaultdict(list)
for e in events:
    if e.get("cat") != "kernel" or e.get("ph") != "X":
        continue
    name = e["name"]
    if "MoEDynamicKernel" in name or "TopKSum" in name or "W4A8" in name or "Phase1" in name:
        kind = ("topk_sum" if "TopKSum" in name else "moe_dynamic" if "MoEDynamic" in name else name[:60])
        groups[(kind, tuple(e.get("args", {}).get("grid", [])))].append(e["dur"])
for (kind, grid), durs in sorted(groups.items(), key=lambda kv: -sum(kv[1])):
    print(f"{kind:14s} grid {str(grid):16s} n {len(durs):6d} mean {statistics.mean(durs):8.1f} us "
          f"total {sum(durs) / 1000:9.1f} ms")
