#!/usr/bin/env python3
"""Per-kernel GPU time of a profiled workload, kept small for cost attribution.

usage: summarize_kernels.py OUT.json TRACE.json.gz

Writes {"span_ms", "kernel_ms", "kernels": [[name, grid, block, launches, ms], ...]}
sorted by time: every GPU kernel in the profiler window (one rank), grouped by
name and launch geometry, so families (and target versus drafter launches,
told apart by grid) can be assigned afterwards without the raw trace.
"""
import gzip
import json
import sys
from collections import defaultdict

out, path = sys.argv[1], sys.argv[2]
with (gzip.open if path.endswith(".gz") else open)(path, "rt") as handle:
    events = json.load(handle)["traceEvents"]
kernels = [e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"]
groups = defaultdict(lambda: [0, 0.0])
for e in kernels:
    args = e.get("args", {})
    key = (e["name"], json.dumps(args.get("grid")), json.dumps(args.get("block")))
    groups[key][0] += 1
    groups[key][1] += e["dur"] / 1000
span = (max(e["ts"] + e["dur"] for e in kernels) - min(e["ts"] for e in kernels)) / 1000 if kernels else 0.0
rows = sorted(([n, json.loads(g), json.loads(b), c, round(ms, 4)] for (n, g, b), (c, ms) in groups.items()),
              key=lambda r: -r[4])
json.dump({"trace": path, "span_ms": round(span, 3), "kernel_ms": round(sum(r[4] for r in rows), 3),
           "kernels": rows}, open(out, "w"))
print(json.dumps({"trace": path.rsplit("/", 1)[-1], "span_ms": round(span, 1), "kernels": len(rows),
                  "kernel_ms": round(sum(r[4] for r in rows), 1)}))
