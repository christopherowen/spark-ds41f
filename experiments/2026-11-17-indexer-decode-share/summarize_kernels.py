#!/usr/bin/env python3
"""Per-kernel GPU time of every profiler window of one rank.

usage: summarize_kernels.py OUT.json TRACE.json.gz

The lab passes one trace of the rank's directory; every trace beside it is summarized,
in the order the windows were written (the trace's first kernel timestamp). Writes
{"windows": [{"trace", "first_kernel_ts", "span_ms", "kernel_ms", "launches",
"kernels": [[name, launches, ms], ...]}, ...]}, kernels sorted by time and grouped by
name, so families can be assigned afterwards without the raw traces.
"""
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

out, given = sys.argv[1], Path(sys.argv[2])
windows = []
for path in sorted(given.parent.glob("*.json*")):
    with (gzip.open if path.name.endswith(".gz") else open)(path, "rt") as handle:
        events = json.load(handle)["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"]
    del events
    groups = defaultdict(lambda: [0, 0.0])
    for e in kernels:
        groups[e["name"]][0] += 1
        groups[e["name"]][1] += e["dur"] / 1000
    first = min((e["ts"] for e in kernels), default=0)
    last = max((e["ts"] + e["dur"] for e in kernels), default=0)
    rows = sorted(([name, count, round(ms, 4)] for name, (count, ms) in groups.items()), key=lambda r: -r[2])
    windows.append({"trace": path.name, "first_kernel_ts": first, "span_ms": round((last - first) / 1000, 3),
                    "kernel_ms": round(sum(r[2] for r in rows), 3), "launches": len(kernels), "kernels": rows})
    del kernels
windows.sort(key=lambda w: w["first_kernel_ts"])
json.dump({"windows": windows}, open(out, "w"))
print(json.dumps([{"trace": w["trace"], "span_ms": w["span_ms"], "kernel_ms": w["kernel_ms"],
                   "launches": w["launches"]} for w in windows]))
