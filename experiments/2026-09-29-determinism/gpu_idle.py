#!/usr/bin/env python3
"""GPU busy time (union of kernel intervals, all streams) and idle gaps per target step.

usage: gpu_idle.py TRACE.json.gz STEPS
Prints span, busy union, idle, and the largest idle gaps with the kernels on either side.
"""
import gzip
import json
import sys

events = json.load(gzip.open(sys.argv[1], "rt"))["traceEvents"]
steps = float(sys.argv[2])
kernels = sorted((e["ts"], e["ts"] + e["dur"], e["name"]) for e in events if e.get("cat") == "kernel" and e.get("ph") == "X")
busy, gaps = 0.0, []
start, end, last = kernels[0][0], kernels[0][1], kernels[0][2]
for s, e, n in kernels[1:]:
    if s > end:
        busy += end - start
        gaps.append((s - end, last[:50], n[:50]))
        start, end = s, e
    if e > end:
        end, last = e, n
busy += end - start
span = kernels[-1][1] - kernels[0][0]
idle = span - busy
print(f"span {span / 1000 / steps:.3f} ms/step  busy {busy / 1000 / steps:.3f}  idle {idle / 1000 / steps:.3f}  "
      f"gaps>50us {sum(g > 50 for g, _, _ in gaps) / steps:.1f}/step")
gaps.sort(reverse=True)
for g, a, b in gaps[:6]:
    print(f"   {g / 1000:7.3f} ms after {a}  before {b}")
