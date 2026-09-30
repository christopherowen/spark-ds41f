#!/usr/bin/env python3
"""GPU kernel time per verification step, by category, from one rank's trace.

usage: analyze_steps.py TRACE.json.gz [TRACE2.json.gz]

Target and drafter routed-MoE launches are told apart by grid (96 vs 90 CTAs);
steps = target MoE launches / 40 layers. Prints, per trace: steps, kernel
time per step in total and by category (routed MoE, top-k sum, dense GEMM,
quantize, attention, all-reduce, mHC, L2 prefetch, other), and the span.
"""
import gzip
import json
import sys
from collections import defaultdict

LAYERS = 40


def category(name, grid):
    if "MoEDynamicKernel" in name:
        return "moe_target" if grid and grid[-1] == 96 else "moe_drafter"
    for key, label in (("TopKSum", "topk_sum"), ("DenseGemm", "dense_gemm"),
                       ("MXFP8RowsQuant", "quantize"), ("mla", "attention"), ("attention", "attention"),
                       ("Roce", "allreduce"), ("mhc", "mhc"), ("MHC", "mhc"),
                       ("L2Prefetch", "l2_prefetch"), ("gemv", "gemv"), ("Gemv", "gemv")):
        if key in name:
            return label
    return "other"


def summarize(path):
    with gzip.open(path, "rt") as handle:
        events = json.load(handle)["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"]
    totals = defaultdict(float)
    counts = defaultdict(int)
    for e in kernels:
        c = category(e["name"], e.get("args", {}).get("grid"))
        totals[c] += e["dur"] / 1000
        counts[c] += 1
    steps = counts["moe_target"] / LAYERS
    span = (max(e["ts"] + e["dur"] for e in kernels) - min(e["ts"] for e in kernels)) / 1000
    return steps, totals, counts, span


rows = [summarize(p) for p in sys.argv[1:]]
cats = sorted({c for _, t, _, _ in rows for c in t}, key=lambda c: -rows[0][1].get(c, 0))
print("category        " + "".join(f"{'trace ' + str(i + 1):>22s}" for i in range(len(rows))))
print("steps           " + "".join(f"{s:22.1f}" for s, _, _, _ in rows))
print("span ms         " + "".join(f"{sp:22.1f}" for _, _, _, sp in rows))
print("kernel ms/step  " + "".join(f"{sum(t.values()) / s:22.3f}" for s, t, _, _ in rows))
for c in cats:
    print(f"{c:16s}" + "".join(f"{t.get(c, 0) / s:16.3f} ms/st" for s, t, _, _ in rows))
