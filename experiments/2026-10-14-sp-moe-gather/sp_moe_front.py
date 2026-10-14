#!/usr/bin/env python3
"""The MoE front of each sequence-parallel layer in a torch prefill trace.

usage: sp_moe_front.py TRACE.json.gz [--rows 8096]

Anchors on the routed gate/up GEMM of a ROWS-row step (grid [9, tiles], tiles =
ceil(6 * ROWS / 32) + 384) and walks the main stream back to the preceding mHC
finalize. Reports per-layer medians of what the change moves to local rows: the
MoE input all-gather, the MXFP8 casts of the gathered rows (grid ROWS * 5120 /
2048), the router GEMM (grid [6, ceil(ROWS / 64)]) and the top-k; plus the
attention front's cast of local rows, the measured cost of one cast of L rows.
"""
import gzip
import json
import statistics
import sys

path = sys.argv[1]
rows = int(sys.argv[sys.argv.index("--rows") + 1]) if "--rows" in sys.argv else 8096
tiles = -(-6 * rows // 32) + 384
gathered_cast = rows * 5120 // 2048
local_cast = gathered_cast // 4

events = json.load(gzip.open(path))["traceEvents"]
kernels = [e for e in events if e.get("ph") == "X" and e.get("cat") == "kernel"]
main_stream = max({e["args"].get("stream") for e in kernels},
                  key=lambda s: sum(1 for e in kernels if e["args"].get("stream") == s))
main = sorted((e for e in kernels if e["args"].get("stream") == main_stream), key=lambda e: e["ts"])


def grid(event):
    return tuple(event["args"].get("grid", []))


layers = []
for i, event in enumerate(main):
    if not (event["name"].startswith("moe_grouped") and grid(event) == (9, tiles, 1)):
        continue
    j = i - 1
    while j >= 0 and not main[j]["name"].startswith("mhc_finalize"):
        j -= 1
    front = main[j + 1:i]
    after = main[i:i + 8]
    layers.append({
        "all_gather": sum(k["dur"] for k in front if "AllGather" in k["name"]),
        "casts": [k["dur"] for k in front
                  if k["name"].startswith("per_token_cast") and grid(k)[0] == gathered_cast],
        "router": sum(k["dur"] for k in front
                      if k["name"].startswith("gemm_bf16_kernel") and grid(k)[:2] == (6, -(-rows // 64))),
        "topk": sum(k["dur"] for k in front + after if k["name"].startswith("moe_topk_gate")),
    })

local = [e["dur"] for e in main if e["name"].startswith("per_token_cast") and grid(e)[0] == local_cast]
span = (main[-1]["ts"] + main[-1]["dur"] - main[0]["ts"]) / 1000


def median(values):
    values = list(values)
    return statistics.median(values) if values else 0.0


print(json.dumps({
    "trace": path.rsplit("/", 1)[-1],
    "rows": rows,
    "sp_layers": len(layers),
    "span_ms": round(span, 1),
    "median_us": {
        "all_gather": round(median(l["all_gather"] for l in layers)),
        "gathered_casts": round(median(sum(l["casts"]) for l in layers)),
        "gathered_casts_per_layer": median(len(l["casts"]) for l in layers),
        "router": round(median(l["router"] for l in layers)),
        "topk": round(median(l["topk"] for l in layers)),
        "local_cast": round(median(local)),
    },
}, indent=1))
