#!/usr/bin/env python3
"""The routed experts' SwiGLU kernel between its two grouped GEMMs, in a torch trace.

usage: swiglu_gap.py TRACE.json.gz

For every `swiglu_forward_and_per_token_cast` launch that sits between two
`moe_grouped_fp8_fp4_gemm` launches on one stream, reports by the gate/up GEMM's
tile count (its grid's second dimension): the SwiGLU kernel's duration, its
overlap with the gate/up GEMM (negative: a gap), the time from the gate/up GEMM's
end to the down GEMM's start (what a faster or fused SwiGLU can remove), and the
two GEMMs' durations. Medians, microseconds.
"""
import gzip
import json
import statistics
import sys

events = json.load(gzip.open(sys.argv[1]))["traceEvents"]
streams = {}
for event in events:
    if event.get("ph") == "X" and event.get("cat") == "kernel":
        streams.setdefault(event["args"].get("stream"), []).append(event)

groups = {}
for launches in streams.values():
    launches.sort(key=lambda e: e["ts"])
    for before, event, after in zip(launches, launches[1:], launches[2:]):
        if not (event["name"].startswith("swiglu_forward_and_per_token_cast")
                and before["name"].startswith("moe_grouped") and after["name"].startswith("moe_grouped")):
            continue
        end = before["ts"] + before["dur"]
        groups.setdefault(before["args"]["grid"][1], []).append(
            (event["dur"], end - event["ts"], after["ts"] - end, before["dur"], after["dur"]))

print(f"{'tiles':>6} {'calls':>6} {'swiglu':>8} {'overlap':>8} {'serial':>8} {'gate/up':>9} {'down':>8}")
for tiles, rows in sorted(groups.items(), key=lambda item: -len(item[1])):
    medians = [statistics.median(r[k] for r in rows) for k in range(5)]
    print(f"{tiles:>6} {len(rows):>6} " + " ".join(f"{m:8.1f}" for m in medians[:3])
          + f" {medians[3]:9.1f} {medians[4]:8.1f}")
