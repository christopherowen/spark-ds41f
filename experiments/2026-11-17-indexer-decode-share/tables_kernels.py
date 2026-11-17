#!/usr/bin/env python3
"""Indexer kernel time per decode step, by window and rank.

usage: tables_kernels.py RECEIPTS_DIR

Reads decode.json (profile_decode.py's lines), decode-kernels-NODE.json
(summarize_kernels.py) and decode-busy-NODE.json (busy_time.py) of a profile job.
Windows alternate real-profiled and grid-profiled, in the depths' order.
"""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
phases = [json.loads(line) for line in (root / "decode.json").read_text().splitlines() if line.startswith("{")]
windows = {p["window"]: p for p in phases if "window" in p}
INDEXER = ("indexer_score_kernel", "indexer_topk_segment_kernel", "indexer_topk_merge_kernel",
           "indexer_mxfp4_quantize_kernel")

print("window  kind           depth      node  steps  launches/step  step ms  busy ms  idle ms  "
      "indexer sum / owned / alone ms   MoE owned ms")
for path in sorted(root.glob("decode-busy-*.json")):
    node = path.stem.rsplit("-", 1)[1]
    for index, w in enumerate(json.loads(path.read_text())["windows"]):
        steps, fam = w["steps"], w["families"]
        part = [sum(v[key] for label, v in fam.items() if label.startswith("indexer")) / steps
                for key in ("sum_ms", "owned_ms", "alone_ms")]
        phase = windows[index]
        print(f"{index:>6}  {phase['phase']:<13}  {phase['depth']:>9,}  {node:>4}  {steps:>5}  {w['launches'] / steps:>13.0f}  "
              f"{w['span_ms'] / steps:>7.2f}  {w['busy_ms'] / steps:>7.2f}  {w['idle_ms'] / steps:>7.2f}  "
              f"{part[0]:>11.3f} / {part[1]:.3f} / {part[2]:.3f}   {fam['MoE']['owned_ms'] / steps:>12.2f}")

print("\nindexer kernels, mean over ranks: launches per step, microseconds per launch, ms per step")
ranks = [json.loads(path.read_text())["windows"] for path in sorted(root.glob("decode-kernels-*.json"))]
steps_of = [w["steps"] for w in json.loads(sorted(root.glob("decode-busy-*.json"))[0].read_text())["windows"]]
for index, steps in enumerate(steps_of):
    phase = windows[index]
    cells = []
    for name in INDEXER:
        rows = [next(([n, c, ms] for n, c, ms in rank[index]["kernels"] if n == name), [name, 0, 0.0]) for rank in ranks]
        count = sum(r[1] for r in rows) / len(rows)
        ms = sum(r[2] for r in rows) / len(rows)
        cells.append(f"{name[8:-7]} {count / steps:.0f} x {1000 * ms / count if count else 0:.1f} us = {ms / steps:.3f}")
    print(f"  {phase['phase']:<13} {phase['depth']:>9,}  " + "; ".join(cells))
