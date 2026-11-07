#!/usr/bin/env python3
"""Reduce the warp-specialized vision-linear stall to one small matrix multiply."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("operation", choices=["vision"])
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--stream", action="store_true")
parser.add_argument("--rows", type=int, default=128)
parser.add_argument("--inner", type=int, default=640)
parser.add_argument("--uniform", action="store_true")
parser.add_argument("--managed", action="store_true")
parser.add_argument("--no-sync", action="store_true")
parser.add_argument("--swiglu", action="store_true")
parser.add_argument("--repeats", type=int, default=1)
args = parser.parse_args()
import torch

collector = None
if args.stream:
    from stream.collector import enable

    collector = enable(
        args.out / "stream",
        ring_bytes=8 << 20,
        patched=True,
        uniform_registers=args.uniform,
        managed=args.managed,
        synchronize_warps=not args.no_sync,
    )
from vllm.models.deepseek_v4_1.tilelang.vision import (
    linear_kernel,
    swiglu_linear_kernel,
)

width = 2816 if args.swiglu else 1024
kernel = (
    swiglu_linear_kernel(width, args.inner)
    if args.swiglu
    else linear_kernel(width, args.inner, True)
)
x = torch.full((args.rows, args.inner), 0.5, dtype=torch.bfloat16, device="cuda")
w = torch.ones(
    (width * (2 if args.swiglu else 1), args.inner), dtype=torch.bfloat16, device="cuda"
)
bias = torch.ones(1024, dtype=torch.bfloat16, device="cuda")
out = torch.empty((args.rows, width), dtype=torch.bfloat16, device="cuda")
print("compiled vision linear; launching", flush=True)
for _ in range(args.repeats):
    if args.swiglu:
        kernel(x, w, out)
    else:
        kernel(x, w, bias, out)
torch.cuda.synchronize()
expected = torch.tensor(
    (args.inner / 2) ** 2 if args.swiglu else args.inner / 2 + 1, dtype=torch.bfloat16
).item()
assert bool((out == expected).all())
result = {
    "operation": "vision",
    "uniform": args.uniform,
    "sha256": hashlib.sha256(out.float().cpu().numpy().tobytes()).hexdigest(),
    "registers": kernel.n_regs,
    "spills": kernel.n_spills,
}
if collector:
    collector.close()
    from stream.export import records

    counts = Counter()
    for row in records(args.out / "stream"):
        counts["records"] += 1
        counts[f"kind_{row[9]}"] += 1
    assert counts["kind_0"] == counts["kind_3"] and counts["kind_1"] == counts["kind_2"]
    result["trace"] = dict(counts)
(args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
