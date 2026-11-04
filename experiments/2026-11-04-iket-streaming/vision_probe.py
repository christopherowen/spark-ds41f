#!/usr/bin/env python3
"""Reduce the warp-specialized vision-linear stall to one small matrix multiply."""

import argparse
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
args = parser.parse_args()
import torch

collector = None
if args.stream:
    from stream.collector import enable

    collector = enable(
        args.out / "stream",
        ring_bytes=1 << 20,
        patched=True,
        uniform_registers=args.uniform,
        managed=args.managed,
    )
from vllm.models.deepseek_v4_1.tilelang.vision import linear_kernel

kernel = linear_kernel(1024, args.inner, True)
x = torch.full((args.rows, args.inner), 0.5, dtype=torch.bfloat16, device="cuda")
w = torch.ones((1024, args.inner), dtype=torch.bfloat16, device="cuda")
bias = torch.ones(1024, dtype=torch.bfloat16, device="cuda")
out = torch.empty((args.rows, 1024), dtype=torch.bfloat16, device="cuda")
print("compiled vision linear; launching", flush=True)
kernel(x, w, bias, out)
torch.cuda.synchronize()
expected = torch.tensor(args.inner / 2 + 1, dtype=torch.bfloat16).item()
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
    from stream.export import export

    result["trace"] = export(args.out / "stream", args.out / "trace.json")
(args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
