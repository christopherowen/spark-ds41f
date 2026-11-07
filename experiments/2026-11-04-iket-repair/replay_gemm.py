#!/usr/bin/env python3
"""Candidate-38 GEMM replay, with compile-only capacity preflight."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("operation", choices=["gemm"])
parser.add_argument("--out", required=True, type=Path)
parser.add_argument("--instrument", action="store_true")
parser.add_argument("--compile-only", action="store_true")
parser.add_argument("--reference-sources", type=Path)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)

import torch

torch.cuda.init()
if args.instrument:
    from vllm.models.deepseek_v4_1.tilelang import iket
    iket.enable(args.out, ctas=1, iters=1, warps=1, granularity="phase")
from vllm.models.deepseek_v4_1.tilelang.gemm import bf16_gemm

gemm = bf16_gemm(128, 512, out_dtype="float32")

def manifests(directory):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.glob("*.json"))}

sources = manifests(args.out / "sources")
if args.instrument and not sources:
    raise RuntimeError("compilation emitted no event manifests")
if args.reference_sources and sources != manifests(args.reference_sources):
    raise RuntimeError("replay manifests differ from the certified preflight")
receipt = {"operation": "gemm", "shape": [6, 128, 512], "instrument": args.instrument,
           "compile_only": args.compile_only, "manifests_sha256": sources}
if not args.compile_only:
    # Binary fractions give an exact, non-uniform FP32 reference on the CPU.
    rows = torch.arange(1, 7, dtype=torch.float32) / 8
    signs = torch.where(torch.arange(128) % 2 == 0, 1., -1.)
    x = rows[:, None].expand(6, 512).contiguous().to(device="cuda", dtype=torch.bfloat16)
    weight = signs[:, None].expand(128, 512).contiguous().to(device="cuda", dtype=torch.bfloat16)
    output = torch.empty((6, 128), device="cuda", dtype=torch.float32)
    gemm(x, weight, output)
    torch.cuda.synchronize()
    result = output.cpu()
    assert torch.equal(result, rows[:, None] * signs[None, :] * 512)
    receipt["sha256"] = hashlib.sha256(result.numpy().tobytes()).hexdigest()
    receipt["bytes"] = result.numel() * result.element_size()
(args.out / "result.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt), flush=True)
