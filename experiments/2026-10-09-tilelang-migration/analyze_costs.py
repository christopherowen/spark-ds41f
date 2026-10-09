#!/usr/bin/env python3
"""Kernel time per target step by family, two arms on the same captured workload.

usage: analyze_costs.py BASE-kernels.json CANDIDATE-kernels.json [--total | --steps-base N --steps-candidate N] [--top N]

Inputs are summarize_kernels.py outputs (rank 0). Steps are the target model's
forward passes, counted by its LM-head projection (once per step: the Triton
row kernel or F.linear's cuBLAS kernel on the 43136-row shard); --total
reports whole-window totals instead (prefill: the arms chunk differently). Families
follow the reference changes they measure: the target MoE (deterministic MoE,
vllm-0034), attention decode and extend kernels (0035), mHC (0033), B12X GEMVs
(0029, 0032), cuBLAS and Triton kernels (the vocabulary projection, 0038, and
torch-backend GEMVs), B12X dense GEMMs (projections, WO, the drafter's heads:
0030, 0037), NCCL and elementwise kernels (prefill reduce-scatter, 0031), the
RoCE collectives and the rest. Prints ms per step for both arms and the
difference, then the kernels whose time per step changed most.
"""
import json
import sys
from collections import defaultdict

LAYERS = 40
RULES = (  # (substring, family); first match wins
    ("Materialized", "moe target"), ("MoEDynamicKernel", "moe target"), ("moe", "moe target"),
    ("TopKSum", "moe target"),
    ("MHC", "mhc"), ("mhc", "mhc"), ("HyperConnection", "mhc"),
    ("Gemv", "b12x gemv"), ("gemv", "b12x gemv"),
    ("Decode", "attention decode"), ("Extend", "attention extend"), ("Prefill", "attention extend"),
    ("mla", "attention other"), ("attention", "attention other"), ("indexed_pages", "attention other"),
    ("_row_kernel", "triton vocabulary row"), ("_row_loop_kernel", "triton vocabulary row"),
    ("nvjet", "cublas"), ("cutlass_80", "cublas"), ("gemv2", "cublas"), ("gemvx", "cublas"), ("gemmSN", "cublas"),
    ("sm90_xmma", "cublas"), ("cublas", "cublas"),
    ("DenseGemm", "b12x dense gemm"), ("MXFP8RowsQuant", "quantize"), ("Quant", "quantize"),
    ("Roce", "roce collectives"), ("nccl", "nccl"),
    ("L2Prefetch", "l2 prefetch"),
    ("at::native", "torch elementwise"), ("triton", "triton other"),
)


def family(name, grid):
    for key, label in RULES:
        if key in name:
            if label == "moe target" and grid and grid[-1] != 96 and ("Dynamic" in name or "Materialized" in name):
                return "moe drafter"
            return label
    return "other"


def head_launch(name, grid):
    """The target LM head's vocabulary projection on rank 0's 43136-row shard."""
    return bool(grid) and (("_row_kernel" in name and grid[0] == 43136)
                           or ("nvjet" in name and grid[0] == 2696)
                           or ("cutlass_80" in name and grid[1] == 337))


def load(path):
    d = json.load(open(path))
    fam, by_kernel = defaultdict(float), defaultdict(float)
    steps = 0
    for name, grid, block, launches, ms in d["kernels"]:
        f = family(name, grid)
        fam[f] += ms
        by_kernel[(f, name[:90], json.dumps(grid))] += ms
        steps += launches if head_launch(name, grid) else 0
    return d, fam, by_kernel, steps


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


(db, fb, kb, sb), (dc, fc, kc, sc) = load(sys.argv[1]), load(sys.argv[2])
sb = 1.0 if "--total" in sys.argv else float(arg("--steps-base", sb))
sc = 1.0 if "--total" in sys.argv else float(arg("--steps-candidate", sc))
top = int(arg("--top", "12"))
unit = "window" if "--total" in sys.argv else "step"
print(f"steps {sb:.1f} / {sc:.1f}   span ms/step {db['span_ms'] / sb:.3f} / {dc['span_ms'] / sc:.3f}   "
      f"kernel ms/step {db['kernel_ms'] / sb:.3f} / {dc['kernel_ms'] / sc:.3f} "
      f"({dc['kernel_ms'] / sc - db['kernel_ms'] / sb:+.3f})")
for f in sorted(set(fb) | set(fc), key=lambda f: -abs(fc.get(f, 0) / sc - fb.get(f, 0) / sb)):
    b, c = fb.get(f, 0) / sb, fc.get(f, 0) / sc
    print(f"  {f:24s} {b:9.3f} {c:9.3f} {c - b:+9.3f} ms/{unit}")
print(f"kernels changing most (ms/{unit}, base -> candidate):")
for key in sorted(set(kb) | set(kc), key=lambda k: -abs(kc.get(k, 0) / sc - kb.get(k, 0) / sb))[:top]:
    b, c = kb.get(key, 0) / sb, kc.get(key, 0) / sc
    print(f"  {c - b:+8.3f} ({b:7.3f} -> {c:7.3f})  [{key[0]}] {key[1]} grid {key[2]}")
