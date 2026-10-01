#!/usr/bin/env python3
"""Kernel time per target step by family, two arms on the same captured workload.

usage: analyze_costs.py BASE-kernels.json CANDIDATE-kernels.json [--steps-base N --steps-candidate N] [--top N]

Inputs are summarize_kernels.py outputs (rank 0). Steps default to the target
model's routed-MoE launches over 40 layers (target launches use 96 CTAs, the
drafter's fewer; the M=1 materialized path counts its first phase). Families
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
    ("Decode", "attention decode"), ("Extend", "attention extend"), ("Prefill", "attention extend"),
    ("mla", "attention other"), ("attention", "attention other"), ("indexed_pages", "attention other"),
    ("MHC", "mhc"), ("mhc", "mhc"), ("HyperConnection", "mhc"),
    ("Gemv", "b12x gemv"), ("gemv", "b12x gemv"),
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


def load(path):
    d = json.load(open(path))
    fam, by_kernel = defaultdict(float), defaultdict(float)
    moe_launches = 0
    for name, grid, block, launches, ms in d["kernels"]:
        f = family(name, grid)
        fam[f] += ms
        by_kernel[(f, name[:90], json.dumps(grid))] += ms
        if f == "moe target" and ("MoEDynamicKernel" in name or "Phase1" in name):
            moe_launches += launches
    return d, fam, by_kernel, moe_launches / LAYERS


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


(db, fb, kb, sb), (dc, fc, kc, sc) = load(sys.argv[1]), load(sys.argv[2])
sb = float(arg("--steps-base", sb))
sc = float(arg("--steps-candidate", sc))
top = int(arg("--top", "12"))
print(f"steps {sb:.1f} / {sc:.1f}   span ms/step {db['span_ms'] / sb:.3f} / {dc['span_ms'] / sc:.3f}   "
      f"kernel ms/step {db['kernel_ms'] / sb:.3f} / {dc['kernel_ms'] / sc:.3f} "
      f"({dc['kernel_ms'] / sc - db['kernel_ms'] / sb:+.3f})")
for f in sorted(set(fb) | set(fc), key=lambda f: -abs(fc.get(f, 0) / sc - fb.get(f, 0) / sb)):
    b, c = fb.get(f, 0) / sb, fc.get(f, 0) / sc
    print(f"  {f:24s} {b:9.3f} {c:9.3f} {c - b:+9.3f} ms/step")
print("kernels changing most (ms/step, base -> candidate):")
for key in sorted(set(kb) | set(kc), key=lambda k: -abs(kc.get(k, 0) / sc - kb.get(k, 0) / sb))[:top]:
    b, c = kb.get(key, 0) / sb, kc.get(key, 0) / sc
    print(f"  {c - b:+8.3f} ({b:7.3f} -> {c:7.3f})  [{key[0]}] {key[1]} grid {key[2]}")
