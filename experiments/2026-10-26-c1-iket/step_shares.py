#!/usr/bin/env python3
"""Each kernel family's share of a TP4 single-stream decode step, from rank 0's traces.

usage: step_shares.py [TRACES] [--windows greedy sampled] [--json OUT]

TRACES is request-inventory-5's ports directory (results/private/determinism/
request-inventory-5/ports), the traces #598 and the fusion survey read. Steps, and each
kernel's ``excl`` (the time it runs with no other kernel but the L2 prefetch beside it)
and ``gap`` (idle time before it), come from the fusion survey's ceilings.py. A
kernel's critical-path cost is excl + gap; the shared expert's kernels on their own
stream mostly overlap the routed experts, so their cost is small next to their duration.

Families are named by kernel and grid at TP4's shapes (gemm.DECODE_FP8_CONFIGS):
gemm_mxfp8 grid 28 is the fused Q-A/KV (1792 / 64), 64 Q-B and the indexer's Q-B, 40
WO-B and the shared down (5120 / 128), 36 the shared gate/up (1152 / 32); the routed
grouped GEMM with 9 column blocks is FC1 (1152 / 128), with 40 FC2.
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "2026-10-25-fusion-survey"))
import ceilings  # noqa: E402

TRACES = os.path.expanduser("~/projects/spark-ds41f/results/private/determinism/request-inventory-5/ports")


def family(name, grid, drafter):
    g = grid[0] if grid else 0
    prefix = "drafter " if drafter else ""
    if name.startswith("moe_grouped"):
        return prefix + ("routed FC1" if g == 9 else "routed FC2")
    if name.startswith(("moe_dispatch", "moe_swiglu_cast", "moe_combine")):
        return prefix + "routed dispatch, SwiGLU, combine"
    if name.startswith("oneshot_allreduce"):
        return prefix + "all-reduce"
    if name.startswith("oneshot_allgather"):
        return prefix + "all-gather"
    if name.startswith("vocab_packed"):
        return prefix + "LM head"
    if name.startswith("gemm_mxfp8_kernel"):
        return prefix + {28: "Q-A/KV GEMM", 64: "Q-B GEMM", 40: "WO-B and shared down GEMMs",
                         36: "shared gate/up GEMM", 200: "DSpark main projection",
                         20: "Engram wkv GEMM"}.get(g, f"gemm_mxfp8 grid {g}")
    if name.startswith("gemm_fp8w_bf16_partials") or name.startswith("gemm_splitk_reduce_cast"):
        return prefix + "WO-A (partials, reduce-cast)"
    if name.startswith(("gemm_bf16_partials", "gemm_splitk_reduce_kernel")):
        return prefix + "router gate and small BF16 projections"
    if name.startswith(("gemm_bf16_splitk", "gemm_bf16_kernel", "gemm_mxfp8_partials")):
        return prefix + "other GEMMs (compressors, Markov head, KV)"
    if name.startswith("mhc_"):
        return prefix + "mHC (project, finalize, post)"
    if name.startswith(("sparse_mla",)):
        return prefix + "sparse MLA"
    if name.startswith(("indexer_", "metadata_", "cache_writer", "rope_", "compress")):
        return prefix + "attention small (indexer, metadata, cache, RoPE, compress)"
    if name.startswith(("per_token_cast", "norm_forward")):
        return prefix + "casts and norms"
    return prefix + "other small kernels and copies"


def window(path, skip_edges=True):
    ks = ceilings.load(path)
    pairs = ceilings.step_pairs(ks)
    if skip_edges:
        pairs = pairs[1:-1]
    steps = []
    for target, drafter, start, end in pairs:
        fam = defaultdict(lambda: {"dur": 0.0, "cost": 0.0, "count": 0})
        for rows, is_drafter in ((target, False), (drafter, True)):
            for r in rows:
                f = fam[family(r["name"], r["grid"], is_drafter)]
                f["dur"] += r["dur"]
                f["cost"] += r["excl"] + r["gap"]
                f["count"] += 1
        steps.append({"us": end - start, "families": fam})
    return steps


def summarize(steps):
    n = len(steps)
    step_us = sum(s["us"] for s in steps) / n
    names = sorted({k for s in steps for k in s["families"]})
    out = []
    for name in names:
        dur = sum(s["families"][name]["dur"] for s in steps if name in s["families"]) / n
        cost = sum(s["families"][name]["cost"] for s in steps if name in s["families"]) / n
        count = sum(s["families"][name]["count"] for s in steps if name in s["families"]) / n
        out.append({"family": name, "launches": round(count, 1), "dur_us": round(dur, 1),
                    "cost_us": round(cost, 1), "share": round(cost / step_us, 4)})
    out.sort(key=lambda e: -e["cost_us"])
    covered = sum(e["cost_us"] for e in out)
    return {"steps": n, "step_us": round(step_us, 1), "covered_us": round(covered, 1),
            "launch_gaps_and_host_us": round(step_us - covered, 1), "families": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="?", default=TRACES)
    ap.add_argument("--windows", nargs="*", default=["greedy", "sampled"])
    ap.add_argument("--json")
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.traces, "decode-trace", "dgx1", "*.pt.trace.json.gz")))
    by_name = dict(zip(ceilings.DECODE, files))
    result = {}
    for name in args.windows:
        result[name] = summarize(window(by_name[name]))
        r = result[name]
        print(f"\n## c1 {name}: {r['steps']} steps of {r['step_us'] / 1000:.2f} ms; "
              f"kernels' excl + gap {r['covered_us'] / 1000:.2f} ms\n")
        print("| family | launches | kernel us | critical us | share |")
        print("|---|---:|---:|---:|---:|")
        for e in r["families"]:
            print(f"| {e['family']} | {e['launches']} | {e['dur_us']:.0f} | {e['cost_us']:.0f} | {e['share']:.1%} |")
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(result, handle, indent=1)
            handle.write("\n")


if __name__ == "__main__":
    main()
