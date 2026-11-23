#!/usr/bin/env python3
"""Standalone TileLang MoE tile-schedule microbenchmark for DS4.1 TP=1 and TP=2 on GB10."""
from __future__ import annotations

import gc
import time
import torch

from vllm.models.deepseek_v4_1.tilelang import linear
from vllm.models.deepseek_v4_1.tilelang.moe import MoeExpertsV41Plan

HIDDEN = 5120
TOPK = 6
NUM_EXPERTS = 384


def bench_one(
    *,
    label: str,
    intermediate: int,
    num_slots: int,
    rows_list: list[int],
    decode_block_M: int = 16,
    block_M: int = 32,
    prefill_block_M: int = 64,
    prefill_swizzle_panel: int = 4,
    block_N: int = 128,
    num_stages: int = 3,
    down_num_stages: int = 2,
    narrow_codes: bool = True,
) -> dict[int, float]:
    g = torch.Generator(device="cuda").manual_seed(42)
    w13 = torch.randint(0, 256, (num_slots, intermediate * 2, HIDDEN // 2), dtype=torch.uint8, device="cuda", generator=g)
    s13 = torch.randint(122, 132, (num_slots, intermediate * 2, HIDDEN // 32), dtype=torch.uint8, device="cuda", generator=g)
    w2 = torch.randint(0, 256, (num_slots, HIDDEN, intermediate // 2), dtype=torch.uint8, device="cuda", generator=g)
    s2 = torch.randint(122, 132, (num_slots, HIDDEN, intermediate // 32), dtype=torch.uint8, device="cuda", generator=g)

    plan = MoeExpertsV41Plan(
        num_slots,
        TOPK,
        HIDDEN,
        intermediate,
        swiglu_limit=7.0,
        decode_block_M=decode_block_M,
        block_M=block_M,
        prefill_block_M=prefill_block_M,
        prefill_swizzle_panel=prefill_swizzle_panel,
        block_N=block_N,
        num_stages=num_stages,
        down_num_stages=down_num_stages,
        out_dtype=torch.float32,
        idx_dtypes=(torch.int64,),
        narrow_codes=narrow_codes,
        slot_map=False,
    )
    plan.declare_kernels()
    plan.bind()
    weights = plan.prepare_weights(w13, s13, w2, s2)
    del w13, s13, w2, s2

    res: dict[int, float] = {}
    for rows in rows_list:
        x_bf16 = torch.randn(rows, HIDDEN, dtype=torch.bfloat16, device="cuda")
        x_q = linear.cast_mxfp8(x_bf16)
        # Realistic Zipfian-ish routing across active slots
        ids = torch.randint(0, num_slots, (rows, TOPK), dtype=torch.int64, device="cuda", generator=g)
        w = torch.softmax(torch.randn(rows, TOPK, dtype=torch.float32, device="cuda"), dim=-1)
        out = torch.empty((rows, HIDDEN), dtype=torch.float32, device="cuda")

        # Warmup
        for _ in range(5):
            plan.run(x_q, ids, w, weights, out)
        torch.cuda.synchronize()

        iters = 40 if rows <= 64 else (15 if rows <= 2048 else 8)
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        for _ in range(iters):
            plan.run(x_q, ids, w, weights, out)
        ev1.record()
        torch.cuda.synchronize()
        ms = ev0.elapsed_time(ev1) / iters
        res[rows] = ms

    print(
        f"{label:42s} | "
        + " | ".join(f"r={r:4d}: {res[r]:6.3f}ms" for r in rows_list),
        flush=True,
    )
    del weights, plan
    gc.collect()
    torch.cuda.empty_cache()
    return res


def main() -> None:
    print("=== TP=2 (inter=1152, slots=384, narrow=True — pp2-tp2 & tp2) ===", flush=True)
    rows_tp2 = [6, 12, 24, 192, 4096]
    configs_tp2 = [
        ("baseline(dM16,mM32,pM64,sw4,N128,s3/2)", dict(decode_block_M=16, block_M=32, prefill_block_M=64, prefill_swizzle_panel=4, block_N=128, num_stages=3, down_num_stages=2)),
        ("down_stages=3", dict(decode_block_M=16, block_M=32, prefill_block_M=64, prefill_swizzle_panel=4, block_N=128, num_stages=3, down_num_stages=3)),
        ("stages=2/2", dict(decode_block_M=16, block_M=32, prefill_block_M=64, prefill_swizzle_panel=4, block_N=128, num_stages=2, down_num_stages=2)),
        ("prefill_M=32,sw=4", dict(decode_block_M=16, block_M=32, prefill_block_M=32, prefill_swizzle_panel=4, block_N=128, num_stages=3, down_num_stages=2)),
        ("prefill_M=128,sw=4", dict(decode_block_M=16, block_M=32, prefill_block_M=128, prefill_swizzle_panel=4, block_N=128, num_stages=3, down_num_stages=2)),
        ("prefill_sw=0", dict(decode_block_M=16, block_M=32, prefill_block_M=64, prefill_swizzle_panel=0, block_N=128, num_stages=3, down_num_stages=2)),
        ("prefill_sw=2", dict(decode_block_M=16, block_M=32, prefill_block_M=64, prefill_swizzle_panel=2, block_N=128, num_stages=3, down_num_stages=2)),
        ("prefill_sw=8", dict(decode_block_M=16, block_M=32, prefill_block_M=64, prefill_swizzle_panel=8, block_N=128, num_stages=3, down_num_stages=2)),
        ("decode_M=32", dict(decode_block_M=32, block_M=32, prefill_block_M=64, prefill_swizzle_panel=4, block_N=128, num_stages=3, down_num_stages=2)),
    ]
    for lbl, kw in configs_tp2:
        try:
            bench_one(label=lbl, intermediate=1152, num_slots=384, rows_list=rows_tp2, narrow_codes=True, **kw)
        except Exception as exc:
            print(f"{lbl:42s} | ERROR: {exc}", flush=True)

    print("\n=== TP=1 (inter=2304, slots=108, narrow=False — tp1 slot cache) ===", flush=True)
    rows_tp1 = [6, 24, 4096]
    configs_tp1 = [
        ("baseline(dM16,mM32,pM64,sw4,N128,s3/2)", dict(decode_block_M=16, block_M=32, prefill_block_M=64, prefill_swizzle_panel=4, block_N=128, num_stages=3, down_num_stages=2)),
        ("down_stages=3", dict(decode_block_M=16, block_M=32, prefill_block_M=64, prefill_swizzle_panel=4, block_N=128, num_stages=3, down_num_stages=3)),
        ("prefill_M=128,sw=4", dict(decode_block_M=16, block_M=32, prefill_block_M=128, prefill_swizzle_panel=4, block_N=128, num_stages=3, down_num_stages=2)),
        ("prefill_sw=2", dict(decode_block_M=16, block_M=32, prefill_block_M=64, prefill_swizzle_panel=2, block_N=128, num_stages=3, down_num_stages=2)),
        ("prefill_sw=8", dict(decode_block_M=16, block_M=32, prefill_block_M=64, prefill_swizzle_panel=8, block_N=128, num_stages=3, down_num_stages=2)),
    ]
    for lbl, kw in configs_tp1:
        try:
            bench_one(label=lbl, intermediate=2304, num_slots=108, rows_list=rows_tp1, narrow_codes=False, **kw)
        except Exception as exc:
            print(f"{lbl:42s} | ERROR: {exc}", flush=True)


if __name__ == "__main__":
    main()
