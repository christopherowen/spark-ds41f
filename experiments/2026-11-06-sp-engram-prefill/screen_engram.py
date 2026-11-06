#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""2-boot GPU screening & isolated micro-benchmark harness for SP Engram prefill.

Do NOT execute without an active cluster hold (~/spark-hold.json) and explicit
window authorization.

Usage (inside a scheduled GPU window):
  # 1. Isolated multi-rank GPU microbenchmark & bit-identity check (TP4 or TP3):
  torchrun --nproc_per_node=4 experiments/2026-11-06-sp-engram-prefill/screen_engram.py \
      --tokens 2048,4096,8192,16384 --warmup 10 --iters 50

  # 2. End-to-end 2-boot bin/spark bench screen (against recorded Candidate 1107 baseline):
  bin/spark cluster start --cluster-config experiments/2026-11-06-sp-engram-prefill/tp4.json --apply
  bin/spark bench --cluster-config experiments/2026-11-06-sp-engram-prefill/tp4.json \
      --output experiments/2026-11-06-sp-engram-prefill/bench-tp4-boot1.json
  bin/spark cluster restart --cluster-config experiments/2026-11-06-sp-engram-prefill/tp4.json --apply
  bin/spark bench --cluster-config experiments/2026-11-06-sp-engram-prefill/tp4.json \
      --output experiments/2026-11-06-sp-engram-prefill/bench-tp4-boot2.json
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.distributed as dist


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokens",
        type=str,
        default="2048,4096,6144,8192,16384",
        help="Comma-separated prefill chunk token counts to screen",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--output", type=str, default="")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    token_counts = [int(x) for x in args.tokens.split(",") if x.strip()]
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    device = torch.device("cuda", torch.cuda.current_device())

    from vllm.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.models.deepseek_v4_1 import sp_prefill
    from vllm.models.deepseek_v4_1.common import engram as engram_ops

    if not sp_prefill.model_parallel_is_initialized():
        initialize_model_parallel(tensor_model_parallel_size=world_size)

    max_cols = (24 + world_size - 1) // world_size
    results = []
    for num_tokens in token_counts:
        layout = sp_prefill.SPRows(num_tokens, world_size, rank)
        gen = torch.Generator(device="cpu").manual_seed(1700 + 13 * rank + num_tokens)
        staged = torch.randn(
            (num_tokens, max_cols * 256), generator=gen, dtype=torch.bfloat16
        ).to(device)

        # Verify bit-identical equality between interleaved_rows slice and interleaved_sp_rows
        full_rows = engram_ops.interleaved_rows(staged)
        local_rows = engram_ops.interleaved_sp_rows(staged, layout)
        real = layout.real_rows
        expected = full_rows[layout.start : layout.start + real]
        assert torch.equal(
            local_rows.view(torch.int16), expected.view(torch.int16)
        ), f"Rank {rank}: mismatch at T={num_tokens}"

        for _ in range(args.warmup):
            _ = engram_ops.interleaved_rows(staged)
            _ = engram_ops.interleaved_sp_rows(staged, layout)
        torch.cuda.synchronize(device)

        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)

        start_evt.record()
        for _ in range(args.iters):
            _ = engram_ops.interleaved_rows(staged)
        end_evt.record()
        torch.cuda.synchronize(device)
        base_ms = start_evt.elapsed_time(end_evt) / args.iters

        start_evt.record()
        for _ in range(args.iters):
            _ = engram_ops.interleaved_sp_rows(staged, layout)
        end_evt.record()
        torch.cuda.synchronize(device)
        cand_ms = start_evt.elapsed_time(end_evt) / args.iters

        rec = {
            "tokens": num_tokens,
            "world_size": world_size,
            "local_rows": layout.local_rows,
            "real_rows": real,
            "interleaved_rows_ms": round(base_ms, 4),
            "interleaved_sp_rows_ms": round(cand_ms, 4),
            "speedup": round(base_ms / max(cand_ms, 1e-6), 3),
            "bit_identical": True,
        }
        results.append(rec)
        if rank == 0:
            print(json.dumps(rec), flush=True)

    if rank == 0 and args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"results": results}, f, indent=2)
            f.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
