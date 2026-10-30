# SPDX-License-Identifier: Apache-2.0
"""Benchmark and bit-for-bit verification of indexer query q_cast reuse + fused RoPE+MXFP4 quantize."""

import json
import torch
from torch import nn

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.models.deepseek_v4_1.tilelang.indexer import (
    INDEX_DIM,
    INDEX_HEADS,
    quantize_mxfp4,
)
from vllm.models.deepseek_v4_1.tilelang.linear import (
    TileLangBlock32Rows,
    cast_mxfp8,
    provide_mxfp8,
    slice_mxfp8,
)
from vllm.models.deepseek_v4_1.tilelang.rope import rope_
from vllm.v1.worker.workspace import use_preallocated_workspace

DEVICE = "cuda"


def _rope_table(positions_max=8192, rope_dim=64):
    inv = 1.0 / (
        10000 ** (torch.arange(0, rope_dim, 2, device=DEVICE).double() / rope_dim)
    )
    angles = torch.arange(positions_max, device=DEVICE).double()[:, None] * inv
    return torch.cat((angles.cos(), angles.sin()), dim=-1).float()


def main():
    gen = torch.Generator(device=DEVICE).manual_seed(42)
    n, k = INDEX_HEADS * INDEX_DIM, 1280  # (4096, 1280)
    weight = torch.randint(0, 256, (n, k), dtype=torch.uint8, device=DEVICE).view(
        torch.float8_e4m3fn
    )
    exps = torch.randint(120, 130, (n // 32, k // 32), dtype=torch.uint8, device=DEVICE)

    base_layer = nn.Module()
    base_layer.weight, base_layer.weight_scale_inv = weight, exps
    with set_current_vllm_config(VllmConfig()):
        wqb_casting = TileLangBlock32Rows(base_layer, 0)
        wqb_casting.declare_kernels()

        wqb_fed = TileLangBlock32Rows(base_layer, 0)
        wqb_fed.tilelang_mxfp8_input = True
        wqb_fed.declare_kernels()

    table = _rope_table()
    scratch = torch.empty(32 << 20, dtype=torch.uint8, device=DEVICE)

    results = {}
    with use_preallocated_workspace(scratch):
        for rows in (1, 4, 8, 16, 32, 64, 512, 2048, 4096):
            qr = torch.randn((rows, k), generator=gen, device=DEVICE).bfloat16()
            positions = torch.randint(0, 8192, (rows,), generator=gen, device=DEVICE)
            # In attention._forward, q_cast is already produced before wq_b and _index_query.
            q_cast = cast_mxfp8(qr)

            def run_baseline():
                iq = wqb_casting(qr).view(rows, INDEX_HEADS, INDEX_DIM)
                rope_(iq, positions, table, 1, False)
                q_out = torch.empty(
                    (rows, INDEX_HEADS, 64), dtype=torch.uint8, device=DEVICE
                )
                s_out = torch.empty(
                    (rows, INDEX_HEADS, 4), dtype=torch.uint8, device=DEVICE
                )
                quantize_mxfp4(iq, q_out, s_out)
                return q_out, s_out

            def run_fused():
                with provide_mxfp8(slice_mxfp8(q_cast, 0, rows)):
                    iq = wqb_fed(qr).view(rows, INDEX_HEADS, INDEX_DIM)
                q_out = torch.empty(
                    (rows, INDEX_HEADS, 64), dtype=torch.uint8, device=DEVICE
                )
                s_out = torch.empty(
                    (rows, INDEX_HEADS, 4), dtype=torch.uint8, device=DEVICE
                )
                quantize_mxfp4(iq, q_out, s_out, positions=positions, cos_sin=table)
                return q_out, s_out

            ref_q, ref_s = run_baseline()
            opt_q, opt_s = run_fused()
            assert torch.equal(ref_q, opt_q), f"q mismatch at rows={rows}"
            assert torch.equal(ref_s, opt_s), f"s mismatch at rows={rows}"

            # Warmup
            for _ in range(25):
                run_baseline()
                run_fused()
            torch.cuda.synchronize()

            iters = 200 if rows <= 64 else 80
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            for _ in range(iters):
                run_baseline()
            end.record()
            torch.cuda.synchronize()
            base_us = start.elapsed_time(end) * 1000.0 / iters

            start.record()
            for _ in range(iters):
                run_fused()
            end.record()
            torch.cuda.synchronize()
            opt_us = start.elapsed_time(end) * 1000.0 / iters

            results[str(rows)] = {
                "rows": rows,
                "baseline_us_per_layer": round(base_us, 2),
                "fused_us_per_layer": round(opt_us, 2),
                "saved_us_per_layer": round(base_us - opt_us, 2),
                "saved_us_11_layers": round((base_us - opt_us) * 11, 2),
                "speedup": round(base_us / opt_us, 3),
                "launches_per_layer_before": 4,
                "launches_per_layer_after": 2,
                "bit_exact": True,
            }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
