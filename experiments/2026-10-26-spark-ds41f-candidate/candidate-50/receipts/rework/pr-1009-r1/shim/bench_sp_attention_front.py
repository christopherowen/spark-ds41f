#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Bit-exactness verification and GPU kernel benchmark for SP prefill local
attention front (`sp_attention_front` + `write_swa_records` + local-only
`index_weights`) on DeepSeek-V4.1 (54 non-compressor layers)."""

import argparse
import json
import math
import time

import torch

from vllm.models.deepseek_v4_1 import kernels, sp_prefill
from vllm.models.deepseek_v4_1.tilelang import columns
from vllm.models.deepseek_v4_1.tilelang.cache_writer import (
    RECORD_BYTES,
    write_cache,
    write_swa_records,
)
from vllm.models.deepseek_v4_1.tilelang.front import (
    KV_LATENT,
    Q_LATENT,
    ROPE_DIM,
    sp_attention_front,
)
from vllm.models.deepseek_v4_1.tilelang.linear import (
    MXFP8Rows,
    empty_mxfp8,
    mxfp8_rows,
    slice_mxfp8,
)
from vllm.models.deepseek_v4_1.tilelang.pdl import norm_forward
from vllm.models.deepseek_v4_1.tilelang.rope import rope_

DEVICE = "cuda"
EPS = 1e-6
HEADS = 32
PAGE_SIZE = 64
NON_COMPRESSOR_LAYERS = 54


def _make_cos_sin(max_pos: int = 32768) -> torch.Tensor:
    inv = 1.0 / (
        10000 ** (torch.arange(0, ROPE_DIM, 2, device=DEVICE).double() / ROPE_DIM)
    )
    angles = torch.arange(max_pos, device=DEVICE).double()[:, None] * inv
    return torch.cat((angles.cos(), angles.sin()), dim=-1).float()


def _sim_gather_many(
    layouts: list[sp_prefill.SPRows],
    per_rank_tensors: list[tuple[torch.Tensor, ...]],
    padded_rows: tuple[int, ...] = (),
) -> tuple[torch.Tensor, ...]:
    num_tokens = layouts[0].num_tokens
    n_tensors = len(per_rank_tensors[0])
    limits = padded_rows + (num_tokens,) * (n_tensors - len(padded_rows))
    outs = []
    for idx, limit in enumerate(limits):
        cat = torch.cat([rank_t[idx] for rank_t in per_rank_tensors], dim=0)
        if limit > cat.shape[0]:
            buf = cat.new_zeros((limit, *cat.shape[1:]))
            buf[: cat.shape[0]].copy_(cat)
            outs.append(buf)
        else:
            outs.append(cat[:limit].contiguous())
    return tuple(outs)


def verify_bit_exactness(cos_sin: torch.Tensor) -> list[dict]:
    results = []
    q_weight = (torch.randn(Q_LATENT, device=DEVICE) * 0.3 + 1.0).bfloat16()
    kv_weight = (torch.randn(KV_LATENT, device=DEVICE) * 0.3 + 1.0).bfloat16()
    q_weight[:32] = 0.0
    q_weight[32:64] *= 1024.0
    w_proj = (torch.randn((HEADS, 128), device=DEVICE) * 0.1).bfloat16()


    for world_size in (3, 4):
        for num_tokens in (60, 65, 205, 206, 207, 512, 4096):
            gen = torch.Generator(device=DEVICE).manual_seed(world_size * 10000 + num_tokens)
            layouts = [
                sp_prefill.SPRows(num_tokens, world_size, r) for r in range(world_size)
            ]
            padded = layouts[0].padded_rows
            local = layouts[0].local_rows

            full_qr_kv = (
                torch.randn((padded, Q_LATENT + KV_LATENT), generator=gen, device=DEVICE)
                * 2.5
            ).bfloat16()
            full_qr_kv[0, :64] = -0.0
            full_qr_kv[1, :] = 0.0
            full_qr_kv[2, Q_LATENT + 12] = 1e4
            full_qr_kv[3, :] *= 1e-30
            full_h = (
                torch.randn((padded, 128), generator=gen, device=DEVICE) * 0.5
            ).bfloat16()
            positions = torch.randint(
                0, 16384, (num_tokens,), generator=gen, device=DEVICE, dtype=torch.int64
            )
            num_pages = (num_tokens // PAGE_SIZE) + 8
            slots = torch.randperm(
                num_pages * PAGE_SIZE, generator=gen, device=DEVICE, dtype=torch.int64
            )[:num_tokens]
            slots[num_tokens // 3] = -1

            base_pages = torch.empty(
                (num_pages, PAGE_SIZE * RECORD_BYTES["swa"]),
                dtype=torch.uint8,
                device=DEVICE,
            )
            base_pages.random_(generator=gen)

            # 1. Baseline 1107 (_gather_front + cast_mxfp8(T) + rope_(T) + write_cache(T))
            widths = (Q_LATENT, KV_LATENT, HEADS)
            fronts = []
            for r, sp in enumerate(layouts):
                qr_kv_r = full_qr_kv[sp.start : sp.start + local]
                h_r = full_h[sp.start : sp.start + local]
                qr_r, kv_r = qr_kv_r.split((Q_LATENT, KV_LATENT), dim=-1)
                front_r = qr_r.new_empty((local, sum(widths)))
                norm_forward(qr_r.contiguous(), q_weight, EPS, out=front_r[:, :Q_LATENT])
                norm_forward(
                    kv_r.contiguous(),
                    kv_weight,
                    EPS,
                    out=front_r[:, Q_LATENT : Q_LATENT + KV_LATENT],
                )
                w_r = torch.nn.functional.linear(h_r, w_proj)
                columns.copy_columns(w_r, front_r[:, Q_LATENT + KV_LATENT :])
                fronts.append(front_r)
            gathered_front = torch.cat(fronts, dim=0)[:num_tokens]
            qr_base, kv_base, w_base = columns.split_columns(gathered_front, widths)
            q_cast_base = kernels.cast_mxfp8(qr_base)
            kv_base_rot = kv_base.clone()
            rope_(kv_base_rot, positions, cos_sin, 1, False)
            pages_base = base_pages.clone()
            write_cache(
                kv_base_rot, pages_base, slots, page_size=PAGE_SIZE, cache_kind="swa"
            )

            # 2. Option A (Local q_norm + cast_mxfp8 + local kv_norm + rope_ + local index_weights)
            rank_pairs_a = []
            local_w_a = []
            for r, sp in enumerate(layouts):
                qr_kv_r = full_qr_kv[sp.start : sp.start + local]
                h_r = full_h[sp.start : sp.start + local]
                real = sp.real_rows
                local_w_a.append(torch.nn.functional.linear(h_r, w_proj)[:real])
                qr_r, kv_r = qr_kv_r.split((Q_LATENT, KV_LATENT), dim=-1)
                qn_r = torch.empty((local, Q_LATENT), dtype=torch.bfloat16, device=DEVICE)
                kvn_r = torch.empty((local, KV_LATENT), dtype=torch.bfloat16, device=DEVICE)
                norm_forward(qr_r.contiguous(), q_weight, EPS, out=qn_r)
                norm_forward(kv_r.contiguous(), kv_weight, EPS, out=kvn_r)
                q_cast_r = kernels.cast_mxfp8(qn_r)
                if real > 0:
                    rope_(
                        kvn_r[:real],
                        positions[sp.start : sp.start + real],
                        cos_sin,
                        1,
                        False,
                    )
                rank_pairs_a.append((q_cast_r.values[:local], q_cast_r.scales[:local], kvn_r))
            rows_q = mxfp8_rows(num_tokens)
            vals_a, scales_a, kv_a = _sim_gather_many(
                layouts, rank_pairs_a, padded_rows=(rows_q, rows_q)
            )
            pages_a = base_pages.clone()
            write_cache(kv_a, pages_a, slots, page_size=PAGE_SIZE, cache_kind="swa")

            # 3. Option B (Promoted: sp_attention_front + gather_mxfp8 + write_swa_records + local index_weights)
            rank_pairs_b = []
            local_w_b = []
            for r, sp in enumerate(layouts):
                qr_kv_r = full_qr_kv[sp.start : sp.start + local]
                h_r = full_h[sp.start : sp.start + local]
                real = sp.real_rows
                local_w_b.append(torch.nn.functional.linear(h_r, w_proj)[:real])
                local_qr = torch.empty(
                    (local, Q_LATENT), dtype=torch.bfloat16, device=DEVICE
                )
                local_q_cast = empty_mxfp8(local_qr)
                local_records = torch.empty(
                    (local, RECORD_BYTES["swa"]), dtype=torch.uint8, device=DEVICE
                )
                if real > 0:
                    sp_attention_front(
                        qr_kv_r[:real],
                        q_weight,
                        kv_weight,
                        EPS,
                        positions[sp.start : sp.start + real],
                        cos_sin,
                        local_q_cast.values[:real],
                        local_q_cast.scales[:real],
                        local_records[:real],
                    )
                rank_pairs_b.append(
                    (
                        local_q_cast.values[:local],
                        local_q_cast.scales[:local],
                        local_records,
                    )
                )
            vals_b, scales_b, records_b = _sim_gather_many(
                layouts, rank_pairs_b, padded_rows=(rows_q, rows_q)
            )
            q_cast_b = MXFP8Rows(
                torch.empty((num_tokens, Q_LATENT), dtype=torch.bfloat16, device=DEVICE),
                vals_b,
                scales_b,
            )
            pages_b = base_pages.clone()
            write_swa_records(records_b, pages_b, slots, page_size=PAGE_SIZE)

            q_val_diffs_a = int(
                (
                    vals_a[:num_tokens].view(torch.uint8)
                    != q_cast_base.values[:num_tokens].view(torch.uint8)
                ).sum()
            )
            q_scale_diffs_a = int(
                (
                    scales_a[:num_tokens].view(torch.uint8)
                    != q_cast_base.scales[:num_tokens].view(torch.uint8)
                ).sum()
            )
            cache_diffs_a = int((pages_a != pages_base).sum())

            q_val_diffs_b = int(
                (
                    vals_b[:num_tokens].view(torch.uint8)
                    != q_cast_base.values[:num_tokens].view(torch.uint8)
                ).sum()
            )
            q_scale_diffs_b = int(
                (
                    scales_b[:num_tokens].view(torch.uint8)
                    != q_cast_base.scales[:num_tokens].view(torch.uint8)
                ).sum()
            )
            cache_diffs_b = int((pages_b != pages_base).sum())

            w_diffs_b = 0
            slice_diffs_b = 0
            for r, sp in enumerate(layouts):
                first, last = sp.start, sp.start + sp.real_rows
                w_ref_r = w_base[first:last]
                w_diffs_b += int(
                    (local_w_b[r].view(torch.uint16) != w_ref_r.view(torch.uint16)).sum()
                )
                s_base = slice_mxfp8(q_cast_base, first, last)
                s_b = slice_mxfp8(q_cast_b, first, last)
                m = last - first
                if m > 0:
                    slice_diffs_b += int(
                        (
                            s_b.values[:m].view(torch.uint8)
                            != s_base.values[:m].view(torch.uint8)
                        ).sum()
                    )
                    slice_diffs_b += int(
                        (
                            s_b.scales[:m].view(torch.uint8)
                            != s_base.scales[:m].view(torch.uint8)
                        ).sum()
                    )

            assert (
                q_val_diffs_a == 0
                and q_scale_diffs_a == 0
                and cache_diffs_a == 0
                and q_val_diffs_b == 0
                and q_scale_diffs_b == 0
                and cache_diffs_b == 0
                and w_diffs_b == 0
                and slice_diffs_b == 0
            ), f"Bit mismatch at TP={world_size}, T={num_tokens}"

            results.append(
                {
                    "tp_size": world_size,
                    "num_tokens": num_tokens,
                    "padded_rows": padded,
                    "local_rows": local,
                    "last_rank_real_rows": layouts[-1].real_rows,
                    "option_a_diffs": {
                        "q_values": q_val_diffs_a,
                        "q_scales": q_scale_diffs_a,
                        "swa_cache": cache_diffs_a,
                    },
                    "option_b_diffs": {
                        "q_values": q_val_diffs_b,
                        "q_scales": q_scale_diffs_b,
                        "swa_cache": cache_diffs_b,
                        "index_weights": w_diffs_b,
                        "index_slice_mxfp8": slice_diffs_b,
                    },
                }
            )
    return results


def _time_cuda(fn, warmup: int = 40, iters: int = 200) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times_us = sorted(s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends))
    n = len(times_us)
    median = times_us[n // 2]
    q25 = times_us[n // 4]
    q75 = times_us[(3 * n) // 4]
    p90 = times_us[int(0.9 * n)]
    return {
        "median_us": round(median, 3),
        "iqr_us": round(q75 - q25, 3),
        "p90_us": round(p90, 3),
        "min_us": round(times_us[0], 3),
        "max_us": round(times_us[-1], 3),
    }


def benchmark_shapes(cos_sin: torch.Tensor, warmup: int, iters: int) -> list[dict]:
    rows_out = []
    q_weight = torch.ones(Q_LATENT, dtype=torch.bfloat16, device=DEVICE)
    kv_weight = torch.ones(KV_LATENT, dtype=torch.bfloat16, device=DEVICE)
    widths = (Q_LATENT, KV_LATENT, HEADS)

    for world_size in (3, 4):
        for num_tokens in (4096, 8192, 16384):
            sp = sp_prefill.SPRows(num_tokens, world_size, 0)
            local = sp.local_rows
            real = sp.real_rows
            padded = sp.padded_rows
            rows_q = mxfp8_rows(num_tokens)

            qr_kv_local = torch.randn(
                (local, Q_LATENT + KV_LATENT), dtype=torch.bfloat16, device=DEVICE
            )
            qr_local, kv_local = qr_kv_local.split((Q_LATENT, KV_LATENT), dim=-1)
            w_local = torch.randn((local, HEADS), dtype=torch.bfloat16, device=DEVICE)
            front_local = torch.empty(
                (local, sum(widths)), dtype=torch.bfloat16, device=DEVICE
            )
            gathered_front = torch.randn(
                (num_tokens, sum(widths)), dtype=torch.bfloat16, device=DEVICE
            )
            positions = torch.arange(num_tokens, dtype=torch.int64, device=DEVICE)
            local_pos = positions[:real]
            num_pages = (num_tokens // PAGE_SIZE) + 4
            slots = torch.arange(num_tokens, dtype=torch.int64, device=DEVICE)
            pages = torch.zeros(
                (num_pages, PAGE_SIZE * RECORD_BYTES["swa"]),
                dtype=torch.uint8,
                device=DEVICE,
            )

            # Baseline 1107 GPU kernel sequence
            def run_baseline():
                norm_forward(qr_local, q_weight, EPS, out=front_local[:, :Q_LATENT])
                norm_forward(
                    kv_local,
                    kv_weight,
                    EPS,
                    out=front_local[:, Q_LATENT : Q_LATENT + KV_LATENT],
                )
                columns.copy_columns(w_local, front_local[:, Q_LATENT + KV_LATENT :])
                qr_t, kv_t, _ = columns.split_columns(gathered_front, widths)
                _ = kernels.cast_mxfp8(qr_t)
                rope_(kv_t, positions, cos_sin, 1, False)
                write_cache(
                    kv_t, pages, slots, page_size=PAGE_SIZE, cache_kind="swa"
                )

            # Option A GPU kernel sequence
            qn_a = torch.empty((local, Q_LATENT), dtype=torch.bfloat16, device=DEVICE)
            kvn_a = torch.empty((local, KV_LATENT), dtype=torch.bfloat16, device=DEVICE)
            kv_gathered_a = torch.randn(
                (num_tokens, KV_LATENT), dtype=torch.bfloat16, device=DEVICE
            )

            def run_option_a():
                norm_forward(qr_local, q_weight, EPS, out=qn_a)
                _ = kernels.cast_mxfp8(qn_a)
                norm_forward(kv_local, kv_weight, EPS, out=kvn_a)
                rope_(kvn_a[:real], local_pos, cos_sin, 1, False)
                write_cache(
                    kv_gathered_a, pages, slots, page_size=PAGE_SIZE, cache_kind="swa"
                )

            # Option B (Promoted) GPU kernel sequence
            local_qr_b = torch.empty(
                (local, Q_LATENT), dtype=torch.bfloat16, device=DEVICE
            )
            local_q_cast_b = empty_mxfp8(local_qr_b)
            local_records_b = torch.empty(
                (local, RECORD_BYTES["swa"]), dtype=torch.uint8, device=DEVICE
            )
            gathered_records_b = torch.zeros(
                (num_tokens, RECORD_BYTES["swa"]), dtype=torch.uint8, device=DEVICE
            )

            def run_option_b():
                sp_attention_front(
                    qr_kv_local[:real],
                    q_weight,
                    kv_weight,
                    EPS,
                    local_pos,
                    cos_sin,
                    local_q_cast_b.values[:real],
                    local_q_cast_b.scales[:real],
                    local_records_b[:real],
                )
                write_swa_records(
                    gathered_records_b, pages, slots, page_size=PAGE_SIZE
                )

            def run_sp_front_only():
                sp_attention_front(
                    qr_kv_local[:real],
                    q_weight,
                    kv_weight,
                    EPS,
                    local_pos,
                    cos_sin,
                    local_q_cast_b.values[:real],
                    local_q_cast_b.scales[:real],
                    local_records_b[:real],
                )

            def run_write_records_only():
                write_swa_records(
                    gathered_records_b, pages, slots, page_size=PAGE_SIZE
                )

            t_base = _time_cuda(run_baseline, warmup=warmup, iters=iters)
            t_opt_a = _time_cuda(run_option_a, warmup=warmup, iters=iters)
            t_opt_b = _time_cuda(run_option_b, warmup=warmup, iters=iters)
            t_front = _time_cuda(run_sp_front_only, warmup=warmup, iters=iters)
            t_write = _time_cuda(run_write_records_only, warmup=warmup, iters=iters)

            base_row_bytes = sum(widths) * 2  # 3648 B/row
            opt_a_row_bytes = (Q_LATENT + 40) + (KV_LATENT * 2)  # 2344 B/row
            opt_b_row_bytes = (Q_LATENT + 40) + RECORD_BYTES["swa"]  # 1848 B/row

            rows_out.append(
                {
                    "tp_size": world_size,
                    "num_tokens": num_tokens,
                    "padded_rows": padded,
                    "local_rows": local,
                    "real_rows": real,
                    "baseline_us": t_base,
                    "option_a_us": t_opt_a,
                    "option_b_us": t_opt_b,
                    "sp_attention_front_only_us": t_front,
                    "write_swa_records_only_us": t_write,
                    "speedup_option_a_vs_baseline": round(
                        t_base["median_us"] / t_opt_a["median_us"], 3
                    ),
                    "speedup_option_b_vs_baseline": round(
                        t_base["median_us"] / t_opt_b["median_us"], 3
                    ),
                    "saved_us_per_layer_option_b": round(
                        t_base["median_us"] - t_opt_b["median_us"], 3
                    ),
                    "saved_ms_54_layers_option_b": round(
                        (t_base["median_us"] - t_opt_b["median_us"])
                        * NON_COMPRESSOR_LAYERS
                        / 1000.0,
                        3,
                    ),
                    "collective_bytes_per_row": {
                        "baseline": base_row_bytes,
                        "option_a": opt_a_row_bytes,
                        "option_b": opt_b_row_bytes,
                        "reduction_pct_option_b": round(
                            (1.0 - opt_b_row_bytes / base_row_bytes) * 100.0, 2
                        ),
                    },
                    "collective_mib_54_layers": {
                        "baseline": round(
                            padded
                            * base_row_bytes
                            * NON_COMPRESSOR_LAYERS
                            / (1024.0 * 1024.0),
                            3,
                        ),
                        "option_a": round(
                            padded
                            * opt_a_row_bytes
                            * NON_COMPRESSOR_LAYERS
                            / (1024.0 * 1024.0),
                            3,
                        ),
                        "option_b": round(
                            padded
                            * opt_b_row_bytes
                            * NON_COMPRESSOR_LAYERS
                            / (1024.0 * 1024.0),
                            3,
                        ),
                    },
                }
            )
    return rows_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    cos_sin = _make_cos_sin()
    exactness = verify_bit_exactness(cos_sin)
    benchmarks = benchmark_shapes(cos_sin, warmup=args.warmup, iters=args.iters)

    payload = {
        "device": torch.cuda.get_device_name(0),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "warmup": args.warmup,
        "iters": args.iters,
        "non_compressor_layers": NON_COMPRESSOR_LAYERS,
        "bit_exactness": exactness,
        "benchmarks": benchmarks,
    }
    print(json.dumps(payload, indent=2))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")


if __name__ == "__main__":
    main()
