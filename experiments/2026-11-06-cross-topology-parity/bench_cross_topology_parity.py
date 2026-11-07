#!/usr/bin/env python3
"""2-boot GPU verification and benchmark suite for cross-topology parity."""

from __future__ import annotations

import json
import sys
import time

from types import SimpleNamespace
import torch
from torch import nn

from vllm.models.deepseek_v4_1 import sp_prefill
from vllm.models.deepseek_v4_1.compressor import _compressor_part_cols
from vllm.models.deepseek_v4_1.tilelang import collectives, prepare
from vllm.models.deepseek_v4_1.tilelang import linear as tl_linear
from vllm.models.deepseek_v4_1.tilelang import wo as tl_wo
from vllm.models.deepseek_v4_1.tilelang.gemm import (
    DECODE_FP8_CONFIGS,
    DECODE_ROWS,
    SPLIT_BF16,
    WIDE_DECODE,
    fp8_decode_config,
    fp8_prefill_config,
    mxfp8_gemm,
    mxfp8_gemm_decode,
    pack_scale_words,
)
from vllm.models.deepseek_v4_1.tilelang.linear import (
    TileLangFP8LinearMethod,
    TileLangSplitLinearMethod,
    cast_mxfp8,
    empty_mxfp8,
    project_parts,
)
from vllm.v1.worker.workspace import use_preallocated_workspace


def bench_cuda_us(fn, warmup: int = 25, iters: int = 100) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return (start.elapsed_time(end) * 1000.0) / iters


def make_block32_weight(n: int, k: int, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    exps = torch.randint(
        118, 125, (n // 32, k // 32), generator=gen, device="cuda", dtype=torch.uint8
    )
    codes = torch.randint(0, 120, (n, k), generator=gen, device="cuda", dtype=torch.uint8)
    weight = codes.view(torch.float8_e4m3fn)
    return weight, exps, pack_scale_words(exps)


def build_split_bf16_layer(weight: torch.Tensor, ratio: int) -> tuple[TileLangSplitLinearMethod, nn.Module]:
    layer = nn.Module()
    layer.weight = nn.Parameter(weight.clone(), requires_grad=False)
    method = TileLangSplitLinearMethod(ratio)
    method.process_weights_after_loading(layer)
    method.declare_layer_kernels(layer)
    return method, layer


def verify_and_bench_compressor(scratch: torch.Tensor) -> dict:
    print("=== Task 1: compressor.fused_wkv_wgate Verification & Benchmarks ===", flush=True)
    device = torch.device("cuda")
    k = 5120
    rows_list = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256]
    max_rows = max(rows_list)
    gen = torch.Generator(device=device).manual_seed(20261106)
    x_full = torch.randn((max_rows, k), generator=gen, device=device, dtype=torch.bfloat16)

    exactness = {}
    with use_preallocated_workspace(scratch):
        for ratio in (1, 2):
            wkv_full = (torch.randn((512, k), generator=gen, device=device) * 0.02).bfloat16()
            wgate_full = (torch.randn((512, k), generator=gen, device=device) * 0.02).bfloat16()
            full_weight = wkv_full if ratio == 1 else torch.cat([wkv_full, wgate_full], dim=0)

            _, ref_layer = build_split_bf16_layer(full_weight, ratio)
            ref_outputs = {}
            for r in rows_list:
                out_wkv = torch.empty((r, 512), dtype=torch.bfloat16, device=device)
                out_wgate = (
                    torch.empty((r, 512), dtype=torch.bfloat16, device=device)
                    if ratio == 2
                    else None
                )
                outputs = (out_wkv,) if ratio == 1 else (out_wkv, out_wgate)
                project_parts(ref_layer, x_full[:r], outputs, r)
                ref_outputs[r] = (out_wkv, out_wgate)

            ratio_results = {}
            for tp_size in (1, 2, 3, 4, 5, 6, 7, 8):
                part_cols = _compressor_part_cols(tp_size)
                rank_layers = []
                raw_shards = []
                for tp_rank in range(tp_size):
                    lo = tp_rank * part_cols
                    hi = min(512, lo + part_cols)
                    valid = max(0, hi - lo)
                    parts = []
                    for src in ([wkv_full] if ratio == 1 else [wkv_full, wgate_full]):
                        shard = torch.zeros((part_cols, k), dtype=torch.bfloat16, device=device)
                        if valid > 0:
                            shard[:valid].copy_(src[lo:hi])
                        parts.append(shard)
                    w_rank = torch.cat(parts, dim=0)
                    raw_shards.append(w_rank.clone())
                    _, layer = build_split_bf16_layer(w_rank.clone(), ratio)
                    if hasattr(tl_linear, "unpack_bf16_layer"):
                        assert torch.equal(tl_linear.unpack_bf16_layer(layer), w_rank)
                    rank_layers.append(layer)

                # Verify _sp_weight unpadding matches full weight bit-for-bit
                if ratio == 1:
                    sp_w = torch.cat(raw_shards, dim=0)[:512]
                    sp_weight_ok = bool(torch.equal(sp_w, wkv_full))
                else:
                    wkv_s = [s[:part_cols] for s in raw_shards]
                    wgate_s = [s[part_cols:] for s in raw_shards]
                    sp_wkv = torch.cat(wkv_s, dim=0)[:512]
                    sp_wgate = torch.cat(wgate_s, dim=0)[:512]
                    sp_weight_ok = bool(
                        torch.equal(sp_wkv, wkv_full) and torch.equal(sp_wgate, wgate_full)
                    )

                all_rows_ok = True
                for r in rows_list:
                    wkv_parts = []
                    wgate_parts = []
                    for rk in range(tp_size):
                        o_kv = torch.empty((r, part_cols), dtype=torch.bfloat16, device=device)
                        if ratio == 1:
                            project_parts(rank_layers[rk], x_full[:r], (o_kv,), r)
                            wkv_parts.append(o_kv)
                        else:
                            o_gate = torch.empty(
                                (r, part_cols), dtype=torch.bfloat16, device=device
                            )
                            project_parts(rank_layers[rk], x_full[:r], (o_kv, o_gate), r)
                            wkv_parts.append(o_kv)
                            wgate_parts.append(o_gate)

                    wkv_got = torch.cat(wkv_parts, dim=-1)[:, :512]
                    wgate_got = (
                        torch.cat(wgate_parts, dim=-1)[:, :512] if ratio == 2 else None
                    )
                    wkv_ref, wgate_ref = ref_outputs[r]
                    row_ok = bool(torch.equal(wkv_got, wkv_ref))
                    if ratio == 2:
                        row_ok = row_ok and bool(torch.equal(wgate_got, wgate_ref))
                    if not row_ok:
                        all_rows_ok = False
                        max_diff = (wkv_got.float() - wkv_ref.float()).abs().max().item()
                        print(
                            f"  MISMATCH ratio={ratio} tp={tp_size} rows={r} max_diff={max_diff}",
                            flush=True,
                        )

                ratio_results[f"tp{tp_size}"] = {
                    "part_cols": part_cols,
                    "local_n": part_cols * ratio,
                    "sp_weight_bit_exact": sp_weight_ok,
                    "project_bit_exact_all_rows": all_rows_ok,
                }
                print(
                    f"  ratio={ratio} tp={tp_size} local_n={part_cols * ratio}: "
                    f"sp_weight_exact={sp_weight_ok} project_exact={all_rows_ok}",
                    flush=True,
                )
            exactness[f"ratio_{ratio}"] = ratio_results

        # Benchmark TP3 sharded vs Replicated (TP1 fallback) and TP4 sharded (warm & cold/multi-layer)
        bench_results = {}
        n_pool = 16  # 16 layers * 10MB = 160MB > L2 cache to measure realistic multi-layer streaming
        for ratio in (1, 2):
            ratio_bench = {}
            shapes = {
                "replicated_tp1_fallback": 512,
                "tp3_sharded": 192,
                "tp4_sharded": 128,
            }
            layers_by_name = {}
            for name, part_c in shapes.items():
                n_cols = part_c * ratio
                w = (torch.randn((n_cols, k), generator=gen, device=device) * 0.02).bfloat16()
                _, layer = build_split_bf16_layer(w, ratio)
                # Build pool of weight clones sharing compiled kernels for multi-layer streaming bench
                pool_layers = []
                for _ in range(n_pool):
                    pl = nn.Module()
                    pl.weight = nn.Parameter(
                        layer.weight.data.clone(), requires_grad=False
                    )
                    for attr in (
                        "tilelang_parts",
                        "tilelang_shards",
                        "tilelang_gemm",
                        "tilelang_splitk",
                        "tilelang_counters",
                        "tilelang_unpack",
                        "packed_bf16_in_features",
                        "packed_bf16_exponent_base",
                        "packed_bf16_exceptions",
                        "weight_exception_entries",
                        "weight_row_offsets",
                    ):
                        if hasattr(layer, attr):
                            setattr(pl, attr, getattr(layer, attr))
                    pool_layers.append(pl)
                o1 = torch.empty((max_rows, part_c), dtype=torch.bfloat16, device=device)
                o2 = (
                    torch.empty((max_rows, part_c), dtype=torch.bfloat16, device=device)
                    if ratio == 2
                    else None
                )
                outs = (o1,) if ratio == 1 else (o1, o2)
                layers_by_name[name] = (layer, pool_layers, outs, n_cols)

            for r in rows_list:
                xr = x_full[:r]
                row_entry = {}
                for name, (layer, pool_layers, outs, n_cols) in layers_by_name.items():
                    us_warm = bench_cuda_us(
                        lambda l=layer, x=xr, o=outs, rows=r: project_parts(l, x, o, rows)
                    )
                    step_idx = [0]

                    def run_cold(pl=pool_layers, x=xr, o=outs, rows=r, idx=step_idx):
                        i = idx[0]
                        idx[0] = (i + 1) % n_pool
                        project_parts(pl[i], x, o, rows)

                    us_cold = bench_cuda_us(run_cold, warmup=16, iters=64)
                    row_entry[name] = {
                        "n_cols": n_cols,
                        "warm_latency_us": round(us_warm, 3),
                        "cold_latency_us": round(us_cold, 3),
                        "cold_61_layers_ms": round(us_cold * 61.0 / 1000.0, 3),
                    }
                rep_warm = row_entry["replicated_tp1_fallback"]["warm_latency_us"]
                tp3_warm = row_entry["tp3_sharded"]["warm_latency_us"]
                rep_cold = row_entry["replicated_tp1_fallback"]["cold_latency_us"]
                tp3_cold = row_entry["tp3_sharded"]["cold_latency_us"]
                row_entry["tp3_warm_speedup"] = round(rep_warm / tp3_warm, 3)
                row_entry["tp3_cold_speedup"] = round(rep_cold / tp3_cold, 3)
                row_entry["tp3_cold_saved_61_layers_ms"] = round(
                    (rep_cold - tp3_cold) * 61.0 / 1000.0, 3
                )
                ratio_bench[str(r)] = row_entry
                if r in (1, 4, 16, 32, 64, 128, 256):
                    print(
                        f"  bench ratio={ratio} M={r:3d}: "
                        f"cold rep({512*ratio})={rep_cold:6.2f}us -> tp3({192*ratio})={tp3_cold:6.2f}us "
                        f"({rep_cold/tp3_cold:.2f}x, 61L saved={(rep_cold-tp3_cold)*61/1000:.2f}ms) | "
                        f"warm {rep_warm:5.2f}us -> {tp3_warm:5.2f}us ({rep_warm/tp3_warm:.2f}x)",
                        flush=True,
                    )
            bench_results[f"ratio_{ratio}"] = ratio_bench

    return {"exactness": exactness, "benchmarks": bench_results}


def verify_and_bench_wqa_wkv_tp5_tp8(scratch: torch.Tensor) -> dict:
    print("=== Task 2: fused_wqa_wkv TP5..TP8 Decode Configs Benchmarks ===", flush=True)
    device = torch.device("cuda")
    k = 5120
    rows_list = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128]
    max_rows = max(rows_list)
    gen = torch.Generator(device=device).manual_seed(20261107)
    x_bf16 = torch.randn((max_rows, k), generator=gen, device=device, dtype=torch.bfloat16)
    mxfp8_in = cast_mxfp8(x_bf16)
    xq = mxfp8_in.values
    words = mxfp8_in.scales.contiguous().view(torch.uint32)

    w_full, exps_full, sf_full = make_block32_weight(1792, k, 2901)

    tp_shapes = [(2, 896), (3, 608), (4, 448), (5, 384), (6, 320), (7, 256), (8, 224)]
    sharded_exactness = {}
    with use_preallocated_workspace(scratch):
        # Reference TP1 layer
        ref_layer = nn.Module()
        ref_layer.weight = nn.Parameter(w_full.clone(), requires_grad=False)
        ref_layer.weight_scale_inv = nn.Parameter(exps_full.clone(), requires_grad=False)
        ref_method = TileLangFP8LinearMethod(SimpleNamespace(weight_block_size=[32, 32]))
        ref_method.process_weights_after_loading(ref_layer)
        ref_method.declare_layer_kernels(ref_layer)
        ref_outs = {r: ref_method.apply(ref_layer, x_bf16[:r]) for r in rows_list}

        for tp_size, part_n in tp_shapes:
            total_n = tp_size * part_n
            if total_n > 1792:
                pad = total_n - 1792
                w_pad = torch.cat(
                    [w_full, torch.zeros((pad, k), dtype=w_full.dtype, device=device)],
                    dim=0,
                )
                s_pad = torch.cat(
                    [
                        exps_full,
                        torch.full(
                            (pad // 32, exps_full.shape[1]),
                            0x7F,
                            dtype=exps_full.dtype,
                            device=device,
                        ),
                    ],
                    dim=0,
                )
            else:
                w_pad, s_pad = w_full, exps_full

            methods_layers = []
            for rk in range(tp_size):
                layer = nn.Module()
                layer.weight = nn.Parameter(
                    w_pad[rk * part_n : (rk + 1) * part_n].contiguous(),
                    requires_grad=False,
                )
                layer.weight_scale_inv = nn.Parameter(
                    s_pad[rk * (part_n // 32) : (rk + 1) * (part_n // 32)].contiguous(),
                    requires_grad=False,
                )
                m = TileLangFP8LinearMethod(SimpleNamespace(weight_block_size=[32, 32]))
                m.process_weights_after_loading(layer)
                m.declare_layer_kernels(layer)
                methods_layers.append((m, layer))

            all_exact = True
            for r in rows_list:
                parts = [m.apply(l, x_bf16[:r]) for m, l in methods_layers]
                gathered = torch.cat(parts, dim=-1)[:, :1792]
                if not torch.equal(gathered, ref_outs[r]):
                    all_exact = False
                    print(f"  MISMATCH tp={tp_size} rows={r}", flush=True)
            sharded_exactness[f"tp{tp_size}_{part_n}"] = all_exact
            print(
                f"  tp={tp_size} part_n={part_n} bit_exact_vs_tp1_all_rows={all_exact}",
                flush=True,
            )

    # Benchmark TP5..TP8 shapes across M in rows_list
    bench_shapes = [(5, 384), (6, 320), (7, 256), (8, 224)]
    shape_benchmarks = {}
    for tp_size, n in bench_shapes:
        w, exps, w_sf = make_block32_weight(n, k, 3000 + n)
        out = torch.empty((max_rows, n), dtype=torch.bfloat16, device=device)

        tuned_kernels = {
            bm: mxfp8_gemm_decode(
                n, k, **fp8_decode_config(n, k, bm), padded_rows=True
            )
            for bm in (16, 32, 64)
        }
        tuned_bm64_s3 = mxfp8_gemm_decode(
            n,
            k,
            block_M=64,
            block_N=32,
            block_K=256,
            num_stages=3,
            padded_rows=True,
        )
        generic_kernels = {
            bm: mxfp8_gemm_decode(
                n,
                k,
                block_M=bm,
                block_N=64,
                block_K=128,
                num_stages=3,
                padded_rows=True,
            )
            for bm in (16, 32, 64)
        }
        prefill_k = mxfp8_gemm(n, k, **fp8_prefill_config(n, k))
        for _ in range(10):
            tuned_kernels[16](xq, w, words, w_sf, out[:16])
            generic_kernels[16](xq[:DECODE_ROWS], w, words[:DECODE_ROWS], w_sf, out[:16])
        torch.cuda.synchronize()

        rows_bench = {}
        for r in rows_list:
            out_r = out[:r]

            if r <= 16:
                bm_tuned = 16
            elif r <= 32:
                bm_tuned = 32
            elif r <= 64:
                bm_tuned = 64
            elif r <= 96:
                bm_tuned = WIDE_DECODE[(n, k)][96]
            else:
                bm_tuned = WIDE_DECODE[(n, k)][128]
            k_tuned = tuned_kernels[bm_tuned]

            us_tuned = bench_cuda_us(
                lambda kt=k_tuned, o=out_r: kt(xq, w, words, w_sf, o)
            )

            if r <= 16:
                k_gen = generic_kernels[16]
                us_gen = bench_cuda_us(
                    lambda kg=k_gen, o=out_r: kg(
                        xq[:DECODE_ROWS], w, words[:DECODE_ROWS], w_sf, o
                    )
                )
            elif r <= 32:
                k_gen = generic_kernels[32]
                us_gen = bench_cuda_us(
                    lambda kg=k_gen, o=out_r: kg(
                        xq[:DECODE_ROWS], w, words[:DECODE_ROWS], w_sf, o
                    )
                )
            elif r <= 64:
                k_gen = generic_kernels[64]
                us_gen = bench_cuda_us(
                    lambda kg=k_gen, o=out_r: kg(
                        xq[:DECODE_ROWS], w, words[:DECODE_ROWS], w_sf, o
                    )
                )
            else:
                us_gen = bench_cuda_us(
                    lambda kp=prefill_k, x=xq[:r], wd=words[:r], o=out_r: kp(
                        x, w, wd, w_sf, o
                    )
                )

            entry = {
                "tuned_block_M": bm_tuned,
                "tuned_us": round(us_tuned, 3),
                "generic_us": round(us_gen, 3),
                "speedup": round(us_gen / us_tuned, 3),
                "saved_61_layers_ms": round((us_gen - us_tuned) * 61.0 / 1000.0, 3),
            }
            if bm_tuned == 64:
                us_s3 = bench_cuda_us(
                    lambda ks=tuned_bm64_s3, o=out_r: ks(xq, w, words, w_sf, o)
                )
                entry["tuned_bm64_stages3_us"] = round(us_s3, 3)

            rows_bench[str(r)] = entry
            print(
                f"  TP{tp_size} ({n}, 5120) M={r:3d}: generic={us_gen:6.2f}us "
                f"tuned(bm={bm_tuned})={us_tuned:6.2f}us speedup={us_gen/us_tuned:.2f}x",
                flush=True,
            )
        shape_benchmarks[f"tp{tp_size}_{n}x5120"] = rows_bench

    return {"exactness": sharded_exactness, "benchmarks": shape_benchmarks}


def verify_and_bench_trim_padded_groups(scratch: torch.Tensor) -> dict:
    print("=== Task 3: _trim_padded_groups (real_groups == 0 & < groups) ===", flush=True)
    device = torch.device("cuda")
    rank_width = 1024
    head_dim = 512
    hidden = 5120
    gen = torch.Generator(device=device).manual_seed(20261108)

    with use_preallocated_workspace(scratch):
        topologies = {
            "tp3": {"real": (3, 3, 2), "padded": (3, 3, 3)},
            "tp5": {"real": (2, 2, 2, 2, 0), "padded": (2, 2, 2, 2, 2)},
            "tp6": {"real": (2, 2, 2, 2, 0, 0), "padded": (2, 2, 2, 2, 2, 2)},
            "tp7": {"real": (2, 2, 2, 2, 0, 0, 0), "padded": (2, 2, 2, 2, 2, 2, 2)},
        }
        test_rows = [1, 4, 16, 64, 128, 256, 2048]
        max_rows = max(test_rows)

        full_cols = 8 * rank_width
        w_a_full, exps_a_full, _ = make_block32_weight(full_cols, 8 * head_dim, 4100)
        w_b_full, exps_b_full, _ = make_block32_weight(hidden, full_cols, 4200)
        o_full = (
            torch.randn((max_rows, 64, head_dim), generator=gen, device=device) * 0.25
        ).bfloat16()

        exactness = {}
        for topo_name, cfg in topologies.items():
            real_groups = cfg["real"]
            padded_groups = cfg["padded"]
            world_size = len(real_groups)

            def build_projections(groups_tuple, real_tuple):
                projs = []
                g_offset = 0
                for rk, (g_alloc, g_real) in enumerate(zip(groups_tuple, real_tuple)):
                    cols = g_alloc * rank_width
                    wo_a = nn.Module()
                    wo_a.o_lora_rank = rank_width
                    w_a = torch.zeros(
                        (cols, 8 * head_dim), dtype=torch.float8_e4m3fn, device=device
                    )
                    s_a = torch.full(
                        (cols // 32, (8 * head_dim) // 32),
                        0x7F,
                        dtype=torch.uint8,
                        device=device,
                    )
                    if g_real > 0:
                        real_c = g_real * rank_width
                        w_a[:real_c].copy_(
                            w_a_full[g_offset * rank_width : (g_offset + g_real) * rank_width]
                        )
                        s_a[: real_c // 32].copy_(
                            exps_a_full[
                                (g_offset * rank_width) // 32 : ((g_offset + g_real) * rank_width)
                                // 32
                            ]
                        )
                    wo_a.weight = nn.Parameter(w_a, requires_grad=False)
                    wo_a.weight_scale_inv = nn.Parameter(s_a, requires_grad=False)

                    wo_b = nn.Module()
                    w_b = torch.zeros(
                        (hidden, cols), dtype=torch.float8_e4m3fn, device=device
                    )
                    s_b = torch.full(
                        (hidden // 32, cols // 32), 0x7F, dtype=torch.uint8, device=device
                    )
                    if g_real > 0:
                        real_c = g_real * rank_width
                        w_b[:, :real_c].copy_(
                            w_b_full[
                                :, g_offset * rank_width : (g_offset + g_real) * rank_width
                            ]
                        )
                        s_b[:, : real_c // 32].copy_(
                            exps_b_full[
                                :,
                                (g_offset * rank_width)
                                // 32 : ((g_offset + g_real) * rank_width) // 32,
                            ]
                        )
                    wo_b.weight = nn.Parameter(w_b, requires_grad=False)
                    wo_b.weight_scale_inv = nn.Parameter(s_b, requires_grad=False)

                    proj = tl_wo.TileLangWOProjection(wo_a, wo_b, groups=g_alloc)
                    proj.declare_kernels()
                    proj.enable_sp(groups_tuple)
                    sp_calls = proj._sp_prefill_calls()
                    proj._sp_prefill_by_cols = {
                        c: prepare.launch(call) for c, call in sp_calls.items()
                    }
                    projs.append((proj, g_offset, g_real, g_alloc))
                    g_offset += g_real
                for rk, (proj, _, _, _) in enumerate(projs):
                    proj.sp_shards = tuple(
                        proj.b
                        if peer == rk
                        else tl_wo._PeerWeight(
                            projs[peer][0].b.weight,
                            projs[peer][0].b.tilelang_weight_sf,
                            proj._sp_prefill_by_cols.get(projs[peer][0].a.columns),
                        )
                        for peer in range(world_size)
                    )
                return projs

            trimmed_projs = build_projections(real_groups, real_groups)
            untrimmed_projs = build_projections(padded_groups, real_groups)

            decode_prefill_ok = True
            sp_prefill_ok = True
            for r in test_rows:
                trimmed_outs = []
                untrimmed_outs = []
                for rk in range(world_size):
                    proj_t, g_off, g_real, _ = trimmed_projs[rk]
                    proj_u, _, _, g_pad = untrimmed_projs[rk]

                    o_t = o_full[:r, g_off * 8 : (g_off + g_real) * 8].contiguous()
                    out_t = torch.full((r, hidden), 7.0, dtype=torch.bfloat16, device=device)
                    proj_t(o_t, out_t)
                    trimmed_outs.append(out_t)

                    o_u = torch.zeros(
                        (r, g_pad * 8, head_dim), dtype=torch.bfloat16, device=device
                    )
                    if g_real > 0:
                        o_u[:, : g_real * 8].copy_(o_t)
                    out_u = torch.full((r, hidden), 7.0, dtype=torch.bfloat16, device=device)
                    proj_u(o_u, out_u)
                    untrimmed_outs.append(out_u)

                for rk in range(world_size):
                    if not torch.equal(trimmed_outs[rk], untrimmed_outs[rk]):
                        decode_prefill_ok = False
                        print(
                            f"  MISMATCH {topo_name} rank={rk} rows={r}",
                            flush=True,
                        )
                sum_t = collectives.rank_order_sum(trimmed_outs)
                sum_u = collectives.rank_order_sum(untrimmed_outs)
                if not torch.equal(sum_t, sum_u):
                    decode_prefill_ok = False

                if r >= 128:
                    padded_r = sp_prefill.SPRows(r, world_size, 0).padded_rows
                    local_r = padded_r // world_size
                    for rk in range(world_size):
                        sp = sp_prefill.SPRows(r, world_size, rk, unreduced=True)
                        real_r = sp.real_rows
                        proj_t = trimmed_projs[rk][0]
                        parts = torch.empty(
                            (world_size, local_r, hidden),
                            dtype=torch.bfloat16,
                            device=device,
                        )
                        for peer in range(world_size):
                            peer_proj, g_off_p, g_real_p, _ = trimmed_projs[peer]
                            cols_p = peer_proj.a.columns
                            xq_p = torch.zeros(
                                (padded_r, cols_p),
                                dtype=torch.float8_e4m3fn,
                                device=device,
                            )
                            sf_p = torch.zeros(
                                (padded_r, cols_p // 32),
                                dtype=torch.uint8,
                                device=device,
                            )
                            if cols_p > 0:
                                o_p = o_full[:r, g_off_p * 8 : (g_off_p + g_real_p) * 8].contiguous()
                                p_specs = peer_proj.a.partial_specs(r)
                                partials = (
                                    tl_linear._scratch(p_specs, scratch)[0]
                                    if p_specs
                                    else None
                                )
                                peer_proj.a(
                                    o_p.reshape(r, -1), xq_p[:r], sf_p[:r], partials
                                )
                            xq_local = xq_p.view(world_size, local_r, cols_p)[rk]
                            sf_local = sf_p.view(world_size, local_r, cols_p // 32)[rk]
                            proj_t._project_sp_shard(
                                peer, xq_local, sf_local, parts[peer], real_r, local_r
                            )
                        sum_sp = collectives.rank_order_sum(list(parts.unbind(0)))[:real_r]
                        full_padded_outs = []
                        for peer in range(world_size):
                            p_proj, g_off_p, g_real_p, _ = trimmed_projs[peer]
                            f_out = torch.zeros(
                                (padded_r, hidden), dtype=torch.bfloat16, device=device
                            )
                            o_p = o_full[:r, g_off_p * 8 : (g_off_p + g_real_p) * 8].contiguous()
                            p_proj(o_p, f_out[:r])
                            full_padded_outs.append(
                                f_out.view(world_size, local_r, hidden)[rk]
                            )
                        expected_sp = collectives.rank_order_sum(full_padded_outs)[:real_r]
                        if not torch.equal(sum_sp, expected_sp):
                            sp_prefill_ok = False
                            print(
                                f"  SP MISMATCH {topo_name} rank={rk} rows={r}",
                                flush=True,
                            )

            exactness[topo_name] = {
                "real_groups": list(real_groups),
                "padded_groups": list(padded_groups),
                "decode_prefill_bit_exact": decode_prefill_ok,
                "sp_prefill_bit_exact": sp_prefill_ok,
            }
            print(
                f"  {topo_name} real={real_groups} vs padded={padded_groups}: "
                f"decode_prefill_exact={decode_prefill_ok} sp_prefill_exact={sp_prefill_ok}",
                flush=True,
            )

        # Benchmark wholly padded rank (real_groups == 0 vs untrimmed 2 groups: wq_b + wo_a + wo_b)
        wq_b_w, wq_b_exps, _ = make_block32_weight(8192, 1280, 4300)
        wq_b_layer = nn.Module()
        wq_b_layer.weight = nn.Parameter(wq_b_w, requires_grad=False)
        wq_b_layer.weight_scale_inv = nn.Parameter(wq_b_exps, requires_grad=False)
        wq_b_method = TileLangFP8LinearMethod(SimpleNamespace(weight_block_size=[32, 32]))
        wq_b_method.process_weights_after_loading(wq_b_layer)
        wq_b_method.declare_layer_kernels(wq_b_layer)

        qr_full = torch.randn(
            (max_rows, 1280), generator=gen, device=device, dtype=torch.bfloat16
        )
        o_dummy = torch.zeros((max_rows, 16, head_dim), dtype=torch.bfloat16, device=device)
        o_empty = torch.empty((max_rows, 0, head_dim), dtype=torch.bfloat16, device=device)
        out_buf = torch.empty((max_rows, hidden), dtype=torch.bfloat16, device=device)

        proj_untrimmed_2g = untrimmed_projs[4][0]  # TP7 rank 4 untrimmed (2 groups)
        proj_trimmed_0g = trimmed_projs[4][0]      # TP7 rank 4 trimmed (0 groups)

        bench_rows = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 2048]
        wholly_padded_bench = {}
        for r in bench_rows:
            qr_r = qr_full[:r]
            o_d_r = o_dummy[:r]
            o_e_r = o_empty[:r]
            out_r = out_buf[:r]

            us_untrimmed_wo = bench_cuda_us(
                lambda p=proj_untrimmed_2g, o=o_d_r, out=out_r: p(o, out)
            )
            us_untrimmed_wqb_plus_wo = bench_cuda_us(
                lambda m=wq_b_method, l=wq_b_layer, q=qr_r, p=proj_untrimmed_2g, o=o_d_r, out=out_r: (
                    m.apply(l, q),
                    p(o, out),
                )
            )
            us_trimmed_0g = bench_cuda_us(
                lambda p=proj_trimmed_0g, o=o_e_r, out=out_r: p(o, out)
            )
            wholly_padded_bench[str(r)] = {
                "untrimmed_wo_2groups_us": round(us_untrimmed_wo, 3),
                "untrimmed_wqb_plus_wo_2groups_us": round(us_untrimmed_wqb_plus_wo, 3),
                "trimmed_0groups_us": round(us_trimmed_0g, 3),
                "wo_speedup": round(us_untrimmed_wo / us_trimmed_0g, 2),
                "wqb_plus_wo_speedup": round(us_untrimmed_wqb_plus_wo / us_trimmed_0g, 2),
                "saved_61_layers_ms": round(
                    (us_untrimmed_wqb_plus_wo - us_trimmed_0g) * 61.0 / 1000.0, 3
                ),
            }
            if r in (1, 4, 16, 32, 64, 128, 256, 2048):
                print(
                    f"  wholly-padded rank M={r:4d}: untrimmed(wq_b+wo)={us_untrimmed_wqb_plus_wo:7.2f}us "
                    f"(wo={us_untrimmed_wo:6.2f}us) -> trimmed(0g)={us_trimmed_0g:5.2f}us "
                    f"(saved 61L={(us_untrimmed_wqb_plus_wo - us_trimmed_0g)*61/1000:.2f}ms)",
                    flush=True,
                )

        return {"exactness": exactness, "benchmarks": wholly_padded_bench}


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "boot_results.json"
    t0 = time.time()
    tl_linear._max_rows = lambda: 2048
    scratch = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "task1_compressor": verify_and_bench_compressor(scratch),
        "task2_wqa_wkv_tp5_tp8": verify_and_bench_wqa_wkv_tp5_tp8(scratch),
        "task3_trim_padded_groups": verify_and_bench_trim_padded_groups(scratch),
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_path} in {results['elapsed_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
