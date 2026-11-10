#!/usr/bin/env python3
"""Benchmark and bit-exactness verification for Q-B and indexer Q-B tile-contiguous decode."""

from __future__ import annotations

import argparse
import json
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.models.deepseek_v4_1 import l2_prefetch as ds_l2pf
from vllm.models.deepseek_v4_1.tilelang import gemm as ds_gemm
from vllm.models.deepseek_v4_1.tilelang.gemm import (
    DECODE_ROWS,
    fp8_decode_config,
    fp8_prefill_config,
    mxfp8_gemm,
    mxfp8_gemm_decode,
    mxfp8_gemm_rope_mxfp4_decode,
    pack_scale_words,
    pack_weight_tiles,
)
from vllm.models.deepseek_v4_1.tilelang.indexer import quantize_mxfp4
from vllm.models.deepseek_v4_1.tilelang.linear import cast_mxfp8
from tests.kernels.deepseek_v41_pdl_chains import (
    ROWS as CHAIN_ROWS,
    LayerChain,
)

OLD_1107_CONFIGS: dict[tuple[int, int], dict[int, dict[str, int]]] = {
    (16384, 1280): {
        16: {"block_M": 16, "block_N": 64, "block_K": 128, "num_stages": 3, "threads": 128},
        32: {"block_M": 32, "block_N": 64, "block_K": 128, "num_stages": 3, "threads": 128},
        64: {"block_M": 64, "block_N": 64, "block_K": 128, "num_stages": 3, "threads": 128},
    },
    (12288, 1280): {
        16: {"block_M": 16, "block_N": 128, "block_K": 128, "num_stages": 3, "threads": 128},
        32: {"block_M": 32, "block_N": 64, "block_K": 128, "num_stages": 3, "threads": 128},
        64: {"block_M": 64, "block_N": 64, "block_K": 128, "num_stages": 2, "threads": 128},
    },
    (8192, 1280): {
        16: {"block_M": 16, "block_N": 128, "block_K": 128, "num_stages": 3, "threads": 128},
        32: {"block_M": 32, "block_N": 64, "block_K": 128, "num_stages": 3, "threads": 128},
        64: {"block_M": 64, "block_N": 64, "block_K": 128, "num_stages": 3, "threads": 128},
    },
    (6144, 1280): {
        16: {"block_M": 16, "block_N": 64, "block_K": 128, "num_stages": 3, "threads": 128},
        32: {"block_M": 32, "block_N": 64, "block_K": 128, "num_stages": 3, "threads": 128},
        64: {"block_M": 64, "block_N": 64, "block_K": 128, "num_stages": 3, "threads": 128},
    },
    (4096, 1280): {
        16: {"block_M": 16, "block_N": 64, "block_K": 128, "num_stages": 3, "threads": 128},
        32: {"block_M": 32, "block_N": 128, "block_K": 128, "num_stages": 3, "threads": 128},
        64: {"block_M": 64, "block_N": 64, "block_K": 128, "num_stages": 2, "threads": 128},
    },
    (2048, 1280): {
        16: {"block_M": 16, "block_N": 64, "block_K": 128, "num_stages": 3, "threads": 128},
        32: {"block_M": 32, "block_N": 64, "block_K": 128, "num_stages": 3, "threads": 128},
        64: {"block_M": 64, "block_N": 64, "block_K": 128, "num_stages": 3, "threads": 128},
    },
}

BATCH_SIZES = (1, 2, 3, 6, 12, 24, 48)
QB_SHAPES = (
    (16384, 1280),
    (12288, 1280),
    (8192, 1280),
    (6144, 1280),
    (4096, 1280),
    (2048, 1280),
)


def _block_m_for(m: int) -> int:
    return 16 if m <= 16 else (32 if m <= 32 else 64)


def _trimmed_mean(vals: list[float], trim_frac: float = 0.25) -> float:
    s = sorted(vals)
    lo = int(len(s) * trim_frac)
    hi = len(s) - lo
    sub = s[lo:hi] if hi > lo else s
    return round(sum(sub) / len(sub), 2)


def _measure_warm(fn, warmup: int = 15, outer: int = 30, inner: int = 20) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(outer)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(outer)]
    for i in range(outer):
        starts[i].record()
        for _ in range(inner):
            fn()
        ends[i].record()
    torch.cuda.synchronize()
    vals = [s.elapsed_time(e) * 1000.0 / inner for s, e in zip(starts, ends)]
    return _trimmed_mean(vals)


def _measure_single(fn, prep_fn, warmup: int = 20, iters: int = 100) -> float:
    for _ in range(warmup):
        prep_fn()
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        prep_fn()
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    vals = [s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends)]
    return _trimmed_mean(vals)


def _make_prefetch_plan(tensors: list[torch.Tensor], device: torch.device):
    segs = []
    for idx, t in enumerate(tensors):
        seg = ds_l2pf._l2pf.tensor_segment(f"t{idx}", t, min_bytes=0)
        if seg is not None:
            segs.append(seg)
    return ds_l2pf.make_plan(segs, ds_l2pf.BUDGET_NEXT, device)


def _capture_chain(chain: LayerChain) -> torch.cuda.CUDAGraph:
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            chain.run()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        chain.run()
    return g


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    torch.manual_seed(20261108)
    device = torch.device("cuda")
    ds_l2pf.warmup()
    flush_buf = torch.empty((48 * 1024 * 1024,), dtype=torch.uint8, device=device)

    results: dict = {
        "device": torch.cuda.get_device_name(0),
        "bit_exactness": {},
        "qb_isolated": {},
        "indexer_qb_isolated": {},
        "chained_qb_plus_indexer_qb": {},
        "layer_chain_pdl": {},
    }

    x_bf16_all = torch.randn((DECODE_ROWS, 1280), device=device, dtype=torch.bfloat16)
    mxfp8_all = cast_mxfp8(x_bf16_all)
    a_fp8 = mxfp8_all.values
    s_a_w = mxfp8_all.scales.contiguous().view(torch.uint32)

    # 1. Verify bit-exactness and benchmark isolated Q-B across all shapes
    for n, k in QB_SHAPES:
        key = f"{n}x{k}"
        w_rm = (torch.randn((n, k), device=device, dtype=torch.bfloat16) * 0.1).to(
            torch.float8_e4m3fn
        )
        w_tc = pack_weight_tiles(w_rm)
        s_b = torch.randint(120, 126, (n // 32, k // 32), device=device, dtype=torch.uint8)
        s_b_w = pack_scale_words(s_b)
        plan_rm = _make_prefetch_plan([w_rm, s_b_w], device)
        plan_tc = _make_prefetch_plan([w_tc, s_b_w], device)

        k_pre = mxfp8_gemm(n, k, **fp8_prefill_config(n, k))
        y_pre_full = torch.empty((DECODE_ROWS, n), device=device, dtype=torch.bfloat16)
        k_pre(a_fp8, w_rm, s_a_w, s_b_w, y_pre_full)
        torch.cuda.synchronize()

        shape_exact: dict[str, dict] = {}
        shape_perf: dict[str, dict] = {}

        for m in BATCH_SIZES:
            block_m = _block_m_for(m)
            cfg_aligned = fp8_decode_config(n, k, block_m)
            cfg_1107 = OLD_1107_CONFIGS[(n, k)][block_m]
            old_was_tc = (n, k) in ((12288, 1280), (8192, 1280), (4096, 1280))

            k_rm = mxfp8_gemm_decode(
                n, k, **cfg_aligned, padded_rows=True, tile_contiguous=False
            )
            k_1107 = mxfp8_gemm_decode(
                n, k, **cfg_1107, padded_rows=True, tile_contiguous=old_was_tc
            )
            k_tc = mxfp8_gemm_decode(
                n, k, **cfg_aligned, padded_rows=True, tile_contiguous=True
            )
            w_1107 = w_tc if old_was_tc else w_rm
            plan_1107 = plan_tc if old_was_tc else plan_rm

            y_rm = torch.empty((m, n), device=device, dtype=torch.bfloat16)
            y_1107 = torch.empty((m, n), device=device, dtype=torch.bfloat16)
            y_tc = torch.empty((m, n), device=device, dtype=torch.bfloat16)

            k_rm(a_fp8, w_rm, s_a_w, s_b_w, y_rm)
            k_1107(a_fp8, w_1107, s_a_w, s_b_w, y_1107)
            k_tc(a_fp8, w_tc, s_a_w, s_b_w, y_tc)
            torch.cuda.synchronize()

            diff_pre = int(
                (y_tc.view(torch.uint16) != y_pre_full[:m].view(torch.uint16)).sum().item()
            )
            diff_rm = int(
                (y_tc.view(torch.uint16) != y_rm.view(torch.uint16)).sum().item()
            )
            diff_1107 = int(
                (y_tc.view(torch.uint16) != y_1107.view(torch.uint16)).sum().item()
            )
            assert diff_pre == 0 and diff_rm == 0 and diff_1107 == 0, (
                n, k, m, diff_pre, diff_rm, diff_1107
            )
            shape_exact[str(m)] = {
                "diff_vs_prefill_u16": diff_pre,
                "diff_vs_rm_decode_u16": diff_rm,
                "diff_vs_1107_decode_u16": diff_1107,
            }

            def prep_cold():
                flush_buf.add_(1)

            def prep_next_rm():
                flush_buf.add_(1)
                ds_l2pf.issue(plan_rm, m)
                ds_l2pf.join()

            def prep_next_1107():
                flush_buf.add_(1)
                ds_l2pf.issue(plan_1107, m)
                ds_l2pf.join()

            def prep_next_tc():
                flush_buf.add_(1)
                ds_l2pf.issue(plan_tc, m)
                ds_l2pf.join()

            shape_perf[str(m)] = {
                "warm_us": {
                    "rm": _measure_warm(lambda: k_rm(a_fp8, w_rm, s_a_w, s_b_w, y_rm)),
                    "1107": _measure_warm(lambda: k_1107(a_fp8, w_1107, s_a_w, s_b_w, y_1107)),
                    "tc_aligned": _measure_warm(lambda: k_tc(a_fp8, w_tc, s_a_w, s_b_w, y_tc)),
                },
                "cold_us": {
                    "rm": _measure_single(lambda: k_rm(a_fp8, w_rm, s_a_w, s_b_w, y_rm), prep_cold),
                    "1107": _measure_single(lambda: k_1107(a_fp8, w_1107, s_a_w, s_b_w, y_1107), prep_cold),
                    "tc_aligned": _measure_single(lambda: k_tc(a_fp8, w_tc, s_a_w, s_b_w, y_tc), prep_cold),
                },
                "next_us": {
                    "rm": _measure_single(lambda: k_rm(a_fp8, w_rm, s_a_w, s_b_w, y_rm), prep_next_rm),
                    "1107": _measure_single(lambda: k_1107(a_fp8, w_1107, s_a_w, s_b_w, y_1107), prep_next_1107),
                    "tc_aligned": _measure_single(lambda: k_tc(a_fp8, w_tc, s_a_w, s_b_w, y_tc), prep_next_tc),
                },
            }

        results["bit_exactness"][key] = shape_exact
        results["qb_isolated"][key] = shape_perf

    # 2. Isolated indexer Q-B fused RoPE + MXFP4 epilogue (4096, 1280)
    n_idx, k_idx, heads, head_dim, rope_dim = 4096, 1280, 32, 128, 64
    w_idx_rm = (torch.randn((n_idx, k_idx), device=device, dtype=torch.bfloat16) * 0.1).to(
        torch.float8_e4m3fn
    )
    w_idx_tc = pack_weight_tiles(w_idx_rm)
    s_idx_b = torch.randint(120, 126, (n_idx // 32, k_idx // 32), device=device, dtype=torch.uint8)
    s_idx_bw = pack_scale_words(s_idx_b)
    cos_sin = torch.randn((4096, rope_dim), device=device, dtype=torch.float32)
    pos_all = torch.arange(DECODE_ROWS, device=device, dtype=torch.int64)

    k_idx_pre = mxfp8_gemm(n_idx, k_idx, **fp8_prefill_config(n_idx, k_idx))
    iq_ref_all = torch.empty((DECODE_ROWS, n_idx), device=device, dtype=torch.bfloat16)
    k_idx_pre(a_fp8, w_idx_rm, s_a_w, s_idx_bw, iq_ref_all)
    torch.cuda.synchronize()

    idx_exact: dict[str, dict] = {}
    idx_perf: dict[str, dict] = {}
    for m in BATCH_SIZES:
        block_m = _block_m_for(m)
        cfg_aligned = fp8_decode_config(n_idx, k_idx, block_m)
        cfg_1107 = OLD_1107_CONFIGS[(n_idx, k_idx)][block_m]

        ki_rm = mxfp8_gemm_rope_mxfp4_decode(
            n_idx, k_idx, head_dim=head_dim, rope_dim=rope_dim, **cfg_aligned, padded_rows=True, tile_contiguous=False
        )
        ki_1107 = mxfp8_gemm_rope_mxfp4_decode(
            n_idx, k_idx, head_dim=head_dim, rope_dim=rope_dim, **cfg_1107, padded_rows=True, tile_contiguous=True
        )
        ki_tc = mxfp8_gemm_rope_mxfp4_decode(
            n_idx, k_idx, head_dim=head_dim, rope_dim=rope_dim, **cfg_aligned, padded_rows=True, tile_contiguous=True
        )

        pos = pos_all[:m]
        p_ref = torch.empty((m, heads, head_dim // 2), device=device, dtype=torch.uint8)
        s_ref = torch.empty((m, heads, head_dim // 32), device=device, dtype=torch.uint8)
        quantize_mxfp4(
            iq_ref_all[:m].view(m, heads, head_dim),
            p_ref,
            s_ref,
            positions=pos,
            cos_sin=cos_sin,
        )

        p_rm = torch.empty_like(p_ref)
        sc_rm = torch.empty_like(s_ref)
        p_1107 = torch.empty_like(p_ref)
        sc_1107 = torch.empty_like(s_ref)
        p_tc = torch.empty_like(p_ref)
        sc_tc = torch.empty_like(s_ref)

        ki_rm(a_fp8, w_idx_rm, s_a_w, s_idx_bw, pos, cos_sin, p_rm.view(m, -1), sc_rm.view(m, -1))
        ki_1107(a_fp8, w_idx_tc, s_a_w, s_idx_bw, pos, cos_sin, p_1107.view(m, -1), sc_1107.view(m, -1))
        ki_tc(a_fp8, w_idx_tc, s_a_w, s_idx_bw, pos, cos_sin, p_tc.view(m, -1), sc_tc.view(m, -1))
        torch.cuda.synchronize()

        dp_ref = int((p_tc != p_ref).sum().item())
        ds_ref = int((sc_tc != s_ref).sum().item())
        dp_rm = int((p_tc != p_rm).sum().item())
        ds_rm = int((sc_tc != sc_rm).sum().item())
        dp_1107 = int((p_tc != p_1107).sum().item())
        ds_1107 = int((sc_tc != sc_1107).sum().item())
        assert dp_ref == 0 and ds_ref == 0 and dp_rm == 0 and ds_rm == 0 and dp_1107 == 0 and ds_1107 == 0
        idx_exact[str(m)] = {
            "packed_diff_vs_ref": dp_ref,
            "scale_diff_vs_ref": ds_ref,
            "packed_diff_vs_rm": dp_rm,
            "scale_diff_vs_rm": ds_rm,
            "packed_diff_vs_1107": dp_1107,
            "scale_diff_vs_1107": ds_1107,
        }

        def prep_cold():
            flush_buf.add_(1)

        p_rm_2d = p_rm.view(m, -1)
        sc_rm_2d = sc_rm.view(m, -1)
        p_1107_2d = p_1107.view(m, -1)
        sc_1107_2d = sc_1107.view(m, -1)
        p_tc_2d = p_tc.view(m, -1)
        sc_tc_2d = sc_tc.view(m, -1)

        idx_perf[str(m)] = {
            "warm_us": {
                "rm": _measure_warm(
                    lambda: ki_rm(a_fp8, w_idx_rm, s_a_w, s_idx_bw, pos, cos_sin, p_rm_2d, sc_rm_2d)
                ),
                "1107": _measure_warm(
                    lambda: ki_1107(a_fp8, w_idx_tc, s_a_w, s_idx_bw, pos, cos_sin, p_1107_2d, sc_1107_2d)
                ),
                "tc_aligned": _measure_warm(
                    lambda: ki_tc(a_fp8, w_idx_tc, s_a_w, s_idx_bw, pos, cos_sin, p_tc_2d, sc_tc_2d)
                ),
            },
            "cold_us": {
                "rm": _measure_single(
                    lambda: ki_rm(a_fp8, w_idx_rm, s_a_w, s_idx_bw, pos, cos_sin, p_rm_2d, sc_rm_2d),
                    prep_cold,
                ),
                "1107": _measure_single(
                    lambda: ki_1107(a_fp8, w_idx_tc, s_a_w, s_idx_bw, pos, cos_sin, p_1107_2d, sc_1107_2d),
                    prep_cold,
                ),
                "tc_aligned": _measure_single(
                    lambda: ki_tc(a_fp8, w_idx_tc, s_a_w, s_idx_bw, pos, cos_sin, p_tc_2d, sc_tc_2d),
                    prep_cold,
                ),
            },
        }

    results["bit_exactness"]["indexer_wq_b_4096x1280"] = idx_exact
    results["indexer_qb_isolated"]["4096x1280"] = idx_perf

    # 3. Chained Q-B + indexer Q-B (TP4: 8192 + 4096, TP3: 12288 + 4096)
    for label, n_qb in (("tp4_8192_plus_4096", 8192), ("tp3_12288_plus_4096", 12288)):
        w_q_rm = (torch.randn((n_qb, 1280), device=device, dtype=torch.bfloat16) * 0.1).to(
            torch.float8_e4m3fn
        )
        w_q_tc = pack_weight_tiles(w_q_rm)
        s_q_b = torch.randint(120, 126, (n_qb // 32, 40), device=device, dtype=torch.uint8)
        s_q_bw = pack_scale_words(s_q_b)
        plan_rm = _make_prefetch_plan([w_q_rm, s_q_bw, w_idx_rm, s_idx_bw], device)
        plan_tc = _make_prefetch_plan([w_q_tc, s_q_bw, w_idx_tc, s_idx_bw], device)
        chained_perf: dict[str, dict] = {}
        for m in BATCH_SIZES:
            block_m = _block_m_for(m)
            cfg_q_al = fp8_decode_config(n_qb, 1280, block_m)
            cfg_q_1107 = OLD_1107_CONFIGS[(n_qb, 1280)][block_m]
            cfg_i_al = fp8_decode_config(4096, 1280, block_m)
            cfg_i_1107 = OLD_1107_CONFIGS[(4096, 1280)][block_m]

            kq_rm = mxfp8_gemm_decode(n_qb, 1280, **cfg_q_al, padded_rows=True, tile_contiguous=False)
            ki_rm = mxfp8_gemm_rope_mxfp4_decode(
                4096, 1280, head_dim=head_dim, rope_dim=rope_dim, **cfg_i_al, padded_rows=True, tile_contiguous=False
            )

            kq_1107 = mxfp8_gemm_decode(n_qb, 1280, **cfg_q_1107, padded_rows=True, tile_contiguous=True)
            ki_1107 = mxfp8_gemm_rope_mxfp4_decode(
                4096, 1280, head_dim=head_dim, rope_dim=rope_dim, **cfg_i_1107, padded_rows=True, tile_contiguous=True
            )

            kq_tc = mxfp8_gemm_decode(n_qb, 1280, **cfg_q_al, padded_rows=True, tile_contiguous=True)
            ki_tc = mxfp8_gemm_rope_mxfp4_decode(
                4096, 1280, head_dim=head_dim, rope_dim=rope_dim, **cfg_i_al, padded_rows=True, tile_contiguous=True
            )

            pos = pos_all[:m]
            y_q = torch.empty((m, n_qb), device=device, dtype=torch.bfloat16)
            p_i = torch.empty((m, heads * (head_dim // 2)), device=device, dtype=torch.uint8)
            sc_i = torch.empty((m, heads * (head_dim // 32)), device=device, dtype=torch.uint8)

            def run_rm():
                kq_rm(a_fp8, w_q_rm, s_a_w, s_q_bw, y_q)
                ki_rm(a_fp8, w_idx_rm, s_a_w, s_idx_bw, pos, cos_sin, p_i, sc_i)

            def run_1107():
                kq_1107(a_fp8, w_q_tc, s_a_w, s_q_bw, y_q)
                ki_1107(a_fp8, w_idx_tc, s_a_w, s_idx_bw, pos, cos_sin, p_i, sc_i)

            def run_tc():
                kq_tc(a_fp8, w_q_tc, s_a_w, s_q_bw, y_q)
                ki_tc(a_fp8, w_idx_tc, s_a_w, s_idx_bw, pos, cos_sin, p_i, sc_i)

            def prep_cold():
                flush_buf.add_(1)

            def prep_next_rm():
                flush_buf.add_(1)
                ds_l2pf.issue(plan_rm, m)
                ds_l2pf.join()

            def prep_next_tc():
                flush_buf.add_(1)
                ds_l2pf.issue(plan_tc, m)
                ds_l2pf.join()

            chained_perf[str(m)] = {
                "warm_us": {
                    "rm": _measure_warm(run_rm),
                    "1107": _measure_warm(run_1107),
                    "tc_aligned": _measure_warm(run_tc),
                },
                "cold_us": {
                    "rm": _measure_single(run_rm, prep_cold),
                    "1107": _measure_single(run_1107, prep_cold),
                    "tc_aligned": _measure_single(run_tc, prep_cold),
                },
                "next_us": {
                    "rm": _measure_single(run_rm, prep_next_rm),
                    "1107": _measure_single(run_1107, prep_next_tc),
                    "tc_aligned": _measure_single(run_tc, prep_next_tc),
                },
            }
        results["chained_qb_plus_indexer_qb"][label] = chained_perf

    # 4. PDL LayerChain CUDA Graph replay (c1=6, c8=24, c8x6=48)
    saved_tc = set(ds_gemm.TILE_CONTIGUOUS_FP8)
    with set_current_vllm_config(VllmConfig()):
        for label, m in CHAIN_ROWS.items():
            ds_gemm.TILE_CONTIGUOUS_FP8.clear()
            chain_rm = LayerChain(m, seed=100 + m)
            chain_rm.poison()
            chain_rm.run()
            torch.cuda.synchronize()
            dig_rm = chain_rm.digests()

            ds_gemm.TILE_CONTIGUOUS_FP8.update(saved_tc)
            chain_tc = LayerChain(m, seed=100 + m)
            chain_tc.poison()
            chain_tc.run()
            torch.cuda.synchronize()
            dig_tc = chain_tc.digests()
            assert dig_rm == dig_tc, (label, m, dig_rm, dig_tc)

            g_rm = _capture_chain(chain_rm)
            g_tc = _capture_chain(chain_tc)

            def prep_cold():
                flush_buf.add_(1)

            results["layer_chain_pdl"][f"{label}_M{m}"] = {
                "digests_equal": dig_rm == dig_tc,
                "warm_us": {
                    "rm": _measure_warm(lambda: g_rm.replay(), warmup=15, outer=25, inner=10),
                    "tc_aligned": _measure_warm(lambda: g_tc.replay(), warmup=15, outer=25, inner=10),
                },
                "cold_us": {
                    "rm": _measure_single(lambda: g_rm.replay(), prep_cold, warmup=15, iters=60),
                    "tc_aligned": _measure_single(lambda: g_tc.replay(), prep_cold, warmup=15, iters=60),
                },
            }
            del g_rm, g_tc, chain_rm, chain_tc
            torch.cuda.empty_cache()

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
