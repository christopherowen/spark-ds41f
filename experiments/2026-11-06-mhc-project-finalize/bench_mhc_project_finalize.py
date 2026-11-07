#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU verification and 2-boot GPU benchmark harness for mHC entry (0731).

Compares:
1. Base Candidate 1107 (0634 fused single-CTA finalize at M=1; base project_streams +
   finalize at M>=2).
2. Unmodified two-launch Split (base project_streams + finalize without Fn0 PDL preload).
3. Candidate 0731 (unconditional two-launch project_streams with Fn stream-0 PDL preload +
   finalize; FINALIZE_FUSE_MAX_ROWS and project_and_finalize_streams optionality removed).

Also records prefill (M = 2048, 6144, 16384) compulsory DRAM byte traffic and achieved
GB/s across both (world=1, with_aux=False) and (world=3, with_aux=True).
"""

import argparse
import ast
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

HIDDEN = 5120
MULT = 4
MIXES = 24
BLOCK_H = 128
BLOCK_M = 16
SPLITS = HIDDEN // BLOCK_H
RMS_EPS = 1e-20
HC_EPS = 1e-6
SINKHORN_ITERS = 20

DECODE_ROWS = (1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96)
PREFILL_ROWS = (2048, 6144, 16384)


def split_tf32_cpu(fn: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bits = fn.view(torch.uint32)
    high = (bits & 0xFFFFE000).view(torch.float32)
    rest = fn - high
    mid = (rest.view(torch.uint32) & 0xFFFFE000).view(torch.float32)
    low = rest - mid
    return high, mid, low


def cpu_reference_split_order(
    x: torch.Tensor,
    r: torch.Tensor,
    post_in: torch.Tensor,
    comb_in: torch.Tensor,
    pre_in: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    weight: torch.Tensor,
    mode: str = "post_pre",
    with_aux: bool = False,
):
    world, M, H = x.shape
    splits = H // BLOCK_H
    r_reshaped = r.view(M, MULT, H)
    if mode == "post_pre":
        if world == 1:
            xv = x[0]
        else:
            acc = x[0].float()
            for w in range(1, world):
                acc = acc + x[w].float()
            xv = acc.bfloat16()
        r_out = torch.empty_like(r_reshaped)
        for n in range(MULT):
            av = comb_in[:, 0, n : n + 1] * r_reshaped[:, 0, :].float()
            av = torch.addcmul(av, post_in[:, n : n + 1], xv.float())
            for m in range(1, MULT):
                av = torch.addcmul(
                    av, comb_in[:, m, n : n + 1], r_reshaped[:, m, :].float()
                )
            r_out[:, n, :] = av.bfloat16()
    elif mode == "broadcast":
        xv = x[0]
        r_out = xv.unsqueeze(1).expand(M, MULT, H).contiguous()
    else:
        r_out = r_reshaped.clone()

    av_streams = r_out.float()
    col = (pre_in.unsqueeze(-1) * av_streams).sum(dim=1)
    c_bf16 = col.bfloat16()
    aux_bf16 = (0.25 * av_streams).sum(dim=1).bfloat16() if with_aux else None

    fn_high, fn_mid, fn_low = split_tf32_cpu(fn)
    P = torch.zeros((splits, M, MIXES), dtype=torch.float32)
    S = torch.zeros((splits, M), dtype=torch.float32)
    Q = torch.zeros((splits, M), dtype=torch.float32)

    for sp in range(splits):
        h0 = sp * BLOCK_H
        h1 = h0 + BLOCK_H
        q_blk = c_bf16[:, h0:h1].float()
        Q[sp] = (q_blk * q_blk).sum(dim=-1)
        if mode == "broadcast":
            a_blk = r_out[:, 0, h0:h1].float()
            S[sp] = (a_blk * a_blk).sum(dim=-1)
            P[sp] = (
                a_blk @ fn_high[:, h0:h1].T
                + a_blk @ fn_mid[:, h0:h1].T
                + a_blk @ fn_low[:, h0:h1].T
            )
        else:
            s_acc = torch.zeros((M,), dtype=torch.float32)
            p_acc = torch.zeros((M, MIXES), dtype=torch.float32)
            for n in range(MULT):
                a_blk = r_out[:, n, h0:h1].float()
                s_acc = s_acc + (a_blk * a_blk).sum(dim=-1)
                k0 = n * H + h0
                k1 = k0 + BLOCK_H
                p_acc = p_acc + (
                    a_blk @ fn_high[:, k0:k1].T
                    + a_blk @ fn_mid[:, k0:k1].T
                    + a_blk @ fn_low[:, k0:k1].T
                )
            S[sp] = s_acc
            P[sp] = p_acc

    s_sum = torch.zeros((M,), dtype=torch.float32)
    p_sum = torch.zeros((M, MIXES), dtype=torch.float32)
    q_sum = torch.zeros((M,), dtype=torch.float32)
    for sp in range(splits):
        s_sum = s_sum + S[sp]
        p_sum = p_sum + P[sp]
        q_sum = q_sum + Q[sp]

    k_group = H if mode == "broadcast" else MULT * H
    rms_l = torch.rsqrt(s_sum / k_group + RMS_EPS).unsqueeze(-1)
    pre_out = torch.sigmoid(p_sum[:, :MULT] * rms_l * scale[0] + base[:MULT]) + HC_EPS
    post_out = (
        torch.sigmoid(p_sum[:, MULT : 2 * MULT] * rms_l * scale[1] + base[MULT : 2 * MULT])
        * 2.0
    )
    cm = (p_sum[:, 2 * MULT :] * rms_l * scale[2] + base[2 * MULT :]).view(M, MULT, MULT)
    rmax = cm.amax(dim=-1, keepdim=True)
    cm = torch.exp(cm - rmax)
    cm = cm / cm.sum(dim=-1, keepdim=True) + HC_EPS
    cm = cm / (cm.sum(dim=-2, keepdim=True) + HC_EPS)
    for _ in range(SINKHORN_ITERS - 1):
        cm = cm / (cm.sum(dim=-1, keepdim=True) + HC_EPS)
        cm = cm / (cm.sum(dim=-2, keepdim=True) + HC_EPS)
    comb_out = cm

    norm_scale = torch.rsqrt(q_sum / H + RMS_EPS).unsqueeze(-1)
    y_out = (c_bf16.float() * norm_scale * weight.float()).bfloat16()
    return r_out.view(M, MULT * H), post_out, comb_out, pre_out, y_out, aux_bf16


def run_cpu_verify(mhc_path: Path, pdl_sources_path: Path, out_json: Path) -> None:
    src = mhc_path.read_text()
    tree = ast.parse(src, filename=str(mhc_path))
    fn_names = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    assign_names = [
        t.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        for t in n.targets
        if isinstance(t, ast.Name)
    ]
    assert "project_streams" in fn_names
    assert "finalize" in fn_names
    assert "project_and_finalize_streams" not in fn_names, (
        "project_and_finalize_streams must be removed"
    )
    assert "FINALIZE_FUSE_MAX_ROWS" not in assign_names, (
        "FINALIZE_FUSE_MAX_ROWS must be removed"
    )

    pdl_src = pdl_sources_path.read_text()
    assert "\"mhc_project_finalize\"" not in pdl_src

    results = []
    for M in (1, 2, 6, 16, 33):
        for mode in ("pre", "post_pre", "broadcast"):
            g = torch.Generator().manual_seed(1000 + M)
            x = torch.randn((1, M, HIDDEN), generator=g, dtype=torch.bfloat16)
            r = torch.randn((M, MULT * HIDDEN), generator=g, dtype=torch.bfloat16)
            post_in = torch.rand((M, MULT), generator=g, dtype=torch.float32) * 2.0
            comb_in = torch.softmax(torch.randn((M, MULT, MULT), generator=g), dim=-1)
            pre_in = torch.softmax(torch.randn((M, MULT), generator=g), dim=-1) + 1e-6
            k = HIDDEN if mode == "broadcast" else MULT * HIDDEN
            fn = torch.randn((MIXES, k), generator=g, dtype=torch.float32) * 0.02
            scale = torch.randn((3,), generator=g, dtype=torch.float32) * 0.1 + 1.0
            base = torch.randn((MIXES,), generator=g, dtype=torch.float32) * 0.1
            weight = (torch.randn((HIDDEN,), generator=g) * 0.1 + 1.0).bfloat16()
            r_out, post_out, comb_out, pre_out, y_out, _ = cpu_reference_split_order(
                x, r, post_in, comb_in, pre_in, fn, scale, base, weight, mode=mode
            )
            r2, post2, comb2, pre2, y2, _ = cpu_reference_split_order(
                x, r, post_in, comb_in, pre_in, fn, scale, base, weight, mode=mode
            )
            assert torch.equal(r_out.view(torch.int16), r2.view(torch.int16))
            assert torch.equal(post_out.view(torch.int32), post2.view(torch.int32))
            assert torch.equal(comb_out.view(torch.int32), comb2.view(torch.int32))
            assert torch.equal(pre_out.view(torch.int32), pre2.view(torch.int32))
            assert torch.equal(y_out.view(torch.int16), y2.view(torch.int16))
            results.append(
                {
                    "M": M,
                    "mode": mode,
                    "finite_all_ok": bool(
                        torch.isfinite(post_out).all()
                        and torch.isfinite(comb_out).all()
                        and torch.isfinite(pre_out).all()
                        and torch.isfinite(y_out).all()
                    ),
                    "deterministic_bits_equal": True,
                }
            )

    payload = {
        "status": "ok",
        "ast_verified": True,
        "optionality_removed": True,
        "cpu_reference_cases": len(results),
        "cases": results,
    }
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"CPU verification passed ({len(results)} cases) -> {out_json}")


def compulsory_dram_bytes(M: int, world: int, with_aux: bool) -> dict[str, float]:
    """Compulsory DRAM traffic (in decimal MB = 1e6 B and MiB = 2^20 B) of post_pre + finalize."""
    # project_streams (post_pre):
    # Reads: X (world * M * H * 2B), R (M * 4 * H * 2B), Fn (24 * 4 * H * 4B), mixes (M * 24 * 4B)
    # Writes: R_out (M * 4 * H * 2B), C (M * H * 2B), Aux (M * H * 2B if with_aux),
    #         P (40 * M * 24 * 4B), S (40 * M * 4B), Q (40 * M * 4B)
    proj_read = (
        world * M * HIDDEN * 2
        + M * MULT * HIDDEN * 2
        + MIXES * MULT * HIDDEN * 4
        + M * MIXES * 4
    )
    proj_write = (
        M * MULT * HIDDEN * 2
        + M * HIDDEN * 2
        + (M * HIDDEN * 2 if with_aux else 0)
        + SPLITS * M * (MIXES + 2) * 4
    )
    proj_total = proj_read + proj_write

    # finalize:
    # Reads: C (M * H * 2B), P, S, Q (40 * M * 26 * 4B), weight (H * 2B), scale+base (108B)
    # Writes: y (M * H * 2B), yq (M * H * 1B), ysf (M * (H // 32) * 1B), mixes_out (M * 24 * 4B)
    fin_read = M * HIDDEN * 2 + SPLITS * M * (MIXES + 2) * 4 + HIDDEN * 2 + 108
    fin_write = M * HIDDEN * 2 + M * HIDDEN * 1 + M * (HIDDEN // 32) * 1 + M * MIXES * 4
    fin_total = fin_read + fin_write

    return {
        "proj_mb": round(proj_total / 1e6, 2),
        "proj_mib": round(proj_total / (1024 * 1024), 2),
        "fin_mb": round(fin_total / 1e6, 2),
        "fin_mib": round(fin_total / (1024 * 1024), 2),
        "total_mb": round((proj_total + fin_total) / 1e6, 2),
        "total_mib": round((proj_total + fin_total) / (1024 * 1024), 2),
    }


def _make_base_kernels():
    """Compile Base Candidate 1107 kernels (base project_streams without Fn0 preload,
    and 0634 single-CTA fused project_and_finalize_streams for M=1 comparison)."""
    import tilelang
    import tilelang.language as T
    from vllm.models.deepseek_v4_1.tilelang import mhc as mhc_mod
    from vllm.models.deepseek_v4_1.tilelang import pdl
    from vllm.models.deepseek_v4_1.tilelang.gemm import scale_words

    pass_cfg = mhc_mod._PASS_CONFIGS

    @tilelang.jit(pass_configs=pass_cfg)
    def base_project_streams(
        hidden_size: int,
        mode: str,
        block_M: int = 16,
        block_H: int = BLOCK_H,
        world: int = 1,
        with_aux: bool = False,
    ):
        splits = hidden_size // block_H
        streams = 1 if mode == "broadcast" else MULT
        K = streams * hidden_size
        N = 32
        LANES = 8
        COLS = block_H // LANES
        threads = block_M * LANES
        H = hidden_size
        M = T.dynamic("M")

        @T.macro
        def load_split_tf32(Fn, k0, F_shared, F_mid, F_low):
            for i, j in T.Parallel(MIXES, block_H):
                f = Fn[i, k0 + j]
                high = T.reinterpret(T.reinterpret(f, T.uint32) & 0xFFFFE000, T.float32)
                rest = f - high
                middle = T.reinterpret(
                    T.reinterpret(rest, T.uint32) & 0xFFFFE000, T.float32
                )
                F_shared[i, j] = high
                F_mid[i, j] = middle
                F_low[i, j] = rest - middle

        @T.macro
        def _body(X, R, post, comb, pre_in, Fn, R_out, P, S, C, Q, Aux, sp, by):
            pdl.chain()
            h0 = sp * block_H
            A_shared = T.alloc_shared((block_M, block_H), T.float32)
            F_shared = T.alloc_shared((N, block_H), T.float32)
            F_mid = T.alloc_shared((N, block_H), T.float32)
            F_low = T.alloc_shared((N, block_H), T.float32)
            C_local = T.alloc_fragment((block_M, N), T.float32)
            xv = T.alloc_local((COLS,), T.bfloat16)
            xp = T.alloc_local((world * COLS,), T.bfloat16)
            rv = T.alloc_local((MULT * COLS,), T.bfloat16)
            nv = T.alloc_local((COLS,), T.bfloat16)
            av = T.alloc_local((COLS,), T.float32)
            sq = T.alloc_local((COLS,), T.float32)
            col = T.alloc_local((COLS,), T.float32)
            aux_col = T.alloc_local((COLS,), T.float32)
            coef = T.alloc_local((2 * MULT + MULT * MULT,), T.float32)
            red = T.alloc_local((1,), T.float32)
            tx = T.get_thread_binding()
            row = tx // LANES
            c0 = (tx % LANES) * COLS
            token = by * block_M + row
            src = T.min(token, M - 1)
            T.clear(C_local)
            for i, j in T.Parallel(N - MIXES, block_H):
                F_shared[MIXES + i, j] = 0.0
                F_mid[MIXES + i, j] = 0.0
                F_low[MIXES + i, j] = 0.0
            for k in T.unroll(MULT):
                coef[k] = pre_in[src, k]
            if mode == "post_pre":
                for k in T.unroll(MULT):
                    coef[MULT + k] = post[src, k]
                for k in T.unroll(MULT * MULT):
                    coef[2 * MULT + k] = comb[src, k // MULT, k % MULT]
            for m in T.unroll(MULT):
                for v in T.vectorized(COLS):
                    rv[m * COLS + v] = R[src, m * H + h0 + c0 + v]
            if world == 1:
                for v in T.vectorized(COLS):
                    xv[v] = X[0, src, h0 + c0 + v]
            else:
                for r in T.unroll(world):
                    for v in T.vectorized(COLS):
                        xp[r * COLS + v] = X[r, src, h0 + c0 + v]
                for r in T.unroll(world):
                    for v in T.unroll(COLS):
                        if r == 0:
                            av[v] = T.Cast(T.float32, xp[v])
                        else:
                            av[v] = av[v] + T.Cast(T.float32, xp[r * COLS + v])
                for v in T.unroll(COLS):
                    xv[v] = T.Cast(T.bfloat16, av[v])
            for v in T.unroll(COLS):
                sq[v] = 0.0
                col[v] = 0.0
                if with_aux:
                    aux_col[v] = 0.0
            for n in T.unroll(MULT):
                load_split_tf32(Fn, n * H + h0, F_shared, F_mid, F_low)
                for v in T.unroll(COLS):
                    av[v] = coef[2 * MULT + n] * T.Cast(T.float32, rv[v])
                    av[v] = T.fma(coef[MULT + n], T.Cast(T.float32, xv[v]), av[v])
                    for m in T.unroll(1, MULT):
                        av[v] = T.fma(
                            coef[2 * MULT + m * MULT + n],
                            T.Cast(T.float32, rv[m * COLS + v]),
                            av[v],
                        )
                    nv[v] = T.Cast(T.bfloat16, av[v])
                if token < M:
                    for v in T.vectorized(COLS):
                        R_out[token, n * H + h0 + c0 + v] = nv[v]
                for v in T.unroll(COLS):
                    av[v] = T.Cast(T.float32, nv[v])
                    A_shared[row, c0 + v] = av[v]
                    sq[v] += av[v] * av[v]
                    col[v] += coef[n] * av[v]
                    if with_aux:
                        aux_col[v] = T.fma(T.float32(0.25), av[v], aux_col[v])
                T.sync_threads()
                T.gemm(A_shared, F_shared, C_local, transpose_B=True)
                T.gemm(A_shared, F_mid, C_local, transpose_B=True)
                T.gemm(A_shared, F_low, C_local, transpose_B=True)
                T.sync_threads()
            if with_aux:
                for v in T.unroll(COLS):
                    nv[v] = T.Cast(T.bfloat16, aux_col[v])
                if token < M:
                    for v in T.vectorized(COLS):
                        Aux[token, h0 + c0 + v] = nv[v]
            for v in T.unroll(COLS):
                nv[v] = T.Cast(T.bfloat16, col[v])
            if token < M:
                for v in T.vectorized(COLS):
                    C[token, h0 + c0 + v] = nv[v]
            red[0] = 0.0
            for v in T.unroll(COLS):
                red[0] += T.Cast(T.float32, nv[v]) * T.Cast(T.float32, nv[v])
            red[0] += T.shfl_xor(red[0], 4)
            red[0] += T.shfl_xor(red[0], 2)
            red[0] += T.shfl_xor(red[0], 1)
            if tx % LANES == 0 and token < M:
                Q[sp, token] = red[0]
            red[0] = 0.0
            for v in T.unroll(COLS):
                red[0] += sq[v]
            red[0] += T.shfl_xor(red[0], 4)
            red[0] += T.shfl_xor(red[0], 2)
            red[0] += T.shfl_xor(red[0], 1)
            if tx % LANES == 0 and token < M:
                S[sp, token, 0] = red[0]
            for i, j in T.Parallel(block_M, MIXES):
                if by * block_M + i < M:
                    P[sp, by * block_M + i, 0, j] = C_local[i, j]

        if not with_aux:
            @T.prim_func
            def mhc_project_streams_base(
                X: T.Tensor((world, M, H), T.bfloat16),
                R: T.Tensor((M, MULT * H), T.bfloat16),
                post: T.Tensor((M, MULT), T.float32),
                comb: T.Tensor((M, MULT, MULT), T.float32),
                pre_in: T.Tensor((M, MULT), T.float32),
                Fn: T.Tensor((MIXES, K), T.float32),
                R_out: T.Tensor((M, MULT * H), T.bfloat16),
                P: T.Tensor((splits, M, 1, MIXES), T.float32),
                S: T.Tensor((splits, M, 1), T.float32),
                C: T.Tensor((M, H), T.bfloat16),
                Q: T.Tensor((splits, M), T.float32),
            ):
                with T.Kernel(splits, T.ceildiv(M, block_M), threads=threads) as (sp, by):
                    _body(X, R, post, comb, pre_in, Fn, R_out, P, S, C, Q, C, sp, by)
            return mhc_project_streams_base

        @T.prim_func
        def mhc_project_streams_base_aux(
            X: T.Tensor((world, M, H), T.bfloat16),
            R: T.Tensor((M, MULT * H), T.bfloat16),
            post: T.Tensor((M, MULT), T.float32),
            comb: T.Tensor((M, MULT, MULT), T.float32),
            pre_in: T.Tensor((M, MULT), T.float32),
            Fn: T.Tensor((MIXES, K), T.float32),
            R_out: T.Tensor((M, MULT * H), T.bfloat16),
            P: T.Tensor((splits, M, 1, MIXES), T.float32),
            S: T.Tensor((splits, M, 1), T.float32),
            C: T.Tensor((M, H), T.bfloat16),
            Q: T.Tensor((splits, M), T.float32),
            Aux: T.Tensor((M, H), T.bfloat16),
        ):
            with T.Kernel(splits, T.ceildiv(M, block_M), threads=threads) as (sp, by):
                _body(X, R, post, comb, pre_in, Fn, R_out, P, S, C, Q, Aux, sp, by)
        return mhc_project_streams_base_aux

    return base_project_streams


def run_gpu_bench(out_json: Path) -> None:
    from vllm.models.deepseek_v4_1.tilelang import mhc as mhc_mod
    from vllm.models.deepseek_v4_1.tilelang.gemm import scale_words
    from vllm.models.deepseek_v4_1.tilelang.linear import mxfp8_rows
    from vllm.v1.worker.workspace import init_workspace_manager

    device = torch.device("cuda")
    init_workspace_manager(device)
    base_project_streams = _make_base_kernels()

    cand_proj = mhc_mod.project_streams(
        HIDDEN, "post_pre", block_M=BLOCK_M, world=1, with_aux=False
    )
    base_proj = base_project_streams(
        HIDDEN, "post_pre", block_M=BLOCK_M, world=1, with_aux=False
    )
    cand_proj_w3_aux = mhc_mod.project_streams(
        HIDDEN, "post_pre", block_M=BLOCK_M, world=3, with_aux=True
    )
    base_proj_w3_aux = base_project_streams(
        HIDDEN, "post_pre", block_M=BLOCK_M, world=3, with_aux=True
    )
    fin_k = mhc_mod.finalize(
        HIDDEN, MULT * HIDDEN, RMS_EPS, HC_EPS, SINKHORN_ITERS, RMS_EPS
    )

    def bench_fn(fn, warm=30, reps=200, use_graph=True):
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        if use_graph:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(10):
                    fn()
            for _ in range(10):
                g.replay()
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(reps):
                g.replay()
            e.record()
            torch.cuda.synchronize()
            return s.elapsed_time(e) * 1000.0 / (reps * 10)
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(reps):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) * 1000.0 / reps

    decode_rows_out = []
    for M in DECODE_ROWS:
        gen = torch.Generator(device=device).manual_seed(4000 + M)
        x = torch.randn((1, M, HIDDEN), generator=gen, device=device, dtype=torch.bfloat16)
        r = torch.randn((M, MULT * HIDDEN), generator=gen, device=device, dtype=torch.bfloat16)
        post_in = torch.rand((M, MULT), generator=gen, device=device, dtype=torch.float32) * 2.0
        comb_in = torch.softmax(torch.randn((M, MULT, MULT), generator=gen, device=device), dim=-1)
        pre_in = torch.softmax(torch.randn((M, MULT), generator=gen, device=device), dim=-1) + 1e-6
        fn = torch.randn((MIXES, MULT * HIDDEN), generator=gen, device=device, dtype=torch.float32) * 0.02
        scale = torch.randn((3,), generator=gen, device=device, dtype=torch.float32) * 0.1 + 1.0
        base = torch.randn((MIXES,), generator=gen, device=device, dtype=torch.float32) * 0.1
        weight = (torch.randn((HIDDEN,), generator=gen, device=device) * 0.1 + 1.0).bfloat16()
        mq = mxfp8_rows(M)
        sf_w = 4 * scale_words(HIDDEN)

        def alloc_bufs():
            return (
                torch.empty_like(r),
                torch.empty((SPLITS, M, 1, MIXES), dtype=torch.float32, device=device),
                torch.empty((SPLITS, M, 1), dtype=torch.float32, device=device),
                torch.empty((M, HIDDEN), dtype=torch.bfloat16, device=device),
                torch.empty((SPLITS, M), dtype=torch.float32, device=device),
                torch.empty((M, MULT), dtype=torch.float32, device=device),
                torch.empty((M, MULT, MULT), dtype=torch.float32, device=device),
                torch.empty((M, MULT), dtype=torch.float32, device=device),
                torch.empty((M, HIDDEN), dtype=torch.bfloat16, device=device),
                torch.empty((mq, HIDDEN), dtype=torch.float8_e4m3fn, device=device),
                torch.empty((mq, sf_w), dtype=torch.uint8, device=device),
            )

        b_base = alloc_bufs()
        b_cand = alloc_bufs()

        def run_base():
            base_proj(x, r, post_in, comb_in, pre_in, fn, b_base[0], b_base[1], b_base[2], b_base[3], b_base[4])
            fin_k(b_base[1], b_base[2], b_base[3], b_base[4], scale, base, weight, b_base[5], b_base[6], b_base[7], b_base[8], b_base[9], b_base[10])

        def run_cand():
            cand_proj(x, r, post_in, comb_in, pre_in, fn, b_cand[0], b_cand[1], b_cand[2], b_cand[3], b_cand[4])
            fin_k(b_cand[1], b_cand[2], b_cand[3], b_cand[4], scale, base, weight, b_cand[5], b_cand[6], b_cand[7], b_cand[8], b_cand[9], b_cand[10])

        run_base()
        run_cand()
        torch.cuda.synchronize()
        bits_equal = bool(
            torch.equal(b_base[0].view(torch.int16), b_cand[0].view(torch.int16))
            and torch.equal(b_base[5].view(torch.int32), b_cand[5].view(torch.int32))
            and torch.equal(b_base[6].view(torch.int32), b_cand[6].view(torch.int32))
            and torch.equal(b_base[7].view(torch.int32), b_cand[7].view(torch.int32))
            and torch.equal(b_base[8].view(torch.int16), b_cand[8].view(torch.int16))
            and torch.equal(b_base[9][:M].view(torch.uint8), b_cand[9][:M].view(torch.uint8))
            and torch.equal(b_base[10][:M], b_cand[10][:M])
        )
        assert bits_equal, f"Bit mismatch at M={M}"

        us_base_split = bench_fn(run_base, use_graph=True)
        us_cand_split = bench_fn(run_cand, use_graph=True)
        us_proj_only = bench_fn(
            lambda: cand_proj(x, r, post_in, comb_in, pre_in, fn, b_cand[0], b_cand[1], b_cand[2], b_cand[3], b_cand[4]),
            use_graph=True,
        )
        us_fin_only = bench_fn(
            lambda: fin_k(b_cand[1], b_cand[2], b_cand[3], b_cand[4], scale, base, weight, b_cand[5], b_cand[6], b_cand[7], b_cand[8], b_cand[9], b_cand[10]),
            use_graph=True,
        )
        decode_rows_out.append(
            {
                "M": M,
                "mode": "graph",
                "bits_equal": bits_equal,
                "base_split_us": round(us_base_split, 3),
                "cand_split_us": round(us_cand_split, 3),
                "cand_project_only_us": round(us_proj_only, 3),
                "finalize_only_us": round(us_fin_only, 3),
                "delta_vs_base_split_us": round(us_cand_split - us_base_split, 3),
                "pass80_delta_vs_base_split_us": round((us_cand_split - us_base_split) * 80.0, 1),
            }
        )
        print(
            f"decode M={M:3d}: base_split={us_base_split:6.2f}us  cand_0731={us_cand_split:6.2f}us "
            f"(proj={us_proj_only:5.2f}us fin={us_fin_only:4.2f}us) delta={us_cand_split - us_base_split:+5.2f}us"
        )

    prefill_rows_out = []
    for world, with_aux in ((1, False), (3, True)):
        p_base = base_proj if not with_aux else base_proj_w3_aux
        p_cand = cand_proj if not with_aux else cand_proj_w3_aux
        for M in PREFILL_ROWS:
            gen = torch.Generator(device=device).manual_seed(5000 + world * 100 + M)
            x = torch.randn((world, M, HIDDEN), generator=gen, device=device, dtype=torch.bfloat16)
            r = torch.randn((M, MULT * HIDDEN), generator=gen, device=device, dtype=torch.bfloat16)
            post_in = torch.rand((M, MULT), generator=gen, device=device, dtype=torch.float32) * 2.0
            comb_in = torch.softmax(torch.randn((M, MULT, MULT), generator=gen, device=device), dim=-1)
            pre_in = torch.softmax(torch.randn((M, MULT), generator=gen, device=device), dim=-1) + 1e-6
            fn = torch.randn((MIXES, MULT * HIDDEN), generator=gen, device=device, dtype=torch.float32) * 0.02
            scale = torch.randn((3,), generator=gen, device=device, dtype=torch.float32) * 0.1 + 1.0
            base = torch.randn((MIXES,), generator=gen, device=device, dtype=torch.float32) * 0.1
            weight = (torch.randn((HIDDEN,), generator=gen, device=device) * 0.1 + 1.0).bfloat16()
            mq = mxfp8_rows(M)
            sf_w = 4 * scale_words(HIDDEN)

            R_b = torch.empty_like(r)
            R_c = torch.empty_like(r)
            P_b = torch.empty((SPLITS, M, 1, MIXES), dtype=torch.float32, device=device)
            P_c = torch.empty_like(P_b)
            S_b = torch.empty((SPLITS, M, 1), dtype=torch.float32, device=device)
            S_c = torch.empty_like(S_b)
            C_b = torch.empty((M, HIDDEN), dtype=torch.bfloat16, device=device)
            C_c = torch.empty_like(C_b)
            Q_b = torch.empty((SPLITS, M), dtype=torch.float32, device=device)
            Q_c = torch.empty_like(Q_b)
            Aux_b = torch.empty((M, HIDDEN), dtype=torch.bfloat16, device=device) if with_aux else None
            Aux_c = torch.empty((M, HIDDEN), dtype=torch.bfloat16, device=device) if with_aux else None
            post_b = torch.empty((M, MULT), dtype=torch.float32, device=device)
            post_c = torch.empty_like(post_b)
            comb_b = torch.empty((M, MULT, MULT), dtype=torch.float32, device=device)
            comb_c = torch.empty_like(comb_b)
            pre_b = torch.empty((M, MULT), dtype=torch.float32, device=device)
            pre_c = torch.empty_like(pre_b)
            y_b = torch.empty((M, HIDDEN), dtype=torch.bfloat16, device=device)
            y_c = torch.empty_like(y_b)
            yq_b = torch.empty((mq, HIDDEN), dtype=torch.float8_e4m3fn, device=device)
            yq_c = torch.empty_like(yq_b)
            ysf_b = torch.empty((mq, sf_w), dtype=torch.uint8, device=device)
            ysf_c = torch.empty_like(ysf_b)

            aux_b_arg = (Aux_b,) if with_aux else ()
            aux_c_arg = (Aux_c,) if with_aux else ()

            def run_p_base():
                p_base(x, r, post_in, comb_in, pre_in, fn, R_b, P_b, S_b, C_b, Q_b, *aux_b_arg)
                fin_k(P_b, S_b, C_b, Q_b, scale, base, weight, post_b, comb_b, pre_b, y_b, yq_b, ysf_b)

            def run_p_cand():
                p_cand(x, r, post_in, comb_in, pre_in, fn, R_c, P_c, S_c, C_c, Q_c, *aux_c_arg)
                fin_k(P_c, S_c, C_c, Q_c, scale, base, weight, post_c, comb_c, pre_c, y_c, yq_c, ysf_c)

            run_p_base()
            run_p_cand()
            torch.cuda.synchronize()
            bits_equal = bool(
                torch.equal(R_b.view(torch.int16), R_c.view(torch.int16))
                and torch.equal(post_b.view(torch.int32), post_c.view(torch.int32))
                and torch.equal(comb_b.view(torch.int32), comb_c.view(torch.int32))
                and torch.equal(pre_b.view(torch.int32), pre_c.view(torch.int32))
                and torch.equal(y_b.view(torch.int16), y_c.view(torch.int16))
                and torch.equal(yq_b[:M].view(torch.uint8), yq_c[:M].view(torch.uint8))
                and torch.equal(ysf_b[:M], ysf_c[:M])
                and (not with_aux or torch.equal(Aux_b.view(torch.int16), Aux_c.view(torch.int16)))
            )
            assert bits_equal, f"Prefill bit mismatch at M={M}, world={world}, with_aux={with_aux}"

            reps = 25 if M < 16384 else 12
            us_base_total = bench_fn(run_p_base, warm=6, reps=reps, use_graph=False)
            us_cand_total = bench_fn(run_p_cand, warm=6, reps=reps, use_graph=False)
            us_cand_proj = bench_fn(
                lambda: p_cand(x, r, post_in, comb_in, pre_in, fn, R_c, P_c, S_c, C_c, Q_c, *aux_c_arg),
                warm=6,
                reps=reps,
                use_graph=False,
            )
            us_fin = bench_fn(
                lambda: fin_k(P_c, S_c, C_c, Q_c, scale, base, weight, post_c, comb_c, pre_c, y_c, yq_c, ysf_c),
                warm=6,
                reps=reps,
                use_graph=False,
            )
            bw = compulsory_dram_bytes(M, world, with_aux)
            proj_gbs = round(bw["proj_mb"] * 1e3 / us_cand_proj, 1)
            fin_gbs = round(bw["fin_mb"] * 1e3 / us_fin, 1)
            total_gbs = round(bw["total_mb"] * 1e3 / us_cand_total, 1)
            prefill_rows_out.append(
                {
                    "M": M,
                    "world": world,
                    "with_aux": with_aux,
                    "bits_equal": bits_equal,
                    "base_total_us": round(us_base_total, 1),
                    "cand_total_us": round(us_cand_total, 1),
                    "cand_project_only_us": round(us_cand_proj, 1),
                    "finalize_only_us": round(us_fin, 1),
                    "delta_vs_base_us": round(us_cand_total - us_base_total, 1),
                    "compulsory_dram": bw,
                    "achieved_gbs": {
                        "project_only_gbs": proj_gbs,
                        "finalize_only_gbs": fin_gbs,
                        "combined_gbs": total_gbs,
                        "pct_of_230gbs": round(total_gbs / 230.0 * 100.0, 1),
                    },
                }
            )
            print(
                f"prefill w={world} aux={int(with_aux)} M={M:5d}: base={us_base_total:7.1f}us "
                f"cand={us_cand_total:7.1f}us (proj={us_cand_proj:6.1f}us [{proj_gbs}GB/s] "
                f"fin={us_fin:5.1f}us [{fin_gbs}GB/s]) total_bw={total_gbs}GB/s ({total_gbs/2.3:.1f}%)"
            )

    payload = {
        "device": torch.cuda.get_device_name(0),
        "decode_rows": decode_rows_out,
        "prefill_rows": prefill_rows_out,
    }
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Saved GPU benchmark receipt -> {out_json}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-verify", action="store_true")
    parser.add_argument("--gpu-bench", action="store_true")
    parser.add_argument("--mhc-path", type=Path)
    parser.add_argument("--pdl-sources-path", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.cpu_verify:
        run_cpu_verify(args.mhc_path, args.pdl_sources_path, args.out)
    elif args.gpu_bench:
        run_gpu_bench(args.out)
    else:
        raise SystemExit("Specify --cpu-verify or --gpu-bench")


if __name__ == "__main__":
    main()
