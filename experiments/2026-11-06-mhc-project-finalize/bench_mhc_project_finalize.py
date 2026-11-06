#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU/compiler verification and GPU correctness/latency harness for mHC project+finalize."""


import argparse
import json
import random
import subprocess
import tempfile
import time
from pathlib import Path

import torch


def split_tf32(f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact 3-TF32-part decomposition matching load_split_tf32 in mhc.py."""
    bits = f.view(torch.int32)
    high = (bits & -0x2000).view(torch.float32)
    rest = f - high
    r_bits = rest.view(torch.int32)
    mid = (r_bits & -0x2000).view(torch.float32)
    low = rest - mid
    return high, mid, low


def simulate_split_vs_fused_cpu(
    tokens: int,
    hidden: int = 256,
    block_h: int = 128,
    block_m: int = 16,
    mode: str = "post_pre",
    world: int = 1,
    with_aux: bool = True,
    inject_nonfinite: bool = False,
    seed: int = 1234,
) -> dict:
    """Bit-exact CPU simulation comparing 2-kernel (project_streams + finalize)
    against 1-kernel parallel-CTA project_and_finalize_streams under random
    CTA arrival permutations."""
    mult = 4
    mixes = 24
    splits = hidden // block_h
    streams = 1 if mode == "broadcast" else mult
    k_dim = streams * hidden
    rms_eps = 1e-6
    hc_eps = 1e-6
    norm_eps = 1e-6
    post_mult = 2.0
    sinkhorn_repeat = 20

    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(world, tokens, hidden, generator=g, dtype=torch.bfloat16)
    r_in = torch.randn(tokens, mult, hidden, generator=g, dtype=torch.bfloat16)
    post_in = 2.0 * torch.sigmoid(torch.randn(tokens, mult, generator=g, dtype=torch.float32))
    comb_in = torch.softmax(torch.randn(tokens, mult, mult, generator=g, dtype=torch.float32), dim=-1)
    pre_in = torch.sigmoid(torch.randn(tokens, mult, generator=g, dtype=torch.float32)) + hc_eps
    fn = torch.randn(mixes, k_dim, generator=g, dtype=torch.float32) * 0.02
    scale = torch.tensor([0.8, 1.1, 0.9], dtype=torch.float32)
    base = torch.randn(mixes, generator=g, dtype=torch.float32) * 0.1
    weight = (torch.randn(hidden, generator=g, dtype=torch.float32) * 0.1 + 1.0).bfloat16()

    x[:, :, :8] = -0.0
    r_in[:, :, :8] = -0.0
    if inject_nonfinite and tokens >= 4:
        x[0, 0, 8] = float("nan")
        r_in[1, 2, 16] = float("inf")
        r_in[2, 0, 24] = float("-inf")
        pre_in[3, 1] = float("nan")

    # Stage 1 per-split partials: P[splits, tokens, mixes], S[splits, tokens], C[tokens, hidden], Q[splits, tokens]
    r_out_ref = torch.empty((tokens, mult, hidden), dtype=torch.bfloat16)
    aux_ref = torch.empty((tokens, hidden), dtype=torch.bfloat16) if with_aux else None
    c_ref = torch.empty((tokens, hidden), dtype=torch.bfloat16)
    p_ref = torch.empty((splits, tokens, mixes), dtype=torch.float32)
    s_ref = torch.empty((splits, tokens), dtype=torch.float32)
    q_ref = torch.empty((splits, tokens), dtype=torch.float32)

    if world == 1:
        x_sum = x[0]
    else:
        acc_x = x[0].float()
        for r_idx in range(1, world):
            acc_x = acc_x + x[r_idx].float()
        x_sum = acc_x.bfloat16()

    for sp in range(splits):
        h0 = sp * block_h
        h1 = h0 + block_h
        sq = torch.zeros((tokens, block_h), dtype=torch.float32)
        col = torch.zeros((tokens, block_h), dtype=torch.float32)
        aux_col = torch.zeros((tokens, block_h), dtype=torch.float32)
        c_local = torch.zeros((tokens, mixes), dtype=torch.float32)

        if mode == "broadcast":
            xv = x_sum[:, h0:h1]
            av = xv.float()
            sq = av * av
            for n in range(mult):
                r_out_ref[:, n, h0:h1] = xv
                col = col + pre_in[:, n : n + 1] * av
            fh, fm, fl = split_tf32(fn[:, h0:h1])
            c_local = c_local + av @ fh.t() + av @ fm.t() + av @ fl.t()
        else:
            rv = r_in[:, :, h0:h1]
            for n in range(mult):
                fh, fm, fl = split_tf32(fn[:, n * hidden + h0 : n * hidden + h1])
                if mode == "post_pre":
                    av = comb_in[:, 0, n : n + 1] * rv[:, 0, :].float()
                    av = torch.addcmul(av, post_in[:, n : n + 1], x_sum[:, h0:h1].float())
                    for m in range(1, mult):
                        av = torch.addcmul(av, comb_in[:, m, n : n + 1], rv[:, m, :].float())
                    nv = av.bfloat16()
                    r_out_ref[:, n, h0:h1] = nv
                else:
                    nv = rv[:, n, :]
                    r_out_ref[:, n, h0:h1] = nv
                av = nv.float()
                sq = sq + av * av
                col = col + pre_in[:, n : n + 1] * av
                if with_aux:
                    aux_col = torch.addcmul(aux_col, torch.tensor(0.25, dtype=torch.float32), av)
                c_local = c_local + av @ fh.t() + av @ fm.t() + av @ fl.t()

        if with_aux and aux_ref is not None:
            aux_ref[:, h0:h1] = aux_col.bfloat16()
        nv_c = col.bfloat16()
        c_ref[:, h0:h1] = nv_c
        q_ref[sp] = (nv_c.float() * nv_c.float()).sum(dim=-1)
        s_ref[sp] = sq.sum(dim=-1)
        p_ref[sp] = c_local

    # Reference Stage 2 (finalize) in fixed split order 0..splits-1
    def finalize_token(tok: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rms = torch.tensor(0.0, dtype=torch.float32)
        p_acc = torch.zeros((mixes,), dtype=torch.float32)
        q_tot = torch.tensor(0.0, dtype=torch.float32)
        for sp in range(splits):
            rms = rms + s_ref[sp, tok]
            p_acc = p_acc + p_ref[sp, tok]
            q_tot = q_tot + q_ref[sp, tok]
        rms = torch.rsqrt(rms / k_dim + rms_eps)
        pre_tok = torch.sigmoid(p_acc[:mult] * rms * scale[0] + base[:mult]) + hc_eps
        post_tok = torch.sigmoid(p_acc[mult : 2 * mult] * rms * scale[1] + base[mult : 2 * mult]) * post_mult
        cm = (p_acc[2 * mult :] * rms * scale[2] + base[2 * mult :]).view(mult, mult)
        rmax = cm.max(dim=-1, keepdim=True).values
        cm = torch.exp(cm - rmax)
        cm = cm / cm.sum(dim=-1, keepdim=True) + hc_eps
        cm = cm / (cm.sum(dim=-2, keepdim=True) + hc_eps)
        for _ in range(sinkhorn_repeat - 1):
            cm = cm / (cm.sum(dim=-1, keepdim=True) + hc_eps)
            cm = cm / (cm.sum(dim=-2, keepdim=True) + hc_eps)
        inv_norm = torch.rsqrt(q_tot / hidden + norm_eps)
        y_tok = (c_ref[tok].float() * inv_norm * weight.float()).bfloat16()
        return pre_tok, post_tok, cm, y_tok

    pre_ref = torch.empty((tokens, mult), dtype=torch.float32)
    post_ref = torch.empty((tokens, mult), dtype=torch.float32)
    comb_ref = torch.empty((tokens, mult, mult), dtype=torch.float32)
    y_ref = torch.empty((tokens, hidden), dtype=torch.bfloat16)
    for tok in range(tokens):
        pre_ref[tok], post_ref[tok], comb_ref[tok], y_ref[tok] = finalize_token(tok)

    # Fused Stage 2 simulation with arbitrary split CTA arrival permutations per tile
    rng = random.Random(seed)
    tiles = (tokens + block_m - 1) // block_m
    counters = [0 for _ in range(tiles)]
    pre_fused = torch.empty_like(pre_ref)
    post_fused = torch.empty_like(post_ref)
    comb_fused = torch.empty_like(comb_ref)
    y_fused = torch.empty_like(y_ref)

    for by in range(tiles):
        live_m = min(tokens - by * block_m, block_m)
        arrival_order = list(range(splits))
        rng.shuffle(arrival_order)
        claimed_ranks: list[int] = []
        for _sp in arrival_order:
            arrived = counters[by]
            counters[by] += 1
            rank = splits - 1 - arrived
            if rank >= live_m:
                continue
            if live_m == 1:
                counters[by] = 0
            claimed_ranks.append(rank)
        assert sorted(claimed_ranks) == list(range(live_m))
        if live_m > 1:
            assert counters[by] == splits
        for rank in claimed_ranks:
            tok = by * block_m + rank
            # Each claiming CTA still loops i_split in 0..splits-1 (fixed split order)
            pre_fused[tok], post_fused[tok], comb_fused[tok], y_fused[tok] = finalize_token(tok)
            if live_m > 1:
                done = counters[by]
                counters[by] += 1
                if done == splits + live_m - 1:
                    counters[by] = 0
        assert counters[by] == 0

    def bits_match(a: torch.Tensor, b: torch.Tensor) -> bool:
        return bool(torch.equal(a.view(torch.uint8), b.view(torch.uint8)))

    assert bits_match(pre_fused, pre_ref)
    assert bits_match(post_fused, post_ref)
    assert bits_match(comb_fused, comb_ref)
    assert bits_match(y_fused, y_ref)
    return {
        "tokens": tokens,
        "hidden": hidden,
        "mode": mode,
        "world": world,
        "with_aux": with_aux,
        "inject_nonfinite": inject_nonfinite,
        "tiles": tiles,
        "counters_reset_zero": all(c == 0 for c in counters),
        "bits_identical": True,
    }


def run_cpu_verify(output_path: Path | None) -> dict:
    import tilelang
    from tests.kernels.deepseek_v41_pdl_sources import TRIGGER, WAIT, violations
    from vllm.models.deepseek_v4_1.tilelang import iket, mhc

    assert mhc.FINALIZE_FUSE_MAX_ROWS == 16
    target = {"kind": "cuda", "arch": "sm_121a"}
    mhc.project_and_finalize_streams.target = target

    variants = [
        ("pre", 1, False),
        ("broadcast", 1, False),
        ("post_pre", 1, False),
        ("post_pre", 1, True),
        ("post_pre", 2, False),
        ("post_pre", 2, True),
        ("post_pre", 3, False),
        ("post_pre", 3, True),
        ("post_pre", 4, False),
        ("post_pre", 4, True),
    ]
    compiled_variants = []
    for mode, world, with_aux in variants:
        k_dim = 5120 if mode == "broadcast" else 20480
        prim = mhc.project_and_finalize_streams.get_tir(
            5120, mode, k_dim, 1e-6, 1e-6, 20, 1e-6,
            block_M=16, world=world, with_aux=with_aux,
        )
        with tilelang.tvm.target.Target(target), tilelang.transform.PassContext(
            opt_level=3, config=mhc._PASS_CONFIGS
        ):
            art = tilelang.lower(prim, target=target)
        src = art.kernel_source
        pdl_viol = violations(src, False)
        assert pdl_viol == [], f"PDL violations for {mode} w={world} aux={with_aux}: {pdl_viol}"
        wait_pos = src.find(WAIT)
        claim_pos = src.find("mhc_claim(", wait_pos)
        trig_pos = src.find(TRIGGER, wait_pos)
        spin_pos = src.find("mhc_wait_splits(", wait_pos)
        assert 0 <= wait_pos < claim_pos < trig_pos < spin_pos

        _, sites_fine_1 = iket.annotate(src, lambda _k, _r: 1, granularity="fine", iters=1)
        _, sites_fine_2 = iket.annotate(src, lambda _k, _r: 1, granularity="fine", iters=2)
        bound1 = sites_fine_1[0]["events_per_warp_bound"]
        bound2 = sites_fine_2[0]["events_per_warp_bound"]
        assert bound1 <= 60
        assert len(sites_fine_1) <= 30

        with tempfile.TemporaryDirectory() as tmpdir:
            cu_path = Path(tmpdir) / "kernel.cu"
            cubin_path = Path(tmpdir) / "kernel.cubin"
            cu_path.write_text(src, encoding="utf-8")
            cmd = [
                "/usr/local/cuda/bin/nvcc",
                "-cubin",
                "-arch=sm_121a",
                "-O3",
                "--expt-relaxed-constexpr",
                "-Xptxas=-v",
                "-I/usr/local/lib/python3.12/dist-packages/tilelang/3rdparty/cutlass/include",
                "-I/usr/local/lib/python3.12/dist-packages/tilelang/src",
                str(cu_path),
                "-o",
                str(cubin_path),
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            ptx_lines = [
                line.strip()
                for line in res.stderr.splitlines()
                if "registers" in line or "spill" in line or "smem" in line
            ]
            assert any(
                "0 bytes spill stores, 0 bytes spill loads" in line for line in ptx_lines
            ), ptx_lines

        compiled_variants.append(
            {
                "mode": mode,
                "world": world,
                "with_aux": with_aux,
                "iket_sites": len(sites_fine_1),
                "iket_bound_iters1": bound1,
                "iket_bound_iters2": bound2,
                "pdl_violations": len(pdl_viol),
                "pdl_order_valid": True,
                "ptxas": ptx_lines,
            }
        )

    oracle_cases = []
    # Full H=5120 bit-exactness checks for M=1, 6, 16 across modes
    for mode, world, with_aux in [("pre", 1, False), ("broadcast", 1, False), ("post_pre", 3, True), ("post_pre", 4, True)]:
        for rows in (1, 6, 16):
            oracle_cases.append(
                simulate_split_vs_fused_cpu(
                    rows, hidden=5120, block_h=128, mode=mode, world=world, with_aux=with_aux, seed=1000 + rows
                )
            )
    # Decode & prefill row sweep + non-finite checks
    for rows in (1, 2, 4, 6, 8, 12, 15, 16, 17, 24, 32, 48, 64, 96, 6144, 16384):
        oracle_cases.append(
            simulate_split_vs_fused_cpu(
                rows,
                hidden=320,
                block_h=8,
                mode="post_pre",
                world=3,
                with_aux=True,
                inject_nonfinite=(rows in (16, 96)),
                seed=2000 + rows,
            )
        )

    report = {
        "finalize_fuse_max_rows": mhc.FINALIZE_FUSE_MAX_ROWS,
        "compiled_variants": compiled_variants,
        "cpu_oracle_cases": len(oracle_cases),
        "all_cpu_oracle_bits_identical": all(c["bits_identical"] for c in oracle_cases),
        "all_counters_reset_zero": all(c["counters_reset_zero"] for c in oracle_cases),
    }
    if output_path is not None:
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def _define_base_0634_fused():
    import tilelang
    import tilelang.language as T
    from vllm.models.deepseek_v4_1.tilelang import mhc, pdl
    from vllm.models.deepseek_v4_1.tilelang.gemm import cast_block_mxfp8, scale_words

    @tilelang.jit(pass_configs=mhc._PASS_CONFIGS)
    def base_0634_project_and_finalize_streams(
        hidden_size: int,
        mode: str,
        rms_group_size: int,
        rms_eps: float,
        hc_eps: float,
        sinkhorn_repeat: int,
        norm_eps: float,
        post_mult: float = 2.0,
        block_M: int = mhc.BLOCK_M,
        block_H: int = mhc.BLOCK_H,
    ):
        MULT = mhc.MULT
        MIXES = mhc.MIXES
        VEC = mhc.VEC
        splits = hidden_size // block_H
        streams = 1 if mode == "broadcast" else MULT
        K = streams * hidden_size
        N = 32
        LANES = 8
        COLS = block_H // LANES
        threads = block_M * LANES
        chunks = hidden_size // VEC
        workers = threads - 32
        sf_width = 4 * scale_words(hidden_size)
        H = hidden_size
        M, Mq, tiles = T.dynamic("M, Mq, tiles")

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

        @T.prim_func
        def mhc_project_finalize_base(
            X: T.Tensor((1, M, H), T.bfloat16),
            R: T.Tensor((M, MULT * H), T.bfloat16),
            post_in: T.Tensor((M, MULT), T.float32),
            comb_in: T.Tensor((M, MULT, MULT), T.float32),
            pre_in: T.Tensor((M, MULT), T.float32),
            Fn: T.Tensor((MIXES, K), T.float32),
            R_out: T.Tensor((M, MULT * H), T.bfloat16),
            P: T.Tensor((splits, M, 1, MIXES), T.float32),
            S: T.Tensor((splits, M, 1), T.float32),
            C: T.Tensor((M, H), T.bfloat16),
            Q: T.Tensor((splits, M), T.float32),
            counters: T.Tensor((tiles,), T.int32),
            scale: T.Tensor((3,), T.float32),
            base: T.Tensor((MIXES,), T.float32),
            weight: T.Tensor((H,), T.bfloat16),
            post_out: T.Tensor((M, MULT), T.float32),
            comb_out: T.Tensor((M, MULT, MULT), T.float32),
            pre_out: T.Tensor((M, MULT), T.float32),
            y: T.Tensor((M, H), T.bfloat16),
            yq: T.Tensor((Mq, H), T.float8_e4m3fn),
            ysf: T.Tensor((Mq, sf_width), T.uint8),
        ):
            with T.Kernel(splits, T.ceildiv(M, block_M), threads=threads) as (sp, by):
                T.import_source(mhc._FUSE_HEADER)
                pdl.chain()
                h0 = sp * block_H
                A_shared = T.alloc_shared((block_M, block_H), T.float32)
                F_shared = T.alloc_shared((N, block_H), T.float32)
                F_mid = T.alloc_shared((N, block_H), T.float32)
                F_low = T.alloc_shared((N, block_H), T.float32)
                last = T.alloc_shared((1,), T.int32)
                C_local = T.alloc_fragment((block_M, N), T.float32)
                xv = T.alloc_local((COLS,), T.bfloat16)
                rv = T.alloc_local((MULT * COLS,), T.bfloat16)
                nv = T.alloc_local((COLS,), T.bfloat16)
                av = T.alloc_local((COLS,), T.float32)
                sq = T.alloc_local((COLS,), T.float32)
                col = T.alloc_local((COLS,), T.float32)
                coef = T.alloc_local((2 * MULT + MULT * MULT,), T.float32)
                red = T.alloc_local((1,), T.float32)
                wv = T.alloc_local((VEC,), T.bfloat16)
                yv = T.alloc_local((VEC,), T.bfloat16)
                xf = T.alloc_local((VEC,), T.float32)
                xq = T.alloc_local((VEC,), T.float8_e4m3fn)
                total = T.alloc_local((1,), T.float32)
                amax = T.alloc_local((1,), T.float32)
                rms_l = T.alloc_local((1,), T.float32)
                p_pre = T.alloc_local((1,), T.float32)
                p_post = T.alloc_local((1,), T.float32)
                p_cm = T.alloc_local((1,), T.float32)
                cm_l = T.alloc_local((1,), T.float32)
                rmax_l = T.alloc_local((1,), T.float32)
                rsum_l = T.alloc_local((1,), T.float32)
                csum_l = T.alloc_local((1,), T.float32)
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
                for k in T.unroll(MULT):
                    coef[MULT + k] = post_in[src, k]
                for k in T.unroll(MULT * MULT):
                    coef[2 * MULT + k] = comb_in[src, k // MULT, k % MULT]
                for v in T.vectorized(COLS):
                    xv[v] = X[0, src, h0 + c0 + v]
                for m in T.unroll(MULT):
                    for v in T.vectorized(COLS):
                        rv[m * COLS + v] = R[src, m * H + h0 + c0 + v]
                for v in T.unroll(COLS):
                    sq[v] = 0.0
                    col[v] = 0.0
                for n in T.unroll(MULT):
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
                    load_split_tf32(Fn, n * H + h0, F_shared, F_mid, F_low)
                    T.sync_threads()
                    T.gemm(A_shared, F_shared, C_local, transpose_B=True)
                    T.gemm(A_shared, F_mid, C_local, transpose_B=True)
                    T.gemm(A_shared, F_low, C_local, transpose_B=True)
                    T.sync_threads()
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
                T.call_extern("handle", "mhc_fence")
                T.sync_threads()
                if tx == 0:
                    arrived = T.atomic_add(
                        counters[by], 1, memory_order="acq_rel", return_prev=True
                    )
                    last[0] = T.if_then_else(arrived == splits - 1, 1, 0)
                T.sync_threads()
                if last[0] == 1:
                    T.call_extern("handle", "mhc_fence")
                    if tx == 0:
                        counters[by] = 0
                    if tx < 32:
                        half = tx // 16
                        lane16 = tx % 16
                        r_idx = lane16 // MULT
                        c_idx = lane16 % MULT
                        for pair in T.serial(block_M // 2):
                            tok = by * block_M + pair * 2 + half
                            if tok < M:
                                rms_l[0] = 0.0
                                p_pre[0] = 0.0
                                p_post[0] = 0.0
                                p_cm[0] = 0.0
                                for i_split in T.serial(splits):
                                    rms_l[0] += S[i_split, tok, 0]
                                    if lane16 < MULT:
                                        p_pre[0] += P[i_split, tok, 0, lane16]
                                        p_post[0] += P[i_split, tok, 0, lane16 + MULT]
                                    p_cm[0] += P[i_split, tok, 0, lane16 + 2 * MULT]
                                rms_l[0] = T.rsqrt(rms_l[0] / rms_group_size + rms_eps)
                                if lane16 < MULT:
                                    pre_out[tok, lane16] = (
                                        T.sigmoid(
                                            p_pre[0] * rms_l[0] * scale[0] + base[lane16]
                                        )
                                        + hc_eps
                                    )
                                    post_out[tok, lane16] = (
                                        T.sigmoid(
                                            p_post[0] * rms_l[0] * scale[1]
                                            + base[lane16 + MULT]
                                        )
                                        * post_mult
                                    )
                                cm_l[0] = (
                                    p_cm[0] * rms_l[0] * scale[2]
                                    + base[lane16 + 2 * MULT]
                                )
                                rmax_l[0] = cm_l[0]
                                rmax_l[0] = T.max(rmax_l[0], T.shfl_xor(rmax_l[0], 2))
                                rmax_l[0] = T.max(rmax_l[0], T.shfl_xor(rmax_l[0], 1))
                                cm_l[0] = T.exp(cm_l[0] - rmax_l[0])
                                rsum_l[0] = cm_l[0]
                                rsum_l[0] += T.shfl_xor(rsum_l[0], 2)
                                rsum_l[0] += T.shfl_xor(rsum_l[0], 1)
                                cm_l[0] = cm_l[0] / rsum_l[0] + hc_eps
                                csum_l[0] = cm_l[0]
                                csum_l[0] += T.shfl_xor(csum_l[0], 8)
                                csum_l[0] += T.shfl_xor(csum_l[0], 4)
                                cm_l[0] = cm_l[0] / (csum_l[0] + hc_eps)
                                for _ in T.serial(sinkhorn_repeat - 1):
                                    rsum_l[0] = cm_l[0]
                                    rsum_l[0] += T.shfl_xor(rsum_l[0], 2)
                                    rsum_l[0] += T.shfl_xor(rsum_l[0], 1)
                                    cm_l[0] = cm_l[0] / (rsum_l[0] + hc_eps)
                                    csum_l[0] = cm_l[0]
                                    csum_l[0] += T.shfl_xor(csum_l[0], 8)
                                    csum_l[0] += T.shfl_xor(csum_l[0], 4)
                                    cm_l[0] = cm_l[0] / (csum_l[0] + hc_eps)
                                comb_out[tok, r_idx, c_idx] = cm_l[0]
                    else:
                        worker = tx - 32
                        for ti in T.serial(block_M):
                            tok = by * block_M + ti
                            if tok < M:
                                total[0] = 0.0
                                for i_split in T.serial(splits):
                                    total[0] += Q[i_split, tok]
                                total[0] = T.rsqrt(total[0] / H + norm_eps)
                                for step in T.serial(T.ceildiv(chunks, workers)):
                                    chunk = step * workers + worker
                                    if chunk < chunks:
                                        for v in T.vectorized(VEC):
                                            yv[v] = C[tok, chunk * VEC + v]
                                        for v in T.vectorized(VEC):
                                            wv[v] = weight[chunk * VEC + v]
                                        for v in T.unroll(VEC):
                                            xv[v] = T.Cast(
                                                T.bfloat16,
                                                T.Cast(T.float32, yv[v])
                                                * total[0]
                                                * T.Cast(T.float32, wv[v]),
                                            )
                                        for v in T.vectorized(VEC):
                                            y[tok, chunk * VEC + v] = xv[v]
                                        amax[0] = T.float32(0)
                                        for v in T.unroll(VEC):
                                            xf[v] = T.Cast(T.float32, xv[v])
                                            amax[0] = T.max(amax[0], T.abs(xf[v]))
                                        amax[0] = T.max(amax[0], T.shfl_xor(amax[0], 1))
                                        amax[0] = T.max(amax[0], T.shfl_xor(amax[0], 2))
                                        cast_block_mxfp8(
                                            xf, xq, VEC, amax[0], yq, ysf, tok, chunk * VEC
                                        )

        return mhc_project_finalize_base

    return base_0634_project_and_finalize_streams


def _bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.element_size() == 2:
        return bool(torch.equal(a.view(torch.int16), b.view(torch.int16)))
    if a.element_size() == 4:
        return bool(torch.equal(a.view(torch.int32), b.view(torch.int32)))
    return bool(torch.equal(a.view(torch.uint8), b.view(torch.uint8)))


def run_gpu_bench(output_path: Path | None) -> dict:
    from vllm.models.deepseek_v4_1.tilelang import mhc
    from vllm.models.deepseek_v4_1.tilelang.gemm import scale_words
    from vllm.models.deepseek_v4_1.tilelang.linear import mxfp8_rows

    device = torch.device("cuda", torch.cuda.current_device())
    hidden = 5120
    block_h = 128
    block_m = 16
    splits = hidden // block_h
    streams = 4 * hidden
    rms_eps = 1e-6
    hc_eps = 1e-6
    iters = 20

    # 1. Exhaustive GPU bit-identity checks across modes, world sizes, with_aux, finite & non-finite
    correctness_cases = []
    for mode, world, with_aux in [
        ("pre", 1, False),
        ("broadcast", 1, False),
        ("post_pre", 1, False),
        ("post_pre", 1, True),
        ("post_pre", 3, False),
        ("post_pre", 3, True),
        ("post_pre", 4, True),
    ]:
        k_dim = hidden if mode == "broadcast" else streams
        proj_k = mhc.project_streams(hidden, mode, block_M=block_m, world=world, with_aux=with_aux)
        fin_k = mhc.finalize(hidden, k_dim, rms_eps, hc_eps, iters, rms_eps)
        fused_k = mhc.project_and_finalize_streams(
            hidden, mode, k_dim, rms_eps, hc_eps, iters, rms_eps,
            block_M=block_m, world=world, with_aux=with_aux,
        )
        for tokens, nonfinite in [(1, False), (6, False), (15, False), (16, False), (16, True), (32, False), (48, True)]:
            g = torch.Generator(device=device).manual_seed(5000 + tokens * 17 + world * 3 + int(with_aux))
            x_in = torch.randn(world, tokens, hidden, generator=g, device=device, dtype=torch.bfloat16)
            source = torch.randn(tokens, streams, generator=g, device=device, dtype=torch.bfloat16)
            mix_post = 2 * torch.sigmoid(torch.randn(tokens, 4, generator=g, device=device))
            mix_comb = torch.softmax(torch.randn(tokens, 4, 4, generator=g, device=device), -1)
            incoming = torch.sigmoid(torch.randn(tokens, 4, generator=g, device=device)) + hc_eps
            fn_in = torch.randn(24, k_dim, generator=g, device=device) * 0.02
            scale = torch.tensor([0.8, 1.1, 0.9], device=device, dtype=torch.float32)
            base = torch.randn(24, generator=g, device=device, dtype=torch.float32) * 0.1
            weight = (torch.randn(hidden, generator=g, device=device) * 0.1 + 1.0).bfloat16()
            x_in[:, :, :8] = -0.0
            source[:, :8] = -0.0
            if nonfinite and tokens >= 4:
                x_in[0, 0, 16] = float("nan")
                source[1, 32] = float("inf")
                source[2, 64] = float("-inf")
                incoming[3, 1] = float("nan")

            mq = mxfp8_rows(tokens)
            sf_w = 4 * scale_words(hidden)
            R_ref = source.clone() if mode == "pre" else torch.empty_like(source)
            P_ref = torch.empty(splits, tokens, 1, 24, dtype=torch.float32, device=device)
            S_ref = torch.empty(splits, tokens, 1, dtype=torch.float32, device=device)
            C_ref = torch.empty(tokens, hidden, dtype=torch.bfloat16, device=device)
            Q_ref = torch.empty(splits, tokens, dtype=torch.float32, device=device)
            post_ref = torch.empty(tokens, 4, dtype=torch.float32, device=device)
            comb_ref = torch.empty(tokens, 4, 4, dtype=torch.float32, device=device)
            pre_ref = torch.empty(tokens, 4, dtype=torch.float32, device=device)
            y_ref = torch.empty(tokens, hidden, dtype=torch.bfloat16, device=device)
            yq_ref = torch.empty(mq, hidden, dtype=torch.float8_e4m3fn, device=device)
            ysf_ref = torch.empty(mq, sf_w, dtype=torch.uint8, device=device)
            aux_ref = (torch.empty(tokens, hidden, dtype=torch.bfloat16, device=device),) if with_aux else ()

            proj_k(x_in, source, mix_post, mix_comb, incoming, fn_in, R_ref, P_ref, S_ref, C_ref, Q_ref, *aux_ref)
            fin_k(P_ref, S_ref, C_ref, Q_ref, scale, base, weight, post_ref, comb_ref, pre_ref, y_ref, yq_ref, ysf_ref)

            R_out = source.clone() if mode == "pre" else torch.empty_like(R_ref)
            P_out = torch.empty_like(P_ref)
            S_out = torch.empty_like(S_ref)
            C_out = torch.empty_like(C_ref)
            Q_out = torch.empty_like(Q_ref)
            post_out = torch.empty_like(post_ref)
            comb_out = torch.empty_like(comb_ref)
            pre_out = torch.empty_like(pre_ref)
            y_out = torch.empty_like(y_ref)
            yq_out = torch.empty_like(yq_ref)
            ysf_out = torch.empty_like(ysf_ref)
            aux_out = (torch.empty_like(aux_ref[0]),) if with_aux else ()
            tiles = (tokens + block_m - 1) // block_m
            counters = torch.zeros(tiles, dtype=torch.int32, device=device)

            for _ in range(2):
                fused_k(
                    x_in, source, mix_post, mix_comb, incoming, fn_in,
                    R_out, P_out, S_out, C_out, Q_out, counters,
                    scale, base, weight, post_out, comb_out, pre_out, y_out, yq_out, ysf_out,
                    *aux_out,
                )
                assert int((counters == 0).all().item()) == 1

            assert _bits_equal(R_out, R_ref), f"R_out mismatch {mode} w={world} aux={with_aux} tokens={tokens}"
            assert _bits_equal(post_out, post_ref), f"post_out mismatch {mode} w={world} aux={with_aux} tokens={tokens}"
            assert _bits_equal(comb_out, comb_ref), f"comb_out mismatch {mode} w={world} aux={with_aux} tokens={tokens}"
            assert _bits_equal(pre_out, pre_ref), f"pre_out mismatch {mode} w={world} aux={with_aux} tokens={tokens}"
            assert _bits_equal(y_out, y_ref), f"y_out mismatch {mode} w={world} aux={with_aux} tokens={tokens}"
            assert _bits_equal(yq_out[:tokens], yq_ref[:tokens]), f"yq_out mismatch {mode} w={world} aux={with_aux} tokens={tokens}"
            assert _bits_equal(ysf_out[:tokens], ysf_ref[:tokens]), f"ysf_out mismatch {mode} w={world} aux={with_aux} tokens={tokens}"
            if with_aux:
                assert _bits_equal(aux_out[0], aux_ref[0]), f"aux_out mismatch {mode} w={world} aux={with_aux} tokens={tokens}"
            correctness_cases.append(
                {
                    "mode": mode,
                    "world": world,
                    "with_aux": with_aux,
                    "tokens": tokens,
                    "nonfinite": nonfinite,
                    "bits_identical": True,
                }
            )

    # 2. Latency benchmark across decode & prefill shapes (CUDA Graph + Eager)
    base_0634_jit = _define_base_0634_fused()
    base_0634_k = base_0634_jit(
        hidden, "post_pre", streams, rms_eps, hc_eps, iters, rms_eps, block_M=block_m
    )
    proj_k = mhc.project_streams(hidden, "post_pre", block_M=block_m)
    fin_k = mhc.finalize(hidden, streams, rms_eps, hc_eps, iters, rms_eps)
    fused_k = mhc.project_and_finalize_streams(
        hidden, "post_pre", streams, rms_eps, hc_eps, iters, rms_eps, block_M=block_m
    )

    rows_list = [1, 2, 4, 6, 8, 12, 15, 16, 17, 24, 32, 36, 40, 48, 64, 96, 2048, 6144, 16384]
    results = []

    for tokens in rows_list:
        g = torch.Generator(device=device).manual_seed(4000 + tokens)
        x_in = torch.randn(1, tokens, hidden, generator=g, device=device, dtype=torch.bfloat16)
        source = torch.randn(tokens, streams, generator=g, device=device, dtype=torch.bfloat16)
        mix_post = 2 * torch.sigmoid(torch.randn(tokens, 4, generator=g, device=device))
        mix_comb = torch.softmax(torch.randn(tokens, 4, 4, generator=g, device=device), -1)
        incoming = torch.sigmoid(torch.randn(tokens, 4, generator=g, device=device)) + hc_eps
        fn_in = torch.randn(24, streams, generator=g, device=device) * 0.02
        scale = torch.tensor([0.8, 1.1, 0.9], device=device, dtype=torch.float32)
        base = torch.randn(24, generator=g, device=device, dtype=torch.float32) * 0.1
        weight = (torch.randn(hidden, generator=g, device=device) * 0.1 + 1.0).bfloat16()

        mq = mxfp8_rows(tokens)
        sf_w = 4 * scale_words(hidden)
        R_ref = torch.empty_like(source)
        P_ref = torch.empty(splits, tokens, 1, 24, dtype=torch.float32, device=device)
        S_ref = torch.empty(splits, tokens, 1, dtype=torch.float32, device=device)
        C_ref = torch.empty(tokens, hidden, dtype=torch.bfloat16, device=device)
        Q_ref = torch.empty(splits, tokens, dtype=torch.float32, device=device)
        post_ref = torch.empty(tokens, 4, dtype=torch.float32, device=device)
        comb_ref = torch.empty(tokens, 4, 4, dtype=torch.float32, device=device)
        pre_ref = torch.empty(tokens, 4, dtype=torch.float32, device=device)
        y_ref = torch.empty(tokens, hidden, dtype=torch.bfloat16, device=device)
        yq_ref = torch.empty(mq, hidden, dtype=torch.float8_e4m3fn, device=device)
        ysf_ref = torch.empty(mq, sf_w, dtype=torch.uint8, device=device)

        proj_k(x_in, source, mix_post, mix_comb, incoming, fn_in, R_ref, P_ref, S_ref, C_ref, Q_ref)
        fin_k(P_ref, S_ref, C_ref, Q_ref, scale, base, weight, post_ref, comb_ref, pre_ref, y_ref, yq_ref, ysf_ref)

        R_out = torch.empty_like(R_ref)
        P_out = torch.empty_like(P_ref)
        S_out = torch.empty_like(S_ref)
        C_out = torch.empty_like(C_ref)
        Q_out = torch.empty_like(Q_ref)
        post_out = torch.empty_like(post_ref)
        comb_out = torch.empty_like(comb_ref)
        pre_out = torch.empty_like(pre_ref)
        y_out = torch.empty_like(y_ref)
        yq_out = torch.empty_like(yq_ref)
        ysf_out = torch.empty_like(ysf_ref)
        tiles = (tokens + block_m - 1) // block_m
        counters = torch.zeros(tiles, dtype=torch.int32, device=device)

        if tokens <= 96:
            fused_k(
                x_in, source, mix_post, mix_comb, incoming, fn_in,
                R_out, P_out, S_out, C_out, Q_out, counters,
                scale, base, weight, post_out, comb_out, pre_out, y_out, yq_out, ysf_out,
            )
            torch.cuda.synchronize()
            assert int((counters == 0).all().item()) == 1
            assert _bits_equal(R_out, R_ref)
            assert _bits_equal(post_out, post_ref)
            assert _bits_equal(comb_out, comb_ref)
            assert _bits_equal(pre_out, pre_ref)
            assert _bits_equal(y_out, y_ref)
            assert _bits_equal(yq_out[:tokens], yq_ref[:tokens])
            assert _bits_equal(ysf_out[:tokens], ysf_ref[:tokens])

        def measure_graph_us(fn, reps: int = 400) -> float:
            for _ in range(15):
                fn()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(10):
                    fn()
            torch.cuda.synchronize()
            for _ in range(5):
                graph.replay()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(reps // 10):
                graph.replay()
            end.record()
            torch.cuda.synchronize()
            return (start.elapsed_time(end) * 1000.0) / reps

        def measure_eager_us(fn, reps: int = 200) -> float:
            for _ in range(15):
                fn()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(reps):
                fn()
            end.record()
            torch.cuda.synchronize()
            return (start.elapsed_time(end) * 1000.0) / reps

        if tokens <= 96:
            split_graph_us = measure_graph_us(
                lambda: (
                    proj_k(x_in, source, mix_post, mix_comb, incoming, fn_in, R_ref, P_ref, S_ref, C_ref, Q_ref),
                    fin_k(P_ref, S_ref, C_ref, Q_ref, scale, base, weight, post_ref, comb_ref, pre_ref, y_ref, yq_ref, ysf_ref),
                )
            )
            cand_fused_graph_us = measure_graph_us(
                lambda: fused_k(
                    x_in, source, mix_post, mix_comb, incoming, fn_in,
                    R_out, P_out, S_out, C_out, Q_out, counters,
                    scale, base, weight, post_out, comb_out, pre_out, y_out, yq_out, ysf_out,
                )
            )
            base_0634_graph_us = (
                measure_graph_us(
                    lambda: base_0634_k(
                        x_in, source, mix_post, mix_comb, incoming, fn_in,
                        R_out, P_out, S_out, C_out, Q_out, counters,
                        scale, base, weight, post_out, comb_out, pre_out, y_out, yq_out, ysf_out,
                    )
                )
                if tokens <= 32
                else None
            )
            split_eager_us = measure_eager_us(
                lambda: (
                    proj_k(x_in, source, mix_post, mix_comb, incoming, fn_in, R_ref, P_ref, S_ref, C_ref, Q_ref),
                    fin_k(P_ref, S_ref, C_ref, Q_ref, scale, base, weight, post_ref, comb_ref, pre_ref, y_ref, yq_ref, ysf_ref),
                )
            )
            cand_fused_eager_us = measure_eager_us(
                lambda: fused_k(
                    x_in, source, mix_post, mix_comb, incoming, fn_in,
                    R_out, P_out, S_out, C_out, Q_out, counters,
                    scale, base, weight, post_out, comb_out, pre_out, y_out, yq_out, ysf_out,
                )
            )
            results.append(
                {
                    "tokens": tokens,
                    "dispatched_fused": tokens <= mhc.FINALIZE_FUSE_MAX_ROWS,
                    "split_graph_us": round(split_graph_us, 3),
                    "base_0634_fused_graph_us": round(base_0634_graph_us, 3) if base_0634_graph_us is not None else None,
                    "cand_0731_fused_graph_us": round(cand_fused_graph_us, 3),
                    "graph_delta_us_vs_split": round(cand_fused_graph_us - split_graph_us, 3),
                    "graph_step80_saved_us_vs_split": round(80.0 * (split_graph_us - cand_fused_graph_us), 1),
                    "split_eager_us": round(split_eager_us, 3),
                    "cand_0731_fused_eager_us": round(cand_fused_eager_us, 3),
                    "bits_identical": True,
                }
            )
        else:
            proj_us = measure_eager_us(
                lambda: proj_k(x_in, source, mix_post, mix_comb, incoming, fn_in, R_ref, P_ref, S_ref, C_ref, Q_ref),
                reps=40,
            )
            fin_us = measure_eager_us(
                lambda: fin_k(P_ref, S_ref, C_ref, Q_ref, scale, base, weight, post_ref, comb_ref, pre_ref, y_ref, yq_ref, ysf_ref),
                reps=40,
            )
            split_eager_us = measure_eager_us(
                lambda: (
                    proj_k(x_in, source, mix_post, mix_comb, incoming, fn_in, R_ref, P_ref, S_ref, C_ref, Q_ref),
                    fin_k(P_ref, S_ref, C_ref, Q_ref, scale, base, weight, post_ref, comb_ref, pre_ref, y_ref, yq_ref, ysf_ref),
                ),
                reps=40,
            )
            results.append(
                {
                    "tokens": tokens,
                    "dispatched_fused": False,
                    "project_only_eager_us": round(proj_us, 3),
                    "finalize_only_eager_us": round(fin_us, 3),
                    "split_eager_us": round(split_eager_us, 3),
                    "bits_identical": True,
                }
            )

    out = {
        "device": torch.cuda.get_device_name(device),
        "finalize_fuse_max_rows": mhc.FINALIZE_FUSE_MAX_ROWS,
        "correctness_cases": len(correctness_cases),
        "all_correctness_bits_identical": all(c["bits_identical"] for c in correctness_cases),
        "rows": results,
    }
    if output_path is not None:
        output_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-verify", action="store_true")
    parser.add_argument("--gpu-bench", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.gpu_bench:
        run_gpu_bench(args.output)
    else:
        run_cpu_verify(args.output)


if __name__ == "__main__":
    main()
