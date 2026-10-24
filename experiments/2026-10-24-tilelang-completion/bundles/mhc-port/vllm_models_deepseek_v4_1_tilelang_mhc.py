# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 lagged mHC on TileKernels (``--linear-backend tilelang``).

V4.1's hyper-connections keep four residual streams. Each sublayer's ``pre``
projects the flattened streams through ``fn`` (RMS-normalized), splits the 24
mixes into a pre-mix (sigmoid), a post-mix (2 * sigmoid) and a 4 x 4
combination (softmax, then Sinkhorn), and feeds the sublayer the streams
collapsed with the *incoming* pre-mix, RMS-normalized with the norm weight.
The new pre-mix is returned for the next sublayer (the lag). ``post`` folds a
sublayer's output back into the streams.

TileKernels provides DeepSeek's mHC kernels. A standalone ``post`` (where no
sublayer entry follows to fuse it: Engram layers, GLP steering, the last layer)
and the stream collapse (the head's input, the drafter's auxiliary states) are
TileLang ports of TileKernels' ``mhc_post_fwd`` and ``mhc_pre_apply_mix_fwd``
with their arithmetic to the bit:

- ``post_streams``: ``bf16(post[n] * x + sum_m comb[m, n] * streams[m])``;
- ``collapse_streams``: ``bf16(sum_m mix[m] * streams[m])``.

A sublayer's entry needs more: DeepSeek moved the projection GEMM to DeepGEMM,
which does not run on SM120/SM121, TileKernels' fused pre step applies the mix
it has just computed rather than the lagged one, and in prefill the four streams
are large enough that reading them once matters. Two TileLang kernels make each
entry:

- ``project_streams``: per hidden block, the previous sublayer's ``post``
  (writing the updated streams), the projection's split partials on TF32
  tensor cores at FP32 precision (fn split into exact TF32 parts) with the
  streams' sum of squares, and
  the streams collapsed with the incoming pre-mix in BF16, with its sum of
  squares. Each thread folds and accumulates one token's columns in
  registers; the GEMM reads the folded streams from shared memory. The split
  is fixed and every token tile accumulates in the same order, so a token's
  results do not depend on the batch.
- ``finalize``: per token, TileKernels' fused reduce, RMS, mix split and
  Sinkhorn (``mhc_pre_big_fuse``), emitting the new pre-mix, beside the
  collapse's RMSNorm with the sublayer's weight.
"""

from functools import cache

import tilelang
import tilelang.language as T
import torch
from torch import nn

from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.models.deepseek_v4_1 import sp_prefill
from vllm.models.deepseek_v4_1.tilelang import prepare
from vllm.models.deepseek_v4_1.tilelang.collectives import RankOrderParts
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    retain_cuda_graph_capture_resource,
)

MULT = 4  # residual streams
MIXES = (2 + MULT) * MULT  # pre, post and combination mixes
BLOCK_H = 128  # hidden columns per projection split, fixed for batch invariance
BLOCK_M = 16  # token tile of project_streams; larger tiles measured slower
VEC = 8  # BF16 elements per vector access

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_VECTORIZE_256: True,
}


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def project_streams(
    hidden_size: int,
    mode: str,
    block_M: int = 16,
    block_H: int = BLOCK_H,
    world: int = 1,
):
    """The streams' projection partials and their collapse, for one hidden block.

    ``mode`` is the sublayer's entry: ``"post_pre"`` first folds the previous
    sublayer's output into the streams (``post``) and writes them, ``"pre"``
    reads streams already updated, and ``"broadcast"`` expands an embedding to
    four copies (the first layer, whose ``fn`` is the stream blocks summed).
    ``X`` is ``(world, M, H)``: with ``world`` > 1 (``"post_pre"`` only) the
    previous output arrives as the ranks' unreduced partials, which each thread
    adds in FP32 in rank order and rounds once, as the rank-order reduce-scatter
    does, so the output has the same bits as that sum read back.

    Each CTA owns ``block_H`` hidden columns of every stream, a K split of the
    projection, for ``block_M`` tokens. Each thread owns one token's
    ``block_H / 8`` contiguous columns: it loads them for every stream into
    registers, folds the previous output in, writes the updated streams, and
    keeps its share of the streams' sum of squares ``S`` and of the collapse
    with the incoming pre-mix. The projection ``P = streams @ fn^T`` runs on
    TF32 tensor cores at FP32 precision, one stream at a time: the streams are
    BF16 (exact in TF32) and each FP32 ``fn`` value is split exactly into three
    TF32-representable parts, so every product is exact and the three GEMMs
    accumulate in FP32, as DeepSeek's reference computes the mixes. The
    collapse is rounded to BF16, with its sum of squares ``Q``.
    """
    assert hidden_size % block_H == 0 and mode in ("pre", "post_pre", "broadcast")
    assert 1 <= world <= 4 and (world == 1 or mode == "post_pre")
    splits = hidden_size // block_H
    streams = 1 if mode == "broadcast" else MULT
    K = streams * hidden_size
    N = 32  # the 24 mixes, padded
    LANES = 8  # threads per token: adjacent lanes of one warp
    COLS = block_H // LANES  # contiguous columns per thread
    threads = block_M * LANES
    H = hidden_size
    M = T.dynamic("M")

    @T.macro
    def load_split_tf32(Fn, k0, F_shared, F_mid, F_low):
        # fn = high + middle + low, each exact in TF32 (11 significant bits): the
        # high part keeps fn's top 11 bits, the middle the next 11, the low the rest.
        # Each value goes from global memory through registers into the three tiles
        # in one pass.
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
    def mhc_project_streams(
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
            h0 = sp * block_H
            A_shared = T.alloc_shared((block_M, block_H), T.float32)
            F_shared = T.alloc_shared((N, block_H), T.float32)
            F_mid = T.alloc_shared((N, block_H), T.float32)
            F_low = T.alloc_shared((N, block_H), T.float32)
            C_local = T.alloc_fragment((block_M, N), T.float32)
            xv = T.alloc_local((COLS,), T.bfloat16)
            rv = T.alloc_local((MULT * COLS,), T.bfloat16)
            nv = T.alloc_local((COLS,), T.bfloat16)
            av = T.alloc_local((COLS,), T.float32)
            sq = T.alloc_local((COLS,), T.float32)
            col = T.alloc_local((COLS,), T.float32)
            coef = T.alloc_local((2 * MULT + MULT * MULT,), T.float32)
            red = T.alloc_local((1,), T.float32)
            tx = T.get_thread_binding()
            row = tx // LANES
            c0 = (tx % LANES) * COLS
            token = by * block_M + row
            src = T.min(token, M - 1)
            T.clear(C_local)
            # The padding mixes stay zero; each stream's copy fills the rest.
            for i, j in T.Parallel(N - MIXES, block_H):
                F_shared[MIXES + i, j] = 0.0
                F_mid[MIXES + i, j] = 0.0
                F_low[MIXES + i, j] = 0.0
            # This thread's token: its mixes, then its columns of x and the streams.
            for k in T.unroll(MULT):
                coef[k] = pre_in[src, k]
            if mode == "post_pre":
                for k in T.unroll(MULT):
                    coef[MULT + k] = post[src, k]
                for k in T.unroll(MULT * MULT):
                    coef[2 * MULT + k] = comb[src, k // MULT, k % MULT]
            if mode != "pre" and world == 1:
                for v in T.vectorized(COLS):
                    xv[v] = X[0, src, h0 + c0 + v]
            if world > 1:
                # The first part as is (signed zeros survive), then the others.
                for r in T.unroll(world):
                    for v in T.vectorized(COLS):
                        xv[v] = X[r, src, h0 + c0 + v]
                    for v in T.unroll(COLS):
                        if r == 0:
                            av[v] = T.Cast(T.float32, xv[v])
                        else:
                            av[v] = av[v] + T.Cast(T.float32, xv[v])
                for v in T.unroll(COLS):
                    xv[v] = T.Cast(T.bfloat16, av[v])
            if mode != "broadcast":
                for m in T.unroll(MULT):
                    for v in T.vectorized(COLS):
                        rv[m * COLS + v] = R[src, m * H + h0 + c0 + v]
            for v in T.unroll(COLS):
                sq[v] = 0.0
                col[v] = 0.0

            if mode == "broadcast":
                # Four copies of the embedding; fn already sums the blocks.
                for v in T.unroll(COLS):
                    av[v] = T.Cast(T.float32, xv[v])
                    A_shared[row, c0 + v] = av[v]
                    sq[v] = av[v] * av[v]
                for n in T.unroll(MULT):
                    if token < M:
                        for v in T.vectorized(COLS):
                            R_out[token, n * H + h0 + c0 + v] = xv[v]
                    for v in T.unroll(COLS):
                        col[v] += coef[n] * av[v]
                load_split_tf32(Fn, h0, F_shared, F_mid, F_low)
                T.sync_threads()
                T.gemm(A_shared, F_shared, C_local, transpose_B=True)
                T.gemm(A_shared, F_mid, C_local, transpose_B=True)
                T.gemm(A_shared, F_low, C_local, transpose_B=True)
            else:
                for n in T.unroll(MULT):
                    if mode == "post_pre":
                        # post[n] * x + sum_m comb[m, n] * streams[m] in FP32, with
                        # post_streams' (TileKernels') instruction sequence.
                        for v in T.unroll(COLS):
                            av[v] = coef[2 * MULT + n] * T.Cast(T.float32, rv[v])
                            av[v] = T.fma(
                                coef[MULT + n], T.Cast(T.float32, xv[v]), av[v]
                            )
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
                    else:
                        for v in T.unroll(COLS):
                            nv[v] = rv[n * COLS + v]
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

            # The collapse in BF16 and its sum of squares, then the streams'.
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

    return mhc_project_streams


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def finalize(
    hidden_size: int,
    rms_group_size: int,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_repeat: int,
    norm_eps: float,
    post_mult: float = 2.0,
    block_H: int = BLOCK_H,
    threads: int = 128,
):
    """One token per CTA: warp 0 turns the projection partials into the new
    pre-mix, the post-mix and the Sinkhorn combination (TileKernels'
    ``mhc_pre_big_fuse``); the other warps RMS-normalize the collapse."""
    assert hidden_size % VEC == 0
    splits = hidden_size // block_H
    chunks = hidden_size // VEC
    workers = threads - 32
    M = T.dynamic("M")

    @T.prim_func
    def mhc_finalize(
        P: T.Tensor((splits, M, 1, MIXES), T.float32),
        S: T.Tensor((splits, M, 1), T.float32),
        C: T.Tensor((M, hidden_size), T.bfloat16),
        Q: T.Tensor((splits, M), T.float32),
        scale: T.Tensor((3,), T.float32),
        base: T.Tensor((MIXES,), T.float32),
        weight: T.Tensor((hidden_size,), T.bfloat16),
        post: T.Tensor((M, MULT), T.float32),
        comb: T.Tensor((M, MULT, MULT), T.float32),
        pre_out: T.Tensor((M, MULT), T.float32),
        y: T.Tensor((M, hidden_size), T.bfloat16),
    ):
        with T.Kernel(M, threads=threads) as pid:
            tx = T.get_thread_binding()
            mixes_shared = T.alloc_shared((MIXES,), T.float32)
            xv = T.alloc_local((VEC,), T.bfloat16)
            wv = T.alloc_local((VEC,), T.bfloat16)
            yv = T.alloc_local((VEC,), T.bfloat16)
            total = T.alloc_local((1,), T.float32)

            if tx < 32:
                # Reduce the split partials; RMS over the flattened streams.
                rms = T.alloc_fragment(1, T.float32)
                mixes = T.alloc_fragment(MIXES, T.float32)
                rms[0] = 0
                for i_split in T.serial(splits):
                    rms[0] += S[i_split, pid, 0]
                rms[0] = T.rsqrt(rms[0] / rms_group_size + rms_eps)
                for j in T.Parallel(MIXES):
                    mixes[j] = 0
                    for i_split in T.serial(splits):
                        mixes[j] += P[i_split, pid, 0, j]
                    mixes[j] *= rms[0]
                T.copy(mixes, mixes_shared, disable_tma=True)

            if tx < 32:
                # The new pre-mix, the post-mix and the Sinkhorn combination.
                cm = T.alloc_fragment((MULT, MULT), T.float32)
                for j in T.Parallel(MULT):
                    pre_out[pid, j] = (
                        T.sigmoid(mixes_shared[j] * scale[0] + base[j]) + hc_eps
                    )
                    post[pid, j] = (
                        T.sigmoid(mixes_shared[j + MULT] * scale[1] + base[j + MULT])
                        * post_mult
                    )
                for j, k in T.Parallel(MULT, MULT):
                    cm[j, k] = (
                        mixes_shared[j * MULT + k + MULT * 2] * scale[2]
                        + base[j * MULT + k + MULT * 2]
                    )
                row_sum = T.alloc_fragment(MULT, T.float32)
                col_sum = T.alloc_fragment(MULT, T.float32)
                row_max = T.alloc_fragment(MULT, T.float32)
                T.reduce_max(cm, row_max, dim=1)
                for j, k in T.Parallel(MULT, MULT):
                    cm[j, k] = T.exp(cm[j, k] - row_max[j])
                T.reduce_sum(cm, row_sum, dim=1)
                for j, k in T.Parallel(MULT, MULT):
                    cm[j, k] = cm[j, k] / row_sum[j] + hc_eps
                T.reduce_sum(cm, col_sum, dim=0)
                for j, k in T.Parallel(MULT, MULT):
                    cm[j, k] = cm[j, k] / (col_sum[k] + hc_eps)
                for _ in T.serial(sinkhorn_repeat - 1):
                    T.reduce_sum(cm, row_sum, dim=1)
                    for j, k in T.Parallel(MULT, MULT):
                        cm[j, k] = cm[j, k] / (row_sum[j] + hc_eps)
                    T.reduce_sum(cm, col_sum, dim=0)
                    for j, k in T.Parallel(MULT, MULT):
                        cm[j, k] = cm[j, k] / (col_sum[k] + hc_eps)
                for j, k in T.Parallel(MULT, MULT):
                    comb[pid, j, k] = cm[j, k]
            else:
                # RMSNorm of the collapse, with the sublayer's weight.
                worker = tx - 32
                total[0] = 0
                for i_split in T.serial(splits):
                    total[0] += Q[i_split, pid]
                total[0] = T.rsqrt(total[0] / hidden_size + norm_eps)
                for step in T.serial(T.ceildiv(chunks, workers)):
                    chunk = step * workers + worker
                    if chunk < chunks:
                        for v in T.vectorized(VEC):
                            yv[v] = C[pid, chunk * VEC + v]
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
                            y[pid, chunk * VEC + v] = xv[v]

    return mhc_finalize


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def post_streams(hidden_size: int, threads: int = 128):
    """A sublayer's output ``x`` folded into the streams, TileKernels' ``mhc_post_fwd``
    to the bit: ``out[n] = bf16(post[n] * x + sum_m comb[m, n] * residual[m])``.

    Per element, as TileKernels' compiled kernel computes it: the first stream's
    product ``comb[0, n] * residual[0]`` rounded to FP32, ``post[n] * x`` added to
    it by a fused multiply-add, then streams 1 to 3 likewise, in order, and one
    rounding to BF16. That is how nvcc contracts TileKernels'
    ``post * x + comb[0] * residual[0] + ...``: of the first addition's two
    products it fuses the left one, so the first stream's product is the one
    rounded (PTX ``mul.f32``, then four ``fma.rn.f32``). Here the ``T.fma`` calls
    fix that sequence: the rounded product's only use is an FMA's addend, which
    nothing can contract. The fused ``post_pre`` entry computes the same sequence.

    Each thread owns eight contiguous columns of one token: it loads them from
    ``x`` and every stream (16-byte loads) and its token's mixes before any
    arithmetic, so a token costs one memory round trip rather than TileKernels'
    pipelined loop over 1,024-column blocks. The grid is flat, a token's column
    blocks adjacent, so any token count launches. No element reads another, so a
    token's result does not depend on the batch.
    """
    assert hidden_size % (threads * VEC) == 0
    blocks = hidden_size // (threads * VEC)
    M = T.dynamic("M")

    @T.prim_func
    def mhc_post(
        x: T.Tensor((M, hidden_size), T.bfloat16),
        residual: T.Tensor((M, MULT, hidden_size), T.bfloat16),
        post: T.Tensor((M, MULT), T.float32),
        comb: T.Tensor((M, MULT, MULT), T.float32),
        out: T.Tensor((M, MULT, hidden_size), T.bfloat16),
    ):
        with T.Kernel(M * blocks, threads=threads) as pid:
            token = pid // blocks
            col = (pid % blocks * threads + T.get_thread_binding()) * VEC
            xv = T.alloc_local((VEC,), T.bfloat16)
            rv = T.alloc_local((MULT * VEC,), T.bfloat16)
            ov = T.alloc_local((MULT * VEC,), T.bfloat16)
            pm = T.alloc_local((MULT,), T.float32)
            cm = T.alloc_local((MULT * MULT,), T.float32)
            acc = T.alloc_var(T.float32)
            for n in T.vectorized(MULT):
                pm[n] = post[token, n]
            for m in T.unroll(MULT):
                for n in T.vectorized(MULT):
                    cm[m * MULT + n] = comb[token, m, n]
            for v in T.vectorized(VEC):
                xv[v] = x[token, col + v]
            for m in T.unroll(MULT):
                for v in T.vectorized(VEC):
                    rv[m * VEC + v] = residual[token, m, col + v]
            for n in T.unroll(MULT):
                for v in T.unroll(VEC):
                    acc = cm[n] * T.Cast(T.float32, rv[v])
                    acc = T.fma(pm[n], T.Cast(T.float32, xv[v]), acc)
                    for m in T.unroll(1, MULT):
                        acc = T.fma(
                            cm[m * MULT + n], T.Cast(T.float32, rv[m * VEC + v]), acc
                        )
                    ov[n * VEC + v] = T.Cast(T.bfloat16, acc)
            for n in T.unroll(MULT):
                for v in T.vectorized(VEC):
                    out[token, n, col + v] = ov[n * VEC + v]

    return mhc_post


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def collapse_streams(hidden_size: int, threads: int = 128):
    """The streams' weighted sum, TileKernels' ``mhc_pre_apply_mix_fwd`` to the bit:
    ``out = bf16(sum_m mix[m] * state[m])``.

    Per element, as TileKernels' compiled kernel computes it: from a positive
    zero, each stream's term added by one fused multiply-add, streams in order,
    and one rounding to BF16. Starting from zero rather than the first product
    matters only in sign: four negative-zero products sum to positive zero, as
    TileKernels' do. The thread mapping is ``post_streams``'.
    """
    assert hidden_size % (threads * VEC) == 0
    blocks = hidden_size // (threads * VEC)
    M = T.dynamic("M")

    @T.prim_func
    def mhc_collapse(
        state: T.Tensor((M, MULT, hidden_size), T.bfloat16),
        mix: T.Tensor((M, MULT), T.float32),
        out: T.Tensor((M, hidden_size), T.bfloat16),
    ):
        with T.Kernel(M * blocks, threads=threads) as pid:
            token = pid // blocks
            col = (pid % blocks * threads + T.get_thread_binding()) * VEC
            sv = T.alloc_local((MULT * VEC,), T.bfloat16)
            ov = T.alloc_local((VEC,), T.bfloat16)
            mv = T.alloc_local((MULT,), T.float32)
            acc = T.alloc_var(T.float32)
            for m in T.vectorized(MULT):
                mv[m] = mix[token, m]
            for m in T.unroll(MULT):
                for v in T.vectorized(VEC):
                    sv[m * VEC + v] = state[token, m, col + v]
            for v in T.unroll(VEC):
                acc = T.float32(0)
                for m in T.unroll(MULT):
                    acc = T.fma(mv[m], T.Cast(T.float32, sv[m * VEC + v]), acc)
                ov[v] = T.Cast(T.bfloat16, acc)
            for v in T.vectorized(VEC):
                out[token, col + v] = ov[v]

    return mhc_collapse


def _project_call(hidden: int, mode: str, world: int) -> prepare.TileLangCall:
    """The ``project_streams`` call of a sublayer entry."""
    return prepare.call(project_streams, hidden, mode, block_M=BLOCK_M, world=world)


def _finalize_call(
    hidden: int, k: int, rms_eps: float, hc_eps: float, iterations: int
) -> prepare.TileLangCall:
    """The ``finalize`` call of a sublayer entry projecting ``k`` stream columns."""
    return prepare.call(finalize, hidden, k, rms_eps, hc_eps, iterations, rms_eps)


def _post_call(hidden: int) -> prepare.TileLangCall:
    """The standalone ``post``'s call."""
    return prepare.call(post_streams, hidden)


def _collapse_call(hidden: int) -> prepare.TileLangCall:
    """The stream collapse's call."""
    return prepare.call(collapse_streams, hidden)


def _make(call: prepare.TileLangCall):
    function, args, kwargs = call
    return function(*args, **kwargs)


@cache
def _project(hidden: int, mode: str, world: int):
    return _make(_project_call(hidden, mode, world))


@cache
def _finalize(hidden: int, k: int, rms_eps: float, hc_eps: float, iterations: int):
    return _make(_finalize_call(hidden, k, rms_eps, hc_eps, iterations))


@cache
def _post(hidden: int):
    return _make(_post_call(hidden))


@cache
def _collapse(hidden: int):
    return _make(_collapse_call(hidden))


@torch.library.custom_op(
    "vllm::dsv41_tilelang_mhc_pre",
    mutates_args=("residual_out", "y", "post", "comb", "pre_out"),
)
def _mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    norm: torch.Tensor,
    pre: torch.Tensor,
    residual_out: torch.Tensor | None,
    y: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    pre_out: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    iterations: int,
    previous_output: torch.Tensor | None = None,
    previous_post: torch.Tensor | None = None,
    previous_comb: torch.Tensor | None = None,
) -> None:
    tokens, hidden = int(residual.shape[0]), int(residual.shape[-1])
    if tokens == 0:
        return
    streams = MULT * hidden
    world = 1
    # Unused operands of a mode take a buffer of the right shape; none is read.
    if previous_output is not None:
        # (T, H), or the ranks' unreduced partials of it, (world, T, H).
        if previous_output.ndim == 3:
            world = int(previous_output.shape[0])
        mode, x, k = "post_pre", previous_output, streams
        source, updated = residual.view(tokens, streams), residual_out
        mix_post, mix_comb = previous_post, previous_comb
    elif residual.ndim == 2:
        mode, x, k = "broadcast", residual, hidden
        source, updated = residual_out.view(tokens, streams), residual_out
        mix_post, mix_comb = pre, comb
    else:
        mode, x, k = "pre", y, streams
        source = residual.view(tokens, streams)
        updated, mix_post, mix_comb = source, pre, comb
    splits = hidden // BLOCK_H
    scratch = current_workspace_manager().get_simultaneous(
        ((splits, tokens, 1, MIXES), torch.float32),
        ((splits, tokens, 1), torch.float32),
        ((tokens, hidden), torch.bfloat16),
        ((splits, tokens), torch.float32),
    )
    retain_cuda_graph_capture_resource(scratch)
    P, S, C, Q = scratch
    _project(hidden, mode, world)(
        x if x.ndim == 3 else x.unsqueeze(0),
        source,
        mix_post,
        mix_comb,
        pre,
        fn,
        updated.view(tokens, streams),
        P,
        S,
        C,
        Q,
    )
    _finalize(hidden, k, rms_eps, hc_eps, iterations)(
        P, S, C, Q, scale, base, norm, post, comb, pre_out, y
    )


@_mhc_pre.register_fake
def _mhc_pre_fake(
    residual,
    fn,
    scale,
    base,
    norm,
    pre,
    residual_out,
    y,
    post,
    comb,
    pre_out,
    rms_eps,
    hc_eps,
    iterations,
    previous_output=None,
    previous_post=None,
    previous_comb=None,
):
    return None


@torch.library.custom_op("vllm::dsv41_tilelang_mhc_post", mutates_args=("out",))
def _mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    out: torch.Tensor,
) -> None:
    tokens, hidden = int(residual.shape[0]), int(residual.shape[-1])
    if tokens == 0:
        return
    _post(hidden)(
        x.view(tokens, hidden),
        residual.view(tokens, MULT, hidden),
        post.view(tokens, MULT),
        comb.view(tokens, MULT, MULT),
        out.view(tokens, MULT, hidden),
    )


@_mhc_post.register_fake
def _mhc_post_fake(x, residual, post, comb, out):
    return None


@torch.library.custom_op("vllm::dsv41_tilelang_mhc_collapse", mutates_args=("out",))
def _mhc_collapse(state: torch.Tensor, mix: torch.Tensor, out: torch.Tensor) -> None:
    tokens, hidden = int(state.shape[0]), int(state.shape[-1])
    if tokens == 0:
        return
    _collapse(hidden)(
        state.view(tokens, MULT, hidden),
        mix.view(tokens, MULT),
        out.view(tokens, hidden),
    )


@_mhc_collapse.register_fake
def _mhc_collapse_fake(state, mix, out):
    return None


class TileKernelsMHC(nn.Module):
    """V4.1 lagged mHC for a decoder layer, with ``B12xMHC``'s interface."""

    def __init__(self, config):
        super().__init__()
        if config.hc_mult != MULT:
            raise ValueError("V4.1 mHC requires four streams")
        self.hidden_size = config.hidden_size
        self.rms_eps = config.rms_norm_eps
        self.hc_eps = config.hc_eps
        self.iterations = config.hc_sinkhorn_iters

    def _entries(self) -> list[tuple[str, int]]:
        """The (mode, world) of every sublayer entry this layer can launch."""
        entries = [(mode, 1) for mode in ("pre", "post_pre", "broadcast")]
        if sp_prefill.unreduced_enabled():
            # Sequence-parallel steps hand post_pre the ranks' unreduced parts.
            entries.append(("post_pre", get_tensor_model_parallel_world_size()))
        return entries

    def declare_kernels(self) -> None:
        """Declare every sublayer entry's kernels, the standalone post and the stream
        collapse, so that no forward (or CUDA graph capture) compiles."""
        hidden = self.hidden_size
        entries = self._entries()
        # The broadcast entry projects one stream; the others all four.
        widths = (hidden, MULT * hidden)
        mix = (self.rms_eps, self.hc_eps, self.iterations)

        def calls() -> list[prepare.TileLangCall]:
            return [
                *(_project_call(hidden, mode, world) for mode, world in entries),
                *(_finalize_call(hidden, k, *mix) for k in widths),
                _post_call(hidden),
                _collapse_call(hidden),
            ]

        def bind() -> None:
            # The forward's cached kernels, as its lookups name them.
            for mode, world in entries:
                _project(hidden, mode, world)
            for k in widths:
                _finalize(hidden, k, *mix)
            _post(hidden)
            _collapse(hidden)

        prepare.declare(calls, bind=bind)

    def pre(
        self,
        residual,
        fn,
        scale,
        base,
        norm,
        pre,
        *,
        previous_output=None,
        previous_post=None,
        previous_comb=None,
    ):
        tokens, device = int(residual.shape[0]), residual.device
        if isinstance(previous_output, RankOrderParts):
            # Added inside the projection kernel, in rank order.
            previous_output = previous_output.packed
        if pre is None:
            pre = torch.zeros((tokens, MULT), dtype=torch.float32, device=device)
            pre[:, 0].fill_(1)
        unchanged = previous_output is None and residual.ndim == 3
        residual_out = (
            None
            if unchanged
            else torch.empty(
                (tokens, MULT, self.hidden_size), dtype=residual.dtype, device=device
            )
        )
        y = torch.empty((tokens, self.hidden_size), dtype=residual.dtype, device=device)
        post = torch.empty((tokens, MULT), dtype=torch.float32, device=device)
        comb = torch.empty((tokens, MULT, MULT), dtype=torch.float32, device=device)
        pre_out = torch.empty((tokens, MULT), dtype=torch.float32, device=device)
        _mhc_pre(
            residual,
            fn,
            scale,
            base,
            norm,
            pre,
            residual_out,
            y,
            post,
            comb,
            pre_out,
            self.rms_eps,
            self.hc_eps,
            self.iterations,
            previous_output,
            previous_post,
            previous_comb,
        )
        return residual if unchanged else residual_out, post, comb, y, pre_out

    def post_pre(self, x, residual, post, comb, fn, scale, base, norm, pre):
        return self.pre(
            residual,
            fn,
            scale,
            base,
            norm,
            pre,
            previous_output=x,
            previous_post=post,
            previous_comb=comb,
        )

    def post(self, x, residual, post, comb):
        if isinstance(x, RankOrderParts):
            x = x.sum()
        out = torch.empty_like(residual)
        _mhc_post(x, residual, post, comb, out)
        return out

    def collapse(
        self, state: torch.Tensor, pre: torch.Tensor | None = None
    ) -> torch.Tensor:
        """The streams' weighted sum with ``pre``, or their mean."""
        if pre is None:
            pre = torch.full(
                (state.shape[0], MULT), 0.25, dtype=torch.float32, device=state.device
            )
        out = torch.empty(
            (state.shape[0], state.shape[-1]), dtype=state.dtype, device=state.device
        )
        _mhc_collapse(state, pre, out)
        return out


__all__ = [
    "TileKernelsMHC",
    "collapse_streams",
    "finalize",
    "post_streams",
    "project_streams",
]
