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

TileKernels provides DeepSeek's mHC kernels. A standalone ``post`` and the
final stream collapse use them as they are. A sublayer's entry needs more:
DeepSeek moved the projection GEMM to DeepGEMM, which does not run on
SM120/SM121, TileKernels' fused pre step applies the mix it has just computed
rather than the lagged one, and in prefill the four streams are large enough
that reading them once matters. Two TileLang kernels make each entry:

- ``project_streams``: per hidden block, the previous sublayer's ``post``
  (writing the updated streams), the projection's split partials on TF32
  tensor cores (DeepGEMM's precision) with the streams' sum of squares, and
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
):
    """The streams' projection partials and their collapse, for one hidden block.

    ``mode`` is the sublayer's entry: ``"post_pre"`` first folds the previous
    sublayer's output into the streams (``post``) and writes them, ``"pre"``
    reads streams already updated, and ``"broadcast"`` expands an embedding to
    four copies (the first layer, whose ``fn`` is the stream blocks summed).

    Each CTA owns ``block_H`` hidden columns of every stream, a K split of the
    projection, for ``block_M`` tokens. Each thread owns one token's
    ``block_H / 8`` contiguous columns: it loads them for every stream into
    registers, folds the previous output in, writes the updated streams, and
    keeps its share of the streams' sum of squares ``S`` and of the collapse
    with the incoming pre-mix. The projection ``P = streams @ fn^T`` runs on
    TF32 tensor cores (DeepGEMM's precision), one stream at a time. The
    collapse is rounded to BF16, with its sum of squares ``Q``.
    """
    assert hidden_size % block_H == 0 and mode in ("pre", "post_pre", "broadcast")
    splits = hidden_size // block_H
    streams = 1 if mode == "broadcast" else MULT
    K = streams * hidden_size
    N = 32  # the 24 mixes, padded
    LANES = 8  # threads per token: adjacent lanes of one warp
    COLS = block_H // LANES  # contiguous columns per thread
    threads = block_M * LANES
    H = hidden_size
    M = T.dynamic("M")

    @T.prim_func
    def main(
        X: T.Tensor((M, H), T.bfloat16),
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
            # This thread's token: its mixes, then its columns of x and the streams.
            for k in T.unroll(MULT):
                coef[k] = pre_in[src, k]
            if mode == "post_pre":
                for k in T.unroll(MULT):
                    coef[MULT + k] = post[src, k]
                for k in T.unroll(MULT * MULT):
                    coef[2 * MULT + k] = comb[src, k // MULT, k % MULT]
            if mode != "pre":
                for v in T.vectorized(COLS):
                    xv[v] = X[src, h0 + c0 + v]
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
                T.copy(Fn[0:MIXES, h0 : h0 + block_H], F_shared[0:MIXES, :])
                T.sync_threads()
                T.gemm(A_shared, F_shared, C_local, transpose_B=True)
            else:
                for n in T.unroll(MULT):
                    if mode == "post_pre":
                        # post[n] * x + sum_m comb[m, n] * streams[m], in FP32.
                        for v in T.unroll(COLS):
                            av[v] = coef[MULT + n] * T.Cast(T.float32, xv[v])
                            for m in T.unroll(MULT):
                                av[v] += coef[2 * MULT + m * MULT + n] * T.Cast(
                                    T.float32, rv[m * COLS + v]
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
                    T.copy(
                        Fn[0:MIXES, n * H + h0 : n * H + h0 + block_H],
                        F_shared[0:MIXES, :],
                    )
                    T.sync_threads()
                    T.gemm(A_shared, F_shared, C_local, transpose_B=True)
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

    return main


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
    def main(
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

    return main


@cache
def _project(hidden: int, mode: str):
    return project_streams(hidden, mode, block_M=BLOCK_M)


@cache
def _finalize(hidden: int, k: int, rms_eps: float, hc_eps: float, iterations: int):
    return finalize(hidden, k, rms_eps, hc_eps, iterations, rms_eps)


def _tile_kernels_mhc():
    import tile_kernels

    return tile_kernels.mhc


@torch.library.custom_op(
    "vllm::bench_0040_mhc_pre",
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
    # Unused operands of a mode take a buffer of the right shape; none is read.
    if previous_output is not None:
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
    _project(hidden, mode)(
        x,
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


@torch.library.custom_op("vllm::bench_0040_mhc_post", mutates_args=("out",))
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
    _tile_kernels_mhc().mhc_post_fwd(
        x.view(1, tokens, hidden),
        residual.view(1, tokens, MULT, hidden),
        post.view(1, tokens, MULT, 1),
        comb.view(1, tokens, MULT, MULT),
        out=out.view(1, tokens, MULT, hidden),
    )


@_mhc_post.register_fake
def _mhc_post_fake(x, residual, post, comb, out):
    return None


@torch.library.custom_op("vllm::bench_0040_mhc_collapse", mutates_args=("out",))
def _mhc_collapse(state: torch.Tensor, mix: torch.Tensor, out: torch.Tensor) -> None:
    if state.shape[0] == 0:
        return
    _tile_kernels_mhc().mhc_pre_apply_mix_fwd(state, mix, out)


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
        # Compile every kernel now, never during CUDA graph capture.
        for mode in ("pre", "post_pre", "broadcast"):
            _project(self.hidden_size, mode)
        for k in (self.hidden_size, MULT * self.hidden_size):
            _finalize(self.hidden_size, k, self.rms_eps, self.hc_eps, self.iterations)

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


__all__ = ["TileKernelsMHC", "finalize", "project_streams"]
