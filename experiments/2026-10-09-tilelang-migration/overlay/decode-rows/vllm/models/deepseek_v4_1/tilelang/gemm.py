# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1-Flash block-32 FP8 linear for SM120 / SM121.

V4.1 Flash stores its dense projections as E4M3 weights with one UE8M0 scale
per 32x32 block (``weight_block_size = [32, 32]``, ``scale_fmt = "ue8m0"``) and
quantizes activations dynamically to E4M3 with one UE8M0 scale per 32
elements. Both operands are therefore MXFP8, and the GEMM maps directly onto
SM120's ``mma.sync.m16n8k32.kind::mxf8f6f4.block_scale.scale_vec::1X``: the
tensor core applies both scales per K32 atom and accumulates in FP32, so no
per-block promotion loop is needed.

Scales reach ``T.mma_gemm_blockscaled`` as uint32 words, each packing the
UE8M0 exponents of four consecutive K32 blocks of one row (byte ``j`` covers
K block ``4 * word + j``). The weight's 32x32 block scales are broadcast to
per-row words once on the host.
"""

import tilelang
import tilelang.language as T
import torch

SF_BLOCK = 32  # K elements per UE8M0 scale
SF_WORD_K = 4 * SF_BLOCK  # K elements per packed scale word


def scale_words(k: int) -> int:
    """Packed UE8M0 words per row of a K-wide operand: four K32 exponents each."""
    return tilelang.cdiv(k, SF_WORD_K)


def block_scale_words(block_K: int) -> int:
    """Scale words one K block reads: blocks off a 128 boundary straddle two words."""
    return tilelang.cdiv(block_K + block_K % SF_WORD_K, SF_WORD_K)


def pick_block_K(K: int, widest: int = 192) -> int:
    """128 when it divides K; else the widest multiple of 64 up to ``widest`` that does.

    TP3's 768-wide expert rows take 128; TP4's 576 take 192 (or 64).
    """
    for block_K in (128, *range(widest - widest % 64, 0, -64)):
        if K % block_K == 0:
            return block_K
    raise ValueError(f"K={K} must be a multiple of 64")


def _mxfp8_gemm(
    N: int,
    K: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 128,
    num_stages: int = 2,
    threads: int = 128,
    out_dtype: T.dtype = T.bfloat16,
    padded_rows: bool = False,
    swizzle_panel: int = 0,
    groups: int = 1,
    shards: int = 1,
):
    """``C[M, N] = (A * SFA) @ (B * SFB)^T`` for MXFP8 ``A[M, K]``, ``B[N, K]``.

    M is dynamic. ``SFA`` / ``SFB`` hold ``ceil(K / 128)`` packed UE8M0 words
    per row. ``block_K`` is any multiple of 64 dividing K; a block that starts
    off a 128-element boundary reads the two words it straddles and tells the
    MMA where it starts (``k_start``), so no operand is padded.

    With ``padded_rows``, ``A`` and ``SFA`` have ``Mp >= M`` rows, a multiple
    of ``block_M`` (the decode workspace): every activation tile is loaded
    whole and only the ``M`` live rows are stored. A partial tile's missing
    rows otherwise cost about 0.4 us each on SM121, which made one decode row
    nearly three times slower than 64.

    Each output accumulates over K in the same order whatever the tile shape
    or padding, so a row's result does not depend on M or the configuration.
    ``swizzle_panel`` rasterizes the CTAs in panels of that many N tiles, so a
    weight larger than L2 is read once per panel rather than once per row of
    tiles; it only changes the order tiles run in.

    With ``groups`` > 1 the projection is grouped (the attention's WO-A): ``A`` is
    ``[M, groups * K]``, ``B`` and ``SFB`` stack the groups' ``[N, K]`` weights,
    and ``C`` is ``[M, groups * N]``; output columns of group ``g`` read only A's
    columns ``g * K`` to ``(g + 1) * K``. One launch covers every group.

    With ``shards`` > 1 each of the ``shards`` equal K ranges is summed from
    zero and the shards are added in order: the arithmetic of
    ``mxfp8_gemm_partials`` plus ``splitk_reduce``, so a narrow projection's
    split-K decode rows give its prefill rows' bits.
    """
    assert K % block_K == 0 and block_K % 64 == 0, (
        "block_K must divide K and be a multiple of 64"
    )
    assert groups == 1 or (N % block_N == 0 and K % SF_WORD_K == 0), (
        "grouped tiles must not straddle a group"
    )
    assert shards == 1 or (groups == 1 and (K // block_K) % shards == 0), (
        "shards must split K into whole blocks"
    )
    k_blocks = K // block_K
    M = T.dynamic("M")
    Mp = T.dynamic("Mp") if padded_rows else M
    in_dtype = T.float8_e4m3fn
    accum_dtype = T.float32
    sf_words = scale_words(K)
    block_words = block_scale_words(block_K)
    total_N = groups * N

    @T.prim_func
    def main(
        A: T.Tensor((Mp, groups * K), in_dtype),
        B: T.Tensor((total_N, K), in_dtype),
        SFA: T.Tensor((Mp, groups * sf_words), T.uint32),
        SFB: T.Tensor((total_N, sf_words), T.uint32),
        C: T.Tensor((M, total_N), out_dtype),
    ):
        with T.Kernel(
            T.ceildiv(total_N, block_N), T.ceildiv(M, block_M), threads=threads
        ) as (bx, by):
            if swizzle_panel:
                T.use_swizzle(panel_size=swizzle_panel)
            # The group's activation columns and scale words.
            group = bx * block_N // N
            a_k = group * K if groups > 1 else 0
            a_word = group * sf_words if groups > 1 else 0
            A_shared = T.alloc_shared((block_M, block_K), in_dtype)
            B_shared = T.alloc_shared((block_N, block_K), in_dtype)
            SFA_shared = T.alloc_shared((block_M, block_words), T.uint32)
            SFB_shared = T.alloc_shared((block_N, block_words), T.uint32)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_total = T.alloc_fragment(
                (block_M, block_N) if shards > 1 else (1, 1), accum_dtype
            )
            T.clear(C_local)
            if shards > 1:
                T.clear(C_total)
            for ko in T.Pipelined(k_blocks, num_stages=num_stages):
                T.copy(A[by * block_M, a_k + ko * block_K], A_shared)
                T.copy(B[bx * block_N, ko * block_K], B_shared)
                # Edge tiles clamp their scale rows; those rows are never stored.
                word0 = ko * block_K // SF_WORD_K
                for i, w in T.Parallel(block_M, block_words):
                    SFA_shared[i, w] = SFA[
                        T.min(by * block_M + i, Mp - 1),
                        a_word + T.min(word0 + w, sf_words - 1),
                    ]
                for j, w in T.Parallel(block_N, block_words):
                    SFB_shared[j, w] = SFB[
                        T.min(bx * block_N + j, total_N - 1),
                        T.min(word0 + w, sf_words - 1),
                    ]
                # The staged words start at the block's first word; k_start
                # places the block inside it.
                T.mma_gemm_blockscaled(
                    A_shared,
                    B_shared,
                    C_local,
                    SFA_shared,
                    SFB_shared,
                    transpose_B=True,
                    k_start=ko * block_K % SF_WORD_K,
                    sf_a_granularity_k=SF_BLOCK,
                    sf_b_granularity_k=SF_BLOCK,
                )
                if shards > 1:  # noqa: SIM102
                    if (ko + 1) % (k_blocks // shards) == 0:
                        for i, j in T.Parallel(block_M, block_N):
                            C_total[i, j] = T.if_then_else(
                                ko + 1 == k_blocks // shards,
                                C_local[i, j],
                                C_total[i, j] + C_local[i, j],
                            )
                        T.clear(C_local)
            if shards > 1:
                T.copy(C_total, C[by * block_M, bx * block_N])
            else:
                T.copy(C_local, C[by * block_M, bx * block_N])

    return main


# Prefill tiles use TileLang's warp-specialized pipeline (TMA producer warps,
# up to 17% faster there). Decode tiles do not: with warp specialization the
# cp.async scale staging races on SM121 at decode tile shapes (16-row tiles
# returned wrong results in up to every run; 64-row tiles were not seen to),
# while the unspecialized pipeline is race-free and as fast for decode. Both
# run the same arithmetic, so their bits agree.
def _mxfp8_gemm_partials(
    N: int,
    K: int,
    shards: int,
    block_M: int = 64,
    block_N: int = 64,
    block_K: int = 128,
    num_stages: int = 3,
    threads: int = 128,
):
    """Split-K MXFP8 GEMM for decode rows: FP32 ``P[s] = (A * SFA) @ (B * SFB)^T``
    over K shard ``s``, A and SFA padded to whole ``block_M`` tiles (the decode
    workspace). A narrow projection has few N tiles; its shards fill the GPU.
    """
    assert K % block_K == 0 and block_K % 64 == 0 and (K // block_K) % shards == 0
    shard_blocks = K // block_K // shards
    M = T.dynamic("M")
    Mp = T.dynamic("Mp")
    in_dtype = T.float8_e4m3fn
    sf_words = scale_words(K)
    block_words = block_scale_words(block_K)

    @T.prim_func
    def main(
        A: T.Tensor((Mp, K), in_dtype),
        B: T.Tensor((N, K), in_dtype),
        SFA: T.Tensor((Mp, sf_words), T.uint32),
        SFB: T.Tensor((N, sf_words), T.uint32),
        P: T.Tensor((shards, M, N), T.float32),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N), shards, T.ceildiv(M, block_M), threads=threads
        ) as (bx, s, by):
            A_shared = T.alloc_shared((block_M, block_K), in_dtype)
            B_shared = T.alloc_shared((block_N, block_K), in_dtype)
            SFA_shared = T.alloc_shared((block_M, block_words), T.uint32)
            SFB_shared = T.alloc_shared((block_N, block_words), T.uint32)
            C_local = T.alloc_fragment((block_M, block_N), T.float32)
            T.clear(C_local)
            for kb in T.Pipelined(shard_blocks, num_stages=num_stages):
                ko = s * shard_blocks + kb
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[bx * block_N, ko * block_K], B_shared)
                word0 = ko * block_K // SF_WORD_K
                for i, w in T.Parallel(block_M, block_words):
                    SFA_shared[i, w] = SFA[
                        T.min(by * block_M + i, Mp - 1),
                        T.min(word0 + w, sf_words - 1),
                    ]
                for j, w in T.Parallel(block_N, block_words):
                    SFB_shared[j, w] = SFB[
                        T.min(bx * block_N + j, N - 1),
                        T.min(word0 + w, sf_words - 1),
                    ]
                T.mma_gemm_blockscaled(
                    A_shared,
                    B_shared,
                    C_local,
                    SFA_shared,
                    SFB_shared,
                    transpose_B=True,
                    k_start=ko * block_K % SF_WORD_K,
                    sf_a_granularity_k=SF_BLOCK,
                    sf_b_granularity_k=SF_BLOCK,
                )
            T.copy(C_local, P[s, by * block_M, bx * block_N])

    return main


mxfp8_gemm = tilelang.jit(_mxfp8_gemm)
mxfp8_gemm_decode = tilelang.jit(
    pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True}
)(_mxfp8_gemm)
mxfp8_gemm_partials = tilelang.jit(
    pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True}
)(_mxfp8_gemm_partials)


@tilelang.jit
def splitk_reduce(
    N: int,
    shards: int,
    out_dtype: T.dtype = T.bfloat16,
    block_N: int = 128,
    threads: int = 128,
):
    """``C[m, n] = P[0, m, n] + P[1, m, n] + ...``, added in shard order in FP32.

    The sum starts from shard 0 itself, not from zero, so it matches the
    in-kernel shard accumulation of ``bf16_gemm`` bit for bit, signed zeros
    included.
    """
    M = T.dynamic("M")

    @T.prim_func
    def main(P: T.Tensor((shards, M, N), T.float32), C: T.Tensor((M, N), out_dtype)):
        with T.Kernel(T.ceildiv(N, block_N), M, threads=threads) as (bx, m):
            acc = T.alloc_fragment((block_N,), T.float32)
            # Columns past N (a narrow projection's last block) are skipped.
            for j in T.Parallel(block_N):
                n = T.min(bx * block_N + j, N - 1)
                acc[j] = P[0, m, n]
            for s in T.serial(1, shards):
                for j in T.Parallel(block_N):
                    n = T.min(bx * block_N + j, N - 1)
                    acc[j] += P[s, m, n]
            for j in T.Parallel(block_N):
                if bx * block_N + j < N:
                    C[m, bx * block_N + j] = acc[j]

    return main


@tilelang.jit
def bf16_gemm(
    N: int,
    K: int,
    block_M: int = 64,
    block_N: int = 64,
    block_K: int = 64,
    num_stages: int = 3,
    threads: int = 128,
    out_dtype: T.dtype = T.bfloat16,
    shards: int = 1,
    blocked: bool = False,
):
    """``C[M, N] = A @ B^T`` for BF16 ``A[M, K]`` and ``B[N, K]``, accumulated in FP32.

    M is dynamic. With ``shards`` > 1 each K shard is summed from zero and
    the shards added in order, the arithmetic of ``bf16_gemm_partials`` plus
    ``splitk_reduce``, so a row's result never depends on the batch it runs in.
    """
    k_blocks = tilelang.cdiv(K, block_K)
    assert shards == 1 or (K % block_K == 0 and k_blocks % shards == 0)
    M = T.dynamic("M")
    dtype = T.bfloat16
    shard_blocks = k_blocks // shards

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads
        ) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), T.float32)
            if blocked:
                C_block = T.alloc_fragment((block_M, block_N), T.float32)
            C_total = T.alloc_fragment(
                (block_M, block_N) if shards > 1 else (1, 1), T.float32
            )
            T.clear(C_local)
            if shards > 1:
                T.clear(C_total)
            for ko in T.Pipelined(k_blocks, num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[bx * block_N, ko * block_K], B_shared)
                if (
                    blocked
                ):  # each K block summed alone, then added: shorter FP32 chains
                    T.clear(C_block)
                    T.gemm(A_shared, B_shared, C_block, transpose_B=True)
                    for i, j in T.Parallel(block_M, block_N):
                        C_local[i, j] += C_block[i, j]
                else:
                    T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                if shards > 1:  # noqa: SIM102
                    if (ko + 1) % shard_blocks == 0:
                        for i, j in T.Parallel(block_M, block_N):
                            C_total[i, j] = T.if_then_else(
                                ko + 1 == shard_blocks,
                                C_local[i, j],
                                C_total[i, j] + C_local[i, j],
                            )
                        T.clear(C_local)
            if shards > 1:
                T.copy(C_total, C[by * block_M, bx * block_N])
            else:
                T.copy(C_local, C[by * block_M, bx * block_N])

    return main


@tilelang.jit
def splitk_reduce_two(
    N: int,
    shards: int,
    out_dtype: T.dtype = T.bfloat16,
    block_N: int = 128,
    threads: int = 128,
):
    """``splitk_reduce`` of ``2 * N`` columns into two outputs: columns below N
    to ``C0``, the rest to ``C1`` (the compressor's values and gates)."""
    assert N % block_N == 0
    M = T.dynamic("M")
    tiles = N // block_N

    @T.prim_func
    def main(
        P: T.Tensor((shards, M, 2 * N), T.float32),
        C0: T.Tensor((M, N), out_dtype),
        C1: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(2 * tiles, M, threads=threads) as (bx, m):
            acc = T.alloc_fragment((block_N,), T.float32)
            for j in T.Parallel(block_N):
                acc[j] = P[0, m, bx * block_N + j]
            for s in T.serial(1, shards):
                for j in T.Parallel(block_N):
                    acc[j] += P[s, m, bx * block_N + j]
            if bx < tiles:
                for j in T.Parallel(block_N):
                    C0[m, bx * block_N + j] = acc[j]
            else:
                for j in T.Parallel(block_N):
                    C1[m, (bx - tiles) * block_N + j] = acc[j]

    return main


@tilelang.jit
def bf16_gemm_two(
    N: int,
    K: int,
    block_M: int = 64,
    block_N: int = 64,
    block_K: int = 64,
    num_stages: int = 3,
    threads: int = 128,
    out_dtype: T.dtype = T.bfloat16,
    shards: int = 1,
    blocked: bool = False,
):
    """``bf16_gemm`` of a ``[2N, K]`` weight into two ``[M, N]`` outputs: rows below
    N of the weight to ``C0``, the rest to ``C1``, with ``bf16_gemm``'s shard
    arithmetic (so it gives ``bf16_gemm_partials`` plus ``splitk_reduce_two``'s bits).
    """
    k_blocks = tilelang.cdiv(K, block_K)
    assert N % block_N == 0
    assert shards == 1 or (K % block_K == 0 and k_blocks % shards == 0)
    M = T.dynamic("M")
    dtype = T.bfloat16
    shard_blocks = k_blocks // shards
    tiles = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((2 * N, K), dtype),
        C0: T.Tensor((M, N), out_dtype),
        C1: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(2 * tiles, T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), T.float32)
            if blocked:
                C_block = T.alloc_fragment((block_M, block_N), T.float32)
            C_total = T.alloc_fragment(
                (block_M, block_N) if shards > 1 else (1, 1), T.float32
            )
            T.clear(C_local)
            if shards > 1:
                T.clear(C_total)
            for ko in T.Pipelined(k_blocks, num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[bx * block_N, ko * block_K], B_shared)
                if (
                    blocked
                ):  # each K block summed alone, then added: shorter FP32 chains
                    T.clear(C_block)
                    T.gemm(A_shared, B_shared, C_block, transpose_B=True)
                    for i, j in T.Parallel(block_M, block_N):
                        C_local[i, j] += C_block[i, j]
                else:
                    T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                if shards > 1:  # noqa: SIM102
                    if (ko + 1) % shard_blocks == 0:
                        for i, j in T.Parallel(block_M, block_N):
                            C_total[i, j] = T.if_then_else(
                                ko + 1 == shard_blocks,
                                C_local[i, j],
                                C_total[i, j] + C_local[i, j],
                            )
                        T.clear(C_local)
            if bx < tiles:
                if shards > 1:
                    T.copy(C_total, C0[by * block_M, bx * block_N])
                else:
                    T.copy(C_local, C0[by * block_M, bx * block_N])
            else:
                if shards > 1:
                    T.copy(C_total, C1[by * block_M, (bx - tiles) * block_N])
                else:
                    T.copy(C_local, C1[by * block_M, (bx - tiles) * block_N])

    return main


@tilelang.jit
def bf16_gemm_partials(
    N: int,
    K: int,
    shards: int,
    block_M: int = 64,
    block_N: int = 64,
    block_K: int = 64,
    num_stages: int = 3,
    threads: int = 128,
    blocked: bool = False,
):
    """Split-K BF16 GEMM for decode rows: FP32 ``P[s] = A @ B^T`` over K shard ``s``."""
    assert K % block_K == 0 and (K // block_K) % shards == 0
    M = T.dynamic("M")
    dtype = T.bfloat16
    shard_blocks = K // block_K // shards

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        P: T.Tensor((shards, M, N), T.float32),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N), shards, T.ceildiv(M, block_M), threads=threads
        ) as (bx, s, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), T.float32)
            if blocked:
                C_block = T.alloc_fragment((block_M, block_N), T.float32)
            T.clear(C_local)
            for kb in T.Pipelined(shard_blocks, num_stages=num_stages):
                ko = s * shard_blocks + kb
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[bx * block_N, ko * block_K], B_shared)
                if (
                    blocked
                ):  # each K block summed alone, then added: shorter FP32 chains
                    T.clear(C_block)
                    T.gemm(A_shared, B_shared, C_block, transpose_B=True)
                    for i, j in T.Parallel(block_M, block_N):
                        C_local[i, j] += C_block[i, j]
                else:
                    T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            T.copy(
                C_local,
                P[
                    s,
                    by * block_M : (by + 1) * block_M,
                    bx * block_N : (bx + 1) * block_N,
                ],
            )

    return main


# The last CTA to finish a split-K tile adds the shards. Every CTA publishes its
# partial (stores, a device-scope fence, then an acq_rel arrival count); the last
# one fences again before it reads the partials, which no earlier load on its SM
# touched, so none sits stale in its L1.
_SPLITK_HEADER = r"""
__device__ __forceinline__ void splitk_fence() { __threadfence(); }
"""


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def bf16_gemm_splitk(
    N: int,
    K: int,
    shards: int,
    block_M: int = 16,
    block_N: int = 64,
    block_K: int = 64,
    num_stages: int = 3,
    threads: int = 128,
    out_dtype: T.dtype = T.float32,
    parts: int = 1,
    blocked: bool = False,
):
    """Split-K BF16 GEMM in one launch: CTA ``(n, s, m)`` writes shard ``s``'s FP32
    partial of its tile to ``P``; the tile's last CTA adds ``P[0] + P[1] + ...`` in
    shard order (``splitk_reduce``'s arithmetic, so its bits) into the outputs and
    resets the tile's counter. ``counters`` holds one zeroed int32 per tile
    (``cdiv(M, block_M) * cdiv(N, block_N)``); it is left zeroed for the next call.
    With ``parts`` = 2, columns below ``N / 2`` go to ``C0`` and the rest to ``C1``.
    """
    assert K % block_K == 0 and (K // block_K) % shards == 0
    assert (
        parts in (1, 2) and N % parts == 0 and (parts == 1 or (N // 2) % block_N == 0)
    )
    shard_blocks = K // block_K // shards
    tiles_N = tilelang.cdiv(N, block_N)
    part = N // parts
    M = T.dynamic("M")
    tiles = T.dynamic("tiles")
    dtype = T.bfloat16

    @T.macro
    def body(A, B, P, counters, C0, C1):
        with T.Kernel(tiles_N, shards, T.ceildiv(M, block_M), threads=threads) as (
            bx,
            s,
            by,
        ):
            T.import_source(_SPLITK_HEADER)
            tx = T.get_thread_binding()
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), T.float32)
            if blocked:
                C_block = T.alloc_fragment((block_M, block_N), T.float32)
            last = T.alloc_shared((1,), T.int32)
            T.clear(C_local)
            for kb in T.Pipelined(shard_blocks, num_stages=num_stages):
                ko = s * shard_blocks + kb
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[bx * block_N, ko * block_K], B_shared)
                if blocked:
                    T.clear(C_block)
                    T.gemm(A_shared, B_shared, C_block, transpose_B=True)
                    for i, j in T.Parallel(block_M, block_N):
                        C_local[i, j] += C_block[i, j]
                else:
                    T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            T.copy(C_local, P[s, by * block_M, bx * block_N])
            T.call_extern("handle", "splitk_fence")
            T.sync_threads()
            if tx == 0:
                arrived = T.atomic_add(
                    counters[by * tiles_N + bx],
                    1,
                    memory_order="acq_rel",
                    return_prev=True,
                )
                last[0] = T.if_then_else(arrived == shards - 1, 1, 0)
            T.sync_threads()
            if last[0] == 1:
                T.call_extern("handle", "splitk_fence")
                for i, j in T.Parallel(block_M, block_N):
                    m = by * block_M + i
                    n = bx * block_N + j
                    if m < M and n < N:
                        acc = T.alloc_var(T.float32)
                        acc = P[0, m, n]
                        for t in T.serial(1, shards):
                            acc = acc + P[t, m, n]
                        if parts == 1:
                            C0[m, n] = acc
                        else:
                            if n < part:  # noqa: SIM102
                                C0[m, n] = acc
                            else:
                                C1[m, n - part] = acc
                if tx == 0:
                    counters[by * tiles_N + bx] = 0

    if parts == 2:

        @T.prim_func
        def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((N, K), dtype),
            P: T.Tensor((shards, M, N), T.float32),
            counters: T.Tensor((tiles,), T.int32),
            C0: T.Tensor((M, part), out_dtype),
            C1: T.Tensor((M, part), out_dtype),
        ):
            body(A, B, P, counters, C0, C1)

    else:

        @T.prim_func
        def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((N, K), dtype),
            P: T.Tensor((shards, M, N), T.float32),
            counters: T.Tensor((tiles,), T.int32),
            C0: T.Tensor((M, N), out_dtype),
        ):
            body(A, B, P, counters, C0, None)

    return main


# ---------------------------------------------------------------------------
# Scale packing and tile configuration
# ---------------------------------------------------------------------------


def pack_scale_words(exponents: torch.Tensor, rows: int | None = None) -> torch.Tensor:
    """Pack ``[..., K / 32]`` UE8M0 bytes into ``[..., ceil(K / 128)]`` uint32 words.

    With ``rows``, a ``[rows / 32, K / 32]`` block-scale grid is first
    broadcast to every row. Rows whose exponent count is not a multiple of
    four gain unread padding bytes (TP4's 576-wide rows: 18 bytes become 20).
    """
    exponents = exponents.view(torch.uint8)
    if rows is not None and exponents.shape[0] != rows:
        exponents = exponents.repeat_interleave(rows // exponents.shape[0], dim=0)
    pad = -exponents.shape[-1] % 4
    if pad:
        exponents = torch.nn.functional.pad(exponents, (0, pad), value=127)
    return exponents.contiguous().view(torch.uint32)


# SMs of one GB10; a decode split-K grid aims to cover them once.
DECODE_CTAS = 48
DECODE_ROWS = 64  # rows served by the decode tile configuration


def bf16_shards(N: int, K: int, block_N: int = 64, block_K: int = 64) -> int:
    """K shards for a BF16 projection: the most that keep one CTA per SM.

    Few output tiles (the 384-expert router has six) leave most SMs idle and
    each CTA latency-bound on a long K loop; splitting K until tiles x shards
    reaches the SM count fixes both. Every row count uses the same split, so
    results stay batch-invariant.
    """
    if K % block_K:
        return 1
    tiles, k_blocks = tilelang.cdiv(N, block_N), K // block_K
    best = 1
    for shards in range(1, k_blocks // 2 + 1):
        if k_blocks % shards == 0 and tiles * shards <= DECODE_CTAS:
            best = shards
    return best


# Decode tile heights: rows up to DECODE_ROWS run on the shortest that holds
# them (one stream's verified rows fit 16; 32 rows hold four streams').
DECODE_TILE_ROWS = (16, 32, DECODE_ROWS)

# Decode tiles of the block-32 FP8 projections by (N, K) and tile height, from
# the kernel-lab sweep on GB10 at the TP4 serving shapes: whole calls with the
# activation cast against B12X's block_fp8_linear, with the weight warm in L2,
# cold, and racing its own L2 prefetch; each tile is the one whose worst ratio
# to B12X over the three is lowest. Every field only changes speed: each tile
# accumulates K in the same order, so a row's bits do not depend on the tile
# or on prefill.
DECODE_FP8_CONFIGS: dict[tuple[int, ...], dict[int, dict]] = {
    # TP4 attention: Q-B, indexer Q-B, fused Q-A/KV.
    (8192, 1280): {
        16: dict(block_N=128, block_K=128, num_stages=3),
        32: dict(block_N=64, block_K=128, num_stages=3),
        64: dict(block_N=64, block_K=256, num_stages=2),
    },
    (4096, 1280): {
        16: dict(block_N=64, block_K=256, num_stages=4),
        32: dict(block_N=128, block_K=128, num_stages=3),
        64: dict(block_N=64, block_K=128, num_stages=2, threads=64),
    },
    (1792, 5120): {
        16: dict(block_N=64, block_K=256, num_stages=2),
        32: dict(block_N=64, block_K=256, num_stages=3),
        64: dict(block_N=64, block_K=256, num_stages=2),
    },
    # TP4 shared expert: gate/up and down.
    (1152, 5120): {
        16: dict(block_N=32, block_K=256, num_stages=3),
        32: dict(block_N=32, block_K=256, num_stages=3),
        64: dict(block_N=64, block_K=256, num_stages=2),
    },
    (5120, 576): {
        16: dict(block_N=128, block_K=64, num_stages=2, threads=64),
        32: dict(block_N=128, block_K=64, num_stages=2),
        64: dict(block_N=64, block_K=64, num_stages=2),
    },
    # TP4 attention output: the grouped WO-A (2 groups) and WO-B (window 3's WO
    # sweep, ranked against the default tile; WO-B keeps it from 32 rows).
    (1024, 4096, 2): {
        16: dict(block_N=64, block_K=128, num_stages=4),
        32: dict(block_N=64, block_K=128, num_stages=4),
        64: dict(block_N=64, block_K=256, num_stages=2),
    },
    (5120, 2048): {
        16: dict(block_N=128, block_K=128, num_stages=3),
    },
    # TP4 DSpark main projection.
    (6400, 6144): {
        16: dict(block_N=32, block_K=256, num_stages=4, threads=64),
        32: dict(block_N=32, block_K=256, num_stages=4, threads=64),
        64: dict(block_N=64, block_K=256, num_stages=2),
    },
}


# Narrow projections leave most SMs idle at decode (512 columns are 8 tiles of 64
# on 48 SMs). They split K for decode rows up to SPLIT_DECODE_ROWS, write FP32
# partials and add them in shard order; their prefill GEMM accumulates the same
# shards in the same order, so a row's bits do not depend on the batch. Shard
# counts and tiles come from the kernel lab against B12X at the TP4 shapes.
SPLIT_DECODE_ROWS = 128
SPLIT_FP8: dict[tuple[int, int], dict] = {
    # TP4 DSpark context KV: the fused Q-A/KV weight's KV rows. Decode tiles by row
    # bucket (128: rows 65-128 in 64-row tiles); window 2's sweep winners.
    (512, 5120): dict(
        shards=5,
        rows=SPLIT_DECODE_ROWS,
        decode={
            16: dict(block_N=32, block_K=256, num_stages=3, threads=128),
            32: dict(block_N=32, block_K=256, num_stages=2, threads=128),
            64: dict(block_N=32, block_K=128, num_stages=3, threads=128),
            128: dict(block_N=64, block_K=128, num_stages=2, threads=128),
        },
        prefill=dict(block_M=128, block_N=64, block_K=128, num_stages=2),
    ),
}
SPLIT_BF16: dict[tuple[int, int], dict] = {
    # TP4 compressor wkv + wgate (ratio 2, one launch for both) and wkv (ratio 1):
    # five shards are fastest; blocked accumulation keeps the FP32 error under
    # DeepSeek's FP32 reference. An entry's optional "decode" maps a row tile
    # (PARTIAL_ROW_TILES) to the one-launch split-K's block_N, num_stages and
    # threads, which only change speed; block_K stays 64, since blocked
    # accumulation adds one block at a time.
    # Decode tiles: window 5's sweep, the lowest worst warm/cold ratio to B12X.
    (1024, 5120): dict(
        shards=5,
        blocked=True,
        decode={
            16: dict(block_N=64, num_stages=4, threads=128),
            32: dict(block_N=64, num_stages=2, threads=256),
            64: dict(block_N=64, num_stages=3, threads=256),
        },
    ),
    (512, 5120): dict(
        shards=5,
        blocked=True,
        decode={
            16: dict(block_N=32, num_stages=3, threads=128),
            32: dict(block_N=32, num_stages=3, threads=128),
            64: dict(block_N=32, num_stages=2, threads=256),
        },
    ),
}


def split_bucket(rows: int) -> int:
    """Decode bucket of a split-K call: 16, 32 or 64 rows (one tile that tall), or
    128 (rows 65-128 in 64-row tiles)."""
    return next(height for height in (16, 32, 64, 128) if rows <= height)


def split_row_tile(rows: int) -> int:
    """Row tile of a split-K decode GEMM: the shortest that holds ``rows``, else 64."""
    return min(split_bucket(rows), 64) if rows <= 128 else 64


# Block-32 FP8 projections whose decode tiles also serve rows past DECODE_ROWS
# (16-stream steps reach 96 rows), by (N, K): {up to this many rows: tile height}.
# Others take their prefill tile past DECODE_ROWS. Only speed changes: every tile
# adds K in the same order. From the kernel lab's decode-rows bundle (2026-10-10,
# whole calls warm, cold and after their own prefetch): the fused Q-A/KV and the
# shared experts' gate/up and down; Q-B, the indexer's Q-B and the DSpark main
# projection run faster on their prefill tiles.
WIDE_DECODE: dict[tuple[int, int], dict[int, int]] = {
    (1792, 5120): {96: 32, 128: 64},
    (1152, 5120): {128: 64},
    (5120, 576): {96: 32},
}

# BF16 projections whose split-K decode path serves rows past DECODE_ROWS, by
# (N, K): the most rows it serves. The partials plus their reduce add the same
# shards in the same order as the shard GEMM, so the bits do not change. The
# router (8 shards) splits faster up to 256 rows, the indexer's head weights (40
# shards) up to 1024 (decode-rows bundle, 2026-10-10).
BF16_SPLIT_ROWS: dict[tuple[int, int], int] = {(384, 5120): 256, (32, 5120): 1024}


def decode_tile_rows(rows: int, wide: dict[int, int] | None = None) -> int:
    """Height of the decode tile that serves ``rows``. Rows past DECODE_ROWS run in
    whole tiles of the layer's ``wide`` height for them (``WIDE_DECODE``), else of
    DECODE_ROWS (a layer that allows them)."""
    if rows <= DECODE_ROWS:
        return next(height for height in DECODE_TILE_ROWS if rows <= height)
    return next(
        (tile for bound, tile in sorted((wide or {}).items()) if rows <= bound),
        DECODE_ROWS,
    )


def fp8_decode_config(
    N: int, K: int, block_M: int = DECODE_ROWS, groups: int = 1
) -> dict:
    """Decode tile of an (N, K) block-32 FP8 projection, ``block_M`` rows tall;
    grouped projections (``groups`` blocks of N rows) are keyed (N, K, groups)."""
    config = dict(
        block_M=block_M, block_N=64, block_K=pick_block_K(K), num_stages=3, threads=128
    )
    key = (N, K) if groups == 1 else (N, K, groups)
    config.update(DECODE_FP8_CONFIGS.get(key, {}).get(block_M, {}))
    return config


def fp8_prefill_config(N: int, K: int, groups: int = 1) -> dict:
    """Prefill tile of an (N, K) block-32 FP8 projection.

    A weight larger than L2 (the DSpark main projection, 39 MB at TP4) is
    re-read from DRAM by every row of tiles in launch order; panels of eight N
    tiles made a 33 MB projection's 4096-row GEMM three times faster. Smaller
    weights stay in L2 either way and run fastest unswizzled.
    """
    device = torch.cuda.current_device()
    l2_bytes = torch.cuda.get_device_properties(device).L2_cache_size
    swizzle_panel = 8 if l2_bytes < groups * N * K else 0
    return dict(default_config(DECODE_ROWS + 1, K), swizzle_panel=swizzle_panel)


def default_config(M: int, K: int) -> dict:
    """Tile choice by row count: decode needs more, smaller N tiles to fill the GPU.

    The large tiles keep K blocks of 128 or 64 to stay within 99 KiB of
    shared memory. Block-32 FP8 decode tiles come from ``fp8_decode_config``.
    """
    if M <= DECODE_ROWS:
        return dict(
            block_M=DECODE_ROWS, block_N=64, block_K=pick_block_K(K), num_stages=3
        )
    block_K = pick_block_K(K, 128)
    return dict(block_M=128, block_N=128, block_K=block_K, num_stages=2)
