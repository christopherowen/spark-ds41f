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
    """
    assert K % block_K == 0 and block_K % 64 == 0, (
        "block_K must divide K and be a multiple of 64"
    )
    k_blocks = K // block_K
    M = T.dynamic("M")
    Mp = T.dynamic("Mp") if padded_rows else M
    in_dtype = T.float8_e4m3fn
    accum_dtype = T.float32
    sf_words = scale_words(K)
    block_words = block_scale_words(block_K)

    @T.prim_func
    def main(
        A: T.Tensor((Mp, K), in_dtype),
        B: T.Tensor((N, K), in_dtype),
        SFA: T.Tensor((Mp, sf_words), T.uint32),
        SFB: T.Tensor((N, sf_words), T.uint32),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads
        ) as (bx, by):
            if swizzle_panel:
                T.use_swizzle(panel_size=swizzle_panel)
            A_shared = T.alloc_shared((block_M, block_K), in_dtype)
            B_shared = T.alloc_shared((block_N, block_K), in_dtype)
            SFA_shared = T.alloc_shared((block_M, block_words), T.uint32)
            SFB_shared = T.alloc_shared((block_N, block_words), T.uint32)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for ko in T.Pipelined(k_blocks, num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[bx * block_N, ko * block_K], B_shared)
                # Edge tiles clamp their scale rows; those rows are never stored.
                word0 = ko * block_K // SF_WORD_K
                for i, w in T.Parallel(block_M, block_words):
                    SFA_shared[i, w] = SFA[
                        T.min(by * block_M + i, Mp - 1),
                        T.min(word0 + w, sf_words - 1),
                    ]
                for j, w in T.Parallel(block_N, block_words):
                    SFB_shared[j, w] = SFB[
                        T.min(bx * block_N + j, N - 1), T.min(word0 + w, sf_words - 1)
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
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


# Prefill tiles use TileLang's warp-specialized pipeline (TMA producer warps,
# up to 17% faster there). Decode tiles do not: with warp specialization the
# cp.async scale staging races on SM121 at decode tile shapes (16-row tiles
# returned wrong results in up to every run; 64-row tiles were not seen to),
# while the unspecialized pipeline is race-free and as fast for decode. Both
# run the same arithmetic, so their bits agree.
mxfp8_gemm = tilelang.jit(_mxfp8_gemm)
mxfp8_gemm_decode = tilelang.jit(
    pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True}
)(_mxfp8_gemm)


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
            C_total = T.alloc_fragment(
                (block_M, block_N) if shards > 1 else (1, 1), T.float32
            )
            T.clear(C_local)
            if shards > 1:
                T.clear(C_total)
            for ko in T.Pipelined(k_blocks, num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[bx * block_N, ko * block_K], B_shared)
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
def bf16_gemm_partials(
    N: int,
    K: int,
    shards: int,
    block_M: int = 64,
    block_N: int = 64,
    block_K: int = 64,
    num_stages: int = 3,
    threads: int = 128,
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
            T.clear(C_local)
            for kb in T.Pipelined(shard_blocks, num_stages=num_stages):
                ko = s * shard_blocks + kb
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[bx * block_N, ko * block_K], B_shared)
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
DECODE_FP8_CONFIGS: dict[tuple[int, int], dict[int, dict]] = {
    # TP4 attention: Q-B, indexer Q-B, fused Q-A/KV.
    (8192, 1280): {
        16: dict(block_N=64, block_K=128, num_stages=3),
        32: dict(block_N=64, block_K=128, num_stages=3),
        64: dict(block_N=64, block_K=256, num_stages=2),
    },
    (4096, 1280): {
        16: dict(block_N=32, block_K=256, num_stages=3, threads=64),
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
    # TP4 DSpark main projection.
    (6400, 6144): {
        16: dict(block_N=32, block_K=256, num_stages=4, threads=64),
        32: dict(block_N=32, block_K=256, num_stages=4, threads=64),
        64: dict(block_N=64, block_K=256, num_stages=2),
    },
}


def decode_tile_rows(rows: int) -> int:
    """Height of the decode tile that serves ``rows`` (at most DECODE_ROWS)."""
    return next(height for height in DECODE_TILE_ROWS if rows <= height)


def fp8_decode_config(N: int, K: int, block_M: int = DECODE_ROWS) -> dict:
    """Decode tile of an (N, K) block-32 FP8 projection, ``block_M`` rows tall."""
    config = dict(
        block_M=block_M, block_N=64, block_K=pick_block_K(K), num_stages=3, threads=128
    )
    config.update(DECODE_FP8_CONFIGS.get((N, K), {}).get(block_M, {}))
    return config


def fp8_prefill_config(N: int, K: int) -> dict:
    """Prefill tile of an (N, K) block-32 FP8 projection.

    A weight larger than L2 (the DSpark main projection, 39 MB at TP4) is
    re-read from DRAM by every row of tiles in launch order; panels of eight N
    tiles made a 33 MB projection's 4096-row GEMM three times faster. Smaller
    weights stay in L2 either way and run fastest unswizzled.
    """
    device = torch.cuda.current_device()
    l2_bytes = torch.cuda.get_device_properties(device).L2_cache_size
    swizzle_panel = 8 if l2_bytes < N * K else 0
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
