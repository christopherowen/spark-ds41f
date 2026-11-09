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
K block ``4 * word + j``). The weight's 32x32 block scales are stored as that
packed grid; the GEMM loads ``SFB[row // 32]``, the word a per-row broadcast
would have copied to that row. Activation scales stay one word per row.
"""

import tilelang
import tilelang.language as T
import torch

from . import pdl
from .scale_codes import (
    expand_scale_codes as expand_scale_codes,
    expand_scale_codes_2bit as expand_scale_codes_2bit,
    pack_expert_scale_codes as pack_expert_scale_codes,
    pack_scale_codes as pack_scale_codes,
    pack_scale_codes_2bit as pack_scale_codes_2bit,
)
from .scale_pack import pack_scale_words

SF_BLOCK = 32  # K elements per UE8M0 scale
SF_WORD_K = 4 * SF_BLOCK  # K elements per packed scale word
TILE_WEIGHT_N = 128  # N rows per tile-contiguous FP8 weight block
TILE_WEIGHT_K = 128  # K columns per tile-contiguous FP8 weight block


def pack_weight_tiles(
    weight: torch.Tensor,
    tile_n: int = TILE_WEIGHT_N,
    tile_k: int = TILE_WEIGHT_K,
) -> torch.Tensor:
    """Pack row-major ``[N, K]`` FP8 weight into ``[N // tile_n, K // tile_k, tile_n, tile_k]``
    viewed as ``(N * (K // tile_k), tile_k)``, so each ``block_N x 128`` tile (for
    ``block_N`` dividing ``tile_n``) is a single contiguous slab in DRAM."""
    n, k = weight.shape
    if n % tile_n or k % tile_k:
        raise ValueError(
            f"weight shape {(n, k)} must be divisible by tile {(tile_n, tile_k)}"
        )
    return (
        weight.reshape(n // tile_n, tile_n, k // tile_k, tile_k)
        .permute(0, 2, 1, 3)
        .contiguous()
        .view(n * (k // tile_k), tile_k)
    )


def unpack_weight_tiles(
    packed: torch.Tensor,
    n: int,
    k: int,
    tile_n: int = TILE_WEIGHT_N,
    tile_k: int = TILE_WEIGHT_K,
) -> torch.Tensor:
    """Inverse of ``pack_weight_tiles``: restore row-major ``[N, K]`` FP8 weight."""
    return (
        packed.reshape(n // tile_n, k // tile_k, tile_n, tile_k)
        .permute(0, 2, 1, 3)
        .contiguous()
        .view(n, k)
    )


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
    pdl_chain: bool = False,
    tile_contiguous: bool = False,
):
    """``C[M, N] = (A * SFA) @ (B * SFB)^T`` for MXFP8 ``A[M, K]``, ``B[N, K]``.

    M is dynamic. ``SFA`` holds ``ceil(K / 128)`` packed UE8M0 words per
    activation row. ``SFB`` holds those words once per 32 output rows (the
    packed 32x32 block-scale grid): row ``r`` reads ``SFB[r // 32]``, including
    inside a group, where groups are stacked in N. ``block_K`` is any multiple
    of 64 dividing K; a block that starts off a 128-element boundary reads the
    two words it straddles and tells the MMA where it starts (``k_start``), so
    no operand is padded.

    With ``padded_rows``, ``A`` and ``SFA`` have ``Mp >= M`` rows, a multiple
    of ``block_M`` (the decode workspace). Producers need write only the ``M``
    live rows: unused MMA lanes repeat the last live activation and scale row,
    and only live results are stored. No padded scratch contents are read.

    Each output accumulates over K in the same order whatever the tile shape
    or padding, so a row's result does not depend on M or the configuration.
    ``swizzle_panel`` rasterizes the CTAs in panels of that many N tiles, so a
    weight larger than L2 is read once per panel rather than once per row of
    tiles; it only changes the order tiles run in.

    With ``groups`` > 1 the projection is grouped: ``A`` is ``[M, groups * K]``,
    ``B`` stacks the groups' ``[N, K]`` weights, ``SFB`` stacks their block-scale
    rows (the same words as broadcasting each group), and ``C`` is
    ``[M, groups * N]``. Output columns of group ``g`` read only A's columns
    ``g * K`` to ``(g + 1) * K``. One launch covers every group. ``N`` is a
    multiple of 32, so a group starts on a block-scale row.

    With ``shards`` > 1 each of the ``shards`` equal K ranges is summed from
    zero and the shards are added in order: the arithmetic of
    ``mxfp8_gemm_partials`` plus ``splitk_reduce``, so a narrow projection's
    split-K decode rows give its prefill rows' bits.

    ``pdl_chain`` starts the kernel with ``pdl.chain()``: the decode tiles
    (``mxfp8_gemm_decode``), which compile without warp specialization, do; the
    warp-specialized prefill tiles do not.
    """
    assert K % block_K == 0 and block_K % 64 == 0, (
        "block_K must divide K and be a multiple of 64"
    )
    assert N % 32 == 0, "block scales are one row per 32 output rows"
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
    if tile_contiguous:
        assert total_N % TILE_WEIGHT_N == 0 and K % TILE_WEIGHT_K == 0
        assert TILE_WEIGHT_N % block_N == 0 and block_K % TILE_WEIGHT_K == 0
    k_tiles = K // TILE_WEIGHT_K
    sub_tiles_k = block_K // TILE_WEIGHT_K
    b_shape = (
        (total_N * k_tiles, TILE_WEIGHT_K) if tile_contiguous else (total_N, K)
    )

    @T.prim_func
    def gemm_mxfp8(
        A: T.Tensor((Mp, groups * K), in_dtype),
        B: T.Tensor(b_shape, in_dtype),
        SFA: T.Tensor((Mp, groups * sf_words), T.uint32),
        SFB: T.Tensor((total_N // 32, sf_words), T.uint32),
        C: T.Tensor((M, total_N), out_dtype),
    ):
        with T.Kernel(
            T.ceildiv(total_N, block_N), T.ceildiv(M, block_M), threads=threads
        ) as (bx, by):
            if swizzle_panel:
                T.use_swizzle(panel_size=swizzle_panel)
            if pdl_chain:
                pdl.chain()
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
                if padded_rows:
                    for i, k in T.Parallel(block_M, block_K):
                        A_shared[i, k] = A[
                            T.min(by * block_M + i, M - 1),
                            a_k + ko * block_K + k,
                        ]
                else:
                    T.copy(A[by * block_M, a_k + ko * block_K], A_shared)
                if not tile_contiguous:
                    T.copy(B[bx * block_N, ko * block_K], B_shared)
                elif sub_tiles_k == 1:
                    row0 = (
                        ((bx * block_N // TILE_WEIGHT_N) * k_tiles + ko) * TILE_WEIGHT_N
                        + (bx * block_N % TILE_WEIGHT_N)
                    )
                    T.copy(B[row0, 0], B_shared)
                else:
                    for sub_k in T.serial(sub_tiles_k):
                        row0 = (
                            (
                                (bx * block_N // TILE_WEIGHT_N) * k_tiles
                                + ko * sub_tiles_k
                                + sub_k
                            )
                            * TILE_WEIGHT_N
                            + (bx * block_N % TILE_WEIGHT_N)
                        )
                        T.copy(
                            B[row0 : row0 + block_N, 0:TILE_WEIGHT_K],
                            B_shared[
                                0:block_N,
                                sub_k * TILE_WEIGHT_K : (sub_k + 1) * TILE_WEIGHT_K,
                            ],
                        )
                # Edge tiles clamp their scale rows; those rows are never stored.
                # SFB is the packed block grid, so the clamped row is divided by
                # 32. That is the per-row broadcast's word, and it stays on the
                # last block row when the tile runs past N.
                word0 = ko * block_K // SF_WORD_K
                for i, w in T.Parallel(block_M, block_words):
                    SFA_shared[i, w] = SFA[
                        T.min(by * block_M + i, M - 1),
                        a_word + T.min(word0 + w, sf_words - 1),
                    ]
                for j, w in T.Parallel(block_N, block_words):
                    row = T.min(bx * block_N + j, total_N - 1)
                    SFB_shared[j, w] = SFB[
                        row // 32,
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

    return gemm_mxfp8


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
    workspace). Only live rows are read; unused MMA lanes repeat the last live
    row. A narrow projection has few N tiles; its shards fill the GPU.
    """
    assert K % block_K == 0 and block_K % 64 == 0 and (K // block_K) % shards == 0
    assert N % 32 == 0, "block scales are one row per 32 output rows"
    shard_blocks = K // block_K // shards
    M = T.dynamic("M")
    Mp = T.dynamic("Mp")
    in_dtype = T.float8_e4m3fn
    sf_words = scale_words(K)
    block_words = block_scale_words(block_K)

    @T.prim_func
    def gemm_mxfp8_partials(
        A: T.Tensor((Mp, K), in_dtype),
        B: T.Tensor((N, K), in_dtype),
        SFA: T.Tensor((Mp, sf_words), T.uint32),
        SFB: T.Tensor((N // 32, sf_words), T.uint32),
        P: T.Tensor((shards, M, N), T.float32),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N), shards, T.ceildiv(M, block_M), threads=threads
        ) as (bx, s, by):
            pdl.chain()
            A_shared = T.alloc_shared((block_M, block_K), in_dtype)
            B_shared = T.alloc_shared((block_N, block_K), in_dtype)
            SFA_shared = T.alloc_shared((block_M, block_words), T.uint32)
            SFB_shared = T.alloc_shared((block_N, block_words), T.uint32)
            C_local = T.alloc_fragment((block_M, block_N), T.float32)
            T.clear(C_local)
            for kb in T.Pipelined(shard_blocks, num_stages=num_stages):
                ko = s * shard_blocks + kb
                for i, k in T.Parallel(block_M, block_K):
                    A_shared[i, k] = A[
                        T.min(by * block_M + i, M - 1), ko * block_K + k
                    ]
                T.copy(B[bx * block_N, ko * block_K], B_shared)
                word0 = ko * block_K // SF_WORD_K
                for i, w in T.Parallel(block_M, block_words):
                    SFA_shared[i, w] = SFA[
                        T.min(by * block_M + i, M - 1),
                        T.min(word0 + w, sf_words - 1),
                    ]
                for j, w in T.Parallel(block_N, block_words):
                    # Same block-grid address as gemm_mxfp8; clamp, then // 32.
                    row = T.min(bx * block_N + j, N - 1)
                    SFB_shared[j, w] = SFB[
                        row // 32,
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

    return gemm_mxfp8_partials


def _mxfp8_gemm_decode(
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
    tile_contiguous: bool = False,
):
    """``_mxfp8_gemm``'s decode tiles: no warp specialization, on the PDL chain."""
    return _mxfp8_gemm(
        N,
        K,
        block_M,
        block_N,
        block_K,
        num_stages,
        threads,
        out_dtype,
        padded_rows,
        swizzle_panel,
        groups,
        shards,
        pdl_chain=True,
        tile_contiguous=tile_contiguous,
    )


mxfp8_gemm = tilelang.jit(_mxfp8_gemm)
mxfp8_gemm_decode = tilelang.jit(
    pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True}
)(_mxfp8_gemm_decode)
mxfp8_gemm_partials = tilelang.jit(
    pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True}
)(_mxfp8_gemm_partials)


_E2M1_HEADER = r"""
__device__ __forceinline__ uint8_t dsv41_e2m1x2(float v0, float v1) {
  unsigned int packed;
  asm("{\n"
      ".reg .b8 b;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b, %2, %1;\n"
      "cvt.u32.u8 %0, b;\n"
      "}"
      : "=r"(packed)
      : "f"(v0), "f"(v1));
  return (uint8_t)packed;
}

__device__ __forceinline__ uint32_t dsv41_e2m1x8(
    float v0, float v1, float v2, float v3,
    float v4, float v5, float v6, float v7) {
  uint32_t packed;
  asm("{\n"
      ".reg .b8 b0, b1, b2, b3;\n"
      ".reg .b16 w0, w1;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n"
      "mov.b16 w0, {b0, b1};\n"
      "mov.b16 w1, {b2, b3};\n"
      "mov.b32 %0, {w0, w1};\n"
      "}"
      : "=r"(packed)
      : "f"(v0), "f"(v1), "f"(v2), "f"(v3),
        "f"(v4), "f"(v5), "f"(v6), "f"(v7));
  return packed;
}

__device__ __forceinline__ void dsv41_store_e2m1x8(
    uint8_t* ptr,
    float v0, float v1, float v2, float v3,
    float v4, float v5, float v6, float v7) {
  *reinterpret_cast<uint32_t*>(ptr) = dsv41_e2m1x8(
      v0, v1, v2, v3, v4, v5, v6, v7);
}

__device__ __forceinline__ void dsv41_store_e2m1x32(
    uint8_t* ptr, uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3) {
  *reinterpret_cast<uint4*>(ptr) = make_uint4(w0, w1, w2, w3);
}
"""


def _e2m1x2_byte(v0, v1):
    """Two FP32 values rounded to nearest-even E2M1 (``cvt.rn.satfinite.e2m1x2.f32``), low nibble ``v0``."""
    return T.call_extern("uint8", "dsv41_e2m1x2", v0, v1)


def _mxfp8_gemm_rope_mxfp4_decode(
    N: int,
    K: int,
    head_dim: int = 128,
    rope_dim: int = 64,
    block_M: int = 16,
    block_N: int = 128,
    block_K: int = 128,
    num_stages: int = 2,
    threads: int = 256,
    padded_rows: bool = True,
    swizzle_panel: int = 0,
    tile_contiguous: bool = False,
    ratio: int = 1,
):
    """Decode MXFP8 GEMM fused with GPT-J interleaved RoPE and MXFP4 quantization.

    Computes ``C[M, N] = bf16((A * SFA) @ (B * SFB)^T)``, rotates the trailing
    ``rope_dim`` columns of each ``head_dim``-wide head in registers at
    ``Positions``, and quantizes each 32-element group to packed E2M1 ``Q``
    (``[M, N // 2]`` uint8) and UE8M0 scales ``S`` (``[M, N // 32]`` uint8),
    matching ``mxfp8_gemm_decode`` followed by ``mxfp4_quantize_v41`` bit for
    bit.

    ``C_shared`` aliases ``A_shared`` after the K loop (zero additional dynamic
    shared memory). In the epilogue, each 16-lane half-warp processes one
    32-column group in parallel (one pair of adjacent columns per lane),
    reducing ``amax`` across the 16 lanes via four ``__shfl_xor_sync`` steps.
    """
    assert K % block_K == 0 and block_K % 64 == 0, (
        "block_K must divide K and be a multiple of 64"
    )
    assert N % block_N == 0 and block_N % SF_BLOCK == 0, (
        "block_N must divide N and be a multiple of 32"
    )
    assert (
        head_dim % SF_BLOCK == 0
        and 0 < rope_dim <= head_dim
        and rope_dim % SF_BLOCK == 0
    ), "head_dim and rope_dim must be multiples of 32"
    pairs_per_row = block_N // 2
    assert (
        threads % pairs_per_row == 0
        and block_M % (threads // pairs_per_row) == 0
    ), "threads must cover whole rows of column pairs"
    k_blocks = K // block_K
    M = T.dynamic("M")
    Mp = T.dynamic("Mp") if padded_rows else M
    max_pos = T.dynamic("max_pos")
    in_dtype = T.float8_e4m3fn
    accum_dtype = T.float32
    sf_words = scale_words(K)
    block_words = block_scale_words(block_K)
    if tile_contiguous:
        assert N % TILE_WEIGHT_N == 0 and K % TILE_WEIGHT_K == 0
        assert TILE_WEIGHT_N % block_N == 0 and block_K % TILE_WEIGHT_K == 0
    k_tiles = K // TILE_WEIGHT_K
    sub_tiles_k = block_K // TILE_WEIGHT_K
    b_shape = (N * k_tiles, TILE_WEIGHT_K) if tile_contiguous else (N, K)
    half_n = N // 2
    scale_n = N // SF_BLOCK
    rope_start = head_dim - rope_dim
    rope_half = rope_dim // 2
    half = SF_BLOCK // 2
    rows_per_iter = threads // pairs_per_row
    iters = block_M // rows_per_iter

    @T.prim_func
    def gemm_mxfp8_rope_mxfp4(
        A: T.Tensor((Mp, K), in_dtype),
        B: T.Tensor(b_shape, in_dtype),
        SFA: T.Tensor((Mp, sf_words), T.uint32),
        SFB: T.Tensor((N // 32, sf_words), T.uint32),
        Positions: T.Tensor((M,), T.int64),
        CosSin: T.Tensor((max_pos, rope_dim), T.float32),
        Q: T.Tensor((M, half_n), T.uint8),
        S: T.Tensor((M, scale_n), T.uint8),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads
        ) as (bx, by):
            T.import_source(_E2M1_HEADER)
            if swizzle_panel:
                T.use_swizzle(panel_size=swizzle_panel)
            pdl.chain()
            tx = T.get_thread_binding()
            A_shared = T.alloc_shared((block_M, block_K), in_dtype)
            B_shared = T.alloc_shared((block_N, block_K), in_dtype)
            SFA_shared = T.alloc_shared((block_M, block_words), T.uint32)
            SFB_shared = T.alloc_shared((block_N, block_words), T.uint32)
            C_shared = T.alloc_shared((block_M, block_N), T.bfloat16)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for ko in T.Pipelined(k_blocks, num_stages=num_stages):
                if padded_rows:
                    for i, k in T.Parallel(block_M, block_K):
                        A_shared[i, k] = A[
                            T.min(by * block_M + i, M - 1),
                            ko * block_K + k,
                        ]
                else:
                    T.copy(A[by * block_M, ko * block_K], A_shared)
                if not tile_contiguous:
                    T.copy(B[bx * block_N, ko * block_K], B_shared)
                elif sub_tiles_k == 1:
                    row0 = (
                        ((bx * block_N // TILE_WEIGHT_N) * k_tiles + ko) * TILE_WEIGHT_N
                        + (bx * block_N % TILE_WEIGHT_N)
                    )
                    T.copy(B[row0, 0], B_shared)
                else:
                    for sub_k in T.serial(sub_tiles_k):
                        row0 = (
                            (
                                (bx * block_N // TILE_WEIGHT_N) * k_tiles
                                + ko * sub_tiles_k
                                + sub_k
                            )
                            * TILE_WEIGHT_N
                            + (bx * block_N % TILE_WEIGHT_N)
                        )
                        T.copy(
                            B[row0 : row0 + block_N, 0:TILE_WEIGHT_K],
                            B_shared[
                                0:block_N,
                                sub_k * TILE_WEIGHT_K : (sub_k + 1) * TILE_WEIGHT_K,
                            ],
                        )
                word0 = ko * block_K // SF_WORD_K
                for i, w in T.Parallel(block_M, block_words):
                    SFA_shared[i, w] = SFA[
                        T.min(by * block_M + i, M - 1),
                        T.min(word0 + w, sf_words - 1),
                    ]
                for j, w in T.Parallel(block_N, block_words):
                    row = T.min(bx * block_N + j, N - 1)
                    SFB_shared[j, w] = SFB[
                        row // 32,
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

            T.sync_threads()
            T.copy(C_local, C_shared)
            col_pair = tx % pairs_per_row
            row_in_iter = tx // pairs_per_row
            g = col_pair // half
            p = tx % half
            col0 = bx * block_N + g * SF_BLOCK
            in_head = col0 % head_dim
            c0 = 2 * col_pair

            v0 = T.alloc_var(T.float32)
            v1 = T.alloc_var(T.float32)
            pos = T.alloc_var(T.int64)
            cosine = T.alloc_var(T.float32)
            sine = T.alloc_var(T.float32)
            amax = T.alloc_var(T.float32)
            bits = T.alloc_var(T.uint32)
            sf = T.alloc_var(T.int32)
            scale = T.alloc_var(T.float32)
            sv0 = T.alloc_var(T.float32)
            sv1 = T.alloc_var(T.float32)

            # A plain loop: nvcc unrolls it now that the E2M1 body is short. Kept
            # rolled (#pragma unroll 1), each row pays the loop's counter, pointer
            # and branch path, which at block_M=16 costs about the conversion's
            # whole gain at M=1..4 (GB10 kernel time).
            for it in T.serial(iters):
                row_local = it * rows_per_iter + row_in_iter
                m = by * block_M + row_local
                if m < M:
                    v0 = T.cast(C_shared[row_local, c0], T.float32)
                    v1 = T.cast(C_shared[row_local, c0 + 1], T.float32)
                    if in_head >= rope_start:
                        pos = (
                            Positions[m] // ratio * ratio
                            if ratio > 1
                            else Positions[m]
                        )
                        pair_base = (in_head - rope_start) // 2
                        cosine = CosSin[pos, pair_base + p]
                        sine = CosSin[pos, rope_half + pair_base + p]
                        r0 = T.cast(T.ieee_fmaf(v0, cosine, -(v1 * sine)), T.bfloat16)
                        r1 = T.cast(T.ieee_fmaf(v0, sine, v1 * cosine), T.bfloat16)
                        v0 = T.cast(r0, T.float32)
                        v1 = T.cast(r1, T.float32)
                    amax = T.max(
                        T.float32(6.0 * 2.0**-126),
                        T.max(T.abs(v0), T.abs(v1)),
                    )
                    amax = T.max(amax, T.shfl_xor(amax, 8))
                    amax = T.max(amax, T.shfl_xor(amax, 4))
                    amax = T.max(amax, T.shfl_xor(amax, 2))
                    amax = T.max(amax, T.shfl_xor(amax, 1))

                    bits = T.reinterpret(T.fmul(amax, T.float32(1.0 / 6.0)), T.uint32)
                    sf = T.cast(T.shift_right(bits, 23) & 0xFF, T.int32) + T.cast(
                        (bits & 0x7FFFFF) != 0, T.int32
                    )
                    scale = T.reinterpret(
                        T.cast(T.shift_left(254 - sf, 23), T.uint32), T.float32
                    )
                    sv0 = T.fmul(v0, scale)
                    sv1 = T.fmul(v1, scale)
                    Q[m, col0 // 2 + p] = _e2m1x2_byte(sv0, sv1)
                    if p == 0:
                        S[m, col0 // SF_BLOCK] = T.cast(sf, T.uint8)

    return gemm_mxfp8_rope_mxfp4


mxfp8_gemm_rope_mxfp4_decode = tilelang.jit(
    pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True}
)(_mxfp8_gemm_rope_mxfp4_decode)


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
    def gemm_splitk_reduce(
        P: T.Tensor((shards, M, N), T.float32), C: T.Tensor((M, N), out_dtype)
    ):
        with T.Kernel(T.ceildiv(N, block_N), M, threads=threads) as (bx, m):
            pdl.chain()
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

    return gemm_splitk_reduce


# DeepSeek's act_quant(x, 32, "ue8m0") as TileKernels' per_token_cast computes it
# (byte-identical: experiments/2026-10-13-tilelang-wo/check_cast_reference.py).
_FP8_MAX = 448.0
_FP8_MAX_INV = 1 / _FP8_MAX
_AMAX_FLOOR = 1e-4


@T.macro
def _reference_cast(X, amax, inv, Q, S, row0, col0, rows, blocks, M):
    """Quantize ``X[r, e]`` (FP32 copies of BF16 values; row ``r`` is 32-element block
    ``r % blocks`` of output row ``row0 + r // blocks``) as act_quant: amax over the
    block floored at 1e-4, the exponent ceil(log2(amax / 448)) from ``amax * (1/448)``'s
    bits, E4M3 of ``x * 2^-e``, and the UE8M0 byte ``e + 127`` into ``S``."""
    T.reduce_absmax(X, amax, dim=1)
    for r in T.Parallel(rows * blocks):
        bits = T.reinterpret(
            T.max(amax[r], T.float32(_AMAX_FLOOR)) * T.float32(_FP8_MAX_INV), T.uint32
        )
        exponent = T.Cast(T.int32, ((bits - T.uint32(1)) >> 23) + T.uint32(1)) - 127
        inv[r] = T.reinterpret(T.Cast(T.uint32, 127 - exponent) << 23, T.float32)
        if row0 + r // blocks < M:
            S[row0 + r // blocks, col0 // 32 + r % blocks] = T.Cast(
                T.uint8, exponent + 127
            )
    for r, e in T.Parallel(rows * blocks, 32):
        if row0 + r // blocks < M:
            Q[row0 + r // blocks, col0 + (r % blocks) * 32 + e] = T.Cast(
                T.float8_e4m3fn, X[r, e] * inv[r]
            )


@T.macro
def _reference_roundtrip(A_shared, A_q, X, amax, inv, scale, block_M, block_K):
    """``A_q`` = the ``[block_M, block_K]`` BF16 tile ``A_shared`` through the
    reference's ``act_quant(x, 32, "ue8m0", inplace=True)``: per 32-element block of
    each row, the E4M3 of ``x * 2^-e`` (``e`` as ``_reference_cast`` derives it)
    times ``2^e``, which BF16 holds exactly. ``A_q`` may be ``A_shared``."""
    blocks = block_K // 32
    for r, e in T.Parallel(block_M * blocks, 32):
        X[r, e] = T.Cast(T.float32, A_shared[r // blocks, (r % blocks) * 32 + e])
    T.reduce_absmax(X, amax, dim=1)
    for r in T.Parallel(block_M * blocks):
        floored = T.max(amax[r], T.float32(_AMAX_FLOOR))
        bits = T.reinterpret(floored * T.float32(_FP8_MAX_INV), T.uint32)
        exponent = T.Cast(T.int32, ((bits - T.uint32(1)) >> 23) + T.uint32(1)) - 127
        inv[r] = T.reinterpret(T.Cast(T.uint32, 127 - exponent) << 23, T.float32)
        scale[r] = T.reinterpret(T.Cast(T.uint32, 127 + exponent) << 23, T.float32)
    for r, e in T.Parallel(block_M * blocks, 32):
        A_q[r // blocks, (r % blocks) * 32 + e] = T.Cast(
            T.bfloat16,
            T.Cast(T.float32, T.Cast(T.float8_e4m3fn, X[r, e] * inv[r])) * scale[r],
        )


@T.macro
def cast_block_mxfp8(values, quantized, count, amax, Q, S, row, col):
    """One thread's share of a 32-value block in registers through ``_reference_cast``'s
    arithmetic: ``values[0:count]`` (FP32 copies of BF16 values) are columns ``col`` on
    of output row ``row``, ``amax`` the whole block's (the caller reduces it across the
    block's threads). Their E4M3 bytes go through ``quantized`` (local, ``count``
    long) to ``Q`` and, from the thread whose columns start the block, the UE8M0 byte
    to ``S``: the bytes ``per_token_cast`` writes. A producer whose output is in
    registers writes its consumers' MXFP8 input with it."""
    bits = T.reinterpret(
        T.max(amax, T.float32(_AMAX_FLOOR)) * T.float32(_FP8_MAX_INV), T.uint32
    )
    exponent = T.Cast(T.int32, ((bits - T.uint32(1)) >> 23) + T.uint32(1)) - 127
    inv = T.reinterpret(T.Cast(T.uint32, 127 - exponent) << 23, T.float32)
    if col % 32 == 0:
        S[row, col // 32] = T.Cast(T.uint8, exponent + 127)
    for v in T.vectorized(count):
        quantized[v] = T.Cast(T.float8_e4m3fn, values[v] * inv)
    for v in T.vectorized(count):
        Q[row, col + v] = quantized[v]


@tilelang.jit
def splitk_reduce_cast(N: int, shards: int, block_M: int = 4, threads: int = 128):
    """``splitk_reduce`` to BF16, then the reference's activation cast: E4M3 ``Q`` and
    one UE8M0 byte per 32 columns in ``S``, the bytes of BF16 ``C`` through
    ``per_token_cast``. WO-A's split decode path hands WO-B its MXFP8 input directly."""
    assert N % 128 == 0
    M = T.dynamic("M")
    blocks = 4  # 128 columns per CTA

    @T.prim_func
    def gemm_splitk_reduce_cast(
        P: T.Tensor((shards, M, N), T.float32),
        Q: T.Tensor((M, N), T.float8_e4m3fn),
        S: T.Tensor((M, N // 32), T.uint8),
    ):
        with T.Kernel(N // 128, T.ceildiv(M, block_M), threads=threads) as (bx, by):
            pdl.chain()
            X = T.alloc_fragment((block_M * blocks, 32), T.float32)
            amax = T.alloc_fragment((block_M * blocks,), T.float32)
            inv = T.alloc_fragment((block_M * blocks,), T.float32)
            for r, e in T.Parallel(block_M * blocks, 32):
                m = T.min(by * block_M + r // blocks, M - 1)
                n = bx * 128 + (r % blocks) * 32 + e
                X[r, e] = P[0, m, n]
            for s in T.serial(1, shards):
                for r, e in T.Parallel(block_M * blocks, 32):
                    m = T.min(by * block_M + r // blocks, M - 1)
                    X[r, e] += P[s, m, bx * 128 + (r % blocks) * 32 + e]
            for r, e in T.Parallel(block_M * blocks, 32):
                X[r, e] = T.Cast(T.float32, T.Cast(T.bfloat16, X[r, e]))
            _reference_cast(
                X, amax, inv, Q, S, by * block_M, bx * 128, block_M, blocks, M
            )

    return gemm_splitk_reduce_cast


def _bf16_gemm(
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
    def gemm_bf16(
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

    return gemm_bf16


bf16_gemm = tilelang.jit(_bf16_gemm)
bf16_gemm_decode = tilelang.jit(
    pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True}
)(_bf16_gemm)


# Not warp-specialized: a producer warp's scale stores would reach the consumers only
# through the copy barriers, which order the asynchronous copies alone (graph replays
# gave other bits).
@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def fp8w_bf16_gemm(
    N: int,
    K: int,
    groups: int = 1,
    block_M: int = 16,
    block_N: int = 32,
    block_K: int = 64,
    num_stages: int = 2,
    threads: int = 128,
    shards: int = 1,
):
    """``C[:, g*N:(g+1)*N] = A[:, g*K:(g+1)*K] @ W_g^T`` for every group ``g`` in one
    launch: BF16 activations ``A[M, groups*K]`` against block-32 FP8 weights
    ``W[groups*N, K]`` (E4M3, one UE8M0 exponent per 32x32 block in ``E``), each weight
    tile expanded exactly to BF16 (an E4M3 value times a power of two fits BF16) and
    multiplied on BF16 tensor cores with FP32 accumulation; BF16 out. This is DeepSeek's
    reference WO-A: its BF16 einsum on the FP8 weights ``convert.py`` dequantizes, of
    activations already through ``act_quant`` (the sparse MLA kernels store them so,
    ``sparse_mla.wo_input``).

    Every M tile runs the same K loop, so a row's bits do not depend on its batch. With
    ``shards`` > 1 each K shard is summed from zero and the shards added in order, the
    arithmetic of ``fp8w_bf16_gemm_partials`` plus ``splitk_reduce``.
    """
    # Whole 32x32 scale blocks per tile (a 16-wide tile had no scale rows).
    assert K % block_K == 0 and block_K % 32 == 0 and N % block_N == 0
    assert block_N % 32 == 0
    k_blocks = K // block_K
    assert k_blocks % shards == 0
    shard_blocks = k_blocks // shards
    M = T.dynamic("M")
    n_tiles = N // block_N

    @T.prim_func
    def gemm_fp8w_bf16(
        A: T.Tensor((M, groups * K), T.bfloat16),
        W: T.Tensor((groups * N, K), T.float8_e4m3fn),
        E: T.Tensor((groups * N // 32, K // 32), T.uint8),
        C: T.Tensor((M, groups * N), T.bfloat16),
    ):
        with T.Kernel(groups * n_tiles, T.ceildiv(M, block_M), threads=threads) as (
            bx,
            by,
        ):
            pdl.chain()
            g = bx // n_tiles
            n0 = g * N + (bx % n_tiles) * block_N
            A_shared = T.alloc_shared((block_M, block_K), T.bfloat16)
            W_shared = T.alloc_shared((block_N, block_K), T.float8_e4m3fn)
            S_shared = T.alloc_shared((block_N // 32, block_K // 32), T.float32)
            B_shared = T.alloc_shared((block_N, block_K), T.bfloat16)
            C_local = T.alloc_fragment((block_M, block_N), T.float32)
            C_total = T.alloc_fragment(
                (block_M, block_N) if shards > 1 else (1, 1), T.float32
            )
            T.clear(C_local)
            if shards > 1:
                T.clear(C_total)
            for ko in T.Pipelined(k_blocks, num_stages=num_stages):
                _w8a16_step(A, W, E, A_shared, W_shared, S_shared, B_shared, C_local,
                            by, g, n0, ko, K, block_M, block_N, block_K)  # fmt: skip
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
                T.copy(C_total, C[by * block_M, n0])
            else:
                T.copy(C_local, C[by * block_M, n0])

    return gemm_fp8w_bf16


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def fp8w_bf16_gemm_cast(
    N: int,
    K: int,
    groups: int = 1,
    block_M: int = 16,
    block_N: int = 32,
    block_K: int = 64,
    num_stages: int = 2,
    threads: int = 128,
    shards: int = 1,
):
    """``fp8w_bf16_gemm`` whose epilogue rounds each output tile to BF16 and applies
    the reference's activation cast (``_reference_cast``): E4M3 ``Q`` and UE8M0 bytes
    ``S``, the bytes ``per_token_cast`` gives the BF16 output, without writing it."""
    assert K % block_K == 0 and block_K % 32 == 0 and N % block_N == 0
    assert block_N % 32 == 0
    k_blocks = K // block_K
    assert k_blocks % shards == 0
    shard_blocks = k_blocks // shards
    M = T.dynamic("M")
    n_tiles = N // block_N
    blocks = block_N // 32

    @T.prim_func
    def gemm_fp8w_bf16_cast(
        A: T.Tensor((M, groups * K), T.bfloat16),
        W: T.Tensor((groups * N, K), T.float8_e4m3fn),
        E: T.Tensor((groups * N // 32, K // 32), T.uint8),
        Q: T.Tensor((M, groups * N), T.float8_e4m3fn),
        S: T.Tensor((M, groups * N // 32), T.uint8),
    ):
        with T.Kernel(groups * n_tiles, T.ceildiv(M, block_M), threads=threads) as (
            bx,
            by,
        ):
            pdl.chain()
            g = bx // n_tiles
            n0 = g * N + (bx % n_tiles) * block_N
            A_shared = T.alloc_shared((block_M, block_K), T.bfloat16)
            W_shared = T.alloc_shared((block_N, block_K), T.float8_e4m3fn)
            S_shared = T.alloc_shared((block_N // 32, block_K // 32), T.float32)
            B_shared = T.alloc_shared((block_N, block_K), T.bfloat16)
            C_shared = T.alloc_shared((block_M, block_N), T.bfloat16)
            C_local = T.alloc_fragment((block_M, block_N), T.float32)
            C_total = T.alloc_fragment(
                (block_M, block_N) if shards > 1 else (1, 1), T.float32
            )
            X = T.alloc_fragment((block_M * blocks, 32), T.float32)
            amax = T.alloc_fragment((block_M * blocks,), T.float32)
            inv = T.alloc_fragment((block_M * blocks,), T.float32)
            T.clear(C_local)
            if shards > 1:
                T.clear(C_total)
            for ko in T.Pipelined(k_blocks, num_stages=num_stages):
                _w8a16_step(A, W, E, A_shared, W_shared, S_shared, B_shared, C_local,
                            by, g, n0, ko, K, block_M, block_N, block_K)  # fmt: skip
                if shards > 1:  # noqa: SIM102
                    if (ko + 1) % shard_blocks == 0:
                        for i, j in T.Parallel(block_M, block_N):
                            C_total[i, j] = T.if_then_else(
                                ko + 1 == shard_blocks,
                                C_local[i, j],
                                C_total[i, j] + C_local[i, j],
                            )
                        T.clear(C_local)
            # The BF16 output tile, as fp8w_bf16_gemm would store it.
            if shards > 1:
                T.copy(C_total, C_shared)
            else:
                T.copy(C_local, C_shared)
            for r, e in T.Parallel(block_M * blocks, 32):
                X[r, e] = T.Cast(
                    T.float32, C_shared[r // blocks, (r % blocks) * 32 + e]
                )
            _reference_cast(X, amax, inv, Q, S, by * block_M, n0, block_M, blocks, M)

    return gemm_fp8w_bf16_cast


@T.macro
def _w8a16_weights(W, E, W_shared, S_shared, B_shared, n0, ko, block_N, block_K):
    """Copy a weight tile, decode its UE8M0 exponents to FP32 powers of two once (a
    per-element uint8 gather vectorizes to a byte pack nvcc rejects) and expand the
    weights exactly to BF16."""
    T.copy(W[n0, ko * block_K], W_shared)
    for r, c in T.Parallel(block_N // 32, block_K // 32):
        exponent = T.Cast(T.uint32, E[n0 // 32 + r, ko * (block_K // 32) + c])
        S_shared[r, c] = T.reinterpret(T.shift_left(exponent, 23), T.float32)
    for i, j in T.Parallel(block_N, block_K):
        B_shared[i, j] = T.Cast(
            T.bfloat16,
            T.Cast(T.float32, W_shared[i, j]) * S_shared[i // 32, j // 32],
        )


@T.macro
def _w8a16_step(
    A, W, E, A_shared, W_shared, S_shared, B_shared, C_local, by, g, n0, ko, K,
    block_M, block_N, block_K,
):  # fmt: skip
    """One K block of the W8A16 GEMM: the activation tile, the expanded weight tile,
    MMA."""
    T.copy(A[by * block_M, g * K + ko * block_K], A_shared)
    _w8a16_weights(W, E, W_shared, S_shared, B_shared, n0, ko, block_N, block_K)
    T.gemm(A_shared, B_shared, C_local, transpose_B=True)


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def fp8w_bf16_gemm_partials(
    N: int,
    K: int,
    groups: int,
    shards: int,
    block_M: int = 16,
    block_N: int = 32,
    block_K: int = 64,
    num_stages: int = 2,
    threads: int = 128,
):
    """Split-K W8A16 for decode rows: FP32 ``P[s]`` over K shard ``s`` of every group,
    a CTA per (column tile, shard, row tile); ``splitk_reduce`` adds them in order."""
    assert K % block_K == 0 and block_K % 32 == 0 and N % block_N == 0
    assert block_N % 32 == 0
    k_blocks = K // block_K
    assert k_blocks % shards == 0
    shard_blocks = k_blocks // shards
    M = T.dynamic("M")
    n_tiles = N // block_N

    @T.prim_func
    def gemm_fp8w_bf16_partials(
        A: T.Tensor((M, groups * K), T.bfloat16),
        W: T.Tensor((groups * N, K), T.float8_e4m3fn),
        E: T.Tensor((groups * N // 32, K // 32), T.uint8),
        P: T.Tensor((shards, M, groups * N), T.float32),
    ):
        with T.Kernel(
            groups * n_tiles, shards, T.ceildiv(M, block_M), threads=threads
        ) as (bx, s, by):
            pdl.chain()
            g = bx // n_tiles
            n0 = g * N + (bx % n_tiles) * block_N
            A_shared = T.alloc_shared((block_M, block_K), T.bfloat16)
            W_shared = T.alloc_shared((block_N, block_K), T.float8_e4m3fn)
            S_shared = T.alloc_shared((block_N // 32, block_K // 32), T.float32)
            B_shared = T.alloc_shared((block_N, block_K), T.bfloat16)
            C_local = T.alloc_fragment((block_M, block_N), T.float32)
            T.clear(C_local)
            for kb in T.Pipelined(shard_blocks, num_stages=num_stages):
                _w8a16_step(A, W, E, A_shared, W_shared, S_shared, B_shared, C_local,
                            by, g, n0, s * shard_blocks + kb, K, block_M, block_N,
                            block_K)  # fmt: skip
            T.copy(
                C_local,
                P[s, by * block_M : (by + 1) * block_M, n0 : n0 + block_N],
            )

    return gemm_fp8w_bf16_partials


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
    def gemm_splitk_reduce_two(
        P: T.Tensor((shards, M, 2 * N), T.float32),
        C0: T.Tensor((M, N), out_dtype),
        C1: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(2 * tiles, M, threads=threads) as (bx, m):
            pdl.chain()
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

    return gemm_splitk_reduce_two


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
    def gemm_bf16_two(
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

    return gemm_bf16_two


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
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
    def gemm_bf16_partials(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        P: T.Tensor((shards, M, N), T.float32),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N), shards, T.ceildiv(M, block_M), threads=threads
        ) as (bx, s, by):
            pdl.chain()
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

    return gemm_bf16_partials


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
            pdl.chain()
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
        def gemm_bf16_splitk(
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
        def gemm_bf16_splitk(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((N, K), dtype),
            P: T.Tensor((shards, M, N), T.float32),
            counters: T.Tensor((tiles,), T.int32),
            C0: T.Tensor((M, N), out_dtype),
        ):
            body(A, B, P, counters, C0, None)

    return gemm_bf16_splitk


_PACKED_GROUP = 128
_PACKED_HALF = 64
_PACKED_VEC = 16

_PACKED_BF16_HEADER = r"""
#include <stdint.h>

__device__ __forceinline__ void splitk_fence() { __threadfence(); }

__device__ __forceinline__ void vllm_packed_bf16_load_row_bounds(
    int num_exceptions,
    const int* __restrict__ row_offsets,
    int row,
    int* __restrict__ out_start,
    int* __restrict__ out_end) {
  const bool has_exc = (num_exceptions > 0);
  *out_start = has_exc ? __ldg(&row_offsets[row]) : 0;
  *out_end = has_exc ? __ldg(&row_offsets[row + 1]) : 0;
}

__device__ __forceinline__ void vllm_packed_bf16_filter_exc(
    const int* __restrict__ exception_entries,
    int num_exceptions,
    const int* __restrict__ row_offsets,
    int row,
    int col_start,
    int col_end,
    int* __restrict__ out_start,
    int* __restrict__ out_end) {
  if (num_exceptions <= 0) {
    *out_start = 0;
    *out_end = 0;
    return;
  }
  int start = __ldg(&row_offsets[row]);
  int end = __ldg(&row_offsets[row + 1]);
  if (col_start > 0 && start < end) {
    int lo = start;
    int hi = end;
    while (lo < hi) {
      const int mid = (lo + hi) >> 1;
      const int col = (int)(((unsigned int)__ldg(&exception_entries[mid])) >> 16);
      if (col < col_start) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    start = lo;
  }
  if (start < end) {
    int lo = start;
    int hi = end;
    while (lo < hi) {
      const int mid = (lo + hi) >> 1;
      const int col = (int)(((unsigned int)__ldg(&exception_entries[mid])) >> 16);
      if (col < col_end) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    end = lo;
  }
  *out_start = start;
  *out_end = end;
}

template <int BASE_MINUS_ONE>
__device__ __forceinline__ void vllm_packed_bf16_decode_vec16(
    const void* __restrict__ low_a_ptr,
    const void* __restrict__ low_b_ptr,
    const void* __restrict__ codes_ptr,
    void* __restrict__ out_a_ptr,
    void* __restrict__ out_b_ptr,
    const int* __restrict__ exception_entries,
    int* __restrict__ exc_start_ptr,
    int exc_end,
    int col_a0) {
  const uint4 low_a4 = *reinterpret_cast<const uint4*>(low_a_ptr);
  const uint4 low_b4 = *reinterpret_cast<const uint4*>(low_b_ptr);
  const uint4 codes4 = *reinterpret_cast<const uint4*>(codes_ptr);
  const unsigned int* la = reinterpret_cast<const unsigned int*>(&low_a4);
  const unsigned int* lb = reinterpret_cast<const unsigned int*>(&low_b4);
  const unsigned int* cd = reinterpret_cast<const unsigned int*>(&codes4);
  unsigned int* oa = reinterpret_cast<unsigned int*>(out_a_ptr);
  unsigned int* ob = reinterpret_cast<unsigned int*>(out_b_ptr);
  constexpr unsigned int BASE_BYTE = (unsigned int)(BASE_MINUS_ONE & 0xFF);

  #pragma unroll
  for (int w = 0; w < 4; ++w) {
    const unsigned int c_word = cd[w];
    const unsigned int la_word = la[w];
    const unsigned int lb_word = lb[w];

    const unsigned int ca4 = c_word & 0x0F0F0F0Fu;
    const unsigned int cb4 = (c_word >> 4) & 0x0F0F0F0Fu;
    const unsigned int nza4 = ((ca4 + 0x0F0F0F0Fu) >> 4) & 0x01010101u;
    const unsigned int nzb4 = ((cb4 + 0x0F0F0F0Fu) >> 4) & 0x01010101u;
    const unsigned int ea4 = ca4 + nza4 * BASE_BYTE;
    const unsigned int eb4 = cb4 + nzb4 * BASE_BYTE;

    const unsigned int lo_a4 = (la_word & 0x7F7F7F7Fu) | ((ea4 << 7) & 0x80808080u);
    const unsigned int hi_a4 = (la_word & 0x80808080u) | ((ea4 >> 1) & 0x7F7F7F7Fu);
    const unsigned int lo_b4 = (lb_word & 0x7F7F7F7Fu) | ((eb4 << 7) & 0x80808080u);
    const unsigned int hi_b4 = (lb_word & 0x80808080u) | ((eb4 >> 1) & 0x7F7F7F7Fu);

    oa[2 * w + 0] = __byte_perm(lo_a4, hi_a4, 0x5140u);
    oa[2 * w + 1] = __byte_perm(lo_a4, hi_a4, 0x7362u);
    ob[2 * w + 0] = __byte_perm(lo_b4, hi_b4, 0x5140u);
    ob[2 * w + 1] = __byte_perm(lo_b4, hi_b4, 0x7362u);
  }

  int cur = *exc_start_ptr;
  if (__builtin_expect(cur < exc_end, 0)) {
    while (cur < exc_end) {
      const unsigned int entry = (unsigned int)__ldg(&exception_entries[cur]);
      const int col = (int)(entry >> 16);
      if (col < col_a0) {
        ++cur;
        continue;
      }
      if (col >= col_a0 + 80) {
        break;
      }
      const int da = col - col_a0;
      const int db = da - 64;
      if ((unsigned int)da < 16u || (unsigned int)db < 16u) {
        const unsigned int val16 = entry & 0xFFFFu;
        const unsigned int mask = (da & 1) ? 0xFFFF0000u : 0x0000FFFFu;
        const unsigned int shifted = (da & 1) ? (val16 << 16) : val16;
        const int wa = ((unsigned int)da < 16u) ? (da >> 1) : -1;
        const int wb = ((unsigned int)db < 16u) ? (db >> 1) : -1;
        #pragma unroll
        for (int u = 0; u < 8; ++u) {
          if (wa == u) oa[u] = (oa[u] & ~mask) | shifted;
          if (wb == u) ob[u] = (ob[u] & ~mask) | shifted;
        }
      }
      ++cur;
    }
    *exc_start_ptr = cur;
  }
}
"""


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def packed_bf16_unpack(
    N: int,
    K: int,
    exponent_base: int,
    threads: int = 128,
):
    """Unpack 12-bit packed BF16 ``W[N, K + K // 2]`` into ``B[N, K]`` BF16."""
    assert K % _PACKED_GROUP == 0
    k_groups = K // _PACKED_GROUP
    groups_per_row_tile = (
        8 if k_groups % 8 == 0 else (4 if k_groups % 4 == 0 else 1)
    )
    cta_k_blocks = k_groups // groups_per_row_tile
    row_bytes = K + K // 2
    base_minus_one = exponent_base - 1
    slices = _PACKED_HALF // _PACKED_VEC
    threads_per_row = groups_per_row_tile * slices
    rows_per_cta = threads // threads_per_row
    E = T.dynamic("E")

    @T.prim_func
    def unpack_packed_bf16(
        W: T.Tensor((N, row_bytes), T.uint8),
        exception_entries: T.Tensor((E,), T.int32),
        row_offsets: T.Tensor((N + 1,), T.int32),
        B: T.Tensor((N, K), T.bfloat16),
    ):
        with T.Kernel(
            cta_k_blocks, T.ceildiv(N, rows_per_cta), threads=threads
        ) as (kb, by):
            T.import_source(_PACKED_BF16_HEADER)
            pdl.chain()
            tx = T.get_thread_binding()
            j = tx // threads_per_row
            rem = tx % threads_per_row
            g = rem // slices
            vg = rem % slices
            c = vg * _PACKED_VEC
            r = by * rows_per_cta + j
            kg = kb * groups_per_row_tile + g
            col_a0 = kg * _PACKED_GROUP + c
            out_a = T.alloc_local((_PACKED_VEC,), T.bfloat16)
            out_b = T.alloc_local((_PACKED_VEC,), T.bfloat16)
            exc_s = T.alloc_local((1,), T.int32)
            exc_e = T.alloc_local((1,), T.int32)
            if r < N:
                T.call_extern(
                    "handle",
                    "vllm_packed_bf16_filter_exc",
                    T.address_of(exception_entries[0]),
                    E,
                    T.address_of(row_offsets[0]),
                    r,
                    col_a0,
                    (kg + 1) * _PACKED_GROUP,
                    T.address_of(exc_s[0]),
                    T.address_of(exc_e[0]),
                )
                T.call_extern(
                    "handle",
                    f"vllm_packed_bf16_decode_vec16<{base_minus_one}>",
                    T.address_of(W[r, col_a0]),
                    T.address_of(W[r, col_a0 + _PACKED_HALF]),
                    T.address_of(W[r, K + kg * _PACKED_HALF + c]),
                    T.address_of(out_a[0]),
                    T.address_of(out_b[0]),
                    T.address_of(exception_entries[0]),
                    T.address_of(exc_s[0]),
                    exc_e[0],
                    col_a0,
                )
                for v in T.vectorized(_PACKED_VEC):
                    B[r, col_a0 + v] = out_a[v]
                for v in T.vectorized(_PACKED_VEC):
                    B[r, col_a0 + _PACKED_HALF + v] = out_b[v]

    return unpack_packed_bf16


# Column tiles from which a packed GEMM reads its rows' exception bounds directly and
# begins with ``pdl.chain()``. Measured on GB10, both are faster on the wide shapes
# (5120x512 and 4096x1280: 80 and 64 column tiles of 64) and slower on the narrow ones
# (512x10240 and 128x5120: 8 and 2 tiles), where a few CTAs each run a long K loop:
# there the whole-row bounds search before the loop (its result is the row's own end)
# is 2.5-5% faster to keep and the early launch 0.3-0.7% slower, so narrow grids launch
# the kernel as before.
_PACKED_WIDE_TILES = 32


def _packed_bf16_gemm(
    N: int,
    K: int,
    exponent_base: int,
    block_M: int = 64,
    block_N: int = 64,
    num_stages: int = 2,
    threads: int = 128,
    out_dtype: T.dtype = T.bfloat16,
    shards: int = 1,
    blocked: bool = False,
):
    """``C[M, N] = A @ W^T`` for BF16 ``A[M, K]`` and 12-bit packed BF16 ``W``."""
    assert K % _PACKED_GROUP == 0
    k_groups = K // _PACKED_GROUP
    assert shards == 1 or k_groups % shards == 0
    shard_groups = k_groups // shards
    row_bytes = K + K // 2
    base_minus_one = exponent_base - 1
    slices = _PACKED_HALF // _PACKED_VEC
    items = block_N * slices
    iters = tilelang.cdiv(items, threads)
    wide = tilelang.cdiv(N, block_N) >= _PACKED_WIDE_TILES
    single_b = blocked
    pipe_stages = min(num_stages, 2, k_groups)
    n_copies = 4 if blocked else 3
    n_compute = ((8 if single_b else 7) if blocked else 2) + (
        1 if shards > 1 else 0
    )
    n_stmts = n_copies + n_compute
    M = T.dynamic("M")
    E = T.dynamic("E")

    @T.prim_func
    def gemm_packed_bf16(
        A: T.Tensor((M, K), T.bfloat16),
        W: T.Tensor((N, row_bytes), T.uint8),
        exception_entries: T.Tensor((E,), T.int32),
        row_offsets: T.Tensor((N + 1,), T.int32),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads
        ) as (bx, by):
            T.import_source(_PACKED_BF16_HEADER)
            if wide:
                pdl.chain()
            if blocked:
                A0_shared = T.alloc_shared((block_M, _PACKED_HALF), T.bfloat16)
                A1_shared = T.alloc_shared((block_M, _PACKED_HALF), T.bfloat16)
                B0_shared = T.alloc_shared((block_N, _PACKED_HALF), T.bfloat16)
                if not single_b:
                    B1_shared = T.alloc_shared((block_N, _PACKED_HALF), T.bfloat16)
                C_block = T.alloc_fragment((block_M, block_N), T.float32)
            else:
                A_shared = T.alloc_shared((block_M, _PACKED_GROUP), T.bfloat16)
                B_shared = T.alloc_shared((block_N, _PACKED_GROUP), T.bfloat16)
            low_shared = T.alloc_shared((block_N, _PACKED_GROUP), T.uint8)
            code_shared = T.alloc_shared((block_N, _PACKED_HALF), T.uint8)
            C_local = T.alloc_fragment((block_M, block_N), T.float32)
            C_total = T.alloc_fragment(
                (block_M, block_N) if shards > 1 else (1, 1), T.float32
            )
            low_a = T.alloc_local((_PACKED_VEC,), T.uint8)
            low_b = T.alloc_local((_PACKED_VEC,), T.uint8)
            codes = T.alloc_local((_PACKED_VEC,), T.uint8)
            out_a = T.alloc_local((iters, _PACKED_VEC), T.bfloat16)
            out_b = T.alloc_local((iters, _PACKED_VEC), T.bfloat16)
            exc_s = T.alloc_local((iters,), T.int32)
            exc_e = T.alloc_local((iters,), T.int32)
            tx = T.get_thread_binding()
            first = bx * block_N
            for step in T.serial(iters):
                item = step * threads + tx
                if items % threads == 0 or item < items:
                    j = item // slices
                    r_idx = T.min(first + j, N - 1)
                    if wide:
                        T.call_extern(
                            "handle",
                            "vllm_packed_bf16_load_row_bounds",
                            E,
                            T.address_of(row_offsets[0]),
                            r_idx,
                            T.address_of(exc_s[step]),
                            T.address_of(exc_e[step]),
                        )
                    else:
                        T.call_extern(
                            "handle",
                            "vllm_packed_bf16_filter_exc",
                            T.address_of(exception_entries[0]),
                            E,
                            T.address_of(row_offsets[0]),
                            r_idx,
                            0,
                            K,
                            T.address_of(exc_s[step]),
                            T.address_of(exc_e[step]),
                        )
                else:
                    exc_s[step] = 0
                    exc_e[step] = 0
            T.clear(C_local)
            if shards > 1:
                T.clear(C_total)
            last = pipe_stages - 1
            for kg in (
                T.serial(k_groups)
                if pipe_stages <= 1
                else T.Pipelined(
                    k_groups,
                    order=list(range(n_stmts)),
                    stage=[0] * n_copies + [last] * n_compute,
                )
            ):
                if blocked:
                    T.copy(A[by * block_M, kg * _PACKED_GROUP], A0_shared)
                    T.copy(
                        A[by * block_M, kg * _PACKED_GROUP + _PACKED_HALF], A1_shared
                    )
                else:
                    T.copy(A[by * block_M, kg * _PACKED_GROUP], A_shared)
                T.copy(W[bx * block_N, kg * _PACKED_GROUP], low_shared)
                T.copy(W[bx * block_N, K + kg * _PACKED_HALF], code_shared)
                for step in T.serial(iters):
                    item = step * threads + tx
                    if items % threads == 0 or item < items:
                        j = item // slices
                        c = item % slices * _PACKED_VEC
                        for v in T.vectorized(_PACKED_VEC):
                            low_a[v] = low_shared[j, c + v]
                        for v in T.vectorized(_PACKED_VEC):
                            low_b[v] = low_shared[j, _PACKED_HALF + c + v]
                        for v in T.vectorized(_PACKED_VEC):
                            codes[v] = code_shared[j, c + v]
                        T.call_extern(
                            "handle",
                            f"vllm_packed_bf16_decode_vec16<{base_minus_one}>",
                            T.address_of(low_a[0]),
                            T.address_of(low_b[0]),
                            T.address_of(codes[0]),
                            T.address_of(out_a[step, 0]),
                            T.address_of(out_b[step, 0]),
                            T.address_of(exception_entries[0]),
                            T.address_of(exc_s[step]),
                            exc_e[step],
                            kg * _PACKED_GROUP + c,
                        )
                        if blocked:
                            for v in T.vectorized(_PACKED_VEC):
                                B0_shared[j, c + v] = out_a[step, v]
                            if not single_b:
                                for v in T.vectorized(_PACKED_VEC):
                                    B1_shared[j, c + v] = out_b[step, v]
                        else:
                            for v in T.vectorized(_PACKED_VEC):
                                B_shared[j, c + v] = out_a[step, v]
                            for v in T.vectorized(_PACKED_VEC):
                                B_shared[j, _PACKED_HALF + c + v] = out_b[step, v]
                if blocked:
                    T.clear(C_block)
                    T.gemm(A0_shared, B0_shared, C_block, transpose_B=True)
                    for i, j in T.Parallel(block_M, block_N):
                        C_local[i, j] += C_block[i, j]
                    if single_b:
                        for step in T.serial(iters):
                            item = step * threads + tx
                            if items % threads == 0 or item < items:
                                j = item // slices
                                c = item % slices * _PACKED_VEC
                                for v in T.vectorized(_PACKED_VEC):
                                    B0_shared[j, c + v] = out_b[step, v]
                        T.clear(C_block)
                        T.gemm(A1_shared, B0_shared, C_block, transpose_B=True)
                    else:
                        T.clear(C_block)
                        T.gemm(A1_shared, B1_shared, C_block, transpose_B=True)
                    for i, j in T.Parallel(block_M, block_N):
                        C_local[i, j] += C_block[i, j]
                else:
                    T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                if shards > 1:  # noqa: SIM102
                    if (kg + 1) % shard_groups == 0:
                        for i, j in T.Parallel(block_M, block_N):
                            C_total[i, j] = T.if_then_else(
                                kg + 1 == shard_groups,
                                C_local[i, j],
                                C_total[i, j] + C_local[i, j],
                            )
                            C_local[i, j] = T.float32(0)
            if shards > 1:
                T.copy(C_total, C[by * block_M, bx * block_N])
            else:
                T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm_packed_bf16


packed_bf16_gemm = tilelang.jit(
    pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True}
)(_packed_bf16_gemm)
packed_bf16_gemm_decode = packed_bf16_gemm


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def packed_bf16_gemm_two(
    N: int,
    K: int,
    exponent_base: int,
    block_M: int = 64,
    block_N: int = 64,
    num_stages: int = 2,
    threads: int = 128,
    out_dtype: T.dtype = T.bfloat16,
    shards: int = 1,
    blocked: bool = False,
):
    """``packed_bf16_gemm`` of a ``[2N, K]`` packed weight into two ``[M, N]`` outputs."""
    assert K % _PACKED_GROUP == 0 and N % block_N == 0
    k_groups = K // _PACKED_GROUP
    assert shards == 1 or k_groups % shards == 0
    shard_groups = k_groups // shards
    row_bytes = K + K // 2
    base_minus_one = exponent_base - 1
    slices = _PACKED_HALF // _PACKED_VEC
    items = block_N * slices
    iters = tilelang.cdiv(items, threads)
    tiles = N // block_N
    wide = 2 * tiles >= _PACKED_WIDE_TILES
    single_b = blocked
    pipe_stages = min(num_stages, 2, k_groups)
    n_copies = 4 if blocked else 3
    n_compute = ((8 if single_b else 7) if blocked else 2) + (
        1 if shards > 1 else 0
    )
    n_stmts = n_copies + n_compute
    M = T.dynamic("M")
    E = T.dynamic("E")

    @T.prim_func
    def gemm_packed_bf16_two(
        A: T.Tensor((M, K), T.bfloat16),
        W: T.Tensor((2 * N, row_bytes), T.uint8),
        exception_entries: T.Tensor((E,), T.int32),
        row_offsets: T.Tensor((2 * N + 1,), T.int32),
        C0: T.Tensor((M, N), out_dtype),
        C1: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(2 * tiles, T.ceildiv(M, block_M), threads=threads) as (bx, by):
            T.import_source(_PACKED_BF16_HEADER)
            if wide:
                pdl.chain()
            if blocked:
                A0_shared = T.alloc_shared((block_M, _PACKED_HALF), T.bfloat16)
                A1_shared = T.alloc_shared((block_M, _PACKED_HALF), T.bfloat16)
                B0_shared = T.alloc_shared((block_N, _PACKED_HALF), T.bfloat16)
                if not single_b:
                    B1_shared = T.alloc_shared((block_N, _PACKED_HALF), T.bfloat16)
                C_block = T.alloc_fragment((block_M, block_N), T.float32)
            else:
                A_shared = T.alloc_shared((block_M, _PACKED_GROUP), T.bfloat16)
                B_shared = T.alloc_shared((block_N, _PACKED_GROUP), T.bfloat16)
            low_shared = T.alloc_shared((block_N, _PACKED_GROUP), T.uint8)
            code_shared = T.alloc_shared((block_N, _PACKED_HALF), T.uint8)
            C_local = T.alloc_fragment((block_M, block_N), T.float32)
            C_total = T.alloc_fragment(
                (block_M, block_N) if shards > 1 else (1, 1), T.float32
            )
            low_a = T.alloc_local((_PACKED_VEC,), T.uint8)
            low_b = T.alloc_local((_PACKED_VEC,), T.uint8)
            codes = T.alloc_local((_PACKED_VEC,), T.uint8)
            out_a = T.alloc_local((iters, _PACKED_VEC), T.bfloat16)
            out_b = T.alloc_local((iters, _PACKED_VEC), T.bfloat16)
            exc_s = T.alloc_local((iters,), T.int32)
            exc_e = T.alloc_local((iters,), T.int32)
            tx = T.get_thread_binding()
            first = bx * block_N
            for step in T.serial(iters):
                item = step * threads + tx
                if items % threads == 0 or item < items:
                    j = item // slices
                    r_idx = T.min(first + j, 2 * N - 1)
                    if wide:
                        T.call_extern(
                            "handle",
                            "vllm_packed_bf16_load_row_bounds",
                            E,
                            T.address_of(row_offsets[0]),
                            r_idx,
                            T.address_of(exc_s[step]),
                            T.address_of(exc_e[step]),
                        )
                    else:
                        T.call_extern(
                            "handle",
                            "vllm_packed_bf16_filter_exc",
                            T.address_of(exception_entries[0]),
                            E,
                            T.address_of(row_offsets[0]),
                            r_idx,
                            0,
                            K,
                            T.address_of(exc_s[step]),
                            T.address_of(exc_e[step]),
                        )
                else:
                    exc_s[step] = 0
                    exc_e[step] = 0
            T.clear(C_local)
            if shards > 1:
                T.clear(C_total)
            last = pipe_stages - 1
            for kg in (
                T.serial(k_groups)
                if pipe_stages <= 1
                else T.Pipelined(
                    k_groups,
                    order=list(range(n_stmts)),
                    stage=[0] * n_copies + [last] * n_compute,
                )
            ):
                if blocked:
                    T.copy(A[by * block_M, kg * _PACKED_GROUP], A0_shared)
                    T.copy(
                        A[by * block_M, kg * _PACKED_GROUP + _PACKED_HALF], A1_shared
                    )
                else:
                    T.copy(A[by * block_M, kg * _PACKED_GROUP], A_shared)
                T.copy(W[bx * block_N, kg * _PACKED_GROUP], low_shared)
                T.copy(W[bx * block_N, K + kg * _PACKED_HALF], code_shared)
                for step in T.serial(iters):
                    item = step * threads + tx
                    if items % threads == 0 or item < items:
                        j = item // slices
                        c = item % slices * _PACKED_VEC
                        for v in T.vectorized(_PACKED_VEC):
                            low_a[v] = low_shared[j, c + v]
                        for v in T.vectorized(_PACKED_VEC):
                            low_b[v] = low_shared[j, _PACKED_HALF + c + v]
                        for v in T.vectorized(_PACKED_VEC):
                            codes[v] = code_shared[j, c + v]
                        T.call_extern(
                            "handle",
                            f"vllm_packed_bf16_decode_vec16<{base_minus_one}>",
                            T.address_of(low_a[0]),
                            T.address_of(low_b[0]),
                            T.address_of(codes[0]),
                            T.address_of(out_a[step, 0]),
                            T.address_of(out_b[step, 0]),
                            T.address_of(exception_entries[0]),
                            T.address_of(exc_s[step]),
                            exc_e[step],
                            kg * _PACKED_GROUP + c,
                        )
                        if blocked:
                            for v in T.vectorized(_PACKED_VEC):
                                B0_shared[j, c + v] = out_a[step, v]
                            if not single_b:
                                for v in T.vectorized(_PACKED_VEC):
                                    B1_shared[j, c + v] = out_b[step, v]
                        else:
                            for v in T.vectorized(_PACKED_VEC):
                                B_shared[j, c + v] = out_a[step, v]
                            for v in T.vectorized(_PACKED_VEC):
                                B_shared[j, _PACKED_HALF + c + v] = out_b[step, v]
                if blocked:
                    T.clear(C_block)
                    T.gemm(A0_shared, B0_shared, C_block, transpose_B=True)
                    for i, j in T.Parallel(block_M, block_N):
                        C_local[i, j] += C_block[i, j]
                    if single_b:
                        for step in T.serial(iters):
                            item = step * threads + tx
                            if items % threads == 0 or item < items:
                                j = item // slices
                                c = item % slices * _PACKED_VEC
                                for v in T.vectorized(_PACKED_VEC):
                                    B0_shared[j, c + v] = out_b[step, v]
                        T.clear(C_block)
                        T.gemm(A1_shared, B0_shared, C_block, transpose_B=True)
                    else:
                        T.clear(C_block)
                        T.gemm(A1_shared, B1_shared, C_block, transpose_B=True)
                    for i, j in T.Parallel(block_M, block_N):
                        C_local[i, j] += C_block[i, j]
                else:
                    T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                if shards > 1:  # noqa: SIM102
                    if (kg + 1) % shard_groups == 0:
                        for i, j in T.Parallel(block_M, block_N):
                            C_total[i, j] = T.if_then_else(
                                kg + 1 == shard_groups,
                                C_local[i, j],
                                C_total[i, j] + C_local[i, j],
                            )
                            C_local[i, j] = T.float32(0)
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

    return gemm_packed_bf16_two


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def packed_bf16_gemm_partials(
    N: int,
    K: int,
    shards: int,
    exponent_base: int,
    block_M: int = 64,
    block_N: int = 64,
    num_stages: int = 3,
    threads: int = 128,
    blocked: bool = False,
):
    """Split-K 12-bit packed BF16 GEMM for decode rows: FP32 ``P[s] = A @ W^T`` over shard ``s``."""
    assert K % _PACKED_GROUP == 0 and (K // _PACKED_GROUP) % shards == 0
    shard_groups = K // _PACKED_GROUP // shards
    shard_cols = K // shards
    row_bytes = K + K // 2
    base_minus_one = exponent_base - 1
    slices = _PACKED_HALF // _PACKED_VEC
    items = block_N * slices
    iters = tilelang.cdiv(items, threads)
    high_cta = tilelang.cdiv(N, block_N) * shards > 40
    single_b = blocked and high_cta
    pipe_stages = min(num_stages, 2 if high_cta else num_stages, shard_groups)
    n_copies = 4 if blocked else 3
    n_compute = (8 if single_b else 7) if blocked else 2
    n_stmts = n_copies + n_compute
    M = T.dynamic("M")
    E = T.dynamic("E")

    @T.prim_func
    def gemm_packed_bf16_partials(
        A: T.Tensor((M, K), T.bfloat16),
        W: T.Tensor((N, row_bytes), T.uint8),
        exception_entries: T.Tensor((E,), T.int32),
        row_offsets: T.Tensor((N + 1,), T.int32),
        P: T.Tensor((shards, M, N), T.float32),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N), shards, T.ceildiv(M, block_M), threads=threads
        ) as (bx, s, by):
            pdl.chain()
            T.import_source(_PACKED_BF16_HEADER)
            if blocked:
                A0_shared = T.alloc_shared((block_M, _PACKED_HALF), T.bfloat16)
                A1_shared = T.alloc_shared((block_M, _PACKED_HALF), T.bfloat16)
                B0_shared = T.alloc_shared((block_N, _PACKED_HALF), T.bfloat16)
                if not single_b:
                    B1_shared = T.alloc_shared((block_N, _PACKED_HALF), T.bfloat16)
                C_block = T.alloc_fragment((block_M, block_N), T.float32)
            else:
                A_shared = T.alloc_shared((block_M, _PACKED_GROUP), T.bfloat16)
                B_shared = T.alloc_shared((block_N, _PACKED_GROUP), T.bfloat16)
            low_shared = T.alloc_shared((block_N, _PACKED_GROUP), T.uint8)
            code_shared = T.alloc_shared((block_N, _PACKED_HALF), T.uint8)
            C_local = T.alloc_fragment((block_M, block_N), T.float32)
            low_a = T.alloc_local((_PACKED_VEC,), T.uint8)
            low_b = T.alloc_local((_PACKED_VEC,), T.uint8)
            codes = T.alloc_local((_PACKED_VEC,), T.uint8)
            out_a = T.alloc_local((iters, _PACKED_VEC), T.bfloat16)
            out_b = T.alloc_local((iters, _PACKED_VEC), T.bfloat16)
            exc_s = T.alloc_local((iters,), T.int32)
            exc_e = T.alloc_local((iters,), T.int32)
            tx = T.get_thread_binding()
            first = bx * block_N
            for step in T.serial(iters):
                item = step * threads + tx
                if items % threads == 0 or item < items:
                    j = item // slices
                    r_idx = T.min(first + j, N - 1)
                    T.call_extern(
                        "handle",
                        "vllm_packed_bf16_filter_exc",
                        T.address_of(exception_entries[0]),
                        E,
                        T.address_of(row_offsets[0]),
                        r_idx,
                        s * shard_cols,
                        (s + 1) * shard_cols,
                        T.address_of(exc_s[step]),
                        T.address_of(exc_e[step]),
                    )
                else:
                    exc_s[step] = 0
                    exc_e[step] = 0
            T.clear(C_local)
            last = pipe_stages - 1
            for gb in (
                T.serial(shard_groups)
                if pipe_stages <= 1
                else T.Pipelined(
                    shard_groups,
                    order=list(range(n_stmts)),
                    stage=[0] * n_copies + [last] * n_compute,
                )
            ):
                kg = s * shard_groups + gb
                if blocked:
                    T.copy(A[by * block_M, kg * _PACKED_GROUP], A0_shared)
                    T.copy(
                        A[by * block_M, kg * _PACKED_GROUP + _PACKED_HALF], A1_shared
                    )
                else:
                    T.copy(A[by * block_M, kg * _PACKED_GROUP], A_shared)
                T.copy(W[bx * block_N, kg * _PACKED_GROUP], low_shared)
                T.copy(W[bx * block_N, K + kg * _PACKED_HALF], code_shared)
                for step in T.serial(iters):
                    item = step * threads + tx
                    if items % threads == 0 or item < items:
                        j = item // slices
                        c = item % slices * _PACKED_VEC
                        for v in T.vectorized(_PACKED_VEC):
                            low_a[v] = low_shared[j, c + v]
                        for v in T.vectorized(_PACKED_VEC):
                            low_b[v] = low_shared[j, _PACKED_HALF + c + v]
                        for v in T.vectorized(_PACKED_VEC):
                            codes[v] = code_shared[j, c + v]
                        T.call_extern(
                            "handle",
                            f"vllm_packed_bf16_decode_vec16<{base_minus_one}>",
                            T.address_of(low_a[0]),
                            T.address_of(low_b[0]),
                            T.address_of(codes[0]),
                            T.address_of(out_a[step, 0]),
                            T.address_of(out_b[step, 0]),
                            T.address_of(exception_entries[0]),
                            T.address_of(exc_s[step]),
                            exc_e[step],
                            kg * _PACKED_GROUP + c,
                        )
                        if blocked:
                            for v in T.vectorized(_PACKED_VEC):
                                B0_shared[j, c + v] = out_a[step, v]
                            if not single_b:
                                for v in T.vectorized(_PACKED_VEC):
                                    B1_shared[j, c + v] = out_b[step, v]
                        else:
                            for v in T.vectorized(_PACKED_VEC):
                                B_shared[j, c + v] = out_a[step, v]
                            for v in T.vectorized(_PACKED_VEC):
                                B_shared[j, _PACKED_HALF + c + v] = out_b[step, v]
                if blocked:
                    T.clear(C_block)
                    T.gemm(A0_shared, B0_shared, C_block, transpose_B=True)
                    for i, j in T.Parallel(block_M, block_N):
                        C_local[i, j] += C_block[i, j]
                    if single_b:
                        for step in T.serial(iters):
                            item = step * threads + tx
                            if items % threads == 0 or item < items:
                                j = item // slices
                                c = item % slices * _PACKED_VEC
                                for v in T.vectorized(_PACKED_VEC):
                                    B0_shared[j, c + v] = out_b[step, v]
                        T.clear(C_block)
                        T.gemm(A1_shared, B0_shared, C_block, transpose_B=True)
                    else:
                        T.clear(C_block)
                        T.gemm(A1_shared, B1_shared, C_block, transpose_B=True)
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

    return gemm_packed_bf16_partials


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def packed_bf16_gemm_splitk(
    N: int,
    K: int,
    shards: int,
    exponent_base: int,
    block_M: int = 16,
    block_N: int = 64,
    num_stages: int = 2,
    threads: int = 128,
    out_dtype: T.dtype = T.float32,
    parts: int = 1,
    blocked: bool = False,
):
    """Split-K 12-bit packed BF16 GEMM in one launch."""
    assert K % _PACKED_GROUP == 0 and (K // _PACKED_GROUP) % shards == 0
    assert (
        parts in (1, 2) and N % parts == 0 and (parts == 1 or (N // 2) % block_N == 0)
    )
    shard_groups = K // _PACKED_GROUP // shards
    shard_cols = K // shards
    row_bytes = K + K // 2
    base_minus_one = exponent_base - 1
    slices = _PACKED_HALF // _PACKED_VEC
    items = block_N * slices
    iters = tilelang.cdiv(items, threads)
    tiles_N = tilelang.cdiv(N, block_N)
    part = N // parts
    high_cta = tiles_N * shards > 40
    single_b = blocked and high_cta
    pipe_stages = min(num_stages, 2 if high_cta else num_stages, shard_groups)
    n_copies = 4 if blocked else 3
    n_compute = (8 if single_b else 7) if blocked else 2
    n_stmts = n_copies + n_compute
    M = T.dynamic("M")
    E = T.dynamic("E")
    tiles = T.dynamic("tiles")

    @T.macro
    def body(A, W, exception_entries, row_offsets, P, counters, C0, C1):
        with T.Kernel(tiles_N, shards, T.ceildiv(M, block_M), threads=threads) as (
            bx,
            s,
            by,
        ):
            T.import_source(_PACKED_BF16_HEADER)
            pdl.chain()
            tx = T.get_thread_binding()
            if blocked:
                A0_shared = T.alloc_shared((block_M, _PACKED_HALF), T.bfloat16)
                A1_shared = T.alloc_shared((block_M, _PACKED_HALF), T.bfloat16)
                B0_shared = T.alloc_shared((block_N, _PACKED_HALF), T.bfloat16)
                if not single_b:
                    B1_shared = T.alloc_shared((block_N, _PACKED_HALF), T.bfloat16)
                C_block = T.alloc_fragment((block_M, block_N), T.float32)
            else:
                A_shared = T.alloc_shared((block_M, _PACKED_GROUP), T.bfloat16)
                B_shared = T.alloc_shared((block_N, _PACKED_GROUP), T.bfloat16)
            low_shared = T.alloc_shared((block_N, _PACKED_GROUP), T.uint8)
            code_shared = T.alloc_shared((block_N, _PACKED_HALF), T.uint8)
            C_local = T.alloc_fragment((block_M, block_N), T.float32)
            low_a = T.alloc_local((_PACKED_VEC,), T.uint8)
            low_b = T.alloc_local((_PACKED_VEC,), T.uint8)
            codes = T.alloc_local((_PACKED_VEC,), T.uint8)
            out_a = T.alloc_local((iters, _PACKED_VEC), T.bfloat16)
            out_b = T.alloc_local((iters, _PACKED_VEC), T.bfloat16)
            exc_s = T.alloc_local((iters,), T.int32)
            exc_e = T.alloc_local((iters,), T.int32)
            last_cta = T.alloc_shared((1,), T.int32)
            first = bx * block_N
            for step in T.serial(iters):
                item = step * threads + tx
                if items % threads == 0 or item < items:
                    j = item // slices
                    r_idx = T.min(first + j, N - 1)
                    T.call_extern(
                        "handle",
                        "vllm_packed_bf16_filter_exc",
                        T.address_of(exception_entries[0]),
                        E,
                        T.address_of(row_offsets[0]),
                        r_idx,
                        s * shard_cols,
                        (s + 1) * shard_cols,
                        T.address_of(exc_s[step]),
                        T.address_of(exc_e[step]),
                    )
                else:
                    exc_s[step] = 0
                    exc_e[step] = 0
            T.clear(C_local)
            last = pipe_stages - 1
            for gb in (
                T.serial(shard_groups)
                if pipe_stages <= 1
                else T.Pipelined(
                    shard_groups,
                    order=list(range(n_stmts)),
                    stage=[0] * n_copies + [last] * n_compute,
                )
            ):
                kg = s * shard_groups + gb
                if blocked:
                    T.copy(A[by * block_M, kg * _PACKED_GROUP], A0_shared)
                    T.copy(
                        A[by * block_M, kg * _PACKED_GROUP + _PACKED_HALF], A1_shared
                    )
                else:
                    T.copy(A[by * block_M, kg * _PACKED_GROUP], A_shared)
                T.copy(W[bx * block_N, kg * _PACKED_GROUP], low_shared)
                T.copy(W[bx * block_N, K + kg * _PACKED_HALF], code_shared)
                for step in T.serial(iters):
                    item = step * threads + tx
                    if items % threads == 0 or item < items:
                        j = item // slices
                        c = item % slices * _PACKED_VEC
                        for v in T.vectorized(_PACKED_VEC):
                            low_a[v] = low_shared[j, c + v]
                        for v in T.vectorized(_PACKED_VEC):
                            low_b[v] = low_shared[j, _PACKED_HALF + c + v]
                        for v in T.vectorized(_PACKED_VEC):
                            codes[v] = code_shared[j, c + v]
                        T.call_extern(
                            "handle",
                            f"vllm_packed_bf16_decode_vec16<{base_minus_one}>",
                            T.address_of(low_a[0]),
                            T.address_of(low_b[0]),
                            T.address_of(codes[0]),
                            T.address_of(out_a[step, 0]),
                            T.address_of(out_b[step, 0]),
                            T.address_of(exception_entries[0]),
                            T.address_of(exc_s[step]),
                            exc_e[step],
                            kg * _PACKED_GROUP + c,
                        )
                        if blocked:
                            for v in T.vectorized(_PACKED_VEC):
                                B0_shared[j, c + v] = out_a[step, v]
                            if not single_b:
                                for v in T.vectorized(_PACKED_VEC):
                                    B1_shared[j, c + v] = out_b[step, v]
                        else:
                            for v in T.vectorized(_PACKED_VEC):
                                B_shared[j, c + v] = out_a[step, v]
                            for v in T.vectorized(_PACKED_VEC):
                                B_shared[j, _PACKED_HALF + c + v] = out_b[step, v]
                if blocked:
                    T.clear(C_block)
                    T.gemm(A0_shared, B0_shared, C_block, transpose_B=True)
                    for i, j in T.Parallel(block_M, block_N):
                        C_local[i, j] += C_block[i, j]
                    if single_b:
                        for step in T.serial(iters):
                            item = step * threads + tx
                            if items % threads == 0 or item < items:
                                j = item // slices
                                c = item % slices * _PACKED_VEC
                                for v in T.vectorized(_PACKED_VEC):
                                    B0_shared[j, c + v] = out_b[step, v]
                        T.clear(C_block)
                        T.gemm(A1_shared, B0_shared, C_block, transpose_B=True)
                    else:
                        T.clear(C_block)
                        T.gemm(A1_shared, B1_shared, C_block, transpose_B=True)
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
            T.call_extern("handle", "splitk_fence")
            T.sync_threads()
            if tx == 0:
                arrived = T.atomic_add(
                    counters[by * tiles_N + bx],
                    1,
                    memory_order="acq_rel",
                    return_prev=True,
                )
                last_cta[0] = T.if_then_else(arrived == shards - 1, 1, 0)
            T.sync_threads()
            if last_cta[0] == 1:
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
                        elif n < part:
                            C0[m, n] = acc
                        else:
                            C1[m, n - part] = acc
                if tx == 0:
                    counters[by * tiles_N + bx] = 0

    if parts == 2:

        @T.prim_func
        def gemm_packed_bf16_splitk(
            A: T.Tensor((M, K), T.bfloat16),
            W: T.Tensor((N, row_bytes), T.uint8),
            exception_entries: T.Tensor((E,), T.int32),
            row_offsets: T.Tensor((N + 1,), T.int32),
            P: T.Tensor((shards, M, N), T.float32),
            counters: T.Tensor((tiles,), T.int32),
            C0: T.Tensor((M, part), out_dtype),
            C1: T.Tensor((M, part), out_dtype),
        ):
            body(A, W, exception_entries, row_offsets, P, counters, C0, C1)

    else:

        @T.prim_func
        def gemm_packed_bf16_splitk(
            A: T.Tensor((M, K), T.bfloat16),
            W: T.Tensor((N, row_bytes), T.uint8),
            exception_entries: T.Tensor((E,), T.int32),
            row_offsets: T.Tensor((N + 1,), T.int32),
            P: T.Tensor((shards, M, N), T.float32),
            counters: T.Tensor((tiles,), T.int32),
            C0: T.Tensor((M, N), out_dtype),
        ):
            body(A, W, exception_entries, row_offsets, P, counters, C0, None)

    return gemm_packed_bf16_splitk

# ---------------------------------------------------------------------------
# Scale packing and tile configuration
# ---------------------------------------------------------------------------


# 4-bit and 2-bit scale codes live together in scale_codes.py (torch only)
# and are re-exported at the top of this module.


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
# Decode Q-B (8192 x 1280) and indexer Q-B (4096 x 1280) pack their [N, K] weights
# into [N // 128, K // 128, 128, 128] tiles so each decode and prefill tile reads a
# contiguous 128-wide slab per K step while keeping bit-identity with row-major [N, K].
TILE_CONTIGUOUS_FP8: set[tuple[int, int]] = {
    (12288, 1280),
    (8192, 1280),
    (4096, 1280),
}

DECODE_FP8_CONFIGS: dict[tuple[int, ...], dict[int, dict]] = {
    # Attention: Q-B (TP3, TP4..7, TP8 / indexer Q-B) and fused Q-A/KV.
    # 64-row Q-B tiles read whole 128-row tile-contiguous slabs (block_N =
    # TILE_WEIGHT_N); N // 128 >= 64 tiles still cover the 48 SMs.
    (12288, 1280): {
        16: dict(block_N=128, block_K=128, num_stages=3),
        32: dict(block_N=64, block_K=128, num_stages=3),
        64: dict(block_N=128, block_K=128, num_stages=2),
    },
    (8192, 1280): {
        16: dict(block_N=128, block_K=128, num_stages=3),
        32: dict(block_N=64, block_K=128, num_stages=3),
        64: dict(block_N=128, block_K=128, num_stages=2),
    },
    (4096, 1280): {
        16: dict(block_N=64, block_K=128, num_stages=3),
        32: dict(block_N=128, block_K=128, num_stages=3),
        64: dict(block_N=64, block_K=128, num_stages=2, threads=64),
    },
    (1792, 5120): {
        16: dict(block_N=64, block_K=256, num_stages=2),
        32: dict(block_N=64, block_K=256, num_stages=3),
        64: dict(block_N=64, block_K=256, num_stages=2),
    },
    # Shared expert gate/up (TP3, TP4, TP5, TP6/TP7, TP8).
    (1536, 5120): {
        16: dict(block_N=32, block_K=256, num_stages=3),
        32: dict(block_N=32, block_K=256, num_stages=3),
        64: dict(block_N=64, block_K=128, num_stages=2),
    },
    # TP3..TP8 column-sharded target fused Q-A/KV:
    # TP3 (608 x 5120, 1792 padded to 1824), TP4 (448 x 5120),
    # TP5 (384 x 5120, 1792 padded to 1920), TP6 (320 x 5120, 1792 padded to 1920),
    # TP7 (256 x 5120), TP8 (224 x 5120).
    (224, 5120): {
        16: dict(block_N=32, block_K=256, num_stages=4),
        32: dict(block_N=32, block_K=256, num_stages=4),
        64: dict(block_N=32, block_K=256, num_stages=4),
    },
    (256, 5120): {
        16: dict(block_N=32, block_K=256, num_stages=4),
        32: dict(block_N=32, block_K=256, num_stages=4),
        64: dict(block_N=32, block_K=256, num_stages=4),
    },
    (320, 5120): {
        16: dict(block_N=32, block_K=256, num_stages=4),
        32: dict(block_N=32, block_K=256, num_stages=4),
        64: dict(block_N=32, block_K=256, num_stages=4),
    },
    (384, 5120): {
        16: dict(block_N=32, block_K=256, num_stages=4),
        32: dict(block_N=32, block_K=256, num_stages=4),
        64: dict(block_N=32, block_K=256, num_stages=4),
    },
    (448, 5120): {
        16: dict(block_N=32, block_K=256, num_stages=4),
        32: dict(block_N=32, block_K=256, num_stages=4),
        64: dict(block_N=32, block_K=256, num_stages=4),
    },
    (608, 5120): {
        16: dict(block_N=32, block_K=256, num_stages=4),
        32: dict(block_N=32, block_K=256, num_stages=4),
        64: dict(block_N=32, block_K=256, num_stages=4),
    },
    # TP4 shared expert: gate/up and down.
    (1152, 5120): {
        16: dict(block_N=32, block_K=256, num_stages=3),
        32: dict(block_N=32, block_K=256, num_stages=3),
        64: dict(block_N=64, block_K=256, num_stages=2),
    },
    (1024, 5120): {
        16: dict(block_N=32, block_K=256, num_stages=3),
        32: dict(block_N=32, block_K=256, num_stages=3),
        64: dict(block_N=64, block_K=256, num_stages=2),
    },
    (768, 5120): {
        16: dict(block_N=32, block_K=256, num_stages=3),
        32: dict(block_N=32, block_K=256, num_stages=3),
        64: dict(block_N=32, block_K=256, num_stages=2),
    },
    (640, 5120): {
        16: dict(block_N=32, block_K=256, num_stages=3),
        32: dict(block_N=32, block_K=256, num_stages=3),
        64: dict(block_N=32, block_K=256, num_stages=2),
    },
    # Shared expert down (TP3, TP4, TP5, TP6/TP7, TP8).
    (5120, 768): {
        16: dict(block_N=128, block_K=64, num_stages=2, threads=64),
        32: dict(block_N=128, block_K=64, num_stages=2),
        64: dict(block_N=64, block_K=64, num_stages=2),
    },
    (5120, 576): {
        16: dict(block_N=128, block_K=64, num_stages=2, threads=64),
        32: dict(block_N=128, block_K=64, num_stages=2),
        64: dict(block_N=64, block_K=64, num_stages=2),
    },
    (5120, 512): {
        16: dict(block_N=128, block_K=64, num_stages=2, threads=64),
        32: dict(block_N=128, block_K=64, num_stages=2),
        64: dict(block_N=64, block_K=64, num_stages=2),
    },
    (5120, 384): {
        16: dict(block_N=128, block_K=64, num_stages=2, threads=64),
        32: dict(block_N=128, block_K=64, num_stages=2),
        64: dict(block_N=64, block_K=64, num_stages=2),
    },
    (5120, 320): {
        16: dict(block_N=128, block_K=64, num_stages=2, threads=64),
        32: dict(block_N=128, block_K=64, num_stages=2),
        64: dict(block_N=64, block_K=64, num_stages=2),
    },
    # Attention output: grouped WO-A (TP3 3 groups, TP4..7 2 groups, TP8 1 group)
    # and WO-B (TP3, TP4..7, TP8).
    (1024, 4096, 3): {
        16: dict(block_N=64, block_K=128, num_stages=4),
        32: dict(block_N=64, block_K=128, num_stages=4),
        64: dict(block_N=64, block_K=128, num_stages=2),
    },
    (1024, 4096, 2): {
        16: dict(block_N=64, block_K=128, num_stages=4),
        32: dict(block_N=64, block_K=128, num_stages=4),
        64: dict(block_N=64, block_K=256, num_stages=2),
    },
    (1024, 4096): {
        16: dict(block_N=32, block_K=128, num_stages=4),
        32: dict(block_N=32, block_K=128, num_stages=4),
        64: dict(block_N=32, block_K=256, num_stages=2),
    },
    (5120, 3072): {
        16: dict(block_N=128, block_K=128, num_stages=3),
        32: dict(block_N=128, block_K=128, num_stages=3),
        64: dict(block_N=128, block_K=128, num_stages=2),
        96: dict(block_N=64, block_K=128, num_stages=2),
        128: dict(block_N=128, block_K=128, num_stages=2, threads=256),
    },
    (5120, 2048): {
        16: dict(block_N=128, block_K=128, num_stages=3),
        32: dict(block_N=128, block_K=128, num_stages=3),
        # 2026-10-13 sweep (experiments/2026-10-13-tilelang-fp8-cast): 0.87 of the
        # default tile at 48-64 rows; a 96-row tile 0.91 of the prefill tile at 72-96.
        64: dict(block_N=128, block_K=128, num_stages=2),
        96: dict(block_N=64, block_K=128, num_stages=2),
        128: dict(block_N=128, block_K=128, num_stages=2, threads=256),
    },
    (5120, 1024): {
        16: dict(block_N=128, block_K=128, num_stages=3),
        32: dict(block_N=128, block_K=128, num_stages=3),
        64: dict(block_N=128, block_K=128, num_stages=2),
        96: dict(block_N=64, block_K=128, num_stages=2),
        128: dict(block_N=128, block_K=128, num_stages=2, threads=256),
    },
    # DSpark main projection (replicated and TP3..TP8 shards).
    (6400, 6144): {
        16: dict(block_N=32, block_K=256, num_stages=4, threads=64),
        32: dict(block_N=32, block_K=256, num_stages=4, threads=64),
        64: dict(block_N=64, block_K=256, num_stages=2),
    },
    (1728, 15360): {
        16: dict(block_N=32, block_K=256, num_stages=3, threads=128),
        32: dict(block_N=64, block_K=256, num_stages=3, threads=128),
        64: dict(block_N=64, block_K=256, num_stages=3, threads=128),
    },
    (1280, 15360): {
        16: dict(block_N=32, block_K=256, num_stages=3, threads=128),
        32: dict(block_N=64, block_K=256, num_stages=3, threads=128),
        64: dict(block_N=32, block_K=256, num_stages=3, threads=128),
    },
    (1024, 15360): {
        16: dict(block_N=32, block_K=256, num_stages=3, threads=128),
        32: dict(block_N=32, block_K=256, num_stages=3, threads=128),
        64: dict(block_N=32, block_K=256, num_stages=3, threads=128),
    },
    (896, 15360): {
        16: dict(block_N=32, block_K=256, num_stages=3, threads=128),
        32: dict(block_N=32, block_K=256, num_stages=4, threads=128),
        64: dict(block_N=32, block_K=256, num_stages=3, threads=128),
    },
    (768, 15360): {
        16: dict(block_N=32, block_K=256, num_stages=3, threads=128),
        32: dict(block_N=32, block_K=256, num_stages=3, threads=128),
        64: dict(block_N=32, block_K=256, num_stages=3, threads=128),
    },
    (640, 15360): {
        16: dict(block_N=32, block_K=256, num_stages=3, threads=128),
        32: dict(block_N=32, block_K=256, num_stages=3, threads=128),
        64: dict(block_N=32, block_K=256, num_stages=3, threads=128),
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
    (384, 5120): dict(
        shards=5,
        blocked=True,
        decode={
            16: dict(block_N=32, num_stages=3, threads=128),
            32: dict(block_N=32, num_stages=3, threads=128),
            64: dict(block_N=32, num_stages=2, threads=256),
        },
    ),
    (256, 5120): dict(
        shards=5,
        blocked=True,
        decode={
            16: dict(block_N=32, num_stages=3, threads=128),
            32: dict(block_N=32, num_stages=3, threads=128),
            64: dict(block_N=32, num_stages=2, threads=256),
        },
    ),
    (192, 5120): dict(
        shards=5,
        blocked=True,
        decode={
            16: dict(block_N=32, num_stages=3, threads=128),
            32: dict(block_N=32, num_stages=3, threads=128),
            64: dict(block_N=32, num_stages=2, threads=256),
        },
    ),
    (128, 5120): dict(
        shards=5,
        blocked=True,
        decode={
            16: dict(block_N=32, num_stages=3, threads=128),
            32: dict(block_N=32, num_stages=3, threads=128),
            64: dict(block_N=32, num_stages=2, threads=256),
        },
    ),
    (64, 5120): dict(
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
    (1536, 5120): {128: 64},
    (224, 5120): {96: 32, 128: 64},
    (256, 5120): {96: 32, 128: 64},
    (320, 5120): {96: 32, 128: 64},
    (384, 5120): {96: 32, 128: 64},
    (448, 5120): {96: 32, 128: 64},
    (608, 5120): {96: 32, 128: 64},
    (1152, 5120): {128: 64},
    (1024, 5120): {128: 64},
    (768, 5120): {128: 64},
    (640, 5120): {128: 64},
    (5120, 768): {96: 32, 128: 64},
    (5120, 576): {96: 32, 128: 64},
    (5120, 512): {96: 32, 128: 64},
    (5120, 384): {96: 32, 128: 64},
    (5120, 320): {96: 32, 128: 64},
    (5120, 3072): {96: 96, 128: 128},
    (5120, 2048): {96: 96, 128: 128},
    (5120, 1024): {96: 96, 128: 128},
}

# BF16 projections whose split-K decode path serves rows past DECODE_ROWS, by
# (N, K): the most rows it serves. The partials plus their reduce add the same
# shards in the same order as the shard GEMM, so the bits do not change. The
# routers (8 shards at 384 experts, 20 at the drafter's 128) split faster up to
# 256 rows, the indexer's head weights (40 shards) up to 1024 (decode-rows
# bundle, 2026-10-10).
BF16_SPLIT_ROWS: dict[tuple[int, int], int] = {
    (384, 5120): 256,
    (128, 5120): 256,
    (32, 5120): 1024,
}


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
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        l2_bytes = torch.cuda.get_device_properties(device).L2_cache_size
    else:
        l2_bytes = 24 * 1024 * 1024
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
