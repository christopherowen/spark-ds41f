# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1-Flash sparse-attention indexer for SM120 / SM121: scores and top-k.

The V4.1 indexer has 32 heads of width 128. It keeps its keys in a paged
MXFP4 cache: each page holds ``page_size`` rows of 64 bytes of packed E2M1
(low nibble first), followed by ``page_size`` rows of 4 UE8M0 scales (one per
32 dims). Queries are quantized to MXFP4 the same way, and per-head weights
arrive as BF16, already multiplied by ``1/64`` and rounded.

The published score keeps distinct BF16 rounding stages. For each position
``j`` and head ``h``:

    dot   = bf16(q_h . k_j)                     # FP32 accumulation
    c_h   = bf16(max(dot, 0) * w_h)
    score = bf16(c_0 + c_1 + ... + c_31)         # FP32, increasing head order

``indexer_score_v41`` computes ``q . k`` straight from the packed operands with
SM120's MXFP4 block-scaled MMA
(``mma.sync.m16n8k64.kind::mxf4nvf4.block_scale.scale_vec::2X.ue8m0``), then
applies the rounding stages and the ordered head sum in the epilogue. It
scores either every position of a row's context, or only the positions in a
candidate list (V4.1's reindex layers).

``IndexerTopK`` selects the top ``k`` entries of each row exactly. Each
4096-entry segment caches order-preserving 16-bit keys of its BF16 scores in
shared memory, runs a two-pass 8-bit radix select and an ordered compaction
(ties at the threshold keep the lowest positions), and keeps its own top
``k``. Long rows then merge the segment winners with the same selection, so
small batches still spread over many CTAs. Positions come back in ascending
order, padded with ``-1``.
``group=8`` ranks blocks of eight positions by their maximum, always keeping
the block that holds the last visible position: V4.1's candidate-domain
selection at its candidate source layer. It returns the blocks' ids, ascending
and ``-1`` padded, and their count of visible positions. Every selected block
but the last is wholly visible and the last holds the row's last position, so
candidate ``j < count`` of such a list is position
``ids[j // 8] * 8 + j % 8``: the reindex layers read their candidates that way,
from an eighth of the bytes of a position list.

Selection is deterministic: the same scores give the same positions
regardless of thread timing or batch composition.
"""

import tilelang
import tilelang.language as T
import torch

from . import pdl, prepare
from .gemm import SF_BLOCK, _E2M1_HEADER

INDEX_HEADS = 32
INDEX_DIM = 128
INDEX_TOPK = 512
CANDIDATE_BLOCKS = 2048
CANDIDATE_BLOCK = 8
# Order-preserving uint16 key of BF16 -inf: entries at or below it are never selected.
_KEY_NEG_INF = 0x007F


def page_bytes(page_size: int) -> int:
    """Index-cache page bytes: ``[page_size, 64]`` data, ``[page_size, 4]`` UE8M0."""
    return page_size * (INDEX_DIM // 2 + INDEX_DIM // SF_BLOCK)


def _make(kernel_call):
    """The kernel of a ``prepare.call`` (a cache hit once it is prepared)."""
    function, args, kwargs = kernel_call
    return function(*args, **kwargs)


def _score_call(width, page_size, candidates):
    return prepare.call(indexer_score_v41, width, page_size, candidates)


def quantize_call(
    page_size=None,
    *,
    rope_heads: int = 0,
    rope_dim: int = 0,
    ratio: int = 1,
):
    """The MXFP4 quantizer ``quantize_mxfp4`` launches, or with ``page_size`` the
    index-key writer ``write_index_keys`` launches, as a ``prepare.call``."""
    kwargs = {}
    if rope_heads > 0:
        kwargs = dict(rope_heads=rope_heads, rope_dim=rope_dim, ratio=ratio)
    if page_size is None:
        return prepare.call(mxfp4_quantize_v41, **kwargs)
    return prepare.call(mxfp4_quantize_v41, True, page_size, **kwargs)


def norm_quantize_call(
    eps: float,
    page_size: int,
    ratio: int = 1,
    slot_dtype: torch.dtype = torch.int64,
    *,
    shards: int = 1,
):
    """The fused ``k_norm + rope + write_index_keys`` kernel ``norm_rope_write_index_keys``
    launches, as a ``prepare.call``."""
    kwargs = {"shards": shards} if shards > 1 else {}
    return prepare.call(
        mxfp4_norm_rope_quantize_v41,
        eps,
        page_size,
        ratio,
        str(slot_dtype).removeprefix("torch."),
        **kwargs,
    )


def _row_extent(Lengths, Active, row, width):
    """Positions a row selects from: ``min(Lengths, Active, width)``, at least zero.

    Candidate rows have no ``Active`` cap: their length counts candidates.
    """
    extent = T.max(Lengths[row], 0)
    if Active is not None:
        extent = T.min(extent, T.max(Active[0], 0))
    return T.min(extent, width)


def _candidate_block(Candidates, row, j):
    """Candidate ``j``'s block id (``-1``: masked). Candidate rows score whole tiles of
    a width that is a multiple of the tile, so ``j`` never passes the list."""
    return Candidates[row, j // CANDIDATE_BLOCK]


def _candidate_position(Candidates, row, j, count=None):
    """Candidate ``j``'s position, always one the row has written. A masked block reads
    block 0, whole in any row of more than one block. With ``count`` (the tile holding
    the row's count), past it the last block's ids point past the row's end, so the
    offset is 0: the first position of the last kept block, which holds the row's last
    position, or position 0 past the kept blocks; without it ``j`` is below the count.
    As a multiple of the block plus an offset below it, the page arithmetic on it folds
    to the block id's."""
    block = T.max(_candidate_block(Candidates, row, j), 0)
    offset = j % CANDIDATE_BLOCK
    if count is not None:
        offset = T.Select(j < count, offset, 0)
    return block * CANDIDATE_BLOCK + offset


_SCORE_LOAD_HEADER = r"""
#include <tl_templates/cuda/instruction/mma_block_scale.h>
__device__ __forceinline__ void __tl_load_k_shared_dense(
    const void* __restrict__ KPool,
    const uint* __restrict__ KScale,
    const int* __restrict__ PageTable,
    void* __restrict__ K_shared,
    void* __restrict__ SFK_shared,
    int row,
    int p0,
    int extent,
    int page_size,
    int table_width,
    int pool_pages,
    int pool_stride,
    int scale_stride,
    int scale_word0
) {
    int tx = threadIdx.x;
    int logical_page = min(p0 / page_size, table_width - 1);
    int raw_page = (table_width >= 1) ? PageTable[(int64_t)row * table_width + logical_page] : 0;
    int page = min(max(raw_page, 0), pool_pages - 1);
    int64_t page_byte_base = ((int64_t)page * pool_stride) >> 1;
    uint4* k_smem = reinterpret_cast<uint4*>(K_shared);

    #pragma unroll
    for (int half = 0; half < 2; ++half) {
        int p = half * 32 + (tx >> 2);
        int c = tx & 3;
        int pos = p0 + p;
        uint4 val = make_uint4(0, 0, 0, 0);
        if (pos < extent && page >= 0 && page < pool_pages) {
            const uint4* gptr = reinterpret_cast<const uint4*>(
                reinterpret_cast<const uint8_t*>(KPool) + page_byte_base + ((pos % page_size) * 64) + c * 16
            );
            val = *gptr;
        }
        int smem_idx = p * 4 + (c ^ ((p >> 1) & 3));
        k_smem[smem_idx] = val;
    }
    if (tx < 64) {
        int pos = p0 + tx;
        uint sval = 0;
        if (pos < extent && page >= 0 && page < pool_pages) {
            sval = KScale[(int64_t)page * scale_stride + scale_word0 + (pos % page_size)];
        }
        reinterpret_cast<uint*>(SFK_shared)[tx] = sval;
    }
    __syncthreads();
}

__device__ __forceinline__ void __tl_load_k_shared_candidates(
    const void* __restrict__ KPool,
    const uint* __restrict__ KScale,
    const int* __restrict__ PageTable,
    const int* __restrict__ Candidates,
    void* __restrict__ K_shared,
    void* __restrict__ SFK_shared,
    int row,
    int p0,
    int extent,
    int blocks,
    int page_size,
    int table_width,
    int pool_pages,
    int pool_stride,
    int scale_stride,
    int scale_word0
) {
    int tx = threadIdx.x;
    bool whole = (p0 + 64 <= extent);
    uint4* k_smem = reinterpret_cast<uint4*>(K_shared);

    #pragma unroll
    for (int half = 0; half < 2; ++half) {
        int p = half * 32 + (tx >> 2);
        int c = tx & 3;
        int j = p0 + p;
        int blk = max(Candidates[(int64_t)row * blocks + (j >> 3)], 0);
        int offset = (!whole && j >= extent) ? 0 : (j & 7);
        int pos = (blk << 3) + offset;
        int logical_page = min(pos / page_size, table_width - 1);
        int raw_page = (table_width >= 1) ? PageTable[(int64_t)row * table_width + logical_page] : 0;
        int page = min(max(raw_page, 0), pool_pages - 1);
        uint4 val = make_uint4(0, 0, 0, 0);
        if (page >= 0 && page < pool_pages) {
            int64_t page_byte_base = ((int64_t)page * pool_stride) >> 1;
            const uint4* gptr = reinterpret_cast<const uint4*>(
                reinterpret_cast<const uint8_t*>(KPool) + page_byte_base + ((pos % page_size) * 64) + c * 16
            );
            val = *gptr;
        }
        int smem_idx = p * 4 + (c ^ ((p >> 1) & 3));
        k_smem[smem_idx] = val;
    }
    if (tx < 64) {
        int j = p0 + tx;
        int blk = max(Candidates[(int64_t)row * blocks + (j >> 3)], 0);
        int offset = (!whole && j >= extent) ? 0 : (j & 7);
        int pos = (blk << 3) + offset;
        int logical_page = min(pos / page_size, table_width - 1);
        int raw_page = (table_width >= 1) ? PageTable[(int64_t)row * table_width + logical_page] : 0;
        int page = min(max(raw_page, 0), pool_pages - 1);
        uint sval = 0;
        if (page >= 0 && page < pool_pages) {
            sval = KScale[(int64_t)page * scale_stride + scale_word0 + (pos % page_size)];
        }
        reinterpret_cast<uint*>(SFK_shared)[tx] = sval;
    }
    __syncthreads();
}

__device__ __forceinline__ void __tl_reduce_store_scores_candidates(
    const float* __restrict__ contrib,
    const int* __restrict__ Candidates,
    bfloat16_t* __restrict__ Scores,
    int row,
    int p0,
    int extent,
    int width,
    int blocks
) {
    int tx = threadIdx.x;
    __syncthreads();
    if (tx < 64) {
        int j = p0 + tx;
        if (j < extent && j < width) {
            int blk = Candidates[(int64_t)row * blocks + (j >> 3)];
            if (blk >= 0) {
                asm volatile("" ::: "memory");
                float s = 0.0f;
                const float* c_row = contrib + tx * 32;
                #pragma unroll
                for (int h = 0; h < 32; ++h) {
                    s += c_row[h];
                }
                Scores[(int64_t)row * width + j] = (bfloat16_t)s;
            } else {
                Scores[(int64_t)row * width + j] = (std::numeric_limits<bfloat16_t>::infinity() * bfloat16_t(-1.0f));
            }
        }
    }
}
// The dense loader's global reads into registers, for the next tile while this one computes, and
// their store into shared memory: together exactly __tl_load_k_shared_dense.
__device__ __forceinline__ void __tl_fetch_k_dense(
    const void* __restrict__ KPool,
    const uint* __restrict__ KScale,
    const int* __restrict__ PageTable,
    int row, int p0, int extent, int page_size, int table_width, int pool_pages,
    int pool_stride, int scale_stride, int scale_word0,
    uint4 (&v)[2], uint& s
) {
    int tx = threadIdx.x;
    int logical_page = min(p0 / page_size, table_width - 1);
    int raw_page = (table_width >= 1) ? PageTable[(int64_t)row * table_width + logical_page] : 0;
    int page = min(max(raw_page, 0), pool_pages - 1);
    int64_t page_byte_base = ((int64_t)page * pool_stride) >> 1;
    #pragma unroll
    for (int half = 0; half < 2; ++half) {
        int p = half * 32 + (tx >> 2);
        int c = tx & 3;
        int pos = p0 + p;
        uint4 val = make_uint4(0, 0, 0, 0);
        if (pos < extent && page >= 0 && page < pool_pages) {
            val = *reinterpret_cast<const uint4*>(
                reinterpret_cast<const uint8_t*>(KPool) + page_byte_base + ((pos % page_size) * 64) + c * 16
            );
        }
        v[half] = val;
    }
    s = 0;
    if (tx < 64) {
        int pos = p0 + tx;
        if (pos < extent && page >= 0 && page < pool_pages) {
            s = KScale[(int64_t)page * scale_stride + scale_word0 + (pos % page_size)];
        }
    }
}

__device__ __forceinline__ void __tl_store_k_shared(
    void* __restrict__ K_shared, void* __restrict__ SFK_shared, const uint4 (&v)[2], uint s
) {
    int tx = threadIdx.x;
    uint4* k_smem = reinterpret_cast<uint4*>(K_shared);
    #pragma unroll
    for (int half = 0; half < 2; ++half) {
        int p = half * 32 + (tx >> 2);
        int c = tx & 3;
        k_smem[p * 4 + (c ^ ((p >> 1) & 3))] = v[half];
    }
    if (tx < 64) {
        reinterpret_cast<uint*>(SFK_shared)[tx] = s;
    }
    __syncthreads();
}

template <int R>
__device__ __forceinline__ void __tl_score_rows_dense_fast(
    const fp4_e2_t* __restrict__ Q,
    const uint* __restrict__ QScale,
    const bfloat16_t* __restrict__ Weights,
    const void* __restrict__ KPool,
    const uint* __restrict__ KScale,
    const int* __restrict__ PageTable,
    const int* __restrict__ Lengths,
    const int* __restrict__ Active,
    bfloat16_t* __restrict__ Scores,
    void* __restrict__ K_shared,
    void* __restrict__ SFK_shared,
    void* __restrict__ Q_shared,
    float* __restrict__ contrib,
    int row0,
    int cta,
    int ctas,
    int rows,
    int width,
    int page_size,
    int table_width,
    int pool_pages,
    int pool_stride,
    int scale_stride,
    int scale_word0
) {
    int tx = threadIdx.x;
    int act = max(Active[0], 0);
    int exts[R];
    int max_ext = 0;
    int min_ext = width;
    #pragma unroll
    for (int r = 0; r < R; ++r) {
        int row = row0 + r;
        if (row < rows) {
            int e = min(min(max(Lengths[row], 0), act), width);
            exts[r] = e;
            max_ext = max(max_ext, e);
            min_ext = min(min_ext, e);
        } else {
            exts[r] = 0;
            min_ext = 0;
        }
    }
    if (cta * 64 >= max_ext) return;

    fp4_e2_2_t B_reg[R][2][16];
    uint32_t sfq_reg[R][2];
    float2 w_reg[R][2];
    __nv_bfloat162 w2_reg[R][2];

    int smem_q_off = (((tx >> 2) * 64) + ((((((tx & 31) >> 4) + ((tx & 3) >> 1)) & 1) * 32)) + ((((((tx & 15) >> 3) + (tx & 1)) & 1) * 16)));
    int ld_q_base = (((tx >> 6) * 1024) + (((tx & 31) >> 4) * 512) + ((tx & 7) * 64) + ((((((tx & 15) >> 3) + ((tx & 3) >> 1)) & 1) * 16)));
    int sfq_idx0 = ((tx >> 6) * 16) + ((tx & 31) >> 2);

    #pragma unroll
    for (int r = 0; r < R; ++r) {
        int row = row0 + r;
        if (row < rows) {
            *(fp4_e2_32_t*)(((fp4_e2_t*)Q_shared) + smem_q_off) = *(const fp4_e2_32_t*)(Q + (int64_t)row * 2048 + tx * 16);
        }
        __syncthreads();
        #pragma unroll
        for (int ki = 0; ki < 2; ++ki) {
            int ld_q_off = ld_q_base + (((((tx & 7) >> 2) + ki) & 1) * 32);
            tl::ptx_ldmatrix_x4(&(((fp4_e2_t*)Q_shared)[ld_q_off]), &(B_reg[r][ki][0]));
        }
        if (row < rows) {
            const uint* qsc = QScale + (int64_t)row * 32;
            sfq_reg[r][0] = qsc[sfq_idx0];
            sfq_reg[r][1] = qsc[sfq_idx0 + 8];
            const bfloat16_t* wptr = Weights + (int64_t)row * 32;
            #pragma unroll
            for (int w_i = 0; w_i < 2; ++w_i) {
                int h0 = ((tx >> 6) * 16) + (w_i * 8) + ((tx & 3) * 2);
                w_reg[r][w_i] = make_float2((float)wptr[h0], (float)wptr[h0 + 1]);
                w2_reg[r][w_i] = *reinterpret_cast<const __nv_bfloat162*>(wptr + h0);
            }
        } else {
            sfq_reg[r][0] = 0;
            sfq_reg[r][1] = 0;
            w_reg[r][0] = make_float2(0.0f, 0.0f);
            w_reg[r][1] = make_float2(0.0f, 0.0f);
            w2_reg[r][0] = __float2bfloat162_rn(0.0f);
            w2_reg[r][1] = __float2bfloat162_rn(0.0f);
        }
        __syncthreads();
    }

    int num_iters = ((max(((max_ext + 63) >> 6) - cta, 0)) + ctas - 1) / ctas;
    int sfk_base = (((tx & 63) >> 5) * 32) + ((tx & 1) * 8) + ((tx & 31) >> 2);

    uint4 pre_v[2];
    uint pre_s = 0;
    int pre_p0 = -1;
    for (int i = 0; i < num_iters; ++i) {
        int p0 = (cta + i * ctas) * 64;
        int lpage = min(p0 / page_size, table_width - 1);
        bool same_k = (R > 1) && ((p0 % page_size) + 64 <= page_size) && (p0 + 64 <= min_ext) && (row0 + R <= rows);
        if (same_k && table_width >= 1) {
            int p_ref = PageTable[(int64_t)row0 * table_width + lpage];
            #pragma unroll
            for (int r = 1; r < R; ++r) {
                if (PageTable[(int64_t)(row0 + r) * table_width + lpage] != p_ref) {
                    same_k = false;
                }
            }
        }

        fp4_e2_2_t A_reg[2][32];
        uint32_t sfk_reg[2];

        if (same_k && (R > 1) && ((R & 1) == 0)) {
            if (pre_p0 == p0) {
                __tl_store_k_shared(K_shared, SFK_shared, pre_v, pre_s);
            } else {
                __tl_load_k_shared_dense(
                    KPool, KScale, PageTable, K_shared, SFK_shared,
                    row0, p0, max_ext, page_size, table_width,
                    pool_pages, pool_stride, scale_stride, scale_word0
                );
            }
            #pragma unroll
            for (int ki = 0; ki < 2; ++ki) {
                #pragma unroll
                for (int i_2 = 0; i_2 < 2; ++i_2) {
                    int ld_k_off = (((tx & 63) >> 5) * 2048) + (i_2 * 1024) + ((tx & 15) * 64) + (((((tx & 7) >> 2) + ki) & 1) * 32) + (((((tx & 31) >> 4) + ((tx & 3) >> 1)) & 1) * 16);
                    tl::ptx_ldmatrix_x4(&(((fp4_e2_t*)K_shared)[ld_k_off]), &(A_reg[ki][i_2 * 16]));
                }
            }
            sfk_reg[0] = ((const uint*)SFK_shared)[sfk_base];
            sfk_reg[1] = ((const uint*)SFK_shared)[sfk_base + 16];
            // The next tile's reads go out now and land while this tile computes. A fast-path tile
            // reads row0's pages, which every row of the group shares; any other path ignores them.
            pre_p0 = -1;
            if (i + 1 < num_iters) {
                int p0n = (cta + (i + 1) * ctas) * 64;
                __tl_fetch_k_dense(
                    KPool, KScale, PageTable, row0, p0n, max_ext, page_size, table_width,
                    pool_pages, pool_stride, scale_stride, scale_word0, pre_v, pre_s
                );
                pre_p0 = p0n;
            }

            #pragma unroll
            for (int rp = 0; rp < R; rp += 2) {
                if (rp > 0) __syncthreads();
                #pragma unroll
                for (int sub = 0; sub < 2; ++sub) {
                    int r = rp + sub;
                    float acc[16];
                    #pragma unroll
                    for (int z = 0; z < 16; ++z) acc[z] = 0.0f;

                    #pragma unroll
                    for (int ki = 0; ki < 2; ++ki) {
                        #pragma unroll
                        for (int i_3 = 0; i_3 < 2; ++i_3) {
                            tl::sm120_mma_sync_blockscaled<tl::SM120MmaBlockScaledKind::kMxf4nvf4, 2, tl::SM120MmaScaleType::kUE8M0>(
                                acc + (i_3 * 8),
                                reinterpret_cast<const uint32_t*>(A_reg[ki] + (i_3 * 16)),
                                reinterpret_cast<const uint32_t*>(B_reg[r][ki] + 0),
                                acc + (i_3 * 8),
                                sfk_reg[i_3],
                                sfq_reg[r][0],
                                static_cast<uint16_t>(ki * 2), 0, static_cast<uint16_t>(ki * 2), 0
                            );
                            tl::sm120_mma_sync_blockscaled<tl::SM120MmaBlockScaledKind::kMxf4nvf4, 2, tl::SM120MmaScaleType::kUE8M0>(
                                acc + (i_3 * 8 + 4),
                                reinterpret_cast<const uint32_t*>(A_reg[ki] + (i_3 * 16)),
                                reinterpret_cast<const uint32_t*>(B_reg[r][ki] + 8),
                                acc + (i_3 * 8 + 4),
                                sfk_reg[i_3],
                                sfq_reg[r][1],
                                static_cast<uint16_t>(ki * 2), 0, static_cast<uint16_t>(ki * 2), 0
                            );
                        }
                    }

                    #pragma unroll
                    for (int i_4 = 0; i_4 < 8; ++i_4) {
                        float2 v_ = *(float2*)(acc + (i_4 * 2));
                        // bf16(max(bf16(dot), 0) * w): the product of two bf16 values is exact, so one
                        // bf16x2 multiply rounds it as the float path's rounding of the float product does.
                        __nv_bfloat162 __m = __hmax2(__float22bfloat162_rn(v_), __float2bfloat162_rn(0.0f));
                        uint1 __2;
                        (reinterpret_cast<__nv_bfloat162*>(&__2))[0] = __hmul2(__m, w2_reg[r][(i_4 & 3) >> 1]);
                        int p_row = (((tx & 63) >> 5) * 32) + ((i_4 >> 2) * 16) + ((i_4 & 1) * 8) + ((tx & 31) >> 2);
                        int h_col = ((tx >> 6) * 16) + (((i_4 & 3) >> 1) * 8) + ((tx & 3) * 2);
                        int swz_col = h_col ^ ((p_row & 15) << 1);
                        bfloat16_t* contrib_bf16 = reinterpret_cast<bfloat16_t*>(contrib);
                        *(uint1*)(contrib_bf16 + ((sub * 64 + p_row) * 32 + swz_col)) = __2;
                    }
                }
                __syncthreads();
                int sub = tx >> 6;
                int p = tx & 63;
                int r = rp + sub;
                int row = row0 + r;
                int ext = exts[r];
                int j = p0 + p;
                if (j < ext && j < width) {
                    asm volatile("" ::: "memory");
                    float s = 0.0f;
                    const bfloat16_t* c_row = reinterpret_cast<const bfloat16_t*>(contrib) + (sub * 64 + p) * 32;
                    // Heads 2k and 2k + 1 stay adjacent under the even swizzle: one 32-bit load
                    // each, added in head order.
                    const uint32_t* c_pair = reinterpret_cast<const uint32_t*>(c_row);
                    int pmask = (p & 15);
                    #pragma unroll
                    for (int h2 = 0; h2 < 16; ++h2) {
                        uint32_t two = c_pair[h2 ^ pmask];
                        float2 f = tl::__tl_cvt_bf16x2_to_f32x2(*reinterpret_cast<__nv_bfloat162*>(&two));
                        s += f.x;
                        s += f.y;
                    }
                    Scores[(int64_t)row * width + j] = (bfloat16_t)s;
                }
            }
            continue;
        }

        if (same_k) {
            __tl_load_k_shared_dense(
                KPool, KScale, PageTable, K_shared, SFK_shared,
                row0, p0, max_ext, page_size, table_width,
                pool_pages, pool_stride, scale_stride, scale_word0
            );
            #pragma unroll
            for (int ki = 0; ki < 2; ++ki) {
                #pragma unroll
                for (int i_2 = 0; i_2 < 2; ++i_2) {
                    int ld_k_off = (((tx & 63) >> 5) * 2048) + (i_2 * 1024) + ((tx & 15) * 64) + (((((tx & 7) >> 2) + ki) & 1) * 32) + (((((tx & 31) >> 4) + ((tx & 3) >> 1)) & 1) * 16);
                    tl::ptx_ldmatrix_x4(&(((fp4_e2_t*)K_shared)[ld_k_off]), &(A_reg[ki][i_2 * 16]));
                }
            }
            sfk_reg[0] = ((const uint*)SFK_shared)[sfk_base];
            sfk_reg[1] = ((const uint*)SFK_shared)[sfk_base + 16];
        }

        #pragma unroll
        for (int r = 0; r < R; ++r) {
            int row = row0 + r;
            int ext = exts[r];
            if (p0 < ext) {
                if (!same_k) {
                    __syncthreads();
                    __tl_load_k_shared_dense(
                        KPool, KScale, PageTable, K_shared, SFK_shared,
                        row, p0, ext, page_size, table_width,
                        pool_pages, pool_stride, scale_stride, scale_word0
                    );
                    #pragma unroll
                    for (int ki = 0; ki < 2; ++ki) {
                        #pragma unroll
                        for (int i_2 = 0; i_2 < 2; ++i_2) {
                            int ld_k_off = (((tx & 63) >> 5) * 2048) + (i_2 * 1024) + ((tx & 15) * 64) + (((((tx & 7) >> 2) + ki) & 1) * 32) + (((((tx & 31) >> 4) + ((tx & 3) >> 1)) & 1) * 16);
                            tl::ptx_ldmatrix_x4(&(((fp4_e2_t*)K_shared)[ld_k_off]), &(A_reg[ki][i_2 * 16]));
                        }
                    }
                    sfk_reg[0] = ((const uint*)SFK_shared)[sfk_base];
                    sfk_reg[1] = ((const uint*)SFK_shared)[sfk_base + 16];
                }

                float acc[16];
                #pragma unroll
                for (int z = 0; z < 16; ++z) acc[z] = 0.0f;

                #pragma unroll
                for (int ki = 0; ki < 2; ++ki) {
                    #pragma unroll
                    for (int i_3 = 0; i_3 < 2; ++i_3) {
                        tl::sm120_mma_sync_blockscaled<tl::SM120MmaBlockScaledKind::kMxf4nvf4, 2, tl::SM120MmaScaleType::kUE8M0>(
                            acc + (i_3 * 8),
                            reinterpret_cast<const uint32_t*>(A_reg[ki] + (i_3 * 16)),
                            reinterpret_cast<const uint32_t*>(B_reg[r][ki] + 0),
                            acc + (i_3 * 8),
                            sfk_reg[i_3],
                            sfq_reg[r][0],
                            static_cast<uint16_t>(ki * 2), 0, static_cast<uint16_t>(ki * 2), 0
                        );
                        tl::sm120_mma_sync_blockscaled<tl::SM120MmaBlockScaledKind::kMxf4nvf4, 2, tl::SM120MmaScaleType::kUE8M0>(
                            acc + (i_3 * 8 + 4),
                            reinterpret_cast<const uint32_t*>(A_reg[ki] + (i_3 * 16)),
                            reinterpret_cast<const uint32_t*>(B_reg[r][ki] + 8),
                            acc + (i_3 * 8 + 4),
                            sfk_reg[i_3],
                            sfq_reg[r][1],
                            static_cast<uint16_t>(ki * 2), 0, static_cast<uint16_t>(ki * 2), 0
                        );
                    }
                }

                if (r > 0 || !same_k) __syncthreads();
                #pragma unroll
                for (int i_4 = 0; i_4 < 8; ++i_4) {
                    float2 v_ = *(float2*)(acc + (i_4 * 2));
                    // bf16(max(bf16(dot), 0) * w): the product of two bf16 values is exact, so one
                    // bf16x2 multiply rounds it as the float path's rounding of the float product does.
                    __nv_bfloat162 __m = __hmax2(__float22bfloat162_rn(v_), __float2bfloat162_rn(0.0f));
                    uint1 __2;
                    (reinterpret_cast<__nv_bfloat162*>(&__2))[0] = __hmul2(__m, w2_reg[r][(i_4 & 3) >> 1]);
                    int p_row = (((tx & 63) >> 5) * 32) + ((i_4 >> 2) * 16) + ((i_4 & 1) * 8) + ((tx & 31) >> 2);
                    int h_col = ((tx >> 6) * 16) + (((i_4 & 3) >> 1) * 8) + ((tx & 3) * 2);
                    int swz_col = h_col ^ ((p_row & 15) << 1);
                    bfloat16_t* contrib_bf16 = reinterpret_cast<bfloat16_t*>(contrib);
                    *(uint1*)(contrib_bf16 + (p_row * 32 + swz_col)) = __2;
                }
                __syncthreads();
                if (tx < 64) {
                    int j = p0 + tx;
                    if (j < ext && j < width) {
                        asm volatile("" ::: "memory");
                        float s = 0.0f;
                        const bfloat16_t* c_row = reinterpret_cast<const bfloat16_t*>(contrib) + tx * 32;
                        const uint32_t* c_pair = reinterpret_cast<const uint32_t*>(c_row);
                        int pmask = (tx & 15);
                        #pragma unroll
                        for (int h2 = 0; h2 < 16; ++h2) {
                            uint32_t two = c_pair[h2 ^ pmask];
                            float2 f = tl::__tl_cvt_bf16x2_to_f32x2(*reinterpret_cast<__nv_bfloat162*>(&two));
                            s += f.x;
                            s += f.y;
                        }
                        Scores[(int64_t)row * width + j] = (bfloat16_t)s;
                    }
                }
            }
        }
        __syncthreads();
    }
}

__device__ __forceinline__ void __tl_score_row_candidates_fast(
    const fp4_e2_t* __restrict__ Q,
    const uint* __restrict__ QScale,
    const bfloat16_t* __restrict__ Weights,
    const void* __restrict__ KPool,
    const uint* __restrict__ KScale,
    const int* __restrict__ PageTable,
    const int* __restrict__ Candidates,
    const int* __restrict__ CandidateLengths,
    bfloat16_t* __restrict__ Scores,
    void* __restrict__ K_shared,
    void* __restrict__ SFK_shared,
    void* __restrict__ Q_shared,
    float* __restrict__ contrib,
    int row,
    int cta,
    int ctas,
    int rows,
    int width,
    int blocks,
    int page_size,
    int table_width,
    int pool_pages,
    int pool_stride,
    int scale_stride,
    int scale_word0
) {
    if (row >= rows) return;
    int extent = min(max(CandidateLengths[row], 0), width);
    if (cta * 64 >= extent) return;

    int tx = threadIdx.x;
    fp4_e2_2_t B_reg[2][16];
    uint32_t sfq_reg[2];
    float2 w_reg[2];

    int smem_q_off = (((tx >> 2) * 64) + ((((((tx & 31) >> 4) + ((tx & 3) >> 1)) & 1) * 32)) + ((((((tx & 15) >> 3) + (tx & 1)) & 1) * 16)));
    int ld_q_base = (((tx >> 6) * 1024) + (((tx & 31) >> 4) * 512) + ((tx & 7) * 64) + ((((((tx & 15) >> 3) + ((tx & 3) >> 1)) & 1) * 16)));
    int sfq_idx0 = ((tx >> 6) * 16) + ((tx & 31) >> 2);

    *(fp4_e2_32_t*)(((fp4_e2_t*)Q_shared) + smem_q_off) = *(const fp4_e2_32_t*)(Q + (int64_t)row * 2048 + tx * 16);
    __syncthreads();
    #pragma unroll
    for (int ki = 0; ki < 2; ++ki) {
        int ld_q_off = ld_q_base + (((((tx & 7) >> 2) + ki) & 1) * 32);
        tl::ptx_ldmatrix_x4(&(((fp4_e2_t*)Q_shared)[ld_q_off]), &(B_reg[ki][0]));
    }
    const uint* qsc = QScale + (int64_t)row * 32;
    sfq_reg[0] = qsc[sfq_idx0];
    sfq_reg[1] = qsc[sfq_idx0 + 8];
    const bfloat16_t* wptr = Weights + (int64_t)row * 32;
    #pragma unroll
    for (int w_i = 0; w_i < 2; ++w_i) {
        int h0 = ((tx >> 6) * 16) + (w_i * 8) + ((tx & 3) * 2);
        w_reg[w_i] = make_float2((float)wptr[h0], (float)wptr[h0 + 1]);
    }

    int num_iters = ((max(((extent + 63) >> 6) - cta, 0)) + ctas - 1) / ctas;
    int sfk_base = (((tx & 63) >> 5) * 32) + ((tx & 1) * 8) + ((tx & 31) >> 2);

    for (int i = 0; i < num_iters; ++i) {
        int p0 = (cta + i * ctas) * 64;
        __tl_load_k_shared_candidates(
            KPool, KScale, PageTable, Candidates, K_shared, SFK_shared,
            row, p0, extent, blocks, page_size, table_width,
            pool_pages, pool_stride, scale_stride, scale_word0
        );

        fp4_e2_2_t A_reg[2][32];
        float acc[16];
        #pragma unroll
        for (int z = 0; z < 16; ++z) acc[z] = 0.0f;

        uint32_t sfk0 = ((const uint*)SFK_shared)[sfk_base];
        uint32_t sfk1 = ((const uint*)SFK_shared)[sfk_base + 16];

        #pragma unroll
        for (int ki = 0; ki < 2; ++ki) {
            #pragma unroll
            for (int i_2 = 0; i_2 < 2; ++i_2) {
                int ld_k_off = (((tx & 63) >> 5) * 2048) + (i_2 * 1024) + ((tx & 15) * 64) + (((((tx & 7) >> 2) + ki) & 1) * 32) + (((((tx & 31) >> 4) + ((tx & 3) >> 1)) & 1) * 16);
                tl::ptx_ldmatrix_x4(&(((fp4_e2_t*)K_shared)[ld_k_off]), &(A_reg[ki][i_2 * 16]));
            }
            #pragma unroll
            for (int i_3 = 0; i_3 < 2; ++i_3) {
                uint32_t sfk_val = (i_3 == 0) ? sfk0 : sfk1;
                tl::sm120_mma_sync_blockscaled<tl::SM120MmaBlockScaledKind::kMxf4nvf4, 2, tl::SM120MmaScaleType::kUE8M0>(
                    acc + (i_3 * 8),
                    reinterpret_cast<const uint32_t*>(A_reg[ki] + (i_3 * 16)),
                    reinterpret_cast<const uint32_t*>(B_reg[ki] + 0),
                    acc + (i_3 * 8),
                    sfk_val, sfq_reg[0],
                    static_cast<uint16_t>(ki * 2), 0, static_cast<uint16_t>(ki * 2), 0
                );
                tl::sm120_mma_sync_blockscaled<tl::SM120MmaBlockScaledKind::kMxf4nvf4, 2, tl::SM120MmaScaleType::kUE8M0>(
                    acc + (i_3 * 8 + 4),
                    reinterpret_cast<const uint32_t*>(A_reg[ki] + (i_3 * 16)),
                    reinterpret_cast<const uint32_t*>(B_reg[ki] + 8),
                    acc + (i_3 * 8 + 4),
                    sfk_val, sfq_reg[1],
                    static_cast<uint16_t>(ki * 2), 0, static_cast<uint16_t>(ki * 2), 0
                );
            }
        }

        #pragma unroll
        for (int i_4 = 0; i_4 < 8; ++i_4) {
            float2 v_ = *(float2*)(acc + (i_4 * 2));
            uint1 __6;
            (reinterpret_cast<__nv_bfloat162*>(&__6))[0] = __float22bfloat162_rn(v_);
            float2 __5 = tl::__tl_cvt_bf16x2_to_f32x2((reinterpret_cast<__nv_bfloat162*>(&__6))[0]);
            float2 __4 = tl::max2(__5, make_float2(0.0f, 0.0f));
            float2 __3 = tl::mul2(__4, w_reg[(i_4 & 3) >> 1]);
            uint1 __2;
            (reinterpret_cast<__nv_bfloat162*>(&__2))[0] = __float22bfloat162_rn(__3);
            int p_row = (((tx & 63) >> 5) * 32) + ((i_4 >> 2) * 16) + ((i_4 & 1) * 8) + ((tx & 31) >> 2);
            int h_col = ((tx >> 6) * 16) + (((i_4 & 3) >> 1) * 8) + ((tx & 3) * 2);
            int swz_col = h_col ^ ((p_row & 15) << 1);
            bfloat16_t* contrib_bf16 = reinterpret_cast<bfloat16_t*>(contrib);
            *(uint1*)(contrib_bf16 + (p_row * 32 + swz_col)) = __2;
        }
        __syncthreads();
        if (tx < 64) {
            int j = p0 + tx;
            if (j < extent && j < width) {
                int blk = Candidates[(int64_t)row * blocks + (j >> 3)];
                if (blk >= 0) {
                    asm volatile("" ::: "memory");
                    float s = 0.0f;
                    const bfloat16_t* c_row = reinterpret_cast<const bfloat16_t*>(contrib) + tx * 32;
                    int mask = (tx & 15) << 1;
                    #pragma unroll
                    for (int h = 0; h < 32; ++h) {
                        s += (float)c_row[h ^ mask];
                    }
                    Scores[(int64_t)row * width + j] = (bfloat16_t)s;
                } else {
                    Scores[(int64_t)row * width + j] = (std::numeric_limits<bfloat16_t>::infinity() * bfloat16_t(-1.0f));
                }
            }
        }
    }
}

"""


@tilelang.jit
def indexer_score_v41(
    width: int,
    page_size: int,
    candidates: bool = False,
    heads: int = INDEX_HEADS,
    block_P: int = 64,
    threads: int = 128,
    max_ctas: int = 128,
):
    """Build the score kernel. Inputs:

    - ``Q``: ``[rows, heads, 128]`` E2M1 queries (``[rows, heads, 64]`` int8 storage);
    - ``QScale``: ``[rows, heads]`` uint32, the four UE8M0 scales of each head;
    - ``Weights``: ``[rows, heads]`` BF16 head weights;
    - ``KPool``: E2M1 view (``[pages, page_bytes * 2]`` logical) of the index cache;
    - ``KScale``: uint32 view (``[pages, page_bytes / 4]``) of the same cache.
      Both may keep the allocator's page stride when it pads pages;
    - ``PageTable``: ``[rows, table_width]`` int32 physical page per logical page;
    - ``Lengths``: ``[rows]`` int32 positions per row;
    - ``Active``: ``[1]`` int32 width cap;
    - candidate mode: ``Candidates`` ``[rows, width // CANDIDATE_BLOCK]`` int32
      block ids (``-1``: a masked block) and ``CandidateLengths`` ``[rows]``,
      candidate ``j`` being position ``Candidates[j // 8] * 8 + j % 8``; its
      source layer writes the ids whole, ``-1`` padded.

    Writes ``Scores`` (``[rows, width]`` BF16) up to each row's extent, ``-inf``
    for masked candidates; entries past the extent are left unwritten and the
    top-k never reads them. No key or scale is read past the visible positions
    either.
    Each row gets at most ``max_ctas`` CTAs, which loop over its live tiles, so
    the launch scales with the context in use rather than the planned width.
    """
    assert page_size % block_P == 0 or candidates, (
        "dense tiles must not straddle a page"
    )
    assert not candidates or width % block_P == 0, "candidate rows score whole tiles"
    blocks = width // CANDIDATE_BLOCK
    rows, pool_pages, table_width, pool_stride, scale_stride = T.dynamic(
        "rows, pool_pages, table_width, pool_stride, scale_stride"
    )
    D = INDEX_DIM
    fp4 = T.float4_e2m1fn
    accum_dtype = T.float32
    page_words = page_bytes(page_size) // 4
    scale_word0 = page_size * D // 2 // 4  # first scale word of a page
    ctas = min(tilelang.cdiv(width, block_P), max_ctas)

    @T.macro
    def load_keys(
        KPool, KScale, PageTable, row, p0, position, loads, K_shared, SFK_shared
    ):
        for p, d in T.Parallel(block_P, D):
            pos = position(p0, p)
            page = T.min(
                T.max(PageTable[row, T.min(pos // page_size, table_width - 1)], 0),
                pool_pages - 1,
            )
            K_shared[p, d] = T.if_then_else(
                loads(p0, p),
                KPool[page, (pos % page_size) * D + d],
                T.cast(0, fp4),
            )
        for p in T.Parallel(block_P):
            pos = position(p0, p)
            page = T.min(
                T.max(PageTable[row, T.min(pos // page_size, table_width - 1)], 0),
                pool_pages - 1,
            )
            SFK_shared[p, 0] = T.if_then_else(
                loads(p0, p),
                KScale[page, scale_word0 + pos % page_size],
                T.cast(0, T.uint32),
            )

    use_vec128 = block_P == 64 and D == 128 and threads == 128

    @T.macro
    def score_row(
        Q,
        QScale,
        Weights,
        KPool,
        KScale,
        PageTable,
        Candidates,
        row,
        cta,
        extent,
        position,
        slot_valid,
        loads,
        whole,
        whole_position,
        Scores,
    ):
        K_shared = T.alloc_shared([block_P, D], fp4)
        Q_shared = T.alloc_shared([heads, D], fp4)
        SFK_shared = T.alloc_shared([block_P, 1], T.uint32)
        SFQ_shared = T.alloc_shared([heads, 1], T.uint32)
        W_shared = T.alloc_shared([heads], accum_dtype)
        contrib = T.alloc_shared([block_P, heads], accum_dtype)
        acc = T.alloc_fragment([block_P, heads], accum_dtype)
        total = T.alloc_fragment([block_P], accum_dtype)

        if cta * block_P < extent:
            for h, d in T.Parallel(heads, D):
                Q_shared[h, d] = Q[row, h, d]
            for h in T.Parallel(heads):
                SFQ_shared[h, 0] = QScale[row, h]
                W_shared[h] = T.cast(Weights[row, h], accum_dtype)
            for i in T.serial(
                T.ceildiv(T.max(T.ceildiv(extent, block_P) - cta, 0), ctas)
            ):
                p0 = (cta + i * ctas) * block_P
                # No load reaches past the visible positions (DeepSeek's reference reads
                # ``index_k[:, : end_pos // ratio]``). Tile rows that ``loads`` refuses
                # (a dense row's rows past its extent) load no key or scale and score
                # zeros, which the store drops. Only the key and scale loads are
                # predicated: the page read stays unconditional, so a dense tile's
                # addresses stay affine and its loads issue together. Every candidate
                # row loads: its position is always a visible one (a masked or padding
                # candidate's too, ``_candidate_position``), and its score is masked at
                # the end. Only the tile holding a candidate row's count bounds its
                # positions by it: a tile ``whole`` below the count takes
                # ``whole_position``, chosen once for the tile.
                if use_vec128 and not candidates:
                    T.call_extern(
                        "handle",
                        "__tl_load_k_shared_dense",
                        T.address_of(KPool[0, 0]),
                        T.address_of(KScale[0, 0]),
                        T.address_of(PageTable[0, 0]),
                        T.address_of(K_shared[0, 0]),
                        T.address_of(SFK_shared[0, 0]),
                        row,
                        p0,
                        extent,
                        page_size,
                        table_width,
                        pool_pages,
                        pool_stride,
                        scale_stride,
                        scale_word0,
                    )
                elif use_vec128 and candidates:
                    T.call_extern(
                        "handle",
                        "__tl_load_k_shared_candidates",
                        T.address_of(KPool[0, 0]),
                        T.address_of(KScale[0, 0]),
                        T.address_of(PageTable[0, 0]),
                        T.address_of(Candidates[0, 0]),
                        T.address_of(K_shared[0, 0]),
                        T.address_of(SFK_shared[0, 0]),
                        row,
                        p0,
                        extent,
                        blocks,
                        page_size,
                        table_width,
                        pool_pages,
                        pool_stride,
                        scale_stride,
                        scale_word0,
                    )
                elif whole is None:
                    load_keys(
                        KPool,
                        KScale,
                        PageTable,
                        row,
                        p0,
                        position,
                        loads,
                        K_shared,
                        SFK_shared,
                    )
                elif whole(p0):
                    load_keys(
                        KPool,
                        KScale,
                        PageTable,
                        row,
                        p0,
                        whole_position,
                        loads,
                        K_shared,
                        SFK_shared,
                    )
                else:
                    load_keys(
                        KPool,
                        KScale,
                        PageTable,
                        row,
                        p0,
                        position,
                        loads,
                        K_shared,
                        SFK_shared,
                    )
                T.mma_gemm_blockscaled(
                    K_shared,
                    Q_shared,
                    acc,
                    SFK_shared,
                    SFQ_shared,
                    transpose_B=True,
                    clear_accum=True,
                    k_start=0,
                    sf_a_granularity_k=SF_BLOCK,
                    sf_b_granularity_k=SF_BLOCK,
                )
                # The published rounding stages: BF16 dot, ReLU, BF16 weighted product.
                for p, h in T.Parallel(block_P, heads):
                    dot = T.max(
                        T.cast(T.cast(acc[p, h], T.bfloat16), accum_dtype), 0.0
                    )
                    contrib[p, h] = T.cast(
                        T.cast(T.fmul(dot, W_shared[h]), T.bfloat16), accum_dtype
                    )
                if use_vec128 and candidates:
                    T.call_extern(
                        "handle",
                        "__tl_reduce_store_scores_candidates",
                        T.address_of(contrib[0, 0]),
                        T.address_of(Candidates[0, 0]),
                        T.address_of(Scores[0, 0]),
                        row,
                        p0,
                        extent,
                        width,
                        blocks,
                    )
                else:
                    T.fill(total, 0)
                    for h in T.serial(heads):  # increasing head order, FP32
                        for p in T.Parallel(block_P):
                            total[p] += contrib[p, h]
                    for p in T.Parallel(block_P):
                        if p0 + p < extent:
                            Scores[row, p0 + p] = T.if_then_else(
                                slot_valid(p0, p),
                                T.cast(total[p], T.bfloat16),
                                -T.infinity(T.bfloat16),
                            )

    fast_vec128 = (
        use_vec128 and block_P == 64 and D == 128 and threads == 128 and heads == 32
    )
    scale_word0 = page_size * D // 2 // 4
    if not candidates and fast_vec128:
        row_blocks = T.if_then_else(rows >= 64, T.ceildiv(rows, 4), rows)

        @T.prim_func
        def indexer_score(
            Q: T.Tensor([rows, heads, D], fp4),
            QScale: T.Tensor([rows, heads], T.uint32),
            Weights: T.Tensor([rows, heads], T.bfloat16),
            KPool: T.StridedTensor([pool_pages, page_words * 8], [pool_stride, 1], fp4),
            KScale: T.StridedTensor(
                [pool_pages, page_words], [scale_stride, 1], T.uint32
            ),
            PageTable: T.Tensor([rows, table_width], T.int32),
            Lengths: T.Tensor([rows], T.int32),
            Active: T.Tensor([1], T.int32),
            Scores: T.Tensor([rows, width], T.bfloat16),
        ):
            with T.Kernel(ctas, row_blocks, threads=threads) as (bx, by):
                T.import_source(_SCORE_LOAD_HEADER)
                pdl.chain()
                linear_block = by * ctas + bx
                cta = linear_block // row_blocks
                rb = linear_block % row_blocks
                K_shared = T.alloc_shared([block_P, D], fp4)
                SFK_shared = T.alloc_shared([block_P, 1], T.uint32)
                Q_shared = T.alloc_shared([heads, D], fp4)
                contrib = T.alloc_shared([block_P, heads], T.float32)
                if rows >= 64:
                    T.call_extern(
                        "handle",
                        "__tl_score_rows_dense_fast<4>",
                        T.address_of(Q[0, 0, 0]),
                        T.address_of(QScale[0, 0]),
                        T.address_of(Weights[0, 0]),
                        T.address_of(KPool[0, 0]),
                        T.address_of(KScale[0, 0]),
                        T.address_of(PageTable[0, 0]),
                        T.address_of(Lengths[0]),
                        T.address_of(Active[0]),
                        T.address_of(Scores[0, 0]),
                        T.address_of(K_shared[0, 0]),
                        T.address_of(SFK_shared[0, 0]),
                        T.address_of(Q_shared[0, 0]),
                        T.address_of(contrib[0, 0]),
                        rb * 4,
                        cta,
                        ctas,
                        rows,
                        width,
                        page_size,
                        table_width,
                        pool_pages,
                        pool_stride,
                        scale_stride,
                        scale_word0,
                    )
                else:
                    T.call_extern(
                        "handle",
                        "__tl_score_rows_dense_fast<1>",
                        T.address_of(Q[0, 0, 0]),
                        T.address_of(QScale[0, 0]),
                        T.address_of(Weights[0, 0]),
                        T.address_of(KPool[0, 0]),
                        T.address_of(KScale[0, 0]),
                        T.address_of(PageTable[0, 0]),
                        T.address_of(Lengths[0]),
                        T.address_of(Active[0]),
                        T.address_of(Scores[0, 0]),
                        T.address_of(K_shared[0, 0]),
                        T.address_of(SFK_shared[0, 0]),
                        T.address_of(Q_shared[0, 0]),
                        T.address_of(contrib[0, 0]),
                        rb,
                        cta,
                        ctas,
                        rows,
                        width,
                        page_size,
                        table_width,
                        pool_pages,
                        pool_stride,
                        scale_stride,
                        scale_word0,
                    )

        return indexer_score

    if not candidates:

        @T.prim_func
        def indexer_score(
            Q: T.Tensor([rows, heads, D], fp4),
            QScale: T.Tensor([rows, heads], T.uint32),
            Weights: T.Tensor([rows, heads], T.bfloat16),
            KPool: T.StridedTensor([pool_pages, page_words * 8], [pool_stride, 1], fp4),
            KScale: T.StridedTensor(
                [pool_pages, page_words], [scale_stride, 1], T.uint32
            ),
            PageTable: T.Tensor([rows, table_width], T.int32),
            Lengths: T.Tensor([rows], T.int32),
            Active: T.Tensor([1], T.int32),
            Scores: T.Tensor([rows, width], T.bfloat16),
        ):
            with T.Kernel(ctas, rows, threads=threads) as (bx, by):
                if use_vec128:
                    T.import_source(_SCORE_LOAD_HEADER)
                pdl.chain()
                linear_block = by * ctas + bx
                cta = linear_block // rows
                row = linear_block % rows
                extent = _row_extent(Lengths, Active, row, width)
                score_row(
                    Q,
                    QScale,
                    Weights,
                    KPool,
                    KScale,
                    PageTable,
                    None,
                    row,
                    cta,
                    extent,
                    lambda p0, p: p0 + p,
                    lambda p0, p: True,
                    lambda p0, p: p0 + p < extent,
                    None,
                    None,
                    Scores,
                )

        return indexer_score

    if fast_vec128:

        @T.prim_func
        def indexer_score(
            Q: T.Tensor([rows, heads, D], fp4),
            QScale: T.Tensor([rows, heads], T.uint32),
            Weights: T.Tensor([rows, heads], T.bfloat16),
            KPool: T.StridedTensor([pool_pages, page_words * 8], [pool_stride, 1], fp4),
            KScale: T.StridedTensor([pool_pages, page_words], [scale_stride, 1], T.uint32),
            PageTable: T.Tensor([rows, table_width], T.int32),
            Lengths: T.Tensor([rows], T.int32),
            Active: T.Tensor([1], T.int32),
            Candidates: T.Tensor([rows, blocks], T.int32),
            CandidateLengths: T.Tensor([rows], T.int32),
            Scores: T.Tensor([rows, width], T.bfloat16),
        ):
            with T.Kernel(ctas, rows, threads=threads) as (bx, by):
                T.import_source(_SCORE_LOAD_HEADER)
                pdl.chain()
                linear_block = by * ctas + bx
                cta = linear_block // rows
                row = linear_block % rows
                K_shared = T.alloc_shared([block_P, D], fp4)
                SFK_shared = T.alloc_shared([block_P, 1], T.uint32)
                Q_shared = T.alloc_shared([heads, D], fp4)
                contrib = T.alloc_shared([block_P, heads], T.float32)
                T.call_extern(
                    "handle",
                    "__tl_score_row_candidates_fast",
                    T.address_of(Q[0, 0, 0]),
                    T.address_of(QScale[0, 0]),
                    T.address_of(Weights[0, 0]),
                    T.address_of(KPool[0, 0]),
                    T.address_of(KScale[0, 0]),
                    T.address_of(PageTable[0, 0]),
                    T.address_of(Candidates[0, 0]),
                    T.address_of(CandidateLengths[0]),
                    T.address_of(Scores[0, 0]),
                    T.address_of(K_shared[0, 0]),
                    T.address_of(SFK_shared[0, 0]),
                    T.address_of(Q_shared[0, 0]),
                    T.address_of(contrib[0, 0]),
                    row,
                    cta,
                    ctas,
                    rows,
                    width,
                    blocks,
                    page_size,
                    table_width,
                    pool_pages,
                    pool_stride,
                    scale_stride,
                    scale_word0,
                )

        return indexer_score

    @T.prim_func
    def indexer_score(
        Q: T.Tensor([rows, heads, D], fp4),
        QScale: T.Tensor([rows, heads], T.uint32),
        Weights: T.Tensor([rows, heads], T.bfloat16),
        KPool: T.StridedTensor([pool_pages, page_words * 8], [pool_stride, 1], fp4),
        KScale: T.StridedTensor([pool_pages, page_words], [scale_stride, 1], T.uint32),
        PageTable: T.Tensor([rows, table_width], T.int32),
        Lengths: T.Tensor([rows], T.int32),
        Active: T.Tensor([1], T.int32),
        Candidates: T.Tensor([rows, blocks], T.int32),
        CandidateLengths: T.Tensor([rows], T.int32),
        Scores: T.Tensor([rows, width], T.bfloat16),
    ):
        with T.Kernel(ctas, rows, threads=threads) as (cta, row):
            if use_vec128:
                T.import_source(_SCORE_LOAD_HEADER)
            pdl.chain()
            extent = _row_extent(CandidateLengths, None, row, width)
            score_row(
                Q,
                QScale,
                Weights,
                KPool,
                KScale,
                PageTable,
                Candidates,
                row,
                cta,
                extent,
                lambda p0, p: _candidate_position(Candidates, row, p0 + p, extent),
                lambda p0, p: _candidate_block(Candidates, row, p0 + p) >= 0,
                lambda p0, p: True,
                lambda p0: p0 + block_P <= extent,
                lambda p0, p: _candidate_position(Candidates, row, p0 + p),
                Scores,
            )

    return indexer_score


def _mxfp4_word(vals, first, scale):
    """Eight scaled values ``vals[first : first + 8]`` packed as four E2M1 bytes."""
    return T.call_extern(
        "uint32",
        "dsv41_e2m1x8",
        T.fmul(T.cast(vals[first], T.float32), scale),
        T.fmul(T.cast(vals[first + 1], T.float32), scale),
        T.fmul(T.cast(vals[first + 2], T.float32), scale),
        T.fmul(T.cast(vals[first + 3], T.float32), scale),
        T.fmul(T.cast(vals[first + 4], T.float32), scale),
        T.fmul(T.cast(vals[first + 5], T.float32), scale),
        T.fmul(T.cast(vals[first + 6], T.float32), scale),
        T.fmul(T.cast(vals[first + 7], T.float32), scale),
    )


def _mxfp4_store_group(ptr, vals, scale):
    """Quantize and store 32 values (16 packed E2M1 bytes) with a 128-bit store."""
    return T.call_extern(
        "void",
        "dsv41_store_e2m1x32",
        ptr,
        _mxfp4_word(vals, 0, scale),
        _mxfp4_word(vals, 8, scale),
        _mxfp4_word(vals, 16, scale),
        _mxfp4_word(vals, 24, scale),
    )


@tilelang.jit
def mxfp4_quantize_v41(
    paged: bool = False,
    page_size: int = 64,
    threads: int = 128,
    rope_heads: int = 0,
    rope_dim: int = 0,
    ratio: int = 1,
):
    """MXFP4-quantize ``[rows, 128]`` BF16 vectors, one thread per 32-value group.

    Each group's UE8M0 scale is ``pow2_ceil(max(amax, 6 * 2^-126) * fp32(1/6))``
    and its values round to nearest even E2M1, as B12X's quantizer does, so
    both backends see identical index queries and keys. With ``rope_heads > 0``,
    the trailing ``rope_dim`` columns of each head are rotated in registers at
    ``Positions`` (``rope_kernel``'s GPT-J interleaved ``ieee_fmaf`` arithmetic)
    before quantization. The output is either contiguous (``Q`` ``[rows, 64]``
    packed, ``S`` ``[rows, 4]``) or, with ``paged``, written through ``Slots``
    (``[rows]`` int64; negative slots skip) into index-cache pages ``Pool``
    (``[pages, page bytes]``, any row stride).
    """
    rows, pool_pages, pool_stride = T.dynamic("rows, pool_pages, pool_stride")
    tokens, positions_max = T.dynamic("tokens, positions_max")
    groups = INDEX_DIM // SF_BLOCK
    half = SF_BLOCK // 2
    assert rope_heads == 0 or (rope_dim > 0 and rope_dim % SF_BLOCK == 0)
    start_group = groups - (rope_dim // SF_BLOCK) if rope_heads > 0 else groups
    rope_half = rope_dim // 2

    @T.macro
    def body(X, Positions, CosSin, Q, S, Slots, Pool):
        with T.Kernel(T.ceildiv(rows * groups, threads), threads=threads) as bx:
            T.import_source(_E2M1_HEADER)
            tx = T.get_thread_binding()
            g = bx * threads + tx
            row, k = g // groups, g % groups
            vals = T.alloc_local([SF_BLOCK], T.bfloat16)
            pos_l = T.alloc_local([1], T.int64)
            slot_l = T.alloc_local([1], T.int64)
            amax = T.alloc_var(T.float32)
            if row < rows:
                if rope_heads > 0:
                    if k >= start_group:
                        token = row // rope_heads
                        pos_l[0] = T.cast(
                            Positions[token] // ratio * ratio
                            if ratio > 1
                            else Positions[token],
                            T.int64,
                        )
                if paged:
                    slot_l[0] = Slots[row]
            pdl.chain()
            if row < rows:
                for i in T.serial(SF_BLOCK):
                    vals[i] = X[row, k * SF_BLOCK + i]
                if rope_heads > 0:
                    if k >= start_group:
                        pos = pos_l[0]
                        pair_base = (k - start_group) * half
                        for i in T.unroll(half):
                            first = T.cast(vals[2 * i], T.float32)
                            second = T.cast(vals[2 * i + 1], T.float32)
                            cosine = CosSin[pos, pair_base + i]
                            sine = CosSin[pos, rope_half + pair_base + i]
                            vals[2 * i] = T.cast(
                                T.ieee_fmaf(first, cosine, -(second * sine)),
                                T.bfloat16,
                            )
                            vals[2 * i + 1] = T.cast(
                                T.ieee_fmaf(first, sine, second * cosine),
                                T.bfloat16,
                            )
                amax = 6.0 * 2.0**-126
                for i in T.serial(SF_BLOCK):
                    amax = T.max(amax, T.abs(T.cast(vals[i], T.float32)))
                bits = T.reinterpret(T.fmul(amax, T.float32(1.0 / 6.0)), T.uint32)
                sf = T.cast(T.shift_right(bits, 23) & 0xFF, T.int32) + T.cast(
                    (bits & 0x7FFFFF) != 0, T.int32
                )
                scale = T.reinterpret(
                    T.cast(T.shift_left(254 - sf, 23), T.uint32), T.float32
                )
                if paged:
                    slot = slot_l[0]
                    page, token = slot // page_size, slot % page_size
                    if (slot >= 0) & (page < pool_pages):
                        _mxfp4_store_group(
                            T.address_of(
                                Pool[page, token * (INDEX_DIM // 2) + k * half]
                            ),
                            vals,
                            scale,
                        )
                        Pool[
                            page, page_size * (INDEX_DIM // 2) + token * groups + k
                        ] = T.cast(sf, T.uint8)
                else:
                    _mxfp4_store_group(
                        T.address_of(Q[row, k * half]),
                        vals,
                        scale,
                    )
                    S[row, k] = T.cast(sf, T.uint8)

    if paged:
        if rope_heads > 0:

            @T.prim_func
            def indexer_mxfp4_quantize(
                X: T.Tensor([rows, INDEX_DIM], T.bfloat16),
                Positions: T.Tensor([tokens], T.int64),
                CosSin: T.Tensor([positions_max, rope_dim], T.float32),
                Slots: T.Tensor([rows], T.int64),
                Pool: T.StridedTensor(
                    [pool_pages, page_bytes(page_size)], [pool_stride, 1], T.uint8
                ),
            ):
                body(X, Positions, CosSin, None, None, Slots, Pool)

            return indexer_mxfp4_quantize

        @T.prim_func
        def indexer_mxfp4_quantize(
            X: T.Tensor([rows, INDEX_DIM], T.bfloat16),
            Slots: T.Tensor([rows], T.int64),
            Pool: T.StridedTensor(
                [pool_pages, page_bytes(page_size)], [pool_stride, 1], T.uint8
            ),
        ):
            body(X, None, None, None, None, Slots, Pool)

        return indexer_mxfp4_quantize

    if rope_heads > 0:

        @T.prim_func
        def indexer_mxfp4_quantize(
            X: T.Tensor([rows, INDEX_DIM], T.bfloat16),
            Positions: T.Tensor([tokens], T.int64),
            CosSin: T.Tensor([positions_max, rope_dim], T.float32),
            Q: T.Tensor([rows, INDEX_DIM // 2], T.uint8),
            S: T.Tensor([rows, groups], T.uint8),
        ):
            body(X, Positions, CosSin, Q, S, None, None)

        return indexer_mxfp4_quantize

    @T.prim_func
    def indexer_mxfp4_quantize(
        X: T.Tensor([rows, INDEX_DIM], T.bfloat16),
        Q: T.Tensor([rows, INDEX_DIM // 2], T.uint8),
        S: T.Tensor([rows, groups], T.uint8),
    ):
        body(X, None, None, Q, S, None, None)

    return indexer_mxfp4_quantize


def quantize_mxfp4(x, q, scales, *, positions=None, cos_sin=None, ratio: int = 1):
    """MXFP4-quantize ``[..., 128]`` BF16 into ``q`` and ``scales``.

    ``q`` is ``[..., 64]`` packed E2M1; ``scales`` is ``[..., 4]`` UE8M0. With
    ``positions`` and ``cos_sin``, rotates the trailing ``cos_sin.shape[1]``
    columns of each head in registers before quantizing.
    """
    if x.shape[0] == 0:
        return q, scales
    if positions is not None and cos_sin is not None:
        heads = x.shape[1] if x.ndim == 3 else 1
        _make(
            quantize_call(
                rope_heads=heads, rope_dim=cos_sin.shape[1], ratio=ratio
            )
        )(
            x.reshape(-1, INDEX_DIM),
            positions,
            cos_sin,
            q.view(-1, INDEX_DIM // 2),
            scales.view(-1, INDEX_DIM // SF_BLOCK),
        )
        return q, scales
    _make(quantize_call())(
        x.reshape(-1, INDEX_DIM),
        q.view(-1, INDEX_DIM // 2),
        scales.view(-1, INDEX_DIM // SF_BLOCK),
    )
    return q, scales


def write_index_keys(keys, pool, slots, page_size):
    """Quantize ``[rows, 128]`` keys into index-cache pages at ``slots`` (-1 skips)."""
    kernel = _make(quantize_call(page_size))
    kernel(keys, slots.to(torch.int64), pool[:, : page_bytes(page_size)])


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def mxfp4_norm_rope_quantize_v41(
    eps: float,
    page_size: int = 64,
    ratio: int = 1,
    slot_dtype: str = "int64",
    shards: int = 1,
):
    """Fused (optional split-K ``shards`` reduction +) 128-column RMSNorm
    (TileKernels' ``norm_forward`` reduction), trailing 64-column GPT-J RoPE
    (``rope_kernel``) and paged MXFP4 key write (``mxfp4_quantize_v41(paged=True)``)
    in one launch, bit for bit."""
    from .front import ROPE_DIM, _THREADS, _VEC, _row_split

    k = _row_split(INDEX_DIM)
    assert (
        k["num_row_warps"] == 1
        and k["row_threads"] == 8
        and k["num_thread_vec"] == 2
        and ROPE_DIM == 64
    )
    kb, kt = k["block_m"], k["row_threads"]
    half = ROPE_DIM // 2
    pairs = _VEC // 2
    out_scale = 1.0
    rows, positions_max = T.dynamic("rows, positions_max")
    pool_pages, pool_stride = T.dynamic("pool_pages, pool_stride")

    @T.macro
    def body(X, Weight, Positions, CosSin, Slots, Pool):
        with T.Kernel(T.ceildiv(rows, kb), threads=_THREADS) as bx:
            T.import_source(_E2M1_HEADER)
            tid = T.get_thread_binding()
            k_sum_partial = T.alloc_fragment((kb, kt), T.float32)
            k_warp_reducer = T.alloc_reducer((kb, 1), T.float32, op="sum")
            k_warp_fragment = T.alloc_fragment((kb, 1), T.float32)
            k_sum_shared = T.alloc_shared((kb, 1), T.float32)
            k_row_partial = T.alloc_fragment((kb, kt, 1), T.float32)
            k_sum_reducer = T.alloc_reducer((kb, kt), T.float32, op="sum")
            k_sum_fragment = T.alloc_fragment((kb, kt), T.float32)
            T.annotate_layout(
                {
                    k_sum_partial: k["row_layout"],
                    k_warp_fragment: k["warp_layout"],
                    k_row_partial: k["row_reduce_layout"],
                    k_sum_fragment: k["row_layout"],
                }
            )
            k_x = T.alloc_local((k["num_thread_vec"], _VEC), T.bfloat16)
            k_y = T.alloc_local((k["num_thread_vec"], _VEC), T.bfloat16)
            acc_local = T.alloc_local((_VEC,), T.float32)
            y_fp32_local = T.alloc_local((_VEC,), T.float32)
            sum_local = T.alloc_local((2,), T.float32)
            weight_local = T.alloc_local((k["num_thread_vec"], _VEC), T.bfloat16)
            cs = T.alloc_local((_VEC,), T.float32)
            slot_local = T.alloc_local((1,), T.int64)

            row_id = tid // kt
            lane = tid % kt
            token_id = bx * kb + row_id
            for vi in T.unroll(k["num_thread_vec"]):
                token_offset = (vi * kt + lane) * _VEC
                for j in T.vectorized(_VEC):
                    weight_local[vi, j] = Weight[token_offset + j]
            if token_id < rows:
                slot_local[0] = T.cast(Slots[token_id], T.int64)
                pos = (
                    Positions[token_id] // ratio * ratio
                    if ratio > 1
                    else Positions[token_id]
                )
                for i in T.vectorized(pairs):
                    cs[i] = CosSin[pos, lane * pairs + i]
                for i in T.vectorized(pairs):
                    cs[pairs + i] = CosSin[pos, half + lane * pairs + i]
            else:
                slot_local[0] = T.int64(-1)
            pdl.chain()
            T.clear(sum_local)
            for vi in T.unroll(k["num_thread_vec"]):
                token_offset = (vi * kt + lane) * _VEC
                if token_id < rows:
                    if shards > 1:
                        for j in T.vectorized(_VEC):
                            acc_local[j] = X[0, token_id, token_offset + j]
                        for s in T.serial(1, shards):
                            for j in T.vectorized(_VEC):
                                acc_local[j] += X[s, token_id, token_offset + j]
                        for j in T.unroll(_VEC):
                            k_x[vi, j] = T.cast(acc_local[j], T.bfloat16)
                    else:
                        for j in T.vectorized(_VEC):
                            k_x[vi, j] = X[token_id, token_offset + j]
                else:
                    T.clear(k_x[vi, :])
                for j in T.vectorized(_VEC):
                    sum_local[j % 2] += T.float32(k_x[vi, j]) * T.float32(k_x[vi, j])
            k_sum_partial[row_id, lane] = sum_local[0] + sum_local[1]
            T.reducer_init(k_warp_reducer)
            for i, w, j in T.Parallel(
                kb, 1, kt, loop_layout=k["warp_reduce_layout"]
            ):
                T.reducer_update(k_warp_reducer[i, w], k_sum_partial[i, w * kt + j])
            T.finalize_reducer(k_warp_reducer, k_warp_fragment)
            if lane == 0:
                k_sum_shared[row_id, 0] = k_warp_fragment[row_id, 0]
            T.sync_threads()
            for i, j, w in T.Parallel(kb, kt, 1, loop_layout=k["row_reduce_layout"]):
                k_row_partial[i, j, w] = k_sum_shared[i, w]
            T.reducer_init(k_sum_reducer)
            for i, j, w in T.Parallel(kb, kt, 1, loop_layout=k["row_reduce_layout"]):
                T.reducer_update(k_sum_reducer[i, j], k_row_partial[i, j, w])
            T.finalize_reducer(k_sum_reducer, k_sum_fragment)

            rstd_value = T.alloc_var(
                T.float32,
                init=T.rsqrt(k_sum_fragment[row_id, lane] / INDEX_DIM + eps),
            )
            for vi in T.unroll(k["num_thread_vec"]):
                for j in T.vectorized(_VEC):
                    y_fp32_local[j] = (
                        T.float32(k_x[vi, j])
                        * rstd_value
                        * (T.float32(weight_local[vi, j]) * out_scale)
                    )
                for j in T.vectorized(_VEC):
                    k_y[vi, j] = y_fp32_local[j]

            if token_id < rows:
                for j in T.unroll(pairs):
                    first = T.cast(k_y[1, 2 * j], T.float32)
                    second = T.cast(k_y[1, 2 * j + 1], T.float32)
                    cosine = cs[j]
                    sine = cs[pairs + j]
                    k_y[1, 2 * j] = T.cast(
                        T.ieee_fmaf(first, cosine, -(second * sine)), T.bfloat16
                    )
                    k_y[1, 2 * j + 1] = T.cast(
                        T.ieee_fmaf(first, sine, second * cosine), T.bfloat16
                    )

            for vi in T.unroll(k["num_thread_vec"]):
                group = vi * 2 + (lane // 4)
                group_lane = lane % 4
                amax = T.alloc_var(T.float32, init=6.0 * 2.0**-126)
                for j in T.unroll(_VEC):
                    amax = T.max(amax, T.abs(T.cast(k_y[vi, j], T.float32)))
                amax = T.max(amax, T.shfl_xor(amax, 1))
                amax = T.max(amax, T.shfl_xor(amax, 2))
                if token_id < rows:
                    slot = slot_local[0]
                    page = slot // page_size
                    tok = slot % page_size
                    if (slot >= 0) & (page < pool_pages):
                        bits = T.reinterpret(
                            T.fmul(amax, T.float32(1.0 / 6.0)), T.uint32
                        )
                        sf = T.cast(T.shift_right(bits, 23) & 0xFF, T.int32) + T.cast(
                            (bits & 0x7FFFFF) != 0, T.int32
                        )
                        scale = T.reinterpret(
                            T.cast(T.shift_left(254 - sf, 23), T.uint32), T.float32
                        )
                        T.call_extern(
                            "void",
                            "dsv41_store_e2m1x8",
                            T.address_of(
                                Pool[
                                    page,
                                    tok * (INDEX_DIM // 2)
                                    + group * 16
                                    + group_lane * pairs,
                                ]
                            ),
                            T.fmul(T.cast(k_y[vi, 0], T.float32), scale),
                            T.fmul(T.cast(k_y[vi, 1], T.float32), scale),
                            T.fmul(T.cast(k_y[vi, 2], T.float32), scale),
                            T.fmul(T.cast(k_y[vi, 3], T.float32), scale),
                            T.fmul(T.cast(k_y[vi, 4], T.float32), scale),
                            T.fmul(T.cast(k_y[vi, 5], T.float32), scale),
                            T.fmul(T.cast(k_y[vi, 6], T.float32), scale),
                            T.fmul(T.cast(k_y[vi, 7], T.float32), scale),
                        )
                        if group_lane == 0:
                            Pool[
                                page,
                                page_size * (INDEX_DIM // 2) + tok * 4 + group,
                            ] = T.cast(sf, T.uint8)

    if shards > 1:

        @T.prim_func
        def indexer_mxfp4_quantize(
            P: T.Tensor([shards, rows, INDEX_DIM], T.float32),
            Weight: T.Tensor([INDEX_DIM], T.bfloat16),
            Positions: T.Tensor([rows], T.int64),
            CosSin: T.Tensor([positions_max, ROPE_DIM], T.float32),
            Slots: T.Tensor([rows], slot_dtype),
            Pool: T.StridedTensor(
                [pool_pages, page_bytes(page_size)], [pool_stride, 1], T.uint8
            ),
        ):
            body(P, Weight, Positions, CosSin, Slots, Pool)

        return indexer_mxfp4_quantize

    @T.prim_func
    def indexer_mxfp4_quantize(
        X: T.Tensor([rows, INDEX_DIM], T.bfloat16),
        Weight: T.Tensor([INDEX_DIM], T.bfloat16),
        Positions: T.Tensor([rows], T.int64),
        CosSin: T.Tensor([positions_max, ROPE_DIM], T.float32),
        Slots: T.Tensor([rows], slot_dtype),
        Pool: T.StridedTensor(
            [pool_pages, page_bytes(page_size)], [pool_stride, 1], T.uint8
        ),
    ):
        body(X, Weight, Positions, CosSin, Slots, Pool)

    return indexer_mxfp4_quantize


def norm_rope_write_index_keys(
    raw_keys, weight, eps: float, positions, cos_sin, pool, slots, page_size: int, ratio: int = 1
):
    """Fused (optional split-K reduction +) RMSNorm + RoPE + paged MXFP4 quantize/write
    of ``[rows, 128]`` BF16 keys or ``[shards, rows, 128]`` FP32 split-K partials."""
    if raw_keys.shape[-2] == 0:
        return
    shards = raw_keys.shape[0] if raw_keys.ndim == 3 else 1
    kernel = _make(
        norm_quantize_call(eps, page_size, ratio, slots.dtype, shards=shards)
    )
    kernel(
        raw_keys,
        weight,
        positions,
        cos_sin,
        slots,
        pool[:, : page_bytes(page_size)],
    )


def _key16(x):
    """Order-preserving uint16 key of a BF16 value, as int32."""
    bits = T.cast(T.reinterpret(x, T.uint16), T.int32)
    return T.if_then_else(bits >= 0x8000, bits ^ 0xFFFF, bits | 0x8000)


def _value16(key):
    """The BF16 value of an order-preserving key."""
    return T.reinterpret(
        T.cast(T.if_then_else(key >= 0x8000, key ^ 0x8000, key ^ 0xFFFF), T.uint16),
        T.bfloat16,
    )


_TOPK_THREADS = 512
_SEGMENT_ENTRIES = _TOPK_THREADS * 8
_LONG_TOPK_MAX = CANDIDATE_BLOCKS
_LONG_SEGMENT_ENTRIES = _TOPK_THREADS * 32


def _segment_entries(topk: int) -> int:
    return _LONG_SEGMENT_ENTRIES if topk <= _LONG_TOPK_MAX else _SEGMENT_ENTRIES


def _swizzle_key_idx(idx, swizzle: bool = False):
    return (idx ^ ((idx >> 5) & 31)) if swizzle else idx


def _select_macros(threads, topk, swizzle: bool = False):
    """Radix threshold and ordered compaction over keys held in shared memory."""
    radix = 256
    steps = threads.bit_length() - 1

    @T.macro
    def block_scan(values, warp_sums, tx, n_steps, warp_offset: int = 0):
        # Inclusive scan of values[0:(1 << n_steps)] via intra-warp shuffle and warp spine.
        val = T.alloc_var(T.int32)
        prior = T.alloc_var(T.int32)
        if n_steps <= 5:
            count = 1 << n_steps
            val = T.if_then_else(tx < count, values[T.min(tx, count - 1)], 0)
            for s in T.unroll(n_steps):
                prior = T.shfl_up(val, 1 << s)
                if tx >= (1 << s):
                    val += prior
            if tx < count:
                values[tx] = val
        else:
            count = 1 << n_steps
            warps = count // 32
            warp_steps = n_steps - 5
            w_val = T.alloc_var(T.int32)
            warp_base = T.alloc_var(T.int32)
            lane = tx % 32
            warp = tx // 32
            val = T.if_then_else(tx < count, values[T.min(tx, count - 1)], 0)
            for s in T.unroll(5):
                prior = T.shfl_up(val, 1 << s)
                if lane >= (1 << s):
                    val += prior
            if (warp < warps) & (lane == 31):
                warp_sums[warp_offset + warp] = val
            T.sync_threads()
            w_val = T.if_then_else(
                lane < warps, warp_sums[warp_offset + T.min(lane, warps - 1)], 0
            )
            for s in T.unroll(warp_steps):
                prior = T.shfl_up(w_val, 1 << s)
                if lane >= (1 << s):
                    w_val += prior
            warp_base = T.if_then_else(
                warp > 0, T.shfl_sync(w_val, T.max(warp - 1, 0)), 0
            )
            if tx < count:
                values[tx] = warp_base + val

    @T.macro
    def suffix_counts(hist, scan, tx):
        # hist[b] <- number of keys whose digit is >= b; hist[radix] = 0.
        if tx < radix:
            scan[tx] = hist[radix - 1 - tx]
        T.sync_threads()
        if threads >= 512:
            block_scan(scan, scan, tx, 8, warp_offset=256)
        else:
            warp_sums = T.alloc_shared([32], T.int32)
            block_scan(scan, warp_sums, tx, 8, warp_offset=0)
        if tx < radix:
            hist[radix - 1 - tx] = scan[tx]
        if tx == 0:
            hist[radix] = 0
        T.sync_threads()

    @T.macro
    def threshold(keys, n, tx, hist, scan, params):
        """Top ``topk`` of keys[0:n]: params[0] <- threshold, params[1] <- ties kept."""
        key = T.alloc_var(T.int32)
        if tx < radix + 1:
            hist[tx] = 0
        if tx == 0:
            params[2] = 0
            params[3] = 0
        T.sync_threads()
        for i in T.serial(T.ceildiv(n, threads)):
            if i * threads + tx < n:
                key = T.cast(keys[_swizzle_key_idx(i * threads + tx, swizzle)], T.int32)
                if key > _KEY_NEG_INF:
                    T.atomic_add(hist[key >> 8], 1)
        T.sync_threads()
        suffix_counts(hist, scan, tx)
        if tx < radix:  # noqa: SIM102 - the guard keeps hist[tx + 1] in bounds; TIR & is not short-circuit
            if (hist[tx] >= topk) & (hist[tx + 1] < topk):
                params[2] = tx
        T.sync_threads()
        hi = T.alloc_var(T.int32)
        above = T.alloc_var(T.int32)
        selectable = T.alloc_var(T.int32)
        hi = params[2]
        above = hist[hi + 1]
        selectable = hist[0]
        T.sync_threads()
        if tx < radix + 1:
            hist[tx] = 0
        T.sync_threads()
        for i in T.serial(T.ceildiv(n, threads)):
            if i * threads + tx < n:
                key = T.cast(keys[_swizzle_key_idx(i * threads + tx, swizzle)], T.int32)
                if (key > _KEY_NEG_INF) & ((key >> 8) == hi):
                    T.atomic_add(hist[key & 0xFF], 1)
        T.sync_threads()
        suffix_counts(hist, scan, tx)
        if tx < radix:  # noqa: SIM102 - the guard keeps hist[tx + 1] in bounds
            if (above + hist[tx] >= topk) & (above + hist[tx + 1] < topk):
                params[3] = tx
        T.sync_threads()
        if tx == 0:
            if selectable <= topk:
                # Everything selectable survives.
                params[0] = _KEY_NEG_INF + 1
                params[1] = selectable
            else:
                params[0] = (hi << 8) | params[3]
                params[1] = topk - above - hist[params[3] + 1]
        T.sync_threads()

    @T.macro
    def compact(keys, per, tx, scan, params, sel_idx, sel_key, n=None, warp_sums=None):
        """Write the selected indices and keys to sel_idx/sel_key in index order.

        Ties at the threshold keep the lowest indices. Each thread owns
        keys[tx * own : (tx + 1) * own], where ``own`` is ``per``, or with an
        active count ``n`` the ``ceil(n / threads)`` that spreads keys[0:n]
        over every thread. Leaves the selected count in scan[threads - 1].
        """
        key = T.alloc_var(T.int32)
        n_gt = T.alloc_var(T.int32)
        n_eq = T.alloc_var(T.int32)
        sel_eq = T.alloc_var(T.int32)
        out_pos = T.alloc_var(T.int32)
        taken = T.alloc_var(T.int32)
        thr = T.alloc_var(T.int32)
        need = T.alloc_var(T.int32)
        thr = params[0]
        need = params[1]
        n_gt = 0
        n_eq = 0
        own = per if n is None else T.ceildiv(n, threads)
        if n is None or tx * own < n:
            for v in T.serial(own):
                if n is None or tx * own + v < n:
                    key = T.cast(keys[_swizzle_key_idx(tx * own + v, swizzle)], T.int32)
                    if key > thr:
                        n_gt += 1
                    elif key == thr:
                        n_eq += 1
        if warp_sums is None:
            warp_sums = T.alloc_shared([32], T.int32)
        if per * threads < 32768:
            # One scan of packed (ties << 16 | above) counts: the ties kept before
            # this thread are min(need, ties before it), so out_pos needs no
            # second scan. Both fields stay below 2**15.
            scan[tx] = (n_eq << 16) | n_gt
            T.sync_threads()
            block_scan(scan, warp_sums, tx, steps)
            eq_before = T.alloc_var(T.int32)
            gt_before = T.alloc_var(T.int32)
            eq_before = (scan[tx] >> 16) - n_eq
            gt_before = (scan[tx] & 0xFFFF) - n_gt
            sel_eq = T.max(T.min(need - eq_before, n_eq), 0)
            out_pos = gt_before + T.min(T.max(need, 0), eq_before)
            T.sync_threads()
            if tx == threads - 1:
                scan[tx] = gt_before + n_gt + T.min(T.max(need, 0), eq_before + n_eq)
        else:
            scan[tx] = n_eq
            T.sync_threads()
            block_scan(scan, warp_sums, tx, steps)
            sel_eq = T.max(T.min(need - (scan[tx] - n_eq), n_eq), 0)
            T.sync_threads()
            scan[tx] = n_gt + sel_eq
            T.sync_threads()
            block_scan(scan, warp_sums, tx, steps)
            out_pos = scan[tx] - n_gt - sel_eq
        taken = 0
        if (n_gt > 0) | (sel_eq > 0):
            for v in T.serial(own):
                if n is None or tx * own + v < n:
                    key = T.cast(keys[_swizzle_key_idx(tx * own + v, swizzle)], T.int32)
                    if (key > thr) | ((key == thr) & (taken < sel_eq)):
                        if key == thr:
                            taken += 1
                        sel_idx[out_pos] = tx * own + v
                        sel_key[out_pos] = T.cast(key, sel_key.dtype)
                        out_pos += 1
        T.sync_threads()

    return threshold, compact


def _final_writer(threads, topk, group):
    @T.macro
    def write_final(
        Indices, Values, Counts, Gather, row, extent, valid, entry, value, params
    ):
        """Write the selected entries (``entry(slot)`` where ``valid(slot)``), ``-1`` padded.

        Grouped entries are block ids; ``Counts`` receives their positions below
        ``extent``. ``Gather`` (candidate rows) holds candidate block ids:
        entry ``e`` is position ``Gather[e // CANDIDATE_BLOCK] * CANDIDATE_BLOCK +
        e % CANDIDATE_BLOCK``.
        """
        tx = T.get_thread_binding()
        count = T.alloc_var(T.int32)
        count = 0
        for i in T.serial(T.ceildiv(topk, threads)):
            slot = i * threads + tx
            if slot < topk:
                if valid(slot):
                    e = entry(slot)
                    Values[row, slot] = value(slot)
                    if Gather is not None:
                        Indices[row, slot] = (
                            Gather[row, e // CANDIDATE_BLOCK] * CANDIDATE_BLOCK
                            + e % CANDIDATE_BLOCK
                        )
                    else:
                        Indices[row, slot] = e
                    if group > 1:
                        count += T.min(group, extent - e * group)
                else:
                    Values[row, slot] = -T.infinity(T.bfloat16)
                    Indices[row, slot] = -1
        if Counts is not None:
            T.sync_threads()
            if tx == 0:
                params[0] = 0
            T.sync_threads()
            T.atomic_add(params[0], count)
            T.sync_threads()
            if tx == 0:
                Counts[row] = params[0]

    @T.macro
    def write_empty_final(Indices, Values, Counts, row):
        tx = T.get_thread_binding()
        for i in T.serial(T.ceildiv(topk, threads)):
            slot = i * threads + tx
            if slot < topk:
                Values[row, slot] = -T.infinity(T.bfloat16)
                Indices[row, slot] = -1
        if Counts is not None:
            if tx == 0:
                Counts[row] = 0

    return write_final, write_empty_final


@tilelang.jit(
    pass_configs={tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True}
)
def indexer_topk_segment_v41(
    width: int,
    topk: int = INDEX_TOPK,
    group: int = 1,
    final: bool = True,
    candidates: bool = False,
    threads: int = _TOPK_THREADS,
    gather_scores: bool = False,
):
    """Stage 1: the exact top-``topk`` of each segment of a row.

    Inputs: ``Scores`` (``[rows, width]`` BF16), ``Lengths`` (``[rows]`` int32)
    and, unless ``candidates`` or ``gather_scores``, ``Active`` (``[1]`` int32):
    a row selects from its first ``min(Lengths, Active, width)`` positions. With
    ``group > 1`` an entry is a block of ``group`` positions keyed by its
    maximum; the block holding the last position always ranks first.

    ``final`` (one segment per row) writes ``Indices`` (``[rows, topk]`` int32
    ascending positions, or block ids with ``group > 1``, ``-1`` padded) and
    ``Values`` (``[rows, topk]`` BF16); ``candidates`` and ``gather_scores`` map
    entries through ``Gather`` (block ids), with ``gather_scores`` reading dense
    ``Scores`` at the gathered block positions, and ``group > 1`` also writes
    ``Counts`` (``[rows]`` positions kept). Otherwise it writes each live
    segment's winners, in entry order, to ``WorkValues`` / ``WorkEntries``
    (``[rows, segments * topk]``, ``-1`` padded); segments past a row's extent
    are left unwritten.
    """
    assert not ((candidates or gather_scores) and group > 1), (
        "candidate rows select positions, not blocks"
    )
    rows = T.dynamic("rows")
    entries = (
        CANDIDATE_BLOCKS * CANDIDATE_BLOCK
        if gather_scores
        else tilelang.cdiv(width, group)
    )
    segment = _segment_entries(topk)
    key_dtype = T.uint16 if topk <= _LONG_TOPK_MAX else T.int32
    swizzle = topk <= _LONG_TOPK_MAX
    segments = tilelang.cdiv(entries, segment)
    assert not final or segments == 1, "a final segment must cover the row"
    per = segment // threads
    threshold, compact = _select_macros(threads, topk, swizzle)
    write_final, write_empty_final = _final_writer(threads, topk, group)

    @T.macro
    def select_segment(
        Scores, Gather, extent, row, seg, keys, hist, scan, params, sel_idx, sel_key
    ):
        tx = T.get_thread_binding()
        best = T.alloc_var(T.int32)
        n_entries = T.ceildiv(extent, group)
        n_active = T.min(segment, T.max(n_entries - seg * segment, 0))
        for i in T.serial(T.ceildiv(n_active, threads)):
            j = i * threads + tx
            if j < n_active:
                e = seg * segment + j
                if gather_scores:
                    blk = Gather[row, e // CANDIDATE_BLOCK]
                    pos = blk * CANDIDATE_BLOCK + e % CANDIDATE_BLOCK
                    keys[_swizzle_key_idx(j, swizzle)] = T.cast(
                        T.if_then_else(
                            blk >= 0,
                            _key16(Scores[row, T.min(T.max(pos, 0), width - 1)]),
                            0,
                        ),
                        key_dtype,
                    )
                elif group == 1:
                    keys[_swizzle_key_idx(j, swizzle)] = T.cast(
                        _key16(Scores[row, T.min(e, width - 1)]), key_dtype
                    )
                else:
                    best = 0
                    for o in T.serial(group):
                        if e * group + o < extent:
                            best = T.max(
                                best,
                                _key16(Scores[row, T.min(e * group + o, width - 1)]),
                            )
                    # The block holding the last visible position always survives.
                    keys[_swizzle_key_idx(j, swizzle)] = T.cast(
                        T.if_then_else(e == (extent - 1) // group, 0xFF80, best),
                        key_dtype,
                    )
        T.sync_threads()
        threshold(keys, n_active, tx, hist, scan, params)
        compact(keys, per, tx, scan, params, sel_idx, sel_key, n_active, warp_sums=hist)

    @T.macro
    def body(
        Scores,
        Lengths,
        Active,
        Gather,
        Indices,
        Values,
        Counts,
        WorkValues,
        WorkEntries,
    ):
        with T.Kernel(segments, rows, threads=threads) as (seg, row):
            pdl.chain()
            tx = T.get_thread_binding()
            keys = T.alloc_shared([segment], key_dtype)
            hist = T.alloc_shared([257], T.int32)
            scan = T.alloc_shared([threads], T.int32)
            params = T.alloc_shared([4], T.int32)
            sel_idx = T.alloc_shared([topk], T.int32)
            sel_key = T.alloc_shared([topk], key_dtype)
            extent = _row_extent(
                Lengths, Active, row, entries if gather_scores else width
            )
            if final:
                if extent > 0:
                    select_segment(
                        Scores,
                        Gather,
                        extent,
                        row,
                        0,
                        keys,
                        hist,
                        scan,
                        params,
                        sel_idx,
                        sel_key,
                    )
                    total = scan[threads - 1]
                    write_final(
                        Indices,
                        Values,
                        Counts,
                        Gather,
                        row,
                        extent,
                        lambda slot: slot < total,
                        lambda slot: sel_idx[slot],
                        lambda slot: _value16(T.cast(sel_key[slot], T.int32)),
                        params,
                    )
                else:
                    write_empty_final(Indices, Values, Counts, row)
            elif seg * segment < T.ceildiv(extent, group):
                # Segments past the row's extent hold nothing; the merges skip them.
                select_segment(
                    Scores,
                    Gather,
                    extent,
                    row,
                    seg,
                    keys,
                    hist,
                    scan,
                    params,
                    sel_idx,
                    sel_key,
                )
                total = scan[threads - 1]
                for i in T.serial(T.ceildiv(topk, threads)):
                    slot = i * threads + tx
                    if slot < topk:
                        if slot < total:
                            WorkValues[row, seg * topk + slot] = _value16(
                                T.cast(sel_key[slot], T.int32)
                            )
                            WorkEntries[row, seg * topk + slot] = (
                                seg * segment + sel_idx[slot]
                            )
                        else:
                            WorkValues[row, seg * topk + slot] = -T.infinity(T.bfloat16)
                            WorkEntries[row, seg * topk + slot] = -1

    if not final:
        if candidates:

            @T.prim_func
            def indexer_topk_segment(
                Scores: T.Tensor([rows, width], T.bfloat16),
                Lengths: T.Tensor([rows], T.int32),
                WorkValues: T.Tensor([rows, segments * topk], T.bfloat16),
                WorkEntries: T.Tensor([rows, segments * topk], T.int32),
            ):
                body(
                    Scores,
                    Lengths,
                    None,
                    None,
                    None,
                    None,
                    None,
                    WorkValues,
                    WorkEntries,
                )

            return indexer_topk_segment

        @T.prim_func
        def indexer_topk_segment(
            Scores: T.Tensor([rows, width], T.bfloat16),
            Lengths: T.Tensor([rows], T.int32),
            Active: T.Tensor([1], T.int32),
            WorkValues: T.Tensor([rows, segments * topk], T.bfloat16),
            WorkEntries: T.Tensor([rows, segments * topk], T.int32),
        ):
            body(
                Scores, Lengths, Active, None, None, None, None, WorkValues, WorkEntries
            )

        return indexer_topk_segment

    if candidates or gather_scores:
        gather_cols = (
            CANDIDATE_BLOCKS if gather_scores else width // CANDIDATE_BLOCK
        )

        @T.prim_func
        def indexer_topk_segment(
            Scores: T.Tensor([rows, width], T.bfloat16),
            Lengths: T.Tensor([rows], T.int32),
            Gather: T.Tensor([rows, gather_cols], T.int32),
            Indices: T.Tensor([rows, topk], T.int32),
            Values: T.Tensor([rows, topk], T.bfloat16),
        ):
            body(Scores, Lengths, None, Gather, Indices, Values, None, None, None)

        return indexer_topk_segment

    if group > 1:

        @T.prim_func
        def indexer_topk_segment(
            Scores: T.Tensor([rows, width], T.bfloat16),
            Lengths: T.Tensor([rows], T.int32),
            Active: T.Tensor([1], T.int32),
            Indices: T.Tensor([rows, topk], T.int32),
            Values: T.Tensor([rows, topk], T.bfloat16),
            Counts: T.Tensor([rows], T.int32),
        ):
            body(Scores, Lengths, Active, None, Indices, Values, Counts, None, None)

        return indexer_topk_segment

    @T.prim_func
    def indexer_topk_segment(
        Scores: T.Tensor([rows, width], T.bfloat16),
        Lengths: T.Tensor([rows], T.int32),
        Active: T.Tensor([1], T.int32),
        Indices: T.Tensor([rows, topk], T.int32),
        Values: T.Tensor([rows, topk], T.bfloat16),
    ):
        body(Scores, Lengths, Active, None, Indices, Values, None, None, None)

    return indexer_topk_segment


@tilelang.jit(
    pass_configs={tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True}
)
def indexer_topk_merge_v41(
    width: int,
    topk: int = INDEX_TOPK,
    group: int = 1,
    lists: int = 2,
    span: int = _SEGMENT_ENTRIES,
    fan_in: int = 16,
    candidates: bool = False,
    threads: int = _TOPK_THREADS,
):
    """Merge winner lists: the exact top-``topk`` of every ``fan_in`` lists.

    Input list ``l`` of a row holds the winners, in ascending entry order, of
    entries ``[l * span, (l + 1) * span)``; lists past the row's extent
    (``Lengths`` and ``Active`` as in ``indexer_topk_segment_v41``) are never
    read. One output list leaves each group of ``fan_in`` inputs, again in
    entry order, so ties keep the lowest positions at every level exactly as
    a one-pass selection would. With one group the merge is final and writes
    ``Indices``, ``Values`` (and ``Counts``) like the final segment kernel;
    otherwise it writes ``OutValues`` / ``OutEntries`` (``[rows, groups *
    topk]``), leaving groups past the extent unwritten.
    """
    assert not (candidates and group > 1), "candidate rows select positions, not blocks"
    rows = T.dynamic("rows")
    groups = tilelang.cdiv(lists, fan_in)
    final = groups == 1
    n = min(lists, fan_in) * topk
    n_pad = tilelang.cdiv(n, threads) * threads
    per = n_pad // threads
    key_dtype = T.uint16 if topk <= _LONG_TOPK_MAX else T.int32
    key_bytes = 2 if topk <= _LONG_TOPK_MAX else 4
    swizzle = topk <= _LONG_TOPK_MAX
    assert n_pad * key_bytes <= 64 * 1024, (
        f"{n} merge keys exceed shared memory; lower fan_in"
    )
    threshold, compact = _select_macros(threads, topk, swizzle)
    write_final, write_empty_final = _final_writer(threads, topk, group)

    @T.macro
    def body(WorkValues, WorkEntries, Lengths, Active, Gather, Indices, Values, Counts):
        with T.Kernel(groups, rows, threads=threads) as (g, row):
            pdl.chain()
            tx = T.get_thread_binding()
            keys = T.alloc_shared([n_pad], key_dtype)
            hist = T.alloc_shared([257], T.int32)
            scan = T.alloc_shared([threads], T.int32)
            params = T.alloc_shared([4], T.int32)
            sel_idx = T.alloc_shared([topk], T.int32)
            sel_key = T.alloc_shared([topk], key_dtype)
            extent = _row_extent(Lengths, Active, row, width)
            live = T.min(
                T.ceildiv(T.ceildiv(extent, group), span), lists
            )  # input lists with entries
            base = g * fan_in * topk
            group_live = T.min(T.max(live - g * fan_in, 0), fan_in)
            if group_live <= 0:
                if final:
                    write_empty_final(Indices, Values, Counts, row)
            elif group_live == 1:
                # A single live input list is already sorted by ascending entry index
                # with -1 / -inf padding; forward it without radix selection.
                if final:
                    write_final(
                        Indices,
                        Values,
                        Counts,
                        Gather,
                        row,
                        extent,
                        lambda slot: WorkEntries[row, base + slot] >= 0,
                        lambda slot: WorkEntries[row, base + slot],
                        lambda slot: WorkValues[row, base + slot],
                        params,
                    )
                else:
                    for i in T.serial(T.ceildiv(topk, threads)):
                        slot = i * threads + tx
                        if slot < topk:
                            Values[row, g * topk + slot] = WorkValues[row, base + slot]
                            Indices[row, g * topk + slot] = WorkEntries[
                                row, base + slot
                            ]
            else:
                n_active = group_live * topk
                for i in T.serial(T.ceildiv(n_active, threads)):
                    j = i * threads + tx
                    if j < n_active:
                        c = base + j
                        keys[_swizzle_key_idx(j, swizzle)] = T.cast(
                            T.if_then_else(
                                WorkEntries[row, c] >= 0,
                                _key16(WorkValues[row, c]),
                                0,
                            ),
                            key_dtype,
                        )
                T.sync_threads()
                threshold(keys, n_active, tx, hist, scan, params)
                compact(
                    keys,
                    per,
                    tx,
                    scan,
                    params,
                    sel_idx,
                    sel_key,
                    n_active,
                    warp_sums=hist,
                )
                total = scan[threads - 1]
                if final:
                    write_final(
                        Indices,
                        Values,
                        Counts,
                        Gather,
                        row,
                        extent,
                        lambda slot: slot < total,
                        lambda slot: WorkEntries[row, sel_idx[slot]],
                        lambda slot: _value16(T.cast(sel_key[slot], T.int32)),
                        params,
                    )
                else:
                    for i in T.serial(T.ceildiv(topk, threads)):
                        slot = i * threads + tx
                        if slot < topk:
                            if slot < total:
                                Values[row, g * topk + slot] = _value16(
                                    T.cast(sel_key[slot], T.int32)
                                )
                                Indices[row, g * topk + slot] = WorkEntries[
                                    row, base + sel_idx[slot]
                                ]
                            else:
                                Values[row, g * topk + slot] = -T.infinity(T.bfloat16)
                                Indices[row, g * topk + slot] = -1

    if not final:
        if candidates:

            @T.prim_func
            def indexer_topk_merge(
                WorkValues: T.Tensor([rows, lists * topk], T.bfloat16),
                WorkEntries: T.Tensor([rows, lists * topk], T.int32),
                Lengths: T.Tensor([rows], T.int32),
                OutValues: T.Tensor([rows, groups * topk], T.bfloat16),
                OutEntries: T.Tensor([rows, groups * topk], T.int32),
            ):
                body(
                    WorkValues,
                    WorkEntries,
                    Lengths,
                    None,
                    None,
                    OutEntries,
                    OutValues,
                    None,
                )

            return indexer_topk_merge

        @T.prim_func
        def indexer_topk_merge(
            WorkValues: T.Tensor([rows, lists * topk], T.bfloat16),
            WorkEntries: T.Tensor([rows, lists * topk], T.int32),
            Lengths: T.Tensor([rows], T.int32),
            Active: T.Tensor([1], T.int32),
            OutValues: T.Tensor([rows, groups * topk], T.bfloat16),
            OutEntries: T.Tensor([rows, groups * topk], T.int32),
        ):
            body(
                WorkValues,
                WorkEntries,
                Lengths,
                Active,
                None,
                OutEntries,
                OutValues,
                None,
            )

        return indexer_topk_merge

    if candidates:

        @T.prim_func
        def indexer_topk_merge(
            WorkValues: T.Tensor([rows, lists * topk], T.bfloat16),
            WorkEntries: T.Tensor([rows, lists * topk], T.int32),
            Lengths: T.Tensor([rows], T.int32),
            Gather: T.Tensor([rows, width // CANDIDATE_BLOCK], T.int32),
            Indices: T.Tensor([rows, topk], T.int32),
            Values: T.Tensor([rows, topk], T.bfloat16),
        ):
            body(WorkValues, WorkEntries, Lengths, None, Gather, Indices, Values, None)

        return indexer_topk_merge

    if group > 1:

        @T.prim_func
        def indexer_topk_merge(
            WorkValues: T.Tensor([rows, lists * topk], T.bfloat16),
            WorkEntries: T.Tensor([rows, lists * topk], T.int32),
            Lengths: T.Tensor([rows], T.int32),
            Active: T.Tensor([1], T.int32),
            Indices: T.Tensor([rows, topk], T.int32),
            Values: T.Tensor([rows, topk], T.bfloat16),
            Counts: T.Tensor([rows], T.int32),
        ):
            body(
                WorkValues, WorkEntries, Lengths, Active, None, Indices, Values, Counts
            )

        return indexer_topk_merge

    @T.prim_func
    def indexer_topk_merge(
        WorkValues: T.Tensor([rows, lists * topk], T.bfloat16),
        WorkEntries: T.Tensor([rows, lists * topk], T.int32),
        Lengths: T.Tensor([rows], T.int32),
        Active: T.Tensor([1], T.int32),
        Indices: T.Tensor([rows, topk], T.int32),
        Values: T.Tensor([rows, topk], T.bfloat16),
    ):
        body(WorkValues, WorkEntries, Lengths, Active, None, Indices, Values, None)

    return indexer_topk_merge


_MERGE_KEYS = 8192  # keys one merge CTA holds in shared memory (32 KiB)
_LONG_MERGE_KEYS = 16384  # uint16 keys one merge CTA holds in shared memory (32 KiB)


def _merge_keys(topk: int) -> int:
    return _LONG_MERGE_KEYS if topk <= _LONG_TOPK_MAX else _MERGE_KEYS


def _take(scratch, specs):
    """Split caller scratch into ``specs`` tensors, allocating any that are missing."""
    if scratch is None:
        return [
            torch.empty(shape, dtype=dtype, device="cuda") for shape, dtype in specs
        ]
    assert len(scratch) == len(specs), (
        f"expected {len(specs)} scratch tensors, got {len(scratch)}"
    )
    return list(scratch)


class IndexerTopK:
    """Exact top-k over BF16 row scores: one pass, or segments plus merges.

    Rows longer than one segment select each segment's winners, then merge
    them ``fan_in`` lists at a time until one list remains, so any width fits
    SM120 shared memory. Every route returns the same positions: each merge
    sees exact winners in entry order and applies the same score-then-position
    order. Segments and merge groups past a row's extent are skipped.
    ``candidates`` rows select from their first ``Lengths`` candidates (a list
    of candidate block ids) and return the candidates' positions; grouped rows
    return block ids.
    """

    def __init__(
        self,
        width,
        topk=INDEX_TOPK,
        group=1,
        candidates=False,
        *,
        gather_scores=False,
    ):
        self.topk, self.group, self.candidates = topk, group, candidates
        self.gather_scores = gather_scores
        self.segments, first, levels = self.routes(
            width, topk, group, candidates, gather_scores=gather_scores
        )
        if self.segments == 1:
            self.single = _make(first)
        else:
            self.stage1 = _make(first)
        # (kernel, input lists, output lists)
        self.levels = [(_make(kernel), lists, out) for kernel, lists, out in levels]

    @staticmethod
    def routes(
        width,
        topk=INDEX_TOPK,
        group=1,
        candidates=False,
        *,
        gather_scores=False,
    ):
        """``(segments, first, levels)``: the segment count, the first kernel (the
        final segment kernel of a one-segment row, else stage 1) and the merge
        levels ``(kernel, input lists, output lists)``, kernels as
        ``prepare.call``s."""
        if gather_scores:
            first = prepare.call(
                indexer_topk_segment_v41,
                width,
                topk,
                group,
                True,
                candidates,
                _TOPK_THREADS,
                True,
            )
            return 1, first, []
        entries = tilelang.cdiv(width, group)
        seg_entries = _segment_entries(topk)
        segments = tilelang.cdiv(entries, seg_entries)
        levels = []
        if segments == 1:
            first = prepare.call(
                indexer_topk_segment_v41, width, topk, group, True, candidates
            )
            return segments, first, levels
        first = prepare.call(
            indexer_topk_segment_v41, width, topk, group, False, candidates
        )
        fan_in = max(2, _merge_keys(topk) // topk)
        lists, span = segments, seg_entries
        merge = indexer_topk_merge_v41
        if topk > 4096 and group == 1 and not candidates:
            from .wide_topk_merge import wide_indexer_topk_merge
            merge = wide_indexer_topk_merge
        while True:
            outputs = tilelang.cdiv(lists, fan_in)
            kernel = prepare.call(
                merge,
                width,
                topk,
                group,
                lists,
                span,
                fan_in,
                candidates,
            )
            levels.append((kernel, lists, outputs))
            if outputs == 1:
                break
            lists, span = outputs, span * fan_in
        return segments, first, levels

    @classmethod
    def calls(
        cls,
        width,
        topk=INDEX_TOPK,
        group=1,
        candidates=False,
        *,
        gather_scores=False,
    ) -> list:
        """The kernel calls ``IndexerTopK(width, topk, group, candidates)`` makes."""
        _, first, levels = cls.routes(
            width, topk, group, candidates, gather_scores=gather_scores
        )
        return [first, *(kernel for kernel, _, _ in levels)]

    def scratch_specs(self, rows):
        """Two ping-pong winner buffers (values, entries) for the merge levels."""
        if self.segments == 1:
            return ()
        widest = (self.segments * self.topk, self.levels[0][2] * self.topk)
        return tuple(
            ((rows * cols,), dtype)
            for cols in widest
            for dtype in (torch.bfloat16, torch.int32)
        )

    def run(
        self,
        scores,
        lengths,
        active,
        indices,
        values,
        *,
        gather=None,
        counts=None,
        scratch=None,
    ):
        """Select into ``indices`` / ``values``; grouped selections fill ``counts``.

        ``active`` is the ``[1]`` width cap of dense rows; candidate rows take
        ``gather`` (their candidate block ids) instead.
        """
        rows = scores.shape[0]
        uses_gather = self.candidates or self.gather_scores
        cap = () if uses_gather else (active,)
        final = (
            *((gather,) if uses_gather else ()),
            indices,
            values,
            *((counts,) if self.group > 1 else ()),
        )
        if self.segments == 1:
            self.single(scores, lengths, *cap, *final)
            return indices
        flat = _take(scratch, self.scratch_specs(rows))
        buffers = ((flat[0], flat[1]), (flat[2], flat[3]))

        def lists_of(buffer, lists):
            return tuple(
                t[: rows * lists * self.topk].view(rows, lists * self.topk)
                for t in buffer
            )

        self.stage1(scores, lengths, *cap, *lists_of(buffers[0], self.segments))
        for level, (kernel, lists, outputs) in enumerate(self.levels):
            work = lists_of(buffers[level % 2], lists)
            if outputs == 1:
                kernel(*work, lengths, *cap, *final)
            else:
                kernel(
                    *work, lengths, *cap, *lists_of(buffers[(level + 1) % 2], outputs)
                )
        return indices


class IndexerV41Plan:
    """Score and top-k kernels for one context width.

    ``candidates=True`` scores only the positions in a candidate list (V4.1's
    reindex layers). ``candidate_blocks=True`` also selects the top
    ``CANDIDATE_BLOCKS`` blocks of ``CANDIDATE_BLOCK`` positions, the
    candidate list a source layer publishes for the reindex layers after it.

    The plan holds no buffers: ``scratch_specs(rows)`` lists the scratch a
    call needs, which a caller can borrow from a shared workspace.
    """

    @staticmethod
    def _gather_from_blocks(width, topk, candidate_blocks) -> bool:
        return (
            candidate_blocks
            and width > CANDIDATE_BLOCKS * CANDIDATE_BLOCK
            and topk < CANDIDATE_BLOCKS
        )

    def __init__(
        self,
        width,
        page_size,
        *,
        topk=INDEX_TOPK,
        candidates=False,
        candidate_blocks=False,
    ):
        assert not (candidates and candidate_blocks), (
            "a layer reads or publishes candidates, not both"
        )
        self.width, self.page_size, self.topk = width, page_size, topk
        self.candidates, self.candidate_blocks = candidates, candidate_blocks
        self.score = _make(_score_call(width, page_size, candidates))
        gather_scores = self._gather_from_blocks(width, topk, candidate_blocks)
        self.select = IndexerTopK(
            width, topk, 1, candidates, gather_scores=gather_scores
        )
        self.select_blocks = (
            IndexerTopK(width, CANDIDATE_BLOCKS, CANDIDATE_BLOCK)
            if candidate_blocks
            else None
        )

    @staticmethod
    def calls(
        width,
        page_size,
        *,
        topk=INDEX_TOPK,
        candidates=False,
        candidate_blocks=False,
    ) -> list:
        """The kernel calls ``IndexerV41Plan`` of these arguments makes."""
        gather_scores = IndexerV41Plan._gather_from_blocks(
            width, topk, candidate_blocks
        )
        calls = [
            _score_call(width, page_size, candidates),
            *IndexerTopK.calls(
                width, topk, 1, candidates, gather_scores=gather_scores
            ),
        ]
        if candidate_blocks:
            calls.extend(IndexerTopK.calls(width, CANDIDATE_BLOCKS, CANDIDATE_BLOCK))
        return calls

    def scratch_specs(self, rows):
        specs = [
            ((rows, self.width), torch.bfloat16),  # scores
            ((rows, self.topk), torch.bfloat16),  # selected values
            *self.select.scratch_specs(rows),
        ]
        if self.select_blocks is not None:
            specs.append(((rows, CANDIDATE_BLOCKS), torch.bfloat16))
            specs.extend(self.select_blocks.scratch_specs(rows))
        return tuple(specs)

    def run(
        self,
        q_packed,
        q_scales,
        weights,
        pool,
        page_table,
        lengths,
        active,
        candidates=None,
        candidate_lengths=None,
        *,
        out=None,
        candidate_out=None,
        candidate_out_lengths=None,
        scratch=None,
    ):
        """Select each row's top positions into ``out``; returns ``(out, scores)``.

        Reindex plans take ``candidates`` (``[rows, CANDIDATE_BLOCKS]`` block ids)
        / ``candidate_lengths`` (positions); source plans write ``candidate_out``
        (``[rows, CANDIDATE_BLOCKS]`` ascending block ids, ``-1`` padded) and
        ``candidate_out_lengths`` (its count of positions). Scores past a row's
        extent are unwritten.
        """
        rows = q_packed.shape[0]
        buffers = _take(scratch, self.scratch_specs(rows))
        scores, values = buffers[:2]
        select_scratch = len(self.select.scratch_specs(rows))
        select_buffers = buffers[2 : 2 + select_scratch]
        if out is None:
            out = torch.empty(
                (rows, self.topk), dtype=torch.int32, device=q_packed.device
            )
        nbytes = page_bytes(self.page_size)
        args = (
            q_packed.view(torch.int8),
            q_scales.view(torch.uint32).reshape(rows, -1),
            weights,
            # Pages may carry allocator padding past their records.
            pool.view(torch.int8)[:, :nbytes],
            pool.view(torch.uint32)[:, : nbytes // 4],
            page_table,
            lengths,
            active,
        )
        if self.candidates:
            self.score(*args, candidates, candidate_lengths, scores)
            self.select.run(
                scores,
                candidate_lengths,
                None,
                out,
                values,
                gather=candidates,
                scratch=select_buffers,
            )
            return out, scores
        self.score(*args, scores)
        if self.select_blocks is not None:
            block_values, *block_buffers = buffers[2 + select_scratch :]
            self.select_blocks.run(
                scores,
                lengths,
                active,
                candidate_out,
                block_values,
                counts=candidate_out_lengths,
                scratch=block_buffers,
            )
            if self.select.gather_scores:
                self.select.run(
                    scores,
                    candidate_out_lengths,
                    None,
                    out,
                    values,
                    gather=candidate_out,
                    scratch=select_buffers,
                )
                return out, scores
        self.select.run(scores, lengths, active, out, values, scratch=select_buffers)
        return out, scores
