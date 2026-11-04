# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Frozen pre-stream envelope for endpoint correctness only; never timed.

Extracted from experiment commit 23e06e3d. The decoder and domain constants
are shared because this experiment changes only the reduction's execution.
"""
import tilelang
import tilelang.language as T
from vllm.models.deepseek_v4_1.tilelang.clustered_indexer import (
    _decode, CLUSTER, MIN_SCALE, MAX_SCALE, INDEX_DIM, page_bytes,
)

@tilelang.jit
def reference_envelopes(width: int, page_size: int):
    """Intervals over visible keys only; no read from an unwritten cache slot."""
    assert width > 0 and width % CLUSTER == 0
    assert page_size % CLUSTER == 0
    rows, pages, table_width, stride = T.dynamic("rows, pages, table_width, stride")
    clusters, nbytes = width // CLUSTER, page_bytes(page_size)

    @T.prim_func
    def envelope(
        Pool: T.StridedTensor([pages, nbytes], [stride, 1], T.uint8),
        Table: T.Tensor([rows, table_width], T.int32),
        Lengths: T.Tensor([rows], T.int32),
        Active: T.Tensor([1], T.int32),
        Lower: T.Tensor([rows, clusters, INDEX_DIM], T.float32),
        Upper: T.Tensor([rows, clusters, INDEX_DIM], T.float32),
    ):
        with T.Kernel(clusters, rows, threads=128) as (cluster, row):
            lo = T.alloc_fragment([CLUSTER, INDEX_DIM], T.float32)
            hi = T.alloc_fragment([CLUSTER, INDEX_DIM], T.float32)
            lmin = T.alloc_fragment([INDEX_DIM], T.float32)
            hmax = T.alloc_fragment([INDEX_DIM], T.float32)
            extent = T.min(T.max(Lengths[row], 0), T.max(T.min(Active[0], width), 0))
            for p, d in T.Parallel(CLUSTER, INDEX_DIM):
                pos = cluster * CLUSTER + p
                lo[p, d] = T.infinity(T.float32)
                hi[p, d] = -T.infinity(T.float32)
                if pos < extent:
                    physical = Table[row, pos // page_size]
                    offset = pos % page_size
                    scale = Pool[physical, page_size * 64 + offset * 4 + d // 32]
                    value = _decode(Pool[physical, offset * 64 + d // 2], scale, d)
                    valid = (scale >= MIN_SCALE) & (scale <= MAX_SCALE)
                    lo[p, d] = T.if_then_else(valid, value, -T.infinity(T.float32))
                    hi[p, d] = T.if_then_else(valid, value, T.infinity(T.float32))
            T.reduce_min(lo, lmin, dim=0)
            T.reduce_max(hi, hmax, dim=0)
            for d in T.Parallel(INDEX_DIM):
                Lower[row, cluster, d] = lmin[d]
                Upper[row, cluster, d] = hmax[d]

    return envelope



