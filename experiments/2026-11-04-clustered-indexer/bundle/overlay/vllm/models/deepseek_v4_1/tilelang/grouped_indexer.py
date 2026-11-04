# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact snapshot metadata shared by queries of one explicitly owned cache.

Every call rebuilds metadata through the maximum visible query extent. A bound
on that superset also bounds each query's visible keys. There is no persistent
metadata, inferred cache identity, or reuse across calls. Serving must group rows
by known cache ownership before this API is eligible.
"""
import tilelang
import tilelang.language as T
import torch

from .clustered_indexer import CLUSTER, ClusteredIndexerPlan, cluster_bounds_tensor
from .indexer import INDEX_DIM, INDEX_TOPK, page_bytes


@tilelang.jit
def shared_extent(width: int, row_cap: int = 256):
    rows = T.dynamic('rows')

    @T.prim_func
    def extent(
        Lengths: T.Tensor([rows], T.int32),
        Active: T.Tensor([1], T.int32),
        Maximum: T.Tensor([1], T.int32),
    ):
        with T.Kernel(1, threads=128):
            values = T.alloc_fragment([row_cap], T.int32)
            maximum = T.alloc_fragment([1], T.int32)
            for r in T.Parallel(row_cap):
                values[r] = T.if_then_else(r < rows, T.max(Lengths[r], 0), 0)
            T.reduce_max(values, maximum, dim=0)
            if T.get_thread_binding() == 0:
                Maximum[0] = T.min(maximum[0], T.max(T.min(Active[0], width), 0))

    return extent


@tilelang.jit
def shared_tables(width: int, page_size: int):
    rows, table_width = T.dynamic('rows, table_width')

    @T.prim_func
    def tables(
        Table: T.Tensor([1, table_width], T.int32),
        Lengths: T.Tensor([rows], T.int32),
        Active: T.Tensor([1], T.int32),
        Expanded: T.Tensor([rows, table_width], T.int32),
    ):
        with T.Kernel(T.ceildiv(rows * table_width, 128), threads=128) as block:
            for i in T.Parallel(128):
                slot = block * 128 + i
                if slot < rows * table_width:
                    row = slot // table_width
                    page = slot % table_width
                    extent = T.min(T.max(Lengths[row], 0), T.max(T.min(Active[0], width), 0))
                    Expanded[row, page] = T.if_then_else(page * page_size < extent,
                                                        Table[0, page], 0)

    return tables


class GroupedClusteredIndexerPlan(ClusteredIndexerPlan):
    """One cache owner, 1..256 query rows; all visible keys must be written.

    ``table`` is the owner's single page-table row, covering the maximum visible
    extent. This explicit contract does not accept arbitrary mixed-request rows.
    Each graph replay recomputes the maximum, table and envelopes from live input.
    """

    def __init__(self, width, page_size, *, topk=INDEX_TOPK, seeds=32):
        super().__init__(width, page_size, topk=topk, seeds=seeds, matrix_bounds=True)
        self.bounds = cluster_bounds_tensor(width, shared_metadata=True)
        self.maximum = shared_extent(width)
        self.tables = shared_tables(width, page_size)

    def run(self, q, scales, weights, pool, table, lengths, active):
        rows = q.shape[0]
        if not 0 < rows <= 256 or table.shape[0] != 1:
            raise ValueError('grouped search requires 1..256 queries and one owned page table')
        if table.shape[1] * self.page_size < self.width:
            raise ValueError('owned table must cover the planned width')
        maximum = torch.empty((1,), dtype=torch.int32, device=q.device)
        expanded = torch.empty((rows, table.shape[1]), dtype=torch.int32, device=q.device)
        lower = torch.empty((1, self.width // CLUSTER, INDEX_DIM), dtype=torch.float32, device=q.device)
        upper = torch.empty_like(lower)
        self.maximum(lengths, active, maximum)
        self.tables(table, lengths, active, expanded)
        self.envelope(pool[:, :page_bytes(self.page_size)], table, maximum, active, lower, upper)
        result, evidence = super().run(q, scales, weights, pool, expanded, lengths, active,
                                      _metadata=(lower, upper))
        evidence['metadata_elements'] = lower.numel() + upper.numel()
        return result, evidence
