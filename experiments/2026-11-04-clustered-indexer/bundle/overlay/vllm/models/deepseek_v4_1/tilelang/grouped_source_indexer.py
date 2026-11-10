# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact source handoff with a fresh snapshot shared by one known owner."""
import torch

from .clustered_indexer import CLUSTER, cluster_bounds_tensor
from .grouped_indexer import shared_extent, shared_tables
from .indexer import INDEX_DIM, page_bytes
from .source_indexer import SourceClusteredIndexerPlan


class GroupedSourceClusteredIndexerPlan(SourceClusteredIndexerPlan):
    """One explicitly owned cache table and 1..256 source queries.

    All visible keys must be written. No metadata persists between invocations;
    graph replay rebuilds maximum extent, per-row tables and bounds from live input.
    """

    def __init__(self, width, page_size):
        super().__init__(width, page_size)
        self.maximum = shared_extent(width)
        self.tables = shared_tables(width, page_size)
        if not self.native_fallback:
            self.search.bounds = cluster_bounds_tensor(width, shared_metadata=True)

    def run(self, q, scales, weights, pool, table, lengths, active):
        rows = q.shape[0]
        if not 0 < rows <= 256 or table.shape[0] != 1:
            raise ValueError('grouped source search requires 1..256 queries and one owned page table')
        if table.shape[1] * self.page_size < self.width:
            raise ValueError('owned table must cover the planned width')
        if rows == 1:
            # The owned table already has the native one-query shape; its
            # extent is the query's own length. No expansion/reduction is needed.
            return super().run(q, scales, weights, pool, table, lengths, active)
        expanded = torch.empty((rows, table.shape[1]), dtype=torch.int32, device=q.device)
        self.tables(table, lengths, active, expanded)
        if self.native_fallback:
            return super().run(q, scales, weights, pool, expanded, lengths, active)
        maximum = torch.empty((1,), dtype=torch.int32, device=q.device)
        lower = torch.empty((1, self.width // CLUSTER, INDEX_DIM), dtype=torch.float32, device=q.device)
        upper = torch.empty_like(lower)
        self.maximum(lengths, active, maximum)
        self.search.envelope(pool[:, :page_bytes(self.page_size)], table, maximum, active, lower, upper)
        result, evidence = super().run(q, scales, weights, pool, expanded, lengths, active,
                                      _metadata=(lower, upper))
        evidence['metadata_elements'] = lower.numel() + upper.numel()
        return result, evidence
