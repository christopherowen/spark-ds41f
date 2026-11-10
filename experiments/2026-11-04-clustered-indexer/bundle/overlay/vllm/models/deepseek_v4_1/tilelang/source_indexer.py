# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact source-layer search: preserve winners and the reindex block handoff."""
import tilelang
import tilelang.language as T
import torch

from .clustered_indexer import CLUSTER, ClusteredIndexerPlan
from .indexer import CANDIDATE_BLOCK, CANDIDATE_BLOCKS, IndexerTopK, IndexerV41Plan, _key16, page_bytes


@tilelang.jit
def source_threshold(seed_blocks: int):
    rows = T.dynamic('rows')

    @T.prim_func
    def threshold(
        Blocks: T.Tensor([rows, CANDIDATE_BLOCKS], T.int32),
        Values: T.Tensor([rows, CANDIDATE_BLOCKS], T.bfloat16),
        Counts: T.Tensor([rows], T.int32),
        SeedIDs: T.Tensor([rows, seed_blocks], T.int32),
        CutoffKey: T.Tensor([rows], T.int32),
        CutoffPosition: T.Tensor([rows], T.int32),
        ThresholdValues: T.Tensor([rows, CANDIDATE_BLOCKS], T.bfloat16),
    ):
        with T.Kernel(rows, threads=128) as row:
            last = T.ceildiv(Counts[row], CANDIDATE_BLOCK) - 1
            pairs = T.alloc_fragment([CANDIDATE_BLOCKS], T.uint64)
            minimum = T.alloc_fragment([1], T.uint64)
            for k in T.Parallel(CANDIDATE_BLOCKS):
                # The native block selector forces its last block. Excluding it
                # gives a lower bound for the 2047 ordinary winners; padding
                # stays -inf and disables pruning when the seed is too small.
                forced = (Blocks[row, k] >= 0) & (Blocks[row, k] == last)
                ThresholdValues[row, k] = T.if_then_else(forced,
                    T.cast(3.3895313892515355e38, T.bfloat16), Values[row, k])

                block = Blocks[row, k]
                position = SeedIDs[row, T.max(block, 0)]
                finite = T.abs(T.cast(Values[row, k], T.float32)) < T.infinity(T.float32)
                pair = T.cast(_key16(Values[row, k]), T.uint64) * 4294967296 + (
                    T.cast(4294967295, T.uint64) - T.cast(T.max(position, 0), T.uint64))
                # One native key/original-position pair avoids keeping a second
                # fragment alive across dependent reductions and gathers.
                pairs[k] = T.if_then_else(forced,
                    T.cast(65535, T.uint64) * 4294967296 + 4294967295,
                    T.if_then_else(finite & (block >= 0) & (position >= 0), pair, T.cast(0, T.uint64)))
            T.reduce_min(pairs, minimum, dim=0)
            if T.get_thread_binding() == 0:
                key = T.cast(minimum[0] // 4294967296, T.int32)
                CutoffKey[row] = key
                CutoffPosition[row] = T.if_then_else(key == 0, 2147483647,
                    T.cast(4294967295 - minimum[0] % 4294967296, T.int32))

    return threshold


@tilelang.jit
def source_surviving_candidates(width: int, topk: int):
    """Stable compaction with the native block value/position cutoff."""
    rows = T.dynamic("rows")
    clusters, threads = width // CLUSTER, 128

    @T.prim_func
    def compact(
        Bounds: T.Tensor([rows, clusters], T.bfloat16),
        Values: T.Tensor([rows, topk], T.bfloat16),
        CutoffKey: T.Tensor([rows], T.int32),
        CutoffPosition: T.Tensor([rows], T.int32),
        Lengths: T.Tensor([rows], T.int32),
        Active: T.Tensor([1], T.int32),
        Candidates: T.Tensor([rows, width // 8], T.int32),
        Counts: T.Tensor([rows], T.int32),
    ):
        with T.Kernel(rows, threads=threads) as row:
            tx = T.get_thread_binding()
            values = T.alloc_fragment([topk], T.float32)
            minimum = T.alloc_fragment([1], T.float32)
            scan = T.alloc_shared([threads], T.int32)
            base = T.alloc_shared([1], T.int32)
            v = T.alloc_var(T.int32)
            extent = T.min(T.max(Lengths[row], 0), T.max(T.min(Active[0], width), 0))
            for k in T.Parallel(topk):
                value = T.cast(Values[row, k], T.float32)
                values[k] = T.if_then_else(
                    T.abs(value) < T.infinity(T.float32), value, -T.infinity(T.float32)
                )
            T.reduce_min(values, minimum, dim=0)
            for b in T.Parallel(width // 8):
                Candidates[row, b] = -1
            if tx == 0:
                base[0] = 0
            T.sync_threads()
            for chunk in T.serial(T.ceildiv(clusters, threads)):
                cluster = chunk * threads + tx
                live = T.min(CLUSTER, T.max(extent - cluster * CLUSTER, 0))
                keep = T.if_then_else(
                    cluster < clusters,
                    (Bounds[row, T.min(cluster, clusters - 1)] >= minimum[0]) & (
                        (_key16(Bounds[row, T.min(cluster, clusters - 1)]) != CutoffKey[row])
                        | (cluster * (CLUSTER // CANDIDATE_BLOCK) <= CutoffPosition[row])),
                    False,
                )
                size = T.if_then_else(keep, live, 0)
                scan[tx] = size
                T.sync_threads()
                for step in T.serial(7):
                    offset = 1 << step
                    v = scan[tx]
                    if tx >= offset:
                        v = v + scan[tx - offset]
                    T.sync_threads()
                    scan[tx] = v
                    T.sync_threads()
                if (cluster < clusters) & (size > 0):
                    for b in T.serial(8):
                        if b * 8 < size:
                            Candidates[row, (base[0] + scan[tx] - size) // 8 + b] = (
                                cluster * 8 + b
                            )
                T.sync_threads()
                if tx == 0:
                    base[0] = base[0] + scan[threads - 1]
                T.sync_threads()
            if tx == 0:
                Counts[row] = base[0]

    return compact


@tilelang.jit
def retain_last_cluster(width: int):
    rows = T.dynamic('rows')

    @T.prim_func
    def last(
        Bounds: T.Tensor([rows, width // CLUSTER], T.bfloat16),
        Lengths: T.Tensor([rows], T.int32),
        Active: T.Tensor([1], T.int32),
    ):
        with T.Kernel(rows, threads=128) as row:
            extent = T.min(T.max(Lengths[row], 0), T.max(T.min(Active[0], width), 0))
            if T.get_thread_binding() == 0:
                if extent > 0:
                    Bounds[row, (extent - 1) // CLUSTER] = T.infinity(T.bfloat16)

    return last


@tilelang.jit
def map_source_blocks(width: int):
    rows = T.dynamic('rows')

    @T.prim_func
    def remap(
        Selected: T.Tensor([rows, CANDIDATE_BLOCKS], T.int32),
        Candidates: T.Tensor([rows, width // CANDIDATE_BLOCK], T.int32),
        Original: T.Tensor([rows, CANDIDATE_BLOCKS], T.int32),
    ):
        with T.Kernel(T.ceildiv(rows * CANDIDATE_BLOCKS, 128), threads=128) as tile:
            for i in T.Parallel(128):
                slot = tile * 128 + i
                if slot < rows * CANDIDATE_BLOCKS:
                    row = slot // CANDIDATE_BLOCKS
                    k = slot % CANDIDATE_BLOCKS
                    selected = Selected[row, k]
                    Original[row, k] = T.if_then_else(selected >= 0,
                        Candidates[row, T.max(selected, 0)], -1)

    return remap


class SourceClusteredIndexerPlan:
    """Dense source rows only; no persistent state or serving registration.

    Preserve native top-512 keys and top-2048 eight-key blocks, including the
    mandatory last visible block. Small planned domains use the native source
    plan directly: a seed covering that whole domain cannot save scoring.
    """

    def __init__(self, width, page_size, *, matrix_bounds=True):
        self.width, self.page_size = width, page_size
        self.native_fallback = width <= 512 * CLUSTER
        if self.native_fallback:
            self.native = IndexerV41Plan(width, page_size, candidate_blocks=True)
            return
        self.search = ClusteredIndexerPlan(width, page_size, seeds=512, matrix_bounds=matrix_bounds)
        self.seed_blocks = IndexerTopK(self.search.seeds * CLUSTER, CANDIDATE_BLOCKS, CANDIDATE_BLOCK)
        self.select_blocks = IndexerTopK(width, CANDIDATE_BLOCKS, CANDIDATE_BLOCK)
        self.threshold = source_threshold(self.search.seeds * 8)
        self.last = retain_last_cluster(width)
        self.remap = map_source_blocks(width)
        self.compact = source_surviving_candidates(width, CANDIDATE_BLOCKS)

    def run(self, q, scales, weights, pool, table, lengths, active, *, _metadata=None):
        rows, device = q.shape[0], q.device
        def empty(shape, dtype):
            return torch.empty(shape, dtype=dtype, device=device)
        source_blocks = empty((rows, CANDIDATE_BLOCKS), torch.int32)
        source_lengths = empty((rows,), torch.int32)
        if self.native_fallback:
            out, _ = self.native.run(q, scales, weights, pool, table, lengths, active,
                                    candidate_out=source_blocks, candidate_out_lengths=source_lengths)
            return out, {'source_blocks': source_blocks, 'source_lengths': source_lengths,
                         'native_fallback': True}
        plan = self.search
        clusters = self.width // CLUSTER
        if _metadata is None:
            lower = empty((rows, clusters, 128), torch.float32)
            upper = torch.empty_like(lower)
        else:
            lower, upper = _metadata
        bounds = empty((rows, clusters), torch.bfloat16)
        counts = empty((rows,), torch.int32)
        coarse_ids = empty((rows, plan.seeds), torch.int32)
        coarse_values = empty((rows, plan.seeds), torch.bfloat16)
        seed_ids = empty((rows, plan.seeds * 8), torch.int32)
        seed_counts = empty((rows,), torch.int32)
        seed_scores = empty((rows, plan.seeds * CLUSTER), torch.bfloat16)
        seed_block_ids = empty((rows, CANDIDATE_BLOCKS), torch.int32)
        seed_block_values = empty((rows, CANDIDATE_BLOCKS), torch.bfloat16)
        seed_block_counts = empty((rows,), torch.int32)
        thresholds = torch.empty_like(seed_block_values)
        cutoff_keys = empty((rows,), torch.int32)
        cutoff_positions = empty((rows,), torch.int32)
        candidates = empty((rows, self.width // 8), torch.int32)
        candidate_counts = empty((rows,), torch.int32)
        if _metadata is None:
            plan.envelope(pool[:, :page_bytes(self.page_size)], table, lengths, active, lower, upper)
        plan.bounds(q.view(torch.uint8), scales.view(torch.uint8).reshape(rows, 32, 4),
                    weights, lower, upper, lengths, active, bounds, counts)
        plan.coarse.run(bounds, counts, active, coarse_ids, coarse_values)
        plan.seed(coarse_ids, lengths, active, seed_ids, seed_counts)
        nbytes = page_bytes(self.page_size)
        # Reuse the unchanged native candidate scorer; no unused seed top-512.
        plan.seed_plan.score(q.view(torch.int8), scales.view(torch.uint32).reshape(rows, -1),
            weights, pool.view(torch.int8)[:, :nbytes], pool.view(torch.uint32)[:, :nbytes//4],
            table, lengths, active, seed_ids, seed_counts, seed_scores)
        self.seed_blocks.run(seed_scores, seed_counts, active, seed_block_ids,
                             seed_block_values, counts=seed_block_counts)
        self.threshold(seed_block_ids, seed_block_values, seed_counts, seed_ids,
                       cutoff_keys, cutoff_positions, thresholds)
        self.last(bounds, lengths, active)
        self.compact(bounds, thresholds, cutoff_keys, cutoff_positions, lengths, active,
                     candidates, candidate_counts)
        out, compact_scores = plan.final_plan.run(q, scales, weights, pool, table, lengths, active,
                                                  candidates, candidate_counts)
        compact_blocks = empty((rows, CANDIDATE_BLOCKS), torch.int32)
        block_values = empty((rows, CANDIDATE_BLOCKS), torch.bfloat16)
        # Compaction retains whole 64-key clusters and the global last cluster.
        # Thus eight-key block boundaries, their total order, and the only
        # partial block remain unchanged under the logical-ID remap.
        self.select_blocks.run(compact_scores, candidate_counts, active,
                               compact_blocks, block_values, counts=source_lengths)
        self.remap(compact_blocks, candidates, source_blocks)
        return out, {'source_blocks': source_blocks, 'source_lengths': source_lengths,
                     'native_fallback': False, 'bounds': bounds, 'candidates': candidates,
                     'candidate_counts': candidate_counts, 'compact_scores': compact_scores}
