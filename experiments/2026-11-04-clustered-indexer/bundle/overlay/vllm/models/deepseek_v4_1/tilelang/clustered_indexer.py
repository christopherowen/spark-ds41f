# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental, lossless cluster pruning for the V4.1 MXFP4 indexer.

Clusters are contiguous 64-key groups in the checkpoint's index space. Each
stores a coordinate interval. A seed supplies a lower bound on the global kth
score; a cluster is omitted only when its score upper bound is strictly below
that threshold. All surviving keys use the existing MXFP4 scorer and top-k.

This snapshot plan is deliberately not registered with serving. Envelope rebuilds
are included in its cost; persistent-cache invalidation is not implemented.
"""

import tilelang
import tilelang.language as T
import torch

from .indexer import INDEX_DIM, INDEX_HEADS, INDEX_TOPK, IndexerV41Plan, page_bytes

CLUSTER = 64
# Bound normal FP32 arithmetic only. Other encodings disable pruning.
MIN_SCALE, MAX_SCALE = 96, 144


def _decode(byte, scale, d):
    code = (T.cast(byte, T.int32) >> ((d % 2) * 4)) & 15
    mag = code & 7
    value = T.if_then_else(
        mag < 2,
        T.cast(mag, T.float32) * 0.5,
        (1.0 + T.cast(mag & 1, T.float32) * 0.5)
        * T.cast(1 << T.max(mag // 2 - 1, 0), T.float32),
    )
    safe_scale = T.min(T.max(T.cast(scale, T.int32), MIN_SCALE), MAX_SCALE)
    value = value * T.reinterpret(safe_scale << 23, T.float32)
    return T.if_then_else((code & 8) != 0, -value, value)


@tilelang.jit
def cluster_envelopes(width: int, page_size: int):
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
            # One coordinate per lane. Streaming over positions keeps just two
            # endpoints live instead of two 64x128 reduction fragments.
            lmin = T.alloc_fragment([INDEX_DIM], T.float32)
            hmax = T.alloc_fragment([INDEX_DIM], T.float32)
            extent = T.min(T.max(Lengths[row], 0), T.max(T.min(Active[0], width), 0))
            for d in T.Parallel(INDEX_DIM):
                lmin[d] = T.infinity(T.float32)
                hmax[d] = -T.infinity(T.float32)
            for p in T.serial(CLUSTER):
                pos = cluster * CLUSTER + p
                if pos < extent:
                    for d in T.Parallel(INDEX_DIM):
                        physical = Table[row, pos // page_size]
                        offset = pos % page_size
                        scale = Pool[physical, page_size * 64 + offset * 4 + d // 32]
                        value = _decode(Pool[physical, offset * 64 + d // 2], scale, d)
                        valid = (scale >= MIN_SCALE) & (scale <= MAX_SCALE)
                        lmin[d] = T.min(lmin[d], T.if_then_else(
                            valid, value, -T.infinity(T.float32)))
                        hmax[d] = T.max(hmax[d], T.if_then_else(
                            valid, value, T.infinity(T.float32)))
            for d in T.Parallel(INDEX_DIM):
                Lower[row, cluster, d] = lmin[d]
                Upper[row, cluster, d] = hmax[d]

    return envelope



@tilelang.jit
def cluster_bounds(width: int):
    """Upper bounds in the scorer's BF16 domain, including rounding slack.

    For 128 terms, two FP32 dot reductions differ from their real sums by
    at most 2*gamma(127)*sum(abs(products)). The 1/256 allowance below also
    covers its own reductions and additions (see the experiment proof).
    """
    rows = T.dynamic("rows")
    clusters = width // CLUSTER

    @T.prim_func
    def bounds(
        Q: T.Tensor([rows, INDEX_HEADS, 64], T.uint8),
        Scales: T.Tensor([rows, INDEX_HEADS, 4], T.uint8),
        Weights: T.Tensor([rows, INDEX_HEADS], T.bfloat16),
        Lower: T.Tensor([rows, clusters, INDEX_DIM], T.float32),
        Upper: T.Tensor([rows, clusters, INDEX_DIM], T.float32),
        Lengths: T.Tensor([rows], T.int32),
        Active: T.Tensor([1], T.int32),
        Bounds: T.Tensor([rows, clusters], T.bfloat16),
        Counts: T.Tensor([rows], T.int32),
    ):
        with T.Kernel(clusters, rows, threads=128) as (cluster, row):
            products = T.alloc_fragment([INDEX_HEADS, INDEX_DIM], T.float32)
            absolute = T.alloc_fragment([INDEX_HEADS, INDEX_DIM], T.float32)
            invalid = T.alloc_fragment([INDEX_HEADS, INDEX_DIM], T.int32)
            dots = T.alloc_fragment([INDEX_HEADS], T.float32)
            norms = T.alloc_fragment([INDEX_HEADS], T.float32)
            bad = T.alloc_fragment([INDEX_HEADS], T.int32)
            contributions = T.alloc_shared([INDEX_HEADS], T.float32)
            total = T.alloc_var(T.float32)
            extent = T.min(T.max(Lengths[row], 0), T.max(T.min(Active[0], width), 0))
            if cluster == 0:
                if T.get_thread_binding() == 0:
                    Counts[row] = T.ceildiv(extent, CLUSTER)
            if cluster * CLUSTER < extent:
                for h, d in T.Parallel(INDEX_HEADS, INDEX_DIM):
                    scale = Scales[row, h, d // 32]
                    q = _decode(Q[row, h, d // 2], scale, d)
                    lo = Lower[row, cluster, d]
                    hi = Upper[row, cluster, d]
                    valid = (scale >= MIN_SCALE) & (scale <= MAX_SCALE)
                    valid = valid & (T.abs(lo) < T.infinity(T.float32))
                    valid = valid & (T.abs(hi) < T.infinity(T.float32))
                    endpoint = T.if_then_else(q >= 0, hi, lo)
                    products[h, d] = q * T.if_then_else(valid, endpoint, 0.0)
                    absolute[h, d] = T.abs(q) * T.if_then_else(
                        valid, T.max(T.abs(lo), T.abs(hi)), 0.0
                    )
                    invalid[h, d] = T.if_then_else(valid, 0, 1)
                T.reduce_sum(products, dots, dim=1)
                T.reduce_sum(absolute, norms, dim=1)
                T.reduce_max(invalid, bad, dim=1)
                for h in T.Parallel(INDEX_HEADS):
                    w = T.cast(Weights[row, h], T.float32)
                    valid = (bad[h] == 0) & (T.abs(w) <= 1048576.0)
                    # Tiny positive slack also covers exact zero/cancellation.
                    upper = T.max(dots[h] + norms[h] * (1.0 / 256.0) + 2.0**-50, 0.0)
                    dot_bf16 = T.cast(T.cast(upper, T.bfloat16), T.float32)
                    weighted = T.cast(
                        T.cast(T.fmul(dot_bf16, T.max(w, 0.0)), T.bfloat16),
                        T.float32,
                    )
                    contributions[h] = T.if_then_else(
                        valid, weighted, T.infinity(T.float32)
                    )
                T.sync_threads()
                if T.get_thread_binding() == 0:
                    total = 0.0
                    for h in T.serial(INDEX_HEADS):
                        total = total + contributions[h]
                    Bounds[row, cluster] = T.cast(total, T.bfloat16)
            else:
                if T.get_thread_binding() == 0:
                    Bounds[row, cluster] = -T.infinity(T.bfloat16)

    return bounds


@tilelang.jit
def cluster_bounds_tensor(width: int, block_C: int = 64):
    """The same conservative interval bound, using exact BF16 endpoint inputs.

    Decoded certified FP4 values fit BF16 exactly. Split q into positive and
    negative parts: q+ @ upper + q- @ lower is the real interval bound. The
    existing 1/256 allowance covers both 128-term MMA reductions and the norm.
    The final index scorer and selected-key arithmetic are never changed.
    """
    rows = T.dynamic('rows')
    clusters = width // CLUSTER

    @T.prim_func
    def bounds_tensor(
        Q: T.Tensor([rows, INDEX_HEADS, 64], T.uint8),
        Scales: T.Tensor([rows, INDEX_HEADS, 4], T.uint8),
        Weights: T.Tensor([rows, INDEX_HEADS], T.bfloat16),
        Lower: T.Tensor([rows, clusters, INDEX_DIM], T.float32),
        Upper: T.Tensor([rows, clusters, INDEX_DIM], T.float32),
        Lengths: T.Tensor([rows], T.int32),
        Active: T.Tensor([1], T.int32),
        Bounds: T.Tensor([rows, clusters], T.bfloat16),
        Counts: T.Tensor([rows], T.int32),
    ):
        with T.Kernel(T.ceildiv(clusters, block_C), rows, threads=128) as (tile, row):
            A = T.alloc_shared([block_C, INDEX_DIM], T.bfloat16)
            B = T.alloc_shared([INDEX_HEADS, INDEX_DIM], T.bfloat16)
            key_bad = T.alloc_shared([block_C], T.int32)
            query_bad = T.alloc_shared([INDEX_HEADS], T.int32)
            contrib = T.alloc_shared([block_C, INDEX_HEADS], T.float32)
            estimate = T.alloc_fragment([block_C, INDEX_HEADS], T.float32)
            norm = T.alloc_fragment([block_C, INDEX_HEADS], T.float32)
            total = T.alloc_fragment([block_C], T.float32)
            extent = T.min(T.max(Lengths[row], 0), T.max(T.min(Active[0], width), 0))
            if tile == 0:
                if T.get_thread_binding() == 0:
                    Counts[row] = T.ceildiv(extent, CLUSTER)
            for c in T.Parallel(block_C):
                key_bad[c] = 0
            for h in T.Parallel(INDEX_HEADS):
                s0 = T.cast(Scales[row, h, 0], T.int32)
                s1 = T.cast(Scales[row, h, 1], T.int32)
                s2 = T.cast(Scales[row, h, 2], T.int32)
                s3 = T.cast(Scales[row, h, 3], T.int32)
                query_bad[h] = T.cast(
                    (s0 < MIN_SCALE) | (s0 > MAX_SCALE)
                    | (s1 < MIN_SCALE) | (s1 > MAX_SCALE)
                    | (s2 < MIN_SCALE) | (s2 > MAX_SCALE)
                    | (s3 < MIN_SCALE) | (s3 > MAX_SCALE), T.int32)
            T.sync_threads()
            # Unsupported metadata is tagged explicitly, never sent as NaN/Inf
            # into MMA. Only the exceptional clusters issue shared atomics.
            for c, d in T.Parallel(block_C, INDEX_DIM):
                cluster = tile * block_C + c
                A[c, d] = T.cast(0, T.bfloat16)
                if cluster < clusters:
                    lo = Lower[row, cluster, d]
                    hi = Upper[row, cluster, d]
                    valid = (T.abs(lo) < T.infinity(T.float32)) & (T.abs(hi) < T.infinity(T.float32))
                    A[c, d] = T.cast(T.if_then_else(valid, hi, 0.0), T.bfloat16)
                    if not valid:
                        T.atomic_max(key_bad[c], 1)
            for h, d in T.Parallel(INDEX_HEADS, INDEX_DIM):
                q = _decode(Q[row, h, d // 2], Scales[row, h, d // 32], d)
                B[h, d] = T.cast(T.max(q, 0.0), T.bfloat16)
            T.sync_threads()
            T.gemm(A, B, estimate, transpose_B=True, clear_accum=True)
            T.sync_threads()
            for c, d in T.Parallel(block_C, INDEX_DIM):
                cluster = tile * block_C + c
                A[c, d] = T.cast(0, T.bfloat16)
                if cluster < clusters:
                    A[c, d] = T.cast(T.if_then_else(key_bad[c] == 0,
                                                    Lower[row, cluster, d], 0.0), T.bfloat16)
            for h, d in T.Parallel(INDEX_HEADS, INDEX_DIM):
                q = _decode(Q[row, h, d // 2], Scales[row, h, d // 32], d)
                B[h, d] = T.cast(T.min(q, 0.0), T.bfloat16)
            T.sync_threads()
            T.gemm(A, B, estimate, transpose_B=True, clear_accum=False)
            T.sync_threads()
            for c, d in T.Parallel(block_C, INDEX_DIM):
                cluster = tile * block_C + c
                A[c, d] = T.cast(0, T.bfloat16)
                if cluster < clusters:
                    lo = Lower[row, cluster, d]
                    hi = Upper[row, cluster, d]
                    A[c, d] = T.cast(T.if_then_else(key_bad[c] == 0,
                        T.max(T.abs(lo), T.abs(hi)), 0.0), T.bfloat16)
            for h, d in T.Parallel(INDEX_HEADS, INDEX_DIM):
                q = _decode(Q[row, h, d // 2], Scales[row, h, d // 32], d)
                B[h, d] = T.cast(T.abs(q), T.bfloat16)
            T.sync_threads()
            T.gemm(A, B, norm, transpose_B=True, clear_accum=True)
            for c, h in T.Parallel(block_C, INDEX_HEADS):
                w = T.cast(Weights[row, h], T.float32)
                valid = (key_bad[c] == 0) & (query_bad[h] == 0) & (T.abs(w) <= 1048576.0)
                valid = valid & (T.abs(estimate[c, h]) < T.infinity(T.float32))
                valid = valid & (T.abs(norm[c, h]) < T.infinity(T.float32))
                upper = T.max(estimate[c, h] + norm[c, h] * (1.0 / 256.0) + 2.0**-50, 0.0)
                dot_bf16 = T.cast(T.cast(upper, T.bfloat16), T.float32)
                weighted = T.cast(T.cast(T.fmul(dot_bf16, T.max(w, 0.0)), T.bfloat16), T.float32)
                contrib[c, h] = T.if_then_else(valid, weighted, T.infinity(T.float32))
            T.sync_threads()
            T.fill(total, 0)
            for h in T.serial(INDEX_HEADS):
                for c in T.Parallel(block_C):
                    total[c] += contrib[c, h]
            for c in T.Parallel(block_C):
                cluster = tile * block_C + c
                if cluster < clusters:
                    Bounds[row, cluster] = T.if_then_else(cluster * CLUSTER < extent,
                        T.cast(total[c], T.bfloat16), -T.infinity(T.bfloat16))

    return bounds_tensor


@tilelang.jit
def seed_candidates(width: int, seeds: int):
    rows = T.dynamic("rows")

    @T.prim_func
    def seed(
        Selected: T.Tensor([rows, seeds], T.int32),
        Lengths: T.Tensor([rows], T.int32),
        Active: T.Tensor([1], T.int32),
        Candidates: T.Tensor([rows, seeds * 8], T.int32),
        Counts: T.Tensor([rows], T.int32),
    ):
        with T.Kernel(rows, threads=128) as row:
            sizes = T.alloc_fragment([seeds], T.int32)
            count = T.alloc_fragment([1], T.int32)
            extent = T.min(T.max(Lengths[row], 0), T.max(T.min(Active[0], width), 0))
            for s, b in T.Parallel(seeds, 8):
                cluster = Selected[row, s]
                Candidates[row, s * 8 + b] = T.if_then_else(
                    (cluster >= 0) & ((cluster * 8 + b) * 8 < extent),
                    cluster * 8 + b,
                    -1,
                )
            for s in T.Parallel(seeds):
                cluster = Selected[row, s]
                sizes[s] = T.if_then_else(
                    cluster >= 0, T.min(CLUSTER, extent - cluster * CLUSTER), 0
                )
            T.reduce_sum(sizes, count, dim=0)
            if T.get_thread_binding() == 0:
                Counts[row] = count[0]

    return seed


@tilelang.jit
def surviving_candidates(width: int, topk: int):
    """Stable compaction in position order. Equality always survives the bound."""
    rows = T.dynamic("rows")
    clusters, threads = width // CLUSTER, 128

    @T.prim_func
    def compact(
        Bounds: T.Tensor([rows, clusters], T.bfloat16),
        Values: T.Tensor([rows, topk], T.bfloat16),
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
                    Bounds[row, T.min(cluster, clusters - 1)] >= minimum[0],
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


class ClusteredIndexerPlan:
    """Snapshot experiment: all preprocessing and allocations are explicit.

    No serving flags or attention backends select this class. It currently
    handles dense indexer domains only, not CSA's source/reindex/reuse state.
    """

    def __init__(self, width, page_size, *, topk=INDEX_TOPK, seeds=32, matrix_bounds=False):
        from .indexer import IndexerTopK

        if width <= 0 or width % CLUSTER or page_size % CLUSTER:
            raise ValueError("width and page size must be positive multiples of 64")
        self.width, self.page_size, self.topk = width, page_size, topk
        self.seeds = min(seeds, width // CLUSTER)
        if self.seeds <= 0 or self.seeds * CLUSTER < topk:
            raise ValueError("seed capacity must cover topk")
        self.envelope = cluster_envelopes(width, page_size)
        self.bounds = cluster_bounds_tensor(width) if matrix_bounds else cluster_bounds(width)
        self.coarse = IndexerTopK(width // CLUSTER, self.seeds)
        self.seed = seed_candidates(width, self.seeds)
        self.compact = surviving_candidates(width, topk)
        self.seed_plan = IndexerV41Plan(self.seeds * CLUSTER, page_size, topk=topk, candidates=True)
        self.final_plan = IndexerV41Plan(width, page_size, topk=topk, candidates=True)

    def run(self, q, scales, weights, pool, table, lengths, active):
        rows, clusters = q.shape[0], self.width // CLUSTER
        device = q.device

        def empty(shape, dtype):
            return torch.empty(shape, device=device, dtype=dtype)

        lower = empty((rows, clusters, INDEX_DIM), torch.float32)
        upper = torch.empty_like(lower)
        bounds = empty((rows, clusters), torch.bfloat16)
        counts = empty((rows,), torch.int32)
        coarse_ids = empty((rows, self.seeds), torch.int32)
        coarse_values = empty((rows, self.seeds), torch.bfloat16)
        seed_ids = empty((rows, self.seeds * 8), torch.int32)
        seed_counts = torch.empty_like(counts)
        candidates = empty((rows, self.width // 8), torch.int32)
        candidate_counts = torch.empty_like(counts)
        cap = torch.full((1,), clusters, device=device, dtype=torch.int32)
        seed_scratch = [empty(shape, dtype) for shape, dtype in self.seed_plan.scratch_specs(rows)]
        # Always rebuild: no stale metadata after append, overwrite, rollback or reuse.
        self.envelope(pool[:, :page_bytes(self.page_size)], table, lengths, active, lower, upper)
        self.bounds(q.view(torch.uint8), scales.view(torch.uint8).reshape(rows, INDEX_HEADS, 4),
                    weights, lower, upper, lengths, active, bounds, counts)
        self.coarse.run(bounds, counts, cap, coarse_ids, coarse_values)
        self.seed(coarse_ids, lengths, active, seed_ids, seed_counts)
        self.seed_plan.run(q, scales, weights, pool, table, lengths, active,
                           seed_ids, seed_counts, scratch=seed_scratch)
        self.compact(bounds, seed_scratch[1], lengths, active, candidates, candidate_counts)
        out, _ = self.final_plan.run(q, scales, weights, pool, table, lengths, active,
                                     candidates, candidate_counts)
        return out, {"bounds": bounds, "candidates": candidates,
                     "candidate_counts": candidate_counts, "seed_values": seed_scratch[1]}
