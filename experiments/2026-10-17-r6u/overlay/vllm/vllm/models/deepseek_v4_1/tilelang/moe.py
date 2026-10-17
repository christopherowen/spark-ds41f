# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1-Flash MoE dispatch and combine for SM120 / SM121.

V4.1 Flash routes every token to 6 of 384 experts (the drafter: 3 of 128).
A router, such as TileKernels' ``moe_topk_gate_forward`` or vLLM's
``dsv4_topk``, writes the experts as ``topk_idx[rows, topk]`` (``-1`` for
padding tokens) and their weights as ``topk_weights[rows, topk]``. Each
(row, slot) pair is a route, numbered ``row * topk + slot``.

``moe_dispatch_v41`` groups the routes by expert in the contiguous layout of
grouped GEMMs (as in DeepGEMM's ``m_grouped_*_contiguous``): each expert's
routes are consecutive, in route order, and padded to a multiple of
``block_M``. It writes:

- ``order[tiles * block_M]``: the route at each row of the layout, ``-1`` for
  padding;
- ``route_pos[rows, topk]``: the layout row of each route, ``-1`` for
  padding tokens;
- ``layout_weights[tiles * block_M]``: the route weight at each row, ``0``
  for padding;
- ``tile_expert[tiles]``: the expert of each ``block_M`` tile, ``-1`` past
  the last used tile, so a grouped GEMM with a fixed grid can stop early;
- ``expert_count[E]``, ``expert_offset[E]``, ``num_tiles[1]``, and
  ``psum[E] = expert_offset + expert_count``, the end of each expert's routes
  as TileKernels' ``psum_num_tokens_per_expert`` expects (with
  ``tile_kernels.config.set_token_alignment(block_M)``).

A route's rank within its expert is its position among that expert's routes
in route order, computed with warp match masks and per-warp counts. The
layout is therefore a pure function of ``topk_idx``.

DeepSeek applies each route's weight before the down projection: TileKernels'
``swiglu_forward_and_per_token_cast`` takes ``layout_weights`` and fuses the
weight, the SwiGLU clamp and the MXFP8 cast of the down projection's input.
``moe_combine_v41`` then sums each token's expert outputs, read from the
layout through ``route_pos``, in slot order in FP32, and optionally adds the
shared expert's output. With ``weighted=True`` it applies the weights itself
with explicit ``T.fma``. Neither kernel uses atomics on
results, so a token's output does not depend on the rest of the batch.
"""

import tilelang
import tilelang.language as T
import torch

from vllm.models.deepseek_v4_1.tilelang import prepare

from .gemm import (
    SF_BLOCK,
    pack_scale_words,
    scale_words,
)
from .linear import _calls, _launch, _warm_casts

# Routes handled by one CTA of the dispatch kernels.
_CHUNK = 512


def _dispatch_macros(num_experts, threads):
    warps = threads // 32
    steps = threads.bit_length() - 1
    assert threads >= num_experts and threads & (threads - 1) == 0

    @T.macro
    def rank_routes(TopkIdx, routes, base, topk, tx, warp_count):
        """For route base + tx: its expert, and its rank among the chunk's routes to it.

        Leaves the chunk's per-expert counts in warp_count[warps, :].
        """
        expert = T.alloc_var(T.int32)
        rank = T.alloc_var(T.int32)
        peers = T.alloc_var(T.uint32)
        run = T.alloc_var(T.int32)
        held = T.alloc_var(T.int32)
        lane = tx % 32
        warp = tx // 32
        for i in T.serial(T.ceildiv((warps + 1) * num_experts, threads)):
            if i * threads + tx < (warps + 1) * num_experts:
                warp_count[
                    (i * threads + tx) // num_experts, (i * threads + tx) % num_experts
                ] = 0
        T.sync_threads()
        r = T.min(base + tx, routes - 1)
        expert = T.if_then_else(
            base + tx < routes, T.cast(TopkIdx[r // topk, r % topk], T.int32), -1
        )
        peers = T.match_any_sync(expert)
        rank = T.cast(
            T.popcount(
                peers
                & (T.shift_left(T.uint32(1), T.cast(lane, T.uint32)) - T.uint32(1))
            ),
            T.int32,
        )
        if (expert >= 0) & (rank == 0):
            warp_count[warp, expert] = T.cast(T.popcount(peers), T.int32)
        T.sync_threads()
        # Exclusive per-warp bases of each expert, in warp order.
        if tx < num_experts:
            run = 0
            for w in T.serial(warps):
                held = warp_count[w, tx]
                warp_count[w, tx] = run
                run += held
            warp_count[warps, tx] = run
        T.sync_threads()
        if expert >= 0:
            rank += warp_count[warp, expert]
        return expert, rank

    @T.macro
    def layout(
        count,
        block_M,
        tx,
        scan,
        ExpertCount,
        ExpertOffset,
        Psum,
        Order,
        LayoutWeights,
        TileExpert,
        NumTiles,
        tiles,
    ):
        """From per-expert route counts: padded offsets, tile map and padding rows."""
        padded = T.alloc_var(T.int32)
        carry = T.alloc_var(T.int32)
        offset = T.alloc_var(T.int32)
        used_tiles = T.alloc_var(T.int32)
        padded = T.if_then_else(
            tx < num_experts, T.ceildiv(count, block_M) * block_M, 0
        )
        # scan may share storage with buffers the caller read last.
        T.sync_threads()
        scan[tx] = padded
        T.sync_threads()
        for s in T.serial(steps):
            carry = T.if_then_else(tx >= (1 << s), scan[T.max(tx - (1 << s), 0)], 0)
            T.sync_threads()
            scan[tx] += carry
            T.sync_threads()
        offset = scan[tx] - padded
        used_tiles = scan[threads - 1] // block_M
        if tx < num_experts:
            ExpertCount[tx] = count
            ExpertOffset[tx] = offset
            Psum[tx] = offset + count
            for i in T.serial(padded - count):
                Order[offset + count + i] = -1
                LayoutWeights[offset + count + i] = 0
            for t in T.serial(padded // block_M):
                TileExpert[offset // block_M + t] = tx
        if tx == 0:
            NumTiles[0] = used_tiles
        for i in T.serial(T.ceildiv(tiles, threads)):
            if (i * threads + tx < tiles) & (i * threads + tx >= used_tiles):
                TileExpert[i * threads + tx] = -1
        return offset

    return rank_routes, layout


@tilelang.jit(
    pass_configs={tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True}
)
def moe_dispatch_v41(
    num_experts: int,
    topk: int,
    block_M: int,
    part: str = "fused",
    idx_dtype=T.int64,
    threads: int = _CHUNK,
):
    """Build one dispatch kernel.

    ``part="fused"`` handles up to ``threads`` routes in one CTA. Larger batches
    run ``"count"`` (per-chunk ranks and counts), ``"layout"`` (offsets and the
    tile map) and ``"scatter"`` (positions).
    """
    rows, tiles, chunks = T.dynamic("rows, tiles, chunks")
    warps = threads // 32
    rank_routes, layout = _dispatch_macros(num_experts, threads)

    if part == "fused":

        @T.prim_func
        def moe_dispatch(
            TopkIdx: T.Tensor([rows, topk], idx_dtype),
            TopkWeights: T.Tensor([rows, topk], T.float32),
            Order: T.Tensor([tiles * block_M], T.int32),
            LayoutWeights: T.Tensor([tiles * block_M], T.float32),
            RoutePos: T.Tensor([rows, topk], T.int32),
            TileExpert: T.Tensor([tiles], T.int32),
            ExpertCount: T.Tensor([num_experts], T.int32),
            ExpertOffset: T.Tensor([num_experts], T.int32),
            Psum: T.Tensor([num_experts], T.int32),
            NumTiles: T.Tensor([1], T.int32),
        ):
            with T.Kernel(1, threads=threads):
                warp_count = T.alloc_shared([warps + 1, num_experts], T.int32)
                scan = T.alloc_shared([threads], T.int32)
                count = T.alloc_var(T.int32)
                tx = T.get_thread_binding()
                routes = rows * topk
                expert, rank = rank_routes(TopkIdx, routes, 0, topk, tx, warp_count)
                count = T.if_then_else(
                    tx < num_experts, warp_count[warps, T.min(tx, num_experts - 1)], 0
                )
                offset = layout(
                    count,
                    block_M,
                    tx,
                    scan,
                    ExpertCount,
                    ExpertOffset,
                    Psum,
                    Order,
                    LayoutWeights,
                    TileExpert,
                    NumTiles,
                    tiles,
                )
                T.sync_threads()
                if tx < num_experts:
                    scan[tx] = offset
                T.sync_threads()
                if tx < routes:
                    if expert >= 0:
                        Order[scan[expert] + rank] = tx
                        LayoutWeights[scan[expert] + rank] = TopkWeights[
                            tx // topk, tx % topk
                        ]
                        RoutePos[tx // topk, tx % topk] = scan[expert] + rank
                    else:
                        RoutePos[tx // topk, tx % topk] = -1

        return moe_dispatch

    if part == "count":

        @T.prim_func
        def moe_dispatch(
            TopkIdx: T.Tensor([rows, topk], idx_dtype),
            ChunkCount: T.Tensor([chunks, num_experts], T.int32),
            Rank: T.Tensor([rows * topk], T.int32),
        ):
            with T.Kernel(chunks, threads=threads) as c:
                warp_count = T.alloc_shared([warps + 1, num_experts], T.int32)
                tx = T.get_thread_binding()
                expert, rank = rank_routes(
                    TopkIdx, rows * topk, c * threads, topk, tx, warp_count
                )
                if tx < num_experts:
                    ChunkCount[c, tx] = warp_count[warps, tx]
                if c * threads + tx < rows * topk:
                    Rank[c * threads + tx] = rank

        return moe_dispatch

    if part == "layout":

        @T.prim_func
        def moe_dispatch(
            ChunkCount: T.Tensor([chunks, num_experts], T.int32),
            Order: T.Tensor([tiles * block_M], T.int32),
            LayoutWeights: T.Tensor([tiles * block_M], T.float32),
            TileExpert: T.Tensor([tiles], T.int32),
            ExpertCount: T.Tensor([num_experts], T.int32),
            ExpertOffset: T.Tensor([num_experts], T.int32),
            Psum: T.Tensor([num_experts], T.int32),
            NumTiles: T.Tensor([1], T.int32),
        ):
            with T.Kernel(1, threads=threads):
                scan = T.alloc_shared([threads], T.int32)
                count = T.alloc_var(T.int32)
                tx = T.get_thread_binding()
                # Exclusive per-expert bases of each chunk, in chunk order.
                held = T.alloc_var(T.int32)
                count = 0
                if tx < num_experts:
                    for c in T.serial(chunks):
                        held = ChunkCount[c, tx]
                        ChunkCount[c, tx] = count
                        count += held
                layout(
                    count,
                    block_M,
                    tx,
                    scan,
                    ExpertCount,
                    ExpertOffset,
                    Psum,
                    Order,
                    LayoutWeights,
                    TileExpert,
                    NumTiles,
                    tiles,
                )

        return moe_dispatch

    assert part == "scatter", part

    @T.prim_func
    def moe_dispatch(
        TopkIdx: T.Tensor([rows, topk], idx_dtype),
        TopkWeights: T.Tensor([rows, topk], T.float32),
        ChunkCount: T.Tensor([chunks, num_experts], T.int32),
        Rank: T.Tensor([rows * topk], T.int32),
        ExpertOffset: T.Tensor([num_experts], T.int32),
        Order: T.Tensor([tiles * block_M], T.int32),
        LayoutWeights: T.Tensor([tiles * block_M], T.float32),
        RoutePos: T.Tensor([rows, topk], T.int32),
    ):
        with T.Kernel(chunks, threads=threads) as c:
            tx = T.get_thread_binding()
            expert = T.alloc_var(T.int32)
            r = c * threads + tx
            if r < rows * topk:
                expert = T.cast(TopkIdx[r // topk, r % topk], T.int32)
                if expert >= 0:
                    pos = ExpertOffset[expert] + ChunkCount[c, expert] + Rank[r]
                    Order[pos] = r
                    LayoutWeights[pos] = TopkWeights[r // topk, r % topk]
                    RoutePos[r // topk, r % topk] = pos
                else:
                    RoutePos[r // topk, r % topk] = -1

    return moe_dispatch


@tilelang.jit
def moe_combine_v41(
    hidden: int,
    topk: int,
    shared: bool = False,
    weighted: bool = False,
    block_D: int = 1024,
    threads: int = 128,
    out_dtype: T.dtype = T.bfloat16,
):
    """``Out[t] = sum_k [Weights[t, k] *] Y[RoutePos[t, k]] (+ Shared[t])``.

    FP32 in slot order; ``out_dtype`` out (FP32 keeps the routes' sum unrounded
    until the rank's partial, shared expert included, rounds once).
    """
    rows, layout_rows = T.dynamic("rows, layout_rows")
    shared_rows = T.dynamic("shared_rows")  # rows with a shared input; any otherwise
    while hidden % block_D:
        block_D //= 2
    assert block_D >= 8, f"hidden={hidden} must be a multiple of 8"

    @T.prim_func
    def moe_combine(
        Y: T.Tensor([layout_rows, hidden], T.bfloat16),
        RoutePos: T.Tensor([rows, topk], T.int32),
        Weights: T.Tensor([rows, topk], T.float32),
        Shared: T.Tensor([shared_rows, hidden], T.bfloat16),
        Out: T.Tensor([rows, hidden], out_dtype),
    ):
        with T.Kernel(hidden // block_D, rows, threads=threads) as (bd, row):
            acc = T.alloc_fragment([block_D], T.float32)
            T.clear(acc)
            if shared:
                # Requested before the routes' loads, added after their sum: the
                # same arithmetic, with the shared row's latency hidden.
                sv = T.alloc_fragment([block_D], T.bfloat16)
                for d in T.Parallel(block_D):
                    sv[d] = Shared[row, bd * block_D + d]
            for k in T.serial(topk):
                pos = RoutePos[row, k]
                if pos >= 0:
                    for d in T.Parallel(block_D):
                        if weighted:
                            acc[d] = T.fma(
                                Weights[row, k],
                                T.cast(Y[pos, bd * block_D + d], T.float32),
                                acc[d],
                            )
                        else:
                            acc[d] = acc[d] + T.cast(
                                Y[pos, bd * block_D + d], T.float32
                            )
            for d in T.Parallel(block_D):
                if shared:
                    Out[row, bd * block_D + d] = T.cast(
                        acc[d] + T.cast(sv[d], T.float32), out_dtype
                    )
                else:
                    Out[row, bd * block_D + d] = T.cast(acc[d], out_dtype)

    return moe_combine


def layout_tiles(rows: int, topk: int, block_M: int, num_experts: int) -> int:
    """Tiles of the contiguous layout.

    Each used tile holds a route, plus at most one partial tile per expert.
    """
    routes = rows * topk
    return min(routes, tilelang.cdiv(routes, block_M) + num_experts)


class MoeDispatchV41Plan:
    """Dispatch and combine kernels for one routing shape.

    The plan holds no buffers: ``workspace_specs(rows)`` lists the layout a
    call needs, which a caller can borrow from a shared workspace.
    ``dispatch`` and ``combine`` only launch kernels, so they can be captured
    in a CUDA graph. ``kernels()`` lists the plan's kernels as calls; ``bind()``
    stores them once they are prepared (its owner declares them).
    """

    _LAYOUT = (
        "order",
        "layout_weights",
        "route_pos",
        "tile_expert",
        "expert_count",
        "expert_offset",
        "psum",
        "num_tiles",
    )

    def __init__(
        self,
        num_experts,
        topk,
        block_M,
        hidden,
        idx_dtype=torch.int64,
        out_dtype=torch.bfloat16,
    ):
        self.num_experts, self.topk, self.block_M, self.hidden = (
            num_experts,
            topk,
            block_M,
            hidden,
        )
        self.idx_dtype = idx_dtype
        self.out_dtype = out_dtype

    def kernels(self) -> dict:
        """Every kernel of the plan, as the calls ``bind`` makes."""
        num_experts, topk, block_M, hidden = (
            self.num_experts,
            self.topk,
            self.block_M,
            self.hidden,
        )
        tl_idx = {torch.int64: T.int64, torch.int32: T.int32}[self.idx_dtype]
        tl_out = {torch.bfloat16: T.bfloat16, torch.float32: T.float32}[self.out_dtype]
        kernels = {
            part: prepare.call(
                moe_dispatch_v41, num_experts, topk, block_M, part, tl_idx
            )
            for part in ("fused", "count", "layout", "scatter")
        }
        kernels["combine_kernels"] = {
            (s, w): prepare.call(moe_combine_v41, hidden, topk, s, w, out_dtype=tl_out)
            for s in (False, True)
            for w in (False, True)
        }
        # A wider output (the FP32 partial) can instead take the shared expert's
        # output, add it in FP32 and round once to BF16 (``combine(shared=...)``).
        kernels["fold_kernels"] = (
            {}
            if self.out_dtype == torch.bfloat16
            else {
                w: prepare.call(
                    moe_combine_v41, hidden, topk, True, w, out_dtype=T.bfloat16
                )
                for w in (False, True)
            }
        )
        return kernels

    def bind(self) -> None:
        """Store the plan's kernels (``fused``, ``count``, ``layout``, ``scatter``,
        ``combine_kernels``, ``fold_kernels``), once they are prepared."""
        for name, kernel in _launch(self.kernels()).items():
            setattr(self, name, kernel)

    def tiles(self, rows):
        return layout_tiles(rows, self.topk, self.block_M, self.num_experts)

    def workspace_specs(self, rows):
        routes, tiles, experts = rows * self.topk, self.tiles(rows), self.num_experts
        specs = [
            ((tiles * self.block_M,), torch.int32),  # order
            ((tiles * self.block_M,), torch.float32),  # layout_weights
            ((rows, self.topk), torch.int32),  # route_pos
            ((tiles,), torch.int32),  # tile_expert
            ((experts,), torch.int32),  # expert_count
            ((experts,), torch.int32),  # expert_offset
            ((experts,), torch.int32),  # psum
            ((1,), torch.int32),  # num_tiles
        ]
        if routes > _CHUNK:
            specs += [
                ((tilelang.cdiv(routes, _CHUNK), experts), torch.int32),
                ((routes,), torch.int32),
            ]
        return tuple(specs)

    def dispatch(self, topk_idx, topk_weights, workspace):
        """Lay the routes out by expert in ``workspace`` (from ``workspace_specs``).

        Returns the named layout.
        """
        rows = topk_idx.shape[0]
        layout = dict(zip(self._LAYOUT, workspace))
        if rows * self.topk <= _CHUNK:
            self.fused(topk_idx, topk_weights, *(layout[name] for name in self._LAYOUT))
        else:
            chunk_count, rank = workspace[len(self._LAYOUT) :]
            self.count(topk_idx, chunk_count, rank)
            self.layout(
                chunk_count,
                layout["order"],
                layout["layout_weights"],
                layout["tile_expert"],
                layout["expert_count"],
                layout["expert_offset"],
                layout["psum"],
                layout["num_tiles"],
            )
            self.scatter(
                topk_idx,
                topk_weights,
                chunk_count,
                rank,
                layout["expert_offset"],
                layout["order"],
                layout["layout_weights"],
                layout["route_pos"],
            )
        return layout

    def combine(self, y, layout, out, topk_weights=None, shared=None):
        """Sum each token's expert outputs into ``out``.

        Pass ``topk_weights`` only if the experts did not apply them. A BF16 ``out``
        of a plan with a wider output dtype takes ``shared``: the routes' FP32 sum
        plus the shared output, rounded once.
        """
        rows = layout["route_pos"].shape[0]
        weights = (
            layout["layout_weights"][: rows * self.topk].view(rows, self.topk)
            if topk_weights is None
            else topk_weights
        )
        if out.dtype != self.out_dtype:
            assert out.dtype == torch.bfloat16 and shared is not None, (
                "only a shared input folds a wider combine into BF16"
            )
            kernel = self.fold_kernels[topk_weights is not None]
        else:
            kernel = self.combine_kernels[
                (shared is not None, topk_weights is not None)
            ]
        # Without a shared input the kernel never reads it: any BF16 rows do.
        kernel(y, layout["route_pos"], weights, y if shared is None else shared, out)
        return out


# TMA unpacks E2M1 for the MMA (CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B) only in
# boxes of exactly 128 K elements, from tensors whose K is a multiple of 128.
FP4_TMA_K = 128


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def grouped_fp8_fp4_gemm(
    num_experts: int,
    N: int,
    K: int,
    topk: int = 6,
    gather: bool = True,
    block_M: int = 32,
    block_N: int = 128,
    num_stages: int = 3,
    threads: int = 128,
    out_dtype: T.dtype = T.bfloat16,
):
    """``C[r] = (A[row(r)] * SFA) @ (B[e] * SFB[e])^T``, ``e`` the expert of row ``r``.

    ``A``: ``[rows, K]`` E4M3 with packed UE8M0 words ``SFA[rows, ceil(K / 128)]``.
    The weights are E2M1 with ``SFB[experts, N, ceil(K / 128)]``, in the
    tile-major layout of ``tile_expert_weights``: ``B`` holds the 128-wide K
    tiles, each a contiguous ``[block_N, 128]`` slab, and when ``K % 128 == 64``
    ``BT`` holds each row block's 64-wide K tail as packed bytes.

    A ``num_stages``-deep pipeline loads the next K blocks while the current one
    multiplies: the 128-wide weight tiles by TMA, which unpacks E2M1 for the MMA;
    the tail by ``cp.async``, eight packed bytes into the start of each
    sixteen-byte group of the same unpacked layout; the gathered activation rows
    and the scale words by ``cp.async``. The tail block multiplies only its 64
    columns, so no operand or scale is padded (TP4's 576-wide expert rows are
    four tiles and a tail).
    """
    block_K = FP4_TMA_K
    tail = K % block_K
    k_full = K // block_K
    assert tail in (0, 64) and k_full >= 1, "K must be a multiple of 64, at least 128"
    assert N % block_N == 0, "N must be a multiple of block_N"
    k_blocks = k_full + (tail > 0)
    tokens, tiles = T.dynamic("tokens, tiles")
    a_rows = tokens if gather else tiles * block_M
    accum_dtype = T.float32
    sf_words = scale_words(K)
    row_segments = block_K // 16  # 16-byte cp.async copies of one A row
    a_segments = block_M * row_segments
    a_copies = tilelang.cdiv(a_segments, threads)
    sfa_copies = tilelang.cdiv(block_M, threads)
    sfb_copies = tilelang.cdiv(block_N, threads)
    tail_groups = tail // 16  # sixteen-element groups of one tail row
    bt_copies = tilelang.cdiv(block_N * tail_groups, threads)
    stages = num_stages
    assert stages >= 2
    n_blocks = N // block_N
    b_shape = (num_experts * n_blocks * k_full, block_N, block_K)
    bt_shape = (num_experts * n_blocks, block_N, tail // 2)

    @T.macro
    def load(A, SFA, B, BT, SFB, A_shared, B_shared, SFA_shared, SFB_shared,
             a_row, sfa_row, loaded, expert, bx, tx, ko, s):  # fmt: skip
        """Stage s <- K block ko: weights, gathered activation rows, scale words."""
        if ko < k_full:
            tile = (expert * n_blocks + bx) * k_full + ko
            T.tma_copy(
                B[tile : tile + 1, 0:block_N, 0:block_K],
                B_shared[s : s + 1, :, :],
                barrier=loaded[s],
            )
        # The tail exists at trace time (BT); its block is chosen on device.
        if BT is not None:  # noqa: SIM102
            if ko == k_full:
                for c in T.unroll(bt_copies):
                    e = c * threads + tx
                    if e < block_N * tail_groups:
                        n = e // tail_groups
                        g = e % tail_groups
                        T.ptx_cp_async(
                            T.access_ptr(B_shared[s, n, g * 16], "w", 8),
                            T.access_ptr(BT[expert * n_blocks + bx, n, g * 8], "r", 8),
                            8,
                        )
        # Padding rows (row -1) are zero-filled without reading memory; the
        # tail block copies only its own columns.
        columns = T.if_then_else(ko < k_full, row_segments, tail_groups)
        for c in T.unroll(a_copies):
            e = c * threads + tx
            if e < a_segments and e % row_segments < columns:
                T.ptx_cp_async(
                    T.access_ptr(
                        A_shared[s, e // row_segments, e % row_segments * 16],
                        "w",
                        16,
                    ),
                    T.access_ptr(
                        A[T.max(a_row[c], 0), ko * block_K + e % row_segments * 16],
                        "r",
                        16,
                    ),
                    16,
                    a_row[c] >= 0,
                )
        # One scale word per row covers the block's four K32 atoms.
        for c in T.unroll(sfa_copies):
            i = c * threads + tx
            if i < block_M:
                T.ptx_cp_async(
                    T.access_ptr(SFA_shared[s, i, 0], "w", 1),
                    T.access_ptr(SFA[T.max(sfa_row[c], 0), ko], "r", 1),
                    1,
                    sfa_row[c] >= 0,
                )
        for c in T.unroll(sfb_copies):
            j = c * threads + tx
            if j < block_N:
                T.ptx_cp_async(
                    T.access_ptr(SFB_shared[s, j, 0], "w", 1),
                    T.access_ptr(SFB[expert, bx * block_N + j, ko], "r", 1),
                    1,
                )
        T.ptx_commit_group()
        T.barrier_arrive(loaded[s])

    @T.macro
    def body(A, SFA, B, BT, SFB, Order, TileExpert, C):
        with T.Kernel(T.ceildiv(N, block_N), tiles, threads=threads) as (bx, t):
            A_shared = T.alloc_shared((stages, block_M, block_K), T.float8_e4m3fn)
            B_shared = T.alloc_shared(
                (stages, block_N, block_K), T.float4_e2m1_unpacked
            )
            SFA_shared = T.alloc_shared((stages, block_M, 1), T.uint32)
            SFB_shared = T.alloc_shared((stages, block_N, 1), T.uint32)
            rows_shared = T.alloc_shared((block_M,), T.int32)
            a_row = T.alloc_local((a_copies,), T.int32)
            sfa_row = T.alloc_local((sfa_copies,), T.int32)
            loaded = T.alloc_barrier([threads] * stages)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            tx = T.get_thread_binding()

            expert = TileExpert[t]
            if expert < 0:
                T.thread_return()
            for i in T.Parallel(block_M):
                route = Order[t * block_M + i]
                rows_shared[i] = T.if_then_else(
                    route >= 0, route // topk if gather else t * block_M + i, -1
                )
            T.sync_threads()
            # Each thread copies the same rows at every K block.
            for c in T.unroll(a_copies):
                a_row[c] = rows_shared[
                    T.min((c * threads + tx) // row_segments, block_M - 1)
                ]
            for c in T.unroll(sfa_copies):
                sfa_row[c] = rows_shared[T.min(c * threads + tx, block_M - 1)]

            T.clear(C_local)
            for p in T.unroll(stages - 1):
                if p < k_blocks:
                    load(A, SFA, B, BT, SFB, A_shared, B_shared, SFA_shared,
                         SFB_shared, a_row, sfa_row, loaded, expert, bx, tx, p,
                         p)  # fmt: skip
                else:
                    T.ptx_commit_group()
            for ko in T.serial(k_blocks):
                # The stage loaded next was last read by iteration ko - 1.
                T.sync_threads()
                nxt = ko + stages - 1
                if nxt < k_blocks:
                    load(A, SFA, B, BT, SFB, A_shared, B_shared, SFA_shared,
                         SFB_shared, a_row, sfa_row, loaded, expert, bx, tx, nxt,
                         nxt % stages)  # fmt: skip
                else:
                    T.ptx_commit_group()
                T.ptx_wait_group(stages - 1)
                T.barrier_wait(loaded[ko % stages], (ko // stages) % 2)
                T.sync_threads()
                s = ko % stages
                if ko < k_full:
                    T.mma_gemm_blockscaled(
                        A_shared[s, :, :],
                        B_shared[s, :, :],
                        C_local,
                        SFA_shared[s, :, :],
                        SFB_shared[s, :, :],
                        transpose_B=True,
                        k_start=0,
                        sf_a_granularity_k=SF_BLOCK,
                        sf_b_granularity_k=SF_BLOCK,
                    )
                if BT is not None:  # noqa: SIM102
                    if ko == k_full:
                        T.mma_gemm_blockscaled(
                            A_shared[s, :, 0:tail],
                            B_shared[s, :, 0:tail],
                            C_local,
                            SFA_shared[s, :, :],
                            SFB_shared[s, :, :],
                            transpose_B=True,
                            k_start=0,
                            sf_a_granularity_k=SF_BLOCK,
                            sf_b_granularity_k=SF_BLOCK,
                        )
            # Only real rows are stored; padding rows of the layout stay untouched.
            for i, j in T.Parallel(block_M, block_N):
                if rows_shared[i] >= 0:
                    C[t * block_M + i, bx * block_N + j] = C_local[i, j]

    if tail:

        @T.prim_func
        def moe_grouped_fp8_fp4_gemm(
            A: T.Tensor((a_rows, K), T.float8_e4m3fn),
            SFA: T.Tensor((a_rows, sf_words), T.uint32),
            B: T.Tensor(b_shape, T.float4_e2m1fn),
            BT: T.Tensor(bt_shape, T.uint8),
            SFB: T.Tensor((num_experts, N, sf_words), T.uint32),
            Order: T.Tensor((tiles * block_M,), T.int32),
            TileExpert: T.Tensor((tiles,), T.int32),
            C: T.Tensor((tiles * block_M, N), out_dtype),
        ):
            body(A, SFA, B, BT, SFB, Order, TileExpert, C)

    else:

        @T.prim_func
        def moe_grouped_fp8_fp4_gemm(
            A: T.Tensor((a_rows, K), T.float8_e4m3fn),
            SFA: T.Tensor((a_rows, sf_words), T.uint32),
            B: T.Tensor(b_shape, T.float4_e2m1fn),
            SFB: T.Tensor((num_experts, N, sf_words), T.uint32),
            Order: T.Tensor((tiles * block_M,), T.int32),
            TileExpert: T.Tensor((tiles,), T.int32),
            C: T.Tensor((tiles * block_M, N), out_dtype),
        ):
            body(A, SFA, B, None, SFB, Order, TileExpert, C)

    return moe_grouped_fp8_fp4_gemm


def tile_expert_weights(b_packed, block_N):
    """Tile-major copy of ``[E, N, K / 2]`` packed E2M1 expert weights.

    Returns ``(tiles, tails)``. Tile ``(e, n, k)`` (``block_N`` rows by 128
    columns) becomes one contiguous ``[block_N, 64]``-byte slab of ``tiles``,
    at index ``(e * N / block_N + n) * (K // 128) + k``. When ``K % 128 == 64``,
    ``tails[e * N / block_N + n]`` holds that row block's last 64 columns as
    ``[block_N, 32]`` bytes; otherwise ``tails`` is None.
    """
    E, N, half_k = b_packed.shape
    k_full, tail = divmod(2 * half_k, FP4_TMA_K)
    b = b_packed.view(E, N // block_N, block_N, half_k)
    full = b[..., : k_full * FP4_TMA_K // 2]
    tiles = (
        full.reshape(E, N // block_N, block_N, k_full, FP4_TMA_K // 2)
        .permute(0, 1, 3, 2, 4)
        .reshape(-1, block_N, FP4_TMA_K // 2)
        .contiguous()
    )
    if not tail:
        return tiles, None
    tails = b[..., k_full * FP4_TMA_K // 2 :].reshape(-1, block_N, tail // 2)
    return tiles, tails.contiguous().view(torch.uint8)


def _spans(specs):
    """Byte offset and size of each spec laid out back to back, 256-byte aligned."""
    spans, offset = [], 0
    for shape, dtype in specs:
        nbytes = dtype.itemsize
        for dim in shape:
            nbytes *= dim
        spans.append((offset, nbytes))
        offset += (nbytes + 255) // 256 * 256
    return spans, offset


def carve(scratch, specs):
    """Views of ``specs`` in ``scratch``, laid out as ``workspace_bytes``."""
    spans, _ = _spans(specs)
    return [
        scratch[offset : offset + nbytes].view(dtype).view(shape)
        for (offset, nbytes), (shape, dtype) in zip(spans, specs)
    ]


def workspace_bytes(specs) -> int:
    return _spans(specs)[1]


class MoeExpertsV41Plan:
    """Routed MXFP4 experts, MXFP8 activations: dispatch, two grouped GEMMs, combine.

    Expert weights come from ``prepare_weights``: ``w13`` (``[E, 2I, H / 2]``
    packed E2M1, gate rows first) and ``w2`` (``[E, H, I / 2]``), each with
    its UE8M0 bytes, become tile-major. The TileKernels MXFP8 cast and fused
    SwiGLU (DeepSeek's clamp, route weight before the down projection) run
    between the GEMMs. The plan holds no buffers; ``workspace_specs(rows)``
    lists one call's scratch. Its owner calls ``declare_kernels()`` when
    preparing kernels; ``run`` launches the kernels it binds.
    """

    def __init__(
        self,
        num_experts,
        topk,
        hidden,
        intermediate,
        *,
        swiglu_limit=None,
        block_M=32,
        block_N=128,
        num_stages=3,
        out_dtype=torch.bfloat16,
    ):
        self.num_experts, self.topk, self.hidden, self.intermediate = (
            num_experts,
            topk,
            hidden,
            intermediate,
        )
        self.block_M, self.block_N = block_M, block_N
        self.num_stages = num_stages
        self.swiglu_limit = swiglu_limit
        # Routers emit int32 or int64 expert ids. The combine writes ``out_dtype``.
        self.out_dtype = out_dtype
        self.dispatch = {
            dtype: MoeDispatchV41Plan(
                num_experts, topk, block_M, hidden, dtype, out_dtype
            )
            for dtype in (torch.int32, torch.int64)
        }
        self.tk = _tile_kernels(block_M)

    def kernels(self) -> dict:
        """The grouped GEMMs and TileKernels' fused SwiGLU, as the calls ``bind``
        makes."""
        config = dict(block_N=self.block_N, num_stages=self.num_stages)
        num_experts, topk, hidden, inter = (
            self.num_experts,
            self.topk,
            self.hidden,
            self.intermediate,
        )
        block_M = self.block_M
        return {
            "gate_up": prepare.call(
                grouped_fp8_fp4_gemm,
                num_experts,
                2 * inter,
                hidden,
                topk,
                True,
                block_M,
                **config,
            ),
            "down": prepare.call(
                grouped_fp8_fp4_gemm,
                num_experts,
                hidden,
                inter,
                topk,
                False,
                block_M,
                **config,
            ),
            "swiglu": _swiglu_call(self.tk, inter, num_experts, self.swiglu_limit),
        }

    def declare_kernels(self) -> None:
        """Declare the grouped GEMMs, the fused SwiGLU and each dispatch plan's
        kernels, stored on the plans once prepared, and warm TileKernels' MXFP8
        cast of the tokens for each batch bucket a step can launch."""

        def calls() -> list[prepare.TileLangCall]:
            out = _calls(self.kernels())
            for dispatch in self.dispatch.values():
                out.extend(_calls(dispatch.kernels()))
            return out

        def bind() -> None:
            for name, kernel in _launch(self.kernels()).items():
                setattr(self, name, kernel)
            for dispatch in self.dispatch.values():
                dispatch.bind()

        prepare.declare(calls, bind=bind, warm=lambda: _warm_casts(self.hidden))

    def prepare_weights(self, w13, w13_scale, w2, w2_scale):
        """Per GEMM ``(tiles, tails, scale words)`` from E2M1 weights and UE8M0.

        ``tails`` is None unless the GEMM's K is 64 past a multiple of 128 (the
        down projection of TP4's 576-wide expert rows).
        """
        return tuple(
            (*tile_expert_weights(w.view(torch.int8), self.block_N),
             pack_scale_words(scale))
            for w, scale in ((w13, w13_scale), (w2, w2_scale))
        )  # fmt: skip

    def workspace_specs(self, rows, idx_dtype=torch.int64):
        dispatch = self.dispatch[idx_dtype]
        layout_rows = dispatch.tiles(rows) * self.block_M
        hidden, inter = self.hidden, self.intermediate
        return (
            *dispatch.workspace_specs(rows),
            ((rows, hidden), torch.float8_e4m3fn),  # tokens, MXFP8
            ((rows, 4 * scale_words(hidden)), torch.uint8),
            ((layout_rows, 2 * inter), torch.bfloat16),  # gate/up output
            ((layout_rows, inter), torch.float8_e4m3fn),  # down input, MXFP8
            ((layout_rows, 4 * scale_words(inter)), torch.uint8),
            ((layout_rows, hidden), torch.bfloat16),  # down output
        )

    def run(self, x, topk_ids, topk_weights, weights, out, workspace=None, shared=None):
        """``out[t] = sum_k w[t, k] * down_e(swiglu(gate_up_e(x[t]))) (+ shared[t])``.

        ``e`` is the token's k-th expert. ``weights`` comes from
        ``prepare_weights``; ``workspace`` is a byte buffer of at least
        ``workspace_bytes(workspace_specs(rows))``. ``shared`` (BF16 rows) is added
        to the FP32 sum before the one rounding to ``out``'s dtype; a callable
        ``shared`` is called just before the combine launch (it may wait for
        another stream) and returns those rows.
        """
        rows = x.shape[0]
        dispatch = self.dispatch[topk_ids.dtype]
        specs = self.workspace_specs(rows, topk_ids.dtype)
        if workspace is None:
            workspace = torch.empty(
                workspace_bytes(specs), dtype=torch.uint8, device=x.device
            )
        buffers = carve(workspace, specs)
        count = len(dispatch.workspace_specs(rows))
        xq, x_sf, h, hq, h_sf, y = buffers[count:]
        gate_up, down = weights
        layout = dispatch.dispatch(topk_ids, topk_weights, buffers[:count])
        self.tk.quant.per_token_cast(
            x, "e4m3", SF_BLOCK, round_sf=True, use_packed_ue8m0=True, out=(xq, x_sf)
        )
        order, tile_expert = layout["order"], layout["tile_expert"]
        x_words, h_words = x_sf.view(torch.uint32), h_sf.view(torch.uint32)
        self.gate_up(xq, x_words, *_present(gate_up), order, tile_expert, h)
        self.swiglu(h, hq, h_sf, layout["psum"], layout["layout_weights"], 1.0, None)
        self.down(hq, h_words, *_present(down), order, tile_expert, y)
        if callable(shared):
            shared = shared()
        return dispatch.combine(y, layout, out, shared=shared)


def _present(weights):
    """A GEMM's weight arguments: the tiles, the K tails when it has them, the words."""
    return tuple(w for w in weights if w is not None)


def _tile_kernels(block_M):
    import tile_kernels

    # Expert segments in the dispatch layout start on block_M rows.
    tile_kernels.config.set_token_alignment(block_M)
    return tile_kernels


def _swiglu_call(tk, intermediate, num_experts, limit) -> prepare.TileLangCall:
    """TileKernels' fused SwiGLU, route weight and MXFP8 cast, into caller buffers.

    ``swiglu_forward_and_per_token_cast`` allocates its outputs; its kernel
    writes into ours. The kernel's ``@tilelang.jit`` factory is called here with
    every key, so it is declared like the plan's own kernels.
    """
    from tile_kernels.quant.common import get_cast_output_config
    from tile_kernels.quant.swiglu_forward_cuda import (
        get_swiglu_forward_and_per_token_cast_kernel_cuda,
    )

    return prepare.call(
        get_swiglu_forward_and_per_token_cast_kernel_cuda,
        intermediate,
        num_experts,
        tk.config.get_token_alignment(),
        True,  # route weights
        False,  # routed scaling factor
        limit,
        False,  # clamped count
        in_dtype=T.dtype(torch.bfloat16),
        out_config=get_cast_output_config("e4m3", (1, SF_BLOCK), False, True, True),
        num_sms=tk.config.get_num_sms(),
        use_pdl=tk.config.get_pdl(),
    )
