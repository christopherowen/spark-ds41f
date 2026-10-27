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

DeepSeek applies each route's weight before the down projection:
``moe_swiglu_cast`` reads ``layout_weights`` and fuses the weight, the SwiGLU
clamp and the MXFP8 cast of the down projection's input, with the arithmetic of
TileKernels' ``swiglu_forward_and_per_token_cast``, over the live tiles only.
``moe_combine_v41`` then sums each token's expert outputs, read from the
layout through ``route_pos``, in slot order in FP32, and optionally adds the
shared expert's output. With ``weighted=True`` it applies the weights itself
with explicit ``T.fma``. Neither kernel uses atomics on
results, so a token's output does not depend on the rest of the batch.
"""

import math

import tilelang
import tilelang.language as T
import torch

from vllm.models.deepseek_v4_1.tilelang import pdl, prepare

from .gemm import (
    SF_BLOCK,
    scale_words,
)
from .linear import MXFP8Rows, _calls, _launch
from .scale_codes import pack_expert_scale_codes, tile_byte_codes

# Routes handled by one CTA of the dispatch kernels.
_CHUNK = 512
# From this many rows a call's grouped GEMMs trigger their dependent's launch after
# their K loop, not at their start (see pdl).
LATE_TRIGGER_ROWS = 16


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
                pdl.chain()
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
                pdl.chain()
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
                pdl.chain()
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
            pdl.chain()
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
            pdl.chain()
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

    def combine(
        self, y, layout, out, topk_weights=None, shared=None, start=0, stop=None
    ):
        """Sum each token's expert outputs into ``out``: tokens ``start:stop`` (every
        token by default) into ``out``'s rows.

        Pass ``topk_weights`` only if the experts did not apply them. A BF16 ``out``
        of a plan with a wider output dtype takes ``shared``: the routes' FP32 sum
        plus the shared output, rounded once. A token's sum reads only its own
        routes, so its row has the same bits however the tokens are split.
        """
        route_pos = layout["route_pos"]
        rows = route_pos.shape[0]
        stop = rows if stop is None else stop
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
        kernel(
            y,
            route_pos[start:stop],
            weights[start:stop],
            y if shared is None else shared[start:stop],
            out,
        )
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
    coded_scales: bool = False,
    narrow_codes: bool = False,
    trigger_after_loop: bool = False,
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

    With ``coded_scales`` the weights' scales come from ``pack_scale_codes`` in
    ``tile_codes`` layout: ``SFB`` holds each tile's ``block_N`` 4-bit-delta codes
    per K block contiguously (one 16-byte ``cp.async`` per 8 rows) and ``SFBase``
    each row's base exponent. Every thread expands its row's code into the scale
    word the MMA reads, after the stage lands; the words, and so every output bit,
    are the uncoded ones, from half the scale bytes.

    With ``trigger_after_loop`` the kernel begins with ``pdl.wait()`` and triggers its
    dependent's launch after its K loop (``pdl.trigger()``) instead of at its start, so
    the dependent launches near its end (see ``pdl``).
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
    code_chunks = block_N * 2 // 16  # 16-byte copies of one stage's 4-bit codes
    # A narrow expert's code is one uint8 a row (four 2-bit deltas), half the bytes.
    code_chunks8 = block_N // 16
    assert not coded_scales or (block_N % 8 == 0 and code_chunks <= threads)
    assert not narrow_codes or (
        coded_scales and block_N % 16 == 0 and code_chunks8 <= threads
    )

    @T.macro
    def load(A, SFA, B, BT, SFB, A_shared, B_shared, SFA_shared, SFB_shared,
             a_row, sfa_row, loaded, expert, bx, tx, ko, s, SFB8, SFC8, Narrow):  # fmt: skip
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
        if coded_scales:
            # SFB_shared is this stage's 4-bit codes; the words are expanded later.
            # A narrow expert loads half as many bytes: one uint8 a row. The
            # expert is one per CTA, so the branch is uniform across the block.
            row = (expert * n_blocks + bx) * k_blocks + ko
            if narrow_codes:
                if Narrow[expert] != 0:
                    if tx < code_chunks8:
                        T.ptx_cp_async(
                            T.access_ptr(SFC8[s, tx * 16], "w", 16),
                            T.access_ptr(SFB8[row, tx * 16], "r", 16),
                            16,  # elements: sixteen 1-byte codes
                        )
                else:
                    if tx < code_chunks:
                        T.ptx_cp_async(
                            T.access_ptr(SFB_shared[s, tx * 8], "w", 8),
                            T.access_ptr(SFB[row, tx * 8], "r", 8),
                            8,  # elements: eight 2-byte codes, one 16-byte copy
                        )
            else:
                if tx < code_chunks:
                    T.ptx_cp_async(
                        T.access_ptr(SFB_shared[s, tx * 8], "w", 8),
                        T.access_ptr(SFB[row, tx * 8], "r", 8),
                        8,  # elements: eight 2-byte codes, one 16-byte copy
                    )
        else:
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
    def body(A, SFA, B, BT, SFB, SFBase, Order, TileExpert, C, SFB8=None, Narrow=None):
        with T.Kernel(T.ceildiv(N, block_N), tiles, threads=threads) as (bx, t):
            if trigger_after_loop:
                pdl.wait()
            else:
                pdl.chain()
            A_shared = T.alloc_shared((stages, block_M, block_K), T.float8_e4m3fn)
            B_shared = T.alloc_shared(
                (stages, block_N, block_K), T.float4_e2m1_unpacked
            )
            SFA_shared = T.alloc_shared((stages, block_M, 1), T.uint32)
            SFB_shared = T.alloc_shared((stages, block_N, 1), T.uint32)
            # The coded path's codes and per-row base words; the uncoded path
            # allocates neither.
            SFC_shared = (
                T.alloc_shared((stages, block_N), T.uint16) if coded_scales else None
            )
            # One uint8 a row for a narrow expert's 2-bit codes. Unused when
            # narrow_codes is off, so that kernel does not allocate it.
            SFC8 = (
                T.alloc_shared((stages, block_N), T.uint8) if narrow_codes else None
            )
            # Per-row base words. The thread that loads a code expands it, so
            # the base has to be shared rather than kept in that thread's
            # registers.
            base_shared = (
                T.alloc_shared((block_N,), T.uint32) if coded_scales else None
            )
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
            if coded_scales:
                for c in T.unroll(sfb_copies):
                    j = c * threads + tx
                    if j < block_N:
                        base_shared[j] = T.cast(
                            SFBase[expert, bx * block_N + j], T.uint32
                        ) * T.uint32(0x01010101)
                # The K loop's first sync publishes these before the expand.
            sf_dest = SFC_shared if coded_scales else SFB_shared

            T.clear(C_local)
            for p in T.unroll(stages - 1):
                if p < k_blocks:
                    if narrow_codes:
                        load(A, SFA, B, BT, SFB, A_shared, B_shared, SFA_shared,
                             sf_dest, a_row, sfa_row, loaded, expert, bx, tx, p,
                             p, SFB8, SFC8, Narrow)  # fmt: skip
                    else:
                        load(A, SFA, B, BT, SFB, A_shared, B_shared, SFA_shared,
                             sf_dest, a_row, sfa_row, loaded, expert, bx, tx, p,
                             p, SFB, SFB_shared, SFB)  # fmt: skip
                else:
                    T.ptx_commit_group()
            for ko in T.serial(k_blocks):
                # The stage loaded next was last read by iteration ko - 1.
                T.sync_threads()
                nxt = ko + stages - 1
                if nxt < k_blocks:
                    if narrow_codes:
                        load(A, SFA, B, BT, SFB, A_shared, B_shared, SFA_shared,
                             sf_dest, a_row, sfa_row, loaded, expert, bx, tx, nxt,
                             nxt % stages, SFB8, SFC8, Narrow)  # fmt: skip
                    else:
                        load(A, SFA, B, BT, SFB, A_shared, B_shared, SFA_shared,
                             sf_dest, a_row, sfa_row, loaded, expert, bx, tx, nxt,
                             nxt % stages, SFB, SFB_shared, SFB)  # fmt: skip
                else:
                    T.ptx_commit_group()
                T.ptx_wait_group(stages - 1)
                T.barrier_wait(loaded[ko % stages], (ko // stages) % 2)
                s = ko % stages
                if coded_scales:
                    # The thread that issued the cp.async expands those bytes.
                    # wait_group finishes only that thread's copy, so another
                    # thread cannot read them yet. The store stays inside the
                    # device branch: a name bound there does not survive it.
                    if narrow_codes:
                        if Narrow[expert] != 0:
                            # Uniform across the CTA (one expert). The tx test
                            # stays inside the unroll so the loop itself is not
                            # divergent.
                            for i in T.unroll(16):
                                if tx < code_chunks8:
                                    row = tx * 16 + i
                                    code8 = T.cast(SFC8[s, row], T.uint32)
                                    SFB_shared[s, row, 0] = (
                                        (code8 & T.uint32(0x3))
                                        | ((code8 >> 2) & T.uint32(0x3)) << 8
                                        | ((code8 >> 4) & T.uint32(0x3)) << 16
                                        | ((code8 >> 6) & T.uint32(0x3)) << 24
                                    ) + base_shared[row]
                        else:
                            for i in T.unroll(8):
                                if tx < code_chunks:
                                    row = tx * 8 + i
                                    code = T.cast(SFC_shared[s, row], T.uint32)
                                    SFB_shared[s, row, 0] = (
                                        (code & T.uint32(0xF))
                                        | ((code >> 4) & T.uint32(0xF)) << 8
                                        | ((code >> 8) & T.uint32(0xF)) << 16
                                        | ((code >> 12) & T.uint32(0xF)) << 24
                                    ) + base_shared[row]
                    else:
                        for i in T.unroll(8):
                            if tx < code_chunks:
                                row = tx * 8 + i
                                code = T.cast(SFC_shared[s, row], T.uint32)
                                SFB_shared[s, row, 0] = (
                                    (code & T.uint32(0xF))
                                    | ((code >> 4) & T.uint32(0xF)) << 8
                                    | ((code >> 8) & T.uint32(0xF)) << 16
                                    | ((code >> 12) & T.uint32(0xF)) << 24
                                ) + base_shared[row]
                T.sync_threads()
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
            if trigger_after_loop:
                pdl.trigger()
            # Only real rows are stored; padding rows of the layout stay untouched.
            for i, j in T.Parallel(block_M, block_N):
                if rows_shared[i] >= 0:
                    C[t * block_M + i, bx * block_N + j] = C_local[i, j]

    sfb_spec = (
        T.Tensor((num_experts * n_blocks * k_blocks, block_N), T.uint16)
        if coded_scales
        else T.Tensor((num_experts, N, sf_words), T.uint32)
    )
    base_spec = T.Tensor((num_experts, N), T.uint8)
    sfb8_spec = T.Tensor((num_experts * n_blocks * k_blocks, block_N), T.uint8)
    narrow_spec = T.Tensor((num_experts,), T.uint8)

    if tail and coded_scales and narrow_codes:

        @T.prim_func
        def moe_grouped_fp8_fp4_gemm(
            A: T.Tensor((a_rows, K), T.float8_e4m3fn),
            SFA: T.Tensor((a_rows, sf_words), T.uint32),
            B: T.Tensor(b_shape, T.float4_e2m1fn),
            BT: T.Tensor(bt_shape, T.uint8),
            SFB: sfb_spec,
            SFB8: sfb8_spec,
            Narrow: narrow_spec,
            SFBase: base_spec,
            Order: T.Tensor((tiles * block_M,), T.int32),
            TileExpert: T.Tensor((tiles,), T.int32),
            C: T.Tensor((tiles * block_M, N), out_dtype),
        ):
            body(A, SFA, B, BT, SFB, SFBase, Order, TileExpert, C, SFB8, Narrow)

    elif tail and coded_scales:

        @T.prim_func
        def moe_grouped_fp8_fp4_gemm(
            A: T.Tensor((a_rows, K), T.float8_e4m3fn),
            SFA: T.Tensor((a_rows, sf_words), T.uint32),
            B: T.Tensor(b_shape, T.float4_e2m1fn),
            BT: T.Tensor(bt_shape, T.uint8),
            SFB: sfb_spec,
            SFBase: base_spec,
            Order: T.Tensor((tiles * block_M,), T.int32),
            TileExpert: T.Tensor((tiles,), T.int32),
            C: T.Tensor((tiles * block_M, N), out_dtype),
        ):
            body(A, SFA, B, BT, SFB, SFBase, Order, TileExpert, C)

    elif coded_scales and narrow_codes:

        @T.prim_func
        def moe_grouped_fp8_fp4_gemm(
            A: T.Tensor((a_rows, K), T.float8_e4m3fn),
            SFA: T.Tensor((a_rows, sf_words), T.uint32),
            B: T.Tensor(b_shape, T.float4_e2m1fn),
            SFB: sfb_spec,
            SFB8: sfb8_spec,
            Narrow: narrow_spec,
            SFBase: base_spec,
            Order: T.Tensor((tiles * block_M,), T.int32),
            TileExpert: T.Tensor((tiles,), T.int32),
            C: T.Tensor((tiles * block_M, N), out_dtype),
        ):
            body(A, SFA, B, None, SFB, SFBase, Order, TileExpert, C, SFB8, Narrow)

    elif coded_scales:

        @T.prim_func
        def moe_grouped_fp8_fp4_gemm(
            A: T.Tensor((a_rows, K), T.float8_e4m3fn),
            SFA: T.Tensor((a_rows, sf_words), T.uint32),
            B: T.Tensor(b_shape, T.float4_e2m1fn),
            SFB: sfb_spec,
            SFBase: base_spec,
            Order: T.Tensor((tiles * block_M,), T.int32),
            TileExpert: T.Tensor((tiles,), T.int32),
            C: T.Tensor((tiles * block_M, N), out_dtype),
        ):
            body(A, SFA, B, None, SFB, SFBase, Order, TileExpert, C)

    elif tail:

        @T.prim_func
        def moe_grouped_fp8_fp4_gemm(
            A: T.Tensor((a_rows, K), T.float8_e4m3fn),
            SFA: T.Tensor((a_rows, sf_words), T.uint32),
            B: T.Tensor(b_shape, T.float4_e2m1fn),
            BT: T.Tensor(bt_shape, T.uint8),
            SFB: sfb_spec,
            Order: T.Tensor((tiles * block_M,), T.int32),
            TileExpert: T.Tensor((tiles,), T.int32),
            C: T.Tensor((tiles * block_M, N), out_dtype),
        ):
            body(A, SFA, B, BT, SFB, None, Order, TileExpert, C)

    else:

        @T.prim_func
        def moe_grouped_fp8_fp4_gemm(
            A: T.Tensor((a_rows, K), T.float8_e4m3fn),
            SFA: T.Tensor((a_rows, sf_words), T.uint32),
            B: T.Tensor(b_shape, T.float4_e2m1fn),
            SFB: sfb_spec,
            Order: T.Tensor((tiles * block_M,), T.int32),
            TileExpert: T.Tensor((tiles,), T.int32),
            C: T.Tensor((tiles * block_M, N), out_dtype),
        ):
            body(A, SFA, B, None, SFB, None, Order, TileExpert, C)

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


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    },
)
def moe_swiglu_cast(intermediate, block_M, limit, rows_per_cta=None):
    """The routed experts' SwiGLU, route weight and MXFP8 cast over the live tiles.

    ``H[L, 2I]`` (the gate/up GEMM's BF16 output, gate columns first) becomes
    ``Out[L, I]`` E4M3 with ``OutSF[L, *]`` UE8M0 bytes, one per 32 columns, for
    every layout row with ``Order[row] >= 0``; padding rows are not written, as the
    down GEMM's padding outputs are never read. A CTA takes ``rows_per_cta`` rows of
    one tile, loading all of them before computing any, and the CTAs of tiles past
    ``num_tiles`` (``TileExpert < 0``) stop there, so a decode step pays for its
    live rows only.

    The arithmetic is TileKernels' ``swiglu_forward_and_per_token_cast`` (route
    weights, the clamp at ``limit`` by compare and select, as spark-ds41f's TileKernels
    patch 0001 applies it, so a NaN passes as through ``torch.clamp``), element for
    element: the fast ``__expf``/``__fdividef`` product, its certificate and the
    precise fallback for a warp that fails it, TileKernels' vector width and
    ``get_sf_and_inv``. A warp takes the same ``32 * vec`` consecutive columns of one
    row as TileKernels' 32-thread groups do, so the certificate's warp vote covers
    the same elements and every output byte and scale equals TileKernels'. That
    kernel walks every expert's boundary in each of its 1,024-thread CTAs, one
    expert at a time, about 22 us a call at any decode size (spark-ds41f
    ``experiments/2026-10-14-moe-swiglu``).
    """
    from tile_kernels.quant.common import (
        get_best_vectorize_size,
        get_cast_output_config,
        get_sf_and_inv,
    )

    in_dtype = T.bfloat16
    out_config = get_cast_output_config("e4m3", (1, SF_BLOCK), False, True, True)
    warp = 32
    vec = max(
        math.gcd(intermediate, get_best_vectorize_size(in_dtype) * warp) // warp, 1
    )
    chunk = warp * vec
    assert intermediate % chunk == 0, "a warp's columns must tile the row"
    assert SF_BLOCK % vec == 0, "a scale block spans whole thread vectors"
    warps = intermediate // chunk  # one row's warps
    if rows_per_cta is None:
        # Fewer warps a row (TP3's 768 columns: 3) want fewer rows a CTA in flight
        # (GB10 bench, experiments/2026-10-18-r6w-swiglu).
        rows_per_cta = 4 if warps >= 4 else 2
    assert block_M % rows_per_cta == 0
    threads = warp * warps
    reduce_rounds = (SF_BLOCK // vec).bit_length() - 1
    sf_lanes = SF_BLOCK // vec
    sf_width = 4 * scale_words(intermediate)
    cert_e = 32  # TileKernels' certificate margin with route weights
    tiles = T.dynamic("tiles")
    rows = tiles * block_M

    @T.prim_func
    def moe_swiglu_cast_kernel(
        H: T.Tensor((rows, 2 * intermediate), in_dtype),
        Out: T.Tensor((rows, intermediate), T.float8_e4m3fn),
        OutSF: T.Tensor((rows, sf_width), T.uint8),
        Order: T.Tensor((rows,), T.int32),
        Weights: T.Tensor((rows,), T.float32),
        TileExpert: T.Tensor((tiles,), T.int32),
    ):
        with T.Kernel(block_M // rows_per_cta, tiles, threads=threads) as (slab, t):
            pdl.chain()
            tid = T.get_thread_binding()
            col = (tid // warp) * chunk + (tid % warp) * vec
            orders = T.alloc_local((rows_per_cta,), T.int32)
            weights = T.alloc_local((rows_per_cta,), T.float32)
            xl = T.alloc_local((rows_per_cta, vec), in_dtype)
            xr = T.alloc_local((rows_per_cta, vec), in_dtype)
            xl_f = T.alloc_local((vec,), T.float32)
            xr_f = T.alloc_local((vec,), T.float32)
            x = T.alloc_local((vec,), T.float32)
            stored = T.alloc_local((vec,), T.float8_e4m3fn)
            amax = T.alloc_local((2,), T.float32)
            cert_min_l = T.alloc_local((2,), T.float32)
            weight = T.alloc_var(T.float32)
            base = t * block_M + slab * rows_per_cta
            # Every row's loads issue before any row's arithmetic, so a cold call
            # waits once for the layout words and once for its rows. ``Order`` is
            # read in bounds for unused tiles too; their stale words are not used.
            for r in T.unroll(rows_per_cta):
                orders[r] = Order[base + r]
            if TileExpert[t] >= 0:
                for r in T.unroll(rows_per_cta):
                    if orders[r] >= 0:
                        weights[r] = Weights[base + r]
                        for j in T.vectorized(vec):
                            xl[r, j] = H[base + r, col + j]
                        for j in T.vectorized(vec):
                            xr[r, j] = H[base + r, intermediate + col + j]
                for r in T.unroll(rows_per_cta):
                    row = base + r
                    if orders[r] >= 0:
                        weight = weights[r]
                        for j in T.vectorized(vec):
                            xl_f[j] = T.float32(xl[r, j])
                            xr_f[j] = T.float32(xr[r, j])
                        for j in T.unroll(vec):
                            val_l = T.alloc_var(T.float32, init=xl_f[j])
                            val_r = T.alloc_var(T.float32, init=xr_f[j])
                            if limit is not None:
                                val_l = T.Select(val_l > limit, limit, val_l)
                                val_r = T.Select(val_r > limit, limit, val_r)
                                val_r = T.Select(val_r < -limit, -limit, val_r)
                            val = T.alloc_var(
                                T.float32,
                                init=T.call_extern(
                                    T.float32,
                                    "__fdividef",
                                    val_l,
                                    1 + T.call_extern(T.float32, "__expf", -val_l),
                                )
                                * val_r,
                            )
                            val = val * weight
                            x[j] = val
                        # TileKernels' certificate: the fast product lands in the
                        # precise product's E4M3/scale cell, or the warp recomputes.
                        cert_min_metric = T.alloc_var(T.uint32, init=0xFFFFFFFF)
                        cert_max_metric = T.alloc_var(T.uint32, init=0)
                        for j in T.vectorized(2):
                            cert_min_l[j] = T.float32(0)
                        for j in T.vectorized(vec):
                            cert_min_l[j % 2] = T.min(cert_min_l[j % 2], xl_f[j])
                        cert_min_l[0] = T.min(cert_min_l[0], cert_min_l[1])
                        for j in T.unroll(vec):
                            cert_t = T.reinterpret(x[j], T.uint32) & T.uint32(0x7FFFF)
                            cert_min_metric = T.min(cert_min_metric, cert_t)
                            cert_max_metric = T.max(cert_max_metric, cert_t)
                        if (
                            T.all_sync(
                                (cert_min_l[0] > -16.0)
                                and (cert_min_metric > cert_e)
                                and (cert_max_metric < 0x80000 - cert_e)
                            )
                            == 0
                        ):
                            for j in T.unroll(vec):
                                val_l = T.alloc_var(T.float32, init=xl_f[j])
                                val_r = T.alloc_var(T.float32, init=xr_f[j])
                                if limit is not None:
                                    val_l = T.Select(val_l > limit, limit, val_l)
                                    val_r = T.Select(val_r > limit, limit, val_r)
                                    val_r = T.Select(val_r < -limit, -limit, val_r)
                                val = T.alloc_var(
                                    T.float32,
                                    init=(val_l / (1 + T.exp(-val_l))) * val_r,
                                )
                                val = val * weight
                                x[j] = val
                        for j in T.vectorized(2):
                            amax[j] = 0
                        for j in T.vectorized(vec):
                            amax[j % 2] = T.max(amax[j % 2], T.abs(x[j]))
                        amax[0] = T.max(amax[0], amax[1])
                        for j in T.unroll(reduce_rounds):
                            amax[0] = T.max(amax[0], T.shfl_xor(amax[0], 1 << j))
                        sf, sf_inv = get_sf_and_inv(amax[0], out_config)
                        if (tid % warp) % sf_lanes == 0:
                            OutSF[row, col // SF_BLOCK] = sf
                        for j in T.vectorized(vec):
                            x[j] *= sf_inv
                        for j in T.vectorized(vec):
                            stored[j] = T.cast(x[j], T.float8_e4m3fn)
                        for j in T.vectorized(vec):
                            Out[row, col + j] = stored[j]

    return moe_swiglu_cast_kernel


def tile_codes(codes: torch.Tensor, block_N: int) -> torch.Tensor:
    """``pack_scale_codes``' ``[E, N, words]`` codes in the coded GEMM's layout:
    ``[E * N / block_N * words, block_N]``, each tile's codes for one K block
    contiguous, so a stage loads them in 16-byte copies."""
    E, N, words = codes.shape
    return (
        codes.view(torch.int16)
        .view(E, N // block_N, block_N, words)
        .permute(0, 1, 3, 2)
        .reshape(-1, block_N)
        .contiguous()
        .view(torch.uint16)
    )


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
    its UE8M0 bytes, become tile-major. The tokens arrive cast to MXFP8 by the
    kernel that produced them (``linear.MXFP8Rows``: the mHC entry). The fused
    SwiGLU (``moe_swiglu_cast``: DeepSeek's clamp, route weight before the down
    projection, MXFP8 cast) runs between the GEMMs. The plan holds no buffers;
    ``workspace_specs(rows)`` lists one call's scratch. Its owner calls
    ``declare_kernels()`` when preparing kernels; ``run`` launches the kernels it
    binds.

    Each grouped-GEMM tile holds ``block_M`` routes of one expert and reads that
    expert's weight slab again, so a call's tile height is chosen by its rows
    (``tile_height``): decode keeps ``block_M``, whose partial tiles cost least
    when an expert has a few routes; a prefill call, where the average expert has
    at least one full ``block_M`` tile, uses ``prefill_block_M`` rows a tile and
    reads the weights half as often a route (GB10, one layer's experts at TP3's
    768 columns a rank: 11% faster at 8,192 tokens, 19% at 16,384; spark-ds41f
    experiments/2026-10-26-tp3-budget-memory). A row's sum over K runs in the same
    order at either height, so its output has the same bits.
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
        prefill_block_M=64,
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
        self.block_M, self.prefill_block_M, self.block_N = (
            block_M,
            prefill_block_M,
            block_N,
        )
        self.num_stages = num_stages
        self.swiglu_limit = swiglu_limit
        # Routers emit int32 or int64 expert ids. The combine writes ``out_dtype``.
        self.out_dtype = out_dtype
        self.dispatch, self.prefill_dispatch = (
            {
                dtype: MoeDispatchV41Plan(
                    num_experts, topk, height, hidden, dtype, out_dtype
                )
                for dtype in (torch.int32, torch.int64)
            }
            for height in (block_M, prefill_block_M)
        )
        self.tk = _tile_kernels(block_M)

    def kernels(self) -> dict:
        """The grouped GEMMs and the fused SwiGLU at both tile heights, as the calls
        ``bind`` makes: ``gate_up``, ``down`` and ``swiglu`` at ``block_M``; at the same
        height ``late_gate_up`` and ``late_down``, which trigger after their K loop
        (calls of ``LATE_TRIGGER_ROWS`` rows or more); and with a ``prefill_`` prefix
        at ``prefill_block_M``, whose GEMMs trigger late too."""
        config = dict(
            block_N=self.block_N,
            num_stages=self.num_stages,
            coded_scales=True,
            narrow_codes=True,
        )
        num_experts, topk, hidden, inter = (
            self.num_experts,
            self.topk,
            self.hidden,
            self.intermediate,
        )
        kernels = {}
        for prefix, block_M, late in (
            ("", self.block_M, False),
            ("late_", self.block_M, True),
            ("prefill_", self.prefill_block_M, True),
        ):
            kernels[f"{prefix}gate_up"] = prepare.call(
                grouped_fp8_fp4_gemm,
                num_experts,
                2 * inter,
                hidden,
                topk,
                True,
                block_M,
                **config,
                trigger_after_loop=late,
            )
            kernels[f"{prefix}down"] = prepare.call(
                grouped_fp8_fp4_gemm,
                num_experts,
                hidden,
                inter,
                topk,
                False,
                block_M,
                **config,
                trigger_after_loop=late,
            )
            if prefix != "late_":  # the SwiGLU is the same at either trigger
                kernels[f"{prefix}swiglu"] = prepare.call(
                    moe_swiglu_cast, inter, block_M, self.swiglu_limit
                )
        return kernels

    def tile_height(self, rows: int) -> int:
        """The tile height of a call of ``rows`` tokens: ``prefill_block_M`` once the
        average expert has a full ``block_M`` tile of routes (``rows * topk >=
        block_M * num_experts``: 2,048 tokens for V4.1's 384 experts at top-6),
        else ``block_M``. Decode steps and their CUDA graphs stay below it."""
        if rows * self.topk >= self.block_M * self.num_experts:
            return self.prefill_block_M
        return self.block_M

    def _dispatch_for(self, rows: int, idx_dtype):
        """The dispatch plan of a call at its tile height."""
        if self.tile_height(rows) == self.block_M:
            return self.dispatch[idx_dtype]
        return self.prefill_dispatch[idx_dtype]

    def _kernels_for(self, rows: int):
        """The (gate/up, SwiGLU, down) kernels of a call at its tile height, its
        grouped GEMMs triggering late from ``LATE_TRIGGER_ROWS`` rows."""
        if self.tile_height(rows) != self.block_M:
            return self.prefill_gate_up, self.prefill_swiglu, self.prefill_down
        if rows >= LATE_TRIGGER_ROWS:
            return self.late_gate_up, self.swiglu, self.late_down
        return self.gate_up, self.swiglu, self.down

    def declare_kernels(self) -> None:
        """Declare the grouped GEMMs, the fused SwiGLU and each dispatch plan's
        kernels, stored on the plans once prepared."""

        dispatches = (*self.dispatch.values(), *self.prefill_dispatch.values())

        def calls() -> list[prepare.TileLangCall]:
            out = _calls(self.kernels())
            for dispatch in dispatches:
                out.extend(_calls(dispatch.kernels()))
            return out

        def bind() -> None:
            for name, kernel in _launch(self.kernels()).items():
                setattr(self, name, kernel)
            for dispatch in dispatches:
                dispatch.bind()

        prepare.declare(calls, bind=bind)

    def prepare_weights(self, w13, w13_scale, w2, w2_scale):
        """Per GEMM ``(tiles, tails, codes, codes_2bit, narrow, base)``.

        ``tails`` is None unless the GEMM's K is 64 past a multiple of 128 (the
        down projection of TP4's 576-wide expert rows). ``codes`` is
        ``pack_scale_codes`` in ``tile_codes`` layout, the bytes the coded GEMM
        loads today, for every expert. An expert whose every row has delta <= 3
        also has ``codes_2bit`` in that tile order as contiguous uint8 (four
        2-bit deltas a byte, low bits first); ``narrow`` is uint8 1 for that
        expert and 0 otherwise, and a wide expert's ``codes_2bit`` slice is
        zero. A narrow expert's GEMM loads ``codes_2bit``; a wide expert loads
        ``codes``. The 4-bit expand is the wide expert's path.
        """
        prepared = []
        for w, scale in ((w13, w13_scale), (w2, w2_scale)):
            packed = pack_expert_scale_codes(scale)
            prepared.append(
                (
                    *tile_expert_weights(w.view(torch.int8), self.block_N),
                    tile_codes(packed.codes, self.block_N),
                    tile_byte_codes(packed.codes_2bit, self.block_N),
                    packed.narrow,
                    packed.base,
                )
            )
        return tuple(prepared)

    def workspace_specs(self, rows, idx_dtype=torch.int64):
        """One call's scratch, laid out for the call's tile height (``tile_height``):
        ``prefill_block_M`` tiles pad each expert's routes to more rows."""
        dispatch = self._dispatch_for(rows, idx_dtype)
        layout_rows = dispatch.tiles(rows) * dispatch.block_M
        hidden, inter = self.hidden, self.intermediate
        return (
            *dispatch.workspace_specs(rows),
            ((layout_rows, 2 * inter), torch.bfloat16),  # gate/up output
            ((layout_rows, inter), torch.float8_e4m3fn),  # down input, MXFP8
            ((layout_rows, 4 * scale_words(inter)), torch.uint8),
            ((layout_rows, hidden), torch.bfloat16),  # down output
        )

    def run(
        self,
        x: MXFP8Rows,
        topk_ids,
        topk_weights,
        weights,
        out,
        workspace=None,
        shared=None,
    ):
        """``out[t] = sum_k w[t, k] * down_e(swiglu(gate_up_e(x[t]))) (+ shared[t])``.

        ``e`` is the token's k-th expert. ``x`` is the tokens' MXFP8 rows, as their
        producer wrote them. ``weights`` comes from ``prepare_weights``;
        ``workspace`` is a byte buffer of at least
        ``workspace_bytes(workspace_specs(rows))``. ``shared`` (BF16 rows) is added
        to the FP32 sum before the one rounding to ``out``'s dtype; a callable
        ``shared`` is called just before the combine launch (it may wait for
        another stream) and returns those rows.

        With ``out`` None the combine is not launched: the result is
        ``combine(start, stop, rows_out)``, which writes the output's rows
        ``start:stop`` into ``rows_out`` (in the dtype ``out`` would have) with one
        launch, the same bits however the rows are split. It reads ``workspace``
        and ``shared`` (a callable one is still called here), which must stay
        unchanged until its last call.
        """
        rows = topk_ids.shape[0]
        dispatch = self._dispatch_for(rows, topk_ids.dtype)
        gate_up_kernel, swiglu_kernel, down_kernel = self._kernels_for(rows)
        specs = self.workspace_specs(rows, topk_ids.dtype)
        if workspace is None:
            workspace = torch.empty(
                workspace_bytes(specs), dtype=torch.uint8, device=x.values.device
            )
        buffers = carve(workspace, specs)
        count = len(dispatch.workspace_specs(rows))
        h, hq, h_sf, y = buffers[count:]
        gate_up, down = weights
        layout = dispatch.dispatch(topk_ids, topk_weights, buffers[:count])
        order, tile_expert = layout["order"], layout["tile_expert"]
        # The gather reads the live rows only; padding rows past them are not read.
        x_words, h_words = x.scales.view(torch.uint32), h_sf.view(torch.uint32)
        gate_up_kernel(x.values, x_words, *_present(gate_up), order, tile_expert, h)
        swiglu_kernel(h, hq, h_sf, order, layout["layout_weights"], tile_expert)
        down_kernel(hq, h_words, *_present(down), order, tile_expert, y)
        if callable(shared):
            shared = shared()
        if out is not None:
            return dispatch.combine(y, layout, out, shared=shared)

        def combine(start, stop, rows_out):
            dispatch.combine(y, layout, rows_out, shared=shared, start=start, stop=stop)

        return combine


def _present(weights):
    """A GEMM's weight arguments: the tiles, the K tails when it has them, the scale
    codes and the row bases.

    ``prepare_weights`` returns ``(tiles, tails, codes, codes_2bit, narrow, base)``.
    The kernel loads ``codes_2bit`` for an expert whose ``narrow`` flag is 1 and
    ``codes`` otherwise.
    """
    return tuple(w for w in weights if w is not None)


def _tile_kernels(block_M):
    import tile_kernels

    # Expert segments in the dispatch layout start on block_M rows.
    tile_kernels.config.set_token_alignment(block_M)
    return tile_kernels
