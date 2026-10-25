# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A reduce-scatter with the one-shot all-reduce's arithmetic.

sparknet's one-shot all-reduce adds the ranks' BF16 partials in FP32 in rank order
(source 0, then 1, 2, 3) and rounds once, so a row has the same bits whatever the
message size, the row's position or the rank. NCCL's reduce-scatter, which prefill
sequence parallelism uses from 205 rows, adds a chunk around the ring starting
after its owner and rounds to BF16 after every hop: a row's bits change with the
step's row count. ``reduce_scatter_rank_order`` exchanges the chunks unreduced
(NCCL send/recv on the same communicator, the same bytes) and adds them as the
one-shot all-reduce does, so a row reduces to the same bits on every path.

At four ranks only the neighbours (rank +-1, the ring4 cable order sparknet and
NCCL use) exchange: each half of the opposite rank's chunk is relayed through one
neighbour, so the four links carry equal bytes. Fewer ranks exchange directly.
"""

# No `from __future__ import annotations`: TileLang reads the prim_func
# annotations eagerly, and the symbolic shape must resolve.
import math
from dataclasses import dataclass

import tilelang
import tilelang.language as T
import torch

from vllm.models.deepseek_v4_1.tilelang import prepare

_VEC = 8  # BF16 per thread: one 16-byte access


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def rank_order_sum_kernel(world: int, threads: int = 256):
    """``out = bf16(((s0 + s1) + s2) + s3)`` in FP32, element by element; the
    first source is taken as is (not added to zero), so signed zeros survive."""
    assert 2 <= world <= 4
    m = T.dynamic("m")  # groups of _VEC elements

    @T.macro
    def load(src, xs, total, row, first):
        for i in T.vectorized(_VEC):
            xs[i] = src[row, i]
        for i in T.unroll(_VEC):
            if first:
                total[i] = T.cast(xs[i], T.float32)
            else:
                total[i] = total[i] + T.cast(xs[i], T.float32)

    @T.macro
    def store(out, xs, total, row):
        for i in T.unroll(_VEC):
            xs[i] = T.cast(total[i], T.bfloat16)
        for i in T.vectorized(_VEC):
            out[row, i] = xs[i]

    if world == 2:

        @T.prim_func
        def collectives_rank_order_sum(
            s0: T.Tensor((m, _VEC), T.bfloat16),
            s1: T.Tensor((m, _VEC), T.bfloat16),
            out: T.Tensor((m, _VEC), T.bfloat16),
        ):
            with T.Kernel(T.ceildiv(m, threads), threads=threads) as bx:
                row = bx * threads + T.get_thread_binding()
                xs = T.alloc_local((_VEC,), T.bfloat16)
                total = T.alloc_local((_VEC,), T.float32)
                if row < m:
                    load(s0, xs, total, row, True)
                    load(s1, xs, total, row, False)
                    store(out, xs, total, row)

    elif world == 3:

        @T.prim_func
        def collectives_rank_order_sum(
            s0: T.Tensor((m, _VEC), T.bfloat16),
            s1: T.Tensor((m, _VEC), T.bfloat16),
            s2: T.Tensor((m, _VEC), T.bfloat16),
            out: T.Tensor((m, _VEC), T.bfloat16),
        ):
            with T.Kernel(T.ceildiv(m, threads), threads=threads) as bx:
                row = bx * threads + T.get_thread_binding()
                xs = T.alloc_local((_VEC,), T.bfloat16)
                total = T.alloc_local((_VEC,), T.float32)
                if row < m:
                    load(s0, xs, total, row, True)
                    load(s1, xs, total, row, False)
                    load(s2, xs, total, row, False)
                    store(out, xs, total, row)

    else:

        @T.prim_func
        def collectives_rank_order_sum(
            s0: T.Tensor((m, _VEC), T.bfloat16),
            s1: T.Tensor((m, _VEC), T.bfloat16),
            s2: T.Tensor((m, _VEC), T.bfloat16),
            s3: T.Tensor((m, _VEC), T.bfloat16),
            out: T.Tensor((m, _VEC), T.bfloat16),
        ):
            with T.Kernel(T.ceildiv(m, threads), threads=threads) as bx:
                row = bx * threads + T.get_thread_binding()
                xs = T.alloc_local((_VEC,), T.bfloat16)
                total = T.alloc_local((_VEC,), T.float32)
                if row < m:
                    load(s0, xs, total, row, True)
                    load(s1, xs, total, row, False)
                    load(s2, xs, total, row, False)
                    load(s3, xs, total, row, False)
                    store(out, xs, total, row)

    return collectives_rank_order_sum


def rank_order_sum_calls() -> list[prepare.TileLangCall]:
    """The rank-order sum the TP group's collectives launch, as ``rank_order_sum``
    calls it: one part per rank, when the group's communicator all-reduces and
    reduce-scatters in rank order (DS4.1's TileLang family on the RoCE policy,
    ``rank_order_reduce_scatter``); ``RankOrderParts.sum`` adds as many parts."""
    from vllm.distributed import get_tp_group

    group = get_tp_group()
    world = group.world_size
    comm = group.device_communicator
    if not 2 <= world <= 4 or not getattr(comm, "rank_order_reduce_scatter", False):
        return []
    return [prepare.call(rank_order_sum_kernel, world)]


_RANK_ROWS_THREADS = 128

_RANK_ROWS_HEADER = r"""
__device__ __forceinline__ void vllm_rank_rows_b128(void* d, const void* s) {
  *reinterpret_cast<uint4*>(d) = *reinterpret_cast<const uint4*>(s);
}
"""


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    }
)
def rank_rows_kernel():
    """An all-gather's rank-major shards (``world`` blocks of ``rows`` rows) as their
    rows side by side: ``out[r, w * packs + p] = src[w * rows + r, p]`` in 16-byte
    packs, ``packs`` a shard row. Any dtype: both sides are viewed as int32 words,
    four a pack."""
    src_rows, src_words, out_rows, out_words = T.dynamic(
        "src_rows, src_words, out_rows, out_words"
    )
    threads = _RANK_ROWS_THREADS

    @T.prim_func
    def collectives_rank_rows(
        src: T.Tensor((src_rows, src_words), T.int32),
        out: T.Tensor((out_rows, out_words), T.int32),
        world: T.int32,
        rows: T.int32,
        packs: T.int32,
    ):
        # Rows on the grid's x axis: a prefill gather's rows pass the y axis' 65535.
        with T.Kernel(rows, T.ceildiv(world * packs, threads), threads=threads) as (
            r,
            bx,
        ):
            T.import_source(_RANK_ROWS_HEADER)
            p = bx * threads + T.get_thread_binding()
            if p < world * packs:
                T.call_extern(
                    "handle",
                    "vllm_rank_rows_b128",
                    T.address_of(out[r, p * 4]),
                    T.address_of(src[(p // packs) * rows + r, (p % packs) * 4]),
                )

    return collectives_rank_rows


def rank_rows_calls() -> list[prepare.TileLangCall]:
    """The rank-major-to-concatenated launch of the TP group's last-dimension
    all-gathers that NCCL carries (``concat_rank_shards``)."""
    from vllm.distributed import get_tp_group

    return [] if get_tp_group().world_size < 2 else [prepare.call(rank_rows_kernel)]


def concat_rank_shards(
    gathered: torch.Tensor, world: int, shard_shape: tuple[int, ...]
) -> torch.Tensor | None:
    """An all-gather's rank-major output (``[world * rows, ...]``, each rank's shard
    of ``shard_shape``) as the shards concatenated along their last dimension, in
    one launch, or None when a shard row is not whole 16-byte packs (torch's
    movedim and reshape copy then)."""
    width = shard_shape[-1]
    row_bytes = width * gathered.element_size()
    if len(shard_shape) < 2 or row_bytes % 16 or gathered.data_ptr() % 16:
        return None
    rows = math.prod(shard_shape[:-1])
    out = gathered.new_empty(*shard_shape[:-1], world * width)
    if rows and width:
        rank_rows_kernel()(
            gathered.view(world * rows, width).view(torch.int32),
            out.view(rows, world * width).view(torch.int32),
            world,
            rows,
            row_bytes // 16,
        )
    return out


def rank_order_sum(
    parts: list[torch.Tensor], out: torch.Tensor | None = None
) -> torch.Tensor:
    """Equal-shape tensors added in FP32 in list order and rounded once: BF16 CUDA
    tensors whose element count is a multiple of 8 go through the TileLang kernel,
    anything else through the same arithmetic in torch."""
    if out is None:
        out = torch.empty_like(parts[0])
    n = parts[0].numel()
    if (
        parts[0].is_cuda
        and parts[0].dtype == torch.bfloat16
        and 2 <= len(parts) <= 4
        and n % _VEC == 0
        and all(p.is_contiguous() for p in parts)
        and out.is_contiguous()
    ):
        if n:
            rank_order_sum_kernel(len(parts))(
                *(p.view(-1, _VEC) for p in parts), out.view(-1, _VEC)
            )
        return out
    total = parts[0].to(torch.float32, copy=True)  # never add into an input
    for part in parts[1:]:
        total += part.float()
    return out.copy_(total)


def exchange_into(
    comm, chunks: torch.Tensor, parts: torch.Tensor, rank: int, world: int
) -> None:
    """Send ``chunks[r]`` to every rank ``r`` and receive its chunk for this rank
    into ``parts[r]``; ``chunks[rank]`` and ``parts[rank]`` are not touched.

    ``chunks`` and ``parts`` are ``(world, L, ...)``, each ``[r]`` contiguous (a
    row range of a larger buffer will do). ``comm`` is the group's PyNCCL
    communicator (``send``, ``recv``, ``group_start``, ``group_end``)."""
    if world == 4:
        nxt, prv, opp = (rank + 1) % 4, (rank + 3) % 4, (rank + 2) % 4
        half = -(-chunks.shape[1] // 2)
        # Halves of the opposite rank's chunks, in transit: from the previous rank
        # for the next one (rows :half), from the next rank for the previous one.
        relay_a = torch.empty_like(chunks[0, :half])
        relay_b = torch.empty_like(chunks[0, half:])
        comm.group_start()
        _send(comm, chunks[nxt], nxt)
        _send(comm, chunks[opp, :half], nxt)
        _send(comm, chunks[prv], prv)
        _send(comm, chunks[opp, half:], prv)
        _recv(comm, parts[prv], prv)
        _recv(comm, relay_a, prv)
        _recv(comm, parts[nxt], nxt)
        _recv(comm, relay_b, nxt)
        comm.group_end()
        comm.group_start()
        _send(comm, relay_a, nxt)
        _send(comm, relay_b, prv)
        _recv(comm, parts[opp, :half], prv)
        _recv(comm, parts[opp, half:], nxt)
        comm.group_end()
    else:
        comm.group_start()
        for peer in range(world):
            if peer != rank:
                _send(comm, chunks[peer], peer)
                _recv(comm, parts[peer], peer)
        comm.group_end()


def exchange_rank_order(
    comm, input_tensor: torch.Tensor, rank: int, world: int
) -> list[torch.Tensor]:
    """Every rank's chunk ``rank`` of ``input_tensor`` (contiguous, ``world * L``
    rows), unreduced and in rank order; this rank's own is a view of the input."""
    chunks = input_tensor.view(world, -1, *input_tensor.shape[1:])
    parts = torch.empty_like(chunks)
    exchange_into(comm, chunks, parts, rank, world)
    return [chunks[r] if r == rank else parts[r] for r in range(world)]


@dataclass(frozen=True)
class RankOrderParts:
    """This rank's rows of a sum over ranks, not yet added: ``packed[r]`` is rank
    ``r``'s partial, ``(world, L, ...)`` contiguous. A consumer that reads every
    part can add them itself (TileLang mHC's ``post_pre``); ``sum()`` gives the
    rank-order reduce-scatter's result, the same bits."""

    packed: torch.Tensor

    @property
    def world(self) -> int:
        return int(self.packed.shape[0])

    @property
    def shape(self) -> torch.Size:
        return self.packed.shape[1:]

    def sum(self, out: torch.Tensor | None = None) -> torch.Tensor:
        return rank_order_sum(list(self.packed.unbind(0)), out)


def reduce_scatter_rank_order(
    comm, input_tensor: torch.Tensor, rank: int, world: int
) -> torch.Tensor:
    """Rows ``[rank * L, (rank + 1) * L)`` of the sum over ranks of ``input_tensor``
    (contiguous, ``world * L`` rows), with the one-shot all-reduce's arithmetic: the
    unreduced exchange, then the rank-order sum (this rank's chunk read in place)."""
    return rank_order_sum(exchange_rank_order(comm, input_tensor, rank, world))


def all_reduce_rank_order(
    comm, input_tensor: torch.Tensor, rank: int, world: int
) -> torch.Tensor:
    """The sum over ranks of ``input_tensor`` at any size, with the one-shot
    all-reduce's arithmetic: its rows (padded to a multiple of ``world``) are
    reduce-scattered in rank order, then all-gathered, which moves bits without
    arithmetic. A row has the same bits as the one-shot path would give it. The
    padding rows (zeros, TileLang's fill) are summed and gathered like the others,
    then dropped."""
    rows = input_tensor.shape[0]
    flat = input_tensor.reshape(rows, -1)
    padded = -(-rows // world) * world
    if padded != rows or not flat.is_contiguous():
        from vllm.models.deepseek_v4_1.tilelang import columns

        buffer = flat.new_empty((padded, flat.shape[1]))
        buffer[:rows].copy_(flat)
        columns.zero_(buffer[rows:])
    else:
        buffer = flat
    own = reduce_scatter_rank_order(comm, buffer, rank, world)
    gathered = torch.empty_like(buffer)
    comm.all_gather(gathered, own)
    return gathered[:rows].view(input_tensor.shape)


def _send(comm, tensor: torch.Tensor, peer: int) -> None:
    if tensor.numel():
        comm.send(tensor, peer)


def _recv(comm, tensor: torch.Tensor, peer: int) -> None:
    if tensor.numel():
        comm.recv(tensor, peer)


__all__ = [
    "RankOrderParts",
    "all_reduce_rank_order",
    "exchange_into",
    "exchange_rank_order",
    "rank_order_sum",
    "rank_order_sum_kernel",
    "reduce_scatter_rank_order",
]
