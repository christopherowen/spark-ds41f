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
import tilelang
import tilelang.language as T
import torch

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
        def rank_order_sum(
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
        def rank_order_sum(
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
        def rank_order_sum(
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

    return rank_order_sum


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
    total = parts[0].float()
    for part in parts[1:]:
        total += part.float()
    return out.copy_(total)


# Bytes of one rank's chunk per pipeline slice: larger chunks are exchanged in
# slices so each slice's relay hop and sum run under the next slice's transfer.
SLICE_BYTES = 4 << 20
MAX_SLICES = 8


def pipeline_slices(chunk_bytes: int) -> int:
    """Pipeline slices for a reduce-scatter whose per-rank chunk has ``chunk_bytes``."""
    return max(1, min(MAX_SLICES, chunk_bytes // SLICE_BYTES))


def reduce_scatter_rank_order(
    comm, input_tensor: torch.Tensor, rank: int, world: int, slices: int | None = None
) -> torch.Tensor:
    """Rows ``[rank * L, (rank + 1) * L)`` of the sum over ranks of ``input_tensor``
    (contiguous, ``world * L`` rows), with the one-shot all-reduce's arithmetic.

    ``comm`` is the group's PyNCCL communicator (``send``, ``recv``,
    ``group_start``, ``group_end``). At four ranks the rows go in ``slices``
    (default by size): each NCCL group carries one slice's direct chunks and
    relay halves and the previous slice's relayed halves, and the previous slice
    is summed on a side stream meanwhile. Slicing changes no sum.
    """
    chunks = input_tensor.view(world, -1, *input_tensor.shape[1:])
    rows = chunks.shape[1]
    parts = torch.empty_like(chunks)
    out = torch.empty_like(chunks[0])

    def sources(lo, hi):  # this rank's own chunk is read in place
        return [
            chunks[r, lo:hi] if r == rank else parts[r, lo:hi] for r in range(world)
        ]

    if world != 4:
        comm.group_start()
        for peer in range(world):
            if peer != rank:
                _send(comm, chunks[peer], peer)
                _recv(comm, parts[peer], peer)
        comm.group_end()
        return rank_order_sum(sources(0, rows), out)
    if slices is None:
        slices = pipeline_slices(chunks[0].numel() * chunks.element_size())
    slices = max(1, min(slices, rows))
    bounds = [rows * i // slices for i in range(slices + 1)]
    nxt, prv, opp = (rank + 1) % 4, (rank + 3) % 4, (rank + 2) % 4
    side = None
    if slices > 1 and input_tensor.is_cuda:
        main = torch.cuda.current_stream(input_tensor.device)
        side = _side_stream(input_tensor.device)
    relays = []
    for s in range(slices + 1):
        comm.group_start()
        if s < slices:
            # Slice s: the neighbours' chunks, and halves of the opposite rank's
            # chunk to relay (rows lo:mid via the next rank, mid:hi via the
            # previous one); the halves in transit arrive from the neighbours.
            lo, hi = bounds[s], bounds[s + 1]
            mid = lo + -(-(hi - lo) // 2)
            relay_a = torch.empty_like(chunks[0, lo:mid])
            relay_b = torch.empty_like(chunks[0, mid:hi])
            relays.append((relay_a, relay_b))
            _send(comm, chunks[nxt, lo:hi], nxt)
            _send(comm, chunks[opp, lo:mid], nxt)
            _send(comm, chunks[prv, lo:hi], prv)
            _send(comm, chunks[opp, mid:hi], prv)
            _recv(comm, parts[prv, lo:hi], prv)
            _recv(comm, relay_a, prv)
            _recv(comm, parts[nxt, lo:hi], nxt)
            _recv(comm, relay_b, nxt)
        if s > 0:
            # Slice s - 1: forward the halves in transit; receive the opposite
            # rank's halves of this rank's own rows.
            lo, hi = bounds[s - 1], bounds[s]
            mid = lo + -(-(hi - lo) // 2)
            relay_a, relay_b = relays[s - 1]
            _send(comm, relay_a, nxt)
            _send(comm, relay_b, prv)
            _recv(comm, parts[opp, lo:mid], prv)
            _recv(comm, parts[opp, mid:hi], nxt)
        comm.group_end()
        if s > 0:
            lo, hi = bounds[s - 1], bounds[s]
            if side is None:
                rank_order_sum(sources(lo, hi), out[lo:hi])
            else:
                side.wait_stream(main)
                with torch.cuda.stream(side):
                    rank_order_sum(sources(lo, hi), out[lo:hi])
    if side is not None:
        main.wait_stream(side)
    return out


_SIDE_STREAMS: dict = {}


def _side_stream(device: torch.device) -> torch.cuda.Stream:
    """One side stream per device for the pipelined sums."""
    if device not in _SIDE_STREAMS:
        _SIDE_STREAMS[device] = torch.cuda.Stream(device)
    return _SIDE_STREAMS[device]


def _send(comm, tensor: torch.Tensor, peer: int) -> None:
    if tensor.numel():
        comm.send(tensor, peer)


def _recv(comm, tensor: torch.Tensor, peer: int) -> None:
    if tensor.numel():
        comm.recv(tensor, peer)


__all__ = [
    "pipeline_slices",
    "rank_order_sum",
    "rank_order_sum_kernel",
    "reduce_scatter_rank_order",
]
