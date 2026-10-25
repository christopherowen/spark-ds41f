# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A sequence-parallel projection reduce-scattered in slices lands every row where
one projection and one reduce-scatter put it (one rank's view, the others'
contributions stood in by zeros); the row exchange gives each rank its rows of every
block, over the cabled pairs only."""

import collections
import threading

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device", allow_module_level=True)

from vllm.models.deepseek_v4_1 import sp_prefill


@pytest.mark.parametrize("tokens", [37, 40, 205])
@pytest.mark.parametrize("rank", [0, 2, 3])
@pytest.mark.parametrize("slices", [1, 2, 3])
def test_slices_land_on_the_owned_rows(monkeypatch, tokens, rank, slices):
    world = 4
    rows = sp_prefill.SPRows(num_tokens=tokens, world_size=world, rank=rank)

    def own_block(x, dim=0):  # this rank's block; other ranks add zeros
        return x.view(world, -1, *x.shape[1:])[rank].clone()

    monkeypatch.setattr(sp_prefill, "tensor_model_parallel_reduce_scatter", own_block)
    source = torch.randn(tokens, 8, device="cuda")
    calls = []

    def project(start, stop, out):
        calls.append((start, stop))
        out.copy_(source[start:stop])

    order = []
    result = rows.project_reduce_scatter(
        project,
        (8,),
        source.dtype,
        source.device,
        slices=slices,
        then=lambda: order.append(len(calls)),
    )
    torch.cuda.synchronize()
    padded = torch.zeros(rows.padded_rows, 8, device="cuda")
    padded[:tokens] = source
    assert torch.equal(result, own_block(padded))
    # Every real row projected once; then() after the last projection.
    projected = sorted(i for start, stop in calls for i in range(start, stop))
    assert projected == list(range(tokens))
    assert order == [len(calls)]


def test_wo_slices():
    assert sp_prefill.wo_slices(2047) == 1
    assert sp_prefill.wo_slices(2048) == 2


@pytest.mark.parametrize("tokens", [37, 40, 205])
@pytest.mark.parametrize("rank", [0, 2, 3])
@pytest.mark.parametrize("slices", [1, 2, 3])
@pytest.mark.parametrize("per_block", [False, True])
def test_unreduced_parts_hold_every_block(monkeypatch, tokens, rank, slices, per_block):
    """Unreduced rows send every other rank's block, slice by slice, and keep this
    rank's own block in its slot of the parts (the other ranks stood in by
    copies of it). Only real rows are projected; the pad rows of the last rank's
    block are zeroed by TileLang's fill (never torch's), so the parts and their
    sums are the zero-padded projection's, pad rows included."""
    from types import SimpleNamespace

    from vllm.models.deepseek_v4_1.tilelang import collectives

    world = 4
    rows = sp_prefill.SPRows(
        num_tokens=tokens, world_size=world, rank=rank, unreduced=True
    )
    group = SimpleNamespace(device_communicator=SimpleNamespace(pynccl_comm="comm"))
    monkeypatch.setattr(sp_prefill, "get_tp_group", lambda: group)

    def refuse(self, *args, **kwargs):
        raise AssertionError("a pad row was zeroed by torch")

    monkeypatch.setattr(torch.Tensor, "zero_", refuse)
    sent = []

    def exchange_into(comm, chunks, parts, r, w):
        assert (comm, r, w) == ("comm", rank, world)
        sent.append(chunks.clone())
        for peer in range(world):
            if peer != rank:
                parts[peer].copy_(parts[rank])

    monkeypatch.setattr(collectives, "exchange_into", exchange_into)
    source = torch.randn(tokens, 8, device="cuda")
    calls, order = [], []

    def project(start, stop, out):
        calls.append((start, stop))
        out.copy_(source[start:stop])

    parts = rows.project_reduce_scatter(
        project,
        (8,),
        source.dtype,
        source.device,
        slices=slices,
        then=lambda: order.append(len(calls)),
        per_block=per_block,
    )
    torch.cuda.synchronize()
    monkeypatch.undo()
    if slices == 1 and not per_block:
        assert calls == [(0, tokens)]  # one projection
    padded = torch.zeros(rows.padded_rows, 8, device="cuda")
    padded[:tokens] = source
    blocks = padded.view(world, rows.local_rows, 8)
    local = rows.local_rows
    assert parts.shape == (world, local, 8)
    for peer in range(world):
        assert torch.equal(parts[peer], blocks[rank])
    sent = torch.cat(sent, dim=1)
    for peer in range(world):
        if peer != rank:
            assert torch.equal(sent[peer], blocks[peer])
    projected = sorted(i for start, stop in calls for i in range(start, stop))
    assert projected == list(range(tokens))
    assert order == [len(calls)]
    total = sp_prefill.materialize(sp_prefill.unreduced_parts(parts))
    assert torch.equal(total, collectives.rank_order_sum(list(parts)))
    zero_padded = collectives.rank_order_sum([blocks[rank]] * world)
    assert torch.equal(total, zero_padded)


@pytest.mark.parametrize("tokens", [37, 40, 205])
@pytest.mark.parametrize("world", [3, 4])
def test_row_exchange_gives_each_rank_its_rows_of_every_block(
    monkeypatch, tokens, world
):
    """Every rank receives every rank's column block of its own real rows, and the
    blocks packed side by side equal its rows of the gathered product cut to the
    logical width (the other ranks' all-to-all blocks simulated, each sending only
    real rows)."""
    width, block = 50, 17  # logical width over world * block padded columns
    product = torch.randn(tokens, world * block, device="cuda")
    ranks = [
        sp_prefill.SPRows(num_tokens=tokens, world_size=world, rank=r)
        for r in range(world)
    ]
    locals_ = [product[:, r * block : (r + 1) * block] for r in range(world)]
    for r, rows in enumerate(ranks):
        own, real = rows.local_rows, rows.real_rows

        def all_to_all_rows(x, n, r=r, own=own, real=real):
            assert torch.equal(x, locals_[r]) and n == own
            out = torch.full((world, n, block), float("nan"), device="cuda")
            for s in range(world):  # rank s sends its rows of rank r's block
                out[s, :real] = locals_[s][r * n : r * n + real]
            return out

        monkeypatch.setattr(
            sp_prefill, "tensor_model_parallel_all_to_all_rows", all_to_all_rows
        )
        blocks = rows.exchange_rows(locals_[r])
        assert blocks.shape == (world, own, block)
        for s in range(world):
            assert torch.equal(
                blocks[s, :real],
                product[rows.start : rows.start + real, s * block : (s + 1) * block],
            )
        packed = rows.pack_columns(blocks, width)
        assert packed.shape == (own, width) and packed.is_contiguous()
        assert torch.equal(
            packed[:real], product[rows.start : rows.start + real, :width]
        )


class _Fabric:
    """PyNCCL ranks as threads: when every rank has ended a group, each receive takes
    the next send from its peer to it (NCCL's per-pair order), shapes checked and
    every send consumed. Records which pairs exchanged."""

    def __init__(self, world: int):
        self.world = world
        self.barrier = threading.Barrier(world, timeout=30)
        self.ops: list[list] = [[] for _ in range(world)]
        self.pairs: set[frozenset] = set()

    def comm(self, rank: int):
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

        comm = PyNcclCommunicator.__new__(PyNcclCommunicator)
        comm.disabled, comm.world_size, comm.rank = False, self.world, rank
        comm.group_start = lambda: None
        comm.send = lambda t, peer, stream: self.ops[rank].append(("send", peer, t))
        comm.recv = lambda t, peer, stream: self.ops[rank].append(("recv", peer, t))
        comm.group_end = lambda: self._end(rank)
        return comm

    def _end(self, rank: int) -> None:
        self.barrier.wait()
        if rank == 0:
            sends = collections.defaultdict(collections.deque)
            for source, ops in enumerate(self.ops):
                for kind, peer, tensor in ops:
                    if kind == "send":
                        sends[(source, peer)].append(tensor)
            for destination, ops in enumerate(self.ops):
                for kind, peer, tensor in ops:
                    if kind == "recv":
                        sent = sends[(peer, destination)].popleft()
                        assert sent.shape == tensor.shape, (peer, destination)
                        tensor.copy_(sent)
                        self.pairs.add(frozenset((peer, destination)))
            assert not any(sends.values()), "a send nobody received"
            self.ops = [[] for _ in range(self.world)]
        self.barrier.wait()


@pytest.mark.parametrize("world", [3, 4])
@pytest.mark.parametrize("short", [0, 3, 9, 10, 25])
def test_row_all_to_all_over_cabled_pairs(world, short, rows=10):
    """PyNCCL's ``all_to_all_rows``, every rank a thread: each rank receives every
    rank's block of its real rows bit for bit and later rows stay unwritten, with the
    last blocks short or empty; at four ranks (the switchless ring) only neighbours
    exchange, since ranks two apart share no cable."""
    total = max(1, world * rows - short)
    fabric = _Fabric(world)
    inputs = [torch.randn(total, 6) for _ in range(world)]
    outputs = [torch.full((world, rows, 6), float("nan")) for _ in range(world)]
    errors = []

    def run(rank: int) -> None:
        try:
            fabric.comm(rank).all_to_all_rows(
                outputs[rank], inputs[rank], rows, stream=object()
            )
        except BaseException as error:  # reported below, with the rank
            errors.append((rank, error))
            fabric.barrier.abort()

    threads = [threading.Thread(target=run, args=(r,)) for r in range(world)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors, errors
    for me in range(world):
        real = max(0, min(rows, total - me * rows))
        for source in range(world):
            got = outputs[me][source]
            assert torch.equal(got[:real], inputs[source][me * rows : me * rows + real])
            assert got[real:].isnan().all(), (me, source)
    if world == 4:
        ring = {frozenset((r, (r + 1) % 4)) for r in range(4)}
        assert fabric.pairs <= ring, sorted(map(sorted, fabric.pairs - ring))
