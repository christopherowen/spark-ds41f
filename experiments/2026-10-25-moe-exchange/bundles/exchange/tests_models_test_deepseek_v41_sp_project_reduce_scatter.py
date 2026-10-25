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


def test_moe_slices():
    assert sp_prefill.moe_slices(205) == 1
    assert sp_prefill.moe_slices(8191) == 1
    assert sp_prefill.moe_slices(8192) == 2
    assert sp_prefill.moe_slices(15395) == 3  # request inventory 5's prompt
    assert sp_prefill.moe_slices(16288) == 3  # the TP4 recipe's long-prefill chunk
    assert sp_prefill.moe_slices(16384) == 4  # a full step of its token budget


def _fake_exchange(monkeypatch, rank, world, own_rows, events=None):
    """The TP group and ``exchange_into`` faked for one rank: each call's chunks are
    cloned as sent, and every other rank's partial of this rank's rows is stood in
    by this rank's own (``own_rows``, ``(L, ...)``), delivered on the exchange's
    stream; the slice comes from the parts' row offset. Returns the sent clones."""
    from types import SimpleNamespace

    from vllm.models.deepseek_v4_1.tilelang import collectives

    group = SimpleNamespace(device_communicator=SimpleNamespace(pynccl_comm="comm"))
    monkeypatch.setattr(sp_prefill, "get_tp_group", lambda: group)
    sent = []

    def exchange_into(comm, chunks, parts, r, w):
        assert (comm, r, w) == ("comm", rank, world)
        lo = parts.storage_offset() // parts.stride(1)
        hi = lo + parts.shape[1]
        if events is not None:
            events.append(("exchange", lo, hi))
        sent.append(chunks.clone())
        for peer in range(world):
            if peer != rank:
                parts[peer].copy_(own_rows[lo:hi])

    monkeypatch.setattr(collectives, "exchange_into", exchange_into)
    return sent


def _assert_exchanged_under_own_block(events, rank, world, local, tokens):
    """Every slice's exchange comes after the other ranks' projected rows of it and
    before this rank's own."""
    for kind, lo, hi in events:
        if kind != "exchange":
            continue
        at = events.index((kind, lo, hi))
        for owner in range(world):
            start, stop = owner * local + lo, min(owner * local + hi, tokens)
            if stop > start:
                index = events.index(("project", start, stop))
                assert (index > at) == (owner == rank), (owner, lo, hi)


@pytest.mark.parametrize("tokens", [37, 40, 205])
@pytest.mark.parametrize("rank", [0, 2, 3])
@pytest.mark.parametrize("slices", [1, 2, 3])
@pytest.mark.parametrize("per_block", [False, True])
def test_unreduced_parts_hold_every_block(monkeypatch, tokens, rank, slices, per_block):
    """Unreduced rows send every other rank's block, slice by slice, and keep this
    rank's own block in its slot of the parts (the other ranks stood in by
    copies of it). Only real rows are projected; the pad rows of the last rank's
    block are zeroed by TileLang's fill (never torch's), so the parts and their
    sums are the zero-padded projection's, pad rows included. Block by block, a
    slice is exchanged once the other ranks' blocks of it are projected and
    before this rank's own is."""
    from vllm.models.deepseek_v4_1.tilelang import collectives

    world = 4
    rows = sp_prefill.SPRows(
        num_tokens=tokens, world_size=world, rank=rank, unreduced=True
    )
    local = rows.local_rows
    source = torch.randn(tokens, 8, device="cuda")
    padded = torch.zeros(rows.padded_rows, 8, device="cuda")
    padded[:tokens] = source
    blocks = padded.view(world, local, 8)
    events = []
    sent = _fake_exchange(monkeypatch, rank, world, blocks[rank], events)

    def refuse(self, *args, **kwargs):
        raise AssertionError("a pad row was zeroed by torch")

    monkeypatch.setattr(torch.Tensor, "zero_", refuse)
    order = []

    def project(start, stop, out):
        events.append(("project", start, stop))
        out.copy_(source[start:stop])

    parts = rows.project_reduce_scatter(
        project,
        (8,),
        source.dtype,
        source.device,
        slices=slices,
        then=lambda: order.append(len(events)),
        per_block=per_block,
    )
    torch.cuda.synchronize()
    monkeypatch.undo()
    calls = [(start, stop) for kind, start, stop in events if kind == "project"]
    if slices == 1 and not per_block:
        assert calls == [(0, tokens)]  # one projection
        assert events[-1] == ("exchange", 0, local)
        assert order == [len(events) - 1]  # then() before the exchange
    else:
        _assert_exchanged_under_own_block(events, rank, world, local, tokens)
        assert order == [len(events)]  # then() after the last projection
    assert parts.shape == (world, local, 8)
    for peer in range(world):
        assert torch.equal(parts[peer], blocks[rank])
    sent = torch.cat(sent, dim=1)
    for peer in range(world):
        if peer != rank:
            assert torch.equal(sent[peer], blocks[peer])
    projected = sorted(i for start, stop in calls for i in range(start, stop))
    assert projected == list(range(tokens))
    total = sp_prefill.materialize(sp_prefill.unreduced_parts(parts))
    assert torch.equal(total, collectives.rank_order_sum(list(parts)))
    zero_padded = collectives.rank_order_sum([blocks[rank]] * world)
    assert torch.equal(total, zero_padded)


def _moe_case(tokens, hidden=384, intermediate=576, num_experts=32, topk=6):
    """A V4.1 TileLang MoE plan with the FP32 combine the shared expert folds into,
    its prepared weights, and one forward's tokens, routes and shared output."""
    from vllm.models.deepseek_v4_1.tilelang.moe import MoeExpertsV41Plan

    gen = torch.Generator(device="cuda").manual_seed(0)

    def experts(n, k):  # packed E2M1 [E, n, k / 2] and UE8M0 [E, n, k / 32]
        packed = torch.randint(
            0, 256, (num_experts, n, k // 2), generator=gen, device="cuda"
        ).to(torch.uint8)
        exps = torch.randint(
            118, 122, (num_experts, n, k // 32), generator=gen, device="cuda"
        ).to(torch.uint8)
        return packed, exps

    plan = MoeExpertsV41Plan(
        num_experts, topk, hidden, intermediate, swiglu_limit=10.0,
        out_dtype=torch.float32,
    )  # fmt: skip
    plan.declare_kernels()
    weights = plan.prepare_weights(
        *experts(2 * intermediate, hidden), *experts(hidden, intermediate)
    )
    x = torch.randn((tokens, hidden), generator=gen, device="cuda") * 0.5
    shared = torch.randn((tokens, hidden), generator=gen, device="cuda")
    scores = torch.rand((tokens, num_experts), generator=gen, device="cuda")
    ids = scores.argsort(dim=1)[:, :topk].to(torch.int32).contiguous()
    route_weights = torch.rand((tokens, topk), generator=gen, device="cuda") + 0.1
    return plan, weights, x.bfloat16(), shared.bfloat16(), ids, route_weights


@pytest.mark.parametrize("tokens", [37, 205])
@pytest.mark.parametrize("rank", [0, 3])
@pytest.mark.parametrize("slice_tokens", [4096, 64, 16])
def test_moe_combine_writes_the_exchange_slots(
    monkeypatch, default_vllm_config, tokens, rank, slice_tokens
):
    """The MoE runner's reduce-scatter hook, given the experts' deferred combine
    (shared expert folded in), has it write the exchange slots block by block and
    slice by slice: each other rank's block into its send slot and this rank's into
    its part carry the bits the one combine into the whole output gave, which the
    copies moved there before. Pad rows are TileLang's zeros, and each slice is
    exchanged before this rank's own block of it is combined."""
    from vllm.platforms import current_platform

    if not current_platform.is_device_capability_family(120):
        pytest.skip("TileLang V4.1 kernels target SM120/SM121")
    pytest.importorskip("tilelang")
    pytest.importorskip("tile_kernels")
    world, hidden = 4, 384
    plan, weights, x, shared, ids, route_weights = _moe_case(tokens, hidden)
    whole = torch.empty((tokens, hidden), dtype=torch.bfloat16, device="cuda")
    plan.run(x, ids, route_weights, weights, whole, shared=shared)
    combine = plan.run(x, ids, route_weights, weights, None, shared=shared)

    rows = sp_prefill.SPRows(
        num_tokens=tokens, world_size=world, rank=rank, unreduced=True
    )
    local = rows.local_rows
    padded = torch.zeros(
        (rows.padded_rows, hidden), dtype=torch.bfloat16, device="cuda"
    )
    padded[:tokens] = whole
    blocks = padded.view(world, local, hidden)
    events = []
    sent = _fake_exchange(monkeypatch, rank, world, blocks[rank], events)
    monkeypatch.setattr(sp_prefill, "MOE_SLICE_TOKENS", slice_tokens)
    slices = sp_prefill.moe_slices(tokens)

    def refuse(self, *args, **kwargs):
        raise AssertionError("a pad row was zeroed by torch")

    monkeypatch.setattr(torch.Tensor, "zero_", refuse)

    def project(start, stop, out):
        events.append(("project", start, stop))
        combine(start, stop, out)

    with sp_prefill.activate(rows):
        parts = sp_prefill.project_reduce_scatter_current(
            project, tokens, rows.padded_rows, (hidden,), torch.bfloat16, x.device
        )
    torch.cuda.synchronize()
    monkeypatch.undo()
    assert sum(kind == "exchange" for kind, _, _ in events) == slices
    _assert_exchanged_under_own_block(events, rank, world, local, tokens)
    assert torch.equal(parts[rank], blocks[rank])
    sent = torch.cat(sent, dim=1)
    for peer in range(world):
        if peer != rank:
            assert torch.equal(sent[peer], blocks[peer])


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
