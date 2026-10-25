# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The rank-order reduce-scatter: the exchange (simulated ranks on the host, the
neighbour relay at four ranks) and the TileLang sum (GPU)."""

import threading
from collections import defaultdict, deque

import pytest
import torch

from vllm.models.deepseek_v4_1.tilelang.collectives import (
    RankOrderParts,
    all_reduce_rank_order,
    exchange_into,
    rank_order_sum,
    reduce_scatter_rank_order,
)


class _Fabric:
    """Point-to-point links between simulated ranks. At four ranks only the ring
    neighbours are cabled, as on ring4: a send to the opposite rank fails."""

    def __init__(self, world: int):
        self.world = world
        self.barrier = threading.Barrier(world)
        self.queues = defaultdict(deque)
        self.gathered = {}
        self.lock = threading.Lock()


class _Comm:
    """One rank's PyNCCL stand-in: sends post at group end, receives match them in
    order per (source, destination) pair after every rank has posted."""

    def __init__(self, fabric: _Fabric, rank: int):
        self.fabric, self.rank = fabric, rank
        self.sends, self.recvs = [], []

    def group_start(self):
        self.sends, self.recvs = [], []

    def send(self, tensor, peer):
        if self.fabric.world == 4 and (peer - self.rank) % 4 == 2:
            raise AssertionError(f"rank {self.rank} sent to its opposite rank {peer}")
        self.sends.append((peer, tensor.clone()))

    def recv(self, tensor, peer):
        self.recvs.append((peer, tensor))

    def all_gather(self, output, tensor):
        """Every rank's ``tensor`` into ``output`` in rank order (data movement)."""
        fabric = self.fabric
        with fabric.lock:
            fabric.gathered[self.rank] = tensor.clone()
        fabric.barrier.wait()
        output.copy_(torch.cat([fabric.gathered[r] for r in range(fabric.world)]))
        fabric.barrier.wait()

    def group_end(self):
        fabric = self.fabric
        with fabric.lock:
            for peer, tensor in self.sends:
                fabric.queues[(self.rank, peer)].append(tensor)
        fabric.barrier.wait()
        for peer, tensor in self.recvs:
            with fabric.lock:
                tensor.copy_(fabric.queues[(peer, self.rank)].popleft())
        fabric.barrier.wait()


@pytest.fixture(autouse=True)
def _host_zero_fill(monkeypatch):
    """The rank-order all-reduce zeroes its padding rows with TileLang's 16-byte
    fill, a CUDA kernel. These tests run the collectives on host tensors, where a
    host fill stands in for it; on CUDA tensors the kernel itself runs."""
    from vllm.models.deepseek_v4_1.tilelang import columns

    zero_ = columns.zero_
    monkeypatch.setattr(
        columns, "zero_", lambda out: zero_(out) if out.is_cuda else out.fill_(0)
    )


def _run(inputs, collective=None):
    world = len(inputs)
    fabric = _Fabric(world)
    out, errors = [None] * world, []
    if collective is None:
        collective = reduce_scatter_rank_order

    def rank_main(rank):
        try:
            out[rank] = collective(_Comm(fabric, rank), inputs[rank], rank, world)
        except BaseException as error:  # surface failures from the threads
            errors.append(error)
            fabric.barrier.abort()

    threads = [threading.Thread(target=rank_main, args=(r,)) for r in range(world)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    assert all(not q for q in fabric.queues.values()), "unmatched sends"
    return out


def _reference(values):
    """The one-shot all-reduce's arithmetic: FP32 in rank order, one rounding."""
    total = values[0].float()
    for value in values[1:]:
        total += value.float()
    return total.to(values[0].dtype)


@pytest.mark.parametrize("world", [2, 3, 4])
@pytest.mark.parametrize("local", [1, 4, 7])
def test_rows_add_as_the_one_shot_all_reduce(world, local):
    torch.manual_seed(world * 10 + local)
    width = 40
    inputs = [
        torch.randn(world * local, width).to(torch.bfloat16) for _ in range(world)
    ]
    out = _run(inputs)
    for rank in range(world):
        rows = slice(rank * local, (rank + 1) * local)
        assert torch.equal(out[rank], _reference([x[rows] for x in inputs]))


def _exchange_in_slices(bounds):
    """The exchange of every row range ``bounds[i]:bounds[i + 1]`` of each rank's
    block into one parts buffer, this rank's own block written in its slot."""

    def collective(comm, x, rank, world):
        chunks = x.view(world, -1, *x.shape[1:])
        parts = torch.full_like(chunks, float("nan"))
        parts[rank] = chunks[rank]
        for lo, hi in zip(bounds, bounds[1:]):
            exchange_into(comm, chunks[:, lo:hi], parts[:, lo:hi], rank, world)
        return RankOrderParts(parts)

    return collective


@pytest.mark.parametrize("world", [2, 3, 4])
@pytest.mark.parametrize("bounds", [(0, 7), (0, 3, 7), (0, 1, 2, 7)])
def test_sliced_exchange_into_parts_sums_alike(world, bounds):
    torch.manual_seed(world * 100 + len(bounds))
    local, width = bounds[-1], 24
    inputs = [
        torch.randn(world * local, width).to(torch.bfloat16) for _ in range(world)
    ]
    whole = _run(inputs)
    sliced = _run(inputs, _exchange_in_slices(bounds))
    for rank in range(world):
        rows = slice(rank * local, (rank + 1) * local)
        assert sliced[rank].world == world and sliced[rank].shape == (local, width)
        for r in range(world):
            assert torch.equal(sliced[rank].packed[r], inputs[r][rows])
        assert torch.equal(sliced[rank].sum(), whole[rank])


@pytest.mark.parametrize("world", [2, 3, 4])
@pytest.mark.parametrize("rows", [1, 5, 8, 13])
def test_all_reduce_adds_as_the_one_shot_all_reduce(world, rows):
    # Every rank gets every row, each summed in rank order with one rounding,
    # whatever the row count (padding rows are dropped).
    torch.manual_seed(world * 100 + rows)
    width = 24
    inputs = [torch.randn(rows, width).to(torch.bfloat16) for _ in range(world)]
    out = _run(inputs, all_reduce_rank_order)
    want = _reference(inputs)
    for rank in range(world):
        assert out[rank].shape == (rows, width)
        assert torch.equal(out[rank], want)


@pytest.mark.parametrize("world", [2, 3, 4])
@pytest.mark.parametrize("rows", [1, 5, 13])
def test_all_reduce_pads_with_tilelang_zeros(monkeypatch, world, rows):
    """The padding rows are zeroed by TileLang's fill (stood in for on the host),
    then summed, gathered and dropped: with the padded buffer NaN where the
    allocator left it, only the padding rows are filled, by that fill alone, and
    every real row is the one-shot sum, bit for bit."""
    from vllm.models.deepseek_v4_1.tilelang import columns

    allocate = torch.Tensor.new_empty
    zeroed = []

    def poisoned(self, *args, **kwargs):
        return allocate(self, *args, **kwargs).fill_(float("nan"))

    def refuse(self, *args, **kwargs):
        raise AssertionError("the padded buffer was zero-filled by torch")

    def zero_(out):
        zeroed.append(out.shape[0])
        return out.fill_(0)

    monkeypatch.setattr(torch.Tensor, "new_empty", poisoned)
    monkeypatch.setattr(torch.Tensor, "new_zeros", refuse)
    monkeypatch.setattr(columns, "zero_", zero_)
    torch.manual_seed(world * 1000 + rows)
    width = 24
    inputs = [torch.randn(rows, width).to(torch.bfloat16) for _ in range(world)]
    out = _run(inputs, all_reduce_rank_order)
    want = _reference(inputs)
    padding = -(-rows // world) * world - rows
    assert zeroed == ([padding] * world if padding else [])
    for rank in range(world):
        assert out[rank].shape == (rows, width)
        assert torch.equal(out[rank], want)


def test_row_bits_do_not_depend_on_owner_or_batch():
    # One row's partials at every position of batches of two sizes: the reduced
    # row has the same bits wherever it lands and whichever rank owns it.
    torch.manual_seed(1)
    world, width = 4, 64
    row = [torch.randn(width).to(torch.bfloat16) * 10**k for k in range(world)]
    results = set()
    for local in (3, 6):
        for position in range(world * local):
            inputs = [
                torch.randn(world * local, width).to(torch.bfloat16)
                for _ in range(world)
            ]
            for x, part in zip(inputs, row):
                x[position] = part
            owner, offset = divmod(position, local)
            results.add(_run(inputs)[owner][offset].view(torch.int16).numpy().tobytes())
    assert len(results) == 1


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_torch_sum_leaves_its_inputs(dtype):
    parts = [torch.randn(3, 8).to(dtype) for _ in range(4)]
    before = [p.clone() for p in parts]
    first = rank_order_sum(parts)
    assert all(torch.equal(p, b) for p, b in zip(parts, before))
    assert torch.equal(rank_order_sum(parts), first)


def test_one_rounding_differs_from_rounding_each_add():
    # FP32 accumulation is observable: rounding after each BF16 add (as NCCL does
    # per hop) gives other bits, so the tests above would catch it.
    torch.manual_seed(2)
    values = [torch.randn(4096).to(torch.bfloat16) * 10**k for k in range(4)]
    each = values[0].clone()
    for value in values[1:]:
        each.add_(value)
    assert not torch.equal(each, _reference(values))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("world", [2, 3, 4])
def test_tilelang_sum_rounds_as_float32_in_rank_order(world):
    torch.manual_seed(world)
    parts = [
        torch.randn(4001, 5120, device="cuda").bfloat16() * 10**k for k in range(world)
    ]
    big = torch.finfo(torch.bfloat16).max
    tiny = torch.finfo(torch.bfloat16).tiny
    specials = torch.tensor(
        [
            0.0,
            -0.0,
            tiny,
            -tiny,
            tiny / 4,
            -tiny / 4,
            big,
            -big,
            float("inf"),
            float("-inf"),
            float("nan"),
            3.0,
            -3.0,
            1e-30,
        ],
        device="cuda",
    ).bfloat16()
    for i, part in enumerate(parts):
        part.view(-1)[: specials.numel()] = specials.roll(i)
    got = rank_order_sum(parts)
    want = _reference(parts)
    _assert_same_bits(got, want)
    # Equal under CUDA graph replays, and on views of one buffer (the exchange's).
    stacked = torch.stack(parts)
    out = torch.empty_like(parts[0])
    graph = torch.cuda.CUDAGraph()
    rank_order_sum(list(stacked), out)
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        rank_order_sum(list(stacked), out)
    for _ in range(20):
        graph.replay()
        torch.cuda.synchronize()
        _assert_same_bits(out, want)


def _assert_same_bits(got, want):
    """Bit for bit, except that a NaN only has to be a NaN (payloads differ by
    conversion routine, and a NaN hidden state is lost either way)."""
    nan = want.isnan()
    assert torch.equal(got.isnan(), nan)
    assert torch.equal(got.view(torch.int16)[~nan], want.view(torch.int16)[~nan])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("world", [2, 3, 4])
@pytest.mark.parametrize(
    "shard, dtype",
    [
        ((8192 * 6, 256), torch.bfloat16),  # TP4 disk Engram columns, 8K prefill
        # The profile run's 16K tokens: rows past the grid's y-axis limit (65535).
        ((16384 * 6, 256), torch.bfloat16),
        ((17, 3, 2048), torch.bfloat16),
        ((5, 32320), torch.float32),  # vocab-parallel logits
        ((9, 20), torch.int64),
        ((1, 8), torch.bfloat16),
    ],
)
def test_rank_shards_concatenate_like_movedim(world, shard, dtype):
    """NCCL's rank-major all-gather output as the shards side by side along their
    last dimension, as the communicator's movedim and reshape gave it, any dtype
    (16-byte packs moved as bits)."""
    from vllm.models.deepseek_v4_1.tilelang.collectives import concat_rank_shards

    torch.manual_seed(world)
    # Random bits of the dtype's width: NaN and infinity patterns included.
    bits = {2: torch.int16, 4: torch.int32, 8: torch.int64}[dtype.itemsize]
    gathered = torch.randint(
        -(2**15), 2**15, (world * shard[0], *shard[1:]), device="cuda"
    )
    gathered = (gathered * 40503).to(bits).view(dtype)
    want = (
        gathered.reshape((world, *shard))
        .movedim(0, len(shard) - 1)
        .reshape(*shard[:-1], world * shard[-1])
    )
    got = concat_rank_shards(gathered, world, shard)
    assert got is not None and got.shape == want.shape and got.is_contiguous()
    assert torch.equal(got.view(torch.uint8), want.contiguous().view(torch.uint8))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_rank_shards_of_partial_packs_are_left_to_torch():
    from vllm.models.deepseek_v4_1.tilelang.collectives import concat_rank_shards

    gathered = torch.zeros(8, 5, dtype=torch.bfloat16, device="cuda")
    assert concat_rank_shards(gathered, 2, (4, 5)) is None
