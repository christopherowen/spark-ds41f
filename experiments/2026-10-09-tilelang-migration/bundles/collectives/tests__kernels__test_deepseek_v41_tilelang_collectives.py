# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The rank-order reduce-scatter: the exchange (simulated ranks on the host, the
neighbour relay at four ranks) and the TileLang sum (GPU)."""

import threading
from collections import defaultdict, deque

import pytest
import torch

from vllm.models.deepseek_v4_1.tilelang.collectives import (
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


def _run(inputs, slices=None):
    world = len(inputs)
    fabric = _Fabric(world)
    out, errors = [None] * world, []

    def rank_main(rank):
        try:
            out[rank] = reduce_scatter_rank_order(
                _Comm(fabric, rank), inputs[rank], rank, world, slices=slices
            )
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
@pytest.mark.parametrize("slices", [None, 1, 2, 3, 8])
def test_rows_add_as_the_one_shot_all_reduce(world, local, slices):
    torch.manual_seed(world * 10 + local)
    width = 40
    inputs = [
        torch.randn(world * local, width).to(torch.bfloat16) for _ in range(world)
    ]
    out = _run(inputs, slices)
    for rank in range(world):
        rows = slice(rank * local, (rank + 1) * local)
        assert torch.equal(out[rank], _reference([x[rows] for x in inputs]))


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
