# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The multi-node RoCE policy has no fallback: every collective runs on the
selected backends (one-shot runtime, PyNCCL) or raises. Nothing reaches the
PyTorch NCCL process group, which the policy never creates (about 3.4 GB of
unified memory per rank on GB10). CPU only."""

import inspect
import re

import pytest
import torch
import torch.distributed

from vllm.distributed.device_communicators import cuda_communicator
from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)
from vllm.distributed.parallel_state import GroupCoordinator

PROCESS_GROUP_COLLECTIVES = (
    "all_reduce",
    "all_gather",
    "all_gather_into_tensor",
    "reduce_scatter_tensor",
    "gather",
    "broadcast",
    "send",
    "recv",
    "all_to_all_single",
    "batch_isend_irecv",
    "barrier",
)


class _PyNccl:
    disabled = False

    def __init__(self):
        self.calls = []

    def broadcast(self, tensor, src):
        self.calls.append(("broadcast", src))

    def batch_isend_irecv(self, p2p_ops):
        self.calls.append(("batch_isend_irecv", len(p2p_ops)))


class _OneShot:
    disabled = False


def _policy_communicator(pynccl=None):
    comm = object.__new__(cuda_communicator.CudaCommunicator)
    comm._roce_policy = True
    comm.b12x_ar_comm = _OneShot()
    comm.pynccl_comm = pynccl if pynccl is not None else _PyNccl()
    comm.world_size = 4
    comm.rank_in_group = 0
    comm.device_group = object()
    return comm


@pytest.fixture(autouse=True)
def no_process_group(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the RoCE policy reached the PyTorch process group")

    for name in PROCESS_GROUP_COLLECTIVES:
        monkeypatch.setattr(torch.distributed, name, refuse, raising=False)


def test_in_place_all_reduce_is_not_part_of_the_policy():
    with pytest.raises(RuntimeError, match="no in-place all-reduce"):
        _policy_communicator().all_reduce_in_place(torch.zeros(4))


def test_gather_is_not_part_of_the_policy():
    with pytest.raises(RuntimeError, match="no gather"):
        _policy_communicator().gather(torch.zeros(4))


def test_batch_isend_irecv_runs_on_pynccl():
    pynccl = _PyNccl()
    _policy_communicator(pynccl).batch_isend_irecv([object(), object()])
    assert pynccl.calls == [("batch_isend_irecv", 2)]


@pytest.mark.parametrize("call", ["broadcast", "batch_isend_irecv"])
def test_lost_pynccl_stops_instead_of_switching_backend(call):
    pynccl = _PyNccl()
    pynccl.disabled = True
    comm = _policy_communicator(pynccl)
    args = (torch.zeros(4),) if call == "broadcast" else ([object()],)
    with pytest.raises(RuntimeError, match="lost its PyNCCL communicator"):
        getattr(comm, call)(*args)


def test_no_inherited_collective_uses_the_process_group():
    """A base-class collective that calls torch.distributed must be overridden,
    so a new upstream collective cannot bypass the policy unnoticed."""
    collective = re.compile(
        r"\b(?:torch\.distributed|dist)\.(?:"
        + "|".join(PROCESS_GROUP_COLLECTIVES)
        + r")\("
    )
    inherited = []
    for name, method in inspect.getmembers(DeviceCommunicatorBase, inspect.isfunction):
        if not collective.search(inspect.getsource(method)):
            continue
        if getattr(cuda_communicator.CudaCommunicator, name) is method:
            inherited.append(name)
    assert inherited == []


class _SparknetAdapter(_OneShot):
    def __init__(self, report: str | None = None):
        self.report = report

    def failure(self) -> str | None:
        return self.report


def test_failure_report_comes_from_the_one_shot_runtime():
    comm = _policy_communicator()
    comm.b12x_ar_comm = _SparknetAdapter(report="saw nothing from rank 2")
    assert comm.roce_failure() == "saw nothing from rank 2"
    comm.b12x_ar_comm.report = None
    assert comm.roce_failure() is None


class _GatheringAdapter(_OneShot):
    """Sparknet's adapter as a broadcast sees it: rank r's copy is
    ``arange(n) + 100 * r``, so the kept slice names its rank."""

    def __init__(self, world: int, eligible: bool = True):
        self.world = world
        self.eligible = eligible
        self.gathers = []

    def should_all_gather(self, inp, dim):
        return self.eligible and inp.is_contiguous() and dim == 0

    def all_gather(self, inp, dim):
        self.gathers.append((tuple(inp.shape), dim))
        base = torch.arange(inp.numel(), dtype=inp.dtype)
        return torch.cat([base + 100 * r for r in range(self.world)])


def _one_shot_communicator(rank, eligible=True, pynccl=None):
    comm = _policy_communicator(pynccl)
    comm.rank_in_group = rank
    comm.b12x_ar_comm = _GatheringAdapter(comm.world_size, eligible)
    return comm


def test_broadcast_keeps_the_source_copy_of_the_one_shot_all_gather():
    comm = _one_shot_communicator(rank=1)
    tensor = torch.full((2, 3), -1.0)
    assert comm.broadcast(tensor, src=2) is tensor
    assert torch.equal(tensor, (torch.arange(6.0) + 200).view(2, 3))
    assert comm.b12x_ar_comm.gathers == [((6,), 0)]
    assert comm.pynccl_comm.calls == []


def test_broadcast_source_keeps_its_own_bytes():
    comm = _one_shot_communicator(rank=2)
    tensor = torch.full((2, 3), 7.0)
    comm.broadcast(tensor, src=2)
    assert torch.equal(tensor, torch.full((2, 3), 7.0))
    # The source takes part in the collective like every other rank.
    assert comm.b12x_ar_comm.gathers == [((6,), 0)]


def test_broadcast_refuses_what_the_all_gather_cannot_take():
    comm = _one_shot_communicator(rank=1, eligible=False)
    with pytest.raises(RuntimeError, match="one-shot broadcast cannot take"):
        comm.broadcast(torch.zeros(4), src=0)
    assert comm.pynccl_comm.calls == []


def _group(device_communicator):
    group = object.__new__(GroupCoordinator)
    group.world_size = 4
    group.ranks = [0, 1, 2, 3]
    group.device_communicator = device_communicator
    group.device_group = object()
    return group


def test_group_broadcast_goes_through_the_policy():
    """The TP group's broadcast (DSpark's per-step confidences) never reaches
    the PyTorch process group under the policy: the one-shot all-gather takes it."""
    pynccl = _PyNccl()
    comm = _one_shot_communicator(rank=1, pynccl=pynccl)
    tensor = torch.zeros(4)
    assert _group(comm).broadcast(tensor, src=0) is tensor
    assert torch.equal(tensor, torch.arange(4.0))
    assert comm.b12x_ar_comm.gathers == [((4,), 0)]
    assert pynccl.calls == []


def test_group_broadcast_without_a_policy_keeps_the_process_group(monkeypatch):
    calls = []
    monkeypatch.setattr(
        torch.distributed,
        "broadcast",
        lambda tensor, src, group: calls.append((src, group)),
    )
    group = _group(device_communicator=None)
    group.broadcast(torch.zeros(4), src=1)
    assert calls == [(1, group.device_group)]


class _PrefillOneShotAdapter(_OneShot):
    def __init__(self, world: int, rank: int, eligible: bool = True):
        self.world = world
        self.rank = rank
        self.eligible = eligible
        self.calls = []

    def should_custom_ar(self, inp):
        return False

    def should_reduce_scatter(self, inp, dim=0):
        return self.eligible and dim == 0

    def reduce_scatter(self, inp, dim=0):
        self.calls.append(("reduce_scatter", tuple(inp.shape), dim))
        chunks = inp.view(self.world, -1, *inp.shape[1:])
        return chunks[self.rank] * float(self.world)

    def should_all_gather(self, inp, dim):
        return self.eligible and dim == 0

    def all_gather(self, inp, dim):
        self.calls.append(("all_gather", tuple(inp.shape), dim))
        return torch.cat([inp + 10.0 * r for r in range(self.world)], dim=0)

    def should_all_to_all_rows(self, inp, rows):
        return self.eligible

    def all_to_all_rows(self, inp, rows):
        self.calls.append(("all_to_all_rows", tuple(inp.shape), rows))
        return torch.full((self.world, rows, *inp.shape[1:]), 42.0, dtype=inp.dtype)


def test_roce_policy_routes_reduce_scatter_all_to_all_and_large_all_reduce():
    pynccl = _PyNccl()
    comm = _policy_communicator(pynccl)
    comm.rank_in_group = 1
    adapter = _PrefillOneShotAdapter(comm.world_size, comm.rank_in_group, eligible=True)
    comm.b12x_ar_comm = adapter

    rs_in = torch.ones(8, 16)
    rs_out = comm.reduce_scatter(rs_in, dim=0)
    assert rs_out.shape == (2, 16)
    assert torch.all(rs_out == 4.0)

    a2a_in = torch.ones(12, 16)
    a2a_out = comm.all_to_all_rows(a2a_in, rows=4)
    assert a2a_out.shape == (4, 4, 16)
    assert torch.all(a2a_out == 42.0)

    ar_in = torch.ones(8, 16)
    ar_out = comm.all_reduce(ar_in)
    assert ar_out.shape == (8, 16)

    assert adapter.calls == [
        ("reduce_scatter", (8, 16), 0),
        ("all_to_all_rows", (12, 16), 4),
        ("reduce_scatter", (8, 16), 0),
        ("all_gather", (2, 16), 0),
    ]
    assert pynccl.calls == []
