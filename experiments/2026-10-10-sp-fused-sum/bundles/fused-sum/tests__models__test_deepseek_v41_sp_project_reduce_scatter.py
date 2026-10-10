# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A sequence-parallel projection reduce-scattered in slices lands every row where
one projection and one reduce-scatter put it (one rank's view, the others'
contributions stood in by zeros)."""

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
    copies of it)."""
    from types import SimpleNamespace

    from vllm.models.deepseek_v4_1.tilelang import collectives

    world = 4
    rows = sp_prefill.SPRows(
        num_tokens=tokens, world_size=world, rank=rank, unreduced=True
    )
    group = SimpleNamespace(device_communicator=SimpleNamespace(pynccl_comm="comm"))
    monkeypatch.setattr(sp_prefill, "get_tp_group", lambda: group)
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
    if slices == 1 and not per_block:
        assert calls == [(0, tokens)]  # one projection
    padded = torch.zeros(rows.padded_rows, 8, device="cuda")
    padded[:tokens] = source
    blocks = padded.view(world, rows.local_rows, 8)
    assert parts.shape == (world, rows.local_rows, 8)
    for peer in range(world):
        assert torch.equal(parts[peer], blocks[rank])
    sent = torch.cat(sent, dim=1)
    for peer in range(world):
        if peer != rank:
            assert torch.equal(sent[peer], blocks[peer])
    projected = sorted(i for start, stop in calls for i in range(start, stop))
    assert projected == list(range(tokens))
    assert order == [len(calls)]
    assert torch.equal(
        sp_prefill.materialize(sp_prefill.unreduced_parts(parts)),
        collectives.rank_order_sum(list(parts)),
    )
