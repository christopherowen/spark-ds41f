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
