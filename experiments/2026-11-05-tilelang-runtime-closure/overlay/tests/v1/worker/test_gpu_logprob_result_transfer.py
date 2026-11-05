# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The actual result handoff preserves strided IDs and asynchronous lifetimes."""

import pytest
import torch

from vllm.v1.outputs import LogprobsTensors

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("rows", [0, 1, 3, 96])
@pytest.mark.parametrize("stride", [1, 2, 33])
def test_native_column_transfer(dtype, rows, stride):
    # Shifted storage offset and inter-row holes, including an empty result.
    storage = torch.arange(7 + max(rows, 1) * stride, dtype=dtype, device="cuda")
    ids = storage[7 : 7 + rows * stride : stride].unsqueeze(1)
    scores = torch.randn((rows, 1), device="cuda")
    ranks = torch.arange(rows, device="cuda", dtype=torch.int64)
    boundaries = torch.tensor([0, rows], device="cuda", dtype=torch.int32)
    gpu = LogprobsTensors(ids, scores, ranks, None, boundaries)
    cpu = gpu.to_cpu_nonblocking()
    torch.cuda.synchronize()
    assert torch.equal(cpu.logprob_token_ids, ids.cpu())
    assert torch.equal(cpu.logprobs, scores.cpu())
    assert torch.equal(cpu.selected_token_ranks, ranks.cpu())
    assert cpu.logprob_token_ids.dtype == dtype
    if rows > 1:
        assert cpu.logprob_token_ids.stride() == ids.stride()
    assert cpu.tolists().cu_num_generated_tokens == [0, rows]
    assert cpu.to_cpu_nonblocking() is cpu


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_side_stream_result_lifetime(dtype):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        ids = torch.arange(96 * 33, dtype=dtype, device="cuda")[::33, None]
        expected = ids.cpu()
        gpu = LogprobsTensors(
            ids,
            torch.ones((96, 1), device="cuda"),
            torch.ones(96, device="cuda", dtype=torch.int64),
        )
        cpu = gpu.to_cpu_nonblocking()
        done = torch.cuda.Event()
        done.record()
        del gpu, ids
        # Reuse the allocator while the transfer's originating stream is live.
        scratch = torch.empty(96 * 33, device="cuda", dtype=dtype)
        scratch.fill_(-1)
    done.synchronize()
    assert torch.equal(cpu.logprob_token_ids, expected)
