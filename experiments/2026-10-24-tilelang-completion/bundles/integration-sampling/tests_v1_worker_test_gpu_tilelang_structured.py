# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The TileLang grammar bitmask against the Triton kernel it replaced, bit for bit."""

import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from vllm.v1.worker.gpu import tilelang_structured
from vllm.v1.worker.gpu.structured_outputs import _apply_grammar_bitmask_kernel

DEVICE = torch.device("cuda")
MASK_STRIDE = 8


def _case(vocab, seed, dtype, invalid):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    counts = [6, 1, 4, 6, 3]
    cu = torch.tensor([0, 6, 7, 11, 17, 20], dtype=torch.int32, device=DEVICE)
    num_logits = cu[-1].item()
    logits = torch.randn(num_logits, vocab + 5, generator=gen, device=DEVICE).to(dtype)
    logits = logits[:, :vocab]
    # Masks for requests 0, 2 and 4 at every scheduled position, one past the
    # request's logits (inactive) for request 2.
    keys = [r * MASK_STRIDE + p for r, n in ((0, 6), (2, 5), (4, 3)) for p in range(n)]
    if invalid:
        keys[4] = -keys[4] - 1
        keys[13] = -keys[13] - 1
    keys = torch.tensor(keys, dtype=torch.int32, device=DEVICE)
    words = -(-vocab // 32)
    bitmask = torch.randint(
        -(2**31), 2**31 - 1, (keys.shape[0], words), generator=gen, device=DEVICE
    ).int()
    bitmask[1] = -1  # nothing masked
    bitmask[2] = 0  # everything masked
    input_ids = torch.arange(100, 100 + num_logits, device=DEVICE).int()
    input_logits_indices = torch.randperm(num_logits, generator=gen, device=DEVICE)
    assert sum(counts) == num_logits
    return logits, keys, cu, input_ids, input_logits_indices, bitmask


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("vocab", [129280, 8193, 31])
@pytest.mark.parametrize("invalid", [False, True])
def test_bitmask_matches_triton(dtype, vocab, invalid):
    logits, keys, cu, input_ids, input_indices, bitmask = _case(
        vocab, 7, dtype, invalid
    )
    expected, expected_ids = logits.clone(), input_ids.clone()
    _apply_grammar_bitmask_kernel[(keys.shape[0], -(-vocab // 8192))](
        expected,
        expected.stride(0),
        keys,
        cu,
        expected_ids if invalid else None,
        input_indices if invalid else None,
        bitmask,
        bitmask.stride(0),
        vocab,
        MASK_STRIDE=MASK_STRIDE,
        BLOCK_SIZE=8192,
    )
    tilelang_structured.apply_grammar_bitmask(
        logits,
        keys,
        cu,
        input_ids if invalid else None,
        input_indices if invalid else None,
        bitmask,
        MASK_STRIDE,
    )
    assert torch.equal(
        logits.view(torch.int16 if dtype == torch.bfloat16 else torch.int32),
        expected.view(torch.int16 if dtype == torch.bfloat16 else torch.int32),
    )
    assert torch.equal(input_ids, expected_ids)
