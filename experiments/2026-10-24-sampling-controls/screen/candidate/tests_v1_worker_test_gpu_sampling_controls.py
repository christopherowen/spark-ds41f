# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-slot, draft-prefix and vocabulary-boundary contracts for V2 controls."""

import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA required for sampling controls", allow_module_level=True)

from vllm.v1.worker.gpu.sample import tilelang_controls as candidate
from vllm.v1.worker.gpu.sample.min_p import _min_p_kernel
from vllm.v1.worker.gpu.sample.penalties import _bincount_kernel, _penalties_kernel


def assert_bits(actual, expected):
    if actual.is_floating_point():
        # NaN payloads are not part of the logits-processor contract.
        assert torch.equal(torch.isnan(actual), torch.isnan(expected))
        valid = ~torch.isnan(expected)
        assert torch.equal(
            actual[valid].view(torch.int32), expected[valid].view(torch.int32)
        )
    else:
        assert torch.equal(actual, expected)


def min_p_case(rows, vocab, index_dtype, *, padded=False, nonfinite=False):
    torch.manual_seed(18093)
    logits = torch.randn((rows, vocab + (19 if padded else 0)), device="cuda")
    logits = logits[:, :vocab]
    mapping = torch.randperm(rows + 7, device="cuda")[:rows].to(index_dtype)
    thresholds = torch.tensor(
        [0.0, 0.01, 0.1, 0.7, 1.0] * ((rows + 11) // 5), device="cuda"
    )[: rows + 7]
    if nonfinite:
        logits[0].fill_(-float("inf"))
        if rows > 1:
            logits[1, 0] = float("nan")
        if rows > 2:
            logits[2, -1] = float("inf")
    return logits, mapping, thresholds


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("vocab", [1, 33, 1023, 1025, 129280])
@pytest.mark.parametrize("nonfinite", [False, True])
def test_min_p_matches_triton_at_vocabulary_boundaries(index_dtype, vocab, nonfinite):
    logits, mapping, thresholds = min_p_case(
        6, vocab, index_dtype, padded=True, nonfinite=nonfinite
    )
    expected = torch.empty_strided(logits.shape, logits.stride(), device="cuda")
    expected.copy_(logits)
    _min_p_kernel[(logits.shape[0],)](
        expected, expected.stride(0), mapping, thresholds, vocab, BLOCK_SIZE=1024
    )
    candidate.apply_min_p(logits, mapping, thresholds)
    assert_bits(logits, expected)


def penalties_case(rows, vocab, index_dtype, *, padded=False):
    torch.manual_seed(31037)
    states = rows + 7
    stride = vocab + (17 if padded else 0)
    logits = torch.randn((rows, stride), device="cuda")[:, :vocab]
    mapping = (torch.arange(rows, device="cuda") // 6 + 2).to(index_dtype)
    positions = (torch.arange(rows, device="cuda") % 6).int()
    tokens = torch.randint(vocab, (rows,), dtype=torch.int32, device="cuda")
    rep = torch.tensor([1.0, 0.5, 1.3, 2.0] * ((states + 3) // 4), device="cuda")[
        :states
    ]
    freq = torch.tensor([0.0, 0.25, -0.3] * ((states + 2) // 3), device="cuda")[:states]
    pres = torch.tensor([0.0, -0.5, 0.75] * ((states + 2) // 3), device="cuda")[:states]
    mask = torch.randint(
        -(2**31),
        2**31 - 1,
        (states, (vocab + 31) // 32),
        dtype=torch.int32,
        device="cuda",
    )
    counts = torch.randint(0, 8, (states, vocab), dtype=torch.int32, device="cuda")
    # One completely disabled request alongside requests with every penalty enabled.
    rep[2], freq[2], pres[2] = 1, 0, 0
    return logits, mapping, tokens, positions, rep, freq, pres, mask, counts


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("vocab", [33, 8191, 8193, 129280])
@pytest.mark.parametrize("nonfinite", [False, True])
def test_penalties_count_only_preceding_drafts(index_dtype, vocab, nonfinite):
    args = penalties_case(17, vocab, index_dtype, padded=True)
    logits, mapping, tokens, pos, rep, freq, pres, mask, counts = args
    if nonfinite:
        logits[:, :8] = torch.tensor(
            [
                -float("inf"),
                float("inf"),
                float("nan"),
                0.0,
                -0.0,
                1.17549435e-38,
                1.40129846e-45,
                -1.40129846e-45,
            ],
            device="cuda",
        )
    expected = torch.empty_strided(logits.shape, logits.stride(), device="cuda")
    expected.copy_(logits)
    _penalties_kernel[(17, (vocab + 8191) // 8192)](
        expected,
        expected.stride(0),
        mapping,
        tokens,
        pos,
        rep,
        freq,
        pres,
        mask,
        mask.stride(0),
        counts,
        counts.stride(0),
        vocab,
        BLOCK_SIZE=8192,
    )
    candidate.apply_penalties(*args)
    assert_bits(logits, expected)


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("length", [1, 1023, 1025, 8192])
def test_bincount_distinguishes_prompt_bits_and_output_counts(index_dtype, length):
    torch.manual_seed(29401)
    states, rows, vocab = 9, 4, 129280
    tokens = torch.randint(
        vocab, (states, length + 11), dtype=torch.int32, device="cuda"
    )
    tokens[:, ::3] = 31
    mapping = torch.tensor([8, 1, 5, 2], dtype=index_dtype, device="cuda")
    prompt = torch.full((states,), length // 2, dtype=torch.int32, device="cuda")
    prefill = torch.full((states,), length, dtype=torch.int32, device="cuda")
    mask = torch.zeros((states, (vocab + 31) // 32), dtype=torch.int32, device="cuda")
    counts = torch.zeros((states, vocab), dtype=torch.int32, device="cuda")
    expected_mask, expected_counts = mask.clone(), counts.clone()
    _bincount_kernel[(rows, (length + 1023) // 1024)](
        mapping,
        tokens,
        tokens.stride(0),
        prompt,
        prefill,
        expected_mask,
        mask.stride(0),
        expected_counts,
        counts.stride(0),
        BLOCK_SIZE=1024,
    )
    candidate.bincount(mapping, tokens, prompt, prefill, mask, counts, length)
    assert_bits(mask, expected_mask)
    assert_bits(counts, expected_counts)
