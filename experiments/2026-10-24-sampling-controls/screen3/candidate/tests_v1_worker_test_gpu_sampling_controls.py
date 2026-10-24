# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-slot, draft-prefix and vocabulary-boundary contracts for V2 controls."""

import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA required for sampling controls", allow_module_level=True)

from vllm.v1.worker.gpu.sample import tilelang_auxiliary as auxiliary
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


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("word_count", [1, 7, 33, 129])
def test_bad_words_match_across_committed_and_draft_boundary(index_dtype, word_count):
    from vllm.v1.worker.gpu.sample.bad_words import _bad_words_kernel

    rows, states, vocab = 18, 11, 129280
    logits = torch.randn((rows, vocab + 13), device="cuda")[:, :vocab]
    mapping = (torch.arange(rows, device="cuda") // 6 + 2).to(index_dtype)
    pos = (torch.arange(rows, device="cuda") % 6).int()
    inputs = (torch.arange(rows, device="cuda") % 6 + 50).int()
    tokens = torch.full((states, 50), 50, dtype=torch.int32, device="cuda")
    prompt = torch.full((states,), 10, dtype=torch.int32, device="cuda")
    total = torch.full_like(prompt, 16)
    words = torch.full((states, word_count * 3), 50, dtype=torch.int32, device="cuda")
    words[:, 2::3] = torch.arange(word_count, device="cuda") + 80
    offsets = (torch.arange(word_count + 1, device="cuda") * 3).int().repeat(states, 1)
    num_words = torch.full_like(prompt, word_count)
    num_words[3] = 0
    expected = logits.clone()
    _bad_words_kernel[(rows, word_count)](
        expected,
        expected.stride(0),
        mapping,
        words,
        words.stride(0),
        offsets,
        offsets.stride(0),
        num_words,
        tokens,
        tokens.stride(0),
        prompt,
        total,
        inputs,
        pos,
    )
    auxiliary.apply_bad_words(
        logits,
        mapping,
        words,
        offsets,
        num_words,
        tokens,
        prompt,
        total,
        inputs,
        pos,
        word_count,
    )
    assert_bits(logits, expected)


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("length", [0, 1, 1023, 1025, 8192])
def test_prompt_ids_shift_each_chunk_to_its_next_token(index_dtype, length):
    from vllm.v1.worker.gpu.sample.prompt_logprob import (
        _prompt_logprobs_token_ids_kernel,
    )

    states = 9
    mapping = torch.tensor([8, 2, 4], device="cuda", dtype=index_dtype)
    starts = torch.tensor(
        [0, length, length + 3, length + 10], device="cuda", dtype=torch.int32
    )
    computed = torch.arange(states, device="cuda", dtype=torch.int32) * 3
    tokens = torch.randint(
        129280, (states, length + 100), device="cuda", dtype=torch.int32
    )
    output = torch.full((length + 10,), -7, device="cuda", dtype=torch.int64)
    expected = output.clone()
    _prompt_logprobs_token_ids_kernel[(3,)](
        expected, starts, mapping, computed, tokens, tokens.stride(0), BLOCK_SIZE=1024
    )
    auxiliary.prompt_token_ids(output, starts, mapping, computed, tokens)
    assert_bits(output, expected)


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("topk", [0, 1, 7, 1024])
def test_custom_logprob_ids_override_topk_without_touching_padding(index_dtype, topk):
    from vllm.v1.worker.gpu.sample.logprob import _fill_logprob_token_ids_kernel

    states, rows = 11, 6
    cols = max(topk, 17)
    output = torch.full((rows, cols + 1), -19, device="cuda", dtype=torch.int64)
    valid = torch.zeros_like(output, dtype=torch.bool)
    mapping = torch.tensor([10, 2, 4, 7, 5, 8], device="cuda", dtype=index_dtype)
    sampled = torch.randint(129280, (rows,), device="cuda", dtype=torch.int64)
    top_ids = torch.randint(
        129280, (rows, max(topk, 1)), device="cuda", dtype=torch.int32
    )
    count = torch.zeros(states, device="cuda", dtype=torch.int32)
    count[mapping[::2]] = 17
    custom = torch.randint(129280, (states, 17), device="cuda", dtype=torch.int32)
    expected, expected_valid = output.clone(), valid.clone()
    _fill_logprob_token_ids_kernel[(rows,)](
        expected,
        expected.stride(0),
        expected_valid,
        valid.stride(0),
        sampled,
        top_ids,
        top_ids.stride(0),
        mapping,
        count,
        custom,
        custom.stride(0),
        NUM_TOPK=topk,
        PADDED_COLS=1 << (cols - 1).bit_length(),
    )
    auxiliary.fill_token_ids(
        output, valid, sampled, top_ids, mapping, count, custom, topk
    )
    assert_bits(output, expected)
    assert_bits(valid, expected_valid)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("vocab", [33, 8193, 129280])
def test_nan_count_ignores_padded_columns(dtype, vocab):
    from vllm.v1.worker.gpu.metrics.logits import _num_nans_kernel

    logits = torch.randn((6, vocab + 17), device="cuda", dtype=dtype)
    logits[:, vocab:] = float("nan")
    logits[1::2, ::19] = float("nan")
    logits = logits[:, :vocab]
    expected = torch.empty(6, device="cuda", dtype=torch.int32)
    _num_nans_kernel[(6,)](logits, logits.stride(0), expected, vocab, BLOCK_SIZE=8192)
    output = auxiliary.count_nans(logits)
    assert_bits(output, expected)
    assert torch.equal(output, torch.isnan(logits).sum(-1))


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("vocab", [128, 8193, 129280])
@pytest.mark.parametrize("restore", [False, True])
def test_bias_preserves_allowed_logits_and_applies_bias_before_stop_mask(
    index_dtype, vocab, restore
):
    from vllm.v1.worker.gpu.sample import tilelang_bias
    from vllm.v1.worker.gpu.sample.logit_bias import _bias_kernel

    torch.manual_seed(81419)
    states, rows, width = 11, 6, 32
    logits = torch.randn((rows, vocab + 13), device="cuda")[:, :vocab]
    mapping = torch.tensor([8, 2, 4, 7, 1, 5], device="cuda", dtype=index_dtype)
    positions = torch.arange(rows, device="cuda", dtype=torch.int64) + 3
    num_allowed = torch.zeros(states, device="cuda", dtype=torch.int32)
    num_allowed[mapping[::2]] = width
    allowed = torch.arange(width, device="cuda", dtype=torch.int32).repeat(states, 1)
    num_bias = torch.full_like(num_allowed, width)
    bias_ids = allowed.clone()
    biases = torch.randn((states, width), device="cuda")
    min_lens = torch.full_like(num_allowed, 5)
    num_stops = torch.full_like(num_allowed, 3)
    restore_rows = torch.ones_like(num_allowed)
    stops = torch.tensor([0, 1, 2], device="cuda", dtype=torch.int32).repeat(states, 1)
    expected = torch.empty_strided(logits.shape, logits.stride(), device="cuda")
    expected.copy_(logits)
    _bias_kernel[(rows,)](
        expected,
        expected.stride(0),
        vocab,
        mapping,
        num_allowed,
        allowed,
        allowed.stride(0),
        num_bias,
        bias_ids,
        bias_ids.stride(0),
        biases,
        biases.stride(0),
        positions,
        min_lens,
        num_stops,
        restore_rows,
        stops,
        stops.stride(0),
        BLOCK_SIZE=width,
        LOGITS_BLOCK_SIZE=8192,
        CHECK_ALL_MASKED_ROWS=restore,
    )
    tilelang_bias.apply_logit_bias(
        logits,
        mapping,
        positions,
        num_allowed,
        allowed,
        num_bias,
        bias_ids,
        biases,
        min_lens,
        num_stops,
        restore_rows,
        stops,
        restore,
    )
    assert_bits(logits, expected)
