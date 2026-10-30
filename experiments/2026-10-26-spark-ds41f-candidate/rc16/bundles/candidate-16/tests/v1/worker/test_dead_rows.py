# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch

from vllm.v1.worker.gpu.input_batch import (
    DeadRows,
    combine_sampled_and_draft_tokens,
    get_num_sampled_and_rejected,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _combine(dead_rows):
    dev = "cuda"
    # Two decode requests (slots 3 and 1) with 5 drafts each, one prefill.
    idx_mapping = torch.tensor([3, 1, 0], dtype=torch.int32, device=dev)
    query_start_loc = torch.tensor([0, 6, 12, 20], dtype=torch.int32, device=dev)
    seq_lens = torch.tensor([106, 206, 8], dtype=torch.int32, device=dev)
    prefill_len = torch.tensor([50, 100, 0, 100], dtype=torch.int32, device=dev)
    last_sampled = torch.tensor([10, 11, 12, 13], dtype=torch.int32, device=dev)
    draft_tokens = torch.arange(20, dtype=torch.int32, device=dev).view(4, 5) + 100
    cu_num_logits = torch.tensor([0, 6, 12, 13], dtype=torch.int32, device=dev)
    input_ids = torch.zeros(24, dtype=torch.int32, device=dev)
    combine_sampled_and_draft_tokens(
        input_ids, idx_mapping, last_sampled, query_start_loc, seq_lens,
        prefill_len, draft_tokens, cu_num_logits, 13, 1, dead_rows=dead_rows,
    )
    return input_ids


def test_dead_rows_mark_rows_past_the_survival_cut():
    dev = "cuda"
    confidence = torch.ones(4, 5, dtype=torch.float32, device=dev)
    # Slot 3: survival 0.9, 0.72, 0.36, ... -> cut 2 at tau 0.5.
    confidence[3] = torch.tensor([0.9, 0.8, 0.5, 0.9, 0.9])
    # Slot 1: first draft already below tau -> cut 0.
    confidence[1] = torch.tensor([0.4, 0.9, 0.9, 0.9, 0.9])
    is_padding = torch.zeros(24, dtype=torch.bool, device=dev)
    cut = torch.empty(3, dtype=torch.int32, device=dev)
    ids_dead = _combine(DeadRows(confidence, is_padding, cut, math.log(0.5)))
    ids_plain = _combine(None)
    assert torch.equal(ids_dead, ids_plain)
    assert cut.tolist() == [2, 0, 9]  # the prefill request is never clamped
    expected = torch.zeros(24, dtype=torch.bool)
    expected[3:6] = True    # request 0: rows of drafts 3-5
    expected[7:12] = True   # request 1: rows of drafts 1-5
    assert torch.equal(is_padding.cpu(), expected)


def test_dead_row_cut_clamps_committed_tokens():
    dev = "cuda"
    num_sampled = torch.tensor([6, 1, 4], dtype=torch.int32, device=dev)
    seq_lens = torch.tensor([106, 206, 306], dtype=torch.int32, device=dev)
    cu_num_logits = torch.tensor([0, 6, 12, 18], dtype=torch.int32, device=dev)
    idx_mapping = torch.tensor([0, 1, 2], dtype=torch.int32, device=dev)
    prefill_len = torch.tensor([10, 10, 10], dtype=torch.int32, device=dev)
    cut = torch.tensor([2, 0, 6], dtype=torch.int32, device=dev)
    sampled, rejected = get_num_sampled_and_rejected(
        num_sampled, seq_lens, cu_num_logits, idx_mapping, prefill_len,
        dead_row_cut=cut,
    )
    assert sampled.tolist() == [3, 1, 4]
    assert rejected.tolist() == [3, 5, 2]


def test_dead_rows_continuation_value_preserves_bridge_draft():
    dev = "cuda"
    confidence = torch.ones(4, 5, dtype=torch.float32, device=dev)
    # Slot 3: survival [0.9, 0.72, 0.36, 0.324, 0.2916]. At tau=0.5, myopic cut is 2,
    # whereas continuation_alpha=0.25 lifts draft 2 to 0.36 + 0.25*(0.324+0.2916) =
    # 0.5139 >= 0.5, preserving the bridge draft (cut=3).
    confidence[3] = torch.tensor([0.9, 0.8, 0.5, 0.9, 0.9], device=dev)
    # Slot 1: same prefix up to draft 2, but dead tail (0.1, 0.1 -> 0.036, 0.0036),
    # so draft 2 scores 0.36 + 0.25*(0.036+0.0036) = 0.3699 < 0.5 and stays cut=2.
    confidence[1] = torch.tensor([0.9, 0.8, 0.5, 0.1, 0.1], device=dev)
    is_padding = torch.zeros(24, dtype=torch.bool, device=dev)
    cut = torch.empty(3, dtype=torch.int32, device=dev)
    _combine(DeadRows(confidence, is_padding, cut, math.log(0.5), 0.25))
    assert cut.tolist() == [3, 2, 9]
    expected = torch.zeros(24, dtype=torch.bool)
    expected[4:6] = True    # request 0 (slot 3): drafts 4-5 padded, draft 3 kept
    expected[9:12] = True   # request 1 (slot 1): drafts 3-5 padded
    assert torch.equal(is_padding.cpu(), expected)
