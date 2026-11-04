# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import MethodType, SimpleNamespace
from typing import get_args

import numpy as np
import pytest
import torch

from vllm.config.model import PROCESSED_LOGPROBS_MODES, LogprobsMode
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import (
    RejectionSampler,
    _iter_request_chunks,
)


def test_iter_request_chunks_preserves_request_boundaries():
    cu_num_logits = np.array([0, 3, 4, 11, 13], dtype=np.int32)

    assert list(_iter_request_chunks(cu_num_logits, max_chunk_logits=5)) == [
        (0, 2),
        (2, 3),
        (3, 4),
    ]


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize("logprobs_mode", get_args(LogprobsMode))
@pytest.mark.parametrize("custom_ids", [False, True])
def test_chunked_scores_match_full_batch(logprobs_mode: str, custom_ids: bool):
    device = torch.device("cuda")
    cu_num_logits_np = np.array([0, 3, 4, 8, 10], dtype=np.int32)
    num_logits_per_req = np.diff(cu_num_logits_np)
    idx_mapping_np = np.array([7, 2, 9, 1], dtype=np.int32)
    input_batch = SimpleNamespace(
        num_reqs=4,
        num_draft_tokens_per_req=num_logits_per_req - 1,
        cu_num_logits_np=cu_num_logits_np,
        cu_num_logits=torch.from_numpy(cu_num_logits_np).to(device),
        idx_mapping_np=idx_mapping_np,
        idx_mapping=torch.from_numpy(idx_mapping_np).to(device),
        expanded_idx_mapping=torch.from_numpy(
            np.repeat(idx_mapping_np, num_logits_per_req)
        ).to(device),
        expanded_local_pos=torch.from_numpy(
            np.concatenate(
                [np.arange(count, dtype=np.int32) for count in num_logits_per_req]
            )
        ).to(device),
    )
    rejection_sampler = object.__new__(RejectionSampler)
    id_state = None
    if custom_ids:
        counts = torch.zeros(10, device=device, dtype=torch.int32)
        counts[7], counts[9] = 1, 3
        tokens = torch.zeros(10, 3, device=device, dtype=torch.int32)
        tokens[7, 0] = 16
        tokens[9] = torch.tensor([3, 11, 5], device=device)
        id_state = SimpleNamespace(
            num_token_ids=SimpleNamespace(gpu=counts),
            token_ids=SimpleNamespace(gpu=tokens),
        )
    rejection_sampler.sampler = SimpleNamespace(
        logprobs_mode=logprobs_mode, logprob_token_ids_state=id_state
    )
    dims = (2, 3 if custom_ids else 0)
    rejection_sampler.num_speculative_steps = 3
    rejection_sampler.num_draft_steps = 3
    rejection_sampler.enable_adaptive_verification = False
    chunk_offsets: list[torch.Tensor] = []

    def fake_verify(
        self,
        logits,
        _draft_logits,
        _draft_sampled,
        _pos,
        cu_num_logits,
        idx_mapping,
        *_mappings,
    ):
        chunk_offsets.append(cu_num_logits)
        num_sampled = torch.diff(cu_num_logits).to(torch.int32)
        sampled = (
            idx_mapping.to(torch.int64).unsqueeze(1) + torch.arange(4, device=device)
        ) % logits.shape[1]
        return logits.float() + 1, sampled, num_sampled

    rejection_sampler._verify = MethodType(fake_verify, rejection_sampler)
    logits = torch.arange(170, dtype=torch.float32, device=device).view(10, 17)

    sampled, num_sampled, chunked_logprobs = rejection_sampler._verify_in_chunks(
        logits,
        input_batch,
        draft_logits=None,
        draft_sampled=torch.arange(10, device=device),
        pos=torch.arange(10, device=device),
        max_chunk_logits=5,
        logprobs_dims=dims,
    )
    score_logits = logits + 1 if logprobs_mode in PROCESSED_LOGPROBS_MODES else logits
    full_logprobs = rejection_sampler._get_logprobs_tensors(
        sampled,
        num_sampled,
        score_logits,
        input_batch.cu_num_logits,
        input_batch.cu_num_logits_np,
        input_batch.expanded_idx_mapping,
        logprobs_dims=dims,
    )

    # Chunks (0, 2), (2, 3), (3, 4): the first chunk's offsets are a view of the
    # batch's (no launch), the later ones the host's uploaded, from 0.
    assert [c.tolist() for c in chunk_offsets] == [[0, 3, 4], [0, 4], [0, 2]]
    assert all(c.dtype == torch.int32 and c.is_cuda for c in chunk_offsets)
    first = chunk_offsets[0]
    assert first.data_ptr() == input_batch.cu_num_logits.data_ptr()
    assert sampled[:, 0].tolist() == idx_mapping_np.tolist()
    assert num_sampled.tolist() == num_logits_per_req.tolist()
    assert chunked_logprobs is not None
    assert full_logprobs is not None
    assert torch.equal(
        chunked_logprobs.logprob_token_ids,
        full_logprobs.logprob_token_ids,
    )
    assert torch.equal(chunked_logprobs.logprobs, full_logprobs.logprobs)
    if logprobs_mode in ("raw_logits", "processed_logits"):
        expected = score_logits.gather(1, full_logprobs.logprob_token_ids)
        if custom_ids:
            widths = torch.tensor([1, 2, 3, 2], device=device).repeat_interleave(
                torch.tensor(num_logits_per_req, device=device)
            )
            valid = torch.arange(4, device=device)[None, :] <= widths[:, None]
            expected.masked_fill_(~valid, -torch.inf)
        assert torch.equal(full_logprobs.logprobs, expected)
    assert torch.equal(
        chunked_logprobs.selected_token_ranks,
        full_logprobs.selected_token_ranks,
    )
    assert (
        chunked_logprobs.cu_num_generated_tokens
        == full_logprobs.cu_num_generated_tokens
    )
