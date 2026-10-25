# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Greedy verification without gathering the target's logits.

When every request in a step is greedy and nothing reads or changes its logits
(no logprobs, penalties, logit bias, bad words, thinking budget, grammar, top-k,
top-p or min-p), verification needs only the target's argmax at each verified
position. Each tensor-parallel rank reduces its vocabulary shard to one
``(max, global index)`` pair per row; the pairs are all-gathered (16-byte rows per
rank instead of the shard's logits) and, in the verification's own launch, reduced
to the first maximum, which is ``torch.argmax`` of the full row, ties and NaNs
included. The acceptance rule is ``rejection_sample``'s greedy branch: a draft is
accepted while it equals the target's argmax; the first mismatch commits the
argmax, and a fully accepted block commits the argmax at the bonus position.
"""

import torch

from vllm.distributed import tensor_model_parallel_all_gather
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.spec_decode.dspark.greedy import (
    greedy_shard_pair,
    register_greedy_shard_pair_warmup,
    scratch_shape,
)

# Columns of a rank's pair row: 16 bytes, which the one-shot all-gather writes in
# place (the pair in the first two).
PAIR_WIDTH = 4


def vocab_parallel_pairs(
    local_logits: torch.Tensor, vocab: int, vocab_start: int, tp_size: int
) -> torch.Tensor:
    """Every rank's ``(max, global index)`` pair of each row, all-gathered
    ([rows, PAIR_WIDTH * tp_size] FP32, in rank order), from this rank's shard of
    the logits (``vocab`` real columns starting at ``vocab_start``):
    ``greedy_rejection_sample`` selects each row's first maximum from them."""
    rows = local_logits.shape[0]
    shape = scratch_shape(rows, vocab)
    values = torch.empty(shape, dtype=torch.float32, device=local_logits.device)
    indices = torch.empty(shape, dtype=torch.int32, device=local_logits.device)
    pairs = torch.empty(
        (rows, PAIR_WIDTH), dtype=torch.float32, device=local_logits.device
    )
    greedy_shard_pair(local_logits, vocab, vocab_start, pairs, values, indices)
    if tp_size > 1:
        pairs = tensor_model_parallel_all_gather(pairs, dim=-1)
    return pairs


# Reference for ``tilelang_sampling`` (after dspark.greedy._select_first_argmax);
# the runner no longer launches it.
@triton.jit
def _greedy_rejection_kernel(
    sampled_ptr,
    sampled_stride,
    num_sampled_ptr,
    target_ids_ptr,
    draft_sampled_ptr,
    cu_num_logits_ptr,
):
    req_idx = tl.program_id(0)
    start_idx = tl.load(cu_num_logits_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1).to(tl.int64)
    num_draft_tokens = end_idx - start_idx - 1
    accepted_length = tl.zeros((), tl.int64)
    verifying = True
    for i in range(num_draft_tokens):
        if verifying:
            draft = tl.load(draft_sampled_ptr + start_idx + i + 1).to(tl.int64)
            target = tl.load(target_ids_ptr + start_idx + i)
            # -1 is a placeholder draft that is always rejected.
            accepted = (target == draft) & (draft >= 0)
            tl.store(
                sampled_ptr + req_idx * sampled_stride + i,
                tl.where(accepted, draft, target),
            )
            verifying = accepted
            accepted_length += accepted.to(tl.int64)
    if accepted_length == num_draft_tokens:
        # Every draft accepted: the bonus position's argmax.
        tl.store(
            sampled_ptr + req_idx * sampled_stride + num_draft_tokens,
            tl.load(target_ids_ptr + start_idx + num_draft_tokens),
        )
    tl.store(num_sampled_ptr + req_idx, (accepted_length + 1).to(tl.int32))


def register_greedy_verification_warmups(
    *,
    logits_dtype: torch.dtype,
    logits_stride: int,
    vocab: int,
    vocab_start: int,
    tp_size: int,
    num_speculative_steps: int,
) -> None:
    """Declare the kernels ``vocab_parallel_pairs`` and ``greedy_rejection_sample``
    launch: this rank's shard of ``compute_logits_local`` (``logits_dtype``, rows
    ``logits_stride`` elements apart, ``vocab`` real columns from global index
    ``vocab_start``), the pairs of ``tp_size`` ranks, drafts and logits offsets from
    the input batch."""
    register_greedy_shard_pair_warmup(
        logits_dtype=logits_dtype,
        logits_stride=logits_stride,
        vocab=vocab,
        vocab_start=vocab_start,
    )
    from vllm.v1.worker.gpu.sample import tilelang_sampling

    tilelang_sampling.declare(
        lambda: tilelang_sampling.greedy_calls(
            rejection=[(num_speculative_steps + 1, tp_size, PAIR_WIDTH)]
        )
    )


def greedy_rejection_sample(
    pairs: torch.Tensor,
    draft_sampled: torch.Tensor,
    cu_num_logits: torch.Tensor,
    num_speculative_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``rejection_sample`` for an all-greedy batch from the gathered shard pairs
    (``vocab_parallel_pairs``): each row's target is the first maximum over its
    ranks' pairs, selected in the verification's launch.

    Returns ``sampled`` ([num_reqs, num_speculative_steps + 1], int64) and
    ``num_sampled`` ([num_reqs], int32), the committed prefix of each row.
    """
    from vllm.v1.worker.gpu.sample import tilelang_sampling

    return tilelang_sampling.greedy_rejection_sample(
        pairs, draft_sampled, cu_num_logits, num_speculative_steps, PAIR_WIDTH
    )
