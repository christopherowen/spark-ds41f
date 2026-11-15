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

from math import prod

import torch

from vllm.distributed import tensor_model_parallel_all_gather
from vllm.utils.math_utils import round_up
from vllm.v1.worker.gpu.spec_decode.dspark.greedy import (
    greedy_shard_pair,
    register_greedy_shard_pair_warmup,
    scratch_shape,
)
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    is_workspace_manager_initialized,
)

# Columns of a rank's pair row: 16 bytes, which the one-shot all-gather writes in
# place (the pair in the first two).
PAIR_WIDTH = 4


def _greedy_pair_buffers(
    rows: int, vocab: int, tp_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = scratch_shape(rows, vocab)
    pair_shape = (rows, PAIR_WIDTH)
    specs: list[tuple[tuple[int, ...], torch.dtype]] = [
        (shape, torch.float32),
        (shape, torch.int32),
    ]
    if tp_size > 1:
        specs.append((pair_shape, torch.float32))
    if is_workspace_manager_initialized():
        manager = current_workspace_manager()
        required_bytes = sum(round_up(prod(s) * d.itemsize, 256) for s, d in specs)
        if (
            manager._device.type == device.type
            and manager._device.index in (None, device.index)
            and (not manager.is_locked() or manager.available_bytes() >= required_bytes)
        ):
            buffers = manager.get_simultaneous(*specs)
            if tp_size > 1:
                return buffers[0], buffers[1], buffers[2]
            pairs = torch.empty(pair_shape, dtype=torch.float32, device=device)
            return buffers[0], buffers[1], pairs
    values = torch.empty(shape, dtype=torch.float32, device=device)
    indices = torch.empty(shape, dtype=torch.int32, device=device)
    pairs = torch.empty(pair_shape, dtype=torch.float32, device=device)
    return values, indices, pairs


def vocab_parallel_pairs(
    local_logits: torch.Tensor, vocab: int, vocab_start: int, tp_size: int
) -> torch.Tensor:
    """Every rank's ``(max, global index)`` pair of each row, all-gathered
    ([rows, PAIR_WIDTH * tp_size] FP32, in rank order), from this rank's shard of
    the logits (``vocab`` real columns starting at ``vocab_start``):
    ``greedy_rejection_sample`` selects each row's first maximum from them."""
    rows = local_logits.shape[0]
    values, indices, pairs = _greedy_pair_buffers(
        rows, vocab, tp_size, local_logits.device
    )
    greedy_shard_pair(local_logits, vocab, vocab_start, pairs, values, indices)
    if tp_size > 1:
        pairs = tensor_model_parallel_all_gather(pairs, dim=-1)
    return pairs


def vocab_parallel_pairs_from_head(
    head: torch.nn.Module,
    hidden_states: torch.Tensor,
    vocab: int,
    vocab_start: int,
    tp_size: int,
    *,
    cut_cols: torch.Tensor | None = None,
    top_k: int = 1,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Every rank's ``(max, global index)`` pair of each row, all-gathered
    ([rows, PAIR_WIDTH * tp_size] FP32, in rank order), projected and reduced
    directly from a prepared packed vocabulary head without materializing the
    shard's logits. When ``top_k > 1``, returns ``(pairs, row_topk)`` where
    ``row_topk`` holds this rank's top-``top_k`` candidates per row."""
    from vllm.models.deepseek_v4_1.tilelang.vocab import project_head_greedy_pairs

    out = project_head_greedy_pairs(
        head,
        hidden_states,
        vocab=vocab,
        vocab_start=vocab_start,
        pair_width=PAIR_WIDTH,
        scratch_pairs=tp_size > 1,
        cut_cols=cut_cols,
        top_k=top_k,
    )
    if top_k > 1:
        pairs, row_topk = out
    else:
        pairs = out
        row_topk = None
    if tp_size > 1:
        pairs = tensor_model_parallel_all_gather(pairs, dim=-1)
    if row_topk is not None:
        return pairs, row_topk
    return pairs


def register_greedy_verification_warmups(
    *,
    logits_dtype: torch.dtype,
    logits_stride: int,
    vocab: int,
    vocab_start: int,
    tp_size: int,
    num_speculative_steps: int,
    step_widths: tuple[int, ...] = (),
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

    widths = dict.fromkeys(step_widths or (num_speculative_steps,))
    tilelang_sampling.declare(
        lambda: tilelang_sampling.greedy_calls(
            rejection=[(steps + 1, tp_size, PAIR_WIDTH) for steps in widths]
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


_TRITON_REFS = frozenset(("_greedy_rejection_kernel",))


def __getattr__(name: str):
    if name in _TRITON_REFS:
        from vllm.v1.worker.gpu import _triton_refs

        return getattr(_triton_refs, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
