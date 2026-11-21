# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Allocation-free greedy reduction of base logits and sequential Markov bias."""

import torch

from vllm.utils.math_utils import cdiv

_BLOCK = 2048


def register_greedy_markov_shard_pair_warmup(
    *,
    base_dtype: torch.dtype,
    base_stride: int,
    base_aligned: bool,
    bias_dtype: torch.dtype,
    bias_stride: int,
    vocab: int,
    vocab_start: int,
    top2: bool = False,
) -> None:
    """Declare the kernels ``greedy_markov_shard_pair`` launches for base logits
    and a Markov bias of these dtypes, strides, alignment and shard width."""
    from vllm.v1.worker.gpu.sample import tilelang_sampling

    vector = (
        base_dtype.itemsize == 2
        and bias_dtype.itemsize == 2
        and base_aligned
        and (base_stride * 2) % 16 == 0
        and (bias_stride * 2) % 16 == 0
        and vocab % 8 == 0
    )
    tilelang_sampling.declare(
        lambda: tilelang_sampling.greedy_calls(
            markov=[(base_dtype, bias_dtype, vector)], top2=top2
        )
    )


def register_greedy_shard_pair_warmup(
    *, logits_dtype: torch.dtype, logits_stride: int, vocab: int, vocab_start: int
) -> None:
    """Declare the kernels ``greedy_shard_pair`` launches for logits of
    ``logits_dtype``, ``logits_stride`` and shard width ``vocab``."""
    from vllm.v1.worker.gpu.sample import tilelang_sampling

    vector = (
        logits_dtype.itemsize == 2
        and (logits_stride * 2) % 16 == 0
        and vocab % 8 == 0
    )
    tilelang_sampling.declare(
        lambda: tilelang_sampling.greedy_calls(logits=[(logits_dtype, vector)])
    )


def register_select_first_argmax_warmup(
    *,
    ranks: int,
    output_dtype: torch.dtype,
    output_stride: int,
    output_aligned: bool,
    pair_width: int = 4,
    top2: bool = False,
) -> None:
    """Declare ``select_first_argmax`` for an output column of ``output_dtype``
    over pairs of ``pair_width`` columns (the runner's 16-byte rows)."""
    from vllm.v1.worker.gpu.sample import tilelang_sampling

    tilelang_sampling.declare(
        lambda: tilelang_sampling.greedy_calls(
            select=[(output_dtype, ranks, pair_width)], top2=top2
        )
    )


def register_sample_greedy_markov_warmup(
    *, dtype: torch.dtype, output_dtype: torch.dtype
) -> None:
    """Declare the kernels ``sample_greedy_markov`` launches for base logits and
    a Markov bias of ``dtype`` and an output column of ``output_dtype``."""
    from vllm.v1.worker.gpu.sample import tilelang_sampling

    tilelang_sampling.declare(
        lambda: tilelang_sampling.greedy_calls(
            markov=[(dtype, dtype)], finish_outputs=[output_dtype]
        )
    )


def greedy_markov_shard_pair(
    base_logits: torch.Tensor,
    markov_bias: torch.Tensor,
    vocab_start: int,
    pairs: torch.Tensor,
    partial_values: torch.Tensor,
    partial_indices: torch.Tensor,
    *,
    top2: bool = False,
) -> None:
    """Write each row's ``(max, global index)`` of ``base_logits + markov_bias``
    over one vocabulary shard into the first two columns of FP32 ``pairs`` ([rows,
    2], or [rows, 4]: 16-byte rows, which the one-shot all-gather writes in place;
    with ``top2=True``, columns 2 and 3 receive the shard's runner-up value and
    global index).

    The value keeps the materialized sum's rounding, and the index is the
    shard-local first maximum offset by ``vocab_start`` (exact in FP32 below
    2**24). All-gathering the pairs across shards and passing them to
    ``select_first_argmax`` reproduces the full-vocabulary argmax, ties and
    NaNs included, because shards hold ascending index ranges.
    """
    from vllm.v1.worker.gpu.sample import tilelang_sampling

    tilelang_sampling.greedy_markov_shard_pair(
        base_logits,
        markov_bias,
        vocab_start,
        pairs,
        partial_values,
        partial_indices,
        top2=top2,
    )


def greedy_shard_pair(
    logits: torch.Tensor,
    vocab: int,
    vocab_start: int,
    pairs: torch.Tensor,
    partial_values: torch.Tensor,
    partial_indices: torch.Tensor,
) -> None:
    """``greedy_markov_shard_pair`` without a bias: each row's ``(max, global
    index)`` over the first ``vocab`` columns of one vocabulary shard of
    ``logits`` (its padding columns excluded), into the first two columns of FP32
    ``pairs`` ([rows, 2] or [rows, 4])."""
    from vllm.v1.worker.gpu.sample import tilelang_sampling

    tilelang_sampling.greedy_shard_pair(
        logits, vocab, vocab_start, pairs, partial_values, partial_indices
    )


def select_first_argmax(
    pairs: torch.Tensor,
    output: torch.Tensor,
    pair_width: int = 2,
    embed: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    top2_branch: (
        tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, float, float] | None
    ) = None,
) -> None:
    """Reduce all-gathered shard pairs ([rows, pair_width * ranks], FP32, each
    rank's pair first in its ``pair_width`` columns) to the first maximum's global
    index, written to ``output`` (one per row). With ``embed`` (a BF16 Markov W1
    table and two row outputs), also the selected tokens' rows of the table. With
    ``top2_branch``, also reduces the global runner-up and promotes it when
    Markov transition confidence outweighs a narrow top1-top2 logit gap."""
    from vllm.v1.worker.gpu.sample import tilelang_sampling

    tilelang_sampling.select_first_argmax(
        pairs, output, pair_width, embed, top2_branch=top2_branch
    )


def markov_anchor_embed(
    ids: torch.Tensor,
    index: torch.Tensor,
    table: torch.Tensor,
    step_out: torch.Tensor,
    confidence_out: torch.Tensor,
) -> None:
    """The rows of the BF16 Markov W1 ``table`` for the tokens ``ids[index]``, into
    ``step_out`` and ``confidence_out``."""
    from vllm.v1.worker.gpu.sample import tilelang_sampling

    tilelang_sampling.markov_anchor_embed(ids, index, table, step_out, confidence_out)


def scratch_shape(max_requests: int, vocabulary_size: int, *, top2: bool = False) -> tuple[int, int]:
    return max_requests, cdiv(vocabulary_size, _BLOCK) * (2 if top2 else 1)


def sample_greedy_markov(
    base_logits: torch.Tensor,
    markov_bias: torch.Tensor,
    output: torch.Tensor,
    partial_values: torch.Tensor,
    partial_indices: torch.Tensor,
) -> None:
    """Write exactly ``(base_logits + markov_bias).argmax(-1)`` into output.

    Both inputs share FP32, FP16, or BF16 dtype and have contiguous vocabulary
    columns. Request strides may differ. The caller owns disjoint FP32/int32
    scratch of ``scratch_shape(capacity, vocabulary_size)`` and int32/int64
    output, which may be a strided speculative-token column. Ties, NaNs, and
    addition rounding follow PyTorch argmax. No proposal truncation is used.

    Args:
        base_logits: Per-request full-vocabulary base logits.
        markov_bias: Transition bias with the same shape and dtype as base_logits.
        output: Caller-owned column receiving one token ID per request.
        partial_values: Contiguous FP32 scratch for the partial maxima.
        partial_indices: Contiguous int32 scratch for the partial token indices.

    Raises:
        ValueError: Input shapes differ or scratch cannot hold every partial.
    """
    from vllm.v1.worker.gpu.sample import tilelang_sampling

    tilelang_sampling.sample_greedy_markov(
        base_logits, markov_bias, output, partial_values, partial_indices
    )


_TRITON_REFS = frozenset(
    (
        "_first_argmax",
        "_partial_argmax",
        "_partial_argmax_logits",
        "_finish_argmax",
        "_finish_argmax_pair",
        "_select_first_argmax",
    )
)


def __getattr__(name: str):
    if name in _TRITON_REFS:
        from vllm.v1.worker.gpu import _triton_refs

        return getattr(_triton_refs, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
