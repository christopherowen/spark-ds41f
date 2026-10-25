# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Allocation-free greedy reduction of base logits and sequential Markov bias."""

import torch

from vllm.triton_utils import tl, triton

_BLOCK = 2048


@triton.jit
def _first_argmax(av, ai, bv, bi):
    a_nan = av != av
    b_nan = bv != bv
    take_a = (a_nan & (~b_nan | (ai < bi))) | (
        ~a_nan & ~b_nan & ((av > bv) | ((av == bv) & (ai < bi)))
    )
    return tl.where(take_a, av, bv), tl.where(take_a, ai, bi)


# Reference for ``tilelang_sampling``; the runner no longer launches it.
@triton.jit
def _partial_argmax(
    base,
    bias,
    values,
    indices,
    base_stride,
    bias_stride,
    VOCAB: tl.constexpr,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    part = tl.program_id(1)
    col = part * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(base + row * base_stride + col, col < VOCAB, other=0)
    b = tl.load(bias + row * bias_stride + col, col < VOCAB, other=0)
    # Preserve the materialized torch.add rounding before comparing logits.
    summed = (
        (a.to(tl.float32) + b.to(tl.float32)).to(base.dtype.element_ty).to(tl.float32)
    )
    summed = tl.where(col < VOCAB, summed, -float("inf"))
    idx = tl.where(col < VOCAB, col, 2147483647)
    value, index = tl.reduce((summed, idx), 0, _first_argmax)
    tl.store(values + row * PARTS + part, value)
    tl.store(indices + row * PARTS + part, index)


# Reference for ``tilelang_sampling``; the runner no longer launches it.
@triton.jit
def _partial_argmax_logits(
    logits,
    values,
    indices,
    logits_stride,
    VOCAB: tl.constexpr,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    part = tl.program_id(1)
    col = part * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(logits + row * logits_stride + col, col < VOCAB, other=0)
    value = tl.where(col < VOCAB, value.to(tl.float32), -float("inf"))
    idx = tl.where(col < VOCAB, col, 2147483647)
    best, index = tl.reduce((value, idx), 0, _first_argmax)
    tl.store(values + row * PARTS + part, best)
    tl.store(indices + row * PARTS + part, index)


# Reference for ``tilelang_sampling``; the runner no longer launches it.
@triton.jit
def _finish_argmax(
    values,
    indices,
    output,
    output_stride,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    part = tl.arange(0, BLOCK)
    value = tl.load(values + row * PARTS + part, part < PARTS, other=-float("inf"))
    index = tl.load(indices + row * PARTS + part, part < PARTS, other=2147483647)
    _, result = tl.reduce((value, index), 0, _first_argmax)
    tl.store(output + row * output_stride, result)


# Reference for ``tilelang_sampling``; the runner no longer launches it.
@triton.jit
def _finish_argmax_pair(
    values,
    indices,
    pairs,
    vocab_start,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    part = tl.arange(0, BLOCK)
    value = tl.load(values + row * PARTS + part, part < PARTS, other=-float("inf"))
    index = tl.load(indices + row * PARTS + part, part < PARTS, other=2147483647)
    best, result = tl.reduce((value, index), 0, _first_argmax)
    tl.store(pairs + row * 2, best)
    tl.store(pairs + row * 2 + 1, (result + vocab_start).to(tl.float32))


# Reference for ``tilelang_sampling``; the runner no longer launches it.
@triton.jit
def _select_first_argmax(
    pairs,
    output,
    output_stride,
    RANKS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    rank = tl.arange(0, BLOCK)
    value = tl.load(
        pairs + row * 2 * RANKS + 2 * rank, rank < RANKS, other=-float("inf")
    )
    index = tl.load(
        pairs + row * 2 * RANKS + 2 * rank + 1, rank < RANKS, other=1073741824.0
    )
    _, result = tl.reduce((value, index.to(tl.int32)), 0, _first_argmax)
    tl.store(output + row * output_stride, result)


def register_greedy_markov_shard_pair_warmup(
    *,
    base_dtype: torch.dtype,
    base_stride: int,
    base_aligned: bool,
    bias_dtype: torch.dtype,
    bias_stride: int,
    vocab: int,
    vocab_start: int,
) -> None:
    """Declare the kernels ``greedy_markov_shard_pair`` launches for base logits
    and a Markov bias of these dtypes (the TileLang kernels take any stride,
    alignment, shard and offset)."""
    from vllm.v1.worker.gpu.sample import tilelang_sampling

    tilelang_sampling.declare(
        lambda: tilelang_sampling.greedy_calls(markov=[(base_dtype, bias_dtype)])
    )


def register_greedy_shard_pair_warmup(
    *, logits_dtype: torch.dtype, logits_stride: int, vocab: int, vocab_start: int
) -> None:
    """Declare the kernels ``greedy_shard_pair`` launches for logits of
    ``logits_dtype``."""
    from vllm.v1.worker.gpu.sample import tilelang_sampling

    tilelang_sampling.declare(
        lambda: tilelang_sampling.greedy_calls(logits=[logits_dtype])
    )


def register_select_first_argmax_warmup(
    *,
    ranks: int,
    output_dtype: torch.dtype,
    output_stride: int,
    output_aligned: bool,
    pair_width: int = 4,
) -> None:
    """Declare ``select_first_argmax`` for an output column of ``output_dtype``
    over pairs of ``pair_width`` columns (the runner's 16-byte rows)."""
    from vllm.v1.worker.gpu.sample import tilelang_sampling

    tilelang_sampling.declare(
        lambda: tilelang_sampling.greedy_calls(
            select=[(output_dtype, ranks, pair_width)]
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
) -> None:
    """Write each row's ``(max, global index)`` of ``base_logits + markov_bias``
    over one vocabulary shard into the first two columns of FP32 ``pairs`` ([rows,
    2], or [rows, 4]: 16-byte rows, which the one-shot all-gather writes in place).

    The value keeps the materialized sum's rounding, and the index is the
    shard-local first maximum offset by ``vocab_start`` (exact in FP32 below
    2**24). All-gathering the pairs across shards and passing them to
    ``select_first_argmax`` reproduces the full-vocabulary argmax, ties and
    NaNs included, because shards hold ascending index ranges.
    """
    from vllm.v1.worker.gpu.sample import tilelang_sampling

    tilelang_sampling.greedy_markov_shard_pair(
        base_logits, markov_bias, vocab_start, pairs, partial_values, partial_indices
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
) -> None:
    """Reduce all-gathered shard pairs ([rows, pair_width * ranks], FP32, each
    rank's pair first in its ``pair_width`` columns) to the first maximum's global
    index, written to ``output`` (one per row). With ``embed`` (a BF16 Markov W1
    table and two row outputs), also the selected tokens' rows of the table."""
    from vllm.v1.worker.gpu.sample import tilelang_sampling

    tilelang_sampling.select_first_argmax(pairs, output, pair_width, embed)


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


def scratch_shape(max_requests: int, vocabulary_size: int) -> tuple[int, int]:
    return max_requests, triton.cdiv(vocabulary_size, _BLOCK)


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
