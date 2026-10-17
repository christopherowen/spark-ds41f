# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Allocation-free greedy reduction of base logits and sequential Markov bias."""

from typing import Any

import torch

from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    TritonWarmupTensor,
    triton_kernel_dispatcher_with_warmup,
    triton_warmup_inputs,
)
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


def _partial_argmax_warmup_inputs(
    *,
    base_dtype: torch.dtype,
    base_stride: int,
    base_aligned: bool,
    bias_dtype: torch.dtype,
    bias_stride: int,
    vocab: int,
) -> dict[str, Any]:
    parts = triton.cdiv(vocab, _BLOCK)
    return triton_warmup_inputs(
        _partial_argmax,
        TritonWarmupTensor(base_dtype, aligned=base_aligned),
        TritonWarmupTensor(bias_dtype),
        TritonWarmupTensor(torch.float32),
        TritonWarmupTensor(torch.int32),
        base_stride,
        bias_stride,
        vocab,
        parts,
        _BLOCK,
        grid=(1,),
    )


def _partial_argmax_logits_warmup_inputs(
    *, dtype: torch.dtype, stride: int, vocab: int
) -> dict[str, Any]:
    return triton_warmup_inputs(
        _partial_argmax_logits,
        TritonWarmupTensor(dtype),
        TritonWarmupTensor(torch.float32),
        TritonWarmupTensor(torch.int32),
        stride,
        vocab,
        triton.cdiv(vocab, _BLOCK),
        _BLOCK,
        grid=(1,),
    )


def _finish_argmax_pair_warmup_inputs(
    *, vocab: int, vocab_start: int
) -> dict[str, Any]:
    parts = triton.cdiv(vocab, _BLOCK)
    return triton_warmup_inputs(
        _finish_argmax_pair,
        TritonWarmupTensor(torch.float32),
        TritonWarmupTensor(torch.int32),
        TritonWarmupTensor(torch.float32),
        vocab_start,
        parts,
        triton.next_power_of_2(parts),
        grid=(1,),
    )


def _select_first_argmax_warmup_inputs(
    *,
    ranks: int,
    output_dtype: torch.dtype,
    output_stride: int,
    output_aligned: bool,
) -> dict[str, Any]:
    return triton_warmup_inputs(
        _select_first_argmax,
        TritonWarmupTensor(torch.float32),
        TritonWarmupTensor(output_dtype, aligned=output_aligned),
        output_stride,
        ranks,
        triton.next_power_of_2(ranks),
        grid=(1,),
    )


_partial_argmax_launcher = triton_kernel_dispatcher_with_warmup(
    warmup_inputs=_partial_argmax_warmup_inputs
)(_partial_argmax)
_partial_argmax_logits_launcher = triton_kernel_dispatcher_with_warmup(
    warmup_inputs=_partial_argmax_logits_warmup_inputs
)(_partial_argmax_logits)
_finish_argmax_pair_launcher = triton_kernel_dispatcher_with_warmup(
    warmup_inputs=_finish_argmax_pair_warmup_inputs
)(_finish_argmax_pair)
_select_first_argmax_launcher = triton_kernel_dispatcher_with_warmup(
    warmup_inputs=_select_first_argmax_warmup_inputs
)(_select_first_argmax)


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
    """Register the kernels ``greedy_markov_shard_pair`` launches for one layout
    of its inputs: rows ``base_stride`` and ``bias_stride`` elements apart, the
    base logits' first element 16-byte aligned or not, ``vocab`` columns of the
    shard starting at global index ``vocab_start``; FP32/int32 scratch."""
    _partial_argmax_launcher.register_warmup(
        base_dtype=base_dtype,
        base_stride=base_stride,
        base_aligned=base_aligned,
        bias_dtype=bias_dtype,
        bias_stride=bias_stride,
        vocab=vocab,
    )
    _finish_argmax_pair_launcher.register_warmup(vocab=vocab, vocab_start=vocab_start)


def register_greedy_shard_pair_warmup(
    *, logits_dtype: torch.dtype, logits_stride: int, vocab: int, vocab_start: int
) -> None:
    """Register the kernels ``greedy_shard_pair`` launches for freshly allocated
    logits of ``logits_dtype`` whose rows are ``logits_stride`` elements apart."""
    _partial_argmax_logits_launcher.register_warmup(
        dtype=logits_dtype, stride=logits_stride, vocab=vocab
    )
    _finish_argmax_pair_launcher.register_warmup(vocab=vocab, vocab_start=vocab_start)


def register_select_first_argmax_warmup(
    *,
    ranks: int,
    output_dtype: torch.dtype,
    output_stride: int,
    output_aligned: bool,
) -> None:
    """Register ``select_first_argmax`` for ``ranks`` freshly gathered pairs per
    row and an output column of ``output_dtype``, ``output_stride`` elements per
    row, whose first element is 16-byte aligned or not."""
    _select_first_argmax_launcher.register_warmup(
        ranks=ranks,
        output_dtype=output_dtype,
        output_stride=output_stride,
        output_aligned=output_aligned,
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
    over one vocabulary shard into FP32 ``pairs`` ([rows, 2]).

    The value keeps the materialized sum's rounding, and the index is the
    shard-local first maximum offset by ``vocab_start`` (exact in FP32 below
    2**24). All-gathering the pairs across shards and passing them to
    ``select_first_argmax`` reproduces the full-vocabulary argmax, ties and
    NaNs included, because shards hold ascending index ranges.
    """
    if base_logits.shape != markov_bias.shape:
        raise ValueError(
            "DSpark greedy base logits and Markov bias must have identical shapes: "
            f"{tuple(base_logits.shape)} != {tuple(markov_bias.shape)}"
        )
    rows, vocab = base_logits.shape
    if vocab_start + vocab > 1 << 24:
        raise ValueError("vocabulary indices must stay exact in FP32")
    parts = triton.cdiv(vocab, _BLOCK)
    if min(partial_values.numel(), partial_indices.numel()) < rows * parts:
        raise ValueError("DSpark greedy scratch is smaller than the logits require")
    _partial_argmax_launcher[(rows, parts)](
        base_logits,
        markov_bias,
        partial_values,
        partial_indices,
        base_logits.stride(0),
        markov_bias.stride(0),
        vocab,
        parts,
        _BLOCK,
    )
    _finish_argmax_pair_launcher[(rows,)](
        partial_values,
        partial_indices,
        pairs,
        vocab_start,
        parts,
        triton.next_power_of_2(parts),
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
    ``logits`` (its padding columns excluded), into FP32 ``pairs`` ([rows, 2])."""
    rows = logits.shape[0]
    if vocab > logits.shape[1]:
        raise ValueError("the shard's real vocabulary exceeds its logits")
    if vocab_start + vocab > 1 << 24:
        raise ValueError("vocabulary indices must stay exact in FP32")
    parts = triton.cdiv(vocab, _BLOCK)
    if min(partial_values.numel(), partial_indices.numel()) < rows * parts:
        raise ValueError("greedy scratch is smaller than the logits require")
    _partial_argmax_logits_launcher[(rows, parts)](
        logits,
        partial_values,
        partial_indices,
        logits.stride(0),
        vocab,
        parts,
        _BLOCK,
    )
    _finish_argmax_pair_launcher[(rows,)](
        partial_values,
        partial_indices,
        pairs,
        vocab_start,
        parts,
        triton.next_power_of_2(parts),
    )


def select_first_argmax(pairs: torch.Tensor, output: torch.Tensor) -> None:
    """Reduce all-gathered shard pairs ([rows, 2 * ranks], FP32) to the first
    maximum's global index, written to ``output`` (one per row)."""
    rows, width = pairs.shape
    ranks = width // 2
    _select_first_argmax_launcher[(rows,)](
        pairs,
        output,
        output.stride(0),
        ranks,
        triton.next_power_of_2(ranks),
    )


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
    if base_logits.shape != markov_bias.shape:
        raise ValueError(
            "DSpark greedy base logits and Markov bias must have identical shapes: "
            f"{tuple(base_logits.shape)} != {tuple(markov_bias.shape)}"
        )
    rows, vocab = base_logits.shape
    parts = triton.cdiv(vocab, _BLOCK)
    if min(partial_values.numel(), partial_indices.numel()) < rows * parts:
        raise ValueError("DSpark greedy scratch is smaller than the logits require")
    _partial_argmax_launcher[(rows, parts)](
        base_logits,
        markov_bias,
        partial_values,
        partial_indices,
        base_logits.stride(0),
        markov_bias.stride(0),
        vocab,
        parts,
        _BLOCK,
    )
    _finish_argmax[(rows,)](
        partial_values,
        partial_indices,
        output,
        output.stride(0),
        parts,
        triton.next_power_of_2(parts),
    )
