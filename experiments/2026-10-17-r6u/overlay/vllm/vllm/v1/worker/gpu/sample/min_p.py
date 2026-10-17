# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch

from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    TritonWarmupTensor,
    triton_kernel_dispatcher_with_warmup,
    triton_warmup_inputs,
)
from vllm.triton_utils import tl, triton


@triton.jit
def _min_p_kernel(
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    min_p_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    min_p = tl.load(min_p_ptr + req_state_idx).to(tl.float32)
    if min_p == 0.0:
        return

    max_val = float("-inf")
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < vocab_size
        logits = tl.load(
            logits_ptr + token_idx * logits_stride + block,
            mask=mask,
            other=float("-inf"),
        )
        max_val = tl.max(tl.maximum(logits, max_val))
    max_val = max_val.to(tl.float32)  # type: ignore

    threshold = max_val + tl.log(min_p)
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < vocab_size
        logits = tl.load(
            logits_ptr + token_idx * logits_stride + block,
            mask=mask,
            other=float("-inf"),
        )
        logits = tl.where(logits < threshold, float("-inf"), logits)
        tl.store(logits_ptr + token_idx * logits_stride + block, logits, mask=mask)


def _min_p_warmup_inputs(*, logits_stride: int, vocab_size: int) -> dict[str, Any]:
    return triton_warmup_inputs(
        _min_p_kernel,
        TritonWarmupTensor(torch.float32),
        logits_stride,
        TritonWarmupTensor(torch.int64),
        TritonWarmupTensor(torch.float32),
        vocab_size,
        grid=(1,),
        BLOCK_SIZE=1024,
    )


_min_p_launcher = triton_kernel_dispatcher_with_warmup(
    warmup_inputs=_min_p_warmup_inputs
)(_min_p_kernel)


def register_min_p_warmup(*, logits_stride: int, vocab_size: int) -> None:
    """Register ``apply_min_p`` for processed FP32 logits, rows ``logits_stride``
    apart, int64 request indices and FP32 thresholds."""
    _min_p_launcher.register_warmup(logits_stride=logits_stride, vocab_size=vocab_size)


def apply_min_p(
    logits: torch.Tensor, expanded_idx_mapping: torch.Tensor, min_p: torch.Tensor
) -> None:
    num_tokens, vocab_size = logits.shape
    BLOCK_SIZE = 1024
    _min_p_launcher[(num_tokens,)](
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        min_p,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
