# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample import tilelang_controls


# The reference the TileLang min-p (``tilelang_controls.min_p_kernel``) is tested
# against; the sampler no longer launches it.
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


def register_min_p_warmup(*, logits_stride: int, vocab_size: int) -> None:
    """Declare ``apply_min_p`` for processed FP32 logits, rows ``logits_stride``
    apart, and FP32 thresholds."""
    tilelang_controls.register_min_p(logits_stride=logits_stride, vocab_size=vocab_size)


def apply_min_p(
    logits: torch.Tensor, expanded_idx_mapping: torch.Tensor, min_p: torch.Tensor
) -> None:
    tilelang_controls.apply_min_p(logits, expanded_idx_mapping, min_p)
