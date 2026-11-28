# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from spark_ds41.sampling import tilelang_controls


def register_min_p_warmup(*, logits_stride: int, vocab_size: int) -> None:
    """Declare ``apply_min_p`` for processed FP32 logits, rows ``logits_stride``
    apart, and FP32 thresholds."""
    tilelang_controls.register_min_p(logits_stride=logits_stride, vocab_size=vocab_size)


def apply_min_p(
    logits: torch.Tensor, expanded_idx_mapping: torch.Tensor, min_p: torch.Tensor
) -> None:
    tilelang_controls.apply_min_p(logits, expanded_idx_mapping, min_p)


_TRITON_REFS = frozenset(("_min_p_kernel",))


def __getattr__(name: str):
    if name in _TRITON_REFS:
        from spark_ds41.sampling import _triton_refs

        return getattr(_triton_refs, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
