# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TileLang output projection of the DeepSeek-V4.1 attention, as DeepSeek's
reference computes it::

    o[..., -64:] = inverse_rope(o[..., -64:])          (BF16, in place)
    a = wo_a(o.view(T, groups, -1))                    (grouped, block-32 FP8, BF16 out)
    out = wo_b(a.flatten(1))                           (block-32 FP8, BF16 out)

with MXFP8 activations (DeepSeek's E4M3 and one UE8M0 scale per 32) for both GEMMs.
WO-A runs every group in one launch of the grouped MXFP8 GEMM; WO-B is the
ordinary block-32 projection. The checkpoint tensors are read in place.
"""

import torch
from torch import nn

from vllm.v1.worker.workspace import current_preallocated_workspace

from .. import l2_prefetch
from .linear import _block32_linear, _prepare_block32
from .rope import rope_


class _Weight:
    """A block-32 FP8 weight and its scales; weak-referenceable for the custom op."""

    def __init__(self, layer: nn.Module, groups: int):
        self.weight = layer.weight.detach()
        self.weight_scale_inv = layer.weight_scale_inv.detach()
        _prepare_block32(self, groups)
        # 16-stream steps (up to 96 rows) stay on decode tiles, two of 64 rows.
        self.tilelang_decode_rows = 128


class TileLangWOProjection:
    def __init__(self, wo_a: nn.Module, wo_b: nn.Module, *, groups: int):
        self.a = _Weight(wo_a, groups)
        self.b = _Weight(wo_b, 1)
        self.hidden = wo_b.weight.shape[0]

    def l2_segments(self) -> list:
        """The bytes both GEMMs read, WO-A first: each weight and its scale words."""
        return l2_prefetch.linear_segments(
            "wo_a", self.a
        ) + l2_prefetch.linear_segments("wo_b", self.b)

    def __call__(
        self,
        o: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        out: torch.Tensor,
    ) -> torch.Tensor:
        """Project ``o`` ([T, heads, head_dim], contiguous BF16; its RoPE columns
        are overwritten) into ``out`` ([T, hidden] BF16)."""
        rows = o.shape[0]
        rope_(o, positions, cos_sin_cache, 1, True)
        scratch = (
            None if torch.compiler.is_compiling() else current_preallocated_workspace()
        )
        a = torch.empty(
            (rows, self.a.weight.shape[0]), dtype=torch.bfloat16, device=o.device
        )
        _block32_linear(o.view(rows, -1), a, self.a.tilelang_key, scratch)
        _block32_linear(a, out, self.b.tilelang_key, scratch)
        return out


__all__ = ["TileLangWOProjection"]
