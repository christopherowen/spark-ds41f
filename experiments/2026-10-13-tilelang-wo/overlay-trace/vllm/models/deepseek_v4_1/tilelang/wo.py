# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TileLang output projection of the DeepSeek-V4.1 attention, as DeepSeek's
reference computes it::

    o[..., -64:] = inverse_rope(o[..., -64:])          (BF16, in place)
    a = wo_a(o.view(T, groups, -1))                    (grouped, block-32 FP8, BF16 out)
    out = wo_b(a.flatten(1))                           (block-32 FP8, BF16 out)

WO-A keeps the BF16 activations, as the reference does (``convert.py`` dequantizes
``wo_a`` to BF16 and the projection is a BF16 einsum): one launch of the grouped
W8A16 GEMM expands each FP8 weight tile exactly to BF16 and accumulates in FP32.
WO-B is the ordinary block-32 projection with MXFP8 activations (DeepSeek's E4M3 and
one UE8M0 scale per 32), as the reference's ``linear`` quantizes them. The
checkpoint tensors are read in place.
"""

import torch
from torch import nn

from vllm.v1.worker.workspace import current_preallocated_workspace

from .. import l2_prefetch
from .gemm import fp8w_bf16_gemm, fp8w_bf16_gemm_partials, splitk_reduce
from .linear import _block32_linear, _prepare_block32, _scratch
from .rope import rope_

# WO-A's W8A16 GEMM sums K in _WO_A_SHARDS shards at every row count, each from zero,
# added in shard order, so a row's bits do not depend on the path. A bucket either
# splits the shards over CTAs (FP32 partials, then splitk_reduce: the SMs a narrow grid
# leaves idle at decode) or adds them inside the kernel (no partials to write and read
# back). Path and tile per bucket from the split sweep (bits depend on neither): 64
# columns and two stages throughout; four shards beat two at decode.
_WO_A_SHARDS = 4
_WO_A_TILES = {
    16: dict(split=True, block_M=16),
    32: dict(split=True, block_M=32),
    64: dict(split=True, block_M=64),
    96: dict(split=False, block_M=32),
    128: dict(split=True, block_M=64),
    None: dict(split=False, block_M=64),
}
_WO_A_COMMON = dict(block_N=64, num_stages=2)


class _Weight:
    """A block-32 FP8 weight and its scales; weak-referenceable for the custom op."""

    def __init__(self, layer: nn.Module, groups: int):
        self.weight = layer.weight.detach()
        self.weight_scale_inv = layer.weight_scale_inv.detach()
        _prepare_block32(self, groups)
        # 16-stream steps (up to 96 rows) stay on decode tiles, two of 64 rows.
        self.tilelang_decode_rows = 128


class _GroupedBF16:
    """WO-A for the W8A16 GEMM: the FP8 weight, its UE8M0 exponents (also what the L2
    prefetch reads beside it, as ``tilelang_weight_sf``) and one kernel per row tile."""

    def __init__(self, layer: nn.Module, groups: int):
        self.weight = layer.weight.detach()
        self.tilelang_weight_sf = layer.weight_scale_inv.detach().view(torch.uint8)
        n, k = self.weight.shape
        self.columns = n
        self.kernels = {
            bucket: fp8w_bf16_gemm_partials(
                n // groups,
                k,
                groups,
                _WO_A_SHARDS,
                block_M=tile["block_M"],
                **_WO_A_COMMON,
            )
            if tile["split"]
            else fp8w_bf16_gemm(
                n // groups,
                k,
                groups,
                shards=_WO_A_SHARDS,
                block_M=tile["block_M"],
                **_WO_A_COMMON,
            )
            for bucket, tile in _WO_A_TILES.items()
        }
        self.reduce = splitk_reduce(n, _WO_A_SHARDS)

    def __call__(self, x: torch.Tensor, out: torch.Tensor) -> None:
        rows = x.shape[0]
        bucket = next((b for b in _WO_A_TILES if b is not None and rows <= b), None)
        if not _WO_A_TILES[bucket]["split"]:
            self.kernels[bucket](x, self.weight, self.tilelang_weight_sf, out)
            return
        (partials,) = _scratch(
            (((_WO_A_SHARDS, rows, self.columns), torch.float32),), None
        )
        self.kernels[bucket](x, self.weight, self.tilelang_weight_sf, partials)
        self.reduce(partials, out)


class TileLangWOProjection:
    def __init__(self, wo_a: nn.Module, wo_b: nn.Module, *, groups: int):
        self.a = _GroupedBF16(wo_a, groups)
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
        if rows:
            self.a(o.view(rows, -1), a)
        _block32_linear(a, out, self.b.tilelang_key, scratch)
        return out


__all__ = ["TileLangWOProjection"]
