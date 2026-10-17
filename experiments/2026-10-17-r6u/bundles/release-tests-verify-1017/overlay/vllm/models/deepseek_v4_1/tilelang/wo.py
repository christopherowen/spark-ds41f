# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TileLang output projection of the DeepSeek-V4.1 attention, as DeepSeek's
reference computes it::

    o[..., -64:] = inverse_rope(o[..., -64:])          (BF16, in place)
    o = act_quant(o, 32, "ue8m0", inplace=True)        (MXFP8 values, back in BF16)
    a = wo_a(o.view(T, groups, -1))                    (grouped, block-32 FP8, BF16 out)
    out = wo_b(a.flatten(1))                           (block-32 FP8, BF16 out)

WO-A quantizes its input as the reference does (since its 2026-10-12 revision, for
both V4.1 Flash checkpoints): each 32-value block of ``o`` to E4M3 with a UE8M0 scale
and back to BF16, then the BF16 einsum on the weights ``convert.py`` dequantizes. One
launch of the grouped W8A16 GEMM does both: it quantizes each activation tile as it
loads it, expands each FP8 weight tile exactly to BF16 and accumulates in FP32.
WO-B is the ordinary block-32 projection with MXFP8 activations (DeepSeek's E4M3 and
one UE8M0 scale per 32), as the reference's ``linear`` quantizes them. WO-A's last
kernel (the GEMM's epilogue, or the split-K reduce at decode) rounds ``a`` to BF16 and
applies that cast itself, so ``a`` never reaches memory in BF16: the bytes are those
of ``per_token_cast`` on the BF16 ``a``. The checkpoint tensors are read in place.
"""

import torch
from torch import nn

from vllm.v1.worker.workspace import current_preallocated_workspace

from .. import l2_prefetch
from . import prepare
from .gemm import (
    SPLIT_FP8,
    fp8w_bf16_gemm_cast,
    fp8w_bf16_gemm_partials,
    splitk_reduce_cast,
)
from .linear import (
    _activation_specs,
    _block32_gemm,
    _block32_scale_words,
    _calls,
    _declare_block32,
    _launch,
    _scratch,
)
from .rope import rope_

# WO-A's W8A16 GEMM sums K in _WO_A_SHARDS shards at every row count, each from zero,
# added in shard order, so a row's bits do not depend on the path. A bucket either
# splits the shards over CTAs (FP32 partials, then splitk_reduce: the SMs a narrow grid
# leaves idle at decode) or adds them inside the kernel (no partials to write and read
# back). Path and tile per bucket from the split sweeps (bits depend on neither): 64
# columns and two stages throughout; four shards beat two at decode, and tiles as tall
# as the bucket from 65 rows (2026-10-13 sweep 2: 72-96 rows 27.9-29.5 against 34.5 us,
# 512 rows 119.1 against 131.8 us, 8192 rows 1675 against 1758 us).
_WO_A_SHARDS = 4
_WO_A_TILES = {
    16: dict(split=True, block_M=16),
    32: dict(split=True, block_M=32),
    64: dict(split=True, block_M=64),
    96: dict(split=True, block_M=96),
    128: dict(split=True, block_M=128),
    None: dict(split=False, block_M=96),
}
_WO_A_COMMON = dict(block_N=64, num_stages=2)


class _Weight:
    """A block-32 FP8 weight and its scales; weak-referenceable for the custom op."""

    def __init__(self, layer: nn.Module, groups: int):
        self.weight = layer.weight.detach()
        self.weight_scale_inv = layer.weight_scale_inv.detach()
        self.groups = groups
        _block32_scale_words(self)

    def declare_kernels(self) -> None:
        """Declare the GEMMs; the caller hands them MXFP8 activations, so there is
        no per-token cast to prepare."""
        _declare_block32(self, self.groups, cast=False)
        # Verification steps stay on decode tiles to 128 rows: one 96-row tile to 96
        # (WIDE_DECODE), two of 64 rows past it.
        self.tilelang_decode_rows = 128


class _GroupedBF16:
    """WO-A for the W8A16 GEMM: the FP8 weight, its UE8M0 exponents (also what the L2
    prefetch reads beside it, as ``tilelang_weight_sf``) and one kernel per row tile."""

    def __init__(self, layer: nn.Module, groups: int):
        self.weight = layer.weight.detach()
        self.tilelang_weight_sf = layer.weight_scale_inv.detach().view(torch.uint8)
        self.columns = self.weight.shape[0]
        self.groups = groups

    def _kernels(self) -> dict:
        """Every kernel of WO-A, as calls: one per row bucket, and the split path's
        reduce."""
        n, k = self.weight.shape
        groups = self.groups
        return {
            "kernels": {
                bucket: prepare.call(
                    fp8w_bf16_gemm_partials,
                    n // groups,
                    k,
                    groups,
                    _WO_A_SHARDS,
                    block_M=tile["block_M"],
                    **_WO_A_COMMON,
                )
                if tile["split"]
                else prepare.call(
                    fp8w_bf16_gemm_cast,
                    n // groups,
                    k,
                    groups,
                    shards=_WO_A_SHARDS,
                    block_M=tile["block_M"],
                    **_WO_A_COMMON,
                )
                for bucket, tile in _WO_A_TILES.items()
            },
            "reduce": prepare.call(splitk_reduce_cast, n, _WO_A_SHARDS),
        }

    def declare_kernels(self) -> None:
        """Declare WO-A's kernels, bound as ``kernels`` and ``reduce``."""
        kernels = self._kernels()

        def bind() -> None:
            for name, kernel in _launch(kernels).items():
                setattr(self, name, kernel)

        prepare.declare(lambda: _calls(kernels), bind=bind)

    @staticmethod
    def bucket(rows: int):
        return next((b for b in _WO_A_TILES if b is not None and rows <= b), None)

    def partial_specs(self, rows: int):
        """The split path's FP32 partials (nothing for the in-kernel path)."""
        if not _WO_A_TILES[self.bucket(rows)]["split"]:
            return ()
        return (((_WO_A_SHARDS, rows, self.columns), torch.float32),)

    def __call__(self, x, xq, sf, partials=None) -> None:
        """WO-A of ``x``, cast as the reference's ``act_quant`` into ``xq`` (E4M3) and
        ``sf`` (UE8M0 bytes); ``partials`` from ``partial_specs`` on the split path."""
        bucket = self.bucket(x.shape[0])
        if not _WO_A_TILES[bucket]["split"]:
            self.kernels[bucket](x, self.weight, self.tilelang_weight_sf, xq, sf)
            return
        self.kernels[bucket](x, self.weight, self.tilelang_weight_sf, partials)
        self.reduce(partials, xq, sf)


class TileLangWOProjection:
    """Inverse RoPE, WO-A and WO-B of one attention layer (built after loading; its
    layer declares the kernels with ``declare_kernels``)."""

    def __init__(self, wo_a: nn.Module, wo_b: nn.Module, *, groups: int):
        self.a = _GroupedBF16(wo_a, groups)
        self.b = _Weight(wo_b, 1)
        if SPLIT_FP8.get(tuple(self.b.weight.shape)):
            raise ValueError("WO-B takes WO-A's MXFP8 output: no split-K path")
        self.hidden = wo_b.weight.shape[0]

    def declare_kernels(self) -> None:
        """Declare WO-A's and WO-B's kernels (the inverse RoPE is the layer's)."""
        self.a.declare_kernels()
        self.b.declare_kernels()

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
        if not rows:
            return out
        rope_(o, positions, cos_sin_cache, 1, True)
        scratch = (
            None if torch.compiler.is_compiling() else current_preallocated_workspace()
        )
        # One borrow for everything live at once: WO-A's partials, then WO-B's MXFP8
        # input in whole decode tiles.
        partial_specs = self.a.partial_specs(rows)
        views = _scratch(
            (
                *partial_specs,
                *_activation_specs(rows, self.a.columns, self.b.tilelang_decode_rows),
            ),
            scratch,
        )
        partials = views[0] if partial_specs else None
        xq, sf = views[-2], views[-1]
        self.a(o.view(rows, -1), xq[:rows], sf[:rows], partials)
        _block32_gemm(self.b, xq, sf, out, rows)
        return out


__all__ = ["TileLangWOProjection"]
