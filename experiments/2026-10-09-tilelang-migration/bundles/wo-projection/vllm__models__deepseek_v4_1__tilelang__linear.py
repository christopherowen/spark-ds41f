# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TileLang linear methods for DeepSeek-V4.1 (``--linear-backend tilelang``).

Block-32 FP8 projections quantize activations to MXFP8 with TileKernels'
per-token cast (DeepSeek's E4M3, one UE8M0 scale per 32 elements) and run
``mxfp8_gemm``; decode rows go through whole 16-, 32- or 64-row activation
tiles from the padded workspace. Unquantized projections run ``bf16_gemm``;
projections with few output tiles (the router) split K over the SMs for
decode rows and reduce the FP32 partials in a fixed order. Every path keeps
each row's accumulation order fixed, so results do not depend on the batch.

The kernels run inside custom ops, which keeps them opaque to torch.compile,
and borrow activation scratch from vLLM's workspace.
"""

import math
from weakref import WeakValueDictionary

import torch
from torch import nn

from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.models.deepseek_v4_1.quant_config import Block32FP8LinearMethod
from vllm.v1.worker.workspace import (
    current_preallocated_workspace,
    current_workspace_manager,
    retain_cuda_graph_capture_resource,
)

from .gemm import (
    DECODE_ROWS,
    DECODE_TILE_ROWS,
    SF_BLOCK,
    bf16_gemm,
    bf16_gemm_partials,
    bf16_shards,
    decode_tile_rows,
    fp8_decode_config,
    fp8_prefill_config,
    mxfp8_gemm,
    mxfp8_gemm_decode,
    pack_scale_words,
    scale_words,
    splitk_reduce,
)

_LAYERS: WeakValueDictionary[int, nn.Module] = WeakValueDictionary()
_TORCH_TO_TL = {torch.bfloat16: "bfloat16", torch.float32: "float32"}


def _tile_kernels():
    import tile_kernels

    return tile_kernels


def _activation_specs(rows: int, k: int):
    """MXFP8 activations and packed scales; decode rows fill one whole tile.

    TileKernels pads each row's packed UE8M0 exponents to whole words. Rows
    past ``rows`` in a decode tile are never stored by the GEMM.
    """
    tile_rows = DECODE_ROWS if rows <= DECODE_ROWS else rows
    return (
        ((tile_rows, k), torch.float8_e4m3fn),
        ((tile_rows, 4 * scale_words(k)), torch.uint8),
    )


# Row tiles of the split-K partials kernel. A 64-row tile computes one decode
# row in twice the time of a 16-row tile; every height gives the same
# partials bit for bit, so each row count takes the smallest tile that fits.
PARTIAL_ROW_TILES = (16, 32, 64)


def partials_kernel(layer: nn.Module, rows: int):
    """The layer's split-K partials kernel with the smallest row tile for ``rows``."""
    for block_M in PARTIAL_ROW_TILES:
        if rows <= block_M:
            return layer.tilelang_partials[block_M]
    return layer.tilelang_partials[PARTIAL_ROW_TILES[-1]]


def _partial_specs(layer: nn.Module, rows: int):
    """FP32 split-K partials of a BF16 projection's decode rows, if it splits."""
    shards = layer.tilelang_shards
    if shards == 1 or rows > DECODE_ROWS:
        return ()
    return (((shards, rows, layer.weight.shape[0]), torch.float32),)


def _spec_bytes(specs) -> int:
    total = 0
    for shape, dtype in specs:
        nbytes = dtype.itemsize
        for dim in shape:
            nbytes *= dim
        total += (nbytes + 255) // 256 * 256
    return total


def _scratch(specs, scratch: torch.Tensor | None):
    """Views of ``specs`` in the caller's scratch, else in vLLM's workspace."""
    if scratch is None:
        scratch = current_preallocated_workspace()
    if scratch is not None:
        return _carve(scratch.view(-1).view(torch.uint8), specs)
    views = current_workspace_manager().get_simultaneous(*specs)
    retain_cuda_graph_capture_resource(views[0])
    return views


def _carve(scratch: torch.Tensor, specs):
    """Views of ``specs`` laid out back to back, 256-byte aligned, in ``scratch``."""
    views, offset = [], 0
    for shape, dtype in specs:
        nbytes = dtype.itemsize
        for dim in shape:
            nbytes *= dim
        views.append(scratch[offset : offset + nbytes].view(dtype).view(shape))
        offset += (nbytes + 255) // 256 * 256
    return views


@torch.library.custom_op(
    "vllm::dsv41_tilelang_block32_linear", mutates_args=("out", "scratch")
)
def _block32_linear(
    x: torch.Tensor, out: torch.Tensor, key: int, scratch: torch.Tensor | None
) -> None:
    layer = _LAYERS[key]
    n, k = layer.weight.shape
    k *= layer.tilelang_groups  # a grouped weight reads every group's columns
    x = x.reshape(-1, k)
    rows = x.shape[0]
    xq, sf = _scratch(_activation_specs(rows, k), scratch)
    _tile_kernels().quant.per_token_cast(
        x,
        "e4m3",
        SF_BLOCK,
        round_sf=True,
        use_packed_ue8m0=True,
        out=(xq[:rows], sf[:rows]),
    )
    if rows > DECODE_ROWS:
        gemm = layer.tilelang_prefill
    else:
        gemm = layer.tilelang_decode[decode_tile_rows(rows)]
    gemm(
        xq,
        layer.weight,
        sf.view(torch.uint32),
        layer.tilelang_weight_sf,
        out.view(rows, n),
    )


@_block32_linear.register_fake
def _block32_linear_fake(x, out, key, scratch):
    return None


@torch.library.custom_op(
    "vllm::dsv41_tilelang_bf16_linear", mutates_args=("out", "scratch")
)
def _bf16_linear(
    x: torch.Tensor, out: torch.Tensor, key: int, scratch: torch.Tensor | None
) -> None:
    layer = _LAYERS[key]
    n, k = layer.weight.shape
    rows = x.numel() // k
    x, out = x.reshape(rows, k), out.view(rows, n)
    specs = _partial_specs(layer, rows)
    if specs:
        (partials,) = _scratch(specs, scratch)
        partials_kernel(layer, rows)(x, layer.weight, partials)
        layer.tilelang_reduce(partials, out)
    else:
        layer.tilelang_gemm(x, layer.weight, out)


@_bf16_linear.register_fake
def _bf16_linear_fake(x, out, key, scratch):
    return None


def _prepare_block32(layer, groups: int = 1) -> None:
    """Compile the GEMMs of ``layer.weight`` (E4M3 ``[N, K]``) and its 32x32 block
    scales ``layer.weight_scale_inv``, and register the layer for the custom op.

    With ``groups`` > 1 the weight stacks ``groups`` blocks of ``N / groups`` rows,
    each applied to its own K columns of ``groups * K``-wide activations.
    """
    n, k = layer.weight.shape
    part = n // groups
    # The 32x32 block scales become per-row words of four K32 exponents.
    layer.tilelang_weight_sf = pack_scale_words(
        layer.weight_scale_inv.view(torch.uint8), rows=n
    )
    layer.tilelang_groups = groups
    layer.tilelang_decode = {
        block_M: mxfp8_gemm_decode(
            part,
            k,
            **fp8_decode_config(part, k, block_M, groups),
            padded_rows=True,
            groups=groups,
        )
        for block_M in DECODE_TILE_ROWS
    }
    layer.tilelang_prefill = mxfp8_gemm(
        part, k, **fp8_prefill_config(part, k, groups), groups=groups
    )
    layer.tilelang_key = id(layer)
    _LAYERS[layer.tilelang_key] = layer


def _apply_block32(layer, x: torch.Tensor) -> torch.Tensor:
    out = torch.empty(
        (*x.shape[:-1], layer.weight.shape[0]),
        dtype=torch.bfloat16,
        device=x.device,
    )
    scratch = (
        None if torch.compiler.is_compiling() else current_preallocated_workspace()
    )
    _block32_linear(x, out, layer.tilelang_key, scratch)
    return out


class TileLangFP8LinearMethod(Block32FP8LinearMethod):
    """Checkpoint block-32 FP8 weights with MXFP8 activations."""

    def process_weights_after_loading(self, layer):
        _prepare_block32(layer)

    def get_workspace_size(self, layer, num_tokens: int) -> int:
        return _spec_bytes(_activation_specs(num_tokens, layer.weight.shape[1]))

    def apply(self, layer, x, bias=None):
        if bias is not None:
            raise ValueError("V4.1 block32 projections require bias-free weights")
        return _apply_block32(layer, x)


class TileLangBlock32Rows:
    """Output rows ``start:`` of a loaded block-32 FP8 projection as their own GEMM.

    The DSpark drafter projects the target's hidden states through only the KV
    rows of each layer's fused Q-A/KV weight. Views of the weight and its scales
    share the checkpoint storage; the owner must keep this object alive.
    """

    def __init__(self, layer: nn.Module, start: int):
        if start % 32:
            raise ValueError(f"rows from {start} do not start on a 32-row scale block")
        self.weight = layer.weight[start:]
        self.weight_scale_inv = layer.weight_scale_inv[start // 32 :]
        _prepare_block32(self)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return _apply_block32(self, x)


class TileLangLinearMethod(UnquantizedLinearMethod):
    """BF16 weights, BF16 or FP32 output (``layer.out_dtype``)."""

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        n, k = layer.weight.shape
        out_dtype = _TORCH_TO_TL[getattr(layer, "out_dtype", torch.bfloat16)]
        shards = layer.tilelang_shards = bf16_shards(n, k)
        layer.tilelang_gemm = bf16_gemm(n, k, out_dtype=out_dtype, shards=shards)
        if shards > 1:
            layer.tilelang_partials = {
                block_M: bf16_gemm_partials(n, k, shards, block_M=block_M)
                for block_M in PARTIAL_ROW_TILES
            }
            layer.tilelang_reduce = splitk_reduce(n, shards, out_dtype=out_dtype)
        layer.tilelang_key = id(layer)
        _LAYERS[layer.tilelang_key] = layer

    def apply(
        self, layer: nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        if bias is not None:
            raise ValueError("V4.1 native projections require bias-free weights")
        out = torch.empty(
            (*x.shape[:-1], layer.weight.shape[0]),
            dtype=getattr(layer, "out_dtype", torch.bfloat16),
            device=x.device,
        )
        scratch = (
            None if torch.compiler.is_compiling() else current_preallocated_workspace()
        )
        _bf16_linear(x, out, layer.tilelang_key, scratch)
        return out

    def get_workspace_size(self, layer, num_tokens: int) -> int:
        return _spec_bytes(_partial_specs(layer, num_tokens))


class _Part:
    """One row block of a split weight; weak-referenceable for ``_LAYERS``."""

    def __init__(self, weight: torch.Tensor, out_dtype: torch.dtype):
        self.weight = weight
        self.out_dtype = out_dtype


class TileLangScaledLinearMethod(TileLangLinearMethod):
    """``TileLangLinearMethod`` with a power-of-two output scale folded into the
    weight at load. Scaling by a power of two is exact in BF16 and commutes with
    the FP32 accumulation, so the output equals the unscaled projection's times
    ``scale``, bit for bit (checked on the weight), without a separate pass."""

    def __init__(self, scale: float):
        super().__init__()
        if scale <= 0 or math.frexp(scale)[0] != 0.5:
            raise ValueError(f"{scale} is not a power of two")
        self.scale = scale

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        weight = layer.weight.data
        scaled = weight * self.scale
        if not torch.equal(scaled / self.scale, weight):
            raise ValueError(f"{self.scale} does not scale this weight exactly")
        weight.copy_(scaled)
        super().process_weights_after_loading(layer)


class TileLangSplitLinearMethod(UnquantizedLinearMethod):
    """A BF16 weight made of equal row blocks (the compressor's wkv and wgate), each
    projected into its own output by the BF16 GEMM of ``TileLangLinearMethod``."""

    def __init__(self, parts: int):
        super().__init__()
        self.parts = parts

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        super().process_weights_after_loading(layer)
        n = layer.weight.shape[0]
        if n % self.parts:
            raise ValueError(f"{n} rows do not split into {self.parts} parts")
        rows = n // self.parts
        out_dtype = getattr(layer, "out_dtype", torch.bfloat16)
        layer.tilelang_parts = []
        for index in range(self.parts):
            part = _Part(
                layer.weight.data[index * rows : (index + 1) * rows], out_dtype
            )
            TileLangLinearMethod().process_weights_after_loading(part)
            layer.tilelang_parts.append(part)

    def get_workspace_size(self, layer, num_tokens: int) -> int:
        return max(
            _spec_bytes(_partial_specs(part, num_tokens))
            for part in layer.tilelang_parts
        )


def project_parts(layer: nn.Module, x: torch.Tensor, outputs, rows: int) -> None:
    """Project ``x`` (rows x K) through each row block of a split layer into the
    matching output's first ``rows`` rows."""
    scratch = (
        None if torch.compiler.is_compiling() else current_preallocated_workspace()
    )
    for part, out in zip(layer.tilelang_parts, outputs):
        _bf16_linear(x, out[:rows], part.tilelang_key, scratch)


__all__ = [
    "TileLangBlock32Rows",
    "TileLangFP8LinearMethod",
    "TileLangLinearMethod",
    "TileLangScaledLinearMethod",
    "TileLangSplitLinearMethod",
    "project_parts",
]
