# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TileLang linear methods for DeepSeek-V4.1 (``--linear-backend tilelang``).

Block-32 FP8 projections quantize activations to MXFP8 with TileKernels'
per-token cast (DeepSeek's E4M3, one UE8M0 scale per 32 elements) and run
``mxfp8_gemm``; decode rows go through whole 64-row activation tiles from the
padded workspace. Unquantized projections run ``bf16_gemm``; projections with
few output tiles (the router) split K over the SMs for decode rows and reduce
the FP32 partials in a fixed order. Every path keeps each row's accumulation
order fixed, so results do not depend on the batch.

The kernels run inside custom ops, which keeps them opaque to torch.compile,
and borrow activation scratch from vLLM's workspace.
"""

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
    SF_BLOCK,
    bf16_gemm,
    bf16_gemm_partials,
    bf16_shards,
    default_config,
    fp8_decode_config,
    mxfp8_gemm,
    mxfp8_gemm_partials,
    pack_scale_words,
    scale_words,
    splitk_reduce,
)

_LAYERS: WeakValueDictionary[int, nn.Module] = WeakValueDictionary()
_TORCH_TO_TL = {torch.bfloat16: "bfloat16", torch.float32: "float32"}


def _tile_kernels():
    import tile_kernels

    return tile_kernels


def _activation_specs(rows: int, k: int, n: int = 0, shards: int = 1):
    """MXFP8 activations and packed scales; decode rows fill one whole tile.

    TileKernels pads each row's packed UE8M0 exponents to whole words. Rows
    past ``rows`` in a decode tile are never stored by the GEMM. A split-K
    decode route adds its FP32 shard partials.
    """
    tile_rows = DECODE_ROWS if rows <= DECODE_ROWS else rows
    specs = (
        ((tile_rows, k), torch.float8_e4m3fn),
        ((tile_rows, 4 * scale_words(k)), torch.uint8),
    )
    if shards > 1 and rows <= DECODE_ROWS:
        specs += (((shards, rows, n), torch.float32),)
    return specs


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
    x = x.reshape(-1, k)
    rows = x.shape[0]
    shards = layer.tilelang_shards
    xq, sf, *partials = _scratch(_activation_specs(rows, k, n, shards), scratch)
    _tile_kernels().quant.per_token_cast(
        x,
        "e4m3",
        SF_BLOCK,
        round_sf=True,
        use_packed_ue8m0=True,
        out=(xq[:rows], sf[:rows]),
    )
    args = (xq, layer.weight, sf.view(torch.uint32), layer.tilelang_weight_sf)
    if rows > DECODE_ROWS:
        layer.tilelang_gemms[1](*args, out.view(rows, n))
    elif partials:
        layer.tilelang_gemms[0](*args, partials[0])
        layer.tilelang_reduce(partials[0], out.view(rows, n))
    else:
        layer.tilelang_gemms[0](*args, out.view(rows, n))


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


class TileLangFP8LinearMethod(Block32FP8LinearMethod):
    """Checkpoint block-32 FP8 weights with MXFP8 activations."""

    def process_weights_after_loading(self, layer):
        n, k = layer.weight.shape
        # The 32x32 block scales become per-row words of four K32 exponents.
        layer.tilelang_weight_sf = pack_scale_words(
            layer.weight_scale_inv.view(torch.uint8), rows=n
        )
        # The decode configuration fixes the K shards; prefill folds the same.
        config = fp8_decode_config(n, k)
        shards = layer.tilelang_shards = config.pop("shards")
        tile = dict(block_M=DECODE_ROWS, **config)
        decode = (
            mxfp8_gemm_partials(n, k, shards, **tile)
            if shards > 1
            else mxfp8_gemm(n, k, **tile, padded_rows=True)
        )
        layer.tilelang_gemms = (
            decode,
            mxfp8_gemm(n, k, **default_config(DECODE_ROWS + 1, k), shards=shards),
        )
        if shards > 1:
            layer.tilelang_reduce = splitk_reduce(n, shards)
        layer.tilelang_key = id(layer)
        _LAYERS[layer.tilelang_key] = layer

    def get_workspace_size(self, layer, num_tokens: int) -> int:
        n, k = layer.weight.shape
        shards = fp8_decode_config(n, k)["shards"]
        return _spec_bytes(_activation_specs(num_tokens, k, n, shards))

    def apply(self, layer, x, bias=None):
        if bias is not None:
            raise ValueError("V4.1 block32 projections require bias-free weights")
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


__all__ = ["TileLangFP8LinearMethod", "TileLangLinearMethod"]
