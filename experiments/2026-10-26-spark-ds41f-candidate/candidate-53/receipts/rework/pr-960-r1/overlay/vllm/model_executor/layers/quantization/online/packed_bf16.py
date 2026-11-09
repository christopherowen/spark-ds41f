# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vocabulary heads stored in B12X's exact 12-bit packed BF16 form.

The checkpoint's BF16 head loads unchanged; after loading it is packed, in
its own storage, into one sign-and-mantissa byte plus a 4-bit exponent code
per value, with the rare out-of-window values kept exactly in a per-row
exception list. Packing
verifies that every bit round-trips and refuses otherwise, so the served head
is the checkpoint tensor in 75% of the bytes, not an approximation.

The packed head runs only through the logits processor's vocabulary
projection, which reads the packed bytes directly: B12X's prepared projection,
or TileLang's with the TileLang linear backend.
"""

import importlib.util

import torch
from torch.nn import Parameter

from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.utils import replace_parameter, set_weight_attrs
from vllm.platforms import current_platform
from vllm.utils.b12x import get_b12x_packed_bf16_vocab_projection

logger = init_logger(__name__)


def _tilelang_packed_bf16_supported() -> bool:
    return (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(120)
        and importlib.util.find_spec("tilelang") is not None
    )


def packed_bf16_supported(input_size: int, params_dtype: torch.dtype) -> bool:
    if params_dtype != torch.bfloat16 or input_size <= 0 or input_size % 128 != 0:
        return False
    kernel_config = get_current_vllm_config().kernel_config
    if kernel_config.linear_backend == "tilelang":
        return _tilelang_packed_bf16_supported()
    projection = get_b12x_packed_bf16_vocab_projection()
    return projection is not None and projection.is_supported()


class PackedBf16LmHeadMethod(QuantizeMethodBase):
    """A BF16 LM head served from its exact 12-bit packed form."""

    def __init__(self) -> None:
        super().__init__()
        kernel_config = get_current_vllm_config().kernel_config
        self.tilelang = kernel_config.linear_backend == "tilelang"

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from vllm.v1.worker import display_carveout

        n = sum(output_partition_sizes)
        k = input_size_per_partition
        shape = (n, k)
        # In the GB10 display carve-out when it is required: the checkpoint
        # streams straight in and the head is then packed where it lies.
        # With the TileLang backend, reserve 2.5 * n * k bytes (1.25 * n * k
        # BF16 elements) in that single carve-out buffer so the packed 12-bit
        # head (1.5 * n * k bytes) and its 8-bit screening table (1.0 * n * k
        # bytes) share one carve-out mapping without allocating host memory.
        alloc_shape = (n, k + k // 4) if self.tilelang else shape
        raw = display_carveout.allocate(
            alloc_shape, params_dtype, torch.get_default_device()
        )
        if raw is not None and self.tilelang:
            data = raw.view(-1)[: n * k].view(shape)
            row_bytes = k + k // 2
            layer._carveout_screen_wq = (
                raw.view(torch.uint8)
                .view(-1)[n * row_bytes : n * (row_bytes + k)]
                .view(shape)
            )
        else:
            data = raw
        weight = Parameter(
            data if data is not None else torch.empty(*shape, dtype=params_dtype),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return
        weight = layer.weight.data
        in_features = int(weight.shape[1])
        # In place: the packed rows take the first three quarters of the BF16
        # head's own storage, so loading frees nothing the BF16 head would not.
        # On GB10 freed memory stays charged to the process, so an extra
        # allocation and free here would cost host memory for good.
        if self.tilelang:
            from vllm.models.deepseek_v4_1.tilelang.vocab import pack_bf16_weight

            packed = pack_bf16_weight(weight, in_place=True)
        else:
            projection = get_b12x_packed_bf16_vocab_projection()
            if projection is not None:
                packed = projection.pack(weight, in_place=True)
            else:
                from vllm.models.deepseek_v4_1.tilelang.vocab import pack_bf16_weight

                packed = pack_bf16_weight(weight, in_place=True)
        from vllm.v1.worker import display_carveout

        del weight
        replace_parameter(layer, "weight", display_carveout.compact(packed.values))
        replace_parameter(layer, "weight_exception_rows", packed.exception_rows)
        replace_parameter(layer, "weight_exception_entries", packed.exception_entries)
        replace_parameter(layer, "weight_row_offsets", packed.row_offsets)
        layer.packed_bf16_in_features = in_features
        layer.packed_bf16_exponent_base = packed.exponent_base
        layer.packed_bf16_exceptions = packed.exceptions
        layer._already_called_process_weights_after_loading = True
        if self.tilelang:
            from vllm.models.deepseek_v4_1.tilelang.vocab import _ensure_screen_weights

            _ensure_screen_weights(layer)
            # The model's head dtype: FP32 logits as DeepSeek's reference head, or
            # the model dtype. The projection is declared with it when the worker
            # prepares kernels (declare_layer_kernels).
            model_config = get_current_vllm_config().model_config
            layer.tilelang_vocab_dtype = (
                model_config.head_dtype if model_config else torch.bfloat16
            )
        logger.info_once(
            "Packed the BF16 LM head exactly into 12 bits per weight: "
            "exponents %d..%d, %d out-of-window values kept exactly per shard.",
            packed.exponent_base,
            packed.exponent_base + 14,
            packed.exceptions,
        )

    def declare_layer_kernels(self, layer: torch.nn.Module) -> None:
        """With the TileLang backend, declare the packed head's projection: once the
        head is packed, since its kernels depend on the exponent base packing found."""
        if not self.tilelang or not hasattr(layer, "packed_bf16_exponent_base"):
            return
        from vllm.models.deepseek_v4_1.tilelang.vocab import prepare_packed_head

        prepare_packed_head(layer, layer.tilelang_vocab_dtype)

    @staticmethod
    def packed_weight(layer: torch.nn.Module):
        """The layer's current packed tensors (after any relocation)."""
        tilelang_method = getattr(getattr(layer, "quant_method", None), "tilelang", False)
        if not tilelang_method:
            projection = get_b12x_packed_bf16_vocab_projection()
            if projection is not None:
                return projection.PackedBf16Weight(
                    layer.weight,
                    layer.packed_bf16_exponent_base,
                    layer.weight_exception_rows,
                    layer.weight_exception_entries,
                    layer.weight_row_offsets,
                    layer.packed_bf16_exceptions,
                )
        from vllm.models.deepseek_v4_1.tilelang.vocab import PackedBf16Weight

        return PackedBf16Weight(
            layer.weight,
            layer.packed_bf16_exponent_base,
            layer.weight_exception_rows,
            layer.weight_exception_entries,
            layer.weight_row_offsets,
            layer.packed_bf16_exceptions,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise RuntimeError(
            "A packed BF16 LM head runs only through the logits processor's "
            "vocabulary projection (linear backend b12x or tilelang, no bias)."
        )

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("A packed BF16 LM head is not an input embedding.")
