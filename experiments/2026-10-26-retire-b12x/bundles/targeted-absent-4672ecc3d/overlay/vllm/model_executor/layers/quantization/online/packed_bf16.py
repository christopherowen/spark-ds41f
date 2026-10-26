# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vocabulary heads stored in an exact 12-bit packed BF16 form.

The checkpoint's BF16 head loads unchanged; after loading it is packed, in
its own storage, into one sign-and-mantissa byte plus a 4-bit exponent code
per value, with the rare out-of-window values kept exactly in a per-row
exception list (``vllm.models.deepseek_v4_1.tilelang.vocab_pack``, B12X's
format). Packing verifies that every bit round-trips and refuses otherwise,
so the served head is the checkpoint tensor in 75% of the bytes, not an
approximation.

The packed head runs only through the logits processor's TileLang vocabulary
projection, which reads the packed bytes directly.
"""

import importlib.util

import torch
from torch.nn import Parameter

from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.utils import replace_parameter, set_weight_attrs

logger = init_logger(__name__)


def packed_bf16_supported(input_size: int, params_dtype: torch.dtype) -> bool:
    if importlib.util.find_spec("tilelang") is None:
        return False
    from vllm.models.deepseek_v4_1.tilelang import vocab_pack

    return (
        vocab_pack.is_supported()
        and params_dtype == torch.bfloat16
        and input_size % vocab_pack.GROUP == 0
    )


class PackedBf16LmHeadMethod(QuantizeMethodBase):
    """A BF16 LM head served from its exact 12-bit packed form."""

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

        shape = (sum(output_partition_sizes), input_size_per_partition)
        # In the GB10 display carve-out when it is required: the checkpoint
        # streams straight in and the head is then packed where it lies.
        data = display_carveout.allocate(
            shape, params_dtype, torch.get_default_device()
        )
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
        from vllm.models.deepseek_v4_1.tilelang.vocab_pack import pack

        weight = layer.weight.data
        in_features = int(weight.shape[1])
        # In place: the packed rows take the first three quarters of the BF16
        # head's own storage, so loading frees nothing the BF16 head would not.
        # On GB10 freed memory stays charged to the process, so an extra
        # allocation and free here would cost host memory for good.
        packed = pack(weight, in_place=True)
        del weight
        replace_parameter(layer, "weight", packed.values)
        replace_parameter(layer, "weight_exception_rows", packed.exception_rows)
        replace_parameter(layer, "weight_exception_entries", packed.exception_entries)
        replace_parameter(layer, "weight_row_offsets", packed.row_offsets)
        layer.packed_bf16_in_features = in_features
        layer.packed_bf16_exponent_base = packed.exponent_base
        layer.packed_bf16_exceptions = packed.exceptions
        layer._already_called_process_weights_after_loading = True
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
        """Declare the packed head's TileLang projection: once the head is packed,
        since its kernels depend on the exponent base packing found."""
        if not hasattr(layer, "packed_bf16_exponent_base"):
            return
        from vllm.models.deepseek_v4_1.tilelang.vocab import prepare_packed_head

        prepare_packed_head(layer, layer.tilelang_vocab_dtype)

    @staticmethod
    def packed_weight(layer: torch.nn.Module):
        """The layer's current packed tensors (after any relocation)."""
        from vllm.models.deepseek_v4_1.tilelang.vocab_pack import PackedBf16Weight

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
            "TileLang vocabulary projection (no bias)."
        )

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("A packed BF16 LM head is not an input embedding.")
