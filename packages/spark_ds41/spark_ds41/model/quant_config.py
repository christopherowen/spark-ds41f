# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 checkpoint quantization configuration."""

import torch

from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    create_fp8_scale_parameter,
    create_fp8_weight_parameter,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.model_executor.parameter import BlockQuantScaleParameter


class Block32FP8LinearMethod(LinearMethodBase):
    """Checkpoint block-32 FP8 weights: E4M3 values, one UE8M0 scale per 32x32 block.

    Kernel families subclass it for processing and execution.
    """

    def __init__(self, quant_config):
        if quant_config.weight_block_size != [32, 32]:
            raise ValueError("V4.1 requires checkpoint block32 FP8 linears")

    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **extra_weight_attrs,
    ):
        loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = sum(output_partition_sizes)
        layer.orig_dtype = params_dtype
        layer.weight_block_size = [32, 32]
        layer.register_parameter(
            "weight",
            create_fp8_weight_parameter(
                sum(output_partition_sizes), input_size_per_partition, loader
            ),
        )
        layer.register_parameter(
            "weight_scale_inv",
            create_fp8_scale_parameter(
                BlockQuantScaleParameter,
                output_partition_sizes,
                input_size_per_partition,
                [32, 32],
                loader,
                scale_dtype=torch.float8_e8m0fnu,
            ),
        )


class DeepseekV41FP8Config(Fp8Config):
    is_scale_e8m0 = True

    @classmethod
    def get_name(cls):
        return "deepseek_v41_fp8"

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
        quant_method = hf_quant_cfg.get("quant_method")
        if quant_method == "deepseek_v41_fp8" or (
            getattr(hf_config, "model_type", None)
            in (
                "deepseek_v41",
                "deepseek_v41_text",
            )
            and quant_method == "fp8"
        ):
            return cls.get_name()
        return None

    def get_quant_method(self, layer, prefix):
        from vllm.model_executor.layers.fused_moe import RoutedExperts
        from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            VocabParallelEmbedding,
        )

        from . import kernels

        if isinstance(layer, LinearBase):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
                match_mode=self.ignored_layers_match_mode,
            ):
                return kernels.linear_method()
            return kernels.block_fp8_linear_method(self)
        if isinstance(layer, VocabParallelEmbedding):
            return kernels.embedding_method()
        if isinstance(layer, RoutedExperts):
            return Mxfp4MoEMethod(layer.moe_config)
        return None
