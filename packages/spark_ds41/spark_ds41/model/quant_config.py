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
            return SparkMxfp4MoEMethod.create(layer.moe_config)
        return None


class SparkMxfp4MoEMethod:
    @staticmethod
    def create(moe_config):
        from vllm import envs
        from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod
        from vllm.model_executor.utils import set_weight_attrs

        class _SparkMxfp4MoEMethod(Mxfp4MoEMethod):
            def create_weights(
                self,
                layer,
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                params_dtype,
                **extra_weight_attrs,
            ):
                slot_target = envs.VLLM_DS41_EXPERT_SLOTS
                prefix = getattr(layer, "prefix", None) or getattr(layer, "layer_name", "") or ""
                is_draft_layer = "dspark" in prefix or "mtp" in prefix
                if not is_draft_layer and "layers." in prefix:
                    from vllm.model_executor.models.utils import extract_layer_index
                    try:
                        is_draft_layer = extract_layer_index(prefix) >= 40
                    except ValueError:
                        pass
                use_slot_meta = (
                    slot_target != 0
                    and not is_draft_layer
                    and (slot_target < 0 or slot_target < num_experts)
                )
                from spark_ds41.direct_io.direct_loader import (
                    is_deferring_arena_weight_alloc,
                )
                defer_arena_meta = (not use_slot_meta) and is_deferring_arena_weight_alloc()
                if not use_slot_meta and not defer_arena_meta:
                    return super().create_weights(
                        layer,
                        num_experts,
                        hidden_size,
                        intermediate_size_per_partition,
                        params_dtype,
                        **extra_weight_attrs,
                    )
                self.num_experts = num_experts
                self.hidden_size = hidden_size
                self.intermediate_size = self.moe.intermediate_size_per_partition
                self.intermediate_size_per_partition = self.moe.intermediate_size_per_partition
                weight_dtype = torch.uint8
                scale_dtype = torch.uint8
                mxfp4_block = 32
                intermediate_size_per_partition = self.moe.intermediate_size_per_partition
                weight_loader = extra_weight_attrs.pop("weight_loader")
                scale_weight_loader = Mxfp4MoEMethod.get_scale_weight_loader(weight_loader)
                if use_slot_meta:
                    self.uses_meta_device = True
                    alloc_fn = lambda shape, dt: torch.empty(shape, dtype=dt, device=torch.device("meta"))
                else:
                    alloc_fn = lambda shape, dt: torch.empty((1,), dtype=dt).expand(shape)

                def _mark(p):
                    if defer_arena_meta:
                        p._direct_deferred_alloc = True
                    return p

                w13_weight = _mark(torch.nn.Parameter(alloc_fn((num_experts, self.moe.w13_num_shards * intermediate_size_per_partition, hidden_size // 2), weight_dtype), requires_grad=False))
                layer.register_parameter("w13_weight", w13_weight)
                set_weight_attrs(w13_weight, extra_weight_attrs)
                set_weight_attrs(w13_weight, {"weight_loader": weight_loader})

                w13_weight_scale = _mark(torch.nn.Parameter(alloc_fn((num_experts, self.moe.w13_num_shards * intermediate_size_per_partition, hidden_size // mxfp4_block), scale_dtype), requires_grad=False))
                layer.register_parameter("w13_weight_scale", w13_weight_scale)
                set_weight_attrs(w13_weight_scale, extra_weight_attrs)
                set_weight_attrs(w13_weight_scale, {"weight_loader": scale_weight_loader})
                w13_weight_scale.quant_method = "block"

                w2_weight = _mark(torch.nn.Parameter(alloc_fn((num_experts, hidden_size, intermediate_size_per_partition // 2), weight_dtype), requires_grad=False))
                layer.register_parameter("w2_weight", w2_weight)
                set_weight_attrs(w2_weight, extra_weight_attrs)
                set_weight_attrs(w2_weight, {"weight_loader": weight_loader})

                w2_weight_scale = _mark(torch.nn.Parameter(alloc_fn((num_experts, hidden_size, intermediate_size_per_partition // mxfp4_block), scale_dtype), requires_grad=False))
                layer.register_parameter("w2_weight_scale", w2_weight_scale)
                set_weight_attrs(w2_weight_scale, extra_weight_attrs)
                set_weight_attrs(w2_weight_scale, {"weight_loader": scale_weight_loader})
                w2_weight_scale.quant_method = "block"

                if self.moe.has_bias:
                    w13_bias = torch.nn.Parameter(alloc_fn((num_experts, self.moe.w13_num_shards * intermediate_size_per_partition), torch.bfloat16), requires_grad=False)
                    layer.register_parameter("w13_bias", w13_bias)
                    set_weight_attrs(w13_bias, extra_weight_attrs)
                    set_weight_attrs(w13_bias, {"weight_loader": weight_loader})
                    w2_bias = torch.nn.Parameter(alloc_fn((num_experts, hidden_size), torch.bfloat16), requires_grad=False)
                    layer.register_parameter("w2_bias", w2_bias)
                    set_weight_attrs(w2_bias, extra_weight_attrs)
                    set_weight_attrs(w2_bias, {"weight_loader": weight_loader})

                if use_slot_meta:
                    from vllm.model_executor.model_loader.reload.layerwise import (
                        initialize_online_processing,
                    )
                    initialize_online_processing(layer)

            def process_weights_after_loading(self, layer):
                if getattr(layer, "_already_called_process_weights_after_loading", False):
                    return
                layer._already_called_process_weights_after_loading = True
                super().process_weights_after_loading(layer)
                if getattr(layer, "slot_cache_config", None) is not None:
                    torch.cuda.empty_cache()

        return _SparkMxfp4MoEMethod(moe_config)
