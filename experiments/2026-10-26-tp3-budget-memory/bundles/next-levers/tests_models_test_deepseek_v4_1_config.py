# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 configuration on GB10: kernel routing and TP3 head padding."""

from types import SimpleNamespace

import pytest

from vllm.config import ModelConfig
from vllm.model_executor.models.config import DeepseekV41ForCausalLMConfig
from vllm.models.deepseek_v4_1.nvidia.model import _select_dsv4_attn_cls
from vllm.v1.attention.backends.registry import AttentionBackendEnum


@pytest.mark.parametrize("component", ["linear_backend", "moe_backend"])
def test_tilelang_runs_deepseek_v41(component) -> None:
    """TileLang's kernels run the whole model: its attention backend and class, the
    linear and MoE backends, and aligned prefill chunks; another backend fails."""
    from vllm.models.deepseek_v4_1.tilelang.attention import (
        DeepseekV41TileLangAttention,
    )

    backend = AttentionBackendEnum.TILELANG
    kernel = SimpleNamespace(linear_backend="auto", moe_backend="auto")
    config = SimpleNamespace(
        attention_config=SimpleNamespace(backend=None),
        kernel_config=kernel,
        scheduler_config=SimpleNamespace(align_prefill_chunks=False),
        speculative_config=None,
    )

    DeepseekV41ForCausalLMConfig.verify_and_update_config(config)

    assert config.attention_config.backend is backend
    assert _select_dsv4_attn_cls(config) is DeepseekV41TileLangAttention
    assert kernel.linear_backend == kernel.moe_backend == "tilelang"
    assert config.scheduler_config.align_prefill_chunks

    setattr(kernel, component, "b12x")
    with pytest.raises(ValueError, match="requires the tilelang"):
        DeepseekV41ForCausalLMConfig.verify_and_update_config(config)
    setattr(kernel, component, "tilelang")

    config.attention_config.backend = AttentionBackendEnum.FLASH_ATTN
    with pytest.raises(ValueError, match=f"requires the {backend.name} attention"):
        DeepseekV41ForCausalLMConfig.verify_and_update_config(config)


def test_deepseek_v41_tp3_padding_uses_generic_parallel_hook() -> None:
    text_config = SimpleNamespace(num_attention_heads=64, o_groups=8)
    model_config = SimpleNamespace(
        architecture="DSparkV41DraftModel",
        hf_config=text_config,
        hf_text_config=text_config,
        model_arch_config=SimpleNamespace(total_num_attention_heads=64),
    )
    model_config.get_model_arch_config = lambda: SimpleNamespace(
        total_num_attention_heads=text_config.num_attention_heads
    )
    parallel_config = SimpleNamespace(tensor_parallel_size=3)

    ModelConfig._update_model_config_for_parallelism(model_config, parallel_config)
    ModelConfig._update_model_config_for_parallelism(model_config, parallel_config)

    assert text_config.original_num_attention_heads == 64
    assert text_config.original_o_groups == 8
    assert text_config.num_attention_heads == 72
    assert text_config.o_groups == 9
    assert model_config.model_arch_config.total_num_attention_heads == 72
