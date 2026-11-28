"""DeepSeek-V4.1 model configuration hooks (FP32 head_dtype + TP=3 padding)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import vllm.envs as envs
from vllm.utils.math_utils import round_up

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig


def update_deepseek_v41_model_config(model_config: ModelConfig) -> None:
    if getattr(model_config.hf_config, "head_dtype", None) is None:
        model_config.hf_config.head_dtype = envs.VLLM_DS41_HEAD_DTYPE


def _pad_for_tp(model_config: ModelConfig, tp_size: int) -> None:
    if tp_size <= 1:
        return
    text_config = getattr(model_config.hf_config, "text_config", model_config.hf_config)
    original_heads = getattr(
        text_config,
        "original_num_attention_heads",
        getattr(text_config, "num_attention_heads", None),
    )
    original_groups = getattr(
        text_config,
        "original_o_groups",
        getattr(text_config, "o_groups", None),
    )
    padded_groups = (
        round_up(original_groups, tp_size) if original_groups is not None else None
    )
    padded_heads = (
        padded_groups * (original_heads // original_groups)
        if padded_groups is not None and original_heads is not None and original_groups
        else original_heads
    )
    original_moe_intermediate = getattr(
        text_config,
        "original_moe_intermediate_size",
        getattr(text_config, "moe_intermediate_size", None),
    )
    padded_moe_intermediate = (
        round_up(original_moe_intermediate, tp_size * 64)
        if original_moe_intermediate is not None
        else None
    )
    if (
        padded_groups == original_groups
        and padded_moe_intermediate == original_moe_intermediate
    ):
        return

    seen: set[int] = set()
    for cfg in (
        model_config.hf_config,
        getattr(model_config, "hf_text_config", None),
        text_config,
    ):
        if cfg is None or id(cfg) in seen:
            continue
        seen.add(id(cfg))
        cfg_any: Any = cfg
        if padded_groups is not None and padded_groups != original_groups:
            if hasattr(cfg, "num_attention_heads"):
                cfg_any.original_num_attention_heads = original_heads
                cfg.num_attention_heads = padded_heads
            if hasattr(cfg, "o_groups"):
                cfg_any.original_o_groups = original_groups
                cfg.o_groups = padded_groups
        if (
            padded_moe_intermediate is not None
            and padded_moe_intermediate != original_moe_intermediate
            and hasattr(cfg, "moe_intermediate_size")
        ):
            cfg_any.original_moe_intermediate_size = original_moe_intermediate
            cfg.moe_intermediate_size = padded_moe_intermediate

    if hasattr(model_config, "get_model_arch_config"):
        model_config.model_arch_config = model_config.get_model_arch_config()


def update_deepseek_v41_vllm_config(vllm_config: VllmConfig) -> None:
    tp_size = vllm_config.parallel_config.tensor_parallel_size
    _pad_for_tp(vllm_config.model_config, tp_size)
    spec = vllm_config.speculative_config
    if spec is not None and spec.draft_model_config is not None:
        draft_tp = spec.draft_tensor_parallel_size or tp_size
        update_deepseek_v41_model_config(spec.draft_model_config)
        _pad_for_tp(spec.draft_model_config, draft_tp)
