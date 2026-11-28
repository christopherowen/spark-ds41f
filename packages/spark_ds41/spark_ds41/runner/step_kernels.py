# SPDX-License-Identifier: Apache-2.0
"""Thin TileLang step-kernel declaration and KV-spec query helpers for Spark."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import Any, cast

import torch

from spark_ds41.sampling import tilelang_inputs, tilelang_sampling, tilelang_token_counts
from vllm import envs
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroup,
    KVCacheSpec,
)
from vllm.v1.worker.utils import AttentionGroup, prepare_kernel_block_sizes


def iter_layer_specs(kv_cache_spec: Any) -> Collection[KVCacheSpec]:
    if hasattr(kv_cache_spec, "kv_cache_specs"):
        return kv_cache_spec.kv_cache_specs.values()
    return (kv_cache_spec,)


def dead_rows_enabled() -> bool:
    return float(getattr(envs, "SPARK_DS41F_DSPARK_DEAD_ROWS_TAU", 0.0) or 0.0) > 0.0


def declare_block_table_kernels(*, cp_size: int, cp_interleave: int) -> None:
    tilelang_inputs.declare(
        lambda: tilelang_inputs.block_table_calls(cp_size, cp_interleave)
    )


def register_input_batch_warmups(*, dead_rows: bool) -> None:
    for dead in (False, True) if dead_rows else (False,):
        tilelang_token_counts.register_counts(dead_rows=dead)
    tilelang_token_counts.register_counts(dead_rows=False, one=True)


def declare_input_batch_kernels(
    *,
    num_lookahead: int,
    num_speculative_steps: int,
    num_new_sampled_tokens: int,
    dead_rows: bool,
    sampled_widths: Sequence[int],
    commits: Sequence[tuple[bool, bool, bool]],
    draft_widths: Sequence[int] = (),
) -> None:
    tilelang_inputs.declare(
        lambda: tilelang_inputs.input_batch_calls(
            num_lookahead=num_lookahead,
            num_speculative_steps=num_speculative_steps,
            num_new_sampled_tokens=num_new_sampled_tokens,
            dead_rows=dead_rows,
            sampled_widths=sampled_widths,
            commits=commits,
            draft_widths=draft_widths,
        )
    )


def declare_staged_write_kernels() -> None:
    tilelang_inputs.declare(tilelang_inputs.staged_write_calls)


def declare_zero_kv_blocks_kernel() -> None:
    if getattr(envs, "VLLM_KV_CACHE_POISON", False):
        tilelang_inputs.declare(tilelang_inputs.fill_kv_blocks_calls)
    else:
        tilelang_inputs.declare(tilelang_inputs.zero_kv_blocks_calls)


def register_drafting_warmups(
    *,
    logits_dtypes: Sequence[torch.dtype],
    cache_dtype: torch.dtype,
    vocab_size: int,
    use_fp64: bool,
) -> None:
    if use_fp64:
        raise ValueError(
            "use_fp64_gumbel: the V2 runner\x27s TileLang sampler draws in FP32 only"
        )
    tilelang_sampling.declare(
        lambda: [
            call
            for dtype in dict.fromkeys(logits_dtypes)
            for col_dtype in (torch.int32, torch.int64)
            for per_token_col in (False, True)
            for call in tilelang_sampling.drafting_calls(
                dtype, cache_dtype, col_dtype, per_token_col
            )
        ]
        + tilelang_sampling.gumbel_finish_calls(vocab_size)
    )


def select_kernel_block_sizes(
    kv_cache_groups: list[KVCacheGroup], vllm_config: VllmConfig
) -> list[int]:
    layer_type = cast(type[Any], AttentionLayerBase)
    attn_groups: list[list[AttentionGroup]] = []
    for kv_cache_group_id, kv_cache_group in enumerate(kv_cache_groups):
        specs = kv_cache_group.kv_cache_specs
        if not any(spec.has_layer_views for spec in specs.values()):
            attn_groups.append([])
            continue
        attn_layers = get_layers_from_vllm_config(
            vllm_config, layer_type, kv_cache_group.layer_names
        )
        attn_groups.append(
            [
                AttentionGroup(
                    layer.get_attn_backend(),
                    [layer_name],
                    specs[layer_name],
                    kv_cache_group_id,
                )
                for layer_name, layer in attn_layers.items()
            ]
        )
    return prepare_kernel_block_sizes(
        KVCacheConfig(
            num_blocks=0, kv_cache_tensors=[], kv_cache_groups=kv_cache_groups
        ),
        attn_groups,
    )
