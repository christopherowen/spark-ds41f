# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native ratio-one/two compression with request-owned partial-state rings."""

import torch
import torch.nn as nn

from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm.models.deepseek_v4_1.sparse_mla import SLOT_DTYPE, DeepseekV41B12xBackend
from vllm.v1.attention.backend import MultipleOf
from vllm.v1.kv_cache_interface import CircularBufferSpec


class CompressorBackend(DeepseekV41B12xBackend):
    @staticmethod
    def get_name():
        return "B12X_DSV41_COMPRESSOR"

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [MultipleOf(1)]


class CompressorStateCache(nn.Module, AttentionLayerBase):
    def __init__(self, config, prefix):
        super().__init__()
        self.prefix = prefix
        self.kv_cache = torch.tensor([])
        spec = config.speculative_config
        draft_tokens = spec.num_speculative_tokens if spec is not None else 0
        # Drafts, the bonus token, and the previous row must coexist until
        # rejection selects the accepted boundary, as in upstream V4.1.
        self.block_size = max(8, 1 << (draft_tokens + 1).bit_length())
        context = config.compilation_config.static_forward_context
        if prefix in context:
            raise ValueError(f"Duplicate V4.1 cache {prefix}")
        context[prefix] = self

    def bind_kv_cache(self, cache):
        self.kv_cache = cache.view(cache.shape[0], -1)

    def get_kv_cache_spec(self, config):
        return CircularBufferSpec(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=1024,
            head_size_v=0,
            dtype=torch.float32,
        )

    def get_attn_backend(self):
        return CompressorBackend

    def forward(self):
        raise RuntimeError("compression state is consumed by its owning compressor")


class DeepseekCompressor(nn.Module):
    def __init__(
        self,
        vllm_config,
        compress_ratio,
        hidden_size,
        head_dim,
        rotate=False,
        prefix="",
        k_cache_prefix="",
    ):
        super().__init__()
        if compress_ratio not in (1, 2) or head_dim != 512:
            raise ValueError("V4.1 compression requires ratio1/2 and head512")
        self.prefix, self.compress_ratio = prefix, compress_ratio
        self.capacity = vllm_config.scheduler_config.max_num_batched_tokens
        self.requests = vllm_config.scheduler_config.max_num_seqs
        self.fused_wkv_wgate = MergedColumnParallelLinear(
            hidden_size,
            [512] * compress_ratio,
            bias=False,
            return_bias=False,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.fused_wkv_wgate",
            params_dtype=torch.bfloat16,
        )
        from vllm.models.deepseek_v4_1.tilelang.linear import TileLangSplitLinearMethod

        # wkv and wgate (ratio 2) each project into their own buffer.
        self.fused_wkv_wgate.quant_method = TileLangSplitLinearMethod(compress_ratio)
        self.fused_wkv_wgate.out_dtype = (
            torch.float32 if compress_ratio == 2 else torch.bfloat16
        )
        # The TileLang compression applies the norm's weight itself: the module
        # only holds the parameter, under its checkpoint name.
        self.norm = nn.Module()
        self.norm.weight = nn.Parameter(
            torch.ones(512, dtype=torch.float32), requires_grad=False
        )
        self.state_cache = (
            CompressorStateCache(vllm_config, f"{prefix}.state_cache")
            if compress_ratio == 2
            else None
        )
        self._values = self._latent = self._emitted = self._slots = None
        self._gates = self._pending_values = self._pending_gates = None
        self._pending_tags = self._local_ids = self._row_ids = None

    def _ensure_capacity_buffers(self, device, t, r) -> None:
        """Allocate the reusable serving storage for ``t`` tokens and ``r``
        requests once, before memory profiling, when the kernels are bound."""
        if self._values is not None:
            if self._values.shape[0] != t:
                raise RuntimeError("compressor capacity changed after preparation")
            return

        def alloc(shape, dtype):
            return torch.empty(shape, dtype=dtype, device=device)

        ratio = self.compress_ratio
        self._values = alloc((t, 512), torch.float32 if ratio == 2 else torch.bfloat16)
        self._latent = alloc((t, 512), torch.bfloat16)
        self._emitted = alloc((t,), torch.bool)
        self._slots = alloc((t,), SLOT_DTYPE)
        self._row_ids = torch.arange(r, dtype=torch.int64, device=device)
        if ratio == 2:
            self._gates = alloc((t, 512), torch.float32)
            self._pending_values = alloc((r, 512), torch.float32)
            self._pending_gates = alloc((r, 512), torch.float32)
            self._pending_tags = alloc((r,), torch.int64)
            self._local_ids = alloc((r,), torch.int64)

    def _project(self, hidden_states):
        from vllm.models.deepseek_v4_1.tilelang.linear import project_parts

        project_parts(
            self.fused_wkv_wgate,
            hidden_states,
            (self._values, self._gates)[: self.compress_ratio],
            hidden_states.shape[0],
        )

    def _tilelang_calls(self) -> list:
        """The TileLang compression kernels ``_forward_tilelang`` launches, as
        ``prepare.call``s, for the step metadata's ``max_num_seqs`` requests."""
        from vllm.models.deepseek_v4_1.tilelang import compress as tl
        from vllm.models.deepseek_v4_1.tilelang import prepare

        if self.compress_ratio == 1:
            return [prepare.call(tl.compress_kernel, 1, self.requests, 1)]
        # The ring holds one state per token, so its metadata's page
        # (``state_metadata.block_size``) is its block size.
        page = self.state_cache.block_size
        return [
            prepare.call(tl.gather_partial_state_kernel, self.requests, page),
            prepare.call(tl.compress_kernel, 2, self.requests, page),
        ]

    def declare_kernels(self) -> None:
        """Declare the TileLang compression kernels (the worker calls this after
        loading); binding allocates the serving storage and ratio 1's placeholder
        operands before memory profiling, so the first call, possibly under graph
        capture, allocates nothing."""
        from vllm.models.deepseek_v4_1.tilelang import compress as tl
        from vllm.models.deepseek_v4_1.tilelang import prepare

        def bind() -> None:
            device = self.fused_wkv_wgate.weight.device
            self._ensure_capacity_buffers(device, self.capacity, self.requests)
            if self.compress_ratio == 1:
                tl.placeholder(self.requests, device)

        prepare.declare(self._tilelang_calls, bind=bind)

    def _forward_tilelang(self, rows, metadata, state_metadata):
        """The TileLang compression: at ratio 2 the requests' carries from the state
        ring, then one launch over the step's rows that also saves the ring."""
        from vllm.models.deepseek_v4_1.tilelang import compress as tl

        if self.compress_ratio == 1:
            tl.compress(
                rows, 1, self._values, None, self.norm.weight,
                metadata.query_start_loc, metadata.request_positions, self._row_ids,
                metadata.slot_mapping, metadata.live_counts, self._latent,
                self._emitted, self._slots,
            )  # fmt: skip
            return
        if state_metadata is None:
            raise RuntimeError("ratio2 compression requires request-state metadata")
        pool, page = self.state_cache.kv_cache, state_metadata.block_size
        tl.gather_partial_state(
            pool, state_metadata.block_table, state_metadata.query_start_loc,
            state_metadata.request_positions, state_metadata.live_counts,
            self._pending_values, self._pending_gates, self._pending_tags,
            self._local_ids, page=page,
        )  # fmt: skip
        tl.compress(
            rows, 2, self._values, self._gates, self.norm.weight,
            metadata.query_start_loc, metadata.request_positions, self._local_ids,
            metadata.slot_mapping, metadata.live_counts, self._latent, self._emitted,
            self._slots, pending_values=self._pending_values,
            pending_gates=self._pending_gates, tags=self._pending_tags, pool=pool,
            state_slots=state_metadata.slot_mapping, page=page,
        )  # fmt: skip

    def forward(self, hidden_states, metadata, state_metadata=None):
        rows = hidden_states.shape[0]
        if self._values is None:
            raise RuntimeError("compressor kernels must be bound before forwarding")
        self._project(hidden_states)
        self._forward_tilelang(rows, metadata, state_metadata)
        return self._latent[:rows], self._slots[:rows]
