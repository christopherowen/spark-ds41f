# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native ratio-one/two compression with request-owned partial-state rings."""

from weakref import WeakValueDictionary

import torch
import torch.nn as nn

from vllm.distributed.communication_op import tensor_model_parallel_all_gather
from vllm.model_executor.layers import linear as linear_layers
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm.models.deepseek_v4_1.sparse_mla import SLOT_DTYPE, DeepseekV41B12xBackend
from vllm.v1.attention.backend import MultipleOf
from vllm.v1.kv_cache_interface import CircularBufferSpec


def _compressor_part_cols(tp_size: int) -> int:
    """Per-rank column width per 512-column compressor part, padded to a multiple of 64."""
    if tp_size <= 1:
        return 512
    block = tp_size * 64
    return ((512 + block - 1) // block) * 64


class _PaddedSplitColumnParallelLinear(MergedColumnParallelLinear):
    """MergedColumnParallelLinear that pads each 512-column logical part to a
    64-column multiple per TP rank and zeroes any trailing padded rows."""

    def __init__(self, input_size: int, compress_ratio: int, **kwargs) -> None:
        disable_tp = bool(kwargs.get("disable_tp", False))
        tp_size = (
            1
            if disable_tp
            else linear_layers.get_tensor_model_parallel_world_size()
        )
        self.logical_part_size = 512
        self.part_cols = _compressor_part_cols(tp_size)
        padded_part_size = self.part_cols * tp_size
        super().__init__(
            input_size,
            [padded_part_size] * compress_ratio,
            **kwargs,
        )
        rank_start = self.tp_rank * self.part_cols
        copy_end = min(self.logical_part_size, rank_start + self.part_cols)
        self.valid_part_cols = max(0, copy_end - rank_start)
        with torch.no_grad():
            self.weight.zero_()

    def _zero_pad_rows(self) -> None:
        if (
            self.valid_part_cols >= self.part_cols
            or not hasattr(self, "weight")
            or self.weight.data.device.type == "meta"
            or hasattr(self, "packed_bf16_exponent_base")
        ):
            return
        with torch.no_grad():
            self.weight.data[self.valid_part_cols : self.part_cols].zero_()
            if len(self.output_sizes) > 1:
                self.weight.data[self.part_cols + self.valid_part_cols :].zero_()

    def process_weights_after_loading(self) -> None:
        self._zero_pad_rows()
        if hasattr(super(), "process_weights_after_loading"):
            super().process_weights_after_loading()
        elif self.quant_method is not None:
            self.quant_method.process_weights_after_loading(self)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int | None = None,
    ) -> None:
        from vllm.model_executor.model_loader.weight_utils import copy_weight

        if loaded_shard_id is None:
            if (
                loaded_weight.shape[0]
                == len(self.output_sizes) * self.logical_part_size
            ):
                for shard_id in range(len(self.output_sizes)):
                    shard = loaded_weight.narrow(
                        0,
                        shard_id * self.logical_part_size,
                        self.logical_part_size,
                    )
                    self.weight_loader(param, shard, loaded_shard_id=shard_id)
                return
            copy_weight(param.data, loaded_weight)
            return
        assert 0 <= loaded_shard_id < len(self.output_sizes)
        assert loaded_weight.shape[0] == self.logical_part_size
        param_data = param.data.narrow(
            0, loaded_shard_id * self.part_cols, self.part_cols
        )
        param_data.zero_()
        rank_start = self.tp_rank * self.part_cols
        copy_end = min(self.logical_part_size, rank_start + self.part_cols)
        copy_len = max(0, copy_end - rank_start)
        if copy_len > 0:
            shard = loaded_weight.narrow(0, rank_start, copy_len)
            assert param_data.shape[1:] == shard.shape[1:]
            copy_weight(param_data.narrow(0, 0, copy_len), shard)

    weight_loader_v2 = weight_loader


class _SharedCompressorBuffers:
    """Transient capacity scratch shared across sequential compressor layers."""

    def __init__(
        self, device: torch.device, t: int, r: int, ratio: int, tp_size: int = 1
    ) -> None:
        def alloc(shape, dtype):
            return torch.empty(shape, dtype=dtype, device=device)

        val_dtype = torch.float32 if ratio == 2 else torch.bfloat16
        part_cols = _compressor_part_cols(tp_size)
        self.values = alloc((t, 512), val_dtype)
        self.local_shards = (
            alloc((ratio * t, part_cols), val_dtype) if tp_size > 1 else None
        )
        self.latent = alloc((t, 512), torch.bfloat16)
        self.emitted = alloc((t,), torch.bool)
        self.slots = alloc((t,), SLOT_DTYPE)
        self.row_ids = torch.arange(r, dtype=torch.int64, device=device)
        if ratio == 2:
            self.gates = alloc((t, 512), torch.float32)
            self.pending_values = alloc((r, 512), torch.float32)
            self.pending_gates = alloc((r, 512), torch.float32)
            self.pending_tags = alloc((r,), torch.int64)
            self.local_ids = alloc((r,), torch.int64)
        else:
            self.gates = self.pending_values = self.pending_gates = None
            self.pending_tags = self.local_ids = None


_SHARED_BUFFERS: WeakValueDictionary[
    tuple[torch.device, int, int, int, int], _SharedCompressorBuffers
] = WeakValueDictionary()


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
        # Every draft a request verifies in a step (copy drafts widen it).
        draft_tokens = spec.num_verification_tokens if spec is not None else 0
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
        # The gather reads a request's slot only after the request wrote it (an
        # odd first position; every ring starts at an even one).
        return CircularBufferSpec(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=1024,
            head_size_v=0,
            dtype=torch.float32,
            uses_only_written_slots=True,
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
        cache_cfg = getattr(vllm_config, "cache_config", None)
        block_size = (
            getattr(cache_cfg, "block_size", 64) if cache_cfg is not None else 64
        )
        self._main_page = block_size // compress_ratio
        model_cfg = getattr(vllm_config, "model_config", None)
        hf = getattr(model_cfg, "hf_config", None) if model_cfg is not None else None
        self._rope_dim = int(getattr(hf, "qk_rope_head_dim", 64))
        tp_size = linear_layers.get_tensor_model_parallel_world_size()
        self._sp_enabled = compress_ratio in (1, 2) and tp_size > 1
        self._sp_weight = self._sp_kernel = None
        # Replicate the 4 compressor layers (30 MiB total across the model) in
        # uncompressed BF16 up through TP4 so decode avoids 4 RoCE all-gathers,
        # 4 column-gather launches, and L2-resident 12-bit unpack overhead.
        disable_tp = tp_size <= 4
        self.fused_wkv_wgate = _PaddedSplitColumnParallelLinear(
            hidden_size,
            compress_ratio,
            bias=False,
            return_bias=False,
            gather_output=False,
            quant_config=None,
            disable_tp=disable_tp,
            prefix=f"{prefix}.fused_wkv_wgate",
            params_dtype=torch.bfloat16,
        )
        from vllm.models.deepseek_v4_1.tilelang.linear import TileLangSplitLinearMethod

        # wkv and wgate (ratio 2) each project into their own buffer.
        self.fused_wkv_wgate.pack_bf16 = not disable_tp
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
        self._local_shards = None

    def _ensure_capacity_buffers(self, device, t, r) -> None:
        """Allocate the reusable serving storage for ``t`` tokens and ``r``
        requests once, before memory profiling, when the kernels are bound."""
        if self._values is not None:
            if self._values.shape[0] != t:
                raise RuntimeError("compressor capacity changed after preparation")
            return
        dev = torch.device(device)
        if dev.type == "cuda" and dev.index is None:
            dev = torch.device("cuda", torch.cuda.current_device())
        tp_size = int(getattr(getattr(self, "fused_wkv_wgate", None), "tp_size", 1))
        key = (dev, int(t), int(r), int(self.compress_ratio), tp_size)
        shared = _SHARED_BUFFERS.get(key)
        if shared is None:
            shared = _SharedCompressorBuffers(
                dev, int(t), int(r), int(self.compress_ratio), tp_size
            )
            _SHARED_BUFFERS[key] = shared
        self._shared_buffers = shared
        self._values = shared.values
        self._local_shards = shared.local_shards
        self._latent = shared.latent
        self._emitted = shared.emitted
        self._slots = shared.slots
        self._row_ids = shared.row_ids
        self._gates = shared.gates
        self._pending_values = shared.pending_values
        self._pending_gates = shared.pending_gates
        self._pending_tags = shared.pending_tags
        self._local_ids = shared.local_ids

    def _project_local(self, hidden_states):
        from vllm.models.deepseek_v4_1.tilelang.linear import project_parts

        rows = hidden_states.shape[0]
        tp_size = int(getattr(getattr(self, "fused_wkv_wgate", None), "tp_size", 1))
        if tp_size <= 1:
            project_parts(
                self.fused_wkv_wgate,
                hidden_states,
                (self._values, self._gates)[: self.compress_ratio],
                rows,
            )
            return None
        part_cols = _compressor_part_cols(tp_size)
        local = self._local_shards[: self.compress_ratio * rows].view(
            self.compress_ratio, rows, part_cols
        )
        project_parts(
            self.fused_wkv_wgate,
            hidden_states,
            tuple(local),
            rows,
        )
        return tensor_model_parallel_all_gather(
            local.view(self.compress_ratio * rows, part_cols), dim=0
        )

    def _finish_project(self, gathered, rows: int) -> None:
        if gathered is None:
            return
        from vllm.models.deepseek_v4_1.tilelang.gather import gather_columns_into

        tp_size = int(getattr(getattr(self, "fused_wkv_wgate", None), "tp_size", 1))
        outputs = (self._values[:rows],)
        if self.compress_ratio == 2:
            outputs += (self._gates[:rows],)
        gather_columns_into(gathered, outputs, tp_size)

    def _project(self, hidden_states):
        self._finish_project(
            self._project_local(hidden_states), hidden_states.shape[0]
        )

    def _tilelang_calls(self) -> list:
        """The TileLang compression kernels ``_forward_tilelang`` launches, as
        ``prepare.call``s, for the step metadata's ``max_num_seqs`` requests."""
        from vllm.models.deepseek_v4_1.tilelang import compress as tl
        from vllm.models.deepseek_v4_1.tilelang import prepare

        main_page = getattr(self, "_main_page", 64 // self.compress_ratio)
        rope_dim = getattr(self, "_rope_dim", 64)
        from vllm.models.deepseek_v4_1.tilelang.gather import gather_call

        tp_size = int(getattr(getattr(self, "fused_wkv_wgate", None), "tp_size", 1))
        part_cols = _compressor_part_cols(tp_size)
        dtype = torch.bfloat16 if self.compress_ratio == 1 else torch.float32
        gather_calls = (
            [
                gather_call(
                    tp_size, part_cols, 512, self.compress_ratio, dtype, packed
                )
                for packed in (False, True)
            ]
            if tp_size > 1
            else []
        )
        if self.compress_ratio == 1:
            return gather_calls + [
                prepare.call(tl.compress_kernel, 1, self.requests, 1),
                prepare.call(
                    tl.compress_kernel,
                    1,
                    self.requests,
                    1,
                    write_kv=True,
                    kv_page_size=main_page,
                    rope_dim=rope_dim,
                ),
            ]
        # The ring holds one state per token, so its metadata's page
        # (``state_metadata.block_size``) is its block size.
        page = self.state_cache.block_size
        return gather_calls + [
            prepare.call(tl.gather_partial_state_kernel, self.requests, page),
            prepare.call(tl.compress_kernel, 2, self.requests, page),
            prepare.call(
                tl.compress_kernel,
                2,
                self.requests,
                page,
                write_kv=True,
                kv_page_size=main_page,
                rope_dim=rope_dim,
            ),
        ]

    def _sp_projection_call(self):
        """Use the original TP shard's K reduction for the full row projection."""
        from vllm.models.deepseek_v4_1.tilelang import prepare
        from vllm.models.deepseek_v4_1.tilelang.gemm import bf16_gemm, bf16_gemm_two
        from vllm.models.deepseek_v4_1.tilelang.linear import _bf16_shape

        layer = self.fused_wkv_wgate
        split = layer.quant_method._split(layer)
        if self.compress_ratio == 2:
            return prepare.call(
                bf16_gemm_two,
                512,
                _bf16_shape(layer)[1],
                out_dtype="float32",
                shards=split["shards"],
                blocked=split["blocked"],
            )
        return prepare.call(
            bf16_gemm,
            512,
            _bf16_shape(layer)[1],
            out_dtype="bfloat16",
            shards=split["shards"],
            blocked=split["blocked"],
        )

    def declare_kernels(self) -> None:
        """Declare the TileLang compression kernels (the worker calls this after
        loading); binding allocates the serving storage and ratio 1's placeholder
        operands before memory profiling, so the first call, possibly under graph
        capture, allocates nothing."""
        from vllm.models.deepseek_v4_1.tilelang import compress as tl
        from vllm.models.deepseek_v4_1.tilelang import prepare

        sp_call = (
            self._sp_projection_call() if getattr(self, "_sp_enabled", False) else None
        )

        def bind() -> None:
            from vllm.models.deepseek_v4_1.tilelang.linear import unpack_bf16_layer

            device = self.fused_wkv_wgate.weight.device
            self._ensure_capacity_buffers(device, self.capacity, self.requests)
            if self.compress_ratio == 1:
                tl.placeholder(self.requests, device)
            if sp_call is not None:
                # The decode projection retains its original TP weight and kernels.
                # Through TP4 that weight is replicated, and SP prefill reads it in
                # place. Above TP4 SP prefill gathers the weight once before memory
                # profiling (5 MiB at ratio one, 10 MiB at ratio two, H=5120) and
                # slices each part's trailing TP column padding.
                weight = unpack_bf16_layer(self.fused_wkv_wgate).detach()
                tp_size = self.fused_wkv_wgate.tp_size
                if tp_size <= 1:
                    self._sp_weight = weight
                elif self.compress_ratio == 1:
                    self._sp_weight = tensor_model_parallel_all_gather(weight, dim=0)[
                        :512
                    ]
                else:
                    part_cols = _compressor_part_cols(tp_size)
                    gathered = tensor_model_parallel_all_gather(weight, dim=0)
                    self._sp_weight = (
                        gathered.view(tp_size, self.compress_ratio, part_cols, -1)
                        .permute(1, 0, 2, 3)
                        .reshape(self.compress_ratio, tp_size * part_cols, -1)[:, :512]
                        .reshape(self.compress_ratio * 512, -1)
                        .contiguous()
                    )
                self._sp_kernel = prepare.launch(sp_call)

        prepare.declare(
            lambda: self._tilelang_calls() + ([] if sp_call is None else [sp_call]),
            bind=bind,
        )

    def _forward_tilelang(
        self,
        rows,
        metadata,
        state_metadata,
        *,
        kv_cache=None,
        token_positions=None,
        cos_sin=None,
        kv_page_size=None,
        values=None,
        gates=None,
    ):
        """The TileLang compression: at ratio 2 the requests' carries from the state
        ring, then one launch over the step's rows that also saves the ring."""
        from vllm.models.deepseek_v4_1.tilelang import compress as tl

        values = self._values if values is None else values
        gates = self._gates if gates is None else gates
        kv_kwargs = {}
        if kv_cache is not None and token_positions is not None and cos_sin is not None:
            kv_kwargs = dict(
                kv_cache=kv_cache,
                token_positions=token_positions,
                cos_sin=cos_sin,
                kv_page_size=kv_page_size or self._main_page,
            )
        if self.compress_ratio == 1:
            tl.compress(
                rows, 1, values, None, self.norm.weight,
                metadata.query_start_loc, metadata.request_positions, self._row_ids,
                metadata.slot_mapping, metadata.live_counts, self._latent,
                self._emitted, self._slots, **kv_kwargs,
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
            rows, 2, values, gates, self.norm.weight,
            metadata.query_start_loc, metadata.request_positions, self._local_ids,
            metadata.slot_mapping, metadata.live_counts, self._latent, self._emitted,
            self._slots, pending_values=self._pending_values,
            pending_gates=self._pending_gates, tags=self._pending_tags, pool=pool,
            state_slots=state_metadata.slot_mapping, page=page, **kv_kwargs,
        )  # fmt: skip

    def forward(
        self,
        hidden_states,
        metadata,
        state_metadata=None,
        *,
        kv_cache=None,
        token_positions=None,
        cos_sin=None,
        kv_page_size=None,
        sp_rows=None,
    ):
        rows = hidden_states.shape[0]
        if self._values is None:
            raise RuntimeError("compressor kernels must be bound before forwarding")
        values = gates = None
        if sp_rows is not None:
            if self.compress_ratio not in (1, 2) or self._sp_kernel is None:
                raise RuntimeError(
                    "ratio-one/two SP projection must be bound before forwarding"
                )
            if rows != sp_rows.local_rows:
                raise ValueError("SP compressor input does not match local rows")
            local = self._values[:rows]
            # Always use the prefill GEMM, even for a short local slice: it keeps
            # the global prefill's K reduction and needs no decode workspace.
            if self.compress_ratio == 1:
                self._sp_kernel(hidden_states, self._sp_weight, local)
                values = sp_rows.gather(local)
                fg_kwargs = {"values": values}
            else:
                local_gates = self._gates[:rows]
                self._sp_kernel(hidden_states, self._sp_weight, local, local_gates)
                values, gates = sp_rows.gather_many(local, local_gates)
                fg_kwargs = {"values": values, "gates": gates}
            rows = sp_rows.num_tokens
        else:
            self._project(hidden_states)
            fg_kwargs = {"values": None}
        self._forward_tilelang(
            rows,
            metadata,
            state_metadata,
            kv_cache=kv_cache,
            token_positions=token_positions,
            cos_sin=cos_sin,
            kv_page_size=kv_page_size,
            **fg_kwargs,
        )
        return self._latent[:rows], self._slots[:rows]
