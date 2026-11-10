# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

import numpy as np
import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.config import VllmConfig, replace
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.utils.math_utils import next_power_of_2
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    get_kv_cache_spec_sliding_window,
)
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cp_utils import maybe_prepare_dcp_local_seq_lens
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.dp_utils import DPSyncState, dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import DFlashCudaGraphManager
from vllm.v1.worker.gpu.spec_decode.dflash.utils import load_dflash_model
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
from vllm.v1.worker.gpu.spec_decode.utils import get_parallel_drafting_token_id
from vllm.v1.worker.utils import AttentionGroup

logger = init_logger(__name__)


class DraftBlockTableLayout(NamedTuple):
    """One draft KV cache group's block table, as ``BlockTables`` lays it out."""

    group_id: int
    width: int
    kernel_block_size: int
    cp_rank: int
    cp_size: int
    cp_interleave: int


class DFlashSpeculator(DraftModelSpeculator):
    _speculator_name = "DFlash"  # For logging, so we can share methods with subclasses

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        parallel_config = vllm_config.parallel_config
        if parallel_config.prefill_context_parallel_size > 1:
            vllm_config = copy.copy(vllm_config)
            vllm_config.parallel_config = replace(
                parallel_config,
                prefill_context_parallel_size=1,
            )
        super().__init__(vllm_config, device)

        self.hidden_states = torch.zeros(
            self.max_num_tokens, self.hidden_size, dtype=self.dtype, device=device
        )
        # Rows that pad the query batch, as the MoE routers read them from the
        # forward context (VLLM_MOE_SKIP_PADDING). Every row starts as padding,
        # so CUDA graph capture routes no dummy row; each step marks its live
        # rows. Persistent: captured graphs read it on replay.
        self._is_padding = torch.ones(
            self.max_num_tokens, dtype=torch.bool, device=device
        )

        # Multimodal inputs not currently supported.
        self.supports_mm_inputs = False

        # Each request emits exactly (bonus + N mask) query tokens per step.
        self.num_query_per_req = 1 + self.num_speculative_steps

        self.parallel_drafting_token_id = get_parallel_drafting_token_id(
            self.draft_model_config.hf_config
        )

        from vllm.model_executor.models.qwen3_dflash import dflash_has_any_non_causal

        self.requires_non_causal = dflash_has_any_non_causal(
            self.draft_model_config.hf_config
        )

        # Whether the anchor query position is itself a prediction. DFlash default uses
        # the anchor as the bonus token (only mask tokens predict); DSpark samples from
        # the anchor and the N-1 mask token positions. See _prepare_dflash_inputs_kernel
        dflash_config = (
            getattr(self.draft_model_config.hf_config, "dflash_config", None) or {}
        )
        if dflash_config.get("sample_from_anchor", False):
            raise ValueError(
                "sample_from_anchor=True is not supported for DFlash. "
                "DFlash uses a fixed 1+N query layout where the anchor "
                "is the bonus token."
            )
        self.sample_from_anchor = False

        # Context positions for K/V precompute, populated by prepare_dflash_inputs.
        # V4.1 DSpark can also read this fixed owner buffer in a context graph.
        self.context_positions = torch.zeros(
            self.max_num_tokens, dtype=torch.int64, device=device
        )

        # Per-mask-token sampling buffers. Flattened from (num_reqs, num_spec_tokens).
        max_num_sampled_tokens = self.max_num_reqs * self.num_speculative_steps
        self.sample_indices = torch.zeros(
            max_num_sampled_tokens, dtype=torch.int64, device=device
        )
        self.sample_pos = torch.zeros(
            max_num_sampled_tokens, dtype=torch.int64, device=device
        )
        # -1 marks an inert sampling row. CUDA graph capture can execute the
        # full buffer before a real batch has populated it, so zero would make
        # every padding row scatter into request slot 0.
        self.sample_idx_mapping = torch.full(
            (max_num_sampled_tokens,), -1, dtype=torch.int32, device=device
        )
        # [0, 1, ..., N-1, 0, 1, ..., N-1, ...] -> the per-token column index into
        # draft_logits[req, step, :].
        self.sample_col = torch.arange(
            self.num_speculative_steps, dtype=torch.int32, device=device
        ).repeat(self.max_num_reqs)

        self.query_cudagraph_manager: DFlashCudaGraphManager | None = None
        self.draft_kv_cache_group_id: int = -1
        self._context_preparer = None

    @property
    def attn_vllm_config(self) -> VllmConfig:
        # The draft's attention differs from the target's in causality.
        config = copy.copy(super().attn_vllm_config)
        config.attention_config = replace(
            self.vllm_config.attention_config,
            use_non_causal=self.requires_non_causal,
        )
        return config

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        wants_full = cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
        supports_full = (
            self.attn_cg_support.min_cg_support.value
            >= AttentionCGSupport.UNIFORM_BATCH.value
        )
        if wants_full and not supports_full:
            logger.warning(
                "%s draft attention (%s) does not support full CUDA graphs; "
                "running the draft eagerly.",
                self._speculator_name,
                self.attn_cg_support.min_cg_attn_backend,
            )
        # PIECEWISE cudagraphs are not supported for dflash.
        if wants_full and supports_full:
            cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
        else:
            cudagraph_mode = CUDAGraphMode.NONE

        self.query_cudagraph_manager = DFlashCudaGraphManager(
            self.vllm_config,
            self.device,
            cudagraph_mode,
            decode_query_len=self.num_query_per_req,
        )

    def capture(self) -> None:
        logger.info("Capturing model for %s speculator...", self._speculator_name)
        # Padded sample rows must not scatter into a live request during capture.
        self.sample_indices.zero_()
        self.sample_pos.zero_()
        self.sample_idx_mapping.fill_(-1)
        assert self.query_cudagraph_manager is not None
        # Capture runs dummy rows only: route none of them.
        self._is_padding.fill_(True)
        self.query_cudagraph_manager.capture(
            self._generate_draft,
            self.input_buffers,
            self.block_tables,
            self.attn_groups,
            self.kv_cache_config,
            self.max_model_len,
            causal=self._group_causal,
            progress_bar_desc=f"Capturing {self._speculator_name.lower()} CUDA graphs",
            capture_attn_metadata=not self._full_cg_needs_attn_metadata_build,
        )
        if torch.cuda.is_available():
            torch.accelerator.synchronize()
            torch.accelerator.empty_cache()
        # Only V4.1 DSpark supplies this hook. Other draft models retain their
        # existing eager context path and graph policy.
        capture_context = getattr(self.model, "capture_context_preparation", None)
        if capture_context is not None and self.query_cudagraph_manager.needs_capture():
            self._context_preparer = capture_context(
                self.vllm_config,
                self.hidden_states,
                self.context_positions,
                self._context_slot_mappings,
                self._layer_group_idx,
                # A decode step's target rows, copy drafts' included.
                self.vllm_config.max_decode_rows,
            )

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        model = load_dflash_model(target_model, self.vllm_config)
        bind_auxiliary_stream = getattr(model, "bind_target_auxiliary_stream", None)
        if callable(bind_auxiliary_stream):
            bind_auxiliary_stream(target_model, self.hidden_states)
        self._setup_vocab_parallel_drafting(model)
        return model

    def register_warmups(
        self, block_table_layouts: Sequence[DraftBlockTableLayout]
    ) -> None:
        """Declare the input preparation ``propose`` launches for each draft KV
        cache group's block table, given in ``draft_kv_cache_group_ids`` order
        before the KV cache exists (only its context parallelism keys the kernel;
        widths, block sizes and the batch sizes are launch data)."""
        from vllm.v1.worker.gpu import tilelang_inputs

        sample_from_anchor = self.sample_from_anchor
        layouts = [
            (layout.cp_size, layout.cp_interleave) for layout in block_table_layouts
        ]
        tilelang_inputs.declare(
            lambda: tilelang_inputs.dflash_calls(sample_from_anchor, layouts)
            + tilelang_inputs.padding_rows_calls()
        )

    def _setup_vocab_parallel_drafting(self, model: nn.Module) -> None:
        """Keep draft logits vocab-sharded (VLLM_DFLASH_VOCAB_PARALLEL_DRAFT=1).

        Probabilistic drafts are Gumbel-sampled per vocab shard and combined
        from per-rank winners (spec_decode.vocab_parallel), so the [rows, vocab]
        logits all-gather disappears; the cache holds this rank's shard.  Any
        layout the sampler does not cover keeps the full-vocab path.
        """
        from vllm.distributed import get_tensor_model_parallel_world_size

        self._vocab_parallel = None
        if not envs.VLLM_DFLASH_VOCAB_PARALLEL_DRAFT or self.draft_logits is None:
            return
        head = getattr(model, "lm_head", None)
        si = getattr(head, "shard_indices", None)
        tp = get_tensor_model_parallel_world_size()
        if (
            tp == 1
            or si is None
            or not hasattr(model, "compute_local_logits")
            or getattr(model, "draft_id_to_target_id", None) is not None
            or self.draft_watermarker is not None
            or self.speculative_config.rejection_sample_method != "standard"
            or si.num_org_vocab_padding != 0
            or si.num_added_elements != 0
            or si.num_org_elements * tp != self.vocab_size
        ):
            logger.info_once(
                "DFlash vocab-parallel drafting not applicable; "
                "keeping the full-vocab draft path"
            )
            return
        from vllm.v1.worker.gpu.spec_decode.vocab_parallel import VPDraftCache

        dtype, fill = self.draft_logits_spec(self.vllm_config)
        self.draft_logits = torch.full(
            (self.max_num_reqs, self.num_speculative_steps, si.num_org_elements),
            fill,
            dtype=dtype,
            device=self.device,
        )
        self._vocab_parallel = VPDraftCache(
            self.draft_logits, si.org_vocab_start_index, self.vocab_size, None
        )
        logger.info_once(
            "DFlash vocab-parallel drafting: shard [%d, %d) of %d, no draft-logits "
            "all-gather",
            si.org_vocab_start_index,
            si.org_vocab_end_index,
            self.vocab_size,
        )

    def sample_draft(
        self,
        hidden_states: torch.Tensor,
        sample_src_positions: torch.Tensor,
        idx_mapping: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        draft_step: torch.Tensor,
        draft_logits: torch.Tensor | None,
    ) -> torch.Tensor:
        vp = getattr(self, "_vocab_parallel", None)
        if vp is None or draft_logits is None or draft_logits is not vp.logits:
            return super().sample_draft(
                hidden_states,
                sample_src_positions,
                idx_mapping,
                temperature,
                seeds,
                draft_step,
                draft_logits,
            )
        from vllm.v1.worker.gpu.spec_decode import vocab_parallel as vp_sampling

        local = self.model.compute_local_logits(hidden_states)
        col = draft_step if draft_step.dim() > 0 else draft_step.expand(local.shape[0])
        tokens, maxes, sums = vp_sampling.draft_sample(
            local,
            idx_mapping,
            temperature,
            seeds,
            sample_src_positions,
            col,
            vp,
            use_fp64=self.use_fp64_gumbel,
        )
        est = self.acceptance_estimator
        if est is not None:
            est.predict_from_stats(
                maxes,
                sums,
                idx_mapping,
                draft_step,
                self.draft_token_confidence_probs,
            )
        return tokens

    def set_attn(
        self,
        model_state: ModelState,
        kv_cache_config: KVCacheConfig,
        block_tables: BlockTables,
        target_input_buffers: InputBuffers,
        target_attn_groups: list[list[AttentionGroup]],
    ) -> None:
        super().set_attn(
            model_state,
            kv_cache_config,
            block_tables,
            target_input_buffers,
            target_attn_groups,
        )

        # FlashAttention's AOT split schedule is wrong for a windowed drafter,
        # and `_get_sliding_window_configs` leaves it on or off depending on
        # whether the target also runs FlashAttention. Decide it here instead.
        for groups in self.attn_groups:
            for group in groups:
                builder = group.get_metadata_builder()
                if getattr(
                    builder, "aot_schedule", False
                ) and get_kv_cache_spec_sliding_window(builder.kv_cache_spec):
                    # `aot_schedule` belongs to FlashAttention's builder, not
                    # to the base class this loop is typed against.
                    builder.aot_schedule = False  # type: ignore[attr-defined]

        self.draft_kv_cache_group_ids = [
            gid for gid, g in enumerate(self.attn_groups) if g
        ]
        assert self.draft_kv_cache_group_ids, "No draft attention groups found."
        self.draft_kv_cache_group_id = self.draft_kv_cache_group_ids[0]
        draft_cp_parameters = {
            block_tables.get_group_cp_parameters(gid)
            for gid in self.draft_kv_cache_group_ids
        }
        if len(draft_cp_parameters) != 1:
            raise NotImplementedError(
                "DFlash draft attention groups must use one DCP layout."
            )
        self.draft_cp_rank, self.draft_cp_size, self.draft_cp_interleave = (
            draft_cp_parameters.pop()
        )

        # Per-group context slot buffers for the precompute (one row per group).
        self._context_slot_mappings = torch.zeros(
            len(self.draft_kv_cache_group_ids),
            self.max_num_tokens,
            dtype=torch.int64,
            device=self.device,
        )
        self._draft_query_slot_ptrs = torch.tensor(
            [
                block_tables.slot_mappings[gid].data_ptr()
                for gid in self.draft_kv_cache_group_ids
            ],
            dtype=torch.int64,
            device=self.device,
        )
        self._draft_table_ptrs = torch.tensor(
            [
                block_tables.input_block_tables[gid].data_ptr()
                for gid in self.draft_kv_cache_group_ids
            ],
            dtype=torch.int64,
            device=self.device,
        )
        self._draft_table_strides = torch.tensor(
            [
                block_tables.input_block_tables[gid].stride(0)
                for gid in self.draft_kv_cache_group_ids
            ],
            dtype=torch.int64,
            device=self.device,
        )
        self._draft_block_sizes = torch.tensor(
            [
                block_tables.kernel_block_sizes[gid]
                for gid in self.draft_kv_cache_group_ids
            ],
            dtype=torch.int32,
            device=self.device,
        )

        # Map each draft decoder layer to the index (within draft_kv_cache_group_ids)
        # of the kv-cache group its cache belongs to. Models that share a single group
        # leave this as None and share one context slot mapping.
        self._layer_group_idx: list[int] | None = None
        # Per-KV-group causal, falling back to whether the drafter is all-causal.
        self._group_causal: dict[int, bool] | bool = not self.requires_non_causal
        if hasattr(self.model, "get_draft_kv_cache_layer_names"):
            layer_names = self.model.get_draft_kv_cache_layer_names()
            name_to_gid = {
                ln: gid
                for gid, group in enumerate(kv_cache_config.kv_cache_groups)
                for ln in group.layer_names
            }
            gid_to_idx = {gid: i for i, gid in enumerate(self.draft_kv_cache_group_ids)}
            self._layer_group_idx = [
                gid_to_idx[name_to_gid[name]] for name in layer_names
            ]
            if hasattr(self.model, "get_draft_attn_causal"):
                self._group_causal = {
                    name_to_gid[name]: layer_causal
                    for name, layer_causal in zip(
                        layer_names, self.model.get_draft_attn_causal()
                    )
                }
        self._full_cg_needs_attn_metadata_build = (
            self._compute_full_cg_needs_attn_metadata_build()
        )

    def _compute_full_cg_needs_attn_metadata_build(self) -> bool:
        """Return True if any draft attention builder or model state mutates
        captured state during ``CUDAGraphMode.FULL`` replay."""
        if self.draft_cp_size > 1 or getattr(self.block_tables, "cp_size", 1) > 1:
            return True
        if (
            getattr(
                type(self.model_state),
                "prepare_draft_attn_metadata",
                ModelState.prepare_draft_attn_metadata,
            )
            is not ModelState.prepare_draft_attn_metadata
        ):
            return True
        for groups in self.attn_groups:
            for group in groups:
                builder = group.get_metadata_builder()
                if callable(getattr(builder, "prepare_cudagraph_replay", None)):
                    # The query graph refreshes this builder's device buffers
                    # before attention reads them, including padded requests.
                    continue
                if (
                    type(builder).__name__ != "FlashAttentionMetadataBuilder"
                    or getattr(builder, "aot_schedule", True)
                    or getattr(builder, "dcp_world_size", 1) > 1
                    or getattr(builder, "rswa_window", None) is not None
                    or getattr(builder, "mm_prefix_query_ranges_np", None) is not None
                ):
                    return True
        return False

    def reset_attn(self) -> None:
        """Release DFlash state derived from the target KV-cache layout."""
        # Drop graphs before detaching the KV cache tensors they reference.
        if self._context_preparer is not None:
            close_context = getattr(self._context_preparer, "close", None)
            if callable(close_context):
                close_context()
        self._context_preparer = None
        for name in (
            "draft_kv_cache_group_ids",
            "draft_kv_cache_group_id",
            "draft_cp_rank",
            "draft_cp_size",
            "draft_cp_interleave",
            "_context_slot_mappings",
            "_draft_query_slot_ptrs",
            "_draft_table_ptrs",
            "_draft_table_strides",
            "_draft_block_sizes",
            "_layer_group_idx",
            "_group_causal",
            "_full_cg_needs_attn_metadata_build",
        ):
            if hasattr(self, name):
                delattr(self, name)
        super().reset_attn()

    @torch.inference_mode()
    def _run_model(
        self,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> torch.Tensor:
        batch_descriptor = BatchDescriptor(num_tokens=num_tokens)
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            num_tokens_across_dp=num_tokens_across_dp,
            slot_mapping=slot_mappings,
            batch_descriptor=batch_descriptor,
            is_padding=self._is_padding[:num_tokens],
        ):
            last_hidden_states = self.model(
                input_ids=self.input_buffers.input_ids[:num_tokens],
                positions=self.input_buffers.positions[:num_tokens],
                inputs_embeds=None,
            )
        return last_hidden_states

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        last_hidden_states = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        num_sample = num_reqs * self.num_speculative_steps
        sample_hidden_states = last_hidden_states[self.sample_indices[:num_sample]]
        # sample_pos is the predicted token's position P. Sampling keys a draw
        # by the position before the sampled token, P-1.
        draft_tokens = self.sample_draft(
            sample_hidden_states,
            self.sample_pos[:num_sample] - 1,
            self.sample_idx_mapping[:num_sample],
            self.temperature,
            self.seeds,
            self.sample_col[:num_sample],
            self.draft_logits,
        )
        self.draft_tokens[:num_reqs] = draft_tokens.view(
            num_reqs, self.num_speculative_steps
        )

    def _build_attn_metadata(
        self,
        num_reqs: int,
        batch_desc: BatchExecutionDescriptor,
        query_start_loc_np: np.ndarray,
        seq_lens_cpu_upper_bound: torch.Tensor,
        step: int,
        causal: bool | Mapping[int, bool] = False,
        dcp_local_seq_lens: torch.Tensor | None = None,
        max_query_len: int | None = None,
    ) -> dict[str, Any] | None:
        if not self.draft_attn_layer_names:
            return None
        if dcp_local_seq_lens is None:
            if self.draft_cp_size > 1:
                dcp_local_seq_lens = maybe_prepare_dcp_local_seq_lens(
                    self.input_buffers.dcp_local_seq_lens,
                    self.input_buffers.seq_lens,
                    num_reqs,
                    self.draft_cp_size,
                    self.draft_cp_rank,
                    self.draft_cp_interleave,
                )
            elif getattr(self.block_tables, "cp_size", 1) > 1:
                dcp_local_seq_lens = self.input_buffers.seq_lens
        return super()._build_attn_metadata(
            num_reqs=num_reqs,
            batch_desc=batch_desc,
            query_start_loc_np=query_start_loc_np,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            step=step,
            causal=causal,
            dcp_local_seq_lens=dcp_local_seq_lens,
            max_query_len=max_query_len,
        )

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        # [num_tokens, hidden_size]
        last_hidden_states: torch.Tensor,
        # num_layers x [num_tokens, hidden_size]
        aux_hidden_states: list[torch.Tensor] | None,
        # [num_reqs]
        num_sampled: torch.Tensor,
        # [num_reqs]
        num_rejected: torch.Tensor,
        # [max_num_reqs]
        last_sampled: torch.Tensor,
        # [max_num_reqs]
        next_prefill_tokens: torch.Tensor,
        # [max_num_reqs]
        temperature: torch.Tensor,
        # [max_num_reqs]
        seeds: torch.Tensor,
        dp_sync: DPSyncState | None = None,
        num_speculative_tokens: int | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
        context_kv_is_restored: bool = False,
    ) -> torch.Tensor:
        num_reqs = input_batch.num_reqs
        num_target_tokens = input_batch.num_tokens
        num_query_tokens = num_reqs * self.num_query_per_req

        get_ced_indices = getattr(
            getattr(self, "model_state", None), "get_ced_indices", None
        )
        ced_indices = get_ced_indices() if get_ced_indices is not None else None
        num_context_tokens = (
            num_target_tokens if ced_indices is None else ced_indices.numel()
        )
        if ced_indices is not None:
            from vllm.models.deepseek_v4_1.ced import gather_rows

        # NOTE: To avoid CPU-GPU synchronization without CPU knowing the
        # number of rejected tokens, we maintain the size of input_ids and
        # hidden_states the same as the target model's. This means, we pad each
        # request's query length to include any rejected positions.
        is_streamed_context = getattr(self.model, "is_streamed_context_states", None)
        context_states_are_streamed = bool(
            aux_hidden_states
            and not context_kv_is_restored
            and ced_indices is None
            and callable(is_streamed_context)
            and is_streamed_context(aux_hidden_states)
        )
        context_states = self.hidden_states[:num_context_tokens]
        use_context_graph = (
            not dummy_run
            and ced_indices is None
            and not is_profile
            and not context_kv_is_restored
            and not context_states_are_streamed
            and bool(aux_hidden_states)
            and self._context_preparer is not None
            and self._context_preparer.can_run(num_target_tokens)
        )
        if not context_kv_is_restored and not use_context_graph:
            if ced_indices is not None:
                # Target aux keeps the original-row ABI. Pack before the expensive
                # combine projection; discarded decoder rows have no context KV.
                if aux_hidden_states:
                    aux_hidden_states = [
                        gather_rows(hidden, ced_indices) for hidden in aux_hidden_states
                    ]
                else:
                    last_hidden_states = gather_rows(last_hidden_states, ced_indices)
            combine_into = getattr(self.model, "combine_aux_hidden_states_into", None)
            if context_states_are_streamed:
                assert aux_hidden_states is not None
                context_states = aux_hidden_states[0]
            elif aux_hidden_states and combine_into is not None:
                # The model joins the aux states and writes the combined rows
                # into the context buffer itself.
                combine_into(aux_hidden_states, context_states)
            else:
                if aux_hidden_states:
                    hidden_states = self.model.combine_hidden_states(
                        torch.cat(aux_hidden_states, dim=-1)
                    )
                else:
                    hidden_states = last_hidden_states
                context_states.copy_(hidden_states[:num_context_tokens])

        if dummy_run and skip_attn_for_dummy_run:
            # Memory profiling path: block_tables / kv_cache_config are not initialized.
            # Since DFlash needs to build its own attention metadata, we must skip the
            # preparation in this path and run a minimal forward pass.
            self.model.precompute_and_store_context_kv(
                context_states,
                self.context_positions[:num_context_tokens],
            )
            # DFlash processes all speculative tokens in one forward pass,
            # so the real token count is num_query_tokens.
            self._prepare_eplb_forward(num_query_tokens)
            # A memory-profile batch is all padding, like the target's.
            self._is_padding[:num_query_tokens].fill_(True)
            self._generate_draft(
                num_reqs,
                num_query_tokens,
                attn_metadata=None,
                slot_mappings=None,
                num_tokens_across_dp=None,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            )
            return self.draft_tokens[:num_reqs]

        if self.pcp_manager is not None and not dummy_run:
            self.block_tables.gather_block_tables(
                input_batch.idx_mapping, num_reqs_padded=num_reqs
            )

        # The query slot mapping is written into the shared BlockTables slot_mappings.
        # That buffer's address is what the captured CUDA graph reads from at replay.
        assert self.draft_kv_cache_group_id >= 0
        max_query_slot_tokens = min(
            self.max_num_tokens, self.max_num_reqs * self.num_query_per_req
        )
        if hasattr(self, "draft_cp_rank"):
            cp_rank = self.draft_cp_rank
            cp_size = self.draft_cp_size
            cp_interleave = self.draft_cp_interleave
        else:
            cp_rank, cp_size, cp_interleave = self.block_tables.get_group_cp_parameters(
                self.draft_kv_cache_group_id
            )
        if len(self.draft_kv_cache_group_ids) == 1:
            gid = self.draft_kv_cache_group_ids[0]
            prepare_dflash_inputs(
                self.input_buffers,
                self.block_tables.slot_mappings[gid],
                self.context_positions,
                self._context_slot_mappings[0],
                self.sample_indices,
                self.sample_pos,
                self.sample_idx_mapping,
                self.temperature,
                self.seeds,
                input_batch,
                num_sampled,
                num_rejected,
                last_sampled,
                next_prefill_tokens,
                temperature,
                seeds,
                self.block_tables.input_block_tables[gid],
                self.block_tables.kernel_block_sizes[gid],
                cp_rank,
                cp_size,
                cp_interleave,
                self.parallel_drafting_token_id,
                self.num_query_per_req,
                self.num_speculative_steps,
                self.max_num_reqs,
                max_query_slot_tokens,
                self.max_model_len,
                self.sample_from_anchor,
            )
        else:
            prepare_dflash_inputs_multi_group(
                self.input_buffers,
                self._draft_query_slot_ptrs,
                self.context_positions,
                self._context_slot_mappings,
                self.sample_indices,
                self.sample_pos,
                self.sample_idx_mapping,
                self.temperature,
                self.seeds,
                input_batch,
                num_sampled,
                num_rejected,
                last_sampled,
                next_prefill_tokens,
                temperature,
                seeds,
                self._draft_table_ptrs,
                self._draft_table_strides,
                self._draft_block_sizes,
                cp_rank,
                cp_size,
                cp_interleave,
                self.parallel_drafting_token_id,
                self.num_query_per_req,
                self.num_speculative_steps,
                self.max_num_reqs,
                max_query_slot_tokens,
                self.max_model_len,
                self.sample_from_anchor,
            )

        if use_context_graph:
            assert self._context_preparer is not None
            self._context_preparer.run(aux_hidden_states, num_target_tokens)
        elif not context_kv_is_restored:
            context_positions = self.context_positions[:num_target_tokens]
            packed_context_slots = None
            if ced_indices is not None:
                # Keep query anchors and rejection masking in the ORIGINAL target
                # coordinates. Only the prepared context inputs are packed afterward.
                context_positions = gather_rows(self.context_positions, ced_indices)
                from vllm.models.deepseek_v4_1.tilelang import ced as ced_kernels

                packed_context_slots = [
                    ced_kernels.gather_slots(slots, ced_indices)
                    for slots in self._context_slot_mappings
                ]
            # Each layer uses the context slots of its own KV-cache group. V4.1 can
            # replay bounded decode context graphs; compact prefills use eager
            # projection.
            # Dummy runs reserve workspace without clobbering real cache entries.
            if dummy_run:
                context_slots: torch.Tensor | list[torch.Tensor | None] | None = None
            else:
                group_slots = (
                    packed_context_slots
                    if packed_context_slots is not None
                    else [
                        slots[:num_context_tokens]
                        for slots in self._context_slot_mappings
                    ]
                )
                if self._layer_group_idx is not None:
                    context_slots = [
                        group_slots[gidx] for gidx in self._layer_group_idx
                    ]
                else:
                    context_slots = group_slots[0]
            self.model.precompute_and_store_context_kv(
                context_states,
                context_positions,
                context_slots,
            )

        if num_speculative_tokens == 0:
            # No drafts are needed: the context K/V has been stored above to
            # keep the drafter in sync, so skip the query pass.
            return self.draft_tokens[:num_reqs]

        batch_sync, num_batch_tokens = (
            self._build_uniform_batch_dp_sync(dp_sync, num_reqs, self.num_query_per_req)
            if dp_sync is not None
            else (None, num_query_tokens)
        )
        # Every DFlash step has exactly num_query_per_req tokens, so we can use FULL CGs
        batch_desc, batch_sync = dispatch_cg_and_sync_dp(
            self.query_cudagraph_manager,
            num_reqs,
            num_batch_tokens,
            uniform_token_count=self.num_query_per_req,
            dp_size=self.dp_size,
            dp_rank=self.dp_rank,
            need_eager=is_profile,
            dp_sync=batch_sync,
        )
        num_tokens_padded = batch_desc.num_tokens

        # DFlash processes all speculative tokens in one forward pass,
        # so the real token count is num_query_tokens.
        self._prepare_eplb_forward(num_query_tokens)

        # Rows past the live query rows pad the batch, and a dummy batch is all
        # padding, like the target's. Written eagerly before the graph replays,
        # so the routers skip exactly these rows.
        from vllm.v1.worker.gpu import tilelang_inputs

        live = 0 if dummy_run else num_query_tokens
        tilelang_inputs.padding_rows(self._is_padding, live, num_tokens_padded)

        if batch_desc.cg_mode == CUDAGraphMode.FULL:
            if getattr(self, "_full_cg_needs_attn_metadata_build", True):
                max_seq_len = int(input_batch.seq_lens_cpu_upper_bound[:num_reqs].max())
                self.draft_max_seq_len = min(
                    max_seq_len + self.num_query_per_req, self.max_model_len
                )
                self._build_uniform_attn_metadata(
                    num_reqs=num_reqs,
                    batch_desc=batch_desc,
                    num_query_per_req=self.num_query_per_req,
                    seq_lens_cpu_upper_bound=input_batch.seq_lens_cpu_upper_bound,
                    step=self.num_query_per_req,
                    causal=self._group_causal,
                )
            assert self.query_cudagraph_manager is not None
            self.query_cudagraph_manager.run_fullgraph(batch_desc)
        else:
            max_seq_len = int(input_batch.seq_lens_cpu_upper_bound[:num_reqs].max())
            self.draft_max_seq_len = min(
                max_seq_len + self.num_query_per_req, self.max_model_len
            )
            draft_attn_metadata = self._build_uniform_attn_metadata(
                num_reqs=num_reqs,
                batch_desc=batch_desc,
                num_query_per_req=self.num_query_per_req,
                seq_lens_cpu_upper_bound=input_batch.seq_lens_cpu_upper_bound,
                step=self.num_query_per_req,
                causal=self._group_causal,
            )
            draft_slot_mappings_by_layer = build_slot_mappings_by_layer(
                self.block_tables.slot_mappings[:, :num_tokens_padded],
                self.kv_cache_config,
            )
            num_tokens_across_dp = (
                batch_sync.num_tokens_across_dp if batch_sync is not None else None
            )
            self._generate_draft(
                num_reqs,
                num_tokens_padded,
                draft_attn_metadata,
                draft_slot_mappings_by_layer,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=batch_desc.cg_mode,
            )

        return self.draft_tokens[:num_reqs]


# The Triton reference's BLOCK_SIZE (rows a program), for its launch in the tests.
_PREPARE_BLOCK_SIZE = 256


def prepare_dflash_inputs(
    input_buffers: InputBuffers,
    query_slot_mapping: torch.Tensor,
    context_positions: torch.Tensor,
    context_slot_mapping: torch.Tensor,
    sample_indices: torch.Tensor,
    sample_pos: torch.Tensor,
    sample_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    seeds: torch.Tensor,
    input_batch: InputBatch,
    # [num_reqs]
    num_sampled: torch.Tensor,
    # [num_reqs]
    num_rejected: torch.Tensor,
    # [max_num_reqs]
    last_sampled: torch.Tensor,
    # [max_num_reqs]
    next_prefill_tokens: torch.Tensor,
    # [max_num_reqs]
    input_temperature: torch.Tensor,
    # [max_num_reqs]
    input_seeds: torch.Tensor,
    # [max_num_reqs, max_num_blocks]
    block_table: torch.Tensor,
    block_size: int,
    cp_rank: int,
    cp_size: int,
    cp_interleave: int,
    parallel_drafting_token_id: int,
    num_query_per_req: int,
    num_speculative_steps: int,
    max_num_reqs: int,
    max_num_tokens: int,
    max_model_len: int,
    sample_from_anchor: bool = False,
) -> None:
    num_reqs = input_batch.num_reqs
    assert num_reqs > 0
    # Cover the longest possible per-request span (ctx + query). Use the max
    # per-request query length, not the total token count across the batch.
    max_target_query_len = int(input_batch.num_scheduled_tokens.max())
    max_tokens_per_req = max_target_query_len + num_query_per_req
    from vllm.v1.worker.gpu import tilelang_inputs

    # Live context/query lengths control the grid, never the compiled kernel.
    tilelang_inputs.prepare_dflash_inputs(
        input_buffers.input_ids,
        input_buffers.positions,
        input_buffers.query_start_loc,
        input_buffers.seq_lens,
        query_slot_mapping,
        context_positions,
        context_slot_mapping,
        sample_indices,
        sample_pos,
        sample_idx_mapping,
        temperature,
        seeds,
        input_batch.positions,
        input_batch.query_start_loc,
        input_batch.idx_mapping,
        last_sampled,
        next_prefill_tokens,
        num_sampled,
        num_rejected,
        input_temperature,
        input_seeds,
        block_table,
        parallel_drafting_token_id,
        block_size,
        num_query_per_req,
        num_speculative_steps,
        max_num_reqs,
        max_num_tokens,
        max_model_len,
        cp_rank,
        cp_size,
        cp_interleave,
        sample_from_anchor,
        num_reqs,
        max_tokens_per_req,
    )


def prepare_dflash_inputs_multi_group(
    input_buffers: InputBuffers,
    query_slot_ptrs: torch.Tensor,
    context_positions: torch.Tensor,
    context_slot_mappings: torch.Tensor,
    sample_indices: torch.Tensor,
    sample_pos: torch.Tensor,
    sample_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    seeds: torch.Tensor,
    input_batch: InputBatch,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    last_sampled: torch.Tensor,
    next_prefill_tokens: torch.Tensor,
    input_temperature: torch.Tensor,
    input_seeds: torch.Tensor,
    table_ptrs: torch.Tensor,
    table_strides: torch.Tensor,
    block_sizes: torch.Tensor,
    cp_rank: int,
    cp_size: int,
    cp_interleave: int,
    parallel_drafting_token_id: int,
    num_query_per_req: int,
    num_speculative_steps: int,
    max_num_reqs: int,
    max_num_tokens: int,
    max_model_len: int,
    sample_from_anchor: bool = False,
) -> None:
    num_reqs = input_batch.num_reqs
    assert num_reqs > 0
    max_target_query_len = int(input_batch.num_scheduled_tokens.max())
    max_tokens_per_req = max_target_query_len + num_query_per_req
    from vllm.v1.worker.gpu import tilelang_inputs

    tilelang_inputs.prepare_dflash_inputs_multi_group(
        input_buffers.input_ids,
        input_buffers.positions,
        input_buffers.query_start_loc,
        input_buffers.seq_lens,
        query_slot_ptrs,
        context_positions,
        context_slot_mappings,
        sample_indices,
        sample_pos,
        sample_idx_mapping,
        temperature,
        seeds,
        input_batch.positions,
        input_batch.query_start_loc,
        input_batch.idx_mapping,
        last_sampled,
        next_prefill_tokens,
        num_sampled,
        num_rejected,
        input_temperature,
        input_seeds,
        table_ptrs,
        table_strides,
        block_sizes,
        parallel_drafting_token_id,
        num_query_per_req,
        num_speculative_steps,
        max_num_reqs,
        max_num_tokens,
        max_model_len,
        cp_rank,
        cp_size,
        cp_interleave,
        sample_from_anchor,
        num_reqs,
        max_tokens_per_req,
    )


_TRITON_REFS = frozenset(("_prepare_dflash_inputs_kernel",))


def __getattr__(name: str):
    if name in _TRITON_REFS:
        from vllm.v1.worker.gpu import _triton_refs

        return getattr(_triton_refs, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
