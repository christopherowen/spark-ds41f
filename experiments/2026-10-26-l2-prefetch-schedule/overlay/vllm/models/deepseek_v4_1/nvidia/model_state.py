# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from vllm.compilation.breakable_cudagraph import is_breakable_cudagraph_enabled
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.models.deepseek_v4_1.ced import (
    CED_WINDOW,
    CEDState,
    ced_decoder_start,
    replay_rows,
)
from vllm.models.deepseek_v4_1.tilelang import ced as ced_kernels
from vllm.models.deepseek_v4_1.sparse_mla import DeepseekV41B12xMetadata
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.utils import AttentionGroup


class DeepseekV41ModelState(DefaultModelState):
    """DefaultModelState plus the engram lookback window.

    The engram n-gram hash needs the ids of the ``depth`` tokens preceding
    each request's chunk start (see ``common/engram.py``). The runner keeps
    the full token history on device, so the window is gathered there every
    step: exact for prompt and generated tokens alike, whatever instance
    produced their KV.

    With disk Engram tables, the host also keeps each request's prefill tokens
    and hashes the rows of a step's prefill chunks when it prepares the step,
    a step before the GPU runs it, so the reader can read them while the
    previous step computes (``engram_rows/host_hash.py``). Rows of tokens the
    GPU samples are hashed on the GPU as before; one step can hold both.
    """

    _ced_streaming_prefix_start: tuple[str, int] | None = None

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ):
        super().__init__(vllm_config, model, encoder_cache, device)
        self.ced_state: CEDState | None = None
        self._ced_prompt_logprobs: set[str] = set()
        self._ced_prefix_start: dict[str, int] = {}
        self._ced_batch: InputBatch | None = None
        self.max_cudagraph_query_len: int | None = None
        if ced_decoder_start(self.model_config.hf_config) is not None:
            compact_max = min(self.max_num_tokens, self.max_num_reqs * CED_WINDOW)
            # Compaction changes the model's optional-input branch; FULL graphs
            # remain valid for every short/varlen decode split, never long chunks
            # or a short chunk that compacts (compacts_short_queries).
            self.max_cudagraph_query_len = CED_WINDOW
            capacities = tuple(
                sorted(
                    {
                        self.max_num_tokens,
                        compact_max,
                        *range(CED_WINDOW, compact_max + 1, CED_WINDOW),
                        *(
                            size
                            for size in (
                                vllm_config.compilation_config.cudagraph_capture_sizes
                                or ()
                            )
                            if 0 < size <= self.max_num_tokens
                        ),
                    }
                )
            )
            self.ced_state = CEDState(
                self.max_num_reqs, self.max_num_tokens, capacities, device
            )
            hf = self.model_config.hf_config
            parallel = vllm_config.parallel_config
            self.ced_state.warmup(
                hf.hidden_size,
                hf.hc_mult,
                pipeline=parallel.pipeline_parallel_size > 1,
                sink_columns=model.get_sink_residual_columns() is not None,
            )
            # The drafter (loaded before the model state) keeps context only for
            # the decoder's kept rows: its windows start at the same replay bound,
            # read from the buffer its captured graphs were recorded with.
            from vllm.models.deepseek_v4_1.attention import DeepseekV4Attention

            for layer in get_layers_from_vllm_config(
                vllm_config, DeepseekV4Attention
            ).values():
                if layer.is_draft:
                    layer.bind_draft_replay_start(self.ced_state.replay_start)
            if (
                is_breakable_cudagraph_enabled()
                and self.max_num_tokens == 4096
                and parallel.decode_context_parallel_size == 1
                and parallel.pipeline_parallel_size == 1
                and parallel.data_parallel_size == 1
                and vllm_config.lora_config is None
            ):
                self.single_request_prefill_cudagraph_tokens = 4096
        depth = model.token_lookback_depth
        self.lookback_token_ids: torch.Tensor | None = None
        if depth > 0:
            # Persistent so a captured graph can read it on replay.
            self.lookback_token_ids = torch.full(
                (self.max_num_reqs, depth), -1, dtype=torch.int32, device=device
            )
            # Rows of every batch size share one kernel; the runner's
            # batch-to-request-state index is int64 (np.intp).
            from vllm.models.deepseek_v4_1.tilelang import lookback, prepare

            prepare.declare(
                lambda: [
                    prepare.call(
                        lookback.gather_lookback_kernel,
                        *lookback.gather_lookback_key(depth, "int64"),
                    )
                ]
            )
        from .model import DeepseekV4Model

        self.disk_engram_models = tuple(
            module
            for module in model.modules()
            if isinstance(module, DeepseekV4Model) and module.disk_engram
        )
        # Disk Engram: each live request's prefill tokens (prompt, and on resumption
        # the outputs it recomputes), for the host's hash of its prefill chunks.
        self._prefill_tokens: dict[str, np.ndarray] = {}
        self._host_engram_hash = None

    def prepare_streaming_update(self, req_id: str) -> None:
        """Keep decoder history when the runner refreshes a live request."""
        self._ced_streaming_prefix_start = (
            req_id,
            self._ced_prefix_start.get(req_id, 0),
        )

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        super().add_request(req_index, new_req_data)
        if self.disk_engram_models and new_req_data.prefill_token_ids is not None:
            self._prefill_tokens[new_req_data.req_id] = np.asarray(
                new_req_data.prefill_token_ids, dtype=np.int32
            )
        sampling = new_req_data.sampling_params
        if sampling is not None and sampling.prompt_logprobs is not None:
            self._ced_prompt_logprobs.add(new_req_data.req_id)
        else:
            self._ced_prompt_logprobs.discard(new_req_data.req_id)
        # Fresh decoder pages lack cached history; streaming updates retain
        # the live request's private pages and therefore its original boundary.
        pending = self._ced_streaming_prefix_start
        streaming = pending is not None and pending[0] == new_req_data.req_id
        prefix_start = pending[1] if streaming else new_req_data.num_computed_tokens
        self._ced_streaming_prefix_start = None
        self._ced_prefix_start[new_req_data.req_id] = prefix_start

    def remove_request(self, req_id: str) -> None:
        super().remove_request(req_id)
        self._prefill_tokens.pop(req_id, None)
        self._ced_prompt_logprobs.discard(req_id)
        self._ced_prefix_start.pop(req_id, None)

    def compacts_short_queries(self, batch) -> bool:
        """Whether CED compacts a request whose chunk fits the graphs' query bound
        (``max_cudagraph_query_len``): a prefill chunk of at most ``CED_WINDOW`` rows
        that keeps fewer, such as the first chunk after a prefix hit just before a
        chunk boundary. A compacted step runs eagerly; ``batch`` is the runner's
        request state before dispatch."""
        if self.ced_state is None or batch is None or not batch.has_prefill:
            return False
        keep_full, _, limits = self._ced_rows(batch)
        return any(
            not full and limit < q <= CED_WINDOW
            for q, full, limit in zip(batch.num_scheduled_tokens, keep_full, limits)
        )

    def can_use_single_request_prefill_graph(self, num_reqs, num_tokens, req_ids):
        return (
            self.single_request_prefill_cudagraph_tokens > 0
            and num_reqs == 1
            and num_tokens == self.single_request_prefill_cudagraph_tokens
            and not any(req in self._ced_prompt_logprobs for req in req_ids)
        )

    def finalize_cudagraph_inputs(self, model_inputs, cg_mode):
        if cg_mode == CUDAGraphMode.PIECEWISE:
            # prepare_attn stages CED after prepare_dummy_inputs. Capture the
            # same compact branch and persistent index buffer used by serving.
            model_inputs["ced_indices"] = self.get_ced_indices()

    def get_ced_indices(self) -> torch.Tensor | None:
        return None if self.ced_state is None else self.ced_state.get_indices()

    def _ced_rows(self, batch) -> tuple[list[bool], list[int], list[int]]:
        """``keep_full`` and ``replay_rows``' floors and limits of a batch's requests:
        an ``InputBatch``, or the runner's request state before dispatch (the same
        fields)."""
        req_ids = batch.req_ids
        nr = len(req_ids)
        keep_full = [req_id in self._ced_prompt_logprobs for req_id in req_ids]
        floors, limits = replay_rows(
            batch.num_scheduled_tokens[:nr],
            batch.num_computed_prefill_tokens_np[:nr],
            batch.prefill_len_np[:nr],
            [self._ced_prefix_start.get(req_id, 0) for req_id in req_ids],
            keep_full,
        )
        return keep_full, floors, limits

    def _stage_ced(self, input_batch: InputBatch, *, force_full: bool = False) -> None:
        state = self.ced_state
        if state is None:
            return
        nr = input_batch.num_reqs
        if force_full:
            keep_full, floors, limits = [True] * nr, [0] * nr, None
        else:
            keep_full, floors, limits = self._ced_rows(input_batch)
        state.stage(
            input_batch.query_start_loc,
            input_batch.seq_lens,
            input_batch.num_scheduled_tokens[:nr],
            keep_full,
            floors,
            limits,
        )
        self._ced_batch = input_batch

    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
    ) -> dict[str, Any]:
        # Runtime prepares attention before model inputs. Capture does the reverse;
        # both paths populate the same persistent packing buffers.
        self._stage_ced(input_batch, force_full=for_capture)
        metadata = super().prepare_attn(
            input_batch,
            cudagraph_mode,
            block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
            for_capture,
        )
        state = self.ced_state
        if state is not None:
            # The decoder's rows are the step's rows gathered by CED's indices.
            # Adaptive verification marks dead rows inside the batch, and the
            # plan's CPU upper bound leaves -1 indices at its end, so the
            # step's padding mask is gathered, not truncated. The mask is a new
            # tensor each step: CED steps must not replay a CUDA graph.
            indices = state.get_indices()
            routing_mask = None
            if indices is not None:
                if cudagraph_mode != CUDAGraphMode.NONE:
                    raise RuntimeError("CED decoder routing requires eager execution")
                routing_mask = ced_kernels.routing_mask(indices, input_batch.is_padding)
            seen: set[int] = set()
            for value in metadata.values():
                if isinstance(value, DeepseekV41B12xMetadata) and id(value) not in seen:
                    seen.add(id(value))
                    value.swa_replay_start = state.replay_start
                    if state.plan is not None:
                        value.decoder = state.decoder_metadata(value)
                        value.decoder.routing_mask = routing_mask
        return metadata

    def prepare_inputs(
        self, input_batch: InputBatch, req_states: RequestState
    ) -> dict[str, torch.Tensor | None]:
        model_inputs = super().prepare_inputs(input_batch, req_states)
        if self.ced_state is not None:
            if self._ced_batch is not input_batch:
                self._stage_ced(input_batch)
            model_inputs["ced_indices"] = self.get_ced_indices()
        window = self.lookback_token_ids
        if window is None:
            return model_inputs
        from vllm.models.deepseek_v4_1.tilelang.lookback import gather_lookback

        # Every row: the batch's requests' windows, then -1.
        gather_lookback(
            window,
            input_batch.idx_mapping,
            req_states.num_computed_tokens.gpu,
            req_states.all_token_ids.gpu,
        )
        model_inputs["lookback_token_ids"] = window
        for model in self.disk_engram_models:
            model.prepare_disk_engram(
                input_batch.input_ids,
                input_batch.query_start_loc,
                window,
                self._early_engram_ids(model, input_batch),
            )
        return model_inputs

    def _early_engram_ids(self, model, input_batch: InputBatch) -> np.ndarray | None:
        """The step's Engram row IDs the host can hash now, ``[tokens, layers, 24]``
        (``host_hash.UNKNOWN`` for rows of tokens the GPU samples, -1 for padding),
        or None when the step has no prefill chunk to hash."""
        if not input_batch.has_prefill:
            return None
        from vllm.models.deepseek_v4_1.engram_rows.host_hash import HostEngramHash

        if self._host_engram_hash is None:
            layout = model.engram_hash.layout
            geometry = layout.geometry
            self._host_engram_hash = HostEngramHash(
                layout.token_map,
                geometry.multipliers,
                geometry.primes,
                geometry.offsets,
            )
        starts = input_batch.query_start_loc_np
        requests = []
        for i, req_id in enumerate(input_batch.req_ids):
            first, rows = int(starts[i]), int(input_batch.num_scheduled_tokens[i])
            tokens = self._prefill_tokens.get(req_id)
            positions = np.empty(0, dtype=np.int64)
            if tokens is not None and input_batch.is_prefilling_np[i]:
                start = int(input_batch.num_computed_prefill_tokens_np[i])
                positions = np.arange(
                    start, min(start + rows, len(tokens)), dtype=np.int64
                )
            requests.append((first, rows, tokens, positions))
        hashed = [i for i, (*_, positions) in enumerate(requests) if len(positions)]
        if not hashed:
            return None
        # Adaptive verification gives verifying requests row counts the GPU chooses,
        # but keeps them at the front of the batch with the host's total, so every
        # later request starts at the same row on the host and the GPU.
        drafts = input_batch.num_draft_tokens_per_req
        if drafts is not None and drafts[hashed[0] :].any():
            raise RuntimeError(
                "Engram host hash: a verifying request follows a prefill chunk, so "
                "the chunk's rows are placed by the GPU"
            )
        return self._host_engram_hash.step_ids(input_batch.input_ids.shape[0], requests)

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        model_inputs = super().prepare_dummy_inputs(num_reqs, num_tokens)
        if self.ced_state is not None:
            # FULL descriptors may pad a short request far beyond128 rows.
            # Capture their noncompact branch, regardless of dummy split;
            # long real queries are forced off FULL graphs by the runner.
            self.ced_state.plan = None
            self.ced_state.replay_start.zero_()
            self._ced_batch = None
            model_inputs["ced_indices"] = None
        if self.lookback_token_ids is not None:
            # The captured graph reads this buffer; replays refill it in place.
            self.lookback_token_ids.fill_(-1)
            model_inputs["lookback_token_ids"] = self.lookback_token_ids
        for model in self.disk_engram_models:
            model.prepare_dummy_engram(num_tokens)
        return model_inputs
