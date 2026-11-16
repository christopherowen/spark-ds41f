# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adaptive verification for DSpark speculative decoding."""

import os
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import init_logger
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.gpu import tilelang_inputs
from vllm.v1.worker.gpu.async_utils import stream
from vllm.v1.worker.gpu.attn_utils import (
    get_attn_cg_support,
    get_query_lens_mismatch_unsupported_backend,
)
from vllm.v1.worker.gpu.input_batch import DeadRows

logger = init_logger(__name__)
# Dead rows: verification rows past the first draft whose survival (running
# product of this step's confidences) falls below this threshold are marked
# as padding, so routed MoE skips them, and the request commits at most one
# token past its last live row. Decided on device, inside the draft combine,
# from the current drafts' confidences; the batch shape is unchanged. 0 = off.
_DEAD_ROWS_TAU = float(os.environ.get("SPARK_DS41F_DSPARK_DEAD_ROWS_TAU", "0"))


def dead_rows_enabled() -> bool:
    """Whether an adaptive verification manager marks dead verification rows."""
    return _DEAD_ROWS_TAU > 0


if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.worker.gpu.attn_utils import AttentionCGSupportInfo
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.states import RequestState
    from vllm.v1.worker.utils import AttentionGroup


# Consecutive steps that may schedule drafts yet verify none before one step
# verifies a draft per request anyway (see ``get_num_tokens``).
EXPLORE_AFTER_IDLE_STEPS = 32

# Multi-regime verification row-cost curve on SM120 / SM121 (calibrated to the
# tp4-ring 4-boot engine grid):
# 1) rows <= 16: single 16-row Tensor Core MMA tile (block_M=16) and early
#    MoE PDL trigger (slope 1.0x).
# 2) 17 <= rows <= 32: second 16-row MMA tile (decode_32) shares active MoE
#    expert tiles at a 0.55x marginal slope. A single request spilling past
#    16 rows (copy-draft burst) has no cross-request expert coalescing and
#    pays the 1.0x slope plus SINGLE_REQ_TILE_STEP_ROWS.
# 3) 33 <= rows <= 64: third and fourth 16-row MMA tiles (decode_64). When at
#    least COALESCED_MIN_REQS (10) requests are active, additional draft rows
#    coalesce onto already-active MoE experts and the single decode_64 launch
#    at COALESCED_ROW_COST_SCALE (0.28x, matching the tp4-ring grid slope).
# 4) rows > 64: dense projections exceed DECODE_ROWS=64 into prefill_128 /
#    multi-tile waves and attention indexers split into a second 64-row
#    DECODE_CHUNK, incurring the tp4-ring second-wave boundary step
#    (SECOND_WAVE_STEP_ROWS = 6.714 row units, or 3.301 ms) at
#    SATURATED_ROW_COST_SCALE = 0.28.
HIGH_ROW_COST_KNEE = 16
HIGH_ROW_COST_SCALE = 0.55
COALESCED_ROW_COST_KNEE = 32
COALESCED_ROW_COST_SCALE = 0.28
COALESCED_MIN_REQS = 10
SATURATED_ROW_COST_KNEE = 64
SATURATED_ROW_COST_SCALE = 0.28
MMA_TILE_STEP_ROWS = 0.0
SINGLE_REQ_TILE_STEP_ROWS = 1.0
SECOND_WAVE_STEP_ROWS = 6.714
COHORT_CONFIDENCE_TOL = 0.025

# Context-length marginal row-cost scaling: on tp4-ring, Sparse-MLA (top-k=2048)
# and the TP-sharded 8-layer Indexer keep step time flat through 262,144 tokens
# (64K index positions per rank). Above CONTEXT_ROW_COST_KNEE (262,144 tokens,
# or when overridden for single-rank profiles), excess per-row index scan cost
# scales smoothly up to CONTEXT_MAX_ROW_COST_SCALE.
CONTEXT_ROW_COST_KNEE = 262144
CONTEXT_REINDEX_SATURATION = 524288
CONTEXT_REINDEX_WEIGHT = 0.12
CONTEXT_SOURCE_SPAN = 524288
CONTEXT_SOURCE_SLOPE = 0.22
CONTEXT_MAX_ROW_COST_SCALE = 2.25


def resolve_adaptive_cudagraph_mode(
    mode: CUDAGraphMode, *, piecewise_capture_available: bool
) -> CUDAGraphMode:
    """Select a separate decode route for variable-length verification."""
    assert mode.has_full_cudagraphs()
    if mode == CUDAGraphMode.FULL_DECODE_ONLY or not piecewise_capture_available:
        return CUDAGraphMode.FULL_DECODE_ONLY
    return CUDAGraphMode.FULL_AND_PIECEWISE


def preparation_tail_sizes(
    capture_sizes: Iterable[int], max_num_tokens: int
) -> tuple[int, ...]:
    """Exact eager token counts past the capture limit, warmed once at startup."""
    capture_sizes = tuple(capture_sizes)
    size = capture_sizes[-1] if capture_sizes else 0
    tail_sizes = set()
    if size:
        tail_sizes.add(min(size + size // 2, max_num_tokens))
        while size < max_num_tokens:
            size = min(size * 2, max_num_tokens)
            tail_sizes.add(size)
        tail_sizes -= set(capture_sizes)
    return tuple(sorted(tail_sizes))


class AdaptiveVerificationManager:
    def __init__(
        self,
        req_states: "RequestState",
        query_start_loc: torch.Tensor,
        num_bonus_tokens: int,
        max_total_logits: int,
        row_cost: float,
        planned_confidences: bool = False,
        num_draft_steps: int | None = None,
        *,
        tp_invariant_confidence: bool = False,
        calibrated_confidence: bool | Callable[[], bool] = False,
    ):
        self.req_states = req_states
        self.num_speculative_steps = req_states.num_speculative_steps
        self.num_draft_steps = num_draft_steps or self.num_speculative_steps
        self.step_steps = self.num_speculative_steps
        device = req_states.device
        self._copy_stream = torch.cuda.Stream(device)

        self.num_bonus_tokens = num_bonus_tokens
        # Rejection sampling verifies logits in one contiguous chunk; the
        # chunked path indexes by scheduled (untrimmed) offsets and cannot
        # address the compacted layout, so the budget must fit one chunk.
        self._max_total_logits = max_total_logits
        # The cost of one more verification row as a fraction of a step's fixed
        # cost: a recipe constant, so the budget is a rule of the inputs, the same
        # on every build and boot, with nothing timed.
        self.row_cost = row_cost
        self.high_row_cost_knee = HIGH_ROW_COST_KNEE
        self.high_row_cost_scale = HIGH_ROW_COST_SCALE
        self.coalesced_row_cost_knee = COALESCED_ROW_COST_KNEE
        self.coalesced_row_cost_scale = COALESCED_ROW_COST_SCALE
        self.coalesced_min_reqs = COALESCED_MIN_REQS
        self.saturated_row_cost_knee = SATURATED_ROW_COST_KNEE
        self.saturated_row_cost_scale = SATURATED_ROW_COST_SCALE
        self.mma_tile_step_rows = MMA_TILE_STEP_ROWS
        self.single_req_tile_step_rows = SINGLE_REQ_TILE_STEP_ROWS
        self.second_wave_step_rows = SECOND_WAVE_STEP_ROWS
        self.context_row_cost_knee = CONTEXT_ROW_COST_KNEE
        self.context_reindex_saturation = CONTEXT_REINDEX_SATURATION
        self.context_reindex_weight = CONTEXT_REINDEX_WEIGHT
        self.context_source_span = CONTEXT_SOURCE_SPAN
        self.context_source_slope = CONTEXT_SOURCE_SLOPE
        self.context_max_row_cost_scale = CONTEXT_MAX_ROW_COST_SCALE
        self._batch_context_scale = 1.0
        self.tp_invariant_confidence = tp_invariant_confidence
        self.calibrated_confidence = calibrated_confidence
        self.cohort_confidence_tol = COHORT_CONFIDENCE_TOL
        self._capacities_cpu_override: dict[str, int] | None = None
        self._cohort_locked = False
        self.dead_rows_tau = _DEAD_ROWS_TAU
        self._has_admitted_drafts = False
        self._dead_row_cut = torch.empty(
            req_states.max_num_reqs, dtype=torch.int32, device=device
        )
        if self.dead_rows_tau > 0:
            logger.info(
                "DSpark dead verification rows below survival %.3g", self.dead_rows_tau
            )
        self.query_start_loc = query_start_loc
        self._batch_budget: tuple[dict[str, int], dict[str, int], int] | None = None
        max_num_reqs = req_states.max_num_reqs
        # Current per-slot confidences, and with copy drafts (planned_confidences)
        # the confidences a later step's budget reads: a copy's are those of the
        # proposal after the current one, whose rows the scheduler lays out from
        # the same copy state. One buffer, so one broadcast keeps both TP-wide.
        self._confidence_buffer = torch.empty(
            (1 + planned_confidences, max_num_reqs, self.num_speculative_steps),
            dtype=torch.float32,
            device=device,
        )
        self._confidence_probs = self._confidence_buffer[0]
        self._planned_probs = self._confidence_buffer[-1]
        self._batch_draft_capacity = torch.empty(
            max_num_reqs, dtype=torch.int32, device=device
        )
        self._num_non_draft_tokens = torch.empty_like(query_start_loc[:-1])
        self._cu_num_logits = torch.empty_like(query_start_loc)

        # Two D2H slots preserve stale inputs for budget selection.
        self._stale_confidences = [
            CpuGpuBuffer(
                max_num_reqs,
                self.num_speculative_steps,
                dtype=torch.float32,
                device=device,
            )
            for _ in range(2)
        ]
        self._copy_events = [torch.cuda.Event(blocking=True) for _ in range(2)]
        self._pending_resets: list[int] = []
        # The new requests' confidence rows, set to 1 together (apply_staged_writes).
        self._new_rows: list[int] = []
        self._stale_idx = 0
        for slot in self._stale_confidences:
            slot.np.fill(1.0)

    def add_request(self, req_idx: int) -> None:
        self._stale_confidences[self._stale_idx].np[req_idx].fill(1.0)
        self._pending_resets.append(req_idx)
        self._new_rows.append(req_idx)

    def apply_staged_writes(self) -> None:
        """Set the new requests' confidences to 1 in one launch."""
        if self._new_rows:
            rows = async_tensor_h2d(
                self._new_rows, dtype=torch.int64, device=self._confidence_probs.device
            )
            tilelang_inputs.fill_rows(rows, self._confidence_probs, 1.0)
            if self._confidence_buffer.shape[0] > 1:
                # Nothing of the slot's previous request reaches a later budget.
                tilelang_inputs.fill_rows(rows, self._planned_probs, 1.0)
            self._new_rows.clear()

    def warm_tail_shapes(
        self, run_dummy: Callable[..., object], capture_sizes: list[int]
    ) -> None:
        """Run each eager token count past the capture limit once, untimed, so its
        kernels compile before the first long prefill does."""
        max_num_tokens = self.req_states.max_num_batched_tokens
        for num_tokens in preparation_tail_sizes(capture_sizes, max_num_tokens):
            run_dummy(num_tokens=num_tokens)
            if torch.cuda.is_available():
                torch.accelerator.synchronize()
                torch.accelerator.empty_cache()

    def load_draft_budget_kernels(self) -> None:
        """Run the draft-budget selection once at its smallest and largest shapes,
        so its kernels are loaded at startup rather than on the first step that
        trims a budget."""
        device = self._confidence_probs.device
        num_draft_steps = getattr(self, "num_draft_steps", self.num_speculative_steps)
        probs = torch.ones_like(self._confidence_probs)
        max_reqs = probs.shape[0]
        for steps in dict.fromkeys((num_draft_steps, self.num_speculative_steps)):
            step_probs = probs[:, :steps]
            for num_reqs, draft_budget in ((1, 1), (max_reqs, max_reqs * steps - 1)):
                if not 0 < draft_budget < num_reqs * steps:
                    continue
                capacities = torch.full(
                    (num_reqs,),
                    steps,
                    dtype=self._batch_draft_capacity.dtype,
                    device=device,
                )
                idx_mapping = torch.arange(num_reqs, dtype=torch.int64, device=device)
                tilelang_inputs.assign_draft_budget(
                    step_probs, idx_mapping, capacities, draft_budget, steps
                )
        torch.cuda.synchronize(device)

    def record_confidences(
        self,
        confidence_probs: torch.Tensor,
        input_batch: "InputBatch",
        planned: torch.Tensor | None = None,
    ) -> None:
        """Publish this step's raw confidences for the ranking kernel and start
        copying them to the CPU, where a later step's budget reads them; with
        copy drafts, ``planned`` holds what that budget reads instead."""
        ready_idx = self._stale_idx ^ 1
        with gpu_sync_allowed():
            self._copy_events[ready_idx].synchronize()
        if self._pending_resets:
            self._stale_confidences[ready_idx].np[self._pending_resets] = 1.0
            self._pending_resets.clear()
        # Last step's copy has landed: budgets read it, this step overwrites the
        # slot they were reading before.
        self._stale_idx, write_idx = ready_idx, self._stale_idx

        tilelang_inputs.scatter_rows(
            input_batch.idx_mapping, confidence_probs, self._confidence_probs
        )
        assert (planned is not None) == (self._confidence_buffer.shape[0] > 1)
        if planned is not None:
            tilelang_inputs.scatter_rows(
                input_batch.idx_mapping, planned, self._planned_probs
            )
        if getattr(input_batch, "has_prefill", False):
            # Requests still mid-prefill after this step have not reached the
            # end of their prompt; keep their confidences at 1 until their
            # final prefill chunk emits a real draft proposal.
            incomplete_prefill_np = (
                input_batch.num_computed_prefill_tokens_np
                + input_batch.num_scheduled_tokens
                < input_batch.prefill_len_np
            )
            if incomplete_prefill_np.any():
                mid_prefill_slots = input_batch.idx_mapping_np[
                    incomplete_prefill_np
                ].tolist()
                self._stale_confidences[self._stale_idx].np[mid_prefill_slots] = 1.0
                self._pending_resets.extend(mid_prefill_slots)
                self._new_rows.extend(mid_prefill_slots)
                self.apply_staged_writes()
        # Replicated projections may differ through unordered floating-point
        # reductions unless both the TP group's reductions and the confidence
        # head are rank-invariant. Both CPU graph selection and GPU request
        # compaction must consume one TP-wide confidence snapshot.
        tp_group = get_tp_group()
        if tp_group.world_size > 1 and not (
            getattr(self, "tp_invariant_confidence", False)
            and getattr(
                getattr(tp_group, "device_communicator", None),
                "rank_order_reduce_scatter",
                False,
            )
        ):
            tp_group.broadcast(self._confidence_buffer, src=0)
        write_slot = self._stale_confidences[write_idx]
        write_slot.gpu.copy_(self._planned_probs)

        current_stream = torch.cuda.current_stream(self.req_states.device)
        self._copy_stream.wait_stream(current_stream)
        with stream(self._copy_stream, current_stream):
            write_slot.copy_to_cpu()
            self._copy_events[write_idx].record()

    def _context_row_cost_scale(
        self, seq_lens: np.ndarray | int | float
    ) -> np.ndarray:
        """Per-row verification cost multiplier as a function of active context length."""
        ctx_knee = getattr(self, "context_row_cost_knee", CONTEXT_ROW_COST_KNEE)
        seq_arr = np.asarray(seq_lens, dtype=np.float64)
        if ctx_knee <= 0:
            return np.ones_like(seq_arr, dtype=np.float64)
        excess = np.maximum(seq_arr - ctx_knee, 0.0)
        reindex_span = max(
            1.0,
            float(
                getattr(
                    self,
                    "context_reindex_saturation",
                    CONTEXT_REINDEX_SATURATION,
                )
                - ctx_knee
            ),
        )
        reindex_weight = float(
            getattr(self, "context_reindex_weight", CONTEXT_REINDEX_WEIGHT)
        )
        source_span = max(
            1.0, float(getattr(self, "context_source_span", CONTEXT_SOURCE_SPAN))
        )
        source_slope = float(
            getattr(self, "context_source_slope", CONTEXT_SOURCE_SLOPE)
        )
        max_scale = float(
            getattr(self, "context_max_row_cost_scale", CONTEXT_MAX_ROW_COST_SCALE)
        )
        scale = (
            1.0
            + reindex_weight * np.minimum(excess / reindex_span, 1.0)
            + source_slope * (excess / source_span)
        )
        return np.minimum(scale, max_scale)

    def _step_cost(
        self,
        rows: np.ndarray,
        *,
        context_excess_rows: np.ndarray | float = 0.0,
        num_reqs: int | None = None,
        num_sampling_requests: int = 1,
    ) -> np.ndarray:
        """Relative verification step cost with MMA-tile boundaries, 64-row wave
        saturation, and uncoalesced context-length indexer cost."""
        rows_f = np.asarray(rows, dtype=np.float64)
        knee = getattr(self, "high_row_cost_knee", HIGH_ROW_COST_KNEE)
        scale = getattr(self, "high_row_cost_scale", HIGH_ROW_COST_SCALE)
        coal_knee = getattr(
            self, "coalesced_row_cost_knee", COALESCED_ROW_COST_KNEE
        )
        coal_scale = getattr(
            self, "coalesced_row_cost_scale", COALESCED_ROW_COST_SCALE
        )
        coal_min_reqs = getattr(
            self, "coalesced_min_reqs", COALESCED_MIN_REQS
        )
        sat_knee = getattr(
            self, "saturated_row_cost_knee", SATURATED_ROW_COST_KNEE
        )
        sat_scale = getattr(
            self, "saturated_row_cost_scale", SATURATED_ROW_COST_SCALE
        )
        effective_reqs = num_reqs if num_reqs is not None else num_sampling_requests
        tile_step = (
            float(
                getattr(
                    self,
                    "single_req_tile_step_rows",
                    SINGLE_REQ_TILE_STEP_ROWS,
                )
            )
            if (effective_reqs <= 1 or num_sampling_requests <= 1) and num_sampling_requests <= 1
            else 0.0
        )
        single_req = tile_step > 0.0
        mid_scale = 1.0 if single_req else scale
        use_coal = (
            not single_req
            and coal_knee > knee
            and num_reqs is not None
            and num_reqs >= coal_min_reqs
        )
        band2b_scale = coal_scale if use_coal else mid_scale
        if knee > 0:
            band1 = np.minimum(rows_f, knee)
            if sat_knee > knee:
                eff_coal_knee = (
                    min(float(coal_knee), float(sat_knee))
                    if coal_knee > knee
                    else float(knee)
                )
                band2a = np.clip(rows_f - knee, 0.0, eff_coal_knee - knee)
                band2b = np.clip(
                    rows_f - eff_coal_knee, 0.0, sat_knee - eff_coal_knee
                )
                band3 = np.maximum(rows_f - sat_knee, 0.0)
                effective_rows = (
                    band1
                    + mid_scale * band2a
                    + band2b_scale * band2b
                    + sat_scale * band3
                )
            else:
                effective_rows = band1 + mid_scale * np.maximum(rows_f - knee, 0.0)
            if single_req:
                effective_rows = effective_rows + tile_step * (rows_f > knee)
            else:
                mma_step = getattr(
                    self, "mma_tile_step_rows", MMA_TILE_STEP_ROWS
                )
                if mma_step > 0.0:
                    effective_rows = effective_rows + mma_step * (
                        (rows_f > knee).astype(np.float64)
                        + 3.0 * (rows_f > 2 * knee).astype(np.float64)
                    )
            if sat_knee > 0:
                wave_step = getattr(
                    self, "second_wave_step_rows", SECOND_WAVE_STEP_ROWS
                )
                if wave_step > 0.0:
                    extra_waves = np.maximum(
                        np.ceil(np.maximum(rows_f, 1.0) / sat_knee) - 1.0, 0.0
                    )
                    effective_rows = effective_rows + wave_step * extra_waves
        else:
            effective_rows = rows_f
        effective_rows = effective_rows + context_excess_rows
        return 1.0 + self.row_cost * effective_rows

    def effective_dead_rows_tau(
        self, context_scale: float | None = None
    ) -> float:
        """Survival threshold for dead-row pruning on host and device.

        Raw ``confidence_head`` survival is inflated by ~3x at tail positions,
        which ``_DEAD_ROWS_TAU`` (0.20) was scaled to cut. Once online IRLS
        calibration fits the per-position intercepts, survival probabilities are
        already on the true acceptance scale, so the threshold matches the
        single-request marginal row-cost break-even ``2 * row_cost / (1 + 3 * row_cost)``
        (scaled by the active context-length row-cost multiplier) instead of
        double-counting the tail decay.
        """
        tau = float(getattr(self, "dead_rows_tau", 0.0))
        if tau <= 0.0:
            return 0.0
        calibrated = getattr(self, "calibrated_confidence", False)
        if callable(calibrated):
            calibrated = bool(calibrated())
        if calibrated and self.row_cost > 0.0:
            if context_scale is None:
                context_scale = float(getattr(self, "_batch_context_scale", 1.0))
            effective_row_cost = self.row_cost * max(1.0, context_scale)
            return min(
                tau,
                2.0 * effective_row_cost / (1.0 + 3.0 * effective_row_cost),
            )
        return tau

    def get_num_tokens(
        self,
        num_tokens_per_req: dict[str, int],
        draft_tokens: dict[str, list[int]],
    ) -> int:
        """Token count once the draft budget is trimmed to fit.

        The budget maximises expected tokens per unit of step cost: the step's
        sampling requests plus the survival probabilities of the admitted draft
        slots (best first), over ``1 + row_cost * rows``, where ``rows`` counts
        the step's non-draft rows and admitted draft rows; within the one-chunk
        logits limit. Stashes it in ``_batch_budget`` for the compaction and
        reallocation that follow in the same step.
        """
        req_ids = list(num_tokens_per_req)
        num_reqs = len(req_ids)
        scheduled_tokens = np.fromiter(
            num_tokens_per_req.values(), dtype=np.int32, count=num_reqs
        )
        scheduled_drafts = np.fromiter(
            (len(draft_tokens.get(req_id, ())) for req_id in req_ids),
            dtype=np.int32,
            count=num_reqs,
        )
        num_non_draft_tokens = scheduled_tokens - scheduled_drafts
        slots = np.fromiter(
            (self.req_states.req_id_to_index[req_id] for req_id in req_ids),
            dtype=np.int32,
            count=len(req_ids),
        )
        seq_lens = (
            self.req_states.num_computed_tokens_np[slots] + num_non_draft_tokens
        )
        sampling_mask = seq_lens >= self.req_states.prefill_len.np[slots]
        num_sampling_requests = int(np.count_nonzero(sampling_mask))
        req_context_scales = self._context_row_cost_scale(
            self.req_states.num_computed_tokens_np[slots]
        )
        self._batch_context_scale = (
            float(np.min(req_context_scales[sampling_mask]))
            if num_sampling_requests > 0
            else 1.0
        )
        # The scheduler's drafts, not a rank's own view of the latest proposal:
        # the budget reads only what every rank holds alike (these and the
        # confidence snapshot), so every rank lays out the same step (#852).
        eligible_drafts = scheduled_drafts
        num_draft_steps = getattr(self, "num_draft_steps", self.num_speculative_steps)
        step_steps = (
            num_draft_steps
            if int(eligible_drafts.max(initial=0)) <= num_draft_steps
            else self.num_speculative_steps
        )
        stale_confidences = self._stale_confidences[self._stale_idx].np[
            slots, :step_steps
        ].astype(np.float64, copy=True)
        computed_np = self.req_states.num_computed_tokens_np[slots]
        prefill_np = self.req_states.prefill_len.np[slots]
        tol = float(getattr(self, "cohort_confidence_tol", COHORT_CONFIDENCE_TOL))
        cohorts: dict[tuple[int, int, int], list[int]] = {}
        if num_reqs >= 2 and step_steps > 0 and tol > 0:
            groups: dict[tuple[int, int, int], list[int]] = {}
            for i in range(num_reqs):
                if sampling_mask[i] and eligible_drafts[i] > 0:
                    key = (
                        int(computed_np[i]),
                        int(prefill_np[i]),
                        int(eligible_drafts[i]),
                    )
                    groups.setdefault(key, []).append(i)
            for key, idxs in groups.items():
                if len(idxs) >= 2:
                    group_conf = stale_confidences[idxs]
                    if float(np.ptp(group_conf, axis=0).max(initial=0.0)) <= tol:
                        cohorts[key] = idxs
                        stale_confidences[idxs] = group_conf.mean(axis=0, keepdims=True)

        survival_probability = np.cumprod(stale_confidences, axis=1)
        steps = np.arange(step_steps)
        valid = steps[None, :] < eligible_drafts[:, None]
        if float(getattr(self, "dead_rows_tau", 0.0)) > 0.0:
            # Never admit draft slots on the host that the device dead-row
            # threshold will immediately mask as padding.
            req_taus = np.fromiter(
                (
                    self.effective_dead_rows_tau(float(scale))
                    for scale in req_context_scales
                ),
                dtype=np.float64,
                count=num_reqs,
            )
            valid &= survival_probability >= req_taus[:, None]
        max_draft_budget = min(
            int(eligible_drafts.sum()),
            max(0, self._max_total_logits - num_reqs * self.num_bonus_tokens),
        )
        token_spread = float(np.ptp(computed_np)) if num_reqs > 1 else 0.0
        if (
            token_spread > 0.0
            and getattr(self, "context_row_cost_knee", CONTEXT_ROW_COST_KNEE) > 0
        ):
            het_slope = float(
                getattr(self, "context_source_slope", CONTEXT_SOURCE_SLOPE)
            )
            het_scales = 1.0 + het_slope * (
                np.maximum(computed_np - float(np.min(computed_np)), 0.0)
                / 65536.0
            )
            req_context_scales = np.maximum(req_context_scales, het_scales)
        valid_req_idx, _ = np.nonzero(valid)
        valid_scores = survival_probability[valid]
        valid_scales = req_context_scales[valid_req_idx]
        heterogeneous_context = (
            num_reqs > 1
            and valid_scores.size > 0
            and float(np.ptp(req_context_scales)) > 1e-6
        )
        num_non_draft_tokens_per_req = {
            req_id: int(num_tokens)
            for req_id, num_tokens in zip(req_ids, num_non_draft_tokens, strict=True)
        }
        base_context_excess = float(
            np.dot(num_non_draft_tokens, np.maximum(req_context_scales - 1.0, 0.0))
        )
        capacities_cpu: np.ndarray | None = None
        if cohorts and getattr(self, "_confidence_buffer", torch.empty(1)).shape[0] <= 1:
            in_cohort = np.zeros(num_reqs, dtype=bool)
            units: list[tuple[float, list[int]]] = []
            for idxs in cohorts.values():
                in_cohort[idxs] = True
                rep = idxs[0]
                for d in range(int(eligible_drafts[rep])):
                    if valid[rep, d]:
                        units.append((float(survival_probability[rep, d]), idxs))
                    else:
                        break
            for i in range(num_reqs):
                if not in_cohort[i]:
                    for d in range(int(eligible_drafts[i])):
                        if valid[i, d]:
                            units.append((float(survival_probability[i, d]), [i]))
                        else:
                            break
            units.sort(key=lambda u: u[0], reverse=True)
            capacities_cpu = np.zeros(num_reqs, dtype=np.int32)
            base_rows = int(num_non_draft_tokens.sum())
            best_util = float(num_sampling_requests) / float(
                self._step_cost(
                    np.array([base_rows], dtype=np.float64),
                    context_excess_rows=base_context_excess,
                    num_reqs=num_reqs,
                    num_sampling_requests=int(num_sampling_requests),
                )[0]
            )
            running_rows = base_rows
            running_excess = base_context_excess
            running_tok = float(num_sampling_requests)
            admitted_rows = 0
            best_units = 0
            for idx_u, (surv, idxs) in enumerate(units, start=1):
                if admitted_rows + len(idxs) > max_draft_budget:
                    continue
                admitted_rows += len(idxs)
                running_rows += len(idxs)
                running_excess += float(
                    np.sum(np.maximum(req_context_scales[idxs] - 1.0, 0.0))
                )
                running_tok += surv * len(idxs)
                cand_util = running_tok / float(
                    self._step_cost(
                        np.array([running_rows], dtype=np.float64),
                        context_excess_rows=running_excess,
                        num_reqs=num_reqs,
                        num_sampling_requests=int(num_sampling_requests),
                    )[0]
                )
                if cand_util > best_util:
                    best_util = cand_util
                    best_units = idx_u
            used_rows = 0
            for surv, idxs in units[:best_units]:
                if used_rows + len(idxs) <= max_draft_budget:
                    used_rows += len(idxs)
                    capacities_cpu[idxs] += 1

            # Phase-locking catch-up: if a minority lead subset is ahead by
            # 1..step_steps+1 tokens from the main cohort, hold it to 1 token
            # and cap the main cohort so both merge onto the same position.
            if cohorts:
                largest_key = max(cohorts, key=lambda k: len(cohorts[k]))
                largest_idxs = cohorts[largest_key]
                pos_b = largest_key[0]
                prefill_b = largest_key[1]
                k_b = int(capacities_cpu[largest_idxs[0]])
                if k_b > 0:
                    lead_idxs: list[int] = []
                    min_gap: int | None = None
                    for i in range(num_reqs):
                        if (
                            sampling_mask[i]
                            and int(prefill_np[i]) == prefill_b
                            and int(computed_np[i]) > pos_b
                        ):
                            gap = int(computed_np[i]) - pos_b
                            if 1 <= gap <= step_steps + 1:
                                lead_idxs.append(i)
                                if min_gap is None or gap < min_gap:
                                    min_gap = gap
                    if (
                        lead_idxs
                        and len(lead_idxs) < len(largest_idxs)
                        and min_gap is not None
                    ):
                        capacities_cpu[lead_idxs] = 0
                        capacities_cpu[largest_idxs] = min(k_b, min_gap)
            draft_budget = int(capacities_cpu.sum())
            ordered_req_idx = np.empty(0, dtype=np.intp)
        else:
            if heterogeneous_context:
                priorities = valid_scores - self.row_cost * np.maximum(
                    valid_scales - 1.0, 0.0
                )
                order = np.lexsort((-valid_scores, -priorities))[:max_draft_budget]
            else:
                order = np.argsort(valid_scores, kind="stable")[::-1][
                    :max_draft_budget
                ]
            scores = valid_scores[order]
            ordered_scales = valid_scales[order]
            ordered_req_idx = valid_req_idx[order]
            context_excess_rows = base_context_excess + np.concatenate(
                ([0.0], np.cumsum(np.maximum(ordered_scales - 1.0, 0.0)))
            )
            expected_tokens = np.concatenate(
                ([num_sampling_requests], num_sampling_requests + np.cumsum(scores))
            )
            rows = int(num_non_draft_tokens.sum()) + np.arange(len(scores) + 1)
            utilities = expected_tokens / self._step_cost(
                rows,
                context_excess_rows=context_excess_rows,
                num_reqs=num_reqs,
                num_sampling_requests=int(num_sampling_requests),
            )
            draft_budget = int(np.argmax(utilities))
        # The confidences come from an estimator that learns only from admitted
        # drafts, so a long run of empty budgets would leave it without the
        # evidence to ever admit one again. Periodically verify one draft per
        # request to keep it grounded.
        idle = getattr(self, "_steps_without_drafts", 0)
        if draft_budget == 0 and max_draft_budget > 0:
            idle += 1
            if idle >= EXPLORE_AFTER_IDLE_STEPS:
                draft_budget = min(num_reqs, max_draft_budget)
                if capacities_cpu is not None:
                    capacities_cpu[:] = 0
                    capacities_cpu[:draft_budget] = 1
                idle = 0
        else:
            idle = 0
        self._steps_without_drafts = idle
        if (
            heterogeneous_context
            and 0 < draft_budget < int(eligible_drafts.sum())
            and getattr(self, "_confidence_buffer", torch.empty(1)).shape[0] <= 1
        ):
            admitted_per_req = np.bincount(
                ordered_req_idx[:draft_budget], minlength=num_reqs
            )
            num_drafts_per_req = {
                req_id: int(admitted)
                for req_id, admitted in zip(req_ids, admitted_per_req, strict=True)
            }
        else:
            num_drafts_per_req = {
                req_id: int(num_drafts)
                for req_id, num_drafts in zip(req_ids, eligible_drafts, strict=True)
            }
        logger.debug(
            "DSpark adaptive verification selected %d/%d draft rows for %d "
            "requests at row cost %.3g",
            draft_budget,
            int(scheduled_drafts.sum()),
            num_reqs,
            self.row_cost,
        )
        self._capacities_cpu_override = (
            {req_id: int(c) for req_id, c in zip(req_ids, capacities_cpu, strict=True)}
            if capacities_cpu is not None
            else None
        )
        self._cohort_locked = capacities_cpu is not None
        self._batch_budget = (
            num_drafts_per_req,
            num_non_draft_tokens_per_req,
            draft_budget,
        )
        return sum(num_non_draft_tokens_per_req.values()) + draft_budget

    def compact_batch(
        self,
        num_draft_tokens_per_req: np.ndarray,
        num_scheduled_tokens: np.ndarray,
        cu_num_logits_np: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compact the CPU batch to the chosen draft budget.

        Returns the compacted per-request token counts and the CPU cu_num_logits_np.
        If the draft budget is 0, we can know cu_num_logits_np exactly, otherwise
        its unchanged/an-upper-bound.
        """
        batch_budget = self._batch_budget
        assert batch_budget is not None
        _, _, draft_budget = batch_budget
        num_drafts = int(num_draft_tokens_per_req.sum())
        if draft_budget == num_drafts:
            return num_scheduled_tokens, cu_num_logits_np

        num_non_draft_tokens = num_scheduled_tokens - num_draft_tokens_per_req
        if draft_budget == 0:
            # The draft budget is 0, so we can know cu_num_logits_np exactly. This helps
            # when we would exceed the sampler logit chunk size.
            num_reqs = num_scheduled_tokens.shape[0]
            cu_num_logits_np = (
                np.arange(num_reqs + 1, dtype=cu_num_logits_np.dtype)
                * self.num_bonus_tokens
            )
            return num_non_draft_tokens, cu_num_logits_np

        is_verification_request = num_draft_tokens_per_req > 0
        num_verification_reqs = int(is_verification_request.sum())
        # sort_batch_req_ids keeps verification requests at the front.
        assert np.all(is_verification_request[:num_verification_reqs])
        # for the CPU side buffer we distribute draft tokens evenly
        draft_lens_cpu = np.zeros_like(num_non_draft_tokens)
        draft_lens_cpu[:num_verification_reqs] = draft_budget // num_verification_reqs
        draft_lens_cpu[: draft_budget % num_verification_reqs] += 1
        return num_non_draft_tokens + draft_lens_cpu, cu_num_logits_np

    def dead_row_args(self, is_padding: torch.Tensor, num_reqs: int) -> DeadRows | None:
        """Arguments for the fused dead-row marking, or None when disabled.

        The confidences are the current drafts', broadcast from rank 0, so
        every rank marks the same rows.
        """
        if getattr(self, "_cohort_locked", False):
            return None
        tau = self.effective_dead_rows_tau()
        if tau <= 0 or not getattr(self, "_has_admitted_drafts", True):
            return None
        return DeadRows(
            self._confidence_probs,
            is_padding,
            self._dead_row_cut[:num_reqs],
            float(np.log(tau)),
        )

    def get_verified_draft_counts(
        self, num_reqs: int, dead_row_cut: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return the device-selected live draft count for each request."""
        capacities = self._batch_draft_capacity[:num_reqs]
        if dead_row_cut is not None:
            torch.minimum(capacities, dead_row_cut[:num_reqs], out=capacities)
        return capacities

    def reallocate_drafts(
        self, req_ids: list[str], idx_mapping: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, torch.Tensor]:
        """The step's logits offsets, query starts and draft budget, and its
        expanded state indices and positions (``expand_idx_mapping``'s), all
        from the admitted drafts."""
        batch_budget, self._batch_budget = self._batch_budget, None
        assert batch_budget is not None
        num_drafts_per_req, num_non_draft_tokens_per_req, draft_budget = batch_budget
        override = getattr(self, "_capacities_cpu_override", None)
        self._capacities_cpu_override = None
        num_reqs = idx_mapping.shape[0]
        scheduled_drafts = np.fromiter(
            (num_drafts_per_req[req_id] for req_id in req_ids),
            dtype=np.int32,
            count=num_reqs,
        )
        num_non_draft_tokens = np.fromiter(
            (num_non_draft_tokens_per_req[req_id] for req_id in req_ids),
            dtype=np.int32,
            count=num_reqs,
        )
        num_tokens = int(num_non_draft_tokens.sum()) + draft_budget

        # Rank draft slots by survival probability and admit the best prefix.
        # capacities enters holding each request's valid draft count (the kernel
        # uses it to bound eligible slots) and leaves holding the admitted count.
        num_draft_steps = getattr(self, "num_draft_steps", self.num_speculative_steps)
        step_steps = (
            num_draft_steps
            if int(scheduled_drafts.max(initial=0)) <= num_draft_steps
            else self.num_speculative_steps
        )
        self.step_steps = step_steps
        self._has_admitted_drafts = draft_budget > 0
        capacities = self._batch_draft_capacity[:num_reqs]
        if draft_budget:
            if override is not None:
                cohort_caps = np.fromiter(
                    (override[req_id] for req_id in req_ids),
                    dtype=np.int32,
                    count=num_reqs,
                )
                async_tensor_h2d(cohort_caps, out=capacities)
            else:
                if num_reqs == 1:
                    scheduled_drafts[0] = draft_budget
                async_tensor_h2d(scheduled_drafts, out=capacities)
                if draft_budget < int(scheduled_drafts.sum()):
                    tilelang_inputs.assign_draft_budget(
                        self._confidence_probs[:, :step_steps],
                        idx_mapping,
                        capacities,
                        draft_budget,
                        step_steps,
                    )

        num_non_draft_tokens_gpu = self._num_non_draft_tokens[:num_reqs]
        async_tensor_h2d(num_non_draft_tokens, out=num_non_draft_tokens_gpu)
        # Both prefix sums, query_start_loc's tail and the expanded logits rows in
        # one kernel, which zeroes capacities on a step without drafts.
        expanded_idx_mapping, expanded_local_pos = tilelang_inputs.adaptive_offsets(
            capacities,
            num_non_draft_tokens_gpu,
            self._cu_num_logits,
            self.query_start_loc,
            idx_mapping,
            num_reqs * self.num_bonus_tokens + draft_budget,
            step_steps + self.num_bonus_tokens,
            self.num_bonus_tokens,
            num_tokens,
            drafted=draft_budget > 0,
        )
        return (
            self._cu_num_logits[: num_reqs + 1],
            self.query_start_loc,
            draft_budget,
            expanded_idx_mapping,
            expanded_local_pos,
        )


def maybe_create_adaptive_verification_manager(
    *,
    enable_adaptive_verification: bool,
    attn_groups: list[list["AttentionGroup"]],
    attn_cg_support: "AttentionCGSupportInfo",
    req_states: "RequestState",
    query_start_loc: torch.Tensor,
    num_bonus_tokens: int,
    max_total_logits: int,
    vllm_config: "VllmConfig",
    adaptive_verification_row_cost: float = 0.02,
    target_layer_names: set[str] | None = None,
    additional_attn_cg_support: tuple[AttentionCGSupport, str | None] | None = None,
    planned_confidences: bool = False,
    tp_invariant_confidence: bool = False,
    calibrated_confidence: bool | Callable[[], bool] = False,
) -> AdaptiveVerificationManager | None:
    if not enable_adaptive_verification:
        return None

    # The selector rejects unsupported backends, but models that
    # hard-wire theirs (e.g. DeepSeek-V4) never go through it.
    backend = get_query_lens_mismatch_unsupported_backend(
        attn_groups,
        checked_layer_names=target_layer_names,
    )
    if backend is not None:
        raise ValueError(
            "Adaptive verification trims verification requests on device, which"
            f" the {backend} attention backend does not support. Pass "
            "enable_adaptive_verification=false in the speculative config, or "
            "use a backend that does."
        )

    target_attn_cg_support = attn_cg_support
    if target_layer_names is not None:
        target_attn_cg_support = get_attn_cg_support(
            attn_groups,
            vllm_config,
            checked_layer_names=target_layer_names,
        )
        if additional_attn_cg_support is not None:
            target_attn_cg_support = target_attn_cg_support.narrow(
                *additional_attn_cg_support
            )
    varlen_unsupported = []
    groups_to_check = (
        attn_groups
        if target_attn_cg_support.min_cg_support != AttentionCGSupport.ALWAYS
        else []
    )
    for groups in groups_to_check:
        for group in groups:
            if target_layer_names is not None and target_layer_names.isdisjoint(
                group.layer_names
            ):
                continue
            builder = group.get_metadata_builder(0)
            support = builder.get_cudagraph_support(vllm_config, group.kv_cache_spec)
            if support != AttentionCGSupport.ALWAYS and not (
                support == AttentionCGSupport.UNIFORM_BATCH
                and getattr(builder, "supports_varlen_decode_cudagraph", False)
            ):
                varlen_unsupported.append(group.backend.__name__)
    if (
        additional_attn_cg_support is not None
        and additional_attn_cg_support[0] != AttentionCGSupport.ALWAYS
    ):
        varlen_unsupported.append(str(additional_attn_cg_support[1]))
    if varlen_unsupported:
        raise ValueError(
            "Adaptive verification requires variable-length decode CUDA graphs; "
            f"unsupported target backends: {', '.join(varlen_unsupported)}. Pass "
            "enable_adaptive_verification=false in the speculative config, or "
            "use a backend that does."
        )

    return AdaptiveVerificationManager(
        req_states,
        query_start_loc,
        num_bonus_tokens,
        max_total_logits=max_total_logits,
        row_cost=adaptive_verification_row_cost,
        planned_confidences=planned_confidences,
        num_draft_steps=getattr(
            getattr(vllm_config, "speculative_config", None),
            "num_speculative_tokens",
            None,
        )
        or None,
        tp_invariant_confidence=tp_invariant_confidence,
        calibrated_confidence=calibrated_confidence,
    )


_TRITON_REFS = frozenset(("_assign_draft_token_budget",))


def __getattr__(name: str):
    if name in _TRITON_REFS:
        from vllm.v1.worker.gpu import _triton_refs

        return getattr(_triton_refs, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
