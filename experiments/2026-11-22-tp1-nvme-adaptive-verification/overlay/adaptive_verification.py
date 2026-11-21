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

# Two-slope verification row-cost knee: once a step crosses one 16-row Tensor
# Core MMA tile, additional verification rows share the already-launched expert
# and dense weight tiles at a reduced marginal cost.
HIGH_ROW_COST_KNEE = 16
HIGH_ROW_COST_SCALE = 0.55
SINGLE_REQ_TILE_SPILL_SCALE = 2.0
# For single-request (c1) DSpark verification, cold-L2 routed MoE touches ~5.5
# distinct experts per live draft row (~+2.5-4.1 ms/row) without cross-request
# coalescing; scale the calibrated c1 row cost accordingly while leaving
# multi-request concurrency (c2..c16) and long copy-draft proposals untouched.
SINGLE_REQ_CALIBRATED_ROW_COST_SCALE = 2.5
WIDE_ROW_COST_KNEE = 32
CALIBRATED_HIGH_ROW_COST_SCALE = 0.65
CALIBRATED_WIDE_ROW_COST_SCALE = 0.50
SMALL_BATCH_L2_ROW_COST_BONUS = 0.76
SMALL_BATCH_COALESCE_REQS = 8
CALIBRATED_TILE_COST_SHARE = 0.30
CALIBRATED_STEP_TAU_CAP = 0.25
CALIBRATED_STALE_FIRST_DRAFT_SHRINK = 0.25
CALIBRATED_STALE_FIRST_DRAFT_PRIOR = 0.60
_CALIBRATED_INIT_PROBS: tuple[float, ...] = (0.68, 0.55, 0.50, 0.48, 0.46)


def _wave_padded_rows(rows: np.ndarray, num_reqs: int) -> np.ndarray:
    """Round admitted draft counts up to full active-request waves."""
    r = np.asarray(rows, dtype=np.int64)
    if r.size == 0 or num_reqs <= 1:
        return r.astype(np.float64)
    base = int(r.reshape(-1)[0])
    drafts = np.maximum(r - base, 0)
    padded = base + ((drafts + num_reqs - 1) // num_reqs) * num_reqs
    return padded.astype(np.float64)


def resolve_adaptive_cudagraph_mode(
    mode: CUDAGraphMode, *, piecewise_capture_available: bool
) -> CUDAGraphMode:
    """Select a separate decode route for variable-length verification."""
    if not mode.has_full_cudagraphs():
        return mode
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
        single_req_calibrated_row_cost_scale: float = SINGLE_REQ_CALIBRATED_ROW_COST_SCALE,
        calibrated_high_row_cost_scale: float = CALIBRATED_HIGH_ROW_COST_SCALE,
        wide_row_cost_knee: int = WIDE_ROW_COST_KNEE,
        calibrated_wide_row_cost_scale: float = CALIBRATED_WIDE_ROW_COST_SCALE,
        max_cudagraph_tokens: int = 0,
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
        self.single_req_tile_spill_scale = SINGLE_REQ_TILE_SPILL_SCALE
        self.tp_invariant_confidence = tp_invariant_confidence
        self.calibrated_confidence = calibrated_confidence
        self.planned_confidences = bool(planned_confidences)
        self.single_req_calibrated_row_cost_scale = float(
            single_req_calibrated_row_cost_scale
        )
        self.calibrated_high_row_cost_scale = float(calibrated_high_row_cost_scale)
        self.wide_row_cost_knee = int(wide_row_cost_knee)
        self.calibrated_wide_row_cost_scale = float(calibrated_wide_row_cost_scale)
        self.max_cudagraph_tokens = int(max_cudagraph_tokens)
        self.dead_rows_tau = _DEAD_ROWS_TAU
        self._has_admitted_drafts = False
        self._step_dead_rows_tau: float | None = None
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
        init_stale = self._initial_stale_values()
        for slot in self._stale_confidences:
            slot.np[:] = init_stale

    def _initial_stale_values(self) -> np.ndarray:
        if self._is_calibrated():
            priors = _CALIBRATED_INIT_PROBS
            return np.array(
                [
                    priors[min(i, len(priors) - 1)]
                    for i in range(self.num_speculative_steps)
                ],
                dtype=np.float32,
            )
        return np.ones(self.num_speculative_steps, dtype=np.float32)

    def add_request(self, req_idx: int) -> None:
        self._stale_confidences[self._stale_idx].np[req_idx] = (
            self._initial_stale_values()
        )
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
            self._stale_confidences[ready_idx].np[self._pending_resets] = (
                self._initial_stale_values()
            )
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
                self._stale_confidences[self._stale_idx].np[mid_prefill_slots] = (
                    self._initial_stale_values()
                )
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

    def _is_calibrated(self) -> bool:
        calibrated = getattr(self, "calibrated_confidence", False)
        if callable(calibrated):
            return bool(calibrated())
        return bool(calibrated)

    def _calibrated_row_cost(self, *, num_reqs: int = 1) -> float:
        if not (self.row_cost > 0.0 and self._is_calibrated()):
            return self.row_cost
        if num_reqs <= 1:
            c1_scale = float(
                getattr(
                    self,
                    "single_req_calibrated_row_cost_scale",
                    SINGLE_REQ_CALIBRATED_ROW_COST_SCALE,
                )
            )
            return self.row_cost * c1_scale
        l2_bonus = float(
            getattr(
                self,
                "small_batch_l2_row_cost_bonus",
                SMALL_BATCH_L2_ROW_COST_BONUS,
            )
        )
        req_scale = (
            1.0 + l2_bonus / float(num_reqs)
            if num_reqs < SMALL_BATCH_COALESCE_REQS
            else 1.0
        )
        return self.row_cost * req_scale

    def _c1_calibrated_row_cost(self, *, num_reqs: int = 1, max_rows: int = 0) -> float:
        del max_rows
        return self._calibrated_row_cost(num_reqs=num_reqs)

    def _step_cost(
        self, rows: np.ndarray, *, num_sampling_requests: int = 1
    ) -> np.ndarray:
        """Relative verification step cost with Tensor Core MMA and MoE knees.

        With calibrated confidences, prices each base-block / post-knee region
        continuously without discontinuities across batch sizes or TP ranks.
        """
        knee = getattr(self, "high_row_cost_knee", HIGH_ROW_COST_KNEE)
        if self._is_calibrated() and self.row_cost > 0.0:
            if num_sampling_requests <= 1:
                base_block = int(
                    getattr(self, "num_bonus_tokens", 1)
                    + getattr(
                        self,
                        "num_draft_steps",
                        getattr(self, "num_speculative_steps", 5),
                    )
                )
                c1_rc = self._calibrated_row_cost(num_reqs=1)
                spill = float(
                    getattr(
                        self,
                        "single_req_tile_spill_scale",
                        SINGLE_REQ_TILE_SPILL_SCALE,
                    )
                )
                r1 = np.minimum(rows, base_block)
                if knee > base_block:
                    r2 = np.maximum(np.minimum(rows, knee) - base_block, 0)
                    r3 = np.maximum(rows - knee, 0)
                    return 1.0 + c1_rc * r1 + self.row_cost * r2 + (c1_rc * spill) * r3
                return (
                    1.0
                    + c1_rc * r1
                    + (c1_rc * spill) * np.maximum(rows - base_block, 0)
                )
            eff_rc = self._calibrated_row_cost(num_reqs=num_sampling_requests)
            custom_hi = getattr(self, "high_row_cost_scale", HIGH_ROW_COST_SCALE)
            if custom_hi != HIGH_ROW_COST_SCALE:
                hi_scale = float(custom_hi)
            else:
                hi_scale = float(
                    getattr(
                        self,
                        "calibrated_high_row_cost_scale",
                        CALIBRATED_HIGH_ROW_COST_SCALE,
                    )
                )
            wide_knee = int(getattr(self, "wide_row_cost_knee", WIDE_ROW_COST_KNEE))
            wide_scale = float(
                getattr(
                    self,
                    "calibrated_wide_row_cost_scale",
                    CALIBRATED_WIDE_ROW_COST_SCALE,
                )
            )
            tile_share = float(
                getattr(self, "calibrated_tile_cost_share", CALIBRATED_TILE_COST_SHARE)
            )
            if knee > 0:

                def _piecewise(x: np.ndarray) -> np.ndarray:
                    r1 = np.minimum(x, knee)
                    if wide_knee > knee:
                        r2 = np.maximum(np.minimum(x, wide_knee) - knee, 0)
                        r3 = np.maximum(x - wide_knee, 0)
                        return r1 + hi_scale * r2 + wide_scale * r3
                    return r1 + hi_scale * np.maximum(x - knee, 0)

                rows_f64 = np.asarray(rows, dtype=np.float64)
                effective_rows = (1.0 - tile_share) * _piecewise(rows_f64)
                if tile_share > 0.0:
                    effective_rows = effective_rows + tile_share * _piecewise(
                        _wave_padded_rows(rows, int(num_sampling_requests))
                    )
            else:
                effective_rows = rows
            return 1.0 + eff_rc * effective_rows

        if num_sampling_requests <= 1:
            scale = getattr(
                self,
                "single_req_tile_spill_scale",
                SINGLE_REQ_TILE_SPILL_SCALE,
            )
        else:
            scale = getattr(self, "high_row_cost_scale", HIGH_ROW_COST_SCALE)
        if knee > 0 and scale != 1.0:
            effective_rows = np.minimum(rows, knee) + scale * np.maximum(
                rows - knee, 0
            )
        else:
            effective_rows = rows
        return 1.0 + self.row_cost * effective_rows

    def effective_dead_rows_tau(self, num_reqs: int = 1) -> float:
        """Survival threshold for dead-row pruning on host and device."""
        tau = float(getattr(self, "dead_rows_tau", 0.0))
        if tau <= 0.0:
            return 0.0
        if self._is_calibrated() and self.row_cost > 0.0:
            eff_row_cost = self._calibrated_row_cost(num_reqs=num_reqs)
            return min(tau, 2.0 * eff_row_cost / (1.0 + 3.0 * eff_row_cost))
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
        max_draft_budget = min(
            int(eligible_drafts.sum()),
            max(0, self._max_total_logits - num_reqs * self.num_bonus_tokens),
        )
        max_cg = int(getattr(self, "max_cudagraph_tokens", 0))
        non_draft_total = int(num_non_draft_tokens.sum())
        max_non_draft = int(num_non_draft_tokens.max(initial=0))
        if (
            max_cg > 0
            and num_reqs > 0
            and max_non_draft <= self.num_bonus_tokens
            and non_draft_total <= max_cg
        ):
            # Keep pure-decode verification (e.g. concurrent CopyDrafter bursts)
            # within max_cudagraph_capture_size instead of spilling to eager.
            max_draft_budget = min(max_draft_budget, max_cg - non_draft_total)
        num_drafts_per_req = {
            req_id: int(num_drafts)
            for req_id, num_drafts in zip(req_ids, eligible_drafts, strict=True)
        }
        num_non_draft_tokens_per_req = {
            req_id: int(num_tokens)
            for req_id, num_tokens in zip(req_ids, num_non_draft_tokens, strict=True)
        }
        num_sampling_requests = np.count_nonzero(
            self.req_states.num_computed_tokens_np[slots] + num_non_draft_tokens
            >= self.req_states.prefill_len.np[slots]
        )
        verifying_mask = eligible_drafts > 0
        num_verification_reqs = int(np.count_nonzero(verifying_mask))
        eff_reqs = num_verification_reqs if num_verification_reqs > 0 else num_reqs
        tau = self.effective_dead_rows_tau(num_reqs=eff_reqs)
        if (
            num_reqs == 1
            and self._is_calibrated()
            and float(getattr(self, "dead_rows_tau", 0.0)) > 0.0
            and int(eligible_drafts.max(initial=0)) <= num_draft_steps
            and int(num_sampling_requests) > 0
        ):
            # All <=5 DSpark slots fit in the single 16-row Tensor Core tile;
            # let device DeadRows prune on current-step confidences rather than
            # intersecting against 1-step-stale host confidences (p_k^2).
            draft_budget = max_draft_budget
            self._step_dead_rows_tau = tau
        else:
            stale_confidences = self._stale_confidences[self._stale_idx].np[
                slots, :step_steps
            ]
            stale_f64 = stale_confidences.astype(np.float64, copy=True)
            if self._is_calibrated() and stale_f64.shape[1] > 0:
                dspark_mask = eligible_drafts <= num_draft_steps
                shrink0 = float(
                    getattr(
                        self,
                        "calibrated_stale_first_draft_shrink",
                        CALIBRATED_STALE_FIRST_DRAFT_SHRINK,
                    )
                )
                prior0 = float(
                    getattr(
                        self,
                        "calibrated_stale_first_draft_prior",
                        CALIBRATED_STALE_FIRST_DRAFT_PRIOR,
                    )
                )
                if shrink0 > 0.0 and np.any(dspark_mask):
                    stale_f64[dspark_mask, 0] = (
                        1.0 - shrink0
                    ) * stale_f64[dspark_mask, 0] + shrink0 * prior0
            survival_probability = np.cumprod(stale_f64, axis=1)
            steps = np.arange(step_steps)
            valid = steps[None, :] < eligible_drafts[:, None]
            if tau > 0:
                # Never admit draft slots on the host that the device dead-row
                # threshold will immediately mask as padding.
                valid &= survival_probability >= tau
            scores = np.sort(survival_probability[valid])[::-1]
            scores = scores[:max_draft_budget]
            if 0 < num_verification_reqs < num_reqs:
                # Isolate decode-draft utility from sunk concurrent prefill rows.
                base_tokens = num_verification_reqs * self.num_bonus_tokens
                expected_tokens = np.concatenate(
                    ([base_tokens], base_tokens + np.cumsum(scores))
                )
                rows = int(num_non_draft_tokens[verifying_mask].sum()) + np.arange(
                    len(scores) + 1
                )
                costs = self._step_cost(
                    rows, num_sampling_requests=num_verification_reqs
                )
            else:
                expected_tokens = np.concatenate(
                    ([num_sampling_requests], num_sampling_requests + np.cumsum(scores))
                )
                rows = int(num_non_draft_tokens.sum()) + np.arange(len(scores) + 1)
                costs = self._step_cost(
                    rows, num_sampling_requests=int(num_sampling_requests)
                )
            utilities = expected_tokens / costs
            draft_budget = int(np.argmax(utilities))
            if (
                0 < num_verification_reqs < num_reqs
                and self._is_calibrated()
                and tau > 0.0
                and draft_budget > 0
                and len(scores) > 0
            ):
                k_ref = min(draft_budget, len(scores) - 1)
                marginal_s = float(
                    expected_tokens[k_ref]
                    * (costs[k_ref + 1] - costs[k_ref])
                    / costs[k_ref]
                )
                step_cap = float(
                    getattr(self, "calibrated_step_tau_cap", CALIBRATED_STEP_TAU_CAP)
                )
                self._step_dead_rows_tau = max(tau, min(step_cap, marginal_s))
            else:
                self._step_dead_rows_tau = tau
        # The confidences come from an estimator that learns only from admitted
        # drafts, so a long run of empty budgets would leave it without the
        # evidence to ever admit one again. Periodically verify one draft per
        # request to keep it grounded.
        idle = getattr(self, "_steps_without_drafts", 0)
        if draft_budget == 0 and max_draft_budget > 0:
            idle += 1
            if idle >= EXPLORE_AFTER_IDLE_STEPS:
                draft_budget = min(num_reqs, max_draft_budget)
                idle = 0
        else:
            idle = 0
        self._steps_without_drafts = idle
        logger.debug(
            "DSpark adaptive verification selected %d/%d draft rows for %d "
            "requests at row cost %.3g",
            draft_budget,
            int(scheduled_drafts.sum()),
            num_reqs,
            self.row_cost,
        )
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
        step_tau = getattr(self, "_step_dead_rows_tau", None)
        self._step_dead_rows_tau = None
        tau = (
            self.effective_dead_rows_tau(num_reqs=num_reqs)
            if step_tau is None
            else float(step_tau)
        )
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
    adaptive_verification_single_req_row_cost_scale: float = SINGLE_REQ_CALIBRATED_ROW_COST_SCALE,
    adaptive_verification_high_row_cost_scale: float = CALIBRATED_HIGH_ROW_COST_SCALE,
    adaptive_verification_wide_row_cost_knee: int = WIDE_ROW_COST_KNEE,
    adaptive_verification_wide_row_cost_scale: float = CALIBRATED_WIDE_ROW_COST_SCALE,
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
        single_req_calibrated_row_cost_scale=(
            adaptive_verification_single_req_row_cost_scale
        ),
        calibrated_high_row_cost_scale=(
            adaptive_verification_high_row_cost_scale
        ),
        wide_row_cost_knee=adaptive_verification_wide_row_cost_knee,
        calibrated_wide_row_cost_scale=(
            adaptive_verification_wide_row_cost_scale
        ),
        max_cudagraph_tokens=int(
            getattr(
                getattr(vllm_config, "compilation_config", None),
                "max_cudagraph_capture_size",
                None,
            )
            or 0
        ),
    )


_TRITON_REFS = frozenset(("_assign_draft_token_budget",))


def __getattr__(name: str):
    if name in _TRITON_REFS:
        from vllm.v1.worker.gpu import _triton_refs

        return getattr(_triton_refs, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
