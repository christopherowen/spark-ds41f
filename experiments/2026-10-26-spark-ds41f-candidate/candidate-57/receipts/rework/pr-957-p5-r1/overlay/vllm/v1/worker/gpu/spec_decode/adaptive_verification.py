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
# and dense weight tiles at a reduced marginal cost. On TP3 each rank holds a
# wider MoE shard (768 vs 576) and 24 MLA heads (2 head tiles vs 1), so rows
# past the knee of a multi-request step remain closer to the base marginal cost
# and single-request tail pruning uses a higher calibrated survival floor. A
# single request keeps SINGLE_REQ_TILE_SPILL_SCALE on every TP size.
HIGH_ROW_COST_KNEE = 16
HIGH_ROW_COST_SCALE = 0.55
HIGH_ROW_COST_SCALE_TP3 = 0.85
SINGLE_REQ_TILE_SPILL_SCALE = 2.0
MIN_CALIBRATED_DEAD_ROWS_TAU_TP3 = 0.12


def _tp_world_size() -> int:
    try:
        return int(get_tp_group().world_size)
    except Exception:
        return 1


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
        tp_size = _tp_world_size()
        self.high_row_cost_knee = HIGH_ROW_COST_KNEE
        self.high_row_cost_scale = (
            HIGH_ROW_COST_SCALE_TP3 if tp_size == 3 else HIGH_ROW_COST_SCALE
        )
        self.single_req_tile_spill_scale = SINGLE_REQ_TILE_SPILL_SCALE
        self.min_calibrated_dead_rows_tau = (
            MIN_CALIBRATED_DEAD_ROWS_TAU_TP3 if tp_size == 3 else 0.0
        )
        self.tp_invariant_confidence = tp_invariant_confidence
        self.calibrated_confidence = calibrated_confidence
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

    def _step_cost(
        self, rows: np.ndarray, *, num_sampling_requests: int = 1
    ) -> np.ndarray:
        """Relative verification step cost with a 16-row Tensor Core MMA tile knee.

        Across concurrent decode requests (``num_sampling_requests > 1``), rows
        above ``HIGH_ROW_COST_KNEE`` (16) share already-launched expert and
        dense weight tiles at a discounted marginal cost (``high_row_cost_scale``,
        ``HIGH_ROW_COST_SCALE_TP3`` on TP3). For a single decode request
        (``num_sampling_requests <= 1``) on every TP size, crossing
        16 rows spills 1 row into a second 16-row Tensor Core MMA wave and the
        24-row CUDA graph, facing a higher marginal cost unless sustained copy
        acceptance justifies the second wave.
        """
        knee = getattr(self, "high_row_cost_knee", HIGH_ROW_COST_KNEE)
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

    def effective_dead_rows_tau(self) -> float:
        """Survival threshold for dead-row pruning on host and device.

        Raw ``confidence_head`` survival is inflated by ~3x at tail positions,
        which ``_DEAD_ROWS_TAU`` (0.20) was scaled to cut. Once online IRLS
        calibration fits the per-position intercepts, survival probabilities are
        already on the true acceptance scale, so the threshold matches the
        single-request marginal row-cost break-even ``2 * row_cost / (1 + 3 * row_cost)``
        instead of double-counting the tail decay. On TP3 that break-even is
        floored at ``MIN_CALIBRATED_DEAD_ROWS_TAU_TP3``, never above ``tau``.
        """
        tau = float(getattr(self, "dead_rows_tau", 0.0))
        if tau <= 0.0:
            return 0.0
        calibrated = getattr(self, "calibrated_confidence", False)
        if callable(calibrated):
            calibrated = bool(calibrated())
        if calibrated and self.row_cost > 0.0:
            floor = float(getattr(self, "min_calibrated_dead_rows_tau", 0.0))
            return min(
                tau,
                max(floor, 2.0 * self.row_cost / (1.0 + 3.0 * self.row_cost)),
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
        ]
        survival_probability = np.cumprod(stale_confidences.astype(np.float64), axis=1)
        steps = np.arange(step_steps)
        valid = steps[None, :] < eligible_drafts[:, None]
        tau = self.effective_dead_rows_tau()
        if tau > 0:
            # Never admit draft slots on the host that the device dead-row
            # threshold will immediately mask as padding.
            valid &= survival_probability >= tau
        scores = np.sort(survival_probability[valid])[::-1]
        max_draft_budget = min(
            int(eligible_drafts.sum()),
            max(0, self._max_total_logits - num_reqs * self.num_bonus_tokens),
        )
        scores = scores[:max_draft_budget]
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
        expected_tokens = np.concatenate(
            ([num_sampling_requests], num_sampling_requests + np.cumsum(scores))
        )
        rows = int(num_non_draft_tokens.sum()) + np.arange(len(scores) + 1)
        utilities = expected_tokens / self._step_cost(
            rows, num_sampling_requests=int(num_sampling_requests)
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
