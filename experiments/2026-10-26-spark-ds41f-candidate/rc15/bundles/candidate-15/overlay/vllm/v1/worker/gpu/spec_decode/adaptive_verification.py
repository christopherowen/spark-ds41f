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
# Continuation-value weight: scores draft slot j by S_j + alpha * sum_{m>j} S_m so
# bridge tokens followed by high-confidence continuations survive myopic pruning
# while preserving non-increasing slot order along a request. 0 = off.
_CONTINUATION_ALPHA = float(
    os.environ.get("SPARK_DS41F_DSPARK_CONTINUATION_ALPHA", "0")
)

GRAPH_PAD_COST_FRACTION = 0.5
GEMM_TILE_ROW_EQUIV = 1.0
LM_HEAD_TILE_ROW_EQUIV = 1.0


def dead_rows_enabled() -> bool:
    """Whether an adaptive verification manager marks dead verification rows."""
    return _DEAD_ROWS_TAU > 0


def continuation_enabled() -> bool:
    """Whether an adaptive verification manager weights continuation yield."""
    return _CONTINUATION_ALPHA > 0


def continuation_alpha_enabled() -> bool:
    """Alias for continuation_enabled()."""
    return _CONTINUATION_ALPHA > 0


def effective_survival_scores(
    survival_probability: np.ndarray,
    valid: np.ndarray,
    continuation_alpha: float = 0.0,
) -> np.ndarray:
    """Score valid draft slots by prefix survival plus weighted suffix yield.

    Because prefix survival ``S_j`` is non-increasing and non-negative, the tail
    yield ``sum_{m>j} S_m`` is also non-increasing in ``j``, so ``S_j + alpha *
    sum_{m>j} S_m`` stays monotonically non-increasing along each request while
    preserving bridge tokens followed by high-confidence continuations.
    """
    if continuation_alpha <= 0.0:
        return survival_probability
    alpha = float(continuation_alpha)
    valid_survival = np.where(valid, survival_probability, 0.0)
    scores = survival_probability.copy()
    suffix = np.zeros(scores.shape[0], dtype=scores.dtype)
    for j in range(scores.shape[1] - 1, -1, -1):
        scores[:, j] = scores[:, j] + alpha * suffix
        suffix = suffix + valid_survival[:, j]
    return scores


if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.sampling_params import SamplingParams
    from vllm.v1.worker.gpu.attn_utils import AttentionCGSupportInfo
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.states import RequestState
    from vllm.v1.worker.utils import AttentionGroup


# Consecutive steps that may schedule drafts yet verify none before one step
# verifies a draft per request anyway (see ``get_num_tokens``).
EXPLORE_AFTER_IDLE_STEPS = 32


def resolve_adaptive_cudagraph_mode(
    mode: CUDAGraphMode, *, piecewise_capture_available: bool
) -> CUDAGraphMode:
    """Select a separate decode route for variable-length verification."""
    assert mode.has_full_cudagraphs()
    if mode == CUDAGraphMode.FULL_DECODE_ONLY or not piecewise_capture_available:
        return CUDAGraphMode.FULL_DECODE_ONLY
    return CUDAGraphMode.FULL_AND_PIECEWISE


# The reference ``tilelang_inputs.assign_draft_budget`` is tested against; the
# manager no longer launches it.
def _assign_draft_token_budget(
    confidence_probs: torch.Tensor,
    idx_mapping: torch.Tensor,
    capacities: torch.Tensor,
    draft_budget: int,
    num_steps: int,
    continuation_alpha: float = 0.0,
) -> None:
    """Admit the globally best `draft_budget` draft slots, in place.

    Every (request, step) slot is scored by its survival probability (plus
    ``continuation_alpha`` times its valid suffix survival sum when enabled), the
    running product of that request's per-position confidences, and the highest
    scores win. Both prefix survival and suffix survival sum decrease along a
    request, so a global top-k always admits continuously along steps within a
    request.

    `capacities` enters holding each request's scheduled draft count (which bounds its
    eligible slots) and leaves holding the admitted count. The caller only calls this
    when `draft_budget < capacities.sum()`, so every winner is a real draft slot.
    """
    survival = confidence_probs[idx_mapping].cumprod(dim=1)
    steps = torch.arange(num_steps, device=survival.device)
    # Out-of-range slots score -inf so they never outrank a real draft.
    out_of_range = steps[None, :] >= capacities[:, None]
    if continuation_alpha > 0.0:
        alpha = float(np.float32(continuation_alpha))
        valid_survival = survival.masked_fill(out_of_range, 0.0)
        suffix = torch.zeros_like(survival[:, 0])
        for j in range(num_steps - 1, -1, -1):
            score_j = survival[:, j] + alpha * suffix
            suffix = suffix + valid_survival[:, j]
            survival[:, j] = score_j
    survival = survival.masked_fill(out_of_range, -float("inf"))
    flat = survival.flatten()
    winners = flat.topk(draft_budget).indices
    admitted = torch.zeros_like(flat, dtype=torch.bool).index_fill_(0, winners, True)
    torch.sum(admitted.view_as(survival), dim=1, dtype=capacities.dtype, out=capacities)


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
        capture_sizes: Iterable[int] = (),
        tile_cost: float = 0.0,
        moe_row_cost: float = 0.0,
        continuation_alpha: float | None = None,
        use_capture_step_cost: bool = True,
        filter_dead_rows_on_host: bool = True,
        clamp_verified_to_dead_cut: bool = True,
    ):
        self.req_states = req_states
        self.num_speculative_steps = req_states.num_speculative_steps
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
        self.tile_cost = tile_cost
        self.moe_row_cost = moe_row_cost
        self.use_capture_step_cost = use_capture_step_cost
        self.filter_dead_rows_on_host = filter_dead_rows_on_host
        self.clamp_verified_to_dead_cut = clamp_verified_to_dead_cut
        self._capture_sizes: tuple[int, ...] = tuple(
            sorted(int(s) for s in capture_sizes if int(s) > 0)
        )
        self.dead_rows_tau = _DEAD_ROWS_TAU
        self._dead_row_cut = torch.empty(
            req_states.max_num_reqs, dtype=torch.int32, device=device
        )
        self._active_dead_row_reqs = 0
        if self.dead_rows_tau > 0:
            logger.info(
                "DSpark dead verification rows below survival %.3g", self.dead_rows_tau
            )
        self.continuation_alpha = (
            _CONTINUATION_ALPHA
            if continuation_alpha is None
            else float(continuation_alpha)
        )
        if self.continuation_alpha > 0:
            logger.info(
                "DSpark continuation-weighted survival alpha=%.3g",
                self.continuation_alpha,
            )
        self.query_start_loc = query_start_loc
        self._batch_budget: tuple[dict[str, int], dict[str, int], int] | None = None
        max_num_reqs = req_states.max_num_reqs
        # Current per-slot confidences, and with copy drafts (planned_confidences)
        # the confidences a later step's budget reads: a copy's are those of the
        # proposal after the current one, whose rows the scheduler lays out from
        # the same copy state. One extra sentinel row (+inf) pins seeded sampled
        # requests' per-request budgets in mixed batches.
        self._confidence_buffer = torch.empty(
            (1 + planned_confidences, max_num_reqs + 1, self.num_speculative_steps),
            dtype=torch.float32,
            device=device,
        )
        self._confidence_buffer[:, max_num_reqs].fill_(float("inf"))
        self._confidence_probs_with_sentinel = self._confidence_buffer[0]
        self._confidence_probs = self._confidence_probs_with_sentinel[:max_num_reqs]
        self._planned_probs = self._confidence_buffer[-1, :max_num_reqs]
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
        self._seeded_sampled = np.zeros(max_num_reqs, dtype=bool)
        self._req_steps_without_drafts = np.zeros(max_num_reqs, dtype=np.int32)
        self._req_stale_confidences = np.ones(
            (max_num_reqs, self.num_speculative_steps), dtype=np.float32
        )
        self._next_stale_confidences = np.ones(
            (max_num_reqs, self.num_speculative_steps), dtype=np.float32
        )
        self._has_next_confidences = np.zeros(max_num_reqs, dtype=bool)
        self._in_flight_slots: list[int] = []
        self._has_recorded_confidences = False
        self._stale_idx = 0
        for slot in self._stale_confidences:
            slot.np.fill(1.0)

    def add_request(
        self, req_idx: int, sampling_params: "SamplingParams | None" = None
    ) -> None:
        self._stale_confidences[self._stale_idx].np[req_idx].fill(1.0)
        if req_idx not in self._pending_resets:
            self._pending_resets.append(req_idx)
        self._new_rows.append(req_idx)
        is_seeded = (
            sampling_params is not None
            and getattr(sampling_params, "seed", None) is not None
            and float(getattr(sampling_params, "temperature", 0.0)) > 0.0
        )
        if hasattr(self, "_seeded_sampled"):
            self._seeded_sampled[req_idx] = is_seeded
        if hasattr(self, "_req_steps_without_drafts"):
            self._req_steps_without_drafts[req_idx] = 0
        if hasattr(self, "_req_stale_confidences"):
            self._req_stale_confidences[req_idx].fill(1.0)
        if hasattr(self, "_next_stale_confidences"):
            self._next_stale_confidences[req_idx].fill(1.0)
        if hasattr(self, "_has_next_confidences"):
            self._has_next_confidences[req_idx] = False
        if hasattr(self, "_in_flight_slots") and req_idx in self._in_flight_slots:
            self._in_flight_slots.remove(req_idx)

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
        if (
            capture_sizes
            and not getattr(self, "_capture_sizes", ())
            and getattr(self, "use_capture_step_cost", True)
        ):
            self._capture_sizes = tuple(
                sorted(int(s) for s in capture_sizes if int(s) > 0)
            )
        max_num_tokens = self.req_states.max_num_batched_tokens
        for num_tokens in preparation_tail_sizes(capture_sizes, max_num_tokens):
            run_dummy(num_tokens=num_tokens)

    def padded_capture_sizes(self, rows: np.ndarray) -> np.ndarray:
        """Return the padded CUDA-graph capture size (or exact eager row count)
        that executes for each candidate ``rows`` count."""
        rows = np.asarray(rows, dtype=np.int64)
        capture_sizes = getattr(self, "_capture_sizes", ())
        if not capture_sizes:
            return rows
        captures = np.asarray(capture_sizes, dtype=np.int64)
        idx = np.searchsorted(captures, rows, side="left")
        in_graph = idx < len(captures)
        clamped_idx = np.minimum(idx, len(captures) - 1)
        return np.where(in_graph, captures[clamped_idx], rows)

    effective_rows = padded_capture_sizes

    def step_costs(
        self,
        rows: np.ndarray,
        scores: np.ndarray | None = None,
        *,
        use_capture_sizes: bool = True,
    ) -> np.ndarray:
        """Piecewise step-cost model for candidate row budgets ``rows[0..K]``.

        When ``_capture_sizes`` is populated or ``dead_rows_tau > 0``:
        - Dense projections, MLA and shared experts pad up to ``capture_size(rows)``
          and pay a discrete step when crossing 16-row GEMM tiles (``(padded - 1) // 16``)
          or 64-row LM-head tiles (``(padded - 1) // 64``).
        - Routed MoE skips trailing CUDA-graph padding rows and dead verification
          rows (``scores < dead_rows_tau``), so only ``live_rows`` pay ``moe_row_cost``.
        When neither is active, reduces to ``1.0 + row_cost * rows``.
        """
        rows = np.asarray(rows, dtype=np.int64)
        capture_sizes = (
            getattr(self, "_capture_sizes", ())
            if use_capture_sizes and getattr(self, "use_capture_step_cost", True)
            else ()
        )
        dead_rows_tau = (
            getattr(self, "dead_rows_tau", 0.0)
            if getattr(self, "filter_dead_rows_on_host", True)
            else 0.0
        )
        if not capture_sizes and dead_rows_tau <= 0.0:
            return 1.0 + self.row_cost * rows

        padded_rows = self.padded_capture_sizes(rows) if capture_sizes else rows
        if capture_sizes:
            gemm_tiles = np.maximum(0, padded_rows - 1) // 16
            head_tiles = np.maximum(0, padded_rows - 1) // 64
        else:
            gemm_tiles = np.zeros_like(padded_rows)
            head_tiles = np.zeros_like(padded_rows)

        if dead_rows_tau > 0.0 and scores is not None and len(scores) > 0:
            live_drafts = np.cumsum(scores >= dead_rows_tau, dtype=rows.dtype)
            live_rows = int(rows[0]) + np.concatenate(
                (np.zeros(1, dtype=rows.dtype), live_drafts)
            )
        else:
            live_rows = rows

        if not capture_sizes:
            return 1.0 + self.row_cost * live_rows

        graph_step_costs = 1.0 + self.row_cost * (
            GRAPH_PAD_COST_FRACTION * padded_rows
            + GEMM_TILE_ROW_EQUIV * gemm_tiles
            + LM_HEAD_TILE_ROW_EQUIV * head_tiles
        )
        moe_row_cost = (1.0 - GRAPH_PAD_COST_FRACTION) * self.row_cost
        return graph_step_costs + moe_row_cost * live_rows

    def _budget_for_scores(
        self,
        scores: np.ndarray,
        num_sampling_requests: int,
        num_non_draft_rows: int,
        *,
        use_capture_sizes: bool = True,
    ) -> int:
        dead_rows_tau = (
            getattr(self, "dead_rows_tau", 0.0)
            if getattr(self, "filter_dead_rows_on_host", True)
            else 0.0
        )
        if dead_rows_tau > 0.0 and len(scores) > 0:
            live_scores = np.where(scores >= dead_rows_tau, scores, 0.0)
        else:
            live_scores = scores
        expected_tokens = np.concatenate(
            ([num_sampling_requests], num_sampling_requests + np.cumsum(live_scores))
        )
        rows = num_non_draft_rows + np.arange(len(scores) + 1)
        utilities = expected_tokens / self.step_costs(
            rows, scores, use_capture_sizes=use_capture_sizes
        )
        return int(np.argmax(utilities))

    def load_draft_budget_kernels(self) -> None:
        """Run the draft-budget selection once at its smallest and largest shapes,
        so its kernels are loaded at startup rather than on the first step that
        trims a budget."""
        device = self._confidence_probs.device
        steps = self.num_speculative_steps
        probs = torch.ones_like(self._confidence_probs)
        max_reqs = probs.shape[0]
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
                probs, idx_mapping, capacities, draft_budget, steps
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
        has_seeded = bool(
            getattr(self, "_seeded_sampled", None) is not None
            and np.any(self._seeded_sampled)
        )
        mid_prefill_slots: list[int] = []
        if has_seeded:
            if (
                hasattr(input_batch, "idx_mapping_np")
                and hasattr(input_batch, "num_computed_tokens_np")
                and hasattr(input_batch, "num_scheduled_tokens")
                and hasattr(input_batch, "prefill_len_np")
            ):
                completed_mask = (
                    input_batch.num_computed_tokens_np + input_batch.num_scheduled_tokens
                    >= input_batch.prefill_len_np
                )
                completed_slots = set(
                    input_batch.idx_mapping_np[completed_mask].tolist()
                )
                mid_prefill_slots = input_batch.idx_mapping_np[
                    ~completed_mask
                ].tolist()
            else:
                completed_slots = set(input_batch.idx_mapping.tolist())
            if self._pending_resets:
                self._stale_confidences[ready_idx].np[self._pending_resets] = 1.0
                self._pending_resets = [
                    r for r in self._pending_resets if r not in completed_slots
                ]
            for slot in mid_prefill_slots:
                if slot not in self._pending_resets:
                    self._pending_resets.append(slot)
            req_stale = getattr(self, "_req_stale_confidences", None)
            if req_stale is not None:
                self._has_recorded_confidences = True
                for slot in self._in_flight_slots:
                    self._next_stale_confidences[slot] = self._stale_confidences[
                        ready_idx
                    ].np[slot]
                    self._has_next_confidences[slot] = True
                for slot in completed_slots:
                    if self._has_next_confidences[slot]:
                        req_stale[slot] = self._next_stale_confidences[slot]
                        self._has_next_confidences[slot] = False
                self._in_flight_slots = sorted(completed_slots)
        elif self._pending_resets:
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
        if mid_prefill_slots:
            rows = async_tensor_h2d(
                mid_prefill_slots,
                dtype=torch.int64,
                device=self._confidence_probs.device,
            )
            tilelang_inputs.fill_rows(rows, self._confidence_probs, 1.0)
            if self._confidence_buffer.shape[0] > 1:
                tilelang_inputs.fill_rows(rows, self._planned_probs, 1.0)
        # Replicated projections may differ through unordered floating-point
        # reductions. Both CPU graph selection and GPU request compaction must
        # consume one TP-wide confidence snapshot to issue matching collectives.
        tp_group = get_tp_group()
        if tp_group.world_size > 1:
            tp_group.broadcast(self._confidence_buffer, src=0)
        write_slot = self._stale_confidences[write_idx]
        write_slot.gpu.copy_(self._planned_probs)

        current_stream = torch.cuda.current_stream(self.req_states.device)
        self._copy_stream.wait_stream(current_stream)
        with stream(self._copy_stream, current_stream):
            write_slot.copy_to_cpu()
            self._copy_events[write_idx].record()

    def get_num_tokens(
        self,
        num_tokens_per_req: dict[str, int],
        draft_tokens: dict[str, list[int]],
    ) -> int:
        """Token count once the draft budget is trimmed to fit.

        The budget maximises expected tokens per unit of step cost: the step's
        sampling requests plus the survival probabilities of the admitted draft
        slots (best first), over ``step_costs(rows, scores)``, where ``rows``
        counts the step's non-draft rows and admitted draft rows; within the
        one-chunk logits limit. Seeded sampled requests budget their own draft
        rows from their own survival probabilities alone, so their verified
        blocks do not depend on co-scheduled requests. Stashes it in
        ``_batch_budget`` for the compaction and reallocation that follow in
        the same step.
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
        seeded_sampled = getattr(self, "_seeded_sampled", None)
        has_seeded = bool(seeded_sampled is not None and np.any(seeded_sampled))
        req_stale = getattr(self, "_req_stale_confidences", None)
        stale_confidences = (
            req_stale[slots]
            if has_seeded
            and req_stale is not None
            and getattr(self, "_has_recorded_confidences", True)
            else self._stale_confidences[self._stale_idx].np[slots]
        )
        survival_probability = np.cumprod(stale_confidences.astype(np.float64), axis=1)
        steps = np.arange(self.num_speculative_steps)
        max_draft_budget = min(
            int(scheduled_drafts.sum()),
            max(0, self._max_total_logits - num_reqs * self.num_bonus_tokens),
        )
        is_sampling = (
            self.req_states.num_computed_tokens_np[slots] + num_non_draft_tokens
            >= self.req_states.prefill_len.np[slots]
        )
        effective_drafts = scheduled_drafts.copy()
        seeded_mask = (
            seeded_sampled[slots]
            if seeded_sampled is not None
            else np.zeros(num_reqs, dtype=bool)
        )
        dead_rows_tau = (
            getattr(self, "dead_rows_tau", 0.0)
            if getattr(self, "filter_dead_rows_on_host", True)
            else 0.0
        )
        continuation_alpha = getattr(self, "continuation_alpha", 0.0)
        pinned_budget = 0
        if np.any(seeded_mask):
            req_idle_steps = getattr(self, "_req_steps_without_drafts", None)
            if req_idle_steps is None:
                req_idle_steps = np.zeros(int(slots.max()) + 1, dtype=np.int32)
                self._req_steps_without_drafts = req_idle_steps
            for i in np.flatnonzero(seeded_mask):
                req_drafts = int(scheduled_drafts[i])
                req_max_budget = min(
                    req_drafts,
                    max(0, self._max_total_logits - self.num_bonus_tokens),
                )
                req_valid = steps[None, :] < req_drafts
                req_slot_scores = effective_survival_scores(
                    survival_probability[int(i) : int(i) + 1],
                    req_valid,
                    continuation_alpha,
                )
                if dead_rows_tau > 0.0:
                    req_valid = req_valid & (req_slot_scores >= dead_rows_tau)
                req_scores = np.sort(req_slot_scores[req_valid])[::-1][:req_max_budget]
                req_budget = self._budget_for_scores(
                    req_scores,
                    int(is_sampling[i]),
                    int(num_non_draft_tokens[i]),
                    use_capture_sizes=False,
                )
                slot = int(slots[i])
                req_idle = int(req_idle_steps[slot])
                if req_budget == 0 and req_max_budget > 0:
                    req_idle += 1
                    if req_idle >= EXPLORE_AFTER_IDLE_STEPS:
                        req_budget = min(1, req_max_budget)
                        req_idle = 0
                else:
                    req_idle = 0
                req_idle_steps[slot] = req_idle
                effective_drafts[i] = req_budget
                pinned_budget += req_budget

        unpinned_mask = ~seeded_mask
        unpinned_budget = 0
        if np.any(unpinned_mask):
            unpinned_drafts = scheduled_drafts[unpinned_mask]
            unpinned_non_draft = num_non_draft_tokens[unpinned_mask]
            unpinned_survival = survival_probability[unpinned_mask]
            valid = steps[None, :] < unpinned_drafts[:, None]
            slot_scores = effective_survival_scores(
                unpinned_survival,
                valid,
                continuation_alpha,
            )
            if dead_rows_tau > 0.0:
                # Never budget draft slots that the device dead-row cut will mark
                # dead: they cannot commit tokens beyond the first dead row and
                # would only inflate the selected CUDA-graph capture bucket.
                valid = valid & (slot_scores >= dead_rows_tau)
            scores = np.sort(slot_scores[valid])[::-1]
            unpinned_max_budget = min(
                int(unpinned_drafts.sum()),
                max(0, max_draft_budget - pinned_budget),
            )
            scores = scores[:unpinned_max_budget]
            num_sampling_requests = int(np.count_nonzero(is_sampling[unpinned_mask]))
            unpinned_budget = self._budget_for_scores(
                scores,
                num_sampling_requests,
                int(unpinned_non_draft.sum()),
            )
            # The confidences come from an estimator that learns only from admitted
            # drafts, so a long run of empty budgets would leave it without the
            # evidence to ever admit one again. Periodically verify one draft per
            # request to keep it grounded.
            idle = getattr(self, "_steps_without_drafts", 0)
            num_unpinned = int(np.count_nonzero(unpinned_mask))
            if unpinned_budget == 0 and unpinned_max_budget > 0:
                idle += 1
                if idle >= EXPLORE_AFTER_IDLE_STEPS:
                    unpinned_budget = min(num_unpinned, unpinned_max_budget)
                    idle = 0
            else:
                idle = 0
            self._steps_without_drafts = idle

        draft_budget = min(max_draft_budget, pinned_budget + unpinned_budget)
        num_drafts_per_req = {
            req_id: int(num_drafts)
            for req_id, num_drafts in zip(req_ids, effective_drafts, strict=True)
        }
        num_non_draft_tokens_per_req = {
            req_id: int(num_tokens)
            for req_id, num_tokens in zip(req_ids, num_non_draft_tokens, strict=True)
        }
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
        if self.dead_rows_tau <= 0:
            self._active_dead_row_reqs = 0
            return None
        self._active_dead_row_reqs = num_reqs
        continuation_alpha = getattr(self, "continuation_alpha", 0.0)
        if continuation_alpha > 0.0:
            return DeadRows(
                self._confidence_probs,
                is_padding,
                self._dead_row_cut[:num_reqs],
                float(np.log(self.dead_rows_tau)),
                float(continuation_alpha),
            )
        return DeadRows(
            self._confidence_probs,
            is_padding,
            self._dead_row_cut[:num_reqs],
            float(np.log(self.dead_rows_tau)),
        )

    def get_verified_draft_counts(
        self,
        num_reqs: int,
        dead_row_cut: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the device-selected verified draft count for each request.

        When dead-row marking ran this step, rows past ``dead_row_cut`` skipped
        routed MoE and cannot be committed as verified drafts, so the verified
        draft count is ``min(admitted, dead_row_cut)``.
        """
        admitted = self._batch_draft_capacity[:num_reqs]
        if (
            not getattr(self, "clamp_verified_to_dead_cut", True)
            or self.dead_rows_tau <= 0
            or getattr(self, "_active_dead_row_reqs", 0) != num_reqs
        ):
            return admitted
        cut = self._dead_row_cut if dead_row_cut is None else dead_row_cut
        if cut is None:
            return admitted
        torch.minimum(admitted, cut[:num_reqs], out=admitted)
        return admitted

    def reallocate_drafts(
        self, req_ids: list[str], idx_mapping: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, torch.Tensor]:
        """The step's logits offsets, query starts and draft budget, and its
        expanded state indices and positions (``expand_idx_mapping``'s), all
        from the admitted drafts."""
        self._active_dead_row_reqs = 0
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
        capacities = self._batch_draft_capacity[:num_reqs]
        if draft_budget:
            async_tensor_h2d(scheduled_drafts, out=capacities)
            if draft_budget < int(scheduled_drafts.sum()):
                probs = self._confidence_probs
                budget_idx_mapping = idx_mapping
                seeded_sampled = getattr(self, "_seeded_sampled", None)
                sentinel_probs = getattr(self, "_confidence_probs_with_sentinel", None)
                if seeded_sampled is not None and sentinel_probs is not None:
                    slots = np.fromiter(
                        (
                            self.req_states.req_id_to_index[req_id]
                            for req_id in req_ids
                        ),
                        dtype=np.int64,
                        count=num_reqs,
                    )
                    seeded_mask = seeded_sampled[slots]
                    if (
                        np.any(seeded_mask)
                        and int(scheduled_drafts[seeded_mask].sum()) <= draft_budget
                    ):
                        probs = sentinel_probs
                        slots[seeded_mask] = self.req_states.max_num_reqs
                        budget_idx_mapping = async_tensor_h2d(
                            slots,
                            dtype=idx_mapping.dtype,
                            device=idx_mapping.device,
                        )
                continuation_alpha = getattr(self, "continuation_alpha", 0.0)
                if continuation_alpha > 0.0:
                    tilelang_inputs.assign_draft_budget(
                        probs,
                        budget_idx_mapping,
                        capacities,
                        draft_budget,
                        self.num_speculative_steps,
                        continuation_alpha=continuation_alpha,
                    )
                else:
                    tilelang_inputs.assign_draft_budget(
                        probs,
                        budget_idx_mapping,
                        capacities,
                        draft_budget,
                        self.num_speculative_steps,
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
            self.num_speculative_steps + self.num_bonus_tokens,
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
        use_capture_step_cost=os.environ.get(
            "SPARK_DS41F_DSPARK_PIECEWISE_STEP_COST", "0"
        )
        == "1",
        filter_dead_rows_on_host=os.environ.get(
            "SPARK_DS41F_DSPARK_HOST_DEAD_ROWS", "0"
        )
        == "1",
        clamp_verified_to_dead_cut=os.environ.get(
            "SPARK_DS41F_DSPARK_CLAMP_DEAD_CAPACITY", "0"
        )
        == "1",
    )
