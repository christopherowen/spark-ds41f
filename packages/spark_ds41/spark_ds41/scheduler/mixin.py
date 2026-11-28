"""Spark scheduler helpers: prefill_yield=short (deep-context chunk stepping,
elapsed-stall K=2 decode interleave, short-probe admission), max_parallel_prefills,
and continues_prefill for late admission."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request


class SparkSchedulerMixin:
    _DEEP_CONTEXT_THRESHOLD_TOKENS: int = 262144
    _ULTRA_DEEP_CONTEXT_THRESHOLD_TOKENS: int = 524288
    _DEEP_CONTEXT_CHUNK_GRID: int = 2048
    _DEEP_CONTEXT_CHUNK_TIER: int = 4096
    _ULTRA_DEEP_CONTEXT_CHUNK_TIER: int = 2048
    _PREFILL_DECODE_INTERLEAVE_STEPS: int = 2
    _PREFILL_DECODE_STALL_THRESHOLD_S: float = 1.0
    _PREFILL_STEP_FIXED_S: float = 0.1517
    _PREFILL_STEP_MARGINAL_S_PER_TOK: float = 1.969e-4
    _PREFILL_STEP_ATTN_S_PER_TOK2: float = 2.044e-9
    _PREFILL_WALL_CLOCK_MIN_S: float = 0.05

    def _spark_init_scheduler(self) -> None:
        self.prefill_yield: bool = (
            getattr(self.scheduler_config, "prefill_yield", "none") == "short"
        )
        self.prefill_yield_limit_s: float = float(
            getattr(self.scheduler_config, "prefill_yield_limit_s", 10.0)
        )
        self.max_parallel_prefills: int = int(
            getattr(self.scheduler_config, "max_parallel_prefills", 0) or 0
        )
        self._prefill_yield_since: dict[str, float] = {}
        self._last_prefill_step_at: float | None = None
        self._last_prefill_step_est_s: float = 0.0
        self._last_prefill_output_id: int | None = None
        self._prefill_stall_accum_s: float = 0.0
        self._prefill_decode_interleave_remaining: int = 0

    def continues_prefill(self, scheduler_output: SchedulerOutput) -> bool:
        """Whether a scheduled step carries a prompt chunk that is not its
        request's last, so late admission decides the next step immediately."""
        for req_id in scheduler_output.num_scheduled_tokens:
            req = self.requests.get(req_id)
            if req is not None and req.is_prefill_chunk:
                return True
        return False

    def _spark_context_aware_chunk_threshold(
        self,
        request: Request,
        base_threshold: int,
        num_runnable_decodes: int,
    ) -> int:
        if (
            base_threshold <= 0
            or num_runnable_decodes <= 0
            or not getattr(self, "prefill_yield", False)
        ):
            return base_threshold
        prompt_len = request.num_prompt_tokens
        if prompt_len >= self._ULTRA_DEEP_CONTEXT_THRESHOLD_TOKENS:
            return min(base_threshold, self._ULTRA_DEEP_CONTEXT_CHUNK_TIER)
        if prompt_len >= self._DEEP_CONTEXT_THRESHOLD_TOKENS:
            return min(base_threshold, self._DEEP_CONTEXT_CHUNK_TIER)
        return base_threshold

    def _spark_estimate_prefill_step_s(
        self, chunk_tokens: int, computed_tokens: int
    ) -> float:
        if chunk_tokens <= 0:
            return 0.0
        avg_ctx = computed_tokens + 0.5 * chunk_tokens
        return (
            self._PREFILL_STEP_FIXED_S
            + self._PREFILL_STEP_MARGINAL_S_PER_TOK * chunk_tokens
            + self._PREFILL_STEP_ATTN_S_PER_TOK2 * chunk_tokens * avg_ctx
        )

    def _spark_prefill_yield_expired(self, request_id: str, now: float) -> bool:
        since = self._prefill_yield_since.get(request_id)
        return since is not None and (now - since) >= self.prefill_yield_limit_s

    def _spark_prefill_finishes_in_chunk(
        self,
        request: Request,
        num_computed_tokens: int,
        long_prefill_threshold: int,
        token_budget: int,
    ) -> bool:
        remaining = request.num_tokens - num_computed_tokens
        if remaining <= 0:
            return True
        cap = token_budget
        if long_prefill_threshold > 0:
            cap = min(cap, long_prefill_threshold)
        return remaining <= cap

    def _spark_short_prefill_waiting(
        self,
        long_prefill_threshold: int,
        token_budget: int,
    ) -> bool:
        if not self.waiting:
            return False
        cap = token_budget
        if long_prefill_threshold > 0:
            cap = min(cap, long_prefill_threshold)
        for req in self.waiting:
            remaining = req.num_tokens - req.num_computed_tokens
            if 0 < remaining <= cap:
                return True
        return False

    def _spark_begin_step_yield(
        self,
        long_prefill_threshold: int,
        token_budget: int,
    ) -> tuple[float, bool, bool, bool, int]:
        """Return (yield_now, interleave_decode_due, short_waiting, yield_long, num_runnable_decodes)."""
        num_runnable_decodes = sum(1 for r in self.running if not r.is_prefill_chunk)
        if not getattr(self, "prefill_yield", False):
            return 0.0, False, False, False, num_runnable_decodes

        yield_now = time.monotonic()
        if self._last_prefill_step_at is not None:
            wall_dt = max(0.0, yield_now - self._last_prefill_step_at)
            prefill_still_in_flight = (
                self.scheduler_config.async_scheduling
                and self._last_prefill_output_id is not None
            )
            step_stall_s = (
                wall_dt
                if (
                    not prefill_still_in_flight
                    and wall_dt >= self._PREFILL_WALL_CLOCK_MIN_S
                )
                else max(wall_dt, self._last_prefill_step_est_s)
            )
            self._prefill_stall_accum_s += step_stall_s
            self._last_prefill_step_at = None
            self._last_prefill_step_est_s = 0.0
            self._last_prefill_output_id = None
            if self._prefill_stall_accum_s >= self._PREFILL_DECODE_STALL_THRESHOLD_S:
                self._prefill_decode_interleave_remaining = (
                    self._PREFILL_DECODE_INTERLEAVE_STEPS
                )
                self._prefill_stall_accum_s = 0.0

        has_eligible_decode = num_runnable_decodes > 0
        has_inflight_multi_chunk_prefill = bool(self._inflight_prefills) or any(
            r.is_prefill_chunk for r in self.running
        )
        if not has_eligible_decode or not has_inflight_multi_chunk_prefill:
            self._prefill_decode_interleave_remaining = 0
            self._prefill_stall_accum_s = 0.0
            self._last_prefill_output_id = None

        interleave_decode_due = (
            has_eligible_decode
            and has_inflight_multi_chunk_prefill
            and self._prefill_decode_interleave_remaining > 0
        )
        admission_open = (
            len(self.running) + self.num_waiting_for_streaming_input
            < self.max_num_running_reqs
        )
        short_waiting = (
            admission_open
            and bool(self.waiting)
            and self._spark_short_prefill_waiting(long_prefill_threshold, token_budget)
        )
        yield_long = interleave_decode_due or short_waiting
        return (
            yield_now,
            interleave_decode_due,
            short_waiting,
            yield_long,
            num_runnable_decodes,
        )

    def _spark_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        if not getattr(self, "prefill_yield", False):
            return
        max_prefill_est = 0.0
        scheduled_any_prefill = False
        for req_id, num_toks in scheduler_output.num_scheduled_tokens.items():
            req = self.requests.get(req_id)
            if req is None:
                continue
            if req.is_prefill_chunk or (
                num_toks > 1 and req.num_computed_tokens <= req.num_prompt_tokens
            ):
                scheduled_any_prefill = True
                self._prefill_yield_since.pop(req_id, None)
                prev_computed = max(0, req.num_computed_tokens - num_toks)
                est = self._spark_estimate_prefill_step_s(num_toks, prev_computed)
                if est > max_prefill_est:
                    max_prefill_est = est
        if scheduled_any_prefill:
            self._last_prefill_step_at = time.monotonic()
            self._last_prefill_step_est_s = max_prefill_est
            self._last_prefill_output_id = id(scheduler_output)
        elif self._prefill_decode_interleave_remaining > 0:
            self._prefill_decode_interleave_remaining -= 1

    def _spark_on_output(self, scheduler_output: SchedulerOutput) -> None:
        if (
            getattr(self, "_last_prefill_output_id", None) is not None
            and self._last_prefill_output_id == id(scheduler_output)
        ):
            self._last_prefill_output_id = None

    def _spark_forget_request(self, request_id: str) -> None:
        if hasattr(self, "_prefill_yield_since"):
            self._prefill_yield_since.pop(request_id, None)


    def _spark_compute_copy_rows_left(self) -> int:
        if not getattr(self, "copy_drafts", False):
            return 0
        if any(getattr(r, "is_prefill_chunk", False) for r in self.running):
            return 0
        slots = min(
            self.max_num_running_reqs,
            len(self.running) + len(self.waiting) + len(getattr(self, "skipped_waiting", ())),
        )
        uniform_rows = slots * (1 + self.num_spec_tokens)
        return max(0, getattr(self, "max_decode_rows", uniform_rows) - uniform_rows)

    def _spark_widen_copy_placeholders(self, request: Any, rows_left: int) -> int:
        placeholders = request.spec_token_ids
        if (
            not placeholders
            or len(placeholders) != self.num_spec_tokens
            or placeholders[-1] != -1
        ):
            return 0
        alloc = getattr(request, "copy_draft_alloc", 0)
        if alloc <= self.num_spec_tokens:
            return 0
        max_new_tokens = (
            self._max_decode_new_tokens(request)
            if hasattr(self, "_max_decode_new_tokens")
            else None
        )
        if max_new_tokens is not None:
            alloc = min(alloc, max_new_tokens - 1)
        rows = max(0, min(alloc - self.num_spec_tokens, rows_left))
        if rows > 0:
            request.spec_token_ids = [-1] * (self.num_spec_tokens + rows)
        return rows
