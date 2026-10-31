# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Copy drafts beside DSpark (``SpeculativeConfig.copy_drafts``).

After DSpark drafts, each request whose context ends with text it has seen
before (an exact suffix match of at least ``min_match`` tokens over its own
prompt and output) takes the tokens that followed the latest match in place of
DSpark's drafts, up to its cap; every other request keeps DSpark's. The copy's
confidences (the request's acceptance estimate at every draft) enter adaptive
verification's budget beside DSpark's, which decides how many rows a step
verifies; verification commits only the target's own tokens, so drafts change
the speed, never the text.

Everything is the request's own: the history is its row of the request state
up to its length, and its copy state (cap, acceptance estimate, pending
proposal) lives in its slot, reset when a request takes the slot. Nothing is
read from or kept for any other request.

The scheduler lays out a request's rows before the step that follows its
proposal is verified (async scheduling): each proposal also reports
``alloc``, the drafts of the proposal after it if this one is accepted in full,
which the scheduler gives the request's step after next (within its decode-row
envelope), and the planned confidences the budget reads at that step.
"""

from typing import Any

import numpy as np
import torch

from vllm.config.speculative import CopyDraftsConfig
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.gpu import tilelang_inputs
from vllm.v1.worker.gpu.spec_decode import tilelang_copy
from vllm.v1.worker.gpu.states import RequestState


class CopyDrafter:
    def __init__(
        self,
        config: CopyDraftsConfig,
        req_states: RequestState,
        num_draft_steps: int,
        device: torch.device,
    ):
        self.config = config
        self.req_states = req_states
        self.min_match = config.min_match
        self.initial = config.initial_drafts
        # The verification lookahead: the request state's draft width.
        self.lookahead = req_states.num_speculative_steps
        assert self.lookahead == max(config.max_drafts, num_draft_steps)
        self.num_draft_steps = num_draft_steps
        max_reqs = req_states.max_num_reqs
        levels = len(config.levels())
        columns = tilelang_copy.STATE_COLUMNS
        # Per slot: (cap doublings, drafts of the pending copy proposal) and
        # (accepted, judged) drafts, decayed. All zero is a request's start.
        self.int_state = torch.zeros(
            (max_reqs, columns), dtype=torch.int32, device=device
        )
        self.float_state = torch.zeros(
            (max_reqs, columns), dtype=torch.float32, device=device
        )
        # The step's proposal, in batch order.
        self.sources = torch.full(
            (max_reqs, tilelang_copy.CHUNKS, levels),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.drafts = torch.zeros(
            (max_reqs, self.lookahead), dtype=torch.int64, device=device
        )
        self.confidences = torch.zeros(
            (max_reqs, self.lookahead), dtype=torch.float32, device=device
        )
        self.planned = torch.zeros_like(self.confidences)
        # Ring of pre-allocated buffers and events for async output delivery of
        # alloc and slot_draft_limits without per-step tensor or event allocation.
        self._ring_size = 3
        self._ring_idx = 0
        self._alloc_buffers = [
            CpuGpuBuffer(max_reqs, dtype=torch.int32, device=device)
            for _ in range(self._ring_size)
        ]
        self._limits_staging = [
            CpuGpuBuffer(max_reqs, dtype=torch.int32, device=device)
            for _ in range(self._ring_size)
        ]
        for buf in self._limits_staging:
            buf.np.fill(self.lookahead)
        self._ring_events = [
            torch.cuda.Event(blocking=True) for _ in range(self._ring_size)
        ]
        self.alloc = self._alloc_buffers[0].gpu
        self._select_alloc = torch.zeros(max_reqs, dtype=torch.int32, device=device)
        # Actual valid draft count per slot from the latest proposal, mirrored to
        # host non-blockingly so the adaptive budget caps stale wide schedules.
        self.slot_draft_limits = CpuGpuBuffer(
            max_reqs, dtype=torch.int32, device=device
        )
        self.slot_draft_limits.np.fill(self.lookahead)
        self.slot_draft_limits.gpu.fill_(self.lookahead)
        self._limits_event: torch.cuda.Event | None = None
        self._limits_staging_np: np.ndarray = self._limits_staging[0].np
        self._pending_limit_resets: list[int] = []
        # A step that verified no drafts: nothing to judge.
        self._no_drafts = torch.zeros(max_reqs, dtype=torch.int32, device=device)
        self._new_rows: list[int] = []

    def sync_slot_draft_limits(self) -> np.ndarray:
        if self._limits_event is not None and self._limits_event.query():
            np.copyto(self.slot_draft_limits.np, self._limits_staging_np)
            if self._pending_limit_resets:
                self.slot_draft_limits.np[self._pending_limit_resets] = self.lookahead
                self._pending_limit_resets.clear()
            self._limits_event = None
        return self.slot_draft_limits.np

    def register_warmups(self) -> None:
        """Declare the copy kernels and the copy state's reset."""
        tilelang_inputs.declare(
            lambda: (
                tilelang_copy.copy_calls(
                    self.min_match, self.lookahead, self.num_draft_steps, self.initial
                )
                + tilelang_inputs.fill_rows_calls(torch.int64, torch.int32)
                + tilelang_inputs.fill_rows_calls(torch.int64, torch.float32)
            )
        )

    def add_request(self, req_idx: int) -> None:
        self.sync_slot_draft_limits()[req_idx] = self.lookahead
        if self._limits_event is not None:
            self._pending_limit_resets.append(req_idx)
        self._new_rows.append(req_idx)

    def apply_staged_writes(self) -> None:
        """Reset the new requests' copy state: nothing of a slot's previous
        request carries over."""
        if self._new_rows:
            rows = async_tensor_h2d(
                self._new_rows, dtype=torch.int64, device=self.int_state.device
            )
            tilelang_inputs.fill_rows(rows, self.int_state, 0)
            tilelang_inputs.fill_rows(rows, self.float_state, 0.0)
            tilelang_inputs.fill_rows(
                rows, self.slot_draft_limits.gpu[:, None], self.lookahead
            )
            self._new_rows.clear()

    def prepare_alloc(
        self,
        idx_mapping: torch.Tensor,
        num_sampled: torch.Tensor,
        verified: torch.Tensor | None,
        dead_row_cut: torch.Tensor | None,
        *,
        match_idx_mapping: torch.Tensor | None = None,
    ) -> None:
        """Match the batch's updated histories and write ``alloc[:num_reqs]`` and
        ``slot_draft_limits.gpu`` before the drafter runs, so the async output
        copy does not wait behind the drafter."""
        self.sync_slot_draft_limits()
        self._ring_idx = (self._ring_idx + 1) % self._ring_size
        self.alloc = self._alloc_buffers[self._ring_idx].gpu
        num_reqs = idx_mapping.shape[0]
        if verified is None:
            verified = self._no_drafts[:num_reqs]
        history = self.req_states.all_token_ids.gpu
        total_len = self.req_states.total_len.gpu
        sources = self.sources[:num_reqs]
        tilelang_copy.copy_match(
            idx_mapping if match_idx_mapping is None else match_idx_mapping,
            history,
            total_len,
            sources,
            self.min_match,
        )
        tilelang_copy.copy_alloc(
            idx_mapping,
            total_len,
            sources,
            num_sampled,
            verified,
            verified if dead_row_cut is None else dead_row_cut,
            self.int_state,
            self.alloc[:num_reqs],
            self.slot_draft_limits.gpu,
            self.min_match,
            self.lookahead,
            self.num_draft_steps,
            self.initial,
        )

    def attach_async_output(
        self,
        async_output: Any,
        num_reqs: int,
        copy_stream: torch.cuda.Stream,
        main_stream: torch.cuda.Stream,
    ) -> None:
        """Queue async D2H copy of this step's ``alloc`` and ``slot_draft_limits``
        using pre-allocated ring buffers and events."""
        idx = self._ring_idx
        alloc_buf = self._alloc_buffers[idx]
        limits_staging = self._limits_staging[idx]
        event = self._ring_events[idx]
        async_output.attach_copy_draft_alloc(
            alloc_buf,
            copy_stream,
            main_stream,
            num_reqs=num_reqs,
            slot_draft_limits_gpu=self.slot_draft_limits.gpu,
            slot_draft_limits_cpu=limits_staging.cpu,
            alloc_event=event,
        )
        self._limits_staging_np = limits_staging.np
        self._limits_event = event
        self._pending_limit_resets.clear()

    def propose(
        self,
        idx_mapping: torch.Tensor,
        dspark_drafts: torch.Tensor,
        dspark_confidences: torch.Tensor,
        num_sampled: torch.Tensor,
        verified: torch.Tensor | None,
        dead_row_cut: torch.Tensor | None,
        *,
        matched: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The batch's drafts, confidences and planned confidences
        (``[num_reqs, lookahead]``), and ``alloc[:num_reqs]``.

        Args:
            idx_mapping: The batch's request slots.
            dspark_drafts: DSpark's drafts, ``[num_reqs, num_draft_steps]``.
            dspark_confidences: Their confidences (``[>= num_reqs, steps]``).
            num_sampled: The tokens this step committed per request.
            verified: The drafts this step verified per request (adaptive
                verification's admitted counts), None if it verified none.
            dead_row_cut: The live drafts per request, None without dead rows.
            matched: Whether ``prepare_alloc`` already matched this step's
                histories into ``sources[:num_reqs]``.
        """
        num_reqs = idx_mapping.shape[0]
        if verified is None:
            verified = self._no_drafts[:num_reqs]
        history = self.req_states.all_token_ids.gpu
        total_len = self.req_states.total_len.gpu
        sources = self.sources[:num_reqs]
        if not matched:
            tilelang_copy.copy_match(
                idx_mapping, history, total_len, sources, self.min_match
            )
        drafts = self.drafts[:num_reqs]
        confidences = self.confidences[:num_reqs]
        planned = self.planned[:num_reqs]
        tilelang_copy.copy_select(
            idx_mapping,
            history,
            total_len,
            sources,
            dspark_drafts[:num_reqs],
            dspark_confidences[:num_reqs, : self.num_draft_steps],
            num_sampled,
            verified,
            verified if dead_row_cut is None else dead_row_cut,
            self.int_state,
            self.float_state,
            drafts,
            confidences,
            planned,
            self._select_alloc[:num_reqs] if matched else self.alloc[:num_reqs],
            self.min_match,
            self.initial,
        )
        return drafts, confidences, planned
