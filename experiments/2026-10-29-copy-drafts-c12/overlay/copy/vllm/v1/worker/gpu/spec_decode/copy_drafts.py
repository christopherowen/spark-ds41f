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

import torch

from vllm.config.speculative import CopyDraftsConfig
from vllm.utils.torch_utils import async_tensor_h2d
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
        self.alloc = torch.zeros(max_reqs, dtype=torch.int32, device=device)
        # A step that verified no drafts: nothing to judge.
        self._no_drafts = torch.zeros(max_reqs, dtype=torch.int32, device=device)
        self._new_rows: list[int] = []

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
            self._new_rows.clear()

    def propose(
        self,
        idx_mapping: torch.Tensor,
        dspark_drafts: torch.Tensor,
        dspark_confidences: torch.Tensor,
        num_sampled: torch.Tensor,
        verified: torch.Tensor | None,
        dead_row_cut: torch.Tensor | None,
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
        """
        num_reqs = idx_mapping.shape[0]
        if verified is None:
            verified = self._no_drafts[:num_reqs]
        history = self.req_states.all_token_ids.gpu
        total_len = self.req_states.total_len.gpu
        sources = self.sources[:num_reqs]
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
            self.alloc[:num_reqs],
            self.min_match,
            self.initial,
        )
        return drafts, confidences, planned
