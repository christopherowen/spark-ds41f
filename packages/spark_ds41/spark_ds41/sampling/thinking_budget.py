# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.sampling_params import SamplingParams
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor
from spark_ds41.sampling import tilelang_thinking
from vllm.v1.worker.gpu.states import RequestState

if TYPE_CHECKING:
    from vllm.config.reasoning import ReasoningConfig

_INT32_MAX = np.iinfo(np.int32).max
# The Triton marker cache's cold-scan block (tests launch the Triton kernels as
# references).
_COLD_SCAN_BLOCK = 1024


class ThinkingBudgetState:
    """Model Runner V2 state for per-request thinking token budgets."""

    def __init__(
        self,
        req_states: RequestState,
        reasoning_config: "ReasoningConfig | None",
    ):
        self.req_states = req_states
        self.max_num_reqs = req_states.max_num_reqs
        self.device = req_states.device

        start_ids = (
            []
            if reasoning_config is None
            else reasoning_config.reasoning_start_token_ids or []
        )
        end_ids = (
            []
            if reasoning_config is None
            else reasoning_config.reasoning_end_token_ids or []
        )
        natural_end_ids = (
            []
            if reasoning_config is None
            else reasoning_config.natural_reasoning_end_token_ids or []
        )
        self.enabled = bool(start_ids and end_ids and natural_end_ids)
        if not self.enabled:
            return

        self.thinking_token_budget = UvaBackedTensor(
            self.max_num_reqs, dtype=torch.int32
        )
        self.thinking_token_budget.np.fill(-1)
        self.thinking_token_budget.copy_to_uva()
        self.configured_budget = np.full(self.max_num_reqs, -1, dtype=np.int32)
        self.use_thinking_budget = np.zeros(self.max_num_reqs, dtype=bool)
        self._forced_reqs: set[int] = set()

        self.cached_last_start = torch.full(
            (self.max_num_reqs,), -1, dtype=torch.int32, device=self.device
        )
        self.cached_last_end = torch.full(
            (self.max_num_reqs,), -1, dtype=torch.int32, device=self.device
        )
        self.cached_scan_pos = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=self.device
        )
        self._reset_reqs: list[int] = []
        self._budget_dirty = False

        self.reasoning_start_token_ids = torch.tensor(
            start_ids, dtype=torch.int32, device=self.device
        )
        self.natural_reasoning_end_token_ids = torch.tensor(
            natural_end_ids, dtype=torch.int32, device=self.device
        )
        self.reasoning_end_token_ids = torch.tensor(
            end_ids, dtype=torch.int32, device=self.device
        )

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> None:
        if not self.enabled:
            return
        self._forced_reqs.discard(req_idx)
        budget = sampling_params.thinking_token_budget
        self.use_thinking_budget[req_idx] = budget is not None
        if budget is None:
            budget = -1
        else:
            budget = min(budget, _INT32_MAX)
            self._reset_reqs.append(req_idx)
        self.configured_budget[req_idx] = budget
        if self.thinking_token_budget.np[req_idx] != budget:
            self.thinking_token_budget.np[req_idx] = budget
            self._budget_dirty = True

    def set_forced_end_requests(
        self, req_indices: set[int]
    ) -> tuple[set[int], set[int]]:
        """Force immediate reasoning termination (budget=0) for ``req_indices``
        and restore configured budgets for any slots removed from the set."""
        if not self.enabled or (not req_indices and not self._forced_reqs):
            return set(), set()
        newly_forced = req_indices - self._forced_reqs
        unforced = self._forced_reqs - req_indices
        changed = newly_forced | unforced
        if not changed:
            return set(), set()
        self._forced_reqs = set(req_indices)
        for req_idx in newly_forced:
            self.use_thinking_budget[req_idx] = True
            if self.configured_budget[req_idx] < 0:
                self._reset_reqs.append(req_idx)
            if self.thinking_token_budget.np[req_idx] != 0:
                self.thinking_token_budget.np[req_idx] = 0
                self._budget_dirty = True
        for req_idx in unforced:
            cfg_budget = int(self.configured_budget[req_idx])
            self.use_thinking_budget[req_idx] = cfg_budget >= 0
            if self.thinking_token_budget.np[req_idx] != cfg_budget:
                self.thinking_token_budget.np[req_idx] = cfg_budget
                self._budget_dirty = True
        return changed, newly_forced

    def apply_staged_writes(self) -> None:
        if not self.enabled:
            return
        if self._reset_reqs:
            idx = async_tensor_h2d(
                self._reset_reqs, dtype=torch.int64, device=self.device
            )
            # The three caches' rows at idx in one launch (three index_fill_ before).
            tilelang_thinking.reset_markers(
                idx, self.cached_last_start, self.cached_last_end, self.cached_scan_pos
            )
            self._reset_reqs.clear()
        if self._budget_dirty:
            self.thinking_token_budget.copy_to_uva()
            self._budget_dirty = False

    def register_warmups(self, logits_strides: tuple[int, ...]) -> None:
        """Declare the marker cache and budget kernels for processed FP32 logits
        with each row stride in ``logits_strides``."""
        if not self.enabled:
            return
        tilelang_thinking.register_thinking(
            logits_strides=logits_strides,
            tokens_stride=self.req_states.all_token_ids.gpu.stride(0),
            start_len=self.reasoning_start_token_ids.numel(),
            natural_len=self.natural_reasoning_end_token_ids.numel(),
            end_len=self.reasoning_end_token_ids.numel(),
        )

    def apply(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
    ) -> None:
        if not self.enabled or not np.any(self.use_thinking_budget[idx_mapping_np]):
            return

        apply_thinking_budget(
            logits,
            idx_mapping,
            expanded_idx_mapping,
            self.thinking_token_budget.gpu,
            self.req_states.all_token_ids.gpu,
            self.req_states.total_len.gpu,
            input_ids,
            expanded_local_pos,
            self.cached_last_start,
            self.cached_last_end,
            self.cached_scan_pos,
            self.reasoning_start_token_ids,
            self.natural_reasoning_end_token_ids,
            self.reasoning_end_token_ids,
        )


def apply_thinking_budget(
    logits: torch.Tensor,
    req_ids: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    thinking_token_budget: torch.Tensor,
    all_token_ids: torch.Tensor,
    total_len: torch.Tensor,
    input_ids: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    cached_last_start: torch.Tensor,
    cached_last_end: torch.Tensor,
    cached_scan_pos: torch.Tensor,
    reasoning_start_token_ids: torch.Tensor,
    natural_reasoning_end_token_ids: torch.Tensor,
    reasoning_end_token_ids: torch.Tensor,
) -> None:
    tilelang_thinking.apply_thinking_budget(
        logits,
        req_ids,
        expanded_idx_mapping,
        thinking_token_budget,
        all_token_ids,
        total_len,
        input_ids,
        expanded_local_pos,
        cached_last_start,
        cached_last_end,
        cached_scan_pos,
        reasoning_start_token_ids,
        natural_reasoning_end_token_ids,
        reasoning_end_token_ids,
    )
