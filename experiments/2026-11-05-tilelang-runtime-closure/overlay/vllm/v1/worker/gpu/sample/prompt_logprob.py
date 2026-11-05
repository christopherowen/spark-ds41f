# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable, Sequence

import numpy as np
import torch

from vllm.config.model import LogprobsMode
from vllm.sampling_params import SamplingParams
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.sample import (
    tilelang_auxiliary,
    tilelang_logprob,
    tilelang_topk_ids,
)
from vllm.v1.worker.gpu.sample.logprob import (
    register_topk_scores_warmup,
)


class PromptLogprobsWorker:
    def __init__(self, max_num_reqs: int, logprobs_mode: LogprobsMode = "raw_logprobs"):
        self.max_num_reqs = max_num_reqs
        self.logprobs_mode = logprobs_mode

        self.uses_prompt_logprobs = np.zeros(self.max_num_reqs, dtype=bool)
        self.num_prompt_logprobs = np.zeros(self.max_num_reqs, dtype=np.int32)
        # req_idx -> list of in-progress LogprobsTensors
        self.in_progress_prompt_logprobs: dict[str, list[LogprobsTensors]] = {}

    def add_request(self, req_id: str, req_idx: int, sampling_params: SamplingParams):
        uses_prompt_logprobs = sampling_params.prompt_logprobs is not None
        self.uses_prompt_logprobs[req_idx] = uses_prompt_logprobs
        self.num_prompt_logprobs[req_idx] = sampling_params.prompt_logprobs or 0
        if uses_prompt_logprobs:
            self.in_progress_prompt_logprobs[req_id] = []

    def remove_request(self, req_id: str) -> None:
        self.in_progress_prompt_logprobs.pop(req_id, None)

    def register_warmups(
        self,
        *,
        logits: Sequence[tuple[torch.dtype, int]],
        vocab_size: int,
        all_token_ids_stride: int,
    ) -> None:
        """Register the prompt token gather and the scores of prompt logits from
        ``logits_fn`` (``(dtype, row stride)`` pairs)."""
        tilelang_auxiliary.register_prompt_ids(
            all_token_ids_stride=all_token_ids_stride
        )
        register_topk_scores_warmup(
            logits=logits,
            vocab_size=vocab_size,
            logits_mode=self.logprobs_mode in ("raw_logits", "processed_logits"),
        )

    def compute_prompt_logprobs(
        self,
        logits_fn: Callable[[torch.Tensor], torch.Tensor],
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        # [max_num_reqs, max_model_len]
        all_token_ids: torch.Tensor,
        # [max_num_reqs]
        num_computed_tokens: torch.Tensor,
        # [max_num_reqs]
        prompt_lens: np.ndarray,
    ) -> dict[str, LogprobsTensors]:
        idx_mapping_np = input_batch.idx_mapping_np
        needs_prompt_logprobs = self.uses_prompt_logprobs[idx_mapping_np]
        if not np.any(needs_prompt_logprobs):
            # Common case: No request asks for prompt logprobs.
            return {}

        num_prompt_logprobs = self.num_prompt_logprobs[idx_mapping_np]
        prompt_lens = prompt_lens[idx_mapping_np]
        computed_prefill = input_batch.num_computed_prefill_tokens_np
        includes_prompt = computed_prefill < prompt_lens
        # NOTE(woosuk): If the request was resumed after preemption, its prompt
        # logprobs must have been computed before preemption. Skip.
        resumed_after_prompt = prompt_lens < input_batch.prefill_len_np
        needs_prompt_logprobs &= includes_prompt & ~resumed_after_prompt
        if not np.any(needs_prompt_logprobs):
            return {}

        # get the maximum number in this batch
        requested_num_prompt_logprobs = num_prompt_logprobs[needs_prompt_logprobs]
        max_num_prompt_logprobs = (
            -1
            if np.any(requested_num_prompt_logprobs == -1)
            else int(requested_num_prompt_logprobs.max())
        )

        # Get the prompt logprobs token_ids.
        prompt_logprobs_token_ids = get_prompt_logprobs_token_ids(
            input_batch.num_tokens,
            input_batch.query_start_loc,
            input_batch.idx_mapping,
            num_computed_tokens,
            all_token_ids,
        )
        prompt_token_ids, prompt_logprobs, prompt_ranks = (
            compute_prompt_logprobs_with_chunking(
                prompt_logprobs_token_ids,
                hidden_states[: input_batch.num_tokens],
                logits_fn,
                max_num_prompt_logprobs,
                self.logprobs_mode,
            )
        )

        pos_after_step = computed_prefill + input_batch.num_scheduled_tokens
        is_prompt_chunked = pos_after_step < prompt_lens

        query_start_loc_np = input_batch.query_start_loc_np
        prompt_logprobs_dict: dict[str, LogprobsTensors] = {}
        for i, req_id in enumerate(input_batch.req_ids):
            if not needs_prompt_logprobs[i]:
                continue

            req_is_prompt_chunked = is_prompt_chunked[i]
            req_num_prompt_logprobs = int(num_prompt_logprobs[i])
            start_idx = query_start_loc_np[i]
            end_idx = query_start_loc_np[i + 1]
            assert start_idx < end_idx, (
                f"start_idx ({start_idx}) >= end_idx ({end_idx})"
            )
            if not req_is_prompt_chunked:
                end_idx -= 1

            width = (
                prompt_logprobs.shape[1]
                if req_num_prompt_logprobs == -1
                else req_num_prompt_logprobs + 1
            )
            # no logprobs if start_idx >= end_idx
            logprobs = (
                None
                if start_idx >= end_idx
                else LogprobsTensors(
                    logprob_token_ids=prompt_token_ids[start_idx:end_idx, :width],
                    logprobs=prompt_logprobs[start_idx:end_idx, :width],
                    selected_token_ranks=prompt_ranks[start_idx:end_idx],
                )
            )

            prompt_logprobs_list = self.in_progress_prompt_logprobs[req_id]
            if logprobs is not None and (req_is_prompt_chunked or prompt_logprobs_list):
                prompt_logprobs_list.append(logprobs)
            if req_is_prompt_chunked:
                # Prompt is chunked. Do not return the logprobs yet.
                continue

            if prompt_logprobs_list:
                # Merge the in-progress logprobs.
                logprobs = LogprobsTensors.cat(prompt_logprobs_list)
                prompt_logprobs_list.clear()

            if logprobs is None:
                continue

            prompt_logprobs_dict[req_id] = logprobs
        return prompt_logprobs_dict


def get_prompt_logprobs_token_ids(
    num_tokens: int,
    query_start_loc: torch.Tensor,
    idx_mapping: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    all_token_ids: torch.Tensor,
) -> torch.Tensor:
    token_ids = torch.empty(num_tokens, dtype=torch.int64, device=idx_mapping.device)
    tilelang_auxiliary.prompt_token_ids(
        token_ids, query_start_loc, idx_mapping, num_computed_tokens, all_token_ids
    )
    return token_ids


def compute_prompt_logprobs_with_chunking(
    prompt_token_ids: torch.Tensor,
    prompt_hidden_states: torch.Tensor,
    logits_fn: Callable[[torch.Tensor], torch.Tensor],
    num_prompt_logprobs: int,
    logprobs_mode: LogprobsMode = "raw_logprobs",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Bound the logits workspace while writing every chunk into its final rows.
    CHUNK_SIZE = 1024
    if not 0 <= num_prompt_logprobs <= tilelang_topk_ids.TOPK_IDS_MAX:
        raise ValueError("prompt logprobs exceed the configured top-k range")
    rows = prompt_token_ids.shape[0]
    shape = (rows, 1 + num_prompt_logprobs)
    token_ids = (
        torch.empty(shape, dtype=torch.int64, device=prompt_token_ids.device)
        if num_prompt_logprobs
        else prompt_token_ids.unsqueeze(-1)
    )
    scores = torch.empty(shape, dtype=torch.float32, device=prompt_token_ids.device)
    ranks = torch.empty(rows, dtype=torch.int64, device=prompt_token_ids.device)
    logits_mode = logprobs_mode in ("raw_logits", "processed_logits")
    for start in range(0, rows, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, rows)
        logits = logits_fn(prompt_hidden_states[start:end])
        selected = token_ids[start:end]
        sampled = prompt_token_ids[start:end]
        if num_prompt_logprobs:
            tilelang_topk_ids.logprob_token_ids(
                logits, num_prompt_logprobs, sampled, out=selected
            )
        if logits_mode:
            tilelang_logprob.gather_logits(logits, selected, out=scores[start:end])
        else:
            tilelang_logprob.topk_log_softmax(logits, selected, scores[start:end])
        tilelang_logprob.token_ranks(logits, sampled, ranks[start:end])

    return token_ids, scores, ranks


_TRITON_REFS = frozenset(("_prompt_logprobs_token_ids_kernel",))


def __getattr__(name: str):
    if name in _TRITON_REFS:
        from vllm.v1.worker.gpu import _triton_refs

        return getattr(_triton_refs, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
