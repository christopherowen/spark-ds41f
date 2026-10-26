# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

import numpy as np
import torch

from vllm.sampling_params import MAX_LOGPROB_TOKEN_IDS, SamplingParams
from vllm.triton_utils import tl, triton
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor, UvaBackedTensor
from vllm.v1.worker.gpu.sample import tilelang_auxiliary, tilelang_topk_ids

# Upper bound on the topk kernel's per-iteration gather width.
_MAX_TOPK_BLOCK = 1024


# The arithmetic the TileLang log-softmax (``tilelang_logprob.topk_log_softmax``)
# fixed, which its tests measure it against, launched as production launched it
# (BLOCK_SIZE=1024, TOPK_BLOCK_SIZE=_MAX_TOPK_BLOCK, four warps); the sampler no
# longer launches it.
@triton.jit(do_not_specialize=["topk"])
def _topk_log_softmax_kernel(
    output_ptr,
    logits_ptr,
    logits_stride,
    topk_ids_ptr,
    topk,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    TOPK_BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0).to(tl.int64)
    row_ptr = logits_ptr + req_idx * logits_stride

    max_val = float("-inf")
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(row_ptr + block, mask=block < vocab_size, other=float("-inf"))
        max_val = tl.max(tl.maximum(logits, max_val))
    max_val = max_val.to(tl.float32)  # type: ignore

    se = 0.0
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(row_ptr + block, mask=block < vocab_size, other=0.0)
        # NOTE(woosuk): Make sure that logits and all following operations use FP32.
        logits = logits.to(tl.float32)
        e = tl.exp(logits - max_val)
        e = tl.where(block < vocab_size, e, 0.0)
        se += tl.sum(e)
    lse = tl.log(se)

    for j in range(0, topk, TOPK_BLOCK_SIZE):
        k_offset = j + tl.arange(0, TOPK_BLOCK_SIZE)
        k_mask = k_offset < topk
        topk_ids = tl.load(
            topk_ids_ptr + req_idx * topk + k_offset, mask=k_mask, other=0
        )
        logits = tl.load(row_ptr + topk_ids, mask=k_mask)
        logits = logits.to(tl.float32)
        o = logits - max_val - lse
        tl.store(output_ptr + req_idx * topk + k_offset, o, mask=k_mask)


@triton.jit
def _ranks_kernel(
    output_ptr,
    logits_ptr,
    logits_stride,
    token_ids_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0).to(tl.int64)
    row_ptr = logits_ptr + req_idx * logits_stride

    token_id = tl.load(token_ids_ptr + req_idx)
    x = tl.load(row_ptr + token_id)

    n = 0
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(row_ptr + block, mask=block < vocab_size, other=float("-inf"))
        n += tl.sum((logits >= x).to(tl.int32))
    tl.store(output_ptr + req_idx, n)


def register_topk_scores_warmup(
    *, logits: Sequence[tuple[torch.dtype, int]], vocab_size: int, logits_mode: bool
) -> None:
    """Register ``compute_topk_scores`` for ``vocab_size``-wide logits given as
    ``(dtype, row stride)`` pairs: the top-k token ids (both paths' layouts), the
    selected tokens' ranks, and their log-softmax unless ``logits_mode``."""
    from vllm.config import get_current_vllm_config_or_none
    from vllm.v1.worker.gpu.sample.tilelang_logprob import (
        register_log_softmax,
        register_ranks,
    )

    # The top-k ids kernel takes k up to TOPK_IDS_MAX: a larger --max-logprobs
    # (or -1, the whole vocabulary) is refused here rather than at its request.
    config = get_current_vllm_config_or_none()
    model_config = None if config is None else config.model_config
    if model_config is not None and not (
        0 <= model_config.max_logprobs <= tilelang_topk_ids.TOPK_IDS_MAX
    ):
        raise ValueError(
            f"--max-logprobs {model_config.max_logprobs} is outside the TileLang "
            f"top-k logprobs' range 0..{tilelang_topk_ids.TOPK_IDS_MAX}"
        )
    tilelang_topk_ids.register_topk_ids(logits)
    register_ranks(logits, vocab_size)
    if not logits_mode:
        register_log_softmax(logits)


def compute_token_logprobs(
    logits: torch.Tensor, token_ids: torch.Tensor, valid: torch.Tensor | None = None
) -> torch.Tensor:
    # NOTE(woosuk): To save GPU memory, we do not materialize the full
    # [batch_size, vocab_size] logprobs tensor. The kernel computes
    # max + logsumexp per row and only emits logprobs at `token_ids`.
    from vllm.v1.worker.gpu.sample.tilelang_logprob import topk_log_softmax

    batch_size = logits.shape[0]
    token_ids = token_ids.to(torch.int64).contiguous()
    num_logprobs = token_ids.shape[1]
    logprobs = logits.new_empty((batch_size, num_logprobs), dtype=torch.float32)
    # The requested count is runtime data, including the single selected token
    # for prompt_logprobs=0: one kernel serves every count. With ``valid``, its
    # false columns are -inf.
    topk_log_softmax(logits, token_ids, logprobs, valid)
    return logprobs


def compute_topk_scores(
    logits: torch.Tensor,
    num_logprobs: int,
    sampled_token_ids: torch.Tensor,
    cu_num_logits: list[int] | torch.Tensor | None = None,
    logprob_token_ids_state: "LogprobTokenIdsState | None" = None,
    expanded_idx_mapping: torch.Tensor | None = None,
    max_per_req_token_ids: int = 0,
    logits_mode: bool = False,
) -> LogprobsTensors:
    if not 0 <= num_logprobs <= tilelang_topk_ids.TOPK_IDS_MAX:
        raise ValueError(f"{num_logprobs} logprobs a token: --max-logprobs bounds them")
    batch_size, vocab_size = logits.shape

    if max_per_req_token_ids == 0:
        # Fast path: no request asked for custom logprob_token_ids.
        if num_logprobs > 0:
            # [sampled, top-k...] in torch.topk's order, one launch.
            logprob_token_ids = tilelang_topk_ids.logprob_token_ids(
                logits, num_logprobs, sampled_token_ids
            )
        else:
            logprob_token_ids = sampled_token_ids.unsqueeze(-1)
        if logits_mode:
            scores = logits.gather(-1, logprob_token_ids).to(torch.float32)
        else:
            scores = compute_token_logprobs(logits, logprob_token_ids)
    else:
        # Some requests specified logprob_token_ids. Build the [batch_size,
        # 1 + max_cols] token_ids matrix and validity mask on the GPU via a
        # single kernel, overriding the topk columns with per-request
        # tokens where applicable.
        assert logprob_token_ids_state is not None
        assert expanded_idx_mapping is not None

        if num_logprobs > 0:
            topk_token_ids = tilelang_topk_ids.topk_token_ids(logits, num_logprobs)
        else:
            # This tensor just used as an int32 pointer, data not accessed.
            topk_token_ids = logprob_token_ids_state.token_ids.gpu

        num_cols = max(num_logprobs, max_per_req_token_ids)
        # The fill writes every column (0 and invalid where unused).
        logprob_token_ids = sampled_token_ids.new_empty((batch_size, 1 + num_cols))
        valid_mask = torch.empty_like(logprob_token_ids, dtype=torch.bool)
        tilelang_auxiliary.fill_token_ids(
            logprob_token_ids,
            valid_mask,
            sampled_token_ids,
            topk_token_ids,
            expanded_idx_mapping,
            logprob_token_ids_state.num_token_ids.gpu,
            logprob_token_ids_state.token_ids.gpu,
            num_logprobs,
        )
        if logits_mode:
            scores = logits.gather(-1, logprob_token_ids).to(torch.float32)
            scores = scores.masked_fill(~valid_mask, float("-inf"))
        else:
            scores = compute_token_logprobs(logits, logprob_token_ids, valid_mask)

    token_ranks = torch.empty(batch_size, dtype=torch.int64, device=logits.device)
    from vllm.v1.worker.gpu.sample.tilelang_logprob import token_ranks as fill_ranks

    fill_ranks(logits, sampled_token_ids, token_ranks)
    is_tensor = isinstance(cu_num_logits, torch.Tensor)
    return LogprobsTensors(
        logprob_token_ids=logprob_token_ids,
        logprobs=scores,
        selected_token_ranks=token_ranks,
        cu_num_generated_tokens=None if is_tensor else cu_num_logits,
        cu_num_generated_tokens_tensor=cu_num_logits if is_tensor else None,
    )


# The reference the TileLang fill (``tilelang_auxiliary.fill_ids_kernel``) is tested
# against; the sampler no longer launches it.
@triton.jit
def _fill_logprob_token_ids_kernel(
    # [batch_size, 1 + num_cols]
    out_token_ids_ptr,
    out_token_ids_stride,
    # [batch_size, 1 + num_cols]
    out_valid_mask_ptr,
    out_valid_mask_stride,
    sampled_token_ids_ptr,  # [batch_size]
    topk_indices_ptr,  # [batch_size, NUM_TOPK] (unused when NUM_TOPK == 0)
    topk_indices_stride,
    expanded_idx_mapping_ptr,  # [batch_size] -> req_state_idx
    num_per_req_token_ids_ptr,  # [max_num_reqs]
    per_req_token_ids_ptr,  # [max_num_reqs, MAX_LOGPROB_TOKEN_IDS]
    per_req_token_ids_stride,
    NUM_TOPK: tl.constexpr,
    PADDED_COLS: tl.constexpr,
):
    batch_idx = tl.program_id(0)

    # Column 0: always the sampled token, always valid.
    sampled = tl.load(sampled_token_ids_ptr + batch_idx)
    tl.store(out_token_ids_ptr + batch_idx * out_token_ids_stride, sampled)
    tl.store(out_valid_mask_ptr + batch_idx * out_valid_mask_stride, 1)

    req_state_idx = tl.load(expanded_idx_mapping_ptr + batch_idx)
    num_custom = tl.load(num_per_req_token_ids_ptr + req_state_idx)

    col = tl.arange(0, PADDED_COLS)
    tid_base = out_token_ids_ptr + batch_idx * out_token_ids_stride + 1
    mask_base = out_valid_mask_ptr + batch_idx * out_valid_mask_stride + 1

    if num_custom > 0:
        # Override topk with per-request custom tokens.
        src = per_req_token_ids_ptr + req_state_idx * per_req_token_ids_stride
        valid = col < num_custom
    else:
        # Fill with topk indices (no-op when NUM_TOPK == 0).
        src = topk_indices_ptr + batch_idx * topk_indices_stride
        valid = col < NUM_TOPK

    tokens = tl.load(src + col, mask=valid, other=0).to(tl.int64)
    tl.store(tid_base + col, tokens, mask=valid)
    tl.store(mask_base + col, tl.full([PADDED_COLS], 1, tl.int1), mask=valid)


class LogprobTokenIdsState:
    """Per-request override of which token ids' logprobs to return.

    See `SamplingParams.logprob_token_ids`.
    """

    def __init__(self, max_num_reqs: int, device: torch.device):
        self.max_num_reqs = max_num_reqs
        self.num_token_ids = UvaBackedTensor(max_num_reqs, dtype=torch.int32)
        self.token_ids = StagedWriteTensor(
            (max_num_reqs, MAX_LOGPROB_TOKEN_IDS),
            dtype=torch.int32,
            device=device,
        )

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> None:
        token_ids = sampling_params.logprob_token_ids
        if not token_ids:
            self.num_token_ids.np[req_idx] = 0
            return
        n = len(token_ids)
        if n > MAX_LOGPROB_TOKEN_IDS:
            raise ValueError(
                f"Too many logprob_token_ids: {n}. The max is {MAX_LOGPROB_TOKEN_IDS}."
            )
        self.num_token_ids.np[req_idx] = n
        self.token_ids.stage_write(req_idx, 0, token_ids)

    def apply_staged_writes(self) -> None:
        self.num_token_ids.copy_to_uva()
        self.token_ids.apply_write()

    def register_warmups(self) -> None:
        self.token_ids.register_warmup()
        tilelang_auxiliary.register_fill_ids()

    def max_num_token_ids(self, idx_mapping_np: np.ndarray) -> int:
        return int(self.num_token_ids.np[idx_mapping_np].max(initial=0))
