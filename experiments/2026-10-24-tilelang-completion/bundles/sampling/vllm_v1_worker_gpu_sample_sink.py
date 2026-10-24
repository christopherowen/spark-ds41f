# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The attention-sink detector and draft-zero rejection on the V2 model runner
(the policy: vllm.v1.sample.sink).

Each step the detector flags the sampled rows whose watched residual column is
sunk. With rejection on, a flagged row 0 rejects draft zero and takes the next
alternate of the draw that produced it: the same scores (the logits, plus the
same Gumbel noise for random rows) in descending order after the sampled token,
i.e. sampling without replacement. A flagged drafter row cuts the verification:
its draft is masked in the row before, and restored after the verification.
Everything stays on the device; each rank counts the rows it flagged and logs
the counts without a synchronization. That count covers every step the rank
executes, including steps whose output the scheduler then discards (the step in
flight when a request finishes or is aborted, stale output after a preemption);
``vllm:sink_*`` counts the flags in committed outputs, so the log reads at least
the metric.
"""

from dataclasses import dataclass

import torch

from vllm.distributed import tensor_model_parallel_all_gather
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.sample.sink import NUM_SINK_EVENT_FIELDS, SinkConfig, SinkRow
from vllm.v1.worker.gpu.sample.gumbel import gumbel_noised_scores

logger = init_logger(__name__)

# Vocabulary block of the candidates kernel; each block contributes
# max_retries + 1 candidates to the merge.
_CANDIDATE_BLOCK_SIZE = 1024


# Reference for ``tilelang_sampling``; the runner no longer launches it.
@triton.jit
def _sink_candidates_kernel(
    # [num_reqs, num_blocks * NUM_CANDIDATES]
    values_ptr,
    indices_ptr,
    candidates_stride,
    # [num_logits, logits_stride]: the scores' logits, after sampling params
    logits_ptr,
    logits_stride,
    vocab_size,
    # [num_reqs + 1]
    cu_num_logits_ptr,
    # [num_reqs]
    num_sampled_ptr,
    # [num_reqs, sampled_stride]
    sampled_ptr,
    sampled_stride,
    # [num_logits]: each row's input token
    draft_ptr,
    # [num_reqs]
    idx_mapping_ptr,
    # [max_num_reqs]
    temp_ptr,
    seeds_ptr,
    # [num_logits]
    pos_ptr,
    BLOCK_SIZE: tl.constexpr,
    NUM_CANDIDATES: tl.constexpr,
    USE_FP64: tl.constexpr,
    MASK_REJECTED_DRAFT: tl.constexpr,
):
    req = tl.program_id(0)
    block_idx = tl.program_id(1)
    start = tl.load(cu_num_logits_ptr + req)
    end = tl.load(cu_num_logits_ptr + req + 1)
    local = tl.maximum(tl.load(num_sampled_ptr + req) - 1, 0)
    # The row that produced the request's last sampled token.
    row = (start + local).to(tl.int64)
    req_state_idx = tl.load(idx_mapping_ptr + req).to(tl.int64)
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)

    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    scores = tl.load(
        logits_ptr + row * logits_stride + block, mask=mask, other=float("-inf")
    ).to(tl.float32)
    if MASK_REJECTED_DRAFT:
        # A random row that rejected its draft resampled the residual, which
        # for one-hot drafts is the target without the draft.
        is_bonus = row == end - 1
        verified = tl.load(draft_ptr + row + 1, mask=not is_bonus, other=-1)
        emitted = tl.load(sampled_ptr + req * sampled_stride + local)
        rejected = (
            (not is_bonus) & (verified >= 0) & (verified != emitted) & (temp != 0.0)
        )
        scores = tl.where(rejected & (block == verified), float("-inf"), scores)
    seed = tl.load(seeds_ptr + req_state_idx)
    pos = tl.load(pos_ptr + row)
    scores = gumbel_noised_scores(
        scores,
        block,
        mask,
        seed,
        pos,
        temp,
        IS_DRAFTING=False,
        USE_FP64=USE_FP64,
        APPLY_TEMPERATURE=False,
    )
    offsets = tl.arange(0, BLOCK_SIZE)
    out = req * candidates_stride + block_idx * NUM_CANDIDATES
    for i in tl.static_range(NUM_CANDIDATES):
        value, idx = tl.max(scores, axis=0, return_indices=True)
        tl.store(values_ptr + out + i, value)
        tl.store(indices_ptr + out + i, block_idx * BLOCK_SIZE + idx)
        scores = tl.where(offsets == idx, float("-inf"), scores)


# Reference for ``tilelang_sampling``; the runner no longer launches it.
@triton.jit
def _sink_policy_kernel(
    # Verification results, updated in place for rejected draft zeros.
    # [num_reqs, SAMPLED_WIDTH] int64, [num_reqs] int32, [num_reqs] int32
    sampled_ptr,
    num_sampled_ptr,
    num_rejected_ptr,
    # Host outputs: [num_reqs, SAMPLED_WIDTH + 1] int64, [num_reqs] int32,
    # [num_reqs, NUM_EVENT_FIELDS] int32, [num_reqs] float32 / int64
    out_tokens_ptr,
    num_out_ptr,
    events_ptr,
    recovered_logprobs_ptr,
    recovered_ranks_ptr,
    # Batch.
    cu_num_logits_ptr,
    idx_mapping_ptr,
    # [num_reqs] int8: the request was decoding (row 0's input is draft zero)
    decoding_ptr,
    # [num_logits] int8, [num_logits] float32 (each row's largest |column|),
    # [num_logits]
    flags_ptr,
    values_ptr,
    draft_ptr,
    # This step's alternates: [num_reqs, NUM_ALTERNATES]
    new_alternates_ptr,
    new_alternate_logprobs_ptr,
    new_alternate_ranks_ptr,
    # Per-slot state: [max_num_reqs, NUM_ALTERNATES], [max_num_reqs]
    alternates_ptr,
    alternate_logprobs_ptr,
    alternate_ranks_ptr,
    tries_ptr,
    # [max_num_reqs] int32, [max_num_reqs, vocab] int32
    total_len_ptr,
    bin_counts_ptr,
    bin_counts_stride,
    # [3] int64: this rank's flagged rows per SinkRow
    counts_ptr,
    NUM_ALTERNATES: tl.constexpr,
    REJECTION: tl.constexpr,
    LOGPROBS: tl.constexpr,
    SAMPLED_WIDTH: tl.constexpr,
    PADDED_WIDTH: tl.constexpr,
    PADDED_ALTERNATES: tl.constexpr,
    NUM_EVENT_FIELDS: tl.constexpr,
):
    req = tl.program_id(0)
    start = tl.load(cu_num_logits_ptr + req)
    num_logits = tl.load(cu_num_logits_ptr + req + 1) - start
    num_sampled = tl.load(num_sampled_ptr + req)
    events = events_ptr + req * NUM_EVENT_FIELDS
    if num_sampled == 0:
        # A prefill chunk: its row is not sampled.
        tl.store(num_out_ptr + req, 0)
        tl.store(events + 0, 0)
        tl.store(events + 1, 2)
        tl.store(events + 2, 0)
        tl.store(events + 3, -1)
        tl.store(events + 4, 0)
        tl.store(events + 5, -1)
        return

    req_state_idx = tl.load(idx_mapping_ptr + req).to(tl.int64)
    decoding = tl.load(decoding_ptr + req) != 0
    row0_flag = tl.load(flags_ptr + start) != 0
    rows = tl.arange(0, PADDED_WIDTH)
    row_flags = tl.load(flags_ptr + start + rows, mask=rows < num_logits, other=0) != 0
    # Drafter rows whose inputs were accepted (1 .. num_sampled - 1), and the
    # row of the first rejected draft.
    accepted_flags = tl.sum(
        (row_flags & (rows >= 1) & (rows < num_sampled)).to(tl.int32), axis=0
    )
    cut = (tl.sum((row_flags & (rows == num_sampled)).to(tl.int32), axis=0) > 0) & (
        num_sampled < num_logits
    )
    draft_zero_flag = decoding & row0_flag
    # Prompt row flagged: FLAGGED; it cannot be replaced.
    outcome = tl.where((not decoding) & row0_flag, 1, 0)
    out_row = out_tokens_ptr + req * (SAMPLED_WIDTH + 1)
    sampled_row = sampled_ptr + req * SAMPLED_WIDTH
    if REJECTION:
        tries = tl.load(tries_ptr + req_state_idx)
        slot = tl.minimum(tries, NUM_ALTERNATES - 1)
        state = req_state_idx * NUM_ALTERNATES
        alternate = tl.load(alternates_ptr + state + slot)
        rejected = draft_zero_flag & (tries < NUM_ALTERNATES) & (alternate >= 0)
        outcome = tl.where(rejected, 2, tl.where(draft_zero_flag, 3, outcome))
        drafter_flags = tl.where(rejected, 0, accepted_flags + cut.to(tl.int32))
        cut_row = tl.where(cut & (not rejected), num_sampled, -1)
        if rejected:
            # Draft zero is replaced by its next alternate; the request's
            # position does not advance (all of its rows are rejected).
            draft_zero = tl.load(draft_ptr + start).to(tl.int64)
            tl.store(sampled_row, alternate)
            tl.store(num_sampled_ptr + req, 1)
            tl.store(num_rejected_ptr + req, num_logits)
            tl.store(tries_ptr + req_state_idx, tries + 1)
            if LOGPROBS:
                tl.store(
                    recovered_logprobs_ptr + req,
                    tl.load(alternate_logprobs_ptr + state + slot),
                )
                tl.store(
                    recovered_ranks_ptr + req,
                    tl.load(alternate_ranks_ptr + state + slot),
                )
            # The request state already holds draft zero as its last token:
            # take it out, so the update appends the alternate in its place.
            tl.store(
                total_len_ptr + req_state_idx,
                tl.load(total_len_ptr + req_state_idx) - 1,
            )
            count_ptr = bin_counts_ptr + req_state_idx * bin_counts_stride + draft_zero
            tl.store(count_ptr, tl.load(count_ptr) - 1)
            tl.store(out_row, alternate)
            tl.store(num_out_ptr + req, 1)
        else:
            # The last sampled token's draw provides the next alternates.
            a = tl.arange(0, PADDED_ALTERNATES)
            in_range = a < NUM_ALTERNATES
            new = req * NUM_ALTERNATES + a
            tl.store(
                alternates_ptr + state + a,
                tl.load(new_alternates_ptr + new, mask=in_range),
                mask=in_range,
            )
            if LOGPROBS:
                tl.store(
                    alternate_logprobs_ptr + state + a,
                    tl.load(new_alternate_logprobs_ptr + new, mask=in_range),
                    mask=in_range,
                )
                tl.store(
                    alternate_ranks_ptr + state + a,
                    tl.load(new_alternate_ranks_ptr + new, mask=in_range),
                    mask=in_range,
                )
            tl.store(tries_ptr + req_state_idx, 0)
            # Draft zero first, then the step's sampled tokens.
            shift = decoding.to(tl.int32)
            tokens = tl.load(sampled_row + rows, mask=rows < num_sampled)
            tl.store(out_row + shift + rows, tokens, mask=rows < num_sampled)
            if decoding:
                tl.store(out_row, tl.load(draft_ptr + start).to(tl.int64))
            tl.store(num_out_ptr + req, num_sampled + shift)
    else:
        outcome = tl.where(draft_zero_flag, 1, outcome)
        drafter_flags = accepted_flags
        cut_row = -1
    # The largest flagged value among the rows the step judged (up to the
    # first rejected draft) and its row, for the log line.
    judged = row_flags & (rows <= num_sampled)
    row_values = tl.load(values_ptr + start + rows, mask=rows < num_logits, other=0.0)
    value, value_row = tl.max(
        tl.where(judged, row_values, -1.0), axis=0, return_indices=True
    )
    flagged_any = value >= 0.0
    tl.store(events + 0, outcome)
    tl.store(events + 1, tl.where(decoding, 0, 2))
    tl.store(events + 2, drafter_flags)
    tl.store(events + 3, cut_row)
    tl.store(events + 4, tl.where(flagged_any, value, 0.0).to(tl.int32))
    tl.store(events + 5, tl.where(flagged_any, value_row, -1))
    tl.atomic_add(counts_ptr + 0, draft_zero_flag.to(tl.int64))
    tl.atomic_add(counts_ptr + 1, drafter_flags.to(tl.int64))
    tl.atomic_add(counts_ptr + 2, ((not decoding) & row0_flag).to(tl.int64))


@dataclass
class SinkCut:
    """The logits the cut overwrote, for the restore."""

    logits: torch.Tensor
    saved: torch.Tensor
    cols: torch.Tensor

    def restore(self) -> None:
        self.logits.scatter_(1, self.cols, self.saved)


@dataclass
class SinkAlternates:
    """The next candidates of each request's last draw: [num_reqs, max_retries]
    tokens (-1 past the possible ones), with logprobs and ranks when the batch
    returns logprobs."""

    tokens: torch.Tensor
    logprobs: torch.Tensor | None = None
    ranks: torch.Tensor | None = None


@dataclass
class SinkStepOutput:
    """The policy's results for the host. ``tokens`` / ``num_tokens`` replace
    the sampled tokens with rejection on: draft zero first for a decoding
    request, or the alternate that replaced it."""

    # [num_reqs, NUM_SINK_EVENT_FIELDS] int32
    events: torch.Tensor
    tokens: torch.Tensor | None = None
    num_tokens: torch.Tensor | None = None
    recovered_logprobs: torch.Tensor | None = None
    recovered_ranks: torch.Tensor | None = None


class SinkSampler:
    """The detector's flags, the drafter-row cut, the alternates of each draw
    and the draft-zero policy, with their per-slot state."""

    def __init__(
        self,
        config: SinkConfig,
        max_num_reqs: int,
        hidden_size: int,
        vocab_size: int,
        num_speculative_steps: int,
        device: torch.device,
        use_fp64_gumbel: bool,
    ):
        if use_fp64_gumbel:
            raise ValueError(
                "use_fp64_gumbel: the V2 runner's TileLang sampler draws in FP32 only"
            )
        if config.column >= hidden_size:
            raise ValueError(
                f"sink detector column {config.column} is outside the hidden "
                f"size {hidden_size}"
            )
        self.config = config
        self.rejection = config.rejection
        self.max_num_reqs = max_num_reqs
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_speculative_steps = num_speculative_steps
        self.device = device
        self.use_fp64_gumbel = use_fp64_gumbel
        self.num_alternates = config.num_alternates
        # Candidates per block (or per rank): the sampled token and the
        # alternates after it.
        self.num_candidates = self.num_alternates + 1
        num_state = max(self.num_alternates, 1)
        self.alternates = torch.full(
            (max_num_reqs, num_state), -1, dtype=torch.int64, device=device
        )
        self.alternate_logprobs = torch.zeros(
            (max_num_reqs, num_state), dtype=torch.float32, device=device
        )
        self.alternate_ranks = torch.zeros(
            (max_num_reqs, num_state), dtype=torch.int64, device=device
        )
        self.tries = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        # Rows this rank flagged since startup, per SinkRow.
        self.counts = torch.zeros(len(SinkRow), dtype=torch.int64, device=device)
        self._counts_host = torch.zeros(
            len(SinkRow), dtype=torch.int64, pin_memory=True
        )
        self._counts_event: torch.cuda.Event | None = None
        self._logged_counts = (0,) * len(SinkRow)
        # Set during the startup warmup (end_warmup clears it).
        self.warming = False
        # Each logits row's largest |column| of the last flags() call.
        self.row_values: torch.Tensor | None = None

    def flags(
        self,
        columns: torch.Tensor,
        logits_indices: torch.Tensor,
        cu_num_logits: torch.Tensor,
        num_reqs: int,
        dead_cut: torch.Tensor | None,
    ) -> torch.Tensor:
        """One int8 flag per logits row: some residual stream has
        ``|residual[token, column]| > threshold``. ``columns`` is the model's
        watched column of every token's pre-norm residual streams,
        ``[tokens, streams]`` float32. Rows past a request's live drafts
        (``dead_cut``) are never flagged: they skipped routed MoE."""
        rows = columns[logits_indices].abs()
        self.row_values = rows.amax(dim=1).nan_to_num(0.0)
        flags = (rows > self.config.threshold).any(dim=1)
        if dead_cut is not None:
            rows = torch.arange(
                flags.shape[0], dtype=cu_num_logits.dtype, device=self.device
            )
            req = torch.searchsorted(cu_num_logits[1 : num_reqs + 1], rows, right=True)
            flags &= rows - cu_num_logits[req] <= dead_cut[req]
        return flags.to(torch.int8)

    def cut(
        self,
        logits: torch.Tensor,
        flags: torch.Tensor,
        drafts: torch.Tensor,
        cu_num_logits: torch.Tensor,
        num_reqs: int,
        vocab_start: int = 0,
        vocab_cols: int | None = None,
    ) -> SinkCut:
        """Masks, in place, the draft of every flagged drafter row in the row
        before it, which verifies it. ``logits`` may be this rank's vocabulary
        shard (columns from ``vocab_start``); a draft outside it is left
        alone."""
        num_logits, num_cols = logits.shape
        # Row r + 1 holds a draft of row r's request unless it starts one.
        starts = torch.zeros(num_logits + 1, dtype=torch.bool, device=self.device)
        starts[cu_num_logits[:num_reqs].long()] = True
        cols = drafts[1:num_logits].long() - vocab_start
        cut = flags[1:num_logits].bool() & ~starts[1:num_logits] & (cols >= 0)
        cut &= cols < (num_cols if vocab_cols is None else vocab_cols)
        cols = cols.where(cut, 0)[:, None]
        rows = logits[: num_logits - 1]
        saved = rows.gather(1, cols)
        rows.scatter_(1, cols, saved.masked_fill(cut[:, None], float("-inf")))
        return SinkCut(rows, saved, cols)

    def alternates_of(
        self,
        logits: torch.Tensor,
        cu_num_logits: torch.Tensor,
        num_sampled: torch.Tensor,
        sampled: torch.Tensor,
        drafts: torch.Tensor,
        idx_mapping: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        pos: torch.Tensor,
        mask_rejected_draft: bool,
    ) -> torch.Tensor:
        """The next ``max_retries`` candidates of each request's last draw
        from the full-vocabulary scores it was drawn from (after sampling
        parameters, with the cut applied)."""
        from vllm.v1.worker.gpu.sample import tilelang_sampling

        values, indices = tilelang_sampling.sink_candidates(
            logits,
            min(logits.shape[1], self.vocab_size),
            cu_num_logits,
            num_sampled,
            sampled,
            drafts,
            idx_mapping,
            temperature,
            seeds,
            pos,
            self.num_candidates,
            mask_rejected_draft,
        )
        return self._merge(values, indices, num_sampled, sampled)

    def alternates_of_greedy_shards(
        self,
        local_logits: torch.Tensor,
        cu_num_logits: torch.Tensor,
        num_sampled: torch.Tensor,
        sampled: torch.Tensor,
        vocab_start: int,
        vocab_cols: int,
        tp_size: int,
    ) -> torch.Tensor:
        """``alternates_of`` for an all-greedy step verified from vocabulary
        shards: each rank's top candidates of the last sampled rows, gathered
        (as ``(value, index)`` floats, exact for any vocabulary index) and
        merged identically on every rank."""
        num_reqs = num_sampled.shape[0]
        rows = cu_num_logits[:num_reqs].to(torch.int64) + (
            num_sampled.to(torch.int64) - 1
        ).clamp(min=0)
        local = local_logits[rows, :vocab_cols].float()
        values, indices = local.topk(self.num_candidates, dim=-1)
        values = values.masked_fill(values.isnan(), float("-inf"))
        packed = torch.cat([values, (indices + vocab_start).float()], dim=1)
        if tp_size > 1:
            packed = tensor_model_parallel_all_gather(packed, dim=-1)
        packed = packed.view(num_reqs, tp_size, 2, self.num_candidates)
        values = packed[:, :, 0].reshape(num_reqs, -1).contiguous()
        indices = packed[:, :, 1].reshape(num_reqs, -1).to(torch.int64)
        return self._merge(values, indices, num_sampled, sampled)

    def _merge(
        self,
        values: torch.Tensor,
        indices: torch.Tensor,
        num_sampled: torch.Tensor,
        sampled: torch.Tensor,
    ) -> torch.Tensor:
        """The best candidates after each request's last sampled token, ties
        to the lower token as in the samplers' argmax (candidates come in
        vocabulary order). The sampled token and NaN scores are no alternates;
        impossible tokens (-inf) become -1."""
        last = (num_sampled.long() - 1).clamp(min=0)[:, None]
        excluded = (indices == sampled.gather(1, last)) | values.isnan()
        values = values.masked_fill(excluded, float("-inf"))
        values, order = values.sort(dim=1, descending=True, stable=True)
        alternates = indices.gather(1, order[:, : self.num_alternates])
        return alternates.masked_fill(
            values[:, : self.num_alternates] == float("-inf"), -1
        )

    def alternate_logprobs_of(
        self,
        logits: torch.Tensor,
        cu_num_logits: torch.Tensor,
        num_sampled: torch.Tensor,
        alternates: torch.Tensor,
        logits_mode: bool,
    ) -> SinkAlternates:
        """The logprob and rank of each alternate in the distribution of the
        row that drew it, as the sampled token's are computed for logprobs."""
        from vllm.v1.worker.gpu.sample.logprob import compute_topk_scores

        num_reqs = alternates.shape[0]
        rows = cu_num_logits[:num_reqs].to(torch.int64) + (
            num_sampled.to(torch.int64) - 1
        ).clamp(min=0)
        row_logits = logits[rows]
        logprobs = torch.empty(
            (num_reqs, self.num_alternates), dtype=torch.float32, device=self.device
        )
        ranks = torch.empty(
            (num_reqs, self.num_alternates), dtype=torch.int64, device=self.device
        )
        for i in range(self.num_alternates):
            tokens = alternates[:, i].clamp(min=0)
            scores = compute_topk_scores(
                row_logits, 0, tokens, None, logits_mode=logits_mode
            )
            logprobs[:, i] = scores.logprobs[:, 0]
            ranks[:, i] = scores.selected_token_ranks
        return SinkAlternates(alternates, logprobs, ranks)

    def apply(
        self,
        sampled: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        cu_num_logits: torch.Tensor,
        idx_mapping: torch.Tensor,
        decoding: torch.Tensor,
        flags: torch.Tensor,
        drafts: torch.Tensor,
        alternates: SinkAlternates | None,
        total_len: torch.Tensor,
        bin_counts: torch.Tensor,
        values: torch.Tensor | None = None,
    ) -> SinkStepOutput:
        """The draft-zero policy, the alternates' bookkeeping and the step's
        events. With rejection on, updates ``sampled``, ``num_sampled`` and
        ``num_rejected`` in place for the requests whose draft zero it
        rejects, and takes draft zero out of their request state."""
        num_reqs, sampled_width = sampled.shape
        if values is None:
            values = self.row_values
        if values is None:
            values = torch.zeros(flags.shape, dtype=torch.float32, device=self.device)
        events = torch.empty(
            (num_reqs, NUM_SINK_EVENT_FIELDS), dtype=torch.int32, device=self.device
        )
        logprobs = alternates is not None and alternates.logprobs is not None
        out = SinkStepOutput(events)
        if self.rejection:
            assert alternates is not None
            out.tokens = torch.empty(
                (num_reqs, sampled_width + 1), dtype=torch.int64, device=self.device
            )
            out.num_tokens = torch.empty(
                num_reqs, dtype=torch.int32, device=self.device
            )
            if logprobs:
                out.recovered_logprobs = torch.zeros(
                    num_reqs, dtype=torch.float32, device=self.device
                )
                out.recovered_ranks = torch.zeros(
                    num_reqs, dtype=torch.int64, device=self.device
                )
        new_tokens = alternates.tokens if alternates is not None else self.alternates
        from vllm.v1.worker.gpu.sample import tilelang_sampling

        tilelang_sampling.sink_policy(
            sampled,
            num_sampled,
            num_rejected,
            out.tokens if out.tokens is not None else sampled,
            out.num_tokens if out.num_tokens is not None else num_sampled,
            events,
            out.recovered_logprobs if logprobs else self.alternate_logprobs,
            out.recovered_ranks if logprobs else self.alternate_ranks,
            cu_num_logits,
            idx_mapping,
            decoding,
            flags,
            values,
            drafts,
            new_tokens,
            alternates.logprobs if logprobs else self.alternate_logprobs,
            alternates.ranks if logprobs else self.alternate_ranks,
            self.alternates,
            self.alternate_logprobs,
            self.alternate_ranks,
            self.tries,
            total_len,
            bin_counts,
            self.counts,
            self.num_alternates,
            self.rejection,
            logprobs,
            NUM_SINK_EVENT_FIELDS,
        )
        return out

    def end_warmup(self) -> None:
        """Forget the rows the startup warmup flagged, and log from now on. Its
        synthetic inputs hold token 0, DeepSeek V4.1's BOS, the sink token
        itself, which the detector rightly flags."""
        self.warming = False
        self.counts.zero_()
        self._counts_event = None
        self._logged_counts = (0,) * len(SinkRow)

    def log_counts(self, rank: int) -> None:
        """Logs this rank's flagged-row counts when they changed, except during
        the startup warmup. The counts come from the previous step's copy, read
        only once it completed: no synchronization. Then queues this step's
        copy."""
        if self.warming:
            return
        event = self._counts_event
        if event is not None and event.query():
            counts = tuple(int(c) for c in self._counts_host.tolist())
            if counts != self._logged_counts:
                self._logged_counts = counts
                logger.info(
                    "Sink detector, rank %d: %d flagged rows in executed steps "
                    "(draft zero %d, drafter %d, prompt %d; vllm:sink_* counts "
                    "committed outputs only)",
                    rank,
                    sum(counts),
                    *counts,
                )
        if event is None:
            event = self._counts_event = torch.cuda.Event()
        elif not event.query():
            return
        self._counts_host.copy_(self.counts, non_blocking=True)
        event.record()

    def register_warmups(
        self,
        *,
        logits: tuple[tuple[torch.dtype, int], ...],
        processed_logits_strides: tuple[int, ...],
        sampled_widths: tuple[int, ...],
        bin_counts_stride: int,
    ) -> None:
        """Declare the kernels a step launches: the policy for each sampled
        width, and the alternates of the raw and processed target logits
        (``(dtype, row stride)``; the TileLang kernels take any stride)."""
        from vllm.v1.worker.gpu.sample import tilelang_sampling

        logits_dtypes = [dtype for dtype, _ in logits]
        if processed_logits_strides:
            logits_dtypes.append(torch.float32)
        tilelang_sampling.declare(
            lambda: tilelang_sampling.sink_calls(
                logits_dtypes=logits_dtypes,
                sampled_widths=sampled_widths,
                num_alternates=self.num_alternates,
                rejection=self.rejection,
                num_event_fields=NUM_SINK_EVENT_FIELDS,
            )
        )
