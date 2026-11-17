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
from vllm.v1.sample.sink import NUM_SINK_EVENT_FIELDS, SinkConfig, SinkRow

logger = init_logger(__name__)

# Vocabulary block of the candidates kernel; each block contributes
# max_retries + 1 candidates to the merge.
_CANDIDATE_BLOCK_SIZE = 1024


@dataclass
class SinkCut:
    """The logits the cut overwrote, for the restore."""

    logits: torch.Tensor
    saved: torch.Tensor
    cols: torch.Tensor
    flags: torch.Tensor
    drafts: torch.Tensor
    cu_num_logits: torch.Tensor

    def restore(self) -> None:
        from vllm.v1.worker.gpu.sample import tilelang_sampling

        tilelang_sampling.sink_restore(
            self.logits,
            self.saved,
            self.cols,
            self.flags,
            self.drafts,
            self.cu_num_logits,
        )


@dataclass
class SinkAlternates:
    """The next candidates of each request's last draw: [num_reqs, max_retries]
    tokens (-1 past the possible ones), with logprobs and ranks when the batch
    returns logprobs (``[num_reqs, max_retries]`` each, the ranks possibly a view
    of alternate-major storage)."""

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
        from vllm.v1.worker.gpu.sample import tilelang_sampling

        flags, self.row_values = tilelang_sampling.sink_flags(
            columns,
            logits_indices,
            cu_num_logits,
            num_reqs,
            dead_cut,
            self.config.threshold,
        )
        return flags

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
        from vllm.v1.worker.gpu.sample import tilelang_sampling

        num_cols = logits.shape[1]
        # Row r + 1 holds a draft of row r's request unless it starts one.
        saved, cols = tilelang_sampling.sink_cut(
            logits,
            flags,
            drafts,
            cu_num_logits,
            num_reqs,
            vocab_start,
            num_cols if vocab_cols is None else vocab_cols,
        )
        return SinkCut(logits, saved, cols, flags, drafts, cu_num_logits)

    def cut_cols(
        self,
        flags: torch.Tensor,
        drafts: torch.Tensor,
        cu_num_logits: torch.Tensor,
        num_reqs: int,
        vocab_start: int = 0,
        vocab_cols: int | None = None,
    ) -> torch.Tensor:
        """Per-row local column index masked to ``-inf`` on this vocabulary
        shard (``-1`` where no draft is cut on this shard)."""
        from vllm.v1.worker.gpu.sample import tilelang_sampling

        limit = self.vocab_size if vocab_cols is None else vocab_cols
        return tilelang_sampling.sink_cut_cols(
            flags,
            drafts,
            cu_num_logits,
            num_reqs,
            vocab_start,
            limit,
        )

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
        dead_cut: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The next ``max_retries`` candidates of each request's last draw
        from the full-vocabulary scores it was drawn from (after sampling
        parameters, with the cut applied). ``dead_cut``: verification ended each
        request's block at its dead-row cut, so a draw at the cut was the bonus,
        which masked no draft."""
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
            dead_cut,
        )
        return tilelang_sampling.sink_merge(
            values, indices, sampled, num_sampled, self.num_alternates
        )

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
        from vllm.v1.worker.gpu.sample import tilelang_sampling

        packed = tilelang_sampling.sink_shard_top(
            local_logits,
            cu_num_logits,
            num_sampled,
            vocab_start,
            vocab_cols,
            self.num_candidates,
        )
        if tp_size > 1:
            packed = tensor_model_parallel_all_gather(packed, dim=-1)
        return tilelang_sampling.sink_merge(
            packed,
            None,
            sampled,
            num_sampled,
            self.num_alternates,
            self.num_candidates,
        )

    def alternates_of_greedy_top_candidates(
        self,
        row_topk: torch.Tensor,
        cu_num_logits: torch.Tensor,
        num_sampled: torch.Tensor,
        sampled: torch.Tensor,
        tp_size: int,
    ) -> torch.Tensor:
        """``alternates_of_greedy_shards`` from per-row top-``num_candidates``
        shard projections (``[num_logits, 2 * num_candidates]`` FP32)."""
        from vllm.v1.worker.gpu.sample import tilelang_sampling

        packed = tilelang_sampling.sink_select_top_rows(
            row_topk,
            cu_num_logits,
            num_sampled,
            self.num_candidates,
        )
        if tp_size > 1:
            packed = tensor_model_parallel_all_gather(packed, dim=-1)
        return tilelang_sampling.sink_merge(
            packed,
            None,
            sampled,
            num_sampled,
            self.num_alternates,
            self.num_candidates,
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
        row that drew it, as the sampled token's are computed for logprobs
        (``compute_topk_scores`` of each alternate, the bits it gives): one launch
        copies the rows densely and clamps the alternates (``sink_rows``), one
        log-softmax launch scores every alternate of a row (with ``logits_mode``,
        the copy takes their logits instead), and the rank kernel ranks one
        alternate a launch, into its own row of alternate-major ranks."""
        from vllm.v1.worker.gpu.sample import tilelang_sampling
        from vllm.v1.worker.gpu.sample.tilelang_logprob import (
            token_ranks,
            topk_log_softmax,
        )

        num_reqs = alternates.shape[0]
        num_alternates = self.num_alternates
        if alternates.shape[1] != num_alternates:
            raise ValueError(
                f"{alternates.shape[1]} alternates a request, not {num_alternates}"
            )
        logprobs = torch.empty(
            (num_reqs, num_alternates), dtype=torch.float32, device=self.device
        )
        ranks = torch.empty(
            (num_alternates, num_reqs), dtype=torch.int64, device=self.device
        )
        if num_reqs and num_alternates:
            rows, tokens, tokens_by_alt = tilelang_sampling.sink_rows(
                logits, cu_num_logits, num_sampled, alternates, logprobs, logits_mode
            )
            if not logits_mode:
                topk_log_softmax(rows, tokens, logprobs)
            for i in range(num_alternates):
                token_ranks(rows, tokens_by_alt[i], ranks[i])
        return SinkAlternates(alternates, logprobs, ranks.t())

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
                # The policy writes every request's: 0 where nothing was recovered.
                out.recovered_logprobs = torch.empty(
                    num_reqs, dtype=torch.float32, device=self.device
                )
                out.recovered_ranks = torch.empty(
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
        vocab_size: int,
        logits_mode: bool = False,
    ) -> None:
        """Declare the kernels a step launches: the policy for each sampled
        width, and the alternates of the raw and processed target logits
        (``(dtype, row stride)``; the TileLang kernels take any stride), with
        their logprobs as ``logits_mode`` (the logprobs mode is a logits mode)
        computes them."""
        from vllm.v1.worker.gpu.sample import tilelang_sampling
        from vllm.v1.worker.gpu.sample.tilelang_logprob import (
            register_log_softmax,
            register_ranks,
        )

        logits_dtypes = [dtype for dtype, _ in logits]
        logits_strides = list(logits)
        if processed_logits_strides:
            logits_dtypes.append(torch.float32)
            logits_strides.extend(
                (torch.float32, stride) for stride in processed_logits_strides
            )
        # alternate_logprobs_of ranks and scores the alternates in dense copies of
        # their rows, vocab_size apart whatever the logits' (padded) row stride.
        dense = [(dtype, vocab_size) for dtype in dict.fromkeys(logits_dtypes)]
        register_ranks(dense, vocab_size)
        if self.rejection and not logits_mode:
            register_log_softmax(dense)
        tilelang_sampling.declare(
            lambda: tilelang_sampling.sink_calls(
                logits_dtypes=logits_dtypes,
                sampled_widths=sampled_widths,
                num_alternates=self.num_alternates,
                rejection=self.rejection,
                num_event_fields=NUM_SINK_EVENT_FIELDS,
                logits_mode=logits_mode,
                logits_strides=logits_strides,
            )
        )


_TRITON_REFS = frozenset(("_sink_candidates_kernel", "_sink_policy_kernel"))


def __getattr__(name: str):
    if name in _TRITON_REFS:
        from vllm.v1.worker.gpu import _triton_refs

        return getattr(_triton_refs, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
