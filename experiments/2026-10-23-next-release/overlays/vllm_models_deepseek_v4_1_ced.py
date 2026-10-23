# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CED row packing and batch metadata; no model computation lives here."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import torch

from .tilelang import ced as ced_kernels
from .tilelang import prepare

if TYPE_CHECKING:
    from .sparse_mla import DeepseekV41B12xMetadata

CED_WINDOW = 128
# The kept-row limit of a request that keeps every row (prompt logprobs, capture).
KEEP_ALL = 2**31 - 1


def ced_decoder_start(hf_config: Any) -> int | None:
    """Find the full-resolution KV source following the compressed encoder."""
    ratios = getattr(hf_config, "compress_ratios", ())
    layers = getattr(hf_config, "num_hidden_layers", len(ratios))
    for source in sorted(getattr(hf_config, "kv_source_layer_ids", ())):
        if (
            0 < source < min(layers, len(ratios))
            and ratios[source] == 1
            and any(ratio > 1 for ratio in ratios[:source])
        ):
            return source
    return None


@torch.library.custom_op("vllm::dsv41_gather_rows", mutates_args=())
def gather_rows(source: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather first-axis rows, zeroing invalid indices (including graph padding)."""
    output = source.new_empty((indices.numel(), *source.shape[1:]))
    if output.numel():
        ced_kernels.copy_rows(source, indices, output, scatter=False)
    return output


@gather_rows.register_fake
def _gather_rows_fake(source, indices):
    return source.new_empty((indices.numel(), *source.shape[1:]))


@torch.library.custom_op("vllm::dsv41_scatter_rows", mutates_args=())
def scatter_rows(
    source: torch.Tensor, indices: torch.Tensor, full_rows: int
) -> torch.Tensor:
    """Restore the full-row ABI; discarded rows are always zero."""
    output = source.new_zeros((full_rows, *source.shape[1:]))
    if source.numel():
        ced_kernels.copy_rows(source, indices, output, scatter=True)
    return output


@scatter_rows.register_fake
def _scatter_rows_fake(source, indices, full_rows):
    return source.new_empty((full_rows, *source.shape[1:]))


@dataclass(frozen=True)
class CEDPlan:
    """CPU scheduling bounds; GPU starts remain authoritative for actual rows."""

    capacity: int
    num_tokens: int
    max_query_len: int
    keep_full: tuple[bool, ...]


def plan_ced(
    query_lengths, keep_full, capacities: tuple[int, ...], limits=None
) -> CEDPlan | None:
    """The CPU bounds of a step's decoder rows: each request's last ``CED_WINDOW``
    rows, or every row where ``keep_full``. ``limits`` (``nonfinal_limits``) lowers a
    request's kept rows further; ``max_query_len`` keeps the window bound, so the
    decoder runs in the mode it would without the limits."""
    bounds = tuple(
        int(q) if full else min(int(q), CED_WINDOW)
        for q, full in zip(query_lengths, keep_full)
    )
    if all(b == int(q) for b, q in zip(bounds, query_lengths)):
        return None
    kept = (
        bounds
        if limits is None
        else tuple(min(b, int(limit)) for b, limit in zip(bounds, limits))
    )
    count = sum(kept)
    return CEDPlan(
        capacities[bisect_left(capacities, count)],
        count,
        max(bounds, default=0),
        tuple(keep_full),
    )


def nonfinal_limits(
    query_lengths,
    computed,
    prefill_lens,
    keep_full,
    eligible,
    *,
    min_remaining: int,
    chunk_align: int,
) -> list[int] | None:
    """Kept-row limits for prefill chunks whose decoder rows are never read.

    A request's chunk keeps one decoder row instead of its last ``CED_WINDOW``
    when the chunk is truncated anyway (more than ``CED_WINDOW`` rows, not
    ``keep_full``), it is ``eligible``, it ends on a multiple of ``chunk_align``
    (the scheduler's aligned chunk boundaries), and at least ``min_remaining``
    prefill rows follow it. The next chunk then has more than ``CED_WINDOW`` rows,
    so its decoder rows replay from inside it, and every later decoder and draft
    window starts past this chunk. The one row is the chunk's last, whose logits
    the runner discards (a chunked prefill samples no token). Returns None when no
    request qualifies.
    """
    limits = [CED_WINDOW] * len(query_lengths)
    skipped = False
    for i, (q, start, total, full, ok) in enumerate(
        zip(query_lengths, computed, prefill_lens, keep_full, eligible)
    ):
        end = int(start) + int(q)
        if (
            ok
            and not full
            and int(q) > CED_WINDOW
            and end % chunk_align == 0
            and int(total) - end >= min_remaining
        ):
            limits[i] = 1
            skipped = True
    return limits if skipped else None


class CEDState:
    """Stable per-model staging buffers shared by dummy, capture, and runtime."""

    def __init__(
        self, max_requests: int, max_tokens: int, capacities: tuple[int, ...], device
    ):
        self.capacities = capacities
        self.max_requests = max_requests
        self.indices = torch.empty(max_tokens, dtype=torch.int64, device=device)
        self.starts = torch.empty(max_requests + 1, dtype=torch.int32, device=device)
        self.request_positions = torch.empty(
            max_requests, dtype=torch.int64, device=device
        )
        self.replay_start = torch.empty_like(self.request_positions)
        self.counts = torch.empty(2, dtype=torch.int32, device=device)
        self.limits = torch.empty(max_requests, dtype=torch.int32, device=device)
        self.prefix_start = torch.empty_like(self.request_positions)
        self.plan: CEDPlan | None = None

    def warmup(
        self, hidden_size: int, hc_mult: int, *, pipeline: bool, sink_columns: bool
    ) -> None:
        """Resolve the row mapping before graph capture, and declare the metadata
        kernels and the packing layouts the deployment launches for preparation
        (compiled before memory profiling); ``pipeline``: pipeline parallel, whose
        stages also scatter the pre-mix; ``sink_columns``: the model keeps the
        attention-sink detector's columns, which a compacted decoder scatters back
        to every row in the pre-mix's layout.
        """
        device = self.indices.device
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("CED packing must be warmed before graph capture")
        if self.indices.numel() <= CED_WINDOW:
            return
        count = CED_WINDOW + 1
        starts = torch.tensor([0, count], dtype=torch.int32, device=device)
        seq = torch.tensor([count], dtype=torch.int32, device=device)
        rows, streams, mixes = (hidden_size,), (hc_mult, hidden_size), (hc_mult,)
        copies = (
            # Gathered: the decoder boundary's rows, mixes, positions and token
            # ids, the decoder's attention metadata, the step's padding mask and
            # the drafter's context rows, positions and slots.
            (torch.int32, (), False),
            (torch.int64, (), False),
            (torch.bool, (), False),
            (torch.bfloat16, rows, False),
            (torch.bfloat16, streams, False),
            (torch.float32, mixes, False),
            (torch.float32, (hc_mult, hc_mult), False),
            # Scattered back to every row: hidden states (collapsed, or the MTP
            # buffer's streams or a pipeline stage's), and a stage's pre-mix or the
            # sink detector's columns (both [tokens, hc_mult] float32).
            (torch.bfloat16, rows, True),
            (torch.bfloat16, streams, True),
            *(((torch.float32, mixes, True),) if pipeline or sink_columns else ()),
        )

        def warm_metadata():
            self.stage(starts, seq, [count], [False], [0])
            self.plan = None
            self.replay_start.zero_()

        prepare.declare(
            lambda: [
                prepare.call(ced_kernels.indices_kernel, self.max_requests),
                # Packed rows of each layout. A source view of wider rows would
                # stop its first CED step with KernelNotPreparedError.
                *(
                    prepare.call(
                        ced_kernels.copy_rows_kernel,
                        *ced_kernels.packed_copy_rows_key(
                            str(dtype).removeprefix("torch."), tail, scatter
                        ),
                    )
                    for dtype, tail, scatter in copies
                ),
                # Unchanged rows, then a compaction of one request and of several.
                *(
                    prepare.call(
                        ced_kernels.requests_kernel, self.max_requests, compact, single
                    )
                    for compact, single in ((False, False), (True, True), (True, False))
                ),
            ],
            warm=warm_metadata,
        )

    def stage(
        self,
        starts: torch.Tensor,
        seq: torch.Tensor,
        query_lengths,
        keep_full,
        prefix_start,
        limits=None,
    ) -> None:
        """``limits``: optional per-request kept-row limits (``nonfinal_limits``)."""
        nr = len(query_lengths)
        self.plan = plan_ced(query_lengths, keep_full, self.capacities, limits)
        # Fresh pinned sources remain owned by the async-copy allocator until
        # transfer completion; reusing a CPU staging buffer could race overlap.
        # A request keeps at most its limit of rows: all of them, the window, or
        # its nonfinal limit.
        limits_host = torch.tensor(
            [
                KEEP_ALL if full else CED_WINDOW if limits is None else limits[i]
                for i, full in enumerate(keep_full)
            ],
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        prefix_host = torch.tensor(
            prefix_start, dtype=torch.int64, device="cpu", pin_memory=True
        )
        self.limits[:nr].copy_(limits_host, non_blocking=True)
        self.prefix_start[:nr].copy_(prefix_host, non_blocking=True)
        ced_kernels.requests(
            starts,
            seq,
            self.limits,
            self.prefix_start,
            self.starts,
            self.request_positions,
            self.replay_start,
            self.counts,
            nr,
            compact=self.plan is not None,
        )
        if self.plan is not None:
            ced_kernels.indices(
                starts,
                self.starts,
                self.indices[: self.plan.capacity],
                nr,
                self.plan.num_tokens,
            )

    def get_indices(self) -> torch.Tensor | None:
        return None if self.plan is None else self.indices[: self.plan.num_tokens]

    def decoder_metadata(
        self, metadata: DeepseekV41B12xMetadata
    ) -> DeepseekV41B12xMetadata:
        plan = self.plan
        assert plan is not None
        indices = self.indices[: plan.num_tokens]
        # Generic packing zeros invalid rows; native metadata uses -1 sentinels.
        valid = indices >= 0

        def gather(field, sentinel=0):
            result = gather_rows(field, indices)
            return result if sentinel == 0 else result.masked_fill(~valid, sentinel)

        return replace(
            metadata,
            num_actual_tokens=plan.num_tokens,
            max_query_len=plan.max_query_len,
            query_start_loc=self.starts,
            request_positions=self.request_positions,
            live_counts=self.counts,
            positions=gather(metadata.positions, -1),
            req_id_per_token=gather(metadata.req_id_per_token, -1),
            slot_mapping=gather(metadata.slot_mapping, -1),
            cache_lengths=gather(metadata.cache_lengths),
            is_decode=plan.max_query_len <= 1,
            decoder=None,
            swa_replay_start=self.replay_start,
        )
