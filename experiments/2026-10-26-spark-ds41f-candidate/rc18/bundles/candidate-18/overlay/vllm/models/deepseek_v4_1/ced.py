# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CED row packing and batch metadata; no model computation lives here."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import torch

from .tilelang import ced as ced_kernels
from .tilelang import columns, prepare

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


def gather_rows(source: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather first-axis rows, zeroing invalid indices (including graph padding)."""
    output = source.new_empty((indices.numel(), *source.shape[1:]))
    if output.numel():
        ced_kernels.copy_rows(source, indices, output, scatter=False)
    return output


def scatter_rows(
    source: torch.Tensor, indices: torch.Tensor, full_rows: int
) -> torch.Tensor:
    """Restore the full-row ABI; discarded rows are always zero."""
    output = columns.zero_(source.new_empty((full_rows, *source.shape[1:])))
    if source.numel():
        ced_kernels.copy_rows(source, indices, output, scatter=True)
    return output


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
    rows, or every row where ``keep_full``. ``limits`` (``replay_rows``) lowers a
    request's kept rows further. None when every request keeps all its rows;
    ``max_query_len`` keeps the window bound, so the decoder runs in the mode it
    would without the limits."""
    bounds = tuple(
        int(q) if full else min(int(q), CED_WINDOW)
        for q, full in zip(query_lengths, keep_full)
    )
    kept = (
        bounds
        if limits is None
        else tuple(min(b, int(limit)) for b, limit in zip(bounds, limits))
    )
    if all(k == int(q) for k, q in zip(kept, query_lengths)):
        return None
    count = sum(kept)
    return CEDPlan(
        capacities[bisect_left(capacities, count)],
        count,
        max(bounds, default=0),
        tuple(keep_full),
    )


def replay_rows(
    query_lengths, computed, prefill_lens, prefix_start, keep_full
) -> tuple[list[int], list[int]]:
    """Each request's replay floor and kept-row limit, anchored at its prompt end.

    The decoder replays the last ``CED_WINDOW`` tokens of the prompt (the V4.1
    paper's bounded replay): a prefilling request's floor is ``prefill_len -
    CED_WINDOW``, or its prefix start where that is later (the decoder has no rows
    before it), and every chunk of the prompt keeps exactly its rows from the floor
    on. A chunk that ends at or before the floor keeps one row, its last, whose
    logits the runner discards (a chunked prefill samples no token); the requests
    kernel replays that row from itself. Wherever the prompt's chunks end, each
    replayed row then sees the same decoder rows, and the drafter, whose context
    rows are the kept rows, the same context. ``prefill_len`` is what the runner
    prefills: the prompt, and the outputs a resumed request recomputes.

    A decoding request keeps up to ``CED_WINDOW`` rows from its prefix start: its
    rows lie past the prompt's floor, whose window no longer reaches it. A
    ``keep_full`` request (prompt logprobs) keeps every row from its prefix start.
    """
    floors, limits = [], []
    for q, start, total, prefix, full in zip(
        query_lengths, computed, prefill_lens, prefix_start, keep_full
    ):
        start, total, prefix = int(start), int(total), int(prefix)
        if full:
            floors.append(prefix)
            limits.append(KEEP_ALL)
        elif start < total:
            floor = max(prefix, total - CED_WINDOW)
            floors.append(floor)
            limits.append(min(max(start + int(q) - floor, 1), CED_WINDOW))
        else:
            floors.append(prefix)
            limits.append(CED_WINDOW)
    return floors, limits


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
        self.floors = torch.empty_like(self.request_positions)
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
                prepare.call(ced_kernels.token_metadata_kernel),
                prepare.call(ced_kernels.routing_mask_kernel),
                prepare.call(ced_kernels.slots_kernel),
                prepare.call(columns.zero_packs_kernel),
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
        floors,
        limits=None,
    ) -> None:
        """``floors``: each request's replay floor; ``limits``: optional per-request
        kept-row limits (both from ``replay_rows``)."""
        nr = len(query_lengths)
        self.plan = plan_ced(query_lengths, keep_full, self.capacities, limits)
        # Fresh pinned sources remain owned by the async-copy allocator until
        # transfer completion; reusing a CPU staging buffer could race overlap.
        # A request keeps at most its limit of rows: all of them, the window, or
        # the rows from its replay floor on.
        limits_host = torch.tensor(
            [
                KEEP_ALL if full else CED_WINDOW if limits is None else limits[i]
                for i, full in enumerate(keep_full)
            ],
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        floors_host = torch.tensor(
            floors, dtype=torch.int64, device="cpu", pin_memory=True
        )
        self.limits[:nr].copy_(limits_host, non_blocking=True)
        self.floors[:nr].copy_(floors_host, non_blocking=True)
        ced_kernels.requests(
            starts,
            seq,
            self.limits,
            self.floors,
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
        # The native metadata's sentinels at invalid rows: -1, and 0 lengths.
        positions, reqs, slots, lengths = ced_kernels.token_metadata(
            indices,
            metadata.positions,
            metadata.req_id_per_token,
            metadata.slot_mapping,
            metadata.cache_lengths,
        )
        return replace(
            metadata,
            num_actual_tokens=plan.num_tokens,
            max_query_len=plan.max_query_len,
            query_start_loc=self.starts,
            request_positions=self.request_positions,
            live_counts=self.counts,
            positions=positions,
            req_id_per_token=reqs,
            slot_mapping=slots,
            cache_lengths=lengths,
            is_decode=plan.max_query_len <= 1,
            decoder=None,
            swa_replay_start=self.replay_start,
        )
