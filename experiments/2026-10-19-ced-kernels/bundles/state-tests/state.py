# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CED row packing and batch metadata; no model computation lives here."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    TritonWarmupTensor,
    triton_kernel_dispatcher_with_warmup,
    triton_warmup_inputs,
)
from vllm.triton_utils import tl, triton

from .tilelang import ced as ced_kernels
from .tilelang import prepare

if TYPE_CHECKING:
    from .sparse_mla import DeepseekV41B12xMetadata

CED_WINDOW = 128


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


@triton.jit(do_not_specialize=["source_rows", "output_rows"])
def _copy_rows(
    Source,
    Indices,
    Output,
    source_rows,
    output_rows,
    DIMS: tl.constexpr,
    STRIDES: tl.constexpr,
    WIDTH: tl.constexpr,
    SCATTER: tl.constexpr,
    B: tl.constexpr,
):
    offset = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B).to(tl.int64)
    row = offset // WIDTH
    col = offset % WIDTH
    limit = source_rows if SCATTER else output_rows
    index = tl.load(Indices + row, row < limit, other=-1).to(tl.int64)
    valid = (row < limit) & (index >= 0)
    valid &= index < (output_rows if SCATTER else source_rows)
    source_row = row if SCATTER else index
    source_offset = source_row * STRIDES[0]
    remainder = col
    for axis in tl.static_range(len(DIMS) - 1, -1, -1):
        source_offset += (remainder % DIMS[axis]) * STRIDES[axis + 1]
        remainder = remainder // DIMS[axis]
    value = tl.load(Source + source_offset, valid, other=0)
    if SCATTER:
        tl.store(Output + index * WIDTH + col, value, valid)
    else:
        tl.store(Output + offset, value, row < output_rows)


def _copy_rows_warmup_inputs(
    *, layouts: tuple[tuple[torch.dtype, tuple[int, ...], bool], ...]
) -> list[dict[str, Any]]:
    cases = []
    for dtype, tail, scatter in layouts:
        rows = TritonWarmupTensor(dtype, shape=(4, *tail))
        cases.append(
            triton_warmup_inputs(
                _copy_rows,
                rows,
                TritonWarmupTensor(torch.int64),
                rows,
                4,
                4,
                tail,
                rows.stride(),
                int(np.prod(tail)),
                scatter,
                512,
                grid=(1,),
            )
        )
    return cases


_copy_rows_launcher = triton_kernel_dispatcher_with_warmup(
    warmup_inputs=_copy_rows_warmup_inputs
)(_copy_rows)


def register_copy_rows_warmup(
    layouts: tuple[tuple[torch.dtype, tuple[int, ...], bool], ...],
) -> None:
    """Register ``gather_rows`` / ``scatter_rows`` for contiguous, 16-byte aligned
    sources of each ``(dtype, shape[1:], scatter)`` layout with int64 indices."""
    _copy_rows_launcher.register_warmup(layouts=layouts)


@torch.library.custom_op("vllm::dsv41_gather_rows", mutates_args=())
def gather_rows(source: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather first-axis rows, zeroing invalid indices (including graph padding)."""
    output = source.new_empty((indices.numel(), *source.shape[1:]))
    width = (
        source.numel() // source.shape[0]
        if source.shape[0]
        else int(np.prod(source.shape[1:]))
    )
    if output.numel():
        _copy_rows_launcher[(triton.cdiv(output.numel(), 512),)](
            source,
            indices,
            output,
            source.shape[0],
            indices.numel(),
            tuple(source.shape[1:]),
            source.stride(),
            width,
            False,
            512,
        )
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
    width = (
        source.numel() // source.shape[0]
        if source.shape[0]
        else int(np.prod(source.shape[1:]))
    )
    if source.numel():
        _copy_rows_launcher[(triton.cdiv(source.numel(), 512),)](
            source,
            indices,
            output,
            source.shape[0],
            full_rows,
            tuple(source.shape[1:]),
            source.stride(),
            width,
            True,
            512,
        )
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


def plan_ced(query_lengths, keep_full, capacities: tuple[int, ...]) -> CEDPlan | None:
    kept = tuple(
        int(q) if full else min(int(q), CED_WINDOW)
        for q, full in zip(query_lengths, keep_full)
    )
    if all(k == int(q) for k, q in zip(kept, query_lengths)):
        return None
    count = sum(kept)
    return CEDPlan(
        capacities[bisect_left(capacities, count)],
        count,
        max(kept, default=0),
        tuple(keep_full),
    )


@triton.jit(do_not_specialize=["nr"])
def _decoder_requests(
    Starts,
    Seq,
    KeepFull,
    PrefixStart,
    OutStarts,
    RequestPositions,
    ReplayStart,
    Counts,
    nr,
    R: tl.constexpr,
    B: tl.constexpr,
    WINDOW: tl.constexpr,
):
    r = tl.arange(0, B)
    valid = r < nr
    start = tl.load(Starts + r, valid, other=0)
    end = tl.load(Starts + r + 1, valid, other=0)
    seq = tl.load(Seq + r, valid, other=0).to(tl.int64)
    full = tl.load(KeepFull + r, valid, other=0)
    q = end - start
    kept = tl.where(full, q, tl.minimum(q, WINDOW))
    cumulative = tl.cumsum(kept)
    count = tl.sum(kept)
    tl.store(OutStarts + r, tl.where(valid, cumulative - kept, count), r <= R)
    tl.store(OutStarts + nr, count)
    tl.store(RequestPositions + r, tl.where(valid, seq - kept, -1), r < R)
    prefix = tl.load(PrefixStart + r, valid, other=0)
    replay = tl.maximum(prefix, tl.where(q > kept, seq - kept, 0))
    tl.store(ReplayStart + r, replay, r < R)
    tl.store(Counts, count)
    tl.store(Counts + 1, nr)


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
        self.keep_full = torch.empty(max_requests, dtype=torch.bool, device=device)
        self.prefix_start = torch.empty_like(self.request_positions)
        self.plan: CEDPlan | None = None

    def warmup(self, hidden_size: int, hc_mult: int) -> None:
        """Resolve the row mapping before graph capture, and register both packing
        directions with the runner's JIT warmup (compiled before memory profiling).
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
        register_copy_rows_warmup(
            (
                # Gathered: the decoder boundary's rows, mixes, positions and token
                # ids, the decoder's attention metadata, the step's padding mask
                # and the drafter's context rows, positions and slots.
                (torch.int32, (), False),
                (torch.int64, (), False),
                (torch.bool, (), False),
                (torch.bfloat16, rows, False),
                (torch.bfloat16, streams, False),
                (torch.float32, mixes, False),
                (torch.float32, (hc_mult, hc_mult), False),
                # Scattered back to every row: hidden states (collapsed, or the MTP
                # buffer's streams) and a pipeline stage's pre-mix.
                (torch.bfloat16, rows, True),
                (torch.bfloat16, streams, True),
                (torch.float32, mixes, True),
            )
        )

        def warm_metadata():
            self.stage(starts, seq, [count], [False], [0])
            self.plan = None
            self.replay_start.zero_()

        prepare.declare(
            lambda: [prepare.call(ced_kernels.indices_kernel, self.max_requests)],
            warm=warm_metadata,
        )

    def stage(
        self,
        starts: torch.Tensor,
        seq: torch.Tensor,
        query_lengths,
        keep_full,
        prefix_start,
    ) -> None:
        nr = len(query_lengths)
        self.plan = plan_ced(query_lengths, keep_full, self.capacities)
        # Fresh pinned sources remain owned by the async-copy allocator until
        # transfer completion; reusing a CPU staging buffer could race overlap.
        keep_host = torch.tensor(
            keep_full, dtype=torch.bool, device="cpu", pin_memory=True
        )
        prefix_host = torch.tensor(
            prefix_start, dtype=torch.int64, device="cpu", pin_memory=True
        )
        self.keep_full[:nr].copy_(keep_host, non_blocking=True)
        self.prefix_start[:nr].copy_(prefix_host, non_blocking=True)
        _decoder_requests[(1,)](
            starts,
            seq,
            self.keep_full,
            self.prefix_start,
            self.starts,
            self.request_positions,
            self.replay_start,
            self.counts,
            nr,
            self.max_requests,
            triton.next_power_of_2(self.max_requests + 1),
            CED_WINDOW,
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
