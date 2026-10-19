# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Frozen r6x Triton references for experimental CED port qualification."""

from vllm.triton_utils import tl, triton


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


@triton.jit(do_not_specialize=["nr", "nt", "capacity"])
def _decoder_indices(Starts, CompactStarts, Indices, nr, nt, capacity, B: tl.constexpr):
    row = tl.program_id(0) * B + tl.arange(0, B)
    lo = tl.full((B,), 0, tl.int32)
    hi = tl.full((B,), nr, tl.int32)
    while tl.sum((lo < hi).to(tl.int32), 0) > 0:
        mid = (lo + hi) // 2
        end = tl.load(CompactStarts + mid + 1, mid < nr, other=nt)
        right = end <= row
        active = lo < hi
        lo = tl.where(active & right, mid + 1, lo)
        hi = tl.where(active & ~right, mid, hi)
    valid = lo < nr
    compact_end = tl.load(CompactStarts + lo + 1, valid, other=0)
    original_end = tl.load(Starts + lo + 1, valid, other=0).to(tl.int64)
    tl.store(
        Indices + row,
        tl.where(valid, original_end - compact_end + row, -1),
        row < capacity,
    )
