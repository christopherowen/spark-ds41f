# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integer row selection for the DeepSeek-V4.1 CED decoder."""

import tilelang
import tilelang.language as T

_PASS = {tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True}


@tilelang.jit(pass_configs=_PASS)
def requests_kernel(max_requests: int):
    starts_len = T.dynamic("starts_len")
    seq_len = T.dynamic("seq_len")
    block = max(32, 1 << max_requests.bit_length())

    @T.prim_func
    def ced_requests(
        starts: T.Tensor((starts_len,), T.int32),
        seq: T.Tensor((seq_len,), T.int32),
        limits: T.Tensor((max_requests,), T.int32),
        prefix: T.Tensor((max_requests,), T.int64),
        out_starts: T.Tensor((max_requests + 1,), T.int32),
        positions: T.Tensor((max_requests,), T.int64),
        replay: T.Tensor((max_requests,), T.int64),
        counts: T.Tensor((2,), T.int32),
        nr: T.int32,
    ):
        with T.Kernel(1, threads=128):
            kept = T.alloc_shared((block,), T.int32)
            cumulative = T.alloc_shared((block,), T.int32)
            for r in T.Parallel(block):
                kept[r] = T.if_then_else(
                    r < nr, T.min(starts[r + 1] - starts[r], limits[r]), 0
                )
            T.copy(kept, cumulative)
            T.cumsum(cumulative, dim=0)
            for r in T.Parallel(max_requests + 1):
                out_starts[r] = T.if_then_else(
                    r < nr, cumulative[r] - kept[r], cumulative[block - 1]
                )
            for r in T.Parallel(max_requests):
                if r < nr:
                    q = starts[r + 1] - starts[r]
                    position = T.cast(seq[r], T.int64) - kept[r]
                    positions[r] = position
                    replay[r] = T.max(
                        prefix[r], T.if_then_else(q > kept[r], position, 0)
                    )
                else:
                    positions[r] = -1
                    replay[r] = 0
            for i in T.Parallel(2):
                counts[i] = T.if_then_else(i == 0, cumulative[block - 1], nr)

    return ced_requests


def requests(starts, seq, limits, prefix, out_starts, positions, replay, counts, nr):
    requests_kernel(positions.shape[0])(
        starts, seq, limits, prefix, out_starts, positions, replay, counts, nr
    )


@tilelang.jit(pass_configs=_PASS)
def indices_kernel(max_requests: int):
    starts_len = T.dynamic("starts_len")
    capacity = T.dynamic("capacity")
    steps = max(1, (max_requests + 1).bit_length())

    @T.prim_func
    def ced_indices(
        starts: T.Tensor((starts_len,), T.int32),
        compact_starts: T.Tensor((max_requests + 1,), T.int32),
        indices: T.Tensor((capacity,), T.int64),
        nr: T.int32,
        nt: T.int32,
    ):
        with T.Kernel(T.ceildiv(capacity, 128), threads=128) as bx:
            row = bx * 128 + T.get_thread_binding()
            bounds = T.alloc_local((2,), T.int32)
            if row < capacity:
                bounds[0] = 0
                bounds[1] = nr
                for _ in T.serial(steps):
                    if bounds[0] < bounds[1]:
                        mid = (bounds[0] + bounds[1]) // 2
                        end = T.if_then_else(mid < nr, compact_starts[mid + 1], nt)
                        if end <= row:
                            bounds[0] = mid + 1
                        else:
                            bounds[1] = mid
                request = bounds[0]
                if request < nr:
                    indices[row] = (
                        T.cast(starts[request + 1], T.int64)
                        - compact_starts[request + 1]
                        + row
                    )
                else:
                    indices[row] = -1

    return ced_indices


def indices(starts, compact_starts, output, nr, nt):
    indices_kernel(compact_starts.shape[0] - 1)(starts, compact_starts, output, nr, nt)


@tilelang.jit(pass_configs=_PASS)
def copy_rows_kernel(
    dtype: str, tail: tuple[int, ...], strides: tuple[int, ...], scatter: bool
):
    source_span = T.dynamic("source_span")
    index_rows = T.dynamic("index_rows")
    output_span = T.dynamic("output_span")
    width = 1
    for dim in tail:
        width *= dim

    @T.prim_func
    def ced_copy_rows(
        source: T.Tensor((source_span,), dtype),
        row_indices: T.Tensor((index_rows,), T.int64),
        output: T.Tensor((output_span,), dtype),
        source_rows: T.int64,
        output_rows: T.int64,
    ):
        with T.Kernel(
            T.ceildiv((source_rows if scatter else output_rows) * width, 512),
            threads=128,
        ) as bx:
            for i in T.Parallel(512):
                offset = T.cast(bx * 512 + i, T.int64)
                row = offset // width
                col = offset % width
                limit = source_rows if scatter else output_rows
                if row < limit:
                    index = row_indices[row]
                    valid = (index >= 0) & (
                        index < (output_rows if scatter else source_rows)
                    )
                    if valid:
                        source_row = row if scatter else index
                        source_offset = source_row * strides[0]
                        remainder = col
                        for reverse in T.unroll(len(tail)):
                            axis = len(tail) - 1 - reverse
                            source_offset += (remainder % tail[axis]) * strides[
                                axis + 1
                            ]
                            remainder = remainder // tail[axis]
                        if scatter:
                            output[index * width + col] = source[source_offset]
                        else:
                            output[offset] = source[source_offset]
                    elif not scatter:
                        output[offset] = T.cast(0, dtype)

    return ced_copy_rows


def copy_rows(source, row_indices, output, scatter: bool):
    """Copy logical rows from a possibly strided source without copying its storage."""
    span = 1 + sum((n - 1) * s for n, s in zip(source.shape, source.stride()))
    flat_source = source.as_strided((span,), (1,))
    copy_rows_kernel(
        str(source.dtype).removeprefix("torch."),
        tuple(source.shape[1:]),
        tuple(source.stride()),
        scatter,
    )(flat_source, row_indices, output.view(-1), source.shape[0], output.shape[0])
