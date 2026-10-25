# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integer row selection for the DeepSeek-V4.1 CED decoder."""

import math

import tilelang
import tilelang.language as T

_PASS = {tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True}


# Predicated global loads (zero when off). They are volatile, so the compiler
# keeps them where they are written, ahead of the stores and branches that use
# them, instead of sinking each into its first use.
def _load32(load, pred=None):
    return T.reinterpret(T.ldg32(load, pred=pred), T.int32)


def _load64(load, pred=None):
    return T.reinterpret(T.ldg64(load, pred=pred), T.int64)


@tilelang.jit(
    pass_configs={
        **_PASS,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    }
)
def requests_kernel(max_requests: int, compact: bool = True, single: bool = False):
    starts_len = T.dynamic("starts_len")
    seq_len = T.dynamic("seq_len")
    chunks = (max_requests + 31) // 32
    scan_steps = min(5, (max_requests - 1).bit_length())
    last_lane = min(max_requests, 32) - 1

    if compact and single:

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
            with T.Kernel(T.ceildiv(max_requests + 1, 128), threads=128) as bx:
                r = bx * 128 + T.get_thread_binding()
                q = _load32(starts[1]) - _load32(starts[0])
                kept = T.min(q, _load32(limits[0]))
                last = T.cast(_load32(seq[0]), T.int64)
                bound = _load64(prefix[0])
                if r <= max_requests:
                    out_starts[r] = T.if_then_else(r == 0, 0, kept)
                if r < max_requests:
                    if r == 0:
                        position = last - kept
                        positions[0] = position
                        replay[0] = T.max(bound, T.if_then_else(q > kept, position, 0))
                    else:
                        positions[r] = -1
                        replay[r] = 0
                if r == 0:
                    counts[0] = kept
                    counts[1] = 1

        return ced_requests

    if not compact:

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
            with T.Kernel(T.ceildiv(max_requests + 1, 128), threads=128) as bx:
                r = bx * 128 + T.get_thread_binding()
                live = r < nr
                start = _load32(starts[T.min(r, nr)])
                end = _load32(starts[r + 1], live)
                last = T.cast(_load32(seq[r], live), T.int64)
                bound = _load64(prefix[r], live)
                total = _load32(starts[nr])
                if r <= max_requests:
                    out_starts[r] = start
                if r < max_requests:
                    positions[r] = T.if_then_else(live, last - (end - start), -1)
                    replay[r] = T.if_then_else(live, bound, 0)
                if r == 0:
                    counts[0] = total
                    counts[1] = nr

        return ced_requests

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
        with T.Kernel(1, threads=32):
            lane = T.get_thread_binding()
            kept = T.alloc_var(T.int32)
            cumulative = T.alloc_var(T.int32)
            carry = T.alloc_var(T.int32)
            carry = 0
            for chunk in T.serial(chunks):
                r = chunk * 32 + lane
                # Every input loads before the scan, so one memory latency covers
                # the request's lengths, sequence end and prefix bound.
                live = r < nr
                q = _load32(starts[r + 1], live) - _load32(starts[r], live)
                limit = _load32(limits[r], live)
                end = T.cast(_load32(seq[r], live), T.int64)
                bound = _load64(prefix[r], live)
                kept = T.min(q, limit)
                if nr <= 1:
                    cumulative = T.shfl_sync(kept, 0) + carry
                else:
                    cumulative = kept
                    for step in T.unroll(scan_steps):
                        prior = T.shfl_up(cumulative, 1 << step)
                        cumulative = cumulative + T.if_then_else(
                            lane >= 1 << step, prior, 0
                        )
                    cumulative = cumulative + carry
                if r < max_requests:
                    out_starts[r] = cumulative - kept
                    position = end - kept
                    positions[r] = T.if_then_else(live, position, -1)
                    replay[r] = T.if_then_else(
                        live, T.max(bound, T.if_then_else(q > kept, position, 0)), 0
                    )
                carry = T.shfl_sync(cumulative, last_lane)
            if lane == 0:
                out_starts[max_requests] = carry
                counts[0] = carry
                counts[1] = nr

    return ced_requests


def requests(
    starts,
    seq,
    limits,
    prefix,
    out_starts,
    positions,
    replay,
    counts,
    nr,
    *,
    compact=True,
):
    requests_kernel(positions.shape[0], compact, compact and nr == 1)(
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
def token_metadata_kernel():
    """The decoder's per-token attention metadata at the CED indices, a thread a
    decoder row: positions, request ids and slots (-1 at an invalid index, as the
    native metadata's sentinels) and cache lengths (0 there); an index past the
    step's rows reads 0, as ``gather_rows`` gives it. ``decoder_metadata``
    gathered each field and masked three of them with torch. Every load issues
    before the stores, clamped into the step's rows."""
    rows, tokens = T.dynamic("rows, tokens")

    @T.prim_func
    def ced_token_metadata(
        indices: T.Tensor((rows,), T.int64),
        positions: T.Tensor((tokens,), T.int64),
        reqs: T.Tensor((tokens,), T.int32),
        slots: T.Tensor((tokens,), T.int64),
        lengths: T.Tensor((tokens,), T.int32),
        positions_out: T.Tensor((rows,), T.int64),
        reqs_out: T.Tensor((rows,), T.int32),
        slots_out: T.Tensor((rows,), T.int64),
        lengths_out: T.Tensor((rows,), T.int32),
    ):
        with T.Kernel(T.ceildiv(rows, 128), threads=128) as bx:
            row = bx * 128 + T.get_thread_binding()
            index = T.alloc_var(T.int64, init=indices[T.min(row, rows - 1)])
            source = T.alloc_var(T.int64, init=T.max(T.min(index, tokens - 1), 0))
            position = T.alloc_var(T.int64, init=positions[source])
            req = T.alloc_var(T.int32, init=reqs[source])
            slot = T.alloc_var(T.int64, init=slots[source])
            length = T.alloc_var(T.int32, init=lengths[source])
            if row < rows:
                # gather_rows' zero past the step's rows, then the sentinels at the
                # negative indices, as the torch chain wrote them.
                valid = index >= 0
                inside = valid & (index < tokens)
                positions_out[row] = T.if_then_else(
                    valid, T.if_then_else(inside, position, T.int64(0)), T.int64(-1)
                )
                reqs_out[row] = T.if_then_else(
                    valid, T.if_then_else(inside, req, 0), -1
                )
                slots_out[row] = T.if_then_else(
                    valid, T.if_then_else(inside, slot, T.int64(0)), T.int64(-1)
                )
                lengths_out[row] = T.if_then_else(inside, length, 0)

    return ced_token_metadata


def token_metadata(indices, positions, reqs, slots, lengths):
    """``(positions, req_id_per_token, slot_mapping, cache_lengths)`` of the
    decoder rows ``indices`` selects."""
    rows = indices.shape[0]
    outs = tuple(field.new_empty(rows) for field in (positions, reqs, slots, lengths))
    if rows:
        token_metadata_kernel()(indices, positions, reqs, slots, lengths, *outs)
    return outs


@tilelang.jit(pass_configs=_PASS)
def slots_kernel():
    """Slots at the CED indices, a thread a decoder row: -1 (``PAD_SLOT_ID``) at an
    invalid index and 0 past the step's rows, as ``gather_rows`` then
    ``masked_fill`` wrote them for the drafter's packed context slots."""
    rows, tokens = T.dynamic("rows, tokens")

    @T.prim_func
    def ced_slots(
        indices: T.Tensor((rows,), T.int64),
        slots: T.Tensor((tokens,), T.int64),
        out: T.Tensor((rows,), T.int64),
    ):
        with T.Kernel(T.ceildiv(rows, 128), threads=128) as bx:
            row = bx * 128 + T.get_thread_binding()
            index = T.alloc_var(T.int64, init=indices[T.min(row, rows - 1)])
            slot = T.alloc_var(T.int64, init=slots[T.max(T.min(index, tokens - 1), 0)])
            if row < rows:
                out[row] = T.if_then_else(
                    index >= 0,
                    T.if_then_else(index < tokens, slot, T.int64(0)),
                    T.int64(-1),
                )

    return ced_slots


def gather_slots(slots, indices):
    """``gather_rows(slots, indices).masked_fill(indices < 0, PAD_SLOT_ID)``."""
    out = slots.new_empty(indices.shape[0])
    if out.numel():
        slots_kernel()(indices, slots, out)
    return out


@tilelang.jit(pass_configs=_PASS)
def routing_mask_kernel():
    """The CED decoder's MoE routing mask, a thread a decoder row: a row routes when
    its index is valid and the step's row there is not padding (a row past the
    step's rows routes, ``gather_rows`` reading its padding as false).
    ``prepare_attn`` wrote it as a comparison, a gather, a negation and a
    conjunction."""
    rows, tokens = T.dynamic("rows, tokens")

    @T.prim_func
    def ced_routing_mask(
        indices: T.Tensor((rows,), T.int64),
        is_padding: T.Tensor((tokens,), T.bool),
        mask: T.Tensor((rows,), T.bool),
    ):
        with T.Kernel(T.ceildiv(rows, 128), threads=128) as bx:
            row = bx * 128 + T.get_thread_binding()
            index = T.alloc_var(T.int64, init=indices[T.min(row, rows - 1)])
            padding = T.alloc_var(
                T.bool, init=is_padding[T.max(T.min(index, tokens - 1), 0)]
            )
            if row < rows:
                mask[row] = T.if_then_else(
                    index < 0,
                    T.bool(False),
                    T.if_then_else(
                        index < tokens,
                        T.if_then_else(padding, T.bool(False), T.bool(True)),
                        T.bool(True),
                    ),
                )

    return ced_routing_mask


def routing_mask(indices, is_padding):
    """``(indices >= 0) & ~gather_rows(is_padding, indices)``."""
    mask = is_padding.new_empty(indices.shape[0])
    if mask.numel():
        routing_mask_kernel()(indices, is_padding, mask)
    return mask


_ITEMSIZE = {
    "bool": 1,
    "int8": 1,
    "uint8": 1,
    "int16": 2,
    "float16": 2,
    "bfloat16": 2,
    "int32": 4,
    "float32": 4,
    "int64": 8,
    "float64": 8,
}
_THREADS = 128

# Predicated global loads for the row copies (zero when off), in the form of
# TileLang's tl::ldg32: 1 and 2 bytes, which it lacks, and 8 bytes into one
# 64-bit register, which spares the register copies its uint2 result costs
# before each index is compared.
_COPY_LOADS = r"""
__device__ __forceinline__ unsigned char ced_ldg8(const void *ptr, bool pred) {
  unsigned int data;
  asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %2, 0;\n mov.b32 %0, 0;\n"
               " @p ld.global.u8 %0, [%1];\n}\n"
               : "=r"(data) : "l"(ptr), "r"((int)pred));
  return (unsigned char)data;
}
__device__ __forceinline__ unsigned short ced_ldg16(const void *ptr, bool pred) {
  unsigned short data;
  asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %2, 0;\n mov.b16 %0, 0;\n"
               " @p ld.global.u16 %0, [%1];\n}\n"
               : "=h"(data) : "l"(ptr), "r"((int)pred));
  return data;
}
__device__ __forceinline__ long long ced_ldg64(const void *ptr, bool pred) {
  long long data;
  asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %2, 0;\n mov.b64 %0, 0;\n"
               " @p ld.global.b64 %0, [%1];\n}\n"
               : "=l"(data) : "l"(ptr), "r"((int)pred));
  return data;
}
"""


def _copy_load(dtype, load, pred):
    """``load``'s element by a predicated load, zero when ``pred`` is off."""
    nbytes = _ITEMSIZE[dtype]
    if nbytes == 4:
        return T.reinterpret(T.ldg32(load, pred=pred), dtype)
    if nbytes == 1:
        return T.cast(
            T.call_extern("uint8", "ced_ldg8", T.address_of(load), pred), dtype
        )
    bits = {2: ("uint16", "ced_ldg16"), 8: ("int64", "ced_ldg64")}[nbytes]
    return T.reinterpret(
        T.call_extern(bits[0], bits[1], T.address_of(load), pred), dtype
    )


def _copy_layout(dtype: str, tail: tuple[int, ...], tail_strides: tuple[int, ...]):
    """A row's width and the elements one thread copies.

    A row whose tail is packed and divides into 16-byte pieces is copied a piece
    per thread (``pieces``), which needs 16-byte aligned rows; any other layout is
    copied four elements per thread, which may span rows.
    """
    width = 1
    packed = []
    for dim in reversed(tail):
        packed.append(width)
        width *= dim
    vector = 16 // _ITEMSIZE[dtype]
    pieces = tuple(reversed(packed)) == tuple(tail_strides) and width % vector == 0
    return width, (vector if pieces else 4), pieces


@tilelang.jit(
    pass_configs={
        **_PASS,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    }
)
def copy_rows_kernel(
    dtype: str,
    tail: tuple[int, ...],
    tail_strides: tuple[int, ...],
    packed: bool,
    scatter: bool,
):
    """Specialized by element type, row shape, in-row strides and whether source
    rows are packed (row stride = row width, a constant then). A view of wider
    rows passes its row stride, so all such views share one kernel."""
    source_span = T.dynamic("source_span")
    index_rows = T.dynamic("index_rows")
    output_span = T.dynamic("output_span")
    width, group, pieces = _copy_layout(dtype, tail, tail_strides)

    def column_offset(col):
        # Build the strided address at specialization time; a TIR loop variable
        # cannot index Python's layout tuples.
        result = 0
        divisor = 1
        for dim, stride in zip(reversed(tail), reversed(tail_strides)):
            result = result + (col // divisor % dim) * stride
            divisor *= dim
        return result

    # Loads issue before any branch: the row indices, then the rows, then the
    # stores. Branching on validity first serializes those latencies per element.
    # Indices, whole-row pieces and gathered elements use predicated loads, which
    # touch no memory when off and then read as zero bits (a gathered invalid
    # row's zero); they are volatile, so the compiler cannot sink a scatter's row
    # load behind the store's validity test. Scattered elements load from their
    # clamped row (a launched block has a row). The source is not marked
    # restrict, so its loads take the coherent path: the read-only path a
    # restrict source takes is slower for these dependent loads. Element layouts
    # keep Triton's coalescing: each thread holds 16-byte aligned runs of up to
    # four elements, repeated a block-width apart.
    run = min(4, 16 // _ITEMSIZE[dtype])

    def element(bx, tx, k):
        return (
            T.cast(bx, T.int64) * (_THREADS * group)
            + k // run * (_THREADS * run)
            + tx * run
            + k % run
        )

    @T.prim_func
    def ced_copy_rows(
        source: T.Tensor((source_span,), dtype),
        row_indices: T.Tensor((index_rows,), T.int64),
        output: T.Tensor((output_span,), dtype),
        source_rows: T.int64,
        output_rows: T.int64,
        row_stride: T.int64,
    ):
        with T.Kernel(
            T.ceildiv(
                (source_rows if scatter else output_rows) * width, _THREADS * group
            ),
            threads=_THREADS,
            prelude=_COPY_LOADS,
        ) as bx:
            T.annotate_restrict_buffers(source)
            limit = source_rows if scatter else output_rows
            bound = output_rows if scatter else source_rows
            stride = width if packed else row_stride
            tx = T.get_thread_binding()
            if pieces:
                first = (T.cast(bx, T.int64) * _THREADS + tx) * group
                row = first // width
                col = first % width
                live = row < limit
                index = _copy_load("int64", row_indices[row], live)
                valid = live & (index >= 0) & (index < bound)
                if scatter:
                    piece = T.ldg128(source[row * stride + col], pred=live)
                    T.stg128(output[index * width + col], piece, pred=valid)
                else:
                    piece = T.ldg128(source[index * stride + col], pred=valid)
                    T.stg128(output[first], piece, pred=live)
            else:
                indices = T.alloc_local((group,), T.int64)
                values = T.alloc_local((group,), dtype)
                for k in T.unroll(group):
                    indices[k] = _copy_load(
                        "int64",
                        row_indices[element(bx, tx, k) // width],
                        element(bx, tx, k) // width < limit,
                    )
                for k in T.unroll(group):
                    row = element(bx, tx, k) // width
                    col = element(bx, tx, k) % width
                    if scatter:
                        values[k] = source[
                            T.min(row, limit - 1) * stride + column_offset(col)
                        ]
                    else:
                        valid = (row < limit) & (indices[k] >= 0) & (indices[k] < bound)
                        values[k] = _copy_load(
                            dtype,
                            source[indices[k] * stride + column_offset(col)],
                            valid,
                        )
                for k in T.unroll(group):
                    row = element(bx, tx, k) // width
                    col = element(bx, tx, k) % width
                    valid = (row < limit) & (indices[k] >= 0) & (indices[k] < bound)
                    if scatter:
                        if valid:
                            output[indices[k] * width + col] = values[k]
                    elif row < limit:
                        output[element(bx, tx, k)] = values[k]

    return ced_copy_rows


def copy_rows_key(dtype: str, tail, tail_strides, row_stride: int, scatter: bool):
    """``copy_rows_kernel``'s arguments for a source layout: what the copy launches
    and what an owner declares for preparation."""
    return (
        dtype,
        tuple(tail),
        tuple(tail_strides),
        row_stride == math.prod(tail),
        scatter,
    )


def packed_copy_rows_key(dtype: str, tail, scatter: bool):
    """``copy_rows_key`` for a source of packed rows of shape ``tail``."""
    strides = [1]
    for dim in reversed(tail[1:]):
        strides.insert(0, strides[0] * dim)
    return copy_rows_key(dtype, tail, strides[: len(tail)], math.prod(tail), scatter)


def copy_rows(source, row_indices, output, scatter: bool):
    """Copy logical rows from a possibly strided source without copying its storage.

    Gathering reads ``source[row_indices[i]]`` into ``output[i]``; scattering writes
    ``source[i]`` to ``output[row_indices[i]]``. Indices outside the indexed tensor
    are skipped (gathered as zero). There must be a row to copy, and a row copied
    in 16-byte pieces needs 16-byte aligned source rows and output storage.
    """
    dtype = str(source.dtype).removeprefix("torch.")
    tail = tuple(source.shape[1:])
    tail_strides = tuple(source.stride()[1:])
    row_stride = source.stride(0)
    if row_indices.numel() != (source.shape[0] if scatter else output.shape[0]):
        raise ValueError("CED row copies need one index per copied row")
    if not row_indices.numel():
        raise ValueError("CED row copies need a row to copy")
    if _copy_layout(dtype, tail, tail_strides)[2] and (
        source.data_ptr() % 16
        or output.data_ptr() % 16
        or row_stride * source.element_size() % 16
    ):
        raise ValueError("CED row copies of whole rows need 16-byte aligned rows")
    span = (
        1 + sum((n - 1) * s for n, s in zip(source.shape, source.stride()))
        if source.numel()
        else 0
    )
    flat_source = source.as_strided((span,), (1,))
    copy_rows_kernel(*copy_rows_key(dtype, tail, tail_strides, row_stride, scatter))(
        flat_source,
        row_indices,
        output.view(-1),
        source.shape[0],
        output.shape[0],
        row_stride,
    )
