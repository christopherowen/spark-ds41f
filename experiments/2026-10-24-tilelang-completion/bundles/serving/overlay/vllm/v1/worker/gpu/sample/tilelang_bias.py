# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Allowed-token, sparse-bias and minimum-length candidate."""

from vllm.tilelang_utils import T, tilelang
from vllm.v1.worker.gpu.sample.tilelang_controls import (
    _DTYPES,
    _HEADER,
    _PASS,
    _declare,
)


def register_bias(
    *,
    vocab_size,
    logits_strides,
    allowed_stride,
    bias_ids_stride,
    bias_stride,
    stop_stride,
    width,
):
    _declare(
        bias_kernel,
        [
            (
                vocab_size,
                stride,
                allowed_stride,
                bias_ids_stride,
                bias_stride,
                stop_stride,
                width,
                restore,
                index_dtype,
                position_dtype,
                run,
            )
            for stride in logits_strides
            for restore in (False, True)
            for index_dtype in _DTYPES.values()
            for position_dtype in _DTYPES.values()
            for run in (1, 4)
        ],
    )


_THREADS = 128
_ROW_TILE = (
    8192  # Triton's LOGITS_BLOCK_SIZE: the row's fill and scan, a tile at a time
)


@tilelang.jit(pass_configs=_PASS)
def bias_kernel(
    vocab: int,
    stride: int,
    allowed_stride: int,
    bias_ids_stride: int,
    bias_stride: int,
    stop_stride: int,
    width: int,
    restore: bool,
    index_dtype: str,
    position_dtype: str,
    run: int,
):
    """``_bias_kernel`` in its tile form: a block a row, each list's ids and values
    loaded (predicated on the list's length) before the gathered logits, and those
    before any store, every masked store predicated. The row's -inf fill and the
    all-masked scan take 8192 columns a tile; only the last, partial tile clamps
    its loads and predicates its stores; with ``run`` 4 (16-byte aligned rows, the
    launch checked) the whole tiles are filled and read as vectors in Triton's
    layout. As Triton's maximum, the scan ignores NaN: a row is alive when some
    logit is above -inf."""
    rows, states, allowed_cols, bias_cols, bias_ids_cols, stop_cols = T.dynamic(
        "rows, states, allowed_cols, bias_cols, bias_ids_cols, stop_cols"
    )
    per = max(width // _THREADS, 1)
    row_per = _ROW_TILE // _THREADS
    full = vocab // _ROW_TILE
    warps = _THREADS // 32

    def load_if(dtype, address, other, ok):
        helper = (
            "vllm_control_load_if_b32"
            if dtype == "int32"
            else ("vllm_control_load_if_f32")
        )
        return T.call_extern(dtype, helper, address, other, ok)

    def store_if(address, value, ok):
        return T.call_extern("handle", "vllm_control_store_if", address, value, ok)

    @T.macro
    def fill_row(logits, row, tx):
        for tile in T.serial(full):
            for r in T.unroll(row_per // run):
                col0 = tile * _ROW_TILE + r * (_THREADS * run) + run * tx
                for k in T.vectorized(run):
                    logits[row, col0 + k] = -T.infinity("float32")
        if vocab % _ROW_TILE:
            for j in T.unroll(row_per):
                col = full * _ROW_TILE + j * _THREADS + tx
                store_if(
                    T.address_of(logits[row, col]), -T.infinity("float32"), col < vocab
                )

    @T.macro
    def scan_row(logits, row, tx, row_values, alive):
        for tile in T.serial(full):
            for r in T.unroll(row_per // run):
                col0 = tile * _ROW_TILE + r * (_THREADS * run) + run * tx
                for k in T.vectorized(run):
                    row_values[r * run + k] = logits[row, col0 + k]
            for j in T.unroll(row_per):
                alive[0] = alive[0] | T.cast(
                    row_values[j] > -T.infinity("float32"), "int32"
                )
        if vocab % _ROW_TILE:
            # A repeated last column does not change whether any is alive.
            for j in T.unroll(row_per):
                col = full * _ROW_TILE + j * _THREADS + tx
                row_values[j] = logits[row, T.min(col, vocab - 1)]
            for j in T.unroll(row_per):
                alive[0] = alive[0] | T.cast(
                    row_values[j] > -T.infinity("float32"), "int32"
                )

    @T.prim_func
    def bias(
        logits: T.StridedTensor((rows, vocab), (stride, 1), "float32"),
        mapping: T.Tensor((rows,), index_dtype),
        positions: T.Tensor((rows,), position_dtype),
        num_allowed: T.Tensor((states,), "int32"),
        allowed: T.StridedTensor((states, allowed_cols), (allowed_stride, 1), "int32"),
        num_bias: T.Tensor((states,), "int32"),
        bias_ids: T.StridedTensor(
            (states, bias_ids_cols), (bias_ids_stride, 1), "int32"
        ),
        biases: T.StridedTensor((states, bias_cols), (bias_stride, 1), "float32"),
        min_lens: T.Tensor((states,), "int32"),
        num_stops: T.Tensor((states,), "int32"),
        restore_row: T.Tensor((states,), "int32"),
        stops: T.StridedTensor((states, stop_cols), (stop_stride, 1), "int32"),
    ):
        with T.Kernel(rows, threads=_THREADS) as block:
            T.import_source(_HEADER)
            tx = T.get_thread_binding()
            row = T.cast(block, "int64")
            req = T.alloc_var(T.int64, init=T.cast(mapping[block], "int64"))
            n_allowed = T.alloc_var(T.int32, init=num_allowed[req])
            n_bias = T.alloc_var(T.int32, init=num_bias[req])
            n_stops = T.alloc_var(T.int32, init=num_stops[req])
            position = T.alloc_var(position_dtype, init=positions[block])
            min_len = T.alloc_var(T.int32, init=min_lens[req])
            # Whether this row restores its stop tokens when nothing else is left.
            keep = T.alloc_var(T.int32, init=restore_row[req] if restore else 0)
            ids = T.alloc_local((per,), "int32")
            values = T.alloc_local((per,), "float32")
            extra = T.alloc_local((per,), "float32")
            row_values = T.alloc_local((row_per,), "float32")
            alive = T.alloc_local((1,), "int32")
            partial = T.alloc_shared((warps,), "int32")
            if n_allowed > 0:
                for k in T.unroll(per):
                    col = k * _THREADS + tx
                    ids[k] = load_if(
                        "int32", T.address_of(allowed[req, col]), 0, col < n_allowed
                    )
                for k in T.unroll(per):
                    values[k] = load_if(
                        "float32",
                        T.address_of(logits[row, ids[k]]),
                        T.float32(0),
                        k * _THREADS + tx < n_allowed,
                    )
                T.sync_threads()
                fill_row(logits, row, tx)
                T.sync_threads()
                for k in T.unroll(per):
                    store_if(
                        T.address_of(logits[row, ids[k]]),
                        values[k],
                        k * _THREADS + tx < n_allowed,
                    )
                T.sync_threads()
            if n_bias > 0:
                for k in T.unroll(per):
                    col = k * _THREADS + tx
                    ids[k] = load_if(
                        "int32", T.address_of(bias_ids[req, col]), 0, col < n_bias
                    )
                    extra[k] = load_if(
                        "float32",
                        T.address_of(biases[req, col]),
                        T.float32(0),
                        col < n_bias,
                    )
                for k in T.unroll(per):
                    values[k] = load_if(
                        "float32",
                        T.address_of(logits[row, ids[k]]),
                        T.float32(0),
                        k * _THREADS + tx < n_bias,
                    )
                for k in T.unroll(per):
                    store_if(
                        T.address_of(logits[row, ids[k]]),
                        values[k] + extra[k],
                        k * _THREADS + tx < n_bias,
                    )
                T.sync_threads()
            if (n_stops > 0) & (position + 1 < min_len):
                for k in T.unroll(per):
                    col = k * _THREADS + tx
                    ids[k] = load_if(
                        "int32", T.address_of(stops[req, col]), 0, col < n_stops
                    )
                if keep != 0:
                    for k in T.unroll(per):
                        values[k] = load_if(
                            "float32",
                            T.address_of(logits[row, ids[k]]),
                            -T.infinity("float32"),
                            k * _THREADS + tx < n_stops,
                        )
                    T.sync_threads()
                for k in T.unroll(per):
                    store_if(
                        T.address_of(logits[row, ids[k]]),
                        -T.infinity("float32"),
                        k * _THREADS + tx < n_stops,
                    )
                if keep != 0:
                    T.sync_threads()
                    alive[0] = 0
                    scan_row(logits, row, tx, row_values, alive)
                    alive[0] = alive[0] | T.shfl_xor(alive[0], 16)
                    alive[0] = alive[0] | T.shfl_xor(alive[0], 8)
                    alive[0] = alive[0] | T.shfl_xor(alive[0], 4)
                    alive[0] = alive[0] | T.shfl_xor(alive[0], 2)
                    alive[0] = alive[0] | T.shfl_xor(alive[0], 1)
                    if tx % 32 == 0:
                        partial[tx // 32] = alive[0]
                    T.sync_threads()
                    alive[0] = partial[0]
                    for w in T.unroll(1, warps):
                        alive[0] = alive[0] | partial[w]
                    for k in T.unroll(per):
                        store_if(
                            T.address_of(logits[row, ids[k]]),
                            values[k],
                            (alive[0] == 0)
                            & (k * _THREADS + tx < n_stops)
                            & (values[k] > -T.infinity("float32"))
                            & (values[k] < T.infinity("float32")),
                        )

    return bias


def _run(logits) -> int:
    """4 when every row starts 16-byte aligned, else 1."""
    return 4 if logits.data_ptr() % 16 == 0 and logits.stride(0) % 4 == 0 else 1


def apply_logit_bias(
    logits,
    mapping,
    positions,
    num_allowed,
    allowed,
    num_bias,
    bias_ids,
    biases,
    min_lens,
    num_stops,
    restore_row,
    stops,
    check_all_masked_rows=False,
):
    if logits.shape[0]:
        width = (
            1
            << (
                max(allowed.shape[1], bias_ids.shape[1], stops.shape[1]) - 1
            ).bit_length()
        )
        bias_kernel(
            logits.shape[1],
            logits.stride(0),
            allowed.stride(0),
            bias_ids.stride(0),
            biases.stride(0),
            stops.stride(0),
            width,
            check_all_masked_rows,
            _DTYPES[mapping.dtype],
            _DTYPES[positions.dtype],
            _run(logits),
        )(
            logits,
            mapping,
            positions,
            num_allowed,
            allowed,
            num_bias,
            bias_ids,
            biases,
            min_lens,
            num_stops,
            restore_row,
            stops,
        )
