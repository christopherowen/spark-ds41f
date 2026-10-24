# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Allowed-token, sparse-bias and minimum-length candidate."""

from vllm.tilelang_utils import T, tilelang
from vllm.v1.worker.gpu.sample.tilelang_controls import _DTYPES, _PASS


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
):
    rows, states, allowed_cols, bias_cols, bias_ids_cols, stop_cols = T.dynamic(
        "rows, states, allowed_cols, bias_cols, bias_ids_cols, stop_cols"
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
        with T.Kernel(rows, threads=128) as row:
            req = mapping[row]
            saved = T.alloc_fragment((width,), "float32")
            saved_ids = T.alloc_fragment((width,), "int32")
            alive = T.alloc_fragment((8192,), "int32")
            any_alive = T.alloc_fragment((1,), "int32")
            if num_allowed[req] > 0:
                for col in T.Parallel(width):
                    if col < num_allowed[req]:
                        saved_ids[col] = allowed[req, col]
                        saved[col] = logits[T.cast(row, "int64"), saved_ids[col]]
                T.sync_threads()
                for block in T.serial(T.ceildiv(vocab, 8192)):
                    for col in T.Parallel(8192):
                        index = block * 8192 + col
                        if index < vocab:
                            logits[T.cast(row, "int64"), index] = -T.infinity("float32")
                T.sync_threads()
                for col in T.Parallel(width):
                    if col < num_allowed[req]:
                        logits[T.cast(row, "int64"), saved_ids[col]] = saved[col]
                T.sync_threads()
            if num_bias[req] > 0:
                for col in T.Parallel(width):
                    if col < num_bias[req]:
                        token = bias_ids[req, col]
                        logits[T.cast(row, "int64"), token] += biases[req, col]
                T.sync_threads()
            if (num_stops[req] > 0) & (positions[row] + 1 < min_lens[req]):
                for col in T.Parallel(width):
                    if col < num_stops[req]:
                        saved_ids[col] = stops[req, col]
                        if restore:
                            saved[col] = logits[T.cast(row, "int64"), saved_ids[col]]
                T.sync_threads()
                for col in T.Parallel(width):
                    if col < num_stops[req]:
                        logits[T.cast(row, "int64"), saved_ids[col]] = -T.infinity(
                            "float32"
                        )
                T.sync_threads()
                if restore and restore_row[req] != 0:
                    T.clear(alive)
                    for block in T.serial(T.ceildiv(vocab, 8192)):
                        for col in T.Parallel(8192):
                            index = block * 8192 + col
                            value = T.if_then_else(
                                index < vocab,
                                logits[T.cast(row, "int64"), index],
                                -T.infinity("float32"),
                            )
                            alive[col] |= T.cast(
                                value != -T.infinity("float32"), "int32"
                            )
                    T.reduce_max(alive, any_alive, dim=0)
                    if any_alive[0] == 0:
                        for col in T.Parallel(width):
                            if col < num_stops[req]:  # noqa: SIM102
                                if (saved[col] > -T.infinity("float32")) & (
                                    saved[col] < T.infinity("float32")
                                ):
                                    logits[T.cast(row, "int64"), saved_ids[col]] = (
                                        saved[col]
                                    )

    return bias


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
