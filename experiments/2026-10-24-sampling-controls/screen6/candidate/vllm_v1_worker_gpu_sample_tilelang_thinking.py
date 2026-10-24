# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reasoning-marker cache and budget candidates, including draft boundaries."""

from vllm.tilelang_utils import T, tilelang
from vllm.v1.worker.gpu.sample.tilelang_controls import _DTYPES, _PASS


@tilelang.jit(pass_configs=_PASS)
def marker_cache_kernel(
    tokens_stride: int, index_dtype: str, start_len: int, end_len: int
):
    reqs, states, columns = T.dynamic("reqs, states, columns")

    @T.prim_func
    def update_markers(
        mapping: T.Tensor((reqs,), index_dtype),
        budget: T.Tensor((states,), "int32"),
        tokens: T.StridedTensor((states, columns), (tokens_stride, 1), "int32"),
        lengths: T.Tensor((states,), "int32"),
        cached_start: T.Tensor((states,), "int32"),
        cached_end: T.Tensor((states,), "int32"),
        cached_pos: T.Tensor((states,), "int32"),
        start_ids: T.Tensor((start_len,), "int32"),
        end_ids: T.Tensor((end_len,), "int32"),
    ):
        with T.Kernel(reqs, threads=128) as row:
            req = mapping[row]
            scan = T.alloc_local((1,), "int32")
            last_start = T.alloc_local((1,), "int32")
            last_end = T.alloc_local((1,), "int32")
            hi = T.alloc_local((1,), "int32")
            lo = T.alloc_local((1,), "int32")
            match = T.alloc_local((1,), "bool")
            starts = T.alloc_fragment((1024,), "int32")
            ends = T.alloc_fragment((1024,), "int32")
            start_max = T.alloc_fragment((1,), "int32")
            end_max = T.alloc_fragment((1,), "int32")
            if budget[req] >= 0:
                scan[0] = cached_pos[req]
                last_start[0] = cached_start[req]
                last_end[0] = cached_end[req]
                if scan[0] > lengths[req]:
                    scan[0] = 0
                    last_start[0] = -1
                    last_end[0] = -1
                if (scan[0] == 0) & (last_start[0] < 0) & (last_end[0] < 0):
                    hi[0] = lengths[req]
                    while (hi[0] > 0) & (last_start[0] < 0) & (last_end[0] < 0):
                        lo[0] = T.max(hi[0] - 1024, 0)
                        for col in T.Parallel(1024):
                            pos = lo[0] + col
                            match[0] = (pos < hi[0]) & (pos + start_len <= lengths[req])
                            for j in T.unroll(start_len):
                                if pos + j < lengths[req]:
                                    match[0] = match[0] & (
                                        tokens[req, pos + j] == start_ids[j]
                                    )
                                else:
                                    match[0] = False
                            starts[col] = T.if_then_else(match[0], pos, -1)
                            match[0] = (pos < hi[0]) & (pos + end_len <= lengths[req])
                            for j in T.unroll(end_len):
                                if pos + j < lengths[req]:
                                    match[0] = match[0] & (
                                        tokens[req, pos + j] == end_ids[j]
                                    )
                                else:
                                    match[0] = False
                            ends[col] = T.if_then_else(match[0], pos, -1)
                        T.reduce_max(starts, start_max, dim=0)
                        T.reduce_max(ends, end_max, dim=0)
                        last_start[0] = start_max[0]
                        last_end[0] = end_max[0]
                        hi[0] = lo[0]
                else:
                    if T.get_thread_binding() == 0:
                        for pos in T.serial(scan[0], lengths[req]):
                            if pos + start_len <= lengths[req]:
                                match[0] = True
                                for j in T.unroll(start_len):
                                    match[0] = match[0] & (
                                        tokens[req, pos + j] == start_ids[j]
                                    )
                                if match[0]:
                                    last_start[0] = pos
                            if pos + end_len <= lengths[req]:
                                match[0] = True
                                for j in T.unroll(end_len):
                                    match[0] = match[0] & (
                                        tokens[req, pos + j] == end_ids[j]
                                    )
                                if match[0]:
                                    last_end[0] = pos
                if T.get_thread_binding() == 0:
                    cached_start[req] = last_start[0]
                    cached_end[req] = last_end[0]
                    cached_pos[req] = T.max(
                        lengths[req] - (max(start_len, end_len) - 1), 0
                    )

    return update_markers


@tilelang.jit(pass_configs=_PASS)
def thinking_kernel(
    logits_stride: int,
    tokens_stride: int,
    index_dtype: str,
    start_len: int,
    natural_len: int,
    end_len: int,
):
    rows, vocab, states, columns = T.dynamic("rows, vocab, states, columns")

    @T.macro
    def effective_token(value, tokens, inputs, req, first, total, pos):
        if pos < total:
            value[0] = tokens[req, pos]
        else:
            value[0] = inputs[first + pos - total + 1]

    @T.prim_func
    def thinking(
        logits: T.StridedTensor((rows, vocab), (logits_stride, 1), "float32"),
        mapping: T.Tensor((rows,), index_dtype),
        budgets: T.Tensor((states,), "int32"),
        tokens: T.StridedTensor((states, columns), (tokens_stride, 1), "int32"),
        lengths: T.Tensor((states,), "int32"),
        inputs: T.Tensor((rows,), "int32"),
        positions: T.Tensor((rows,), "int32"),
        cached_start: T.Tensor((states,), "int32"),
        cached_end: T.Tensor((states,), "int32"),
        start_ids: T.Tensor((start_len,), "int32"),
        natural_ids: T.Tensor((natural_len,), "int32"),
        end_ids: T.Tensor((end_len,), "int32"),
    ):
        with T.Kernel(rows, threads=1) as row:
            req = mapping[row]
            first = T.cast(row, "int64") - positions[row]
            total = lengths[req]
            length = total + positions[row]
            last_start = T.alloc_local((1,), "int32")
            last_end = T.alloc_local((1,), "int32")
            prefix = T.alloc_local((1,), "int32")
            actual = T.alloc_local((1,), "int32")
            match = T.alloc_local((1,), "bool")
            if budgets[req] >= 0:
                last_start[0] = cached_start[req]
                last_end[0] = cached_end[req]
                for pos in T.serial(
                    T.max(total - start_len + 1, 0), length - start_len + 1
                ):
                    match[0] = True
                    for j in T.unroll(start_len):
                        effective_token(
                            actual, tokens, inputs, req, first, total, pos + j
                        )
                        match[0] = match[0] & (actual[0] == start_ids[j])
                    if match[0]:
                        last_start[0] = pos
                for pos in T.serial(
                    T.max(total - natural_len + 1, 0), length - natural_len + 1
                ):
                    match[0] = True
                    for j in T.unroll(natural_len):
                        effective_token(
                            actual, tokens, inputs, req, first, total, pos + j
                        )
                        match[0] = match[0] & (actual[0] == natural_ids[j])
                    if match[0]:
                        last_end[0] = pos
                if (
                    (last_start[0] >= 0)
                    & (last_start[0] > last_end[0])
                    & (length - last_start[0] - start_len >= budgets[req])
                ):
                    prefix[0] = 0
                    for n in T.unroll(1, end_len):
                        if n <= length:
                            match[0] = True
                            for j in T.unroll(n):
                                effective_token(
                                    actual,
                                    tokens,
                                    inputs,
                                    req,
                                    first,
                                    total,
                                    length - n + j,
                                )
                                match[0] = match[0] & (actual[0] == end_ids[j])
                            if match[0]:
                                prefix[0] = n
                    logits[T.cast(row, "int64"), end_ids[prefix[0]]] = 1.0e9

    return thinking


def apply_thinking_budget(
    logits,
    req_ids,
    mapping,
    budgets,
    tokens,
    lengths,
    inputs,
    positions,
    cached_start,
    cached_end,
    cached_pos,
    start_ids,
    natural_ids,
    end_ids,
):
    if req_ids.numel():
        marker_cache_kernel(
            tokens.stride(0),
            _DTYPES[req_ids.dtype],
            start_ids.numel(),
            natural_ids.numel(),
        )(
            req_ids,
            budgets,
            tokens,
            lengths,
            cached_start,
            cached_end,
            cached_pos,
            start_ids,
            natural_ids,
        )
    if logits.shape[0]:
        thinking_kernel(
            logits.stride(0),
            tokens.stride(0),
            _DTYPES[mapping.dtype],
            start_ids.numel(),
            natural_ids.numel(),
            end_ids.numel(),
        )(
            logits,
            mapping,
            budgets,
            tokens,
            lengths,
            inputs,
            positions,
            cached_start,
            cached_end,
            start_ids,
            natural_ids,
            end_ids,
        )
