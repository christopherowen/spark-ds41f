# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integer-only sampler bookkeeping and forbidden-sequence candidates."""

import torch

from vllm.tilelang_utils import T, tilelang
from vllm.v1.worker.gpu.sample.tilelang_controls import (
    _DTYPES,
    _HEADER,
    _PASS,
    _declare,
)


def register_bad_words(
    *,
    logits_stride,
    bad_word_token_ids_stride,
    bad_word_offsets_stride,
    all_token_ids_stride,
):
    _declare(
        bad_words_kernel,
        [
            (
                logits_stride,
                bad_word_token_ids_stride,
                bad_word_offsets_stride,
                all_token_ids_stride,
                dtype,
            )
            for dtype in _DTYPES.values()
        ],
    )


def register_prompt_ids(*, all_token_ids_stride):
    _declare(
        prompt_ids_kernel, [(all_token_ids_stride, dtype) for dtype in _DTYPES.values()]
    )


def register_fill_ids():
    _declare(
        fill_ids_kernel,
        [
            (mapping, sampled, topk)
            for mapping in _DTYPES.values()
            for sampled in _DTYPES.values()
            for topk in _DTYPES.values()
        ],
    )


def register_nan_counts(*, logits, vocab_size):
    _declare(
        nan_count_kernel,
        [(vocab_size, stride, str(dtype)[6:]) for dtype, stride in logits],
    )


@tilelang.jit(pass_configs=_PASS)
def bad_words_kernel(
    logits_stride: int,
    words_stride: int,
    offsets_stride: int,
    tokens_stride: int,
    index_dtype: str,
):
    rows, vocab, states, word_cols, offset_cols, token_cols = T.dynamic(
        "rows, vocab, states, word_cols, offset_cols, token_cols"
    )

    @T.prim_func
    def bad_words(
        logits: T.StridedTensor((rows, vocab), (logits_stride, 1), "float32"),
        mapping: T.Tensor((rows,), index_dtype),
        words: T.StridedTensor((states, word_cols), (words_stride, 1), "int32"),
        offsets: T.StridedTensor((states, offset_cols), (offsets_stride, 1), "int32"),
        num_words: T.Tensor((states,), "int32"),
        tokens: T.StridedTensor((states, token_cols), (tokens_stride, 1), "int32"),
        prompt_len: T.Tensor((states,), "int32"),
        total_len: T.Tensor((states,), "int32"),
        inputs: T.Tensor((rows,), "int32"),
        local_pos: T.Tensor((rows,), "int32"),
        max_words: T.int32,
    ):
        with T.Kernel(rows, T.ceildiv(max_words, 32), threads=32) as (row, group):
            word = group * 32 + T.get_thread_binding()
            req = T.alloc_local((1,), "int64")
            actual = T.alloc_local((1,), "int32")
            matches = T.alloc_local((1,), "int32")
            req[0] = mapping[row]
            if word < num_words[req[0]]:
                start = offsets[req[0], word]
                end = offsets[req[0], word + 1]
                prefix = end - start - 1
                output_len = total_len[req[0]] - prompt_len[req[0]]
                effective_len = output_len + local_pos[row]
                if prefix <= effective_len:
                    matches[0] = 1
                    for i in T.serial(prefix):
                        pos = effective_len - prefix + i
                        if pos >= output_len:
                            actual[0] = inputs[
                                row - local_pos[row] + pos - output_len + 1
                            ]
                        else:
                            actual[0] = tokens[req[0], prompt_len[req[0]] + pos]
                        matches[0] &= T.cast(
                            words[req[0], start + i] == actual[0], "int32"
                        )
                    if matches[0] != 0:
                        logits[
                            T.cast(row, "int64"), words[req[0], end - 1]
                        ] = -T.infinity("float32")

    return bad_words


def apply_bad_words(
    logits,
    mapping,
    words,
    offsets,
    num_words,
    tokens,
    prompt_len,
    total_len,
    inputs,
    local_pos,
    max_words,
):
    if logits.shape[0] and max_words:
        bad_words_kernel(
            logits.stride(0),
            words.stride(0),
            offsets.stride(0),
            tokens.stride(0),
            _DTYPES[mapping.dtype],
        )(
            logits,
            mapping,
            words,
            offsets,
            num_words,
            tokens,
            prompt_len,
            total_len,
            inputs,
            local_pos,
            max_words,
        )


_PROMPT_THREADS = 128
_PROMPT_GROUP = 1024  # tokens a group: eight a thread
_PROMPT_GROUPS = 4  # groups a pass: a 16K chunk in four passes


@tilelang.jit(pass_configs=_PASS)
def prompt_ids_kernel(tokens_stride: int, index_dtype: str):
    """A block a request: the token after each of its chunk's positions. A pass
    covers four 1024-token groups and issues all of its loads before its stores
    (both predicated on the chunk's length), so a long chunk waits on memory once a
    pass; a group the chunk does not reach is skipped, on the length alone."""
    rows, output_len, states, token_cols = T.dynamic(
        "rows, output_len, states, token_cols"
    )
    per = _PROMPT_GROUP // _PROMPT_THREADS
    tile = _PROMPT_GROUP * _PROMPT_GROUPS

    @T.prim_func
    def prompt_ids(
        output: T.Tensor((output_len,), "int64"),
        query_starts: T.Tensor((rows + 1,), "int32"),
        mapping: T.Tensor((rows,), index_dtype),
        computed: T.Tensor((states,), "int32"),
        tokens: T.StridedTensor((states, token_cols), (tokens_stride, 1), "int32"),
    ):
        with T.Kernel(rows, threads=_PROMPT_THREADS) as row:
            T.import_source(_HEADER)
            tx = T.get_thread_binding()
            req = T.alloc_var(T.int64, init=T.cast(mapping[row], "int64"))
            start = T.alloc_var(T.int32, init=query_starts[row])
            length = T.alloc_var(T.int32, init=query_starts[row + 1] - start)
            first = T.alloc_var(T.int32, init=computed[req] + 1)
            values = T.alloc_local((per * _PROMPT_GROUPS,), "int32")
            for block in T.serial(T.ceildiv(length, tile)):
                for g in T.unroll(_PROMPT_GROUPS):
                    if block * tile + g * _PROMPT_GROUP < length:
                        for k in T.unroll(per):
                            pos = (
                                block * tile + g * _PROMPT_GROUP
                                + k * _PROMPT_THREADS + tx
                            )  # fmt: skip
                            values[g * per + k] = T.call_extern(
                                "int32",
                                "vllm_control_load_if_b32",
                                T.address_of(tokens[req, first + pos]),
                                0,
                                pos < length,
                            )
                for g in T.unroll(_PROMPT_GROUPS):
                    if block * tile + g * _PROMPT_GROUP < length:
                        for k in T.unroll(per):
                            pos = (
                                block * tile + g * _PROMPT_GROUP
                                + k * _PROMPT_THREADS + tx
                            )  # fmt: skip
                            T.call_extern(
                                "handle",
                                "vllm_control_store_if_b64",
                                T.address_of(output[start + pos]),
                                T.cast(values[g * per + k], "int64"),
                                pos < length,
                            )

    return prompt_ids


def prompt_token_ids(output, query_starts, mapping, computed, tokens):
    if mapping.numel():
        prompt_ids_kernel(tokens.stride(0), _DTYPES[mapping.dtype])(
            output, query_starts, mapping, computed, tokens
        )


@tilelang.jit(pass_configs=_PASS)
def fill_ids_kernel(index_dtype: str, sampled_dtype: str, topk_dtype: str):
    output_stride, mask_stride, topk_stride, custom_stride = T.dynamic(
        "output_stride, mask_stride, topk_stride, custom_stride", "int64"
    )
    rows, out_cols, mask_cols, topk_rows, topk_cols, states, custom_cols = T.dynamic(
        "rows, out_cols, mask_cols, topk_rows, topk_cols, states, custom_cols"
    )

    @T.prim_func
    def fill_ids(
        output: T.StridedTensor((rows, out_cols), (output_stride, 1), "int64"),
        valid: T.StridedTensor((rows, mask_cols), (mask_stride, 1), "bool"),
        sampled: T.Tensor((rows,), sampled_dtype),
        topk_ids: T.StridedTensor((topk_rows, topk_cols), (topk_stride, 1), topk_dtype),
        mapping: T.Tensor((rows,), index_dtype),
        num_custom: T.Tensor((states,), "int32"),
        custom_ids: T.StridedTensor((states, custom_cols), (custom_stride, 1), "int32"),
        topk: T.int32,
    ):
        with T.Kernel(rows, threads=128) as row:
            req = mapping[row]
            if T.get_thread_binding() == 0:
                output[row, 0] = T.cast(sampled[row], "int64")
                valid[row, 0] = True
            # Every column is written: an unused one is 0 and invalid (the zeros
            # the outputs were allocated with before).
            for i in T.serial(T.ceildiv(out_cols - 1, 128)):
                col = i * 128 + T.get_thread_binding()
                if col + 1 < out_cols:
                    if num_custom[req] > 0:
                        if col < num_custom[req]:
                            output[row, col + 1] = T.cast(custom_ids[req, col], "int64")
                            valid[row, col + 1] = True
                        else:
                            output[row, col + 1] = T.int64(0)
                            valid[row, col + 1] = False
                    elif col < topk:
                        output[row, col + 1] = T.cast(topk_ids[row, col], "int64")
                        valid[row, col + 1] = True
                    else:
                        output[row, col + 1] = T.int64(0)
                        valid[row, col + 1] = False

    return fill_ids


def fill_token_ids(
    output, valid, sampled, topk_ids, mapping, num_custom, custom_ids, topk
):
    if mapping.numel():
        fill_ids_kernel(
            _DTYPES[mapping.dtype], _DTYPES[sampled.dtype], _DTYPES[topk_ids.dtype]
        )(output, valid, sampled, topk_ids, mapping, num_custom, custom_ids, topk)


@tilelang.jit(pass_configs=_PASS)
def nan_count_kernel(vocab: int, stride: int, dtype: str):
    rows = T.dynamic("rows")

    @T.prim_func
    def nan_count(
        logits: T.StridedTensor((rows, vocab), (stride, 1), dtype),
        output: T.Tensor((rows,), "int32"),
    ):
        with T.Kernel(rows, threads=256) as row:
            counts = T.alloc_fragment((8192,), "int32")
            total = T.alloc_fragment((1,), "int32")
            T.clear(counts)
            for block in T.serial(T.ceildiv(vocab, 8192)):
                for col in T.Parallel(8192):
                    index = block * 8192 + col
                    value = T.if_then_else(
                        index < vocab, logits[T.cast(row, "int64"), index], 0
                    )
                    counts[col] += T.cast(T.isnan(T.cast(value, "float32")), "int32")
            T.reduce_sum(counts, total, dim=0)
            output[row] = total[0]

    return nan_count


def count_nans(logits):
    output = torch.empty(logits.shape[0], dtype=torch.int32, device=logits.device)
    if logits.shape[0]:
        nan_count_kernel(logits.shape[1], logits.stride(0), str(logits.dtype)[6:])(
            logits, output
        )
    return output
