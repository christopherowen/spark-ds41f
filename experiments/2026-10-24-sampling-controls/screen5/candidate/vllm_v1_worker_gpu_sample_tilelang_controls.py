# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-dependent logits processors with the V2 sampler's tensor layouts."""

import torch

from vllm.tilelang_utils import T, tilelang

_PASS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
}
_DTYPES = {torch.int32: "int32", torch.int64: "int64"}
_HEADER = r"""
__device__ __forceinline__ float vllm_control_log(float x) {
  float y;
  asm("lg2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return __fmul_rn(y, 0.693147182464599609375f);
}
__device__ __forceinline__ float vllm_control_reciprocal(float x) {
  float y;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float vllm_control_penalty(float x, float scale,
    float freq, int count, float presence) {
  x = __fmul_rn(x, scale);
  x = __fmaf_rn(-freq, (float)count, x);
  return __fmaf_rn(-presence, (float)(count > 0), x);
}
"""


def _launch(factory, key, *tensors):
    factory(*key)(*tensors)


def _declare(factory, keys):
    from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry

    if JitWarmupRegistry.active() is None:
        return
    from vllm.models.deepseek_v4_1.tilelang.prepare import call, declare

    declare(lambda: [call(factory, *key) for key in keys])


@tilelang.jit(pass_configs=_PASS)
def min_p_kernel(vocab: int, stride: int, index_dtype: str, threads: int = 128):
    rows, states = T.dynamic("rows, states")
    block = 1024

    @T.prim_func
    def min_p(
        logits: T.StridedTensor((rows, vocab), (stride, 1), "float32"),
        mapping: T.Tensor((rows,), index_dtype),
        thresholds: T.Tensor((states,), "float32"),
    ):
        with T.Kernel(rows, threads=threads) as row:
            T.import_source(_HEADER)
            maximum = T.alloc_fragment((block,), "float32")
            reduced = T.alloc_fragment((1,), "float32")
            value = T.alloc_local((1,), "float32")
            minimum = T.alloc_local((1,), "float32")
            minimum[0] = thresholds[mapping[row]]
            if minimum[0] != 0:
                T.fill(maximum, -T.infinity("float32"))
                for start in T.serial(T.ceildiv(vocab, block)):
                    for col in T.Parallel(block):
                        index = start * block + col
                        value[0] = -T.infinity("float32")
                        if index < vocab:
                            value[0] = logits[T.cast(row, "int64"), index]
                        maximum[col] = T.max(maximum[col], value[0])
                T.reduce_max(maximum, reduced, dim=0)
                minimum[0] = reduced[0] + T.call_extern(
                    "float32", "vllm_control_log", minimum[0]
                )
                # The maximum reduction ignores isolated NaNs, as Triton does.
                for start in T.serial(T.ceildiv(vocab, block)):
                    for col in T.Parallel(block):
                        index = start * block + col
                        if index < vocab:
                            value[0] = logits[T.cast(row, "int64"), index]
                            logits[T.cast(row, "int64"), index] = T.if_then_else(
                                value[0] < minimum[0],
                                -T.infinity("float32"),
                                value[0],
                            )

    return min_p


def register_min_p(*, logits_stride: int, vocab_size: int):
    _declare(
        min_p_kernel,
        [(vocab_size, logits_stride, dtype) for dtype in _DTYPES.values()],
    )


def apply_min_p(logits, mapping, thresholds):
    if logits.shape[0]:
        _launch(
            min_p_kernel,
            (logits.shape[1], logits.stride(0), _DTYPES[mapping.dtype]),
            logits,
            mapping,
            thresholds,
        )


@tilelang.jit(
    pass_configs={**_PASS, tilelang.PassConfigKey.TL_DISABLE_VECTORIZE_256: True}
)
def penalties_kernel(
    vocab: int,
    logits_stride: int,
    mask_stride: int,
    counts_stride: int,
    index_dtype: str,
    threads: int = 256,
):
    rows, states, mask_rows, count_rows = T.dynamic(
        "rows, states, mask_rows, count_rows"
    )
    block = 8192

    @T.prim_func
    def penalties(
        logits: T.StridedTensor((rows, vocab), (logits_stride, 1), "float32"),
        mapping: T.Tensor((rows,), index_dtype),
        token_ids: T.Tensor((rows,), "int32"),
        local_pos: T.Tensor((rows,), "int32"),
        repetition: T.Tensor((states,), "float32"),
        frequency: T.Tensor((states,), "float32"),
        presence: T.Tensor((states,), "float32"),
        prompt_mask: T.StridedTensor(
            (mask_rows, T.ceildiv(vocab, 32)), (mask_stride, 1), "int32"
        ),
        output_counts: T.StridedTensor(
            (count_rows, vocab), (counts_stride, 1), "int32"
        ),
    ):
        with T.Kernel(rows, T.ceildiv(vocab, block), threads=threads) as (row, blk):
            T.import_source(_HEADER)
            req = T.alloc_local((1,), "int64")
            rep = T.alloc_local((1,), "float32")
            freq = T.alloc_local((1,), "float32")
            pres = T.alloc_local((1,), "float32")
            scale = T.alloc_local((1,), "float32")
            values = T.alloc_fragment((block,), "float32")
            counts = T.alloc_fragment((block,), "int32")
            req[0] = mapping[row]
            rep[0] = repetition[req[0]]
            freq[0] = frequency[req[0]]
            pres[0] = presence[req[0]]
            if (rep[0] != 1) | (freq[0] != 0) | (pres[0] != 0):
                for col in T.Parallel(block):
                    index = blk * block + col
                    values[col] = T.if_then_else(
                        index < vocab, logits[T.cast(row, "int64"), index], 0
                    )
                    counts[col] = T.if_then_else(
                        index < vocab, output_counts[req[0], index], 0
                    )
                for prev in T.serial(local_pos[row]):
                    for col in T.Parallel(block):
                        counts[col] += T.cast(
                            blk * block + col
                            == token_ids[row - local_pos[row] + prev + 1],
                            "int32",
                        )
                for col in T.Parallel(block):
                    index = blk * block + col
                    if index < vocab:
                        scale[0] = 1
                        if rep[0] != 1:
                            appeared = (
                                (prompt_mask[req[0], index // 32] >> (index % 32)) & 1
                            ) | T.cast(counts[col] > 0, "int32")
                            scale[0] = T.if_then_else(appeared != 0, rep[0], 1)
                            if values[col] > 0:
                                scale[0] = T.call_extern(
                                    "float32", "vllm_control_reciprocal", scale[0]
                                )
                        logits[T.cast(row, "int64"), index] = T.call_extern(
                            "float32",
                            "vllm_control_penalty",
                            values[col],
                            scale[0],
                            freq[0],
                            counts[col],
                            pres[0],
                        )

    return penalties


def register_penalties(
    *,
    logits_stride: int,
    prompt_bin_mask_stride: int,
    output_bin_counts_stride: int,
    vocab_size: int,
):
    _declare(
        penalties_kernel,
        [
            (
                vocab_size,
                logits_stride,
                prompt_bin_mask_stride,
                output_bin_counts_stride,
                dtype,
            )
            for dtype in _DTYPES.values()
        ],
    )


def apply_penalties(logits, mapping, tokens, positions, rep, freq, pres, mask, counts):
    if logits.shape[0]:
        _launch(
            penalties_kernel,
            (
                logits.shape[1],
                logits.stride(0),
                mask.stride(0),
                counts.stride(0),
                _DTYPES[mapping.dtype],
            ),
            logits,
            mapping,
            tokens,
            positions,
            rep,
            freq,
            pres,
            mask,
            counts,
        )


@tilelang.jit(pass_configs=_PASS)
def bincount_kernel(
    tokens_stride: int,
    mask_stride: int,
    counts_stride: int,
    index_dtype: str,
    threads: int = 128,
):
    rows, states, columns, mask_cols, counts_cols = T.dynamic(
        "rows, states, columns, mask_cols, counts_cols"
    )

    @T.prim_func
    def bincount(
        mapping: T.Tensor((rows,), index_dtype),
        tokens: T.StridedTensor((states, columns), (tokens_stride, 1), "int32"),
        prompt_len: T.Tensor((states,), "int32"),
        prefill_len: T.Tensor((states,), "int32"),
        mask: T.StridedTensor((states, mask_cols), (mask_stride, 1), "int32"),
        counts: T.StridedTensor((states, counts_cols), (counts_stride, 1), "int32"),
        num_blocks: T.int32,
    ):
        with T.Kernel(rows, num_blocks, threads=threads) as (row, blk):
            req = T.alloc_local((1,), "int64")
            token = T.alloc_local((1,), "int32")
            req[0] = mapping[row]
            for i in T.Parallel(1024):
                pos = blk * 1024 + i
                if pos < prefill_len[req[0]]:
                    token[0] = tokens[req[0], pos]
                    if pos < prompt_len[req[0]]:
                        T.atomic_or(mask[req[0], token[0] // 32], 1 << (token[0] % 32))
                    else:
                        T.atomic_add(counts[req[0], token[0]], 1)

    return bincount


def register_bincount(
    *,
    all_token_ids_stride: int,
    prompt_bin_mask_stride: int,
    output_bin_counts_stride: int,
):
    _declare(
        bincount_kernel,
        [
            (
                all_token_ids_stride,
                prompt_bin_mask_stride,
                output_bin_counts_stride,
                dtype,
            )
            for dtype in _DTYPES.values()
        ],
    )


def bincount(mapping, tokens, prompt, prefill, mask, counts, max_prefill_len):
    if mapping.numel() and max_prefill_len:
        _launch(
            bincount_kernel,
            (
                tokens.stride(0),
                mask.stride(0),
                counts.stride(0),
                _DTYPES[mapping.dtype],
            ),
            mapping,
            tokens,
            prompt,
            prefill,
            mask,
            counts,
            (max_prefill_len + 1023) // 1024,
        )
