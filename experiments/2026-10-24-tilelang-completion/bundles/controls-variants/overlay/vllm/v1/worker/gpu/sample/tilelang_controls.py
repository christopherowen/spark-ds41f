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
// Triton's bundled libdevice __nv_logf, as used by the sampling family.
// CUDA's logf and lg2.approx * ln(2) can move a min-p threshold by one ULP.
__device__ __forceinline__ float vllm_control_log(float x) {
  float result;
  asm("{\n"
      ".reg .pred tiny, fin, zero;\n"
      ".reg .b32 z, o, e, m, f, r, p, q, l, w;\n"
      "setp.lt.f32 tiny, %1, 0f00800000;\n"
      "mul.rn.f32 z, %1, 0f4B000000;\n"
      "selp.f32 z, z, %1, tiny;\n"
      "selp.f32 o, 0fC1B80000, 0f00000000, tiny;\n"
      "add.s32 e, z, -1059760811;\n"
      "and.b32 e, e, -8388608;\n"
      "sub.s32 m, z, e;\n"
      "cvt.rn.f32.s32 f, e;\n"
      "fma.rn.ftz.f32 f, f, 0f34000000, o;\n"
      "add.rn.f32 r, m, 0fBF800000;\n"
      "fma.rn.ftz.f32 p, 0fBE055027, r, 0f3E1039F6;\n"
      "fma.rn.ftz.f32 p, p, r, 0fBDF8CDCC;\n"
      "fma.rn.ftz.f32 p, p, r, 0f3E0F2955;\n"
      "fma.rn.ftz.f32 p, p, r, 0fBE2AD8B9;\n"
      "fma.rn.ftz.f32 p, p, r, 0f3E4CED0B;\n"
      "fma.rn.ftz.f32 p, p, r, 0fBE7FFF22;\n"
      "fma.rn.ftz.f32 p, p, r, 0f3EAAAA78;\n"
      "fma.rn.ftz.f32 p, p, r, 0fBF000000;\n"
      "mul.rn.f32 q, r, p;\n"
      "fma.rn.ftz.f32 q, q, r, r;\n"
      "fma.rn.ftz.f32 l, f, 0f3F317218, q;\n"
      // Preserve the bundled libdevice exceptional-value handling.
      "setp.lt.u32 fin, z, 2139095040;\n"
      "fma.rn.ftz.f32 w, z, 0f7F800000, 0f7F800000;\n"
      "selp.f32 l, l, w, fin;\n"
      "setp.eq.f32 zero, z, 0f00000000;\n"
      "selp.f32 %0, 0fFF800000, l, zero;\n"
      "}"
      : "=f"(result) : "f"(x));
  return result;
}
__device__ __forceinline__ void vllm_control_store_if(void* p, float v, bool ok) {
  asm volatile("{\n .reg .pred q;\n setp.ne.b32 q, %2, 0;\n"
               " @q st.global.f32 [%0], %1;\n}"
               :: "l"(p), "f"(v), "r"((int)ok) : "memory");
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


def _min_p_column(start, j, tx, tile, threads, run):
    """Column of thread ``tx``'s ``j``-th element in tile ``start``: runs of ``run``
    consecutive columns, ``threads * run`` apart (Triton's layout for its loads)."""
    return start * tile + (j // run) * (threads * run) + run * tx + j % run


@tilelang.jit(pass_configs=_PASS)
def min_p_kernel(
    vocab: int,
    stride: int,
    index_dtype: str,
    run: int,
    threads: int = 128,
    tile: int = 1024,
):
    """One row a block, in tiles of ``tile`` columns. Each tile's loads are issued
    before its values are used: with ``run`` 4 (16-byte aligned rows, the launch
    checked) as vectors, and in the last, partial tile at columns clamped to the row
    (a repeated column does not change the maximum; its stores are predicated)."""
    rows, states = T.dynamic("rows, states")
    per_thread = tile // threads
    full = vocab // tile
    warps = threads // 32

    @T.macro
    def load_tile(values, src, row, start, tx):
        for r in T.unroll(per_thread // run):
            col0 = _min_p_column(start, r * run, tx, tile, threads, run)
            for k in T.vectorized(run):
                values[r * run + k] = src[row, col0 + k]

    @T.prim_func
    def min_p(
        logits: T.StridedTensor((rows, vocab), (stride, 1), "float32"),
        mapping: T.Tensor((rows,), index_dtype),
        thresholds: T.Tensor((states,), "float32"),
    ):
        with T.Kernel(rows, threads=threads) as block:
            T.import_source(_HEADER)
            tx = T.get_thread_binding()
            row = T.cast(block, "int64")
            values = T.alloc_local((per_thread,), "float32")
            best = T.alloc_local((1,), "float32")
            minimum = T.alloc_local((1,), "float32")
            partial = T.alloc_shared((warps,), "float32")
            minimum[0] = thresholds[mapping[block]]
            if minimum[0] != 0:
                best[0] = -T.infinity("float32")
                for start in T.serial(full):
                    load_tile(values, logits, row, start, tx)
                    for j in T.unroll(per_thread):
                        best[0] = T.max(best[0], values[j])
                if vocab % tile:
                    for j in T.unroll(per_thread):
                        col = _min_p_column(full, j, tx, tile, threads, run)
                        values[j] = logits[row, T.min(col, vocab - 1)]
                    for j in T.unroll(per_thread):
                        best[0] = T.max(best[0], values[j])
                # The maximum ignores isolated NaNs, as Triton's does.
                best[0] = T.max(best[0], T.shfl_xor(best[0], 16))
                best[0] = T.max(best[0], T.shfl_xor(best[0], 8))
                best[0] = T.max(best[0], T.shfl_xor(best[0], 4))
                best[0] = T.max(best[0], T.shfl_xor(best[0], 2))
                best[0] = T.max(best[0], T.shfl_xor(best[0], 1))
                if tx % 32 == 0:
                    partial[tx // 32] = best[0]
                T.sync_threads()
                best[0] = partial[0]
                for w in T.unroll(1, warps):
                    best[0] = T.max(best[0], partial[w])
                minimum[0] = best[0] + T.call_extern(
                    "float32", "vllm_control_log", minimum[0]
                )
                for start in T.serial(full):
                    load_tile(values, logits, row, start, tx)
                    for r in T.unroll(per_thread // run):
                        col0 = _min_p_column(start, r * run, tx, tile, threads, run)
                        for k in T.vectorized(run):
                            logits[row, col0 + k] = T.if_then_else(
                                values[r * run + k] < minimum[0],
                                -T.infinity("float32"),
                                values[r * run + k],
                            )
                if vocab % tile:
                    for j in T.unroll(per_thread):
                        col = _min_p_column(full, j, tx, tile, threads, run)
                        values[j] = logits[row, T.min(col, vocab - 1)]
                    for j in T.unroll(per_thread):
                        col = _min_p_column(full, j, tx, tile, threads, run)
                        T.call_extern(
                            "handle",
                            "vllm_control_store_if",
                            T.address_of(logits[row, col]),
                            T.if_then_else(
                                values[j] < minimum[0],
                                -T.infinity("float32"),
                                values[j],
                            ),
                            col < vocab,
                        )

    return min_p


def _min_p_run(logits) -> int:
    """4 when every row starts 16-byte aligned, else 1."""
    return 4 if logits.data_ptr() % 16 == 0 and logits.stride(0) % 4 == 0 else 1


def register_min_p(*, logits_stride: int, vocab_size: int):
    _declare(
        min_p_kernel,
        [
            (vocab_size, logits_stride, dtype, run)
            for dtype in _DTYPES.values()
            for run in (1, 4)
        ],
    )


def apply_min_p(logits, mapping, thresholds):
    if logits.shape[0]:
        _launch(
            min_p_kernel,
            (
                logits.shape[1],
                logits.stride(0),
                _DTYPES[mapping.dtype],
                _min_p_run(logits),
            ),
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
    block = 2048

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


def accumulate_bincount(
    mapping, tokens, prompt, prefill, mask, counts, max_prefill_len
):
    """Kernel portion only; the owner still clears selected rows before accumulation."""
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
