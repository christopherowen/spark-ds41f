# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TileLang kernels of the V2 runner's rejection sampler, for production's
verification: one-hot (greedy) drafts, block verification, FP32 draws and no
watermark. Each kernel is the contract of the Triton kernels it replaces in
``rejection_sampler_utils``, bit for bit, NaN and infinite logits included, and
writes the elements they write and no others (the Triton kernels stay importable
as the tests' reference and serve every other configuration):

- ``local_stats``: ``_compute_local_logits_stats_kernel`` without draft logits
- ``verify``: ``_compute_cumulative_log_p_kernel`` without draft logits and
  ``_rejection_kernel`` with block verification, as one kernel
- ``resample``: ``_resample_kernel`` without draft logits or watermark, FP32

``rejection_sample`` inserts the resampled token, as for every configuration
(``tilelang_resampled``).

The arithmetic is Triton 3.7's for sm_121a, read from the kernels' PTX and SASS:
``tl.exp`` is ``ex2.approx.f32`` (subnormals kept) of the product with log2(e),
``tl.log`` libdevice's ``__nv_logf`` as Triton bundles it (``vllm_control_log``),
``/`` is ``div.full.f32``, and every add, product and FMA rounds to nearest as the
header writes it (nvcc would contract them otherwise). The random stream, the Gumbel
noise and the argmax order are ``tilelang_sampling``'s.

A vocabulary block's statistics (8192 columns, four warps) follow Triton's layout
(``_triton_run``): a thread holds 64 elements, sums their exponentials in register
order, then the warp sums at XOR distances 16 to 1 and threads 0-3 at 2 and 1. The
maxima ignore NaN, so any order gives them; the argmax is the Gumbel sampler's.

A row's log-sum-exp over its blocks (``_compute_global_logsumexp``) runs on one warp
in Triton, the blocks padded to a power of two, block ``k`` at lanes ``k`` and
``k + padded`` (and so on), reduced at XOR distances ``padded / 2`` to 1. LLVM fuses
the first level's product into an FMA: lane 0 adds its own product unrounded to its
partner's rounded one, so lanes reach different sums, and lane 0, which stores, is
the contract. Here ``padded`` lanes take a row, the first of them as Triton's lane 0.

Every comparison NaN decides is made in the header: TVM's simplifier rewrites
``a <= b`` as ``!(b < a)``, which a NaN flips.
"""

# No `from __future__ import annotations`: TileLang reads the prim_func
# annotations eagerly, and the symbolic shapes must resolve.
from collections.abc import Sequence

import torch

from vllm.tilelang_utils import T, tilelang
from vllm.v1.worker.gpu.sample import tilelang_controls, tilelang_sampling
from vllm.v1.worker.gpu.sample.tilelang_sampling import (
    _DTYPES,
    _PASS,
    _PASSES,
    _THREADS,
    _WARPS,
    GUMBEL_BLOCK,
    RUNNER_IDX_DTYPE,
    _block_argmax,
    _call,
    _element,
    _exchange,
    _keep_or_take,
    _kernel,
    _load_run,
    _load_run_at,
    _randint_at,
    _runs,
    _triton_run,
    declare,
)

# rejection_sampler_utils._VOCAB_BLOCK_SIZE and _RESAMPLE_BLOCK_SIZE.
STATS_BLOCK = 8192
RESAMPLE_BLOCK = GUMBEL_BLOCK
_STATS_PER_THREAD = STATS_BLOCK // _THREADS
# A row's blocks, padded, are at most one warp of lanes.
_MAX_LANES = 32

_HEADER = (
    # Philox, the Gumbel noise and the argmax combine.
    tilelang_sampling._HEADER
    # vllm_control_log: tl.log, Triton's bundled libdevice __nv_logf.
    + tilelang_controls._HEADER
    + r"""
// tl.exp of an FP32 value: the product with log2(e), then ex2.approx.f32 without
// flush-to-zero (ptxas keeps subnormal results).
__device__ __forceinline__ float vllm_exp(float x) {
  float y;
  asm("{\n"
      ".reg .b32 t;\n"
      "mul.rn.f32 t, %1, 0f3FB8AA3B;\n"
      "ex2.approx.f32 %0, t;\n"
      "}"
      : "=f"(y)
      : "f"(x));
  return y;
}

// FP32 arithmetic rounded to nearest, out of nvcc's contraction and the TileLang
// simplifier's reach.
__device__ __forceinline__ float vllm_add(float a, float b) {
  float c;
  asm("add.rn.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
  return c;
}
__device__ __forceinline__ float vllm_sub(float a, float b) {
  float c;
  asm("sub.rn.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
  return c;
}
__device__ __forceinline__ float vllm_mul(float a, float b) {
  float c;
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
  return c;
}
__device__ __forceinline__ float vllm_fma(float a, float b, float c) {
  float d;
  asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c));
  return d;
}
// max.f32: a NaN gives way to the other operand.
__device__ __forceinline__ float vllm_max(float a, float b) {
  float c;
  asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
  return c;
}

// tl.exp(x - m).
__device__ __forceinline__ float vllm_exp_from(float x, float m) {
  return vllm_exp(vllm_sub(x, m));
}

// A row's log-sum-exp from its maximum and its rescaled sum: m + log(s).
__device__ __forceinline__ float vllm_lse(float m, float s) {
  return vllm_add(m, vllm_control_log(s));
}

// A sampled row's block sum of exponentials, kept while the block's maximum is
// above -inf (a NaN maximum is not), else 0.
__device__ __forceinline__ float vllm_live_sum(float m, float s) {
  float y;
  asm("{\n"
      ".reg .pred live;\n"
      "setp.gt.f32 live, %1, 0fFF800000;\n"
      "selp.f32 %0, %2, 0f00000000, live;\n"
      "}"
      : "=f"(y)
      : "f"(m), "f"(s));
  return y;
}

// A step of the cumulative log ratio: min(log_p + log_prob, 0), min.f32 (a NaN sum
// gives 0).
__device__ __forceinline__ float vllm_log_ratio(float log_p, float log_prob) {
  float y;
  asm("{\n"
      ".reg .b32 t;\n"
      "add.rn.f32 t, %1, %2;\n"
      "min.f32 %0, t, 0f00000000;\n"
      "}"
      : "=f"(y)
      : "f"(log_p), "f"(log_prob));
  return y;
}

// tl_rand32(includes_zero=False) of a Philox word: uint_to_uniform_float, then the
// clamp to _TL_RAND_MIN (0f2FFFFFFF).
__device__ __forceinline__ float vllm_uniform(unsigned int word) {
  float u;
  asm("{\n"
      ".reg .b32 s, x, v;\n"
      "shr.s32 s, %1, 31;\n"
      "xor.b32 x, s, %1;\n"
      "cvt.rn.f32.s32 v, x;\n"
      "mul.rn.f32 v, v, 0f2FFFFFFF;\n"
      "max.f32 %0, v, 0f2FFFFFFF;\n"
      "}"
      : "=f"(u)
      : "r"(word));
  return u;
}

// Block verification's acceptance threshold for a one-hot draft followed by a valid
// one, from the prefix's joint ratio p and the next draft's target probability m:
// the residual mass p * (1 - m) over residual + 1 - p, which Triton contracts to
// fma(p, 1 - m, 1) - p; div.full.f32, and 1 where that is not positive.
__device__ __forceinline__ float vllm_block_threshold(float p, float m) {
  float h;
  asm("{\n"
      ".reg .pred positive;\n"
      ".reg .b32 rest, residual, denom, ratio;\n"
      "sub.rn.f32 rest, 0f3F800000, %2;\n"
      "mul.rn.f32 residual, %1, rest;\n"
      "fma.rn.f32 denom, %1, rest, 0f3F800000;\n"
      "sub.rn.f32 denom, denom, %1;\n"
      "setp.gt.f32 positive, denom, 0f00000000;\n"
      "div.full.f32 ratio, residual, denom;\n"
      "selp.f32 %0, ratio, 0f3F800000, positive;\n"
      "}"
      : "=f"(h)
      : "f"(p), "f"(m));
  return h;
}

// u <= h, ordered: a NaN threshold accepts nothing.
__device__ __forceinline__ int vllm_at_most(float u, float h) { return u <= h; }
"""
)


def _distances(lanes: int) -> tuple[int, ...]:
    """The XOR distances of a butterfly over ``lanes`` lanes, the widest first."""
    return tuple(lanes >> i for i in range(1, lanes.bit_length()))


@T.macro
def _block_reduce(value, partial, tx, combine):
    """Combine every thread's ``value[0]`` across the block in Triton's order into
    thread 0's: lanes at XOR distances 16 to 1, lane 0 of each warp into ``partial``,
    then threads 0-3 at distances 2 and 1 (the other lanes exchange what they hold,
    as in Triton, and their results are unused). ``combine`` is the header's
    ``vllm_max`` or ``vllm_add``, the thread's own value first."""
    for distance in (16, 8, 4, 2, 1):
        value[0] = T.call_extern(
            "float32", combine, value[0], T.shfl_xor(value[0], distance)
        )
    if tx % 32 == 0:
        partial[tx // 32] = value[0]
    T.sync_threads()
    if tx < _WARPS:
        value[0] = partial[tx]
    for distance in (2, 1):
        value[0] = T.call_extern(
            "float32", combine, value[0], T.shfl_xor(value[0], distance)
        )


# --- Vocabulary-block statistics ----------------------------------------------


@tilelang.jit(pass_configs=_PASS)
def local_stats_kernel(logits_dtype: str, idx_dtype: str, step_dtype: str, run: int):
    """``_compute_local_logits_stats_kernel`` without draft logits: one block of 128
    threads per ``(logits row, vocabulary block of 8192)``; rows at a bonus position
    (``expanded_local_pos >= num_speculative_steps``) are left alone. A greedy row
    stores its block's argmax (global, int64) and maximum, a sampled row its block's
    maximum and the sum of ``exp(logit - maximum)`` (0 unless the maximum is above
    -inf). A thread's 64 elements come in runs of ``run`` (``_triton_run``), all
    loaded with the request's state index: Triton loads them once the temperature is
    known, a round trip later."""
    rows, vocab, blocks = T.dynamic("rows, vocab, blocks")
    # Inputs may be longer than the batch: each has its own length.
    idx_len, step_len, temp_len = T.dynamic("idx_len, step_len, temp_len")
    logits_stride, argmax_stride, max_stride, sumexp_stride = T.dynamic(
        "logits_stride, argmax_stride, max_stride, sumexp_stride", "int64"
    )

    @T.prim_func
    def local_stats(
        local_argmax: T.StridedTensor((rows, blocks), (argmax_stride, 1), T.int64),
        local_max: T.StridedTensor((rows, blocks), (max_stride, 1), T.float32),
        local_sumexp: T.StridedTensor((rows, blocks), (sumexp_stride, 1), T.float32),
        logits: T.StridedTensor((rows, vocab), (logits_stride, 1), logits_dtype),
        expanded_idx_mapping: T.Tensor((idx_len,), idx_dtype),
        expanded_local_pos: T.Tensor((step_len,), step_dtype),
        temperature: T.Tensor((temp_len,), T.float32),
        num_speculative_steps: T.int32,
    ):
        with T.Kernel(rows, blocks, threads=_THREADS) as (t, blk):
            T.import_source(_HEADER)
            tx = T.get_thread_binding()
            raw = T.alloc_local((_STATS_PER_THREAD,), logits_dtype)
            values = T.alloc_local((_STATS_PER_THREAD,), T.float32)
            best = T.alloc_local((1,), T.float32)
            best_index = T.alloc_local((1,), T.int32)
            other = T.alloc_local((1,), T.float32)
            other_index = T.alloc_local((1,), T.int32)
            total = T.alloc_local((1,), T.float32)
            partial = T.alloc_shared((_WARPS,), T.float32)
            partial_index = T.alloc_shared((_WARPS,), T.int32)
            partial_sum = T.alloc_shared((_WARPS,), T.float32)
            maximum = T.alloc_shared((1,), T.float32)
            if run > 1:
                # The launch checked it: a 16-byte base and a stride of 16 elements.
                T.assume(logits_stride % 16 == 0)

            # Bonus rows need no statistics.
            if T.cast(expanded_local_pos[t], T.int64) < T.cast(
                num_speculative_steps, T.int64
            ):
                state = T.alloc_var(
                    T.int64, init=T.cast(expanded_idx_mapping[t], T.int64)
                )
                for r in T.unroll(_STATS_PER_THREAD // run):
                    col0 = blk * STATS_BLOCK + r * (_THREADS * run) + run * tx
                    _load_run_at(logits, raw, r * run, t, col0, vocab, run)
                temp = T.alloc_var(T.float32, init=temperature[state])
                # Triton's masked lanes read -inf.
                for j in T.unroll(_STATS_PER_THREAD):
                    values[j] = T.if_then_else(
                        blk * STATS_BLOCK + _element(j, tx, run) < vocab,
                        T.cast(raw[j], T.float32),
                        -T.infinity(T.float32),
                    )
                # The thread's maximum before the temperature's test, so the loads
                # issue ahead of it rather than in its branches.
                best[0] = values[0]
                for j in T.unroll(1, _STATS_PER_THREAD):
                    best[0] = T.call_extern("float32", "vllm_max", best[0], values[j])
                if temp == T.float32(0):
                    best[0] = values[0]
                    best_index[0] = _element(0, tx, run)
                    for j in T.unroll(1, _STATS_PER_THREAD):
                        _keep_or_take(best, best_index, values[j], _element(j, tx, run))
                    _block_argmax(
                        best, best_index, other, other_index, partial, partial_index, tx
                    )
                    if tx == 0:
                        local_argmax[t, blk] = T.cast(
                            blk * STATS_BLOCK + best_index[0], T.int64
                        )
                        local_max[t, blk] = best[0]
                else:
                    _block_reduce(best, partial, tx, "vllm_max")
                    if tx == 0:
                        maximum[0] = best[0]
                    T.sync_threads()
                    best[0] = maximum[0]
                    # The exponentials in register order, the first two added first.
                    total[0] = T.call_extern(
                        "float32", "vllm_exp_from", values[0], best[0]
                    )
                    for j in T.unroll(1, _STATS_PER_THREAD):
                        total[0] = T.call_extern(
                            "float32",
                            "vllm_add",
                            total[0],
                            T.call_extern(
                                "float32", "vllm_exp_from", values[j], best[0]
                            ),
                        )
                    _block_reduce(total, partial_sum, tx, "vllm_add")
                    if tx == 0:
                        local_max[t, blk] = best[0]
                        local_sumexp[t, blk] = T.call_extern(
                            "float32", "vllm_live_sum", best[0], total[0]
                        )

    return local_stats


def local_stats_key(
    logits_dtype: torch.dtype,
    idx_dtype: torch.dtype,
    step_dtype: torch.dtype,
    run: int = 1,
) -> tuple:
    return (_DTYPES[logits_dtype], _DTYPES[idx_dtype], _DTYPES[step_dtype], run)


def compute_local_stats(
    local_argmax: torch.Tensor,  # [num_logits, blocks]
    local_max: torch.Tensor,  # [num_logits, blocks]
    local_sumexp: torch.Tensor,  # [num_logits, blocks]
    logits: torch.Tensor,  # [num_logits, vocab_size]
    expanded_idx_mapping: torch.Tensor,  # [num_logits]
    expanded_local_pos: torch.Tensor,  # [num_logits]
    temperature: torch.Tensor,  # [max_num_reqs]
    num_speculative_steps: int,
) -> None:
    """``_compute_local_logits_stats_kernel``'s launch without draft logits, into
    the given ``[num_logits, cdiv(vocab_size, 8192)]`` statistics."""
    if logits.shape[0] == 0:
        return
    kernel = _kernel(
        local_stats_kernel,
        *local_stats_key(
            logits.dtype,
            expanded_idx_mapping.dtype,
            expanded_local_pos.dtype,
            _triton_run(logits),
        ),
    )
    kernel(
        local_argmax,
        local_max,
        local_sumexp,
        logits,
        expanded_idx_mapping.contiguous(),
        expanded_local_pos.contiguous(),
        temperature,
        num_speculative_steps,
    )


# --- Cumulative ratio and rejection -----------------------------------------


@tilelang.jit(pass_configs=_PASS)
def verify_kernel(
    logits_dtype: str,
    drafts_dtype: str,
    idx_dtype: str,
    pos_dtype: str,
    lanes: int,
    width: int,
):
    """``_compute_cumulative_log_p_kernel`` and ``_rejection_kernel`` (block
    verification, one-hot drafts), a block a request. ``lanes`` is Triton's
    PADDED_VOCAB_NUM_BLOCKS and ``width`` the sampled tokens' (num_speculative_steps
    + 1). Each draft step takes ``lanes`` threads, one a vocabulary block: they load
    its statistics, its draft and the draft's target logit at once, every load at an
    index that exists whatever the masks say, the masks as selects, and reduce them
    to the step's log-sum-exp and the draft's log probability (as Triton's lane 0),
    the greedy step's target argmax, and the step's uniform draw. Thread 0 then runs
    both kernels' loops over the drafts from shared memory, where Triton's take two
    or three dependent round trips to memory a step. The step's cumulative log ratio
    is stored as the cumulative kernel stores it, for sampled requests only; the
    sampled tokens, counts and rejected rows' log-sum-exps as the rejection
    kernel's."""
    steps = width - 1
    if not 1 <= lanes <= _MAX_LANES:
        raise ValueError(f"{lanes} vocabulary blocks: a row's blocks take one warp")
    threads = max(32, -(-max(steps, 1) * lanes // 32) * 32)
    if threads > 1024:
        raise ValueError(f"{steps} draft steps of {lanes} blocks exceed a block")
    groups = threads // lanes
    reqs, num_logits, vocab, blocks = T.dynamic("reqs, num_logits, vocab, blocks")
    drafts_len, idx_len, temp_len, seed_len, pos_len = T.dynamic(
        "drafts_len, idx_len, temp_len, seed_len, pos_len"
    )
    sampled_stride, logits_stride, argmax_stride, max_stride, sumexp_stride = T.dynamic(
        "sampled_stride, logits_stride, argmax_stride, max_stride, sumexp_stride",
        "int64",
    )

    @T.prim_func
    def verify(
        sampled: T.StridedTensor((reqs, width), (sampled_stride, 1), T.int64),
        num_sampled: T.Tensor((reqs,), T.int32),
        target_lse: T.Tensor((reqs,), T.float32),
        draft_lse: T.Tensor((reqs,), T.float32),
        cumulative: T.Tensor((num_logits,), T.float32),
        logits: T.StridedTensor((num_logits, vocab), (logits_stride, 1), logits_dtype),
        local_argmax: T.StridedTensor(
            (num_logits, blocks), (argmax_stride, 1), T.int64
        ),
        local_max: T.StridedTensor((num_logits, blocks), (max_stride, 1), T.float32),
        local_sumexp: T.StridedTensor(
            (num_logits, blocks), (sumexp_stride, 1), T.float32
        ),
        drafts: T.Tensor((drafts_len,), drafts_dtype),
        cu_num_logits: T.Tensor((reqs + 1,), T.int32),
        idx_mapping: T.Tensor((idx_len,), idx_dtype),
        temperature: T.Tensor((temp_len,), T.float32),
        seeds: T.Tensor((seed_len,), T.int64),
        pos: T.Tensor((pos_len,), pos_dtype),
    ):
        with T.Kernel(reqs, threads=threads) as req:
            T.import_source(_HEADER)
            tx = T.get_thread_binding()
            step = tx // lanes
            block = tx % lanes
            row_max = T.alloc_local((1,), T.float32)
            total = T.alloc_local((1,), T.float32)
            best = T.alloc_local((1,), T.float32)
            best_index = T.alloc_local((1,), T.int32)
            other = T.alloc_local((1,), T.float32)
            other_index = T.alloc_local((1,), T.int32)
            lse_s = T.alloc_shared((groups,), T.float32)
            log_prob_s = T.alloc_shared((groups,), T.float32)
            uniform_s = T.alloc_shared((groups,), T.float32)
            draft_s = T.alloc_shared((groups,), T.int64)
            target_s = T.alloc_shared((groups,), T.int64)

            start = T.alloc_var(T.int32, init=cu_num_logits[req])
            end = T.alloc_var(T.int32, init=cu_num_logits[req + 1])
            state = T.alloc_var(T.int64, init=T.cast(idx_mapping[req], T.int64))
            num_drafts = end - start - 1
            # Step s verifies the draft at row s + 1 with row s's logits; the steps
            # past the request's drafts read its last rows and are masked.
            live = step < num_drafts
            row = T.alloc_var(T.int32, init=start + T.min(step, num_drafts))
            draft = T.alloc_var(
                T.int64,
                init=T.cast(drafts[start + T.min(step + 1, num_drafts)], T.int64),
            )
            maximum = T.alloc_var(
                T.float32, init=local_max[row, T.min(block, blocks - 1)]
            )
            sumexp = T.alloc_var(
                T.float32, init=local_sumexp[row, T.min(block, blocks - 1)]
            )
            position = T.alloc_var(pos_dtype, init=pos[row])
            temp = T.alloc_var(T.float32, init=temperature[state])
            seed = T.alloc_var(T.int64, init=seeds[state])
            in_row = live & (block < blocks)
            maximum = T.if_then_else(in_row, maximum, -T.infinity(T.float32))
            sumexp = T.if_then_else(in_row, sumexp, T.float32(0))
            draft = T.if_then_else(live, draft, T.int64(-1))
            logit = T.alloc_var(
                T.float32,
                init=T.cast(logits[row, T.max(draft, T.int64(0))], T.float32),
            )

            # _compute_global_logsumexp: the blocks' maximum, then their sums
            # rescaled to it, the first level's product fused into its add.
            row_max[0] = maximum
            for distance in _distances(lanes):
                row_max[0] = T.call_extern(
                    "float32", "vllm_max", row_max[0], T.shfl_xor(row_max[0], distance)
                )
            scale = T.alloc_var(
                T.float32,
                init=T.call_extern("float32", "vllm_exp_from", maximum, row_max[0]),
            )
            if lanes == 1:
                total[0] = T.call_extern("float32", "vllm_mul", scale, sumexp)
            else:
                total[0] = T.call_extern(
                    "float32",
                    "vllm_fma",
                    scale,
                    sumexp,
                    T.shfl_xor(
                        T.call_extern("float32", "vllm_mul", scale, sumexp), lanes // 2
                    ),
                )
                for distance in _distances(lanes)[1:]:
                    total[0] = T.call_extern(
                        "float32", "vllm_add", total[0], T.shfl_xor(total[0], distance)
                    )
            lse = T.alloc_var(
                T.float32,
                init=T.call_extern("float32", "vllm_lse", row_max[0], total[0]),
            )

            # _compute_global_target_argmax: NaN maxima count as -inf, and the lower
            # block wins a tie.
            best[0] = T.if_then_else(T.isnan(maximum), -T.infinity(T.float32), maximum)
            best_index[0] = block
            for distance in _distances(lanes):
                _exchange(best, best_index, other, other_index, distance)
            target = T.alloc_var(T.int64, init=local_argmax[row, best_index[0]])

            # The step's draw: tl_rand32 at spec_step_seed(seed, step).
            word = _randint_at(
                seed ^ T.shift_left(T.cast(step, T.int64), T.int64(32)),
                position,
                pos_dtype,
                False,
            )
            if block == 0:
                lse_s[step] = lse
                log_prob_s[step] = T.call_extern("float32", "vllm_sub", logit, lse)
                uniform_s[step] = T.call_extern("float32", "vllm_uniform", word)
                draft_s[step] = draft
                target_s[step] = target
            T.sync_threads()

            if tx == 0:
                greedy = T.alloc_var(
                    T.int32, init=T.cast(temp == T.float32(0), T.int32)
                )
                log_p = T.alloc_var(T.float32, init=T.float32(0))
                accepted = T.alloc_var(T.int64, init=T.int64(0))
                verifying = T.alloc_var(T.int32, init=1)
                for i in tuple(range(steps)):
                    if i < num_drafts:
                        drafted = T.alloc_var(T.int64, init=draft_s[i])
                        valid = T.cast(drafted >= 0, T.int32)
                        token = T.max(drafted, T.int64(0))
                        if greedy == 0:
                            # The cumulative ratio, which a -1 placeholder leaves
                            # as it was; a placeholder ends verification.
                            log_p = T.if_then_else(
                                valid != 0,
                                T.call_extern(
                                    "float32", "vllm_log_ratio", log_p, log_prob_s[i]
                                ),
                                log_p,
                            )
                            cumulative[start + i] = log_p
                            verifying = verifying & valid
                        if verifying != 0:
                            if greedy != 0:
                                # Accepted while the draft is the target argmax,
                                # which replaces the first one that is not.
                                hit = T.alloc_var(
                                    T.int32,
                                    init=T.cast(target_s[i] == token, T.int32) & valid,
                                )
                                sampled[req, i] = T.if_then_else(
                                    hit != 0, token, target_s[i]
                                )
                                accepted = accepted + T.cast(hit, T.int64)
                                verifying = hit
                            else:
                                # Block verification: the prefix's joint ratio,
                                # or the threshold of the residual mass when a
                                # valid draft follows (its log probability is the
                                # next step's).
                                threshold = T.alloc_var(
                                    T.float32,
                                    init=T.call_extern("float32", "vllm_exp", log_p),
                                )
                                if i + 1 < steps:  # noqa: SIM102
                                    if draft_s[i + 1] >= 0:
                                        threshold = T.call_extern(
                                            "float32",
                                            "vllm_block_threshold",
                                            threshold,
                                            T.call_extern(
                                                "float32", "vllm_exp", log_prob_s[i + 1]
                                            ),
                                        )
                                accepted = T.if_then_else(
                                    T.call_extern(
                                        "int32",
                                        "vllm_at_most",
                                        uniform_s[i],
                                        threshold,
                                    )
                                    != 0,
                                    T.int64(i + 1),
                                    accepted,
                                )
                                sampled[req, i] = token
                num_sampled[req] = T.cast(accepted, T.int32)
                # The first rejected row's log-sum-exp, which resampling with draft
                # logits reads; 0 for greedy requests and fully accepted blocks.
                target_lse[req] = T.if_then_else(
                    (greedy == 0) & (accepted < T.cast(num_drafts, T.int64)),
                    lse_s[T.min(accepted, T.int64(groups - 1))],
                    T.float32(0),
                )
                draft_lse[req] = T.float32(0)

    return verify


def _padded_blocks(vocab_size: int) -> int:
    """Triton's PADDED_VOCAB_NUM_BLOCKS for ``vocab_size`` columns."""
    blocks = -(-vocab_size // STATS_BLOCK)
    return 1 << max(blocks - 1, 0).bit_length()


def verify_key(
    logits_dtype: torch.dtype,
    drafts_dtype: torch.dtype,
    idx_dtype: torch.dtype,
    pos_dtype: torch.dtype,
    vocab_size: int,
    num_speculative_steps: int,
) -> tuple:
    return (
        _DTYPES[logits_dtype],
        _DTYPES[drafts_dtype],
        _DTYPES[idx_dtype],
        _DTYPES[pos_dtype],
        _padded_blocks(vocab_size),
        num_speculative_steps + 1,
    )


def verify(
    sampled: torch.Tensor,  # [num_reqs, num_speculative_steps + 1]
    num_sampled: torch.Tensor,  # [num_reqs]
    target_rejected_logsumexp: torch.Tensor,  # [num_reqs]
    draft_rejected_logsumexp: torch.Tensor,  # [num_reqs]
    cumulative_log_p: torch.Tensor,  # [num_logits]
    target_logits: torch.Tensor,  # [num_logits, vocab_size]
    target_local_argmax: torch.Tensor,  # [num_logits, blocks]
    target_local_max: torch.Tensor,  # [num_logits, blocks]
    target_local_sumexp: torch.Tensor,  # [num_logits, blocks]
    draft_sampled: torch.Tensor,  # [num_logits]
    cu_num_logits: torch.Tensor,  # [num_reqs + 1]
    idx_mapping: torch.Tensor,  # [num_reqs]
    temperature: torch.Tensor,  # [max_num_reqs]
    seed: torch.Tensor,  # [max_num_reqs]
    pos: torch.Tensor,  # [num_logits]
    num_speculative_steps: int,
) -> None:
    """The launches of ``_compute_cumulative_log_p_kernel`` and
    ``_rejection_kernel`` (block verification, one-hot drafts) as one, into the
    given outputs."""
    if num_sampled.shape[0] == 0:
        return
    kernel = _kernel(
        verify_kernel,
        *verify_key(
            target_logits.dtype,
            draft_sampled.dtype,
            idx_mapping.dtype,
            pos.dtype,
            target_logits.shape[1],
            num_speculative_steps,
        ),
    )
    kernel(
        sampled,
        num_sampled,
        target_rejected_logsumexp,
        draft_rejected_logsumexp,
        cumulative_log_p,
        target_logits,
        target_local_argmax,
        target_local_max,
        target_local_sumexp,
        draft_sampled.contiguous(),
        cu_num_logits,
        idx_mapping.contiguous(),
        temperature,
        seed,
        pos.contiguous(),
    )


# --- Resampling ---------------------------------------------------------------


@tilelang.jit(pass_configs=_PASS)
def resample_kernel(
    logits_dtype: str, idx_dtype: str, drafts_dtype: str, pos_dtype: str, run: int
):
    """``_resample_kernel`` without draft logits or watermark, FP32: one block of
    128 threads per ``(request, vocabulary block of 1024)``, a thread's elements in
    runs of ``run`` (``_triton_run``). The request's first unaccepted row is
    resampled: the target logits with a rejected draft masked to -inf (the residual of
    a one-hot draft; a bonus row and a -1 placeholder mask nothing), plus the draw's
    Gumbel noise unless the temperature is 0; each block's argmax index (global,
    int64) and score. A greedy request's rejected row is left alone: verification
    stored its target argmax."""
    reqs, num_logits, vocab, blocks = T.dynamic("reqs, num_logits, vocab, blocks")
    # Inputs may be longer than the batch: each has its own length.
    idx_len, drafts_len, temp_len, seed_len, pos_len = T.dynamic(
        "idx_len, drafts_len, temp_len, seed_len, pos_len"
    )
    logits_stride, argmax_stride, max_stride = T.dynamic(
        "logits_stride, argmax_stride, max_stride", "int64"
    )

    @T.macro
    def passes(logits, raw, score, best, best_index, row, rejected, word_seed, blk,
               tx, noisy):  # fmt: skip
        """The thread's residual scores, combined into ``(best, best_index)``; with
        ``noisy`` each element's Philox word is drawn ahead of its bound check (the
        Gumbel sampler's ``passes``)."""
        for r in T.unroll(_PASSES // run):
            col0 = blk * RESAMPLE_BLOCK + r * (_THREADS * run) + run * tx
            _load_run(logits, raw, row, col0, vocab, run)
            for k in T.unroll(run):
                local = r * (_THREADS * run) + run * tx + k
                index = col0 + k
                if noisy:
                    word = T.alloc_var(
                        T.uint32,
                        init=T.call_extern(
                            "uint32", "vllm_randint32", word_seed[0], index
                        ),
                    )
                score[0] = -T.infinity(T.float32)
                if index < vocab:
                    score[0] = T.if_then_else(
                        T.cast(index, T.int64) == rejected[0],
                        -T.infinity(T.float32),
                        T.cast(raw[k], T.float32),
                    )
                    if noisy:
                        score[0] = T.call_extern(
                            "float32", "vllm_gumbel_score", score[0], word
                        )
                if r * run + k == 0:
                    best[0] = score[0]
                    best_index[0] = local
                else:
                    _keep_or_take(best, best_index, score[0], local)

    @T.prim_func
    def resample(
        local_argmax: T.StridedTensor((reqs, blocks), (argmax_stride, 1), T.int64),
        local_max: T.StridedTensor((reqs, blocks), (max_stride, 1), T.float32),
        logits: T.StridedTensor((num_logits, vocab), (logits_stride, 1), logits_dtype),
        num_sampled: T.Tensor((reqs,), T.int32),
        cu_num_logits: T.Tensor((reqs + 1,), T.int32),
        expanded_idx_mapping: T.Tensor((idx_len,), idx_dtype),
        drafts: T.Tensor((drafts_len,), drafts_dtype),
        temperature: T.Tensor((temp_len,), T.float32),
        seeds: T.Tensor((seed_len,), T.int64),
        pos: T.Tensor((pos_len,), pos_dtype),
    ):
        with T.Kernel(reqs, blocks, threads=_THREADS) as (req, blk):
            T.import_source(_HEADER)
            tx = T.get_thread_binding()
            raw = T.alloc_local((run,), logits_dtype)
            score = T.alloc_local((1,), T.float32)
            best = T.alloc_local((1,), T.float32)
            best_index = T.alloc_local((1,), T.int32)
            other = T.alloc_local((1,), T.float32)
            other_index = T.alloc_local((1,), T.int32)
            rejected = T.alloc_local((1,), T.int64)
            word_seed = T.alloc_local((1,), T.uint32)
            partial = T.alloc_shared((_WARPS,), T.float32)
            partial_index = T.alloc_shared((_WARPS,), T.int32)
            if run > 1:
                # The launch checked it: a 16-byte base and a stride of 16 elements.
                T.assume(logits_stride % 16 == 0)

            end = T.alloc_var(T.int32, init=cu_num_logits[req + 1])
            row = T.alloc_var(
                T.int64,
                init=T.cast(cu_num_logits[req], T.int64)
                + T.cast(num_sampled[req], T.int64),
            )
            state = T.alloc_var(
                T.int64, init=T.cast(expanded_idx_mapping[row], T.int64)
            )
            # As Triton's, the test's temperature is read without a validity check.
            temp = T.alloc_var(T.float32, init=temperature[state])
            bonus = row == T.cast(end - 1, T.int64)
            if (temp != T.float32(0)) | bonus:
                # The rejected draft is the next row's input; a bonus row has none.
                draft = T.alloc_var(
                    T.int64,
                    init=T.cast(
                        drafts[T.min(row + 1, T.cast(end - 1, T.int64))], T.int64
                    ),
                )
                rejected[0] = T.if_then_else(bonus | (draft < 0), T.int64(-1), draft)
                # gumbel_block_argmax: a padded row draws with seed 0 and no noise.
                seed = T.alloc_var(
                    T.int64,
                    init=T.if_then_else(
                        state >= 0, seeds[T.max(state, T.int64(0))], T.int64(0)
                    ),
                )
                word_seed[0] = _randint_at(seed, pos[row], pos_dtype, False)
                if T.if_then_else(state >= 0, temp, T.float32(0)) != T.float32(0):
                    passes(logits, raw, score, best, best_index, row, rejected,
                           word_seed, blk, tx, True)  # fmt: skip
                else:
                    passes(logits, raw, score, best, best_index, row, rejected,
                           word_seed, blk, tx, False)  # fmt: skip
                _block_argmax(
                    best, best_index, other, other_index, partial, partial_index, tx
                )
                if tx == 0:
                    local_argmax[req, blk] = T.cast(
                        blk * RESAMPLE_BLOCK + best_index[0], T.int64
                    )
                    local_max[req, blk] = best[0]

    return resample


def resample_key(
    logits_dtype: torch.dtype,
    idx_dtype: torch.dtype,
    drafts_dtype: torch.dtype,
    pos_dtype: torch.dtype,
    run: int = 1,
) -> tuple:
    return (
        _DTYPES[logits_dtype],
        _DTYPES[idx_dtype],
        _DTYPES[drafts_dtype],
        _DTYPES[pos_dtype],
        run,
    )


def resample(
    resampled_local_argmax: torch.Tensor,  # [num_reqs, blocks]
    resampled_local_max: torch.Tensor,  # [num_reqs, blocks]
    target_logits: torch.Tensor,  # [num_logits, vocab_size]
    num_sampled: torch.Tensor,  # [num_reqs]
    cu_num_logits: torch.Tensor,  # [num_reqs + 1]
    expanded_idx_mapping: torch.Tensor,  # [num_logits]
    draft_sampled: torch.Tensor,  # [num_logits]
    temperature: torch.Tensor,  # [max_num_reqs]
    seed: torch.Tensor,  # [max_num_reqs]
    pos: torch.Tensor,  # [num_logits]
) -> None:
    """``_resample_kernel``'s launch without draft logits or watermark (FP32), into
    the given ``[num_reqs, cdiv(vocab_size, 1024)]`` block winners."""
    if num_sampled.shape[0] == 0:
        return
    kernel = _kernel(
        resample_kernel,
        *resample_key(
            target_logits.dtype,
            expanded_idx_mapping.dtype,
            draft_sampled.dtype,
            pos.dtype,
            _triton_run(target_logits),
        ),
    )
    kernel(
        resampled_local_argmax,
        resampled_local_max,
        target_logits,
        num_sampled,
        cu_num_logits,
        expanded_idx_mapping.contiguous(),
        draft_sampled.contiguous(),
        temperature,
        seed,
        pos.contiguous(),
    )


# --- rejection_sample ---------------------------------------------------------


def verify_and_resample(
    # [num_logits, V]
    target_logits: torch.Tensor,
    # [num_logits]
    draft_sampled: torch.Tensor,
    # [num_reqs + 1]
    cu_num_logits: torch.Tensor,
    # [num_logits]
    pos: torch.Tensor,
    # [num_reqs]
    idx_mapping: torch.Tensor,
    # [num_logits]
    expanded_idx_mapping: torch.Tensor,
    # [num_logits]
    expanded_local_pos: torch.Tensor,
    # [max_num_reqs]
    temperature: torch.Tensor,
    # [max_num_reqs]
    seed: torch.Tensor,
    num_speculative_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``rejection_sample``'s launches before the insertion, with one-hot drafts,
    block verification, FP32 draws and no watermark: the sampled tokens
    ([num_reqs, num_speculative_steps + 1], int64) up to the first rejected or bonus
    token, their counts ([num_reqs], int32), and the block winners of that token's
    resampling (``[num_reqs, cdiv(V, 1024)]`` indices, int64, and scores, FP32)."""
    num_reqs = cu_num_logits.shape[0] - 1
    num_logits, vocab_size = target_logits.shape

    # The per-vocab-block logits stats: target argmax for greedy requests, target
    # max and summed exponentials for the others.
    blocks = -(-vocab_size // STATS_BLOCK)
    target_local_argmax = target_logits.new_empty(num_logits, blocks, dtype=torch.int64)
    target_local_max = target_logits.new_empty(num_logits, blocks, dtype=torch.float32)
    target_local_sumexp = target_logits.new_empty(
        num_logits, blocks, dtype=torch.float32
    )
    compute_local_stats(
        target_local_argmax,
        target_local_max,
        target_local_sumexp,
        target_logits,
        expanded_idx_mapping,
        expanded_local_pos,
        temperature,
        num_speculative_steps,
    )

    # Verification up to the first rejected or bonus token.
    cumulative_log_p = target_logits.new_empty(num_logits, dtype=torch.float32)
    sampled = draft_sampled.new_empty(
        num_reqs, num_speculative_steps + 1, dtype=torch.int64
    )
    num_sampled = sampled.new_empty(num_reqs, dtype=torch.int32)
    target_rejected_logsumexp = target_logits.new_empty(num_reqs, dtype=torch.float32)
    draft_rejected_logsumexp = target_logits.new_empty(num_reqs, dtype=torch.float32)
    verify(
        sampled,
        num_sampled,
        target_rejected_logsumexp,
        draft_rejected_logsumexp,
        cumulative_log_p,
        target_logits,
        target_local_argmax,
        target_local_max,
        target_local_sumexp,
        draft_sampled,
        cu_num_logits,
        idx_mapping,
        temperature,
        seed,
        pos,
        num_speculative_steps,
    )

    # Resample the rejected/bonus tokens.
    resample_blocks = -(-vocab_size // RESAMPLE_BLOCK)
    resampled_local_argmax = target_logits.new_empty(
        num_reqs, resample_blocks, dtype=torch.int64
    )
    resampled_local_max = target_logits.new_empty(
        num_reqs, resample_blocks, dtype=torch.float32
    )
    resample(
        resampled_local_argmax,
        resampled_local_max,
        target_logits,
        num_sampled,
        cu_num_logits,
        expanded_idx_mapping,
        draft_sampled,
        temperature,
        seed,
        pos,
    )
    return sampled, num_sampled, resampled_local_argmax, resampled_local_max


# --- Preparation --------------------------------------------------------------


def rejection_calls(
    logits_dtypes: Sequence[torch.dtype], vocab_size: int, num_speculative_steps: int
) -> list:
    """``verify_and_resample``'s calls for ``vocab_size``-wide target logits of
    each of ``logits_dtypes``, in both layouts a launch may take (``_runs``), with
    the input batch's int32 drafts and local positions and its int64 indices and
    positions."""
    calls = []
    for dtype in dict.fromkeys(logits_dtypes):
        for run in _runs(dtype):
            calls.append(
                _call(
                    local_stats_kernel,
                    *local_stats_key(dtype, RUNNER_IDX_DTYPE, torch.int32, run),
                )
            )
            calls.append(
                _call(
                    resample_kernel,
                    *resample_key(
                        dtype, RUNNER_IDX_DTYPE, torch.int32, torch.int64, run
                    ),
                )
            )
        calls.append(
            _call(
                verify_kernel,
                *verify_key(
                    dtype,
                    torch.int32,
                    RUNNER_IDX_DTYPE,
                    torch.int64,
                    vocab_size,
                    num_speculative_steps,
                ),
            )
        )
    return calls


def register_block_verification(
    logits_dtypes: Sequence[torch.dtype], vocab_size: int, num_speculative_steps: int
) -> None:
    """Declare ``rejection_calls`` with the runner's JIT registry (a no-op where
    none collects)."""
    declare(lambda: rejection_calls(logits_dtypes, vocab_size, num_speculative_steps))
