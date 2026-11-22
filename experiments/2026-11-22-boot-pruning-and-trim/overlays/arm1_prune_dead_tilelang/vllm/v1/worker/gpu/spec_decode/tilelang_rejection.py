# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TileLang kernels of the V2 runner's rejection sampler, for production's
verification: one-hot (greedy) drafts, block verification, FP32 draws and no
watermark. Each kernel replaces Triton kernels of ``rejection_sampler_utils`` and
writes the elements they write and no others (the Triton kernels stay importable
as the tests' baseline and serve every other configuration):

- ``local_stats``: ``_compute_local_logits_stats_kernel`` without draft logits
- ``verify``: ``_compute_cumulative_log_p_kernel`` without draft logits and
  ``_rejection_kernel`` with block verification, as one kernel
- ``resample``: ``_resample_kernel`` without draft logits or watermark, FP32

``rejection_sample`` inserts the resampled token, as for every configuration
(``tilelang_resampled``).

With a dead-row cut (adaptive verification's dead rows: the drafts past a request's
cut were verification rows that skipped the routed MoE, so their logits are not the
target's), ``verify`` and ``resample`` end each request's block at its cut: its
drafts from the cut on are -1 placeholders, which end a block as the drafter's own
do. The last live draft takes the last draft's threshold (the joint ratio), no row
past the cut weighs in any threshold or ratio, and the cut's row draws the bonus
token: the outcome is block verification's of the truncated draft, bit for bit.

The argmaxes, maxima, uniform draws, greedy verification and resampling are the
Triton kernels' bit for bit, NaN and infinite logits included. The probabilities
follow Barron, Gnanapragasam and Tschannen's fix ("A Fix for Softmax and
Cross-Entropy", Eq. 5), which Triton's arithmetic lacks. With M a row's maximum and
S the sum of exp(z - M) over its other columns, the log-softmax is the shifted
logit z - M (exactly 0 at the maximum) minus log1p(S). Triton sums every column, the
maximum's exp(0) = 1 among them, and subtracts M + log(sum) from the logit: the sum
rounds to 1 once S is below half an FP32 ULP of 1 (a margin past 16.6 nats), every
block added after the maximum's loses what it holds below that half ULP, and
M + log(sum) rounds at the magnitude of M (an ULP of 32 is 3.8e-6), so a confident
draft's log probability came out 0 or a few ULPs of M off. Block verification's
threshold then took 1 - m and 1 - p of probabilities near 1, which cancel. Here:

- ``local_stats``: a sampled row's block sum leaves out one column holding the
  block's maximum (its other ties add exp(0) each, ``vllm_others``).
- ``verify``: a step's S is the block sums rescaled to the row's maximum, each with
  its left-out column back (1 + sum), except the first block that holds the row's
  maximum; the draft's log probability is its shifted logit minus log1p(S), the
  rejected row's log-sum-exp M + log1p(S); the threshold p (1 - m) / (1 - p m)
  takes both complements from ``expm1`` of the logs (``vllm_block_threshold``).

The rest of the arithmetic is Triton 3.7's for sm_121a, read from the kernels' PTX
and SASS: ``tl.exp`` is ``ex2.approx.f32`` (subnormals kept) of the product with
log2(e), and every add, product and FMA rounds to nearest as the header writes it
(nvcc would contract them otherwise). ``log1p`` and ``expm1`` are CUDA's (libdevice,
within an ULP; these kernels compile without fast math). The random stream, the
Gumbel noise and the argmax order are ``tilelang_sampling``'s.

A vocabulary block's statistics (8192 columns, four warps) follow Triton's layout
(``_triton_run``): a thread holds 64 elements, sums their exponentials in register
order, then the warp sums at XOR distances 16 to 1 and threads 0-3 at 2 and 1; its
count of the maximum's ties (exact in FP32) reduces the same way. The maxima ignore
NaN, so any order gives them; the argmax is the Gumbel sampler's.

A row's S over its blocks runs on ``padded`` lanes, Triton's padded block count (a
power of two, at most one warp): lane ``k`` takes block ``k``, its term rounded,
reduced at XOR distances ``padded / 2`` to 1 (every lane reaches the same sum).

Every comparison NaN decides is made in the header: TVM's simplifier rewrites
``a <= b`` as ``!(b < a)``, which a NaN flips.
"""

# No `from __future__ import annotations`: TileLang reads the prim_func
# annotations eagerly, and the symbolic shapes must resolve.
from collections.abc import Callable, Sequence

import torch

from vllm.tilelang_utils import T, tilelang
from vllm.v1.worker.gpu.sample import tilelang_sampling
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

// expm1 within an ULP: CUDA's (libdevice), compiled without fast math.
__device__ __forceinline__ float vllm_expm1(float x) { return expm1f(x); }

// Barron, Gnanapragasam and Tschannen's log-softmax ("A Fix for Softmax and
// Cross-Entropy", Eq. 5), from a row's maximum m. These compute every value and
// select, in PTX: a C conditional lets nvcc predicate the exponential instead,
// which costs a column several instructions.
//
// A logit's shift x - m, exactly 0 where x is the maximum (an infinite one
// included).
__device__ __forceinline__ float vllm_shifted(float x, float m) {
  float y;
  asm("{\n"
      ".reg .pred tie;\n"
      ".reg .b32 d;\n"
      "sub.rn.f32 d, %1, %2;\n"
      "setp.eq.f32 tie, %1, %2;\n"
      "selp.f32 %0, 0f00000000, d, tie;\n"
      "}"
      : "=f"(y)
      : "f"(x), "f"(m));
  return y;
}

// tl.exp(x - m) of a column that is not the maximum (NaN is not), 0 for one that is.
__device__ __forceinline__ float vllm_other_exp(float x, float m) {
  float y;
  asm("{\n"
      ".reg .pred tie;\n"
      ".reg .b32 t, e;\n"
      "sub.rn.f32 t, %1, %2;\n"
      "mul.rn.f32 t, t, 0f3FB8AA3B;\n"
      "ex2.approx.f32 e, t;\n"
      "setp.eq.f32 tie, %1, %2;\n"
      "selp.f32 %0, 0f00000000, e, tie;\n"
      "}"
      : "=f"(y)
      : "f"(x), "f"(m));
  return y;
}

// 1 for a column holding the maximum, else 0: its ties, counted exactly in FP32.
__device__ __forceinline__ float vllm_tie(float x, float m) {
  float y;
  asm("set.eq.f32.f32 %0, %1, %2;" : "=f"(y) : "f"(x), "f"(m));
  return y;
}

// int: x is the maximum m.
__device__ __forceinline__ int vllm_is_max(float x, float m) {
  int y;
  asm("{\n"
      ".reg .pred tie;\n"
      "setp.eq.f32 tie, %1, %2;\n"
      "selp.s32 %0, 1, 0, tie;\n"
      "}"
      : "=r"(y)
      : "f"(x), "f"(m));
  return y;
}

// S, the others' sum: the columns that are not the maximum (sum), and of the
// maximum's ``ties`` all but one, exp(m - m) each (1; NaN for an infinite maximum,
// as the paper's z - z_max).
__device__ __forceinline__ float vllm_others(float sum, float ties, float m) {
  return ties > 1.0f
             ? vllm_add(sum, vllm_mul(vllm_sub(ties, 1.0f), vllm_exp_from(m, m)))
             : sum;
}

// A row's S from its block sums: NaN where its two or more columns are all -inf
// (the paper's z - z_max is -inf - -inf there, as in ``vllm_others``), which the
// block sums cannot tell, since a block whose maximum is -inf sums to 0.
__device__ __forceinline__ float vllm_row_others(float others, float m, int columns) {
  return (m == __uint_as_float(0xff800000u) && columns > 1)
             ? __uint_as_float(0x7fffffffu)
             : others;
}

// The log-sum-exp past the maximum, softplus(-margin) = log1p(S): CUDA's log1pf
// (libdevice, within an ULP).
__device__ __forceinline__ float vllm_softplus(float others) { return log1pf(others); }

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
// one, from the logs of the prefix's joint ratio p and of the next draft's target
// probability m: the residual mass p (1 - m) over residual + 1 - p, which is
// 1 - p m. Both complements are -expm1 of a log, never 1 - x of a probability near
// 1, where it cancels (1 - m was 0 once m rounded to 1); the quotient is IEEE's,
// and the threshold 1 where the denominator is not positive (p = m = 1, or NaN).
__device__ __forceinline__ float vllm_block_threshold(float log_p, float log_m) {
  const float residual = vllm_mul(vllm_exp(log_p), -vllm_expm1(log_m));
  const float denom = -vllm_expm1(vllm_add(log_p, log_m));
  float ratio;
  asm("div.rn.f32 %0, %1, %2;" : "=f"(ratio) : "f"(residual), "f"(denom));
  return denom > 0.0f ? ratio : 1.0f;
}

// u <= h, ordered: a NaN threshold accepts nothing.
__device__ __forceinline__ int vllm_at_most(float u, float h) { return u <= h; }
"""
)


def _distances(lanes: int) -> tuple[int, ...]:
    """The XOR distances of a butterfly over ``lanes`` lanes, the widest first."""
    return tuple(lanes >> i for i in range(1, lanes.bit_length()))


def _live_draft(live, step, dead_row_cut, req, cut: bool):
    """Whether step ``step`` verifies a draft: one of the request's (``live``) and,
    with a dead-row ``cut``, before the request's cut."""
    if cut:
        return live & (step < dead_row_cut[req])
    return live


def _ends_block(ended, count, dead_row_cut, req, cut: bool):
    """Whether the row after ``count`` accepted drafts ends the request's block: a
    bonus row or a placeholder (``ended``) or, with a dead-row ``cut``, a row at or
    past the request's cut."""
    if cut:
        return ended | (count >= dead_row_cut[req])
    return ended


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
    maximum and the sum of ``exp(logit - maximum)`` over its columns but one holding
    the maximum (``vllm_others``; 0 unless the maximum is above -inf). A thread's 64
    elements come in runs of ``run`` (``_triton_run``), all loaded with the request's
    state index: Triton loads them once the temperature is known, a round trip
    later."""
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
        apply_temperature: T.int32,
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
            ties = T.alloc_local((1,), T.float32)
            partial = T.alloc_shared((_WARPS,), T.float32)
            partial_index = T.alloc_shared((_WARPS,), T.int32)
            partial_sum = T.alloc_shared((_WARPS,), T.float32)
            partial_ties = T.alloc_shared((_WARPS,), T.float32)
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
                    if (apply_temperature != 0) & (temp != T.float32(1)):
                        for j in T.unroll(_STATS_PER_THREAD):
                            if blk * STATS_BLOCK + _element(j, tx, run) < vocab:
                                values[j] = T.call_extern(
                                    "float32", "vllm_div_full", values[j], temp
                                )
                        best[0] = values[0]
                        for j in T.unroll(1, _STATS_PER_THREAD):
                            best[0] = T.call_extern(
                                "float32", "vllm_max", best[0], values[j]
                            )
                    _block_reduce(best, partial, tx, "vllm_max")
                    if tx == 0:
                        maximum[0] = best[0]
                    T.sync_threads()
                    best[0] = maximum[0]
                    # The other columns' exponentials in register order, the first
                    # two added first; the maximum's columns counted.
                    total[0] = T.call_extern(
                        "float32", "vllm_other_exp", values[0], best[0]
                    )
                    ties[0] = T.call_extern("float32", "vllm_tie", values[0], best[0])
                    for j in T.unroll(1, _STATS_PER_THREAD):
                        total[0] = T.call_extern(
                            "float32",
                            "vllm_add",
                            total[0],
                            T.call_extern(
                                "float32", "vllm_other_exp", values[j], best[0]
                            ),
                        )
                        ties[0] = T.call_extern(
                            "float32",
                            "vllm_add",
                            ties[0],
                            T.call_extern("float32", "vllm_tie", values[j], best[0]),
                        )
                    _block_reduce(total, partial_sum, tx, "vllm_add")
                    _block_reduce(ties, partial_ties, tx, "vllm_add")
                    if tx == 0:
                        local_max[t, blk] = best[0]
                        local_sumexp[t, blk] = T.call_extern(
                            "float32",
                            "vllm_live_sum",
                            best[0],
                            T.call_extern(
                                "float32", "vllm_others", total[0], ties[0], best[0]
                            ),
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
    apply_temperature: bool = False,
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
        int(apply_temperature),
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
    cut: bool = False,
):
    """``_compute_cumulative_log_p_kernel`` and ``_rejection_kernel`` (block
    verification, one-hot drafts), a block a request. ``lanes`` is Triton's
    PADDED_VOCAB_NUM_BLOCKS and ``width`` the sampled tokens' (num_speculative_steps
    + 1); with ``cut``, the drafts from each request's ``dead_row_cut`` on are
    placeholders (see the module docstring). Each draft step takes ``lanes`` threads,
    one a vocabulary block: they load its statistics, its draft and the draft's
    target logit at once, every load at an index that exists whatever the masks say,
    the masks as selects, and reduce them to the step's log-sum-exp and the draft's
    log probability (the paper's Eq. 5), the greedy step's target argmax, and the
    step's uniform draw; each step's first thread then folds its cumulative log
    ratio and threshold. Thread 0 then runs
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
    drafts_len, idx_len, temp_len, seed_len, pos_len, cut_len = T.dynamic(
        "drafts_len, idx_len, temp_len, seed_len, pos_len, cut_len"
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
        dead_row_cut: T.Tensor((cut_len,), T.int32),
        apply_temperature: T.int32,
    ):
        with T.Kernel(reqs, threads=threads) as req:
            T.import_source(_HEADER)
            tx = T.get_thread_binding()
            step = tx // lanes
            block = tx % lanes
            row_max = T.alloc_local((1,), T.float32)
            first = T.alloc_local((1,), T.int32)
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
            cumulative_s = T.alloc_shared((groups,), T.float32)
            threshold_s = T.alloc_shared((groups,), T.float32)

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
            position = T.alloc_var(pos_dtype, init=pos[row])
            temp = T.alloc_var(T.float32, init=temperature[state])
            seed = T.alloc_var(T.int64, init=seeds[state])
            in_row = live & (block < blocks)
            b_clamped = T.min(block, blocks - 1)
            maximum = T.alloc_var(T.float32, init=-T.infinity(T.float32))
            sumexp = T.alloc_var(T.float32, init=T.float32(0))
            if in_row:
                maximum = local_max[row, b_clamped]
                if temp != T.float32(0):
                    sumexp = local_sumexp[row, b_clamped]
            # The cut's own row is live: it draws the truncated block's bonus (and a
            # greedy request's target argmax there).
            draft = T.if_then_else(
                _live_draft(live, step, dead_row_cut, req, cut), draft, T.int64(-1)
            )
            logit = T.alloc_var(
                T.float32,
                init=T.cast(logits[row, T.max(draft, T.int64(0))], T.float32),
            )
            if (
                (apply_temperature != 0)
                & (temp != T.float32(0))
                & (temp != T.float32(1))
            ):
                logit = T.call_extern("float32", "vllm_div_full", logit, temp)

            # The step's S: the blocks' maximum, the first block holding it (a NaN
            # maximum holds none), then the block sums rescaled to it, each with its
            # left-out column back (1 + sum) except the first block's.
            row_max[0] = maximum
            for distance in _distances(lanes):
                row_max[0] = T.call_extern(
                    "float32", "vllm_max", row_max[0], T.shfl_xor(row_max[0], distance)
                )
            first[0] = T.if_then_else(
                T.call_extern("int32", "vllm_is_max", maximum, row_max[0]) != 0,
                block,
                lanes,
            )
            for distance in _distances(lanes):
                first[0] = T.min(first[0], T.shfl_xor(first[0], distance))
            total[0] = T.if_then_else(
                block == first[0],
                sumexp,
                T.call_extern(
                    "float32",
                    "vllm_mul",
                    T.call_extern("float32", "vllm_exp_from", maximum, row_max[0]),
                    T.call_extern("float32", "vllm_add", T.float32(1), sumexp),
                ),
            )
            for distance in _distances(lanes):
                total[0] = T.call_extern(
                    "float32", "vllm_add", total[0], T.shfl_xor(total[0], distance)
                )
            softplus = T.alloc_var(
                T.float32,
                init=T.call_extern(
                    "float32",
                    "vllm_softplus",
                    T.call_extern(
                        "float32", "vllm_row_others", total[0], row_max[0], vocab
                    ),
                ),
            )

            # _compute_global_target_argmax: NaN maxima count as -inf, and the lower
            # block wins a tie.
            best[0] = T.if_then_else(T.isnan(maximum), -T.infinity(T.float32), maximum)
            best_index[0] = block
            for distance in _distances(lanes):
                _exchange(best, best_index, other, other_index, distance)
            target = T.alloc_var(T.int64, init=T.int64(0))
            if live & (temp == T.float32(0)):
                target = local_argmax[row, T.min(best_index[0], blocks - 1)]

            # The step's draw: tl_rand32 at spec_step_seed(seed, step).
            word = _randint_at(
                seed ^ T.shift_left(T.cast(step, T.int64), T.int64(32)),
                position,
                pos_dtype,
                False,
            )
            if block == 0:
                lse_s[step] = T.call_extern("float32", "vllm_add", row_max[0], softplus)
                log_prob_s[step] = T.call_extern(
                    "float32",
                    "vllm_sub",
                    T.call_extern("float32", "vllm_shifted", logit, row_max[0]),
                    softplus,
                )
                uniform_s[step] = T.call_extern("float32", "vllm_uniform", word)
                draft_s[step] = draft
                target_s[step] = target
            T.sync_threads()

            # Each step's cumulative log ratio and block verification's threshold at
            # once: the step's first thread folds the steps through it in order,
            # with a -1 placeholder leaving the ratio as it was, so the bits are a
            # serial loop's. The threshold is the prefix's joint ratio, or that of
            # the residual mass when a valid draft follows (from the logs of the
            # ratio and of the next step's draft probability).
            if block == 0:  # noqa: SIM102
                if step < steps:
                    log_p = T.alloc_var(T.float32, init=T.float32(0))
                    for k in tuple(range(steps)):
                        if k <= step:
                            log_p = T.if_then_else(
                                draft_s[k] >= 0,
                                T.call_extern(
                                    "float32", "vllm_log_ratio", log_p, log_prob_s[k]
                                ),
                                log_p,
                            )
                    cumulative_s[step] = log_p
                    after = T.alloc_var(T.int32, init=T.min(step + 1, steps - 1))
                    threshold_s[step] = T.if_then_else(
                        (step + 1 < steps) & (draft_s[after] >= 0),
                        T.call_extern(
                            "float32", "vllm_block_threshold", log_p, log_prob_s[after]
                        ),
                        T.call_extern("float32", "vllm_exp", log_p),
                    )
            T.sync_threads()

            if tx == 0:
                greedy = T.alloc_var(
                    T.int32, init=T.cast(temp == T.float32(0), T.int32)
                )
                accepted = T.alloc_var(T.int64, init=T.int64(0))
                verifying = T.alloc_var(T.int32, init=1)
                for i in tuple(range(steps)):
                    if i < num_drafts:
                        drafted = T.alloc_var(T.int64, init=draft_s[i])
                        valid = T.cast(drafted >= 0, T.int32)
                        token = T.max(drafted, T.int64(0))
                        if greedy == 0:
                            # The cumulative ratio; a placeholder ends
                            # verification.
                            cumulative[start + i] = cumulative_s[i]
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
                                # Block verification: the last step whose draw is
                                # at most its threshold ends the accepted block.
                                accepted = T.if_then_else(
                                    T.call_extern(
                                        "int32",
                                        "vllm_at_most",
                                        uniform_s[i],
                                        threshold_s[i],
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
    return _padded_blocks_count(blocks)


def _padded_blocks_count(blocks: int) -> int:
    """Next power of two at or above ``blocks`` (at least 1)."""
    return 1 << max(blocks - 1, 0).bit_length()


def verify_key(
    logits_dtype: torch.dtype,
    drafts_dtype: torch.dtype,
    idx_dtype: torch.dtype,
    pos_dtype: torch.dtype,
    vocab_size: int,
    num_speculative_steps: int,
    cut: bool = False,
) -> tuple:
    return (
        _DTYPES[logits_dtype],
        _DTYPES[drafts_dtype],
        _DTYPES[idx_dtype],
        _DTYPES[pos_dtype],
        _padded_blocks(vocab_size),
        num_speculative_steps + 1,
        cut,
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
    dead_row_cut: torch.Tensor | None = None,  # [num_reqs] int32
    apply_temperature: bool = False,
) -> None:
    """The launches of ``_compute_cumulative_log_p_kernel`` and
    ``_rejection_kernel`` (block verification, one-hot drafts) as one, into the
    given outputs; with ``dead_row_cut``, each request's block ends at its cut."""
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
            dead_row_cut is not None,
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
        # Without a cut it is not read: the counts stand in.
        num_sampled if dead_row_cut is None else dead_row_cut,
        int(apply_temperature),
    )


# --- Resampling ---------------------------------------------------------------


@tilelang.jit(pass_configs=_PASS)
def resample_kernel(
    logits_dtype: str,
    idx_dtype: str,
    drafts_dtype: str,
    pos_dtype: str,
    run: int,
    cut: bool = False,
):
    """``_resample_kernel`` without draft logits or watermark, FP32: one block of
    128 threads per ``(request, vocabulary block of 1024)``, a thread's elements in
    runs of ``run`` (``_triton_run``). The request's first unaccepted row is
    resampled: the target logits with a rejected draft masked to -inf (the residual of
    a one-hot draft; a bonus row and a -1 placeholder mask nothing, nor with ``cut``
    the row at or past the request's ``dead_row_cut``, the bonus of its truncated
    block), plus the draw's Gumbel noise unless the temperature is 0; each block's
    argmax index (global, int64) and score. A greedy request's rejected row is left
    alone: verification stored its target argmax."""
    reqs, num_logits, vocab, blocks = T.dynamic("reqs, num_logits, vocab, blocks")
    # Inputs may be longer than the batch: each has its own length.
    idx_len, drafts_len, temp_len, seed_len, pos_len, cut_len = T.dynamic(
        "idx_len, drafts_len, temp_len, seed_len, pos_len, cut_len"
    )
    logits_stride, argmax_stride, max_stride = T.dynamic(
        "logits_stride, argmax_stride, max_stride", "int64"
    )

    @T.macro
    def passes(logits, raw, score, best, best_index, row, rejected, word_seed, temp,
               apply_temperature, blk, tx, noisy):  # fmt: skip
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
                    score[0] = T.cast(raw[k], T.float32)
                    if noisy:
                        if (apply_temperature != 0) & (temp != T.float32(1)):
                            score[0] = T.call_extern(
                                "float32", "vllm_div_full", score[0], temp
                            )
                    score[0] = T.if_then_else(
                        T.cast(index, T.int64) == rejected[0],
                        -T.infinity(T.float32),
                        score[0],
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
        dead_row_cut: T.Tensor((cut_len,), T.int32),
        apply_temperature: T.int32,
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
                rejected[0] = T.if_then_else(
                    _ends_block(
                        bonus | (draft < 0), num_sampled[req], dead_row_cut, req, cut
                    ),
                    T.int64(-1),
                    draft,
                )
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
                           word_seed, temp, apply_temperature, blk, tx, True)  # fmt: skip
                else:
                    passes(logits, raw, score, best, best_index, row, rejected,
                           word_seed, temp, apply_temperature, blk, tx, False)  # fmt: skip
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
    cut: bool = False,
) -> tuple:
    return (
        _DTYPES[logits_dtype],
        _DTYPES[idx_dtype],
        _DTYPES[drafts_dtype],
        _DTYPES[pos_dtype],
        run,
        cut,
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
    dead_row_cut: torch.Tensor | None = None,  # [num_reqs] int32
    apply_temperature: bool = False,
) -> None:
    """``_resample_kernel``'s launch without draft logits or watermark (FP32), into
    the given ``[num_reqs, cdiv(vocab_size, 1024)]`` block winners; with
    ``dead_row_cut``, a row at a request's cut draws its block's bonus."""
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
            dead_row_cut is not None,
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
        # Without a cut it is not read: the counts stand in.
        num_sampled if dead_row_cut is None else dead_row_cut,
        int(apply_temperature),
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
    # [num_reqs] int32
    dead_row_cut: torch.Tensor | None = None,
    kernel_steps: int | None = None,
    apply_temperature: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``rejection_sample``'s launches before the insertion, with one-hot drafts,
    block verification, FP32 draws and no watermark: the sampled tokens
    ([num_reqs, num_speculative_steps + 1], int64) up to the first rejected or bonus
    token, their counts ([num_reqs], int32), and the block winners of that token's
    resampling (``[num_reqs, cdiv(V, 1024)]`` indices, int64, and scores, FP32).
    With ``dead_row_cut``, each request's block ends at its cut: a count is at most
    the cut, and the token after it comes from a live row. ``kernel_steps``: a step
    whose requests verify at most this many drafts runs the kernels at that width
    (copy drafts: DSpark's block on a step without a copy); the sampled tokens keep
    every column."""
    num_reqs = cu_num_logits.shape[0] - 1
    num_logits, vocab_size = target_logits.shape
    steps = num_speculative_steps if kernel_steps is None else kernel_steps
    assert steps <= num_speculative_steps

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
        steps,
        apply_temperature,
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
        sampled[:, : steps + 1],
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
        steps,
        dead_row_cut,
        apply_temperature,
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
        dead_row_cut,
        apply_temperature,
    )
    return sampled, num_sampled, resampled_local_argmax, resampled_local_max


# --- Preparation --------------------------------------------------------------


def _runs_for_strides(
    dtype: torch.dtype, strides: Sequence[int] | None = None
) -> tuple[int, ...]:
    if not strides:
        return _runs(dtype)
    vec = 16 // torch.empty((), dtype=dtype).element_size()
    return tuple(dict.fromkeys(vec if stride % 16 == 0 else 1 for stride in strides))


def rejection_calls(
    logits_dtypes: Sequence[torch.dtype],
    vocab_size: int,
    num_speculative_steps: int,
    cuts: Sequence[bool] = (False,),
    *,
    logits_strides: Sequence[tuple[torch.dtype, int]] | None = None,
) -> list:
    """``verify_and_resample``'s calls for ``vocab_size``-wide target logits of
    each of ``logits_dtypes``, covering the layouts reachable from
    ``logits_strides``, with the input batch's int32 drafts and local positions
    and its int64 indices and positions; verification and resampling for each of
    ``cuts`` (True: with a dead-row cut)."""
    strides_by_dtype: dict[torch.dtype, list[int]] = {}
    if logits_strides is not None:
        for dtype, stride in logits_strides:
            strides_by_dtype.setdefault(dtype, []).append(stride)
    calls = []
    for dtype in dict.fromkeys(logits_dtypes):
        for run in _runs_for_strides(dtype, strides_by_dtype.get(dtype)):
            calls.append(
                _call(
                    local_stats_kernel,
                    *local_stats_key(dtype, RUNNER_IDX_DTYPE, torch.int32, run),
                )
            )
            calls += [
                _call(
                    resample_kernel,
                    *resample_key(
                        dtype, RUNNER_IDX_DTYPE, torch.int32, torch.int64, run, cut
                    ),
                )
                for cut in cuts
            ]
        calls += [
            _call(
                verify_kernel,
                *verify_key(
                    dtype,
                    torch.int32,
                    RUNNER_IDX_DTYPE,
                    torch.int64,
                    vocab_size,
                    num_speculative_steps,
                    cut,
                ),
            )
            for cut in cuts
        ]
    return calls


def register_block_verification(
    logits_dtypes: Sequence[torch.dtype],
    vocab_size: int,
    num_speculative_steps: int,
    dead_rows: bool = False,
    *,
    logits_strides: Sequence[tuple[torch.dtype, int]] | None = None,
) -> None:
    """Declare ``rejection_calls`` with the runner's JIT registry (a no-op where
    none collects); with ``dead_rows``, with a dead-row cut too (the steps that
    verify), beside the steps without one (the warmup's)."""
    cuts = (False, True) if dead_rows else (False,)
    strides = tuple(logits_strides) if logits_strides is not None else None
    declare(
        lambda: rejection_calls(
            logits_dtypes,
            vocab_size,
            num_speculative_steps,
            cuts,
            logits_strides=strides,
        )
    )



# --- Shard-local verification and resampling ----------------------------------


@tilelang.jit(pass_configs=_PASS)
def local_stats_shard_kernel(
    logits_dtype: str,
    drafts_dtype: str,
    idx_dtype: str,
    step_dtype: str,
    run: int,
):
    """Per-rank vocabulary-block statistics for shard-local verification.

    Packs 4 FP32 values per 8192-column block into ``local_stats`` of shape
    ``[rows, 4 * blocks]`` (16 bytes per block, 16-byte aligned row width for
    one-shot ``all_gather(..., dim=-1)``):
    - Greedy row (``temp == 0``): ``[block_max, 0, float32(global_argmax), 0]``
    - Sampled row (``temp != 0``): ``[block_max, block_sumexp, owns_draft, draft_logit]``
      where ``owns_draft`` and ``draft_logit`` are populated in ``blk == 0`` when
      ``drafts[t + 1]`` falls in ``[vocab_start, vocab_start + vocab)``.
    """
    rows, vocab, blocks = T.dynamic("rows, vocab, blocks")
    drafts_len, idx_len, step_len, temp_len = T.dynamic(
        "drafts_len, idx_len, step_len, temp_len"
    )
    logits_stride, stats_stride = T.dynamic("logits_stride, stats_stride", "int64")

    @T.prim_func
    def local_stats_shard(
        local_stats: T.StridedTensor(
            (rows, 4 * blocks), (stats_stride, 1), T.float32
        ),
        logits: T.StridedTensor((rows, vocab), (logits_stride, 1), logits_dtype),
        drafts: T.Tensor((drafts_len,), drafts_dtype),
        expanded_idx_mapping: T.Tensor((idx_len,), idx_dtype),
        expanded_local_pos: T.Tensor((step_len,), step_dtype),
        temperature: T.Tensor((temp_len,), T.float32),
        num_speculative_steps: T.int32,
        apply_temperature: T.int32,
        vocab_start: T.int32,
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
            ties = T.alloc_local((1,), T.float32)
            partial = T.alloc_shared((_WARPS,), T.float32)
            partial_index = T.alloc_shared((_WARPS,), T.int32)
            partial_sum = T.alloc_shared((_WARPS,), T.float32)
            partial_ties = T.alloc_shared((_WARPS,), T.float32)
            maximum = T.alloc_shared((1,), T.float32)
            if run > 1:
                T.assume(logits_stride % 16 == 0)

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
                for j in T.unroll(_STATS_PER_THREAD):
                    values[j] = T.if_then_else(
                        blk * STATS_BLOCK + _element(j, tx, run) < vocab,
                        T.cast(raw[j], T.float32),
                        -T.infinity(T.float32),
                    )
                best[0] = values[0]
                for j in T.unroll(1, _STATS_PER_THREAD):
                    best[0] = T.call_extern("float32", "vllm_max", best[0], values[j])
                if temp == T.float32(0):
                    best[0] = values[0]
                    best_index[0] = _element(0, tx, run)
                    for j in T.unroll(1, _STATS_PER_THREAD):
                        _keep_or_take(
                            best, best_index, values[j], _element(j, tx, run)
                        )
                    _block_argmax(
                        best,
                        best_index,
                        other,
                        other_index,
                        partial,
                        partial_index,
                        tx,
                    )
                    if tx == 0:
                        local_stats[t, blk * 4 + 0] = best[0]
                        local_stats[t, blk * 4 + 1] = T.float32(0)
                        local_stats[t, blk * 4 + 2] = T.cast(
                            vocab_start + blk * STATS_BLOCK + best_index[0],
                            T.float32,
                        )
                        local_stats[t, blk * 4 + 3] = T.float32(0)
                else:
                    if (apply_temperature != 0) & (temp != T.float32(1)):
                        for j in T.unroll(_STATS_PER_THREAD):
                            if blk * STATS_BLOCK + _element(j, tx, run) < vocab:
                                values[j] = T.call_extern(
                                    "float32", "vllm_div_full", values[j], temp
                                )
                        best[0] = values[0]
                        for j in T.unroll(1, _STATS_PER_THREAD):
                            best[0] = T.call_extern(
                                "float32", "vllm_max", best[0], values[j]
                            )
                    _block_reduce(best, partial, tx, "vllm_max")
                    if tx == 0:
                        maximum[0] = best[0]
                    T.sync_threads()
                    best[0] = maximum[0]
                    total[0] = T.call_extern(
                        "float32", "vllm_other_exp", values[0], best[0]
                    )
                    ties[0] = T.call_extern("float32", "vllm_tie", values[0], best[0])
                    for j in T.unroll(1, _STATS_PER_THREAD):
                        total[0] = T.call_extern(
                            "float32",
                            "vllm_add",
                            total[0],
                            T.call_extern(
                                "float32", "vllm_other_exp", values[j], best[0]
                            ),
                        )
                        ties[0] = T.call_extern(
                            "float32",
                            "vllm_add",
                            ties[0],
                            T.call_extern("float32", "vllm_tie", values[j], best[0]),
                        )
                    _block_reduce(total, partial_sum, tx, "vllm_add")
                    _block_reduce(ties, partial_ties, tx, "vllm_add")
                    if tx == 0:
                        local_stats[t, blk * 4 + 0] = best[0]
                        local_stats[t, blk * 4 + 1] = T.call_extern(
                            "float32",
                            "vllm_live_sum",
                            best[0],
                            T.call_extern(
                                "float32",
                                "vllm_others",
                                total[0],
                                ties[0],
                                best[0],
                            ),
                        )
                        if blk == 0:
                            draft_tok = T.alloc_var(
                                T.int64,
                                init=T.cast(
                                    drafts[T.min(t + 1, drafts_len - 1)], T.int64
                                ),
                            )
                            draft_col = T.alloc_var(
                                T.int64,
                                init=draft_tok - T.cast(vocab_start, T.int64),
                            )
                            owns_draft = (draft_col >= T.int64(0)) & (
                                draft_col < T.cast(vocab, T.int64)
                            )
                            d_val = T.alloc_var(
                                T.float32,
                                init=T.if_then_else(
                                    owns_draft,
                                    T.cast(
                                        logits[
                                            t,
                                            T.min(
                                                T.max(
                                                    T.cast(draft_col, T.int32), 0
                                                ),
                                                vocab - 1,
                                            ),
                                        ],
                                        T.float32,
                                    ),
                                    T.float32(0),
                                ),
                            )
                            if (apply_temperature != 0) & (temp != T.float32(1)):
                                d_val = T.call_extern(
                                    "float32", "vllm_div_full", d_val, temp
                                )
                            local_stats[t, 2] = T.if_then_else(
                                owns_draft, T.float32(1), T.float32(0)
                            )
                            local_stats[t, 3] = T.if_then_else(
                                owns_draft, d_val, T.float32(0)
                            )
                        else:
                            local_stats[t, blk * 4 + 2] = T.float32(0)
                            local_stats[t, blk * 4 + 3] = T.float32(0)
            elif tx == 0:
                local_stats[t, blk * 4 + 0] = -T.infinity(T.float32)
                local_stats[t, blk * 4 + 1] = T.float32(0)
                local_stats[t, blk * 4 + 2] = T.float32(0)
                local_stats[t, blk * 4 + 3] = T.float32(0)

    return local_stats_shard


def local_stats_shard_key(
    logits_dtype: torch.dtype,
    drafts_dtype: torch.dtype,
    idx_dtype: torch.dtype,
    step_dtype: torch.dtype,
    run: int = 1,
) -> tuple:
    return (
        _DTYPES[logits_dtype],
        _DTYPES[drafts_dtype],
        _DTYPES[idx_dtype],
        _DTYPES[step_dtype],
        run,
    )


def compute_local_stats_shard(
    local_stats: torch.Tensor,  # [num_logits, 4 * local_blocks]
    logits: torch.Tensor,  # [num_logits, shard_vocab_size]
    draft_sampled: torch.Tensor,  # [num_logits]
    expanded_idx_mapping: torch.Tensor,  # [num_logits]
    expanded_local_pos: torch.Tensor,  # [num_logits]
    temperature: torch.Tensor,  # [max_num_reqs]
    num_speculative_steps: int,
    vocab_start: int,
    apply_temperature: bool = False,
) -> None:
    if logits.shape[0] == 0:
        return
    kernel = _kernel(
        local_stats_shard_kernel,
        *local_stats_shard_key(
            logits.dtype,
            draft_sampled.dtype,
            expanded_idx_mapping.dtype,
            expanded_local_pos.dtype,
            _triton_run(logits),
        ),
    )
    kernel(
        local_stats,
        logits,
        draft_sampled.contiguous(),
        expanded_idx_mapping.contiguous(),
        expanded_local_pos.contiguous(),
        temperature,
        num_speculative_steps,
        int(apply_temperature),
        int(vocab_start),
    )


@tilelang.jit(pass_configs=_PASS)
def verify_shard_kernel(
    drafts_dtype: str,
    idx_dtype: str,
    pos_dtype: str,
    lanes: int,
    width: int,
    cut: bool = False,
):
    """Block verification from all-gathered per-rank vocabulary-block statistics."""
    steps = width - 1
    if not 1 <= lanes <= _MAX_LANES:
        raise ValueError(f"{lanes} vocabulary blocks: a row's blocks take one warp")
    threads = max(32, -(-max(steps, 1) * lanes // 32) * 32)
    if threads > 1024:
        raise ValueError(f"{steps} draft steps of {lanes} blocks exceed a block")
    groups = threads // lanes
    reqs, num_logits, blocks = T.dynamic("reqs, num_logits, blocks")
    drafts_len, idx_len, temp_len, seed_len, pos_len, cut_len = T.dynamic(
        "drafts_len, idx_len, temp_len, seed_len, pos_len, cut_len"
    )
    sampled_stride, stats_stride = T.dynamic("sampled_stride, stats_stride", "int64")

    @T.prim_func
    def verify_shard(
        sampled: T.StridedTensor((reqs, width), (sampled_stride, 1), T.int64),
        num_sampled: T.Tensor((reqs,), T.int32),
        target_lse: T.Tensor((reqs,), T.float32),
        draft_lse: T.Tensor((reqs,), T.float32),
        cumulative: T.Tensor((num_logits,), T.float32),
        gathered_stats: T.StridedTensor(
            (num_logits, 4 * blocks), (stats_stride, 1), T.float32
        ),
        drafts: T.Tensor((drafts_len,), drafts_dtype),
        cu_num_logits: T.Tensor((reqs + 1,), T.int32),
        idx_mapping: T.Tensor((idx_len,), idx_dtype),
        temperature: T.Tensor((temp_len,), T.float32),
        seeds: T.Tensor((seed_len,), T.int64),
        pos: T.Tensor((pos_len,), pos_dtype),
        dead_row_cut: T.Tensor((cut_len,), T.int32),
        vocab: T.int32,
    ):
        with T.Kernel(reqs, threads=threads) as req:
            T.import_source(_HEADER)
            tx = T.get_thread_binding()
            step = tx // lanes
            block = tx % lanes
            row_max = T.alloc_local((1,), T.float32)
            first = T.alloc_local((1,), T.int32)
            total = T.alloc_local((1,), T.float32)
            best = T.alloc_local((1,), T.float32)
            best_index = T.alloc_local((1,), T.int32)
            other = T.alloc_local((1,), T.float32)
            other_index = T.alloc_local((1,), T.int32)
            has_draft = T.alloc_local((1,), T.int32)
            draft_logit = T.alloc_local((1,), T.float32)
            lse_s = T.alloc_shared((groups,), T.float32)
            log_prob_s = T.alloc_shared((groups,), T.float32)
            uniform_s = T.alloc_shared((groups,), T.float32)
            draft_s = T.alloc_shared((groups,), T.int64)
            target_s = T.alloc_shared((groups,), T.int64)
            cumulative_s = T.alloc_shared((groups,), T.float32)
            threshold_s = T.alloc_shared((groups,), T.float32)

            start = T.alloc_var(T.int32, init=cu_num_logits[req])
            end = T.alloc_var(T.int32, init=cu_num_logits[req + 1])
            state = T.alloc_var(T.int64, init=T.cast(idx_mapping[req], T.int64))
            num_drafts = end - start - 1
            live = step < num_drafts
            row = T.alloc_var(T.int32, init=start + T.min(step, num_drafts))
            draft = T.alloc_var(
                T.int64,
                init=T.cast(drafts[start + T.min(step + 1, num_drafts)], T.int64),
            )
            b_clamped = T.min(block, blocks - 1)
            position = T.alloc_var(pos_dtype, init=pos[row])
            temp = T.alloc_var(T.float32, init=temperature[state])
            seed = T.alloc_var(T.int64, init=seeds[state])
            in_row = live & (block < blocks)
            maximum = T.alloc_var(T.float32, init=-T.infinity(T.float32))
            sumexp = T.alloc_var(T.float32, init=T.float32(0))
            has_draft[0] = 0
            draft_logit[0] = T.float32(0)
            if in_row:
                maximum = gathered_stats[row, b_clamped * 4 + 0]
                if temp != T.float32(0):
                    sumexp = gathered_stats[row, b_clamped * 4 + 1]
                    if gathered_stats[row, b_clamped * 4 + 2] != T.float32(0):
                        has_draft[0] = 1
                        draft_logit[0] = gathered_stats[row, b_clamped * 4 + 3]
            draft = T.if_then_else(
                _live_draft(live, step, dead_row_cut, req, cut), draft, T.int64(-1)
            )
            for distance in _distances(lanes):
                other_has = T.shfl_xor(has_draft[0], distance)
                other_logit = T.shfl_xor(draft_logit[0], distance)
                draft_logit[0] = T.if_then_else(
                    has_draft[0] != 0, draft_logit[0], other_logit
                )
                has_draft[0] = has_draft[0] | other_has
            logit = T.alloc_var(T.float32, init=draft_logit[0])

            row_max[0] = maximum
            for distance in _distances(lanes):
                row_max[0] = T.call_extern(
                    "float32", "vllm_max", row_max[0], T.shfl_xor(row_max[0], distance)
                )
            first[0] = T.if_then_else(
                T.call_extern("int32", "vllm_is_max", maximum, row_max[0]) != 0,
                block,
                lanes,
            )
            for distance in _distances(lanes):
                first[0] = T.min(first[0], T.shfl_xor(first[0], distance))
            total[0] = T.if_then_else(
                block == first[0],
                sumexp,
                T.call_extern(
                    "float32",
                    "vllm_mul",
                    T.call_extern("float32", "vllm_exp_from", maximum, row_max[0]),
                    T.call_extern("float32", "vllm_add", T.float32(1), sumexp),
                ),
            )
            for distance in _distances(lanes):
                total[0] = T.call_extern(
                    "float32", "vllm_add", total[0], T.shfl_xor(total[0], distance)
                )
            softplus = T.alloc_var(
                T.float32,
                init=T.call_extern(
                    "float32",
                    "vllm_softplus",
                    T.call_extern(
                        "float32", "vllm_row_others", total[0], row_max[0], vocab
                    ),
                ),
            )

            best[0] = T.if_then_else(T.isnan(maximum), -T.infinity(T.float32), maximum)
            best_index[0] = block
            for distance in _distances(lanes):
                _exchange(best, best_index, other, other_index, distance)
            target = T.alloc_var(T.int64, init=T.int64(0))
            if live & (temp == T.float32(0)):
                target = T.cast(
                    gathered_stats[row, T.min(best_index[0], blocks - 1) * 4 + 2],
                    T.int64,
                )

            word = _randint_at(
                seed ^ T.shift_left(T.cast(step, T.int64), T.int64(32)),
                position,
                pos_dtype,
                False,
            )
            if block == 0:
                lse_s[step] = T.call_extern("float32", "vllm_add", row_max[0], softplus)
                log_prob_s[step] = T.call_extern(
                    "float32",
                    "vllm_sub",
                    T.call_extern("float32", "vllm_shifted", logit, row_max[0]),
                    softplus,
                )
                uniform_s[step] = T.call_extern("float32", "vllm_uniform", word)
                draft_s[step] = draft
                target_s[step] = target
            T.sync_threads()

            if block == 0:  # noqa: SIM102
                if step < steps:
                    log_p = T.alloc_var(T.float32, init=T.float32(0))
                    for k in tuple(range(steps)):
                        if k <= step:
                            log_p = T.if_then_else(
                                draft_s[k] >= 0,
                                T.call_extern(
                                    "float32", "vllm_log_ratio", log_p, log_prob_s[k]
                                ),
                                log_p,
                            )
                    cumulative_s[step] = log_p
                    after = T.alloc_var(T.int32, init=T.min(step + 1, steps - 1))
                    threshold_s[step] = T.if_then_else(
                        (step + 1 < steps) & (draft_s[after] >= 0),
                        T.call_extern(
                            "float32", "vllm_block_threshold", log_p, log_prob_s[after]
                        ),
                        T.call_extern("float32", "vllm_exp", log_p),
                    )
            T.sync_threads()

            if tx == 0:
                greedy = T.alloc_var(
                    T.int32, init=T.cast(temp == T.float32(0), T.int32)
                )
                accepted = T.alloc_var(T.int64, init=T.int64(0))
                verifying = T.alloc_var(T.int32, init=1)
                for i in tuple(range(steps)):
                    if i < num_drafts:
                        drafted = T.alloc_var(T.int64, init=draft_s[i])
                        valid = T.cast(drafted >= 0, T.int32)
                        token = T.max(drafted, T.int64(0))
                        if greedy == 0:
                            cumulative[start + i] = cumulative_s[i]
                            verifying = verifying & valid
                        if verifying != 0:
                            if greedy != 0:
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
                                accepted = T.if_then_else(
                                    T.call_extern(
                                        "int32",
                                        "vllm_at_most",
                                        uniform_s[i],
                                        threshold_s[i],
                                    )
                                    != 0,
                                    T.int64(i + 1),
                                    accepted,
                                )
                                sampled[req, i] = token
                num_sampled[req] = T.cast(accepted, T.int32)
                target_lse[req] = T.if_then_else(
                    (greedy == 0) & (accepted < T.cast(num_drafts, T.int64)),
                    lse_s[T.min(accepted, T.int64(groups - 1))],
                    T.float32(0),
                )
                draft_lse[req] = T.float32(0)

    return verify_shard


def verify_shard_key(
    drafts_dtype: torch.dtype,
    idx_dtype: torch.dtype,
    pos_dtype: torch.dtype,
    total_blocks: int,
    num_speculative_steps: int,
    cut: bool = False,
) -> tuple:
    return (
        _DTYPES[drafts_dtype],
        _DTYPES[idx_dtype],
        _DTYPES[pos_dtype],
        _padded_blocks_count(total_blocks),
        num_speculative_steps + 1,
        cut,
    )


def verify_shard(
    sampled: torch.Tensor,
    num_sampled: torch.Tensor,
    target_rejected_logsumexp: torch.Tensor,
    draft_rejected_logsumexp: torch.Tensor,
    cumulative_log_p: torch.Tensor,
    gathered_stats: torch.Tensor,
    draft_sampled: torch.Tensor,
    cu_num_logits: torch.Tensor,
    idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    seed: torch.Tensor,
    pos: torch.Tensor,
    num_speculative_steps: int,
    vocab_size: int,
    dead_row_cut: torch.Tensor | None = None,
) -> None:
    if num_sampled.shape[0] == 0:
        return
    total_blocks = gathered_stats.shape[1] // 4
    kernel = _kernel(
        verify_shard_kernel,
        *verify_shard_key(
            draft_sampled.dtype,
            idx_mapping.dtype,
            pos.dtype,
            total_blocks,
            num_speculative_steps,
            dead_row_cut is not None,
        ),
    )
    kernel(
        sampled,
        num_sampled,
        target_rejected_logsumexp,
        draft_rejected_logsumexp,
        cumulative_log_p,
        gathered_stats,
        draft_sampled.contiguous(),
        cu_num_logits,
        idx_mapping.contiguous(),
        temperature,
        seed,
        pos.contiguous(),
        num_sampled if dead_row_cut is None else dead_row_cut,
        int(vocab_size),
    )


@tilelang.jit(pass_configs=_PASS)
def resample_shard_kernel(
    logits_dtype: str,
    idx_dtype: str,
    drafts_dtype: str,
    pos_dtype: str,
    run: int,
    cut: bool = False,
):
    """Shard-local ``resample_kernel`` with global Philox offsets and token indices."""
    reqs, num_logits, vocab, blocks = T.dynamic("reqs, num_logits, vocab, blocks")
    idx_len, drafts_len, temp_len, seed_len, pos_len, cut_len = T.dynamic(
        "idx_len, drafts_len, temp_len, seed_len, pos_len, cut_len"
    )
    logits_stride, argmax_stride, max_stride = T.dynamic(
        "logits_stride, argmax_stride, max_stride", "int64"
    )

    @T.macro
    def passes(logits, raw, score, best, best_index, row, rejected, word_seed, temp,
               apply_temperature, vocab_start, blk, tx, noisy):  # fmt: skip
        for r in T.unroll(_PASSES // run):
            col0 = blk * RESAMPLE_BLOCK + r * (_THREADS * run) + run * tx
            _load_run(logits, raw, row, col0, vocab, run)
            for k in T.unroll(run):
                local = r * (_THREADS * run) + run * tx + k
                index = col0 + k
                global_index = vocab_start + index
                if noisy:
                    word = T.alloc_var(
                        T.uint32,
                        init=T.call_extern(
                            "uint32", "vllm_randint32", word_seed[0], global_index
                        ),
                    )
                score[0] = -T.infinity(T.float32)
                if index < vocab:
                    score[0] = T.cast(raw[k], T.float32)
                    if noisy:
                        if (apply_temperature != 0) & (temp != T.float32(1)):
                            score[0] = T.call_extern(
                                "float32", "vllm_div_full", score[0], temp
                            )
                    score[0] = T.if_then_else(
                        T.cast(global_index, T.int64) == rejected[0],
                        -T.infinity(T.float32),
                        score[0],
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
    def resample_shard(
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
        dead_row_cut: T.Tensor((cut_len,), T.int32),
        apply_temperature: T.int32,
        vocab_start: T.int32,
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
            temp = T.alloc_var(T.float32, init=temperature[state])
            bonus = row == T.cast(end - 1, T.int64)
            if (temp != T.float32(0)) | bonus:
                draft = T.alloc_var(
                    T.int64,
                    init=T.cast(
                        drafts[T.min(row + 1, T.cast(end - 1, T.int64))], T.int64
                    ),
                )
                rejected[0] = T.if_then_else(
                    _ends_block(
                        bonus | (draft < 0), num_sampled[req], dead_row_cut, req, cut
                    ),
                    T.int64(-1),
                    draft,
                )
                seed = T.alloc_var(
                    T.int64,
                    init=T.if_then_else(
                        state >= 0, seeds[T.max(state, T.int64(0))], T.int64(0)
                    ),
                )
                word_seed[0] = _randint_at(seed, pos[row], pos_dtype, False)
                if T.if_then_else(state >= 0, temp, T.float32(0)) != T.float32(0):
                    passes(logits, raw, score, best, best_index, row, rejected,
                           word_seed, temp, apply_temperature, vocab_start, blk, tx, True)  # fmt: skip
                else:
                    passes(logits, raw, score, best, best_index, row, rejected,
                           word_seed, temp, apply_temperature, vocab_start, blk, tx, False)  # fmt: skip
                _block_argmax(
                    best, best_index, other, other_index, partial, partial_index, tx
                )
                if tx == 0:
                    local_argmax[req, blk] = T.cast(
                        vocab_start + blk * RESAMPLE_BLOCK + best_index[0], T.int64
                    )
                    local_max[req, blk] = best[0]
            elif tx == 0:
                local_argmax[req, blk] = T.int64(0)
                local_max[req, blk] = -T.infinity(T.float32)

    return resample_shard


def resample_shard(
    resampled_local_argmax: torch.Tensor,
    resampled_local_max: torch.Tensor,
    target_logits: torch.Tensor,
    num_sampled: torch.Tensor,
    cu_num_logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    draft_sampled: torch.Tensor,
    temperature: torch.Tensor,
    seed: torch.Tensor,
    pos: torch.Tensor,
    vocab_start: int,
    dead_row_cut: torch.Tensor | None = None,
    apply_temperature: bool = False,
) -> None:
    if num_sampled.shape[0] == 0:
        return
    kernel = _kernel(
        resample_shard_kernel,
        *resample_key(
            target_logits.dtype,
            expanded_idx_mapping.dtype,
            draft_sampled.dtype,
            pos.dtype,
            _triton_run(target_logits),
            dead_row_cut is not None,
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
        num_sampled if dead_row_cut is None else dead_row_cut,
        int(apply_temperature),
        int(vocab_start),
    )


@tilelang.jit(pass_configs=_PASS)
def resample_shard_pair_kernel(blocks: int):
    """Reduce each request's ``blocks`` shard winners to one 16-byte pair row
    ``[best_max, float32(best_id), 0, 0]``."""
    requests = T.dynamic("requests")
    id_stride, value_stride, pair_stride = T.dynamic(
        "id_stride, value_stride, pair_stride", "int64"
    )
    padded = 1 << max(blocks - 1, 0).bit_length()
    per = max(1, (blocks + 31) // 32)

    @T.prim_func
    def resample_shard_pair(
        local_pairs: T.StridedTensor((requests, 4), (pair_stride, 1), T.float32),
        local_ids: T.StridedTensor((requests, blocks), (id_stride, 1), T.int64),
        local_max: T.StridedTensor((requests, blocks), (value_stride, 1), T.float32),
    ):
        with T.Kernel(requests, threads=32) as row:
            thread = T.get_thread_binding()
            values = T.alloc_local((per,), T.float32)
            ids = T.alloc_local((per,), T.int64)
            for k in T.unroll(per):
                i = T.min(k * 32 + thread, blocks - 1)
                values[k] = local_max[row, i]
                ids[k] = local_ids[row, i]
            best = T.alloc_local((1,), T.float32)
            best_index = T.alloc_local((1,), T.int32)
            best_id = T.alloc_local((1,), T.int64)
            best[0] = -T.infinity(T.float32)
            best_index[0] = padded
            best_id[0] = T.int64(0)
            for k in T.unroll(per):
                idx = k * 32 + thread
                valid = idx < blocks
                val = T.if_then_else(
                    T.isnan(values[k]), -T.infinity(T.float32), values[k]
                )
                take = valid & (
                    (val > best[0]) | ((val == best[0]) & (idx < best_index[0]))
                )
                best_id[0] = T.if_then_else(take, ids[k], best_id[0])
                best_index[0] = T.if_then_else(take, idx, best_index[0])
                best[0] = T.if_then_else(take, val, best[0])
            for distance in (16, 8, 4, 2, 1):
                other_val = T.shfl_xor(best[0], distance)
                other_idx = T.shfl_xor(best_index[0], distance)
                other_id = T.shfl_xor(best_id[0], distance)
                take = (other_val > best[0]) | (
                    (other_val == best[0]) & (other_idx < best_index[0])
                )
                best_id[0] = T.if_then_else(take, other_id, best_id[0])
                best_index[0] = T.if_then_else(take, other_idx, best_index[0])
                best[0] = T.if_then_else(take, other_val, best[0])
            if thread == 0:
                local_pairs[row, 0] = best[0]
                local_pairs[row, 1] = T.cast(best_id[0], T.float32)
                local_pairs[row, 2] = T.float32(0)
                local_pairs[row, 3] = T.float32(0)

    return resample_shard_pair


@tilelang.jit(pass_configs=_PASS)
def insert_shard_resampled_kernel(
    tp_size: int,
    index_dtype: str,
    limited: bool = False,
):
    """Select the winning resampled or bonus token across ``tp_size`` gathered rank
    pairs and commit each request's count."""
    requests, width = T.dynamic("requests, width")
    mapping_len, state_len, cut_len = T.dynamic("mapping_len, state_len, cut_len")
    sampled_stride, pair_stride = T.dynamic("sampled_stride, pair_stride", "int64")

    @T.prim_func
    def insert_shard_resampled(
        sampled: T.StridedTensor((requests, width), (sampled_stride, 1), T.int64),
        counts: T.Tensor((requests,), T.int32),
        gathered_pairs: T.StridedTensor(
            (requests, 4 * tp_size), (pair_stride, 1), T.float32
        ),
        cu: T.Tensor((requests + 1,), T.int32),
        mapping: T.Tensor((mapping_len,), index_dtype),
        temperature: T.Tensor((state_len,), T.float32),
        dead_row_cut: T.Tensor((cut_len,), T.int32),
    ):
        with T.Kernel(requests, threads=1) as row:
            count = T.alloc_var(T.int32, init=counts[row])
            if limited:
                counts[row] = T.min(count + 1, dead_row_cut[row] + 1)
            else:
                counts[row] = count + 1
            position = cu[row] + count
            state = mapping[position]
            if (
                T.cast(temperature[state] != 0, T.int32)
                | T.cast(position == cu[row + 1] - 1, T.int32)
            ) != 0:
                best = T.alloc_var(T.float32, init=-T.infinity(T.float32))
                winner = T.alloc_var(T.int64, init=T.int64(0))
                for r in T.unroll(tp_size):
                    raw_val = gathered_pairs[row, r * 4]
                    value = T.if_then_else(
                        T.isnan(raw_val), -T.infinity(T.float32), raw_val
                    )
                    tok = T.cast(gathered_pairs[row, r * 4 + 1], T.int64)
                    if (r == 0) | (value > best):
                        best = value
                        winner = tok
                sampled[row, count] = winner

    return insert_shard_resampled


def verify_and_resample_from_shards(
    target_logits: torch.Tensor,
    draft_sampled: torch.Tensor,
    cu_num_logits: torch.Tensor,
    pos: torch.Tensor,
    idx_mapping: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    temperature: torch.Tensor,
    seed: torch.Tensor,
    num_speculative_steps: int,
    vocab_start: int,
    vocab_size: int,
    tp_size: int,
    all_gather_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    dead_row_cut: torch.Tensor | None = None,
    kernel_steps: int | None = None,
    apply_temperature: bool = False,
    local_stats: torch.Tensor | None = None,
    resampled_local_argmax: torch.Tensor | None = None,
    resampled_local_max: torch.Tensor | None = None,
    local_pairs: torch.Tensor | None = None,
    sampled: torch.Tensor | None = None,
    num_sampled: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shard-local block verification and resampling without all-gathering
    the full vocabulary logits across tensor-parallel ranks."""
    num_reqs = cu_num_logits.shape[0] - 1
    num_logits, shard_vocab = target_logits.shape
    steps = num_speculative_steps if kernel_steps is None else kernel_steps
    assert steps <= num_speculative_steps

    local_blocks = -(-max(shard_vocab, target_logits.stride(0)) // STATS_BLOCK)
    if local_stats is None:
        local_stats = target_logits.new_empty(
            num_logits, 4 * local_blocks, dtype=torch.float32
        )
    compute_local_stats_shard(
        local_stats,
        target_logits,
        draft_sampled,
        expanded_idx_mapping,
        expanded_local_pos,
        temperature,
        steps,
        vocab_start,
        apply_temperature,
    )

    if tp_size > 1 and all_gather_fn is not None:
        gathered_stats = all_gather_fn(local_stats)
    else:
        gathered_stats = local_stats

    cumulative_log_p = target_logits.new_empty(num_logits, dtype=torch.float32)
    if sampled is None:
        sampled = draft_sampled.new_empty(
            num_reqs, num_speculative_steps + 1, dtype=torch.int64
        )
    if num_sampled is None:
        num_sampled = sampled.new_empty(num_reqs, dtype=torch.int32)
    target_rejected_logsumexp = target_logits.new_empty(num_reqs, dtype=torch.float32)
    draft_rejected_logsumexp = target_logits.new_empty(num_reqs, dtype=torch.float32)
    verify_shard(
        sampled[:, : steps + 1],
        num_sampled,
        target_rejected_logsumexp,
        draft_rejected_logsumexp,
        cumulative_log_p,
        gathered_stats,
        draft_sampled,
        cu_num_logits,
        idx_mapping,
        temperature,
        seed,
        pos,
        steps,
        vocab_size,
        dead_row_cut,
    )

    resample_blocks = -(-max(shard_vocab, target_logits.stride(0)) // RESAMPLE_BLOCK)
    if resampled_local_argmax is None:
        resampled_local_argmax = target_logits.new_empty(
            num_reqs, resample_blocks, dtype=torch.int64
        )
    if resampled_local_max is None:
        resampled_local_max = target_logits.new_empty(
            num_reqs, resample_blocks, dtype=torch.float32
        )
    resample_shard(
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
        vocab_start,
        dead_row_cut,
        apply_temperature,
    )

    if local_pairs is None:
        local_pairs = target_logits.new_empty(num_reqs, 4, dtype=torch.float32)
    if num_reqs > 0:
        pair_fn = _kernel(resample_shard_pair_kernel, resample_blocks)
        pair_fn(local_pairs, resampled_local_argmax, resampled_local_max)

    if tp_size > 1 and all_gather_fn is not None:
        gathered_pairs = all_gather_fn(local_pairs)
    else:
        gathered_pairs = local_pairs

    if num_reqs > 0:
        insert_fn = _kernel(
            insert_shard_resampled_kernel,
            int(tp_size),
            _DTYPES[expanded_idx_mapping.dtype],
            dead_row_cut is not None,
        )
        insert_fn(
            sampled,
            num_sampled,
            gathered_pairs,
            cu_num_logits,
            expanded_idx_mapping.contiguous(),
            temperature,
            num_sampled if dead_row_cut is None else dead_row_cut,
        )
    return sampled, num_sampled


def resample_from_shards(
    target_logits: torch.Tensor,
    draft_sampled: torch.Tensor,
    cu_num_logits: torch.Tensor,
    pos: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    seed: torch.Tensor,
    vocab_start: int,
    tp_size: int,
    all_gather_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    dead_row_cut: torch.Tensor | None = None,
    apply_temperature: bool = False,
    resampled_local_argmax: torch.Tensor | None = None,
    resampled_local_max: torch.Tensor | None = None,
    local_pairs: torch.Tensor | None = None,
    sampled: torch.Tensor | None = None,
    num_sampled: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shard-local Gumbel/greedy sampling for 1-token-per-request steps."""
    num_reqs = cu_num_logits.shape[0] - 1
    _, shard_vocab = target_logits.shape
    if sampled is None:
        sampled = draft_sampled.new_empty(num_reqs, 1, dtype=torch.int64)
    if num_sampled is None:
        num_sampled = sampled.new_zeros(num_reqs, dtype=torch.int32)
    else:
        num_sampled.zero_()

    resample_blocks = -(-max(shard_vocab, target_logits.stride(0)) // RESAMPLE_BLOCK)
    if resampled_local_argmax is None:
        resampled_local_argmax = target_logits.new_empty(
            num_reqs, resample_blocks, dtype=torch.int64
        )
    if resampled_local_max is None:
        resampled_local_max = target_logits.new_empty(
            num_reqs, resample_blocks, dtype=torch.float32
        )
    resample_shard(
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
        vocab_start,
        dead_row_cut,
        apply_temperature,
    )

    if local_pairs is None:
        local_pairs = target_logits.new_empty(num_reqs, 4, dtype=torch.float32)
    if num_reqs > 0:
        pair_fn = _kernel(resample_shard_pair_kernel, resample_blocks)
        pair_fn(local_pairs, resampled_local_argmax, resampled_local_max)

    if tp_size > 1 and all_gather_fn is not None:
        gathered_pairs = all_gather_fn(local_pairs)
    else:
        gathered_pairs = local_pairs

    if num_reqs > 0:
        insert_fn = _kernel(
            insert_shard_resampled_kernel,
            int(tp_size),
            _DTYPES[expanded_idx_mapping.dtype],
            dead_row_cut is not None,
        )
        insert_fn(
            sampled,
            num_sampled,
            gathered_pairs,
            cu_num_logits,
            expanded_idx_mapping.contiguous(),
            temperature,
            num_sampled if dead_row_cut is None else dead_row_cut,
        )
    return sampled, num_sampled


def shard_rejection_calls(
    logits_dtypes: Sequence[torch.dtype],
    shard_vocab_size: int,
    tp_size: int,
    step_widths: Sequence[int],
    cuts: Sequence[bool] = (False,),
) -> list:
    local_blocks = -(-shard_vocab_size // STATS_BLOCK)
    total_blocks = tp_size * local_blocks
    resample_blocks = -(-shard_vocab_size // RESAMPLE_BLOCK)
    calls = [
        _call(resample_shard_pair_kernel, resample_blocks),
    ]
    for cut in cuts:
        calls.append(
            _call(
                insert_shard_resampled_kernel,
                int(tp_size),
                _DTYPES[RUNNER_IDX_DTYPE],
                cut,
            )
        )
        for steps in dict.fromkeys(step_widths):
            calls.append(
                _call(
                    verify_shard_kernel,
                    *verify_shard_key(
                        torch.int32,
                        RUNNER_IDX_DTYPE,
                        torch.int64,
                        total_blocks,
                        steps,
                        cut,
                    ),
                )
            )
    for dtype in dict.fromkeys(logits_dtypes):
        for run in _runs_for_strides(dtype, (shard_vocab_size,)):
            calls.append(
                _call(
                    local_stats_shard_kernel,
                    *local_stats_shard_key(
                        dtype, torch.int32, RUNNER_IDX_DTYPE, torch.int32, run
                    ),
                )
            )
            calls += [
                _call(
                    resample_shard_kernel,
                    *resample_key(
                        dtype, RUNNER_IDX_DTYPE, torch.int32, torch.int64, run, cut
                    ),
                )
                for cut in cuts
            ]
    return calls


def register_shard_block_verification(
    logits_dtypes: Sequence[torch.dtype],
    shard_vocab_size: int,
    tp_size: int,
    num_speculative_steps: int,
    step_widths: Sequence[int] = (),
    dead_rows: bool = False,
) -> None:
    cuts = (False, True) if dead_rows else (False,)
    widths = tuple(dict.fromkeys(step_widths or (num_speculative_steps,)))
    declare(
        lambda: shard_rejection_calls(
            logits_dtypes, shard_vocab_size, tp_size, widths, cuts
        )
    )
