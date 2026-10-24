# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TileLang kernels of the V2 model runner's sampling.

Each kernel is the contract of the Triton kernel it replaces, bit for bit, NaN
scores included (the Triton kernels stay importable as the tests' reference):

- ``gumbel_sample``: ``gumbel._gumbel_sample_kernel``
- ``apply_temperature``: ``gumbel._temperature_kernel``

The random stream is Triton 3.5's, so seeded outputs do not move: ``tl.randint``
is Philox-4x32 with ten rounds (the 64-bit seed is the key, low word first; the
offset's low and high words are the first two counter words), and ``tl.rand`` maps
the word to ``[0, 1)`` with ``uint_to_uniform_float``. The FP32 Gumbel noise
``-log(-log1p(-u))`` is libdevice's ``__nv_log1pf`` and ``__nv_logf`` as Triton
inlines them from its bundled libdevice, with flush-to-zero FMAs: the header below
carries those instructions in inline PTX, because CUDA's own libdevice is another
implementation. Each branch of theirs becomes a select, which computes the same
value. The FP32 division by the temperature is Triton's ``/``, ``div.full.f32``.

A block's argmax is ``tl.max(..., return_indices=True)`` in Triton's order at four
warps. Triton 3.7 (the image's) lays the block out by the load's alignment: when the
logits' base and row stride are divisible by 16 (its argument specialization), each
thread holds runs of one 16-byte vector (four FP32 or eight 16-bit values), runs 512
or 1024 apart; otherwise one element, 128 apart. A thread keeps its running pair
while it is ``>=`` the next of its elements, in order; lanes then exchange pairs at
XOR distances 16, 8, 4, 2 and 1, each keeping its own pair while it is greater, or
equal with the lower index; lane 0 of each warp publishes, and threads 0-3 finish at
distances 2 and 1. A comparison with NaN is false, so a NaN score gives way to
whatever it meets first, and the order decides which index a block with NaN
reports: the launch picks the layout by Triton's rule (``_triton_run``).

The draw is FP32 only: the TileLang sampler does not implement
``use_fp64_gumbel``, which the runner rejects.
"""

# No `from __future__ import annotations`: TileLang reads the prim_func
# annotations eagerly, and the symbolic shapes must resolve.
from collections.abc import Callable, Sequence
from typing import Any

import torch

from vllm.tilelang_utils import T, tilelang

_PASS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    # Every access is guarded by the kernel's own masks, as in Triton.
    tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
}

# gumbel._GUMBEL_BLOCK_SIZE, and Triton's default four warps.
GUMBEL_BLOCK = 1024
_THREADS = 128
_WARPS = _THREADS // 32
_PASSES = GUMBEL_BLOCK // _THREADS
# gumbel._TEMPERATURE_BLOCK_SIZE.
_TEMPERATURE_BLOCK = 8192
# Temperature's threads a block: fewer serial runs a thread than at 128, the shape
# Triton reaches with its loads all issued up front.
_TEMPERATURE_THREADS = 256
# gumbel._DRAFT_NOISE_SALT.
_DRAFT_NOISE_SALT = 1 << 30

_DTYPES = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
    torch.int32: "int32",
    torch.int64: "int64",
}

_HEADER = r"""
// Triton 3.5 philox_impl (triton/language/random.py) on 32-bit words, ten
// rounds; the first word.
__device__ __forceinline__ unsigned int vllm_philox(unsigned int c0, unsigned int c1,
                                                   unsigned int c2, unsigned int c3,
                                                   unsigned int k0, unsigned int k1) {
#pragma unroll
  for (int round = 0; round < 10; ++round) {
    const unsigned int a = c0;
    const unsigned int b = c2;
    c0 = __umulhi(0xCD9E8D57u, b) ^ c1 ^ k0;
    c2 = __umulhi(0xD2511F53u, a) ^ c3 ^ k1;
    c1 = 0xCD9E8D57u * b;
    c3 = 0xD2511F53u * a;
    k0 += 0x9E3779B9u;
    k1 += 0xBB67AE85u;
  }
  return c0;
}

// tl.randint(seed, offset) for an int64 seed and an int64 offset: the offset's
// low word, then its arithmetic high word.
__device__ __forceinline__ unsigned int vllm_randint64(long long seed,
                                                      long long offset) {
  const unsigned long long key = (unsigned long long)seed;
  return vllm_philox((unsigned int)offset, (unsigned int)(offset >> 32), 0u, 0u,
                     (unsigned int)key, (unsigned int)(key >> 32));
}

// tl.randint(seed, offset) for an int64 seed and an int32 offset: Triton gives a
// 32-bit offset no high word.
__device__ __forceinline__ unsigned int vllm_randint64_offset32(long long seed,
                                                               unsigned int offset) {
  const unsigned long long key = (unsigned long long)seed;
  return vllm_philox(offset, 0u, 0u, 0u, (unsigned int)key, (unsigned int)(key >> 32));
}

// tl.randint(seed, offset) for a uint32 seed (tl.randint's own result, widened
// with zeros) and an int32 offset (no high word).
__device__ __forceinline__ unsigned int vllm_randint32(unsigned int seed, int offset) {
  return vllm_philox((unsigned int)offset, 0u, 0u, 0u, seed, 0u);
}

// gumbel.py's FP32 noise of one Philox word, as Triton 3.5 compiles
// tl_rand32(includes_zero=False) and -tl.log(-tldevice.log1p(-u)) for sm_121a:
// uint_to_uniform_float, the clamp to _TL_RAND_MIN (0f2FFFFFFF), then libdevice's
// __nv_log1pf of 0 - u and __nv_logf of 0 - that, negated.
__device__ __forceinline__ float vllm_gumbel_noise(unsigned int word) {
  float noise;
  asm("{\n"
      ".reg .pred fin, inside, zero, tiny;\n"
      ".reg .b32 x, s, u, v, t, e, m, k, a, r, f, p, q, l, w, y, z, o;\n"
      // uint_to_uniform_float: where(x < 0, -x - 1, x) * 4.6566127342e-10
      "shr.s32 s, %1, 31;\n"
      "xor.b32 x, s, %1;\n"
      "cvt.rn.f32.s32 u, x;\n"
      "mul.rn.f32 u, u, 0f2FFFFFFF;\n"
      "max.f32 u, u, 0f2FFFFFFF;\n"
      // __nv_log1pf(v), v = 0 - u
      "sub.rn.f32 v, 0f00000000, u;\n"
      "add.rz.ftz.f32 t, v, 0f3F800000;\n"
      "add.s32 e, t, -1061158912;\n"
      "and.b32 e, e, -8388608;\n"
      "sub.s32 m, v, e;\n"
      "sub.s32 k, 1082130432, e;\n"
      "fma.rn.ftz.f32 a, 0f3E800000, k, 0fBF800000;\n"
      "add.rn.f32 r, a, m;\n"
      "cvt.rn.f32.s32 f, e;\n"
      "mul.rn.f32 f, f, 0f34000000;\n"
      "fma.rn.ftz.f32 p, 0fBD39BF78, r, 0f3DD80012;\n"
      "fma.rn.ftz.f32 p, p, r, 0fBE0778E0;\n"
      "fma.rn.ftz.f32 p, p, r, 0f3E146475;\n"
      "fma.rn.ftz.f32 p, p, r, 0fBE2A68DD;\n"
      "fma.rn.ftz.f32 p, p, r, 0f3E4CAF9E;\n"
      "fma.rn.ftz.f32 p, p, r, 0fBE800042;\n"
      "fma.rn.ftz.f32 p, p, r, 0f3EAAAAE6;\n"
      "fma.rn.ftz.f32 p, p, r, 0fBF000000;\n"
      "mul.rn.f32 q, r, p;\n"
      "fma.rn.ftz.f32 q, q, r, r;\n"
      "fma.rn.ftz.f32 l, f, 0f3F317218, q;\n"
      // Not a finite non-negative v: inside [-1, -0] keep, else fma(v, inf, inf);
      // then a zero v gives -0.
      "setp.lt.u32 fin, v, 2139095040;\n"
      "setp.lt.s32 inside, v, -1082130431;\n"
      "fma.rn.ftz.f32 w, v, 0f7F800000, 0f7F800000;\n"
      "selp.f32 w, l, w, inside;\n"
      "setp.eq.f32 zero, v, 0f00000000;\n"
      "selp.f32 w, 0f80000000, w, zero;\n"
      "selp.f32 l, l, w, fin;\n"
      // __nv_logf(y), y = 0 - log1p(v): subnormal y scaled by 2^23
      "sub.rn.f32 y, 0f00000000, l;\n"
      "setp.lt.f32 tiny, y, 0f00800000;\n"
      "mul.rn.f32 z, y, 0f4B000000;\n"
      "selp.f32 z, z, y, tiny;\n"
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
      // Not finite non-negative: fma(z, inf, inf). The noise is 0 - logf(y), or
      // +inf where z is zero.
      "setp.lt.u32 fin, z, 2139095040;\n"
      "fma.rn.ftz.f32 w, z, 0f7F800000, 0f7F800000;\n"
      "selp.f32 l, l, w, fin;\n"
      "sub.rn.f32 l, 0f00000000, l;\n"
      "setp.eq.f32 zero, z, 0f00000000;\n"
      "selp.f32 %0, 0f7F800000, l, zero;\n"
      "}"
      : "=f"(noise)
      : "r"(word));
  return noise;
}

// A logit plus the noise of its word: Triton's FP32 add.
__device__ __forceinline__ float vllm_gumbel_score(float logit, unsigned int word) {
  const float noise = vllm_gumbel_noise(word);
  float score;
  asm("add.rn.f32 %0, %1, %2;" : "=f"(score) : "f"(logit), "f"(noise));
  return score;
}

// Triton's FP32 `/`: div.full.f32.
__device__ __forceinline__ float vllm_div_full(float x, float y) {
  float q;
  asm("div.full.f32 %0, %1, %2;" : "=f"(q) : "f"(x), "f"(y));
  return q;
}

// dspark.greedy._first_argmax: keep (av, ai) over (bv, bi) when av is NaN and bv
// is not or has a higher index, or neither is NaN and av > bv, or av == bv with
// ai < bi: torch.argmax's first maximum, NaN above everything. Bitwise, not
// short-circuit, operators: every comparison is evaluated and combined in
// predicates, as Triton's tl.where combine is, instead of a branch for each.
__device__ __forceinline__ int vllm_first_argmax_keep(float av, int ai, float bv,
                                                      int bi) {
  const bool a_nan = av != av;
  const bool b_nan = bv != bv;
  return (a_nan & (!b_nan | (ai < bi))) |
         (!a_nan & !b_nan & ((av > bv) | ((av == bv) & (ai < bi))));
}

// tl.max(..., return_indices=True)'s combine (tie break to the left): keep
// (v, i) over (w, j) when v > w, or v == w and i < j. Comparisons with NaN are
// false. Compiled here, out of the TileLang simplifier's reach; bitwise
// operators, as above.
__device__ __forceinline__ int vllm_argmax_keep(float v, int i, float w, int j) {
  return (v > w) | ((v == w) & (i < j));
}
"""


_KERNELS: dict[tuple[int, tuple[Any, ...]], Any] = {}


def _kernel(factory: Any, *key: Any) -> Any:
    """``factory(*key)``, resolved once: before the kernel cache freezes it
    compiles on a miss; after, an undeclared key raises by name."""
    kernel = _KERNELS.get((id(factory), key))
    if kernel is None:
        kernel = _KERNELS[(id(factory), key)] = factory(*key)
    return kernel


def _triton_run(rows: torch.Tensor) -> int:
    """Elements a thread holds contiguously in Triton 3.7's layout for a load from
    ``rows``: one 16-byte vector when the base address and the row stride are both
    divisible by 16 (the specialization its coalescing pass reads), else one."""
    if rows.data_ptr() % 16 == 0 and rows.stride(0) % 16 == 0:
        return 16 // rows.element_size()
    return 1


def _element(j, tx, run: int):
    """The block-local index of thread ``tx``'s ``j``-th element: runs of ``run``
    elements, ``128 * run`` apart."""
    return (j // run) * (_THREADS * run) + run * tx + j % run


@T.macro
def _load_run(src, raw, row, col0, vocab, run):
    """``raw[k] = src[row, col0 + k]`` for the run's columns below ``vocab``: one
    vector load when the run is whole (the launch guaranteed its alignment)."""
    if run == 1:
        if col0 < vocab:
            raw[0] = src[row, col0]
    else:
        if col0 + run <= vocab:
            for k in T.vectorized(run):
                raw[k] = src[row, col0 + k]
        else:
            for k in T.unroll(run):
                if col0 + k < vocab:
                    raw[k] = src[row, col0 + k]


@T.macro
def _store_run(dst, raw, row, col0, vocab, run):
    """``_load_run``'s store."""
    if run == 1:
        if col0 < vocab:
            dst[row, col0] = raw[0]
    else:
        if col0 + run <= vocab:
            for k in T.vectorized(run):
                dst[row, col0 + k] = raw[k]
        else:
            for k in T.unroll(run):
                if col0 + k < vocab:
                    dst[row, col0 + k] = raw[k]


@T.macro
def _load_run_at(src, raw, base, row, col0, vocab, run):
    """``_load_run`` into ``raw[base:base + run]``."""
    if run == 1:
        if col0 < vocab:
            raw[base] = src[row, col0]
    else:
        if col0 + run <= vocab:
            for k in T.vectorized(run):
                raw[base + k] = src[row, col0 + k]
        else:
            for k in T.unroll(run):
                if col0 + k < vocab:
                    raw[base + k] = src[row, col0 + k]


@T.macro
def _store_run_at(dst, raw, base, row, col0, vocab, run):
    """``_store_run`` from ``raw[base:base + run]``."""
    if run == 1:
        if col0 < vocab:
            dst[row, col0] = raw[base]
    else:
        if col0 + run <= vocab:
            for k in T.vectorized(run):
                dst[row, col0 + k] = raw[base + k]
        else:
            for k in T.unroll(run):
                if col0 + k < vocab:
                    dst[row, col0 + k] = raw[base + k]


@T.macro
def _keep_or_take(best, best_index, value, index):
    """``(best, best_index)`` = Triton's argmax combine of itself and ``(value,
    index)``, by selects (Triton's tl.where), the decision held in a register so
    both selects read the pair as it was."""
    keep = T.alloc_var(
        T.int32,
        init=T.call_extern(
            "int32", "vllm_argmax_keep", best[0], best_index[0], value, index
        ),
    )
    best_index[0] = T.if_then_else(keep != 0, best_index[0], index)
    best[0] = T.if_then_else(keep != 0, best[0], value)


@T.macro
def _exchange(best, best_index, other, other_index, distance):
    """Combine with the lane at XOR ``distance``."""
    other[0] = T.shfl_xor(best[0], distance)
    other_index[0] = T.shfl_xor(best_index[0], distance)
    _keep_or_take(best, best_index, other[0], other_index[0])


@T.macro
def _block_argmax(best, best_index, other, other_index, partial, partial_index, tx):
    """Reduce each thread's ``(best, best_index)`` across the block in Triton's
    order into thread 0's: lanes at XOR distances 16 to 1, lane 0 of each warp into
    ``partial``, then threads 0-3 at distances 2 and 1 (the other lanes exchange
    what they hold, as in Triton, and their results are unused)."""
    _exchange(best, best_index, other, other_index, 16)
    _exchange(best, best_index, other, other_index, 8)
    _exchange(best, best_index, other, other_index, 4)
    _exchange(best, best_index, other, other_index, 2)
    _exchange(best, best_index, other, other_index, 1)
    if tx % 32 == 0:
        partial[tx // 32] = best[0]
        partial_index[tx // 32] = best_index[0]
    T.sync_threads()
    if tx < _WARPS:
        best[0] = partial[tx]
        best_index[0] = partial_index[tx]
    _exchange(best, best_index, other, other_index, 2)
    _exchange(best, best_index, other, other_index, 1)


# --- Gumbel-max sampling ------------------------------------------------------


def _randint_at(seed, position, pos_dtype: str, salted: bool):
    """``tl.randint(seed, position)`` (``position + _DRAFT_NOISE_SALT`` when
    ``salted``) for an int64 seed and a position of ``pos_dtype``: an int64 offset
    gives its two words, an int32 one its own word, the salt added in 32 bits."""
    if pos_dtype == "int64":
        offset = position + T.int64(_DRAFT_NOISE_SALT) if salted else position
        return T.call_extern("uint32", "vllm_randint64", seed, offset)
    word = T.cast(position, "uint32")
    if salted:
        word = word + T.uint32(_DRAFT_NOISE_SALT)
    return T.call_extern("uint32", "vllm_randint64_offset32", seed, word)


@tilelang.jit(pass_configs=_PASS)
def gumbel_sample_kernel(
    logits_dtype: str,
    idx_dtype: str,
    apply_temperature: bool,
    is_drafting: bool,
    cache_dtype: str | None,
    col_dtype: str | None,
    per_token_col: bool,
    run: int,
    pos_dtype: str = "int64",
):
    """``gumbel._gumbel_sample_kernel``: one block of 128 threads per
    ``(token, vocabulary block of 1024)``; each block's argmax index (global, int64)
    and score, a thread's elements in runs of ``run`` (``_triton_run``). With a
    logits cache (``cache_dtype``), the logits before the temperature are stored at
    the step's column (``col_dtype``, one per token with ``per_token_col``) and the
    seed takes the column in its high word. Positions of ``pos_dtype`` place the
    draw as Triton's ``tl.randint`` does: an int32 position (salted in 32 bits when
    drafting) is the offset's only word."""
    tokens, vocab, blocks = T.dynamic("tokens, vocab, blocks")
    # Inputs may be longer than the batch: each has its own length.
    idx_len, pos_len, seed_len, temp_len = T.dynamic(
        "idx_len, pos_len, seed_len, temp_len"
    )
    logits_stride, argmax_stride, max_stride = T.dynamic(
        "logits_stride, argmax_stride, max_stride", "int64"
    )
    cached = cache_dtype is not None
    cache_reqs, cache_cols, cache_vocab, num_cols = T.dynamic(
        "cache_reqs, cache_cols, cache_vocab, num_cols"
    )
    cache_stride_0, cache_stride_1 = T.dynamic(
        "cache_stride_0, cache_stride_1", "int64"
    )
    if not cached:
        # Placeholders the variant never reads.
        cache_dtype, col_dtype = "float32", "int64"

    @T.macro
    def passes(logits, raw, score, best, best_index, cache, req, col, word_seed, temp,
               t, blk, tx, noisy):  # fmt: skip
        """The thread's elements, combined into ``(best, best_index)``. With
        ``noisy`` (the token's temperature is not 0) each element's Philox word is
        drawn unconditionally, ahead of its bound check, as Triton draws a whole
        block: the key schedule, the same for every element of the token, is then
        computed once a thread instead of once an element."""
        for r in T.unroll(_PASSES // run):
            col0 = blk * GUMBEL_BLOCK + r * (_THREADS * run) + run * tx
            _load_run(logits, raw, t, col0, vocab, run)
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
                    if cached:  # noqa: SIM102
                        if req[0] >= 0:
                            cache[req[0], col[0], index] = T.cast(score[0], cache_dtype)
                    if noisy:
                        if apply_temperature:
                            score[0] = T.call_extern(
                                "float32", "vllm_div_full", score[0], temp[0]
                            )
                        score[0] = T.call_extern(
                            "float32", "vllm_gumbel_score", score[0], word
                        )
                if r * run + k == 0:
                    best[0] = score[0]
                    best_index[0] = local
                else:
                    _keep_or_take(best, best_index, score[0], local)

    @T.prim_func
    def gumbel_sample(
        local_argmax: T.StridedTensor((tokens, blocks), (argmax_stride, 1), T.int64),
        local_max: T.StridedTensor((tokens, blocks), (max_stride, 1), T.float32),
        logits: T.StridedTensor((tokens, vocab), (logits_stride, 1), logits_dtype),
        idx_mapping: T.Tensor((idx_len,), idx_dtype),
        seeds: T.Tensor((seed_len,), T.int64),
        pos: T.Tensor((pos_len,), pos_dtype),
        temperature: T.Tensor((temp_len,), T.float32),
        cache: T.StridedTensor(
            (cache_reqs, cache_cols, cache_vocab),
            (cache_stride_0, cache_stride_1, 1),
            cache_dtype,
        ),
        cache_col: T.Tensor((num_cols,), col_dtype),
    ):
        with T.Kernel(tokens, blocks, threads=_THREADS) as (t, blk):
            T.import_source(_HEADER)
            tx = T.get_thread_binding()
            req = T.alloc_local((1,), T.int64)
            temp = T.alloc_local((1,), T.float32)
            seed = T.alloc_local((1,), T.int64)
            col = T.alloc_local((1,), T.int64)
            word_seed = T.alloc_local((1,), T.uint32)
            raw = T.alloc_local((run,), logits_dtype)
            score = T.alloc_local((1,), T.float32)
            best = T.alloc_local((1,), T.float32)
            best_index = T.alloc_local((1,), T.int32)
            other = T.alloc_local((1,), T.float32)
            other_index = T.alloc_local((1,), T.int32)
            partial = T.alloc_shared((_WARPS,), T.float32)
            partial_index = T.alloc_shared((_WARPS,), T.int32)
            if run > 1:
                # The launch checked it: a 16-byte base and a stride of 16 elements.
                T.assume(logits_stride % 16 == 0)

            req[0] = T.cast(idx_mapping[t], T.int64)
            temp[0] = T.float32(0)
            seed[0] = T.int64(0)
            if req[0] >= 0:
                temp[0] = temperature[req[0]]
                seed[0] = seeds[req[0]]
            if cached:
                if per_token_col:
                    col[0] = T.cast(cache_col[t], T.int64)
                else:
                    col[0] = T.cast(cache_col[0], T.int64)
                # gumbel.spec_step_seed
                seed[0] = seed[0] ^ T.shift_left(col[0], T.int64(32))
            word_seed[0] = _randint_at(seed[0], pos[t], pos_dtype, is_drafting)
            # One test of the temperature for the whole block, as Triton's kernel.
            if temp[0] != T.float32(0):
                passes(logits, raw, score, best, best_index, cache, req, col, word_seed,
                       temp, t, blk, tx, True)  # fmt: skip
            else:
                passes(logits, raw, score, best, best_index, cache, req, col, word_seed,
                       temp, t, blk, tx, False)  # fmt: skip
            _block_argmax(
                best, best_index, other, other_index, partial, partial_index, tx
            )
            if tx == 0:
                local_argmax[t, blk] = T.cast(
                    blk * GUMBEL_BLOCK + best_index[0], T.int64
                )
                local_max[t, blk] = best[0]

    return gumbel_sample


def gumbel_sample_key(
    logits_dtype: torch.dtype,
    idx_dtype: torch.dtype,
    apply_temperature: bool,
    is_drafting: bool,
    cache_dtype: torch.dtype | None = None,
    col_dtype: torch.dtype | None = None,
    per_token_col: bool = False,
    run: int = 1,
    pos_dtype: torch.dtype = torch.int64,
) -> tuple:
    cached = cache_dtype is not None
    return (
        _DTYPES[logits_dtype],
        _DTYPES[idx_dtype],
        apply_temperature,
        is_drafting,
        _DTYPES[cache_dtype] if cached else None,
        _DTYPES[col_dtype] if cached else None,
        per_token_col and cached,
        run,
        _DTYPES[pos_dtype],
    )


def gumbel_sample(
    logits: torch.Tensor,  # [num_tokens, vocab_size]
    expanded_idx_mapping: torch.Tensor,  # [num_tokens]
    temperature: torch.Tensor,  # [max_num_reqs]
    seed: torch.Tensor,  # [max_num_reqs]
    pos: torch.Tensor,  # [num_tokens]
    apply_temperature: bool,
    is_drafting: bool,
    logits_cache: torch.Tensor | None = None,  # [max_num_reqs, num_cols, vocab_size]
    logits_cache_col: torch.Tensor | None = None,  # scalar or [num_tokens]
    use_fp64: bool = False,
) -> torch.Tensor:
    """``gumbel.gumbel_sample``: each token's sampled id ([num_tokens], int64)."""
    if use_fp64:
        raise ValueError("the TileLang sampler draws in FP32 only (use_fp64_gumbel)")
    expanded_idx_mapping = expanded_idx_mapping.contiguous()
    pos = pos.contiguous()
    num_tokens, vocab_size = logits.shape
    num_blocks = -(-vocab_size // GUMBEL_BLOCK)
    local_argmax = logits.new_empty(num_tokens, num_blocks, dtype=torch.int64)
    local_max = logits.new_empty(num_tokens, num_blocks, dtype=torch.float32)
    cached = logits_cache is not None
    if cached:
        assert logits_cache_col is not None
        assert logits_cache.size(-1) >= vocab_size, (
            f"draft logits cache vocab dim ({logits_cache.size(-1)}) is narrower "
            f"than the sampled logits ({vocab_size}). Cached logits would be "
            "truncated."
        )
        logits_cache_col = logits_cache_col.contiguous()
    per_token_col = cached and logits_cache_col.dim() > 0
    if num_tokens > 0:
        kernel = _kernel(
            gumbel_sample_kernel,
            *gumbel_sample_key(
                logits.dtype,
                expanded_idx_mapping.dtype,
                apply_temperature,
                is_drafting,
                logits_cache.dtype if cached else None,
                logits_cache_col.dtype if cached else None,
                per_token_col,
                _triton_run(logits),
                pos.dtype,
            ),
        )
        kernel(
            local_argmax,
            local_max,
            logits,
            expanded_idx_mapping,
            seed,
            pos,
            temperature,
            # Placeholders of the declared types when there is no cache.
            logits_cache if cached else local_max.unsqueeze(1),
            logits_cache_col.view(-1) if cached else local_argmax.view(-1),
        )
    # NOTE(woosuk): Use int64 for later indexing.
    max_block_idx = local_max.argmax(dim=-1, keepdim=True)
    return local_argmax.gather(dim=-1, index=max_block_idx).view(-1)


# --- Sink detector ------------------------------------------------------------

# sink._CANDIDATE_BLOCK_SIZE
SINK_CANDIDATE_BLOCK = 1024


@tilelang.jit(pass_configs=_PASS)
def sink_candidates_kernel(
    logits_dtype: str,
    idx_dtype: str,
    num_candidates: int,
    mask_rejected_draft: bool,
    run: int,
    pos_dtype: str = "int64",
):
    """``sink._sink_candidates_kernel``: one block of 128 threads per ``(request,
    vocabulary block of 1024)``, a thread's elements in runs of ``run``
    (``_triton_run``). The scores are the request's last sampled row's logits (with a
    rejected one-hot draft masked when ``mask_rejected_draft``) plus the draw's
    noise; the block's ``num_candidates`` best, in descending order, each found by
    Triton's argmax and then masked."""
    reqs, rows, vocab, candidates = T.dynamic("reqs, rows, vocab, candidates")
    # Inputs may be longer than the batch: each has its own length.
    cu_len, num_len, sampled_rows, sampled_width = T.dynamic(
        "cu_len, num_len, sampled_rows, sampled_width"
    )
    drafts_len, idx_len, temp_len, seed_len, pos_len = T.dynamic(
        "drafts_len, idx_len, temp_len, seed_len, pos_len"
    )
    logits_stride, candidates_stride, sampled_stride = T.dynamic(
        "logits_stride, candidates_stride, sampled_stride", "int64"
    )

    @T.macro
    def passes(logits, raw, scores, row, rejected, verified, word_seed, blk, tx, noisy):
        """The thread's scores; with ``noisy`` each element's Philox word is drawn
        unconditionally, ahead of its bound check (the Gumbel sampler's
        ``passes``)."""
        for r in T.unroll(_PASSES // run):
            col0 = blk * SINK_CANDIDATE_BLOCK + r * (_THREADS * run) + run * tx
            _load_run(logits, raw, row[0], col0, vocab, run)
            for k in T.unroll(run):
                index = col0 + k
                if noisy:
                    word = T.alloc_var(
                        T.uint32,
                        init=T.call_extern(
                            "uint32", "vllm_randint32", word_seed[0], index
                        ),
                    )
                scores[r * run + k] = -T.infinity(T.float32)
                if index < vocab:
                    scores[r * run + k] = T.cast(raw[k], T.float32)
                    if (rejected[0] != 0) & (index == verified[0]):
                        scores[r * run + k] = -T.infinity(T.float32)
                    if noisy:
                        scores[r * run + k] = T.call_extern(
                            "float32", "vllm_gumbel_score", scores[r * run + k], word
                        )

    @T.prim_func
    def sink_candidates(
        values: T.StridedTensor((reqs, candidates), (candidates_stride, 1), T.float32),
        indices: T.StridedTensor((reqs, candidates), (candidates_stride, 1), T.int64),
        logits: T.StridedTensor((rows, vocab), (logits_stride, 1), logits_dtype),
        cu_num_logits: T.Tensor((cu_len,), T.int32),
        num_sampled: T.Tensor((num_len,), T.int32),
        sampled: T.StridedTensor(
            (sampled_rows, sampled_width), (sampled_stride, 1), T.int64
        ),
        drafts: T.Tensor((drafts_len,), T.int32),
        idx_mapping: T.Tensor((idx_len,), idx_dtype),
        temperature: T.Tensor((temp_len,), T.float32),
        seeds: T.Tensor((seed_len,), T.int64),
        pos: T.Tensor((pos_len,), pos_dtype),
    ):
        with T.Kernel(
            reqs, T.ceildiv(vocab, SINK_CANDIDATE_BLOCK), threads=_THREADS
        ) as (req, blk):
            T.import_source(_HEADER)
            tx = T.get_thread_binding()
            row = T.alloc_local((1,), T.int64)
            local = T.alloc_local((1,), T.int32)
            state = T.alloc_local((1,), T.int64)
            temp = T.alloc_local((1,), T.float32)
            rejected = T.alloc_local((1,), T.int32)
            verified = T.alloc_local((1,), T.int32)
            word_seed = T.alloc_local((1,), T.uint32)
            raw = T.alloc_local((run,), logits_dtype)
            scores = T.alloc_local((_PASSES,), T.float32)
            best = T.alloc_local((1,), T.float32)
            best_index = T.alloc_local((1,), T.int32)
            other = T.alloc_local((1,), T.float32)
            other_index = T.alloc_local((1,), T.int32)
            partial = T.alloc_shared((_WARPS,), T.float32)
            partial_index = T.alloc_shared((_WARPS,), T.int32)
            chosen = T.alloc_shared((1,), T.int32)
            if run > 1:
                # The launch checked it: a 16-byte base and a stride of 16 elements.
                T.assume(logits_stride % 16 == 0)

            local[0] = T.max(num_sampled[req] - 1, 0)
            # The row that produced the request's last sampled token.
            row[0] = T.cast(cu_num_logits[req] + local[0], T.int64)
            state[0] = T.cast(idx_mapping[req], T.int64)
            temp[0] = temperature[state[0]]
            rejected[0] = 0
            verified[0] = -1
            if mask_rejected_draft:  # noqa: SIM102
                # A random row that rejected its draft resampled the residual,
                # which for one-hot drafts is the target without the draft.
                if row[0] != T.cast(cu_num_logits[req + 1] - 1, T.int64):
                    verified[0] = drafts[row[0] + 1]
                    if (
                        (verified[0] >= 0)
                        & (T.cast(verified[0], T.int64) != sampled[req, local[0]])
                        & (temp[0] != T.float32(0))
                    ):
                        rejected[0] = 1
            word_seed[0] = _randint_at(seeds[state[0]], pos[row[0]], pos_dtype, False)
            # One test of the temperature for the whole block, as the Gumbel sampler.
            if temp[0] != T.float32(0):
                passes(logits, raw, scores, row, rejected, verified, word_seed, blk, tx,
                       True)  # fmt: skip
            else:
                passes(logits, raw, scores, row, rejected, verified, word_seed, blk, tx,
                       False)  # fmt: skip
            for i in T.serial(num_candidates):
                best[0] = scores[0]
                best_index[0] = _element(0, tx, run)
                for j in T.unroll(1, _PASSES):
                    _keep_or_take(best, best_index, scores[j], _element(j, tx, run))
                _block_argmax(
                    best, best_index, other, other_index, partial, partial_index, tx
                )
                if tx == 0:
                    values[req, blk * num_candidates + i] = best[0]
                    indices[req, blk * num_candidates + i] = T.cast(
                        blk * SINK_CANDIDATE_BLOCK + best_index[0], T.int64
                    )
                    chosen[0] = best_index[0]
                T.sync_threads()
                for j in T.unroll(_PASSES):
                    if _element(j, tx, run) == chosen[0]:
                        scores[j] = -T.infinity(T.float32)

    return sink_candidates


def sink_candidates_key(
    logits_dtype: torch.dtype,
    idx_dtype: torch.dtype,
    num_candidates: int,
    mask_rejected_draft: bool,
    run: int = 1,
    pos_dtype: torch.dtype = torch.int64,
) -> tuple:
    return (
        _DTYPES[logits_dtype],
        _DTYPES[idx_dtype],
        num_candidates,
        mask_rejected_draft,
        run,
        _DTYPES[pos_dtype],
    )


def sink_candidates(
    logits: torch.Tensor,
    vocab_size: int,
    cu_num_logits: torch.Tensor,
    num_sampled: torch.Tensor,
    sampled: torch.Tensor,
    drafts: torch.Tensor,
    idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    seeds: torch.Tensor,
    pos: torch.Tensor,
    num_candidates: int,
    mask_rejected_draft: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``SinkSampler.alternates_of``'s launch: each request's candidates per
    vocabulary block, ``(values, indices)`` of ``[num_reqs, blocks *
    num_candidates]``, in vocabulary order of the blocks."""
    num_reqs = idx_mapping.shape[0]
    num_blocks = -(-vocab_size // SINK_CANDIDATE_BLOCK)
    width = num_blocks * num_candidates
    values = logits.new_empty((num_reqs, width), dtype=torch.float32)
    indices = logits.new_empty((num_reqs, width), dtype=torch.int64)
    if num_reqs > 0:
        kernel = _kernel(
            sink_candidates_kernel,
            *sink_candidates_key(
                logits.dtype,
                idx_mapping.dtype,
                num_candidates,
                mask_rejected_draft,
                _triton_run(logits),
                pos.dtype,
            ),
        )
        kernel(
            values,
            indices,
            logits[:, :vocab_size],
            cu_num_logits,
            num_sampled,
            sampled,
            drafts,
            idx_mapping,
            temperature,
            seeds,
            pos,
        )
    return values, indices


def _next_power_of_2(value: int) -> int:
    return 1 << max(value - 1, 0).bit_length()


@tilelang.jit(pass_configs=_PASS)
def sink_policy_kernel(
    idx_dtype: str,
    num_alternates: int,
    rejection: bool,
    logprobs: bool,
    sampled_width: int,
    num_event_fields: int,
):
    """``sink._sink_policy_kernel``: one block per request, whose first thread
    applies the draft-zero policy, keeps the alternates and writes the events in
    Triton's order. The flagged-value maximum takes the lowest row among equal
    values (``row_values`` are finite: the detector stores them through
    ``nan_to_num``). Each buffer is passed flat; ``num_alternates`` is at least 1,
    as the Triton launch's ``NUM_ALTERNATES``."""
    alts = num_alternates
    width = sampled_width
    padded = _next_power_of_2(width + 1)
    reqs, cu_len, flags_len, values_len, drafts_len = T.dynamic(
        "reqs, cu_len, flags_len, values_len, drafts_len"
    )
    num_len, rejected_len, decoding_len = T.dynamic(
        "num_len, rejected_len, decoding_len"
    )
    out_len, num_out_len, lp_len, rank_len = T.dynamic(
        "out_len, num_out_len, lp_len, rank_len"
    )
    new_len, new_lp_len, new_rank_len = T.dynamic("new_len, new_lp_len, new_rank_len")
    state_len, state_lp_len, state_rank_len = T.dynamic(
        "state_len, state_lp_len, state_rank_len"
    )
    idx_len, tries_len, total_len_len, count_rows, count_cols = T.dynamic(
        "idx_len, tries_len, total_len_len, count_rows, count_cols"
    )
    count_stride = T.dynamic("count_stride", "int64")

    @T.prim_func
    def sink_policy(
        sampled: T.Tensor((reqs, width), T.int64),
        num_sampled: T.Tensor((num_len,), T.int32),
        num_rejected: T.Tensor((rejected_len,), T.int32),
        out_tokens: T.Tensor((out_len,), T.int64),
        num_out: T.Tensor((num_out_len,), T.int32),
        events: T.Tensor((reqs, num_event_fields), T.int32),
        recovered_logprobs: T.Tensor((lp_len,), T.float32),
        recovered_ranks: T.Tensor((rank_len,), T.int64),
        cu_num_logits: T.Tensor((cu_len,), T.int32),
        idx_mapping: T.Tensor((idx_len,), idx_dtype),
        decoding: T.Tensor((decoding_len,), T.int8),
        flags: T.Tensor((flags_len,), T.int8),
        row_values: T.Tensor((values_len,), T.float32),
        drafts: T.Tensor((drafts_len,), T.int32),
        new_alternates: T.Tensor((new_len,), T.int64),
        new_alternate_logprobs: T.Tensor((new_lp_len,), T.float32),
        new_alternate_ranks: T.Tensor((new_rank_len,), T.int64),
        alternates: T.Tensor((state_len,), T.int64),
        alternate_logprobs: T.Tensor((state_lp_len,), T.float32),
        alternate_ranks: T.Tensor((state_rank_len,), T.int64),
        tries: T.Tensor((tries_len,), T.int32),
        total_len: T.Tensor((total_len_len,), T.int32),
        bin_counts: T.StridedTensor(
            (count_rows, count_cols), (count_stride, 1), T.int32
        ),
        counts: T.Tensor((3,), T.int64),
    ):
        with T.Kernel(reqs, threads=32) as req:
            tx = T.get_thread_binding()
            start = T.alloc_local((1,), T.int32)
            logits_n = T.alloc_local((1,), T.int32)
            sampled_n = T.alloc_local((1,), T.int32)
            state = T.alloc_local((1,), T.int64)
            is_decoding = T.alloc_local((1,), T.int32)
            row0_flag = T.alloc_local((1,), T.int32)
            accepted_flags = T.alloc_local((1,), T.int32)
            cut = T.alloc_local((1,), T.int32)
            draft_zero_flag = T.alloc_local((1,), T.int32)
            outcome = T.alloc_local((1,), T.int32)
            drafter_flags = T.alloc_local((1,), T.int32)
            cut_row = T.alloc_local((1,), T.int32)
            attempts = T.alloc_local((1,), T.int32)
            slot = T.alloc_local((1,), T.int64)
            alternate = T.alloc_local((1,), T.int64)
            rejected = T.alloc_local((1,), T.int32)
            draft_zero = T.alloc_local((1,), T.int64)
            shift = T.alloc_local((1,), T.int32)
            value = T.alloc_local((1,), T.float32)
            best = T.alloc_local((1,), T.float32)
            best_row = T.alloc_local((1,), T.int32)
            flagged = T.alloc_local((1,), T.int32)
            if tx == 0:
                start[0] = cu_num_logits[req]
                logits_n[0] = cu_num_logits[req + 1] - start[0]
                sampled_n[0] = num_sampled[req]
                if sampled_n[0] == 0:
                    # A prefill chunk: its row is not sampled.
                    num_out[req] = 0
                    events[req, 0] = 0
                    events[req, 1] = 2
                    events[req, 2] = 0
                    events[req, 3] = -1
                    events[req, 4] = 0
                    events[req, 5] = -1
                else:
                    state[0] = T.cast(idx_mapping[req], T.int64)
                    is_decoding[0] = T.if_then_else(decoding[req] != T.int8(0), 1, 0)
                    row0_flag[0] = T.if_then_else(flags[start[0]] != T.int8(0), 1, 0)
                    # Drafter rows whose inputs were accepted (1 .. num_sampled
                    # - 1), and the row of the first rejected draft.
                    accepted_flags[0] = 0
                    cut[0] = 0
                    for r in T.serial(1, padded):
                        # The flag is read only for the request's rows.
                        if r < logits_n[0]:  # noqa: SIM102
                            if flags[start[0] + r] != T.int8(0):
                                if r < sampled_n[0]:
                                    accepted_flags[0] = accepted_flags[0] + 1
                                if r == sampled_n[0]:
                                    cut[0] = 1
                    if sampled_n[0] >= logits_n[0]:
                        cut[0] = 0
                    draft_zero_flag[0] = is_decoding[0] & row0_flag[0]
                    # Prompt row flagged: FLAGGED; it cannot be replaced.
                    outcome[0] = T.if_then_else(
                        (is_decoding[0] == 0) & (row0_flag[0] != 0), 1, 0
                    )
                    if rejection:
                        attempts[0] = tries[state[0]]
                        slot[0] = state[0] * alts + T.cast(
                            T.min(attempts[0], alts - 1), T.int64
                        )
                        alternate[0] = alternates[slot[0]]
                        rejected[0] = T.if_then_else(
                            (draft_zero_flag[0] != 0)
                            & (attempts[0] < alts)
                            & (alternate[0] >= 0),
                            1,
                            0,
                        )
                        outcome[0] = T.if_then_else(
                            rejected[0] != 0,
                            2,
                            T.if_then_else(draft_zero_flag[0] != 0, 3, outcome[0]),
                        )
                        drafter_flags[0] = T.if_then_else(
                            rejected[0] != 0, 0, accepted_flags[0] + cut[0]
                        )
                        cut_row[0] = T.if_then_else(
                            (cut[0] != 0) & (rejected[0] == 0), sampled_n[0], -1
                        )
                        if rejected[0] != 0:
                            # Draft zero is replaced by its next alternate; the
                            # request's position does not advance.
                            draft_zero[0] = T.cast(drafts[start[0]], T.int64)
                            sampled[req, 0] = alternate[0]
                            num_sampled[req] = 1
                            num_rejected[req] = logits_n[0]
                            tries[state[0]] = attempts[0] + 1
                            if logprobs:
                                recovered_logprobs[req] = alternate_logprobs[slot[0]]
                                recovered_ranks[req] = alternate_ranks[slot[0]]
                            # The request state already holds draft zero as its
                            # last token: take it out.
                            total_len[state[0]] = total_len[state[0]] - 1
                            bin_counts[state[0], draft_zero[0]] = (
                                bin_counts[state[0], draft_zero[0]] - 1
                            )
                            out_tokens[T.cast(req, T.int64) * (width + 1)] = alternate[
                                0
                            ]
                            num_out[req] = 1
                        else:
                            # The last sampled token's draw provides the next
                            # alternates.
                            for a in T.serial(alts):
                                alternates[state[0] * alts + a] = new_alternates[
                                    T.cast(req, T.int64) * alts + a
                                ]
                                if logprobs:
                                    alternate_logprobs[state[0] * alts + a] = (
                                        new_alternate_logprobs[
                                            T.cast(req, T.int64) * alts + a
                                        ]
                                    )
                                    alternate_ranks[state[0] * alts + a] = (
                                        new_alternate_ranks[
                                            T.cast(req, T.int64) * alts + a
                                        ]
                                    )
                            tries[state[0]] = 0
                            # Draft zero first, then the step's sampled tokens.
                            shift[0] = is_decoding[0]
                            for r in T.serial(width):
                                if r < sampled_n[0]:
                                    out_tokens[
                                        T.cast(req, T.int64) * (width + 1)
                                        + shift[0]
                                        + r
                                    ] = sampled[req, r]
                            if is_decoding[0] != 0:
                                out_tokens[T.cast(req, T.int64) * (width + 1)] = T.cast(
                                    drafts[start[0]], T.int64
                                )
                            num_out[req] = sampled_n[0] + shift[0]
                    else:
                        outcome[0] = T.if_then_else(
                            draft_zero_flag[0] != 0, 1, outcome[0]
                        )
                        drafter_flags[0] = accepted_flags[0]
                        cut_row[0] = -1
                    # The largest flagged value among the rows the step judged
                    # (up to the first rejected draft) and its row.
                    for r in T.serial(padded):
                        value[0] = T.float32(-1)
                        if (r < logits_n[0]) & (r <= sampled_n[0]):  # noqa: SIM102
                            if flags[start[0] + r] != T.int8(0):
                                value[0] = row_values[start[0] + r]
                        if (r == 0) | (value[0] > best[0]):
                            best[0] = value[0]
                            best_row[0] = r
                    flagged[0] = T.if_then_else(best[0] >= T.float32(0), 1, 0)
                    events[req, 0] = outcome[0]
                    events[req, 1] = T.if_then_else(is_decoding[0] != 0, 0, 2)
                    events[req, 2] = drafter_flags[0]
                    events[req, 3] = cut_row[0]
                    events[req, 4] = T.if_then_else(
                        flagged[0] != 0, T.cast(best[0], T.int32), 0
                    )
                    events[req, 5] = T.if_then_else(flagged[0] != 0, best_row[0], -1)
                    T.atomic_add(counts[0], T.cast(draft_zero_flag[0], T.int64))
                    T.atomic_add(counts[1], T.cast(drafter_flags[0], T.int64))
                    T.atomic_add(
                        counts[2],
                        T.cast((is_decoding[0] == 0) & (row0_flag[0] != 0), T.int64),
                    )

    return sink_policy


def sink_policy(
    sampled: torch.Tensor,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    out_tokens: torch.Tensor,
    num_out: torch.Tensor,
    events: torch.Tensor,
    recovered_logprobs: torch.Tensor,
    recovered_ranks: torch.Tensor,
    cu_num_logits: torch.Tensor,
    idx_mapping: torch.Tensor,
    decoding: torch.Tensor,
    flags: torch.Tensor,
    row_values: torch.Tensor,
    drafts: torch.Tensor,
    new_alternates: torch.Tensor,
    new_alternate_logprobs: torch.Tensor,
    new_alternate_ranks: torch.Tensor,
    alternates: torch.Tensor,
    alternate_logprobs: torch.Tensor,
    alternate_ranks: torch.Tensor,
    tries: torch.Tensor,
    total_len: torch.Tensor,
    bin_counts: torch.Tensor,
    counts: torch.Tensor,
    num_alternates: int,
    rejection: bool,
    logprobs: bool,
    num_event_fields: int,
) -> None:
    """``SinkSampler.apply``'s launch, with the Triton launch's arguments (the
    placeholders where a variant has no buffer included); the per-slot and step
    buffers are contiguous."""
    num_reqs, sampled_width = sampled.shape
    if num_reqs == 0:
        return
    kernel = _kernel(
        sink_policy_kernel,
        *sink_policy_key(
            idx_mapping.dtype,
            num_alternates,
            rejection,
            logprobs,
            sampled_width,
            num_event_fields,
        ),
    )
    kernel(
        sampled,
        num_sampled,
        num_rejected,
        out_tokens.view(-1),
        num_out.view(-1),
        events,
        recovered_logprobs.view(-1),
        recovered_ranks.view(-1),
        cu_num_logits,
        idx_mapping,
        decoding,
        flags,
        row_values,
        drafts,
        new_alternates.view(-1),
        new_alternate_logprobs.view(-1),
        new_alternate_ranks.view(-1),
        alternates.view(-1),
        alternate_logprobs.view(-1),
        alternate_ranks.view(-1),
        tries,
        total_len,
        bin_counts,
        counts,
    )


def sink_policy_key(
    idx_dtype: torch.dtype,
    num_alternates: int,
    rejection: bool,
    logprobs: bool,
    sampled_width: int,
    num_event_fields: int,
) -> tuple:
    return (
        _DTYPES[idx_dtype],
        max(num_alternates, 1),
        rejection,
        logprobs and rejection,
        sampled_width,
        num_event_fields,
    )


# --- Greedy argmax and verification -----------------------------------------

# dspark.greedy._BLOCK
GREEDY_BLOCK = 2048
_VEC = 8  # values in one 16-byte load of 16-bit logits
_INT32_MAX = 2147483647


@T.macro
def _first_keep_or_take(best, best_index, value, index):
    """``(best, best_index)`` = ``_first_argmax`` of itself and ``(value, index)``,
    by selects."""
    keep = T.alloc_var(
        T.int32,
        init=T.call_extern(
            "int32", "vllm_first_argmax_keep", best[0], best_index[0], value, index
        ),
    )
    best_index[0] = T.if_then_else(keep != 0, best_index[0], index)
    best[0] = T.if_then_else(keep != 0, best[0], value)


@T.macro
def _first_exchange(best, best_index, other, other_index, distance):
    other[0] = T.shfl_xor(best[0], distance)
    other_index[0] = T.shfl_xor(best_index[0], distance)
    _first_keep_or_take(best, best_index, other[0], other_index[0])


@T.macro
def _warp_first_argmax(best, best_index, other, other_index):
    """Every lane's ``(best, best_index)`` = the warp's first maximum (the
    combine is a total order, so the pairing does not matter)."""
    _first_exchange(best, best_index, other, other_index, 16)
    _first_exchange(best, best_index, other, other_index, 8)
    _first_exchange(best, best_index, other, other_index, 4)
    _first_exchange(best, best_index, other, other_index, 2)
    _first_exchange(best, best_index, other, other_index, 1)


@tilelang.jit(pass_configs=_PASS)
def partial_argmax_kernel(
    base_dtype: str, bias_dtype: str | None, vector: bool, threads: int = _THREADS
):
    """``dspark.greedy._partial_argmax`` (with a bias, each sum rounded to the base
    dtype as the materialized ``torch.add``) and ``_partial_argmax_logits``
    (without): the first maximum of each 2048-column part of each row, by value
    and shard-local column, into flat FP32 / int32 scratch at ``row * parts +
    part``. ``threads`` a block; with ``vector`` (16-byte aligned rows, columns a
    multiple of eight), each thread reads runs of eight consecutive values. The
    combine is a total order, so the layout does not change the result."""
    rows, vocab, values_len, indices_len = T.dynamic(
        "rows, vocab, values_len, indices_len"
    )
    base_stride, bias_stride = T.dynamic("base_stride, bias_stride", "int64")
    biased = bias_dtype is not None
    if not biased:
        bias_dtype = base_dtype  # placeholder the variant never reads
    warps = threads // 32
    runs = GREEDY_BLOCK // (threads * _VEC)
    passes = GREEDY_BLOCK // threads

    @T.macro
    def visit(base, bias, best, best_index, row, col):
        value = T.cast(base[row, col], T.float32)
        if biased:
            value = T.cast(
                T.cast(value + T.cast(bias[row, col], T.float32), base_dtype),
                T.float32,
            )
        _first_keep_or_take(best, best_index, value, col)

    @T.prim_func
    def partial_argmax(
        base: T.StridedTensor((rows, vocab), (base_stride, 1), base_dtype),
        bias: T.StridedTensor((rows, vocab), (bias_stride, 1), bias_dtype),
        values: T.Tensor((values_len,), T.float32),
        indices: T.Tensor((indices_len,), T.int32),
    ):
        with T.Kernel(rows, T.ceildiv(vocab, GREEDY_BLOCK), threads=threads) as (
            row,
            part,
        ):
            T.import_source(_HEADER)
            tx = T.get_thread_binding()
            a = T.alloc_local((_VEC,), base_dtype)
            b = T.alloc_local((_VEC,), bias_dtype)
            best = T.alloc_local((1,), T.float32)
            best_index = T.alloc_local((1,), T.int32)
            other = T.alloc_local((1,), T.float32)
            other_index = T.alloc_local((1,), T.int32)
            partial = T.alloc_shared((warps,), T.float32)
            partial_index = T.alloc_shared((warps,), T.int32)
            # Triton's masked lanes, which no column beats.
            best[0] = -T.infinity(T.float32)
            best_index[0] = _INT32_MAX
            if vector:
                # The launch checked it: rows 16-byte aligned, columns whole runs.
                T.assume(base_stride % _VEC == 0)
                T.assume(bias_stride % _VEC == 0)
                T.assume(vocab % _VEC == 0)
                for run in T.unroll(runs):
                    col0 = part * GREEDY_BLOCK + (run * threads + tx) * _VEC
                    if col0 < vocab:
                        for i in T.vectorized(_VEC):
                            a[i] = base[row, col0 + i]
                        if biased:
                            for i in T.vectorized(_VEC):
                                b[i] = bias[row, col0 + i]
                        for i in T.unroll(_VEC):
                            value = T.cast(a[i], T.float32)
                            if biased:
                                value = T.cast(
                                    T.cast(value + T.cast(b[i], T.float32), base_dtype),
                                    T.float32,
                                )
                            _first_keep_or_take(best, best_index, value, col0 + i)
            else:
                for j in T.unroll(passes):
                    col = part * GREEDY_BLOCK + j * threads + tx
                    if col < vocab:
                        visit(base, bias, best, best_index, row, col)
            _warp_first_argmax(best, best_index, other, other_index)
            if tx % 32 == 0:
                partial[tx // 32] = best[0]
                partial_index[tx // 32] = best_index[0]
            T.sync_threads()
            if tx == 0:
                for w in T.unroll(1, warps):
                    _first_keep_or_take(best, best_index, partial[w], partial_index[w])
                slot = T.cast(row, T.int64) * T.ceildiv(vocab, GREEDY_BLOCK) + part
                values[slot] = best[0]
                indices[slot] = best_index[0]

    return partial_argmax


@tilelang.jit(pass_configs=_PASS)
def finish_argmax_kernel(output: str, rows_per_block: int = 0):
    """``dspark.greedy._finish_argmax_pair`` (``output`` "pair": the row's first
    maximum and its global index ``+ vocab_start`` as FP32 into ``pairs[row]``) and
    ``_finish_argmax`` ("int32" / "int64": the index into ``out[row *
    out_stride]``), from the parts' scratch: one warp per row, or with
    ``rows_per_block`` one thread per row, the parts in order."""
    rows, values_len, indices_len, out_len = T.dynamic(
        "rows, values_len, indices_len, out_len"
    )
    pair = output == "pair"
    out_dtype = "float32" if pair else output

    @T.macro
    def _finish_store(out, best, best_index, row, out_stride, vocab_start):
        if pair:
            out[T.cast(row, T.int64) * 2] = best[0]
            out[T.cast(row, T.int64) * 2 + 1] = T.cast(
                best_index[0] + vocab_start, T.float32
            )
        else:
            out[T.cast(row, T.int64) * out_stride] = T.cast(best_index[0], out_dtype)

    @T.prim_func
    def finish_argmax(
        values: T.Tensor((values_len,), T.float32),
        indices: T.Tensor((indices_len,), T.int32),
        out: T.Tensor((out_len,), out_dtype),
        parts: T.int32,
        out_rows: T.int32,
        out_stride: T.int64,
        vocab_start: T.int32,
    ):
        if rows_per_block:
            with T.Kernel(
                T.ceildiv(out_rows, rows_per_block), threads=rows_per_block
            ) as bx:
                T.import_source(_HEADER)
                row = bx * rows_per_block + T.get_thread_binding()
                best = T.alloc_local((1,), T.float32)
                best_index = T.alloc_local((1,), T.int32)
                best[0] = -T.infinity(T.float32)
                best_index[0] = _INT32_MAX
                if row < out_rows:
                    for k in T.serial(parts):
                        slot = T.cast(row, T.int64) * parts + k
                        _first_keep_or_take(
                            best, best_index, values[slot], indices[slot]
                        )
                    _finish_store(out, best, best_index, row, out_stride, vocab_start)
        else:
            with T.Kernel(out_rows, threads=32) as row:
                T.import_source(_HEADER)
                tx = T.get_thread_binding()
                best = T.alloc_local((1,), T.float32)
                best_index = T.alloc_local((1,), T.int32)
                other = T.alloc_local((1,), T.float32)
                other_index = T.alloc_local((1,), T.int32)
                best[0] = -T.infinity(T.float32)
                best_index[0] = _INT32_MAX
                for k in T.serial(T.ceildiv(parts - tx, 32)):
                    slot = T.cast(row, T.int64) * parts + tx + k * 32
                    _first_keep_or_take(best, best_index, values[slot], indices[slot])
                _warp_first_argmax(best, best_index, other, other_index)
                if tx == 0:
                    _finish_store(out, best, best_index, row, out_stride, vocab_start)

    return finish_argmax


@tilelang.jit(pass_configs=_PASS)
def select_first_argmax_kernel(output_dtype: str, ranks: int):
    """``dspark.greedy._select_first_argmax``: the first maximum's global index
    over the all-gathered ``(value, index)`` pairs of each row ([rows, 2 * ranks]
    FP32; indices exact in FP32), into ``out[row * out_stride]``, a warp a row.
    As in Triton, the number of ranks is fixed when the kernel is built and every
    load is issued before any combine: lane ``k < ranks`` reads rank ``k``'s pair,
    the other lanes hold Triton's masked values (-inf, 2**30), and the lanes reduce
    over the next power of two (the combine is a total order)."""
    assert 1 <= ranks <= 32
    rows, width, out_len = T.dynamic("rows, width, out_len")
    pairs_stride = T.dynamic("pairs_stride", "int64")
    span = 1 << max(ranks - 1, 0).bit_length()

    @T.prim_func
    def select_first_argmax(
        pairs: T.StridedTensor((rows, width), (pairs_stride, 1), T.float32),
        out: T.Tensor((out_len,), output_dtype),
        out_stride: T.int64,
    ):
        with T.Kernel(rows, threads=32) as row:
            T.import_source(_HEADER)
            tx = T.get_thread_binding()
            best = T.alloc_local((1,), T.float32)
            best_index = T.alloc_local((1,), T.int32)
            other = T.alloc_local((1,), T.float32)
            other_index = T.alloc_local((1,), T.int32)
            lane = T.min(tx, ranks - 1)
            value = pairs[row, 2 * lane]
            index = pairs[row, 2 * lane + 1]
            best[0] = T.if_then_else(tx < ranks, value, -T.infinity(T.float32))
            best_index[0] = T.if_then_else(
                tx < ranks, T.cast(index, T.int32), T.int32(1 << 30)
            )
            if span >= 32:
                _first_exchange(best, best_index, other, other_index, 16)
            if span >= 16:
                _first_exchange(best, best_index, other, other_index, 8)
            if span >= 8:
                _first_exchange(best, best_index, other, other_index, 4)
            if span >= 4:
                _first_exchange(best, best_index, other, other_index, 2)
            if span >= 2:
                _first_exchange(best, best_index, other, other_index, 1)
            if tx == 0:
                out[T.cast(row, T.int64) * out_stride] = T.cast(
                    best_index[0], output_dtype
                )

    return select_first_argmax


def _flat(column: torch.Tensor) -> torch.Tensor:
    """The span of a (possibly strided) column as a flat view, which a kernel
    indexes at ``row * stride``."""
    rows = column.shape[0]
    span = (rows - 1) * column.stride(0) + 1 if rows > 0 else 0
    return column.as_strided((span,), (1,))


def _vector_rows(*tensors: torch.Tensor) -> bool:
    """Every row of each 16-bit tensor starts 16-byte aligned and its columns are
    whole runs of eight."""
    return all(
        t.element_size() == 2
        and t.data_ptr() % 16 == 0
        and (t.stride(0) * 2) % 16 == 0
        and t.shape[1] % _VEC == 0
        for t in tensors
    )


def partial_argmax_key(
    base_dtype: torch.dtype, bias_dtype: torch.dtype | None, vector: bool
) -> tuple:
    return (
        _DTYPES[base_dtype],
        _DTYPES[bias_dtype] if bias_dtype is not None else None,
        vector,
    )


def _partial_argmax(
    base: torch.Tensor,
    bias: torch.Tensor | None,
    partial_values: torch.Tensor,
    partial_indices: torch.Tensor,
) -> None:
    rows, vocab = base.shape
    if rows == 0:
        return
    vector = _vector_rows(base) if bias is None else _vector_rows(base, bias)
    kernel = _kernel(
        partial_argmax_kernel,
        *partial_argmax_key(base.dtype, None if bias is None else bias.dtype, vector),
    )
    kernel(
        base,
        base if bias is None else bias,
        partial_values.view(-1),
        partial_indices.view(-1),
    )


def _finish(
    partial_values: torch.Tensor,
    partial_indices: torch.Tensor,
    out: torch.Tensor,
    rows: int,
    parts: int,
    out_stride: int,
    vocab_start: int,
    output: str,
) -> None:
    if rows == 0:
        return
    kernel = _kernel(finish_argmax_kernel, output)
    kernel(
        partial_values.view(-1),
        partial_indices.view(-1),
        out,
        parts,
        rows,
        out_stride,
        vocab_start,
    )


def greedy_markov_shard_pair(
    base_logits: torch.Tensor,
    markov_bias: torch.Tensor,
    vocab_start: int,
    pairs: torch.Tensor,
    partial_values: torch.Tensor,
    partial_indices: torch.Tensor,
) -> None:
    """``dspark.greedy.greedy_markov_shard_pair``."""
    if base_logits.shape != markov_bias.shape:
        raise ValueError(
            "DSpark greedy base logits and Markov bias must have identical shapes: "
            f"{tuple(base_logits.shape)} != {tuple(markov_bias.shape)}"
        )
    rows, vocab = base_logits.shape
    if vocab_start + vocab > 1 << 24:
        raise ValueError("vocabulary indices must stay exact in FP32")
    parts = -(-vocab // GREEDY_BLOCK)
    if min(partial_values.numel(), partial_indices.numel()) < rows * parts:
        raise ValueError("DSpark greedy scratch is smaller than the logits require")
    _partial_argmax(base_logits, markov_bias, partial_values, partial_indices)
    _finish(
        partial_values,
        partial_indices,
        pairs.view(-1),
        rows,
        parts,
        2,
        vocab_start,
        "pair",
    )


def greedy_shard_pair(
    logits: torch.Tensor,
    vocab: int,
    vocab_start: int,
    pairs: torch.Tensor,
    partial_values: torch.Tensor,
    partial_indices: torch.Tensor,
) -> None:
    """``dspark.greedy.greedy_shard_pair``."""
    rows = logits.shape[0]
    if vocab > logits.shape[1]:
        raise ValueError("the shard's real vocabulary exceeds its logits")
    if vocab_start + vocab > 1 << 24:
        raise ValueError("vocabulary indices must stay exact in FP32")
    parts = -(-vocab // GREEDY_BLOCK)
    if min(partial_values.numel(), partial_indices.numel()) < rows * parts:
        raise ValueError("greedy scratch is smaller than the logits require")
    _partial_argmax(logits[:, :vocab], None, partial_values, partial_indices)
    _finish(
        partial_values,
        partial_indices,
        pairs.view(-1),
        rows,
        parts,
        2,
        vocab_start,
        "pair",
    )


def select_first_argmax(pairs: torch.Tensor, output: torch.Tensor) -> None:
    """``dspark.greedy.select_first_argmax``."""
    rows = pairs.shape[0]
    if rows == 0:
        return
    kernel = _kernel(
        select_first_argmax_kernel, _DTYPES[output.dtype], pairs.shape[1] // 2
    )
    kernel(pairs, _flat(output), output.stride(0))


def sample_greedy_markov(
    base_logits: torch.Tensor,
    markov_bias: torch.Tensor,
    output: torch.Tensor,
    partial_values: torch.Tensor,
    partial_indices: torch.Tensor,
) -> None:
    """``dspark.greedy.sample_greedy_markov``."""
    if base_logits.shape != markov_bias.shape:
        raise ValueError(
            "DSpark greedy base logits and Markov bias must have identical shapes: "
            f"{tuple(base_logits.shape)} != {tuple(markov_bias.shape)}"
        )
    rows, vocab = base_logits.shape
    parts = -(-vocab // GREEDY_BLOCK)
    if min(partial_values.numel(), partial_indices.numel()) < rows * parts:
        raise ValueError("DSpark greedy scratch is smaller than the logits require")
    _partial_argmax(base_logits, markov_bias, partial_values, partial_indices)
    _finish(
        partial_values,
        partial_indices,
        _flat(output),
        rows,
        parts,
        output.stride(0),
        0,
        _DTYPES[output.dtype],
    )


@tilelang.jit(pass_configs=_PASS)
def greedy_rejection_kernel(width: int, drafts_dtype: str):
    """``greedy_verification._greedy_rejection_kernel``: per request, drafts are
    accepted while they equal the target's argmax (a -1 draft never is); the first
    mismatch commits the argmax, and a fully accepted block commits the bonus
    position's argmax. Positions after the first mismatch are not written."""
    reqs, cu_len, targets_len, drafts_len = T.dynamic(
        "reqs, cu_len, targets_len, drafts_len"
    )

    @T.prim_func
    def greedy_rejection(
        sampled: T.Tensor((reqs, width), T.int64),
        num_sampled: T.Tensor((reqs,), T.int32),
        target_ids: T.Tensor((targets_len,), T.int64),
        draft_sampled: T.Tensor((drafts_len,), drafts_dtype),
        cu_num_logits: T.Tensor((cu_len,), T.int32),
    ):
        with T.Kernel(reqs, threads=32) as req:
            tx = T.get_thread_binding()
            start = T.alloc_local((1,), T.int64)
            num_drafts = T.alloc_local((1,), T.int64)
            accepted = T.alloc_local((1,), T.int64)
            verifying = T.alloc_local((1,), T.int32)
            draft = T.alloc_local((1,), T.int64)
            target = T.alloc_local((1,), T.int64)
            if tx == 0:
                start[0] = T.cast(cu_num_logits[req], T.int64)
                num_drafts[0] = T.cast(cu_num_logits[req + 1], T.int64) - start[0] - 1
                accepted[0] = 0
                verifying[0] = 1
                for i in T.serial(num_drafts[0]):
                    if verifying[0] != 0:
                        draft[0] = T.cast(draft_sampled[start[0] + i + 1], T.int64)
                        target[0] = target_ids[start[0] + i]
                        # -1 is a placeholder draft that is always rejected.
                        if (target[0] == draft[0]) & (draft[0] >= 0):
                            sampled[req, i] = draft[0]
                            accepted[0] = accepted[0] + 1
                        else:
                            sampled[req, i] = target[0]
                            verifying[0] = 0
                if accepted[0] == num_drafts[0]:
                    # Every draft accepted: the bonus position's argmax.
                    sampled[req, num_drafts[0]] = target_ids[start[0] + num_drafts[0]]
                num_sampled[req] = T.cast(accepted[0] + 1, T.int32)

    return greedy_rejection


def greedy_rejection_sample(
    target_ids: torch.Tensor,
    draft_sampled: torch.Tensor,
    cu_num_logits: torch.Tensor,
    num_speculative_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``greedy_verification.greedy_rejection_sample``."""
    num_reqs = cu_num_logits.shape[0] - 1
    sampled = draft_sampled.new_empty(
        num_reqs, num_speculative_steps + 1, dtype=torch.int64
    )
    num_sampled = sampled.new_empty(num_reqs, dtype=torch.int32)
    if num_reqs > 0:
        kernel = _kernel(
            greedy_rejection_kernel,
            num_speculative_steps + 1,
            _DTYPES[draft_sampled.dtype],
        )
        kernel(sampled, num_sampled, target_ids, draft_sampled, cu_num_logits)
    return sampled, num_sampled


# --- Temperature --------------------------------------------------------------


@tilelang.jit(pass_configs=_PASS)
def temperature_kernel(idx_dtype: str, run: int, threads: int = _TEMPERATURE_THREADS):
    """``gumbel._temperature_kernel``: FP32 logits divided in place by their
    request's temperature (``div.full.f32``), rows at temperature 0 or 1 left
    alone; a block of ``threads`` per ``(token, 8192 columns)``, each thread's
    elements in runs of ``run`` (``_triton_run``), one vector load and store each
    when aligned. As in Triton, the thread's runs are all loaded, then divided,
    then stored: the logits are read and written through one pointer, so a run
    stored before the next run's load would make every load wait for a store. As
    in Triton, the temperature is read at the mapped index without a validity
    check."""
    tokens, vocab, idx_len, temp_len = T.dynamic("tokens, vocab, idx_len, temp_len")
    logits_stride = T.dynamic("logits_stride", "int64")
    runs = _TEMPERATURE_BLOCK // (threads * run)

    @T.prim_func
    def temperature(
        logits: T.StridedTensor((tokens, vocab), (logits_stride, 1), T.float32),
        idx_mapping: T.Tensor((idx_len,), idx_dtype),
        temperature: T.Tensor((temp_len,), T.float32),
    ):
        with T.Kernel(
            tokens, T.ceildiv(vocab, _TEMPERATURE_BLOCK), threads=threads
        ) as (t, blk):
            T.import_source(_HEADER)
            tx = T.get_thread_binding()
            temp = T.alloc_local((1,), T.float32)
            raw = T.alloc_local((runs * run,), T.float32)
            if run > 1:
                # The launch checked it: a 16-byte base and a stride of 16 elements.
                T.assume(logits_stride % 16 == 0)
            temp[0] = temperature[T.cast(idx_mapping[t], T.int64)]
            if (temp[0] != T.float32(0)) & (temp[0] != T.float32(1)):
                for r in T.unroll(runs):
                    col0 = blk * _TEMPERATURE_BLOCK + r * (threads * run) + run * tx
                    _load_run_at(logits, raw, r * run, t, col0, vocab, run)
                for k in T.unroll(runs * run):
                    raw[k] = T.call_extern("float32", "vllm_div_full", raw[k], temp[0])
                for r in T.unroll(runs):
                    col0 = blk * _TEMPERATURE_BLOCK + r * (threads * run) + run * tx
                    _store_run_at(logits, raw, r * run, t, col0, vocab, run)

    return temperature


def apply_temperature(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
) -> None:
    """``gumbel.apply_temperature``."""
    if logits.shape[0] == 0:
        return
    kernel = _kernel(
        temperature_kernel, _DTYPES[expanded_idx_mapping.dtype], _triton_run(logits)
    )
    kernel(logits, expanded_idx_mapping.contiguous(), temperature)


# --- Preparation --------------------------------------------------------------

# The runner's batch-to-request-state index is ``np.intp``.
RUNNER_IDX_DTYPE = torch.int64


def _runs(dtype: torch.dtype) -> tuple[int, int]:
    """Both layouts a launch may take for logits of ``dtype`` (``_triton_run``)."""
    return 1, 16 // torch.empty((), dtype=dtype).element_size()


def _call(factory: Any, *key: Any) -> tuple:
    from vllm.models.deepseek_v4_1.tilelang import prepare

    return prepare.call(factory, *key)


def declare(calls: Callable[[], list]) -> None:
    """Declare an owner's calls with the runner's JIT registry: its TileLang owner
    compiles them with the deployment's other TileLang calls before memory
    profiling, and the kernel cache freezes. Like a Triton dispatcher's
    ``register_warmup``, a no-op where no registry collects (the first launch then
    compiles, the cache not being frozen)."""
    from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry

    if JitWarmupRegistry.active() is None:
        return
    from vllm.models.deepseek_v4_1.tilelang import prepare

    prepare.declare(calls)


def sampling_calls(
    logits_dtypes: Sequence[torch.dtype], temperature: bool = True
) -> list:
    """The target sampler's calls: Gumbel-max sampling of logits of each of
    ``logits_dtypes`` without a logits cache, the temperature already applied,
    and the temperature of processed FP32 logits."""
    calls = [
        _call(
            gumbel_sample_kernel,
            *gumbel_sample_key(dtype, RUNNER_IDX_DTYPE, False, False, run=run),
        )
        for dtype in dict.fromkeys(logits_dtypes)
        for run in _runs(dtype)
    ]
    if temperature:
        calls += [
            _call(temperature_kernel, _DTYPES[RUNNER_IDX_DTYPE], run)
            for run in _runs(torch.float32)
        ]
    return calls


def drafting_calls(
    logits_dtype: torch.dtype,
    cache_dtype: torch.dtype | None,
    col_dtype: torch.dtype | None,
    per_token_col: bool,
) -> list:
    """A speculator's draft sampling: temperature applied in the kernel, the
    drafting noise stream, and the logits cache when there is one."""
    return [
        _call(
            gumbel_sample_kernel,
            *gumbel_sample_key(
                logits_dtype,
                RUNNER_IDX_DTYPE,
                True,
                True,
                cache_dtype,
                col_dtype,
                per_token_col,
                run,
            ),
        )
        for run in _runs(logits_dtype)
    ]


def _unique(calls: list) -> list:
    """``calls`` without repeats (a call's keyword arguments are unhashable)."""
    seen: set = set()
    unique = []
    for call in calls:
        key = (id(call[0]), call[1])
        if key not in seen:
            seen.add(key)
            unique.append(call)
    return unique


def sink_calls(
    *,
    logits_dtypes: Sequence[torch.dtype],
    sampled_widths: Sequence[int],
    num_alternates: int,
    rejection: bool,
    num_event_fields: int,
) -> list:
    """``SinkSampler``'s calls: the policy for each sampled width, with and
    without logprobs; with rejection on, the candidates of logits of each of
    ``logits_dtypes`` (a rejected draft masked when more than one row is
    sampled)."""
    calls = [
        _call(
            sink_policy_kernel,
            *sink_policy_key(
                RUNNER_IDX_DTYPE,
                num_alternates,
                rejection,
                logprobs,
                width,
                num_event_fields,
            ),
        )
        for logprobs in (False, True)
        for width in sampled_widths
    ]
    if rejection:
        calls += [
            _call(
                sink_candidates_kernel,
                *sink_candidates_key(
                    dtype, RUNNER_IDX_DTYPE, num_alternates + 1, width > 1, run
                ),
            )
            for dtype in logits_dtypes
            for run in _runs(dtype)
            for width in sampled_widths
        ]
    return _unique(calls)


def greedy_calls(
    *,
    markov: Sequence[tuple[torch.dtype, torch.dtype]] = (),
    logits: Sequence[torch.dtype] = (),
    select: Sequence[tuple[torch.dtype, int]] = (),
    finish_outputs: Sequence[torch.dtype] = (),
    rejection_widths: Sequence[int] = (),
) -> list:
    """The greedy reductions' calls: ``greedy_markov_shard_pair`` /
    ``sample_greedy_markov`` for each ``(base, bias)`` dtype pair, and
    ``greedy_shard_pair`` for each logits dtype (both the 16-byte and the
    element-wise reads, which the launch picks from the rows' alignment); the
    pair finish when there is a pair, ``select_first_argmax`` for each
    ``(output dtype, ranks)``, ``sample_greedy_markov``'s index finish for each of
    ``finish_outputs``, and greedy verification for each sampled width."""
    calls = [
        _call(partial_argmax_kernel, *partial_argmax_key(base, bias, vector))
        for base, bias in markov
        for vector in (True, False)
    ]
    calls += [
        _call(partial_argmax_kernel, *partial_argmax_key(dtype, None, vector))
        for dtype in logits
        for vector in (True, False)
    ]
    if markov or logits:
        calls.append(_call(finish_argmax_kernel, "pair"))
    calls += [_call(finish_argmax_kernel, _DTYPES[dtype]) for dtype in finish_outputs]
    calls += [
        _call(select_first_argmax_kernel, _DTYPES[dtype], ranks)
        for dtype, ranks in select
    ]
    calls += [
        _call(greedy_rejection_kernel, width, _DTYPES[dtype])
        for width in rejection_widths
        for dtype in (torch.int32, torch.int64)
    ]
    return _unique(calls)
