
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
#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <math_constants.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void topk_log_softmax_kernel(const float* __restrict__ logits, float* __restrict__ output, const int* __restrict__ token_ids, int64_t logits_stride, int rows, int topk, int vocab);
extern "C" __global__ void __launch_bounds__(1024, 1) topk_log_softmax_kernel(const float* __restrict__ logits, float* __restrict__ output, const int* __restrict__ token_ids, int64_t logits_stride, int rows, int topk, int vocab) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* selected = ((void*)((char*)buf_dyn_shmem + 0));
  void* max_partial = ((void*)((char*)buf_dyn_shmem + 4096));
  void* sum_partial = ((void*)((char*)buf_dyn_shmem + 4096));
  void* tie_partial = ((void*)((char*)buf_dyn_shmem + 4096));
  float peak[2];
  float raw[8];
  float best[1];
  float maximum = 0x0p+0f/*0.000000e+00*/;
  float total[2];
  float tally[2];
  float se[1];
  float ties[1];
  float lse = 0x0p+0f/*0.000000e+00*/;
  if (((int)threadIdx.x) < topk) {
    ((float*)selected)[((int)threadIdx.x)] = logits[((((int64_t)((int)blockIdx.x)) * logits_stride) + ((int64_t)token_ids[(((int64_t)((int)blockIdx.x)) + (((int64_t)((int)threadIdx.x)) * ((int64_t)rows)))]))];
  }
  peak[0] = -CUDART_INF_F;
  peak[1] = -CUDART_INF_F;
  if (0 < (vocab & 8191)) {
    raw[0] = logits[((((int64_t)((int)blockIdx.x)) * logits_stride) + ((int64_t)min((((vocab >> 13) * 8192) + (((int)threadIdx.x) * 4)), (vocab - 1))))];
    raw[1] = logits[((((int64_t)((int)blockIdx.x)) * logits_stride) + ((int64_t)min(((((vocab >> 13) * 8192) + (((int)threadIdx.x) * 4)) + 1), (vocab - 1))))];
    raw[2] = logits[((((int64_t)((int)blockIdx.x)) * logits_stride) + ((int64_t)min(((((vocab >> 13) * 8192) + (((int)threadIdx.x) * 4)) + 2), (vocab - 1))))];
    raw[3] = logits[((((int64_t)((int)blockIdx.x)) * logits_stride) + ((int64_t)min(((((vocab >> 13) * 8192) + (((int)threadIdx.x) * 4)) + 3), (vocab - 1))))];
    raw[4] = logits[((((int64_t)((int)blockIdx.x)) * logits_stride) + ((int64_t)min(((((vocab >> 13) * 8192) + (((int)threadIdx.x) * 4)) + 4096), (vocab - 1))))];
    raw[5] = logits[((((int64_t)((int)blockIdx.x)) * logits_stride) + ((int64_t)min(((((vocab >> 13) * 8192) + (((int)threadIdx.x) * 4)) + 4097), (vocab - 1))))];
    raw[6] = logits[((((int64_t)((int)blockIdx.x)) * logits_stride) + ((int64_t)min(((((vocab >> 13) * 8192) + (((int)threadIdx.x) * 4)) + 4098), (vocab - 1))))];
    raw[7] = logits[((((int64_t)((int)blockIdx.x)) * logits_stride) + ((int64_t)min(((((vocab >> 13) * 8192) + (((int)threadIdx.x) * 4)) + 4099), (vocab - 1))))];
    float condval;
    if (((((vocab >> 13) * 8192) + (((int)threadIdx.x) * 4)) < vocab)) {
      condval = raw[0];
    } else {
      condval = -CUDART_INF_F;
    }
    float value = condval;
    peak[0] = vllm_max(peak[0], value);
    float condval_1;
    if ((((((vocab >> 13) * 8192) + (((int)threadIdx.x) * 4)) + 1) < vocab)) {
      condval_1 = raw[1];
    } else {
      condval_1 = -CUDART_INF_F;
    }
    float value_1 = condval_1;
    peak[0] = vllm_max(peak[0], value_1);
    float condval_2;
    if ((((((vocab >> 13) * 8192) + (((int)threadIdx.x) * 4)) + 2) < vocab)) {
      condval_2 = raw[2];
    } else {
      condval_2 = -CUDART_INF_F;
    }
    float value_2 = condval_2;
    peak[0] = vllm_max(peak[0], value_2);
    float condval_3;
    if ((((((vocab >> 13) * 8192) + (((int)threadIdx.x) * 4)) + 3) < vocab)) {
      condval_3 = raw[3];
    } else {
      condval_3 = -CUDART_INF_F;
    }
    float value_3 = condval_3;
    peak[0] = vllm_max(peak[0], value_3);
    float condval_4;
    if ((((((vocab >> 13) * 8192) + (((int)threadIdx.x) * 4)) + 4096) < vocab)) {
      condval_4 = raw[4];
    } else {
      condval_4 = -CUDART_INF_F;
    }
    float value_4 = condval_4;
    peak[1] = vllm_max(peak[1], value_4);
    float condval_5;
    if ((((((vocab >> 13) * 8192) + (((int)threadIdx.x) * 4)) + 4097) < vocab)) {
      condval_5 = raw[5];
    } else {
      condval_5 = -CUDART_INF_F;
    }
    float value_5 = condval_5;
    peak[1] = vllm_max(peak[1], value_5);
    float condval_6;
    if ((((((vocab >> 13) * 8192) + (((int)threadIdx.x) * 4)) + 4098) < vocab)) {
      condval_6 = raw[6];
    } else {
      condval_6 = -CUDART_INF_F;
    }
    float value_6 = condval_6;
    peak[1] = vllm_max(peak[1], value_6);
    float condval_7;
    if ((((((vocab >> 13) * 8192) + (((int)threadIdx.x) * 4)) + 4099) < vocab)) {
      condval_7 = raw[7];
    } else {
      condval_7 = -CUDART_INF_F;
    }
    float value_7 = condval_7;
    peak[1] = vllm_max(peak[1], value_7);
  }
  for (int step = 0; step < (vocab >> 13); ++step) {
    *(float4*)(raw + 0) = *(float4*)(logits + ((((((((int64_t)vocab) >> (int64_t)13) * (int64_t)8192) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)) + (((int64_t)((int)blockIdx.x)) * logits_stride)) - (((int64_t)step) * (int64_t)8192)) - (int64_t)8192));
    *(float4*)(raw + 4) = *(float4*)(logits + ((((((((int64_t)vocab) >> (int64_t)13) * (int64_t)8192) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)) + (((int64_t)((int)blockIdx.x)) * logits_stride)) - (((int64_t)step) * (int64_t)8192)) - (int64_t)4096));
    float value_8 = raw[0];
    peak[0] = vllm_max(peak[0], value_8);
    float value_9 = raw[1];
    peak[0] = vllm_max(peak[0], value_9);
    float value_10 = raw[2];
    peak[0] = vllm_max(peak[0], value_10);
    float value_11 = raw[3];
    peak[0] = vllm_max(peak[0], value_11);
    float value_12 = raw[4];
    peak[1] = vllm_max(peak[1], value_12);
    float value_13 = raw[5];
    peak[1] = vllm_max(peak[1], value_13);
    float value_14 = raw[6];
    peak[1] = vllm_max(peak[1], value_14);
    float value_15 = raw[7];
    peak[1] = vllm_max(peak[1], value_15);
  }
  best[0] = peak[0];
  best[0] = vllm_max(best[0], peak[1]);
  best[0] = vllm_max(best[0], __shfl_xor_sync((uint)4294967295, best[0], 16, 32));
  best[0] = vllm_max(best[0], __shfl_xor_sync((uint)4294967295, best[0], 8, 32));
  best[0] = vllm_max(best[0], __shfl_xor_sync((uint)4294967295, best[0], 4, 32));
  best[0] = vllm_max(best[0], __shfl_xor_sync((uint)4294967295, best[0], 2, 32));
  best[0] = vllm_max(best[0], __shfl_xor_sync((uint)4294967295, best[0], 1, 32));
  if ((((int)threadIdx.x) % 32) == 0) {
    ((float*)max_partial)[(((int)threadIdx.x) >> 5)] = best[0];
  }
  __syncthreads();
  best[0] = ((float*)max_partial)[0];
  best[0] = vllm_max(best[0], ((float*)max_partial)[1]);
  best[0] = vllm_max(best[0], ((float*)max_partial)[2]);
  best[0] = vllm_max(best[0], ((float*)max_partial)[3]);
  best[0] = vllm_max(best[0], ((float*)max_partial)[4]);
  best[0] = vllm_max(best[0], ((float*)max_partial)[5]);
  best[0] = vllm_max(best[0], ((float*)max_partial)[6]);
  best[0] = vllm_max(best[0], ((float*)max_partial)[7]);
  best[0] = vllm_max(best[0], ((float*)max_partial)[8]);
  best[0] = vllm_max(best[0], ((float*)max_partial)[9]);
  best[0] = vllm_max(best[0], ((float*)max_partial)[10]);
  best[0] = vllm_max(best[0], ((float*)max_partial)[11]);
  best[0] = vllm_max(best[0], ((float*)max_partial)[12]);
  best[0] = vllm_max(best[0], ((float*)max_partial)[13]);
  best[0] = vllm_max(best[0], ((float*)max_partial)[14]);
  best[0] = vllm_max(best[0], ((float*)max_partial)[15]);
  best[0] = vllm_max(best[0], ((float*)max_partial)[16]);
  best[0] = vllm_max(best[0], ((float*)max_partial)[17]);
  best[0] = vllm_max(best[0], ((float*)max_partial)[18]);
  best[0] = vllm_max(best[0], ((float*)max_partial)[19]);
  best[0] = vllm_max(best[0], ((float*)max_partial)[20]);
  best[0] = vllm_max(best[0], ((float*)max_partial)[21]);
  best[0] = vllm_max(best[0], ((float*)max_partial)[22]);
  best[0] = vllm_max(best[0], ((float*)max_partial)[23]);
  best[0] = vllm_max(best[0], ((float*)max_partial)[24]);
  best[0] = vllm_max(best[0], ((float*)max_partial)[25]);
  best[0] = vllm_max(best[0], ((float*)max_partial)[26]);
  best[0] = vllm_max(best[0], ((float*)max_partial)[27]);
  best[0] = vllm_max(best[0], ((float*)max_partial)[28]);
  best[0] = vllm_max(best[0], ((float*)max_partial)[29]);
  best[0] = vllm_max(best[0], ((float*)max_partial)[30]);
  best[0] = vllm_max(best[0], ((float*)max_partial)[31]);
  maximum = best[0];
  total[0] = 0x0p+0f/*0.000000e+00*/;
  tally[0] = 0x0p+0f/*0.000000e+00*/;
  total[1] = 0x0p+0f/*0.000000e+00*/;
  tally[1] = 0x0p+0f/*0.000000e+00*/;
  for (int step_1 = 0; step_1 < (vocab >> 13); ++step_1) {
    *(float4*)(raw + 0) = *(float4*)(logits + (((((int64_t)step_1) * (int64_t)8192) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)) + (((int64_t)((int)blockIdx.x)) * logits_stride)));
    *(float4*)(raw + 4) = *(float4*)(logits + ((((((int64_t)step_1) * (int64_t)8192) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)) + (((int64_t)((int)blockIdx.x)) * logits_stride)) + (int64_t)4096));
    float maximum_1 = maximum;
    float value_16 = raw[0];
    float term = vllm_other_exp(value_16, maximum_1);
    float tie = vllm_tie(value_16, maximum_1);
    total[0] = vllm_add(total[0], term);
    tally[0] = vllm_add(tally[0], tie);
    float value_17 = raw[1];
    float term_1 = vllm_other_exp(value_17, maximum_1);
    float tie_1 = vllm_tie(value_17, maximum_1);
    total[0] = vllm_add(total[0], term_1);
    tally[0] = vllm_add(tally[0], tie_1);
    float value_18 = raw[2];
    float term_2 = vllm_other_exp(value_18, maximum_1);
    float tie_2 = vllm_tie(value_18, maximum_1);
    total[0] = vllm_add(total[0], term_2);
    tally[0] = vllm_add(tally[0], tie_2);
    float value_19 = raw[3];
    float term_3 = vllm_other_exp(value_19, maximum_1);
    float tie_3 = vllm_tie(value_19, maximum_1);
    total[0] = vllm_add(total[0], term_3);
    tally[0] = vllm_add(tally[0], tie_3);
    float value_20 = raw[4];
    float term_4 = vllm_other_exp(value_20, maximum_1);
    float tie_4 = vllm_tie(value_20, maximum_1);
    total[1] = vllm_add(total[1], term_4);
    tally[1] = vllm_add(tally[1], tie_4);
    float value_21 = raw[5];
    float term_5 = vllm_other_exp(value_21, maximum_1);
    float tie_5 = vllm_tie(value_21, maximum_1);
    total[1] = vllm_add(total[1], term_5);
    tally[1] = vllm_add(tally[1], tie_5);
    float value_22 = raw[6];
    float term_6 = vllm_other_exp(value_22, maximum_1);
    float tie_6 = vllm_tie(value_22, maximum_1);
    total[1] = vllm_add(total[1], term_6);
    tally[1] = vllm_add(tally[1], tie_6);
    float value_23 = raw[7];
    float term_7 = vllm_other_exp(value_23, maximum_1);
    float tie_7 = vllm_tie(value_23, maximum_1);
    total[1] = vllm_add(total[1], term_7);
    tally[1] = vllm_add(tally[1], tie_7);
  }
  if (0 < (vocab & 8191)) {
    raw[0] = logits[((((int64_t)((int)blockIdx.x)) * logits_stride) + ((int64_t)min((((vocab >> 13) * 8192) + (((int)threadIdx.x) * 4)), (vocab - 1))))];
    raw[1] = logits[((((int64_t)((int)blockIdx.x)) * logits_stride) + ((int64_t)min(((((vocab >> 13) * 8192) + (((int)threadIdx.x) * 4)) + 1), (vocab - 1))))];
    raw[2] = logits[((((int64_t)((int)blockIdx.x)) * logits_stride) + ((int64_t)min(((((vocab >> 13) * 8192) + (((int)threadIdx.x) * 4)) + 2), (vocab - 1))))];
    raw[3] = logits[((((int64_t)((int)blockIdx.x)) * logits_stride) + ((int64_t)min(((((vocab >> 13) * 8192) + (((int)threadIdx.x) * 4)) + 3), (vocab - 1))))];
    raw[4] = logits[((((int64_t)((int)blockIdx.x)) * logits_stride) + ((int64_t)min(((((vocab >> 13) * 8192) + (((int)threadIdx.x) * 4)) + 4096), (vocab - 1))))];
    raw[5] = logits[((((int64_t)((int)blockIdx.x)) * logits_stride) + ((int64_t)min(((((vocab >> 13) * 8192) + (((int)threadIdx.x) * 4)) + 4097), (vocab - 1))))];
    raw[6] = logits[((((int64_t)((int)blockIdx.x)) * logits_stride) + ((int64_t)min(((((vocab >> 13) * 8192) + (((int)threadIdx.x) * 4)) + 4098), (vocab - 1))))];
    raw[7] = logits[((((int64_t)((int)blockIdx.x)) * logits_stride) + ((int64_t)min(((((vocab >> 13) * 8192) + (((int)threadIdx.x) * 4)) + 4099), (vocab - 1))))];
    float value_24 = raw[0];
    float condval_8;
    if (((((vocab >> 13) * 8192) + (((int)threadIdx.x) * 4)) < vocab)) {
      condval_8 = vllm_other_exp(value_24, maximum);
    } else {
      condval_8 = 0x0p+0f/*0.000000e+00*/;
    }
    float term_8 = condval_8;
    float condval_9;
    if (((((vocab >> 13) * 8192) + (((int)threadIdx.x) * 4)) < vocab)) {
      condval_9 = vllm_tie(value_24, maximum);
    } else {
      condval_9 = 0x0p+0f/*0.000000e+00*/;
    }
    float tie_8 = condval_9;
    total[0] = vllm_add(total[0], term_8);
    tally[0] = vllm_add(tally[0], tie_8);
    float value_25 = raw[1];
    float condval_10;
    if ((((((vocab >> 13) * 8192) + (((int)threadIdx.x) * 4)) + 1) < vocab)) {
      condval_10 = vllm_other_exp(value_25, maximum);
    } else {
      condval_10 = 0x0p+0f/*0.000000e+00*/;
    }
    float term_9 = condval_10;
    float condval_11;
    if ((((((vocab >> 13) * 8192) + (((int)threadIdx.x) * 4)) + 1) < vocab)) {
      condval_11 = vllm_tie(value_25, maximum);
    } else {
      condval_11 = 0x0p+0f/*0.000000e+00*/;
    }
    float tie_9 = condval_11;
    total[0] = vllm_add(total[0], term_9);
    tally[0] = vllm_add(tally[0], tie_9);
    float value_26 = raw[2];
    float condval_12;
    if ((((((vocab >> 13) * 8192) + (((int)threadIdx.x) * 4)) + 2) < vocab)) {
      condval_12 = vllm_other_exp(value_26, maximum);
    } else {
      condval_12 = 0x0p+0f/*0.000000e+00*/;
    }
    float term_10 = condval_12;
    float condval_13;
    if ((((((vocab >> 13) * 8192) + (((int)threadIdx.x) * 4)) + 2) < vocab)) {
      condval_13 = vllm_tie(value_26, maximum);
    } else {
      condval_13 = 0x0p+0f/*0.000000e+00*/;
    }
    float tie_10 = condval_13;
    total[0] = vllm_add(total[0], term_10);
    tally[0] = vllm_add(tally[0], tie_10);
    float value_27 = raw[3];
    float condval_14;
    if ((((((vocab >> 13) * 8192) + (((int)threadIdx.x) * 4)) + 3) < vocab)) {
      condval_14 = vllm_other_exp(value_27, maximum);
    } else {
      condval_14 = 0x0p+0f/*0.000000e+00*/;
    }
    float term_11 = condval_14;
    float condval_15;
    if ((((((vocab >> 13) * 8192) + (((int)threadIdx.x) * 4)) + 3) < vocab)) {
      condval_15 = vllm_tie(value_27, maximum);
    } else {
      condval_15 = 0x0p+0f/*0.000000e+00*/;
    }
    float tie_11 = condval_15;
    total[0] = vllm_add(total[0], term_11);
    tally[0] = vllm_add(tally[0], tie_11);
    float value_28 = raw[4];
    float condval_16;
    if ((((((vocab >> 13) * 8192) + (((int)threadIdx.x) * 4)) + 4096) < vocab)) {
      condval_16 = vllm_other_exp(value_28, maximum);
    } else {
      condval_16 = 0x0p+0f/*0.000000e+00*/;
    }
    float term_12 = condval_16;
    float condval_17;
    if ((((((vocab >> 13) * 8192) + (((int)threadIdx.x) * 4)) + 4096) < vocab)) {
      condval_17 = vllm_tie(value_28, maximum);
    } else {
      condval_17 = 0x0p+0f/*0.000000e+00*/;
    }
    float tie_12 = condval_17;
    total[1] = vllm_add(total[1], term_12);
    tally[1] = vllm_add(tally[1], tie_12);
    float value_29 = raw[5];
    float condval_18;
    if ((((((vocab >> 13) * 8192) + (((int)threadIdx.x) * 4)) + 4097) < vocab)) {
      condval_18 = vllm_other_exp(value_29, maximum);
    } else {
      condval_18 = 0x0p+0f/*0.000000e+00*/;
    }
    float term_13 = condval_18;
    float condval_19;
    if ((((((vocab >> 13) * 8192) + (((int)threadIdx.x) * 4)) + 4097) < vocab)) {
      condval_19 = vllm_tie(value_29, maximum);
    } else {
      condval_19 = 0x0p+0f/*0.000000e+00*/;
    }
    float tie_13 = condval_19;
    total[1] = vllm_add(total[1], term_13);
    tally[1] = vllm_add(tally[1], tie_13);
    float value_30 = raw[6];
    float condval_20;
    if ((((((vocab >> 13) * 8192) + (((int)threadIdx.x) * 4)) + 4098) < vocab)) {
      condval_20 = vllm_other_exp(value_30, maximum);
    } else {
      condval_20 = 0x0p+0f/*0.000000e+00*/;
    }
    float term_14 = condval_20;
    float condval_21;
    if ((((((vocab >> 13) * 8192) + (((int)threadIdx.x) * 4)) + 4098) < vocab)) {
      condval_21 = vllm_tie(value_30, maximum);
    } else {
      condval_21 = 0x0p+0f/*0.000000e+00*/;
    }
    float tie_14 = condval_21;
    total[1] = vllm_add(total[1], term_14);
    tally[1] = vllm_add(tally[1], tie_14);
    float value_31 = raw[7];
    float condval_22;
    if ((((((vocab >> 13) * 8192) + (((int)threadIdx.x) * 4)) + 4099) < vocab)) {
      condval_22 = vllm_other_exp(value_31, maximum);
    } else {
      condval_22 = 0x0p+0f/*0.000000e+00*/;
    }
    float term_15 = condval_22;
    float condval_23;
    if ((((((vocab >> 13) * 8192) + (((int)threadIdx.x) * 4)) + 4099) < vocab)) {
      condval_23 = vllm_tie(value_31, maximum);
    } else {
      condval_23 = 0x0p+0f/*0.000000e+00*/;
    }
    float tie_15 = condval_23;
    total[1] = vllm_add(total[1], term_15);
    tally[1] = vllm_add(tally[1], tie_15);
  }
  se[0] = total[0];
  ties[0] = tally[0];
  se[0] = vllm_add(se[0], total[1]);
  ties[0] = vllm_add(ties[0], tally[1]);
  se[0] = vllm_add(se[0], __shfl_xor_sync((uint)4294967295, se[0], 16, 32));
  se[0] = vllm_add(se[0], __shfl_xor_sync((uint)4294967295, se[0], 8, 32));
  se[0] = vllm_add(se[0], __shfl_xor_sync((uint)4294967295, se[0], 4, 32));
  se[0] = vllm_add(se[0], __shfl_xor_sync((uint)4294967295, se[0], 2, 32));
  se[0] = vllm_add(se[0], __shfl_xor_sync((uint)4294967295, se[0], 1, 32));
  __syncthreads();
  if ((((int)threadIdx.x) % 32) == 0) {
    ((float*)sum_partial)[(((int)threadIdx.x) >> 5)] = se[0];
  }
  __syncthreads();
  se[0] = ((float*)sum_partial)[0];
  se[0] = vllm_add(se[0], ((float*)sum_partial)[1]);
  se[0] = vllm_add(se[0], ((float*)sum_partial)[2]);
  se[0] = vllm_add(se[0], ((float*)sum_partial)[3]);
  se[0] = vllm_add(se[0], ((float*)sum_partial)[4]);
  se[0] = vllm_add(se[0], ((float*)sum_partial)[5]);
  se[0] = vllm_add(se[0], ((float*)sum_partial)[6]);
  se[0] = vllm_add(se[0], ((float*)sum_partial)[7]);
  se[0] = vllm_add(se[0], ((float*)sum_partial)[8]);
  se[0] = vllm_add(se[0], ((float*)sum_partial)[9]);
  se[0] = vllm_add(se[0], ((float*)sum_partial)[10]);
  se[0] = vllm_add(se[0], ((float*)sum_partial)[11]);
  se[0] = vllm_add(se[0], ((float*)sum_partial)[12]);
  se[0] = vllm_add(se[0], ((float*)sum_partial)[13]);
  se[0] = vllm_add(se[0], ((float*)sum_partial)[14]);
  se[0] = vllm_add(se[0], ((float*)sum_partial)[15]);
  se[0] = vllm_add(se[0], ((float*)sum_partial)[16]);
  se[0] = vllm_add(se[0], ((float*)sum_partial)[17]);
  se[0] = vllm_add(se[0], ((float*)sum_partial)[18]);
  se[0] = vllm_add(se[0], ((float*)sum_partial)[19]);
  se[0] = vllm_add(se[0], ((float*)sum_partial)[20]);
  se[0] = vllm_add(se[0], ((float*)sum_partial)[21]);
  se[0] = vllm_add(se[0], ((float*)sum_partial)[22]);
  se[0] = vllm_add(se[0], ((float*)sum_partial)[23]);
  se[0] = vllm_add(se[0], ((float*)sum_partial)[24]);
  se[0] = vllm_add(se[0], ((float*)sum_partial)[25]);
  se[0] = vllm_add(se[0], ((float*)sum_partial)[26]);
  se[0] = vllm_add(se[0], ((float*)sum_partial)[27]);
  se[0] = vllm_add(se[0], ((float*)sum_partial)[28]);
  se[0] = vllm_add(se[0], ((float*)sum_partial)[29]);
  se[0] = vllm_add(se[0], ((float*)sum_partial)[30]);
  se[0] = vllm_add(se[0], ((float*)sum_partial)[31]);
  ties[0] = vllm_add(ties[0], __shfl_xor_sync((uint)4294967295, ties[0], 16, 32));
  ties[0] = vllm_add(ties[0], __shfl_xor_sync((uint)4294967295, ties[0], 8, 32));
  ties[0] = vllm_add(ties[0], __shfl_xor_sync((uint)4294967295, ties[0], 4, 32));
  ties[0] = vllm_add(ties[0], __shfl_xor_sync((uint)4294967295, ties[0], 2, 32));
  ties[0] = vllm_add(ties[0], __shfl_xor_sync((uint)4294967295, ties[0], 1, 32));
  __syncthreads();
  if ((((int)threadIdx.x) % 32) == 0) {
    ((float*)tie_partial)[(((int)threadIdx.x) >> 5)] = ties[0];
  }
  __syncthreads();
  ties[0] = ((float*)tie_partial)[0];
  ties[0] = vllm_add(ties[0], ((float*)tie_partial)[1]);
  ties[0] = vllm_add(ties[0], ((float*)tie_partial)[2]);
  ties[0] = vllm_add(ties[0], ((float*)tie_partial)[3]);
  ties[0] = vllm_add(ties[0], ((float*)tie_partial)[4]);
  ties[0] = vllm_add(ties[0], ((float*)tie_partial)[5]);
  ties[0] = vllm_add(ties[0], ((float*)tie_partial)[6]);
  ties[0] = vllm_add(ties[0], ((float*)tie_partial)[7]);
  ties[0] = vllm_add(ties[0], ((float*)tie_partial)[8]);
  ties[0] = vllm_add(ties[0], ((float*)tie_partial)[9]);
  ties[0] = vllm_add(ties[0], ((float*)tie_partial)[10]);
  ties[0] = vllm_add(ties[0], ((float*)tie_partial)[11]);
  ties[0] = vllm_add(ties[0], ((float*)tie_partial)[12]);
  ties[0] = vllm_add(ties[0], ((float*)tie_partial)[13]);
  ties[0] = vllm_add(ties[0], ((float*)tie_partial)[14]);
  ties[0] = vllm_add(ties[0], ((float*)tie_partial)[15]);
  ties[0] = vllm_add(ties[0], ((float*)tie_partial)[16]);
  ties[0] = vllm_add(ties[0], ((float*)tie_partial)[17]);
  ties[0] = vllm_add(ties[0], ((float*)tie_partial)[18]);
  ties[0] = vllm_add(ties[0], ((float*)tie_partial)[19]);
  ties[0] = vllm_add(ties[0], ((float*)tie_partial)[20]);
  ties[0] = vllm_add(ties[0], ((float*)tie_partial)[21]);
  ties[0] = vllm_add(ties[0], ((float*)tie_partial)[22]);
  ties[0] = vllm_add(ties[0], ((float*)tie_partial)[23]);
  ties[0] = vllm_add(ties[0], ((float*)tie_partial)[24]);
  ties[0] = vllm_add(ties[0], ((float*)tie_partial)[25]);
  ties[0] = vllm_add(ties[0], ((float*)tie_partial)[26]);
  ties[0] = vllm_add(ties[0], ((float*)tie_partial)[27]);
  ties[0] = vllm_add(ties[0], ((float*)tie_partial)[28]);
  ties[0] = vllm_add(ties[0], ((float*)tie_partial)[29]);
  ties[0] = vllm_add(ties[0], ((float*)tie_partial)[30]);
  ties[0] = vllm_add(ties[0], ((float*)tie_partial)[31]);
  lse = vllm_softplus(vllm_others(se[0], ties[0], maximum));
  if (((int)threadIdx.x) < topk) {
    float value_32 = vllm_sub(vllm_shifted(((float*)selected)[((int)threadIdx.x)], maximum), lse);
    output[((((int64_t)((int)blockIdx.x)) * ((int64_t)topk)) + ((int64_t)((int)threadIdx.x)))] = value_32;
  }
}

