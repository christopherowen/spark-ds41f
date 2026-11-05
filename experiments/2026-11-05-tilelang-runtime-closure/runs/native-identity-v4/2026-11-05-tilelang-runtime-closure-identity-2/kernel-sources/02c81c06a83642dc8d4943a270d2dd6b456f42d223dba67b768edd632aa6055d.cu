
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

extern "C" __global__ void topk_log_softmax_kernel(const bfloat16_t* __restrict__ logits, float* __restrict__ output, const int* __restrict__ token_ids, int64_t logits_stride, int rows, int topk, int vocab);
extern "C" __global__ void __launch_bounds__(1024, 1) topk_log_softmax_kernel(const bfloat16_t* __restrict__ logits, float* __restrict__ output, const int* __restrict__ token_ids, int64_t logits_stride, int rows, int topk, int vocab) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* selected = ((void*)((char*)buf_dyn_shmem + 0));
  void* max_partial = ((void*)((char*)buf_dyn_shmem + 4096));
  void* sum_partial = ((void*)((char*)buf_dyn_shmem + 4096));
  void* tie_partial = ((void*)((char*)buf_dyn_shmem + 4096));
  float peak[2];
  bfloat16_t raw[16];
  float best[1];
  float maximum = 0x0p+0f/*0.000000e+00*/;
  float total[2];
  float tally[2];
  float se[1];
  float ties[1];
  float lse = 0x0p+0f/*0.000000e+00*/;
  if (((int)threadIdx.x) < topk) {
    ((float*)selected)[((int)threadIdx.x)] = ((float)logits[((((int64_t)((int)blockIdx.x)) * logits_stride) + ((int64_t)token_ids[(((int64_t)((int)blockIdx.x)) + (((int64_t)((int)threadIdx.x)) * ((int64_t)rows)))]))]);
  }
  peak[0] = -CUDART_INF_F;
  peak[1] = -CUDART_INF_F;
  if (0 < (vocab & 16383)) {
    raw[0] = logits[((((int64_t)((int)blockIdx.x)) * logits_stride) + ((int64_t)min((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)), (vocab - 1))))];
    raw[1] = logits[((((int64_t)((int)blockIdx.x)) * logits_stride) + ((int64_t)min(((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 1), (vocab - 1))))];
    raw[2] = logits[((((int64_t)((int)blockIdx.x)) * logits_stride) + ((int64_t)min(((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 2), (vocab - 1))))];
    raw[3] = logits[((((int64_t)((int)blockIdx.x)) * logits_stride) + ((int64_t)min(((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 3), (vocab - 1))))];
    raw[4] = logits[((((int64_t)((int)blockIdx.x)) * logits_stride) + ((int64_t)min(((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 4), (vocab - 1))))];
    raw[5] = logits[((((int64_t)((int)blockIdx.x)) * logits_stride) + ((int64_t)min(((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 5), (vocab - 1))))];
    raw[6] = logits[((((int64_t)((int)blockIdx.x)) * logits_stride) + ((int64_t)min(((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 6), (vocab - 1))))];
    raw[7] = logits[((((int64_t)((int)blockIdx.x)) * logits_stride) + ((int64_t)min(((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 7), (vocab - 1))))];
    raw[8] = logits[((((int64_t)((int)blockIdx.x)) * logits_stride) + ((int64_t)min(((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 8192), (vocab - 1))))];
    raw[9] = logits[((((int64_t)((int)blockIdx.x)) * logits_stride) + ((int64_t)min(((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 8193), (vocab - 1))))];
    raw[10] = logits[((((int64_t)((int)blockIdx.x)) * logits_stride) + ((int64_t)min(((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 8194), (vocab - 1))))];
    raw[11] = logits[((((int64_t)((int)blockIdx.x)) * logits_stride) + ((int64_t)min(((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 8195), (vocab - 1))))];
    raw[12] = logits[((((int64_t)((int)blockIdx.x)) * logits_stride) + ((int64_t)min(((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 8196), (vocab - 1))))];
    raw[13] = logits[((((int64_t)((int)blockIdx.x)) * logits_stride) + ((int64_t)min(((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 8197), (vocab - 1))))];
    raw[14] = logits[((((int64_t)((int)blockIdx.x)) * logits_stride) + ((int64_t)min(((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 8198), (vocab - 1))))];
    raw[15] = logits[((((int64_t)((int)blockIdx.x)) * logits_stride) + ((int64_t)min(((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 8199), (vocab - 1))))];
    float condval;
    if (((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) < vocab)) {
      condval = ((float)raw[0]);
    } else {
      condval = -CUDART_INF_F;
    }
    float value = condval;
    peak[0] = vllm_max(peak[0], value);
    float condval_1;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 1) < vocab)) {
      condval_1 = ((float)raw[1]);
    } else {
      condval_1 = -CUDART_INF_F;
    }
    float value_1 = condval_1;
    peak[0] = vllm_max(peak[0], value_1);
    float condval_2;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 2) < vocab)) {
      condval_2 = ((float)raw[2]);
    } else {
      condval_2 = -CUDART_INF_F;
    }
    float value_2 = condval_2;
    peak[0] = vllm_max(peak[0], value_2);
    float condval_3;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 3) < vocab)) {
      condval_3 = ((float)raw[3]);
    } else {
      condval_3 = -CUDART_INF_F;
    }
    float value_3 = condval_3;
    peak[0] = vllm_max(peak[0], value_3);
    float condval_4;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 4) < vocab)) {
      condval_4 = ((float)raw[4]);
    } else {
      condval_4 = -CUDART_INF_F;
    }
    float value_4 = condval_4;
    peak[0] = vllm_max(peak[0], value_4);
    float condval_5;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 5) < vocab)) {
      condval_5 = ((float)raw[5]);
    } else {
      condval_5 = -CUDART_INF_F;
    }
    float value_5 = condval_5;
    peak[0] = vllm_max(peak[0], value_5);
    float condval_6;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 6) < vocab)) {
      condval_6 = ((float)raw[6]);
    } else {
      condval_6 = -CUDART_INF_F;
    }
    float value_6 = condval_6;
    peak[0] = vllm_max(peak[0], value_6);
    float condval_7;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 7) < vocab)) {
      condval_7 = ((float)raw[7]);
    } else {
      condval_7 = -CUDART_INF_F;
    }
    float value_7 = condval_7;
    peak[0] = vllm_max(peak[0], value_7);
    float condval_8;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 8192) < vocab)) {
      condval_8 = ((float)raw[8]);
    } else {
      condval_8 = -CUDART_INF_F;
    }
    float value_8 = condval_8;
    peak[1] = vllm_max(peak[1], value_8);
    float condval_9;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 8193) < vocab)) {
      condval_9 = ((float)raw[9]);
    } else {
      condval_9 = -CUDART_INF_F;
    }
    float value_9 = condval_9;
    peak[1] = vllm_max(peak[1], value_9);
    float condval_10;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 8194) < vocab)) {
      condval_10 = ((float)raw[10]);
    } else {
      condval_10 = -CUDART_INF_F;
    }
    float value_10 = condval_10;
    peak[1] = vllm_max(peak[1], value_10);
    float condval_11;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 8195) < vocab)) {
      condval_11 = ((float)raw[11]);
    } else {
      condval_11 = -CUDART_INF_F;
    }
    float value_11 = condval_11;
    peak[1] = vllm_max(peak[1], value_11);
    float condval_12;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 8196) < vocab)) {
      condval_12 = ((float)raw[12]);
    } else {
      condval_12 = -CUDART_INF_F;
    }
    float value_12 = condval_12;
    peak[1] = vllm_max(peak[1], value_12);
    float condval_13;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 8197) < vocab)) {
      condval_13 = ((float)raw[13]);
    } else {
      condval_13 = -CUDART_INF_F;
    }
    float value_13 = condval_13;
    peak[1] = vllm_max(peak[1], value_13);
    float condval_14;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 8198) < vocab)) {
      condval_14 = ((float)raw[14]);
    } else {
      condval_14 = -CUDART_INF_F;
    }
    float value_14 = condval_14;
    peak[1] = vllm_max(peak[1], value_14);
    float condval_15;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 8199) < vocab)) {
      condval_15 = ((float)raw[15]);
    } else {
      condval_15 = -CUDART_INF_F;
    }
    float value_15 = condval_15;
    peak[1] = vllm_max(peak[1], value_15);
  }
  for (int step = 0; step < (vocab >> 14); ++step) {
    *(uint4*)(raw + 0) = *(uint4*)(logits + ((((((((int64_t)vocab) >> (int64_t)14) * (int64_t)16384) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)) + (((int64_t)((int)blockIdx.x)) * logits_stride)) - (((int64_t)step) * (int64_t)16384)) - (int64_t)16384));
    *(uint4*)(raw + 8) = *(uint4*)(logits + ((((((((int64_t)vocab) >> (int64_t)14) * (int64_t)16384) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)) + (((int64_t)((int)blockIdx.x)) * logits_stride)) - (((int64_t)step) * (int64_t)16384)) - (int64_t)8192));
    float value_16 = ((float)raw[0]);
    peak[0] = vllm_max(peak[0], value_16);
    float value_17 = ((float)raw[1]);
    peak[0] = vllm_max(peak[0], value_17);
    float value_18 = ((float)raw[2]);
    peak[0] = vllm_max(peak[0], value_18);
    float value_19 = ((float)raw[3]);
    peak[0] = vllm_max(peak[0], value_19);
    float value_20 = ((float)raw[4]);
    peak[0] = vllm_max(peak[0], value_20);
    float value_21 = ((float)raw[5]);
    peak[0] = vllm_max(peak[0], value_21);
    float value_22 = ((float)raw[6]);
    peak[0] = vllm_max(peak[0], value_22);
    float value_23 = ((float)raw[7]);
    peak[0] = vllm_max(peak[0], value_23);
    float value_24 = ((float)raw[8]);
    peak[1] = vllm_max(peak[1], value_24);
    float value_25 = ((float)raw[9]);
    peak[1] = vllm_max(peak[1], value_25);
    float value_26 = ((float)raw[10]);
    peak[1] = vllm_max(peak[1], value_26);
    float value_27 = ((float)raw[11]);
    peak[1] = vllm_max(peak[1], value_27);
    float value_28 = ((float)raw[12]);
    peak[1] = vllm_max(peak[1], value_28);
    float value_29 = ((float)raw[13]);
    peak[1] = vllm_max(peak[1], value_29);
    float value_30 = ((float)raw[14]);
    peak[1] = vllm_max(peak[1], value_30);
    float value_31 = ((float)raw[15]);
    peak[1] = vllm_max(peak[1], value_31);
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
  for (int step_1 = 0; step_1 < (vocab >> 14); ++step_1) {
    *(uint4*)(raw + 0) = *(uint4*)(logits + (((((int64_t)step_1) * (int64_t)16384) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)) + (((int64_t)((int)blockIdx.x)) * logits_stride)));
    *(uint4*)(raw + 8) = *(uint4*)(logits + ((((((int64_t)step_1) * (int64_t)16384) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)) + (((int64_t)((int)blockIdx.x)) * logits_stride)) + (int64_t)8192));
    float maximum_1 = maximum;
    float value_32 = ((float)raw[0]);
    float term = vllm_other_exp(value_32, maximum_1);
    float tie = vllm_tie(value_32, maximum_1);
    total[0] = vllm_add(total[0], term);
    tally[0] = vllm_add(tally[0], tie);
    float value_33 = ((float)raw[1]);
    float term_1 = vllm_other_exp(value_33, maximum_1);
    float tie_1 = vllm_tie(value_33, maximum_1);
    total[0] = vllm_add(total[0], term_1);
    tally[0] = vllm_add(tally[0], tie_1);
    float value_34 = ((float)raw[2]);
    float term_2 = vllm_other_exp(value_34, maximum_1);
    float tie_2 = vllm_tie(value_34, maximum_1);
    total[0] = vllm_add(total[0], term_2);
    tally[0] = vllm_add(tally[0], tie_2);
    float value_35 = ((float)raw[3]);
    float term_3 = vllm_other_exp(value_35, maximum_1);
    float tie_3 = vllm_tie(value_35, maximum_1);
    total[0] = vllm_add(total[0], term_3);
    tally[0] = vllm_add(tally[0], tie_3);
    float value_36 = ((float)raw[4]);
    float term_4 = vllm_other_exp(value_36, maximum_1);
    float tie_4 = vllm_tie(value_36, maximum_1);
    total[0] = vllm_add(total[0], term_4);
    tally[0] = vllm_add(tally[0], tie_4);
    float value_37 = ((float)raw[5]);
    float term_5 = vllm_other_exp(value_37, maximum_1);
    float tie_5 = vllm_tie(value_37, maximum_1);
    total[0] = vllm_add(total[0], term_5);
    tally[0] = vllm_add(tally[0], tie_5);
    float value_38 = ((float)raw[6]);
    float term_6 = vllm_other_exp(value_38, maximum_1);
    float tie_6 = vllm_tie(value_38, maximum_1);
    total[0] = vllm_add(total[0], term_6);
    tally[0] = vllm_add(tally[0], tie_6);
    float value_39 = ((float)raw[7]);
    float term_7 = vllm_other_exp(value_39, maximum_1);
    float tie_7 = vllm_tie(value_39, maximum_1);
    total[0] = vllm_add(total[0], term_7);
    tally[0] = vllm_add(tally[0], tie_7);
    float value_40 = ((float)raw[8]);
    float term_8 = vllm_other_exp(value_40, maximum_1);
    float tie_8 = vllm_tie(value_40, maximum_1);
    total[1] = vllm_add(total[1], term_8);
    tally[1] = vllm_add(tally[1], tie_8);
    float value_41 = ((float)raw[9]);
    float term_9 = vllm_other_exp(value_41, maximum_1);
    float tie_9 = vllm_tie(value_41, maximum_1);
    total[1] = vllm_add(total[1], term_9);
    tally[1] = vllm_add(tally[1], tie_9);
    float value_42 = ((float)raw[10]);
    float term_10 = vllm_other_exp(value_42, maximum_1);
    float tie_10 = vllm_tie(value_42, maximum_1);
    total[1] = vllm_add(total[1], term_10);
    tally[1] = vllm_add(tally[1], tie_10);
    float value_43 = ((float)raw[11]);
    float term_11 = vllm_other_exp(value_43, maximum_1);
    float tie_11 = vllm_tie(value_43, maximum_1);
    total[1] = vllm_add(total[1], term_11);
    tally[1] = vllm_add(tally[1], tie_11);
    float value_44 = ((float)raw[12]);
    float term_12 = vllm_other_exp(value_44, maximum_1);
    float tie_12 = vllm_tie(value_44, maximum_1);
    total[1] = vllm_add(total[1], term_12);
    tally[1] = vllm_add(tally[1], tie_12);
    float value_45 = ((float)raw[13]);
    float term_13 = vllm_other_exp(value_45, maximum_1);
    float tie_13 = vllm_tie(value_45, maximum_1);
    total[1] = vllm_add(total[1], term_13);
    tally[1] = vllm_add(tally[1], tie_13);
    float value_46 = ((float)raw[14]);
    float term_14 = vllm_other_exp(value_46, maximum_1);
    float tie_14 = vllm_tie(value_46, maximum_1);
    total[1] = vllm_add(total[1], term_14);
    tally[1] = vllm_add(tally[1], tie_14);
    float value_47 = ((float)raw[15]);
    float term_15 = vllm_other_exp(value_47, maximum_1);
    float tie_15 = vllm_tie(value_47, maximum_1);
    total[1] = vllm_add(total[1], term_15);
    tally[1] = vllm_add(tally[1], tie_15);
  }
  if (0 < (vocab & 16383)) {
    raw[0] = logits[((((int64_t)((int)blockIdx.x)) * logits_stride) + ((int64_t)min((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)), (vocab - 1))))];
    raw[1] = logits[((((int64_t)((int)blockIdx.x)) * logits_stride) + ((int64_t)min(((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 1), (vocab - 1))))];
    raw[2] = logits[((((int64_t)((int)blockIdx.x)) * logits_stride) + ((int64_t)min(((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 2), (vocab - 1))))];
    raw[3] = logits[((((int64_t)((int)blockIdx.x)) * logits_stride) + ((int64_t)min(((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 3), (vocab - 1))))];
    raw[4] = logits[((((int64_t)((int)blockIdx.x)) * logits_stride) + ((int64_t)min(((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 4), (vocab - 1))))];
    raw[5] = logits[((((int64_t)((int)blockIdx.x)) * logits_stride) + ((int64_t)min(((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 5), (vocab - 1))))];
    raw[6] = logits[((((int64_t)((int)blockIdx.x)) * logits_stride) + ((int64_t)min(((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 6), (vocab - 1))))];
    raw[7] = logits[((((int64_t)((int)blockIdx.x)) * logits_stride) + ((int64_t)min(((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 7), (vocab - 1))))];
    raw[8] = logits[((((int64_t)((int)blockIdx.x)) * logits_stride) + ((int64_t)min(((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 8192), (vocab - 1))))];
    raw[9] = logits[((((int64_t)((int)blockIdx.x)) * logits_stride) + ((int64_t)min(((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 8193), (vocab - 1))))];
    raw[10] = logits[((((int64_t)((int)blockIdx.x)) * logits_stride) + ((int64_t)min(((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 8194), (vocab - 1))))];
    raw[11] = logits[((((int64_t)((int)blockIdx.x)) * logits_stride) + ((int64_t)min(((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 8195), (vocab - 1))))];
    raw[12] = logits[((((int64_t)((int)blockIdx.x)) * logits_stride) + ((int64_t)min(((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 8196), (vocab - 1))))];
    raw[13] = logits[((((int64_t)((int)blockIdx.x)) * logits_stride) + ((int64_t)min(((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 8197), (vocab - 1))))];
    raw[14] = logits[((((int64_t)((int)blockIdx.x)) * logits_stride) + ((int64_t)min(((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 8198), (vocab - 1))))];
    raw[15] = logits[((((int64_t)((int)blockIdx.x)) * logits_stride) + ((int64_t)min(((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 8199), (vocab - 1))))];
    float value_48 = ((float)raw[0]);
    float condval_16;
    if (((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) < vocab)) {
      condval_16 = vllm_other_exp(value_48, maximum);
    } else {
      condval_16 = 0x0p+0f/*0.000000e+00*/;
    }
    float term_16 = condval_16;
    float condval_17;
    if (((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) < vocab)) {
      condval_17 = vllm_tie(value_48, maximum);
    } else {
      condval_17 = 0x0p+0f/*0.000000e+00*/;
    }
    float tie_16 = condval_17;
    total[0] = vllm_add(total[0], term_16);
    tally[0] = vllm_add(tally[0], tie_16);
    float value_49 = ((float)raw[1]);
    float condval_18;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 1) < vocab)) {
      condval_18 = vllm_other_exp(value_49, maximum);
    } else {
      condval_18 = 0x0p+0f/*0.000000e+00*/;
    }
    float term_17 = condval_18;
    float condval_19;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 1) < vocab)) {
      condval_19 = vllm_tie(value_49, maximum);
    } else {
      condval_19 = 0x0p+0f/*0.000000e+00*/;
    }
    float tie_17 = condval_19;
    total[0] = vllm_add(total[0], term_17);
    tally[0] = vllm_add(tally[0], tie_17);
    float value_50 = ((float)raw[2]);
    float condval_20;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 2) < vocab)) {
      condval_20 = vllm_other_exp(value_50, maximum);
    } else {
      condval_20 = 0x0p+0f/*0.000000e+00*/;
    }
    float term_18 = condval_20;
    float condval_21;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 2) < vocab)) {
      condval_21 = vllm_tie(value_50, maximum);
    } else {
      condval_21 = 0x0p+0f/*0.000000e+00*/;
    }
    float tie_18 = condval_21;
    total[0] = vllm_add(total[0], term_18);
    tally[0] = vllm_add(tally[0], tie_18);
    float value_51 = ((float)raw[3]);
    float condval_22;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 3) < vocab)) {
      condval_22 = vllm_other_exp(value_51, maximum);
    } else {
      condval_22 = 0x0p+0f/*0.000000e+00*/;
    }
    float term_19 = condval_22;
    float condval_23;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 3) < vocab)) {
      condval_23 = vllm_tie(value_51, maximum);
    } else {
      condval_23 = 0x0p+0f/*0.000000e+00*/;
    }
    float tie_19 = condval_23;
    total[0] = vllm_add(total[0], term_19);
    tally[0] = vllm_add(tally[0], tie_19);
    float value_52 = ((float)raw[4]);
    float condval_24;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 4) < vocab)) {
      condval_24 = vllm_other_exp(value_52, maximum);
    } else {
      condval_24 = 0x0p+0f/*0.000000e+00*/;
    }
    float term_20 = condval_24;
    float condval_25;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 4) < vocab)) {
      condval_25 = vllm_tie(value_52, maximum);
    } else {
      condval_25 = 0x0p+0f/*0.000000e+00*/;
    }
    float tie_20 = condval_25;
    total[0] = vllm_add(total[0], term_20);
    tally[0] = vllm_add(tally[0], tie_20);
    float value_53 = ((float)raw[5]);
    float condval_26;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 5) < vocab)) {
      condval_26 = vllm_other_exp(value_53, maximum);
    } else {
      condval_26 = 0x0p+0f/*0.000000e+00*/;
    }
    float term_21 = condval_26;
    float condval_27;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 5) < vocab)) {
      condval_27 = vllm_tie(value_53, maximum);
    } else {
      condval_27 = 0x0p+0f/*0.000000e+00*/;
    }
    float tie_21 = condval_27;
    total[0] = vllm_add(total[0], term_21);
    tally[0] = vllm_add(tally[0], tie_21);
    float value_54 = ((float)raw[6]);
    float condval_28;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 6) < vocab)) {
      condval_28 = vllm_other_exp(value_54, maximum);
    } else {
      condval_28 = 0x0p+0f/*0.000000e+00*/;
    }
    float term_22 = condval_28;
    float condval_29;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 6) < vocab)) {
      condval_29 = vllm_tie(value_54, maximum);
    } else {
      condval_29 = 0x0p+0f/*0.000000e+00*/;
    }
    float tie_22 = condval_29;
    total[0] = vllm_add(total[0], term_22);
    tally[0] = vllm_add(tally[0], tie_22);
    float value_55 = ((float)raw[7]);
    float condval_30;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 7) < vocab)) {
      condval_30 = vllm_other_exp(value_55, maximum);
    } else {
      condval_30 = 0x0p+0f/*0.000000e+00*/;
    }
    float term_23 = condval_30;
    float condval_31;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 7) < vocab)) {
      condval_31 = vllm_tie(value_55, maximum);
    } else {
      condval_31 = 0x0p+0f/*0.000000e+00*/;
    }
    float tie_23 = condval_31;
    total[0] = vllm_add(total[0], term_23);
    tally[0] = vllm_add(tally[0], tie_23);
    float value_56 = ((float)raw[8]);
    float condval_32;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 8192) < vocab)) {
      condval_32 = vllm_other_exp(value_56, maximum);
    } else {
      condval_32 = 0x0p+0f/*0.000000e+00*/;
    }
    float term_24 = condval_32;
    float condval_33;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 8192) < vocab)) {
      condval_33 = vllm_tie(value_56, maximum);
    } else {
      condval_33 = 0x0p+0f/*0.000000e+00*/;
    }
    float tie_24 = condval_33;
    total[1] = vllm_add(total[1], term_24);
    tally[1] = vllm_add(tally[1], tie_24);
    float value_57 = ((float)raw[9]);
    float condval_34;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 8193) < vocab)) {
      condval_34 = vllm_other_exp(value_57, maximum);
    } else {
      condval_34 = 0x0p+0f/*0.000000e+00*/;
    }
    float term_25 = condval_34;
    float condval_35;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 8193) < vocab)) {
      condval_35 = vllm_tie(value_57, maximum);
    } else {
      condval_35 = 0x0p+0f/*0.000000e+00*/;
    }
    float tie_25 = condval_35;
    total[1] = vllm_add(total[1], term_25);
    tally[1] = vllm_add(tally[1], tie_25);
    float value_58 = ((float)raw[10]);
    float condval_36;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 8194) < vocab)) {
      condval_36 = vllm_other_exp(value_58, maximum);
    } else {
      condval_36 = 0x0p+0f/*0.000000e+00*/;
    }
    float term_26 = condval_36;
    float condval_37;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 8194) < vocab)) {
      condval_37 = vllm_tie(value_58, maximum);
    } else {
      condval_37 = 0x0p+0f/*0.000000e+00*/;
    }
    float tie_26 = condval_37;
    total[1] = vllm_add(total[1], term_26);
    tally[1] = vllm_add(tally[1], tie_26);
    float value_59 = ((float)raw[11]);
    float condval_38;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 8195) < vocab)) {
      condval_38 = vllm_other_exp(value_59, maximum);
    } else {
      condval_38 = 0x0p+0f/*0.000000e+00*/;
    }
    float term_27 = condval_38;
    float condval_39;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 8195) < vocab)) {
      condval_39 = vllm_tie(value_59, maximum);
    } else {
      condval_39 = 0x0p+0f/*0.000000e+00*/;
    }
    float tie_27 = condval_39;
    total[1] = vllm_add(total[1], term_27);
    tally[1] = vllm_add(tally[1], tie_27);
    float value_60 = ((float)raw[12]);
    float condval_40;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 8196) < vocab)) {
      condval_40 = vllm_other_exp(value_60, maximum);
    } else {
      condval_40 = 0x0p+0f/*0.000000e+00*/;
    }
    float term_28 = condval_40;
    float condval_41;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 8196) < vocab)) {
      condval_41 = vllm_tie(value_60, maximum);
    } else {
      condval_41 = 0x0p+0f/*0.000000e+00*/;
    }
    float tie_28 = condval_41;
    total[1] = vllm_add(total[1], term_28);
    tally[1] = vllm_add(tally[1], tie_28);
    float value_61 = ((float)raw[13]);
    float condval_42;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 8197) < vocab)) {
      condval_42 = vllm_other_exp(value_61, maximum);
    } else {
      condval_42 = 0x0p+0f/*0.000000e+00*/;
    }
    float term_29 = condval_42;
    float condval_43;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 8197) < vocab)) {
      condval_43 = vllm_tie(value_61, maximum);
    } else {
      condval_43 = 0x0p+0f/*0.000000e+00*/;
    }
    float tie_29 = condval_43;
    total[1] = vllm_add(total[1], term_29);
    tally[1] = vllm_add(tally[1], tie_29);
    float value_62 = ((float)raw[14]);
    float condval_44;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 8198) < vocab)) {
      condval_44 = vllm_other_exp(value_62, maximum);
    } else {
      condval_44 = 0x0p+0f/*0.000000e+00*/;
    }
    float term_30 = condval_44;
    float condval_45;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 8198) < vocab)) {
      condval_45 = vllm_tie(value_62, maximum);
    } else {
      condval_45 = 0x0p+0f/*0.000000e+00*/;
    }
    float tie_30 = condval_45;
    total[1] = vllm_add(total[1], term_30);
    tally[1] = vllm_add(tally[1], tie_30);
    float value_63 = ((float)raw[15]);
    float condval_46;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 8199) < vocab)) {
      condval_46 = vllm_other_exp(value_63, maximum);
    } else {
      condval_46 = 0x0p+0f/*0.000000e+00*/;
    }
    float term_31 = condval_46;
    float condval_47;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 8)) + 8199) < vocab)) {
      condval_47 = vllm_tie(value_63, maximum);
    } else {
      condval_47 = 0x0p+0f/*0.000000e+00*/;
    }
    float tie_31 = condval_47;
    total[1] = vllm_add(total[1], term_31);
    tally[1] = vllm_add(tally[1], tie_31);
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
    float value_64 = vllm_sub(vllm_shifted(((float*)selected)[((int)threadIdx.x)], maximum), lse);
    output[((((int64_t)((int)blockIdx.x)) * ((int64_t)topk)) + ((int64_t)((int)threadIdx.x)))] = value_64;
  }
}

