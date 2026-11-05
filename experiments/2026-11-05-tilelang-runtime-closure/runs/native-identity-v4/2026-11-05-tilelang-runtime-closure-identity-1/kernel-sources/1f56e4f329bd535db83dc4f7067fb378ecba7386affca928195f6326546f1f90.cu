
// TopKTypeConfig<float>::convert: an unsigned key that orders as the value, -0 below
// +0, every NaN above +inf.
__device__ __forceinline__ unsigned int vllm_tki_key(float v) {
  const unsigned int x = __float_as_uint(v);
  if ((x & 0x7fffffffu) > 0x7f800000u) return 0xffffffffu;
  return x ^ ((x & 0x80000000u) ? 0xffffffffu : 0x80000000u);
}

// An element's entry: its key, then the complement of its column, so that entries
// are unique and the larger one is the one torch.topk selects first.
__device__ __forceinline__ unsigned long long vllm_tki_entry(float v, int col) {
  return ((unsigned long long)vllm_tki_key(v) << 32) |
         (unsigned long long)(~(unsigned int)col);
}

// Whether a value can rank above an entry of value `floor`: it is not below it (a
// NaN value, or a NaN floor, passes).
__device__ __forceinline__ int vllm_tki_pass(float v, float floor) {
  return !(v < floor);
}

// The value of an entry's key; -inf for an empty entry (0).
__device__ __forceinline__ float vllm_tki_floor(unsigned long long entry) {
  const unsigned int key = (unsigned int)(entry >> 32);
  if (key == 0u) return __uint_as_float(0xff800000u);
  return __uint_as_float(key ^ ((key & 0x80000000u) ? 0x80000000u : 0xffffffffu));
}

__device__ __forceinline__ unsigned int vllm_tki_ballot(int flag) {
  return __ballot_sync(0xffffffffu, flag);
}

__device__ __forceinline__ unsigned long long vllm_tki_lane(unsigned long long v,
                                                            int lane) {
  return __shfl_sync(0xffffffffu, v, lane);
}

// The warp's list (lane i holds its i-th largest entry, 0 where empty) with the
// entries of the lanes in `mask` inserted, the lowest lane first: the entries below
// one move down a lane, and the last falls off.
__device__ __forceinline__ unsigned long long vllm_tki_insert(unsigned long long list,
                                                              unsigned long long entry,
                                                              unsigned int mask) {
  const int lane = threadIdx.x & 31;
  while (mask != 0u) {
    const int src = __ffs(mask) - 1;
    mask &= mask - 1u;
    const unsigned long long x = __shfl_sync(0xffffffffu, entry, src);
    const unsigned long long up = __shfl_up_sync(0xffffffffu, list, 1);
    list = (lane > 0 && x > up) ? up : (x > list ? x : list);
  }
  return list;
}

__device__ __forceinline__ bool vllm_tki_zero(unsigned int key) {
  return key == 0x80000000u || key == 0x7fffffffu;
}

// GTOp<scalar_t, true> on the values of keys a and b: a NaN is above any other
// value and equal to a NaN; otherwise a > b, -0 equal to +0.
__device__ __forceinline__ bool vllm_tki_gt(unsigned int a, unsigned int b) {
  return b != 0xffffffffu && a > b && !(vllm_tki_zero(a) && vllm_tki_zero(b));
}

// bitonicSwap between this lane's slot and the slot `stride` lanes away: the lower
// slot is A, the upper B; an invalid slot (column -1) sorts last.
__device__ __forceinline__ void vllm_tki_exchange(unsigned int& key, int& col,
                                                  int stride, bool dir) {
  const int lane = threadIdx.x & 31;
  const unsigned int other_key = __shfl_xor_sync(0xffffffffu, key, stride);
  const int other_col = __shfl_xor_sync(0xffffffffu, col, stride);
  const bool lower = (lane & stride) == 0;
  const unsigned int key_a = lower ? key : other_key;
  const unsigned int key_b = lower ? other_key : key;
  const bool valid_a = (lower ? col : other_col) >= 0;
  const bool valid_b = (lower ? other_col : col) >= 0;
  const bool swap = (vllm_tki_gt(key_a, key_b) && valid_a) || !valid_b;
  if (swap == dir) {
    key = other_key;
    col = other_col;
  }
}

// torch.topk's order of its k <= 32 selected entries, lane i holding the entry
// ranked i (i < k): gatherTopK's layout (the entries above the k-th key in column
// order, then those equal to it, in column order already), then
// bitonicSortKVInPlace over 32 slots, a slot a lane. Lane p returns the column at
// position p.
__device__ __forceinline__ int vllm_tki_order(unsigned long long entry, int k) {
  const int lane = threadIdx.x & 31;
  const unsigned int key = (unsigned int)(entry >> 32);
  const int col = (int)~(unsigned int)entry;
  const unsigned int kth = __shfl_sync(0xffffffffu, key, k - 1);
  const int above = (lane < k && key != kth) ? 1 : 0;
  int before = 0;
#pragma unroll
  for (int j = 0; j < 32; ++j) {
    const int above_j = __shfl_sync(0xffffffffu, above, j);
    const int col_j = __shfl_sync(0xffffffffu, col, j);
    before += (above_j != 0 && col_j < col) ? 1 : 0;
  }
  const int at = above ? before : lane;
  unsigned int slot_key = 0u;
  int slot_col = -1;
#pragma unroll
  for (int j = 0; j < 32; ++j) {
    const int at_j = __shfl_sync(0xffffffffu, at, j);
    const unsigned int key_j = __shfl_sync(0xffffffffu, key, j);
    const int col_j = __shfl_sync(0xffffffffu, col, j);
    if (j < k && at_j == lane) {
      slot_key = key_j;
      slot_col = col_j;
    }
  }
  // bitonicSort<32>: the flag of thread t, (t & (size / 2)) != 0, is bit `size` of
  // its slots' positions.
#pragma unroll
  for (int size = 2; size < 32; size <<= 1) {
    const bool dir = (lane & size) != 0;
#pragma unroll
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      vllm_tki_exchange(slot_key, slot_col, stride, dir);
    }
  }
#pragma unroll
  for (int stride = 16; stride > 0; stride >>= 1) {
    vllm_tki_exchange(slot_key, slot_col, stride, false);
  }
  return slot_col;
}
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

extern "C" __global__ void topk_ids_kernel(const float* __restrict__ logits, int64_t* __restrict__ out, const int64_t* __restrict__ sampled, int64_t logits_stride, int rows, int vocab, int width);
extern "C" __global__ void __launch_bounds__(512, 1) topk_ids_kernel(const float* __restrict__ logits, int64_t* __restrict__ out, const int64_t* __restrict__ sampled, int64_t logits_stride, int rows, int vocab, int width) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* chosen = ((void*)((char*)buf_dyn_shmem + 0));
  void* lists = ((void*)((char*)buf_dyn_shmem + 0));
  uint64_t best[1];
  uint64_t thr[1];
  float floor_1[1];
  float raw[32];
  uint bits[1];
  uint64_t entry[1];
  uint mask[1];
  int rank[1];
  int found[1];
  int picked[1];
  best[0] = (uint64_t)0;
  thr[0] = (uint64_t)0;
  floor_1[0] = -CUDART_INF_F;
  for (int t = 0; t < (vocab >> 14); ++t) {
    *(float4*)(raw + 0) = *(float4*)(logits + (((((int64_t)t) * (int64_t)16384) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)) + (((int64_t)((int)blockIdx.x)) * logits_stride)));
    *(float4*)(raw + 4) = *(float4*)(logits + ((((((int64_t)t) * (int64_t)16384) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)) + (((int64_t)((int)blockIdx.x)) * logits_stride)) + (int64_t)2048));
    *(float4*)(raw + 8) = *(float4*)(logits + ((((((int64_t)t) * (int64_t)16384) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)) + (((int64_t)((int)blockIdx.x)) * logits_stride)) + (int64_t)4096));
    *(float4*)(raw + 12) = *(float4*)(logits + ((((((int64_t)t) * (int64_t)16384) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)) + (((int64_t)((int)blockIdx.x)) * logits_stride)) + (int64_t)6144));
    *(float4*)(raw + 16) = *(float4*)(logits + ((((((int64_t)t) * (int64_t)16384) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)) + (((int64_t)((int)blockIdx.x)) * logits_stride)) + (int64_t)8192));
    *(float4*)(raw + 20) = *(float4*)(logits + ((((((int64_t)t) * (int64_t)16384) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)) + (((int64_t)((int)blockIdx.x)) * logits_stride)) + (int64_t)10240));
    *(float4*)(raw + 24) = *(float4*)(logits + ((((((int64_t)t) * (int64_t)16384) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)) + (((int64_t)((int)blockIdx.x)) * logits_stride)) + (int64_t)12288));
    *(float4*)(raw + 28) = *(float4*)(logits + ((((((int64_t)t) * (int64_t)16384) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)) + (((int64_t)((int)blockIdx.x)) * logits_stride)) + (int64_t)14336));
    bits[0] = (uint)0;
    int passes = vllm_tki_pass(raw[0], floor_1[0]);
    uint condval;
    if ((passes != 0)) {
      condval = (uint)1;
    } else {
      condval = (uint)0;
    }
    bits[0] = (bits[0] | condval);
    int passes_1 = vllm_tki_pass(raw[1], floor_1[0]);
    uint condval_1;
    if ((passes_1 != 0)) {
      condval_1 = (uint)2;
    } else {
      condval_1 = (uint)0;
    }
    bits[0] = (bits[0] | condval_1);
    int passes_2 = vllm_tki_pass(raw[2], floor_1[0]);
    uint condval_2;
    if ((passes_2 != 0)) {
      condval_2 = (uint)4;
    } else {
      condval_2 = (uint)0;
    }
    bits[0] = (bits[0] | condval_2);
    int passes_3 = vllm_tki_pass(raw[3], floor_1[0]);
    uint condval_3;
    if ((passes_3 != 0)) {
      condval_3 = (uint)8;
    } else {
      condval_3 = (uint)0;
    }
    bits[0] = (bits[0] | condval_3);
    int passes_4 = vllm_tki_pass(raw[4], floor_1[0]);
    uint condval_4;
    if ((passes_4 != 0)) {
      condval_4 = (uint)16;
    } else {
      condval_4 = (uint)0;
    }
    bits[0] = (bits[0] | condval_4);
    int passes_5 = vllm_tki_pass(raw[5], floor_1[0]);
    uint condval_5;
    if ((passes_5 != 0)) {
      condval_5 = (uint)32;
    } else {
      condval_5 = (uint)0;
    }
    bits[0] = (bits[0] | condval_5);
    int passes_6 = vllm_tki_pass(raw[6], floor_1[0]);
    uint condval_6;
    if ((passes_6 != 0)) {
      condval_6 = (uint)64;
    } else {
      condval_6 = (uint)0;
    }
    bits[0] = (bits[0] | condval_6);
    int passes_7 = vllm_tki_pass(raw[7], floor_1[0]);
    uint condval_7;
    if ((passes_7 != 0)) {
      condval_7 = (uint)128;
    } else {
      condval_7 = (uint)0;
    }
    bits[0] = (bits[0] | condval_7);
    int passes_8 = vllm_tki_pass(raw[8], floor_1[0]);
    uint condval_8;
    if ((passes_8 != 0)) {
      condval_8 = (uint)256;
    } else {
      condval_8 = (uint)0;
    }
    bits[0] = (bits[0] | condval_8);
    int passes_9 = vllm_tki_pass(raw[9], floor_1[0]);
    uint condval_9;
    if ((passes_9 != 0)) {
      condval_9 = (uint)512;
    } else {
      condval_9 = (uint)0;
    }
    bits[0] = (bits[0] | condval_9);
    int passes_10 = vllm_tki_pass(raw[10], floor_1[0]);
    uint condval_10;
    if ((passes_10 != 0)) {
      condval_10 = (uint)1024;
    } else {
      condval_10 = (uint)0;
    }
    bits[0] = (bits[0] | condval_10);
    int passes_11 = vllm_tki_pass(raw[11], floor_1[0]);
    uint condval_11;
    if ((passes_11 != 0)) {
      condval_11 = (uint)2048;
    } else {
      condval_11 = (uint)0;
    }
    bits[0] = (bits[0] | condval_11);
    int passes_12 = vllm_tki_pass(raw[12], floor_1[0]);
    uint condval_12;
    if ((passes_12 != 0)) {
      condval_12 = (uint)4096;
    } else {
      condval_12 = (uint)0;
    }
    bits[0] = (bits[0] | condval_12);
    int passes_13 = vllm_tki_pass(raw[13], floor_1[0]);
    uint condval_13;
    if ((passes_13 != 0)) {
      condval_13 = (uint)8192;
    } else {
      condval_13 = (uint)0;
    }
    bits[0] = (bits[0] | condval_13);
    int passes_14 = vllm_tki_pass(raw[14], floor_1[0]);
    uint condval_14;
    if ((passes_14 != 0)) {
      condval_14 = (uint)16384;
    } else {
      condval_14 = (uint)0;
    }
    bits[0] = (bits[0] | condval_14);
    int passes_15 = vllm_tki_pass(raw[15], floor_1[0]);
    uint condval_15;
    if ((passes_15 != 0)) {
      condval_15 = (uint)32768;
    } else {
      condval_15 = (uint)0;
    }
    bits[0] = (bits[0] | condval_15);
    int passes_16 = vllm_tki_pass(raw[16], floor_1[0]);
    uint condval_16;
    if ((passes_16 != 0)) {
      condval_16 = (uint)65536;
    } else {
      condval_16 = (uint)0;
    }
    bits[0] = (bits[0] | condval_16);
    int passes_17 = vllm_tki_pass(raw[17], floor_1[0]);
    uint condval_17;
    if ((passes_17 != 0)) {
      condval_17 = (uint)131072;
    } else {
      condval_17 = (uint)0;
    }
    bits[0] = (bits[0] | condval_17);
    int passes_18 = vllm_tki_pass(raw[18], floor_1[0]);
    uint condval_18;
    if ((passes_18 != 0)) {
      condval_18 = (uint)262144;
    } else {
      condval_18 = (uint)0;
    }
    bits[0] = (bits[0] | condval_18);
    int passes_19 = vllm_tki_pass(raw[19], floor_1[0]);
    uint condval_19;
    if ((passes_19 != 0)) {
      condval_19 = (uint)524288;
    } else {
      condval_19 = (uint)0;
    }
    bits[0] = (bits[0] | condval_19);
    int passes_20 = vllm_tki_pass(raw[20], floor_1[0]);
    uint condval_20;
    if ((passes_20 != 0)) {
      condval_20 = (uint)1048576;
    } else {
      condval_20 = (uint)0;
    }
    bits[0] = (bits[0] | condval_20);
    int passes_21 = vllm_tki_pass(raw[21], floor_1[0]);
    uint condval_21;
    if ((passes_21 != 0)) {
      condval_21 = (uint)2097152;
    } else {
      condval_21 = (uint)0;
    }
    bits[0] = (bits[0] | condval_21);
    int passes_22 = vllm_tki_pass(raw[22], floor_1[0]);
    uint condval_22;
    if ((passes_22 != 0)) {
      condval_22 = (uint)4194304;
    } else {
      condval_22 = (uint)0;
    }
    bits[0] = (bits[0] | condval_22);
    int passes_23 = vllm_tki_pass(raw[23], floor_1[0]);
    uint condval_23;
    if ((passes_23 != 0)) {
      condval_23 = (uint)8388608;
    } else {
      condval_23 = (uint)0;
    }
    bits[0] = (bits[0] | condval_23);
    int passes_24 = vllm_tki_pass(raw[24], floor_1[0]);
    uint condval_24;
    if ((passes_24 != 0)) {
      condval_24 = (uint)16777216;
    } else {
      condval_24 = (uint)0;
    }
    bits[0] = (bits[0] | condval_24);
    int passes_25 = vllm_tki_pass(raw[25], floor_1[0]);
    uint condval_25;
    if ((passes_25 != 0)) {
      condval_25 = (uint)33554432;
    } else {
      condval_25 = (uint)0;
    }
    bits[0] = (bits[0] | condval_25);
    int passes_26 = vllm_tki_pass(raw[26], floor_1[0]);
    uint condval_26;
    if ((passes_26 != 0)) {
      condval_26 = (uint)67108864;
    } else {
      condval_26 = (uint)0;
    }
    bits[0] = (bits[0] | condval_26);
    int passes_27 = vllm_tki_pass(raw[27], floor_1[0]);
    uint condval_27;
    if ((passes_27 != 0)) {
      condval_27 = (uint)134217728;
    } else {
      condval_27 = (uint)0;
    }
    bits[0] = (bits[0] | condval_27);
    int passes_28 = vllm_tki_pass(raw[28], floor_1[0]);
    uint condval_28;
    if ((passes_28 != 0)) {
      condval_28 = (uint)268435456;
    } else {
      condval_28 = (uint)0;
    }
    bits[0] = (bits[0] | condval_28);
    int passes_29 = vllm_tki_pass(raw[29], floor_1[0]);
    uint condval_29;
    if ((passes_29 != 0)) {
      condval_29 = (uint)536870912;
    } else {
      condval_29 = (uint)0;
    }
    bits[0] = (bits[0] | condval_29);
    int passes_30 = vllm_tki_pass(raw[30], floor_1[0]);
    uint condval_30;
    if ((passes_30 != 0)) {
      condval_30 = (uint)1073741824;
    } else {
      condval_30 = (uint)0;
    }
    bits[0] = (bits[0] | condval_30);
    int passes_31 = vllm_tki_pass(raw[31], floor_1[0]);
    uint condval_31;
    if ((passes_31 != 0)) {
      condval_31 = (uint)2147483648;
    } else {
      condval_31 = (uint)0;
    }
    bits[0] = (bits[0] | condval_31);
    bool flag = ((bits[0] & (uint)1) != (uint)0);
    bool if_condition = (vllm_tki_ballot(((int)flag)) != (uint)0);
    if (if_condition) {
      entry[0] = vllm_tki_entry(raw[0], ((t * 16384) + (((int)threadIdx.x) * 4)));
      mask[0] = vllm_tki_ballot(((int)(flag & (thr[0] < entry[0]))));
      if (mask[0] != (uint)0) {
        best[0] = vllm_tki_insert(best[0], entry[0], mask[0]);
      }
      if (mask[0] != (uint)0) {
        thr[0] = vllm_tki_lane(best[0], (width - 2));
      }
      if (mask[0] != (uint)0) {
        floor_1[0] = vllm_tki_floor(thr[0]);
      }
    }
    bool flag_1 = ((bits[0] & (uint)2) != (uint)0);
    bool if_condition_1 = (vllm_tki_ballot(((int)flag_1)) != (uint)0);
    if (if_condition_1) {
      entry[0] = vllm_tki_entry(raw[1], (((t * 16384) + (((int)threadIdx.x) * 4)) + 1));
      mask[0] = vllm_tki_ballot(((int)(flag_1 & (thr[0] < entry[0]))));
      if (mask[0] != (uint)0) {
        best[0] = vllm_tki_insert(best[0], entry[0], mask[0]);
      }
      if (mask[0] != (uint)0) {
        thr[0] = vllm_tki_lane(best[0], (width - 2));
      }
      if (mask[0] != (uint)0) {
        floor_1[0] = vllm_tki_floor(thr[0]);
      }
    }
    bool flag_2 = ((bits[0] & (uint)4) != (uint)0);
    bool if_condition_2 = (vllm_tki_ballot(((int)flag_2)) != (uint)0);
    if (if_condition_2) {
      entry[0] = vllm_tki_entry(raw[2], (((t * 16384) + (((int)threadIdx.x) * 4)) + 2));
      mask[0] = vllm_tki_ballot(((int)(flag_2 & (thr[0] < entry[0]))));
      if (mask[0] != (uint)0) {
        best[0] = vllm_tki_insert(best[0], entry[0], mask[0]);
      }
      if (mask[0] != (uint)0) {
        thr[0] = vllm_tki_lane(best[0], (width - 2));
      }
      if (mask[0] != (uint)0) {
        floor_1[0] = vllm_tki_floor(thr[0]);
      }
    }
    bool flag_3 = ((bits[0] & (uint)8) != (uint)0);
    bool if_condition_3 = (vllm_tki_ballot(((int)flag_3)) != (uint)0);
    if (if_condition_3) {
      entry[0] = vllm_tki_entry(raw[3], (((t * 16384) + (((int)threadIdx.x) * 4)) + 3));
      mask[0] = vllm_tki_ballot(((int)(flag_3 & (thr[0] < entry[0]))));
      if (mask[0] != (uint)0) {
        best[0] = vllm_tki_insert(best[0], entry[0], mask[0]);
      }
      if (mask[0] != (uint)0) {
        thr[0] = vllm_tki_lane(best[0], (width - 2));
      }
      if (mask[0] != (uint)0) {
        floor_1[0] = vllm_tki_floor(thr[0]);
      }
    }
    bool flag_4 = ((bits[0] & (uint)16) != (uint)0);
    bool if_condition_4 = (vllm_tki_ballot(((int)flag_4)) != (uint)0);
    if (if_condition_4) {
      entry[0] = vllm_tki_entry(raw[4], (((t * 16384) + (((int)threadIdx.x) * 4)) + 2048));
      mask[0] = vllm_tki_ballot(((int)(flag_4 & (thr[0] < entry[0]))));
      if (mask[0] != (uint)0) {
        best[0] = vllm_tki_insert(best[0], entry[0], mask[0]);
      }
      if (mask[0] != (uint)0) {
        thr[0] = vllm_tki_lane(best[0], (width - 2));
      }
      if (mask[0] != (uint)0) {
        floor_1[0] = vllm_tki_floor(thr[0]);
      }
    }
    bool flag_5 = ((bits[0] & (uint)32) != (uint)0);
    bool if_condition_5 = (vllm_tki_ballot(((int)flag_5)) != (uint)0);
    if (if_condition_5) {
      entry[0] = vllm_tki_entry(raw[5], (((t * 16384) + (((int)threadIdx.x) * 4)) + 2049));
      mask[0] = vllm_tki_ballot(((int)(flag_5 & (thr[0] < entry[0]))));
      if (mask[0] != (uint)0) {
        best[0] = vllm_tki_insert(best[0], entry[0], mask[0]);
      }
      if (mask[0] != (uint)0) {
        thr[0] = vllm_tki_lane(best[0], (width - 2));
      }
      if (mask[0] != (uint)0) {
        floor_1[0] = vllm_tki_floor(thr[0]);
      }
    }
    bool flag_6 = ((bits[0] & (uint)64) != (uint)0);
    bool if_condition_6 = (vllm_tki_ballot(((int)flag_6)) != (uint)0);
    if (if_condition_6) {
      entry[0] = vllm_tki_entry(raw[6], (((t * 16384) + (((int)threadIdx.x) * 4)) + 2050));
      mask[0] = vllm_tki_ballot(((int)(flag_6 & (thr[0] < entry[0]))));
      if (mask[0] != (uint)0) {
        best[0] = vllm_tki_insert(best[0], entry[0], mask[0]);
      }
      if (mask[0] != (uint)0) {
        thr[0] = vllm_tki_lane(best[0], (width - 2));
      }
      if (mask[0] != (uint)0) {
        floor_1[0] = vllm_tki_floor(thr[0]);
      }
    }
    bool flag_7 = ((bits[0] & (uint)128) != (uint)0);
    bool if_condition_7 = (vllm_tki_ballot(((int)flag_7)) != (uint)0);
    if (if_condition_7) {
      entry[0] = vllm_tki_entry(raw[7], (((t * 16384) + (((int)threadIdx.x) * 4)) + 2051));
      mask[0] = vllm_tki_ballot(((int)(flag_7 & (thr[0] < entry[0]))));
      if (mask[0] != (uint)0) {
        best[0] = vllm_tki_insert(best[0], entry[0], mask[0]);
      }
      if (mask[0] != (uint)0) {
        thr[0] = vllm_tki_lane(best[0], (width - 2));
      }
      if (mask[0] != (uint)0) {
        floor_1[0] = vllm_tki_floor(thr[0]);
      }
    }
    bool flag_8 = ((bits[0] & (uint)256) != (uint)0);
    bool if_condition_8 = (vllm_tki_ballot(((int)flag_8)) != (uint)0);
    if (if_condition_8) {
      entry[0] = vllm_tki_entry(raw[8], (((t * 16384) + (((int)threadIdx.x) * 4)) + 4096));
      mask[0] = vllm_tki_ballot(((int)(flag_8 & (thr[0] < entry[0]))));
      if (mask[0] != (uint)0) {
        best[0] = vllm_tki_insert(best[0], entry[0], mask[0]);
      }
      if (mask[0] != (uint)0) {
        thr[0] = vllm_tki_lane(best[0], (width - 2));
      }
      if (mask[0] != (uint)0) {
        floor_1[0] = vllm_tki_floor(thr[0]);
      }
    }
    bool flag_9 = ((bits[0] & (uint)512) != (uint)0);
    bool if_condition_9 = (vllm_tki_ballot(((int)flag_9)) != (uint)0);
    if (if_condition_9) {
      entry[0] = vllm_tki_entry(raw[9], (((t * 16384) + (((int)threadIdx.x) * 4)) + 4097));
      mask[0] = vllm_tki_ballot(((int)(flag_9 & (thr[0] < entry[0]))));
      if (mask[0] != (uint)0) {
        best[0] = vllm_tki_insert(best[0], entry[0], mask[0]);
      }
      if (mask[0] != (uint)0) {
        thr[0] = vllm_tki_lane(best[0], (width - 2));
      }
      if (mask[0] != (uint)0) {
        floor_1[0] = vllm_tki_floor(thr[0]);
      }
    }
    bool flag_10 = ((bits[0] & (uint)1024) != (uint)0);
    bool if_condition_10 = (vllm_tki_ballot(((int)flag_10)) != (uint)0);
    if (if_condition_10) {
      entry[0] = vllm_tki_entry(raw[10], (((t * 16384) + (((int)threadIdx.x) * 4)) + 4098));
      mask[0] = vllm_tki_ballot(((int)(flag_10 & (thr[0] < entry[0]))));
      if (mask[0] != (uint)0) {
        best[0] = vllm_tki_insert(best[0], entry[0], mask[0]);
      }
      if (mask[0] != (uint)0) {
        thr[0] = vllm_tki_lane(best[0], (width - 2));
      }
      if (mask[0] != (uint)0) {
        floor_1[0] = vllm_tki_floor(thr[0]);
      }
    }
    bool flag_11 = ((bits[0] & (uint)2048) != (uint)0);
    bool if_condition_11 = (vllm_tki_ballot(((int)flag_11)) != (uint)0);
    if (if_condition_11) {
      entry[0] = vllm_tki_entry(raw[11], (((t * 16384) + (((int)threadIdx.x) * 4)) + 4099));
      mask[0] = vllm_tki_ballot(((int)(flag_11 & (thr[0] < entry[0]))));
      if (mask[0] != (uint)0) {
        best[0] = vllm_tki_insert(best[0], entry[0], mask[0]);
      }
      if (mask[0] != (uint)0) {
        thr[0] = vllm_tki_lane(best[0], (width - 2));
      }
      if (mask[0] != (uint)0) {
        floor_1[0] = vllm_tki_floor(thr[0]);
      }
    }
    bool flag_12 = ((bits[0] & (uint)4096) != (uint)0);
    bool if_condition_12 = (vllm_tki_ballot(((int)flag_12)) != (uint)0);
    if (if_condition_12) {
      entry[0] = vllm_tki_entry(raw[12], (((t * 16384) + (((int)threadIdx.x) * 4)) + 6144));
      mask[0] = vllm_tki_ballot(((int)(flag_12 & (thr[0] < entry[0]))));
      if (mask[0] != (uint)0) {
        best[0] = vllm_tki_insert(best[0], entry[0], mask[0]);
      }
      if (mask[0] != (uint)0) {
        thr[0] = vllm_tki_lane(best[0], (width - 2));
      }
      if (mask[0] != (uint)0) {
        floor_1[0] = vllm_tki_floor(thr[0]);
      }
    }
    bool flag_13 = ((bits[0] & (uint)8192) != (uint)0);
    bool if_condition_13 = (vllm_tki_ballot(((int)flag_13)) != (uint)0);
    if (if_condition_13) {
      entry[0] = vllm_tki_entry(raw[13], (((t * 16384) + (((int)threadIdx.x) * 4)) + 6145));
      mask[0] = vllm_tki_ballot(((int)(flag_13 & (thr[0] < entry[0]))));
      if (mask[0] != (uint)0) {
        best[0] = vllm_tki_insert(best[0], entry[0], mask[0]);
      }
      if (mask[0] != (uint)0) {
        thr[0] = vllm_tki_lane(best[0], (width - 2));
      }
      if (mask[0] != (uint)0) {
        floor_1[0] = vllm_tki_floor(thr[0]);
      }
    }
    bool flag_14 = ((bits[0] & (uint)16384) != (uint)0);
    bool if_condition_14 = (vllm_tki_ballot(((int)flag_14)) != (uint)0);
    if (if_condition_14) {
      entry[0] = vllm_tki_entry(raw[14], (((t * 16384) + (((int)threadIdx.x) * 4)) + 6146));
      mask[0] = vllm_tki_ballot(((int)(flag_14 & (thr[0] < entry[0]))));
      if (mask[0] != (uint)0) {
        best[0] = vllm_tki_insert(best[0], entry[0], mask[0]);
      }
      if (mask[0] != (uint)0) {
        thr[0] = vllm_tki_lane(best[0], (width - 2));
      }
      if (mask[0] != (uint)0) {
        floor_1[0] = vllm_tki_floor(thr[0]);
      }
    }
    bool flag_15 = ((bits[0] & (uint)32768) != (uint)0);
    bool if_condition_15 = (vllm_tki_ballot(((int)flag_15)) != (uint)0);
    if (if_condition_15) {
      entry[0] = vllm_tki_entry(raw[15], (((t * 16384) + (((int)threadIdx.x) * 4)) + 6147));
      mask[0] = vllm_tki_ballot(((int)(flag_15 & (thr[0] < entry[0]))));
      if (mask[0] != (uint)0) {
        best[0] = vllm_tki_insert(best[0], entry[0], mask[0]);
      }
      if (mask[0] != (uint)0) {
        thr[0] = vllm_tki_lane(best[0], (width - 2));
      }
      if (mask[0] != (uint)0) {
        floor_1[0] = vllm_tki_floor(thr[0]);
      }
    }
    bool flag_16 = ((bits[0] & (uint)65536) != (uint)0);
    bool if_condition_16 = (vllm_tki_ballot(((int)flag_16)) != (uint)0);
    if (if_condition_16) {
      entry[0] = vllm_tki_entry(raw[16], (((t * 16384) + (((int)threadIdx.x) * 4)) + 8192));
      mask[0] = vllm_tki_ballot(((int)(flag_16 & (thr[0] < entry[0]))));
      if (mask[0] != (uint)0) {
        best[0] = vllm_tki_insert(best[0], entry[0], mask[0]);
      }
      if (mask[0] != (uint)0) {
        thr[0] = vllm_tki_lane(best[0], (width - 2));
      }
      if (mask[0] != (uint)0) {
        floor_1[0] = vllm_tki_floor(thr[0]);
      }
    }
    bool flag_17 = ((bits[0] & (uint)131072) != (uint)0);
    bool if_condition_17 = (vllm_tki_ballot(((int)flag_17)) != (uint)0);
    if (if_condition_17) {
      entry[0] = vllm_tki_entry(raw[17], (((t * 16384) + (((int)threadIdx.x) * 4)) + 8193));
      mask[0] = vllm_tki_ballot(((int)(flag_17 & (thr[0] < entry[0]))));
      if (mask[0] != (uint)0) {
        best[0] = vllm_tki_insert(best[0], entry[0], mask[0]);
      }
      if (mask[0] != (uint)0) {
        thr[0] = vllm_tki_lane(best[0], (width - 2));
      }
      if (mask[0] != (uint)0) {
        floor_1[0] = vllm_tki_floor(thr[0]);
      }
    }
    bool flag_18 = ((bits[0] & (uint)262144) != (uint)0);
    bool if_condition_18 = (vllm_tki_ballot(((int)flag_18)) != (uint)0);
    if (if_condition_18) {
      entry[0] = vllm_tki_entry(raw[18], (((t * 16384) + (((int)threadIdx.x) * 4)) + 8194));
      mask[0] = vllm_tki_ballot(((int)(flag_18 & (thr[0] < entry[0]))));
      if (mask[0] != (uint)0) {
        best[0] = vllm_tki_insert(best[0], entry[0], mask[0]);
      }
      if (mask[0] != (uint)0) {
        thr[0] = vllm_tki_lane(best[0], (width - 2));
      }
      if (mask[0] != (uint)0) {
        floor_1[0] = vllm_tki_floor(thr[0]);
      }
    }
    bool flag_19 = ((bits[0] & (uint)524288) != (uint)0);
    bool if_condition_19 = (vllm_tki_ballot(((int)flag_19)) != (uint)0);
    if (if_condition_19) {
      entry[0] = vllm_tki_entry(raw[19], (((t * 16384) + (((int)threadIdx.x) * 4)) + 8195));
      mask[0] = vllm_tki_ballot(((int)(flag_19 & (thr[0] < entry[0]))));
      if (mask[0] != (uint)0) {
        best[0] = vllm_tki_insert(best[0], entry[0], mask[0]);
      }
      if (mask[0] != (uint)0) {
        thr[0] = vllm_tki_lane(best[0], (width - 2));
      }
      if (mask[0] != (uint)0) {
        floor_1[0] = vllm_tki_floor(thr[0]);
      }
    }
    bool flag_20 = ((bits[0] & (uint)1048576) != (uint)0);
    bool if_condition_20 = (vllm_tki_ballot(((int)flag_20)) != (uint)0);
    if (if_condition_20) {
      entry[0] = vllm_tki_entry(raw[20], (((t * 16384) + (((int)threadIdx.x) * 4)) + 10240));
      mask[0] = vllm_tki_ballot(((int)(flag_20 & (thr[0] < entry[0]))));
      if (mask[0] != (uint)0) {
        best[0] = vllm_tki_insert(best[0], entry[0], mask[0]);
      }
      if (mask[0] != (uint)0) {
        thr[0] = vllm_tki_lane(best[0], (width - 2));
      }
      if (mask[0] != (uint)0) {
        floor_1[0] = vllm_tki_floor(thr[0]);
      }
    }
    bool flag_21 = ((bits[0] & (uint)2097152) != (uint)0);
    bool if_condition_21 = (vllm_tki_ballot(((int)flag_21)) != (uint)0);
    if (if_condition_21) {
      entry[0] = vllm_tki_entry(raw[21], (((t * 16384) + (((int)threadIdx.x) * 4)) + 10241));
      mask[0] = vllm_tki_ballot(((int)(flag_21 & (thr[0] < entry[0]))));
      if (mask[0] != (uint)0) {
        best[0] = vllm_tki_insert(best[0], entry[0], mask[0]);
      }
      if (mask[0] != (uint)0) {
        thr[0] = vllm_tki_lane(best[0], (width - 2));
      }
      if (mask[0] != (uint)0) {
        floor_1[0] = vllm_tki_floor(thr[0]);
      }
    }
    bool flag_22 = ((bits[0] & (uint)4194304) != (uint)0);
    bool if_condition_22 = (vllm_tki_ballot(((int)flag_22)) != (uint)0);
    if (if_condition_22) {
      entry[0] = vllm_tki_entry(raw[22], (((t * 16384) + (((int)threadIdx.x) * 4)) + 10242));
      mask[0] = vllm_tki_ballot(((int)(flag_22 & (thr[0] < entry[0]))));
      if (mask[0] != (uint)0) {
        best[0] = vllm_tki_insert(best[0], entry[0], mask[0]);
      }
      if (mask[0] != (uint)0) {
        thr[0] = vllm_tki_lane(best[0], (width - 2));
      }
      if (mask[0] != (uint)0) {
        floor_1[0] = vllm_tki_floor(thr[0]);
      }
    }
    bool flag_23 = ((bits[0] & (uint)8388608) != (uint)0);
    bool if_condition_23 = (vllm_tki_ballot(((int)flag_23)) != (uint)0);
    if (if_condition_23) {
      entry[0] = vllm_tki_entry(raw[23], (((t * 16384) + (((int)threadIdx.x) * 4)) + 10243));
      mask[0] = vllm_tki_ballot(((int)(flag_23 & (thr[0] < entry[0]))));
      if (mask[0] != (uint)0) {
        best[0] = vllm_tki_insert(best[0], entry[0], mask[0]);
      }
      if (mask[0] != (uint)0) {
        thr[0] = vllm_tki_lane(best[0], (width - 2));
      }
      if (mask[0] != (uint)0) {
        floor_1[0] = vllm_tki_floor(thr[0]);
      }
    }
    bool flag_24 = ((bits[0] & (uint)16777216) != (uint)0);
    bool if_condition_24 = (vllm_tki_ballot(((int)flag_24)) != (uint)0);
    if (if_condition_24) {
      entry[0] = vllm_tki_entry(raw[24], (((t * 16384) + (((int)threadIdx.x) * 4)) + 12288));
      mask[0] = vllm_tki_ballot(((int)(flag_24 & (thr[0] < entry[0]))));
      if (mask[0] != (uint)0) {
        best[0] = vllm_tki_insert(best[0], entry[0], mask[0]);
      }
      if (mask[0] != (uint)0) {
        thr[0] = vllm_tki_lane(best[0], (width - 2));
      }
      if (mask[0] != (uint)0) {
        floor_1[0] = vllm_tki_floor(thr[0]);
      }
    }
    bool flag_25 = ((bits[0] & (uint)33554432) != (uint)0);
    bool if_condition_25 = (vllm_tki_ballot(((int)flag_25)) != (uint)0);
    if (if_condition_25) {
      entry[0] = vllm_tki_entry(raw[25], (((t * 16384) + (((int)threadIdx.x) * 4)) + 12289));
      mask[0] = vllm_tki_ballot(((int)(flag_25 & (thr[0] < entry[0]))));
      if (mask[0] != (uint)0) {
        best[0] = vllm_tki_insert(best[0], entry[0], mask[0]);
      }
      if (mask[0] != (uint)0) {
        thr[0] = vllm_tki_lane(best[0], (width - 2));
      }
      if (mask[0] != (uint)0) {
        floor_1[0] = vllm_tki_floor(thr[0]);
      }
    }
    bool flag_26 = ((bits[0] & (uint)67108864) != (uint)0);
    bool if_condition_26 = (vllm_tki_ballot(((int)flag_26)) != (uint)0);
    if (if_condition_26) {
      entry[0] = vllm_tki_entry(raw[26], (((t * 16384) + (((int)threadIdx.x) * 4)) + 12290));
      mask[0] = vllm_tki_ballot(((int)(flag_26 & (thr[0] < entry[0]))));
      if (mask[0] != (uint)0) {
        best[0] = vllm_tki_insert(best[0], entry[0], mask[0]);
      }
      if (mask[0] != (uint)0) {
        thr[0] = vllm_tki_lane(best[0], (width - 2));
      }
      if (mask[0] != (uint)0) {
        floor_1[0] = vllm_tki_floor(thr[0]);
      }
    }
    bool flag_27 = ((bits[0] & (uint)134217728) != (uint)0);
    bool if_condition_27 = (vllm_tki_ballot(((int)flag_27)) != (uint)0);
    if (if_condition_27) {
      entry[0] = vllm_tki_entry(raw[27], (((t * 16384) + (((int)threadIdx.x) * 4)) + 12291));
      mask[0] = vllm_tki_ballot(((int)(flag_27 & (thr[0] < entry[0]))));
      if (mask[0] != (uint)0) {
        best[0] = vllm_tki_insert(best[0], entry[0], mask[0]);
      }
      if (mask[0] != (uint)0) {
        thr[0] = vllm_tki_lane(best[0], (width - 2));
      }
      if (mask[0] != (uint)0) {
        floor_1[0] = vllm_tki_floor(thr[0]);
      }
    }
    bool flag_28 = ((bits[0] & (uint)268435456) != (uint)0);
    bool if_condition_28 = (vllm_tki_ballot(((int)flag_28)) != (uint)0);
    if (if_condition_28) {
      entry[0] = vllm_tki_entry(raw[28], (((t * 16384) + (((int)threadIdx.x) * 4)) + 14336));
      mask[0] = vllm_tki_ballot(((int)(flag_28 & (thr[0] < entry[0]))));
      if (mask[0] != (uint)0) {
        best[0] = vllm_tki_insert(best[0], entry[0], mask[0]);
      }
      if (mask[0] != (uint)0) {
        thr[0] = vllm_tki_lane(best[0], (width - 2));
      }
      if (mask[0] != (uint)0) {
        floor_1[0] = vllm_tki_floor(thr[0]);
      }
    }
    bool flag_29 = ((bits[0] & (uint)536870912) != (uint)0);
    bool if_condition_29 = (vllm_tki_ballot(((int)flag_29)) != (uint)0);
    if (if_condition_29) {
      entry[0] = vllm_tki_entry(raw[29], (((t * 16384) + (((int)threadIdx.x) * 4)) + 14337));
      mask[0] = vllm_tki_ballot(((int)(flag_29 & (thr[0] < entry[0]))));
      if (mask[0] != (uint)0) {
        best[0] = vllm_tki_insert(best[0], entry[0], mask[0]);
      }
      if (mask[0] != (uint)0) {
        thr[0] = vllm_tki_lane(best[0], (width - 2));
      }
      if (mask[0] != (uint)0) {
        floor_1[0] = vllm_tki_floor(thr[0]);
      }
    }
    bool flag_30 = ((bits[0] & (uint)1073741824) != (uint)0);
    bool if_condition_30 = (vllm_tki_ballot(((int)flag_30)) != (uint)0);
    if (if_condition_30) {
      entry[0] = vllm_tki_entry(raw[30], (((t * 16384) + (((int)threadIdx.x) * 4)) + 14338));
      mask[0] = vllm_tki_ballot(((int)(flag_30 & (thr[0] < entry[0]))));
      if (mask[0] != (uint)0) {
        best[0] = vllm_tki_insert(best[0], entry[0], mask[0]);
      }
      if (mask[0] != (uint)0) {
        thr[0] = vllm_tki_lane(best[0], (width - 2));
      }
      if (mask[0] != (uint)0) {
        floor_1[0] = vllm_tki_floor(thr[0]);
      }
    }
    bool flag_31 = ((bits[0] & (uint)2147483648) != (uint)0);
    bool if_condition_31 = (vllm_tki_ballot(((int)flag_31)) != (uint)0);
    if (if_condition_31) {
      entry[0] = vllm_tki_entry(raw[31], (((t * 16384) + (((int)threadIdx.x) * 4)) + 14339));
      mask[0] = vllm_tki_ballot(((int)(flag_31 & (thr[0] < entry[0]))));
      if (mask[0] != (uint)0) {
        best[0] = vllm_tki_insert(best[0], entry[0], mask[0]);
      }
      if (mask[0] != (uint)0) {
        thr[0] = vllm_tki_lane(best[0], (width - 2));
      }
      if (mask[0] != (uint)0) {
        floor_1[0] = vllm_tki_floor(thr[0]);
      }
    }
  }
  if (0 < (vocab & 16383)) {
    if (((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 4) <= vocab) {
      *(float4*)(raw + 0) = *(float4*)(logits + ((((((int64_t)vocab) >> (int64_t)14) * (int64_t)16384) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)) + (((int64_t)((int)blockIdx.x)) * logits_stride)));
    } else {
      #pragma unroll
      for (int v = 0; v < 4; ++v) {
        if (((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + v) < vocab) {
          raw[v] = logits[(((((((int64_t)vocab) >> (int64_t)14) * (int64_t)16384) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)) + (((int64_t)((int)blockIdx.x)) * logits_stride)) + ((int64_t)v))];
        }
      }
    }
    if (((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 2052) <= vocab) {
      *(float4*)(raw + 4) = *(float4*)(logits + (((((((int64_t)vocab) >> (int64_t)14) * (int64_t)16384) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)) + (((int64_t)((int)blockIdx.x)) * logits_stride)) + (int64_t)2048));
    } else {
      #pragma unroll
      for (int v_1 = 0; v_1 < 4; ++v_1) {
        if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + v_1) + 2048) < vocab) {
          raw[(v_1 + 4)] = logits[((((((((int64_t)vocab) >> (int64_t)14) * (int64_t)16384) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)) + (((int64_t)((int)blockIdx.x)) * logits_stride)) + ((int64_t)v_1)) + (int64_t)2048)];
        }
      }
    }
    if (((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 4100) <= vocab) {
      *(float4*)(raw + 8) = *(float4*)(logits + (((((((int64_t)vocab) >> (int64_t)14) * (int64_t)16384) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)) + (((int64_t)((int)blockIdx.x)) * logits_stride)) + (int64_t)4096));
    } else {
      #pragma unroll
      for (int v_2 = 0; v_2 < 4; ++v_2) {
        if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + v_2) + 4096) < vocab) {
          raw[(v_2 + 8)] = logits[((((((((int64_t)vocab) >> (int64_t)14) * (int64_t)16384) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)) + (((int64_t)((int)blockIdx.x)) * logits_stride)) + ((int64_t)v_2)) + (int64_t)4096)];
        }
      }
    }
    if (((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 6148) <= vocab) {
      *(float4*)(raw + 12) = *(float4*)(logits + (((((((int64_t)vocab) >> (int64_t)14) * (int64_t)16384) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)) + (((int64_t)((int)blockIdx.x)) * logits_stride)) + (int64_t)6144));
    } else {
      #pragma unroll
      for (int v_3 = 0; v_3 < 4; ++v_3) {
        if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + v_3) + 6144) < vocab) {
          raw[(v_3 + 12)] = logits[((((((((int64_t)vocab) >> (int64_t)14) * (int64_t)16384) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)) + (((int64_t)((int)blockIdx.x)) * logits_stride)) + ((int64_t)v_3)) + (int64_t)6144)];
        }
      }
    }
    if (((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 8196) <= vocab) {
      *(float4*)(raw + 16) = *(float4*)(logits + (((((((int64_t)vocab) >> (int64_t)14) * (int64_t)16384) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)) + (((int64_t)((int)blockIdx.x)) * logits_stride)) + (int64_t)8192));
    } else {
      #pragma unroll
      for (int v_4 = 0; v_4 < 4; ++v_4) {
        if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + v_4) + 8192) < vocab) {
          raw[(v_4 + 16)] = logits[((((((((int64_t)vocab) >> (int64_t)14) * (int64_t)16384) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)) + (((int64_t)((int)blockIdx.x)) * logits_stride)) + ((int64_t)v_4)) + (int64_t)8192)];
        }
      }
    }
    if (((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 10244) <= vocab) {
      *(float4*)(raw + 20) = *(float4*)(logits + (((((((int64_t)vocab) >> (int64_t)14) * (int64_t)16384) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)) + (((int64_t)((int)blockIdx.x)) * logits_stride)) + (int64_t)10240));
    } else {
      #pragma unroll
      for (int v_5 = 0; v_5 < 4; ++v_5) {
        if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + v_5) + 10240) < vocab) {
          raw[(v_5 + 20)] = logits[((((((((int64_t)vocab) >> (int64_t)14) * (int64_t)16384) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)) + (((int64_t)((int)blockIdx.x)) * logits_stride)) + ((int64_t)v_5)) + (int64_t)10240)];
        }
      }
    }
    if (((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 12292) <= vocab) {
      *(float4*)(raw + 24) = *(float4*)(logits + (((((((int64_t)vocab) >> (int64_t)14) * (int64_t)16384) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)) + (((int64_t)((int)blockIdx.x)) * logits_stride)) + (int64_t)12288));
    } else {
      #pragma unroll
      for (int v_6 = 0; v_6 < 4; ++v_6) {
        if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + v_6) + 12288) < vocab) {
          raw[(v_6 + 24)] = logits[((((((((int64_t)vocab) >> (int64_t)14) * (int64_t)16384) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)) + (((int64_t)((int)blockIdx.x)) * logits_stride)) + ((int64_t)v_6)) + (int64_t)12288)];
        }
      }
    }
    if (((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 14340) <= vocab) {
      *(float4*)(raw + 28) = *(float4*)(logits + (((((((int64_t)vocab) >> (int64_t)14) * (int64_t)16384) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)) + (((int64_t)((int)blockIdx.x)) * logits_stride)) + (int64_t)14336));
    } else {
      #pragma unroll
      for (int v_7 = 0; v_7 < 4; ++v_7) {
        if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + v_7) + 14336) < vocab) {
          raw[(v_7 + 28)] = logits[((((((((int64_t)vocab) >> (int64_t)14) * (int64_t)16384) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)) + (((int64_t)((int)blockIdx.x)) * logits_stride)) + ((int64_t)v_7)) + (int64_t)14336)];
        }
      }
    }
    bits[0] = (uint)0;
    int passes_32 = vllm_tki_pass(raw[0], floor_1[0]);
    int condval_33;
    if (((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) < vocab)) {
      condval_33 = passes_32;
    } else {
      condval_33 = 0;
    }
    uint condval_32;
    if ((condval_33 != 0)) {
      condval_32 = (uint)1;
    } else {
      condval_32 = (uint)0;
    }
    bits[0] = (bits[0] | condval_32);
    int passes_33 = vllm_tki_pass(raw[1], floor_1[0]);
    int condval_35;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 1) < vocab)) {
      condval_35 = passes_33;
    } else {
      condval_35 = 0;
    }
    uint condval_34;
    if ((condval_35 != 0)) {
      condval_34 = (uint)2;
    } else {
      condval_34 = (uint)0;
    }
    bits[0] = (bits[0] | condval_34);
    int passes_34 = vllm_tki_pass(raw[2], floor_1[0]);
    int condval_37;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 2) < vocab)) {
      condval_37 = passes_34;
    } else {
      condval_37 = 0;
    }
    uint condval_36;
    if ((condval_37 != 0)) {
      condval_36 = (uint)4;
    } else {
      condval_36 = (uint)0;
    }
    bits[0] = (bits[0] | condval_36);
    int passes_35 = vllm_tki_pass(raw[3], floor_1[0]);
    int condval_39;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 3) < vocab)) {
      condval_39 = passes_35;
    } else {
      condval_39 = 0;
    }
    uint condval_38;
    if ((condval_39 != 0)) {
      condval_38 = (uint)8;
    } else {
      condval_38 = (uint)0;
    }
    bits[0] = (bits[0] | condval_38);
    int passes_36 = vllm_tki_pass(raw[4], floor_1[0]);
    int condval_41;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 2048) < vocab)) {
      condval_41 = passes_36;
    } else {
      condval_41 = 0;
    }
    uint condval_40;
    if ((condval_41 != 0)) {
      condval_40 = (uint)16;
    } else {
      condval_40 = (uint)0;
    }
    bits[0] = (bits[0] | condval_40);
    int passes_37 = vllm_tki_pass(raw[5], floor_1[0]);
    int condval_43;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 2049) < vocab)) {
      condval_43 = passes_37;
    } else {
      condval_43 = 0;
    }
    uint condval_42;
    if ((condval_43 != 0)) {
      condval_42 = (uint)32;
    } else {
      condval_42 = (uint)0;
    }
    bits[0] = (bits[0] | condval_42);
    int passes_38 = vllm_tki_pass(raw[6], floor_1[0]);
    int condval_45;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 2050) < vocab)) {
      condval_45 = passes_38;
    } else {
      condval_45 = 0;
    }
    uint condval_44;
    if ((condval_45 != 0)) {
      condval_44 = (uint)64;
    } else {
      condval_44 = (uint)0;
    }
    bits[0] = (bits[0] | condval_44);
    int passes_39 = vllm_tki_pass(raw[7], floor_1[0]);
    int condval_47;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 2051) < vocab)) {
      condval_47 = passes_39;
    } else {
      condval_47 = 0;
    }
    uint condval_46;
    if ((condval_47 != 0)) {
      condval_46 = (uint)128;
    } else {
      condval_46 = (uint)0;
    }
    bits[0] = (bits[0] | condval_46);
    int passes_40 = vllm_tki_pass(raw[8], floor_1[0]);
    int condval_49;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 4096) < vocab)) {
      condval_49 = passes_40;
    } else {
      condval_49 = 0;
    }
    uint condval_48;
    if ((condval_49 != 0)) {
      condval_48 = (uint)256;
    } else {
      condval_48 = (uint)0;
    }
    bits[0] = (bits[0] | condval_48);
    int passes_41 = vllm_tki_pass(raw[9], floor_1[0]);
    int condval_51;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 4097) < vocab)) {
      condval_51 = passes_41;
    } else {
      condval_51 = 0;
    }
    uint condval_50;
    if ((condval_51 != 0)) {
      condval_50 = (uint)512;
    } else {
      condval_50 = (uint)0;
    }
    bits[0] = (bits[0] | condval_50);
    int passes_42 = vllm_tki_pass(raw[10], floor_1[0]);
    int condval_53;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 4098) < vocab)) {
      condval_53 = passes_42;
    } else {
      condval_53 = 0;
    }
    uint condval_52;
    if ((condval_53 != 0)) {
      condval_52 = (uint)1024;
    } else {
      condval_52 = (uint)0;
    }
    bits[0] = (bits[0] | condval_52);
    int passes_43 = vllm_tki_pass(raw[11], floor_1[0]);
    int condval_55;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 4099) < vocab)) {
      condval_55 = passes_43;
    } else {
      condval_55 = 0;
    }
    uint condval_54;
    if ((condval_55 != 0)) {
      condval_54 = (uint)2048;
    } else {
      condval_54 = (uint)0;
    }
    bits[0] = (bits[0] | condval_54);
    int passes_44 = vllm_tki_pass(raw[12], floor_1[0]);
    int condval_57;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 6144) < vocab)) {
      condval_57 = passes_44;
    } else {
      condval_57 = 0;
    }
    uint condval_56;
    if ((condval_57 != 0)) {
      condval_56 = (uint)4096;
    } else {
      condval_56 = (uint)0;
    }
    bits[0] = (bits[0] | condval_56);
    int passes_45 = vllm_tki_pass(raw[13], floor_1[0]);
    int condval_59;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 6145) < vocab)) {
      condval_59 = passes_45;
    } else {
      condval_59 = 0;
    }
    uint condval_58;
    if ((condval_59 != 0)) {
      condval_58 = (uint)8192;
    } else {
      condval_58 = (uint)0;
    }
    bits[0] = (bits[0] | condval_58);
    int passes_46 = vllm_tki_pass(raw[14], floor_1[0]);
    int condval_61;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 6146) < vocab)) {
      condval_61 = passes_46;
    } else {
      condval_61 = 0;
    }
    uint condval_60;
    if ((condval_61 != 0)) {
      condval_60 = (uint)16384;
    } else {
      condval_60 = (uint)0;
    }
    bits[0] = (bits[0] | condval_60);
    int passes_47 = vllm_tki_pass(raw[15], floor_1[0]);
    int condval_63;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 6147) < vocab)) {
      condval_63 = passes_47;
    } else {
      condval_63 = 0;
    }
    uint condval_62;
    if ((condval_63 != 0)) {
      condval_62 = (uint)32768;
    } else {
      condval_62 = (uint)0;
    }
    bits[0] = (bits[0] | condval_62);
    int passes_48 = vllm_tki_pass(raw[16], floor_1[0]);
    int condval_65;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 8192) < vocab)) {
      condval_65 = passes_48;
    } else {
      condval_65 = 0;
    }
    uint condval_64;
    if ((condval_65 != 0)) {
      condval_64 = (uint)65536;
    } else {
      condval_64 = (uint)0;
    }
    bits[0] = (bits[0] | condval_64);
    int passes_49 = vllm_tki_pass(raw[17], floor_1[0]);
    int condval_67;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 8193) < vocab)) {
      condval_67 = passes_49;
    } else {
      condval_67 = 0;
    }
    uint condval_66;
    if ((condval_67 != 0)) {
      condval_66 = (uint)131072;
    } else {
      condval_66 = (uint)0;
    }
    bits[0] = (bits[0] | condval_66);
    int passes_50 = vllm_tki_pass(raw[18], floor_1[0]);
    int condval_69;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 8194) < vocab)) {
      condval_69 = passes_50;
    } else {
      condval_69 = 0;
    }
    uint condval_68;
    if ((condval_69 != 0)) {
      condval_68 = (uint)262144;
    } else {
      condval_68 = (uint)0;
    }
    bits[0] = (bits[0] | condval_68);
    int passes_51 = vllm_tki_pass(raw[19], floor_1[0]);
    int condval_71;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 8195) < vocab)) {
      condval_71 = passes_51;
    } else {
      condval_71 = 0;
    }
    uint condval_70;
    if ((condval_71 != 0)) {
      condval_70 = (uint)524288;
    } else {
      condval_70 = (uint)0;
    }
    bits[0] = (bits[0] | condval_70);
    int passes_52 = vllm_tki_pass(raw[20], floor_1[0]);
    int condval_73;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 10240) < vocab)) {
      condval_73 = passes_52;
    } else {
      condval_73 = 0;
    }
    uint condval_72;
    if ((condval_73 != 0)) {
      condval_72 = (uint)1048576;
    } else {
      condval_72 = (uint)0;
    }
    bits[0] = (bits[0] | condval_72);
    int passes_53 = vllm_tki_pass(raw[21], floor_1[0]);
    int condval_75;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 10241) < vocab)) {
      condval_75 = passes_53;
    } else {
      condval_75 = 0;
    }
    uint condval_74;
    if ((condval_75 != 0)) {
      condval_74 = (uint)2097152;
    } else {
      condval_74 = (uint)0;
    }
    bits[0] = (bits[0] | condval_74);
    int passes_54 = vllm_tki_pass(raw[22], floor_1[0]);
    int condval_77;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 10242) < vocab)) {
      condval_77 = passes_54;
    } else {
      condval_77 = 0;
    }
    uint condval_76;
    if ((condval_77 != 0)) {
      condval_76 = (uint)4194304;
    } else {
      condval_76 = (uint)0;
    }
    bits[0] = (bits[0] | condval_76);
    int passes_55 = vllm_tki_pass(raw[23], floor_1[0]);
    int condval_79;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 10243) < vocab)) {
      condval_79 = passes_55;
    } else {
      condval_79 = 0;
    }
    uint condval_78;
    if ((condval_79 != 0)) {
      condval_78 = (uint)8388608;
    } else {
      condval_78 = (uint)0;
    }
    bits[0] = (bits[0] | condval_78);
    int passes_56 = vllm_tki_pass(raw[24], floor_1[0]);
    int condval_81;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 12288) < vocab)) {
      condval_81 = passes_56;
    } else {
      condval_81 = 0;
    }
    uint condval_80;
    if ((condval_81 != 0)) {
      condval_80 = (uint)16777216;
    } else {
      condval_80 = (uint)0;
    }
    bits[0] = (bits[0] | condval_80);
    int passes_57 = vllm_tki_pass(raw[25], floor_1[0]);
    int condval_83;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 12289) < vocab)) {
      condval_83 = passes_57;
    } else {
      condval_83 = 0;
    }
    uint condval_82;
    if ((condval_83 != 0)) {
      condval_82 = (uint)33554432;
    } else {
      condval_82 = (uint)0;
    }
    bits[0] = (bits[0] | condval_82);
    int passes_58 = vllm_tki_pass(raw[26], floor_1[0]);
    int condval_85;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 12290) < vocab)) {
      condval_85 = passes_58;
    } else {
      condval_85 = 0;
    }
    uint condval_84;
    if ((condval_85 != 0)) {
      condval_84 = (uint)67108864;
    } else {
      condval_84 = (uint)0;
    }
    bits[0] = (bits[0] | condval_84);
    int passes_59 = vllm_tki_pass(raw[27], floor_1[0]);
    int condval_87;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 12291) < vocab)) {
      condval_87 = passes_59;
    } else {
      condval_87 = 0;
    }
    uint condval_86;
    if ((condval_87 != 0)) {
      condval_86 = (uint)134217728;
    } else {
      condval_86 = (uint)0;
    }
    bits[0] = (bits[0] | condval_86);
    int passes_60 = vllm_tki_pass(raw[28], floor_1[0]);
    int condval_89;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 14336) < vocab)) {
      condval_89 = passes_60;
    } else {
      condval_89 = 0;
    }
    uint condval_88;
    if ((condval_89 != 0)) {
      condval_88 = (uint)268435456;
    } else {
      condval_88 = (uint)0;
    }
    bits[0] = (bits[0] | condval_88);
    int passes_61 = vllm_tki_pass(raw[29], floor_1[0]);
    int condval_91;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 14337) < vocab)) {
      condval_91 = passes_61;
    } else {
      condval_91 = 0;
    }
    uint condval_90;
    if ((condval_91 != 0)) {
      condval_90 = (uint)536870912;
    } else {
      condval_90 = (uint)0;
    }
    bits[0] = (bits[0] | condval_90);
    int passes_62 = vllm_tki_pass(raw[30], floor_1[0]);
    int condval_93;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 14338) < vocab)) {
      condval_93 = passes_62;
    } else {
      condval_93 = 0;
    }
    uint condval_92;
    if ((condval_93 != 0)) {
      condval_92 = (uint)1073741824;
    } else {
      condval_92 = (uint)0;
    }
    bits[0] = (bits[0] | condval_92);
    int passes_63 = vllm_tki_pass(raw[31], floor_1[0]);
    int condval_95;
    if ((((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 14339) < vocab)) {
      condval_95 = passes_63;
    } else {
      condval_95 = 0;
    }
    uint condval_94;
    if ((condval_95 != 0)) {
      condval_94 = (uint)2147483648;
    } else {
      condval_94 = (uint)0;
    }
    bits[0] = (bits[0] | condval_94);
    bool flag_32 = ((bits[0] & (uint)1) != (uint)0);
    bool if_condition_32 = (vllm_tki_ballot(((int)flag_32)) != (uint)0);
    if (if_condition_32) {
      entry[0] = vllm_tki_entry(raw[0], (((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)));
      mask[0] = vllm_tki_ballot(((int)(flag_32 & (thr[0] < entry[0]))));
      if (mask[0] != (uint)0) {
        best[0] = vllm_tki_insert(best[0], entry[0], mask[0]);
      }
      if (mask[0] != (uint)0) {
        thr[0] = vllm_tki_lane(best[0], (width - 2));
      }
      if (mask[0] != (uint)0) {
        floor_1[0] = vllm_tki_floor(thr[0]);
      }
    }
    bool flag_33 = ((bits[0] & (uint)2) != (uint)0);
    bool if_condition_33 = (vllm_tki_ballot(((int)flag_33)) != (uint)0);
    if (if_condition_33) {
      entry[0] = vllm_tki_entry(raw[1], ((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 1));
      mask[0] = vllm_tki_ballot(((int)(flag_33 & (thr[0] < entry[0]))));
      if (mask[0] != (uint)0) {
        best[0] = vllm_tki_insert(best[0], entry[0], mask[0]);
      }
      if (mask[0] != (uint)0) {
        thr[0] = vllm_tki_lane(best[0], (width - 2));
      }
      if (mask[0] != (uint)0) {
        floor_1[0] = vllm_tki_floor(thr[0]);
      }
    }
    bool flag_34 = ((bits[0] & (uint)4) != (uint)0);
    bool if_condition_34 = (vllm_tki_ballot(((int)flag_34)) != (uint)0);
    if (if_condition_34) {
      entry[0] = vllm_tki_entry(raw[2], ((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 2));
      mask[0] = vllm_tki_ballot(((int)(flag_34 & (thr[0] < entry[0]))));
      if (mask[0] != (uint)0) {
        best[0] = vllm_tki_insert(best[0], entry[0], mask[0]);
      }
      if (mask[0] != (uint)0) {
        thr[0] = vllm_tki_lane(best[0], (width - 2));
      }
      if (mask[0] != (uint)0) {
        floor_1[0] = vllm_tki_floor(thr[0]);
      }
    }
    bool flag_35 = ((bits[0] & (uint)8) != (uint)0);
    bool if_condition_35 = (vllm_tki_ballot(((int)flag_35)) != (uint)0);
    if (if_condition_35) {
      entry[0] = vllm_tki_entry(raw[3], ((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 3));
      mask[0] = vllm_tki_ballot(((int)(flag_35 & (thr[0] < entry[0]))));
      if (mask[0] != (uint)0) {
        best[0] = vllm_tki_insert(best[0], entry[0], mask[0]);
      }
      if (mask[0] != (uint)0) {
        thr[0] = vllm_tki_lane(best[0], (width - 2));
      }
      if (mask[0] != (uint)0) {
        floor_1[0] = vllm_tki_floor(thr[0]);
      }
    }
    bool flag_36 = ((bits[0] & (uint)16) != (uint)0);
    bool if_condition_36 = (vllm_tki_ballot(((int)flag_36)) != (uint)0);
    if (if_condition_36) {
      entry[0] = vllm_tki_entry(raw[4], ((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 2048));
      mask[0] = vllm_tki_ballot(((int)(flag_36 & (thr[0] < entry[0]))));
      if (mask[0] != (uint)0) {
        best[0] = vllm_tki_insert(best[0], entry[0], mask[0]);
      }
      if (mask[0] != (uint)0) {
        thr[0] = vllm_tki_lane(best[0], (width - 2));
      }
      if (mask[0] != (uint)0) {
        floor_1[0] = vllm_tki_floor(thr[0]);
      }
    }
    bool flag_37 = ((bits[0] & (uint)32) != (uint)0);
    bool if_condition_37 = (vllm_tki_ballot(((int)flag_37)) != (uint)0);
    if (if_condition_37) {
      entry[0] = vllm_tki_entry(raw[5], ((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 2049));
      mask[0] = vllm_tki_ballot(((int)(flag_37 & (thr[0] < entry[0]))));
      if (mask[0] != (uint)0) {
        best[0] = vllm_tki_insert(best[0], entry[0], mask[0]);
      }
      if (mask[0] != (uint)0) {
        thr[0] = vllm_tki_lane(best[0], (width - 2));
      }
      if (mask[0] != (uint)0) {
        floor_1[0] = vllm_tki_floor(thr[0]);
      }
    }
    bool flag_38 = ((bits[0] & (uint)64) != (uint)0);
    bool if_condition_38 = (vllm_tki_ballot(((int)flag_38)) != (uint)0);
    if (if_condition_38) {
      entry[0] = vllm_tki_entry(raw[6], ((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 2050));
      mask[0] = vllm_tki_ballot(((int)(flag_38 & (thr[0] < entry[0]))));
      if (mask[0] != (uint)0) {
        best[0] = vllm_tki_insert(best[0], entry[0], mask[0]);
      }
      if (mask[0] != (uint)0) {
        thr[0] = vllm_tki_lane(best[0], (width - 2));
      }
      if (mask[0] != (uint)0) {
        floor_1[0] = vllm_tki_floor(thr[0]);
      }
    }
    bool flag_39 = ((bits[0] & (uint)128) != (uint)0);
    bool if_condition_39 = (vllm_tki_ballot(((int)flag_39)) != (uint)0);
    if (if_condition_39) {
      entry[0] = vllm_tki_entry(raw[7], ((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 2051));
      mask[0] = vllm_tki_ballot(((int)(flag_39 & (thr[0] < entry[0]))));
      if (mask[0] != (uint)0) {
        best[0] = vllm_tki_insert(best[0], entry[0], mask[0]);
      }
      if (mask[0] != (uint)0) {
        thr[0] = vllm_tki_lane(best[0], (width - 2));
      }
      if (mask[0] != (uint)0) {
        floor_1[0] = vllm_tki_floor(thr[0]);
      }
    }
    bool flag_40 = ((bits[0] & (uint)256) != (uint)0);
    bool if_condition_40 = (vllm_tki_ballot(((int)flag_40)) != (uint)0);
    if (if_condition_40) {
      entry[0] = vllm_tki_entry(raw[8], ((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 4096));
      mask[0] = vllm_tki_ballot(((int)(flag_40 & (thr[0] < entry[0]))));
      if (mask[0] != (uint)0) {
        best[0] = vllm_tki_insert(best[0], entry[0], mask[0]);
      }
      if (mask[0] != (uint)0) {
        thr[0] = vllm_tki_lane(best[0], (width - 2));
      }
      if (mask[0] != (uint)0) {
        floor_1[0] = vllm_tki_floor(thr[0]);
      }
    }
    bool flag_41 = ((bits[0] & (uint)512) != (uint)0);
    bool if_condition_41 = (vllm_tki_ballot(((int)flag_41)) != (uint)0);
    if (if_condition_41) {
      entry[0] = vllm_tki_entry(raw[9], ((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 4097));
      mask[0] = vllm_tki_ballot(((int)(flag_41 & (thr[0] < entry[0]))));
      if (mask[0] != (uint)0) {
        best[0] = vllm_tki_insert(best[0], entry[0], mask[0]);
      }
      if (mask[0] != (uint)0) {
        thr[0] = vllm_tki_lane(best[0], (width - 2));
      }
      if (mask[0] != (uint)0) {
        floor_1[0] = vllm_tki_floor(thr[0]);
      }
    }
    bool flag_42 = ((bits[0] & (uint)1024) != (uint)0);
    bool if_condition_42 = (vllm_tki_ballot(((int)flag_42)) != (uint)0);
    if (if_condition_42) {
      entry[0] = vllm_tki_entry(raw[10], ((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 4098));
      mask[0] = vllm_tki_ballot(((int)(flag_42 & (thr[0] < entry[0]))));
      if (mask[0] != (uint)0) {
        best[0] = vllm_tki_insert(best[0], entry[0], mask[0]);
      }
      if (mask[0] != (uint)0) {
        thr[0] = vllm_tki_lane(best[0], (width - 2));
      }
      if (mask[0] != (uint)0) {
        floor_1[0] = vllm_tki_floor(thr[0]);
      }
    }
    bool flag_43 = ((bits[0] & (uint)2048) != (uint)0);
    bool if_condition_43 = (vllm_tki_ballot(((int)flag_43)) != (uint)0);
    if (if_condition_43) {
      entry[0] = vllm_tki_entry(raw[11], ((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 4099));
      mask[0] = vllm_tki_ballot(((int)(flag_43 & (thr[0] < entry[0]))));
      if (mask[0] != (uint)0) {
        best[0] = vllm_tki_insert(best[0], entry[0], mask[0]);
      }
      if (mask[0] != (uint)0) {
        thr[0] = vllm_tki_lane(best[0], (width - 2));
      }
      if (mask[0] != (uint)0) {
        floor_1[0] = vllm_tki_floor(thr[0]);
      }
    }
    bool flag_44 = ((bits[0] & (uint)4096) != (uint)0);
    bool if_condition_44 = (vllm_tki_ballot(((int)flag_44)) != (uint)0);
    if (if_condition_44) {
      entry[0] = vllm_tki_entry(raw[12], ((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 6144));
      mask[0] = vllm_tki_ballot(((int)(flag_44 & (thr[0] < entry[0]))));
      if (mask[0] != (uint)0) {
        best[0] = vllm_tki_insert(best[0], entry[0], mask[0]);
      }
      if (mask[0] != (uint)0) {
        thr[0] = vllm_tki_lane(best[0], (width - 2));
      }
      if (mask[0] != (uint)0) {
        floor_1[0] = vllm_tki_floor(thr[0]);
      }
    }
    bool flag_45 = ((bits[0] & (uint)8192) != (uint)0);
    bool if_condition_45 = (vllm_tki_ballot(((int)flag_45)) != (uint)0);
    if (if_condition_45) {
      entry[0] = vllm_tki_entry(raw[13], ((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 6145));
      mask[0] = vllm_tki_ballot(((int)(flag_45 & (thr[0] < entry[0]))));
      if (mask[0] != (uint)0) {
        best[0] = vllm_tki_insert(best[0], entry[0], mask[0]);
      }
      if (mask[0] != (uint)0) {
        thr[0] = vllm_tki_lane(best[0], (width - 2));
      }
      if (mask[0] != (uint)0) {
        floor_1[0] = vllm_tki_floor(thr[0]);
      }
    }
    bool flag_46 = ((bits[0] & (uint)16384) != (uint)0);
    bool if_condition_46 = (vllm_tki_ballot(((int)flag_46)) != (uint)0);
    if (if_condition_46) {
      entry[0] = vllm_tki_entry(raw[14], ((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 6146));
      mask[0] = vllm_tki_ballot(((int)(flag_46 & (thr[0] < entry[0]))));
      if (mask[0] != (uint)0) {
        best[0] = vllm_tki_insert(best[0], entry[0], mask[0]);
      }
      if (mask[0] != (uint)0) {
        thr[0] = vllm_tki_lane(best[0], (width - 2));
      }
      if (mask[0] != (uint)0) {
        floor_1[0] = vllm_tki_floor(thr[0]);
      }
    }
    bool flag_47 = ((bits[0] & (uint)32768) != (uint)0);
    bool if_condition_47 = (vllm_tki_ballot(((int)flag_47)) != (uint)0);
    if (if_condition_47) {
      entry[0] = vllm_tki_entry(raw[15], ((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 6147));
      mask[0] = vllm_tki_ballot(((int)(flag_47 & (thr[0] < entry[0]))));
      if (mask[0] != (uint)0) {
        best[0] = vllm_tki_insert(best[0], entry[0], mask[0]);
      }
      if (mask[0] != (uint)0) {
        thr[0] = vllm_tki_lane(best[0], (width - 2));
      }
      if (mask[0] != (uint)0) {
        floor_1[0] = vllm_tki_floor(thr[0]);
      }
    }
    bool flag_48 = ((bits[0] & (uint)65536) != (uint)0);
    bool if_condition_48 = (vllm_tki_ballot(((int)flag_48)) != (uint)0);
    if (if_condition_48) {
      entry[0] = vllm_tki_entry(raw[16], ((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 8192));
      mask[0] = vllm_tki_ballot(((int)(flag_48 & (thr[0] < entry[0]))));
      if (mask[0] != (uint)0) {
        best[0] = vllm_tki_insert(best[0], entry[0], mask[0]);
      }
      if (mask[0] != (uint)0) {
        thr[0] = vllm_tki_lane(best[0], (width - 2));
      }
      if (mask[0] != (uint)0) {
        floor_1[0] = vllm_tki_floor(thr[0]);
      }
    }
    bool flag_49 = ((bits[0] & (uint)131072) != (uint)0);
    bool if_condition_49 = (vllm_tki_ballot(((int)flag_49)) != (uint)0);
    if (if_condition_49) {
      entry[0] = vllm_tki_entry(raw[17], ((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 8193));
      mask[0] = vllm_tki_ballot(((int)(flag_49 & (thr[0] < entry[0]))));
      if (mask[0] != (uint)0) {
        best[0] = vllm_tki_insert(best[0], entry[0], mask[0]);
      }
      if (mask[0] != (uint)0) {
        thr[0] = vllm_tki_lane(best[0], (width - 2));
      }
      if (mask[0] != (uint)0) {
        floor_1[0] = vllm_tki_floor(thr[0]);
      }
    }
    bool flag_50 = ((bits[0] & (uint)262144) != (uint)0);
    bool if_condition_50 = (vllm_tki_ballot(((int)flag_50)) != (uint)0);
    if (if_condition_50) {
      entry[0] = vllm_tki_entry(raw[18], ((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 8194));
      mask[0] = vllm_tki_ballot(((int)(flag_50 & (thr[0] < entry[0]))));
      if (mask[0] != (uint)0) {
        best[0] = vllm_tki_insert(best[0], entry[0], mask[0]);
      }
      if (mask[0] != (uint)0) {
        thr[0] = vllm_tki_lane(best[0], (width - 2));
      }
      if (mask[0] != (uint)0) {
        floor_1[0] = vllm_tki_floor(thr[0]);
      }
    }
    bool flag_51 = ((bits[0] & (uint)524288) != (uint)0);
    bool if_condition_51 = (vllm_tki_ballot(((int)flag_51)) != (uint)0);
    if (if_condition_51) {
      entry[0] = vllm_tki_entry(raw[19], ((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 8195));
      mask[0] = vllm_tki_ballot(((int)(flag_51 & (thr[0] < entry[0]))));
      if (mask[0] != (uint)0) {
        best[0] = vllm_tki_insert(best[0], entry[0], mask[0]);
      }
      if (mask[0] != (uint)0) {
        thr[0] = vllm_tki_lane(best[0], (width - 2));
      }
      if (mask[0] != (uint)0) {
        floor_1[0] = vllm_tki_floor(thr[0]);
      }
    }
    bool flag_52 = ((bits[0] & (uint)1048576) != (uint)0);
    bool if_condition_52 = (vllm_tki_ballot(((int)flag_52)) != (uint)0);
    if (if_condition_52) {
      entry[0] = vllm_tki_entry(raw[20], ((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 10240));
      mask[0] = vllm_tki_ballot(((int)(flag_52 & (thr[0] < entry[0]))));
      if (mask[0] != (uint)0) {
        best[0] = vllm_tki_insert(best[0], entry[0], mask[0]);
      }
      if (mask[0] != (uint)0) {
        thr[0] = vllm_tki_lane(best[0], (width - 2));
      }
      if (mask[0] != (uint)0) {
        floor_1[0] = vllm_tki_floor(thr[0]);
      }
    }
    bool flag_53 = ((bits[0] & (uint)2097152) != (uint)0);
    bool if_condition_53 = (vllm_tki_ballot(((int)flag_53)) != (uint)0);
    if (if_condition_53) {
      entry[0] = vllm_tki_entry(raw[21], ((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 10241));
      mask[0] = vllm_tki_ballot(((int)(flag_53 & (thr[0] < entry[0]))));
      if (mask[0] != (uint)0) {
        best[0] = vllm_tki_insert(best[0], entry[0], mask[0]);
      }
      if (mask[0] != (uint)0) {
        thr[0] = vllm_tki_lane(best[0], (width - 2));
      }
      if (mask[0] != (uint)0) {
        floor_1[0] = vllm_tki_floor(thr[0]);
      }
    }
    bool flag_54 = ((bits[0] & (uint)4194304) != (uint)0);
    bool if_condition_54 = (vllm_tki_ballot(((int)flag_54)) != (uint)0);
    if (if_condition_54) {
      entry[0] = vllm_tki_entry(raw[22], ((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 10242));
      mask[0] = vllm_tki_ballot(((int)(flag_54 & (thr[0] < entry[0]))));
      if (mask[0] != (uint)0) {
        best[0] = vllm_tki_insert(best[0], entry[0], mask[0]);
      }
      if (mask[0] != (uint)0) {
        thr[0] = vllm_tki_lane(best[0], (width - 2));
      }
      if (mask[0] != (uint)0) {
        floor_1[0] = vllm_tki_floor(thr[0]);
      }
    }
    bool flag_55 = ((bits[0] & (uint)8388608) != (uint)0);
    bool if_condition_55 = (vllm_tki_ballot(((int)flag_55)) != (uint)0);
    if (if_condition_55) {
      entry[0] = vllm_tki_entry(raw[23], ((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 10243));
      mask[0] = vllm_tki_ballot(((int)(flag_55 & (thr[0] < entry[0]))));
      if (mask[0] != (uint)0) {
        best[0] = vllm_tki_insert(best[0], entry[0], mask[0]);
      }
      if (mask[0] != (uint)0) {
        thr[0] = vllm_tki_lane(best[0], (width - 2));
      }
      if (mask[0] != (uint)0) {
        floor_1[0] = vllm_tki_floor(thr[0]);
      }
    }
    bool flag_56 = ((bits[0] & (uint)16777216) != (uint)0);
    bool if_condition_56 = (vllm_tki_ballot(((int)flag_56)) != (uint)0);
    if (if_condition_56) {
      entry[0] = vllm_tki_entry(raw[24], ((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 12288));
      mask[0] = vllm_tki_ballot(((int)(flag_56 & (thr[0] < entry[0]))));
      if (mask[0] != (uint)0) {
        best[0] = vllm_tki_insert(best[0], entry[0], mask[0]);
      }
      if (mask[0] != (uint)0) {
        thr[0] = vllm_tki_lane(best[0], (width - 2));
      }
      if (mask[0] != (uint)0) {
        floor_1[0] = vllm_tki_floor(thr[0]);
      }
    }
    bool flag_57 = ((bits[0] & (uint)33554432) != (uint)0);
    bool if_condition_57 = (vllm_tki_ballot(((int)flag_57)) != (uint)0);
    if (if_condition_57) {
      entry[0] = vllm_tki_entry(raw[25], ((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 12289));
      mask[0] = vllm_tki_ballot(((int)(flag_57 & (thr[0] < entry[0]))));
      if (mask[0] != (uint)0) {
        best[0] = vllm_tki_insert(best[0], entry[0], mask[0]);
      }
      if (mask[0] != (uint)0) {
        thr[0] = vllm_tki_lane(best[0], (width - 2));
      }
      if (mask[0] != (uint)0) {
        floor_1[0] = vllm_tki_floor(thr[0]);
      }
    }
    bool flag_58 = ((bits[0] & (uint)67108864) != (uint)0);
    bool if_condition_58 = (vllm_tki_ballot(((int)flag_58)) != (uint)0);
    if (if_condition_58) {
      entry[0] = vllm_tki_entry(raw[26], ((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 12290));
      mask[0] = vllm_tki_ballot(((int)(flag_58 & (thr[0] < entry[0]))));
      if (mask[0] != (uint)0) {
        best[0] = vllm_tki_insert(best[0], entry[0], mask[0]);
      }
      if (mask[0] != (uint)0) {
        thr[0] = vllm_tki_lane(best[0], (width - 2));
      }
      if (mask[0] != (uint)0) {
        floor_1[0] = vllm_tki_floor(thr[0]);
      }
    }
    bool flag_59 = ((bits[0] & (uint)134217728) != (uint)0);
    bool if_condition_59 = (vllm_tki_ballot(((int)flag_59)) != (uint)0);
    if (if_condition_59) {
      entry[0] = vllm_tki_entry(raw[27], ((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 12291));
      mask[0] = vllm_tki_ballot(((int)(flag_59 & (thr[0] < entry[0]))));
      if (mask[0] != (uint)0) {
        best[0] = vllm_tki_insert(best[0], entry[0], mask[0]);
      }
      if (mask[0] != (uint)0) {
        thr[0] = vllm_tki_lane(best[0], (width - 2));
      }
      if (mask[0] != (uint)0) {
        floor_1[0] = vllm_tki_floor(thr[0]);
      }
    }
    bool flag_60 = ((bits[0] & (uint)268435456) != (uint)0);
    bool if_condition_60 = (vllm_tki_ballot(((int)flag_60)) != (uint)0);
    if (if_condition_60) {
      entry[0] = vllm_tki_entry(raw[28], ((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 14336));
      mask[0] = vllm_tki_ballot(((int)(flag_60 & (thr[0] < entry[0]))));
      if (mask[0] != (uint)0) {
        best[0] = vllm_tki_insert(best[0], entry[0], mask[0]);
      }
      if (mask[0] != (uint)0) {
        thr[0] = vllm_tki_lane(best[0], (width - 2));
      }
      if (mask[0] != (uint)0) {
        floor_1[0] = vllm_tki_floor(thr[0]);
      }
    }
    bool flag_61 = ((bits[0] & (uint)536870912) != (uint)0);
    bool if_condition_61 = (vllm_tki_ballot(((int)flag_61)) != (uint)0);
    if (if_condition_61) {
      entry[0] = vllm_tki_entry(raw[29], ((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 14337));
      mask[0] = vllm_tki_ballot(((int)(flag_61 & (thr[0] < entry[0]))));
      if (mask[0] != (uint)0) {
        best[0] = vllm_tki_insert(best[0], entry[0], mask[0]);
      }
      if (mask[0] != (uint)0) {
        thr[0] = vllm_tki_lane(best[0], (width - 2));
      }
      if (mask[0] != (uint)0) {
        floor_1[0] = vllm_tki_floor(thr[0]);
      }
    }
    bool flag_62 = ((bits[0] & (uint)1073741824) != (uint)0);
    bool if_condition_62 = (vllm_tki_ballot(((int)flag_62)) != (uint)0);
    if (if_condition_62) {
      entry[0] = vllm_tki_entry(raw[30], ((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 14338));
      mask[0] = vllm_tki_ballot(((int)(flag_62 & (thr[0] < entry[0]))));
      if (mask[0] != (uint)0) {
        best[0] = vllm_tki_insert(best[0], entry[0], mask[0]);
      }
      if (mask[0] != (uint)0) {
        thr[0] = vllm_tki_lane(best[0], (width - 2));
      }
      if (mask[0] != (uint)0) {
        floor_1[0] = vllm_tki_floor(thr[0]);
      }
    }
    bool flag_63 = ((bits[0] & (uint)2147483648) != (uint)0);
    bool if_condition_63 = (vllm_tki_ballot(((int)flag_63)) != (uint)0);
    if (if_condition_63) {
      entry[0] = vllm_tki_entry(raw[31], ((((vocab >> 14) * 16384) + (((int)threadIdx.x) * 4)) + 14339));
      mask[0] = vllm_tki_ballot(((int)(flag_63 & (thr[0] < entry[0]))));
      if (mask[0] != (uint)0) {
        best[0] = vllm_tki_insert(best[0], entry[0], mask[0]);
      }
      if (mask[0] != (uint)0) {
        thr[0] = vllm_tki_lane(best[0], (width - 2));
      }
      if (mask[0] != (uint)0) {
        floor_1[0] = vllm_tki_floor(thr[0]);
      }
    }
  }
  ((uint64_t*)lists)[((int)threadIdx.x)] = best[0];
  __syncthreads();
  rank[0] = 0;
  found[0] = 0;
  int condval_96;
  if ((((found[0] + 16) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)15)]))) {
    condval_96 = 16;
  } else {
    condval_96 = 0;
  }
  found[0] = (found[0] + condval_96);
  int condval_97;
  if ((((found[0] + 8) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)7)]))) {
    condval_97 = 8;
  } else {
    condval_97 = 0;
  }
  found[0] = (found[0] + condval_97);
  int condval_98;
  if ((((found[0] + 4) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)3)]))) {
    condval_98 = 4;
  } else {
    condval_98 = 0;
  }
  found[0] = (found[0] + condval_98);
  int condval_99;
  if ((((found[0] + 2) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)1)]))) {
    condval_99 = 2;
  } else {
    condval_99 = 0;
  }
  found[0] = (found[0] + condval_99);
  int condval_100;
  if ((((found[0] + 1) < width) & (best[0] < ((uint64_t*)lists)[found[(int64_t)0]]))) {
    condval_100 = 1;
  } else {
    condval_100 = 0;
  }
  found[0] = (found[0] + condval_100);
  int condval_101;
  if ((((found[0] + 1) < width) & (best[0] < ((uint64_t*)lists)[found[(int64_t)0]]))) {
    condval_101 = 1;
  } else {
    condval_101 = 0;
  }
  found[0] = (found[0] + condval_101);
  rank[0] = (rank[0] + found[0]);
  found[0] = 0;
  int condval_102;
  if ((((found[0] + 16) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)47)]))) {
    condval_102 = 16;
  } else {
    condval_102 = 0;
  }
  found[0] = (found[0] + condval_102);
  int condval_103;
  if ((((found[0] + 8) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)39)]))) {
    condval_103 = 8;
  } else {
    condval_103 = 0;
  }
  found[0] = (found[0] + condval_103);
  int condval_104;
  if ((((found[0] + 4) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)35)]))) {
    condval_104 = 4;
  } else {
    condval_104 = 0;
  }
  found[0] = (found[0] + condval_104);
  int condval_105;
  if ((((found[0] + 2) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)33)]))) {
    condval_105 = 2;
  } else {
    condval_105 = 0;
  }
  found[0] = (found[0] + condval_105);
  int condval_106;
  if ((((found[0] + 1) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)32)]))) {
    condval_106 = 1;
  } else {
    condval_106 = 0;
  }
  found[0] = (found[0] + condval_106);
  int condval_107;
  if ((((found[0] + 1) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)32)]))) {
    condval_107 = 1;
  } else {
    condval_107 = 0;
  }
  found[0] = (found[0] + condval_107);
  rank[0] = (rank[0] + found[0]);
  found[0] = 0;
  int condval_108;
  if ((((found[0] + 16) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)79)]))) {
    condval_108 = 16;
  } else {
    condval_108 = 0;
  }
  found[0] = (found[0] + condval_108);
  int condval_109;
  if ((((found[0] + 8) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)71)]))) {
    condval_109 = 8;
  } else {
    condval_109 = 0;
  }
  found[0] = (found[0] + condval_109);
  int condval_110;
  if ((((found[0] + 4) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)67)]))) {
    condval_110 = 4;
  } else {
    condval_110 = 0;
  }
  found[0] = (found[0] + condval_110);
  int condval_111;
  if ((((found[0] + 2) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)65)]))) {
    condval_111 = 2;
  } else {
    condval_111 = 0;
  }
  found[0] = (found[0] + condval_111);
  int condval_112;
  if ((((found[0] + 1) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)64)]))) {
    condval_112 = 1;
  } else {
    condval_112 = 0;
  }
  found[0] = (found[0] + condval_112);
  int condval_113;
  if ((((found[0] + 1) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)64)]))) {
    condval_113 = 1;
  } else {
    condval_113 = 0;
  }
  found[0] = (found[0] + condval_113);
  rank[0] = (rank[0] + found[0]);
  found[0] = 0;
  int condval_114;
  if ((((found[0] + 16) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)111)]))) {
    condval_114 = 16;
  } else {
    condval_114 = 0;
  }
  found[0] = (found[0] + condval_114);
  int condval_115;
  if ((((found[0] + 8) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)103)]))) {
    condval_115 = 8;
  } else {
    condval_115 = 0;
  }
  found[0] = (found[0] + condval_115);
  int condval_116;
  if ((((found[0] + 4) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)99)]))) {
    condval_116 = 4;
  } else {
    condval_116 = 0;
  }
  found[0] = (found[0] + condval_116);
  int condval_117;
  if ((((found[0] + 2) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)97)]))) {
    condval_117 = 2;
  } else {
    condval_117 = 0;
  }
  found[0] = (found[0] + condval_117);
  int condval_118;
  if ((((found[0] + 1) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)96)]))) {
    condval_118 = 1;
  } else {
    condval_118 = 0;
  }
  found[0] = (found[0] + condval_118);
  int condval_119;
  if ((((found[0] + 1) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)96)]))) {
    condval_119 = 1;
  } else {
    condval_119 = 0;
  }
  found[0] = (found[0] + condval_119);
  rank[0] = (rank[0] + found[0]);
  found[0] = 0;
  int condval_120;
  if ((((found[0] + 16) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)143)]))) {
    condval_120 = 16;
  } else {
    condval_120 = 0;
  }
  found[0] = (found[0] + condval_120);
  int condval_121;
  if ((((found[0] + 8) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)135)]))) {
    condval_121 = 8;
  } else {
    condval_121 = 0;
  }
  found[0] = (found[0] + condval_121);
  int condval_122;
  if ((((found[0] + 4) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)131)]))) {
    condval_122 = 4;
  } else {
    condval_122 = 0;
  }
  found[0] = (found[0] + condval_122);
  int condval_123;
  if ((((found[0] + 2) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)129)]))) {
    condval_123 = 2;
  } else {
    condval_123 = 0;
  }
  found[0] = (found[0] + condval_123);
  int condval_124;
  if ((((found[0] + 1) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)128)]))) {
    condval_124 = 1;
  } else {
    condval_124 = 0;
  }
  found[0] = (found[0] + condval_124);
  int condval_125;
  if ((((found[0] + 1) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)128)]))) {
    condval_125 = 1;
  } else {
    condval_125 = 0;
  }
  found[0] = (found[0] + condval_125);
  rank[0] = (rank[0] + found[0]);
  found[0] = 0;
  int condval_126;
  if ((((found[0] + 16) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)175)]))) {
    condval_126 = 16;
  } else {
    condval_126 = 0;
  }
  found[0] = (found[0] + condval_126);
  int condval_127;
  if ((((found[0] + 8) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)167)]))) {
    condval_127 = 8;
  } else {
    condval_127 = 0;
  }
  found[0] = (found[0] + condval_127);
  int condval_128;
  if ((((found[0] + 4) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)163)]))) {
    condval_128 = 4;
  } else {
    condval_128 = 0;
  }
  found[0] = (found[0] + condval_128);
  int condval_129;
  if ((((found[0] + 2) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)161)]))) {
    condval_129 = 2;
  } else {
    condval_129 = 0;
  }
  found[0] = (found[0] + condval_129);
  int condval_130;
  if ((((found[0] + 1) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)160)]))) {
    condval_130 = 1;
  } else {
    condval_130 = 0;
  }
  found[0] = (found[0] + condval_130);
  int condval_131;
  if ((((found[0] + 1) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)160)]))) {
    condval_131 = 1;
  } else {
    condval_131 = 0;
  }
  found[0] = (found[0] + condval_131);
  rank[0] = (rank[0] + found[0]);
  found[0] = 0;
  int condval_132;
  if ((((found[0] + 16) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)207)]))) {
    condval_132 = 16;
  } else {
    condval_132 = 0;
  }
  found[0] = (found[0] + condval_132);
  int condval_133;
  if ((((found[0] + 8) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)199)]))) {
    condval_133 = 8;
  } else {
    condval_133 = 0;
  }
  found[0] = (found[0] + condval_133);
  int condval_134;
  if ((((found[0] + 4) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)195)]))) {
    condval_134 = 4;
  } else {
    condval_134 = 0;
  }
  found[0] = (found[0] + condval_134);
  int condval_135;
  if ((((found[0] + 2) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)193)]))) {
    condval_135 = 2;
  } else {
    condval_135 = 0;
  }
  found[0] = (found[0] + condval_135);
  int condval_136;
  if ((((found[0] + 1) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)192)]))) {
    condval_136 = 1;
  } else {
    condval_136 = 0;
  }
  found[0] = (found[0] + condval_136);
  int condval_137;
  if ((((found[0] + 1) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)192)]))) {
    condval_137 = 1;
  } else {
    condval_137 = 0;
  }
  found[0] = (found[0] + condval_137);
  rank[0] = (rank[0] + found[0]);
  found[0] = 0;
  int condval_138;
  if ((((found[0] + 16) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)239)]))) {
    condval_138 = 16;
  } else {
    condval_138 = 0;
  }
  found[0] = (found[0] + condval_138);
  int condval_139;
  if ((((found[0] + 8) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)231)]))) {
    condval_139 = 8;
  } else {
    condval_139 = 0;
  }
  found[0] = (found[0] + condval_139);
  int condval_140;
  if ((((found[0] + 4) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)227)]))) {
    condval_140 = 4;
  } else {
    condval_140 = 0;
  }
  found[0] = (found[0] + condval_140);
  int condval_141;
  if ((((found[0] + 2) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)225)]))) {
    condval_141 = 2;
  } else {
    condval_141 = 0;
  }
  found[0] = (found[0] + condval_141);
  int condval_142;
  if ((((found[0] + 1) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)224)]))) {
    condval_142 = 1;
  } else {
    condval_142 = 0;
  }
  found[0] = (found[0] + condval_142);
  int condval_143;
  if ((((found[0] + 1) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)224)]))) {
    condval_143 = 1;
  } else {
    condval_143 = 0;
  }
  found[0] = (found[0] + condval_143);
  rank[0] = (rank[0] + found[0]);
  found[0] = 0;
  int condval_144;
  if ((((found[0] + 16) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)271)]))) {
    condval_144 = 16;
  } else {
    condval_144 = 0;
  }
  found[0] = (found[0] + condval_144);
  int condval_145;
  if ((((found[0] + 8) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)263)]))) {
    condval_145 = 8;
  } else {
    condval_145 = 0;
  }
  found[0] = (found[0] + condval_145);
  int condval_146;
  if ((((found[0] + 4) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)259)]))) {
    condval_146 = 4;
  } else {
    condval_146 = 0;
  }
  found[0] = (found[0] + condval_146);
  int condval_147;
  if ((((found[0] + 2) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)257)]))) {
    condval_147 = 2;
  } else {
    condval_147 = 0;
  }
  found[0] = (found[0] + condval_147);
  int condval_148;
  if ((((found[0] + 1) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)256)]))) {
    condval_148 = 1;
  } else {
    condval_148 = 0;
  }
  found[0] = (found[0] + condval_148);
  int condval_149;
  if ((((found[0] + 1) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)256)]))) {
    condval_149 = 1;
  } else {
    condval_149 = 0;
  }
  found[0] = (found[0] + condval_149);
  rank[0] = (rank[0] + found[0]);
  found[0] = 0;
  int condval_150;
  if ((((found[0] + 16) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)303)]))) {
    condval_150 = 16;
  } else {
    condval_150 = 0;
  }
  found[0] = (found[0] + condval_150);
  int condval_151;
  if ((((found[0] + 8) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)295)]))) {
    condval_151 = 8;
  } else {
    condval_151 = 0;
  }
  found[0] = (found[0] + condval_151);
  int condval_152;
  if ((((found[0] + 4) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)291)]))) {
    condval_152 = 4;
  } else {
    condval_152 = 0;
  }
  found[0] = (found[0] + condval_152);
  int condval_153;
  if ((((found[0] + 2) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)289)]))) {
    condval_153 = 2;
  } else {
    condval_153 = 0;
  }
  found[0] = (found[0] + condval_153);
  int condval_154;
  if ((((found[0] + 1) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)288)]))) {
    condval_154 = 1;
  } else {
    condval_154 = 0;
  }
  found[0] = (found[0] + condval_154);
  int condval_155;
  if ((((found[0] + 1) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)288)]))) {
    condval_155 = 1;
  } else {
    condval_155 = 0;
  }
  found[0] = (found[0] + condval_155);
  rank[0] = (rank[0] + found[0]);
  found[0] = 0;
  int condval_156;
  if ((((found[0] + 16) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)335)]))) {
    condval_156 = 16;
  } else {
    condval_156 = 0;
  }
  found[0] = (found[0] + condval_156);
  int condval_157;
  if ((((found[0] + 8) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)327)]))) {
    condval_157 = 8;
  } else {
    condval_157 = 0;
  }
  found[0] = (found[0] + condval_157);
  int condval_158;
  if ((((found[0] + 4) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)323)]))) {
    condval_158 = 4;
  } else {
    condval_158 = 0;
  }
  found[0] = (found[0] + condval_158);
  int condval_159;
  if ((((found[0] + 2) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)321)]))) {
    condval_159 = 2;
  } else {
    condval_159 = 0;
  }
  found[0] = (found[0] + condval_159);
  int condval_160;
  if ((((found[0] + 1) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)320)]))) {
    condval_160 = 1;
  } else {
    condval_160 = 0;
  }
  found[0] = (found[0] + condval_160);
  int condval_161;
  if ((((found[0] + 1) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)320)]))) {
    condval_161 = 1;
  } else {
    condval_161 = 0;
  }
  found[0] = (found[0] + condval_161);
  rank[0] = (rank[0] + found[0]);
  found[0] = 0;
  int condval_162;
  if ((((found[0] + 16) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)367)]))) {
    condval_162 = 16;
  } else {
    condval_162 = 0;
  }
  found[0] = (found[0] + condval_162);
  int condval_163;
  if ((((found[0] + 8) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)359)]))) {
    condval_163 = 8;
  } else {
    condval_163 = 0;
  }
  found[0] = (found[0] + condval_163);
  int condval_164;
  if ((((found[0] + 4) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)355)]))) {
    condval_164 = 4;
  } else {
    condval_164 = 0;
  }
  found[0] = (found[0] + condval_164);
  int condval_165;
  if ((((found[0] + 2) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)353)]))) {
    condval_165 = 2;
  } else {
    condval_165 = 0;
  }
  found[0] = (found[0] + condval_165);
  int condval_166;
  if ((((found[0] + 1) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)352)]))) {
    condval_166 = 1;
  } else {
    condval_166 = 0;
  }
  found[0] = (found[0] + condval_166);
  int condval_167;
  if ((((found[0] + 1) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)352)]))) {
    condval_167 = 1;
  } else {
    condval_167 = 0;
  }
  found[0] = (found[0] + condval_167);
  rank[0] = (rank[0] + found[0]);
  found[0] = 0;
  int condval_168;
  if ((((found[0] + 16) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)399)]))) {
    condval_168 = 16;
  } else {
    condval_168 = 0;
  }
  found[0] = (found[0] + condval_168);
  int condval_169;
  if ((((found[0] + 8) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)391)]))) {
    condval_169 = 8;
  } else {
    condval_169 = 0;
  }
  found[0] = (found[0] + condval_169);
  int condval_170;
  if ((((found[0] + 4) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)387)]))) {
    condval_170 = 4;
  } else {
    condval_170 = 0;
  }
  found[0] = (found[0] + condval_170);
  int condval_171;
  if ((((found[0] + 2) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)385)]))) {
    condval_171 = 2;
  } else {
    condval_171 = 0;
  }
  found[0] = (found[0] + condval_171);
  int condval_172;
  if ((((found[0] + 1) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)384)]))) {
    condval_172 = 1;
  } else {
    condval_172 = 0;
  }
  found[0] = (found[0] + condval_172);
  int condval_173;
  if ((((found[0] + 1) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)384)]))) {
    condval_173 = 1;
  } else {
    condval_173 = 0;
  }
  found[0] = (found[0] + condval_173);
  rank[0] = (rank[0] + found[0]);
  found[0] = 0;
  int condval_174;
  if ((((found[0] + 16) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)431)]))) {
    condval_174 = 16;
  } else {
    condval_174 = 0;
  }
  found[0] = (found[0] + condval_174);
  int condval_175;
  if ((((found[0] + 8) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)423)]))) {
    condval_175 = 8;
  } else {
    condval_175 = 0;
  }
  found[0] = (found[0] + condval_175);
  int condval_176;
  if ((((found[0] + 4) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)419)]))) {
    condval_176 = 4;
  } else {
    condval_176 = 0;
  }
  found[0] = (found[0] + condval_176);
  int condval_177;
  if ((((found[0] + 2) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)417)]))) {
    condval_177 = 2;
  } else {
    condval_177 = 0;
  }
  found[0] = (found[0] + condval_177);
  int condval_178;
  if ((((found[0] + 1) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)416)]))) {
    condval_178 = 1;
  } else {
    condval_178 = 0;
  }
  found[0] = (found[0] + condval_178);
  int condval_179;
  if ((((found[0] + 1) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)416)]))) {
    condval_179 = 1;
  } else {
    condval_179 = 0;
  }
  found[0] = (found[0] + condval_179);
  rank[0] = (rank[0] + found[0]);
  found[0] = 0;
  int condval_180;
  if ((((found[0] + 16) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)463)]))) {
    condval_180 = 16;
  } else {
    condval_180 = 0;
  }
  found[0] = (found[0] + condval_180);
  int condval_181;
  if ((((found[0] + 8) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)455)]))) {
    condval_181 = 8;
  } else {
    condval_181 = 0;
  }
  found[0] = (found[0] + condval_181);
  int condval_182;
  if ((((found[0] + 4) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)451)]))) {
    condval_182 = 4;
  } else {
    condval_182 = 0;
  }
  found[0] = (found[0] + condval_182);
  int condval_183;
  if ((((found[0] + 2) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)449)]))) {
    condval_183 = 2;
  } else {
    condval_183 = 0;
  }
  found[0] = (found[0] + condval_183);
  int condval_184;
  if ((((found[0] + 1) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)448)]))) {
    condval_184 = 1;
  } else {
    condval_184 = 0;
  }
  found[0] = (found[0] + condval_184);
  int condval_185;
  if ((((found[0] + 1) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)448)]))) {
    condval_185 = 1;
  } else {
    condval_185 = 0;
  }
  found[0] = (found[0] + condval_185);
  rank[0] = (rank[0] + found[0]);
  found[0] = 0;
  int condval_186;
  if ((((found[0] + 16) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)495)]))) {
    condval_186 = 16;
  } else {
    condval_186 = 0;
  }
  found[0] = (found[0] + condval_186);
  int condval_187;
  if ((((found[0] + 8) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)487)]))) {
    condval_187 = 8;
  } else {
    condval_187 = 0;
  }
  found[0] = (found[0] + condval_187);
  int condval_188;
  if ((((found[0] + 4) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)483)]))) {
    condval_188 = 4;
  } else {
    condval_188 = 0;
  }
  found[0] = (found[0] + condval_188);
  int condval_189;
  if ((((found[0] + 2) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)481)]))) {
    condval_189 = 2;
  } else {
    condval_189 = 0;
  }
  found[0] = (found[0] + condval_189);
  int condval_190;
  if ((((found[0] + 1) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)480)]))) {
    condval_190 = 1;
  } else {
    condval_190 = 0;
  }
  found[0] = (found[0] + condval_190);
  int condval_191;
  if ((((found[0] + 1) < width) & (best[0] < ((uint64_t*)lists)[(((int64_t)found[(int64_t)0]) + (int64_t)480)]))) {
    condval_191 = 1;
  } else {
    condval_191 = 0;
  }
  found[0] = (found[0] + condval_191);
  rank[0] = (rank[0] + found[0]);
  __syncthreads();
  if (((((((int)threadIdx.x) & 31) + 1) < width) & (best[0] != (uint64_t)0)) & ((rank[0] + 1) < width)) {
    ((uint64_t*)chosen)[rank[(int64_t)0]] = best[0];
  }
  __syncthreads();
  if ((((int)threadIdx.x) >> 5) == 0) {
    uint64_t condval_192;
    if ((((((int)threadIdx.x) & 31) + 1) < width)) {
      condval_192 = ((uint64_t*)chosen)[(((int)threadIdx.x) & 31)];
    } else {
      condval_192 = (uint64_t)0;
    }
    picked[0] = vllm_tki_order(condval_192, (width - 1));
    if (((((int)threadIdx.x) & 31) + 1) < width) {
      out[(((((int64_t)((int)blockIdx.x)) * ((int64_t)width)) + (((int64_t)((int)threadIdx.x)) & (int64_t)31)) + (int64_t)1)] = ((int64_t)picked[0]);
    }
  }
  if (((int)threadIdx.x) == 0) {
    out[(((int64_t)((int)blockIdx.x)) * ((int64_t)width))] = sampled[((int64_t)((int)blockIdx.x))];
  }
}

