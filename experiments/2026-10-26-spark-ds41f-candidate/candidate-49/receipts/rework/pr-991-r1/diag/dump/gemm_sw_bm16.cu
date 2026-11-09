#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/instruction/mma_block_scale.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/cuda_fp8.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void gemm_mxfp8_rope_mxfp4_kernel(const fp8_e4_t* A, const fp8_e4_t* B, const float* CosSin, const int64_t* Positions, uchar* Q, uchar* S, const uint* SFA, const uint* SFB, int M, int Mp, int max_pos);
extern "C" __global__ void __launch_bounds__(256, 1) gemm_mxfp8_rope_mxfp4_kernel(const fp8_e4_t* A, const fp8_e4_t* B, const float* CosSin, const int64_t* Positions, uchar* Q, uchar* S, const uint* SFA, const uint* SFB, int M, int Mp, int max_pos) {
  extern __shared__ __align__(16) uchar buf_dyn_shmem[];
  void* A_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* C_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* B_shared = ((void*)((char*)buf_dyn_shmem + 4096));
  void* SFA_shared = ((void*)((char*)buf_dyn_shmem + 36864));
  void* SFB_shared = ((void*)((char*)buf_dyn_shmem + 36992));
  float C_local[8];
  float v0 = 0x0p+0f/*0.000000e+00*/;
  float v1 = 0x0p+0f/*0.000000e+00*/;
  int64_t pos = (int64_t)0;
  float cosine = 0x0p+0f/*0.000000e+00*/;
  float sine = 0x0p+0f/*0.000000e+00*/;
  float amax = 0x0p+0f/*0.000000e+00*/;
  uint bits = (uint)0;
  int sf = 0;
  float scale = 0x0p+0f/*0.000000e+00*/;
  float sv0 = 0x0p+0f/*0.000000e+00*/;
  float sv1 = 0x0p+0f/*0.000000e+00*/;
  int c_lo = 0;
  int c_hi = 0;
  cudaGridDependencySynchronize();
  cudaTriggerProgrammaticLaunchCompletion();
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(C_local + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  tl::cp_async_gs_conditional<8>((&(((fp8_e4_t*)A_shared)[((((((((int)threadIdx.x) >> 4) * 128) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + ((((int)threadIdx.x) & 1) * 8))])), (&(A[((min(((((int64_t)((int)blockIdx.y)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4)), (((int64_t)M) - (int64_t)1)) * (int64_t)1280) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8))])), (((((((int)blockIdx.y) * 16) + (((int)threadIdx.x) >> 4)) < Mp) || (M <= Mp)) && (1 <= M)));
  #pragma unroll
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    tl::cp_async_gs<16>((&(((fp8_e4_t*)B_shared)[(((((i_1 * 4096) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16))])), (&(B[(((((int)blockIdx.x) * 163840) + (i_1 * 4096)) + (((int)threadIdx.x) * 16))])));
  }
  if (((int)threadIdx.x) < 16) {
    tl::cp_async_gs_conditional<4>((&(((uint*)SFA_shared)[((int)threadIdx.x)])), (&(SFA[(min(((((int64_t)((int)blockIdx.y)) * (int64_t)16) + ((int64_t)((int)threadIdx.x))), (((int64_t)M) - (int64_t)1)) * (int64_t)10)])), (((((((int)blockIdx.y) * 16) + ((int)threadIdx.x)) < Mp) || (M <= Mp)) && (1 <= M)));
  }
  if (((int)threadIdx.x) < 128) {
    tl::cp_async_gs<4>((&(((uint*)SFB_shared)[((int)threadIdx.x)])), (&(SFB[((((int)blockIdx.x) * 40) + ((((int)threadIdx.x) >> 5) * 10))])));
  }
  tl::cp_async_commit();
  for (int ko = 0; ko < 9; ++ko) {
    __syncthreads();
    tl::cp_async_gs_conditional<8>((&(((fp8_e4_t*)A_shared)[((((((((ko + 1) & 1) * 2048) + ((((int)threadIdx.x) >> 4) * 128)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + ((((int)threadIdx.x) & 1) * 8))])), (&(A[((((min(((((int64_t)((int)blockIdx.y)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4)), (((int64_t)M) - (int64_t)1)) * (int64_t)1280) + (((int64_t)ko) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) + (int64_t)128)])), (((((((int)blockIdx.y) * 16) + (((int)threadIdx.x) >> 4)) < Mp) || (M <= Mp)) && (1 <= M)));
    #pragma unroll
    for (int i_2 = 0; i_2 < 4; ++i_2) {
      tl::cp_async_gs<16>((&(((fp8_e4_t*)B_shared)[((((((((ko + 1) & 1) * 16384) + (i_2 * 4096)) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16))])), (&(B[(((((((int)blockIdx.x) * 163840) + (ko * 16384)) + (i_2 * 4096)) + (((int)threadIdx.x) * 16)) + 16384)])));
    }
    if (((int)threadIdx.x) < 16) {
      tl::cp_async_gs_conditional<4>((&(((uint*)SFA_shared)[((((ko + 1) & 1) * 16) + ((int)threadIdx.x))])), (&(SFA[(((min(((((int64_t)((int)blockIdx.y)) * (int64_t)16) + ((int64_t)((int)threadIdx.x))), (((int64_t)M) - (int64_t)1)) * (int64_t)10) + ((int64_t)ko)) + (int64_t)1)])), (((((((int)blockIdx.y) * 16) + ((int)threadIdx.x)) < Mp) || (M <= Mp)) && (1 <= M)));
    }
    if (((int)threadIdx.x) < 128) {
      tl::cp_async_gs<4>((&(((uint*)SFB_shared)[((((ko + 1) & 1) * 128) + ((int)threadIdx.x))])), (&(SFB[((((((int)blockIdx.x) * 40) + ((((int)threadIdx.x) >> 5) * 10)) + ko) + 1)])));
    }
    tl::cp_async_commit();
    tl::cp_async_wait<1>();
    __syncthreads();
    {
      fp8_e4_t A_local[16];
      fp8_e4_t B_local[16];
      for (int ki = 0; ki < 4; ++ki) {
        tl::ptx_ldmatrix_x4((&(((fp8_e4_t*)A_shared)[((((((ko & 1) * 2048) + ((((int)threadIdx.x) & 15) * 128)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 64)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16))])), (&(A_local[0])));
        tl::ptx_ldmatrix_x4((&(((fp8_e4_t*)B_shared)[(((((((ko & 1) * 16384) + ((((int)threadIdx.x) >> 4) * 1024)) + ((((int)threadIdx.x) & 7) * 128)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 64)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16))])), (&(B_local[0])));
        tl::sm120_mma_sync_blockscaled<tl::SM120MmaBlockScaledKind::kMxf8f6f4, 1, tl::SM120MmaScaleType::kUE8M0, tl::SM120MmaOperandType::kE4M3, tl::SM120MmaOperandType::kE4M3>(reinterpret_cast<float*>(C_local + 0), reinterpret_cast<const uint32_t*>(A_local + 0), reinterpret_cast<const uint32_t*>(B_local + 0), reinterpret_cast<const float*>(C_local + 0), (*reinterpret_cast<const uint32_t*>((&(((uint*)SFA_shared)[((((ko & 1) * 16) + ((((int)threadIdx.x) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))])))), (*reinterpret_cast<const uint32_t*>((&(((uint*)SFB_shared)[((((ko & 1) * 128) + ((((int)threadIdx.x) >> 5) * 16)) + ((((int)threadIdx.x) & 31) >> 2))])))), static_cast<uint16_t>(ki), static_cast<uint16_t>(0), static_cast<uint16_t>(ki), static_cast<uint16_t>(0));
        tl::sm120_mma_sync_blockscaled<tl::SM120MmaBlockScaledKind::kMxf8f6f4, 1, tl::SM120MmaScaleType::kUE8M0, tl::SM120MmaOperandType::kE4M3, tl::SM120MmaOperandType::kE4M3>(reinterpret_cast<float*>(C_local + 4), reinterpret_cast<const uint32_t*>(A_local + 0), reinterpret_cast<const uint32_t*>(B_local + 8), reinterpret_cast<const float*>(C_local + 4), (*reinterpret_cast<const uint32_t*>((&(((uint*)SFA_shared)[((((ko & 1) * 16) + ((((int)threadIdx.x) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))])))), (*reinterpret_cast<const uint32_t*>((&(((uint*)SFB_shared)[(((((ko & 1) * 128) + ((((int)threadIdx.x) >> 5) * 16)) + ((((int)threadIdx.x) & 31) >> 2)) + 8)])))), static_cast<uint16_t>(ki), static_cast<uint16_t>(0), static_cast<uint16_t>(ki), static_cast<uint16_t>(0));
      }
    }
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  {
    fp8_e4_t A_local_1[16];
    fp8_e4_t B_local_1[16];
    for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
      tl::ptx_ldmatrix_x4((&(((fp8_e4_t*)A_shared)[((((((((int)threadIdx.x) & 15) * 128) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 >> 1)) & 1) * 64)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 2048)])), (&(A_local_1[0])));
      tl::ptx_ldmatrix_x4((&(((fp8_e4_t*)B_shared)[(((((((((int)threadIdx.x) >> 4) * 1024) + ((((int)threadIdx.x) & 7) * 128)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 >> 1)) & 1) * 64)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 16384)])), (&(B_local_1[0])));
      tl::sm120_mma_sync_blockscaled<tl::SM120MmaBlockScaledKind::kMxf8f6f4, 1, tl::SM120MmaScaleType::kUE8M0, tl::SM120MmaOperandType::kE4M3, tl::SM120MmaOperandType::kE4M3>(reinterpret_cast<float*>(C_local + 0), reinterpret_cast<const uint32_t*>(A_local_1 + 0), reinterpret_cast<const uint32_t*>(B_local_1 + 0), reinterpret_cast<const float*>(C_local + 0), (*reinterpret_cast<const uint32_t*>((&(((uint*)SFA_shared)[((((((int)threadIdx.x) & 1) * 8) + ((((int)threadIdx.x) & 31) >> 2)) + 16)])))), (*reinterpret_cast<const uint32_t*>((&(((uint*)SFB_shared)[((((((int)threadIdx.x) >> 5) * 16) + ((((int)threadIdx.x) & 31) >> 2)) + 128)])))), static_cast<uint16_t>(ki_1), static_cast<uint16_t>(0), static_cast<uint16_t>(ki_1), static_cast<uint16_t>(0));
      tl::sm120_mma_sync_blockscaled<tl::SM120MmaBlockScaledKind::kMxf8f6f4, 1, tl::SM120MmaScaleType::kUE8M0, tl::SM120MmaOperandType::kE4M3, tl::SM120MmaOperandType::kE4M3>(reinterpret_cast<float*>(C_local + 4), reinterpret_cast<const uint32_t*>(A_local_1 + 0), reinterpret_cast<const uint32_t*>(B_local_1 + 8), reinterpret_cast<const float*>(C_local + 4), (*reinterpret_cast<const uint32_t*>((&(((uint*)SFA_shared)[((((((int)threadIdx.x) & 1) * 8) + ((((int)threadIdx.x) & 31) >> 2)) + 16)])))), (*reinterpret_cast<const uint32_t*>((&(((uint*)SFB_shared)[((((((int)threadIdx.x) >> 5) * 16) + ((((int)threadIdx.x) & 31) >> 2)) + 136)])))), static_cast<uint16_t>(ki_1), static_cast<uint16_t>(0), static_cast<uint16_t>(ki_1), static_cast<uint16_t>(0));
    }
  }
  __syncthreads();
  tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)C_shared)[(((((int)threadIdx.x) & 15) * 128) + ((((int)threadIdx.x) >> 4) * 8))])), __pack_half2(((bfloat16_t)C_local[0]), ((bfloat16_t)C_local[1])), __pack_half2(((bfloat16_t)C_local[2]), ((bfloat16_t)C_local[3])), __pack_half2(((bfloat16_t)C_local[4]), ((bfloat16_t)C_local[5])), __pack_half2(((bfloat16_t)C_local[6]), ((bfloat16_t)C_local[7])));
  __syncthreads();
  for (int it = 0; it < 4; ++it) {
    if ((((((int)blockIdx.y) * 16) + (it * 4)) + (((int)threadIdx.x) >> 6)) < M) {
      v0 = ((float)((bfloat16_t*)C_shared)[((it * 512) + (((int)threadIdx.x) * 2))]);
      v1 = ((float)((bfloat16_t*)C_shared)[(((it * 512) + (((int)threadIdx.x) * 2)) + 1)]);
      if (32 <= (((int)threadIdx.x) & 63)) {
        pos = Positions[(((((int64_t)((int)blockIdx.y)) * (int64_t)16) + (((int64_t)it) * (int64_t)4)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)6))];
        float condval;
        if ((((int64_t)0 <= pos) && (pos < ((int64_t)max_pos)))) {
          condval = CosSin[(((pos * (int64_t)64) + (((int64_t)((int)threadIdx.x)) & (int64_t)63)) - (int64_t)32)];
        } else {
          condval = 0x0p+0f/*0.000000e+00*/;
        }
        cosine = condval;
        float condval_1;
        if ((((int64_t)0 <= pos) && (pos < ((int64_t)max_pos)))) {
          condval_1 = CosSin[((pos * (int64_t)64) + (((int64_t)((int)threadIdx.x)) & (int64_t)63))];
        } else {
          condval_1 = 0x0p+0f/*0.000000e+00*/;
        }
        sine = condval_1;
        bfloat16_t r0 = ((bfloat16_t)__fmaf_rn(v0, cosine, ((v1 * sine) * -0x1p+0f/*-1.000000e+00*/)));
        bfloat16_t r1 = ((bfloat16_t)__fmaf_rn(v0, sine, (v1 * cosine)));
        v0 = ((float)r0);
        v1 = ((float)r1);
      }
      amax = max(0x1.8p-124f/*7.052966e-38*/, max(fabsf(v0), fabsf(v1)));
      amax = max(amax, __shfl_xor_sync((uint)4294967295, amax, 8, 32));
      amax = max(amax, __shfl_xor_sync((uint)4294967295, amax, 4, 32));
      amax = max(amax, __shfl_xor_sync((uint)4294967295, amax, 2, 32));
      amax = max(amax, __shfl_xor_sync((uint)4294967295, amax, 1, 32));
      float v_ = __fmul_rn(amax, 0x1.5555555555555p-3f/*1.666667e-01*/);
      bits = (*(uint *)(&(v_)));
      sf = (((int)((bits >> (uint)23) & (uint)255)) + ((int)((bits & (uint)8388607) != (uint)0)));
      uint v__1 = (uint)((254 - sf) << 23);
      scale = (*(float *)(&(v__1)));
      sv0 = __fmul_rn(v0, scale);
      sv1 = __fmul_rn(v1, scale);
      float _reinterpret_tmp = sv0;
      c_lo = (((((((((int)(0x1p-2f/*2.500000e-01*/ < fabsf(sv0))) + ((int)(0x1.8p-1f/*7.500000e-01*/ <= fabsf(sv0)))) + ((int)(0x1.4p+0f/*1.250000e+00*/ < fabsf(sv0)))) + ((int)(0x1.cp+0f/*1.750000e+00*/ <= fabsf(sv0)))) + ((int)(0x1.4p+1f/*2.500000e+00*/ < fabsf(sv0)))) + ((int)(0x1.cp+1f/*3.500000e+00*/ <= fabsf(sv0)))) + ((int)(0x1.4p+2f/*5.000000e+00*/ < fabsf(sv0)))) | (((int)((*(uint *)(&(_reinterpret_tmp))) >> (uint)31)) << 3));
      float _reinterpret_tmp_1 = sv1;
      c_hi = (((((((((int)(0x1p-2f/*2.500000e-01*/ < fabsf(sv1))) + ((int)(0x1.8p-1f/*7.500000e-01*/ <= fabsf(sv1)))) + ((int)(0x1.4p+0f/*1.250000e+00*/ < fabsf(sv1)))) + ((int)(0x1.cp+0f/*1.750000e+00*/ <= fabsf(sv1)))) + ((int)(0x1.4p+1f/*2.500000e+00*/ < fabsf(sv1)))) + ((int)(0x1.cp+1f/*3.500000e+00*/ <= fabsf(sv1)))) + ((int)(0x1.4p+2f/*5.000000e+00*/ < fabsf(sv1)))) | (((int)((*(uint *)(&(_reinterpret_tmp_1))) >> (uint)31)) << 3));
      Q[(((((((int64_t)((int)blockIdx.y)) * (int64_t)32768) + (((int64_t)it) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)2048)) + (((int64_t)((int)blockIdx.x)) * (int64_t)64)) + (((int64_t)((int)threadIdx.x)) & (int64_t)63))] = ((uchar)(c_lo | (c_hi << 4)));
      if ((((int)threadIdx.x) % 16) == 0) {
        S[(((((((int64_t)((int)blockIdx.y)) * (int64_t)2048) + (((int64_t)it) * (int64_t)512)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)128)) + (((int64_t)((int)blockIdx.x)) * (int64_t)4)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)4))] = ((uchar)sf);
      }
    }
  }
}

