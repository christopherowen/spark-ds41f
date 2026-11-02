trace /c/iket/full-43l-exp/run3b_1pass/trace/iket/pid_0x34/iket.decoded_results.json: 15908 launches, graphLaunches dict of 14

## Timeline: 8289 launches in 655977.8 us; the GPU is busy 199561.3 us (30%) and idle 456416.5 us in 3086 gaps; gaps over 200 us split it into 606 segments

The 15 of 606 segments with the most busy time:

| segment | launches | wall us | busy us | busy | first kernel |
|---:|---:|---:|---:|---:|---|
| 311 | 5 | 4572.2 | 4244.1 | 93% | mhc_collapse_column_kernel |
| 344 | 1 | 4208.5 | 4208.5 | 100% | vocab_packed_projection_kernel |
| 363 | 1222 | 18173.5 | 18027.2 | 99% | mhc_project_streams_kernel |
| 364 | 3 | 4235.4 | 4038.4 | 95% | sink_flags_kernel |
| 371 | 16 | 4862.0 | 4139.1 | 85% | gemm_bf16_partials_kernel |
| 380 | 1257 | 20671.7 | 19648.6 | 95% | finish_argmax_kernel |
| 382 | 2 | 4045.2 | 4027.0 | 100% | ced_copy_rows_kernel |
| 442 | 1228 | 21011.1 | 20629.9 | 98% | metadata_builder_kernel |
| 444 | 2 | 4088.2 | 4076.9 | 100% | ced_copy_rows_kernel |
| 488 | 4 | 4236.8 | 4017.1 | 95% | mhc_collapse_kernel |
| 498 | 1245 | 20988.8 | 20394.6 | 97% | metadata_builder_kernel |
| 500 | 1 | 4022.4 | 4022.4 | 100% | vocab_packed_projection_kernel |
| 540 | 8 | 4279.5 | 4054.1 | 95% | moe_swiglu_cast_kernel_kernel |
| 554 | 1222 | 19808.6 | 19660.0 | 99% | mhc_project_streams_kernel |
| 556 | 2 | 4033.0 | 4028.0 | 100% | ced_copy_rows_kernel |

| idle us | after | before |
|---:|---|---|
| 70654.4 | scatter_rows_kernel | padding_rows_kernel |
| 16638.2 | norm_forward_and_per_token_cast_kernel_kernel | sink_flags_kernel |
| 16584.9 | norm_forward_and_per_token_cast_kernel_kernel | sink_flags_kernel |
| 16469.9 | norm_forward_and_per_token_cast_kernel_kernel | sink_flags_kernel |
| 16382.1 | norm_forward_and_per_token_cast_kernel_kernel | sink_flags_kernel |
| 16174.5 | norm_forward_and_per_token_cast_kernel_kernel | sink_flags_kernel |
| 1266.7 | gemm_mxfp8_kernel | activation_swiglu_clamp_kernel |
| 1236.2 | mhc_finalize_kernel | gemm_bf16_partials_kernel |
| 1235.6 | metadata_decode_kernel | sparse_mla_kernel |
| 1158.8 | vocab_packed_projection_kernel | logits_rows_kernel |

## Engine steps

| step | kind | launches | wall us | busy us | busy | most span |
|---:|---|---:|---:|---:|---:|---|
| 0 | prefill | 1272 | 228018.1 | 43580.5 | 19% | moe_grouped_fp8_fp4_gemm_kernel 12522, gemm_mxfp8_kernel 11261, vocab_packed_projection_kernel 8445 |
| 1 | decode | 1404 | 60900.8 | 29496.0 | 48% | vocab_packed_projection_kernel 8033, l2_prefetch_kernel 7720, gemm_mxfp8_kernel 6732 |
| 2 | decode | 1404 | 143647.3 | 31142.8 | 22% | vocab_packed_projection_kernel 8032, moe_grouped_fp8_fp4_gemm_kernel 7960, l2_prefetch_kernel 7542 |
| 3 | decode | 1404 | 76240.4 | 32244.4 | 42% | l2_prefetch_kernel 10346, moe_grouped_fp8_fp4_gemm_kernel 8518, vocab_packed_projection_kernel 8088 |
| 4 | decode | 1404 | 73756.7 | 31898.2 | 43% | moe_grouped_fp8_fp4_gemm_kernel 8301, vocab_packed_projection_kernel 8035, l2_prefetch_kernel 8007 |
| 5 | decode | 1401 | 72773.2 | 31199.5 | 43% | moe_grouped_fp8_fp4_gemm_kernel 8021, vocab_packed_projection_kernel 7998, l2_prefetch_kernel 7958 |

## Overview over 5 decode steps: 7017 launches of 115 kernel shapes, spans summing to 247311.2 us (49462.2 a step)

| kernel | grid | launches a step | span us a step | share | span us a launch | last warp: wait | load | compute | store | unattributed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| l2_prefetch_kernel | (2, 1, 1) | 119 | 8314.7 | 16.8% | 71.10 | 0% | 99% | 0% | 0% | 1% |
| vocab_packed_projection_kernel | (1, 4040, 1) | 2 | 8037.3 | 16.2% | 4017.57 | 73% | 9% | 9% | 0% | 9% |
| moe_grouped_fp8_fp4_gemm_kernel | (9, 36, 1) | 40 | 4666.6 | 9.4% | 118.29 | 66% | 23% | 17% | 1% | 0% |
| moe_grouped_fp8_fp4_gemm_kernel | (40, 36, 1) | 40 | 2955.6 | 6.0% | 75.15 | 17% | 45% | 14% | 3% | 21% |
| mhc_project_streams_kernel | (40, 1, 1) | 86 | 2244.5 | 4.5% | 23.49 | 37% | 23% | 11% | 2% | 27% |
| gemm_mxfp8_kernel | (28, 1, 1) | 43 | 2204.3 | 4.5% | 52.26 | 83% | 5% | 8% | 0% | 4% |
| attention_front_kernel | (4, 1, 1) | 43 | 2018.8 | 4.1% | 49.44 | 86% | 1% | 1% | 1% | 12% |
| mhc_finalize_kernel | (6, 1, 1) | 80 | 1906.1 | 3.9% | 22.72 | 66% | 5% | 10% | 10% | 9% |
| gemm_fp8w_bf16_partials_kernel | (32, 4, 1) | 43 | 1617.8 | 3.3% | 37.34 | 60% | 17% | 9% | 0% | 14% |
| gemm_bf16_kernel | (2020, 1, 1) | 5 | 1613.6 | 3.3% | 324.22 | 6% | 7% | 0% | 0% | 87% |
| gemm_mxfp8_kernel | (64, 1, 1) | 51 | 1541.0 | 3.1% | 26.56 | 56% | 10% | 9% | 0% | 25% |
| gemm_bf16_partials_kernel | (6, 8, 1) | 40 | 1463.8 | 3.0% | 36.40 | 15% | 5% | 0% | 0% | 79% |
| gemm_mxfp8_kernel | (36, 1, 1) | 43 | 1423.1 | 2.9% | 33.66 | 74% | 11% | 10% | 0% | 5% |
| moe_swiglu_cast_kernel_kernel | (8, 36, 1) | 40 | 1373.2 | 2.8% | 34.99 | 94% | 1% | 1% | 1% | 4% |
| gemm_splitk_reduce_cast_kernel | (16, 2, 1) | 43 | 1266.0 | 2.6% | 29.38 | 94% | 1% | 0% | 0% | 4% |
| sparse_mla_kernel_1 | (16, 6, 1) | 40 | 967.3 | 2.0% | 23.33 | 6% | 1% | 0% | 0% | 94% |
| sparse_mla_kernel | (5, 1, 6) | 38 | 949.4 | 1.9% | 24.02 | 19% | 43% | 20% | 3% | 14% |
| gemm_mxfp8_kernel | (40, 1, 1) | 43 | 928.5 | 1.9% | 17.76 | 44% | 18% | 18% | 0% | 20% |
| gemm_mxfp8_kernel | (40, 1, 1) | 43 | 515.6 | 1.0% | 9.79 | 22% | 39% | 16% | 1% | 21% |
| gemm_mxfp8_kernel | (80, 1, 1) | 1 | 385.0 | 0.8% | 383.36 | 90% | 6% | 4% | 0% | 0% |

| wait | kernel | grid | last-warp us a step | a launch | first statement |
|---|---|---|---:|---:|---|
| wait#2*40 | moe_grouped_fp8_fp4_gemm_kernel | (9, 36, 1) | 1678.1 | 41.95 | `tl::cp_async_wait<2>();` |
| wait.pdl | attention_front_kernel | (4, 1, 1) | 1673.6 | 38.92 | `cudaGridDependencySynchronize();` |
| wait*19 | gemm_mxfp8_kernel | (28, 1, 1) | 1430.4 | 33.26 | `tl::cp_async_wait<1>();` |
| wait.pdl | moe_swiglu_cast_kernel_kernel | (8, 36, 1) | 1258.3 | 31.46 | `cudaGridDependencySynchronize();` |
| wait.pdl | mhc_finalize_kernel | (6, 1, 1) | 1237.0 | 15.46 | `cudaGridDependencySynchronize();` |
| wait.pdl | gemm_splitk_reduce_cast_kernel | (16, 2, 1) | 1189.5 | 27.66 | `cudaGridDependencySynchronize();` |
| wait*18 | gemm_mxfp8_kernel | (36, 1, 1) | 877.9 | 20.42 | `tl::cp_async_wait<2>();` |
| wait.pdl | mhc_project_streams_kernel | (40, 1, 1) | 595.2 | 6.92 | `cudaGridDependencySynchronize();` |
| wait*14 | gemm_fp8w_bf16_partials_kernel | (32, 4, 1) | 461.8 | 10.74 | `tl::cp_async_wait<3>();` |
| wait*118 | gemm_mxfp8_kernel | (80, 1, 1) | 333.0 | 333.04 | `tl::cp_async_wait<2>();` |
| wait.pdl | rope_kernel | (6, 1, 1) | 318.0 | 7.95 | `cudaGridDependencySynchronize();` |
| wait*14 | gemm_mxfp8_kernel | (40, 1, 1) | 225.7 | 5.25 | `tl::cp_async_wait<2>();` |
| wait*8 | gemm_mxfp8_kernel | (64, 1, 1) | 219.0 | 4.29 | `tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| wait.pdl | compress_gather_partial_state_kernel | (16, 1, 1) | 180.8 | 60.27 | `cudaGridDependencySynchronize();` |
| wait | moe_dispatch_kernel | (1, 1, 1) | 147.1 | 3.42 | `cudaGridDependencySynchronize();` |
| wait.sync*4 | mhc_project_streams_kernel | (40, 1, 1) | 140.6 | 1.64 | `__syncthreads();` |
| wait.pdl | gemm_mxfp8_kernel | (64, 1, 1) | 138.0 | 2.71 | `cudaGridDependencySynchronize();` |
| wait*4 | sparse_mla_kernel | (5, 1, 6) | 124.6 | 3.28 | `tl::cp_async_wait<0>();` |
| wait.pdl | activation_swiglu_clamp_kernel | (1, 6, 1) | 112.3 | 2.81 | `cudaGridDependencySynchronize();` |
| wait.mbarrier*10 | gemm_bf16_partials_kernel | (6, 8, 1) | 84.9 | 2.12 | `mbarrier[((kb % 3) + 3)].wait((((kb % 6) / 3) ^ 1));` |

### l2_prefetch_kernel grid (2, 1, 1) block (128, 1, 1)

595 launches, 8 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 71.10 us, warp 71.01 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 70.56 | 99.4% | 69.04 | 99.3% |
| unattributed | 0.45 | 0.6% | 0.48 | 0.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 70.56 | 99.4% | 1.0 | 70.56 | `for (int s = 0; s < (nseg2 >> 1); ++s)` |

### vocab_packed_projection_kernel grid (1, 4040, 1) block (128, 1, 1)

10 launches, 16160 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 4017.57 us, warp 53.33 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 28.16 | 52.8% | 40.50 | 73.2% |
| unattributed | 16.56 | 31.1% | 4.98 | 9.0% |
| compute | 5.02 | 9.4% | 4.83 | 8.7% |
| load | 3.52 | 6.6% | 4.98 | 9.0% |
| store | 0.06 | 0.1% | 0.06 | 0.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait*17 (est.: sampled iterations x17) | 23.66 | 44.4% | 1.0 | 23.66 | `tl::cp_async_wait<6>();` |
| compute.mma*17 (est.: sampled iterations x17) | 4.35 | 8.2% | 1.0 | 4.35 | `{` |
| wait#3 | 1.78 | 3.3% | 1.0 | 1.78 | `tl::cp_async_wait<2>();` |
| load.cp_async#2*17 (est.: sampled iterations x17) [16.0 GB/s/warp] | 1.63 | 3.1% | 1.0 | 1.63 | `for (int i_6 = 0; i_6 < 8; ++i_6) \| for (int i_9 = 0; i_9 < 4; ++i_9)` |
| wait#2 | 1.63 | 3.1% | 1.0 | 1.63 | `tl::cp_async_wait<4>();` |
| load.global | 1.60 | 3.0% | 1.0 | 1.60 | `int start = row_offsets[(((int)blockIdx.y) * 16)]; \| int start = row_offsets[(((int)blockI` |
| wait#4 | 0.54 | 1.0% | 1.0 | 0.54 | `tl::cp_async_wait<0>();` |
| wait.sync*17 (est.: sampled iterations x17) | 0.54 | 1.0% | 1.0 | 0.54 | `__syncthreads();` |
| load.cp_async [13.333 GB/s/warp] | 0.29 | 0.5% | 1.0 | 0.29 | `for (int i = 0; i < 4; ++i) \| for (int i = 0; i < 8; ++i) \| for (int i_1 = 0; i_1 < 4; ++i` |
| compute.mma#2 | 0.22 | 0.4% | 1.0 | 0.22 | `{` |
| compute.mma#3 | 0.22 | 0.4% | 1.0 | 0.22 | `{` |
| compute.mma#4 | 0.22 | 0.4% | 1.0 | 0.22 | `{` |
| store.global | 0.06 | 0.1% | 1.0 | 0.06 | `for (int i_13 = 0; i_13 < 2; ++i_13) \| for (int i_14 = 0; i_14 < 4; ++i_14) \| for (int i_9` |

### moe_grouped_fp8_fp4_gemm_kernel grid (9, 36, 1) block (128, 1, 1)

200 launches, 1296 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 118.29 us, warp 68.99 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 34.14 | 49.5% | 46.51 | 66.4% |
| load | 15.30 | 22.2% | 16.07 | 23.0% |
| compute | 11.52 | 16.7% | 12.02 | 17.2% |
| unattributed | 7.68 | 11.1% | 0.00 | 0.0% |
| store | 0.35 | 0.5% | 0.36 | 0.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait#2*40 (est.: sampled iterations x40) | 30.08 | 43.6% | 1.0 | 30.08 | `tl::cp_async_wait<2>();` |
| compute.mma*40 (est.: sampled iterations x40) | 11.52 | 16.7% | 1.0 | 11.52 | `{` |
| load#2*40 (est.: sampled iterations x40) [36.571 GB/s/warp] | 8.96 | 13.0% | 1.0 | 8.96 | `if (ko < 38)` |
| load [2.207 GB/s/warp] | 3.71 | 5.4% | 1.0 | 3.71 | `if (((int)threadIdx.x) < 32) \| if (((int)threadIdx.x) < 64)` |
| load.global#2*40 (est.: sampled iterations x40) | 2.56 | 3.7% | 1.0 | 2.56 | `if (((expert < 128) && (0 <= expert))) \| if (((expert < 384) && (0 <= expert)))` |
| wait | 1.50 | 2.2% | 1.0 | 1.50 | `__syncthreads();` |
| wait.sync#2*40 (est.: sampled iterations x40) | 1.28 | 1.9% | 1.0 | 1.28 | `__syncthreads();` |
| wait.sync*40 (est.: sampled iterations x40) | 1.28 | 1.9% | 1.0 | 1.28 | `__syncthreads();` |
| store.global | 0.35 | 0.5% | 1.0 | 0.35 | `for (int i_2 = 0; i_2 < 16; ++i_2) \| for (int i_3 = 0; i_3 < 16; ++i_3) \| for (int i_4 = 0` |
| load.global | 0.06 | 0.1% | 1.0 | 0.06 | `int expert = TileExpert[((int64_t)((int)blockIdx.y))];` |

### moe_grouped_fp8_fp4_gemm_kernel grid (40, 36, 1) block (128, 1, 1)

200 launches, 5760 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 75.15 us, warp 11.54 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 4.35 | 37.7% | 2.46 | 21.1% |
| load | 4.34 | 37.6% | 5.21 | 44.7% |
| compute | 1.60 | 13.9% | 1.59 | 13.6% |
| wait | 0.90 | 7.8% | 2.03 | 17.4% |
| store | 0.35 | 3.1% | 0.37 | 3.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load [2.723 GB/s/warp] | 3.01 | 26.1% | 1.0 | 3.01 | `if (((int)threadIdx.x) < 32) \| if (((int)threadIdx.x) < 64)` |
| compute.mma*5 (est.: sampled iterations x5) | 1.60 | 13.9% | 1.0 | 1.60 | `if (ko < 4)` |
| load#2*5 (est.: sampled iterations x5) [42.667 GB/s/warp] | 0.96 | 8.3% | 1.0 | 0.96 | `if (ko < 3)` |
| wait#2*5 (est.: sampled iterations x5) | 0.48 | 4.2% | 1.0 | 0.48 | `tl::cp_async_wait<2>();` |
| store.global | 0.35 | 3.1% | 1.0 | 0.35 | `for (int i_2 = 0; i_2 < 16; ++i_2) \| for (int i_3 = 0; i_3 < 16; ++i_3) \| for (int i_4 = 0` |
| load.global#2*5 (est.: sampled iterations x5) | 0.32 | 2.8% | 1.0 | 0.32 | `if (((expert < 128) && (0 <= expert))) \| if (((expert < 384) && (0 <= expert)))` |
| wait.sync#2*5 (est.: sampled iterations x5) | 0.16 | 1.4% | 1.0 | 0.16 | `__syncthreads();` |
| wait.sync*5 (est.: sampled iterations x5) | 0.16 | 1.4% | 1.0 | 0.16 | `__syncthreads();` |
| wait | 0.10 | 0.8% | 1.0 | 0.10 | `__syncthreads();` |
| load.global | 0.05 | 0.4% | 1.0 | 0.05 | `int expert = TileExpert[((int64_t)((int)blockIdx.y))];` |

### mhc_project_streams_kernel grid (40, 1, 1) block (128, 1, 1)

430 launches, 160 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 23.49 us, warp 19.33 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 12.58 | 65.1% | 6.26 | 26.7% |
| load | 3.14 | 16.2% | 5.29 | 22.6% |
| compute | 2.69 | 13.9% | 2.69 | 11.5% |
| store | 0.48 | 2.5% | 0.50 | 2.1% |
| wait | 0.45 | 2.3% | 8.73 | 37.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2*4 (est.: sampled iterations x4) | 2.94 | 15.2% | 1.0 | 2.94 | `for (int i_1 = 0; i_1 < 6; ++i_1)` |
| compute.mma*4 (est.: sampled iterations x4) | 2.56 | 13.2% | 1.0 | 2.56 | `{` |
| store.global*4 (est.: sampled iterations x4) | 0.26 | 1.3% | 1.0 | 0.26 | `if (((((int)blockIdx.y) * 16) + (((int)threadIdx.x) >> 3)) < M)` |
| wait.sync*4 (est.: sampled iterations x4) | 0.26 | 1.3% | 1.0 | 0.26 | `__syncthreads();` |
| load.global | 0.19 | 1.0% | 1.0 | 0.19 | `for (int k = 0; k < 4; ++k)` |
| wait.sync#2*4 (est.: sampled iterations x4) | 0.13 | 0.7% | 1.0 | 0.13 | `__syncthreads();` |
| store.global#4 | 0.10 | 0.5% | 1.0 | 0.10 | `if (((((int)threadIdx.x) % 8) == 0) && (((((int)blockIdx.y) * 16) + (((int)threadIdx.x) >>` |
| compute.reduce | 0.06 | 0.3% | 1.0 | 0.06 | `red[0] = (red[0] + __shfl_xor_sync((uint)4294967295, red[0], 4, 32));` |
| compute.reduce#2 | 0.06 | 0.3% | 1.0 | 0.06 | `red[0] = (red[0] + __shfl_xor_sync((uint)4294967295, red[0], 4, 32));` |
| store.global#2 | 0.06 | 0.3% | 1.0 | 0.06 | `if (((((int)blockIdx.y) * 16) + (((int)threadIdx.x) >> 3)) < M) \| if (((((int)threadIdx.x)` |
| store.global#3 | 0.06 | 0.3% | 1.0 | 0.06 | `if (((((int)threadIdx.x) % 8) == 0) && (((((int)blockIdx.y) * 16) + (((int)threadIdx.x) >>` |
| wait.pdl | 0.06 | 0.3% | 1.0 | 0.06 | `cudaGridDependencySynchronize();` |
| compute.mma | 0.00 | 0.0% | 0.0 | 0.00 | `{` |
| load.global#2 | 0.00 | 0.0% | 0.0 | 0.00 | `for (int i_1 = 0; i_1 < 6; ++i_1)` |
| store.global | 0.00 | 0.0% | 0.0 | 0.00 | `for (int n = 0; n < 4; ++n) \| if (((((int)blockIdx.y) * 16) + (((int)threadIdx.x) >> 3)) <` |
| wait.sync | 0.00 | 0.0% | 0.0 | 0.00 | `__syncthreads();` |

### gemm_mxfp8_kernel grid (28, 1, 1) block (128, 1, 1)

215 launches, 112 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 52.26 us, warp 43.71 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 19.52 | 44.7% | 36.88 | 83.1% |
| unattributed | 18.98 | 43.4% | 1.95 | 4.4% |
| compute | 3.20 | 7.3% | 3.42 | 7.7% |
| load | 1.95 | 4.5% | 2.09 | 4.7% |
| store | 0.06 | 0.1% | 0.06 | 0.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait*19 (est.: sampled iterations x19) | 16.42 | 37.6% | 1.0 | 16.42 | `tl::cp_async_wait<1>();` |
| compute.mma*19 (est.: sampled iterations x19) | 3.04 | 7.0% | 1.0 | 3.04 | `{` |
| wait.pdl | 1.98 | 4.5% | 1.0 | 1.98 | `cudaGridDependencySynchronize();` |
| load.cp_async#2*19 (est.: sampled iterations x19) [13.333 GB/s/warp] | 1.82 | 4.2% | 1.0 | 1.82 | `for (int i_3 = 0; i_3 < 2; ++i_3) \| for (int i_3 = 0; i_3 < 8; ++i_3)` |
| wait.sync*19 (est.: sampled iterations x19) | 0.61 | 1.4% | 1.0 | 0.61 | `__syncthreads();` |
| wait#2 | 0.51 | 1.2% | 1.0 | 0.51 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| compute.mma#2 | 0.16 | 0.4% | 1.0 | 0.16 | `{` |
| load.cp_async [10.0 GB/s/warp] | 0.13 | 0.3% | 1.0 | 0.13 | `for (int i = 0; i < 2; ++i) \| for (int i_1 = 0; i_1 < 2; ++i_1) \| for (int i_1 = 0; i_1 < ` |
| store.global | 0.06 | 0.1% | 1.0 | 0.06 | `for (int i_10 = 0; i_10 < 16; ++i_10) \| for (int i_10 = 0; i_10 < 8; ++i_10) \| for (int i_` |

### attention_front_kernel grid (4, 1, 1) block (128, 1, 1)

215 launches, 16 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 49.44 us, warp 47.78 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 42.05 | 88.0% | 38.96 | 86.2% |
| unattributed | 4.77 | 10.0% | 5.26 | 11.6% |
| compute | 0.32 | 0.7% | 0.28 | 0.6% |
| load | 0.32 | 0.7% | 0.33 | 0.7% |
| store | 0.32 | 0.7% | 0.36 | 0.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.pdl | 42.02 | 87.9% | 1.0 | 42.02 | `cudaGridDependencySynchronize();` |
| load.global | 0.22 | 0.5% | 1.0 | 0.22 | `for (int vi = 0; vi < 3; ++vi)` |
| store.global#2 | 0.22 | 0.5% | 1.0 | 0.22 | `for (int vi_2 = 0; vi_2 < 3; ++vi_2)` |
| compute.reduce#2*3 (est.: sampled iterations x3) | 0.19 | 0.4% | 1.0 | 0.19 | `for (int i_1 = 0; i_1 < 2; ++i_1)` |
| compute.reduce | 0.13 | 0.3% | 1.0 | 0.13 | `q_warp_reducer[0] = tl::AllReduce<tl::SumOp, 32, 1, 0, tl::NamedBarrier<128>>::run(q_warp_` |
| load.global#2*3 (est.: sampled iterations x3) | 0.10 | 0.2% | 1.0 | 0.10 | `if (((vi_1 * 2) + ((((int)threadIdx.x) & 63) >> 5)) < 5)` |
| store.global*3 (est.: sampled iterations x3) | 0.10 | 0.2% | 1.0 | 0.10 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 6)) < M) && (((vi_1 * 2) + ((((int)` |
| wait.sync | 0.03 | 0.1% | 1.0 | 0.03 | `__syncthreads();` |

### mhc_finalize_kernel grid (6, 1, 1) block (128, 1, 1)

400 launches, 24 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 22.72 us, warp 22.22 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 15.10 | 68.0% | 15.50 | 66.4% |
| compute | 2.27 | 10.2% | 2.28 | 9.8% |
| store | 2.02 | 9.1% | 2.40 | 10.3% |
| unattributed | 1.94 | 8.7% | 1.99 | 8.5% |
| load | 0.90 | 4.0% | 1.17 | 5.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.pdl | 15.07 | 67.8% | 1.0 | 15.07 | `cudaGridDependencySynchronize();` |
| compute.reduce | 2.27 | 10.2% | 1.0 | 2.27 | `row_max[0] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<32>>::run(row_max[0]);` |
| store.global | 1.95 | 8.8% | 1.0 | 1.95 | `if (((int)threadIdx.x) < 4)` |
| load.global | 0.74 | 3.3% | 1.0 | 0.74 | `if (((int)threadIdx.x) < 32)` |
| load.global#2 | 0.16 | 0.7% | 1.0 | 0.16 | `cm[0] = ((mixes_shared[((((int)threadIdx.x) & 15) + 8)] * scale[2]) + base[((((int)threadI` |
| store.global#2 | 0.06 | 0.3% | 1.0 | 0.06 | `if ((((int)threadIdx.x) >> 4) == 0)` |
| wait.sync | 0.03 | 0.1% | 1.0 | 0.03 | `__syncthreads();` |

### gemm_fp8w_bf16_partials_kernel grid (32, 4, 1) block (128, 1, 1)

215 launches, 512 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 37.34 us, warp 21.60 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 11.52 | 53.3% | 3.48 | 13.8% |
| wait | 5.41 | 25.0% | 15.01 | 59.6% |
| load | 2.56 | 11.9% | 4.39 | 17.5% |
| compute | 2.05 | 9.5% | 2.22 | 8.8% |
| store | 0.06 | 0.3% | 0.07 | 0.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait*14 (est.: sampled iterations x14) | 3.14 | 14.5% | 1.0 | 3.14 | `tl::cp_async_wait<3>();` |
| compute.mma*14 (est.: sampled iterations x14) | 1.79 | 8.3% | 1.0 | 1.79 | `{` |
| load.global*14 (est.: sampled iterations x14) | 1.34 | 6.2% | 1.0 | 1.34 | `if (((int)threadIdx.x) < 4)` |
| load.cp_async#2*14 (est.: sampled iterations x14) [16.0 GB/s/warp] | 0.45 | 2.1% | 1.0 | 0.45 | `for (int i_4 = 0; i_4 < 2; ++i_4) \| for (int i_6 = 0; i_6 < 2; ++i_6)` |
| load.cp_async#3*14 (est.: sampled iterations x14) [16.0 GB/s/warp] | 0.45 | 2.1% | 1.0 | 0.45 | `for (int i_10 = 0; i_10 < 4; ++i_10) \| for (int i_10 = 0; i_10 < 6; ++i_10) \| for (int i_1` |
| wait#2*14 (est.: sampled iterations x14) | 0.45 | 2.1% | 1.0 | 0.45 | `tl::cp_async_wait<3>();` |
| wait.sync#2*14 (est.: sampled iterations x14) | 0.45 | 2.1% | 1.0 | 0.45 | `__syncthreads();` |
| wait.sync*14 (est.: sampled iterations x14) | 0.45 | 2.1% | 1.0 | 0.45 | `__syncthreads();` |
| wait.pdl | 0.32 | 1.5% | 1.0 | 0.32 | `cudaGridDependencySynchronize();` |
| wait#4 | 0.29 | 1.3% | 1.0 | 0.29 | `tl::cp_async_wait<1>();` |
| wait#3 | 0.26 | 1.2% | 1.0 | 0.26 | `tl::cp_async_wait<3>();` |
| load.cp_async [10.667 GB/s/warp] | 0.19 | 0.9% | 1.0 | 0.19 | `for (int i_1 = 0; i_1 < 2; ++i_1)` |
| compute.mma#2 | 0.13 | 0.6% | 1.0 | 0.13 | `{` |
| compute.mma#3 | 0.13 | 0.6% | 1.0 | 0.13 | `{` |
| load.global#2 | 0.06 | 0.3% | 1.0 | 0.06 | `if (((int)threadIdx.x) < 4)` |
| load.global#3 | 0.06 | 0.3% | 1.0 | 0.06 | `if (((int)threadIdx.x) < 4)` |
| store.global | 0.06 | 0.3% | 1.0 | 0.06 | `for (int i_13 = 0; i_13 < 8; ++i_13) \| for (int i_19 = 0; i_19 < 16; ++i_19) \| for (int i_` |
| wait.sync#3 | 0.03 | 0.1% | 1.0 | 0.03 | `__syncthreads();` |
| wait.sync#4 | 0.03 | 0.1% | 1.0 | 0.03 | `__syncthreads();` |

### gemm_bf16_kernel grid (2020, 1, 1) block (256, 1, 1)

25 launches, 16160 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 324.22 us, warp 1.09 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.83 | 76.5% | 1.59 | 87.1% |
| load | 0.13 | 11.8% | 0.12 | 6.6% |
| wait | 0.13 | 11.8% | 0.12 | 6.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.tma [384.0 GB/s/warp] | 0.13 | 11.8% | 1.0 | 0.13 | `if (tl::tl_shuffle_elect<128>())` |
| wait.mbarrier | 0.06 | 5.9% | 1.0 | 0.06 | `mbarrier[2].wait(1);` |
| wait.sync | 0.06 | 5.9% | 1.0 | 0.06 | `__syncthreads();` |

### gemm_mxfp8_kernel grid (64, 1, 1) block (128, 1, 1)

255 launches, 256 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 26.56 us, warp 15.71 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 9.15 | 58.2% | 4.59 | 25.1% |
| wait | 3.90 | 24.8% | 10.18 | 55.6% |
| compute | 1.60 | 10.2% | 1.58 | 8.6% |
| load | 0.99 | 6.3% | 1.89 | 10.3% |
| store | 0.06 | 0.4% | 0.07 | 0.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.pdl | 2.62 | 16.7% | 1.0 | 2.62 | `cudaGridDependencySynchronize();` |
| compute.mma*8 (est.: sampled iterations x8) | 1.28 | 8.1% | 1.0 | 1.28 | `{` |
| load.cp_async#2*8 (est.: sampled iterations x8) [13.333 GB/s/warp] | 0.77 | 4.9% | 1.0 | 0.77 | `for (int i_3 = 0; i_3 < 2; ++i_3) \| for (int i_5 = 0; i_5 < 2; ++i_5) \| tl::cp_async_gs_co` |
| wait*8 (est.: sampled iterations x8) | 0.51 | 3.3% | 1.0 | 0.51 | `tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| wait#3 | 0.29 | 1.8% | 1.0 | 0.29 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>();` |
| wait.sync*8 (est.: sampled iterations x8) | 0.26 | 1.6% | 1.0 | 0.26 | `__syncthreads();` |
| load.cp_async [5.714 GB/s/warp] | 0.22 | 1.4% | 1.0 | 0.22 | `for (int i = 0; i < 2; ++i) \| for (int i_1 = 0; i_1 < 2; ++i_1) \| for (int i_1 = 0; i_1 < ` |
| wait#2 | 0.22 | 1.4% | 1.0 | 0.22 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| compute.mma#2 | 0.16 | 1.0% | 1.0 | 0.16 | `{` |
| compute.mma#3 | 0.16 | 1.0% | 1.0 | 0.16 | `{` |
| store.global | 0.06 | 0.4% | 1.0 | 0.06 | `for (int i_10 = 0; i_10 < 16; ++i_10) \| for (int i_10 = 0; i_10 < 8; ++i_10) \| for (int i_` |
| compute.mma#4 | 0.00 | 0.0% | 0.0 | 0.00 | `{` |
| compute.mma*2 (est.: sampled iterations x2) | 0.00 | 0.0% | 0.0 | 0.00 | `{` |
| load.cp_async#2*2 (est.: sampled iterations x2) | 0.00 | 0.0% | 0.0 | 0.00 | `for (int i_7 = 0; i_7 < 2; ++i_7)` |
| wait#4 | 0.00 | 0.0% | 0.0 | 0.00 | `tl::cp_async_wait<0>();` |
| wait*2 (est.: sampled iterations x2) | 0.00 | 0.0% | 0.0 | 0.00 | `tl::cp_async_wait<3>();` |
| wait.sync*2 (est.: sampled iterations x2) | 0.00 | 0.0% | 0.0 | 0.00 | `__syncthreads();` |

### gemm_bf16_partials_kernel grid (6, 8, 1) block (256, 1, 1)

200 launches, 384 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 36.40 us, warp 14.85 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 13.22 | 89.0% | 12.18 | 79.4% |
| load | 0.96 | 6.5% | 0.84 | 5.5% |
| wait | 0.67 | 4.5% | 2.32 | 15.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.tma*10 (est.: sampled iterations x10) [128.0 GB/s/warp] | 0.96 | 6.5% | 1.0 | 0.96 | `if (tl::tl_shuffle_elect<128>())` |
| wait.mbarrier*10 (est.: sampled iterations x10) | 0.64 | 4.3% | 1.0 | 0.64 | `mbarrier[((kb % 3) + 3)].wait((((kb % 6) / 3) ^ 1));` |
| wait.sync | 0.03 | 0.2% | 1.0 | 0.03 | `__syncthreads();` |

### gemm_mxfp8_kernel grid (36, 1, 1) block (128, 1, 1)

215 launches, 144 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 33.66 us, warp 30.40 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 13.76 | 45.3% | 1.58 | 5.2% |
| wait | 11.49 | 37.8% | 22.35 | 73.8% |
| compute | 3.17 | 10.4% | 3.06 | 10.1% |
| load | 1.92 | 6.3% | 3.22 | 10.6% |
| store | 0.06 | 0.2% | 0.05 | 0.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait*18 (est.: sampled iterations x18) | 10.37 | 34.1% | 1.0 | 10.37 | `tl::cp_async_wait<2>();` |
| compute.mma*18 (est.: sampled iterations x18) | 2.88 | 9.5% | 1.0 | 2.88 | `{` |
| load.cp_async#2*18 (est.: sampled iterations x18) [13.333 GB/s/warp] | 1.73 | 5.7% | 1.0 | 1.73 | `for (int i_4 = 0; i_4 < 2; ++i_4) \| for (int i_5 = 0; i_5 < 4; ++i_5)` |
| wait.sync*18 (est.: sampled iterations x18) | 0.58 | 1.9% | 1.0 | 0.58 | `__syncthreads();` |
| wait#3 | 0.38 | 1.3% | 1.0 | 0.38 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>();` |
| load.cp_async [6.667 GB/s/warp] | 0.19 | 0.6% | 1.0 | 0.19 | `for (int i = 0; i < 2; ++i) \| for (int i_1 = 0; i_1 < 2; ++i_1) \| for (int i_1 = 0; i_1 < ` |
| compute.mma#2 | 0.16 | 0.5% | 1.0 | 0.16 | `{` |
| compute.mma#3 | 0.13 | 0.4% | 1.0 | 0.13 | `{` |
| wait#2 | 0.13 | 0.4% | 1.0 | 0.13 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| store.global | 0.06 | 0.2% | 1.0 | 0.06 | `for (int i_10 = 0; i_10 < 16; ++i_10) \| for (int i_10 = 0; i_10 < 8; ++i_10) \| for (int i_` |
| wait.pdl | 0.03 | 0.1% | 1.0 | 0.03 | `cudaGridDependencySynchronize();` |

### moe_swiglu_cast_kernel_kernel grid (8, 36, 1) block (288, 1, 1)

200 launches, 2592 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 34.99 us, warp 34.00 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 31.98 | 94.1% | 31.46 | 93.7% |
| unattributed | 1.18 | 3.5% | 1.35 | 4.0% |
| load | 0.32 | 0.9% | 0.31 | 0.9% |
| compute | 0.26 | 0.8% | 0.24 | 0.7% |
| store | 0.26 | 0.8% | 0.21 | 0.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.pdl | 31.98 | 94.1% | 1.0 | 31.98 | `cudaGridDependencySynchronize();` |
| load.global | 0.32 | 0.9% | 1.0 | 0.32 | `for (int r = 0; r < 4; ++r)` |
| compute.reduce*4 (est.: sampled iterations x4) | 0.26 | 0.8% | 1.0 | 0.26 | `if (0 <= orders[r_2])` |
| store.global*4 (est.: sampled iterations x4) | 0.26 | 0.8% | 1.0 | 0.26 | `if (0 <= orders[r_2])` |

### gemm_splitk_reduce_cast_kernel grid (16, 2, 1) block (128, 1, 1)

215 launches, 128 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 29.38 us, warp 28.86 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 27.49 | 95.2% | 27.66 | 94.1% |
| unattributed | 0.83 | 2.9% | 1.17 | 4.0% |
| load | 0.35 | 1.2% | 0.38 | 1.3% |
| compute | 0.10 | 0.3% | 0.08 | 0.3% |
| store | 0.10 | 0.3% | 0.09 | 0.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.pdl | 27.49 | 95.2% | 1.0 | 27.49 | `cudaGridDependencySynchronize();` |
| load.global | 0.35 | 1.2% | 1.0 | 0.35 | `if ((1 <= M))` |
| compute.reduce | 0.10 | 0.3% | 1.0 | 0.10 | `amax[0] = tl::AllReduce<tl::MaxOp, 8, 1, 0, tl::NamedBarrier<128>>::run(amax[0]);` |
| store.global | 0.10 | 0.3% | 1.0 | 0.10 | `if (((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 5)) < M)` |

### sparse_mla_kernel_1 grid (16, 6, 1) block (128, 1, 1)

200 launches, 96 CTAs, 384 warps; span 23.33 us, warp 22.39 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 22.34 | 99.8% | 22.59 | 93.5% |
| wait | 0.05 | 0.2% | 1.35 | 5.6% |
| load | 0.00 | 0.0% | 0.21 | 0.8% |
| compute | 0.00 | 0.0% | 0.01 | 0.0% |
| store | 0.00 | 0.0% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.pdl#2 | 0.05 | 0.2% | 0.0 | 18.22 | `cudaGridDependencySynchronize();` |
| load.global | 0.00 | 0.0% | 0.0 | 1.89 | `M = __fmul_rn(Sinks[((int)blockIdx.x)], 0x1.7154764ee6c2fp+0f/*1.442695e+00*/);` |
| compute.reduce | 0.00 | 0.0% | 0.0 | 0.10 | `amax[0] = tl::AllReduce<tl::MaxOp, 8, 1, 0, tl::NamedBarrier<128>>::run(amax[0]);` |
| store.global#2 | 0.00 | 0.0% | 0.0 | 0.06 | `*(uint2*)(Output + (((((int64_t)((int)blockIdx.y)) * (int64_t)8192) + (((int64_t)((int)blo` |
| wait.sync | 0.00 | 0.0% | 0.0 | 0.03 | `__syncthreads();` |

### sparse_mla_kernel grid (5, 1, 6) block (128, 1, 1)

190 launches, 120 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 24.02 us, warp 19.20 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 7.31 | 38.1% | 8.63 | 42.9% |
| compute | 4.10 | 21.3% | 4.09 | 20.4% |
| unattributed | 3.97 | 20.7% | 2.87 | 14.3% |
| wait | 3.38 | 17.6% | 3.90 | 19.4% |
| store | 0.45 | 2.3% | 0.61 | 3.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load [0.16 GB/s/warp] | 6.42 | 33.4% | 1.0 | 6.42 | `for (int c = 0; c < 4; ++c) \| if (((int)threadIdx.x) < 128) \| if (((int)threadIdx.x) < 192` |
| compute*4 (est.: sampled iterations x4) | 4.10 | 21.3% | 1.0 | 4.10 | `if (((((int)blockIdx.x) * 2) + (c_1 >> 1)) < 3) \| {` |
| wait*4 (est.: sampled iterations x4) | 3.33 | 17.3% | 1.0 | 3.33 | `tl::cp_async_wait<0>();` |
| load.cp_async*4 (est.: sampled iterations x4) [4.571 GB/s/warp] | 0.90 | 4.7% | 1.0 | 0.90 | `if (((((int)blockIdx.x) * 2) + (c_1 >> 1)) < 3) \| if ((c_1 < 3) & (((((int)blockIdx.x) * 4` |
| store.global | 0.45 | 2.3% | 1.0 | 0.45 | `for (int i_10 = 0; i_10 < 32; ++i_10) \| for (int i_11 = 0; i_11 < 32; ++i_11) \| for (int i` |
| wait.pdl | 0.05 | 0.2% | 1.0 | 0.05 | `cudaGridDependencySynchronize();` |

### gemm_mxfp8_kernel grid (40, 1, 1) block (128, 1, 1)

215 launches, 160 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 17.76 us, warp 11.81 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 6.14 | 52.0% | 3.17 | 20.3% |
| compute | 2.56 | 21.7% | 2.80 | 17.9% |
| load | 1.54 | 13.0% | 2.76 | 17.7% |
| wait | 1.50 | 12.7% | 6.82 | 43.7% |
| store | 0.06 | 0.5% | 0.07 | 0.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*14 (est.: sampled iterations x14) | 2.24 | 19.0% | 1.0 | 2.24 | `{` |
| load.cp_async#2*14 (est.: sampled iterations x14) [13.333 GB/s/warp] | 1.34 | 11.4% | 1.0 | 1.34 | `for (int i_5 = 0; i_5 < 2; ++i_5) \| tl::cp_async_gs_conditional<16>((&(((fp8_e4_t*)A_share` |
| wait*14 (est.: sampled iterations x14) | 0.90 | 7.6% | 1.0 | 0.90 | `tl::cp_async_wait<2>();` |
| wait.sync*14 (est.: sampled iterations x14) | 0.45 | 3.8% | 1.0 | 0.45 | `__syncthreads();` |
| load.cp_async [6.667 GB/s/warp] | 0.19 | 1.6% | 1.0 | 0.19 | `for (int i = 0; i < 2; ++i) \| for (int i_1 = 0; i_1 < 2; ++i_1) \| for (int i_1 = 0; i_1 < ` |
| compute.mma#2 | 0.16 | 1.4% | 1.0 | 0.16 | `{` |
| compute.mma#3 | 0.16 | 1.4% | 1.0 | 0.16 | `{` |
| store.global | 0.06 | 0.5% | 1.0 | 0.06 | `for (int i_10 = 0; i_10 < 16; ++i_10) \| for (int i_10 = 0; i_10 < 8; ++i_10) \| for (int i_` |
| wait#2 | 0.06 | 0.5% | 1.0 | 0.06 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| wait#3 | 0.06 | 0.5% | 1.0 | 0.06 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>();` |
| wait.pdl | 0.03 | 0.3% | 1.0 | 0.03 | `cudaGridDependencySynchronize();` |

### gemm_mxfp8_kernel grid (40, 1, 1) block (64, 1, 1)

215 launches, 80 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 9.79 us, warp 7.94 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 3.81 | 48.0% | 1.93 | 20.9% |
| load | 1.70 | 21.4% | 3.64 | 39.4% |
| compute | 1.44 | 18.1% | 1.52 | 16.4% |
| wait | 0.90 | 11.3% | 2.03 | 22.0% |
| store | 0.10 | 1.2% | 0.12 | 1.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*8 (est.: sampled iterations x8) | 1.28 | 16.1% | 1.0 | 1.28 | `{` |
| load [1.241 GB/s/warp] | 0.93 | 11.7% | 1.0 | 0.93 | `tl::cp_async_gs_conditional<16>((&(((fp8_e4_t*)A_shared)[((((((int)threadIdx.x) >> 2) * 64` |
| load#2*8 (est.: sampled iterations x8) [12.0 GB/s/warp] | 0.77 | 9.7% | 1.0 | 0.77 | `tl::cp_async_gs_conditional<16>((&(((fp8_e4_t*)A_shared)[((((((ko + 1) & 1) * 1024) + ((((` |
| wait*8 (est.: sampled iterations x8) | 0.51 | 6.5% | 1.0 | 0.51 | `tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| wait.sync*8 (est.: sampled iterations x8) | 0.26 | 3.2% | 1.0 | 0.26 | `__syncthreads();` |
| compute.mma#2 | 0.16 | 2.0% | 1.0 | 0.16 | `{` |
| store.global | 0.10 | 1.2% | 1.0 | 0.10 | `for (int i_10 = 0; i_10 < 16; ++i_10) \| for (int i_10 = 0; i_10 < 8; ++i_10) \| for (int i_` |
| wait#2 | 0.10 | 1.2% | 1.0 | 0.10 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| wait.pdl | 0.03 | 0.4% | 1.0 | 0.03 | `cudaGridDependencySynchronize();` |

### gemm_mxfp8_kernel grid (80, 1, 1) block (128, 1, 1)

5 launches, 320 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 383.36 us, warp 375.65 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 273.95 | 72.9% | 339.71 | 90.1% |
| unattributed | 63.94 | 17.0% | 0.00 | 0.0% |
| load | 22.37 | 6.0% | 23.72 | 6.3% |
| compute | 15.33 | 4.1% | 14.57 | 3.9% |
| store | 0.06 | 0.0% | 0.06 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait*118 (est.: sampled iterations x118) | 268.10 | 71.4% | 1.0 | 268.10 | `tl::cp_async_wait<2>();` |
| compute.mma*118 (est.: sampled iterations x118) | 15.10 | 4.0% | 1.0 | 15.10 | `{` |
| load.cp_async [0.086 GB/s/warp] | 14.82 | 3.9% | 1.0 | 14.82 | `for (int i = 0; i < 2; ++i) \| for (int i_1 = 0; i_1 < 2; ++i_1) \| for (int i_1 = 0; i_1 < ` |
| load.cp_async#2*118 (est.: sampled iterations x118) [20.0 GB/s/warp] | 7.55 | 2.0% | 1.0 | 7.55 | `for (int i_5 = 0; i_5 < 2; ++i_5) \| for (int i_5 = 0; i_5 < 4; ++i_5) \| tl::cp_async_gs_co` |
| wait.sync*118 (est.: sampled iterations x118) | 3.78 | 1.0% | 1.0 | 3.78 | `__syncthreads();` |
| wait#2 | 1.34 | 0.4% | 1.0 | 1.34 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| wait#3 | 0.70 | 0.2% | 1.0 | 0.70 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>();` |
| compute.mma#2 | 0.13 | 0.0% | 1.0 | 0.13 | `{` |
| compute.mma#3 | 0.10 | 0.0% | 1.0 | 0.10 | `{` |
| store.global | 0.06 | 0.0% | 1.0 | 0.06 | `for (int i_10 = 0; i_10 < 16; ++i_10) \| for (int i_10 = 0; i_10 < 8; ++i_10) \| for (int i_` |
| wait.pdl | 0.03 | 0.0% | 1.0 | 0.03 | `cudaGridDependencySynchronize();` |

### rope_kernel grid (6, 1, 1) block (128, 1, 1)

200 launches, 24 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 9.06 us, warp 9.01 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 7.78 | 86.3% | 7.95 | 84.7% |
| unattributed | 1.01 | 11.2% | 1.20 | 12.8% |
| load | 0.19 | 2.1% | 0.19 | 2.0% |
| store | 0.03 | 0.4% | 0.04 | 0.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.pdl | 7.78 | 86.3% | 1.0 | 7.78 | `cudaGridDependencySynchronize();` |
| load.global | 0.19 | 2.1% | 1.0 | 0.19 | `if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < tokens)) \| int64_t position ` |
| store.global | 0.03 | 0.4% | 1.0 | 0.03 | `*(uint4*)(x + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + ((((int64_t)((int)thread` |

### moe_dispatch_kernel grid (1, 1, 1) block (512, 1, 1)

215 launches, 16 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 6.14 us, warp 6.14 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 4.67 | 76.0% | 4.56 | 72.3% |
| store | 1.18 | 19.3% | 1.17 | 18.6% |
| unattributed, with loops' unsampled iterations | 0.22 | 3.6% | 0.47 | 7.5% |
| load | 0.06 | 1.0% | 0.10 | 1.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait | 3.39 | 55.2% | 1.0 | 3.39 | `cudaGridDependencySynchronize();` |
| store.global (sampled iterations) | 1.18 | 19.3% | 1.0 | 1.18 | `ChunkCount[((((int64_t)c) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = count; \| Chu` |
| wait.sync#2*9 (est.: sampled iterations x9) | 0.86 | 14.1% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync | 0.42 | 6.8% | 1.0 | 0.42 | `__syncthreads();` |
| load.global (sampled iterations) | 0.06 | 1.0% | 1.0 | 0.06 | `expert = ((int)TopkIdx[((((int64_t)((int)blockIdx.x)) * (int64_t)512) + ((int64_t)((int)th` |

### gemm_bf16_splitk_kernel grid (16, 5, 1) block (128, 1, 1)

20 launches, 320 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 64.69 us, warp 57.71 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 33.90 | 58.7% | 9.80 | 17.2% |
| wait | 13.46 | 23.3% | 34.72 | 61.0% |
| load | 8.69 | 15.1% | 10.41 | 18.3% |
| compute | 1.54 | 2.7% | 1.84 | 3.2% |
| store | 0.13 | 0.2% | 0.14 | 0.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.cp_async [0.411 GB/s/warp] | 7.47 | 12.9% | 1.0 | 7.47 | `for (int i_1 = 0; i_1 < 2; ++i_1) \| tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)A_sha` |
| wait*12 (est.: sampled iterations x12) | 6.53 | 11.3% | 1.0 | 6.53 | `tl::cp_async_wait<3>();` |
| wait#2 | 4.16 | 7.2% | 1.0 | 4.16 | `tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>(); \| tl::cp_async_wait<3>();` |
| compute.mma*12 (est.: sampled iterations x12) | 1.15 | 2.0% | 1.0 | 1.15 | `{` |
| wait#3 | 1.10 | 1.9% | 1.0 | 1.10 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| load.cp_async#2*12 (est.: sampled iterations x12) [16.0 GB/s/warp] | 0.77 | 1.3% | 1.0 | 0.77 | `tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)A_shared)[((((((kb & 3) * 1024) + ((((int` |
| wait.sync*12 (est.: sampled iterations x12) | 0.77 | 1.3% | 1.0 | 0.77 | `__syncthreads();` |
| load.global | 0.45 | 0.8% | 1.0 | 0.45 | `if (((int)threadIdx.x) == 0)` |
| wait#4 | 0.45 | 0.8% | 1.0 | 0.45 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>();` |
| wait#5 | 0.35 | 0.6% | 1.0 | 0.35 | `tl::cp_async_wait<0>();` |
| compute.mma#2 | 0.10 | 0.2% | 1.0 | 0.10 | `{` |
| compute.mma#3 | 0.10 | 0.2% | 1.0 | 0.10 | `{` |
| compute.mma#4 | 0.10 | 0.2% | 1.0 | 0.10 | `{` |
| compute.mma#5 | 0.10 | 0.2% | 1.0 | 0.10 | `{` |
| store.global | 0.06 | 0.1% | 1.0 | 0.06 | `for (int i_10 = 0; i_10 < 4; ++i_10) \| for (int i_16 = 0; i_16 < 4; ++i_16) \| for (int i_1` |
| store.global#3 | 0.06 | 0.1% | 1.0 | 0.06 | `if (((int*)last)[0] == 1)` |
| wait.pdl | 0.03 | 0.1% | 1.0 | 0.03 | `cudaGridDependencySynchronize();` |
| wait.sync#2 | 0.03 | 0.1% | 1.0 | 0.03 | `__syncthreads();` |
| wait.sync#3 | 0.03 | 0.1% | 1.0 | 0.03 | `__syncthreads();` |
| compute.mma*13 (est.: sampled iterations x13) | 0.00 | 0.0% | 0.0 | 0.00 | `{` |
| load.cp_async#2*13 (est.: sampled iterations x13) | 0.00 | 0.0% | 0.0 | 0.00 | `for (int i_10 = 0; i_10 < 2; ++i_10) \| for (int i_8 = 0; i_8 < 2; ++i_8) \| tl::cp_async_gs` |
| wait*13 (est.: sampled iterations x13) | 0.00 | 0.0% | 0.0 | 0.00 | `tl::cp_async_wait<2>();` |
| wait.sync*13 (est.: sampled iterations x13) | 0.00 | 0.0% | 0.0 | 0.00 | `__syncthreads();` |

### router_partials_kernel grid (3, 1, 1) block (64, 1, 1)

215 launches, 6 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 4.99 us, warp 4.99 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 2.24 | 44.9% | 2.44 | 44.1% |
| compute | 1.34 | 26.9% | 1.33 | 24.0% |
| load | 1.22 | 24.4% | 1.53 | 27.6% |
| store | 0.16 | 3.2% | 0.20 | 3.7% |
| wait | 0.03 | 0.6% | 0.03 | 0.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.reduce | 1.34 | 26.9% | 1.0 | 1.34 | `for (int k = 0; k < 3; ++k) \| for (int k = 0; k < 6; ++k)` |
| load.global#4 | 0.61 | 12.2% | 1.0 | 0.61 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < image_len)) \| if ((((((int)bl` |
| load.global | 0.45 | 9.0% | 1.0 | 0.45 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < mask_len))` |
| load.global#2 | 0.10 | 1.9% | 1.0 | 0.10 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < mask_len))` |
| store.global#3 | 0.10 | 1.9% | 1.0 | 0.10 | `if ((((int)threadIdx.x) & 31) < 3) \| if ((((int)threadIdx.x) & 31) < 6)` |
| load.global#3 | 0.06 | 1.3% | 1.0 | 0.06 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < mask_len))` |
| store.global | 0.03 | 0.6% | 1.0 | 0.03 | `if (condval == (bool)0) \| if (condval == (bool)1)` |
| store.global#2 | 0.03 | 0.6% | 1.0 | 0.03 | `if (condval_1 == (bool)0) \| if (condval_1 == (bool)1)` |
| wait.pdl | 0.03 | 0.6% | 1.0 | 0.03 | `cudaGridDependencySynchronize();` |

### compress_gather_partial_state_kernel grid (16, 1, 1) block (128, 1, 1)

15 launches, 64 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 71.07 us, warp 70.34 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 61.28 | 87.1% | 60.27 | 86.5% |
| load | 4.74 | 6.7% | 4.71 | 6.8% |
| unattributed | 3.17 | 4.5% | 1.89 | 2.7% |
| store | 1.15 | 1.6% | 2.80 | 4.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.pdl | 61.28 | 87.1% | 1.0 | 61.28 | `cudaGridDependencySynchronize();` |
| load.global | 4.67 | 6.6% | 1.0 | 4.67 | `int nt = counts[0];` |
| store.global*2 (est.: sampled iterations x2) | 1.09 | 1.5% | 1.0 | 1.09 | `if (half == 0)` |
| load.global#2*2 (est.: sampled iterations x2) | 0.06 | 0.1% | 1.0 | 0.06 | `for (int i = 0; i < 4; ++i)` |
| store.global#2 | 0.06 | 0.1% | 1.0 | 0.06 | `if (((int)threadIdx.x) == 0)` |

### activation_swiglu_clamp_kernel grid (1, 6, 1) block (128, 1, 1)

200 launches, 24 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 2.02 us, warp 2.00 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 0.99 | 49.6% | 2.81 | 72.3% |
| unattributed | 0.88 | 44.0% | 0.97 | 25.1% |
| load | 0.06 | 3.2% | 0.05 | 1.3% |
| store | 0.06 | 3.2% | 0.05 | 1.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.pdl | 0.99 | 49.6% | 1.0 | 0.99 | `cudaGridDependencySynchronize();` |
| load.global | 0.06 | 3.2% | 1.0 | 0.06 | `*(uint4*)(gate + 0) = *(uint4*)(X + ((((int64_t)((int)blockIdx.y)) * (int64_t)1152) + (((i` |
| store.global | 0.06 | 3.2% | 1.0 | 0.06 | `*(uint4*)(Y + ((((int64_t)((int)blockIdx.y)) * (int64_t)576) + (((int64_t)((int)threadIdx.` |

### moe_grouped_fp8_fp4_gemm_kernel grid (9, 15, 1) block (128, 1, 1)

15 launches, 540 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 47.01 us, warp 44.77 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 13.25 | 29.6% | 13.88 | 31.8% |
| unattributed | 13.22 | 29.5% | 0.00 | 0.0% |
| compute | 11.52 | 25.7% | 11.95 | 27.4% |
| wait | 6.43 | 14.4% | 29.22 | 66.9% |
| store | 0.35 | 0.8% | 0.36 | 0.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*40 (est.: sampled iterations x40) | 11.52 | 25.7% | 1.0 | 11.52 | `{` |
| load#2*40 (est.: sampled iterations x40) [36.571 GB/s/warp] | 8.96 | 20.0% | 1.0 | 8.96 | `if (ko < 38)` |
| wait#2*40 (est.: sampled iterations x40) | 3.84 | 8.6% | 1.0 | 3.84 | `tl::cp_async_wait<2>();` |
| load.global#2*40 (est.: sampled iterations x40) | 2.56 | 5.7% | 1.0 | 2.56 | `if (((expert < 128) && (0 <= expert))) \| if (((expert < 384) && (0 <= expert)))` |
| load [4.923 GB/s/warp] | 1.66 | 3.7% | 1.0 | 1.66 | `if (((int)threadIdx.x) < 32) \| if (((int)threadIdx.x) < 64)` |
| wait.sync#2*40 (est.: sampled iterations x40) | 1.28 | 2.9% | 1.0 | 1.28 | `__syncthreads();` |
| wait.sync*40 (est.: sampled iterations x40) | 1.28 | 2.9% | 1.0 | 1.28 | `__syncthreads();` |
| store.global | 0.35 | 0.8% | 1.0 | 0.35 | `for (int i_2 = 0; i_2 < 16; ++i_2) \| for (int i_3 = 0; i_3 < 16; ++i_3) \| for (int i_4 = 0` |
| load.global | 0.06 | 0.1% | 1.0 | 0.06 | `int expert = TileExpert[((int64_t)((int)blockIdx.y))];` |
| wait | 0.03 | 0.1% | 1.0 | 0.03 | `__syncthreads();` |

### compress_kernel grid (2, 1, 1) block (128, 1, 1)

20 launches, 8 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 25.28 us, warp 25.28 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 11.60 | 45.9% | 6.85 | 21.1% |
| load | 6.96 | 27.5% | 6.49 | 20.0% |
| wait | 5.90 | 23.4% | 17.05 | 52.4% |
| store | 0.72 | 2.8% | 2.02 | 6.2% |
| compute | 0.10 | 0.4% | 0.09 | 0.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.pdl | 5.90 | 23.4% | 1.0 | 5.90 | `cudaGridDependencySynchronize();` |
| load.global | 4.21 | 16.6% | 1.0 | 4.21 | `if ((((0 <= counts[0]) & (0 < counts[1])) & (counts[1] <= 16)) & (((((int)blockIdx.x) * 4)` |
| load.global#2*4 (est.: sampled iterations x4) | 2.56 | 10.1% | 1.0 | 2.56 | `*(float4*)(v1 + 0) = *(float4*)(values + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048)` |
| store.global#2*4 (est.: sampled iterations x4) | 0.26 | 1.0% | 1.0 | 0.26 | `if ((slot >> (int64_t)3) < ((int64_t)blocks))` |
| store.global*4 (est.: sampled iterations x4) | 0.26 | 1.0% | 1.0 | 0.26 | `*(uint2*)(out + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + ((((int64_t)((int)thre` |
| load.global#3*4 (est.: sampled iterations x4) | 0.13 | 0.5% | 1.0 | 0.13 | `*(float4*)(w + 0) = *(float4*)(weight + ((c_1 * 128) + ((((int)threadIdx.x) & 31) * 4)));` |
| store.global#3*4 (est.: sampled iterations x4) | 0.13 | 0.5% | 1.0 | 0.13 | `*(uint2*)(out + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + ((((int64_t)((int)thre` |
| compute.reduce | 0.10 | 0.4% | 1.0 | 0.10 | `float total = tl::warp_reduce_sum(ss[0]);` |
| load.global#4 | 0.06 | 0.3% | 1.0 | 0.06 | `if (valid)` |
| store.global#5 | 0.05 | 0.2% | 1.0 | 0.05 | `emitted_slots[((((int64_t)((int)blockIdx.x)) * (int64_t)4) + (((int64_t)((int)threadIdx.x)` |
| store.global#4 | 0.03 | 0.1% | 1.0 | 0.03 | `emitted[((((int64_t)((int)blockIdx.x)) * (int64_t)4) + (((int64_t)((int)threadIdx.x)) >> (` |
| load.global#3 | 0.00 | 0.0% | 0.0 | 0.00 | `if (valid)` |
| store.global#2 | 0.00 | 0.0% | 0.0 | 0.00 | `emitted[((((int64_t)((int)blockIdx.x)) * (int64_t)4) + (((int64_t)((int)threadIdx.x)) >> (` |
| store.global#3 | 0.00 | 0.0% | 0.0 | 0.00 | `emitted_slots[((((int64_t)((int)blockIdx.x)) * (int64_t)4) + (((int64_t)((int)threadIdx.x)` |

### moe_combine_kernel grid (5, 6, 1) block (128, 1, 1)

200 launches, 120 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 2.27 us, warp 2.21 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.57 | 71.0% | 1.56 | 68.0% |
| unattributed | 0.54 | 24.6% | 0.65 | 28.2% |
| store | 0.06 | 2.9% | 0.05 | 2.2% |
| wait | 0.03 | 1.4% | 0.04 | 1.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.57 | 71.0% | 1.0 | 1.57 | `for (int k = 0; k < 3; ++k) \| for (int k = 0; k < 6; ++k) \| if ((((int)blockIdx.y) < share` |
| store.global | 0.06 | 2.9% | 1.0 | 0.06 | `{` |
| wait.pdl | 0.03 | 1.4% | 1.0 | 0.03 | `cudaGridDependencySynchronize();` |

### moe_grouped_fp8_fp4_gemm_kernel grid (40, 15, 1) block (128, 1, 1)

15 launches, 2400 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 29.79 us, warp 11.23 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 4.64 | 41.3% | 5.15 | 42.3% |
| unattributed | 3.81 | 33.9% | 1.73 | 14.2% |
| compute | 1.60 | 14.2% | 1.60 | 13.1% |
| wait | 0.83 | 7.4% | 3.32 | 27.3% |
| store | 0.35 | 3.1% | 0.37 | 3.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load [2.586 GB/s/warp] | 3.17 | 28.2% | 1.0 | 3.17 | `if (((int)threadIdx.x) < 32) \| if (((int)threadIdx.x) < 64)` |
| compute.mma*5 (est.: sampled iterations x5) | 1.60 | 14.2% | 1.0 | 1.60 | `if (ko < 4)` |
| load#2*5 (est.: sampled iterations x5) [36.571 GB/s/warp] | 1.12 | 10.0% | 1.0 | 1.12 | `if (ko < 3)` |
| wait#2*5 (est.: sampled iterations x5) | 0.48 | 4.3% | 1.0 | 0.48 | `tl::cp_async_wait<2>();` |
| store.global | 0.35 | 3.1% | 1.0 | 0.35 | `for (int i_2 = 0; i_2 < 16; ++i_2) \| for (int i_3 = 0; i_3 < 16; ++i_3) \| for (int i_4 = 0` |
| load.global#2*5 (est.: sampled iterations x5) | 0.32 | 2.8% | 1.0 | 0.32 | `if (((expert < 128) && (0 <= expert))) \| if (((expert < 384) && (0 <= expert)))` |
| wait.sync#2*5 (est.: sampled iterations x5) | 0.16 | 1.4% | 1.0 | 0.16 | `__syncthreads();` |
| wait.sync*5 (est.: sampled iterations x5) | 0.16 | 1.4% | 1.0 | 0.16 | `__syncthreads();` |
| load.global | 0.03 | 0.3% | 1.0 | 0.03 | `int expert = TileExpert[((int64_t)((int)blockIdx.y))];` |
| wait | 0.03 | 0.3% | 1.0 | 0.03 | `__syncthreads();` |

### per_token_cast_kernel_kernel grid (9, 1, 1) block (64, 1, 1)

215 launches, 18 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 1.09 us, warp 1.09 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.58 | 52.9% | 0.87 | 58.4% |
| load | 0.26 | 23.5% | 0.31 | 20.5% |
| compute | 0.13 | 11.8% | 0.10 | 7.0% |
| store | 0.10 | 8.8% | 0.16 | 11.0% |
| wait | 0.03 | 2.9% | 0.05 | 3.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.19 | 17.6% | 1.0 | 0.19 | `for (int i = 0; i < 2; ++i)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.13 | 11.8% | 1.0 | 0.13 | `amax_fragment[i_1] = tl::AllReduce<tl::MaxOp, 2, 1, 0, tl::NamedBarrier<64>>::run(amax_fra` |
| store.global#2 | 0.10 | 8.8% | 1.0 | 0.10 | `for (int i_3 = 0; i_3 < 4; ++i_3)` |
| load.global#2*2 (est.: sampled iterations x2) | 0.06 | 5.9% | 1.0 | 0.06 | `amax_fragment[i_1] = cutlass::bfloat16_t(__hmax((bfloat16_t(((nv_bfloat162*)(&(amax_fragme` |
| wait.pdl | 0.03 | 2.9% | 1.0 | 0.03 | `cudaGridDependencySynchronize();` |

### metadata_decode_kernel grid (6, 2, 1) block (128, 1, 1)

75 launches, 48 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 3.58 us, warp 2.72 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.57 | 57.6% | 2.12 | 53.2% |
| unattributed | 0.99 | 36.5% | 0.58 | 14.5% |
| store | 0.13 | 4.7% | 0.14 | 3.6% |
| wait | 0.03 | 1.2% | 1.14 | 28.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.38 | 50.6% | 1.0 | 1.38 | `if ((((offset + ((int)blockIdx.x)) < src_rows) && (0 <= (offset + ((int)blockIdx.x)))))` |
| load.global#2 | 0.19 | 7.1% | 1.0 | 0.19 | `if (((0 <= req) & ((int64_t)0 <= pos)))` |
| store.global | 0.06 | 2.4% | 1.0 | 0.06 | `swa[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = con` |
| store.global#2 | 0.03 | 1.2% | 1.0 | 0.03 | `lengths[((int64_t)((int)blockIdx.x))] = ((int)max(info[1], (int64_t)0));` |
| store.global#3 | 0.03 | 1.2% | 1.0 | 0.03 | `top_lengths[((int64_t)((int)blockIdx.x))] = condval_6; \| top_lengths[((int64_t)((int)block` |
| wait.pdl | 0.03 | 1.2% | 1.0 | 0.03 | `cudaGridDependencySynchronize();` |

### gemm_bf16_partials_kernel grid (2, 20, 1) block (256, 1, 1)

15 launches, 320 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 13.18 us, warp 4.64 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 4.10 | 88.3% | 6.29 | 91.3% |
| wait | 0.29 | 6.2% | 0.29 | 4.1% |
| load | 0.26 | 5.5% | 0.32 | 4.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.tma*4 (est.: sampled iterations x4) [160.0 GB/s/warp] | 0.26 | 5.5% | 1.0 | 0.26 | `if (tl::tl_shuffle_elect<128>())` |
| wait.mbarrier*4 (est.: sampled iterations x4) | 0.26 | 5.5% | 1.0 | 0.26 | `mbarrier[((kb % 3) + 3)].wait(((kb / 3) ^ 1));` |
| wait.sync | 0.03 | 0.7% | 1.0 | 0.03 | `__syncthreads();` |

### indexer_score_kernel grid (128, 6, 1) block (128, 1, 1)

25 launches, 3072 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 6.56 us, warp 6.08 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 3.68 | 60.5% | 6.68 | 67.1% |
| unattributed, with loops' unsampled iterations | 1.92 | 31.6% | 2.40 | 24.1% |
| wait | 0.26 | 4.2% | 0.67 | 6.7% |
| compute | 0.16 | 2.6% | 0.15 | 1.5% |
| store | 0.06 | 1.1% | 0.05 | 0.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 (sampled iterations) | 3.04 | 50.0% | 1.0 | 3.04 | `for (int i_1 = 0; i_1 < 32; ++i_1) \| if ((((i * 8192) + (((int)blockIdx.x) * 64)) + 64) <=` |
| load.global | 0.58 | 9.5% | 1.0 | 0.58 | `int extent = min(max(CandidateLengths[((int64_t)((int)blockIdx.y))], 0), 16384); \| int ext` |
| wait.sync (sampled iterations) | 0.19 | 3.2% | 1.0 | 0.19 | `__syncthreads();` |
| compute.mma (sampled iterations) | 0.16 | 2.6% | 1.0 | 0.16 | `{` |
| load.global#3 (sampled iterations) | 0.06 | 1.1% | 1.0 | 0.06 | `if ((0 <= Candidates[((((((int64_t)((int)blockIdx.y)) * (int64_t)2048) + (((int64_t)i) * (` |
| store.global (sampled iterations) | 0.06 | 1.1% | 1.0 | 0.06 | `Scores[((((((int64_t)((int)blockIdx.y)) * (int64_t)16384) + (((int64_t)i) * (int64_t)8192)` |
| wait.pdl | 0.06 | 1.1% | 1.0 | 0.06 | `cudaGridDependencySynchronize();` |

### rope_kernel grid (6, 1, 1) block (256, 1, 1)

40 launches, 48 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 5.25 us, warp 4.98 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 3.81 | 76.5% | 4.80 | 76.2% |
| unattributed | 0.94 | 19.0% | 1.25 | 19.9% |
| load | 0.19 | 3.9% | 0.20 | 3.2% |
| store | 0.03 | 0.6% | 0.04 | 0.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.pdl | 3.81 | 76.5% | 1.0 | 3.81 | `cudaGridDependencySynchronize();` |
| load.global | 0.19 | 3.9% | 1.0 | 0.19 | `if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < tokens)) \| int64_t position ` |
| store.global | 0.03 | 0.6% | 1.0 | 0.03 | `*(uint4*)(x + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + ((((int64_t)((int)thread` |

### indexer_mxfp4_quantize_kernel grid (6, 1, 1) block (128, 1, 1)

40 launches, 24 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 4.70 us, warp 4.70 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 2.08 | 44.2% | 2.47 | 46.0% |
| store | 2.08 | 44.2% | 1.86 | 34.7% |
| unattributed | 0.51 | 10.9% | 0.88 | 16.3% |
| wait | 0.03 | 0.7% | 0.16 | 3.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global*16 (est.: sampled iterations x16) | 2.05 | 43.5% | 1.0 | 2.05 | `Pool[(((((slot & (int64_t)127) * (int64_t)64) + ((((int64_t)((int)threadIdx.x)) & (int64_t` |
| load.global | 1.06 | 22.4% | 1.0 | 1.06 | `for (int i = 0; i < 32; ++i)` |
| load.global#2*16 (est.: sampled iterations x16) | 1.02 | 21.8% | 1.0 | 1.02 | `float v__2 = __fmul_rn(((float)X[(((((int64_t)((int)blockIdx.x)) * (int64_t)4096) + (((int` |
| store.global#2 | 0.03 | 0.7% | 1.0 | 0.03 | `Pool[(((((slot & (int64_t)127) * (int64_t)4) + ((slot >> (int64_t)7) * ((int64_t)pool_stri` |
| wait.pdl | 0.03 | 0.7% | 1.0 | 0.03 | `cudaGridDependencySynchronize();` |

### mhc_finalize_kernel grid (5, 1, 1) block (128, 1, 1)

30 launches, 20 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 6.88 us, warp 4.93 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| compute | 2.29 | 46.4% | 2.31 | 43.5% |
| load | 1.18 | 24.0% | 1.44 | 27.1% |
| unattributed | 0.93 | 18.8% | 0.99 | 18.6% |
| store | 0.46 | 9.4% | 0.49 | 9.3% |
| wait | 0.06 | 1.3% | 0.08 | 1.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.reduce | 2.29 | 46.4% | 1.0 | 2.29 | `row_max[0] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<32>>::run(row_max[0]);` |
| load.global | 0.80 | 16.2% | 1.0 | 0.80 | `if (((int)threadIdx.x) < 32)` |
| store.global | 0.42 | 8.4% | 1.0 | 0.42 | `if (((int)threadIdx.x) < 4)` |
| load.global#2 | 0.38 | 7.8% | 1.0 | 0.38 | `cm[0] = ((mixes_shared[((((int)threadIdx.x) & 15) + 8)] * scale[2]) + base[((((int)threadI` |
| store.global#2 | 0.05 | 1.0% | 1.0 | 0.05 | `if ((((int)threadIdx.x) >> 4) == 0)` |
| wait.pdl | 0.03 | 0.6% | 1.0 | 0.03 | `cudaGridDependencySynchronize();` |
| wait.sync | 0.03 | 0.6% | 1.0 | 0.03 | `__syncthreads();` |

### indexer_topk_merge_kernel grid (1, 6, 1) block (512, 1, 1)

25 launches, 96 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 8.80 us, warp 8.80 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 6.94 | 78.9% | 7.43 | 84.0% |
| load | 0.99 | 11.3% | 1.02 | 11.5% |
| unattributed | 0.77 | 8.7% | 0.30 | 3.4% |
| store | 0.10 | 1.1% | 0.09 | 1.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.pdl | 3.81 | 43.3% | 1.0 | 3.81 | `cudaGridDependencySynchronize();` |
| load.global | 0.77 | 8.7% | 1.0 | 0.77 | `int extent = min(max(Lengths[((int64_t)((int)blockIdx.y))], 0), 16384); \| int extent = min` |
| wait.sync#6*9 (est.: sampled iterations x9) | 0.58 | 6.5% | 1.0 | 0.58 | `__syncthreads();` |
| wait.sync#8*9 (est.: sampled iterations x9) | 0.58 | 6.5% | 1.0 | 0.58 | `__syncthreads();` |
| wait.sync#2*8 (est.: sampled iterations x8) | 0.51 | 5.8% | 1.0 | 0.51 | `__syncthreads();` |
| wait.sync#4*8 (est.: sampled iterations x8) | 0.51 | 5.8% | 1.0 | 0.51 | `__syncthreads();` |
| wait.sync#3 | 0.35 | 4.0% | 1.0 | 0.35 | `__syncthreads();` |
| wait.sync | 0.26 | 2.9% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#5 | 0.22 | 2.5% | 1.0 | 0.22 | `__syncthreads();` |
| load.global#3 | 0.13 | 1.5% | 1.0 | 0.13 | `if (((0 <= e) && (e < 16384)))` |
| load.global#2 | 0.10 | 1.1% | 1.0 | 0.10 | `if (((0 <= ((int*)sel_idx)[((int)threadIdx.x)]) && (((int*)sel_idx)[((int)threadIdx.x)] < ` |
| wait.sync#7 | 0.10 | 1.1% | 1.0 | 0.10 | `__syncthreads();` |
| store.global | 0.06 | 0.7% | 1.0 | 0.06 | `Values[((((int64_t)((int)blockIdx.y)) * (int64_t)512) + ((int64_t)((int)threadIdx.x)))] = ` |
| store.global#2 | 0.03 | 0.4% | 1.0 | 0.03 | `Indices[((((int64_t)((int)blockIdx.y)) * (int64_t)512) + ((int64_t)((int)threadIdx.x)))] =` |
| wait.sync#9 | 0.03 | 0.4% | 1.0 | 0.03 | `__syncthreads();` |

### gemm_mxfp8_partials_kernel grid (16, 5, 1) block (128, 1, 1)

15 launches, 320 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 12.16 us, warp 11.55 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 5.98 | 51.8% | 5.73 | 47.3% |
| unattributed | 4.45 | 38.5% | 3.67 | 30.3% |
| compute | 0.61 | 5.3% | 0.62 | 5.1% |
| load | 0.45 | 3.9% | 2.03 | 16.8% |
| store | 0.06 | 0.6% | 0.05 | 0.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait*2 (est.: sampled iterations x2) | 2.69 | 23.3% | 1.0 | 2.69 | `tl::cp_async_wait<2>();` |
| wait#2 | 1.63 | 14.1% | 1.0 | 1.63 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>();` |
| wait#3 | 1.57 | 13.6% | 1.0 | 1.57 | `tl::cp_async_wait<0>();` |
| compute.mma*2 (est.: sampled iterations x2) | 0.32 | 2.8% | 1.0 | 0.32 | `{` |
| load.cp_async [10.0 GB/s/warp] | 0.26 | 2.2% | 1.0 | 0.26 | `for (int i = 0; i < 2; ++i) \| for (int i_1 = 0; i_1 < 4; ++i_1)` |
| load.cp_async#2*2 (est.: sampled iterations x2) [13.333 GB/s/warp] | 0.19 | 1.7% | 1.0 | 0.19 | `for (int i_4 = 0; i_4 < 2; ++i_4)` |
| compute.mma#2 | 0.16 | 1.4% | 1.0 | 0.16 | `{` |
| compute.mma#3 | 0.13 | 1.1% | 1.0 | 0.13 | `{` |
| store.global | 0.06 | 0.6% | 1.0 | 0.06 | `for (int i_11 = 0; i_11 < 16; ++i_11) \| for (int i_13 = 0; i_13 < 8; ++i_13) \| for (int i_` |
| wait.sync*2 (est.: sampled iterations x2) | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| wait.pdl | 0.03 | 0.3% | 1.0 | 0.03 | `cudaGridDependencySynchronize();` |

### sparse_mla_kernel grid (1, 1, 6) block (128, 1, 1)

10 launches, 24 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 18.24 us, warp 18.24 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 6.30 | 34.6% | 6.70 | 34.9% |
| compute | 4.10 | 22.5% | 4.08 | 21.3% |
| wait | 3.79 | 20.8% | 3.81 | 19.8% |
| unattributed | 3.38 | 18.5% | 3.74 | 19.5% |
| store | 0.67 | 3.7% | 0.88 | 4.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load [0.181 GB/s/warp] | 5.66 | 31.1% | 1.0 | 5.66 | `for (int c = 0; c < 4; ++c) \| if (((int)threadIdx.x) < 128) \| if (((int)threadIdx.x) < 192` |
| compute*4 (est.: sampled iterations x4) | 4.10 | 22.5% | 1.0 | 4.10 | `if (((((int)blockIdx.x) * 2) + (c_1 >> 1)) < 3) \| {` |
| wait*4 (est.: sampled iterations x4) | 3.46 | 18.9% | 1.0 | 3.46 | `tl::cp_async_wait<0>();` |
| store.global | 0.67 | 3.7% | 1.0 | 0.67 | `for (int i_10 = 0; i_10 < 32; ++i_10) \| for (int i_11 = 0; i_11 < 32; ++i_11) \| for (int i` |
| load.cp_async*4 (est.: sampled iterations x4) [6.4 GB/s/warp] | 0.64 | 3.5% | 1.0 | 0.64 | `if (((((int)blockIdx.x) * 2) + (c_1 >> 1)) < 3) \| if ((c_1 < 3) & (((((int)blockIdx.x) * 4` |
| wait.pdl | 0.34 | 1.8% | 1.0 | 0.34 | `cudaGridDependencySynchronize();` |

### gemm_bf16_partials_kernel grid (1, 40, 1) block (256, 1, 1)

40 launches, 320 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 2.72 us, warp 0.83 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.48 | 57.7% | 0.98 | 54.8% |
| load | 0.19 | 23.1% | 0.26 | 14.5% |
| wait | 0.16 | 19.2% | 0.55 | 30.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.tma*2 (est.: sampled iterations x2) [170.667 GB/s/warp] | 0.19 | 23.1% | 1.0 | 0.19 | `if (tl::tl_shuffle_elect<128>())` |
| wait.mbarrier*2 (est.: sampled iterations x2) | 0.13 | 15.4% | 1.0 | 0.13 | `mbarrier[(kb + 3)].wait(1);` |
| wait.sync | 0.03 | 3.8% | 1.0 | 0.03 | `__syncthreads();` |

### sparse_mla_kernel grid (2, 1, 5) block (128, 1, 1)

15 launches, 40 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 11.78 us, warp 11.78 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| compute | 4.35 | 37.0% | 4.39 | 37.3% |
| wait | 3.87 | 32.9% | 3.82 | 32.5% |
| load | 2.05 | 17.4% | 1.99 | 16.9% |
| unattributed | 0.99 | 8.4% | 1.05 | 8.9% |
| store | 0.51 | 4.3% | 0.52 | 4.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute*4 (est.: sampled iterations x4) | 4.35 | 37.0% | 1.0 | 4.35 | `if (((((int)blockIdx.x) * 2) + (c_1 >> 1)) < 3) \| {` |
| wait*4 (est.: sampled iterations x4) | 3.33 | 28.3% | 1.0 | 3.33 | `tl::cp_async_wait<0>();` |
| load [0.727 GB/s/warp] | 1.41 | 12.0% | 1.0 | 1.41 | `for (int c = 0; c < 4; ++c) \| if (((int)threadIdx.x) < 128) \| if (((int)threadIdx.x) < 192` |
| load.cp_async*4 (est.: sampled iterations x4) [6.4 GB/s/warp] | 0.64 | 5.4% | 1.0 | 0.64 | `if (((((int)blockIdx.x) * 2) + (c_1 >> 1)) < 3) \| if ((c_1 < 3) & (((((int)blockIdx.x) * 4` |
| store.global | 0.51 | 4.3% | 1.0 | 0.51 | `for (int i_10 = 0; i_10 < 32; ++i_10) \| for (int i_11 = 0; i_11 < 32; ++i_11) \| for (int i` |
| wait.sync#2*4 (est.: sampled iterations x4) | 0.26 | 2.2% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync*4 (est.: sampled iterations x4) | 0.26 | 2.2% | 1.0 | 0.26 | `__syncthreads();` |
| wait.pdl | 0.03 | 0.3% | 1.0 | 0.03 | `cudaGridDependencySynchronize();` |

### metadata_builder_kernel grid (2, 1, 1) block (128, 1, 1)

130 launches, 8 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 1.25 us, warp 1.02 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.45 | 43.8% | 0.50 | 47.3% |
| load | 0.35 | 34.4% | 0.33 | 31.3% |
| store | 0.19 | 18.8% | 0.19 | 18.1% |
| wait | 0.03 | 3.1% | 0.04 | 3.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 | 0.22 | 21.9% | 1.0 | 0.22 | `if ((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) < nr))` |
| load.global | 0.13 | 12.5% | 1.0 | 0.13 | `if ((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) <= nr))` |
| store.global | 0.10 | 9.4% | 1.0 | 0.10 | `out_starts[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] = condval;` |
| store.global#3 | 0.06 | 6.2% | 1.0 | 0.06 | `if (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) == 0)` |
| store.global#2 | 0.03 | 3.1% | 1.0 | 0.03 | `request_positions[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] = condval_2;` |
| wait.pdl | 0.03 | 3.1% | 1.0 | 0.03 | `cudaGridDependencySynchronize();` |

### indexer_topk_segment_kernel grid (4, 6, 1) block (512, 1, 1)

20 launches, 384 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 6.69 us, warp 6.67 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 5.70 | 85.4% | 6.93 | 85.9% |
| unattributed | 0.46 | 7.0% | 0.60 | 7.5% |
| load | 0.42 | 6.2% | 0.44 | 5.4% |
| store | 0.10 | 1.4% | 0.09 | 1.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.pdl | 1.71 | 25.7% | 1.0 | 1.71 | `cudaGridDependencySynchronize();` |
| wait.sync#8*9 (est.: sampled iterations x9) | 0.86 | 12.9% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#2*8 (est.: sampled iterations x8) | 0.77 | 11.5% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync#6*9 (est.: sampled iterations x9) | 0.58 | 8.6% | 1.0 | 0.58 | `__syncthreads();` |
| wait.sync#4*8 (est.: sampled iterations x8) | 0.51 | 7.7% | 1.0 | 0.51 | `__syncthreads();` |
| wait.sync#3 | 0.45 | 6.7% | 1.0 | 0.45 | `__syncthreads();` |
| wait.sync | 0.37 | 5.5% | 1.0 | 0.37 | `__syncthreads();` |
| wait.sync#5 | 0.32 | 4.8% | 1.0 | 0.32 | `__syncthreads();` |
| load.global#2 | 0.26 | 3.8% | 1.0 | 0.26 | `for (int i = 0; i < 8; ++i)` |
| load.global | 0.16 | 2.4% | 1.0 | 0.16 | `int extent = min(max(Lengths[((int64_t)((int)blockIdx.y))], 0), 16384); \| int extent = min` |
| store.global | 0.10 | 1.4% | 1.0 | 0.10 | `for (int i_3 = 0; i_3 < 4; ++i_3) \| if (((int)threadIdx.x) < total)` |
| wait.sync#7 | 0.06 | 1.0% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#9 | 0.06 | 1.0% | 1.0 | 0.06 | `__syncthreads();` |

### indexer_topk_segment_kernel grid (1, 6, 1) block (512, 1, 1)

20 launches, 96 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 6.05 us, warp 6.05 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 4.90 | 81.0% | 5.56 | 80.1% |
| unattributed | 0.67 | 11.1% | 0.49 | 7.1% |
| load | 0.42 | 6.9% | 0.74 | 10.6% |
| store | 0.06 | 1.1% | 0.15 | 2.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.pdl | 1.15 | 19.0% | 1.0 | 1.15 | `cudaGridDependencySynchronize();` |
| wait.sync#8*9 (est.: sampled iterations x9) | 0.86 | 14.3% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#6*9 (est.: sampled iterations x9) | 0.58 | 9.5% | 1.0 | 0.58 | `__syncthreads();` |
| wait.sync#2*8 (est.: sampled iterations x8) | 0.51 | 8.5% | 1.0 | 0.51 | `__syncthreads();` |
| wait.sync#4*8 (est.: sampled iterations x8) | 0.51 | 8.5% | 1.0 | 0.51 | `__syncthreads();` |
| wait.sync#3 | 0.45 | 7.4% | 1.0 | 0.45 | `__syncthreads();` |
| load.global | 0.42 | 6.9% | 1.0 | 0.42 | `int extent = min(max(Lengths[((int64_t)((int)blockIdx.y))], 0), 16384); \| int extent = min` |
| wait.sync | 0.38 | 6.3% | 1.0 | 0.38 | `__syncthreads();` |
| wait.sync#5 | 0.32 | 5.3% | 1.0 | 0.32 | `__syncthreads();` |
| store.global | 0.06 | 1.1% | 1.0 | 0.06 | `for (int i_3 = 0; i_3 < 4; ++i_3) \| if (((int)threadIdx.x) < total)` |
| wait.sync#7 | 0.06 | 1.1% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#9 | 0.06 | 1.1% | 1.0 | 0.06 | `__syncthreads();` |

### rope_kernel grid (1, 1, 1) block (128, 1, 1)

55 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 1.47 us, warp 1.47 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.15 | 78.3% | 0.91 | 42.0% |
| load | 0.26 | 17.4% | 0.35 | 16.3% |
| store | 0.03 | 2.2% | 0.04 | 1.9% |
| wait | 0.03 | 2.2% | 0.86 | 39.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.19 | 13.0% | 1.0 | 0.19 | `if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < tokens)) \| int64_t position ` |
| load.global#2 | 0.06 | 4.3% | 1.0 | 0.06 | `if (((position < ((int64_t)positions_max)) && ((int64_t)0 <= position)))` |
| store.global | 0.03 | 2.2% | 1.0 | 0.03 | `*(uint4*)(x + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + ((((int64_t)((int)thread` |
| wait.pdl | 0.03 | 2.2% | 1.0 | 0.03 | `cudaGridDependencySynchronize();` |

### partial_argmax_kernel grid (1, 64, 1) block (128, 1, 1)

25 launches, 256 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 4.58 us, warp 4.58 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 4.06 | 88.8% | 4.10 | 87.7% |
| load | 0.19 | 4.2% | 0.18 | 3.9% |
| compute | 0.13 | 2.8% | 0.13 | 2.7% |
| store | 0.13 | 2.8% | 0.13 | 2.7% |
| wait | 0.06 | 1.4% | 0.14 | 3.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.19 | 4.2% | 1.0 | 0.19 | `for (int j = 0; j < 16; ++j) \| for (int run = 0; run < 2; ++run)` |
| compute.reduce | 0.13 | 2.8% | 1.0 | 0.13 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| store.global | 0.13 | 2.8% | 1.0 | 0.13 | `if (((int)threadIdx.x) == 0)` |
| wait.sync | 0.06 | 1.4% | 1.0 | 0.06 | `__syncthreads();` |

### token_ranks_kernel grid (1, 1, 1) block (256, 1, 1)

15 launches, 8 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 7.52 us, warp 7.52 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 6.82 | 90.6% | 6.90 | 90.5% |
| unattributed | 0.35 | 4.7% | 0.37 | 4.8% |
| compute | 0.29 | 3.8% | 0.30 | 3.9% |
| store | 0.06 | 0.9% | 0.06 | 0.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 6.82 | 90.6% | 1.0 | 6.82 | `if ((((int64_t)0 <= token_ids[((int64_t)((int)blockIdx.x))]) && (token_ids[((int64_t)((int` |
| compute.reduce | 0.29 | 3.8% | 1.0 | 0.29 | `total[0] = tl::AllReduce<tl::SumOp, 256, 1, 0, tl::NamedBarrier<256>>::run(total[0], (&(wo` |
| store.global | 0.06 | 0.9% | 1.0 | 0.06 | `output[((int64_t)((int)blockIdx.x))] = ((int64_t)total[0]);` |

### norm_forward_and_per_token_cast_kernel_kernel grid (1, 1, 1) block (128, 1, 1)

35 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 3.14 us, warp 3.14 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.79 | 57.1% | 1.00 | 31.0% |
| store | 1.02 | 32.7% | 1.78 | 55.0% |
| load | 0.16 | 5.1% | 0.28 | 8.6% |
| compute | 0.10 | 3.1% | 0.10 | 2.9% |
| wait | 0.06 | 2.0% | 0.08 | 2.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*2 (est.: sampled iterations x2) | 0.70 | 22.4% | 1.0 | 0.70 | `for (int j_2 = 0; j_2 < 2; ++j_2)` |
| store.global*2 (est.: sampled iterations x2) | 0.19 | 6.1% | 1.0 | 0.19 | `for (int j_1 = 0; j_1 < 4; ++j_1)` |
| compute.reduce | 0.10 | 3.1% | 1.0 | 0.10 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| load.global#3*2 (est.: sampled iterations x2) | 0.06 | 2.0% | 1.0 | 0.06 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 64) + ((((int)threadIdx.x) & 7) ` |
| load.global*2 (est.: sampled iterations x2) | 0.06 | 2.0% | 1.0 | 0.06 | `if (((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < num_tokens)` |
| store.global#5*2 (est.: sampled iterations x2) | 0.06 | 2.0% | 1.0 | 0.06 | `*(uint4*)(out + ((((((int64_t)vi_1) * (int64_t)64) + ((((int64_t)((int)threadIdx.x)) & (in` |
| load.global#2 | 0.03 | 1.0% | 1.0 | 0.03 | `*(float2*)(row_partial + 0) = *(float2*)(sum_shared + ((((int)threadIdx.x) >> 6) * 2)); \| ` |
| store.global#2 | 0.03 | 1.0% | 1.0 | 0.03 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| store.global#3 | 0.03 | 1.0% | 1.0 | 0.03 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |
| wait.pdl | 0.03 | 1.0% | 1.0 | 0.03 | `cudaGridDependencySynchronize();` |
| wait.sync | 0.03 | 1.0% | 1.0 | 0.03 | `__syncthreads();` |
| load.global#3*4 (est.: sampled iterations x4) | 0.00 | 0.0% | 0.0 | 0.00 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 128) + ((((int)threadIdx.x) & 15` |
| load.global*4 (est.: sampled iterations x4) | 0.00 | 0.0% | 0.0 | 0.00 | `if (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 4)) < num_tokens)` |
| store.global#4*4 (est.: sampled iterations x4) | 0.00 | 0.0% | 0.0 | 0.00 | `for (int j_2 = 0; j_2 < 2; ++j_2)` |
| store.global#5*4 (est.: sampled iterations x4) | 0.00 | 0.0% | 0.0 | 0.00 | `*(uint4*)(out + ((((((int64_t)vi_1) * (int64_t)128) + ((((int64_t)((int)threadIdx.x)) & (i` |
| store.global*4 (est.: sampled iterations x4) | 0.00 | 0.0% | 0.0 | 0.00 | `for (int j_1 = 0; j_1 < 4; ++j_1)` |

### metadata_pages_kernel grid (6, 1, 1) block (128, 1, 1)

40 launches, 24 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 1.31 us, warp 1.31 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.80 | 61.0% | 1.61 | 59.7% |
| store | 0.35 | 26.8% | 0.91 | 33.8% |
| load | 0.13 | 9.8% | 0.14 | 5.0% |
| wait | 0.03 | 2.4% | 0.04 | 1.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.35 | 26.8% | 1.0 | 0.35 | `if (((int)threadIdx.x) < 32)` |
| load.global | 0.06 | 4.9% | 1.0 | 0.06 | `if ((((offset + ((int)blockIdx.x)) < src_rows) && (0 <= (offset + ((int)blockIdx.x)))))` |
| load.global#2 | 0.06 | 4.9% | 1.0 | 0.06 | `if (((req < table_rows) && (((int)threadIdx.x) < table_cols)))` |
| wait.pdl | 0.03 | 2.4% | 1.0 | 0.03 | `cudaGridDependencySynchronize();` |

### indexer_mxfp4_quantize_kernel grid (1, 1, 1) block (128, 1, 1)

20 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 4.05 us, warp 4.05 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 3.52 | 87.0% | 4.26 | 93.4% |
| store | 1.58 | 39.1% | 1.58 | 34.7% |
| wait | 0.03 | 0.8% | 0.15 | 3.3% |
| unattributed | 0.00 | 0.0% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2*16 (est.: sampled iterations x16) | 3.07 | 75.9% | 1.0 | 3.07 | `float v__2 = __fmul_rn(((float)X[(((((int64_t)((int)blockIdx.x)) * (int64_t)4096) + (((int` |
| store.global*16 (est.: sampled iterations x16) | 1.54 | 37.9% | 1.0 | 1.54 | `Pool[(((((slot & (int64_t)127) * (int64_t)64) + ((((int64_t)((int)threadIdx.x)) & (int64_t` |
| load.global | 0.45 | 11.1% | 1.0 | 0.45 | `for (int i = 0; i < 32; ++i)` |
| store.global#2 | 0.05 | 1.2% | 1.0 | 0.05 | `Pool[(((((slot & (int64_t)127) * (int64_t)4) + ((slot >> (int64_t)7) * ((int64_t)pool_stri` |
| wait.pdl | 0.03 | 0.8% | 1.0 | 0.03 | `cudaGridDependencySynchronize();` |

### gemm_bf16_partials_kernel grid (2, 4, 1) block (256, 1, 1)

20 launches, 64 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 3.92 us, warp 0.93 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.61 | 65.5% | 1.13 | 77.7% |
| load | 0.19 | 20.7% | 0.19 | 12.7% |
| wait | 0.13 | 13.8% | 0.14 | 9.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.tma*2 (est.: sampled iterations x2) [170.667 GB/s/warp] | 0.19 | 20.7% | 1.0 | 0.19 | `if (tl::tl_shuffle_elect<128>())` |
| wait.mbarrier*2 (est.: sampled iterations x2) | 0.10 | 10.3% | 1.0 | 0.10 | `mbarrier[(kb + 3)].wait(1);` |
| wait.sync | 0.03 | 3.4% | 1.0 | 0.03 | `__syncthreads();` |

### topk_ids_kernel grid (5, 1, 1) block (512, 1, 1)

4 launches, 80 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 20.74 us, warp 20.72 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 15.90 | 76.8% | 15.79 | 76.2% |
| store | 2.72 | 13.1% | 2.67 | 12.9% |
| wait | 1.79 | 8.6% | 1.83 | 8.8% |
| unattributed | 0.30 | 1.5% | 0.43 | 2.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 15.90 | 76.8% | 1.0 | 15.90 | `for (int t = 0; t < (vocab >> 14); ++t)` |
| store.global | 2.72 | 13.1% | 1.0 | 2.72 | `if ((((int)threadIdx.x) >> 5) == 0)` |
| wait.sync | 1.79 | 8.6% | 1.0 | 1.79 | `__syncthreads();` |

### topk_log_softmax_kernel grid (1, 1, 1) block (1024, 1, 1)

5 launches, 32 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 15.94 us, warp 15.90 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 13.54 | 85.1% | 13.57 | 85.0% |
| unattributed, with loops' unsampled iterations | 1.47 | 9.3% | 1.36 | 8.5% |
| compute | 0.67 | 4.2% | 0.76 | 4.8% |
| wait | 0.16 | 1.0% | 0.22 | 1.4% |
| store | 0.06 | 0.4% | 0.05 | 0.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 6.62 | 41.6% | 1.0 | 6.62 | `if (0 < (vocab & 2047)) \| if (0 < (vocab & 8191))` |
| load.global#2 | 6.34 | 39.8% | 1.0 | 6.34 | `for (int step_1 = 0; step_1 < (vocab >> 11); ++step_1) \| for (int step_1 = 0; step_1 < (vo` |
| compute.reduce#2 | 0.58 | 3.6% | 1.0 | 0.58 | `se[0] = vllm_add(se[0], __shfl_xor_sync((uint)4294967295, se[0], 16, 32));` |
| load.global#3 (sampled iterations) | 0.58 | 3.6% | 1.0 | 0.58 | `float value_32 = vllm_sub(vllm_shifted(logits[((((int64_t)((int)blockIdx.x)) * logits_stri` |
| compute.reduce | 0.10 | 0.6% | 1.0 | 0.10 | `best[0] = vllm_max(best[0], __shfl_xor_sync((uint)4294967295, best[0], 16, 32));` |
| wait.sync#2 | 0.10 | 0.6% | 1.0 | 0.10 | `__syncthreads();` |
| store.global (sampled iterations) | 0.06 | 0.4% | 1.0 | 0.06 | `output[(((((int64_t)c) * (int64_t)1024) + (((int64_t)((int)blockIdx.x)) * ((int64_t)topk))` |
| wait.sync | 0.06 | 0.4% | 1.0 | 0.06 | `__syncthreads();` |

### per_token_cast_kernel_kernel grid (5, 1, 1) block (64, 1, 1)

40 launches, 10 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 1.22 us, warp 1.22 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.51 | 42.1% | 0.95 | 55.6% |
| load | 0.29 | 23.7% | 0.29 | 17.1% |
| store | 0.26 | 21.1% | 0.28 | 16.3% |
| compute | 0.13 | 10.5% | 0.10 | 5.7% |
| wait | 0.03 | 2.6% | 0.09 | 5.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.22 | 18.4% | 1.0 | 0.22 | `for (int i = 0; i < 2; ++i)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.13 | 10.5% | 1.0 | 0.13 | `amax_fragment[i_1] = tl::AllReduce<tl::MaxOp, 2, 1, 0, tl::NamedBarrier<64>>::run(amax_fra` |
| store.global#2 | 0.13 | 10.5% | 1.0 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3)` |
| load.global#2*2 (est.: sampled iterations x2) | 0.06 | 5.3% | 1.0 | 0.06 | `amax_fragment[i_1] = cutlass::bfloat16_t(__hmax((bfloat16_t(((nv_bfloat162*)(&(amax_fragme` |
| store.global#3*2 (est.: sampled iterations x2) | 0.06 | 5.3% | 1.0 | 0.06 | `*(fp8_e4_16_t*)(out + ((((((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int64_t)10240) +` |
| store.global*2 (est.: sampled iterations x2) | 0.06 | 5.3% | 1.0 | 0.06 | `out_sf[((((((int64_t)((int)blockIdx.x)) % (int64_t)15) * (int64_t)32) + (((int64_t)((int)t` |
| wait.pdl | 0.03 | 2.6% | 1.0 | 0.03 | `cudaGridDependencySynchronize();` |

### topk_log_softmax_kernel grid (5, 1, 1) block (1024, 1, 1)

4 launches, 160 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 16.05 us, warp 16.00 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 13.62 | 85.1% | 13.61 | 84.9% |
| unattributed, with loops' unsampled iterations | 1.49 | 9.3% | 1.50 | 9.3% |
| compute | 0.64 | 4.0% | 0.68 | 4.2% |
| wait | 0.19 | 1.2% | 0.19 | 1.2% |
| store | 0.06 | 0.4% | 0.06 | 0.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 | 6.66 | 41.6% | 1.0 | 6.66 | `for (int step_1 = 0; step_1 < (vocab >> 11); ++step_1) \| for (int step_1 = 0; step_1 < (vo` |
| load.global | 6.61 | 41.3% | 1.0 | 6.61 | `if (0 < (vocab & 2047)) \| if (0 < (vocab & 8191))` |
| compute.reduce#2 | 0.54 | 3.4% | 1.0 | 0.54 | `se[0] = vllm_add(se[0], __shfl_xor_sync((uint)4294967295, se[0], 16, 32));` |
| load.global#3 (sampled iterations) | 0.35 | 2.2% | 1.0 | 0.35 | `float value_32 = vllm_sub(vllm_shifted(logits[((((int64_t)((int)blockIdx.x)) * logits_stri` |
| wait.sync#2 | 0.11 | 0.7% | 1.0 | 0.11 | `__syncthreads();` |
| compute.reduce | 0.10 | 0.6% | 1.0 | 0.10 | `best[0] = vllm_max(best[0], __shfl_xor_sync((uint)4294967295, best[0], 16, 32));` |
| wait.sync | 0.08 | 0.5% | 1.0 | 0.08 | `__syncthreads();` |
| store.global (sampled iterations) | 0.06 | 0.4% | 1.0 | 0.06 | `output[(((((int64_t)c) * (int64_t)1024) + (((int64_t)((int)blockIdx.x)) * ((int64_t)topk))` |

### mhc_post_kernel grid (30, 1, 1) block (128, 1, 1)

20 launches, 120 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 3.14 us, warp 2.91 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 2.75 | 94.5% | 2.59 | 87.9% |
| load | 0.06 | 2.2% | 0.08 | 2.6% |
| store | 0.06 | 2.2% | 0.05 | 1.7% |
| wait | 0.03 | 1.1% | 0.23 | 7.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.06 | 2.2% | 1.0 | 0.06 | `*(float4*)(pm + 0) = *(float4*)(post + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int` |
| store.global | 0.06 | 2.2% | 1.0 | 0.06 | `for (int n_1 = 0; n_1 < 4; ++n_1)` |
| wait.pdl | 0.03 | 1.1% | 1.0 | 0.03 | `cudaGridDependencySynchronize();` |

### local_stats_kernel grid (5, 16, 1) block (128, 1, 1)

4 launches, 320 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 15.26 us, warp 15.04 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 13.81 | 91.8% | 13.76 | 91.6% |
| unattributed | 0.99 | 6.6% | 1.01 | 6.7% |
| compute | 0.16 | 1.1% | 0.17 | 1.1% |
| store | 0.08 | 0.5% | 0.08 | 0.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 13.81 | 91.8% | 1.0 | 13.81 | `if (expanded_local_pos[((int64_t)((int)blockIdx.x))] < num_speculative_steps)` |
| compute.reduce | 0.16 | 1.1% | 1.0 | 0.16 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| store.global | 0.08 | 0.5% | 1.0 | 0.08 | `if (((int)threadIdx.x) == 0)` |

### gemm_splitk_reduce_kernel grid (1, 6, 1) block (128, 1, 1)

60 launches, 24 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 0.90 us, warp 0.90 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.45 | 50.0% | 0.60 | 59.3% |
| load | 0.35 | 39.3% | 0.31 | 31.3% |
| store | 0.06 | 7.1% | 0.06 | 5.5% |
| wait | 0.03 | 3.6% | 0.04 | 3.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.35 | 39.3% | 1.0 | 0.35 | `acc[0] = P[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x))` |
| store.global | 0.06 | 7.1% | 1.0 | 0.06 | `C[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x)) * (int64` |
| wait.pdl | 0.03 | 3.6% | 1.0 | 0.03 | `cudaGridDependencySynchronize();` |

### indexer_score_kernel grid (64, 6, 1) block (128, 1, 1)

15 launches, 1536 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 3.68 us, warp 3.65 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 2.18 | 59.6% | 2.22 | 56.9% |
| unattributed, with loops' unsampled iterations | 1.09 | 29.8% | 1.12 | 28.7% |
| wait | 0.19 | 5.3% | 0.38 | 9.6% |
| compute | 0.13 | 3.5% | 0.13 | 3.4% |
| store | 0.06 | 1.8% | 0.05 | 1.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 (sampled iterations) | 1.70 | 46.5% | 1.0 | 1.70 | `for (int i_1 = 0; i_1 < 32; ++i_1) \| if ((((i * 8192) + (((int)blockIdx.x) * 64)) + 64) <=` |
| load.global | 0.48 | 13.2% | 1.0 | 0.48 | `int extent = min(max(CandidateLengths[((int64_t)((int)blockIdx.y))], 0), 16384); \| int ext` |
| wait.sync (sampled iterations) | 0.16 | 4.4% | 1.0 | 0.16 | `__syncthreads();` |
| compute.mma (sampled iterations) | 0.13 | 3.5% | 1.0 | 0.13 | `{` |
| store.global (sampled iterations) | 0.06 | 1.8% | 1.0 | 0.06 | `Scores[((((((int64_t)((int)blockIdx.y)) * (int64_t)16384) + (((int64_t)i) * (int64_t)8192)` |
| wait.pdl | 0.03 | 0.9% | 1.0 | 0.03 | `cudaGridDependencySynchronize();` |

### moe_swiglu_cast_kernel_kernel grid (8, 15, 1) block (288, 1, 1)

15 launches, 1080 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 2.88 us, warp 2.88 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.89 | 65.6% | 2.20 | 69.1% |
| compute | 0.38 | 13.3% | 0.38 | 12.1% |
| load | 0.29 | 10.0% | 0.30 | 9.4% |
| store | 0.26 | 8.9% | 0.25 | 7.8% |
| wait | 0.06 | 2.2% | 0.05 | 1.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.reduce*4 (est.: sampled iterations x4) | 0.38 | 13.3% | 1.0 | 0.38 | `if (0 <= orders[r_2])` |
| load.global | 0.29 | 10.0% | 1.0 | 0.29 | `for (int r = 0; r < 4; ++r)` |
| store.global*4 (est.: sampled iterations x4) | 0.26 | 8.9% | 1.0 | 0.26 | `if (0 <= orders[r_2])` |
| wait.pdl | 0.06 | 2.2% | 1.0 | 0.06 | `cudaGridDependencySynchronize();` |

### cache_writer_kernel grid (6, 1, 1) block (128, 1, 1)

23 launches, 24 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 1.31 us, warp 1.12 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.48 | 42.9% | 0.66 | 48.1% |
| store | 0.29 | 25.7% | 0.37 | 27.4% |
| load | 0.26 | 22.9% | 0.24 | 17.2% |
| compute | 0.06 | 5.7% | 0.07 | 4.8% |
| wait | 0.03 | 2.9% | 0.03 | 2.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.29 | 25.7% | 1.0 | 0.29 | `if (((int64_t)0 <= slot) & ((slot >> (int64_t)7) < ((int64_t)pages))) \| if (((int64_t)0 <=` |
| load.global | 0.19 | 17.1% | 1.0 | 0.19 | `for (int i = 0; i < 4; ++i)` |
| compute.reduce | 0.06 | 5.7% | 1.0 | 0.06 | `amax = max(amax, __shfl_xor_sync((uint)4294967295, amax, 1, 32));` |
| load.global#2 | 0.06 | 5.7% | 1.0 | 0.06 | `int64_t slot = slots[((int64_t)((int)blockIdx.x))];` |
| wait.pdl | 0.03 | 2.9% | 1.0 | 0.03 | `cudaGridDependencySynchronize();` |

### mhc_collapse_kernel grid (30, 1, 1) block (128, 1, 1)

15 launches, 120 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 2.72 us, warp 2.34 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 1.22 | 52.1% | 1.13 | 49.7% |
| unattributed | 0.99 | 42.5% | 1.04 | 45.4% |
| load | 0.06 | 2.7% | 0.06 | 2.7% |
| store | 0.06 | 2.7% | 0.05 | 2.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.pdl | 1.22 | 52.1% | 1.0 | 1.22 | `cudaGridDependencySynchronize();` |
| load.global | 0.06 | 2.7% | 1.0 | 0.06 | `*(float4*)(mv + 0) = *(float4*)(mix + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int6` |
| store.global | 0.06 | 2.7% | 1.0 | 0.06 | `*(uint4*)(out + ((((int64_t)((int)blockIdx.x)) * (int64_t)1024) + (((int64_t)((int)threadI` |

### norm_forward_and_per_token_cast_kernel_kernel grid (5, 1, 1) block (128, 1, 1)

9 launches, 20 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 3.46 us, warp 3.46 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 2.21 | 63.9% | 2.20 | 59.7% |
| unattributed | 0.74 | 21.3% | 0.91 | 24.7% |
| load | 0.32 | 9.3% | 0.36 | 9.7% |
| compute | 0.13 | 3.7% | 0.13 | 3.6% |
| wait | 0.06 | 1.9% | 0.09 | 2.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*5 (est.: sampled iterations x5) | 1.92 | 55.6% | 1.0 | 1.92 | `for (int j_1 = 0; j_1 < 2; ++j_1)` |
| load.global#2*5 (est.: sampled iterations x5) | 0.16 | 4.6% | 1.0 | 0.16 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 1024) + (((int)threadIdx.x) * 8)` |
| load.global*5 (est.: sampled iterations x5) | 0.16 | 4.6% | 1.0 | 0.16 | `*(uint4*)(x_local + (vi * 8)) = *(uint4*)(x + ((((((int64_t)vi) * (int64_t)1024) + (((int6` |
| store.global*5 (est.: sampled iterations x5) | 0.16 | 4.6% | 1.0 | 0.16 | `for (int j = 0; j < 4; ++j)` |
| compute.reduce | 0.13 | 3.7% | 1.0 | 0.13 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| store.global#3 | 0.10 | 2.8% | 1.0 | 0.10 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |
| store.global#2 | 0.03 | 0.9% | 1.0 | 0.03 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| wait.pdl | 0.03 | 0.9% | 1.0 | 0.03 | `cudaGridDependencySynchronize();` |
| wait.sync | 0.03 | 0.9% | 1.0 | 0.03 | `__syncthreads();` |

### metadata_chunk_kernel grid (5, 1, 1) block (192, 1, 1)

15 launches, 30 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 2.34 us, warp 2.30 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.60 | 69.4% | 1.48 | 67.9% |
| unattributed | 0.58 | 25.0% | 0.53 | 24.3% |
| store | 0.10 | 4.2% | 0.13 | 6.1% |
| wait | 0.03 | 1.4% | 0.04 | 1.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.31 | 56.9% | 1.0 | 1.31 | `if ((((offset + ((int)blockIdx.x)) < src_rows) && (0 <= (offset + ((int)blockIdx.x)))))` |
| load.global#2 | 0.29 | 12.5% | 1.0 | 0.29 | `if (((0 <= req) & ((int64_t)0 <= pos)))` |
| store.global | 0.03 | 1.4% | 1.0 | 0.03 | `swa[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = con` |
| store.global#2 | 0.03 | 1.4% | 1.0 | 0.03 | `lengths[((int64_t)((int)blockIdx.x))] = ((int)max(info[1], (int64_t)0));` |
| store.global#3 | 0.03 | 1.4% | 1.0 | 0.03 | `top_lengths[((int64_t)((int)blockIdx.x))] = condval_12; \| top_lengths[((int64_t)((int)bloc` |
| wait.pdl | 0.03 | 1.4% | 1.0 | 0.03 | `cudaGridDependencySynchronize();` |

### token_ranks_kernel grid (5, 1, 1) block (256, 1, 1)

4 launches, 40 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 7.71 us, warp 7.66 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 6.80 | 88.7% | 6.81 | 88.6% |
| compute | 0.42 | 5.4% | 0.43 | 5.6% |
| unattributed | 0.38 | 5.0% | 0.39 | 5.1% |
| store | 0.06 | 0.8% | 0.06 | 0.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 6.80 | 88.7% | 1.0 | 6.80 | `if ((((int64_t)0 <= token_ids[((int64_t)((int)blockIdx.x))]) && (token_ids[((int64_t)((int` |
| compute.reduce | 0.42 | 5.4% | 1.0 | 0.42 | `total[0] = tl::AllReduce<tl::SumOp, 256, 1, 0, tl::NamedBarrier<256>>::run(total[0], (&(wo` |
| store.global | 0.06 | 0.8% | 1.0 | 0.06 | `output[((int64_t)((int)blockIdx.x))] = ((int64_t)total[0]);` |

### prepare_dflash_inputs_kernel grid (9, 1, 1) block (256, 1, 1)

15 launches, 72 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 1.79 us, warp 1.79 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.18 | 66.1% | 1.36 | 67.7% |
| unattributed | 0.35 | 19.6% | 0.38 | 19.0% |
| store | 0.26 | 14.3% | 0.27 | 13.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.12 | 62.5% | 1.0 | 1.12 | `req = idx_mapping[((int)blockIdx.y)];` |
| store.global#2 | 0.26 | 14.3% | 1.0 | 0.26 | `if (((int)blockIdx.y) == (num_reqs - 1))` |
| load.global#2 | 0.06 | 3.6% | 1.0 | 0.06 | `vllm_store_if_b32((&(out_query_start_loc[((int)blockIdx.y)])), (((int)blockIdx.y) * num_qu` |

### indexer_topk_segment_kernel grid (2, 6, 1) block (512, 1, 1)

5 launches, 192 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 5.31 us, warp 5.31 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 4.29 | 80.7% | 4.30 | 80.2% |
| unattributed | 0.48 | 9.0% | 0.52 | 9.7% |
| load | 0.45 | 8.4% | 0.45 | 8.4% |
| store | 0.10 | 1.8% | 0.10 | 1.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync#8*9 (est.: sampled iterations x9) | 0.86 | 16.3% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#4*8 (est.: sampled iterations x8) | 0.77 | 14.5% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync#6*9 (est.: sampled iterations x9) | 0.58 | 10.8% | 1.0 | 0.58 | `__syncthreads();` |
| wait.sync#2*8 (est.: sampled iterations x8) | 0.51 | 9.6% | 1.0 | 0.51 | `__syncthreads();` |
| wait.sync#3 | 0.45 | 8.4% | 1.0 | 0.45 | `__syncthreads();` |
| wait.sync | 0.35 | 6.6% | 1.0 | 0.35 | `__syncthreads();` |
| wait.pdl | 0.32 | 6.0% | 1.0 | 0.32 | `cudaGridDependencySynchronize();` |
| wait.sync#5 | 0.32 | 6.0% | 1.0 | 0.32 | `__syncthreads();` |
| load.global#2 | 0.26 | 4.8% | 1.0 | 0.26 | `for (int i = 0; i < 8; ++i)` |
| load.global | 0.19 | 3.6% | 1.0 | 0.19 | `int extent = min(max(Lengths[((int64_t)((int)blockIdx.y))], 0), 16384); \| int extent = min` |
| store.global | 0.10 | 1.8% | 1.0 | 0.10 | `for (int i_3 = 0; i_3 < 4; ++i_3) \| if (((int)threadIdx.x) < total)` |
| wait.sync#7 | 0.06 | 1.2% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#9 | 0.06 | 1.2% | 1.0 | 0.06 | `__syncthreads();` |

### topk_ids_kernel grid (6, 1, 1) block (512, 1, 1)

1 launches, 96 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 24.99 us, warp 21.66 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 16.54 | 76.4% | 16.54 | 76.4% |
| store | 3.17 | 14.6% | 3.17 | 14.6% |
| wait | 1.54 | 7.1% | 1.54 | 7.1% |
| unattributed | 0.42 | 1.9% | 0.42 | 1.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 16.54 | 76.4% | 1.0 | 16.54 | `for (int t = 0; t < (vocab >> 14); ++t)` |
| store.global | 3.17 | 14.6% | 1.0 | 3.17 | `if ((((int)threadIdx.x) >> 5) == 0)` |
| wait.sync | 1.54 | 7.1% | 1.0 | 1.54 | `__syncthreads();` |

### sink_merge_kernel grid (1, 1, 1) block (128, 1, 1)

5 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 4.19 us, warp 4.19 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 2.05 | 48.9% | 1.96 | 39.6% |
| unattributed | 0.90 | 21.4% | 1.72 | 34.8% |
| compute | 0.58 | 13.7% | 0.54 | 10.9% |
| wait | 0.58 | 13.7% | 0.61 | 12.4% |
| store | 0.10 | 2.3% | 0.12 | 2.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2*3 (est.: sampled iterations x3) | 1.63 | 38.9% | 1.0 | 1.63 | `for (int step = 0; step < ((width + 127) >> 7); ++step) \| for (int step = 0; step < ((widt` |
| compute.reduce*3 (est.: sampled iterations x3) | 0.58 | 13.7% | 1.0 | 0.58 | `float other = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| wait.sync*3 (est.: sampled iterations x3) | 0.48 | 11.5% | 1.0 | 0.48 | `__syncthreads();` |
| load.global | 0.22 | 5.3% | 1.0 | 0.22 | `last = sampled[(max((((int64_t)num_sampled[((int64_t)((int)blockIdx.x))]) - (int64_t)1), (` |
| load.global#3*3 (est.: sampled iterations x3) | 0.19 | 4.6% | 1.0 | 0.19 | `if ((best[0] == -CUDART_INF_F))` |
| store.global*3 (est.: sampled iterations x3) | 0.10 | 2.3% | 1.0 | 0.10 | `alternates[((((int64_t)((int)blockIdx.x)) * (int64_t)3) + ((int64_t)rank))] = condval_31;` |
| wait.sync#2*3 (est.: sampled iterations x3) | 0.10 | 2.3% | 1.0 | 0.10 | `__syncthreads();` |

### sparse_mla_kernel_1 grid (16, 5, 1) block (128, 1, 1)

15 launches, 80 CTAs, 320 warps; span 1.44 us, warp 1.25 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.24 | 99.8% | 0.63 | 43.5% |
| load | 0.00 | 0.2% | 0.61 | 42.0% |
| compute | 0.00 | 0.0% | 0.09 | 6.0% |
| wait | 0.00 | 0.0% | 0.08 | 5.3% |
| store | 0.00 | 0.0% | 0.05 | 3.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.00 | 0.2% | 0.0 | 0.61 | `M = __fmul_rn(Sinks[((int)blockIdx.x)], 0x1.7154764ee6c2fp+0f/*1.442695e+00*/);` |
| compute.reduce | 0.00 | 0.0% | 0.0 | 0.10 | `amax[0] = tl::AllReduce<tl::MaxOp, 8, 1, 0, tl::NamedBarrier<128>>::run(amax[0]);` |
| store.global#2 | 0.00 | 0.0% | 0.0 | 0.03 | `*(uint2*)(Output + (((((int64_t)((int)blockIdx.y)) * (int64_t)8192) + (((int64_t)((int)blo` |
| wait.pdl#2 | 0.00 | 0.0% | 0.0 | 0.03 | `cudaGridDependencySynchronize();` |
| wait.sync#3 | 0.00 | 0.0% | 0.0 | 0.03 | `__syncthreads();` |

### sink_candidates_kernel grid (1, 127, 1) block (128, 1, 1)

5 launches, 508 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 3.65 us, warp 3.62 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.38 | 38.1% | 1.67 | 39.5% |
| compute | 1.02 | 28.3% | 0.97 | 23.0% |
| load | 0.70 | 19.5% | 1.07 | 25.3% |
| store | 0.26 | 7.1% | 0.28 | 6.7% |
| wait | 0.26 | 7.1% | 0.23 | 5.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.reduce*4 (est.: sampled iterations x4) | 1.02 | 28.3% | 1.0 | 1.02 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| load.global | 0.70 | 19.5% | 1.0 | 0.70 | `local[0] = max((num_sampled[((int64_t)((int)blockIdx.x))] - 1), 0);` |
| store.global*4 (est.: sampled iterations x4) | 0.26 | 7.1% | 1.0 | 0.26 | `if (((int)threadIdx.x) == 0)` |
| wait.sync*4 (est.: sampled iterations x4) | 0.26 | 7.1% | 1.0 | 0.26 | `__syncthreads();` |

### moe_combine_kernel grid (5, 5, 1) block (128, 1, 1)

15 launches, 100 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 1.41 us, warp 1.38 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.80 | 58.1% | 0.80 | 58.4% |
| unattributed | 0.48 | 34.9% | 0.48 | 35.1% |
| store | 0.06 | 4.7% | 0.05 | 3.9% |
| wait | 0.03 | 2.3% | 0.04 | 2.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.80 | 58.1% | 1.0 | 0.80 | `for (int k = 0; k < 3; ++k) \| for (int k = 0; k < 6; ++k) \| if ((((int)blockIdx.y) < share` |
| store.global | 0.06 | 4.7% | 1.0 | 0.06 | `{` |
| wait.pdl | 0.03 | 2.3% | 1.0 | 0.03 | `cudaGridDependencySynchronize();` |

### gemm_splitk_reduce_kernel grid (4, 5, 1) block (128, 1, 1)

12 launches, 80 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 0.67 us, warp 0.64 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.37 | 57.5% | 0.38 | 58.3% |
| load | 0.19 | 30.0% | 0.19 | 28.7% |
| store | 0.05 | 7.5% | 0.05 | 7.3% |
| wait | 0.03 | 5.0% | 0.04 | 5.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.19 | 30.0% | 1.0 | 0.19 | `acc[0] = P[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x))` |
| store.global | 0.05 | 7.5% | 1.0 | 0.05 | `C[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x)) * (int64` |
| wait.pdl | 0.03 | 5.0% | 1.0 | 0.03 | `cudaGridDependencySynchronize();` |

### norm_forward_and_per_token_cast_kernel_kernel grid (6, 1, 1) block (128, 1, 1)

6 launches, 24 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 3.42 us, warp 3.42 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 2.37 | 69.2% | 2.29 | 66.7% |
| unattributed | 0.54 | 15.9% | 0.61 | 17.9% |
| load | 0.32 | 9.3% | 0.32 | 9.3% |
| compute | 0.13 | 3.7% | 0.13 | 3.7% |
| wait | 0.06 | 1.9% | 0.08 | 2.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*5 (est.: sampled iterations x5) | 2.08 | 60.7% | 1.0 | 2.08 | `for (int j_1 = 0; j_1 < 2; ++j_1)` |
| load.global#2*5 (est.: sampled iterations x5) | 0.16 | 4.7% | 1.0 | 0.16 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 1024) + (((int)threadIdx.x) * 8)` |
| load.global*5 (est.: sampled iterations x5) | 0.16 | 4.7% | 1.0 | 0.16 | `*(uint4*)(x_local + (vi * 8)) = *(uint4*)(x + ((((((int64_t)vi) * (int64_t)1024) + (((int6` |
| store.global*5 (est.: sampled iterations x5) | 0.16 | 4.7% | 1.0 | 0.16 | `for (int j = 0; j < 4; ++j)` |
| compute.reduce | 0.13 | 3.7% | 1.0 | 0.13 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| store.global#3 | 0.10 | 2.8% | 1.0 | 0.10 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |
| store.global#2 | 0.03 | 0.9% | 1.0 | 0.03 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| wait.pdl | 0.03 | 0.9% | 1.0 | 0.03 | `cudaGridDependencySynchronize();` |
| wait.sync | 0.03 | 0.9% | 1.0 | 0.03 | `__syncthreads();` |

### per_token_cast_kernel_kernel grid (15, 1, 1) block (64, 1, 1)

15 launches, 30 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 1.15 us, warp 1.15 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.45 | 38.9% | 0.46 | 39.7% |
| load | 0.29 | 25.0% | 0.28 | 24.0% |
| store | 0.26 | 22.2% | 0.28 | 24.2% |
| compute | 0.13 | 11.1% | 0.10 | 8.9% |
| wait | 0.03 | 2.8% | 0.04 | 3.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.22 | 19.4% | 1.0 | 0.22 | `for (int i = 0; i < 2; ++i)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.13 | 11.1% | 1.0 | 0.13 | `amax_fragment[i_1] = tl::AllReduce<tl::MaxOp, 2, 1, 0, tl::NamedBarrier<64>>::run(amax_fra` |
| store.global#2 | 0.13 | 11.1% | 1.0 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3)` |
| load.global#2*2 (est.: sampled iterations x2) | 0.06 | 5.6% | 1.0 | 0.06 | `amax_fragment[i_1] = cutlass::bfloat16_t(__hmax((bfloat16_t(((nv_bfloat162*)(&(amax_fragme` |
| store.global#3*2 (est.: sampled iterations x2) | 0.06 | 5.6% | 1.0 | 0.06 | `*(fp8_e4_16_t*)(out + ((((((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int64_t)10240) +` |
| store.global*2 (est.: sampled iterations x2) | 0.06 | 5.6% | 1.0 | 0.06 | `out_sf[((((((int64_t)((int)blockIdx.x)) % (int64_t)15) * (int64_t)32) + (((int64_t)((int)t` |
| wait.pdl | 0.03 | 2.8% | 1.0 | 0.03 | `cudaGridDependencySynchronize();` |

### finish_argmax_kernel grid (1, 1, 1) block (32, 1, 1)

25 launches, 1 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 0.77 us, warp 0.77 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.45 | 58.3% | 0.47 | 59.5% |
| compute | 0.13 | 16.7% | 0.11 | 14.4% |
| load | 0.13 | 16.7% | 0.13 | 16.9% |
| store | 0.06 | 8.3% | 0.07 | 9.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.reduce | 0.13 | 16.7% | 1.0 | 0.13 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| load.global | 0.13 | 16.7% | 1.0 | 0.13 | `for (int k = 0; k < 2; ++k) \| if ((((int)threadIdx.x) < parts))` |
| store.global | 0.06 | 8.3% | 1.0 | 0.06 | `if (((int)threadIdx.x) == 0)` |

### activation_swiglu_clamp_kernel grid (1, 5, 1) block (128, 1, 1)

15 launches, 20 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 1.06 us, warp 1.06 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.93 | 87.9% | 1.16 | 89.6% |
| load | 0.06 | 6.1% | 0.05 | 4.1% |
| store | 0.03 | 3.0% | 0.05 | 3.6% |
| wait | 0.03 | 3.0% | 0.03 | 2.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.06 | 6.1% | 1.0 | 0.06 | `*(uint4*)(gate + 0) = *(uint4*)(X + ((((int64_t)((int)blockIdx.y)) * (int64_t)1152) + (((i` |
| store.global | 0.03 | 3.0% | 1.0 | 0.03 | `*(uint4*)(Y + ((((int64_t)((int)blockIdx.y)) * (int64_t)576) + (((int64_t)((int)threadIdx.` |
| wait.pdl | 0.03 | 3.0% | 1.0 | 0.03 | `cudaGridDependencySynchronize();` |

### metadata_chunk_kernel grid (6, 1, 1) block (128, 1, 1)

10 launches, 24 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 1.95 us, warp 1.92 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.96 | 50.0% | 0.99 | 57.8% |
| unattributed | 0.82 | 42.5% | 0.56 | 32.8% |
| store | 0.11 | 5.8% | 0.13 | 7.5% |
| wait | 0.03 | 1.7% | 0.03 | 1.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.67 | 35.0% | 1.0 | 0.67 | `if ((((offset + ((int)blockIdx.x)) < src_rows) && (0 <= (offset + ((int)blockIdx.x)))))` |
| load.global#2 | 0.29 | 15.0% | 1.0 | 0.29 | `if (((0 <= req) & ((int64_t)0 <= pos)))` |
| store.global | 0.05 | 2.5% | 1.0 | 0.05 | `swa[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = con` |
| store.global#2 | 0.03 | 1.7% | 1.0 | 0.03 | `lengths[((int64_t)((int)blockIdx.x))] = ((int)max(info[1], (int64_t)0));` |
| store.global#3 | 0.03 | 1.7% | 1.0 | 0.03 | `top_lengths[((int64_t)((int)blockIdx.x))] = condval_12; \| top_lengths[((int64_t)((int)bloc` |
| wait.pdl | 0.03 | 1.7% | 1.0 | 0.03 | `cudaGridDependencySynchronize();` |

### topk_log_softmax_kernel grid (6, 1, 1) block (1024, 1, 1)

1 launches, 192 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 16.00 us, warp 15.84 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 13.34 | 84.2% | 13.34 | 84.2% |
| unattributed, with loops' unsampled iterations | 1.41 | 8.9% | 1.41 | 8.9% |
| compute | 0.77 | 4.8% | 0.77 | 4.8% |
| wait | 0.29 | 1.8% | 0.29 | 1.8% |
| store | 0.03 | 0.2% | 0.03 | 0.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 6.53 | 41.2% | 1.0 | 6.53 | `if (0 < (vocab & 2047)) \| if (0 < (vocab & 8191))` |
| load.global#2 | 6.43 | 40.6% | 1.0 | 6.43 | `for (int step_1 = 0; step_1 < (vocab >> 11); ++step_1) \| for (int step_1 = 0; step_1 < (vo` |
| compute.reduce#2 | 0.64 | 4.0% | 1.0 | 0.64 | `se[0] = vllm_add(se[0], __shfl_xor_sync((uint)4294967295, se[0], 16, 32));` |
| load.global#3 (sampled iterations) | 0.38 | 2.4% | 1.0 | 0.38 | `float value_32 = vllm_sub(vllm_shifted(logits[((((int64_t)((int)blockIdx.x)) * logits_stri` |
| wait.sync | 0.16 | 1.0% | 1.0 | 0.16 | `__syncthreads();` |
| compute.reduce | 0.13 | 0.8% | 1.0 | 0.13 | `best[0] = vllm_max(best[0], __shfl_xor_sync((uint)4294967295, best[0], 16, 32));` |
| wait.sync#2 | 0.13 | 0.8% | 1.0 | 0.13 | `__syncthreads();` |
| store.global (sampled iterations) | 0.03 | 0.2% | 1.0 | 0.03 | `output[(((((int64_t)c) * (int64_t)1024) + (((int64_t)((int)blockIdx.x)) * ((int64_t)topk))` |

### verify_kernel grid (1, 1, 1) block (96, 1, 1)

5 launches, 3 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 2.78 us, warp 2.78 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.06 | 37.9% | 1.11 | 38.5% |
| unattributed | 0.77 | 27.6% | 0.78 | 27.2% |
| wait | 0.42 | 14.9% | 0.42 | 14.5% |
| store | 0.29 | 10.3% | 0.30 | 10.5% |
| compute | 0.26 | 9.2% | 0.27 | 9.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.99 | 35.6% | 1.0 | 0.99 | `start = cu_num_logits[((int64_t)((int)blockIdx.x))];` |
| wait.sync | 0.42 | 14.9% | 1.0 | 0.42 | `__syncthreads();` |
| store.global | 0.29 | 10.3% | 1.0 | 0.29 | `if (((int)threadIdx.x) == 0)` |
| compute.reduce | 0.26 | 9.2% | 1.0 | 0.26 | `row_max[0] = vllm_max(row_max[0], __shfl_xor_sync((uint)4294967295, row_max[0], 8, 32));` |
| load.global#2 | 0.06 | 2.3% | 1.0 | 0.06 | `target = local_argmax[((((int64_t)row) * argmax_stride) + ((int64_t)best_index[0]))];` |

### sink_cut_kernel grid (1, 1, 1) block (128, 1, 1)

10 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 1.41 us, warp 1.41 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.53 | 37.5% | 0.53 | 37.5% |
| store | 0.50 | 35.2% | 0.50 | 35.0% |
| unattributed | 0.38 | 27.3% | 0.39 | 27.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.53 | 37.5% | 0.5 | 0.53 | `for (int b = 0; b < num_reqs; ++b)` |
| store.global | 0.50 | 35.2% | 1.0 | 0.50 | `if (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) < num_cut) \| saved[((((int64_t)((int)` |

### resample_kernel grid (1, 127, 1) block (128, 1, 1)

5 launches, 508 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 2.43 us, warp 2.43 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.63 | 67.1% | 2.09 | 75.2% |
| unattributed | 0.77 | 31.6% | 0.65 | 23.4% |
| compute | 0.03 | 1.3% | 0.04 | 1.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.63 | 67.1% | 1.0 | 1.63 | `end = cu_num_logits[(((int64_t)((int)blockIdx.x)) + (int64_t)1)];` |
| compute.reduce | 0.03 | 1.3% | 1.0 | 0.03 | `if ((temp != 0x0p+0f/*0.000000e+00*/) \| bonus)` |

### local_stats_kernel grid (6, 16, 1) block (128, 1, 1)

1 launches, 384 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 13.86 us, warp 13.41 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 12.19 | 90.9% | 12.19 | 90.9% |
| unattributed | 0.99 | 7.4% | 0.99 | 7.4% |
| compute | 0.16 | 1.2% | 0.16 | 1.2% |
| store | 0.06 | 0.5% | 0.06 | 0.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 12.19 | 90.9% | 1.0 | 12.19 | `if (expanded_local_pos[((int64_t)((int)blockIdx.x))] < num_speculative_steps)` |
| compute.reduce | 0.16 | 1.2% | 1.0 | 0.16 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| store.global | 0.06 | 0.5% | 1.0 | 0.06 | `if (((int)threadIdx.x) == 0)` |

### rope_kernel grid (5, 1, 1) block (128, 1, 1)

15 launches, 20 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 0.86 us, warp 0.86 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.58 | 66.7% | 0.60 | 69.2% |
| load | 0.19 | 22.2% | 0.18 | 20.8% |
| wait | 0.06 | 7.4% | 0.05 | 5.7% |
| store | 0.03 | 3.7% | 0.04 | 4.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.19 | 22.2% | 1.0 | 0.19 | `if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < tokens)) \| int64_t position ` |
| wait.pdl | 0.06 | 7.4% | 1.0 | 0.06 | `cudaGridDependencySynchronize();` |
| store.global | 0.03 | 3.7% | 1.0 | 0.03 | `*(uint4*)(x + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + ((((int64_t)((int)thread` |

### sink_policy_kernel grid (1, 1, 1) block (32, 1, 1)

5 launches, 1 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 2.56 us, warp 2.56 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.09 | 42.5% | 1.11 | 43.5% |
| unattributed | 0.96 | 37.5% | 0.93 | 36.2% |
| store | 0.51 | 20.0% | 0.52 | 20.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 | 0.64 | 25.0% | 1.0 | 0.64 | `state[0] = idx_mapping[((int64_t)((int)blockIdx.x))];` |
| store.global#3 | 0.45 | 17.5% | 1.0 | 0.45 | `for (int a = 0; a < 3; ++a) \| recovered_logprobs[((int64_t)((int)blockIdx.x))] = 0x0p+0f/*` |
| load.global | 0.22 | 8.8% | 1.0 | 0.22 | `start[0] = cu_num_logits[((int64_t)((int)blockIdx.x))];` |
| load.global#4 | 0.16 | 6.2% | 1.0 | 0.16 | `for (int r = 0; r < 2; ++r) \| for (int r_2 = 0; r_2 < 8; ++r_2)` |
| load.global#5 | 0.06 | 2.5% | 1.0 | 0.06 | `AtomicAdd((&(counts[0])), draft_zero_flag[0]);` |
| store.global#4 | 0.06 | 2.5% | 1.0 | 0.06 | `events[(((int64_t)((int)blockIdx.x)) * (int64_t)6)] = outcome[0];` |

### cache_writer_kernel grid (5, 1, 1) block (128, 1, 1)

12 launches, 20 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 1.02 us, warp 0.96 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.45 | 46.7% | 0.47 | 48.0% |
| load | 0.26 | 26.7% | 0.24 | 24.1% |
| store | 0.16 | 16.7% | 0.16 | 16.5% |
| compute | 0.06 | 6.7% | 0.07 | 7.6% |
| wait | 0.03 | 3.3% | 0.04 | 3.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.19 | 20.0% | 1.0 | 0.19 | `for (int i = 0; i < 4; ++i)` |
| store.global | 0.16 | 16.7% | 1.0 | 0.16 | `if (((int64_t)0 <= slot) & ((slot >> (int64_t)7) < ((int64_t)pages))) \| if (((int64_t)0 <=` |
| compute.reduce | 0.06 | 6.7% | 1.0 | 0.06 | `amax = max(amax, __shfl_xor_sync((uint)4294967295, amax, 1, 32));` |
| load.global#2 | 0.06 | 6.7% | 1.0 | 0.06 | `int64_t slot = slots[((int64_t)((int)blockIdx.x))];` |
| wait.pdl | 0.03 | 3.3% | 1.0 | 0.03 | `cudaGridDependencySynchronize();` |

### compute_slot_mappings_kernel grid (1, 17, 1) block (256, 1, 1)

5 launches, 136 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 2.34 us, warp 2.34 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.57 | 67.1% | 1.58 | 67.3% |
| load | 0.77 | 32.9% | 0.77 | 32.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.77 | 32.9% | 1.0 | 0.77 | `int first_token = query_start_loc[0];` |

### post_update_kernel grid (1, 1, 1) block (32, 1, 1)

5 launches, 1 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 1.95 us, warp 1.95 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.70 | 36.1% | 0.70 | 30.0% |
| load | 0.64 | 32.8% | 0.65 | 28.1% |
| store | 0.61 | 31.1% | 0.97 | 41.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 | 0.48 | 24.6% | 1.0 | 0.48 | `int count = num_sampled[((((int64_t)((int)blockIdx.x)) * (int64_t)32) + ((int64_t)((int)th` |
| store.global#2 | 0.45 | 23.0% | 1.0 | 0.45 | `if (0 < count)` |
| store.global | 0.16 | 8.2% | 1.0 | 0.16 | `if (0 < count)` |
| load.global#3 | 0.10 | 4.9% | 1.0 | 0.10 | `if ((0 < count))` |
| load.global | 0.06 | 3.3% | 1.0 | 0.06 | `int64_t req = idx_mapping[((((int64_t)((int)blockIdx.x)) * (int64_t)32) + ((int64_t)((int)` |

### sink_rows_kernel grid (1, 64, 1) block (256, 1, 1)

5 launches, 512 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 1.70 us, warp 1.70 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 0.77 | 45.3% | 0.70 | 31.5% |
| unattributed | 0.54 | 32.1% | 0.53 | 23.8% |
| load | 0.38 | 22.6% | 1.00 | 44.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#2 | 0.51 | 30.2% | 1.0 | 0.51 | `tokens[((((int64_t)((int)blockIdx.x)) * (int64_t)3) + ((int64_t)((int)threadIdx.x)))] = to` |
| load.global | 0.29 | 17.0% | 1.0 | 0.29 | `row = (max((((int64_t)num_sampled[((int64_t)((int)blockIdx.x))]) - (int64_t)1), (int64_t)0` |
| store.global | 0.26 | 15.1% | 1.0 | 0.26 | `for (int k_1 = 0; k_1 < 8; ++k_1)` |
| load.global#2 | 0.10 | 5.7% | 1.0 | 0.10 | `token = max(alternates[((((int64_t)((int)blockIdx.x)) * (int64_t)3) + ((int64_t)((int)thre` |

### sink_flags_kernel grid (1, 1, 1) block (128, 1, 1)

5 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 2.08 us, warp 2.08 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.28 | 61.5% | 1.35 | 63.6% |
| unattributed | 0.45 | 21.5% | 0.42 | 19.6% |
| store | 0.35 | 16.9% | 0.36 | 16.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.28 | 61.5% | 1.0 | 1.28 | `token = logits_indices[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)th` |
| store.global | 0.35 | 16.9% | 1.0 | 0.35 | `flags[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = (` |

### dspark_confidence_kernel grid (5, 1, 1) block (128, 1, 1)

5 launches, 20 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 1.92 us, warp 1.92 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.50 | 78.3% | 1.38 | 72.5% |
| load | 0.16 | 8.3% | 0.17 | 8.7% |
| compute | 0.10 | 5.0% | 0.10 | 5.0% |
| store | 0.10 | 5.0% | 0.10 | 5.0% |
| wait | 0.06 | 3.3% | 0.17 | 8.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.13 | 6.7% | 1.0 | 0.13 | `for (int c = 0; c < 5; ++c)` |
| compute.reduce | 0.10 | 5.0% | 1.0 | 0.10 | `acc[0] = tl::warp_reduce_sum(acc[0]);` |
| store.global | 0.10 | 5.0% | 1.0 | 0.10 | `out[((int64_t)((int)blockIdx.x))] = (0x1p+0f/*1.000000e+00*/ / (0x1p+0f/*1.000000e+00*/ + ` |
| wait.sync | 0.06 | 3.3% | 1.0 | 0.06 | `__syncthreads();` |
| load.global#2 | 0.03 | 1.7% | 1.0 | 0.03 | `total[0] = 0x0p+0f/*0.000000e+00*/;` |

### adaptive_offsets_kernel grid (1, 1, 1) block (256, 1, 1)

4 launches, 8 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 2.27 us, warp 2.27 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed, with loops' unsampled iterations | 0.72 | 31.7% | 0.71 | 31.3% |
| load | 0.59 | 26.1% | 0.59 | 26.1% |
| wait | 0.50 | 21.8% | 0.50 | 21.8% |
| store | 0.46 | 20.4% | 0.47 | 20.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 (sampled iterations) | 0.46 | 20.4% | 1.0 | 0.46 | `if ((((chunk * 256) + ((int)threadIdx.x)) < num_reqs))` |
| wait.sync (sampled iterations) | 0.43 | 19.0% | 1.0 | 0.43 | `__syncthreads();` |
| store.global#5 | 0.14 | 6.3% | 1.0 | 0.14 | `for (int chunk_1 = 0; chunk_1 < ((starts_len + 255) >> 8); ++chunk_1)` |
| store.global#4 (sampled iterations) | 0.13 | 5.6% | 1.0 | 0.13 | `for (int j = 0; j < 6; ++j)` |
| store.global | 0.10 | 4.2% | 1.0 | 0.10 | `if (((int)threadIdx.x) == 0)` |
| load.global (sampled iterations) | 0.08 | 3.5% | 1.0 | 0.08 | `if (((((chunk * 256) + ((int)threadIdx.x)) < num_reqs) & (drafted != 0)))` |
| store.global#2 (sampled iterations) | 0.06 | 2.8% | 1.0 | 0.06 | `if ((((chunk * 256) + ((int)threadIdx.x)) < num_reqs) & (drafted == 0))` |
| wait.sync#2 (sampled iterations) | 0.06 | 2.8% | 1.0 | 0.06 | `__syncthreads();` |
| load.global#3 (sampled iterations) | 0.05 | 2.1% | 1.0 | 0.05 | `state = idx_mapping[((((int64_t)chunk) * (int64_t)256) + ((int64_t)((int)threadIdx.x)))];` |
| store.global#3 (sampled iterations) | 0.03 | 1.4% | 1.0 | 0.03 | `cu_num_logits[(((((int64_t)chunk) * (int64_t)256) + ((int64_t)((int)threadIdx.x))) + (int6` |

### gather_block_tables_kernel grid (17, 1, 1) block (256, 1, 1)

5 launches, 136 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 1.70 us, warp 1.70 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.38 | 81.1% | 1.37 | 81.4% |
| unattributed | 0.32 | 18.9% | 0.31 | 18.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.38 | 81.1% | 1.0 | 1.38 | `int g = chunk_groups[((int64_t)((int)blockIdx.x))];` |

### ced_copy_rows_kernel grid (25, 1, 1) block (128, 1, 1)

9 launches, 100 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 0.93 us, warp 0.93 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.51 | 55.2% | 0.47 | 54.3% |
| unattributed | 0.42 | 44.8% | 0.40 | 45.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.51 | 55.2% | 1.0 | 0.51 | `for (int k = 0; k < 4; ++k) \| int64_t index = ced_ldg64((&(row_indices[((((int64_t)blockId` |

### token_ranks_kernel grid (6, 1, 1) block (256, 1, 1)

1 launches, 48 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 7.68 us, warp 7.65 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 6.69 | 87.4% | 6.69 | 87.4% |
| compute | 0.51 | 6.7% | 0.51 | 6.7% |
| unattributed | 0.38 | 5.0% | 0.38 | 5.0% |
| store | 0.06 | 0.8% | 0.06 | 0.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 6.69 | 87.4% | 1.0 | 6.69 | `if ((((int64_t)0 <= token_ids[((int64_t)((int)blockIdx.x))]) && (token_ids[((int64_t)((int` |
| compute.reduce | 0.51 | 6.7% | 1.0 | 0.51 | `total[0] = tl::AllReduce<tl::SumOp, 256, 1, 0, tl::NamedBarrier<256>>::run(total[0], (&(wo` |
| store.global | 0.06 | 0.8% | 1.0 | 0.06 | `output[((int64_t)((int)blockIdx.x))] = ((int64_t)total[0]);` |

### insert_kernel grid (1, 1, 1) block (32, 1, 1)

5 launches, 1 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 1.50 us, warp 1.50 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.64 | 42.6% | 0.65 | 43.0% |
| unattributed | 0.51 | 34.0% | 0.52 | 34.2% |
| store | 0.32 | 21.3% | 0.31 | 20.3% |
| compute | 0.03 | 2.1% | 0.04 | 2.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#3 | 0.51 | 34.0% | 1.0 | 0.51 | `int state = mapping[((int64_t)position)]; \| int64_t state = mapping[((int64_t)position)];` |
| store.global | 0.32 | 21.3% | 1.0 | 0.32 | `counts[((int64_t)((int)blockIdx.x))] = (count_local + 1);` |
| load.global | 0.10 | 6.4% | 1.0 | 0.10 | `for (int k = 0; k < 4; ++k)` |
| compute.reduce | 0.03 | 2.1% | 1.0 | 0.03 | `int count = __shfl_sync((uint)4294967295, count_local, 0, 32);` |
| load.global#2 | 0.03 | 2.1% | 1.0 | 0.03 | `count_local = counts[((int64_t)((int)blockIdx.x))];` |

### combine_sampled_and_draft_tokens_kernel grid (1, 1, 1) block (32, 1, 1)

5 launches, 1 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 1.50 us, warp 1.50 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.90 | 59.6% | 0.89 | 58.9% |
| unattributed | 0.51 | 34.0% | 0.52 | 34.7% |
| store | 0.10 | 6.4% | 0.10 | 6.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.90 | 59.6% | 1.0 | 0.90 | `if (((((int)blockIdx.x) * 32) + ((int)threadIdx.x)) < num_reqs) \| req = idx_mapping[((int6` |
| store.global | 0.10 | 6.4% | 1.0 | 0.10 | `if ((prefill < seq) && (1 < num_logits))` |

### assign_draft_budget_kernel grid (1, 1, 1) block (256, 1, 1)

4 launches, 8 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 1.68 us, warp 1.68 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed, with loops' unsampled iterations | 0.70 | 41.9% | 0.69 | 41.1% |
| load | 0.69 | 41.0% | 0.69 | 41.1% |
| store | 0.14 | 8.6% | 0.14 | 8.6% |
| wait | 0.14 | 8.6% | 0.15 | 9.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.69 | 41.0% | 1.0 | 0.69 | `for (int chunk = 0; chunk < ((num_reqs + 255) >> 8); ++chunk)` |
| store.global | 0.14 | 8.6% | 1.0 | 0.14 | `for (int chunk_2 = 0; chunk_2 < ((num_reqs + 255) >> 8); ++chunk_2)` |
| wait.sync#2 (sampled iterations) | 0.06 | 3.8% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync | 0.05 | 2.9% | 1.0 | 0.05 | `__syncthreads();` |
| wait.sync#3 | 0.03 | 1.9% | 1.0 | 0.03 | `__syncthreads();` |

### logits_rows_kernel grid (1, 1, 1) block (128, 1, 1)

5 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 1.31 us, warp 1.31 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.48 | 36.6% | 0.47 | 36.5% |
| unattributed | 0.45 | 34.1% | 0.44 | 34.0% |
| store | 0.38 | 29.3% | 0.38 | 29.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.48 | 36.6% | 1.0 | 0.48 | `index = logits_indices[min(((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((in` |
| store.global | 0.38 | 29.3% | 1.0 | 0.38 | `if (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) < num_logits)` |

### per_token_cast_kernel_kernel grid (45, 1, 1) block (64, 1, 1)

5 launches, 90 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 1.25 us, warp 1.25 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.48 | 38.5% | 0.51 | 41.2% |
| store | 0.32 | 25.6% | 0.29 | 23.7% |
| load | 0.29 | 23.1% | 0.29 | 23.2% |
| compute | 0.13 | 10.3% | 0.12 | 9.3% |
| wait | 0.03 | 2.6% | 0.03 | 2.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.22 | 17.9% | 1.0 | 0.22 | `for (int i = 0; i < 2; ++i)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.13 | 10.3% | 1.0 | 0.13 | `amax_fragment[i_1] = tl::AllReduce<tl::MaxOp, 2, 1, 0, tl::NamedBarrier<64>>::run(amax_fra` |
| store.global#2 | 0.13 | 10.3% | 1.0 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3)` |
| store.global*2 (est.: sampled iterations x2) | 0.13 | 10.3% | 1.0 | 0.13 | `out_sf[((((((int64_t)((int)blockIdx.x)) % (int64_t)15) * (int64_t)32) + (((int64_t)((int)t` |
| load.global#2*2 (est.: sampled iterations x2) | 0.06 | 5.1% | 1.0 | 0.06 | `amax_fragment[i_1] = cutlass::bfloat16_t(__hmax((bfloat16_t(((nv_bfloat162*)(&(amax_fragme` |
| store.global#3*2 (est.: sampled iterations x2) | 0.06 | 5.1% | 1.0 | 0.06 | `*(fp8_e4_16_t*)(out + ((((((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int64_t)10240) +` |
| wait.pdl | 0.03 | 2.6% | 1.0 | 0.03 | `cudaGridDependencySynchronize();` |

### token_counts_kernel grid (1, 1, 1) block (128, 1, 1)

5 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 1.22 us, warp 1.22 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 0.58 | 47.4% | 0.58 | 46.9% |
| unattributed | 0.42 | 34.2% | 0.43 | 34.9% |
| load | 0.22 | 18.4% | 0.22 | 18.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.58 | 47.4% | 1.0 | 0.58 | `if (((int)threadIdx.x) == 0) \| sampled[((int64_t)((int)blockIdx.x))] = (((int)(prefill_len` |
| load.global | 0.22 | 18.4% | 1.0 | 0.22 | `int count = sampled[((int64_t)((int)blockIdx.x))];` |

### mhc_collapse_column_kernel grid (30, 1, 1) block (128, 1, 1)

5 launches, 120 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 1.22 us, warp 1.18 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.80 | 67.6% | 0.79 | 66.7% |
| wait | 0.26 | 21.6% | 0.26 | 22.0% |
| load | 0.06 | 5.4% | 0.08 | 6.5% |
| store | 0.06 | 5.4% | 0.06 | 4.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.pdl | 0.26 | 21.6% | 1.0 | 0.26 | `cudaGridDependencySynchronize();` |
| load.global | 0.06 | 5.4% | 1.0 | 0.06 | `*(float4*)(mv + 0) = *(float4*)(mix + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int6` |
| store.global | 0.06 | 5.4% | 1.0 | 0.06 | `*(uint4*)(out + ((((int64_t)((int)blockIdx.x)) * (int64_t)1024) + (((int64_t)((int)threadI` |

### split_columns_kernel grid (5, 15, 1) block (128, 1, 1)

4 launches, 300 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 1.50 us, warp 1.33 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.91 | 68.7% | 0.92 | 68.9% |
| unattributed | 0.42 | 31.3% | 0.42 | 31.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.91 | 68.7% | 1.0 | 0.91 | `if (((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 5)) < 5) \| if (((int)blockIdx.y) < 5` |

### flatten_sampled_kernel grid (1, 1, 1) block (128, 1, 1)

5 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 1.15 us, warp 1.15 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.58 | 50.0% | 0.59 | 50.8% |
| unattributed | 0.48 | 41.7% | 0.47 | 40.9% |
| store | 0.10 | 8.3% | 0.10 | 8.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.35 | 30.6% | 1.0 | 0.35 | `for (int r = 0; r < num_reqs; ++r)` |
| load.global#2 | 0.22 | 19.4% | 1.0 | 0.22 | `int i = (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) - cu_num_logits[req]);` |
| store.global | 0.10 | 8.3% | 1.0 | 0.10 | `flat[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = co` |

### mhc_post_kernel grid (25, 1, 1) block (128, 1, 1)

5 launches, 100 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 1.09 us, warp 0.99 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.83 | 83.9% | 0.87 | 84.0% |
| load | 0.06 | 6.5% | 0.07 | 6.8% |
| store | 0.06 | 6.5% | 0.06 | 5.6% |
| wait | 0.03 | 3.2% | 0.04 | 3.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.06 | 6.5% | 1.0 | 0.06 | `*(float4*)(pm + 0) = *(float4*)(post + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int` |
| store.global | 0.06 | 6.5% | 1.0 | 0.06 | `for (int n_1 = 0; n_1 < 4; ++n_1)` |
| wait.pdl | 0.03 | 3.2% | 1.0 | 0.03 | `cudaGridDependencySynchronize();` |

### prepare_pos_seq_lens_kernel grid (2, 1, 1) block (256, 1, 1)

5 launches, 16 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 1.06 us, warp 1.06 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.54 | 51.5% | 0.56 | 53.0% |
| unattributed | 0.42 | 39.4% | 0.39 | 37.2% |
| store | 0.10 | 9.1% | 0.10 | 9.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.54 | 51.5% | 1.0 | 0.54 | `int computed = num_computed_tokens[idx_mapping[((int64_t)((int)blockIdx.x))]];` |
| store.global#2 | 0.10 | 9.1% | 1.0 | 0.10 | `if (((int)threadIdx.x) == 0)` |

### scatter_rows_gather_kernel grid (1, 1, 1) block (128, 1, 1)

5 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 0.99 us, warp 0.99 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.48 | 48.4% | 0.51 | 50.0% |
| store | 0.42 | 41.9% | 0.42 | 41.8% |
| load | 0.10 | 9.7% | 0.08 | 8.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.42 | 41.9% | 1.0 | 0.42 | `if (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) < (num_reqs * width))` |
| load.global | 0.10 | 9.7% | 1.0 | 0.10 | `index = idx_mapping[min((((((int64_t)0 <= ((int64_t)width)) && ((int64_t)0 <= rmod_1)) \|\| ` |

### ced_requests_kernel grid (1, 1, 1) block (128, 1, 1)

5 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 0.96 us, warp 0.96 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 0.45 | 46.7% | 0.45 | 45.2% |
| unattributed | 0.42 | 43.3% | 0.46 | 45.9% |
| load | 0.10 | 10.0% | 0.09 | 8.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.45 | 46.7% | 1.0 | 0.45 | `if (((int)threadIdx.x) < 16) \| if (((int)threadIdx.x) <= 16)` |
| load.global | 0.10 | 10.0% | 1.0 | 0.10 | `uint v_ = tl::load_global_32((&(starts[1]))); \| uint v_ = tl::load_global_32((&(starts[min` |

### mhc_collapse_kernel grid (25, 1, 1) block (128, 1, 1)

5 launches, 100 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 0.96 us, warp 0.90 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.74 | 82.1% | 0.80 | 83.9% |
| load | 0.06 | 7.1% | 0.06 | 6.7% |
| store | 0.06 | 7.1% | 0.05 | 5.4% |
| wait | 0.03 | 3.6% | 0.04 | 4.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.06 | 7.1% | 1.0 | 0.06 | `*(float4*)(mv + 0) = *(float4*)(mix + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int6` |
| store.global | 0.06 | 7.1% | 1.0 | 0.06 | `*(uint4*)(out + ((((int64_t)((int)blockIdx.x)) * (int64_t)1024) + (((int64_t)((int)threadI` |
| wait.pdl | 0.03 | 3.6% | 1.0 | 0.03 | `cudaGridDependencySynchronize();` |

### scatter_rows_kernel grid (1, 1, 1) block (128, 1, 1)

5 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 0.96 us, warp 0.96 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.48 | 50.0% | 0.48 | 49.7% |
| store | 0.38 | 40.0% | 0.40 | 41.1% |
| load | 0.10 | 10.0% | 0.09 | 9.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.38 | 40.0% | 1.0 | 0.38 | `if (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) < (num_reqs * width))` |
| load.global | 0.10 | 10.0% | 1.0 | 0.10 | `index = idx_mapping[min((((((int64_t)0 <= ((int64_t)width)) && ((int64_t)0 <= rmod_1)) \|\| ` |

### padding_rows_kernel grid (1, 1, 1) block (256, 1, 1)

9 launches, 8 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 0.42 us, warp 0.42 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.32 | 76.9% | 0.33 | 80.2% |
| store | 0.10 | 23.1% | 0.08 | 19.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.10 | 23.1% | 1.0 | 0.10 | `if (((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) < rows)` |

### gemm_splitk_reduce_kernel grid (4, 6, 1) block (128, 1, 1)

3 launches, 96 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 0.64 us, warp 0.64 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.35 | 55.0% | 0.37 | 56.5% |
| load | 0.19 | 30.0% | 0.19 | 29.0% |
| wait | 0.06 | 10.0% | 0.05 | 8.1% |
| store | 0.03 | 5.0% | 0.04 | 6.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.19 | 30.0% | 1.0 | 0.19 | `acc[0] = P[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x))` |
| wait.pdl | 0.06 | 10.0% | 1.0 | 0.06 | `cudaGridDependencySynchronize();` |
| store.global | 0.03 | 5.0% | 1.0 | 0.03 | `C[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x)) * (int64` |

### split_columns_kernel grid (6, 15, 1) block (128, 1, 1)

1 launches, 360 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 1.82 us, warp 1.57 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.12 | 71.4% | 1.12 | 71.4% |
| unattributed | 0.45 | 28.6% | 0.45 | 28.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.12 | 71.4% | 1.0 | 1.12 | `if (((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 5)) < 5) \| if (((int)blockIdx.y) < 5` |

### ced_copy_rows_kernel grid (30, 1, 1) block (128, 1, 1)

1 launches, 120 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 1.34 us, warp 1.31 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.93 | 70.7% | 0.93 | 70.7% |
| unattributed | 0.38 | 29.3% | 0.38 | 29.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.93 | 70.7% | 1.0 | 0.93 | `for (int k = 0; k < 4; ++k) \| int64_t index = ced_ldg64((&(row_indices[((((int64_t)blockId` |
