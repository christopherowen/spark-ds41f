=== timeline and tables
trace /c/iket/step-tp4-c8-20261026T162621Z/trace/iket/pid_0x18dd/iket.decoded_results.json: 16677 launches, graphLaunches dict of 90

## Timeline: 7547 launches in 1296514.3 us; the GPU is busy 337638.9 us (26%) and idle 958875.4 us in 5922 gaps; gaps over 200 us split it into 1055 segments

The 15 of 1055 segments with the most busy time:

| segment | launches | wall us | busy us | busy | first kernel |
|---:|---:|---:|---:|---:|---|
| 5 | 118 | 17384.7 | 8120.1 | 47% | per_token_cast_kernel_kernel |
| 17 | 12 | 7203.2 | 5903.6 | 82% | moe_dispatch_kernel |
| 22 | 38 | 9349.1 | 6316.8 | 68% | mhc_project_streams_kernel |
| 24 | 269 | 9079.0 | 5942.6 | 65% | finish_argmax_kernel |
| 47 | 35 | 7416.0 | 5495.0 | 74% | moe_combine_kernel |
| 55 | 261 | 5983.0 | 4988.0 | 83% | finish_argmax_kernel |
| 240 | 255 | 5850.7 | 4980.6 | 85% | adaptive_offsets_kernel |
| 471 | 263 | 7738.5 | 5673.1 | 73% | gemm_bf16_kernel |
| 513 | 31 | 7274.8 | 5296.7 | 73% | norm_forward_and_per_token_cast_kernel_kernel |
| 516 | 258 | 6762.2 | 4997.1 | 74% | scatter_rows_kernel |
| 856 | 261 | 6702.0 | 4978.4 | 74% | finish_argmax_kernel |
| 881 | 93 | 11822.3 | 6458.3 | 55% | padding_rows_kernel |
| 945 | 257 | 6308.3 | 5002.3 | 79% | padding_rows_kernel |
| 996 | 93 | 12322.2 | 7156.5 | 58% | mhc_project_streams_kernel |
| 1008 | 255 | 6197.7 | 4988.6 | 80% | adaptive_offsets_kernel |

| idle us | after | before |
|---:|---|---|
| 99506.1 | resample_kernel | insert_kernel |
| 75685.5 | scatter_rows_kernel | padding_rows_kernel |
| 11393.0 | norm_forward_and_per_token_cast_kernel_kernel | sink_flags_kernel |
| 11280.6 | norm_forward_and_per_token_cast_kernel_kernel | sink_flags_kernel |
| 11273.9 | norm_forward_and_per_token_cast_kernel_kernel | sink_flags_kernel |
| 11036.4 | norm_forward_and_per_token_cast_kernel_kernel | sink_flags_kernel |
| 11027.7 | norm_forward_and_per_token_cast_kernel_kernel | sink_flags_kernel |
| 10879.6 | norm_forward_and_per_token_cast_kernel_kernel | sink_flags_kernel |
| 10875.0 | norm_forward_and_per_token_cast_kernel_kernel | sink_flags_kernel |
| 10761.3 | norm_forward_and_per_token_cast_kernel_kernel | sink_flags_kernel |

## Engine steps

| step | kind | launches | wall us | busy us | busy | most span |
|---:|---|---:|---:|---:|---:|---|
| 0 | prefill | 410 | 70889.2 | 29419.8 | 42% | vocab_packed_projection_kernel 9671, moe_grouped_fp8_fp4_gemm_kernel 5560, gemm_mxfp8_kernel 4701 |
| 1 | decode | 420 | 146237.7 | 19026.0 | 13% | vocab_packed_projection_kernel 9409, gemm_mxfp8_kernel 2207, moe_grouped_fp8_fp4_gemm_kernel 2159 |
| 2 | decode | 420 | 130988.8 | 18095.4 | 14% | vocab_packed_projection_kernel 8822, moe_grouped_fp8_fp4_gemm_kernel 2382, gemm_mxfp8_kernel 2232 |
| 3 | decode | 420 | 54899.2 | 17938.8 | 33% | vocab_packed_projection_kernel 8687, moe_grouped_fp8_fp4_gemm_kernel 2386, gemm_mxfp8_kernel 2218 |
| 4 | decode | 420 | 62855.0 | 18088.2 | 29% | vocab_packed_projection_kernel 8733, moe_grouped_fp8_fp4_gemm_kernel 2310, gemm_mxfp8_kernel 2245 |
| 5 | decode | 420 | 58850.8 | 18070.8 | 31% | vocab_packed_projection_kernel 8853, moe_grouped_fp8_fp4_gemm_kernel 2320, gemm_mxfp8_kernel 2160 |
| 6 | decode | 420 | 63378.9 | 17934.0 | 28% | vocab_packed_projection_kernel 8756, moe_grouped_fp8_fp4_gemm_kernel 2305, gemm_mxfp8_kernel 2138 |
| 7 | decode | 420 | 71555.7 | 17998.4 | 25% | vocab_packed_projection_kernel 8748, moe_grouped_fp8_fp4_gemm_kernel 2333, gemm_mxfp8_kernel 2170 |
| 8 | decode | 420 | 55628.6 | 18428.4 | 33% | vocab_packed_projection_kernel 8839, moe_grouped_fp8_fp4_gemm_kernel 2396, gemm_mxfp8_kernel 2242 |
| 9 | decode | 420 | 78823.1 | 18055.2 | 23% | vocab_packed_projection_kernel 8780, moe_grouped_fp8_fp4_gemm_kernel 2312, gemm_mxfp8_kernel 2132 |
| 10 | decode | 420 | 86938.1 | 17877.4 | 21% | vocab_packed_projection_kernel 8722, moe_grouped_fp8_fp4_gemm_kernel 2297, gemm_mxfp8_kernel 2131 |
| 11 | decode | 420 | 63103.7 | 18160.7 | 29% | vocab_packed_projection_kernel 8932, moe_grouped_fp8_fp4_gemm_kernel 2319, gemm_mxfp8_kernel 2137 |
| 12 | decode | 420 | 69081.4 | 17972.5 | 26% | vocab_packed_projection_kernel 8700, moe_grouped_fp8_fp4_gemm_kernel 2312, gemm_mxfp8_kernel 2200 |
| 13 | decode | 420 | 52777.4 | 17930.2 | 34% | vocab_packed_projection_kernel 8688, moe_grouped_fp8_fp4_gemm_kernel 2320, gemm_mxfp8_kernel 2189 |
| 14 | decode | 420 | 63201.8 | 18039.1 | 29% | vocab_packed_projection_kernel 8756, moe_grouped_fp8_fp4_gemm_kernel 2312, gemm_mxfp8_kernel 2172 |
| 15 | decode | 420 | 53721.4 | 17883.3 | 33% | vocab_packed_projection_kernel 8661, moe_grouped_fp8_fp4_gemm_kernel 2312, gemm_mxfp8_kernel 2117 |
| 16 | decode | 420 | 50200.3 | 18393.6 | 37% | vocab_packed_projection_kernel 9155, moe_grouped_fp8_fp4_gemm_kernel 2301, gemm_mxfp8_kernel 2139 |
| 17 | decode | 417 | 60539.6 | 18327.2 | 30% | vocab_packed_projection_kernel 9012, moe_grouped_fp8_fp4_gemm_kernel 2314, gemm_mxfp8_kernel 2191 |

## Overview over 17 decode steps: 7137 launches of 127 kernel shapes, spans summing to 333161.6 us (19597.7 a step)

| kernel | grid | launches a step | span us a step | share | span us a launch | last warp: wait | load | compute | store | unattributed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vocab_packed_projection_kernel | (1, 4040, 1) | 1.05882 | 4965.3 | 25.3% | 4656.96 | 38% | 26% | 24% | 0% | 11% |
| vocab_packed_projection_kernel | (1, 4040, 1) | 0.941176 | 3873.1 | 19.8% | 4078.93 | 54% | 33% | 10% | 0% | 3% |
| gemm_bf16_kernel | (2020, 1, 1) | 5 | 1609.1 | 8.2% | 326.18 | 91% | 0% | 4% | 1% | 4% |
| l2_prefetch_kernel | (2, 1, 1) | 17 | 1295.5 | 6.6% | 69.79 | 0% | 100% | 0% | 0% | 0% |
| moe_grouped_fp8_fp4_gemm_kernel | (40, 288, 1) | 6 | 1028.1 | 5.2% | 170.05 | 8% | 56% | 0% | 0% | 36% |
| mhc_project_streams_kernel | (40, 3, 1) | 18 | 838.6 | 4.3% | 52.38 | 8% | 36% | 18% | 4% | 35% |
| moe_grouped_fp8_fp4_gemm_kernel | (9, 288, 1) | 6 | 803.2 | 4.1% | 133.86 | 12% | 15% | 0% | 0% | 73% |
| gemm_mxfp8_kernel | (80, 1, 1) | 10 | 570.2 | 2.9% | 16.34 | 72% | 5% | 7% | 1% | 15% |
| gemm_mxfp8_kernel | (128, 1, 1) | 9 | 461.7 | 2.4% | 40.64 | 113% | 5% | 17% | 1% | 0% |
| gemm_fp8w_bf16_partials_kernel | (32, 4, 1) | 9 | 410.5 | 2.1% | 58.53 | 30% | 12% | 13% | 0% | 45% |
| gemm_mxfp8_kernel | (28, 1, 1) | 9 | 407.8 | 2.1% | 43.10 | 44% | 9% | 24% | 0% | 24% |
| gemm_mxfp8_kernel | (40, 1, 1) | 9 | 405.5 | 2.1% | 44.77 | 52% | 6% | 19% | 2% | 21% |
| gemm_mxfp8_kernel | (18, 1, 1) | 9 | 282.1 | 1.4% | 26.94 | 42% | 13% | 35% | 1% | 10% |
| sparse_mla_kernel | (1, 48, 1) | 4 | 275.5 | 1.4% | 61.50 | 2% | 8% | 2% | 0% | 87% |
| moe_grouped_fp8_fp4_gemm_kernel | (9, 120, 1) | 3 | 244.0 | 1.2% | 81.34 | 12% | 14% | 0% | 0% | 73% |
| moe_grouped_fp8_fp4_gemm_kernel | (40, 120, 1) | 3 | 241.7 | 1.2% | 79.17 | 14% | 21% | 0% | 0% | 66% |
| partial_argmax_kernel | (8, 64, 1) | 5 | 148.1 | 0.8% | 28.64 | 2% | 70% | 2% | 0% | 25% |
| gemm_bf16_splitk_kernel | (16, 5, 1) | 2 | 135.1 | 0.7% | 65.34 | 102% | 12% | 7% | 2% | 0% |
| gemm_bf16_partials_kernel | (6, 8, 1) | 6 | 117.3 | 0.6% | 18.40 | 31% | 0% | 42% | 3% | 24% |
| moe_swiglu_cast_kernel_kernel | (8, 288, 1) | 6 | 91.5 | 0.5% | 14.99 | 0% | 68% | 0% | 0% | 32% |

| wait | kernel | grid | last-warp us a step | a launch | first statement |
|---|---|---|---:|---:|---|
| wait.cp_async*118 | gemm_mxfp8_kernel | (80, 1, 1) | 207.0 | 20.70 | `tl::cp_async_wait<2>();` |
| wait.cp_async*15 | gemm_mxfp8_kernel | (40, 1, 1) | 137.5 | 15.28 | `tl::cp_async_wait<1>();` |
| wait.cp_async*4 | gemm_mxfp8_kernel | (128, 1, 1) | 110.0 | 12.23 | `tl::cp_async_wait<1>();` |
| wait.cp_async*19 | gemm_mxfp8_kernel | (28, 1, 1) | 93.3 | 10.36 | `tl::cp_async_wait<1>();` |
| wait.cp_async*13 | gemm_bf16_splitk_kernel | (16, 5, 1) | 73.9 | 36.93 | `tl::cp_async_wait<2>();` |
| wait.sync#2*118 | gemm_mxfp8_kernel | (80, 1, 1) | 69.3 | 6.93 | `__syncthreads();` |
| wait.sync#2*19 | gemm_mxfp8_kernel | (28, 1, 1) | 65.0 | 7.22 | `__syncthreads();` |
| wait.cp_async*8 | gemm_mxfp8_kernel | (80, 1, 1) | 60.3 | 6.03 | `tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| wait.sync#2*15 | gemm_mxfp8_kernel | (40, 1, 1) | 59.2 | 6.58 | `__syncthreads();` |
| wait.sync#2*19 | gemm_mxfp8_kernel | (18, 1, 1) | 52.7 | 5.86 | `__syncthreads();` |
| wait.cp_async*19 | gemm_mxfp8_kernel | (18, 1, 1) | 47.5 | 5.28 | `tl::cp_async_wait<1>();` |
| wait.mbarrier#2 | gemm_bf16_kernel | (2020, 1, 1) | 34.6 | 6.92 | `mbarrier[0].wait(0);` |
| wait.cp_async*14 | gemm_fp8w_bf16_partials_kernel | (32, 4, 1) | 32.2 | 3.58 | `tl::cp_async_wait<3>();` |
| wait.cp_async*9 | gemm_mxfp8_kernel | (64, 1, 1) | 30.3 | 15.17 | `tl::cp_async_wait<1>();` |
| wait.sync#2*8 | gemm_mxfp8_kernel | (80, 1, 1) | 26.7 | 2.67 | `__syncthreads();` |
| wait.sync#2*14 | gemm_fp8w_bf16_partials_kernel | (32, 4, 1) | 23.4 | 2.60 | `__syncthreads();` |
| wait.sync*13 | gemm_bf16_splitk_kernel | (16, 5, 1) | 22.3 | 11.13 | `__syncthreads();` |
| wait.sync*14 | gemm_bf16_splitk_kernel | (16, 5, 1) | 20.1 | 10.04 | `__syncthreads();` |
| wait.sync#2*4 | gemm_mxfp8_kernel | (128, 1, 1) | 18.6 | 2.07 | `__syncthreads();` |
| wait.sync*4 | mhc_project_streams_kernel | (40, 3, 1) | 15.7 | 0.87 | `__syncthreads();` |

### vocab_packed_projection_kernel grid (1, 4040, 1) block (256, 1, 1)

18 launches, 32320 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 4656.96 us, warp 57.89 us, nsys graphs 4546.11 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 24.58 | 42.5% | 21.59 | 38.1% |
| compute | 13.60 | 23.5% | 13.77 | 24.3% |
| load | 11.24 | 19.4% | 14.90 | 26.3% |
| unattributed | 8.40 | 14.5% | 6.29 | 11.1% |
| store | 0.06 | 0.1% | 0.06 | 0.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*37 (est.: sampled iterations x37) | 12.65 | 21.9% | 1.0 | 13.62 | `{` |
| wait.sync#2*37 (est.: sampled iterations x37) | 9.27 | 16.0% | 1.0 | 25.46 | `__syncthreads();` |
| wait.sync#3*37 (est.: sampled iterations x37) | 5.72 | 9.9% | 1.0 | 9.47 | `__syncthreads();` |
| load.global | 5.62 | 9.7% | 1.0 | 11.15 | `int start = row_offsets[(((int)blockIdx.y) * 16)]; \| int start = row_offsets[(((int)blockI` |
| load.cp_async#2*37 (est.: sampled iterations x37) | 5.27 | 9.1% | 1.0 | 7.10 | `for (int i_4 = 0; i_4 < 4; ++i_4)` |
| wait.cp_async*37 (est.: sampled iterations x37) | 3.61 | 6.2% | 1.0 | 23.68 | `tl::cp_async_wait<6>();` |
| wait.sync*37 (est.: sampled iterations x37) | 2.50 | 4.3% | 1.0 | 3.55 | `__syncthreads();` |
| wait.cp_async#2*37 (est.: sampled iterations x37) | 2.22 | 3.8% | 1.0 | 2.37 | `tl::cp_async_wait<7>();` |
| load.cp_async | 0.35 | 0.6% | 1.0 | 0.48 | `for (int i = 0; i < 4; ++i) \| for (int i = 0; i < 8; ++i) \| for (int i_1 = 0; i_1 < 4; ++i` |
| compute.mma#4 | 0.32 | 0.5% | 1.0 | 0.35 | `{` |
| compute.mma#2 | 0.31 | 0.5% | 1.0 | 0.35 | `{` |
| compute.mma#3 | 0.31 | 0.5% | 1.0 | 0.32 | `{` |
| wait.sync#8 | 0.25 | 0.4% | 1.0 | 0.90 | `__syncthreads();` |
| wait.sync#4 | 0.18 | 0.3% | 1.0 | 0.62 | `__syncthreads();` |
| wait.sync#7 | 0.15 | 0.3% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#5 | 0.14 | 0.2% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#9 | 0.14 | 0.2% | 1.0 | 0.26 | `__syncthreads();` |
| wait.cp_async#7 | 0.10 | 0.2% | 1.0 | 0.90 | `tl::cp_async_wait<0>();` |
| wait.sync#6 | 0.07 | 0.1% | 1.0 | 0.11 | `__syncthreads();` |
| wait.cp_async#3 | 0.07 | 0.1% | 1.0 | 0.61 | `tl::cp_async_wait<4>();` |
| store.global | 0.06 | 0.1% | 1.0 | 0.10 | `for (int i_13 = 0; i_13 < 2; ++i_13) \| for (int i_14 = 0; i_14 < 4; ++i_14) \| for (int i_9` |
| wait.cp_async#5 | 0.05 | 0.1% | 1.0 | 0.10 | `tl::cp_async_wait<2>();` |
| wait.cp_async#8 | 0.05 | 0.1% | 1.0 | 0.06 | `tl::cp_async_wait<1>();` |
| wait.cp_async#4 | 0.04 | 0.1% | 1.0 | 0.06 | `tl::cp_async_wait<5>();` |
| wait.cp_async#6 | 0.04 | 0.1% | 1.0 | 0.06 | `tl::cp_async_wait<3>();` |

### vocab_packed_projection_kernel grid (1, 4040, 1) block (128, 1, 1)

16 launches, 16160 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 4078.93 us, warp 50.75 us, nsys graphs 4056.77 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 26.19 | 51.6% | 25.97 | 53.8% |
| load | 12.49 | 24.6% | 15.82 | 32.8% |
| unattributed | 6.96 | 13.7% | 1.46 | 3.0% |
| compute | 5.05 | 10.0% | 4.99 | 10.3% |
| store | 0.05 | 0.1% | 0.05 | 0.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.cp_async*17 (est.: sampled iterations x17) | 10.56 | 20.8% | 1.0 | 40.26 | `tl::cp_async_wait<6>();` |
| wait.sync#2*17 (est.: sampled iterations x17) | 9.93 | 19.6% | 1.0 | 28.56 | `__syncthreads();` |
| load.global | 9.90 | 19.5% | 1.0 | 16.98 | `int start = row_offsets[(((int)blockIdx.y) * 16)]; \| int start = row_offsets[(((int)blockI` |
| compute.mma*17 (est.: sampled iterations x17) | 4.35 | 8.6% | 1.0 | 4.35 | `{` |
| load.cp_async#2*17 (est.: sampled iterations x17) | 2.29 | 4.5% | 1.0 | 2.72 | `for (int i_6 = 0; i_6 < 8; ++i_6) \| for (int i_9 = 0; i_9 < 4; ++i_9)` |
| wait.sync*17 (est.: sampled iterations x17) | 1.09 | 2.1% | 1.0 | 1.09 | `__syncthreads();` |
| wait.sync#3*17 (est.: sampled iterations x17) | 1.04 | 2.0% | 1.0 | 1.36 | `__syncthreads();` |
| wait.cp_async#2*17 (est.: sampled iterations x17) | 0.92 | 1.8% | 1.0 | 1.09 | `tl::cp_async_wait<7>();` |
| wait.cp_async#3 | 0.65 | 1.3% | 1.0 | 2.18 | `tl::cp_async_wait<4>();` |
| wait.sync#6 | 0.47 | 0.9% | 1.0 | 1.55 | `__syncthreads();` |
| wait.sync#4 | 0.47 | 0.9% | 1.0 | 1.41 | `__syncthreads();` |
| wait.cp_async#5 | 0.42 | 0.8% | 1.0 | 1.78 | `tl::cp_async_wait<2>();` |
| load.cp_async | 0.29 | 0.6% | 1.0 | 0.37 | `for (int i = 0; i < 4; ++i) \| for (int i = 0; i < 8; ++i) \| for (int i_1 = 0; i_1 < 4; ++i` |
| compute.mma#2 | 0.24 | 0.5% | 1.0 | 0.26 | `{` |
| compute.mma#3 | 0.23 | 0.5% | 1.0 | 0.26 | `{` |
| compute.mma#4 | 0.23 | 0.5% | 1.0 | 0.26 | `{` |
| wait.cp_async#7 | 0.19 | 0.4% | 1.0 | 0.98 | `tl::cp_async_wait<0>();` |
| wait.sync#8 | 0.17 | 0.3% | 1.0 | 0.93 | `__syncthreads();` |
| wait.sync#5 | 0.05 | 0.1% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#9 | 0.05 | 0.1% | 1.0 | 0.06 | `__syncthreads();` |
| store.global | 0.05 | 0.1% | 1.0 | 0.06 | `for (int i_13 = 0; i_13 < 2; ++i_13) \| for (int i_14 = 0; i_14 < 4; ++i_14) \| for (int i_9` |
| wait.sync#7 | 0.05 | 0.1% | 1.0 | 0.06 | `__syncthreads();` |
| wait.cp_async#4 | 0.04 | 0.1% | 1.0 | 0.06 | `tl::cp_async_wait<5>();` |
| wait.cp_async#6 | 0.04 | 0.1% | 1.0 | 0.06 | `tl::cp_async_wait<3>();` |
| wait.cp_async#8 | 0.04 | 0.1% | 1.0 | 0.06 | `tl::cp_async_wait<1>();` |

### gemm_bf16_kernel grid (2020, 1, 1) block (256, 1, 1)

85 launches, 16160 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 326.18 us, warp 4.50 us, nsys graphs 271.97 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 3.83 | 85.1% | 7.00 | 91.2% |
| unattributed | 0.45 | 10.1% | 0.32 | 4.2% |
| compute | 0.14 | 3.2% | 0.29 | 3.7% |
| load | 0.04 | 0.8% | 0.00 | 0.0% |
| store | 0.04 | 0.8% | 0.07 | 0.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.mbarrier#2 | 3.72 | 82.8% | 0.5 | 10.30 | `mbarrier[0].wait(0);` |
| compute.mma | 0.14 | 3.2% | 0.5 | 0.29 | `{` |
| wait.sync | 0.08 | 1.7% | 1.0 | 0.10 | `__syncthreads();` |
| load.tma | 0.04 | 0.8% | 0.5 | 0.13 | `if (tl::tl_shuffle_elect<128>())` |
| store.global | 0.04 | 0.8% | 0.5 | 0.10 | `for (int i_3 = 0; i_3 < 4; ++i_3) \| for (int i_4 = 0; i_4 < 8; ++i_4) \| for (int i_6 = 0; ` |
| wait.mbarrier | 0.03 | 0.6% | 0.5 | 0.06 | `mbarrier[2].wait(1);` |

### l2_prefetch_kernel grid (2, 1, 1) block (128, 1, 1)

289 launches, 8 warps; ranges, warp and last warp from the 2 recording CTAs' 8 warps; span 69.79 us, warp 68.38 us, nsys graphs 69.44 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 67.96 | 99.4% | 75.85 | 99.5% |
| unattributed | 0.42 | 0.6% | 0.36 | 0.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 67.96 | 99.4% | 1.0 | 69.41 | `for (int s = 0; s < (nseg2 >> 1); ++s)` |

### moe_grouped_fp8_fp4_gemm_kernel grid (40, 288, 1) block (128, 1, 1)

102 launches, 46080 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 170.05 us, warp 3.07 us, nsys graphs 118.75 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.36 | 44.1% | 0.20 | 36.3% |
| load | 0.71 | 23.2% | 0.31 | 55.9% |
| wait | 0.50 | 16.3% | 0.04 | 7.9% |
| compute | 0.44 | 14.3% | 0.00 | 0.0% |
| store | 0.06 | 2.0% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*5 (est.: sampled iterations x5) | 0.44 | 14.3% | 0.2 | 1.76 | `if (ko < 4)` |
| load.global | 0.31 | 10.0% | 1.0 | 0.38 | `int expert = TileExpert[((int64_t)((int)blockIdx.y))];` |
| load#2*5 (est.: sampled iterations x5) | 0.23 | 7.3% | 0.2 | 0.96 | `if (ko < 3)` |
| load.global#3 | 0.11 | 3.6% | 0.2 | 0.45 | `if (((expert < 128) && (0 <= expert))) \| if (((expert < 384) && (0 <= expert)))` |
| wait.sync#5*5 (est.: sampled iterations x5) | 0.10 | 3.3% | 0.2 | 0.48 | `__syncthreads();` |
| wait.mbarrier*5 (est.: sampled iterations x5) | 0.09 | 2.9% | 0.2 | 0.48 | `loaded[(ko % 3)].wait((ko / 3));` |
| wait.sync#4*5 (est.: sampled iterations x5) | 0.08 | 2.6% | 0.2 | 0.32 | `__syncthreads();` |
| wait.sync#6*5 (est.: sampled iterations x5) | 0.08 | 2.6% | 0.2 | 0.32 | `__syncthreads();` |
| store.global | 0.06 | 2.0% | 0.2 | 0.29 | `for (int i_2 = 0; i_2 < 16; ++i_2) \| for (int i_3 = 0; i_3 < 16; ++i_3) \| for (int i_4 = 0` |
| wait.cp_async*5 (est.: sampled iterations x5) | 0.06 | 2.0% | 0.2 | 0.32 | `tl::cp_async_wait<2>();` |
| load | 0.04 | 1.4% | 0.2 | 0.26 | `for (int s = 0; s < 2; ++s)` |
| wait.sync | 0.04 | 1.4% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#2 | 0.04 | 1.2% | 0.2 | 0.19 | `__syncthreads();` |
| load.global#2 | 0.03 | 0.8% | 0.2 | 0.22 | `if (((int)threadIdx.x) < 32) \| if (((int)threadIdx.x) < 64)` |
| wait.sync#3 | 0.01 | 0.3% | 0.2 | 0.03 | `__syncthreads();` |

### mhc_project_streams_kernel grid (40, 3, 1) block (128, 1, 1)

306 launches, 480 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 52.38 us, warp 20.56 us, nsys graphs 32.37 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 9.43 | 45.9% | 5.19 | 36.5% |
| unattributed | 6.29 | 30.6% | 4.93 | 34.6% |
| compute | 2.76 | 13.4% | 2.54 | 17.8% |
| wait | 1.60 | 7.8% | 1.08 | 7.6% |
| store | 0.48 | 2.3% | 0.50 | 3.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2*4 (est.: sampled iterations x4) | 9.23 | 44.9% | 1.0 | 17.15 | `for (int i_1 = 0; i_1 < 6; ++i_1)` |
| compute.mma*4 (est.: sampled iterations x4) | 2.62 | 12.8% | 1.0 | 2.69 | `{` |
| wait.sync*4 (est.: sampled iterations x4) | 1.41 | 6.8% | 1.0 | 5.31 | `__syncthreads();` |
| store.global*4 (est.: sampled iterations x4) | 0.26 | 1.2% | 1.0 | 0.26 | `if (((((int)blockIdx.y) * 16) + (((int)threadIdx.x) >> 3)) < M)` |
| load.global | 0.20 | 1.0% | 1.0 | 0.26 | `for (int k = 0; k < 4; ++k)` |
| wait.sync#2*4 (est.: sampled iterations x4) | 0.19 | 0.9% | 1.0 | 0.26 | `__syncthreads();` |
| store.global#4 | 0.10 | 0.5% | 1.0 | 0.13 | `if (((((int)threadIdx.x) % 8) == 0) && (((((int)blockIdx.y) * 16) + (((int)threadIdx.x) >>` |
| compute.reduce | 0.07 | 0.3% | 1.0 | 0.10 | `red[0] = (red[0] + __shfl_xor_sync((uint)4294967295, red[0], 4, 32));` |
| compute.reduce#2 | 0.07 | 0.3% | 1.0 | 0.10 | `red[0] = (red[0] + __shfl_xor_sync((uint)4294967295, red[0], 4, 32));` |
| store.global#2 | 0.06 | 0.3% | 1.0 | 0.06 | `if (((((int)blockIdx.y) * 16) + (((int)threadIdx.x) >> 3)) < M) \| if (((((int)threadIdx.x)` |
| store.global#3 | 0.06 | 0.3% | 1.0 | 0.06 | `if (((((int)threadIdx.x) % 8) == 0) && (((((int)blockIdx.y) * 16) + (((int)threadIdx.x) >>` |
| compute.mma | 0.00 | 0.0% | 0.0 | 0.00 | `{` |
| load.global#2 | 0.00 | 0.0% | 0.0 | 0.00 | `for (int i_1 = 0; i_1 < 6; ++i_1)` |
| store.global | 0.00 | 0.0% | 0.0 | 0.00 | `for (int n = 0; n < 4; ++n) \| if (((((int)blockIdx.y) * 16) + (((int)threadIdx.x) >> 3)) <` |
| wait.sync | 0.00 | 0.0% | 0.0 | 0.00 | `__syncthreads();` |

### moe_grouped_fp8_fp4_gemm_kernel grid (9, 288, 1) block (128, 1, 1)

102 launches, 10368 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 133.86 us, warp 21.07 us, nsys graphs 122.56 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 8.55 | 40.6% | 0.29 | 73.4% |
| wait | 7.25 | 34.4% | 0.05 | 11.7% |
| compute | 3.36 | 15.9% | 0.00 | 0.0% |
| load | 1.85 | 8.8% | 0.06 | 14.8% |
| store | 0.06 | 0.3% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.mbarrier*40 (est.: sampled iterations x40) | 3.88 | 18.4% | 0.2 | 16.00 | `loaded[(ko % 3)].wait(((ko % 6) / 3));` |
| compute.mma*40 (est.: sampled iterations x40) | 3.36 | 15.9% | 0.2 | 14.08 | `{` |
| load#2*40 (est.: sampled iterations x40) | 1.52 | 7.2% | 0.2 | 8.96 | `if (ko < 38)` |
| wait.sync#5*40 (est.: sampled iterations x40) | 1.36 | 6.5% | 0.2 | 6.40 | `__syncthreads();` |
| wait.cp_async*40 (est.: sampled iterations x40) | 0.64 | 3.0% | 0.2 | 2.56 | `tl::cp_async_wait<2>();` |
| wait.sync#4*40 (est.: sampled iterations x40) | 0.64 | 3.0% | 0.2 | 2.56 | `__syncthreads();` |
| wait.sync#6*40 (est.: sampled iterations x40) | 0.64 | 3.0% | 0.2 | 2.56 | `__syncthreads();` |
| load.global | 0.14 | 0.7% | 1.0 | 0.38 | `int expert = TileExpert[((int64_t)((int)blockIdx.y))];` |
| load.global#3 | 0.13 | 0.6% | 0.2 | 0.51 | `if (((expert < 128) && (0 <= expert))) \| if (((expert < 384) && (0 <= expert)))` |
| store.global | 0.06 | 0.3% | 0.2 | 0.26 | `for (int i_2 = 0; i_2 < 16; ++i_2) \| for (int i_3 = 0; i_3 < 16; ++i_3) \| for (int i_4 = 0` |
| load | 0.05 | 0.2% | 0.2 | 0.26 | `for (int s = 0; s < 2; ++s)` |
| wait.sync | 0.04 | 0.2% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#2 | 0.03 | 0.2% | 0.2 | 0.16 | `__syncthreads();` |
| load.global#2 | 0.02 | 0.1% | 0.2 | 0.19 | `if (((int)threadIdx.x) < 32) \| if (((int)threadIdx.x) < 64)` |
| wait.sync#3 | 0.01 | 0.1% | 0.2 | 0.06 | `__syncthreads();` |

### gemm_mxfp8_kernel grid (80, 1, 1) block (128, 1, 1)

170 launches, 320 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 16.34 us, warp 13.50 us, nsys graphs 18.13 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 5.36 | 39.7% | 38.75 | 72.0% |
| unattributed | 5.32 | 39.4% | 8.18 | 15.2% |
| compute | 1.71 | 12.7% | 3.91 | 7.3% |
| load | 0.82 | 6.1% | 2.58 | 4.8% |
| store | 0.28 | 2.1% | 0.37 | 0.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync#2*8 (est.: sampled iterations x8) | 2.14 | 15.9% | 1.0 | 5.76 | `__syncthreads();` |
| wait.cp_async*8 (est.: sampled iterations x8) | 1.91 | 14.2% | 1.0 | 6.66 | `tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| compute.mma*8 (est.: sampled iterations x8) | 1.52 | 11.3% | 1.0 | 1.79 | `{` |
| load.cp_async#2*8 (est.: sampled iterations x8) | 0.69 | 5.1% | 1.0 | 1.02 | `for (int i_3 = 0; i_3 < 2; ++i_3) \| for (int i_5 = 0; i_5 < 2; ++i_5) \| tl::cp_async_gs_co` |
| wait.cp_async#2 | 0.53 | 3.9% | 1.0 | 1.57 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| wait.sync*8 (est.: sampled iterations x8) | 0.42 | 3.1% | 1.0 | 0.51 | `__syncthreads();` |
| wait.sync#3 | 0.35 | 2.6% | 1.0 | 1.09 | `__syncthreads();` |
| store.global | 0.28 | 2.1% | 1.0 | 0.48 | `for (int i_10 = 0; i_10 < 16; ++i_10) \| for (int i_10 = 0; i_10 < 8; ++i_10) \| for (int i_` |
| compute.mma#2 | 0.19 | 1.4% | 1.0 | 0.26 | `{` |
| load.cp_async | 0.13 | 1.0% | 1.0 | 0.19 | `for (int i = 0; i < 2; ++i) \| for (int i_1 = 0; i_1 < 2; ++i_1) \| for (int i_1 = 0; i_1 < ` |
| compute.mma#3 | 0.00 | 0.0% | 0.0 | 0.00 | `{` |
| compute.mma*118 (est.: sampled iterations x118) | 0.00 | 0.0% | 0.0 | 0.00 | `{` |
| load.cp_async#2*118 (est.: sampled iterations x118) | 0.00 | 0.0% | 0.0 | 0.00 | `for (int i_5 = 0; i_5 < 2; ++i_5) \| for (int i_5 = 0; i_5 < 4; ++i_5) \| tl::cp_async_gs_co` |
| wait.cp_async#3 | 0.00 | 0.0% | 0.0 | 0.00 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>();` |
| wait.cp_async*118 (est.: sampled iterations x118) | 0.00 | 0.0% | 0.0 | 0.00 | `tl::cp_async_wait<2>();` |
| wait.sync#2*118 (est.: sampled iterations x118) | 0.00 | 0.0% | 0.0 | 0.00 | `__syncthreads();` |
| wait.sync#4 | 0.00 | 0.0% | 0.0 | 0.00 | `__syncthreads();` |
| wait.sync*118 (est.: sampled iterations x118) | 0.00 | 0.0% | 0.0 | 0.00 | `__syncthreads();` |

### gemm_mxfp8_kernel grid (128, 1, 1) block (128, 1, 1)

153 launches, 512 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 40.64 us, warp 14.08 us, nsys graphs 61.92 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 10.31 | 73.2% | 16.59 | 113.3% |
| compute | 2.55 | 18.1% | 2.54 | 17.3% |
| load | 0.76 | 5.4% | 0.76 | 5.2% |
| unattributed | 0.30 | 2.1% | 0.00 | 0.0% |
| store | 0.15 | 1.1% | 0.16 | 1.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.cp_async*4 (est.: sampled iterations x4) | 6.89 | 48.9% | 1.0 | 16.00 | `tl::cp_async_wait<1>();` |
| compute.mma*4 (est.: sampled iterations x4) | 2.04 | 14.5% | 1.0 | 2.05 | `{` |
| wait.sync#2*4 (est.: sampled iterations x4) | 1.62 | 11.5% | 1.0 | 5.12 | `__syncthreads();` |
| wait.cp_async#2 | 1.22 | 8.6% | 1.0 | 2.78 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| load.cp_async#2*4 (est.: sampled iterations x4) | 0.56 | 4.0% | 1.0 | 0.64 | `for (int i_3 = 0; i_3 < 8; ++i_3)` |
| compute.mma#2 | 0.51 | 3.7% | 1.0 | 0.54 | `{` |
| wait.sync#3 | 0.40 | 2.8% | 1.0 | 1.31 | `__syncthreads();` |
| load.cp_async | 0.20 | 1.4% | 1.0 | 0.26 | `for (int i = 0; i < 2; ++i) \| for (int i_1 = 0; i_1 < 2; ++i_1) \| for (int i_1 = 0; i_1 < ` |
| wait.sync*4 (est.: sampled iterations x4) | 0.19 | 1.4% | 1.0 | 0.26 | `__syncthreads();` |
| store.global | 0.15 | 1.1% | 1.0 | 0.19 | `for (int i_10 = 0; i_10 < 16; ++i_10) \| for (int i_10 = 0; i_10 < 8; ++i_10) \| for (int i_` |

### gemm_fp8w_bf16_partials_kernel grid (32, 4, 1) block (128, 1, 1)

153 launches, 512 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 58.53 us, warp 56.55 us, nsys graphs 45.12 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 34.80 | 61.5% | 20.14 | 44.7% |
| wait | 12.09 | 21.4% | 13.51 | 30.0% |
| compute | 5.58 | 9.9% | 5.93 | 13.2% |
| load | 3.92 | 6.9% | 5.33 | 11.8% |
| store | 0.16 | 0.3% | 0.15 | 0.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*14 (est.: sampled iterations x14) | 4.98 | 8.8% | 1.0 | 6.27 | `{` |
| wait.sync#2*14 (est.: sampled iterations x14) | 2.83 | 5.0% | 1.0 | 8.06 | `__syncthreads();` |
| wait.cp_async*14 (est.: sampled iterations x14) | 2.27 | 4.0% | 1.0 | 11.65 | `tl::cp_async_wait<3>();` |
| wait.cp_async#3 | 1.45 | 2.6% | 1.0 | 3.68 | `tl::cp_async_wait<3>();` |
| load.cp_async#3*14 (est.: sampled iterations x14) | 1.26 | 2.2% | 1.0 | 1.79 | `for (int i_10 = 0; i_10 < 4; ++i_10) \| for (int i_10 = 0; i_10 < 6; ++i_10) \| for (int i_1` |
| load.global*14 (est.: sampled iterations x14) | 1.06 | 1.9% | 1.0 | 1.79 | `if (((int)threadIdx.x) < 4)` |
| load.cp_async#2*14 (est.: sampled iterations x14) | 1.01 | 1.8% | 1.0 | 1.34 | `for (int i_4 = 0; i_4 < 2; ++i_4) \| for (int i_6 = 0; i_6 < 2; ++i_6)` |
| wait.sync#5*14 (est.: sampled iterations x14) | 0.95 | 1.7% | 1.0 | 1.34 | `__syncthreads();` |
| wait.sync*14 (est.: sampled iterations x14) | 0.95 | 1.7% | 1.0 | 1.34 | `__syncthreads();` |
| wait.sync#3*14 (est.: sampled iterations x14) | 0.92 | 1.6% | 1.0 | 1.34 | `__syncthreads();` |
| wait.sync#4*14 (est.: sampled iterations x14) | 0.92 | 1.6% | 1.0 | 1.34 | `__syncthreads();` |
| wait.cp_async#2*14 (est.: sampled iterations x14) | 0.81 | 1.4% | 1.0 | 0.90 | `tl::cp_async_wait<3>();` |
| load.cp_async | 0.46 | 0.8% | 1.0 | 1.09 | `for (int i_1 = 0; i_1 < 2; ++i_1)` |
| wait.sync#7 | 0.40 | 0.7% | 1.0 | 1.18 | `__syncthreads();` |
| compute.mma#2 | 0.31 | 0.6% | 1.0 | 0.42 | `{` |
| compute.mma#3 | 0.28 | 0.5% | 1.0 | 0.35 | `{` |
| store.global | 0.16 | 0.3% | 1.0 | 0.19 | `for (int i_13 = 0; i_13 < 8; ++i_13) \| for (int i_19 = 0; i_19 < 16; ++i_19) \| for (int i_` |
| wait.sync#10 | 0.16 | 0.3% | 1.0 | 0.48 | `__syncthreads();` |
| wait.cp_async#5 | 0.11 | 0.2% | 1.0 | 0.77 | `tl::cp_async_wait<1>();` |
| load.global#2 | 0.06 | 0.1% | 1.0 | 0.10 | `if (((int)threadIdx.x) < 4)` |
| load.global#3 | 0.06 | 0.1% | 1.0 | 0.16 | `if (((int)threadIdx.x) < 4)` |
| wait.cp_async#6 | 0.06 | 0.1% | 1.0 | 0.06 | `tl::cp_async_wait<0>();` |
| wait.sync#6 | 0.05 | 0.1% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#8 | 0.05 | 0.1% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#11 | 0.05 | 0.1% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#9 | 0.05 | 0.1% | 1.0 | 0.10 | `__syncthreads();` |
| wait.cp_async#4 | 0.04 | 0.1% | 1.0 | 0.06 | `tl::cp_async_wait<2>();` |

### gemm_mxfp8_kernel grid (28, 1, 1) block (128, 1, 1)

153 launches, 112 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 43.10 us, warp 42.41 us, nsys graphs 42.91 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 17.40 | 41.0% | 19.66 | 43.7% |
| compute | 10.61 | 25.0% | 10.59 | 23.5% |
| unattributed | 10.54 | 24.8% | 10.77 | 23.9% |
| load | 3.68 | 8.7% | 3.83 | 8.5% |
| store | 0.18 | 0.4% | 0.16 | 0.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*19 (est.: sampled iterations x19) | 10.11 | 23.8% | 1.0 | 10.34 | `{` |
| wait.cp_async*19 (est.: sampled iterations x19) | 9.80 | 23.1% | 1.0 | 28.58 | `tl::cp_async_wait<1>();` |
| wait.sync#2*19 (est.: sampled iterations x19) | 5.55 | 13.1% | 1.0 | 17.63 | `__syncthreads();` |
| load.cp_async#2*19 (est.: sampled iterations x19) | 3.46 | 8.2% | 1.0 | 3.65 | `for (int i_3 = 0; i_3 < 2; ++i_3) \| for (int i_3 = 0; i_3 < 8; ++i_3)` |
| wait.sync*19 (est.: sampled iterations x19) | 1.22 | 2.9% | 1.0 | 1.82 | `__syncthreads();` |
| wait.cp_async#2 | 0.61 | 1.4% | 1.0 | 1.44 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| compute.mma#2 | 0.50 | 1.2% | 1.0 | 0.51 | `{` |
| load.cp_async | 0.22 | 0.5% | 1.0 | 0.26 | `for (int i = 0; i < 2; ++i) \| for (int i_1 = 0; i_1 < 2; ++i_1) \| for (int i_1 = 0; i_1 < ` |
| wait.sync#3 | 0.22 | 0.5% | 1.0 | 0.74 | `__syncthreads();` |
| store.global | 0.18 | 0.4% | 1.0 | 0.22 | `for (int i_10 = 0; i_10 < 16; ++i_10) \| for (int i_10 = 0; i_10 < 8; ++i_10) \| for (int i_` |

### gemm_mxfp8_kernel grid (40, 1, 1) block (128, 1, 1)

153 launches, 160 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 44.77 us, warp 44.16 us, nsys graphs 47.68 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 19.83 | 44.9% | 23.36 | 52.3% |
| unattributed | 12.72 | 28.8% | 9.31 | 20.9% |
| compute | 8.56 | 19.4% | 8.58 | 19.2% |
| load | 2.43 | 5.5% | 2.57 | 5.8% |
| store | 0.62 | 1.4% | 0.82 | 1.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.cp_async*15 (est.: sampled iterations x15) | 12.60 | 28.5% | 1.0 | 33.12 | `tl::cp_async_wait<1>();` |
| compute.mma*15 (est.: sampled iterations x15) | 8.07 | 18.3% | 1.0 | 8.16 | `{` |
| wait.sync#2*15 (est.: sampled iterations x15) | 5.58 | 12.6% | 1.0 | 16.32 | `__syncthreads();` |
| load.cp_async#2*15 (est.: sampled iterations x15) | 2.25 | 5.1% | 1.0 | 2.40 | `for (int i_3 = 0; i_3 < 4; ++i_3) \| for (int i_3 = 0; i_3 < 6; ++i_3)` |
| wait.sync*15 (est.: sampled iterations x15) | 0.90 | 2.0% | 1.0 | 0.96 | `__syncthreads();` |
| store.global | 0.62 | 1.4% | 1.0 | 0.86 | `for (int i_10 = 0; i_10 < 16; ++i_10) \| for (int i_10 = 0; i_10 < 8; ++i_10) \| for (int i_` |
| wait.cp_async#2 | 0.52 | 1.2% | 1.0 | 1.57 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| compute.mma#2 | 0.49 | 1.1% | 1.0 | 0.51 | `{` |
| wait.sync#3 | 0.22 | 0.5% | 1.0 | 0.80 | `__syncthreads();` |
| load.cp_async | 0.18 | 0.4% | 1.0 | 0.22 | `for (int i = 0; i < 2; ++i) \| for (int i_1 = 0; i_1 < 2; ++i_1) \| for (int i_1 = 0; i_1 < ` |

### gemm_mxfp8_kernel grid (18, 1, 1) block (128, 1, 1)

153 launches, 72 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 26.94 us, warp 26.71 us, nsys graphs 32.96 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| compute | 10.56 | 39.5% | 10.55 | 34.7% |
| wait | 7.45 | 27.9% | 12.70 | 41.8% |
| unattributed | 4.86 | 18.2% | 3.01 | 9.9% |
| load | 3.66 | 13.7% | 3.96 | 13.0% |
| store | 0.19 | 0.7% | 0.17 | 0.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*19 (est.: sampled iterations x19) | 10.06 | 37.7% | 1.0 | 10.34 | `{` |
| load.cp_async#2*19 (est.: sampled iterations x19) | 3.44 | 12.9% | 1.0 | 3.65 | `for (int i_3 = 0; i_3 < 2; ++i_3) \| for (int i_3 = 0; i_3 < 8; ++i_3)` |
| wait.sync#2*19 (est.: sampled iterations x19) | 3.37 | 12.6% | 1.0 | 9.12 | `__syncthreads();` |
| wait.cp_async*19 (est.: sampled iterations x19) | 2.74 | 10.2% | 1.0 | 9.73 | `tl::cp_async_wait<1>();` |
| wait.sync*19 (est.: sampled iterations x19) | 1.22 | 4.6% | 1.0 | 1.82 | `__syncthreads();` |
| compute.mma#2 | 0.50 | 1.9% | 1.0 | 0.51 | `{` |
| load.cp_async | 0.22 | 0.8% | 1.0 | 0.26 | `for (int i = 0; i < 2; ++i) \| for (int i_1 = 0; i_1 < 2; ++i_1) \| for (int i_1 = 0; i_1 < ` |
| store.global | 0.19 | 0.7% | 1.0 | 0.22 | `for (int i_10 = 0; i_10 < 16; ++i_10) \| for (int i_10 = 0; i_10 < 8; ++i_10) \| for (int i_` |
| wait.cp_async#2 | 0.06 | 0.2% | 1.0 | 0.13 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| wait.sync#3 | 0.06 | 0.2% | 1.0 | 0.13 | `__syncthreads();` |

### sparse_mla_kernel grid (1, 48, 1) block (256, 1, 1)

68 launches, 384 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 61.50 us, warp 61.34 us, nsys graphs 48.78 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed, with loops' unsampled iterations | 56.02 | 91.3% | 56.90 | 87.3% |
| load | 2.40 | 3.9% | 5.40 | 8.3% |
| compute | 1.48 | 2.4% | 1.49 | 2.3% |
| wait | 1.35 | 2.2% | 1.27 | 1.9% |
| store | 0.10 | 0.2% | 0.10 | 0.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global (sampled iterations) | 2.03 | 3.3% | 4.0 | 2.30 | `for (int c = 0; c < 4; ++c) \| for (int i = 0; i < 8; ++i) \| for (int i_1 = 0; i_1 < 8; ++i` |
| wait.sync (sampled iterations) | 1.28 | 2.1% | 18.0 | 1.63 | `__syncthreads();` |
| compute.mma (sampled iterations) | 1.21 | 2.0% | 2.0 | 1.25 | `for (int ki = 0; ki < 32; ++ki) \| for (int ki_1 = 0; ki_1 < 2; ++ki_1)` |
| load.cp_async (sampled iterations) | 0.37 | 0.6% | 2.0 | 0.42 | `for (int k_1 = 0; k_1 < 5; ++k_1) \| for (int k_1 = 0; k_1 < 9; ++k_1) \| for (int k_3 = 0; ` |
| compute.reduce (sampled iterations) | 0.27 | 0.4% | 2.0 | 0.29 | `m_g[i_5] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(m_g[i_5], (&((` |
| store.global | 0.10 | 0.2% | 1.0 | 0.13 | `for (int i_10 = 0; i_10 < 32; ++i_10) \| for (int i_11 = 0; i_11 < 32; ++i_11) \| for (int i` |
| wait.cp_async (sampled iterations) | 0.07 | 0.1% | 1.0 | 0.10 | `tl::cp_async_wait<0>();` |

### moe_grouped_fp8_fp4_gemm_kernel grid (9, 120, 1) block (128, 1, 1)

51 launches, 4320 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 81.34 us, warp 13.45 us, nsys graphs 66.24 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 4.29 | 31.9% | 0.29 | 73.2% |
| wait | 4.01 | 29.8% | 0.05 | 12.3% |
| compute | 3.36 | 25.0% | 0.00 | 0.0% |
| load | 1.72 | 12.8% | 0.06 | 14.4% |
| store | 0.07 | 0.5% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*40 (est.: sampled iterations x40) | 3.36 | 25.0% | 0.2 | 14.08 | `{` |
| load#2*40 (est.: sampled iterations x40) | 1.52 | 11.3% | 0.2 | 8.96 | `if (ko < 38)` |
| wait.sync#5*40 (est.: sampled iterations x40) | 1.12 | 8.3% | 0.2 | 5.12 | `__syncthreads();` |
| wait.mbarrier*40 (est.: sampled iterations x40) | 0.96 | 7.1% | 0.2 | 3.84 | `loaded[(ko % 3)].wait(((ko % 6) / 3));` |
| wait.sync#4*40 (est.: sampled iterations x40) | 0.64 | 4.8% | 0.2 | 2.56 | `__syncthreads();` |
| wait.sync#6*40 (est.: sampled iterations x40) | 0.64 | 4.8% | 0.2 | 2.56 | `__syncthreads();` |
| wait.cp_async*40 (est.: sampled iterations x40) | 0.56 | 4.2% | 0.2 | 2.56 | `tl::cp_async_wait<2>();` |
| store.global | 0.07 | 0.5% | 0.2 | 0.32 | `for (int i_2 = 0; i_2 < 16; ++i_2) \| for (int i_3 = 0; i_3 < 16; ++i_3) \| for (int i_4 = 0` |
| load.global | 0.07 | 0.5% | 1.0 | 0.10 | `int expert = TileExpert[((int64_t)((int)blockIdx.y))];` |
| load.global#3 | 0.06 | 0.5% | 0.2 | 0.26 | `if (((expert < 128) && (0 <= expert))) \| if (((expert < 384) && (0 <= expert)))` |
| load | 0.05 | 0.3% | 0.2 | 0.26 | `for (int s = 0; s < 2; ++s)` |
| wait.sync | 0.05 | 0.3% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#2 | 0.03 | 0.2% | 0.2 | 0.16 | `__syncthreads();` |
| load.global#2 | 0.02 | 0.2% | 0.2 | 0.19 | `if (((int)threadIdx.x) < 32) \| if (((int)threadIdx.x) < 64)` |
| wait.sync#3 | 0.01 | 0.1% | 0.2 | 0.03 | `__syncthreads();` |

### moe_grouped_fp8_fp4_gemm_kernel grid (40, 120, 1) block (128, 1, 1)

51 launches, 19200 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 79.17 us, warp 2.76 us, nsys graphs 55.26 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.31 | 47.6% | 0.20 | 65.7% |
| wait | 0.48 | 17.5% | 0.04 | 13.8% |
| load | 0.45 | 16.2% | 0.06 | 20.5% |
| compute | 0.44 | 16.0% | 0.00 | 0.0% |
| store | 0.08 | 2.8% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*5 (est.: sampled iterations x5) | 0.44 | 16.0% | 0.2 | 1.76 | `if (ko < 4)` |
| load#2*5 (est.: sampled iterations x5) | 0.22 | 8.0% | 0.2 | 0.96 | `if (ko < 3)` |
| load.global | 0.10 | 3.6% | 1.0 | 0.22 | `int expert = TileExpert[((int64_t)((int)blockIdx.y))];` |
| wait.sync#5*5 (est.: sampled iterations x5) | 0.09 | 3.3% | 0.2 | 0.48 | `__syncthreads();` |
| wait.mbarrier*5 (est.: sampled iterations x5) | 0.08 | 2.9% | 0.2 | 0.32 | `loaded[(ko % 3)].wait((ko / 3));` |
| wait.sync#4*5 (est.: sampled iterations x5) | 0.08 | 2.9% | 0.2 | 0.32 | `__syncthreads();` |
| wait.sync#6*5 (est.: sampled iterations x5) | 0.08 | 2.9% | 0.2 | 0.32 | `__syncthreads();` |
| store.global | 0.08 | 2.8% | 0.2 | 0.32 | `for (int i_2 = 0; i_2 < 16; ++i_2) \| for (int i_3 = 0; i_3 < 16; ++i_3) \| for (int i_4 = 0` |
| wait.cp_async*5 (est.: sampled iterations x5) | 0.06 | 2.2% | 0.2 | 0.32 | `tl::cp_async_wait<2>();` |
| load.global#3 | 0.06 | 2.0% | 0.2 | 0.22 | `if (((expert < 128) && (0 <= expert))) \| if (((expert < 384) && (0 <= expert)))` |
| load | 0.05 | 1.7% | 0.2 | 0.26 | `for (int s = 0; s < 2; ++s)` |
| wait.sync | 0.05 | 1.7% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#2 | 0.04 | 1.4% | 0.2 | 0.19 | `__syncthreads();` |
| load.global#2 | 0.02 | 0.9% | 0.2 | 0.19 | `if (((int)threadIdx.x) < 32) \| if (((int)threadIdx.x) < 64)` |
| wait.sync#3 | 0.01 | 0.3% | 0.2 | 0.03 | `__syncthreads();` |

### partial_argmax_kernel grid (8, 64, 1) block (128, 1, 1)

85 launches, 2048 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 28.64 us, warp 22.28 us, nsys graphs 27.23 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 9.97 | 44.8% | 7.02 | 25.1% |
| load | 8.25 | 37.0% | 19.69 | 70.5% |
| compute | 2.65 | 11.9% | 0.51 | 1.8% |
| wait | 1.07 | 4.8% | 0.60 | 2.1% |
| store | 0.34 | 1.5% | 0.13 | 0.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 8.25 | 37.0% | 1.0 | 19.81 | `for (int j = 0; j < 16; ++j) \| for (int run = 0; run < 2; ++run)` |
| compute.reduce | 2.65 | 11.9% | 1.0 | 6.02 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| wait.sync | 1.07 | 4.8% | 1.0 | 3.68 | `__syncthreads();` |
| store.global | 0.34 | 1.5% | 1.0 | 1.54 | `if (((int)threadIdx.x) == 0)` |

### gemm_bf16_splitk_kernel grid (16, 5, 1) block (256, 1, 1)

34 launches, 640 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 65.34 us, warp 62.38 us, nsys graphs 67.47 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 34.48 | 55.3% | 68.08 | 101.9% |
| unattributed | 20.43 | 32.8% | 0.00 | 0.0% |
| compute | 4.05 | 6.5% | 4.46 | 6.7% |
| load | 3.23 | 5.2% | 7.83 | 11.7% |
| store | 0.19 | 0.3% | 1.11 | 1.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.cp_async*13 (est.: sampled iterations x13) | 17.89 | 28.7% | 0.5 | 33.90 | `tl::cp_async_wait<2>();` |
| wait.sync*13 (est.: sampled iterations x13) | 7.03 | 11.3% | 0.5 | 16.02 | `__syncthreads();` |
| wait.sync*14 (est.: sampled iterations x14) | 2.96 | 4.7% | 0.5 | 8.96 | `__syncthreads();` |
| compute.mma*13 (est.: sampled iterations x13) | 1.96 | 3.1% | 0.5 | 2.08 | `{` |
| load.cp_async | 1.72 | 2.8% | 1.0 | 3.82 | `for (int i_1 = 0; i_1 < 2; ++i_1) \| tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)A_sha` |
| compute.mma*14 (est.: sampled iterations x14) | 1.46 | 2.3% | 0.5 | 1.57 | `{` |
| wait.sync#3 | 0.97 | 1.5% | 1.0 | 2.40 | `__syncthreads();` |
| wait.sync#4 | 0.94 | 1.5% | 1.0 | 2.34 | `__syncthreads();` |
| wait.cp_async*14 (est.: sampled iterations x14) | 0.92 | 1.5% | 0.5 | 12.32 | `tl::cp_async_wait<1>();` |
| load.cp_async#2*13 (est.: sampled iterations x13) | 0.72 | 1.2% | 0.5 | 0.83 | `for (int i_10 = 0; i_10 < 2; ++i_10) \| for (int i_8 = 0; i_8 < 2; ++i_8) \| tl::cp_async_gs` |
| wait.cp_async#2 | 0.71 | 1.1% | 1.0 | 2.85 | `tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>(); \| tl::cp_async_wait<3>();` |
| load.cp_async#2*14 (est.: sampled iterations x14) | 0.66 | 1.1% | 0.5 | 0.90 | `for (int i_6 = 0; i_6 < 2; ++i_6) \| tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)A_sha` |
| wait.cp_async#3 | 0.62 | 1.0% | 1.0 | 2.99 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| wait.cp_async#4 | 0.53 | 0.8% | 0.5 | 1.81 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>();` |
| wait.sync#5 | 0.52 | 0.8% | 1.0 | 1.44 | `__syncthreads();` |
| wait.sync#2*14 (est.: sampled iterations x14) | 0.46 | 0.7% | 0.5 | 0.67 | `__syncthreads();` |
| wait.sync#2*13 (est.: sampled iterations x13) | 0.41 | 0.7% | 0.5 | 0.62 | `__syncthreads();` |
| wait.sync#6 | 0.33 | 0.5% | 1.0 | 0.64 | `__syncthreads();` |
| compute.mma#3 | 0.25 | 0.4% | 1.0 | 0.29 | `{` |
| compute.mma#2 | 0.25 | 0.4% | 1.0 | 0.29 | `{` |
| wait.sync#7 | 0.19 | 0.3% | 0.5 | 0.24 | `__syncthreads();` |
| load.global | 0.13 | 0.2% | 1.0 | 0.91 | `if (((int)threadIdx.x) == 0)` |
| compute.mma#4 | 0.13 | 0.2% | 0.5 | 0.14 | `{` |
| store.global | 0.12 | 0.2% | 1.0 | 0.19 | `for (int i_10 = 0; i_10 < 4; ++i_10) \| for (int i_16 = 0; i_16 < 4; ++i_16) \| for (int i_1` |
| store.global#3 | 0.06 | 0.1% | 1.0 | 0.10 | `if (((int*)last)[0] == 1)` |
| load.global#2*16 (est.: sampled iterations x16) | 0.00 | 0.0% | 0.0 | 0.00 | `acc = P[(((((((int64_t)((int)blockIdx.z)) * (int64_t)65536) + (((int64_t)i_26) * (int64_t)` |
| load.global#2*8 (est.: sampled iterations x8) | 0.00 | 0.0% | 0.0 | 0.00 | `acc = P[(((((((int64_t)((int)blockIdx.z)) * (int64_t)16384) + (((int64_t)i_17) * (int64_t)` |
| store.global#2*16 (est.: sampled iterations x16) | 0.00 | 0.0% | 0.0 | 0.00 | `if (((int)blockIdx.x) < 8)` |
| store.global#2*8 (est.: sampled iterations x8) | 0.00 | 0.0% | 0.0 | 0.00 | `C0[(((((((int64_t)((int)blockIdx.z)) * (int64_t)16384) + (((int64_t)i_18) * (int64_t)2048)` |

### gemm_bf16_partials_kernel grid (6, 8, 1) block (256, 1, 1)

102 launches, 384 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 18.40 us, warp 8.80 us, nsys graphs 18.72 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 3.86 | 43.9% | 1.75 | 24.4% |
| wait | 2.94 | 33.4% | 2.21 | 30.8% |
| compute | 1.49 | 16.9% | 3.00 | 41.7% |
| load | 0.39 | 4.4% | 0.00 | 0.0% |
| store | 0.12 | 1.3% | 0.22 | 3.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*10 (est.: sampled iterations x10) | 1.49 | 16.9% | 0.5 | 3.20 | `{` |
| wait.mbarrier*10 (est.: sampled iterations x10) | 1.49 | 16.9% | 0.5 | 3.20 | `mbarrier[((kb % 3) + 3)].wait((((kb % 6) / 3) ^ 1));` |
| wait.mbarrier#2*10 (est.: sampled iterations x10) | 1.30 | 14.8% | 0.5 | 5.44 | `mbarrier[(kb_1 % 3)].wait(((kb_1 % 6) / 3));` |
| load.tma*10 (est.: sampled iterations x10) | 0.39 | 4.4% | 0.5 | 1.28 | `if (tl::tl_shuffle_elect<128>())` |
| wait.sync | 0.14 | 1.6% | 1.0 | 0.22 | `__syncthreads();` |
| store.global | 0.12 | 1.3% | 0.5 | 0.29 | `for (int i_1 = 0; i_1 < 4; ++i_1) \| for (int i_2 = 0; i_2 < 8; ++i_2) \| for (int i_4 = 0; ` |

### moe_swiglu_cast_kernel_kernel grid (8, 288, 1) block (288, 1, 1)

102 launches, 20736 warps; ranges, warp and last warp from the 4 recording CTAs' 36 warps; span 14.99 us, warp 1.54 us, nsys graphs 6.26 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.77 | 50.3% | 0.29 | 32.4% |
| load | 0.56 | 36.4% | 0.61 | 67.6% |
| compute | 0.12 | 8.1% | 0.00 | 0.0% |
| store | 0.08 | 5.3% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.56 | 36.4% | 1.0 | 1.06 | `for (int r = 0; r < 4; ++r)` |
| compute.reduce*4 (est.: sampled iterations x4) | 0.12 | 8.1% | 0.2 | 0.51 | `if (0 <= orders[r_2])` |
| store.global*4 (est.: sampled iterations x4) | 0.08 | 5.3% | 0.2 | 0.38 | `if (0 <= orders[r_2])` |

### mhc_finalize_kernel grid (48, 1, 1) block (128, 1, 1)

204 launches, 192 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 6.50 us, warp 5.64 us, nsys graphs 7.79 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 2.06 | 36.5% | 3.15 | 45.3% |
| compute | 0.99 | 17.6% | 0.60 | 8.6% |
| store | 0.99 | 17.5% | 1.18 | 17.0% |
| load | 0.96 | 17.1% | 1.08 | 15.5% |
| wait | 0.64 | 11.3% | 0.95 | 13.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync | 0.64 | 11.3% | 1.0 | 0.86 | `__syncthreads();` |
| compute.reduce | 0.57 | 10.2% | 0.2 | 2.30 | `row_max[0] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<32>>::run(row_max[0]);` |
| store.global#4*7 (est.: sampled iterations x7) | 0.56 | 9.9% | 0.8 | 0.90 | `if ((((int)threadIdx.x) % 4) == 0)` |
| compute.reduce#2*7 (est.: sampled iterations x7) | 0.42 | 7.4% | 0.8 | 0.67 | `amax[0] = max(amax[0], __shfl_xor_sync((uint)4294967295, amax[0], 1, 32));` |
| load.global#4*7 (est.: sampled iterations x7) | 0.36 | 6.5% | 0.8 | 0.67 | `*(uint4*)(yv + 0) = *(uint4*)(C + ((((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((i` |
| store.global#3*7 (est.: sampled iterations x7) | 0.32 | 5.7% | 0.8 | 0.45 | `*(uint4*)(y + ((((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((int64_t)step) * (int6` |
| load.global | 0.31 | 5.5% | 1.0 | 0.93 | `if (((int)threadIdx.x) < 32)` |
| load.global#3 | 0.24 | 4.3% | 0.8 | 0.35 | `for (int i_split_2 = 0; i_split_2 < 40; ++i_split_2)` |
| store.global | 0.09 | 1.7% | 0.2 | 0.38 | `if (((int)threadIdx.x) < 4)` |
| load.global#2 | 0.05 | 0.9% | 0.2 | 0.19 | `cm[0] = ((mixes_shared[((((int)threadIdx.x) & 15) + 8)] * scale[2]) + base[((((int)threadI` |
| store.global#2 | 0.01 | 0.2% | 0.2 | 0.06 | `if ((((int)threadIdx.x) >> 4) == 0)` |

### sparse_mla_kernel grid (2, 1, 40) block (256, 1, 1)

51 launches, 640 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 26.24 us, warp 14.43 us, nsys graphs 22.66 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| compute | 6.92 | 47.9% | 6.92 | 45.3% |
| wait | 2.96 | 20.5% | 2.85 | 18.6% |
| unattributed, with loops' unsampled iterations | 2.60 | 18.0% | 2.58 | 16.9% |
| load | 1.68 | 11.7% | 2.36 | 15.5% |
| store | 0.27 | 1.9% | 0.56 | 3.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*4 (est.: sampled iterations x4) | 2.72 | 18.8% | 1.0 | 2.82 | `for (int ki = 0; ki < 32; ++ki) \| {` |
| compute.mma#2*4 (est.: sampled iterations x4) | 2.17 | 15.0% | 1.0 | 2.30 | `for (int ki_1 = 0; ki_1 < 2; ++ki_1) \| if (((((int)blockIdx.x) * 2) + (c_1 >> 1)) < 3)` |
| compute.reduce#2*8 (est.: sampled iterations x8) | 1.02 | 7.0% | 1.0 | 1.02 | `row_sum[i_7] = tl::AllReduce<tl::SumOp, 128, 32, 0, tl::NamedBarrier<128>>::run(row_sum[i_` |
| compute.reduce*8 (est.: sampled iterations x8) | 1.02 | 7.0% | 1.0 | 1.02 | `m_g[i_4] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(m_g[i_4], (&((` |
| load.global#2 | 0.71 | 4.9% | 1.0 | 0.86 | `for (int i = 0; i < 8; ++i)` |
| load.cp_async#2*4 (est.: sampled iterations x4) | 0.65 | 4.5% | 1.0 | 0.77 | `if (((((int)blockIdx.x) * 2) + (c_1 >> 1)) < 3) \| if ((c_1 < 3) & (c_1 < 3)) \| if (c < 3)` |
| wait.sync#10*8 (est.: sampled iterations x8) | 0.37 | 2.5% | 1.0 | 0.51 | `__syncthreads();` |
| wait.sync#8*8 (est.: sampled iterations x8) | 0.37 | 2.5% | 1.0 | 0.51 | `__syncthreads();` |
| wait.sync#7*4 (est.: sampled iterations x4) | 0.30 | 2.1% | 1.0 | 0.38 | `__syncthreads();` |
| wait.sync#5*4 (est.: sampled iterations x4) | 0.28 | 2.0% | 1.0 | 0.38 | `__syncthreads();` |
| wait.sync#4*4 (est.: sampled iterations x4) | 0.28 | 1.9% | 1.0 | 0.38 | `__syncthreads();` |
| wait.cp_async*4 (est.: sampled iterations x4) | 0.28 | 1.9% | 1.0 | 0.38 | `tl::cp_async_wait<0>();` |
| store.global | 0.27 | 1.9% | 1.0 | 0.58 | `for (int i_10 = 0; i_10 < 32; ++i_10) \| for (int i_11 = 0; i_11 < 32; ++i_11) \| for (int i` |
| wait.sync#6*4 (est.: sampled iterations x4) | 0.26 | 1.8% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#9*4 (est.: sampled iterations x4) | 0.26 | 1.8% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#11*4 (est.: sampled iterations x4) | 0.25 | 1.7% | 1.0 | 0.26 | `__syncthreads();` |
| load.cp_async (sampled iterations) | 0.23 | 1.6% | 1.0 | 0.26 | `for (int k_1 = 0; k_1 < 5; ++k_1) \| for (int k_1 = 0; k_1 < 9; ++k_1) \| for (int k_3 = 0; ` |
| wait.sync (sampled iterations) | 0.20 | 1.4% | 1.0 | 0.26 | `__syncthreads();` |
| load.global (sampled iterations) | 0.10 | 0.7% | 1.0 | 0.29 | `for (int c = 0; c < 4; ++c) \| for (int i = 0; i < 8; ++i) \| for (int i_1 = 0; i_1 < 8; ++i` |
| wait.sync#3 | 0.07 | 0.5% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#2 | 0.05 | 0.4% | 1.0 | 0.06 | `__syncthreads();` |

### mhc_post_kernel grid (240, 1, 1) block (128, 1, 1)

68 launches, 960 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 16.48 us, warp 7.27 us, nsys graphs 9.94 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 7.03 | 96.8% | 8.51 | 79.5% |
| load | 0.17 | 2.3% | 1.60 | 15.0% |
| store | 0.07 | 0.9% | 0.58 | 5.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.17 | 2.3% | 1.0 | 0.22 | `*(float4*)(pm + 0) = *(float4*)(post + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int` |
| store.global | 0.07 | 0.9% | 1.0 | 0.26 | `for (int n_1 = 0; n_1 < 4; ++n_1)` |

### gemm_bf16_partials_kernel grid (2, 20, 1) block (256, 1, 1)

51 launches, 320 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 19.52 us, warp 8.39 us, nsys graphs 12.80 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 6.67 | 79.5% | 6.75 | 75.8% |
| wait | 0.94 | 11.2% | 0.88 | 9.8% |
| compute | 0.55 | 6.6% | 1.10 | 12.4% |
| load | 0.14 | 1.6% | 0.00 | 0.0% |
| store | 0.09 | 1.1% | 0.18 | 2.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.mbarrier#2*4 (est.: sampled iterations x4) | 0.71 | 8.5% | 0.5 | 3.20 | `mbarrier[(kb_1 % 3)].wait((kb_1 / 3));` |
| compute.mma*4 (est.: sampled iterations x4) | 0.55 | 6.6% | 0.5 | 1.15 | `{` |
| load.tma*4 (est.: sampled iterations x4) | 0.14 | 1.6% | 0.5 | 0.38 | `if (tl::tl_shuffle_elect<128>())` |
| wait.mbarrier*4 (est.: sampled iterations x4) | 0.12 | 1.5% | 0.5 | 0.26 | `mbarrier[((kb % 3) + 3)].wait(((kb / 3) ^ 1));` |
| wait.sync | 0.10 | 1.2% | 1.0 | 0.13 | `__syncthreads();` |
| store.global | 0.09 | 1.1% | 0.5 | 0.22 | `for (int i_1 = 0; i_1 < 4; ++i_1) \| for (int i_2 = 0; i_2 < 8; ++i_2) \| for (int i_4 = 0; ` |

### sparse_mla_kernel grid (1, 1, 48) block (128, 1, 1)

34 launches, 192 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 28.86 us, warp 18.97 us, nsys graphs 24.03 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed, with loops' unsampled iterations | 6.62 | 34.9% | 6.39 | 30.8% |
| compute | 4.65 | 24.5% | 4.64 | 22.3% |
| load | 3.89 | 20.5% | 5.54 | 26.7% |
| wait | 2.67 | 14.1% | 3.41 | 16.4% |
| store | 1.13 | 6.0% | 0.78 | 3.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 | 2.72 | 14.4% | 1.0 | 4.35 | `for (int i = 0; i < 8; ++i)` |
| compute.mma*4 (est.: sampled iterations x4) | 1.72 | 9.0% | 1.0 | 1.79 | `for (int ki = 0; ki < 32; ++ki) \| {` |
| compute.mma#2*4 (est.: sampled iterations x4) | 1.16 | 6.1% | 1.0 | 1.28 | `for (int ki_1 = 0; ki_1 < 2; ++ki_1) \| if (((((int)blockIdx.x) * 2) + (c_1 >> 1)) < 3)` |
| store.global | 1.13 | 6.0% | 1.0 | 1.92 | `for (int i_10 = 0; i_10 < 32; ++i_10) \| for (int i_11 = 0; i_11 < 32; ++i_11) \| for (int i` |
| compute.reduce#2*8 (est.: sampled iterations x8) | 0.91 | 4.8% | 1.0 | 1.02 | `row_sum[i_7] = tl::AllReduce<tl::SumOp, 128, 32, 0, tl::NamedBarrier<128>>::run(row_sum[i_` |
| compute.reduce*8 (est.: sampled iterations x8) | 0.86 | 4.6% | 1.0 | 1.02 | `m_g[i_4] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(m_g[i_4], (&((` |
| load.cp_async#2*4 (est.: sampled iterations x4) | 0.67 | 3.5% | 1.0 | 0.77 | `if (((((int)blockIdx.x) * 2) + (c_1 >> 1)) < 3) \| if ((c_1 < 3) & (c_1 < 3)) \| if (c < 3)` |
| load.cp_async (sampled iterations) | 0.36 | 1.9% | 1.0 | 0.38 | `for (int k_1 = 0; k_1 < 5; ++k_1) \| for (int k_1 = 0; k_1 < 9; ++k_1) \| for (int k_3 = 0; ` |
| wait.sync#8*8 (est.: sampled iterations x8) | 0.35 | 1.9% | 1.0 | 0.51 | `__syncthreads();` |
| wait.sync#10*8 (est.: sampled iterations x8) | 0.32 | 1.7% | 1.0 | 0.51 | `__syncthreads();` |
| wait.cp_async*4 (est.: sampled iterations x4) | 0.28 | 1.5% | 1.0 | 0.38 | `tl::cp_async_wait<0>();` |
| wait.sync#5*4 (est.: sampled iterations x4) | 0.26 | 1.3% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#11*4 (est.: sampled iterations x4) | 0.24 | 1.3% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#4*4 (est.: sampled iterations x4) | 0.24 | 1.3% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#7*4 (est.: sampled iterations x4) | 0.24 | 1.3% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#6*4 (est.: sampled iterations x4) | 0.23 | 1.2% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#9*4 (est.: sampled iterations x4) | 0.23 | 1.2% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync (sampled iterations) | 0.18 | 0.9% | 1.0 | 0.22 | `__syncthreads();` |
| load.global (sampled iterations) | 0.13 | 0.7% | 1.0 | 0.37 | `for (int c = 0; c < 4; ++c) \| for (int i = 0; i < 8; ++i) \| for (int i_1 = 0; i_1 < 8; ++i` |
| wait.sync#3 | 0.06 | 0.3% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#2 | 0.04 | 0.2% | 1.0 | 0.06 | `__syncthreads();` |

### cache_writer_kernel grid (48, 1, 1) block (128, 1, 1)

139 launches, 192 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 5.60 us, warp 5.41 us, nsys graphs 6.82 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 3.14 | 58.1% | 2.35 | 34.7% |
| store | 1.90 | 35.1% | 3.40 | 50.2% |
| load | 0.30 | 5.5% | 0.85 | 12.5% |
| compute | 0.07 | 1.4% | 0.18 | 2.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 1.90 | 35.1% | 1.0 | 2.24 | `if (((int64_t)0 <= slot) & ((slot >> (int64_t)7) < ((int64_t)pages))) \| if (((int64_t)0 <=` |
| load.global | 0.24 | 4.4% | 1.0 | 0.26 | `for (int i = 0; i < 4; ++i)` |
| compute.reduce | 0.07 | 1.4% | 1.0 | 0.10 | `amax = max(amax, __shfl_xor_sync((uint)4294967295, amax, 1, 32));` |
| load.global#2 | 0.06 | 1.1% | 1.0 | 0.06 | `int64_t slot = slots[((int64_t)((int)blockIdx.x))];` |

### gemm_mxfp8_kernel grid (64, 1, 1) block (64, 1, 1)

34 launches, 128 warps; ranges, warp and last warp from the 4 recording CTAs' 8 warps; span 25.15 us, warp 23.79 us, nsys graphs 25.84 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 17.02 | 71.6% | 18.22 | 74.2% |
| compute | 5.10 | 21.4% | 5.08 | 20.7% |
| load | 1.25 | 5.3% | 1.28 | 5.2% |
| unattributed | 0.23 | 1.0% | 0.00 | 0.0% |
| store | 0.19 | 0.8% | 0.23 | 0.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.cp_async*9 (est.: sampled iterations x9) | 13.39 | 56.3% | 1.0 | 22.61 | `tl::cp_async_wait<1>();` |
| compute.mma*9 (est.: sampled iterations x9) | 4.61 | 19.4% | 1.0 | 4.61 | `{` |
| wait.sync#2*9 (est.: sampled iterations x9) | 2.09 | 8.8% | 1.0 | 7.20 | `__syncthreads();` |
| load.cp_async#2*9 (est.: sampled iterations x9) | 1.08 | 4.5% | 1.0 | 1.15 | `for (int i_3 = 0; i_3 < 8; ++i_3)` |
| wait.cp_async#2 | 0.85 | 3.6% | 1.0 | 2.11 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| wait.sync*9 (est.: sampled iterations x9) | 0.56 | 2.3% | 1.0 | 0.58 | `__syncthreads();` |
| compute.mma#2 | 0.49 | 2.1% | 1.0 | 0.51 | `{` |
| store.global | 0.19 | 0.8% | 1.0 | 0.22 | `for (int i_10 = 0; i_10 < 16; ++i_10) \| for (int i_10 = 0; i_10 < 8; ++i_10) \| for (int i_` |
| load.cp_async | 0.17 | 0.7% | 1.0 | 0.22 | `for (int i = 0; i < 2; ++i) \| for (int i_1 = 0; i_1 < 2; ++i_1) \| for (int i_1 = 0; i_1 < ` |
| wait.sync#3 | 0.13 | 0.6% | 1.0 | 0.51 | `__syncthreads();` |

### mhc_finalize_kernel grid (40, 1, 1) block (128, 1, 1)

102 launches, 160 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 6.69 us, warp 5.97 us, nsys graphs 5.82 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 2.32 | 38.8% | 3.38 | 47.3% |
| load | 1.01 | 17.0% | 0.94 | 13.1% |
| store | 1.00 | 16.7% | 1.29 | 18.0% |
| compute | 0.99 | 16.6% | 0.60 | 8.3% |
| wait | 0.65 | 10.9% | 0.96 | 13.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync | 0.65 | 10.9% | 1.0 | 0.86 | `__syncthreads();` |
| compute.reduce | 0.57 | 9.6% | 0.2 | 2.30 | `row_max[0] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<32>>::run(row_max[0]);` |
| store.global#4*7 (est.: sampled iterations x7) | 0.56 | 9.4% | 0.8 | 0.90 | `if ((((int)threadIdx.x) % 4) == 0)` |
| compute.reduce#2*7 (est.: sampled iterations x7) | 0.42 | 7.0% | 0.8 | 0.67 | `amax[0] = max(amax[0], __shfl_xor_sync((uint)4294967295, amax[0], 1, 32));` |
| load.global#4*7 (est.: sampled iterations x7) | 0.36 | 6.1% | 0.8 | 0.67 | `*(uint4*)(yv + 0) = *(uint4*)(C + ((((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((i` |
| store.global#3*7 (est.: sampled iterations x7) | 0.32 | 5.4% | 0.8 | 0.45 | `*(uint4*)(y + ((((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((int64_t)step) * (int6` |
| load.global | 0.32 | 5.3% | 1.0 | 0.93 | `if (((int)threadIdx.x) < 32)` |
| load.global#3 | 0.24 | 4.0% | 0.8 | 0.32 | `for (int i_split_2 = 0; i_split_2 < 40; ++i_split_2)` |
| store.global | 0.10 | 1.7% | 0.2 | 0.42 | `if (((int)threadIdx.x) < 4)` |
| load.global#2 | 0.09 | 1.6% | 0.2 | 0.38 | `cm[0] = ((mixes_shared[((((int)threadIdx.x) & 15) + 8)] * scale[2]) + base[((((int)threadI` |
| store.global#2 | 0.01 | 0.2% | 0.2 | 0.06 | `if ((((int)threadIdx.x) >> 4) == 0)` |

### moe_dispatch_kernel grid (1, 1, 1) block (512, 1, 1)

153 launches, 16 warps; ranges, warp and last warp from the 1 recording CTAs' 16 warps; span 4.29 us, warp 4.07 us, nsys graphs 2.62 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 2.18 | 53.6% | 1.82 | 42.3% |
| unattributed, with loops' unsampled iterations | 1.29 | 31.6% | 1.35 | 31.4% |
| store | 0.52 | 12.7% | 1.05 | 24.4% |
| load | 0.08 | 2.0% | 0.09 | 2.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync#7*9 (est.: sampled iterations x9) | 0.67 | 16.4% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#6*9 (est.: sampled iterations x9) | 0.63 | 15.5% | 1.0 | 0.86 | `__syncthreads();` |
| store.global (sampled iterations) | 0.41 | 10.0% | 1.0 | 0.70 | `ChunkCount[((((int64_t)c) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = count; \| Chu` |
| wait.sync#8 | 0.37 | 9.0% | 1.0 | 0.54 | `__syncthreads();` |
| wait.sync#3 | 0.14 | 3.3% | 1.0 | 0.35 | `__syncthreads();` |
| wait.sync#2 | 0.13 | 3.1% | 1.0 | 0.19 | `__syncthreads();` |
| store.global#2 | 0.11 | 2.7% | 1.0 | 0.26 | `RoutePos[((((int64_t)((int)blockIdx.x)) * (int64_t)512) + ((int64_t)((int)threadIdx.x)))] ` |
| load.global (sampled iterations) | 0.08 | 2.0% | 1.0 | 0.10 | `expert = ((int)TopkIdx[((((int64_t)((int)blockIdx.x)) * (int64_t)512) + ((int64_t)((int)th` |
| wait.sync | 0.07 | 1.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#4 | 0.06 | 1.6% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#5 | 0.06 | 1.6% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#9 | 0.06 | 1.5% | 1.0 | 0.06 | `__syncthreads();` |

### gemm_mxfp8_partials_kernel grid (16, 5, 1) block (128, 1, 1)

51 launches, 320 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 12.38 us, warp 11.63 us, nsys graphs 13.60 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 6.19 | 53.2% | 6.38 | 50.8% |
| wait | 4.32 | 37.2% | 4.02 | 32.0% |
| compute | 0.61 | 5.3% | 0.69 | 5.5% |
| load | 0.45 | 3.9% | 1.43 | 11.4% |
| store | 0.05 | 0.5% | 0.05 | 0.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.cp_async#3 | 1.43 | 12.3% | 1.0 | 3.46 | `tl::cp_async_wait<0>();` |
| wait.cp_async#2 | 1.08 | 9.3% | 1.0 | 2.43 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>();` |
| wait.sync#2*2 (est.: sampled iterations x2) | 0.48 | 4.2% | 1.0 | 1.86 | `__syncthreads();` |
| wait.sync#3 | 0.48 | 4.1% | 1.0 | 1.63 | `__syncthreads();` |
| wait.sync#4 | 0.45 | 3.9% | 1.0 | 1.57 | `__syncthreads();` |
| wait.cp_async*2 (est.: sampled iterations x2) | 0.31 | 2.6% | 1.0 | 2.11 | `tl::cp_async_wait<2>();` |
| compute.mma*2 (est.: sampled iterations x2) | 0.30 | 2.6% | 1.0 | 0.32 | `{` |
| load.cp_async | 0.23 | 2.0% | 1.0 | 0.32 | `for (int i = 0; i < 2; ++i) \| for (int i_1 = 0; i_1 < 4; ++i_1)` |
| load.cp_async#2*2 (est.: sampled iterations x2) | 0.22 | 1.9% | 1.0 | 0.26 | `for (int i_4 = 0; i_4 < 2; ++i_4)` |
| compute.mma#2 | 0.15 | 1.3% | 1.0 | 0.16 | `{` |
| compute.mma#3 | 0.15 | 1.3% | 1.0 | 0.19 | `{` |
| wait.sync*2 (est.: sampled iterations x2) | 0.09 | 0.8% | 1.0 | 0.13 | `__syncthreads();` |
| store.global | 0.05 | 0.5% | 1.0 | 0.06 | `for (int i_11 = 0; i_11 < 16; ++i_11) \| for (int i_13 = 0; i_13 < 8; ++i_13) \| for (int i_` |
| compute.mma*6 (est.: sampled iterations x6) | 0.00 | 0.0% | 0.0 | 0.00 | `{` |
| load.cp_async#2*6 (est.: sampled iterations x6) | 0.00 | 0.0% | 0.0 | 0.00 | `for (int i_5 = 0; i_5 < 4; ++i_5)` |
| wait.cp_async*6 (est.: sampled iterations x6) | 0.00 | 0.0% | 0.0 | 0.00 | `tl::cp_async_wait<2>();` |
| wait.sync#2*6 (est.: sampled iterations x6) | 0.00 | 0.0% | 0.0 | 0.00 | `__syncthreads();` |
| wait.sync*6 (est.: sampled iterations x6) | 0.00 | 0.0% | 0.0 | 0.00 | `__syncthreads();` |

### metadata_decode_kernel grid (48, 2, 1) block (128, 1, 1)

68 launches, 384 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 2.26 us, warp 1.30 us, nsys graphs 2.10 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.66 | 50.5% | 5.16 | 62.1% |
| unattributed | 0.56 | 43.2% | 2.22 | 26.7% |
| store | 0.08 | 6.3% | 0.93 | 11.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.41 | 31.8% | 0.5 | 1.34 | `if ((((offset + ((int)blockIdx.x)) < src_rows) && (0 <= (offset + ((int)blockIdx.x)))))` |
| load.global#3 | 0.23 | 18.0% | 0.5 | 0.51 | `if ((((offset + ((int)blockIdx.x)) < main_src_rows) && (0 <= (offset + ((int)blockIdx.x)))` |
| store.global#4 | 0.04 | 3.1% | 0.5 | 0.22 | `if (0 < live)` |
| store.global | 0.03 | 2.3% | 0.5 | 0.06 | `swa[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = con` |
| load.global#2 | 0.01 | 0.6% | 0.1 | 0.06 | `if (((0 <= req) & ((int64_t)0 <= pos)))` |
| store.global#2 | 0.01 | 0.5% | 0.1 | 0.06 | `lengths[((int64_t)((int)blockIdx.x))] = ((int)max(info[1], (int64_t)0));` |
| store.global#3 | 0.01 | 0.5% | 0.1 | 0.06 | `top_lengths[((int64_t)((int)blockIdx.x))] = condval_6; \| top_lengths[((int64_t)((int)block` |

### rope_kernel grid (3, 1, 1) block (128, 1, 1)

224 launches, 12 warps; ranges, warp and last warp from the 3 recording CTAs' 12 warps; span 1.25 us, warp 1.21 us, nsys graphs 1.31 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.99 | 81.3% | 2.10 | 79.6% |
| load | 0.19 | 15.8% | 0.47 | 17.8% |
| store | 0.03 | 2.9% | 0.07 | 2.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.10 | 7.9% | 1.0 | 0.10 | `if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < tokens)) \| int64_t position ` |
| load.global#2 | 0.10 | 7.9% | 1.0 | 0.10 | `if (((position < ((int64_t)positions_max)) && ((int64_t)0 <= position)))` |
| store.global | 0.03 | 2.9% | 1.0 | 0.06 | `*(uint4*)(x + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + ((((int64_t)((int)thread` |

### local_stats_kernel grid (16, 16, 1) block (128, 1, 1)

16 launches, 1024 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 35.17 us, warp 33.31 us, nsys graphs 35.89 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 32.20 | 96.7% | 33.10 | 95.8% |
| unattributed | 0.74 | 2.2% | 0.80 | 2.3% |
| compute | 0.19 | 0.6% | 0.17 | 0.5% |
| wait | 0.11 | 0.3% | 0.39 | 1.1% |
| store | 0.07 | 0.2% | 0.08 | 0.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 32.20 | 96.7% | 1.0 | 33.17 | `if (expanded_local_pos[((int64_t)((int)blockIdx.x))] < num_speculative_steps)` |
| compute.reduce | 0.13 | 0.4% | 1.0 | 0.16 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| wait.sync | 0.11 | 0.3% | 1.0 | 0.46 | `__syncthreads();` |
| store.global | 0.07 | 0.2% | 1.0 | 0.10 | `if (((int)threadIdx.x) == 0)` |
| compute.reduce#2 | 0.06 | 0.2% | 1.0 | 0.10 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 2, 32);` |

### mhc_collapse_kernel grid (240, 1, 1) block (128, 1, 1)

51 launches, 960 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 10.62 us, warp 4.92 us, nsys graphs 9.76 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 3.97 | 80.7% | 3.75 | 59.1% |
| load | 0.89 | 18.1% | 2.45 | 38.6% |
| store | 0.06 | 1.2% | 0.14 | 2.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.89 | 18.1% | 1.0 | 2.94 | `*(float4*)(mv + 0) = *(float4*)(mix + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int6` |
| store.global | 0.06 | 1.2% | 1.0 | 0.10 | `*(uint4*)(out + ((((int64_t)((int)blockIdx.x)) * (int64_t)1024) + (((int64_t)((int)threadI` |

### router_partials_kernel grid (24, 1, 1) block (64, 1, 1)

102 launches, 48 warps; ranges, warp and last warp from the 4 recording CTAs' 8 warps; span 5.12 us, warp 3.19 us, nsys graphs 4.70 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.31 | 41.2% | 1.98 | 36.6% |
| load | 1.06 | 33.1% | 1.87 | 34.7% |
| compute | 0.67 | 21.1% | 1.35 | 25.0% |
| store | 0.14 | 4.5% | 0.20 | 3.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.reduce | 0.67 | 21.1% | 0.5 | 1.34 | `for (int k = 0; k < 3; ++k) \| for (int k = 0; k < 6; ++k)` |
| load.global | 0.48 | 14.9% | 1.0 | 0.58 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < mask_len))` |
| load.global#4 | 0.45 | 14.2% | 0.5 | 1.02 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < image_len)) \| if ((((((int)bl` |
| load.global#2 | 0.06 | 2.0% | 1.0 | 0.06 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < mask_len))` |
| load.global#3 | 0.06 | 2.0% | 1.0 | 0.06 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < mask_len))` |
| store.global | 0.05 | 1.5% | 1.0 | 0.06 | `if (condval == (bool)0) \| if (condval == (bool)1)` |
| store.global#2 | 0.05 | 1.5% | 1.0 | 0.06 | `if (condval_1 == (bool)0) \| if (condval_1 == (bool)1)` |
| store.global#3 | 0.05 | 1.5% | 0.5 | 0.10 | `if ((((int)threadIdx.x) & 31) < 3) \| if ((((int)threadIdx.x) & 31) < 6)` |

### metadata_builder_kernel grid (2, 1, 1) block (128, 1, 1)

289 launches, 8 warps; ranges, warp and last warp from the 2 recording CTAs' 8 warps; span 1.54 us, warp 0.83 us, nsys graphs 1.54 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.46 | 54.8% | 1.33 | 75.2% |
| unattributed | 0.30 | 36.5% | 0.38 | 21.5% |
| store | 0.07 | 8.7% | 0.06 | 3.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#3 | 0.42 | 50.0% | 0.5 | 1.15 | `for (int __1 = 0; __1 < 4; ++__1)` |
| load.global#2 | 0.03 | 3.4% | 0.1 | 0.22 | `if ((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) < nr))` |
| store.global#3 | 0.03 | 3.4% | 0.5 | 0.06 | `if (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) == 0)` |
| store.global#4 | 0.03 | 3.4% | 0.5 | 0.06 | `positions[(((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))` |
| load.global | 0.01 | 1.4% | 0.1 | 0.10 | `if ((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) <= nr))` |
| store.global | 0.01 | 1.4% | 0.1 | 0.10 | `out_starts[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] = condval;` |
| store.global#2 | 0.00 | 0.5% | 0.1 | 0.03 | `request_positions[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] = condval_2;` |

### moe_combine_kernel grid (5, 48, 1) block (128, 1, 1)

102 launches, 960 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 4.16 us, warp 3.12 us, nsys graphs 2.62 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 2.40 | 77.0% | 3.58 | 85.9% |
| unattributed | 0.49 | 15.7% | 0.54 | 12.9% |
| store | 0.23 | 7.2% | 0.05 | 1.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 2.40 | 77.0% | 1.0 | 3.50 | `for (int k = 0; k < 3; ++k) \| for (int k = 0; k < 6; ++k) \| if ((((int)blockIdx.y) < share` |
| store.global | 0.23 | 7.2% | 1.0 | 0.67 | `{` |

### token_ranks_kernel grid (8, 1, 1) block (256, 1, 1)

51 launches, 64 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 7.81 us, warp 7.71 us, nsys graphs 9.02 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 6.95 | 90.1% | 7.18 | 89.6% |
| unattributed | 0.36 | 4.7% | 0.37 | 4.6% |
| compute | 0.36 | 4.6% | 0.41 | 5.2% |
| store | 0.04 | 0.6% | 0.05 | 0.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 6.95 | 90.1% | 1.0 | 7.23 | `if ((((int64_t)0 <= token_ids[((int64_t)((int)blockIdx.x))]) && (token_ids[((int64_t)((int` |
| compute.reduce | 0.36 | 4.6% | 1.0 | 0.67 | `total[0] = tl::AllReduce<tl::SumOp, 256, 1, 0, tl::NamedBarrier<256>>::run(total[0], (&(wo` |
| store.global | 0.04 | 0.6% | 1.0 | 0.06 | `output[((int64_t)((int)blockIdx.x))] = ((int64_t)total[0]);` |

### indexer_topk_segment_kernel grid (1, 48, 1) block (512, 1, 1)

34 launches, 768 warps; ranges, warp and last warp from the 4 recording CTAs' 64 warps; span 10.94 us, warp 10.69 us, nsys graphs 6.29 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed, with loops' unsampled iterations | 5.68 | 53.1% | 5.44 | 50.7% |
| wait | 3.63 | 34.0% | 3.85 | 35.9% |
| load | 1.23 | 11.5% | 1.27 | 11.9% |
| store | 0.15 | 1.4% | 0.17 | 1.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.23 | 11.5% | 1.0 | 1.46 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 4096); \| i` |
| wait.sync (sampled iterations) | 0.98 | 9.1% | 14.5 | 1.15 | `__syncthreads();` |
| wait.sync#24*9 (est.: sampled iterations x9) | 0.31 | 2.9% | 0.5 | 0.43 | `__syncthreads();` |
| wait.sync#20*9 (est.: sampled iterations x9) | 0.30 | 2.8% | 0.5 | 0.43 | `__syncthreads();` |
| wait.sync#19*9 (est.: sampled iterations x9) | 0.30 | 2.8% | 0.5 | 0.43 | `__syncthreads();` |
| wait.sync#23*9 (est.: sampled iterations x9) | 0.29 | 2.7% | 0.5 | 0.29 | `__syncthreads();` |
| wait.sync#14*8 (est.: sampled iterations x8) | 0.28 | 2.6% | 0.5 | 0.38 | `__syncthreads();` |
| wait.sync#6*8 (est.: sampled iterations x8) | 0.28 | 2.6% | 0.5 | 0.38 | `__syncthreads();` |
| wait.sync#5*8 (est.: sampled iterations x8) | 0.26 | 2.5% | 0.5 | 0.38 | `__syncthreads();` |
| wait.sync#13*8 (est.: sampled iterations x8) | 0.18 | 1.7% | 0.5 | 0.26 | `__syncthreads();` |
| store.global | 0.15 | 1.4% | 1.5 | 0.19 | `for (int i_3 = 0; i_3 < 4; ++i_3) \| if (((int)threadIdx.x) < total) \| if (((int)threadIdx.` |
| wait.sync#25 | 0.04 | 0.3% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#8 | 0.04 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#18 | 0.03 | 0.3% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#16 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#17 | 0.03 | 0.3% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#3 | 0.03 | 0.3% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#9 | 0.03 | 0.3% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#2 | 0.03 | 0.3% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#7 | 0.03 | 0.3% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#15 | 0.03 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#11 | 0.03 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#22 | 0.03 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#12 | 0.03 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#21 | 0.03 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#4 | 0.02 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#10 | 0.02 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |

### norm_forward_and_per_token_cast_kernel_kernel grid (6, 1, 1) block (128, 1, 1)

105 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 2.91 us, warp 2.91 us, nsys graphs 1.54 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 2.45 | 84.3% | 2.64 | 78.3% |
| load | 0.45 | 15.3% | 0.44 | 12.9% |
| compute | 0.11 | 3.8% | 0.12 | 3.5% |
| wait | 0.05 | 1.8% | 0.05 | 1.5% |
| unattributed | 0.00 | 0.0% | 0.12 | 3.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*4 (est.: sampled iterations x4) | 1.78 | 61.2% | 1.0 | 1.79 | `for (int j_2 = 0; j_2 < 2; ++j_2)` |
| store.global*4 (est.: sampled iterations x4) | 0.41 | 14.1% | 1.0 | 0.51 | `for (int j_1 = 0; j_1 < 4; ++j_1)` |
| load.global*4 (est.: sampled iterations x4) | 0.23 | 8.1% | 1.0 | 0.26 | `if (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 4)) < num_tokens)` |
| load.global#3*4 (est.: sampled iterations x4) | 0.18 | 6.0% | 1.0 | 0.26 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 128) + ((((int)threadIdx.x) & 15` |
| store.global#5*4 (est.: sampled iterations x4) | 0.17 | 5.7% | 1.0 | 0.26 | `*(uint4*)(out + ((((((int64_t)vi_1) * (int64_t)128) + ((((int64_t)((int)threadIdx.x)) & (i` |
| compute.reduce | 0.11 | 3.8% | 1.0 | 0.13 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| store.global#3 | 0.05 | 1.8% | 1.0 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |
| wait.sync | 0.05 | 1.8% | 1.0 | 0.10 | `__syncthreads();` |
| store.global#2 | 0.05 | 1.6% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| load.global#2 | 0.04 | 1.2% | 1.0 | 0.06 | `*(float2*)(row_partial + 0) = *(float2*)(sum_shared + ((((int)threadIdx.x) >> 6) * 2)); \| ` |

### indexer_score_kernel grid (128, 48, 1) block (128, 1, 1)

17 launches, 24576 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 20.74 us, warp 2.54 us, nsys graphs 11.33 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.40 | 55.4% | 0.59 | 63.6% |
| unattributed, with loops' unsampled iterations | 0.78 | 30.6% | 0.30 | 33.1% |
| wait | 0.25 | 10.0% | 0.02 | 1.8% |
| compute | 0.08 | 3.0% | 0.01 | 1.0% |
| store | 0.03 | 1.0% | 0.00 | 0.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 (sampled iterations) | 0.86 | 33.9% | 0.5 | 1.86 | `for (int i_1 = 0; i_1 < 32; ++i_1)` |
| load.global | 0.39 | 15.3% | 1.0 | 0.64 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 4096); \| i` |
| load.global#3 (sampled iterations) | 0.16 | 6.2% | 0.5 | 0.64 | `if (((int)threadIdx.x) < 64)` |
| wait.sync#2 (sampled iterations) | 0.15 | 5.9% | 0.5 | 0.64 | `__syncthreads();` |
| compute.mma (sampled iterations) | 0.08 | 3.0% | 0.5 | 0.16 | `for (int ki = 0; ki < 2; ++ki)` |
| wait.sync (sampled iterations) | 0.04 | 1.4% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#4 (sampled iterations) | 0.04 | 1.4% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#3 (sampled iterations) | 0.03 | 1.3% | 0.5 | 0.10 | `__syncthreads();` |
| store.global (sampled iterations) | 0.03 | 1.0% | 0.5 | 0.06 | `if ((((int)threadIdx.x) >> 6) == 0)` |

### topk_ids_kernel grid (16, 1, 1) block (512, 1, 1)

16 launches, 256 warps; ranges, warp and last warp from the 4 recording CTAs' 64 warps; span 20.82 us, warp 18.99 us, nsys graphs 21.71 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 15.98 | 84.2% | 17.05 | 78.6% |
| unattributed | 1.93 | 10.2% | 1.94 | 8.9% |
| wait | 0.86 | 4.6% | 0.68 | 3.1% |
| store | 0.21 | 1.1% | 2.04 | 9.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 15.98 | 84.2% | 1.0 | 17.15 | `for (int t = 0; t < (vocab >> 14); ++t)` |
| wait.sync | 0.73 | 3.8% | 1.0 | 1.79 | `__syncthreads();` |
| store.global | 0.21 | 1.1% | 1.0 | 2.50 | `if ((((int)threadIdx.x) >> 5) == 0)` |
| wait.sync#2 | 0.08 | 0.4% | 1.0 | 0.16 | `__syncthreads();` |
| wait.sync#3 | 0.06 | 0.3% | 1.0 | 0.06 | `__syncthreads();` |

### moe_swiglu_cast_kernel_kernel grid (8, 120, 1) block (288, 1, 1)

51 launches, 8640 warps; ranges, warp and last warp from the 4 recording CTAs' 36 warps; span 6.43 us, warp 1.40 us, nsys graphs 3.62 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.77 | 54.6% | 0.40 | 39.4% |
| load | 0.42 | 30.2% | 0.52 | 51.1% |
| compute | 0.13 | 9.1% | 0.09 | 8.3% |
| store | 0.09 | 6.1% | 0.01 | 1.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.42 | 30.2% | 1.0 | 0.80 | `for (int r = 0; r < 4; ++r)` |
| compute.reduce*4 (est.: sampled iterations x4) | 0.13 | 9.1% | 0.2 | 0.64 | `if (0 <= orders[r_2])` |
| store.global*4 (est.: sampled iterations x4) | 0.09 | 6.1% | 0.2 | 0.38 | `if (0 <= orders[r_2])` |

### topk_log_softmax_kernel grid (16, 1, 1) block (1024, 1, 1)

16 launches, 512 warps; ranges, warp and last warp from the 4 recording CTAs' 128 warps; span 20.98 us, warp 19.20 us, nsys graphs 18.42 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 15.86 | 82.6% | 17.55 | 84.9% |
| unattributed, with loops' unsampled iterations | 1.67 | 8.7% | 1.72 | 8.3% |
| wait | 1.27 | 6.6% | 0.96 | 4.7% |
| compute | 0.39 | 2.0% | 0.38 | 1.8% |
| store | 0.00 | 0.0% | 0.06 | 0.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 8.41 | 43.8% | 1.0 | 9.87 | `if (0 < (vocab & 2047)) \| if (0 < (vocab & 8191))` |
| load.global#2 | 7.44 | 38.8% | 1.0 | 8.18 | `for (int step_1 = 0; step_1 < (vocab >> 11); ++step_1) \| for (int step_1 = 0; step_1 < (vo` |
| wait.sync#2 | 0.52 | 2.7% | 1.0 | 1.17 | `__syncthreads();` |
| wait.sync | 0.50 | 2.6% | 1.0 | 1.49 | `__syncthreads();` |
| compute.reduce#3 | 0.17 | 0.9% | 1.0 | 0.22 | `ties[0] = vllm_add(ties[0], __shfl_xor_sync((uint)4294967295, ties[0], 16, 32));` |
| compute.reduce#2 | 0.11 | 0.6% | 1.0 | 0.16 | `se[0] = vllm_add(se[0], __shfl_xor_sync((uint)4294967295, se[0], 16, 32));` |
| compute.reduce | 0.10 | 0.5% | 1.0 | 0.13 | `best[0] = vllm_max(best[0], __shfl_xor_sync((uint)4294967295, best[0], 16, 32));` |
| wait.sync#4 | 0.10 | 0.5% | 1.0 | 0.16 | `__syncthreads();` |
| wait.sync#5 | 0.08 | 0.4% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#3 | 0.08 | 0.4% | 1.0 | 0.10 | `__syncthreads();` |
| load.global#3 (sampled iterations) | 0.01 | 0.1% | 0.0 | 0.38 | `float value_32 = vllm_sub(vllm_shifted(logits[((((int64_t)((int)blockIdx.x)) * logits_stri` |
| store.global (sampled iterations) | 0.00 | 0.0% | 0.0 | 0.06 | `output[(((((int64_t)c) * (int64_t)1024) + (((int64_t)((int)blockIdx.x)) * ((int64_t)topk))` |

### topk_log_softmax_kernel grid (8, 1, 1) block (1024, 1, 1)

17 launches, 256 warps; ranges, warp and last warp from the 4 recording CTAs' 128 warps; span 17.86 us, warp 16.90 us, nsys graphs 16.45 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 13.83 | 81.8% | 14.78 | 82.7% |
| unattributed, with loops' unsampled iterations | 1.59 | 9.4% | 1.76 | 9.9% |
| wait | 1.09 | 6.4% | 0.89 | 5.0% |
| compute | 0.39 | 2.3% | 0.38 | 2.1% |
| store | 0.00 | 0.0% | 0.07 | 0.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 7.12 | 42.1% | 1.0 | 7.55 | `if (0 < (vocab & 2047)) \| if (0 < (vocab & 8191))` |
| load.global#2 | 6.70 | 39.6% | 1.0 | 7.39 | `for (int step_1 = 0; step_1 < (vocab >> 11); ++step_1) \| for (int step_1 = 0; step_1 < (vo` |
| wait.sync#2 | 0.51 | 3.0% | 1.0 | 1.09 | `__syncthreads();` |
| wait.sync | 0.32 | 1.9% | 1.0 | 0.90 | `__syncthreads();` |
| compute.reduce#3 | 0.17 | 1.0% | 1.0 | 0.22 | `ties[0] = vllm_add(ties[0], __shfl_xor_sync((uint)4294967295, ties[0], 16, 32));` |
| compute.reduce#2 | 0.11 | 0.7% | 1.0 | 0.16 | `se[0] = vllm_add(se[0], __shfl_xor_sync((uint)4294967295, se[0], 16, 32));` |
| compute.reduce | 0.11 | 0.6% | 1.0 | 0.13 | `best[0] = vllm_max(best[0], __shfl_xor_sync((uint)4294967295, best[0], 16, 32));` |
| wait.sync#4 | 0.10 | 0.6% | 1.0 | 0.16 | `__syncthreads();` |
| wait.sync#5 | 0.08 | 0.5% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#3 | 0.08 | 0.5% | 1.0 | 0.10 | `__syncthreads();` |
| load.global#3 (sampled iterations) | 0.01 | 0.1% | 0.0 | 0.38 | `float value_32 = vllm_sub(vllm_shifted(logits[((((int64_t)((int)blockIdx.x)) * logits_stri` |
| store.global (sampled iterations) | 0.00 | 0.0% | 0.0 | 0.06 | `output[(((((int64_t)c) * (int64_t)1024) + (((int64_t)((int)blockIdx.x)) * ((int64_t)topk))` |

### norm_forward_and_per_token_cast_kernel_kernel grid (24, 1, 1) block (128, 1, 1)

102 launches, 96 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 2.50 us, warp 2.30 us, nsys graphs 1.92 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 1.58 | 68.4% | 2.08 | 72.0% |
| load | 0.35 | 15.0% | 0.34 | 11.6% |
| unattributed | 0.15 | 6.6% | 0.31 | 10.7% |
| compute | 0.12 | 5.1% | 0.12 | 4.0% |
| wait | 0.11 | 4.9% | 0.05 | 1.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*3 (est.: sampled iterations x3) | 1.06 | 46.2% | 1.0 | 1.15 | `for (int j_2 = 0; j_2 < 2; ++j_2)` |
| store.global*3 (est.: sampled iterations x3) | 0.31 | 13.3% | 1.0 | 0.38 | `for (int j_1 = 0; j_1 < 4; ++j_1)` |
| load.global*3 (est.: sampled iterations x3) | 0.18 | 7.8% | 1.0 | 0.19 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 6)) < num_tokens) && (((vi * 2) + (` |
| load.global#3*3 (est.: sampled iterations x3) | 0.13 | 5.7% | 1.0 | 0.19 | `if (((vi_1 * 2) + ((((int)threadIdx.x) & 63) >> 5)) < 5)` |
| store.global#5*3 (est.: sampled iterations x3) | 0.13 | 5.7% | 1.0 | 0.19 | `*(uint4*)(out + ((((((int64_t)vi_1) * (int64_t)512) + ((((int64_t)((int)threadIdx.x)) & (i` |
| compute.reduce | 0.12 | 5.1% | 1.0 | 0.13 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| wait.sync | 0.11 | 4.9% | 1.0 | 0.19 | `__syncthreads();` |
| store.global#2 | 0.04 | 1.8% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| load.global#2 | 0.03 | 1.5% | 1.0 | 0.06 | `*(float2*)(row_partial + 0) = *(float2*)(sum_shared + ((((int)threadIdx.x) >> 6) * 2)); \| ` |
| store.global#3 | 0.03 | 1.4% | 0.5 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |

### sink_rows_kernel grid (8, 64, 1) block (256, 1, 1)

17 launches, 4096 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 15.87 us, warp 4.77 us, nsys graphs 7.42 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 2.03 | 42.5% | 3.40 | 44.6% |
| unattributed | 1.65 | 34.7% | 1.12 | 14.7% |
| load | 1.09 | 22.8% | 3.12 | 40.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 2.00 | 42.0% | 1.0 | 5.44 | `for (int k_1 = 0; k_1 < 8; ++k_1)` |
| load.global | 1.09 | 22.8% | 1.0 | 4.26 | `row = (max((((int64_t)num_sampled[((int64_t)((int)blockIdx.x))]) - (int64_t)1), (int64_t)0` |
| store.global#2 | 0.02 | 0.5% | 0.0 | 0.77 | `tokens[((((int64_t)((int)blockIdx.x)) * (int64_t)3) + ((int64_t)((int)threadIdx.x)))] = to` |
| load.global#2 | 0.00 | 0.1% | 0.0 | 0.10 | `token = max(alternates[((((int64_t)((int)blockIdx.x)) * (int64_t)3) + ((int64_t)((int)thre` |

### metadata_chunk_kernel grid (48, 1, 1) block (128, 1, 1)

34 launches, 192 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 4.05 us, warp 3.25 us, nsys graphs 8.40 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 2.25 | 69.2% | 2.26 | 31.5% |
| load | 0.92 | 28.2% | 4.77 | 66.6% |
| store | 0.08 | 2.6% | 0.14 | 1.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.85 | 26.3% | 1.0 | 1.23 | `if ((((offset + ((int)blockIdx.x)) < src_rows) && (0 <= (offset + ((int)blockIdx.x)))))` |
| load.global#2 | 0.06 | 1.9% | 0.2 | 0.50 | `if (((0 <= req) & ((int64_t)0 <= pos)))` |
| store.global | 0.06 | 1.9% | 1.0 | 0.10 | `swa[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = con` |
| store.global#2 | 0.01 | 0.4% | 0.2 | 0.06 | `lengths[((int64_t)((int)blockIdx.x))] = ((int)max(info[1], (int64_t)0));` |
| store.global#3 | 0.01 | 0.3% | 0.2 | 0.06 | `top_lengths[((int64_t)((int)blockIdx.x))] = condval_12; \| top_lengths[((int64_t)((int)bloc` |

### sink_candidates_kernel grid (8, 127, 1) block (128, 1, 1)

17 launches, 4064 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 11.94 us, warp 4.81 us, nsys graphs 9.76 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.72 | 35.7% | 1.12 | 24.7% |
| compute | 1.44 | 29.9% | 1.33 | 29.2% |
| load | 0.76 | 15.7% | 1.20 | 26.4% |
| wait | 0.60 | 12.5% | 0.53 | 11.6% |
| store | 0.30 | 6.1% | 0.36 | 8.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.reduce*4 (est.: sampled iterations x4) | 1.05 | 21.8% | 1.0 | 1.28 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| load.global | 0.76 | 15.7% | 1.0 | 1.12 | `local[0] = max((num_sampled[((int64_t)((int)blockIdx.x))] - 1), 0);` |
| compute.reduce#2*4 (est.: sampled iterations x4) | 0.39 | 8.1% | 1.0 | 0.51 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 2, 32);` |
| wait.sync#2*4 (est.: sampled iterations x4) | 0.30 | 6.3% | 1.0 | 0.51 | `__syncthreads();` |
| store.global*4 (est.: sampled iterations x4) | 0.30 | 6.1% | 1.0 | 0.38 | `if (((int)threadIdx.x) == 0)` |
| wait.sync*4 (est.: sampled iterations x4) | 0.30 | 6.1% | 1.0 | 0.51 | `__syncthreads();` |

### sparse_mla_kernel_1 grid (16, 48, 1) block (128, 1, 1)

34 launches, 768 CTAs, 3072 warps; span 6.37 us, warp 3.54 us, nsys graphs 7.65 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 3.53 | 99.7% | 1.83 | 92.0% |
| load | 0.01 | 0.2% | 0.08 | 3.9% |
| wait | 0.00 | 0.0% | 0.01 | 0.7% |
| compute | 0.00 | 0.0% | 0.06 | 3.2% |
| store | 0.00 | 0.0% | 0.01 | 0.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#3 | 0.01 | 0.2% | 0.0 | 1.58 | `M = __fmul_rn(Sinks[((int)blockIdx.x)], 0x1.7154764ee6c2fp+0f/*1.442695e+00*/);` |
| wait.sync#13 | 0.00 | 0.0% | 0.0 | 0.35 | `__syncthreads();` |
| compute.reduce#3 | 0.00 | 0.0% | 0.0 | 0.19 | `amax[0] = tl::AllReduce<tl::MaxOp, 8, 1, 0, tl::NamedBarrier<128>>::run(amax[0]);` |
| load.global#4 | 0.00 | 0.0% | 0.0 | 0.35 | `if (((int)threadIdx.x) < 32)` |
| wait.sync#12 | 0.00 | 0.0% | 0.0 | 0.13 | `__syncthreads();` |
| store.global#2 | 0.00 | 0.0% | 0.0 | 0.10 | `*(uint2*)(Output + (((((int64_t)((int)blockIdx.y)) * (int64_t)8192) + (((int64_t)((int)blo` |

### sparse_mla_kernel_1 grid (16, 40, 1) block (128, 1, 1)

51 launches, 640 CTAs, 2560 warps; span 4.13 us, warp 2.72 us, nsys graphs 3.65 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 2.71 | 99.5% | 1.53 | 89.7% |
| load | 0.01 | 0.4% | 0.16 | 9.2% |
| wait | 0.00 | 0.1% | 0.01 | 0.8% |
| compute | 0.00 | 0.0% | 0.00 | 0.2% |
| store | 0.00 | 0.0% | 0.00 | 0.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#3 | 0.01 | 0.4% | 0.0 | 2.08 | `M = __fmul_rn(Sinks[((int)blockIdx.x)], 0x1.7154764ee6c2fp+0f/*1.442695e+00*/);` |
| wait.sync#13 | 0.00 | 0.1% | 0.0 | 0.80 | `__syncthreads();` |
| load.global#4 | 0.00 | 0.0% | 0.0 | 0.80 | `if (((int)threadIdx.x) < 32)` |
| compute.reduce#3 | 0.00 | 0.0% | 0.0 | 0.19 | `amax[0] = tl::AllReduce<tl::MaxOp, 8, 1, 0, tl::NamedBarrier<128>>::run(amax[0]);` |
| wait.sync#12 | 0.00 | 0.0% | 0.0 | 0.29 | `__syncthreads();` |
| store.global#2 | 0.00 | 0.0% | 0.0 | 0.10 | `*(uint2*)(Output + (((((int64_t)((int)blockIdx.y)) * (int64_t)8192) + (((int64_t)((int)blo` |

### compress_kernel grid (12, 1, 1) block (128, 1, 1)

34 launches, 48 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 6.21 us, warp 4.85 us, nsys graphs 5.97 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 2.37 | 48.9% | 1.42 | 25.2% |
| load | 1.94 | 39.9% | 3.18 | 56.5% |
| store | 0.45 | 9.2% | 0.93 | 16.6% |
| compute | 0.10 | 2.0% | 0.09 | 1.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.20 | 24.7% | 1.0 | 1.50 | `if ((((0 <= counts[0]) & (0 < counts[1])) & (counts[1] <= 8)) & (((((int)blockIdx.x) * 4) ` |
| load.global#2*4 (est.: sampled iterations x4) | 0.63 | 13.0% | 1.0 | 1.34 | `*(float4*)(v1 + 0) = *(float4*)(values + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048)` |
| store.global*4 (est.: sampled iterations x4) | 0.15 | 3.1% | 1.0 | 0.32 | `*(uint2*)(out + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + ((((int64_t)((int)thre` |
| compute.reduce | 0.10 | 2.0% | 1.0 | 0.10 | `float total = tl::warp_reduce_sum(ss[0]);` |
| store.global#3 | 0.09 | 1.8% | 0.5 | 0.16 | `emitted_slots[((((int64_t)((int)blockIdx.x)) * (int64_t)4) + (((int64_t)((int)threadIdx.x)` |
| load.global#3*4 (est.: sampled iterations x4) | 0.07 | 1.4% | 0.5 | 0.13 | `*(float4*)(w + 0) = *(float4*)(weight + ((c_1 * 128) + ((((int)threadIdx.x) & 31) * 4)));` |
| store.global#3*4 (est.: sampled iterations x4) | 0.07 | 1.4% | 0.5 | 0.13 | `*(uint2*)(out + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + ((((int64_t)((int)thre` |
| store.global#2*4 (est.: sampled iterations x4) | 0.06 | 1.2% | 0.2 | 0.13 | `if ((slot >> (int64_t)3) < ((int64_t)blocks))` |
| store.global#5 | 0.05 | 0.9% | 0.5 | 0.16 | `emitted_slots[((((int64_t)((int)blockIdx.x)) * (int64_t)4) + (((int64_t)((int)threadIdx.x)` |
| load.global#3 | 0.02 | 0.5% | 0.5 | 0.03 | `if (valid)` |
| load.global#4 | 0.02 | 0.4% | 0.5 | 0.03 | `if (valid)` |
| store.global#2 | 0.02 | 0.3% | 0.5 | 0.02 | `emitted[((((int64_t)((int)blockIdx.x)) * (int64_t)4) + (((int64_t)((int)threadIdx.x)) >> (` |
| store.global#4 | 0.02 | 0.3% | 0.5 | 0.02 | `emitted[((((int64_t)((int)blockIdx.x)) * (int64_t)4) + (((int64_t)((int)threadIdx.x)) >> (` |

### per_token_cast_kernel_kernel grid (18, 1, 1) block (64, 1, 1)

153 launches, 36 warps; ranges, warp and last warp from the 5 recording CTAs' 10 warps; span 1.18 us, warp 1.04 us, nsys graphs 1.66 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.38 | 36.7% | 0.40 | 33.8% |
| load | 0.34 | 32.4% | 0.38 | 32.0% |
| store | 0.22 | 21.6% | 0.31 | 26.1% |
| compute | 0.10 | 9.3% | 0.10 | 8.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.25 | 23.8% | 1.0 | 0.29 | `for (int i = 0; i < 2; ++i)` |
| store.global#2 | 0.11 | 10.5% | 1.0 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.10 | 9.3% | 1.0 | 0.13 | `amax_fragment[i_1] = tl::AllReduce<tl::MaxOp, 2, 1, 0, tl::NamedBarrier<64>>::run(amax_fra` |
| load.global#2*2 (est.: sampled iterations x2) | 0.09 | 8.6% | 1.0 | 0.13 | `amax_fragment[i_1] = cutlass::bfloat16_t(__hmax((bfloat16_t(((nv_bfloat162*)(&(amax_fragme` |
| store.global*2 (est.: sampled iterations x2) | 0.06 | 6.2% | 0.6 | 0.13 | `out_sf[((((((int64_t)((int)blockIdx.x)) % (int64_t)15) * (int64_t)32) + (((int64_t)((int)t` |
| store.global#3*2 (est.: sampled iterations x2) | 0.05 | 4.9% | 0.6 | 0.13 | `*(fp8_e4_16_t*)(out + ((((((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int64_t)10240) +` |

### indexer_score_kernel grid (64, 48, 1) block (128, 1, 1)

17 launches, 12288 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 10.56 us, warp 2.48 us, nsys graphs 6.82 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.19 | 47.9% | 0.58 | 66.0% |
| unattributed, with loops' unsampled iterations | 0.90 | 36.2% | 0.28 | 31.8% |
| wait | 0.29 | 11.7% | 0.01 | 1.1% |
| compute | 0.08 | 3.1% | 0.01 | 0.9% |
| store | 0.03 | 1.1% | 0.00 | 0.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 (sampled iterations) | 0.66 | 26.7% | 0.5 | 1.47 | `for (int i_1 = 0; i_1 < 32; ++i_1)` |
| load.global | 0.42 | 16.7% | 1.0 | 0.67 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 4096); \| i` |
| wait.sync#2 (sampled iterations) | 0.17 | 6.9% | 0.5 | 0.51 | `__syncthreads();` |
| load.global#3 (sampled iterations) | 0.11 | 4.5% | 0.5 | 0.58 | `if (((int)threadIdx.x) < 64)` |
| compute.mma (sampled iterations) | 0.08 | 3.1% | 0.5 | 0.19 | `for (int ki = 0; ki < 2; ++ki)` |
| wait.sync#4 (sampled iterations) | 0.04 | 1.8% | 0.5 | 0.13 | `__syncthreads();` |
| wait.sync (sampled iterations) | 0.04 | 1.5% | 0.5 | 0.13 | `__syncthreads();` |
| wait.sync#3 (sampled iterations) | 0.04 | 1.4% | 0.5 | 0.10 | `__syncthreads();` |
| store.global (sampled iterations) | 0.03 | 1.1% | 0.5 | 0.10 | `if ((((int)threadIdx.x) >> 6) == 0)` |

### gemm_splitk_reduce_cast_kernel grid (16, 12, 1) block (128, 1, 1)

102 launches, 768 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 1.76 us, warp 1.59 us, nsys graphs 1.79 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.96 | 59.9% | 1.06 | 61.3% |
| unattributed | 0.46 | 28.7% | 0.48 | 27.4% |
| compute | 0.09 | 5.8% | 0.10 | 5.9% |
| store | 0.09 | 5.6% | 0.09 | 5.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.96 | 59.9% | 1.0 | 1.06 | `if ((1 <= M))` |
| compute.reduce | 0.09 | 5.8% | 1.0 | 0.13 | `amax[0] = tl::AllReduce<tl::MaxOp, 8, 1, 0, tl::NamedBarrier<128>>::run(amax[0]);` |
| store.global | 0.09 | 5.6% | 1.0 | 0.10 | `if (((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 5)) < M)` |

### indexer_topk_segment_kernel grid (2, 48, 1) block (512, 1, 1)

17 launches, 1536 warps; ranges, warp and last warp from the 4 recording CTAs' 64 warps; span 10.30 us, warp 5.33 us, nsys graphs 4.54 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 3.14 | 59.0% | 6.07 | 58.4% |
| unattributed, with loops' unsampled iterations | 1.78 | 33.5% | 3.28 | 31.5% |
| load | 0.36 | 6.7% | 0.94 | 9.0% |
| store | 0.04 | 0.8% | 0.11 | 1.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync#24*9 (est.: sampled iterations x9) | 0.35 | 6.6% | 0.5 | 0.86 | `__syncthreads();` |
| wait.sync#20*9 (est.: sampled iterations x9) | 0.33 | 6.3% | 0.5 | 0.86 | `__syncthreads();` |
| wait.sync#23*9 (est.: sampled iterations x9) | 0.32 | 6.0% | 0.5 | 0.86 | `__syncthreads();` |
| wait.sync#6*8 (est.: sampled iterations x8) | 0.31 | 5.8% | 0.5 | 0.77 | `__syncthreads();` |
| wait.sync#19*9 (est.: sampled iterations x9) | 0.31 | 5.7% | 0.5 | 0.86 | `__syncthreads();` |
| wait.sync#14*8 (est.: sampled iterations x8) | 0.29 | 5.4% | 0.5 | 0.77 | `__syncthreads();` |
| wait.sync#13*8 (est.: sampled iterations x8) | 0.28 | 5.2% | 0.5 | 0.77 | `__syncthreads();` |
| wait.sync#5*8 (est.: sampled iterations x8) | 0.27 | 5.1% | 0.5 | 0.77 | `__syncthreads();` |
| load.global | 0.20 | 3.8% | 1.0 | 0.26 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 4096); \| i` |
| load.global#2 | 0.16 | 3.0% | 0.5 | 0.74 | `for (int i = 0; i < 8; ++i)` |
| wait.sync (sampled iterations) | 0.14 | 2.7% | 0.5 | 0.38 | `__syncthreads();` |
| wait.sync#25 | 0.05 | 0.9% | 0.5 | 0.16 | `__syncthreads();` |
| store.global | 0.04 | 0.8% | 0.5 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3) \| if (((int)threadIdx.x) < total) \| if (((int)threadIdx.` |
| wait.sync#17 | 0.04 | 0.8% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#18 | 0.04 | 0.7% | 0.5 | 0.16 | `__syncthreads();` |
| wait.sync#16 | 0.04 | 0.7% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#8 | 0.04 | 0.7% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#2 | 0.03 | 0.6% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#3 | 0.03 | 0.6% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#7 | 0.03 | 0.6% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#9 | 0.03 | 0.6% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#11 | 0.03 | 0.6% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#15 | 0.03 | 0.6% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#21 | 0.03 | 0.6% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#22 | 0.03 | 0.6% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#12 | 0.03 | 0.6% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#4 | 0.03 | 0.5% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#10 | 0.03 | 0.5% | 0.5 | 0.06 | `__syncthreads();` |

### per_token_cast_kernel_kernel grid (30, 1, 1) block (64, 1, 1)

136 launches, 60 warps; ranges, warp and last warp from the 5 recording CTAs' 10 warps; span 1.25 us, warp 1.18 us, nsys graphs 1.12 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.44 | 37.5% | 0.43 | 34.5% |
| load | 0.35 | 29.6% | 0.40 | 31.8% |
| store | 0.29 | 24.7% | 0.33 | 26.3% |
| compute | 0.10 | 8.2% | 0.09 | 7.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.27 | 22.6% | 1.0 | 0.29 | `for (int i = 0; i < 2; ++i)` |
| store.global#2 | 0.11 | 9.5% | 1.0 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.10 | 8.2% | 1.0 | 0.13 | `amax_fragment[i_1] = tl::AllReduce<tl::MaxOp, 2, 1, 0, tl::NamedBarrier<64>>::run(amax_fra` |
| store.global#3*2 (est.: sampled iterations x2) | 0.09 | 7.6% | 1.0 | 0.13 | `*(fp8_e4_16_t*)(out + ((((((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int64_t)10240) +` |
| store.global*2 (est.: sampled iterations x2) | 0.09 | 7.6% | 1.0 | 0.13 | `out_sf[((((((int64_t)((int)blockIdx.x)) % (int64_t)15) * (int64_t)32) + (((int64_t)((int)t` |
| load.global#2*2 (est.: sampled iterations x2) | 0.08 | 7.1% | 1.0 | 0.13 | `amax_fragment[i_1] = cutlass::bfloat16_t(__hmax((bfloat16_t(((nv_bfloat162*)(&(amax_fragme` |

### indexer_topk_merge_kernel grid (1, 48, 1) block (512, 1, 1)

17 launches, 768 warps; ranges, warp and last warp from the 4 recording CTAs' 64 warps; span 9.76 us, warp 9.48 us, nsys graphs 4.45 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 6.13 | 64.7% | 5.92 | 60.5% |
| unattributed | 2.21 | 23.3% | 2.43 | 24.8% |
| load | 1.07 | 11.3% | 1.34 | 13.6% |
| store | 0.07 | 0.7% | 0.10 | 1.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.06 | 11.2% | 1.0 | 1.25 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 8192);` |
| wait.sync#20*9 (est.: sampled iterations x9) | 0.68 | 7.1% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#24*9 (est.: sampled iterations x9) | 0.66 | 7.0% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#19*9 (est.: sampled iterations x9) | 0.66 | 6.9% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#23*9 (est.: sampled iterations x9) | 0.63 | 6.7% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#14*8 (est.: sampled iterations x8) | 0.61 | 6.4% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync#6*8 (est.: sampled iterations x8) | 0.61 | 6.4% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync#5*8 (est.: sampled iterations x8) | 0.57 | 6.0% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync#13*8 (est.: sampled iterations x8) | 0.56 | 5.9% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync | 0.14 | 1.4% | 1.0 | 0.35 | `__syncthreads();` |
| wait.sync#17 | 0.08 | 0.8% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#16 | 0.08 | 0.8% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#8 | 0.07 | 0.8% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#25 | 0.07 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#2 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#9 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#3 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| store.global#2 | 0.06 | 0.7% | 1.0 | 0.10 | `Values[((((int64_t)((int)blockIdx.y)) * (int64_t)512) + ((int64_t)((int)threadIdx.x)))] = ` |
| wait.sync#15 | 0.06 | 0.6% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#4 | 0.06 | 0.6% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#7 | 0.06 | 0.6% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#12 | 0.06 | 0.6% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#11 | 0.06 | 0.6% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#10 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#22 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#18 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#21 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| load.global#2 | 0.01 | 0.1% | 0.1 | 0.10 | `if (((0 <= ((int*)sel_idx)[((int)threadIdx.x)]) && (((int*)sel_idx)[((int)threadIdx.x)] < ` |
| store.global | 0.00 | 0.0% | 0.1 | 0.10 | `Values[((((int64_t)((int)blockIdx.y)) * (int64_t)512) + ((int64_t)((int)threadIdx.x)))] = ` |

### mhc_post_kernel grid (200, 1, 1) block (128, 1, 1)

17 launches, 800 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 9.60 us, warp 6.97 us, nsys graphs 3.28 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 6.75 | 96.9% | 7.30 | 85.6% |
| load | 0.15 | 2.2% | 1.05 | 12.3% |
| store | 0.06 | 0.9% | 0.17 | 2.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.15 | 2.2% | 1.0 | 0.19 | `*(float4*)(pm + 0) = *(float4*)(post + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int` |
| store.global | 0.06 | 0.9% | 1.0 | 0.16 | `for (int n_1 = 0; n_1 < 4; ++n_1)` |

### norm_forward_and_per_token_cast_kernel_kernel grid (2, 1, 1) block (128, 1, 1)

48 launches, 8 warps; ranges, warp and last warp from the 2 recording CTAs' 8 warps; span 2.85 us, warp 2.84 us, nsys graphs 1.41 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 2.54 | 89.2% | 2.52 | 81.4% |
| load | 0.43 | 15.2% | 0.44 | 14.1% |
| compute | 0.11 | 3.8% | 0.11 | 3.5% |
| wait | 0.06 | 2.0% | 0.15 | 4.7% |
| unattributed | 0.00 | 0.0% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*4 (est.: sampled iterations x4) | 1.87 | 65.9% | 1.0 | 1.92 | `for (int j_2 = 0; j_2 < 2; ++j_2)` |
| store.global*4 (est.: sampled iterations x4) | 0.43 | 15.2% | 1.0 | 0.51 | `for (int j_1 = 0; j_1 < 4; ++j_1)` |
| load.global*4 (est.: sampled iterations x4) | 0.24 | 8.4% | 1.0 | 0.26 | `if (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 4)) < num_tokens)` |
| load.global#3*4 (est.: sampled iterations x4) | 0.16 | 5.6% | 1.0 | 0.26 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 128) + ((((int)threadIdx.x) & 15` |
| store.global#5*4 (est.: sampled iterations x4) | 0.14 | 4.8% | 1.0 | 0.19 | `*(uint4*)(out + ((((((int64_t)vi_1) * (int64_t)128) + ((((int64_t)((int)threadIdx.x)) & (i` |
| compute.reduce | 0.11 | 3.8% | 1.0 | 0.13 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| wait.sync | 0.06 | 2.0% | 1.0 | 0.06 | `__syncthreads();` |
| store.global#3 | 0.05 | 1.8% | 1.0 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |
| store.global#2 | 0.04 | 1.5% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| load.global#2 | 0.03 | 1.1% | 1.0 | 0.03 | `*(float2*)(row_partial + 0) = *(float2*)(sum_shared + ((((int)threadIdx.x) >> 6) * 2)); \| ` |

### norm_forward_and_per_token_cast_kernel_kernel grid (5, 1, 1) block (128, 1, 1)

51 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 2.75 us, warp 2.75 us, nsys graphs 1.76 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 2.24 | 81.5% | 2.29 | 79.0% |
| load | 0.43 | 15.5% | 0.42 | 14.6% |
| compute | 0.11 | 4.0% | 0.11 | 3.8% |
| wait | 0.05 | 1.9% | 0.06 | 2.1% |
| unattributed | 0.00 | 0.0% | 0.02 | 0.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*4 (est.: sampled iterations x4) | 1.56 | 56.5% | 1.0 | 1.66 | `for (int j_2 = 0; j_2 < 2; ++j_2)` |
| store.global*4 (est.: sampled iterations x4) | 0.42 | 15.1% | 1.0 | 0.51 | `for (int j_1 = 0; j_1 < 4; ++j_1)` |
| load.global*4 (est.: sampled iterations x4) | 0.23 | 8.4% | 1.0 | 0.26 | `if (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 4)) < num_tokens)` |
| store.global#5*4 (est.: sampled iterations x4) | 0.17 | 6.3% | 1.0 | 0.26 | `*(uint4*)(out + ((((((int64_t)vi_1) * (int64_t)128) + ((((int64_t)((int)threadIdx.x)) & (i` |
| load.global#3*4 (est.: sampled iterations x4) | 0.16 | 5.8% | 1.0 | 0.26 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 128) + ((((int)threadIdx.x) & 15` |
| compute.reduce | 0.11 | 4.0% | 1.0 | 0.13 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| store.global#3 | 0.05 | 1.9% | 1.0 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |
| wait.sync | 0.05 | 1.9% | 1.0 | 0.10 | `__syncthreads();` |
| store.global#2 | 0.05 | 1.7% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| load.global#2 | 0.04 | 1.3% | 1.0 | 0.06 | `*(float2*)(row_partial + 0) = *(float2*)(sum_shared + ((((int)threadIdx.x) >> 6) * 2)); \| ` |

### topk_ids_kernel grid (48, 1, 1) block (512, 1, 1)

1 launches, 768 warps; ranges, warp and last warp from the 4 recording CTAs' 64 warps; span 141.34 us, warp 137.87 us, nsys graphs 127.49 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 113.04 | 82.0% | 136.51 | 97.4% |
| wait | 22.68 | 16.5% | 0.19 | 0.1% |
| unattributed | 1.96 | 1.4% | 1.92 | 1.4% |
| store | 0.18 | 0.1% | 1.54 | 1.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 113.04 | 82.0% | 1.0 | 136.51 | `for (int t = 0; t < (vocab >> 14); ++t)` |
| wait.sync | 22.56 | 16.4% | 1.0 | 75.36 | `__syncthreads();` |
| store.global | 0.18 | 0.1% | 1.0 | 1.54 | `if ((((int)threadIdx.x) >> 5) == 0)` |
| wait.sync#2 | 0.07 | 0.1% | 1.0 | 0.13 | `__syncthreads();` |
| wait.sync#3 | 0.05 | 0.0% | 1.0 | 0.06 | `__syncthreads();` |

### token_ranks_kernel grid (16, 1, 1) block (256, 1, 1)

16 launches, 128 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 8.46 us, warp 8.27 us, nsys graphs 9.66 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 7.45 | 90.1% | 7.72 | 90.1% |
| compute | 0.41 | 4.9% | 0.44 | 5.1% |
| unattributed | 0.36 | 4.4% | 0.36 | 4.3% |
| store | 0.05 | 0.6% | 0.05 | 0.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 7.45 | 90.1% | 1.0 | 7.84 | `if ((((int64_t)0 <= token_ids[((int64_t)((int)blockIdx.x))]) && (token_ids[((int64_t)((int` |
| compute.reduce | 0.41 | 4.9% | 1.0 | 0.75 | `total[0] = tl::AllReduce<tl::SumOp, 256, 1, 0, tl::NamedBarrier<256>>::run(total[0], (&(wo` |
| store.global | 0.05 | 0.6% | 1.0 | 0.06 | `output[((int64_t)((int)blockIdx.x))] = ((int64_t)total[0]);` |

### router_partials_kernel grid (20, 1, 1) block (64, 1, 1)

51 launches, 40 warps; ranges, warp and last warp from the 4 recording CTAs' 8 warps; span 2.53 us, warp 2.50 us, nsys graphs 5.07 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.02 | 40.7% | 1.09 | 42.0% |
| unattributed | 0.93 | 37.1% | 0.94 | 36.1% |
| compute | 0.34 | 13.7% | 0.35 | 13.3% |
| store | 0.21 | 8.5% | 0.22 | 8.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#4 | 0.51 | 20.3% | 1.0 | 0.51 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < image_len)) \| if ((((((int)bl` |
| load.global | 0.38 | 15.3% | 1.0 | 0.38 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < mask_len))` |
| compute.reduce | 0.34 | 13.7% | 1.0 | 0.35 | `for (int k = 0; k < 3; ++k) \| for (int k = 0; k < 6; ++k)` |
| store.global#3 | 0.11 | 4.5% | 1.0 | 0.13 | `if ((((int)threadIdx.x) & 31) < 3) \| if ((((int)threadIdx.x) & 31) < 6)` |
| load.global#2 | 0.06 | 2.6% | 1.0 | 0.06 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < mask_len))` |
| load.global#3 | 0.06 | 2.6% | 1.0 | 0.06 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < mask_len))` |
| store.global | 0.05 | 2.1% | 1.0 | 0.06 | `if (condval == (bool)0) \| if (condval == (bool)1)` |
| store.global#2 | 0.05 | 1.9% | 1.0 | 0.06 | `if (condval_1 == (bool)0) \| if (condval_1 == (bool)1)` |

### indexer_mxfp4_quantize_kernel grid (2, 1, 1) block (128, 1, 1)

34 launches, 8 warps; ranges, warp and last warp from the 2 recording CTAs' 8 warps; span 3.66 us, warp 1.64 us, nsys graphs 2.66 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.98 | 59.7% | 2.27 | 59.3% |
| unattributed | 0.40 | 24.1% | 0.47 | 12.2% |
| store | 0.27 | 16.2% | 1.09 | 28.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.69 | 42.1% | 0.8 | 1.09 | `for (int i = 0; i < 32; ++i)` |
| load.global#2*16 (est.: sampled iterations x16) | 0.29 | 17.5% | 0.2 | 1.28 | `float v__2 = __fmul_rn(((float)X[(((((int64_t)((int)blockIdx.x)) * (int64_t)4096) + (((int` |
| store.global*16 (est.: sampled iterations x16) | 0.26 | 15.6% | 0.2 | 1.02 | `Pool[(((((slot & (int64_t)127) * (int64_t)64) + ((((int64_t)((int)threadIdx.x)) & (int64_t` |
| store.global#2 | 0.01 | 0.6% | 0.2 | 0.03 | `Pool[(((((slot & (int64_t)127) * (int64_t)4) + ((slot >> (int64_t)7) * ((int64_t)pool_stri` |

### indexer_mxfp4_quantize_kernel grid (48, 1, 1) block (128, 1, 1)

34 launches, 192 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 3.81 us, warp 3.71 us, nsys graphs 2.98 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 2.32 | 62.5% | 2.40 | 63.3% |
| store | 1.13 | 30.5% | 1.13 | 29.8% |
| unattributed | 0.26 | 7.1% | 0.26 | 6.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2*16 (est.: sampled iterations x16) | 1.28 | 34.5% | 1.0 | 1.54 | `float v__2 = __fmul_rn(((float)X[(((((int64_t)((int)blockIdx.x)) * (int64_t)4096) + (((int` |
| store.global*16 (est.: sampled iterations x16) | 1.09 | 29.3% | 1.0 | 1.54 | `Pool[(((((slot & (int64_t)127) * (int64_t)64) + ((((int64_t)((int)threadIdx.x)) & (int64_t` |
| load.global | 1.04 | 28.0% | 1.0 | 1.12 | `for (int i = 0; i < 32; ++i)` |
| store.global#2 | 0.04 | 1.1% | 1.0 | 0.06 | `Pool[(((((slot & (int64_t)127) * (int64_t)4) + ((slot >> (int64_t)7) * ((int64_t)pool_stri` |

### rope_kernel grid (48, 1, 1) block (128, 1, 1)

102 launches, 192 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 1.20 us, warp 1.16 us, nsys graphs 1.54 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.68 | 58.4% | 0.75 | 60.6% |
| load | 0.44 | 38.3% | 0.45 | 36.2% |
| store | 0.04 | 3.3% | 0.04 | 3.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.44 | 38.3% | 1.0 | 0.45 | `if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < tokens)) \| int64_t position ` |
| store.global | 0.04 | 3.3% | 1.0 | 0.06 | `*(uint4*)(x + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + ((((int64_t)((int)thread` |

### moe_combine_kernel grid (5, 40, 1) block (128, 1, 1)

51 launches, 800 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 2.24 us, warp 1.74 us, nsys graphs 2.05 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.36 | 78.0% | 1.93 | 83.8% |
| unattributed | 0.33 | 18.8% | 0.32 | 13.9% |
| store | 0.06 | 3.2% | 0.05 | 2.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.36 | 78.0% | 1.0 | 1.66 | `for (int k = 0; k < 3; ++k) \| for (int k = 0; k < 6; ++k) \| if ((((int)blockIdx.y) < share` |
| store.global | 0.06 | 3.2% | 1.0 | 0.06 | `{` |

### norm_forward_and_per_token_cast_kernel_kernel grid (20, 1, 1) block (128, 1, 1)

51 launches, 80 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 2.40 us, warp 2.20 us, nsys graphs 2.11 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 1.56 | 71.1% | 1.51 | 63.0% |
| load | 0.34 | 15.6% | 0.33 | 13.8% |
| compute | 0.12 | 5.4% | 0.12 | 5.0% |
| wait | 0.11 | 5.0% | 0.05 | 2.0% |
| unattributed | 0.06 | 2.9% | 0.39 | 16.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*3 (est.: sampled iterations x3) | 1.06 | 48.3% | 1.0 | 1.15 | `for (int j_2 = 0; j_2 < 2; ++j_2)` |
| store.global*3 (est.: sampled iterations x3) | 0.29 | 13.4% | 1.0 | 0.38 | `for (int j_1 = 0; j_1 < 4; ++j_1)` |
| load.global*3 (est.: sampled iterations x3) | 0.18 | 8.2% | 1.0 | 0.19 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 6)) < num_tokens) && (((vi * 2) + (` |
| store.global#5*3 (est.: sampled iterations x3) | 0.14 | 6.3% | 1.0 | 0.19 | `*(uint4*)(out + ((((((int64_t)vi_1) * (int64_t)512) + ((((int64_t)((int)threadIdx.x)) & (i` |
| load.global#3*3 (est.: sampled iterations x3) | 0.13 | 6.0% | 1.0 | 0.19 | `if (((vi_1 * 2) + ((((int)threadIdx.x) & 63) >> 5)) < 5)` |
| compute.reduce | 0.12 | 5.4% | 1.0 | 0.13 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| wait.sync | 0.11 | 5.0% | 1.0 | 0.19 | `__syncthreads();` |
| store.global#2 | 0.04 | 1.9% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| load.global#2 | 0.03 | 1.5% | 1.0 | 0.06 | `*(float2*)(row_partial + 0) = *(float2*)(sum_shared + ((((int)threadIdx.x) >> 6) * 2)); \| ` |
| store.global#3 | 0.03 | 1.3% | 0.5 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |

### gemm_bf16_partials_kernel grid (1, 40, 1) block (256, 1, 1)

34 launches, 320 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 3.26 us, warp 1.98 us, nsys graphs 3.26 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.31 | 66.5% | 2.28 | 67.4% |
| compute | 0.29 | 14.6% | 0.59 | 17.4% |
| wait | 0.23 | 11.7% | 0.37 | 11.0% |
| store | 0.07 | 3.7% | 0.14 | 4.3% |
| load | 0.07 | 3.5% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*2 (est.: sampled iterations x2) | 0.29 | 14.6% | 0.5 | 0.64 | `{` |
| wait.sync | 0.10 | 5.2% | 1.0 | 0.13 | `__syncthreads();` |
| store.global | 0.07 | 3.7% | 0.5 | 0.18 | `for (int i_1 = 0; i_1 < 4; ++i_1) \| for (int i_2 = 0; i_2 < 8; ++i_2) \| for (int i_4 = 0; ` |
| load.tma*2 (est.: sampled iterations x2) | 0.07 | 3.5% | 0.5 | 0.19 | `if (tl::tl_shuffle_elect<128>())` |
| wait.mbarrier#2*2 (est.: sampled iterations x2) | 0.06 | 3.2% | 0.5 | 0.13 | `mbarrier[kb_1].wait(0);` |
| wait.mbarrier*2 (est.: sampled iterations x2) | 0.06 | 3.2% | 0.5 | 0.13 | `mbarrier[(kb + 3)].wait(1);` |

### metadata_chunk_kernel grid (40, 1, 1) block (192, 1, 1)

51 launches, 240 warps; ranges, warp and last warp from the 4 recording CTAs' 24 warps; span 2.18 us, warp 1.50 us, nsys graphs 2.34 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.08 | 71.9% | 1.51 | 68.3% |
| unattributed | 0.36 | 24.0% | 0.57 | 25.8% |
| store | 0.06 | 4.1% | 0.13 | 6.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.04 | 69.0% | 1.0 | 1.34 | `if ((((offset + ((int)blockIdx.x)) < src_rows) && (0 <= (offset + ((int)blockIdx.x)))))` |
| store.global | 0.05 | 3.1% | 1.0 | 0.06 | `swa[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = con` |
| load.global#2 | 0.04 | 2.8% | 0.2 | 0.29 | `if (((0 <= req) & ((int64_t)0 <= pos)))` |
| store.global#2 | 0.01 | 0.5% | 0.2 | 0.06 | `lengths[((int64_t)((int)blockIdx.x))] = ((int)max(info[1], (int64_t)0));` |
| store.global#3 | 0.01 | 0.4% | 0.2 | 0.06 | `top_lengths[((int64_t)((int)blockIdx.x))] = condval_12; \| top_lengths[((int64_t)((int)bloc` |

### resample_kernel grid (8, 127, 1) block (128, 1, 1)

17 launches, 4064 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 6.62 us, warp 2.77 us, nsys graphs 6.40 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.66 | 59.8% | 1.13 | 51.3% |
| unattributed | 0.65 | 23.4% | 0.61 | 27.7% |
| compute | 0.31 | 11.0% | 0.28 | 12.9% |
| wait | 0.09 | 3.1% | 0.09 | 3.9% |
| store | 0.08 | 2.7% | 0.09 | 4.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.66 | 59.8% | 1.0 | 2.40 | `end = cu_num_logits[(((int64_t)((int)blockIdx.x)) + (int64_t)1)];` |
| compute.reduce#2 | 0.15 | 5.5% | 1.0 | 0.19 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 8, 32);` |
| wait.sync | 0.09 | 3.1% | 1.0 | 0.16 | `__syncthreads();` |
| compute.reduce#3 | 0.08 | 2.9% | 1.0 | 0.10 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 2, 32);` |
| store.global | 0.08 | 2.7% | 1.0 | 0.13 | `if (((int)threadIdx.x) == 0)` |
| compute.reduce | 0.07 | 2.7% | 1.0 | 0.10 | `if ((temp != 0x0p+0f/*0.000000e+00*/) \| bonus)` |

### prepare_dflash_inputs_kernel grid (9, 8, 1) block (256, 1, 1)

51 launches, 576 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 2.18 us, warp 1.89 us, nsys graphs 2.40 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.46 | 77.2% | 1.71 | 79.3% |
| unattributed | 0.37 | 19.4% | 0.39 | 18.1% |
| store | 0.06 | 3.3% | 0.06 | 2.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.40 | 74.1% | 1.0 | 1.66 | `req = idx_mapping[((int)blockIdx.y)];` |
| store.global#2 | 0.06 | 3.3% | 1.0 | 0.10 | `if (((int)blockIdx.y) == (num_reqs - 1))` |
| load.global#2 | 0.06 | 3.2% | 1.0 | 0.06 | `vllm_store_if_b32((&(out_query_start_loc[((int)blockIdx.y)])), (((int)blockIdx.y) * num_qu` |

### local_stats_kernel grid (48, 16, 1) block (128, 1, 1)

1 launches, 3072 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 109.28 us, warp 38.28 us, nsys graphs 89.47 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 36.26 | 94.7% | 47.07 | 97.2% |
| unattributed | 0.89 | 2.3% | 0.93 | 1.9% |
| wait | 0.82 | 2.1% | 0.19 | 0.4% |
| compute | 0.25 | 0.6% | 0.16 | 0.3% |
| store | 0.07 | 0.2% | 0.10 | 0.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 36.26 | 94.7% | 1.0 | 47.17 | `if (expanded_local_pos[((int64_t)((int)blockIdx.x))] < num_speculative_steps)` |
| wait.sync | 0.82 | 2.1% | 1.0 | 3.39 | `__syncthreads();` |
| compute.reduce | 0.19 | 0.5% | 1.0 | 1.25 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| store.global | 0.07 | 0.2% | 1.0 | 0.10 | `if (((int)threadIdx.x) == 0)` |
| compute.reduce#2 | 0.06 | 0.2% | 1.0 | 0.06 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 2, 32);` |

### compress_gather_partial_state_kernel grid (8, 1, 1) block (128, 1, 1)

17 launches, 32 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 5.41 us, warp 4.35 us, nsys graphs 2.93 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 2.42 | 55.7% | 2.77 | 43.4% |
| load | 1.06 | 24.3% | 2.46 | 38.5% |
| store | 0.87 | 20.0% | 1.16 | 18.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.94 | 21.5% | 1.0 | 0.96 | `int nt = counts[0];` |
| store.global*2 (est.: sampled iterations x2) | 0.82 | 18.8% | 1.0 | 0.90 | `if (half == 0)` |
| load.global#2*2 (est.: sampled iterations x2) | 0.12 | 2.8% | 1.0 | 0.13 | `for (int i = 0; i < 4; ++i)` |
| store.global#2 | 0.06 | 1.3% | 1.0 | 0.10 | `if (((int)threadIdx.x) == 0)` |

### activation_swiglu_clamp_kernel grid (1, 48, 1) block (128, 1, 1)

102 launches, 192 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 0.96 us, warp 0.73 us, nsys graphs 1.89 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.63 | 85.6% | 0.81 | 84.8% |
| load | 0.07 | 9.8% | 0.10 | 10.2% |
| store | 0.03 | 4.6% | 0.05 | 5.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.07 | 9.8% | 0.8 | 0.10 | `*(uint4*)(gate + 0) = *(uint4*)(X + ((((int64_t)((int)blockIdx.y)) * (int64_t)1152) + (((i` |
| store.global | 0.03 | 4.6% | 0.8 | 0.06 | `*(uint4*)(Y + ((((int64_t)((int)blockIdx.y)) * (int64_t)576) + (((int64_t)((int)threadIdx.` |

### gemm_bf16_partials_kernel grid (2, 4, 1) block (256, 1, 1)

34 launches, 64 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 2.50 us, warp 1.62 us, nsys graphs 2.70 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.94 | 57.8% | 1.46 | 51.7% |
| compute | 0.29 | 17.7% | 0.57 | 20.4% |
| wait | 0.23 | 14.0% | 0.58 | 20.6% |
| store | 0.10 | 6.5% | 0.21 | 7.4% |
| load | 0.07 | 4.1% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*2 (est.: sampled iterations x2) | 0.29 | 17.7% | 0.5 | 0.58 | `{` |
| wait.sync | 0.11 | 6.7% | 1.0 | 0.13 | `__syncthreads();` |
| store.global | 0.10 | 6.5% | 0.5 | 0.22 | `for (int i_1 = 0; i_1 < 4; ++i_1) \| for (int i_2 = 0; i_2 < 8; ++i_2) \| for (int i_4 = 0; ` |
| load.tma*2 (est.: sampled iterations x2) | 0.07 | 4.1% | 0.5 | 0.19 | `if (tl::tl_shuffle_elect<128>())` |
| wait.mbarrier#2*2 (est.: sampled iterations x2) | 0.06 | 3.6% | 0.5 | 0.13 | `mbarrier[kb_1].wait(0);` |
| wait.mbarrier*2 (est.: sampled iterations x2) | 0.06 | 3.6% | 0.5 | 0.13 | `mbarrier[(kb + 3)].wait(1);` |

### norm_forward_and_per_token_cast_kernel_kernel grid (3, 1, 1) block (128, 1, 1)

34 launches, 12 warps; ranges, warp and last warp from the 3 recording CTAs' 12 warps; span 2.14 us, warp 2.14 us, nsys graphs 1.89 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 1.10 | 51.4% | 1.23 | 49.7% |
| unattributed | 0.67 | 31.4% | 0.76 | 30.7% |
| load | 0.23 | 10.7% | 0.30 | 12.0% |
| compute | 0.09 | 4.0% | 0.09 | 3.5% |
| wait | 0.05 | 2.5% | 0.10 | 4.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*2 (est.: sampled iterations x2) | 0.73 | 34.0% | 1.0 | 0.77 | `for (int j_2 = 0; j_2 < 2; ++j_2)` |
| store.global*2 (est.: sampled iterations x2) | 0.21 | 9.7% | 1.0 | 0.26 | `for (int j_1 = 0; j_1 < 4; ++j_1)` |
| load.global*2 (est.: sampled iterations x2) | 0.12 | 5.5% | 1.0 | 0.13 | `if (((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < num_tokens)` |
| compute.reduce | 0.09 | 4.0% | 1.0 | 0.10 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| load.global#3*2 (est.: sampled iterations x2) | 0.08 | 3.7% | 1.0 | 0.13 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 64) + ((((int)threadIdx.x) & 7) ` |
| store.global#5*2 (est.: sampled iterations x2) | 0.08 | 3.6% | 1.0 | 0.13 | `*(uint4*)(out + ((((((int64_t)vi_1) * (int64_t)64) + ((((int64_t)((int)threadIdx.x)) & (in` |
| wait.sync | 0.05 | 2.5% | 1.0 | 0.06 | `__syncthreads();` |
| store.global#3 | 0.05 | 2.3% | 1.0 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |
| store.global#2 | 0.04 | 1.9% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| load.global#2 | 0.03 | 1.5% | 1.0 | 0.03 | `*(float2*)(row_partial + 0) = *(float2*)(sum_shared + ((((int)threadIdx.x) >> 6) * 2)); \| ` |

### gemm_splitk_reduce_cast_kernel grid (16, 10, 1) block (128, 1, 1)

51 launches, 640 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 1.44 us, warp 1.20 us, nsys graphs 1.63 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.66 | 54.7% | 0.94 | 61.7% |
| unattributed | 0.37 | 30.8% | 0.40 | 26.5% |
| store | 0.09 | 7.3% | 0.09 | 5.8% |
| compute | 0.09 | 7.2% | 0.09 | 6.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.66 | 54.7% | 1.0 | 0.86 | `if ((1 <= M))` |
| store.global | 0.09 | 7.3% | 1.0 | 0.10 | `if (((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 5)) < M)` |
| compute.reduce | 0.09 | 7.2% | 1.0 | 0.10 | `amax[0] = tl::AllReduce<tl::MaxOp, 8, 1, 0, tl::NamedBarrier<128>>::run(amax[0]);` |

### topk_log_softmax_kernel grid (48, 1, 1) block (1024, 1, 1)

1 launches, 1536 warps; ranges, warp and last warp from the 4 recording CTAs' 128 warps; span 76.74 us, warp 70.52 us, nsys graphs 65.50 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 60.73 | 86.1% | 72.32 | 94.6% |
| wait | 7.82 | 11.1% | 1.89 | 2.5% |
| unattributed, with loops' unsampled iterations | 1.59 | 2.3% | 1.79 | 2.3% |
| compute | 0.37 | 0.5% | 0.38 | 0.5% |
| store | 0.00 | 0.0% | 0.06 | 0.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 41.28 | 58.5% | 1.0 | 49.38 | `if (0 < (vocab & 2047)) \| if (0 < (vocab & 8191))` |
| load.global#2 | 19.39 | 27.5% | 1.0 | 24.54 | `for (int step_1 = 0; step_1 < (vocab >> 11); ++step_1) \| for (int step_1 = 0; step_1 < (vo` |
| wait.sync | 5.32 | 7.5% | 1.0 | 20.13 | `__syncthreads();` |
| wait.sync#2 | 2.25 | 3.2% | 1.0 | 7.62 | `__syncthreads();` |
| compute.reduce#3 | 0.17 | 0.2% | 1.0 | 0.22 | `ties[0] = vllm_add(ties[0], __shfl_xor_sync((uint)4294967295, ties[0], 16, 32));` |
| compute.reduce#2 | 0.10 | 0.1% | 1.0 | 0.13 | `se[0] = vllm_add(se[0], __shfl_xor_sync((uint)4294967295, se[0], 16, 32));` |
| wait.sync#4 | 0.10 | 0.1% | 1.0 | 0.16 | `__syncthreads();` |
| compute.reduce | 0.10 | 0.1% | 1.0 | 0.13 | `best[0] = vllm_max(best[0], __shfl_xor_sync((uint)4294967295, best[0], 16, 32));` |
| wait.sync#5 | 0.08 | 0.1% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#3 | 0.08 | 0.1% | 1.0 | 0.10 | `__syncthreads();` |
| load.global#3 (sampled iterations) | 0.06 | 0.1% | 0.0 | 5.44 | `float value_32 = vllm_sub(vllm_shifted(logits[((((int64_t)((int)blockIdx.x)) * logits_stri` |
| store.global (sampled iterations) | 0.00 | 0.0% | 0.0 | 0.06 | `output[(((((int64_t)c) * (int64_t)1024) + (((int64_t)((int)blockIdx.x)) * ((int64_t)topk))` |

### mhc_collapse_kernel grid (200, 1, 1) block (128, 1, 1)

17 launches, 800 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 3.90 us, warp 2.59 us, nsys graphs 2.30 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 2.33 | 89.8% | 2.54 | 74.8% |
| load | 0.21 | 8.2% | 0.82 | 24.0% |
| store | 0.05 | 1.9% | 0.04 | 1.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.21 | 8.2% | 1.0 | 0.58 | `*(float4*)(mv + 0) = *(float4*)(mix + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int6` |
| store.global | 0.05 | 1.9% | 1.0 | 0.06 | `*(uint4*)(out + ((((int64_t)((int)blockIdx.x)) * (int64_t)1024) + (((int64_t)((int)threadI` |

### sink_merge_kernel grid (8, 1, 1) block (128, 1, 1)

17 launches, 32 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 4.06 us, warp 4.03 us, nsys graphs 4.45 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.91 | 47.4% | 2.07 | 50.4% |
| wait | 0.83 | 20.5% | 0.44 | 10.7% |
| unattributed | 0.70 | 17.5% | 0.91 | 22.3% |
| compute | 0.56 | 13.8% | 0.55 | 13.4% |
| store | 0.03 | 0.7% | 0.13 | 3.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2*3 (est.: sampled iterations x3) | 1.63 | 40.5% | 1.0 | 1.73 | `for (int step = 0; step < ((width + 127) >> 7); ++step) \| for (int step = 0; step < ((widt` |
| compute.reduce*3 (est.: sampled iterations x3) | 0.56 | 13.8% | 1.0 | 0.58 | `float other = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| wait.sync#3*3 (est.: sampled iterations x3) | 0.52 | 12.9% | 1.0 | 0.67 | `__syncthreads();` |
| load.global | 0.23 | 5.8% | 1.0 | 0.26 | `last = sampled[(max((((int64_t)num_sampled[((int64_t)((int)blockIdx.x))]) - (int64_t)1), (` |
| wait.sync*3 (est.: sampled iterations x3) | 0.16 | 4.0% | 1.0 | 0.19 | `__syncthreads();` |
| wait.sync#2*3 (est.: sampled iterations x3) | 0.14 | 3.6% | 1.0 | 0.19 | `__syncthreads();` |
| load.global#3*3 (est.: sampled iterations x3) | 0.05 | 1.2% | 0.2 | 0.19 | `if ((best[0] == -CUDART_INF_F))` |
| store.global*3 (est.: sampled iterations x3) | 0.03 | 0.7% | 0.2 | 0.19 | `alternates[((((int64_t)((int)blockIdx.x)) * (int64_t)3) + ((int64_t)rank))] = condval_31;` |

### per_token_cast_kernel_kernel grid (25, 1, 1) block (64, 1, 1)

51 launches, 50 warps; ranges, warp and last warp from the 5 recording CTAs' 10 warps; span 1.15 us, warp 1.10 us, nsys graphs 1.38 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.37 | 33.3% | 0.39 | 29.7% |
| load | 0.35 | 31.6% | 0.37 | 28.5% |
| store | 0.29 | 26.4% | 0.44 | 34.0% |
| compute | 0.10 | 8.7% | 0.10 | 7.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.26 | 23.5% | 1.0 | 0.29 | `for (int i = 0; i < 2; ++i)` |
| store.global#2 | 0.11 | 10.1% | 1.0 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.10 | 8.7% | 1.0 | 0.13 | `amax_fragment[i_1] = tl::AllReduce<tl::MaxOp, 2, 1, 0, tl::NamedBarrier<64>>::run(amax_fra` |
| load.global#2*2 (est.: sampled iterations x2) | 0.09 | 8.1% | 1.0 | 0.13 | `amax_fragment[i_1] = cutlass::bfloat16_t(__hmax((bfloat16_t(((nv_bfloat162*)(&(amax_fragme` |
| store.global#3*2 (est.: sampled iterations x2) | 0.09 | 8.1% | 1.0 | 0.13 | `*(fp8_e4_16_t*)(out + ((((((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int64_t)10240) +` |
| store.global*2 (est.: sampled iterations x2) | 0.09 | 8.1% | 1.0 | 0.13 | `out_sf[((((((int64_t)((int)blockIdx.x)) % (int64_t)15) * (int64_t)32) + (((int64_t)((int)t` |

### activation_swiglu_clamp_kernel grid (1, 40, 1) block (128, 1, 1)

51 launches, 160 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 0.93 us, warp 0.74 us, nsys graphs 1.49 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.64 | 85.8% | 1.06 | 82.1% |
| load | 0.07 | 9.7% | 0.17 | 12.8% |
| store | 0.03 | 4.6% | 0.06 | 5.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.07 | 9.7% | 0.8 | 0.10 | `*(uint4*)(gate + 0) = *(uint4*)(X + ((((int64_t)((int)blockIdx.y)) * (int64_t)1152) + (((i` |
| store.global | 0.03 | 4.6% | 0.8 | 0.06 | `*(uint4*)(Y + ((((int64_t)((int)blockIdx.y)) * (int64_t)576) + (((int64_t)((int)threadIdx.` |

### norm_forward_and_per_token_cast_kernel_kernel grid (48, 1, 1) block (128, 1, 1)

18 launches, 192 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 3.36 us, warp 3.18 us, nsys graphs 2.46 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 1.32 | 41.5% | 2.34 | 65.0% |
| unattributed | 1.25 | 39.2% | 0.66 | 18.3% |
| load | 0.43 | 13.5% | 0.42 | 11.6% |
| compute | 0.13 | 4.1% | 0.13 | 3.6% |
| wait | 0.06 | 1.8% | 0.05 | 1.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*5 (est.: sampled iterations x5) | 1.02 | 32.2% | 1.0 | 1.36 | `for (int j_1 = 0; j_1 < 2; ++j_1)` |
| store.global*5 (est.: sampled iterations x5) | 0.24 | 7.5% | 1.0 | 0.32 | `for (int j = 0; j < 4; ++j)` |
| load.global*5 (est.: sampled iterations x5) | 0.22 | 6.9% | 1.0 | 0.32 | `*(uint4*)(x_local + (vi * 8)) = *(uint4*)(x + ((((((int64_t)vi) * (int64_t)1024) + (((int6` |
| load.global#2*5 (est.: sampled iterations x5) | 0.21 | 6.6% | 1.0 | 0.32 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 1024) + (((int)threadIdx.x) * 8)` |
| compute.reduce | 0.13 | 4.1% | 1.0 | 0.16 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| wait.sync | 0.06 | 1.8% | 1.0 | 0.10 | `__syncthreads();` |
| store.global#2 | 0.04 | 1.3% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| store.global#3 | 0.01 | 0.4% | 0.2 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |

### gemm_splitk_reduce_kernel grid (1, 48, 1) block (128, 1, 1)

68 launches, 192 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 0.86 us, warp 0.82 us, nsys graphs 1.02 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.45 | 55.2% | 0.55 | 59.7% |
| load | 0.32 | 39.0% | 0.32 | 34.9% |
| store | 0.05 | 5.8% | 0.05 | 5.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.32 | 39.0% | 1.0 | 0.34 | `acc[0] = P[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x))` |
| store.global | 0.05 | 5.8% | 1.0 | 0.06 | `C[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x)) * (int64` |

### norm_forward_and_per_token_cast_kernel_kernel grid (40, 1, 1) block (128, 1, 1)

17 launches, 160 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 3.39 us, warp 3.21 us, nsys graphs 2.40 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 1.89 | 58.9% | 2.85 | 74.8% |
| unattributed | 0.68 | 21.1% | 0.24 | 6.2% |
| load | 0.45 | 14.0% | 0.51 | 13.4% |
| compute | 0.13 | 4.0% | 0.13 | 3.4% |
| wait | 0.06 | 2.0% | 0.08 | 2.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*5 (est.: sampled iterations x5) | 1.64 | 51.0% | 1.0 | 2.24 | `for (int j_1 = 0; j_1 < 2; ++j_1)` |
| load.global*5 (est.: sampled iterations x5) | 0.23 | 7.2% | 1.0 | 0.32 | `*(uint4*)(x_local + (vi * 8)) = *(uint4*)(x + ((((((int64_t)vi) * (int64_t)1024) + (((int6` |
| load.global#2*5 (est.: sampled iterations x5) | 0.22 | 6.8% | 1.0 | 0.32 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 1024) + (((int)threadIdx.x) * 8)` |
| store.global*5 (est.: sampled iterations x5) | 0.20 | 6.2% | 1.0 | 0.32 | `for (int j = 0; j < 4; ++j)` |
| compute.reduce | 0.13 | 4.0% | 1.0 | 0.16 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| wait.sync | 0.06 | 2.0% | 1.0 | 0.10 | `__syncthreads();` |
| store.global#2 | 0.04 | 1.2% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| store.global#3 | 0.01 | 0.4% | 0.2 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |

### finish_argmax_kernel grid (8, 1, 1) block (32, 1, 1)

85 launches, 8 warps; ranges, warp and last warp from the 4 recording CTAs' 4 warps; span 0.74 us, warp 0.73 us, nsys graphs 1.06 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.38 | 52.7% | 0.39 | 52.8% |
| load | 0.13 | 17.6% | 0.13 | 17.2% |
| compute | 0.12 | 16.5% | 0.12 | 16.2% |
| store | 0.10 | 13.2% | 0.10 | 13.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.13 | 17.6% | 1.0 | 0.13 | `for (int k = 0; k < 2; ++k) \| if ((((int)threadIdx.x) < parts))` |
| compute.reduce | 0.12 | 16.5% | 1.0 | 0.13 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| store.global | 0.10 | 13.2% | 1.0 | 0.10 | `if (((int)threadIdx.x) == 0)` |

### context_inputs_kernel grid (5, 16, 1) block (128, 1, 1)

16 launches, 320 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 3.76 us, warp 2.49 us, nsys graphs 3.97 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 2.10 | 84.2% | 3.27 | 89.2% |
| unattributed | 0.33 | 13.1% | 0.33 | 9.1% |
| store | 0.07 | 2.7% | 0.06 | 1.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 2.10 | 84.2% | 1.0 | 3.12 | `vllm_dspark_context_b128((&(aux[(((((int64_t)((int)blockIdx.y)) * (int64_t)15360) + (((int` |
| store.global | 0.07 | 2.7% | 1.0 | 0.10 | `if (((((int)blockIdx.x) == 0) & (((int)threadIdx.x) == 0)) & (num_tokens <= ((int)blockIdx` |

### rope_kernel grid (40, 1, 1) block (128, 1, 1)

51 launches, 160 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 1.22 us, warp 1.17 us, nsys graphs 1.66 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.67 | 57.2% | 0.70 | 57.9% |
| load | 0.46 | 39.6% | 0.47 | 38.8% |
| store | 0.04 | 3.2% | 0.04 | 3.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.46 | 39.6% | 1.0 | 0.48 | `if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < tokens)) \| int64_t position ` |
| store.global | 0.04 | 3.2% | 1.0 | 0.06 | `*(uint4*)(x + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + ((((int64_t)((int)thread` |

### mhc_collapse_column_kernel grid (240, 1, 1) block (128, 1, 1)

17 launches, 960 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 3.55 us, warp 1.40 us, nsys graphs 2.18 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.10 | 78.5% | 1.21 | 60.8% |
| load | 0.24 | 17.0% | 0.72 | 36.5% |
| store | 0.06 | 4.6% | 0.05 | 2.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.24 | 17.0% | 1.0 | 0.67 | `*(float4*)(mv + 0) = *(float4*)(mix + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int6` |
| store.global | 0.06 | 4.6% | 1.0 | 0.10 | `*(uint4*)(out + ((((int64_t)((int)blockIdx.x)) * (int64_t)1024) + (((int64_t)((int)threadI` |

### dspark_confidence_kernel grid (40, 1, 1) block (128, 1, 1)

17 launches, 160 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 3.39 us, warp 3.08 us, nsys graphs 3.23 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 2.63 | 85.6% | 2.99 | 84.9% |
| load | 0.17 | 5.5% | 0.21 | 5.9% |
| wait | 0.16 | 5.1% | 0.14 | 4.0% |
| compute | 0.09 | 3.1% | 0.10 | 2.7% |
| store | 0.02 | 0.7% | 0.09 | 2.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.16 | 5.2% | 1.0 | 0.16 | `for (int c = 0; c < 5; ++c)` |
| wait.sync | 0.16 | 5.1% | 1.0 | 0.42 | `__syncthreads();` |
| compute.reduce | 0.09 | 3.1% | 1.0 | 0.10 | `acc[0] = tl::warp_reduce_sum(acc[0]);` |
| store.global | 0.02 | 0.7% | 0.2 | 0.10 | `out[((int64_t)((int)blockIdx.x))] = (0x1p+0f/*1.000000e+00*/ / (0x1p+0f/*1.000000e+00*/ + ` |
| load.global#2 | 0.01 | 0.3% | 0.2 | 0.06 | `total[0] = 0x0p+0f/*0.000000e+00*/;` |

### per_token_cast_kernel_kernel grid (40, 1, 1) block (64, 1, 1)

48 launches, 80 warps; ranges, warp and last warp from the 4 recording CTAs' 8 warps; span 1.10 us, warp 1.07 us, nsys graphs 1.06 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.38 | 35.2% | 0.39 | 33.2% |
| store | 0.30 | 28.5% | 0.31 | 26.4% |
| unattributed | 0.29 | 27.3% | 0.39 | 33.0% |
| compute | 0.10 | 9.0% | 0.09 | 7.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.29 | 27.0% | 1.0 | 0.29 | `for (int i = 0; i < 2; ++i)` |
| store.global#2 | 0.11 | 10.5% | 1.0 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.10 | 9.0% | 1.0 | 0.13 | `amax_fragment[i_1] = tl::AllReduce<tl::MaxOp, 2, 1, 0, tl::NamedBarrier<64>>::run(amax_fra` |
| store.global#3*2 (est.: sampled iterations x2) | 0.10 | 9.0% | 1.0 | 0.13 | `*(fp8_e4_16_t*)(out + ((((((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int64_t)10240) +` |
| store.global*2 (est.: sampled iterations x2) | 0.10 | 9.0% | 1.0 | 0.13 | `out_sf[((((((int64_t)((int)blockIdx.x)) % (int64_t)15) * (int64_t)32) + (((int64_t)((int)t` |
| load.global#2*2 (est.: sampled iterations x2) | 0.09 | 8.2% | 1.0 | 0.13 | `amax_fragment[i_1] = cutlass::bfloat16_t(__hmax((bfloat16_t(((nv_bfloat162*)(&(amax_fragme` |

### verify_kernel grid (8, 1, 1) block (96, 1, 1)

17 launches, 24 warps; ranges, warp and last warp from the 4 recording CTAs' 12 warps; span 2.59 us, warp 2.50 us, nsys graphs 2.24 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.06 | 42.5% | 1.58 | 47.1% |
| unattributed | 0.89 | 35.6% | 1.10 | 32.7% |
| compute | 0.30 | 11.9% | 0.30 | 8.9% |
| wait | 0.15 | 6.2% | 0.21 | 6.4% |
| store | 0.10 | 3.8% | 0.16 | 4.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.00 | 39.9% | 1.0 | 1.02 | `start = cu_num_logits[((int64_t)((int)blockIdx.x))];` |
| compute.reduce | 0.30 | 11.9% | 1.0 | 0.32 | `row_max[0] = vllm_max(row_max[0], __shfl_xor_sync((uint)4294967295, row_max[0], 8, 32));` |
| store.global | 0.10 | 3.8% | 1.0 | 0.16 | `if (((int)threadIdx.x) == 0)` |
| load.global#2 | 0.06 | 2.6% | 1.0 | 0.06 | `target = local_argmax[((((int64_t)row) * argmax_stride) + ((int64_t)best_index[0]))];` |
| wait.sync#2 | 0.06 | 2.6% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync | 0.05 | 1.9% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#3 | 0.04 | 1.7% | 1.0 | 0.06 | `__syncthreads();` |

### adaptive_offsets_kernel grid (1, 1, 1) block (256, 1, 1)

16 launches, 8 warps; ranges, warp and last warp from the 1 recording CTAs' 8 warps; span 3.52 us, warp 3.52 us, nsys graphs 1.95 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 1.57 | 44.5% | 1.10 | 31.2% |
| unattributed, with loops' unsampled iterations | 1.44 | 40.8% | 1.57 | 44.3% |
| store | 0.33 | 9.4% | 0.49 | 13.8% |
| load | 0.18 | 5.2% | 0.38 | 10.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync#18 (sampled iterations) | 0.34 | 9.7% | 1.0 | 0.38 | `__syncthreads();` |
| wait.sync (sampled iterations) | 0.20 | 5.8% | 1.0 | 0.22 | `__syncthreads();` |
| store.global#5 | 0.12 | 3.5% | 1.0 | 0.13 | `for (int chunk_1 = 0; chunk_1 < ((starts_len + 255) >> 8); ++chunk_1)` |
| load.global#2 (sampled iterations) | 0.10 | 2.8% | 1.0 | 0.22 | `if ((((chunk * 256) + ((int)threadIdx.x)) < num_reqs))` |
| store.global | 0.09 | 2.6% | 1.0 | 0.10 | `if (((int)threadIdx.x) == 0)` |
| store.global#2 (sampled iterations) | 0.09 | 2.6% | 1.0 | 0.10 | `if ((((chunk * 256) + ((int)threadIdx.x)) < num_reqs) & (drafted == 0))` |
| load.global (sampled iterations) | 0.08 | 2.2% | 1.0 | 0.10 | `if (((((chunk * 256) + ((int)threadIdx.x)) < num_reqs) & (drafted != 0)))` |
| wait.sync#10 (sampled iterations) | 0.06 | 1.8% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#11 (sampled iterations) | 0.06 | 1.8% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#12 (sampled iterations) | 0.06 | 1.8% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#13 (sampled iterations) | 0.06 | 1.8% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#15 (sampled iterations) | 0.06 | 1.8% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#16 (sampled iterations) | 0.06 | 1.8% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#17 (sampled iterations) | 0.06 | 1.8% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#2 (sampled iterations) | 0.06 | 1.8% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#3 (sampled iterations) | 0.06 | 1.8% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#4 (sampled iterations) | 0.06 | 1.8% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#5 (sampled iterations) | 0.06 | 1.8% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#6 (sampled iterations) | 0.06 | 1.8% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#7 (sampled iterations) | 0.06 | 1.8% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#8 (sampled iterations) | 0.06 | 1.8% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#9 (sampled iterations) | 0.06 | 1.8% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#14 (sampled iterations) | 0.06 | 1.8% | 1.0 | 0.06 | `__syncthreads();` |
| store.global#4 (sampled iterations) | 0.02 | 0.5% | 0.1 | 0.13 | `for (int j = 0; j < 6; ++j)` |
| load.global#3 (sampled iterations) | 0.01 | 0.2% | 0.1 | 0.06 | `state = idx_mapping[((((int64_t)chunk) * (int64_t)256) + ((int64_t)((int)threadIdx.x)))];` |
| store.global#3 (sampled iterations) | 0.01 | 0.2% | 0.1 | 0.06 | `cu_num_logits[(((((int64_t)chunk) * (int64_t)256) + ((int64_t)((int)threadIdx.x))) + (int6` |

### token_ranks_kernel grid (48, 1, 1) block (256, 1, 1)

1 launches, 384 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 55.42 us, warp 48.00 us, nsys graphs 42.98 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 44.33 | 92.3% | 52.51 | 95.3% |
| compute | 3.28 | 6.8% | 2.18 | 3.9% |
| unattributed | 0.35 | 0.7% | 0.38 | 0.7% |
| store | 0.04 | 0.1% | 0.03 | 0.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 44.33 | 92.3% | 1.0 | 54.50 | `if ((((int64_t)0 <= token_ids[((int64_t)((int)blockIdx.x))]) && (token_ids[((int64_t)((int` |
| compute.reduce | 3.28 | 6.8% | 1.0 | 12.61 | `total[0] = tl::AllReduce<tl::SumOp, 256, 1, 0, tl::NamedBarrier<256>>::run(total[0], (&(wo` |
| store.global | 0.04 | 0.1% | 1.0 | 0.06 | `output[((int64_t)((int)blockIdx.x))] = ((int64_t)total[0]);` |

### cache_writer_kernel grid (40, 1, 1) block (128, 1, 1)

51 launches, 160 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 1.06 us, warp 1.05 us, nsys graphs 1.41 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.39 | 37.2% | 0.40 | 38.4% |
| load | 0.29 | 27.9% | 0.30 | 28.8% |
| store | 0.29 | 27.5% | 0.26 | 25.5% |
| compute | 0.08 | 7.4% | 0.08 | 7.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.29 | 27.5% | 1.0 | 0.32 | `if (((int64_t)0 <= slot) & ((slot >> (int64_t)7) < ((int64_t)pages))) \| if (((int64_t)0 <=` |
| load.global | 0.23 | 22.3% | 1.0 | 0.26 | `for (int i = 0; i < 4; ++i)` |
| compute.reduce | 0.08 | 7.4% | 1.0 | 0.10 | `amax = max(amax, __shfl_xor_sync((uint)4294967295, amax, 1, 32));` |
| load.global#2 | 0.06 | 5.5% | 1.0 | 0.06 | `int64_t slot = slots[((int64_t)((int)blockIdx.x))];` |

### cache_writer_kernel grid (16, 1, 1) block (128, 1, 1)

48 launches, 64 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 0.99 us, warp 0.97 us, nsys graphs 1.12 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.32 | 33.4% | 0.33 | 33.1% |
| store | 0.29 | 29.8% | 0.29 | 29.5% |
| load | 0.28 | 29.1% | 0.29 | 29.7% |
| compute | 0.07 | 7.6% | 0.08 | 7.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.29 | 29.8% | 1.0 | 0.32 | `if (((int64_t)0 <= slot) & ((slot >> (int64_t)7) < ((int64_t)pages))) \| if (((int64_t)0 <=` |
| load.global | 0.23 | 23.4% | 1.0 | 0.26 | `for (int i = 0; i < 4; ++i)` |
| compute.reduce | 0.07 | 7.6% | 1.0 | 0.10 | `amax = max(amax, __shfl_xor_sync((uint)4294967295, amax, 1, 32));` |
| load.global#2 | 0.06 | 5.8% | 1.0 | 0.06 | `int64_t slot = slots[((int64_t)((int)blockIdx.x))];` |

### sink_policy_kernel grid (8, 1, 1) block (32, 1, 1)

17 launches, 8 warps; ranges, warp and last warp from the 4 recording CTAs' 4 warps; span 2.40 us, warp 2.37 us, nsys graphs 2.56 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.98 | 41.2% | 1.04 | 38.6% |
| unattributed | 0.88 | 37.2% | 1.11 | 41.2% |
| store | 0.51 | 21.6% | 0.54 | 20.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 | 0.57 | 24.0% | 1.0 | 0.58 | `state[0] = idx_mapping[((int64_t)((int)blockIdx.x))];` |
| store.global#3 | 0.46 | 19.3% | 1.0 | 0.48 | `for (int a = 0; a < 3; ++a) \| recovered_logprobs[((int64_t)((int)blockIdx.x))] = 0x0p+0f/*` |
| load.global | 0.22 | 9.1% | 1.0 | 0.22 | `start[0] = cu_num_logits[((int64_t)((int)blockIdx.x))];` |
| load.global#4 | 0.14 | 5.7% | 1.0 | 0.16 | `for (int r = 0; r < 2; ++r) \| for (int r_2 = 0; r_2 < 8; ++r_2)` |
| load.global#5 | 0.06 | 2.4% | 1.0 | 0.06 | `AtomicAdd((&(counts[0])), draft_zero_flag[0]);` |
| store.global#4 | 0.06 | 2.4% | 1.0 | 0.06 | `events[(((int64_t)((int)blockIdx.x)) * (int64_t)6)] = outcome[0];` |

### sink_cut_kernel grid (1, 1, 1) block (128, 1, 1)

34 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 1.34 us, warp 0.60 us, nsys graphs 2.11 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.32 | 52.7% | 0.31 | 22.9% |
| store | 0.16 | 27.3% | 0.53 | 39.3% |
| load | 0.12 | 20.0% | 0.51 | 37.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.16 | 27.3% | 0.8 | 0.51 | `if (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) < num_cut) \| saved[((((int64_t)((int)` |
| load.global | 0.12 | 20.0% | 0.1 | 0.48 | `for (int b = 0; b < num_reqs; ++b)` |

### metadata_pages_kernel grid (48, 1, 1) block (128, 1, 1)

34 launches, 192 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 1.28 us, warp 0.86 us, nsys graphs 1.50 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.68 | 79.6% | 0.77 | 60.1% |
| load | 0.14 | 15.8% | 0.19 | 15.0% |
| store | 0.04 | 4.7% | 0.32 | 24.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.13 | 14.8% | 1.0 | 0.13 | `if ((((offset + ((int)blockIdx.x)) < src_rows) && (0 <= (offset + ((int)blockIdx.x)))))` |
| store.global | 0.04 | 4.7% | 0.1 | 0.32 | `if (((int)threadIdx.x) < 32)` |
| load.global#2 | 0.01 | 0.9% | 0.1 | 0.06 | `if (((req < table_rows) && (((int)threadIdx.x) < table_cols)))` |

### norm_forward_and_per_token_cast_kernel_kernel grid (16, 1, 1) block (128, 1, 1)

16 launches, 64 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 2.40 us, warp 2.32 us, nsys graphs 1.89 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 2.02 | 86.9% | 2.64 | 99.4% |
| load | 0.44 | 18.9% | 0.43 | 16.2% |
| compute | 0.13 | 5.5% | 0.13 | 4.9% |
| wait | 0.06 | 2.4% | 0.06 | 2.3% |
| unattributed | 0.00 | 0.0% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*5 (est.: sampled iterations x5) | 1.76 | 75.8% | 1.0 | 2.08 | `for (int j_1 = 0; j_1 < 2; ++j_1)` |
| load.global#2*5 (est.: sampled iterations x5) | 0.22 | 9.5% | 1.0 | 0.32 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 1024) + (((int)threadIdx.x) * 8)` |
| load.global*5 (est.: sampled iterations x5) | 0.22 | 9.5% | 1.0 | 0.32 | `*(uint4*)(x_local + (vi * 8)) = *(uint4*)(x + ((((((int64_t)vi) * (int64_t)1024) + (((int6` |
| store.global*5 (est.: sampled iterations x5) | 0.21 | 8.8% | 1.0 | 0.32 | `for (int j = 0; j < 4; ++j)` |
| compute.reduce | 0.13 | 5.5% | 1.0 | 0.16 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| wait.sync | 0.06 | 2.4% | 1.0 | 0.10 | `__syncthreads();` |
| store.global#2 | 0.04 | 1.8% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| store.global#3 | 0.01 | 0.4% | 0.2 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |

### ced_copy_rows_kernel grid (200, 1, 1) block (128, 1, 1)

17 launches, 800 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 1.25 us, warp 1.06 us, nsys graphs 1.57 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.73 | 69.5% | 1.98 | 85.7% |
| unattributed | 0.32 | 30.5% | 0.33 | 14.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.73 | 69.5% | 1.0 | 0.80 | `for (int k = 0; k < 4; ++k) \| int64_t index = ced_ldg64((&(row_indices[((((int64_t)blockId` |

### rope_kernel grid (1, 1, 1) block (128, 1, 1)

48 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 0.67 us, warp 0.67 us, nsys graphs 1.06 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.42 | 61.9% | 0.55 | 62.9% |
| load | 0.22 | 33.3% | 0.28 | 32.6% |
| store | 0.03 | 4.8% | 0.04 | 4.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 | 0.13 | 19.0% | 1.0 | 0.13 | `if (((position < ((int64_t)positions_max)) && ((int64_t)0 <= position)))` |
| load.global | 0.10 | 14.3% | 1.0 | 0.10 | `if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < tokens)) \| int64_t position ` |
| store.global | 0.03 | 4.8% | 1.0 | 0.03 | `*(uint4*)(x + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + ((((int64_t)((int)thread` |

### compute_slot_mappings_kernel grid (1, 11, 1) block (256, 1, 1)

17 launches, 88 warps; ranges, warp and last warp from the 6 recording CTAs' 48 warps; span 2.34 us, warp 1.44 us, nsys graphs 2.69 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.81 | 56.7% | 0.81 | 35.1% |
| unattributed | 0.62 | 43.3% | 1.50 | 64.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.81 | 56.7% | 1.0 | 0.83 | `int first_token = query_start_loc[0];` |

### gather_block_tables_kernel grid (11, 8, 1) block (256, 1, 1)

17 launches, 704 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 2.37 us, warp 1.45 us, nsys graphs 2.35 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.15 | 79.3% | 1.74 | 84.2% |
| unattributed | 0.30 | 20.7% | 0.33 | 15.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.15 | 79.3% | 1.0 | 1.79 | `int g = chunk_groups[((int64_t)((int)blockIdx.x))];` |

### post_update_kernel grid (1, 1, 1) block (32, 1, 1)

17 launches, 1 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 1.95 us, warp 1.95 us, nsys graphs 2.30 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.70 | 36.1% | 0.69 | 30.4% |
| store | 0.67 | 34.4% | 1.01 | 44.6% |
| unattributed | 0.58 | 29.5% | 0.56 | 25.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 | 0.51 | 26.2% | 1.0 | 0.51 | `int count = num_sampled[((((int64_t)((int)blockIdx.x)) * (int64_t)32) + ((int64_t)((int)th` |
| store.global#2 | 0.51 | 26.2% | 1.0 | 0.51 | `if (0 < count)` |
| store.global | 0.16 | 8.2% | 1.0 | 0.16 | `if (0 < count)` |
| load.global | 0.10 | 4.9% | 1.0 | 0.10 | `int64_t req = idx_mapping[((((int64_t)((int)blockIdx.x)) * (int64_t)32) + ((int64_t)((int)` |
| load.global#3 | 0.10 | 4.9% | 1.0 | 0.10 | `if ((0 < count))` |

### per_token_cast_kernel_kernel grid (120, 1, 1) block (64, 1, 1)

19 launches, 240 warps; ranges, warp and last warp from the 4 recording CTAs' 8 warps; span 1.82 us, warp 1.72 us, nsys graphs 1.44 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.96 | 55.8% | 1.02 | 55.0% |
| unattributed | 0.36 | 20.9% | 0.43 | 23.5% |
| store | 0.30 | 17.2% | 0.30 | 16.2% |
| compute | 0.10 | 6.0% | 0.10 | 5.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.87 | 50.7% | 1.0 | 0.99 | `for (int i = 0; i < 2; ++i)` |
| store.global#2 | 0.11 | 6.5% | 1.0 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.10 | 6.0% | 1.0 | 0.13 | `amax_fragment[i_1] = tl::AllReduce<tl::MaxOp, 2, 1, 0, tl::NamedBarrier<64>>::run(amax_fra` |
| store.global*2 (est.: sampled iterations x2) | 0.10 | 5.6% | 1.0 | 0.13 | `out_sf[((((((int64_t)((int)blockIdx.x)) % (int64_t)15) * (int64_t)32) + (((int64_t)((int)t` |
| load.global#2*2 (est.: sampled iterations x2) | 0.09 | 5.1% | 1.0 | 0.13 | `amax_fragment[i_1] = cutlass::bfloat16_t(__hmax((bfloat16_t(((nv_bfloat162*)(&(amax_fragme` |
| store.global#3*2 (est.: sampled iterations x2) | 0.09 | 5.1% | 1.0 | 0.13 | `*(fp8_e4_16_t*)(out + ((((((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int64_t)10240) +` |

### rope_kernel grid (48, 1, 1) block (256, 1, 1)

34 launches, 384 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 0.99 us, warp 0.88 us, nsys graphs 1.34 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.59 | 67.3% | 0.69 | 69.6% |
| load | 0.25 | 28.1% | 0.26 | 25.9% |
| store | 0.04 | 4.5% | 0.04 | 4.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.25 | 28.1% | 1.0 | 0.26 | `if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < tokens)) \| int64_t position ` |
| store.global | 0.04 | 4.5% | 1.0 | 0.06 | `*(uint4*)(x + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + ((((int64_t)((int)thread` |

### insert_kernel grid (8, 1, 1) block (32, 1, 1)

17 launches, 8 warps; ranges, warp and last warp from the 4 recording CTAs' 4 warps; span 1.79 us, warp 1.74 us, nsys graphs 1.63 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.75 | 43.1% | 0.87 | 43.4% |
| unattributed | 0.50 | 28.9% | 0.49 | 24.7% |
| store | 0.34 | 19.3% | 0.44 | 22.0% |
| compute | 0.15 | 8.7% | 0.20 | 10.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#3 | 0.58 | 33.0% | 1.0 | 0.61 | `int state = mapping[((int64_t)position)]; \| int64_t state = mapping[((int64_t)position)];` |
| store.global | 0.29 | 16.5% | 1.0 | 0.29 | `counts[((int64_t)((int)blockIdx.x))] = (count_local + 1); \| counts[((int64_t)((int)blockId` |
| compute.reduce#2 | 0.12 | 6.9% | 1.0 | 0.13 | `float value_1 = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| load.global | 0.11 | 6.4% | 1.0 | 0.13 | `for (int k = 0; k < 4; ++k)` |
| load.global#2 | 0.06 | 3.7% | 1.0 | 0.06 | `count_local = counts[((int64_t)((int)blockIdx.x))];` |
| store.global#2 | 0.05 | 2.8% | 1.0 | 0.06 | `if (((int)threadIdx.x) == 0)` |
| compute.reduce | 0.03 | 1.8% | 1.0 | 0.03 | `int count = __shfl_sync((uint)4294967295, count_local, 0, 32);` |

### sink_flags_kernel grid (1, 1, 1) block (128, 1, 1)

17 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 1.95 us, warp 0.66 us, nsys graphs 2.85 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.33 | 50.0% | 1.30 | 66.2% |
| unattributed | 0.24 | 36.6% | 0.32 | 16.1% |
| store | 0.09 | 13.4% | 0.35 | 17.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.33 | 50.0% | 0.2 | 1.28 | `token = logits_indices[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)th` |
| store.global | 0.09 | 13.4% | 0.2 | 0.35 | `flags[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = (` |

### logits_rows_kernel grid (1, 1, 1) block (128, 1, 1)

17 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 1.15 us, warp 1.10 us, nsys graphs 2.05 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.51 | 46.4% | 0.86 | 49.4% |
| unattributed | 0.46 | 41.3% | 0.29 | 16.6% |
| store | 0.14 | 12.3% | 0.59 | 34.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.51 | 46.4% | 1.0 | 0.51 | `index = logits_indices[min(((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((in` |
| store.global | 0.14 | 12.3% | 1.0 | 0.35 | `if (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) < num_logits)` |

### gemm_splitk_reduce_kernel grid (4, 16, 1) block (128, 1, 1)

48 launches, 256 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 0.61 us, warp 0.53 us, nsys graphs 0.77 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.25 | 46.8% | 0.27 | 45.4% |
| load | 0.24 | 45.3% | 0.27 | 46.5% |
| store | 0.04 | 7.9% | 0.05 | 8.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.24 | 45.3% | 1.0 | 0.29 | `acc[0] = P[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x))` |
| store.global | 0.04 | 7.9% | 1.0 | 0.06 | `C[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x)) * (int64` |

### token_counts_kernel grid (8, 1, 1) block (128, 1, 1)

17 launches, 32 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 1.15 us, warp 0.48 us, nsys graphs 1.28 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.28 | 58.9% | 0.38 | 23.8% |
| store | 0.14 | 29.5% | 0.99 | 61.7% |
| load | 0.06 | 11.6% | 0.23 | 14.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.14 | 29.5% | 0.2 | 0.58 | `if (((int)threadIdx.x) == 0) \| sampled[((int64_t)((int)blockIdx.x))] = (((int)(prefill_len` |
| load.global | 0.06 | 11.6% | 0.2 | 0.22 | `int count = sampled[((int64_t)((int)blockIdx.x))];` |

### assign_draft_budget_kernel grid (1, 1, 1) block (256, 1, 1)

16 launches, 8 warps; ranges, warp and last warp from the 1 recording CTAs' 8 warps; span 1.65 us, warp 1.63 us, nsys graphs 1.89 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 0.62 | 38.2% | 0.39 | 23.7% |
| unattributed, with loops' unsampled iterations | 0.62 | 37.8% | 0.76 | 45.5% |
| load | 0.20 | 12.2% | 0.32 | 19.4% |
| store | 0.19 | 11.8% | 0.19 | 11.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync | 0.31 | 19.1% | 1.0 | 0.35 | `__syncthreads();` |
| wait.sync#3 | 0.25 | 15.2% | 1.0 | 0.32 | `__syncthreads();` |
| load.global | 0.20 | 12.2% | 1.0 | 0.48 | `for (int chunk = 0; chunk < ((num_reqs + 255) >> 8); ++chunk)` |
| store.global | 0.19 | 11.8% | 1.0 | 0.19 | `for (int chunk_2 = 0; chunk_2 < ((num_reqs + 255) >> 8); ++chunk_2)` |
| wait.sync#2 (sampled iterations) | 0.06 | 3.9% | 1.0 | 0.06 | `__syncthreads();` |

### combine_sampled_and_draft_tokens_kernel grid (8, 1, 1) block (32, 1, 1)

17 launches, 8 warps; ranges, warp and last warp from the 4 recording CTAs' 4 warps; span 1.31 us, warp 1.17 us, nsys graphs 1.44 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.64 | 54.8% | 0.68 | 56.4% |
| unattributed | 0.43 | 37.0% | 0.44 | 36.0% |
| store | 0.10 | 8.2% | 0.09 | 7.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.64 | 54.8% | 1.0 | 0.67 | `if (((((int)blockIdx.x) * 32) + ((int)threadIdx.x)) < num_reqs) \| req = idx_mapping[((int6` |
| store.global | 0.10 | 8.2% | 1.0 | 0.10 | `if ((prefill < seq) && (1 < num_logits))` |

### flatten_sampled_kernel grid (1, 1, 1) block (128, 1, 1)

17 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 1.09 us, warp 0.80 us, nsys graphs 1.31 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.47 | 59.0% | 0.72 | 60.5% |
| unattributed | 0.30 | 38.0% | 0.38 | 31.7% |
| store | 0.02 | 3.0% | 0.09 | 7.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.42 | 52.0% | 1.0 | 0.42 | `for (int r = 0; r < num_reqs; ++r)` |
| load.global#2 | 0.06 | 7.0% | 0.2 | 0.22 | `int i = (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) - cu_num_logits[req]);` |
| store.global | 0.02 | 3.0% | 0.2 | 0.10 | `flat[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = co` |

### prepare_pos_seq_lens_kernel grid (9, 1, 1) block (256, 1, 1)

17 launches, 72 warps; ranges, warp and last warp from the 5 recording CTAs' 40 warps; span 1.12 us, warp 0.95 us, nsys graphs 1.30 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.46 | 47.7% | 0.57 | 51.4% |
| unattributed | 0.38 | 40.2% | 0.41 | 36.8% |
| store | 0.12 | 12.1% | 0.13 | 11.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.46 | 47.7% | 0.8 | 0.58 | `int computed = num_computed_tokens[idx_mapping[((int64_t)((int)blockIdx.x))]];` |
| store.global#2 | 0.10 | 10.7% | 0.8 | 0.13 | `if (((int)threadIdx.x) == 0)` |
| store.global | 0.01 | 1.3% | 0.2 | 0.06 | `for (int it = 0; it < (((max_reqs + 255) - num_reqs) >> 8); ++it)` |

### scatter_rows_gather_kernel grid (1, 1, 1) block (128, 1, 1)

17 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 0.93 us, warp 0.86 us, nsys graphs 1.33 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.53 | 61.7% | 0.40 | 42.8% |
| store | 0.23 | 27.1% | 0.42 | 45.6% |
| load | 0.10 | 11.2% | 0.11 | 11.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.23 | 27.1% | 1.0 | 0.42 | `if (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) < (num_reqs * width))` |
| load.global | 0.10 | 11.2% | 1.0 | 0.10 | `index = idx_mapping[min((((((int64_t)0 <= ((int64_t)width)) && ((int64_t)0 <= rmod_1)) \|\| ` |

### padding_rows_kernel grid (1, 1, 1) block (256, 1, 1)

33 launches, 8 warps; ranges, warp and last warp from the 1 recording CTAs' 8 warps; span 0.38 us, warp 0.36 us, nsys graphs 0.74 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.27 | 75.3% | 0.37 | 79.2% |
| store | 0.09 | 24.7% | 0.10 | 20.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.09 | 24.7% | 1.0 | 0.10 | `if (((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) < rows)` |

### ced_copy_rows_kernel grid (80, 1, 1) block (128, 1, 1)

16 launches, 320 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 0.67 us, warp 0.58 us, nsys graphs 1.74 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.35 | 59.5% | 0.40 | 60.3% |
| unattributed | 0.24 | 40.5% | 0.26 | 39.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.35 | 59.5% | 1.0 | 0.42 | `for (int k = 0; k < 4; ++k) \| int64_t index = ced_ldg64((&(row_indices[((((int64_t)blockId` |

### scatter_rows_kernel grid (1, 1, 1) block (128, 1, 1)

17 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 0.64 us, warp 0.59 us, nsys graphs 0.99 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.39 | 66.2% | 0.39 | 62.3% |
| store | 0.10 | 17.6% | 0.13 | 21.3% |
| load | 0.10 | 16.2% | 0.10 | 16.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.10 | 17.6% | 1.0 | 0.13 | `if (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) < (num_reqs * width))` |
| load.global | 0.10 | 16.2% | 1.0 | 0.10 | `index = idx_mapping[min((((((int64_t)0 <= ((int64_t)width)) && ((int64_t)0 <= rmod_1)) \|\| ` |

### ced_requests_kernel grid (1, 1, 1) block (128, 1, 1)

17 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 0.61 us, warp 0.54 us, nsys graphs 1.15 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.32 | 59.7% | 0.32 | 51.8% |
| load | 0.12 | 22.4% | 0.11 | 18.6% |
| store | 0.10 | 17.9% | 0.18 | 29.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.12 | 22.4% | 1.0 | 0.13 | `uint v_ = tl::load_global_32((&(starts[1]))); \| uint v_ = tl::load_global_32((&(starts[min` |
| store.global | 0.10 | 17.9% | 1.0 | 0.19 | `if (((int)threadIdx.x) < 8) \| if (((int)threadIdx.x) <= 8)` |

### context_inputs_kernel grid (5, 48, 1) block (128, 1, 1)

1 launches, 960 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 8.67 us, warp 7.73 us, nsys graphs 11.55 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 7.12 | 92.1% | 7.55 | 87.1% |
| unattributed | 0.45 | 5.9% | 0.74 | 8.5% |
| store | 0.15 | 2.0% | 0.38 | 4.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 7.12 | 92.1% | 1.0 | 7.71 | `vllm_dspark_context_b128((&(aux[(((((int64_t)((int)blockIdx.y)) * (int64_t)15360) + (((int` |
| store.global | 0.15 | 2.0% | 1.0 | 0.42 | `if (((((int)blockIdx.x) == 0) & (((int)threadIdx.x) == 0)) & (num_tokens <= ((int)blockIdx` |

### per_token_cast_kernel_kernel grid (360, 1, 1) block (64, 1, 1)

1 launches, 720 warps; ranges, warp and last warp from the 4 recording CTAs' 8 warps; span 3.58 us, warp 3.18 us, nsys graphs 4.61 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 2.21 | 69.6% | 2.27 | 64.0% |
| load | 0.74 | 23.4% | 0.90 | 25.2% |
| compute | 0.22 | 6.8% | 0.13 | 3.6% |
| unattributed | 0.00 | 0.1% | 0.26 | 7.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#3*2 (est.: sampled iterations x2) | 1.12 | 35.3% | 1.0 | 3.46 | `*(fp8_e4_16_t*)(out + ((((((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int64_t)10240) +` |
| load.global | 0.66 | 20.9% | 1.0 | 0.80 | `for (int i = 0; i < 2; ++i)` |
| store.global*2 (est.: sampled iterations x2) | 0.63 | 19.9% | 1.0 | 2.05 | `out_sf[((((((int64_t)((int)blockIdx.x)) % (int64_t)15) * (int64_t)32) + (((int64_t)((int)t` |
| store.global#2 | 0.46 | 14.5% | 1.0 | 1.06 | `for (int i_3 = 0; i_3 < 4; ++i_3)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.22 | 6.8% | 1.0 | 0.90 | `amax_fragment[i_1] = tl::AllReduce<tl::MaxOp, 2, 1, 0, tl::NamedBarrier<64>>::run(amax_fra` |
| load.global#2*2 (est.: sampled iterations x2) | 0.08 | 2.5% | 1.0 | 0.13 | `amax_fragment[i_1] = cutlass::bfloat16_t(__hmax((bfloat16_t(((nv_bfloat162*)(&(amax_fragme` |

### gemm_splitk_reduce_kernel grid (4, 48, 1) block (128, 1, 1)

3 launches, 768 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 0.77 us, warp 0.71 us, nsys graphs 1.22 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.39 | 54.5% | 0.42 | 55.7% |
| unattributed | 0.27 | 38.2% | 0.28 | 37.1% |
| store | 0.05 | 7.3% | 0.05 | 7.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.39 | 54.5% | 1.0 | 0.42 | `acc[0] = P[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x))` |
| store.global | 0.05 | 7.3% | 1.0 | 0.06 | `C[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x)) * (int64` |

### ced_copy_rows_kernel grid (240, 1, 1) block (128, 1, 1)

1 launches, 960 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 1.12 us, warp 0.80 us, nsys graphs 2.50 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.50 | 62.3% | 0.54 | 65.4% |
| unattributed | 0.30 | 37.7% | 0.29 | 34.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.50 | 62.3% | 1.0 | 0.54 | `for (int k = 0; k < 4; ++k) \| int64_t index = ced_ldg64((&(row_indices[((((int64_t)blockId` |
=== timeline and tables: exit 0 in 36 s
