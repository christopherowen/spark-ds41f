=== timeline and tables
trace /c/iket/step-tp3-c8-20261026T192627Z/trace/iket/pid_0x18df/iket.decoded_results.json: 16143 launches, graphLaunches dict of 89

## Timeline: 7547 launches in 1195419.3 us; the GPU is busy 364016.3 us (30%) and idle 831403.0 us in 5843 gaps; gaps over 200 us split it into 892 segments

The 15 of 892 segments with the most busy time:

| segment | launches | wall us | busy us | busy | first kernel |
|---:|---:|---:|---:|---:|---|
| 118 | 93 | 10708.8 | 6839.8 | 64% | padding_rows_kernel |
| 124 | 259 | 7057.4 | 6315.9 | 89% | scatter_rows_gather_kernel |
| 170 | 238 | 6314.8 | 6033.8 | 96% | metadata_builder_kernel |
| 220 | 250 | 6508.7 | 6030.9 | 93% | ced_requests_kernel |
| 257 | 256 | 6705.9 | 6059.3 | 90% | assign_draft_budget_kernel |
| 304 | 257 | 7148.4 | 6034.1 | 84% | padding_rows_kernel |
| 345 | 262 | 7815.5 | 6076.8 | 78% | partial_argmax_kernel |
| 386 | 261 | 8162.0 | 6083.7 | 75% | finish_argmax_kernel |
| 415 | 264 | 7920.4 | 6498.2 | 82% | finish_argmax_kernel |
| 442 | 251 | 6883.3 | 6065.8 | 88% | compute_slot_mappings_kernel |
| 491 | 238 | 6521.3 | 6036.1 | 93% | metadata_builder_kernel |
| 700 | 248 | 7564.3 | 6613.9 | 87% | metadata_builder_kernel |
| 753 | 242 | 7219.6 | 6437.2 | 89% | metadata_builder_kernel |
| 821 | 239 | 6461.8 | 6044.4 | 94% | metadata_builder_kernel |
| 865 | 256 | 7598.8 | 6090.6 | 80% | assign_draft_budget_kernel |

| idle us | after | before |
|---:|---|---|
| 105700.8 | resample_kernel | insert_kernel |
| 72657.4 | scatter_rows_kernel | padding_rows_kernel |
| 10524.0 | norm_forward_and_per_token_cast_kernel_kernel | sink_flags_kernel |
| 10505.3 | norm_forward_and_per_token_cast_kernel_kernel | sink_flags_kernel |
| 10504.0 | norm_forward_and_per_token_cast_kernel_kernel | sink_flags_kernel |
| 10489.3 | norm_forward_and_per_token_cast_kernel_kernel | sink_flags_kernel |
| 10286.0 | norm_forward_and_per_token_cast_kernel_kernel | sink_flags_kernel |
| 10193.8 | norm_forward_and_per_token_cast_kernel_kernel | sink_flags_kernel |
| 10120.8 | norm_forward_and_per_token_cast_kernel_kernel | sink_flags_kernel |
| 10103.3 | norm_forward_and_per_token_cast_kernel_kernel | sink_flags_kernel |

## Engine steps

| step | kind | launches | wall us | busy us | busy | most span |
|---:|---|---:|---:|---:|---:|---|
| 0 | prefill | 410 | 84689.0 | 30972.5 | 37% | vocab_packed_projection_kernel 8990, moe_grouped_fp8_fp4_gemm_kernel 6699, gemm_mxfp8_kernel 5269 |
| 1 | decode | 420 | 150622.5 | 20063.6 | 13% | vocab_packed_projection_kernel 9180, gemm_mxfp8_kernel 2869, moe_grouped_fp8_fp4_gemm_kernel 2437 |
| 2 | decode | 420 | 120345.9 | 19816.6 | 16% | vocab_packed_projection_kernel 8753, gemm_mxfp8_kernel 3014, moe_grouped_fp8_fp4_gemm_kernel 2867 |
| 3 | decode | 420 | 50196.7 | 19640.6 | 39% | vocab_packed_projection_kernel 8812, gemm_mxfp8_kernel 2939, moe_grouped_fp8_fp4_gemm_kernel 2855 |
| 4 | decode | 420 | 49111.3 | 19443.5 | 40% | vocab_packed_projection_kernel 8653, gemm_mxfp8_kernel 2847, moe_grouped_fp8_fp4_gemm_kernel 2793 |
| 5 | decode | 420 | 52198.2 | 19533.4 | 37% | vocab_packed_projection_kernel 8585, gemm_mxfp8_kernel 2875, moe_grouped_fp8_fp4_gemm_kernel 2811 |
| 6 | decode | 420 | 48627.9 | 19499.9 | 40% | vocab_packed_projection_kernel 8534, gemm_mxfp8_kernel 3012, moe_grouped_fp8_fp4_gemm_kernel 2839 |
| 7 | decode | 420 | 50173.7 | 19455.8 | 39% | vocab_packed_projection_kernel 8582, gemm_mxfp8_kernel 2808, moe_grouped_fp8_fp4_gemm_kernel 2774 |
| 8 | decode | 420 | 46013.0 | 19343.7 | 42% | vocab_packed_projection_kernel 8562, gemm_mxfp8_kernel 2844, moe_grouped_fp8_fp4_gemm_kernel 2788 |
| 9 | decode | 420 | 46371.6 | 19825.6 | 43% | vocab_packed_projection_kernel 8928, gemm_mxfp8_kernel 2837, moe_grouped_fp8_fp4_gemm_kernel 2828 |
| 10 | decode | 420 | 52955.6 | 19403.9 | 37% | vocab_packed_projection_kernel 8638, gemm_mxfp8_kernel 2846, moe_grouped_fp8_fp4_gemm_kernel 2782 |
| 11 | decode | 420 | 76769.8 | 19445.9 | 25% | vocab_packed_projection_kernel 8664, gemm_mxfp8_kernel 2808, moe_grouped_fp8_fp4_gemm_kernel 2774 |
| 12 | decode | 420 | 57465.3 | 19283.5 | 34% | vocab_packed_projection_kernel 8522, gemm_mxfp8_kernel 2814, moe_grouped_fp8_fp4_gemm_kernel 2752 |
| 13 | decode | 420 | 68955.4 | 19441.4 | 28% | vocab_packed_projection_kernel 8723, gemm_mxfp8_kernel 2817, moe_grouped_fp8_fp4_gemm_kernel 2778 |
| 14 | decode | 420 | 58922.8 | 20118.0 | 34% | vocab_packed_projection_kernel 8702, gemm_mxfp8_kernel 3076, moe_grouped_fp8_fp4_gemm_kernel 2913 |
| 15 | decode | 420 | 69085.4 | 19827.9 | 29% | vocab_packed_projection_kernel 8475, moe_grouped_fp8_fp4_gemm_kernel 2951, gemm_mxfp8_kernel 2949 |
| 16 | decode | 420 | 59367.9 | 19354.3 | 33% | vocab_packed_projection_kernel 8575, gemm_mxfp8_kernel 2823, moe_grouped_fp8_fp4_gemm_kernel 2788 |
| 17 | decode | 417 | 50675.8 | 19546.0 | 39% | vocab_packed_projection_kernel 8757, gemm_mxfp8_kernel 2849, moe_grouped_fp8_fp4_gemm_kernel 2828 |

## Overview over 17 decode steps: 7137 launches of 127 kernel shapes, spans summing to 358448.0 us (21085.2 a step)

| kernel | grid | launches a step | span us a step | share | span us a launch | last warp: wait | load | compute | store | unattributed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vocab_packed_projection_kernel | (1, 4040, 1) | 1.05882 | 4878.1 | 23.1% | 4621.94 | 25% | 27% | 25% | 0% | 23% |
| vocab_packed_projection_kernel | (1, 4040, 1) | 0.941176 | 3806.9 | 18.1% | 4034.86 | 44% | 28% | 11% | 0% | 17% |
| gemm_bf16_kernel | (2020, 1, 1) | 5 | 1724.8 | 8.2% | 346.14 | 90% | 0% | 4% | 1% | 5% |
| l2_prefetch_kernel | (2, 1, 1) | 17 | 1381.3 | 6.6% | 72.29 | 0% | 100% | 0% | 0% | 0% |
| gemm_mxfp8_kernel | (80, 1, 1) | 19 | 1316.0 | 6.2% | 60.93 | 54% | 9% | 13% | 1% | 23% |
| moe_grouped_fp8_fp4_gemm_kernel | (40, 288, 1) | 6 | 1175.4 | 5.6% | 194.69 | 8% | 9% | 0% | 0% | 84% |
| moe_grouped_fp8_fp4_gemm_kernel | (12, 288, 1) | 6 | 1104.7 | 5.2% | 186.29 | 12% | 14% | 0% | 0% | 73% |
| mhc_project_streams_kernel | (40, 3, 1) | 18 | 848.6 | 4.0% | 49.70 | 8% | 34% | 19% | 4% | 35% |
| gemm_mxfp8_kernel | (192, 1, 1) | 9 | 740.0 | 3.5% | 70.85 | 36% | 8% | 15% | 1% | 40% |
| gemm_fp8w_bf16_partials_kernel | (48, 4, 1) | 9 | 636.0 | 3.0% | 80.93 | 37% | 13% | 17% | 1% | 32% |
| gemm_mxfp8_kernel | (28, 1, 1) | 9 | 405.8 | 1.9% | 41.60 | 41% | 9% | 24% | 0% | 25% |
| gemm_mxfp8_kernel | (24, 1, 1) | 9 | 372.0 | 1.8% | 40.74 | 45% | 17% | 33% | 1% | 5% |
| sparse_mla_kernel | (1, 48, 1) | 4 | 300.0 | 1.4% | 65.95 | 2% | 9% | 2% | 1% | 86% |
| moe_grouped_fp8_fp4_gemm_kernel | (40, 120, 1) | 3 | 266.8 | 1.3% | 86.72 | 15% | 19% | 0% | 0% | 66% |
| moe_grouped_fp8_fp4_gemm_kernel | (12, 120, 1) | 3 | 250.7 | 1.2% | 81.47 | 16% | 19% | 0% | 0% | 65% |
| partial_argmax_kernel | (8, 64, 1) | 5 | 159.9 | 0.8% | 31.62 | 2% | 69% | 1% | 0% | 27% |
| gemm_bf16_splitk_kernel | (16, 5, 1) | 2 | 145.4 | 0.7% | 73.79 | 84% | 14% | 6% | 1% | 0% |
| gemm_bf16_partials_kernel | (6, 8, 1) | 6 | 136.4 | 0.6% | 20.50 | 25% | 0% | 29% | 2% | 45% |
| mhc_finalize_kernel | (48, 1, 1) | 12 | 95.5 | 0.5% | 6.83 | 12% | 19% | 9% | 17% | 43% |
| sparse_mla_kernel | (2, 1, 40) | 3 | 83.2 | 0.4% | 27.17 | 19% | 12% | 46% | 6% | 17% |

| wait | kernel | grid | last-warp us a step | a launch | first statement |
|---|---|---|---:|---:|---|
| wait.cp_async*118 | gemm_mxfp8_kernel | (80, 1, 1) | 184.1 | 9.69 | `tl::cp_async_wait<2>();` |
| wait.cp_async*19 | gemm_mxfp8_kernel | (28, 1, 1) | 98.5 | 10.94 | `tl::cp_async_wait<1>();` |
| wait.cp_async*22 | gemm_mxfp8_kernel | (80, 1, 1) | 70.6 | 3.71 | `tl::cp_async_wait<2>();` |
| wait.cp_async*38 | gemm_mxfp8_kernel | (24, 1, 1) | 67.5 | 7.49 | `tl::cp_async_wait<2>();` |
| wait.sync#2*38 | gemm_mxfp8_kernel | (24, 1, 1) | 66.2 | 7.36 | `__syncthreads();` |
| wait.sync#2*118 | gemm_mxfp8_kernel | (80, 1, 1) | 65.7 | 3.46 | `__syncthreads();` |
| wait.cp_async*13 | gemm_bf16_splitk_kernel | (16, 5, 1) | 55.1 | 27.55 | `tl::cp_async_wait<2>();` |
| wait.sync#2*22 | gemm_mxfp8_kernel | (80, 1, 1) | 50.6 | 2.66 | `__syncthreads();` |
| wait.sync#2*19 | gemm_mxfp8_kernel | (28, 1, 1) | 47.4 | 5.27 | `__syncthreads();` |
| wait.mbarrier#2 | gemm_bf16_kernel | (2020, 1, 1) | 30.4 | 6.09 | `mbarrier[0].wait(0);` |
| wait.cp_async*14 | gemm_fp8w_bf16_partials_kernel | (48, 4, 1) | 27.8 | 3.09 | `tl::cp_async_wait<3>();` |
| wait.cp_async*9 | gemm_mxfp8_kernel | (64, 1, 1) | 26.8 | 13.40 | `tl::cp_async_wait<1>();` |
| wait.sync*13 | gemm_bf16_splitk_kernel | (16, 5, 1) | 25.0 | 12.52 | `__syncthreads();` |
| wait.sync#2*14 | gemm_fp8w_bf16_partials_kernel | (48, 4, 1) | 24.8 | 2.76 | `__syncthreads();` |
| wait.sync*38 | gemm_mxfp8_kernel | (24, 1, 1) | 21.0 | 2.33 | `__syncthreads();` |
| wait.sync#2*8 | gemm_mxfp8_kernel | (192, 1, 1) | 18.4 | 2.04 | `__syncthreads();` |
| wait.cp_async#3 | gemm_mxfp8_kernel | (192, 1, 1) | 15.6 | 1.73 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>();` |
| wait.cp_async*8 | gemm_mxfp8_kernel | (192, 1, 1) | 15.4 | 1.71 | `tl::cp_async_wait<2>();` |
| wait.mbarrier#2*10 | gemm_bf16_partials_kernel | (6, 8, 1) | 15.1 | 2.51 | `mbarrier[(kb_1 % 3)].wait(((kb_1 % 6) / 3));` |
| wait.sync*4 | mhc_project_streams_kernel | (40, 3, 1) | 14.9 | 0.83 | `__syncthreads();` |

### vocab_packed_projection_kernel grid (1, 4040, 1) block (256, 1, 1)

18 launches, 32320 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 4621.94 us, warp 57.98 us, nsys graphs 4522.08 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 19.12 | 33.0% | 14.17 | 25.3% |
| unattributed | 14.82 | 25.6% | 12.85 | 22.9% |
| compute | 13.72 | 23.7% | 14.03 | 25.0% |
| load | 10.26 | 17.7% | 14.92 | 26.6% |
| store | 0.06 | 0.1% | 0.06 | 0.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*37 (est.: sampled iterations x37) | 12.78 | 22.0% | 1.0 | 14.21 | `{` |
| wait.sync#3*37 (est.: sampled iterations x37) | 5.90 | 10.2% | 1.0 | 9.47 | `__syncthreads();` |
| load.cp_async#2*37 (est.: sampled iterations x37) | 5.27 | 9.1% | 1.0 | 6.51 | `for (int i_4 = 0; i_4 < 4; ++i_4)` |
| wait.sync#2*37 (est.: sampled iterations x37) | 4.68 | 8.1% | 1.0 | 10.66 | `__syncthreads();` |
| load.global | 4.64 | 8.0% | 1.0 | 10.19 | `int start = row_offsets[(((int)blockIdx.y) * 16)]; \| int start = row_offsets[(((int)blockI` |
| wait.sync*37 (est.: sampled iterations x37) | 2.52 | 4.3% | 1.0 | 3.55 | `__syncthreads();` |
| wait.cp_async*37 (est.: sampled iterations x37) | 2.37 | 4.1% | 1.0 | 10.06 | `tl::cp_async_wait<6>();` |
| wait.cp_async#2*37 (est.: sampled iterations x37) | 2.28 | 3.9% | 1.0 | 2.96 | `tl::cp_async_wait<7>();` |
| load.cp_async | 0.35 | 0.6% | 1.0 | 0.46 | `for (int i = 0; i < 4; ++i) \| for (int i = 0; i < 8; ++i) \| for (int i_1 = 0; i_1 < 4; ++i` |
| compute.mma#4 | 0.31 | 0.5% | 1.0 | 0.35 | `{` |
| compute.mma#3 | 0.31 | 0.5% | 1.0 | 0.34 | `{` |
| compute.mma#2 | 0.31 | 0.5% | 1.0 | 0.35 | `{` |
| wait.sync#8 | 0.24 | 0.4% | 1.0 | 0.80 | `__syncthreads();` |
| wait.sync#4 | 0.17 | 0.3% | 1.0 | 0.58 | `__syncthreads();` |
| wait.sync#6 | 0.16 | 0.3% | 1.0 | 0.58 | `__syncthreads();` |
| wait.sync#7 | 0.15 | 0.3% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#9 | 0.15 | 0.3% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#5 | 0.14 | 0.2% | 1.0 | 0.26 | `__syncthreads();` |
| wait.cp_async#7 | 0.09 | 0.2% | 1.0 | 0.78 | `tl::cp_async_wait<0>();` |
| wait.cp_async#5 | 0.07 | 0.1% | 1.0 | 0.58 | `tl::cp_async_wait<2>();` |
| wait.cp_async#3 | 0.07 | 0.1% | 1.0 | 0.56 | `tl::cp_async_wait<4>();` |
| store.global | 0.06 | 0.1% | 1.0 | 0.10 | `for (int i_13 = 0; i_13 < 2; ++i_13) \| for (int i_14 = 0; i_14 < 4; ++i_14) \| for (int i_9` |
| wait.cp_async#6 | 0.05 | 0.1% | 1.0 | 0.06 | `tl::cp_async_wait<3>();` |
| wait.cp_async#4 | 0.04 | 0.1% | 1.0 | 0.06 | `tl::cp_async_wait<5>();` |
| wait.cp_async#8 | 0.04 | 0.1% | 1.0 | 0.06 | `tl::cp_async_wait<1>();` |

### vocab_packed_projection_kernel grid (1, 4040, 1) block (128, 1, 1)

16 launches, 16160 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 4034.86 us, warp 52.69 us, nsys graphs 4089.25 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 20.46 | 38.8% | 20.58 | 43.8% |
| unattributed | 13.86 | 26.3% | 8.05 | 17.1% |
| load | 13.29 | 25.2% | 13.31 | 28.3% |
| compute | 5.04 | 9.6% | 5.01 | 10.7% |
| store | 0.05 | 0.1% | 0.05 | 0.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 10.67 | 20.3% | 1.0 | 26.46 | `int start = row_offsets[(((int)blockIdx.y) * 16)]; \| int start = row_offsets[(((int)blockI` |
| wait.cp_async*17 (est.: sampled iterations x17) | 7.96 | 15.1% | 1.0 | 29.38 | `tl::cp_async_wait<6>();` |
| wait.sync#2*17 (est.: sampled iterations x17) | 7.09 | 13.5% | 1.0 | 21.76 | `__syncthreads();` |
| compute.mma*17 (est.: sampled iterations x17) | 4.34 | 8.2% | 1.0 | 4.62 | `{` |
| load.cp_async#2*17 (est.: sampled iterations x17) | 2.31 | 4.4% | 1.0 | 2.72 | `for (int i_6 = 0; i_6 < 8; ++i_6) \| for (int i_9 = 0; i_9 < 4; ++i_9)` |
| wait.sync*17 (est.: sampled iterations x17) | 1.09 | 2.1% | 1.0 | 1.09 | `__syncthreads();` |
| wait.sync#3*17 (est.: sampled iterations x17) | 1.07 | 2.0% | 1.0 | 1.09 | `__syncthreads();` |
| wait.cp_async#2*17 (est.: sampled iterations x17) | 0.95 | 1.8% | 1.0 | 1.09 | `tl::cp_async_wait<7>();` |
| wait.sync#4 | 0.47 | 0.9% | 1.0 | 1.41 | `__syncthreads();` |
| wait.cp_async#3 | 0.43 | 0.8% | 1.0 | 1.87 | `tl::cp_async_wait<4>();` |
| wait.sync#6 | 0.38 | 0.7% | 1.0 | 1.30 | `__syncthreads();` |
| wait.cp_async#5 | 0.36 | 0.7% | 1.0 | 1.70 | `tl::cp_async_wait<2>();` |
| load.cp_async | 0.30 | 0.6% | 1.0 | 0.42 | `for (int i = 0; i < 4; ++i) \| for (int i = 0; i < 8; ++i) \| for (int i_1 = 0; i_1 < 4; ++i` |
| compute.mma#2 | 0.24 | 0.5% | 1.0 | 0.26 | `{` |
| compute.mma#3 | 0.23 | 0.4% | 1.0 | 0.26 | `{` |
| compute.mma#4 | 0.23 | 0.4% | 1.0 | 0.26 | `{` |
| wait.sync#8 | 0.21 | 0.4% | 1.0 | 1.09 | `__syncthreads();` |
| wait.cp_async#7 | 0.17 | 0.3% | 1.0 | 1.47 | `tl::cp_async_wait<0>();` |
| wait.sync#5 | 0.06 | 0.1% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#7 | 0.05 | 0.1% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#9 | 0.05 | 0.1% | 1.0 | 0.06 | `__syncthreads();` |
| store.global | 0.05 | 0.1% | 1.0 | 0.06 | `for (int i_13 = 0; i_13 < 2; ++i_13) \| for (int i_14 = 0; i_14 < 4; ++i_14) \| for (int i_9` |
| wait.cp_async#4 | 0.04 | 0.1% | 1.0 | 0.06 | `tl::cp_async_wait<5>();` |
| wait.cp_async#6 | 0.04 | 0.1% | 1.0 | 0.06 | `tl::cp_async_wait<3>();` |
| wait.cp_async#8 | 0.04 | 0.1% | 1.0 | 0.06 | `tl::cp_async_wait<1>();` |

### gemm_bf16_kernel grid (2020, 1, 1) block (256, 1, 1)

85 launches, 16160 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 346.14 us, warp 4.61 us, nsys graphs 272.22 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 4.04 | 87.7% | 6.17 | 90.1% |
| unattributed | 0.35 | 7.6% | 0.33 | 4.8% |
| compute | 0.14 | 3.1% | 0.29 | 4.2% |
| load | 0.04 | 0.8% | 0.00 | 0.0% |
| store | 0.04 | 0.8% | 0.06 | 0.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.mbarrier#2 | 3.94 | 85.4% | 0.5 | 10.56 | `mbarrier[0].wait(0);` |
| compute.mma | 0.14 | 3.1% | 0.5 | 0.29 | `{` |
| wait.sync | 0.08 | 1.7% | 1.0 | 0.10 | `__syncthreads();` |
| load.tma | 0.04 | 0.8% | 0.5 | 0.13 | `if (tl::tl_shuffle_elect<128>())` |
| store.global | 0.04 | 0.8% | 0.5 | 0.10 | `for (int i_3 = 0; i_3 < 4; ++i_3) \| for (int i_4 = 0; i_4 < 8; ++i_4) \| for (int i_6 = 0; ` |
| wait.mbarrier | 0.03 | 0.7% | 0.5 | 0.06 | `mbarrier[2].wait(1);` |

### l2_prefetch_kernel grid (2, 1, 1) block (128, 1, 1)

289 launches, 8 warps; ranges, warp and last warp from the 2 recording CTAs' 8 warps; span 72.29 us, warp 71.44 us, nsys graphs 72.90 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 70.94 | 99.3% | 80.90 | 99.6% |
| unattributed | 0.50 | 0.7% | 0.35 | 0.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 70.94 | 99.3% | 1.0 | 71.84 | `for (int s = 0; s < (nseg2 >> 1); ++s)` |

### gemm_mxfp8_kernel grid (80, 1, 1) block (128, 1, 1)

323 launches, 320 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 60.93 us, warp 26.72 us, nsys graphs 50.43 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 24.79 | 92.8% | 9.66 | 23.1% |
| wait | 0.84 | 3.2% | 22.58 | 54.0% |
| compute | 0.61 | 2.3% | 5.58 | 13.3% |
| load | 0.24 | 0.9% | 3.75 | 9.0% |
| store | 0.23 | 0.9% | 0.26 | 0.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.cp_async#2 | 0.32 | 1.2% | 1.0 | 1.22 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| compute.mma#2 | 0.30 | 1.1% | 1.0 | 0.32 | `{` |
| compute.mma#3 | 0.30 | 1.1% | 1.0 | 0.32 | `{` |
| load.cp_async | 0.24 | 0.9% | 1.0 | 0.29 | `for (int i_1 = 0; i_1 < 2; ++i_1) \| for (int i_1 = 0; i_1 < 4; ++i_1) \| for (int i_1 = 0; ` |
| store.global | 0.23 | 0.9% | 1.0 | 0.32 | `for (int i_10 = 0; i_10 < 16; ++i_10) \| for (int i_10 = 0; i_10 < 8; ++i_10) \| for (int i_` |
| wait.sync#3 | 0.23 | 0.8% | 1.0 | 0.80 | `__syncthreads();` |
| wait.cp_async#3 | 0.17 | 0.6% | 1.0 | 0.67 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>();` |
| wait.sync#4 | 0.13 | 0.5% | 1.0 | 0.45 | `__syncthreads();` |
| compute.mma*118 (est.: sampled iterations x118) | 0.00 | 0.0% | 0.0 | 0.00 | `{` |
| compute.mma*22 (est.: sampled iterations x22) | 0.00 | 0.0% | 0.0 | 0.00 | `{` |
| compute.mma*4 (est.: sampled iterations x4) | 0.00 | 0.0% | 0.0 | 0.00 | `{` |
| load.cp_async#2*118 (est.: sampled iterations x118) | 0.00 | 0.0% | 0.0 | 0.00 | `for (int i_5 = 0; i_5 < 2; ++i_5) \| for (int i_5 = 0; i_5 < 4; ++i_5) \| tl::cp_async_gs_co` |
| load.cp_async#2*22 (est.: sampled iterations x22) | 0.00 | 0.0% | 0.0 | 0.00 | `for (int i_5 = 0; i_5 < 2; ++i_5) \| for (int i_5 = 0; i_5 < 4; ++i_5) \| tl::cp_async_gs_co` |
| load.cp_async#2*4 (est.: sampled iterations x4) | 0.00 | 0.0% | 0.0 | 0.00 | `for (int i_5 = 0; i_5 < 2; ++i_5) \| for (int i_5 = 0; i_5 < 4; ++i_5) \| tl::cp_async_gs_co` |
| wait.cp_async*118 (est.: sampled iterations x118) | 0.00 | 0.0% | 0.0 | 0.00 | `tl::cp_async_wait<2>();` |
| wait.cp_async*22 (est.: sampled iterations x22) | 0.00 | 0.0% | 0.0 | 0.00 | `tl::cp_async_wait<2>();` |
| wait.cp_async*4 (est.: sampled iterations x4) | 0.00 | 0.0% | 0.0 | 0.00 | `tl::cp_async_wait<2>();` |
| wait.sync#2*118 (est.: sampled iterations x118) | 0.00 | 0.0% | 0.0 | 0.00 | `__syncthreads();` |
| wait.sync#2*22 (est.: sampled iterations x22) | 0.00 | 0.0% | 0.0 | 0.00 | `__syncthreads();` |
| wait.sync#2*4 (est.: sampled iterations x4) | 0.00 | 0.0% | 0.0 | 0.00 | `__syncthreads();` |
| wait.sync*118 (est.: sampled iterations x118) | 0.00 | 0.0% | 0.0 | 0.00 | `__syncthreads();` |
| wait.sync*22 (est.: sampled iterations x22) | 0.00 | 0.0% | 0.0 | 0.00 | `__syncthreads();` |
| wait.sync*4 (est.: sampled iterations x4) | 0.00 | 0.0% | 0.0 | 0.00 | `__syncthreads();` |

### moe_grouped_fp8_fp4_gemm_kernel grid (40, 288, 1) block (128, 1, 1)

102 launches, 46080 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 194.69 us, warp 3.58 us, nsys graphs 134.00 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.72 | 47.9% | 0.56 | 84.0% |
| wait | 0.77 | 21.6% | 0.05 | 7.5% |
| load | 0.52 | 14.4% | 0.06 | 8.5% |
| compute | 0.52 | 14.4% | 0.00 | 0.0% |
| store | 0.06 | 1.7% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*6 (est.: sampled iterations x6) | 0.52 | 14.4% | 0.2 | 2.11 | `{` |
| wait.sync#5*6 (est.: sampled iterations x6) | 0.24 | 6.7% | 0.2 | 1.15 | `__syncthreads();` |
| load#2*6 (est.: sampled iterations x6) | 0.22 | 6.0% | 0.2 | 1.15 | `if (ko < 4)` |
| wait.mbarrier*6 (est.: sampled iterations x6) | 0.14 | 4.0% | 0.2 | 0.58 | `loaded[(ko % 3)].wait((ko / 3));` |
| load.global | 0.12 | 3.3% | 1.0 | 0.29 | `int expert = TileExpert[((int64_t)((int)blockIdx.y))];` |
| load.global#3 | 0.11 | 3.2% | 0.2 | 0.48 | `if (((expert < 128) && (0 <= expert))) \| if (((expert < 384) && (0 <= expert)))` |
| wait.cp_async*6 (est.: sampled iterations x6) | 0.11 | 3.2% | 0.2 | 0.77 | `tl::cp_async_wait<2>();` |
| wait.sync#4*6 (est.: sampled iterations x6) | 0.10 | 2.7% | 0.2 | 0.38 | `__syncthreads();` |
| wait.sync#6*6 (est.: sampled iterations x6) | 0.10 | 2.7% | 0.2 | 0.38 | `__syncthreads();` |
| store.global | 0.06 | 1.7% | 0.2 | 0.26 | `for (int i_2 = 0; i_2 < 16; ++i_2) \| for (int i_4 = 0; i_4 < 32; ++i_4)` |
| load | 0.04 | 1.2% | 0.2 | 0.22 | `for (int s = 0; s < 2; ++s)` |
| wait.sync | 0.04 | 1.2% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#2 | 0.03 | 0.9% | 0.2 | 0.16 | `__syncthreads();` |
| load.global#2 | 0.02 | 0.7% | 0.2 | 0.19 | `if (((int)threadIdx.x) < 32) \| if (((int)threadIdx.x) < 64)` |
| wait.sync#3 | 0.01 | 0.3% | 0.2 | 0.05 | `__syncthreads();` |

### moe_grouped_fp8_fp4_gemm_kernel grid (12, 288, 1) block (128, 1, 1)

102 launches, 13824 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 186.29 us, warp 21.44 us, nsys graphs 169.95 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 11.92 | 55.6% | 0.30 | 73.2% |
| wait | 4.25 | 19.8% | 0.05 | 12.5% |
| compute | 3.44 | 16.0% | 0.00 | 0.0% |
| load | 1.77 | 8.3% | 0.06 | 14.3% |
| store | 0.06 | 0.3% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*40 (est.: sampled iterations x40) | 3.44 | 16.0% | 0.2 | 14.08 | `{` |
| load#2*40 (est.: sampled iterations x40) | 1.52 | 7.1% | 0.2 | 8.96 | `if (ko < 38)` |
| wait.sync#5*40 (est.: sampled iterations x40) | 1.28 | 6.0% | 0.2 | 6.40 | `__syncthreads();` |
| wait.mbarrier*40 (est.: sampled iterations x40) | 0.96 | 4.5% | 0.2 | 3.84 | `loaded[(ko % 3)].wait(((ko % 6) / 3));` |
| wait.cp_async*40 (est.: sampled iterations x40) | 0.64 | 3.0% | 0.2 | 2.56 | `tl::cp_async_wait<2>();` |
| wait.sync#4*40 (est.: sampled iterations x40) | 0.64 | 3.0% | 0.2 | 2.56 | `__syncthreads();` |
| wait.sync#6*40 (est.: sampled iterations x40) | 0.64 | 3.0% | 0.2 | 2.56 | `__syncthreads();` |
| load.global#3 | 0.11 | 0.5% | 0.2 | 0.45 | `if (((expert < 128) && (0 <= expert))) \| if (((expert < 384) && (0 <= expert)))` |
| load.global | 0.07 | 0.3% | 1.0 | 0.10 | `int expert = TileExpert[((int64_t)((int)blockIdx.y))];` |
| store.global | 0.06 | 0.3% | 0.2 | 0.26 | `for (int i_2 = 0; i_2 < 16; ++i_2) \| for (int i_4 = 0; i_4 < 32; ++i_4)` |
| wait.sync | 0.05 | 0.2% | 1.0 | 0.06 | `__syncthreads();` |
| load | 0.04 | 0.2% | 0.2 | 0.22 | `for (int s = 0; s < 2; ++s)` |
| wait.sync#2 | 0.03 | 0.1% | 0.2 | 0.16 | `__syncthreads();` |
| load.global#2 | 0.02 | 0.1% | 0.2 | 0.19 | `if (((int)threadIdx.x) < 32) \| if (((int)threadIdx.x) < 64)` |
| wait.sync#3 | 0.01 | 0.1% | 0.2 | 0.06 | `__syncthreads();` |

### mhc_project_streams_kernel grid (40, 3, 1) block (128, 1, 1)

306 launches, 480 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 49.70 us, warp 20.23 us, nsys graphs 34.48 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 7.92 | 39.1% | 4.47 | 33.9% |
| unattributed | 7.46 | 36.9% | 4.62 | 35.1% |
| compute | 2.77 | 13.7% | 2.56 | 19.4% |
| wait | 1.60 | 7.9% | 1.03 | 7.9% |
| store | 0.48 | 2.4% | 0.50 | 3.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2*4 (est.: sampled iterations x4) | 7.71 | 38.1% | 1.0 | 17.34 | `for (int i_1 = 0; i_1 < 6; ++i_1)` |
| compute.mma*4 (est.: sampled iterations x4) | 2.64 | 13.0% | 1.0 | 2.69 | `{` |
| wait.sync*4 (est.: sampled iterations x4) | 1.42 | 7.0% | 1.0 | 5.50 | `__syncthreads();` |
| store.global*4 (est.: sampled iterations x4) | 0.26 | 1.3% | 1.0 | 0.26 | `if (((((int)blockIdx.y) * 16) + (((int)threadIdx.x) >> 3)) < M)` |
| load.global | 0.21 | 1.0% | 1.0 | 0.26 | `for (int k = 0; k < 4; ++k)` |
| wait.sync#2*4 (est.: sampled iterations x4) | 0.18 | 0.9% | 1.0 | 0.26 | `__syncthreads();` |
| store.global#4 | 0.10 | 0.5% | 1.0 | 0.13 | `if (((((int)threadIdx.x) % 8) == 0) && (((((int)blockIdx.y) * 16) + (((int)threadIdx.x) >>` |
| compute.reduce | 0.07 | 0.3% | 1.0 | 0.10 | `red[0] = (red[0] + __shfl_xor_sync((uint)4294967295, red[0], 4, 32));` |
| compute.reduce#2 | 0.07 | 0.3% | 1.0 | 0.10 | `red[0] = (red[0] + __shfl_xor_sync((uint)4294967295, red[0], 4, 32));` |
| store.global#2 | 0.06 | 0.3% | 1.0 | 0.06 | `if (((((int)blockIdx.y) * 16) + (((int)threadIdx.x) >> 3)) < M) \| if (((((int)threadIdx.x)` |
| store.global#3 | 0.06 | 0.3% | 1.0 | 0.06 | `if (((((int)threadIdx.x) % 8) == 0) && (((((int)blockIdx.y) * 16) + (((int)threadIdx.x) >>` |
| compute.mma | 0.00 | 0.0% | 0.0 | 0.00 | `{` |
| load.global#2 | 0.00 | 0.0% | 0.0 | 0.00 | `for (int i_1 = 0; i_1 < 6; ++i_1)` |
| store.global | 0.00 | 0.0% | 0.0 | 0.00 | `for (int n = 0; n < 4; ++n) \| if (((((int)blockIdx.y) * 16) + (((int)threadIdx.x) >> 3)) <` |
| wait.sync | 0.00 | 0.0% | 0.0 | 0.00 | `__syncthreads();` |

### gemm_mxfp8_kernel grid (192, 1, 1) block (128, 1, 1)

153 launches, 768 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 70.85 us, warp 17.93 us, nsys graphs 74.66 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 8.92 | 49.8% | 7.16 | 36.5% |
| unattributed | 4.58 | 25.6% | 7.77 | 39.5% |
| compute | 2.94 | 16.4% | 2.95 | 15.0% |
| load | 1.28 | 7.1% | 1.58 | 8.1% |
| store | 0.20 | 1.1% | 0.19 | 1.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.cp_async*8 (est.: sampled iterations x8) | 3.34 | 18.7% | 1.0 | 12.54 | `tl::cp_async_wait<2>();` |
| wait.sync#2*8 (est.: sampled iterations x8) | 2.91 | 16.2% | 1.0 | 9.47 | `__syncthreads();` |
| compute.mma*8 (est.: sampled iterations x8) | 2.34 | 13.0% | 1.0 | 2.56 | `{` |
| load.cp_async#2*8 (est.: sampled iterations x8) | 1.09 | 6.1% | 1.0 | 1.28 | `for (int i_5 = 0; i_5 < 2; ++i_5) \| for (int i_5 = 0; i_5 < 4; ++i_5) \| tl::cp_async_gs_co` |
| wait.cp_async#3 | 1.06 | 5.9% | 1.0 | 3.01 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>();` |
| wait.sync#4 | 0.52 | 2.9% | 1.0 | 1.86 | `__syncthreads();` |
| wait.sync*8 (est.: sampled iterations x8) | 0.37 | 2.1% | 1.0 | 0.51 | `__syncthreads();` |
| wait.cp_async#2 | 0.36 | 2.0% | 1.0 | 1.57 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| wait.sync#3 | 0.35 | 2.0% | 1.0 | 1.22 | `__syncthreads();` |
| compute.mma#2 | 0.30 | 1.7% | 1.0 | 0.32 | `{` |
| compute.mma#3 | 0.30 | 1.7% | 1.0 | 0.32 | `{` |
| store.global | 0.20 | 1.1% | 1.0 | 0.26 | `for (int i_10 = 0; i_10 < 16; ++i_10) \| for (int i_10 = 0; i_10 < 8; ++i_10) \| for (int i_` |
| load.cp_async | 0.19 | 1.1% | 1.0 | 0.22 | `for (int i_1 = 0; i_1 < 2; ++i_1) \| for (int i_1 = 0; i_1 < 4; ++i_1) \| for (int i_1 = 0; ` |

### gemm_fp8w_bf16_partials_kernel grid (48, 4, 1) block (128, 1, 1)

153 launches, 768 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 80.93 us, warp 50.40 us, nsys graphs 64.83 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 27.77 | 55.1% | 9.74 | 31.9% |
| wait | 12.77 | 25.3% | 11.19 | 36.7% |
| compute | 5.52 | 11.0% | 5.27 | 17.3% |
| load | 4.14 | 8.2% | 4.11 | 13.5% |
| store | 0.19 | 0.4% | 0.18 | 0.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*14 (est.: sampled iterations x14) | 4.90 | 9.7% | 1.0 | 6.27 | `{` |
| wait.sync#2*14 (est.: sampled iterations x14) | 3.39 | 6.7% | 1.0 | 12.99 | `__syncthreads();` |
| wait.cp_async*14 (est.: sampled iterations x14) | 2.72 | 5.4% | 1.0 | 15.68 | `tl::cp_async_wait<3>();` |
| load.cp_async#3*14 (est.: sampled iterations x14) | 1.26 | 2.5% | 1.0 | 1.79 | `for (int i_10 = 0; i_10 < 4; ++i_10) \| for (int i_10 = 0; i_10 < 6; ++i_10) \| for (int i_1` |
| load.global*14 (est.: sampled iterations x14) | 1.06 | 2.1% | 1.0 | 1.79 | `if (((int)threadIdx.x) < 4)` |
| load.cp_async#2*14 (est.: sampled iterations x14) | 1.04 | 2.1% | 1.0 | 1.34 | `for (int i_4 = 0; i_4 < 2; ++i_4) \| for (int i_6 = 0; i_6 < 2; ++i_6)` |
| wait.sync#4*14 (est.: sampled iterations x14) | 0.98 | 1.9% | 1.0 | 1.34 | `__syncthreads();` |
| wait.sync*14 (est.: sampled iterations x14) | 0.98 | 1.9% | 1.0 | 1.34 | `__syncthreads();` |
| wait.sync#5*14 (est.: sampled iterations x14) | 0.95 | 1.9% | 1.0 | 1.34 | `__syncthreads();` |
| wait.sync#3*14 (est.: sampled iterations x14) | 0.92 | 1.8% | 1.0 | 1.34 | `__syncthreads();` |
| wait.cp_async#2*14 (est.: sampled iterations x14) | 0.81 | 1.6% | 1.0 | 0.90 | `tl::cp_async_wait<3>();` |
| wait.cp_async#3 | 0.72 | 1.4% | 1.0 | 2.69 | `tl::cp_async_wait<3>();` |
| load.cp_async | 0.66 | 1.3% | 1.0 | 1.66 | `for (int i_1 = 0; i_1 < 2; ++i_1)` |
| wait.sync#7 | 0.49 | 1.0% | 1.0 | 1.54 | `__syncthreads();` |
| compute.mma#2 | 0.32 | 0.6% | 1.0 | 0.42 | `{` |
| compute.mma#3 | 0.30 | 0.6% | 1.0 | 0.38 | `{` |
| wait.sync#10 | 0.28 | 0.6% | 1.0 | 1.12 | `__syncthreads();` |
| wait.cp_async#5 | 0.21 | 0.4% | 1.0 | 1.31 | `tl::cp_async_wait<1>();` |
| store.global | 0.19 | 0.4% | 1.0 | 0.26 | `for (int i_13 = 0; i_13 < 8; ++i_13) \| for (int i_19 = 0; i_19 < 16; ++i_19) \| for (int i_` |
| load.global#3 | 0.06 | 0.1% | 1.0 | 0.10 | `if (((int)threadIdx.x) < 4)` |
| load.global#2 | 0.06 | 0.1% | 1.0 | 0.10 | `if (((int)threadIdx.x) < 4)` |
| wait.cp_async#6 | 0.06 | 0.1% | 1.0 | 0.06 | `tl::cp_async_wait<0>();` |
| wait.sync#6 | 0.06 | 0.1% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#8 | 0.05 | 0.1% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#11 | 0.05 | 0.1% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#9 | 0.05 | 0.1% | 1.0 | 0.10 | `__syncthreads();` |
| wait.cp_async#4 | 0.05 | 0.1% | 1.0 | 0.06 | `tl::cp_async_wait<2>();` |

### gemm_mxfp8_kernel grid (28, 1, 1) block (128, 1, 1)

153 launches, 112 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 41.60 us, warp 40.99 us, nsys graphs 42.59 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 13.54 | 33.0% | 18.41 | 41.4% |
| unattributed | 12.87 | 31.4% | 11.27 | 25.3% |
| compute | 10.69 | 26.1% | 10.67 | 24.0% |
| load | 3.72 | 9.1% | 3.99 | 9.0% |
| store | 0.18 | 0.4% | 0.16 | 0.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*19 (est.: sampled iterations x19) | 10.18 | 24.8% | 1.0 | 10.34 | `{` |
| wait.cp_async*19 (est.: sampled iterations x19) | 6.73 | 16.4% | 1.0 | 22.50 | `tl::cp_async_wait<1>();` |
| wait.sync#2*19 (est.: sampled iterations x19) | 4.64 | 11.3% | 1.0 | 14.59 | `__syncthreads();` |
| load.cp_async#2*19 (est.: sampled iterations x19) | 3.50 | 8.5% | 1.0 | 3.65 | `for (int i_3 = 0; i_3 < 2; ++i_3) \| for (int i_3 = 0; i_3 < 8; ++i_3)` |
| wait.sync*19 (est.: sampled iterations x19) | 1.22 | 3.0% | 1.0 | 1.82 | `__syncthreads();` |
| wait.cp_async#2 | 0.64 | 1.6% | 1.0 | 1.47 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| compute.mma#2 | 0.50 | 1.2% | 1.0 | 0.51 | `{` |
| wait.sync#3 | 0.32 | 0.8% | 1.0 | 1.02 | `__syncthreads();` |
| load.cp_async | 0.23 | 0.6% | 1.0 | 0.26 | `for (int i_1 = 0; i_1 < 2; ++i_1) \| for (int i_1 = 0; i_1 < 4; ++i_1) \| for (int i_1 = 0; ` |
| store.global | 0.18 | 0.4% | 1.0 | 0.22 | `for (int i_10 = 0; i_10 < 16; ++i_10) \| for (int i_10 = 0; i_10 < 8; ++i_10) \| for (int i_` |

### gemm_mxfp8_kernel grid (24, 1, 1) block (128, 1, 1)

153 launches, 96 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 40.74 us, warp 38.78 us, nsys graphs 39.52 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| compute | 13.14 | 33.9% | 13.18 | 32.8% |
| wait | 11.46 | 29.5% | 17.94 | 44.6% |
| unattributed | 8.66 | 22.3% | 2.07 | 5.2% |
| load | 5.33 | 13.7% | 6.74 | 16.8% |
| store | 0.20 | 0.5% | 0.29 | 0.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*38 (est.: sampled iterations x38) | 12.54 | 32.3% | 1.0 | 13.38 | `{` |
| load.cp_async#2*38 (est.: sampled iterations x38) | 5.02 | 12.9% | 1.0 | 6.08 | `for (int i_5 = 0; i_5 < 2; ++i_5) \| for (int i_5 = 0; i_5 < 4; ++i_5) \| tl::cp_async_gs_co` |
| wait.sync#2*38 (est.: sampled iterations x38) | 4.86 | 12.5% | 1.0 | 10.94 | `__syncthreads();` |
| wait.cp_async*38 (est.: sampled iterations x38) | 3.95 | 10.2% | 1.0 | 12.16 | `tl::cp_async_wait<2>();` |
| wait.sync*38 (est.: sampled iterations x38) | 2.43 | 6.3% | 1.0 | 2.43 | `__syncthreads();` |
| load.cp_async | 0.31 | 0.8% | 1.0 | 0.35 | `for (int i_1 = 0; i_1 < 2; ++i_1) \| for (int i_1 = 0; i_1 < 4; ++i_1) \| for (int i_1 = 0; ` |
| compute.mma#2 | 0.30 | 0.8% | 1.0 | 0.32 | `{` |
| compute.mma#3 | 0.30 | 0.8% | 1.0 | 0.32 | `{` |
| store.global | 0.20 | 0.5% | 1.0 | 0.26 | `for (int i_10 = 0; i_10 < 16; ++i_10) \| for (int i_10 = 0; i_10 < 8; ++i_10) \| for (int i_` |
| wait.cp_async#3 | 0.06 | 0.1% | 1.0 | 0.06 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>();` |
| wait.sync#3 | 0.05 | 0.1% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#4 | 0.05 | 0.1% | 1.0 | 0.06 | `__syncthreads();` |
| wait.cp_async#2 | 0.05 | 0.1% | 1.0 | 0.06 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |

### sparse_mla_kernel grid (1, 48, 1) block (256, 1, 1)

68 launches, 384 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 65.95 us, warp 64.24 us, nsys graphs 51.82 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed, with loops' unsampled iterations | 57.01 | 88.7% | 58.13 | 86.1% |
| load | 4.22 | 6.6% | 5.98 | 8.9% |
| compute | 1.49 | 2.3% | 1.50 | 2.2% |
| wait | 1.37 | 2.1% | 1.35 | 2.0% |
| store | 0.17 | 0.3% | 0.53 | 0.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global (sampled iterations) | 3.84 | 6.0% | 4.0 | 4.27 | `for (int c = 0; c < 4; ++c) \| for (int i = 0; i < 8; ++i) \| for (int i_1 = 0; i_1 < 8; ++i` |
| wait.sync (sampled iterations) | 1.30 | 2.0% | 18.0 | 1.44 | `__syncthreads();` |
| compute.mma (sampled iterations) | 1.21 | 1.9% | 2.0 | 1.25 | `for (int ki = 0; ki < 32; ++ki) \| for (int ki_1 = 0; ki_1 < 2; ++ki_1)` |
| load.cp_async (sampled iterations) | 0.38 | 0.6% | 2.0 | 0.45 | `for (int k_1 = 0; k_1 < 5; ++k_1) \| for (int k_1 = 0; k_1 < 9; ++k_1) \| for (int k_3 = 0; ` |
| compute.reduce (sampled iterations) | 0.27 | 0.4% | 2.0 | 0.32 | `m_g[i_5] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(m_g[i_5], (&((` |
| store.global | 0.17 | 0.3% | 1.0 | 0.22 | `for (int i_10 = 0; i_10 < 32; ++i_10) \| for (int i_11 = 0; i_11 < 32; ++i_11) \| for (int i` |
| wait.cp_async (sampled iterations) | 0.07 | 0.1% | 1.0 | 0.10 | `tl::cp_async_wait<0>();` |

### moe_grouped_fp8_fp4_gemm_kernel grid (40, 120, 1) block (128, 1, 1)

51 launches, 19200 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 86.72 us, warp 3.31 us, nsys graphs 60.37 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.67 | 50.6% | 0.21 | 66.2% |
| wait | 0.64 | 19.3% | 0.05 | 15.1% |
| compute | 0.52 | 15.6% | 0.00 | 0.0% |
| load | 0.40 | 12.1% | 0.06 | 18.7% |
| store | 0.08 | 2.4% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*6 (est.: sampled iterations x6) | 0.52 | 15.6% | 0.2 | 2.11 | `{` |
| load#2*6 (est.: sampled iterations x6) | 0.22 | 6.5% | 0.2 | 1.15 | `if (ko < 4)` |
| wait.sync#5*6 (est.: sampled iterations x6) | 0.17 | 5.1% | 0.2 | 0.77 | `__syncthreads();` |
| wait.cp_async*6 (est.: sampled iterations x6) | 0.10 | 2.9% | 0.2 | 0.38 | `tl::cp_async_wait<2>();` |
| wait.mbarrier*6 (est.: sampled iterations x6) | 0.10 | 2.9% | 0.2 | 0.38 | `loaded[(ko % 3)].wait((ko / 3));` |
| wait.sync#4*6 (est.: sampled iterations x6) | 0.10 | 2.9% | 0.2 | 0.38 | `__syncthreads();` |
| wait.sync#6*6 (est.: sampled iterations x6) | 0.10 | 2.9% | 0.2 | 0.38 | `__syncthreads();` |
| store.global | 0.08 | 2.4% | 0.2 | 0.32 | `for (int i_2 = 0; i_2 < 16; ++i_2) \| for (int i_4 = 0; i_4 < 32; ++i_4)` |
| load.global | 0.07 | 2.1% | 1.0 | 0.10 | `int expert = TileExpert[((int64_t)((int)blockIdx.y))];` |
| load.global#3 | 0.05 | 1.5% | 0.2 | 0.19 | `if (((expert < 128) && (0 <= expert))) \| if (((expert < 384) && (0 <= expert)))` |
| wait.sync | 0.05 | 1.5% | 1.0 | 0.06 | `__syncthreads();` |
| load | 0.04 | 1.3% | 0.2 | 0.22 | `for (int s = 0; s < 2; ++s)` |
| wait.sync#2 | 0.03 | 1.0% | 0.2 | 0.16 | `__syncthreads();` |
| load.global#2 | 0.02 | 0.7% | 0.2 | 0.19 | `if (((int)threadIdx.x) < 32) \| if (((int)threadIdx.x) < 64)` |
| wait.sync#3 | 0.01 | 0.2% | 0.2 | 0.03 | `__syncthreads();` |

### moe_grouped_fp8_fp4_gemm_kernel grid (12, 120, 1) block (128, 1, 1)

51 launches, 5760 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 81.47 us, warp 12.41 us, nsys graphs 66.13 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 4.08 | 32.9% | 0.05 | 15.6% |
| compute | 3.44 | 27.7% | 0.00 | 0.0% |
| unattributed | 2.93 | 23.6% | 0.20 | 65.1% |
| load | 1.88 | 15.1% | 0.06 | 19.2% |
| store | 0.08 | 0.6% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*40 (est.: sampled iterations x40) | 3.44 | 27.7% | 0.2 | 14.08 | `{` |
| load#2*40 (est.: sampled iterations x40) | 1.52 | 12.3% | 0.2 | 8.96 | `if (ko < 38)` |
| wait.sync#5*40 (est.: sampled iterations x40) | 1.20 | 9.7% | 0.2 | 6.40 | `__syncthreads();` |
| wait.mbarrier*40 (est.: sampled iterations x40) | 0.96 | 7.7% | 0.2 | 3.84 | `loaded[(ko % 3)].wait(((ko % 6) / 3));` |
| wait.sync#4*40 (est.: sampled iterations x40) | 0.64 | 5.2% | 0.2 | 2.56 | `__syncthreads();` |
| wait.sync#6*40 (est.: sampled iterations x40) | 0.64 | 5.2% | 0.2 | 2.56 | `__syncthreads();` |
| wait.cp_async*40 (est.: sampled iterations x40) | 0.56 | 4.5% | 0.2 | 2.56 | `tl::cp_async_wait<2>();` |
| load.global#3 | 0.22 | 1.7% | 0.2 | 0.86 | `if (((expert < 128) && (0 <= expert))) \| if (((expert < 384) && (0 <= expert)))` |
| store.global | 0.08 | 0.6% | 0.2 | 0.32 | `for (int i_2 = 0; i_2 < 16; ++i_2) \| for (int i_4 = 0; i_4 < 32; ++i_4)` |
| load.global | 0.07 | 0.6% | 1.0 | 0.10 | `int expert = TileExpert[((int64_t)((int)blockIdx.y))];` |
| load | 0.05 | 0.4% | 0.2 | 0.26 | `for (int s = 0; s < 2; ++s)` |
| wait.sync | 0.04 | 0.4% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#2 | 0.03 | 0.3% | 0.2 | 0.16 | `__syncthreads();` |
| load.global#2 | 0.02 | 0.2% | 0.2 | 0.19 | `if (((int)threadIdx.x) < 32) \| if (((int)threadIdx.x) < 64)` |
| wait.sync#3 | 0.01 | 0.1% | 0.2 | 0.03 | `__syncthreads();` |

### partial_argmax_kernel grid (8, 64, 1) block (128, 1, 1)

85 launches, 2048 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 31.62 us, warp 23.49 us, nsys graphs 26.88 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 10.87 | 46.3% | 8.13 | 27.2% |
| load | 8.53 | 36.3% | 20.77 | 69.5% |
| compute | 2.51 | 10.7% | 0.33 | 1.1% |
| wait | 1.14 | 4.9% | 0.53 | 1.8% |
| store | 0.44 | 1.9% | 0.13 | 0.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 8.53 | 36.3% | 1.0 | 21.47 | `for (int j = 0; j < 16; ++j) \| for (int run = 0; run < 2; ++run)` |
| compute.reduce | 2.51 | 10.7% | 1.0 | 5.66 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| wait.sync | 1.14 | 4.9% | 1.0 | 3.39 | `__syncthreads();` |
| store.global | 0.44 | 1.9% | 1.0 | 1.79 | `if (((int)threadIdx.x) == 0)` |

### gemm_bf16_splitk_kernel grid (16, 5, 1) block (256, 1, 1)

34 launches, 640 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 73.79 us, warp 67.84 us, nsys graphs 76.03 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 32.94 | 48.5% | 60.86 | 84.5% |
| unattributed | 27.77 | 40.9% | 0.00 | 0.0% |
| compute | 4.12 | 6.1% | 4.65 | 6.5% |
| load | 2.78 | 4.1% | 10.33 | 14.3% |
| store | 0.23 | 0.3% | 0.93 | 1.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.cp_async*13 (est.: sampled iterations x13) | 15.66 | 23.1% | 0.5 | 35.36 | `tl::cp_async_wait<2>();` |
| wait.sync*13 (est.: sampled iterations x13) | 6.90 | 10.2% | 0.5 | 19.34 | `__syncthreads();` |
| wait.cp_async*14 (est.: sampled iterations x14) | 2.11 | 3.1% | 0.5 | 11.42 | `tl::cp_async_wait<1>();` |
| compute.mma*13 (est.: sampled iterations x13) | 2.02 | 3.0% | 0.5 | 2.08 | `{` |
| wait.sync*14 (est.: sampled iterations x14) | 1.91 | 2.8% | 0.5 | 7.84 | `__syncthreads();` |
| compute.mma*14 (est.: sampled iterations x14) | 1.48 | 2.2% | 0.5 | 1.57 | `{` |
| load.cp_async | 1.04 | 1.5% | 1.0 | 3.90 | `for (int i_1 = 0; i_1 < 2; ++i_1) \| tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)A_sha` |
| wait.sync#4 | 1.03 | 1.5% | 1.0 | 3.25 | `__syncthreads();` |
| wait.sync#5 | 0.90 | 1.3% | 1.0 | 3.02 | `__syncthreads();` |
| wait.sync#3 | 0.86 | 1.3% | 1.0 | 2.80 | `__syncthreads();` |
| wait.cp_async#2 | 0.82 | 1.2% | 1.0 | 3.18 | `tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>(); \| tl::cp_async_wait<3>();` |
| wait.cp_async#3 | 0.78 | 1.2% | 1.0 | 3.52 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| load.cp_async#2*13 (est.: sampled iterations x13) | 0.75 | 1.1% | 0.5 | 1.04 | `for (int i_10 = 0; i_10 < 2; ++i_10) \| for (int i_8 = 0; i_8 < 2; ++i_8) \| tl::cp_async_gs` |
| load.cp_async#2*14 (est.: sampled iterations x14) | 0.68 | 1.0% | 0.5 | 0.90 | `for (int i_6 = 0; i_6 < 2; ++i_6) \| tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)A_sha` |
| wait.cp_async#4 | 0.54 | 0.8% | 0.5 | 2.34 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>();` |
| wait.sync#2*14 (est.: sampled iterations x14) | 0.49 | 0.7% | 0.5 | 0.67 | `__syncthreads();` |
| wait.sync#2*13 (est.: sampled iterations x13) | 0.42 | 0.6% | 0.5 | 0.62 | `__syncthreads();` |
| wait.sync#6 | 0.33 | 0.5% | 1.0 | 0.74 | `__syncthreads();` |
| compute.mma#2 | 0.25 | 0.4% | 1.0 | 0.29 | `{` |
| compute.mma#3 | 0.25 | 0.4% | 1.0 | 0.29 | `{` |
| wait.sync#7 | 0.19 | 0.3% | 0.5 | 0.22 | `__syncthreads();` |
| load.global#2*8 (est.: sampled iterations x8) | 0.16 | 0.2% | 0.1 | 0.64 | `acc = P[(((((((int64_t)((int)blockIdx.z)) * (int64_t)16384) + (((int64_t)i_17) * (int64_t)` |
| load.global | 0.15 | 0.2% | 1.0 | 1.26 | `if (((int)threadIdx.x) == 0)` |
| compute.mma#4 | 0.13 | 0.2% | 0.5 | 0.14 | `{` |
| store.global | 0.13 | 0.2% | 1.0 | 0.22 | `for (int i_10 = 0; i_10 < 4; ++i_10) \| for (int i_16 = 0; i_16 < 4; ++i_16) \| for (int i_1` |
| store.global#3 | 0.07 | 0.1% | 1.0 | 0.10 | `if (((int*)last)[0] == 1)` |
| store.global#2*8 (est.: sampled iterations x8) | 0.04 | 0.1% | 0.1 | 0.26 | `C0[(((((((int64_t)((int)blockIdx.z)) * (int64_t)16384) + (((int64_t)i_18) * (int64_t)2048)` |
| load.global#2*16 (est.: sampled iterations x16) | 0.00 | 0.0% | 0.0 | 0.00 | `acc = P[(((((((int64_t)((int)blockIdx.z)) * (int64_t)65536) + (((int64_t)i_26) * (int64_t)` |
| store.global#2*16 (est.: sampled iterations x16) | 0.00 | 0.0% | 0.0 | 0.00 | `if (((int)blockIdx.x) < 8)` |

### gemm_bf16_partials_kernel grid (6, 8, 1) block (256, 1, 1)

102 launches, 384 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 20.50 us, warp 8.71 us, nsys graphs 18.66 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 3.61 | 41.4% | 4.73 | 44.6% |
| wait | 3.09 | 35.5% | 2.61 | 24.6% |
| compute | 1.50 | 17.2% | 3.04 | 28.6% |
| load | 0.39 | 4.5% | 0.00 | 0.0% |
| store | 0.12 | 1.4% | 0.24 | 2.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.mbarrier*10 (est.: sampled iterations x10) | 1.54 | 17.7% | 0.5 | 3.52 | `mbarrier[((kb % 3) + 3)].wait((((kb % 6) / 3) ^ 1));` |
| compute.mma*10 (est.: sampled iterations x10) | 1.50 | 17.2% | 0.5 | 3.20 | `{` |
| wait.mbarrier#2*10 (est.: sampled iterations x10) | 1.44 | 16.5% | 0.5 | 5.76 | `mbarrier[(kb_1 % 3)].wait(((kb_1 % 6) / 3));` |
| load.tma*10 (est.: sampled iterations x10) | 0.39 | 4.5% | 0.5 | 1.28 | `if (tl::tl_shuffle_elect<128>())` |
| store.global | 0.12 | 1.4% | 0.5 | 0.29 | `for (int i_1 = 0; i_1 < 4; ++i_1) \| for (int i_2 = 0; i_2 < 8; ++i_2) \| for (int i_4 = 0; ` |
| wait.sync | 0.11 | 1.2% | 1.0 | 0.14 | `__syncthreads();` |

### mhc_finalize_kernel grid (48, 1, 1) block (128, 1, 1)

204 launches, 192 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 6.83 us, warp 5.83 us, nsys graphs 9.20 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 2.23 | 38.2% | 3.30 | 43.4% |
| store | 1.00 | 17.2% | 1.27 | 16.7% |
| compute | 1.00 | 17.1% | 0.66 | 8.7% |
| load | 0.97 | 16.7% | 1.42 | 18.7% |
| wait | 0.63 | 10.8% | 0.95 | 12.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync | 0.63 | 10.8% | 1.0 | 0.86 | `__syncthreads();` |
| compute.reduce | 0.58 | 9.9% | 0.2 | 2.30 | `row_max[0] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<32>>::run(row_max[0]);` |
| store.global#4*7 (est.: sampled iterations x7) | 0.56 | 9.6% | 0.8 | 0.90 | `if ((((int)threadIdx.x) % 4) == 0)` |
| compute.reduce#2*7 (est.: sampled iterations x7) | 0.42 | 7.2% | 0.8 | 0.67 | `amax[0] = max(amax[0], __shfl_xor_sync((uint)4294967295, amax[0], 1, 32));` |
| load.global#4*7 (est.: sampled iterations x7) | 0.36 | 6.2% | 0.8 | 0.67 | `*(uint4*)(yv + 0) = *(uint4*)(C + ((((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((i` |
| store.global#3*7 (est.: sampled iterations x7) | 0.32 | 5.5% | 0.8 | 0.45 | `*(uint4*)(y + ((((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((int64_t)step) * (int6` |
| load.global | 0.32 | 5.4% | 1.0 | 0.93 | `if (((int)threadIdx.x) < 32)` |
| load.global#3 | 0.25 | 4.2% | 0.8 | 0.35 | `for (int i_split_2 = 0; i_split_2 < 40; ++i_split_2)` |
| store.global | 0.11 | 1.8% | 0.2 | 0.45 | `if (((int)threadIdx.x) < 4)` |
| load.global#2 | 0.05 | 0.8% | 0.2 | 0.19 | `cm[0] = ((mixes_shared[((((int)threadIdx.x) & 15) + 8)] * scale[2]) + base[((((int)threadI` |
| store.global#2 | 0.01 | 0.2% | 0.2 | 0.06 | `if ((((int)threadIdx.x) >> 4) == 0)` |

### sparse_mla_kernel grid (2, 1, 40) block (256, 1, 1)

51 launches, 640 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 27.17 us, warp 15.18 us, nsys graphs 23.84 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| compute | 6.99 | 46.0% | 6.94 | 46.2% |
| wait | 3.01 | 19.8% | 2.85 | 19.0% |
| unattributed, with loops' unsampled iterations | 2.63 | 17.3% | 2.55 | 17.0% |
| load | 1.86 | 12.3% | 1.84 | 12.3% |
| store | 0.69 | 4.5% | 0.83 | 5.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*4 (est.: sampled iterations x4) | 2.73 | 18.0% | 1.0 | 2.82 | `for (int ki = 0; ki < 32; ++ki) \| {` |
| compute.mma#2*4 (est.: sampled iterations x4) | 2.21 | 14.5% | 1.0 | 2.30 | `for (int ki_1 = 0; ki_1 < 2; ++ki_1) \| if (((((int)blockIdx.x) * 2) + (c_1 >> 1)) < 3)` |
| compute.reduce#2*8 (est.: sampled iterations x8) | 1.02 | 6.7% | 1.0 | 1.02 | `row_sum[i_7] = tl::AllReduce<tl::SumOp, 128, 32, 0, tl::NamedBarrier<128>>::run(row_sum[i_` |
| compute.reduce*8 (est.: sampled iterations x8) | 1.02 | 6.7% | 1.0 | 1.02 | `m_g[i_4] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(m_g[i_4], (&((` |
| load.global#2 | 0.90 | 5.9% | 1.0 | 1.02 | `for (int i = 0; i < 8; ++i)` |
| store.global | 0.69 | 4.5% | 1.0 | 0.86 | `for (int i_10 = 0; i_10 < 32; ++i_10) \| for (int i_11 = 0; i_11 < 32; ++i_11) \| for (int i` |
| load.cp_async#2*4 (est.: sampled iterations x4) | 0.65 | 4.3% | 1.0 | 0.77 | `if (((((int)blockIdx.x) * 2) + (c_1 >> 1)) < 3) \| if ((c_1 < 3) & (c_1 < 3)) \| if (c < 3)` |
| wait.sync#10*8 (est.: sampled iterations x8) | 0.38 | 2.5% | 1.0 | 0.51 | `__syncthreads();` |
| wait.sync#8*8 (est.: sampled iterations x8) | 0.38 | 2.5% | 1.0 | 0.51 | `__syncthreads();` |
| wait.sync#7*4 (est.: sampled iterations x4) | 0.31 | 2.0% | 1.0 | 0.38 | `__syncthreads();` |
| wait.sync#5*4 (est.: sampled iterations x4) | 0.30 | 2.0% | 1.0 | 0.38 | `__syncthreads();` |
| wait.cp_async*4 (est.: sampled iterations x4) | 0.28 | 1.8% | 1.0 | 0.38 | `tl::cp_async_wait<0>();` |
| wait.sync#4*4 (est.: sampled iterations x4) | 0.28 | 1.8% | 1.0 | 0.38 | `__syncthreads();` |
| wait.sync#9*4 (est.: sampled iterations x4) | 0.26 | 1.7% | 1.0 | 0.38 | `__syncthreads();` |
| wait.sync#11*4 (est.: sampled iterations x4) | 0.26 | 1.7% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#6*4 (est.: sampled iterations x4) | 0.25 | 1.7% | 1.0 | 0.38 | `__syncthreads();` |
| load.cp_async (sampled iterations) | 0.21 | 1.4% | 1.0 | 0.26 | `for (int k_1 = 0; k_1 < 5; ++k_1) \| for (int k_1 = 0; k_1 < 9; ++k_1) \| for (int k_3 = 0; ` |
| wait.sync (sampled iterations) | 0.20 | 1.3% | 1.0 | 0.26 | `__syncthreads();` |
| load.global (sampled iterations) | 0.10 | 0.6% | 1.0 | 0.29 | `for (int c = 0; c < 4; ++c) \| for (int i = 0; i < 8; ++i) \| for (int i_1 = 0; i_1 < 8; ++i` |
| wait.sync#3 | 0.07 | 0.5% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#2 | 0.05 | 0.3% | 1.0 | 0.06 | `__syncthreads();` |

### moe_swiglu_cast_kernel_kernel grid (16, 288, 1) block (96, 1, 1)

102 launches, 13824 warps; ranges, warp and last warp from the 4 recording CTAs' 12 warps; span 10.34 us, warp 1.58 us, nsys graphs 5.39 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.12 | 70.6% | 0.26 | 36.9% |
| load | 0.38 | 24.3% | 0.44 | 63.1% |
| compute | 0.04 | 2.7% | 0.00 | 0.0% |
| store | 0.04 | 2.4% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.38 | 24.3% | 1.0 | 0.72 | `for (int r = 0; r < 2; ++r)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.04 | 2.7% | 0.2 | 0.19 | `if (0 <= orders[r_2])` |
| store.global*2 (est.: sampled iterations x2) | 0.04 | 2.4% | 0.2 | 0.19 | `if (0 <= orders[r_2])` |

### sparse_mla_kernel grid (1, 1, 48) block (256, 1, 1)

34 launches, 384 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 31.73 us, warp 21.66 us, nsys graphs 31.22 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| compute | 6.93 | 32.0% | 6.92 | 25.4% |
| load | 5.34 | 24.7% | 6.64 | 24.3% |
| unattributed, with loops' unsampled iterations | 5.30 | 24.5% | 9.81 | 35.9% |
| wait | 3.06 | 14.1% | 3.03 | 11.1% |
| store | 1.04 | 4.8% | 0.90 | 3.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 | 4.37 | 20.2% | 1.0 | 8.74 | `for (int i = 0; i < 8; ++i)` |
| compute.mma*4 (est.: sampled iterations x4) | 2.72 | 12.6% | 1.0 | 2.82 | `for (int ki = 0; ki < 32; ++ki) \| {` |
| compute.mma#2*4 (est.: sampled iterations x4) | 2.16 | 10.0% | 1.0 | 2.30 | `for (int ki_1 = 0; ki_1 < 2; ++ki_1) \| if (((((int)blockIdx.x) * 2) + (c_1 >> 1)) < 3)` |
| store.global | 1.04 | 4.8% | 1.0 | 1.65 | `for (int i_10 = 0; i_10 < 32; ++i_10) \| for (int i_11 = 0; i_11 < 32; ++i_11) \| for (int i` |
| compute.reduce#2*8 (est.: sampled iterations x8) | 1.02 | 4.7% | 1.0 | 1.02 | `row_sum[i_7] = tl::AllReduce<tl::SumOp, 128, 32, 0, tl::NamedBarrier<128>>::run(row_sum[i_` |
| compute.reduce*8 (est.: sampled iterations x8) | 1.02 | 4.7% | 1.0 | 1.02 | `m_g[i_4] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(m_g[i_4], (&((` |
| load.cp_async#2*4 (est.: sampled iterations x4) | 0.63 | 2.9% | 1.0 | 0.77 | `if (((((int)blockIdx.x) * 2) + (c_1 >> 1)) < 3) \| if ((c_1 < 3) & (c_1 < 3)) \| if (c < 3)` |
| wait.sync#10*8 (est.: sampled iterations x8) | 0.38 | 1.8% | 1.0 | 0.51 | `__syncthreads();` |
| wait.sync#8*8 (est.: sampled iterations x8) | 0.38 | 1.7% | 1.0 | 0.51 | `__syncthreads();` |
| wait.sync#5*4 (est.: sampled iterations x4) | 0.32 | 1.5% | 1.0 | 0.38 | `__syncthreads();` |
| wait.sync#7*4 (est.: sampled iterations x4) | 0.29 | 1.4% | 1.0 | 0.38 | `__syncthreads();` |
| wait.sync#4*4 (est.: sampled iterations x4) | 0.28 | 1.3% | 1.0 | 0.38 | `__syncthreads();` |
| wait.cp_async*4 (est.: sampled iterations x4) | 0.28 | 1.3% | 1.0 | 0.38 | `tl::cp_async_wait<0>();` |
| wait.sync#9*4 (est.: sampled iterations x4) | 0.26 | 1.2% | 1.0 | 0.38 | `__syncthreads();` |
| wait.sync#11*4 (est.: sampled iterations x4) | 0.26 | 1.2% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#6*4 (est.: sampled iterations x4) | 0.25 | 1.2% | 1.0 | 0.26 | `__syncthreads();` |
| load.cp_async (sampled iterations) | 0.25 | 1.1% | 1.0 | 0.29 | `for (int k_1 = 0; k_1 < 5; ++k_1) \| for (int k_1 = 0; k_1 < 9; ++k_1) \| for (int k_3 = 0; ` |
| wait.sync (sampled iterations) | 0.22 | 1.0% | 1.0 | 0.26 | `__syncthreads();` |
| load.global (sampled iterations) | 0.09 | 0.4% | 1.0 | 0.35 | `for (int c = 0; c < 4; ++c) \| for (int i = 0; i < 8; ++i) \| for (int i_1 = 0; i_1 < 8; ++i` |
| wait.sync#3 | 0.09 | 0.4% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#2 | 0.05 | 0.2% | 1.0 | 0.06 | `__syncthreads();` |

### mhc_post_kernel grid (240, 1, 1) block (128, 1, 1)

68 launches, 960 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 12.40 us, warp 7.35 us, nsys graphs 10.54 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 7.12 | 96.9% | 9.31 | 86.2% |
| load | 0.17 | 2.3% | 0.87 | 8.0% |
| store | 0.06 | 0.8% | 0.63 | 5.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.17 | 2.3% | 1.0 | 0.22 | `*(float4*)(pm + 0) = *(float4*)(post + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int` |
| store.global | 0.06 | 0.8% | 1.0 | 0.13 | `for (int n_1 = 0; n_1 < 4; ++n_1)` |

### gemm_mxfp8_kernel grid (64, 1, 1) block (64, 1, 1)

34 launches, 128 warps; ranges, warp and last warp from the 4 recording CTAs' 8 warps; span 25.41 us, warp 23.73 us, nsys graphs 26.27 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 15.34 | 64.6% | 15.94 | 64.7% |
| compute | 5.10 | 21.5% | 5.12 | 20.8% |
| unattributed | 1.86 | 7.8% | 2.06 | 8.4% |
| load | 1.26 | 5.3% | 1.28 | 5.2% |
| store | 0.18 | 0.8% | 0.22 | 0.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.cp_async*9 (est.: sampled iterations x9) | 11.95 | 50.4% | 1.0 | 21.02 | `tl::cp_async_wait<1>();` |
| compute.mma*9 (est.: sampled iterations x9) | 4.61 | 19.4% | 1.0 | 4.61 | `{` |
| wait.sync#2*9 (est.: sampled iterations x9) | 1.91 | 8.0% | 1.0 | 6.48 | `__syncthreads();` |
| load.cp_async#2*9 (est.: sampled iterations x9) | 1.08 | 4.6% | 1.0 | 1.15 | `for (int i_3 = 0; i_3 < 8; ++i_3)` |
| wait.cp_async#2 | 0.82 | 3.5% | 1.0 | 2.02 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| wait.sync*9 (est.: sampled iterations x9) | 0.54 | 2.3% | 1.0 | 0.58 | `__syncthreads();` |
| compute.mma#2 | 0.49 | 2.1% | 1.0 | 0.51 | `{` |
| store.global | 0.18 | 0.8% | 1.0 | 0.22 | `for (int i_10 = 0; i_10 < 16; ++i_10) \| for (int i_10 = 0; i_10 < 8; ++i_10) \| for (int i_` |
| load.cp_async | 0.18 | 0.7% | 1.0 | 0.22 | `for (int i_1 = 0; i_1 < 2; ++i_1) \| for (int i_1 = 0; i_1 < 4; ++i_1) \| for (int i_1 = 0; ` |
| wait.sync#3 | 0.12 | 0.5% | 1.0 | 0.51 | `__syncthreads();` |

### cache_writer_kernel grid (48, 1, 1) block (128, 1, 1)

139 launches, 192 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 4.38 us, warp 3.66 us, nsys graphs 6.78 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 2.14 | 58.6% | 3.41 | 68.4% |
| unattributed | 1.14 | 31.3% | 0.89 | 17.8% |
| load | 0.29 | 8.0% | 0.61 | 12.2% |
| compute | 0.07 | 2.0% | 0.08 | 1.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 2.14 | 58.6% | 1.0 | 2.46 | `if (((int64_t)0 <= slot) & ((slot >> (int64_t)7) < ((int64_t)pages))) \| if (((int64_t)0 <=` |
| load.global | 0.24 | 6.5% | 1.0 | 0.26 | `for (int i = 0; i < 4; ++i)` |
| compute.reduce | 0.07 | 2.0% | 1.0 | 0.10 | `amax = max(amax, __shfl_xor_sync((uint)4294967295, amax, 1, 32));` |
| load.global#2 | 0.06 | 1.6% | 1.0 | 0.06 | `int64_t slot = slots[((int64_t)((int)blockIdx.x))];` |

### gemm_bf16_partials_kernel grid (2, 20, 1) block (256, 1, 1)

51 launches, 320 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 14.82 us, warp 7.03 us, nsys graphs 11.84 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 5.36 | 76.2% | 4.41 | 55.8% |
| wait | 0.90 | 12.8% | 2.19 | 27.6% |
| compute | 0.56 | 7.9% | 1.12 | 14.2% |
| load | 0.13 | 1.9% | 0.00 | 0.0% |
| store | 0.09 | 1.3% | 0.19 | 2.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.mbarrier#2*4 (est.: sampled iterations x4) | 0.68 | 9.6% | 0.5 | 2.94 | `mbarrier[(kb_1 % 3)].wait((kb_1 / 3));` |
| compute.mma*4 (est.: sampled iterations x4) | 0.56 | 7.9% | 0.5 | 1.15 | `{` |
| load.tma*4 (est.: sampled iterations x4) | 0.13 | 1.9% | 0.5 | 0.38 | `if (tl::tl_shuffle_elect<128>())` |
| wait.mbarrier*4 (est.: sampled iterations x4) | 0.12 | 1.7% | 0.5 | 0.26 | `mbarrier[((kb % 3) + 3)].wait(((kb / 3) ^ 1));` |
| wait.sync | 0.10 | 1.4% | 1.0 | 0.13 | `__syncthreads();` |
| store.global | 0.09 | 1.3% | 0.5 | 0.22 | `for (int i_1 = 0; i_1 < 4; ++i_1) \| for (int i_2 = 0; i_2 < 8; ++i_2) \| for (int i_4 = 0; ` |

### mhc_finalize_kernel grid (40, 1, 1) block (128, 1, 1)

102 launches, 160 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 6.62 us, warp 5.91 us, nsys graphs 5.98 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 2.25 | 38.0% | 3.45 | 48.9% |
| load | 1.02 | 17.2% | 1.01 | 14.3% |
| store | 1.00 | 17.0% | 1.16 | 16.4% |
| compute | 1.00 | 16.8% | 0.57 | 8.1% |
| wait | 0.65 | 11.0% | 0.86 | 12.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync | 0.65 | 11.0% | 1.0 | 0.86 | `__syncthreads();` |
| compute.reduce | 0.58 | 9.7% | 0.2 | 2.30 | `row_max[0] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<32>>::run(row_max[0]);` |
| store.global#4*7 (est.: sampled iterations x7) | 0.57 | 9.7% | 0.8 | 0.90 | `if ((((int)threadIdx.x) % 4) == 0)` |
| compute.reduce#2*7 (est.: sampled iterations x7) | 0.42 | 7.1% | 0.8 | 0.67 | `amax[0] = max(amax[0], __shfl_xor_sync((uint)4294967295, amax[0], 1, 32));` |
| load.global#4*7 (est.: sampled iterations x7) | 0.36 | 6.2% | 0.8 | 0.67 | `*(uint4*)(yv + 0) = *(uint4*)(C + ((((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((i` |
| store.global#3*7 (est.: sampled iterations x7) | 0.32 | 5.4% | 0.8 | 0.45 | `*(uint4*)(y + ((((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((int64_t)step) * (int6` |
| load.global | 0.31 | 5.3% | 1.0 | 0.93 | `if (((int)threadIdx.x) < 32)` |
| load.global#3 | 0.24 | 4.1% | 0.8 | 0.35 | `for (int i_split_2 = 0; i_split_2 < 40; ++i_split_2)` |
| load.global#2 | 0.10 | 1.6% | 0.2 | 0.38 | `cm[0] = ((mixes_shared[((((int)threadIdx.x) & 15) + 8)] * scale[2]) + base[((((int)threadI` |
| store.global | 0.09 | 1.6% | 0.2 | 0.42 | `if (((int)threadIdx.x) < 4)` |
| store.global#2 | 0.01 | 0.2% | 0.2 | 0.06 | `if ((((int)threadIdx.x) >> 4) == 0)` |

### moe_dispatch_kernel grid (1, 1, 1) block (512, 1, 1)

153 launches, 16 warps; ranges, warp and last warp from the 1 recording CTAs' 16 warps; span 4.32 us, warp 4.16 us, nsys graphs 2.72 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 2.16 | 52.0% | 1.78 | 41.6% |
| unattributed, with loops' unsampled iterations | 1.40 | 33.6% | 1.45 | 34.0% |
| store | 0.52 | 12.5% | 0.95 | 22.3% |
| load | 0.08 | 2.0% | 0.09 | 2.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync#7*9 (est.: sampled iterations x9) | 0.65 | 15.6% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#6*9 (est.: sampled iterations x9) | 0.63 | 15.2% | 1.0 | 0.86 | `__syncthreads();` |
| store.global (sampled iterations) | 0.41 | 9.9% | 1.0 | 0.70 | `ChunkCount[((((int64_t)c) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = count; \| Chu` |
| wait.sync#8 | 0.36 | 8.8% | 1.0 | 0.54 | `__syncthreads();` |
| wait.sync#3 | 0.14 | 3.3% | 1.0 | 0.35 | `__syncthreads();` |
| wait.sync#2 | 0.13 | 3.1% | 1.0 | 0.19 | `__syncthreads();` |
| store.global#2 | 0.11 | 2.6% | 1.0 | 0.26 | `RoutePos[((((int64_t)((int)blockIdx.x)) * (int64_t)512) + ((int64_t)((int)threadIdx.x)))] ` |
| load.global (sampled iterations) | 0.08 | 2.0% | 1.0 | 0.10 | `expert = ((int)TopkIdx[((((int64_t)((int)blockIdx.x)) * (int64_t)512) + ((int64_t)((int)th` |
| wait.sync | 0.07 | 1.6% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#4 | 0.06 | 1.5% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#9 | 0.06 | 1.5% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#5 | 0.06 | 1.4% | 1.0 | 0.06 | `__syncthreads();` |

### gemm_mxfp8_partials_kernel grid (16, 5, 1) block (128, 1, 1)

51 launches, 320 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 11.87 us, warp 11.28 us, nsys graphs 12.48 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 5.95 | 52.7% | 5.58 | 44.8% |
| wait | 4.21 | 37.3% | 3.85 | 30.9% |
| compute | 0.61 | 5.4% | 0.71 | 5.7% |
| load | 0.46 | 4.1% | 2.28 | 18.3% |
| store | 0.05 | 0.5% | 0.05 | 0.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.cp_async#3 | 1.57 | 13.9% | 1.0 | 3.42 | `tl::cp_async_wait<0>();` |
| wait.cp_async#2 | 0.90 | 8.0% | 1.0 | 1.92 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>();` |
| wait.sync#2*2 (est.: sampled iterations x2) | 0.50 | 4.4% | 1.0 | 1.47 | `__syncthreads();` |
| wait.sync#3 | 0.49 | 4.3% | 1.0 | 1.47 | `__syncthreads();` |
| wait.sync#4 | 0.40 | 3.6% | 1.0 | 1.28 | `__syncthreads();` |
| compute.mma*2 (est.: sampled iterations x2) | 0.31 | 2.7% | 1.0 | 0.32 | `{` |
| wait.cp_async*2 (est.: sampled iterations x2) | 0.25 | 2.2% | 1.0 | 1.41 | `tl::cp_async_wait<2>();` |
| load.cp_async | 0.24 | 2.1% | 1.0 | 0.35 | `for (int i = 0; i < 2; ++i) \| for (int i_1 = 0; i_1 < 4; ++i_1)` |
| load.cp_async#2*2 (est.: sampled iterations x2) | 0.22 | 2.0% | 1.0 | 0.26 | `for (int i_4 = 0; i_4 < 2; ++i_4)` |
| compute.mma#3 | 0.15 | 1.4% | 1.0 | 0.19 | `{` |
| compute.mma#2 | 0.15 | 1.3% | 1.0 | 0.16 | `{` |
| wait.sync*2 (est.: sampled iterations x2) | 0.09 | 0.8% | 1.0 | 0.13 | `__syncthreads();` |
| store.global | 0.05 | 0.5% | 1.0 | 0.06 | `for (int i_11 = 0; i_11 < 16; ++i_11) \| for (int i_13 = 0; i_13 < 8; ++i_13) \| for (int i_` |
| compute.mma*6 (est.: sampled iterations x6) | 0.00 | 0.0% | 0.0 | 0.00 | `{` |
| load.cp_async#2*6 (est.: sampled iterations x6) | 0.00 | 0.0% | 0.0 | 0.00 | `for (int i_5 = 0; i_5 < 4; ++i_5)` |
| wait.cp_async*6 (est.: sampled iterations x6) | 0.00 | 0.0% | 0.0 | 0.00 | `tl::cp_async_wait<2>();` |
| wait.sync#2*6 (est.: sampled iterations x6) | 0.00 | 0.0% | 0.0 | 0.00 | `__syncthreads();` |
| wait.sync*6 (est.: sampled iterations x6) | 0.00 | 0.0% | 0.0 | 0.00 | `__syncthreads();` |

### local_stats_kernel grid (16, 16, 1) block (128, 1, 1)

16 launches, 1024 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 38.82 us, warp 34.86 us, nsys graphs 36.00 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 33.47 | 96.0% | 36.19 | 96.4% |
| unattributed | 0.99 | 2.8% | 0.86 | 2.3% |
| compute | 0.20 | 0.6% | 0.19 | 0.5% |
| wait | 0.13 | 0.4% | 0.21 | 0.6% |
| store | 0.07 | 0.2% | 0.09 | 0.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 33.47 | 96.0% | 1.0 | 35.97 | `if (expanded_local_pos[((int64_t)((int)blockIdx.x))] < num_speculative_steps)` |
| compute.reduce | 0.13 | 0.4% | 1.0 | 0.16 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| wait.sync | 0.13 | 0.4% | 1.0 | 0.54 | `__syncthreads();` |
| store.global | 0.07 | 0.2% | 1.0 | 0.10 | `if (((int)threadIdx.x) == 0)` |
| compute.reduce#2 | 0.06 | 0.2% | 1.0 | 0.06 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 2, 32);` |

### metadata_decode_kernel grid (48, 2, 1) block (128, 1, 1)

68 launches, 384 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 2.34 us, warp 1.32 us, nsys graphs 3.02 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.67 | 50.4% | 5.59 | 70.8% |
| unattributed | 0.57 | 43.4% | 1.94 | 24.6% |
| store | 0.08 | 6.2% | 0.37 | 4.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.42 | 31.8% | 0.5 | 1.34 | `if ((((offset + ((int)blockIdx.x)) < src_rows) && (0 <= (offset + ((int)blockIdx.x)))))` |
| load.global#3 | 0.24 | 18.0% | 0.5 | 0.53 | `if ((((offset + ((int)blockIdx.x)) < main_src_rows) && (0 <= (offset + ((int)blockIdx.x)))` |
| store.global#4 | 0.04 | 3.2% | 0.5 | 0.22 | `if (0 < live)` |
| store.global | 0.03 | 2.3% | 0.5 | 0.06 | `swa[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = con` |
| load.global#2 | 0.01 | 0.6% | 0.1 | 0.10 | `if (((0 <= req) & ((int64_t)0 <= pos)))` |
| store.global#2 | 0.01 | 0.5% | 0.1 | 0.06 | `lengths[((int64_t)((int)blockIdx.x))] = ((int)max(info[1], (int64_t)0));` |
| store.global#3 | 0.00 | 0.3% | 0.1 | 0.03 | `top_lengths[((int64_t)((int)blockIdx.x))] = condval_6; \| top_lengths[((int64_t)((int)block` |

### router_partials_kernel grid (24, 1, 1) block (64, 1, 1)

102 launches, 48 warps; ranges, warp and last warp from the 4 recording CTAs' 8 warps; span 4.93 us, warp 3.09 us, nsys graphs 4.54 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.26 | 40.8% | 1.93 | 36.4% |
| load | 1.00 | 32.5% | 1.81 | 34.1% |
| compute | 0.68 | 22.0% | 1.36 | 25.7% |
| store | 0.14 | 4.7% | 0.20 | 3.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.reduce | 0.68 | 22.0% | 0.5 | 1.38 | `for (int k = 0; k < 3; ++k) \| for (int k = 0; k < 6; ++k)` |
| load.global | 0.45 | 14.5% | 1.0 | 0.48 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < mask_len))` |
| load.global#4 | 0.43 | 13.9% | 0.5 | 0.93 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < image_len)) \| if ((((((int)bl` |
| load.global#2 | 0.06 | 2.1% | 1.0 | 0.06 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < mask_len))` |
| load.global#3 | 0.06 | 2.1% | 1.0 | 0.06 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < mask_len))` |
| store.global | 0.05 | 1.6% | 1.0 | 0.06 | `if (condval == (bool)0) \| if (condval == (bool)1)` |
| store.global#2 | 0.05 | 1.6% | 1.0 | 0.06 | `if (condval_1 == (bool)0) \| if (condval_1 == (bool)1)` |
| store.global#3 | 0.05 | 1.6% | 0.5 | 0.10 | `if ((((int)threadIdx.x) & 31) < 3) \| if ((((int)threadIdx.x) & 31) < 6)` |

### mhc_collapse_kernel grid (240, 1, 1) block (128, 1, 1)

51 launches, 960 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 10.14 us, warp 4.64 us, nsys graphs 6.94 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 4.04 | 87.2% | 3.78 | 68.1% |
| load | 0.53 | 11.5% | 1.70 | 30.6% |
| store | 0.06 | 1.3% | 0.07 | 1.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.53 | 11.5% | 1.0 | 1.66 | `*(float4*)(mv + 0) = *(float4*)(mix + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int6` |
| store.global | 0.06 | 1.3% | 1.0 | 0.10 | `*(uint4*)(out + ((((int64_t)((int)blockIdx.x)) * (int64_t)1024) + (((int64_t)((int)threadI` |

### metadata_builder_kernel grid (2, 1, 1) block (128, 1, 1)

289 launches, 8 warps; ranges, warp and last warp from the 2 recording CTAs' 8 warps; span 1.54 us, warp 0.84 us, nsys graphs 1.57 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.46 | 54.5% | 1.27 | 76.0% |
| unattributed | 0.31 | 36.8% | 0.34 | 20.4% |
| store | 0.07 | 8.6% | 0.06 | 3.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#3 | 0.42 | 49.8% | 0.5 | 1.15 | `for (int __1 = 0; __1 < 4; ++__1)` |
| load.global#2 | 0.03 | 3.3% | 0.1 | 0.22 | `if ((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) < nr))` |
| store.global#3 | 0.03 | 3.3% | 0.5 | 0.06 | `if (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) == 0)` |
| store.global#4 | 0.03 | 3.3% | 0.5 | 0.06 | `positions[(((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))` |
| load.global | 0.01 | 1.4% | 0.1 | 0.10 | `if ((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) <= nr))` |
| store.global | 0.01 | 1.4% | 0.1 | 0.10 | `out_starts[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] = condval;` |
| store.global#2 | 0.00 | 0.5% | 0.1 | 0.03 | `request_positions[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] = condval_2;` |

### token_ranks_kernel grid (8, 1, 1) block (256, 1, 1)

51 launches, 64 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 7.81 us, warp 7.75 us, nsys graphs 8.86 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 6.99 | 90.2% | 8.13 | 91.3% |
| unattributed | 0.37 | 4.8% | 0.37 | 4.1% |
| compute | 0.34 | 4.4% | 0.36 | 4.1% |
| store | 0.05 | 0.6% | 0.05 | 0.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 6.99 | 90.2% | 1.0 | 7.23 | `if ((((int64_t)0 <= token_ids[((int64_t)((int)blockIdx.x))]) && (token_ids[((int64_t)((int` |
| compute.reduce | 0.34 | 4.4% | 1.0 | 0.61 | `total[0] = tl::AllReduce<tl::SumOp, 256, 1, 0, tl::NamedBarrier<256>>::run(total[0], (&(wo` |
| store.global | 0.05 | 0.6% | 1.0 | 0.06 | `output[((int64_t)((int)blockIdx.x))] = ((int64_t)total[0]);` |

### indexer_topk_segment_kernel grid (1, 48, 1) block (512, 1, 1)

34 launches, 768 warps; ranges, warp and last warp from the 4 recording CTAs' 64 warps; span 11.01 us, warp 10.80 us, nsys graphs 0.00 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed, with loops' unsampled iterations | 5.57 | 51.6% | 5.51 | 50.9% |
| wait | 3.76 | 34.8% | 3.88 | 35.9% |
| load | 1.31 | 12.1% | 1.23 | 11.3% |
| store | 0.15 | 1.4% | 0.20 | 1.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.31 | 12.1% | 1.0 | 1.49 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 4096); \| i` |
| wait.sync (sampled iterations) | 0.99 | 9.1% | 14.5 | 1.14 | `__syncthreads();` |
| wait.sync#20*9 (est.: sampled iterations x9) | 0.32 | 2.9% | 0.5 | 0.43 | `__syncthreads();` |
| wait.sync#24*9 (est.: sampled iterations x9) | 0.31 | 2.8% | 0.5 | 0.43 | `__syncthreads();` |
| wait.sync#19*9 (est.: sampled iterations x9) | 0.30 | 2.8% | 0.5 | 0.43 | `__syncthreads();` |
| wait.sync#23*9 (est.: sampled iterations x9) | 0.29 | 2.7% | 0.5 | 0.43 | `__syncthreads();` |
| wait.sync#14*8 (est.: sampled iterations x8) | 0.29 | 2.7% | 0.5 | 0.38 | `__syncthreads();` |
| wait.sync#6*8 (est.: sampled iterations x8) | 0.28 | 2.6% | 0.5 | 0.38 | `__syncthreads();` |
| wait.sync#13*8 (est.: sampled iterations x8) | 0.26 | 2.4% | 0.5 | 0.38 | `__syncthreads();` |
| wait.sync#5*8 (est.: sampled iterations x8) | 0.26 | 2.4% | 0.5 | 0.38 | `__syncthreads();` |
| store.global | 0.15 | 1.4% | 1.5 | 0.19 | `for (int i_3 = 0; i_3 < 4; ++i_3) \| if (((int)threadIdx.x) < total) \| if (((int)threadIdx.` |
| wait.sync#25 | 0.04 | 0.4% | 0.5 | 0.08 | `__syncthreads();` |
| wait.sync#16 | 0.04 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#17 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#8 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#18 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#3 | 0.03 | 0.3% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#11 | 0.03 | 0.3% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#2 | 0.03 | 0.3% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#15 | 0.03 | 0.3% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#7 | 0.03 | 0.3% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#9 | 0.03 | 0.3% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#4 | 0.03 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#22 | 0.03 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#10 | 0.03 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#12 | 0.02 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#21 | 0.02 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |

### norm_forward_and_per_token_cast_kernel_kernel grid (6, 1, 1) block (128, 1, 1)

105 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 2.94 us, warp 2.91 us, nsys graphs 1.57 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 2.47 | 84.6% | 3.41 | 98.3% |
| load | 0.44 | 15.2% | 0.61 | 17.6% |
| compute | 0.11 | 3.8% | 0.11 | 3.2% |
| wait | 0.05 | 1.8% | 0.05 | 1.5% |
| unattributed | 0.00 | 0.0% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*4 (est.: sampled iterations x4) | 1.79 | 61.5% | 1.0 | 1.79 | `for (int j_2 = 0; j_2 < 2; ++j_2)` |
| store.global*4 (est.: sampled iterations x4) | 0.41 | 14.1% | 1.0 | 0.51 | `for (int j_1 = 0; j_1 < 4; ++j_1)` |
| load.global*4 (est.: sampled iterations x4) | 0.24 | 8.2% | 1.0 | 0.26 | `if (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 4)) < num_tokens)` |
| load.global#3*4 (est.: sampled iterations x4) | 0.17 | 5.7% | 1.0 | 0.26 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 128) + ((((int)threadIdx.x) & 15` |
| store.global#5*4 (est.: sampled iterations x4) | 0.17 | 5.7% | 1.0 | 0.26 | `*(uint4*)(out + ((((((int64_t)vi_1) * (int64_t)128) + ((((int64_t)((int)threadIdx.x)) & (i` |
| compute.reduce | 0.11 | 3.8% | 1.0 | 0.13 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| store.global#3 | 0.05 | 1.8% | 1.0 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |
| wait.sync | 0.05 | 1.8% | 1.0 | 0.10 | `__syncthreads();` |
| store.global#2 | 0.05 | 1.6% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| load.global#2 | 0.04 | 1.3% | 1.0 | 0.06 | `*(float2*)(row_partial + 0) = *(float2*)(sum_shared + ((((int)threadIdx.x) >> 6) * 2)); \| ` |

### indexer_score_kernel grid (128, 48, 1) block (128, 1, 1)

17 launches, 24576 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 20.99 us, warp 2.63 us, nsys graphs 15.71 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.38 | 52.5% | 0.70 | 67.5% |
| unattributed, with loops' unsampled iterations | 0.93 | 35.2% | 0.31 | 30.0% |
| wait | 0.22 | 8.5% | 0.02 | 1.4% |
| compute | 0.07 | 2.8% | 0.01 | 0.7% |
| store | 0.03 | 1.0% | 0.00 | 0.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 (sampled iterations) | 0.85 | 32.1% | 0.5 | 1.98 | `for (int i_1 = 0; i_1 < 32; ++i_1)` |
| load.global | 0.40 | 15.0% | 1.0 | 0.93 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 4096); \| i` |
| load.global#3 (sampled iterations) | 0.14 | 5.3% | 0.5 | 0.58 | `if (((int)threadIdx.x) < 64)` |
| wait.sync#2 (sampled iterations) | 0.13 | 4.9% | 0.5 | 0.54 | `__syncthreads();` |
| compute.mma (sampled iterations) | 0.07 | 2.8% | 0.5 | 0.16 | `for (int ki = 0; ki < 2; ++ki)` |
| wait.sync (sampled iterations) | 0.03 | 1.2% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#4 (sampled iterations) | 0.03 | 1.2% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#3 (sampled iterations) | 0.03 | 1.1% | 0.5 | 0.06 | `__syncthreads();` |
| store.global (sampled iterations) | 0.03 | 1.0% | 0.5 | 0.06 | `if ((((int)threadIdx.x) >> 6) == 0)` |

### metadata_chunk_kernel grid (48, 1, 1) block (128, 1, 1)

34 launches, 192 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 10.74 us, warp 3.42 us, nsys graphs 10.06 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.78 | 52.2% | 1.11 | 11.0% |
| load | 1.55 | 45.4% | 8.84 | 87.6% |
| store | 0.08 | 2.4% | 0.14 | 1.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.40 | 40.9% | 1.0 | 3.18 | `if ((((offset + ((int)blockIdx.x)) < src_rows) && (0 <= (offset + ((int)blockIdx.x)))))` |
| load.global#2 | 0.15 | 4.5% | 0.2 | 1.42 | `if (((0 <= req) & ((int64_t)0 <= pos)))` |
| store.global | 0.06 | 1.8% | 1.0 | 0.10 | `swa[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = con` |
| store.global#2 | 0.01 | 0.4% | 0.2 | 0.06 | `lengths[((int64_t)((int)blockIdx.x))] = ((int)max(info[1], (int64_t)0));` |
| store.global#3 | 0.01 | 0.3% | 0.2 | 0.06 | `top_lengths[((int64_t)((int)blockIdx.x))] = condval_12; \| top_lengths[((int64_t)((int)bloc` |

### topk_ids_kernel grid (16, 1, 1) block (512, 1, 1)

16 launches, 256 warps; ranges, warp and last warp from the 4 recording CTAs' 64 warps; span 21.02 us, warp 19.11 us, nsys graphs 21.06 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 15.91 | 83.2% | 16.27 | 77.4% |
| unattributed | 2.09 | 10.9% | 1.96 | 9.3% |
| wait | 0.91 | 4.7% | 0.71 | 3.4% |
| store | 0.21 | 1.1% | 2.08 | 9.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 15.91 | 83.2% | 1.0 | 17.02 | `for (int t = 0; t < (vocab >> 14); ++t)` |
| wait.sync | 0.77 | 4.0% | 1.0 | 1.73 | `__syncthreads();` |
| store.global | 0.21 | 1.1% | 1.0 | 2.58 | `if ((((int)threadIdx.x) >> 5) == 0)` |
| wait.sync#2 | 0.08 | 0.4% | 1.0 | 0.16 | `__syncthreads();` |
| wait.sync#3 | 0.06 | 0.3% | 1.0 | 0.06 | `__syncthreads();` |

### topk_log_softmax_kernel grid (16, 1, 1) block (1024, 1, 1)

16 launches, 512 warps; ranges, warp and last warp from the 4 recording CTAs' 128 warps; span 20.86 us, warp 18.99 us, nsys graphs 18.16 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 15.70 | 82.7% | 17.28 | 85.1% |
| unattributed, with loops' unsampled iterations | 1.55 | 8.2% | 1.70 | 8.4% |
| wait | 1.35 | 7.1% | 0.89 | 4.4% |
| compute | 0.39 | 2.0% | 0.37 | 1.8% |
| store | 0.00 | 0.0% | 0.07 | 0.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 8.22 | 43.3% | 1.0 | 9.62 | `if (0 < (vocab & 2047)) \| if (0 < (vocab & 8191))` |
| load.global#2 | 7.47 | 39.4% | 1.0 | 8.16 | `for (int step_1 = 0; step_1 < (vocab >> 11); ++step_1) \| for (int step_1 = 0; step_1 < (vo` |
| wait.sync | 0.59 | 3.1% | 1.0 | 1.38 | `__syncthreads();` |
| wait.sync#2 | 0.50 | 2.6% | 1.0 | 1.22 | `__syncthreads();` |
| compute.reduce#3 | 0.17 | 0.9% | 1.0 | 0.21 | `ties[0] = vllm_add(ties[0], __shfl_xor_sync((uint)4294967295, ties[0], 16, 32));` |
| compute.reduce#2 | 0.11 | 0.6% | 1.0 | 0.16 | `se[0] = vllm_add(se[0], __shfl_xor_sync((uint)4294967295, se[0], 16, 32));` |
| compute.reduce | 0.10 | 0.5% | 1.0 | 0.13 | `best[0] = vllm_max(best[0], __shfl_xor_sync((uint)4294967295, best[0], 16, 32));` |
| wait.sync#4 | 0.10 | 0.5% | 1.0 | 0.16 | `__syncthreads();` |
| wait.sync#5 | 0.08 | 0.4% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#3 | 0.08 | 0.4% | 1.0 | 0.10 | `__syncthreads();` |
| load.global#3 (sampled iterations) | 0.01 | 0.1% | 0.0 | 0.38 | `float value_32 = vllm_sub(vllm_shifted(logits[((((int64_t)((int)blockIdx.x)) * logits_stri` |
| store.global (sampled iterations) | 0.00 | 0.0% | 0.0 | 0.06 | `output[(((((int64_t)c) * (int64_t)1024) + (((int64_t)((int)blockIdx.x)) * ((int64_t)topk))` |

### topk_log_softmax_kernel grid (8, 1, 1) block (1024, 1, 1)

17 launches, 256 warps; ranges, warp and last warp from the 4 recording CTAs' 128 warps; span 17.66 us, warp 16.87 us, nsys graphs 16.45 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 13.75 | 81.5% | 14.96 | 83.2% |
| unattributed, with loops' unsampled iterations | 1.63 | 9.6% | 1.74 | 9.7% |
| wait | 1.09 | 6.5% | 0.82 | 4.6% |
| compute | 0.39 | 2.3% | 0.39 | 2.2% |
| store | 0.00 | 0.0% | 0.07 | 0.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 7.05 | 41.8% | 1.0 | 7.52 | `if (0 < (vocab & 2047)) \| if (0 < (vocab & 8191))` |
| load.global#2 | 6.70 | 39.7% | 1.0 | 7.39 | `for (int step_1 = 0; step_1 < (vocab >> 11); ++step_1) \| for (int step_1 = 0; step_1 < (vo` |
| wait.sync#2 | 0.50 | 2.9% | 1.0 | 1.06 | `__syncthreads();` |
| wait.sync | 0.34 | 2.0% | 1.0 | 0.90 | `__syncthreads();` |
| compute.reduce#3 | 0.17 | 1.0% | 1.0 | 0.22 | `ties[0] = vllm_add(ties[0], __shfl_xor_sync((uint)4294967295, ties[0], 16, 32));` |
| compute.reduce#2 | 0.11 | 0.7% | 1.0 | 0.16 | `se[0] = vllm_add(se[0], __shfl_xor_sync((uint)4294967295, se[0], 16, 32));` |
| compute.reduce | 0.11 | 0.6% | 1.0 | 0.13 | `best[0] = vllm_max(best[0], __shfl_xor_sync((uint)4294967295, best[0], 16, 32));` |
| wait.sync#4 | 0.10 | 0.6% | 1.0 | 0.16 | `__syncthreads();` |
| wait.sync#3 | 0.08 | 0.5% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#5 | 0.08 | 0.5% | 1.0 | 0.10 | `__syncthreads();` |
| load.global#3 (sampled iterations) | 0.01 | 0.1% | 0.0 | 0.38 | `float value_32 = vllm_sub(vllm_shifted(logits[((((int64_t)((int)blockIdx.x)) * logits_stri` |
| store.global (sampled iterations) | 0.00 | 0.0% | 0.0 | 0.06 | `output[(((((int64_t)c) * (int64_t)1024) + (((int64_t)((int)blockIdx.x)) * ((int64_t)topk))` |

### sparse_mla_kernel_1 grid (24, 40, 1) block (128, 1, 1)

51 launches, 960 CTAs, 3840 warps; span 5.76 us, warp 2.30 us, nsys graphs 4.54 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 2.29 | 99.7% | 2.43 | 97.7% |
| load | 0.01 | 0.2% | 0.05 | 1.9% |
| wait | 0.00 | 0.1% | 0.01 | 0.3% |
| compute | 0.00 | 0.0% | 0.00 | 0.1% |
| store | 0.00 | 0.0% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#3 | 0.00 | 0.2% | 0.0 | 1.76 | `M = __fmul_rn(Sinks[((int)blockIdx.x)], 0x1.7154764ee6c2fp+0f/*1.442695e+00*/);` |
| wait.sync#13 | 0.00 | 0.0% | 0.0 | 0.80 | `__syncthreads();` |
| load.global#4 | 0.00 | 0.0% | 0.0 | 0.83 | `if (((int)threadIdx.x) < 32)` |
| wait.sync#12 | 0.00 | 0.0% | 0.0 | 0.38 | `__syncthreads();` |
| compute.reduce#3 | 0.00 | 0.0% | 0.0 | 0.19 | `amax[0] = tl::AllReduce<tl::MaxOp, 8, 1, 0, tl::NamedBarrier<128>>::run(amax[0]);` |
| store.global#2 | 0.00 | 0.0% | 0.0 | 0.10 | `*(uint2*)(Output + (((((int64_t)((int)blockIdx.y)) * (int64_t)12288) + (((int64_t)((int)bl` |

### moe_combine_kernel grid (5, 48, 1) block (128, 1, 1)

102 launches, 960 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 2.66 us, warp 1.79 us, nsys graphs 2.75 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.30 | 72.7% | 2.36 | 84.0% |
| unattributed | 0.43 | 24.2% | 0.40 | 14.2% |
| store | 0.06 | 3.1% | 0.05 | 1.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.30 | 72.7% | 1.0 | 2.11 | `for (int k = 0; k < 3; ++k) \| for (int k = 0; k < 6; ++k) \| if ((((int)blockIdx.y) < share` |
| store.global | 0.06 | 3.1% | 1.0 | 0.06 | `{` |

### sink_rows_kernel grid (8, 64, 1) block (256, 1, 1)

17 launches, 4096 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 16.45 us, warp 4.96 us, nsys graphs 7.76 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 2.38 | 48.1% | 4.29 | 55.1% |
| load | 1.34 | 27.0% | 2.13 | 27.3% |
| unattributed | 1.24 | 25.0% | 1.38 | 17.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 2.35 | 47.4% | 1.0 | 5.54 | `for (int k_1 = 0; k_1 < 8; ++k_1)` |
| load.global | 1.33 | 26.9% | 1.0 | 3.58 | `row = (max((((int64_t)num_sampled[((int64_t)((int)blockIdx.x))]) - (int64_t)1), (int64_t)0` |
| store.global#2 | 0.03 | 0.7% | 0.0 | 1.06 | `tokens[((((int64_t)((int)blockIdx.x)) * (int64_t)3) + ((int64_t)((int)threadIdx.x)))] = to` |
| load.global#2 | 0.00 | 0.1% | 0.0 | 0.13 | `token = max(alternates[((((int64_t)((int)blockIdx.x)) * (int64_t)3) + ((int64_t)((int)thre` |

### norm_forward_and_per_token_cast_kernel_kernel grid (24, 1, 1) block (128, 1, 1)

102 launches, 96 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 2.50 us, warp 2.29 us, nsys graphs 1.95 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 1.57 | 68.9% | 2.03 | 70.5% |
| load | 0.34 | 15.0% | 0.35 | 12.1% |
| unattributed | 0.14 | 6.1% | 0.33 | 11.6% |
| compute | 0.12 | 5.2% | 0.12 | 4.1% |
| wait | 0.11 | 4.9% | 0.05 | 1.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*3 (est.: sampled iterations x3) | 1.06 | 46.2% | 1.0 | 1.15 | `for (int j_2 = 0; j_2 < 2; ++j_2)` |
| store.global*3 (est.: sampled iterations x3) | 0.31 | 13.7% | 1.0 | 0.38 | `for (int j_1 = 0; j_1 < 4; ++j_1)` |
| load.global*3 (est.: sampled iterations x3) | 0.17 | 7.6% | 1.0 | 0.19 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 6)) < num_tokens) && (((vi * 2) + (` |
| load.global#3*3 (est.: sampled iterations x3) | 0.13 | 5.8% | 1.0 | 0.19 | `if (((vi_1 * 2) + ((((int)threadIdx.x) & 63) >> 5)) < 5)` |
| store.global#5*3 (est.: sampled iterations x3) | 0.13 | 5.8% | 1.0 | 0.19 | `*(uint4*)(out + ((((((int64_t)vi_1) * (int64_t)512) + ((((int64_t)((int)threadIdx.x)) & (i` |
| compute.reduce | 0.12 | 5.2% | 1.0 | 0.13 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| wait.sync | 0.11 | 4.9% | 1.0 | 0.19 | `__syncthreads();` |
| store.global#2 | 0.04 | 1.8% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| load.global#2 | 0.04 | 1.6% | 1.0 | 0.06 | `*(float2*)(row_partial + 0) = *(float2*)(sum_shared + ((((int)threadIdx.x) >> 6) * 2)); \| ` |
| store.global#3 | 0.03 | 1.4% | 0.5 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |

### rope_kernel grid (3, 1, 1) block (128, 1, 1)

224 launches, 12 warps; ranges, warp and last warp from the 3 recording CTAs' 12 warps; span 0.99 us, warp 0.95 us, nsys graphs 1.30 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.72 | 76.2% | 0.90 | 74.0% |
| load | 0.19 | 20.2% | 0.28 | 22.8% |
| store | 0.03 | 3.6% | 0.04 | 3.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.10 | 10.1% | 1.0 | 0.10 | `if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < tokens)) \| int64_t position ` |
| load.global#2 | 0.10 | 10.1% | 1.0 | 0.10 | `if (((position < ((int64_t)positions_max)) && ((int64_t)0 <= position)))` |
| store.global | 0.03 | 3.6% | 1.0 | 0.06 | `*(uint4*)(x + ((((((int64_t)((int)blockIdx.x)) * (int64_t)12288) + ((((int64_t)((int)threa` |

### moe_swiglu_cast_kernel_kernel grid (16, 120, 1) block (96, 1, 1)

51 launches, 5760 warps; ranges, warp and last warp from the 4 recording CTAs' 12 warps; span 4.51 us, warp 1.44 us, nsys graphs 4.05 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.10 | 76.6% | 3.45 | 79.8% |
| load | 0.27 | 18.6% | 0.56 | 13.0% |
| compute | 0.04 | 2.6% | 0.16 | 3.7% |
| store | 0.03 | 2.2% | 0.15 | 3.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.27 | 18.6% | 1.0 | 0.58 | `for (int r = 0; r < 2; ++r)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.04 | 2.6% | 0.2 | 0.19 | `if (0 <= orders[r_2])` |
| store.global*2 (est.: sampled iterations x2) | 0.03 | 2.2% | 0.2 | 0.13 | `if (0 <= orders[r_2])` |

### gemm_splitk_reduce_cast_kernel grid (24, 12, 1) block (128, 1, 1)

102 launches, 1152 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 2.05 us, warp 1.69 us, nsys graphs 2.24 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.02 | 60.3% | 1.56 | 69.8% |
| unattributed | 0.50 | 29.3% | 0.51 | 22.6% |
| store | 0.09 | 5.3% | 0.09 | 3.9% |
| compute | 0.09 | 5.1% | 0.08 | 3.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.02 | 60.3% | 1.0 | 1.31 | `if ((1 <= M))` |
| store.global | 0.09 | 5.3% | 1.0 | 0.10 | `if (((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 5)) < M)` |
| compute.reduce | 0.09 | 5.1% | 1.0 | 0.10 | `amax[0] = tl::AllReduce<tl::MaxOp, 8, 1, 0, tl::NamedBarrier<128>>::run(amax[0]);` |

### compress_kernel grid (12, 1, 1) block (128, 1, 1)

34 launches, 48 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 6.18 us, warp 4.80 us, nsys graphs 7.63 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 2.34 | 48.8% | 1.60 | 24.0% |
| load | 1.92 | 40.0% | 3.86 | 57.8% |
| store | 0.45 | 9.3% | 1.12 | 16.8% |
| compute | 0.09 | 2.0% | 0.09 | 1.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.18 | 24.5% | 1.0 | 1.49 | `if ((((0 <= counts[0]) & (0 < counts[1])) & (counts[1] <= 8)) & (((((int)blockIdx.x) * 4) ` |
| load.global#2*4 (est.: sampled iterations x4) | 0.64 | 13.2% | 1.0 | 1.34 | `*(float4*)(v1 + 0) = *(float4*)(values + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048)` |
| store.global*4 (est.: sampled iterations x4) | 0.16 | 3.2% | 1.0 | 0.26 | `*(uint2*)(out + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + ((((int64_t)((int)thre` |
| compute.reduce | 0.09 | 2.0% | 1.0 | 0.10 | `float total = tl::warp_reduce_sum(ss[0]);` |
| store.global#3 | 0.09 | 1.8% | 0.5 | 0.16 | `emitted_slots[((((int64_t)((int)blockIdx.x)) * (int64_t)4) + (((int64_t)((int)threadIdx.x)` |
| load.global#3*4 (est.: sampled iterations x4) | 0.07 | 1.4% | 0.5 | 0.13 | `*(float4*)(w + 0) = *(float4*)(weight + ((c_1 * 128) + ((((int)threadIdx.x) & 31) * 4)));` |
| store.global#3*4 (est.: sampled iterations x4) | 0.06 | 1.3% | 0.5 | 0.06 | `*(uint2*)(out + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + ((((int64_t)((int)thre` |
| store.global#2*4 (est.: sampled iterations x4) | 0.06 | 1.2% | 0.2 | 0.13 | `if ((slot >> (int64_t)3) < ((int64_t)blocks))` |
| store.global#5 | 0.05 | 1.0% | 0.5 | 0.16 | `emitted_slots[((((int64_t)((int)blockIdx.x)) * (int64_t)4) + (((int64_t)((int)threadIdx.x)` |
| load.global#3 | 0.02 | 0.4% | 0.5 | 0.03 | `if (valid)` |
| load.global#4 | 0.02 | 0.4% | 0.5 | 0.03 | `if (valid)` |
| store.global#2 | 0.02 | 0.4% | 0.5 | 0.03 | `emitted[((((int64_t)((int)blockIdx.x)) * (int64_t)4) + (((int64_t)((int)threadIdx.x)) >> (` |
| store.global#4 | 0.02 | 0.3% | 0.5 | 0.02 | `emitted[((((int64_t)((int)blockIdx.x)) * (int64_t)4) + (((int64_t)((int)threadIdx.x)) >> (` |

### sink_candidates_kernel grid (8, 127, 1) block (128, 1, 1)

17 launches, 4064 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 11.14 us, warp 4.51 us, nsys graphs 9.38 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| compute | 1.42 | 31.4% | 1.37 | 32.2% |
| unattributed | 1.30 | 28.8% | 0.88 | 20.6% |
| load | 0.91 | 20.1% | 1.15 | 27.1% |
| wait | 0.60 | 13.3% | 0.50 | 11.8% |
| store | 0.29 | 6.4% | 0.35 | 8.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.reduce*4 (est.: sampled iterations x4) | 1.02 | 22.7% | 1.0 | 1.28 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| load.global | 0.91 | 20.1% | 1.0 | 1.41 | `local[0] = max((num_sampled[((int64_t)((int)blockIdx.x))] - 1), 0);` |
| compute.reduce#2*4 (est.: sampled iterations x4) | 0.39 | 8.7% | 1.0 | 0.51 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 2, 32);` |
| wait.sync*4 (est.: sampled iterations x4) | 0.30 | 6.7% | 1.0 | 0.51 | `__syncthreads();` |
| wait.sync#2*4 (est.: sampled iterations x4) | 0.30 | 6.6% | 1.0 | 0.38 | `__syncthreads();` |
| store.global*4 (est.: sampled iterations x4) | 0.29 | 6.4% | 1.0 | 0.38 | `if (((int)threadIdx.x) == 0)` |

### sparse_mla_kernel_1 grid (24, 48, 1) block (128, 1, 1)

34 launches, 1152 CTAs, 4608 warps; span 5.90 us, warp 2.22 us, nsys graphs 6.48 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 2.22 | 99.8% | 1.89 | 100.0% |
| load | 0.00 | 0.2% | 0.00 | 0.0% |
| wait | 0.00 | 0.0% | 0.00 | 0.0% |
| compute | 0.00 | 0.0% | 0.00 | 0.0% |
| store | 0.00 | 0.0% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#3 | 0.00 | 0.1% | 0.0 | 1.49 | `M = __fmul_rn(Sinks[((int)blockIdx.x)], 0x1.7154764ee6c2fp+0f/*1.442695e+00*/);` |
| wait.sync#13 | 0.00 | 0.0% | 0.0 | 0.30 | `__syncthreads();` |
| compute.reduce#3 | 0.00 | 0.0% | 0.0 | 0.19 | `amax[0] = tl::AllReduce<tl::MaxOp, 8, 1, 0, tl::NamedBarrier<128>>::run(amax[0]);` |
| load.global#4 | 0.00 | 0.0% | 0.0 | 0.34 | `if (((int)threadIdx.x) < 32)` |
| wait.sync#12 | 0.00 | 0.0% | 0.0 | 0.22 | `__syncthreads();` |
| store.global#2 | 0.00 | 0.0% | 0.0 | 0.10 | `*(uint2*)(Output + (((((int64_t)((int)blockIdx.y)) * (int64_t)12288) + (((int64_t)((int)bl` |

### mhc_post_kernel grid (200, 1, 1) block (128, 1, 1)

17 launches, 800 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 11.39 us, warp 7.79 us, nsys graphs 4.27 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 7.57 | 97.3% | 10.33 | 92.6% |
| load | 0.16 | 2.0% | 0.66 | 5.9% |
| store | 0.06 | 0.7% | 0.17 | 1.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.16 | 2.0% | 1.0 | 0.19 | `*(float4*)(pm + 0) = *(float4*)(post + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int` |
| store.global | 0.06 | 0.7% | 1.0 | 0.13 | `for (int n_1 = 0; n_1 < 4; ++n_1)` |

### per_token_cast_kernel_kernel grid (30, 1, 1) block (64, 1, 1)

136 launches, 60 warps; ranges, warp and last warp from the 5 recording CTAs' 10 warps; span 1.25 us, warp 1.18 us, nsys graphs 1.12 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.43 | 36.2% | 0.44 | 30.4% |
| load | 0.36 | 30.8% | 0.60 | 42.0% |
| store | 0.29 | 24.9% | 0.29 | 20.2% |
| compute | 0.10 | 8.1% | 0.11 | 7.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.28 | 23.2% | 1.0 | 0.29 | `for (int i = 0; i < 2; ++i)` |
| store.global#2 | 0.11 | 9.5% | 1.0 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.10 | 8.1% | 1.0 | 0.13 | `amax_fragment[i_1] = tl::AllReduce<tl::MaxOp, 2, 1, 0, tl::NamedBarrier<64>>::run(amax_fra` |
| store.global*2 (est.: sampled iterations x2) | 0.09 | 7.8% | 1.0 | 0.13 | `out_sf[((((((int64_t)((int)blockIdx.x)) % (int64_t)15) * (int64_t)32) + (((int64_t)((int)t` |
| load.global#2*2 (est.: sampled iterations x2) | 0.09 | 7.6% | 1.0 | 0.13 | `amax_fragment[i_1] = cutlass::bfloat16_t(__hmax((bfloat16_t(((nv_bfloat162*)(&(amax_fragme` |
| store.global#3*2 (est.: sampled iterations x2) | 0.09 | 7.6% | 1.0 | 0.13 | `*(fp8_e4_16_t*)(out + ((((((((int64_t)((int)blockIdx.x)) / (int64_t)3) * (int64_t)6144) + ` |

### indexer_score_kernel grid (64, 48, 1) block (128, 1, 1)

17 launches, 12288 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 10.85 us, warp 2.43 us, nsys graphs 0.00 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.14 | 46.9% | 0.51 | 61.5% |
| unattributed, with loops' unsampled iterations | 0.86 | 35.5% | 0.29 | 35.5% |
| wait | 0.32 | 13.1% | 0.01 | 1.6% |
| compute | 0.08 | 3.3% | 0.01 | 0.9% |
| store | 0.03 | 1.2% | 0.00 | 0.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 (sampled iterations) | 0.64 | 26.5% | 0.5 | 1.44 | `for (int i_1 = 0; i_1 < 32; ++i_1)` |
| load.global | 0.39 | 16.2% | 1.0 | 0.93 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 4096); \| i` |
| wait.sync#2 (sampled iterations) | 0.18 | 7.3% | 0.5 | 0.51 | `__syncthreads();` |
| load.global#3 (sampled iterations) | 0.10 | 4.1% | 0.5 | 0.51 | `if (((int)threadIdx.x) < 64)` |
| compute.mma (sampled iterations) | 0.08 | 3.3% | 0.5 | 0.19 | `for (int ki = 0; ki < 2; ++ki)` |
| wait.sync (sampled iterations) | 0.05 | 2.1% | 0.5 | 0.19 | `__syncthreads();` |
| wait.sync#3 (sampled iterations) | 0.05 | 1.9% | 0.5 | 0.13 | `__syncthreads();` |
| wait.sync#4 (sampled iterations) | 0.04 | 1.7% | 0.5 | 0.10 | `__syncthreads();` |
| store.global (sampled iterations) | 0.03 | 1.2% | 0.5 | 0.06 | `if ((((int)threadIdx.x) >> 6) == 0)` |

### indexer_topk_segment_kernel grid (2, 48, 1) block (512, 1, 1)

17 launches, 1536 warps; ranges, warp and last warp from the 4 recording CTAs' 64 warps; span 9.98 us, warp 5.28 us, nsys graphs 0.00 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 3.06 | 58.0% | 6.16 | 59.6% |
| unattributed, with loops' unsampled iterations | 1.83 | 34.7% | 3.25 | 31.4% |
| load | 0.34 | 6.5% | 0.82 | 7.9% |
| store | 0.04 | 0.9% | 0.11 | 1.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync#20*9 (est.: sampled iterations x9) | 0.33 | 6.3% | 0.5 | 0.86 | `__syncthreads();` |
| wait.sync#19*9 (est.: sampled iterations x9) | 0.33 | 6.2% | 0.5 | 0.86 | `__syncthreads();` |
| wait.sync#24*9 (est.: sampled iterations x9) | 0.32 | 6.1% | 0.5 | 0.86 | `__syncthreads();` |
| wait.sync#23*9 (est.: sampled iterations x9) | 0.32 | 6.0% | 0.5 | 0.86 | `__syncthreads();` |
| wait.sync#6*8 (est.: sampled iterations x8) | 0.30 | 5.7% | 0.5 | 0.77 | `__syncthreads();` |
| wait.sync#14*8 (est.: sampled iterations x8) | 0.30 | 5.6% | 0.5 | 0.77 | `__syncthreads();` |
| wait.sync#5*8 (est.: sampled iterations x8) | 0.29 | 5.5% | 0.5 | 0.77 | `__syncthreads();` |
| wait.sync#13*8 (est.: sampled iterations x8) | 0.28 | 5.2% | 0.5 | 0.77 | `__syncthreads();` |
| load.global | 0.19 | 3.6% | 1.0 | 0.22 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 4096); \| i` |
| load.global#2 | 0.15 | 2.8% | 0.5 | 0.38 | `for (int i = 0; i < 8; ++i)` |
| wait.sync (sampled iterations) | 0.06 | 1.1% | 0.5 | 0.19 | `__syncthreads();` |
| wait.sync#25 | 0.05 | 0.9% | 0.5 | 0.16 | `__syncthreads();` |
| store.global | 0.04 | 0.9% | 0.5 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3) \| if (((int)threadIdx.x) < total) \| if (((int)threadIdx.` |
| wait.sync#17 | 0.04 | 0.8% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#18 | 0.04 | 0.7% | 0.5 | 0.13 | `__syncthreads();` |
| wait.sync#16 | 0.04 | 0.7% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#8 | 0.04 | 0.7% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#9 | 0.03 | 0.6% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#3 | 0.03 | 0.6% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#2 | 0.03 | 0.6% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#7 | 0.03 | 0.6% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#10 | 0.03 | 0.6% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#15 | 0.03 | 0.6% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#12 | 0.03 | 0.6% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#4 | 0.03 | 0.6% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#11 | 0.03 | 0.6% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#22 | 0.03 | 0.5% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#21 | 0.03 | 0.5% | 0.5 | 0.06 | `__syncthreads();` |

### moe_combine_kernel grid (5, 40, 1) block (128, 1, 1)

51 launches, 800 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 3.10 us, warp 1.94 us, nsys graphs 2.08 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.56 | 80.5% | 2.38 | 86.1% |
| unattributed | 0.32 | 16.6% | 0.33 | 11.9% |
| store | 0.06 | 2.9% | 0.06 | 2.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.56 | 80.5% | 1.0 | 2.11 | `for (int k = 0; k < 3; ++k) \| for (int k = 0; k < 6; ++k) \| if ((((int)blockIdx.y) < share` |
| store.global | 0.06 | 2.9% | 1.0 | 0.06 | `{` |

### indexer_topk_merge_kernel grid (1, 48, 1) block (512, 1, 1)

17 launches, 768 warps; ranges, warp and last warp from the 4 recording CTAs' 64 warps; span 9.79 us, warp 9.49 us, nsys graphs 8.03 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 6.15 | 64.7% | 5.85 | 59.6% |
| unattributed | 2.27 | 23.9% | 2.53 | 25.8% |
| load | 1.01 | 10.6% | 1.31 | 13.4% |
| store | 0.07 | 0.7% | 0.11 | 1.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.01 | 10.6% | 1.0 | 1.18 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 8192);` |
| wait.sync#20*9 (est.: sampled iterations x9) | 0.66 | 7.0% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#24*9 (est.: sampled iterations x9) | 0.66 | 7.0% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#19*9 (est.: sampled iterations x9) | 0.64 | 6.8% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#23*9 (est.: sampled iterations x9) | 0.63 | 6.7% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#6*8 (est.: sampled iterations x8) | 0.62 | 6.5% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync#14*8 (est.: sampled iterations x8) | 0.60 | 6.4% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync#13*8 (est.: sampled iterations x8) | 0.58 | 6.1% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync#5*8 (est.: sampled iterations x8) | 0.58 | 6.1% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync | 0.14 | 1.5% | 1.0 | 0.32 | `__syncthreads();` |
| wait.sync#17 | 0.08 | 0.8% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#16 | 0.08 | 0.8% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#8 | 0.07 | 0.8% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#2 | 0.07 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#25 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#3 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#9 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| store.global#2 | 0.06 | 0.7% | 1.0 | 0.10 | `Values[((((int64_t)((int)blockIdx.y)) * (int64_t)512) + ((int64_t)((int)threadIdx.x)))] = ` |
| wait.sync#15 | 0.06 | 0.6% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#7 | 0.06 | 0.6% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#11 | 0.06 | 0.6% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#12 | 0.06 | 0.6% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#18 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#21 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#22 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#4 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#10 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| load.global#2 | 0.01 | 0.1% | 0.1 | 0.10 | `if (((0 <= ((int*)sel_idx)[((int)threadIdx.x)]) && (((int*)sel_idx)[((int)threadIdx.x)] < ` |
| store.global | 0.00 | 0.0% | 0.1 | 0.10 | `Values[((((int64_t)((int)blockIdx.y)) * (int64_t)512) + ((int64_t)((int)threadIdx.x)))] = ` |

### per_token_cast_kernel_kernel grid (18, 1, 1) block (64, 1, 1)

102 launches, 36 warps; ranges, warp and last warp from the 5 recording CTAs' 10 warps; span 1.39 us, warp 1.24 us, nsys graphs 1.82 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.47 | 37.8% | 0.63 | 40.3% |
| unattributed | 0.33 | 26.9% | 0.40 | 25.3% |
| store | 0.33 | 26.4% | 0.41 | 26.3% |
| compute | 0.11 | 8.8% | 0.13 | 8.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.37 | 30.1% | 1.0 | 0.50 | `for (int i = 0; i < 2; ++i)` |
| store.global#2 | 0.12 | 9.3% | 1.0 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3)` |
| store.global*2 (est.: sampled iterations x2) | 0.12 | 9.3% | 1.0 | 0.13 | `out_sf[((((((int64_t)((int)blockIdx.x)) % (int64_t)15) * (int64_t)32) + (((int64_t)((int)t` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.11 | 8.8% | 1.0 | 0.13 | `amax_fragment[i_1] = tl::AllReduce<tl::MaxOp, 2, 1, 0, tl::NamedBarrier<64>>::run(amax_fra` |
| load.global#2*2 (est.: sampled iterations x2) | 0.10 | 7.8% | 1.0 | 0.13 | `amax_fragment[i_1] = cutlass::bfloat16_t(__hmax((bfloat16_t(((nv_bfloat162*)(&(amax_fragme` |
| store.global#3*2 (est.: sampled iterations x2) | 0.10 | 7.8% | 1.0 | 0.13 | `*(fp8_e4_16_t*)(out + ((((((((int64_t)((int)blockIdx.x)) / (int64_t)3) * (int64_t)6144) + ` |

### activation_swiglu_clamp_kernel grid (1, 48, 1) block (128, 1, 1)

102 launches, 192 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 1.02 us, warp 0.76 us, nsys graphs 2.27 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.65 | 85.8% | 1.11 | 84.7% |
| load | 0.07 | 9.5% | 0.15 | 11.4% |
| store | 0.04 | 4.7% | 0.05 | 3.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.07 | 9.5% | 0.8 | 0.10 | `*(uint4*)(gate + 0) = *(uint4*)(X + ((((int64_t)((int)blockIdx.y)) * (int64_t)1536) + (((i` |
| store.global | 0.04 | 4.7% | 0.8 | 0.06 | `*(uint4*)(Y + ((((int64_t)((int)blockIdx.y)) * (int64_t)768) + (((int64_t)((int)threadIdx.` |

### router_partials_kernel grid (20, 1, 1) block (64, 1, 1)

51 launches, 40 warps; ranges, warp and last warp from the 4 recording CTAs' 8 warps; span 2.53 us, warp 2.48 us, nsys graphs 6.61 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.03 | 41.6% | 1.12 | 39.5% |
| unattributed | 0.89 | 35.8% | 0.97 | 34.3% |
| compute | 0.34 | 13.9% | 0.34 | 12.1% |
| store | 0.22 | 8.7% | 0.40 | 14.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#4 | 0.52 | 21.0% | 1.0 | 0.54 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < image_len)) \| if ((((((int)bl` |
| load.global | 0.38 | 15.5% | 1.0 | 0.38 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < mask_len))` |
| compute.reduce | 0.34 | 13.9% | 1.0 | 0.35 | `for (int k = 0; k < 3; ++k) \| for (int k = 0; k < 6; ++k)` |
| store.global#3 | 0.11 | 4.5% | 1.0 | 0.13 | `if ((((int)threadIdx.x) & 31) < 3) \| if ((((int)threadIdx.x) & 31) < 6)` |
| load.global#2 | 0.06 | 2.6% | 1.0 | 0.06 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < mask_len))` |
| load.global#3 | 0.06 | 2.6% | 1.0 | 0.06 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < mask_len))` |
| store.global | 0.05 | 2.1% | 1.0 | 0.06 | `if (condval == (bool)0) \| if (condval == (bool)1)` |
| store.global#2 | 0.05 | 2.1% | 1.0 | 0.06 | `if (condval_1 == (bool)0) \| if (condval_1 == (bool)1)` |

### norm_forward_and_per_token_cast_kernel_kernel grid (2, 1, 1) block (128, 1, 1)

48 launches, 8 warps; ranges, warp and last warp from the 2 recording CTAs' 8 warps; span 2.85 us, warp 2.85 us, nsys graphs 1.41 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 2.55 | 89.6% | 2.55 | 88.9% |
| load | 0.42 | 14.9% | 0.44 | 15.3% |
| compute | 0.11 | 3.9% | 0.11 | 3.9% |
| wait | 0.05 | 1.8% | 0.07 | 2.3% |
| unattributed | 0.00 | 0.0% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*4 (est.: sampled iterations x4) | 1.90 | 66.9% | 1.0 | 1.92 | `for (int j_2 = 0; j_2 < 2; ++j_2)` |
| store.global*4 (est.: sampled iterations x4) | 0.42 | 14.6% | 1.0 | 0.51 | `for (int j_1 = 0; j_1 < 4; ++j_1)` |
| load.global*4 (est.: sampled iterations x4) | 0.23 | 8.1% | 1.0 | 0.26 | `if (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 4)) < num_tokens)` |
| load.global#3*4 (est.: sampled iterations x4) | 0.16 | 5.6% | 1.0 | 0.26 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 128) + ((((int)threadIdx.x) & 15` |
| store.global#5*4 (est.: sampled iterations x4) | 0.13 | 4.5% | 1.0 | 0.13 | `*(uint4*)(out + ((((((int64_t)vi_1) * (int64_t)128) + ((((int64_t)((int)threadIdx.x)) & (i` |
| compute.reduce | 0.11 | 3.9% | 1.0 | 0.13 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| store.global#3 | 0.06 | 2.0% | 1.0 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |
| wait.sync | 0.05 | 1.8% | 1.0 | 0.06 | `__syncthreads();` |
| store.global#2 | 0.05 | 1.7% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| load.global#2 | 0.03 | 1.1% | 1.0 | 0.03 | `*(float2*)(row_partial + 0) = *(float2*)(sum_shared + ((((int)threadIdx.x) >> 6) * 2)); \| ` |

### topk_ids_kernel grid (48, 1, 1) block (512, 1, 1)

1 launches, 768 warps; ranges, warp and last warp from the 4 recording CTAs' 64 warps; span 137.41 us, warp 120.51 us, nsys graphs 113.63 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 88.12 | 73.1% | 116.26 | 87.0% |
| wait | 29.10 | 24.1% | 13.82 | 10.3% |
| unattributed | 1.96 | 1.6% | 1.98 | 1.5% |
| store | 1.32 | 1.1% | 1.54 | 1.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 88.12 | 73.1% | 1.0 | 129.92 | `for (int t = 0; t < (vocab >> 14); ++t)` |
| wait.sync | 28.96 | 24.0% | 1.0 | 76.00 | `__syncthreads();` |
| store.global | 1.32 | 1.1% | 1.0 | 18.02 | `if ((((int)threadIdx.x) >> 5) == 0)` |
| wait.sync#2 | 0.08 | 0.1% | 1.0 | 0.19 | `__syncthreads();` |
| wait.sync#3 | 0.06 | 0.0% | 1.0 | 0.06 | `__syncthreads();` |

### rope_kernel grid (48, 1, 1) block (192, 1, 1)

102 launches, 288 warps; ranges, warp and last warp from the 4 recording CTAs' 24 warps; span 1.22 us, warp 1.18 us, nsys graphs 1.63 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.68 | 57.8% | 0.79 | 61.0% |
| load | 0.45 | 38.7% | 0.46 | 35.7% |
| store | 0.04 | 3.5% | 0.04 | 3.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.45 | 38.7% | 1.0 | 0.48 | `if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < tokens)) \| int64_t position ` |
| store.global | 0.04 | 3.5% | 1.0 | 0.06 | `*(uint4*)(x + ((((((int64_t)((int)blockIdx.x)) * (int64_t)12288) + ((((int64_t)((int)threa` |

### token_ranks_kernel grid (16, 1, 1) block (256, 1, 1)

16 launches, 128 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 8.43 us, warp 8.33 us, nsys graphs 9.50 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 7.51 | 90.2% | 7.60 | 89.7% |
| compute | 0.39 | 4.7% | 0.46 | 5.5% |
| unattributed | 0.38 | 4.5% | 0.36 | 4.2% |
| store | 0.05 | 0.6% | 0.05 | 0.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 7.51 | 90.2% | 1.0 | 7.82 | `if ((((int64_t)0 <= token_ids[((int64_t)((int)blockIdx.x))]) && (token_ids[((int64_t)((int` |
| compute.reduce | 0.39 | 4.7% | 1.0 | 0.77 | `total[0] = tl::AllReduce<tl::SumOp, 256, 1, 0, tl::NamedBarrier<256>>::run(total[0], (&(wo` |
| store.global | 0.05 | 0.6% | 1.0 | 0.06 | `output[((int64_t)((int)blockIdx.x))] = ((int64_t)total[0]);` |

### gemm_splitk_reduce_cast_kernel grid (24, 10, 1) block (128, 1, 1)

51 launches, 960 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 2.08 us, warp 1.60 us, nsys graphs 2.05 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.05 | 65.8% | 1.84 | 73.0% |
| unattributed | 0.37 | 23.2% | 0.51 | 20.2% |
| store | 0.09 | 5.6% | 0.09 | 3.5% |
| compute | 0.09 | 5.4% | 0.08 | 3.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.05 | 65.8% | 1.0 | 1.50 | `if ((1 <= M))` |
| store.global | 0.09 | 5.6% | 1.0 | 0.10 | `if (((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 5)) < M)` |
| compute.reduce | 0.09 | 5.4% | 1.0 | 0.10 | `amax[0] = tl::AllReduce<tl::MaxOp, 8, 1, 0, tl::NamedBarrier<128>>::run(amax[0]);` |

### norm_forward_and_per_token_cast_kernel_kernel grid (5, 1, 1) block (128, 1, 1)

51 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 2.75 us, warp 2.75 us, nsys graphs 1.79 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 2.28 | 82.8% | 2.05 | 80.2% |
| load | 0.46 | 16.7% | 0.45 | 17.8% |
| compute | 0.11 | 4.1% | 0.11 | 4.4% |
| wait | 0.05 | 2.0% | 0.06 | 2.2% |
| unattributed | 0.00 | 0.0% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*4 (est.: sampled iterations x4) | 1.60 | 58.1% | 1.0 | 1.66 | `for (int j_2 = 0; j_2 < 2; ++j_2)` |
| store.global*4 (est.: sampled iterations x4) | 0.42 | 15.1% | 1.0 | 0.51 | `for (int j_1 = 0; j_1 < 4; ++j_1)` |
| load.global*4 (est.: sampled iterations x4) | 0.24 | 8.6% | 1.0 | 0.26 | `if (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 4)) < num_tokens)` |
| load.global#3*4 (est.: sampled iterations x4) | 0.19 | 6.7% | 1.0 | 0.26 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 128) + ((((int)threadIdx.x) & 15` |
| store.global#5*4 (est.: sampled iterations x4) | 0.17 | 6.0% | 1.0 | 0.26 | `*(uint4*)(out + ((((((int64_t)vi_1) * (int64_t)128) + ((((int64_t)((int)threadIdx.x)) & (i` |
| compute.reduce | 0.11 | 4.1% | 1.0 | 0.13 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| wait.sync | 0.05 | 2.0% | 1.0 | 0.10 | `__syncthreads();` |
| store.global#3 | 0.05 | 1.9% | 1.0 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |
| store.global#2 | 0.05 | 1.7% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| load.global#2 | 0.04 | 1.3% | 1.0 | 0.06 | `*(float2*)(row_partial + 0) = *(float2*)(sum_shared + ((((int)threadIdx.x) >> 6) * 2)); \| ` |

### indexer_mxfp4_quantize_kernel grid (48, 1, 1) block (128, 1, 1)

34 launches, 192 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 3.84 us, warp 3.73 us, nsys graphs 3.01 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 2.33 | 62.4% | 2.38 | 62.4% |
| store | 1.13 | 30.3% | 1.16 | 30.4% |
| unattributed | 0.27 | 7.3% | 0.27 | 7.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2*16 (est.: sampled iterations x16) | 1.28 | 34.3% | 1.0 | 1.54 | `float v__2 = __fmul_rn(((float)X[(((((int64_t)((int)blockIdx.x)) * (int64_t)4096) + (((int` |
| store.global*16 (est.: sampled iterations x16) | 1.09 | 29.2% | 1.0 | 1.54 | `Pool[(((((slot & (int64_t)127) * (int64_t)64) + ((((int64_t)((int)threadIdx.x)) & (int64_t` |
| load.global | 1.05 | 28.1% | 1.0 | 1.12 | `for (int i = 0; i < 32; ++i)` |
| store.global#2 | 0.04 | 1.1% | 1.0 | 0.06 | `Pool[(((((slot & (int64_t)127) * (int64_t)4) + ((slot >> (int64_t)7) * ((int64_t)pool_stri` |

### compress_gather_partial_state_kernel grid (8, 1, 1) block (128, 1, 1)

17 launches, 32 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 6.43 us, warp 5.94 us, nsys graphs 3.94 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 2.74 | 46.1% | 4.35 | 58.1% |
| unattributed | 2.33 | 39.3% | 1.20 | 16.1% |
| store | 0.87 | 14.6% | 1.94 | 25.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 2.62 | 44.1% | 1.0 | 2.66 | `int nt = counts[0];` |
| store.global*2 (est.: sampled iterations x2) | 0.81 | 13.7% | 1.0 | 0.96 | `if (half == 0)` |
| load.global#2*2 (est.: sampled iterations x2) | 0.12 | 2.0% | 1.0 | 0.13 | `for (int i = 0; i < 4; ++i)` |
| store.global#2 | 0.06 | 1.0% | 1.0 | 0.10 | `if (((int)threadIdx.x) == 0)` |

### indexer_mxfp4_quantize_kernel grid (2, 1, 1) block (128, 1, 1)

34 launches, 8 warps; ranges, warp and last warp from the 2 recording CTAs' 8 warps; span 3.74 us, warp 1.64 us, nsys graphs 2.58 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.98 | 59.8% | 2.28 | 60.1% |
| unattributed | 0.39 | 23.8% | 0.37 | 9.9% |
| store | 0.27 | 16.4% | 1.14 | 30.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.69 | 42.2% | 0.8 | 1.09 | `for (int i = 0; i < 32; ++i)` |
| load.global#2*16 (est.: sampled iterations x16) | 0.29 | 17.6% | 0.2 | 1.02 | `float v__2 = __fmul_rn(((float)X[(((((int64_t)((int)blockIdx.x)) * (int64_t)4096) + (((int` |
| store.global*16 (est.: sampled iterations x16) | 0.26 | 15.6% | 0.2 | 1.02 | `Pool[(((((slot & (int64_t)127) * (int64_t)64) + ((((int64_t)((int)threadIdx.x)) & (int64_t` |
| store.global#2 | 0.01 | 0.7% | 0.2 | 0.06 | `Pool[(((((slot & (int64_t)127) * (int64_t)4) + ((slot >> (int64_t)7) * ((int64_t)pool_stri` |

### norm_forward_and_per_token_cast_kernel_kernel grid (20, 1, 1) block (128, 1, 1)

51 launches, 80 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 2.37 us, warp 2.19 us, nsys graphs 2.10 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 1.57 | 71.5% | 1.76 | 70.7% |
| load | 0.36 | 16.3% | 0.35 | 13.9% |
| compute | 0.12 | 5.4% | 0.12 | 4.8% |
| wait | 0.11 | 5.0% | 0.06 | 2.6% |
| unattributed | 0.04 | 1.8% | 0.20 | 7.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*3 (est.: sampled iterations x3) | 1.07 | 48.8% | 1.0 | 1.15 | `for (int j_2 = 0; j_2 < 2; ++j_2)` |
| store.global*3 (est.: sampled iterations x3) | 0.29 | 13.2% | 1.0 | 0.38 | `for (int j_1 = 0; j_1 < 4; ++j_1)` |
| load.global*3 (est.: sampled iterations x3) | 0.18 | 8.2% | 1.0 | 0.19 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 6)) < num_tokens) && (((vi * 2) + (` |
| load.global#3*3 (est.: sampled iterations x3) | 0.14 | 6.3% | 1.0 | 0.19 | `if (((vi_1 * 2) + ((((int)threadIdx.x) & 63) >> 5)) < 5)` |
| store.global#5*3 (est.: sampled iterations x3) | 0.14 | 6.3% | 1.0 | 0.19 | `*(uint4*)(out + ((((((int64_t)vi_1) * (int64_t)512) + ((((int64_t)((int)threadIdx.x)) & (i` |
| compute.reduce | 0.12 | 5.4% | 1.0 | 0.13 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| wait.sync | 0.11 | 5.0% | 1.0 | 0.19 | `__syncthreads();` |
| store.global#2 | 0.04 | 2.0% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| load.global#2 | 0.04 | 1.7% | 1.0 | 0.06 | `*(float2*)(row_partial + 0) = *(float2*)(sum_shared + ((((int)threadIdx.x) >> 6) * 2)); \| ` |
| store.global#3 | 0.03 | 1.3% | 0.5 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |

### gemm_bf16_partials_kernel grid (1, 40, 1) block (256, 1, 1)

34 launches, 320 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 3.58 us, warp 2.14 us, nsys graphs 3.23 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.38 | 64.2% | 2.31 | 63.7% |
| wait | 0.33 | 15.4% | 0.58 | 16.0% |
| compute | 0.29 | 13.6% | 0.59 | 16.4% |
| store | 0.07 | 3.5% | 0.14 | 4.0% |
| load | 0.07 | 3.2% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*2 (est.: sampled iterations x2) | 0.29 | 13.6% | 0.5 | 0.64 | `{` |
| wait.mbarrier#2*2 (est.: sampled iterations x2) | 0.17 | 7.9% | 0.5 | 0.61 | `mbarrier[kb_1].wait(0);` |
| wait.sync | 0.10 | 4.9% | 1.0 | 0.13 | `__syncthreads();` |
| store.global | 0.07 | 3.5% | 0.5 | 0.19 | `for (int i_1 = 0; i_1 < 4; ++i_1) \| for (int i_2 = 0; i_2 < 8; ++i_2) \| for (int i_4 = 0; ` |
| load.tma*2 (est.: sampled iterations x2) | 0.07 | 3.2% | 0.5 | 0.19 | `if (tl::tl_shuffle_elect<128>())` |
| wait.mbarrier*2 (est.: sampled iterations x2) | 0.06 | 2.7% | 0.5 | 0.13 | `mbarrier[(kb + 3)].wait(1);` |

### local_stats_kernel grid (48, 16, 1) block (128, 1, 1)

1 launches, 3072 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 119.46 us, warp 48.61 us, nsys graphs 84.42 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 44.17 | 90.9% | 56.51 | 89.1% |
| compute | 2.07 | 4.3% | 3.46 | 5.4% |
| unattributed | 2.03 | 4.2% | 3.30 | 5.2% |
| wait | 0.28 | 0.6% | 0.06 | 0.1% |
| store | 0.07 | 0.1% | 0.10 | 0.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 44.17 | 90.9% | 1.0 | 60.10 | `if (expanded_local_pos[((int64_t)((int)blockIdx.x))] < num_speculative_steps)` |
| compute.reduce | 2.01 | 4.1% | 1.0 | 9.50 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| wait.sync | 0.28 | 0.6% | 1.0 | 1.79 | `__syncthreads();` |
| store.global | 0.07 | 0.1% | 1.0 | 0.10 | `if (((int)threadIdx.x) == 0)` |
| compute.reduce#2 | 0.07 | 0.1% | 1.0 | 0.10 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 2, 32);` |

### metadata_chunk_kernel grid (40, 1, 1) block (192, 1, 1)

51 launches, 240 warps; ranges, warp and last warp from the 4 recording CTAs' 24 warps; span 2.18 us, warp 1.51 us, nsys graphs 2.34 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.09 | 71.9% | 1.71 | 74.2% |
| unattributed | 0.36 | 24.0% | 0.46 | 20.1% |
| store | 0.06 | 4.1% | 0.13 | 5.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.05 | 69.2% | 1.0 | 1.34 | `if ((((offset + ((int)blockIdx.x)) < src_rows) && (0 <= (offset + ((int)blockIdx.x)))))` |
| store.global | 0.05 | 3.2% | 1.0 | 0.06 | `swa[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = con` |
| load.global#2 | 0.04 | 2.7% | 0.2 | 0.26 | `if (((0 <= req) & ((int64_t)0 <= pos)))` |
| store.global#2 | 0.01 | 0.4% | 0.2 | 0.06 | `lengths[((int64_t)((int)blockIdx.x))] = ((int)max(info[1], (int64_t)0));` |
| store.global#3 | 0.01 | 0.4% | 0.2 | 0.06 | `top_lengths[((int64_t)((int)blockIdx.x))] = condval_12; \| top_lengths[((int64_t)((int)bloc` |

### gemm_bf16_partials_kernel grid (2, 4, 1) block (256, 1, 1)

34 launches, 64 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 2.53 us, warp 1.65 us, nsys graphs 2.72 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.96 | 58.2% | 1.43 | 43.1% |
| compute | 0.29 | 17.5% | 0.58 | 17.5% |
| wait | 0.23 | 13.8% | 1.04 | 31.2% |
| store | 0.11 | 6.5% | 0.27 | 8.3% |
| load | 0.07 | 4.0% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*2 (est.: sampled iterations x2) | 0.29 | 17.5% | 0.5 | 0.64 | `{` |
| store.global | 0.11 | 6.5% | 0.5 | 0.26 | `for (int i_1 = 0; i_1 < 4; ++i_1) \| for (int i_2 = 0; i_2 < 8; ++i_2) \| for (int i_4 = 0; ` |
| wait.sync | 0.11 | 6.4% | 1.0 | 0.13 | `__syncthreads();` |
| load.tma*2 (est.: sampled iterations x2) | 0.07 | 4.0% | 0.5 | 0.19 | `if (tl::tl_shuffle_elect<128>())` |
| wait.mbarrier*2 (est.: sampled iterations x2) | 0.06 | 3.8% | 0.5 | 0.13 | `mbarrier[(kb + 3)].wait(1);` |
| wait.mbarrier#2*2 (est.: sampled iterations x2) | 0.06 | 3.6% | 0.5 | 0.13 | `mbarrier[kb_1].wait(0);` |

### resample_kernel grid (8, 127, 1) block (128, 1, 1)

17 launches, 4064 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 6.50 us, warp 2.74 us, nsys graphs 5.70 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.57 | 57.4% | 1.13 | 50.2% |
| unattributed | 0.69 | 25.3% | 0.70 | 31.1% |
| compute | 0.30 | 10.9% | 0.25 | 11.2% |
| wait | 0.10 | 3.5% | 0.08 | 3.5% |
| store | 0.08 | 2.9% | 0.09 | 4.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.57 | 57.4% | 1.0 | 2.37 | `end = cu_num_logits[(((int64_t)((int)blockIdx.x)) + (int64_t)1)];` |
| compute.reduce#2 | 0.15 | 5.3% | 1.0 | 0.19 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 8, 32);` |
| wait.sync | 0.10 | 3.5% | 1.0 | 0.16 | `__syncthreads();` |
| compute.reduce#3 | 0.08 | 3.0% | 1.0 | 0.10 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 2, 32);` |
| store.global | 0.08 | 2.9% | 1.0 | 0.13 | `if (((int)threadIdx.x) == 0)` |
| compute.reduce | 0.07 | 2.6% | 1.0 | 0.10 | `if ((temp != 0x0p+0f/*0.000000e+00*/) \| bonus)` |

### prepare_dflash_inputs_kernel grid (9, 8, 1) block (256, 1, 1)

51 launches, 576 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 1.95 us, warp 1.71 us, nsys graphs 2.37 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.30 | 75.9% | 1.47 | 76.9% |
| unattributed | 0.35 | 20.4% | 0.38 | 20.0% |
| store | 0.06 | 3.7% | 0.06 | 3.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.24 | 72.4% | 1.0 | 1.38 | `req = idx_mapping[((int)blockIdx.y)];` |
| store.global#2 | 0.06 | 3.7% | 1.0 | 0.10 | `if (((int)blockIdx.y) == (num_reqs - 1))` |
| load.global#2 | 0.06 | 3.5% | 1.0 | 0.06 | `vllm_store_if_b32((&(out_query_start_loc[((int)blockIdx.y)])), (((int)blockIdx.y) * num_qu` |

### norm_forward_and_per_token_cast_kernel_kernel grid (3, 1, 1) block (128, 1, 1)

34 launches, 12 warps; ranges, warp and last warp from the 3 recording CTAs' 12 warps; span 2.18 us, warp 2.15 us, nsys graphs 1.89 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 1.13 | 52.6% | 1.49 | 58.5% |
| unattributed | 0.65 | 30.0% | 0.68 | 26.8% |
| load | 0.24 | 11.0% | 0.24 | 9.4% |
| compute | 0.09 | 4.0% | 0.09 | 3.4% |
| wait | 0.05 | 2.4% | 0.05 | 1.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*2 (est.: sampled iterations x2) | 0.74 | 34.3% | 1.0 | 0.77 | `for (int j_2 = 0; j_2 < 2; ++j_2)` |
| store.global*2 (est.: sampled iterations x2) | 0.21 | 9.7% | 1.0 | 0.26 | `for (int j_1 = 0; j_1 < 4; ++j_1)` |
| load.global*2 (est.: sampled iterations x2) | 0.12 | 5.7% | 1.0 | 0.13 | `if (((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < num_tokens)` |
| store.global#5*2 (est.: sampled iterations x2) | 0.09 | 4.2% | 1.0 | 0.13 | `*(uint4*)(out + ((((((int64_t)vi_1) * (int64_t)64) + ((((int64_t)((int)threadIdx.x)) & (in` |
| compute.reduce | 0.09 | 4.0% | 1.0 | 0.10 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| load.global#3*2 (est.: sampled iterations x2) | 0.08 | 3.8% | 1.0 | 0.13 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 64) + ((((int)threadIdx.x) & 7) ` |
| store.global#3 | 0.05 | 2.5% | 1.0 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |
| wait.sync | 0.05 | 2.4% | 1.0 | 0.06 | `__syncthreads();` |
| store.global#2 | 0.04 | 1.9% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| load.global#2 | 0.03 | 1.5% | 1.0 | 0.03 | `*(float2*)(row_partial + 0) = *(float2*)(sum_shared + ((((int)threadIdx.x) >> 6) * 2)); \| ` |

### mhc_collapse_kernel grid (200, 1, 1) block (128, 1, 1)

17 launches, 800 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 4.64 us, warp 2.93 us, nsys graphs 1.92 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 2.62 | 89.2% | 3.32 | 80.5% |
| load | 0.27 | 9.1% | 0.75 | 18.3% |
| store | 0.05 | 1.7% | 0.05 | 1.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.27 | 9.1% | 1.0 | 0.99 | `*(float4*)(mv + 0) = *(float4*)(mix + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int6` |
| store.global | 0.05 | 1.7% | 1.0 | 0.10 | `*(uint4*)(out + ((((int64_t)((int)blockIdx.x)) * (int64_t)1024) + (((int64_t)((int)threadI` |

### activation_swiglu_clamp_kernel grid (1, 40, 1) block (128, 1, 1)

51 launches, 160 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 0.96 us, warp 0.76 us, nsys graphs 1.58 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.65 | 85.8% | 1.25 | 83.2% |
| load | 0.07 | 9.5% | 0.19 | 12.4% |
| store | 0.04 | 4.7% | 0.07 | 4.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.07 | 9.5% | 0.8 | 0.10 | `*(uint4*)(gate + 0) = *(uint4*)(X + ((((int64_t)((int)blockIdx.y)) * (int64_t)1536) + (((i` |
| store.global | 0.04 | 4.7% | 0.8 | 0.06 | `*(uint4*)(Y + ((((int64_t)((int)blockIdx.y)) * (int64_t)768) + (((int64_t)((int)threadIdx.` |

### topk_log_softmax_kernel grid (48, 1, 1) block (1024, 1, 1)

1 launches, 1536 warps; ranges, warp and last warp from the 4 recording CTAs' 128 warps; span 75.23 us, warp 71.29 us, nsys graphs 70.05 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 61.66 | 86.5% | 70.85 | 94.7% |
| wait | 7.68 | 10.8% | 1.73 | 2.3% |
| unattributed, with loops' unsampled iterations | 1.58 | 2.2% | 1.79 | 2.4% |
| compute | 0.37 | 0.5% | 0.35 | 0.5% |
| store | 0.00 | 0.0% | 0.06 | 0.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 40.14 | 56.3% | 1.0 | 47.71 | `if (0 < (vocab & 2047)) \| if (0 < (vocab & 8191))` |
| load.global#2 | 21.47 | 30.1% | 1.0 | 25.54 | `for (int step_1 = 0; step_1 < (vocab >> 11); ++step_1) \| for (int step_1 = 0; step_1 < (vo` |
| wait.sync | 5.04 | 7.1% | 1.0 | 19.33 | `__syncthreads();` |
| wait.sync#2 | 2.38 | 3.3% | 1.0 | 8.90 | `__syncthreads();` |
| compute.reduce#3 | 0.18 | 0.2% | 1.0 | 0.19 | `ties[0] = vllm_add(ties[0], __shfl_xor_sync((uint)4294967295, ties[0], 16, 32));` |
| compute.reduce#2 | 0.10 | 0.1% | 1.0 | 0.13 | `se[0] = vllm_add(se[0], __shfl_xor_sync((uint)4294967295, se[0], 16, 32));` |
| wait.sync#4 | 0.10 | 0.1% | 1.0 | 0.16 | `__syncthreads();` |
| compute.reduce | 0.10 | 0.1% | 1.0 | 0.13 | `best[0] = vllm_max(best[0], __shfl_xor_sync((uint)4294967295, best[0], 16, 32));` |
| wait.sync#5 | 0.08 | 0.1% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#3 | 0.08 | 0.1% | 1.0 | 0.10 | `__syncthreads();` |
| load.global#3 (sampled iterations) | 0.04 | 0.1% | 0.0 | 3.65 | `float value_32 = vllm_sub(vllm_shifted(logits[((((int64_t)((int)blockIdx.x)) * logits_stri` |
| store.global (sampled iterations) | 0.00 | 0.0% | 0.0 | 0.10 | `output[(((((int64_t)c) * (int64_t)1024) + (((int64_t)((int)blockIdx.x)) * ((int64_t)topk))` |

### rope_kernel grid (40, 1, 1) block (192, 1, 1)

51 launches, 240 warps; ranges, warp and last warp from the 4 recording CTAs' 24 warps; span 1.25 us, warp 1.20 us, nsys graphs 1.54 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.68 | 56.5% | 0.83 | 60.9% |
| load | 0.48 | 40.1% | 0.49 | 36.0% |
| store | 0.04 | 3.3% | 0.04 | 3.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.48 | 40.1% | 1.0 | 0.48 | `if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < tokens)) \| int64_t position ` |
| store.global | 0.04 | 3.3% | 1.0 | 0.06 | `*(uint4*)(x + ((((((int64_t)((int)blockIdx.x)) * (int64_t)12288) + ((((int64_t)((int)threa` |

### sink_merge_kernel grid (8, 1, 1) block (128, 1, 1)

17 launches, 32 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 4.10 us, warp 4.05 us, nsys graphs 4.42 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.92 | 47.5% | 2.06 | 50.4% |
| wait | 0.82 | 20.3% | 0.47 | 11.6% |
| unattributed | 0.71 | 17.5% | 0.84 | 20.5% |
| compute | 0.56 | 13.8% | 0.57 | 14.0% |
| store | 0.04 | 0.9% | 0.14 | 3.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2*3 (est.: sampled iterations x3) | 1.64 | 40.6% | 1.0 | 1.73 | `for (int step = 0; step < ((width + 127) >> 7); ++step) \| for (int step = 0; step < ((widt` |
| compute.reduce*3 (est.: sampled iterations x3) | 0.56 | 13.8% | 1.0 | 0.58 | `float other = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| wait.sync#3*3 (est.: sampled iterations x3) | 0.52 | 12.9% | 1.0 | 0.67 | `__syncthreads();` |
| load.global | 0.23 | 5.7% | 1.0 | 0.26 | `last = sampled[(max((((int64_t)num_sampled[((int64_t)((int)blockIdx.x))]) - (int64_t)1), (` |
| wait.sync*3 (est.: sampled iterations x3) | 0.17 | 4.3% | 1.0 | 0.19 | `__syncthreads();` |
| wait.sync#2*3 (est.: sampled iterations x3) | 0.13 | 3.1% | 1.0 | 0.19 | `__syncthreads();` |
| load.global#3*3 (est.: sampled iterations x3) | 0.05 | 1.2% | 0.2 | 0.19 | `if ((best[0] == -CUDART_INF_F))` |
| store.global*3 (est.: sampled iterations x3) | 0.04 | 0.9% | 0.2 | 0.19 | `alternates[((((int64_t)((int)blockIdx.x)) * (int64_t)3) + ((int64_t)rank))] = condval_31;` |

### finish_argmax_kernel grid (8, 1, 1) block (32, 1, 1)

85 launches, 8 warps; ranges, warp and last warp from the 4 recording CTAs' 4 warps; span 0.77 us, warp 0.75 us, nsys graphs 1.06 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.41 | 54.3% | 0.46 | 56.6% |
| load | 0.13 | 17.0% | 0.13 | 15.8% |
| compute | 0.12 | 16.0% | 0.12 | 14.9% |
| store | 0.10 | 12.8% | 0.10 | 12.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.13 | 17.0% | 1.0 | 0.13 | `for (int k = 0; k < 2; ++k) \| if ((((int)threadIdx.x) < parts))` |
| compute.reduce | 0.12 | 16.0% | 1.0 | 0.13 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| store.global | 0.10 | 12.8% | 1.0 | 0.10 | `if (((int)threadIdx.x) == 0)` |

### context_inputs_kernel grid (5, 16, 1) block (128, 1, 1)

16 launches, 320 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 4.00 us, warp 2.45 us, nsys graphs 3.90 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 2.06 | 84.1% | 3.30 | 84.0% |
| unattributed | 0.32 | 13.2% | 0.42 | 10.7% |
| store | 0.07 | 2.7% | 0.21 | 5.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 2.06 | 84.1% | 1.0 | 3.34 | `vllm_dspark_context_b128((&(aux[(((((int64_t)((int)blockIdx.y)) * (int64_t)15360) + (((int` |
| store.global | 0.07 | 2.7% | 1.0 | 0.10 | `if (((((int)blockIdx.x) == 0) & (((int)threadIdx.x) == 0)) & (num_tokens <= ((int)blockIdx` |

### per_token_cast_kernel_kernel grid (25, 1, 1) block (64, 1, 1)

51 launches, 50 warps; ranges, warp and last warp from the 5 recording CTAs' 10 warps; span 1.15 us, warp 1.09 us, nsys graphs 1.41 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.35 | 32.2% | 0.49 | 39.3% |
| unattributed | 0.35 | 32.2% | 0.36 | 28.4% |
| store | 0.29 | 26.9% | 0.31 | 24.9% |
| compute | 0.10 | 8.8% | 0.09 | 7.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.26 | 24.0% | 1.0 | 0.29 | `for (int i = 0; i < 2; ++i)` |
| store.global#2 | 0.12 | 10.5% | 1.0 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.10 | 8.8% | 1.0 | 0.13 | `amax_fragment[i_1] = tl::AllReduce<tl::MaxOp, 2, 1, 0, tl::NamedBarrier<64>>::run(amax_fra` |
| load.global#2*2 (est.: sampled iterations x2) | 0.09 | 8.2% | 1.0 | 0.13 | `amax_fragment[i_1] = cutlass::bfloat16_t(__hmax((bfloat16_t(((nv_bfloat162*)(&(amax_fragme` |
| store.global#3*2 (est.: sampled iterations x2) | 0.09 | 8.2% | 1.0 | 0.13 | `*(fp8_e4_16_t*)(out + ((((((((int64_t)((int)blockIdx.x)) / (int64_t)3) * (int64_t)6144) + ` |
| store.global*2 (est.: sampled iterations x2) | 0.09 | 8.2% | 1.0 | 0.13 | `out_sf[((((((int64_t)((int)blockIdx.x)) % (int64_t)15) * (int64_t)32) + (((int64_t)((int)t` |

### norm_forward_and_per_token_cast_kernel_kernel grid (40, 1, 1) block (128, 1, 1)

17 launches, 160 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 3.65 us, warp 3.53 us, nsys graphs 2.42 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 2.35 | 66.7% | 3.11 | 82.5% |
| unattributed | 0.50 | 14.1% | 0.00 | 0.0% |
| load | 0.44 | 12.5% | 0.42 | 11.2% |
| compute | 0.13 | 3.7% | 0.13 | 3.4% |
| wait | 0.11 | 3.0% | 0.21 | 5.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*5 (est.: sampled iterations x5) | 2.08 | 59.0% | 1.0 | 2.72 | `for (int j_1 = 0; j_1 < 2; ++j_1)` |
| load.global#2*5 (est.: sampled iterations x5) | 0.22 | 6.2% | 1.0 | 0.32 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 1024) + (((int)threadIdx.x) * 8)` |
| load.global*5 (est.: sampled iterations x5) | 0.22 | 6.2% | 1.0 | 0.32 | `*(uint4*)(x_local + (vi * 8)) = *(uint4*)(x + ((((((int64_t)vi) * (int64_t)1024) + (((int6` |
| store.global*5 (est.: sampled iterations x5) | 0.22 | 6.2% | 1.0 | 0.32 | `for (int j = 0; j < 4; ++j)` |
| compute.reduce | 0.13 | 3.7% | 1.0 | 0.16 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| wait.sync | 0.11 | 3.0% | 1.0 | 0.29 | `__syncthreads();` |
| store.global#2 | 0.04 | 1.2% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| store.global#3 | 0.01 | 0.3% | 0.2 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |

### gemm_splitk_reduce_kernel grid (1, 48, 1) block (128, 1, 1)

68 launches, 192 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 0.90 us, warp 0.85 us, nsys graphs 1.06 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.41 | 48.1% | 0.45 | 52.3% |
| load | 0.39 | 46.2% | 0.36 | 42.0% |
| store | 0.05 | 5.7% | 0.05 | 5.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.39 | 46.2% | 1.0 | 0.42 | `acc[0] = P[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x))` |
| store.global | 0.05 | 5.7% | 1.0 | 0.06 | `C[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x)) * (int64` |

### dspark_confidence_kernel grid (40, 1, 1) block (128, 1, 1)

17 launches, 160 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 3.55 us, warp 3.14 us, nsys graphs 3.50 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 2.66 | 84.6% | 2.98 | 83.4% |
| wait | 0.19 | 6.0% | 0.20 | 5.6% |
| load | 0.18 | 5.6% | 0.21 | 5.8% |
| compute | 0.10 | 3.1% | 0.09 | 2.6% |
| store | 0.02 | 0.7% | 0.09 | 2.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync | 0.19 | 6.0% | 1.0 | 0.42 | `__syncthreads();` |
| load.global | 0.16 | 5.2% | 1.0 | 0.19 | `for (int c = 0; c < 5; ++c)` |
| compute.reduce | 0.10 | 3.1% | 1.0 | 0.10 | `acc[0] = tl::warp_reduce_sum(acc[0]);` |
| store.global | 0.02 | 0.7% | 0.2 | 0.10 | `out[((int64_t)((int)blockIdx.x))] = (0x1p+0f/*1.000000e+00*/ / (0x1p+0f/*1.000000e+00*/ + ` |
| load.global#2 | 0.01 | 0.4% | 0.2 | 0.06 | `total[0] = 0x0p+0f/*0.000000e+00*/;` |

### cache_writer_kernel grid (40, 1, 1) block (128, 1, 1)

51 launches, 160 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 1.06 us, warp 1.03 us, nsys graphs 1.31 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.37 | 35.9% | 0.45 | 40.6% |
| load | 0.29 | 28.3% | 0.31 | 27.7% |
| store | 0.29 | 28.3% | 0.28 | 24.7% |
| compute | 0.08 | 7.6% | 0.08 | 7.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.29 | 28.3% | 1.0 | 0.32 | `if (((int64_t)0 <= slot) & ((slot >> (int64_t)7) < ((int64_t)pages))) \| if (((int64_t)0 <=` |
| load.global | 0.23 | 22.5% | 1.0 | 0.26 | `for (int i = 0; i < 4; ++i)` |
| compute.reduce | 0.08 | 7.6% | 1.0 | 0.10 | `amax = max(amax, __shfl_xor_sync((uint)4294967295, amax, 1, 32));` |
| load.global#2 | 0.06 | 5.8% | 1.0 | 0.06 | `int64_t slot = slots[((int64_t)((int)blockIdx.x))];` |

### mhc_collapse_column_kernel grid (240, 1, 1) block (128, 1, 1)

17 launches, 960 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 3.36 us, warp 1.60 us, nsys graphs 2.14 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.29 | 80.5% | 1.43 | 61.8% |
| load | 0.23 | 14.4% | 0.82 | 35.6% |
| store | 0.08 | 5.1% | 0.06 | 2.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.23 | 14.4% | 1.0 | 0.67 | `*(float4*)(mv + 0) = *(float4*)(mix + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int6` |
| store.global | 0.08 | 5.1% | 1.0 | 0.35 | `*(uint4*)(out + ((((int64_t)((int)blockIdx.x)) * (int64_t)1024) + (((int64_t)((int)threadI` |

### token_ranks_kernel grid (48, 1, 1) block (256, 1, 1)

1 launches, 384 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 58.08 us, warp 39.66 us, nsys graphs 48.29 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 33.12 | 83.5% | 55.46 | 99.0% |
| compute | 6.12 | 15.4% | 0.22 | 0.4% |
| unattributed | 0.36 | 0.9% | 0.32 | 0.6% |
| store | 0.05 | 0.1% | 0.03 | 0.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 33.12 | 83.5% | 1.0 | 55.46 | `if ((((int64_t)0 <= token_ids[((int64_t)((int)blockIdx.x))]) && (token_ids[((int64_t)((int` |
| compute.reduce | 6.12 | 15.4% | 1.0 | 21.22 | `total[0] = tl::AllReduce<tl::SumOp, 256, 1, 0, tl::NamedBarrier<256>>::run(total[0], (&(wo` |
| store.global | 0.05 | 0.1% | 1.0 | 0.06 | `output[((int64_t)((int)blockIdx.x))] = ((int64_t)total[0]);` |

### per_token_cast_kernel_kernel grid (15, 1, 1) block (64, 1, 1)

51 launches, 30 warps; ranges, warp and last warp from the 5 recording CTAs' 10 warps; span 1.12 us, warp 1.08 us, nsys graphs 1.34 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.36 | 32.8% | 0.37 | 33.1% |
| unattributed | 0.33 | 30.5% | 0.34 | 30.3% |
| store | 0.29 | 27.2% | 0.31 | 27.4% |
| compute | 0.10 | 9.5% | 0.10 | 9.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.27 | 24.6% | 1.0 | 0.29 | `for (int i = 0; i < 2; ++i)` |
| store.global#2 | 0.11 | 10.1% | 1.0 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.10 | 9.5% | 1.0 | 0.13 | `amax_fragment[i_1] = tl::AllReduce<tl::MaxOp, 2, 1, 0, tl::NamedBarrier<64>>::run(amax_fra` |
| store.global*2 (est.: sampled iterations x2) | 0.10 | 9.5% | 1.0 | 0.13 | `out_sf[((((((int64_t)((int)blockIdx.x)) % (int64_t)15) * (int64_t)32) + (((int64_t)((int)t` |
| load.global#2*2 (est.: sampled iterations x2) | 0.09 | 8.3% | 1.0 | 0.13 | `amax_fragment[i_1] = cutlass::bfloat16_t(__hmax((bfloat16_t(((nv_bfloat162*)(&(amax_fragme` |
| store.global#3*2 (est.: sampled iterations x2) | 0.08 | 7.7% | 1.0 | 0.13 | `*(fp8_e4_16_t*)(out + ((((((((int64_t)((int)blockIdx.x)) / (int64_t)3) * (int64_t)6144) + ` |

### adaptive_offsets_kernel grid (1, 1, 1) block (256, 1, 1)

16 launches, 8 warps; ranges, warp and last warp from the 1 recording CTAs' 8 warps; span 3.52 us, warp 3.50 us, nsys graphs 1.87 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 1.57 | 44.9% | 1.11 | 31.7% |
| unattributed, with loops' unsampled iterations | 1.44 | 41.3% | 1.53 | 43.5% |
| store | 0.31 | 8.8% | 0.49 | 14.0% |
| load | 0.17 | 5.0% | 0.38 | 10.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync#18 (sampled iterations) | 0.34 | 9.8% | 1.0 | 0.38 | `__syncthreads();` |
| wait.sync (sampled iterations) | 0.20 | 5.8% | 1.0 | 0.22 | `__syncthreads();` |
| store.global#5 | 0.12 | 3.4% | 1.0 | 0.13 | `for (int chunk_1 = 0; chunk_1 < ((starts_len + 255) >> 8); ++chunk_1)` |
| load.global#2 (sampled iterations) | 0.10 | 2.7% | 1.0 | 0.22 | `if ((((chunk * 256) + ((int)threadIdx.x)) < num_reqs))` |
| store.global#2 (sampled iterations) | 0.09 | 2.6% | 1.0 | 0.10 | `if ((((chunk * 256) + ((int)threadIdx.x)) < num_reqs) & (drafted == 0))` |
| store.global | 0.07 | 2.1% | 1.0 | 0.10 | `if (((int)threadIdx.x) == 0)` |
| load.global (sampled iterations) | 0.07 | 2.0% | 1.0 | 0.10 | `if (((((chunk * 256) + ((int)threadIdx.x)) < num_reqs) & (drafted != 0)))` |
| wait.sync#10 (sampled iterations) | 0.06 | 1.8% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#11 (sampled iterations) | 0.06 | 1.8% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#12 (sampled iterations) | 0.06 | 1.8% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#13 (sampled iterations) | 0.06 | 1.8% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#14 (sampled iterations) | 0.06 | 1.8% | 1.0 | 0.06 | `__syncthreads();` |
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
| store.global#4 (sampled iterations) | 0.02 | 0.5% | 0.1 | 0.13 | `for (int j = 0; j < 6; ++j)` |
| load.global#3 (sampled iterations) | 0.01 | 0.2% | 0.1 | 0.06 | `state = idx_mapping[((((int64_t)chunk) * (int64_t)256) + ((int64_t)((int)threadIdx.x)))];` |
| store.global#3 (sampled iterations) | 0.01 | 0.2% | 0.1 | 0.06 | `cu_num_logits[(((((int64_t)chunk) * (int64_t)256) + ((int64_t)((int)threadIdx.x))) + (int6` |

### per_token_cast_kernel_kernel grid (40, 1, 1) block (64, 1, 1)

48 launches, 80 warps; ranges, warp and last warp from the 4 recording CTAs' 8 warps; span 1.15 us, warp 1.09 us, nsys graphs 0.99 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.37 | 34.2% | 0.39 | 34.9% |
| unattributed | 0.32 | 29.8% | 0.33 | 29.6% |
| store | 0.30 | 27.2% | 0.30 | 27.0% |
| compute | 0.10 | 8.8% | 0.10 | 8.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.28 | 26.1% | 1.0 | 0.32 | `for (int i = 0; i < 2; ++i)` |
| store.global#2 | 0.11 | 10.3% | 1.0 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.10 | 8.8% | 1.0 | 0.13 | `amax_fragment[i_1] = tl::AllReduce<tl::MaxOp, 2, 1, 0, tl::NamedBarrier<64>>::run(amax_fra` |
| store.global*2 (est.: sampled iterations x2) | 0.10 | 8.8% | 1.0 | 0.13 | `out_sf[((((((int64_t)((int)blockIdx.x)) % (int64_t)15) * (int64_t)32) + (((int64_t)((int)t` |
| load.global#2*2 (est.: sampled iterations x2) | 0.09 | 8.1% | 1.0 | 0.13 | `amax_fragment[i_1] = cutlass::bfloat16_t(__hmax((bfloat16_t(((nv_bfloat162*)(&(amax_fragme` |
| store.global#3*2 (est.: sampled iterations x2) | 0.09 | 8.1% | 1.0 | 0.13 | `*(fp8_e4_16_t*)(out + ((((((((int64_t)((int)blockIdx.x)) / (int64_t)3) * (int64_t)6144) + ` |

### cache_writer_kernel grid (16, 1, 1) block (128, 1, 1)

48 launches, 64 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 0.99 us, warp 0.98 us, nsys graphs 1.12 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.32 | 33.1% | 0.33 | 32.1% |
| store | 0.29 | 29.5% | 0.30 | 28.5% |
| load | 0.29 | 29.4% | 0.33 | 31.8% |
| compute | 0.08 | 8.0% | 0.08 | 7.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.29 | 29.5% | 1.0 | 0.32 | `if (((int64_t)0 <= slot) & ((slot >> (int64_t)7) < ((int64_t)pages))) \| if (((int64_t)0 <=` |
| load.global | 0.23 | 23.3% | 1.0 | 0.26 | `for (int i = 0; i < 4; ++i)` |
| compute.reduce | 0.08 | 8.0% | 1.0 | 0.10 | `amax = max(amax, __shfl_xor_sync((uint)4294967295, amax, 1, 32));` |
| load.global#2 | 0.06 | 6.1% | 1.0 | 0.06 | `int64_t slot = slots[((int64_t)((int)blockIdx.x))];` |

### norm_forward_and_per_token_cast_kernel_kernel grid (48, 1, 1) block (128, 1, 1)

18 launches, 192 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 2.50 us, warp 2.38 us, nsys graphs 2.30 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 1.76 | 73.8% | 2.98 | 111.7% |
| load | 0.42 | 17.4% | 0.43 | 16.0% |
| compute | 0.13 | 5.4% | 0.13 | 4.9% |
| wait | 0.06 | 2.4% | 0.06 | 2.4% |
| unattributed | 0.02 | 1.0% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*5 (est.: sampled iterations x5) | 1.49 | 62.5% | 1.0 | 1.76 | `for (int j_1 = 0; j_1 < 2; ++j_1)` |
| load.global#2*5 (est.: sampled iterations x5) | 0.23 | 9.4% | 1.0 | 0.32 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 1024) + (((int)threadIdx.x) * 8)` |
| store.global*5 (est.: sampled iterations x5) | 0.21 | 9.0% | 1.0 | 0.32 | `for (int j = 0; j < 4; ++j)` |
| load.global*5 (est.: sampled iterations x5) | 0.19 | 8.0% | 1.0 | 0.32 | `*(uint4*)(x_local + (vi * 8)) = *(uint4*)(x + ((((((int64_t)vi) * (int64_t)1024) + (((int6` |
| compute.reduce | 0.13 | 5.4% | 1.0 | 0.16 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| wait.sync | 0.06 | 2.4% | 1.0 | 0.10 | `__syncthreads();` |
| store.global#2 | 0.04 | 1.8% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| store.global#3 | 0.01 | 0.5% | 0.2 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |

### sink_cut_kernel grid (1, 1, 1) block (128, 1, 1)

34 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 1.36 us, warp 0.60 us, nsys graphs 2.11 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.32 | 52.3% | 0.30 | 22.6% |
| store | 0.17 | 27.8% | 0.54 | 40.2% |
| load | 0.12 | 19.9% | 0.50 | 37.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.17 | 27.8% | 0.8 | 0.54 | `if (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) < num_cut) \| saved[((((int64_t)((int)` |
| load.global | 0.12 | 19.9% | 0.1 | 0.48 | `for (int b = 0; b < num_reqs; ++b)` |

### verify_kernel grid (8, 1, 1) block (96, 1, 1)

17 launches, 24 warps; ranges, warp and last warp from the 4 recording CTAs' 12 warps; span 2.59 us, warp 2.51 us, nsys graphs 2.30 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.06 | 42.2% | 1.07 | 40.8% |
| unattributed | 0.91 | 36.2% | 0.88 | 33.5% |
| compute | 0.30 | 12.1% | 0.30 | 11.5% |
| wait | 0.15 | 5.9% | 0.21 | 8.1% |
| store | 0.09 | 3.6% | 0.16 | 6.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.99 | 39.7% | 1.0 | 1.02 | `start = cu_num_logits[((int64_t)((int)blockIdx.x))];` |
| compute.reduce | 0.30 | 12.1% | 1.0 | 0.32 | `row_max[0] = vllm_max(row_max[0], __shfl_xor_sync((uint)4294967295, row_max[0], 8, 32));` |
| store.global | 0.09 | 3.6% | 1.0 | 0.16 | `if (((int)threadIdx.x) == 0)` |
| load.global#2 | 0.06 | 2.6% | 1.0 | 0.06 | `target = local_argmax[((((int64_t)row) * argmax_stride) + ((int64_t)best_index[0]))];` |
| wait.sync#2 | 0.06 | 2.6% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#3 | 0.04 | 1.7% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync | 0.04 | 1.6% | 1.0 | 0.06 | `__syncthreads();` |

### metadata_pages_kernel grid (48, 1, 1) block (128, 1, 1)

34 launches, 192 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 1.28 us, warp 0.83 us, nsys graphs 0.00 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.66 | 78.9% | 0.74 | 58.5% |
| load | 0.14 | 16.3% | 0.20 | 15.6% |
| store | 0.04 | 4.8% | 0.33 | 26.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.13 | 15.4% | 1.0 | 0.13 | `if ((((offset + ((int)blockIdx.x)) < src_rows) && (0 <= (offset + ((int)blockIdx.x)))))` |
| store.global | 0.04 | 4.8% | 0.1 | 0.32 | `if (((int)threadIdx.x) < 32)` |
| load.global#2 | 0.01 | 1.0% | 0.1 | 0.06 | `if (((req < table_rows) && (((int)threadIdx.x) < table_cols)))` |

### sink_policy_kernel grid (8, 1, 1) block (32, 1, 1)

17 launches, 8 warps; ranges, warp and last warp from the 4 recording CTAs' 4 warps; span 2.40 us, warp 2.38 us, nsys graphs 2.74 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.95 | 40.1% | 1.04 | 41.4% |
| unattributed | 0.90 | 38.0% | 0.90 | 36.0% |
| store | 0.52 | 21.9% | 0.56 | 22.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 | 0.57 | 23.9% | 1.0 | 0.58 | `state[0] = idx_mapping[((int64_t)((int)blockIdx.x))];` |
| store.global#3 | 0.46 | 19.5% | 1.0 | 0.48 | `for (int a = 0; a < 3; ++a) \| recovered_logprobs[((int64_t)((int)blockIdx.x))] = 0x0p+0f/*` |
| load.global | 0.21 | 8.8% | 1.0 | 0.22 | `start[0] = cu_num_logits[((int64_t)((int)blockIdx.x))];` |
| load.global#4 | 0.13 | 5.4% | 1.0 | 0.13 | `for (int r = 0; r < 2; ++r) \| for (int r_2 = 0; r_2 < 8; ++r_2)` |
| store.global#4 | 0.06 | 2.4% | 1.0 | 0.06 | `events[(((int64_t)((int)blockIdx.x)) * (int64_t)6)] = outcome[0];` |
| load.global#5 | 0.05 | 2.0% | 1.0 | 0.06 | `AtomicAdd((&(counts[0])), draft_zero_flag[0]);` |

### norm_forward_and_per_token_cast_kernel_kernel grid (16, 1, 1) block (128, 1, 1)

16 launches, 64 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 2.37 us, warp 2.29 us, nsys graphs 1.89 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 1.94 | 85.0% | 3.34 | 127.3% |
| load | 0.43 | 18.8% | 0.41 | 15.6% |
| compute | 0.13 | 5.7% | 0.13 | 5.0% |
| wait | 0.06 | 2.5% | 0.06 | 2.3% |
| unattributed | 0.00 | 0.0% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*5 (est.: sampled iterations x5) | 1.67 | 73.2% | 1.0 | 2.00 | `for (int j_1 = 0; j_1 < 2; ++j_1)` |
| load.global#2*5 (est.: sampled iterations x5) | 0.22 | 9.6% | 1.0 | 0.32 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 1024) + (((int)threadIdx.x) * 8)` |
| store.global*5 (est.: sampled iterations x5) | 0.21 | 9.4% | 1.0 | 0.32 | `for (int j = 0; j < 4; ++j)` |
| load.global*5 (est.: sampled iterations x5) | 0.21 | 9.2% | 1.0 | 0.32 | `*(uint4*)(x_local + (vi * 8)) = *(uint4*)(x + ((((((int64_t)vi) * (int64_t)1024) + (((int6` |
| compute.reduce | 0.13 | 5.7% | 1.0 | 0.16 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| wait.sync | 0.06 | 2.5% | 1.0 | 0.10 | `__syncthreads();` |
| store.global#2 | 0.04 | 1.8% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| store.global#3 | 0.01 | 0.5% | 0.2 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |

### rope_kernel grid (1, 1, 1) block (128, 1, 1)

48 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 0.67 us, warp 0.67 us, nsys graphs 1.06 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.42 | 61.9% | 0.53 | 62.0% |
| load | 0.22 | 33.3% | 0.29 | 33.3% |
| store | 0.03 | 4.8% | 0.04 | 4.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 | 0.13 | 19.0% | 1.0 | 0.13 | `if (((position < ((int64_t)positions_max)) && ((int64_t)0 <= position)))` |
| load.global | 0.10 | 14.3% | 1.0 | 0.10 | `if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < tokens)) \| int64_t position ` |
| store.global | 0.03 | 4.8% | 1.0 | 0.03 | `*(uint4*)(x + ((((((int64_t)((int)blockIdx.x)) * (int64_t)12288) + ((((int64_t)((int)threa` |

### post_update_kernel grid (1, 1, 1) block (32, 1, 1)

17 launches, 1 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 1.92 us, warp 1.92 us, nsys graphs 2.53 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 0.67 | 35.0% | 1.07 | 46.4% |
| load | 0.64 | 33.3% | 0.67 | 29.1% |
| unattributed | 0.61 | 31.7% | 0.56 | 24.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#2 | 0.51 | 26.7% | 1.0 | 0.51 | `if (0 < count)` |
| load.global#2 | 0.48 | 25.0% | 1.0 | 0.48 | `int count = num_sampled[((((int64_t)((int)blockIdx.x)) * (int64_t)32) + ((int64_t)((int)th` |
| store.global | 0.16 | 8.3% | 1.0 | 0.16 | `if (0 < count)` |
| load.global#3 | 0.10 | 5.0% | 1.0 | 0.10 | `if ((0 < count))` |
| load.global | 0.06 | 3.3% | 1.0 | 0.06 | `int64_t req = idx_mapping[((((int64_t)((int)blockIdx.x)) * (int64_t)32) + ((int64_t)((int)` |

### compute_slot_mappings_kernel grid (1, 11, 1) block (256, 1, 1)

17 launches, 88 warps; ranges, warp and last warp from the 6 recording CTAs' 48 warps; span 2.27 us, warp 1.43 us, nsys graphs 2.72 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.80 | 56.1% | 0.80 | 35.1% |
| unattributed | 0.63 | 43.9% | 1.47 | 64.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.80 | 56.1% | 1.0 | 0.83 | `int first_token = query_start_loc[0];` |

### gather_block_tables_kernel grid (11, 8, 1) block (256, 1, 1)

17 launches, 704 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 2.27 us, warp 1.40 us, nsys graphs 0.00 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.09 | 78.0% | 1.65 | 83.8% |
| unattributed | 0.31 | 22.0% | 0.32 | 16.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.09 | 78.0% | 1.0 | 1.73 | `int g = chunk_groups[((int64_t)((int)blockIdx.x))];` |

### per_token_cast_kernel_kernel grid (120, 1, 1) block (64, 1, 1)

19 launches, 240 warps; ranges, warp and last warp from the 4 recording CTAs' 8 warps; span 1.79 us, warp 1.60 us, nsys graphs 1.38 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.86 | 53.5% | 0.97 | 52.1% |
| unattributed | 0.32 | 20.0% | 0.15 | 8.3% |
| store | 0.32 | 20.0% | 0.53 | 28.1% |
| compute | 0.10 | 6.5% | 0.22 | 11.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.76 | 47.5% | 1.0 | 0.90 | `for (int i = 0; i < 2; ++i)` |
| store.global#2 | 0.11 | 7.0% | 1.0 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3)` |
| store.global*2 (est.: sampled iterations x2) | 0.11 | 7.0% | 1.0 | 0.13 | `out_sf[((((((int64_t)((int)blockIdx.x)) % (int64_t)15) * (int64_t)32) + (((int64_t)((int)t` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.10 | 6.5% | 1.0 | 0.13 | `amax_fragment[i_1] = tl::AllReduce<tl::MaxOp, 2, 1, 0, tl::NamedBarrier<64>>::run(amax_fra` |
| load.global#2*2 (est.: sampled iterations x2) | 0.10 | 6.0% | 1.0 | 0.13 | `amax_fragment[i_1] = cutlass::bfloat16_t(__hmax((bfloat16_t(((nv_bfloat162*)(&(amax_fragme` |
| store.global#3*2 (est.: sampled iterations x2) | 0.10 | 6.0% | 1.0 | 0.13 | `*(fp8_e4_16_t*)(out + ((((((((int64_t)((int)blockIdx.x)) / (int64_t)3) * (int64_t)6144) + ` |

### rope_kernel grid (48, 1, 1) block (256, 1, 1)

34 launches, 384 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 1.01 us, warp 0.89 us, nsys graphs 1.33 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.59 | 66.6% | 0.70 | 70.0% |
| load | 0.26 | 28.9% | 0.26 | 25.8% |
| store | 0.04 | 4.5% | 0.04 | 4.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.26 | 28.9% | 1.0 | 0.29 | `if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < tokens)) \| int64_t position ` |
| store.global | 0.04 | 4.5% | 1.0 | 0.06 | `*(uint4*)(x + ((((((int64_t)((int)blockIdx.x)) * (int64_t)12288) + ((((int64_t)((int)threa` |

### sink_flags_kernel grid (1, 1, 1) block (128, 1, 1)

17 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 1.98 us, warp 0.66 us, nsys graphs 2.85 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.33 | 49.4% | 1.30 | 66.0% |
| unattributed | 0.25 | 37.3% | 0.32 | 16.3% |
| store | 0.09 | 13.3% | 0.35 | 17.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.33 | 49.4% | 0.2 | 1.31 | `token = logits_indices[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)th` |
| store.global | 0.09 | 13.3% | 0.2 | 0.35 | `flags[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = (` |

### insert_kernel grid (8, 1, 1) block (32, 1, 1)

17 launches, 8 warps; ranges, warp and last warp from the 4 recording CTAs' 4 warps; span 1.73 us, warp 1.70 us, nsys graphs 1.73 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.70 | 41.3% | 0.78 | 41.6% |
| unattributed | 0.50 | 29.1% | 0.49 | 26.1% |
| store | 0.34 | 19.7% | 0.45 | 23.7% |
| compute | 0.17 | 9.9% | 0.16 | 8.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#3 | 0.53 | 31.0% | 1.0 | 0.54 | `int state = mapping[((int64_t)position)]; \| int64_t state = mapping[((int64_t)position)];` |
| store.global | 0.29 | 16.9% | 1.0 | 0.29 | `counts[((int64_t)((int)blockIdx.x))] = (count_local + 1); \| counts[((int64_t)((int)blockId` |
| compute.reduce#2 | 0.12 | 7.0% | 1.0 | 0.13 | `float value_1 = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| load.global | 0.12 | 7.0% | 1.0 | 0.13 | `for (int k = 0; k < 4; ++k)` |
| load.global#2 | 0.06 | 3.3% | 1.0 | 0.06 | `count_local = counts[((int64_t)((int)blockIdx.x))];` |
| compute.reduce | 0.05 | 2.8% | 1.0 | 0.06 | `int count = __shfl_sync((uint)4294967295, count_local, 0, 32);` |
| store.global#2 | 0.05 | 2.8% | 1.0 | 0.06 | `if (((int)threadIdx.x) == 0)` |

### gemm_splitk_reduce_kernel grid (4, 16, 1) block (128, 1, 1)

48 launches, 256 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 0.61 us, warp 0.54 us, nsys graphs 0.77 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.25 | 46.6% | 0.27 | 46.7% |
| load | 0.24 | 45.1% | 0.26 | 45.1% |
| store | 0.04 | 8.2% | 0.05 | 8.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.24 | 45.1% | 1.0 | 0.29 | `acc[0] = P[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x))` |
| store.global | 0.04 | 8.2% | 1.0 | 0.06 | `C[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x)) * (int64` |

### assign_draft_budget_kernel grid (1, 1, 1) block (256, 1, 1)

16 launches, 8 warps; ranges, warp and last warp from the 1 recording CTAs' 8 warps; span 1.70 us, warp 1.68 us, nsys graphs 1.84 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed, with loops' unsampled iterations | 0.66 | 39.2% | 0.82 | 47.8% |
| wait | 0.63 | 37.4% | 0.41 | 23.9% |
| load | 0.20 | 12.0% | 0.30 | 17.2% |
| store | 0.19 | 11.4% | 0.19 | 11.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync | 0.32 | 18.8% | 1.0 | 0.35 | `__syncthreads();` |
| wait.sync#3 | 0.25 | 14.8% | 1.0 | 0.32 | `__syncthreads();` |
| load.global | 0.20 | 12.0% | 1.0 | 0.48 | `for (int chunk = 0; chunk < ((num_reqs + 255) >> 8); ++chunk)` |
| store.global | 0.19 | 11.4% | 1.0 | 0.19 | `for (int chunk_2 = 0; chunk_2 < ((num_reqs + 255) >> 8); ++chunk_2)` |
| wait.sync#2 (sampled iterations) | 0.06 | 3.8% | 1.0 | 0.06 | `__syncthreads();` |

### ced_copy_rows_kernel grid (200, 1, 1) block (128, 1, 1)

17 launches, 800 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 1.18 us, warp 1.02 us, nsys graphs 1.92 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.76 | 74.9% | 0.91 | 75.9% |
| unattributed | 0.26 | 25.1% | 0.29 | 24.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.76 | 74.9% | 1.0 | 0.86 | `for (int k = 0; k < 4; ++k) \| int64_t index = ced_ldg64((&(row_indices[((((int64_t)blockId` |

### combine_sampled_and_draft_tokens_kernel grid (8, 1, 1) block (32, 1, 1)

17 launches, 8 warps; ranges, warp and last warp from the 4 recording CTAs' 4 warps; span 1.31 us, warp 1.18 us, nsys graphs 1.44 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.65 | 54.7% | 0.67 | 54.9% |
| unattributed | 0.44 | 37.2% | 0.46 | 37.4% |
| store | 0.10 | 8.1% | 0.09 | 7.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.65 | 54.7% | 1.0 | 0.67 | `if (((((int)blockIdx.x) * 32) + ((int)threadIdx.x)) < num_reqs) \| req = idx_mapping[((int6` |
| store.global | 0.10 | 8.1% | 1.0 | 0.10 | `if ((prefill < seq) && (1 < num_logits))` |

### logits_rows_kernel grid (1, 1, 1) block (128, 1, 1)

17 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 1.18 us, warp 1.13 us, nsys graphs 2.08 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.51 | 45.4% | 0.51 | 39.9% |
| unattributed | 0.48 | 42.6% | 0.30 | 23.2% |
| store | 0.14 | 12.1% | 0.47 | 36.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.51 | 45.4% | 1.0 | 0.51 | `index = logits_indices[min(((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((in` |
| store.global | 0.14 | 12.1% | 1.0 | 0.35 | `if (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) < num_logits)` |

### scatter_rows_gather_kernel grid (1, 1, 1) block (128, 1, 1)

17 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 0.93 us, warp 0.85 us, nsys graphs 1.41 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.51 | 60.4% | 0.40 | 34.3% |
| store | 0.24 | 28.3% | 0.67 | 56.5% |
| load | 0.10 | 11.3% | 0.11 | 9.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.24 | 28.3% | 1.0 | 0.42 | `if (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) < (num_reqs * width))` |
| load.global | 0.10 | 11.3% | 1.0 | 0.10 | `index = idx_mapping[min((((((int64_t)0 <= ((int64_t)width)) && ((int64_t)0 <= rmod_1)) \|\| ` |

### token_counts_kernel grid (8, 1, 1) block (128, 1, 1)

17 launches, 32 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 1.15 us, warp 0.45 us, nsys graphs 1.47 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.26 | 57.3% | 0.32 | 27.5% |
| store | 0.14 | 30.4% | 0.60 | 52.0% |
| load | 0.06 | 12.3% | 0.24 | 20.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.14 | 30.4% | 0.2 | 0.58 | `if (((int)threadIdx.x) == 0) \| sampled[((int64_t)((int)blockIdx.x))] = (((int)(prefill_len` |
| load.global | 0.06 | 12.3% | 0.2 | 0.22 | `int count = sampled[((int64_t)((int)blockIdx.x))];` |

### flatten_sampled_kernel grid (1, 1, 1) block (128, 1, 1)

17 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 1.12 us, warp 0.83 us, nsys graphs 1.31 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.47 | 56.7% | 0.64 | 56.7% |
| unattributed | 0.34 | 40.4% | 0.39 | 34.9% |
| store | 0.02 | 2.9% | 0.09 | 8.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.42 | 50.0% | 1.0 | 0.42 | `for (int r = 0; r < num_reqs; ++r)` |
| load.global#2 | 0.06 | 6.7% | 0.2 | 0.22 | `int i = (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) - cu_num_logits[req]);` |
| store.global | 0.02 | 2.9% | 0.2 | 0.10 | `flat[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = co` |

### prepare_pos_seq_lens_kernel grid (9, 1, 1) block (256, 1, 1)

17 launches, 72 warps; ranges, warp and last warp from the 5 recording CTAs' 40 warps; span 1.12 us, warp 0.96 us, nsys graphs 1.28 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.46 | 48.2% | 0.58 | 52.2% |
| unattributed | 0.38 | 39.7% | 0.40 | 36.3% |
| store | 0.12 | 12.1% | 0.13 | 11.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.46 | 48.2% | 0.8 | 0.61 | `int computed = num_computed_tokens[idx_mapping[((int64_t)((int)blockIdx.x))]];` |
| store.global#2 | 0.10 | 10.7% | 0.8 | 0.13 | `if (((int)threadIdx.x) == 0)` |
| store.global | 0.01 | 1.3% | 0.2 | 0.06 | `for (int it = 0; it < (((max_reqs + 255) - num_reqs) >> 8); ++it)` |

### padding_rows_kernel grid (1, 1, 1) block (256, 1, 1)

33 launches, 8 warps; ranges, warp and last warp from the 1 recording CTAs' 8 warps; span 0.38 us, warp 0.36 us, nsys graphs 0.74 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.26 | 73.0% | 0.27 | 67.5% |
| store | 0.10 | 27.0% | 0.13 | 32.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.10 | 27.0% | 1.0 | 0.10 | `if (((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) < rows)` |

### context_inputs_kernel grid (5, 48, 1) block (128, 1, 1)

1 launches, 960 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 11.55 us, warp 10.93 us, nsys graphs 8.61 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 8.67 | 79.3% | 8.86 | 76.9% |
| unattributed | 1.35 | 12.4% | 0.58 | 5.0% |
| store | 0.91 | 8.3% | 2.08 | 18.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 8.67 | 79.3% | 1.0 | 10.82 | `vllm_dspark_context_b128((&(aux[(((((int64_t)((int)blockIdx.y)) * (int64_t)15360) + (((int` |
| store.global | 0.91 | 8.3% | 1.0 | 4.16 | `if (((((int)blockIdx.x) == 0) & (((int)threadIdx.x) == 0)) & (num_tokens <= ((int)blockIdx` |

### ced_copy_rows_kernel grid (80, 1, 1) block (128, 1, 1)

16 launches, 320 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 0.70 us, warp 0.61 us, nsys graphs 1.10 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.37 | 59.9% | 0.42 | 61.3% |
| unattributed | 0.25 | 40.1% | 0.27 | 38.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.37 | 59.9% | 1.0 | 0.42 | `for (int k = 0; k < 4; ++k) \| int64_t index = ced_ldg64((&(row_indices[((((int64_t)blockId` |

### scatter_rows_kernel grid (1, 1, 1) block (128, 1, 1)

17 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 0.64 us, warp 0.59 us, nsys graphs 1.01 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.39 | 66.2% | 0.39 | 61.0% |
| load | 0.10 | 17.6% | 0.11 | 17.9% |
| store | 0.10 | 16.2% | 0.13 | 21.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.10 | 17.6% | 1.0 | 0.13 | `index = idx_mapping[min((((((int64_t)0 <= ((int64_t)width)) && ((int64_t)0 <= rmod_1)) \|\| ` |
| store.global | 0.10 | 16.2% | 1.0 | 0.13 | `if (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) < (num_reqs * width))` |

### ced_requests_kernel grid (1, 1, 1) block (128, 1, 1)

17 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 0.61 us, warp 0.54 us, nsys graphs 1.15 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.34 | 62.7% | 0.34 | 53.6% |
| store | 0.10 | 19.4% | 0.19 | 29.6% |
| load | 0.10 | 17.9% | 0.11 | 16.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.10 | 19.4% | 1.0 | 0.19 | `if (((int)threadIdx.x) < 8) \| if (((int)threadIdx.x) <= 8)` |
| load.global | 0.10 | 17.9% | 1.0 | 0.10 | `uint v_ = tl::load_global_32((&(starts[1]))); \| uint v_ = tl::load_global_32((&(starts[min` |

### per_token_cast_kernel_kernel grid (360, 1, 1) block (64, 1, 1)

1 launches, 720 warps; ranges, warp and last warp from the 4 recording CTAs' 8 warps; span 4.10 us, warp 3.25 us, nsys graphs 3.39 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 1.44 | 44.2% | 1.66 | 41.9% |
| load | 0.79 | 24.4% | 0.96 | 24.2% |
| unattributed | 0.54 | 16.7% | 1.22 | 30.6% |
| compute | 0.48 | 14.8% | 0.13 | 3.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.67 | 20.7% | 1.0 | 0.90 | `for (int i = 0; i < 2; ++i)` |
| store.global#2 | 0.56 | 17.1% | 1.0 | 1.70 | `for (int i_3 = 0; i_3 < 4; ++i_3)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.48 | 14.8% | 1.0 | 3.01 | `amax_fragment[i_1] = tl::AllReduce<tl::MaxOp, 2, 1, 0, tl::NamedBarrier<64>>::run(amax_fra` |
| store.global*2 (est.: sampled iterations x2) | 0.46 | 14.0% | 1.0 | 1.41 | `out_sf[((((((int64_t)((int)blockIdx.x)) % (int64_t)15) * (int64_t)32) + (((int64_t)((int)t` |
| store.global#3*2 (est.: sampled iterations x2) | 0.42 | 13.0% | 1.0 | 2.69 | `*(fp8_e4_16_t*)(out + ((((((((int64_t)((int)blockIdx.x)) / (int64_t)3) * (int64_t)6144) + ` |
| load.global#2*2 (est.: sampled iterations x2) | 0.12 | 3.7% | 1.0 | 0.19 | `amax_fragment[i_1] = cutlass::bfloat16_t(__hmax((bfloat16_t(((nv_bfloat162*)(&(amax_fragme` |

### gemm_splitk_reduce_kernel grid (4, 48, 1) block (128, 1, 1)

3 launches, 768 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 0.80 us, warp 0.74 us, nsys graphs 1.31 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.40 | 54.2% | 0.42 | 45.9% |
| unattributed | 0.29 | 39.6% | 0.45 | 49.4% |
| store | 0.05 | 6.2% | 0.04 | 4.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.40 | 54.2% | 1.0 | 0.42 | `acc[0] = P[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x))` |
| store.global | 0.05 | 6.2% | 1.0 | 0.06 | `C[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x)) * (int64` |

### ced_copy_rows_kernel grid (240, 1, 1) block (128, 1, 1)

1 launches, 960 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 1.34 us, warp 1.11 us, nsys graphs 1.54 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.81 | 73.2% | 1.02 | 78.0% |
| unattributed | 0.30 | 26.8% | 0.29 | 22.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.81 | 73.2% | 1.0 | 1.02 | `for (int k = 0; k < 4; ++k) \| int64_t index = ced_ldg64((&(row_indices[((((int64_t)blockId` |
=== timeline and tables: exit 0 in 36 s
