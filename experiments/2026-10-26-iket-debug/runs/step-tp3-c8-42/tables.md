=== timeline and tables
trace /c/iket/step-tp3-c8-20261026T170609Z/trace/iket/pid_0x18c2/iket.decoded_results.json: 16677 launches, graphLaunches dict of 90

## Timeline: 7547 launches in 1245741.3 us; the GPU is busy 366944.6 us (29%) and idle 878796.6 us in 5864 gaps; gaps over 200 us split it into 989 segments

The 15 of 989 segments with the most busy time:

| segment | launches | wall us | busy us | busy | first kernel |
|---:|---:|---:|---:|---:|---|
| 166 | 265 | 7922.4 | 6388.3 | 81% | partial_argmax_kernel |
| 231 | 262 | 7916.5 | 6512.1 | 82% | partial_argmax_kernel |
| 286 | 244 | 6831.6 | 5958.0 | 87% | metadata_builder_kernel |
| 341 | 241 | 6777.0 | 6316.2 | 93% | metadata_builder_kernel |
| 406 | 242 | 6397.4 | 6008.4 | 94% | metadata_builder_kernel |
| 464 | 259 | 7297.3 | 6046.2 | 83% | scatter_rows_gather_kernel |
| 520 | 238 | 6801.3 | 6399.3 | 94% | metadata_builder_kernel |
| 571 | 246 | 6651.9 | 5977.8 | 90% | metadata_builder_kernel |
| 615 | 262 | 7399.4 | 6110.7 | 83% | partial_argmax_kernel |
| 667 | 235 | 6118.0 | 5972.3 | 98% | mhc_project_streams_kernel |
| 716 | 240 | 6657.6 | 5980.1 | 90% | metadata_builder_kernel |
| 773 | 238 | 6563.0 | 6035.2 | 92% | metadata_builder_kernel |
| 813 | 263 | 7983.9 | 6384.3 | 80% | gemm_bf16_kernel |
| 880 | 236 | 6119.5 | 5949.8 | 97% | metadata_builder_kernel |
| 938 | 252 | 6922.7 | 5991.5 | 87% | gather_block_tables_kernel |

| idle us | after | before |
|---:|---|---|
| 106523.8 | resample_kernel | insert_kernel |
| 81551.9 | scatter_rows_kernel | padding_rows_kernel |
| 11183.9 | norm_forward_and_per_token_cast_kernel_kernel | sink_flags_kernel |
| 11146.4 | norm_forward_and_per_token_cast_kernel_kernel | sink_flags_kernel |
| 11123.0 | norm_forward_and_per_token_cast_kernel_kernel | sink_flags_kernel |
| 11023.6 | norm_forward_and_per_token_cast_kernel_kernel | sink_flags_kernel |
| 11004.4 | norm_forward_and_per_token_cast_kernel_kernel | sink_flags_kernel |
| 10937.6 | norm_forward_and_per_token_cast_kernel_kernel | sink_flags_kernel |
| 10734.7 | norm_forward_and_per_token_cast_kernel_kernel | sink_flags_kernel |
| 10733.4 | norm_forward_and_per_token_cast_kernel_kernel | sink_flags_kernel |

## Engine steps

| step | kind | launches | wall us | busy us | busy | most span |
|---:|---|---:|---:|---:|---:|---|
| 0 | prefill | 410 | 99678.9 | 31081.6 | 31% | vocab_packed_projection_kernel 9315, moe_grouped_fp8_fp4_gemm_kernel 6591, gemm_mxfp8_kernel 5100 |
| 1 | decode | 420 | 152587.6 | 20384.1 | 13% | vocab_packed_projection_kernel 9622, gemm_mxfp8_kernel 2865, moe_grouped_fp8_fp4_gemm_kernel 2403 |
| 2 | decode | 420 | 134734.5 | 19624.9 | 15% | vocab_packed_projection_kernel 8743, gemm_mxfp8_kernel 2989, moe_grouped_fp8_fp4_gemm_kernel 2867 |
| 3 | decode | 420 | 49306.8 | 19727.1 | 40% | vocab_packed_projection_kernel 8791, gemm_mxfp8_kernel 2990, moe_grouped_fp8_fp4_gemm_kernel 2898 |
| 4 | decode | 420 | 55887.6 | 20517.5 | 37% | vocab_packed_projection_kernel 9310, moe_grouped_fp8_fp4_gemm_kernel 2923, gemm_mxfp8_kernel 2912 |
| 5 | decode | 420 | 60600.9 | 19716.8 | 33% | vocab_packed_projection_kernel 9049, gemm_mxfp8_kernel 2773, moe_grouped_fp8_fp4_gemm_kernel 2746 |
| 6 | decode | 420 | 57578.0 | 20020.5 | 35% | vocab_packed_projection_kernel 8913, moe_grouped_fp8_fp4_gemm_kernel 2955, gemm_mxfp8_kernel 2879 |
| 7 | decode | 420 | 61741.7 | 19548.9 | 32% | vocab_packed_projection_kernel 8751, moe_grouped_fp8_fp4_gemm_kernel 2840, gemm_mxfp8_kernel 2788 |
| 8 | decode | 420 | 61701.9 | 19509.0 | 32% | vocab_packed_projection_kernel 8757, moe_grouped_fp8_fp4_gemm_kernel 2790, gemm_mxfp8_kernel 2788 |
| 9 | decode | 420 | 56137.7 | 20238.0 | 36% | vocab_packed_projection_kernel 9077, gemm_mxfp8_kernel 2935, moe_grouped_fp8_fp4_gemm_kernel 2915 |
| 10 | decode | 420 | 57761.8 | 20058.4 | 35% | vocab_packed_projection_kernel 9268, gemm_mxfp8_kernel 2797, moe_grouped_fp8_fp4_gemm_kernel 2791 |
| 11 | decode | 420 | 53277.5 | 19605.9 | 37% | vocab_packed_projection_kernel 8730, moe_grouped_fp8_fp4_gemm_kernel 2908, gemm_mxfp8_kernel 2809 |
| 12 | decode | 420 | 57309.5 | 19373.2 | 34% | vocab_packed_projection_kernel 8685, moe_grouped_fp8_fp4_gemm_kernel 2793, gemm_mxfp8_kernel 2760 |
| 13 | decode | 420 | 59013.9 | 19788.9 | 34% | vocab_packed_projection_kernel 9043, gemm_mxfp8_kernel 2786, moe_grouped_fp8_fp4_gemm_kernel 2772 |
| 14 | decode | 420 | 54111.1 | 19450.7 | 36% | vocab_packed_projection_kernel 8666, gemm_mxfp8_kernel 2789, moe_grouped_fp8_fp4_gemm_kernel 2785 |
| 15 | decode | 420 | 59762.2 | 19436.0 | 33% | vocab_packed_projection_kernel 8752, gemm_mxfp8_kernel 2812, moe_grouped_fp8_fp4_gemm_kernel 2798 |
| 16 | decode | 420 | 57056.5 | 19498.0 | 34% | vocab_packed_projection_kernel 8806, gemm_mxfp8_kernel 2768, moe_grouped_fp8_fp4_gemm_kernel 2767 |
| 17 | decode | 417 | 55537.7 | 19365.2 | 35% | vocab_packed_projection_kernel 8710, gemm_mxfp8_kernel 2809, moe_grouped_fp8_fp4_gemm_kernel 2770 |

## Overview over 17 decode steps: 7137 launches of 127 kernel shapes, spans summing to 361300.4 us (21253.0 a step)

| kernel | grid | launches a step | span us a step | share | span us a launch | last warp: wait | load | compute | store | unattributed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vocab_packed_projection_kernel | (1, 4040, 1) | 1.05882 | 5077.0 | 23.9% | 4691.58 | 32% | 26% | 24% | 0% | 18% |
| vocab_packed_projection_kernel | (1, 4040, 1) | 0.941176 | 3845.0 | 18.1% | 4058.11 | 45% | 38% | 10% | 0% | 7% |
| gemm_bf16_kernel | (2020, 1, 1) | 5 | 1670.9 | 7.9% | 335.07 | 92% | 0% | 4% | 1% | 4% |
| l2_prefetch_kernel | (2, 1, 1) | 17 | 1349.4 | 6.3% | 72.80 | 0% | 100% | 0% | 0% | 0% |
| gemm_mxfp8_kernel | (80, 1, 1) | 19 | 1263.4 | 5.9% | 55.33 | 64% | 8% | 13% | 1% | 13% |
| moe_grouped_fp8_fp4_gemm_kernel | (40, 288, 1) | 6 | 1179.3 | 5.5% | 195.07 | 7% | 9% | 0% | 0% | 84% |
| moe_grouped_fp8_fp4_gemm_kernel | (12, 288, 1) | 6 | 1120.0 | 5.3% | 186.91 | 10% | 11% | 0% | 0% | 79% |
| mhc_project_streams_kernel | (40, 3, 1) | 18 | 869.6 | 4.1% | 53.14 | 9% | 35% | 19% | 4% | 34% |
| gemm_mxfp8_kernel | (192, 1, 1) | 9 | 756.7 | 3.6% | 71.97 | 34% | 7% | 15% | 1% | 43% |
| gemm_fp8w_bf16_partials_kernel | (48, 4, 1) | 9 | 656.4 | 3.1% | 89.60 | 43% | 13% | 17% | 1% | 26% |
| gemm_mxfp8_kernel | (28, 1, 1) | 9 | 407.2 | 1.9% | 41.54 | 42% | 9% | 24% | 0% | 25% |
| gemm_mxfp8_kernel | (24, 1, 1) | 9 | 359.6 | 1.7% | 33.06 | 43% | 16% | 34% | 0% | 7% |
| sparse_mla_kernel | (1, 48, 1) | 4 | 303.1 | 1.4% | 77.46 | 2% | 11% | 2% | 1% | 85% |
| moe_grouped_fp8_fp4_gemm_kernel | (40, 120, 1) | 3 | 265.7 | 1.2% | 86.59 | 14% | 17% | 0% | 0% | 68% |
| moe_grouped_fp8_fp4_gemm_kernel | (12, 120, 1) | 3 | 242.0 | 1.1% | 76.96 | 17% | 19% | 0% | 0% | 64% |
| partial_argmax_kernel | (8, 64, 1) | 5 | 154.4 | 0.7% | 30.78 | 2% | 70% | 2% | 1% | 25% |
| gemm_bf16_splitk_kernel | (16, 5, 1) | 2 | 142.1 | 0.7% | 71.20 | 93% | 10% | 7% | 1% | 0% |
| gemm_bf16_partials_kernel | (6, 8, 1) | 6 | 131.8 | 0.6% | 20.05 | 31% | 0% | 32% | 2% | 35% |
| mhc_finalize_kernel | (48, 1, 1) | 12 | 91.0 | 0.4% | 6.56 | 12% | 17% | 9% | 18% | 44% |
| sparse_mla_kernel | (2, 1, 40) | 3 | 82.0 | 0.4% | 27.17 | 20% | 12% | 46% | 6% | 16% |

| wait | kernel | grid | last-warp us a step | a launch | first statement |
|---|---|---|---:|---:|---|
| wait.cp_async*118 | gemm_mxfp8_kernel | (80, 1, 1) | 189.0 | 9.95 | `tl::cp_async_wait<2>();` |
| wait.sync#2*118 | gemm_mxfp8_kernel | (80, 1, 1) | 126.6 | 6.66 | `__syncthreads();` |
| wait.cp_async*19 | gemm_mxfp8_kernel | (28, 1, 1) | 82.7 | 9.19 | `tl::cp_async_wait<1>();` |
| wait.cp_async*22 | gemm_mxfp8_kernel | (80, 1, 1) | 77.6 | 4.08 | `tl::cp_async_wait<2>();` |
| wait.sync#2*38 | gemm_mxfp8_kernel | (24, 1, 1) | 74.0 | 8.23 | `__syncthreads();` |
| wait.cp_async*13 | gemm_bf16_splitk_kernel | (16, 5, 1) | 70.8 | 35.40 | `tl::cp_async_wait<2>();` |
| wait.sync#2*19 | gemm_mxfp8_kernel | (28, 1, 1) | 66.4 | 7.38 | `__syncthreads();` |
| wait.sync#2*22 | gemm_mxfp8_kernel | (80, 1, 1) | 57.3 | 3.01 | `__syncthreads();` |
| wait.cp_async*38 | gemm_mxfp8_kernel | (24, 1, 1) | 48.4 | 5.37 | `tl::cp_async_wait<2>();` |
| wait.cp_async*14 | gemm_fp8w_bf16_partials_kernel | (48, 4, 1) | 37.8 | 4.20 | `tl::cp_async_wait<3>();` |
| wait.cp_async*9 | gemm_mxfp8_kernel | (64, 1, 1) | 36.6 | 18.30 | `tl::cp_async_wait<1>();` |
| wait.mbarrier#2 | gemm_bf16_kernel | (2020, 1, 1) | 36.2 | 7.23 | `mbarrier[0].wait(0);` |
| wait.sync#2*14 | gemm_fp8w_bf16_partials_kernel | (48, 4, 1) | 27.2 | 3.02 | `__syncthreads();` |
| wait.sync*38 | gemm_mxfp8_kernel | (24, 1, 1) | 20.7 | 2.30 | `__syncthreads();` |
| wait.cp_async*14 | gemm_bf16_splitk_kernel | (16, 5, 1) | 18.0 | 9.00 | `tl::cp_async_wait<1>();` |
| wait.mbarrier#2*10 | gemm_bf16_partials_kernel | (6, 8, 1) | 17.3 | 2.88 | `mbarrier[(kb_1 % 3)].wait(((kb_1 % 6) / 3));` |
| wait.sync*4 | mhc_project_streams_kernel | (40, 3, 1) | 16.8 | 0.94 | `__syncthreads();` |
| wait.cp_async#3 | gemm_mxfp8_kernel | (192, 1, 1) | 16.0 | 1.78 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>();` |
| wait.cp_async*8 | gemm_mxfp8_kernel | (192, 1, 1) | 15.0 | 1.67 | `tl::cp_async_wait<2>();` |
| wait.sync*13 | gemm_bf16_splitk_kernel | (16, 5, 1) | 14.1 | 7.07 | `__syncthreads();` |

### vocab_packed_projection_kernel grid (1, 4040, 1) block (256, 1, 1)

18 launches, 32320 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 4691.58 us, warp 59.74 us, nsys graphs 4543.94 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 22.17 | 37.1% | 18.26 | 32.1% |
| compute | 13.69 | 22.9% | 13.71 | 24.1% |
| unattributed | 13.01 | 21.8% | 10.01 | 17.6% |
| load | 10.81 | 18.1% | 14.88 | 26.1% |
| store | 0.06 | 0.1% | 0.06 | 0.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*37 (est.: sampled iterations x37) | 12.75 | 21.3% | 1.0 | 13.62 | `{` |
| wait.sync#2*37 (est.: sampled iterations x37) | 7.25 | 12.1% | 1.0 | 17.17 | `__syncthreads();` |
| wait.sync#3*37 (est.: sampled iterations x37) | 5.77 | 9.7% | 1.0 | 9.47 | `__syncthreads();` |
| load.cp_async#2*37 (est.: sampled iterations x37) | 5.31 | 8.9% | 1.0 | 7.10 | `for (int i_4 = 0; i_4 < 4; ++i_4)` |
| load.global | 5.15 | 8.6% | 1.0 | 12.19 | `int start = row_offsets[(((int)blockIdx.y) * 16)]; \| int start = row_offsets[(((int)blockI` |
| wait.cp_async*37 (est.: sampled iterations x37) | 2.90 | 4.9% | 1.0 | 16.58 | `tl::cp_async_wait<6>();` |
| wait.sync*37 (est.: sampled iterations x37) | 2.59 | 4.3% | 1.0 | 3.55 | `__syncthreads();` |
| wait.cp_async#2*37 (est.: sampled iterations x37) | 2.26 | 3.8% | 1.0 | 2.96 | `tl::cp_async_wait<7>();` |
| load.cp_async | 0.35 | 0.6% | 1.0 | 0.46 | `for (int i = 0; i < 4; ++i) \| for (int i = 0; i < 8; ++i) \| for (int i_1 = 0; i_1 < 4; ++i` |
| wait.sync#4 | 0.32 | 0.5% | 1.0 | 1.20 | `__syncthreads();` |
| compute.mma#4 | 0.32 | 0.5% | 1.0 | 0.35 | `{` |
| compute.mma#3 | 0.31 | 0.5% | 1.0 | 0.32 | `{` |
| compute.mma#2 | 0.31 | 0.5% | 1.0 | 0.35 | `{` |
| wait.cp_async#3 | 0.15 | 0.2% | 1.0 | 1.20 | `tl::cp_async_wait<4>();` |
| wait.sync#5 | 0.14 | 0.2% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#9 | 0.14 | 0.2% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#7 | 0.14 | 0.2% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#6 | 0.14 | 0.2% | 1.0 | 0.82 | `__syncthreads();` |
| wait.cp_async#5 | 0.09 | 0.1% | 1.0 | 0.78 | `tl::cp_async_wait<2>();` |
| wait.sync#8 | 0.07 | 0.1% | 1.0 | 0.16 | `__syncthreads();` |
| wait.cp_async#7 | 0.06 | 0.1% | 1.0 | 0.14 | `tl::cp_async_wait<0>();` |
| store.global | 0.06 | 0.1% | 1.0 | 0.08 | `for (int i_13 = 0; i_13 < 2; ++i_13) \| for (int i_14 = 0; i_14 < 4; ++i_14) \| for (int i_9` |
| wait.cp_async#8 | 0.05 | 0.1% | 1.0 | 0.06 | `tl::cp_async_wait<1>();` |
| wait.cp_async#4 | 0.05 | 0.1% | 1.0 | 0.06 | `tl::cp_async_wait<5>();` |
| wait.cp_async#6 | 0.05 | 0.1% | 1.0 | 0.06 | `tl::cp_async_wait<3>();` |

### vocab_packed_projection_kernel grid (1, 4040, 1) block (128, 1, 1)

16 launches, 16160 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 4058.11 us, warp 54.17 us, nsys graphs 4061.76 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 20.88 | 38.5% | 23.81 | 45.0% |
| load | 14.48 | 26.7% | 20.15 | 38.1% |
| unattributed | 13.73 | 25.3% | 3.90 | 7.4% |
| compute | 5.03 | 9.3% | 5.05 | 9.5% |
| store | 0.05 | 0.1% | 0.05 | 0.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 11.89 | 21.9% | 1.0 | 21.12 | `int start = row_offsets[(((int)blockIdx.y) * 16)]; \| int start = row_offsets[(((int)blockI` |
| wait.cp_async*17 (est.: sampled iterations x17) | 8.43 | 15.6% | 1.0 | 35.63 | `tl::cp_async_wait<6>();` |
| wait.sync#2*17 (est.: sampled iterations x17) | 7.40 | 13.7% | 1.0 | 22.58 | `__syncthreads();` |
| compute.mma*17 (est.: sampled iterations x17) | 4.34 | 8.0% | 1.0 | 4.35 | `{` |
| load.cp_async#2*17 (est.: sampled iterations x17) | 2.29 | 4.2% | 1.0 | 2.72 | `for (int i_6 = 0; i_6 < 8; ++i_6) \| for (int i_9 = 0; i_9 < 4; ++i_9)` |
| wait.sync#3*17 (est.: sampled iterations x17) | 1.10 | 2.0% | 1.0 | 1.63 | `__syncthreads();` |
| wait.sync*17 (est.: sampled iterations x17) | 1.02 | 1.9% | 1.0 | 1.09 | `__syncthreads();` |
| wait.cp_async#2*17 (est.: sampled iterations x17) | 0.95 | 1.8% | 1.0 | 1.09 | `tl::cp_async_wait<7>();` |
| wait.sync#4 | 0.38 | 0.7% | 1.0 | 1.55 | `__syncthreads();` |
| wait.cp_async#3 | 0.35 | 0.7% | 1.0 | 1.90 | `tl::cp_async_wait<4>();` |
| wait.sync#6 | 0.32 | 0.6% | 1.0 | 1.07 | `__syncthreads();` |
| load.cp_async | 0.30 | 0.6% | 1.0 | 0.40 | `for (int i = 0; i < 4; ++i) \| for (int i = 0; i < 8; ++i) \| for (int i_1 = 0; i_1 < 4; ++i` |
| wait.cp_async#5 | 0.29 | 0.5% | 1.0 | 1.42 | `tl::cp_async_wait<2>();` |
| compute.mma#2 | 0.24 | 0.4% | 1.0 | 0.26 | `{` |
| compute.mma#3 | 0.23 | 0.4% | 1.0 | 0.26 | `{` |
| compute.mma#4 | 0.23 | 0.4% | 1.0 | 0.26 | `{` |
| wait.sync#8 | 0.20 | 0.4% | 1.0 | 0.72 | `__syncthreads();` |
| wait.cp_async#7 | 0.15 | 0.3% | 1.0 | 0.93 | `tl::cp_async_wait<0>();` |
| wait.sync#5 | 0.06 | 0.1% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#7 | 0.05 | 0.1% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#9 | 0.05 | 0.1% | 1.0 | 0.06 | `__syncthreads();` |
| store.global | 0.05 | 0.1% | 1.0 | 0.06 | `for (int i_13 = 0; i_13 < 2; ++i_13) \| for (int i_14 = 0; i_14 < 4; ++i_14) \| for (int i_9` |
| wait.cp_async#6 | 0.04 | 0.1% | 1.0 | 0.06 | `tl::cp_async_wait<3>();` |
| wait.cp_async#4 | 0.04 | 0.1% | 1.0 | 0.06 | `tl::cp_async_wait<5>();` |
| wait.cp_async#8 | 0.04 | 0.1% | 1.0 | 0.06 | `tl::cp_async_wait<1>();` |

### gemm_bf16_kernel grid (2020, 1, 1) block (256, 1, 1)

85 launches, 16160 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 335.07 us, warp 4.62 us, nsys graphs 271.30 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 4.06 | 87.9% | 7.32 | 91.5% |
| unattributed | 0.34 | 7.4% | 0.32 | 4.0% |
| compute | 0.14 | 3.1% | 0.29 | 3.6% |
| load | 0.04 | 0.8% | 0.00 | 0.0% |
| store | 0.04 | 0.8% | 0.07 | 0.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.mbarrier#2 | 3.96 | 85.6% | 0.5 | 10.11 | `mbarrier[0].wait(0);` |
| compute.mma | 0.14 | 3.1% | 0.5 | 0.29 | `{` |
| wait.sync | 0.08 | 1.7% | 1.0 | 0.10 | `__syncthreads();` |
| load.tma | 0.04 | 0.8% | 0.5 | 0.13 | `if (tl::tl_shuffle_elect<128>())` |
| store.global | 0.04 | 0.8% | 0.5 | 0.10 | `for (int i_3 = 0; i_3 < 4; ++i_3) \| for (int i_4 = 0; i_4 < 8; ++i_4) \| for (int i_6 = 0; ` |
| wait.mbarrier | 0.03 | 0.6% | 0.5 | 0.06 | `mbarrier[2].wait(1);` |

### l2_prefetch_kernel grid (2, 1, 1) block (128, 1, 1)

289 launches, 8 warps; ranges, warp and last warp from the 2 recording CTAs' 8 warps; span 72.80 us, warp 71.80 us, nsys graphs 71.46 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 71.40 | 99.4% | 79.04 | 99.6% |
| unattributed | 0.40 | 0.6% | 0.34 | 0.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 71.40 | 99.4% | 1.0 | 72.42 | `for (int s = 0; s < (nseg2 >> 1); ++s)` |

### gemm_mxfp8_kernel grid (80, 1, 1) block (128, 1, 1)

323 launches, 320 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 55.33 us, warp 25.98 us, nsys graphs 51.09 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 23.97 | 92.3% | 5.58 | 13.4% |
| wait | 0.94 | 3.6% | 26.80 | 64.5% |
| compute | 0.60 | 2.3% | 5.60 | 13.5% |
| load | 0.24 | 0.9% | 3.32 | 8.0% |
| store | 0.22 | 0.9% | 0.25 | 0.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.cp_async#2 | 0.31 | 1.2% | 1.0 | 1.25 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| compute.mma#2 | 0.30 | 1.2% | 1.0 | 0.32 | `{` |
| compute.mma#3 | 0.30 | 1.2% | 1.0 | 0.32 | `{` |
| load.cp_async | 0.24 | 0.9% | 1.0 | 0.29 | `for (int i_1 = 0; i_1 < 2; ++i_1) \| for (int i_1 = 0; i_1 < 4; ++i_1) \| for (int i_1 = 0; ` |
| wait.cp_async#3 | 0.24 | 0.9% | 1.0 | 0.99 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>();` |
| store.global | 0.22 | 0.9% | 1.0 | 0.29 | `for (int i_10 = 0; i_10 < 16; ++i_10) \| for (int i_10 = 0; i_10 < 8; ++i_10) \| for (int i_` |
| wait.sync#3 | 0.22 | 0.9% | 1.0 | 0.80 | `__syncthreads();` |
| wait.sync#4 | 0.16 | 0.6% | 1.0 | 0.67 | `__syncthreads();` |
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

102 launches, 46080 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 195.07 us, warp 3.55 us, nsys graphs 136.53 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.76 | 49.5% | 0.54 | 83.7% |
| wait | 0.67 | 18.9% | 0.05 | 7.4% |
| compute | 0.53 | 14.9% | 0.00 | 0.0% |
| load | 0.53 | 14.9% | 0.06 | 8.9% |
| store | 0.06 | 1.7% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*6 (est.: sampled iterations x6) | 0.53 | 14.9% | 0.2 | 2.11 | `{` |
| load#2*6 (est.: sampled iterations x6) | 0.22 | 6.1% | 0.2 | 1.15 | `if (ko < 4)` |
| wait.sync#5*6 (est.: sampled iterations x6) | 0.17 | 4.7% | 0.2 | 0.77 | `__syncthreads();` |
| wait.mbarrier*6 (est.: sampled iterations x6) | 0.13 | 3.7% | 0.2 | 0.58 | `loaded[(ko % 3)].wait((ko / 3));` |
| load.global | 0.12 | 3.5% | 1.0 | 0.32 | `int expert = TileExpert[((int64_t)((int)blockIdx.y))];` |
| load.global#3 | 0.12 | 3.4% | 0.2 | 0.48 | `if (((expert < 128) && (0 <= expert))) \| if (((expert < 384) && (0 <= expert)))` |
| wait.cp_async*6 (est.: sampled iterations x6) | 0.10 | 2.7% | 0.2 | 0.38 | `tl::cp_async_wait<2>();` |
| wait.sync#4*6 (est.: sampled iterations x6) | 0.10 | 2.7% | 0.2 | 0.38 | `__syncthreads();` |
| wait.sync#6*6 (est.: sampled iterations x6) | 0.10 | 2.7% | 0.2 | 0.38 | `__syncthreads();` |
| store.global | 0.06 | 1.7% | 0.2 | 0.27 | `for (int i_2 = 0; i_2 < 16; ++i_2) \| for (int i_4 = 0; i_4 < 32; ++i_4)` |
| load | 0.04 | 1.2% | 0.2 | 0.22 | `for (int s = 0; s < 2; ++s)` |
| wait.sync | 0.04 | 1.2% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#2 | 0.03 | 0.9% | 0.2 | 0.16 | `__syncthreads();` |
| load.global#2 | 0.02 | 0.7% | 0.2 | 0.19 | `if (((int)threadIdx.x) < 32) \| if (((int)threadIdx.x) < 64)` |
| wait.sync#3 | 0.01 | 0.2% | 0.2 | 0.03 | `__syncthreads();` |

### moe_grouped_fp8_fp4_gemm_kernel grid (12, 288, 1) block (128, 1, 1)

102 launches, 13824 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 186.91 us, warp 21.43 us, nsys graphs 170.64 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 10.76 | 50.2% | 0.40 | 79.2% |
| wait | 5.45 | 25.4% | 0.05 | 9.6% |
| compute | 3.36 | 15.7% | 0.00 | 0.0% |
| load | 1.81 | 8.4% | 0.06 | 11.2% |
| store | 0.06 | 0.3% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*40 (est.: sampled iterations x40) | 3.36 | 15.7% | 0.2 | 14.08 | `{` |
| wait.mbarrier*40 (est.: sampled iterations x40) | 2.08 | 9.7% | 0.2 | 8.32 | `loaded[(ko % 3)].wait(((ko % 6) / 3));` |
| load#2*40 (est.: sampled iterations x40) | 1.52 | 7.1% | 0.2 | 7.68 | `if (ko < 38)` |
| wait.sync#5*40 (est.: sampled iterations x40) | 1.36 | 6.3% | 0.2 | 6.40 | `__syncthreads();` |
| wait.cp_async*40 (est.: sampled iterations x40) | 0.64 | 3.0% | 0.2 | 2.56 | `tl::cp_async_wait<2>();` |
| wait.sync#4*40 (est.: sampled iterations x40) | 0.64 | 3.0% | 0.2 | 2.56 | `__syncthreads();` |
| wait.sync#6*40 (est.: sampled iterations x40) | 0.64 | 3.0% | 0.2 | 2.56 | `__syncthreads();` |
| load.global#3 | 0.11 | 0.5% | 0.2 | 0.45 | `if (((expert < 128) && (0 <= expert))) \| if (((expert < 384) && (0 <= expert)))` |
| load.global | 0.10 | 0.5% | 1.0 | 0.26 | `int expert = TileExpert[((int64_t)((int)blockIdx.y))];` |
| store.global | 0.06 | 0.3% | 0.2 | 0.26 | `for (int i_2 = 0; i_2 < 16; ++i_2) \| for (int i_4 = 0; i_4 < 32; ++i_4)` |
| wait.sync | 0.05 | 0.2% | 1.0 | 0.06 | `__syncthreads();` |
| load | 0.04 | 0.2% | 0.2 | 0.22 | `for (int s = 0; s < 2; ++s)` |
| wait.sync#2 | 0.03 | 0.1% | 0.2 | 0.16 | `__syncthreads();` |
| load.global#2 | 0.02 | 0.1% | 0.2 | 0.19 | `if (((int)threadIdx.x) < 32) \| if (((int)threadIdx.x) < 64)` |
| wait.sync#3 | 0.01 | 0.0% | 0.2 | 0.05 | `__syncthreads();` |

### mhc_project_streams_kernel grid (40, 3, 1) block (128, 1, 1)

306 launches, 480 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 53.14 us, warp 20.68 us, nsys graphs 34.96 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 9.77 | 47.2% | 4.64 | 34.7% |
| unattributed | 5.83 | 28.2% | 4.53 | 33.9% |
| compute | 2.77 | 13.4% | 2.56 | 19.1% |
| wait | 1.83 | 8.8% | 1.15 | 8.6% |
| store | 0.48 | 2.3% | 0.50 | 3.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2*4 (est.: sampled iterations x4) | 9.56 | 46.2% | 1.0 | 19.26 | `for (int i_1 = 0; i_1 < 6; ++i_1)` |
| compute.mma*4 (est.: sampled iterations x4) | 2.64 | 12.8% | 1.0 | 2.69 | `{` |
| wait.sync*4 (est.: sampled iterations x4) | 1.64 | 8.0% | 1.0 | 5.89 | `__syncthreads();` |
| store.global*4 (est.: sampled iterations x4) | 0.26 | 1.2% | 1.0 | 0.26 | `if (((((int)blockIdx.y) * 16) + (((int)threadIdx.x) >> 3)) < M)` |
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

153 launches, 768 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 71.97 us, warp 17.26 us, nsys graphs 73.54 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 8.26 | 47.9% | 6.74 | 33.8% |
| unattributed | 4.58 | 26.5% | 8.55 | 42.9% |
| compute | 2.94 | 17.0% | 2.96 | 14.8% |
| load | 1.28 | 7.4% | 1.48 | 7.4% |
| store | 0.20 | 1.2% | 0.19 | 0.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.cp_async*8 (est.: sampled iterations x8) | 3.06 | 17.7% | 1.0 | 11.78 | `tl::cp_async_wait<2>();` |
| wait.sync#2*8 (est.: sampled iterations x8) | 2.53 | 14.6% | 1.0 | 9.22 | `__syncthreads();` |
| compute.mma*8 (est.: sampled iterations x8) | 2.34 | 13.5% | 1.0 | 2.56 | `{` |
| load.cp_async#2*8 (est.: sampled iterations x8) | 1.09 | 6.3% | 1.0 | 1.28 | `for (int i_5 = 0; i_5 < 2; ++i_5) \| for (int i_5 = 0; i_5 < 4; ++i_5) \| tl::cp_async_gs_co` |
| wait.cp_async#3 | 1.08 | 6.2% | 1.0 | 3.26 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>();` |
| wait.sync#4 | 0.52 | 3.0% | 1.0 | 1.79 | `__syncthreads();` |
| wait.sync*8 (est.: sampled iterations x8) | 0.38 | 2.2% | 1.0 | 0.51 | `__syncthreads();` |
| wait.cp_async#2 | 0.36 | 2.1% | 1.0 | 1.54 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| wait.sync#3 | 0.34 | 2.0% | 1.0 | 1.25 | `__syncthreads();` |
| compute.mma#2 | 0.30 | 1.7% | 1.0 | 0.32 | `{` |
| compute.mma#3 | 0.30 | 1.7% | 1.0 | 0.32 | `{` |
| store.global | 0.20 | 1.2% | 1.0 | 0.26 | `for (int i_10 = 0; i_10 < 16; ++i_10) \| for (int i_10 = 0; i_10 < 8; ++i_10) \| for (int i_` |
| load.cp_async | 0.19 | 1.1% | 1.0 | 0.22 | `for (int i_1 = 0; i_1 < 2; ++i_1) \| for (int i_1 = 0; i_1 < 4; ++i_1) \| for (int i_1 = 0; ` |

### gemm_fp8w_bf16_partials_kernel grid (48, 4, 1) block (128, 1, 1)

153 launches, 768 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 89.60 us, warp 55.33 us, nsys graphs 66.02 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 30.83 | 55.7% | 7.84 | 25.9% |
| wait | 14.55 | 26.3% | 12.99 | 42.9% |
| compute | 5.62 | 10.2% | 5.25 | 17.4% |
| load | 4.15 | 7.5% | 3.98 | 13.2% |
| store | 0.19 | 0.3% | 0.18 | 0.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*14 (est.: sampled iterations x14) | 4.98 | 9.0% | 1.0 | 6.27 | `{` |
| wait.cp_async*14 (est.: sampled iterations x14) | 4.26 | 7.7% | 1.0 | 18.37 | `tl::cp_async_wait<3>();` |
| wait.sync#2*14 (est.: sampled iterations x14) | 3.28 | 5.9% | 1.0 | 12.10 | `__syncthreads();` |
| load.cp_async#3*14 (est.: sampled iterations x14) | 1.29 | 2.3% | 1.0 | 1.79 | `for (int i_10 = 0; i_10 < 4; ++i_10) \| for (int i_10 = 0; i_10 < 6; ++i_10) \| for (int i_1` |
| wait.cp_async#3 | 1.11 | 2.0% | 1.0 | 3.04 | `tl::cp_async_wait<3>();` |
| load.global*14 (est.: sampled iterations x14) | 1.06 | 1.9% | 1.0 | 1.79 | `if (((int)threadIdx.x) < 4)` |
| load.cp_async#2*14 (est.: sampled iterations x14) | 1.04 | 1.9% | 1.0 | 1.34 | `for (int i_4 = 0; i_4 < 2; ++i_4) \| for (int i_6 = 0; i_6 < 2; ++i_6)` |
| wait.sync#4*14 (est.: sampled iterations x14) | 0.98 | 1.8% | 1.0 | 1.34 | `__syncthreads();` |
| wait.sync*14 (est.: sampled iterations x14) | 0.98 | 1.8% | 1.0 | 1.34 | `__syncthreads();` |
| wait.sync#5*14 (est.: sampled iterations x14) | 0.95 | 1.7% | 1.0 | 1.34 | `__syncthreads();` |
| wait.sync#3*14 (est.: sampled iterations x14) | 0.92 | 1.7% | 1.0 | 1.34 | `__syncthreads();` |
| wait.cp_async#2*14 (est.: sampled iterations x14) | 0.84 | 1.5% | 1.0 | 0.90 | `tl::cp_async_wait<3>();` |
| load.cp_async | 0.63 | 1.1% | 1.0 | 1.66 | `for (int i_1 = 0; i_1 < 2; ++i_1)` |
| wait.sync#7 | 0.39 | 0.7% | 1.0 | 1.38 | `__syncthreads();` |
| compute.mma#2 | 0.33 | 0.6% | 1.0 | 0.45 | `{` |
| compute.mma#3 | 0.30 | 0.5% | 1.0 | 0.38 | `{` |
| wait.cp_async#5 | 0.29 | 0.5% | 1.0 | 1.44 | `tl::cp_async_wait<1>();` |
| wait.sync#10 | 0.23 | 0.4% | 1.0 | 0.86 | `__syncthreads();` |
| store.global | 0.19 | 0.3% | 1.0 | 0.26 | `for (int i_13 = 0; i_13 < 8; ++i_13) \| for (int i_19 = 0; i_19 < 16; ++i_19) \| for (int i_` |
| load.global#3 | 0.06 | 0.1% | 1.0 | 0.10 | `if (((int)threadIdx.x) < 4)` |
| load.global#2 | 0.06 | 0.1% | 1.0 | 0.10 | `if (((int)threadIdx.x) < 4)` |
| wait.cp_async#6 | 0.06 | 0.1% | 1.0 | 0.06 | `tl::cp_async_wait<0>();` |
| wait.sync#9 | 0.06 | 0.1% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#6 | 0.05 | 0.1% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#8 | 0.05 | 0.1% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#11 | 0.05 | 0.1% | 1.0 | 0.06 | `__syncthreads();` |
| wait.cp_async#4 | 0.05 | 0.1% | 1.0 | 0.06 | `tl::cp_async_wait<2>();` |

### gemm_mxfp8_kernel grid (28, 1, 1) block (128, 1, 1)

153 launches, 112 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 41.54 us, warp 39.58 us, nsys graphs 41.66 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 13.19 | 33.3% | 18.71 | 41.9% |
| unattributed | 11.81 | 29.8% | 11.34 | 25.4% |
| compute | 10.69 | 27.0% | 10.59 | 23.7% |
| load | 3.72 | 9.4% | 3.85 | 8.6% |
| store | 0.17 | 0.4% | 0.16 | 0.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*19 (est.: sampled iterations x19) | 10.18 | 25.7% | 1.0 | 10.34 | `{` |
| wait.cp_async*19 (est.: sampled iterations x19) | 6.16 | 15.6% | 1.0 | 21.89 | `tl::cp_async_wait<1>();` |
| wait.sync#2*19 (est.: sampled iterations x19) | 4.94 | 12.5% | 1.0 | 15.81 | `__syncthreads();` |
| load.cp_async#2*19 (est.: sampled iterations x19) | 3.50 | 8.8% | 1.0 | 3.65 | `for (int i_3 = 0; i_3 < 2; ++i_3) \| for (int i_3 = 0; i_3 < 8; ++i_3)` |
| wait.sync*19 (est.: sampled iterations x19) | 1.22 | 3.1% | 1.0 | 1.82 | `__syncthreads();` |
| wait.cp_async#2 | 0.61 | 1.5% | 1.0 | 1.50 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| compute.mma#2 | 0.50 | 1.3% | 1.0 | 0.51 | `{` |
| wait.sync#3 | 0.26 | 0.7% | 1.0 | 0.77 | `__syncthreads();` |
| load.cp_async | 0.23 | 0.6% | 1.0 | 0.26 | `for (int i_1 = 0; i_1 < 2; ++i_1) \| for (int i_1 = 0; i_1 < 4; ++i_1) \| for (int i_1 = 0; ` |
| store.global | 0.17 | 0.4% | 1.0 | 0.22 | `for (int i_10 = 0; i_10 < 16; ++i_10) \| for (int i_10 = 0; i_10 < 8; ++i_10) \| for (int i_` |

### gemm_mxfp8_kernel grid (24, 1, 1) block (128, 1, 1)

153 launches, 96 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 33.06 us, warp 31.93 us, nsys graphs 39.97 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| compute | 13.14 | 41.1% | 13.10 | 33.8% |
| wait | 12.29 | 38.5% | 16.56 | 42.7% |
| load | 5.41 | 16.9% | 6.34 | 16.3% |
| unattributed | 0.90 | 2.8% | 2.63 | 6.8% |
| store | 0.20 | 0.6% | 0.18 | 0.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*38 (est.: sampled iterations x38) | 12.54 | 39.3% | 1.0 | 13.38 | `{` |
| wait.sync#2*38 (est.: sampled iterations x38) | 5.78 | 18.1% | 1.0 | 15.81 | `__syncthreads();` |
| load.cp_async#2*38 (est.: sampled iterations x38) | 5.09 | 15.9% | 1.0 | 6.08 | `for (int i_5 = 0; i_5 < 2; ++i_5) \| for (int i_5 = 0; i_5 < 4; ++i_5) \| tl::cp_async_gs_co` |
| wait.cp_async*38 (est.: sampled iterations x38) | 3.95 | 12.4% | 1.0 | 17.02 | `tl::cp_async_wait<2>();` |
| wait.sync*38 (est.: sampled iterations x38) | 2.36 | 7.4% | 1.0 | 2.43 | `__syncthreads();` |
| load.cp_async | 0.31 | 1.0% | 1.0 | 0.35 | `for (int i_1 = 0; i_1 < 2; ++i_1) \| for (int i_1 = 0; i_1 < 4; ++i_1) \| for (int i_1 = 0; ` |
| compute.mma#2 | 0.30 | 0.9% | 1.0 | 0.32 | `{` |
| compute.mma#3 | 0.30 | 0.9% | 1.0 | 0.32 | `{` |
| store.global | 0.20 | 0.6% | 1.0 | 0.22 | `for (int i_10 = 0; i_10 < 16; ++i_10) \| for (int i_10 = 0; i_10 < 8; ++i_10) \| for (int i_` |
| wait.cp_async#3 | 0.06 | 0.2% | 1.0 | 0.06 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>();` |
| wait.sync#3 | 0.05 | 0.2% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#4 | 0.05 | 0.2% | 1.0 | 0.06 | `__syncthreads();` |
| wait.cp_async#2 | 0.05 | 0.1% | 1.0 | 0.06 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |

### sparse_mla_kernel grid (1, 48, 1) block (256, 1, 1)

68 launches, 384 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 77.46 us, warp 64.77 us, nsys graphs 51.97 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed, with loops' unsampled iterations | 56.90 | 87.8% | 58.64 | 84.8% |
| load | 4.85 | 7.5% | 7.36 | 10.6% |
| compute | 1.49 | 2.3% | 1.50 | 2.2% |
| wait | 1.37 | 2.1% | 1.29 | 1.9% |
| store | 0.17 | 0.3% | 0.38 | 0.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global (sampled iterations) | 4.48 | 6.9% | 4.0 | 6.05 | `for (int c = 0; c < 4; ++c) \| for (int i = 0; i < 8; ++i) \| for (int i_1 = 0; i_1 < 8; ++i` |
| wait.sync (sampled iterations) | 1.30 | 2.0% | 18.0 | 1.47 | `__syncthreads();` |
| compute.mma (sampled iterations) | 1.21 | 1.9% | 2.0 | 1.28 | `for (int ki = 0; ki < 32; ++ki) \| for (int ki_1 = 0; ki_1 < 2; ++ki_1)` |
| load.cp_async (sampled iterations) | 0.37 | 0.6% | 2.0 | 0.45 | `for (int k_1 = 0; k_1 < 5; ++k_1) \| for (int k_1 = 0; k_1 < 9; ++k_1) \| for (int k_3 = 0; ` |
| compute.reduce (sampled iterations) | 0.27 | 0.4% | 2.0 | 0.32 | `m_g[i_5] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(m_g[i_5], (&((` |
| store.global | 0.17 | 0.3% | 1.0 | 0.22 | `for (int i_10 = 0; i_10 < 32; ++i_10) \| for (int i_11 = 0; i_11 < 32; ++i_11) \| for (int i` |
| wait.cp_async (sampled iterations) | 0.07 | 0.1% | 1.0 | 0.10 | `tl::cp_async_wait<0>();` |

### moe_grouped_fp8_fp4_gemm_kernel grid (40, 120, 1) block (128, 1, 1)

51 launches, 19200 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 86.59 us, warp 3.25 us, nsys graphs 60.43 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.61 | 49.5% | 0.22 | 68.2% |
| wait | 0.65 | 19.9% | 0.05 | 14.4% |
| compute | 0.49 | 15.2% | 0.00 | 0.0% |
| load | 0.40 | 12.4% | 0.06 | 17.4% |
| store | 0.10 | 3.0% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*6 (est.: sampled iterations x6) | 0.49 | 15.2% | 0.2 | 2.11 | `{` |
| load#2*6 (est.: sampled iterations x6) | 0.22 | 6.7% | 0.2 | 1.15 | `if (ko < 4)` |
| wait.sync#5*6 (est.: sampled iterations x6) | 0.17 | 5.2% | 0.2 | 0.77 | `__syncthreads();` |
| wait.mbarrier*6 (est.: sampled iterations x6) | 0.12 | 3.7% | 0.2 | 0.58 | `loaded[(ko % 3)].wait((ko / 3));` |
| store.global | 0.10 | 3.0% | 0.2 | 0.42 | `for (int i_2 = 0; i_2 < 16; ++i_2) \| for (int i_4 = 0; i_4 < 32; ++i_4)` |
| wait.sync#4*6 (est.: sampled iterations x6) | 0.10 | 3.0% | 0.2 | 0.38 | `__syncthreads();` |
| wait.sync#6*6 (est.: sampled iterations x6) | 0.10 | 3.0% | 0.2 | 0.38 | `__syncthreads();` |
| wait.cp_async*6 (est.: sampled iterations x6) | 0.08 | 2.6% | 0.2 | 0.38 | `tl::cp_async_wait<2>();` |
| load.global | 0.06 | 2.0% | 1.0 | 0.10 | `int expert = TileExpert[((int64_t)((int)blockIdx.y))];` |
| load.global#3 | 0.06 | 1.7% | 0.2 | 0.22 | `if (((expert < 128) && (0 <= expert))) \| if (((expert < 384) && (0 <= expert)))` |
| load | 0.04 | 1.4% | 0.2 | 0.22 | `for (int s = 0; s < 2; ++s)` |
| wait.sync | 0.04 | 1.3% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#2 | 0.03 | 1.0% | 0.2 | 0.16 | `__syncthreads();` |
| load.global#2 | 0.02 | 0.7% | 0.2 | 0.19 | `if (((int)threadIdx.x) < 32) \| if (((int)threadIdx.x) < 64)` |
| wait.sync#3 | 0.01 | 0.2% | 0.2 | 0.03 | `__syncthreads();` |

### moe_grouped_fp8_fp4_gemm_kernel grid (12, 120, 1) block (128, 1, 1)

51 launches, 5760 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 76.96 us, warp 12.04 us, nsys graphs 65.97 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 4.09 | 33.9% | 0.05 | 16.5% |
| compute | 3.44 | 28.6% | 0.00 | 0.0% |
| unattributed | 2.72 | 22.6% | 0.20 | 64.3% |
| load | 1.72 | 14.3% | 0.06 | 19.2% |
| store | 0.07 | 0.6% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*40 (est.: sampled iterations x40) | 3.44 | 28.6% | 0.2 | 14.08 | `{` |
| load#2*40 (est.: sampled iterations x40) | 1.52 | 12.6% | 0.2 | 8.96 | `if (ko < 38)` |
| wait.sync#5*40 (est.: sampled iterations x40) | 1.28 | 10.6% | 0.2 | 6.40 | `__syncthreads();` |
| wait.mbarrier*40 (est.: sampled iterations x40) | 0.88 | 7.3% | 0.2 | 3.84 | `loaded[(ko % 3)].wait(((ko % 6) / 3));` |
| wait.sync#4*40 (est.: sampled iterations x40) | 0.64 | 5.3% | 0.2 | 2.56 | `__syncthreads();` |
| wait.sync#6*40 (est.: sampled iterations x40) | 0.64 | 5.3% | 0.2 | 2.56 | `__syncthreads();` |
| wait.cp_async*40 (est.: sampled iterations x40) | 0.56 | 4.7% | 0.2 | 2.56 | `tl::cp_async_wait<2>();` |
| store.global | 0.07 | 0.6% | 0.2 | 0.32 | `for (int i_2 = 0; i_2 < 16; ++i_2) \| for (int i_4 = 0; i_4 < 32; ++i_4)` |
| load.global | 0.07 | 0.5% | 1.0 | 0.10 | `int expert = TileExpert[((int64_t)((int)blockIdx.y))];` |
| load.global#3 | 0.06 | 0.5% | 0.2 | 0.26 | `if (((expert < 128) && (0 <= expert))) \| if (((expert < 384) && (0 <= expert)))` |
| load | 0.05 | 0.4% | 0.2 | 0.26 | `for (int s = 0; s < 2; ++s)` |
| wait.sync | 0.05 | 0.4% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#2 | 0.03 | 0.3% | 0.2 | 0.16 | `__syncthreads();` |
| load.global#2 | 0.02 | 0.2% | 0.2 | 0.19 | `if (((int)threadIdx.x) < 32) \| if (((int)threadIdx.x) < 64)` |
| wait.sync#3 | 0.01 | 0.1% | 0.2 | 0.03 | `__syncthreads();` |

### partial_argmax_kernel grid (8, 64, 1) block (128, 1, 1)

85 launches, 2048 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 30.78 us, warp 23.49 us, nsys graphs 26.98 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 10.67 | 45.4% | 7.29 | 25.3% |
| load | 8.80 | 37.4% | 20.10 | 69.9% |
| compute | 2.57 | 10.9% | 0.64 | 2.2% |
| wait | 1.06 | 4.5% | 0.54 | 1.9% |
| store | 0.39 | 1.7% | 0.19 | 0.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 8.80 | 37.4% | 1.0 | 21.02 | `for (int j = 0; j < 16; ++j) \| for (int run = 0; run < 2; ++run)` |
| compute.reduce | 2.57 | 10.9% | 1.0 | 5.89 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| wait.sync | 1.06 | 4.5% | 1.0 | 3.78 | `__syncthreads();` |
| store.global | 0.39 | 1.7% | 1.0 | 1.86 | `if (((int)threadIdx.x) == 0)` |

### gemm_bf16_splitk_kernel grid (16, 5, 1) block (256, 1, 1)

34 launches, 640 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 71.20 us, warp 64.75 us, nsys graphs 75.20 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 33.33 | 51.5% | 0.00 | 0.0% |
| wait | 24.62 | 38.0% | 63.11 | 93.5% |
| compute | 4.04 | 6.2% | 4.45 | 6.6% |
| load | 2.57 | 4.0% | 7.05 | 10.4% |
| store | 0.19 | 0.3% | 0.70 | 1.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.cp_async*13 (est.: sampled iterations x13) | 11.13 | 17.2% | 0.5 | 34.74 | `tl::cp_async_wait<2>();` |
| wait.sync*13 (est.: sampled iterations x13) | 5.49 | 8.5% | 0.5 | 13.94 | `__syncthreads();` |
| compute.mma*13 (est.: sampled iterations x13) | 1.98 | 3.1% | 0.5 | 2.08 | `{` |
| compute.mma*14 (est.: sampled iterations x14) | 1.47 | 2.3% | 0.5 | 1.57 | `{` |
| wait.sync*14 (est.: sampled iterations x14) | 1.14 | 1.8% | 0.5 | 3.58 | `__syncthreads();` |
| wait.cp_async#2 | 1.04 | 1.6% | 1.0 | 4.14 | `tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>(); \| tl::cp_async_wait<3>();` |
| wait.sync#3 | 1.00 | 1.5% | 1.0 | 2.64 | `__syncthreads();` |
| wait.sync#4 | 0.92 | 1.4% | 1.0 | 2.46 | `__syncthreads();` |
| load.cp_async | 0.91 | 1.4% | 1.0 | 3.73 | `for (int i_1 = 0; i_1 < 2; ++i_1) \| tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)A_sha` |
| load.cp_async#2*13 (est.: sampled iterations x13) | 0.76 | 1.2% | 0.5 | 1.04 | `for (int i_10 = 0; i_10 < 2; ++i_10) \| for (int i_8 = 0; i_8 < 2; ++i_8) \| tl::cp_async_gs` |
| wait.cp_async#3 | 0.73 | 1.1% | 1.0 | 3.34 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| load.cp_async#2*14 (est.: sampled iterations x14) | 0.67 | 1.0% | 0.5 | 0.90 | `for (int i_6 = 0; i_6 < 2; ++i_6) \| tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)A_sha` |
| wait.sync#5 | 0.65 | 1.0% | 1.0 | 2.19 | `__syncthreads();` |
| wait.cp_async#4 | 0.58 | 0.9% | 0.5 | 1.87 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>();` |
| wait.cp_async*14 (est.: sampled iterations x14) | 0.50 | 0.8% | 0.5 | 3.36 | `tl::cp_async_wait<1>();` |
| wait.sync#2*14 (est.: sampled iterations x14) | 0.47 | 0.7% | 0.5 | 0.67 | `__syncthreads();` |
| wait.sync#2*13 (est.: sampled iterations x13) | 0.42 | 0.7% | 0.5 | 0.62 | `__syncthreads();` |
| wait.sync#6 | 0.37 | 0.6% | 1.0 | 1.66 | `__syncthreads();` |
| compute.mma#2 | 0.23 | 0.4% | 1.0 | 0.29 | `{` |
| compute.mma#3 | 0.23 | 0.4% | 1.0 | 0.29 | `{` |
| load.global | 0.22 | 0.3% | 1.0 | 2.21 | `if (((int)threadIdx.x) == 0)` |
| wait.sync#7 | 0.18 | 0.3% | 0.5 | 0.21 | `__syncthreads();` |
| compute.mma#4 | 0.13 | 0.2% | 0.5 | 0.14 | `{` |
| store.global | 0.12 | 0.2% | 1.0 | 0.21 | `for (int i_10 = 0; i_10 < 4; ++i_10) \| for (int i_16 = 0; i_16 < 4; ++i_16) \| for (int i_1` |
| store.global#3 | 0.07 | 0.1% | 1.0 | 0.10 | `if (((int*)last)[0] == 1)` |
| load.global#2*16 (est.: sampled iterations x16) | 0.00 | 0.0% | 0.0 | 0.00 | `acc = P[(((((((int64_t)((int)blockIdx.z)) * (int64_t)65536) + (((int64_t)i_26) * (int64_t)` |
| load.global#2*8 (est.: sampled iterations x8) | 0.00 | 0.0% | 0.0 | 0.00 | `acc = P[(((((((int64_t)((int)blockIdx.z)) * (int64_t)16384) + (((int64_t)i_17) * (int64_t)` |
| store.global#2*16 (est.: sampled iterations x16) | 0.00 | 0.0% | 0.0 | 0.00 | `if (((int)blockIdx.x) < 8)` |
| store.global#2*8 (est.: sampled iterations x8) | 0.00 | 0.0% | 0.0 | 0.00 | `C0[(((((((int64_t)((int)blockIdx.z)) * (int64_t)16384) + (((int64_t)i_18) * (int64_t)2048)` |

### gemm_bf16_partials_kernel grid (6, 8, 1) block (256, 1, 1)

102 launches, 384 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 20.05 us, warp 8.39 us, nsys graphs 18.43 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 3.44 | 41.0% | 3.33 | 34.9% |
| wait | 2.96 | 35.2% | 2.97 | 31.2% |
| compute | 1.49 | 17.8% | 3.01 | 31.5% |
| load | 0.39 | 4.6% | 0.00 | 0.0% |
| store | 0.12 | 1.4% | 0.23 | 2.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*10 (est.: sampled iterations x10) | 1.49 | 17.8% | 0.5 | 3.20 | `{` |
| wait.mbarrier*10 (est.: sampled iterations x10) | 1.49 | 17.8% | 0.5 | 3.52 | `mbarrier[((kb % 3) + 3)].wait((((kb % 6) / 3) ^ 1));` |
| wait.mbarrier#2*10 (est.: sampled iterations x10) | 1.36 | 16.2% | 0.5 | 6.72 | `mbarrier[(kb_1 % 3)].wait(((kb_1 % 6) / 3));` |
| load.tma*10 (est.: sampled iterations x10) | 0.39 | 4.6% | 0.5 | 1.28 | `if (tl::tl_shuffle_elect<128>())` |
| store.global | 0.12 | 1.4% | 0.5 | 0.29 | `for (int i_1 = 0; i_1 < 4; ++i_1) \| for (int i_2 = 0; i_2 < 8; ++i_2) \| for (int i_4 = 0; ` |
| wait.sync | 0.11 | 1.3% | 1.0 | 0.16 | `__syncthreads();` |

### mhc_finalize_kernel grid (48, 1, 1) block (128, 1, 1)

204 launches, 192 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 6.56 us, warp 5.81 us, nsys graphs 9.22 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 2.20 | 37.9% | 3.24 | 44.0% |
| compute | 1.00 | 17.1% | 0.64 | 8.7% |
| store | 1.00 | 17.1% | 1.30 | 17.6% |
| load | 0.98 | 16.9% | 1.28 | 17.4% |
| wait | 0.63 | 10.9% | 0.90 | 12.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync | 0.63 | 10.9% | 1.0 | 0.86 | `__syncthreads();` |
| compute.reduce | 0.58 | 9.9% | 0.2 | 2.30 | `row_max[0] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<32>>::run(row_max[0]);` |
| store.global#4*7 (est.: sampled iterations x7) | 0.56 | 9.6% | 0.8 | 0.90 | `if ((((int)threadIdx.x) % 4) == 0)` |
| compute.reduce#2*7 (est.: sampled iterations x7) | 0.42 | 7.2% | 0.8 | 0.67 | `amax[0] = max(amax[0], __shfl_xor_sync((uint)4294967295, amax[0], 1, 32));` |
| load.global#4*7 (est.: sampled iterations x7) | 0.37 | 6.4% | 0.8 | 0.67 | `*(uint4*)(yv + 0) = *(uint4*)(C + ((((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((i` |
| store.global#3*7 (est.: sampled iterations x7) | 0.32 | 5.5% | 0.8 | 0.45 | `*(uint4*)(y + ((((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((int64_t)step) * (int6` |
| load.global | 0.31 | 5.4% | 1.0 | 0.93 | `if (((int)threadIdx.x) < 32)` |
| load.global#3 | 0.25 | 4.3% | 0.8 | 0.35 | `for (int i_split_2 = 0; i_split_2 < 40; ++i_split_2)` |
| store.global | 0.10 | 1.7% | 0.2 | 0.42 | `if (((int)threadIdx.x) < 4)` |
| load.global#2 | 0.05 | 0.8% | 0.2 | 0.19 | `cm[0] = ((mixes_shared[((((int)threadIdx.x) & 15) + 8)] * scale[2]) + base[((((int)threadI` |
| store.global#2 | 0.01 | 0.2% | 0.2 | 0.06 | `if ((((int)threadIdx.x) >> 4) == 0)` |

### sparse_mla_kernel grid (2, 1, 40) block (256, 1, 1)

51 launches, 640 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 27.17 us, warp 15.22 us, nsys graphs 23.65 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| compute | 6.96 | 45.7% | 6.96 | 46.1% |
| wait | 3.02 | 19.8% | 2.98 | 19.8% |
| unattributed, with loops' unsampled iterations | 2.68 | 17.6% | 2.46 | 16.3% |
| load | 1.87 | 12.3% | 1.85 | 12.3% |
| store | 0.68 | 4.5% | 0.83 | 5.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*4 (est.: sampled iterations x4) | 2.72 | 17.9% | 1.0 | 2.82 | `for (int ki = 0; ki < 32; ++ki) \| {` |
| compute.mma#2*4 (est.: sampled iterations x4) | 2.20 | 14.4% | 1.0 | 2.30 | `for (int ki_1 = 0; ki_1 < 2; ++ki_1) \| if (((((int)blockIdx.x) * 2) + (c_1 >> 1)) < 3)` |
| compute.reduce#2*8 (est.: sampled iterations x8) | 1.02 | 6.7% | 1.0 | 1.28 | `row_sum[i_7] = tl::AllReduce<tl::SumOp, 128, 32, 0, tl::NamedBarrier<128>>::run(row_sum[i_` |
| compute.reduce*8 (est.: sampled iterations x8) | 1.02 | 6.7% | 1.0 | 1.02 | `m_g[i_4] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(m_g[i_4], (&((` |
| load.global#2 | 0.92 | 6.0% | 1.0 | 1.09 | `for (int i = 0; i < 8; ++i)` |
| store.global | 0.68 | 4.5% | 1.0 | 0.86 | `for (int i_10 = 0; i_10 < 32; ++i_10) \| for (int i_11 = 0; i_11 < 32; ++i_11) \| for (int i` |
| load.cp_async#2*4 (est.: sampled iterations x4) | 0.64 | 4.2% | 1.0 | 0.77 | `if (((((int)blockIdx.x) * 2) + (c_1 >> 1)) < 3) \| if ((c_1 < 3) & (c_1 < 3)) \| if (c < 3)` |
| wait.sync#10*8 (est.: sampled iterations x8) | 0.38 | 2.5% | 1.0 | 0.51 | `__syncthreads();` |
| wait.sync#8*8 (est.: sampled iterations x8) | 0.38 | 2.5% | 1.0 | 0.51 | `__syncthreads();` |
| wait.sync#5*4 (est.: sampled iterations x4) | 0.31 | 2.0% | 1.0 | 0.38 | `__syncthreads();` |
| wait.sync#7*4 (est.: sampled iterations x4) | 0.30 | 2.0% | 1.0 | 0.38 | `__syncthreads();` |
| wait.cp_async*4 (est.: sampled iterations x4) | 0.28 | 1.8% | 1.0 | 0.38 | `tl::cp_async_wait<0>();` |
| wait.sync#4*4 (est.: sampled iterations x4) | 0.28 | 1.8% | 1.0 | 0.38 | `__syncthreads();` |
| wait.sync#6*4 (est.: sampled iterations x4) | 0.26 | 1.7% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#9*4 (est.: sampled iterations x4) | 0.26 | 1.7% | 1.0 | 0.38 | `__syncthreads();` |
| wait.sync#11*4 (est.: sampled iterations x4) | 0.25 | 1.7% | 1.0 | 0.26 | `__syncthreads();` |
| load.cp_async (sampled iterations) | 0.21 | 1.4% | 1.0 | 0.26 | `for (int k_1 = 0; k_1 < 5; ++k_1) \| for (int k_1 = 0; k_1 < 9; ++k_1) \| for (int k_3 = 0; ` |
| wait.sync (sampled iterations) | 0.20 | 1.3% | 1.0 | 0.26 | `__syncthreads();` |
| load.global (sampled iterations) | 0.10 | 0.7% | 1.0 | 0.29 | `for (int c = 0; c < 4; ++c) \| for (int i = 0; i < 8; ++i) \| for (int i_1 = 0; i_1 < 8; ++i` |
| wait.sync#3 | 0.08 | 0.5% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#2 | 0.05 | 0.3% | 1.0 | 0.06 | `__syncthreads();` |

### moe_swiglu_cast_kernel_kernel grid (16, 288, 1) block (96, 1, 1)

102 launches, 13824 warps; ranges, warp and last warp from the 4 recording CTAs' 12 warps; span 10.26 us, warp 1.64 us, nsys graphs 5.38 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.17 | 71.1% | 0.26 | 37.2% |
| load | 0.39 | 24.0% | 0.45 | 62.8% |
| compute | 0.04 | 2.6% | 0.00 | 0.0% |
| store | 0.04 | 2.3% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.39 | 24.0% | 1.0 | 0.74 | `for (int r = 0; r < 2; ++r)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.04 | 2.6% | 0.2 | 0.19 | `if (0 <= orders[r_2])` |
| store.global*2 (est.: sampled iterations x2) | 0.04 | 2.3% | 0.2 | 0.19 | `if (0 <= orders[r_2])` |

### sparse_mla_kernel grid (1, 1, 48) block (256, 1, 1)

34 launches, 384 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 31.23 us, warp 19.72 us, nsys graphs 29.76 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| compute | 6.92 | 35.1% | 6.87 | 33.6% |
| load | 4.95 | 25.1% | 5.05 | 24.7% |
| wait | 3.05 | 15.5% | 3.22 | 15.7% |
| unattributed, with loops' unsampled iterations | 3.01 | 15.3% | 3.44 | 16.8% |
| store | 1.79 | 9.1% | 1.89 | 9.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 | 4.00 | 20.3% | 1.0 | 4.66 | `for (int i = 0; i < 8; ++i)` |
| compute.mma*4 (est.: sampled iterations x4) | 2.72 | 13.8% | 1.0 | 2.82 | `for (int ki = 0; ki < 32; ++ki) \| {` |
| compute.mma#2*4 (est.: sampled iterations x4) | 2.16 | 10.9% | 1.0 | 2.30 | `for (int ki_1 = 0; ki_1 < 2; ++ki_1) \| if (((((int)blockIdx.x) * 2) + (c_1 >> 1)) < 3)` |
| store.global | 1.79 | 9.1% | 1.0 | 2.69 | `for (int i_10 = 0; i_10 < 32; ++i_10) \| for (int i_11 = 0; i_11 < 32; ++i_11) \| for (int i` |
| compute.reduce#2*8 (est.: sampled iterations x8) | 1.02 | 5.2% | 1.0 | 1.28 | `row_sum[i_7] = tl::AllReduce<tl::SumOp, 128, 32, 0, tl::NamedBarrier<128>>::run(row_sum[i_` |
| compute.reduce*8 (est.: sampled iterations x8) | 1.02 | 5.2% | 1.0 | 1.02 | `m_g[i_4] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(m_g[i_4], (&((` |
| load.cp_async#2*4 (est.: sampled iterations x4) | 0.62 | 3.1% | 1.0 | 0.77 | `if (((((int)blockIdx.x) * 2) + (c_1 >> 1)) < 3) \| if ((c_1 < 3) & (c_1 < 3)) \| if (c < 3)` |
| wait.sync#10*8 (est.: sampled iterations x8) | 0.38 | 1.9% | 1.0 | 0.51 | `__syncthreads();` |
| wait.sync#8*8 (est.: sampled iterations x8) | 0.38 | 1.9% | 1.0 | 0.51 | `__syncthreads();` |
| wait.sync#5*4 (est.: sampled iterations x4) | 0.32 | 1.6% | 1.0 | 0.38 | `__syncthreads();` |
| wait.sync#7*4 (est.: sampled iterations x4) | 0.29 | 1.5% | 1.0 | 0.38 | `__syncthreads();` |
| wait.sync#4*4 (est.: sampled iterations x4) | 0.28 | 1.4% | 1.0 | 0.38 | `__syncthreads();` |
| wait.cp_async*4 (est.: sampled iterations x4) | 0.28 | 1.4% | 1.0 | 0.38 | `tl::cp_async_wait<0>();` |
| wait.sync#9*4 (est.: sampled iterations x4) | 0.26 | 1.3% | 1.0 | 0.38 | `__syncthreads();` |
| wait.sync#11*4 (est.: sampled iterations x4) | 0.25 | 1.3% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#6*4 (est.: sampled iterations x4) | 0.25 | 1.3% | 1.0 | 0.26 | `__syncthreads();` |
| load.cp_async (sampled iterations) | 0.24 | 1.2% | 1.0 | 0.29 | `for (int k_1 = 0; k_1 < 5; ++k_1) \| for (int k_1 = 0; k_1 < 9; ++k_1) \| for (int k_3 = 0; ` |
| wait.sync (sampled iterations) | 0.21 | 1.1% | 1.0 | 0.26 | `__syncthreads();` |
| load.global (sampled iterations) | 0.09 | 0.5% | 1.0 | 0.26 | `for (int c = 0; c < 4; ++c) \| for (int i = 0; i < 8; ++i) \| for (int i_1 = 0; i_1 < 8; ++i` |
| wait.sync#3 | 0.09 | 0.4% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#2 | 0.05 | 0.3% | 1.0 | 0.06 | `__syncthreads();` |

### mhc_post_kernel grid (240, 1, 1) block (128, 1, 1)

68 launches, 960 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 15.58 us, warp 7.50 us, nsys graphs 10.08 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 7.26 | 96.7% | 9.73 | 88.9% |
| load | 0.17 | 2.3% | 0.75 | 6.8% |
| store | 0.08 | 1.0% | 0.46 | 4.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.17 | 2.3% | 1.0 | 0.22 | `*(float4*)(pm + 0) = *(float4*)(post + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int` |
| store.global | 0.08 | 1.0% | 1.0 | 0.48 | `for (int n_1 = 0; n_1 < 4; ++n_1)` |

### gemm_mxfp8_kernel grid (64, 1, 1) block (64, 1, 1)

34 launches, 128 warps; ranges, warp and last warp from the 4 recording CTAs' 8 warps; span 25.41 us, warp 23.54 us, nsys graphs 26.74 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 18.13 | 77.0% | 20.37 | 81.3% |
| compute | 5.10 | 21.6% | 5.12 | 20.4% |
| load | 1.26 | 5.3% | 1.28 | 5.1% |
| store | 0.18 | 0.8% | 0.21 | 0.8% |
| unattributed | 0.00 | 0.0% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.cp_async*9 (est.: sampled iterations x9) | 14.94 | 63.5% | 1.0 | 23.18 | `tl::cp_async_wait<1>();` |
| compute.mma*9 (est.: sampled iterations x9) | 4.61 | 19.6% | 1.0 | 4.61 | `{` |
| wait.sync#2*9 (est.: sampled iterations x9) | 1.94 | 8.3% | 1.0 | 7.20 | `__syncthreads();` |
| load.cp_async#2*9 (est.: sampled iterations x9) | 1.08 | 4.6% | 1.0 | 1.15 | `for (int i_3 = 0; i_3 < 8; ++i_3)` |
| wait.cp_async#2 | 0.64 | 2.7% | 1.0 | 2.05 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| wait.sync*9 (est.: sampled iterations x9) | 0.50 | 2.1% | 1.0 | 0.58 | `__syncthreads();` |
| compute.mma#2 | 0.49 | 2.1% | 1.0 | 0.51 | `{` |
| store.global | 0.18 | 0.8% | 1.0 | 0.22 | `for (int i_10 = 0; i_10 < 16; ++i_10) \| for (int i_10 = 0; i_10 < 8; ++i_10) \| for (int i_` |
| load.cp_async | 0.18 | 0.7% | 1.0 | 0.22 | `for (int i_1 = 0; i_1 < 2; ++i_1) \| for (int i_1 = 0; i_1 < 4; ++i_1) \| for (int i_1 = 0; ` |
| wait.sync#3 | 0.11 | 0.5% | 1.0 | 0.45 | `__syncthreads();` |

### gemm_bf16_partials_kernel grid (2, 20, 1) block (256, 1, 1)

51 launches, 320 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 14.59 us, warp 6.89 us, nsys graphs 11.94 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 5.06 | 73.5% | 6.57 | 71.3% |
| wait | 1.05 | 15.3% | 1.29 | 14.0% |
| compute | 0.55 | 8.0% | 1.12 | 12.1% |
| load | 0.13 | 1.9% | 0.00 | 0.0% |
| store | 0.09 | 1.4% | 0.24 | 2.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.mbarrier#2*4 (est.: sampled iterations x4) | 0.82 | 11.9% | 0.5 | 3.58 | `mbarrier[(kb_1 % 3)].wait((kb_1 / 3));` |
| compute.mma*4 (est.: sampled iterations x4) | 0.55 | 8.0% | 0.5 | 1.15 | `{` |
| load.tma*4 (est.: sampled iterations x4) | 0.13 | 1.9% | 0.5 | 0.38 | `if (tl::tl_shuffle_elect<128>())` |
| wait.mbarrier*4 (est.: sampled iterations x4) | 0.12 | 1.8% | 0.5 | 0.26 | `mbarrier[((kb % 3) + 3)].wait(((kb / 3) ^ 1));` |
| wait.sync | 0.11 | 1.6% | 1.0 | 0.13 | `__syncthreads();` |
| store.global | 0.09 | 1.4% | 0.5 | 0.22 | `for (int i_1 = 0; i_1 < 4; ++i_1) \| for (int i_2 = 0; i_2 < 8; ++i_2) \| for (int i_4 = 0; ` |

### cache_writer_kernel grid (48, 1, 1) block (128, 1, 1)

139 launches, 192 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 4.00 us, warp 3.62 us, nsys graphs 6.37 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 1.70 | 47.0% | 2.53 | 47.3% |
| unattributed | 1.55 | 42.9% | 1.76 | 32.9% |
| load | 0.29 | 8.1% | 0.98 | 18.3% |
| compute | 0.07 | 2.0% | 0.08 | 1.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 1.70 | 47.0% | 1.0 | 2.46 | `if (((int64_t)0 <= slot) & ((slot >> (int64_t)7) < ((int64_t)pages))) \| if (((int64_t)0 <=` |
| load.global | 0.24 | 6.5% | 1.0 | 0.26 | `for (int i = 0; i < 4; ++i)` |
| compute.reduce | 0.07 | 2.0% | 1.0 | 0.10 | `amax = max(amax, __shfl_xor_sync((uint)4294967295, amax, 1, 32));` |
| load.global#2 | 0.06 | 1.6% | 1.0 | 0.06 | `int64_t slot = slots[((int64_t)((int)blockIdx.x))];` |

### moe_dispatch_kernel grid (1, 1, 1) block (512, 1, 1)

153 launches, 16 warps; ranges, warp and last warp from the 1 recording CTAs' 16 warps; span 4.32 us, warp 4.17 us, nsys graphs 2.66 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 2.15 | 51.5% | 1.79 | 38.0% |
| unattributed, with loops' unsampled iterations | 1.42 | 34.2% | 1.60 | 34.0% |
| store | 0.52 | 12.4% | 1.22 | 26.0% |
| load | 0.08 | 2.0% | 0.10 | 2.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync#7*9 (est.: sampled iterations x9) | 0.65 | 15.5% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#6*9 (est.: sampled iterations x9) | 0.61 | 14.7% | 1.0 | 0.86 | `__syncthreads();` |
| store.global (sampled iterations) | 0.41 | 9.8% | 1.0 | 0.70 | `ChunkCount[((((int64_t)c) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = count; \| Chu` |
| wait.sync#8 | 0.37 | 8.8% | 1.0 | 0.54 | `__syncthreads();` |
| wait.sync#3 | 0.14 | 3.3% | 1.0 | 0.32 | `__syncthreads();` |
| wait.sync#2 | 0.13 | 3.0% | 1.0 | 0.19 | `__syncthreads();` |
| store.global#2 | 0.11 | 2.5% | 1.0 | 0.26 | `RoutePos[((((int64_t)((int)blockIdx.x)) * (int64_t)512) + ((int64_t)((int)threadIdx.x)))] ` |
| load.global (sampled iterations) | 0.08 | 2.0% | 1.0 | 0.10 | `expert = ((int)TopkIdx[((((int64_t)((int)blockIdx.x)) * (int64_t)512) + ((int64_t)((int)th` |
| wait.sync | 0.07 | 1.6% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#5 | 0.06 | 1.5% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#4 | 0.06 | 1.5% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#9 | 0.06 | 1.5% | 1.0 | 0.06 | `__syncthreads();` |

### mhc_finalize_kernel grid (40, 1, 1) block (128, 1, 1)

102 launches, 160 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 6.62 us, warp 5.89 us, nsys graphs 6.03 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 2.24 | 38.1% | 3.29 | 47.1% |
| store | 1.01 | 17.1% | 1.21 | 17.3% |
| load | 1.01 | 17.1% | 1.03 | 14.7% |
| compute | 1.00 | 16.9% | 0.55 | 7.9% |
| wait | 0.64 | 10.9% | 0.91 | 13.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync | 0.64 | 10.9% | 1.0 | 0.86 | `__syncthreads();` |
| compute.reduce | 0.58 | 9.8% | 0.2 | 2.30 | `row_max[0] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<32>>::run(row_max[0]);` |
| store.global#4*7 (est.: sampled iterations x7) | 0.56 | 9.5% | 0.8 | 0.90 | `if ((((int)threadIdx.x) % 4) == 0)` |
| compute.reduce#2*7 (est.: sampled iterations x7) | 0.42 | 7.1% | 0.8 | 0.67 | `amax[0] = max(amax[0], __shfl_xor_sync((uint)4294967295, amax[0], 1, 32));` |
| load.global#4*7 (est.: sampled iterations x7) | 0.36 | 6.2% | 0.8 | 0.67 | `*(uint4*)(yv + 0) = *(uint4*)(C + ((((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((i` |
| store.global#3*7 (est.: sampled iterations x7) | 0.34 | 5.7% | 0.8 | 0.45 | `*(uint4*)(y + ((((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((int64_t)step) * (int6` |
| load.global | 0.31 | 5.3% | 1.0 | 0.93 | `if (((int)threadIdx.x) < 32)` |
| load.global#3 | 0.24 | 4.1% | 0.8 | 0.35 | `for (int i_split_2 = 0; i_split_2 < 40; ++i_split_2)` |
| store.global | 0.10 | 1.7% | 0.2 | 0.42 | `if (((int)threadIdx.x) < 4)` |
| load.global#2 | 0.09 | 1.5% | 0.2 | 0.38 | `cm[0] = ((mixes_shared[((((int)threadIdx.x) & 15) + 8)] * scale[2]) + base[((((int)threadI` |
| store.global#2 | 0.01 | 0.2% | 0.2 | 0.06 | `if ((((int)threadIdx.x) >> 4) == 0)` |

### local_stats_kernel grid (16, 16, 1) block (128, 1, 1)

16 launches, 1024 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 41.14 us, warp 39.28 us, nsys graphs 36.42 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 38.05 | 96.9% | 36.75 | 96.2% |
| unattributed | 0.86 | 2.2% | 0.92 | 2.4% |
| compute | 0.20 | 0.5% | 0.20 | 0.5% |
| wait | 0.11 | 0.3% | 0.24 | 0.6% |
| store | 0.06 | 0.2% | 0.08 | 0.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 38.05 | 96.9% | 1.0 | 39.17 | `if (expanded_local_pos[((int64_t)((int)blockIdx.x))] < num_speculative_steps)` |
| compute.reduce | 0.14 | 0.3% | 1.0 | 0.18 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| wait.sync | 0.11 | 0.3% | 1.0 | 0.35 | `__syncthreads();` |
| compute.reduce#2 | 0.06 | 0.2% | 1.0 | 0.08 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 2, 32);` |
| store.global | 0.06 | 0.2% | 1.0 | 0.10 | `if (((int)threadIdx.x) == 0)` |

### gemm_mxfp8_partials_kernel grid (16, 5, 1) block (128, 1, 1)

51 launches, 320 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 11.74 us, warp 11.20 us, nsys graphs 12.70 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 5.12 | 45.7% | 5.78 | 47.6% |
| wait | 4.95 | 44.2% | 3.82 | 31.5% |
| compute | 0.61 | 5.4% | 0.69 | 5.7% |
| load | 0.46 | 4.1% | 1.80 | 14.8% |
| store | 0.05 | 0.5% | 0.05 | 0.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.cp_async#3 | 1.69 | 15.1% | 1.0 | 3.58 | `tl::cp_async_wait<0>();` |
| wait.cp_async#2 | 1.05 | 9.3% | 1.0 | 2.08 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>();` |
| wait.sync#2*2 (est.: sampled iterations x2) | 0.89 | 8.0% | 1.0 | 2.82 | `__syncthreads();` |
| wait.cp_async*2 (est.: sampled iterations x2) | 0.48 | 4.3% | 1.0 | 2.82 | `tl::cp_async_wait<2>();` |
| wait.sync#3 | 0.38 | 3.4% | 1.0 | 1.15 | `__syncthreads();` |
| wait.sync#4 | 0.36 | 3.3% | 1.0 | 1.15 | `__syncthreads();` |
| compute.mma*2 (est.: sampled iterations x2) | 0.30 | 2.7% | 1.0 | 0.32 | `{` |
| load.cp_async | 0.24 | 2.1% | 1.0 | 0.32 | `for (int i = 0; i < 2; ++i) \| for (int i_1 = 0; i_1 < 4; ++i_1)` |
| load.cp_async#2*2 (est.: sampled iterations x2) | 0.22 | 2.0% | 1.0 | 0.26 | `for (int i_4 = 0; i_4 < 2; ++i_4)` |
| compute.mma#2 | 0.15 | 1.4% | 1.0 | 0.16 | `{` |
| compute.mma#3 | 0.15 | 1.4% | 1.0 | 0.16 | `{` |
| wait.sync*2 (est.: sampled iterations x2) | 0.10 | 0.9% | 1.0 | 0.13 | `__syncthreads();` |
| store.global | 0.05 | 0.5% | 1.0 | 0.06 | `for (int i_11 = 0; i_11 < 16; ++i_11) \| for (int i_13 = 0; i_13 < 8; ++i_13) \| for (int i_` |
| compute.mma*6 (est.: sampled iterations x6) | 0.00 | 0.0% | 0.0 | 0.00 | `{` |
| load.cp_async#2*6 (est.: sampled iterations x6) | 0.00 | 0.0% | 0.0 | 0.00 | `for (int i_5 = 0; i_5 < 4; ++i_5)` |
| wait.cp_async*6 (est.: sampled iterations x6) | 0.00 | 0.0% | 0.0 | 0.00 | `tl::cp_async_wait<2>();` |
| wait.sync#2*6 (est.: sampled iterations x6) | 0.00 | 0.0% | 0.0 | 0.00 | `__syncthreads();` |
| wait.sync*6 (est.: sampled iterations x6) | 0.00 | 0.0% | 0.0 | 0.00 | `__syncthreads();` |

### mhc_collapse_kernel grid (240, 1, 1) block (128, 1, 1)

51 launches, 960 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 10.02 us, warp 6.17 us, nsys graphs 9.28 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 5.22 | 84.5% | 5.34 | 67.6% |
| load | 0.90 | 14.5% | 2.51 | 31.7% |
| store | 0.06 | 0.9% | 0.06 | 0.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.90 | 14.5% | 1.0 | 2.75 | `*(float4*)(mv + 0) = *(float4*)(mix + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int6` |
| store.global | 0.06 | 0.9% | 1.0 | 0.10 | `*(uint4*)(out + ((((int64_t)((int)blockIdx.x)) * (int64_t)1024) + (((int64_t)((int)threadI` |

### metadata_decode_kernel grid (48, 2, 1) block (128, 1, 1)

68 launches, 384 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 3.10 us, warp 1.57 us, nsys graphs 2.11 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.78 | 49.9% | 2.34 | 30.2% |
| load | 0.70 | 44.8% | 5.07 | 65.7% |
| store | 0.08 | 5.3% | 0.31 | 4.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.45 | 28.4% | 0.5 | 1.31 | `if ((((offset + ((int)blockIdx.x)) < src_rows) && (0 <= (offset + ((int)blockIdx.x)))))` |
| load.global#3 | 0.25 | 15.9% | 0.5 | 0.54 | `if ((((offset + ((int)blockIdx.x)) < main_src_rows) && (0 <= (offset + ((int)blockIdx.x)))` |
| store.global#4 | 0.04 | 2.5% | 0.5 | 0.13 | `if (0 < live)` |
| store.global | 0.03 | 2.0% | 0.5 | 0.06 | `swa[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = con` |
| load.global#2 | 0.01 | 0.5% | 0.1 | 0.10 | `if (((0 <= req) & ((int64_t)0 <= pos)))` |
| store.global#2 | 0.01 | 0.4% | 0.1 | 0.06 | `lengths[((int64_t)((int)blockIdx.x))] = ((int)max(info[1], (int64_t)0));` |
| store.global#3 | 0.01 | 0.4% | 0.1 | 0.06 | `top_lengths[((int64_t)((int)blockIdx.x))] = condval_6; \| top_lengths[((int64_t)((int)block` |

### router_partials_kernel grid (24, 1, 1) block (64, 1, 1)

102 launches, 48 warps; ranges, warp and last warp from the 4 recording CTAs' 8 warps; span 4.86 us, warp 3.05 us, nsys graphs 4.45 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.22 | 40.0% | 1.78 | 35.1% |
| load | 1.00 | 33.0% | 1.73 | 34.1% |
| compute | 0.68 | 22.3% | 1.36 | 26.9% |
| store | 0.14 | 4.7% | 0.20 | 3.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.reduce | 0.68 | 22.3% | 0.5 | 1.38 | `for (int k = 0; k < 3; ++k) \| for (int k = 0; k < 6; ++k)` |
| load.global | 0.45 | 14.7% | 1.0 | 0.48 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < mask_len))` |
| load.global#4 | 0.43 | 14.1% | 0.5 | 0.96 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < image_len)) \| if ((((((int)bl` |
| load.global#2 | 0.06 | 2.1% | 1.0 | 0.06 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < mask_len))` |
| load.global#3 | 0.06 | 2.1% | 1.0 | 0.06 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < mask_len))` |
| store.global | 0.05 | 1.6% | 1.0 | 0.06 | `if (condval == (bool)0) \| if (condval == (bool)1)` |
| store.global#2 | 0.05 | 1.6% | 1.0 | 0.06 | `if (condval_1 == (bool)0) \| if (condval_1 == (bool)1)` |
| store.global#3 | 0.05 | 1.6% | 0.5 | 0.10 | `if ((((int)threadIdx.x) & 31) < 3) \| if ((((int)threadIdx.x) & 31) < 6)` |

### metadata_builder_kernel grid (2, 1, 1) block (128, 1, 1)

289 launches, 8 warps; ranges, warp and last warp from the 2 recording CTAs' 8 warps; span 1.54 us, warp 0.84 us, nsys graphs 1.54 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.46 | 55.2% | 1.25 | 75.9% |
| unattributed | 0.30 | 35.7% | 0.34 | 20.6% |
| store | 0.08 | 9.0% | 0.06 | 3.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#3 | 0.42 | 50.5% | 0.5 | 1.18 | `for (int __1 = 0; __1 < 4; ++__1)` |
| load.global#2 | 0.03 | 3.3% | 0.1 | 0.22 | `if ((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) < nr))` |
| store.global#3 | 0.03 | 3.3% | 0.5 | 0.06 | `if (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) == 0)` |
| store.global#4 | 0.03 | 3.3% | 0.5 | 0.06 | `positions[(((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))` |
| load.global | 0.01 | 1.4% | 0.1 | 0.10 | `if ((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) <= nr))` |
| store.global | 0.01 | 1.4% | 0.1 | 0.10 | `out_starts[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] = condval;` |
| store.global#2 | 0.01 | 1.0% | 0.1 | 0.06 | `request_positions[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] = condval_2;` |

### token_ranks_kernel grid (8, 1, 1) block (256, 1, 1)

51 launches, 64 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 7.78 us, warp 7.68 us, nsys graphs 8.96 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 6.96 | 90.7% | 7.08 | 90.2% |
| unattributed | 0.34 | 4.4% | 0.37 | 4.7% |
| compute | 0.33 | 4.3% | 0.35 | 4.5% |
| store | 0.05 | 0.6% | 0.04 | 0.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 6.96 | 90.7% | 1.0 | 7.17 | `if ((((int64_t)0 <= token_ids[((int64_t)((int)blockIdx.x))]) && (token_ids[((int64_t)((int` |
| compute.reduce | 0.33 | 4.3% | 1.0 | 0.61 | `total[0] = tl::AllReduce<tl::SumOp, 256, 1, 0, tl::NamedBarrier<256>>::run(total[0], (&(wo` |
| store.global | 0.05 | 0.6% | 1.0 | 0.06 | `output[((int64_t)((int)blockIdx.x))] = ((int64_t)total[0]);` |

### indexer_topk_segment_kernel grid (1, 48, 1) block (512, 1, 1)

34 launches, 768 warps; ranges, warp and last warp from the 4 recording CTAs' 64 warps; span 11.01 us, warp 10.79 us, nsys graphs 5.57 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed, with loops' unsampled iterations | 5.62 | 52.1% | 5.39 | 50.1% |
| wait | 3.69 | 34.2% | 3.95 | 36.8% |
| load | 1.32 | 12.2% | 1.20 | 11.1% |
| store | 0.15 | 1.4% | 0.21 | 2.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.32 | 12.2% | 1.0 | 1.49 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 4096); \| i` |
| wait.sync (sampled iterations) | 0.98 | 9.1% | 14.5 | 1.14 | `__syncthreads();` |
| wait.sync#24*9 (est.: sampled iterations x9) | 0.31 | 2.9% | 0.5 | 0.43 | `__syncthreads();` |
| wait.sync#20*9 (est.: sampled iterations x9) | 0.30 | 2.8% | 0.5 | 0.43 | `__syncthreads();` |
| wait.sync#19*9 (est.: sampled iterations x9) | 0.30 | 2.8% | 0.5 | 0.43 | `__syncthreads();` |
| wait.sync#23*9 (est.: sampled iterations x9) | 0.29 | 2.7% | 0.5 | 0.43 | `__syncthreads();` |
| wait.sync#14*8 (est.: sampled iterations x8) | 0.27 | 2.5% | 0.5 | 0.38 | `__syncthreads();` |
| wait.sync#6*8 (est.: sampled iterations x8) | 0.26 | 2.4% | 0.5 | 0.38 | `__syncthreads();` |
| wait.sync#5*8 (est.: sampled iterations x8) | 0.26 | 2.4% | 0.5 | 0.38 | `__syncthreads();` |
| wait.sync#13*8 (est.: sampled iterations x8) | 0.26 | 2.4% | 0.5 | 0.38 | `__syncthreads();` |
| store.global | 0.15 | 1.4% | 1.5 | 0.19 | `for (int i_3 = 0; i_3 < 4; ++i_3) \| if (((int)threadIdx.x) < total) \| if (((int)threadIdx.` |
| wait.sync#25 | 0.04 | 0.4% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#16 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#8 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#17 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#18 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#2 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#3 | 0.03 | 0.3% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#7 | 0.03 | 0.3% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#11 | 0.03 | 0.3% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#15 | 0.03 | 0.3% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#9 | 0.03 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#12 | 0.03 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#4 | 0.02 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#21 | 0.02 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#22 | 0.02 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#10 | 0.02 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |

### rope_kernel grid (3, 1, 1) block (128, 1, 1)

224 launches, 12 warps; ranges, warp and last warp from the 3 recording CTAs' 12 warps; span 1.12 us, warp 1.09 us, nsys graphs 1.31 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.86 | 78.7% | 1.22 | 75.7% |
| load | 0.20 | 18.1% | 0.35 | 21.8% |
| store | 0.03 | 3.2% | 0.04 | 2.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 | 0.10 | 9.3% | 1.0 | 0.13 | `if (((position < ((int64_t)positions_max)) && ((int64_t)0 <= position)))` |
| load.global | 0.10 | 8.8% | 1.0 | 0.10 | `if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < tokens)) \| int64_t position ` |
| store.global | 0.03 | 3.2% | 1.0 | 0.06 | `*(uint4*)(x + ((((((int64_t)((int)blockIdx.x)) * (int64_t)12288) + ((((int64_t)((int)threa` |

### norm_forward_and_per_token_cast_kernel_kernel grid (6, 1, 1) block (128, 1, 1)

105 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 2.94 us, warp 2.94 us, nsys graphs 1.54 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 2.47 | 83.8% | 2.86 | 84.5% |
| load | 0.44 | 14.9% | 0.47 | 13.9% |
| compute | 0.11 | 3.8% | 0.11 | 3.3% |
| wait | 0.05 | 1.7% | 0.05 | 1.6% |
| unattributed | 0.00 | 0.0% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*4 (est.: sampled iterations x4) | 1.79 | 60.9% | 1.0 | 1.92 | `for (int j_2 = 0; j_2 < 2; ++j_2)` |
| store.global*4 (est.: sampled iterations x4) | 0.41 | 13.9% | 1.0 | 0.51 | `for (int j_1 = 0; j_1 < 4; ++j_1)` |
| load.global*4 (est.: sampled iterations x4) | 0.24 | 8.2% | 1.0 | 0.26 | `if (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 4)) < num_tokens)` |
| load.global#3*4 (est.: sampled iterations x4) | 0.17 | 5.6% | 1.0 | 0.26 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 128) + ((((int)threadIdx.x) & 15` |
| store.global#5*4 (est.: sampled iterations x4) | 0.17 | 5.6% | 1.0 | 0.26 | `*(uint4*)(out + ((((((int64_t)vi_1) * (int64_t)128) + ((((int64_t)((int)threadIdx.x)) & (i` |
| compute.reduce | 0.11 | 3.8% | 1.0 | 0.13 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| store.global#3 | 0.05 | 1.8% | 1.0 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |
| wait.sync | 0.05 | 1.7% | 1.0 | 0.10 | `__syncthreads();` |
| store.global#2 | 0.05 | 1.5% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| load.global#2 | 0.03 | 1.2% | 1.0 | 0.06 | `*(float2*)(row_partial + 0) = *(float2*)(sum_shared + ((((int)threadIdx.x) >> 6) * 2)); \| ` |

### indexer_score_kernel grid (128, 48, 1) block (128, 1, 1)

17 launches, 24576 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 20.67 us, warp 2.54 us, nsys graphs 11.74 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.37 | 53.7% | 0.60 | 64.6% |
| unattributed, with loops' unsampled iterations | 0.85 | 33.4% | 0.31 | 32.8% |
| wait | 0.23 | 9.0% | 0.01 | 1.2% |
| compute | 0.07 | 2.8% | 0.01 | 1.0% |
| store | 0.03 | 1.1% | 0.00 | 0.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 (sampled iterations) | 0.87 | 34.1% | 0.5 | 2.02 | `for (int i_1 = 0; i_1 < 32; ++i_1)` |
| load.global | 0.36 | 14.1% | 1.0 | 0.86 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 4096); \| i` |
| load.global#3 (sampled iterations) | 0.14 | 5.6% | 0.5 | 0.54 | `if (((int)threadIdx.x) < 64)` |
| wait.sync#2 (sampled iterations) | 0.13 | 5.1% | 0.5 | 0.54 | `__syncthreads();` |
| compute.mma (sampled iterations) | 0.07 | 2.8% | 0.5 | 0.16 | `for (int ki = 0; ki < 2; ++ki)` |
| wait.sync#4 (sampled iterations) | 0.04 | 1.4% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync (sampled iterations) | 0.03 | 1.3% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#3 (sampled iterations) | 0.03 | 1.2% | 0.5 | 0.06 | `__syncthreads();` |
| store.global (sampled iterations) | 0.03 | 1.1% | 0.5 | 0.06 | `if ((((int)threadIdx.x) >> 6) == 0)` |

### topk_ids_kernel grid (16, 1, 1) block (512, 1, 1)

16 launches, 256 warps; ranges, warp and last warp from the 4 recording CTAs' 64 warps; span 20.99 us, warp 19.09 us, nsys graphs 21.17 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 15.99 | 83.8% | 16.63 | 79.2% |
| unattributed | 2.01 | 10.5% | 1.93 | 9.2% |
| wait | 0.87 | 4.6% | 0.64 | 3.0% |
| store | 0.21 | 1.1% | 1.78 | 8.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 15.99 | 83.8% | 1.0 | 17.12 | `for (int t = 0; t < (vocab >> 14); ++t)` |
| wait.sync | 0.73 | 3.8% | 1.0 | 1.79 | `__syncthreads();` |
| store.global | 0.21 | 1.1% | 1.0 | 2.53 | `if ((((int)threadIdx.x) >> 5) == 0)` |
| wait.sync#2 | 0.08 | 0.4% | 1.0 | 0.16 | `__syncthreads();` |
| wait.sync#3 | 0.06 | 0.3% | 1.0 | 0.06 | `__syncthreads();` |

### topk_log_softmax_kernel grid (16, 1, 1) block (1024, 1, 1)

16 launches, 512 warps; ranges, warp and last warp from the 4 recording CTAs' 128 warps; span 20.40 us, warp 18.66 us, nsys graphs 18.74 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 15.51 | 83.1% | 17.49 | 85.9% |
| unattributed, with loops' unsampled iterations | 1.51 | 8.1% | 1.69 | 8.3% |
| wait | 1.26 | 6.7% | 0.75 | 3.7% |
| compute | 0.39 | 2.1% | 0.38 | 1.8% |
| store | 0.00 | 0.0% | 0.07 | 0.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 8.12 | 43.5% | 1.0 | 9.02 | `if (0 < (vocab & 2047)) \| if (0 < (vocab & 8191))` |
| load.global#2 | 7.38 | 39.5% | 1.0 | 8.05 | `for (int step_1 = 0; step_1 < (vocab >> 11); ++step_1) \| for (int step_1 = 0; step_1 < (vo` |
| wait.sync#2 | 0.51 | 2.7% | 1.0 | 1.15 | `__syncthreads();` |
| wait.sync | 0.49 | 2.7% | 1.0 | 1.33 | `__syncthreads();` |
| compute.reduce#3 | 0.17 | 0.9% | 1.0 | 0.19 | `ties[0] = vllm_add(ties[0], __shfl_xor_sync((uint)4294967295, ties[0], 16, 32));` |
| compute.reduce#2 | 0.11 | 0.6% | 1.0 | 0.19 | `se[0] = vllm_add(se[0], __shfl_xor_sync((uint)4294967295, se[0], 16, 32));` |
| compute.reduce | 0.10 | 0.5% | 1.0 | 0.13 | `best[0] = vllm_max(best[0], __shfl_xor_sync((uint)4294967295, best[0], 16, 32));` |
| wait.sync#4 | 0.10 | 0.5% | 1.0 | 0.16 | `__syncthreads();` |
| wait.sync#3 | 0.08 | 0.4% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#5 | 0.08 | 0.4% | 1.0 | 0.10 | `__syncthreads();` |
| load.global#3 (sampled iterations) | 0.01 | 0.1% | 0.0 | 0.38 | `float value_32 = vllm_sub(vllm_shifted(logits[((((int64_t)((int)blockIdx.x)) * logits_stri` |
| store.global (sampled iterations) | 0.00 | 0.0% | 0.0 | 0.06 | `output[(((((int64_t)c) * (int64_t)1024) + (((int64_t)((int)blockIdx.x)) * ((int64_t)topk))` |

### metadata_chunk_kernel grid (48, 1, 1) block (128, 1, 1)

34 launches, 192 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 10.32 us, warp 5.27 us, nsys graphs 10.18 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 4.27 | 81.0% | 7.82 | 83.5% |
| unattributed | 0.92 | 17.5% | 1.41 | 15.0% |
| store | 0.08 | 1.6% | 0.14 | 1.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 4.22 | 80.0% | 1.0 | 6.98 | `if ((((offset + ((int)blockIdx.x)) < src_rows) && (0 <= (offset + ((int)blockIdx.x)))))` |
| store.global | 0.06 | 1.1% | 1.0 | 0.10 | `swa[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = con` |
| load.global#2 | 0.05 | 0.9% | 0.2 | 0.51 | `if (((0 <= req) & ((int64_t)0 <= pos)))` |
| store.global#2 | 0.01 | 0.2% | 0.2 | 0.06 | `lengths[((int64_t)((int)blockIdx.x))] = ((int)max(info[1], (int64_t)0));` |
| store.global#3 | 0.01 | 0.2% | 0.2 | 0.06 | `top_lengths[((int64_t)((int)blockIdx.x))] = condval_12; \| top_lengths[((int64_t)((int)bloc` |

### moe_combine_kernel grid (5, 48, 1) block (128, 1, 1)

102 launches, 960 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 2.69 us, warp 1.80 us, nsys graphs 2.98 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.31 | 72.9% | 2.32 | 81.7% |
| unattributed | 0.43 | 24.1% | 0.47 | 16.4% |
| store | 0.05 | 3.0% | 0.06 | 2.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.31 | 72.9% | 1.0 | 2.11 | `for (int k = 0; k < 3; ++k) \| for (int k = 0; k < 6; ++k) \| if ((((int)blockIdx.y) < share` |
| store.global | 0.05 | 3.0% | 1.0 | 0.06 | `{` |

### topk_log_softmax_kernel grid (8, 1, 1) block (1024, 1, 1)

17 launches, 256 warps; ranges, warp and last warp from the 4 recording CTAs' 128 warps; span 17.82 us, warp 16.96 us, nsys graphs 16.51 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 13.72 | 80.9% | 14.77 | 83.1% |
| unattributed, with loops' unsampled iterations | 1.76 | 10.4% | 1.77 | 10.0% |
| wait | 1.09 | 6.4% | 0.78 | 4.4% |
| compute | 0.39 | 2.3% | 0.39 | 2.2% |
| store | 0.00 | 0.0% | 0.06 | 0.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 6.99 | 41.2% | 1.0 | 7.39 | `if (0 < (vocab & 2047)) \| if (0 < (vocab & 8191))` |
| load.global#2 | 6.72 | 39.6% | 1.0 | 7.46 | `for (int step_1 = 0; step_1 < (vocab >> 11); ++step_1) \| for (int step_1 = 0; step_1 < (vo` |
| wait.sync#2 | 0.51 | 3.0% | 1.0 | 1.15 | `__syncthreads();` |
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

102 launches, 96 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 2.50 us, warp 2.29 us, nsys graphs 1.92 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 1.58 | 69.0% | 2.02 | 69.3% |
| load | 0.35 | 15.5% | 0.37 | 12.6% |
| unattributed | 0.12 | 5.5% | 0.36 | 12.3% |
| compute | 0.12 | 5.2% | 0.12 | 4.0% |
| wait | 0.11 | 4.9% | 0.05 | 1.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*3 (est.: sampled iterations x3) | 1.07 | 46.7% | 1.0 | 1.15 | `for (int j_2 = 0; j_2 < 2; ++j_2)` |
| store.global*3 (est.: sampled iterations x3) | 0.31 | 13.4% | 1.0 | 0.38 | `for (int j_1 = 0; j_1 < 4; ++j_1)` |
| load.global*3 (est.: sampled iterations x3) | 0.18 | 7.9% | 1.0 | 0.19 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 6)) < num_tokens) && (((vi * 2) + (` |
| load.global#3*3 (est.: sampled iterations x3) | 0.14 | 6.0% | 1.0 | 0.19 | `if (((vi_1 * 2) + ((((int)threadIdx.x) & 63) >> 5)) < 5)` |
| store.global#5*3 (est.: sampled iterations x3) | 0.13 | 5.8% | 1.0 | 0.19 | `*(uint4*)(out + ((((((int64_t)vi_1) * (int64_t)512) + ((((int64_t)((int)threadIdx.x)) & (i` |
| compute.reduce | 0.12 | 5.2% | 1.0 | 0.13 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| wait.sync | 0.11 | 4.9% | 1.0 | 0.19 | `__syncthreads();` |
| store.global#2 | 0.04 | 1.7% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| load.global#2 | 0.04 | 1.6% | 1.0 | 0.06 | `*(float2*)(row_partial + 0) = *(float2*)(sum_shared + ((((int)threadIdx.x) >> 6) * 2)); \| ` |
| store.global#3 | 0.03 | 1.4% | 0.5 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |

### sink_rows_kernel grid (8, 64, 1) block (256, 1, 1)

17 launches, 4096 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 16.83 us, warp 5.12 us, nsys graphs 7.89 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 2.53 | 49.4% | 4.49 | 49.3% |
| load | 1.48 | 28.8% | 2.87 | 31.5% |
| unattributed | 1.12 | 21.8% | 1.74 | 19.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 2.50 | 48.8% | 1.0 | 5.73 | `for (int k_1 = 0; k_1 < 8; ++k_1)` |
| load.global | 1.47 | 28.7% | 1.0 | 3.74 | `row = (max((((int64_t)num_sampled[((int64_t)((int)blockIdx.x))]) - (int64_t)1), (int64_t)0` |
| store.global#2 | 0.03 | 0.6% | 0.0 | 0.93 | `tokens[((((int64_t)((int)blockIdx.x)) * (int64_t)3) + ((int64_t)((int)threadIdx.x)))] = to` |
| load.global#2 | 0.00 | 0.1% | 0.0 | 0.13 | `token = max(alternates[((((int64_t)((int)blockIdx.x)) * (int64_t)3) + ((int64_t)((int)thre` |

### sparse_mla_kernel_1 grid (24, 40, 1) block (128, 1, 1)

51 launches, 960 CTAs, 3840 warps; span 5.31 us, warp 2.24 us, nsys graphs 4.51 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 2.23 | 99.7% | 2.40 | 100.0% |
| load | 0.01 | 0.2% | 0.00 | 0.0% |
| wait | 0.00 | 0.1% | 0.00 | 0.0% |
| compute | 0.00 | 0.0% | 0.00 | 0.0% |
| store | 0.00 | 0.0% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#3 | 0.00 | 0.2% | 0.0 | 1.57 | `M = __fmul_rn(Sinks[((int)blockIdx.x)], 0x1.7154764ee6c2fp+0f/*1.442695e+00*/);` |
| wait.sync#13 | 0.00 | 0.0% | 0.0 | 0.83 | `__syncthreads();` |
| load.global#4 | 0.00 | 0.0% | 0.0 | 0.83 | `if (((int)threadIdx.x) < 32)` |
| compute.reduce#3 | 0.00 | 0.0% | 0.0 | 0.19 | `amax[0] = tl::AllReduce<tl::MaxOp, 8, 1, 0, tl::NamedBarrier<128>>::run(amax[0]);` |
| wait.sync#12 | 0.00 | 0.0% | 0.0 | 0.29 | `__syncthreads();` |
| store.global#2 | 0.00 | 0.0% | 0.0 | 0.10 | `*(uint2*)(Output + (((((int64_t)((int)blockIdx.y)) * (int64_t)12288) + (((int64_t)((int)bl` |

### gemm_splitk_reduce_cast_kernel grid (24, 12, 1) block (128, 1, 1)

102 launches, 1152 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 2.05 us, warp 1.69 us, nsys graphs 2.14 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.02 | 60.3% | 1.50 | 68.8% |
| unattributed | 0.49 | 29.3% | 0.50 | 23.2% |
| store | 0.09 | 5.3% | 0.09 | 4.0% |
| compute | 0.09 | 5.1% | 0.09 | 4.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.02 | 60.3% | 1.0 | 1.34 | `if ((1 <= M))` |
| store.global | 0.09 | 5.3% | 1.0 | 0.10 | `if (((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 5)) < M)` |
| compute.reduce | 0.09 | 5.1% | 1.0 | 0.10 | `amax[0] = tl::AllReduce<tl::MaxOp, 8, 1, 0, tl::NamedBarrier<128>>::run(amax[0]);` |

### moe_swiglu_cast_kernel_kernel grid (16, 120, 1) block (96, 1, 1)

51 launches, 5760 warps; ranges, warp and last warp from the 4 recording CTAs' 12 warps; span 4.35 us, warp 1.41 us, nsys graphs 4.03 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.07 | 76.1% | 3.59 | 84.0% |
| load | 0.26 | 18.2% | 0.38 | 8.8% |
| compute | 0.04 | 3.0% | 0.16 | 3.8% |
| store | 0.04 | 2.7% | 0.15 | 3.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.26 | 18.2% | 1.0 | 0.54 | `for (int r = 0; r < 2; ++r)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.04 | 3.0% | 0.2 | 0.19 | `if (0 <= orders[r_2])` |
| store.global*2 (est.: sampled iterations x2) | 0.04 | 2.7% | 0.2 | 0.19 | `if (0 <= orders[r_2])` |

### compress_gather_partial_state_kernel grid (8, 1, 1) block (128, 1, 1)

17 launches, 32 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 12.80 us, warp 10.42 us, nsys graphs 5.46 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 5.02 | 48.2% | 6.15 | 45.7% |
| store | 4.13 | 39.7% | 5.23 | 38.8% |
| unattributed | 1.27 | 12.2% | 2.08 | 15.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 4.90 | 47.0% | 1.0 | 4.90 | `int nt = counts[0];` |
| store.global*2 (est.: sampled iterations x2) | 4.07 | 39.1% | 1.0 | 6.34 | `if (half == 0)` |
| load.global#2*2 (est.: sampled iterations x2) | 0.12 | 1.2% | 1.0 | 0.13 | `for (int i = 0; i < 4; ++i)` |
| store.global#2 | 0.06 | 0.6% | 1.0 | 0.10 | `if (((int)threadIdx.x) == 0)` |

### sink_candidates_kernel grid (8, 127, 1) block (128, 1, 1)

17 launches, 4064 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 11.65 us, warp 4.61 us, nsys graphs 9.38 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.56 | 33.9% | 0.96 | 22.7% |
| compute | 1.42 | 30.7% | 1.36 | 32.2% |
| load | 0.73 | 15.9% | 1.01 | 23.9% |
| wait | 0.61 | 13.2% | 0.52 | 12.3% |
| store | 0.29 | 6.2% | 0.38 | 8.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.reduce*4 (est.: sampled iterations x4) | 1.02 | 22.0% | 1.0 | 1.28 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| load.global | 0.73 | 15.9% | 1.0 | 1.15 | `local[0] = max((num_sampled[((int64_t)((int)blockIdx.x))] - 1), 0);` |
| compute.reduce#2*4 (est.: sampled iterations x4) | 0.40 | 8.7% | 1.0 | 0.51 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 2, 32);` |
| wait.sync#2*4 (est.: sampled iterations x4) | 0.31 | 6.8% | 1.0 | 0.51 | `__syncthreads();` |
| wait.sync*4 (est.: sampled iterations x4) | 0.30 | 6.4% | 1.0 | 0.51 | `__syncthreads();` |
| store.global*4 (est.: sampled iterations x4) | 0.29 | 6.2% | 1.0 | 0.51 | `if (((int)threadIdx.x) == 0)` |

### compress_kernel grid (12, 1, 1) block (128, 1, 1)

34 launches, 48 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 6.21 us, warp 4.84 us, nsys graphs 5.79 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 2.34 | 48.3% | 1.78 | 27.3% |
| load | 1.97 | 40.8% | 3.84 | 58.9% |
| store | 0.44 | 9.0% | 0.81 | 12.4% |
| compute | 0.09 | 1.9% | 0.09 | 1.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.23 | 25.5% | 1.0 | 1.55 | `if ((((0 <= counts[0]) & (0 < counts[1])) & (counts[1] <= 8)) & (((((int)blockIdx.x) * 4) ` |
| load.global#2*4 (est.: sampled iterations x4) | 0.64 | 13.1% | 1.0 | 1.41 | `*(float4*)(v1 + 0) = *(float4*)(values + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048)` |
| store.global*4 (est.: sampled iterations x4) | 0.15 | 3.1% | 1.0 | 0.26 | `*(uint2*)(out + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + ((((int64_t)((int)thre` |
| compute.reduce | 0.09 | 1.9% | 1.0 | 0.10 | `float total = tl::warp_reduce_sum(ss[0]);` |
| store.global#3 | 0.08 | 1.7% | 0.5 | 0.14 | `emitted_slots[((((int64_t)((int)blockIdx.x)) * (int64_t)4) + (((int64_t)((int)threadIdx.x)` |
| store.global#3*4 (est.: sampled iterations x4) | 0.07 | 1.4% | 0.5 | 0.13 | `*(uint2*)(out + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + ((((int64_t)((int)thre` |
| load.global#3*4 (est.: sampled iterations x4) | 0.06 | 1.3% | 0.5 | 0.06 | `*(float4*)(w + 0) = *(float4*)(weight + ((c_1 * 128) + ((((int)threadIdx.x) & 31) * 4)));` |
| store.global#2*4 (est.: sampled iterations x4) | 0.06 | 1.2% | 0.2 | 0.13 | `if ((slot >> (int64_t)3) < ((int64_t)blocks))` |
| store.global#5 | 0.04 | 0.9% | 0.5 | 0.16 | `emitted_slots[((((int64_t)((int)blockIdx.x)) * (int64_t)4) + (((int64_t)((int)threadIdx.x)` |
| load.global#3 | 0.02 | 0.5% | 0.5 | 0.03 | `if (valid)` |
| load.global#4 | 0.02 | 0.4% | 0.5 | 0.03 | `if (valid)` |
| store.global#2 | 0.02 | 0.3% | 0.5 | 0.02 | `emitted[((((int64_t)((int)blockIdx.x)) * (int64_t)4) + (((int64_t)((int)threadIdx.x)) >> (` |
| store.global#4 | 0.02 | 0.3% | 0.5 | 0.02 | `emitted[((((int64_t)((int)blockIdx.x)) * (int64_t)4) + (((int64_t)((int)threadIdx.x)) >> (` |

### sparse_mla_kernel_1 grid (24, 48, 1) block (128, 1, 1)

34 launches, 1152 CTAs, 4608 warps; span 5.55 us, warp 2.10 us, nsys graphs 5.07 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 2.10 | 99.7% | 1.71 | 100.0% |
| load | 0.00 | 0.2% | 0.00 | 0.0% |
| wait | 0.00 | 0.0% | 0.00 | 0.0% |
| compute | 0.00 | 0.0% | 0.00 | 0.0% |
| store | 0.00 | 0.0% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#3 | 0.00 | 0.2% | 0.0 | 1.63 | `M = __fmul_rn(Sinks[((int)blockIdx.x)], 0x1.7154764ee6c2fp+0f/*1.442695e+00*/);` |
| wait.sync#13 | 0.00 | 0.0% | 0.0 | 0.32 | `__syncthreads();` |
| compute.reduce#3 | 0.00 | 0.0% | 0.0 | 0.16 | `amax[0] = tl::AllReduce<tl::MaxOp, 8, 1, 0, tl::NamedBarrier<128>>::run(amax[0]);` |
| load.global#4 | 0.00 | 0.0% | 0.0 | 0.32 | `if (((int)threadIdx.x) < 32)` |
| wait.sync#12 | 0.00 | 0.0% | 0.0 | 0.19 | `__syncthreads();` |
| store.global#2 | 0.00 | 0.0% | 0.0 | 0.10 | `*(uint2*)(Output + (((((int64_t)((int)blockIdx.y)) * (int64_t)12288) + (((int64_t)((int)bl` |

### mhc_post_kernel grid (200, 1, 1) block (128, 1, 1)

17 launches, 800 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 11.26 us, warp 8.16 us, nsys graphs 4.14 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 7.96 | 97.5% | 9.66 | 92.8% |
| load | 0.15 | 1.9% | 0.58 | 5.6% |
| store | 0.05 | 0.7% | 0.17 | 1.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.15 | 1.9% | 1.0 | 0.19 | `*(float4*)(pm + 0) = *(float4*)(post + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int` |
| store.global | 0.05 | 0.7% | 1.0 | 0.13 | `for (int n_1 = 0; n_1 < 4; ++n_1)` |

### rope_kernel grid (48, 1, 1) block (192, 1, 1)

102 launches, 288 warps; ranges, warp and last warp from the 4 recording CTAs' 24 warps; span 1.28 us, warp 1.23 us, nsys graphs 1.57 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.74 | 59.7% | 1.29 | 70.0% |
| load | 0.46 | 37.1% | 0.51 | 27.6% |
| store | 0.04 | 3.2% | 0.04 | 2.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.46 | 37.1% | 1.0 | 0.48 | `if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < tokens)) \| int64_t position ` |
| store.global | 0.04 | 3.2% | 1.0 | 0.06 | `*(uint4*)(x + ((((((int64_t)((int)blockIdx.x)) * (int64_t)12288) + ((((int64_t)((int)threa` |

### moe_combine_kernel grid (5, 40, 1) block (128, 1, 1)

51 launches, 800 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 3.01 us, warp 2.29 us, nsys graphs 2.05 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.87 | 81.7% | 2.80 | 87.0% |
| unattributed | 0.36 | 15.8% | 0.36 | 11.3% |
| store | 0.06 | 2.4% | 0.05 | 1.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.87 | 81.7% | 1.0 | 2.37 | `for (int k = 0; k < 3; ++k) \| for (int k = 0; k < 6; ++k) \| if ((((int)blockIdx.y) < share` |
| store.global | 0.06 | 2.4% | 1.0 | 0.06 | `{` |

### indexer_score_kernel grid (64, 48, 1) block (128, 1, 1)

17 launches, 12288 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 10.46 us, warp 2.42 us, nsys graphs 6.88 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.18 | 48.8% | 0.62 | 66.8% |
| unattributed, with loops' unsampled iterations | 0.86 | 35.4% | 0.29 | 31.0% |
| wait | 0.28 | 11.4% | 0.01 | 1.2% |
| compute | 0.08 | 3.4% | 0.01 | 0.8% |
| store | 0.03 | 1.1% | 0.00 | 0.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 (sampled iterations) | 0.66 | 27.4% | 0.5 | 1.50 | `for (int i_1 = 0; i_1 < 32; ++i_1)` |
| load.global | 0.42 | 17.4% | 1.0 | 0.86 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 4096); \| i` |
| wait.sync#2 (sampled iterations) | 0.15 | 6.3% | 0.5 | 0.51 | `__syncthreads();` |
| load.global#3 (sampled iterations) | 0.10 | 4.0% | 0.5 | 0.54 | `if (((int)threadIdx.x) < 64)` |
| compute.mma (sampled iterations) | 0.08 | 3.4% | 0.5 | 0.19 | `for (int ki = 0; ki < 2; ++ki)` |
| wait.sync#4 (sampled iterations) | 0.05 | 2.1% | 0.5 | 0.16 | `__syncthreads();` |
| wait.sync (sampled iterations) | 0.04 | 1.5% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#3 (sampled iterations) | 0.04 | 1.5% | 0.5 | 0.10 | `__syncthreads();` |
| store.global (sampled iterations) | 0.03 | 1.1% | 0.5 | 0.10 | `if ((((int)threadIdx.x) >> 6) == 0)` |

### per_token_cast_kernel_kernel grid (30, 1, 1) block (64, 1, 1)

136 launches, 60 warps; ranges, warp and last warp from the 5 recording CTAs' 10 warps; span 1.22 us, warp 1.17 us, nsys graphs 1.12 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.43 | 36.4% | 0.40 | 30.2% |
| load | 0.36 | 30.4% | 0.47 | 35.5% |
| store | 0.29 | 24.9% | 0.36 | 27.1% |
| compute | 0.10 | 8.2% | 0.09 | 7.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.27 | 22.7% | 1.0 | 0.29 | `for (int i = 0; i < 2; ++i)` |
| store.global#2 | 0.11 | 9.6% | 1.0 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.10 | 8.2% | 1.0 | 0.13 | `amax_fragment[i_1] = tl::AllReduce<tl::MaxOp, 2, 1, 0, tl::NamedBarrier<64>>::run(amax_fra` |
| load.global#2*2 (est.: sampled iterations x2) | 0.09 | 7.7% | 1.0 | 0.13 | `amax_fragment[i_1] = cutlass::bfloat16_t(__hmax((bfloat16_t(((nv_bfloat162*)(&(amax_fragme` |
| store.global#3*2 (est.: sampled iterations x2) | 0.09 | 7.7% | 1.0 | 0.13 | `*(fp8_e4_16_t*)(out + ((((((((int64_t)((int)blockIdx.x)) / (int64_t)3) * (int64_t)6144) + ` |
| store.global*2 (est.: sampled iterations x2) | 0.09 | 7.7% | 1.0 | 0.13 | `out_sf[((((((int64_t)((int)blockIdx.x)) % (int64_t)15) * (int64_t)32) + (((int64_t)((int)t` |

### indexer_topk_segment_kernel grid (2, 48, 1) block (512, 1, 1)

17 launches, 1536 warps; ranges, warp and last warp from the 4 recording CTAs' 64 warps; span 9.95 us, warp 5.27 us, nsys graphs 4.51 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 3.07 | 58.3% | 6.14 | 59.3% |
| unattributed, with loops' unsampled iterations | 1.80 | 34.1% | 3.24 | 31.3% |
| load | 0.35 | 6.7% | 0.87 | 8.4% |
| store | 0.05 | 0.9% | 0.11 | 1.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync#24*9 (est.: sampled iterations x9) | 0.34 | 6.5% | 0.5 | 0.86 | `__syncthreads();` |
| wait.sync#20*9 (est.: sampled iterations x9) | 0.34 | 6.4% | 0.5 | 0.86 | `__syncthreads();` |
| wait.sync#19*9 (est.: sampled iterations x9) | 0.32 | 6.1% | 0.5 | 0.86 | `__syncthreads();` |
| wait.sync#23*9 (est.: sampled iterations x9) | 0.32 | 6.0% | 0.5 | 0.86 | `__syncthreads();` |
| wait.sync#14*8 (est.: sampled iterations x8) | 0.30 | 5.8% | 0.5 | 0.77 | `__syncthreads();` |
| wait.sync#13*8 (est.: sampled iterations x8) | 0.30 | 5.7% | 0.5 | 0.77 | `__syncthreads();` |
| wait.sync#6*8 (est.: sampled iterations x8) | 0.30 | 5.6% | 0.5 | 0.77 | `__syncthreads();` |
| wait.sync#5*8 (est.: sampled iterations x8) | 0.28 | 5.2% | 0.5 | 0.77 | `__syncthreads();` |
| load.global | 0.20 | 3.8% | 1.0 | 0.26 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 4096); \| i` |
| load.global#2 | 0.15 | 2.9% | 0.5 | 0.38 | `for (int i = 0; i < 8; ++i)` |
| wait.sync (sampled iterations) | 0.06 | 1.1% | 0.5 | 0.19 | `__syncthreads();` |
| wait.sync#25 | 0.05 | 0.9% | 0.5 | 0.16 | `__syncthreads();` |
| store.global | 0.05 | 0.9% | 0.5 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3) \| if (((int)threadIdx.x) < total) \| if (((int)threadIdx.` |
| wait.sync#17 | 0.04 | 0.7% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#18 | 0.04 | 0.7% | 0.5 | 0.13 | `__syncthreads();` |
| wait.sync#16 | 0.04 | 0.7% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#8 | 0.04 | 0.7% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#11 | 0.03 | 0.6% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#15 | 0.03 | 0.6% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#3 | 0.03 | 0.6% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#9 | 0.03 | 0.6% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#2 | 0.03 | 0.6% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#7 | 0.03 | 0.6% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#12 | 0.03 | 0.5% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#21 | 0.03 | 0.5% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#4 | 0.03 | 0.5% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#22 | 0.03 | 0.5% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#10 | 0.03 | 0.5% | 0.5 | 0.06 | `__syncthreads();` |

### per_token_cast_kernel_kernel grid (18, 1, 1) block (64, 1, 1)

102 launches, 36 warps; ranges, warp and last warp from the 5 recording CTAs' 10 warps; span 1.50 us, warp 1.30 us, nsys graphs 1.86 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.50 | 38.3% | 0.79 | 47.3% |
| unattributed | 0.37 | 28.2% | 0.41 | 24.5% |
| store | 0.33 | 25.6% | 0.36 | 21.7% |
| compute | 0.10 | 7.9% | 0.11 | 6.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.41 | 31.4% | 1.0 | 0.54 | `for (int i = 0; i < 2; ++i)` |
| store.global#2 | 0.12 | 8.9% | 1.0 | 0.14 | `for (int i_3 = 0; i_3 < 4; ++i_3)` |
| store.global*2 (est.: sampled iterations x2) | 0.12 | 8.9% | 1.0 | 0.13 | `out_sf[((((((int64_t)((int)blockIdx.x)) % (int64_t)15) * (int64_t)32) + (((int64_t)((int)t` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.10 | 7.9% | 1.0 | 0.13 | `amax_fragment[i_1] = tl::AllReduce<tl::MaxOp, 2, 1, 0, tl::NamedBarrier<64>>::run(amax_fra` |
| store.global#3*2 (est.: sampled iterations x2) | 0.10 | 7.9% | 1.0 | 0.13 | `*(fp8_e4_16_t*)(out + ((((((((int64_t)((int)blockIdx.x)) / (int64_t)3) * (int64_t)6144) + ` |
| load.global#2*2 (est.: sampled iterations x2) | 0.09 | 6.9% | 1.0 | 0.13 | `amax_fragment[i_1] = cutlass::bfloat16_t(__hmax((bfloat16_t(((nv_bfloat162*)(&(amax_fragme` |

### indexer_topk_merge_kernel grid (1, 48, 1) block (512, 1, 1)

17 launches, 768 warps; ranges, warp and last warp from the 4 recording CTAs' 64 warps; span 9.79 us, warp 9.49 us, nsys graphs 4.48 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 6.18 | 65.1% | 5.94 | 60.7% |
| unattributed | 2.27 | 23.9% | 2.46 | 25.1% |
| load | 0.98 | 10.3% | 1.28 | 13.1% |
| store | 0.07 | 0.7% | 0.11 | 1.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.98 | 10.3% | 1.0 | 1.22 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 8192);` |
| wait.sync#20*9 (est.: sampled iterations x9) | 0.69 | 7.3% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#19*9 (est.: sampled iterations x9) | 0.66 | 6.9% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#24*9 (est.: sampled iterations x9) | 0.66 | 6.9% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#23*9 (est.: sampled iterations x9) | 0.63 | 6.7% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#14*8 (est.: sampled iterations x8) | 0.61 | 6.4% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync#6*8 (est.: sampled iterations x8) | 0.60 | 6.4% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync#5*8 (est.: sampled iterations x8) | 0.58 | 6.1% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync#13*8 (est.: sampled iterations x8) | 0.56 | 5.9% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync | 0.17 | 1.8% | 1.0 | 0.45 | `__syncthreads();` |
| wait.sync#17 | 0.08 | 0.8% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#16 | 0.08 | 0.8% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#8 | 0.07 | 0.8% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#2 | 0.07 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#9 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#25 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| store.global#2 | 0.06 | 0.7% | 1.0 | 0.10 | `Values[((((int64_t)((int)blockIdx.y)) * (int64_t)512) + ((int64_t)((int)threadIdx.x)))] = ` |
| wait.sync#15 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#3 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#11 | 0.06 | 0.6% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#7 | 0.06 | 0.6% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#12 | 0.06 | 0.6% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#22 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#18 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#4 | 0.06 | 0.6% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#10 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#21 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| load.global#2 | 0.01 | 0.1% | 0.1 | 0.10 | `if (((0 <= ((int*)sel_idx)[((int)threadIdx.x)]) && (((int*)sel_idx)[((int)threadIdx.x)] < ` |
| store.global | 0.00 | 0.0% | 0.1 | 0.06 | `Values[((((int64_t)((int)blockIdx.y)) * (int64_t)512) + ((int64_t)((int)threadIdx.x)))] = ` |

### activation_swiglu_clamp_kernel grid (1, 48, 1) block (128, 1, 1)

102 launches, 192 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 0.99 us, warp 0.76 us, nsys graphs 1.95 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.65 | 85.8% | 1.34 | 88.2% |
| load | 0.07 | 9.5% | 0.12 | 8.1% |
| store | 0.04 | 4.7% | 0.06 | 3.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.07 | 9.5% | 0.8 | 0.10 | `*(uint4*)(gate + 0) = *(uint4*)(X + ((((int64_t)((int)blockIdx.y)) * (int64_t)1536) + (((i` |
| store.global | 0.04 | 4.7% | 0.8 | 0.06 | `*(uint4*)(Y + ((((int64_t)((int)blockIdx.y)) * (int64_t)768) + (((int64_t)((int)threadIdx.` |

### norm_forward_and_per_token_cast_kernel_kernel grid (20, 1, 1) block (128, 1, 1)

51 launches, 80 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 2.40 us, warp 2.20 us, nsys graphs 2.14 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 1.58 | 71.9% | 1.74 | 61.4% |
| load | 0.35 | 16.0% | 0.34 | 11.9% |
| compute | 0.12 | 5.4% | 0.12 | 4.2% |
| wait | 0.11 | 5.0% | 0.04 | 1.5% |
| unattributed | 0.04 | 1.7% | 0.60 | 21.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*3 (est.: sampled iterations x3) | 1.08 | 49.2% | 1.0 | 1.15 | `for (int j_2 = 0; j_2 < 2; ++j_2)` |
| store.global*3 (est.: sampled iterations x3) | 0.29 | 13.4% | 1.0 | 0.38 | `for (int j_1 = 0; j_1 < 4; ++j_1)` |
| load.global*3 (est.: sampled iterations x3) | 0.18 | 8.2% | 1.0 | 0.19 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 6)) < num_tokens) && (((vi * 2) + (` |
| load.global#3*3 (est.: sampled iterations x3) | 0.14 | 6.3% | 1.0 | 0.19 | `if (((vi_1 * 2) + ((((int)threadIdx.x) & 63) >> 5)) < 5)` |
| store.global#5*3 (est.: sampled iterations x3) | 0.13 | 6.0% | 1.0 | 0.19 | `*(uint4*)(out + ((((((int64_t)vi_1) * (int64_t)512) + ((((int64_t)((int)threadIdx.x)) & (i` |
| compute.reduce | 0.12 | 5.4% | 1.0 | 0.13 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| wait.sync | 0.11 | 5.0% | 1.0 | 0.19 | `__syncthreads();` |
| store.global#2 | 0.04 | 2.0% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| load.global#2 | 0.03 | 1.5% | 1.0 | 0.06 | `*(float2*)(row_partial + 0) = *(float2*)(sum_shared + ((((int)threadIdx.x) >> 6) * 2)); \| ` |
| store.global#3 | 0.03 | 1.3% | 0.5 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |

### norm_forward_and_per_token_cast_kernel_kernel grid (5, 1, 1) block (128, 1, 1)

51 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 2.78 us, warp 2.76 us, nsys graphs 1.79 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 2.22 | 80.6% | 2.00 | 72.0% |
| load | 0.46 | 16.6% | 0.45 | 16.3% |
| compute | 0.11 | 4.1% | 0.12 | 4.1% |
| wait | 0.05 | 1.9% | 0.06 | 2.3% |
| unattributed | 0.00 | 0.0% | 0.15 | 5.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*4 (est.: sampled iterations x4) | 1.54 | 55.7% | 1.0 | 1.54 | `for (int j_2 = 0; j_2 < 2; ++j_2)` |
| store.global*4 (est.: sampled iterations x4) | 0.42 | 15.1% | 1.0 | 0.51 | `for (int j_1 = 0; j_1 < 4; ++j_1)` |
| load.global*4 (est.: sampled iterations x4) | 0.24 | 8.6% | 1.0 | 0.26 | `if (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 4)) < num_tokens)` |
| load.global#3*4 (est.: sampled iterations x4) | 0.19 | 6.7% | 1.0 | 0.26 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 128) + ((((int)threadIdx.x) & 15` |
| store.global#5*4 (est.: sampled iterations x4) | 0.17 | 6.3% | 1.0 | 0.26 | `*(uint4*)(out + ((((((int64_t)vi_1) * (int64_t)128) + ((((int64_t)((int)threadIdx.x)) & (i` |
| compute.reduce | 0.11 | 4.1% | 1.0 | 0.13 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| wait.sync | 0.05 | 1.9% | 1.0 | 0.10 | `__syncthreads();` |
| store.global#3 | 0.05 | 1.9% | 1.0 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |
| store.global#2 | 0.04 | 1.6% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| load.global#2 | 0.04 | 1.3% | 1.0 | 0.06 | `*(float2*)(row_partial + 0) = *(float2*)(sum_shared + ((((int)threadIdx.x) >> 6) * 2)); \| ` |

### router_partials_kernel grid (20, 1, 1) block (64, 1, 1)

51 launches, 40 warps; ranges, warp and last warp from the 4 recording CTAs' 8 warps; span 2.53 us, warp 2.49 us, nsys graphs 6.86 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.04 | 42.0% | 1.07 | 39.4% |
| unattributed | 0.87 | 34.9% | 0.94 | 34.7% |
| compute | 0.35 | 14.0% | 0.35 | 12.9% |
| store | 0.23 | 9.2% | 0.35 | 12.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#4 | 0.52 | 20.7% | 1.0 | 0.54 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < image_len)) \| if ((((((int)bl` |
| load.global | 0.40 | 16.1% | 1.0 | 0.42 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < mask_len))` |
| compute.reduce | 0.35 | 14.0% | 1.0 | 0.35 | `for (int k = 0; k < 3; ++k) \| for (int k = 0; k < 6; ++k)` |
| store.global#3 | 0.12 | 4.7% | 1.0 | 0.13 | `if ((((int)threadIdx.x) & 31) < 3) \| if ((((int)threadIdx.x) & 31) < 6)` |
| load.global#2 | 0.06 | 2.6% | 1.0 | 0.06 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < mask_len))` |
| load.global#3 | 0.06 | 2.6% | 1.0 | 0.06 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < mask_len))` |
| store.global | 0.06 | 2.3% | 1.0 | 0.06 | `if (condval == (bool)0) \| if (condval == (bool)1)` |
| store.global#2 | 0.06 | 2.3% | 1.0 | 0.06 | `if (condval_1 == (bool)0) \| if (condval_1 == (bool)1)` |

### norm_forward_and_per_token_cast_kernel_kernel grid (2, 1, 1) block (128, 1, 1)

48 launches, 8 warps; ranges, warp and last warp from the 2 recording CTAs' 8 warps; span 2.85 us, warp 2.85 us, nsys graphs 1.41 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 2.64 | 92.6% | 2.61 | 91.1% |
| load | 0.43 | 15.2% | 0.45 | 15.6% |
| compute | 0.11 | 3.9% | 0.11 | 3.8% |
| wait | 0.05 | 1.9% | 0.05 | 1.7% |
| unattributed | 0.00 | 0.0% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*4 (est.: sampled iterations x4) | 1.92 | 67.4% | 1.0 | 1.92 | `for (int j_2 = 0; j_2 < 2; ++j_2)` |
| store.global*4 (est.: sampled iterations x4) | 0.43 | 15.2% | 1.0 | 0.51 | `for (int j_1 = 0; j_1 < 4; ++j_1)` |
| load.global*4 (est.: sampled iterations x4) | 0.24 | 8.4% | 1.0 | 0.26 | `if (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 4)) < num_tokens)` |
| store.global#5*4 (est.: sampled iterations x4) | 0.18 | 6.5% | 1.0 | 0.26 | `*(uint4*)(out + ((((((int64_t)vi_1) * (int64_t)128) + ((((int64_t)((int)threadIdx.x)) & (i` |
| load.global#3*4 (est.: sampled iterations x4) | 0.16 | 5.6% | 1.0 | 0.26 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 128) + ((((int)threadIdx.x) & 15` |
| compute.reduce | 0.11 | 3.9% | 1.0 | 0.13 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| wait.sync | 0.05 | 1.9% | 1.0 | 0.06 | `__syncthreads();` |
| store.global#3 | 0.05 | 1.8% | 1.0 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |
| store.global#2 | 0.05 | 1.7% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| load.global#2 | 0.03 | 1.1% | 1.0 | 0.03 | `*(float2*)(row_partial + 0) = *(float2*)(sum_shared + ((((int)threadIdx.x) >> 6) * 2)); \| ` |

### token_ranks_kernel grid (16, 1, 1) block (256, 1, 1)

16 launches, 128 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 8.48 us, warp 8.28 us, nsys graphs 9.87 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 7.48 | 90.3% | 7.79 | 90.8% |
| compute | 0.39 | 4.7% | 0.37 | 4.4% |
| unattributed | 0.37 | 4.5% | 0.38 | 4.4% |
| store | 0.04 | 0.5% | 0.04 | 0.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 7.48 | 90.3% | 1.0 | 7.86 | `if ((((int64_t)0 <= token_ids[((int64_t)((int)blockIdx.x))]) && (token_ids[((int64_t)((int` |
| compute.reduce | 0.39 | 4.7% | 1.0 | 0.77 | `total[0] = tl::AllReduce<tl::SumOp, 256, 1, 0, tl::NamedBarrier<256>>::run(total[0], (&(wo` |
| store.global | 0.04 | 0.5% | 1.0 | 0.06 | `output[((int64_t)((int)blockIdx.x))] = ((int64_t)total[0]);` |

### topk_ids_kernel grid (48, 1, 1) block (512, 1, 1)

1 launches, 768 warps; ranges, warp and last warp from the 4 recording CTAs' 64 warps; span 137.22 us, warp 117.20 us, nsys graphs 123.46 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 88.04 | 75.1% | 116.00 | 89.7% |
| wait | 25.26 | 21.5% | 8.10 | 6.3% |
| store | 1.95 | 1.7% | 3.23 | 2.5% |
| unattributed | 1.95 | 1.7% | 1.98 | 1.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 88.04 | 75.1% | 1.0 | 123.90 | `for (int t = 0; t < (vocab >> 14); ++t)` |
| wait.sync | 25.12 | 21.4% | 1.0 | 80.70 | `__syncthreads();` |
| store.global | 1.95 | 1.7% | 1.0 | 20.86 | `if ((((int)threadIdx.x) >> 5) == 0)` |
| wait.sync#2 | 0.08 | 0.1% | 1.0 | 0.13 | `__syncthreads();` |
| wait.sync#3 | 0.06 | 0.1% | 1.0 | 0.06 | `__syncthreads();` |

### gemm_splitk_reduce_cast_kernel grid (24, 10, 1) block (128, 1, 1)

51 launches, 960 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 2.18 us, warp 1.66 us, nsys graphs 2.11 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.11 | 66.9% | 1.87 | 73.1% |
| unattributed | 0.38 | 22.8% | 0.52 | 20.4% |
| store | 0.09 | 5.3% | 0.08 | 3.2% |
| compute | 0.08 | 5.1% | 0.08 | 3.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.11 | 66.9% | 1.0 | 1.63 | `if ((1 <= M))` |
| store.global | 0.09 | 5.3% | 1.0 | 0.10 | `if (((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 5)) < M)` |
| compute.reduce | 0.08 | 5.1% | 1.0 | 0.10 | `amax[0] = tl::AllReduce<tl::MaxOp, 8, 1, 0, tl::NamedBarrier<128>>::run(amax[0]);` |

### indexer_mxfp4_quantize_kernel grid (2, 1, 1) block (128, 1, 1)

34 launches, 8 warps; ranges, warp and last warp from the 2 recording CTAs' 8 warps; span 3.71 us, warp 1.63 us, nsys graphs 2.56 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.95 | 58.0% | 2.25 | 58.6% |
| unattributed | 0.42 | 25.6% | 0.43 | 11.3% |
| store | 0.27 | 16.4% | 1.16 | 30.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.69 | 42.3% | 0.8 | 1.09 | `for (int i = 0; i < 32; ++i)` |
| load.global#2*16 (est.: sampled iterations x16) | 0.26 | 15.7% | 0.2 | 1.02 | `float v__2 = __fmul_rn(((float)X[(((((int64_t)((int)blockIdx.x)) * (int64_t)4096) + (((int` |
| store.global*16 (est.: sampled iterations x16) | 0.26 | 15.7% | 0.2 | 1.02 | `Pool[(((((slot & (int64_t)127) * (int64_t)64) + ((((int64_t)((int)threadIdx.x)) & (int64_t` |
| store.global#2 | 0.01 | 0.7% | 0.2 | 0.06 | `Pool[(((((slot & (int64_t)127) * (int64_t)4) + ((slot >> (int64_t)7) * ((int64_t)pool_stri` |

### indexer_mxfp4_quantize_kernel grid (48, 1, 1) block (128, 1, 1)

34 launches, 192 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 3.81 us, warp 3.71 us, nsys graphs 3.01 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 2.31 | 62.2% | 2.35 | 61.7% |
| store | 1.13 | 30.5% | 1.15 | 30.1% |
| unattributed | 0.27 | 7.4% | 0.31 | 8.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2*16 (est.: sampled iterations x16) | 1.26 | 34.0% | 1.0 | 1.54 | `float v__2 = __fmul_rn(((float)X[(((((int64_t)((int)blockIdx.x)) * (int64_t)4096) + (((int` |
| store.global*16 (est.: sampled iterations x16) | 1.09 | 29.3% | 1.0 | 1.54 | `Pool[(((((slot & (int64_t)127) * (int64_t)64) + ((((int64_t)((int)threadIdx.x)) & (int64_t` |
| load.global | 1.04 | 28.1% | 1.0 | 1.12 | `for (int i = 0; i < 32; ++i)` |
| store.global#2 | 0.04 | 1.2% | 1.0 | 0.06 | `Pool[(((((slot & (int64_t)127) * (int64_t)4) + ((slot >> (int64_t)7) * ((int64_t)pool_stri` |

### gemm_bf16_partials_kernel grid (1, 40, 1) block (256, 1, 1)

34 launches, 320 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 3.57 us, warp 2.13 us, nsys graphs 3.18 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.34 | 62.8% | 2.53 | 69.5% |
| wait | 0.36 | 16.7% | 0.37 | 10.2% |
| compute | 0.29 | 13.7% | 0.59 | 16.2% |
| store | 0.07 | 3.5% | 0.15 | 4.1% |
| load | 0.07 | 3.3% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*2 (est.: sampled iterations x2) | 0.29 | 13.7% | 0.5 | 0.64 | `{` |
| wait.mbarrier#2*2 (est.: sampled iterations x2) | 0.20 | 9.2% | 0.5 | 0.74 | `mbarrier[kb_1].wait(0);` |
| wait.sync | 0.10 | 4.6% | 1.0 | 0.13 | `__syncthreads();` |
| store.global | 0.07 | 3.5% | 0.5 | 0.18 | `for (int i_1 = 0; i_1 < 4; ++i_1) \| for (int i_2 = 0; i_2 < 8; ++i_2) \| for (int i_4 = 0; ` |
| load.tma*2 (est.: sampled iterations x2) | 0.07 | 3.3% | 0.5 | 0.19 | `if (tl::tl_shuffle_elect<128>())` |
| wait.mbarrier*2 (est.: sampled iterations x2) | 0.06 | 3.0% | 0.5 | 0.13 | `mbarrier[(kb + 3)].wait(1);` |

### prepare_dflash_inputs_kernel grid (9, 8, 1) block (256, 1, 1)

51 launches, 576 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 2.11 us, warp 1.86 us, nsys graphs 2.40 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.41 | 75.7% | 1.70 | 78.2% |
| unattributed | 0.39 | 20.9% | 0.41 | 19.0% |
| store | 0.06 | 3.4% | 0.06 | 2.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.35 | 72.5% | 1.0 | 1.60 | `req = idx_mapping[((int)blockIdx.y)];` |
| store.global#2 | 0.06 | 3.4% | 1.0 | 0.10 | `if (((int)blockIdx.y) == (num_reqs - 1))` |
| load.global#2 | 0.06 | 3.2% | 1.0 | 0.10 | `vllm_store_if_b32((&(out_query_start_loc[((int)blockIdx.y)])), (((int)blockIdx.y) * num_qu` |

### resample_kernel grid (8, 127, 1) block (128, 1, 1)

17 launches, 4064 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 6.59 us, warp 2.85 us, nsys graphs 6.21 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.69 | 59.2% | 1.13 | 49.7% |
| unattributed | 0.69 | 24.1% | 0.65 | 28.3% |
| compute | 0.30 | 10.7% | 0.32 | 13.9% |
| wait | 0.10 | 3.4% | 0.09 | 4.0% |
| store | 0.08 | 2.7% | 0.09 | 4.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.69 | 59.2% | 1.0 | 2.50 | `end = cu_num_logits[(((int64_t)((int)blockIdx.x)) + (int64_t)1)];` |
| compute.reduce#2 | 0.15 | 5.3% | 1.0 | 0.19 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 8, 32);` |
| wait.sync | 0.10 | 3.4% | 1.0 | 0.19 | `__syncthreads();` |
| compute.reduce#3 | 0.08 | 2.8% | 1.0 | 0.10 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 2, 32);` |
| store.global | 0.08 | 2.7% | 1.0 | 0.13 | `if (((int)threadIdx.x) == 0)` |
| compute.reduce | 0.07 | 2.6% | 1.0 | 0.13 | `if ((temp != 0x0p+0f/*0.000000e+00*/) \| bonus)` |

### local_stats_kernel grid (48, 16, 1) block (128, 1, 1)

1 launches, 3072 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 108.58 us, warp 34.37 us, nsys graphs 84.16 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 30.70 | 89.3% | 24.74 | 70.4% |
| wait | 1.86 | 5.4% | 2.69 | 7.7% |
| unattributed | 1.22 | 3.6% | 6.05 | 17.2% |
| compute | 0.51 | 1.5% | 1.54 | 4.4% |
| store | 0.07 | 0.2% | 0.13 | 0.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 30.70 | 89.3% | 1.0 | 43.36 | `if (expanded_local_pos[((int64_t)((int)blockIdx.x))] < num_speculative_steps)` |
| wait.sync | 1.86 | 5.4% | 1.0 | 8.67 | `__syncthreads();` |
| compute.reduce | 0.45 | 1.3% | 1.0 | 4.06 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| store.global | 0.07 | 0.2% | 1.0 | 0.13 | `if (((int)threadIdx.x) == 0)` |
| compute.reduce#2 | 0.06 | 0.2% | 1.0 | 0.06 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 2, 32);` |

### metadata_chunk_kernel grid (40, 1, 1) block (192, 1, 1)

51 launches, 240 warps; ranges, warp and last warp from the 4 recording CTAs' 24 warps; span 2.21 us, warp 1.53 us, nsys graphs 2.30 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.09 | 71.3% | 1.49 | 70.9% |
| unattributed | 0.38 | 24.7% | 0.48 | 23.0% |
| store | 0.06 | 4.0% | 0.13 | 6.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.05 | 68.6% | 1.0 | 1.34 | `if ((((offset + ((int)blockIdx.x)) < src_rows) && (0 <= (offset + ((int)blockIdx.x)))))` |
| store.global | 0.05 | 3.1% | 1.0 | 0.06 | `swa[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = con` |
| load.global#2 | 0.04 | 2.7% | 0.2 | 0.26 | `if (((0 <= req) & ((int64_t)0 <= pos)))` |
| store.global#2 | 0.01 | 0.4% | 0.2 | 0.06 | `lengths[((int64_t)((int)blockIdx.x))] = ((int)max(info[1], (int64_t)0));` |
| store.global#3 | 0.01 | 0.4% | 0.2 | 0.06 | `top_lengths[((int64_t)((int)blockIdx.x))] = condval_12; \| top_lengths[((int64_t)((int)bloc` |

### gemm_bf16_partials_kernel grid (2, 4, 1) block (256, 1, 1)

34 launches, 64 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 2.50 us, warp 1.63 us, nsys graphs 2.69 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.95 | 58.4% | 1.59 | 59.8% |
| compute | 0.29 | 17.6% | 0.58 | 21.6% |
| wait | 0.22 | 13.6% | 0.29 | 10.8% |
| store | 0.11 | 6.5% | 0.21 | 7.8% |
| load | 0.06 | 3.9% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*2 (est.: sampled iterations x2) | 0.29 | 17.6% | 0.5 | 0.58 | `{` |
| store.global | 0.11 | 6.5% | 0.5 | 0.22 | `for (int i_1 = 0; i_1 < 4; ++i_1) \| for (int i_2 = 0; i_2 < 8; ++i_2) \| for (int i_4 = 0; ` |
| wait.sync | 0.10 | 6.5% | 1.0 | 0.13 | `__syncthreads();` |
| load.tma*2 (est.: sampled iterations x2) | 0.06 | 3.9% | 0.5 | 0.19 | `if (tl::tl_shuffle_elect<128>())` |
| wait.mbarrier#2*2 (est.: sampled iterations x2) | 0.06 | 3.6% | 0.5 | 0.13 | `mbarrier[kb_1].wait(0);` |
| wait.mbarrier*2 (est.: sampled iterations x2) | 0.06 | 3.6% | 0.5 | 0.13 | `mbarrier[(kb + 3)].wait(1);` |

### norm_forward_and_per_token_cast_kernel_kernel grid (3, 1, 1) block (128, 1, 1)

34 launches, 12 warps; ranges, warp and last warp from the 3 recording CTAs' 12 warps; span 2.18 us, warp 2.15 us, nsys graphs 1.89 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 1.15 | 53.5% | 1.30 | 54.7% |
| unattributed | 0.63 | 29.4% | 0.68 | 28.5% |
| load | 0.23 | 10.6% | 0.26 | 10.7% |
| compute | 0.09 | 4.0% | 0.09 | 3.6% |
| wait | 0.05 | 2.5% | 0.06 | 2.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*2 (est.: sampled iterations x2) | 0.74 | 34.6% | 1.0 | 0.77 | `for (int j_2 = 0; j_2 < 2; ++j_2)` |
| store.global*2 (est.: sampled iterations x2) | 0.21 | 9.7% | 1.0 | 0.26 | `for (int j_1 = 0; j_1 < 4; ++j_1)` |
| load.global*2 (est.: sampled iterations x2) | 0.12 | 5.5% | 1.0 | 0.13 | `if (((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < num_tokens)` |
| store.global#5*2 (est.: sampled iterations x2) | 0.10 | 4.5% | 1.0 | 0.13 | `*(uint4*)(out + ((((((int64_t)vi_1) * (int64_t)64) + ((((int64_t)((int)threadIdx.x)) & (in` |
| compute.reduce | 0.09 | 4.0% | 1.0 | 0.10 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| load.global#3*2 (est.: sampled iterations x2) | 0.08 | 3.6% | 1.0 | 0.13 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 64) + ((((int)threadIdx.x) & 7) ` |
| store.global#3 | 0.06 | 2.9% | 1.0 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |
| wait.sync | 0.05 | 2.5% | 1.0 | 0.06 | `__syncthreads();` |
| store.global#2 | 0.04 | 1.9% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| load.global#2 | 0.03 | 1.5% | 1.0 | 0.03 | `*(float2*)(row_partial + 0) = *(float2*)(sum_shared + ((((int)threadIdx.x) >> 6) * 2)); \| ` |

### topk_log_softmax_kernel grid (48, 1, 1) block (1024, 1, 1)

1 launches, 1536 warps; ranges, warp and last warp from the 4 recording CTAs' 128 warps; span 78.02 us, warp 72.58 us, nsys graphs 62.27 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 61.14 | 84.2% | 69.92 | 89.8% |
| wait | 9.47 | 13.1% | 5.73 | 7.4% |
| unattributed, with loops' unsampled iterations | 1.59 | 2.2% | 1.79 | 2.3% |
| compute | 0.37 | 0.5% | 0.35 | 0.5% |
| store | 0.00 | 0.0% | 0.06 | 0.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 39.50 | 54.4% | 1.0 | 48.26 | `if (0 < (vocab & 2047)) \| if (0 < (vocab & 8191))` |
| load.global#2 | 21.55 | 29.7% | 1.0 | 27.78 | `for (int step_1 = 0; step_1 < (vocab >> 11); ++step_1) \| for (int step_1 = 0; step_1 < (vo` |
| wait.sync | 5.57 | 7.7% | 1.0 | 20.54 | `__syncthreads();` |
| wait.sync#2 | 3.63 | 5.0% | 1.0 | 11.42 | `__syncthreads();` |
| compute.reduce#3 | 0.17 | 0.2% | 1.0 | 0.22 | `ties[0] = vllm_add(ties[0], __shfl_xor_sync((uint)4294967295, ties[0], 16, 32));` |
| wait.sync#4 | 0.11 | 0.1% | 1.0 | 0.16 | `__syncthreads();` |
| compute.reduce#2 | 0.10 | 0.1% | 1.0 | 0.16 | `se[0] = vllm_add(se[0], __shfl_xor_sync((uint)4294967295, se[0], 16, 32));` |
| compute.reduce | 0.10 | 0.1% | 1.0 | 0.10 | `best[0] = vllm_max(best[0], __shfl_xor_sync((uint)4294967295, best[0], 16, 32));` |
| load.global#3 (sampled iterations) | 0.09 | 0.1% | 0.0 | 9.06 | `float value_32 = vllm_sub(vllm_shifted(logits[((((int64_t)((int)blockIdx.x)) * logits_stri` |
| wait.sync#5 | 0.08 | 0.1% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#3 | 0.08 | 0.1% | 1.0 | 0.10 | `__syncthreads();` |
| store.global (sampled iterations) | 0.00 | 0.0% | 0.0 | 0.06 | `output[(((((int64_t)c) * (int64_t)1024) + (((int64_t)((int)blockIdx.x)) * ((int64_t)topk))` |

### per_token_cast_kernel_kernel grid (25, 1, 1) block (64, 1, 1)

51 launches, 50 warps; ranges, warp and last warp from the 5 recording CTAs' 10 warps; span 1.15 us, warp 1.09 us, nsys graphs 1.34 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.36 | 32.9% | 0.36 | 24.8% |
| load | 0.34 | 31.5% | 0.54 | 37.4% |
| store | 0.29 | 26.8% | 0.45 | 30.9% |
| compute | 0.10 | 8.8% | 0.10 | 6.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.26 | 23.8% | 1.0 | 0.29 | `for (int i = 0; i < 2; ++i)` |
| store.global#2 | 0.11 | 10.3% | 1.0 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.10 | 8.8% | 1.0 | 0.13 | `amax_fragment[i_1] = tl::AllReduce<tl::MaxOp, 2, 1, 0, tl::NamedBarrier<64>>::run(amax_fra` |
| store.global#3*2 (est.: sampled iterations x2) | 0.09 | 8.2% | 1.0 | 0.13 | `*(fp8_e4_16_t*)(out + ((((((((int64_t)((int)blockIdx.x)) / (int64_t)3) * (int64_t)6144) + ` |
| store.global*2 (est.: sampled iterations x2) | 0.09 | 8.2% | 1.0 | 0.13 | `out_sf[((((((int64_t)((int)blockIdx.x)) % (int64_t)15) * (int64_t)32) + (((int64_t)((int)t` |
| load.global#2*2 (est.: sampled iterations x2) | 0.08 | 7.6% | 1.0 | 0.13 | `amax_fragment[i_1] = cutlass::bfloat16_t(__hmax((bfloat16_t(((nv_bfloat162*)(&(amax_fragme` |

### mhc_collapse_kernel grid (200, 1, 1) block (128, 1, 1)

17 launches, 800 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 4.42 us, warp 2.90 us, nsys graphs 1.94 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 2.60 | 89.4% | 2.64 | 69.7% |
| load | 0.25 | 8.7% | 1.09 | 28.8% |
| store | 0.06 | 1.9% | 0.05 | 1.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.25 | 8.7% | 1.0 | 0.96 | `*(float4*)(mv + 0) = *(float4*)(mix + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int6` |
| store.global | 0.06 | 1.9% | 1.0 | 0.13 | `*(uint4*)(out + ((((int64_t)((int)blockIdx.x)) * (int64_t)1024) + (((int64_t)((int)threadI` |

### norm_forward_and_per_token_cast_kernel_kernel grid (40, 1, 1) block (128, 1, 1)

17 launches, 160 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 3.42 us, warp 3.15 us, nsys graphs 2.37 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 1.86 | 59.1% | 2.42 | 56.8% |
| unattributed | 0.64 | 20.2% | 1.16 | 27.3% |
| load | 0.44 | 14.0% | 0.45 | 10.6% |
| compute | 0.13 | 4.2% | 0.14 | 3.2% |
| wait | 0.08 | 2.5% | 0.09 | 2.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*5 (est.: sampled iterations x5) | 1.59 | 50.4% | 1.0 | 2.24 | `for (int j_1 = 0; j_1 < 2; ++j_1)` |
| load.global#2*5 (est.: sampled iterations x5) | 0.22 | 7.0% | 1.0 | 0.32 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 1024) + (((int)threadIdx.x) * 8)` |
| load.global*5 (est.: sampled iterations x5) | 0.22 | 7.0% | 1.0 | 0.32 | `*(uint4*)(x_local + (vi * 8)) = *(uint4*)(x + ((((((int64_t)vi) * (int64_t)1024) + (((int6` |
| store.global*5 (est.: sampled iterations x5) | 0.22 | 7.0% | 1.0 | 0.32 | `for (int j = 0; j < 4; ++j)` |
| compute.reduce | 0.13 | 4.2% | 1.0 | 0.16 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| wait.sync | 0.08 | 2.5% | 1.0 | 0.19 | `__syncthreads();` |
| store.global#2 | 0.04 | 1.3% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| store.global#3 | 0.01 | 0.4% | 0.2 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |

### sink_merge_kernel grid (8, 1, 1) block (128, 1, 1)

17 launches, 32 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 4.10 us, warp 4.05 us, nsys graphs 4.45 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.93 | 47.6% | 2.07 | 50.7% |
| wait | 0.82 | 20.3% | 0.44 | 10.8% |
| unattributed | 0.71 | 17.4% | 0.87 | 21.3% |
| compute | 0.56 | 13.9% | 0.57 | 13.9% |
| store | 0.03 | 0.7% | 0.14 | 3.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2*3 (est.: sampled iterations x3) | 1.64 | 40.5% | 1.0 | 1.73 | `for (int step = 0; step < ((width + 127) >> 7); ++step) \| for (int step = 0; step < ((widt` |
| compute.reduce*3 (est.: sampled iterations x3) | 0.56 | 13.9% | 1.0 | 0.58 | `float other = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| wait.sync#3*3 (est.: sampled iterations x3) | 0.52 | 12.7% | 1.0 | 0.67 | `__syncthreads();` |
| load.global | 0.24 | 5.9% | 1.0 | 0.26 | `last = sampled[(max((((int64_t)num_sampled[((int64_t)((int)blockIdx.x))]) - (int64_t)1), (` |
| wait.sync*3 (est.: sampled iterations x3) | 0.17 | 4.2% | 1.0 | 0.19 | `__syncthreads();` |
| wait.sync#2*3 (est.: sampled iterations x3) | 0.14 | 3.4% | 1.0 | 0.19 | `__syncthreads();` |
| load.global#3*3 (est.: sampled iterations x3) | 0.05 | 1.2% | 0.2 | 0.19 | `if ((best[0] == -CUDART_INF_F))` |
| store.global*3 (est.: sampled iterations x3) | 0.03 | 0.7% | 0.2 | 0.19 | `alternates[((((int64_t)((int)blockIdx.x)) * (int64_t)3) + ((int64_t)rank))] = condval_31;` |

### context_inputs_kernel grid (5, 16, 1) block (128, 1, 1)

16 launches, 320 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 3.81 us, warp 2.56 us, nsys graphs 4.13 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 2.16 | 84.2% | 3.66 | 87.9% |
| unattributed | 0.33 | 13.1% | 0.44 | 10.6% |
| store | 0.07 | 2.7% | 0.06 | 1.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 2.16 | 84.2% | 1.0 | 3.30 | `vllm_dspark_context_b128((&(aux[(((((int64_t)((int)blockIdx.y)) * (int64_t)15360) + (((int` |
| store.global | 0.07 | 2.7% | 1.0 | 0.10 | `if (((((int)blockIdx.x) == 0) & (((int)threadIdx.x) == 0)) & (num_tokens <= ((int)blockIdx` |

### finish_argmax_kernel grid (8, 1, 1) block (32, 1, 1)

85 launches, 8 warps; ranges, warp and last warp from the 4 recording CTAs' 4 warps; span 0.77 us, warp 0.73 us, nsys graphs 1.06 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.38 | 51.6% | 0.45 | 56.4% |
| load | 0.13 | 17.6% | 0.12 | 15.8% |
| compute | 0.12 | 16.5% | 0.12 | 15.1% |
| store | 0.10 | 14.3% | 0.10 | 12.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.13 | 17.6% | 1.0 | 0.13 | `for (int k = 0; k < 2; ++k) \| if ((((int)threadIdx.x) < parts))` |
| compute.reduce | 0.12 | 16.5% | 1.0 | 0.13 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| store.global | 0.10 | 14.3% | 1.0 | 0.13 | `if (((int)threadIdx.x) == 0)` |

### per_token_cast_kernel_kernel grid (15, 1, 1) block (64, 1, 1)

51 launches, 30 warps; ranges, warp and last warp from the 5 recording CTAs' 10 warps; span 1.12 us, warp 1.08 us, nsys graphs 1.33 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.36 | 33.0% | 0.43 | 33.9% |
| unattributed | 0.31 | 29.2% | 0.41 | 31.9% |
| store | 0.30 | 28.3% | 0.34 | 27.0% |
| compute | 0.10 | 9.5% | 0.09 | 7.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.27 | 24.7% | 1.0 | 0.29 | `for (int i = 0; i < 2; ++i)` |
| store.global*2 (est.: sampled iterations x2) | 0.11 | 10.1% | 1.0 | 0.13 | `out_sf[((((((int64_t)((int)blockIdx.x)) % (int64_t)15) * (int64_t)32) + (((int64_t)((int)t` |
| store.global#2 | 0.11 | 9.8% | 1.0 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.10 | 9.5% | 1.0 | 0.13 | `amax_fragment[i_1] = tl::AllReduce<tl::MaxOp, 2, 1, 0, tl::NamedBarrier<64>>::run(amax_fra` |
| load.global#2*2 (est.: sampled iterations x2) | 0.09 | 8.3% | 1.0 | 0.13 | `amax_fragment[i_1] = cutlass::bfloat16_t(__hmax((bfloat16_t(((nv_bfloat162*)(&(amax_fragme` |
| store.global#3*2 (est.: sampled iterations x2) | 0.09 | 8.3% | 1.0 | 0.13 | `*(fp8_e4_16_t*)(out + ((((((((int64_t)((int)blockIdx.x)) / (int64_t)3) * (int64_t)6144) + ` |

### mhc_collapse_column_kernel grid (240, 1, 1) block (128, 1, 1)

17 launches, 960 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 3.55 us, warp 1.57 us, nsys graphs 2.14 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.23 | 78.0% | 1.56 | 57.4% |
| load | 0.27 | 16.9% | 1.11 | 40.6% |
| store | 0.08 | 5.1% | 0.05 | 2.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.27 | 16.9% | 1.0 | 0.77 | `*(float4*)(mv + 0) = *(float4*)(mix + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int6` |
| store.global | 0.08 | 5.1% | 1.0 | 0.26 | `*(uint4*)(out + ((((int64_t)((int)blockIdx.x)) * (int64_t)1024) + (((int64_t)((int)threadI` |

### rope_kernel grid (40, 1, 1) block (192, 1, 1)

51 launches, 240 warps; ranges, warp and last warp from the 4 recording CTAs' 24 warps; span 1.25 us, warp 1.19 us, nsys graphs 1.57 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.68 | 57.3% | 0.73 | 58.6% |
| load | 0.47 | 39.4% | 0.47 | 37.9% |
| store | 0.04 | 3.3% | 0.04 | 3.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.47 | 39.4% | 1.0 | 0.48 | `if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < tokens)) \| int64_t position ` |
| store.global | 0.04 | 3.3% | 1.0 | 0.06 | `*(uint4*)(x + ((((((int64_t)((int)blockIdx.x)) * (int64_t)12288) + ((((int64_t)((int)threa` |

### activation_swiglu_clamp_kernel grid (1, 40, 1) block (128, 1, 1)

51 launches, 160 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 0.96 us, warp 0.75 us, nsys graphs 1.73 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.65 | 85.9% | 0.98 | 81.4% |
| load | 0.07 | 9.6% | 0.17 | 13.8% |
| store | 0.03 | 4.5% | 0.06 | 4.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.07 | 9.6% | 0.8 | 0.10 | `*(uint4*)(gate + 0) = *(uint4*)(X + ((((int64_t)((int)blockIdx.y)) * (int64_t)1536) + (((i` |
| store.global | 0.03 | 4.5% | 0.8 | 0.06 | `*(uint4*)(Y + ((((int64_t)((int)blockIdx.y)) * (int64_t)768) + (((int64_t)((int)threadIdx.` |

### dspark_confidence_kernel grid (40, 1, 1) block (128, 1, 1)

17 launches, 160 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 3.42 us, warp 3.13 us, nsys graphs 3.47 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 2.65 | 84.7% | 2.76 | 81.2% |
| wait | 0.18 | 5.8% | 0.16 | 4.8% |
| load | 0.18 | 5.6% | 0.29 | 8.6% |
| compute | 0.10 | 3.1% | 0.10 | 2.8% |
| store | 0.02 | 0.8% | 0.09 | 2.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync | 0.18 | 5.8% | 1.0 | 0.58 | `__syncthreads();` |
| load.global | 0.16 | 5.2% | 1.0 | 0.19 | `for (int c = 0; c < 5; ++c)` |
| compute.reduce | 0.10 | 3.1% | 1.0 | 0.10 | `acc[0] = tl::warp_reduce_sum(acc[0]);` |
| store.global | 0.02 | 0.8% | 0.2 | 0.10 | `out[((int64_t)((int)blockIdx.x))] = (0x1p+0f/*1.000000e+00*/ / (0x1p+0f/*1.000000e+00*/ + ` |
| load.global#2 | 0.01 | 0.4% | 0.2 | 0.06 | `total[0] = 0x0p+0f/*0.000000e+00*/;` |

### norm_forward_and_per_token_cast_kernel_kernel grid (48, 1, 1) block (128, 1, 1)

18 launches, 192 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 2.50 us, warp 2.37 us, nsys graphs 2.34 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 1.58 | 66.6% | 2.43 | 79.1% |
| load | 0.42 | 17.9% | 0.46 | 15.0% |
| unattributed | 0.18 | 7.6% | 0.00 | 0.0% |
| compute | 0.13 | 5.5% | 0.13 | 4.2% |
| wait | 0.06 | 2.4% | 0.06 | 2.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*5 (est.: sampled iterations x5) | 1.29 | 54.2% | 1.0 | 1.76 | `for (int j_1 = 0; j_1 < 2; ++j_1)` |
| store.global*5 (est.: sampled iterations x5) | 0.24 | 10.1% | 1.0 | 0.32 | `for (int j = 0; j < 4; ++j)` |
| load.global*5 (est.: sampled iterations x5) | 0.21 | 9.1% | 1.0 | 0.32 | `*(uint4*)(x_local + (vi * 8)) = *(uint4*)(x + ((((((int64_t)vi) * (int64_t)1024) + (((int6` |
| load.global#2*5 (est.: sampled iterations x5) | 0.21 | 8.9% | 1.0 | 0.32 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 1024) + (((int)threadIdx.x) * 8)` |
| compute.reduce | 0.13 | 5.5% | 1.0 | 0.16 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| wait.sync | 0.06 | 2.4% | 1.0 | 0.10 | `__syncthreads();` |
| store.global#2 | 0.04 | 1.7% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| store.global#3 | 0.01 | 0.5% | 0.2 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |

### adaptive_offsets_kernel grid (1, 1, 1) block (256, 1, 1)

16 launches, 8 warps; ranges, warp and last warp from the 1 recording CTAs' 8 warps; span 3.52 us, warp 3.52 us, nsys graphs 1.89 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 1.57 | 44.7% | 1.13 | 32.0% |
| unattributed, with loops' unsampled iterations | 1.46 | 41.6% | 1.52 | 43.1% |
| store | 0.31 | 8.7% | 0.49 | 13.9% |
| load | 0.18 | 5.1% | 0.39 | 11.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync#18 (sampled iterations) | 0.34 | 9.8% | 1.0 | 0.38 | `__syncthreads();` |
| wait.sync (sampled iterations) | 0.20 | 5.8% | 1.0 | 0.22 | `__syncthreads();` |
| store.global#5 | 0.12 | 3.5% | 1.0 | 0.13 | `for (int chunk_1 = 0; chunk_1 < ((starts_len + 255) >> 8); ++chunk_1)` |
| load.global#2 (sampled iterations) | 0.10 | 2.7% | 1.0 | 0.22 | `if ((((chunk * 256) + ((int)threadIdx.x)) < num_reqs))` |
| store.global#2 (sampled iterations) | 0.09 | 2.5% | 1.0 | 0.10 | `if ((((chunk * 256) + ((int)threadIdx.x)) < num_reqs) & (drafted == 0))` |
| load.global (sampled iterations) | 0.07 | 2.1% | 1.0 | 0.10 | `if (((((chunk * 256) + ((int)threadIdx.x)) < num_reqs) & (drafted != 0)))` |
| store.global | 0.07 | 2.1% | 1.0 | 0.10 | `if (((int)threadIdx.x) == 0)` |
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
| store.global#4 (sampled iterations) | 0.01 | 0.3% | 0.1 | 0.10 | `for (int j = 0; j < 6; ++j)` |
| load.global#3 (sampled iterations) | 0.01 | 0.2% | 0.1 | 0.06 | `state = idx_mapping[((((int64_t)chunk) * (int64_t)256) + ((int64_t)((int)threadIdx.x)))];` |
| store.global#3 (sampled iterations) | 0.01 | 0.2% | 0.1 | 0.06 | `cu_num_logits[(((((int64_t)chunk) * (int64_t)256) + ((int64_t)((int)threadIdx.x))) + (int6` |

### cache_writer_kernel grid (40, 1, 1) block (128, 1, 1)

51 launches, 160 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 1.09 us, warp 1.08 us, nsys graphs 1.31 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.41 | 38.4% | 0.41 | 37.8% |
| store | 0.30 | 27.5% | 0.26 | 23.7% |
| load | 0.29 | 27.1% | 0.35 | 31.5% |
| compute | 0.08 | 7.1% | 0.08 | 7.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.30 | 27.5% | 1.0 | 0.32 | `if (((int64_t)0 <= slot) & ((slot >> (int64_t)7) < ((int64_t)pages))) \| if (((int64_t)0 <=` |
| load.global | 0.23 | 21.7% | 1.0 | 0.26 | `for (int i = 0; i < 4; ++i)` |
| compute.reduce | 0.08 | 7.1% | 1.0 | 0.10 | `amax = max(amax, __shfl_xor_sync((uint)4294967295, amax, 1, 32));` |
| load.global#2 | 0.06 | 5.4% | 1.0 | 0.06 | `int64_t slot = slots[((int64_t)((int)blockIdx.x))];` |

### gather_block_tables_kernel grid (11, 8, 1) block (256, 1, 1)

17 launches, 704 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 2.34 us, warp 1.44 us, nsys graphs 2.43 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.14 | 79.3% | 2.76 | 89.8% |
| unattributed | 0.30 | 20.7% | 0.31 | 10.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.14 | 79.3% | 1.0 | 1.76 | `int g = chunk_groups[((int64_t)((int)blockIdx.x))];` |

### gemm_splitk_reduce_kernel grid (1, 48, 1) block (128, 1, 1)

68 launches, 192 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 0.86 us, warp 0.82 us, nsys graphs 0.99 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.44 | 53.8% | 0.40 | 51.1% |
| load | 0.33 | 40.1% | 0.33 | 42.1% |
| store | 0.05 | 6.1% | 0.05 | 6.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.33 | 40.1% | 1.0 | 0.40 | `acc[0] = P[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x))` |
| store.global | 0.05 | 6.1% | 1.0 | 0.06 | `C[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x)) * (int64` |

### per_token_cast_kernel_kernel grid (40, 1, 1) block (64, 1, 1)

48 launches, 80 warps; ranges, warp and last warp from the 4 recording CTAs' 8 warps; span 1.15 us, warp 1.08 us, nsys graphs 1.02 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.38 | 34.8% | 0.39 | 34.4% |
| unattributed | 0.31 | 29.0% | 0.32 | 28.4% |
| store | 0.30 | 27.4% | 0.32 | 28.3% |
| compute | 0.10 | 8.9% | 0.10 | 8.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.29 | 26.6% | 1.0 | 0.32 | `for (int i = 0; i < 2; ++i)` |
| store.global#2 | 0.11 | 10.4% | 1.0 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.10 | 8.9% | 1.0 | 0.13 | `amax_fragment[i_1] = tl::AllReduce<tl::MaxOp, 2, 1, 0, tl::NamedBarrier<64>>::run(amax_fra` |
| store.global*2 (est.: sampled iterations x2) | 0.10 | 8.9% | 1.0 | 0.13 | `out_sf[((((((int64_t)((int)blockIdx.x)) % (int64_t)15) * (int64_t)32) + (((int64_t)((int)t` |
| load.global#2*2 (est.: sampled iterations x2) | 0.09 | 8.1% | 1.0 | 0.13 | `amax_fragment[i_1] = cutlass::bfloat16_t(__hmax((bfloat16_t(((nv_bfloat162*)(&(amax_fragme` |
| store.global#3*2 (est.: sampled iterations x2) | 0.09 | 8.1% | 1.0 | 0.13 | `*(fp8_e4_16_t*)(out + ((((((((int64_t)((int)blockIdx.x)) / (int64_t)3) * (int64_t)6144) + ` |

### token_ranks_kernel grid (48, 1, 1) block (256, 1, 1)

1 launches, 384 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 54.27 us, warp 44.40 us, nsys graphs 43.33 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 39.95 | 90.0% | 53.18 | 98.5% |
| compute | 4.05 | 9.1% | 0.38 | 0.7% |
| unattributed | 0.35 | 0.8% | 0.35 | 0.7% |
| store | 0.05 | 0.1% | 0.06 | 0.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 39.95 | 90.0% | 1.0 | 53.38 | `if ((((int64_t)0 <= token_ids[((int64_t)((int)blockIdx.x))]) && (token_ids[((int64_t)((int` |
| compute.reduce | 4.05 | 9.1% | 1.0 | 16.80 | `total[0] = tl::AllReduce<tl::SumOp, 256, 1, 0, tl::NamedBarrier<256>>::run(total[0], (&(wo` |
| store.global | 0.05 | 0.1% | 1.0 | 0.06 | `output[((int64_t)((int)blockIdx.x))] = ((int64_t)total[0]);` |

### sink_cut_kernel grid (1, 1, 1) block (128, 1, 1)

34 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 1.86 us, warp 0.63 us, nsys graphs 2.08 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.34 | 54.4% | 0.47 | 30.9% |
| store | 0.17 | 26.6% | 0.54 | 36.1% |
| load | 0.12 | 19.0% | 0.50 | 33.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.17 | 26.6% | 0.8 | 0.54 | `if (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) < num_cut) \| saved[((((int64_t)((int)` |
| load.global | 0.12 | 19.0% | 0.1 | 0.48 | `for (int b = 0; b < num_reqs; ++b)` |

### ced_copy_rows_kernel grid (200, 1, 1) block (128, 1, 1)

17 launches, 800 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 1.28 us, warp 1.08 us, nsys graphs 1.79 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.77 | 71.0% | 1.80 | 77.6% |
| unattributed | 0.31 | 29.0% | 0.52 | 22.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.77 | 71.0% | 1.0 | 0.90 | `for (int k = 0; k < 4; ++k) \| int64_t index = ced_ldg64((&(row_indices[((((int64_t)blockId` |

### sink_policy_kernel grid (8, 1, 1) block (32, 1, 1)

17 launches, 8 warps; ranges, warp and last warp from the 4 recording CTAs' 4 warps; span 2.56 us, warp 2.55 us, nsys graphs 2.46 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.18 | 46.1% | 1.51 | 51.6% |
| unattributed | 0.87 | 34.2% | 0.89 | 30.2% |
| store | 0.50 | 19.7% | 0.53 | 18.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 | 0.78 | 30.7% | 1.0 | 0.80 | `state[0] = idx_mapping[((int64_t)((int)blockIdx.x))];` |
| store.global#3 | 0.45 | 17.6% | 1.0 | 0.45 | `for (int a = 0; a < 3; ++a) \| recovered_logprobs[((int64_t)((int)blockIdx.x))] = 0x0p+0f/*` |
| load.global | 0.21 | 8.2% | 1.0 | 0.22 | `start[0] = cu_num_logits[((int64_t)((int)blockIdx.x))];` |
| load.global#4 | 0.14 | 5.3% | 1.0 | 0.16 | `for (int r = 0; r < 2; ++r) \| for (int r_2 = 0; r_2 < 8; ++r_2)` |
| store.global#4 | 0.06 | 2.2% | 1.0 | 0.06 | `events[(((int64_t)((int)blockIdx.x)) * (int64_t)6)] = outcome[0];` |
| load.global#5 | 0.05 | 1.9% | 1.0 | 0.06 | `AtomicAdd((&(counts[0])), draft_zero_flag[0]);` |

### compute_slot_mappings_kernel grid (1, 11, 1) block (256, 1, 1)

17 launches, 88 warps; ranges, warp and last warp from the 6 recording CTAs' 48 warps; span 2.30 us, warp 1.43 us, nsys graphs 2.69 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.79 | 55.5% | 0.79 | 27.2% |
| unattributed | 0.64 | 44.5% | 2.13 | 72.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.79 | 55.5% | 1.0 | 0.80 | `int first_token = query_start_loc[0];` |

### cache_writer_kernel grid (16, 1, 1) block (128, 1, 1)

48 launches, 64 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 0.99 us, warp 0.98 us, nsys graphs 1.09 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.32 | 32.8% | 0.33 | 33.0% |
| load | 0.29 | 29.7% | 0.30 | 30.2% |
| store | 0.29 | 29.5% | 0.29 | 29.2% |
| compute | 0.08 | 8.0% | 0.08 | 7.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.29 | 29.5% | 1.0 | 0.29 | `if (((int64_t)0 <= slot) & ((slot >> (int64_t)7) < ((int64_t)pages))) \| if (((int64_t)0 <=` |
| load.global | 0.23 | 23.6% | 1.0 | 0.26 | `for (int i = 0; i < 4; ++i)` |
| compute.reduce | 0.08 | 8.0% | 1.0 | 0.10 | `amax = max(amax, __shfl_xor_sync((uint)4294967295, amax, 1, 32));` |
| load.global#2 | 0.06 | 6.1% | 1.0 | 0.06 | `int64_t slot = slots[((int64_t)((int)blockIdx.x))];` |

### verify_kernel grid (8, 1, 1) block (96, 1, 1)

17 launches, 24 warps; ranges, warp and last warp from the 4 recording CTAs' 12 warps; span 2.56 us, warp 2.49 us, nsys graphs 2.24 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.04 | 41.9% | 1.07 | 40.9% |
| unattributed | 0.90 | 36.3% | 0.87 | 33.3% |
| compute | 0.30 | 12.0% | 0.30 | 11.6% |
| wait | 0.15 | 6.1% | 0.21 | 7.9% |
| store | 0.09 | 3.6% | 0.17 | 6.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.98 | 39.3% | 1.0 | 0.99 | `start = cu_num_logits[((int64_t)((int)blockIdx.x))];` |
| compute.reduce | 0.30 | 12.0% | 1.0 | 0.32 | `row_max[0] = vllm_max(row_max[0], __shfl_xor_sync((uint)4294967295, row_max[0], 8, 32));` |
| store.global | 0.09 | 3.6% | 1.0 | 0.16 | `if (((int)threadIdx.x) == 0)` |
| load.global#2 | 0.06 | 2.6% | 1.0 | 0.06 | `target = local_argmax[((((int64_t)row) * argmax_stride) + ((int64_t)best_index[0]))];` |
| wait.sync#2 | 0.06 | 2.6% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#3 | 0.05 | 1.8% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync | 0.04 | 1.7% | 1.0 | 0.06 | `__syncthreads();` |

### metadata_pages_kernel grid (48, 1, 1) block (128, 1, 1)

34 launches, 192 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 1.28 us, warp 0.85 us, nsys graphs 1.50 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.68 | 79.5% | 0.75 | 59.7% |
| load | 0.14 | 16.0% | 0.19 | 15.1% |
| store | 0.04 | 4.5% | 0.31 | 25.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.13 | 15.1% | 1.0 | 0.13 | `if ((((offset + ((int)blockIdx.x)) < src_rows) && (0 <= (offset + ((int)blockIdx.x)))))` |
| store.global | 0.04 | 4.5% | 0.1 | 0.32 | `if (((int)threadIdx.x) < 32)` |
| load.global#2 | 0.01 | 0.9% | 0.1 | 0.06 | `if (((req < table_rows) && (((int)threadIdx.x) < table_cols)))` |

### post_update_kernel grid (1, 1, 1) block (32, 1, 1)

17 launches, 1 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 1.92 us, warp 1.92 us, nsys graphs 2.67 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.70 | 36.7% | 0.74 | 29.4% |
| store | 0.64 | 33.3% | 1.22 | 48.2% |
| unattributed | 0.58 | 30.0% | 0.56 | 22.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 | 0.51 | 26.7% | 1.0 | 0.51 | `int count = num_sampled[((((int64_t)((int)blockIdx.x)) * (int64_t)32) + ((int64_t)((int)th` |
| store.global#2 | 0.48 | 25.0% | 1.0 | 0.48 | `if (0 < count)` |
| store.global | 0.16 | 8.3% | 1.0 | 0.16 | `if (0 < count)` |
| load.global | 0.10 | 5.0% | 1.0 | 0.10 | `int64_t req = idx_mapping[((((int64_t)((int)blockIdx.x)) * (int64_t)32) + ((int64_t)((int)` |
| load.global#3 | 0.10 | 5.0% | 1.0 | 0.10 | `if ((0 < count))` |

### rope_kernel grid (1, 1, 1) block (128, 1, 1)

48 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 0.67 us, warp 0.67 us, nsys graphs 1.12 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.42 | 61.9% | 0.53 | 61.0% |
| load | 0.22 | 33.3% | 0.30 | 34.6% |
| store | 0.03 | 4.8% | 0.04 | 4.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 | 0.13 | 19.0% | 1.0 | 0.13 | `if (((position < ((int64_t)positions_max)) && ((int64_t)0 <= position)))` |
| load.global | 0.10 | 14.3% | 1.0 | 0.10 | `if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < tokens)) \| int64_t position ` |
| store.global | 0.03 | 4.8% | 1.0 | 0.03 | `*(uint4*)(x + ((((((int64_t)((int)blockIdx.x)) * (int64_t)12288) + ((((int64_t)((int)threa` |

### norm_forward_and_per_token_cast_kernel_kernel grid (16, 1, 1) block (128, 1, 1)

16 launches, 64 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 2.37 us, warp 2.30 us, nsys graphs 1.89 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 1.92 | 83.5% | 2.58 | 104.5% |
| load | 0.46 | 19.8% | 0.42 | 17.0% |
| compute | 0.13 | 5.7% | 0.13 | 5.1% |
| wait | 0.06 | 2.5% | 0.05 | 1.9% |
| unattributed | 0.00 | 0.0% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*5 (est.: sampled iterations x5) | 1.63 | 71.1% | 1.0 | 2.08 | `for (int j_1 = 0; j_1 < 2; ++j_1)` |
| load.global*5 (est.: sampled iterations x5) | 0.23 | 10.0% | 1.0 | 0.32 | `*(uint4*)(x_local + (vi * 8)) = *(uint4*)(x + ((((((int64_t)vi) * (int64_t)1024) + (((int6` |
| store.global*5 (est.: sampled iterations x5) | 0.23 | 10.0% | 1.0 | 0.32 | `for (int j = 0; j < 4; ++j)` |
| load.global#2*5 (est.: sampled iterations x5) | 0.23 | 9.8% | 1.0 | 0.32 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 1024) + (((int)threadIdx.x) * 8)` |
| compute.reduce | 0.13 | 5.7% | 1.0 | 0.16 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| wait.sync | 0.06 | 2.5% | 1.0 | 0.10 | `__syncthreads();` |
| store.global#2 | 0.04 | 1.8% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| store.global#3 | 0.01 | 0.6% | 0.2 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |

### rope_kernel grid (48, 1, 1) block (256, 1, 1)

34 launches, 384 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 1.10 us, warp 0.95 us, nsys graphs 1.22 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.65 | 68.3% | 0.81 | 72.5% |
| load | 0.26 | 27.4% | 0.26 | 23.4% |
| store | 0.04 | 4.3% | 0.05 | 4.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.26 | 27.4% | 1.0 | 0.29 | `if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < tokens)) \| int64_t position ` |
| store.global | 0.04 | 4.3% | 1.0 | 0.06 | `*(uint4*)(x + ((((((int64_t)((int)blockIdx.x)) * (int64_t)12288) + ((((int64_t)((int)threa` |

### per_token_cast_kernel_kernel grid (120, 1, 1) block (64, 1, 1)

19 launches, 240 warps; ranges, warp and last warp from the 4 recording CTAs' 8 warps; span 1.86 us, warp 1.65 us, nsys graphs 1.41 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.91 | 55.1% | 1.01 | 53.6% |
| unattributed | 0.35 | 21.1% | 0.25 | 13.5% |
| store | 0.30 | 18.0% | 0.29 | 15.5% |
| compute | 0.10 | 5.8% | 0.33 | 17.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.81 | 49.3% | 1.0 | 1.06 | `for (int i = 0; i < 2; ++i)` |
| store.global#2 | 0.11 | 6.8% | 1.0 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.10 | 5.8% | 1.0 | 0.13 | `amax_fragment[i_1] = tl::AllReduce<tl::MaxOp, 2, 1, 0, tl::NamedBarrier<64>>::run(amax_fra` |
| load.global#2*2 (est.: sampled iterations x2) | 0.10 | 5.8% | 1.0 | 0.13 | `amax_fragment[i_1] = cutlass::bfloat16_t(__hmax((bfloat16_t(((nv_bfloat162*)(&(amax_fragme` |
| store.global*2 (est.: sampled iterations x2) | 0.10 | 5.8% | 1.0 | 0.13 | `out_sf[((((((int64_t)((int)blockIdx.x)) % (int64_t)15) * (int64_t)32) + (((int64_t)((int)t` |
| store.global#3*2 (est.: sampled iterations x2) | 0.09 | 5.3% | 1.0 | 0.13 | `*(fp8_e4_16_t*)(out + ((((((((int64_t)((int)blockIdx.x)) / (int64_t)3) * (int64_t)6144) + ` |

### sink_flags_kernel grid (1, 1, 1) block (128, 1, 1)

17 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 1.95 us, warp 0.66 us, nsys graphs 2.88 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.32 | 48.8% | 1.35 | 67.0% |
| unattributed | 0.25 | 37.8% | 0.32 | 15.8% |
| store | 0.09 | 13.4% | 0.35 | 17.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.32 | 48.8% | 0.2 | 1.28 | `token = logits_indices[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)th` |
| store.global | 0.09 | 13.4% | 0.2 | 0.35 | `flags[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = (` |

### insert_kernel grid (8, 1, 1) block (32, 1, 1)

17 launches, 8 warps; ranges, warp and last warp from the 4 recording CTAs' 4 warps; span 1.76 us, warp 1.74 us, nsys graphs 1.63 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.76 | 43.8% | 0.88 | 45.3% |
| unattributed | 0.48 | 27.6% | 0.48 | 24.5% |
| store | 0.34 | 19.4% | 0.43 | 22.0% |
| compute | 0.16 | 9.2% | 0.16 | 8.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#3 | 0.58 | 33.2% | 1.0 | 0.61 | `int state = mapping[((int64_t)position)]; \| int64_t state = mapping[((int64_t)position)];` |
| store.global | 0.29 | 16.6% | 1.0 | 0.29 | `counts[((int64_t)((int)blockIdx.x))] = (count_local + 1); \| counts[((int64_t)((int)blockId` |
| load.global | 0.12 | 6.9% | 1.0 | 0.13 | `for (int k = 0; k < 4; ++k)` |
| compute.reduce#2 | 0.11 | 6.5% | 1.0 | 0.13 | `float value_1 = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| load.global#2 | 0.06 | 3.7% | 1.0 | 0.06 | `count_local = counts[((int64_t)((int)blockIdx.x))];` |
| compute.reduce | 0.05 | 2.8% | 1.0 | 0.06 | `int count = __shfl_sync((uint)4294967295, count_local, 0, 32);` |
| store.global#2 | 0.05 | 2.8% | 1.0 | 0.06 | `if (((int)threadIdx.x) == 0)` |

### gemm_splitk_reduce_kernel grid (4, 16, 1) block (128, 1, 1)

48 launches, 256 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 0.61 us, warp 0.53 us, nsys graphs 0.77 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.25 | 46.3% | 0.27 | 45.9% |
| load | 0.24 | 45.4% | 0.27 | 46.0% |
| store | 0.04 | 8.3% | 0.05 | 8.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.24 | 45.4% | 1.0 | 0.29 | `acc[0] = P[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x))` |
| store.global | 0.04 | 8.3% | 1.0 | 0.06 | `C[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x)) * (int64` |

### combine_sampled_and_draft_tokens_kernel grid (8, 1, 1) block (32, 1, 1)

17 launches, 8 warps; ranges, warp and last warp from the 4 recording CTAs' 4 warps; span 1.31 us, warp 1.19 us, nsys graphs 1.44 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.64 | 53.7% | 0.66 | 54.2% |
| unattributed | 0.46 | 38.3% | 0.47 | 38.1% |
| store | 0.10 | 8.1% | 0.09 | 7.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.64 | 53.7% | 1.0 | 0.67 | `if (((((int)blockIdx.x) * 32) + ((int)threadIdx.x)) < num_reqs) \| req = idx_mapping[((int6` |
| store.global | 0.10 | 8.1% | 1.0 | 0.10 | `if ((prefill < seq) && (1 < num_logits))` |

### assign_draft_budget_kernel grid (1, 1, 1) block (256, 1, 1)

16 launches, 8 warps; ranges, warp and last warp from the 1 recording CTAs' 8 warps; span 1.66 us, warp 1.66 us, nsys graphs 1.70 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed, with loops' unsampled iterations | 0.65 | 38.9% | 0.86 | 49.6% |
| wait | 0.62 | 37.5% | 0.31 | 18.2% |
| load | 0.20 | 12.0% | 0.37 | 21.2% |
| store | 0.19 | 11.5% | 0.19 | 11.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync | 0.31 | 18.8% | 1.0 | 0.35 | `__syncthreads();` |
| wait.sync#3 | 0.25 | 14.9% | 1.0 | 0.32 | `__syncthreads();` |
| load.global | 0.20 | 12.0% | 1.0 | 0.48 | `for (int chunk = 0; chunk < ((num_reqs + 255) >> 8); ++chunk)` |
| store.global | 0.19 | 11.5% | 1.0 | 0.19 | `for (int chunk_2 = 0; chunk_2 < ((num_reqs + 255) >> 8); ++chunk_2)` |
| wait.sync#2 (sampled iterations) | 0.06 | 3.8% | 1.0 | 0.06 | `__syncthreads();` |

### token_counts_kernel grid (8, 1, 1) block (128, 1, 1)

17 launches, 32 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 1.15 us, warp 0.48 us, nsys graphs 1.44 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.29 | 60.3% | 0.35 | 22.1% |
| store | 0.14 | 28.5% | 0.99 | 63.2% |
| load | 0.05 | 11.2% | 0.23 | 14.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.14 | 28.5% | 0.2 | 0.58 | `if (((int)threadIdx.x) == 0) \| sampled[((int64_t)((int)blockIdx.x))] = (((int)(prefill_len` |
| load.global | 0.05 | 11.2% | 0.2 | 0.22 | `int count = sampled[((int64_t)((int)blockIdx.x))];` |

### logits_rows_kernel grid (1, 1, 1) block (128, 1, 1)

17 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 1.18 us, warp 1.14 us, nsys graphs 2.11 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.51 | 45.1% | 0.67 | 50.1% |
| unattributed | 0.49 | 43.0% | 0.30 | 22.1% |
| store | 0.14 | 12.0% | 0.37 | 27.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.51 | 45.1% | 1.0 | 0.51 | `index = logits_indices[min(((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((in` |
| store.global | 0.14 | 12.0% | 1.0 | 0.38 | `if (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) < num_logits)` |

### prepare_pos_seq_lens_kernel grid (9, 1, 1) block (256, 1, 1)

17 launches, 72 warps; ranges, warp and last warp from the 5 recording CTAs' 40 warps; span 1.09 us, warp 0.95 us, nsys graphs 1.38 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.46 | 48.2% | 0.57 | 51.6% |
| unattributed | 0.38 | 39.6% | 0.40 | 36.2% |
| store | 0.12 | 12.2% | 0.13 | 12.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.46 | 48.2% | 0.8 | 0.58 | `int computed = num_computed_tokens[idx_mapping[((int64_t)((int)blockIdx.x))]];` |
| store.global#2 | 0.10 | 10.8% | 0.8 | 0.13 | `if (((int)threadIdx.x) == 0)` |
| store.global | 0.01 | 1.4% | 0.2 | 0.06 | `for (int it = 0; it < (((max_reqs + 255) - num_reqs) >> 8); ++it)` |

### flatten_sampled_kernel grid (1, 1, 1) block (128, 1, 1)

17 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 1.12 us, warp 0.82 us, nsys graphs 1.31 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.47 | 57.8% | 0.62 | 56.5% |
| unattributed | 0.32 | 39.2% | 0.39 | 35.5% |
| store | 0.02 | 2.9% | 0.09 | 8.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.42 | 51.0% | 1.0 | 0.42 | `for (int r = 0; r < num_reqs; ++r)` |
| load.global#2 | 0.06 | 6.9% | 0.2 | 0.22 | `int i = (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) - cu_num_logits[req]);` |
| store.global | 0.02 | 2.9% | 0.2 | 0.10 | `flat[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = co` |

### scatter_rows_gather_kernel grid (1, 1, 1) block (128, 1, 1)

17 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 0.90 us, warp 0.83 us, nsys graphs 1.68 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.50 | 60.6% | 0.38 | 42.0% |
| store | 0.23 | 27.9% | 0.41 | 46.0% |
| load | 0.10 | 11.5% | 0.11 | 12.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.23 | 27.9% | 1.0 | 0.42 | `if (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) < (num_reqs * width))` |
| load.global | 0.10 | 11.5% | 1.0 | 0.10 | `index = idx_mapping[min((((((int64_t)0 <= ((int64_t)width)) && ((int64_t)0 <= rmod_1)) \|\| ` |

### padding_rows_kernel grid (1, 1, 1) block (256, 1, 1)

33 launches, 8 warps; ranges, warp and last warp from the 1 recording CTAs' 8 warps; span 0.38 us, warp 0.36 us, nsys graphs 0.77 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.26 | 73.0% | 0.28 | 73.7% |
| store | 0.10 | 27.0% | 0.10 | 26.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.10 | 27.0% | 1.0 | 0.10 | `if (((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) < rows)` |

### ced_copy_rows_kernel grid (80, 1, 1) block (128, 1, 1)

16 launches, 320 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 0.70 us, warp 0.62 us, nsys graphs 1.34 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.37 | 59.4% | 0.43 | 60.9% |
| unattributed | 0.25 | 40.6% | 0.27 | 39.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.37 | 59.4% | 1.0 | 0.42 | `for (int k = 0; k < 4; ++k) \| int64_t index = ced_ldg64((&(row_indices[((((int64_t)blockId` |

### scatter_rows_kernel grid (1, 1, 1) block (128, 1, 1)

17 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 0.61 us, warp 0.58 us, nsys graphs 0.99 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.38 | 66.7% | 0.38 | 61.2% |
| load | 0.10 | 16.7% | 0.11 | 17.6% |
| store | 0.10 | 16.7% | 0.13 | 21.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.10 | 16.7% | 1.0 | 0.10 | `index = idx_mapping[min((((((int64_t)0 <= ((int64_t)width)) && ((int64_t)0 <= rmod_1)) \|\| ` |
| store.global | 0.10 | 16.7% | 1.0 | 0.13 | `if (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) < (num_reqs * width))` |

### ced_requests_kernel grid (1, 1, 1) block (128, 1, 1)

17 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 0.61 us, warp 0.54 us, nsys graphs 1.15 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.34 | 62.7% | 0.33 | 53.1% |
| load | 0.10 | 19.4% | 0.11 | 18.4% |
| store | 0.10 | 17.9% | 0.18 | 28.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.10 | 19.4% | 1.0 | 0.13 | `uint v_ = tl::load_global_32((&(starts[1]))); \| uint v_ = tl::load_global_32((&(starts[min` |
| store.global | 0.10 | 17.9% | 1.0 | 0.16 | `if (((int)threadIdx.x) < 8) \| if (((int)threadIdx.x) <= 8)` |

### context_inputs_kernel grid (5, 48, 1) block (128, 1, 1)

1 launches, 960 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 9.54 us, warp 8.22 us, nsys graphs 9.34 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 6.89 | 83.8% | 7.36 | 77.7% |
| unattributed | 0.92 | 11.2% | 2.05 | 21.6% |
| store | 0.42 | 5.1% | 0.06 | 0.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 6.89 | 83.8% | 1.0 | 8.19 | `vllm_dspark_context_b128((&(aux[(((((int64_t)((int)blockIdx.y)) * (int64_t)15360) + (((int` |
| store.global | 0.42 | 5.1% | 1.0 | 1.15 | `if (((((int)blockIdx.x) == 0) & (((int)threadIdx.x) == 0)) & (num_tokens <= ((int)blockIdx` |

### per_token_cast_kernel_kernel grid (360, 1, 1) block (64, 1, 1)

1 launches, 720 warps; ranges, warp and last warp from the 4 recording CTAs' 8 warps; span 3.42 us, warp 2.89 us, nsys graphs 2.45 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 2.08 | 72.1% | 3.14 | 94.2% |
| load | 0.79 | 27.2% | 1.02 | 30.8% |
| compute | 0.35 | 12.2% | 0.13 | 3.8% |
| unattributed | 0.00 | 0.0% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#3*2 (est.: sampled iterations x2) | 1.10 | 37.9% | 1.0 | 3.01 | `*(fp8_e4_16_t*)(out + ((((((((int64_t)((int)blockIdx.x)) / (int64_t)3) * (int64_t)6144) + ` |
| store.global*2 (est.: sampled iterations x2) | 0.82 | 28.5% | 1.0 | 2.88 | `out_sf[((((((int64_t)((int)blockIdx.x)) % (int64_t)15) * (int64_t)32) + (((int64_t)((int)t` |
| load.global | 0.69 | 23.9% | 1.0 | 0.90 | `for (int i = 0; i < 2; ++i)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.35 | 12.2% | 1.0 | 1.98 | `amax_fragment[i_1] = tl::AllReduce<tl::MaxOp, 2, 1, 0, tl::NamedBarrier<64>>::run(amax_fra` |
| store.global#2 | 0.16 | 5.7% | 1.0 | 0.32 | `for (int i_3 = 0; i_3 < 4; ++i_3)` |
| load.global#2*2 (est.: sampled iterations x2) | 0.10 | 3.3% | 1.0 | 0.13 | `amax_fragment[i_1] = cutlass::bfloat16_t(__hmax((bfloat16_t(((nv_bfloat162*)(&(amax_fragme` |

### gemm_splitk_reduce_kernel grid (4, 48, 1) block (128, 1, 1)

3 launches, 768 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 0.77 us, warp 0.71 us, nsys graphs 1.22 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.39 | 54.7% | 0.42 | 55.7% |
| unattributed | 0.28 | 39.1% | 0.28 | 37.1% |
| store | 0.04 | 6.2% | 0.05 | 7.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.39 | 54.7% | 1.0 | 0.42 | `acc[0] = P[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x))` |
| store.global | 0.04 | 6.2% | 1.0 | 0.06 | `C[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x)) * (int64` |

### ced_copy_rows_kernel grid (240, 1, 1) block (128, 1, 1)

1 launches, 960 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 0.99 us, warp 0.84 us, nsys graphs 1.44 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.52 | 62.1% | 0.61 | 61.3% |
| unattributed | 0.32 | 37.9% | 0.38 | 38.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.52 | 62.1% | 1.0 | 0.61 | `for (int k = 0; k < 4; ++k) \| int64_t index = ced_ldg64((&(row_indices[((((int64_t)blockId` |
=== timeline and tables: exit 0 in 37 s
