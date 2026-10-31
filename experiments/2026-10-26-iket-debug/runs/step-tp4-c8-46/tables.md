=== timeline and tables
trace /c/iket/step-tp4-c8-20261026T190623Z/trace/iket/pid_0x18f5/iket.decoded_results.json: 16143 launches, graphLaunches dict of 89

## Timeline: 7547 launches in 1166277.1 us; the GPU is busy 334258.3 us (29%) and idle 832018.8 us in 5868 gaps; gaps over 200 us split it into 835 segments

The 15 of 835 segments with the most busy time:

| segment | launches | wall us | busy us | busy | first kernel |
|---:|---:|---:|---:|---:|---|
| 105 | 263 | 6063.2 | 5150.7 | 85% | finish_argmax_kernel |
| 115 | 92 | 10523.8 | 6511.1 | 62% | mhc_project_streams_kernel |
| 117 | 274 | 8013.4 | 6319.6 | 79% | partial_argmax_kernel |
| 122 | 102 | 13069.1 | 7676.2 | 59% | mhc_project_streams_kernel |
| 244 | 5 | 5608.5 | 5022.9 | 90% | mhc_post_kernel |
| 312 | 235 | 5360.4 | 5172.1 | 96% | mhc_project_streams_kernel |
| 374 | 249 | 5524.0 | 4942.0 | 89% | metadata_builder_kernel |
| 486 | 238 | 5462.1 | 5094.4 | 93% | metadata_builder_kernel |
| 514 | 46 | 8030.9 | 5930.4 | 74% | mhc_finalize_kernel |
| 627 | 262 | 6855.6 | 4945.2 | 72% | partial_argmax_kernel |
| 673 | 259 | 6346.5 | 4938.8 | 78% | scatter_rows_gather_kernel |
| 714 | 255 | 6257.3 | 4939.3 | 79% | adaptive_offsets_kernel |
| 746 | 63 | 10197.6 | 5710.8 | 56% | mhc_project_streams_kernel |
| 756 | 260 | 6833.3 | 5011.8 | 73% | dspark_confidence_kernel |
| 823 | 92 | 12909.8 | 6620.7 | 51% | mhc_project_streams_kernel |

| idle us | after | before |
|---:|---|---|
| 127278.4 | resample_kernel | insert_kernel |
| 91666.7 | scatter_rows_kernel | padding_rows_kernel |
| 11450.2 | norm_forward_and_per_token_cast_kernel_kernel | sink_flags_kernel |
| 11344.6 | norm_forward_and_per_token_cast_kernel_kernel | sink_flags_kernel |
| 11217.6 | norm_forward_and_per_token_cast_kernel_kernel | sink_flags_kernel |
| 11077.8 | norm_forward_and_per_token_cast_kernel_kernel | sink_flags_kernel |
| 10950.0 | norm_forward_and_per_token_cast_kernel_kernel | sink_flags_kernel |
| 10909.4 | norm_forward_and_per_token_cast_kernel_kernel | sink_flags_kernel |
| 10901.3 | norm_forward_and_per_token_cast_kernel_kernel | sink_flags_kernel |
| 10887.5 | norm_forward_and_per_token_cast_kernel_kernel | sink_flags_kernel |

## Engine steps

| step | kind | launches | wall us | busy us | busy | most span |
|---:|---|---:|---:|---:|---:|---|
| 0 | prefill | 410 | 74559.4 | 27937.8 | 37% | vocab_packed_projection_kernel 9102, moe_grouped_fp8_fp4_gemm_kernel 5497, gemm_mxfp8_kernel 4351 |
| 1 | decode | 420 | 166179.1 | 18535.3 | 11% | vocab_packed_projection_kernel 9204, moe_grouped_fp8_fp4_gemm_kernel 2148, gemm_mxfp8_kernel 2137 |
| 2 | decode | 420 | 131602.9 | 17893.7 | 14% | vocab_packed_projection_kernel 8723, moe_grouped_fp8_fp4_gemm_kernel 2391, gemm_mxfp8_kernel 2168 |
| 3 | decode | 420 | 50030.8 | 17826.3 | 36% | vocab_packed_projection_kernel 8636, moe_grouped_fp8_fp4_gemm_kernel 2395, gemm_mxfp8_kernel 2161 |
| 4 | decode | 420 | 53139.0 | 17892.0 | 34% | vocab_packed_projection_kernel 8674, moe_grouped_fp8_fp4_gemm_kernel 2279, gemm_mxfp8_kernel 2106 |
| 5 | decode | 420 | 47570.6 | 18372.4 | 39% | vocab_packed_projection_kernel 9128, moe_grouped_fp8_fp4_gemm_kernel 2321, gemm_mxfp8_kernel 2094 |
| 6 | decode | 420 | 51025.1 | 17862.6 | 35% | vocab_packed_projection_kernel 8700, moe_grouped_fp8_fp4_gemm_kernel 2378, gemm_mxfp8_kernel 2194 |
| 7 | decode | 420 | 55041.6 | 18146.9 | 33% | vocab_packed_projection_kernel 8766, moe_grouped_fp8_fp4_gemm_kernel 2385, gemm_mxfp8_kernel 2257 |
| 8 | decode | 420 | 52431.6 | 17922.9 | 34% | vocab_packed_projection_kernel 8659, moe_grouped_fp8_fp4_gemm_kernel 2308, gemm_mxfp8_kernel 2185 |
| 9 | decode | 420 | 51307.8 | 17880.7 | 35% | vocab_packed_projection_kernel 8677, moe_grouped_fp8_fp4_gemm_kernel 2326, gemm_mxfp8_kernel 2168 |
| 10 | decode | 420 | 49295.9 | 18757.3 | 38% | vocab_packed_projection_kernel 9298, moe_grouped_fp8_fp4_gemm_kernel 2334, gemm_mxfp8_kernel 2150 |
| 11 | decode | 420 | 57226.8 | 17816.1 | 31% | vocab_packed_projection_kernel 8688, moe_grouped_fp8_fp4_gemm_kernel 2305, gemm_mxfp8_kernel 2116 |
| 12 | decode | 420 | 53945.8 | 17776.6 | 33% | vocab_packed_projection_kernel 8662, moe_grouped_fp8_fp4_gemm_kernel 2314, gemm_mxfp8_kernel 2110 |
| 13 | decode | 420 | 52463.2 | 17897.3 | 34% | vocab_packed_projection_kernel 8742, moe_grouped_fp8_fp4_gemm_kernel 2310, gemm_mxfp8_kernel 2120 |
| 14 | decode | 420 | 54357.5 | 17718.6 | 33% | vocab_packed_projection_kernel 8644, moe_grouped_fp8_fp4_gemm_kernel 2298, gemm_mxfp8_kernel 2112 |
| 15 | decode | 420 | 55239.3 | 17664.4 | 32% | vocab_packed_projection_kernel 8586, moe_grouped_fp8_fp4_gemm_kernel 2288, gemm_mxfp8_kernel 2117 |
| 16 | decode | 420 | 60466.7 | 18175.7 | 30% | vocab_packed_projection_kernel 8934, moe_grouped_fp8_fp4_gemm_kernel 2324, gemm_mxfp8_kernel 2127 |
| 17 | decode | 417 | 48235.4 | 18181.5 | 38% | vocab_packed_projection_kernel 9121, moe_grouped_fp8_fp4_gemm_kernel 2293, gemm_mxfp8_kernel 2085 |

## Overview over 17 decode steps: 7137 launches of 127 kernel shapes, spans summing to 330307.7 us (19429.9 a step)

| kernel | grid | launches a step | span us a step | share | span us a launch | last warp: wait | load | compute | store | unattributed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vocab_packed_projection_kernel | (1, 4040, 1) | 1.05882 | 4915.8 | 25.3% | 4587.18 | 27% | 25% | 25% | 0% | 23% |
| vocab_packed_projection_kernel | (1, 4040, 1) | 0.941176 | 3898.4 | 20.1% | 4102.11 | 52% | 30% | 10% | 0% | 8% |
| gemm_bf16_kernel | (2020, 1, 1) | 5 | 1560.6 | 8.0% | 313.73 | 90% | 0% | 4% | 1% | 5% |
| l2_prefetch_kernel | (2, 1, 1) | 17 | 1312.7 | 6.8% | 71.46 | 0% | 100% | 0% | 0% | 0% |
| moe_grouped_fp8_fp4_gemm_kernel | (40, 288, 1) | 6 | 1023.0 | 5.3% | 169.39 | 9% | 48% | 0% | 0% | 42% |
| moe_grouped_fp8_fp4_gemm_kernel | (9, 288, 1) | 6 | 803.7 | 4.1% | 134.02 | 11% | 13% | 0% | 0% | 75% |
| mhc_project_streams_kernel | (40, 3, 1) | 18 | 800.4 | 4.1% | 42.05 | 7% | 36% | 20% | 4% | 34% |
| gemm_mxfp8_kernel | (80, 1, 1) | 10 | 559.5 | 2.9% | 14.96 | 62% | 6% | 8% | 1% | 24% |
| gemm_mxfp8_kernel | (128, 1, 1) | 9 | 472.8 | 2.4% | 44.93 | 114% | 6% | 16% | 1% | 0% |
| gemm_mxfp8_kernel | (40, 1, 1) | 9 | 405.0 | 2.1% | 44.00 | 53% | 6% | 19% | 2% | 20% |
| gemm_fp8w_bf16_partials_kernel | (32, 4, 1) | 9 | 398.9 | 2.1% | 55.01 | 30% | 12% | 13% | 0% | 44% |
| gemm_mxfp8_kernel | (28, 1, 1) | 9 | 383.1 | 2.0% | 38.82 | 37% | 10% | 25% | 0% | 28% |
| sparse_mla_kernel | (1, 48, 1) | 4 | 288.4 | 1.5% | 61.33 | 2% | 7% | 2% | 0% | 88% |
| gemm_mxfp8_kernel | (18, 1, 1) | 9 | 267.1 | 1.4% | 25.73 | 34% | 14% | 37% | 1% | 14% |
| moe_grouped_fp8_fp4_gemm_kernel | (9, 120, 1) | 3 | 249.0 | 1.3% | 82.11 | 13% | 13% | 0% | 0% | 74% |
| moe_grouped_fp8_fp4_gemm_kernel | (40, 120, 1) | 3 | 241.8 | 1.2% | 79.33 | 14% | 21% | 0% | 0% | 64% |
| partial_argmax_kernel | (8, 64, 1) | 5 | 152.4 | 0.8% | 29.44 | 2% | 69% | 1% | 1% | 28% |
| gemm_bf16_splitk_kernel | (16, 5, 1) | 2 | 142.7 | 0.7% | 72.93 | 85% | 16% | 6% | 1% | 0% |
| gemm_bf16_partials_kernel | (6, 8, 1) | 6 | 108.6 | 0.6% | 17.70 | 28% | 0% | 40% | 3% | 29% |
| moe_swiglu_cast_kernel_kernel | (8, 288, 1) | 6 | 90.0 | 0.5% | 14.86 | 0% | 66% | 0% | 0% | 34% |

| wait | kernel | grid | last-warp us a step | a launch | first statement |
|---|---|---|---:|---:|---|
| wait.cp_async*118 | gemm_mxfp8_kernel | (80, 1, 1) | 147.3 | 14.73 | `tl::cp_async_wait<2>();` |
| wait.cp_async*15 | gemm_mxfp8_kernel | (40, 1, 1) | 144.7 | 16.08 | `tl::cp_async_wait<1>();` |
| wait.cp_async*4 | gemm_mxfp8_kernel | (128, 1, 1) | 115.1 | 12.79 | `tl::cp_async_wait<1>();` |
| wait.sync#2*118 | gemm_mxfp8_kernel | (80, 1, 1) | 93.5 | 9.35 | `__syncthreads();` |
| wait.cp_async*13 | gemm_bf16_splitk_kernel | (16, 5, 1) | 69.3 | 34.67 | `tl::cp_async_wait<2>();` |
| wait.cp_async*19 | gemm_mxfp8_kernel | (28, 1, 1) | 67.3 | 7.48 | `tl::cp_async_wait<1>();` |
| wait.sync#2*15 | gemm_mxfp8_kernel | (40, 1, 1) | 52.6 | 5.85 | `__syncthreads();` |
| wait.sync#2*19 | gemm_mxfp8_kernel | (28, 1, 1) | 52.0 | 5.78 | `__syncthreads();` |
| wait.cp_async*8 | gemm_mxfp8_kernel | (80, 1, 1) | 43.1 | 4.31 | `tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| wait.cp_async*19 | gemm_mxfp8_kernel | (18, 1, 1) | 41.4 | 4.60 | `tl::cp_async_wait<1>();` |
| wait.mbarrier#2 | gemm_bf16_kernel | (2020, 1, 1) | 31.9 | 6.38 | `mbarrier[0].wait(0);` |
| wait.sync#2*19 | gemm_mxfp8_kernel | (18, 1, 1) | 30.9 | 3.43 | `__syncthreads();` |
| wait.cp_async*9 | gemm_mxfp8_kernel | (64, 1, 1) | 29.8 | 14.89 | `tl::cp_async_wait<1>();` |
| wait.cp_async*14 | gemm_fp8w_bf16_partials_kernel | (32, 4, 1) | 29.2 | 3.24 | `tl::cp_async_wait<3>();` |
| wait.sync#2*14 | gemm_fp8w_bf16_partials_kernel | (32, 4, 1) | 22.5 | 2.50 | `__syncthreads();` |
| wait.sync#2*4 | gemm_mxfp8_kernel | (128, 1, 1) | 21.0 | 2.33 | `__syncthreads();` |
| wait.sync*13 | gemm_bf16_splitk_kernel | (16, 5, 1) | 19.8 | 9.90 | `__syncthreads();` |
| wait.sync#2*8 | gemm_mxfp8_kernel | (80, 1, 1) | 18.9 | 1.89 | `__syncthreads();` |
| wait.cp_async#2 | gemm_mxfp8_kernel | (128, 1, 1) | 16.2 | 1.80 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| wait.cp_async#3 | gemm_fp8w_bf16_partials_kernel | (32, 4, 1) | 14.0 | 1.55 | `tl::cp_async_wait<3>();` |

### vocab_packed_projection_kernel grid (1, 4040, 1) block (256, 1, 1)

18 launches, 32320 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 4587.18 us, warp 56.98 us, nsys graphs 4637.02 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 21.95 | 38.5% | 14.97 | 27.0% |
| compute | 13.70 | 24.0% | 13.90 | 25.1% |
| unattributed | 11.43 | 20.1% | 12.59 | 22.7% |
| load | 9.84 | 17.3% | 13.97 | 25.2% |
| store | 0.06 | 0.1% | 0.06 | 0.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*37 (est.: sampled iterations x37) | 12.76 | 22.4% | 1.0 | 14.21 | `{` |
| wait.sync#2*37 (est.: sampled iterations x37) | 6.83 | 12.0% | 1.0 | 22.50 | `__syncthreads();` |
| wait.sync#3*37 (est.: sampled iterations x37) | 5.79 | 10.2% | 1.0 | 9.47 | `__syncthreads();` |
| load.cp_async#2*37 (est.: sampled iterations x37) | 5.29 | 9.3% | 1.0 | 5.92 | `for (int i_4 = 0; i_4 < 4; ++i_4)` |
| load.global | 4.21 | 7.4% | 1.0 | 9.09 | `int start = row_offsets[(((int)blockIdx.y) * 16)]; \| int start = row_offsets[(((int)blockI` |
| wait.cp_async*37 (est.: sampled iterations x37) | 3.22 | 5.6% | 1.0 | 21.90 | `tl::cp_async_wait<6>();` |
| wait.sync*37 (est.: sampled iterations x37) | 2.61 | 4.6% | 1.0 | 3.55 | `__syncthreads();` |
| wait.cp_async#2*37 (est.: sampled iterations x37) | 2.20 | 3.9% | 1.0 | 2.96 | `tl::cp_async_wait<7>();` |
| load.cp_async | 0.35 | 0.6% | 1.0 | 0.45 | `for (int i = 0; i < 4; ++i) \| for (int i = 0; i < 8; ++i) \| for (int i_1 = 0; i_1 < 4; ++i` |
| compute.mma#3 | 0.32 | 0.6% | 1.0 | 0.32 | `{` |
| compute.mma#4 | 0.32 | 0.6% | 1.0 | 0.35 | `{` |
| compute.mma#2 | 0.31 | 0.5% | 1.0 | 0.34 | `{` |
| wait.sync#4 | 0.22 | 0.4% | 1.0 | 0.72 | `__syncthreads();` |
| wait.sync#6 | 0.17 | 0.3% | 1.0 | 0.50 | `__syncthreads();` |
| wait.sync#7 | 0.15 | 0.3% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#9 | 0.14 | 0.3% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#5 | 0.14 | 0.3% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#8 | 0.14 | 0.2% | 1.0 | 0.46 | `__syncthreads();` |
| wait.cp_async#7 | 0.07 | 0.1% | 1.0 | 0.46 | `tl::cp_async_wait<0>();` |
| wait.cp_async#3 | 0.07 | 0.1% | 1.0 | 0.70 | `tl::cp_async_wait<4>();` |
| wait.cp_async#5 | 0.06 | 0.1% | 1.0 | 0.46 | `tl::cp_async_wait<2>();` |
| store.global | 0.06 | 0.1% | 1.0 | 0.10 | `for (int i_13 = 0; i_13 < 2; ++i_13) \| for (int i_14 = 0; i_14 < 4; ++i_14) \| for (int i_9` |
| wait.cp_async#4 | 0.05 | 0.1% | 1.0 | 0.06 | `tl::cp_async_wait<5>();` |
| wait.cp_async#8 | 0.05 | 0.1% | 1.0 | 0.06 | `tl::cp_async_wait<1>();` |
| wait.cp_async#6 | 0.05 | 0.1% | 1.0 | 0.06 | `tl::cp_async_wait<3>();` |

### vocab_packed_projection_kernel grid (1, 4040, 1) block (128, 1, 1)

16 launches, 16160 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 4102.11 us, warp 52.59 us, nsys graphs 4114.58 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 30.53 | 58.1% | 25.55 | 52.0% |
| load | 11.79 | 22.4% | 14.65 | 29.8% |
| unattributed | 5.16 | 9.8% | 3.82 | 7.8% |
| compute | 5.06 | 9.6% | 5.10 | 10.4% |
| store | 0.05 | 0.1% | 0.05 | 0.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.cp_async*17 (est.: sampled iterations x17) | 14.72 | 28.0% | 1.0 | 47.60 | `tl::cp_async_wait<6>();` |
| wait.sync#2*17 (est.: sampled iterations x17) | 10.46 | 19.9% | 1.0 | 38.35 | `__syncthreads();` |
| load.global | 9.21 | 17.5% | 1.0 | 19.46 | `int start = row_offsets[(((int)blockIdx.y) * 16)]; \| int start = row_offsets[(((int)blockI` |
| compute.mma*17 (est.: sampled iterations x17) | 4.35 | 8.3% | 1.0 | 4.35 | `{` |
| load.cp_async#2*17 (est.: sampled iterations x17) | 2.29 | 4.4% | 1.0 | 2.72 | `for (int i_6 = 0; i_6 < 8; ++i_6) \| for (int i_9 = 0; i_9 < 4; ++i_9)` |
| wait.sync#3*17 (est.: sampled iterations x17) | 1.09 | 2.1% | 1.0 | 1.63 | `__syncthreads();` |
| wait.sync*17 (est.: sampled iterations x17) | 1.09 | 2.1% | 1.0 | 1.09 | `__syncthreads();` |
| wait.cp_async#2*17 (est.: sampled iterations x17) | 0.95 | 1.8% | 1.0 | 1.09 | `tl::cp_async_wait<7>();` |
| wait.cp_async#3 | 0.63 | 1.2% | 1.0 | 2.21 | `tl::cp_async_wait<4>();` |
| wait.sync#4 | 0.40 | 0.8% | 1.0 | 1.58 | `__syncthreads();` |
| wait.sync#6 | 0.33 | 0.6% | 1.0 | 1.06 | `__syncthreads();` |
| load.cp_async | 0.28 | 0.5% | 1.0 | 0.35 | `for (int i = 0; i < 4; ++i) \| for (int i = 0; i < 8; ++i) \| for (int i_1 = 0; i_1 < 4; ++i` |
| compute.mma#2 | 0.24 | 0.5% | 1.0 | 0.26 | `{` |
| wait.cp_async#5 | 0.23 | 0.4% | 1.0 | 1.31 | `tl::cp_async_wait<2>();` |
| compute.mma#3 | 0.23 | 0.4% | 1.0 | 0.26 | `{` |
| compute.mma#4 | 0.23 | 0.4% | 1.0 | 0.26 | `{` |
| wait.sync#8 | 0.21 | 0.4% | 1.0 | 0.72 | `__syncthreads();` |
| wait.cp_async#7 | 0.14 | 0.3% | 1.0 | 0.74 | `tl::cp_async_wait<0>();` |
| wait.sync#5 | 0.06 | 0.1% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#9 | 0.05 | 0.1% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#7 | 0.05 | 0.1% | 1.0 | 0.06 | `__syncthreads();` |
| store.global | 0.05 | 0.1% | 1.0 | 0.06 | `for (int i_13 = 0; i_13 < 2; ++i_13) \| for (int i_14 = 0; i_14 < 4; ++i_14) \| for (int i_9` |
| wait.cp_async#4 | 0.04 | 0.1% | 1.0 | 0.06 | `tl::cp_async_wait<5>();` |
| wait.cp_async#6 | 0.04 | 0.1% | 1.0 | 0.06 | `tl::cp_async_wait<3>();` |
| wait.cp_async#8 | 0.04 | 0.1% | 1.0 | 0.06 | `tl::cp_async_wait<1>();` |

### gemm_bf16_kernel grid (2020, 1, 1) block (256, 1, 1)

85 launches, 16160 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 313.73 us, warp 4.44 us, nsys graphs 287.07 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 3.83 | 86.2% | 6.47 | 90.5% |
| unattributed | 0.40 | 8.9% | 0.32 | 4.5% |
| compute | 0.14 | 3.2% | 0.29 | 4.0% |
| load | 0.04 | 0.8% | 0.00 | 0.0% |
| store | 0.04 | 0.8% | 0.07 | 0.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.mbarrier#2 | 3.72 | 83.9% | 0.5 | 9.82 | `mbarrier[0].wait(0);` |
| compute.mma | 0.14 | 3.2% | 0.5 | 0.29 | `{` |
| wait.sync | 0.08 | 1.7% | 1.0 | 0.10 | `__syncthreads();` |
| load.tma | 0.04 | 0.8% | 0.5 | 0.13 | `if (tl::tl_shuffle_elect<128>())` |
| store.global | 0.04 | 0.8% | 0.5 | 0.10 | `for (int i_3 = 0; i_3 < 4; ++i_3) \| for (int i_4 = 0; i_4 < 8; ++i_4) \| for (int i_6 = 0; ` |
| wait.mbarrier | 0.03 | 0.7% | 0.5 | 0.06 | `mbarrier[2].wait(1);` |

### l2_prefetch_kernel grid (2, 1, 1) block (128, 1, 1)

289 launches, 8 warps; ranges, warp and last warp from the 2 recording CTAs' 8 warps; span 71.46 us, warp 69.44 us, nsys graphs 68.06 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 69.08 | 99.5% | 76.87 | 99.6% |
| unattributed | 0.36 | 0.5% | 0.34 | 0.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 69.08 | 99.5% | 1.0 | 71.07 | `for (int s = 0; s < (nseg2 >> 1); ++s)` |

### moe_grouped_fp8_fp4_gemm_kernel grid (40, 288, 1) block (128, 1, 1)

102 launches, 46080 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 169.39 us, warp 3.05 us, nsys graphs 118.24 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.42 | 46.4% | 0.20 | 42.5% |
| load | 0.63 | 20.6% | 0.23 | 48.4% |
| wait | 0.51 | 16.6% | 0.04 | 9.2% |
| compute | 0.44 | 14.4% | 0.00 | 0.0% |
| store | 0.06 | 2.0% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*5 (est.: sampled iterations x5) | 0.44 | 14.4% | 0.2 | 1.76 | `if (ko < 4)` |
| load#2*5 (est.: sampled iterations x5) | 0.23 | 7.5% | 0.2 | 0.96 | `if (ko < 3)` |
| load.global | 0.22 | 7.1% | 1.0 | 0.35 | `int expert = TileExpert[((int64_t)((int)blockIdx.y))];` |
| load.global#3 | 0.11 | 3.7% | 0.2 | 0.45 | `if (((expert < 128) && (0 <= expert))) \| if (((expert < 384) && (0 <= expert)))` |
| wait.sync#5*5 (est.: sampled iterations x5) | 0.11 | 3.6% | 0.2 | 0.48 | `__syncthreads();` |
| wait.mbarrier*5 (est.: sampled iterations x5) | 0.08 | 2.6% | 0.2 | 0.32 | `loaded[(ko % 3)].wait((ko / 3));` |
| wait.sync#4*5 (est.: sampled iterations x5) | 0.08 | 2.6% | 0.2 | 0.32 | `__syncthreads();` |
| wait.sync#6*5 (est.: sampled iterations x5) | 0.08 | 2.6% | 0.2 | 0.32 | `__syncthreads();` |
| wait.cp_async*5 (est.: sampled iterations x5) | 0.07 | 2.3% | 0.2 | 0.32 | `tl::cp_async_wait<2>();` |
| store.global | 0.06 | 2.0% | 0.2 | 0.26 | `for (int i_2 = 0; i_2 < 16; ++i_2) \| for (int i_3 = 0; i_3 < 16; ++i_3) \| for (int i_4 = 0` |
| load | 0.05 | 1.5% | 0.2 | 0.26 | `for (int s = 0; s < 2; ++s)` |
| wait.sync | 0.04 | 1.4% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#2 | 0.04 | 1.2% | 0.2 | 0.19 | `__syncthreads();` |
| load.global#2 | 0.02 | 0.8% | 0.2 | 0.19 | `if (((int)threadIdx.x) < 32) \| if (((int)threadIdx.x) < 64)` |
| wait.sync#3 | 0.01 | 0.3% | 0.2 | 0.03 | `__syncthreads();` |

### moe_grouped_fp8_fp4_gemm_kernel grid (9, 288, 1) block (128, 1, 1)

102 launches, 10368 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 134.02 us, warp 21.56 us, nsys graphs 120.22 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 9.29 | 43.1% | 0.05 | 11.3% |
| unattributed | 7.10 | 32.9% | 0.32 | 75.5% |
| compute | 3.36 | 15.6% | 0.00 | 0.0% |
| load | 1.76 | 8.2% | 0.06 | 13.3% |
| store | 0.06 | 0.3% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.mbarrier*40 (est.: sampled iterations x40) | 5.92 | 27.5% | 0.2 | 23.68 | `loaded[(ko % 3)].wait(((ko % 6) / 3));` |
| compute.mma*40 (est.: sampled iterations x40) | 3.36 | 15.6% | 0.2 | 14.08 | `{` |
| load#2*40 (est.: sampled iterations x40) | 1.44 | 6.7% | 0.2 | 7.68 | `if (ko < 38)` |
| wait.sync#5*40 (est.: sampled iterations x40) | 1.36 | 6.3% | 0.2 | 6.40 | `__syncthreads();` |
| wait.cp_async*40 (est.: sampled iterations x40) | 0.64 | 3.0% | 0.2 | 2.56 | `tl::cp_async_wait<2>();` |
| wait.sync#4*40 (est.: sampled iterations x40) | 0.64 | 3.0% | 0.2 | 2.56 | `__syncthreads();` |
| wait.sync#6*40 (est.: sampled iterations x40) | 0.64 | 3.0% | 0.2 | 2.56 | `__syncthreads();` |
| load.global | 0.13 | 0.6% | 1.0 | 0.35 | `int expert = TileExpert[((int64_t)((int)blockIdx.y))];` |
| load.global#3 | 0.12 | 0.6% | 0.2 | 0.48 | `if (((expert < 128) && (0 <= expert))) \| if (((expert < 384) && (0 <= expert)))` |
| store.global | 0.06 | 0.3% | 0.2 | 0.26 | `for (int i_2 = 0; i_2 < 16; ++i_2) \| for (int i_3 = 0; i_3 < 16; ++i_3) \| for (int i_4 = 0` |
| load | 0.05 | 0.2% | 0.2 | 0.26 | `for (int s = 0; s < 2; ++s)` |
| wait.sync | 0.05 | 0.2% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#2 | 0.03 | 0.1% | 0.2 | 0.16 | `__syncthreads();` |
| load.global#2 | 0.02 | 0.1% | 0.2 | 0.19 | `if (((int)threadIdx.x) < 32) \| if (((int)threadIdx.x) < 64)` |
| wait.sync#3 | 0.01 | 0.0% | 0.2 | 0.06 | `__syncthreads();` |

### mhc_project_streams_kernel grid (40, 3, 1) block (128, 1, 1)

306 launches, 480 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 42.05 us, warp 16.31 us, nsys graphs 41.71 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 7.10 | 43.5% | 4.65 | 35.7% |
| unattributed | 4.51 | 27.7% | 4.41 | 33.9% |
| compute | 2.77 | 17.0% | 2.55 | 19.6% |
| wait | 1.44 | 8.9% | 0.87 | 6.7% |
| store | 0.48 | 3.0% | 0.54 | 4.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2*4 (est.: sampled iterations x4) | 6.90 | 42.3% | 1.0 | 14.98 | `for (int i_1 = 0; i_1 < 6; ++i_1)` |
| compute.mma*4 (est.: sampled iterations x4) | 2.64 | 16.2% | 1.0 | 2.69 | `{` |
| wait.sync*4 (est.: sampled iterations x4) | 1.26 | 7.7% | 1.0 | 4.86 | `__syncthreads();` |
| store.global*4 (est.: sampled iterations x4) | 0.26 | 1.6% | 1.0 | 0.26 | `if (((((int)blockIdx.y) * 16) + (((int)threadIdx.x) >> 3)) < M)` |
| load.global | 0.21 | 1.3% | 1.0 | 0.26 | `for (int k = 0; k < 4; ++k)` |
| wait.sync#2*4 (est.: sampled iterations x4) | 0.18 | 1.1% | 1.0 | 0.26 | `__syncthreads();` |
| store.global#4 | 0.10 | 0.6% | 1.0 | 0.13 | `if (((((int)threadIdx.x) % 8) == 0) && (((((int)blockIdx.y) * 16) + (((int)threadIdx.x) >>` |
| compute.reduce | 0.07 | 0.4% | 1.0 | 0.10 | `red[0] = (red[0] + __shfl_xor_sync((uint)4294967295, red[0], 4, 32));` |
| compute.reduce#2 | 0.07 | 0.4% | 1.0 | 0.10 | `red[0] = (red[0] + __shfl_xor_sync((uint)4294967295, red[0], 4, 32));` |
| store.global#2 | 0.06 | 0.4% | 1.0 | 0.06 | `if (((((int)blockIdx.y) * 16) + (((int)threadIdx.x) >> 3)) < M) \| if (((((int)threadIdx.x)` |
| store.global#3 | 0.06 | 0.4% | 1.0 | 0.06 | `if (((((int)threadIdx.x) % 8) == 0) && (((((int)blockIdx.y) * 16) + (((int)threadIdx.x) >>` |
| compute.mma | 0.00 | 0.0% | 0.0 | 0.00 | `{` |
| load.global#2 | 0.00 | 0.0% | 0.0 | 0.00 | `for (int i_1 = 0; i_1 < 6; ++i_1)` |
| store.global | 0.00 | 0.0% | 0.0 | 0.00 | `for (int n = 0; n < 4; ++n) \| if (((((int)blockIdx.y) * 16) + (((int)threadIdx.x) >> 3)) <` |
| wait.sync | 0.00 | 0.0% | 0.0 | 0.00 | `__syncthreads();` |

### gemm_mxfp8_kernel grid (80, 1, 1) block (128, 1, 1)

170 launches, 320 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 14.96 us, warp 10.28 us, nsys graphs 20.37 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 3.98 | 38.7% | 12.62 | 24.0% |
| wait | 3.44 | 33.5% | 32.71 | 62.1% |
| compute | 1.76 | 17.1% | 3.96 | 7.5% |
| load | 0.84 | 8.2% | 2.91 | 5.5% |
| store | 0.26 | 2.5% | 0.48 | 0.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*8 (est.: sampled iterations x8) | 1.57 | 15.3% | 1.0 | 1.79 | `{` |
| wait.sync#2*8 (est.: sampled iterations x8) | 1.15 | 11.2% | 1.0 | 3.20 | `__syncthreads();` |
| wait.cp_async*8 (est.: sampled iterations x8) | 0.90 | 8.7% | 1.0 | 3.33 | `tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| load.cp_async#2*8 (est.: sampled iterations x8) | 0.70 | 6.8% | 1.0 | 1.02 | `for (int i_3 = 0; i_3 < 2; ++i_3) \| for (int i_5 = 0; i_5 < 2; ++i_5) \| tl::cp_async_gs_co` |
| wait.cp_async#2 | 0.61 | 6.0% | 1.0 | 1.63 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| wait.sync*8 (est.: sampled iterations x8) | 0.43 | 4.2% | 1.0 | 0.51 | `__syncthreads();` |
| wait.sync#3 | 0.34 | 3.3% | 1.0 | 0.96 | `__syncthreads();` |
| store.global | 0.26 | 2.5% | 1.0 | 0.42 | `for (int i_10 = 0; i_10 < 16; ++i_10) \| for (int i_10 = 0; i_10 < 8; ++i_10) \| for (int i_` |
| compute.mma#2 | 0.19 | 1.9% | 1.0 | 0.26 | `{` |
| load.cp_async | 0.14 | 1.3% | 1.0 | 0.19 | `for (int i = 0; i < 2; ++i) \| for (int i_1 = 0; i_1 < 2; ++i_1) \| for (int i_1 = 0; i_1 < ` |
| compute.mma#3 | 0.00 | 0.0% | 0.0 | 0.00 | `{` |
| compute.mma*118 (est.: sampled iterations x118) | 0.00 | 0.0% | 0.0 | 0.00 | `{` |
| load.cp_async#2*118 (est.: sampled iterations x118) | 0.00 | 0.0% | 0.0 | 0.00 | `for (int i_5 = 0; i_5 < 2; ++i_5) \| for (int i_5 = 0; i_5 < 4; ++i_5) \| tl::cp_async_gs_co` |
| wait.cp_async#3 | 0.00 | 0.0% | 0.0 | 0.00 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>();` |
| wait.cp_async*118 (est.: sampled iterations x118) | 0.00 | 0.0% | 0.0 | 0.00 | `tl::cp_async_wait<2>();` |
| wait.sync#2*118 (est.: sampled iterations x118) | 0.00 | 0.0% | 0.0 | 0.00 | `__syncthreads();` |
| wait.sync#4 | 0.00 | 0.0% | 0.0 | 0.00 | `__syncthreads();` |
| wait.sync*118 (est.: sampled iterations x118) | 0.00 | 0.0% | 0.0 | 0.00 | `__syncthreads();` |

### gemm_mxfp8_kernel grid (128, 1, 1) block (128, 1, 1)

153 launches, 512 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 44.93 us, warp 16.75 us, nsys graphs 40.10 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 12.80 | 76.4% | 17.74 | 114.4% |
| compute | 2.56 | 15.3% | 2.55 | 16.5% |
| load | 0.77 | 4.6% | 0.87 | 5.6% |
| unattributed | 0.47 | 2.8% | 0.00 | 0.0% |
| store | 0.15 | 0.9% | 0.16 | 1.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.cp_async*4 (est.: sampled iterations x4) | 8.73 | 52.1% | 1.0 | 17.28 | `tl::cp_async_wait<1>();` |
| compute.mma*4 (est.: sampled iterations x4) | 2.05 | 12.2% | 1.0 | 2.05 | `{` |
| wait.sync#2*4 (est.: sampled iterations x4) | 1.94 | 11.6% | 1.0 | 6.40 | `__syncthreads();` |
| wait.cp_async#2 | 1.50 | 8.9% | 1.0 | 3.58 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| load.cp_async#2*4 (est.: sampled iterations x4) | 0.56 | 3.3% | 1.0 | 0.64 | `for (int i_3 = 0; i_3 < 8; ++i_3)` |
| compute.mma#2 | 0.52 | 3.1% | 1.0 | 0.54 | `{` |
| wait.sync#3 | 0.44 | 2.6% | 1.0 | 1.44 | `__syncthreads();` |
| load.cp_async | 0.21 | 1.2% | 1.0 | 0.26 | `for (int i = 0; i < 2; ++i) \| for (int i_1 = 0; i_1 < 2; ++i_1) \| for (int i_1 = 0; i_1 < ` |
| wait.sync*4 (est.: sampled iterations x4) | 0.19 | 1.1% | 1.0 | 0.26 | `__syncthreads();` |
| store.global | 0.15 | 0.9% | 1.0 | 0.19 | `for (int i_10 = 0; i_10 < 16; ++i_10) \| for (int i_10 = 0; i_10 < 8; ++i_10) \| for (int i_` |

### gemm_mxfp8_kernel grid (40, 1, 1) block (128, 1, 1)

153 launches, 160 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 44.00 us, warp 43.22 us, nsys graphs 36.80 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 18.19 | 42.1% | 23.73 | 53.0% |
| unattributed | 13.35 | 30.9% | 8.97 | 20.1% |
| compute | 8.63 | 20.0% | 8.58 | 19.2% |
| load | 2.43 | 5.6% | 2.73 | 6.1% |
| store | 0.62 | 1.4% | 0.75 | 1.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.cp_async*15 (est.: sampled iterations x15) | 11.37 | 26.3% | 1.0 | 28.80 | `tl::cp_async_wait<1>();` |
| compute.mma*15 (est.: sampled iterations x15) | 8.13 | 18.8% | 1.0 | 8.16 | `{` |
| wait.sync#2*15 (est.: sampled iterations x15) | 4.95 | 11.5% | 1.0 | 14.40 | `__syncthreads();` |
| load.cp_async#2*15 (est.: sampled iterations x15) | 2.25 | 5.2% | 1.0 | 2.40 | `for (int i_3 = 0; i_3 < 4; ++i_3) \| for (int i_3 = 0; i_3 < 6; ++i_3)` |
| wait.sync*15 (est.: sampled iterations x15) | 0.93 | 2.2% | 1.0 | 0.96 | `__syncthreads();` |
| wait.cp_async#2 | 0.67 | 1.5% | 1.0 | 1.82 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| store.global | 0.62 | 1.4% | 1.0 | 0.86 | `for (int i_10 = 0; i_10 < 16; ++i_10) \| for (int i_10 = 0; i_10 < 8; ++i_10) \| for (int i_` |
| compute.mma#2 | 0.50 | 1.1% | 1.0 | 0.51 | `{` |
| wait.sync#3 | 0.27 | 0.6% | 1.0 | 0.96 | `__syncthreads();` |
| load.cp_async | 0.18 | 0.4% | 1.0 | 0.22 | `for (int i = 0; i < 2; ++i) \| for (int i_1 = 0; i_1 < 2; ++i_1) \| for (int i_1 = 0; i_1 < ` |

### gemm_fp8w_bf16_partials_kernel grid (32, 4, 1) block (128, 1, 1)

153 launches, 512 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 55.01 us, warp 53.49 us, nsys graphs 44.83 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 32.31 | 60.4% | 19.35 | 44.2% |
| wait | 11.49 | 21.5% | 13.13 | 30.0% |
| compute | 5.61 | 10.5% | 5.85 | 13.4% |
| load | 3.92 | 7.3% | 5.26 | 12.0% |
| store | 0.16 | 0.3% | 0.16 | 0.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*14 (est.: sampled iterations x14) | 5.01 | 9.4% | 1.0 | 6.27 | `{` |
| wait.sync#2*14 (est.: sampled iterations x14) | 2.69 | 5.0% | 1.0 | 8.96 | `__syncthreads();` |
| wait.cp_async*14 (est.: sampled iterations x14) | 2.30 | 4.3% | 1.0 | 12.99 | `tl::cp_async_wait<3>();` |
| load.cp_async#3*14 (est.: sampled iterations x14) | 1.23 | 2.3% | 1.0 | 1.79 | `for (int i_10 = 0; i_10 < 4; ++i_10) \| for (int i_10 = 0; i_10 < 6; ++i_10) \| for (int i_1` |
| load.global*14 (est.: sampled iterations x14) | 1.06 | 2.0% | 1.0 | 1.79 | `if (((int)threadIdx.x) < 4)` |
| load.cp_async#2*14 (est.: sampled iterations x14) | 1.04 | 1.9% | 1.0 | 1.34 | `for (int i_4 = 0; i_4 < 2; ++i_4) \| for (int i_6 = 0; i_6 < 2; ++i_6)` |
| wait.cp_async#3 | 0.97 | 1.8% | 1.0 | 3.01 | `tl::cp_async_wait<3>();` |
| wait.sync#4*14 (est.: sampled iterations x14) | 0.95 | 1.8% | 1.0 | 1.34 | `__syncthreads();` |
| wait.sync#5*14 (est.: sampled iterations x14) | 0.95 | 1.8% | 1.0 | 1.34 | `__syncthreads();` |
| wait.sync*14 (est.: sampled iterations x14) | 0.95 | 1.8% | 1.0 | 1.34 | `__syncthreads();` |
| wait.sync#3*14 (est.: sampled iterations x14) | 0.92 | 1.7% | 1.0 | 1.34 | `__syncthreads();` |
| wait.cp_async#2*14 (est.: sampled iterations x14) | 0.81 | 1.5% | 1.0 | 0.90 | `tl::cp_async_wait<3>();` |
| load.cp_async | 0.47 | 0.9% | 1.0 | 1.09 | `for (int i_1 = 0; i_1 < 2; ++i_1)` |
| wait.sync#7 | 0.33 | 0.6% | 1.0 | 1.02 | `__syncthreads();` |
| compute.mma#2 | 0.31 | 0.6% | 1.0 | 0.38 | `{` |
| compute.mma#3 | 0.28 | 0.5% | 1.0 | 0.32 | `{` |
| wait.sync#10 | 0.17 | 0.3% | 1.0 | 0.64 | `__syncthreads();` |
| store.global | 0.16 | 0.3% | 1.0 | 0.19 | `for (int i_13 = 0; i_13 < 8; ++i_13) \| for (int i_19 = 0; i_19 < 16; ++i_19) \| for (int i_` |
| wait.cp_async#5 | 0.13 | 0.2% | 1.0 | 0.77 | `tl::cp_async_wait<1>();` |
| load.global#2 | 0.06 | 0.1% | 1.0 | 0.10 | `if (((int)threadIdx.x) < 4)` |
| load.global#3 | 0.06 | 0.1% | 1.0 | 0.16 | `if (((int)threadIdx.x) < 4)` |
| wait.cp_async#6 | 0.06 | 0.1% | 1.0 | 0.06 | `tl::cp_async_wait<0>();` |
| wait.sync#6 | 0.05 | 0.1% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#9 | 0.05 | 0.1% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#11 | 0.05 | 0.1% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#8 | 0.05 | 0.1% | 1.0 | 0.06 | `__syncthreads();` |
| wait.cp_async#4 | 0.04 | 0.1% | 1.0 | 0.06 | `tl::cp_async_wait<2>();` |

### gemm_mxfp8_kernel grid (28, 1, 1) block (128, 1, 1)

153 launches, 112 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 38.82 us, warp 36.91 us, nsys graphs 42.30 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 13.14 | 35.6% | 15.43 | 36.7% |
| compute | 10.69 | 29.0% | 10.63 | 25.3% |
| unattributed | 9.22 | 25.0% | 11.77 | 28.0% |
| load | 3.68 | 10.0% | 4.00 | 9.5% |
| store | 0.18 | 0.5% | 0.16 | 0.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*19 (est.: sampled iterations x19) | 10.18 | 27.6% | 1.0 | 10.34 | `{` |
| wait.cp_async*19 (est.: sampled iterations x19) | 6.23 | 16.9% | 1.0 | 21.28 | `tl::cp_async_wait<1>();` |
| wait.sync#2*19 (est.: sampled iterations x19) | 4.86 | 13.2% | 1.0 | 16.42 | `__syncthreads();` |
| load.cp_async#2*19 (est.: sampled iterations x19) | 3.46 | 9.4% | 1.0 | 3.65 | `for (int i_3 = 0; i_3 < 2; ++i_3) \| for (int i_3 = 0; i_3 < 8; ++i_3)` |
| wait.sync*19 (est.: sampled iterations x19) | 1.22 | 3.3% | 1.0 | 1.82 | `__syncthreads();` |
| wait.cp_async#2 | 0.54 | 1.5% | 1.0 | 1.50 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| compute.mma#2 | 0.50 | 1.4% | 1.0 | 0.51 | `{` |
| wait.sync#3 | 0.29 | 0.8% | 1.0 | 0.99 | `__syncthreads();` |
| load.cp_async | 0.23 | 0.6% | 1.0 | 0.26 | `for (int i = 0; i < 2; ++i) \| for (int i_1 = 0; i_1 < 2; ++i_1) \| for (int i_1 = 0; i_1 < ` |
| store.global | 0.18 | 0.5% | 1.0 | 0.22 | `for (int i_10 = 0; i_10 < 16; ++i_10) \| for (int i_10 = 0; i_10 < 8; ++i_10) \| for (int i_` |

### sparse_mla_kernel grid (1, 48, 1) block (256, 1, 1)

68 launches, 384 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 61.33 us, warp 61.11 us, nsys graphs 52.51 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed, with loops' unsampled iterations | 55.84 | 91.4% | 58.69 | 88.3% |
| load | 2.33 | 3.8% | 4.95 | 7.4% |
| compute | 1.49 | 2.4% | 1.51 | 2.3% |
| wait | 1.35 | 2.2% | 1.26 | 1.9% |
| store | 0.10 | 0.2% | 0.10 | 0.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global (sampled iterations) | 1.96 | 3.2% | 4.0 | 2.26 | `for (int c = 0; c < 4; ++c) \| for (int i = 0; i < 8; ++i) \| for (int i_1 = 0; i_1 < 8; ++i` |
| wait.sync (sampled iterations) | 1.28 | 2.1% | 18.0 | 1.63 | `__syncthreads();` |
| compute.mma (sampled iterations) | 1.22 | 2.0% | 2.0 | 1.25 | `for (int ki = 0; ki < 32; ++ki) \| for (int ki_1 = 0; ki_1 < 2; ++ki_1)` |
| load.cp_async (sampled iterations) | 0.38 | 0.6% | 2.0 | 0.45 | `for (int k_1 = 0; k_1 < 5; ++k_1) \| for (int k_1 = 0; k_1 < 9; ++k_1) \| for (int k_3 = 0; ` |
| compute.reduce (sampled iterations) | 0.27 | 0.4% | 2.0 | 0.32 | `m_g[i_5] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(m_g[i_5], (&((` |
| store.global | 0.10 | 0.2% | 1.0 | 0.13 | `for (int i_10 = 0; i_10 < 32; ++i_10) \| for (int i_11 = 0; i_11 < 32; ++i_11) \| for (int i` |
| wait.cp_async (sampled iterations) | 0.07 | 0.1% | 1.0 | 0.10 | `tl::cp_async_wait<0>();` |

### gemm_mxfp8_kernel grid (18, 1, 1) block (128, 1, 1)

153 launches, 72 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 25.73 us, warp 25.14 us, nsys graphs 25.38 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| compute | 10.62 | 42.2% | 10.65 | 37.4% |
| wait | 6.74 | 26.8% | 9.61 | 33.7% |
| unattributed | 3.87 | 15.4% | 4.00 | 14.0% |
| load | 3.72 | 14.8% | 4.09 | 14.3% |
| store | 0.19 | 0.7% | 0.17 | 0.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*19 (est.: sampled iterations x19) | 10.12 | 40.3% | 1.0 | 10.34 | `{` |
| load.cp_async#2*19 (est.: sampled iterations x19) | 3.50 | 13.9% | 1.0 | 3.65 | `for (int i_3 = 0; i_3 < 2; ++i_3) \| for (int i_3 = 0; i_3 < 8; ++i_3)` |
| wait.sync#2*19 (est.: sampled iterations x19) | 3.01 | 12.0% | 1.0 | 9.73 | `__syncthreads();` |
| wait.cp_async*19 (est.: sampled iterations x19) | 2.37 | 9.4% | 1.0 | 11.55 | `tl::cp_async_wait<1>();` |
| wait.sync*19 (est.: sampled iterations x19) | 1.22 | 4.8% | 1.0 | 1.82 | `__syncthreads();` |
| compute.mma#2 | 0.50 | 2.0% | 1.0 | 0.51 | `{` |
| load.cp_async | 0.23 | 0.9% | 1.0 | 0.26 | `for (int i = 0; i < 2; ++i) \| for (int i_1 = 0; i_1 < 2; ++i_1) \| for (int i_1 = 0; i_1 < ` |
| store.global | 0.19 | 0.7% | 1.0 | 0.22 | `for (int i_10 = 0; i_10 < 16; ++i_10) \| for (int i_10 = 0; i_10 < 8; ++i_10) \| for (int i_` |
| wait.cp_async#2 | 0.07 | 0.3% | 1.0 | 0.26 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| wait.sync#3 | 0.07 | 0.3% | 1.0 | 0.19 | `__syncthreads();` |

### moe_grouped_fp8_fp4_gemm_kernel grid (9, 120, 1) block (128, 1, 1)

51 launches, 4320 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 82.11 us, warp 13.51 us, nsys graphs 72.70 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 4.09 | 30.3% | 0.05 | 12.7% |
| unattributed | 3.96 | 29.3% | 0.30 | 73.9% |
| compute | 3.52 | 26.0% | 0.00 | 0.0% |
| load | 1.87 | 13.8% | 0.06 | 13.5% |
| store | 0.07 | 0.5% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*40 (est.: sampled iterations x40) | 3.52 | 26.0% | 0.2 | 14.08 | `{` |
| load#2*40 (est.: sampled iterations x40) | 1.52 | 11.2% | 0.2 | 8.96 | `if (ko < 38)` |
| wait.sync#5*40 (est.: sampled iterations x40) | 1.20 | 8.9% | 0.2 | 6.40 | `__syncthreads();` |
| wait.mbarrier*40 (est.: sampled iterations x40) | 0.96 | 7.1% | 0.2 | 3.84 | `loaded[(ko % 3)].wait(((ko % 6) / 3));` |
| wait.sync#4*40 (est.: sampled iterations x40) | 0.64 | 4.7% | 0.2 | 2.56 | `__syncthreads();` |
| wait.sync#6*40 (est.: sampled iterations x40) | 0.64 | 4.7% | 0.2 | 2.56 | `__syncthreads();` |
| wait.cp_async*40 (est.: sampled iterations x40) | 0.56 | 4.1% | 0.2 | 2.56 | `tl::cp_async_wait<2>();` |
| load.global#3 | 0.21 | 1.5% | 0.2 | 0.83 | `if (((expert < 128) && (0 <= expert))) \| if (((expert < 384) && (0 <= expert)))` |
| store.global | 0.07 | 0.5% | 0.2 | 0.32 | `for (int i_2 = 0; i_2 < 16; ++i_2) \| for (int i_3 = 0; i_3 < 16; ++i_3) \| for (int i_4 = 0` |
| load.global | 0.07 | 0.5% | 1.0 | 0.10 | `int expert = TileExpert[((int64_t)((int)blockIdx.y))];` |
| wait.sync | 0.05 | 0.4% | 1.0 | 0.06 | `__syncthreads();` |
| load | 0.05 | 0.3% | 0.2 | 0.26 | `for (int s = 0; s < 2; ++s)` |
| wait.sync#2 | 0.03 | 0.2% | 0.2 | 0.16 | `__syncthreads();` |
| load.global#2 | 0.02 | 0.2% | 0.2 | 0.19 | `if (((int)threadIdx.x) < 32) \| if (((int)threadIdx.x) < 64)` |
| wait.sync#3 | 0.01 | 0.1% | 0.2 | 0.06 | `__syncthreads();` |

### moe_grouped_fp8_fp4_gemm_kernel grid (40, 120, 1) block (128, 1, 1)

51 launches, 19200 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 79.33 us, warp 2.76 us, nsys graphs 58.27 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.28 | 46.5% | 0.19 | 64.2% |
| wait | 0.50 | 18.2% | 0.04 | 14.4% |
| load | 0.45 | 16.1% | 0.06 | 21.3% |
| compute | 0.44 | 15.9% | 0.00 | 0.0% |
| store | 0.09 | 3.2% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*5 (est.: sampled iterations x5) | 0.44 | 15.9% | 0.2 | 1.76 | `if (ko < 4)` |
| load#2*5 (est.: sampled iterations x5) | 0.22 | 8.0% | 0.2 | 0.96 | `if (ko < 3)` |
| wait.sync#5*5 (est.: sampled iterations x5) | 0.10 | 3.6% | 0.2 | 0.48 | `__syncthreads();` |
| load.global | 0.10 | 3.5% | 1.0 | 0.19 | `int expert = TileExpert[((int64_t)((int)blockIdx.y))];` |
| store.global | 0.09 | 3.2% | 0.2 | 0.38 | `for (int i_2 = 0; i_2 < 16; ++i_2) \| for (int i_3 = 0; i_3 < 16; ++i_3) \| for (int i_4 = 0` |
| wait.mbarrier*5 (est.: sampled iterations x5) | 0.08 | 2.9% | 0.2 | 0.32 | `loaded[(ko % 3)].wait((ko / 3));` |
| wait.sync#4*5 (est.: sampled iterations x5) | 0.08 | 2.9% | 0.2 | 0.32 | `__syncthreads();` |
| wait.sync#6*5 (est.: sampled iterations x5) | 0.08 | 2.9% | 0.2 | 0.32 | `__syncthreads();` |
| wait.cp_async*5 (est.: sampled iterations x5) | 0.07 | 2.5% | 0.2 | 0.32 | `tl::cp_async_wait<2>();` |
| load.global#3 | 0.06 | 2.0% | 0.2 | 0.22 | `if (((expert < 128) && (0 <= expert))) \| if (((expert < 384) && (0 <= expert)))` |
| load | 0.05 | 1.7% | 0.2 | 0.26 | `for (int s = 0; s < 2; ++s)` |
| wait.sync | 0.05 | 1.7% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#2 | 0.04 | 1.4% | 0.2 | 0.19 | `__syncthreads();` |
| load.global#2 | 0.03 | 0.9% | 0.2 | 0.22 | `if (((int)threadIdx.x) < 32) \| if (((int)threadIdx.x) < 64)` |
| wait.sync#3 | 0.01 | 0.3% | 0.2 | 0.03 | `__syncthreads();` |

### partial_argmax_kernel grid (8, 64, 1) block (128, 1, 1)

85 launches, 2048 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 29.44 us, warp 22.31 us, nsys graphs 27.26 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 10.70 | 48.0% | 7.87 | 27.6% |
| load | 8.14 | 36.5% | 19.51 | 68.5% |
| compute | 1.97 | 8.8% | 0.35 | 1.2% |
| wait | 1.05 | 4.7% | 0.59 | 2.1% |
| store | 0.44 | 2.0% | 0.15 | 0.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 8.14 | 36.5% | 1.0 | 19.97 | `for (int j = 0; j < 16; ++j) \| for (int run = 0; run < 2; ++run)` |
| compute.reduce | 1.97 | 8.8% | 1.0 | 5.06 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| wait.sync | 1.05 | 4.7% | 1.0 | 3.26 | `__syncthreads();` |
| store.global | 0.44 | 2.0% | 1.0 | 1.66 | `if (((int)threadIdx.x) == 0)` |

### gemm_bf16_splitk_kernel grid (16, 5, 1) block (256, 1, 1)

34 launches, 640 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 72.93 us, warp 64.33 us, nsys graphs 76.77 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 33.69 | 52.4% | 58.23 | 84.6% |
| unattributed | 23.57 | 36.6% | 0.00 | 0.0% |
| compute | 4.08 | 6.3% | 4.37 | 6.3% |
| load | 2.81 | 4.4% | 10.71 | 15.6% |
| store | 0.19 | 0.3% | 0.79 | 1.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.cp_async*13 (est.: sampled iterations x13) | 20.24 | 31.5% | 0.5 | 36.82 | `tl::cp_async_wait<2>();` |
| wait.sync*13 (est.: sampled iterations x13) | 5.03 | 7.8% | 0.5 | 14.77 | `__syncthreads();` |
| compute.mma*13 (est.: sampled iterations x13) | 2.00 | 3.1% | 0.5 | 2.08 | `{` |
| wait.sync*14 (est.: sampled iterations x14) | 1.80 | 2.8% | 0.5 | 6.94 | `__syncthreads();` |
| compute.mma*14 (est.: sampled iterations x14) | 1.50 | 2.3% | 0.5 | 1.79 | `{` |
| load.cp_async | 1.19 | 1.9% | 1.0 | 4.37 | `for (int i_1 = 0; i_1 < 2; ++i_1) \| tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)A_sha` |
| wait.sync#3 | 1.01 | 1.6% | 1.0 | 2.80 | `__syncthreads();` |
| wait.cp_async#2 | 0.83 | 1.3% | 1.0 | 3.66 | `tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>(); \| tl::cp_async_wait<3>();` |
| wait.sync#4 | 0.79 | 1.2% | 1.0 | 2.34 | `__syncthreads();` |
| wait.sync#5 | 0.75 | 1.2% | 1.0 | 2.45 | `__syncthreads();` |
| load.cp_async#2*13 (est.: sampled iterations x13) | 0.74 | 1.2% | 0.5 | 0.83 | `for (int i_10 = 0; i_10 < 2; ++i_10) \| for (int i_8 = 0; i_8 < 2; ++i_8) \| tl::cp_async_gs` |
| load.cp_async#2*14 (est.: sampled iterations x14) | 0.68 | 1.1% | 0.5 | 0.90 | `for (int i_6 = 0; i_6 < 2; ++i_6) \| tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)A_sha` |
| wait.cp_async*14 (est.: sampled iterations x14) | 0.68 | 1.1% | 0.5 | 6.72 | `tl::cp_async_wait<1>();` |
| wait.cp_async#3 | 0.65 | 1.0% | 1.0 | 2.74 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| wait.sync#2*14 (est.: sampled iterations x14) | 0.48 | 0.8% | 0.5 | 0.67 | `__syncthreads();` |
| wait.cp_async#4 | 0.43 | 0.7% | 0.5 | 1.68 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>();` |
| wait.sync#2*13 (est.: sampled iterations x13) | 0.40 | 0.6% | 0.5 | 0.42 | `__syncthreads();` |
| wait.sync#6 | 0.39 | 0.6% | 1.0 | 1.58 | `__syncthreads();` |
| compute.mma#2 | 0.24 | 0.4% | 1.0 | 0.29 | `{` |
| compute.mma#3 | 0.22 | 0.3% | 1.0 | 0.26 | `{` |
| wait.sync#7 | 0.20 | 0.3% | 0.5 | 0.26 | `__syncthreads();` |
| load.global | 0.19 | 0.3% | 1.0 | 2.08 | `if (((int)threadIdx.x) == 0)` |
| compute.mma#4 | 0.13 | 0.2% | 0.5 | 0.14 | `{` |
| store.global | 0.12 | 0.2% | 1.0 | 0.19 | `for (int i_10 = 0; i_10 < 4; ++i_10) \| for (int i_16 = 0; i_16 < 4; ++i_16) \| for (int i_1` |
| store.global#3 | 0.07 | 0.1% | 1.0 | 0.10 | `if (((int*)last)[0] == 1)` |
| load.global#2*16 (est.: sampled iterations x16) | 0.00 | 0.0% | 0.0 | 0.00 | `acc = P[(((((((int64_t)((int)blockIdx.z)) * (int64_t)65536) + (((int64_t)i_26) * (int64_t)` |
| load.global#2*8 (est.: sampled iterations x8) | 0.00 | 0.0% | 0.0 | 0.00 | `acc = P[(((((((int64_t)((int)blockIdx.z)) * (int64_t)16384) + (((int64_t)i_17) * (int64_t)` |
| store.global#2*16 (est.: sampled iterations x16) | 0.00 | 0.0% | 0.0 | 0.00 | `if (((int)blockIdx.x) < 8)` |
| store.global#2*8 (est.: sampled iterations x8) | 0.00 | 0.0% | 0.0 | 0.00 | `C0[(((((((int64_t)((int)blockIdx.z)) * (int64_t)16384) + (((int64_t)i_18) * (int64_t)2048)` |

### gemm_bf16_partials_kernel grid (6, 8, 1) block (256, 1, 1)

102 launches, 384 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 17.70 us, warp 8.53 us, nsys graphs 18.85 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 4.05 | 47.5% | 2.18 | 28.9% |
| wait | 2.47 | 28.9% | 2.15 | 28.4% |
| compute | 1.50 | 17.6% | 3.00 | 39.7% |
| load | 0.39 | 4.6% | 0.00 | 0.0% |
| store | 0.12 | 1.4% | 0.23 | 3.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*10 (est.: sampled iterations x10) | 1.50 | 17.6% | 0.5 | 3.20 | `{` |
| wait.mbarrier*10 (est.: sampled iterations x10) | 1.48 | 17.3% | 0.5 | 3.20 | `mbarrier[((kb % 3) + 3)].wait((((kb % 6) / 3) ^ 1));` |
| wait.mbarrier#2*10 (est.: sampled iterations x10) | 0.87 | 10.2% | 0.5 | 4.16 | `mbarrier[(kb_1 % 3)].wait(((kb_1 % 6) / 3));` |
| load.tma*10 (est.: sampled iterations x10) | 0.39 | 4.6% | 0.5 | 1.28 | `if (tl::tl_shuffle_elect<128>())` |
| store.global | 0.12 | 1.4% | 0.5 | 0.29 | `for (int i_1 = 0; i_1 < 4; ++i_1) \| for (int i_2 = 0; i_2 < 8; ++i_2) \| for (int i_4 = 0; ` |
| wait.sync | 0.12 | 1.4% | 1.0 | 0.16 | `__syncthreads();` |

### moe_swiglu_cast_kernel_kernel grid (8, 288, 1) block (288, 1, 1)

102 launches, 20736 warps; ranges, warp and last warp from the 4 recording CTAs' 36 warps; span 14.86 us, warp 1.46 us, nsys graphs 6.11 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.78 | 53.6% | 0.32 | 34.2% |
| load | 0.48 | 32.6% | 0.61 | 65.8% |
| compute | 0.12 | 8.5% | 0.00 | 0.0% |
| store | 0.08 | 5.3% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.48 | 32.6% | 1.0 | 0.98 | `for (int r = 0; r < 4; ++r)` |
| compute.reduce*4 (est.: sampled iterations x4) | 0.12 | 8.5% | 0.2 | 0.51 | `if (0 <= orders[r_2])` |
| store.global*4 (est.: sampled iterations x4) | 0.08 | 5.3% | 0.2 | 0.38 | `if (0 <= orders[r_2])` |

### mhc_finalize_kernel grid (48, 1, 1) block (128, 1, 1)

204 launches, 192 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 6.46 us, warp 5.37 us, nsys graphs 8.50 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.76 | 32.7% | 2.87 | 42.3% |
| compute | 1.00 | 18.5% | 0.62 | 9.1% |
| store | 0.99 | 18.5% | 1.23 | 18.2% |
| load | 0.98 | 18.2% | 1.14 | 16.8% |
| wait | 0.65 | 12.0% | 0.92 | 13.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync | 0.65 | 12.0% | 1.0 | 0.86 | `__syncthreads();` |
| compute.reduce | 0.58 | 10.7% | 0.2 | 2.30 | `row_max[0] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<32>>::run(row_max[0]);` |
| store.global#4*7 (est.: sampled iterations x7) | 0.56 | 10.4% | 0.8 | 0.90 | `if ((((int)threadIdx.x) % 4) == 0)` |
| compute.reduce#2*7 (est.: sampled iterations x7) | 0.42 | 7.8% | 0.8 | 0.67 | `amax[0] = max(amax[0], __shfl_xor_sync((uint)4294967295, amax[0], 1, 32));` |
| load.global#4*7 (est.: sampled iterations x7) | 0.38 | 7.0% | 0.8 | 0.67 | `*(uint4*)(yv + 0) = *(uint4*)(C + ((((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((i` |
| store.global#3*7 (est.: sampled iterations x7) | 0.32 | 6.0% | 0.8 | 0.45 | `*(uint4*)(y + ((((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((int64_t)step) * (int6` |
| load.global | 0.31 | 5.8% | 1.0 | 0.93 | `if (((int)threadIdx.x) < 32)` |
| load.global#3 | 0.24 | 4.5% | 0.8 | 0.35 | `for (int i_split_2 = 0; i_split_2 < 40; ++i_split_2)` |
| store.global | 0.10 | 1.9% | 0.2 | 0.42 | `if (((int)threadIdx.x) < 4)` |
| load.global#2 | 0.05 | 0.9% | 0.2 | 0.19 | `cm[0] = ((mixes_shared[((((int)threadIdx.x) & 15) + 8)] * scale[2]) + base[((((int)threadI` |
| store.global#2 | 0.01 | 0.2% | 0.2 | 0.06 | `if ((((int)threadIdx.x) >> 4) == 0)` |

### sparse_mla_kernel grid (2, 1, 40) block (256, 1, 1)

51 launches, 640 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 26.43 us, warp 14.58 us, nsys graphs 22.67 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| compute | 6.96 | 47.7% | 6.95 | 45.6% |
| wait | 2.99 | 20.5% | 2.81 | 18.5% |
| unattributed, with loops' unsampled iterations | 2.63 | 18.0% | 2.49 | 16.3% |
| load | 1.73 | 11.8% | 2.43 | 15.9% |
| store | 0.28 | 1.9% | 0.56 | 3.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*4 (est.: sampled iterations x4) | 2.73 | 18.7% | 1.0 | 2.82 | `for (int ki = 0; ki < 32; ++ki) \| {` |
| compute.mma#2*4 (est.: sampled iterations x4) | 2.18 | 15.0% | 1.0 | 2.30 | `for (int ki_1 = 0; ki_1 < 2; ++ki_1) \| if (((((int)blockIdx.x) * 2) + (c_1 >> 1)) < 3)` |
| compute.reduce#2*8 (est.: sampled iterations x8) | 1.02 | 7.0% | 1.0 | 1.02 | `row_sum[i_7] = tl::AllReduce<tl::SumOp, 128, 32, 0, tl::NamedBarrier<128>>::run(row_sum[i_` |
| compute.reduce*8 (est.: sampled iterations x8) | 1.02 | 7.0% | 1.0 | 1.02 | `m_g[i_4] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(m_g[i_4], (&((` |
| load.global#2 | 0.75 | 5.2% | 1.0 | 0.96 | `for (int i = 0; i < 8; ++i)` |
| load.cp_async#2*4 (est.: sampled iterations x4) | 0.64 | 4.4% | 1.0 | 0.77 | `if (((((int)blockIdx.x) * 2) + (c_1 >> 1)) < 3) \| if ((c_1 < 3) & (c_1 < 3)) \| if (c < 3)` |
| wait.sync#8*8 (est.: sampled iterations x8) | 0.39 | 2.7% | 1.0 | 0.51 | `__syncthreads();` |
| wait.sync#10*8 (est.: sampled iterations x8) | 0.38 | 2.6% | 1.0 | 0.51 | `__syncthreads();` |
| wait.sync#7*4 (est.: sampled iterations x4) | 0.30 | 2.0% | 1.0 | 0.38 | `__syncthreads();` |
| wait.sync#5*4 (est.: sampled iterations x4) | 0.28 | 1.9% | 1.0 | 0.38 | `__syncthreads();` |
| wait.cp_async*4 (est.: sampled iterations x4) | 0.28 | 1.9% | 1.0 | 0.38 | `tl::cp_async_wait<0>();` |
| wait.sync#4*4 (est.: sampled iterations x4) | 0.28 | 1.9% | 1.0 | 0.38 | `__syncthreads();` |
| store.global | 0.28 | 1.9% | 1.0 | 0.58 | `for (int i_10 = 0; i_10 < 32; ++i_10) \| for (int i_11 = 0; i_11 < 32; ++i_11) \| for (int i` |
| wait.sync#11*4 (est.: sampled iterations x4) | 0.26 | 1.8% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#6*4 (est.: sampled iterations x4) | 0.26 | 1.8% | 1.0 | 0.38 | `__syncthreads();` |
| wait.sync#9*4 (est.: sampled iterations x4) | 0.26 | 1.8% | 1.0 | 0.38 | `__syncthreads();` |
| load.cp_async (sampled iterations) | 0.23 | 1.6% | 1.0 | 0.26 | `for (int k_1 = 0; k_1 < 5; ++k_1) \| for (int k_1 = 0; k_1 < 9; ++k_1) \| for (int k_3 = 0; ` |
| wait.sync (sampled iterations) | 0.20 | 1.4% | 1.0 | 0.26 | `__syncthreads();` |
| load.global (sampled iterations) | 0.10 | 0.7% | 1.0 | 0.29 | `for (int c = 0; c < 4; ++c) \| for (int i = 0; i < 8; ++i) \| for (int i_1 = 0; i_1 < 8; ++i` |
| wait.sync#3 | 0.07 | 0.5% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#2 | 0.05 | 0.4% | 1.0 | 0.06 | `__syncthreads();` |

### sparse_mla_kernel grid (1, 1, 48) block (128, 1, 1)

34 launches, 192 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 28.72 us, warp 18.82 us, nsys graphs 26.26 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed, with loops' unsampled iterations | 6.36 | 33.8% | 5.93 | 29.3% |
| compute | 4.70 | 25.0% | 4.71 | 23.3% |
| load | 3.92 | 20.8% | 5.51 | 27.3% |
| wait | 2.68 | 14.3% | 3.23 | 16.0% |
| store | 1.15 | 6.1% | 0.85 | 4.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 | 2.70 | 14.3% | 1.0 | 3.78 | `for (int i = 0; i < 8; ++i)` |
| compute.mma*4 (est.: sampled iterations x4) | 1.74 | 9.2% | 1.0 | 1.79 | `for (int ki = 0; ki < 32; ++ki) \| {` |
| compute.mma#2*4 (est.: sampled iterations x4) | 1.17 | 6.2% | 1.0 | 1.28 | `for (int ki_1 = 0; ki_1 < 2; ++ki_1) \| if (((((int)blockIdx.x) * 2) + (c_1 >> 1)) < 3)` |
| store.global | 1.15 | 6.1% | 1.0 | 1.92 | `for (int i_10 = 0; i_10 < 32; ++i_10) \| for (int i_11 = 0; i_11 < 32; ++i_11) \| for (int i` |
| compute.reduce#2*8 (est.: sampled iterations x8) | 0.90 | 4.8% | 1.0 | 1.02 | `row_sum[i_7] = tl::AllReduce<tl::SumOp, 128, 32, 0, tl::NamedBarrier<128>>::run(row_sum[i_` |
| compute.reduce*8 (est.: sampled iterations x8) | 0.90 | 4.8% | 1.0 | 1.02 | `m_g[i_4] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(m_g[i_4], (&((` |
| load.cp_async#2*4 (est.: sampled iterations x4) | 0.66 | 3.5% | 1.0 | 0.77 | `if (((((int)blockIdx.x) * 2) + (c_1 >> 1)) < 3) \| if ((c_1 < 3) & (c_1 < 3)) \| if (c < 3)` |
| load.cp_async (sampled iterations) | 0.36 | 1.9% | 1.0 | 0.38 | `for (int k_1 = 0; k_1 < 5; ++k_1) \| for (int k_1 = 0; k_1 < 9; ++k_1) \| for (int k_3 = 0; ` |
| wait.sync#10*8 (est.: sampled iterations x8) | 0.33 | 1.7% | 1.0 | 0.51 | `__syncthreads();` |
| wait.sync#8*8 (est.: sampled iterations x8) | 0.32 | 1.7% | 1.0 | 0.51 | `__syncthreads();` |
| wait.cp_async*4 (est.: sampled iterations x4) | 0.28 | 1.5% | 1.0 | 0.38 | `tl::cp_async_wait<0>();` |
| wait.sync#5*4 (est.: sampled iterations x4) | 0.26 | 1.4% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#7*4 (est.: sampled iterations x4) | 0.26 | 1.4% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#9*4 (est.: sampled iterations x4) | 0.25 | 1.3% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#4*4 (est.: sampled iterations x4) | 0.25 | 1.3% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#11*4 (est.: sampled iterations x4) | 0.23 | 1.2% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#6*4 (est.: sampled iterations x4) | 0.22 | 1.2% | 1.0 | 0.26 | `__syncthreads();` |
| load.global (sampled iterations) | 0.21 | 1.1% | 1.0 | 0.30 | `for (int c = 0; c < 4; ++c) \| for (int i = 0; i < 8; ++i) \| for (int i_1 = 0; i_1 < 8; ++i` |
| wait.sync (sampled iterations) | 0.18 | 0.9% | 1.0 | 0.22 | `__syncthreads();` |
| wait.sync#3 | 0.06 | 0.3% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#2 | 0.05 | 0.2% | 1.0 | 0.06 | `__syncthreads();` |

### gemm_mxfp8_kernel grid (64, 1, 1) block (64, 1, 1)

34 launches, 128 warps; ranges, warp and last warp from the 4 recording CTAs' 8 warps; span 26.02 us, warp 24.60 us, nsys graphs 25.34 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 20.87 | 84.8% | 19.63 | 74.9% |
| compute | 5.10 | 20.7% | 5.10 | 19.4% |
| load | 1.26 | 5.1% | 1.34 | 5.1% |
| store | 0.19 | 0.8% | 0.21 | 0.8% |
| unattributed | 0.00 | 0.0% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.cp_async*9 (est.: sampled iterations x9) | 17.26 | 70.2% | 1.0 | 25.63 | `tl::cp_async_wait<1>();` |
| compute.mma*9 (est.: sampled iterations x9) | 4.61 | 18.7% | 1.0 | 4.61 | `{` |
| wait.sync#2*9 (est.: sampled iterations x9) | 2.14 | 8.7% | 1.0 | 8.06 | `__syncthreads();` |
| load.cp_async#2*9 (est.: sampled iterations x9) | 1.08 | 4.4% | 1.0 | 1.15 | `for (int i_3 = 0; i_3 < 8; ++i_3)` |
| wait.cp_async#2 | 0.83 | 3.4% | 1.0 | 1.98 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| wait.sync*9 (est.: sampled iterations x9) | 0.50 | 2.0% | 1.0 | 0.58 | `__syncthreads();` |
| compute.mma#2 | 0.49 | 2.0% | 1.0 | 0.51 | `{` |
| store.global | 0.19 | 0.8% | 1.0 | 0.22 | `for (int i_10 = 0; i_10 < 16; ++i_10) \| for (int i_10 = 0; i_10 < 8; ++i_10) \| for (int i_` |
| load.cp_async | 0.18 | 0.7% | 1.0 | 0.22 | `for (int i = 0; i < 2; ++i) \| for (int i_1 = 0; i_1 < 2; ++i_1) \| for (int i_1 = 0; i_1 < ` |
| wait.sync#3 | 0.13 | 0.5% | 1.0 | 0.46 | `__syncthreads();` |

### mhc_post_kernel grid (240, 1, 1) block (128, 1, 1)

68 launches, 960 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 12.11 us, warp 7.45 us, nsys graphs 9.70 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 7.20 | 96.6% | 8.73 | 83.5% |
| load | 0.17 | 2.3% | 0.86 | 8.2% |
| store | 0.08 | 1.0% | 0.87 | 8.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.17 | 2.3% | 1.0 | 0.22 | `*(float4*)(pm + 0) = *(float4*)(post + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int` |
| store.global | 0.08 | 1.0% | 1.0 | 0.45 | `for (int n_1 = 0; n_1 < 4; ++n_1)` |

### gemm_bf16_partials_kernel grid (2, 20, 1) block (256, 1, 1)

51 launches, 320 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 17.09 us, warp 7.69 us, nsys graphs 13.18 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 6.03 | 78.4% | 5.61 | 68.2% |
| wait | 0.88 | 11.4% | 1.29 | 15.7% |
| compute | 0.55 | 7.2% | 1.13 | 13.7% |
| load | 0.14 | 1.8% | 0.00 | 0.0% |
| store | 0.09 | 1.2% | 0.20 | 2.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.mbarrier#2*4 (est.: sampled iterations x4) | 0.65 | 8.4% | 0.5 | 2.94 | `mbarrier[(kb_1 % 3)].wait((kb_1 / 3));` |
| compute.mma*4 (est.: sampled iterations x4) | 0.55 | 7.2% | 0.5 | 1.15 | `{` |
| load.tma*4 (est.: sampled iterations x4) | 0.14 | 1.8% | 0.5 | 0.38 | `if (tl::tl_shuffle_elect<128>())` |
| wait.mbarrier*4 (est.: sampled iterations x4) | 0.12 | 1.6% | 0.5 | 0.26 | `mbarrier[((kb % 3) + 3)].wait(((kb / 3) ^ 1));` |
| wait.sync | 0.11 | 1.4% | 1.0 | 0.13 | `__syncthreads();` |
| store.global | 0.09 | 1.2% | 0.5 | 0.22 | `for (int i_1 = 0; i_1 < 4; ++i_1) \| for (int i_2 = 0; i_2 < 8; ++i_2) \| for (int i_4 = 0; ` |

### cache_writer_kernel grid (48, 1, 1) block (128, 1, 1)

139 launches, 192 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 3.97 us, warp 3.62 us, nsys graphs 5.95 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.80 | 49.7% | 2.23 | 42.8% |
| store | 1.45 | 40.0% | 2.11 | 40.4% |
| load | 0.30 | 8.2% | 0.72 | 13.7% |
| compute | 0.08 | 2.1% | 0.17 | 3.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 1.45 | 40.0% | 1.0 | 1.66 | `if (((int64_t)0 <= slot) & ((slot >> (int64_t)7) < ((int64_t)pages))) \| if (((int64_t)0 <=` |
| load.global | 0.24 | 6.5% | 1.0 | 0.26 | `for (int i = 0; i < 4; ++i)` |
| compute.reduce | 0.08 | 2.1% | 1.0 | 0.10 | `amax = max(amax, __shfl_xor_sync((uint)4294967295, amax, 1, 32));` |
| load.global#2 | 0.06 | 1.7% | 1.0 | 0.06 | `int64_t slot = slots[((int64_t)((int)blockIdx.x))];` |

### mhc_finalize_kernel grid (40, 1, 1) block (128, 1, 1)

102 launches, 160 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 6.66 us, warp 5.95 us, nsys graphs 6.22 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 2.30 | 38.7% | 3.49 | 47.9% |
| load | 1.01 | 17.0% | 0.93 | 12.8% |
| compute | 1.00 | 16.7% | 0.56 | 7.7% |
| store | 0.99 | 16.7% | 1.19 | 16.3% |
| wait | 0.65 | 10.9% | 1.12 | 15.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync | 0.65 | 10.9% | 1.0 | 0.86 | `__syncthreads();` |
| compute.reduce | 0.58 | 9.7% | 0.2 | 2.30 | `row_max[0] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<32>>::run(row_max[0]);` |
| store.global#4*7 (est.: sampled iterations x7) | 0.56 | 9.4% | 0.8 | 0.90 | `if ((((int)threadIdx.x) % 4) == 0)` |
| compute.reduce#2*7 (est.: sampled iterations x7) | 0.42 | 7.1% | 0.8 | 0.67 | `amax[0] = max(amax[0], __shfl_xor_sync((uint)4294967295, amax[0], 1, 32));` |
| load.global#4*7 (est.: sampled iterations x7) | 0.36 | 6.1% | 0.8 | 0.67 | `*(uint4*)(yv + 0) = *(uint4*)(C + ((((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((i` |
| store.global#3*7 (est.: sampled iterations x7) | 0.32 | 5.4% | 0.8 | 0.45 | `*(uint4*)(y + ((((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((int64_t)step) * (int6` |
| load.global | 0.31 | 5.2% | 1.0 | 0.93 | `if (((int)threadIdx.x) < 32)` |
| load.global#3 | 0.24 | 4.0% | 0.8 | 0.32 | `for (int i_split_2 = 0; i_split_2 < 40; ++i_split_2)` |
| store.global | 0.10 | 1.6% | 0.2 | 0.42 | `if (((int)threadIdx.x) < 4)` |
| load.global#2 | 0.09 | 1.6% | 0.2 | 0.38 | `cm[0] = ((mixes_shared[((((int)threadIdx.x) & 15) + 8)] * scale[2]) + base[((((int)threadI` |
| store.global#2 | 0.01 | 0.2% | 0.2 | 0.06 | `if ((((int)threadIdx.x) >> 4) == 0)` |

### moe_dispatch_kernel grid (1, 1, 1) block (512, 1, 1)

153 launches, 16 warps; ranges, warp and last warp from the 1 recording CTAs' 16 warps; span 4.22 us, warp 4.05 us, nsys graphs 2.53 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 2.17 | 53.6% | 1.77 | 41.4% |
| unattributed, with loops' unsampled iterations | 1.28 | 31.6% | 1.39 | 32.7% |
| store | 0.52 | 12.8% | 1.02 | 23.9% |
| load | 0.08 | 2.0% | 0.09 | 2.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync#7*9 (est.: sampled iterations x9) | 0.65 | 16.0% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#6*9 (est.: sampled iterations x9) | 0.63 | 15.5% | 1.0 | 0.86 | `__syncthreads();` |
| store.global (sampled iterations) | 0.41 | 10.1% | 1.0 | 0.70 | `ChunkCount[((((int64_t)c) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = count; \| Chu` |
| wait.sync#8 | 0.37 | 9.1% | 1.0 | 0.54 | `__syncthreads();` |
| wait.sync#3 | 0.14 | 3.4% | 1.0 | 0.35 | `__syncthreads();` |
| wait.sync#2 | 0.13 | 3.2% | 1.0 | 0.19 | `__syncthreads();` |
| store.global#2 | 0.11 | 2.7% | 1.0 | 0.26 | `RoutePos[((((int64_t)((int)blockIdx.x)) * (int64_t)512) + ((int64_t)((int)threadIdx.x)))] ` |
| load.global (sampled iterations) | 0.08 | 2.0% | 1.0 | 0.10 | `expert = ((int)TopkIdx[((((int64_t)((int)blockIdx.x)) * (int64_t)512) + ((int64_t)((int)th` |
| wait.sync | 0.07 | 1.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#9 | 0.06 | 1.6% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#4 | 0.06 | 1.5% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#5 | 0.06 | 1.5% | 1.0 | 0.06 | `__syncthreads();` |

### gemm_mxfp8_partials_kernel grid (16, 5, 1) block (128, 1, 1)

51 launches, 320 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 11.94 us, warp 11.41 us, nsys graphs 13.34 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 5.56 | 48.7% | 5.90 | 48.5% |
| wait | 4.73 | 41.4% | 3.91 | 32.1% |
| compute | 0.62 | 5.4% | 0.72 | 6.0% |
| load | 0.46 | 4.0% | 1.58 | 13.0% |
| store | 0.05 | 0.5% | 0.05 | 0.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.cp_async#3 | 1.65 | 14.4% | 1.0 | 3.58 | `tl::cp_async_wait<0>();` |
| wait.cp_async#2 | 0.99 | 8.7% | 1.0 | 2.14 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>();` |
| wait.sync#2*2 (est.: sampled iterations x2) | 0.76 | 6.7% | 1.0 | 2.75 | `__syncthreads();` |
| wait.sync#3 | 0.48 | 4.2% | 1.0 | 1.31 | `__syncthreads();` |
| wait.sync#4 | 0.40 | 3.5% | 1.0 | 1.28 | `__syncthreads();` |
| wait.cp_async*2 (est.: sampled iterations x2) | 0.35 | 3.1% | 1.0 | 2.62 | `tl::cp_async_wait<2>();` |
| compute.mma*2 (est.: sampled iterations x2) | 0.31 | 2.7% | 1.0 | 0.32 | `{` |
| load.cp_async | 0.23 | 2.0% | 1.0 | 0.32 | `for (int i = 0; i < 2; ++i) \| for (int i_1 = 0; i_1 < 4; ++i_1)` |
| load.cp_async#2*2 (est.: sampled iterations x2) | 0.22 | 2.0% | 1.0 | 0.26 | `for (int i_4 = 0; i_4 < 2; ++i_4)` |
| compute.mma#2 | 0.15 | 1.3% | 1.0 | 0.16 | `{` |
| compute.mma#3 | 0.15 | 1.3% | 1.0 | 0.19 | `{` |
| wait.sync*2 (est.: sampled iterations x2) | 0.09 | 0.8% | 1.0 | 0.13 | `__syncthreads();` |
| store.global | 0.05 | 0.5% | 1.0 | 0.06 | `for (int i_11 = 0; i_11 < 16; ++i_11) \| for (int i_13 = 0; i_13 < 8; ++i_13) \| for (int i_` |
| compute.mma*6 (est.: sampled iterations x6) | 0.00 | 0.0% | 0.0 | 0.00 | `{` |
| load.cp_async#2*6 (est.: sampled iterations x6) | 0.00 | 0.0% | 0.0 | 0.00 | `for (int i_5 = 0; i_5 < 4; ++i_5)` |
| wait.cp_async*6 (est.: sampled iterations x6) | 0.00 | 0.0% | 0.0 | 0.00 | `tl::cp_async_wait<2>();` |
| wait.sync#2*6 (est.: sampled iterations x6) | 0.00 | 0.0% | 0.0 | 0.00 | `__syncthreads();` |
| wait.sync*6 (est.: sampled iterations x6) | 0.00 | 0.0% | 0.0 | 0.00 | `__syncthreads();` |

### local_stats_kernel grid (16, 16, 1) block (128, 1, 1)

16 launches, 1024 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 34.56 us, warp 33.11 us, nsys graphs 36.02 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 31.95 | 96.5% | 32.98 | 96.3% |
| unattributed | 0.81 | 2.4% | 0.80 | 2.3% |
| compute | 0.19 | 0.6% | 0.19 | 0.5% |
| wait | 0.09 | 0.3% | 0.20 | 0.6% |
| store | 0.06 | 0.2% | 0.09 | 0.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 31.95 | 96.5% | 1.0 | 32.46 | `if (expanded_local_pos[((int64_t)((int)blockIdx.x))] < num_speculative_steps)` |
| compute.reduce | 0.13 | 0.4% | 1.0 | 0.16 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| wait.sync | 0.09 | 0.3% | 1.0 | 0.37 | `__syncthreads();` |
| store.global | 0.06 | 0.2% | 1.0 | 0.10 | `if (((int)threadIdx.x) == 0)` |
| compute.reduce#2 | 0.06 | 0.2% | 1.0 | 0.10 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 2, 32);` |

### metadata_decode_kernel grid (48, 2, 1) block (128, 1, 1)

68 launches, 384 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 2.32 us, warp 1.46 us, nsys graphs 3.54 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.69 | 47.2% | 1.49 | 20.4% |
| load | 0.69 | 47.1% | 4.87 | 66.4% |
| store | 0.08 | 5.7% | 0.97 | 13.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.45 | 30.6% | 0.5 | 1.34 | `if ((((offset + ((int)blockIdx.x)) < src_rows) && (0 <= (offset + ((int)blockIdx.x)))))` |
| load.global#3 | 0.23 | 16.0% | 0.5 | 0.51 | `if ((((offset + ((int)blockIdx.x)) < main_src_rows) && (0 <= (offset + ((int)blockIdx.x)))` |
| store.global#4 | 0.04 | 2.7% | 0.5 | 0.22 | `if (0 < live)` |
| store.global | 0.03 | 2.2% | 0.5 | 0.06 | `swa[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = con` |
| load.global#2 | 0.01 | 0.5% | 0.1 | 0.10 | `if (((0 <= req) & ((int64_t)0 <= pos)))` |
| store.global#2 | 0.01 | 0.4% | 0.1 | 0.06 | `lengths[((int64_t)((int)blockIdx.x))] = ((int)max(info[1], (int64_t)0));` |
| store.global#3 | 0.01 | 0.3% | 0.1 | 0.05 | `top_lengths[((int64_t)((int)blockIdx.x))] = condval_6; \| top_lengths[((int64_t)((int)block` |

### router_partials_kernel grid (24, 1, 1) block (64, 1, 1)

102 launches, 48 warps; ranges, warp and last warp from the 4 recording CTAs' 8 warps; span 5.02 us, warp 3.15 us, nsys graphs 4.64 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.28 | 40.6% | 1.98 | 37.2% |
| load | 1.05 | 33.3% | 1.79 | 33.6% |
| compute | 0.68 | 21.5% | 1.36 | 25.5% |
| store | 0.14 | 4.6% | 0.20 | 3.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.reduce | 0.68 | 21.5% | 0.5 | 1.38 | `for (int k = 0; k < 3; ++k) \| for (int k = 0; k < 6; ++k)` |
| load.global | 0.47 | 15.0% | 1.0 | 0.54 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < mask_len))` |
| load.global#4 | 0.45 | 14.3% | 0.5 | 0.96 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < image_len)) \| if ((((((int)bl` |
| load.global#2 | 0.06 | 2.0% | 1.0 | 0.06 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < mask_len))` |
| load.global#3 | 0.06 | 2.0% | 1.0 | 0.06 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < mask_len))` |
| store.global | 0.05 | 1.5% | 1.0 | 0.06 | `if (condval == (bool)0) \| if (condval == (bool)1)` |
| store.global#2 | 0.05 | 1.5% | 1.0 | 0.06 | `if (condval_1 == (bool)0) \| if (condval_1 == (bool)1)` |
| store.global#3 | 0.05 | 1.5% | 0.5 | 0.10 | `if ((((int)threadIdx.x) & 31) < 3) \| if ((((int)threadIdx.x) & 31) < 6)` |

### metadata_builder_kernel grid (2, 1, 1) block (128, 1, 1)

289 launches, 8 warps; ranges, warp and last warp from the 2 recording CTAs' 8 warps; span 1.54 us, warp 0.84 us, nsys graphs 1.66 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.46 | 55.5% | 1.25 | 74.1% |
| unattributed | 0.30 | 35.9% | 0.38 | 22.5% |
| store | 0.07 | 8.6% | 0.06 | 3.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#3 | 0.42 | 50.7% | 0.5 | 1.18 | `for (int __1 = 0; __1 < 4; ++__1)` |
| load.global#2 | 0.03 | 3.3% | 0.1 | 0.22 | `if ((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) < nr))` |
| store.global#3 | 0.03 | 3.3% | 0.5 | 0.06 | `if (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) == 0)` |
| store.global#4 | 0.03 | 3.3% | 0.5 | 0.06 | `positions[(((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))` |
| load.global | 0.01 | 1.4% | 0.1 | 0.10 | `if ((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) <= nr))` |
| store.global | 0.01 | 1.4% | 0.1 | 0.10 | `out_starts[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] = condval;` |
| store.global#2 | 0.00 | 0.5% | 0.1 | 0.03 | `request_positions[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] = condval_2;` |

### rope_kernel grid (3, 1, 1) block (128, 1, 1)

224 launches, 12 warps; ranges, warp and last warp from the 3 recording CTAs' 12 warps; span 0.99 us, warp 0.94 us, nsys graphs 1.44 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.72 | 76.0% | 1.49 | 78.0% |
| load | 0.19 | 20.3% | 0.38 | 20.0% |
| store | 0.03 | 3.7% | 0.04 | 2.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.10 | 10.2% | 1.0 | 0.10 | `if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < tokens)) \| int64_t position ` |
| load.global#2 | 0.10 | 10.2% | 1.0 | 0.13 | `if (((position < ((int64_t)positions_max)) && ((int64_t)0 <= position)))` |
| store.global | 0.03 | 3.7% | 1.0 | 0.06 | `*(uint4*)(x + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + ((((int64_t)((int)thread` |

### mhc_collapse_kernel grid (240, 1, 1) block (128, 1, 1)

51 launches, 960 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 8.54 us, warp 4.84 us, nsys graphs 5.92 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 3.71 | 76.7% | 2.68 | 48.5% |
| load | 1.07 | 22.1% | 2.73 | 49.5% |
| store | 0.06 | 1.2% | 0.11 | 1.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.07 | 22.1% | 1.0 | 3.49 | `*(float4*)(mv + 0) = *(float4*)(mix + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int6` |
| store.global | 0.06 | 1.2% | 1.0 | 0.10 | `*(uint4*)(out + ((((int64_t)((int)blockIdx.x)) * (int64_t)1024) + (((int64_t)((int)threadI` |

### moe_combine_kernel grid (5, 48, 1) block (128, 1, 1)

102 launches, 960 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 4.13 us, warp 3.02 us, nsys graphs 2.56 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 2.18 | 72.1% | 3.36 | 85.0% |
| unattributed | 0.55 | 18.2% | 0.51 | 12.9% |
| store | 0.29 | 9.7% | 0.09 | 2.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 2.18 | 72.1% | 1.0 | 3.46 | `for (int k = 0; k < 3; ++k) \| for (int k = 0; k < 6; ++k) \| if ((((int)blockIdx.y) < share` |
| store.global | 0.29 | 9.7% | 1.0 | 1.07 | `{` |

### token_ranks_kernel grid (8, 1, 1) block (256, 1, 1)

51 launches, 64 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 7.81 us, warp 7.73 us, nsys graphs 9.57 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 6.98 | 90.3% | 7.16 | 89.7% |
| unattributed | 0.36 | 4.6% | 0.38 | 4.7% |
| compute | 0.35 | 4.5% | 0.40 | 5.0% |
| store | 0.04 | 0.6% | 0.05 | 0.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 6.98 | 90.3% | 1.0 | 7.23 | `if ((((int64_t)0 <= token_ids[((int64_t)((int)blockIdx.x))]) && (token_ids[((int64_t)((int` |
| compute.reduce | 0.35 | 4.5% | 1.0 | 0.67 | `total[0] = tl::AllReduce<tl::SumOp, 256, 1, 0, tl::NamedBarrier<256>>::run(total[0], (&(wo` |
| store.global | 0.04 | 0.6% | 1.0 | 0.06 | `output[((int64_t)((int)blockIdx.x))] = ((int64_t)total[0]);` |

### norm_forward_and_per_token_cast_kernel_kernel grid (6, 1, 1) block (128, 1, 1)

105 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 2.94 us, warp 2.94 us, nsys graphs 1.54 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 2.47 | 83.9% | 3.23 | 86.9% |
| load | 0.44 | 14.9% | 0.46 | 12.3% |
| compute | 0.11 | 3.8% | 0.11 | 3.0% |
| wait | 0.05 | 1.8% | 0.06 | 1.6% |
| unattributed | 0.00 | 0.0% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*4 (est.: sampled iterations x4) | 1.79 | 60.9% | 1.0 | 1.92 | `for (int j_2 = 0; j_2 < 2; ++j_2)` |
| store.global*4 (est.: sampled iterations x4) | 0.41 | 13.9% | 1.0 | 0.51 | `for (int j_1 = 0; j_1 < 4; ++j_1)` |
| load.global*4 (est.: sampled iterations x4) | 0.23 | 8.0% | 1.0 | 0.26 | `if (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 4)) < num_tokens)` |
| load.global#3*4 (est.: sampled iterations x4) | 0.17 | 5.8% | 1.0 | 0.26 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 128) + ((((int)threadIdx.x) & 15` |
| store.global#5*4 (est.: sampled iterations x4) | 0.17 | 5.8% | 1.0 | 0.26 | `*(uint4*)(out + ((((((int64_t)vi_1) * (int64_t)128) + ((((int64_t)((int)threadIdx.x)) & (i` |
| compute.reduce | 0.11 | 3.8% | 1.0 | 0.13 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| wait.sync | 0.05 | 1.8% | 1.0 | 0.10 | `__syncthreads();` |
| store.global#3 | 0.05 | 1.8% | 1.0 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |
| store.global#2 | 0.05 | 1.5% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| load.global#2 | 0.03 | 1.2% | 1.0 | 0.06 | `*(float2*)(row_partial + 0) = *(float2*)(sum_shared + ((((int)threadIdx.x) >> 6) * 2)); \| ` |

### indexer_topk_segment_kernel grid (1, 48, 1) block (512, 1, 1)

34 launches, 768 warps; ranges, warp and last warp from the 4 recording CTAs' 64 warps; span 10.98 us, warp 10.79 us, nsys graphs 0.00 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed, with loops' unsampled iterations | 5.58 | 51.7% | 5.50 | 50.6% |
| wait | 3.73 | 34.5% | 3.90 | 35.9% |
| load | 1.33 | 12.3% | 1.28 | 11.8% |
| store | 0.15 | 1.4% | 0.18 | 1.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.33 | 12.3% | 1.0 | 1.52 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 4096); \| i` |
| wait.sync (sampled iterations) | 0.98 | 9.1% | 14.5 | 1.14 | `__syncthreads();` |
| wait.sync#20*9 (est.: sampled iterations x9) | 0.31 | 2.9% | 0.5 | 0.43 | `__syncthreads();` |
| wait.sync#24*9 (est.: sampled iterations x9) | 0.30 | 2.8% | 0.5 | 0.43 | `__syncthreads();` |
| wait.sync#19*9 (est.: sampled iterations x9) | 0.29 | 2.7% | 0.5 | 0.43 | `__syncthreads();` |
| wait.sync#23*9 (est.: sampled iterations x9) | 0.29 | 2.7% | 0.5 | 0.43 | `__syncthreads();` |
| wait.sync#6*8 (est.: sampled iterations x8) | 0.29 | 2.7% | 0.5 | 0.38 | `__syncthreads();` |
| wait.sync#14*8 (est.: sampled iterations x8) | 0.28 | 2.6% | 0.5 | 0.38 | `__syncthreads();` |
| wait.sync#5*8 (est.: sampled iterations x8) | 0.26 | 2.4% | 0.5 | 0.38 | `__syncthreads();` |
| wait.sync#13*8 (est.: sampled iterations x8) | 0.26 | 2.4% | 0.5 | 0.38 | `__syncthreads();` |
| store.global | 0.15 | 1.4% | 1.5 | 0.19 | `for (int i_3 = 0; i_3 < 4; ++i_3) \| if (((int)threadIdx.x) < total) \| if (((int)threadIdx.` |
| wait.sync#25 | 0.04 | 0.4% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#17 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#16 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#8 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#18 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#2 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#9 | 0.03 | 0.3% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#7 | 0.03 | 0.3% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#11 | 0.03 | 0.3% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#12 | 0.03 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#15 | 0.03 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#10 | 0.03 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#4 | 0.02 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#22 | 0.02 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#21 | 0.02 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#3 | 0.02 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |

### norm_forward_and_per_token_cast_kernel_kernel grid (24, 1, 1) block (128, 1, 1)

102 launches, 96 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 2.53 us, warp 2.30 us, nsys graphs 1.92 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 1.61 | 69.9% | 3.00 | 87.2% |
| load | 0.34 | 14.7% | 0.35 | 10.2% |
| unattributed | 0.13 | 5.5% | 0.00 | 0.0% |
| compute | 0.12 | 5.1% | 0.12 | 3.5% |
| wait | 0.11 | 4.9% | 0.05 | 1.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*3 (est.: sampled iterations x3) | 1.10 | 47.9% | 1.0 | 1.15 | `for (int j_2 = 0; j_2 < 2; ++j_2)` |
| store.global*3 (est.: sampled iterations x3) | 0.31 | 13.3% | 1.0 | 0.38 | `for (int j_1 = 0; j_1 < 4; ++j_1)` |
| load.global*3 (est.: sampled iterations x3) | 0.17 | 7.6% | 1.0 | 0.19 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 6)) < num_tokens) && (((vi * 2) + (` |
| load.global#3*3 (est.: sampled iterations x3) | 0.13 | 5.7% | 1.0 | 0.19 | `if (((vi_1 * 2) + ((((int)threadIdx.x) & 63) >> 5)) < 5)` |
| store.global#5*3 (est.: sampled iterations x3) | 0.13 | 5.5% | 1.0 | 0.19 | `*(uint4*)(out + ((((((int64_t)vi_1) * (int64_t)512) + ((((int64_t)((int)threadIdx.x)) & (i` |
| compute.reduce | 0.12 | 5.1% | 1.0 | 0.13 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| wait.sync | 0.11 | 4.9% | 1.0 | 0.19 | `__syncthreads();` |
| store.global#2 | 0.04 | 1.8% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| load.global#2 | 0.03 | 1.4% | 1.0 | 0.06 | `*(float2*)(row_partial + 0) = *(float2*)(sum_shared + ((((int)threadIdx.x) >> 6) * 2)); \| ` |
| store.global#3 | 0.03 | 1.4% | 0.5 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |

### indexer_score_kernel grid (128, 48, 1) block (128, 1, 1)

17 launches, 24576 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 20.51 us, warp 2.67 us, nsys graphs 15.47 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.34 | 50.2% | 0.53 | 59.0% |
| unattributed, with loops' unsampled iterations | 0.98 | 36.8% | 0.32 | 34.9% |
| wait | 0.25 | 9.4% | 0.04 | 4.8% |
| compute | 0.07 | 2.7% | 0.01 | 0.8% |
| store | 0.03 | 1.0% | 0.00 | 0.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 (sampled iterations) | 0.88 | 32.8% | 0.5 | 1.92 | `for (int i_1 = 0; i_1 < 32; ++i_1)` |
| load.global | 0.30 | 11.3% | 1.0 | 0.45 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 4096); \| i` |
| load.global#3 (sampled iterations) | 0.16 | 6.1% | 0.5 | 0.64 | `if (((int)threadIdx.x) < 64)` |
| wait.sync#2 (sampled iterations) | 0.15 | 5.6% | 0.5 | 0.61 | `__syncthreads();` |
| compute.mma (sampled iterations) | 0.07 | 2.7% | 0.5 | 0.16 | `for (int ki = 0; ki < 2; ++ki)` |
| wait.sync (sampled iterations) | 0.04 | 1.3% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#4 (sampled iterations) | 0.04 | 1.3% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#3 (sampled iterations) | 0.03 | 1.0% | 0.5 | 0.10 | `__syncthreads();` |
| store.global (sampled iterations) | 0.03 | 1.0% | 0.5 | 0.06 | `if ((((int)threadIdx.x) >> 6) == 0)` |

### topk_ids_kernel grid (16, 1, 1) block (512, 1, 1)

16 launches, 256 warps; ranges, warp and last warp from the 4 recording CTAs' 64 warps; span 20.98 us, warp 19.04 us, nsys graphs 21.34 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 15.87 | 83.4% | 16.02 | 74.1% |
| unattributed | 2.06 | 10.8% | 1.94 | 9.0% |
| wait | 0.88 | 4.6% | 0.77 | 3.6% |
| store | 0.23 | 1.2% | 2.88 | 13.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 15.87 | 83.4% | 1.0 | 16.98 | `for (int t = 0; t < (vocab >> 14); ++t)` |
| wait.sync | 0.74 | 3.9% | 1.0 | 1.76 | `__syncthreads();` |
| store.global | 0.23 | 1.2% | 1.0 | 2.50 | `if ((((int)threadIdx.x) >> 5) == 0)` |
| wait.sync#2 | 0.08 | 0.4% | 1.0 | 0.16 | `__syncthreads();` |
| wait.sync#3 | 0.06 | 0.3% | 1.0 | 0.06 | `__syncthreads();` |

### moe_swiglu_cast_kernel_kernel grid (8, 120, 1) block (288, 1, 1)

51 launches, 8640 warps; ranges, warp and last warp from the 4 recording CTAs' 36 warps; span 6.43 us, warp 1.39 us, nsys graphs 4.06 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.75 | 53.8% | 0.20 | 20.3% |
| load | 0.42 | 30.4% | 0.54 | 56.1% |
| compute | 0.14 | 9.7% | 0.22 | 23.0% |
| store | 0.09 | 6.1% | 0.01 | 0.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.42 | 30.4% | 1.0 | 0.93 | `for (int r = 0; r < 4; ++r)` |
| compute.reduce*4 (est.: sampled iterations x4) | 0.14 | 9.7% | 0.2 | 0.64 | `if (0 <= orders[r_2])` |
| store.global*4 (est.: sampled iterations x4) | 0.09 | 6.1% | 0.2 | 0.38 | `if (0 <= orders[r_2])` |

### topk_log_softmax_kernel grid (8, 1, 1) block (1024, 1, 1)

17 launches, 256 warps; ranges, warp and last warp from the 4 recording CTAs' 128 warps; span 17.70 us, warp 16.87 us, nsys graphs 16.99 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 13.81 | 81.9% | 16.76 | 84.3% |
| unattributed, with loops' unsampled iterations | 1.54 | 9.1% | 1.74 | 8.7% |
| wait | 1.12 | 6.7% | 0.92 | 4.7% |
| compute | 0.39 | 2.3% | 0.39 | 1.9% |
| store | 0.00 | 0.0% | 0.06 | 0.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 7.03 | 41.7% | 1.0 | 7.36 | `if (0 < (vocab & 2047)) \| if (0 < (vocab & 8191))` |
| load.global#2 | 6.77 | 40.1% | 1.0 | 7.49 | `for (int step_1 = 0; step_1 < (vocab >> 11); ++step_1) \| for (int step_1 = 0; step_1 < (vo` |
| wait.sync#2 | 0.52 | 3.1% | 1.0 | 1.15 | `__syncthreads();` |
| wait.sync | 0.35 | 2.1% | 1.0 | 0.99 | `__syncthreads();` |
| compute.reduce#3 | 0.17 | 1.0% | 1.0 | 0.22 | `ties[0] = vllm_add(ties[0], __shfl_xor_sync((uint)4294967295, ties[0], 16, 32));` |
| compute.reduce#2 | 0.11 | 0.7% | 1.0 | 0.16 | `se[0] = vllm_add(se[0], __shfl_xor_sync((uint)4294967295, se[0], 16, 32));` |
| compute.reduce | 0.11 | 0.6% | 1.0 | 0.13 | `best[0] = vllm_max(best[0], __shfl_xor_sync((uint)4294967295, best[0], 16, 32));` |
| wait.sync#4 | 0.10 | 0.6% | 1.0 | 0.16 | `__syncthreads();` |
| wait.sync#3 | 0.08 | 0.5% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#5 | 0.08 | 0.5% | 1.0 | 0.10 | `__syncthreads();` |
| load.global#3 (sampled iterations) | 0.01 | 0.1% | 0.0 | 0.38 | `float value_32 = vllm_sub(vllm_shifted(logits[((((int64_t)((int)blockIdx.x)) * logits_stri` |
| store.global (sampled iterations) | 0.00 | 0.0% | 0.0 | 0.06 | `output[(((((int64_t)c) * (int64_t)1024) + (((int64_t)((int)blockIdx.x)) * ((int64_t)topk))` |

### topk_log_softmax_kernel grid (16, 1, 1) block (1024, 1, 1)

16 launches, 512 warps; ranges, warp and last warp from the 4 recording CTAs' 128 warps; span 20.90 us, warp 18.87 us, nsys graphs 18.50 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 15.56 | 82.5% | 17.24 | 84.6% |
| unattributed, with loops' unsampled iterations | 1.56 | 8.3% | 1.72 | 8.4% |
| wait | 1.35 | 7.2% | 0.99 | 4.9% |
| compute | 0.39 | 2.1% | 0.37 | 1.8% |
| store | 0.00 | 0.0% | 0.06 | 0.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 8.17 | 43.3% | 1.0 | 9.62 | `if (0 < (vocab & 2047)) \| if (0 < (vocab & 8191))` |
| load.global#2 | 7.38 | 39.1% | 1.0 | 8.08 | `for (int step_1 = 0; step_1 < (vocab >> 11); ++step_1) \| for (int step_1 = 0; step_1 < (vo` |
| wait.sync | 0.57 | 3.0% | 1.0 | 1.44 | `__syncthreads();` |
| wait.sync#2 | 0.52 | 2.8% | 1.0 | 1.20 | `__syncthreads();` |
| compute.reduce#3 | 0.17 | 0.9% | 1.0 | 0.22 | `ties[0] = vllm_add(ties[0], __shfl_xor_sync((uint)4294967295, ties[0], 16, 32));` |
| compute.reduce#2 | 0.11 | 0.6% | 1.0 | 0.18 | `se[0] = vllm_add(se[0], __shfl_xor_sync((uint)4294967295, se[0], 16, 32));` |
| compute.reduce | 0.10 | 0.5% | 1.0 | 0.13 | `best[0] = vllm_max(best[0], __shfl_xor_sync((uint)4294967295, best[0], 16, 32));` |
| wait.sync#4 | 0.10 | 0.5% | 1.0 | 0.16 | `__syncthreads();` |
| wait.sync#5 | 0.08 | 0.4% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#3 | 0.08 | 0.4% | 1.0 | 0.10 | `__syncthreads();` |
| load.global#3 (sampled iterations) | 0.01 | 0.1% | 0.0 | 0.38 | `float value_32 = vllm_sub(vllm_shifted(logits[((((int64_t)((int)blockIdx.x)) * logits_stri` |
| store.global (sampled iterations) | 0.00 | 0.0% | 0.0 | 0.06 | `output[(((((int64_t)c) * (int64_t)1024) + (((int64_t)((int)blockIdx.x)) * ((int64_t)topk))` |

### sink_rows_kernel grid (8, 64, 1) block (256, 1, 1)

17 launches, 4096 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 16.96 us, warp 5.07 us, nsys graphs 8.11 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 2.48 | 49.0% | 3.18 | 42.2% |
| load | 1.54 | 30.4% | 2.95 | 39.1% |
| unattributed | 1.04 | 20.6% | 1.40 | 18.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 2.43 | 48.0% | 1.0 | 5.44 | `for (int k_1 = 0; k_1 < 8; ++k_1)` |
| load.global | 1.54 | 30.4% | 1.0 | 4.29 | `row = (max((((int64_t)num_sampled[((int64_t)((int)blockIdx.x))]) - (int64_t)1), (int64_t)0` |
| store.global#2 | 0.05 | 1.0% | 0.0 | 1.60 | `tokens[((((int64_t)((int)blockIdx.x)) * (int64_t)3) + ((int64_t)((int)threadIdx.x)))] = to` |
| load.global#2 | 0.00 | 0.1% | 0.0 | 0.10 | `token = max(alternates[((((int64_t)((int)blockIdx.x)) * (int64_t)3) + ((int64_t)((int)thre` |

### compress_kernel grid (12, 1, 1) block (128, 1, 1)

34 launches, 48 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 6.29 us, warp 5.26 us, nsys graphs 6.59 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 2.76 | 52.4% | 2.22 | 30.4% |
| load | 1.97 | 37.5% | 4.05 | 55.4% |
| store | 0.44 | 8.3% | 0.95 | 13.0% |
| compute | 0.09 | 1.8% | 0.09 | 1.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.24 | 23.6% | 1.0 | 1.54 | `if ((((0 <= counts[0]) & (0 < counts[1])) & (counts[1] <= 8)) & (((((int)blockIdx.x) * 4) ` |
| load.global#2*4 (est.: sampled iterations x4) | 0.62 | 11.9% | 1.0 | 1.34 | `*(float4*)(v1 + 0) = *(float4*)(values + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048)` |
| store.global*4 (est.: sampled iterations x4) | 0.16 | 3.0% | 1.0 | 0.26 | `*(uint2*)(out + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + ((((int64_t)((int)thre` |
| compute.reduce | 0.09 | 1.8% | 1.0 | 0.10 | `float total = tl::warp_reduce_sum(ss[0]);` |
| store.global#3 | 0.08 | 1.6% | 0.5 | 0.16 | `emitted_slots[((((int64_t)((int)blockIdx.x)) * (int64_t)4) + (((int64_t)((int)threadIdx.x)` |
| load.global#3*4 (est.: sampled iterations x4) | 0.07 | 1.3% | 0.5 | 0.13 | `*(float4*)(w + 0) = *(float4*)(weight + ((c_1 * 128) + ((((int)threadIdx.x) & 31) * 4)));` |
| store.global#3*4 (est.: sampled iterations x4) | 0.07 | 1.3% | 0.5 | 0.13 | `*(uint2*)(out + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + ((((int64_t)((int)thre` |
| store.global#2*4 (est.: sampled iterations x4) | 0.06 | 1.1% | 0.2 | 0.13 | `if ((slot >> (int64_t)3) < ((int64_t)blocks))` |
| store.global#5 | 0.04 | 0.7% | 0.5 | 0.14 | `emitted_slots[((((int64_t)((int)blockIdx.x)) * (int64_t)4) + (((int64_t)((int)threadIdx.x)` |
| load.global#3 | 0.02 | 0.4% | 0.5 | 0.03 | `if (valid)` |
| load.global#4 | 0.02 | 0.3% | 0.5 | 0.03 | `if (valid)` |
| store.global#4 | 0.02 | 0.3% | 0.5 | 0.03 | `emitted[((((int64_t)((int)blockIdx.x)) * (int64_t)4) + (((int64_t)((int)threadIdx.x)) >> (` |
| store.global#2 | 0.02 | 0.3% | 0.5 | 0.02 | `emitted[((((int64_t)((int)blockIdx.x)) * (int64_t)4) + (((int64_t)((int)threadIdx.x)) >> (` |

### sink_candidates_kernel grid (8, 127, 1) block (128, 1, 1)

17 launches, 4064 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 11.62 us, warp 4.59 us, nsys graphs 9.70 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.46 | 31.8% | 0.80 | 19.1% |
| compute | 1.40 | 30.5% | 1.32 | 31.7% |
| load | 0.81 | 17.7% | 1.11 | 26.8% |
| wait | 0.62 | 13.4% | 0.57 | 13.7% |
| store | 0.30 | 6.6% | 0.36 | 8.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.reduce*4 (est.: sampled iterations x4) | 1.01 | 22.0% | 1.0 | 1.28 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| load.global | 0.81 | 17.7% | 1.0 | 1.18 | `local[0] = max((num_sampled[((int64_t)((int)blockIdx.x))] - 1), 0);` |
| compute.reduce#2*4 (est.: sampled iterations x4) | 0.39 | 8.5% | 1.0 | 0.51 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 2, 32);` |
| wait.sync#2*4 (est.: sampled iterations x4) | 0.31 | 6.8% | 1.0 | 0.51 | `__syncthreads();` |
| store.global*4 (est.: sampled iterations x4) | 0.30 | 6.6% | 1.0 | 0.38 | `if (((int)threadIdx.x) == 0)` |
| wait.sync*4 (est.: sampled iterations x4) | 0.30 | 6.6% | 1.0 | 0.51 | `__syncthreads();` |

### metadata_chunk_kernel grid (48, 1, 1) block (128, 1, 1)

34 launches, 192 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 5.55 us, warp 4.23 us, nsys graphs 5.30 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 2.38 | 56.3% | 2.31 | 35.8% |
| load | 1.76 | 41.7% | 3.99 | 62.0% |
| store | 0.08 | 2.0% | 0.14 | 2.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.72 | 40.7% | 1.0 | 2.91 | `if ((((offset + ((int)blockIdx.x)) < src_rows) && (0 <= (offset + ((int)blockIdx.x)))))` |
| store.global | 0.06 | 1.5% | 1.0 | 0.06 | `swa[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = con` |
| load.global#2 | 0.04 | 1.0% | 0.2 | 0.32 | `if (((0 <= req) & ((int64_t)0 <= pos)))` |
| store.global#2 | 0.01 | 0.3% | 0.2 | 0.06 | `lengths[((int64_t)((int)blockIdx.x))] = ((int)max(info[1], (int64_t)0));` |
| store.global#3 | 0.01 | 0.2% | 0.2 | 0.06 | `top_lengths[((int64_t)((int)blockIdx.x))] = condval_12; \| top_lengths[((int64_t)((int)bloc` |

### sparse_mla_kernel_1 grid (16, 40, 1) block (128, 1, 1)

51 launches, 640 CTAs, 2560 warps; span 3.90 us, warp 2.33 us, nsys graphs 3.74 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 2.32 | 99.5% | 1.33 | 100.0% |
| load | 0.01 | 0.4% | 0.00 | 0.0% |
| wait | 0.00 | 0.1% | 0.00 | 0.0% |
| compute | 0.00 | 0.0% | 0.00 | 0.0% |
| store | 0.00 | 0.0% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#3 | 0.01 | 0.3% | 0.0 | 1.57 | `M = __fmul_rn(Sinks[((int)blockIdx.x)], 0x1.7154764ee6c2fp+0f/*1.442695e+00*/);` |
| wait.sync#13 | 0.00 | 0.1% | 0.0 | 0.83 | `__syncthreads();` |
| load.global#4 | 0.00 | 0.0% | 0.0 | 0.83 | `if (((int)threadIdx.x) < 32)` |
| compute.reduce#3 | 0.00 | 0.0% | 0.0 | 0.16 | `amax[0] = tl::AllReduce<tl::MaxOp, 8, 1, 0, tl::NamedBarrier<128>>::run(amax[0]);` |
| wait.sync#12 | 0.00 | 0.0% | 0.0 | 0.29 | `__syncthreads();` |
| store.global#2 | 0.00 | 0.0% | 0.0 | 0.10 | `*(uint2*)(Output + (((((int64_t)((int)blockIdx.y)) * (int64_t)8192) + (((int64_t)((int)blo` |

### compress_gather_partial_state_kernel grid (8, 1, 1) block (128, 1, 1)

17 launches, 32 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 12.45 us, warp 10.46 us, nsys graphs 0.00 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 4.38 | 41.9% | 4.18 | 36.7% |
| unattributed | 4.32 | 41.3% | 2.90 | 25.5% |
| store | 1.76 | 16.9% | 4.30 | 37.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 4.26 | 40.7% | 1.0 | 4.26 | `int nt = counts[0];` |
| store.global*2 (est.: sampled iterations x2) | 1.71 | 16.3% | 1.0 | 3.01 | `if (half == 0)` |
| load.global#2*2 (est.: sampled iterations x2) | 0.12 | 1.1% | 1.0 | 0.13 | `for (int i = 0; i < 4; ++i)` |
| store.global#2 | 0.06 | 0.5% | 1.0 | 0.10 | `if (((int)threadIdx.x) == 0)` |

### gemm_splitk_reduce_cast_kernel grid (16, 12, 1) block (128, 1, 1)

102 launches, 768 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 1.76 us, warp 1.61 us, nsys graphs 1.70 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.94 | 58.3% | 1.07 | 58.0% |
| unattributed | 0.49 | 30.6% | 0.59 | 31.8% |
| compute | 0.09 | 5.6% | 0.09 | 5.0% |
| store | 0.09 | 5.6% | 0.10 | 5.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.94 | 58.3% | 1.0 | 1.06 | `if ((1 <= M))` |
| compute.reduce | 0.09 | 5.6% | 1.0 | 0.10 | `amax[0] = tl::AllReduce<tl::MaxOp, 8, 1, 0, tl::NamedBarrier<128>>::run(amax[0]);` |
| store.global | 0.09 | 5.6% | 1.0 | 0.13 | `if (((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 5)) < M)` |

### per_token_cast_kernel_kernel grid (18, 1, 1) block (64, 1, 1)

153 launches, 36 warps; ranges, warp and last warp from the 5 recording CTAs' 10 warps; span 1.15 us, warp 1.00 us, nsys graphs 1.66 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.34 | 33.7% | 0.41 | 32.8% |
| unattributed | 0.34 | 33.7% | 0.36 | 28.5% |
| store | 0.22 | 22.4% | 0.33 | 26.1% |
| compute | 0.10 | 10.3% | 0.16 | 12.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.25 | 24.7% | 1.0 | 0.29 | `for (int i = 0; i < 2; ++i)` |
| store.global#2 | 0.11 | 10.9% | 1.0 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.10 | 10.3% | 1.0 | 0.13 | `amax_fragment[i_1] = tl::AllReduce<tl::MaxOp, 2, 1, 0, tl::NamedBarrier<64>>::run(amax_fra` |
| load.global#2*2 (est.: sampled iterations x2) | 0.09 | 9.0% | 1.0 | 0.13 | `amax_fragment[i_1] = cutlass::bfloat16_t(__hmax((bfloat16_t(((nv_bfloat162*)(&(amax_fragme` |
| store.global*2 (est.: sampled iterations x2) | 0.06 | 6.4% | 0.6 | 0.13 | `out_sf[((((((int64_t)((int)blockIdx.x)) % (int64_t)15) * (int64_t)32) + (((int64_t)((int)t` |
| store.global#3*2 (est.: sampled iterations x2) | 0.05 | 5.1% | 0.6 | 0.13 | `*(fp8_e4_16_t*)(out + ((((((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int64_t)10240) +` |

### per_token_cast_kernel_kernel grid (30, 1, 1) block (64, 1, 1)

136 launches, 60 warps; ranges, warp and last warp from the 5 recording CTAs' 10 warps; span 1.25 us, warp 1.19 us, nsys graphs 1.12 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.45 | 37.7% | 0.53 | 38.0% |
| load | 0.35 | 29.7% | 0.48 | 33.9% |
| store | 0.29 | 24.6% | 0.30 | 21.3% |
| compute | 0.10 | 8.1% | 0.10 | 6.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.27 | 22.7% | 1.0 | 0.29 | `for (int i = 0; i < 2; ++i)` |
| store.global#2 | 0.11 | 9.4% | 1.0 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.10 | 8.1% | 1.0 | 0.13 | `amax_fragment[i_1] = tl::AllReduce<tl::MaxOp, 2, 1, 0, tl::NamedBarrier<64>>::run(amax_fra` |
| store.global#3*2 (est.: sampled iterations x2) | 0.09 | 7.6% | 1.0 | 0.13 | `*(fp8_e4_16_t*)(out + ((((((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int64_t)10240) +` |
| store.global*2 (est.: sampled iterations x2) | 0.09 | 7.6% | 1.0 | 0.13 | `out_sf[((((((int64_t)((int)blockIdx.x)) % (int64_t)15) * (int64_t)32) + (((int64_t)((int)t` |
| load.global#2*2 (est.: sampled iterations x2) | 0.08 | 7.0% | 1.0 | 0.13 | `amax_fragment[i_1] = cutlass::bfloat16_t(__hmax((bfloat16_t(((nv_bfloat162*)(&(amax_fragme` |

### sparse_mla_kernel_1 grid (16, 48, 1) block (128, 1, 1)

34 launches, 768 CTAs, 3072 warps; span 4.85 us, warp 2.46 us, nsys graphs 7.87 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 2.45 | 99.7% | 1.58 | 90.2% |
| load | 0.01 | 0.2% | 0.12 | 7.1% |
| wait | 0.00 | 0.1% | 0.02 | 1.4% |
| compute | 0.00 | 0.0% | 0.02 | 0.9% |
| store | 0.00 | 0.0% | 0.01 | 0.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#3 | 0.01 | 0.2% | 0.0 | 1.30 | `M = __fmul_rn(Sinks[((int)blockIdx.x)], 0x1.7154764ee6c2fp+0f/*1.442695e+00*/);` |
| wait.sync#13 | 0.00 | 0.0% | 0.0 | 0.35 | `__syncthreads();` |
| compute.reduce#3 | 0.00 | 0.0% | 0.0 | 0.19 | `amax[0] = tl::AllReduce<tl::MaxOp, 8, 1, 0, tl::NamedBarrier<128>>::run(amax[0]);` |
| load.global#4 | 0.00 | 0.0% | 0.0 | 0.35 | `if (((int)threadIdx.x) < 32)` |
| wait.sync#12 | 0.00 | 0.0% | 0.0 | 0.13 | `__syncthreads();` |
| store.global#2 | 0.00 | 0.0% | 0.0 | 0.10 | `*(uint2*)(Output + (((((int64_t)((int)blockIdx.y)) * (int64_t)8192) + (((int64_t)((int)blo` |

### indexer_topk_segment_kernel grid (2, 48, 1) block (512, 1, 1)

17 launches, 1536 warps; ranges, warp and last warp from the 4 recording CTAs' 64 warps; span 10.18 us, warp 5.33 us, nsys graphs 0.00 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 3.13 | 58.7% | 6.36 | 60.4% |
| unattributed, with loops' unsampled iterations | 1.79 | 33.6% | 3.02 | 28.7% |
| load | 0.36 | 6.8% | 1.03 | 9.8% |
| store | 0.05 | 0.9% | 0.11 | 1.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync#20*9 (est.: sampled iterations x9) | 0.34 | 6.3% | 0.5 | 0.86 | `__syncthreads();` |
| wait.sync#24*9 (est.: sampled iterations x9) | 0.34 | 6.3% | 0.5 | 0.86 | `__syncthreads();` |
| wait.sync#19*9 (est.: sampled iterations x9) | 0.32 | 6.0% | 0.5 | 0.86 | `__syncthreads();` |
| wait.sync#23*9 (est.: sampled iterations x9) | 0.32 | 6.0% | 0.5 | 0.86 | `__syncthreads();` |
| wait.sync#6*8 (est.: sampled iterations x8) | 0.30 | 5.6% | 0.5 | 0.77 | `__syncthreads();` |
| wait.sync#14*8 (est.: sampled iterations x8) | 0.29 | 5.5% | 0.5 | 0.77 | `__syncthreads();` |
| wait.sync#13*8 (est.: sampled iterations x8) | 0.28 | 5.3% | 0.5 | 0.77 | `__syncthreads();` |
| wait.sync#5*8 (est.: sampled iterations x8) | 0.28 | 5.2% | 0.5 | 0.77 | `__syncthreads();` |
| load.global | 0.21 | 3.9% | 1.0 | 0.26 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 4096); \| i` |
| load.global#2 | 0.16 | 3.0% | 0.5 | 0.64 | `for (int i = 0; i < 8; ++i)` |
| wait.sync (sampled iterations) | 0.11 | 2.1% | 0.5 | 0.42 | `__syncthreads();` |
| wait.sync#25 | 0.05 | 0.9% | 0.5 | 0.16 | `__syncthreads();` |
| store.global | 0.05 | 0.9% | 0.5 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3) \| if (((int)threadIdx.x) < total) \| if (((int)threadIdx.` |
| wait.sync#18 | 0.04 | 0.8% | 0.5 | 0.13 | `__syncthreads();` |
| wait.sync#17 | 0.04 | 0.7% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#8 | 0.04 | 0.7% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#16 | 0.04 | 0.7% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#12 | 0.03 | 0.6% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#15 | 0.03 | 0.6% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#9 | 0.03 | 0.6% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#22 | 0.03 | 0.6% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#4 | 0.03 | 0.6% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#11 | 0.03 | 0.6% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#2 | 0.03 | 0.6% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#3 | 0.03 | 0.6% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#7 | 0.03 | 0.6% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#21 | 0.03 | 0.6% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#10 | 0.03 | 0.6% | 0.5 | 0.06 | `__syncthreads();` |

### indexer_score_kernel grid (64, 48, 1) block (128, 1, 1)

17 launches, 12288 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 10.50 us, warp 2.46 us, nsys graphs 0.00 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.19 | 48.3% | 0.53 | 59.9% |
| unattributed, with loops' unsampled iterations | 0.90 | 36.7% | 0.29 | 32.9% |
| wait | 0.26 | 10.6% | 0.03 | 3.6% |
| compute | 0.08 | 3.2% | 0.03 | 3.4% |
| store | 0.03 | 1.1% | 0.00 | 0.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 (sampled iterations) | 0.69 | 27.9% | 0.5 | 1.47 | `for (int i_1 = 0; i_1 < 32; ++i_1)` |
| load.global | 0.42 | 17.0% | 1.0 | 0.80 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 4096); \| i` |
| wait.sync#2 (sampled iterations) | 0.13 | 5.4% | 0.5 | 0.45 | `__syncthreads();` |
| load.global#3 (sampled iterations) | 0.08 | 3.4% | 0.5 | 0.51 | `if (((int)threadIdx.x) < 64)` |
| compute.mma (sampled iterations) | 0.08 | 3.2% | 0.5 | 0.19 | `for (int ki = 0; ki < 2; ++ki)` |
| wait.sync#4 (sampled iterations) | 0.06 | 2.4% | 0.5 | 0.19 | `__syncthreads();` |
| wait.sync (sampled iterations) | 0.04 | 1.5% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#3 (sampled iterations) | 0.03 | 1.4% | 0.5 | 0.10 | `__syncthreads();` |
| store.global (sampled iterations) | 0.03 | 1.1% | 0.5 | 0.10 | `if ((((int)threadIdx.x) >> 6) == 0)` |

### indexer_topk_merge_kernel grid (1, 48, 1) block (512, 1, 1)

17 launches, 768 warps; ranges, warp and last warp from the 4 recording CTAs' 64 warps; span 9.79 us, warp 9.50 us, nsys graphs 8.00 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 6.15 | 64.7% | 6.03 | 59.6% |
| unattributed | 2.28 | 24.0% | 2.38 | 23.4% |
| load | 1.01 | 10.6% | 1.60 | 15.8% |
| store | 0.07 | 0.7% | 0.12 | 1.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.00 | 10.6% | 1.0 | 1.18 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 8192);` |
| wait.sync#24*9 (est.: sampled iterations x9) | 0.68 | 7.1% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#20*9 (est.: sampled iterations x9) | 0.66 | 7.0% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#19*9 (est.: sampled iterations x9) | 0.65 | 6.8% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#23*9 (est.: sampled iterations x9) | 0.62 | 6.5% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#14*8 (est.: sampled iterations x8) | 0.61 | 6.4% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync#6*8 (est.: sampled iterations x8) | 0.60 | 6.3% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync#5*8 (est.: sampled iterations x8) | 0.58 | 6.1% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync#13*8 (est.: sampled iterations x8) | 0.57 | 6.0% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync | 0.15 | 1.6% | 1.0 | 0.35 | `__syncthreads();` |
| wait.sync#17 | 0.08 | 0.9% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#16 | 0.08 | 0.8% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#8 | 0.07 | 0.8% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#2 | 0.07 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#25 | 0.07 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#9 | 0.07 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#7 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#3 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| store.global#2 | 0.06 | 0.7% | 1.0 | 0.10 | `Values[((((int64_t)((int)blockIdx.y)) * (int64_t)512) + ((int64_t)((int)threadIdx.x)))] = ` |
| wait.sync#15 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#22 | 0.06 | 0.6% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#11 | 0.06 | 0.6% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#4 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#12 | 0.06 | 0.6% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#21 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#18 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#10 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| load.global#2 | 0.01 | 0.1% | 0.1 | 0.10 | `if (((0 <= ((int*)sel_idx)[((int)threadIdx.x)]) && (((int*)sel_idx)[((int)threadIdx.x)] < ` |
| store.global | 0.00 | 0.0% | 0.1 | 0.10 | `Values[((((int64_t)((int)blockIdx.y)) * (int64_t)512) + ((int64_t)((int)threadIdx.x)))] = ` |

### mhc_post_kernel grid (200, 1, 1) block (128, 1, 1)

17 launches, 800 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 10.53 us, warp 6.72 us, nsys graphs 3.33 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 6.52 | 97.0% | 8.23 | 94.8% |
| load | 0.15 | 2.2% | 0.33 | 3.8% |
| store | 0.05 | 0.8% | 0.12 | 1.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.15 | 2.2% | 1.0 | 0.19 | `*(float4*)(pm + 0) = *(float4*)(post + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int` |
| store.global | 0.05 | 0.8% | 1.0 | 0.06 | `for (int n_1 = 0; n_1 < 4; ++n_1)` |

### norm_forward_and_per_token_cast_kernel_kernel grid (5, 1, 1) block (128, 1, 1)

51 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 2.78 us, warp 2.76 us, nsys graphs 1.86 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 2.27 | 82.2% | 2.64 | 80.5% |
| load | 0.44 | 15.9% | 0.46 | 13.9% |
| compute | 0.11 | 4.1% | 0.11 | 3.4% |
| wait | 0.05 | 1.9% | 0.36 | 10.9% |
| unattributed | 0.00 | 0.0% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*4 (est.: sampled iterations x4) | 1.58 | 57.3% | 1.0 | 1.66 | `for (int j_2 = 0; j_2 < 2; ++j_2)` |
| store.global*4 (est.: sampled iterations x4) | 0.41 | 14.9% | 1.0 | 0.51 | `for (int j_1 = 0; j_1 < 4; ++j_1)` |
| load.global*4 (est.: sampled iterations x4) | 0.24 | 8.6% | 1.0 | 0.26 | `if (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 4)) < num_tokens)` |
| store.global#5*4 (est.: sampled iterations x4) | 0.18 | 6.5% | 1.0 | 0.26 | `*(uint4*)(out + ((((((int64_t)vi_1) * (int64_t)128) + ((((int64_t)((int)threadIdx.x)) & (i` |
| load.global#3*4 (est.: sampled iterations x4) | 0.17 | 6.0% | 1.0 | 0.26 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 128) + ((((int)threadIdx.x) & 15` |
| compute.reduce | 0.11 | 4.1% | 1.0 | 0.13 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| wait.sync | 0.05 | 1.9% | 1.0 | 0.10 | `__syncthreads();` |
| store.global#3 | 0.05 | 1.9% | 1.0 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |
| store.global#2 | 0.05 | 1.7% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| load.global#2 | 0.04 | 1.3% | 1.0 | 0.06 | `*(float2*)(row_partial + 0) = *(float2*)(sum_shared + ((((int)threadIdx.x) >> 6) * 2)); \| ` |

### rope_kernel grid (48, 1, 1) block (128, 1, 1)

102 launches, 192 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 1.25 us, warp 1.19 us, nsys graphs 1.57 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.70 | 58.5% | 1.00 | 66.7% |
| load | 0.46 | 38.3% | 0.46 | 30.8% |
| store | 0.04 | 3.3% | 0.04 | 2.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.46 | 38.3% | 1.0 | 0.48 | `if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < tokens)) \| int64_t position ` |
| store.global | 0.04 | 3.3% | 1.0 | 0.06 | `*(uint4*)(x + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + ((((int64_t)((int)thread` |

### moe_combine_kernel grid (5, 40, 1) block (128, 1, 1)

51 launches, 800 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 2.82 us, warp 2.06 us, nsys graphs 2.54 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.69 | 82.2% | 2.29 | 85.3% |
| unattributed | 0.31 | 15.2% | 0.34 | 12.6% |
| store | 0.05 | 2.6% | 0.06 | 2.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.69 | 82.2% | 1.0 | 2.02 | `for (int k = 0; k < 3; ++k) \| for (int k = 0; k < 6; ++k) \| if ((((int)blockIdx.y) < share` |
| store.global | 0.05 | 2.6% | 1.0 | 0.06 | `{` |

### norm_forward_and_per_token_cast_kernel_kernel grid (20, 1, 1) block (128, 1, 1)

51 launches, 80 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 2.40 us, warp 2.19 us, nsys graphs 2.14 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 1.56 | 71.1% | 1.57 | 58.4% |
| load | 0.35 | 15.8% | 0.34 | 12.5% |
| compute | 0.12 | 5.5% | 0.12 | 4.4% |
| wait | 0.11 | 4.9% | 0.06 | 2.2% |
| unattributed | 0.06 | 2.7% | 0.60 | 22.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*3 (est.: sampled iterations x3) | 1.07 | 48.7% | 1.0 | 1.15 | `for (int j_2 = 0; j_2 < 2; ++j_2)` |
| store.global*3 (est.: sampled iterations x3) | 0.29 | 13.4% | 1.0 | 0.38 | `for (int j_1 = 0; j_1 < 4; ++j_1)` |
| load.global*3 (est.: sampled iterations x3) | 0.18 | 8.2% | 1.0 | 0.19 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 6)) < num_tokens) && (((vi * 2) + (` |
| load.global#3*3 (est.: sampled iterations x3) | 0.13 | 6.0% | 1.0 | 0.19 | `if (((vi_1 * 2) + ((((int)threadIdx.x) & 63) >> 5)) < 5)` |
| store.global#5*3 (est.: sampled iterations x3) | 0.13 | 5.7% | 1.0 | 0.19 | `*(uint4*)(out + ((((((int64_t)vi_1) * (int64_t)512) + ((((int64_t)((int)threadIdx.x)) & (i` |
| compute.reduce | 0.12 | 5.5% | 1.0 | 0.13 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| wait.sync | 0.11 | 4.9% | 1.0 | 0.19 | `__syncthreads();` |
| store.global#2 | 0.04 | 1.9% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| load.global#2 | 0.03 | 1.6% | 1.0 | 0.06 | `*(float2*)(row_partial + 0) = *(float2*)(sum_shared + ((((int)threadIdx.x) >> 6) * 2)); \| ` |
| store.global#3 | 0.03 | 1.3% | 0.5 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |

### norm_forward_and_per_token_cast_kernel_kernel grid (2, 1, 1) block (128, 1, 1)

48 launches, 8 warps; ranges, warp and last warp from the 2 recording CTAs' 8 warps; span 2.85 us, warp 2.84 us, nsys graphs 1.41 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 2.48 | 87.1% | 2.53 | 88.8% |
| load | 0.48 | 16.9% | 0.46 | 16.0% |
| compute | 0.11 | 3.9% | 0.11 | 3.9% |
| wait | 0.05 | 1.7% | 0.04 | 1.5% |
| unattributed | 0.00 | 0.0% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*4 (est.: sampled iterations x4) | 1.82 | 64.1% | 1.0 | 1.92 | `for (int j_2 = 0; j_2 < 2; ++j_2)` |
| store.global*4 (est.: sampled iterations x4) | 0.42 | 14.6% | 1.0 | 0.51 | `for (int j_1 = 0; j_1 < 4; ++j_1)` |
| load.global*4 (est.: sampled iterations x4) | 0.24 | 8.4% | 1.0 | 0.26 | `if (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 4)) < num_tokens)` |
| load.global#3*4 (est.: sampled iterations x4) | 0.21 | 7.3% | 1.0 | 0.26 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 128) + ((((int)threadIdx.x) & 15` |
| store.global#5*4 (est.: sampled iterations x4) | 0.14 | 5.1% | 1.0 | 0.26 | `*(uint4*)(out + ((((((int64_t)vi_1) * (int64_t)128) + ((((int64_t)((int)threadIdx.x)) & (i` |
| compute.reduce | 0.11 | 3.9% | 1.0 | 0.13 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| store.global#3 | 0.05 | 1.7% | 1.0 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |
| wait.sync | 0.05 | 1.7% | 1.0 | 0.06 | `__syncthreads();` |
| store.global#2 | 0.04 | 1.5% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| load.global#2 | 0.03 | 1.1% | 1.0 | 0.03 | `*(float2*)(row_partial + 0) = *(float2*)(sum_shared + ((((int)threadIdx.x) >> 6) * 2)); \| ` |

### token_ranks_kernel grid (16, 1, 1) block (256, 1, 1)

16 launches, 128 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 8.45 us, warp 8.30 us, nsys graphs 9.36 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 7.48 | 90.1% | 7.61 | 90.2% |
| compute | 0.42 | 5.0% | 0.41 | 4.9% |
| unattributed | 0.36 | 4.4% | 0.36 | 4.3% |
| store | 0.04 | 0.5% | 0.05 | 0.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 7.48 | 90.1% | 1.0 | 7.84 | `if ((((int64_t)0 <= token_ids[((int64_t)((int)blockIdx.x))]) && (token_ids[((int64_t)((int` |
| compute.reduce | 0.42 | 5.0% | 1.0 | 0.80 | `total[0] = tl::AllReduce<tl::SumOp, 256, 1, 0, tl::NamedBarrier<256>>::run(total[0], (&(wo` |
| store.global | 0.04 | 0.5% | 1.0 | 0.06 | `output[((int64_t)((int)blockIdx.x))] = ((int64_t)total[0]);` |

### topk_ids_kernel grid (48, 1, 1) block (512, 1, 1)

1 launches, 768 warps; ranges, warp and last warp from the 4 recording CTAs' 64 warps; span 133.15 us, warp 123.54 us, nsys graphs 114.30 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 100.11 | 81.0% | 92.45 | 71.2% |
| wait | 21.27 | 17.2% | 33.89 | 26.1% |
| unattributed | 1.97 | 1.6% | 1.98 | 1.5% |
| store | 0.18 | 0.1% | 1.57 | 1.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 100.11 | 81.0% | 1.0 | 126.18 | `for (int t = 0; t < (vocab >> 14); ++t)` |
| wait.sync | 21.14 | 17.1% | 1.0 | 58.53 | `__syncthreads();` |
| store.global | 0.18 | 0.1% | 1.0 | 1.57 | `if ((((int)threadIdx.x) >> 5) == 0)` |
| wait.sync#2 | 0.08 | 0.1% | 1.0 | 0.16 | `__syncthreads();` |
| wait.sync#3 | 0.06 | 0.0% | 1.0 | 0.06 | `__syncthreads();` |

### router_partials_kernel grid (20, 1, 1) block (64, 1, 1)

51 launches, 40 warps; ranges, warp and last warp from the 4 recording CTAs' 8 warps; span 2.56 us, warp 2.52 us, nsys graphs 6.94 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.03 | 40.9% | 1.06 | 41.3% |
| unattributed | 0.93 | 36.9% | 0.95 | 36.9% |
| compute | 0.35 | 13.8% | 0.35 | 13.5% |
| store | 0.21 | 8.4% | 0.22 | 8.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#4 | 0.51 | 20.3% | 1.0 | 0.54 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < image_len)) \| if ((((((int)bl` |
| load.global | 0.39 | 15.5% | 1.0 | 0.42 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < mask_len))` |
| compute.reduce | 0.35 | 13.8% | 1.0 | 0.35 | `for (int k = 0; k < 3; ++k) \| for (int k = 0; k < 6; ++k)` |
| store.global#3 | 0.11 | 4.4% | 1.0 | 0.13 | `if ((((int)threadIdx.x) & 31) < 3) \| if ((((int)threadIdx.x) & 31) < 6)` |
| load.global#2 | 0.06 | 2.5% | 1.0 | 0.06 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < mask_len))` |
| load.global#3 | 0.06 | 2.5% | 1.0 | 0.06 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < mask_len))` |
| store.global | 0.05 | 2.1% | 1.0 | 0.06 | `if (condval == (bool)0) \| if (condval == (bool)1)` |
| store.global#2 | 0.05 | 1.9% | 1.0 | 0.06 | `if (condval_1 == (bool)0) \| if (condval_1 == (bool)1)` |

### indexer_mxfp4_quantize_kernel grid (2, 1, 1) block (128, 1, 1)

34 launches, 8 warps; ranges, warp and last warp from the 2 recording CTAs' 8 warps; span 3.74 us, warp 1.66 us, nsys graphs 2.56 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.95 | 56.9% | 2.20 | 57.2% |
| unattributed | 0.45 | 27.0% | 0.44 | 11.4% |
| store | 0.27 | 16.1% | 1.20 | 31.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.69 | 41.5% | 0.8 | 1.09 | `for (int i = 0; i < 32; ++i)` |
| load.global#2*16 (est.: sampled iterations x16) | 0.26 | 15.4% | 0.2 | 1.02 | `float v__2 = __fmul_rn(((float)X[(((((int64_t)((int)blockIdx.x)) * (int64_t)4096) + (((int` |
| store.global*16 (est.: sampled iterations x16) | 0.26 | 15.4% | 0.2 | 1.02 | `Pool[(((((slot & (int64_t)127) * (int64_t)64) + ((((int64_t)((int)threadIdx.x)) & (int64_t` |
| store.global#2 | 0.01 | 0.7% | 0.2 | 0.06 | `Pool[(((((slot & (int64_t)127) * (int64_t)4) + ((slot >> (int64_t)7) * ((int64_t)pool_stri` |

### indexer_mxfp4_quantize_kernel grid (48, 1, 1) block (128, 1, 1)

34 launches, 192 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 3.84 us, warp 3.72 us, nsys graphs 3.01 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 2.35 | 63.0% | 2.39 | 62.8% |
| store | 1.13 | 30.4% | 1.13 | 29.6% |
| unattributed | 0.24 | 6.6% | 0.29 | 7.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2*16 (est.: sampled iterations x16) | 1.30 | 34.8% | 1.0 | 1.54 | `float v__2 = __fmul_rn(((float)X[(((((int64_t)((int)blockIdx.x)) * (int64_t)4096) + (((int` |
| store.global*16 (est.: sampled iterations x16) | 1.09 | 29.2% | 1.0 | 1.54 | `Pool[(((((slot & (int64_t)127) * (int64_t)64) + ((((int64_t)((int)threadIdx.x)) & (int64_t` |
| load.global | 1.05 | 28.2% | 1.0 | 1.12 | `for (int i = 0; i < 32; ++i)` |
| store.global#2 | 0.04 | 1.2% | 1.0 | 0.06 | `Pool[(((((slot & (int64_t)127) * (int64_t)4) + ((slot >> (int64_t)7) * ((int64_t)pool_stri` |

### metadata_chunk_kernel grid (40, 1, 1) block (192, 1, 1)

51 launches, 240 warps; ranges, warp and last warp from the 4 recording CTAs' 24 warps; span 2.18 us, warp 1.52 us, nsys graphs 2.40 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.09 | 71.6% | 1.63 | 69.7% |
| unattributed | 0.37 | 24.4% | 0.57 | 24.5% |
| store | 0.06 | 4.0% | 0.14 | 5.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.05 | 69.0% | 1.0 | 1.34 | `if ((((offset + ((int)blockIdx.x)) < src_rows) && (0 <= (offset + ((int)blockIdx.x)))))` |
| store.global | 0.05 | 3.2% | 1.0 | 0.06 | `swa[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = con` |
| load.global#2 | 0.04 | 2.6% | 0.2 | 0.26 | `if (((0 <= req) & ((int64_t)0 <= pos)))` |
| store.global#2 | 0.01 | 0.5% | 0.2 | 0.06 | `lengths[((int64_t)((int)blockIdx.x))] = ((int)max(info[1], (int64_t)0));` |
| store.global#3 | 0.01 | 0.4% | 0.2 | 0.03 | `top_lengths[((int64_t)((int)blockIdx.x))] = condval_12; \| top_lengths[((int64_t)((int)bloc` |

### gemm_bf16_partials_kernel grid (1, 40, 1) block (256, 1, 1)

34 launches, 320 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 3.52 us, warp 2.09 us, nsys graphs 2.99 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.41 | 67.6% | 2.43 | 69.9% |
| compute | 0.29 | 13.8% | 0.59 | 16.9% |
| wait | 0.24 | 11.7% | 0.32 | 9.2% |
| store | 0.07 | 3.5% | 0.14 | 4.0% |
| load | 0.07 | 3.4% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*2 (est.: sampled iterations x2) | 0.29 | 13.8% | 0.5 | 0.64 | `{` |
| wait.sync | 0.11 | 5.1% | 1.0 | 0.13 | `__syncthreads();` |
| wait.mbarrier#2*2 (est.: sampled iterations x2) | 0.07 | 3.6% | 0.5 | 0.26 | `mbarrier[kb_1].wait(0);` |
| store.global | 0.07 | 3.5% | 0.5 | 0.16 | `for (int i_1 = 0; i_1 < 4; ++i_1) \| for (int i_2 = 0; i_2 < 8; ++i_2) \| for (int i_4 = 0; ` |
| load.tma*2 (est.: sampled iterations x2) | 0.07 | 3.4% | 0.5 | 0.19 | `if (tl::tl_shuffle_elect<128>())` |
| wait.mbarrier*2 (est.: sampled iterations x2) | 0.06 | 3.0% | 0.5 | 0.13 | `mbarrier[(kb + 3)].wait(1);` |

### resample_kernel grid (8, 127, 1) block (128, 1, 1)

17 launches, 4064 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 6.72 us, warp 2.99 us, nsys graphs 6.46 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.74 | 58.2% | 1.51 | 57.0% |
| unattributed | 0.77 | 25.7% | 0.63 | 23.9% |
| compute | 0.31 | 10.3% | 0.30 | 11.2% |
| wait | 0.10 | 3.2% | 0.11 | 4.2% |
| store | 0.08 | 2.6% | 0.10 | 3.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.74 | 58.2% | 1.0 | 2.50 | `end = cu_num_logits[(((int64_t)((int)blockIdx.x)) + (int64_t)1)];` |
| compute.reduce#2 | 0.15 | 5.0% | 1.0 | 0.19 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 8, 32);` |
| wait.sync | 0.10 | 3.2% | 1.0 | 0.19 | `__syncthreads();` |
| compute.reduce#3 | 0.08 | 2.7% | 1.0 | 0.10 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 2, 32);` |
| store.global | 0.08 | 2.6% | 1.0 | 0.13 | `if (((int)threadIdx.x) == 0)` |
| compute.reduce | 0.08 | 2.5% | 1.0 | 0.10 | `if ((temp != 0x0p+0f/*0.000000e+00*/) \| bonus)` |

### prepare_dflash_inputs_kernel grid (9, 8, 1) block (256, 1, 1)

51 launches, 576 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 2.11 us, warp 1.88 us, nsys graphs 2.48 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.41 | 74.9% | 1.64 | 78.7% |
| unattributed | 0.41 | 21.7% | 0.38 | 18.5% |
| store | 0.06 | 3.4% | 0.06 | 2.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.35 | 71.7% | 1.0 | 1.57 | `req = idx_mapping[((int)blockIdx.y)];` |
| store.global#2 | 0.06 | 3.4% | 1.0 | 0.10 | `if (((int)blockIdx.y) == (num_reqs - 1))` |
| load.global#2 | 0.06 | 3.2% | 1.0 | 0.10 | `vllm_store_if_b32((&(out_query_start_loc[((int)blockIdx.y)])), (((int)blockIdx.y) * num_qu` |

### activation_swiglu_clamp_kernel grid (1, 48, 1) block (128, 1, 1)

102 launches, 192 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 0.99 us, warp 0.74 us, nsys graphs 1.95 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.64 | 85.7% | 0.85 | 83.2% |
| load | 0.07 | 9.7% | 0.12 | 11.3% |
| store | 0.03 | 4.6% | 0.06 | 5.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.07 | 9.7% | 0.8 | 0.10 | `*(uint4*)(gate + 0) = *(uint4*)(X + ((((int64_t)((int)blockIdx.y)) * (int64_t)1152) + (((i` |
| store.global | 0.03 | 4.6% | 0.8 | 0.06 | `*(uint4*)(Y + ((((int64_t)((int)blockIdx.y)) * (int64_t)576) + (((int64_t)((int)threadIdx.` |

### gemm_bf16_partials_kernel grid (2, 4, 1) block (256, 1, 1)

34 launches, 64 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 2.53 us, warp 1.64 us, nsys graphs 2.42 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.95 | 57.7% | 1.52 | 50.5% |
| compute | 0.29 | 17.5% | 0.58 | 19.2% |
| wait | 0.23 | 14.1% | 0.61 | 20.2% |
| store | 0.11 | 6.5% | 0.30 | 10.1% |
| load | 0.07 | 4.1% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*2 (est.: sampled iterations x2) | 0.29 | 17.5% | 0.5 | 0.64 | `{` |
| wait.sync | 0.11 | 6.7% | 1.0 | 0.13 | `__syncthreads();` |
| store.global | 0.11 | 6.5% | 0.5 | 0.24 | `for (int i_1 = 0; i_1 < 4; ++i_1) \| for (int i_2 = 0; i_2 < 8; ++i_2) \| for (int i_4 = 0; ` |
| load.tma*2 (est.: sampled iterations x2) | 0.07 | 4.1% | 0.5 | 0.19 | `if (tl::tl_shuffle_elect<128>())` |
| wait.mbarrier*2 (est.: sampled iterations x2) | 0.06 | 3.8% | 0.5 | 0.13 | `mbarrier[(kb + 3)].wait(1);` |
| wait.mbarrier#2*2 (est.: sampled iterations x2) | 0.06 | 3.7% | 0.5 | 0.13 | `mbarrier[kb_1].wait(0);` |

### local_stats_kernel grid (48, 16, 1) block (128, 1, 1)

1 launches, 3072 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 95.90 us, warp 42.56 us, nsys graphs 120.58 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 39.33 | 92.4% | 53.76 | 96.8% |
| unattributed | 1.95 | 4.6% | 0.86 | 1.6% |
| compute | 0.64 | 1.5% | 0.16 | 0.3% |
| wait | 0.57 | 1.3% | 0.67 | 1.2% |
| store | 0.07 | 0.2% | 0.10 | 0.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 39.33 | 92.4% | 1.0 | 54.56 | `if (expanded_local_pos[((int64_t)((int)blockIdx.x))] < num_speculative_steps)` |
| compute.reduce | 0.58 | 1.4% | 1.0 | 1.54 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| wait.sync | 0.57 | 1.3% | 1.0 | 3.23 | `__syncthreads();` |
| store.global | 0.07 | 0.2% | 1.0 | 0.10 | `if (((int)threadIdx.x) == 0)` |
| compute.reduce#2 | 0.06 | 0.2% | 1.0 | 0.06 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 2, 32);` |

### norm_forward_and_per_token_cast_kernel_kernel grid (3, 1, 1) block (128, 1, 1)

34 launches, 12 warps; ranges, warp and last warp from the 3 recording CTAs' 12 warps; span 2.18 us, warp 2.16 us, nsys graphs 1.89 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 1.12 | 51.9% | 1.35 | 53.3% |
| unattributed | 0.67 | 31.1% | 0.74 | 29.2% |
| load | 0.23 | 10.7% | 0.30 | 12.1% |
| compute | 0.09 | 3.9% | 0.09 | 3.5% |
| wait | 0.05 | 2.3% | 0.05 | 1.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*2 (est.: sampled iterations x2) | 0.72 | 33.4% | 1.0 | 0.77 | `for (int j_2 = 0; j_2 < 2; ++j_2)` |
| store.global*2 (est.: sampled iterations x2) | 0.21 | 9.6% | 1.0 | 0.26 | `for (int j_1 = 0; j_1 < 4; ++j_1)` |
| load.global*2 (est.: sampled iterations x2) | 0.12 | 5.4% | 1.0 | 0.13 | `if (((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < num_tokens)` |
| store.global#5*2 (est.: sampled iterations x2) | 0.10 | 4.4% | 1.0 | 0.13 | `*(uint4*)(out + ((((((int64_t)vi_1) * (int64_t)64) + ((((int64_t)((int)threadIdx.x)) & (in` |
| compute.reduce | 0.09 | 3.9% | 1.0 | 0.10 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| load.global#3*2 (est.: sampled iterations x2) | 0.08 | 3.8% | 1.0 | 0.13 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 64) + ((((int)threadIdx.x) & 7) ` |
| store.global#3 | 0.05 | 2.5% | 1.0 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |
| wait.sync | 0.05 | 2.3% | 1.0 | 0.06 | `__syncthreads();` |
| store.global#2 | 0.04 | 2.0% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| load.global#2 | 0.03 | 1.5% | 1.0 | 0.03 | `*(float2*)(row_partial + 0) = *(float2*)(sum_shared + ((((int)threadIdx.x) >> 6) * 2)); \| ` |

### verify_kernel grid (8, 1, 1) block (96, 1, 1)

17 launches, 24 warps; ranges, warp and last warp from the 4 recording CTAs' 12 warps; span 2.59 us, warp 2.49 us, nsys graphs 2.18 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.06 | 42.7% | 2.00 | 41.8% |
| unattributed | 0.87 | 34.9% | 2.14 | 44.7% |
| compute | 0.30 | 12.1% | 0.29 | 6.1% |
| wait | 0.16 | 6.4% | 0.18 | 3.9% |
| store | 0.10 | 3.9% | 0.17 | 3.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.00 | 40.1% | 1.0 | 1.02 | `start = cu_num_logits[((int64_t)((int)blockIdx.x))];` |
| compute.reduce | 0.30 | 12.1% | 1.0 | 0.32 | `row_max[0] = vllm_max(row_max[0], __shfl_xor_sync((uint)4294967295, row_max[0], 8, 32));` |
| store.global | 0.10 | 3.9% | 1.0 | 0.16 | `if (((int)threadIdx.x) == 0)` |
| load.global#2 | 0.06 | 2.6% | 1.0 | 0.06 | `target = local_argmax[((((int64_t)row) * argmax_stride) + ((int64_t)best_index[0]))];` |
| wait.sync#2 | 0.06 | 2.6% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync | 0.05 | 1.9% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#3 | 0.05 | 1.9% | 1.0 | 0.06 | `__syncthreads();` |

### gemm_splitk_reduce_cast_kernel grid (16, 10, 1) block (128, 1, 1)

51 launches, 640 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 1.50 us, warp 1.26 us, nsys graphs 1.84 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.68 | 54.0% | 0.95 | 58.3% |
| unattributed | 0.40 | 32.0% | 0.50 | 30.7% |
| store | 0.09 | 7.1% | 0.09 | 5.4% |
| compute | 0.09 | 7.0% | 0.09 | 5.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.68 | 54.0% | 1.0 | 0.90 | `if ((1 <= M))` |
| store.global | 0.09 | 7.1% | 1.0 | 0.10 | `if (((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 5)) < M)` |
| compute.reduce | 0.09 | 7.0% | 1.0 | 0.10 | `amax[0] = tl::AllReduce<tl::MaxOp, 8, 1, 0, tl::NamedBarrier<128>>::run(amax[0]);` |

### per_token_cast_kernel_kernel grid (25, 1, 1) block (64, 1, 1)

51 launches, 50 warps; ranges, warp and last warp from the 5 recording CTAs' 10 warps; span 1.15 us, warp 1.10 us, nsys graphs 1.49 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.36 | 32.8% | 0.38 | 23.6% |
| load | 0.35 | 32.0% | 0.82 | 51.1% |
| store | 0.29 | 26.5% | 0.29 | 18.4% |
| compute | 0.10 | 8.7% | 0.11 | 6.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.26 | 23.8% | 1.0 | 0.29 | `for (int i = 0; i < 2; ++i)` |
| store.global#2 | 0.11 | 10.2% | 1.0 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.10 | 8.7% | 1.0 | 0.13 | `amax_fragment[i_1] = tl::AllReduce<tl::MaxOp, 2, 1, 0, tl::NamedBarrier<64>>::run(amax_fra` |
| load.global#2*2 (est.: sampled iterations x2) | 0.09 | 8.1% | 1.0 | 0.13 | `amax_fragment[i_1] = cutlass::bfloat16_t(__hmax((bfloat16_t(((nv_bfloat162*)(&(amax_fragme` |
| store.global#3*2 (est.: sampled iterations x2) | 0.09 | 8.1% | 1.0 | 0.13 | `*(fp8_e4_16_t*)(out + ((((((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int64_t)10240) +` |
| store.global*2 (est.: sampled iterations x2) | 0.09 | 8.1% | 1.0 | 0.13 | `out_sf[((((((int64_t)((int)blockIdx.x)) % (int64_t)15) * (int64_t)32) + (((int64_t)((int)t` |

### topk_log_softmax_kernel grid (48, 1, 1) block (1024, 1, 1)

1 launches, 1536 warps; ranges, warp and last warp from the 4 recording CTAs' 128 warps; span 75.55 us, warp 68.06 us, nsys graphs 107.14 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 58.19 | 85.5% | 72.54 | 96.6% |
| wait | 7.88 | 11.6% | 0.35 | 0.5% |
| unattributed, with loops' unsampled iterations | 1.60 | 2.4% | 1.79 | 2.4% |
| compute | 0.38 | 0.6% | 0.38 | 0.5% |
| store | 0.00 | 0.0% | 0.06 | 0.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 38.58 | 56.7% | 1.0 | 45.76 | `if (0 < (vocab & 2047)) \| if (0 < (vocab & 8191))` |
| load.global#2 | 19.53 | 28.7% | 1.0 | 26.02 | `for (int step_1 = 0; step_1 < (vocab >> 11); ++step_1) \| for (int step_1 = 0; step_1 < (vo` |
| wait.sync | 5.01 | 7.4% | 1.0 | 18.62 | `__syncthreads();` |
| wait.sync#2 | 2.61 | 3.8% | 1.0 | 9.47 | `__syncthreads();` |
| compute.reduce#3 | 0.18 | 0.3% | 1.0 | 0.22 | `ties[0] = vllm_add(ties[0], __shfl_xor_sync((uint)4294967295, ties[0], 16, 32));` |
| compute.reduce#2 | 0.10 | 0.1% | 1.0 | 0.13 | `se[0] = vllm_add(se[0], __shfl_xor_sync((uint)4294967295, se[0], 16, 32));` |
| wait.sync#4 | 0.10 | 0.1% | 1.0 | 0.16 | `__syncthreads();` |
| compute.reduce | 0.09 | 0.1% | 1.0 | 0.10 | `best[0] = vllm_max(best[0], __shfl_xor_sync((uint)4294967295, best[0], 16, 32));` |
| load.global#3 (sampled iterations) | 0.09 | 0.1% | 0.0 | 7.97 | `float value_32 = vllm_sub(vllm_shifted(logits[((((int64_t)((int)blockIdx.x)) * logits_stri` |
| wait.sync#3 | 0.08 | 0.1% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#5 | 0.08 | 0.1% | 1.0 | 0.10 | `__syncthreads();` |
| store.global (sampled iterations) | 0.00 | 0.0% | 0.0 | 0.06 | `output[(((((int64_t)c) * (int64_t)1024) + (((int64_t)((int)blockIdx.x)) * ((int64_t)topk))` |

### activation_swiglu_clamp_kernel grid (1, 40, 1) block (128, 1, 1)

51 launches, 160 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 0.96 us, warp 0.74 us, nsys graphs 1.55 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.64 | 85.5% | 1.22 | 83.8% |
| load | 0.07 | 9.7% | 0.16 | 11.2% |
| store | 0.04 | 4.8% | 0.07 | 5.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.07 | 9.7% | 0.8 | 0.10 | `*(uint4*)(gate + 0) = *(uint4*)(X + ((((int64_t)((int)blockIdx.y)) * (int64_t)1152) + (((i` |
| store.global | 0.04 | 4.8% | 0.8 | 0.06 | `*(uint4*)(Y + ((((int64_t)((int)blockIdx.y)) * (int64_t)576) + (((int64_t)((int)threadIdx.` |

### sink_merge_kernel grid (8, 1, 1) block (128, 1, 1)

17 launches, 32 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 4.10 us, warp 4.06 us, nsys graphs 4.40 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.92 | 47.4% | 2.07 | 47.9% |
| wait | 0.84 | 20.7% | 0.43 | 10.0% |
| unattributed | 0.70 | 17.3% | 1.14 | 26.2% |
| compute | 0.56 | 13.9% | 0.58 | 13.3% |
| store | 0.03 | 0.7% | 0.11 | 2.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2*3 (est.: sampled iterations x3) | 1.64 | 40.4% | 1.0 | 1.73 | `for (int step = 0; step < ((width + 127) >> 7); ++step) \| for (int step = 0; step < ((widt` |
| compute.reduce*3 (est.: sampled iterations x3) | 0.56 | 13.9% | 1.0 | 0.58 | `float other = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| wait.sync#3*3 (est.: sampled iterations x3) | 0.52 | 12.9% | 1.0 | 0.67 | `__syncthreads();` |
| load.global | 0.24 | 5.8% | 1.0 | 0.26 | `last = sampled[(max((((int64_t)num_sampled[((int64_t)((int)blockIdx.x))]) - (int64_t)1), (` |
| wait.sync*3 (est.: sampled iterations x3) | 0.17 | 4.3% | 1.0 | 0.19 | `__syncthreads();` |
| wait.sync#2*3 (est.: sampled iterations x3) | 0.14 | 3.6% | 1.0 | 0.19 | `__syncthreads();` |
| load.global#3*3 (est.: sampled iterations x3) | 0.05 | 1.2% | 0.2 | 0.19 | `if ((best[0] == -CUDART_INF_F))` |
| store.global*3 (est.: sampled iterations x3) | 0.03 | 0.7% | 0.2 | 0.19 | `alternates[((((int64_t)((int)blockIdx.x)) * (int64_t)3) + ((int64_t)rank))] = condval_31;` |

### finish_argmax_kernel grid (8, 1, 1) block (32, 1, 1)

85 launches, 8 warps; ranges, warp and last warp from the 4 recording CTAs' 4 warps; span 0.77 us, warp 0.74 us, nsys graphs 1.06 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.38 | 52.2% | 0.46 | 57.1% |
| load | 0.13 | 17.4% | 0.12 | 15.5% |
| compute | 0.12 | 16.3% | 0.12 | 15.0% |
| store | 0.10 | 14.1% | 0.10 | 12.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.13 | 17.4% | 1.0 | 0.13 | `for (int k = 0; k < 2; ++k) \| if ((((int)threadIdx.x) < parts))` |
| compute.reduce | 0.12 | 16.3% | 1.0 | 0.13 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| store.global | 0.10 | 14.1% | 1.0 | 0.13 | `if (((int)threadIdx.x) == 0)` |

### mhc_collapse_column_kernel grid (240, 1, 1) block (128, 1, 1)

17 launches, 960 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 3.97 us, warp 1.75 us, nsys graphs 2.18 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.41 | 80.5% | 1.63 | 61.6% |
| load | 0.26 | 15.1% | 0.96 | 36.2% |
| store | 0.08 | 4.5% | 0.06 | 2.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.26 | 15.1% | 1.0 | 0.83 | `*(float4*)(mv + 0) = *(float4*)(mix + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int6` |
| store.global | 0.08 | 4.5% | 1.0 | 0.26 | `*(uint4*)(out + ((((int64_t)((int)blockIdx.x)) * (int64_t)1024) + (((int64_t)((int)threadI` |

### cache_writer_kernel grid (40, 1, 1) block (128, 1, 1)

51 launches, 160 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 1.09 us, warp 1.07 us, nsys graphs 1.50 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.41 | 38.1% | 0.42 | 39.1% |
| store | 0.30 | 28.0% | 0.28 | 26.1% |
| load | 0.29 | 26.9% | 0.29 | 27.2% |
| compute | 0.08 | 7.1% | 0.08 | 7.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.30 | 28.0% | 1.0 | 0.32 | `if (((int64_t)0 <= slot) & ((slot >> (int64_t)7) < ((int64_t)pages))) \| if (((int64_t)0 <=` |
| load.global | 0.23 | 21.5% | 1.0 | 0.26 | `for (int i = 0; i < 4; ++i)` |
| compute.reduce | 0.08 | 7.1% | 1.0 | 0.10 | `amax = max(amax, __shfl_xor_sync((uint)4294967295, amax, 1, 32));` |
| load.global#2 | 0.06 | 5.4% | 1.0 | 0.06 | `int64_t slot = slots[((int64_t)((int)blockIdx.x))];` |

### rope_kernel grid (40, 1, 1) block (128, 1, 1)

51 launches, 160 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 1.28 us, warp 1.22 us, nsys graphs 1.73 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.71 | 57.7% | 0.77 | 58.6% |
| load | 0.48 | 39.2% | 0.50 | 38.5% |
| store | 0.04 | 3.1% | 0.04 | 2.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.48 | 39.2% | 1.0 | 0.48 | `if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < tokens)) \| int64_t position ` |
| store.global | 0.04 | 3.1% | 1.0 | 0.06 | `*(uint4*)(x + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + ((((int64_t)((int)thread` |

### gemm_splitk_reduce_kernel grid (1, 48, 1) block (128, 1, 1)

68 launches, 192 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 0.86 us, warp 0.83 us, nsys graphs 1.02 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.47 | 56.6% | 0.53 | 58.3% |
| load | 0.31 | 37.4% | 0.33 | 36.0% |
| store | 0.05 | 6.0% | 0.05 | 5.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.31 | 37.4% | 1.0 | 0.34 | `acc[0] = P[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x))` |
| store.global | 0.05 | 6.0% | 1.0 | 0.06 | `C[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x)) * (int64` |

### mhc_collapse_kernel grid (200, 1, 1) block (128, 1, 1)

17 launches, 800 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 3.58 us, warp 2.29 us, nsys graphs 2.37 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 2.01 | 88.0% | 2.23 | 74.4% |
| load | 0.22 | 9.7% | 0.71 | 23.8% |
| store | 0.05 | 2.3% | 0.05 | 1.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.22 | 9.7% | 1.0 | 0.58 | `*(float4*)(mv + 0) = *(float4*)(mix + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int6` |
| store.global | 0.05 | 2.3% | 1.0 | 0.06 | `*(uint4*)(out + ((((int64_t)((int)blockIdx.x)) * (int64_t)1024) + (((int64_t)((int)threadI` |

### context_inputs_kernel grid (5, 16, 1) block (128, 1, 1)

16 launches, 320 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 3.65 us, warp 2.17 us, nsys graphs 4.18 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.76 | 81.2% | 3.03 | 80.5% |
| unattributed | 0.34 | 15.7% | 0.53 | 14.1% |
| store | 0.07 | 3.1% | 0.20 | 5.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.76 | 81.2% | 1.0 | 3.09 | `vllm_dspark_context_b128((&(aux[(((((int64_t)((int)blockIdx.y)) * (int64_t)15360) + (((int` |
| store.global | 0.07 | 3.1% | 1.0 | 0.10 | `if (((((int)blockIdx.x) == 0) & (((int)threadIdx.x) == 0)) & (num_tokens <= ((int)blockIdx` |

### dspark_confidence_kernel grid (40, 1, 1) block (128, 1, 1)

17 launches, 160 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 3.39 us, warp 3.19 us, nsys graphs 3.46 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 2.77 | 86.9% | 2.98 | 86.2% |
| load | 0.17 | 5.5% | 0.21 | 6.0% |
| wait | 0.12 | 3.8% | 0.09 | 2.5% |
| compute | 0.10 | 3.0% | 0.09 | 2.7% |
| store | 0.02 | 0.8% | 0.09 | 2.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.16 | 5.1% | 1.0 | 0.19 | `for (int c = 0; c < 5; ++c)` |
| wait.sync | 0.12 | 3.8% | 1.0 | 0.35 | `__syncthreads();` |
| compute.reduce | 0.10 | 3.0% | 1.0 | 0.10 | `acc[0] = tl::warp_reduce_sum(acc[0]);` |
| store.global | 0.02 | 0.8% | 0.2 | 0.10 | `out[((int64_t)((int)blockIdx.x))] = (0x1p+0f/*1.000000e+00*/ / (0x1p+0f/*1.000000e+00*/ + ` |
| load.global#2 | 0.01 | 0.4% | 0.2 | 0.06 | `total[0] = 0x0p+0f/*0.000000e+00*/;` |

### norm_forward_and_per_token_cast_kernel_kernel grid (48, 1, 1) block (128, 1, 1)

18 launches, 192 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 3.10 us, warp 2.84 us, nsys graphs 2.42 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 1.43 | 50.5% | 2.62 | 81.5% |
| unattributed | 0.77 | 27.0% | 0.00 | 0.0% |
| load | 0.45 | 15.8% | 0.46 | 14.4% |
| compute | 0.13 | 4.6% | 0.13 | 4.0% |
| wait | 0.06 | 2.1% | 0.06 | 1.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*5 (est.: sampled iterations x5) | 1.17 | 41.2% | 1.0 | 1.92 | `for (int j_1 = 0; j_1 < 2; ++j_1)` |
| load.global#2*5 (est.: sampled iterations x5) | 0.23 | 7.9% | 1.0 | 0.32 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 1024) + (((int)threadIdx.x) * 8)` |
| load.global*5 (est.: sampled iterations x5) | 0.23 | 7.9% | 1.0 | 0.32 | `*(uint4*)(x_local + (vi * 8)) = *(uint4*)(x + ((((((int64_t)vi) * (int64_t)1024) + (((int6` |
| store.global*5 (est.: sampled iterations x5) | 0.21 | 7.4% | 1.0 | 0.32 | `for (int j = 0; j < 4; ++j)` |
| compute.reduce | 0.13 | 4.6% | 1.0 | 0.16 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| wait.sync | 0.06 | 2.1% | 1.0 | 0.10 | `__syncthreads();` |
| store.global#2 | 0.04 | 1.5% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| store.global#3 | 0.01 | 0.4% | 0.2 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |

### norm_forward_and_per_token_cast_kernel_kernel grid (40, 1, 1) block (128, 1, 1)

17 launches, 160 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 3.33 us, warp 3.12 us, nsys graphs 1.94 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 1.51 | 48.5% | 2.07 | 62.6% |
| unattributed | 0.97 | 31.2% | 0.55 | 16.6% |
| load | 0.44 | 14.1% | 0.45 | 13.7% |
| compute | 0.13 | 4.2% | 0.13 | 3.9% |
| wait | 0.06 | 2.1% | 0.11 | 3.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*5 (est.: sampled iterations x5) | 1.24 | 39.7% | 1.0 | 1.76 | `for (int j_1 = 0; j_1 < 2; ++j_1)` |
| load.global#2*5 (est.: sampled iterations x5) | 0.22 | 7.1% | 1.0 | 0.32 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 1024) + (((int)threadIdx.x) * 8)` |
| load.global*5 (est.: sampled iterations x5) | 0.22 | 7.1% | 1.0 | 0.32 | `*(uint4*)(x_local + (vi * 8)) = *(uint4*)(x + ((((((int64_t)vi) * (int64_t)1024) + (((int6` |
| store.global*5 (est.: sampled iterations x5) | 0.22 | 7.1% | 1.0 | 0.32 | `for (int j = 0; j < 4; ++j)` |
| compute.reduce | 0.13 | 4.2% | 1.0 | 0.16 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| wait.sync | 0.06 | 2.1% | 1.0 | 0.13 | `__syncthreads();` |
| store.global#2 | 0.04 | 1.3% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| store.global#3 | 0.01 | 0.4% | 0.2 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |

### adaptive_offsets_kernel grid (1, 1, 1) block (256, 1, 1)

16 launches, 8 warps; ranges, warp and last warp from the 1 recording CTAs' 8 warps; span 3.52 us, warp 3.51 us, nsys graphs 1.82 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 1.57 | 44.7% | 1.13 | 32.1% |
| unattributed, with loops' unsampled iterations | 1.46 | 41.6% | 1.52 | 43.2% |
| store | 0.30 | 8.5% | 0.49 | 13.8% |
| load | 0.18 | 5.1% | 0.38 | 10.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync#18 (sampled iterations) | 0.34 | 9.8% | 1.0 | 0.38 | `__syncthreads();` |
| wait.sync (sampled iterations) | 0.20 | 5.8% | 1.0 | 0.22 | `__syncthreads();` |
| store.global#5 | 0.12 | 3.5% | 1.0 | 0.13 | `for (int chunk_1 = 0; chunk_1 < ((starts_len + 255) >> 8); ++chunk_1)` |
| load.global#2 (sampled iterations) | 0.09 | 2.6% | 1.0 | 0.22 | `if ((((chunk * 256) + ((int)threadIdx.x)) < num_reqs))` |
| store.global#2 (sampled iterations) | 0.08 | 2.4% | 1.0 | 0.10 | `if ((((chunk * 256) + ((int)threadIdx.x)) < num_reqs) & (drafted == 0))` |
| load.global (sampled iterations) | 0.08 | 2.3% | 1.0 | 0.10 | `if (((((chunk * 256) + ((int)threadIdx.x)) < num_reqs) & (drafted != 0)))` |
| store.global | 0.07 | 2.0% | 1.0 | 0.10 | `if (((int)threadIdx.x) == 0)` |
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

### per_token_cast_kernel_kernel grid (40, 1, 1) block (64, 1, 1)

48 launches, 80 warps; ranges, warp and last warp from the 4 recording CTAs' 8 warps; span 1.12 us, warp 1.08 us, nsys graphs 1.01 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.38 | 35.7% | 0.40 | 36.4% |
| unattributed | 0.30 | 27.9% | 0.28 | 25.8% |
| store | 0.30 | 27.5% | 0.32 | 29.4% |
| compute | 0.10 | 8.9% | 0.09 | 8.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.30 | 27.5% | 1.0 | 0.32 | `for (int i = 0; i < 2; ++i)` |
| store.global#2 | 0.11 | 10.4% | 1.0 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.10 | 8.9% | 1.0 | 0.13 | `amax_fragment[i_1] = tl::AllReduce<tl::MaxOp, 2, 1, 0, tl::NamedBarrier<64>>::run(amax_fra` |
| store.global*2 (est.: sampled iterations x2) | 0.10 | 8.9% | 1.0 | 0.13 | `out_sf[((((((int64_t)((int)blockIdx.x)) % (int64_t)15) * (int64_t)32) + (((int64_t)((int)t` |
| load.global#2*2 (est.: sampled iterations x2) | 0.09 | 8.2% | 1.0 | 0.13 | `amax_fragment[i_1] = cutlass::bfloat16_t(__hmax((bfloat16_t(((nv_bfloat162*)(&(amax_fragme` |
| store.global#3*2 (est.: sampled iterations x2) | 0.09 | 8.2% | 1.0 | 0.13 | `*(fp8_e4_16_t*)(out + ((((((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int64_t)10240) +` |

### sink_cut_kernel grid (1, 1, 1) block (128, 1, 1)

34 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 1.79 us, warp 0.60 us, nsys graphs 2.14 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.32 | 53.3% | 0.45 | 28.6% |
| store | 0.16 | 26.7% | 0.56 | 35.6% |
| load | 0.12 | 20.0% | 0.57 | 35.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.16 | 26.7% | 0.8 | 0.51 | `if (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) < num_cut) \| saved[((((int64_t)((int)` |
| load.global | 0.12 | 20.0% | 0.1 | 0.48 | `for (int b = 0; b < num_reqs; ++b)` |

### token_ranks_kernel grid (48, 1, 1) block (256, 1, 1)

1 launches, 384 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 48.99 us, warp 44.50 us, nsys graphs 89.15 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 35.83 | 80.5% | 45.02 | 94.1% |
| compute | 8.25 | 18.5% | 2.40 | 5.0% |
| unattributed | 0.37 | 0.8% | 0.35 | 0.7% |
| store | 0.06 | 0.1% | 0.06 | 0.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 35.83 | 80.5% | 1.0 | 47.23 | `if ((((int64_t)0 <= token_ids[((int64_t)((int)blockIdx.x))]) && (token_ids[((int64_t)((int` |
| compute.reduce | 8.25 | 18.5% | 1.0 | 28.48 | `total[0] = tl::AllReduce<tl::SumOp, 256, 1, 0, tl::NamedBarrier<256>>::run(total[0], (&(wo` |
| store.global | 0.06 | 0.1% | 1.0 | 0.06 | `output[((int64_t)((int)blockIdx.x))] = ((int64_t)total[0]);` |

### cache_writer_kernel grid (16, 1, 1) block (128, 1, 1)

48 launches, 64 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 0.99 us, warp 0.98 us, nsys graphs 1.12 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.33 | 33.7% | 0.33 | 33.4% |
| store | 0.29 | 29.5% | 0.29 | 29.3% |
| load | 0.29 | 29.1% | 0.29 | 29.6% |
| compute | 0.08 | 7.7% | 0.08 | 7.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.29 | 29.5% | 1.0 | 0.32 | `if (((int64_t)0 <= slot) & ((slot >> (int64_t)7) < ((int64_t)pages))) \| if (((int64_t)0 <=` |
| load.global | 0.23 | 23.2% | 1.0 | 0.26 | `for (int i = 0; i < 4; ++i)` |
| compute.reduce | 0.08 | 7.7% | 1.0 | 0.10 | `amax = max(amax, __shfl_xor_sync((uint)4294967295, amax, 1, 32));` |
| load.global#2 | 0.06 | 5.9% | 1.0 | 0.06 | `int64_t slot = slots[((int64_t)((int)blockIdx.x))];` |

### metadata_pages_kernel grid (48, 1, 1) block (128, 1, 1)

34 launches, 192 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 1.28 us, warp 0.86 us, nsys graphs 0.00 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.68 | 79.7% | 0.76 | 59.7% |
| load | 0.14 | 15.9% | 0.19 | 15.1% |
| store | 0.04 | 4.4% | 0.32 | 25.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.13 | 15.0% | 1.0 | 0.13 | `if ((((offset + ((int)blockIdx.x)) < src_rows) && (0 <= (offset + ((int)blockIdx.x)))))` |
| store.global | 0.04 | 4.4% | 0.1 | 0.32 | `if (((int)threadIdx.x) < 32)` |
| load.global#2 | 0.01 | 0.9% | 0.1 | 0.06 | `if (((req < table_rows) && (((int)threadIdx.x) < table_cols)))` |

### sink_policy_kernel grid (8, 1, 1) block (32, 1, 1)

17 launches, 8 warps; ranges, warp and last warp from the 4 recording CTAs' 4 warps; span 2.46 us, warp 2.40 us, nsys graphs 2.46 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.94 | 39.3% | 1.02 | 39.9% |
| unattributed | 0.94 | 39.3% | 0.98 | 38.3% |
| store | 0.51 | 21.3% | 0.56 | 21.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 | 0.55 | 23.0% | 1.0 | 0.58 | `state[0] = idx_mapping[((int64_t)((int)blockIdx.x))];` |
| store.global#3 | 0.46 | 19.0% | 1.0 | 0.48 | `for (int a = 0; a < 3; ++a) \| recovered_logprobs[((int64_t)((int)blockIdx.x))] = 0x0p+0f/*` |
| load.global | 0.21 | 8.7% | 1.0 | 0.22 | `start[0] = cu_num_logits[((int64_t)((int)blockIdx.x))];` |
| load.global#4 | 0.14 | 5.7% | 1.0 | 0.16 | `for (int r = 0; r < 2; ++r) \| for (int r_2 = 0; r_2 < 8; ++r_2)` |
| store.global#4 | 0.06 | 2.3% | 1.0 | 0.06 | `events[(((int64_t)((int)blockIdx.x)) * (int64_t)6)] = outcome[0];` |
| load.global#5 | 0.05 | 2.0% | 1.0 | 0.06 | `AtomicAdd((&(counts[0])), draft_zero_flag[0]);` |

### sink_flags_kernel grid (1, 1, 1) block (128, 1, 1)

17 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 1.98 us, warp 0.66 us, nsys graphs 2.85 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.33 | 49.4% | 1.30 | 51.1% |
| unattributed | 0.25 | 37.3% | 0.42 | 16.4% |
| store | 0.09 | 13.3% | 0.82 | 32.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.33 | 49.4% | 0.2 | 1.31 | `token = logits_indices[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)th` |
| store.global | 0.09 | 13.3% | 0.2 | 0.35 | `flags[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = (` |

### norm_forward_and_per_token_cast_kernel_kernel grid (16, 1, 1) block (128, 1, 1)

16 launches, 64 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 2.43 us, warp 2.33 us, nsys graphs 1.92 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 2.11 | 90.4% | 3.42 | 129.8% |
| load | 0.43 | 18.6% | 0.41 | 15.6% |
| compute | 0.13 | 5.5% | 0.13 | 5.0% |
| wait | 0.06 | 2.4% | 0.06 | 2.1% |
| unattributed | 0.00 | 0.0% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*5 (est.: sampled iterations x5) | 1.82 | 78.2% | 1.0 | 2.08 | `for (int j_1 = 0; j_1 < 2; ++j_1)` |
| store.global*5 (est.: sampled iterations x5) | 0.23 | 9.9% | 1.0 | 0.32 | `for (int j = 0; j < 4; ++j)` |
| load.global*5 (est.: sampled iterations x5) | 0.22 | 9.4% | 1.0 | 0.32 | `*(uint4*)(x_local + (vi * 8)) = *(uint4*)(x + ((((((int64_t)vi) * (int64_t)1024) + (((int6` |
| load.global#2*5 (est.: sampled iterations x5) | 0.21 | 9.2% | 1.0 | 0.32 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 1024) + (((int)threadIdx.x) * 8)` |
| compute.reduce | 0.13 | 5.5% | 1.0 | 0.16 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| wait.sync | 0.06 | 2.4% | 1.0 | 0.10 | `__syncthreads();` |
| store.global#2 | 0.04 | 1.7% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| store.global#3 | 0.01 | 0.6% | 0.2 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |

### rope_kernel grid (1, 1, 1) block (128, 1, 1)

48 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 0.67 us, warp 0.67 us, nsys graphs 1.06 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.42 | 61.9% | 0.53 | 61.6% |
| load | 0.22 | 33.3% | 0.30 | 34.2% |
| store | 0.03 | 4.8% | 0.04 | 4.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 | 0.13 | 19.0% | 1.0 | 0.13 | `if (((position < ((int64_t)positions_max)) && ((int64_t)0 <= position)))` |
| load.global | 0.10 | 14.3% | 1.0 | 0.10 | `if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < tokens)) \| int64_t position ` |
| store.global | 0.03 | 4.8% | 1.0 | 0.03 | `*(uint4*)(x + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + ((((int64_t)((int)thread` |

### post_update_kernel grid (1, 1, 1) block (32, 1, 1)

17 launches, 1 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 1.98 us, warp 1.98 us, nsys graphs 2.58 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 0.74 | 37.1% | 1.11 | 47.2% |
| load | 0.70 | 35.5% | 0.68 | 28.9% |
| unattributed | 0.54 | 27.4% | 0.56 | 23.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#2 | 0.58 | 29.0% | 1.0 | 0.58 | `if (0 < count)` |
| load.global#2 | 0.51 | 25.8% | 1.0 | 0.51 | `int count = num_sampled[((((int64_t)((int)blockIdx.x)) * (int64_t)32) + ((int64_t)((int)th` |
| store.global | 0.16 | 8.1% | 1.0 | 0.16 | `if (0 < count)` |
| load.global | 0.10 | 4.8% | 1.0 | 0.10 | `int64_t req = idx_mapping[((((int64_t)((int)blockIdx.x)) * (int64_t)32) + ((int64_t)((int)` |
| load.global#3 | 0.10 | 4.8% | 1.0 | 0.10 | `if ((0 < count))` |

### compute_slot_mappings_kernel grid (1, 11, 1) block (256, 1, 1)

17 launches, 88 warps; ranges, warp and last warp from the 6 recording CTAs' 48 warps; span 2.30 us, warp 1.42 us, nsys graphs 2.80 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.78 | 55.3% | 0.79 | 34.5% |
| unattributed | 0.63 | 44.7% | 1.50 | 65.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.78 | 55.3% | 1.0 | 0.80 | `int first_token = query_start_loc[0];` |

### per_token_cast_kernel_kernel grid (120, 1, 1) block (64, 1, 1)

19 launches, 240 warps; ranges, warp and last warp from the 4 recording CTAs' 8 warps; span 1.89 us, warp 1.69 us, nsys graphs 1.41 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.90 | 53.6% | 1.02 | 54.0% |
| unattributed | 0.37 | 22.0% | 0.35 | 18.6% |
| store | 0.31 | 18.2% | 0.41 | 22.0% |
| compute | 0.10 | 6.2% | 0.10 | 5.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.82 | 48.3% | 1.0 | 1.06 | `for (int i = 0; i < 2; ++i)` |
| store.global#2 | 0.12 | 6.9% | 1.0 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.10 | 6.2% | 1.0 | 0.13 | `amax_fragment[i_1] = tl::AllReduce<tl::MaxOp, 2, 1, 0, tl::NamedBarrier<64>>::run(amax_fra` |
| store.global*2 (est.: sampled iterations x2) | 0.10 | 6.2% | 1.0 | 0.13 | `out_sf[((((((int64_t)((int)blockIdx.x)) % (int64_t)15) * (int64_t)32) + (((int64_t)((int)t` |
| load.global#2*2 (est.: sampled iterations x2) | 0.09 | 5.2% | 1.0 | 0.13 | `amax_fragment[i_1] = cutlass::bfloat16_t(__hmax((bfloat16_t(((nv_bfloat162*)(&(amax_fragme` |
| store.global#3*2 (est.: sampled iterations x2) | 0.09 | 5.2% | 1.0 | 0.13 | `*(fp8_e4_16_t*)(out + ((((((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int64_t)10240) +` |

### gather_block_tables_kernel grid (11, 8, 1) block (256, 1, 1)

17 launches, 704 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 2.05 us, warp 1.34 us, nsys graphs 0.00 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.03 | 76.6% | 1.53 | 82.4% |
| unattributed | 0.31 | 23.4% | 0.33 | 17.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.03 | 76.6% | 1.0 | 1.47 | `int g = chunk_groups[((int64_t)((int)blockIdx.x))];` |

### rope_kernel grid (48, 1, 1) block (256, 1, 1)

34 launches, 384 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 1.01 us, warp 0.89 us, nsys graphs 1.28 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.59 | 66.1% | 0.70 | 69.6% |
| load | 0.26 | 29.4% | 0.26 | 26.2% |
| store | 0.04 | 4.5% | 0.04 | 4.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.26 | 29.4% | 1.0 | 0.29 | `if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < tokens)) \| int64_t position ` |
| store.global | 0.04 | 4.5% | 1.0 | 0.06 | `*(uint4*)(x + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + ((((int64_t)((int)thread` |

### insert_kernel grid (8, 1, 1) block (32, 1, 1)

17 launches, 8 warps; ranges, warp and last warp from the 4 recording CTAs' 4 warps; span 1.76 us, warp 1.74 us, nsys graphs 1.63 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.75 | 43.3% | 0.83 | 44.3% |
| unattributed | 0.49 | 28.1% | 0.49 | 26.4% |
| store | 0.34 | 19.4% | 0.40 | 21.7% |
| compute | 0.16 | 9.2% | 0.14 | 7.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#3 | 0.57 | 32.7% | 1.0 | 0.61 | `int state = mapping[((int64_t)position)]; \| int64_t state = mapping[((int64_t)position)];` |
| store.global | 0.29 | 16.6% | 1.0 | 0.29 | `counts[((int64_t)((int)blockIdx.x))] = (count_local + 1); \| counts[((int64_t)((int)blockId` |
| load.global | 0.12 | 6.9% | 1.0 | 0.13 | `for (int k = 0; k < 4; ++k)` |
| compute.reduce#2 | 0.11 | 6.5% | 1.0 | 0.13 | `float value_1 = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| load.global#2 | 0.06 | 3.7% | 1.0 | 0.06 | `count_local = counts[((int64_t)((int)blockIdx.x))];` |
| compute.reduce | 0.05 | 2.8% | 1.0 | 0.06 | `int count = __shfl_sync((uint)4294967295, count_local, 0, 32);` |
| store.global#2 | 0.05 | 2.8% | 1.0 | 0.06 | `if (((int)threadIdx.x) == 0)` |

### ced_copy_rows_kernel grid (200, 1, 1) block (128, 1, 1)

17 launches, 800 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 1.63 us, warp 1.22 us, nsys graphs 1.87 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.90 | 73.7% | 1.37 | 80.0% |
| unattributed | 0.32 | 26.3% | 0.34 | 20.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.90 | 73.7% | 1.0 | 1.22 | `for (int k = 0; k < 4; ++k) \| int64_t index = ced_ldg64((&(row_indices[((((int64_t)blockId` |

### gemm_splitk_reduce_kernel grid (4, 16, 1) block (128, 1, 1)

48 launches, 256 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 0.61 us, warp 0.54 us, nsys graphs 0.77 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.25 | 46.4% | 0.27 | 46.3% |
| load | 0.25 | 45.8% | 0.27 | 45.4% |
| store | 0.04 | 7.8% | 0.05 | 8.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.25 | 45.8% | 1.0 | 0.29 | `acc[0] = P[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x))` |
| store.global | 0.04 | 7.8% | 1.0 | 0.06 | `C[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x)) * (int64` |

### assign_draft_budget_kernel grid (1, 1, 1) block (256, 1, 1)

16 launches, 8 warps; ranges, warp and last warp from the 1 recording CTAs' 8 warps; span 1.73 us, warp 1.70 us, nsys graphs 1.68 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed, with loops' unsampled iterations | 0.68 | 40.0% | 0.78 | 45.5% |
| wait | 0.63 | 36.9% | 0.40 | 23.2% |
| load | 0.20 | 11.8% | 0.35 | 20.2% |
| store | 0.19 | 11.3% | 0.19 | 11.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync | 0.32 | 18.6% | 1.0 | 0.35 | `__syncthreads();` |
| wait.sync#3 | 0.25 | 14.6% | 1.0 | 0.32 | `__syncthreads();` |
| load.global | 0.20 | 11.8% | 1.0 | 0.48 | `for (int chunk = 0; chunk < ((num_reqs + 255) >> 8); ++chunk)` |
| store.global | 0.19 | 11.3% | 1.0 | 0.19 | `for (int chunk_2 = 0; chunk_2 < ((num_reqs + 255) >> 8); ++chunk_2)` |
| wait.sync#2 (sampled iterations) | 0.06 | 3.8% | 1.0 | 0.06 | `__syncthreads();` |

### combine_sampled_and_draft_tokens_kernel grid (8, 1, 1) block (32, 1, 1)

17 launches, 8 warps; ranges, warp and last warp from the 4 recording CTAs' 4 warps; span 1.31 us, warp 1.18 us, nsys graphs 1.57 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.65 | 55.1% | 0.67 | 54.9% |
| unattributed | 0.43 | 36.7% | 0.46 | 37.3% |
| store | 0.10 | 8.2% | 0.10 | 7.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.65 | 55.1% | 1.0 | 0.67 | `if (((((int)blockIdx.x) * 32) + ((int)threadIdx.x)) < num_reqs) \| req = idx_mapping[((int6` |
| store.global | 0.10 | 8.2% | 1.0 | 0.10 | `if ((prefill < seq) && (1 < num_logits))` |

### token_counts_kernel grid (8, 1, 1) block (128, 1, 1)

17 launches, 32 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 1.15 us, warp 0.48 us, nsys graphs 1.26 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.28 | 59.2% | 0.35 | 28.6% |
| store | 0.14 | 29.0% | 0.64 | 52.2% |
| load | 0.06 | 11.8% | 0.24 | 19.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.14 | 29.0% | 0.2 | 0.58 | `if (((int)threadIdx.x) == 0) \| sampled[((int64_t)((int)blockIdx.x))] = (((int)(prefill_len` |
| load.global | 0.06 | 11.8% | 0.2 | 0.22 | `int count = sampled[((int64_t)((int)blockIdx.x))];` |

### logits_rows_kernel grid (1, 1, 1) block (128, 1, 1)

17 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 1.18 us, warp 1.14 us, nsys graphs 2.11 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.51 | 45.1% | 0.50 | 42.3% |
| unattributed | 0.48 | 42.3% | 0.30 | 25.0% |
| store | 0.14 | 12.7% | 0.39 | 32.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.51 | 45.1% | 1.0 | 0.51 | `index = logits_indices[min(((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((in` |
| store.global | 0.14 | 12.7% | 1.0 | 0.38 | `if (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) < num_logits)` |

### prepare_pos_seq_lens_kernel grid (9, 1, 1) block (256, 1, 1)

17 launches, 72 warps; ranges, warp and last warp from the 5 recording CTAs' 40 warps; span 1.12 us, warp 0.98 us, nsys graphs 1.28 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.46 | 46.5% | 0.56 | 50.2% |
| unattributed | 0.41 | 41.8% | 0.43 | 38.3% |
| store | 0.12 | 11.7% | 0.13 | 11.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.46 | 46.5% | 0.8 | 0.58 | `int computed = num_computed_tokens[idx_mapping[((int64_t)((int)blockIdx.x))]];` |
| store.global#2 | 0.10 | 10.4% | 0.8 | 0.13 | `if (((int)threadIdx.x) == 0)` |
| store.global | 0.01 | 1.3% | 0.2 | 0.06 | `for (int it = 0; it < (((max_reqs + 255) - num_reqs) >> 8); ++it)` |

### flatten_sampled_kernel grid (1, 1, 1) block (128, 1, 1)

17 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 1.12 us, warp 0.81 us, nsys graphs 1.34 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.47 | 58.4% | 0.64 | 57.4% |
| unattributed | 0.31 | 38.6% | 0.38 | 34.1% |
| store | 0.02 | 3.0% | 0.09 | 8.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.42 | 51.5% | 1.0 | 0.42 | `for (int r = 0; r < num_reqs; ++r)` |
| load.global#2 | 0.06 | 6.9% | 0.2 | 0.22 | `int i = (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) - cu_num_logits[req]);` |
| store.global | 0.02 | 3.0% | 0.2 | 0.10 | `flat[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = co` |

### scatter_rows_gather_kernel grid (1, 1, 1) block (128, 1, 1)

17 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 0.90 us, warp 0.83 us, nsys graphs 1.70 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.50 | 60.6% | 0.55 | 51.1% |
| store | 0.23 | 27.9% | 0.42 | 38.8% |
| load | 0.10 | 11.5% | 0.11 | 10.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.23 | 27.9% | 1.0 | 0.42 | `if (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) < (num_reqs * width))` |
| load.global | 0.10 | 11.5% | 1.0 | 0.10 | `index = idx_mapping[min((((((int64_t)0 <= ((int64_t)width)) && ((int64_t)0 <= rmod_1)) \|\| ` |

### padding_rows_kernel grid (1, 1, 1) block (256, 1, 1)

33 launches, 8 warps; ranges, warp and last warp from the 1 recording CTAs' 8 warps; span 0.38 us, warp 0.36 us, nsys graphs 0.74 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.26 | 73.0% | 0.28 | 73.6% |
| store | 0.10 | 27.0% | 0.10 | 26.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.10 | 27.0% | 1.0 | 0.10 | `if (((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) < rows)` |

### ced_requests_kernel grid (1, 1, 1) block (128, 1, 1)

17 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 0.61 us, warp 0.54 us, nsys graphs 1.15 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.34 | 64.2% | 0.34 | 52.8% |
| load | 0.10 | 17.9% | 0.11 | 16.7% |
| store | 0.10 | 17.9% | 0.20 | 30.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.10 | 17.9% | 1.0 | 0.10 | `uint v_ = tl::load_global_32((&(starts[1]))); \| uint v_ = tl::load_global_32((&(starts[min` |
| store.global | 0.10 | 17.9% | 1.0 | 0.19 | `if (((int)threadIdx.x) < 8) \| if (((int)threadIdx.x) <= 8)` |

### ced_copy_rows_kernel grid (80, 1, 1) block (128, 1, 1)

16 launches, 320 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 0.67 us, warp 0.59 us, nsys graphs 2.16 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.36 | 60.2% | 0.40 | 60.6% |
| unattributed | 0.23 | 39.8% | 0.26 | 39.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.36 | 60.2% | 1.0 | 0.38 | `for (int k = 0; k < 4; ++k) \| int64_t index = ced_ldg64((&(row_indices[((((int64_t)blockId` |

### scatter_rows_kernel grid (1, 1, 1) block (128, 1, 1)

17 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 0.64 us, warp 0.59 us, nsys graphs 0.99 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.40 | 67.6% | 0.39 | 61.9% |
| load | 0.10 | 16.2% | 0.11 | 17.1% |
| store | 0.10 | 16.2% | 0.13 | 21.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.10 | 16.2% | 1.0 | 0.10 | `index = idx_mapping[min((((((int64_t)0 <= ((int64_t)width)) && ((int64_t)0 <= rmod_1)) \|\| ` |
| store.global | 0.10 | 16.2% | 1.0 | 0.13 | `if (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) < (num_reqs * width))` |

### context_inputs_kernel grid (5, 48, 1) block (128, 1, 1)

1 launches, 960 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 9.92 us, warp 8.72 us, nsys graphs 6.85 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 7.04 | 80.8% | 9.15 | 92.3% |
| unattributed | 1.09 | 12.5% | 0.64 | 6.5% |
| store | 0.58 | 6.7% | 0.13 | 1.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 7.04 | 80.8% | 1.0 | 9.34 | `vllm_dspark_context_b128((&(aux[(((((int64_t)((int)blockIdx.y)) * (int64_t)15360) + (((int` |
| store.global | 0.58 | 6.7% | 1.0 | 2.18 | `if (((((int)blockIdx.x) == 0) & (((int)threadIdx.x) == 0)) & (num_tokens <= ((int)blockIdx` |

### per_token_cast_kernel_kernel grid (360, 1, 1) block (64, 1, 1)

1 launches, 720 warps; ranges, warp and last warp from the 4 recording CTAs' 8 warps; span 4.06 us, warp 3.00 us, nsys graphs 4.64 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 1.56 | 52.0% | 1.76 | 47.4% |
| load | 0.84 | 28.0% | 0.86 | 23.3% |
| compute | 0.40 | 13.3% | 0.13 | 3.4% |
| unattributed | 0.20 | 6.7% | 0.96 | 25.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#3*2 (est.: sampled iterations x2) | 0.74 | 24.8% | 1.0 | 2.24 | `*(fp8_e4_16_t*)(out + ((((((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int64_t)10240) +` |
| load.global | 0.71 | 23.7% | 1.0 | 0.93 | `for (int i = 0; i < 2; ++i)` |
| store.global*2 (est.: sampled iterations x2) | 0.42 | 13.9% | 1.0 | 2.11 | `out_sf[((((((int64_t)((int)blockIdx.x)) % (int64_t)15) * (int64_t)32) + (((int64_t)((int)t` |
| store.global#2 | 0.40 | 13.3% | 1.0 | 1.12 | `for (int i_3 = 0; i_3 < 4; ++i_3)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.40 | 13.3% | 1.0 | 2.37 | `amax_fragment[i_1] = tl::AllReduce<tl::MaxOp, 2, 1, 0, tl::NamedBarrier<64>>::run(amax_fra` |
| load.global#2*2 (est.: sampled iterations x2) | 0.13 | 4.3% | 1.0 | 0.19 | `amax_fragment[i_1] = cutlass::bfloat16_t(__hmax((bfloat16_t(((nv_bfloat162*)(&(amax_fragme` |

### gemm_splitk_reduce_kernel grid (4, 48, 1) block (128, 1, 1)

3 launches, 768 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 0.80 us, warp 0.72 us, nsys graphs 1.25 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.39 | 53.7% | 0.42 | 54.9% |
| unattributed | 0.29 | 39.6% | 0.28 | 36.6% |
| store | 0.05 | 6.6% | 0.06 | 8.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.39 | 53.7% | 1.0 | 0.42 | `acc[0] = P[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x))` |
| store.global | 0.05 | 6.6% | 1.0 | 0.06 | `C[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x)) * (int64` |

### ced_copy_rows_kernel grid (240, 1, 1) block (128, 1, 1)

1 launches, 960 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 0.90 us, warp 0.76 us, nsys graphs 1.66 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.48 | 63.0% | 0.54 | 63.0% |
| unattributed | 0.28 | 37.0% | 0.32 | 37.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.48 | 63.0% | 1.0 | 0.58 | `for (int k = 0; k < 4; ++k) \| int64_t index = ced_ldg64((&(row_indices[((((int64_t)blockId` |
=== timeline and tables: exit 0 in 36 s
