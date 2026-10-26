=== timeline and tables
trace /c/iket/step-tp3-c1-20261026T191719Z/trace/iket/pid_0x1843/iket.decoded_results.json: 12184 launches, graphLaunches dict of 124

## Timeline: 7754 launches in 829353.4 us; the GPU is busy 283356.3 us (34%) and idle 545997.2 us in 4926 gaps; gaps over 200 us split it into 480 segments

The 15 of 480 segments with the most busy time:

| segment | launches | wall us | busy us | busy | first kernel |
|---:|---:|---:|---:|---:|---|
| 81 | 117 | 6842.2 | 6743.9 | 99% | padding_rows_kernel |
| 110 | 116 | 6918.9 | 6839.8 | 99% | mhc_project_streams_kernel |
| 145 | 117 | 7011.6 | 6754.7 | 96% | padding_rows_kernel |
| 169 | 117 | 6831.3 | 6740.3 | 99% | padding_rows_kernel |
| 193 | 120 | 7457.7 | 6856.4 | 92% | metadata_builder_kernel |
| 208 | 116 | 6841.8 | 6754.5 | 99% | mhc_project_streams_kernel |
| 224 | 120 | 7353.1 | 6767.0 | 92% | metadata_builder_kernel |
| 246 | 116 | 6841.3 | 6755.6 | 99% | mhc_project_streams_kernel |
| 264 | 120 | 7250.6 | 6767.8 | 93% | metadata_builder_kernel |
| 285 | 117 | 6874.0 | 6771.5 | 99% | padding_rows_kernel |
| 310 | 116 | 6830.1 | 6740.9 | 99% | mhc_project_streams_kernel |
| 354 | 116 | 6835.1 | 6748.3 | 99% | mhc_project_streams_kernel |
| 376 | 119 | 7221.4 | 6766.9 | 94% | metadata_builder_kernel |
| 390 | 120 | 7072.8 | 6744.3 | 95% | metadata_builder_kernel |
| 477 | 117 | 6837.8 | 6743.2 | 99% | padding_rows_kernel |

| idle us | after | before |
|---:|---|---|
| 123619.6 | resample_kernel | insert_kernel |
| 80115.1 | scatter_rows_kernel | padding_rows_kernel |
| 3425.4 | norm_forward_and_per_token_cast_kernel_kernel | sink_flags_kernel |
| 3329.4 | dspark_confidence_kernel | scatter_rows_gather_kernel |
| 3167.7 | norm_forward_and_per_token_cast_kernel_kernel | sink_flags_kernel |
| 3139.9 | norm_forward_and_per_token_cast_kernel_kernel | sink_flags_kernel |
| 3127.7 | norm_forward_and_per_token_cast_kernel_kernel | sink_flags_kernel |
| 3001.1 | dspark_confidence_kernel | scatter_rows_gather_kernel |
| 2973.2 | norm_forward_and_per_token_cast_kernel_kernel | sink_flags_kernel |
| 2948.4 | dspark_confidence_kernel | scatter_rows_gather_kernel |

## Engine steps

| step | kind | launches | wall us | busy us | busy | most span |
|---:|---|---:|---:|---:|---:|---|
| 0 | prefill | 413 | 74820.2 | 17430.5 | 23% | vocab_packed_projection_kernel 8076, gemm_mxfp8_kernel 3173, moe_grouped_fp8_fp4_gemm_kernel 2524 |
| 1 | decode | 432 | 155360.5 | 15309.2 | 10% | vocab_packed_projection_kernel 8181, gemm_mxfp8_kernel 2342, l2_prefetch_kernel 1890 |
| 2 | decode | 432 | 113988.4 | 15806.3 | 14% | vocab_packed_projection_kernel 7981, gemm_mxfp8_kernel 2496, l2_prefetch_kernel 2146 |
| 3 | decode | 432 | 32609.1 | 15602.4 | 48% | vocab_packed_projection_kernel 8120, gemm_mxfp8_kernel 2570, l2_prefetch_kernel 2324 |
| 4 | decode | 432 | 29981.0 | 15656.1 | 52% | vocab_packed_projection_kernel 8098, gemm_mxfp8_kernel 2253, moe_grouped_fp8_fp4_gemm_kernel 1808 |
| 5 | decode | 432 | 29057.4 | 15933.9 | 55% | vocab_packed_projection_kernel 7997, gemm_mxfp8_kernel 2362, moe_grouped_fp8_fp4_gemm_kernel 2027 |
| 6 | decode | 432 | 31237.6 | 15657.7 | 50% | vocab_packed_projection_kernel 7991, gemm_mxfp8_kernel 2259, moe_grouped_fp8_fp4_gemm_kernel 1809 |
| 7 | decode | 432 | 30595.9 | 15966.2 | 52% | vocab_packed_projection_kernel 7999, gemm_mxfp8_kernel 2406, moe_grouped_fp8_fp4_gemm_kernel 1935 |
| 8 | decode | 432 | 33472.4 | 15608.8 | 47% | vocab_packed_projection_kernel 8044, gemm_mxfp8_kernel 2201, moe_grouped_fp8_fp4_gemm_kernel 1808 |
| 9 | decode | 432 | 32191.0 | 15617.4 | 49% | vocab_packed_projection_kernel 8008, gemm_mxfp8_kernel 2243, moe_grouped_fp8_fp4_gemm_kernel 1816 |
| 10 | decode | 432 | 32327.5 | 15626.0 | 48% | vocab_packed_projection_kernel 7988, gemm_mxfp8_kernel 2312, moe_grouped_fp8_fp4_gemm_kernel 1866 |
| 11 | decode | 432 | 29225.0 | 15603.1 | 53% | vocab_packed_projection_kernel 7975, gemm_mxfp8_kernel 2253, moe_grouped_fp8_fp4_gemm_kernel 1804 |
| 12 | decode | 432 | 31870.7 | 15573.5 | 49% | vocab_packed_projection_kernel 7989, gemm_mxfp8_kernel 2278, moe_grouped_fp8_fp4_gemm_kernel 1814 |
| 13 | decode | 432 | 33609.2 | 15603.7 | 46% | vocab_packed_projection_kernel 8011, gemm_mxfp8_kernel 2268, moe_grouped_fp8_fp4_gemm_kernel 1798 |
| 14 | decode | 432 | 29530.8 | 15587.2 | 53% | vocab_packed_projection_kernel 8001, gemm_mxfp8_kernel 2273, moe_grouped_fp8_fp4_gemm_kernel 1817 |
| 15 | decode | 432 | 33722.9 | 15539.4 | 46% | vocab_packed_projection_kernel 7959, gemm_mxfp8_kernel 2272, moe_grouped_fp8_fp4_gemm_kernel 1820 |
| 16 | decode | 432 | 38091.5 | 15582.3 | 41% | vocab_packed_projection_kernel 8023, gemm_mxfp8_kernel 2234, moe_grouped_fp8_fp4_gemm_kernel 1810 |
| 17 | decode | 429 | 32517.8 | 15652.6 | 48% | vocab_packed_projection_kernel 8021, gemm_mxfp8_kernel 2314, moe_grouped_fp8_fp4_gemm_kernel 1807 |

## Overview over 17 decode steps: 7341 launches of 142 kernel shapes, spans summing to 296005.6 us (17412.1 a step)

| kernel | grid | launches a step | span us a step | share | span us a launch | last warp: wait | load | compute | store | unattributed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vocab_packed_projection_kernel | (1, 4040, 1) | 2 | 8022.7 | 46.1% | 3999.38 | 47% | 29% | 10% | 0% | 14% |
| l2_prefetch_kernel | (2, 1, 1) | 25 | 1669.3 | 9.6% | 64.64 | 0% | 100% | 0% | 0% | 0% |
| gemm_bf16_kernel | (2020, 1, 1) | 5 | 1331.5 | 7.6% | 266.18 | 89% | 0% | 5% | 1% | 5% |
| moe_grouped_fp8_fp4_gemm_kernel | (12, 30, 1) | 8.64706 | 1163.0 | 6.7% | 170.34 | 68% | 16% | 21% | 1% | 0% |
| gemm_mxfp8_kernel | (80, 1, 1) | 19 | 1093.5 | 6.3% | 51.65 | 68% | 7% | 5% | 0% | 19% |
| moe_grouped_fp8_fp4_gemm_kernel | (40, 30, 1) | 8.64706 | 597.1 | 3.4% | 81.54 | 15% | 18% | 0% | 0% | 67% |
| gemm_mxfp8_kernel | (192, 1, 1) | 9 | 547.4 | 3.1% | 52.42 | 24% | 6% | 5% | 0% | 65% |
| mhc_project_streams_kernel | (40, 1, 1) | 18 | 334.9 | 1.9% | 18.67 | 28% | 28% | 15% | 3% | 26% |
| gemm_mxfp8_kernel | (28, 1, 1) | 9 | 324.4 | 1.9% | 36.13 | 44% | 10% | 11% | 0% | 35% |
| gemm_fp8w_bf16_partials_kernel | (48, 4, 1) | 9 | 306.6 | 1.8% | 26.78 | 48% | 18% | 9% | 0% | 25% |
| gemm_mxfp8_kernel | (24, 1, 1) | 9 | 301.5 | 1.7% | 31.94 | 35% | 21% | 19% | 0% | 25% |
| gemm_bf16_partials_kernel | (6, 8, 1) | 6 | 138.0 | 0.8% | 22.48 | 39% | 0% | 23% | 1% | 36% |
| mhc_finalize_kernel | (5, 1, 1) | 11.2941 | 122.2 | 0.7% | 11.04 | 7% | 15% | 11% | 23% | 44% |
| gemm_bf16_splitk_kernel | (16, 5, 1) | 2 | 109.0 | 0.6% | 55.34 | 98% | 9% | 4% | 0% | 0% |
| sparse_mla_kernel | (5, 1, 5) | 3.76471 | 85.7 | 0.5% | 18.83 | 17% | 23% | 7% | 4% | 49% |
| mhc_finalize_kernel | (10, 1, 1) | 6 | 70.7 | 0.4% | 12.32 | 10% | 15% | 9% | 19% | 47% |
| sparse_mla_kernel | (2, 2, 10) | 3 | 61.1 | 0.4% | 20.54 | 14% | 25% | 23% | 2% | 36% |
| rope_kernel | (1, 1, 1) | 16 | 49.0 | 0.3% | 1.39 | 0% | 12% | 0% | 2% | 86% |
| gemm_mxfp8_kernel | (64, 1, 1) | 2 | 47.1 | 0.3% | 23.47 | 33% | 16% | 10% | 1% | 40% |
| cache_writer_kernel | (5, 1, 1) | 7.52941 | 46.1 | 0.3% | 5.89 | 0% | 11% | 1% | 33% | 55% |

| wait | kernel | grid | last-warp us a step | a launch | first statement |
|---|---|---|---:|---:|---|
| wait.cp_async*22 | gemm_mxfp8_kernel | (80, 1, 1) | 244.8 | 12.88 | `tl::cp_async_wait<2>();` |
| wait.mbarrier*40 | moe_grouped_fp8_fp4_gemm_kernel | (12, 30, 1) | 179.7 | 20.78 | `loaded[(ko % 3)].wait(((ko % 6) / 3));` |
| wait.cp_async*118 | gemm_mxfp8_kernel | (80, 1, 1) | 178.4 | 9.39 | `tl::cp_async_wait<2>();` |
| wait.sync#2*22 | gemm_mxfp8_kernel | (80, 1, 1) | 135.8 | 7.15 | `__syncthreads();` |
| wait.sync#2*118 | gemm_mxfp8_kernel | (80, 1, 1) | 88.6 | 4.66 | `__syncthreads();` |
| wait.sync*4 | mhc_project_streams_kernel | (40, 1, 1) | 82.3 | 4.57 | `__syncthreads();` |
| wait.sync#2*19 | gemm_mxfp8_kernel | (28, 1, 1) | 69.5 | 7.72 | `__syncthreads();` |
| wait.cp_async*40 | moe_grouped_fp8_fp4_gemm_kernel | (12, 30, 1) | 69.1 | 7.99 | `tl::cp_async_wait<2>();` |
| wait.cp_async*19 | gemm_mxfp8_kernel | (28, 1, 1) | 56.0 | 6.23 | `tl::cp_async_wait<1>();` |
| wait.cp_async*13 | gemm_bf16_splitk_kernel | (16, 5, 1) | 55.0 | 27.50 | `tl::cp_async_wait<2>();` |
| wait.cp_async*14 | gemm_fp8w_bf16_partials_kernel | (48, 4, 1) | 44.1 | 4.90 | `tl::cp_async_wait<3>();` |
| wait.cp_async*38 | gemm_mxfp8_kernel | (24, 1, 1) | 43.8 | 4.87 | `tl::cp_async_wait<2>();` |
| wait.sync#2*38 | gemm_mxfp8_kernel | (24, 1, 1) | 28.2 | 3.13 | `__syncthreads();` |
| wait.mbarrier#2 | gemm_bf16_kernel | (2020, 1, 1) | 26.9 | 5.38 | `mbarrier[0].wait(0);` |
| wait.sync#2*17 | vocab_packed_projection_kernel | (1, 4040, 1) | 24.9 | 12.45 | `__syncthreads();` |
| wait.sync#2*14 | gemm_fp8w_bf16_partials_kernel | (48, 4, 1) | 22.3 | 2.48 | `__syncthreads();` |
| wait.sync*38 | gemm_mxfp8_kernel | (24, 1, 1) | 21.7 | 2.41 | `__syncthreads();` |
| wait.cp_async*4 | gemm_mxfp8_kernel | (80, 1, 1) | 20.9 | 1.10 | `tl::cp_async_wait<2>();` |
| wait.sync#2*4 | gemm_mxfp8_kernel | (80, 1, 1) | 18.7 | 0.98 | `__syncthreads();` |
| wait.sync#5*40 | moe_grouped_fp8_fp4_gemm_kernel | (12, 30, 1) | 18.1 | 2.10 | `__syncthreads();` |

### vocab_packed_projection_kernel grid (1, 4040, 1) block (128, 1, 1)

34 launches, 16160 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 3999.38 us, warp 48.94 us, nsys graphs 4061.54 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 29.02 | 59.3% | 23.56 | 46.9% |
| load | 9.62 | 19.7% | 14.37 | 28.6% |
| unattributed | 5.20 | 10.6% | 7.22 | 14.4% |
| compute | 5.05 | 10.3% | 5.05 | 10.1% |
| store | 0.05 | 0.1% | 0.05 | 0.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.cp_async*17 (est.: sampled iterations x17) | 13.04 | 26.6% | 1.0 | 51.14 | `tl::cp_async_wait<6>();` |
| wait.sync#2*17 (est.: sampled iterations x17) | 10.81 | 22.1% | 1.0 | 34.00 | `__syncthreads();` |
| load.global | 7.13 | 14.6% | 1.0 | 12.74 | `int start = row_offsets[(((int)blockIdx.y) * 16)]; \| int start = row_offsets[(((int)blockI` |
| compute.mma*17 (est.: sampled iterations x17) | 4.35 | 8.9% | 1.0 | 4.35 | `{` |
| load.cp_async#2*17 (est.: sampled iterations x17) | 2.21 | 4.5% | 1.0 | 2.72 | `for (int i_6 = 0; i_6 < 8; ++i_6) \| for (int i_9 = 0; i_9 < 4; ++i_9)` |
| wait.sync#3*17 (est.: sampled iterations x17) | 1.09 | 2.2% | 1.0 | 1.09 | `__syncthreads();` |
| wait.sync*17 (est.: sampled iterations x17) | 1.05 | 2.2% | 1.0 | 1.09 | `__syncthreads();` |
| wait.cp_async#2*17 (est.: sampled iterations x17) | 0.95 | 1.9% | 1.0 | 1.09 | `tl::cp_async_wait<7>();` |
| wait.sync#6 | 0.41 | 0.8% | 1.0 | 1.54 | `__syncthreads();` |
| wait.sync#4 | 0.37 | 0.7% | 1.0 | 1.47 | `__syncthreads();` |
| wait.cp_async#3 | 0.36 | 0.7% | 1.0 | 1.81 | `tl::cp_async_wait<4>();` |
| wait.cp_async#5 | 0.29 | 0.6% | 1.0 | 1.65 | `tl::cp_async_wait<2>();` |
| load.cp_async | 0.28 | 0.6% | 1.0 | 0.32 | `for (int i = 0; i < 4; ++i) \| for (int i = 0; i < 8; ++i) \| for (int i_1 = 0; i_1 < 4; ++i` |
| compute.mma#2 | 0.23 | 0.5% | 1.0 | 0.26 | `{` |
| compute.mma#3 | 0.23 | 0.5% | 1.0 | 0.26 | `{` |
| compute.mma#4 | 0.23 | 0.5% | 1.0 | 0.26 | `{` |
| wait.sync#8 | 0.20 | 0.4% | 1.0 | 0.85 | `__syncthreads();` |
| wait.cp_async#7 | 0.16 | 0.3% | 1.0 | 0.85 | `tl::cp_async_wait<0>();` |
| wait.sync#5 | 0.05 | 0.1% | 1.0 | 0.06 | `__syncthreads();` |
| store.global | 0.05 | 0.1% | 1.0 | 0.06 | `for (int i_13 = 0; i_13 < 2; ++i_13) \| for (int i_14 = 0; i_14 < 4; ++i_14) \| for (int i_9` |
| wait.sync#7 | 0.05 | 0.1% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#9 | 0.05 | 0.1% | 1.0 | 0.06 | `__syncthreads();` |
| wait.cp_async#4 | 0.04 | 0.1% | 1.0 | 0.06 | `tl::cp_async_wait<5>();` |
| wait.cp_async#6 | 0.04 | 0.1% | 1.0 | 0.06 | `tl::cp_async_wait<3>();` |
| wait.cp_async#8 | 0.04 | 0.1% | 1.0 | 0.06 | `tl::cp_async_wait<1>();` |

### l2_prefetch_kernel grid (2, 1, 1) block (128, 1, 1)

425 launches, 8 warps; ranges, warp and last warp from the 2 recording CTAs' 8 warps; span 64.64 us, warp 63.57 us, nsys graphs 66.74 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 63.15 | 99.3% | 66.46 | 99.5% |
| unattributed | 0.42 | 0.7% | 0.31 | 0.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 63.15 | 99.3% | 1.0 | 64.32 | `for (int s = 0; s < (nseg2 >> 1); ++s)` |

### gemm_bf16_kernel grid (2020, 1, 1) block (256, 1, 1)

85 launches, 16160 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 266.18 us, warp 3.62 us, nsys graphs 271.52 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 3.04 | 84.0% | 5.47 | 88.9% |
| unattributed | 0.36 | 10.0% | 0.33 | 5.3% |
| compute | 0.14 | 4.0% | 0.29 | 4.7% |
| load | 0.04 | 1.0% | 0.00 | 0.0% |
| store | 0.04 | 1.0% | 0.07 | 1.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.mbarrier#2 | 2.94 | 81.1% | 0.5 | 7.20 | `mbarrier[0].wait(0);` |
| compute.mma | 0.14 | 4.0% | 0.5 | 0.29 | `{` |
| wait.sync | 0.07 | 2.1% | 1.0 | 0.10 | `__syncthreads();` |
| load.tma | 0.04 | 1.0% | 0.5 | 0.13 | `if (tl::tl_shuffle_elect<128>())` |
| store.global | 0.04 | 1.0% | 0.5 | 0.10 | `for (int i_3 = 0; i_3 < 4; ++i_3) \| for (int i_4 = 0; i_4 < 8; ++i_4) \| for (int i_6 = 0; ` |
| wait.mbarrier | 0.03 | 0.8% | 0.5 | 0.06 | `mbarrier[2].wait(1);` |

### moe_grouped_fp8_fp4_gemm_kernel grid (12, 30, 1) block (128, 1, 1)

147 launches, 1440 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 170.34 us, warp 33.97 us, nsys graphs 158.53 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 13.57 | 39.9% | 34.84 | 68.0% |
| unattributed | 10.05 | 29.6% | 0.00 | 0.0% |
| compute | 6.64 | 19.5% | 10.98 | 21.4% |
| load | 3.57 | 10.5% | 8.31 | 16.2% |
| store | 0.14 | 0.4% | 0.29 | 0.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.mbarrier*40 (est.: sampled iterations x40) | 7.68 | 22.6% | 0.5 | 25.60 | `loaded[(ko % 3)].wait(((ko % 6) / 3));` |
| compute.mma*40 (est.: sampled iterations x40) | 6.64 | 19.5% | 0.5 | 14.08 | `{` |
| load#2*40 (est.: sampled iterations x40) | 2.88 | 8.5% | 0.5 | 8.96 | `if (ko < 38)` |
| wait.sync#5*40 (est.: sampled iterations x40) | 2.40 | 7.1% | 0.5 | 6.40 | `__syncthreads();` |
| wait.sync#6*40 (est.: sampled iterations x40) | 1.20 | 3.5% | 0.5 | 2.56 | `__syncthreads();` |
| wait.sync#4*40 (est.: sampled iterations x40) | 1.12 | 3.3% | 0.5 | 2.56 | `__syncthreads();` |
| wait.cp_async*40 (est.: sampled iterations x40) | 1.04 | 3.1% | 0.5 | 2.56 | `tl::cp_async_wait<2>();` |
| load.global#3 | 0.49 | 1.4% | 0.5 | 1.47 | `if (((expert < 128) && (0 <= expert))) \| if (((expert < 384) && (0 <= expert)))` |
| store.global | 0.14 | 0.4% | 0.5 | 0.38 | `for (int i_2 = 0; i_2 < 16; ++i_2) \| for (int i_4 = 0; i_4 < 32; ++i_4)` |
| load | 0.08 | 0.2% | 0.5 | 0.26 | `for (int s = 0; s < 2; ++s)` |
| load.global | 0.07 | 0.2% | 1.0 | 0.10 | `int expert = TileExpert[((int64_t)((int)blockIdx.y))];` |
| wait.sync#2 | 0.06 | 0.2% | 0.5 | 0.16 | `__syncthreads();` |
| load.global#2 | 0.05 | 0.1% | 0.5 | 0.19 | `if (((int)threadIdx.x) < 32) \| if (((int)threadIdx.x) < 64)` |
| wait.sync | 0.05 | 0.1% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#3 | 0.02 | 0.0% | 0.5 | 0.06 | `__syncthreads();` |

### gemm_mxfp8_kernel grid (80, 1, 1) block (128, 1, 1)

323 launches, 320 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 51.65 us, warp 47.09 us, nsys graphs 48.11 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 44.82 | 95.2% | 10.72 | 18.9% |
| wait | 1.72 | 3.6% | 38.78 | 68.3% |
| compute | 0.25 | 0.5% | 3.06 | 5.4% |
| load | 0.24 | 0.5% | 4.20 | 7.4% |
| store | 0.06 | 0.1% | 0.06 | 0.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.cp_async#2 | 0.52 | 1.1% | 1.0 | 2.02 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| wait.sync#3 | 0.44 | 0.9% | 1.0 | 1.44 | `__syncthreads();` |
| wait.cp_async#3 | 0.40 | 0.9% | 1.0 | 1.54 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>();` |
| wait.sync#4 | 0.35 | 0.8% | 1.0 | 1.25 | `__syncthreads();` |
| load.cp_async | 0.24 | 0.5% | 1.0 | 0.38 | `for (int i_1 = 0; i_1 < 2; ++i_1) \| for (int i_1 = 0; i_1 < 4; ++i_1) \| for (int i_1 = 0; ` |
| compute.mma#3 | 0.13 | 0.3% | 1.0 | 0.16 | `{` |
| compute.mma#2 | 0.12 | 0.3% | 1.0 | 0.13 | `{` |
| store.global | 0.06 | 0.1% | 1.0 | 0.06 | `for (int i_10 = 0; i_10 < 16; ++i_10) \| for (int i_10 = 0; i_10 < 8; ++i_10) \| for (int i_` |
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

### moe_grouped_fp8_fp4_gemm_kernel grid (40, 30, 1) block (128, 1, 1)

147 launches, 4800 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 81.54 us, warp 5.80 us, nsys graphs 76.42 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 2.08 | 35.8% | 0.05 | 14.9% |
| unattributed | 1.61 | 27.7% | 0.21 | 67.1% |
| compute | 1.00 | 17.2% | 0.00 | 0.0% |
| load | 0.98 | 16.8% | 0.06 | 18.0% |
| store | 0.14 | 2.5% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*6 (est.: sampled iterations x6) | 1.00 | 17.2% | 0.5 | 2.11 | `{` |
| wait.sync#5*6 (est.: sampled iterations x6) | 0.71 | 12.2% | 0.5 | 3.07 | `__syncthreads();` |
| wait.mbarrier*6 (est.: sampled iterations x6) | 0.62 | 10.8% | 0.5 | 2.11 | `loaded[(ko % 3)].wait((ko / 3));` |
| load#2*6 (est.: sampled iterations x6) | 0.40 | 6.8% | 0.5 | 1.15 | `if (ko < 4)` |
| load.global#3 | 0.38 | 6.6% | 0.5 | 1.09 | `if (((expert < 128) && (0 <= expert))) \| if (((expert < 384) && (0 <= expert)))` |
| wait.cp_async*6 (est.: sampled iterations x6) | 0.29 | 5.0% | 0.5 | 2.50 | `tl::cp_async_wait<2>();` |
| wait.sync#4*6 (est.: sampled iterations x6) | 0.17 | 2.9% | 0.5 | 0.38 | `__syncthreads();` |
| wait.sync#6*6 (est.: sampled iterations x6) | 0.17 | 2.9% | 0.5 | 0.38 | `__syncthreads();` |
| store.global | 0.14 | 2.5% | 0.5 | 0.38 | `for (int i_2 = 0; i_2 < 16; ++i_2) \| for (int i_4 = 0; i_4 < 32; ++i_4)` |
| load | 0.08 | 1.4% | 0.5 | 0.26 | `for (int s = 0; s < 2; ++s)` |
| load.global | 0.07 | 1.2% | 1.0 | 0.10 | `int expert = TileExpert[((int64_t)((int)blockIdx.y))];` |
| wait.sync#2 | 0.06 | 1.0% | 0.5 | 0.16 | `__syncthreads();` |
| load.global#2 | 0.05 | 0.8% | 0.5 | 0.19 | `if (((int)threadIdx.x) < 32) \| if (((int)threadIdx.x) < 64)` |
| wait.sync | 0.05 | 0.8% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#3 | 0.02 | 0.3% | 0.5 | 0.06 | `__syncthreads();` |

### gemm_mxfp8_kernel grid (192, 1, 1) block (128, 1, 1)

153 launches, 768 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 52.42 us, warp 32.00 us, nsys graphs 71.92 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 17.11 | 53.5% | 16.10 | 65.0% |
| wait | 11.76 | 36.7% | 5.94 | 24.0% |
| load | 1.67 | 5.2% | 1.38 | 5.6% |
| compute | 1.40 | 4.4% | 1.32 | 5.3% |
| store | 0.05 | 0.2% | 0.05 | 0.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync#2*8 (est.: sampled iterations x8) | 3.25 | 10.2% | 1.0 | 12.80 | `__syncthreads();` |
| wait.cp_async*8 (est.: sampled iterations x8) | 2.74 | 8.6% | 1.0 | 14.34 | `tl::cp_async_wait<2>();` |
| wait.cp_async#3 | 1.66 | 5.2% | 1.0 | 4.51 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>();` |
| wait.cp_async#2 | 1.59 | 5.0% | 1.0 | 5.34 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| load.cp_async#2*8 (est.: sampled iterations x8) | 1.34 | 4.2% | 1.0 | 4.10 | `for (int i_5 = 0; i_5 < 2; ++i_5) \| for (int i_5 = 0; i_5 < 4; ++i_5) \| tl::cp_async_gs_co` |
| compute.mma*8 (est.: sampled iterations x8) | 1.15 | 3.6% | 1.0 | 1.54 | `{` |
| wait.sync#4 | 1.08 | 3.4% | 1.0 | 3.26 | `__syncthreads();` |
| wait.sync#3 | 1.02 | 3.2% | 1.0 | 3.68 | `__syncthreads();` |
| wait.sync*8 (est.: sampled iterations x8) | 0.42 | 1.3% | 1.0 | 0.51 | `__syncthreads();` |
| load.cp_async | 0.33 | 1.0% | 1.0 | 0.80 | `for (int i_1 = 0; i_1 < 2; ++i_1) \| for (int i_1 = 0; i_1 < 4; ++i_1) \| for (int i_1 = 0; ` |
| compute.mma#3 | 0.13 | 0.4% | 1.0 | 0.16 | `{` |
| compute.mma#2 | 0.12 | 0.4% | 1.0 | 0.13 | `{` |
| store.global | 0.05 | 0.2% | 1.0 | 0.06 | `for (int i_10 = 0; i_10 < 16; ++i_10) \| for (int i_10 = 0; i_10 < 8; ++i_10) \| for (int i_` |

### mhc_project_streams_kernel grid (40, 1, 1) block (128, 1, 1)

306 launches, 160 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 18.67 us, warp 15.11 us, nsys graphs 16.34 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 7.50 | 49.6% | 4.49 | 26.1% |
| compute | 2.78 | 18.4% | 2.60 | 15.1% |
| load | 2.76 | 18.3% | 4.86 | 28.2% |
| wait | 1.60 | 10.6% | 4.76 | 27.7% |
| store | 0.46 | 3.1% | 0.51 | 2.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*4 (est.: sampled iterations x4) | 2.65 | 17.5% | 1.0 | 2.69 | `{` |
| load.global#2*4 (est.: sampled iterations x4) | 2.52 | 16.7% | 1.0 | 7.30 | `for (int i_1 = 0; i_1 < 6; ++i_1)` |
| wait.sync*4 (est.: sampled iterations x4) | 1.42 | 9.4% | 1.0 | 5.50 | `__syncthreads();` |
| store.global*4 (est.: sampled iterations x4) | 0.25 | 1.6% | 1.0 | 0.38 | `if (((((int)blockIdx.y) * 16) + (((int)threadIdx.x) >> 3)) < M)` |
| load.global | 0.23 | 1.5% | 1.0 | 0.26 | `for (int k = 0; k < 4; ++k)` |
| wait.sync#2*4 (est.: sampled iterations x4) | 0.18 | 1.2% | 1.0 | 0.26 | `__syncthreads();` |
| store.global#4 | 0.09 | 0.6% | 1.0 | 0.13 | `if (((((int)threadIdx.x) % 8) == 0) && (((((int)blockIdx.y) * 16) + (((int)threadIdx.x) >>` |
| compute.reduce | 0.07 | 0.5% | 1.0 | 0.10 | `red[0] = (red[0] + __shfl_xor_sync((uint)4294967295, red[0], 4, 32));` |
| compute.reduce#2 | 0.07 | 0.5% | 1.0 | 0.10 | `red[0] = (red[0] + __shfl_xor_sync((uint)4294967295, red[0], 4, 32));` |
| store.global#2 | 0.06 | 0.4% | 1.0 | 0.10 | `if (((((int)blockIdx.y) * 16) + (((int)threadIdx.x) >> 3)) < M) \| if (((((int)threadIdx.x)` |
| store.global#3 | 0.06 | 0.4% | 1.0 | 0.06 | `if (((((int)threadIdx.x) % 8) == 0) && (((((int)blockIdx.y) * 16) + (((int)threadIdx.x) >>` |
| compute.mma | 0.00 | 0.0% | 0.0 | 0.00 | `{` |
| load.global#2 | 0.00 | 0.0% | 0.0 | 0.00 | `for (int i_1 = 0; i_1 < 6; ++i_1)` |
| store.global | 0.00 | 0.0% | 0.0 | 0.00 | `for (int n = 0; n < 4; ++n) \| if (((((int)blockIdx.y) * 16) + (((int)threadIdx.x) >> 3)) <` |
| wait.sync | 0.00 | 0.0% | 0.0 | 0.00 | `__syncthreads();` |

### gemm_mxfp8_kernel grid (28, 1, 1) block (128, 1, 1)

153 launches, 112 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 36.13 us, warp 35.51 us, nsys graphs 41.90 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 21.78 | 61.3% | 12.40 | 34.6% |
| wait | 6.89 | 19.4% | 15.75 | 43.9% |
| compute | 3.94 | 11.1% | 3.91 | 10.9% |
| load | 2.84 | 8.0% | 3.75 | 10.5% |
| store | 0.05 | 0.2% | 0.05 | 0.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*19 (est.: sampled iterations x19) | 3.76 | 10.6% | 1.0 | 4.26 | `{` |
| wait.sync#2*19 (est.: sampled iterations x19) | 3.12 | 8.8% | 1.0 | 7.90 | `__syncthreads();` |
| load.cp_async#2*19 (est.: sampled iterations x19) | 2.66 | 7.5% | 1.0 | 3.04 | `for (int i_3 = 0; i_3 < 2; ++i_3) \| for (int i_3 = 0; i_3 < 8; ++i_3)` |
| wait.cp_async*19 (est.: sampled iterations x19) | 2.28 | 6.4% | 1.0 | 7.90 | `tl::cp_async_wait<1>();` |
| wait.sync*19 (est.: sampled iterations x19) | 1.22 | 3.4% | 1.0 | 1.22 | `__syncthreads();` |
| load.cp_async | 0.18 | 0.5% | 1.0 | 0.19 | `for (int i_1 = 0; i_1 < 2; ++i_1) \| for (int i_1 = 0; i_1 < 4; ++i_1) \| for (int i_1 = 0; ` |
| compute.mma#2 | 0.17 | 0.5% | 1.0 | 0.19 | `{` |
| wait.cp_async#2 | 0.15 | 0.4% | 1.0 | 0.42 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| wait.sync#3 | 0.13 | 0.4% | 1.0 | 0.35 | `__syncthreads();` |
| store.global | 0.05 | 0.2% | 1.0 | 0.06 | `for (int i_10 = 0; i_10 < 16; ++i_10) \| for (int i_10 = 0; i_10 < 8; ++i_10) \| for (int i_` |

### gemm_fp8w_bf16_partials_kernel grid (48, 4, 1) block (128, 1, 1)

153 launches, 768 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 26.78 us, warp 18.75 us, nsys graphs 47.70 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 7.70 | 41.1% | 14.42 | 47.8% |
| unattributed | 4.40 | 23.5% | 7.62 | 25.3% |
| load | 3.88 | 20.7% | 5.30 | 17.6% |
| compute | 2.70 | 14.4% | 2.77 | 9.2% |
| store | 0.07 | 0.4% | 0.06 | 0.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*14 (est.: sampled iterations x14) | 2.44 | 13.0% | 1.0 | 3.14 | `{` |
| wait.sync#2*14 (est.: sampled iterations x14) | 1.32 | 7.0% | 1.0 | 1.79 | `__syncthreads();` |
| load.global*14 (est.: sampled iterations x14) | 1.15 | 6.1% | 1.0 | 1.79 | `if (((int)threadIdx.x) < 4)` |
| load.cp_async#2*14 (est.: sampled iterations x14) | 1.12 | 6.0% | 1.0 | 1.34 | `for (int i_4 = 0; i_4 < 2; ++i_4) \| for (int i_6 = 0; i_6 < 2; ++i_6)` |
| load.cp_async#3*14 (est.: sampled iterations x14) | 1.06 | 5.7% | 1.0 | 1.34 | `for (int i_10 = 0; i_10 < 4; ++i_10) \| for (int i_10 = 0; i_10 < 6; ++i_10) \| for (int i_1` |
| wait.sync#4*14 (est.: sampled iterations x14) | 1.04 | 5.5% | 1.0 | 1.34 | `__syncthreads();` |
| wait.sync#3*14 (est.: sampled iterations x14) | 1.01 | 5.4% | 1.0 | 1.34 | `__syncthreads();` |
| wait.sync#5*14 (est.: sampled iterations x14) | 1.01 | 5.4% | 1.0 | 1.34 | `__syncthreads();` |
| wait.sync*14 (est.: sampled iterations x14) | 1.01 | 5.4% | 1.0 | 1.34 | `__syncthreads();` |
| wait.cp_async*14 (est.: sampled iterations x14) | 0.90 | 4.8% | 1.0 | 1.34 | `tl::cp_async_wait<3>();` |
| wait.cp_async#2*14 (est.: sampled iterations x14) | 0.87 | 4.6% | 1.0 | 1.34 | `tl::cp_async_wait<3>();` |
| load.cp_async | 0.42 | 2.2% | 1.0 | 0.70 | `for (int i_1 = 0; i_1 < 2; ++i_1)` |
| compute.mma#2 | 0.14 | 0.7% | 1.0 | 0.16 | `{` |
| compute.mma#3 | 0.13 | 0.7% | 1.0 | 0.16 | `{` |
| wait.sync#7 | 0.07 | 0.4% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#10 | 0.07 | 0.4% | 1.0 | 0.10 | `__syncthreads();` |
| store.global | 0.07 | 0.4% | 1.0 | 0.10 | `for (int i_13 = 0; i_13 < 8; ++i_13) \| for (int i_19 = 0; i_19 < 16; ++i_19) \| for (int i_` |
| load.global#2 | 0.06 | 0.3% | 1.0 | 0.10 | `if (((int)threadIdx.x) < 4)` |
| load.global#3 | 0.06 | 0.3% | 1.0 | 0.10 | `if (((int)threadIdx.x) < 4)` |
| wait.cp_async#6 | 0.06 | 0.3% | 1.0 | 0.06 | `tl::cp_async_wait<0>();` |
| wait.cp_async#3 | 0.05 | 0.3% | 1.0 | 0.06 | `tl::cp_async_wait<3>();` |
| wait.sync#8 | 0.05 | 0.3% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#9 | 0.05 | 0.3% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#11 | 0.05 | 0.3% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#6 | 0.05 | 0.3% | 1.0 | 0.06 | `__syncthreads();` |
| wait.cp_async#5 | 0.05 | 0.3% | 1.0 | 0.06 | `tl::cp_async_wait<1>();` |
| wait.cp_async#4 | 0.05 | 0.2% | 1.0 | 0.06 | `tl::cp_async_wait<2>();` |

### gemm_mxfp8_kernel grid (24, 1, 1) block (128, 1, 1)

153 launches, 96 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 31.94 us, warp 30.55 us, nsys graphs 39.04 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 12.38 | 40.5% | 8.31 | 25.2% |
| wait | 7.81 | 25.6% | 11.40 | 34.6% |
| compute | 6.02 | 19.7% | 6.14 | 18.6% |
| load | 4.29 | 14.0% | 7.00 | 21.3% |
| store | 0.05 | 0.2% | 0.08 | 0.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*38 (est.: sampled iterations x38) | 5.78 | 18.9% | 1.0 | 6.08 | `{` |
| load.cp_async#2*38 (est.: sampled iterations x38) | 4.10 | 13.4% | 1.0 | 4.86 | `for (int i_5 = 0; i_5 < 2; ++i_5) \| for (int i_5 = 0; i_5 < 4; ++i_5) \| tl::cp_async_gs_co` |
| wait.sync#2*38 (est.: sampled iterations x38) | 2.89 | 9.5% | 1.0 | 3.65 | `__syncthreads();` |
| wait.sync*38 (est.: sampled iterations x38) | 2.43 | 8.0% | 1.0 | 2.43 | `__syncthreads();` |
| wait.cp_async*38 (est.: sampled iterations x38) | 2.28 | 7.5% | 1.0 | 2.43 | `tl::cp_async_wait<2>();` |
| load.cp_async | 0.18 | 0.6% | 1.0 | 0.26 | `for (int i_1 = 0; i_1 < 2; ++i_1) \| for (int i_1 = 0; i_1 < 4; ++i_1) \| for (int i_1 = 0; ` |
| compute.mma#3 | 0.12 | 0.4% | 1.0 | 0.13 | `{` |
| compute.mma#2 | 0.12 | 0.4% | 1.0 | 0.13 | `{` |
| wait.cp_async#3 | 0.06 | 0.2% | 1.0 | 0.06 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>();` |
| store.global | 0.05 | 0.2% | 1.0 | 0.06 | `for (int i_10 = 0; i_10 < 16; ++i_10) \| for (int i_10 = 0; i_10 < 8; ++i_10) \| for (int i_` |
| wait.sync#4 | 0.05 | 0.2% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#3 | 0.05 | 0.2% | 1.0 | 0.06 | `__syncthreads();` |
| wait.cp_async#2 | 0.05 | 0.2% | 1.0 | 0.06 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |

### gemm_bf16_partials_kernel grid (6, 8, 1) block (256, 1, 1)

102 launches, 384 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 22.48 us, warp 8.15 us, nsys graphs 20.93 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 5.59 | 68.6% | 2.15 | 35.9% |
| wait | 1.44 | 17.7% | 2.36 | 39.4% |
| compute | 0.69 | 8.5% | 1.41 | 23.5% |
| load | 0.39 | 4.8% | 0.00 | 0.0% |
| store | 0.04 | 0.5% | 0.07 | 1.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.mbarrier#2*10 (est.: sampled iterations x10) | 0.70 | 8.6% | 0.5 | 2.56 | `mbarrier[(kb_1 % 3)].wait(((kb_1 % 6) / 3));` |
| compute.mma*10 (est.: sampled iterations x10) | 0.69 | 8.5% | 0.5 | 1.60 | `{` |
| wait.mbarrier*10 (est.: sampled iterations x10) | 0.64 | 7.9% | 0.5 | 1.60 | `mbarrier[((kb % 3) + 3)].wait((((kb % 6) / 3) ^ 1));` |
| load.tma*10 (est.: sampled iterations x10) | 0.39 | 4.8% | 0.5 | 1.28 | `if (tl::tl_shuffle_elect<128>())` |
| wait.sync | 0.10 | 1.2% | 1.0 | 0.13 | `__syncthreads();` |
| store.global | 0.04 | 0.5% | 0.5 | 0.10 | `for (int i_1 = 0; i_1 < 4; ++i_1) \| for (int i_2 = 0; i_2 < 8; ++i_2) \| for (int i_4 = 0; ` |

### mhc_finalize_kernel grid (5, 1, 1) block (128, 1, 1)

192 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 11.04 us, warp 10.08 us, nsys graphs 11.36 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 5.81 | 57.6% | 4.73 | 43.9% |
| store | 1.56 | 15.5% | 2.46 | 22.8% |
| load | 1.05 | 10.4% | 1.61 | 14.9% |
| compute | 1.00 | 9.9% | 1.19 | 11.0% |
| wait | 0.66 | 6.6% | 0.81 | 7.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.66 | 6.6% | 0.2 | 2.69 | `if (((int)threadIdx.x) < 4)` |
| wait.sync | 0.66 | 6.6% | 1.0 | 0.91 | `__syncthreads();` |
| compute.reduce | 0.57 | 5.7% | 0.2 | 2.30 | `row_max[0] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<32>>::run(row_max[0]);` |
| store.global#4*7 (est.: sampled iterations x7) | 0.56 | 5.6% | 0.8 | 0.90 | `if ((((int)threadIdx.x) % 4) == 0)` |
| compute.reduce#2*7 (est.: sampled iterations x7) | 0.43 | 4.2% | 0.8 | 0.67 | `amax[0] = max(amax[0], __shfl_xor_sync((uint)4294967295, amax[0], 1, 32));` |
| load.global#4*7 (est.: sampled iterations x7) | 0.36 | 3.6% | 0.8 | 0.67 | `*(uint4*)(yv + 0) = *(uint4*)(C + ((((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((i` |
| load.global | 0.36 | 3.6% | 1.0 | 1.09 | `if (((int)threadIdx.x) < 32)` |
| store.global#3*7 (est.: sampled iterations x7) | 0.32 | 3.2% | 0.8 | 0.45 | `*(uint4*)(y + ((((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((int64_t)step) * (int6` |
| load.global#3 | 0.28 | 2.8% | 0.8 | 0.38 | `for (int i_split_2 = 0; i_split_2 < 40; ++i_split_2)` |
| load.global#2 | 0.04 | 0.4% | 0.2 | 0.19 | `cm[0] = ((mixes_shared[((((int)threadIdx.x) & 15) + 8)] * scale[2]) + base[((((int)threadI` |
| store.global#2 | 0.01 | 0.1% | 0.2 | 0.06 | `if ((((int)threadIdx.x) >> 4) == 0)` |

### gemm_bf16_splitk_kernel grid (16, 5, 1) block (128, 1, 1)

34 launches, 320 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 55.34 us, warp 50.98 us, nsys graphs 51.89 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 25.01 | 49.1% | 0.00 | 0.0% |
| wait | 21.68 | 42.5% | 52.00 | 98.3% |
| load | 2.15 | 4.2% | 4.64 | 8.8% |
| compute | 2.01 | 3.9% | 2.36 | 4.5% |
| store | 0.13 | 0.3% | 0.16 | 0.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.cp_async*13 (est.: sampled iterations x13) | 7.60 | 14.9% | 0.5 | 16.43 | `tl::cp_async_wait<2>();` |
| wait.sync*13 (est.: sampled iterations x13) | 2.81 | 5.5% | 0.5 | 6.66 | `__syncthreads();` |
| wait.cp_async#2 | 2.06 | 4.0% | 1.0 | 4.83 | `tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>(); \| tl::cp_async_wait<3>();` |
| wait.cp_async*12 (est.: sampled iterations x12) | 1.97 | 3.9% | 0.5 | 8.64 | `tl::cp_async_wait<3>();` |
| wait.sync*12 (est.: sampled iterations x12) | 1.93 | 3.8% | 0.5 | 6.34 | `__syncthreads();` |
| wait.cp_async#4 | 1.07 | 2.1% | 1.0 | 3.25 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>();` |
| load.cp_async | 1.04 | 2.0% | 1.0 | 3.38 | `for (int i_1 = 0; i_1 < 2; ++i_1) \| tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)A_sha` |
| wait.cp_async#3 | 0.96 | 1.9% | 1.0 | 2.96 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| compute.mma*13 (est.: sampled iterations x13) | 0.83 | 1.6% | 0.5 | 0.83 | `{` |
| compute.mma*12 (est.: sampled iterations x12) | 0.82 | 1.6% | 0.5 | 0.96 | `{` |
| wait.sync#3 | 0.71 | 1.4% | 1.0 | 2.19 | `__syncthreads();` |
| wait.sync#5 | 0.67 | 1.3% | 1.0 | 2.13 | `__syncthreads();` |
| wait.sync#4 | 0.67 | 1.3% | 1.0 | 1.86 | `__syncthreads();` |
| load.cp_async#2*12 (est.: sampled iterations x12) | 0.50 | 1.0% | 0.5 | 0.58 | `tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)A_shared)[((((((kb & 3) * 1024) + ((((int` |
| load.cp_async#2*13 (est.: sampled iterations x13) | 0.42 | 0.8% | 0.5 | 0.42 | `for (int i_10 = 0; i_10 < 2; ++i_10) \| for (int i_8 = 0; i_8 < 2; ++i_8) \| tl::cp_async_gs` |
| wait.sync#2*13 (est.: sampled iterations x13) | 0.31 | 0.6% | 0.5 | 0.42 | `__syncthreads();` |
| wait.sync#2*12 (est.: sampled iterations x12) | 0.29 | 0.6% | 0.5 | 0.38 | `__syncthreads();` |
| wait.sync#7 | 0.19 | 0.4% | 1.0 | 0.27 | `__syncthreads();` |
| load.global | 0.18 | 0.4% | 1.0 | 0.70 | `if (((int)threadIdx.x) == 0)` |
| wait.sync#8 | 0.16 | 0.3% | 0.5 | 0.22 | `__syncthreads();` |
| wait.sync#6 | 0.16 | 0.3% | 1.0 | 0.50 | `__syncthreads();` |
| compute.mma#2 | 0.11 | 0.2% | 1.0 | 0.13 | `{` |
| wait.cp_async#5 | 0.11 | 0.2% | 0.5 | 0.48 | `tl::cp_async_wait<0>();` |
| compute.mma#3 | 0.10 | 0.2% | 1.0 | 0.13 | `{` |
| compute.mma#4 | 0.10 | 0.2% | 1.0 | 0.13 | `{` |
| store.global#3 | 0.07 | 0.1% | 1.0 | 0.10 | `if (((int*)last)[0] == 1)` |
| store.global | 0.06 | 0.1% | 1.0 | 0.06 | `for (int i_10 = 0; i_10 < 4; ++i_10) \| for (int i_16 = 0; i_16 < 4; ++i_16) \| for (int i_1` |
| compute.mma#5 | 0.05 | 0.1% | 0.5 | 0.05 | `{` |

### sparse_mla_kernel grid (5, 1, 5) block (256, 1, 1)

64 launches, 200 warps; ranges, warp and last warp from the 5 recording CTAs' 40 warps; span 18.83 us, warp 17.69 us, nsys graphs 16.02 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed, with loops' unsampled iterations | 10.97 | 62.0% | 11.02 | 49.4% |
| wait | 3.12 | 17.6% | 3.86 | 17.3% |
| compute | 1.48 | 8.3% | 1.49 | 6.7% |
| load | 1.46 | 8.2% | 5.06 | 22.7% |
| store | 0.67 | 3.8% | 0.89 | 4.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync (sampled iterations) | 3.05 | 17.2% | 11.0 | 5.10 | `__syncthreads();` |
| compute.mma (sampled iterations) | 1.22 | 6.9% | 2.0 | 1.28 | `for (int ki = 0; ki < 32; ++ki) \| for (int ki_1 = 0; ki_1 < 2; ++ki_1)` |
| load.global (sampled iterations) | 1.16 | 6.5% | 2.0 | 5.06 | `for (int c = 0; c < 4; ++c) \| for (int i = 0; i < 8; ++i) \| for (int i_1 = 0; i_1 < 8; ++i` |
| store.global | 0.67 | 3.8% | 1.0 | 0.83 | `for (int i_10 = 0; i_10 < 32; ++i_10) \| for (int i_11 = 0; i_11 < 32; ++i_11) \| for (int i` |
| load.cp_async (sampled iterations) | 0.30 | 1.7% | 2.0 | 0.42 | `for (int k_1 = 0; k_1 < 5; ++k_1) \| for (int k_1 = 0; k_1 < 9; ++k_1) \| for (int k_3 = 0; ` |
| compute.reduce (sampled iterations) | 0.25 | 1.4% | 2.0 | 0.26 | `m_g[i_5] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(m_g[i_5], (&((` |
| wait.cp_async (sampled iterations) | 0.07 | 0.4% | 1.0 | 0.10 | `tl::cp_async_wait<0>();` |

### mhc_finalize_kernel grid (10, 1, 1) block (128, 1, 1)

102 launches, 40 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 12.32 us, warp 10.79 us, nsys graphs 11.28 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 6.50 | 60.2% | 5.50 | 46.9% |
| store | 1.44 | 13.3% | 2.27 | 19.3% |
| load | 1.14 | 10.5% | 1.70 | 14.5% |
| compute | 1.00 | 9.3% | 1.06 | 9.0% |
| wait | 0.72 | 6.6% | 1.21 | 10.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync | 0.72 | 6.6% | 1.0 | 0.99 | `__syncthreads();` |
| compute.reduce | 0.57 | 5.3% | 0.2 | 2.30 | `row_max[0] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<32>>::run(row_max[0]);` |
| store.global#4*7 (est.: sampled iterations x7) | 0.56 | 5.2% | 0.8 | 0.90 | `if ((((int)threadIdx.x) % 4) == 0)` |
| store.global | 0.54 | 5.0% | 0.2 | 2.21 | `if (((int)threadIdx.x) < 4)` |
| load.global | 0.49 | 4.5% | 1.0 | 1.44 | `if (((int)threadIdx.x) < 32)` |
| compute.reduce#2*7 (est.: sampled iterations x7) | 0.43 | 3.9% | 0.8 | 0.67 | `amax[0] = max(amax[0], __shfl_xor_sync((uint)4294967295, amax[0], 1, 32));` |
| load.global#4*7 (est.: sampled iterations x7) | 0.36 | 3.3% | 0.8 | 0.67 | `*(uint4*)(yv + 0) = *(uint4*)(C + ((((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((i` |
| store.global#3*7 (est.: sampled iterations x7) | 0.32 | 3.0% | 0.8 | 0.45 | `*(uint4*)(y + ((((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((int64_t)step) * (int6` |
| load.global#3 | 0.25 | 2.3% | 0.8 | 0.35 | `for (int i_split_2 = 0; i_split_2 < 40; ++i_split_2)` |
| load.global#2 | 0.04 | 0.4% | 0.2 | 0.19 | `cm[0] = ((mixes_shared[((((int)threadIdx.x) & 15) + 8)] * scale[2]) + base[((((int)threadI` |
| store.global#2 | 0.01 | 0.1% | 0.2 | 0.06 | `if ((((int)threadIdx.x) >> 4) == 0)` |

### sparse_mla_kernel grid (2, 2, 10) block (128, 1, 1)

51 launches, 160 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 20.54 us, warp 18.01 us, nsys graphs 17.92 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed, with loops' unsampled iterations | 7.56 | 42.0% | 7.23 | 35.5% |
| compute | 4.69 | 26.0% | 4.69 | 23.1% |
| wait | 2.74 | 15.2% | 2.86 | 14.0% |
| load | 2.68 | 14.9% | 5.11 | 25.1% |
| store | 0.34 | 1.9% | 0.46 | 2.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*4 (est.: sampled iterations x4) | 1.73 | 9.6% | 1.0 | 1.79 | `for (int ki = 0; ki < 32; ++ki) \| {` |
| load.global#2 | 1.33 | 7.4% | 1.0 | 2.82 | `for (int i = 0; i < 8; ++i)` |
| compute.mma#2*4 (est.: sampled iterations x4) | 1.20 | 6.7% | 1.0 | 1.28 | `for (int ki_1 = 0; ki_1 < 2; ++ki_1) \| if (((((int)blockIdx.x) * 2) + (c_1 >> 1)) < 3)` |
| compute.reduce#2*8 (est.: sampled iterations x8) | 0.88 | 4.9% | 1.0 | 1.02 | `row_sum[i_7] = tl::AllReduce<tl::SumOp, 128, 32, 0, tl::NamedBarrier<128>>::run(row_sum[i_` |
| compute.reduce*8 (est.: sampled iterations x8) | 0.88 | 4.9% | 1.0 | 1.02 | `m_g[i_4] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(m_g[i_4], (&((` |
| load.cp_async#2*4 (est.: sampled iterations x4) | 0.70 | 3.9% | 1.0 | 0.77 | `if (((((int)blockIdx.x) * 2) + (c_1 >> 1)) < 3) \| if ((c_1 < 3) & (c_1 < 3)) \| if (c < 3)` |
| load.cp_async (sampled iterations) | 0.50 | 2.8% | 1.0 | 0.54 | `for (int k_1 = 0; k_1 < 5; ++k_1) \| for (int k_1 = 0; k_1 < 9; ++k_1) \| for (int k_3 = 0; ` |
| store.global | 0.34 | 1.9% | 1.0 | 0.48 | `for (int i_10 = 0; i_10 < 32; ++i_10) \| for (int i_11 = 0; i_11 < 32; ++i_11) \| for (int i` |
| wait.sync#10*8 (est.: sampled iterations x8) | 0.34 | 1.9% | 1.0 | 0.51 | `__syncthreads();` |
| wait.sync#8*8 (est.: sampled iterations x8) | 0.34 | 1.9% | 1.0 | 0.51 | `__syncthreads();` |
| wait.cp_async*4 (est.: sampled iterations x4) | 0.26 | 1.5% | 1.0 | 0.38 | `tl::cp_async_wait<0>();` |
| wait.sync#5*4 (est.: sampled iterations x4) | 0.26 | 1.4% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#7*4 (est.: sampled iterations x4) | 0.26 | 1.4% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#9*4 (est.: sampled iterations x4) | 0.26 | 1.4% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#11*4 (est.: sampled iterations x4) | 0.25 | 1.4% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#4*4 (est.: sampled iterations x4) | 0.25 | 1.4% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#6*4 (est.: sampled iterations x4) | 0.25 | 1.4% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync (sampled iterations) | 0.18 | 1.0% | 1.0 | 0.26 | `__syncthreads();` |
| load.global (sampled iterations) | 0.14 | 0.8% | 1.0 | 0.29 | `for (int c = 0; c < 4; ++c) \| for (int i = 0; i < 8; ++i) \| for (int i_1 = 0; i_1 < 8; ++i` |
| wait.sync#3 | 0.06 | 0.3% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#2 | 0.05 | 0.3% | 1.0 | 0.06 | `__syncthreads();` |

### rope_kernel grid (1, 1, 1) block (128, 1, 1)

272 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 1.39 us, warp 0.96 us, nsys graphs 1.38 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.79 | 82.5% | 2.64 | 86.1% |
| load | 0.15 | 15.8% | 0.36 | 11.8% |
| store | 0.02 | 1.7% | 0.06 | 2.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.10 | 10.0% | 1.0 | 0.10 | `if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < tokens)) \| int64_t position ` |
| load.global#2 | 0.06 | 5.8% | 0.5 | 0.10 | `if (((position < ((int64_t)positions_max)) && ((int64_t)0 <= position)))` |
| store.global | 0.02 | 1.7% | 0.5 | 0.03 | `*(uint4*)(x + ((((((int64_t)((int)blockIdx.x)) * (int64_t)12288) + ((((int64_t)((int)threa` |

### gemm_mxfp8_kernel grid (64, 1, 1) block (128, 1, 1)

34 launches, 256 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 23.47 us, warp 14.98 us, nsys graphs 24.06 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 10.13 | 67.6% | 2.75 | 32.9% |
| unattributed | 3.25 | 21.7% | 3.32 | 39.6% |
| compute | 0.88 | 5.9% | 0.88 | 10.5% |
| load | 0.66 | 4.4% | 1.38 | 16.4% |
| store | 0.06 | 0.4% | 0.06 | 0.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.cp_async*2 (est.: sampled iterations x2) | 2.84 | 19.0% | 1.0 | 6.34 | `tl::cp_async_wait<3>();` |
| wait.cp_async#3 | 1.62 | 10.8% | 1.0 | 3.92 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>();` |
| wait.cp_async#4 | 1.29 | 8.6% | 1.0 | 3.01 | `tl::cp_async_wait<0>();` |
| wait.sync#2*2 (est.: sampled iterations x2) | 1.16 | 7.8% | 1.0 | 4.19 | `__syncthreads();` |
| wait.cp_async#2 | 1.16 | 7.7% | 1.0 | 3.15 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| wait.sync#4 | 0.78 | 5.2% | 1.0 | 2.51 | `__syncthreads();` |
| wait.sync#3 | 0.66 | 4.4% | 1.0 | 2.37 | `__syncthreads();` |
| wait.sync#5 | 0.52 | 3.5% | 1.0 | 1.70 | `__syncthreads();` |
| load.cp_async | 0.40 | 2.7% | 1.0 | 0.51 | `for (int i_1 = 0; i_1 < 2; ++i_1) \| for (int i_1 = 0; i_1 < 4; ++i_1) \| for (int i_1 = 0; ` |
| compute.mma*2 (est.: sampled iterations x2) | 0.36 | 2.4% | 1.0 | 0.38 | `{` |
| load.cp_async#2*2 (est.: sampled iterations x2) | 0.26 | 1.7% | 1.0 | 0.51 | `for (int i_7 = 0; i_7 < 2; ++i_7)` |
| compute.mma#2 | 0.18 | 1.2% | 1.0 | 0.19 | `{` |
| compute.mma#3 | 0.18 | 1.2% | 1.0 | 0.19 | `{` |
| compute.mma#4 | 0.18 | 1.2% | 1.0 | 0.19 | `{` |
| wait.sync*2 (est.: sampled iterations x2) | 0.10 | 0.6% | 1.0 | 0.13 | `__syncthreads();` |
| store.global | 0.06 | 0.4% | 1.0 | 0.06 | `for (int i_10 = 0; i_10 < 16; ++i_10) \| for (int i_10 = 0; i_10 < 8; ++i_10) \| for (int i_` |

### cache_writer_kernel grid (5, 1, 1) block (128, 1, 1)

128 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 5.89 us, warp 5.86 us, nsys graphs 4.37 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 4.01 | 68.5% | 3.36 | 54.9% |
| store | 1.48 | 25.2% | 2.00 | 32.7% |
| load | 0.30 | 5.1% | 0.68 | 11.2% |
| compute | 0.07 | 1.3% | 0.08 | 1.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 1.48 | 25.2% | 1.0 | 1.65 | `if (((int64_t)0 <= slot) & ((slot >> (int64_t)7) < ((int64_t)pages))) \| if (((int64_t)0 <=` |
| load.global | 0.24 | 4.0% | 1.0 | 0.26 | `for (int i = 0; i < 4; ++i)` |
| compute.reduce | 0.07 | 1.3% | 1.0 | 0.10 | `amax = max(amax, __shfl_xor_sync((uint)4294967295, amax, 1, 32));` |
| load.global#2 | 0.06 | 1.0% | 1.0 | 0.06 | `int64_t slot = slots[((int64_t)((int)blockIdx.x))];` |

### moe_dispatch_kernel grid (1, 1, 1) block (512, 1, 1)

153 launches, 16 warps; ranges, warp and last warp from the 1 recording CTAs' 16 warps; span 4.80 us, warp 4.55 us, nsys graphs 2.78 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 2.57 | 56.4% | 1.73 | 35.7% |
| unattributed, with loops' unsampled iterations | 1.29 | 28.5% | 1.75 | 36.2% |
| store | 0.60 | 13.3% | 1.24 | 25.7% |
| load | 0.08 | 1.8% | 0.11 | 2.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync#7*9 (est.: sampled iterations x9) | 0.67 | 14.7% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#6*9 (est.: sampled iterations x9) | 0.61 | 13.5% | 1.0 | 0.86 | `__syncthreads();` |
| store.global (sampled iterations) | 0.53 | 11.6% | 1.0 | 0.96 | `ChunkCount[((((int64_t)c) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = count; \| Chu` |
| wait.sync#8 | 0.50 | 11.1% | 1.0 | 0.80 | `__syncthreads();` |
| wait.sync#2 | 0.39 | 8.7% | 1.0 | 0.42 | `__syncthreads();` |
| wait.sync#3 | 0.13 | 2.9% | 1.0 | 0.32 | `__syncthreads();` |
| load.global (sampled iterations) | 0.08 | 1.8% | 1.0 | 0.10 | `expert = ((int)TopkIdx[((((int64_t)((int)blockIdx.x)) * (int64_t)512) + ((int64_t)((int)th` |
| store.global#2 | 0.08 | 1.7% | 1.0 | 0.29 | `RoutePos[((((int64_t)((int)blockIdx.x)) * (int64_t)512) + ((int64_t)((int)threadIdx.x)))] ` |
| wait.sync | 0.07 | 1.5% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#4 | 0.06 | 1.4% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#5 | 0.06 | 1.4% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#9 | 0.06 | 1.4% | 1.0 | 0.06 | `__syncthreads();` |

### sparse_mla_kernel grid (1, 2, 5) block (128, 1, 1)

32 launches, 40 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 23.89 us, warp 23.84 us, nsys graphs 21.30 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed, with loops' unsampled iterations | 8.87 | 37.2% | 7.99 | 34.7% |
| load | 5.19 | 21.8% | 7.31 | 31.7% |
| compute | 4.72 | 19.8% | 4.72 | 20.5% |
| wait | 4.62 | 19.4% | 2.53 | 11.0% |
| store | 0.43 | 1.8% | 0.50 | 2.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 | 3.38 | 14.2% | 1.0 | 3.81 | `for (int i = 0; i < 8; ++i)` |
| wait.sync (sampled iterations) | 2.11 | 8.9% | 1.0 | 2.82 | `__syncthreads();` |
| compute.mma*4 (est.: sampled iterations x4) | 1.73 | 7.2% | 1.0 | 1.79 | `for (int ki = 0; ki < 32; ++ki) \| {` |
| compute.mma#2*4 (est.: sampled iterations x4) | 1.16 | 4.9% | 1.0 | 1.28 | `for (int ki_1 = 0; ki_1 < 2; ++ki_1) \| if (((((int)blockIdx.x) * 2) + (c_1 >> 1)) < 3)` |
| compute.reduce#2*8 (est.: sampled iterations x8) | 0.92 | 3.8% | 1.0 | 1.02 | `row_sum[i_7] = tl::AllReduce<tl::SumOp, 128, 32, 0, tl::NamedBarrier<128>>::run(row_sum[i_` |
| compute.reduce*8 (est.: sampled iterations x8) | 0.92 | 3.8% | 1.0 | 1.02 | `m_g[i_4] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(m_g[i_4], (&((` |
| load.global (sampled iterations) | 0.81 | 3.4% | 1.0 | 2.85 | `for (int c = 0; c < 4; ++c) \| for (int i = 0; i < 8; ++i) \| for (int i_1 = 0; i_1 < 8; ++i` |
| load.cp_async#2*4 (est.: sampled iterations x4) | 0.66 | 2.8% | 1.0 | 0.77 | `if (((((int)blockIdx.x) * 2) + (c_1 >> 1)) < 3) \| if ((c_1 < 3) & (c_1 < 3)) \| if (c < 3)` |
| store.global | 0.43 | 1.8% | 1.0 | 0.51 | `for (int i_10 = 0; i_10 < 32; ++i_10) \| for (int i_11 = 0; i_11 < 32; ++i_11) \| for (int i` |
| load.cp_async (sampled iterations) | 0.34 | 1.4% | 1.0 | 0.37 | `for (int k_1 = 0; k_1 < 5; ++k_1) \| for (int k_1 = 0; k_1 < 9; ++k_1) \| for (int k_3 = 0; ` |
| wait.sync#10*8 (est.: sampled iterations x8) | 0.32 | 1.3% | 1.0 | 0.51 | `__syncthreads();` |
| wait.sync#8*8 (est.: sampled iterations x8) | 0.29 | 1.2% | 1.0 | 0.51 | `__syncthreads();` |
| wait.cp_async*4 (est.: sampled iterations x4) | 0.27 | 1.1% | 1.0 | 0.38 | `tl::cp_async_wait<0>();` |
| wait.sync#5*4 (est.: sampled iterations x4) | 0.26 | 1.1% | 1.0 | 0.38 | `__syncthreads();` |
| wait.sync#4*4 (est.: sampled iterations x4) | 0.26 | 1.1% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#6*4 (est.: sampled iterations x4) | 0.26 | 1.1% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#7*4 (est.: sampled iterations x4) | 0.26 | 1.1% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#9*4 (est.: sampled iterations x4) | 0.26 | 1.1% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#11*4 (est.: sampled iterations x4) | 0.24 | 1.0% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#3 | 0.06 | 0.2% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#2 | 0.04 | 0.2% | 1.0 | 0.06 | `__syncthreads();` |

### metadata_chunk_kernel grid (10, 1, 1) block (192, 1, 1)

51 launches, 60 warps; ranges, warp and last warp from the 5 recording CTAs' 30 warps; span 13.44 us, warp 8.68 us, nsys graphs 12.70 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 5.62 | 64.7% | 11.15 | 78.8% |
| unattributed | 2.99 | 34.4% | 2.85 | 20.1% |
| store | 0.07 | 0.8% | 0.14 | 1.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 5.49 | 63.3% | 1.0 | 8.99 | `if ((((offset + ((int)blockIdx.x)) < src_rows) && (0 <= (offset + ((int)blockIdx.x)))))` |
| load.global#2 | 0.12 | 1.4% | 0.2 | 1.22 | `if (((0 <= req) & ((int64_t)0 <= pos)))` |
| store.global | 0.05 | 0.6% | 1.0 | 0.10 | `swa[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = con` |
| store.global#2 | 0.01 | 0.1% | 0.2 | 0.06 | `lengths[((int64_t)((int)blockIdx.x))] = ((int)max(info[1], (int64_t)0));` |
| store.global#3 | 0.01 | 0.1% | 0.2 | 0.06 | `top_lengths[((int64_t)((int)blockIdx.x))] = condval_12; \| top_lengths[((int64_t)((int)bloc` |

### moe_grouped_fp8_fp4_gemm_kernel grid (12, 36, 1) block (128, 1, 1)

6 launches, 1728 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 113.34 us, warp 18.27 us, nsys graphs 3.52 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 7.84 | 42.9% | 0.21 | 66.1% |
| wait | 5.29 | 29.0% | 0.05 | 16.9% |
| compute | 3.24 | 17.7% | 0.00 | 0.0% |
| load | 1.83 | 10.0% | 0.05 | 16.9% |
| store | 0.07 | 0.4% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*40 (est.: sampled iterations x40) | 3.24 | 17.7% | 0.2 | 13.44 | `{` |
| wait.mbarrier*40 (est.: sampled iterations x40) | 2.08 | 11.4% | 0.2 | 8.32 | `loaded[(ko % 3)].wait(((ko % 6) / 3));` |
| load#2*40 (est.: sampled iterations x40) | 1.52 | 8.3% | 0.2 | 8.96 | `if (ko < 38)` |
| wait.sync#5*40 (est.: sampled iterations x40) | 1.24 | 6.8% | 0.2 | 5.76 | `__syncthreads();` |
| wait.sync#4*40 (est.: sampled iterations x40) | 0.64 | 3.5% | 0.2 | 2.56 | `__syncthreads();` |
| wait.sync#6*40 (est.: sampled iterations x40) | 0.64 | 3.5% | 0.2 | 2.56 | `__syncthreads();` |
| wait.cp_async*40 (est.: sampled iterations x40) | 0.60 | 3.3% | 0.2 | 2.56 | `tl::cp_async_wait<2>();` |
| load.global | 0.12 | 0.7% | 1.0 | 0.32 | `int expert = TileExpert[((int64_t)((int)blockIdx.y))];` |
| load.global#3 | 0.12 | 0.6% | 0.2 | 0.46 | `if (((expert < 128) && (0 <= expert))) \| if (((expert < 384) && (0 <= expert)))` |
| store.global | 0.07 | 0.4% | 0.2 | 0.35 | `for (int i_2 = 0; i_2 < 16; ++i_2) \| for (int i_4 = 0; i_4 < 32; ++i_4)` |
| wait.sync | 0.05 | 0.3% | 1.0 | 0.06 | `__syncthreads();` |
| load | 0.04 | 0.2% | 0.2 | 0.22 | `for (int s = 0; s < 2; ++s)` |
| wait.sync#2 | 0.03 | 0.2% | 0.2 | 0.16 | `__syncthreads();` |
| load.global#2 | 0.02 | 0.1% | 0.2 | 0.16 | `if (((int)threadIdx.x) < 32) \| if (((int)threadIdx.x) < 64)` |
| wait.sync#3 | 0.01 | 0.1% | 0.2 | 0.06 | `__syncthreads();` |

### gemm_mxfp8_partials_kernel grid (16, 5, 1) block (128, 1, 1)

51 launches, 320 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 12.03 us, warp 11.26 us, nsys graphs 12.67 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 5.55 | 49.2% | 6.80 | 54.7% |
| wait | 4.59 | 40.8% | 3.07 | 24.7% |
| compute | 0.61 | 5.4% | 0.64 | 5.1% |
| load | 0.46 | 4.1% | 1.88 | 15.1% |
| store | 0.05 | 0.5% | 0.05 | 0.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.cp_async#3 | 1.61 | 14.3% | 1.0 | 3.58 | `tl::cp_async_wait<0>();` |
| wait.cp_async#2 | 1.01 | 9.0% | 1.0 | 2.18 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>();` |
| wait.sync#2*2 (est.: sampled iterations x2) | 0.74 | 6.5% | 1.0 | 2.30 | `__syncthreads();` |
| wait.sync#3 | 0.48 | 4.2% | 1.0 | 1.38 | `__syncthreads();` |
| wait.sync#4 | 0.34 | 3.1% | 1.0 | 1.25 | `__syncthreads();` |
| wait.cp_async*2 (est.: sampled iterations x2) | 0.32 | 2.9% | 1.0 | 2.24 | `tl::cp_async_wait<2>();` |
| compute.mma*2 (est.: sampled iterations x2) | 0.31 | 2.7% | 1.0 | 0.32 | `{` |
| load.cp_async | 0.24 | 2.1% | 1.0 | 0.32 | `for (int i = 0; i < 2; ++i) \| for (int i_1 = 0; i_1 < 4; ++i_1)` |
| load.cp_async#2*2 (est.: sampled iterations x2) | 0.22 | 2.0% | 1.0 | 0.26 | `for (int i_4 = 0; i_4 < 2; ++i_4)` |
| compute.mma#3 | 0.15 | 1.3% | 1.0 | 0.16 | `{` |
| compute.mma#2 | 0.15 | 1.3% | 1.0 | 0.16 | `{` |
| wait.sync*2 (est.: sampled iterations x2) | 0.09 | 0.8% | 1.0 | 0.13 | `__syncthreads();` |
| store.global | 0.05 | 0.5% | 1.0 | 0.06 | `for (int i_11 = 0; i_11 < 16; ++i_11) \| for (int i_13 = 0; i_13 < 8; ++i_13) \| for (int i_` |

### partial_argmax_kernel grid (2, 64, 1) block (128, 1, 1)

85 launches, 512 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 7.01 us, warp 5.94 us, nsys graphs 7.30 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 5.18 | 87.1% | 5.80 | 83.7% |
| wait | 0.30 | 5.0% | 0.24 | 3.5% |
| load | 0.27 | 4.5% | 0.63 | 9.1% |
| compute | 0.12 | 2.1% | 0.12 | 1.8% |
| store | 0.07 | 1.2% | 0.12 | 1.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync | 0.30 | 5.0% | 1.0 | 1.12 | `__syncthreads();` |
| load.global | 0.27 | 4.5% | 1.0 | 0.42 | `for (int j = 0; j < 16; ++j) \| for (int run = 0; run < 2; ++run)` |
| compute.reduce | 0.12 | 2.1% | 1.0 | 0.13 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| store.global | 0.07 | 1.2% | 1.0 | 0.13 | `if (((int)threadIdx.x) == 0)` |

### router_partials_kernel grid (3, 1, 1) block (64, 1, 1)

102 launches, 6 warps; ranges, warp and last warp from the 3 recording CTAs' 6 warps; span 5.02 us, warp 3.02 us, nsys graphs 4.38 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.21 | 39.9% | 2.13 | 37.2% |
| load | 1.00 | 33.2% | 2.03 | 35.5% |
| compute | 0.68 | 22.6% | 1.37 | 23.9% |
| store | 0.13 | 4.4% | 0.20 | 3.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.reduce | 0.68 | 22.6% | 0.5 | 1.38 | `for (int k = 0; k < 3; ++k) \| for (int k = 0; k < 6; ++k)` |
| load.global#4 | 0.49 | 16.2% | 0.5 | 1.02 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < image_len)) \| if ((((((int)bl` |
| load.global | 0.41 | 13.4% | 0.8 | 0.51 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < mask_len))` |
| load.global#2 | 0.05 | 1.8% | 0.8 | 0.06 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < mask_len))` |
| load.global#3 | 0.05 | 1.8% | 0.8 | 0.06 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < mask_len))` |
| store.global#3 | 0.05 | 1.6% | 0.5 | 0.10 | `if ((((int)threadIdx.x) & 31) < 3) \| if ((((int)threadIdx.x) & 31) < 6)` |
| store.global | 0.04 | 1.4% | 0.8 | 0.06 | `if (condval == (bool)0) \| if (condval == (bool)1)` |
| store.global#2 | 0.04 | 1.4% | 0.8 | 0.06 | `if (condval_1 == (bool)0) \| if (condval_1 == (bool)1)` |

### moe_swiglu_cast_kernel_kernel grid (16, 30, 1) block (96, 1, 1)

147 launches, 1440 warps; ranges, warp and last warp from the 4 recording CTAs' 12 warps; span 3.81 us, warp 1.21 us, nsys graphs 4.51 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.91 | 75.7% | 2.30 | 78.1% |
| load | 0.20 | 16.8% | 0.41 | 13.9% |
| compute | 0.05 | 4.0% | 0.12 | 4.1% |
| store | 0.04 | 3.5% | 0.11 | 3.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.20 | 16.8% | 1.0 | 0.26 | `for (int r = 0; r < 2; ++r)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.05 | 4.0% | 0.5 | 0.13 | `if (0 <= orders[r_2])` |
| store.global*2 (est.: sampled iterations x2) | 0.04 | 3.5% | 0.5 | 0.13 | `if (0 <= orders[r_2])` |

### norm_forward_and_per_token_cast_kernel_kernel grid (1, 1, 1) block (128, 1, 1)

187 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 2.75 us, warp 2.75 us, nsys graphs 1.28 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 2.10 | 76.5% | 2.39 | 81.9% |
| load | 0.38 | 14.0% | 0.43 | 14.8% |
| compute | 0.11 | 4.1% | 0.10 | 3.6% |
| wait | 0.10 | 3.8% | 0.06 | 1.9% |
| unattributed | 0.05 | 1.7% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*4 (est.: sampled iterations x4) | 1.54 | 55.8% | 1.0 | 1.54 | `for (int j_2 = 0; j_2 < 2; ++j_2)` |
| store.global*4 (est.: sampled iterations x4) | 0.35 | 12.8% | 1.0 | 0.38 | `for (int j_1 = 0; j_1 < 4; ++j_1)` |
| load.global*4 (est.: sampled iterations x4) | 0.22 | 8.1% | 1.0 | 0.26 | `if (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 4)) < num_tokens)` |
| load.global#3*4 (est.: sampled iterations x4) | 0.13 | 4.7% | 1.0 | 0.13 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 128) + ((((int)threadIdx.x) & 15` |
| store.global#5*4 (est.: sampled iterations x4) | 0.13 | 4.7% | 0.8 | 0.13 | `*(uint4*)(out + ((((((int64_t)vi_1) * (int64_t)128) + ((((int64_t)((int)threadIdx.x)) & (i` |
| compute.reduce | 0.11 | 4.1% | 1.0 | 0.13 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| wait.sync | 0.10 | 3.8% | 1.0 | 0.29 | `__syncthreads();` |
| store.global#2 | 0.05 | 1.7% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| store.global#3 | 0.04 | 1.5% | 0.8 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |
| load.global#2 | 0.03 | 1.2% | 1.0 | 0.03 | `*(float2*)(row_partial + 0) = *(float2*)(sum_shared + ((((int)threadIdx.x) >> 6) * 2)); \| ` |
| load.global#3*2 (est.: sampled iterations x2) | 0.00 | 0.0% | 0.0 | 0.00 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 64) + ((((int)threadIdx.x) & 7) ` |
| load.global*2 (est.: sampled iterations x2) | 0.00 | 0.0% | 0.0 | 0.00 | `if (((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < num_tokens)` |
| store.global#4*2 (est.: sampled iterations x2) | 0.00 | 0.0% | 0.0 | 0.00 | `for (int j_2 = 0; j_2 < 2; ++j_2)` |
| store.global#5*2 (est.: sampled iterations x2) | 0.00 | 0.0% | 0.0 | 0.00 | `*(uint4*)(out + ((((((int64_t)vi_1) * (int64_t)64) + ((((int64_t)((int)threadIdx.x)) & (in` |
| store.global*2 (est.: sampled iterations x2) | 0.00 | 0.0% | 0.0 | 0.00 | `for (int j_1 = 0; j_1 < 4; ++j_1)` |

### gemm_bf16_partials_kernel grid (2, 20, 1) block (256, 1, 1)

51 launches, 320 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 7.90 us, warp 4.11 us, nsys graphs 11.92 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 2.78 | 67.5% | 1.38 | 54.9% |
| wait | 0.56 | 13.5% | 0.61 | 24.4% |
| load | 0.50 | 12.3% | 0.00 | 0.0% |
| compute | 0.23 | 5.5% | 0.44 | 17.7% |
| store | 0.05 | 1.1% | 0.08 | 3.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.tma*4 (est.: sampled iterations x4) | 0.50 | 12.3% | 0.5 | 1.41 | `if (tl::tl_shuffle_elect<128>())` |
| wait.mbarrier#2*4 (est.: sampled iterations x4) | 0.32 | 7.9% | 0.5 | 1.28 | `mbarrier[(kb_1 % 3)].wait((kb_1 / 3));` |
| compute.mma*4 (est.: sampled iterations x4) | 0.23 | 5.5% | 0.5 | 0.51 | `{` |
| wait.mbarrier*4 (est.: sampled iterations x4) | 0.12 | 3.0% | 0.5 | 0.26 | `mbarrier[((kb % 3) + 3)].wait(((kb / 3) ^ 1));` |
| wait.sync | 0.11 | 2.6% | 1.0 | 0.13 | `__syncthreads();` |
| store.global | 0.05 | 1.1% | 0.5 | 0.13 | `for (int i_1 = 0; i_1 < 4; ++i_1) \| for (int i_2 = 0; i_2 < 8; ++i_2) \| for (int i_4 = 0; ` |

### token_ranks_kernel grid (1, 1, 1) block (256, 1, 1)

51 launches, 8 warps; ranges, warp and last warp from the 1 recording CTAs' 8 warps; span 7.62 us, warp 7.59 us, nsys graphs 8.03 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 6.88 | 90.7% | 7.19 | 90.3% |
| unattributed | 0.35 | 4.6% | 0.37 | 4.6% |
| compute | 0.31 | 4.1% | 0.36 | 4.5% |
| store | 0.05 | 0.6% | 0.05 | 0.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 6.88 | 90.7% | 1.0 | 7.01 | `if ((((int64_t)0 <= token_ids[((int64_t)((int)blockIdx.x))]) && (token_ids[((int64_t)((int` |
| compute.reduce | 0.31 | 4.1% | 1.0 | 0.42 | `total[0] = tl::AllReduce<tl::SumOp, 256, 1, 0, tl::NamedBarrier<256>>::run(total[0], (&(wo` |
| store.global | 0.05 | 0.6% | 1.0 | 0.06 | `output[((int64_t)((int)blockIdx.x))] = ((int64_t)total[0]);` |

### metadata_builder_kernel grid (2, 1, 1) block (128, 1, 1)

289 launches, 8 warps; ranges, warp and last warp from the 2 recording CTAs' 8 warps; span 1.22 us, warp 0.67 us, nsys graphs 1.47 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.30 | 44.3% | 0.34 | 24.8% |
| load | 0.29 | 43.7% | 0.97 | 70.8% |
| store | 0.08 | 12.0% | 0.06 | 4.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#3 | 0.25 | 37.1% | 0.5 | 0.83 | `for (int __1 = 0; __1 < 2; ++__1)` |
| store.global#4 | 0.04 | 5.4% | 0.5 | 0.10 | `positions[(((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))` |
| load.global#2 | 0.03 | 4.2% | 0.1 | 0.22 | `if ((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) < nr))` |
| store.global#3 | 0.03 | 4.2% | 0.5 | 0.06 | `if (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) == 0)` |
| load.global | 0.02 | 2.4% | 0.1 | 0.13 | `if ((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) <= nr))` |
| store.global | 0.01 | 1.8% | 0.1 | 0.10 | `out_starts[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] = condval;` |
| store.global#2 | 0.00 | 0.6% | 0.1 | 0.03 | `request_positions[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] = condval_2;` |

### metadata_decode_kernel grid (5, 2, 1) block (128, 1, 1)

64 launches, 40 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 3.09 us, warp 2.46 us, nsys graphs 5.92 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.31 | 53.3% | 4.43 | 72.0% |
| unattributed | 1.05 | 42.8% | 1.12 | 18.2% |
| store | 0.09 | 3.8% | 0.61 | 9.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.06 | 43.0% | 0.6 | 2.00 | `if ((((offset + ((int)blockIdx.x)) < src_rows) && (0 <= (offset + ((int)blockIdx.x)))))` |
| load.global#3 | 0.22 | 9.1% | 0.4 | 0.61 | `if ((((offset + ((int)blockIdx.x)) < main_src_rows) && (0 <= (offset + ((int)blockIdx.x)))` |
| store.global#4 | 0.04 | 1.8% | 0.4 | 0.22 | `if (0 < live)` |
| store.global | 0.04 | 1.4% | 0.6 | 0.06 | `swa[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = con` |
| load.global#2 | 0.03 | 1.2% | 0.1 | 0.19 | `if (((0 <= req) & ((int64_t)0 <= pos)))` |
| store.global#2 | 0.01 | 0.3% | 0.1 | 0.06 | `lengths[((int64_t)((int)blockIdx.x))] = ((int)max(info[1], (int64_t)0));` |
| store.global#3 | 0.01 | 0.3% | 0.1 | 0.06 | `top_lengths[((int64_t)((int)blockIdx.x))] = condval_6; \| top_lengths[((int64_t)((int)block` |

### moe_grouped_fp8_fp4_gemm_kernel grid (40, 36, 1) block (128, 1, 1)

6 launches, 5760 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 61.82 us, warp 3.51 us, nsys graphs 9.90 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.57 | 44.7% | 0.21 | 65.0% |
| wait | 0.90 | 25.6% | 0.05 | 16.7% |
| compute | 0.51 | 14.5% | 0.00 | 0.0% |
| load | 0.46 | 13.2% | 0.06 | 18.3% |
| store | 0.07 | 2.1% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*6 (est.: sampled iterations x6) | 0.51 | 14.5% | 0.2 | 2.11 | `{` |
| wait.sync#5*6 (est.: sampled iterations x6) | 0.31 | 8.9% | 0.2 | 1.54 | `__syncthreads();` |
| load#2*6 (est.: sampled iterations x6) | 0.21 | 6.0% | 0.2 | 1.06 | `if (ko < 4)` |
| wait.mbarrier*6 (est.: sampled iterations x6) | 0.17 | 4.8% | 0.2 | 0.77 | `loaded[(ko % 3)].wait((ko / 3));` |
| wait.cp_async*6 (est.: sampled iterations x6) | 0.14 | 3.9% | 0.2 | 1.15 | `tl::cp_async_wait<2>();` |
| load.global#3 | 0.12 | 3.4% | 0.2 | 0.48 | `if (((expert < 128) && (0 <= expert))) \| if (((expert < 384) && (0 <= expert)))` |
| wait.sync#4*6 (est.: sampled iterations x6) | 0.10 | 2.7% | 0.2 | 0.38 | `__syncthreads();` |
| wait.sync#6*6 (est.: sampled iterations x6) | 0.10 | 2.7% | 0.2 | 0.38 | `__syncthreads();` |
| store.global | 0.07 | 2.1% | 0.2 | 0.35 | `for (int i_2 = 0; i_2 < 16; ++i_2) \| for (int i_4 = 0; i_4 < 32; ++i_4)` |
| load.global | 0.06 | 1.8% | 1.0 | 0.10 | `int expert = TileExpert[((int64_t)((int)blockIdx.y))];` |
| wait.sync | 0.05 | 1.3% | 1.0 | 0.06 | `__syncthreads();` |
| load | 0.04 | 1.3% | 0.2 | 0.24 | `for (int s = 0; s < 2; ++s)` |
| wait.sync#2 | 0.03 | 0.9% | 0.2 | 0.16 | `__syncthreads();` |
| load.global#2 | 0.02 | 0.7% | 0.2 | 0.19 | `if (((int)threadIdx.x) < 32) \| if (((int)threadIdx.x) < 64)` |
| wait.sync#3 | 0.01 | 0.2% | 0.2 | 0.03 | `__syncthreads();` |

### cache_writer_kernel grid (10, 1, 1) block (128, 1, 1)

51 launches, 40 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 5.60 us, warp 5.53 us, nsys graphs 5.09 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 2.81 | 50.8% | 3.93 | 54.9% |
| store | 2.35 | 42.6% | 2.43 | 34.0% |
| load | 0.29 | 5.3% | 0.72 | 10.0% |
| compute | 0.08 | 1.4% | 0.07 | 1.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 2.35 | 42.6% | 1.0 | 2.59 | `if (((int64_t)0 <= slot) & ((slot >> (int64_t)7) < ((int64_t)pages))) \| if (((int64_t)0 <=` |
| load.global | 0.24 | 4.3% | 1.0 | 0.26 | `for (int i = 0; i < 4; ++i)` |
| compute.reduce | 0.08 | 1.4% | 1.0 | 0.10 | `amax = max(amax, __shfl_xor_sync((uint)4294967295, amax, 1, 32));` |
| load.global#2 | 0.06 | 1.0% | 1.0 | 0.06 | `int64_t slot = slots[((int64_t)((int)blockIdx.x))];` |

### topk_ids_kernel grid (5, 1, 1) block (512, 1, 1)

16 launches, 80 warps; ranges, warp and last warp from the 5 recording CTAs' 80 warps; span 21.25 us, warp 19.13 us, nsys graphs 20.62 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 15.41 | 80.6% | 16.65 | 73.8% |
| unattributed | 2.31 | 12.1% | 1.95 | 8.6% |
| wait | 0.95 | 5.0% | 0.75 | 3.3% |
| store | 0.46 | 2.4% | 3.21 | 14.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 15.41 | 80.6% | 1.0 | 16.46 | `for (int t = 0; t < (vocab >> 14); ++t)` |
| wait.sync | 0.81 | 4.2% | 1.0 | 1.66 | `__syncthreads();` |
| store.global | 0.46 | 2.4% | 1.0 | 3.09 | `if ((((int)threadIdx.x) >> 5) == 0)` |
| wait.sync#2 | 0.08 | 0.4% | 1.0 | 0.18 | `__syncthreads();` |
| wait.sync#3 | 0.06 | 0.3% | 1.0 | 0.06 | `__syncthreads();` |

### per_token_cast_kernel_kernel grid (6, 1, 1) block (64, 1, 1)

51 launches, 12 warps; ranges, warp and last warp from the 6 recording CTAs' 12 warps; span 6.53 us, warp 6.20 us, nsys graphs 4.51 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 5.18 | 83.6% | 4.37 | 63.2% |
| load | 0.69 | 11.1% | 1.73 | 25.1% |
| store | 0.22 | 3.6% | 0.47 | 6.8% |
| compute | 0.10 | 1.6% | 0.34 | 4.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.61 | 9.8% | 1.0 | 1.02 | `for (int i = 0; i < 2; ++i)` |
| store.global#2 | 0.12 | 2.0% | 1.0 | 0.16 | `for (int i_3 = 0; i_3 < 4; ++i_3)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.10 | 1.6% | 1.0 | 0.13 | `amax_fragment[i_1] = tl::AllReduce<tl::MaxOp, 2, 1, 0, tl::NamedBarrier<64>>::run(amax_fra` |
| load.global#2*2 (est.: sampled iterations x2) | 0.09 | 1.4% | 1.0 | 0.13 | `amax_fragment[i_1] = cutlass::bfloat16_t(__hmax((bfloat16_t(((nv_bfloat162*)(&(amax_fragme` |
| store.global*2 (est.: sampled iterations x2) | 0.06 | 0.9% | 0.5 | 0.13 | `out_sf[((((((int64_t)((int)blockIdx.x)) % (int64_t)15) * (int64_t)32) + (((int64_t)((int)t` |
| store.global#3*2 (est.: sampled iterations x2) | 0.04 | 0.7% | 0.5 | 0.13 | `*(fp8_e4_16_t*)(out + ((((((((int64_t)((int)blockIdx.x)) / (int64_t)3) * (int64_t)6144) + ` |

### indexer_topk_segment_kernel grid (1, 5, 1) block (512, 1, 1)

32 launches, 80 warps; ranges, warp and last warp from the 5 recording CTAs' 80 warps; span 10.72 us, warp 10.63 us, nsys graphs 0.00 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed, with loops' unsampled iterations | 5.70 | 53.6% | 5.49 | 50.9% |
| wait | 3.70 | 34.8% | 3.88 | 36.0% |
| load | 1.06 | 10.0% | 1.10 | 10.2% |
| store | 0.17 | 1.6% | 0.31 | 2.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.06 | 10.0% | 1.0 | 1.12 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 4096); \| i` |
| wait.sync (sampled iterations) | 1.00 | 9.4% | 14.5 | 1.18 | `__syncthreads();` |
| wait.sync#20*9 (est.: sampled iterations x9) | 0.31 | 2.9% | 0.5 | 0.43 | `__syncthreads();` |
| wait.sync#23*9 (est.: sampled iterations x9) | 0.30 | 2.8% | 0.5 | 0.43 | `__syncthreads();` |
| wait.sync#19*9 (est.: sampled iterations x9) | 0.29 | 2.7% | 0.5 | 0.43 | `__syncthreads();` |
| wait.sync#14*8 (est.: sampled iterations x8) | 0.29 | 2.7% | 0.5 | 0.38 | `__syncthreads();` |
| wait.sync#6*8 (est.: sampled iterations x8) | 0.28 | 2.6% | 0.5 | 0.38 | `__syncthreads();` |
| wait.sync#5*8 (est.: sampled iterations x8) | 0.27 | 2.5% | 0.5 | 0.38 | `__syncthreads();` |
| wait.sync#13*8 (est.: sampled iterations x8) | 0.26 | 2.5% | 0.5 | 0.38 | `__syncthreads();` |
| wait.sync#24*9 (est.: sampled iterations x9) | 0.23 | 2.1% | 0.5 | 0.29 | `__syncthreads();` |
| store.global | 0.17 | 1.6% | 1.5 | 0.22 | `for (int i_3 = 0; i_3 < 4; ++i_3) \| if (((int)threadIdx.x) < total) \| if (((int)threadIdx.` |
| wait.sync#25 | 0.05 | 0.4% | 0.5 | 0.08 | `__syncthreads();` |
| wait.sync#17 | 0.04 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#16 | 0.04 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#8 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#18 | 0.03 | 0.3% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#9 | 0.03 | 0.3% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#11 | 0.03 | 0.3% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#3 | 0.03 | 0.3% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#7 | 0.03 | 0.3% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#2 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#15 | 0.03 | 0.3% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#10 | 0.03 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#22 | 0.03 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#4 | 0.02 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#12 | 0.02 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#21 | 0.02 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |

### norm_forward_and_per_token_cast_kernel_kernel grid (3, 1, 1) block (128, 1, 1)

102 launches, 12 warps; ranges, warp and last warp from the 3 recording CTAs' 12 warps; span 2.78 us, warp 2.57 us, nsys graphs 1.92 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 1.62 | 62.9% | 1.82 | 58.6% |
| unattributed | 0.36 | 14.1% | 0.74 | 23.7% |
| load | 0.34 | 13.1% | 0.33 | 10.8% |
| wait | 0.14 | 5.4% | 0.05 | 1.5% |
| compute | 0.12 | 4.6% | 0.17 | 5.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*3 (est.: sampled iterations x3) | 1.15 | 44.8% | 1.0 | 1.15 | `for (int j_2 = 0; j_2 < 2; ++j_2)` |
| store.global*3 (est.: sampled iterations x3) | 0.29 | 11.2% | 1.0 | 0.38 | `for (int j_1 = 0; j_1 < 4; ++j_1)` |
| load.global*3 (est.: sampled iterations x3) | 0.18 | 6.8% | 1.0 | 0.19 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 6)) < num_tokens) && (((vi * 2) + (` |
| wait.sync | 0.14 | 5.4% | 1.0 | 0.26 | `__syncthreads();` |
| load.global#3*3 (est.: sampled iterations x3) | 0.13 | 5.0% | 1.0 | 0.19 | `if (((vi_1 * 2) + ((((int)threadIdx.x) & 63) >> 5)) < 5)` |
| compute.reduce | 0.12 | 4.6% | 1.0 | 0.13 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| store.global#5*3 (est.: sampled iterations x3) | 0.11 | 4.2% | 0.8 | 0.19 | `*(uint4*)(out + ((((((int64_t)vi_1) * (int64_t)512) + ((((int64_t)((int)threadIdx.x)) & (i` |
| store.global#2 | 0.04 | 1.7% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| load.global#2 | 0.03 | 1.2% | 1.0 | 0.05 | `*(float2*)(row_partial + 0) = *(float2*)(sum_shared + ((((int)threadIdx.x) >> 6) * 2)); \| ` |
| store.global#3 | 0.03 | 1.0% | 0.4 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |

### topk_log_softmax_kernel grid (1, 1, 1) block (1024, 1, 1)

17 launches, 32 warps; ranges, warp and last warp from the 1 recording CTAs' 32 warps; span 16.86 us, warp 16.32 us, nsys graphs 15.84 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 13.07 | 80.1% | 14.05 | 78.7% |
| unattributed, with loops' unsampled iterations | 1.64 | 10.0% | 1.75 | 9.8% |
| wait | 1.22 | 7.5% | 1.61 | 9.0% |
| compute | 0.40 | 2.4% | 0.39 | 2.2% |
| store | 0.00 | 0.0% | 0.07 | 0.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 6.60 | 40.4% | 1.0 | 6.91 | `if (0 < (vocab & 2047)) \| if (0 < (vocab & 8191))` |
| load.global#2 | 6.46 | 39.6% | 1.0 | 7.04 | `for (int step_1 = 0; step_1 < (vocab >> 11); ++step_1) \| for (int step_1 = 0; step_1 < (vo` |
| wait.sync#2 | 0.61 | 3.7% | 1.0 | 0.96 | `__syncthreads();` |
| wait.sync | 0.35 | 2.2% | 1.0 | 0.77 | `__syncthreads();` |
| compute.reduce#3 | 0.18 | 1.1% | 1.0 | 0.19 | `ties[0] = vllm_add(ties[0], __shfl_xor_sync((uint)4294967295, ties[0], 16, 32));` |
| compute.reduce#2 | 0.11 | 0.7% | 1.0 | 0.16 | `se[0] = vllm_add(se[0], __shfl_xor_sync((uint)4294967295, se[0], 16, 32));` |
| compute.reduce | 0.11 | 0.7% | 1.0 | 0.13 | `best[0] = vllm_max(best[0], __shfl_xor_sync((uint)4294967295, best[0], 16, 32));` |
| wait.sync#4 | 0.10 | 0.6% | 1.0 | 0.13 | `__syncthreads();` |
| wait.sync#5 | 0.08 | 0.5% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#3 | 0.08 | 0.5% | 1.0 | 0.10 | `__syncthreads();` |
| load.global#3 (sampled iterations) | 0.01 | 0.1% | 0.0 | 0.35 | `float value_32 = vllm_sub(vllm_shifted(logits[((((int64_t)((int)blockIdx.x)) * logits_stri` |
| store.global (sampled iterations) | 0.00 | 0.0% | 0.0 | 0.06 | `output[(((((int64_t)c) * (int64_t)1024) + (((int64_t)((int)blockIdx.x)) * ((int64_t)topk))` |

### topk_log_softmax_kernel grid (5, 1, 1) block (1024, 1, 1)

16 launches, 160 warps; ranges, warp and last warp from the 5 recording CTAs' 160 warps; span 17.12 us, warp 16.23 us, nsys graphs 16.51 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 13.19 | 81.3% | 15.43 | 84.4% |
| unattributed, with loops' unsampled iterations | 1.63 | 10.0% | 1.75 | 9.6% |
| wait | 1.01 | 6.2% | 0.66 | 3.6% |
| compute | 0.40 | 2.4% | 0.38 | 2.1% |
| store | 0.00 | 0.0% | 0.06 | 0.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 6.68 | 41.2% | 1.0 | 7.06 | `if (0 < (vocab & 2047)) \| if (0 < (vocab & 8191))` |
| load.global#2 | 6.50 | 40.0% | 1.0 | 7.12 | `for (int step_1 = 0; step_1 < (vocab >> 11); ++step_1) \| for (int step_1 = 0; step_1 < (vo` |
| wait.sync#2 | 0.45 | 2.8% | 1.0 | 0.96 | `__syncthreads();` |
| wait.sync | 0.31 | 1.9% | 1.0 | 0.90 | `__syncthreads();` |
| compute.reduce#3 | 0.18 | 1.1% | 1.0 | 0.22 | `ties[0] = vllm_add(ties[0], __shfl_xor_sync((uint)4294967295, ties[0], 16, 32));` |
| compute.reduce#2 | 0.11 | 0.7% | 1.0 | 0.16 | `se[0] = vllm_add(se[0], __shfl_xor_sync((uint)4294967295, se[0], 16, 32));` |
| compute.reduce | 0.11 | 0.7% | 1.0 | 0.16 | `best[0] = vllm_max(best[0], __shfl_xor_sync((uint)4294967295, best[0], 16, 32));` |
| wait.sync#4 | 0.10 | 0.6% | 1.0 | 0.16 | `__syncthreads();` |
| wait.sync#3 | 0.08 | 0.5% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#5 | 0.08 | 0.5% | 1.0 | 0.10 | `__syncthreads();` |
| load.global#3 (sampled iterations) | 0.01 | 0.1% | 0.0 | 0.38 | `float value_32 = vllm_sub(vllm_shifted(logits[((((int64_t)((int)blockIdx.x)) * logits_stri` |
| store.global (sampled iterations) | 0.00 | 0.0% | 0.0 | 0.06 | `output[(((((int64_t)c) * (int64_t)1024) + (((int64_t)((int)blockIdx.x)) * ((int64_t)topk))` |

### sparse_mla_kernel_1 grid (24, 5, 1) block (128, 1, 1)

96 launches, 120 CTAs, 480 warps; span 1.98 us, warp 1.52 us, nsys graphs 2.40 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.48 | 97.4% | 1.15 | 43.4% |
| load | 0.03 | 1.8% | 1.23 | 46.6% |
| wait | 0.01 | 0.6% | 0.15 | 5.6% |
| compute | 0.00 | 0.2% | 0.08 | 2.9% |
| store | 0.00 | 0.1% | 0.04 | 1.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.03 | 1.8% | 0.1 | 1.06 | `M = __fmul_rn(Sinks[((int)blockIdx.x)], 0x1.7154764ee6c2fp+0f/*1.442695e+00*/); \| if (((in` |
| wait.sync | 0.01 | 0.6% | 0.1 | 0.42 | `__syncthreads();` |
| compute.reduce | 0.00 | 0.2% | 0.0 | 0.10 | `amax[0] = tl::AllReduce<tl::MaxOp, 8, 1, 0, tl::NamedBarrier<128>>::run(amax[0]);` |
| store.global | 0.00 | 0.1% | 0.0 | 0.06 | `*(uint2*)(Output + (((((int64_t)((int)blockIdx.y)) * (int64_t)12288) + (((int64_t)((int)bl` |
| compute.reduce#3 | 0.00 | 0.0% | 0.0 | 0.00 | `amax[0] = tl::AllReduce<tl::MaxOp, 8, 1, 0, tl::NamedBarrier<128>>::run(amax[0]);` |
| load.global#3 | 0.00 | 0.0% | 0.0 | 0.00 | `M = __fmul_rn(Sinks[((int)blockIdx.x)], 0x1.7154764ee6c2fp+0f/*1.442695e+00*/);` |
| load.global#4 | 0.00 | 0.0% | 0.0 | 0.00 | `if (((int)threadIdx.x) < 32)` |
| store.global#2 | 0.00 | 0.0% | 0.0 | 0.00 | `*(uint2*)(Output + (((((int64_t)((int)blockIdx.y)) * (int64_t)12288) + (((int64_t)((int)bl` |
| wait.sync#12 | 0.00 | 0.0% | 0.0 | 0.00 | `__syncthreads();` |
| wait.sync#13 | 0.00 | 0.0% | 0.0 | 0.00 | `__syncthreads();` |

### moe_combine_kernel grid (5, 5, 1) block (128, 1, 1)

96 launches, 100 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 2.29 us, warp 1.70 us, nsys graphs 2.43 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.26 | 73.8% | 1.96 | 81.4% |
| unattributed | 0.39 | 23.0% | 0.40 | 16.5% |
| store | 0.06 | 3.3% | 0.05 | 2.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.26 | 73.8% | 1.0 | 1.82 | `for (int k = 0; k < 3; ++k) \| for (int k = 0; k < 6; ++k) \| if ((((int)blockIdx.y) < share` |
| store.global | 0.06 | 3.3% | 1.0 | 0.06 | `{` |

### compress_kernel grid (2, 1, 1) block (128, 1, 1)

34 launches, 8 warps; ranges, warp and last warp from the 2 recording CTAs' 8 warps; span 6.53 us, warp 3.79 us, nsys graphs 15.47 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.72 | 45.4% | 2.39 | 34.5% |
| load | 1.58 | 41.7% | 3.44 | 49.7% |
| store | 0.43 | 11.3% | 1.00 | 14.4% |
| compute | 0.06 | 1.6% | 0.09 | 1.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.89 | 23.6% | 0.6 | 1.74 | `if ((((0 <= counts[0]) & (0 < counts[1])) & (counts[1] <= 2)) & (((((int)blockIdx.x) * 4) ` |
| load.global#2*4 (est.: sampled iterations x4) | 0.62 | 16.5% | 0.6 | 1.34 | `*(float4*)(v1 + 0) = *(float4*)(values + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048)` |
| store.global*4 (est.: sampled iterations x4) | 0.16 | 4.2% | 0.6 | 0.26 | `*(uint2*)(out + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + ((((int64_t)((int)thre` |
| store.global#3 | 0.10 | 2.6% | 0.3 | 0.16 | `emitted_slots[((((int64_t)((int)blockIdx.x)) * (int64_t)4) + (((int64_t)((int)threadIdx.x)` |
| store.global#2*4 (est.: sampled iterations x4) | 0.07 | 1.9% | 0.3 | 0.13 | `if ((slot >> (int64_t)3) < ((int64_t)blocks))` |
| compute.reduce | 0.06 | 1.6% | 0.6 | 0.10 | `float total = tl::warp_reduce_sum(ss[0]);` |
| store.global#5 | 0.05 | 1.2% | 0.3 | 0.14 | `emitted_slots[((((int64_t)((int)blockIdx.x)) * (int64_t)4) + (((int64_t)((int)threadIdx.x)` |
| load.global#3*4 (est.: sampled iterations x4) | 0.04 | 1.1% | 0.3 | 0.06 | `*(float4*)(w + 0) = *(float4*)(weight + ((c_1 * 128) + ((((int)threadIdx.x) & 31) * 4)));` |
| store.global#3*4 (est.: sampled iterations x4) | 0.04 | 1.1% | 0.3 | 0.06 | `*(uint2*)(out + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + ((((int64_t)((int)thre` |
| load.global#4 | 0.01 | 0.4% | 0.3 | 0.03 | `if (valid)` |
| load.global#3 | 0.01 | 0.3% | 0.3 | 0.02 | `if (valid)` |
| store.global#4 | 0.01 | 0.3% | 0.3 | 0.02 | `emitted[((((int64_t)((int)blockIdx.x)) * (int64_t)4) + (((int64_t)((int)threadIdx.x)) >> (` |
| store.global#2 | 0.00 | 0.0% | 0.3 | 0.00 | `emitted[((((int64_t)((int)blockIdx.x)) * (int64_t)4) + (((int64_t)((int)threadIdx.x)) >> (` |

### metadata_chunk_kernel grid (5, 1, 1) block (128, 1, 1)

32 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 4.59 us, warp 4.02 us, nsys graphs 6.54 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 3.14 | 78.3% | 3.13 | 44.1% |
| load | 0.80 | 20.0% | 3.80 | 53.6% |
| store | 0.07 | 1.8% | 0.16 | 2.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.72 | 17.9% | 1.0 | 0.93 | `if ((((offset + ((int)blockIdx.x)) < src_rows) && (0 <= (offset + ((int)blockIdx.x)))))` |
| load.global#2 | 0.08 | 2.1% | 0.2 | 0.35 | `if (((0 <= req) & ((int64_t)0 <= pos)))` |
| store.global | 0.05 | 1.3% | 1.0 | 0.06 | `swa[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = con` |
| store.global#2 | 0.01 | 0.2% | 0.2 | 0.06 | `lengths[((int64_t)((int)blockIdx.x))] = ((int)max(info[1], (int64_t)0));` |
| store.global#3 | 0.01 | 0.2% | 0.2 | 0.06 | `top_lengths[((int64_t)((int)blockIdx.x))] = condval_12; \| top_lengths[((int64_t)((int)bloc` |

### mhc_post_kernel grid (25, 1, 1) block (128, 1, 1)

64 launches, 100 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 1.98 us, warp 1.65 us, nsys graphs 2.21 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.50 | 90.8% | 2.51 | 94.1% |
| load | 0.10 | 6.3% | 0.11 | 4.2% |
| store | 0.05 | 2.9% | 0.05 | 1.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.10 | 6.3% | 1.0 | 0.13 | `*(float4*)(pm + 0) = *(float4*)(post + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int` |
| store.global | 0.05 | 2.9% | 1.0 | 0.06 | `for (int n_1 = 0; n_1 < 4; ++n_1)` |

### compress_gather_partial_state_kernel grid (2, 1, 1) block (128, 1, 1)

17 launches, 8 warps; ranges, warp and last warp from the 2 recording CTAs' 8 warps; span 12.00 us, warp 8.34 us, nsys graphs 0.00 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 4.64 | 55.6% | 5.01 | 43.1% |
| unattributed | 2.48 | 29.8% | 3.14 | 27.0% |
| store | 1.22 | 14.6% | 3.47 | 29.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 4.51 | 54.1% | 1.0 | 5.34 | `int nt = counts[0];` |
| store.global*2 (est.: sampled iterations x2) | 1.16 | 13.9% | 1.0 | 2.37 | `if (half == 0)` |
| load.global#2*2 (est.: sampled iterations x2) | 0.13 | 1.5% | 1.0 | 0.13 | `for (int i = 0; i < 4; ++i)` |
| store.global#2 | 0.06 | 0.7% | 1.0 | 0.06 | `if (((int)threadIdx.x) == 0)` |

### per_token_cast_kernel_kernel grid (5, 1, 1) block (64, 1, 1)

136 launches, 10 warps; ranges, warp and last warp from the 5 recording CTAs' 10 warps; span 1.22 us, warp 1.10 us, nsys graphs 1.06 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.45 | 40.8% | 0.53 | 36.5% |
| load | 0.35 | 31.3% | 0.52 | 35.7% |
| store | 0.21 | 19.2% | 0.30 | 20.8% |
| compute | 0.10 | 8.7% | 0.10 | 7.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.26 | 23.2% | 1.0 | 0.26 | `for (int i = 0; i < 2; ++i)` |
| store.global#2 | 0.12 | 10.4% | 1.0 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.10 | 8.7% | 1.0 | 0.13 | `amax_fragment[i_1] = tl::AllReduce<tl::MaxOp, 2, 1, 0, tl::NamedBarrier<64>>::run(amax_fra` |
| load.global#2*2 (est.: sampled iterations x2) | 0.09 | 8.1% | 1.0 | 0.13 | `amax_fragment[i_1] = cutlass::bfloat16_t(__hmax((bfloat16_t(((nv_bfloat162*)(&(amax_fragme` |
| store.global*2 (est.: sampled iterations x2) | 0.05 | 4.6% | 0.5 | 0.13 | `out_sf[((((((int64_t)((int)blockIdx.x)) % (int64_t)15) * (int64_t)32) + (((int64_t)((int)t` |
| store.global#3*2 (est.: sampled iterations x2) | 0.04 | 4.1% | 0.5 | 0.13 | `*(fp8_e4_16_t*)(out + ((((((((int64_t)((int)blockIdx.x)) / (int64_t)3) * (int64_t)6144) + ` |

### local_stats_kernel grid (5, 16, 1) block (128, 1, 1)

16 launches, 320 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 12.19 us, warp 11.50 us, nsys graphs 12.35 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 10.35 | 90.0% | 10.95 | 91.4% |
| unattributed | 0.73 | 6.3% | 0.71 | 5.9% |
| wait | 0.19 | 1.6% | 0.07 | 0.6% |
| compute | 0.17 | 1.5% | 0.17 | 1.5% |
| store | 0.06 | 0.5% | 0.09 | 0.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 10.35 | 90.0% | 1.0 | 10.86 | `if (expanded_local_pos[((int64_t)((int)blockIdx.x))] < num_speculative_steps)` |
| wait.sync | 0.19 | 1.6% | 1.0 | 0.62 | `__syncthreads();` |
| compute.reduce | 0.12 | 1.0% | 1.0 | 0.13 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| store.global | 0.06 | 0.5% | 1.0 | 0.10 | `if (((int)threadIdx.x) == 0)` |
| compute.reduce#2 | 0.06 | 0.5% | 1.0 | 0.06 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 2, 32);` |

### norm_forward_and_per_token_cast_kernel_kernel grid (5, 1, 1) block (128, 1, 1)

67 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 2.82 us, warp 2.58 us, nsys graphs 1.92 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 1.63 | 63.3% | 1.90 | 65.4% |
| unattributed | 0.39 | 15.1% | 0.47 | 16.2% |
| load | 0.33 | 12.8% | 0.37 | 12.8% |
| compute | 0.12 | 4.6% | 0.12 | 4.1% |
| wait | 0.11 | 4.2% | 0.05 | 1.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*3 (est.: sampled iterations x3) | 1.15 | 44.5% | 1.0 | 1.15 | `for (int j_2 = 0; j_2 < 2; ++j_2)` |
| store.global*3 (est.: sampled iterations x3) | 0.30 | 11.5% | 1.0 | 0.38 | `for (int j_1 = 0; j_1 < 4; ++j_1)` |
| load.global*3 (est.: sampled iterations x3) | 0.18 | 6.9% | 1.0 | 0.19 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 6)) < num_tokens) && (((vi * 2) + (` |
| compute.reduce | 0.12 | 4.6% | 1.0 | 0.13 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| load.global#3*3 (est.: sampled iterations x3) | 0.12 | 4.6% | 1.0 | 0.19 | `if (((vi_1 * 2) + ((((int)threadIdx.x) & 63) >> 5)) < 5)` |
| store.global#5*3 (est.: sampled iterations x3) | 0.12 | 4.5% | 1.0 | 0.19 | `*(uint4*)(out + ((((((int64_t)vi_1) * (int64_t)512) + ((((int64_t)((int)threadIdx.x)) & (i` |
| wait.sync | 0.11 | 4.2% | 1.0 | 0.19 | `__syncthreads();` |
| store.global#2 | 0.04 | 1.6% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| load.global#2 | 0.03 | 1.2% | 1.0 | 0.06 | `*(float2*)(row_partial + 0) = *(float2*)(sum_shared + ((((int)threadIdx.x) >> 6) * 2)); \| ` |
| store.global#3 | 0.03 | 1.2% | 0.5 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |
| load.global#2*5 (est.: sampled iterations x5) | 0.00 | 0.0% | 0.0 | 0.00 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 1024) + (((int)threadIdx.x) * 8)` |
| load.global*5 (est.: sampled iterations x5) | 0.00 | 0.0% | 0.0 | 0.00 | `*(uint4*)(x_local + (vi * 8)) = *(uint4*)(x + ((((((int64_t)vi) * (int64_t)1024) + (((int6` |
| store.global#4*5 (est.: sampled iterations x5) | 0.00 | 0.0% | 0.0 | 0.00 | `for (int j_1 = 0; j_1 < 2; ++j_1)` |
| store.global*5 (est.: sampled iterations x5) | 0.00 | 0.0% | 0.0 | 0.00 | `for (int j = 0; j < 4; ++j)` |

### sparse_mla_kernel_1 grid (24, 10, 1) block (128, 1, 1)

51 launches, 240 CTAs, 960 warps; span 2.85 us, warp 2.13 us, nsys graphs 5.22 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 2.12 | 99.1% | 2.36 | 65.8% |
| load | 0.01 | 0.6% | 1.01 | 28.0% |
| wait | 0.00 | 0.2% | 0.10 | 2.8% |
| compute | 0.00 | 0.1% | 0.06 | 1.7% |
| store | 0.00 | 0.0% | 0.06 | 1.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#3 | 0.01 | 0.5% | 0.0 | 0.80 | `M = __fmul_rn(Sinks[((int)blockIdx.x)], 0x1.7154764ee6c2fp+0f/*1.442695e+00*/);` |
| wait.sync#13 | 0.00 | 0.1% | 0.0 | 0.26 | `__syncthreads();` |
| load.global#4 | 0.00 | 0.1% | 0.0 | 0.26 | `if (((int)threadIdx.x) < 32)` |
| compute.reduce#3 | 0.00 | 0.1% | 0.0 | 0.13 | `amax[0] = tl::AllReduce<tl::MaxOp, 8, 1, 0, tl::NamedBarrier<128>>::run(amax[0]);` |
| wait.sync#12 | 0.00 | 0.1% | 0.0 | 0.13 | `__syncthreads();` |
| store.global#2 | 0.00 | 0.0% | 0.0 | 0.06 | `*(uint2*)(Output + (((((int64_t)((int)blockIdx.y)) * (int64_t)12288) + (((int64_t)((int)bl` |

### router_partials_kernel grid (5, 1, 1) block (64, 1, 1)

51 launches, 10 warps; ranges, warp and last warp from the 5 recording CTAs' 10 warps; span 3.01 us, warp 2.21 us, nsys graphs 2.66 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.95 | 43.0% | 1.50 | 42.0% |
| load | 0.88 | 39.9% | 1.47 | 41.2% |
| store | 0.20 | 8.8% | 0.24 | 6.6% |
| compute | 0.18 | 8.2% | 0.37 | 10.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.48 | 21.7% | 1.0 | 0.51 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < mask_len))` |
| load.global#4 | 0.28 | 12.4% | 0.5 | 0.61 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < image_len)) \| if ((((((int)bl` |
| compute.reduce | 0.18 | 8.2% | 0.5 | 0.38 | `for (int k = 0; k < 3; ++k) \| for (int k = 0; k < 6; ++k)` |
| store.global | 0.07 | 3.0% | 1.0 | 0.10 | `if (condval == (bool)0) \| if (condval == (bool)1)` |
| store.global#2 | 0.07 | 3.0% | 1.0 | 0.10 | `if (condval_1 == (bool)0) \| if (condval_1 == (bool)1)` |
| load.global#2 | 0.06 | 2.9% | 1.0 | 0.10 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < mask_len))` |
| load.global#3 | 0.06 | 2.9% | 1.0 | 0.10 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < mask_len))` |
| store.global#3 | 0.06 | 2.7% | 0.5 | 0.13 | `if ((((int)threadIdx.x) & 31) < 3) \| if ((((int)threadIdx.x) & 31) < 6)` |

### indexer_topk_segment_kernel grid (2, 5, 1) block (512, 1, 1)

16 launches, 160 warps; ranges, warp and last warp from the 5 recording CTAs' 80 warps; span 10.10 us, warp 9.88 us, nsys graphs 0.00 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 6.14 | 62.1% | 5.86 | 57.4% |
| unattributed, with loops' unsampled iterations | 3.19 | 32.3% | 3.56 | 34.8% |
| load | 0.47 | 4.7% | 0.69 | 6.7% |
| store | 0.09 | 0.9% | 0.11 | 1.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync#24*9 (est.: sampled iterations x9) | 0.67 | 6.7% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#20*9 (est.: sampled iterations x9) | 0.66 | 6.7% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#23*9 (est.: sampled iterations x9) | 0.64 | 6.4% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#19*9 (est.: sampled iterations x9) | 0.63 | 6.4% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#6*8 (est.: sampled iterations x8) | 0.60 | 6.0% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync#14*8 (est.: sampled iterations x8) | 0.60 | 6.0% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync#5*8 (est.: sampled iterations x8) | 0.59 | 6.0% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync#13*8 (est.: sampled iterations x8) | 0.56 | 5.6% | 1.0 | 0.77 | `__syncthreads();` |
| load.global#2 | 0.26 | 2.6% | 1.0 | 0.51 | `for (int i = 0; i < 8; ++i)` |
| load.global | 0.21 | 2.1% | 1.0 | 0.22 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 4096); \| i` |
| wait.sync (sampled iterations) | 0.14 | 1.4% | 1.0 | 0.35 | `__syncthreads();` |
| wait.sync#25 | 0.10 | 1.0% | 1.0 | 0.16 | `__syncthreads();` |
| store.global | 0.09 | 0.9% | 1.0 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3) \| if (((int)threadIdx.x) < total) \| if (((int)threadIdx.` |
| wait.sync#17 | 0.08 | 0.8% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#18 | 0.08 | 0.8% | 1.0 | 0.16 | `__syncthreads();` |
| wait.sync#8 | 0.07 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#16 | 0.07 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#9 | 0.07 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#2 | 0.07 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#3 | 0.06 | 0.6% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#7 | 0.06 | 0.6% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#15 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#12 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#11 | 0.06 | 0.6% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#10 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#4 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#22 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#21 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |

### rope_kernel grid (5, 1, 1) block (192, 1, 1)

96 launches, 30 warps; ranges, warp and last warp from the 5 recording CTAs' 30 warps; span 1.31 us, warp 1.28 us, nsys graphs 1.47 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.75 | 58.9% | 1.07 | 66.8% |
| load | 0.49 | 37.9% | 0.48 | 30.0% |
| store | 0.04 | 3.2% | 0.05 | 3.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.49 | 37.9% | 1.0 | 0.51 | `if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < tokens)) \| int64_t position ` |
| store.global | 0.04 | 3.2% | 1.0 | 0.06 | `*(uint4*)(x + ((((((int64_t)((int)blockIdx.x)) * (int64_t)12288) + ((((int64_t)((int)threa` |

### mhc_collapse_kernel grid (25, 1, 1) block (128, 1, 1)

48 launches, 100 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 2.45 us, warp 2.40 us, nsys graphs 4.22 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 2.26 | 94.1% | 2.79 | 88.1% |
| load | 0.10 | 4.0% | 0.33 | 10.4% |
| store | 0.04 | 1.9% | 0.05 | 1.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.10 | 4.0% | 1.0 | 0.13 | `*(float4*)(mv + 0) = *(float4*)(mix + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int6` |
| store.global | 0.04 | 1.9% | 1.0 | 0.06 | `*(uint4*)(out + ((((int64_t)((int)blockIdx.x)) * (int64_t)1024) + (((int64_t)((int)threadI` |

### indexer_topk_merge_kernel grid (1, 5, 1) block (512, 1, 1)

16 launches, 80 warps; ranges, warp and last warp from the 5 recording CTAs' 80 warps; span 9.28 us, warp 9.05 us, nsys graphs 5.34 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 6.12 | 67.6% | 6.06 | 63.6% |
| unattributed | 2.27 | 25.1% | 2.30 | 24.2% |
| load | 0.58 | 6.4% | 1.03 | 10.8% |
| store | 0.08 | 0.9% | 0.13 | 1.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync#20*9 (est.: sampled iterations x9) | 0.67 | 7.4% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#24*9 (est.: sampled iterations x9) | 0.66 | 7.3% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#19*9 (est.: sampled iterations x9) | 0.63 | 7.0% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#23*9 (est.: sampled iterations x9) | 0.62 | 6.8% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#14*8 (est.: sampled iterations x8) | 0.61 | 6.8% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync#6*8 (est.: sampled iterations x8) | 0.60 | 6.6% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync#5*8 (est.: sampled iterations x8) | 0.58 | 6.4% | 1.0 | 0.77 | `__syncthreads();` |
| load.global | 0.57 | 6.3% | 1.0 | 0.70 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 8192);` |
| wait.sync#13*8 (est.: sampled iterations x8) | 0.56 | 6.2% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync | 0.15 | 1.7% | 1.0 | 0.22 | `__syncthreads();` |
| wait.sync#17 | 0.08 | 0.9% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#16 | 0.08 | 0.8% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#8 | 0.07 | 0.8% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#25 | 0.07 | 0.8% | 1.0 | 0.10 | `__syncthreads();` |
| store.global#2 | 0.07 | 0.7% | 0.9 | 0.10 | `Values[((((int64_t)((int)blockIdx.y)) * (int64_t)512) + ((int64_t)((int)threadIdx.x)))] = ` |
| wait.sync#2 | 0.07 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#3 | 0.07 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#15 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#12 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#9 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#11 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#7 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#18 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#4 | 0.06 | 0.7% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#22 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#21 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#10 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| load.global#2 | 0.01 | 0.1% | 0.1 | 0.10 | `if (((0 <= ((int*)sel_idx)[((int)threadIdx.x)]) && (((int*)sel_idx)[((int)threadIdx.x)] < ` |
| store.global | 0.01 | 0.1% | 0.1 | 0.10 | `Values[((((int64_t)((int)blockIdx.y)) * (int64_t)512) + ((int64_t)((int)threadIdx.x)))] = ` |

### gemm_splitk_reduce_cast_kernel grid (24, 2, 1) block (128, 1, 1)

102 launches, 192 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 1.06 us, warp 0.97 us, nsys graphs 1.15 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.44 | 45.5% | 0.51 | 37.4% |
| load | 0.37 | 38.6% | 0.70 | 51.1% |
| compute | 0.08 | 8.3% | 0.08 | 5.5% |
| store | 0.07 | 7.6% | 0.08 | 6.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.37 | 38.6% | 1.0 | 0.43 | `if ((1 <= M))` |
| compute.reduce | 0.08 | 8.3% | 1.0 | 0.10 | `amax[0] = tl::AllReduce<tl::MaxOp, 8, 1, 0, tl::NamedBarrier<128>>::run(amax[0]);` |
| store.global | 0.07 | 7.6% | 1.0 | 0.10 | `if (((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 5)) < M)` |

### norm_forward_and_per_token_cast_kernel_kernel grid (2, 1, 1) block (128, 1, 1)

51 launches, 8 warps; ranges, warp and last warp from the 2 recording CTAs' 8 warps; span 2.72 us, warp 2.70 us, nsys graphs 1.34 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 2.03 | 75.4% | 2.22 | 79.0% |
| load | 0.42 | 15.4% | 0.47 | 16.7% |
| wait | 0.14 | 5.0% | 0.04 | 1.6% |
| compute | 0.11 | 4.2% | 0.11 | 3.9% |
| unattributed | 0.00 | 0.0% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*4 (est.: sampled iterations x4) | 1.54 | 57.0% | 1.0 | 1.54 | `for (int j_2 = 0; j_2 < 2; ++j_2)` |
| store.global*4 (est.: sampled iterations x4) | 0.34 | 12.5% | 1.0 | 0.51 | `for (int j_1 = 0; j_1 < 4; ++j_1)` |
| load.global*4 (est.: sampled iterations x4) | 0.24 | 8.9% | 1.0 | 0.26 | `if (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 4)) < num_tokens)` |
| load.global#3*4 (est.: sampled iterations x4) | 0.14 | 5.3% | 1.0 | 0.26 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 128) + ((((int)threadIdx.x) & 15` |
| wait.sync | 0.14 | 5.0% | 1.0 | 0.29 | `__syncthreads();` |
| compute.reduce | 0.11 | 4.2% | 1.0 | 0.13 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| store.global#5*4 (est.: sampled iterations x4) | 0.08 | 3.0% | 0.6 | 0.13 | `*(uint4*)(out + ((((((int64_t)vi_1) * (int64_t)128) + ((((int64_t)((int)threadIdx.x)) & (i` |
| store.global#2 | 0.04 | 1.6% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| store.global#3 | 0.04 | 1.3% | 0.6 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |
| load.global#2 | 0.03 | 1.2% | 1.0 | 0.03 | `*(float2*)(row_partial + 0) = *(float2*)(sum_shared + ((((int)threadIdx.x) >> 6) * 2)); \| ` |

### activation_swiglu_clamp_kernel grid (1, 10, 1) block (128, 1, 1)

51 launches, 40 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 1.95 us, warp 1.05 us, nsys graphs 7.94 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.95 | 90.0% | 2.53 | 91.1% |
| load | 0.07 | 6.8% | 0.20 | 7.3% |
| store | 0.03 | 3.2% | 0.05 | 1.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.07 | 6.8% | 0.8 | 0.10 | `*(uint4*)(gate + 0) = *(uint4*)(X + ((((int64_t)((int)blockIdx.y)) * (int64_t)1536) + (((i` |
| store.global | 0.03 | 3.2% | 0.8 | 0.06 | `*(uint4*)(Y + ((((int64_t)((int)blockIdx.y)) * (int64_t)768) + (((int64_t)((int)threadIdx.` |

### mhc_finalize_kernel grid (6, 1, 1) block (128, 1, 1)

12 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 11.87 us, warp 10.92 us, nsys graphs 9.14 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 6.72 | 61.5% | 5.21 | 45.0% |
| store | 1.56 | 14.3% | 2.93 | 25.3% |
| load | 1.12 | 10.3% | 1.73 | 14.9% |
| compute | 0.98 | 9.0% | 1.15 | 9.9% |
| wait | 0.54 | 5.0% | 0.57 | 4.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.67 | 6.2% | 0.2 | 2.69 | `if (((int)threadIdx.x) < 4)` |
| compute.reduce | 0.57 | 5.2% | 0.2 | 2.30 | `row_max[0] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<32>>::run(row_max[0]);` |
| store.global#4*7 (est.: sampled iterations x7) | 0.55 | 5.0% | 0.8 | 0.90 | `if ((((int)threadIdx.x) % 4) == 0)` |
| wait.sync | 0.54 | 5.0% | 1.0 | 0.74 | `__syncthreads();` |
| compute.reduce#2*7 (est.: sampled iterations x7) | 0.42 | 3.8% | 0.8 | 0.67 | `amax[0] = max(amax[0], __shfl_xor_sync((uint)4294967295, amax[0], 1, 32));` |
| load.global#3 | 0.39 | 3.6% | 0.8 | 0.53 | `for (int i_split_2 = 0; i_split_2 < 40; ++i_split_2)` |
| load.global#4*7 (est.: sampled iterations x7) | 0.36 | 3.3% | 0.8 | 0.67 | `*(uint4*)(yv + 0) = *(uint4*)(C + ((((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((i` |
| load.global | 0.32 | 3.0% | 1.0 | 0.93 | `if (((int)threadIdx.x) < 32)` |
| store.global#3*7 (est.: sampled iterations x7) | 0.32 | 2.9% | 0.8 | 0.45 | `*(uint4*)(y + ((((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((int64_t)step) * (int6` |
| load.global#2 | 0.04 | 0.4% | 0.2 | 0.19 | `cm[0] = ((mixes_shared[((((int)threadIdx.x) & 15) + 8)] * scale[2]) + base[((((int)threadI` |
| store.global#2 | 0.01 | 0.1% | 0.2 | 0.06 | `if ((((int)threadIdx.x) >> 4) == 0)` |

### token_ranks_kernel grid (5, 1, 1) block (256, 1, 1)

16 launches, 40 warps; ranges, warp and last warp from the 5 recording CTAs' 40 warps; span 7.81 us, warp 7.70 us, nsys graphs 8.40 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 6.95 | 90.2% | 7.66 | 89.6% |
| compute | 0.35 | 4.6% | 0.45 | 5.3% |
| unattributed | 0.35 | 4.6% | 0.38 | 4.5% |
| store | 0.05 | 0.6% | 0.05 | 0.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 6.95 | 90.2% | 1.0 | 7.18 | `if ((((int64_t)0 <= token_ids[((int64_t)((int)blockIdx.x))]) && (token_ids[((int64_t)((int` |
| compute.reduce | 0.35 | 4.6% | 1.0 | 0.67 | `total[0] = tl::AllReduce<tl::SumOp, 256, 1, 0, tl::NamedBarrier<256>>::run(total[0], (&(wo` |
| store.global | 0.05 | 0.6% | 1.0 | 0.06 | `output[((int64_t)((int)blockIdx.x))] = ((int64_t)total[0]);` |

### per_token_cast_kernel_kernel grid (3, 1, 1) block (64, 1, 1)

102 launches, 6 warps; ranges, warp and last warp from the 3 recording CTAs' 6 warps; span 1.18 us, warp 1.11 us, nsys graphs 1.60 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.45 | 40.7% | 0.47 | 35.8% |
| load | 0.34 | 30.8% | 0.39 | 29.3% |
| store | 0.21 | 18.8% | 0.31 | 23.8% |
| compute | 0.11 | 9.6% | 0.15 | 11.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.26 | 23.1% | 1.0 | 0.26 | `for (int i = 0; i < 2; ++i)` |
| store.global#2 | 0.11 | 10.1% | 1.0 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.11 | 9.6% | 1.0 | 0.13 | `amax_fragment[i_1] = tl::AllReduce<tl::MaxOp, 2, 1, 0, tl::NamedBarrier<64>>::run(amax_fra` |
| load.global#2*2 (est.: sampled iterations x2) | 0.09 | 7.7% | 1.0 | 0.13 | `amax_fragment[i_1] = cutlass::bfloat16_t(__hmax((bfloat16_t(((nv_bfloat162*)(&(amax_fragme` |
| store.global*2 (est.: sampled iterations x2) | 0.05 | 4.8% | 0.5 | 0.13 | `out_sf[((((((int64_t)((int)blockIdx.x)) % (int64_t)15) * (int64_t)32) + (((int64_t)((int)t` |
| store.global#3*2 (est.: sampled iterations x2) | 0.04 | 3.9% | 0.5 | 0.13 | `*(fp8_e4_16_t*)(out + ((((((((int64_t)((int)blockIdx.x)) / (int64_t)3) * (int64_t)6144) + ` |

### indexer_mxfp4_quantize_kernel grid (5, 1, 1) block (128, 1, 1)

32 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 3.78 us, warp 3.70 us, nsys graphs 2.91 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 2.29 | 62.1% | 2.36 | 62.7% |
| store | 1.12 | 30.2% | 1.10 | 29.3% |
| unattributed | 0.28 | 7.7% | 0.30 | 8.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2*16 (est.: sampled iterations x16) | 1.27 | 34.3% | 1.0 | 1.54 | `float v__2 = __fmul_rn(((float)X[(((((int64_t)((int)blockIdx.x)) * (int64_t)4096) + (((int` |
| store.global*16 (est.: sampled iterations x16) | 1.08 | 29.1% | 1.0 | 1.54 | `Pool[(((((slot & (int64_t)127) * (int64_t)64) + ((((int64_t)((int)threadIdx.x)) & (int64_t` |
| load.global | 1.03 | 27.8% | 1.0 | 1.09 | `for (int i = 0; i < 32; ++i)` |
| store.global#2 | 0.04 | 1.1% | 1.0 | 0.06 | `Pool[(((((slot & (int64_t)127) * (int64_t)4) + ((slot >> (int64_t)7) * ((int64_t)pool_stri` |

### indexer_mxfp4_quantize_kernel grid (1, 1, 1) block (128, 1, 1)

34 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 3.01 us, warp 1.01 us, nsys graphs 2.02 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.39 | 38.3% | 0.56 | 15.9% |
| load | 0.36 | 35.6% | 1.80 | 51.7% |
| store | 0.26 | 26.1% | 1.13 | 32.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2*16 (est.: sampled iterations x16) | 0.26 | 25.3% | 0.2 | 1.02 | `float v__2 = __fmul_rn(((float)X[(((((int64_t)((int)blockIdx.x)) * (int64_t)4096) + (((int` |
| store.global*16 (est.: sampled iterations x16) | 0.26 | 25.3% | 0.2 | 1.02 | `Pool[(((((slot & (int64_t)127) * (int64_t)64) + ((((int64_t)((int)threadIdx.x)) & (int64_t` |
| load.global | 0.10 | 10.3% | 0.2 | 0.42 | `for (int i = 0; i < 32; ++i)` |
| store.global#2 | 0.01 | 0.8% | 0.2 | 0.03 | `Pool[(((((slot & (int64_t)127) * (int64_t)4) + ((slot >> (int64_t)7) * ((int64_t)pool_stri` |

### prepare_dflash_inputs_kernel grid (9, 1, 1) block (256, 1, 1)

51 launches, 72 warps; ranges, warp and last warp from the 5 recording CTAs' 40 warps; span 1.98 us, warp 0.73 us, nsys graphs 2.46 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.25 | 34.3% | 1.39 | 68.9% |
| unattributed | 0.25 | 33.6% | 0.36 | 17.7% |
| store | 0.23 | 32.1% | 0.27 | 13.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.24 | 32.6% | 0.2 | 1.31 | `req = idx_mapping[((int)blockIdx.y)];` |
| store.global | 0.19 | 25.4% | 0.8 | 0.26 | `if (((int)blockIdx.y) == (num_reqs - 1))` |
| store.global#2 | 0.05 | 6.7% | 0.2 | 0.29 | `if (((int)blockIdx.y) == (num_reqs - 1))` |
| load.global#2 | 0.01 | 1.6% | 0.2 | 0.06 | `vllm_store_if_b32((&(out_query_start_loc[((int)blockIdx.y)])), (((int)blockIdx.y) * num_qu` |

### sink_merge_kernel grid (1, 1, 1) block (128, 1, 1)

17 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 4.29 us, warp 4.29 us, nsys graphs 4.32 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.90 | 44.4% | 2.05 | 34.3% |
| unattributed | 0.90 | 20.9% | 2.79 | 46.7% |
| wait | 0.89 | 20.7% | 0.45 | 7.5% |
| compute | 0.58 | 13.4% | 0.55 | 9.3% |
| store | 0.02 | 0.6% | 0.13 | 2.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2*3 (est.: sampled iterations x3) | 1.63 | 38.1% | 1.0 | 1.63 | `for (int step = 0; step < ((width + 127) >> 7); ++step) \| for (int step = 0; step < ((widt` |
| compute.reduce*3 (est.: sampled iterations x3) | 0.58 | 13.4% | 1.0 | 0.58 | `float other = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| wait.sync#3*3 (est.: sampled iterations x3) | 0.53 | 12.3% | 1.0 | 0.67 | `__syncthreads();` |
| load.global | 0.22 | 5.2% | 1.0 | 0.22 | `last = sampled[(max((((int64_t)num_sampled[((int64_t)((int)blockIdx.x))]) - (int64_t)1), (` |
| wait.sync#2*3 (est.: sampled iterations x3) | 0.19 | 4.5% | 1.0 | 0.19 | `__syncthreads();` |
| wait.sync*3 (est.: sampled iterations x3) | 0.17 | 3.9% | 1.0 | 0.19 | `__syncthreads();` |
| load.global#3*3 (est.: sampled iterations x3) | 0.05 | 1.1% | 0.2 | 0.19 | `if ((best[0] == -CUDART_INF_F))` |
| store.global*3 (est.: sampled iterations x3) | 0.02 | 0.6% | 0.2 | 0.10 | `alternates[((((int64_t)((int)blockIdx.x)) * (int64_t)3) + ((int64_t)rank))] = condval_31;` |

### gemm_bf16_partials_kernel grid (1, 40, 1) block (256, 1, 1)

34 launches, 320 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 2.69 us, warp 1.71 us, nsys graphs 2.85 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.25 | 73.1% | 2.10 | 78.5% |
| wait | 0.23 | 13.6% | 0.25 | 9.4% |
| compute | 0.11 | 6.4% | 0.23 | 8.6% |
| load | 0.07 | 4.1% | 0.00 | 0.0% |
| store | 0.05 | 2.8% | 0.09 | 3.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*2 (est.: sampled iterations x2) | 0.11 | 6.4% | 0.5 | 0.26 | `{` |
| wait.sync | 0.11 | 6.2% | 1.0 | 0.13 | `__syncthreads();` |
| load.tma*2 (est.: sampled iterations x2) | 0.07 | 4.1% | 0.5 | 0.19 | `if (tl::tl_shuffle_elect<128>())` |
| wait.mbarrier*2 (est.: sampled iterations x2) | 0.06 | 3.7% | 0.5 | 0.13 | `mbarrier[(kb + 3)].wait(1);` |
| wait.mbarrier#2*2 (est.: sampled iterations x2) | 0.06 | 3.6% | 0.5 | 0.13 | `mbarrier[kb_1].wait(0);` |
| store.global | 0.05 | 2.8% | 0.5 | 0.13 | `for (int i_1 = 0; i_1 < 4; ++i_1) \| for (int i_2 = 0; i_2 < 8; ++i_2) \| for (int i_4 = 0; ` |

### sparse_mla_kernel grid (5, 1, 6) block (256, 1, 1)

4 launches, 240 warps; ranges, warp and last warp from the 5 recording CTAs' 40 warps; span 20.83 us, warp 18.59 us, nsys graphs 14.32 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed, with loops' unsampled iterations | 10.58 | 56.9% | 10.98 | 51.2% |
| wait | 4.21 | 22.6% | 0.94 | 4.4% |
| load | 1.67 | 9.0% | 7.23 | 33.8% |
| compute | 1.47 | 7.9% | 1.50 | 7.0% |
| store | 0.66 | 3.5% | 0.78 | 3.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync (sampled iterations) | 4.14 | 22.3% | 11.0 | 6.99 | `__syncthreads();` |
| load.global (sampled iterations) | 1.37 | 7.4% | 2.0 | 7.33 | `for (int c = 0; c < 4; ++c) \| for (int i = 0; i < 8; ++i) \| for (int i_1 = 0; i_1 < 8; ++i` |
| compute.mma (sampled iterations) | 1.22 | 6.5% | 2.0 | 1.28 | `for (int ki = 0; ki < 32; ++ki) \| for (int ki_1 = 0; ki_1 < 2; ++ki_1)` |
| store.global | 0.66 | 3.5% | 1.0 | 0.85 | `for (int i_10 = 0; i_10 < 32; ++i_10) \| for (int i_11 = 0; i_11 < 32; ++i_11) \| for (int i` |
| load.cp_async (sampled iterations) | 0.30 | 1.6% | 2.0 | 0.42 | `for (int k_1 = 0; k_1 < 5; ++k_1) \| for (int k_1 = 0; k_1 < 9; ++k_1) \| for (int k_3 = 0; ` |
| compute.reduce (sampled iterations) | 0.25 | 1.4% | 2.0 | 0.26 | `m_g[i_5] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(m_g[i_5], (&((` |
| wait.cp_async (sampled iterations) | 0.07 | 0.4% | 1.0 | 0.10 | `tl::cp_async_wait<0>();` |

### activation_swiglu_clamp_kernel grid (1, 5, 1) block (128, 1, 1)

96 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 0.93 us, warp 0.73 us, nsys graphs 1.86 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.63 | 85.8% | 0.79 | 85.0% |
| load | 0.07 | 9.8% | 0.09 | 10.1% |
| store | 0.03 | 4.4% | 0.05 | 4.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.07 | 9.8% | 0.8 | 0.10 | `*(uint4*)(gate + 0) = *(uint4*)(X + ((((int64_t)((int)blockIdx.y)) * (int64_t)1536) + (((i` |
| store.global | 0.03 | 4.4% | 0.8 | 0.06 | `*(uint4*)(Y + ((((int64_t)((int)blockIdx.y)) * (int64_t)768) + (((int64_t)((int)threadIdx.` |

### gemm_bf16_partials_kernel grid (2, 4, 1) block (256, 1, 1)

34 launches, 64 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 1.98 us, warp 1.37 us, nsys graphs 2.11 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.91 | 66.4% | 1.65 | 70.1% |
| wait | 0.23 | 16.6% | 0.38 | 16.1% |
| compute | 0.12 | 8.5% | 0.23 | 9.8% |
| load | 0.07 | 5.0% | 0.00 | 0.0% |
| store | 0.05 | 3.5% | 0.10 | 4.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*2 (est.: sampled iterations x2) | 0.12 | 8.5% | 0.5 | 0.26 | `{` |
| wait.sync | 0.11 | 7.9% | 1.0 | 0.13 | `__syncthreads();` |
| load.tma*2 (est.: sampled iterations x2) | 0.07 | 5.0% | 0.5 | 0.19 | `if (tl::tl_shuffle_elect<128>())` |
| wait.mbarrier*2 (est.: sampled iterations x2) | 0.06 | 4.4% | 0.5 | 0.13 | `mbarrier[(kb + 3)].wait(1);` |
| wait.mbarrier#2*2 (est.: sampled iterations x2) | 0.06 | 4.3% | 0.5 | 0.13 | `mbarrier[kb_1].wait(0);` |
| store.global | 0.05 | 3.5% | 0.5 | 0.13 | `for (int i_1 = 0; i_1 < 4; ++i_1) \| for (int i_2 = 0; i_2 < 8; ++i_2) \| for (int i_4 = 0; ` |

### sink_candidates_kernel grid (1, 127, 1) block (128, 1, 1)

17 launches, 508 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 4.42 us, warp 3.67 us, nsys graphs 3.20 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.54 | 42.0% | 1.96 | 44.1% |
| compute | 0.90 | 24.4% | 0.90 | 20.1% |
| load | 0.54 | 14.7% | 0.91 | 20.3% |
| wait | 0.44 | 12.0% | 0.35 | 8.0% |
| store | 0.25 | 6.8% | 0.33 | 7.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.reduce*4 (est.: sampled iterations x4) | 0.65 | 17.8% | 1.0 | 0.77 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| load.global | 0.54 | 14.7% | 1.0 | 0.64 | `local[0] = max((num_sampled[((int64_t)((int)blockIdx.x))] - 1), 0);` |
| store.global*4 (est.: sampled iterations x4) | 0.25 | 6.8% | 1.0 | 0.38 | `if (((int)threadIdx.x) == 0)` |
| compute.reduce#2*4 (est.: sampled iterations x4) | 0.24 | 6.6% | 1.0 | 0.26 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 2, 32);` |
| wait.sync#2*4 (est.: sampled iterations x4) | 0.24 | 6.6% | 1.0 | 0.38 | `__syncthreads();` |
| wait.sync*4 (est.: sampled iterations x4) | 0.20 | 5.4% | 1.0 | 0.26 | `__syncthreads();` |

### moe_combine_kernel grid (5, 10, 1) block (128, 1, 1)

51 launches, 200 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 1.41 us, warp 1.10 us, nsys graphs 1.50 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.65 | 58.8% | 0.93 | 67.6% |
| unattributed | 0.40 | 36.2% | 0.40 | 28.8% |
| store | 0.05 | 4.9% | 0.05 | 3.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.65 | 58.8% | 1.0 | 0.93 | `for (int k = 0; k < 3; ++k) \| for (int k = 0; k < 6; ++k) \| if ((((int)blockIdx.y) < share` |
| store.global | 0.05 | 4.9% | 1.0 | 0.06 | `{` |

### rope_kernel grid (10, 1, 1) block (192, 1, 1)

51 launches, 60 warps; ranges, warp and last warp from the 5 recording CTAs' 30 warps; span 1.28 us, warp 1.21 us, nsys graphs 1.50 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.72 | 59.0% | 0.83 | 62.0% |
| load | 0.46 | 37.7% | 0.46 | 34.4% |
| store | 0.04 | 3.3% | 0.05 | 3.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.46 | 37.7% | 1.0 | 0.48 | `if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < tokens)) \| int64_t position ` |
| store.global | 0.04 | 3.3% | 1.0 | 0.06 | `*(uint4*)(x + ((((((int64_t)((int)blockIdx.x)) * (int64_t)12288) + ((((int64_t)((int)threa` |

### indexer_score_kernel grid (128, 5, 1) block (128, 1, 1)

16 launches, 2560 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 4.22 us, warp 1.48 us, nsys graphs 3.86 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.86 | 57.9% | 2.60 | 62.5% |
| unattributed, with loops' unsampled iterations | 0.50 | 33.8% | 1.21 | 29.2% |
| wait | 0.08 | 5.5% | 0.18 | 4.4% |
| compute | 0.03 | 2.2% | 0.13 | 3.0% |
| store | 0.01 | 0.7% | 0.04 | 1.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.40 | 27.4% | 1.0 | 0.70 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 4096); \| i` |
| load.global#2 (sampled iterations) | 0.40 | 27.0% | 0.2 | 1.60 | `for (int i_1 = 0; i_1 < 32; ++i_1)` |
| load.global#3 (sampled iterations) | 0.05 | 3.4% | 0.2 | 0.37 | `if (((int)threadIdx.x) < 64)` |
| wait.sync#2 (sampled iterations) | 0.05 | 3.1% | 0.2 | 0.34 | `__syncthreads();` |
| compute.mma (sampled iterations) | 0.03 | 2.2% | 0.2 | 0.13 | `for (int ki = 0; ki < 2; ++ki)` |
| wait.sync (sampled iterations) | 0.01 | 1.0% | 0.2 | 0.06 | `__syncthreads();` |
| store.global (sampled iterations) | 0.01 | 0.7% | 0.2 | 0.06 | `if ((((int)threadIdx.x) >> 6) == 0)` |
| wait.sync#3 (sampled iterations) | 0.01 | 0.7% | 0.2 | 0.06 | `__syncthreads();` |
| wait.sync#4 (sampled iterations) | 0.01 | 0.7% | 0.2 | 0.06 | `__syncthreads();` |

### indexer_score_kernel grid (64, 5, 1) block (128, 1, 1)

16 launches, 1280 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 4.03 us, warp 1.79 us, nsys graphs 0.00 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.92 | 51.1% | 2.00 | 49.4% |
| unattributed, with loops' unsampled iterations | 0.67 | 37.5% | 1.37 | 33.8% |
| wait | 0.16 | 8.6% | 0.49 | 12.0% |
| compute | 0.04 | 2.1% | 0.15 | 3.7% |
| store | 0.01 | 0.7% | 0.04 | 1.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.57 | 32.1% | 1.0 | 0.77 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 4096); \| i` |
| load.global#2 (sampled iterations) | 0.31 | 17.1% | 0.2 | 1.26 | `for (int i_1 = 0; i_1 < 32; ++i_1)` |
| wait.sync#2 (sampled iterations) | 0.06 | 3.5% | 0.2 | 0.32 | `__syncthreads();` |
| wait.sync#3 (sampled iterations) | 0.06 | 3.4% | 0.2 | 0.26 | `__syncthreads();` |
| compute.mma (sampled iterations) | 0.04 | 2.1% | 0.2 | 0.16 | `for (int ki = 0; ki < 2; ++ki)` |
| load.global#3 (sampled iterations) | 0.03 | 1.9% | 0.2 | 0.32 | `if (((int)threadIdx.x) < 64)` |
| wait.sync (sampled iterations) | 0.02 | 0.9% | 0.2 | 0.06 | `__syncthreads();` |
| wait.sync#4 (sampled iterations) | 0.02 | 0.9% | 0.2 | 0.06 | `__syncthreads();` |
| store.global (sampled iterations) | 0.01 | 0.7% | 0.2 | 0.06 | `if ((((int)threadIdx.x) >> 6) == 0)` |

### per_token_cast_kernel_kernel grid (10, 1, 1) block (64, 1, 1)

51 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 10 warps; span 1.22 us, warp 1.08 us, nsys graphs 1.09 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.45 | 41.9% | 0.44 | 36.3% |
| load | 0.31 | 28.9% | 0.36 | 29.3% |
| store | 0.22 | 20.4% | 0.33 | 26.6% |
| compute | 0.10 | 8.8% | 0.10 | 7.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.22 | 20.6% | 1.0 | 0.29 | `for (int i = 0; i < 2; ++i)` |
| store.global#2 | 0.11 | 10.3% | 1.0 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.10 | 8.8% | 1.0 | 0.13 | `amax_fragment[i_1] = tl::AllReduce<tl::MaxOp, 2, 1, 0, tl::NamedBarrier<64>>::run(amax_fra` |
| load.global#2*2 (est.: sampled iterations x2) | 0.09 | 8.3% | 1.0 | 0.13 | `amax_fragment[i_1] = cutlass::bfloat16_t(__hmax((bfloat16_t(((nv_bfloat162*)(&(amax_fragme` |
| store.global*2 (est.: sampled iterations x2) | 0.06 | 5.3% | 0.6 | 0.13 | `out_sf[((((((int64_t)((int)blockIdx.x)) % (int64_t)15) * (int64_t)32) + (((int64_t)((int)t` |
| store.global#3*2 (est.: sampled iterations x2) | 0.05 | 4.7% | 0.6 | 0.13 | `*(fp8_e4_16_t*)(out + ((((((((int64_t)((int)blockIdx.x)) / (int64_t)3) * (int64_t)6144) + ` |

### finish_argmax_kernel grid (2, 1, 1) block (32, 1, 1)

85 launches, 2 warps; ranges, warp and last warp from the 2 recording CTAs' 2 warps; span 0.74 us, warp 0.74 us, nsys graphs 0.90 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.38 | 52.2% | 0.39 | 52.7% |
| compute | 0.13 | 17.4% | 0.12 | 16.3% |
| load | 0.13 | 17.4% | 0.12 | 17.0% |
| store | 0.10 | 13.0% | 0.10 | 13.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.reduce | 0.13 | 17.4% | 1.0 | 0.13 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| load.global | 0.13 | 17.4% | 1.0 | 0.13 | `for (int k = 0; k < 2; ++k) \| if ((((int)threadIdx.x) < parts))` |
| store.global | 0.10 | 13.0% | 1.0 | 0.10 | `if (((int)threadIdx.x) == 0)` |

### verify_kernel grid (1, 1, 1) block (96, 1, 1)

17 launches, 3 warps; ranges, warp and last warp from the 1 recording CTAs' 3 warps; span 2.85 us, warp 2.72 us, nsys graphs 2.43 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.10 | 40.4% | 1.36 | 37.2% |
| unattributed | 0.99 | 36.5% | 1.26 | 34.5% |
| compute | 0.29 | 10.6% | 0.29 | 8.0% |
| wait | 0.18 | 6.7% | 0.44 | 12.1% |
| store | 0.16 | 5.9% | 0.30 | 8.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.03 | 38.0% | 1.0 | 1.06 | `start = cu_num_logits[((int64_t)((int)blockIdx.x))];` |
| compute.reduce | 0.29 | 10.6% | 1.0 | 0.32 | `row_max[0] = vllm_max(row_max[0], __shfl_xor_sync((uint)4294967295, row_max[0], 8, 32));` |
| store.global | 0.16 | 5.9% | 1.0 | 0.29 | `if (((int)threadIdx.x) == 0)` |
| wait.sync#3 | 0.07 | 2.7% | 1.0 | 0.13 | `__syncthreads();` |
| load.global#2 | 0.06 | 2.4% | 1.0 | 0.06 | `target = local_argmax[((((int64_t)row) * argmax_stride) + ((int64_t)best_index[0]))];` |
| wait.sync#2 | 0.06 | 2.4% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync | 0.04 | 1.6% | 1.0 | 0.06 | `__syncthreads();` |

### adaptive_offsets_kernel grid (1, 1, 1) block (256, 1, 1)

16 launches, 8 warps; ranges, warp and last warp from the 1 recording CTAs' 8 warps; span 3.78 us, warp 3.75 us, nsys graphs 1.95 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 1.81 | 48.2% | 1.14 | 30.0% |
| unattributed, with loops' unsampled iterations | 1.43 | 38.0% | 1.49 | 39.1% |
| store | 0.31 | 8.3% | 0.54 | 14.2% |
| load | 0.20 | 5.4% | 0.64 | 16.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync (sampled iterations) | 0.43 | 11.4% | 1.0 | 0.48 | `__syncthreads();` |
| wait.sync#18 (sampled iterations) | 0.35 | 9.4% | 1.0 | 0.42 | `__syncthreads();` |
| store.global#5 | 0.13 | 3.5% | 1.0 | 0.16 | `for (int chunk_1 = 0; chunk_1 < ((starts_len + 255) >> 8); ++chunk_1)` |
| load.global#2 (sampled iterations) | 0.13 | 3.4% | 1.0 | 0.48 | `if ((((chunk * 256) + ((int)threadIdx.x)) < num_reqs))` |
| store.global#2 (sampled iterations) | 0.09 | 2.4% | 1.0 | 0.10 | `if ((((chunk * 256) + ((int)threadIdx.x)) < num_reqs) & (drafted == 0))` |
| load.global (sampled iterations) | 0.07 | 1.8% | 1.0 | 0.10 | `if (((((chunk * 256) + ((int)threadIdx.x)) < num_reqs) & (drafted != 0)))` |
| store.global | 0.07 | 1.8% | 1.0 | 0.10 | `if (((int)threadIdx.x) == 0)` |
| wait.sync#10 (sampled iterations) | 0.06 | 1.7% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#11 (sampled iterations) | 0.06 | 1.7% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#12 (sampled iterations) | 0.06 | 1.7% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#13 (sampled iterations) | 0.06 | 1.7% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#14 (sampled iterations) | 0.06 | 1.7% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#15 (sampled iterations) | 0.06 | 1.7% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#16 (sampled iterations) | 0.06 | 1.7% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#17 (sampled iterations) | 0.06 | 1.7% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#2 (sampled iterations) | 0.06 | 1.7% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#3 (sampled iterations) | 0.06 | 1.7% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#4 (sampled iterations) | 0.06 | 1.7% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#5 (sampled iterations) | 0.06 | 1.7% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#6 (sampled iterations) | 0.06 | 1.7% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#7 (sampled iterations) | 0.06 | 1.7% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#8 (sampled iterations) | 0.06 | 1.7% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#9 (sampled iterations) | 0.06 | 1.7% | 1.0 | 0.06 | `__syncthreads();` |
| store.global#4 (sampled iterations) | 0.02 | 0.4% | 0.1 | 0.13 | `for (int j = 0; j < 6; ++j)` |
| load.global#3 (sampled iterations) | 0.01 | 0.2% | 0.1 | 0.06 | `state = idx_mapping[((((int64_t)chunk) * (int64_t)256) + ((int64_t)((int)threadIdx.x)))];` |
| store.global#3 (sampled iterations) | 0.01 | 0.2% | 0.1 | 0.06 | `cu_num_logits[(((((int64_t)chunk) * (int64_t)256) + ((int64_t)((int)threadIdx.x))) + (int6` |

### gemm_splitk_reduce_cast_kernel grid (24, 3, 1) block (128, 1, 1)

51 launches, 288 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 1.15 us, warp 1.08 us, nsys graphs 1.28 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.47 | 43.3% | 0.50 | 43.0% |
| unattributed | 0.45 | 41.8% | 0.49 | 42.1% |
| compute | 0.08 | 7.6% | 0.09 | 7.5% |
| store | 0.08 | 7.2% | 0.09 | 7.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.47 | 43.3% | 1.0 | 0.51 | `if ((1 <= M))` |
| compute.reduce | 0.08 | 7.6% | 1.0 | 0.10 | `amax[0] = tl::AllReduce<tl::MaxOp, 8, 1, 0, tl::NamedBarrier<128>>::run(amax[0]);` |
| store.global | 0.08 | 7.2% | 1.0 | 0.10 | `if (((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 5)) < M)` |

### cache_writer_kernel grid (8, 1, 1) block (128, 1, 1)

51 launches, 32 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 0.99 us, warp 0.93 us, nsys graphs 1.12 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.32 | 34.6% | 0.33 | 29.5% |
| load | 0.28 | 30.7% | 0.29 | 26.4% |
| store | 0.25 | 26.8% | 0.41 | 37.1% |
| compute | 0.07 | 8.0% | 0.08 | 7.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.25 | 26.8% | 1.0 | 0.32 | `if (((int64_t)0 <= slot) & ((slot >> (int64_t)7) < ((int64_t)pages))) \| if (((int64_t)0 <=` |
| load.global | 0.23 | 24.4% | 1.0 | 0.26 | `for (int i = 0; i < 4; ++i)` |
| compute.reduce | 0.07 | 8.0% | 1.0 | 0.10 | `amax = max(amax, __shfl_xor_sync((uint)4294967295, amax, 1, 32));` |
| load.global#2 | 0.06 | 6.3% | 1.0 | 0.06 | `int64_t slot = slots[((int64_t)((int)blockIdx.x))];` |

### per_token_cast_kernel_kernel grid (20, 1, 1) block (64, 1, 1)

51 launches, 40 warps; ranges, warp and last warp from the 4 recording CTAs' 8 warps; span 1.09 us, warp 1.04 us, nsys graphs 0.99 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.36 | 34.2% | 0.37 | 34.7% |
| store | 0.30 | 28.5% | 0.30 | 27.7% |
| unattributed | 0.29 | 28.1% | 0.30 | 28.1% |
| compute | 0.10 | 9.2% | 0.10 | 9.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.27 | 25.8% | 1.0 | 0.29 | `for (int i = 0; i < 2; ++i)` |
| store.global#2 | 0.11 | 10.8% | 1.0 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.10 | 9.2% | 1.0 | 0.13 | `amax_fragment[i_1] = tl::AllReduce<tl::MaxOp, 2, 1, 0, tl::NamedBarrier<64>>::run(amax_fra` |
| store.global*2 (est.: sampled iterations x2) | 0.10 | 9.2% | 1.0 | 0.13 | `out_sf[((((((int64_t)((int)blockIdx.x)) % (int64_t)15) * (int64_t)32) + (((int64_t)((int)t` |
| load.global#2*2 (est.: sampled iterations x2) | 0.09 | 8.5% | 1.0 | 0.13 | `amax_fragment[i_1] = cutlass::bfloat16_t(__hmax((bfloat16_t(((nv_bfloat162*)(&(amax_fragme` |
| store.global#3*2 (est.: sampled iterations x2) | 0.09 | 8.5% | 1.0 | 0.13 | `*(fp8_e4_16_t*)(out + ((((((((int64_t)((int)blockIdx.x)) / (int64_t)3) * (int64_t)6144) + ` |

### sink_cut_kernel grid (1, 1, 1) block (128, 1, 1)

34 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 1.73 us, warp 0.60 us, nsys graphs 2.11 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.31 | 52.0% | 0.36 | 22.4% |
| store | 0.17 | 28.0% | 0.70 | 43.5% |
| load | 0.12 | 20.0% | 0.55 | 34.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.17 | 28.0% | 0.6 | 0.48 | `if (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) < num_cut) \| saved[((((int64_t)((int)` |
| load.global | 0.12 | 20.0% | 0.1 | 0.48 | `for (int b = 0; b < num_reqs; ++b)` |

### gemm_splitk_reduce_kernel grid (1, 5, 1) block (128, 1, 1)

64 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 0.83 us, warp 0.82 us, nsys graphs 1.02 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.39 | 47.2% | 0.37 | 43.4% |
| unattributed | 0.38 | 47.0% | 0.43 | 50.5% |
| store | 0.05 | 5.8% | 0.05 | 6.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.39 | 47.2% | 1.0 | 0.42 | `acc[0] = P[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x))` |
| store.global | 0.05 | 5.8% | 1.0 | 0.06 | `C[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x)) * (int64` |

### resample_kernel grid (1, 127, 1) block (128, 1, 1)

17 launches, 508 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 2.94 us, warp 2.18 us, nsys graphs 2.21 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.83 | 83.9% | 2.46 | 84.3% |
| unattributed | 0.31 | 14.1% | 0.42 | 14.3% |
| compute | 0.04 | 2.0% | 0.04 | 1.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.83 | 83.9% | 1.0 | 2.56 | `end = cu_num_logits[(((int64_t)((int)blockIdx.x)) + (int64_t)1)];` |
| compute.reduce | 0.04 | 2.0% | 1.0 | 0.06 | `if ((temp != 0x0p+0f/*0.000000e+00*/) \| bonus)` |

### sparse_mla_kernel grid (1, 2, 6) block (128, 1, 1)

2 launches, 48 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 23.58 us, warp 23.43 us, nsys graphs 16.00 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed, with loops' unsampled iterations | 7.31 | 31.2% | 7.30 | 31.0% |
| load | 5.66 | 24.2% | 8.46 | 35.9% |
| wait | 5.42 | 23.1% | 2.35 | 10.0% |
| compute | 4.71 | 20.1% | 4.93 | 20.9% |
| store | 0.33 | 1.4% | 0.51 | 2.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 | 3.61 | 15.4% | 1.0 | 3.82 | `for (int i = 0; i < 8; ++i)` |
| wait.sync (sampled iterations) | 2.94 | 12.5% | 1.0 | 3.92 | `__syncthreads();` |
| compute.mma*4 (est.: sampled iterations x4) | 1.71 | 7.3% | 1.0 | 1.79 | `for (int ki = 0; ki < 32; ++ki) \| {` |
| compute.mma#2*4 (est.: sampled iterations x4) | 1.17 | 5.0% | 1.0 | 1.22 | `for (int ki_1 = 0; ki_1 < 2; ++ki_1) \| if (((((int)blockIdx.x) * 2) + (c_1 >> 1)) < 3)` |
| load.global (sampled iterations) | 1.03 | 4.4% | 1.0 | 3.94 | `for (int c = 0; c < 4; ++c) \| for (int i = 0; i < 8; ++i) \| for (int i_1 = 0; i_1 < 8; ++i` |
| compute.reduce*8 (est.: sampled iterations x8) | 0.94 | 4.0% | 1.0 | 1.02 | `m_g[i_4] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(m_g[i_4], (&((` |
| compute.reduce#2*8 (est.: sampled iterations x8) | 0.90 | 3.8% | 1.0 | 1.02 | `row_sum[i_7] = tl::AllReduce<tl::SumOp, 128, 32, 0, tl::NamedBarrier<128>>::run(row_sum[i_` |
| load.cp_async#2*4 (est.: sampled iterations x4) | 0.68 | 2.9% | 1.0 | 0.77 | `if (((((int)blockIdx.x) * 2) + (c_1 >> 1)) < 3) \| if ((c_1 < 3) & (c_1 < 3)) \| if (c < 3)` |
| wait.sync#10*8 (est.: sampled iterations x8) | 0.37 | 1.6% | 1.0 | 0.51 | `__syncthreads();` |
| load.cp_async (sampled iterations) | 0.34 | 1.4% | 1.0 | 0.35 | `for (int k_1 = 0; k_1 < 5; ++k_1) \| for (int k_1 = 0; k_1 < 9; ++k_1) \| for (int k_3 = 0; ` |
| store.global | 0.33 | 1.4% | 1.0 | 0.51 | `for (int i_10 = 0; i_10 < 32; ++i_10) \| for (int i_11 = 0; i_11 < 32; ++i_11) \| for (int i` |
| wait.sync#8*8 (est.: sampled iterations x8) | 0.29 | 1.2% | 1.0 | 0.38 | `__syncthreads();` |
| wait.cp_async*4 (est.: sampled iterations x4) | 0.27 | 1.2% | 1.0 | 0.38 | `tl::cp_async_wait<0>();` |
| wait.sync#5*4 (est.: sampled iterations x4) | 0.26 | 1.1% | 1.0 | 0.38 | `__syncthreads();` |
| wait.sync#6*4 (est.: sampled iterations x4) | 0.26 | 1.1% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#9*4 (est.: sampled iterations x4) | 0.26 | 1.1% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#11*4 (est.: sampled iterations x4) | 0.23 | 1.0% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#7*4 (est.: sampled iterations x4) | 0.23 | 1.0% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#4*4 (est.: sampled iterations x4) | 0.21 | 0.9% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#3 | 0.07 | 0.3% | 1.0 | 0.08 | `__syncthreads();` |
| wait.sync#2 | 0.04 | 0.2% | 1.0 | 0.06 | `__syncthreads();` |

### compute_slot_mappings_kernel grid (1, 11, 1) block (256, 1, 1)

17 launches, 88 warps; ranges, warp and last warp from the 6 recording CTAs' 48 warps; span 2.66 us, warp 1.45 us, nsys graphs 2.74 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.80 | 55.3% | 0.80 | 30.7% |
| unattributed | 0.65 | 44.7% | 1.80 | 69.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.80 | 55.3% | 1.0 | 0.83 | `int first_token = query_start_loc[0];` |

### sink_policy_kernel grid (1, 1, 1) block (32, 1, 1)

17 launches, 1 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 2.56 us, warp 2.56 us, nsys graphs 2.62 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.09 | 42.5% | 1.12 | 42.8% |
| unattributed | 0.90 | 35.0% | 0.91 | 34.9% |
| store | 0.58 | 22.5% | 0.59 | 22.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 | 0.67 | 26.2% | 1.0 | 0.67 | `state[0] = idx_mapping[((int64_t)((int)blockIdx.x))];` |
| store.global#3 | 0.51 | 20.0% | 1.0 | 0.51 | `for (int a = 0; a < 3; ++a) \| recovered_logprobs[((int64_t)((int)blockIdx.x))] = 0x0p+0f/*` |
| load.global | 0.22 | 8.8% | 1.0 | 0.22 | `start[0] = cu_num_logits[((int64_t)((int)blockIdx.x))];` |
| load.global#4 | 0.16 | 6.2% | 1.0 | 0.16 | `for (int r = 0; r < 2; ++r) \| for (int r_2 = 0; r_2 < 8; ++r_2)` |
| store.global#4 | 0.06 | 2.5% | 1.0 | 0.06 | `events[(((int64_t)((int)blockIdx.x)) * (int64_t)6)] = outcome[0];` |
| load.global#5 | 0.03 | 1.2% | 1.0 | 0.03 | `AtomicAdd((&(counts[0])), draft_zero_flag[0]);` |

### norm_forward_and_per_token_cast_kernel_kernel grid (8, 1, 1) block (128, 1, 1)

17 launches, 32 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 2.34 us, warp 2.29 us, nsys graphs 1.79 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 2.19 | 95.9% | 3.21 | 127.3% |
| load | 0.42 | 18.4% | 0.41 | 16.4% |
| compute | 0.13 | 5.8% | 0.13 | 5.2% |
| wait | 0.06 | 2.4% | 0.06 | 2.5% |
| unattributed | 0.00 | 0.0% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*5 (est.: sampled iterations x5) | 1.92 | 83.9% | 1.0 | 2.08 | `for (int j_1 = 0; j_1 < 2; ++j_1)` |
| store.global*5 (est.: sampled iterations x5) | 0.22 | 9.6% | 1.0 | 0.32 | `for (int j = 0; j < 4; ++j)` |
| load.global#2*5 (est.: sampled iterations x5) | 0.21 | 9.2% | 1.0 | 0.32 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 1024) + (((int)threadIdx.x) * 8)` |
| load.global*5 (est.: sampled iterations x5) | 0.21 | 9.2% | 1.0 | 0.32 | `*(uint4*)(x_local + (vi * 8)) = *(uint4*)(x + ((((((int64_t)vi) * (int64_t)1024) + (((int6` |
| compute.reduce | 0.13 | 5.8% | 1.0 | 0.16 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| wait.sync | 0.06 | 2.4% | 1.0 | 0.10 | `__syncthreads();` |
| store.global#2 | 0.04 | 1.7% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| store.global#3 | 0.01 | 0.6% | 0.2 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |

### metadata_pages_kernel grid (5, 1, 1) block (128, 1, 1)

32 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 1.31 us, warp 0.95 us, nsys graphs 0.00 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.72 | 76.1% | 0.79 | 59.8% |
| load | 0.14 | 15.1% | 0.19 | 14.8% |
| store | 0.08 | 8.7% | 0.34 | 25.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.13 | 13.4% | 1.0 | 0.13 | `if ((((offset + ((int)blockIdx.x)) < src_rows) && (0 <= (offset + ((int)blockIdx.x)))))` |
| store.global | 0.08 | 8.7% | 0.2 | 0.35 | `if (((int)threadIdx.x) < 32)` |
| load.global#2 | 0.02 | 1.7% | 0.2 | 0.06 | `if (((req < table_rows) && (((int)threadIdx.x) < table_cols)))` |

### norm_forward_and_per_token_cast_kernel_kernel grid (10, 1, 1) block (128, 1, 1)

17 launches, 40 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 2.34 us, warp 2.29 us, nsys graphs 1.82 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 2.17 | 95.0% | 2.20 | 94.2% |
| load | 0.43 | 18.9% | 0.48 | 20.5% |
| compute | 0.13 | 5.7% | 0.13 | 5.6% |
| wait | 0.06 | 2.5% | 0.06 | 2.5% |
| unattributed | 0.00 | 0.0% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*5 (est.: sampled iterations x5) | 1.90 | 82.8% | 1.0 | 2.08 | `for (int j_1 = 0; j_1 < 2; ++j_1)` |
| store.global*5 (est.: sampled iterations x5) | 0.22 | 9.8% | 1.0 | 0.32 | `for (int j = 0; j < 4; ++j)` |
| load.global#2*5 (est.: sampled iterations x5) | 0.22 | 9.4% | 1.0 | 0.32 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 1024) + (((int)threadIdx.x) * 8)` |
| load.global*5 (est.: sampled iterations x5) | 0.22 | 9.4% | 1.0 | 0.32 | `*(uint4*)(x_local + (vi * 8)) = *(uint4*)(x + ((((((int64_t)vi) * (int64_t)1024) + (((int6` |
| compute.reduce | 0.13 | 5.7% | 1.0 | 0.16 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| wait.sync | 0.06 | 2.5% | 1.0 | 0.10 | `__syncthreads();` |
| store.global#2 | 0.04 | 1.8% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| store.global#3 | 0.01 | 0.6% | 0.2 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |

### sink_flags_kernel grid (1, 1, 1) block (128, 1, 1)

17 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 2.24 us, warp 0.73 us, nsys graphs 3.10 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.39 | 53.8% | 1.54 | 69.6% |
| unattributed | 0.25 | 34.1% | 0.34 | 15.1% |
| store | 0.09 | 12.1% | 0.34 | 15.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.39 | 53.8% | 0.2 | 1.57 | `token = logits_indices[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)th` |
| store.global | 0.09 | 12.1% | 0.2 | 0.35 | `flags[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = (` |

### cache_writer_kernel grid (6, 1, 1) block (128, 1, 1)

8 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 4.05 us, warp 3.44 us, nsys graphs 3.46 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 2.14 | 62.1% | 2.38 | 51.3% |
| store | 0.94 | 27.3% | 1.88 | 40.4% |
| load | 0.29 | 8.4% | 0.30 | 6.5% |
| compute | 0.07 | 2.1% | 0.08 | 1.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.94 | 27.3% | 1.0 | 1.38 | `if (((int64_t)0 <= slot) & ((slot >> (int64_t)7) < ((int64_t)pages))) \| if (((int64_t)0 <=` |
| load.global | 0.23 | 6.7% | 1.0 | 0.26 | `for (int i = 0; i < 4; ++i)` |
| compute.reduce | 0.07 | 2.1% | 1.0 | 0.10 | `amax = max(amax, __shfl_xor_sync((uint)4294967295, amax, 1, 32));` |
| load.global#2 | 0.06 | 1.7% | 1.0 | 0.06 | `int64_t slot = slots[((int64_t)((int)blockIdx.x))];` |

### post_update_kernel grid (1, 1, 1) block (32, 1, 1)

17 launches, 1 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 1.82 us, warp 1.82 us, nsys graphs 2.21 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.67 | 36.8% | 0.68 | 33.6% |
| unattributed | 0.58 | 31.6% | 0.60 | 29.9% |
| store | 0.58 | 31.6% | 0.74 | 36.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 | 0.48 | 26.3% | 1.0 | 0.48 | `int count = num_sampled[((((int64_t)((int)blockIdx.x)) * (int64_t)32) + ((int64_t)((int)th` |
| store.global#2 | 0.42 | 22.8% | 1.0 | 0.42 | `if (0 < count)` |
| store.global | 0.16 | 8.8% | 1.0 | 0.16 | `if (0 < count)` |
| load.global | 0.10 | 5.3% | 1.0 | 0.10 | `int64_t req = idx_mapping[((((int64_t)((int)blockIdx.x)) * (int64_t)32) + ((int64_t)((int)` |
| load.global#3 | 0.10 | 5.3% | 1.0 | 0.10 | `if ((0 < count))` |

### rope_kernel grid (5, 1, 1) block (256, 1, 1)

32 launches, 40 warps; ranges, warp and last warp from the 5 recording CTAs' 40 warps; span 1.06 us, warp 0.97 us, nsys graphs 1.30 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.67 | 69.3% | 0.73 | 70.4% |
| load | 0.26 | 26.6% | 0.27 | 25.7% |
| store | 0.04 | 4.1% | 0.04 | 3.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.26 | 26.6% | 1.0 | 0.29 | `if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < tokens)) \| int64_t position ` |
| store.global | 0.04 | 4.1% | 1.0 | 0.06 | `*(uint4*)(x + ((((((int64_t)((int)blockIdx.x)) * (int64_t)12288) + ((((int64_t)((int)threa` |

### dspark_confidence_kernel grid (10, 1, 1) block (128, 1, 1)

17 launches, 40 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 1.95 us, warp 1.76 us, nsys graphs 1.86 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.39 | 78.8% | 1.51 | 77.2% |
| load | 0.17 | 9.7% | 0.21 | 10.8% |
| compute | 0.10 | 5.4% | 0.09 | 4.8% |
| wait | 0.08 | 4.7% | 0.05 | 2.4% |
| store | 0.02 | 1.3% | 0.09 | 4.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.16 | 9.1% | 1.0 | 0.16 | `for (int c = 0; c < 5; ++c)` |
| compute.reduce | 0.10 | 5.4% | 1.0 | 0.10 | `acc[0] = tl::warp_reduce_sum(acc[0]);` |
| wait.sync | 0.08 | 4.7% | 1.0 | 0.22 | `__syncthreads();` |
| store.global | 0.02 | 1.3% | 0.2 | 0.10 | `out[((int64_t)((int)blockIdx.x))] = (0x1p+0f/*1.000000e+00*/ / (0x1p+0f/*1.000000e+00*/ + ` |
| load.global#2 | 0.01 | 0.6% | 0.2 | 0.06 | `total[0] = 0x0p+0f/*0.000000e+00*/;` |

### insert_kernel grid (1, 1, 1) block (32, 1, 1)

17 launches, 1 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 1.57 us, warp 1.57 us, nsys graphs 1.38 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.70 | 44.9% | 0.96 | 50.1% |
| unattributed | 0.42 | 26.5% | 0.42 | 21.9% |
| store | 0.42 | 26.5% | 0.50 | 25.9% |
| compute | 0.03 | 2.0% | 0.04 | 2.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#3 | 0.51 | 32.7% | 1.0 | 0.51 | `int state = mapping[((int64_t)position)]; \| int64_t state = mapping[((int64_t)position)];` |
| store.global | 0.42 | 26.5% | 1.0 | 0.42 | `counts[((int64_t)((int)blockIdx.x))] = (count_local + 1); \| counts[((int64_t)((int)blockId` |
| load.global | 0.13 | 8.2% | 1.0 | 0.13 | `for (int k = 0; k < 4; ++k)` |
| load.global#2 | 0.06 | 4.1% | 1.0 | 0.06 | `count_local = counts[((int64_t)((int)blockIdx.x))];` |
| compute.reduce | 0.03 | 2.0% | 1.0 | 0.03 | `int count = __shfl_sync((uint)4294967295, count_local, 0, 32);` |

### gather_block_tables_kernel grid (11, 1, 1) block (256, 1, 1)

17 launches, 88 warps; ranges, warp and last warp from the 6 recording CTAs' 48 warps; span 1.92 us, warp 1.37 us, nsys graphs 0.00 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.06 | 77.1% | 1.58 | 83.6% |
| unattributed | 0.31 | 22.9% | 0.31 | 16.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.06 | 77.1% | 1.0 | 1.60 | `int g = chunk_groups[((int64_t)((int)blockIdx.x))];` |

### sink_rows_kernel grid (1, 64, 1) block (256, 1, 1)

17 launches, 512 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 1.54 us, warp 0.97 us, nsys graphs 1.74 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.38 | 39.0% | 0.46 | 26.2% |
| load | 0.31 | 32.3% | 0.44 | 25.0% |
| store | 0.28 | 28.7% | 0.86 | 48.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.31 | 31.9% | 1.0 | 0.38 | `row = (max((((int64_t)num_sampled[((int64_t)((int)blockIdx.x))]) - (int64_t)1), (int64_t)0` |
| store.global | 0.27 | 27.7% | 1.0 | 0.48 | `for (int k_1 = 0; k_1 < 8; ++k_1)` |
| store.global#2 | 0.01 | 1.0% | 0.0 | 0.32 | `tokens[((((int64_t)((int)blockIdx.x)) * (int64_t)3) + ((int64_t)((int)threadIdx.x)))] = to` |
| load.global#2 | 0.00 | 0.4% | 0.0 | 0.13 | `token = max(alternates[((((int64_t)((int)blockIdx.x)) * (int64_t)3) + ((int64_t)((int)thre` |

### topk_ids_kernel grid (6, 1, 1) block (512, 1, 1)

1 launches, 96 warps; ranges, warp and last warp from the 6 recording CTAs' 96 warps; span 28.83 us, warp 25.94 us, nsys graphs 25.12 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 22.30 | 86.0% | 24.48 | 84.9% |
| unattributed | 1.95 | 7.5% | 1.92 | 6.7% |
| wait | 1.39 | 5.4% | 0.93 | 3.2% |
| store | 0.30 | 1.2% | 1.50 | 5.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 22.30 | 86.0% | 1.0 | 25.22 | `for (int t = 0; t < (vocab >> 14); ++t)` |
| wait.sync | 1.24 | 4.8% | 1.0 | 3.39 | `__syncthreads();` |
| store.global | 0.30 | 1.2% | 1.0 | 3.97 | `if ((((int)threadIdx.x) >> 5) == 0)` |
| wait.sync#2 | 0.08 | 0.3% | 1.0 | 0.16 | `__syncthreads();` |
| wait.sync#3 | 0.06 | 0.2% | 1.0 | 0.10 | `__syncthreads();` |

### context_inputs_kernel grid (5, 8, 1) block (128, 1, 1)

17 launches, 160 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 1.60 us, warp 1.24 us, nsys graphs 2.00 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.82 | 66.5% | 1.27 | 75.9% |
| unattributed | 0.34 | 27.4% | 0.34 | 20.1% |
| store | 0.08 | 6.1% | 0.07 | 4.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.82 | 66.5% | 1.0 | 1.18 | `vllm_dspark_context_b128((&(aux[(((((int64_t)((int)blockIdx.y)) * (int64_t)15360) + (((int` |
| store.global | 0.08 | 6.1% | 1.0 | 0.16 | `if (((((int)blockIdx.x) == 0) & (((int)threadIdx.x) == 0)) & (num_tokens <= ((int)blockIdx` |

### gemm_splitk_reduce_kernel grid (4, 8, 1) block (128, 1, 1)

51 launches, 128 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 0.54 us, warp 0.53 us, nsys graphs 0.70 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.26 | 48.5% | 0.26 | 48.5% |
| load | 0.23 | 43.2% | 0.23 | 43.1% |
| store | 0.04 | 8.3% | 0.05 | 8.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.23 | 43.2% | 1.0 | 0.26 | `acc[0] = P[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x))` |
| store.global | 0.04 | 8.3% | 1.0 | 0.06 | `C[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x)) * (int64` |

### assign_draft_budget_kernel grid (1, 1, 1) block (256, 1, 1)

16 launches, 8 warps; ranges, warp and last warp from the 1 recording CTAs' 8 warps; span 1.76 us, warp 1.75 us, nsys graphs 1.86 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 0.77 | 44.0% | 0.50 | 28.4% |
| unattributed, with loops' unsampled iterations | 0.55 | 31.5% | 0.61 | 34.7% |
| load | 0.24 | 13.5% | 0.45 | 25.7% |
| store | 0.19 | 11.0% | 0.20 | 11.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync | 0.53 | 30.1% | 1.0 | 0.61 | `__syncthreads();` |
| load.global | 0.24 | 13.5% | 1.0 | 0.74 | `for (int chunk = 0; chunk < ((num_reqs + 255) >> 8); ++chunk)` |
| store.global | 0.19 | 11.0% | 1.0 | 0.19 | `for (int chunk_2 = 0; chunk_2 < ((num_reqs + 255) >> 8); ++chunk_2)` |
| wait.sync#3 | 0.18 | 10.2% | 1.0 | 0.21 | `__syncthreads();` |
| wait.sync#2 (sampled iterations) | 0.06 | 3.7% | 1.0 | 0.06 | `__syncthreads();` |

### metadata_decode_kernel grid (6, 2, 1) block (128, 1, 1)

4 launches, 48 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 5.98 us, warp 4.39 us, nsys graphs 2.19 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 2.69 | 61.3% | 4.47 | 69.6% |
| unattributed | 1.55 | 35.3% | 1.10 | 17.1% |
| store | 0.15 | 3.4% | 0.86 | 13.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#3 | 1.40 | 31.9% | 0.5 | 2.83 | `if ((((offset + ((int)blockIdx.x)) < main_src_rows) && (0 <= (offset + ((int)blockIdx.x)))` |
| load.global | 1.27 | 28.9% | 0.5 | 3.31 | `if ((((offset + ((int)blockIdx.x)) < src_rows) && (0 <= (offset + ((int)blockIdx.x)))))` |
| store.global#4 | 0.11 | 2.6% | 0.5 | 0.72 | `if (0 < live)` |
| store.global | 0.03 | 0.6% | 0.5 | 0.06 | `swa[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = con` |
| load.global#2 | 0.03 | 0.6% | 0.1 | 0.21 | `if (((0 <= req) & ((int64_t)0 <= pos)))` |
| store.global#2 | 0.01 | 0.1% | 0.1 | 0.05 | `lengths[((int64_t)((int)blockIdx.x))] = ((int)max(info[1], (int64_t)0));` |
| store.global#3 | 0.01 | 0.1% | 0.1 | 0.05 | `top_lengths[((int64_t)((int)blockIdx.x))] = condval_6; \| top_lengths[((int64_t)((int)block` |

### combine_sampled_and_draft_tokens_kernel grid (1, 1, 1) block (32, 1, 1)

17 launches, 1 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 1.41 us, warp 1.41 us, nsys graphs 1.73 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.90 | 63.6% | 0.91 | 64.2% |
| unattributed | 0.42 | 29.5% | 0.41 | 28.9% |
| store | 0.10 | 6.8% | 0.10 | 6.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.90 | 63.6% | 1.0 | 0.90 | `if (((((int)blockIdx.x) * 32) + ((int)threadIdx.x)) < num_reqs) \| req = idx_mapping[((int6` |
| store.global | 0.10 | 6.8% | 1.0 | 0.10 | `if ((prefill < seq) && (1 < num_logits))` |

### mhc_post_kernel grid (50, 1, 1) block (128, 1, 1)

17 launches, 200 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 1.38 us, warp 1.10 us, nsys graphs 1.60 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.92 | 83.8% | 1.14 | 85.0% |
| load | 0.12 | 11.4% | 0.14 | 10.6% |
| store | 0.05 | 4.8% | 0.06 | 4.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.12 | 11.4% | 1.0 | 0.16 | `*(float4*)(pm + 0) = *(float4*)(post + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int` |
| store.global | 0.05 | 4.8% | 1.0 | 0.10 | `for (int n_1 = 0; n_1 < 4; ++n_1)` |

### moe_swiglu_cast_kernel_kernel grid (16, 36, 1) block (96, 1, 1)

6 launches, 1728 warps; ranges, warp and last warp from the 4 recording CTAs' 12 warps; span 3.65 us, warp 1.30 us, nsys graphs 1.86 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.03 | 79.3% | 3.15 | 82.9% |
| load | 0.20 | 15.4% | 0.38 | 10.1% |
| compute | 0.04 | 2.9% | 0.14 | 3.6% |
| store | 0.03 | 2.5% | 0.13 | 3.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.20 | 15.4% | 1.0 | 0.26 | `for (int r = 0; r < 2; ++r)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.04 | 2.9% | 0.2 | 0.19 | `if (0 <= orders[r_2])` |
| store.global*2 (est.: sampled iterations x2) | 0.03 | 2.5% | 0.2 | 0.13 | `if (0 <= orders[r_2])` |

### per_token_cast_kernel_kernel grid (60, 1, 1) block (64, 1, 1)

17 launches, 120 warps; ranges, warp and last warp from the 4 recording CTAs' 8 warps; span 1.18 us, warp 1.12 us, nsys graphs 1.26 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.40 | 35.9% | 0.45 | 33.8% |
| store | 0.31 | 27.8% | 0.30 | 22.9% |
| unattributed | 0.31 | 27.8% | 0.48 | 36.2% |
| compute | 0.10 | 8.5% | 0.09 | 7.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.32 | 28.1% | 1.0 | 0.35 | `for (int i = 0; i < 2; ++i)` |
| store.global#2 | 0.11 | 10.0% | 1.0 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3)` |
| store.global*2 (est.: sampled iterations x2) | 0.10 | 9.3% | 1.0 | 0.13 | `out_sf[((((((int64_t)((int)blockIdx.x)) % (int64_t)15) * (int64_t)32) + (((int64_t)((int)t` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.10 | 8.5% | 1.0 | 0.13 | `amax_fragment[i_1] = tl::AllReduce<tl::MaxOp, 2, 1, 0, tl::NamedBarrier<64>>::run(amax_fra` |
| store.global#3*2 (est.: sampled iterations x2) | 0.10 | 8.5% | 1.0 | 0.13 | `*(fp8_e4_16_t*)(out + ((((((((int64_t)((int)blockIdx.x)) / (int64_t)3) * (int64_t)6144) + ` |
| load.global#2*2 (est.: sampled iterations x2) | 0.09 | 7.8% | 1.0 | 0.13 | `amax_fragment[i_1] = cutlass::bfloat16_t(__hmax((bfloat16_t(((nv_bfloat162*)(&(amax_fragme` |

### indexer_topk_segment_kernel grid (1, 6, 1) block (512, 1, 1)

2 launches, 96 warps; ranges, warp and last warp from the 6 recording CTAs' 96 warps; span 10.64 us, warp 10.58 us, nsys graphs 0.00 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed, with loops' unsampled iterations | 5.36 | 50.6% | 5.17 | 48.6% |
| wait | 3.99 | 37.7% | 4.14 | 38.9% |
| load | 1.07 | 10.1% | 1.10 | 10.4% |
| store | 0.17 | 1.6% | 0.22 | 2.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.07 | 10.1% | 1.0 | 1.14 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 4096); \| i` |
| wait.sync (sampled iterations) | 1.01 | 9.5% | 14.5 | 1.23 | `__syncthreads();` |
| wait.sync#24*9 (est.: sampled iterations x9) | 0.37 | 3.5% | 0.5 | 0.43 | `__syncthreads();` |
| wait.sync#5*8 (est.: sampled iterations x8) | 0.31 | 3.0% | 0.5 | 0.38 | `__syncthreads();` |
| wait.sync#20*9 (est.: sampled iterations x9) | 0.31 | 2.9% | 0.5 | 0.43 | `__syncthreads();` |
| wait.sync#23*9 (est.: sampled iterations x9) | 0.31 | 2.9% | 0.5 | 0.43 | `__syncthreads();` |
| wait.sync#19*9 (est.: sampled iterations x9) | 0.30 | 2.9% | 0.5 | 0.43 | `__syncthreads();` |
| wait.sync#14*8 (est.: sampled iterations x8) | 0.29 | 2.8% | 0.5 | 0.38 | `__syncthreads();` |
| wait.sync#6*8 (est.: sampled iterations x8) | 0.28 | 2.6% | 0.5 | 0.38 | `__syncthreads();` |
| wait.sync#13*8 (est.: sampled iterations x8) | 0.27 | 2.6% | 0.5 | 0.38 | `__syncthreads();` |
| store.global | 0.17 | 1.6% | 1.5 | 0.22 | `for (int i_3 = 0; i_3 < 4; ++i_3) \| if (((int)threadIdx.x) < total) \| if (((int)threadIdx.` |
| wait.sync#25 | 0.05 | 0.5% | 0.5 | 0.08 | `__syncthreads();` |
| wait.sync#17 | 0.04 | 0.4% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#18 | 0.04 | 0.4% | 0.5 | 0.08 | `__syncthreads();` |
| wait.sync#16 | 0.04 | 0.4% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#8 | 0.04 | 0.4% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#9 | 0.04 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#2 | 0.04 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#10 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#3 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#11 | 0.03 | 0.3% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#12 | 0.03 | 0.3% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#21 | 0.03 | 0.3% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#15 | 0.03 | 0.3% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#7 | 0.03 | 0.3% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#22 | 0.03 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#4 | 0.03 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |

### logits_rows_kernel grid (1, 1, 1) block (128, 1, 1)

17 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 1.18 us, warp 1.11 us, nsys graphs 2.05 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.51 | 46.0% | 0.51 | 42.4% |
| unattributed | 0.46 | 41.7% | 0.30 | 24.5% |
| store | 0.14 | 12.2% | 0.40 | 33.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.51 | 46.0% | 1.0 | 0.51 | `index = logits_indices[min(((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((in` |
| store.global | 0.14 | 12.2% | 1.0 | 0.35 | `if (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) < num_logits)` |

### token_counts_kernel grid (1, 1, 1) block (128, 1, 1)

17 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 1.12 us, warp 0.47 us, nsys graphs 1.31 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.27 | 57.6% | 0.32 | 28.2% |
| store | 0.14 | 30.5% | 0.60 | 52.4% |
| load | 0.06 | 11.9% | 0.22 | 19.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.14 | 30.5% | 0.2 | 0.58 | `if (((int)threadIdx.x) == 0) \| sampled[((int64_t)((int)blockIdx.x))] = (((int)(prefill_len` |
| load.global | 0.06 | 11.9% | 0.2 | 0.22 | `int count = sampled[((int64_t)((int)blockIdx.x))];` |

### flatten_sampled_kernel grid (1, 1, 1) block (128, 1, 1)

17 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 1.12 us, warp 0.83 us, nsys graphs 1.28 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.47 | 56.7% | 0.63 | 55.6% |
| unattributed | 0.34 | 40.4% | 0.41 | 36.6% |
| store | 0.02 | 2.9% | 0.09 | 7.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.42 | 50.0% | 1.0 | 0.42 | `for (int r = 0; r < num_reqs; ++r)` |
| load.global#2 | 0.06 | 6.7% | 0.2 | 0.22 | `int i = (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) - cu_num_logits[req]);` |
| store.global | 0.02 | 2.9% | 0.2 | 0.10 | `flat[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = co` |

### prepare_pos_seq_lens_kernel grid (2, 1, 1) block (256, 1, 1)

17 launches, 16 warps; ranges, warp and last warp from the 2 recording CTAs' 16 warps; span 1.09 us, warp 0.75 us, nsys graphs 1.31 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.32 | 42.4% | 0.36 | 32.8% |
| load | 0.30 | 40.3% | 0.60 | 54.7% |
| store | 0.13 | 17.2% | 0.14 | 12.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.30 | 40.3% | 0.5 | 0.61 | `int computed = num_computed_tokens[idx_mapping[((int64_t)((int)blockIdx.x))]];` |
| store.global | 0.07 | 8.8% | 0.5 | 0.16 | `for (int it = 0; it < (((max_reqs + 255) - num_reqs) >> 8); ++it)` |
| store.global#2 | 0.06 | 8.5% | 0.5 | 0.13 | `if (((int)threadIdx.x) == 0)` |

### metadata_chunk_kernel grid (6, 1, 1) block (128, 1, 1)

2 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 9.25 us, warp 6.63 us, nsys graphs 4.88 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 4.79 | 72.3% | 7.25 | 78.4% |
| unattributed | 1.76 | 26.6% | 1.86 | 20.1% |
| store | 0.08 | 1.1% | 0.14 | 1.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 4.23 | 63.8% | 1.0 | 5.04 | `if ((((offset + ((int)blockIdx.x)) < src_rows) && (0 <= (offset + ((int)blockIdx.x)))))` |
| load.global#2 | 0.56 | 8.4% | 0.2 | 2.24 | `if (((0 <= req) & ((int64_t)0 <= pos)))` |
| store.global | 0.05 | 0.8% | 1.0 | 0.06 | `swa[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = con` |
| store.global#2 | 0.01 | 0.2% | 0.2 | 0.06 | `lengths[((int64_t)((int)blockIdx.x))] = ((int)max(info[1], (int64_t)0));` |
| store.global#3 | 0.01 | 0.2% | 0.2 | 0.05 | `top_lengths[((int64_t)((int)blockIdx.x))] = condval_12; \| top_lengths[((int64_t)((int)bloc` |

### mhc_post_kernel grid (30, 1, 1) block (128, 1, 1)

4 launches, 120 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 3.70 us, warp 2.00 us, nsys graphs 1.31 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.84 | 92.4% | 3.50 | 95.6% |
| load | 0.10 | 5.3% | 0.11 | 3.1% |
| store | 0.05 | 2.4% | 0.05 | 1.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.10 | 5.3% | 1.0 | 0.13 | `*(float4*)(pm + 0) = *(float4*)(post + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int` |
| store.global | 0.05 | 2.4% | 1.0 | 0.06 | `for (int n_1 = 0; n_1 < 4; ++n_1)` |

### topk_log_softmax_kernel grid (6, 1, 1) block (1024, 1, 1)

1 launches, 192 warps; ranges, warp and last warp from the 6 recording CTAs' 192 warps; span 17.47 us, warp 16.27 us, nsys graphs 16.16 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 13.27 | 81.5% | 14.43 | 82.9% |
| unattributed, with loops' unsampled iterations | 1.59 | 9.8% | 1.73 | 9.9% |
| wait | 1.02 | 6.3% | 0.80 | 4.6% |
| compute | 0.39 | 2.4% | 0.38 | 2.2% |
| store | 0.00 | 0.0% | 0.06 | 0.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 6.73 | 41.4% | 1.0 | 7.23 | `if (0 < (vocab & 2047)) \| if (0 < (vocab & 8191))` |
| load.global#2 | 6.52 | 40.1% | 1.0 | 7.33 | `for (int step_1 = 0; step_1 < (vocab >> 11); ++step_1) \| for (int step_1 = 0; step_1 < (vo` |
| wait.sync#2 | 0.48 | 2.9% | 1.0 | 0.96 | `__syncthreads();` |
| wait.sync | 0.28 | 1.7% | 1.0 | 0.83 | `__syncthreads();` |
| compute.reduce#3 | 0.17 | 1.1% | 1.0 | 0.22 | `ties[0] = vllm_add(ties[0], __shfl_xor_sync((uint)4294967295, ties[0], 16, 32));` |
| compute.reduce#2 | 0.11 | 0.7% | 1.0 | 0.16 | `se[0] = vllm_add(se[0], __shfl_xor_sync((uint)4294967295, se[0], 16, 32));` |
| compute.reduce | 0.11 | 0.7% | 1.0 | 0.13 | `best[0] = vllm_max(best[0], __shfl_xor_sync((uint)4294967295, best[0], 16, 32));` |
| wait.sync#4 | 0.10 | 0.6% | 1.0 | 0.16 | `__syncthreads();` |
| wait.sync#5 | 0.08 | 0.5% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#3 | 0.08 | 0.5% | 1.0 | 0.10 | `__syncthreads();` |
| load.global#3 (sampled iterations) | 0.01 | 0.1% | 0.0 | 0.38 | `float value_32 = vllm_sub(vllm_shifted(logits[((((int64_t)((int)blockIdx.x)) * logits_stri` |
| store.global (sampled iterations) | 0.00 | 0.0% | 0.0 | 0.06 | `output[(((((int64_t)c) * (int64_t)1024) + (((int64_t)((int)blockIdx.x)) * ((int64_t)topk))` |

### mhc_collapse_kernel grid (50, 1, 1) block (128, 1, 1)

17 launches, 200 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 0.90 us, warp 0.76 us, nsys graphs 1.09 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.61 | 80.0% | 0.75 | 82.4% |
| load | 0.10 | 13.7% | 0.11 | 11.6% |
| store | 0.05 | 6.3% | 0.05 | 6.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.10 | 13.7% | 1.0 | 0.13 | `*(float4*)(mv + 0) = *(float4*)(mix + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int6` |
| store.global | 0.05 | 6.3% | 1.0 | 0.06 | `*(uint4*)(out + ((((int64_t)((int)blockIdx.x)) * (int64_t)1024) + (((int64_t)((int)threadI` |

### scatter_rows_gather_kernel grid (1, 1, 1) block (128, 1, 1)

17 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 0.90 us, warp 0.82 us, nsys graphs 1.79 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.57 | 68.9% | 0.39 | 42.5% |
| store | 0.15 | 18.4% | 0.41 | 45.0% |
| load | 0.10 | 12.6% | 0.11 | 12.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.15 | 18.4% | 1.0 | 0.42 | `if (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) < (num_reqs * width))` |
| load.global | 0.10 | 12.6% | 1.0 | 0.13 | `index = idx_mapping[min((((((int64_t)0 <= ((int64_t)width)) && ((int64_t)0 <= rmod_1)) \|\| ` |

### ced_copy_rows_kernel grid (50, 1, 1) block (128, 1, 1)

17 launches, 200 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 0.90 us, warp 0.84 us, nsys graphs 1.18 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.59 | 69.8% | 0.63 | 70.6% |
| unattributed | 0.25 | 30.2% | 0.26 | 29.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.59 | 69.8% | 1.0 | 0.64 | `for (int k = 0; k < 4; ++k) \| int64_t index = ced_ldg64((&(row_indices[((((int64_t)blockId` |

### mhc_collapse_column_kernel grid (25, 1, 1) block (128, 1, 1)

16 launches, 100 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 0.80 us, warp 0.78 us, nsys graphs 1.09 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.62 | 80.3% | 0.62 | 65.9% |
| load | 0.10 | 12.8% | 0.26 | 27.9% |
| store | 0.05 | 6.9% | 0.06 | 6.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.10 | 12.8% | 1.0 | 0.13 | `*(float4*)(mv + 0) = *(float4*)(mix + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int6` |
| store.global | 0.05 | 6.9% | 1.0 | 0.06 | `*(uint4*)(out + ((((int64_t)((int)blockIdx.x)) * (int64_t)1024) + (((int64_t)((int)threadI` |

### ced_requests_kernel grid (1, 1, 1) block (128, 1, 1)

17 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 0.90 us, warp 0.61 us, nsys graphs 1.18 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.36 | 59.2% | 0.35 | 40.1% |
| store | 0.15 | 25.0% | 0.42 | 48.0% |
| load | 0.10 | 15.8% | 0.11 | 11.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.15 | 25.0% | 1.0 | 0.42 | `if (((int)threadIdx.x) < 2) \| if (((int)threadIdx.x) <= 2)` |
| load.global | 0.10 | 15.8% | 1.0 | 0.10 | `uint v_ = tl::load_global_32((&(starts[1]))); \| uint v_ = tl::load_global_32((&(starts[min` |

### sparse_mla_kernel_1 grid (24, 6, 1) block (128, 1, 1)

6 launches, 144 CTAs, 576 warps; span 2.18 us, warp 1.60 us, nsys graphs 2.21 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.57 | 97.8% | 1.10 | 48.7% |
| load | 0.02 | 1.5% | 0.94 | 41.8% |
| wait | 0.01 | 0.5% | 0.10 | 4.3% |
| compute | 0.00 | 0.1% | 0.08 | 3.5% |
| store | 0.00 | 0.1% | 0.04 | 1.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.02 | 1.5% | 0.1 | 1.15 | `M = __fmul_rn(Sinks[((int)blockIdx.x)], 0x1.7154764ee6c2fp+0f/*1.442695e+00*/); \| if (((in` |
| wait.sync | 0.01 | 0.5% | 0.1 | 0.46 | `__syncthreads();` |
| compute.reduce | 0.00 | 0.1% | 0.0 | 0.10 | `amax[0] = tl::AllReduce<tl::MaxOp, 8, 1, 0, tl::NamedBarrier<128>>::run(amax[0]);` |
| store.global | 0.00 | 0.1% | 0.0 | 0.06 | `*(uint2*)(Output + (((((int64_t)((int)blockIdx.y)) * (int64_t)12288) + (((int64_t)((int)bl` |
| compute.reduce#3 | 0.00 | 0.0% | 0.0 | 0.00 | `amax[0] = tl::AllReduce<tl::MaxOp, 8, 1, 0, tl::NamedBarrier<128>>::run(amax[0]);` |
| load.global#3 | 0.00 | 0.0% | 0.0 | 0.00 | `M = __fmul_rn(Sinks[((int)blockIdx.x)], 0x1.7154764ee6c2fp+0f/*1.442695e+00*/);` |
| load.global#4 | 0.00 | 0.0% | 0.0 | 0.00 | `if (((int)threadIdx.x) < 32)` |
| store.global#2 | 0.00 | 0.0% | 0.0 | 0.00 | `*(uint2*)(Output + (((((int64_t)((int)blockIdx.y)) * (int64_t)12288) + (((int64_t)((int)bl` |
| wait.sync#12 | 0.00 | 0.0% | 0.0 | 0.00 | `__syncthreads();` |
| wait.sync#13 | 0.00 | 0.0% | 0.0 | 0.00 | `__syncthreads();` |

### mhc_collapse_kernel grid (30, 1, 1) block (128, 1, 1)

3 launches, 120 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 5.22 us, warp 5.19 us, nsys graphs 3.87 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 5.04 | 97.2% | 4.21 | 95.6% |
| load | 0.10 | 1.8% | 0.13 | 2.9% |
| store | 0.05 | 1.0% | 0.06 | 1.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.10 | 1.8% | 1.0 | 0.10 | `*(float4*)(mv + 0) = *(float4*)(mix + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int6` |
| store.global | 0.05 | 1.0% | 1.0 | 0.06 | `*(uint4*)(out + ((((int64_t)((int)blockIdx.x)) * (int64_t)1024) + (((int64_t)((int)threadI` |

### local_stats_kernel grid (6, 16, 1) block (128, 1, 1)

1 launches, 384 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 13.18 us, warp 12.86 us, nsys graphs 12.26 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 11.86 | 92.2% | 12.13 | 92.0% |
| unattributed | 0.69 | 5.3% | 0.67 | 5.1% |
| compute | 0.19 | 1.5% | 0.19 | 1.5% |
| store | 0.08 | 0.6% | 0.13 | 1.0% |
| wait | 0.05 | 0.4% | 0.06 | 0.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 11.86 | 92.2% | 1.0 | 12.13 | `if (expanded_local_pos[((int64_t)((int)blockIdx.x))] < num_speculative_steps)` |
| compute.reduce | 0.13 | 1.0% | 1.0 | 0.16 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| store.global | 0.08 | 0.6% | 1.0 | 0.13 | `if (((int)threadIdx.x) == 0)` |
| compute.reduce#2 | 0.06 | 0.5% | 1.0 | 0.06 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 2, 32);` |
| wait.sync | 0.05 | 0.4% | 1.0 | 0.06 | `__syncthreads();` |

### moe_combine_kernel grid (5, 6, 1) block (128, 1, 1)

6 launches, 120 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 2.05 us, warp 1.55 us, nsys graphs 1.31 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.10 | 71.0% | 1.65 | 78.6% |
| unattributed | 0.39 | 25.3% | 0.39 | 18.8% |
| store | 0.06 | 3.6% | 0.05 | 2.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.10 | 71.0% | 1.0 | 1.58 | `for (int k = 0; k < 3; ++k) \| for (int k = 0; k < 6; ++k) \| if ((((int)blockIdx.y) < share` |
| store.global | 0.06 | 3.6% | 1.0 | 0.06 | `{` |

### padding_rows_kernel grid (1, 1, 1) block (256, 1, 1)

33 launches, 8 warps; ranges, warp and last warp from the 1 recording CTAs' 8 warps; span 0.35 us, warp 0.35 us, nsys graphs 0.77 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.26 | 72.7% | 0.26 | 69.9% |
| store | 0.10 | 27.3% | 0.11 | 30.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.10 | 27.3% | 1.0 | 0.10 | `if (((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) < rows)` |

### scatter_rows_kernel grid (1, 1, 1) block (128, 1, 1)

17 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 0.64 us, warp 0.59 us, nsys graphs 0.99 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.38 | 64.9% | 0.39 | 60.2% |
| load | 0.12 | 20.3% | 0.11 | 17.3% |
| store | 0.09 | 14.9% | 0.15 | 22.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.12 | 20.3% | 1.0 | 0.13 | `index = idx_mapping[min((((((int64_t)0 <= ((int64_t)width)) && ((int64_t)0 <= rmod_1)) \|\| ` |
| store.global | 0.09 | 14.9% | 1.0 | 0.16 | `if (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) < (num_reqs * width))` |

### ced_copy_rows_kernel grid (25, 1, 1) block (128, 1, 1)

16 launches, 100 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 0.67 us, warp 0.62 us, nsys graphs 1.12 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.37 | 60.9% | 0.40 | 61.0% |
| unattributed | 0.24 | 39.1% | 0.26 | 39.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.37 | 60.9% | 1.0 | 0.42 | `for (int k = 0; k < 4; ++k) \| int64_t index = ced_ldg64((&(row_indices[((((int64_t)blockId` |

### indexer_topk_segment_kernel grid (2, 6, 1) block (512, 1, 1)

1 launches, 192 warps; ranges, warp and last warp from the 4 recording CTAs' 64 warps; span 9.86 us, warp 5.20 us, nsys graphs 0.00 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 3.14 | 60.4% | 5.92 | 60.1% |
| unattributed, with loops' unsampled iterations | 1.68 | 32.4% | 3.30 | 33.4% |
| load | 0.33 | 6.4% | 0.54 | 5.5% |
| store | 0.05 | 0.9% | 0.10 | 1.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync#19*9 (est.: sampled iterations x9) | 0.35 | 6.8% | 0.5 | 0.86 | `__syncthreads();` |
| wait.sync#23*9 (est.: sampled iterations x9) | 0.35 | 6.8% | 0.5 | 0.86 | `__syncthreads();` |
| wait.sync#24*9 (est.: sampled iterations x9) | 0.34 | 6.5% | 0.5 | 0.86 | `__syncthreads();` |
| wait.sync#14*8 (est.: sampled iterations x8) | 0.33 | 6.3% | 0.5 | 0.77 | `__syncthreads();` |
| wait.sync#6*8 (est.: sampled iterations x8) | 0.32 | 6.1% | 0.5 | 0.77 | `__syncthreads();` |
| wait.sync#20*9 (est.: sampled iterations x9) | 0.31 | 5.9% | 0.5 | 0.86 | `__syncthreads();` |
| wait.sync#13*8 (est.: sampled iterations x8) | 0.29 | 5.5% | 0.5 | 0.77 | `__syncthreads();` |
| wait.sync#5*8 (est.: sampled iterations x8) | 0.28 | 5.3% | 0.5 | 0.77 | `__syncthreads();` |
| load.global | 0.20 | 3.8% | 1.0 | 0.22 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 4096); \| i` |
| load.global#2 | 0.13 | 2.5% | 0.5 | 0.32 | `for (int i = 0; i < 8; ++i)` |
| wait.sync (sampled iterations) | 0.05 | 1.0% | 0.5 | 0.19 | `__syncthreads();` |
| wait.sync#25 | 0.05 | 1.0% | 0.5 | 0.16 | `__syncthreads();` |
| store.global | 0.05 | 0.9% | 0.5 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3) \| if (((int)threadIdx.x) < total) \| if (((int)threadIdx.` |
| wait.sync#8 | 0.04 | 0.8% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#17 | 0.04 | 0.8% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#16 | 0.04 | 0.7% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#18 | 0.03 | 0.6% | 0.5 | 0.13 | `__syncthreads();` |
| wait.sync#9 | 0.03 | 0.6% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#12 | 0.03 | 0.6% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#3 | 0.03 | 0.6% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#11 | 0.03 | 0.6% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#15 | 0.03 | 0.6% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#2 | 0.03 | 0.6% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#21 | 0.03 | 0.6% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#4 | 0.03 | 0.6% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#7 | 0.03 | 0.6% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#10 | 0.03 | 0.6% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#22 | 0.02 | 0.5% | 0.5 | 0.06 | `__syncthreads();` |

### indexer_topk_merge_kernel grid (1, 6, 1) block (512, 1, 1)

1 launches, 96 warps; ranges, warp and last warp from the 6 recording CTAs' 96 warps; span 9.22 us, warp 9.00 us, nsys graphs 6.11 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 6.18 | 68.7% | 5.89 | 63.9% |
| unattributed | 2.15 | 23.9% | 2.40 | 26.0% |
| load | 0.59 | 6.6% | 0.80 | 8.7% |
| store | 0.08 | 0.9% | 0.13 | 1.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync#24*9 (est.: sampled iterations x9) | 0.73 | 8.1% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#23*9 (est.: sampled iterations x9) | 0.66 | 7.3% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#6*8 (est.: sampled iterations x8) | 0.65 | 7.2% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync#19*9 (est.: sampled iterations x9) | 0.62 | 6.9% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#20*9 (est.: sampled iterations x9) | 0.61 | 6.8% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#14*8 (est.: sampled iterations x8) | 0.59 | 6.6% | 1.0 | 0.77 | `__syncthreads();` |
| load.global | 0.58 | 6.5% | 1.0 | 0.70 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 8192);` |
| wait.sync#5*8 (est.: sampled iterations x8) | 0.58 | 6.4% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync#13*8 (est.: sampled iterations x8) | 0.57 | 6.3% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync | 0.15 | 1.7% | 1.0 | 0.22 | `__syncthreads();` |
| wait.sync#17 | 0.08 | 0.9% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#16 | 0.08 | 0.9% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#8 | 0.07 | 0.8% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#25 | 0.07 | 0.8% | 1.0 | 0.10 | `__syncthreads();` |
| store.global#2 | 0.07 | 0.8% | 0.9 | 0.10 | `Values[((((int64_t)((int)blockIdx.y)) * (int64_t)512) + ((int64_t)((int)threadIdx.x)))] = ` |
| wait.sync#3 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#7 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#2 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#9 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#21 | 0.06 | 0.7% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#22 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#12 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#18 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#10 | 0.06 | 0.6% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#15 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#11 | 0.05 | 0.6% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#4 | 0.05 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| store.global | 0.01 | 0.1% | 0.1 | 0.13 | `Values[((((int64_t)((int)blockIdx.y)) * (int64_t)512) + ((int64_t)((int)threadIdx.x)))] = ` |
| load.global#2 | 0.01 | 0.1% | 0.1 | 0.10 | `if (((0 <= ((int*)sel_idx)[((int)threadIdx.x)]) && (((int*)sel_idx)[((int)threadIdx.x)] < ` |

### token_ranks_kernel grid (6, 1, 1) block (256, 1, 1)

1 launches, 48 warps; ranges, warp and last warp from the 6 recording CTAs' 48 warps; span 7.74 us, warp 7.65 us, nsys graphs 7.94 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 6.91 | 90.3% | 7.14 | 92.1% |
| unattributed | 0.35 | 4.6% | 0.35 | 4.5% |
| compute | 0.34 | 4.5% | 0.19 | 2.5% |
| store | 0.05 | 0.6% | 0.06 | 0.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 6.91 | 90.3% | 1.0 | 7.14 | `if ((((int64_t)0 <= token_ids[((int64_t)((int)blockIdx.x))]) && (token_ids[((int64_t)((int` |
| compute.reduce | 0.34 | 4.5% | 1.0 | 0.61 | `total[0] = tl::AllReduce<tl::SumOp, 256, 1, 0, tl::NamedBarrier<256>>::run(total[0], (&(wo` |
| store.global | 0.05 | 0.6% | 1.0 | 0.06 | `output[((int64_t)((int)blockIdx.x))] = ((int64_t)total[0]);` |

### rope_kernel grid (6, 1, 1) block (192, 1, 1)

6 launches, 36 warps; ranges, warp and last warp from the 6 recording CTAs' 36 warps; span 1.26 us, warp 1.24 us, nsys graphs 1.50 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.70 | 57.0% | 0.76 | 60.9% |
| load | 0.49 | 40.0% | 0.45 | 36.2% |
| store | 0.04 | 3.0% | 0.04 | 3.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.49 | 40.0% | 1.0 | 0.51 | `if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < tokens)) \| int64_t position ` |
| store.global | 0.04 | 3.0% | 1.0 | 0.06 | `*(uint4*)(x + ((((((int64_t)((int)blockIdx.x)) * (int64_t)12288) + ((((int64_t)((int)threa` |

### indexer_mxfp4_quantize_kernel grid (6, 1, 1) block (128, 1, 1)

2 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 3.73 us, warp 3.65 us, nsys graphs 2.91 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 2.27 | 62.3% | 2.37 | 63.5% |
| store | 1.13 | 31.0% | 1.07 | 28.8% |
| unattributed | 0.24 | 6.6% | 0.29 | 7.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2*16 (est.: sampled iterations x16) | 1.25 | 34.2% | 1.0 | 1.54 | `float v__2 = __fmul_rn(((float)X[(((((int64_t)((int)blockIdx.x)) * (int64_t)4096) + (((int` |
| store.global*16 (est.: sampled iterations x16) | 1.09 | 29.8% | 1.0 | 1.54 | `Pool[(((((slot & (int64_t)127) * (int64_t)64) + ((((int64_t)((int)threadIdx.x)) & (int64_t` |
| load.global | 1.03 | 28.1% | 1.0 | 1.10 | `for (int i = 0; i < 32; ++i)` |
| store.global#2 | 0.04 | 1.2% | 1.0 | 0.06 | `Pool[(((((slot & (int64_t)127) * (int64_t)4) + ((slot >> (int64_t)7) * ((int64_t)pool_stri` |

### activation_swiglu_clamp_kernel grid (1, 6, 1) block (128, 1, 1)

6 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 0.93 us, warp 0.73 us, nsys graphs 1.82 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.63 | 85.8% | 0.77 | 84.3% |
| load | 0.07 | 9.9% | 0.09 | 9.3% |
| store | 0.03 | 4.4% | 0.06 | 6.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.07 | 9.9% | 0.8 | 0.10 | `*(uint4*)(gate + 0) = *(uint4*)(X + ((((int64_t)((int)blockIdx.y)) * (int64_t)1536) + (((i` |
| store.global | 0.03 | 4.4% | 0.8 | 0.06 | `*(uint4*)(Y + ((((int64_t)((int)blockIdx.y)) * (int64_t)768) + (((int64_t)((int)threadIdx.` |

### indexer_score_kernel grid (128, 6, 1) block (128, 1, 1)

1 launches, 3072 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 4.83 us, warp 2.17 us, nsys graphs 2.85 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.09 | 50.4% | 1.60 | 45.0% |
| unattributed, with loops' unsampled iterations | 0.81 | 37.3% | 1.50 | 42.3% |
| wait | 0.18 | 8.3% | 0.26 | 7.2% |
| compute | 0.06 | 2.9% | 0.13 | 3.6% |
| store | 0.02 | 1.0% | 0.06 | 1.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 (sampled iterations) | 0.60 | 27.5% | 0.5 | 1.41 | `for (int i_1 = 0; i_1 < 32; ++i_1)` |
| load.global | 0.39 | 18.2% | 1.0 | 0.77 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 4096); \| i` |
| load.global#3 (sampled iterations) | 0.10 | 4.8% | 0.5 | 0.42 | `if (((int)threadIdx.x) < 64)` |
| wait.sync#2 (sampled iterations) | 0.09 | 4.3% | 0.5 | 0.42 | `__syncthreads();` |
| compute.mma (sampled iterations) | 0.06 | 2.9% | 0.5 | 0.13 | `for (int ki = 0; ki < 2; ++ki)` |
| wait.sync#4 (sampled iterations) | 0.04 | 1.7% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync (sampled iterations) | 0.03 | 1.6% | 0.5 | 0.10 | `__syncthreads();` |
| store.global (sampled iterations) | 0.02 | 1.0% | 0.5 | 0.06 | `if ((((int)threadIdx.x) >> 6) == 0)` |
| wait.sync#3 (sampled iterations) | 0.02 | 0.7% | 0.5 | 0.03 | `__syncthreads();` |

### indexer_score_kernel grid (64, 6, 1) block (128, 1, 1)

1 launches, 1536 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 4.26 us, warp 2.46 us, nsys graphs 0.00 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.22 | 49.6% | 2.02 | 48.1% |
| unattributed, with loops' unsampled iterations | 0.89 | 36.3% | 1.47 | 35.1% |
| wait | 0.23 | 9.3% | 0.51 | 12.2% |
| compute | 0.07 | 2.8% | 0.16 | 3.8% |
| store | 0.05 | 1.9% | 0.03 | 0.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 (sampled iterations) | 0.59 | 24.1% | 0.5 | 1.28 | `for (int i_1 = 0; i_1 < 32; ++i_1)` |
| load.global | 0.56 | 22.7% | 1.0 | 0.80 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 4096); \| i` |
| wait.sync#2 (sampled iterations) | 0.09 | 3.8% | 0.5 | 0.29 | `__syncthreads();` |
| wait.sync#4 (sampled iterations) | 0.09 | 3.5% | 0.5 | 0.26 | `__syncthreads();` |
| compute.mma (sampled iterations) | 0.07 | 2.8% | 0.5 | 0.16 | `for (int ki = 0; ki < 2; ++ki)` |
| load.global#3 (sampled iterations) | 0.07 | 2.8% | 0.5 | 0.35 | `if (((int)threadIdx.x) < 64)` |
| store.global (sampled iterations) | 0.05 | 1.9% | 0.5 | 0.16 | `if ((((int)threadIdx.x) >> 6) == 0)` |
| wait.sync#3 (sampled iterations) | 0.03 | 1.1% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync (sampled iterations) | 0.02 | 1.0% | 0.5 | 0.06 | `__syncthreads();` |

### gemm_splitk_reduce_kernel grid (1, 6, 1) block (128, 1, 1)

4 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 0.77 us, warp 0.75 us, nsys graphs 0.99 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.39 | 51.8% | 0.38 | 50.0% |
| load | 0.32 | 42.7% | 0.33 | 42.7% |
| store | 0.04 | 5.6% | 0.06 | 7.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.32 | 42.7% | 1.0 | 0.34 | `acc[0] = P[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x))` |
| store.global | 0.04 | 5.6% | 1.0 | 0.06 | `C[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x)) * (int64` |

### metadata_pages_kernel grid (6, 1, 1) block (128, 1, 1)

2 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 1.33 us, warp 0.96 us, nsys graphs 0.00 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.74 | 76.8% | 0.82 | 61.4% |
| load | 0.14 | 14.9% | 0.19 | 14.5% |
| store | 0.08 | 8.3% | 0.32 | 24.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.13 | 13.3% | 1.0 | 0.13 | `if ((((offset + ((int)blockIdx.x)) < src_rows) && (0 <= (offset + ((int)blockIdx.x)))))` |
| store.global | 0.08 | 8.3% | 0.2 | 0.32 | `if (((int)threadIdx.x) < 32)` |
| load.global#2 | 0.02 | 1.7% | 0.2 | 0.06 | `if (((req < table_rows) && (((int)threadIdx.x) < table_cols)))` |

### norm_forward_and_per_token_cast_kernel_kernel grid (6, 1, 1) block (128, 1, 1)

1 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 2.56 us, warp 2.54 us, nsys graphs 1.76 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 2.33 | 91.7% | 1.98 | 77.5% |
| load | 0.47 | 18.4% | 0.64 | 25.0% |
| compute | 0.13 | 5.0% | 0.13 | 5.0% |
| wait | 0.06 | 2.5% | 0.03 | 1.2% |
| unattributed | 0.00 | 0.0% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*5 (est.: sampled iterations x5) | 2.04 | 80.2% | 1.0 | 2.24 | `for (int j_1 = 0; j_1 < 2; ++j_1)` |
| load.global#2*5 (est.: sampled iterations x5) | 0.25 | 9.7% | 1.0 | 0.32 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 1024) + (((int)threadIdx.x) * 8)` |
| store.global*5 (est.: sampled iterations x5) | 0.24 | 9.4% | 1.0 | 0.32 | `for (int j = 0; j < 4; ++j)` |
| load.global*5 (est.: sampled iterations x5) | 0.22 | 8.7% | 1.0 | 0.32 | `*(uint4*)(x_local + (vi * 8)) = *(uint4*)(x + ((((((int64_t)vi) * (int64_t)1024) + (((int6` |
| compute.reduce | 0.13 | 5.0% | 1.0 | 0.16 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| wait.sync | 0.06 | 2.5% | 1.0 | 0.10 | `__syncthreads();` |
| store.global#2 | 0.04 | 1.7% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| store.global#3 | 0.01 | 0.4% | 0.2 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |

### rope_kernel grid (6, 1, 1) block (256, 1, 1)

2 launches, 48 warps; ranges, warp and last warp from the 6 recording CTAs' 48 warps; span 1.02 us, warp 0.93 us, nsys graphs 1.12 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.63 | 68.2% | 0.72 | 70.3% |
| load | 0.26 | 27.8% | 0.26 | 25.0% |
| store | 0.04 | 4.1% | 0.05 | 4.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.26 | 27.8% | 1.0 | 0.27 | `if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < tokens)) \| int64_t position ` |
| store.global | 0.04 | 4.1% | 1.0 | 0.06 | `*(uint4*)(x + ((((((int64_t)((int)blockIdx.x)) * (int64_t)12288) + ((((int64_t)((int)threa` |

### mhc_collapse_column_kernel grid (30, 1, 1) block (128, 1, 1)

1 launches, 120 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 0.80 us, warp 0.76 us, nsys graphs 0.88 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.62 | 81.8% | 0.61 | 79.2% |
| load | 0.09 | 11.4% | 0.10 | 12.5% |
| store | 0.05 | 6.8% | 0.06 | 8.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.09 | 11.4% | 1.0 | 0.10 | `*(float4*)(mv + 0) = *(float4*)(mix + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int6` |
| store.global | 0.05 | 6.8% | 1.0 | 0.06 | `*(uint4*)(out + ((((int64_t)((int)blockIdx.x)) * (int64_t)1024) + (((int64_t)((int)threadI` |

### ced_copy_rows_kernel grid (30, 1, 1) block (128, 1, 1)

1 launches, 120 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 0.70 us, warp 0.63 us, nsys graphs 1.25 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.37 | 59.4% | 0.42 | 61.9% |
| unattributed | 0.26 | 40.6% | 0.26 | 38.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.37 | 59.4% | 1.0 | 0.42 | `for (int k = 0; k < 4; ++k) \| int64_t index = ced_ldg64((&(row_indices[((((int64_t)blockId` |
=== timeline and tables: exit 0 in 22 s
