=== timeline and tables
trace /c/iket/step-tp4-c1-20261026T161848Z/trace/iket/pid_0x188b/iket.decoded_results.json: 12688 launches, graphLaunches dict of 124

## Timeline: 7754 launches in 867884.4 us; the GPU is busy 263013.2 us (30%) and idle 604871.2 us in 5104 gaps; gaps over 200 us split it into 523 segments

The 15 of 523 segments with the most busy time:

| segment | launches | wall us | busy us | busy | first kernel |
|---:|---:|---:|---:|---:|---|
| 10 | 152 | 8987.7 | 7098.9 | 79% | sink_rows_kernel |
| 45 | 116 | 6599.4 | 6522.8 | 99% | mhc_project_streams_kernel |
| 103 | 116 | 6544.4 | 6468.6 | 99% | mhc_project_streams_kernel |
| 173 | 117 | 6687.2 | 6466.3 | 97% | padding_rows_kernel |
| 190 | 116 | 6565.3 | 6478.3 | 99% | mhc_project_streams_kernel |
| 224 | 120 | 6957.7 | 6468.5 | 93% | metadata_builder_kernel |
| 252 | 116 | 6555.4 | 6463.3 | 99% | mhc_project_streams_kernel |
| 279 | 116 | 6609.3 | 6515.3 | 99% | mhc_project_streams_kernel |
| 307 | 117 | 6610.5 | 6495.7 | 98% | padding_rows_kernel |
| 321 | 116 | 6562.5 | 6476.4 | 99% | mhc_project_streams_kernel |
| 361 | 116 | 6539.8 | 6453.2 | 99% | mhc_project_streams_kernel |
| 398 | 116 | 6768.7 | 6679.6 | 99% | mhc_project_streams_kernel |
| 430 | 116 | 6877.2 | 6775.6 | 99% | mhc_project_streams_kernel |
| 476 | 116 | 6544.5 | 6456.6 | 99% | mhc_project_streams_kernel |
| 495 | 120 | 6951.9 | 6508.1 | 94% | metadata_builder_kernel |

| idle us | after | before |
|---:|---|---|
| 98190.1 | resample_kernel | insert_kernel |
| 93041.5 | scatter_rows_kernel | padding_rows_kernel |
| 3432.9 | dspark_confidence_kernel | scatter_rows_gather_kernel |
| 3420.3 | dspark_confidence_kernel | scatter_rows_gather_kernel |
| 3340.7 | norm_forward_and_per_token_cast_kernel_kernel | sink_flags_kernel |
| 3323.0 | norm_forward_and_per_token_cast_kernel_kernel | sink_flags_kernel |
| 3311.4 | norm_forward_and_per_token_cast_kernel_kernel | sink_flags_kernel |
| 3216.3 | dspark_confidence_kernel | scatter_rows_gather_kernel |
| 3173.7 | dspark_confidence_kernel | scatter_rows_gather_kernel |
| 3078.7 | dspark_confidence_kernel | scatter_rows_gather_kernel |

## Engine steps

| step | kind | launches | wall us | busy us | busy | most span |
|---:|---|---:|---:|---:|---:|---|
| 0 | prefill | 413 | 40579.2 | 16884.2 | 42% | vocab_packed_projection_kernel 8748, gemm_mxfp8_kernel 2561, moe_grouped_fp8_fp4_gemm_kernel 2098 |
| 1 | decode | 432 | 134168.8 | 14617.5 | 11% | vocab_packed_projection_kernel 8155, gemm_mxfp8_kernel 1754, l2_prefetch_kernel 1729 |
| 2 | decode | 432 | 129619.9 | 14577.7 | 11% | vocab_packed_projection_kernel 8210, l2_prefetch_kernel 1948, gemm_mxfp8_kernel 1656 |
| 3 | decode | 432 | 33852.1 | 14483.6 | 43% | vocab_packed_projection_kernel 8107, l2_prefetch_kernel 2106, gemm_mxfp8_kernel 1790 |
| 4 | decode | 432 | 40888.8 | 14490.0 | 35% | vocab_packed_projection_kernel 8033, l2_prefetch_kernel 1560, gemm_mxfp8_kernel 1489 |
| 5 | decode | 432 | 37124.8 | 14436.8 | 39% | vocab_packed_projection_kernel 8055, gemm_mxfp8_kernel 1541, l2_prefetch_kernel 1523 |
| 6 | decode | 432 | 30518.2 | 14390.8 | 47% | vocab_packed_projection_kernel 8048, gemm_mxfp8_kernel 1525, l2_prefetch_kernel 1524 |
| 7 | decode | 432 | 35322.8 | 14360.5 | 41% | vocab_packed_projection_kernel 8031, l2_prefetch_kernel 1517, gemm_mxfp8_kernel 1515 |
| 8 | decode | 432 | 33411.9 | 14419.4 | 43% | vocab_packed_projection_kernel 8030, l2_prefetch_kernel 1515, gemm_mxfp8_kernel 1507 |
| 9 | decode | 432 | 31474.0 | 14402.9 | 46% | vocab_packed_projection_kernel 8065, l2_prefetch_kernel 1519, gemm_mxfp8_kernel 1486 |
| 10 | decode | 432 | 40879.5 | 14395.5 | 35% | vocab_packed_projection_kernel 8056, l2_prefetch_kernel 1521, gemm_mxfp8_kernel 1492 |
| 11 | decode | 432 | 28909.4 | 14378.9 | 50% | vocab_packed_projection_kernel 8029, gemm_mxfp8_kernel 1511, l2_prefetch_kernel 1510 |
| 12 | decode | 432 | 49395.3 | 14402.7 | 29% | vocab_packed_projection_kernel 8055, l2_prefetch_kernel 1515, gemm_mxfp8_kernel 1491 |
| 13 | decode | 432 | 45890.4 | 14847.0 | 32% | vocab_packed_projection_kernel 8209, gemm_mxfp8_kernel 1622, l2_prefetch_kernel 1548 |
| 14 | decode | 432 | 40234.3 | 14788.4 | 37% | vocab_packed_projection_kernel 8147, gemm_mxfp8_kernel 1618, l2_prefetch_kernel 1542 |
| 15 | decode | 432 | 51638.8 | 14389.2 | 28% | vocab_packed_projection_kernel 8050, gemm_mxfp8_kernel 1545, l2_prefetch_kernel 1518 |
| 16 | decode | 432 | 28083.6 | 14420.5 | 51% | vocab_packed_projection_kernel 8039, l2_prefetch_kernel 1530, gemm_mxfp8_kernel 1484 |
| 17 | decode | 429 | 28703.2 | 14327.7 | 50% | vocab_packed_projection_kernel 8018, l2_prefetch_kernel 1550, gemm_mxfp8_kernel 1507 |

## Overview over 17 decode steps: 7341 launches of 142 kernel shapes, spans summing to 273363.4 us (16080.2 a step)

| kernel | grid | launches a step | span us a step | share | span us a launch | last warp: wait | load | compute | store | unattributed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vocab_packed_projection_kernel | (1, 4040, 1) | 2 | 8078.7 | 50.2% | 4021.65 | 56% | 31% | 10% | 0% | 2% |
| l2_prefetch_kernel | (2, 1, 1) | 25 | 1598.5 | 9.9% | 63.49 | 0% | 100% | 0% | 0% | 0% |
| gemm_bf16_kernel | (2020, 1, 1) | 5 | 1314.9 | 8.2% | 262.34 | 89% | 0% | 5% | 1% | 5% |
| moe_grouped_fp8_fp4_gemm_kernel | (9, 30, 1) | 8.64706 | 805.8 | 5.0% | 117.25 | 54% | 21% | 30% | 1% | 0% |
| moe_grouped_fp8_fp4_gemm_kernel | (40, 30, 1) | 8.64706 | 487.0 | 3.0% | 65.66 | 14% | 20% | 0% | 0% | 66% |
| gemm_mxfp8_kernel | (80, 1, 1) | 1 | 394.4 | 2.5% | 392.99 | 77% | 4% | 5% | 0% | 14% |
| gemm_mxfp8_kernel | (64, 1, 1) | 11 | 359.6 | 2.2% | 25.12 | 27% | 11% | 15% | 1% | 46% |
| gemm_mxfp8_kernel | (28, 1, 1) | 9 | 329.9 | 2.1% | 37.18 | 36% | 11% | 11% | 0% | 43% |
| mhc_project_streams_kernel | (40, 1, 1) | 18 | 299.6 | 1.9% | 16.45 | 19% | 32% | 17% | 3% | 28% |
| gemm_fp8w_bf16_partials_kernel | (32, 4, 1) | 9 | 209.5 | 1.3% | 17.95 | 50% | 19% | 11% | 0% | 20% |
| gemm_mxfp8_kernel | (36, 1, 1) | 9 | 203.3 | 1.3% | 22.14 | 39% | 16% | 16% | 0% | 28% |
| gemm_mxfp8_kernel | (40, 1, 1) | 9 | 188.6 | 1.2% | 19.71 | 45% | 16% | 16% | 0% | 23% |
| gemm_bf16_partials_kernel | (6, 8, 1) | 6 | 162.7 | 1.0% | 26.43 | 33% | 0% | 33% | 2% | 32% |
| mhc_finalize_kernel | (5, 1, 1) | 11.2941 | 144.5 | 0.9% | 12.38 | 6% | 14% | 9% | 18% | 54% |
| gemm_bf16_splitk_kernel | (16, 5, 1) | 2 | 112.4 | 0.7% | 56.27 | 99% | 8% | 4% | 0% | 0% |
| gemm_mxfp8_kernel | (40, 1, 1) | 9 | 84.8 | 0.5% | 6.91 | 51% | 13% | 18% | 1% | 18% |
| sparse_mla_kernel | (5, 1, 5) | 3.76471 | 80.0 | 0.5% | 14.70 | 3% | 41% | 5% | 2% | 49% |
| sparse_mla_kernel | (2, 1, 10) | 3 | 71.0 | 0.4% | 24.61 | 12% | 28% | 20% | 2% | 37% |
| mhc_finalize_kernel | (10, 1, 1) | 6 | 68.2 | 0.4% | 12.08 | 5% | 11% | 11% | 24% | 48% |
| sparse_mla_kernel | (1, 1, 5) | 1.88235 | 55.4 | 0.3% | 30.93 | 9% | 36% | 16% | 2% | 36% |

| wait | kernel | grid | last-warp us a step | a launch | first statement |
|---|---|---|---:|---:|---|
| wait.cp_async*118 | gemm_mxfp8_kernel | (80, 1, 1) | 204.6 | 204.57 | `tl::cp_async_wait<2>();` |
| wait.mbarrier*40 | moe_grouped_fp8_fp4_gemm_kernel | (9, 30, 1) | 100.7 | 11.65 | `loaded[(ko % 3)].wait(((ko % 6) / 3));` |
| wait.sync#2*118 | gemm_mxfp8_kernel | (80, 1, 1) | 88.8 | 88.85 | `__syncthreads();` |
| wait.cp_async*13 | gemm_bf16_splitk_kernel | (16, 5, 1) | 62.4 | 31.20 | `tl::cp_async_wait<2>();` |
| wait.cp_async*19 | gemm_mxfp8_kernel | (28, 1, 1) | 59.6 | 6.62 | `tl::cp_async_wait<1>();` |
| wait.sync*4 | mhc_project_streams_kernel | (40, 1, 1) | 51.1 | 2.84 | `__syncthreads();` |
| wait.cp_async*40 | moe_grouped_fp8_fp4_gemm_kernel | (9, 30, 1) | 50.8 | 5.88 | `tl::cp_async_wait<2>();` |
| wait.sync#2*19 | gemm_mxfp8_kernel | (28, 1, 1) | 41.0 | 4.55 | `__syncthreads();` |
| wait.cp_async*14 | gemm_mxfp8_kernel | (40, 1, 1) | 34.9 | 3.88 | `tl::cp_async_wait<2>();` |
| wait.cp_async*14 | gemm_fp8w_bf16_partials_kernel | (32, 4, 1) | 32.0 | 3.55 | `tl::cp_async_wait<3>();` |
| wait.sync#2*18 | gemm_mxfp8_kernel | (36, 1, 1) | 31.6 | 3.51 | `__syncthreads();` |
| wait.cp_async*18 | gemm_mxfp8_kernel | (36, 1, 1) | 30.7 | 3.41 | `tl::cp_async_wait<2>();` |
| wait.sync#2*14 | gemm_mxfp8_kernel | (40, 1, 1) | 29.4 | 3.27 | `__syncthreads();` |
| wait.mbarrier#2 | gemm_bf16_kernel | (2020, 1, 1) | 26.8 | 5.36 | `mbarrier[0].wait(0);` |
| wait.sync#2*17 | vocab_packed_projection_kernel | (1, 4040, 1) | 24.8 | 12.40 | `__syncthreads();` |
| wait.sync#5*40 | moe_grouped_fp8_fp4_gemm_kernel | (9, 30, 1) | 21.0 | 2.43 | `__syncthreads();` |
| wait.cp_async*17 | vocab_packed_projection_kernel | (1, 4040, 1) | 20.8 | 10.38 | `tl::cp_async_wait<6>();` |
| wait.sync#6*40 | moe_grouped_fp8_fp4_gemm_kernel | (9, 30, 1) | 20.6 | 2.39 | `__syncthreads();` |
| wait.sync#4*40 | moe_grouped_fp8_fp4_gemm_kernel | (9, 30, 1) | 20.4 | 2.36 | `__syncthreads();` |
| wait.sync#2*14 | gemm_fp8w_bf16_partials_kernel | (32, 4, 1) | 19.5 | 2.16 | `__syncthreads();` |

### vocab_packed_projection_kernel grid (1, 4040, 1) block (128, 1, 1)

34 launches, 16160 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 4021.65 us, warp 49.76 us, nsys graphs 4016.85 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 30.20 | 60.7% | 27.98 | 56.4% |
| load | 10.82 | 21.7% | 15.57 | 31.4% |
| compute | 5.05 | 10.2% | 5.00 | 10.1% |
| unattributed | 3.63 | 7.3% | 1.02 | 2.1% |
| store | 0.05 | 0.1% | 0.05 | 0.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.cp_async*17 (est.: sampled iterations x17) | 14.60 | 29.3% | 1.0 | 38.90 | `tl::cp_async_wait<6>();` |
| wait.sync#2*17 (est.: sampled iterations x17) | 10.25 | 20.6% | 1.0 | 32.10 | `__syncthreads();` |
| load.global | 8.33 | 16.7% | 1.0 | 14.78 | `int start = row_offsets[(((int)blockIdx.y) * 16)]; \| int start = row_offsets[(((int)blockI` |
| compute.mma*17 (est.: sampled iterations x17) | 4.35 | 8.7% | 1.0 | 4.35 | `{` |
| load.cp_async#2*17 (est.: sampled iterations x17) | 2.21 | 4.4% | 1.0 | 2.72 | `for (int i_6 = 0; i_6 < 8; ++i_6) \| for (int i_9 = 0; i_9 < 4; ++i_9)` |
| wait.sync*17 (est.: sampled iterations x17) | 1.07 | 2.2% | 1.0 | 1.09 | `__syncthreads();` |
| wait.sync#3*17 (est.: sampled iterations x17) | 1.05 | 2.1% | 1.0 | 1.09 | `__syncthreads();` |
| wait.cp_async#2*17 (est.: sampled iterations x17) | 0.95 | 1.9% | 1.0 | 1.09 | `tl::cp_async_wait<7>();` |
| wait.sync#4 | 0.46 | 0.9% | 1.0 | 1.30 | `__syncthreads();` |
| wait.cp_async#3 | 0.45 | 0.9% | 1.0 | 1.89 | `tl::cp_async_wait<4>();` |
| wait.sync#6 | 0.39 | 0.8% | 1.0 | 1.04 | `__syncthreads();` |
| load.cp_async | 0.28 | 0.6% | 1.0 | 0.32 | `for (int i = 0; i < 4; ++i) \| for (int i = 0; i < 8; ++i) \| for (int i_1 = 0; i_1 < 4; ++i` |
| wait.sync#8 | 0.27 | 0.5% | 1.0 | 1.07 | `__syncthreads();` |
| compute.mma#2 | 0.24 | 0.5% | 1.0 | 0.26 | `{` |
| compute.mma#4 | 0.24 | 0.5% | 1.0 | 0.26 | `{` |
| compute.mma#3 | 0.23 | 0.5% | 1.0 | 0.26 | `{` |
| wait.cp_async#5 | 0.23 | 0.5% | 1.0 | 1.22 | `tl::cp_async_wait<2>();` |
| wait.cp_async#7 | 0.19 | 0.4% | 1.0 | 1.14 | `tl::cp_async_wait<0>();` |
| wait.sync#5 | 0.06 | 0.1% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#7 | 0.05 | 0.1% | 1.0 | 0.06 | `__syncthreads();` |
| store.global | 0.05 | 0.1% | 1.0 | 0.06 | `for (int i_13 = 0; i_13 < 2; ++i_13) \| for (int i_14 = 0; i_14 < 4; ++i_14) \| for (int i_9` |
| wait.sync#9 | 0.05 | 0.1% | 1.0 | 0.06 | `__syncthreads();` |
| wait.cp_async#4 | 0.04 | 0.1% | 1.0 | 0.06 | `tl::cp_async_wait<5>();` |
| wait.cp_async#6 | 0.04 | 0.1% | 1.0 | 0.06 | `tl::cp_async_wait<3>();` |
| wait.cp_async#8 | 0.04 | 0.1% | 1.0 | 0.06 | `tl::cp_async_wait<1>();` |

### l2_prefetch_kernel grid (2, 1, 1) block (128, 1, 1)

425 launches, 8 warps; ranges, warp and last warp from the 2 recording CTAs' 8 warps; span 63.49 us, warp 62.09 us, nsys graphs 64.77 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 61.68 | 99.3% | 63.66 | 99.6% |
| unattributed | 0.41 | 0.7% | 0.28 | 0.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 61.68 | 99.3% | 1.0 | 63.23 | `for (int s = 0; s < (nseg2 >> 1); ++s)` |

### gemm_bf16_kernel grid (2020, 1, 1) block (256, 1, 1)

85 launches, 16160 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 262.34 us, warp 3.61 us, nsys graphs 262.05 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 3.05 | 84.5% | 5.45 | 88.9% |
| unattributed | 0.34 | 9.5% | 0.32 | 5.2% |
| compute | 0.14 | 4.0% | 0.29 | 4.7% |
| load | 0.04 | 1.0% | 0.00 | 0.0% |
| store | 0.04 | 1.0% | 0.07 | 1.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.mbarrier#2 | 2.94 | 81.5% | 0.5 | 7.10 | `mbarrier[0].wait(0);` |
| compute.mma | 0.14 | 4.0% | 0.5 | 0.29 | `{` |
| wait.sync | 0.08 | 2.1% | 1.0 | 0.10 | `__syncthreads();` |
| load.tma | 0.04 | 1.0% | 0.5 | 0.13 | `if (tl::tl_shuffle_elect<128>())` |
| store.global | 0.04 | 1.0% | 0.5 | 0.10 | `for (int i_3 = 0; i_3 < 4; ++i_3) \| for (int i_4 = 0; i_4 < 8; ++i_4) \| for (int i_6 = 0; ` |
| wait.mbarrier | 0.03 | 0.8% | 0.5 | 0.06 | `mbarrier[2].wait(1);` |

### moe_grouped_fp8_fp4_gemm_kernel grid (9, 30, 1) block (128, 1, 1)

147 launches, 1080 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 117.25 us, warp 23.41 us, nsys graphs 47.39 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 7.98 | 34.1% | 24.85 | 53.9% |
| unattributed | 7.23 | 30.9% | 0.00 | 0.0% |
| compute | 5.38 | 23.0% | 13.63 | 29.6% |
| load | 2.71 | 11.6% | 9.66 | 20.9% |
| store | 0.12 | 0.5% | 0.36 | 0.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*40 (est.: sampled iterations x40) | 5.38 | 23.0% | 0.4 | 14.08 | `{` |
| wait.mbarrier*40 (est.: sampled iterations x40) | 3.14 | 13.4% | 0.4 | 12.80 | `loaded[(ko % 3)].wait(((ko % 6) / 3));` |
| load#2*40 (est.: sampled iterations x40) | 2.30 | 9.8% | 0.4 | 8.96 | `if (ko < 38)` |
| wait.sync#5*40 (est.: sampled iterations x40) | 1.92 | 8.2% | 0.4 | 6.40 | `__syncthreads();` |
| wait.cp_async*40 (est.: sampled iterations x40) | 0.96 | 4.1% | 0.4 | 2.56 | `tl::cp_async_wait<2>();` |
| wait.sync#4*40 (est.: sampled iterations x40) | 0.96 | 4.1% | 0.4 | 2.56 | `__syncthreads();` |
| wait.sync#6*40 (est.: sampled iterations x40) | 0.90 | 3.8% | 0.4 | 2.56 | `__syncthreads();` |
| load.global#3 | 0.20 | 0.8% | 0.4 | 0.58 | `if (((expert < 128) && (0 <= expert))) \| if (((expert < 384) && (0 <= expert)))` |
| store.global | 0.12 | 0.5% | 0.4 | 0.35 | `for (int i_2 = 0; i_2 < 16; ++i_2) \| for (int i_3 = 0; i_3 < 16; ++i_3) \| for (int i_4 = 0` |
| load.global | 0.10 | 0.4% | 1.0 | 0.29 | `int expert = TileExpert[((int64_t)((int)blockIdx.y))];` |
| load | 0.07 | 0.3% | 0.4 | 0.26 | `for (int s = 0; s < 2; ++s)` |
| wait.sync | 0.05 | 0.2% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#2 | 0.05 | 0.2% | 0.4 | 0.16 | `__syncthreads();` |
| load.global#2 | 0.04 | 0.2% | 0.4 | 0.19 | `if (((int)threadIdx.x) < 32) \| if (((int)threadIdx.x) < 64)` |
| wait.sync#3 | 0.01 | 0.1% | 0.4 | 0.06 | `__syncthreads();` |

### moe_grouped_fp8_fp4_gemm_kernel grid (40, 30, 1) block (128, 1, 1)

147 launches, 4800 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 65.66 us, warp 4.70 us, nsys graphs 29.14 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.86 | 39.5% | 0.20 | 66.2% |
| load | 0.97 | 20.7% | 0.06 | 19.8% |
| compute | 0.88 | 18.7% | 0.00 | 0.0% |
| wait | 0.84 | 18.0% | 0.04 | 14.0% |
| store | 0.14 | 3.1% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*5 (est.: sampled iterations x5) | 0.88 | 18.7% | 0.5 | 1.76 | `if (ko < 4)` |
| load#2*5 (est.: sampled iterations x5) | 0.44 | 9.4% | 0.5 | 0.96 | `if (ko < 3)` |
| load.global#3 | 0.31 | 6.6% | 0.5 | 0.80 | `if (((expert < 128) && (0 <= expert))) \| if (((expert < 384) && (0 <= expert)))` |
| wait.sync#5*5 (est.: sampled iterations x5) | 0.17 | 3.6% | 0.5 | 0.48 | `__syncthreads();` |
| wait.mbarrier*5 (est.: sampled iterations x5) | 0.16 | 3.4% | 0.5 | 0.48 | `loaded[(ko % 3)].wait((ko / 3));` |
| store.global | 0.14 | 3.1% | 0.5 | 0.38 | `for (int i_2 = 0; i_2 < 16; ++i_2) \| for (int i_3 = 0; i_3 < 16; ++i_3) \| for (int i_4 = 0` |
| wait.sync#4*5 (est.: sampled iterations x5) | 0.14 | 3.0% | 0.5 | 0.32 | `__syncthreads();` |
| wait.sync#6*5 (est.: sampled iterations x5) | 0.13 | 2.8% | 0.5 | 0.32 | `__syncthreads();` |
| wait.cp_async*5 (est.: sampled iterations x5) | 0.12 | 2.6% | 0.5 | 0.32 | `tl::cp_async_wait<2>();` |
| load.global | 0.10 | 2.0% | 1.0 | 0.19 | `int expert = TileExpert[((int64_t)((int)blockIdx.y))];` |
| load | 0.08 | 1.8% | 0.5 | 0.26 | `for (int s = 0; s < 2; ++s)` |
| wait.sync#2 | 0.06 | 1.4% | 0.5 | 0.19 | `__syncthreads();` |
| wait.sync | 0.04 | 0.9% | 1.0 | 0.06 | `__syncthreads();` |
| load.global#2 | 0.04 | 0.9% | 0.5 | 0.22 | `if (((int)threadIdx.x) < 32) \| if (((int)threadIdx.x) < 64)` |
| wait.sync#3 | 0.02 | 0.3% | 0.5 | 0.06 | `__syncthreads();` |

### gemm_mxfp8_kernel grid (80, 1, 1) block (128, 1, 1)

17 launches, 320 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 392.99 us, warp 391.04 us, nsys graphs 371.86 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 318.15 | 81.4% | 301.15 | 76.5% |
| unattributed | 41.13 | 10.5% | 56.78 | 14.4% |
| compute | 18.42 | 4.7% | 18.91 | 4.8% |
| load | 13.29 | 3.4% | 16.74 | 4.3% |
| store | 0.06 | 0.0% | 0.05 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.cp_async*118 (est.: sampled iterations x118) | 203.43 | 52.0% | 1.0 | 577.73 | `tl::cp_async_wait<2>();` |
| wait.sync#2*118 (est.: sampled iterations x118) | 105.26 | 26.9% | 1.0 | 309.63 | `__syncthreads();` |
| compute.mma*118 (est.: sampled iterations x118) | 18.17 | 4.6% | 1.0 | 18.88 | `{` |
| load.cp_async#2*118 (est.: sampled iterations x118) | 12.98 | 3.3% | 1.0 | 15.10 | `for (int i_5 = 0; i_5 < 2; ++i_5) \| for (int i_5 = 0; i_5 < 4; ++i_5) \| tl::cp_async_gs_co` |
| wait.sync*118 (est.: sampled iterations x118) | 7.55 | 1.9% | 1.0 | 7.55 | `__syncthreads();` |
| wait.cp_async#2 | 0.69 | 0.2% | 1.0 | 2.21 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| wait.cp_async#3 | 0.51 | 0.1% | 1.0 | 2.14 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>();` |
| wait.sync#3 | 0.37 | 0.1% | 1.0 | 1.28 | `__syncthreads();` |
| wait.sync#4 | 0.34 | 0.1% | 1.0 | 1.28 | `__syncthreads();` |
| load.cp_async | 0.31 | 0.1% | 1.0 | 0.51 | `for (int i = 0; i < 2; ++i) \| for (int i_1 = 0; i_1 < 2; ++i_1) \| for (int i_1 = 0; i_1 < ` |
| compute.mma#3 | 0.13 | 0.0% | 1.0 | 0.16 | `{` |
| compute.mma#2 | 0.12 | 0.0% | 1.0 | 0.13 | `{` |
| store.global | 0.06 | 0.0% | 1.0 | 0.06 | `for (int i_10 = 0; i_10 < 16; ++i_10) \| for (int i_10 = 0; i_10 < 8; ++i_10) \| for (int i_` |

### gemm_mxfp8_kernel grid (64, 1, 1) block (128, 1, 1)

187 launches, 256 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 25.12 us, warp 14.72 us, nsys graphs 24.70 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 7.21 | 48.9% | 5.42 | 45.7% |
| wait | 4.38 | 29.8% | 3.19 | 26.9% |
| compute | 2.03 | 13.8% | 1.83 | 15.5% |
| load | 1.04 | 7.1% | 1.36 | 11.4% |
| store | 0.06 | 0.4% | 0.06 | 0.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*8 (est.: sampled iterations x8) | 1.66 | 11.3% | 1.0 | 1.79 | `{` |
| wait.sync#2*8 (est.: sampled iterations x8) | 1.14 | 7.7% | 1.0 | 4.10 | `__syncthreads();` |
| wait.cp_async*8 (est.: sampled iterations x8) | 0.99 | 6.7% | 1.0 | 4.86 | `tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| load.cp_async#2*8 (est.: sampled iterations x8) | 0.82 | 5.5% | 1.0 | 1.02 | `for (int i_3 = 0; i_3 < 2; ++i_3) \| for (int i_5 = 0; i_5 < 2; ++i_5) \| tl::cp_async_gs_co` |
| wait.cp_async#3 | 0.55 | 3.7% | 1.0 | 1.82 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>();` |
| wait.cp_async#2 | 0.51 | 3.4% | 1.0 | 1.66 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| wait.sync#4 | 0.46 | 3.1% | 1.0 | 1.50 | `__syncthreads();` |
| wait.sync#3 | 0.37 | 2.5% | 1.0 | 1.25 | `__syncthreads();` |
| wait.sync*8 (est.: sampled iterations x8) | 0.37 | 2.5% | 1.0 | 0.51 | `__syncthreads();` |
| load.cp_async | 0.23 | 1.5% | 1.0 | 0.26 | `for (int i = 0; i < 2; ++i) \| for (int i_1 = 0; i_1 < 2; ++i_1) \| for (int i_1 = 0; i_1 < ` |
| compute.mma#3 | 0.19 | 1.3% | 1.0 | 0.19 | `{` |
| compute.mma#2 | 0.18 | 1.2% | 1.0 | 0.19 | `{` |
| store.global | 0.06 | 0.4% | 1.0 | 0.06 | `for (int i_10 = 0; i_10 < 16; ++i_10) \| for (int i_10 = 0; i_10 < 8; ++i_10) \| for (int i_` |
| compute.mma#4 | 0.00 | 0.0% | 0.0 | 0.00 | `{` |
| compute.mma*2 (est.: sampled iterations x2) | 0.00 | 0.0% | 0.0 | 0.00 | `{` |
| load.cp_async#2*2 (est.: sampled iterations x2) | 0.00 | 0.0% | 0.0 | 0.00 | `for (int i_7 = 0; i_7 < 2; ++i_7)` |
| wait.cp_async#4 | 0.00 | 0.0% | 0.0 | 0.00 | `tl::cp_async_wait<0>();` |
| wait.cp_async*2 (est.: sampled iterations x2) | 0.00 | 0.0% | 0.0 | 0.00 | `tl::cp_async_wait<3>();` |
| wait.sync#2*2 (est.: sampled iterations x2) | 0.00 | 0.0% | 0.0 | 0.00 | `__syncthreads();` |
| wait.sync#5 | 0.00 | 0.0% | 0.0 | 0.00 | `__syncthreads();` |
| wait.sync*2 (est.: sampled iterations x2) | 0.00 | 0.0% | 0.0 | 0.00 | `__syncthreads();` |

### gemm_mxfp8_kernel grid (28, 1, 1) block (128, 1, 1)

153 launches, 112 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 37.18 us, warp 35.02 us, nsys graphs 41.66 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 21.02 | 60.0% | 15.29 | 42.5% |
| wait | 7.08 | 20.2% | 12.82 | 35.6% |
| compute | 3.94 | 11.2% | 3.91 | 10.9% |
| load | 2.93 | 8.4% | 3.90 | 10.8% |
| store | 0.05 | 0.1% | 0.05 | 0.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*19 (est.: sampled iterations x19) | 3.76 | 10.7% | 1.0 | 4.26 | `{` |
| wait.sync#2*19 (est.: sampled iterations x19) | 3.08 | 8.8% | 1.0 | 9.73 | `__syncthreads();` |
| load.cp_async#2*19 (est.: sampled iterations x19) | 2.66 | 7.6% | 1.0 | 3.04 | `for (int i_3 = 0; i_3 < 2; ++i_3) \| for (int i_3 = 0; i_3 < 8; ++i_3)` |
| wait.cp_async*19 (est.: sampled iterations x19) | 2.58 | 7.4% | 1.0 | 13.98 | `tl::cp_async_wait<1>();` |
| wait.sync*19 (est.: sampled iterations x19) | 1.18 | 3.4% | 1.0 | 1.22 | `__syncthreads();` |
| load.cp_async | 0.27 | 0.8% | 1.0 | 0.29 | `for (int i = 0; i < 2; ++i) \| for (int i_1 = 0; i_1 < 2; ++i_1) \| for (int i_1 = 0; i_1 < ` |
| compute.mma#2 | 0.18 | 0.5% | 1.0 | 0.19 | `{` |
| wait.cp_async#2 | 0.13 | 0.4% | 1.0 | 0.32 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| wait.sync#3 | 0.10 | 0.3% | 1.0 | 0.26 | `__syncthreads();` |
| store.global | 0.05 | 0.1% | 1.0 | 0.06 | `for (int i_10 = 0; i_10 < 16; ++i_10) \| for (int i_10 = 0; i_10 < 8; ++i_10) \| for (int i_` |

### mhc_project_streams_kernel grid (40, 1, 1) block (128, 1, 1)

306 launches, 160 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 16.45 us, warp 14.24 us, nsys graphs 14.43 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 6.97 | 48.9% | 4.42 | 28.2% |
| compute | 2.79 | 19.6% | 2.61 | 16.6% |
| load | 2.75 | 19.3% | 5.07 | 32.4% |
| wait | 1.27 | 8.9% | 3.03 | 19.3% |
| store | 0.46 | 3.3% | 0.55 | 3.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*4 (est.: sampled iterations x4) | 2.66 | 18.7% | 1.0 | 2.69 | `{` |
| load.global#2*4 (est.: sampled iterations x4) | 2.52 | 17.7% | 1.0 | 5.82 | `for (int i_1 = 0; i_1 < 6; ++i_1)` |
| wait.sync*4 (est.: sampled iterations x4) | 1.08 | 7.6% | 1.0 | 3.71 | `__syncthreads();` |
| store.global*4 (est.: sampled iterations x4) | 0.25 | 1.7% | 1.0 | 0.38 | `if (((((int)blockIdx.y) * 16) + (((int)threadIdx.x) >> 3)) < M)` |
| load.global | 0.23 | 1.6% | 1.0 | 0.26 | `for (int k = 0; k < 4; ++k)` |
| wait.sync#2*4 (est.: sampled iterations x4) | 0.18 | 1.3% | 1.0 | 0.26 | `__syncthreads();` |
| store.global#4 | 0.09 | 0.6% | 1.0 | 0.13 | `if (((((int)threadIdx.x) % 8) == 0) && (((((int)blockIdx.y) * 16) + (((int)threadIdx.x) >>` |
| compute.reduce | 0.07 | 0.5% | 1.0 | 0.10 | `red[0] = (red[0] + __shfl_xor_sync((uint)4294967295, red[0], 4, 32));` |
| compute.reduce#2 | 0.07 | 0.5% | 1.0 | 0.10 | `red[0] = (red[0] + __shfl_xor_sync((uint)4294967295, red[0], 4, 32));` |
| store.global#2 | 0.06 | 0.4% | 1.0 | 0.10 | `if (((((int)blockIdx.y) * 16) + (((int)threadIdx.x) >> 3)) < M) \| if (((((int)threadIdx.x)` |
| store.global#3 | 0.06 | 0.4% | 1.0 | 0.06 | `if (((((int)threadIdx.x) % 8) == 0) && (((((int)blockIdx.y) * 16) + (((int)threadIdx.x) >>` |
| compute.mma | 0.00 | 0.0% | 0.0 | 0.00 | `{` |
| load.global#2 | 0.00 | 0.0% | 0.0 | 0.00 | `for (int i_1 = 0; i_1 < 6; ++i_1)` |
| store.global | 0.00 | 0.0% | 0.0 | 0.00 | `for (int n = 0; n < 4; ++n) \| if (((((int)blockIdx.y) * 16) + (((int)threadIdx.x) >> 3)) <` |
| wait.sync | 0.00 | 0.0% | 0.0 | 0.00 | `__syncthreads();` |

### gemm_fp8w_bf16_partials_kernel grid (32, 4, 1) block (128, 1, 1)

153 launches, 512 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 17.95 us, warp 16.25 us, nsys graphs 28.11 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 7.09 | 43.7% | 11.20 | 49.9% |
| load | 3.46 | 21.3% | 4.15 | 18.5% |
| unattributed | 3.16 | 19.4% | 4.48 | 20.0% |
| compute | 2.48 | 15.3% | 2.54 | 11.3% |
| store | 0.06 | 0.4% | 0.06 | 0.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*14 (est.: sampled iterations x14) | 2.24 | 13.8% | 1.0 | 2.69 | `{` |
| wait.sync#2*14 (est.: sampled iterations x14) | 1.23 | 7.6% | 1.0 | 1.79 | `__syncthreads();` |
| load.global*14 (est.: sampled iterations x14) | 1.09 | 6.7% | 1.0 | 1.79 | `if (((int)threadIdx.x) < 4)` |
| load.cp_async#2*14 (est.: sampled iterations x14) | 1.04 | 6.4% | 1.0 | 1.34 | `for (int i_4 = 0; i_4 < 2; ++i_4) \| for (int i_6 = 0; i_6 < 2; ++i_6)` |
| load.cp_async#3*14 (est.: sampled iterations x14) | 0.95 | 5.9% | 1.0 | 1.34 | `for (int i_10 = 0; i_10 < 4; ++i_10) \| for (int i_10 = 0; i_10 < 6; ++i_10) \| for (int i_1` |
| wait.sync#3*14 (est.: sampled iterations x14) | 0.92 | 5.7% | 1.0 | 1.34 | `__syncthreads();` |
| wait.sync#4*14 (est.: sampled iterations x14) | 0.92 | 5.7% | 1.0 | 1.34 | `__syncthreads();` |
| wait.sync#5*14 (est.: sampled iterations x14) | 0.92 | 5.7% | 1.0 | 1.34 | `__syncthreads();` |
| wait.sync*14 (est.: sampled iterations x14) | 0.90 | 5.5% | 1.0 | 1.34 | `__syncthreads();` |
| wait.cp_async*14 (est.: sampled iterations x14) | 0.87 | 5.3% | 1.0 | 0.90 | `tl::cp_async_wait<3>();` |
| wait.cp_async#2*14 (est.: sampled iterations x14) | 0.81 | 5.0% | 1.0 | 0.90 | `tl::cp_async_wait<3>();` |
| load.cp_async | 0.26 | 1.6% | 1.0 | 0.48 | `for (int i_1 = 0; i_1 < 2; ++i_1)` |
| compute.mma#2 | 0.13 | 0.8% | 1.0 | 0.16 | `{` |
| compute.mma#3 | 0.11 | 0.7% | 1.0 | 0.13 | `{` |
| wait.sync#10 | 0.06 | 0.4% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#7 | 0.06 | 0.4% | 1.0 | 0.10 | `__syncthreads();` |
| load.global#2 | 0.06 | 0.4% | 1.0 | 0.10 | `if (((int)threadIdx.x) < 4)` |
| load.global#3 | 0.06 | 0.4% | 1.0 | 0.10 | `if (((int)threadIdx.x) < 4)` |
| store.global | 0.06 | 0.4% | 1.0 | 0.06 | `for (int i_13 = 0; i_13 < 8; ++i_13) \| for (int i_19 = 0; i_19 < 16; ++i_19) \| for (int i_` |
| wait.cp_async#6 | 0.05 | 0.3% | 1.0 | 0.06 | `tl::cp_async_wait<0>();` |
| wait.sync#8 | 0.05 | 0.3% | 1.0 | 0.06 | `__syncthreads();` |
| wait.cp_async#3 | 0.05 | 0.3% | 1.0 | 0.06 | `tl::cp_async_wait<3>();` |
| wait.sync#11 | 0.05 | 0.3% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#6 | 0.05 | 0.3% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#9 | 0.05 | 0.3% | 1.0 | 0.06 | `__syncthreads();` |
| wait.cp_async#5 | 0.05 | 0.3% | 1.0 | 0.06 | `tl::cp_async_wait<1>();` |
| wait.cp_async#4 | 0.04 | 0.3% | 1.0 | 0.06 | `tl::cp_async_wait<2>();` |

### gemm_mxfp8_kernel grid (36, 1, 1) block (128, 1, 1)

153 launches, 144 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 22.14 us, warp 20.86 us, nsys graphs 28.61 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 10.88 | 52.2% | 6.21 | 28.5% |
| wait | 4.00 | 19.2% | 8.41 | 38.6% |
| compute | 3.53 | 16.9% | 3.55 | 16.3% |
| load | 2.40 | 11.5% | 3.58 | 16.4% |
| store | 0.05 | 0.2% | 0.05 | 0.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*18 (est.: sampled iterations x18) | 3.24 | 15.5% | 1.0 | 3.46 | `{` |
| load.cp_async#2*18 (est.: sampled iterations x18) | 2.16 | 10.4% | 1.0 | 2.30 | `for (int i_4 = 0; i_4 < 2; ++i_4) \| for (int i_5 = 0; i_5 < 4; ++i_5)` |
| wait.sync#2*18 (est.: sampled iterations x18) | 1.48 | 7.1% | 1.0 | 3.46 | `__syncthreads();` |
| wait.cp_async*18 (est.: sampled iterations x18) | 1.15 | 5.5% | 1.0 | 2.88 | `tl::cp_async_wait<2>();` |
| wait.sync*18 (est.: sampled iterations x18) | 1.12 | 5.4% | 1.0 | 1.15 | `__syncthreads();` |
| load.cp_async | 0.24 | 1.1% | 1.0 | 0.29 | `for (int i = 0; i < 2; ++i) \| for (int i_1 = 0; i_1 < 2; ++i_1) \| for (int i_1 = 0; i_1 < ` |
| compute.mma#2 | 0.15 | 0.7% | 1.0 | 0.16 | `{` |
| compute.mma#3 | 0.14 | 0.7% | 1.0 | 0.16 | `{` |
| wait.sync#4 | 0.08 | 0.4% | 1.0 | 0.19 | `__syncthreads();` |
| wait.cp_async#3 | 0.07 | 0.3% | 1.0 | 0.19 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>();` |
| wait.sync#3 | 0.06 | 0.3% | 1.0 | 0.06 | `__syncthreads();` |
| wait.cp_async#2 | 0.05 | 0.2% | 1.0 | 0.06 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| store.global | 0.05 | 0.2% | 1.0 | 0.06 | `for (int i_10 = 0; i_10 < 16; ++i_10) \| for (int i_10 = 0; i_10 < 8; ++i_10) \| for (int i_` |

### gemm_mxfp8_kernel grid (40, 1, 1) block (128, 1, 1)

153 launches, 160 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 19.71 us, warp 17.25 us, nsys graphs 23.71 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 7.35 | 42.6% | 9.28 | 45.2% |
| unattributed | 4.45 | 25.8% | 4.66 | 22.7% |
| compute | 3.23 | 18.7% | 3.24 | 15.8% |
| load | 2.15 | 12.5% | 3.27 | 15.9% |
| store | 0.06 | 0.4% | 0.06 | 0.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*14 (est.: sampled iterations x14) | 2.88 | 16.7% | 1.0 | 3.14 | `{` |
| wait.sync#2*14 (est.: sampled iterations x14) | 2.83 | 16.4% | 1.0 | 9.86 | `__syncthreads();` |
| wait.cp_async*14 (est.: sampled iterations x14) | 2.69 | 15.6% | 1.0 | 11.20 | `tl::cp_async_wait<2>();` |
| load.cp_async#2*14 (est.: sampled iterations x14) | 1.90 | 11.0% | 1.0 | 2.24 | `for (int i_5 = 0; i_5 < 2; ++i_5) \| tl::cp_async_gs_conditional<16>((&(((fp8_e4_t*)A_share` |
| wait.sync*14 (est.: sampled iterations x14) | 0.87 | 5.0% | 1.0 | 0.90 | `__syncthreads();` |
| wait.cp_async#2 | 0.29 | 1.7% | 1.0 | 1.15 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| wait.sync#3 | 0.26 | 1.5% | 1.0 | 0.93 | `__syncthreads();` |
| load.cp_async | 0.25 | 1.4% | 1.0 | 0.26 | `for (int i = 0; i < 2; ++i) \| for (int i_1 = 0; i_1 < 2; ++i_1) \| for (int i_1 = 0; i_1 < ` |
| wait.cp_async#3 | 0.23 | 1.3% | 1.0 | 0.93 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>();` |
| wait.sync#4 | 0.19 | 1.1% | 1.0 | 0.70 | `__syncthreads();` |
| compute.mma#3 | 0.18 | 1.0% | 1.0 | 0.19 | `{` |
| compute.mma#2 | 0.17 | 1.0% | 1.0 | 0.19 | `{` |
| store.global | 0.06 | 0.4% | 1.0 | 0.10 | `for (int i_10 = 0; i_10 < 16; ++i_10) \| for (int i_10 = 0; i_10 < 8; ++i_10) \| for (int i_` |

### gemm_bf16_partials_kernel grid (6, 8, 1) block (256, 1, 1)

102 launches, 384 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 26.43 us, warp 6.20 us, nsys graphs 23.47 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 3.84 | 61.9% | 1.35 | 32.4% |
| wait | 1.24 | 20.0% | 1.37 | 32.8% |
| compute | 0.68 | 11.0% | 1.37 | 32.9% |
| load | 0.40 | 6.4% | 0.00 | 0.0% |
| store | 0.04 | 0.6% | 0.08 | 1.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*10 (est.: sampled iterations x10) | 0.68 | 11.0% | 0.5 | 1.60 | `{` |
| wait.mbarrier*10 (est.: sampled iterations x10) | 0.68 | 10.9% | 0.5 | 1.60 | `mbarrier[((kb % 3) + 3)].wait((((kb % 6) / 3) ^ 1));` |
| wait.mbarrier#2*10 (est.: sampled iterations x10) | 0.47 | 7.6% | 0.5 | 1.76 | `mbarrier[(kb_1 % 3)].wait(((kb_1 % 6) / 3));` |
| load.tma*10 (est.: sampled iterations x10) | 0.40 | 6.4% | 0.5 | 1.28 | `if (tl::tl_shuffle_elect<128>())` |
| wait.sync | 0.10 | 1.6% | 1.0 | 0.13 | `__syncthreads();` |
| store.global | 0.04 | 0.6% | 0.5 | 0.10 | `for (int i_1 = 0; i_1 < 4; ++i_1) \| for (int i_2 = 0; i_2 < 8; ++i_2) \| for (int i_4 = 0; ` |

### mhc_finalize_kernel grid (5, 1, 1) block (128, 1, 1)

192 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 12.38 us, warp 11.20 us, nsys graphs 10.62 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 7.04 | 62.9% | 6.94 | 54.2% |
| store | 1.52 | 13.5% | 2.26 | 17.7% |
| load | 1.02 | 9.1% | 1.75 | 13.7% |
| compute | 1.00 | 8.9% | 1.11 | 8.7% |
| wait | 0.62 | 5.5% | 0.73 | 5.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync | 0.62 | 5.5% | 1.0 | 0.83 | `__syncthreads();` |
| store.global | 0.62 | 5.5% | 0.2 | 2.51 | `if (((int)threadIdx.x) < 4)` |
| compute.reduce | 0.57 | 5.1% | 0.2 | 2.30 | `row_max[0] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<32>>::run(row_max[0]);` |
| store.global#4*7 (est.: sampled iterations x7) | 0.56 | 5.0% | 0.8 | 0.90 | `if ((((int)threadIdx.x) % 4) == 0)` |
| compute.reduce#2*7 (est.: sampled iterations x7) | 0.43 | 3.8% | 0.8 | 0.67 | `amax[0] = max(amax[0], __shfl_xor_sync((uint)4294967295, amax[0], 1, 32));` |
| load.global#4*7 (est.: sampled iterations x7) | 0.36 | 3.2% | 0.8 | 0.67 | `*(uint4*)(yv + 0) = *(uint4*)(C + ((((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((i` |
| load.global | 0.34 | 3.0% | 1.0 | 1.02 | `if (((int)threadIdx.x) < 32)` |
| store.global#3*7 (est.: sampled iterations x7) | 0.32 | 2.9% | 0.8 | 0.45 | `*(uint4*)(y + ((((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((int64_t)step) * (int6` |
| load.global#3 | 0.28 | 2.5% | 0.8 | 0.38 | `for (int i_split_2 = 0; i_split_2 < 40; ++i_split_2)` |
| load.global#2 | 0.04 | 0.4% | 0.2 | 0.19 | `cm[0] = ((mixes_shared[((((int)threadIdx.x) & 15) + 8)] * scale[2]) + base[((((int)threadI` |
| store.global#2 | 0.01 | 0.1% | 0.2 | 0.06 | `if ((((int)threadIdx.x) >> 4) == 0)` |

### gemm_bf16_splitk_kernel grid (16, 5, 1) block (128, 1, 1)

34 launches, 320 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 56.27 us, warp 51.07 us, nsys graphs 53.46 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 23.51 | 46.0% | 53.58 | 99.5% |
| unattributed | 22.85 | 44.7% | 0.00 | 0.0% |
| load | 2.55 | 5.0% | 4.27 | 7.9% |
| compute | 2.04 | 4.0% | 2.13 | 4.0% |
| store | 0.12 | 0.2% | 0.15 | 0.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.cp_async*13 (est.: sampled iterations x13) | 9.96 | 19.5% | 0.5 | 17.68 | `tl::cp_async_wait<2>();` |
| wait.sync*13 (est.: sampled iterations x13) | 2.52 | 4.9% | 0.5 | 7.07 | `__syncthreads();` |
| wait.cp_async#2 | 2.01 | 3.9% | 1.0 | 4.74 | `tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>(); \| tl::cp_async_wait<3>();` |
| wait.cp_async#4 | 1.69 | 3.3% | 1.0 | 4.75 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>();` |
| wait.cp_async*12 (est.: sampled iterations x12) | 1.51 | 3.0% | 0.5 | 8.64 | `tl::cp_async_wait<3>();` |
| wait.sync*12 (est.: sampled iterations x12) | 1.45 | 2.8% | 0.5 | 7.87 | `__syncthreads();` |
| load.cp_async | 1.29 | 2.5% | 1.0 | 2.34 | `for (int i_1 = 0; i_1 < 2; ++i_1) \| tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)A_sha` |
| wait.cp_async#3 | 1.08 | 2.1% | 1.0 | 3.65 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| compute.mma*12 (est.: sampled iterations x12) | 0.85 | 1.7% | 0.5 | 0.96 | `{` |
| compute.mma*13 (est.: sampled iterations x13) | 0.83 | 1.6% | 0.5 | 0.83 | `{` |
| wait.sync#5 | 0.73 | 1.4% | 1.0 | 2.34 | `__syncthreads();` |
| wait.sync#3 | 0.68 | 1.3% | 1.0 | 2.50 | `__syncthreads();` |
| wait.sync#4 | 0.68 | 1.3% | 1.0 | 2.21 | `__syncthreads();` |
| load.cp_async#2*12 (est.: sampled iterations x12) | 0.49 | 1.0% | 0.5 | 0.58 | `tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)A_shared)[((((((kb & 3) * 1024) + ((((int` |
| load.cp_async#2*13 (est.: sampled iterations x13) | 0.42 | 0.8% | 0.5 | 0.42 | `for (int i_10 = 0; i_10 < 2; ++i_10) \| for (int i_8 = 0; i_8 < 2; ++i_8) \| tl::cp_async_gs` |
| load.global | 0.35 | 0.7% | 1.0 | 2.08 | `if (((int)threadIdx.x) == 0)` |
| wait.sync#2*13 (est.: sampled iterations x13) | 0.33 | 0.6% | 0.5 | 0.42 | `__syncthreads();` |
| wait.sync#2*12 (est.: sampled iterations x12) | 0.31 | 0.6% | 0.5 | 0.38 | `__syncthreads();` |
| wait.sync#7 | 0.22 | 0.4% | 1.0 | 0.45 | `__syncthreads();` |
| wait.sync#8 | 0.17 | 0.3% | 0.5 | 0.24 | `__syncthreads();` |
| wait.sync#6 | 0.12 | 0.2% | 1.0 | 0.26 | `__syncthreads();` |
| compute.mma#2 | 0.11 | 0.2% | 1.0 | 0.13 | `{` |
| compute.mma#4 | 0.10 | 0.2% | 1.0 | 0.13 | `{` |
| compute.mma#3 | 0.10 | 0.2% | 1.0 | 0.13 | `{` |
| store.global#3 | 0.06 | 0.1% | 1.0 | 0.10 | `if (((int*)last)[0] == 1)` |
| store.global | 0.06 | 0.1% | 1.0 | 0.06 | `for (int i_10 = 0; i_10 < 4; ++i_10) \| for (int i_16 = 0; i_16 < 4; ++i_16) \| for (int i_1` |
| compute.mma#5 | 0.05 | 0.1% | 0.5 | 0.06 | `{` |
| wait.cp_async#5 | 0.05 | 0.1% | 0.5 | 0.19 | `tl::cp_async_wait<0>();` |

### gemm_mxfp8_kernel grid (40, 1, 1) block (64, 1, 1)

153 launches, 80 warps; ranges, warp and last warp from the 4 recording CTAs' 8 warps; span 6.91 us, warp 6.18 us, nsys graphs 8.32 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 2.00 | 32.4% | 1.49 | 17.7% |
| wait | 1.75 | 28.3% | 4.28 | 50.7% |
| compute | 1.50 | 24.3% | 1.53 | 18.1% |
| load | 0.84 | 13.5% | 1.06 | 12.5% |
| store | 0.08 | 1.4% | 0.09 | 1.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*8 (est.: sampled iterations x8) | 1.34 | 21.7% | 1.0 | 1.54 | `{` |
| load.cp_async#2*8 (est.: sampled iterations x8) | 0.70 | 11.4% | 1.0 | 0.77 | `for (int i_3 = 0; i_3 < 2; ++i_3) \| for (int i_5 = 0; i_5 < 2; ++i_5) \| tl::cp_async_gs_co` |
| wait.cp_async*8 (est.: sampled iterations x8) | 0.61 | 9.8% | 1.0 | 1.54 | `tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| wait.sync#2*8 (est.: sampled iterations x8) | 0.48 | 7.8% | 1.0 | 1.02 | `__syncthreads();` |
| wait.sync*8 (est.: sampled iterations x8) | 0.38 | 6.2% | 1.0 | 0.51 | `__syncthreads();` |
| wait.cp_async#2 | 0.20 | 3.2% | 1.0 | 0.42 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| compute.mma#2 | 0.16 | 2.6% | 1.0 | 0.16 | `{` |
| load.cp_async | 0.13 | 2.1% | 1.0 | 0.16 | `for (int i = 0; i < 2; ++i) \| for (int i_1 = 0; i_1 < 2; ++i_1) \| for (int i_1 = 0; i_1 < ` |
| store.global | 0.08 | 1.4% | 1.0 | 0.10 | `for (int i_10 = 0; i_10 < 16; ++i_10) \| for (int i_10 = 0; i_10 < 8; ++i_10) \| for (int i_` |
| wait.sync#3 | 0.08 | 1.3% | 1.0 | 0.22 | `__syncthreads();` |

### sparse_mla_kernel grid (5, 1, 5) block (128, 1, 1)

64 launches, 100 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 14.70 us, warp 14.32 us, nsys graphs 11.94 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed, with loops' unsampled iterations | 9.81 | 68.5% | 10.38 | 49.2% |
| wait | 1.71 | 11.9% | 0.68 | 3.2% |
| load | 1.40 | 9.7% | 8.56 | 40.6% |
| compute | 0.96 | 6.7% | 0.96 | 4.6% |
| store | 0.44 | 3.1% | 0.52 | 2.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync (sampled iterations) | 1.64 | 11.5% | 11.0 | 2.22 | `__syncthreads();` |
| load.global (sampled iterations) | 1.07 | 7.4% | 2.0 | 2.24 | `for (int c = 0; c < 4; ++c) \| for (int i = 0; i < 8; ++i) \| for (int i_1 = 0; i_1 < 8; ++i` |
| compute.mma (sampled iterations) | 0.74 | 5.2% | 2.0 | 0.77 | `for (int ki = 0; ki < 32; ++ki) \| for (int ki_1 = 0; ki_1 < 2; ++ki_1)` |
| store.global | 0.44 | 3.1% | 1.0 | 0.54 | `for (int i_10 = 0; i_10 < 32; ++i_10) \| for (int i_11 = 0; i_11 < 32; ++i_11) \| for (int i` |
| load.cp_async (sampled iterations) | 0.33 | 2.3% | 2.0 | 0.42 | `for (int k_1 = 0; k_1 < 5; ++k_1) \| for (int k_1 = 0; k_1 < 9; ++k_1) \| for (int k_3 = 0; ` |
| compute.reduce (sampled iterations) | 0.22 | 1.6% | 2.0 | 0.26 | `m_g[i_5] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(m_g[i_5], (&((` |
| wait.cp_async (sampled iterations) | 0.07 | 0.5% | 1.0 | 0.10 | `tl::cp_async_wait<0>();` |

### sparse_mla_kernel grid (2, 1, 10) block (128, 1, 1)

51 launches, 80 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 24.61 us, warp 17.50 us, nsys graphs 19.07 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed, with loops' unsampled iterations | 10.12 | 57.8% | 8.83 | 37.3% |
| compute | 2.50 | 14.3% | 4.75 | 20.1% |
| load | 2.45 | 14.0% | 6.72 | 28.4% |
| wait | 1.99 | 11.4% | 2.85 | 12.0% |
| store | 0.44 | 2.5% | 0.51 | 2.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 | 1.48 | 8.5% | 1.0 | 4.74 | `for (int i = 0; i < 8; ++i)` |
| compute.mma*4 (est.: sampled iterations x4) | 0.86 | 4.9% | 0.5 | 1.79 | `for (int ki = 0; ki < 32; ++ki) \| {` |
| compute.mma#2*4 (est.: sampled iterations x4) | 0.74 | 4.2% | 1.0 | 1.28 | `for (int ki_1 = 0; ki_1 < 2; ++ki_1) \| if (((((int)blockIdx.x) * 2) + (c_1 >> 1)) < 3)` |
| load.cp_async#2*4 (est.: sampled iterations x4) | 0.50 | 2.8% | 1.0 | 0.77 | `if (((((int)blockIdx.x) * 2) + (c_1 >> 1)) < 3) \| if ((c_1 < 3) & (c_1 < 3)) \| if (c < 3)` |
| compute.reduce#2*8 (est.: sampled iterations x8) | 0.45 | 2.6% | 0.5 | 1.02 | `row_sum[i_7] = tl::AllReduce<tl::SumOp, 128, 32, 0, tl::NamedBarrier<128>>::run(row_sum[i_` |
| compute.reduce*8 (est.: sampled iterations x8) | 0.45 | 2.6% | 0.5 | 1.02 | `m_g[i_4] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(m_g[i_4], (&((` |
| store.global | 0.44 | 2.5% | 1.0 | 0.51 | `for (int i_10 = 0; i_10 < 32; ++i_10) \| for (int i_11 = 0; i_11 < 32; ++i_11) \| for (int i` |
| load.cp_async (sampled iterations) | 0.33 | 1.9% | 1.0 | 0.35 | `for (int k_1 = 0; k_1 < 5; ++k_1) \| for (int k_1 = 0; k_1 < 9; ++k_1) \| for (int k_3 = 0; ` |
| wait.sync#7*4 (est.: sampled iterations x4) | 0.25 | 1.4% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#6*4 (est.: sampled iterations x4) | 0.24 | 1.4% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#9*4 (est.: sampled iterations x4) | 0.24 | 1.4% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#11*4 (est.: sampled iterations x4) | 0.23 | 1.3% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync (sampled iterations) | 0.18 | 1.0% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#10*8 (est.: sampled iterations x8) | 0.18 | 1.0% | 0.5 | 0.51 | `__syncthreads();` |
| wait.sync#8*8 (est.: sampled iterations x8) | 0.18 | 1.0% | 0.5 | 0.51 | `__syncthreads();` |
| load.global (sampled iterations) | 0.14 | 0.8% | 1.0 | 0.35 | `for (int c = 0; c < 4; ++c) \| for (int i = 0; i < 8; ++i) \| for (int i_1 = 0; i_1 < 8; ++i` |
| wait.cp_async*4 (est.: sampled iterations x4) | 0.14 | 0.8% | 0.5 | 0.38 | `tl::cp_async_wait<0>();` |
| wait.sync#4*4 (est.: sampled iterations x4) | 0.13 | 0.7% | 0.5 | 0.26 | `__syncthreads();` |
| wait.sync#5*4 (est.: sampled iterations x4) | 0.13 | 0.7% | 0.5 | 0.26 | `__syncthreads();` |
| wait.sync#3 | 0.06 | 0.3% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#2 | 0.05 | 0.3% | 1.0 | 0.06 | `__syncthreads();` |

### mhc_finalize_kernel grid (10, 1, 1) block (128, 1, 1)

102 launches, 40 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 12.08 us, warp 10.66 us, nsys graphs 9.68 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 6.55 | 61.5% | 5.42 | 47.8% |
| store | 1.54 | 14.5% | 2.77 | 24.4% |
| compute | 1.00 | 9.4% | 1.30 | 11.5% |
| load | 0.98 | 9.2% | 1.30 | 11.5% |
| wait | 0.59 | 5.5% | 0.55 | 4.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.63 | 5.9% | 0.2 | 2.59 | `if (((int)threadIdx.x) < 4)` |
| wait.sync | 0.59 | 5.5% | 1.0 | 0.82 | `__syncthreads();` |
| compute.reduce | 0.57 | 5.4% | 0.2 | 2.30 | `row_max[0] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<32>>::run(row_max[0]);` |
| store.global#4*7 (est.: sampled iterations x7) | 0.57 | 5.4% | 0.8 | 0.90 | `if ((((int)threadIdx.x) % 4) == 0)` |
| compute.reduce#2*7 (est.: sampled iterations x7) | 0.43 | 4.0% | 0.8 | 0.67 | `amax[0] = max(amax[0], __shfl_xor_sync((uint)4294967295, amax[0], 1, 32));` |
| load.global#4*7 (est.: sampled iterations x7) | 0.36 | 3.4% | 0.8 | 0.67 | `*(uint4*)(yv + 0) = *(uint4*)(C + ((((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((i` |
| store.global#3*7 (est.: sampled iterations x7) | 0.32 | 3.0% | 0.8 | 0.45 | `*(uint4*)(y + ((((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((int64_t)step) * (int6` |
| load.global | 0.32 | 3.0% | 1.0 | 1.02 | `if (((int)threadIdx.x) < 32)` |
| load.global#3 | 0.25 | 2.3% | 0.8 | 0.35 | `for (int i_split_2 = 0; i_split_2 < 40; ++i_split_2)` |
| load.global#2 | 0.05 | 0.4% | 0.2 | 0.19 | `cm[0] = ((mixes_shared[((((int)threadIdx.x) & 15) + 8)] * scale[2]) + base[((((int)threadI` |
| store.global#2 | 0.01 | 0.1% | 0.2 | 0.06 | `if ((((int)threadIdx.x) >> 4) == 0)` |

### sparse_mla_kernel grid (1, 1, 5) block (128, 1, 1)

32 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 30.93 us, warp 30.87 us, nsys graphs 22.19 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed, with loops' unsampled iterations | 13.69 | 44.3% | 10.68 | 36.3% |
| load | 7.11 | 23.0% | 10.72 | 36.4% |
| wait | 4.78 | 15.5% | 2.67 | 9.1% |
| compute | 4.64 | 15.0% | 4.69 | 15.9% |
| store | 0.65 | 2.1% | 0.70 | 2.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 | 5.30 | 17.2% | 1.0 | 6.86 | `for (int i = 0; i < 8; ++i)` |
| wait.sync (sampled iterations) | 2.22 | 7.2% | 1.0 | 2.94 | `__syncthreads();` |
| compute.mma*4 (est.: sampled iterations x4) | 1.75 | 5.7% | 1.0 | 1.79 | `for (int ki = 0; ki < 32; ++ki) \| {` |
| compute.mma#2*4 (est.: sampled iterations x4) | 1.15 | 3.7% | 1.0 | 1.15 | `for (int ki_1 = 0; ki_1 < 2; ++ki_1) \| if (((((int)blockIdx.x) * 2) + (c_1 >> 1)) < 3)` |
| compute.reduce#2*8 (est.: sampled iterations x8) | 0.90 | 2.9% | 1.0 | 1.02 | `row_sum[i_7] = tl::AllReduce<tl::SumOp, 128, 32, 0, tl::NamedBarrier<128>>::run(row_sum[i_` |
| compute.reduce*8 (est.: sampled iterations x8) | 0.84 | 2.7% | 1.0 | 1.02 | `m_g[i_4] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(m_g[i_4], (&((` |
| load.global (sampled iterations) | 0.80 | 2.6% | 1.0 | 3.02 | `for (int c = 0; c < 4; ++c) \| for (int i = 0; i < 8; ++i) \| for (int i_1 = 0; i_1 < 8; ++i` |
| load.cp_async#2*4 (est.: sampled iterations x4) | 0.65 | 2.1% | 1.0 | 0.77 | `if (((((int)blockIdx.x) * 2) + (c_1 >> 1)) < 3) \| if ((c_1 < 3) & (c_1 < 3)) \| if (c < 3)` |
| store.global | 0.65 | 2.1% | 1.0 | 0.70 | `for (int i_10 = 0; i_10 < 32; ++i_10) \| for (int i_11 = 0; i_11 < 32; ++i_11) \| for (int i` |
| load.cp_async (sampled iterations) | 0.36 | 1.2% | 1.0 | 0.38 | `for (int k_1 = 0; k_1 < 5; ++k_1) \| for (int k_1 = 0; k_1 < 9; ++k_1) \| for (int k_3 = 0; ` |
| wait.sync#10*8 (est.: sampled iterations x8) | 0.33 | 1.1% | 1.0 | 0.51 | `__syncthreads();` |
| wait.sync#8*8 (est.: sampled iterations x8) | 0.33 | 1.1% | 1.0 | 0.51 | `__syncthreads();` |
| wait.cp_async*4 (est.: sampled iterations x4) | 0.27 | 0.9% | 1.0 | 0.38 | `tl::cp_async_wait<0>();` |
| wait.sync#11*4 (est.: sampled iterations x4) | 0.26 | 0.8% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#4*4 (est.: sampled iterations x4) | 0.26 | 0.8% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#5*4 (est.: sampled iterations x4) | 0.26 | 0.8% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#6*4 (est.: sampled iterations x4) | 0.26 | 0.8% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#7*4 (est.: sampled iterations x4) | 0.26 | 0.8% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#9*4 (est.: sampled iterations x4) | 0.26 | 0.8% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#3 | 0.06 | 0.2% | 1.0 | 0.08 | `__syncthreads();` |
| wait.sync#2 | 0.04 | 0.1% | 1.0 | 0.06 | `__syncthreads();` |

### cache_writer_kernel grid (5, 1, 1) block (128, 1, 1)

128 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 8.59 us, warp 8.25 us, nsys graphs 4.56 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 5.40 | 65.5% | 3.58 | 49.0% |
| store | 2.48 | 30.1% | 3.04 | 41.6% |
| load | 0.29 | 3.6% | 0.58 | 7.9% |
| compute | 0.07 | 0.9% | 0.10 | 1.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 2.48 | 30.1% | 1.0 | 2.53 | `if (((int64_t)0 <= slot) & ((slot >> (int64_t)7) < ((int64_t)pages))) \| if (((int64_t)0 <=` |
| load.global | 0.24 | 2.8% | 1.0 | 0.26 | `for (int i = 0; i < 4; ++i)` |
| compute.reduce | 0.07 | 0.9% | 1.0 | 0.10 | `amax = max(amax, __shfl_xor_sync((uint)4294967295, amax, 1, 32));` |
| load.global#2 | 0.06 | 0.7% | 1.0 | 0.06 | `int64_t slot = slots[((int64_t)((int)blockIdx.x))];` |

### metadata_chunk_kernel grid (10, 1, 1) block (192, 1, 1)

51 launches, 60 warps; ranges, warp and last warp from the 5 recording CTAs' 30 warps; span 14.05 us, warp 8.39 us, nsys graphs 10.46 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 5.24 | 62.4% | 13.32 | 86.8% |
| unattributed | 3.08 | 36.7% | 1.90 | 12.4% |
| store | 0.07 | 0.8% | 0.13 | 0.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 5.14 | 61.3% | 1.0 | 9.89 | `if ((((offset + ((int)blockIdx.x)) < src_rows) && (0 <= (offset + ((int)blockIdx.x)))))` |
| load.global#2 | 0.10 | 1.2% | 0.2 | 0.96 | `if (((0 <= req) & ((int64_t)0 <= pos)))` |
| store.global | 0.05 | 0.6% | 1.0 | 0.10 | `swa[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = con` |
| store.global#3 | 0.01 | 0.1% | 0.2 | 0.06 | `top_lengths[((int64_t)((int)blockIdx.x))] = condval_12; \| top_lengths[((int64_t)((int)bloc` |
| store.global#2 | 0.01 | 0.1% | 0.2 | 0.06 | `lengths[((int64_t)((int)blockIdx.x))] = ((int)max(info[1], (int64_t)0));` |

### rope_kernel grid (1, 1, 1) block (128, 1, 1)

272 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 1.52 us, warp 1.36 us, nsys graphs 1.25 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.19 | 87.6% | 2.41 | 86.2% |
| load | 0.15 | 11.2% | 0.34 | 12.2% |
| store | 0.02 | 1.2% | 0.05 | 1.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.10 | 7.1% | 1.0 | 0.10 | `if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < tokens)) \| int64_t position ` |
| load.global#2 | 0.06 | 4.1% | 0.5 | 0.10 | `if (((position < ((int64_t)positions_max)) && ((int64_t)0 <= position)))` |
| store.global | 0.02 | 1.2% | 0.5 | 0.03 | `*(uint4*)(x + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + ((((int64_t)((int)thread` |

### moe_dispatch_kernel grid (1, 1, 1) block (512, 1, 1)

153 launches, 16 warps; ranges, warp and last warp from the 1 recording CTAs' 16 warps; span 4.70 us, warp 4.51 us, nsys graphs 2.94 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 2.57 | 57.0% | 1.66 | 35.3% |
| unattributed, with loops' unsampled iterations | 1.25 | 27.8% | 1.75 | 37.3% |
| store | 0.60 | 13.4% | 1.19 | 25.3% |
| load | 0.08 | 1.8% | 0.10 | 2.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync#7*9 (est.: sampled iterations x9) | 0.67 | 14.8% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#6*9 (est.: sampled iterations x9) | 0.63 | 14.0% | 1.0 | 0.86 | `__syncthreads();` |
| store.global (sampled iterations) | 0.53 | 11.7% | 1.0 | 0.96 | `ChunkCount[((((int64_t)c) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = count; \| Chu` |
| wait.sync#8 | 0.51 | 11.2% | 1.0 | 0.80 | `__syncthreads();` |
| wait.sync#2 | 0.39 | 8.7% | 1.0 | 0.42 | `__syncthreads();` |
| wait.sync#3 | 0.13 | 2.9% | 1.0 | 0.32 | `__syncthreads();` |
| load.global (sampled iterations) | 0.08 | 1.8% | 1.0 | 0.10 | `expert = ((int)TopkIdx[((((int64_t)((int)blockIdx.x)) * (int64_t)512) + ((int64_t)((int)th` |
| store.global#2 | 0.08 | 1.7% | 1.0 | 0.29 | `RoutePos[((((int64_t)((int)blockIdx.x)) * (int64_t)512) + ((int64_t)((int)threadIdx.x)))] ` |
| wait.sync | 0.07 | 1.5% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#9 | 0.06 | 1.4% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#4 | 0.06 | 1.3% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#5 | 0.06 | 1.3% | 1.0 | 0.06 | `__syncthreads();` |

### gemm_bf16_partials_kernel grid (2, 20, 1) block (256, 1, 1)

51 launches, 320 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 12.77 us, warp 3.54 us, nsys graphs 15.44 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 2.57 | 72.6% | 0.94 | 48.4% |
| wait | 0.42 | 11.9% | 0.48 | 24.5% |
| load | 0.28 | 7.8% | 0.00 | 0.0% |
| compute | 0.23 | 6.4% | 0.45 | 23.2% |
| store | 0.04 | 1.3% | 0.08 | 3.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.tma*4 (est.: sampled iterations x4) | 0.28 | 7.8% | 0.5 | 1.02 | `if (tl::tl_shuffle_elect<128>())` |
| compute.mma*4 (est.: sampled iterations x4) | 0.23 | 6.4% | 0.5 | 0.51 | `{` |
| wait.mbarrier#2*4 (est.: sampled iterations x4) | 0.20 | 5.5% | 0.5 | 0.77 | `mbarrier[(kb_1 % 3)].wait((kb_1 / 3));` |
| wait.mbarrier*4 (est.: sampled iterations x4) | 0.12 | 3.4% | 0.5 | 0.26 | `mbarrier[((kb % 3) + 3)].wait(((kb / 3) ^ 1));` |
| wait.sync | 0.10 | 2.9% | 1.0 | 0.16 | `__syncthreads();` |
| store.global | 0.04 | 1.3% | 0.5 | 0.13 | `for (int i_1 = 0; i_1 < 4; ++i_1) \| for (int i_2 = 0; i_2 < 8; ++i_2) \| for (int i_4 = 0; ` |

### gemm_mxfp8_partials_kernel grid (16, 5, 1) block (128, 1, 1)

51 launches, 320 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 12.54 us, warp 11.50 us, nsys graphs 12.45 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 5.64 | 49.1% | 5.90 | 46.6% |
| wait | 4.74 | 41.2% | 4.27 | 33.7% |
| compute | 0.61 | 5.3% | 0.62 | 4.9% |
| load | 0.46 | 4.0% | 1.82 | 14.4% |
| store | 0.05 | 0.5% | 0.05 | 0.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.cp_async#3 | 1.74 | 15.2% | 1.0 | 3.46 | `tl::cp_async_wait<0>();` |
| wait.sync#2*2 (est.: sampled iterations x2) | 0.85 | 7.4% | 1.0 | 2.75 | `__syncthreads();` |
| wait.cp_async#2 | 0.82 | 7.1% | 1.0 | 2.02 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>();` |
| wait.sync#3 | 0.47 | 4.1% | 1.0 | 1.57 | `__syncthreads();` |
| wait.cp_async*2 (est.: sampled iterations x2) | 0.40 | 3.4% | 1.0 | 2.75 | `tl::cp_async_wait<2>();` |
| wait.sync#4 | 0.37 | 3.3% | 1.0 | 1.12 | `__syncthreads();` |
| compute.mma*2 (est.: sampled iterations x2) | 0.31 | 2.7% | 1.0 | 0.32 | `{` |
| load.cp_async | 0.24 | 2.1% | 1.0 | 0.32 | `for (int i = 0; i < 2; ++i) \| for (int i_1 = 0; i_1 < 4; ++i_1)` |
| load.cp_async#2*2 (est.: sampled iterations x2) | 0.22 | 1.9% | 1.0 | 0.26 | `for (int i_4 = 0; i_4 < 2; ++i_4)` |
| compute.mma#3 | 0.15 | 1.3% | 1.0 | 0.16 | `{` |
| compute.mma#2 | 0.15 | 1.3% | 1.0 | 0.16 | `{` |
| wait.sync*2 (est.: sampled iterations x2) | 0.10 | 0.8% | 1.0 | 0.13 | `__syncthreads();` |
| store.global | 0.05 | 0.5% | 1.0 | 0.06 | `for (int i_11 = 0; i_11 < 16; ++i_11) \| for (int i_13 = 0; i_13 < 8; ++i_13) \| for (int i_` |

### moe_grouped_fp8_fp4_gemm_kernel grid (9, 36, 1) block (128, 1, 1)

6 launches, 1296 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 98.62 us, warp 18.03 us, nsys graphs 2.94 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 6.58 | 36.5% | 0.00 | 0.0% |
| wait | 6.37 | 35.3% | 18.63 | 71.1% |
| compute | 3.20 | 17.7% | 4.48 | 17.1% |
| load | 1.82 | 10.1% | 3.18 | 12.1% |
| store | 0.07 | 0.4% | 0.12 | 0.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*40 (est.: sampled iterations x40) | 3.20 | 17.7% | 0.2 | 12.80 | `{` |
| wait.mbarrier*40 (est.: sampled iterations x40) | 3.20 | 17.7% | 0.2 | 12.80 | `loaded[(ko % 3)].wait(((ko % 6) / 3));` |
| load#2*40 (est.: sampled iterations x40) | 1.52 | 8.4% | 0.2 | 8.96 | `if (ko < 38)` |
| wait.sync#5*40 (est.: sampled iterations x40) | 1.24 | 6.9% | 0.2 | 5.76 | `__syncthreads();` |
| wait.sync#4*40 (est.: sampled iterations x40) | 0.64 | 3.5% | 0.2 | 2.56 | `__syncthreads();` |
| wait.cp_async*40 (est.: sampled iterations x40) | 0.60 | 3.3% | 0.2 | 2.56 | `tl::cp_async_wait<2>();` |
| wait.sync#6*40 (est.: sampled iterations x40) | 0.60 | 3.3% | 0.2 | 2.56 | `__syncthreads();` |
| load.global | 0.12 | 0.7% | 1.0 | 0.29 | `int expert = TileExpert[((int64_t)((int)blockIdx.y))];` |
| load.global#3 | 0.11 | 0.6% | 0.2 | 0.45 | `if (((expert < 128) && (0 <= expert))) \| if (((expert < 384) && (0 <= expert)))` |
| store.global | 0.07 | 0.4% | 0.2 | 0.35 | `for (int i_2 = 0; i_2 < 16; ++i_2) \| for (int i_3 = 0; i_3 < 16; ++i_3) \| for (int i_4 = 0` |
| wait.sync | 0.05 | 0.3% | 1.0 | 0.06 | `__syncthreads();` |
| load | 0.04 | 0.2% | 0.2 | 0.22 | `for (int s = 0; s < 2; ++s)` |
| wait.sync#2 | 0.03 | 0.2% | 0.2 | 0.16 | `__syncthreads();` |
| load.global#2 | 0.02 | 0.1% | 0.2 | 0.19 | `if (((int)threadIdx.x) < 32) \| if (((int)threadIdx.x) < 64)` |
| wait.sync#3 | 0.01 | 0.0% | 0.2 | 0.03 | `__syncthreads();` |

### partial_argmax_kernel grid (2, 64, 1) block (128, 1, 1)

85 launches, 512 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 7.42 us, warp 6.20 us, nsys graphs 7.17 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 5.45 | 88.0% | 6.18 | 84.9% |
| wait | 0.28 | 4.5% | 0.16 | 2.2% |
| load | 0.27 | 4.4% | 0.68 | 9.4% |
| compute | 0.12 | 2.0% | 0.12 | 1.7% |
| store | 0.08 | 1.2% | 0.13 | 1.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync | 0.28 | 4.5% | 1.0 | 1.12 | `__syncthreads();` |
| load.global | 0.27 | 4.4% | 1.0 | 0.42 | `for (int j = 0; j < 16; ++j) \| for (int run = 0; run < 2; ++run)` |
| compute.reduce | 0.12 | 2.0% | 1.0 | 0.13 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| store.global | 0.08 | 1.2% | 1.0 | 0.13 | `if (((int)threadIdx.x) == 0)` |

### router_partials_kernel grid (3, 1, 1) block (64, 1, 1)

102 launches, 6 warps; ranges, warp and last warp from the 3 recording CTAs' 6 warps; span 5.18 us, warp 3.14 us, nsys graphs 4.86 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.33 | 42.5% | 2.10 | 37.0% |
| load | 0.99 | 31.6% | 2.02 | 35.5% |
| compute | 0.68 | 21.8% | 1.36 | 24.0% |
| store | 0.13 | 4.1% | 0.20 | 3.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.reduce | 0.68 | 21.8% | 0.5 | 1.38 | `for (int k = 0; k < 3; ++k) \| for (int k = 0; k < 6; ++k)` |
| load.global#4 | 0.50 | 15.8% | 0.5 | 1.06 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < image_len)) \| if ((((((int)bl` |
| load.global | 0.39 | 12.4% | 0.8 | 0.48 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < mask_len))` |
| load.global#2 | 0.05 | 1.7% | 0.8 | 0.06 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < mask_len))` |
| load.global#3 | 0.05 | 1.7% | 0.8 | 0.06 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < mask_len))` |
| store.global#3 | 0.05 | 1.5% | 0.5 | 0.10 | `if ((((int)threadIdx.x) & 31) < 3) \| if ((((int)threadIdx.x) & 31) < 6)` |
| store.global#2 | 0.04 | 1.4% | 0.8 | 0.06 | `if (condval_1 == (bool)0) \| if (condval_1 == (bool)1)` |
| store.global | 0.04 | 1.2% | 0.8 | 0.06 | `if (condval == (bool)0) \| if (condval == (bool)1)` |

### norm_forward_and_per_token_cast_kernel_kernel grid (1, 1, 1) block (128, 1, 1)

187 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 2.75 us, warp 2.75 us, nsys graphs 1.28 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 2.10 | 76.5% | 2.22 | 79.7% |
| load | 0.38 | 14.0% | 0.42 | 14.9% |
| compute | 0.10 | 3.8% | 0.10 | 3.8% |
| wait | 0.10 | 3.8% | 0.05 | 1.8% |
| unattributed | 0.06 | 2.0% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*4 (est.: sampled iterations x4) | 1.54 | 55.8% | 1.0 | 1.54 | `for (int j_2 = 0; j_2 < 2; ++j_2)` |
| store.global*4 (est.: sampled iterations x4) | 0.35 | 12.8% | 1.0 | 0.38 | `for (int j_1 = 0; j_1 < 4; ++j_1)` |
| load.global*4 (est.: sampled iterations x4) | 0.22 | 8.1% | 1.0 | 0.26 | `if (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 4)) < num_tokens)` |
| load.global#3*4 (est.: sampled iterations x4) | 0.13 | 4.7% | 1.0 | 0.13 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 128) + ((((int)threadIdx.x) & 15` |
| store.global#5*4 (est.: sampled iterations x4) | 0.13 | 4.7% | 0.8 | 0.13 | `*(uint4*)(out + ((((((int64_t)vi_1) * (int64_t)128) + ((((int64_t)((int)threadIdx.x)) & (i` |
| compute.reduce | 0.10 | 3.8% | 1.0 | 0.13 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| wait.sync | 0.10 | 3.8% | 1.0 | 0.29 | `__syncthreads();` |
| store.global#2 | 0.05 | 1.7% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| store.global#3 | 0.04 | 1.5% | 0.8 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |
| load.global#2 | 0.03 | 1.2% | 1.0 | 0.03 | `*(float2*)(row_partial + 0) = *(float2*)(sum_shared + ((((int)threadIdx.x) >> 6) * 2)); \| ` |
| load.global#3*2 (est.: sampled iterations x2) | 0.00 | 0.0% | 0.0 | 0.00 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 64) + ((((int)threadIdx.x) & 7) ` |
| load.global*2 (est.: sampled iterations x2) | 0.00 | 0.0% | 0.0 | 0.00 | `if (((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < num_tokens)` |
| store.global#4*2 (est.: sampled iterations x2) | 0.00 | 0.0% | 0.0 | 0.00 | `for (int j_2 = 0; j_2 < 2; ++j_2)` |
| store.global#5*2 (est.: sampled iterations x2) | 0.00 | 0.0% | 0.0 | 0.00 | `*(uint4*)(out + ((((((int64_t)vi_1) * (int64_t)64) + ((((int64_t)((int)threadIdx.x)) & (in` |
| store.global*2 (est.: sampled iterations x2) | 0.00 | 0.0% | 0.0 | 0.00 | `for (int j_1 = 0; j_1 < 4; ++j_1)` |

### metadata_decode_kernel grid (5, 2, 1) block (128, 1, 1)

64 launches, 40 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 3.41 us, warp 2.20 us, nsys graphs 2.85 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.16 | 52.7% | 2.49 | 31.5% |
| load | 0.95 | 43.1% | 5.17 | 65.4% |
| store | 0.09 | 4.2% | 0.24 | 3.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.68 | 30.8% | 0.6 | 1.34 | `if ((((offset + ((int)blockIdx.x)) < src_rows) && (0 <= (offset + ((int)blockIdx.x)))))` |
| load.global#3 | 0.24 | 11.0% | 0.4 | 0.66 | `if ((((offset + ((int)blockIdx.x)) < main_src_rows) && (0 <= (offset + ((int)blockIdx.x)))` |
| store.global#4 | 0.04 | 2.0% | 0.4 | 0.22 | `if (0 < live)` |
| store.global | 0.04 | 1.6% | 0.6 | 0.06 | `swa[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = con` |
| load.global#2 | 0.03 | 1.3% | 0.1 | 0.19 | `if (((0 <= req) & ((int64_t)0 <= pos)))` |
| store.global#2 | 0.01 | 0.3% | 0.1 | 0.06 | `lengths[((int64_t)((int)blockIdx.x))] = ((int)max(info[1], (int64_t)0));` |
| store.global#3 | 0.01 | 0.3% | 0.1 | 0.06 | `top_lengths[((int64_t)((int)blockIdx.x))] = condval_6; \| top_lengths[((int64_t)((int)block` |

### moe_swiglu_cast_kernel_kernel grid (8, 30, 1) block (288, 1, 1)

147 launches, 2160 warps; ranges, warp and last warp from the 4 recording CTAs' 36 warps; span 2.85 us, warp 1.10 us, nsys graphs 3.20 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.64 | 58.5% | 1.24 | 51.9% |
| load | 0.24 | 22.1% | 0.48 | 20.0% |
| compute | 0.11 | 10.1% | 0.42 | 17.7% |
| store | 0.10 | 9.4% | 0.25 | 10.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.24 | 22.1% | 1.0 | 0.35 | `for (int r = 0; r < 4; ++r)` |
| compute.reduce*4 (est.: sampled iterations x4) | 0.11 | 10.1% | 0.5 | 0.38 | `if (0 <= orders[r_2])` |
| store.global*4 (est.: sampled iterations x4) | 0.10 | 9.4% | 0.5 | 0.26 | `if (0 <= orders[r_2])` |

### metadata_builder_kernel grid (2, 1, 1) block (128, 1, 1)

289 launches, 8 warps; ranges, warp and last warp from the 2 recording CTAs' 8 warps; span 1.25 us, warp 0.68 us, nsys graphs 1.44 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.31 | 45.9% | 0.34 | 22.0% |
| load | 0.29 | 42.9% | 1.14 | 74.1% |
| store | 0.08 | 11.2% | 0.06 | 4.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#3 | 0.25 | 37.1% | 0.5 | 0.86 | `for (int __1 = 0; __1 < 2; ++__1)` |
| store.global#4 | 0.03 | 4.7% | 0.5 | 0.06 | `positions[(((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))` |
| load.global#2 | 0.03 | 4.1% | 0.1 | 0.22 | `if ((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) < nr))` |
| store.global#3 | 0.03 | 4.1% | 0.5 | 0.06 | `if (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) == 0)` |
| load.global | 0.01 | 1.8% | 0.1 | 0.10 | `if ((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) <= nr))` |
| store.global | 0.01 | 1.8% | 0.1 | 0.10 | `out_starts[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] = condval;` |
| store.global#2 | 0.00 | 0.6% | 0.1 | 0.03 | `request_positions[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] = condval_2;` |

### token_ranks_kernel grid (1, 1, 1) block (256, 1, 1)

51 launches, 8 warps; ranges, warp and last warp from the 1 recording CTAs' 8 warps; span 7.62 us, warp 7.60 us, nsys graphs 8.03 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 6.85 | 90.1% | 7.05 | 90.1% |
| compute | 0.36 | 4.7% | 0.36 | 4.6% |
| unattributed | 0.36 | 4.7% | 0.37 | 4.7% |
| store | 0.04 | 0.6% | 0.05 | 0.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 6.85 | 90.1% | 1.0 | 7.01 | `if ((((int64_t)0 <= token_ids[((int64_t)((int)blockIdx.x))]) && (token_ids[((int64_t)((int` |
| compute.reduce | 0.36 | 4.7% | 1.0 | 0.54 | `total[0] = tl::AllReduce<tl::SumOp, 256, 1, 0, tl::NamedBarrier<256>>::run(total[0], (&(wo` |
| store.global | 0.04 | 0.6% | 1.0 | 0.06 | `output[((int64_t)((int)blockIdx.x))] = ((int64_t)total[0]);` |

### indexer_topk_segment_kernel grid (1, 5, 1) block (512, 1, 1)

32 launches, 80 warps; ranges, warp and last warp from the 5 recording CTAs' 80 warps; span 11.04 us, warp 10.76 us, nsys graphs 5.33 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed, with loops' unsampled iterations | 5.64 | 52.4% | 5.42 | 50.2% |
| wait | 3.72 | 34.6% | 3.95 | 36.6% |
| load | 1.23 | 11.4% | 1.15 | 10.7% |
| store | 0.17 | 1.6% | 0.27 | 2.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.23 | 11.4% | 1.0 | 1.49 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 4096); \| i` |
| wait.sync (sampled iterations) | 1.00 | 9.3% | 14.5 | 1.18 | `__syncthreads();` |
| wait.sync#24*9 (est.: sampled iterations x9) | 0.31 | 2.9% | 0.5 | 0.43 | `__syncthreads();` |
| wait.sync#20*9 (est.: sampled iterations x9) | 0.31 | 2.9% | 0.5 | 0.43 | `__syncthreads();` |
| wait.sync#19*9 (est.: sampled iterations x9) | 0.30 | 2.8% | 0.5 | 0.43 | `__syncthreads();` |
| wait.sync#14*8 (est.: sampled iterations x8) | 0.28 | 2.6% | 0.5 | 0.38 | `__syncthreads();` |
| wait.sync#6*8 (est.: sampled iterations x8) | 0.28 | 2.6% | 0.5 | 0.38 | `__syncthreads();` |
| wait.sync#13*8 (est.: sampled iterations x8) | 0.27 | 2.5% | 0.5 | 0.38 | `__syncthreads();` |
| wait.sync#5*8 (est.: sampled iterations x8) | 0.26 | 2.4% | 0.5 | 0.38 | `__syncthreads();` |
| wait.sync#23*9 (est.: sampled iterations x9) | 0.25 | 2.3% | 0.5 | 0.43 | `__syncthreads();` |
| store.global | 0.17 | 1.6% | 1.5 | 0.22 | `for (int i_3 = 0; i_3 < 4; ++i_3) \| if (((int)threadIdx.x) < total) \| if (((int)threadIdx.` |
| wait.sync#25 | 0.04 | 0.4% | 0.5 | 0.08 | `__syncthreads();` |
| wait.sync#17 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#8 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#16 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#2 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#18 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#9 | 0.03 | 0.3% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#3 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#15 | 0.03 | 0.3% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#7 | 0.03 | 0.3% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#21 | 0.03 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#22 | 0.03 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#11 | 0.02 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#10 | 0.02 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#4 | 0.02 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#12 | 0.02 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |

### topk_ids_kernel grid (5, 1, 1) block (512, 1, 1)

16 launches, 80 warps; ranges, warp and last warp from the 5 recording CTAs' 80 warps; span 21.22 us, warp 19.03 us, nsys graphs 20.72 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 15.42 | 81.0% | 15.73 | 74.1% |
| unattributed | 2.22 | 11.6% | 1.95 | 9.2% |
| wait | 0.93 | 4.9% | 0.64 | 3.0% |
| store | 0.46 | 2.4% | 2.91 | 13.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 15.42 | 81.0% | 1.0 | 16.48 | `for (int t = 0; t < (vocab >> 14); ++t)` |
| wait.sync | 0.80 | 4.2% | 1.0 | 1.49 | `__syncthreads();` |
| store.global | 0.46 | 2.4% | 1.0 | 2.98 | `if ((((int)threadIdx.x) >> 5) == 0)` |
| wait.sync#2 | 0.08 | 0.4% | 1.0 | 0.16 | `__syncthreads();` |
| wait.sync#3 | 0.06 | 0.3% | 1.0 | 0.06 | `__syncthreads();` |

### cache_writer_kernel grid (10, 1, 1) block (128, 1, 1)

51 launches, 40 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 6.94 us, warp 6.89 us, nsys graphs 4.93 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 4.91 | 71.2% | 3.22 | 49.4% |
| store | 1.61 | 23.4% | 2.73 | 41.8% |
| load | 0.30 | 4.3% | 0.47 | 7.3% |
| compute | 0.08 | 1.1% | 0.10 | 1.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 1.61 | 23.4% | 1.0 | 1.66 | `if (((int64_t)0 <= slot) & ((slot >> (int64_t)7) < ((int64_t)pages))) \| if (((int64_t)0 <=` |
| load.global | 0.24 | 3.5% | 1.0 | 0.26 | `for (int i = 0; i < 4; ++i)` |
| compute.reduce | 0.08 | 1.1% | 1.0 | 0.10 | `amax = max(amax, __shfl_xor_sync((uint)4294967295, amax, 1, 32));` |
| load.global#2 | 0.06 | 0.9% | 1.0 | 0.06 | `int64_t slot = slots[((int64_t)((int)blockIdx.x))];` |

### norm_forward_and_per_token_cast_kernel_kernel grid (3, 1, 1) block (128, 1, 1)

102 launches, 12 warps; ranges, warp and last warp from the 3 recording CTAs' 12 warps; span 2.75 us, warp 2.54 us, nsys graphs 1.92 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 1.61 | 63.6% | 2.19 | 67.4% |
| load | 0.35 | 13.6% | 0.35 | 10.6% |
| unattributed | 0.32 | 12.8% | 0.54 | 16.6% |
| wait | 0.14 | 5.5% | 0.05 | 1.6% |
| compute | 0.12 | 4.6% | 0.12 | 3.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*3 (est.: sampled iterations x3) | 1.15 | 45.4% | 1.0 | 1.15 | `for (int j_2 = 0; j_2 < 2; ++j_2)` |
| store.global*3 (est.: sampled iterations x3) | 0.28 | 11.0% | 1.0 | 0.38 | `for (int j_1 = 0; j_1 < 4; ++j_1)` |
| load.global*3 (est.: sampled iterations x3) | 0.18 | 6.9% | 1.0 | 0.19 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 6)) < num_tokens) && (((vi * 2) + (` |
| wait.sync | 0.14 | 5.5% | 1.0 | 0.26 | `__syncthreads();` |
| load.global#3*3 (est.: sampled iterations x3) | 0.14 | 5.4% | 1.0 | 0.19 | `if (((vi_1 * 2) + ((((int)threadIdx.x) & 63) >> 5)) < 5)` |
| compute.reduce | 0.12 | 4.6% | 1.0 | 0.13 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| store.global#5*3 (est.: sampled iterations x3) | 0.11 | 4.4% | 0.8 | 0.19 | `*(uint4*)(out + ((((((int64_t)vi_1) * (int64_t)512) + ((((int64_t)((int)threadIdx.x)) & (i` |
| store.global#2 | 0.04 | 1.7% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| load.global#2 | 0.03 | 1.3% | 1.0 | 0.06 | `*(float2*)(row_partial + 0) = *(float2*)(sum_shared + ((((int)threadIdx.x) >> 6) * 2)); \| ` |
| store.global#3 | 0.03 | 1.1% | 0.4 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |

### moe_grouped_fp8_fp4_gemm_kernel grid (40, 36, 1) block (128, 1, 1)

6 launches, 5760 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 52.02 us, warp 3.04 us, nsys graphs 9.47 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.54 | 50.5% | 0.19 | 60.0% |
| wait | 0.50 | 16.4% | 0.06 | 20.0% |
| load | 0.50 | 16.3% | 0.06 | 20.0% |
| compute | 0.44 | 14.4% | 0.00 | 0.0% |
| store | 0.07 | 2.4% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*5 (est.: sampled iterations x5) | 0.44 | 14.4% | 0.2 | 1.76 | `if (ko < 4)` |
| load#2*5 (est.: sampled iterations x5) | 0.22 | 7.1% | 0.2 | 0.96 | `if (ko < 3)` |
| load.global#3 | 0.11 | 3.7% | 0.2 | 0.45 | `if (((expert < 128) && (0 <= expert))) \| if (((expert < 384) && (0 <= expert)))` |
| load.global | 0.10 | 3.2% | 1.0 | 0.19 | `int expert = TileExpert[((int64_t)((int)blockIdx.y))];` |
| wait.mbarrier*5 (est.: sampled iterations x5) | 0.10 | 3.1% | 0.2 | 0.40 | `loaded[(ko % 3)].wait((ko / 3));` |
| wait.sync#5*5 (est.: sampled iterations x5) | 0.08 | 2.8% | 0.2 | 0.48 | `__syncthreads();` |
| wait.sync#4*5 (est.: sampled iterations x5) | 0.08 | 2.6% | 0.2 | 0.32 | `__syncthreads();` |
| wait.sync#6*5 (est.: sampled iterations x5) | 0.08 | 2.6% | 0.2 | 0.32 | `__syncthreads();` |
| store.global | 0.07 | 2.4% | 0.2 | 0.37 | `for (int i_2 = 0; i_2 < 16; ++i_2) \| for (int i_3 = 0; i_3 < 16; ++i_3) \| for (int i_4 = 0` |
| wait.cp_async*5 (est.: sampled iterations x5) | 0.06 | 2.0% | 0.2 | 0.32 | `tl::cp_async_wait<2>();` |
| wait.sync | 0.05 | 1.6% | 1.0 | 0.06 | `__syncthreads();` |
| load | 0.05 | 1.5% | 0.2 | 0.26 | `for (int s = 0; s < 2; ++s)` |
| wait.sync#2 | 0.04 | 1.2% | 0.2 | 0.19 | `__syncthreads();` |
| load.global#2 | 0.03 | 0.9% | 0.2 | 0.22 | `if (((int)threadIdx.x) < 32) \| if (((int)threadIdx.x) < 64)` |
| wait.sync#3 | 0.01 | 0.4% | 0.2 | 0.05 | `__syncthreads();` |

### compress_kernel grid (2, 1, 1) block (128, 1, 1)

34 launches, 8 warps; ranges, warp and last warp from the 2 recording CTAs' 8 warps; span 7.10 us, warp 4.36 us, nsys graphs 6.58 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 2.15 | 49.4% | 2.97 | 34.6% |
| load | 1.71 | 39.2% | 4.70 | 54.9% |
| store | 0.44 | 10.1% | 0.81 | 9.4% |
| compute | 0.06 | 1.4% | 0.09 | 1.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.04 | 24.0% | 0.6 | 1.79 | `if ((((0 <= counts[0]) & (0 < counts[1])) & (counts[1] <= 2)) & (((((int)blockIdx.x) * 4) ` |
| load.global#2*4 (est.: sampled iterations x4) | 0.60 | 13.8% | 0.6 | 1.34 | `*(float4*)(v1 + 0) = *(float4*)(values + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048)` |
| store.global*4 (est.: sampled iterations x4) | 0.16 | 3.7% | 0.6 | 0.26 | `*(uint2*)(out + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + ((((int64_t)((int)thre` |
| store.global#3 | 0.10 | 2.3% | 0.3 | 0.16 | `emitted_slots[((((int64_t)((int)blockIdx.x)) * (int64_t)4) + (((int64_t)((int)threadIdx.x)` |
| store.global#2*4 (est.: sampled iterations x4) | 0.07 | 1.7% | 0.3 | 0.13 | `if ((slot >> (int64_t)3) < ((int64_t)blocks))` |
| compute.reduce | 0.06 | 1.4% | 0.6 | 0.10 | `float total = tl::warp_reduce_sum(ss[0]);` |
| store.global#5 | 0.05 | 1.1% | 0.3 | 0.16 | `emitted_slots[((((int64_t)((int)blockIdx.x)) * (int64_t)4) + (((int64_t)((int)threadIdx.x)` |
| load.global#3*4 (est.: sampled iterations x4) | 0.04 | 0.9% | 0.3 | 0.06 | `*(float4*)(w + 0) = *(float4*)(weight + ((c_1 * 128) + ((((int)threadIdx.x) & 31) * 4)));` |
| store.global#3*4 (est.: sampled iterations x4) | 0.04 | 0.9% | 0.3 | 0.06 | `*(uint2*)(out + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + ((((int64_t)((int)thre` |
| load.global#4 | 0.01 | 0.3% | 0.3 | 0.03 | `if (valid)` |
| load.global#3 | 0.01 | 0.2% | 0.3 | 0.02 | `if (valid)` |
| store.global#2 | 0.01 | 0.2% | 0.3 | 0.02 | `emitted[((((int64_t)((int)blockIdx.x)) * (int64_t)4) + (((int64_t)((int)threadIdx.x)) >> (` |
| store.global#4 | 0.01 | 0.2% | 0.3 | 0.02 | `emitted[((((int64_t)((int)blockIdx.x)) * (int64_t)4) + (((int64_t)((int)threadIdx.x)) >> (` |

### topk_log_softmax_kernel grid (1, 1, 1) block (1024, 1, 1)

17 launches, 32 warps; ranges, warp and last warp from the 1 recording CTAs' 32 warps; span 16.54 us, warp 15.95 us, nsys graphs 15.68 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 12.90 | 80.9% | 13.81 | 82.0% |
| unattributed, with loops' unsampled iterations | 1.53 | 9.6% | 1.74 | 10.3% |
| wait | 1.13 | 7.1% | 0.83 | 5.0% |
| compute | 0.39 | 2.4% | 0.38 | 2.3% |
| store | 0.00 | 0.0% | 0.07 | 0.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 6.57 | 41.2% | 1.0 | 6.82 | `if (0 < (vocab & 2047)) \| if (0 < (vocab & 8191))` |
| load.global#2 | 6.32 | 39.6% | 1.0 | 6.78 | `for (int step_1 = 0; step_1 < (vocab >> 11); ++step_1) \| for (int step_1 = 0; step_1 < (vo` |
| wait.sync#2 | 0.57 | 3.6% | 1.0 | 0.93 | `__syncthreads();` |
| wait.sync | 0.31 | 1.9% | 1.0 | 0.96 | `__syncthreads();` |
| compute.reduce#3 | 0.18 | 1.1% | 1.0 | 0.19 | `ties[0] = vllm_add(ties[0], __shfl_xor_sync((uint)4294967295, ties[0], 16, 32));` |
| compute.reduce#2 | 0.11 | 0.7% | 1.0 | 0.16 | `se[0] = vllm_add(se[0], __shfl_xor_sync((uint)4294967295, se[0], 16, 32));` |
| compute.reduce | 0.10 | 0.6% | 1.0 | 0.13 | `best[0] = vllm_max(best[0], __shfl_xor_sync((uint)4294967295, best[0], 16, 32));` |
| wait.sync#4 | 0.10 | 0.6% | 1.0 | 0.16 | `__syncthreads();` |
| wait.sync#5 | 0.08 | 0.5% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#3 | 0.08 | 0.5% | 1.0 | 0.10 | `__syncthreads();` |
| load.global#3 (sampled iterations) | 0.01 | 0.1% | 0.0 | 0.38 | `float value_32 = vllm_sub(vllm_shifted(logits[((((int64_t)((int)blockIdx.x)) * logits_stri` |
| store.global (sampled iterations) | 0.00 | 0.0% | 0.0 | 0.06 | `output[(((((int64_t)c) * (int64_t)1024) + (((int64_t)((int)blockIdx.x)) * ((int64_t)topk))` |

### topk_log_softmax_kernel grid (5, 1, 1) block (1024, 1, 1)

16 launches, 160 warps; ranges, warp and last warp from the 5 recording CTAs' 160 warps; span 17.02 us, warp 16.23 us, nsys graphs 16.32 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 13.24 | 81.6% | 14.21 | 82.6% |
| unattributed, with loops' unsampled iterations | 1.58 | 9.8% | 1.76 | 10.2% |
| wait | 1.01 | 6.2% | 0.78 | 4.5% |
| compute | 0.40 | 2.5% | 0.38 | 2.2% |
| store | 0.00 | 0.0% | 0.07 | 0.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 6.70 | 41.3% | 1.0 | 7.04 | `if (0 < (vocab & 2047)) \| if (0 < (vocab & 8191))` |
| load.global#2 | 6.52 | 40.2% | 1.0 | 7.10 | `for (int step_1 = 0; step_1 < (vocab >> 11); ++step_1) \| for (int step_1 = 0; step_1 < (vo` |
| wait.sync#2 | 0.46 | 2.8% | 1.0 | 1.06 | `__syncthreads();` |
| wait.sync | 0.29 | 1.8% | 1.0 | 0.80 | `__syncthreads();` |
| compute.reduce#3 | 0.18 | 1.1% | 1.0 | 0.22 | `ties[0] = vllm_add(ties[0], __shfl_xor_sync((uint)4294967295, ties[0], 16, 32));` |
| compute.reduce | 0.11 | 0.7% | 1.0 | 0.16 | `best[0] = vllm_max(best[0], __shfl_xor_sync((uint)4294967295, best[0], 16, 32));` |
| compute.reduce#2 | 0.11 | 0.7% | 1.0 | 0.16 | `se[0] = vllm_add(se[0], __shfl_xor_sync((uint)4294967295, se[0], 16, 32));` |
| wait.sync#4 | 0.10 | 0.6% | 1.0 | 0.16 | `__syncthreads();` |
| wait.sync#5 | 0.08 | 0.5% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#3 | 0.08 | 0.5% | 1.0 | 0.10 | `__syncthreads();` |
| load.global#3 (sampled iterations) | 0.01 | 0.1% | 0.0 | 0.42 | `float value_32 = vllm_sub(vllm_shifted(logits[((((int64_t)((int)blockIdx.x)) * logits_stri` |
| store.global (sampled iterations) | 0.00 | 0.0% | 0.0 | 0.06 | `output[(((((int64_t)c) * (int64_t)1024) + (((int64_t)((int)blockIdx.x)) * ((int64_t)topk))` |

### moe_combine_kernel grid (5, 5, 1) block (128, 1, 1)

96 launches, 100 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 2.24 us, warp 1.71 us, nsys graphs 2.43 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.25 | 73.2% | 1.78 | 78.3% |
| unattributed | 0.40 | 23.5% | 0.44 | 19.4% |
| store | 0.06 | 3.3% | 0.05 | 2.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.25 | 73.2% | 1.0 | 1.76 | `for (int k = 0; k < 3; ++k) \| for (int k = 0; k < 6; ++k) \| if ((((int)blockIdx.y) < share` |
| store.global | 0.06 | 3.3% | 1.0 | 0.06 | `{` |

### norm_forward_and_per_token_cast_kernel_kernel grid (5, 1, 1) block (128, 1, 1)

67 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 2.82 us, warp 2.56 us, nsys graphs 1.92 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 1.64 | 64.0% | 2.57 | 80.6% |
| unattributed | 0.37 | 14.6% | 0.08 | 2.5% |
| load | 0.32 | 12.5% | 0.37 | 11.5% |
| compute | 0.12 | 4.6% | 0.12 | 3.9% |
| wait | 0.11 | 4.3% | 0.05 | 1.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*3 (est.: sampled iterations x3) | 1.15 | 44.8% | 1.0 | 1.15 | `for (int j_2 = 0; j_2 < 2; ++j_2)` |
| store.global*3 (est.: sampled iterations x3) | 0.30 | 11.6% | 1.0 | 0.38 | `for (int j_1 = 0; j_1 < 4; ++j_1)` |
| load.global*3 (est.: sampled iterations x3) | 0.17 | 6.7% | 1.0 | 0.19 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 6)) < num_tokens) && (((vi * 2) + (` |
| store.global#5*3 (est.: sampled iterations x3) | 0.12 | 4.7% | 1.0 | 0.19 | `*(uint4*)(out + ((((((int64_t)vi_1) * (int64_t)512) + ((((int64_t)((int)threadIdx.x)) & (i` |
| compute.reduce | 0.12 | 4.6% | 1.0 | 0.13 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| load.global#3*3 (est.: sampled iterations x3) | 0.12 | 4.5% | 1.0 | 0.19 | `if (((vi_1 * 2) + ((((int)threadIdx.x) & 63) >> 5)) < 5)` |
| wait.sync | 0.11 | 4.3% | 1.0 | 0.19 | `__syncthreads();` |
| store.global#2 | 0.04 | 1.6% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| load.global#2 | 0.03 | 1.2% | 1.0 | 0.03 | `*(float2*)(row_partial + 0) = *(float2*)(sum_shared + ((((int)threadIdx.x) >> 6) * 2)); \| ` |
| store.global#3 | 0.03 | 1.2% | 0.5 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |
| load.global#2*5 (est.: sampled iterations x5) | 0.00 | 0.0% | 0.0 | 0.00 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 1024) + (((int)threadIdx.x) * 8)` |
| load.global*5 (est.: sampled iterations x5) | 0.00 | 0.0% | 0.0 | 0.00 | `*(uint4*)(x_local + (vi * 8)) = *(uint4*)(x + ((((((int64_t)vi) * (int64_t)1024) + (((int6` |
| store.global#4*5 (est.: sampled iterations x5) | 0.00 | 0.0% | 0.0 | 0.00 | `for (int j_1 = 0; j_1 < 2; ++j_1)` |
| store.global*5 (est.: sampled iterations x5) | 0.00 | 0.0% | 0.0 | 0.00 | `for (int j = 0; j < 4; ++j)` |

### per_token_cast_kernel_kernel grid (9, 1, 1) block (64, 1, 1)

153 launches, 18 warps; ranges, warp and last warp from the 5 recording CTAs' 10 warps; span 1.02 us, warp 0.92 us, nsys graphs 1.22 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.45 | 49.3% | 0.59 | 42.4% |
| load | 0.26 | 28.0% | 0.36 | 26.1% |
| store | 0.11 | 12.2% | 0.24 | 17.2% |
| compute | 0.10 | 10.5% | 0.20 | 14.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.17 | 18.2% | 1.0 | 0.26 | `for (int i = 0; i < 2; ++i)` |
| store.global#2 | 0.11 | 12.2% | 1.0 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.10 | 10.5% | 1.0 | 0.13 | `amax_fragment[i_1] = tl::AllReduce<tl::MaxOp, 2, 1, 0, tl::NamedBarrier<64>>::run(amax_fra` |
| load.global#2*2 (est.: sampled iterations x2) | 0.09 | 9.8% | 1.0 | 0.13 | `amax_fragment[i_1] = cutlass::bfloat16_t(__hmax((bfloat16_t(((nv_bfloat162*)(&(amax_fragme` |

### local_stats_kernel grid (5, 16, 1) block (128, 1, 1)

16 launches, 320 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 12.54 us, warp 11.68 us, nsys graphs 12.27 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 10.57 | 90.5% | 11.50 | 91.2% |
| unattributed | 0.73 | 6.2% | 0.70 | 5.5% |
| compute | 0.17 | 1.5% | 0.18 | 1.4% |
| wait | 0.15 | 1.2% | 0.15 | 1.2% |
| store | 0.07 | 0.6% | 0.09 | 0.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 10.57 | 90.5% | 1.0 | 10.99 | `if (expanded_local_pos[((int64_t)((int)blockIdx.x))] < num_speculative_steps)` |
| wait.sync | 0.15 | 1.2% | 1.0 | 0.75 | `__syncthreads();` |
| compute.reduce | 0.12 | 1.0% | 1.0 | 0.13 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| store.global | 0.07 | 0.6% | 1.0 | 0.11 | `if (((int)threadIdx.x) == 0)` |
| compute.reduce#2 | 0.06 | 0.5% | 1.0 | 0.06 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 2, 32);` |

### compress_gather_partial_state_kernel grid (2, 1, 1) block (128, 1, 1)

17 launches, 8 warps; ranges, warp and last warp from the 2 recording CTAs' 8 warps; span 12.35 us, warp 9.18 us, nsys graphs 7.34 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 4.76 | 51.8% | 5.46 | 45.8% |
| unattributed | 3.40 | 37.0% | 2.84 | 23.8% |
| store | 1.03 | 11.2% | 3.63 | 30.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 4.64 | 50.5% | 1.0 | 6.27 | `int nt = counts[0];` |
| store.global*2 (est.: sampled iterations x2) | 0.97 | 10.5% | 1.0 | 2.62 | `if (half == 0)` |
| load.global#2*2 (est.: sampled iterations x2) | 0.12 | 1.3% | 1.0 | 0.13 | `for (int i = 0; i < 4; ++i)` |
| store.global#2 | 0.06 | 0.7% | 1.0 | 0.06 | `if (((int)threadIdx.x) == 0)` |

### mhc_post_kernel grid (25, 1, 1) block (128, 1, 1)

64 launches, 100 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 2.18 us, warp 1.67 us, nsys graphs 2.21 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.52 | 90.9% | 2.69 | 94.4% |
| load | 0.10 | 6.2% | 0.11 | 3.7% |
| store | 0.05 | 2.9% | 0.05 | 1.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.10 | 6.2% | 1.0 | 0.13 | `*(float4*)(pm + 0) = *(float4*)(post + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int` |
| store.global | 0.05 | 2.9% | 1.0 | 0.06 | `for (int n_1 = 0; n_1 < 4; ++n_1)` |

### metadata_chunk_kernel grid (5, 1, 1) block (128, 1, 1)

32 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 4.16 us, warp 3.49 us, nsys graphs 6.53 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 2.19 | 62.8% | 2.22 | 36.6% |
| load | 1.22 | 35.1% | 3.69 | 61.1% |
| store | 0.07 | 2.1% | 0.14 | 2.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.12 | 32.2% | 1.0 | 1.31 | `if ((((offset + ((int)blockIdx.x)) < src_rows) && (0 <= (offset + ((int)blockIdx.x)))))` |
| load.global#2 | 0.10 | 2.9% | 0.2 | 0.50 | `if (((0 <= req) & ((int64_t)0 <= pos)))` |
| store.global | 0.05 | 1.5% | 1.0 | 0.06 | `swa[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = con` |
| store.global#2 | 0.01 | 0.3% | 0.2 | 0.06 | `lengths[((int64_t)((int)blockIdx.x))] = ((int)max(info[1], (int64_t)0));` |
| store.global#3 | 0.01 | 0.2% | 0.2 | 0.03 | `top_lengths[((int64_t)((int)blockIdx.x))] = condval_12; \| top_lengths[((int64_t)((int)bloc` |

### sparse_mla_kernel_1 grid (16, 5, 1) block (128, 1, 1)

96 launches, 80 CTAs, 320 warps; span 1.79 us, warp 1.36 us, nsys graphs 1.95 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.31 | 96.3% | 0.74 | 40.3% |
| load | 0.03 | 2.5% | 0.84 | 45.8% |
| wait | 0.01 | 0.8% | 0.13 | 7.0% |
| compute | 0.00 | 0.3% | 0.08 | 4.5% |
| store | 0.00 | 0.1% | 0.04 | 2.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.03 | 2.5% | 0.1 | 0.86 | `M = __fmul_rn(Sinks[((int)blockIdx.x)], 0x1.7154764ee6c2fp+0f/*1.442695e+00*/); \| if (((in` |
| wait.sync | 0.01 | 0.8% | 0.1 | 0.34 | `__syncthreads();` |
| compute.reduce | 0.00 | 0.3% | 0.1 | 0.10 | `amax[0] = tl::AllReduce<tl::MaxOp, 8, 1, 0, tl::NamedBarrier<128>>::run(amax[0]);` |
| store.global | 0.00 | 0.1% | 0.1 | 0.06 | `*(uint2*)(Output + (((((int64_t)((int)blockIdx.y)) * (int64_t)8192) + (((int64_t)((int)blo` |
| compute.reduce#3 | 0.00 | 0.0% | 0.0 | 0.00 | `amax[0] = tl::AllReduce<tl::MaxOp, 8, 1, 0, tl::NamedBarrier<128>>::run(amax[0]);` |
| load.global#3 | 0.00 | 0.0% | 0.0 | 0.00 | `M = __fmul_rn(Sinks[((int)blockIdx.x)], 0x1.7154764ee6c2fp+0f/*1.442695e+00*/);` |
| load.global#4 | 0.00 | 0.0% | 0.0 | 0.00 | `if (((int)threadIdx.x) < 32)` |
| store.global#2 | 0.00 | 0.0% | 0.0 | 0.00 | `*(uint2*)(Output + (((((int64_t)((int)blockIdx.y)) * (int64_t)8192) + (((int64_t)((int)blo` |
| wait.sync#12 | 0.00 | 0.0% | 0.0 | 0.00 | `__syncthreads();` |
| wait.sync#13 | 0.00 | 0.0% | 0.0 | 0.00 | `__syncthreads();` |

### per_token_cast_kernel_kernel grid (5, 1, 1) block (64, 1, 1)

136 launches, 10 warps; ranges, warp and last warp from the 5 recording CTAs' 10 warps; span 1.18 us, warp 1.08 us, nsys graphs 1.06 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.44 | 40.7% | 0.52 | 40.9% |
| load | 0.34 | 31.7% | 0.35 | 27.5% |
| store | 0.20 | 18.7% | 0.30 | 23.7% |
| compute | 0.10 | 8.9% | 0.10 | 7.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.25 | 23.4% | 1.0 | 0.26 | `for (int i = 0; i < 2; ++i)` |
| store.global#2 | 0.11 | 10.4% | 1.0 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.10 | 8.9% | 1.0 | 0.13 | `amax_fragment[i_1] = tl::AllReduce<tl::MaxOp, 2, 1, 0, tl::NamedBarrier<64>>::run(amax_fra` |
| load.global#2*2 (est.: sampled iterations x2) | 0.09 | 8.3% | 1.0 | 0.13 | `amax_fragment[i_1] = cutlass::bfloat16_t(__hmax((bfloat16_t(((nv_bfloat162*)(&(amax_fragme` |
| store.global#3*2 (est.: sampled iterations x2) | 0.04 | 4.1% | 0.5 | 0.13 | `*(fp8_e4_16_t*)(out + ((((((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int64_t)10240) +` |
| store.global*2 (est.: sampled iterations x2) | 0.04 | 4.1% | 0.5 | 0.13 | `out_sf[((((((int64_t)((int)blockIdx.x)) % (int64_t)15) * (int64_t)32) + (((int64_t)((int)t` |

### mhc_finalize_kernel grid (6, 1, 1) block (128, 1, 1)

12 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 13.89 us, warp 11.54 us, nsys graphs 8.10 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 6.58 | 57.1% | 7.90 | 55.4% |
| store | 1.77 | 15.3% | 2.37 | 16.6% |
| load | 1.54 | 13.4% | 1.66 | 11.7% |
| compute | 0.99 | 8.6% | 1.00 | 7.0% |
| wait | 0.64 | 5.6% | 1.33 | 9.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.89 | 7.7% | 0.2 | 3.57 | `if (((int)threadIdx.x) < 4)` |
| load.global | 0.85 | 7.3% | 1.0 | 1.90 | `if (((int)threadIdx.x) < 32)` |
| wait.sync | 0.64 | 5.6% | 1.0 | 0.86 | `__syncthreads();` |
| compute.reduce | 0.57 | 4.9% | 0.2 | 2.29 | `row_max[0] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<32>>::run(row_max[0]);` |
| store.global#4*7 (est.: sampled iterations x7) | 0.55 | 4.8% | 0.8 | 0.90 | `if ((((int)threadIdx.x) % 4) == 0)` |
| compute.reduce#2*7 (est.: sampled iterations x7) | 0.42 | 3.7% | 0.8 | 0.67 | `amax[0] = max(amax[0], __shfl_xor_sync((uint)4294967295, amax[0], 1, 32));` |
| load.global#4*7 (est.: sampled iterations x7) | 0.37 | 3.2% | 0.8 | 0.67 | `*(uint4*)(yv + 0) = *(uint4*)(C + ((((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((i` |
| store.global#3*7 (est.: sampled iterations x7) | 0.32 | 2.8% | 0.8 | 0.45 | `*(uint4*)(y + ((((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((int64_t)step) * (int6` |
| load.global#3 | 0.28 | 2.5% | 0.8 | 0.38 | `for (int i_split_2 = 0; i_split_2 < 40; ++i_split_2)` |
| load.global#2 | 0.05 | 0.4% | 0.2 | 0.19 | `cm[0] = ((mixes_shared[((((int)threadIdx.x) & 15) + 8)] * scale[2]) + base[((((int)threadI` |
| store.global#2 | 0.01 | 0.1% | 0.2 | 0.06 | `if ((((int)threadIdx.x) >> 4) == 0)` |

### indexer_topk_segment_kernel grid (2, 5, 1) block (512, 1, 1)

16 launches, 160 warps; ranges, warp and last warp from the 5 recording CTAs' 80 warps; span 10.11 us, warp 9.91 us, nsys graphs 4.54 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 6.21 | 62.6% | 6.13 | 59.4% |
| unattributed, with loops' unsampled iterations | 3.14 | 31.7% | 3.28 | 31.8% |
| load | 0.47 | 4.8% | 0.79 | 7.7% |
| store | 0.09 | 0.9% | 0.12 | 1.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync#24*9 (est.: sampled iterations x9) | 0.68 | 6.9% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#20*9 (est.: sampled iterations x9) | 0.68 | 6.8% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#19*9 (est.: sampled iterations x9) | 0.64 | 6.5% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#23*9 (est.: sampled iterations x9) | 0.64 | 6.5% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#14*8 (est.: sampled iterations x8) | 0.60 | 6.0% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync#5*8 (est.: sampled iterations x8) | 0.59 | 5.9% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync#6*8 (est.: sampled iterations x8) | 0.59 | 5.9% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync#13*8 (est.: sampled iterations x8) | 0.57 | 5.8% | 1.0 | 0.77 | `__syncthreads();` |
| load.global#2 | 0.26 | 2.6% | 1.0 | 0.51 | `for (int i = 0; i < 8; ++i)` |
| load.global | 0.21 | 2.1% | 1.0 | 0.22 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 4096); \| i` |
| wait.sync (sampled iterations) | 0.14 | 1.4% | 1.0 | 0.35 | `__syncthreads();` |
| wait.sync#25 | 0.10 | 1.0% | 1.0 | 0.16 | `__syncthreads();` |
| store.global | 0.09 | 0.9% | 1.0 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3) \| if (((int)threadIdx.x) < total) \| if (((int)threadIdx.` |
| wait.sync#18 | 0.08 | 0.8% | 1.0 | 0.16 | `__syncthreads();` |
| wait.sync#17 | 0.08 | 0.8% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#8 | 0.08 | 0.8% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#16 | 0.07 | 0.8% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#9 | 0.07 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#3 | 0.07 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#2 | 0.07 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#11 | 0.06 | 0.6% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#15 | 0.06 | 0.6% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#12 | 0.06 | 0.6% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#7 | 0.06 | 0.6% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#10 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#21 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#22 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#4 | 0.06 | 0.6% | 1.0 | 0.08 | `__syncthreads();` |

### mhc_collapse_kernel grid (25, 1, 1) block (128, 1, 1)

48 launches, 100 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 2.74 us, warp 2.71 us, nsys graphs 2.66 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 2.57 | 94.7% | 3.19 | 94.1% |
| load | 0.10 | 3.5% | 0.14 | 4.2% |
| store | 0.05 | 1.7% | 0.05 | 1.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.10 | 3.5% | 1.0 | 0.13 | `*(float4*)(mv + 0) = *(float4*)(mix + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int6` |
| store.global | 0.05 | 1.7% | 1.0 | 0.06 | `*(uint4*)(out + ((((int64_t)((int)blockIdx.x)) * (int64_t)1024) + (((int64_t)((int)threadI` |

### router_partials_kernel grid (5, 1, 1) block (64, 1, 1)

51 launches, 10 warps; ranges, warp and last warp from the 5 recording CTAs' 10 warps; span 2.75 us, warp 2.03 us, nsys graphs 2.40 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.84 | 41.7% | 1.15 | 37.7% |
| load | 0.82 | 40.6% | 1.34 | 44.0% |
| store | 0.19 | 9.2% | 0.21 | 7.0% |
| compute | 0.17 | 8.5% | 0.34 | 11.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.45 | 22.1% | 1.0 | 0.45 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < mask_len))` |
| load.global#4 | 0.25 | 12.2% | 0.5 | 0.54 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < image_len)) \| if ((((((int)bl` |
| compute.reduce | 0.17 | 8.5% | 0.5 | 0.35 | `for (int k = 0; k < 3; ++k) \| for (int k = 0; k < 6; ++k)` |
| load.global#2 | 0.06 | 3.2% | 1.0 | 0.06 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < mask_len))` |
| load.global#3 | 0.06 | 3.2% | 1.0 | 0.06 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < mask_len))` |
| store.global | 0.06 | 3.2% | 1.0 | 0.10 | `if (condval == (bool)0) \| if (condval == (bool)1)` |
| store.global#2 | 0.06 | 3.2% | 1.0 | 0.10 | `if (condval_1 == (bool)0) \| if (condval_1 == (bool)1)` |
| store.global#3 | 0.06 | 2.8% | 0.5 | 0.13 | `if ((((int)threadIdx.x) & 31) < 3) \| if ((((int)threadIdx.x) & 31) < 6)` |

### indexer_topk_merge_kernel grid (1, 5, 1) block (512, 1, 1)

16 launches, 80 warps; ranges, warp and last warp from the 5 recording CTAs' 80 warps; span 9.28 us, warp 9.03 us, nsys graphs 3.87 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 6.14 | 67.9% | 6.09 | 63.6% |
| unattributed | 2.24 | 24.8% | 2.32 | 24.3% |
| load | 0.58 | 6.4% | 1.01 | 10.6% |
| store | 0.08 | 0.9% | 0.15 | 1.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync#24*9 (est.: sampled iterations x9) | 0.68 | 7.6% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#20*9 (est.: sampled iterations x9) | 0.68 | 7.5% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#23*9 (est.: sampled iterations x9) | 0.65 | 7.2% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#19*9 (est.: sampled iterations x9) | 0.64 | 7.1% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#6*8 (est.: sampled iterations x8) | 0.59 | 6.6% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync#14*8 (est.: sampled iterations x8) | 0.59 | 6.5% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync#5*8 (est.: sampled iterations x8) | 0.58 | 6.4% | 1.0 | 0.77 | `__syncthreads();` |
| load.global | 0.57 | 6.3% | 1.0 | 0.70 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 8192);` |
| wait.sync#13*8 (est.: sampled iterations x8) | 0.56 | 6.3% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync | 0.15 | 1.6% | 1.0 | 0.22 | `__syncthreads();` |
| wait.sync#17 | 0.08 | 0.9% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#16 | 0.08 | 0.9% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#8 | 0.07 | 0.8% | 1.0 | 0.10 | `__syncthreads();` |
| store.global#2 | 0.07 | 0.8% | 0.9 | 0.10 | `Values[((((int64_t)((int)blockIdx.y)) * (int64_t)512) + ((int64_t)((int)threadIdx.x)))] = ` |
| wait.sync#2 | 0.07 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#3 | 0.07 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#9 | 0.07 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#25 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#15 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#7 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#12 | 0.06 | 0.7% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#11 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#21 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#18 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#4 | 0.06 | 0.6% | 1.0 | 0.08 | `__syncthreads();` |
| wait.sync#10 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#22 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| store.global | 0.01 | 0.1% | 0.1 | 0.13 | `Values[((((int64_t)((int)blockIdx.y)) * (int64_t)512) + ((int64_t)((int)threadIdx.x)))] = ` |
| load.global#2 | 0.01 | 0.1% | 0.1 | 0.10 | `if (((0 <= ((int*)sel_idx)[((int)threadIdx.x)]) && (((int*)sel_idx)[((int)threadIdx.x)] < ` |

### rope_kernel grid (5, 1, 1) block (128, 1, 1)

96 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 1.31 us, warp 1.27 us, nsys graphs 1.47 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.74 | 58.6% | 0.95 | 65.5% |
| load | 0.49 | 38.3% | 0.46 | 31.7% |
| store | 0.04 | 3.0% | 0.04 | 2.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.49 | 38.3% | 1.0 | 0.51 | `if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < tokens)) \| int64_t position ` |
| store.global | 0.04 | 3.0% | 1.0 | 0.06 | `*(uint4*)(x + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + ((((int64_t)((int)thread` |

### norm_forward_and_per_token_cast_kernel_kernel grid (2, 1, 1) block (128, 1, 1)

51 launches, 8 warps; ranges, warp and last warp from the 2 recording CTAs' 8 warps; span 2.72 us, warp 2.72 us, nsys graphs 1.34 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 2.05 | 75.4% | 1.95 | 74.5% |
| load | 0.43 | 15.9% | 0.44 | 16.7% |
| wait | 0.14 | 5.1% | 0.05 | 2.0% |
| compute | 0.11 | 4.0% | 0.11 | 4.2% |
| unattributed | 0.00 | 0.0% | 0.07 | 2.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*4 (est.: sampled iterations x4) | 1.54 | 56.5% | 1.0 | 1.54 | `for (int j_2 = 0; j_2 < 2; ++j_2)` |
| store.global*4 (est.: sampled iterations x4) | 0.32 | 11.8% | 1.0 | 0.51 | `for (int j_1 = 0; j_1 < 4; ++j_1)` |
| load.global*4 (est.: sampled iterations x4) | 0.22 | 8.2% | 1.0 | 0.26 | `if (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 4)) < num_tokens)` |
| load.global#3*4 (est.: sampled iterations x4) | 0.18 | 6.5% | 1.0 | 0.26 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 128) + ((((int)threadIdx.x) & 15` |
| wait.sync | 0.14 | 5.1% | 1.0 | 0.29 | `__syncthreads();` |
| store.global#5*4 (est.: sampled iterations x4) | 0.11 | 4.1% | 0.6 | 0.26 | `*(uint4*)(out + ((((((int64_t)vi_1) * (int64_t)128) + ((((int64_t)((int)threadIdx.x)) & (i` |
| compute.reduce | 0.11 | 4.0% | 1.0 | 0.13 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| store.global#2 | 0.05 | 1.8% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| store.global#3 | 0.04 | 1.3% | 0.6 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |
| load.global#2 | 0.03 | 1.2% | 1.0 | 0.03 | `*(float2*)(row_partial + 0) = *(float2*)(sum_shared + ((((int)threadIdx.x) >> 6) * 2)); \| ` |

### sparse_mla_kernel_1 grid (16, 10, 1) block (128, 1, 1)

51 launches, 160 CTAs, 640 warps; span 1.76 us, warp 1.33 us, nsys graphs 4.99 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.30 | 98.1% | 1.37 | 54.2% |
| load | 0.02 | 1.2% | 0.93 | 37.0% |
| wait | 0.01 | 0.4% | 0.11 | 4.4% |
| compute | 0.00 | 0.2% | 0.07 | 2.9% |
| store | 0.00 | 0.1% | 0.04 | 1.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#3 | 0.01 | 1.0% | 0.0 | 0.64 | `M = __fmul_rn(Sinks[((int)blockIdx.x)], 0x1.7154764ee6c2fp+0f/*1.442695e+00*/);` |
| wait.sync#13 | 0.00 | 0.3% | 0.0 | 0.22 | `__syncthreads();` |
| load.global#4 | 0.00 | 0.2% | 0.0 | 0.22 | `if (((int)threadIdx.x) < 32)` |
| compute.reduce#3 | 0.00 | 0.2% | 0.0 | 0.10 | `amax[0] = tl::AllReduce<tl::MaxOp, 8, 1, 0, tl::NamedBarrier<128>>::run(amax[0]);` |
| wait.sync#12 | 0.00 | 0.1% | 0.0 | 0.13 | `__syncthreads();` |
| store.global#2 | 0.00 | 0.1% | 0.0 | 0.06 | `*(uint2*)(Output + (((((int64_t)((int)blockIdx.y)) * (int64_t)8192) + (((int64_t)((int)blo` |

### prepare_dflash_inputs_kernel grid (9, 1, 1) block (256, 1, 1)

51 launches, 72 warps; ranges, warp and last warp from the 5 recording CTAs' 40 warps; span 2.02 us, warp 0.74 us, nsys graphs 2.50 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.26 | 34.6% | 1.70 | 67.1% |
| unattributed | 0.25 | 33.5% | 0.47 | 18.7% |
| store | 0.24 | 31.8% | 0.36 | 14.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.24 | 33.0% | 0.2 | 1.34 | `req = idx_mapping[((int)blockIdx.y)];` |
| store.global | 0.19 | 25.2% | 0.8 | 0.26 | `if (((int)blockIdx.y) == (num_reqs - 1))` |
| store.global#2 | 0.05 | 6.6% | 0.2 | 0.29 | `if (((int)blockIdx.y) == (num_reqs - 1))` |
| load.global#2 | 0.01 | 1.6% | 0.2 | 0.06 | `vllm_store_if_b32((&(out_query_start_loc[((int)blockIdx.y)])), (((int)blockIdx.y) * num_qu` |

### token_ranks_kernel grid (5, 1, 1) block (256, 1, 1)

16 launches, 40 warps; ranges, warp and last warp from the 5 recording CTAs' 40 warps; span 7.81 us, warp 7.73 us, nsys graphs 8.77 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 6.95 | 89.9% | 7.14 | 90.3% |
| unattributed | 0.39 | 5.0% | 0.38 | 4.8% |
| compute | 0.35 | 4.5% | 0.34 | 4.3% |
| store | 0.04 | 0.6% | 0.05 | 0.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 6.95 | 89.9% | 1.0 | 7.23 | `if ((((int64_t)0 <= token_ids[((int64_t)((int)blockIdx.x))]) && (token_ids[((int64_t)((int` |
| compute.reduce | 0.35 | 4.5% | 1.0 | 0.70 | `total[0] = tl::AllReduce<tl::SumOp, 256, 1, 0, tl::NamedBarrier<256>>::run(total[0], (&(wo` |
| store.global | 0.04 | 0.6% | 1.0 | 0.06 | `output[((int64_t)((int)blockIdx.x))] = ((int64_t)total[0]);` |

### gemm_splitk_reduce_cast_kernel grid (16, 2, 1) block (128, 1, 1)

102 launches, 128 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 0.99 us, warp 0.92 us, nsys graphs 1.09 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.42 | 45.9% | 0.52 | 42.7% |
| load | 0.35 | 37.9% | 0.54 | 44.1% |
| compute | 0.08 | 8.5% | 0.08 | 6.5% |
| store | 0.07 | 7.7% | 0.08 | 6.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.35 | 37.9% | 1.0 | 0.38 | `if ((1 <= M))` |
| compute.reduce | 0.08 | 8.5% | 1.0 | 0.10 | `amax[0] = tl::AllReduce<tl::MaxOp, 8, 1, 0, tl::NamedBarrier<128>>::run(amax[0]);` |
| store.global | 0.07 | 7.7% | 1.0 | 0.10 | `if (((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 5)) < M)` |

### indexer_mxfp4_quantize_kernel grid (5, 1, 1) block (128, 1, 1)

32 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 3.78 us, warp 3.70 us, nsys graphs 2.93 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 2.31 | 62.4% | 2.35 | 62.1% |
| store | 1.14 | 30.8% | 1.12 | 29.6% |
| unattributed | 0.25 | 6.8% | 0.31 | 8.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2*16 (est.: sampled iterations x16) | 1.28 | 34.6% | 1.0 | 1.54 | `float v__2 = __fmul_rn(((float)X[(((((int64_t)((int)blockIdx.x)) * (int64_t)4096) + (((int` |
| store.global*16 (est.: sampled iterations x16) | 1.10 | 29.7% | 1.0 | 1.54 | `Pool[(((((slot & (int64_t)127) * (int64_t)64) + ((((int64_t)((int)threadIdx.x)) & (int64_t` |
| load.global | 1.03 | 27.8% | 1.0 | 1.09 | `for (int i = 0; i < 32; ++i)` |
| store.global#2 | 0.04 | 1.1% | 1.0 | 0.06 | `Pool[(((((slot & (int64_t)127) * (int64_t)4) + ((slot >> (int64_t)7) * ((int64_t)pool_stri` |

### activation_swiglu_clamp_kernel grid (1, 5, 1) block (128, 1, 1)

96 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 0.93 us, warp 0.73 us, nsys graphs 1.86 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.62 | 85.5% | 0.96 | 85.6% |
| load | 0.07 | 9.9% | 0.11 | 10.0% |
| store | 0.03 | 4.6% | 0.05 | 4.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.07 | 9.9% | 0.8 | 0.10 | `*(uint4*)(gate + 0) = *(uint4*)(X + ((((int64_t)((int)blockIdx.y)) * (int64_t)1152) + (((i` |
| store.global | 0.03 | 4.6% | 0.8 | 0.06 | `*(uint4*)(Y + ((((int64_t)((int)blockIdx.y)) * (int64_t)576) + (((int64_t)((int)threadIdx.` |

### indexer_mxfp4_quantize_kernel grid (1, 1, 1) block (128, 1, 1)

34 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 2.99 us, warp 1.02 us, nsys graphs 1.98 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.39 | 38.6% | 0.42 | 13.5% |
| load | 0.36 | 35.4% | 1.50 | 48.1% |
| store | 0.26 | 26.0% | 1.20 | 38.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2*16 (est.: sampled iterations x16) | 0.26 | 25.2% | 0.2 | 1.02 | `float v__2 = __fmul_rn(((float)X[(((((int64_t)((int)blockIdx.x)) * (int64_t)4096) + (((int` |
| store.global*16 (est.: sampled iterations x16) | 0.26 | 25.2% | 0.2 | 1.02 | `Pool[(((((slot & (int64_t)127) * (int64_t)64) + ((((int64_t)((int)threadIdx.x)) & (int64_t` |
| load.global | 0.10 | 10.2% | 0.2 | 0.42 | `for (int i = 0; i < 32; ++i)` |
| store.global#2 | 0.01 | 0.8% | 0.2 | 0.03 | `Pool[(((((slot & (int64_t)127) * (int64_t)4) + ((slot >> (int64_t)7) * ((int64_t)pool_stri` |

### sparse_mla_kernel grid (5, 1, 6) block (128, 1, 1)

4 launches, 120 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 23.97 us, warp 21.39 us, nsys graphs 11.81 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed, with loops' unsampled iterations | 12.97 | 60.6% | 11.64 | 48.7% |
| wait | 3.52 | 16.5% | 0.62 | 2.6% |
| load | 3.47 | 16.2% | 10.18 | 42.6% |
| compute | 0.97 | 4.5% | 0.94 | 3.9% |
| store | 0.46 | 2.2% | 0.53 | 2.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync (sampled iterations) | 3.45 | 16.1% | 11.0 | 7.76 | `__syncthreads();` |
| load.global (sampled iterations) | 3.14 | 14.7% | 2.0 | 9.12 | `for (int c = 0; c < 4; ++c) \| for (int i = 0; i < 8; ++i) \| for (int i_1 = 0; i_1 < 8; ++i` |
| compute.mma (sampled iterations) | 0.74 | 3.5% | 2.0 | 0.77 | `for (int ki = 0; ki < 32; ++ki) \| for (int ki_1 = 0; ki_1 < 2; ++ki_1)` |
| store.global | 0.46 | 2.2% | 1.0 | 0.61 | `for (int i_10 = 0; i_10 < 32; ++i_10) \| for (int i_11 = 0; i_11 < 32; ++i_11) \| for (int i` |
| load.cp_async (sampled iterations) | 0.33 | 1.5% | 2.0 | 0.40 | `for (int k_1 = 0; k_1 < 5; ++k_1) \| for (int k_1 = 0; k_1 < 9; ++k_1) \| for (int k_3 = 0; ` |
| compute.reduce (sampled iterations) | 0.22 | 1.1% | 2.0 | 0.26 | `m_g[i_5] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(m_g[i_5], (&((` |
| wait.cp_async (sampled iterations) | 0.07 | 0.3% | 1.0 | 0.10 | `tl::cp_async_wait<0>();` |

### gemm_bf16_partials_kernel grid (1, 40, 1) block (256, 1, 1)

34 launches, 320 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 2.62 us, warp 1.68 us, nsys graphs 2.75 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.22 | 72.8% | 2.07 | 79.0% |
| wait | 0.23 | 13.6% | 0.23 | 8.7% |
| compute | 0.11 | 6.6% | 0.23 | 8.8% |
| load | 0.07 | 4.3% | 0.00 | 0.0% |
| store | 0.05 | 2.8% | 0.09 | 3.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*2 (est.: sampled iterations x2) | 0.11 | 6.6% | 0.5 | 0.26 | `{` |
| wait.sync | 0.10 | 6.0% | 1.0 | 0.13 | `__syncthreads();` |
| load.tma*2 (est.: sampled iterations x2) | 0.07 | 4.3% | 0.5 | 0.19 | `if (tl::tl_shuffle_elect<128>())` |
| wait.mbarrier#2*2 (est.: sampled iterations x2) | 0.06 | 3.8% | 0.5 | 0.16 | `mbarrier[kb_1].wait(0);` |
| wait.mbarrier*2 (est.: sampled iterations x2) | 0.06 | 3.8% | 0.5 | 0.13 | `mbarrier[(kb + 3)].wait(1);` |
| store.global | 0.05 | 2.8% | 0.5 | 0.13 | `for (int i_1 = 0; i_1 < 4; ++i_1) \| for (int i_2 = 0; i_2 < 8; ++i_2) \| for (int i_4 = 0; ` |

### sink_candidates_kernel grid (1, 127, 1) block (128, 1, 1)

17 launches, 508 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 4.45 us, warp 3.47 us, nsys graphs 3.20 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.32 | 38.0% | 1.81 | 39.6% |
| compute | 0.92 | 26.4% | 0.93 | 20.4% |
| load | 0.54 | 15.5% | 1.10 | 23.9% |
| wait | 0.45 | 12.9% | 0.40 | 8.7% |
| store | 0.25 | 7.2% | 0.34 | 7.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.reduce*4 (est.: sampled iterations x4) | 0.66 | 19.0% | 1.0 | 0.77 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| load.global | 0.54 | 15.5% | 1.0 | 0.64 | `local[0] = max((num_sampled[((int64_t)((int)blockIdx.x))] - 1), 0);` |
| compute.reduce#2*4 (est.: sampled iterations x4) | 0.26 | 7.4% | 1.0 | 0.38 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 2, 32);` |
| store.global*4 (est.: sampled iterations x4) | 0.25 | 7.2% | 1.0 | 0.38 | `if (((int)threadIdx.x) == 0)` |
| wait.sync#2*4 (est.: sampled iterations x4) | 0.24 | 7.0% | 1.0 | 0.38 | `__syncthreads();` |
| wait.sync*4 (est.: sampled iterations x4) | 0.20 | 5.9% | 1.0 | 0.26 | `__syncthreads();` |

### gemm_bf16_partials_kernel grid (2, 4, 1) block (256, 1, 1)

34 launches, 64 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 2.00 us, warp 1.38 us, nsys graphs 2.11 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.92 | 66.5% | 1.48 | 63.8% |
| wait | 0.23 | 16.6% | 0.51 | 21.8% |
| compute | 0.11 | 8.3% | 0.24 | 10.4% |
| load | 0.07 | 5.1% | 0.00 | 0.0% |
| store | 0.05 | 3.6% | 0.09 | 4.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*2 (est.: sampled iterations x2) | 0.11 | 8.3% | 0.5 | 0.26 | `{` |
| wait.sync | 0.11 | 7.8% | 1.0 | 0.13 | `__syncthreads();` |
| load.tma*2 (est.: sampled iterations x2) | 0.07 | 5.1% | 0.5 | 0.19 | `if (tl::tl_shuffle_elect<128>())` |
| wait.mbarrier*2 (est.: sampled iterations x2) | 0.06 | 4.4% | 0.5 | 0.13 | `mbarrier[(kb + 3)].wait(1);` |
| wait.mbarrier#2*2 (est.: sampled iterations x2) | 0.06 | 4.4% | 0.5 | 0.13 | `mbarrier[kb_1].wait(0);` |
| store.global | 0.05 | 3.6% | 0.5 | 0.13 | `for (int i_1 = 0; i_1 < 4; ++i_1) \| for (int i_2 = 0; i_2 < 8; ++i_2) \| for (int i_4 = 0; ` |

### activation_swiglu_clamp_kernel grid (1, 10, 1) block (128, 1, 1)

51 launches, 40 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 0.93 us, warp 0.72 us, nsys graphs 1.86 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.62 | 85.4% | 1.35 | 87.4% |
| load | 0.07 | 10.0% | 0.14 | 8.9% |
| store | 0.03 | 4.6% | 0.06 | 3.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.07 | 10.0% | 0.8 | 0.10 | `*(uint4*)(gate + 0) = *(uint4*)(X + ((((int64_t)((int)blockIdx.y)) * (int64_t)1152) + (((i` |
| store.global | 0.03 | 4.6% | 0.8 | 0.06 | `*(uint4*)(Y + ((((int64_t)((int)blockIdx.y)) * (int64_t)576) + (((int64_t)((int)threadIdx.` |

### rope_kernel grid (10, 1, 1) block (128, 1, 1)

51 launches, 40 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 1.28 us, warp 1.22 us, nsys graphs 1.52 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.71 | 57.8% | 0.96 | 64.8% |
| load | 0.48 | 39.2% | 0.48 | 32.5% |
| store | 0.04 | 3.0% | 0.04 | 2.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.48 | 39.2% | 1.0 | 0.51 | `if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < tokens)) \| int64_t position ` |
| store.global | 0.04 | 3.0% | 1.0 | 0.06 | `*(uint4*)(x + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + ((((int64_t)((int)thread` |

### sink_merge_kernel grid (1, 1, 1) block (128, 1, 1)

17 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 4.32 us, warp 4.32 us, nsys graphs 4.27 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.95 | 45.2% | 2.08 | 46.7% |
| unattributed | 0.95 | 22.0% | 1.23 | 27.7% |
| wait | 0.82 | 18.9% | 0.45 | 10.0% |
| compute | 0.58 | 13.3% | 0.58 | 12.9% |
| store | 0.02 | 0.6% | 0.12 | 2.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2*3 (est.: sampled iterations x3) | 1.66 | 38.3% | 1.0 | 1.73 | `for (int step = 0; step < ((width + 127) >> 7); ++step) \| for (int step = 0; step < ((widt` |
| compute.reduce*3 (est.: sampled iterations x3) | 0.58 | 13.3% | 1.0 | 0.58 | `float other = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| wait.sync#3*3 (est.: sampled iterations x3) | 0.53 | 12.2% | 1.0 | 0.67 | `__syncthreads();` |
| load.global | 0.25 | 5.7% | 1.0 | 0.26 | `last = sampled[(max((((int64_t)num_sampled[((int64_t)((int)blockIdx.x))]) - (int64_t)1), (` |
| wait.sync*3 (est.: sampled iterations x3) | 0.17 | 3.9% | 1.0 | 0.19 | `__syncthreads();` |
| wait.sync#2*3 (est.: sampled iterations x3) | 0.12 | 2.8% | 1.0 | 0.19 | `__syncthreads();` |
| load.global#3*3 (est.: sampled iterations x3) | 0.05 | 1.1% | 0.2 | 0.19 | `if ((best[0] == -CUDART_INF_F))` |
| store.global*3 (est.: sampled iterations x3) | 0.02 | 0.6% | 0.2 | 0.10 | `alternates[((((int64_t)((int)blockIdx.x)) * (int64_t)3) + ((int64_t)rank))] = condval_31;` |

### moe_combine_kernel grid (5, 10, 1) block (128, 1, 1)

51 launches, 200 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 1.41 us, warp 1.12 us, nsys graphs 1.50 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.65 | 58.0% | 0.92 | 66.4% |
| unattributed | 0.41 | 37.2% | 0.41 | 29.8% |
| store | 0.05 | 4.9% | 0.05 | 3.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.65 | 58.0% | 1.0 | 0.93 | `for (int k = 0; k < 3; ++k) \| for (int k = 0; k < 6; ++k) \| if ((((int)blockIdx.y) < share` |
| store.global | 0.05 | 4.9% | 1.0 | 0.06 | `{` |

### indexer_score_kernel grid (64, 5, 1) block (128, 1, 1)

16 launches, 1280 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 4.32 us, warp 1.97 us, nsys graphs 3.26 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.10 | 55.9% | 2.25 | 52.3% |
| unattributed, with loops' unsampled iterations | 0.68 | 34.5% | 1.40 | 32.5% |
| wait | 0.14 | 7.1% | 0.45 | 10.5% |
| compute | 0.04 | 1.9% | 0.15 | 3.5% |
| store | 0.01 | 0.6% | 0.05 | 1.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.78 | 39.5% | 1.0 | 0.91 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 4096); \| i` |
| load.global#2 (sampled iterations) | 0.29 | 14.8% | 0.2 | 1.18 | `for (int i_1 = 0; i_1 < 32; ++i_1)` |
| wait.sync#3 (sampled iterations) | 0.06 | 2.9% | 0.2 | 0.26 | `__syncthreads();` |
| wait.sync#2 (sampled iterations) | 0.05 | 2.5% | 0.2 | 0.26 | `__syncthreads();` |
| compute.mma (sampled iterations) | 0.04 | 1.9% | 0.2 | 0.16 | `for (int ki = 0; ki < 2; ++ki)` |
| load.global#3 (sampled iterations) | 0.03 | 1.5% | 0.2 | 0.26 | `if (((int)threadIdx.x) < 64)` |
| wait.sync (sampled iterations) | 0.02 | 0.8% | 0.2 | 0.06 | `__syncthreads();` |
| wait.sync#4 (sampled iterations) | 0.02 | 0.8% | 0.2 | 0.06 | `__syncthreads();` |
| store.global (sampled iterations) | 0.01 | 0.6% | 0.2 | 0.06 | `if ((((int)threadIdx.x) >> 6) == 0)` |

### indexer_score_kernel grid (128, 5, 1) block (128, 1, 1)

16 launches, 2560 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 4.05 us, warp 1.45 us, nsys graphs 3.49 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.77 | 52.8% | 2.49 | 61.3% |
| unattributed, with loops' unsampled iterations | 0.56 | 38.3% | 1.22 | 30.1% |
| wait | 0.09 | 6.1% | 0.18 | 4.5% |
| compute | 0.03 | 2.2% | 0.12 | 3.1% |
| store | 0.01 | 0.6% | 0.04 | 1.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.36 | 24.7% | 1.0 | 0.74 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 4096); \| i` |
| load.global#2 (sampled iterations) | 0.35 | 24.0% | 0.2 | 1.50 | `for (int i_1 = 0; i_1 < 32; ++i_1)` |
| load.global#3 (sampled iterations) | 0.06 | 4.1% | 0.2 | 0.42 | `if (((int)threadIdx.x) < 64)` |
| wait.sync#2 (sampled iterations) | 0.05 | 3.6% | 0.2 | 0.38 | `__syncthreads();` |
| compute.mma (sampled iterations) | 0.03 | 2.2% | 0.2 | 0.13 | `for (int ki = 0; ki < 2; ++ki)` |
| wait.sync (sampled iterations) | 0.01 | 1.0% | 0.2 | 0.06 | `__syncthreads();` |
| wait.sync#3 (sampled iterations) | 0.01 | 0.9% | 0.2 | 0.06 | `__syncthreads();` |
| wait.sync#4 (sampled iterations) | 0.01 | 0.7% | 0.2 | 0.06 | `__syncthreads();` |
| store.global (sampled iterations) | 0.01 | 0.6% | 0.2 | 0.05 | `if ((((int)threadIdx.x) >> 6) == 0)` |

### per_token_cast_kernel_kernel grid (10, 1, 1) block (64, 1, 1)

51 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 10 warps; span 1.18 us, warp 1.06 us, nsys graphs 1.09 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.44 | 41.1% | 0.45 | 37.0% |
| load | 0.31 | 29.6% | 0.38 | 30.6% |
| store | 0.21 | 20.2% | 0.30 | 24.2% |
| compute | 0.10 | 9.1% | 0.10 | 8.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.22 | 21.1% | 1.0 | 0.29 | `for (int i = 0; i < 2; ++i)` |
| store.global#2 | 0.11 | 10.6% | 1.0 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.10 | 9.1% | 1.0 | 0.13 | `amax_fragment[i_1] = tl::AllReduce<tl::MaxOp, 2, 1, 0, tl::NamedBarrier<64>>::run(amax_fra` |
| load.global#2*2 (est.: sampled iterations x2) | 0.09 | 8.5% | 1.0 | 0.13 | `amax_fragment[i_1] = cutlass::bfloat16_t(__hmax((bfloat16_t(((nv_bfloat162*)(&(amax_fragme` |
| store.global#3*2 (est.: sampled iterations x2) | 0.05 | 4.8% | 0.6 | 0.13 | `*(fp8_e4_16_t*)(out + ((((((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int64_t)10240) +` |
| store.global*2 (est.: sampled iterations x2) | 0.05 | 4.8% | 0.6 | 0.13 | `out_sf[((((((int64_t)((int)blockIdx.x)) % (int64_t)15) * (int64_t)32) + (((int64_t)((int)t` |

### finish_argmax_kernel grid (2, 1, 1) block (32, 1, 1)

85 launches, 2 warps; ranges, warp and last warp from the 2 recording CTAs' 2 warps; span 0.74 us, warp 0.74 us, nsys graphs 0.90 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.38 | 52.2% | 0.39 | 53.1% |
| compute | 0.13 | 17.4% | 0.12 | 16.2% |
| load | 0.13 | 17.4% | 0.12 | 16.9% |
| store | 0.10 | 13.0% | 0.10 | 13.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.reduce | 0.13 | 17.4% | 1.0 | 0.13 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| load.global | 0.13 | 17.4% | 1.0 | 0.13 | `for (int k = 0; k < 2; ++k) \| if ((((int)threadIdx.x) < parts))` |
| store.global | 0.10 | 13.0% | 1.0 | 0.10 | `if (((int)threadIdx.x) == 0)` |

### cache_writer_kernel grid (6, 1, 1) block (128, 1, 1)

8 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 8.03 us, warp 7.89 us, nsys graphs 3.68 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 5.44 | 68.9% | 3.23 | 41.6% |
| store | 2.08 | 26.4% | 3.50 | 45.1% |
| load | 0.29 | 3.7% | 0.96 | 12.4% |
| compute | 0.08 | 1.0% | 0.08 | 1.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 2.08 | 26.4% | 1.0 | 2.43 | `if (((int64_t)0 <= slot) & ((slot >> (int64_t)7) < ((int64_t)pages))) \| if (((int64_t)0 <=` |
| load.global | 0.23 | 2.9% | 1.0 | 0.26 | `for (int i = 0; i < 4; ++i)` |
| compute.reduce | 0.08 | 1.0% | 1.0 | 0.10 | `amax = max(amax, __shfl_xor_sync((uint)4294967295, amax, 1, 32));` |
| load.global#2 | 0.06 | 0.8% | 1.0 | 0.06 | `int64_t slot = slots[((int64_t)((int)blockIdx.x))];` |

### adaptive_offsets_kernel grid (1, 1, 1) block (256, 1, 1)

16 launches, 8 warps; ranges, warp and last warp from the 1 recording CTAs' 8 warps; span 3.78 us, warp 3.74 us, nsys graphs 1.98 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 1.80 | 48.0% | 1.16 | 30.2% |
| unattributed, with loops' unsampled iterations | 1.42 | 38.0% | 1.48 | 38.8% |
| store | 0.31 | 8.4% | 0.55 | 14.4% |
| load | 0.21 | 5.6% | 0.63 | 16.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync (sampled iterations) | 0.42 | 11.1% | 1.0 | 0.48 | `__syncthreads();` |
| wait.sync#18 (sampled iterations) | 0.35 | 9.4% | 1.0 | 0.42 | `__syncthreads();` |
| store.global#5 | 0.13 | 3.5% | 1.0 | 0.16 | `for (int chunk_1 = 0; chunk_1 < ((starts_len + 255) >> 8); ++chunk_1)` |
| load.global#2 (sampled iterations) | 0.12 | 3.2% | 1.0 | 0.48 | `if ((((chunk * 256) + ((int)threadIdx.x)) < num_reqs))` |
| store.global#2 (sampled iterations) | 0.09 | 2.3% | 1.0 | 0.10 | `if ((((chunk * 256) + ((int)threadIdx.x)) < num_reqs) & (drafted == 0))` |
| load.global (sampled iterations) | 0.08 | 2.2% | 1.0 | 0.10 | `if (((((chunk * 256) + ((int)threadIdx.x)) < num_reqs) & (drafted != 0)))` |
| store.global | 0.07 | 1.9% | 1.0 | 0.10 | `if (((int)threadIdx.x) == 0)` |
| wait.sync#3 (sampled iterations) | 0.07 | 1.8% | 1.0 | 0.08 | `__syncthreads();` |
| wait.sync#5 (sampled iterations) | 0.07 | 1.8% | 1.0 | 0.08 | `__syncthreads();` |
| wait.sync#10 (sampled iterations) | 0.06 | 1.7% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#11 (sampled iterations) | 0.06 | 1.7% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#12 (sampled iterations) | 0.06 | 1.7% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#13 (sampled iterations) | 0.06 | 1.7% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#14 (sampled iterations) | 0.06 | 1.7% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#15 (sampled iterations) | 0.06 | 1.7% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#16 (sampled iterations) | 0.06 | 1.7% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#17 (sampled iterations) | 0.06 | 1.7% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#2 (sampled iterations) | 0.06 | 1.7% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#4 (sampled iterations) | 0.06 | 1.7% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#6 (sampled iterations) | 0.06 | 1.7% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#7 (sampled iterations) | 0.06 | 1.7% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#8 (sampled iterations) | 0.06 | 1.7% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#9 (sampled iterations) | 0.06 | 1.7% | 1.0 | 0.06 | `__syncthreads();` |
| store.global#4 (sampled iterations) | 0.02 | 0.4% | 0.1 | 0.13 | `for (int j = 0; j < 6; ++j)` |
| load.global#3 (sampled iterations) | 0.01 | 0.2% | 0.1 | 0.06 | `state = idx_mapping[((((int64_t)chunk) * (int64_t)256) + ((int64_t)((int)threadIdx.x)))];` |
| store.global#3 (sampled iterations) | 0.01 | 0.2% | 0.1 | 0.06 | `cu_num_logits[(((((int64_t)chunk) * (int64_t)256) + ((int64_t)((int)threadIdx.x))) + (int6` |

### gemm_splitk_reduce_cast_kernel grid (16, 3, 1) block (128, 1, 1)

51 launches, 192 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 1.02 us, warp 0.97 us, nsys graphs 1.15 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.42 | 43.1% | 0.50 | 45.0% |
| load | 0.40 | 40.7% | 0.43 | 38.4% |
| compute | 0.08 | 8.4% | 0.10 | 9.1% |
| store | 0.08 | 7.8% | 0.08 | 7.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.40 | 40.7% | 1.0 | 0.45 | `if ((1 <= M))` |
| compute.reduce | 0.08 | 8.4% | 1.0 | 0.10 | `amax[0] = tl::AllReduce<tl::MaxOp, 8, 1, 0, tl::NamedBarrier<128>>::run(amax[0]);` |
| store.global | 0.08 | 7.8% | 1.0 | 0.10 | `if (((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 5)) < M)` |

### per_token_cast_kernel_kernel grid (20, 1, 1) block (64, 1, 1)

51 launches, 40 warps; ranges, warp and last warp from the 4 recording CTAs' 8 warps; span 1.09 us, warp 1.04 us, nsys graphs 0.99 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.35 | 33.7% | 0.38 | 35.0% |
| unattributed | 0.30 | 28.7% | 0.30 | 27.8% |
| store | 0.30 | 28.4% | 0.31 | 28.3% |
| compute | 0.10 | 9.2% | 0.10 | 8.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.26 | 25.3% | 1.0 | 0.29 | `for (int i = 0; i < 2; ++i)` |
| store.global#2 | 0.11 | 10.7% | 1.0 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.10 | 9.2% | 1.0 | 0.13 | `amax_fragment[i_1] = tl::AllReduce<tl::MaxOp, 2, 1, 0, tl::NamedBarrier<64>>::run(amax_fra` |
| store.global*2 (est.: sampled iterations x2) | 0.10 | 9.2% | 1.0 | 0.13 | `out_sf[((((((int64_t)((int)blockIdx.x)) % (int64_t)15) * (int64_t)32) + (((int64_t)((int)t` |
| load.global#2*2 (est.: sampled iterations x2) | 0.09 | 8.4% | 1.0 | 0.13 | `amax_fragment[i_1] = cutlass::bfloat16_t(__hmax((bfloat16_t(((nv_bfloat162*)(&(amax_fragme` |
| store.global#3*2 (est.: sampled iterations x2) | 0.09 | 8.4% | 1.0 | 0.13 | `*(fp8_e4_16_t*)(out + ((((((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int64_t)10240) +` |

### verify_kernel grid (1, 1, 1) block (96, 1, 1)

17 launches, 3 warps; ranges, warp and last warp from the 1 recording CTAs' 3 warps; span 2.85 us, warp 2.70 us, nsys graphs 2.34 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.10 | 40.7% | 1.49 | 45.8% |
| unattributed | 0.96 | 35.6% | 0.94 | 28.8% |
| compute | 0.29 | 10.7% | 0.29 | 8.8% |
| wait | 0.19 | 7.1% | 0.24 | 7.4% |
| store | 0.16 | 5.9% | 0.30 | 9.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.03 | 38.3% | 1.0 | 1.06 | `start = cu_num_logits[((int64_t)((int)blockIdx.x))];` |
| compute.reduce | 0.29 | 10.7% | 1.0 | 0.32 | `row_max[0] = vllm_max(row_max[0], __shfl_xor_sync((uint)4294967295, row_max[0], 8, 32));` |
| store.global | 0.16 | 5.9% | 1.0 | 0.29 | `if (((int)threadIdx.x) == 0)` |
| wait.sync#3 | 0.09 | 3.2% | 1.0 | 0.13 | `__syncthreads();` |
| load.global#2 | 0.06 | 2.4% | 1.0 | 0.06 | `target = local_argmax[((((int64_t)row) * argmax_stride) + ((int64_t)best_index[0]))];` |
| wait.sync#2 | 0.06 | 2.4% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync | 0.04 | 1.6% | 1.0 | 0.06 | `__syncthreads();` |

### cache_writer_kernel grid (8, 1, 1) block (128, 1, 1)

51 launches, 32 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 1.02 us, warp 0.94 us, nsys graphs 1.12 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.33 | 34.5% | 0.33 | 32.5% |
| load | 0.29 | 30.5% | 0.29 | 28.7% |
| store | 0.25 | 26.9% | 0.32 | 31.1% |
| compute | 0.08 | 8.1% | 0.08 | 7.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.25 | 26.9% | 1.0 | 0.32 | `if (((int64_t)0 <= slot) & ((slot >> (int64_t)7) < ((int64_t)pages))) \| if (((int64_t)0 <=` |
| load.global | 0.23 | 24.4% | 1.0 | 0.26 | `for (int i = 0; i < 4; ++i)` |
| compute.reduce | 0.08 | 8.1% | 1.0 | 0.10 | `amax = max(amax, __shfl_xor_sync((uint)4294967295, amax, 1, 32));` |
| load.global#2 | 0.06 | 6.1% | 1.0 | 0.06 | `int64_t slot = slots[((int64_t)((int)blockIdx.x))];` |

### gemm_splitk_reduce_kernel grid (1, 5, 1) block (128, 1, 1)

64 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 0.83 us, warp 0.82 us, nsys graphs 0.99 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.46 | 55.8% | 0.40 | 51.8% |
| load | 0.32 | 38.4% | 0.32 | 41.5% |
| store | 0.05 | 5.8% | 0.05 | 6.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.32 | 38.4% | 1.0 | 0.34 | `acc[0] = P[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x))` |
| store.global | 0.05 | 5.8% | 1.0 | 0.06 | `C[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x)) * (int64` |

### sink_cut_kernel grid (1, 1, 1) block (128, 1, 1)

34 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 1.79 us, warp 0.62 us, nsys graphs 2.08 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.34 | 55.2% | 0.42 | 29.1% |
| store | 0.16 | 25.3% | 0.51 | 35.1% |
| load | 0.12 | 19.5% | 0.52 | 35.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.16 | 25.3% | 0.6 | 0.51 | `if (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) < num_cut) \| saved[((((int64_t)((int)` |
| load.global | 0.12 | 19.5% | 0.1 | 0.48 | `for (int b = 0; b < num_reqs; ++b)` |

### norm_forward_and_per_token_cast_kernel_kernel grid (8, 1, 1) block (128, 1, 1)

17 launches, 32 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 2.34 us, warp 2.28 us, nsys graphs 1.79 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 2.16 | 94.7% | 3.29 | 114.6% |
| load | 0.43 | 18.8% | 0.44 | 15.4% |
| compute | 0.13 | 5.7% | 0.13 | 4.5% |
| wait | 0.05 | 2.4% | 0.06 | 2.1% |
| unattributed | 0.00 | 0.0% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*5 (est.: sampled iterations x5) | 1.91 | 83.6% | 1.0 | 2.08 | `for (int j_1 = 0; j_1 < 2; ++j_1)` |
| load.global*5 (est.: sampled iterations x5) | 0.22 | 9.6% | 1.0 | 0.32 | `*(uint4*)(x_local + (vi * 8)) = *(uint4*)(x + ((((((int64_t)vi) * (int64_t)1024) + (((int6` |
| load.global#2*5 (est.: sampled iterations x5) | 0.21 | 9.2% | 1.0 | 0.32 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 1024) + (((int)threadIdx.x) * 8)` |
| store.global*5 (est.: sampled iterations x5) | 0.20 | 8.8% | 1.0 | 0.32 | `for (int j = 0; j < 4; ++j)` |
| compute.reduce | 0.13 | 5.7% | 1.0 | 0.16 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| wait.sync | 0.05 | 2.4% | 1.0 | 0.10 | `__syncthreads();` |
| store.global#2 | 0.04 | 1.8% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| store.global#3 | 0.01 | 0.6% | 0.2 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |

### sparse_mla_kernel grid (1, 1, 6) block (128, 1, 1)

2 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 24.11 us, warp 24.10 us, nsys graphs 16.34 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed, with loops' unsampled iterations | 7.03 | 29.2% | 6.94 | 28.8% |
| load | 6.41 | 26.6% | 9.06 | 37.6% |
| wait | 5.25 | 21.8% | 2.58 | 10.7% |
| compute | 4.73 | 19.6% | 4.86 | 20.2% |
| store | 0.68 | 2.8% | 0.67 | 2.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 | 4.47 | 18.5% | 1.0 | 4.78 | `for (int i = 0; i < 8; ++i)` |
| wait.sync (sampled iterations) | 2.66 | 11.0% | 1.0 | 3.54 | `__syncthreads();` |
| compute.mma*4 (est.: sampled iterations x4) | 1.76 | 7.3% | 1.0 | 1.79 | `for (int ki = 0; ki < 32; ++ki) \| {` |
| compute.mma#2*4 (est.: sampled iterations x4) | 1.16 | 4.8% | 1.0 | 1.28 | `for (int ki_1 = 0; ki_1 < 2; ++ki_1) \| if (((((int)blockIdx.x) * 2) + (c_1 >> 1)) < 3)` |
| compute.reduce*8 (est.: sampled iterations x8) | 0.95 | 4.0% | 1.0 | 1.02 | `m_g[i_4] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(m_g[i_4], (&((` |
| load.global (sampled iterations) | 0.93 | 3.9% | 1.0 | 3.55 | `for (int c = 0; c < 4; ++c) \| for (int i = 0; i < 8; ++i) \| for (int i_1 = 0; i_1 < 8; ++i` |
| compute.reduce#2*8 (est.: sampled iterations x8) | 0.86 | 3.6% | 1.0 | 1.02 | `row_sum[i_7] = tl::AllReduce<tl::SumOp, 128, 32, 0, tl::NamedBarrier<128>>::run(row_sum[i_` |
| store.global | 0.68 | 2.8% | 1.0 | 0.70 | `for (int i_10 = 0; i_10 < 32; ++i_10) \| for (int i_11 = 0; i_11 < 32; ++i_11) \| for (int i` |
| load.cp_async#2*4 (est.: sampled iterations x4) | 0.65 | 2.7% | 1.0 | 0.70 | `if (((((int)blockIdx.x) * 2) + (c_1 >> 1)) < 3) \| if ((c_1 < 3) & (c_1 < 3)) \| if (c < 3)` |
| wait.sync#10*8 (est.: sampled iterations x8) | 0.39 | 1.6% | 1.0 | 0.51 | `__syncthreads();` |
| load.cp_async (sampled iterations) | 0.36 | 1.5% | 1.0 | 0.38 | `for (int k_1 = 0; k_1 < 5; ++k_1) \| for (int k_1 = 0; k_1 < 9; ++k_1) \| for (int k_3 = 0; ` |
| wait.sync#8*8 (est.: sampled iterations x8) | 0.32 | 1.3% | 1.0 | 0.51 | `__syncthreads();` |
| wait.cp_async*4 (est.: sampled iterations x4) | 0.28 | 1.2% | 1.0 | 0.38 | `tl::cp_async_wait<0>();` |
| wait.sync#4*4 (est.: sampled iterations x4) | 0.26 | 1.1% | 1.0 | 0.32 | `__syncthreads();` |
| wait.sync#7*4 (est.: sampled iterations x4) | 0.25 | 1.0% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#9*4 (est.: sampled iterations x4) | 0.25 | 1.0% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#11*4 (est.: sampled iterations x4) | 0.25 | 1.0% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#6*4 (est.: sampled iterations x4) | 0.24 | 1.0% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#5*4 (est.: sampled iterations x4) | 0.24 | 1.0% | 1.0 | 0.32 | `__syncthreads();` |
| wait.sync#3 | 0.06 | 0.2% | 1.0 | 0.08 | `__syncthreads();` |
| wait.sync#2 | 0.05 | 0.2% | 1.0 | 0.06 | `__syncthreads();` |

### compute_slot_mappings_kernel grid (1, 11, 1) block (256, 1, 1)

17 launches, 88 warps; ranges, warp and last warp from the 6 recording CTAs' 48 warps; span 2.69 us, warp 1.46 us, nsys graphs 2.85 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.80 | 54.9% | 0.80 | 30.4% |
| unattributed | 0.66 | 45.1% | 1.83 | 69.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.80 | 54.9% | 1.0 | 0.83 | `int first_token = query_start_loc[0];` |

### resample_kernel grid (1, 127, 1) block (128, 1, 1)

17 launches, 508 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 2.02 us, warp 1.69 us, nsys graphs 2.21 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.35 | 79.6% | 2.27 | 86.9% |
| unattributed | 0.30 | 17.8% | 0.30 | 11.4% |
| compute | 0.04 | 2.6% | 0.04 | 1.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.35 | 79.6% | 1.0 | 1.66 | `end = cu_num_logits[(((int64_t)((int)blockIdx.x)) + (int64_t)1)];` |
| compute.reduce | 0.04 | 2.6% | 1.0 | 0.06 | `if ((temp != 0x0p+0f/*0.000000e+00*/) \| bonus)` |

### sink_policy_kernel grid (1, 1, 1) block (32, 1, 1)

17 launches, 1 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 2.53 us, warp 2.53 us, nsys graphs 2.66 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.12 | 44.3% | 1.12 | 43.1% |
| unattributed | 0.83 | 32.9% | 0.89 | 34.5% |
| store | 0.58 | 22.8% | 0.58 | 22.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 | 0.67 | 26.6% | 1.0 | 0.67 | `state[0] = idx_mapping[((int64_t)((int)blockIdx.x))];` |
| store.global#3 | 0.51 | 20.3% | 1.0 | 0.51 | `for (int a = 0; a < 3; ++a) \| recovered_logprobs[((int64_t)((int)blockIdx.x))] = 0x0p+0f/*` |
| load.global | 0.22 | 8.9% | 1.0 | 0.22 | `start[0] = cu_num_logits[((int64_t)((int)blockIdx.x))];` |
| load.global#4 | 0.16 | 6.3% | 1.0 | 0.16 | `for (int r = 0; r < 2; ++r) \| for (int r_2 = 0; r_2 < 8; ++r_2)` |
| load.global#5 | 0.06 | 2.5% | 1.0 | 0.06 | `AtomicAdd((&(counts[0])), draft_zero_flag[0]);` |
| store.global#4 | 0.06 | 2.5% | 1.0 | 0.06 | `events[(((int64_t)((int)blockIdx.x)) * (int64_t)6)] = outcome[0];` |

### sink_rows_kernel grid (1, 64, 1) block (256, 1, 1)

17 launches, 512 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 1.70 us, warp 0.95 us, nsys graphs 1.86 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.36 | 38.4% | 0.90 | 35.5% |
| load | 0.32 | 33.3% | 0.47 | 18.7% |
| store | 0.27 | 28.3% | 1.16 | 45.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.31 | 32.8% | 1.0 | 0.38 | `row = (max((((int64_t)num_sampled[((int64_t)((int)blockIdx.x))]) - (int64_t)1), (int64_t)0` |
| store.global | 0.25 | 26.6% | 1.0 | 0.48 | `for (int k_1 = 0; k_1 < 8; ++k_1)` |
| store.global#2 | 0.02 | 1.7% | 0.0 | 0.51 | `tokens[((((int64_t)((int)blockIdx.x)) * (int64_t)3) + ((int64_t)((int)threadIdx.x)))] = to` |
| load.global#2 | 0.01 | 0.5% | 0.0 | 0.16 | `token = max(alternates[((((int64_t)((int)blockIdx.x)) * (int64_t)3) + ((int64_t)((int)thre` |

### metadata_pages_kernel grid (5, 1, 1) block (128, 1, 1)

32 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 1.31 us, warp 0.95 us, nsys graphs 1.50 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.72 | 76.0% | 0.79 | 59.6% |
| load | 0.14 | 15.1% | 0.19 | 14.5% |
| store | 0.08 | 8.9% | 0.34 | 25.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.13 | 13.4% | 1.0 | 0.13 | `if ((((offset + ((int)blockIdx.x)) < src_rows) && (0 <= (offset + ((int)blockIdx.x)))))` |
| store.global | 0.08 | 8.9% | 0.2 | 0.35 | `if (((int)threadIdx.x) < 32)` |
| load.global#2 | 0.02 | 1.7% | 0.2 | 0.06 | `if (((req < table_rows) && (((int)threadIdx.x) < table_cols)))` |

### norm_forward_and_per_token_cast_kernel_kernel grid (10, 1, 1) block (128, 1, 1)

17 launches, 40 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 2.34 us, warp 2.28 us, nsys graphs 1.82 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 2.15 | 94.3% | 2.20 | 94.1% |
| load | 0.44 | 19.3% | 0.46 | 19.7% |
| compute | 0.13 | 5.7% | 0.13 | 5.6% |
| wait | 0.06 | 2.5% | 0.06 | 2.6% |
| unattributed | 0.00 | 0.0% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*5 (est.: sampled iterations x5) | 1.88 | 82.5% | 1.0 | 2.08 | `for (int j_1 = 0; j_1 < 2; ++j_1)` |
| load.global#2*5 (est.: sampled iterations x5) | 0.22 | 9.8% | 1.0 | 0.32 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 1024) + (((int)threadIdx.x) * 8)` |
| load.global*5 (est.: sampled iterations x5) | 0.22 | 9.5% | 1.0 | 0.32 | `*(uint4*)(x_local + (vi * 8)) = *(uint4*)(x + ((((((int64_t)vi) * (int64_t)1024) + (((int6` |
| store.global*5 (est.: sampled iterations x5) | 0.22 | 9.5% | 1.0 | 0.32 | `for (int j = 0; j < 4; ++j)` |
| compute.reduce | 0.13 | 5.7% | 1.0 | 0.16 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| wait.sync | 0.06 | 2.5% | 1.0 | 0.10 | `__syncthreads();` |
| store.global#2 | 0.04 | 1.8% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| store.global#3 | 0.01 | 0.6% | 0.2 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |

### sink_flags_kernel grid (1, 1, 1) block (128, 1, 1)

17 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 2.18 us, warp 0.71 us, nsys graphs 3.07 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.38 | 53.9% | 1.52 | 69.9% |
| unattributed | 0.25 | 34.8% | 0.32 | 14.9% |
| store | 0.08 | 11.2% | 0.33 | 15.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.38 | 53.9% | 0.2 | 1.54 | `token = logits_indices[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)th` |
| store.global | 0.08 | 11.2% | 0.2 | 0.32 | `flags[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = (` |

### flatten_sampled_kernel grid (1, 1, 1) block (128, 1, 1)

17 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 1.12 us, warp 0.81 us, nsys graphs 1.34 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.47 | 58.4% | 0.64 | 32.6% |
| unattributed | 0.31 | 38.6% | 1.22 | 62.7% |
| store | 0.02 | 3.0% | 0.09 | 4.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.42 | 51.5% | 1.0 | 0.42 | `for (int r = 0; r < num_reqs; ++r)` |
| load.global#2 | 0.06 | 6.9% | 0.2 | 0.22 | `int i = (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) - cu_num_logits[req]);` |
| store.global | 0.02 | 3.0% | 0.2 | 0.10 | `flat[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = co` |

### metadata_decode_kernel grid (6, 2, 1) block (128, 1, 1)

4 launches, 48 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 7.66 us, warp 6.36 us, nsys graphs 1.76 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 3.23 | 50.8% | 7.31 | 88.8% |
| unattributed | 2.99 | 47.0% | 0.78 | 9.5% |
| store | 0.14 | 2.2% | 0.14 | 1.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 2.86 | 44.9% | 0.5 | 6.34 | `if ((((offset + ((int)blockIdx.x)) < src_rows) && (0 <= (offset + ((int)blockIdx.x)))))` |
| load.global#3 | 0.35 | 5.5% | 0.5 | 0.74 | `if ((((offset + ((int)blockIdx.x)) < main_src_rows) && (0 <= (offset + ((int)blockIdx.x)))` |
| store.global#4 | 0.10 | 1.6% | 0.5 | 0.64 | `if (0 < live)` |
| store.global | 0.03 | 0.4% | 0.5 | 0.06 | `swa[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = con` |
| load.global#2 | 0.02 | 0.4% | 0.1 | 0.19 | `if (((0 <= req) & ((int64_t)0 <= pos)))` |
| store.global#2 | 0.01 | 0.1% | 0.1 | 0.05 | `lengths[((int64_t)((int)blockIdx.x))] = ((int)max(info[1], (int64_t)0));` |
| store.global#3 | 0.01 | 0.1% | 0.1 | 0.05 | `top_lengths[((int64_t)((int)blockIdx.x))] = condval_6; \| top_lengths[((int64_t)((int)block` |

### rope_kernel grid (5, 1, 1) block (256, 1, 1)

32 launches, 40 warps; ranges, warp and last warp from the 5 recording CTAs' 40 warps; span 1.09 us, warp 0.99 us, nsys graphs 1.14 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.69 | 70.0% | 0.70 | 69.2% |
| load | 0.26 | 26.2% | 0.27 | 26.5% |
| store | 0.04 | 3.9% | 0.04 | 4.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.26 | 26.2% | 1.0 | 0.29 | `if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < tokens)) \| int64_t position ` |
| store.global | 0.04 | 3.9% | 1.0 | 0.06 | `*(uint4*)(x + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + ((((int64_t)((int)thread` |

### dspark_confidence_kernel grid (10, 1, 1) block (128, 1, 1)

17 launches, 40 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 1.92 us, warp 1.70 us, nsys graphs 1.98 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.34 | 78.8% | 1.47 | 77.2% |
| load | 0.17 | 10.2% | 0.21 | 10.8% |
| compute | 0.10 | 5.6% | 0.09 | 5.0% |
| wait | 0.07 | 4.1% | 0.05 | 2.4% |
| store | 0.02 | 1.3% | 0.09 | 4.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.16 | 9.4% | 1.0 | 0.16 | `for (int c = 0; c < 5; ++c)` |
| compute.reduce | 0.10 | 5.6% | 1.0 | 0.10 | `acc[0] = tl::warp_reduce_sum(acc[0]);` |
| wait.sync | 0.07 | 4.1% | 1.0 | 0.16 | `__syncthreads();` |
| store.global | 0.02 | 1.3% | 0.2 | 0.10 | `out[((int64_t)((int)blockIdx.x))] = (0x1p+0f/*1.000000e+00*/ / (0x1p+0f/*1.000000e+00*/ + ` |
| load.global#2 | 0.01 | 0.8% | 0.2 | 0.06 | `total[0] = 0x0p+0f/*0.000000e+00*/;` |

### gather_block_tables_kernel grid (11, 1, 1) block (256, 1, 1)

17 launches, 88 warps; ranges, warp and last warp from the 6 recording CTAs' 48 warps; span 1.92 us, warp 1.36 us, nsys graphs 2.10 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.06 | 78.3% | 1.58 | 83.8% |
| unattributed | 0.30 | 21.7% | 0.31 | 16.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.06 | 78.3% | 1.0 | 1.63 | `int g = chunk_groups[((int64_t)((int)blockIdx.x))];` |

### post_update_kernel grid (1, 1, 1) block (32, 1, 1)

17 launches, 1 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 1.79 us, warp 1.79 us, nsys graphs 2.26 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.70 | 39.3% | 0.68 | 38.1% |
| store | 0.58 | 32.1% | 0.57 | 31.7% |
| unattributed | 0.51 | 28.6% | 0.54 | 30.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 | 0.51 | 28.6% | 1.0 | 0.51 | `int count = num_sampled[((((int64_t)((int)blockIdx.x)) * (int64_t)32) + ((int64_t)((int)th` |
| store.global#2 | 0.42 | 23.2% | 1.0 | 0.42 | `if (0 < count)` |
| store.global | 0.16 | 8.9% | 1.0 | 0.16 | `if (0 < count)` |
| load.global | 0.10 | 5.4% | 1.0 | 0.10 | `int64_t req = idx_mapping[((((int64_t)((int)blockIdx.x)) * (int64_t)32) + ((int64_t)((int)` |
| load.global#3 | 0.10 | 5.4% | 1.0 | 0.10 | `if ((0 < count))` |

### insert_kernel grid (1, 1, 1) block (32, 1, 1)

17 launches, 1 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 1.57 us, warp 1.57 us, nsys graphs 1.47 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.70 | 44.9% | 0.77 | 44.4% |
| unattributed | 0.42 | 26.5% | 0.42 | 24.2% |
| store | 0.42 | 26.5% | 0.50 | 28.7% |
| compute | 0.03 | 2.0% | 0.05 | 2.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#3 | 0.51 | 32.7% | 1.0 | 0.51 | `int state = mapping[((int64_t)position)]; \| int64_t state = mapping[((int64_t)position)];` |
| store.global | 0.42 | 26.5% | 1.0 | 0.42 | `counts[((int64_t)((int)blockIdx.x))] = (count_local + 1); \| counts[((int64_t)((int)blockId` |
| load.global | 0.13 | 8.2% | 1.0 | 0.13 | `for (int k = 0; k < 4; ++k)` |
| load.global#2 | 0.06 | 4.1% | 1.0 | 0.06 | `count_local = counts[((int64_t)((int)blockIdx.x))];` |
| compute.reduce | 0.03 | 2.0% | 1.0 | 0.03 | `int count = __shfl_sync((uint)4294967295, count_local, 0, 32);` |

### gemm_splitk_reduce_kernel grid (4, 8, 1) block (128, 1, 1)

51 launches, 128 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 0.54 us, warp 0.53 us, nsys graphs 0.70 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.26 | 48.3% | 0.26 | 47.6% |
| load | 0.23 | 43.8% | 0.24 | 44.2% |
| store | 0.04 | 7.9% | 0.05 | 8.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.23 | 43.8% | 1.0 | 0.26 | `acc[0] = P[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x))` |
| store.global | 0.04 | 7.9% | 1.0 | 0.06 | `C[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x)) * (int64` |

### assign_draft_budget_kernel grid (1, 1, 1) block (256, 1, 1)

16 launches, 8 warps; ranges, warp and last warp from the 1 recording CTAs' 8 warps; span 1.79 us, warp 1.79 us, nsys graphs 1.86 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 0.79 | 44.1% | 0.29 | 16.3% |
| unattributed, with loops' unsampled iterations | 0.57 | 31.9% | 0.67 | 37.2% |
| load | 0.24 | 13.3% | 0.64 | 35.8% |
| store | 0.19 | 10.7% | 0.19 | 10.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync | 0.54 | 30.1% | 1.0 | 0.61 | `__syncthreads();` |
| load.global | 0.24 | 13.3% | 1.0 | 0.74 | `for (int chunk = 0; chunk < ((num_reqs + 255) >> 8); ++chunk)` |
| store.global | 0.19 | 10.7% | 1.0 | 0.19 | `for (int chunk_2 = 0; chunk_2 < ((num_reqs + 255) >> 8); ++chunk_2)` |
| wait.sync#3 | 0.19 | 10.4% | 1.0 | 0.22 | `__syncthreads();` |
| wait.sync#2 (sampled iterations) | 0.06 | 3.6% | 1.0 | 0.06 | `__syncthreads();` |

### context_inputs_kernel grid (5, 8, 1) block (128, 1, 1)

17 launches, 160 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 1.57 us, warp 1.23 us, nsys graphs 1.97 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.81 | 66.1% | 1.25 | 75.1% |
| unattributed | 0.34 | 27.9% | 0.34 | 20.7% |
| store | 0.07 | 6.0% | 0.07 | 4.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.81 | 66.1% | 1.0 | 1.15 | `vllm_dspark_context_b128((&(aux[(((((int64_t)((int)blockIdx.y)) * (int64_t)15360) + (((int` |
| store.global | 0.07 | 6.0% | 1.0 | 0.16 | `if (((((int)blockIdx.x) == 0) & (((int)threadIdx.x) == 0)) & (num_tokens <= ((int)blockIdx` |

### topk_ids_kernel grid (6, 1, 1) block (512, 1, 1)

1 launches, 96 warps; ranges, warp and last warp from the 6 recording CTAs' 96 warps; span 27.04 us, warp 24.42 us, nsys graphs 26.11 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 20.60 | 84.4% | 21.95 | 81.2% |
| unattributed | 1.90 | 7.8% | 1.89 | 7.0% |
| wait | 1.42 | 5.8% | 1.70 | 6.3% |
| store | 0.49 | 2.0% | 1.50 | 5.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 20.60 | 84.4% | 1.0 | 23.49 | `for (int t = 0; t < (vocab >> 14); ++t)` |
| wait.sync | 1.29 | 5.3% | 1.0 | 2.91 | `__syncthreads();` |
| store.global | 0.49 | 2.0% | 1.0 | 4.48 | `if ((((int)threadIdx.x) >> 5) == 0)` |
| wait.sync#2 | 0.08 | 0.3% | 1.0 | 0.16 | `__syncthreads();` |
| wait.sync#3 | 0.06 | 0.2% | 1.0 | 0.06 | `__syncthreads();` |

### sparse_mla_kernel_1 grid (16, 6, 1) block (128, 1, 1)

6 launches, 96 CTAs, 384 warps; span 1.97 us, warp 1.53 us, nsys graphs 1.95 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.48 | 96.7% | 2.45 | 59.1% |
| load | 0.04 | 2.3% | 1.46 | 35.2% |
| wait | 0.01 | 0.7% | 0.11 | 2.7% |
| compute | 0.00 | 0.2% | 0.08 | 1.9% |
| store | 0.00 | 0.1% | 0.05 | 1.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.04 | 2.3% | 0.1 | 1.09 | `M = __fmul_rn(Sinks[((int)blockIdx.x)], 0x1.7154764ee6c2fp+0f/*1.442695e+00*/); \| if (((in` |
| wait.sync | 0.01 | 0.7% | 0.1 | 0.42 | `__syncthreads();` |
| compute.reduce | 0.00 | 0.2% | 0.0 | 0.10 | `amax[0] = tl::AllReduce<tl::MaxOp, 8, 1, 0, tl::NamedBarrier<128>>::run(amax[0]);` |
| store.global | 0.00 | 0.1% | 0.0 | 0.06 | `*(uint2*)(Output + (((((int64_t)((int)blockIdx.y)) * (int64_t)8192) + (((int64_t)((int)blo` |
| compute.reduce#3 | 0.00 | 0.0% | 0.0 | 0.00 | `amax[0] = tl::AllReduce<tl::MaxOp, 8, 1, 0, tl::NamedBarrier<128>>::run(amax[0]);` |
| load.global#3 | 0.00 | 0.0% | 0.0 | 0.00 | `M = __fmul_rn(Sinks[((int)blockIdx.x)], 0x1.7154764ee6c2fp+0f/*1.442695e+00*/);` |
| load.global#4 | 0.00 | 0.0% | 0.0 | 0.00 | `if (((int)threadIdx.x) < 32)` |
| store.global#2 | 0.00 | 0.0% | 0.0 | 0.00 | `*(uint2*)(Output + (((((int64_t)((int)blockIdx.y)) * (int64_t)8192) + (((int64_t)((int)blo` |
| wait.sync#12 | 0.00 | 0.0% | 0.0 | 0.00 | `__syncthreads();` |
| wait.sync#13 | 0.00 | 0.0% | 0.0 | 0.00 | `__syncthreads();` |

### combine_sampled_and_draft_tokens_kernel grid (1, 1, 1) block (32, 1, 1)

17 launches, 1 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 1.41 us, warp 1.41 us, nsys graphs 1.94 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.90 | 63.6% | 0.91 | 64.5% |
| unattributed | 0.42 | 29.5% | 0.41 | 28.7% |
| store | 0.10 | 6.8% | 0.10 | 6.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.90 | 63.6% | 1.0 | 0.90 | `if (((((int)blockIdx.x) * 32) + ((int)threadIdx.x)) < num_reqs) \| req = idx_mapping[((int6` |
| store.global | 0.10 | 6.8% | 1.0 | 0.10 | `if ((prefill < seq) && (1 < num_logits))` |

### mhc_post_kernel grid (50, 1, 1) block (128, 1, 1)

17 launches, 200 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 1.31 us, warp 1.02 us, nsys graphs 1.60 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.84 | 82.9% | 1.11 | 83.2% |
| load | 0.12 | 11.8% | 0.14 | 10.9% |
| store | 0.05 | 5.3% | 0.08 | 5.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.12 | 11.8% | 1.0 | 0.16 | `*(float4*)(pm + 0) = *(float4*)(post + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int` |
| store.global | 0.05 | 5.3% | 1.0 | 0.10 | `for (int n_1 = 0; n_1 < 4; ++n_1)` |

### per_token_cast_kernel_kernel grid (60, 1, 1) block (64, 1, 1)

17 launches, 120 warps; ranges, warp and last warp from the 4 recording CTAs' 8 warps; span 1.18 us, warp 1.11 us, nsys graphs 1.26 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.39 | 35.4% | 0.45 | 35.8% |
| unattributed | 0.31 | 27.8% | 0.40 | 31.9% |
| store | 0.30 | 27.4% | 0.30 | 23.9% |
| compute | 0.10 | 9.4% | 0.11 | 8.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.30 | 27.4% | 1.0 | 0.35 | `for (int i = 0; i < 2; ++i)` |
| store.global#2 | 0.11 | 10.1% | 1.0 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.10 | 9.4% | 1.0 | 0.13 | `amax_fragment[i_1] = tl::AllReduce<tl::MaxOp, 2, 1, 0, tl::NamedBarrier<64>>::run(amax_fra` |
| store.global*2 (est.: sampled iterations x2) | 0.10 | 9.4% | 1.0 | 0.13 | `out_sf[((((((int64_t)((int)blockIdx.x)) % (int64_t)15) * (int64_t)32) + (((int64_t)((int)t` |
| load.global#2*2 (est.: sampled iterations x2) | 0.09 | 7.9% | 1.0 | 0.13 | `amax_fragment[i_1] = cutlass::bfloat16_t(__hmax((bfloat16_t(((nv_bfloat162*)(&(amax_fragme` |
| store.global#3*2 (est.: sampled iterations x2) | 0.09 | 7.9% | 1.0 | 0.13 | `*(fp8_e4_16_t*)(out + ((((((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int64_t)10240) +` |

### indexer_topk_segment_kernel grid (1, 6, 1) block (512, 1, 1)

2 launches, 96 warps; ranges, warp and last warp from the 6 recording CTAs' 96 warps; span 10.61 us, warp 10.56 us, nsys graphs 5.28 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed, with loops' unsampled iterations | 5.30 | 50.1% | 5.76 | 54.4% |
| wait | 4.03 | 38.1% | 3.50 | 33.1% |
| load | 1.07 | 10.1% | 1.10 | 10.4% |
| store | 0.17 | 1.6% | 0.22 | 2.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.07 | 10.1% | 1.0 | 1.12 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 4096); \| i` |
| wait.sync (sampled iterations) | 1.02 | 9.6% | 14.5 | 1.25 | `__syncthreads();` |
| wait.sync#20*9 (est.: sampled iterations x9) | 0.33 | 3.2% | 0.5 | 0.43 | `__syncthreads();` |
| wait.sync#19*9 (est.: sampled iterations x9) | 0.33 | 3.1% | 0.5 | 0.43 | `__syncthreads();` |
| wait.sync#5*8 (est.: sampled iterations x8) | 0.32 | 3.1% | 0.5 | 0.38 | `__syncthreads();` |
| wait.sync#24*9 (est.: sampled iterations x9) | 0.32 | 3.0% | 0.5 | 0.43 | `__syncthreads();` |
| wait.sync#13*8 (est.: sampled iterations x8) | 0.31 | 3.0% | 0.5 | 0.38 | `__syncthreads();` |
| wait.sync#23*9 (est.: sampled iterations x9) | 0.30 | 2.9% | 0.5 | 0.43 | `__syncthreads();` |
| wait.sync#14*8 (est.: sampled iterations x8) | 0.29 | 2.8% | 0.5 | 0.38 | `__syncthreads();` |
| wait.sync#6*8 (est.: sampled iterations x8) | 0.28 | 2.7% | 0.5 | 0.38 | `__syncthreads();` |
| store.global | 0.17 | 1.6% | 1.5 | 0.24 | `for (int i_3 = 0; i_3 < 4; ++i_3) \| if (((int)threadIdx.x) < total) \| if (((int)threadIdx.` |
| wait.sync#25 | 0.05 | 0.5% | 0.5 | 0.08 | `__syncthreads();` |
| wait.sync#16 | 0.04 | 0.4% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#8 | 0.04 | 0.4% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#17 | 0.04 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#18 | 0.04 | 0.3% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#9 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#2 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#3 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#15 | 0.03 | 0.3% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#11 | 0.03 | 0.3% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#10 | 0.03 | 0.3% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#12 | 0.03 | 0.3% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#7 | 0.03 | 0.3% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#22 | 0.03 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#21 | 0.03 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#4 | 0.03 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |

### moe_swiglu_cast_kernel_kernel grid (8, 36, 1) block (288, 1, 1)

6 launches, 2592 warps; ranges, warp and last warp from the 4 recording CTAs' 36 warps; span 3.62 us, warp 1.63 us, nsys graphs 1.28 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.76 | 46.7% | 1.78 | 53.7% |
| load | 0.69 | 42.7% | 0.83 | 25.0% |
| compute | 0.10 | 6.4% | 0.45 | 13.5% |
| store | 0.07 | 4.2% | 0.26 | 7.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.69 | 42.7% | 1.0 | 0.99 | `for (int r = 0; r < 4; ++r)` |
| compute.reduce*4 (est.: sampled iterations x4) | 0.10 | 6.4% | 0.2 | 0.51 | `if (0 <= orders[r_2])` |
| store.global*4 (est.: sampled iterations x4) | 0.07 | 4.2% | 0.2 | 0.38 | `if (0 <= orders[r_2])` |

### token_counts_kernel grid (1, 1, 1) block (128, 1, 1)

17 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 1.12 us, warp 0.47 us, nsys graphs 1.31 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.27 | 57.6% | 0.35 | 29.1% |
| store | 0.14 | 30.5% | 0.62 | 52.4% |
| load | 0.06 | 11.9% | 0.22 | 18.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.14 | 30.5% | 0.2 | 0.58 | `if (((int)threadIdx.x) == 0) \| sampled[((int64_t)((int)blockIdx.x))] = (((int)(prefill_len` |
| load.global | 0.06 | 11.9% | 0.2 | 0.22 | `int count = sampled[((int64_t)((int)blockIdx.x))];` |

### logits_rows_kernel grid (1, 1, 1) block (128, 1, 1)

17 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 1.15 us, warp 1.10 us, nsys graphs 2.05 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.51 | 46.4% | 0.50 | 43.4% |
| unattributed | 0.46 | 41.3% | 0.30 | 25.8% |
| store | 0.14 | 12.3% | 0.36 | 30.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.51 | 46.4% | 1.0 | 0.51 | `index = logits_indices[min(((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((in` |
| store.global | 0.14 | 12.3% | 1.0 | 0.35 | `if (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) < num_logits)` |

### prepare_pos_seq_lens_kernel grid (2, 1, 1) block (256, 1, 1)

17 launches, 16 warps; ranges, warp and last warp from the 2 recording CTAs' 16 warps; span 1.12 us, warp 0.76 us, nsys graphs 1.31 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.32 | 42.3% | 0.37 | 33.3% |
| load | 0.31 | 40.4% | 0.62 | 55.2% |
| store | 0.13 | 17.3% | 0.13 | 11.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.31 | 40.4% | 0.5 | 0.64 | `int computed = num_computed_tokens[idx_mapping[((int64_t)((int)blockIdx.x))]];` |
| store.global | 0.07 | 8.9% | 0.5 | 0.16 | `for (int it = 0; it < (((max_reqs + 255) - num_reqs) >> 8); ++it)` |
| store.global#2 | 0.06 | 8.4% | 0.5 | 0.13 | `if (((int)threadIdx.x) == 0)` |

### topk_log_softmax_kernel grid (6, 1, 1) block (1024, 1, 1)

1 launches, 192 warps; ranges, warp and last warp from the 6 recording CTAs' 192 warps; span 16.83 us, warp 16.08 us, nsys graphs 16.35 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 13.12 | 81.6% | 13.86 | 82.5% |
| unattributed, with loops' unsampled iterations | 1.58 | 9.8% | 1.66 | 9.9% |
| wait | 0.98 | 6.1% | 0.77 | 4.6% |
| compute | 0.39 | 2.5% | 0.45 | 2.7% |
| store | 0.00 | 0.0% | 0.06 | 0.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 6.63 | 41.3% | 1.0 | 6.94 | `if (0 < (vocab & 2047)) \| if (0 < (vocab & 8191))` |
| load.global#2 | 6.48 | 40.3% | 1.0 | 7.04 | `for (int step_1 = 0; step_1 < (vocab >> 11); ++step_1) \| for (int step_1 = 0; step_1 < (vo` |
| wait.sync#2 | 0.43 | 2.7% | 1.0 | 1.02 | `__syncthreads();` |
| wait.sync | 0.29 | 1.8% | 1.0 | 0.77 | `__syncthreads();` |
| compute.reduce#3 | 0.17 | 1.1% | 1.0 | 0.22 | `ties[0] = vllm_add(ties[0], __shfl_xor_sync((uint)4294967295, ties[0], 16, 32));` |
| compute.reduce#2 | 0.11 | 0.7% | 1.0 | 0.19 | `se[0] = vllm_add(se[0], __shfl_xor_sync((uint)4294967295, se[0], 16, 32));` |
| compute.reduce | 0.11 | 0.7% | 1.0 | 0.16 | `best[0] = vllm_max(best[0], __shfl_xor_sync((uint)4294967295, best[0], 16, 32));` |
| wait.sync#4 | 0.10 | 0.6% | 1.0 | 0.13 | `__syncthreads();` |
| wait.sync#5 | 0.08 | 0.5% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#3 | 0.08 | 0.5% | 1.0 | 0.10 | `__syncthreads();` |
| load.global#3 (sampled iterations) | 0.01 | 0.1% | 0.0 | 0.38 | `float value_32 = vllm_sub(vllm_shifted(logits[((((int64_t)((int)blockIdx.x)) * logits_stri` |
| store.global (sampled iterations) | 0.00 | 0.0% | 0.0 | 0.06 | `output[(((((int64_t)c) * (int64_t)1024) + (((int64_t)((int)blockIdx.x)) * ((int64_t)topk))` |

### mhc_collapse_column_kernel grid (25, 1, 1) block (128, 1, 1)

16 launches, 100 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 0.80 us, warp 0.78 us, nsys graphs 0.99 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.62 | 79.8% | 0.78 | 79.7% |
| load | 0.10 | 13.2% | 0.14 | 14.0% |
| store | 0.06 | 7.1% | 0.06 | 6.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.10 | 13.2% | 1.0 | 0.13 | `*(float4*)(mv + 0) = *(float4*)(mix + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int6` |
| store.global | 0.06 | 7.1% | 1.0 | 0.06 | `*(uint4*)(out + ((((int64_t)((int)blockIdx.x)) * (int64_t)1024) + (((int64_t)((int)threadI` |

### scatter_rows_gather_kernel grid (1, 1, 1) block (128, 1, 1)

17 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 0.90 us, warp 0.82 us, nsys graphs 1.82 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.54 | 66.0% | 0.38 | 41.7% |
| store | 0.15 | 18.4% | 0.41 | 45.0% |
| load | 0.13 | 15.5% | 0.12 | 13.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.15 | 18.4% | 1.0 | 0.42 | `if (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) < (num_reqs * width))` |
| load.global | 0.13 | 15.5% | 1.0 | 0.13 | `index = idx_mapping[min((((((int64_t)0 <= ((int64_t)width)) && ((int64_t)0 <= rmod_1)) \|\| ` |

### ced_copy_rows_kernel grid (50, 1, 1) block (128, 1, 1)

17 launches, 200 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 0.90 us, warp 0.83 us, nsys graphs 1.15 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.57 | 69.4% | 0.61 | 70.3% |
| unattributed | 0.25 | 30.6% | 0.26 | 29.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.57 | 69.4% | 1.0 | 0.61 | `for (int k = 0; k < 4; ++k) \| int64_t index = ced_ldg64((&(row_indices[((((int64_t)blockId` |

### mhc_collapse_kernel grid (50, 1, 1) block (128, 1, 1)

17 launches, 200 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 0.86 us, warp 0.76 us, nsys graphs 1.06 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.61 | 80.0% | 0.69 | 81.1% |
| load | 0.10 | 13.4% | 0.11 | 13.0% |
| store | 0.05 | 6.5% | 0.05 | 5.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.10 | 13.4% | 1.0 | 0.13 | `*(float4*)(mv + 0) = *(float4*)(mix + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int6` |
| store.global | 0.05 | 6.5% | 1.0 | 0.06 | `*(uint4*)(out + ((((int64_t)((int)blockIdx.x)) * (int64_t)1024) + (((int64_t)((int)threadI` |

### ced_requests_kernel grid (1, 1, 1) block (128, 1, 1)

17 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 0.83 us, warp 0.59 us, nsys graphs 1.15 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.30 | 51.4% | 0.33 | 38.4% |
| store | 0.18 | 29.7% | 0.42 | 49.0% |
| load | 0.11 | 18.9% | 0.11 | 12.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.18 | 29.7% | 1.0 | 0.42 | `if (((int)threadIdx.x) < 2) \| if (((int)threadIdx.x) <= 2)` |
| load.global | 0.11 | 18.9% | 1.0 | 0.13 | `uint v_ = tl::load_global_32((&(starts[1]))); \| uint v_ = tl::load_global_32((&(starts[min` |

### indexer_topk_merge_kernel grid (1, 6, 1) block (512, 1, 1)

1 launches, 96 warps; ranges, warp and last warp from the 6 recording CTAs' 96 warps; span 14.08 us, warp 10.23 us, nsys graphs 3.81 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 6.25 | 61.1% | 10.18 | 72.4% |
| unattributed | 2.77 | 27.1% | 3.04 | 21.6% |
| load | 1.13 | 11.0% | 0.70 | 5.0% |
| store | 0.07 | 0.7% | 0.13 | 0.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.12 | 11.0% | 1.0 | 5.47 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 8192);` |
| wait.sync#19*9 (est.: sampled iterations x9) | 0.74 | 7.2% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#20*9 (est.: sampled iterations x9) | 0.66 | 6.5% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#24*9 (est.: sampled iterations x9) | 0.65 | 6.4% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#23*9 (est.: sampled iterations x9) | 0.61 | 6.0% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#6*8 (est.: sampled iterations x8) | 0.58 | 5.7% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync#14*8 (est.: sampled iterations x8) | 0.55 | 5.4% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync#5*8 (est.: sampled iterations x8) | 0.55 | 5.4% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync#13*8 (est.: sampled iterations x8) | 0.53 | 5.1% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync | 0.39 | 3.8% | 1.0 | 4.96 | `__syncthreads();` |
| wait.sync#16 | 0.08 | 0.8% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#8 | 0.08 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#2 | 0.07 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#3 | 0.07 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#18 | 0.06 | 0.6% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#9 | 0.06 | 0.6% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#17 | 0.06 | 0.6% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#12 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#25 | 0.06 | 0.6% | 1.0 | 0.10 | `__syncthreads();` |
| store.global#2 | 0.06 | 0.6% | 0.9 | 0.10 | `Values[((((int64_t)((int)blockIdx.y)) * (int64_t)512) + ((int64_t)((int)threadIdx.x)))] = ` |
| wait.sync#7 | 0.06 | 0.6% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#15 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#11 | 0.06 | 0.5% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#22 | 0.05 | 0.5% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#4 | 0.05 | 0.5% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#10 | 0.05 | 0.5% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#21 | 0.05 | 0.5% | 1.0 | 0.06 | `__syncthreads();` |
| store.global | 0.01 | 0.1% | 0.1 | 0.13 | `Values[((((int64_t)((int)blockIdx.y)) * (int64_t)512) + ((int64_t)((int)threadIdx.x)))] = ` |
| load.global#2 | 0.01 | 0.1% | 0.1 | 0.10 | `if (((0 <= ((int*)sel_idx)[((int)threadIdx.x)]) && (((int*)sel_idx)[((int)threadIdx.x)] < ` |

### local_stats_kernel grid (6, 16, 1) block (128, 1, 1)

1 launches, 384 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 12.99 us, warp 12.34 us, nsys graphs 12.35 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 11.23 | 91.0% | 11.55 | 91.4% |
| unattributed | 0.77 | 6.2% | 0.83 | 6.6% |
| compute | 0.16 | 1.3% | 0.13 | 1.0% |
| wait | 0.11 | 0.9% | 0.03 | 0.3% |
| store | 0.07 | 0.6% | 0.10 | 0.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 11.23 | 91.0% | 1.0 | 11.55 | `if (expanded_local_pos[((int64_t)((int)blockIdx.x))] < num_speculative_steps)` |
| compute.reduce | 0.11 | 0.9% | 1.0 | 0.13 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| wait.sync | 0.11 | 0.9% | 1.0 | 0.51 | `__syncthreads();` |
| store.global | 0.07 | 0.6% | 1.0 | 0.13 | `if (((int)threadIdx.x) == 0)` |
| compute.reduce#2 | 0.05 | 0.4% | 1.0 | 0.06 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 2, 32);` |

### padding_rows_kernel grid (1, 1, 1) block (256, 1, 1)

33 launches, 8 warps; ranges, warp and last warp from the 1 recording CTAs' 8 warps; span 0.35 us, warp 0.35 us, nsys graphs 0.77 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.26 | 72.7% | 0.26 | 71.4% |
| store | 0.10 | 27.3% | 0.11 | 28.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.10 | 27.3% | 1.0 | 0.10 | `if (((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) < rows)` |

### moe_combine_kernel grid (5, 6, 1) block (128, 1, 1)

6 launches, 120 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 2.02 us, warp 1.50 us, nsys graphs 1.18 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.05 | 70.0% | 1.54 | 77.2% |
| unattributed | 0.40 | 26.3% | 0.40 | 20.1% |
| store | 0.06 | 3.7% | 0.05 | 2.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.05 | 70.0% | 1.0 | 1.54 | `for (int k = 0; k < 3; ++k) \| for (int k = 0; k < 6; ++k) \| if ((((int)blockIdx.y) < share` |
| store.global | 0.06 | 3.7% | 1.0 | 0.06 | `{` |

### indexer_topk_segment_kernel grid (2, 6, 1) block (512, 1, 1)

1 launches, 192 warps; ranges, warp and last warp from the 4 recording CTAs' 64 warps; span 11.78 us, warp 7.14 us, nsys graphs 0.83 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed, with loops' unsampled iterations | 3.77 | 52.9% | 4.83 | 41.0% |
| wait | 2.98 | 41.8% | 6.30 | 53.5% |
| load | 0.34 | 4.8% | 0.54 | 4.6% |
| store | 0.04 | 0.6% | 0.10 | 0.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync#19*9 (est.: sampled iterations x9) | 0.36 | 5.0% | 0.5 | 0.86 | `__syncthreads();` |
| wait.sync#14*8 (est.: sampled iterations x8) | 0.32 | 4.4% | 0.5 | 0.77 | `__syncthreads();` |
| wait.sync#20*9 (est.: sampled iterations x9) | 0.30 | 4.2% | 0.5 | 0.86 | `__syncthreads();` |
| wait.sync#24*9 (est.: sampled iterations x9) | 0.30 | 4.2% | 0.5 | 0.86 | `__syncthreads();` |
| wait.sync#23*9 (est.: sampled iterations x9) | 0.29 | 4.0% | 0.5 | 0.58 | `__syncthreads();` |
| wait.sync#6*8 (est.: sampled iterations x8) | 0.28 | 4.0% | 0.5 | 0.77 | `__syncthreads();` |
| wait.sync#13*8 (est.: sampled iterations x8) | 0.27 | 3.8% | 0.5 | 0.77 | `__syncthreads();` |
| wait.sync#5*8 (est.: sampled iterations x8) | 0.27 | 3.8% | 0.5 | 0.77 | `__syncthreads();` |
| load.global | 0.21 | 2.9% | 1.0 | 0.22 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 4096); \| i` |
| load.global#2 | 0.14 | 1.9% | 0.5 | 0.32 | `for (int i = 0; i < 8; ++i)` |
| wait.sync (sampled iterations) | 0.05 | 0.7% | 0.5 | 0.16 | `__syncthreads();` |
| wait.sync#25 | 0.04 | 0.6% | 0.5 | 0.16 | `__syncthreads();` |
| store.global | 0.04 | 0.6% | 0.5 | 0.10 | `for (int i_3 = 0; i_3 < 4; ++i_3) \| if (((int)threadIdx.x) < total) \| if (((int)threadIdx.` |
| wait.sync#18 | 0.04 | 0.6% | 0.5 | 0.13 | `__syncthreads();` |
| wait.sync#8 | 0.04 | 0.6% | 0.5 | 0.13 | `__syncthreads();` |
| wait.sync#2 | 0.04 | 0.5% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#9 | 0.04 | 0.5% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#3 | 0.03 | 0.5% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#17 | 0.03 | 0.5% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#12 | 0.03 | 0.5% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#16 | 0.03 | 0.5% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#22 | 0.03 | 0.4% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#15 | 0.03 | 0.4% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#4 | 0.03 | 0.4% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#10 | 0.03 | 0.4% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#7 | 0.03 | 0.4% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#11 | 0.03 | 0.4% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#21 | 0.03 | 0.4% | 0.5 | 0.06 | `__syncthreads();` |

### scatter_rows_kernel grid (1, 1, 1) block (128, 1, 1)

17 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 0.64 us, warp 0.59 us, nsys graphs 0.99 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.38 | 63.5% | 0.38 | 59.5% |
| load | 0.13 | 21.6% | 0.12 | 18.1% |
| store | 0.09 | 14.9% | 0.14 | 22.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.13 | 21.6% | 1.0 | 0.13 | `index = idx_mapping[min((((((int64_t)0 <= ((int64_t)width)) && ((int64_t)0 <= rmod_1)) \|\| ` |
| store.global | 0.09 | 14.9% | 1.0 | 0.16 | `if (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) < (num_reqs * width))` |

### rope_kernel grid (6, 1, 1) block (128, 1, 1)

6 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 1.38 us, warp 1.31 us, nsys graphs 1.50 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.78 | 59.8% | 1.30 | 72.3% |
| load | 0.49 | 37.0% | 0.45 | 25.0% |
| store | 0.04 | 3.2% | 0.05 | 2.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.49 | 37.0% | 1.0 | 0.50 | `if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < tokens)) \| int64_t position ` |
| store.global | 0.04 | 3.2% | 1.0 | 0.06 | `*(uint4*)(x + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + ((((int64_t)((int)thread` |

### ced_copy_rows_kernel grid (25, 1, 1) block (128, 1, 1)

16 launches, 100 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 0.67 us, warp 0.61 us, nsys graphs 1.15 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.36 | 59.8% | 0.39 | 59.6% |
| unattributed | 0.25 | 40.2% | 0.27 | 40.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.36 | 59.8% | 1.0 | 0.42 | `for (int k = 0; k < 4; ++k) \| int64_t index = ced_ldg64((&(row_indices[((((int64_t)blockId` |

### mhc_collapse_kernel grid (30, 1, 1) block (128, 1, 1)

3 launches, 120 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 3.68 us, warp 3.67 us, nsys graphs 2.27 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 3.53 | 96.0% | 3.16 | 95.2% |
| load | 0.10 | 2.6% | 0.10 | 2.9% |
| store | 0.05 | 1.4% | 0.06 | 1.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.10 | 2.6% | 1.0 | 0.10 | `*(float4*)(mv + 0) = *(float4*)(mix + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int6` |
| store.global | 0.05 | 1.4% | 1.0 | 0.06 | `*(uint4*)(out + ((((int64_t)((int)blockIdx.x)) * (int64_t)1024) + (((int64_t)((int)threadI` |

### metadata_chunk_kernel grid (6, 1, 1) block (128, 1, 1)

2 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 4.26 us, warp 3.01 us, nsys graphs 4.46 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.57 | 52.0% | 2.64 | 62.0% |
| unattributed | 1.37 | 45.5% | 1.49 | 35.0% |
| store | 0.07 | 2.5% | 0.13 | 3.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.37 | 45.5% | 1.0 | 1.87 | `if ((((offset + ((int)blockIdx.x)) < src_rows) && (0 <= (offset + ((int)blockIdx.x)))))` |
| load.global#2 | 0.20 | 6.6% | 0.2 | 0.80 | `if (((0 <= req) & ((int64_t)0 <= pos)))` |
| store.global | 0.05 | 1.8% | 1.0 | 0.06 | `swa[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = con` |
| store.global#2 | 0.01 | 0.4% | 0.2 | 0.06 | `lengths[((int64_t)((int)blockIdx.x))] = ((int)max(info[1], (int64_t)0));` |
| store.global#3 | 0.01 | 0.3% | 0.2 | 0.05 | `top_lengths[((int64_t)((int)blockIdx.x))] = condval_12; \| top_lengths[((int64_t)((int)bloc` |

### token_ranks_kernel grid (6, 1, 1) block (256, 1, 1)

1 launches, 48 warps; ranges, warp and last warp from the 6 recording CTAs' 48 warps; span 7.97 us, warp 7.72 us, nsys graphs 7.87 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 6.95 | 90.0% | 7.20 | 90.7% |
| unattributed | 0.37 | 4.8% | 0.32 | 4.0% |
| compute | 0.36 | 4.7% | 0.35 | 4.4% |
| store | 0.04 | 0.6% | 0.06 | 0.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 6.95 | 90.0% | 1.0 | 7.36 | `if ((((int64_t)0 <= token_ids[((int64_t)((int)blockIdx.x))]) && (token_ids[((int64_t)((int` |
| compute.reduce | 0.36 | 4.7% | 1.0 | 0.64 | `total[0] = tl::AllReduce<tl::SumOp, 256, 1, 0, tl::NamedBarrier<256>>::run(total[0], (&(wo` |
| store.global | 0.04 | 0.6% | 1.0 | 0.06 | `output[((int64_t)((int)blockIdx.x))] = ((int64_t)total[0]);` |

### indexer_score_kernel grid (128, 6, 1) block (128, 1, 1)

1 launches, 3072 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 7.84 us, warp 3.13 us, nsys graphs 2.27 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 2.16 | 69.0% | 5.15 | 78.5% |
| unattributed, with loops' unsampled iterations | 0.73 | 23.3% | 1.02 | 15.6% |
| wait | 0.16 | 5.1% | 0.19 | 2.9% |
| compute | 0.06 | 1.9% | 0.13 | 2.0% |
| store | 0.02 | 0.8% | 0.06 | 1.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 (sampled iterations) | 1.69 | 54.0% | 0.5 | 4.64 | `for (int i_1 = 0; i_1 < 32; ++i_1)` |
| load.global | 0.38 | 12.0% | 1.0 | 0.64 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 4096); \| i` |
| load.global#3 (sampled iterations) | 0.09 | 2.9% | 0.5 | 0.35 | `if (((int)threadIdx.x) < 64)` |
| wait.sync#2 (sampled iterations) | 0.08 | 2.4% | 0.5 | 0.32 | `__syncthreads();` |
| compute.mma (sampled iterations) | 0.06 | 1.9% | 0.5 | 0.13 | `for (int ki = 0; ki < 2; ++ki)` |
| wait.sync#3 (sampled iterations) | 0.03 | 1.0% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync (sampled iterations) | 0.03 | 0.8% | 0.5 | 0.06 | `__syncthreads();` |
| store.global (sampled iterations) | 0.02 | 0.8% | 0.5 | 0.06 | `if ((((int)threadIdx.x) >> 6) == 0)` |
| wait.sync#4 (sampled iterations) | 0.02 | 0.8% | 0.5 | 0.06 | `__syncthreads();` |

### indexer_mxfp4_quantize_kernel grid (6, 1, 1) block (128, 1, 1)

2 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 3.76 us, warp 3.68 us, nsys graphs 2.91 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 2.27 | 61.8% | 2.35 | 62.6% |
| store | 1.11 | 30.1% | 1.07 | 28.5% |
| unattributed | 0.30 | 8.1% | 0.34 | 8.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2*16 (est.: sampled iterations x16) | 1.26 | 34.2% | 1.0 | 1.54 | `float v__2 = __fmul_rn(((float)X[(((((int64_t)((int)blockIdx.x)) * (int64_t)4096) + (((int` |
| store.global*16 (est.: sampled iterations x16) | 1.07 | 29.0% | 1.0 | 1.54 | `Pool[(((((slot & (int64_t)127) * (int64_t)64) + ((((int64_t)((int)threadIdx.x)) & (int64_t` |
| load.global | 1.01 | 27.6% | 1.0 | 1.09 | `for (int i = 0; i < 32; ++i)` |
| store.global#2 | 0.04 | 1.1% | 1.0 | 0.06 | `Pool[(((((slot & (int64_t)127) * (int64_t)4) + ((slot >> (int64_t)7) * ((int64_t)pool_stri` |

### mhc_post_kernel grid (30, 1, 1) block (128, 1, 1)

4 launches, 120 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 1.55 us, warp 1.31 us, nsys graphs 1.34 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.16 | 88.4% | 1.42 | 91.3% |
| load | 0.10 | 7.9% | 0.10 | 6.2% |
| store | 0.05 | 3.7% | 0.04 | 2.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.10 | 7.9% | 1.0 | 0.13 | `*(float4*)(pm + 0) = *(float4*)(post + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int` |
| store.global | 0.05 | 3.7% | 1.0 | 0.06 | `for (int n_1 = 0; n_1 < 4; ++n_1)` |

### activation_swiglu_clamp_kernel grid (1, 6, 1) block (128, 1, 1)

6 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 0.93 us, warp 0.73 us, nsys graphs 1.86 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.63 | 86.1% | 0.95 | 86.9% |
| load | 0.07 | 9.5% | 0.10 | 8.7% |
| store | 0.03 | 4.4% | 0.05 | 4.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.07 | 9.5% | 0.8 | 0.10 | `*(uint4*)(gate + 0) = *(uint4*)(X + ((((int64_t)((int)blockIdx.y)) * (int64_t)1152) + (((i` |
| store.global | 0.03 | 4.4% | 0.8 | 0.06 | `*(uint4*)(Y + ((((int64_t)((int)blockIdx.y)) * (int64_t)576) + (((int64_t)((int)threadIdx.` |

### indexer_score_kernel grid (64, 6, 1) block (128, 1, 1)

1 launches, 1536 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 4.06 us, warp 2.37 us, nsys graphs 1.70 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.12 | 47.3% | 1.92 | 47.6% |
| unattributed, with loops' unsampled iterations | 0.90 | 38.1% | 1.57 | 38.9% |
| wait | 0.24 | 10.1% | 0.38 | 9.5% |
| compute | 0.06 | 2.6% | 0.13 | 3.2% |
| store | 0.04 | 1.9% | 0.03 | 0.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 (sampled iterations) | 0.55 | 23.0% | 0.5 | 1.18 | `for (int i_1 = 0; i_1 < 32; ++i_1)` |
| load.global | 0.51 | 21.6% | 1.0 | 0.80 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 4096); \| i` |
| wait.sync#2 (sampled iterations) | 0.11 | 4.5% | 0.5 | 0.32 | `__syncthreads();` |
| wait.sync#4 (sampled iterations) | 0.09 | 3.7% | 0.5 | 0.26 | `__syncthreads();` |
| load.global#3 (sampled iterations) | 0.06 | 2.7% | 0.5 | 0.35 | `if (((int)threadIdx.x) < 64)` |
| compute.mma (sampled iterations) | 0.06 | 2.6% | 0.5 | 0.13 | `for (int ki = 0; ki < 2; ++ki)` |
| store.global (sampled iterations) | 0.04 | 1.9% | 0.5 | 0.16 | `if ((((int)threadIdx.x) >> 6) == 0)` |
| wait.sync (sampled iterations) | 0.03 | 1.1% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#3 (sampled iterations) | 0.02 | 0.8% | 0.5 | 0.06 | `__syncthreads();` |

### gemm_splitk_reduce_kernel grid (1, 6, 1) block (128, 1, 1)

4 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 0.77 us, warp 0.74 us, nsys graphs 0.93 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.39 | 52.3% | 0.39 | 50.0% |
| load | 0.31 | 41.4% | 0.33 | 41.8% |
| store | 0.05 | 6.2% | 0.06 | 8.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.31 | 41.4% | 1.0 | 0.32 | `acc[0] = P[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x))` |
| store.global | 0.05 | 6.2% | 1.0 | 0.06 | `C[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x)) * (int64` |

### metadata_pages_kernel grid (6, 1, 1) block (128, 1, 1)

2 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 1.33 us, warp 0.97 us, nsys graphs 1.06 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.74 | 76.3% | 0.80 | 60.2% |
| load | 0.14 | 14.9% | 0.19 | 14.5% |
| store | 0.08 | 8.8% | 0.34 | 25.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.13 | 13.3% | 1.0 | 0.13 | `if ((((offset + ((int)blockIdx.x)) < src_rows) && (0 <= (offset + ((int)blockIdx.x)))))` |
| store.global | 0.08 | 8.8% | 0.2 | 0.35 | `if (((int)threadIdx.x) < 32)` |
| load.global#2 | 0.02 | 1.7% | 0.2 | 0.06 | `if (((req < table_rows) && (((int)threadIdx.x) < table_cols)))` |

### norm_forward_and_per_token_cast_kernel_kernel grid (6, 1, 1) block (128, 1, 1)

1 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 2.59 us, warp 2.55 us, nsys graphs 1.89 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 2.46 | 96.4% | 2.66 | 103.8% |
| load | 0.47 | 18.3% | 0.64 | 25.0% |
| compute | 0.13 | 5.2% | 0.13 | 5.0% |
| wait | 0.06 | 2.3% | 0.06 | 2.5% |
| unattributed | 0.00 | 0.0% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*5 (est.: sampled iterations x5) | 2.19 | 86.1% | 1.0 | 2.24 | `for (int j_1 = 0; j_1 < 2; ++j_1)` |
| load.global*5 (est.: sampled iterations x5) | 0.26 | 10.2% | 1.0 | 0.32 | `*(uint4*)(x_local + (vi * 8)) = *(uint4*)(x + ((((((int64_t)vi) * (int64_t)1024) + (((int6` |
| load.global#2*5 (est.: sampled iterations x5) | 0.21 | 8.1% | 1.0 | 0.32 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 1024) + (((int)threadIdx.x) * 8)` |
| store.global*5 (est.: sampled iterations x5) | 0.21 | 8.1% | 1.0 | 0.32 | `for (int j = 0; j < 4; ++j)` |
| compute.reduce | 0.13 | 5.2% | 1.0 | 0.16 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| wait.sync | 0.06 | 2.3% | 1.0 | 0.10 | `__syncthreads();` |
| store.global#2 | 0.04 | 1.6% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| store.global#3 | 0.02 | 0.6% | 0.2 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |

### rope_kernel grid (6, 1, 1) block (256, 1, 1)

2 launches, 48 warps; ranges, warp and last warp from the 6 recording CTAs' 48 warps; span 1.01 us, warp 0.92 us, nsys graphs 1.12 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.61 | 66.8% | 0.72 | 71.4% |
| load | 0.26 | 28.7% | 0.26 | 25.4% |
| store | 0.04 | 4.5% | 0.03 | 3.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.26 | 28.7% | 1.0 | 0.27 | `if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < tokens)) \| int64_t position ` |
| store.global | 0.04 | 4.5% | 1.0 | 0.06 | `*(uint4*)(x + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + ((((int64_t)((int)thread` |

### mhc_collapse_column_kernel grid (30, 1, 1) block (128, 1, 1)

1 launches, 120 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 0.74 us, warp 0.65 us, nsys graphs 0.90 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.49 | 76.0% | 0.54 | 77.3% |
| load | 0.10 | 15.1% | 0.10 | 13.6% |
| store | 0.06 | 8.9% | 0.06 | 9.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.10 | 15.1% | 1.0 | 0.13 | `*(float4*)(mv + 0) = *(float4*)(mix + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int6` |
| store.global | 0.06 | 8.9% | 1.0 | 0.06 | `*(uint4*)(out + ((((int64_t)((int)blockIdx.x)) * (int64_t)1024) + (((int64_t)((int)threadI` |

### ced_copy_rows_kernel grid (30, 1, 1) block (128, 1, 1)

1 launches, 120 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 0.64 us, warp 0.60 us, nsys graphs 1.02 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.36 | 59.5% | 0.38 | 60.0% |
| unattributed | 0.24 | 40.5% | 0.26 | 40.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.36 | 59.5% | 1.0 | 0.38 | `for (int k = 0; k < 4; ++k) \| int64_t index = ced_ldg64((&(row_indices[((((int64_t)blockId` |
=== timeline and tables: exit 0 in 22 s
