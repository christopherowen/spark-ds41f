=== timeline and tables
trace /c/iket/step-tp4-c1-20261026T185743Z/trace/iket/pid_0x1854/iket.decoded_results.json: 12184 launches, graphLaunches dict of 124

## Timeline: 7754 launches in 797333.4 us; the GPU is busy 262894.4 us (33%) and idle 534439.1 us in 5065 gaps; gaps over 200 us split it into 512 segments

The 15 of 512 segments with the most busy time:

| segment | launches | wall us | busy us | busy | first kernel |
|---:|---:|---:|---:|---:|---|
| 52 | 134 | 7502.9 | 6538.2 | 87% | rope_kernel |
| 81 | 116 | 6618.4 | 6539.8 | 99% | mhc_project_streams_kernel |
| 94 | 120 | 6860.4 | 6461.7 | 94% | metadata_builder_kernel |
| 122 | 116 | 6568.7 | 6491.4 | 99% | mhc_project_streams_kernel |
| 143 | 120 | 6955.2 | 6554.3 | 94% | metadata_builder_kernel |
| 166 | 120 | 6864.1 | 6482.9 | 94% | metadata_builder_kernel |
| 189 | 120 | 6827.9 | 6459.4 | 95% | metadata_builder_kernel |
| 213 | 120 | 6837.2 | 6468.5 | 95% | metadata_builder_kernel |
| 244 | 116 | 6551.0 | 6468.7 | 99% | mhc_project_streams_kernel |
| 279 | 117 | 6975.6 | 6862.2 | 98% | padding_rows_kernel |
| 308 | 116 | 6559.6 | 6472.8 | 99% | mhc_project_streams_kernel |
| 366 | 116 | 6543.8 | 6458.7 | 99% | mhc_project_streams_kernel |
| 450 | 118 | 6991.3 | 6718.1 | 96% | metadata_builder_kernel |
| 476 | 120 | 6850.1 | 6472.4 | 94% | metadata_builder_kernel |
| 509 | 119 | 6736.0 | 6465.7 | 96% | metadata_builder_kernel |

| idle us | after | before |
|---:|---|---|
| 117086.6 | resample_kernel | insert_kernel |
| 80599.5 | scatter_rows_kernel | padding_rows_kernel |
| 3299.4 | dspark_confidence_kernel | scatter_rows_gather_kernel |
| 3236.7 | dspark_confidence_kernel | scatter_rows_gather_kernel |
| 3164.5 | norm_forward_and_per_token_cast_kernel_kernel | sink_flags_kernel |
| 3024.3 | dspark_confidence_kernel | scatter_rows_gather_kernel |
| 2985.7 | norm_forward_and_per_token_cast_kernel_kernel | sink_flags_kernel |
| 2969.5 | norm_forward_and_per_token_cast_kernel_kernel | sink_flags_kernel |
| 2945.7 | dspark_confidence_kernel | scatter_rows_gather_kernel |
| 2899.7 | dspark_confidence_kernel | scatter_rows_gather_kernel |

## Engine steps

| step | kind | launches | wall us | busy us | busy | most span |
|---:|---|---:|---:|---:|---:|---|
| 0 | prefill | 413 | 48361.4 | 16355.5 | 34% | vocab_packed_projection_kernel 8122, gemm_mxfp8_kernel 2590, moe_grouped_fp8_fp4_gemm_kernel 2038 |
| 1 | decode | 432 | 146708.2 | 14351.3 | 10% | vocab_packed_projection_kernel 8178, l2_prefetch_kernel 1687, gemm_mxfp8_kernel 1584 |
| 2 | decode | 432 | 107475.8 | 14387.9 | 13% | vocab_packed_projection_kernel 8058, l2_prefetch_kernel 1963, gemm_mxfp8_kernel 1716 |
| 3 | decode | 432 | 34085.5 | 14531.3 | 43% | vocab_packed_projection_kernel 8135, l2_prefetch_kernel 2112, gemm_mxfp8_kernel 1749 |
| 4 | decode | 432 | 29553.6 | 14538.0 | 49% | vocab_packed_projection_kernel 8119, gemm_mxfp8_kernel 1571, l2_prefetch_kernel 1520 |
| 5 | decode | 432 | 30351.8 | 14435.3 | 48% | vocab_packed_projection_kernel 8081, gemm_mxfp8_kernel 1552, l2_prefetch_kernel 1506 |
| 6 | decode | 432 | 29320.1 | 14749.2 | 50% | vocab_packed_projection_kernel 8048, gemm_mxfp8_kernel 1636, l2_prefetch_kernel 1537 |
| 7 | decode | 432 | 30430.8 | 14391.5 | 47% | vocab_packed_projection_kernel 8041, gemm_mxfp8_kernel 1540, l2_prefetch_kernel 1520 |
| 8 | decode | 432 | 34808.8 | 14426.0 | 41% | vocab_packed_projection_kernel 8055, gemm_mxfp8_kernel 1556, l2_prefetch_kernel 1521 |
| 9 | decode | 432 | 35862.2 | 14783.9 | 41% | vocab_packed_projection_kernel 8380, l2_prefetch_kernel 1564, gemm_mxfp8_kernel 1517 |
| 10 | decode | 432 | 33683.1 | 14425.1 | 43% | vocab_packed_projection_kernel 8069, gemm_mxfp8_kernel 1549, l2_prefetch_kernel 1514 |
| 11 | decode | 432 | 36478.5 | 14452.3 | 40% | vocab_packed_projection_kernel 8103, gemm_mxfp8_kernel 1514, l2_prefetch_kernel 1508 |
| 12 | decode | 432 | 30903.0 | 14424.4 | 47% | vocab_packed_projection_kernel 8060, gemm_mxfp8_kernel 1559, l2_prefetch_kernel 1511 |
| 13 | decode | 432 | 34723.6 | 14664.9 | 42% | vocab_packed_projection_kernel 8043, gemm_mxfp8_kernel 1678, l2_prefetch_kernel 1522 |
| 14 | decode | 432 | 29340.4 | 14470.3 | 49% | vocab_packed_projection_kernel 8110, gemm_mxfp8_kernel 1568, l2_prefetch_kernel 1520 |
| 15 | decode | 432 | 34495.0 | 14699.2 | 43% | vocab_packed_projection_kernel 8239, gemm_mxfp8_kernel 1562, l2_prefetch_kernel 1488 |
| 16 | decode | 432 | 30946.8 | 14411.4 | 47% | vocab_packed_projection_kernel 8053, gemm_mxfp8_kernel 1550, l2_prefetch_kernel 1510 |
| 17 | decode | 429 | 34267.7 | 14396.8 | 42% | vocab_packed_projection_kernel 8051, gemm_mxfp8_kernel 1550, l2_prefetch_kernel 1517 |

## Overview over 17 decode steps: 7341 launches of 142 kernel shapes, spans summing to 272920.1 us (16054.1 a step)

| kernel | grid | launches a step | span us a step | share | span us a launch | last warp: wait | load | compute | store | unattributed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vocab_packed_projection_kernel | (1, 4040, 1) | 2 | 8107.2 | 50.5% | 4029.70 | 65% | 29% | 11% | 0% | 0% |
| l2_prefetch_kernel | (2, 1, 1) | 25 | 1589.4 | 9.9% | 63.39 | 0% | 100% | 0% | 0% | 0% |
| gemm_bf16_kernel | (2020, 1, 1) | 5 | 1330.7 | 8.3% | 264.74 | 89% | 0% | 5% | 1% | 5% |
| moe_grouped_fp8_fp4_gemm_kernel | (9, 30, 1) | 8.64706 | 816.9 | 5.1% | 118.50 | 46% | 20% | 29% | 1% | 4% |
| moe_grouped_fp8_fp4_gemm_kernel | (40, 30, 1) | 8.64706 | 495.2 | 3.1% | 65.86 | 14% | 20% | 0% | 0% | 66% |
| gemm_mxfp8_kernel | (80, 1, 1) | 1 | 409.0 | 2.5% | 409.86 | 75% | 4% | 4% | 0% | 16% |
| gemm_mxfp8_kernel | (64, 1, 1) | 11 | 367.5 | 2.3% | 25.92 | 29% | 10% | 15% | 1% | 45% |
| gemm_mxfp8_kernel | (28, 1, 1) | 9 | 309.8 | 1.9% | 35.10 | 38% | 11% | 12% | 0% | 39% |
| mhc_project_streams_kernel | (40, 1, 1) | 18 | 292.5 | 1.8% | 15.52 | 18% | 31% | 17% | 3% | 30% |
| gemm_fp8w_bf16_partials_kernel | (32, 4, 1) | 9 | 219.3 | 1.4% | 19.26 | 45% | 18% | 11% | 0% | 25% |
| gemm_mxfp8_kernel | (36, 1, 1) | 9 | 208.4 | 1.3% | 23.14 | 45% | 16% | 16% | 0% | 23% |
| gemm_mxfp8_kernel | (40, 1, 1) | 9 | 187.5 | 1.2% | 18.53 | 59% | 15% | 16% | 0% | 10% |
| gemm_bf16_partials_kernel | (6, 8, 1) | 6 | 142.7 | 0.9% | 24.24 | 39% | 0% | 31% | 2% | 28% |
| mhc_finalize_kernel | (5, 1, 1) | 11.2941 | 136.6 | 0.9% | 12.24 | 6% | 16% | 9% | 20% | 50% |
| gemm_bf16_splitk_kernel | (16, 5, 1) | 2 | 115.6 | 0.7% | 57.60 | 116% | 8% | 4% | 0% | 0% |
| gemm_mxfp8_kernel | (40, 1, 1) | 9 | 103.1 | 0.6% | 7.52 | 77% | 12% | 14% | 1% | 0% |
| sparse_mla_kernel | (5, 1, 5) | 3.76471 | 90.8 | 0.6% | 16.59 | 3% | 42% | 4% | 2% | 49% |
| mhc_finalize_kernel | (10, 1, 1) | 6 | 64.8 | 0.4% | 11.33 | 7% | 15% | 10% | 15% | 52% |
| sparse_mla_kernel | (2, 1, 10) | 3 | 62.3 | 0.4% | 21.92 | 13% | 25% | 23% | 2% | 38% |
| cache_writer_kernel | (5, 1, 1) | 7.52941 | 44.3 | 0.3% | 5.12 | 0% | 9% | 3% | 43% | 45% |

| wait | kernel | grid | last-warp us a step | a launch | first statement |
|---|---|---|---:|---:|---|
| wait.cp_async*118 | gemm_mxfp8_kernel | (80, 1, 1) | 230.1 | 230.11 | `tl::cp_async_wait<2>();` |
| wait.cp_async*13 | gemm_bf16_splitk_kernel | (16, 5, 1) | 81.1 | 40.54 | `tl::cp_async_wait<2>();` |
| wait.mbarrier*40 | moe_grouped_fp8_fp4_gemm_kernel | (9, 30, 1) | 77.7 | 8.99 | `loaded[(ko % 3)].wait(((ko % 6) / 3));` |
| wait.sync#2*118 | gemm_mxfp8_kernel | (80, 1, 1) | 68.4 | 68.41 | `__syncthreads();` |
| wait.cp_async*14 | gemm_mxfp8_kernel | (40, 1, 1) | 58.2 | 6.47 | `tl::cp_async_wait<2>();` |
| wait.cp_async*19 | gemm_mxfp8_kernel | (28, 1, 1) | 57.0 | 6.34 | `tl::cp_async_wait<1>();` |
| wait.cp_async*40 | moe_grouped_fp8_fp4_gemm_kernel | (9, 30, 1) | 49.2 | 5.69 | `tl::cp_async_wait<2>();` |
| wait.cp_async*8 | gemm_mxfp8_kernel | (40, 1, 1) | 48.7 | 5.41 | `tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| wait.sync*4 | mhc_project_streams_kernel | (40, 1, 1) | 46.6 | 2.59 | `__syncthreads();` |
| wait.sync#2*19 | gemm_mxfp8_kernel | (28, 1, 1) | 45.3 | 5.04 | `__syncthreads();` |
| wait.cp_async*18 | gemm_mxfp8_kernel | (36, 1, 1) | 44.0 | 4.89 | `tl::cp_async_wait<2>();` |
| wait.sync#2*18 | gemm_mxfp8_kernel | (36, 1, 1) | 34.0 | 3.77 | `__syncthreads();` |
| wait.sync#2*14 | gemm_mxfp8_kernel | (40, 1, 1) | 32.7 | 3.63 | `__syncthreads();` |
| wait.cp_async*17 | vocab_packed_projection_kernel | (1, 4040, 1) | 27.8 | 13.92 | `tl::cp_async_wait<6>();` |
| wait.mbarrier#2 | gemm_bf16_kernel | (2020, 1, 1) | 26.8 | 5.37 | `mbarrier[0].wait(0);` |
| wait.sync#2*17 | vocab_packed_projection_kernel | (1, 4040, 1) | 23.8 | 11.89 | `__syncthreads();` |
| wait.cp_async*14 | gemm_fp8w_bf16_partials_kernel | (32, 4, 1) | 21.3 | 2.36 | `tl::cp_async_wait<3>();` |
| wait.sync#5*40 | moe_grouped_fp8_fp4_gemm_kernel | (9, 30, 1) | 20.9 | 2.42 | `__syncthreads();` |
| wait.sync#6*40 | moe_grouped_fp8_fp4_gemm_kernel | (9, 30, 1) | 20.7 | 2.39 | `__syncthreads();` |
| wait.sync#4*40 | moe_grouped_fp8_fp4_gemm_kernel | (9, 30, 1) | 20.5 | 2.37 | `__syncthreads();` |

### vocab_packed_projection_kernel grid (1, 4040, 1) block (128, 1, 1)

34 launches, 16160 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 4029.70 us, warp 49.34 us, nsys graphs 4046.53 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 31.70 | 64.3% | 31.25 | 65.0% |
| load | 9.72 | 19.7% | 13.81 | 28.7% |
| compute | 5.05 | 10.2% | 5.08 | 10.6% |
| unattributed | 2.82 | 5.7% | 0.00 | 0.0% |
| store | 0.05 | 0.1% | 0.05 | 0.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.cp_async*17 (est.: sampled iterations x17) | 15.04 | 30.5% | 1.0 | 42.98 | `tl::cp_async_wait<6>();` |
| wait.sync#2*17 (est.: sampled iterations x17) | 11.24 | 22.8% | 1.0 | 35.09 | `__syncthreads();` |
| load.global | 7.20 | 14.6% | 1.0 | 13.94 | `int start = row_offsets[(((int)blockIdx.y) * 16)]; \| int start = row_offsets[(((int)blockI` |
| compute.mma*17 (est.: sampled iterations x17) | 4.35 | 8.8% | 1.0 | 4.35 | `{` |
| load.cp_async#2*17 (est.: sampled iterations x17) | 2.24 | 4.5% | 1.0 | 2.72 | `for (int i_6 = 0; i_6 < 8; ++i_6) \| for (int i_9 = 0; i_9 < 4; ++i_9)` |
| wait.sync#3*17 (est.: sampled iterations x17) | 1.09 | 2.2% | 1.0 | 1.63 | `__syncthreads();` |
| wait.sync*17 (est.: sampled iterations x17) | 1.05 | 2.1% | 1.0 | 1.09 | `__syncthreads();` |
| wait.cp_async#2*17 (est.: sampled iterations x17) | 0.99 | 2.0% | 1.0 | 1.09 | `tl::cp_async_wait<7>();` |
| wait.cp_async#3 | 0.52 | 1.0% | 1.0 | 1.92 | `tl::cp_async_wait<4>();` |
| wait.sync#4 | 0.36 | 0.7% | 1.0 | 1.25 | `__syncthreads();` |
| wait.sync#6 | 0.35 | 0.7% | 1.0 | 1.34 | `__syncthreads();` |
| wait.cp_async#5 | 0.32 | 0.6% | 1.0 | 1.60 | `tl::cp_async_wait<2>();` |
| load.cp_async | 0.28 | 0.6% | 1.0 | 0.32 | `for (int i = 0; i < 4; ++i) \| for (int i = 0; i < 8; ++i) \| for (int i_1 = 0; i_1 < 4; ++i` |
| wait.sync#8 | 0.25 | 0.5% | 1.0 | 1.01 | `__syncthreads();` |
| compute.mma#2 | 0.24 | 0.5% | 1.0 | 0.26 | `{` |
| compute.mma#3 | 0.23 | 0.5% | 1.0 | 0.26 | `{` |
| compute.mma#4 | 0.23 | 0.5% | 1.0 | 0.26 | `{` |
| wait.cp_async#7 | 0.21 | 0.4% | 1.0 | 1.06 | `tl::cp_async_wait<0>();` |
| wait.sync#5 | 0.06 | 0.1% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#7 | 0.05 | 0.1% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#9 | 0.05 | 0.1% | 1.0 | 0.06 | `__syncthreads();` |
| store.global | 0.05 | 0.1% | 1.0 | 0.06 | `for (int i_13 = 0; i_13 < 2; ++i_13) \| for (int i_14 = 0; i_14 < 4; ++i_14) \| for (int i_9` |
| wait.cp_async#4 | 0.04 | 0.1% | 1.0 | 0.06 | `tl::cp_async_wait<5>();` |
| wait.cp_async#6 | 0.04 | 0.1% | 1.0 | 0.06 | `tl::cp_async_wait<3>();` |
| wait.cp_async#8 | 0.04 | 0.1% | 1.0 | 0.06 | `tl::cp_async_wait<1>();` |

### l2_prefetch_kernel grid (2, 1, 1) block (128, 1, 1)

425 launches, 8 warps; ranges, warp and last warp from the 2 recording CTAs' 8 warps; span 63.39 us, warp 61.85 us, nsys graphs 64.37 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 61.44 | 99.3% | 63.26 | 99.5% |
| unattributed | 0.41 | 0.7% | 0.32 | 0.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 61.44 | 99.3% | 1.0 | 63.07 | `for (int s = 0; s < (nseg2 >> 1); ++s)` |

### gemm_bf16_kernel grid (2020, 1, 1) block (256, 1, 1)

85 launches, 16160 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 264.74 us, warp 3.63 us, nsys graphs 268.56 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 3.05 | 84.1% | 5.45 | 89.0% |
| unattributed | 0.36 | 10.0% | 0.32 | 5.2% |
| compute | 0.14 | 3.9% | 0.29 | 4.7% |
| load | 0.04 | 1.0% | 0.00 | 0.0% |
| store | 0.04 | 1.0% | 0.07 | 1.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.mbarrier#2 | 2.94 | 81.1% | 0.5 | 7.14 | `mbarrier[0].wait(0);` |
| compute.mma | 0.14 | 3.9% | 0.5 | 0.29 | `{` |
| wait.sync | 0.08 | 2.1% | 1.0 | 0.10 | `__syncthreads();` |
| load.tma | 0.04 | 1.0% | 0.5 | 0.13 | `if (tl::tl_shuffle_elect<128>())` |
| store.global | 0.04 | 1.0% | 0.5 | 0.10 | `for (int i_3 = 0; i_3 < 4; ++i_3) \| for (int i_4 = 0; i_4 < 8; ++i_4) \| for (int i_6 = 0; ` |
| wait.mbarrier | 0.03 | 0.8% | 0.5 | 0.06 | `mbarrier[2].wait(1);` |

### moe_grouped_fp8_fp4_gemm_kernel grid (9, 30, 1) block (128, 1, 1)

147 launches, 1080 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 118.50 us, warp 23.98 us, nsys graphs 112.16 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 9.33 | 38.9% | 1.84 | 3.9% |
| wait | 6.51 | 27.1% | 22.00 | 46.2% |
| compute | 5.31 | 22.2% | 13.70 | 28.8% |
| load | 2.71 | 11.3% | 9.69 | 20.4% |
| store | 0.12 | 0.5% | 0.36 | 0.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*40 (est.: sampled iterations x40) | 5.31 | 22.2% | 0.4 | 14.08 | `{` |
| load#2*40 (est.: sampled iterations x40) | 2.30 | 9.6% | 0.4 | 8.96 | `if (ko < 38)` |
| wait.sync#5*40 (est.: sampled iterations x40) | 1.98 | 8.3% | 0.4 | 6.40 | `__syncthreads();` |
| wait.mbarrier*40 (est.: sampled iterations x40) | 1.73 | 7.2% | 0.4 | 6.40 | `loaded[(ko % 3)].wait(((ko % 6) / 3));` |
| wait.sync#6*40 (est.: sampled iterations x40) | 0.96 | 4.0% | 0.4 | 2.56 | `__syncthreads();` |
| wait.sync#4*40 (est.: sampled iterations x40) | 0.90 | 3.7% | 0.4 | 2.56 | `__syncthreads();` |
| wait.cp_async*40 (est.: sampled iterations x40) | 0.83 | 3.5% | 0.4 | 2.56 | `tl::cp_async_wait<2>();` |
| load.global#3 | 0.20 | 0.8% | 0.4 | 0.58 | `if (((expert < 128) && (0 <= expert))) \| if (((expert < 384) && (0 <= expert)))` |
| store.global | 0.12 | 0.5% | 0.4 | 0.38 | `for (int i_2 = 0; i_2 < 16; ++i_2) \| for (int i_3 = 0; i_3 < 16; ++i_3) \| for (int i_4 = 0` |
| load.global | 0.10 | 0.4% | 1.0 | 0.29 | `int expert = TileExpert[((int64_t)((int)blockIdx.y))];` |
| load | 0.07 | 0.3% | 0.4 | 0.26 | `for (int s = 0; s < 2; ++s)` |
| wait.sync | 0.05 | 0.2% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#2 | 0.05 | 0.2% | 0.4 | 0.16 | `__syncthreads();` |
| load.global#2 | 0.04 | 0.2% | 0.4 | 0.19 | `if (((int)threadIdx.x) < 32) \| if (((int)threadIdx.x) < 64)` |
| wait.sync#3 | 0.01 | 0.1% | 0.4 | 0.06 | `__syncthreads();` |

### moe_grouped_fp8_fp4_gemm_kernel grid (40, 30, 1) block (128, 1, 1)

147 launches, 4800 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 65.86 us, warp 4.57 us, nsys graphs 59.65 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.75 | 38.4% | 0.20 | 65.9% |
| load | 0.93 | 20.4% | 0.06 | 20.2% |
| compute | 0.88 | 19.3% | 0.00 | 0.0% |
| wait | 0.86 | 18.7% | 0.04 | 13.9% |
| store | 0.14 | 3.2% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*5 (est.: sampled iterations x5) | 0.88 | 19.3% | 0.5 | 1.76 | `if (ko < 4)` |
| load#2*5 (est.: sampled iterations x5) | 0.44 | 9.6% | 0.5 | 0.96 | `if (ko < 3)` |
| load.global#3 | 0.27 | 6.0% | 0.5 | 0.61 | `if (((expert < 128) && (0 <= expert))) \| if (((expert < 384) && (0 <= expert)))` |
| wait.mbarrier*5 (est.: sampled iterations x5) | 0.17 | 3.7% | 0.5 | 0.48 | `loaded[(ko % 3)].wait((ko / 3));` |
| wait.sync#5*5 (est.: sampled iterations x5) | 0.17 | 3.7% | 0.5 | 0.48 | `__syncthreads();` |
| store.global | 0.14 | 3.2% | 0.5 | 0.38 | `for (int i_2 = 0; i_2 < 16; ++i_2) \| for (int i_3 = 0; i_3 < 16; ++i_3) \| for (int i_4 = 0` |
| wait.sync#4*5 (est.: sampled iterations x5) | 0.14 | 3.1% | 0.5 | 0.32 | `__syncthreads();` |
| wait.sync#6*5 (est.: sampled iterations x5) | 0.13 | 2.8% | 0.5 | 0.32 | `__syncthreads();` |
| wait.cp_async*5 (est.: sampled iterations x5) | 0.12 | 2.6% | 0.5 | 0.32 | `tl::cp_async_wait<2>();` |
| load.global | 0.10 | 2.1% | 1.0 | 0.22 | `int expert = TileExpert[((int64_t)((int)blockIdx.y))];` |
| load | 0.08 | 1.8% | 0.5 | 0.26 | `for (int s = 0; s < 2; ++s)` |
| wait.sync#2 | 0.06 | 1.4% | 0.5 | 0.19 | `__syncthreads();` |
| wait.sync | 0.05 | 1.0% | 1.0 | 0.06 | `__syncthreads();` |
| load.global#2 | 0.04 | 0.9% | 0.5 | 0.22 | `if (((int)threadIdx.x) < 32) \| if (((int)threadIdx.x) < 64)` |
| wait.sync#3 | 0.02 | 0.4% | 0.5 | 0.06 | `__syncthreads();` |

### gemm_mxfp8_kernel grid (80, 1, 1) block (128, 1, 1)

17 launches, 320 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 409.86 us, warp 405.89 us, nsys graphs 64.19 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 337.37 | 83.1% | 306.51 | 75.2% |
| unattributed | 37.00 | 9.1% | 66.36 | 16.3% |
| compute | 18.19 | 4.5% | 18.24 | 4.5% |
| load | 13.28 | 3.3% | 16.47 | 4.0% |
| store | 0.06 | 0.0% | 0.05 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.cp_async*118 (est.: sampled iterations x118) | 247.33 | 60.9% | 1.0 | 604.16 | `tl::cp_async_wait<2>();` |
| wait.sync#2*118 (est.: sampled iterations x118) | 80.95 | 19.9% | 1.0 | 328.51 | `__syncthreads();` |
| compute.mma*118 (est.: sampled iterations x118) | 17.94 | 4.4% | 1.0 | 18.88 | `{` |
| load.cp_async#2*118 (est.: sampled iterations x118) | 12.98 | 3.2% | 1.0 | 15.10 | `for (int i_5 = 0; i_5 < 2; ++i_5) \| for (int i_5 = 0; i_5 < 4; ++i_5) \| tl::cp_async_gs_co` |
| wait.sync*118 (est.: sampled iterations x118) | 7.08 | 1.7% | 1.0 | 7.55 | `__syncthreads();` |
| wait.cp_async#2 | 0.73 | 0.2% | 1.0 | 2.75 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| wait.cp_async#3 | 0.53 | 0.1% | 1.0 | 1.89 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>();` |
| wait.sync#3 | 0.45 | 0.1% | 1.0 | 2.08 | `__syncthreads();` |
| load.cp_async | 0.30 | 0.1% | 1.0 | 0.51 | `for (int i = 0; i < 2; ++i) \| for (int i_1 = 0; i_1 < 2; ++i_1) \| for (int i_1 = 0; i_1 < ` |
| wait.sync#4 | 0.30 | 0.1% | 1.0 | 0.93 | `__syncthreads();` |
| compute.mma#3 | 0.13 | 0.0% | 1.0 | 0.13 | `{` |
| compute.mma#2 | 0.12 | 0.0% | 1.0 | 0.13 | `{` |
| store.global | 0.06 | 0.0% | 1.0 | 0.06 | `for (int i_10 = 0; i_10 < 16; ++i_10) \| for (int i_10 = 0; i_10 < 8; ++i_10) \| for (int i_` |

### gemm_mxfp8_kernel grid (64, 1, 1) block (128, 1, 1)

187 launches, 256 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 25.92 us, warp 14.91 us, nsys graphs 24.80 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 6.55 | 43.9% | 5.46 | 45.2% |
| wait | 5.23 | 35.1% | 3.48 | 28.8% |
| compute | 2.03 | 13.6% | 1.82 | 15.1% |
| load | 1.04 | 7.0% | 1.25 | 10.3% |
| store | 0.06 | 0.4% | 0.06 | 0.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*8 (est.: sampled iterations x8) | 1.66 | 11.2% | 1.0 | 1.79 | `{` |
| wait.sync#2*8 (est.: sampled iterations x8) | 1.55 | 10.4% | 1.0 | 5.12 | `__syncthreads();` |
| wait.cp_async*8 (est.: sampled iterations x8) | 1.36 | 9.1% | 1.0 | 6.66 | `tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| load.cp_async#2*8 (est.: sampled iterations x8) | 0.82 | 5.5% | 1.0 | 1.02 | `for (int i_3 = 0; i_3 < 2; ++i_3) \| for (int i_5 = 0; i_5 < 2; ++i_5) \| tl::cp_async_gs_co` |
| wait.cp_async#3 | 0.62 | 4.2% | 1.0 | 1.86 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>();` |
| wait.cp_async#2 | 0.53 | 3.5% | 1.0 | 1.86 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| wait.sync#4 | 0.42 | 2.8% | 1.0 | 1.44 | `__syncthreads();` |
| wait.sync#3 | 0.40 | 2.7% | 1.0 | 1.25 | `__syncthreads();` |
| wait.sync*8 (est.: sampled iterations x8) | 0.35 | 2.4% | 1.0 | 0.51 | `__syncthreads();` |
| load.cp_async | 0.23 | 1.5% | 1.0 | 0.26 | `for (int i = 0; i < 2; ++i) \| for (int i_1 = 0; i_1 < 2; ++i_1) \| for (int i_1 = 0; i_1 < ` |
| compute.mma#3 | 0.19 | 1.2% | 1.0 | 0.19 | `{` |
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

153 launches, 112 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 35.10 us, warp 33.08 us, nsys graphs 41.31 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 18.56 | 56.1% | 13.16 | 38.8% |
| wait | 7.54 | 22.8% | 13.04 | 38.5% |
| compute | 3.94 | 11.9% | 3.92 | 11.6% |
| load | 2.99 | 9.0% | 3.71 | 10.9% |
| store | 0.05 | 0.2% | 0.05 | 0.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*19 (est.: sampled iterations x19) | 3.76 | 11.4% | 1.0 | 4.26 | `{` |
| wait.sync#2*19 (est.: sampled iterations x19) | 3.19 | 9.6% | 1.0 | 9.73 | `__syncthreads();` |
| wait.cp_async*19 (est.: sampled iterations x19) | 2.89 | 8.7% | 1.0 | 13.98 | `tl::cp_async_wait<1>();` |
| load.cp_async#2*19 (est.: sampled iterations x19) | 2.70 | 8.2% | 1.0 | 3.04 | `for (int i_3 = 0; i_3 < 2; ++i_3) \| for (int i_3 = 0; i_3 < 8; ++i_3)` |
| wait.sync*19 (est.: sampled iterations x19) | 1.22 | 3.7% | 1.0 | 1.22 | `__syncthreads();` |
| load.cp_async | 0.29 | 0.9% | 1.0 | 0.32 | `for (int i = 0; i < 2; ++i) \| for (int i_1 = 0; i_1 < 2; ++i_1) \| for (int i_1 = 0; i_1 < ` |
| compute.mma#2 | 0.18 | 0.5% | 1.0 | 0.19 | `{` |
| wait.cp_async#2 | 0.14 | 0.4% | 1.0 | 0.35 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| wait.sync#3 | 0.11 | 0.3% | 1.0 | 0.29 | `__syncthreads();` |
| store.global | 0.05 | 0.2% | 1.0 | 0.06 | `for (int i_10 = 0; i_10 < 16; ++i_10) \| for (int i_10 = 0; i_10 < 8; ++i_10) \| for (int i_` |

### mhc_project_streams_kernel grid (40, 1, 1) block (128, 1, 1)

306 launches, 160 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 15.52 us, warp 13.36 us, nsys graphs 15.62 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 6.50 | 48.6% | 4.53 | 30.0% |
| compute | 2.78 | 20.8% | 2.56 | 17.0% |
| load | 2.47 | 18.5% | 4.73 | 31.3% |
| wait | 1.15 | 8.6% | 2.78 | 18.4% |
| store | 0.46 | 3.5% | 0.50 | 3.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*4 (est.: sampled iterations x4) | 2.65 | 19.8% | 1.0 | 2.69 | `{` |
| load.global#2*4 (est.: sampled iterations x4) | 2.23 | 16.7% | 1.0 | 4.86 | `for (int i_1 = 0; i_1 < 6; ++i_1)` |
| wait.sync*4 (est.: sampled iterations x4) | 0.96 | 7.2% | 1.0 | 2.94 | `__syncthreads();` |
| store.global*4 (est.: sampled iterations x4) | 0.25 | 1.9% | 1.0 | 0.38 | `if (((((int)blockIdx.y) * 16) + (((int)threadIdx.x) >> 3)) < M)` |
| load.global | 0.23 | 1.8% | 1.0 | 0.26 | `for (int k = 0; k < 4; ++k)` |
| wait.sync#2*4 (est.: sampled iterations x4) | 0.18 | 1.4% | 1.0 | 0.26 | `__syncthreads();` |
| store.global#4 | 0.09 | 0.7% | 1.0 | 0.13 | `if (((((int)threadIdx.x) % 8) == 0) && (((((int)blockIdx.y) * 16) + (((int)threadIdx.x) >>` |
| compute.reduce | 0.07 | 0.5% | 1.0 | 0.10 | `red[0] = (red[0] + __shfl_xor_sync((uint)4294967295, red[0], 4, 32));` |
| compute.reduce#2 | 0.07 | 0.5% | 1.0 | 0.10 | `red[0] = (red[0] + __shfl_xor_sync((uint)4294967295, red[0], 4, 32));` |
| store.global#2 | 0.06 | 0.5% | 1.0 | 0.10 | `if (((((int)blockIdx.y) * 16) + (((int)threadIdx.x) >> 3)) < M) \| if (((((int)threadIdx.x)` |
| store.global#3 | 0.06 | 0.4% | 1.0 | 0.06 | `if (((((int)threadIdx.x) % 8) == 0) && (((((int)blockIdx.y) * 16) + (((int)threadIdx.x) >>` |
| compute.mma | 0.00 | 0.0% | 0.0 | 0.00 | `{` |
| load.global#2 | 0.00 | 0.0% | 0.0 | 0.00 | `for (int i_1 = 0; i_1 < 6; ++i_1)` |
| store.global | 0.00 | 0.0% | 0.0 | 0.00 | `for (int n = 0; n < 4; ++n) \| if (((((int)blockIdx.y) * 16) + (((int)threadIdx.x) >> 3)) <` |
| wait.sync | 0.00 | 0.0% | 0.0 | 0.00 | `__syncthreads();` |

### gemm_fp8w_bf16_partials_kernel grid (32, 4, 1) block (128, 1, 1)

153 launches, 512 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 19.26 us, warp 16.24 us, nsys graphs 31.60 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 7.15 | 44.0% | 10.20 | 45.2% |
| load | 3.43 | 21.1% | 4.13 | 18.3% |
| unattributed | 3.12 | 19.2% | 5.67 | 25.1% |
| compute | 2.48 | 15.3% | 2.51 | 11.1% |
| store | 0.06 | 0.4% | 0.06 | 0.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*14 (est.: sampled iterations x14) | 2.24 | 13.8% | 1.0 | 2.69 | `{` |
| wait.sync#2*14 (est.: sampled iterations x14) | 1.23 | 7.6% | 1.0 | 1.79 | `__syncthreads();` |
| load.global*14 (est.: sampled iterations x14) | 1.09 | 6.7% | 1.0 | 1.79 | `if (((int)threadIdx.x) < 4)` |
| load.cp_async#2*14 (est.: sampled iterations x14) | 1.01 | 6.2% | 1.0 | 1.34 | `for (int i_4 = 0; i_4 < 2; ++i_4) \| for (int i_6 = 0; i_6 < 2; ++i_6)` |
| load.cp_async#3*14 (est.: sampled iterations x14) | 0.95 | 5.9% | 1.0 | 1.34 | `for (int i_10 = 0; i_10 < 4; ++i_10) \| for (int i_10 = 0; i_10 < 6; ++i_10) \| for (int i_1` |
| wait.sync#4*14 (est.: sampled iterations x14) | 0.95 | 5.9% | 1.0 | 1.34 | `__syncthreads();` |
| wait.sync#3*14 (est.: sampled iterations x14) | 0.92 | 5.7% | 1.0 | 1.34 | `__syncthreads();` |
| wait.sync#5*14 (est.: sampled iterations x14) | 0.92 | 5.7% | 1.0 | 1.34 | `__syncthreads();` |
| wait.sync*14 (est.: sampled iterations x14) | 0.92 | 5.7% | 1.0 | 1.34 | `__syncthreads();` |
| wait.cp_async*14 (est.: sampled iterations x14) | 0.87 | 5.3% | 1.0 | 0.90 | `tl::cp_async_wait<3>();` |
| wait.cp_async#2*14 (est.: sampled iterations x14) | 0.81 | 5.0% | 1.0 | 0.90 | `tl::cp_async_wait<3>();` |
| load.cp_async | 0.26 | 1.6% | 1.0 | 0.48 | `for (int i_1 = 0; i_1 < 2; ++i_1)` |
| compute.mma#2 | 0.12 | 0.8% | 1.0 | 0.16 | `{` |
| compute.mma#3 | 0.12 | 0.7% | 1.0 | 0.13 | `{` |
| wait.sync#10 | 0.06 | 0.4% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#7 | 0.06 | 0.4% | 1.0 | 0.10 | `__syncthreads();` |
| load.global#2 | 0.06 | 0.4% | 1.0 | 0.10 | `if (((int)threadIdx.x) < 4)` |
| store.global | 0.06 | 0.4% | 1.0 | 0.06 | `for (int i_13 = 0; i_13 < 8; ++i_13) \| for (int i_19 = 0; i_19 < 16; ++i_19) \| for (int i_` |
| load.global#3 | 0.06 | 0.4% | 1.0 | 0.10 | `if (((int)threadIdx.x) < 4)` |
| wait.cp_async#6 | 0.06 | 0.3% | 1.0 | 0.06 | `tl::cp_async_wait<0>();` |
| wait.sync#6 | 0.05 | 0.3% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#8 | 0.05 | 0.3% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#11 | 0.05 | 0.3% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#9 | 0.05 | 0.3% | 1.0 | 0.06 | `__syncthreads();` |
| wait.cp_async#3 | 0.05 | 0.3% | 1.0 | 0.06 | `tl::cp_async_wait<3>();` |
| wait.cp_async#5 | 0.05 | 0.3% | 1.0 | 0.06 | `tl::cp_async_wait<1>();` |
| wait.cp_async#4 | 0.04 | 0.3% | 1.0 | 0.06 | `tl::cp_async_wait<2>();` |

### gemm_mxfp8_kernel grid (36, 1, 1) block (128, 1, 1)

153 launches, 144 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 23.14 us, warp 21.84 us, nsys graphs 25.89 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 11.87 | 54.3% | 5.26 | 23.2% |
| wait | 4.03 | 18.5% | 10.14 | 44.8% |
| compute | 3.53 | 16.2% | 3.60 | 15.9% |
| load | 2.36 | 10.8% | 3.58 | 15.8% |
| store | 0.05 | 0.2% | 0.05 | 0.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*18 (est.: sampled iterations x18) | 3.24 | 14.8% | 1.0 | 3.46 | `{` |
| load.cp_async#2*18 (est.: sampled iterations x18) | 2.16 | 9.9% | 1.0 | 2.30 | `for (int i_4 = 0; i_4 < 2; ++i_4) \| for (int i_5 = 0; i_5 < 4; ++i_5)` |
| wait.sync#2*18 (est.: sampled iterations x18) | 1.55 | 7.1% | 1.0 | 2.88 | `__syncthreads();` |
| wait.cp_async*18 (est.: sampled iterations x18) | 1.15 | 5.3% | 1.0 | 2.88 | `tl::cp_async_wait<2>();` |
| wait.sync*18 (est.: sampled iterations x18) | 1.12 | 5.1% | 1.0 | 1.15 | `__syncthreads();` |
| load.cp_async | 0.20 | 0.9% | 1.0 | 0.26 | `for (int i = 0; i < 2; ++i) \| for (int i_1 = 0; i_1 < 2; ++i_1) \| for (int i_1 = 0; i_1 < ` |
| compute.mma#2 | 0.15 | 0.7% | 1.0 | 0.16 | `{` |
| compute.mma#3 | 0.15 | 0.7% | 1.0 | 0.16 | `{` |
| wait.cp_async#3 | 0.06 | 0.3% | 1.0 | 0.06 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>();` |
| wait.sync#3 | 0.06 | 0.3% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#4 | 0.05 | 0.2% | 1.0 | 0.06 | `__syncthreads();` |
| store.global | 0.05 | 0.2% | 1.0 | 0.06 | `for (int i_10 = 0; i_10 < 16; ++i_10) \| for (int i_10 = 0; i_10 < 8; ++i_10) \| for (int i_` |
| wait.cp_async#2 | 0.05 | 0.2% | 1.0 | 0.06 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |

### gemm_mxfp8_kernel grid (40, 1, 1) block (128, 1, 1)

153 launches, 160 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 18.53 us, warp 16.74 us, nsys graphs 18.53 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 6.76 | 40.4% | 12.12 | 58.9% |
| unattributed | 4.53 | 27.1% | 2.12 | 10.3% |
| compute | 3.23 | 19.3% | 3.21 | 15.6% |
| load | 2.15 | 12.9% | 3.06 | 14.9% |
| store | 0.06 | 0.4% | 0.06 | 0.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*14 (est.: sampled iterations x14) | 2.88 | 17.2% | 1.0 | 3.14 | `{` |
| wait.cp_async*14 (est.: sampled iterations x14) | 2.58 | 15.4% | 1.0 | 10.30 | `tl::cp_async_wait<2>();` |
| wait.sync#2*14 (est.: sampled iterations x14) | 2.35 | 14.0% | 1.0 | 8.51 | `__syncthreads();` |
| load.cp_async#2*14 (est.: sampled iterations x14) | 1.90 | 11.4% | 1.0 | 2.24 | `for (int i_5 = 0; i_5 < 2; ++i_5) \| tl::cp_async_gs_conditional<16>((&(((fp8_e4_t*)A_share` |
| wait.sync*14 (est.: sampled iterations x14) | 0.90 | 5.4% | 1.0 | 0.90 | `__syncthreads();` |
| wait.cp_async#2 | 0.33 | 2.0% | 1.0 | 1.18 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| wait.sync#3 | 0.26 | 1.6% | 1.0 | 0.96 | `__syncthreads();` |
| load.cp_async | 0.25 | 1.5% | 1.0 | 0.29 | `for (int i = 0; i < 2; ++i) \| for (int i_1 = 0; i_1 < 2; ++i_1) \| for (int i_1 = 0; i_1 < ` |
| wait.cp_async#3 | 0.19 | 1.2% | 1.0 | 0.67 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>();` |
| compute.mma#3 | 0.18 | 1.1% | 1.0 | 0.19 | `{` |
| compute.mma#2 | 0.17 | 1.0% | 1.0 | 0.19 | `{` |
| wait.sync#4 | 0.16 | 0.9% | 1.0 | 0.54 | `__syncthreads();` |
| store.global | 0.06 | 0.4% | 1.0 | 0.06 | `for (int i_10 = 0; i_10 < 16; ++i_10) \| for (int i_10 = 0; i_10 < 8; ++i_10) \| for (int i_` |

### gemm_bf16_partials_kernel grid (6, 8, 1) block (256, 1, 1)

102 launches, 384 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 24.24 us, warp 5.45 us, nsys graphs 22.38 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 3.18 | 58.4% | 1.21 | 27.8% |
| wait | 1.16 | 21.3% | 1.72 | 39.5% |
| compute | 0.68 | 12.5% | 1.35 | 31.0% |
| load | 0.39 | 7.2% | 0.00 | 0.0% |
| store | 0.04 | 0.7% | 0.08 | 1.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*10 (est.: sampled iterations x10) | 0.68 | 12.5% | 0.5 | 1.60 | `{` |
| wait.mbarrier*10 (est.: sampled iterations x10) | 0.64 | 11.7% | 0.5 | 1.60 | `mbarrier[((kb % 3) + 3)].wait((((kb % 6) / 3) ^ 1));` |
| wait.mbarrier#2*10 (est.: sampled iterations x10) | 0.42 | 7.7% | 0.5 | 1.60 | `mbarrier[(kb_1 % 3)].wait(((kb_1 % 6) / 3));` |
| load.tma*10 (est.: sampled iterations x10) | 0.39 | 7.2% | 0.5 | 1.28 | `if (tl::tl_shuffle_elect<128>())` |
| wait.sync | 0.10 | 1.8% | 1.0 | 0.13 | `__syncthreads();` |
| store.global | 0.04 | 0.7% | 0.5 | 0.10 | `for (int i_1 = 0; i_1 < 4; ++i_1) \| for (int i_2 = 0; i_2 < 8; ++i_2) \| for (int i_4 = 0; ` |

### mhc_finalize_kernel grid (5, 1, 1) block (128, 1, 1)

192 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 12.24 us, warp 11.21 us, nsys graphs 11.68 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 6.90 | 61.5% | 6.03 | 49.8% |
| store | 1.67 | 14.9% | 2.38 | 19.7% |
| load | 1.06 | 9.5% | 1.91 | 15.8% |
| compute | 1.00 | 8.9% | 1.11 | 9.2% |
| wait | 0.58 | 5.2% | 0.67 | 5.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.77 | 6.9% | 0.2 | 3.09 | `if (((int)threadIdx.x) < 4)` |
| wait.sync | 0.58 | 5.2% | 1.0 | 0.80 | `__syncthreads();` |
| compute.reduce | 0.57 | 5.1% | 0.2 | 2.30 | `row_max[0] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<32>>::run(row_max[0]);` |
| store.global#4*7 (est.: sampled iterations x7) | 0.56 | 5.0% | 0.8 | 0.90 | `if ((((int)threadIdx.x) % 4) == 0)` |
| compute.reduce#2*7 (est.: sampled iterations x7) | 0.43 | 3.8% | 0.8 | 0.67 | `amax[0] = max(amax[0], __shfl_xor_sync((uint)4294967295, amax[0], 1, 32));` |
| load.global#4*7 (est.: sampled iterations x7) | 0.37 | 3.3% | 0.8 | 0.67 | `*(uint4*)(yv + 0) = *(uint4*)(C + ((((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((i` |
| load.global | 0.35 | 3.1% | 1.0 | 0.99 | `if (((int)threadIdx.x) < 32)` |
| store.global#3*7 (est.: sampled iterations x7) | 0.32 | 2.9% | 0.8 | 0.45 | `*(uint4*)(y + ((((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((int64_t)step) * (int6` |
| load.global#3 | 0.30 | 2.7% | 0.8 | 0.42 | `for (int i_split_2 = 0; i_split_2 < 40; ++i_split_2)` |
| load.global#2 | 0.04 | 0.4% | 0.2 | 0.19 | `cm[0] = ((mixes_shared[((((int)threadIdx.x) & 15) + 8)] * scale[2]) + base[((((int)threadI` |
| store.global#2 | 0.01 | 0.1% | 0.2 | 0.06 | `if ((((int)threadIdx.x) >> 4) == 0)` |

### gemm_bf16_splitk_kernel grid (16, 5, 1) block (128, 1, 1)

34 launches, 320 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 57.60 us, warp 54.10 us, nsys graphs 51.92 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 33.69 | 62.3% | 65.42 | 115.8% |
| unattributed | 16.45 | 30.4% | 0.00 | 0.0% |
| compute | 1.97 | 3.6% | 2.07 | 3.7% |
| load | 1.87 | 3.5% | 4.29 | 7.6% |
| store | 0.13 | 0.2% | 0.16 | 0.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.cp_async*13 (est.: sampled iterations x13) | 17.61 | 32.6% | 0.5 | 38.69 | `tl::cp_async_wait<2>();` |
| wait.cp_async*12 (est.: sampled iterations x12) | 3.58 | 6.6% | 0.5 | 11.33 | `tl::cp_async_wait<3>();` |
| wait.sync*13 (est.: sampled iterations x13) | 3.00 | 5.6% | 0.5 | 9.36 | `__syncthreads();` |
| wait.sync*12 (est.: sampled iterations x12) | 2.58 | 4.8% | 0.5 | 7.10 | `__syncthreads();` |
| wait.cp_async#2 | 1.58 | 2.9% | 1.0 | 4.93 | `tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>(); \| tl::cp_async_wait<3>();` |
| wait.cp_async#4 | 1.42 | 2.6% | 1.0 | 3.73 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>();` |
| compute.mma*13 (est.: sampled iterations x13) | 0.81 | 1.5% | 0.5 | 0.83 | `{` |
| compute.mma*12 (est.: sampled iterations x12) | 0.80 | 1.5% | 0.5 | 0.96 | `{` |
| wait.sync#5 | 0.74 | 1.4% | 1.0 | 2.26 | `__syncthreads();` |
| wait.cp_async#3 | 0.74 | 1.4% | 1.0 | 2.59 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| load.cp_async | 0.67 | 1.2% | 1.0 | 1.50 | `for (int i_1 = 0; i_1 < 2; ++i_1) \| tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)A_sha` |
| wait.sync#3 | 0.64 | 1.2% | 1.0 | 2.35 | `__syncthreads();` |
| wait.sync#4 | 0.61 | 1.1% | 1.0 | 2.03 | `__syncthreads();` |
| load.cp_async#2*12 (est.: sampled iterations x12) | 0.50 | 0.9% | 0.5 | 0.58 | `tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)A_shared)[((((((kb & 3) * 1024) + ((((int` |
| load.cp_async#2*13 (est.: sampled iterations x13) | 0.43 | 0.8% | 0.5 | 0.62 | `for (int i_10 = 0; i_10 < 2; ++i_10) \| for (int i_8 = 0; i_8 < 2; ++i_8) \| tl::cp_async_gs` |
| wait.sync#2*13 (est.: sampled iterations x13) | 0.36 | 0.7% | 0.5 | 0.42 | `__syncthreads();` |
| wait.sync#2*12 (est.: sampled iterations x12) | 0.30 | 0.6% | 0.5 | 0.38 | `__syncthreads();` |
| load.global | 0.27 | 0.5% | 1.0 | 1.73 | `if (((int)threadIdx.x) == 0)` |
| wait.sync#7 | 0.19 | 0.4% | 1.0 | 0.27 | `__syncthreads();` |
| wait.sync#8 | 0.18 | 0.3% | 0.5 | 0.32 | `__syncthreads();` |
| compute.mma#2 | 0.11 | 0.2% | 1.0 | 0.13 | `{` |
| compute.mma#4 | 0.10 | 0.2% | 1.0 | 0.13 | `{` |
| compute.mma#3 | 0.10 | 0.2% | 1.0 | 0.13 | `{` |
| wait.sync#6 | 0.09 | 0.2% | 1.0 | 0.24 | `__syncthreads();` |
| store.global#3 | 0.07 | 0.1% | 1.0 | 0.10 | `if (((int*)last)[0] == 1)` |
| store.global | 0.06 | 0.1% | 1.0 | 0.06 | `for (int i_10 = 0; i_10 < 4; ++i_10) \| for (int i_16 = 0; i_16 < 4; ++i_16) \| for (int i_1` |
| compute.mma#5 | 0.05 | 0.1% | 0.5 | 0.06 | `{` |
| wait.cp_async#5 | 0.05 | 0.1% | 0.5 | 0.16 | `tl::cp_async_wait<0>();` |

### gemm_mxfp8_kernel grid (40, 1, 1) block (64, 1, 1)

153 launches, 80 warps; ranges, warp and last warp from the 4 recording CTAs' 8 warps; span 7.52 us, warp 6.75 us, nsys graphs 12.51 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 2.40 | 35.6% | 8.36 | 76.8% |
| unattributed | 1.92 | 28.5% | 0.00 | 0.0% |
| compute | 1.50 | 22.3% | 1.54 | 14.2% |
| load | 0.84 | 12.4% | 1.32 | 12.2% |
| store | 0.08 | 1.2% | 0.10 | 0.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*8 (est.: sampled iterations x8) | 1.34 | 19.9% | 1.0 | 1.54 | `{` |
| wait.cp_async*8 (est.: sampled iterations x8) | 1.02 | 15.2% | 1.0 | 3.33 | `tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| load.cp_async#2*8 (est.: sampled iterations x8) | 0.70 | 10.4% | 1.0 | 0.77 | `for (int i_3 = 0; i_3 < 2; ++i_3) \| for (int i_5 = 0; i_5 < 2; ++i_5) \| tl::cp_async_gs_co` |
| wait.sync#2*8 (est.: sampled iterations x8) | 0.64 | 9.5% | 1.0 | 1.79 | `__syncthreads();` |
| wait.sync*8 (est.: sampled iterations x8) | 0.38 | 5.7% | 1.0 | 0.51 | `__syncthreads();` |
| wait.cp_async#2 | 0.26 | 3.9% | 1.0 | 0.54 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| compute.mma#2 | 0.16 | 2.4% | 1.0 | 0.16 | `{` |
| load.cp_async | 0.13 | 2.0% | 1.0 | 0.16 | `for (int i = 0; i < 2; ++i) \| for (int i_1 = 0; i_1 < 2; ++i_1) \| for (int i_1 = 0; i_1 < ` |
| wait.sync#3 | 0.10 | 1.4% | 1.0 | 0.29 | `__syncthreads();` |
| store.global | 0.08 | 1.2% | 1.0 | 0.10 | `for (int i_10 = 0; i_10 < 16; ++i_10) \| for (int i_10 = 0; i_10 < 8; ++i_10) \| for (int i_` |

### sparse_mla_kernel grid (5, 1, 5) block (128, 1, 1)

64 launches, 100 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 16.59 us, warp 15.87 us, nsys graphs 16.06 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed, with loops' unsampled iterations | 10.07 | 63.4% | 11.64 | 48.8% |
| wait | 2.63 | 16.6% | 0.64 | 2.7% |
| load | 1.77 | 11.2% | 10.10 | 42.3% |
| compute | 0.96 | 6.1% | 0.96 | 4.0% |
| store | 0.44 | 2.8% | 0.53 | 2.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync (sampled iterations) | 2.56 | 16.1% | 11.0 | 4.21 | `__syncthreads();` |
| load.global (sampled iterations) | 1.44 | 9.1% | 2.0 | 4.11 | `for (int c = 0; c < 4; ++c) \| for (int i = 0; i < 8; ++i) \| for (int i_1 = 0; i_1 < 8; ++i` |
| compute.mma (sampled iterations) | 0.74 | 4.6% | 2.0 | 0.77 | `for (int ki = 0; ki < 32; ++ki) \| for (int ki_1 = 0; ki_1 < 2; ++ki_1)` |
| store.global | 0.44 | 2.8% | 1.0 | 0.54 | `for (int i_10 = 0; i_10 < 32; ++i_10) \| for (int i_11 = 0; i_11 < 32; ++i_11) \| for (int i` |
| load.cp_async (sampled iterations) | 0.33 | 2.1% | 2.0 | 0.42 | `for (int k_1 = 0; k_1 < 5; ++k_1) \| for (int k_1 = 0; k_1 < 9; ++k_1) \| for (int k_3 = 0; ` |
| compute.reduce (sampled iterations) | 0.22 | 1.4% | 2.0 | 0.26 | `m_g[i_5] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(m_g[i_5], (&((` |
| wait.cp_async (sampled iterations) | 0.07 | 0.4% | 1.0 | 0.10 | `tl::cp_async_wait<0>();` |

### mhc_finalize_kernel grid (10, 1, 1) block (128, 1, 1)

102 launches, 40 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 11.33 us, warp 10.40 us, nsys graphs 9.95 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 6.48 | 62.3% | 5.63 | 52.1% |
| store | 1.38 | 13.3% | 1.65 | 15.2% |
| compute | 1.00 | 9.6% | 1.12 | 10.4% |
| load | 0.98 | 9.4% | 1.62 | 15.0% |
| wait | 0.56 | 5.4% | 0.78 | 7.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.reduce | 0.58 | 5.5% | 0.2 | 2.30 | `row_max[0] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<32>>::run(row_max[0]);` |
| store.global#4*7 (est.: sampled iterations x7) | 0.56 | 5.4% | 0.8 | 0.90 | `if ((((int)threadIdx.x) % 4) == 0)` |
| wait.sync | 0.56 | 5.4% | 1.0 | 0.74 | `__syncthreads();` |
| store.global | 0.48 | 4.7% | 0.2 | 1.95 | `if (((int)threadIdx.x) < 4)` |
| compute.reduce#2*7 (est.: sampled iterations x7) | 0.43 | 4.1% | 0.8 | 0.67 | `amax[0] = max(amax[0], __shfl_xor_sync((uint)4294967295, amax[0], 1, 32));` |
| load.global#4*7 (est.: sampled iterations x7) | 0.37 | 3.6% | 0.8 | 0.67 | `*(uint4*)(yv + 0) = *(uint4*)(C + ((((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((i` |
| store.global#3*7 (est.: sampled iterations x7) | 0.32 | 3.1% | 0.8 | 0.45 | `*(uint4*)(y + ((((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((int64_t)step) * (int6` |
| load.global | 0.31 | 3.0% | 1.0 | 0.90 | `if (((int)threadIdx.x) < 32)` |
| load.global#3 | 0.25 | 2.4% | 0.8 | 0.35 | `for (int i_split_2 = 0; i_split_2 < 40; ++i_split_2)` |
| load.global#2 | 0.04 | 0.4% | 0.2 | 0.19 | `cm[0] = ((mixes_shared[((((int)threadIdx.x) & 15) + 8)] * scale[2]) + base[((((int)threadI` |
| store.global#2 | 0.01 | 0.1% | 0.2 | 0.06 | `if ((((int)threadIdx.x) >> 4) == 0)` |

### sparse_mla_kernel grid (2, 1, 10) block (128, 1, 1)

51 launches, 80 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 21.92 us, warp 15.65 us, nsys graphs 21.09 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed, with loops' unsampled iterations | 8.57 | 54.7% | 7.80 | 37.6% |
| compute | 2.50 | 15.9% | 4.73 | 22.8% |
| load | 2.17 | 13.8% | 5.13 | 24.7% |
| wait | 1.98 | 12.7% | 2.62 | 12.6% |
| store | 0.44 | 2.8% | 0.48 | 2.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 | 1.22 | 7.8% | 1.0 | 3.84 | `for (int i = 0; i < 8; ++i)` |
| compute.mma*4 (est.: sampled iterations x4) | 0.86 | 5.5% | 0.5 | 1.79 | `for (int ki = 0; ki < 32; ++ki) \| {` |
| compute.mma#2*4 (est.: sampled iterations x4) | 0.74 | 4.7% | 1.0 | 1.28 | `for (int ki_1 = 0; ki_1 < 2; ++ki_1) \| if (((((int)blockIdx.x) * 2) + (c_1 >> 1)) < 3)` |
| load.cp_async#2*4 (est.: sampled iterations x4) | 0.50 | 3.2% | 1.0 | 0.77 | `if (((((int)blockIdx.x) * 2) + (c_1 >> 1)) < 3) \| if ((c_1 < 3) & (c_1 < 3)) \| if (c < 3)` |
| compute.reduce#2*8 (est.: sampled iterations x8) | 0.45 | 2.9% | 0.5 | 1.02 | `row_sum[i_7] = tl::AllReduce<tl::SumOp, 128, 32, 0, tl::NamedBarrier<128>>::run(row_sum[i_` |
| compute.reduce*8 (est.: sampled iterations x8) | 0.45 | 2.9% | 0.5 | 1.02 | `m_g[i_4] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(m_g[i_4], (&((` |
| store.global | 0.44 | 2.8% | 1.0 | 0.48 | `for (int i_10 = 0; i_10 < 32; ++i_10) \| for (int i_11 = 0; i_11 < 32; ++i_11) \| for (int i` |
| load.cp_async (sampled iterations) | 0.33 | 2.1% | 1.0 | 0.35 | `for (int k_1 = 0; k_1 < 5; ++k_1) \| for (int k_1 = 0; k_1 < 9; ++k_1) \| for (int k_3 = 0; ` |
| wait.sync#11*4 (est.: sampled iterations x4) | 0.25 | 1.6% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#7*4 (est.: sampled iterations x4) | 0.25 | 1.6% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#6*4 (est.: sampled iterations x4) | 0.24 | 1.5% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#9*4 (est.: sampled iterations x4) | 0.23 | 1.5% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync (sampled iterations) | 0.18 | 1.1% | 1.0 | 0.22 | `__syncthreads();` |
| wait.sync#10*8 (est.: sampled iterations x8) | 0.18 | 1.1% | 0.5 | 0.51 | `__syncthreads();` |
| wait.sync#8*8 (est.: sampled iterations x8) | 0.18 | 1.1% | 0.5 | 0.51 | `__syncthreads();` |
| wait.cp_async*4 (est.: sampled iterations x4) | 0.13 | 0.8% | 0.5 | 0.26 | `tl::cp_async_wait<0>();` |
| wait.sync#4*4 (est.: sampled iterations x4) | 0.13 | 0.8% | 0.5 | 0.26 | `__syncthreads();` |
| wait.sync#5*4 (est.: sampled iterations x4) | 0.13 | 0.8% | 0.5 | 0.26 | `__syncthreads();` |
| load.global (sampled iterations) | 0.13 | 0.8% | 1.0 | 0.26 | `for (int c = 0; c < 4; ++c) \| for (int i = 0; i < 8; ++i) \| for (int i_1 = 0; i_1 < 8; ++i` |
| wait.sync#3 | 0.06 | 0.4% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#2 | 0.04 | 0.3% | 1.0 | 0.06 | `__syncthreads();` |

### cache_writer_kernel grid (5, 1, 1) block (128, 1, 1)

128 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 5.12 us, warp 5.00 us, nsys graphs 4.83 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 3.57 | 71.3% | 2.65 | 45.0% |
| store | 1.06 | 21.2% | 2.55 | 43.3% |
| load | 0.30 | 6.0% | 0.51 | 8.6% |
| compute | 0.07 | 1.5% | 0.18 | 3.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 1.06 | 21.2% | 1.0 | 1.12 | `if (((int64_t)0 <= slot) & ((slot >> (int64_t)7) < ((int64_t)pages))) \| if (((int64_t)0 <=` |
| load.global | 0.24 | 4.8% | 1.0 | 0.26 | `for (int i = 0; i < 4; ++i)` |
| compute.reduce | 0.07 | 1.5% | 1.0 | 0.10 | `amax = max(amax, __shfl_xor_sync((uint)4294967295, amax, 1, 32));` |
| load.global#2 | 0.06 | 1.2% | 1.0 | 0.06 | `int64_t slot = slots[((int64_t)((int)blockIdx.x))];` |

### gemm_bf16_partials_kernel grid (2, 20, 1) block (256, 1, 1)

51 launches, 320 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 13.98 us, warp 2.94 us, nsys graphs 12.51 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.98 | 67.5% | 0.84 | 46.0% |
| wait | 0.54 | 18.4% | 0.48 | 26.1% |
| compute | 0.22 | 7.6% | 0.43 | 23.8% |
| load | 0.15 | 5.0% | 0.00 | 0.0% |
| store | 0.04 | 1.4% | 0.07 | 4.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.mbarrier#2*4 (est.: sampled iterations x4) | 0.31 | 10.5% | 0.5 | 1.15 | `mbarrier[(kb_1 % 3)].wait((kb_1 / 3));` |
| compute.mma*4 (est.: sampled iterations x4) | 0.22 | 7.6% | 0.5 | 0.51 | `{` |
| load.tma*4 (est.: sampled iterations x4) | 0.15 | 5.0% | 0.5 | 0.38 | `if (tl::tl_shuffle_elect<128>())` |
| wait.mbarrier*4 (est.: sampled iterations x4) | 0.12 | 4.2% | 0.5 | 0.26 | `mbarrier[((kb % 3) + 3)].wait(((kb / 3) ^ 1));` |
| wait.sync | 0.11 | 3.7% | 1.0 | 0.19 | `__syncthreads();` |
| store.global | 0.04 | 1.4% | 0.5 | 0.10 | `for (int i_1 = 0; i_1 < 4; ++i_1) \| for (int i_2 = 0; i_2 < 8; ++i_2) \| for (int i_4 = 0; ` |

### moe_dispatch_kernel grid (1, 1, 1) block (512, 1, 1)

153 launches, 16 warps; ranges, warp and last warp from the 1 recording CTAs' 16 warps; span 4.70 us, warp 4.48 us, nsys graphs 2.88 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 2.57 | 57.4% | 1.70 | 37.0% |
| unattributed, with loops' unsampled iterations | 1.23 | 27.4% | 1.63 | 35.3% |
| store | 0.60 | 13.4% | 1.18 | 25.6% |
| load | 0.08 | 1.8% | 0.09 | 2.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync#7*9 (est.: sampled iterations x9) | 0.67 | 14.9% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#6*9 (est.: sampled iterations x9) | 0.63 | 14.1% | 1.0 | 0.86 | `__syncthreads();` |
| store.global (sampled iterations) | 0.52 | 11.7% | 1.0 | 0.96 | `ChunkCount[((((int64_t)c) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = count; \| Chu` |
| wait.sync#8 | 0.50 | 11.2% | 1.0 | 0.80 | `__syncthreads();` |
| wait.sync#2 | 0.39 | 8.7% | 1.0 | 0.42 | `__syncthreads();` |
| wait.sync#3 | 0.14 | 3.0% | 1.0 | 0.32 | `__syncthreads();` |
| load.global (sampled iterations) | 0.08 | 1.8% | 1.0 | 0.10 | `expert = ((int)TopkIdx[((((int64_t)((int)blockIdx.x)) * (int64_t)512) + ((int64_t)((int)th` |
| store.global#2 | 0.08 | 1.7% | 1.0 | 0.29 | `RoutePos[((((int64_t)((int)blockIdx.x)) * (int64_t)512) + ((int64_t)((int)threadIdx.x)))] ` |
| wait.sync | 0.07 | 1.5% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#4 | 0.06 | 1.3% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#5 | 0.06 | 1.3% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#9 | 0.06 | 1.3% | 1.0 | 0.06 | `__syncthreads();` |

### sparse_mla_kernel grid (1, 1, 5) block (128, 1, 1)

32 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 21.30 us, warp 21.24 us, nsys graphs 22.51 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed, with loops' unsampled iterations | 7.34 | 34.6% | 6.61 | 32.0% |
| compute | 4.73 | 22.3% | 4.74 | 23.0% |
| load | 4.64 | 21.9% | 6.05 | 29.3% |
| wait | 3.88 | 18.3% | 2.55 | 12.3% |
| store | 0.65 | 3.0% | 0.69 | 3.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 | 3.12 | 14.7% | 1.0 | 3.36 | `for (int i = 0; i < 8; ++i)` |
| compute.mma*4 (est.: sampled iterations x4) | 1.72 | 8.1% | 1.0 | 1.79 | `for (int ki = 0; ki < 32; ++ki) \| {` |
| wait.sync (sampled iterations) | 1.38 | 6.5% | 1.0 | 1.82 | `__syncthreads();` |
| compute.mma#2*4 (est.: sampled iterations x4) | 1.16 | 5.5% | 1.0 | 1.28 | `for (int ki_1 = 0; ki_1 < 2; ++ki_1) \| if (((((int)blockIdx.x) * 2) + (c_1 >> 1)) < 3)` |
| compute.reduce#2*8 (est.: sampled iterations x8) | 0.95 | 4.5% | 1.0 | 1.02 | `row_sum[i_7] = tl::AllReduce<tl::SumOp, 128, 32, 0, tl::NamedBarrier<128>>::run(row_sum[i_` |
| compute.reduce*8 (est.: sampled iterations x8) | 0.90 | 4.2% | 1.0 | 1.02 | `m_g[i_4] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(m_g[i_4], (&((` |
| load.cp_async#2*4 (est.: sampled iterations x4) | 0.65 | 3.1% | 1.0 | 0.77 | `if (((((int)blockIdx.x) * 2) + (c_1 >> 1)) < 3) \| if ((c_1 < 3) & (c_1 < 3)) \| if (c < 3)` |
| store.global | 0.65 | 3.0% | 1.0 | 0.70 | `for (int i_10 = 0; i_10 < 32; ++i_10) \| for (int i_11 = 0; i_11 < 32; ++i_11) \| for (int i` |
| load.global (sampled iterations) | 0.51 | 2.4% | 1.0 | 1.86 | `for (int c = 0; c < 4; ++c) \| for (int i = 0; i < 8; ++i) \| for (int i_1 = 0; i_1 < 8; ++i` |
| load.cp_async (sampled iterations) | 0.36 | 1.7% | 1.0 | 0.38 | `for (int k_1 = 0; k_1 < 5; ++k_1) \| for (int k_1 = 0; k_1 < 9; ++k_1) \| for (int k_3 = 0; ` |
| wait.sync#8*8 (est.: sampled iterations x8) | 0.32 | 1.5% | 1.0 | 0.51 | `__syncthreads();` |
| wait.sync#10*8 (est.: sampled iterations x8) | 0.31 | 1.4% | 1.0 | 0.51 | `__syncthreads();` |
| wait.cp_async*4 (est.: sampled iterations x4) | 0.27 | 1.3% | 1.0 | 0.38 | `tl::cp_async_wait<0>();` |
| wait.sync#4*4 (est.: sampled iterations x4) | 0.26 | 1.2% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#6*4 (est.: sampled iterations x4) | 0.26 | 1.2% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#11*4 (est.: sampled iterations x4) | 0.25 | 1.2% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#9*4 (est.: sampled iterations x4) | 0.25 | 1.2% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#5*4 (est.: sampled iterations x4) | 0.25 | 1.2% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#7*4 (est.: sampled iterations x4) | 0.24 | 1.1% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#3 | 0.06 | 0.3% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#2 | 0.05 | 0.2% | 1.0 | 0.06 | `__syncthreads();` |

### partial_argmax_kernel grid (2, 64, 1) block (128, 1, 1)

85 launches, 512 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 7.52 us, warp 6.39 us, nsys graphs 6.98 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 5.62 | 88.0% | 6.37 | 85.8% |
| wait | 0.30 | 4.6% | 0.30 | 4.0% |
| load | 0.27 | 4.2% | 0.51 | 6.9% |
| compute | 0.12 | 1.9% | 0.12 | 1.6% |
| store | 0.08 | 1.2% | 0.13 | 1.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync | 0.30 | 4.6% | 1.0 | 1.18 | `__syncthreads();` |
| load.global | 0.27 | 4.2% | 1.0 | 0.42 | `for (int j = 0; j < 16; ++j) \| for (int run = 0; run < 2; ++run)` |
| compute.reduce | 0.12 | 1.9% | 1.0 | 0.13 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| store.global | 0.08 | 1.2% | 1.0 | 0.13 | `if (((int)threadIdx.x) == 0)` |

### gemm_mxfp8_partials_kernel grid (16, 5, 1) block (128, 1, 1)

51 launches, 320 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 12.32 us, warp 11.72 us, nsys graphs 12.77 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 5.44 | 46.4% | 5.80 | 47.3% |
| wait | 5.16 | 44.0% | 4.16 | 34.0% |
| compute | 0.62 | 5.3% | 0.64 | 5.2% |
| load | 0.46 | 3.9% | 1.61 | 13.1% |
| store | 0.05 | 0.5% | 0.05 | 0.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.cp_async#3 | 1.92 | 16.4% | 1.0 | 3.90 | `tl::cp_async_wait<0>();` |
| wait.cp_async#2 | 1.13 | 9.6% | 1.0 | 2.21 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>();` |
| wait.sync#2*2 (est.: sampled iterations x2) | 0.81 | 6.9% | 1.0 | 2.43 | `__syncthreads();` |
| wait.sync#3 | 0.49 | 4.2% | 1.0 | 1.41 | `__syncthreads();` |
| wait.cp_async*2 (est.: sampled iterations x2) | 0.40 | 3.4% | 1.0 | 2.37 | `tl::cp_async_wait<2>();` |
| wait.sync#4 | 0.32 | 2.7% | 1.0 | 0.99 | `__syncthreads();` |
| compute.mma*2 (est.: sampled iterations x2) | 0.31 | 2.7% | 1.0 | 0.32 | `{` |
| load.cp_async | 0.23 | 2.0% | 1.0 | 0.32 | `for (int i = 0; i < 2; ++i) \| for (int i_1 = 0; i_1 < 4; ++i_1)` |
| load.cp_async#2*2 (est.: sampled iterations x2) | 0.22 | 1.9% | 1.0 | 0.26 | `for (int i_4 = 0; i_4 < 2; ++i_4)` |
| compute.mma#2 | 0.15 | 1.3% | 1.0 | 0.16 | `{` |
| compute.mma#3 | 0.15 | 1.3% | 1.0 | 0.16 | `{` |
| wait.sync*2 (est.: sampled iterations x2) | 0.09 | 0.8% | 1.0 | 0.13 | `__syncthreads();` |
| store.global | 0.05 | 0.5% | 1.0 | 0.06 | `for (int i_11 = 0; i_11 < 16; ++i_11) \| for (int i_13 = 0; i_13 < 8; ++i_13) \| for (int i_` |

### moe_grouped_fp8_fp4_gemm_kernel grid (9, 36, 1) block (128, 1, 1)

6 launches, 1296 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 100.05 us, warp 18.38 us, nsys graphs 3.26 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 12.05 | 65.6% | 0.04 | 11.1% |
| compute | 3.20 | 17.4% | 0.00 | 0.0% |
| load | 1.78 | 9.7% | 0.06 | 15.3% |
| unattributed | 1.27 | 6.9% | 0.28 | 73.6% |
| store | 0.07 | 0.4% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.mbarrier*40 (est.: sampled iterations x40) | 8.64 | 47.0% | 0.2 | 34.56 | `loaded[(ko % 3)].wait(((ko % 6) / 3));` |
| compute.mma*40 (est.: sampled iterations x40) | 3.20 | 17.4% | 0.2 | 12.80 | `{` |
| load#2*40 (est.: sampled iterations x40) | 1.52 | 8.3% | 0.2 | 8.96 | `if (ko < 38)` |
| wait.sync#5*40 (est.: sampled iterations x40) | 1.36 | 7.4% | 0.2 | 6.40 | `__syncthreads();` |
| wait.cp_async*40 (est.: sampled iterations x40) | 0.68 | 3.7% | 0.2 | 3.20 | `tl::cp_async_wait<2>();` |
| wait.sync#4*40 (est.: sampled iterations x40) | 0.64 | 3.5% | 0.2 | 2.56 | `__syncthreads();` |
| wait.sync#6*40 (est.: sampled iterations x40) | 0.64 | 3.5% | 0.2 | 2.56 | `__syncthreads();` |
| load.global#3 | 0.11 | 0.6% | 0.2 | 0.45 | `if (((expert < 128) && (0 <= expert))) \| if (((expert < 384) && (0 <= expert)))` |
| load.global | 0.08 | 0.4% | 1.0 | 0.18 | `int expert = TileExpert[((int64_t)((int)blockIdx.y))];` |
| store.global | 0.07 | 0.4% | 0.2 | 0.35 | `for (int i_2 = 0; i_2 < 16; ++i_2) \| for (int i_3 = 0; i_3 < 16; ++i_3) \| for (int i_4 = 0` |
| load | 0.04 | 0.2% | 0.2 | 0.22 | `for (int s = 0; s < 2; ++s)` |
| wait.sync | 0.04 | 0.2% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#2 | 0.03 | 0.2% | 0.2 | 0.16 | `__syncthreads();` |
| load.global#2 | 0.03 | 0.1% | 0.2 | 0.19 | `if (((int)threadIdx.x) < 32) \| if (((int)threadIdx.x) < 64)` |
| wait.sync#3 | 0.01 | 0.1% | 0.2 | 0.06 | `__syncthreads();` |

### router_partials_kernel grid (3, 1, 1) block (64, 1, 1)

102 launches, 6 warps; ranges, warp and last warp from the 3 recording CTAs' 6 warps; span 5.18 us, warp 3.12 us, nsys graphs 4.86 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.34 | 42.8% | 2.00 | 37.2% |
| load | 0.98 | 31.3% | 1.80 | 33.6% |
| compute | 0.68 | 21.9% | 1.37 | 25.4% |
| store | 0.13 | 4.1% | 0.20 | 3.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.reduce | 0.68 | 21.9% | 0.5 | 1.38 | `for (int k = 0; k < 3; ++k) \| for (int k = 0; k < 6; ++k)` |
| load.global#4 | 0.50 | 15.9% | 0.5 | 1.06 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < image_len)) \| if ((((((int)bl` |
| load.global | 0.37 | 12.0% | 0.8 | 0.45 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < mask_len))` |
| load.global#2 | 0.05 | 1.7% | 0.8 | 0.06 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < mask_len))` |
| load.global#3 | 0.05 | 1.7% | 0.8 | 0.06 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < mask_len))` |
| store.global#3 | 0.05 | 1.5% | 0.5 | 0.10 | `if ((((int)threadIdx.x) & 31) < 3) \| if ((((int)threadIdx.x) & 31) < 6)` |
| store.global | 0.04 | 1.4% | 0.8 | 0.06 | `if (condval == (bool)0) \| if (condval == (bool)1)` |
| store.global#2 | 0.04 | 1.2% | 0.8 | 0.06 | `if (condval_1 == (bool)0) \| if (condval_1 == (bool)1)` |

### metadata_chunk_kernel grid (10, 1, 1) block (192, 1, 1)

51 launches, 60 warps; ranges, warp and last warp from the 5 recording CTAs' 30 warps; span 10.02 us, warp 6.79 us, nsys graphs 12.00 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 4.28 | 63.2% | 8.00 | 74.4% |
| unattributed | 2.43 | 35.8% | 2.62 | 24.4% |
| store | 0.07 | 1.0% | 0.13 | 1.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 4.24 | 62.6% | 1.0 | 6.78 | `if ((((offset + ((int)blockIdx.x)) < src_rows) && (0 <= (offset + ((int)blockIdx.x)))))` |
| store.global | 0.05 | 0.8% | 1.0 | 0.10 | `swa[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = con` |
| load.global#2 | 0.04 | 0.6% | 0.2 | 0.38 | `if (((0 <= req) & ((int64_t)0 <= pos)))` |
| store.global#3 | 0.01 | 0.1% | 0.2 | 0.06 | `top_lengths[((int64_t)((int)blockIdx.x))] = condval_12; \| top_lengths[((int64_t)((int)bloc` |
| store.global#2 | 0.01 | 0.1% | 0.2 | 0.06 | `lengths[((int64_t)((int)blockIdx.x))] = ((int)max(info[1], (int64_t)0));` |

### norm_forward_and_per_token_cast_kernel_kernel grid (1, 1, 1) block (128, 1, 1)

187 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 2.72 us, warp 2.72 us, nsys graphs 1.28 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 2.10 | 77.4% | 2.22 | 76.6% |
| load | 0.38 | 14.1% | 0.41 | 14.3% |
| compute | 0.11 | 4.1% | 0.11 | 3.7% |
| wait | 0.10 | 3.8% | 0.10 | 3.4% |
| unattributed | 0.02 | 0.6% | 0.06 | 2.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*4 (est.: sampled iterations x4) | 1.54 | 56.5% | 1.0 | 1.54 | `for (int j_2 = 0; j_2 < 2; ++j_2)` |
| store.global*4 (est.: sampled iterations x4) | 0.35 | 12.9% | 1.0 | 0.38 | `for (int j_1 = 0; j_1 < 4; ++j_1)` |
| load.global*4 (est.: sampled iterations x4) | 0.22 | 8.2% | 1.0 | 0.26 | `if (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 4)) < num_tokens)` |
| load.global#3*4 (est.: sampled iterations x4) | 0.13 | 4.7% | 1.0 | 0.13 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 128) + ((((int)threadIdx.x) & 15` |
| store.global#5*4 (est.: sampled iterations x4) | 0.13 | 4.7% | 0.8 | 0.13 | `*(uint4*)(out + ((((((int64_t)vi_1) * (int64_t)128) + ((((int64_t)((int)threadIdx.x)) & (i` |
| compute.reduce | 0.11 | 4.1% | 1.0 | 0.13 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| wait.sync | 0.10 | 3.8% | 1.0 | 0.29 | `__syncthreads();` |
| store.global#2 | 0.05 | 1.8% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| store.global#3 | 0.04 | 1.5% | 0.8 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |
| load.global#2 | 0.03 | 1.2% | 1.0 | 0.03 | `*(float2*)(row_partial + 0) = *(float2*)(sum_shared + ((((int)threadIdx.x) >> 6) * 2)); \| ` |
| load.global#3*2 (est.: sampled iterations x2) | 0.00 | 0.0% | 0.0 | 0.00 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 64) + ((((int)threadIdx.x) & 7) ` |
| load.global*2 (est.: sampled iterations x2) | 0.00 | 0.0% | 0.0 | 0.00 | `if (((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < num_tokens)` |
| store.global#4*2 (est.: sampled iterations x2) | 0.00 | 0.0% | 0.0 | 0.00 | `for (int j_2 = 0; j_2 < 2; ++j_2)` |
| store.global#5*2 (est.: sampled iterations x2) | 0.00 | 0.0% | 0.0 | 0.00 | `*(uint4*)(out + ((((((int64_t)vi_1) * (int64_t)64) + ((((int64_t)((int)threadIdx.x)) & (in` |
| store.global*2 (est.: sampled iterations x2) | 0.00 | 0.0% | 0.0 | 0.00 | `for (int j_1 = 0; j_1 < 4; ++j_1)` |

### rope_kernel grid (1, 1, 1) block (128, 1, 1)

272 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 1.26 us, warp 0.90 us, nsys graphs 1.31 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.74 | 82.1% | 1.61 | 83.0% |
| load | 0.14 | 16.1% | 0.29 | 14.9% |
| store | 0.02 | 1.8% | 0.04 | 2.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.10 | 10.7% | 1.0 | 0.10 | `if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < tokens)) \| int64_t position ` |
| load.global#2 | 0.05 | 5.4% | 0.5 | 0.10 | `if (((position < ((int64_t)positions_max)) && ((int64_t)0 <= position)))` |
| store.global | 0.02 | 1.8% | 0.5 | 0.03 | `*(uint4*)(x + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + ((((int64_t)((int)thread` |

### moe_swiglu_cast_kernel_kernel grid (8, 30, 1) block (288, 1, 1)

147 launches, 2160 warps; ranges, warp and last warp from the 4 recording CTAs' 36 warps; span 2.82 us, warp 1.10 us, nsys graphs 3.04 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.64 | 58.6% | 1.26 | 54.0% |
| load | 0.24 | 21.7% | 0.46 | 19.6% |
| compute | 0.12 | 10.7% | 0.39 | 16.6% |
| store | 0.10 | 9.1% | 0.23 | 9.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.24 | 21.7% | 1.0 | 0.35 | `for (int r = 0; r < 4; ++r)` |
| compute.reduce*4 (est.: sampled iterations x4) | 0.12 | 10.7% | 0.5 | 0.38 | `if (0 <= orders[r_2])` |
| store.global*4 (est.: sampled iterations x4) | 0.10 | 9.1% | 0.5 | 0.26 | `if (0 <= orders[r_2])` |

### metadata_builder_kernel grid (2, 1, 1) block (128, 1, 1)

289 launches, 8 warps; ranges, warp and last warp from the 2 recording CTAs' 8 warps; span 1.22 us, warp 0.68 us, nsys graphs 1.47 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.31 | 45.6% | 0.34 | 24.4% |
| load | 0.29 | 43.2% | 0.98 | 71.3% |
| store | 0.08 | 11.2% | 0.06 | 4.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#3 | 0.25 | 36.7% | 0.5 | 0.83 | `for (int __1 = 0; __1 < 2; ++__1)` |
| store.global#4 | 0.03 | 4.7% | 0.5 | 0.06 | `positions[(((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))` |
| load.global#2 | 0.03 | 4.1% | 0.1 | 0.22 | `if ((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) < nr))` |
| store.global#3 | 0.03 | 4.1% | 0.5 | 0.06 | `if (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) == 0)` |
| load.global | 0.02 | 2.4% | 0.1 | 0.13 | `if ((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) <= nr))` |
| store.global | 0.01 | 1.8% | 0.1 | 0.10 | `out_starts[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] = condval;` |
| store.global#2 | 0.00 | 0.6% | 0.1 | 0.03 | `request_positions[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] = condval_2;` |

### token_ranks_kernel grid (1, 1, 1) block (256, 1, 1)

51 launches, 8 warps; ranges, warp and last warp from the 1 recording CTAs' 8 warps; span 7.62 us, warp 7.61 us, nsys graphs 7.90 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 6.84 | 89.9% | 6.87 | 89.9% |
| unattributed | 0.39 | 5.1% | 0.36 | 4.7% |
| compute | 0.34 | 4.4% | 0.36 | 4.8% |
| store | 0.05 | 0.6% | 0.05 | 0.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 6.84 | 89.9% | 1.0 | 7.01 | `if ((((int64_t)0 <= token_ids[((int64_t)((int)blockIdx.x))]) && (token_ids[((int64_t)((int` |
| compute.reduce | 0.34 | 4.4% | 1.0 | 0.51 | `total[0] = tl::AllReduce<tl::SumOp, 256, 1, 0, tl::NamedBarrier<256>>::run(total[0], (&(wo` |
| store.global | 0.05 | 0.6% | 1.0 | 0.06 | `output[((int64_t)((int)blockIdx.x))] = ((int64_t)total[0]);` |

### indexer_topk_segment_kernel grid (1, 5, 1) block (512, 1, 1)

32 launches, 80 warps; ranges, warp and last warp from the 5 recording CTAs' 80 warps; span 10.64 us, warp 10.59 us, nsys graphs 0.00 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed, with loops' unsampled iterations | 5.56 | 52.5% | 5.46 | 50.9% |
| wait | 3.78 | 35.7% | 3.88 | 36.2% |
| load | 1.08 | 10.2% | 1.11 | 10.4% |
| store | 0.17 | 1.6% | 0.28 | 2.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.08 | 10.2% | 1.0 | 1.14 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 4096); \| i` |
| wait.sync (sampled iterations) | 1.01 | 9.5% | 14.5 | 1.17 | `__syncthreads();` |
| wait.sync#20*9 (est.: sampled iterations x9) | 0.31 | 2.9% | 0.5 | 0.43 | `__syncthreads();` |
| wait.sync#24*9 (est.: sampled iterations x9) | 0.31 | 2.9% | 0.5 | 0.43 | `__syncthreads();` |
| wait.sync#19*9 (est.: sampled iterations x9) | 0.30 | 2.9% | 0.5 | 0.43 | `__syncthreads();` |
| wait.sync#23*9 (est.: sampled iterations x9) | 0.29 | 2.8% | 0.5 | 0.43 | `__syncthreads();` |
| wait.sync#14*8 (est.: sampled iterations x8) | 0.28 | 2.7% | 0.5 | 0.38 | `__syncthreads();` |
| wait.sync#6*8 (est.: sampled iterations x8) | 0.27 | 2.6% | 0.5 | 0.38 | `__syncthreads();` |
| wait.sync#5*8 (est.: sampled iterations x8) | 0.27 | 2.5% | 0.5 | 0.38 | `__syncthreads();` |
| wait.sync#13*8 (est.: sampled iterations x8) | 0.26 | 2.5% | 0.5 | 0.38 | `__syncthreads();` |
| store.global | 0.17 | 1.6% | 1.5 | 0.22 | `for (int i_3 = 0; i_3 < 4; ++i_3) \| if (((int)threadIdx.x) < total) \| if (((int)threadIdx.` |
| wait.sync#25 | 0.05 | 0.4% | 0.5 | 0.08 | `__syncthreads();` |
| wait.sync#17 | 0.04 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#16 | 0.04 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#8 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#18 | 0.03 | 0.3% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#2 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#9 | 0.03 | 0.3% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#7 | 0.03 | 0.3% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#15 | 0.03 | 0.3% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#11 | 0.03 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#3 | 0.03 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#10 | 0.03 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#4 | 0.03 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#22 | 0.03 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#12 | 0.02 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#21 | 0.02 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |

### topk_ids_kernel grid (5, 1, 1) block (512, 1, 1)

16 launches, 80 warps; ranges, warp and last warp from the 5 recording CTAs' 80 warps; span 21.17 us, warp 19.13 us, nsys graphs 20.62 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 15.43 | 80.7% | 15.53 | 73.4% |
| unattributed | 2.31 | 12.1% | 1.94 | 9.2% |
| wait | 0.92 | 4.8% | 0.73 | 3.4% |
| store | 0.46 | 2.4% | 2.96 | 14.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 15.43 | 80.7% | 1.0 | 16.38 | `for (int t = 0; t < (vocab >> 14); ++t)` |
| wait.sync | 0.78 | 4.1% | 1.0 | 1.50 | `__syncthreads();` |
| store.global | 0.46 | 2.4% | 1.0 | 3.01 | `if ((((int)threadIdx.x) >> 5) == 0)` |
| wait.sync#2 | 0.08 | 0.4% | 1.0 | 0.16 | `__syncthreads();` |
| wait.sync#3 | 0.06 | 0.3% | 1.0 | 0.06 | `__syncthreads();` |

### metadata_decode_kernel grid (5, 2, 1) block (128, 1, 1)

64 launches, 40 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 2.69 us, warp 1.89 us, nsys graphs 5.33 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.99 | 52.4% | 2.96 | 58.7% |
| unattributed | 0.81 | 43.0% | 1.39 | 27.5% |
| store | 0.09 | 4.6% | 0.70 | 13.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.74 | 39.3% | 0.6 | 1.44 | `if ((((offset + ((int)blockIdx.x)) < src_rows) && (0 <= (offset + ((int)blockIdx.x)))))` |
| load.global#3 | 0.22 | 11.5% | 0.4 | 0.58 | `if ((((offset + ((int)blockIdx.x)) < main_src_rows) && (0 <= (offset + ((int)blockIdx.x)))` |
| store.global#4 | 0.04 | 2.2% | 0.4 | 0.22 | `if (0 < live)` |
| store.global | 0.03 | 1.7% | 0.6 | 0.06 | `swa[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = con` |
| load.global#2 | 0.03 | 1.6% | 0.1 | 0.22 | `if (((0 <= req) & ((int64_t)0 <= pos)))` |
| store.global#2 | 0.01 | 0.3% | 0.1 | 0.06 | `lengths[((int64_t)((int)blockIdx.x))] = ((int)max(info[1], (int64_t)0));` |
| store.global#3 | 0.01 | 0.3% | 0.1 | 0.06 | `top_lengths[((int64_t)((int)blockIdx.x))] = condval_6; \| top_lengths[((int64_t)((int)block` |

### moe_grouped_fp8_fp4_gemm_kernel grid (40, 36, 1) block (128, 1, 1)

6 launches, 5760 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 51.82 us, warp 3.00 us, nsys graphs 9.65 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.46 | 48.9% | 0.20 | 66.7% |
| load | 0.52 | 17.2% | 0.06 | 21.1% |
| wait | 0.50 | 16.8% | 0.04 | 12.3% |
| compute | 0.44 | 14.7% | 0.00 | 0.0% |
| store | 0.07 | 2.4% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*5 (est.: sampled iterations x5) | 0.44 | 14.7% | 0.2 | 1.76 | `if (ko < 4)` |
| load#2*5 (est.: sampled iterations x5) | 0.22 | 7.3% | 0.2 | 0.96 | `if (ko < 3)` |
| wait.mbarrier*5 (est.: sampled iterations x5) | 0.12 | 4.0% | 0.2 | 0.48 | `loaded[(ko % 3)].wait((ko / 3));` |
| load.global | 0.11 | 3.8% | 1.0 | 0.21 | `int expert = TileExpert[((int64_t)((int)blockIdx.y))];` |
| load.global#3 | 0.11 | 3.7% | 0.2 | 0.45 | `if (((expert < 128) && (0 <= expert))) \| if (((expert < 384) && (0 <= expert)))` |
| wait.sync#4*5 (est.: sampled iterations x5) | 0.08 | 2.7% | 0.2 | 0.32 | `__syncthreads();` |
| wait.sync#5*5 (est.: sampled iterations x5) | 0.08 | 2.7% | 0.2 | 0.32 | `__syncthreads();` |
| store.global | 0.07 | 2.4% | 0.2 | 0.37 | `for (int i_2 = 0; i_2 < 16; ++i_2) \| for (int i_3 = 0; i_3 < 16; ++i_3) \| for (int i_4 = 0` |
| wait.sync#6*5 (est.: sampled iterations x5) | 0.07 | 2.3% | 0.2 | 0.32 | `__syncthreads();` |
| wait.cp_async*5 (est.: sampled iterations x5) | 0.07 | 2.2% | 0.2 | 0.32 | `tl::cp_async_wait<2>();` |
| load | 0.04 | 1.5% | 0.2 | 0.22 | `for (int s = 0; s < 2; ++s)` |
| wait.sync | 0.04 | 1.4% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#2 | 0.04 | 1.3% | 0.2 | 0.19 | `__syncthreads();` |
| load.global#2 | 0.03 | 0.9% | 0.2 | 0.22 | `if (((int)threadIdx.x) < 32) \| if (((int)threadIdx.x) < 64)` |
| wait.sync#3 | 0.01 | 0.3% | 0.2 | 0.03 | `__syncthreads();` |

### norm_forward_and_per_token_cast_kernel_kernel grid (3, 1, 1) block (128, 1, 1)

102 launches, 12 warps; ranges, warp and last warp from the 3 recording CTAs' 12 warps; span 2.75 us, warp 2.54 us, nsys graphs 1.92 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 1.61 | 63.5% | 1.94 | 64.4% |
| load | 0.34 | 13.2% | 0.34 | 11.4% |
| unattributed | 0.33 | 13.1% | 0.57 | 18.8% |
| wait | 0.14 | 5.5% | 0.05 | 1.5% |
| compute | 0.12 | 4.6% | 0.12 | 3.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*3 (est.: sampled iterations x3) | 1.15 | 45.4% | 1.0 | 1.15 | `for (int j_2 = 0; j_2 < 2; ++j_2)` |
| store.global*3 (est.: sampled iterations x3) | 0.29 | 11.4% | 1.0 | 0.38 | `for (int j_1 = 0; j_1 < 4; ++j_1)` |
| load.global*3 (est.: sampled iterations x3) | 0.18 | 6.9% | 1.0 | 0.19 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 6)) < num_tokens) && (((vi * 2) + (` |
| wait.sync | 0.14 | 5.5% | 1.0 | 0.26 | `__syncthreads();` |
| load.global#3*3 (est.: sampled iterations x3) | 0.13 | 5.0% | 1.0 | 0.19 | `if (((vi_1 * 2) + ((((int)threadIdx.x) & 63) >> 5)) < 5)` |
| compute.reduce | 0.12 | 4.6% | 1.0 | 0.13 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| store.global#5*3 (est.: sampled iterations x3) | 0.10 | 4.1% | 0.8 | 0.19 | `*(uint4*)(out + ((((((int64_t)vi_1) * (int64_t)512) + ((((int64_t)((int)threadIdx.x)) & (i` |
| store.global#2 | 0.04 | 1.6% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| load.global#2 | 0.03 | 1.3% | 1.0 | 0.03 | `*(float2*)(row_partial + 0) = *(float2*)(sum_shared + ((((int)threadIdx.x) >> 6) * 2)); \| ` |
| store.global#3 | 0.03 | 1.1% | 0.4 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |

### topk_log_softmax_kernel grid (1, 1, 1) block (1024, 1, 1)

17 launches, 32 warps; ranges, warp and last warp from the 1 recording CTAs' 32 warps; span 16.42 us, warp 15.85 us, nsys graphs 15.97 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 12.84 | 81.0% | 13.44 | 81.6% |
| unattributed, with loops' unsampled iterations | 1.55 | 9.8% | 1.72 | 10.4% |
| wait | 1.07 | 6.8% | 0.84 | 5.1% |
| compute | 0.39 | 2.5% | 0.40 | 2.4% |
| store | 0.00 | 0.0% | 0.07 | 0.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 6.59 | 41.6% | 1.0 | 6.85 | `if (0 < (vocab & 2047)) \| if (0 < (vocab & 8191))` |
| load.global#2 | 6.24 | 39.3% | 1.0 | 6.66 | `for (int step_1 = 0; step_1 < (vocab >> 11); ++step_1) \| for (int step_1 = 0; step_1 < (vo` |
| wait.sync#2 | 0.50 | 3.2% | 1.0 | 0.93 | `__syncthreads();` |
| wait.sync | 0.32 | 2.0% | 1.0 | 0.64 | `__syncthreads();` |
| compute.reduce#3 | 0.17 | 1.1% | 1.0 | 0.19 | `ties[0] = vllm_add(ties[0], __shfl_xor_sync((uint)4294967295, ties[0], 16, 32));` |
| compute.reduce#2 | 0.11 | 0.7% | 1.0 | 0.16 | `se[0] = vllm_add(se[0], __shfl_xor_sync((uint)4294967295, se[0], 16, 32));` |
| compute.reduce | 0.10 | 0.7% | 1.0 | 0.13 | `best[0] = vllm_max(best[0], __shfl_xor_sync((uint)4294967295, best[0], 16, 32));` |
| wait.sync#4 | 0.10 | 0.6% | 1.0 | 0.13 | `__syncthreads();` |
| wait.sync#5 | 0.08 | 0.5% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#3 | 0.08 | 0.5% | 1.0 | 0.10 | `__syncthreads();` |
| load.global#3 (sampled iterations) | 0.01 | 0.1% | 0.0 | 0.38 | `float value_32 = vllm_sub(vllm_shifted(logits[((((int64_t)((int)blockIdx.x)) * logits_stri` |
| store.global (sampled iterations) | 0.00 | 0.0% | 0.0 | 0.06 | `output[(((((int64_t)c) * (int64_t)1024) + (((int64_t)((int)blockIdx.x)) * ((int64_t)topk))` |

### topk_log_softmax_kernel grid (5, 1, 1) block (1024, 1, 1)

16 launches, 160 warps; ranges, warp and last warp from the 5 recording CTAs' 160 warps; span 16.96 us, warp 16.21 us, nsys graphs 16.66 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 13.19 | 81.4% | 14.02 | 82.7% |
| unattributed, with loops' unsampled iterations | 1.62 | 10.0% | 1.75 | 10.3% |
| wait | 1.00 | 6.2% | 0.75 | 4.4% |
| compute | 0.39 | 2.4% | 0.37 | 2.2% |
| store | 0.00 | 0.0% | 0.07 | 0.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 6.68 | 41.2% | 1.0 | 6.99 | `if (0 < (vocab & 2047)) \| if (0 < (vocab & 8191))` |
| load.global#2 | 6.50 | 40.1% | 1.0 | 7.09 | `for (int step_1 = 0; step_1 < (vocab >> 11); ++step_1) \| for (int step_1 = 0; step_1 < (vo` |
| wait.sync#2 | 0.45 | 2.8% | 1.0 | 1.04 | `__syncthreads();` |
| wait.sync | 0.29 | 1.8% | 1.0 | 0.90 | `__syncthreads();` |
| compute.reduce#3 | 0.17 | 1.1% | 1.0 | 0.22 | `ties[0] = vllm_add(ties[0], __shfl_xor_sync((uint)4294967295, ties[0], 16, 32));` |
| compute.reduce#2 | 0.11 | 0.7% | 1.0 | 0.16 | `se[0] = vllm_add(se[0], __shfl_xor_sync((uint)4294967295, se[0], 16, 32));` |
| compute.reduce | 0.11 | 0.7% | 1.0 | 0.16 | `best[0] = vllm_max(best[0], __shfl_xor_sync((uint)4294967295, best[0], 16, 32));` |
| wait.sync#4 | 0.10 | 0.6% | 1.0 | 0.16 | `__syncthreads();` |
| wait.sync#5 | 0.08 | 0.5% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#3 | 0.08 | 0.5% | 1.0 | 0.10 | `__syncthreads();` |
| load.global#3 (sampled iterations) | 0.01 | 0.1% | 0.0 | 0.38 | `float value_32 = vllm_sub(vllm_shifted(logits[((((int64_t)((int)blockIdx.x)) * logits_stri` |
| store.global (sampled iterations) | 0.00 | 0.0% | 0.0 | 0.06 | `output[(((((int64_t)c) * (int64_t)1024) + (((int64_t)((int)blockIdx.x)) * ((int64_t)topk))` |

### sparse_mla_kernel_1 grid (16, 5, 1) block (128, 1, 1)

96 launches, 80 CTAs, 320 warps; span 1.86 us, warp 1.39 us, nsys graphs 1.89 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.34 | 96.4% | 1.12 | 41.4% |
| load | 0.03 | 2.4% | 1.34 | 49.5% |
| wait | 0.01 | 0.8% | 0.12 | 4.5% |
| compute | 0.00 | 0.3% | 0.08 | 3.0% |
| store | 0.00 | 0.1% | 0.04 | 1.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.03 | 2.4% | 0.1 | 0.83 | `M = __fmul_rn(Sinks[((int)blockIdx.x)], 0x1.7154764ee6c2fp+0f/*1.442695e+00*/); \| if (((in` |
| wait.sync | 0.01 | 0.8% | 0.1 | 0.32 | `__syncthreads();` |
| compute.reduce | 0.00 | 0.3% | 0.1 | 0.10 | `amax[0] = tl::AllReduce<tl::MaxOp, 8, 1, 0, tl::NamedBarrier<128>>::run(amax[0]);` |
| store.global | 0.00 | 0.1% | 0.1 | 0.06 | `*(uint2*)(Output + (((((int64_t)((int)blockIdx.y)) * (int64_t)8192) + (((int64_t)((int)blo` |
| compute.reduce#3 | 0.00 | 0.0% | 0.0 | 0.00 | `amax[0] = tl::AllReduce<tl::MaxOp, 8, 1, 0, tl::NamedBarrier<128>>::run(amax[0]);` |
| load.global#3 | 0.00 | 0.0% | 0.0 | 0.00 | `M = __fmul_rn(Sinks[((int)blockIdx.x)], 0x1.7154764ee6c2fp+0f/*1.442695e+00*/);` |
| load.global#4 | 0.00 | 0.0% | 0.0 | 0.00 | `if (((int)threadIdx.x) < 32)` |
| store.global#2 | 0.00 | 0.0% | 0.0 | 0.00 | `*(uint2*)(Output + (((((int64_t)((int)blockIdx.y)) * (int64_t)8192) + (((int64_t)((int)blo` |
| wait.sync#12 | 0.00 | 0.0% | 0.0 | 0.00 | `__syncthreads();` |
| wait.sync#13 | 0.00 | 0.0% | 0.0 | 0.00 | `__syncthreads();` |

### cache_writer_kernel grid (10, 1, 1) block (128, 1, 1)

51 launches, 40 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 5.73 us, warp 5.70 us, nsys graphs 6.30 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 3.73 | 65.5% | 3.00 | 57.7% |
| store | 1.59 | 28.0% | 1.71 | 32.8% |
| load | 0.29 | 5.1% | 0.40 | 7.8% |
| compute | 0.08 | 1.4% | 0.09 | 1.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 1.59 | 28.0% | 1.0 | 1.63 | `if (((int64_t)0 <= slot) & ((slot >> (int64_t)7) < ((int64_t)pages))) \| if (((int64_t)0 <=` |
| load.global | 0.23 | 4.1% | 1.0 | 0.26 | `for (int i = 0; i < 4; ++i)` |
| compute.reduce | 0.08 | 1.4% | 1.0 | 0.10 | `amax = max(amax, __shfl_xor_sync((uint)4294967295, amax, 1, 32));` |
| load.global#2 | 0.06 | 1.0% | 1.0 | 0.06 | `int64_t slot = slots[((int64_t)((int)blockIdx.x))];` |

### per_token_cast_kernel_kernel grid (9, 1, 1) block (64, 1, 1)

153 launches, 18 warps; ranges, warp and last warp from the 5 recording CTAs' 10 warps; span 1.06 us, warp 0.96 us, nsys graphs 1.17 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.44 | 46.0% | 0.84 | 50.7% |
| load | 0.31 | 32.0% | 0.42 | 25.6% |
| store | 0.12 | 12.0% | 0.29 | 17.3% |
| compute | 0.10 | 10.0% | 0.11 | 6.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.22 | 22.7% | 1.0 | 0.26 | `for (int i = 0; i < 2; ++i)` |
| store.global#2 | 0.12 | 12.0% | 1.0 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.10 | 10.0% | 1.0 | 0.13 | `amax_fragment[i_1] = tl::AllReduce<tl::MaxOp, 2, 1, 0, tl::NamedBarrier<64>>::run(amax_fra` |
| load.global#2*2 (est.: sampled iterations x2) | 0.09 | 9.3% | 1.0 | 0.13 | `amax_fragment[i_1] = cutlass::bfloat16_t(__hmax((bfloat16_t(((nv_bfloat162*)(&(amax_fragme` |

### metadata_chunk_kernel grid (5, 1, 1) block (128, 1, 1)

32 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 4.98 us, warp 4.02 us, nsys graphs 5.20 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 2.62 | 65.1% | 2.56 | 33.2% |
| load | 1.33 | 33.2% | 5.01 | 64.9% |
| store | 0.07 | 1.7% | 0.14 | 1.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.21 | 30.1% | 1.0 | 1.58 | `if ((((offset + ((int)blockIdx.x)) < src_rows) && (0 <= (offset + ((int)blockIdx.x)))))` |
| load.global#2 | 0.12 | 3.0% | 0.2 | 0.53 | `if (((0 <= req) & ((int64_t)0 <= pos)))` |
| store.global | 0.05 | 1.2% | 1.0 | 0.06 | `swa[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = con` |
| store.global#2 | 0.01 | 0.3% | 0.2 | 0.06 | `lengths[((int64_t)((int)blockIdx.x))] = ((int)max(info[1], (int64_t)0));` |
| store.global#3 | 0.01 | 0.2% | 0.2 | 0.05 | `top_lengths[((int64_t)((int)blockIdx.x))] = condval_12; \| top_lengths[((int64_t)((int)bloc` |

### moe_combine_kernel grid (5, 5, 1) block (128, 1, 1)

96 launches, 100 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 2.24 us, warp 1.71 us, nsys graphs 2.43 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.27 | 74.3% | 1.88 | 77.6% |
| unattributed | 0.38 | 22.5% | 0.49 | 20.3% |
| store | 0.06 | 3.3% | 0.05 | 2.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.27 | 74.3% | 1.0 | 1.79 | `for (int k = 0; k < 3; ++k) \| for (int k = 0; k < 6; ++k) \| if ((((int)blockIdx.y) < share` |
| store.global | 0.06 | 3.3% | 1.0 | 0.06 | `{` |

### local_stats_kernel grid (5, 16, 1) block (128, 1, 1)

16 launches, 320 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 12.62 us, warp 11.79 us, nsys graphs 12.29 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 10.71 | 90.8% | 11.49 | 91.1% |
| unattributed | 0.69 | 5.9% | 0.70 | 5.6% |
| compute | 0.17 | 1.4% | 0.17 | 1.3% |
| wait | 0.16 | 1.4% | 0.16 | 1.3% |
| store | 0.07 | 0.6% | 0.09 | 0.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 10.71 | 90.8% | 1.0 | 11.23 | `if (expanded_local_pos[((int64_t)((int)blockIdx.x))] < num_speculative_steps)` |
| wait.sync | 0.16 | 1.4% | 1.0 | 0.51 | `__syncthreads();` |
| compute.reduce | 0.12 | 1.0% | 1.0 | 0.13 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| store.global | 0.07 | 0.6% | 1.0 | 0.13 | `if (((int)threadIdx.x) == 0)` |
| compute.reduce#2 | 0.06 | 0.5% | 1.0 | 0.06 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 2, 32);` |

### compress_kernel grid (2, 1, 1) block (128, 1, 1)

34 launches, 8 warps; ranges, warp and last warp from the 2 recording CTAs' 8 warps; span 6.50 us, warp 3.80 us, nsys graphs 6.93 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.72 | 45.2% | 1.92 | 33.1% |
| load | 1.58 | 41.6% | 2.94 | 50.6% |
| store | 0.44 | 11.6% | 0.85 | 14.6% |
| compute | 0.06 | 1.6% | 0.09 | 1.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.89 | 23.5% | 0.6 | 1.62 | `if ((((0 <= counts[0]) & (0 < counts[1])) & (counts[1] <= 2)) & (((((int)blockIdx.x) * 4) ` |
| load.global#2*4 (est.: sampled iterations x4) | 0.62 | 16.4% | 0.6 | 1.28 | `*(float4*)(v1 + 0) = *(float4*)(values + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048)` |
| store.global*4 (est.: sampled iterations x4) | 0.16 | 4.2% | 0.6 | 0.26 | `*(uint2*)(out + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + ((((int64_t)((int)thre` |
| store.global#3 | 0.10 | 2.6% | 0.3 | 0.16 | `emitted_slots[((((int64_t)((int)blockIdx.x)) * (int64_t)4) + (((int64_t)((int)threadIdx.x)` |
| store.global#2*4 (est.: sampled iterations x4) | 0.07 | 1.9% | 0.3 | 0.13 | `if ((slot >> (int64_t)3) < ((int64_t)blocks))` |
| compute.reduce | 0.06 | 1.6% | 0.6 | 0.10 | `float total = tl::warp_reduce_sum(ss[0]);` |
| store.global#5 | 0.05 | 1.3% | 0.3 | 0.18 | `emitted_slots[((((int64_t)((int)blockIdx.x)) * (int64_t)4) + (((int64_t)((int)threadIdx.x)` |
| load.global#3*4 (est.: sampled iterations x4) | 0.04 | 1.1% | 0.3 | 0.06 | `*(float4*)(w + 0) = *(float4*)(weight + ((c_1 * 128) + ((((int)threadIdx.x) & 31) * 4)));` |
| store.global#3*4 (est.: sampled iterations x4) | 0.04 | 1.1% | 0.3 | 0.06 | `*(uint2*)(out + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + ((((int64_t)((int)thre` |
| load.global#4 | 0.01 | 0.4% | 0.3 | 0.03 | `if (valid)` |
| load.global#3 | 0.01 | 0.3% | 0.3 | 0.03 | `if (valid)` |
| store.global#2 | 0.01 | 0.3% | 0.3 | 0.02 | `emitted[((((int64_t)((int)blockIdx.x)) * (int64_t)4) + (((int64_t)((int)threadIdx.x)) >> (` |
| store.global#4 | 0.01 | 0.3% | 0.3 | 0.02 | `emitted[((((int64_t)((int)blockIdx.x)) * (int64_t)4) + (((int64_t)((int)threadIdx.x)) >> (` |

### sparse_mla_kernel_1 grid (16, 10, 1) block (128, 1, 1)

51 launches, 160 CTAs, 640 warps; span 3.10 us, warp 2.50 us, nsys graphs 3.46 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 2.48 | 99.0% | 2.27 | 60.0% |
| load | 0.02 | 0.6% | 1.23 | 32.5% |
| wait | 0.01 | 0.2% | 0.17 | 4.5% |
| compute | 0.00 | 0.1% | 0.07 | 1.9% |
| store | 0.00 | 0.0% | 0.04 | 1.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#3 | 0.01 | 0.5% | 0.0 | 0.61 | `M = __fmul_rn(Sinks[((int)blockIdx.x)], 0x1.7154764ee6c2fp+0f/*1.442695e+00*/);` |
| wait.sync#13 | 0.00 | 0.2% | 0.0 | 0.22 | `__syncthreads();` |
| load.global#4 | 0.00 | 0.1% | 0.0 | 0.22 | `if (((int)threadIdx.x) < 32)` |
| compute.reduce#3 | 0.00 | 0.1% | 0.0 | 0.10 | `amax[0] = tl::AllReduce<tl::MaxOp, 8, 1, 0, tl::NamedBarrier<128>>::run(amax[0]);` |
| wait.sync#12 | 0.00 | 0.1% | 0.0 | 0.10 | `__syncthreads();` |
| store.global#2 | 0.00 | 0.0% | 0.0 | 0.06 | `*(uint2*)(Output + (((((int64_t)((int)blockIdx.y)) * (int64_t)8192) + (((int64_t)((int)blo` |

### norm_forward_and_per_token_cast_kernel_kernel grid (5, 1, 1) block (128, 1, 1)

67 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 2.78 us, warp 2.55 us, nsys graphs 1.89 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 1.63 | 63.9% | 1.93 | 68.7% |
| unattributed | 0.37 | 14.7% | 0.35 | 12.4% |
| load | 0.32 | 12.4% | 0.36 | 12.8% |
| compute | 0.12 | 4.7% | 0.12 | 4.3% |
| wait | 0.11 | 4.3% | 0.05 | 1.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*3 (est.: sampled iterations x3) | 1.14 | 44.7% | 1.0 | 1.15 | `for (int j_2 = 0; j_2 < 2; ++j_2)` |
| store.global*3 (est.: sampled iterations x3) | 0.30 | 11.7% | 1.0 | 0.38 | `for (int j_1 = 0; j_1 < 4; ++j_1)` |
| load.global*3 (est.: sampled iterations x3) | 0.16 | 6.4% | 1.0 | 0.19 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 6)) < num_tokens) && (((vi * 2) + (` |
| load.global#3*3 (est.: sampled iterations x3) | 0.12 | 4.7% | 1.0 | 0.19 | `if (((vi_1 * 2) + ((((int)threadIdx.x) & 63) >> 5)) < 5)` |
| store.global#5*3 (est.: sampled iterations x3) | 0.12 | 4.7% | 1.0 | 0.19 | `*(uint4*)(out + ((((((int64_t)vi_1) * (int64_t)512) + ((((int64_t)((int)threadIdx.x)) & (i` |
| compute.reduce | 0.12 | 4.7% | 1.0 | 0.13 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| wait.sync | 0.11 | 4.3% | 1.0 | 0.19 | `__syncthreads();` |
| store.global#2 | 0.04 | 1.6% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| load.global#2 | 0.03 | 1.3% | 1.0 | 0.03 | `*(float2*)(row_partial + 0) = *(float2*)(sum_shared + ((((int)threadIdx.x) >> 6) * 2)); \| ` |
| store.global#3 | 0.03 | 1.2% | 0.5 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |
| load.global#2*5 (est.: sampled iterations x5) | 0.00 | 0.0% | 0.0 | 0.00 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 1024) + (((int)threadIdx.x) * 8)` |
| load.global*5 (est.: sampled iterations x5) | 0.00 | 0.0% | 0.0 | 0.00 | `*(uint4*)(x_local + (vi * 8)) = *(uint4*)(x + ((((((int64_t)vi) * (int64_t)1024) + (((int6` |
| store.global#4*5 (est.: sampled iterations x5) | 0.00 | 0.0% | 0.0 | 0.00 | `for (int j_1 = 0; j_1 < 2; ++j_1)` |
| store.global*5 (est.: sampled iterations x5) | 0.00 | 0.0% | 0.0 | 0.00 | `for (int j = 0; j < 4; ++j)` |

### router_partials_kernel grid (5, 1, 1) block (64, 1, 1)

51 launches, 10 warps; ranges, warp and last warp from the 5 recording CTAs' 10 warps; span 2.75 us, warp 2.00 us, nsys graphs 2.34 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.82 | 41.1% | 1.64 | 44.3% |
| load | 0.82 | 40.8% | 1.49 | 40.4% |
| store | 0.19 | 9.4% | 0.21 | 5.7% |
| compute | 0.17 | 8.6% | 0.35 | 9.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.44 | 21.8% | 1.0 | 0.45 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < mask_len))` |
| load.global#4 | 0.25 | 12.6% | 0.5 | 0.54 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < image_len)) \| if ((((((int)bl` |
| compute.reduce | 0.17 | 8.6% | 0.5 | 0.35 | `for (int k = 0; k < 3; ++k) \| for (int k = 0; k < 6; ++k)` |
| store.global | 0.07 | 3.4% | 1.0 | 0.10 | `if (condval == (bool)0) \| if (condval == (bool)1)` |
| load.global#2 | 0.06 | 3.2% | 1.0 | 0.06 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < mask_len))` |
| load.global#3 | 0.06 | 3.2% | 1.0 | 0.06 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < mask_len))` |
| store.global#2 | 0.06 | 3.2% | 1.0 | 0.10 | `if (condval_1 == (bool)0) \| if (condval_1 == (bool)1)` |
| store.global#3 | 0.06 | 2.9% | 0.5 | 0.13 | `if ((((int)threadIdx.x) & 31) < 3) \| if ((((int)threadIdx.x) & 31) < 6)` |

### compress_gather_partial_state_kernel grid (2, 1, 1) block (128, 1, 1)

17 launches, 8 warps; ranges, warp and last warp from the 2 recording CTAs' 8 warps; span 11.20 us, warp 8.80 us, nsys graphs 0.00 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 5.00 | 56.9% | 3.31 | 31.8% |
| load | 3.29 | 37.4% | 5.29 | 50.8% |
| store | 0.50 | 5.7% | 1.82 | 17.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 3.17 | 36.0% | 1.0 | 4.42 | `int nt = counts[0];` |
| store.global*2 (est.: sampled iterations x2) | 0.45 | 5.1% | 1.0 | 0.83 | `if (half == 0)` |
| load.global#2*2 (est.: sampled iterations x2) | 0.12 | 1.4% | 1.0 | 0.13 | `for (int i = 0; i < 4; ++i)` |
| store.global#2 | 0.06 | 0.6% | 1.0 | 0.06 | `if (((int)threadIdx.x) == 0)` |

### per_token_cast_kernel_kernel grid (5, 1, 1) block (64, 1, 1)

136 launches, 10 warps; ranges, warp and last warp from the 5 recording CTAs' 10 warps; span 1.18 us, warp 1.08 us, nsys graphs 1.06 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.45 | 41.2% | 0.49 | 39.6% |
| load | 0.34 | 31.1% | 0.35 | 28.2% |
| store | 0.20 | 18.9% | 0.30 | 23.9% |
| compute | 0.10 | 8.8% | 0.10 | 8.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.25 | 23.2% | 1.0 | 0.26 | `for (int i = 0; i < 2; ++i)` |
| store.global#2 | 0.12 | 10.6% | 1.0 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.10 | 8.8% | 1.0 | 0.13 | `amax_fragment[i_1] = tl::AllReduce<tl::MaxOp, 2, 1, 0, tl::NamedBarrier<64>>::run(amax_fra` |
| load.global#2*2 (est.: sampled iterations x2) | 0.09 | 8.0% | 1.0 | 0.13 | `amax_fragment[i_1] = cutlass::bfloat16_t(__hmax((bfloat16_t(((nv_bfloat162*)(&(amax_fragme` |
| store.global#3*2 (est.: sampled iterations x2) | 0.04 | 4.1% | 0.5 | 0.13 | `*(fp8_e4_16_t*)(out + ((((((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int64_t)10240) +` |
| store.global*2 (est.: sampled iterations x2) | 0.04 | 4.1% | 0.5 | 0.13 | `out_sf[((((((int64_t)((int)blockIdx.x)) % (int64_t)15) * (int64_t)32) + (((int64_t)((int)t` |

### mhc_finalize_kernel grid (6, 1, 1) block (128, 1, 1)

12 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 13.09 us, warp 11.93 us, nsys graphs 7.60 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 7.75 | 64.9% | 8.32 | 62.5% |
| store | 1.44 | 12.1% | 1.67 | 12.5% |
| load | 1.09 | 9.1% | 1.77 | 13.3% |
| compute | 0.99 | 8.3% | 0.83 | 6.2% |
| wait | 0.66 | 5.5% | 0.73 | 5.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync | 0.66 | 5.5% | 1.0 | 0.88 | `__syncthreads();` |
| compute.reduce | 0.57 | 4.8% | 0.2 | 2.30 | `row_max[0] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<32>>::run(row_max[0]);` |
| store.global#4*7 (est.: sampled iterations x7) | 0.56 | 4.7% | 0.8 | 0.90 | `if ((((int)threadIdx.x) % 4) == 0)` |
| store.global | 0.55 | 4.6% | 0.2 | 2.21 | `if (((int)threadIdx.x) < 4)` |
| compute.reduce#2*7 (est.: sampled iterations x7) | 0.42 | 3.5% | 0.8 | 0.67 | `amax[0] = max(amax[0], __shfl_xor_sync((uint)4294967295, amax[0], 1, 32));` |
| load.global#4*7 (est.: sampled iterations x7) | 0.37 | 3.1% | 0.8 | 0.67 | `*(uint4*)(yv + 0) = *(uint4*)(C + ((((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((i` |
| load.global#3 | 0.35 | 2.9% | 0.8 | 0.46 | `for (int i_split_2 = 0; i_split_2 < 40; ++i_split_2)` |
| load.global | 0.33 | 2.8% | 1.0 | 0.99 | `if (((int)threadIdx.x) < 32)` |
| store.global#3*7 (est.: sampled iterations x7) | 0.32 | 2.7% | 0.8 | 0.45 | `*(uint4*)(y + ((((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((int64_t)step) * (int6` |
| load.global#2 | 0.04 | 0.4% | 0.2 | 0.19 | `cm[0] = ((mixes_shared[((((int)threadIdx.x) & 15) + 8)] * scale[2]) + base[((((int)threadI` |
| store.global#2 | 0.01 | 0.1% | 0.2 | 0.06 | `if ((((int)threadIdx.x) >> 4) == 0)` |

### indexer_topk_segment_kernel grid (2, 5, 1) block (512, 1, 1)

16 launches, 160 warps; ranges, warp and last warp from the 5 recording CTAs' 80 warps; span 9.92 us, warp 9.84 us, nsys graphs 0.00 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 6.10 | 62.0% | 6.00 | 60.3% |
| unattributed, with loops' unsampled iterations | 3.18 | 32.3% | 3.22 | 32.3% |
| load | 0.47 | 4.8% | 0.62 | 6.2% |
| store | 0.09 | 0.9% | 0.12 | 1.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync#24*9 (est.: sampled iterations x9) | 0.68 | 6.9% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#20*9 (est.: sampled iterations x9) | 0.65 | 6.6% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#23*9 (est.: sampled iterations x9) | 0.65 | 6.6% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#19*9 (est.: sampled iterations x9) | 0.64 | 6.5% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#14*8 (est.: sampled iterations x8) | 0.59 | 6.0% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync#6*8 (est.: sampled iterations x8) | 0.59 | 6.0% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync#5*8 (est.: sampled iterations x8) | 0.57 | 5.8% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync#13*8 (est.: sampled iterations x8) | 0.56 | 5.7% | 1.0 | 0.77 | `__syncthreads();` |
| load.global#2 | 0.26 | 2.7% | 1.0 | 0.32 | `for (int i = 0; i < 8; ++i)` |
| load.global | 0.21 | 2.1% | 1.0 | 0.22 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 4096); \| i` |
| wait.sync (sampled iterations) | 0.10 | 1.0% | 1.0 | 0.19 | `__syncthreads();` |
| wait.sync#25 | 0.10 | 1.0% | 1.0 | 0.16 | `__syncthreads();` |
| store.global | 0.09 | 0.9% | 1.0 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3) \| if (((int)threadIdx.x) < total) \| if (((int)threadIdx.` |
| wait.sync#17 | 0.08 | 0.8% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#18 | 0.08 | 0.8% | 1.0 | 0.16 | `__syncthreads();` |
| wait.sync#16 | 0.07 | 0.8% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#8 | 0.07 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#3 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#9 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#2 | 0.06 | 0.6% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#11 | 0.06 | 0.6% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#15 | 0.06 | 0.6% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#12 | 0.06 | 0.6% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#7 | 0.06 | 0.6% | 1.0 | 0.08 | `__syncthreads();` |
| wait.sync#4 | 0.06 | 0.6% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#22 | 0.06 | 0.6% | 1.0 | 0.08 | `__syncthreads();` |
| wait.sync#21 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#10 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |

### mhc_post_kernel grid (25, 1, 1) block (128, 1, 1)

64 launches, 100 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 1.89 us, warp 1.61 us, nsys graphs 2.14 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.46 | 90.7% | 1.97 | 91.8% |
| load | 0.10 | 6.4% | 0.11 | 5.2% |
| store | 0.05 | 3.0% | 0.07 | 3.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.10 | 6.4% | 1.0 | 0.13 | `*(float4*)(pm + 0) = *(float4*)(post + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int` |
| store.global | 0.05 | 3.0% | 1.0 | 0.06 | `for (int n_1 = 0; n_1 < 4; ++n_1)` |

### indexer_topk_merge_kernel grid (1, 5, 1) block (512, 1, 1)

16 launches, 80 warps; ranges, warp and last warp from the 5 recording CTAs' 80 warps; span 9.28 us, warp 9.03 us, nsys graphs 5.34 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 6.15 | 68.1% | 5.98 | 64.4% |
| unattributed | 2.22 | 24.6% | 2.37 | 25.6% |
| load | 0.58 | 6.4% | 0.78 | 8.4% |
| store | 0.08 | 0.9% | 0.15 | 1.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync#20*9 (est.: sampled iterations x9) | 0.68 | 7.5% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#24*9 (est.: sampled iterations x9) | 0.66 | 7.3% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#23*9 (est.: sampled iterations x9) | 0.65 | 7.2% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#19*9 (est.: sampled iterations x9) | 0.64 | 7.1% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#14*8 (est.: sampled iterations x8) | 0.60 | 6.7% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync#13*8 (est.: sampled iterations x8) | 0.59 | 6.5% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync#6*8 (est.: sampled iterations x8) | 0.59 | 6.5% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync#5*8 (est.: sampled iterations x8) | 0.57 | 6.4% | 1.0 | 0.77 | `__syncthreads();` |
| load.global | 0.57 | 6.3% | 1.0 | 0.70 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 8192);` |
| wait.sync | 0.15 | 1.6% | 1.0 | 0.22 | `__syncthreads();` |
| wait.sync#17 | 0.08 | 0.9% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#16 | 0.08 | 0.8% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#8 | 0.07 | 0.8% | 1.0 | 0.10 | `__syncthreads();` |
| store.global#2 | 0.07 | 0.8% | 0.9 | 0.10 | `Values[((((int64_t)((int)blockIdx.y)) * (int64_t)512) + ((int64_t)((int)threadIdx.x)))] = ` |
| wait.sync#2 | 0.07 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#3 | 0.07 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#9 | 0.07 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#25 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#15 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#7 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#11 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#12 | 0.06 | 0.7% | 1.0 | 0.08 | `__syncthreads();` |
| wait.sync#10 | 0.06 | 0.7% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#18 | 0.06 | 0.7% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#21 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#4 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#22 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| store.global | 0.01 | 0.1% | 0.1 | 0.13 | `Values[((((int64_t)((int)blockIdx.y)) * (int64_t)512) + ((int64_t)((int)threadIdx.x)))] = ` |
| load.global#2 | 0.01 | 0.1% | 0.1 | 0.10 | `if (((0 <= ((int*)sel_idx)[((int)threadIdx.x)]) && (((int*)sel_idx)[((int)threadIdx.x)] < ` |

### rope_kernel grid (5, 1, 1) block (128, 1, 1)

96 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 1.30 us, warp 1.26 us, nsys graphs 1.47 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.72 | 56.8% | 1.01 | 66.5% |
| load | 0.51 | 40.2% | 0.46 | 30.7% |
| store | 0.04 | 3.0% | 0.04 | 2.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.51 | 40.2% | 1.0 | 0.51 | `if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < tokens)) \| int64_t position ` |
| store.global | 0.04 | 3.0% | 1.0 | 0.06 | `*(uint4*)(x + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + ((((int64_t)((int)thread` |

### mhc_collapse_kernel grid (25, 1, 1) block (128, 1, 1)

48 launches, 100 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 2.66 us, warp 2.55 us, nsys graphs 4.00 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 2.40 | 94.4% | 2.78 | 92.4% |
| load | 0.10 | 3.8% | 0.18 | 5.9% |
| store | 0.05 | 1.9% | 0.05 | 1.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.10 | 3.8% | 1.0 | 0.13 | `*(float4*)(mv + 0) = *(float4*)(mix + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int6` |
| store.global | 0.05 | 1.9% | 1.0 | 0.06 | `*(uint4*)(out + ((((int64_t)((int)blockIdx.x)) * (int64_t)1024) + (((int64_t)((int)threadI` |

### norm_forward_and_per_token_cast_kernel_kernel grid (2, 1, 1) block (128, 1, 1)

51 launches, 8 warps; ranges, warp and last warp from the 2 recording CTAs' 8 warps; span 2.69 us, warp 2.69 us, nsys graphs 1.34 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 2.08 | 77.2% | 2.03 | 75.7% |
| load | 0.43 | 16.1% | 0.44 | 16.2% |
| wait | 0.14 | 5.1% | 0.05 | 1.8% |
| compute | 0.11 | 4.2% | 0.11 | 4.1% |
| unattributed | 0.00 | 0.0% | 0.06 | 2.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*4 (est.: sampled iterations x4) | 1.54 | 57.1% | 1.0 | 1.54 | `for (int j_2 = 0; j_2 < 2; ++j_2)` |
| store.global*4 (est.: sampled iterations x4) | 0.34 | 12.5% | 1.0 | 0.51 | `for (int j_1 = 0; j_1 < 4; ++j_1)` |
| load.global*4 (est.: sampled iterations x4) | 0.24 | 8.9% | 1.0 | 0.26 | `if (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 4)) < num_tokens)` |
| load.global#3*4 (est.: sampled iterations x4) | 0.16 | 6.0% | 1.0 | 0.26 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 128) + ((((int)threadIdx.x) & 15` |
| wait.sync | 0.14 | 5.1% | 1.0 | 0.29 | `__syncthreads();` |
| store.global#5*4 (est.: sampled iterations x4) | 0.13 | 4.8% | 0.6 | 0.26 | `*(uint4*)(out + ((((((int64_t)vi_1) * (int64_t)128) + ((((int64_t)((int)threadIdx.x)) & (i` |
| compute.reduce | 0.11 | 4.2% | 1.0 | 0.13 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| store.global#2 | 0.04 | 1.6% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| load.global#2 | 0.03 | 1.2% | 1.0 | 0.03 | `*(float2*)(row_partial + 0) = *(float2*)(sum_shared + ((((int)threadIdx.x) >> 6) * 2)); \| ` |
| store.global#3 | 0.03 | 1.2% | 0.6 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |

### token_ranks_kernel grid (5, 1, 1) block (256, 1, 1)

16 launches, 40 warps; ranges, warp and last warp from the 5 recording CTAs' 40 warps; span 7.87 us, warp 7.72 us, nsys graphs 8.61 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 6.94 | 89.9% | 7.08 | 89.9% |
| unattributed | 0.38 | 4.9% | 0.39 | 4.9% |
| compute | 0.35 | 4.6% | 0.36 | 4.6% |
| store | 0.04 | 0.6% | 0.04 | 0.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 6.94 | 89.9% | 1.0 | 7.26 | `if ((((int64_t)0 <= token_ids[((int64_t)((int)blockIdx.x))]) && (token_ids[((int64_t)((int` |
| compute.reduce | 0.35 | 4.6% | 1.0 | 0.69 | `total[0] = tl::AllReduce<tl::SumOp, 256, 1, 0, tl::NamedBarrier<256>>::run(total[0], (&(wo` |
| store.global | 0.04 | 0.6% | 1.0 | 0.06 | `output[((int64_t)((int)blockIdx.x))] = ((int64_t)total[0]);` |

### indexer_mxfp4_quantize_kernel grid (5, 1, 1) block (128, 1, 1)

32 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 3.78 us, warp 3.70 us, nsys graphs 2.94 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 2.30 | 62.1% | 2.36 | 62.5% |
| store | 1.14 | 30.9% | 1.14 | 30.2% |
| unattributed | 0.26 | 7.0% | 0.27 | 7.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2*16 (est.: sampled iterations x16) | 1.27 | 34.3% | 1.0 | 1.54 | `float v__2 = __fmul_rn(((float)X[(((((int64_t)((int)blockIdx.x)) * (int64_t)4096) + (((int` |
| store.global*16 (est.: sampled iterations x16) | 1.10 | 29.8% | 1.0 | 1.54 | `Pool[(((((slot & (int64_t)127) * (int64_t)64) + ((((int64_t)((int)threadIdx.x)) & (int64_t` |
| load.global | 1.03 | 27.8% | 1.0 | 1.09 | `for (int i = 0; i < 32; ++i)` |
| store.global#2 | 0.04 | 1.2% | 1.0 | 0.06 | `Pool[(((((slot & (int64_t)127) * (int64_t)4) + ((slot >> (int64_t)7) * ((int64_t)pool_stri` |

### gemm_splitk_reduce_cast_kernel grid (16, 2, 1) block (128, 1, 1)

102 launches, 128 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 0.99 us, warp 0.93 us, nsys graphs 1.09 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.43 | 46.1% | 0.57 | 50.6% |
| load | 0.35 | 37.6% | 0.40 | 35.5% |
| compute | 0.08 | 8.6% | 0.08 | 6.9% |
| store | 0.07 | 7.7% | 0.08 | 6.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.35 | 37.6% | 1.0 | 0.38 | `if ((1 <= M))` |
| compute.reduce | 0.08 | 8.6% | 1.0 | 0.10 | `amax[0] = tl::AllReduce<tl::MaxOp, 8, 1, 0, tl::NamedBarrier<128>>::run(amax[0]);` |
| store.global | 0.07 | 7.7% | 1.0 | 0.10 | `if (((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 5)) < M)` |

### indexer_mxfp4_quantize_kernel grid (1, 1, 1) block (128, 1, 1)

34 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 3.01 us, warp 1.00 us, nsys graphs 1.97 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.37 | 37.3% | 0.36 | 11.6% |
| load | 0.36 | 36.1% | 1.56 | 50.3% |
| store | 0.26 | 26.5% | 1.18 | 38.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2*16 (est.: sampled iterations x16) | 0.26 | 25.7% | 0.2 | 1.02 | `float v__2 = __fmul_rn(((float)X[(((((int64_t)((int)blockIdx.x)) * (int64_t)4096) + (((int` |
| store.global*16 (est.: sampled iterations x16) | 0.26 | 25.7% | 0.2 | 1.02 | `Pool[(((((slot & (int64_t)127) * (int64_t)64) + ((((int64_t)((int)threadIdx.x)) & (int64_t` |
| load.global | 0.10 | 10.4% | 0.2 | 0.42 | `for (int i = 0; i < 32; ++i)` |
| store.global#2 | 0.01 | 0.8% | 0.2 | 0.03 | `Pool[(((((slot & (int64_t)127) * (int64_t)4) + ((slot >> (int64_t)7) * ((int64_t)pool_stri` |

### prepare_dflash_inputs_kernel grid (9, 1, 1) block (256, 1, 1)

51 launches, 72 warps; ranges, warp and last warp from the 5 recording CTAs' 40 warps; span 2.02 us, warp 0.74 us, nsys graphs 2.30 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.26 | 34.7% | 1.40 | 68.7% |
| unattributed | 0.25 | 33.3% | 0.37 | 18.1% |
| store | 0.24 | 31.9% | 0.27 | 13.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.24 | 33.1% | 0.2 | 1.34 | `req = idx_mapping[((int)blockIdx.y)];` |
| store.global | 0.19 | 25.3% | 0.8 | 0.26 | `if (((int)blockIdx.y) == (num_reqs - 1))` |
| store.global#2 | 0.05 | 6.6% | 0.2 | 0.29 | `if (((int)blockIdx.y) == (num_reqs - 1))` |
| load.global#2 | 0.01 | 1.6% | 0.2 | 0.06 | `vllm_store_if_b32((&(out_query_start_loc[((int)blockIdx.y)])), (((int)blockIdx.y) * num_qu` |

### activation_swiglu_clamp_kernel grid (1, 5, 1) block (128, 1, 1)

96 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 0.93 us, warp 0.73 us, nsys graphs 1.86 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.63 | 85.7% | 0.90 | 86.3% |
| load | 0.07 | 9.9% | 0.10 | 9.3% |
| store | 0.03 | 4.4% | 0.05 | 4.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.07 | 9.9% | 0.8 | 0.10 | `*(uint4*)(gate + 0) = *(uint4*)(X + ((((int64_t)((int)blockIdx.y)) * (int64_t)1152) + (((i` |
| store.global | 0.03 | 4.4% | 0.8 | 0.06 | `*(uint4*)(Y + ((((int64_t)((int)blockIdx.y)) * (int64_t)576) + (((int64_t)((int)threadIdx.` |

### sparse_mla_kernel grid (5, 1, 6) block (128, 1, 1)

4 launches, 120 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 22.24 us, warp 20.70 us, nsys graphs 13.97 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed, with loops' unsampled iterations | 12.65 | 61.1% | 10.59 | 44.8% |
| wait | 3.67 | 17.7% | 0.66 | 2.8% |
| load | 2.97 | 14.3% | 10.86 | 46.0% |
| compute | 0.96 | 4.6% | 0.97 | 4.1% |
| store | 0.46 | 2.2% | 0.54 | 2.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync (sampled iterations) | 3.60 | 17.4% | 11.0 | 5.54 | `__syncthreads();` |
| load.global (sampled iterations) | 2.64 | 12.7% | 2.0 | 8.96 | `for (int c = 0; c < 4; ++c) \| for (int i = 0; i < 8; ++i) \| for (int i_1 = 0; i_1 < 8; ++i` |
| compute.mma (sampled iterations) | 0.74 | 3.6% | 2.0 | 0.77 | `for (int ki = 0; ki < 32; ++ki) \| for (int ki_1 = 0; ki_1 < 2; ++ki_1)` |
| store.global | 0.46 | 2.2% | 1.0 | 0.54 | `for (int i_10 = 0; i_10 < 32; ++i_10) \| for (int i_11 = 0; i_11 < 32; ++i_11) \| for (int i` |
| load.cp_async (sampled iterations) | 0.33 | 1.6% | 2.0 | 0.40 | `for (int k_1 = 0; k_1 < 5; ++k_1) \| for (int k_1 = 0; k_1 < 9; ++k_1) \| for (int k_3 = 0; ` |
| compute.reduce (sampled iterations) | 0.22 | 1.1% | 2.0 | 0.24 | `m_g[i_5] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(m_g[i_5], (&((` |
| wait.cp_async (sampled iterations) | 0.07 | 0.3% | 1.0 | 0.10 | `tl::cp_async_wait<0>();` |

### gemm_bf16_partials_kernel grid (1, 40, 1) block (256, 1, 1)

34 launches, 320 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 2.62 us, warp 1.69 us, nsys graphs 2.82 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.23 | 72.8% | 2.06 | 78.3% |
| wait | 0.23 | 13.6% | 0.24 | 9.3% |
| compute | 0.11 | 6.6% | 0.23 | 8.9% |
| load | 0.07 | 4.1% | 0.00 | 0.0% |
| store | 0.05 | 2.8% | 0.09 | 3.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*2 (est.: sampled iterations x2) | 0.11 | 6.6% | 0.5 | 0.26 | `{` |
| wait.sync | 0.10 | 6.2% | 1.0 | 0.13 | `__syncthreads();` |
| load.tma*2 (est.: sampled iterations x2) | 0.07 | 4.1% | 0.5 | 0.19 | `if (tl::tl_shuffle_elect<128>())` |
| wait.mbarrier*2 (est.: sampled iterations x2) | 0.06 | 3.8% | 0.5 | 0.13 | `mbarrier[(kb + 3)].wait(1);` |
| wait.mbarrier#2*2 (est.: sampled iterations x2) | 0.06 | 3.6% | 0.5 | 0.13 | `mbarrier[kb_1].wait(0);` |
| store.global | 0.05 | 2.8% | 0.5 | 0.13 | `for (int i_1 = 0; i_1 < 4; ++i_1) \| for (int i_2 = 0; i_2 < 8; ++i_2) \| for (int i_4 = 0; ` |

### gemm_bf16_partials_kernel grid (2, 4, 1) block (256, 1, 1)

34 launches, 64 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 1.98 us, warp 1.37 us, nsys graphs 2.11 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.91 | 66.2% | 1.59 | 69.1% |
| wait | 0.23 | 16.8% | 0.39 | 16.9% |
| compute | 0.11 | 8.3% | 0.23 | 10.0% |
| load | 0.07 | 5.1% | 0.00 | 0.0% |
| store | 0.05 | 3.6% | 0.09 | 3.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*2 (est.: sampled iterations x2) | 0.11 | 8.3% | 0.5 | 0.26 | `{` |
| wait.sync | 0.11 | 7.9% | 1.0 | 0.13 | `__syncthreads();` |
| load.tma*2 (est.: sampled iterations x2) | 0.07 | 5.1% | 0.5 | 0.19 | `if (tl::tl_shuffle_elect<128>())` |
| wait.mbarrier*2 (est.: sampled iterations x2) | 0.06 | 4.5% | 0.5 | 0.13 | `mbarrier[(kb + 3)].wait(1);` |
| wait.mbarrier#2*2 (est.: sampled iterations x2) | 0.06 | 4.4% | 0.5 | 0.13 | `mbarrier[kb_1].wait(0);` |
| store.global | 0.05 | 3.6% | 0.5 | 0.13 | `for (int i_1 = 0; i_1 < 4; ++i_1) \| for (int i_2 = 0; i_2 < 8; ++i_2) \| for (int i_4 = 0; ` |

### sink_candidates_kernel grid (1, 127, 1) block (128, 1, 1)

17 launches, 508 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 4.32 us, warp 3.51 us, nsys graphs 3.26 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.37 | 39.1% | 1.95 | 44.2% |
| compute | 0.90 | 25.7% | 0.90 | 20.3% |
| load | 0.53 | 15.2% | 0.88 | 20.0% |
| wait | 0.46 | 13.1% | 0.38 | 8.5% |
| store | 0.24 | 6.9% | 0.31 | 7.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.reduce*4 (est.: sampled iterations x4) | 0.65 | 18.6% | 1.0 | 0.77 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| load.global | 0.53 | 15.2% | 1.0 | 0.64 | `local[0] = max((num_sampled[((int64_t)((int)blockIdx.x))] - 1), 0);` |
| wait.sync#2*4 (est.: sampled iterations x4) | 0.26 | 7.3% | 1.0 | 0.38 | `__syncthreads();` |
| compute.reduce#2*4 (est.: sampled iterations x4) | 0.25 | 7.1% | 1.0 | 0.38 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 2, 32);` |
| store.global*4 (est.: sampled iterations x4) | 0.24 | 6.9% | 1.0 | 0.38 | `if (((int)threadIdx.x) == 0)` |
| wait.sync*4 (est.: sampled iterations x4) | 0.20 | 5.8% | 1.0 | 0.26 | `__syncthreads();` |

### sink_merge_kernel grid (1, 1, 1) block (128, 1, 1)

17 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 4.29 us, warp 4.29 us, nsys graphs 4.26 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.90 | 44.4% | 2.03 | 47.8% |
| unattributed | 0.97 | 22.6% | 1.09 | 25.6% |
| wait | 0.82 | 19.0% | 0.46 | 10.7% |
| compute | 0.58 | 13.4% | 0.56 | 13.1% |
| store | 0.02 | 0.6% | 0.12 | 2.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2*3 (est.: sampled iterations x3) | 1.63 | 38.1% | 1.0 | 1.63 | `for (int step = 0; step < ((width + 127) >> 7); ++step) \| for (int step = 0; step < ((widt` |
| compute.reduce*3 (est.: sampled iterations x3) | 0.58 | 13.4% | 1.0 | 0.58 | `float other = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| wait.sync#3*3 (est.: sampled iterations x3) | 0.53 | 12.3% | 1.0 | 0.67 | `__syncthreads();` |
| load.global | 0.22 | 5.2% | 1.0 | 0.22 | `last = sampled[(max((((int64_t)num_sampled[((int64_t)((int)blockIdx.x))]) - (int64_t)1), (` |
| wait.sync*3 (est.: sampled iterations x3) | 0.17 | 3.9% | 1.0 | 0.19 | `__syncthreads();` |
| wait.sync#2*3 (est.: sampled iterations x3) | 0.12 | 2.8% | 1.0 | 0.19 | `__syncthreads();` |
| load.global#3*3 (est.: sampled iterations x3) | 0.05 | 1.1% | 0.2 | 0.19 | `if ((best[0] == -CUDART_INF_F))` |
| store.global*3 (est.: sampled iterations x3) | 0.02 | 0.6% | 0.2 | 0.10 | `alternates[((((int64_t)((int)blockIdx.x)) * (int64_t)3) + ((int64_t)rank))] = condval_31;` |

### moe_combine_kernel grid (5, 10, 1) block (128, 1, 1)

51 launches, 200 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 1.41 us, warp 1.12 us, nsys graphs 1.54 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.65 | 58.0% | 0.92 | 65.7% |
| unattributed | 0.42 | 37.2% | 0.43 | 30.4% |
| store | 0.05 | 4.8% | 0.05 | 3.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.65 | 58.0% | 1.0 | 0.93 | `for (int k = 0; k < 3; ++k) \| for (int k = 0; k < 6; ++k) \| if ((((int)blockIdx.y) < share` |
| store.global | 0.05 | 4.8% | 1.0 | 0.06 | `{` |

### rope_kernel grid (10, 1, 1) block (128, 1, 1)

51 launches, 40 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 1.28 us, warp 1.23 us, nsys graphs 1.54 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.72 | 58.1% | 0.81 | 60.6% |
| load | 0.48 | 38.9% | 0.49 | 36.4% |
| store | 0.04 | 3.0% | 0.04 | 3.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.48 | 38.9% | 1.0 | 0.51 | `if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < tokens)) \| int64_t position ` |
| store.global | 0.04 | 3.0% | 1.0 | 0.06 | `*(uint4*)(x + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + ((((int64_t)((int)thread` |

### indexer_score_kernel grid (128, 5, 1) block (128, 1, 1)

16 launches, 2560 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 4.10 us, warp 1.44 us, nsys graphs 3.81 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.78 | 54.4% | 2.57 | 62.4% |
| unattributed, with loops' unsampled iterations | 0.53 | 36.4% | 1.21 | 29.2% |
| wait | 0.09 | 6.1% | 0.18 | 4.3% |
| compute | 0.03 | 2.2% | 0.12 | 3.0% |
| store | 0.01 | 0.8% | 0.04 | 1.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 (sampled iterations) | 0.37 | 25.9% | 0.2 | 1.58 | `for (int i_1 = 0; i_1 < 32; ++i_1)` |
| load.global | 0.35 | 24.5% | 1.0 | 0.74 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 4096); \| i` |
| load.global#3 (sampled iterations) | 0.06 | 4.0% | 0.2 | 0.38 | `if (((int)threadIdx.x) < 64)` |
| wait.sync#2 (sampled iterations) | 0.05 | 3.5% | 0.2 | 0.37 | `__syncthreads();` |
| compute.mma (sampled iterations) | 0.03 | 2.2% | 0.2 | 0.13 | `for (int ki = 0; ki < 2; ++ki)` |
| wait.sync (sampled iterations) | 0.01 | 1.0% | 0.2 | 0.06 | `__syncthreads();` |
| store.global (sampled iterations) | 0.01 | 0.8% | 0.2 | 0.06 | `if ((((int)threadIdx.x) >> 6) == 0)` |
| wait.sync#3 (sampled iterations) | 0.01 | 0.8% | 0.2 | 0.06 | `__syncthreads();` |
| wait.sync#4 (sampled iterations) | 0.01 | 0.8% | 0.2 | 0.06 | `__syncthreads();` |

### indexer_score_kernel grid (64, 5, 1) block (128, 1, 1)

16 launches, 1280 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 4.10 us, warp 1.83 us, nsys graphs 0.00 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.96 | 52.5% | 2.01 | 49.4% |
| unattributed, with loops' unsampled iterations | 0.66 | 36.1% | 1.35 | 33.2% |
| wait | 0.16 | 8.7% | 0.52 | 12.7% |
| compute | 0.04 | 2.1% | 0.15 | 3.6% |
| store | 0.01 | 0.7% | 0.05 | 1.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.63 | 34.5% | 1.0 | 0.90 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 4096); \| i` |
| load.global#2 (sampled iterations) | 0.30 | 16.2% | 0.2 | 1.33 | `for (int i_1 = 0; i_1 < 32; ++i_1)` |
| wait.sync#3 (sampled iterations) | 0.06 | 3.3% | 0.2 | 0.26 | `__syncthreads();` |
| wait.sync#2 (sampled iterations) | 0.06 | 3.2% | 0.2 | 0.30 | `__syncthreads();` |
| compute.mma (sampled iterations) | 0.04 | 2.1% | 0.2 | 0.16 | `for (int ki = 0; ki < 2; ++ki)` |
| load.global#3 (sampled iterations) | 0.03 | 1.9% | 0.2 | 0.32 | `if (((int)threadIdx.x) < 64)` |
| wait.sync (sampled iterations) | 0.02 | 1.3% | 0.2 | 0.13 | `__syncthreads();` |
| wait.sync#4 (sampled iterations) | 0.02 | 1.0% | 0.2 | 0.10 | `__syncthreads();` |
| store.global (sampled iterations) | 0.01 | 0.7% | 0.2 | 0.06 | `if ((((int)threadIdx.x) >> 6) == 0)` |

### finish_argmax_kernel grid (2, 1, 1) block (32, 1, 1)

85 launches, 2 warps; ranges, warp and last warp from the 2 recording CTAs' 2 warps; span 0.74 us, warp 0.74 us, nsys graphs 0.90 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.38 | 52.2% | 0.39 | 53.1% |
| compute | 0.13 | 17.4% | 0.12 | 16.2% |
| load | 0.13 | 17.4% | 0.13 | 17.2% |
| store | 0.10 | 13.0% | 0.10 | 13.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.reduce | 0.13 | 17.4% | 1.0 | 0.13 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| load.global | 0.13 | 17.4% | 1.0 | 0.13 | `for (int k = 0; k < 2; ++k) \| if ((((int)threadIdx.x) < parts))` |
| store.global | 0.10 | 13.0% | 1.0 | 0.10 | `if (((int)threadIdx.x) == 0)` |

### adaptive_offsets_kernel grid (1, 1, 1) block (256, 1, 1)

16 launches, 8 warps; ranges, warp and last warp from the 1 recording CTAs' 8 warps; span 3.78 us, warp 3.75 us, nsys graphs 1.95 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 1.79 | 47.7% | 1.14 | 30.1% |
| unattributed, with loops' unsampled iterations | 1.43 | 38.1% | 1.50 | 39.7% |
| store | 0.33 | 8.8% | 0.52 | 13.7% |
| load | 0.20 | 5.4% | 0.63 | 16.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync (sampled iterations) | 0.42 | 11.2% | 1.0 | 0.48 | `__syncthreads();` |
| wait.sync#18 (sampled iterations) | 0.34 | 9.2% | 1.0 | 0.38 | `__syncthreads();` |
| store.global#5 | 0.13 | 3.5% | 1.0 | 0.16 | `for (int chunk_1 = 0; chunk_1 < ((starts_len + 255) >> 8); ++chunk_1)` |
| load.global#2 (sampled iterations) | 0.13 | 3.4% | 1.0 | 0.48 | `if ((((chunk * 256) + ((int)threadIdx.x)) < num_reqs))` |
| store.global#2 (sampled iterations) | 0.09 | 2.5% | 1.0 | 0.10 | `if ((((chunk * 256) + ((int)threadIdx.x)) < num_reqs) & (drafted == 0))` |
| store.global | 0.08 | 2.2% | 1.0 | 0.10 | `if (((int)threadIdx.x) == 0)` |
| load.global (sampled iterations) | 0.07 | 1.8% | 1.0 | 0.10 | `if (((((chunk * 256) + ((int)threadIdx.x)) < num_reqs) & (drafted != 0)))` |
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
| store.global#4 (sampled iterations) | 0.01 | 0.3% | 0.1 | 0.10 | `for (int j = 0; j < 6; ++j)` |
| load.global#3 (sampled iterations) | 0.01 | 0.2% | 0.1 | 0.06 | `state = idx_mapping[((((int64_t)chunk) * (int64_t)256) + ((int64_t)((int)threadIdx.x)))];` |
| store.global#3 (sampled iterations) | 0.01 | 0.2% | 0.1 | 0.06 | `cu_num_logits[(((((int64_t)chunk) * (int64_t)256) + ((int64_t)((int)threadIdx.x))) + (int6` |

### per_token_cast_kernel_kernel grid (10, 1, 1) block (64, 1, 1)

51 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 10 warps; span 1.18 us, warp 1.07 us, nsys graphs 1.09 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.43 | 40.2% | 0.42 | 35.8% |
| load | 0.31 | 29.4% | 0.37 | 31.0% |
| store | 0.23 | 21.3% | 0.31 | 25.9% |
| compute | 0.10 | 9.0% | 0.09 | 7.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.22 | 21.0% | 1.0 | 0.29 | `for (int i = 0; i < 2; ++i)` |
| store.global#2 | 0.11 | 10.5% | 1.0 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.10 | 9.0% | 1.0 | 0.13 | `amax_fragment[i_1] = tl::AllReduce<tl::MaxOp, 2, 1, 0, tl::NamedBarrier<64>>::run(amax_fra` |
| load.global#2*2 (est.: sampled iterations x2) | 0.09 | 8.4% | 1.0 | 0.13 | `amax_fragment[i_1] = cutlass::bfloat16_t(__hmax((bfloat16_t(((nv_bfloat162*)(&(amax_fragme` |
| store.global#3*2 (est.: sampled iterations x2) | 0.06 | 5.4% | 0.6 | 0.13 | `*(fp8_e4_16_t*)(out + ((((((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int64_t)10240) +` |
| store.global*2 (est.: sampled iterations x2) | 0.06 | 5.4% | 0.6 | 0.13 | `out_sf[((((((int64_t)((int)blockIdx.x)) % (int64_t)15) * (int64_t)32) + (((int64_t)((int)t` |

### resample_kernel grid (1, 127, 1) block (128, 1, 1)

17 launches, 508 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 2.94 us, warp 2.21 us, nsys graphs 2.24 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.88 | 85.4% | 3.05 | 90.1% |
| unattributed | 0.28 | 12.6% | 0.30 | 8.8% |
| compute | 0.04 | 2.0% | 0.04 | 1.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.88 | 85.4% | 1.0 | 2.59 | `end = cu_num_logits[(((int64_t)((int)blockIdx.x)) + (int64_t)1)];` |
| compute.reduce | 0.04 | 2.0% | 1.0 | 0.06 | `if ((temp != 0x0p+0f/*0.000000e+00*/) \| bonus)` |

### per_token_cast_kernel_kernel grid (20, 1, 1) block (64, 1, 1)

51 launches, 40 warps; ranges, warp and last warp from the 4 recording CTAs' 8 warps; span 1.09 us, warp 1.04 us, nsys graphs 0.99 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.35 | 33.8% | 0.37 | 34.2% |
| store | 0.30 | 29.2% | 0.30 | 27.7% |
| unattributed | 0.28 | 26.9% | 0.31 | 29.1% |
| compute | 0.10 | 10.0% | 0.10 | 9.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.26 | 25.4% | 1.0 | 0.29 | `for (int i = 0; i < 2; ++i)` |
| store.global#2 | 0.11 | 10.8% | 1.0 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.10 | 10.0% | 1.0 | 0.13 | `amax_fragment[i_1] = tl::AllReduce<tl::MaxOp, 2, 1, 0, tl::NamedBarrier<64>>::run(amax_fra` |
| store.global#3*2 (est.: sampled iterations x2) | 0.10 | 9.2% | 1.0 | 0.13 | `*(fp8_e4_16_t*)(out + ((((((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int64_t)10240) +` |
| store.global*2 (est.: sampled iterations x2) | 0.10 | 9.2% | 1.0 | 0.13 | `out_sf[((((((int64_t)((int)blockIdx.x)) % (int64_t)15) * (int64_t)32) + (((int64_t)((int)t` |
| load.global#2*2 (est.: sampled iterations x2) | 0.09 | 8.5% | 1.0 | 0.13 | `amax_fragment[i_1] = cutlass::bfloat16_t(__hmax((bfloat16_t(((nv_bfloat162*)(&(amax_fragme` |

### gemm_splitk_reduce_cast_kernel grid (16, 3, 1) block (128, 1, 1)

51 launches, 192 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 1.02 us, warp 0.98 us, nsys graphs 1.15 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.43 | 44.1% | 0.48 | 46.1% |
| load | 0.40 | 40.4% | 0.41 | 39.2% |
| compute | 0.08 | 8.0% | 0.08 | 7.4% |
| store | 0.07 | 7.6% | 0.08 | 7.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.40 | 40.4% | 1.0 | 0.42 | `if ((1 <= M))` |
| compute.reduce | 0.08 | 8.0% | 1.0 | 0.10 | `amax[0] = tl::AllReduce<tl::MaxOp, 8, 1, 0, tl::NamedBarrier<128>>::run(amax[0]);` |
| store.global | 0.07 | 7.6% | 1.0 | 0.10 | `if (((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 5)) < M)` |

### activation_swiglu_clamp_kernel grid (1, 10, 1) block (128, 1, 1)

51 launches, 40 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 0.93 us, warp 0.72 us, nsys graphs 1.82 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.62 | 85.7% | 0.88 | 83.4% |
| load | 0.07 | 9.7% | 0.12 | 11.4% |
| store | 0.03 | 4.6% | 0.05 | 5.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.07 | 9.7% | 0.8 | 0.10 | `*(uint4*)(gate + 0) = *(uint4*)(X + ((((int64_t)((int)blockIdx.y)) * (int64_t)1152) + (((i` |
| store.global | 0.03 | 4.6% | 0.8 | 0.06 | `*(uint4*)(Y + ((((int64_t)((int)blockIdx.y)) * (int64_t)576) + (((int64_t)((int)threadIdx.` |

### gemm_splitk_reduce_kernel grid (1, 5, 1) block (128, 1, 1)

64 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 0.83 us, warp 0.83 us, nsys graphs 0.99 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.40 | 47.9% | 0.43 | 53.7% |
| load | 0.38 | 46.3% | 0.33 | 40.3% |
| store | 0.05 | 5.8% | 0.05 | 6.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.38 | 46.3% | 1.0 | 0.40 | `acc[0] = P[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x))` |
| store.global | 0.05 | 5.8% | 1.0 | 0.06 | `C[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x)) * (int64` |

### cache_writer_kernel grid (8, 1, 1) block (128, 1, 1)

51 launches, 32 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 1.02 us, warp 0.93 us, nsys graphs 1.12 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.32 | 34.3% | 0.33 | 32.3% |
| load | 0.28 | 30.6% | 0.30 | 29.3% |
| store | 0.25 | 26.9% | 0.31 | 30.7% |
| compute | 0.08 | 8.2% | 0.08 | 7.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.25 | 26.9% | 1.0 | 0.32 | `if (((int64_t)0 <= slot) & ((slot >> (int64_t)7) < ((int64_t)pages))) \| if (((int64_t)0 <=` |
| load.global | 0.23 | 24.6% | 1.0 | 0.26 | `for (int i = 0; i < 4; ++i)` |
| compute.reduce | 0.08 | 8.2% | 1.0 | 0.10 | `amax = max(amax, __shfl_xor_sync((uint)4294967295, amax, 1, 32));` |
| load.global#2 | 0.06 | 6.0% | 1.0 | 0.06 | `int64_t slot = slots[((int64_t)((int)blockIdx.x))];` |

### cache_writer_kernel grid (6, 1, 1) block (128, 1, 1)

8 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 5.76 us, warp 5.60 us, nsys graphs 3.49 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 3.20 | 57.2% | 4.06 | 64.9% |
| store | 2.03 | 36.3% | 1.71 | 27.3% |
| load | 0.29 | 5.3% | 0.42 | 6.7% |
| compute | 0.07 | 1.2% | 0.07 | 1.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 2.03 | 36.3% | 1.0 | 2.10 | `if (((int64_t)0 <= slot) & ((slot >> (int64_t)7) < ((int64_t)pages))) \| if (((int64_t)0 <=` |
| load.global | 0.24 | 4.3% | 1.0 | 0.32 | `for (int i = 0; i < 4; ++i)` |
| compute.reduce | 0.07 | 1.2% | 1.0 | 0.10 | `amax = max(amax, __shfl_xor_sync((uint)4294967295, amax, 1, 32));` |
| load.global#2 | 0.05 | 1.0% | 1.0 | 0.06 | `int64_t slot = slots[((int64_t)((int)blockIdx.x))];` |

### verify_kernel grid (1, 1, 1) block (96, 1, 1)

17 launches, 3 warps; ranges, warp and last warp from the 1 recording CTAs' 3 warps; span 2.82 us, warp 2.69 us, nsys graphs 2.34 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.08 | 40.1% | 1.08 | 38.0% |
| unattributed | 0.99 | 36.9% | 0.95 | 33.4% |
| compute | 0.29 | 10.7% | 0.29 | 10.2% |
| wait | 0.17 | 6.3% | 0.23 | 8.1% |
| store | 0.16 | 6.0% | 0.30 | 10.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.01 | 37.7% | 1.0 | 1.02 | `start = cu_num_logits[((int64_t)((int)blockIdx.x))];` |
| compute.reduce | 0.29 | 10.7% | 1.0 | 0.32 | `row_max[0] = vllm_max(row_max[0], __shfl_xor_sync((uint)4294967295, row_max[0], 8, 32));` |
| store.global | 0.16 | 6.0% | 1.0 | 0.29 | `if (((int)threadIdx.x) == 0)` |
| wait.sync#3 | 0.07 | 2.8% | 1.0 | 0.13 | `__syncthreads();` |
| load.global#2 | 0.06 | 2.4% | 1.0 | 0.06 | `target = local_argmax[((((int64_t)row) * argmax_stride) + ((int64_t)best_index[0]))];` |
| wait.sync#2 | 0.06 | 2.4% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync | 0.03 | 1.2% | 1.0 | 0.03 | `__syncthreads();` |

### compute_slot_mappings_kernel grid (1, 11, 1) block (256, 1, 1)

17 launches, 88 warps; ranges, warp and last warp from the 6 recording CTAs' 48 warps; span 2.72 us, warp 1.46 us, nsys graphs 2.72 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.82 | 56.1% | 0.82 | 30.4% |
| unattributed | 0.64 | 43.9% | 1.87 | 69.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.82 | 56.1% | 1.0 | 0.83 | `int first_token = query_start_loc[0];` |

### sink_cut_kernel grid (1, 1, 1) block (128, 1, 1)

34 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 1.70 us, warp 0.60 us, nsys graphs 2.11 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.33 | 55.0% | 0.31 | 23.5% |
| store | 0.15 | 24.8% | 0.50 | 38.0% |
| load | 0.12 | 20.1% | 0.51 | 38.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.15 | 24.8% | 0.6 | 0.48 | `if (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) < num_cut) \| saved[((((int64_t)((int)` |
| load.global | 0.12 | 20.1% | 0.1 | 0.48 | `for (int b = 0; b < num_reqs; ++b)` |

### sink_policy_kernel grid (1, 1, 1) block (32, 1, 1)

17 launches, 1 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 2.50 us, warp 2.50 us, nsys graphs 2.74 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.09 | 43.6% | 1.12 | 43.6% |
| unattributed | 0.83 | 33.3% | 0.87 | 33.8% |
| store | 0.58 | 23.1% | 0.58 | 22.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 | 0.64 | 25.6% | 1.0 | 0.64 | `state[0] = idx_mapping[((int64_t)((int)blockIdx.x))];` |
| store.global#3 | 0.51 | 20.5% | 1.0 | 0.51 | `for (int a = 0; a < 3; ++a) \| recovered_logprobs[((int64_t)((int)blockIdx.x))] = 0x0p+0f/*` |
| load.global | 0.22 | 9.0% | 1.0 | 0.22 | `start[0] = cu_num_logits[((int64_t)((int)blockIdx.x))];` |
| load.global#4 | 0.16 | 6.4% | 1.0 | 0.16 | `for (int r = 0; r < 2; ++r) \| for (int r_2 = 0; r_2 < 8; ++r_2)` |
| load.global#5 | 0.06 | 2.6% | 1.0 | 0.06 | `AtomicAdd((&(counts[0])), draft_zero_flag[0]);` |
| store.global#4 | 0.06 | 2.6% | 1.0 | 0.06 | `events[(((int64_t)((int)blockIdx.x)) * (int64_t)6)] = outcome[0];` |

### norm_forward_and_per_token_cast_kernel_kernel grid (8, 1, 1) block (128, 1, 1)

17 launches, 32 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 2.34 us, warp 2.29 us, nsys graphs 1.76 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 2.13 | 93.2% | 3.06 | 119.1% |
| load | 0.45 | 19.7% | 0.40 | 15.8% |
| compute | 0.13 | 5.7% | 0.13 | 5.1% |
| wait | 0.05 | 2.4% | 0.06 | 2.4% |
| unattributed | 0.00 | 0.0% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*5 (est.: sampled iterations x5) | 1.86 | 81.2% | 1.0 | 2.08 | `for (int j_1 = 0; j_1 < 2; ++j_1)` |
| load.global#2*5 (est.: sampled iterations x5) | 0.23 | 10.0% | 1.0 | 0.32 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 1024) + (((int)threadIdx.x) * 8)` |
| load.global*5 (est.: sampled iterations x5) | 0.22 | 9.6% | 1.0 | 0.32 | `*(uint4*)(x_local + (vi * 8)) = *(uint4*)(x + ((((((int64_t)vi) * (int64_t)1024) + (((int6` |
| store.global*5 (est.: sampled iterations x5) | 0.22 | 9.6% | 1.0 | 0.32 | `for (int j = 0; j < 4; ++j)` |
| compute.reduce | 0.13 | 5.7% | 1.0 | 0.16 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| wait.sync | 0.05 | 2.4% | 1.0 | 0.06 | `__syncthreads();` |
| store.global#2 | 0.04 | 1.8% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| store.global#3 | 0.01 | 0.5% | 0.2 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |

### metadata_pages_kernel grid (5, 1, 1) block (128, 1, 1)

32 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 1.31 us, warp 0.95 us, nsys graphs 0.00 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.72 | 76.1% | 0.79 | 59.8% |
| load | 0.14 | 15.1% | 0.19 | 14.6% |
| store | 0.08 | 8.8% | 0.34 | 25.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.13 | 13.4% | 1.0 | 0.13 | `if ((((offset + ((int)blockIdx.x)) < src_rows) && (0 <= (offset + ((int)blockIdx.x)))))` |
| store.global | 0.08 | 8.8% | 0.2 | 0.35 | `if (((int)threadIdx.x) < 32)` |
| load.global#2 | 0.02 | 1.7% | 0.2 | 0.06 | `if (((req < table_rows) && (((int)threadIdx.x) < table_cols)))` |

### norm_forward_and_per_token_cast_kernel_kernel grid (10, 1, 1) block (128, 1, 1)

17 launches, 40 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 2.30 us, warp 2.27 us, nsys graphs 1.79 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 2.08 | 91.5% | 2.17 | 93.5% |
| load | 0.44 | 19.4% | 0.38 | 16.2% |
| compute | 0.13 | 5.6% | 0.13 | 5.4% |
| wait | 0.06 | 2.5% | 0.05 | 2.3% |
| unattributed | 0.00 | 0.0% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*5 (est.: sampled iterations x5) | 1.79 | 78.9% | 1.0 | 2.08 | `for (int j_1 = 0; j_1 < 2; ++j_1)` |
| load.global*5 (est.: sampled iterations x5) | 0.23 | 10.2% | 1.0 | 0.32 | `*(uint4*)(x_local + (vi * 8)) = *(uint4*)(x + ((((((int64_t)vi) * (int64_t)1024) + (((int6` |
| store.global*5 (est.: sampled iterations x5) | 0.23 | 10.2% | 1.0 | 0.32 | `for (int j = 0; j < 4; ++j)` |
| load.global#2*5 (est.: sampled iterations x5) | 0.21 | 9.2% | 1.0 | 0.32 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 1024) + (((int)threadIdx.x) * 8)` |
| compute.reduce | 0.13 | 5.6% | 1.0 | 0.16 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| wait.sync | 0.06 | 2.5% | 1.0 | 0.10 | `__syncthreads();` |
| store.global#2 | 0.04 | 1.8% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| store.global#3 | 0.01 | 0.6% | 0.2 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |

### sparse_mla_kernel grid (1, 1, 6) block (128, 1, 1)

2 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 19.65 us, warp 19.62 us, nsys graphs 13.23 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 5.52 | 28.1% | 6.78 | 34.5% |
| unattributed, with loops' unsampled iterations | 4.84 | 24.7% | 5.18 | 26.4% |
| compute | 4.69 | 23.9% | 4.67 | 23.8% |
| wait | 3.87 | 19.8% | 2.32 | 11.8% |
| store | 0.69 | 3.5% | 0.69 | 3.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 | 2.91 | 14.8% | 1.0 | 3.33 | `for (int i = 0; i < 8; ++i)` |
| compute.mma*4 (est.: sampled iterations x4) | 1.73 | 8.8% | 1.0 | 1.79 | `for (int ki = 0; ki < 32; ++ki) \| {` |
| load.global (sampled iterations) | 1.58 | 8.0% | 1.0 | 3.02 | `for (int c = 0; c < 4; ++c) \| for (int i = 0; i < 8; ++i) \| for (int i_1 = 0; i_1 < 8; ++i` |
| wait.sync (sampled iterations) | 1.49 | 7.6% | 1.0 | 1.98 | `__syncthreads();` |
| compute.mma#2*4 (est.: sampled iterations x4) | 1.17 | 6.0% | 1.0 | 1.28 | `for (int ki_1 = 0; ki_1 < 2; ++ki_1) \| if (((((int)blockIdx.x) * 2) + (c_1 >> 1)) < 3)` |
| compute.reduce#2*8 (est.: sampled iterations x8) | 0.92 | 4.7% | 1.0 | 1.02 | `row_sum[i_7] = tl::AllReduce<tl::SumOp, 128, 32, 0, tl::NamedBarrier<128>>::run(row_sum[i_` |
| compute.reduce*8 (est.: sampled iterations x8) | 0.87 | 4.5% | 1.0 | 1.02 | `m_g[i_4] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(m_g[i_4], (&((` |
| store.global | 0.69 | 3.5% | 1.0 | 0.72 | `for (int i_10 = 0; i_10 < 32; ++i_10) \| for (int i_11 = 0; i_11 < 32; ++i_11) \| for (int i` |
| load.cp_async#2*4 (est.: sampled iterations x4) | 0.66 | 3.4% | 1.0 | 0.77 | `if (((((int)blockIdx.x) * 2) + (c_1 >> 1)) < 3) \| if ((c_1 < 3) & (c_1 < 3)) \| if (c < 3)` |
| wait.sync#8*8 (est.: sampled iterations x8) | 0.39 | 2.0% | 1.0 | 0.51 | `__syncthreads();` |
| load.cp_async (sampled iterations) | 0.37 | 1.9% | 1.0 | 0.40 | `for (int k_1 = 0; k_1 < 5; ++k_1) \| for (int k_1 = 0; k_1 < 9; ++k_1) \| for (int k_3 = 0; ` |
| wait.sync#10*8 (est.: sampled iterations x8) | 0.30 | 1.5% | 1.0 | 0.51 | `__syncthreads();` |
| wait.sync#5*4 (est.: sampled iterations x4) | 0.26 | 1.3% | 1.0 | 0.32 | `__syncthreads();` |
| wait.cp_async*4 (est.: sampled iterations x4) | 0.26 | 1.3% | 1.0 | 0.32 | `tl::cp_async_wait<0>();` |
| wait.sync#4*4 (est.: sampled iterations x4) | 0.25 | 1.3% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#7*4 (est.: sampled iterations x4) | 0.22 | 1.1% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#11*4 (est.: sampled iterations x4) | 0.21 | 1.1% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#9*4 (est.: sampled iterations x4) | 0.20 | 1.0% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#6*4 (est.: sampled iterations x4) | 0.19 | 1.0% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#3 | 0.06 | 0.3% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#2 | 0.03 | 0.2% | 1.0 | 0.05 | `__syncthreads();` |

### post_update_kernel grid (1, 1, 1) block (32, 1, 1)

17 launches, 1 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 1.79 us, warp 1.79 us, nsys graphs 2.53 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.67 | 37.5% | 0.67 | 29.2% |
| store | 0.58 | 32.1% | 0.57 | 24.6% |
| unattributed | 0.54 | 30.4% | 1.06 | 46.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 | 0.48 | 26.8% | 1.0 | 0.48 | `int count = num_sampled[((((int64_t)((int)blockIdx.x)) * (int64_t)32) + ((int64_t)((int)th` |
| store.global#2 | 0.42 | 23.2% | 1.0 | 0.42 | `if (0 < count)` |
| store.global | 0.16 | 8.9% | 1.0 | 0.16 | `if (0 < count)` |
| load.global | 0.10 | 5.4% | 1.0 | 0.10 | `int64_t req = idx_mapping[((((int64_t)((int)blockIdx.x)) * (int64_t)32) + ((int64_t)((int)` |
| load.global#3 | 0.10 | 5.4% | 1.0 | 0.10 | `if ((0 < count))` |

### sink_flags_kernel grid (1, 1, 1) block (128, 1, 1)

17 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 2.21 us, warp 0.72 us, nsys graphs 3.12 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.38 | 53.3% | 1.54 | 69.9% |
| unattributed | 0.25 | 34.4% | 0.33 | 14.8% |
| store | 0.09 | 12.2% | 0.34 | 15.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.38 | 53.3% | 0.2 | 1.54 | `token = logits_indices[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)th` |
| store.global | 0.09 | 12.2% | 0.2 | 0.35 | `flags[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = (` |

### assign_draft_budget_kernel grid (1, 1, 1) block (256, 1, 1)

16 launches, 8 warps; ranges, warp and last warp from the 1 recording CTAs' 8 warps; span 1.73 us, warp 1.73 us, nsys graphs 1.86 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 0.78 | 45.1% | 0.47 | 21.7% |
| unattributed, with loops' unsampled iterations | 0.52 | 30.3% | 0.61 | 27.9% |
| load | 0.23 | 13.4% | 0.91 | 41.5% |
| store | 0.19 | 11.1% | 0.19 | 8.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync | 0.52 | 30.1% | 1.0 | 0.61 | `__syncthreads();` |
| load.global | 0.23 | 13.4% | 1.0 | 0.70 | `for (int chunk = 0; chunk < ((num_reqs + 255) >> 8); ++chunk)` |
| wait.sync#3 | 0.20 | 11.3% | 1.0 | 0.22 | `__syncthreads();` |
| store.global | 0.19 | 11.1% | 1.0 | 0.19 | `for (int chunk_2 = 0; chunk_2 < ((num_reqs + 255) >> 8); ++chunk_2)` |
| wait.sync#2 (sampled iterations) | 0.06 | 3.7% | 1.0 | 0.06 | `__syncthreads();` |

### sink_rows_kernel grid (1, 64, 1) block (256, 1, 1)

17 launches, 512 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 1.57 us, warp 0.96 us, nsys graphs 1.74 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.38 | 39.2% | 0.64 | 31.5% |
| load | 0.31 | 32.2% | 0.62 | 30.5% |
| store | 0.27 | 28.6% | 0.78 | 38.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.30 | 31.9% | 1.0 | 0.38 | `row = (max((((int64_t)num_sampled[((int64_t)((int)blockIdx.x))]) - (int64_t)1), (int64_t)0` |
| store.global | 0.26 | 27.2% | 1.0 | 0.45 | `for (int k_1 = 0; k_1 < 8; ++k_1)` |
| store.global#2 | 0.01 | 1.5% | 0.0 | 0.45 | `tokens[((((int64_t)((int)blockIdx.x)) * (int64_t)3) + ((int64_t)((int)threadIdx.x)))] = to` |
| load.global#2 | 0.00 | 0.3% | 0.0 | 0.10 | `token = max(alternates[((((int64_t)((int)blockIdx.x)) * (int64_t)3) + ((int64_t)((int)thre` |

### gather_block_tables_kernel grid (11, 1, 1) block (256, 1, 1)

17 launches, 88 warps; ranges, warp and last warp from the 6 recording CTAs' 48 warps; span 1.95 us, warp 1.35 us, nsys graphs 0.00 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.05 | 78.1% | 1.63 | 84.5% |
| unattributed | 0.30 | 21.9% | 0.30 | 15.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.05 | 78.1% | 1.0 | 1.63 | `int g = chunk_groups[((int64_t)((int)blockIdx.x))];` |

### rope_kernel grid (5, 1, 1) block (256, 1, 1)

32 launches, 40 warps; ranges, warp and last warp from the 5 recording CTAs' 40 warps; span 0.98 us, warp 0.88 us, nsys graphs 1.15 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.59 | 66.4% | 0.71 | 69.7% |
| load | 0.26 | 29.2% | 0.26 | 25.7% |
| store | 0.04 | 4.4% | 0.05 | 4.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.26 | 29.2% | 1.0 | 0.29 | `if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < tokens)) \| int64_t position ` |
| store.global | 0.04 | 4.4% | 1.0 | 0.06 | `*(uint4*)(x + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + ((((int64_t)((int)thread` |

### logits_rows_kernel grid (1, 1, 1) block (128, 1, 1)

17 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 1.15 us, warp 1.13 us, nsys graphs 2.05 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.50 | 44.0% | 0.30 | 15.8% |
| load | 0.49 | 43.3% | 0.50 | 26.2% |
| store | 0.14 | 12.8% | 1.10 | 58.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.49 | 43.3% | 1.0 | 0.51 | `index = logits_indices[min(((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((in` |
| store.global | 0.14 | 12.8% | 1.0 | 0.38 | `if (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) < num_logits)` |

### dspark_confidence_kernel grid (10, 1, 1) block (128, 1, 1)

17 launches, 40 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 1.79 us, warp 1.61 us, nsys graphs 1.82 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.25 | 77.9% | 1.36 | 75.9% |
| load | 0.17 | 10.7% | 0.21 | 11.4% |
| compute | 0.09 | 5.9% | 0.10 | 5.3% |
| wait | 0.07 | 4.1% | 0.05 | 2.7% |
| store | 0.02 | 1.4% | 0.08 | 4.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.16 | 10.0% | 1.0 | 0.16 | `for (int c = 0; c < 5; ++c)` |
| compute.reduce | 0.09 | 5.9% | 1.0 | 0.10 | `acc[0] = tl::warp_reduce_sum(acc[0]);` |
| wait.sync | 0.07 | 4.1% | 1.0 | 0.13 | `__syncthreads();` |
| store.global | 0.02 | 1.4% | 0.2 | 0.10 | `out[((int64_t)((int)blockIdx.x))] = (0x1p+0f/*1.000000e+00*/ / (0x1p+0f/*1.000000e+00*/ + ` |
| load.global#2 | 0.01 | 0.8% | 0.2 | 0.06 | `total[0] = 0x0p+0f/*0.000000e+00*/;` |

### insert_kernel grid (1, 1, 1) block (32, 1, 1)

17 launches, 1 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 1.57 us, warp 1.57 us, nsys graphs 1.41 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.70 | 44.9% | 0.75 | 41.6% |
| unattributed | 0.42 | 26.5% | 0.42 | 23.6% |
| store | 0.42 | 26.5% | 0.58 | 32.5% |
| compute | 0.03 | 2.0% | 0.04 | 2.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#3 | 0.51 | 32.7% | 1.0 | 0.51 | `int state = mapping[((int64_t)position)]; \| int64_t state = mapping[((int64_t)position)];` |
| store.global | 0.42 | 26.5% | 1.0 | 0.42 | `counts[((int64_t)((int)blockIdx.x))] = (count_local + 1); \| counts[((int64_t)((int)blockId` |
| load.global | 0.13 | 8.2% | 1.0 | 0.13 | `for (int k = 0; k < 4; ++k)` |
| load.global#2 | 0.06 | 4.1% | 1.0 | 0.06 | `count_local = counts[((int64_t)((int)blockIdx.x))];` |
| compute.reduce | 0.03 | 2.0% | 1.0 | 0.03 | `int count = __shfl_sync((uint)4294967295, count_local, 0, 32);` |

### gemm_splitk_reduce_kernel grid (4, 8, 1) block (128, 1, 1)

51 launches, 128 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 0.54 us, warp 0.52 us, nsys graphs 0.74 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.25 | 47.9% | 0.26 | 48.1% |
| load | 0.23 | 43.3% | 0.24 | 44.0% |
| store | 0.05 | 8.8% | 0.04 | 7.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.23 | 43.3% | 1.0 | 0.26 | `acc[0] = P[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x))` |
| store.global | 0.05 | 8.8% | 1.0 | 0.06 | `C[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x)) * (int64` |

### context_inputs_kernel grid (5, 8, 1) block (128, 1, 1)

17 launches, 160 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 1.54 us, warp 1.22 us, nsys graphs 1.94 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.81 | 66.5% | 1.23 | 75.5% |
| unattributed | 0.33 | 27.1% | 0.33 | 20.2% |
| store | 0.08 | 6.4% | 0.07 | 4.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.81 | 66.5% | 1.0 | 1.12 | `vllm_dspark_context_b128((&(aux[(((((int64_t)((int)blockIdx.y)) * (int64_t)15360) + (((int` |
| store.global | 0.08 | 6.4% | 1.0 | 0.16 | `if (((((int)blockIdx.x) == 0) & (((int)threadIdx.x) == 0)) & (num_tokens <= ((int)blockIdx` |

### topk_ids_kernel grid (6, 1, 1) block (512, 1, 1)

1 launches, 96 warps; ranges, warp and last warp from the 6 recording CTAs' 96 warps; span 27.90 us, warp 24.91 us, nsys graphs 24.83 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 21.52 | 86.4% | 24.00 | 86.0% |
| unattributed | 1.92 | 7.7% | 1.92 | 6.9% |
| wait | 1.18 | 4.7% | 0.51 | 1.8% |
| store | 0.30 | 1.2% | 1.47 | 5.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 21.52 | 86.4% | 1.0 | 24.32 | `for (int t = 0; t < (vocab >> 14); ++t)` |
| wait.sync | 1.05 | 4.2% | 1.0 | 2.69 | `__syncthreads();` |
| store.global | 0.30 | 1.2% | 1.0 | 4.26 | `if ((((int)threadIdx.x) >> 5) == 0)` |
| wait.sync#2 | 0.07 | 0.3% | 1.0 | 0.16 | `__syncthreads();` |
| wait.sync#3 | 0.05 | 0.2% | 1.0 | 0.10 | `__syncthreads();` |

### per_token_cast_kernel_kernel grid (60, 1, 1) block (64, 1, 1)

17 launches, 120 warps; ranges, warp and last warp from the 4 recording CTAs' 8 warps; span 1.22 us, warp 1.12 us, nsys graphs 1.26 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.40 | 35.5% | 0.44 | 32.0% |
| unattributed | 0.31 | 28.0% | 0.53 | 38.1% |
| store | 0.30 | 27.2% | 0.31 | 22.3% |
| compute | 0.10 | 9.3% | 0.11 | 7.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.31 | 27.6% | 1.0 | 0.35 | `for (int i = 0; i < 2; ++i)` |
| store.global#2 | 0.11 | 10.0% | 1.0 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.10 | 9.3% | 1.0 | 0.13 | `amax_fragment[i_1] = tl::AllReduce<tl::MaxOp, 2, 1, 0, tl::NamedBarrier<64>>::run(amax_fra` |
| store.global*2 (est.: sampled iterations x2) | 0.10 | 9.3% | 1.0 | 0.13 | `out_sf[((((((int64_t)((int)blockIdx.x)) % (int64_t)15) * (int64_t)32) + (((int64_t)((int)t` |
| load.global#2*2 (est.: sampled iterations x2) | 0.09 | 7.9% | 1.0 | 0.13 | `amax_fragment[i_1] = cutlass::bfloat16_t(__hmax((bfloat16_t(((nv_bfloat162*)(&(amax_fragme` |
| store.global#3*2 (est.: sampled iterations x2) | 0.09 | 7.9% | 1.0 | 0.13 | `*(fp8_e4_16_t*)(out + ((((((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int64_t)10240) +` |

### combine_sampled_and_draft_tokens_kernel grid (1, 1, 1) block (32, 1, 1)

17 launches, 1 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 1.38 us, warp 1.38 us, nsys graphs 1.70 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.90 | 65.1% | 0.90 | 64.3% |
| unattributed | 0.38 | 27.9% | 0.40 | 28.9% |
| store | 0.10 | 7.0% | 0.09 | 6.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.90 | 65.1% | 1.0 | 0.90 | `if (((((int)blockIdx.x) * 32) + ((int)threadIdx.x)) < num_reqs) \| req = idx_mapping[((int6` |
| store.global | 0.10 | 7.0% | 1.0 | 0.10 | `if ((prefill < seq) && (1 < num_logits))` |

### mhc_post_kernel grid (50, 1, 1) block (128, 1, 1)

17 launches, 200 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 1.38 us, warp 1.02 us, nsys graphs 1.63 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.84 | 83.1% | 1.13 | 84.3% |
| load | 0.12 | 11.5% | 0.14 | 10.7% |
| store | 0.05 | 5.4% | 0.07 | 5.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.12 | 11.5% | 1.0 | 0.16 | `*(float4*)(pm + 0) = *(float4*)(post + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int` |
| store.global | 0.05 | 5.4% | 1.0 | 0.10 | `for (int n_1 = 0; n_1 < 4; ++n_1)` |

### indexer_topk_segment_kernel grid (1, 6, 1) block (512, 1, 1)

2 launches, 96 warps; ranges, warp and last warp from the 6 recording CTAs' 96 warps; span 10.62 us, warp 10.56 us, nsys graphs 0.00 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed, with loops' unsampled iterations | 5.25 | 49.7% | 5.49 | 51.7% |
| wait | 4.08 | 38.6% | 3.81 | 35.8% |
| load | 1.06 | 10.0% | 1.09 | 10.2% |
| store | 0.17 | 1.6% | 0.24 | 2.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.06 | 10.0% | 1.0 | 1.12 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 4096); \| i` |
| wait.sync (sampled iterations) | 1.01 | 9.6% | 14.5 | 1.18 | `__syncthreads();` |
| wait.sync#24*9 (est.: sampled iterations x9) | 0.37 | 3.5% | 0.5 | 0.43 | `__syncthreads();` |
| wait.sync#23*9 (est.: sampled iterations x9) | 0.33 | 3.1% | 0.5 | 0.43 | `__syncthreads();` |
| wait.sync#20*9 (est.: sampled iterations x9) | 0.32 | 3.1% | 0.5 | 0.43 | `__syncthreads();` |
| wait.sync#14*8 (est.: sampled iterations x8) | 0.32 | 3.0% | 0.5 | 0.38 | `__syncthreads();` |
| wait.sync#6*8 (est.: sampled iterations x8) | 0.31 | 2.9% | 0.5 | 0.38 | `__syncthreads();` |
| wait.sync#19*9 (est.: sampled iterations x9) | 0.31 | 2.9% | 0.5 | 0.43 | `__syncthreads();` |
| wait.sync#5*8 (est.: sampled iterations x8) | 0.30 | 2.8% | 0.5 | 0.38 | `__syncthreads();` |
| wait.sync#13*8 (est.: sampled iterations x8) | 0.28 | 2.7% | 0.5 | 0.38 | `__syncthreads();` |
| store.global | 0.17 | 1.6% | 1.5 | 0.24 | `for (int i_3 = 0; i_3 < 4; ++i_3) \| if (((int)threadIdx.x) < total) \| if (((int)threadIdx.` |
| wait.sync#25 | 0.05 | 0.5% | 0.5 | 0.08 | `__syncthreads();` |
| wait.sync#18 | 0.04 | 0.4% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#17 | 0.04 | 0.4% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#3 | 0.04 | 0.4% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#8 | 0.04 | 0.4% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#16 | 0.04 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#9 | 0.04 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#2 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#11 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#7 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#10 | 0.03 | 0.3% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#15 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#4 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#12 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#21 | 0.03 | 0.3% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#22 | 0.03 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |

### token_counts_kernel grid (1, 1, 1) block (128, 1, 1)

17 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 1.12 us, warp 0.47 us, nsys graphs 1.28 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.27 | 57.6% | 0.34 | 28.1% |
| store | 0.14 | 30.5% | 0.64 | 53.6% |
| load | 0.06 | 11.9% | 0.22 | 18.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.14 | 30.5% | 0.2 | 0.58 | `if (((int)threadIdx.x) == 0) \| sampled[((int64_t)((int)blockIdx.x))] = (((int)(prefill_len` |
| load.global | 0.06 | 11.9% | 0.2 | 0.22 | `int count = sampled[((int64_t)((int)blockIdx.x))];` |

### flatten_sampled_kernel grid (1, 1, 1) block (128, 1, 1)

17 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 1.12 us, warp 0.83 us, nsys graphs 1.31 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.46 | 55.8% | 0.62 | 55.2% |
| unattributed | 0.34 | 41.3% | 0.42 | 37.2% |
| store | 0.02 | 2.9% | 0.09 | 7.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.41 | 49.0% | 1.0 | 0.42 | `for (int r = 0; r < num_reqs; ++r)` |
| load.global#2 | 0.06 | 6.7% | 0.2 | 0.22 | `int i = (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) - cu_num_logits[req]);` |
| store.global | 0.02 | 2.9% | 0.2 | 0.10 | `flat[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = co` |

### metadata_decode_kernel grid (6, 2, 1) block (128, 1, 1)

4 launches, 48 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 4.40 us, warp 3.19 us, nsys graphs 2.18 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 2.02 | 63.5% | 3.07 | 65.6% |
| unattributed | 1.06 | 33.4% | 1.46 | 31.1% |
| store | 0.10 | 3.1% | 0.15 | 3.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.20 | 37.5% | 0.5 | 2.96 | `if ((((offset + ((int)blockIdx.x)) < src_rows) && (0 <= (offset + ((int)blockIdx.x)))))` |
| load.global#3 | 0.81 | 25.3% | 0.5 | 1.65 | `if ((((offset + ((int)blockIdx.x)) < main_src_rows) && (0 <= (offset + ((int)blockIdx.x)))` |
| store.global#4 | 0.06 | 1.8% | 0.5 | 0.26 | `if (0 < live)` |
| store.global | 0.03 | 1.0% | 0.5 | 0.06 | `swa[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = con` |
| load.global#2 | 0.02 | 0.8% | 0.1 | 0.19 | `if (((0 <= req) & ((int64_t)0 <= pos)))` |
| store.global#2 | 0.01 | 0.2% | 0.1 | 0.05 | `lengths[((int64_t)((int)blockIdx.x))] = ((int)max(info[1], (int64_t)0));` |
| store.global#3 | 0.01 | 0.2% | 0.1 | 0.05 | `top_lengths[((int64_t)((int)blockIdx.x))] = condval_6; \| top_lengths[((int64_t)((int)block` |

### prepare_pos_seq_lens_kernel grid (2, 1, 1) block (256, 1, 1)

17 launches, 16 warps; ranges, warp and last warp from the 2 recording CTAs' 16 warps; span 1.09 us, warp 0.75 us, nsys graphs 1.31 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.31 | 41.6% | 0.36 | 32.5% |
| load | 0.30 | 40.3% | 0.61 | 55.5% |
| store | 0.14 | 18.0% | 0.13 | 12.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.30 | 40.3% | 0.5 | 0.61 | `int computed = num_computed_tokens[idx_mapping[((int64_t)((int)blockIdx.x))]];` |
| store.global | 0.07 | 9.5% | 0.5 | 0.19 | `for (int it = 0; it < (((max_reqs + 255) - num_reqs) >> 8); ++it)` |
| store.global#2 | 0.06 | 8.5% | 0.5 | 0.13 | `if (((int)threadIdx.x) == 0)` |

### ced_requests_kernel grid (1, 1, 1) block (128, 1, 1)

17 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 0.86 us, warp 0.60 us, nsys graphs 1.18 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.33 | 54.7% | 0.33 | 31.1% |
| store | 0.17 | 28.0% | 0.63 | 58.7% |
| load | 0.10 | 17.3% | 0.11 | 10.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.17 | 28.0% | 1.0 | 0.42 | `if (((int)threadIdx.x) < 2) \| if (((int)threadIdx.x) <= 2)` |
| load.global | 0.10 | 17.3% | 1.0 | 0.13 | `uint v_ = tl::load_global_32((&(starts[1]))); \| uint v_ = tl::load_global_32((&(starts[min` |

### metadata_chunk_kernel grid (6, 1, 1) block (128, 1, 1)

2 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 9.06 us, warp 6.84 us, nsys graphs 4.37 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 4.15 | 60.7% | 6.22 | 68.7% |
| unattributed | 2.62 | 38.3% | 2.72 | 30.0% |
| store | 0.07 | 1.0% | 0.11 | 1.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 3.82 | 55.9% | 1.0 | 4.91 | `if ((((offset + ((int)blockIdx.x)) < src_rows) && (0 <= (offset + ((int)blockIdx.x)))))` |
| load.global#2 | 0.33 | 4.8% | 0.2 | 1.33 | `if (((0 <= req) & ((int64_t)0 <= pos)))` |
| store.global | 0.05 | 0.7% | 1.0 | 0.06 | `swa[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = con` |
| store.global#2 | 0.01 | 0.2% | 0.2 | 0.05 | `lengths[((int64_t)((int)blockIdx.x))] = ((int)max(info[1], (int64_t)0));` |
| store.global#3 | 0.01 | 0.1% | 0.2 | 0.03 | `top_lengths[((int64_t)((int)blockIdx.x))] = condval_12; \| top_lengths[((int64_t)((int)bloc` |

### topk_log_softmax_kernel grid (6, 1, 1) block (1024, 1, 1)

1 launches, 192 warps; ranges, warp and last warp from the 6 recording CTAs' 192 warps; span 17.63 us, warp 16.61 us, nsys graphs 16.22 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 13.66 | 82.2% | 14.88 | 84.4% |
| unattributed, with loops' unsampled iterations | 1.59 | 9.6% | 1.76 | 10.0% |
| wait | 0.97 | 5.8% | 0.58 | 3.3% |
| compute | 0.39 | 2.3% | 0.35 | 2.0% |
| store | 0.00 | 0.0% | 0.06 | 0.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 7.10 | 42.8% | 1.0 | 7.74 | `if (0 < (vocab & 2047)) \| if (0 < (vocab & 8191))` |
| load.global#2 | 6.54 | 39.4% | 1.0 | 7.04 | `for (int step_1 = 0; step_1 < (vocab >> 11); ++step_1) \| for (int step_1 = 0; step_1 < (vo` |
| wait.sync#2 | 0.43 | 2.6% | 1.0 | 0.99 | `__syncthreads();` |
| wait.sync | 0.29 | 1.8% | 1.0 | 0.80 | `__syncthreads();` |
| compute.reduce#3 | 0.17 | 1.0% | 1.0 | 0.22 | `ties[0] = vllm_add(ties[0], __shfl_xor_sync((uint)4294967295, ties[0], 16, 32));` |
| compute.reduce#2 | 0.11 | 0.7% | 1.0 | 0.16 | `se[0] = vllm_add(se[0], __shfl_xor_sync((uint)4294967295, se[0], 16, 32));` |
| compute.reduce | 0.11 | 0.6% | 1.0 | 0.13 | `best[0] = vllm_max(best[0], __shfl_xor_sync((uint)4294967295, best[0], 16, 32));` |
| wait.sync#4 | 0.10 | 0.6% | 1.0 | 0.16 | `__syncthreads();` |
| wait.sync#3 | 0.08 | 0.5% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#5 | 0.07 | 0.5% | 1.0 | 0.10 | `__syncthreads();` |
| load.global#3 (sampled iterations) | 0.01 | 0.1% | 0.0 | 0.38 | `float value_32 = vllm_sub(vllm_shifted(logits[((((int64_t)((int)blockIdx.x)) * logits_stri` |
| store.global (sampled iterations) | 0.00 | 0.0% | 0.0 | 0.06 | `output[(((((int64_t)c) * (int64_t)1024) + (((int64_t)((int)blockIdx.x)) * ((int64_t)topk))` |

### scatter_rows_gather_kernel grid (1, 1, 1) block (128, 1, 1)

17 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 0.90 us, warp 0.82 us, nsys graphs 1.82 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.56 | 68.0% | 0.39 | 40.6% |
| store | 0.15 | 18.4% | 0.45 | 47.3% |
| load | 0.11 | 13.6% | 0.11 | 12.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.15 | 18.4% | 1.0 | 0.42 | `if (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) < (num_reqs * width))` |
| load.global | 0.11 | 13.6% | 1.0 | 0.13 | `index = idx_mapping[min((((((int64_t)0 <= ((int64_t)width)) && ((int64_t)0 <= rmod_1)) \|\| ` |

### sparse_mla_kernel_1 grid (16, 6, 1) block (128, 1, 1)

6 launches, 96 CTAs, 384 warps; span 1.92 us, warp 1.36 us, nsys graphs 1.98 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.31 | 96.5% | 1.06 | 42.3% |
| load | 0.03 | 2.4% | 1.22 | 48.7% |
| wait | 0.01 | 0.7% | 0.10 | 3.8% |
| compute | 0.00 | 0.2% | 0.08 | 3.2% |
| store | 0.00 | 0.1% | 0.05 | 1.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.03 | 2.4% | 0.1 | 0.98 | `M = __fmul_rn(Sinks[((int)blockIdx.x)], 0x1.7154764ee6c2fp+0f/*1.442695e+00*/); \| if (((in` |
| wait.sync | 0.01 | 0.7% | 0.1 | 0.38 | `__syncthreads();` |
| compute.reduce | 0.00 | 0.2% | 0.0 | 0.10 | `amax[0] = tl::AllReduce<tl::MaxOp, 8, 1, 0, tl::NamedBarrier<128>>::run(amax[0]);` |
| store.global | 0.00 | 0.1% | 0.0 | 0.06 | `*(uint2*)(Output + (((((int64_t)((int)blockIdx.y)) * (int64_t)8192) + (((int64_t)((int)blo` |
| compute.reduce#3 | 0.00 | 0.0% | 0.0 | 0.00 | `amax[0] = tl::AllReduce<tl::MaxOp, 8, 1, 0, tl::NamedBarrier<128>>::run(amax[0]);` |
| load.global#3 | 0.00 | 0.0% | 0.0 | 0.00 | `M = __fmul_rn(Sinks[((int)blockIdx.x)], 0x1.7154764ee6c2fp+0f/*1.442695e+00*/);` |
| load.global#4 | 0.00 | 0.0% | 0.0 | 0.00 | `if (((int)threadIdx.x) < 32)` |
| store.global#2 | 0.00 | 0.0% | 0.0 | 0.00 | `*(uint2*)(Output + (((((int64_t)((int)blockIdx.y)) * (int64_t)8192) + (((int64_t)((int)blo` |
| wait.sync#12 | 0.00 | 0.0% | 0.0 | 0.00 | `__syncthreads();` |
| wait.sync#13 | 0.00 | 0.0% | 0.0 | 0.00 | `__syncthreads();` |

### mhc_collapse_column_kernel grid (25, 1, 1) block (128, 1, 1)

16 launches, 100 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 0.80 us, warp 0.77 us, nsys graphs 1.02 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.62 | 80.6% | 0.79 | 82.7% |
| load | 0.10 | 12.5% | 0.11 | 11.3% |
| store | 0.05 | 6.9% | 0.06 | 6.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.10 | 12.5% | 1.0 | 0.13 | `*(float4*)(mv + 0) = *(float4*)(mix + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int6` |
| store.global | 0.05 | 6.9% | 1.0 | 0.06 | `*(uint4*)(out + ((((int64_t)((int)blockIdx.x)) * (int64_t)1024) + (((int64_t)((int)threadI` |

### moe_swiglu_cast_kernel_kernel grid (8, 36, 1) block (288, 1, 1)

6 launches, 2592 warps; ranges, warp and last warp from the 4 recording CTAs' 36 warps; span 2.56 us, warp 1.04 us, nsys graphs 1.55 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.62 | 60.1% | 1.51 | 59.8% |
| load | 0.25 | 23.8% | 0.34 | 13.3% |
| compute | 0.10 | 9.9% | 0.41 | 16.0% |
| store | 0.06 | 6.2% | 0.28 | 10.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.25 | 23.8% | 1.0 | 0.35 | `for (int r = 0; r < 4; ++r)` |
| compute.reduce*4 (est.: sampled iterations x4) | 0.10 | 9.9% | 0.2 | 0.51 | `if (0 <= orders[r_2])` |
| store.global*4 (est.: sampled iterations x4) | 0.06 | 6.2% | 0.2 | 0.32 | `if (0 <= orders[r_2])` |

### ced_copy_rows_kernel grid (50, 1, 1) block (128, 1, 1)

17 launches, 200 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 0.90 us, warp 0.83 us, nsys graphs 1.15 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.57 | 69.0% | 0.61 | 70.0% |
| unattributed | 0.26 | 31.0% | 0.26 | 30.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.57 | 69.0% | 1.0 | 0.61 | `for (int k = 0; k < 4; ++k) \| int64_t index = ced_ldg64((&(row_indices[((((int64_t)blockId` |

### mhc_collapse_kernel grid (50, 1, 1) block (128, 1, 1)

17 launches, 200 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 0.86 us, warp 0.75 us, nsys graphs 1.06 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.60 | 79.7% | 0.66 | 79.3% |
| load | 0.10 | 13.9% | 0.12 | 14.2% |
| store | 0.05 | 6.4% | 0.05 | 6.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.10 | 13.9% | 1.0 | 0.13 | `*(float4*)(mv + 0) = *(float4*)(mix + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int6` |
| store.global | 0.05 | 6.4% | 1.0 | 0.06 | `*(uint4*)(out + ((((int64_t)((int)blockIdx.x)) * (int64_t)1024) + (((int64_t)((int)threadI` |

### local_stats_kernel grid (6, 16, 1) block (128, 1, 1)

1 launches, 384 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 12.38 us, warp 11.71 us, nsys graphs 12.38 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 10.50 | 89.6% | 10.62 | 87.8% |
| unattributed | 0.71 | 6.1% | 0.74 | 6.1% |
| wait | 0.26 | 2.2% | 0.48 | 4.0% |
| compute | 0.17 | 1.4% | 0.16 | 1.3% |
| store | 0.08 | 0.6% | 0.10 | 0.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 10.50 | 89.6% | 1.0 | 11.07 | `if (expanded_local_pos[((int64_t)((int)blockIdx.x))] < num_speculative_steps)` |
| wait.sync | 0.26 | 2.2% | 1.0 | 0.51 | `__syncthreads();` |
| compute.reduce | 0.11 | 0.9% | 1.0 | 0.13 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| store.global | 0.08 | 0.6% | 1.0 | 0.13 | `if (((int)threadIdx.x) == 0)` |
| compute.reduce#2 | 0.06 | 0.5% | 1.0 | 0.06 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 2, 32);` |

### padding_rows_kernel grid (1, 1, 1) block (256, 1, 1)

33 launches, 8 warps; ranges, warp and last warp from the 1 recording CTAs' 8 warps; span 0.38 us, warp 0.36 us, nsys graphs 0.77 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.26 | 73.0% | 0.27 | 72.9% |
| store | 0.10 | 27.0% | 0.10 | 27.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.10 | 27.0% | 1.0 | 0.10 | `if (((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) < rows)` |

### moe_combine_kernel grid (5, 6, 1) block (128, 1, 1)

6 launches, 120 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 1.98 us, warp 1.49 us, nsys graphs 1.18 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.04 | 70.2% | 1.52 | 77.2% |
| unattributed | 0.39 | 26.3% | 0.40 | 20.3% |
| store | 0.05 | 3.5% | 0.05 | 2.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.04 | 70.2% | 1.0 | 1.52 | `for (int k = 0; k < 3; ++k) \| for (int k = 0; k < 6; ++k) \| if ((((int)blockIdx.y) < share` |
| store.global | 0.05 | 3.5% | 1.0 | 0.06 | `{` |

### scatter_rows_kernel grid (1, 1, 1) block (128, 1, 1)

17 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 0.64 us, warp 0.58 us, nsys graphs 1.02 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.40 | 68.5% | 0.39 | 61.5% |
| load | 0.10 | 17.8% | 0.11 | 17.1% |
| store | 0.08 | 13.7% | 0.14 | 21.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.10 | 17.8% | 1.0 | 0.13 | `index = idx_mapping[min((((((int64_t)0 <= ((int64_t)width)) && ((int64_t)0 <= rmod_1)) \|\| ` |
| store.global | 0.08 | 13.7% | 1.0 | 0.13 | `if (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) < (num_reqs * width))` |

### ced_copy_rows_kernel grid (25, 1, 1) block (128, 1, 1)

16 launches, 100 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 0.67 us, warp 0.59 us, nsys graphs 1.02 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.36 | 60.3% | 0.39 | 60.1% |
| unattributed | 0.23 | 39.7% | 0.26 | 39.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.36 | 60.3% | 1.0 | 0.40 | `for (int k = 0; k < 4; ++k) \| int64_t index = ced_ldg64((&(row_indices[((((int64_t)blockId` |

### mhc_collapse_kernel grid (30, 1, 1) block (128, 1, 1)

3 launches, 120 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 3.62 us, warp 3.44 us, nsys graphs 2.24 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 3.30 | 95.7% | 3.33 | 95.7% |
| load | 0.10 | 2.8% | 0.10 | 2.8% |
| store | 0.05 | 1.5% | 0.05 | 1.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.10 | 2.8% | 1.0 | 0.10 | `*(float4*)(mv + 0) = *(float4*)(mix + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int6` |
| store.global | 0.05 | 1.5% | 1.0 | 0.06 | `*(uint4*)(out + ((((int64_t)((int)blockIdx.x)) * (int64_t)1024) + (((int64_t)((int)threadI` |

### indexer_topk_segment_kernel grid (2, 6, 1) block (512, 1, 1)

1 launches, 192 warps; ranges, warp and last warp from the 4 recording CTAs' 64 warps; span 9.79 us, warp 5.17 us, nsys graphs 0.00 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 2.96 | 57.2% | 5.34 | 54.6% |
| unattributed, with loops' unsampled iterations | 1.82 | 35.3% | 3.81 | 38.9% |
| load | 0.34 | 6.6% | 0.54 | 5.6% |
| store | 0.04 | 0.9% | 0.10 | 1.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync#19*9 (est.: sampled iterations x9) | 0.35 | 6.7% | 0.5 | 0.86 | `__syncthreads();` |
| wait.sync#24*9 (est.: sampled iterations x9) | 0.31 | 5.9% | 0.5 | 0.86 | `__syncthreads();` |
| wait.sync#13*8 (est.: sampled iterations x8) | 0.30 | 5.7% | 0.5 | 0.77 | `__syncthreads();` |
| wait.sync#6*8 (est.: sampled iterations x8) | 0.30 | 5.7% | 0.5 | 0.77 | `__syncthreads();` |
| wait.sync#20*9 (est.: sampled iterations x9) | 0.29 | 5.7% | 0.5 | 0.86 | `__syncthreads();` |
| wait.sync#23*9 (est.: sampled iterations x9) | 0.29 | 5.6% | 0.5 | 0.58 | `__syncthreads();` |
| wait.sync#14*8 (est.: sampled iterations x8) | 0.27 | 5.3% | 0.5 | 0.77 | `__syncthreads();` |
| wait.sync#5*8 (est.: sampled iterations x8) | 0.27 | 5.3% | 0.5 | 0.77 | `__syncthreads();` |
| load.global | 0.21 | 4.0% | 1.0 | 0.22 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 4096); \| i` |
| load.global#2 | 0.13 | 2.6% | 0.5 | 0.32 | `for (int i = 0; i < 8; ++i)` |
| wait.sync (sampled iterations) | 0.05 | 1.0% | 0.5 | 0.16 | `__syncthreads();` |
| wait.sync#25 | 0.05 | 0.9% | 0.5 | 0.16 | `__syncthreads();` |
| store.global | 0.04 | 0.9% | 0.5 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3) \| if (((int)threadIdx.x) < total) \| if (((int)threadIdx.` |
| wait.sync#16 | 0.04 | 0.7% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#17 | 0.04 | 0.7% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#18 | 0.04 | 0.7% | 0.5 | 0.13 | `__syncthreads();` |
| wait.sync#8 | 0.04 | 0.7% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#3 | 0.04 | 0.7% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#9 | 0.03 | 0.7% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#22 | 0.03 | 0.6% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#12 | 0.03 | 0.6% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#10 | 0.03 | 0.6% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#2 | 0.03 | 0.6% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#15 | 0.03 | 0.6% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#7 | 0.03 | 0.6% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#21 | 0.03 | 0.5% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#11 | 0.03 | 0.5% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#4 | 0.03 | 0.5% | 0.5 | 0.06 | `__syncthreads();` |

### indexer_topk_merge_kernel grid (1, 6, 1) block (512, 1, 1)

1 launches, 96 warps; ranges, warp and last warp from the 6 recording CTAs' 96 warps; span 9.31 us, warp 9.09 us, nsys graphs 5.90 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 6.00 | 66.0% | 5.76 | 61.9% |
| unattributed | 2.43 | 26.7% | 2.72 | 29.2% |
| load | 0.59 | 6.4% | 0.74 | 7.9% |
| store | 0.08 | 0.8% | 0.10 | 1.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync#20*9 (est.: sampled iterations x9) | 0.71 | 7.8% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#24*9 (est.: sampled iterations x9) | 0.65 | 7.1% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#19*9 (est.: sampled iterations x9) | 0.62 | 6.8% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#23*9 (est.: sampled iterations x9) | 0.59 | 6.5% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#14*8 (est.: sampled iterations x8) | 0.59 | 6.5% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync#6*8 (est.: sampled iterations x8) | 0.59 | 6.5% | 1.0 | 0.77 | `__syncthreads();` |
| load.global | 0.57 | 6.3% | 1.0 | 0.70 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 8192);` |
| wait.sync#13*8 (est.: sampled iterations x8) | 0.55 | 6.0% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync#5*8 (est.: sampled iterations x8) | 0.54 | 5.9% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync | 0.15 | 1.7% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#17 | 0.08 | 0.9% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#16 | 0.08 | 0.8% | 1.0 | 0.13 | `__syncthreads();` |
| wait.sync#8 | 0.07 | 0.8% | 1.0 | 0.10 | `__syncthreads();` |
| store.global#2 | 0.07 | 0.7% | 0.9 | 0.10 | `Values[((((int64_t)((int)blockIdx.y)) * (int64_t)512) + ((int64_t)((int)threadIdx.x)))] = ` |
| wait.sync#9 | 0.07 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#15 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#2 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#25 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#11 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#3 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#18 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#22 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#10 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#12 | 0.06 | 0.6% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#4 | 0.06 | 0.6% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#7 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#21 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| load.global#2 | 0.01 | 0.1% | 0.1 | 0.10 | `if (((0 <= ((int*)sel_idx)[((int)threadIdx.x)]) && (((int*)sel_idx)[((int)threadIdx.x)] < ` |
| store.global | 0.01 | 0.1% | 0.1 | 0.13 | `Values[((((int64_t)((int)blockIdx.y)) * (int64_t)512) + ((int64_t)((int)threadIdx.x)))] = ` |

### token_ranks_kernel grid (6, 1, 1) block (256, 1, 1)

1 launches, 48 warps; ranges, warp and last warp from the 6 recording CTAs' 48 warps; span 7.78 us, warp 7.69 us, nsys graphs 7.94 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 6.95 | 90.4% | 6.94 | 89.3% |
| unattributed | 0.36 | 4.7% | 0.42 | 5.3% |
| compute | 0.33 | 4.3% | 0.35 | 4.5% |
| store | 0.04 | 0.5% | 0.06 | 0.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 6.95 | 90.4% | 1.0 | 7.17 | `if ((((int64_t)0 <= token_ids[((int64_t)((int)blockIdx.x))]) && (token_ids[((int64_t)((int` |
| compute.reduce | 0.33 | 4.3% | 1.0 | 0.67 | `total[0] = tl::AllReduce<tl::SumOp, 256, 1, 0, tl::NamedBarrier<256>>::run(total[0], (&(wo` |
| store.global | 0.04 | 0.5% | 1.0 | 0.06 | `output[((int64_t)((int)blockIdx.x))] = ((int64_t)total[0]);` |

### rope_kernel grid (6, 1, 1) block (128, 1, 1)

6 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 1.30 us, warp 1.27 us, nsys graphs 1.47 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.72 | 56.4% | 0.74 | 59.1% |
| load | 0.51 | 40.4% | 0.47 | 37.4% |
| store | 0.04 | 3.2% | 0.04 | 3.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.51 | 40.4% | 1.0 | 0.51 | `if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < tokens)) \| int64_t position ` |
| store.global | 0.04 | 3.2% | 1.0 | 0.06 | `*(uint4*)(x + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + ((((int64_t)((int)thread` |

### indexer_mxfp4_quantize_kernel grid (6, 1, 1) block (128, 1, 1)

2 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 3.73 us, warp 3.65 us, nsys graphs 2.94 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 2.36 | 64.6% | 2.61 | 70.0% |
| store | 1.14 | 31.1% | 1.07 | 28.8% |
| unattributed | 0.16 | 4.3% | 0.05 | 1.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2*16 (est.: sampled iterations x16) | 1.35 | 37.1% | 1.0 | 1.54 | `float v__2 = __fmul_rn(((float)X[(((((int64_t)((int)blockIdx.x)) * (int64_t)4096) + (((int` |
| store.global*16 (est.: sampled iterations x16) | 1.10 | 30.1% | 1.0 | 1.54 | `Pool[(((((slot & (int64_t)127) * (int64_t)64) + ((((int64_t)((int)threadIdx.x)) & (int64_t` |
| load.global | 1.01 | 27.5% | 1.0 | 1.07 | `for (int i = 0; i < 32; ++i)` |
| store.global#2 | 0.04 | 1.1% | 1.0 | 0.06 | `Pool[(((((slot & (int64_t)127) * (int64_t)4) + ((slot >> (int64_t)7) * ((int64_t)pool_stri` |

### mhc_post_kernel grid (30, 1, 1) block (128, 1, 1)

4 launches, 120 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 1.84 us, warp 1.62 us, nsys graphs 1.31 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.47 | 90.6% | 1.62 | 90.6% |
| load | 0.10 | 6.5% | 0.12 | 6.7% |
| store | 0.05 | 3.0% | 0.05 | 2.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.10 | 6.5% | 1.0 | 0.13 | `*(float4*)(pm + 0) = *(float4*)(post + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int` |
| store.global | 0.05 | 3.0% | 1.0 | 0.06 | `for (int n_1 = 0; n_1 < 4; ++n_1)` |

### activation_swiglu_clamp_kernel grid (1, 6, 1) block (128, 1, 1)

6 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 0.91 us, warp 0.71 us, nsys graphs 1.86 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.60 | 85.1% | 0.75 | 83.4% |
| load | 0.07 | 10.1% | 0.10 | 10.7% |
| store | 0.03 | 4.8% | 0.05 | 5.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.07 | 10.1% | 0.8 | 0.10 | `*(uint4*)(gate + 0) = *(uint4*)(X + ((((int64_t)((int)blockIdx.y)) * (int64_t)1152) + (((i` |
| store.global | 0.03 | 4.8% | 0.8 | 0.06 | `*(uint4*)(Y + ((((int64_t)((int)blockIdx.y)) * (int64_t)576) + (((int64_t)((int)threadIdx.` |

### indexer_score_kernel grid (128, 6, 1) block (128, 1, 1)

1 launches, 3072 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 4.64 us, warp 2.17 us, nsys graphs 2.83 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.23 | 56.6% | 1.76 | 56.1% |
| unattributed, with loops' unsampled iterations | 0.69 | 31.6% | 0.99 | 31.6% |
| wait | 0.18 | 8.1% | 0.22 | 7.1% |
| compute | 0.06 | 2.8% | 0.13 | 4.1% |
| store | 0.02 | 0.9% | 0.03 | 1.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.57 | 26.0% | 1.0 | 0.77 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 4096); \| i` |
| load.global#2 (sampled iterations) | 0.56 | 25.9% | 0.5 | 1.28 | `for (int i_1 = 0; i_1 < 32; ++i_1)` |
| load.global#3 (sampled iterations) | 0.10 | 4.7% | 0.5 | 0.42 | `if (((int)threadIdx.x) < 64)` |
| wait.sync#2 (sampled iterations) | 0.09 | 4.2% | 0.5 | 0.42 | `__syncthreads();` |
| compute.mma (sampled iterations) | 0.06 | 2.8% | 0.5 | 0.13 | `for (int ki = 0; ki < 2; ++ki)` |
| wait.sync (sampled iterations) | 0.03 | 1.4% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#4 (sampled iterations) | 0.03 | 1.4% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#3 (sampled iterations) | 0.02 | 1.1% | 0.5 | 0.06 | `__syncthreads();` |
| store.global (sampled iterations) | 0.02 | 0.9% | 0.5 | 0.06 | `if ((((int)threadIdx.x) >> 6) == 0)` |

### indexer_score_kernel grid (64, 6, 1) block (128, 1, 1)

1 launches, 1536 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 4.10 us, warp 2.39 us, nsys graphs 0.00 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.13 | 47.4% | 1.82 | 44.9% |
| unattributed, with loops' unsampled iterations | 0.90 | 37.7% | 1.57 | 38.6% |
| wait | 0.24 | 10.1% | 0.51 | 12.6% |
| compute | 0.07 | 3.0% | 0.13 | 3.1% |
| store | 0.04 | 1.8% | 0.03 | 0.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 (sampled iterations) | 0.58 | 24.2% | 0.5 | 1.25 | `for (int i_1 = 0; i_1 < 32; ++i_1)` |
| load.global | 0.49 | 20.5% | 1.0 | 0.83 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 4096); \| i` |
| wait.sync#2 (sampled iterations) | 0.10 | 4.3% | 0.5 | 0.32 | `__syncthreads();` |
| wait.sync#4 (sampled iterations) | 0.08 | 3.5% | 0.5 | 0.26 | `__syncthreads();` |
| compute.mma (sampled iterations) | 0.07 | 3.0% | 0.5 | 0.16 | `for (int ki = 0; ki < 2; ++ki)` |
| load.global#3 (sampled iterations) | 0.06 | 2.7% | 0.5 | 0.32 | `if (((int)threadIdx.x) < 64)` |
| store.global (sampled iterations) | 0.04 | 1.8% | 0.5 | 0.19 | `if ((((int)threadIdx.x) >> 6) == 0)` |
| wait.sync#3 (sampled iterations) | 0.03 | 1.2% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync (sampled iterations) | 0.03 | 1.1% | 0.5 | 0.10 | `__syncthreads();` |

### gemm_splitk_reduce_kernel grid (1, 6, 1) block (128, 1, 1)

4 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 0.77 us, warp 0.75 us, nsys graphs 0.99 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.39 | 52.3% | 0.41 | 52.6% |
| load | 0.31 | 41.4% | 0.31 | 40.2% |
| store | 0.05 | 6.3% | 0.06 | 7.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.31 | 41.4% | 1.0 | 0.34 | `acc[0] = P[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x))` |
| store.global | 0.05 | 6.3% | 1.0 | 0.06 | `C[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x)) * (int64` |

### metadata_pages_kernel grid (6, 1, 1) block (128, 1, 1)

2 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 1.31 us, warp 0.95 us, nsys graphs 0.00 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.73 | 76.6% | 0.80 | 61.0% |
| load | 0.14 | 14.8% | 0.19 | 14.6% |
| store | 0.08 | 8.7% | 0.32 | 24.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.12 | 13.1% | 1.0 | 0.13 | `if ((((offset + ((int)blockIdx.x)) < src_rows) && (0 <= (offset + ((int)blockIdx.x)))))` |
| store.global | 0.08 | 8.7% | 0.2 | 0.34 | `if (((int)threadIdx.x) < 32)` |
| load.global#2 | 0.02 | 1.7% | 0.2 | 0.06 | `if (((req < table_rows) && (((int)threadIdx.x) < table_cols)))` |

### norm_forward_and_per_token_cast_kernel_kernel grid (6, 1, 1) block (128, 1, 1)

1 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 2.59 us, warp 2.55 us, nsys graphs 1.82 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 2.50 | 98.3% | 2.66 | 103.8% |
| load | 0.45 | 17.8% | 0.32 | 12.5% |
| compute | 0.13 | 4.9% | 0.13 | 5.0% |
| wait | 0.05 | 1.8% | 0.06 | 2.5% |
| unattributed | 0.00 | 0.0% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*5 (est.: sampled iterations x5) | 2.21 | 86.9% | 1.0 | 2.40 | `for (int j_1 = 0; j_1 < 2; ++j_1)` |
| load.global#2*5 (est.: sampled iterations x5) | 0.24 | 9.4% | 1.0 | 0.32 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 1024) + (((int)threadIdx.x) * 8)` |
| store.global*5 (est.: sampled iterations x5) | 0.24 | 9.4% | 1.0 | 0.32 | `for (int j = 0; j < 4; ++j)` |
| load.global*5 (est.: sampled iterations x5) | 0.21 | 8.4% | 1.0 | 0.32 | `*(uint4*)(x_local + (vi * 8)) = *(uint4*)(x + ((((((int64_t)vi) * (int64_t)1024) + (((int6` |
| compute.reduce | 0.13 | 4.9% | 1.0 | 0.13 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| wait.sync | 0.05 | 1.8% | 1.0 | 0.06 | `__syncthreads();` |
| store.global#2 | 0.04 | 1.5% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| store.global#3 | 0.01 | 0.5% | 0.2 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |

### rope_kernel grid (6, 1, 1) block (256, 1, 1)

2 launches, 48 warps; ranges, warp and last warp from the 6 recording CTAs' 48 warps; span 0.99 us, warp 0.90 us, nsys graphs 1.12 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.60 | 66.9% | 0.67 | 67.7% |
| load | 0.26 | 28.7% | 0.26 | 25.8% |
| store | 0.04 | 4.4% | 0.06 | 6.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.26 | 28.7% | 1.0 | 0.27 | `if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < tokens)) \| int64_t position ` |
| store.global | 0.04 | 4.4% | 1.0 | 0.06 | `*(uint4*)(x + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + ((((int64_t)((int)thread` |

### mhc_collapse_column_kernel grid (30, 1, 1) block (128, 1, 1)

1 launches, 120 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 0.80 us, warp 0.77 us, nsys graphs 0.88 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.61 | 79.3% | 0.61 | 79.2% |
| load | 0.10 | 12.7% | 0.10 | 12.5% |
| store | 0.06 | 8.1% | 0.06 | 8.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.10 | 12.7% | 1.0 | 0.13 | `*(float4*)(mv + 0) = *(float4*)(mix + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int6` |
| store.global | 0.06 | 8.1% | 1.0 | 0.06 | `*(uint4*)(out + ((((int64_t)((int)blockIdx.x)) * (int64_t)1024) + (((int64_t)((int)threadI` |

### ced_copy_rows_kernel grid (30, 1, 1) block (128, 1, 1)

1 launches, 120 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 0.64 us, warp 0.59 us, nsys graphs 1.09 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.36 | 60.6% | 0.38 | 60.0% |
| unattributed | 0.23 | 39.4% | 0.26 | 40.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.36 | 60.6% | 1.0 | 0.38 | `for (int k = 0; k < 4; ++k) \| int64_t index = ced_ldg64((&(row_indices[((((int64_t)blockId` |
=== timeline and tables: exit 0 in 22 s
