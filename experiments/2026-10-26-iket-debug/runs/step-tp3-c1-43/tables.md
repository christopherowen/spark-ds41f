=== timeline and tables
trace /c/iket/step-tp3-c1-20261026T173405Z/trace/iket/pid_0x1879/iket.decoded_results.json: 12688 launches, graphLaunches dict of 124

## Timeline: 7754 launches in 774230.2 us; the GPU is busy 285022.9 us (37%) and idle 489207.3 us in 4872 gaps; gaps over 200 us split it into 336 segments

The 15 of 336 segments with the most busy time:

| segment | launches | wall us | busy us | busy | first kernel |
|---:|---:|---:|---:|---:|---|
| 89 | 116 | 6898.4 | 6822.2 | 99% | mhc_project_streams_kernel |
| 97 | 120 | 7355.1 | 6768.2 | 92% | metadata_builder_kernel |
| 125 | 120 | 7345.6 | 6783.5 | 92% | metadata_builder_kernel |
| 156 | 116 | 6859.0 | 6774.0 | 99% | mhc_project_streams_kernel |
| 170 | 120 | 7276.6 | 6768.1 | 93% | metadata_builder_kernel |
| 185 | 116 | 6859.2 | 6775.4 | 99% | mhc_project_streams_kernel |
| 203 | 116 | 6856.1 | 6768.2 | 99% | mhc_project_streams_kernel |
| 221 | 117 | 7123.6 | 6987.2 | 98% | padding_rows_kernel |
| 233 | 120 | 7456.2 | 6778.8 | 91% | metadata_builder_kernel |
| 253 | 120 | 7521.7 | 6796.0 | 90% | metadata_builder_kernel |
| 282 | 117 | 7026.2 | 6777.7 | 96% | padding_rows_kernel |
| 296 | 120 | 7208.3 | 6765.0 | 94% | metadata_builder_kernel |
| 305 | 120 | 7166.8 | 6783.6 | 95% | metadata_builder_kernel |
| 322 | 117 | 7258.0 | 7154.4 | 99% | padding_rows_kernel |
| 333 | 120 | 7511.8 | 6774.6 | 90% | metadata_builder_kernel |

| idle us | after | before |
|---:|---|---|
| 126464.4 | resample_kernel | insert_kernel |
| 92392.9 | scatter_rows_kernel | padding_rows_kernel |
| 3301.6 | dspark_confidence_kernel | scatter_rows_gather_kernel |
| 3143.4 | dspark_confidence_kernel | scatter_rows_gather_kernel |
| 3117.8 | dspark_confidence_kernel | scatter_rows_gather_kernel |
| 3108.8 | norm_forward_and_per_token_cast_kernel_kernel | sink_flags_kernel |
| 3102.6 | norm_forward_and_per_token_cast_kernel_kernel | sink_flags_kernel |
| 3100.8 | norm_forward_and_per_token_cast_kernel_kernel | sink_flags_kernel |
| 3092.0 | norm_forward_and_per_token_cast_kernel_kernel | sink_flags_kernel |
| 3088.0 | norm_forward_and_per_token_cast_kernel_kernel | sink_flags_kernel |

## Engine steps

| step | kind | launches | wall us | busy us | busy | most span |
|---:|---|---:|---:|---:|---:|---|
| 0 | prefill | 413 | 57403.6 | 17291.5 | 30% | vocab_packed_projection_kernel 8067, gemm_mxfp8_kernel 3116, moe_grouped_fp8_fp4_gemm_kernel 2417 |
| 1 | decode | 432 | 157248.7 | 15259.4 | 10% | vocab_packed_projection_kernel 8128, gemm_mxfp8_kernel 2380, l2_prefetch_kernel 1871 |
| 2 | decode | 432 | 118820.5 | 15535.9 | 13% | vocab_packed_projection_kernel 8021, gemm_mxfp8_kernel 2470, l2_prefetch_kernel 2135 |
| 3 | decode | 432 | 28847.1 | 15619.4 | 54% | vocab_packed_projection_kernel 8107, gemm_mxfp8_kernel 2520, l2_prefetch_kernel 2332 |
| 4 | decode | 432 | 29248.1 | 15676.9 | 54% | vocab_packed_projection_kernel 8096, gemm_mxfp8_kernel 2281, moe_grouped_fp8_fp4_gemm_kernel 1808 |
| 5 | decode | 432 | 32090.0 | 16004.4 | 50% | vocab_packed_projection_kernel 8020, gemm_mxfp8_kernel 2422, moe_grouped_fp8_fp4_gemm_kernel 2028 |
| 6 | decode | 432 | 28398.0 | 15685.5 | 55% | vocab_packed_projection_kernel 8077, gemm_mxfp8_kernel 2297, moe_grouped_fp8_fp4_gemm_kernel 1832 |
| 7 | decode | 432 | 29315.5 | 16033.0 | 55% | vocab_packed_projection_kernel 8024, gemm_mxfp8_kernel 2410, moe_grouped_fp8_fp4_gemm_kernel 2047 |
| 8 | decode | 432 | 29748.7 | 15604.7 | 52% | vocab_packed_projection_kernel 8005, gemm_mxfp8_kernel 2335, moe_grouped_fp8_fp4_gemm_kernel 1821 |
| 9 | decode | 432 | 31243.2 | 16192.3 | 52% | vocab_packed_projection_kernel 8085, gemm_mxfp8_kernel 2341, moe_grouped_fp8_fp4_gemm_kernel 2008 |
| 10 | decode | 432 | 28100.8 | 15722.0 | 56% | vocab_packed_projection_kernel 8092, gemm_mxfp8_kernel 2266, moe_grouped_fp8_fp4_gemm_kernel 1822 |
| 11 | decode | 432 | 30506.0 | 15955.3 | 52% | vocab_packed_projection_kernel 8037, gemm_mxfp8_kernel 2422, moe_grouped_fp8_fp4_gemm_kernel 1880 |
| 12 | decode | 432 | 29157.5 | 15647.0 | 54% | vocab_packed_projection_kernel 8061, gemm_mxfp8_kernel 2318, moe_grouped_fp8_fp4_gemm_kernel 1824 |
| 13 | decode | 432 | 28866.4 | 15923.9 | 55% | vocab_packed_projection_kernel 8029, gemm_mxfp8_kernel 2344, moe_grouped_fp8_fp4_gemm_kernel 1897 |
| 14 | decode | 432 | 28150.8 | 15633.9 | 56% | vocab_packed_projection_kernel 8022, gemm_mxfp8_kernel 2287, moe_grouped_fp8_fp4_gemm_kernel 1838 |
| 15 | decode | 432 | 27147.3 | 15604.8 | 57% | vocab_packed_projection_kernel 8029, gemm_mxfp8_kernel 2249, moe_grouped_fp8_fp4_gemm_kernel 1812 |
| 16 | decode | 432 | 29601.3 | 15999.6 | 54% | vocab_packed_projection_kernel 8343, gemm_mxfp8_kernel 2302, moe_grouped_fp8_fp4_gemm_kernel 1843 |
| 17 | decode | 429 | 27243.6 | 15633.4 | 57% | vocab_packed_projection_kernel 8059, gemm_mxfp8_kernel 2326, moe_grouped_fp8_fp4_gemm_kernel 1835 |

## Overview over 17 decode steps: 7341 launches of 142 kernel shapes, spans summing to 299684.4 us (17628.5 a step)

| kernel | grid | launches a step | span us a step | share | span us a launch | last warp: wait | load | compute | store | unattributed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vocab_packed_projection_kernel | (1, 4040, 1) | 2 | 8072.5 | 45.8% | 4013.86 | 47% | 32% | 10% | 0% | 10% |
| l2_prefetch_kernel | (2, 1, 1) | 25 | 1703.5 | 9.7% | 64.90 | 0% | 99% | 0% | 0% | 1% |
| gemm_bf16_kernel | (2020, 1, 1) | 5 | 1342.4 | 7.6% | 266.50 | 90% | 0% | 4% | 1% | 5% |
| moe_grouped_fp8_fp4_gemm_kernel | (12, 30, 1) | 8.64706 | 1190.8 | 6.8% | 168.45 | 65% | 16% | 20% | 1% | 0% |
| gemm_mxfp8_kernel | (80, 1, 1) | 19 | 1095.0 | 6.2% | 51.78 | 69% | 8% | 6% | 0% | 18% |
| moe_grouped_fp8_fp4_gemm_kernel | (40, 30, 1) | 8.64706 | 602.1 | 3.4% | 82.18 | 15% | 18% | 0% | 0% | 67% |
| gemm_mxfp8_kernel | (192, 1, 1) | 9 | 546.1 | 3.1% | 51.97 | 25% | 5% | 5% | 1% | 65% |
| gemm_mxfp8_kernel | (28, 1, 1) | 9 | 340.6 | 1.9% | 37.73 | 47% | 10% | 10% | 0% | 32% |
| mhc_project_streams_kernel | (40, 1, 1) | 18 | 326.4 | 1.9% | 17.78 | 23% | 31% | 15% | 3% | 28% |
| gemm_mxfp8_kernel | (24, 1, 1) | 9 | 320.7 | 1.8% | 31.97 | 35% | 19% | 17% | 0% | 29% |
| gemm_fp8w_bf16_partials_kernel | (48, 4, 1) | 9 | 316.7 | 1.8% | 24.61 | 36% | 17% | 9% | 0% | 38% |
| gemm_bf16_partials_kernel | (6, 8, 1) | 6 | 146.6 | 0.8% | 24.35 | 38% | 0% | 26% | 1% | 35% |
| mhc_finalize_kernel | (5, 1, 1) | 11.2941 | 128.0 | 0.7% | 10.96 | 7% | 14% | 11% | 25% | 43% |
| gemm_bf16_splitk_kernel | (16, 5, 1) | 2 | 120.7 | 0.7% | 57.90 | 99% | 7% | 4% | 0% | 0% |
| sparse_mla_kernel | (5, 1, 5) | 3.76471 | 87.2 | 0.5% | 20.90 | 18% | 17% | 6% | 5% | 53% |
| sparse_mla_kernel | (2, 2, 10) | 3 | 63.9 | 0.4% | 21.79 | 14% | 25% | 22% | 2% | 36% |
| mhc_finalize_kernel | (10, 1, 1) | 6 | 63.5 | 0.4% | 10.51 | 5% | 14% | 14% | 27% | 40% |
| gemm_mxfp8_kernel | (64, 1, 1) | 2 | 48.8 | 0.3% | 23.36 | 39% | 17% | 11% | 1% | 33% |
| moe_dispatch_kernel | (1, 1, 1) | 9 | 47.8 | 0.3% | 4.86 | 33% | 2% | 0% | 29% | 36% |
| metadata_chunk_kernel | (10, 1, 1) | 3 | 47.1 | 0.3% | 15.74 | 0% | 77% | 0% | 1% | 23% |

| wait | kernel | grid | last-warp us a step | a launch | first statement |
|---|---|---|---:|---:|---|
| wait.cp_async*118 | gemm_mxfp8_kernel | (80, 1, 1) | 217.5 | 11.44 | `tl::cp_async_wait<2>();` |
| wait.cp_async*22 | gemm_mxfp8_kernel | (80, 1, 1) | 212.9 | 11.21 | `tl::cp_async_wait<2>();` |
| wait.mbarrier*40 | moe_grouped_fp8_fp4_gemm_kernel | (12, 30, 1) | 165.6 | 19.15 | `loaded[(ko % 3)].wait(((ko % 6) / 3));` |
| wait.sync#2*22 | gemm_mxfp8_kernel | (80, 1, 1) | 115.0 | 6.05 | `__syncthreads();` |
| wait.sync#2*118 | gemm_mxfp8_kernel | (80, 1, 1) | 91.5 | 4.82 | `__syncthreads();` |
| wait.cp_async*19 | gemm_mxfp8_kernel | (28, 1, 1) | 91.0 | 10.11 | `tl::cp_async_wait<1>();` |
| wait.cp_async*40 | moe_grouped_fp8_fp4_gemm_kernel | (12, 30, 1) | 79.5 | 9.20 | `tl::cp_async_wait<2>();` |
| wait.sync*4 | mhc_project_streams_kernel | (40, 1, 1) | 65.7 | 3.65 | `__syncthreads();` |
| wait.sync#2*19 | gemm_mxfp8_kernel | (28, 1, 1) | 53.4 | 5.93 | `__syncthreads();` |
| wait.sync#2*38 | gemm_mxfp8_kernel | (24, 1, 1) | 52.5 | 5.83 | `__syncthreads();` |
| wait.cp_async*13 | gemm_bf16_splitk_kernel | (16, 5, 1) | 51.7 | 25.87 | `tl::cp_async_wait<2>();` |
| wait.cp_async*38 | gemm_mxfp8_kernel | (24, 1, 1) | 30.5 | 3.39 | `tl::cp_async_wait<2>();` |
| wait.mbarrier#2 | gemm_bf16_kernel | (2020, 1, 1) | 29.9 | 5.98 | `mbarrier[0].wait(0);` |
| wait.sync*38 | gemm_mxfp8_kernel | (24, 1, 1) | 20.8 | 2.31 | `__syncthreads();` |
| wait.cp_async*14 | gemm_fp8w_bf16_partials_kernel | (48, 4, 1) | 20.6 | 2.29 | `tl::cp_async_wait<3>();` |
| wait.sync#2*4 | gemm_mxfp8_kernel | (80, 1, 1) | 20.2 | 1.07 | `__syncthreads();` |
| wait.cp_async*4 | gemm_mxfp8_kernel | (80, 1, 1) | 20.1 | 1.06 | `tl::cp_async_wait<2>();` |
| wait.cp_async#3 | gemm_mxfp8_kernel | (192, 1, 1) | 18.7 | 2.08 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>();` |
| wait.cp_async*12 | gemm_bf16_splitk_kernel | (16, 5, 1) | 18.4 | 9.19 | `tl::cp_async_wait<3>();` |
| wait.sync#2*17 | vocab_packed_projection_kernel | (1, 4040, 1) | 18.3 | 9.15 | `__syncthreads();` |

### vocab_packed_projection_kernel grid (1, 4040, 1) block (128, 1, 1)

34 launches, 16160 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 4013.86 us, warp 48.93 us, nsys graphs 4038.67 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 27.03 | 55.2% | 23.04 | 47.0% |
| load | 10.26 | 21.0% | 15.80 | 32.2% |
| unattributed | 6.54 | 13.4% | 5.08 | 10.4% |
| compute | 5.05 | 10.3% | 5.09 | 10.4% |
| store | 0.05 | 0.1% | 0.05 | 0.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.cp_async*17 (est.: sampled iterations x17) | 12.31 | 25.2% | 1.0 | 38.35 | `tl::cp_async_wait<6>();` |
| wait.sync#2*17 (est.: sampled iterations x17) | 9.47 | 19.4% | 1.0 | 29.65 | `__syncthreads();` |
| load.global | 7.81 | 16.0% | 1.0 | 14.50 | `int start = row_offsets[(((int)blockIdx.y) * 16)]; \| int start = row_offsets[(((int)blockI` |
| compute.mma*17 (est.: sampled iterations x17) | 4.35 | 8.9% | 1.0 | 4.35 | `{` |
| load.cp_async#2*17 (est.: sampled iterations x17) | 2.18 | 4.4% | 1.0 | 2.72 | `for (int i_6 = 0; i_6 < 8; ++i_6) \| for (int i_9 = 0; i_9 < 4; ++i_9)` |
| wait.sync#3*17 (est.: sampled iterations x17) | 1.09 | 2.2% | 1.0 | 1.63 | `__syncthreads();` |
| wait.sync*17 (est.: sampled iterations x17) | 1.05 | 2.2% | 1.0 | 1.09 | `__syncthreads();` |
| wait.cp_async#2*17 (est.: sampled iterations x17) | 0.99 | 2.0% | 1.0 | 1.09 | `tl::cp_async_wait<7>();` |
| wait.cp_async#3 | 0.46 | 0.9% | 1.0 | 1.92 | `tl::cp_async_wait<4>();` |
| wait.sync#4 | 0.45 | 0.9% | 1.0 | 1.50 | `__syncthreads();` |
| wait.sync#6 | 0.31 | 0.6% | 1.0 | 1.23 | `__syncthreads();` |
| load.cp_async | 0.28 | 0.6% | 1.0 | 0.32 | `for (int i = 0; i < 4; ++i) \| for (int i = 0; i < 8; ++i) \| for (int i_1 = 0; i_1 < 4; ++i` |
| wait.cp_async#5 | 0.25 | 0.5% | 1.0 | 1.42 | `tl::cp_async_wait<2>();` |
| compute.mma#2 | 0.23 | 0.5% | 1.0 | 0.26 | `{` |
| compute.mma#3 | 0.23 | 0.5% | 1.0 | 0.26 | `{` |
| compute.mma#4 | 0.23 | 0.5% | 1.0 | 0.26 | `{` |
| wait.sync#8 | 0.22 | 0.4% | 1.0 | 0.98 | `__syncthreads();` |
| wait.cp_async#7 | 0.15 | 0.3% | 1.0 | 0.98 | `tl::cp_async_wait<0>();` |
| wait.sync#5 | 0.06 | 0.1% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#7 | 0.05 | 0.1% | 1.0 | 0.06 | `__syncthreads();` |
| store.global | 0.05 | 0.1% | 1.0 | 0.06 | `for (int i_13 = 0; i_13 < 2; ++i_13) \| for (int i_14 = 0; i_14 < 4; ++i_14) \| for (int i_9` |
| wait.sync#9 | 0.05 | 0.1% | 1.0 | 0.06 | `__syncthreads();` |
| wait.cp_async#4 | 0.04 | 0.1% | 1.0 | 0.06 | `tl::cp_async_wait<5>();` |
| wait.cp_async#6 | 0.04 | 0.1% | 1.0 | 0.06 | `tl::cp_async_wait<3>();` |
| wait.cp_async#8 | 0.04 | 0.1% | 1.0 | 0.06 | `tl::cp_async_wait<1>();` |

### l2_prefetch_kernel grid (2, 1, 1) block (128, 1, 1)

425 launches, 8 warps; ranges, warp and last warp from the 2 recording CTAs' 8 warps; span 64.90 us, warp 64.15 us, nsys graphs 66.27 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 63.71 | 99.3% | 67.74 | 99.4% |
| unattributed | 0.44 | 0.7% | 0.40 | 0.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 63.71 | 99.3% | 1.0 | 64.58 | `for (int s = 0; s < (nseg2 >> 1); ++s)` |

### gemm_bf16_kernel grid (2020, 1, 1) block (256, 1, 1)

85 launches, 16160 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 266.50 us, warp 3.71 us, nsys graphs 261.20 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 3.14 | 84.7% | 6.07 | 90.0% |
| unattributed | 0.35 | 9.5% | 0.32 | 4.7% |
| compute | 0.14 | 3.9% | 0.29 | 4.3% |
| load | 0.04 | 1.0% | 0.00 | 0.0% |
| store | 0.04 | 1.0% | 0.07 | 1.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.mbarrier#2 | 3.03 | 81.8% | 0.5 | 7.23 | `mbarrier[0].wait(0);` |
| compute.mma | 0.14 | 3.9% | 0.5 | 0.29 | `{` |
| wait.sync | 0.08 | 2.1% | 1.0 | 0.10 | `__syncthreads();` |
| load.tma | 0.04 | 1.0% | 0.5 | 0.13 | `if (tl::tl_shuffle_elect<128>())` |
| store.global | 0.04 | 1.0% | 0.5 | 0.10 | `for (int i_3 = 0; i_3 < 4; ++i_3) \| for (int i_4 = 0; i_4 < 8; ++i_4) \| for (int i_6 = 0; ` |
| wait.mbarrier | 0.03 | 0.8% | 0.5 | 0.06 | `mbarrier[2].wait(1);` |

### moe_grouped_fp8_fp4_gemm_kernel grid (12, 30, 1) block (128, 1, 1)

147 launches, 1440 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 168.45 us, warp 34.99 us, nsys graphs 63.55 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 12.64 | 36.1% | 0.00 | 0.0% |
| wait | 11.88 | 34.0% | 34.06 | 65.4% |
| compute | 6.64 | 19.0% | 10.55 | 20.3% |
| load | 3.68 | 10.5% | 8.10 | 15.6% |
| store | 0.14 | 0.4% | 0.28 | 0.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*40 (est.: sampled iterations x40) | 6.64 | 19.0% | 0.5 | 14.08 | `{` |
| wait.mbarrier*40 (est.: sampled iterations x40) | 5.44 | 15.5% | 0.5 | 20.48 | `loaded[(ko % 3)].wait(((ko % 6) / 3));` |
| load#2*40 (est.: sampled iterations x40) | 2.96 | 8.5% | 0.5 | 8.96 | `if (ko < 38)` |
| wait.sync#5*40 (est.: sampled iterations x40) | 2.64 | 7.5% | 0.5 | 6.40 | `__syncthreads();` |
| wait.cp_async*40 (est.: sampled iterations x40) | 1.28 | 3.7% | 0.5 | 2.56 | `tl::cp_async_wait<2>();` |
| wait.sync#4*40 (est.: sampled iterations x40) | 1.20 | 3.4% | 0.5 | 2.56 | `__syncthreads();` |
| wait.sync#6*40 (est.: sampled iterations x40) | 1.20 | 3.4% | 0.5 | 2.56 | `__syncthreads();` |
| load.global#3 | 0.49 | 1.4% | 0.5 | 1.47 | `if (((expert < 128) && (0 <= expert))) \| if (((expert < 384) && (0 <= expert)))` |
| store.global | 0.14 | 0.4% | 0.5 | 0.38 | `for (int i_2 = 0; i_2 < 16; ++i_2) \| for (int i_4 = 0; i_4 < 32; ++i_4)` |
| load.global | 0.11 | 0.3% | 1.0 | 0.26 | `int expert = TileExpert[((int64_t)((int)blockIdx.y))];` |
| load | 0.08 | 0.2% | 0.5 | 0.26 | `for (int s = 0; s < 2; ++s)` |
| wait.sync#2 | 0.06 | 0.2% | 0.5 | 0.16 | `__syncthreads();` |
| load.global#2 | 0.05 | 0.1% | 0.5 | 0.19 | `if (((int)threadIdx.x) < 32) \| if (((int)threadIdx.x) < 64)` |
| wait.sync | 0.05 | 0.1% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#3 | 0.02 | 0.0% | 0.5 | 0.06 | `__syncthreads();` |

### gemm_mxfp8_kernel grid (80, 1, 1) block (128, 1, 1)

323 launches, 320 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 51.78 us, warp 47.53 us, nsys graphs 47.68 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 45.04 | 94.8% | 9.96 | 17.6% |
| wait | 1.95 | 4.1% | 38.79 | 68.7% |
| compute | 0.25 | 0.5% | 3.18 | 5.6% |
| load | 0.24 | 0.5% | 4.47 | 7.9% |
| store | 0.06 | 0.1% | 0.06 | 0.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.cp_async#2 | 0.63 | 1.3% | 1.0 | 2.24 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| wait.sync#3 | 0.49 | 1.0% | 1.0 | 1.63 | `__syncthreads();` |
| wait.cp_async#3 | 0.45 | 1.0% | 1.0 | 1.73 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>();` |
| wait.sync#4 | 0.37 | 0.8% | 1.0 | 1.47 | `__syncthreads();` |
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

147 launches, 4800 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 82.18 us, warp 5.68 us, nsys graphs 33.52 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 2.28 | 40.1% | 0.05 | 15.2% |
| unattributed | 1.27 | 22.3% | 0.21 | 66.8% |
| compute | 1.00 | 17.5% | 0.00 | 0.0% |
| load | 0.99 | 17.4% | 0.06 | 18.1% |
| store | 0.14 | 2.5% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync#5*6 (est.: sampled iterations x6) | 1.14 | 20.1% | 0.5 | 5.18 | `__syncthreads();` |
| compute.mma*6 (est.: sampled iterations x6) | 1.00 | 17.5% | 0.5 | 2.11 | `{` |
| wait.cp_async*6 (est.: sampled iterations x6) | 0.43 | 7.6% | 0.5 | 4.80 | `tl::cp_async_wait<2>();` |
| load.global#3 | 0.40 | 7.0% | 0.5 | 1.15 | `if (((expert < 128) && (0 <= expert))) \| if (((expert < 384) && (0 <= expert)))` |
| load#2*6 (est.: sampled iterations x6) | 0.40 | 7.0% | 0.5 | 1.15 | `if (ko < 4)` |
| wait.mbarrier*6 (est.: sampled iterations x6) | 0.24 | 4.2% | 0.5 | 0.58 | `loaded[(ko % 3)].wait((ko / 3));` |
| wait.sync#4*6 (est.: sampled iterations x6) | 0.18 | 3.2% | 0.5 | 0.38 | `__syncthreads();` |
| wait.sync#6*6 (est.: sampled iterations x6) | 0.17 | 3.0% | 0.5 | 0.38 | `__syncthreads();` |
| store.global | 0.14 | 2.5% | 0.5 | 0.38 | `for (int i_2 = 0; i_2 < 16; ++i_2) \| for (int i_4 = 0; i_4 < 32; ++i_4)` |
| load | 0.08 | 1.4% | 0.5 | 0.26 | `for (int s = 0; s < 2; ++s)` |
| load.global | 0.07 | 1.2% | 1.0 | 0.10 | `int expert = TileExpert[((int64_t)((int)blockIdx.y))];` |
| wait.sync#2 | 0.06 | 1.0% | 0.5 | 0.16 | `__syncthreads();` |
| load.global#2 | 0.04 | 0.8% | 0.5 | 0.19 | `if (((int)threadIdx.x) < 32) \| if (((int)threadIdx.x) < 64)` |
| wait.sync | 0.04 | 0.8% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#3 | 0.02 | 0.3% | 0.5 | 0.06 | `__syncthreads();` |

### gemm_mxfp8_kernel grid (192, 1, 1) block (128, 1, 1)

153 launches, 768 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 51.97 us, warp 30.77 us, nsys graphs 57.09 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 14.83 | 48.2% | 16.35 | 64.8% |
| wait | 12.82 | 41.7% | 6.19 | 24.5% |
| load | 1.72 | 5.6% | 1.21 | 4.8% |
| compute | 1.35 | 4.4% | 1.32 | 5.2% |
| store | 0.06 | 0.2% | 0.15 | 0.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync#2*8 (est.: sampled iterations x8) | 3.70 | 12.0% | 1.0 | 15.36 | `__syncthreads();` |
| wait.cp_async*8 (est.: sampled iterations x8) | 3.60 | 11.7% | 1.0 | 17.41 | `tl::cp_async_wait<2>();` |
| wait.cp_async#3 | 1.65 | 5.3% | 1.0 | 4.13 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>();` |
| wait.cp_async#2 | 1.48 | 4.8% | 1.0 | 5.15 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| load.cp_async#2*8 (est.: sampled iterations x8) | 1.39 | 4.5% | 1.0 | 4.10 | `for (int i_5 = 0; i_5 < 2; ++i_5) \| for (int i_5 = 0; i_5 < 4; ++i_5) \| tl::cp_async_gs_co` |
| compute.mma*8 (est.: sampled iterations x8) | 1.10 | 3.6% | 1.0 | 1.28 | `{` |
| wait.sync#4 | 1.01 | 3.3% | 1.0 | 3.01 | `__syncthreads();` |
| wait.sync#3 | 0.99 | 3.2% | 1.0 | 3.49 | `__syncthreads();` |
| wait.sync*8 (est.: sampled iterations x8) | 0.40 | 1.3% | 1.0 | 0.51 | `__syncthreads();` |
| load.cp_async | 0.33 | 1.1% | 1.0 | 0.83 | `for (int i_1 = 0; i_1 < 2; ++i_1) \| for (int i_1 = 0; i_1 < 4; ++i_1) \| for (int i_1 = 0; ` |
| compute.mma#3 | 0.13 | 0.4% | 1.0 | 0.16 | `{` |
| compute.mma#2 | 0.12 | 0.4% | 1.0 | 0.13 | `{` |
| store.global | 0.06 | 0.2% | 1.0 | 0.06 | `for (int i_10 = 0; i_10 < 16; ++i_10) \| for (int i_10 = 0; i_10 < 8; ++i_10) \| for (int i_` |

### gemm_mxfp8_kernel grid (28, 1, 1) block (128, 1, 1)

153 launches, 112 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 37.73 us, warp 36.04 us, nsys graphs 42.50 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 19.68 | 54.6% | 12.19 | 32.5% |
| wait | 9.28 | 25.8% | 17.53 | 46.7% |
| compute | 3.94 | 10.9% | 3.94 | 10.5% |
| load | 3.08 | 8.6% | 3.85 | 10.3% |
| store | 0.05 | 0.1% | 0.05 | 0.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync#2*19 (est.: sampled iterations x19) | 4.22 | 11.7% | 1.0 | 14.59 | `__syncthreads();` |
| compute.mma*19 (est.: sampled iterations x19) | 3.76 | 10.4% | 1.0 | 4.26 | `{` |
| wait.cp_async*19 (est.: sampled iterations x19) | 3.65 | 10.1% | 1.0 | 19.46 | `tl::cp_async_wait<1>();` |
| load.cp_async#2*19 (est.: sampled iterations x19) | 2.70 | 7.5% | 1.0 | 3.04 | `for (int i_3 = 0; i_3 < 2; ++i_3) \| for (int i_3 = 0; i_3 < 8; ++i_3)` |
| wait.sync*19 (est.: sampled iterations x19) | 1.18 | 3.3% | 1.0 | 1.22 | `__syncthreads();` |
| load.cp_async | 0.39 | 1.1% | 1.0 | 0.42 | `for (int i_1 = 0; i_1 < 2; ++i_1) \| for (int i_1 = 0; i_1 < 4; ++i_1) \| for (int i_1 = 0; ` |
| compute.mma#2 | 0.18 | 0.5% | 1.0 | 0.19 | `{` |
| wait.cp_async#2 | 0.13 | 0.4% | 1.0 | 0.32 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| wait.sync#3 | 0.11 | 0.3% | 1.0 | 0.29 | `__syncthreads();` |
| store.global | 0.05 | 0.1% | 1.0 | 0.06 | `for (int i_10 = 0; i_10 < 16; ++i_10) \| for (int i_10 = 0; i_10 < 8; ++i_10) \| for (int i_` |

### mhc_project_streams_kernel grid (40, 1, 1) block (128, 1, 1)

306 launches, 160 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 17.78 us, warp 15.20 us, nsys graphs 16.30 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 7.20 | 47.4% | 4.70 | 27.9% |
| load | 2.85 | 18.8% | 5.21 | 30.9% |
| compute | 2.78 | 18.3% | 2.56 | 15.2% |
| wait | 1.90 | 12.5% | 3.85 | 22.8% |
| store | 0.46 | 3.1% | 0.52 | 3.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*4 (est.: sampled iterations x4) | 2.65 | 17.4% | 1.0 | 2.69 | `{` |
| load.global#2*4 (est.: sampled iterations x4) | 2.62 | 17.2% | 1.0 | 7.74 | `for (int i_1 = 0; i_1 < 6; ++i_1)` |
| wait.sync*4 (est.: sampled iterations x4) | 1.72 | 11.3% | 1.0 | 5.44 | `__syncthreads();` |
| store.global*4 (est.: sampled iterations x4) | 0.25 | 1.6% | 1.0 | 0.38 | `if (((((int)blockIdx.y) * 16) + (((int)threadIdx.x) >> 3)) < M)` |
| load.global | 0.23 | 1.5% | 1.0 | 0.26 | `for (int k = 0; k < 4; ++k)` |
| wait.sync#2*4 (est.: sampled iterations x4) | 0.18 | 1.2% | 1.0 | 0.26 | `__syncthreads();` |
| store.global#4 | 0.09 | 0.6% | 1.0 | 0.13 | `if (((((int)threadIdx.x) % 8) == 0) && (((((int)blockIdx.y) * 16) + (((int)threadIdx.x) >>` |
| compute.reduce | 0.07 | 0.4% | 1.0 | 0.10 | `red[0] = (red[0] + __shfl_xor_sync((uint)4294967295, red[0], 4, 32));` |
| compute.reduce#2 | 0.07 | 0.4% | 1.0 | 0.10 | `red[0] = (red[0] + __shfl_xor_sync((uint)4294967295, red[0], 4, 32));` |
| store.global#2 | 0.06 | 0.4% | 1.0 | 0.10 | `if (((((int)blockIdx.y) * 16) + (((int)threadIdx.x) >> 3)) < M) \| if (((((int)threadIdx.x)` |
| store.global#3 | 0.06 | 0.4% | 1.0 | 0.06 | `if (((((int)threadIdx.x) % 8) == 0) && (((((int)blockIdx.y) * 16) + (((int)threadIdx.x) >>` |
| compute.mma | 0.00 | 0.0% | 0.0 | 0.00 | `{` |
| load.global#2 | 0.00 | 0.0% | 0.0 | 0.00 | `for (int i_1 = 0; i_1 < 6; ++i_1)` |
| store.global | 0.00 | 0.0% | 0.0 | 0.00 | `for (int n = 0; n < 4; ++n) \| if (((((int)blockIdx.y) * 16) + (((int)threadIdx.x) >> 3)) <` |
| wait.sync | 0.00 | 0.0% | 0.0 | 0.00 | `__syncthreads();` |

### gemm_mxfp8_kernel grid (24, 1, 1) block (128, 1, 1)

153 launches, 96 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 31.97 us, warp 31.43 us, nsys graphs 36.86 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 13.18 | 41.9% | 10.20 | 29.0% |
| wait | 7.81 | 24.8% | 12.18 | 34.6% |
| compute | 6.02 | 19.2% | 6.00 | 17.0% |
| load | 4.37 | 13.9% | 6.76 | 19.2% |
| store | 0.05 | 0.2% | 0.06 | 0.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*38 (est.: sampled iterations x38) | 5.78 | 18.4% | 1.0 | 6.08 | `{` |
| load.cp_async#2*38 (est.: sampled iterations x38) | 4.18 | 13.3% | 1.0 | 4.86 | `for (int i_5 = 0; i_5 < 2; ++i_5) \| for (int i_5 = 0; i_5 < 4; ++i_5) \| tl::cp_async_gs_co` |
| wait.sync#2*38 (est.: sampled iterations x38) | 2.96 | 9.4% | 1.0 | 3.65 | `__syncthreads();` |
| wait.sync*38 (est.: sampled iterations x38) | 2.36 | 7.5% | 1.0 | 2.43 | `__syncthreads();` |
| wait.cp_async*38 (est.: sampled iterations x38) | 2.28 | 7.3% | 1.0 | 2.43 | `tl::cp_async_wait<2>();` |
| load.cp_async | 0.19 | 0.6% | 1.0 | 0.26 | `for (int i_1 = 0; i_1 < 2; ++i_1) \| for (int i_1 = 0; i_1 < 4; ++i_1) \| for (int i_1 = 0; ` |
| compute.mma#3 | 0.13 | 0.4% | 1.0 | 0.13 | `{` |
| compute.mma#2 | 0.12 | 0.4% | 1.0 | 0.13 | `{` |
| wait.cp_async#3 | 0.06 | 0.2% | 1.0 | 0.06 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>();` |
| store.global | 0.05 | 0.2% | 1.0 | 0.06 | `for (int i_10 = 0; i_10 < 16; ++i_10) \| for (int i_10 = 0; i_10 < 8; ++i_10) \| for (int i_` |
| wait.sync#4 | 0.05 | 0.2% | 1.0 | 0.06 | `__syncthreads();` |
| wait.cp_async#2 | 0.05 | 0.2% | 1.0 | 0.06 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| wait.sync#3 | 0.05 | 0.2% | 1.0 | 0.06 | `__syncthreads();` |

### gemm_fp8w_bf16_partials_kernel grid (48, 4, 1) block (128, 1, 1)

153 launches, 768 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 24.61 us, warp 17.92 us, nsys graphs 41.50 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 7.70 | 43.0% | 11.22 | 35.8% |
| load | 3.88 | 21.7% | 5.41 | 17.2% |
| unattributed | 3.60 | 20.1% | 11.82 | 37.7% |
| compute | 2.67 | 14.9% | 2.87 | 9.1% |
| store | 0.07 | 0.4% | 0.06 | 0.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*14 (est.: sampled iterations x14) | 2.41 | 13.4% | 1.0 | 2.69 | `{` |
| wait.sync#2*14 (est.: sampled iterations x14) | 1.32 | 7.3% | 1.0 | 1.79 | `__syncthreads();` |
| load.global*14 (est.: sampled iterations x14) | 1.18 | 6.6% | 1.0 | 1.79 | `if (((int)threadIdx.x) < 4)` |
| load.cp_async#2*14 (est.: sampled iterations x14) | 1.09 | 6.1% | 1.0 | 1.34 | `for (int i_4 = 0; i_4 < 2; ++i_4) \| for (int i_6 = 0; i_6 < 2; ++i_6)` |
| load.cp_async#3*14 (est.: sampled iterations x14) | 1.06 | 5.9% | 1.0 | 1.34 | `for (int i_10 = 0; i_10 < 4; ++i_10) \| for (int i_10 = 0; i_10 < 6; ++i_10) \| for (int i_1` |
| wait.sync#4*14 (est.: sampled iterations x14) | 1.04 | 5.8% | 1.0 | 1.34 | `__syncthreads();` |
| wait.sync#3*14 (est.: sampled iterations x14) | 1.01 | 5.6% | 1.0 | 1.34 | `__syncthreads();` |
| wait.sync#5*14 (est.: sampled iterations x14) | 1.01 | 5.6% | 1.0 | 1.34 | `__syncthreads();` |
| wait.sync*14 (est.: sampled iterations x14) | 1.01 | 5.6% | 1.0 | 1.34 | `__syncthreads();` |
| wait.cp_async*14 (est.: sampled iterations x14) | 0.90 | 5.0% | 1.0 | 1.34 | `tl::cp_async_wait<3>();` |
| wait.cp_async#2*14 (est.: sampled iterations x14) | 0.87 | 4.8% | 1.0 | 1.34 | `tl::cp_async_wait<3>();` |
| load.cp_async | 0.43 | 2.4% | 1.0 | 0.70 | `for (int i_1 = 0; i_1 < 2; ++i_1)` |
| compute.mma#2 | 0.14 | 0.8% | 1.0 | 0.16 | `{` |
| compute.mma#3 | 0.13 | 0.7% | 1.0 | 0.16 | `{` |
| wait.sync#7 | 0.07 | 0.4% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#10 | 0.07 | 0.4% | 1.0 | 0.10 | `__syncthreads();` |
| load.global#2 | 0.07 | 0.4% | 1.0 | 0.10 | `if (((int)threadIdx.x) < 4)` |
| store.global | 0.07 | 0.4% | 1.0 | 0.10 | `for (int i_13 = 0; i_13 < 8; ++i_13) \| for (int i_19 = 0; i_19 < 16; ++i_19) \| for (int i_` |
| load.global#3 | 0.06 | 0.3% | 1.0 | 0.10 | `if (((int)threadIdx.x) < 4)` |
| wait.cp_async#6 | 0.06 | 0.3% | 1.0 | 0.06 | `tl::cp_async_wait<0>();` |
| wait.sync#8 | 0.06 | 0.3% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#9 | 0.06 | 0.3% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#6 | 0.05 | 0.3% | 1.0 | 0.06 | `__syncthreads();` |
| wait.cp_async#3 | 0.05 | 0.3% | 1.0 | 0.06 | `tl::cp_async_wait<3>();` |
| wait.sync#11 | 0.05 | 0.3% | 1.0 | 0.06 | `__syncthreads();` |
| wait.cp_async#5 | 0.05 | 0.3% | 1.0 | 0.06 | `tl::cp_async_wait<1>();` |
| wait.cp_async#4 | 0.05 | 0.3% | 1.0 | 0.06 | `tl::cp_async_wait<2>();` |

### gemm_bf16_partials_kernel grid (6, 8, 1) block (256, 1, 1)

102 launches, 384 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 24.35 us, warp 8.82 us, nsys graphs 19.89 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 6.36 | 72.1% | 1.85 | 35.2% |
| wait | 1.35 | 15.3% | 1.99 | 37.9% |
| compute | 0.68 | 7.7% | 1.36 | 25.7% |
| load | 0.40 | 4.5% | 0.00 | 0.0% |
| store | 0.04 | 0.4% | 0.06 | 1.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*10 (est.: sampled iterations x10) | 0.68 | 7.7% | 0.5 | 1.60 | `{` |
| wait.mbarrier*10 (est.: sampled iterations x10) | 0.65 | 7.4% | 0.5 | 1.60 | `mbarrier[((kb % 3) + 3)].wait((((kb % 6) / 3) ^ 1));` |
| wait.mbarrier#2*10 (est.: sampled iterations x10) | 0.60 | 6.8% | 0.5 | 2.24 | `mbarrier[(kb_1 % 3)].wait(((kb_1 % 6) / 3));` |
| load.tma*10 (est.: sampled iterations x10) | 0.40 | 4.5% | 0.5 | 1.28 | `if (tl::tl_shuffle_elect<128>())` |
| wait.sync | 0.10 | 1.1% | 1.0 | 0.13 | `__syncthreads();` |
| store.global | 0.04 | 0.4% | 0.5 | 0.10 | `for (int i_1 = 0; i_1 < 4; ++i_1) \| for (int i_2 = 0; i_2 < 8; ++i_2) \| for (int i_4 = 0; ` |

### mhc_finalize_kernel grid (5, 1, 1) block (128, 1, 1)

192 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 10.96 us, warp 9.63 us, nsys graphs 11.41 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 5.36 | 55.7% | 4.93 | 43.5% |
| store | 1.52 | 15.8% | 2.79 | 24.7% |
| load | 1.07 | 11.1% | 1.59 | 14.0% |
| compute | 1.00 | 10.4% | 1.20 | 10.6% |
| wait | 0.68 | 7.0% | 0.82 | 7.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync | 0.68 | 7.0% | 1.0 | 0.91 | `__syncthreads();` |
| store.global | 0.62 | 6.5% | 0.2 | 2.51 | `if (((int)threadIdx.x) < 4)` |
| compute.reduce | 0.57 | 6.0% | 0.2 | 2.30 | `row_max[0] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<32>>::run(row_max[0]);` |
| store.global#4*7 (est.: sampled iterations x7) | 0.56 | 5.8% | 0.8 | 0.90 | `if ((((int)threadIdx.x) % 4) == 0)` |
| compute.reduce#2*7 (est.: sampled iterations x7) | 0.43 | 4.4% | 0.8 | 0.67 | `amax[0] = max(amax[0], __shfl_xor_sync((uint)4294967295, amax[0], 1, 32));` |
| load.global | 0.39 | 4.0% | 1.0 | 1.07 | `if (((int)threadIdx.x) < 32)` |
| load.global#4*7 (est.: sampled iterations x7) | 0.37 | 3.8% | 0.8 | 0.67 | `*(uint4*)(yv + 0) = *(uint4*)(C + ((((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((i` |
| store.global#3*7 (est.: sampled iterations x7) | 0.32 | 3.4% | 0.8 | 0.45 | `*(uint4*)(y + ((((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((int64_t)step) * (int6` |
| load.global#3 | 0.27 | 2.8% | 0.8 | 0.38 | `for (int i_split_2 = 0; i_split_2 < 40; ++i_split_2)` |
| load.global#2 | 0.04 | 0.5% | 0.2 | 0.19 | `cm[0] = ((mixes_shared[((((int)threadIdx.x) & 15) + 8)] * scale[2]) + base[((((int)threadI` |
| store.global#2 | 0.01 | 0.1% | 0.2 | 0.06 | `if ((((int)threadIdx.x) >> 4) == 0)` |

### gemm_bf16_splitk_kernel grid (16, 5, 1) block (128, 1, 1)

34 launches, 320 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 57.90 us, warp 51.66 us, nsys graphs 54.14 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 35.74 | 69.2% | 55.98 | 98.8% |
| unattributed | 11.08 | 21.4% | 0.00 | 0.0% |
| load | 2.77 | 5.4% | 3.71 | 6.5% |
| compute | 1.94 | 3.8% | 2.12 | 3.7% |
| store | 0.13 | 0.2% | 0.14 | 0.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.cp_async*13 (est.: sampled iterations x13) | 17.42 | 33.7% | 0.5 | 33.70 | `tl::cp_async_wait<2>();` |
| wait.cp_async*12 (est.: sampled iterations x12) | 5.16 | 10.0% | 0.5 | 14.78 | `tl::cp_async_wait<3>();` |
| wait.cp_async#2 | 2.72 | 5.3% | 1.0 | 5.01 | `tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>(); \| tl::cp_async_wait<3>();` |
| wait.sync*13 (est.: sampled iterations x13) | 2.42 | 4.7% | 0.5 | 5.82 | `__syncthreads();` |
| wait.sync*12 (est.: sampled iterations x12) | 2.21 | 4.3% | 0.5 | 5.76 | `__syncthreads();` |
| wait.cp_async#4 | 1.71 | 3.3% | 1.0 | 4.29 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>();` |
| load.cp_async | 1.50 | 2.9% | 1.0 | 3.74 | `for (int i_1 = 0; i_1 < 2; ++i_1) \| tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)A_sha` |
| wait.cp_async#3 | 0.84 | 1.6% | 1.0 | 2.75 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| wait.sync#5 | 0.84 | 1.6% | 1.0 | 2.29 | `__syncthreads();` |
| compute.mma*12 (est.: sampled iterations x12) | 0.80 | 1.6% | 0.5 | 0.96 | `{` |
| compute.mma*13 (est.: sampled iterations x13) | 0.78 | 1.5% | 0.5 | 0.83 | `{` |
| wait.sync#3 | 0.62 | 1.2% | 1.0 | 1.81 | `__syncthreads();` |
| load.cp_async#2*12 (est.: sampled iterations x12) | 0.52 | 1.0% | 0.5 | 0.58 | `tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)A_shared)[((((((kb & 3) * 1024) + ((((int` |
| wait.sync#4 | 0.47 | 0.9% | 1.0 | 1.47 | `__syncthreads();` |
| load.cp_async#2*13 (est.: sampled iterations x13) | 0.43 | 0.8% | 0.5 | 0.62 | `for (int i_10 = 0; i_10 < 2; ++i_10) \| for (int i_8 = 0; i_8 < 2; ++i_8) \| tl::cp_async_gs` |
| wait.sync#2*13 (est.: sampled iterations x13) | 0.36 | 0.7% | 0.5 | 0.42 | `__syncthreads();` |
| load.global | 0.33 | 0.6% | 1.0 | 2.32 | `if (((int)threadIdx.x) == 0)` |
| wait.sync#2*12 (est.: sampled iterations x12) | 0.30 | 0.6% | 0.5 | 0.38 | `__syncthreads();` |
| wait.sync#7 | 0.24 | 0.5% | 1.0 | 0.45 | `__syncthreads();` |
| wait.sync#8 | 0.17 | 0.3% | 0.5 | 0.22 | `__syncthreads();` |
| wait.cp_async#5 | 0.13 | 0.3% | 0.5 | 0.77 | `tl::cp_async_wait<0>();` |
| wait.sync#6 | 0.13 | 0.3% | 1.0 | 0.46 | `__syncthreads();` |
| compute.mma#2 | 0.11 | 0.2% | 1.0 | 0.13 | `{` |
| compute.mma#3 | 0.10 | 0.2% | 1.0 | 0.13 | `{` |
| compute.mma#4 | 0.10 | 0.2% | 1.0 | 0.13 | `{` |
| store.global#3 | 0.07 | 0.1% | 1.0 | 0.10 | `if (((int*)last)[0] == 1)` |
| store.global | 0.06 | 0.1% | 1.0 | 0.06 | `for (int i_10 = 0; i_10 < 4; ++i_10) \| for (int i_16 = 0; i_16 < 4; ++i_16) \| for (int i_1` |
| compute.mma#5 | 0.05 | 0.1% | 0.5 | 0.05 | `{` |

### sparse_mla_kernel grid (5, 1, 5) block (256, 1, 1)

64 launches, 200 warps; ranges, warp and last warp from the 5 recording CTAs' 40 warps; span 20.90 us, warp 18.51 us, nsys graphs 13.98 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed, with loops' unsampled iterations | 12.69 | 68.5% | 12.31 | 53.3% |
| wait | 2.00 | 10.8% | 4.20 | 18.2% |
| load | 1.68 | 9.1% | 3.98 | 17.3% |
| compute | 1.48 | 8.0% | 1.49 | 6.5% |
| store | 0.67 | 3.6% | 1.10 | 4.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync (sampled iterations) | 1.93 | 10.4% | 11.0 | 3.78 | `__syncthreads();` |
| load.global (sampled iterations) | 1.38 | 7.5% | 2.0 | 6.22 | `for (int c = 0; c < 4; ++c) \| for (int i = 0; i < 8; ++i) \| for (int i_1 = 0; i_1 < 8; ++i` |
| compute.mma (sampled iterations) | 1.22 | 6.6% | 2.0 | 1.28 | `for (int ki = 0; ki < 32; ++ki) \| for (int ki_1 = 0; ki_1 < 2; ++ki_1)` |
| store.global | 0.67 | 3.6% | 1.0 | 0.83 | `for (int i_10 = 0; i_10 < 32; ++i_10) \| for (int i_11 = 0; i_11 < 32; ++i_11) \| for (int i` |
| load.cp_async (sampled iterations) | 0.30 | 1.6% | 2.0 | 0.42 | `for (int k_1 = 0; k_1 < 5; ++k_1) \| for (int k_1 = 0; k_1 < 9; ++k_1) \| for (int k_3 = 0; ` |
| compute.reduce (sampled iterations) | 0.25 | 1.4% | 2.0 | 0.27 | `m_g[i_5] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(m_g[i_5], (&((` |
| wait.cp_async (sampled iterations) | 0.07 | 0.4% | 1.0 | 0.10 | `tl::cp_async_wait<0>();` |

### sparse_mla_kernel grid (2, 2, 10) block (128, 1, 1)

51 launches, 160 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 21.79 us, warp 19.48 us, nsys graphs 18.98 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed, with loops' unsampled iterations | 8.38 | 43.0% | 7.73 | 36.3% |
| compute | 4.72 | 24.2% | 4.72 | 22.2% |
| load | 3.31 | 17.0% | 5.34 | 25.1% |
| wait | 2.73 | 14.0% | 3.03 | 14.3% |
| store | 0.34 | 1.7% | 0.46 | 2.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 | 2.14 | 11.0% | 1.0 | 4.38 | `for (int i = 0; i < 8; ++i)` |
| compute.mma*4 (est.: sampled iterations x4) | 1.73 | 8.9% | 1.0 | 1.79 | `for (int ki = 0; ki < 32; ++ki) \| {` |
| compute.mma#2*4 (est.: sampled iterations x4) | 1.20 | 6.2% | 1.0 | 1.28 | `for (int ki_1 = 0; ki_1 < 2; ++ki_1) \| if (((((int)blockIdx.x) * 2) + (c_1 >> 1)) < 3)` |
| compute.reduce#2*8 (est.: sampled iterations x8) | 0.90 | 4.6% | 1.0 | 1.02 | `row_sum[i_7] = tl::AllReduce<tl::SumOp, 128, 32, 0, tl::NamedBarrier<128>>::run(row_sum[i_` |
| compute.reduce*8 (est.: sampled iterations x8) | 0.90 | 4.6% | 1.0 | 1.02 | `m_g[i_4] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(m_g[i_4], (&((` |
| load.cp_async#2*4 (est.: sampled iterations x4) | 0.70 | 3.6% | 1.0 | 0.77 | `if (((((int)blockIdx.x) * 2) + (c_1 >> 1)) < 3) \| if ((c_1 < 3) & (c_1 < 3)) \| if (c < 3)` |
| wait.sync#10*8 (est.: sampled iterations x8) | 0.35 | 1.8% | 1.0 | 0.51 | `__syncthreads();` |
| store.global | 0.34 | 1.7% | 1.0 | 0.48 | `for (int i_10 = 0; i_10 < 32; ++i_10) \| for (int i_11 = 0; i_11 < 32; ++i_11) \| for (int i` |
| load.cp_async (sampled iterations) | 0.34 | 1.7% | 1.0 | 0.38 | `for (int k_1 = 0; k_1 < 5; ++k_1) \| for (int k_1 = 0; k_1 < 9; ++k_1) \| for (int k_3 = 0; ` |
| wait.sync#8*8 (est.: sampled iterations x8) | 0.32 | 1.6% | 1.0 | 0.51 | `__syncthreads();` |
| wait.cp_async*4 (est.: sampled iterations x4) | 0.26 | 1.4% | 1.0 | 0.38 | `tl::cp_async_wait<0>();` |
| wait.sync#11*4 (est.: sampled iterations x4) | 0.26 | 1.3% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#5*4 (est.: sampled iterations x4) | 0.26 | 1.3% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#4*4 (est.: sampled iterations x4) | 0.25 | 1.3% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#7*4 (est.: sampled iterations x4) | 0.25 | 1.3% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#9*4 (est.: sampled iterations x4) | 0.25 | 1.3% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#6*4 (est.: sampled iterations x4) | 0.24 | 1.2% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync (sampled iterations) | 0.18 | 0.9% | 1.0 | 0.26 | `__syncthreads();` |
| load.global (sampled iterations) | 0.14 | 0.7% | 1.0 | 0.29 | `for (int c = 0; c < 4; ++c) \| for (int i = 0; i < 8; ++i) \| for (int i_1 = 0; i_1 < 8; ++i` |
| wait.sync#3 | 0.07 | 0.3% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#2 | 0.05 | 0.2% | 1.0 | 0.06 | `__syncthreads();` |

### mhc_finalize_kernel grid (10, 1, 1) block (128, 1, 1)

102 launches, 40 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 10.51 us, warp 9.55 us, nsys graphs 10.35 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 5.43 | 56.9% | 4.20 | 39.9% |
| store | 1.47 | 15.4% | 2.83 | 26.9% |
| load | 1.00 | 10.5% | 1.50 | 14.3% |
| compute | 1.00 | 10.4% | 1.44 | 13.7% |
| wait | 0.65 | 6.8% | 0.56 | 5.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync | 0.65 | 6.8% | 1.0 | 0.86 | `__syncthreads();` |
| store.global | 0.58 | 6.0% | 0.2 | 2.34 | `if (((int)threadIdx.x) < 4)` |
| compute.reduce | 0.57 | 6.0% | 0.2 | 2.30 | `row_max[0] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<32>>::run(row_max[0]);` |
| store.global#4*7 (est.: sampled iterations x7) | 0.56 | 5.9% | 0.8 | 0.90 | `if ((((int)threadIdx.x) % 4) == 0)` |
| compute.reduce#2*7 (est.: sampled iterations x7) | 0.43 | 4.5% | 0.8 | 0.67 | `amax[0] = max(amax[0], __shfl_xor_sync((uint)4294967295, amax[0], 1, 32));` |
| load.global#4*7 (est.: sampled iterations x7) | 0.37 | 3.9% | 0.8 | 0.67 | `*(uint4*)(yv + 0) = *(uint4*)(C + ((((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((i` |
| load.global | 0.34 | 3.6% | 1.0 | 1.04 | `if (((int)threadIdx.x) < 32)` |
| store.global#3*7 (est.: sampled iterations x7) | 0.32 | 3.4% | 0.8 | 0.56 | `*(uint4*)(y + ((((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((int64_t)step) * (int6` |
| load.global#3 | 0.24 | 2.6% | 0.8 | 0.35 | `for (int i_split_2 = 0; i_split_2 < 40; ++i_split_2)` |
| load.global#2 | 0.04 | 0.5% | 0.2 | 0.19 | `cm[0] = ((mixes_shared[((((int)threadIdx.x) & 15) + 8)] * scale[2]) + base[((((int)threadI` |
| store.global#2 | 0.01 | 0.1% | 0.2 | 0.06 | `if ((((int)threadIdx.x) >> 4) == 0)` |

### gemm_mxfp8_kernel grid (64, 1, 1) block (128, 1, 1)

34 launches, 256 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 23.36 us, warp 14.97 us, nsys graphs 23.30 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 9.99 | 66.7% | 3.25 | 39.0% |
| unattributed | 3.31 | 22.1% | 2.73 | 32.7% |
| compute | 0.89 | 5.9% | 0.88 | 10.6% |
| load | 0.73 | 4.9% | 1.42 | 17.1% |
| store | 0.06 | 0.4% | 0.06 | 0.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.cp_async*2 (est.: sampled iterations x2) | 2.99 | 20.0% | 1.0 | 6.24 | `tl::cp_async_wait<3>();` |
| wait.cp_async#3 | 1.60 | 10.7% | 1.0 | 3.58 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>();` |
| wait.cp_async#2 | 1.31 | 8.8% | 1.0 | 3.07 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| wait.cp_async#4 | 1.26 | 8.4% | 1.0 | 2.67 | `tl::cp_async_wait<0>();` |
| wait.sync#2*2 (est.: sampled iterations x2) | 1.19 | 7.9% | 1.0 | 4.29 | `__syncthreads();` |
| wait.sync#4 | 0.61 | 4.1% | 1.0 | 2.14 | `__syncthreads();` |
| wait.sync#3 | 0.55 | 3.7% | 1.0 | 1.97 | `__syncthreads();` |
| load.cp_async | 0.41 | 2.7% | 1.0 | 0.54 | `for (int i_1 = 0; i_1 < 2; ++i_1) \| for (int i_1 = 0; i_1 < 4; ++i_1) \| for (int i_1 = 0; ` |
| wait.sync#5 | 0.38 | 2.5% | 1.0 | 1.33 | `__syncthreads();` |
| compute.mma*2 (est.: sampled iterations x2) | 0.36 | 2.4% | 1.0 | 0.38 | `{` |
| load.cp_async#2*2 (est.: sampled iterations x2) | 0.32 | 2.1% | 1.0 | 0.86 | `for (int i_7 = 0; i_7 < 2; ++i_7)` |
| compute.mma#2 | 0.18 | 1.2% | 1.0 | 0.19 | `{` |
| compute.mma#3 | 0.18 | 1.2% | 1.0 | 0.19 | `{` |
| compute.mma#4 | 0.18 | 1.2% | 1.0 | 0.19 | `{` |
| wait.sync*2 (est.: sampled iterations x2) | 0.09 | 0.6% | 1.0 | 0.13 | `__syncthreads();` |
| store.global | 0.06 | 0.4% | 1.0 | 0.06 | `for (int i_10 = 0; i_10 < 16; ++i_10) \| for (int i_10 = 0; i_10 < 8; ++i_10) \| for (int i_` |

### moe_dispatch_kernel grid (1, 1, 1) block (512, 1, 1)

153 launches, 16 warps; ranges, warp and last warp from the 1 recording CTAs' 16 warps; span 4.86 us, warp 4.62 us, nsys graphs 2.82 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 2.58 | 55.9% | 1.76 | 33.2% |
| unattributed, with loops' unsampled iterations | 1.36 | 29.3% | 1.91 | 36.1% |
| store | 0.60 | 13.1% | 1.53 | 28.8% |
| load | 0.08 | 1.7% | 0.10 | 1.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync#7*9 (est.: sampled iterations x9) | 0.67 | 14.4% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#6*9 (est.: sampled iterations x9) | 0.63 | 13.6% | 1.0 | 0.86 | `__syncthreads();` |
| store.global (sampled iterations) | 0.53 | 11.4% | 1.0 | 0.96 | `ChunkCount[((((int64_t)c) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = count; \| Chu` |
| wait.sync#8 | 0.51 | 10.9% | 1.0 | 0.83 | `__syncthreads();` |
| wait.sync#2 | 0.39 | 8.5% | 1.0 | 0.42 | `__syncthreads();` |
| wait.sync#3 | 0.13 | 2.9% | 1.0 | 0.32 | `__syncthreads();` |
| load.global (sampled iterations) | 0.08 | 1.7% | 1.0 | 0.10 | `expert = ((int)TopkIdx[((((int64_t)((int)blockIdx.x)) * (int64_t)512) + ((int64_t)((int)th` |
| store.global#2 | 0.08 | 1.7% | 1.0 | 0.29 | `RoutePos[((((int64_t)((int)blockIdx.x)) * (int64_t)512) + ((int64_t)((int)threadIdx.x)))] ` |
| wait.sync | 0.07 | 1.5% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#4 | 0.06 | 1.3% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#5 | 0.06 | 1.3% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#9 | 0.06 | 1.3% | 1.0 | 0.06 | `__syncthreads();` |

### metadata_chunk_kernel grid (10, 1, 1) block (192, 1, 1)

51 launches, 60 warps; ranges, warp and last warp from the 5 recording CTAs' 30 warps; span 15.74 us, warp 10.17 us, nsys graphs 12.96 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 6.05 | 59.5% | 12.02 | 76.5% |
| unattributed | 4.05 | 39.8% | 3.55 | 22.6% |
| store | 0.07 | 0.7% | 0.13 | 0.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 5.97 | 58.7% | 1.0 | 9.28 | `if ((((offset + ((int)blockIdx.x)) < src_rows) && (0 <= (offset + ((int)blockIdx.x)))))` |
| load.global#2 | 0.08 | 0.8% | 0.2 | 0.74 | `if (((0 <= req) & ((int64_t)0 <= pos)))` |
| store.global | 0.05 | 0.5% | 1.0 | 0.10 | `swa[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = con` |
| store.global#3 | 0.01 | 0.1% | 0.2 | 0.06 | `top_lengths[((int64_t)((int)blockIdx.x))] = condval_12; \| top_lengths[((int64_t)((int)bloc` |
| store.global#2 | 0.01 | 0.1% | 0.2 | 0.06 | `lengths[((int64_t)((int)blockIdx.x))] = ((int)max(info[1], (int64_t)0));` |

### sparse_mla_kernel grid (1, 2, 5) block (128, 1, 1)

32 launches, 40 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 25.31 us, warp 25.25 us, nsys graphs 19.79 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed, with loops' unsampled iterations | 9.55 | 37.8% | 9.14 | 36.7% |
| load | 5.90 | 23.4% | 7.93 | 31.9% |
| wait | 4.71 | 18.7% | 2.62 | 10.5% |
| compute | 4.66 | 18.4% | 4.69 | 18.8% |
| store | 0.43 | 1.7% | 0.51 | 2.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 | 4.10 | 16.2% | 1.0 | 4.37 | `for (int i = 0; i < 8; ++i)` |
| wait.sync (sampled iterations) | 2.15 | 8.5% | 1.0 | 2.86 | `__syncthreads();` |
| compute.mma*4 (est.: sampled iterations x4) | 1.72 | 6.8% | 1.0 | 1.79 | `for (int ki = 0; ki < 32; ++ki) \| {` |
| compute.mma#2*4 (est.: sampled iterations x4) | 1.16 | 4.6% | 1.0 | 1.28 | `for (int ki_1 = 0; ki_1 < 2; ++ki_1) \| if (((((int)blockIdx.x) * 2) + (c_1 >> 1)) < 3)` |
| compute.reduce*8 (est.: sampled iterations x8) | 0.92 | 3.6% | 1.0 | 1.02 | `m_g[i_4] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(m_g[i_4], (&((` |
| compute.reduce#2*8 (est.: sampled iterations x8) | 0.84 | 3.3% | 1.0 | 1.02 | `row_sum[i_7] = tl::AllReduce<tl::SumOp, 128, 32, 0, tl::NamedBarrier<128>>::run(row_sum[i_` |
| load.global (sampled iterations) | 0.79 | 3.1% | 1.0 | 2.99 | `for (int c = 0; c < 4; ++c) \| for (int i = 0; i < 8; ++i) \| for (int i_1 = 0; i_1 < 8; ++i` |
| load.cp_async#2*4 (est.: sampled iterations x4) | 0.67 | 2.6% | 1.0 | 0.77 | `if (((((int)blockIdx.x) * 2) + (c_1 >> 1)) < 3) \| if ((c_1 < 3) & (c_1 < 3)) \| if (c < 3)` |
| store.global | 0.43 | 1.7% | 1.0 | 0.51 | `for (int i_10 = 0; i_10 < 32; ++i_10) \| for (int i_11 = 0; i_11 < 32; ++i_11) \| for (int i` |
| load.cp_async (sampled iterations) | 0.35 | 1.4% | 1.0 | 0.38 | `for (int k_1 = 0; k_1 < 5; ++k_1) \| for (int k_1 = 0; k_1 < 9; ++k_1) \| for (int k_3 = 0; ` |
| wait.sync#10*8 (est.: sampled iterations x8) | 0.33 | 1.3% | 1.0 | 0.51 | `__syncthreads();` |
| wait.sync#8*8 (est.: sampled iterations x8) | 0.33 | 1.3% | 1.0 | 0.51 | `__syncthreads();` |
| wait.sync#5*4 (est.: sampled iterations x4) | 0.27 | 1.1% | 1.0 | 0.38 | `__syncthreads();` |
| wait.cp_async*4 (est.: sampled iterations x4) | 0.26 | 1.0% | 1.0 | 0.32 | `tl::cp_async_wait<0>();` |
| wait.sync#11*4 (est.: sampled iterations x4) | 0.26 | 1.0% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#4*4 (est.: sampled iterations x4) | 0.26 | 1.0% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#6*4 (est.: sampled iterations x4) | 0.26 | 1.0% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#7*4 (est.: sampled iterations x4) | 0.26 | 1.0% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#9*4 (est.: sampled iterations x4) | 0.25 | 1.0% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#3 | 0.06 | 0.2% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#2 | 0.05 | 0.2% | 1.0 | 0.06 | `__syncthreads();` |

### gemm_bf16_partials_kernel grid (2, 20, 1) block (256, 1, 1)

51 launches, 320 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 14.11 us, warp 7.97 us, nsys graphs 13.10 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 6.34 | 79.5% | 1.41 | 58.6% |
| wait | 0.87 | 10.9% | 0.47 | 19.6% |
| load | 0.49 | 6.1% | 0.00 | 0.0% |
| compute | 0.22 | 2.8% | 0.45 | 18.7% |
| store | 0.05 | 0.7% | 0.08 | 3.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.tma*4 (est.: sampled iterations x4) | 0.49 | 6.1% | 0.5 | 1.28 | `if (tl::tl_shuffle_elect<128>())` |
| wait.mbarrier#2*4 (est.: sampled iterations x4) | 0.47 | 5.9% | 0.5 | 1.28 | `mbarrier[(kb_1 % 3)].wait((kb_1 / 3));` |
| wait.sync | 0.27 | 3.4% | 1.0 | 0.35 | `__syncthreads();` |
| compute.mma*4 (est.: sampled iterations x4) | 0.22 | 2.8% | 0.5 | 0.51 | `{` |
| wait.mbarrier*4 (est.: sampled iterations x4) | 0.13 | 1.6% | 0.5 | 0.26 | `mbarrier[((kb % 3) + 3)].wait(((kb / 3) ^ 1));` |
| store.global | 0.05 | 0.7% | 0.5 | 0.13 | `for (int i_1 = 0; i_1 < 4; ++i_1) \| for (int i_2 = 0; i_2 < 8; ++i_2) \| for (int i_4 = 0; ` |

### moe_grouped_fp8_fp4_gemm_kernel grid (12, 36, 1) block (128, 1, 1)

6 launches, 1728 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 116.40 us, warp 18.54 us, nsys graphs 3.46 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 7.84 | 42.3% | 0.05 | 15.8% |
| unattributed | 5.40 | 29.1% | 0.20 | 66.7% |
| compute | 3.44 | 18.6% | 0.00 | 0.0% |
| load | 1.79 | 9.7% | 0.05 | 17.5% |
| store | 0.07 | 0.4% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.mbarrier*40 (est.: sampled iterations x40) | 4.64 | 25.0% | 0.2 | 18.56 | `loaded[(ko % 3)].wait(((ko % 6) / 3));` |
| compute.mma*40 (est.: sampled iterations x40) | 3.44 | 18.6% | 0.2 | 14.08 | `{` |
| load#2*40 (est.: sampled iterations x40) | 1.48 | 8.0% | 0.2 | 7.68 | `if (ko < 38)` |
| wait.sync#5*40 (est.: sampled iterations x40) | 1.32 | 7.1% | 0.2 | 6.40 | `__syncthreads();` |
| wait.sync#4*40 (est.: sampled iterations x40) | 0.64 | 3.5% | 0.2 | 2.56 | `__syncthreads();` |
| wait.sync#6*40 (est.: sampled iterations x40) | 0.64 | 3.5% | 0.2 | 2.56 | `__syncthreads();` |
| wait.cp_async*40 (est.: sampled iterations x40) | 0.52 | 2.8% | 0.2 | 2.56 | `tl::cp_async_wait<2>();` |
| load.global | 0.12 | 0.6% | 1.0 | 0.29 | `int expert = TileExpert[((int64_t)((int)blockIdx.y))];` |
| load.global#3 | 0.12 | 0.6% | 0.2 | 0.48 | `if (((expert < 128) && (0 <= expert))) \| if (((expert < 384) && (0 <= expert)))` |
| store.global | 0.07 | 0.4% | 0.2 | 0.35 | `for (int i_2 = 0; i_2 < 16; ++i_2) \| for (int i_4 = 0; i_4 < 32; ++i_4)` |
| load | 0.05 | 0.2% | 0.2 | 0.26 | `for (int s = 0; s < 2; ++s)` |
| wait.sync | 0.04 | 0.2% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#2 | 0.03 | 0.2% | 0.2 | 0.16 | `__syncthreads();` |
| load.global#2 | 0.03 | 0.1% | 0.2 | 0.19 | `if (((int)threadIdx.x) < 32) \| if (((int)threadIdx.x) < 64)` |
| wait.sync#3 | 0.01 | 0.0% | 0.2 | 0.03 | `__syncthreads();` |

### gemm_mxfp8_partials_kernel grid (16, 5, 1) block (128, 1, 1)

51 launches, 320 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 11.97 us, warp 11.25 us, nsys graphs 12.45 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 5.47 | 48.6% | 6.66 | 53.8% |
| wait | 4.65 | 41.3% | 3.13 | 25.3% |
| compute | 0.61 | 5.5% | 0.64 | 5.2% |
| load | 0.46 | 4.1% | 1.89 | 15.3% |
| store | 0.05 | 0.5% | 0.06 | 0.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.cp_async#3 | 1.72 | 15.3% | 1.0 | 3.33 | `tl::cp_async_wait<0>();` |
| wait.cp_async#2 | 1.02 | 9.1% | 1.0 | 2.14 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>();` |
| wait.sync#2*2 (est.: sampled iterations x2) | 0.65 | 5.8% | 1.0 | 2.43 | `__syncthreads();` |
| wait.sync#3 | 0.47 | 4.2% | 1.0 | 1.47 | `__syncthreads();` |
| wait.sync#4 | 0.35 | 3.1% | 1.0 | 1.25 | `__syncthreads();` |
| wait.cp_async*2 (est.: sampled iterations x2) | 0.34 | 3.1% | 1.0 | 2.43 | `tl::cp_async_wait<2>();` |
| compute.mma*2 (est.: sampled iterations x2) | 0.31 | 2.7% | 1.0 | 0.32 | `{` |
| load.cp_async | 0.23 | 2.1% | 1.0 | 0.32 | `for (int i = 0; i < 2; ++i) \| for (int i_1 = 0; i_1 < 4; ++i_1)` |
| load.cp_async#2*2 (est.: sampled iterations x2) | 0.23 | 2.0% | 1.0 | 0.26 | `for (int i_4 = 0; i_4 < 2; ++i_4)` |
| compute.mma#3 | 0.15 | 1.4% | 1.0 | 0.19 | `{` |
| compute.mma#2 | 0.15 | 1.4% | 1.0 | 0.16 | `{` |
| wait.sync*2 (est.: sampled iterations x2) | 0.09 | 0.8% | 1.0 | 0.13 | `__syncthreads();` |
| store.global | 0.05 | 0.5% | 1.0 | 0.06 | `for (int i_11 = 0; i_11 < 16; ++i_11) \| for (int i_13 = 0; i_13 < 8; ++i_13) \| for (int i_` |

### partial_argmax_kernel grid (2, 64, 1) block (128, 1, 1)

85 launches, 512 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 7.46 us, warp 6.34 us, nsys graphs 7.46 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 5.54 | 87.4% | 6.32 | 85.6% |
| wait | 0.33 | 5.3% | 0.20 | 2.7% |
| load | 0.27 | 4.3% | 0.62 | 8.3% |
| compute | 0.12 | 1.9% | 0.12 | 1.7% |
| store | 0.08 | 1.2% | 0.13 | 1.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync | 0.33 | 5.3% | 1.0 | 1.28 | `__syncthreads();` |
| load.global | 0.27 | 4.3% | 1.0 | 0.42 | `for (int j = 0; j < 16; ++j) \| for (int run = 0; run < 2; ++run)` |
| compute.reduce | 0.12 | 1.9% | 1.0 | 0.13 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| store.global | 0.08 | 1.2% | 1.0 | 0.13 | `if (((int)threadIdx.x) == 0)` |

### router_partials_kernel grid (3, 1, 1) block (64, 1, 1)

102 launches, 6 warps; ranges, warp and last warp from the 3 recording CTAs' 6 warps; span 4.99 us, warp 3.00 us, nsys graphs 4.51 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.19 | 39.6% | 2.43 | 39.9% |
| load | 1.00 | 33.2% | 2.03 | 33.3% |
| compute | 0.68 | 22.6% | 1.36 | 22.3% |
| store | 0.14 | 4.6% | 0.27 | 4.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.reduce | 0.68 | 22.6% | 0.5 | 1.38 | `for (int k = 0; k < 3; ++k) \| for (int k = 0; k < 6; ++k)` |
| load.global#4 | 0.49 | 16.2% | 0.5 | 1.02 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < image_len)) \| if ((((((int)bl` |
| load.global | 0.41 | 13.5% | 0.8 | 0.51 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < mask_len))` |
| load.global#2 | 0.05 | 1.8% | 0.8 | 0.06 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < mask_len))` |
| load.global#3 | 0.05 | 1.8% | 0.8 | 0.06 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < mask_len))` |
| store.global | 0.05 | 1.6% | 0.8 | 0.06 | `if (condval == (bool)0) \| if (condval == (bool)1)` |
| store.global#3 | 0.05 | 1.6% | 0.5 | 0.10 | `if ((((int)threadIdx.x) & 31) < 3) \| if ((((int)threadIdx.x) & 31) < 6)` |
| store.global#2 | 0.04 | 1.4% | 0.8 | 0.06 | `if (condval_1 == (bool)0) \| if (condval_1 == (bool)1)` |

### cache_writer_kernel grid (5, 1, 1) block (128, 1, 1)

128 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 5.39 us, warp 4.86 us, nsys graphs 6.32 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 2.95 | 60.7% | 2.26 | 47.1% |
| store | 1.54 | 31.8% | 2.02 | 41.9% |
| load | 0.29 | 6.0% | 0.44 | 9.2% |
| compute | 0.07 | 1.5% | 0.08 | 1.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 1.54 | 31.8% | 1.0 | 1.87 | `if (((int64_t)0 <= slot) & ((slot >> (int64_t)7) < ((int64_t)pages))) \| if (((int64_t)0 <=` |
| load.global | 0.23 | 4.7% | 1.0 | 0.26 | `for (int i = 0; i < 4; ++i)` |
| compute.reduce | 0.07 | 1.5% | 1.0 | 0.10 | `amax = max(amax, __shfl_xor_sync((uint)4294967295, amax, 1, 32));` |
| load.global#2 | 0.06 | 1.2% | 1.0 | 0.06 | `int64_t slot = slots[((int64_t)((int)blockIdx.x))];` |

### moe_swiglu_cast_kernel_kernel grid (16, 30, 1) block (96, 1, 1)

147 launches, 1440 warps; ranges, warp and last warp from the 4 recording CTAs' 12 warps; span 3.84 us, warp 1.24 us, nsys graphs 4.35 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.91 | 73.1% | 2.40 | 77.2% |
| load | 0.24 | 19.6% | 0.48 | 15.5% |
| compute | 0.05 | 3.9% | 0.12 | 3.7% |
| store | 0.04 | 3.4% | 0.11 | 3.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.24 | 19.6% | 1.0 | 0.45 | `for (int r = 0; r < 2; ++r)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.05 | 3.9% | 0.5 | 0.13 | `if (0 <= orders[r_2])` |
| store.global*2 (est.: sampled iterations x2) | 0.04 | 3.4% | 0.5 | 0.13 | `if (0 <= orders[r_2])` |

### rope_kernel grid (1, 1, 1) block (128, 1, 1)

272 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 1.47 us, warp 1.17 us, nsys graphs 1.28 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.01 | 86.3% | 1.65 | 78.1% |
| load | 0.14 | 12.3% | 0.40 | 19.2% |
| store | 0.02 | 1.4% | 0.06 | 2.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.10 | 8.2% | 1.0 | 0.10 | `if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < tokens)) \| int64_t position ` |
| load.global#2 | 0.05 | 4.1% | 0.5 | 0.10 | `if (((position < ((int64_t)positions_max)) && ((int64_t)0 <= position)))` |
| store.global | 0.02 | 1.4% | 0.5 | 0.03 | `*(uint4*)(x + ((((((int64_t)((int)blockIdx.x)) * (int64_t)12288) + ((((int64_t)((int)threa` |

### norm_forward_and_per_token_cast_kernel_kernel grid (1, 1, 1) block (128, 1, 1)

187 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 2.75 us, warp 2.75 us, nsys graphs 1.28 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 2.10 | 76.2% | 2.38 | 80.6% |
| load | 0.38 | 14.0% | 0.42 | 14.1% |
| compute | 0.10 | 3.8% | 0.11 | 3.6% |
| wait | 0.10 | 3.5% | 0.05 | 1.9% |
| unattributed | 0.07 | 2.6% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*4 (est.: sampled iterations x4) | 1.54 | 55.8% | 1.0 | 1.54 | `for (int j_2 = 0; j_2 < 2; ++j_2)` |
| store.global*4 (est.: sampled iterations x4) | 0.35 | 12.8% | 1.0 | 0.38 | `for (int j_1 = 0; j_1 < 4; ++j_1)` |
| load.global*4 (est.: sampled iterations x4) | 0.22 | 8.1% | 1.0 | 0.26 | `if (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 4)) < num_tokens)` |
| load.global#3*4 (est.: sampled iterations x4) | 0.13 | 4.7% | 1.0 | 0.13 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 128) + ((((int)threadIdx.x) & 15` |
| store.global#5*4 (est.: sampled iterations x4) | 0.13 | 4.7% | 0.8 | 0.13 | `*(uint4*)(out + ((((((int64_t)vi_1) * (int64_t)128) + ((((int64_t)((int)threadIdx.x)) & (i` |
| compute.reduce | 0.10 | 3.8% | 1.0 | 0.13 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| wait.sync | 0.10 | 3.5% | 1.0 | 0.26 | `__syncthreads();` |
| store.global#2 | 0.05 | 1.7% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| load.global#2 | 0.03 | 1.2% | 1.0 | 0.03 | `*(float2*)(row_partial + 0) = *(float2*)(sum_shared + ((((int)threadIdx.x) >> 6) * 2)); \| ` |
| store.global#3 | 0.03 | 1.2% | 0.8 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |
| load.global#3*2 (est.: sampled iterations x2) | 0.00 | 0.0% | 0.0 | 0.00 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 64) + ((((int)threadIdx.x) & 7) ` |
| load.global*2 (est.: sampled iterations x2) | 0.00 | 0.0% | 0.0 | 0.00 | `if (((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < num_tokens)` |
| store.global#4*2 (est.: sampled iterations x2) | 0.00 | 0.0% | 0.0 | 0.00 | `for (int j_2 = 0; j_2 < 2; ++j_2)` |
| store.global#5*2 (est.: sampled iterations x2) | 0.00 | 0.0% | 0.0 | 0.00 | `*(uint4*)(out + ((((((int64_t)vi_1) * (int64_t)64) + ((((int64_t)((int)threadIdx.x)) & (in` |
| store.global*2 (est.: sampled iterations x2) | 0.00 | 0.0% | 0.0 | 0.00 | `for (int j_1 = 0; j_1 < 4; ++j_1)` |

### metadata_decode_kernel grid (5, 2, 1) block (128, 1, 1)

64 launches, 40 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 4.99 us, warp 3.36 us, nsys graphs 3.17 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.72 | 51.3% | 1.83 | 24.7% |
| load | 1.54 | 45.9% | 4.87 | 65.7% |
| store | 0.09 | 2.7% | 0.71 | 9.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.96 | 28.4% | 0.6 | 1.92 | `if ((((offset + ((int)blockIdx.x)) < src_rows) && (0 <= (offset + ((int)blockIdx.x)))))` |
| load.global#3 | 0.56 | 16.6% | 0.4 | 1.44 | `if ((((offset + ((int)blockIdx.x)) < main_src_rows) && (0 <= (offset + ((int)blockIdx.x)))` |
| store.global#4 | 0.04 | 1.3% | 0.4 | 0.21 | `if (0 < live)` |
| store.global | 0.03 | 1.0% | 0.6 | 0.06 | `swa[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = con` |
| load.global#2 | 0.03 | 0.9% | 0.1 | 0.19 | `if (((0 <= req) & ((int64_t)0 <= pos)))` |
| store.global#2 | 0.01 | 0.2% | 0.1 | 0.06 | `lengths[((int64_t)((int)blockIdx.x))] = ((int)max(info[1], (int64_t)0));` |
| store.global#3 | 0.01 | 0.2% | 0.1 | 0.06 | `top_lengths[((int64_t)((int)blockIdx.x))] = condval_6; \| top_lengths[((int64_t)((int)block` |

### token_ranks_kernel grid (1, 1, 1) block (256, 1, 1)

51 launches, 8 warps; ranges, warp and last warp from the 1 recording CTAs' 8 warps; span 7.58 us, warp 7.58 us, nsys graphs 8.03 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 6.86 | 90.4% | 7.14 | 90.1% |
| compute | 0.36 | 4.7% | 0.39 | 4.9% |
| unattributed | 0.33 | 4.3% | 0.36 | 4.5% |
| store | 0.04 | 0.6% | 0.04 | 0.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 6.86 | 90.4% | 1.0 | 7.04 | `if ((((int64_t)0 <= token_ids[((int64_t)((int)blockIdx.x))]) && (token_ids[((int64_t)((int` |
| compute.reduce | 0.36 | 4.7% | 1.0 | 0.54 | `total[0] = tl::AllReduce<tl::SumOp, 256, 1, 0, tl::NamedBarrier<256>>::run(total[0], (&(wo` |
| store.global | 0.04 | 0.6% | 1.0 | 0.06 | `output[((int64_t)((int)blockIdx.x))] = ((int64_t)total[0]);` |

### metadata_builder_kernel grid (2, 1, 1) block (128, 1, 1)

289 launches, 8 warps; ranges, warp and last warp from the 2 recording CTAs' 8 warps; span 1.25 us, warp 0.68 us, nsys graphs 1.44 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.32 | 46.2% | 0.34 | 24.9% |
| load | 0.29 | 42.7% | 0.97 | 71.0% |
| store | 0.08 | 11.1% | 0.06 | 4.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#3 | 0.25 | 36.8% | 0.5 | 0.86 | `for (int __1 = 0; __1 < 2; ++__1)` |
| store.global#4 | 0.03 | 4.7% | 0.5 | 0.10 | `positions[(((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))` |
| load.global#2 | 0.03 | 4.1% | 0.1 | 0.22 | `if ((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) < nr))` |
| store.global#3 | 0.03 | 4.1% | 0.5 | 0.06 | `if (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) == 0)` |
| load.global | 0.01 | 1.8% | 0.1 | 0.10 | `if ((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) <= nr))` |
| store.global | 0.01 | 1.8% | 0.1 | 0.10 | `out_starts[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] = condval;` |
| store.global#2 | 0.00 | 0.6% | 0.1 | 0.03 | `request_positions[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] = condval_2;` |

### moe_grouped_fp8_fp4_gemm_kernel grid (40, 36, 1) block (128, 1, 1)

6 launches, 5760 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 62.14 us, warp 3.60 us, nsys graphs 9.86 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.53 | 42.5% | 0.21 | 66.7% |
| wait | 1.06 | 29.4% | 0.04 | 13.3% |
| compute | 0.48 | 13.3% | 0.00 | 0.0% |
| load | 0.46 | 12.7% | 0.06 | 20.0% |
| store | 0.07 | 2.0% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*6 (est.: sampled iterations x6) | 0.48 | 13.3% | 0.2 | 1.92 | `{` |
| wait.sync#5*6 (est.: sampled iterations x6) | 0.47 | 13.2% | 0.2 | 2.40 | `__syncthreads();` |
| load#2*6 (est.: sampled iterations x6) | 0.21 | 5.8% | 0.2 | 1.06 | `if (ko < 4)` |
| wait.cp_async*6 (est.: sampled iterations x6) | 0.19 | 5.3% | 0.2 | 2.02 | `tl::cp_async_wait<2>();` |
| wait.mbarrier*6 (est.: sampled iterations x6) | 0.11 | 3.2% | 0.2 | 0.48 | `loaded[(ko % 3)].wait((ko / 3));` |
| load.global#3 | 0.11 | 3.1% | 0.2 | 0.45 | `if (((expert < 128) && (0 <= expert))) \| if (((expert < 384) && (0 <= expert)))` |
| wait.sync#4*6 (est.: sampled iterations x6) | 0.10 | 2.7% | 0.2 | 0.38 | `__syncthreads();` |
| wait.sync#6*6 (est.: sampled iterations x6) | 0.09 | 2.5% | 0.2 | 0.38 | `__syncthreads();` |
| store.global | 0.07 | 2.0% | 0.2 | 0.37 | `for (int i_2 = 0; i_2 < 16; ++i_2) \| for (int i_4 = 0; i_4 < 32; ++i_4)` |
| load.global | 0.07 | 1.8% | 1.0 | 0.10 | `int expert = TileExpert[((int64_t)((int)blockIdx.y))];` |
| wait.sync | 0.04 | 1.3% | 1.0 | 0.06 | `__syncthreads();` |
| load | 0.04 | 1.2% | 0.2 | 0.22 | `for (int s = 0; s < 2; ++s)` |
| wait.sync#2 | 0.03 | 0.9% | 0.2 | 0.16 | `__syncthreads();` |
| load.global#2 | 0.02 | 0.7% | 0.2 | 0.19 | `if (((int)threadIdx.x) < 32) \| if (((int)threadIdx.x) < 64)` |
| wait.sync#3 | 0.02 | 0.4% | 0.2 | 0.06 | `__syncthreads();` |

### cache_writer_kernel grid (10, 1, 1) block (128, 1, 1)

51 launches, 40 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 6.91 us, warp 6.54 us, nsys graphs 4.80 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 4.00 | 61.2% | 4.02 | 55.9% |
| store | 2.12 | 32.4% | 2.09 | 29.1% |
| load | 0.34 | 5.2% | 0.96 | 13.3% |
| compute | 0.08 | 1.2% | 0.12 | 1.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 2.12 | 32.4% | 1.0 | 2.50 | `if (((int64_t)0 <= slot) & ((slot >> (int64_t)7) < ((int64_t)pages))) \| if (((int64_t)0 <=` |
| load.global | 0.28 | 4.3% | 1.0 | 0.32 | `for (int i = 0; i < 4; ++i)` |
| compute.reduce | 0.08 | 1.2% | 1.0 | 0.10 | `amax = max(amax, __shfl_xor_sync((uint)4294967295, amax, 1, 32));` |
| load.global#2 | 0.06 | 0.9% | 1.0 | 0.06 | `int64_t slot = slots[((int64_t)((int)blockIdx.x))];` |

### indexer_topk_segment_kernel grid (1, 5, 1) block (512, 1, 1)

32 launches, 80 warps; ranges, warp and last warp from the 5 recording CTAs' 80 warps; span 10.69 us, warp 10.61 us, nsys graphs 5.97 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed, with loops' unsampled iterations | 5.60 | 52.8% | 5.61 | 51.4% |
| wait | 3.78 | 35.6% | 3.88 | 35.6% |
| load | 1.06 | 10.0% | 1.10 | 10.1% |
| store | 0.17 | 1.6% | 0.32 | 3.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.06 | 10.0% | 1.0 | 1.14 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 4096); \| i` |
| wait.sync (sampled iterations) | 1.00 | 9.4% | 14.5 | 1.18 | `__syncthreads();` |
| wait.sync#24*9 (est.: sampled iterations x9) | 0.31 | 3.0% | 0.5 | 0.43 | `__syncthreads();` |
| wait.sync#20*9 (est.: sampled iterations x9) | 0.31 | 2.9% | 0.5 | 0.43 | `__syncthreads();` |
| wait.sync#23*9 (est.: sampled iterations x9) | 0.30 | 2.8% | 0.5 | 0.43 | `__syncthreads();` |
| wait.sync#19*9 (est.: sampled iterations x9) | 0.30 | 2.8% | 0.5 | 0.43 | `__syncthreads();` |
| wait.sync#14*8 (est.: sampled iterations x8) | 0.28 | 2.6% | 0.5 | 0.38 | `__syncthreads();` |
| wait.sync#6*8 (est.: sampled iterations x8) | 0.28 | 2.6% | 0.5 | 0.38 | `__syncthreads();` |
| wait.sync#13*8 (est.: sampled iterations x8) | 0.27 | 2.5% | 0.5 | 0.38 | `__syncthreads();` |
| wait.sync#5*8 (est.: sampled iterations x8) | 0.26 | 2.4% | 0.5 | 0.26 | `__syncthreads();` |
| store.global | 0.17 | 1.6% | 1.5 | 0.24 | `for (int i_3 = 0; i_3 < 4; ++i_3) \| if (((int)threadIdx.x) < total) \| if (((int)threadIdx.` |
| wait.sync#25 | 0.04 | 0.4% | 0.5 | 0.08 | `__syncthreads();` |
| wait.sync#16 | 0.04 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#8 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#17 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#18 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#9 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#3 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#2 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#11 | 0.03 | 0.3% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#7 | 0.03 | 0.3% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#10 | 0.03 | 0.3% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#15 | 0.03 | 0.3% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#12 | 0.03 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#4 | 0.03 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#21 | 0.02 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#22 | 0.02 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |

### topk_ids_kernel grid (5, 1, 1) block (512, 1, 1)

16 launches, 80 warps; ranges, warp and last warp from the 5 recording CTAs' 80 warps; span 21.12 us, warp 19.04 us, nsys graphs 20.75 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 15.43 | 81.0% | 16.29 | 74.7% |
| unattributed | 2.28 | 12.0% | 1.96 | 9.0% |
| wait | 0.94 | 4.9% | 0.65 | 3.0% |
| store | 0.39 | 2.1% | 2.92 | 13.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 15.43 | 81.0% | 1.0 | 16.42 | `for (int t = 0; t < (vocab >> 14); ++t)` |
| wait.sync | 0.80 | 4.2% | 1.0 | 1.54 | `__syncthreads();` |
| store.global | 0.39 | 2.1% | 1.0 | 2.98 | `if ((((int)threadIdx.x) >> 5) == 0)` |
| wait.sync#2 | 0.08 | 0.4% | 1.0 | 0.16 | `__syncthreads();` |
| wait.sync#3 | 0.06 | 0.3% | 1.0 | 0.06 | `__syncthreads();` |

### compress_kernel grid (2, 1, 1) block (128, 1, 1)

34 launches, 8 warps; ranges, warp and last warp from the 2 recording CTAs' 8 warps; span 7.95 us, warp 5.37 us, nsys graphs 12.59 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 3.17 | 59.0% | 3.73 | 37.8% |
| load | 1.69 | 31.4% | 4.94 | 50.2% |
| store | 0.46 | 8.5% | 1.09 | 11.1% |
| compute | 0.06 | 1.1% | 0.09 | 0.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.01 | 18.9% | 0.6 | 1.84 | `if ((((0 <= counts[0]) & (0 < counts[1])) & (counts[1] <= 2)) & (((((int)blockIdx.x) * 4) ` |
| load.global#2*4 (est.: sampled iterations x4) | 0.61 | 11.3% | 0.6 | 1.34 | `*(float4*)(v1 + 0) = *(float4*)(values + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048)` |
| store.global*4 (est.: sampled iterations x4) | 0.18 | 3.3% | 0.6 | 0.32 | `*(uint2*)(out + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + ((((int64_t)((int)thre` |
| store.global#3 | 0.10 | 1.9% | 0.3 | 0.16 | `emitted_slots[((((int64_t)((int)blockIdx.x)) * (int64_t)4) + (((int64_t)((int)threadIdx.x)` |
| store.global#2*4 (est.: sampled iterations x4) | 0.07 | 1.3% | 0.3 | 0.13 | `if ((slot >> (int64_t)3) < ((int64_t)blocks))` |
| compute.reduce | 0.06 | 1.1% | 0.6 | 0.10 | `float total = tl::warp_reduce_sum(ss[0]);` |
| store.global#5 | 0.05 | 0.9% | 0.3 | 0.16 | `emitted_slots[((((int64_t)((int)blockIdx.x)) * (int64_t)4) + (((int64_t)((int)threadIdx.x)` |
| load.global#3*4 (est.: sampled iterations x4) | 0.04 | 0.7% | 0.3 | 0.06 | `*(float4*)(w + 0) = *(float4*)(weight + ((c_1 * 128) + ((((int)threadIdx.x) & 31) * 4)));` |
| store.global#3*4 (est.: sampled iterations x4) | 0.04 | 0.7% | 0.3 | 0.06 | `*(uint2*)(out + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + ((((int64_t)((int)thre` |
| load.global#4 | 0.01 | 0.3% | 0.3 | 0.03 | `if (valid)` |
| store.global#4 | 0.01 | 0.2% | 0.3 | 0.03 | `emitted[((((int64_t)((int)blockIdx.x)) * (int64_t)4) + (((int64_t)((int)threadIdx.x)) >> (` |
| load.global#3 | 0.01 | 0.2% | 0.3 | 0.02 | `if (valid)` |
| store.global#2 | 0.01 | 0.2% | 0.3 | 0.02 | `emitted[((((int64_t)((int)blockIdx.x)) * (int64_t)4) + (((int64_t)((int)threadIdx.x)) >> (` |

### norm_forward_and_per_token_cast_kernel_kernel grid (3, 1, 1) block (128, 1, 1)

102 launches, 12 warps; ranges, warp and last warp from the 3 recording CTAs' 12 warps; span 2.75 us, warp 2.55 us, nsys graphs 1.89 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 1.62 | 63.5% | 2.05 | 67.9% |
| load | 0.34 | 13.3% | 0.34 | 11.2% |
| unattributed | 0.34 | 13.2% | 0.47 | 15.5% |
| wait | 0.14 | 5.4% | 0.05 | 1.5% |
| compute | 0.12 | 4.6% | 0.12 | 3.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*3 (est.: sampled iterations x3) | 1.15 | 45.1% | 1.0 | 1.15 | `for (int j_2 = 0; j_2 < 2; ++j_2)` |
| store.global*3 (est.: sampled iterations x3) | 0.29 | 11.3% | 1.0 | 0.38 | `for (int j_1 = 0; j_1 < 4; ++j_1)` |
| load.global*3 (est.: sampled iterations x3) | 0.18 | 6.9% | 1.0 | 0.19 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 6)) < num_tokens) && (((vi * 2) + (` |
| wait.sync | 0.14 | 5.4% | 1.0 | 0.26 | `__syncthreads();` |
| load.global#3*3 (est.: sampled iterations x3) | 0.13 | 5.0% | 1.0 | 0.19 | `if (((vi_1 * 2) + ((((int)threadIdx.x) & 63) >> 5)) < 5)` |
| compute.reduce | 0.12 | 4.6% | 1.0 | 0.13 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| store.global#5*3 (est.: sampled iterations x3) | 0.11 | 4.4% | 0.8 | 0.19 | `*(uint4*)(out + ((((((int64_t)vi_1) * (int64_t)512) + ((((int64_t)((int)threadIdx.x)) & (i` |
| store.global#2 | 0.04 | 1.7% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| load.global#2 | 0.03 | 1.4% | 1.0 | 0.06 | `*(float2*)(row_partial + 0) = *(float2*)(sum_shared + ((((int)threadIdx.x) >> 6) * 2)); \| ` |
| store.global#3 | 0.03 | 1.0% | 0.4 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |

### sparse_mla_kernel_1 grid (24, 5, 1) block (128, 1, 1)

96 launches, 120 CTAs, 480 warps; span 2.02 us, warp 1.55 us, nsys graphs 2.11 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.51 | 97.4% | 1.30 | 43.1% |
| load | 0.03 | 1.7% | 1.47 | 48.6% |
| wait | 0.01 | 0.6% | 0.13 | 4.2% |
| compute | 0.00 | 0.2% | 0.08 | 2.8% |
| store | 0.00 | 0.1% | 0.04 | 1.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.03 | 1.7% | 0.1 | 1.06 | `M = __fmul_rn(Sinks[((int)blockIdx.x)], 0x1.7154764ee6c2fp+0f/*1.442695e+00*/); \| if (((in` |
| wait.sync | 0.01 | 0.6% | 0.1 | 0.42 | `__syncthreads();` |
| compute.reduce | 0.00 | 0.2% | 0.0 | 0.10 | `amax[0] = tl::AllReduce<tl::MaxOp, 8, 1, 0, tl::NamedBarrier<128>>::run(amax[0]);` |
| store.global | 0.00 | 0.1% | 0.0 | 0.06 | `*(uint2*)(Output + (((((int64_t)((int)blockIdx.y)) * (int64_t)12288) + (((int64_t)((int)bl` |
| compute.reduce#3 | 0.00 | 0.0% | 0.0 | 0.00 | `amax[0] = tl::AllReduce<tl::MaxOp, 8, 1, 0, tl::NamedBarrier<128>>::run(amax[0]);` |
| load.global#3 | 0.00 | 0.0% | 0.0 | 0.00 | `M = __fmul_rn(Sinks[((int)blockIdx.x)], 0x1.7154764ee6c2fp+0f/*1.442695e+00*/);` |
| load.global#4 | 0.00 | 0.0% | 0.0 | 0.00 | `if (((int)threadIdx.x) < 32)` |
| store.global#2 | 0.00 | 0.0% | 0.0 | 0.00 | `*(uint2*)(Output + (((((int64_t)((int)blockIdx.y)) * (int64_t)12288) + (((int64_t)((int)bl` |
| wait.sync#12 | 0.00 | 0.0% | 0.0 | 0.00 | `__syncthreads();` |
| wait.sync#13 | 0.00 | 0.0% | 0.0 | 0.00 | `__syncthreads();` |

### topk_log_softmax_kernel grid (1, 1, 1) block (1024, 1, 1)

17 launches, 32 warps; ranges, warp and last warp from the 1 recording CTAs' 32 warps; span 16.45 us, warp 15.87 us, nsys graphs 16.19 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 12.89 | 81.2% | 13.55 | 82.4% |
| unattributed, with loops' unsampled iterations | 1.56 | 9.8% | 1.76 | 10.7% |
| wait | 1.03 | 6.5% | 0.70 | 4.2% |
| compute | 0.39 | 2.5% | 0.38 | 2.3% |
| store | 0.00 | 0.0% | 0.06 | 0.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 6.59 | 41.5% | 1.0 | 6.88 | `if (0 < (vocab & 2047)) \| if (0 < (vocab & 8191))` |
| load.global#2 | 6.29 | 39.6% | 1.0 | 6.69 | `for (int step_1 = 0; step_1 < (vocab >> 11); ++step_1) \| for (int step_1 = 0; step_1 < (vo` |
| wait.sync#2 | 0.45 | 2.8% | 1.0 | 0.80 | `__syncthreads();` |
| wait.sync | 0.33 | 2.1% | 1.0 | 0.74 | `__syncthreads();` |
| compute.reduce#3 | 0.18 | 1.1% | 1.0 | 0.19 | `ties[0] = vllm_add(ties[0], __shfl_xor_sync((uint)4294967295, ties[0], 16, 32));` |
| compute.reduce#2 | 0.11 | 0.7% | 1.0 | 0.13 | `se[0] = vllm_add(se[0], __shfl_xor_sync((uint)4294967295, se[0], 16, 32));` |
| compute.reduce | 0.10 | 0.7% | 1.0 | 0.13 | `best[0] = vllm_max(best[0], __shfl_xor_sync((uint)4294967295, best[0], 16, 32));` |
| wait.sync#4 | 0.10 | 0.6% | 1.0 | 0.16 | `__syncthreads();` |
| wait.sync#3 | 0.08 | 0.5% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#5 | 0.08 | 0.5% | 1.0 | 0.10 | `__syncthreads();` |
| load.global#3 (sampled iterations) | 0.01 | 0.1% | 0.0 | 0.38 | `float value_32 = vllm_sub(vllm_shifted(logits[((((int64_t)((int)blockIdx.x)) * logits_stri` |
| store.global (sampled iterations) | 0.00 | 0.0% | 0.0 | 0.06 | `output[(((((int64_t)c) * (int64_t)1024) + (((int64_t)((int)blockIdx.x)) * ((int64_t)topk))` |

### topk_log_softmax_kernel grid (5, 1, 1) block (1024, 1, 1)

16 launches, 160 warps; ranges, warp and last warp from the 5 recording CTAs' 160 warps; span 17.06 us, warp 16.24 us, nsys graphs 16.69 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 13.19 | 81.2% | 14.29 | 83.0% |
| unattributed, with loops' unsampled iterations | 1.60 | 9.8% | 1.74 | 10.1% |
| wait | 1.06 | 6.5% | 0.75 | 4.3% |
| compute | 0.39 | 2.4% | 0.38 | 2.2% |
| store | 0.00 | 0.0% | 0.06 | 0.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 6.68 | 41.1% | 1.0 | 7.07 | `if (0 < (vocab & 2047)) \| if (0 < (vocab & 8191))` |
| load.global#2 | 6.50 | 40.0% | 1.0 | 7.17 | `for (int step_1 = 0; step_1 < (vocab >> 11); ++step_1) \| for (int step_1 = 0; step_1 < (vo` |
| wait.sync#2 | 0.49 | 3.0% | 1.0 | 1.07 | `__syncthreads();` |
| wait.sync | 0.31 | 1.9% | 1.0 | 1.14 | `__syncthreads();` |
| compute.reduce#3 | 0.17 | 1.1% | 1.0 | 0.22 | `ties[0] = vllm_add(ties[0], __shfl_xor_sync((uint)4294967295, ties[0], 16, 32));` |
| compute.reduce#2 | 0.11 | 0.7% | 1.0 | 0.16 | `se[0] = vllm_add(se[0], __shfl_xor_sync((uint)4294967295, se[0], 16, 32));` |
| compute.reduce | 0.11 | 0.7% | 1.0 | 0.16 | `best[0] = vllm_max(best[0], __shfl_xor_sync((uint)4294967295, best[0], 16, 32));` |
| wait.sync#4 | 0.10 | 0.6% | 1.0 | 0.16 | `__syncthreads();` |
| wait.sync#5 | 0.08 | 0.5% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#3 | 0.08 | 0.5% | 1.0 | 0.10 | `__syncthreads();` |
| load.global#3 (sampled iterations) | 0.01 | 0.1% | 0.0 | 0.38 | `float value_32 = vllm_sub(vllm_shifted(logits[((((int64_t)((int)blockIdx.x)) * logits_stri` |
| store.global (sampled iterations) | 0.00 | 0.0% | 0.0 | 0.06 | `output[(((((int64_t)c) * (int64_t)1024) + (((int64_t)((int)blockIdx.x)) * ((int64_t)topk))` |

### metadata_chunk_kernel grid (5, 1, 1) block (128, 1, 1)

32 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 5.07 us, warp 4.59 us, nsys graphs 10.66 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 3.06 | 66.7% | 3.03 | 35.7% |
| load | 1.46 | 31.7% | 5.31 | 62.6% |
| store | 0.07 | 1.5% | 0.15 | 1.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.33 | 29.0% | 1.0 | 1.57 | `if ((((offset + ((int)blockIdx.x)) < src_rows) && (0 <= (offset + ((int)blockIdx.x)))))` |
| load.global#2 | 0.13 | 2.8% | 0.2 | 0.51 | `if (((0 <= req) & ((int64_t)0 <= pos)))` |
| store.global | 0.05 | 1.1% | 1.0 | 0.06 | `swa[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = con` |
| store.global#2 | 0.01 | 0.2% | 0.2 | 0.06 | `lengths[((int64_t)((int)blockIdx.x))] = ((int)max(info[1], (int64_t)0));` |
| store.global#3 | 0.01 | 0.2% | 0.2 | 0.06 | `top_lengths[((int64_t)((int)blockIdx.x))] = condval_12; \| top_lengths[((int64_t)((int)bloc` |

### moe_combine_kernel grid (5, 5, 1) block (128, 1, 1)

96 launches, 100 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 2.32 us, warp 1.75 us, nsys graphs 2.43 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.30 | 74.1% | 1.89 | 79.0% |
| unattributed | 0.40 | 22.7% | 0.42 | 17.5% |
| store | 0.05 | 3.1% | 0.08 | 3.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.30 | 74.1% | 1.0 | 1.86 | `for (int k = 0; k < 3; ++k) \| for (int k = 0; k < 6; ++k) \| if ((((int)blockIdx.y) < share` |
| store.global | 0.05 | 3.1% | 1.0 | 0.06 | `{` |

### router_partials_kernel grid (5, 1, 1) block (64, 1, 1)

51 launches, 10 warps; ranges, warp and last warp from the 5 recording CTAs' 10 warps; span 2.98 us, warp 2.16 us, nsys graphs 2.46 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.94 | 43.6% | 2.10 | 45.9% |
| load | 0.85 | 39.5% | 1.84 | 40.1% |
| store | 0.19 | 8.8% | 0.22 | 4.7% |
| compute | 0.18 | 8.2% | 0.42 | 9.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.45 | 21.1% | 1.0 | 0.48 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < mask_len))` |
| load.global#4 | 0.27 | 12.3% | 0.5 | 0.58 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < image_len)) \| if ((((((int)bl` |
| compute.reduce | 0.18 | 8.2% | 0.5 | 0.38 | `for (int k = 0; k < 3; ++k) \| for (int k = 0; k < 6; ++k)` |
| load.global#2 | 0.07 | 3.1% | 1.0 | 0.10 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < mask_len))` |
| load.global#3 | 0.06 | 3.0% | 1.0 | 0.06 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < mask_len))` |
| store.global | 0.06 | 3.0% | 1.0 | 0.10 | `if (condval == (bool)0) \| if (condval == (bool)1)` |
| store.global#2 | 0.06 | 3.0% | 1.0 | 0.10 | `if (condval_1 == (bool)0) \| if (condval_1 == (bool)1)` |
| store.global#3 | 0.06 | 2.8% | 0.5 | 0.13 | `if ((((int)threadIdx.x) & 31) < 3) \| if ((((int)threadIdx.x) & 31) < 6)` |

### local_stats_kernel grid (5, 16, 1) block (128, 1, 1)

16 launches, 320 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 12.42 us, warp 11.87 us, nsys graphs 12.88 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 10.90 | 91.8% | 12.07 | 90.9% |
| unattributed | 0.63 | 5.3% | 0.72 | 5.4% |
| compute | 0.17 | 1.4% | 0.16 | 1.2% |
| wait | 0.12 | 1.0% | 0.24 | 1.8% |
| store | 0.07 | 0.6% | 0.09 | 0.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 10.90 | 91.8% | 1.0 | 11.41 | `if (expanded_local_pos[((int64_t)((int)blockIdx.x))] < num_speculative_steps)` |
| wait.sync | 0.12 | 1.0% | 1.0 | 0.45 | `__syncthreads();` |
| compute.reduce | 0.11 | 0.9% | 1.0 | 0.13 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| store.global | 0.07 | 0.6% | 1.0 | 0.13 | `if (((int)threadIdx.x) == 0)` |
| compute.reduce#2 | 0.06 | 0.5% | 1.0 | 0.06 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 2, 32);` |

### compress_gather_partial_state_kernel grid (2, 1, 1) block (128, 1, 1)

17 launches, 8 warps; ranges, warp and last warp from the 2 recording CTAs' 8 warps; span 12.64 us, warp 9.74 us, nsys graphs 2.42 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 4.90 | 50.3% | 3.01 | 23.9% |
| load | 3.52 | 36.1% | 5.58 | 44.3% |
| store | 1.32 | 13.6% | 4.01 | 31.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 3.39 | 34.8% | 1.0 | 4.64 | `int nt = counts[0];` |
| store.global*2 (est.: sampled iterations x2) | 1.27 | 13.1% | 1.0 | 3.97 | `if (half == 0)` |
| load.global#2*2 (est.: sampled iterations x2) | 0.13 | 1.3% | 1.0 | 0.13 | `for (int i = 0; i < 4; ++i)` |
| store.global#2 | 0.05 | 0.5% | 1.0 | 0.06 | `if (((int)threadIdx.x) == 0)` |

### mhc_collapse_kernel grid (25, 1, 1) block (128, 1, 1)

48 launches, 100 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 3.70 us, warp 3.65 us, nsys graphs 3.52 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 3.50 | 96.0% | 3.95 | 90.6% |
| load | 0.10 | 2.6% | 0.18 | 4.0% |
| store | 0.05 | 1.4% | 0.23 | 5.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.10 | 2.6% | 1.0 | 0.13 | `*(float4*)(mv + 0) = *(float4*)(mix + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int6` |
| store.global | 0.05 | 1.4% | 1.0 | 0.06 | `*(uint4*)(out + ((((int64_t)((int)blockIdx.x)) * (int64_t)1024) + (((int64_t)((int)threadI` |

### sparse_mla_kernel_1 grid (24, 10, 1) block (128, 1, 1)

51 launches, 240 CTAs, 960 warps; span 2.24 us, warp 1.76 us, nsys graphs 4.38 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.74 | 98.9% | 2.30 | 61.8% |
| load | 0.01 | 0.7% | 1.21 | 32.4% |
| wait | 0.00 | 0.2% | 0.11 | 3.0% |
| compute | 0.00 | 0.1% | 0.07 | 1.9% |
| store | 0.00 | 0.0% | 0.04 | 0.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#3 | 0.01 | 0.6% | 0.0 | 0.77 | `M = __fmul_rn(Sinks[((int)blockIdx.x)], 0x1.7154764ee6c2fp+0f/*1.442695e+00*/);` |
| wait.sync#13 | 0.00 | 0.2% | 0.0 | 0.26 | `__syncthreads();` |
| load.global#4 | 0.00 | 0.1% | 0.0 | 0.26 | `if (((int)threadIdx.x) < 32)` |
| compute.reduce#3 | 0.00 | 0.1% | 0.0 | 0.13 | `amax[0] = tl::AllReduce<tl::MaxOp, 8, 1, 0, tl::NamedBarrier<128>>::run(amax[0]);` |
| wait.sync#12 | 0.00 | 0.1% | 0.0 | 0.16 | `__syncthreads();` |
| store.global#2 | 0.00 | 0.0% | 0.0 | 0.06 | `*(uint2*)(Output + (((((int64_t)((int)blockIdx.y)) * (int64_t)12288) + (((int64_t)((int)bl` |

### norm_forward_and_per_token_cast_kernel_kernel grid (5, 1, 1) block (128, 1, 1)

67 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 2.78 us, warp 2.56 us, nsys graphs 1.92 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 1.64 | 63.9% | 1.94 | 68.1% |
| unattributed | 0.38 | 14.8% | 0.37 | 12.9% |
| load | 0.32 | 12.3% | 0.37 | 12.9% |
| compute | 0.12 | 4.7% | 0.12 | 4.3% |
| wait | 0.11 | 4.3% | 0.05 | 1.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*3 (est.: sampled iterations x3) | 1.13 | 44.2% | 1.0 | 1.15 | `for (int j_2 = 0; j_2 < 2; ++j_2)` |
| store.global*3 (est.: sampled iterations x3) | 0.30 | 11.8% | 1.0 | 0.38 | `for (int j_1 = 0; j_1 < 4; ++j_1)` |
| load.global*3 (est.: sampled iterations x3) | 0.17 | 6.6% | 1.0 | 0.19 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 6)) < num_tokens) && (((vi * 2) + (` |
| store.global#5*3 (est.: sampled iterations x3) | 0.12 | 4.9% | 1.0 | 0.19 | `*(uint4*)(out + ((((((int64_t)vi_1) * (int64_t)512) + ((((int64_t)((int)threadIdx.x)) & (i` |
| compute.reduce | 0.12 | 4.7% | 1.0 | 0.13 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| load.global#3*3 (est.: sampled iterations x3) | 0.12 | 4.5% | 1.0 | 0.19 | `if (((vi_1 * 2) + ((((int)threadIdx.x) & 63) >> 5)) < 5)` |
| wait.sync | 0.11 | 4.3% | 1.0 | 0.19 | `__syncthreads();` |
| store.global#2 | 0.04 | 1.7% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| load.global#2 | 0.03 | 1.2% | 1.0 | 0.06 | `*(float2*)(row_partial + 0) = *(float2*)(sum_shared + ((((int)threadIdx.x) >> 6) * 2)); \| ` |
| store.global#3 | 0.03 | 1.2% | 0.5 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |
| load.global#2*5 (est.: sampled iterations x5) | 0.00 | 0.0% | 0.0 | 0.00 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 1024) + (((int)threadIdx.x) * 8)` |
| load.global*5 (est.: sampled iterations x5) | 0.00 | 0.0% | 0.0 | 0.00 | `*(uint4*)(x_local + (vi * 8)) = *(uint4*)(x + ((((((int64_t)vi) * (int64_t)1024) + (((int6` |
| store.global#4*5 (est.: sampled iterations x5) | 0.00 | 0.0% | 0.0 | 0.00 | `for (int j_1 = 0; j_1 < 2; ++j_1)` |
| store.global*5 (est.: sampled iterations x5) | 0.00 | 0.0% | 0.0 | 0.00 | `for (int j = 0; j < 4; ++j)` |

### mhc_post_kernel grid (25, 1, 1) block (128, 1, 1)

64 launches, 100 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 1.95 us, warp 1.66 us, nsys graphs 2.29 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.51 | 90.9% | 2.65 | 93.9% |
| load | 0.10 | 6.2% | 0.11 | 3.8% |
| store | 0.05 | 2.9% | 0.06 | 2.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.10 | 6.2% | 1.0 | 0.13 | `*(float4*)(pm + 0) = *(float4*)(post + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int` |
| store.global | 0.05 | 2.9% | 1.0 | 0.06 | `for (int n_1 = 0; n_1 < 4; ++n_1)` |

### per_token_cast_kernel_kernel grid (6, 1, 1) block (64, 1, 1)

51 launches, 12 warps; ranges, warp and last warp from the 6 recording CTAs' 12 warps; span 3.68 us, warp 3.21 us, nsys graphs 2.85 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 2.08 | 65.0% | 2.04 | 56.2% |
| load | 0.81 | 25.3% | 1.17 | 32.3% |
| store | 0.21 | 6.6% | 0.32 | 8.9% |
| compute | 0.10 | 3.2% | 0.10 | 2.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.72 | 22.5% | 1.0 | 1.02 | `for (int i = 0; i < 2; ++i)` |
| store.global#2 | 0.12 | 3.7% | 1.0 | 0.16 | `for (int i_3 = 0; i_3 < 4; ++i_3)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.10 | 3.2% | 1.0 | 0.13 | `amax_fragment[i_1] = tl::AllReduce<tl::MaxOp, 2, 1, 0, tl::NamedBarrier<64>>::run(amax_fra` |
| load.global#2*2 (est.: sampled iterations x2) | 0.09 | 2.8% | 1.0 | 0.13 | `amax_fragment[i_1] = cutlass::bfloat16_t(__hmax((bfloat16_t(((nv_bfloat162*)(&(amax_fragme` |
| store.global*2 (est.: sampled iterations x2) | 0.05 | 1.5% | 0.5 | 0.13 | `out_sf[((((((int64_t)((int)blockIdx.x)) % (int64_t)15) * (int64_t)32) + (((int64_t)((int)t` |
| store.global#3*2 (est.: sampled iterations x2) | 0.04 | 1.3% | 0.5 | 0.13 | `*(fp8_e4_16_t*)(out + ((((((((int64_t)((int)blockIdx.x)) / (int64_t)3) * (int64_t)6144) + ` |

### per_token_cast_kernel_kernel grid (5, 1, 1) block (64, 1, 1)

136 launches, 10 warps; ranges, warp and last warp from the 5 recording CTAs' 10 warps; span 1.18 us, warp 1.09 us, nsys graphs 1.06 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.45 | 41.2% | 0.56 | 41.1% |
| load | 0.34 | 31.3% | 0.39 | 28.6% |
| store | 0.20 | 18.7% | 0.31 | 22.7% |
| compute | 0.10 | 8.8% | 0.10 | 7.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.26 | 23.4% | 1.0 | 0.29 | `for (int i = 0; i < 2; ++i)` |
| store.global#2 | 0.12 | 10.5% | 1.0 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.10 | 8.8% | 1.0 | 0.13 | `amax_fragment[i_1] = tl::AllReduce<tl::MaxOp, 2, 1, 0, tl::NamedBarrier<64>>::run(amax_fra` |
| load.global#2*2 (est.: sampled iterations x2) | 0.09 | 7.9% | 1.0 | 0.13 | `amax_fragment[i_1] = cutlass::bfloat16_t(__hmax((bfloat16_t(((nv_bfloat162*)(&(amax_fragme` |
| store.global#3*2 (est.: sampled iterations x2) | 0.04 | 4.1% | 0.5 | 0.13 | `*(fp8_e4_16_t*)(out + ((((((((int64_t)((int)blockIdx.x)) / (int64_t)3) * (int64_t)6144) + ` |
| store.global*2 (est.: sampled iterations x2) | 0.04 | 4.1% | 0.5 | 0.13 | `out_sf[((((((int64_t)((int)blockIdx.x)) % (int64_t)15) * (int64_t)32) + (((int64_t)((int)t` |

### rope_kernel grid (5, 1, 1) block (192, 1, 1)

96 launches, 30 warps; ranges, warp and last warp from the 5 recording CTAs' 30 warps; span 1.34 us, warp 1.31 us, nsys graphs 1.47 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.75 | 57.7% | 1.22 | 65.4% |
| load | 0.51 | 39.2% | 0.60 | 32.0% |
| store | 0.04 | 3.2% | 0.05 | 2.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.51 | 39.2% | 1.0 | 0.51 | `if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < tokens)) \| int64_t position ` |
| store.global | 0.04 | 3.2% | 1.0 | 0.06 | `*(uint4*)(x + ((((((int64_t)((int)blockIdx.x)) * (int64_t)12288) + ((((int64_t)((int)threa` |

### indexer_topk_segment_kernel grid (2, 5, 1) block (512, 1, 1)

16 launches, 160 warps; ranges, warp and last warp from the 5 recording CTAs' 80 warps; span 10.10 us, warp 9.86 us, nsys graphs 4.74 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 6.16 | 62.5% | 6.15 | 57.0% |
| unattributed, with loops' unsampled iterations | 3.14 | 31.8% | 3.67 | 34.1% |
| load | 0.47 | 4.8% | 0.85 | 7.9% |
| store | 0.09 | 0.9% | 0.11 | 1.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync#24*9 (est.: sampled iterations x9) | 0.68 | 6.9% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#20*9 (est.: sampled iterations x9) | 0.67 | 6.8% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#19*9 (est.: sampled iterations x9) | 0.64 | 6.5% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#23*9 (est.: sampled iterations x9) | 0.63 | 6.4% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#6*8 (est.: sampled iterations x8) | 0.61 | 6.2% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync#14*8 (est.: sampled iterations x8) | 0.59 | 5.9% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync#13*8 (est.: sampled iterations x8) | 0.58 | 5.9% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync#5*8 (est.: sampled iterations x8) | 0.56 | 5.6% | 1.0 | 0.77 | `__syncthreads();` |
| load.global#2 | 0.26 | 2.7% | 1.0 | 0.51 | `for (int i = 0; i < 8; ++i)` |
| load.global | 0.21 | 2.1% | 1.0 | 0.26 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 4096); \| i` |
| wait.sync (sampled iterations) | 0.13 | 1.4% | 1.0 | 0.38 | `__syncthreads();` |
| wait.sync#25 | 0.10 | 1.0% | 1.0 | 0.16 | `__syncthreads();` |
| store.global | 0.09 | 0.9% | 1.0 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3) \| if (((int)threadIdx.x) < total) \| if (((int)threadIdx.` |
| wait.sync#17 | 0.08 | 0.8% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#18 | 0.08 | 0.8% | 1.0 | 0.16 | `__syncthreads();` |
| wait.sync#8 | 0.08 | 0.8% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#16 | 0.07 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#3 | 0.07 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#9 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#2 | 0.06 | 0.6% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#4 | 0.06 | 0.6% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#15 | 0.06 | 0.6% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#11 | 0.06 | 0.6% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#7 | 0.06 | 0.6% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#22 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#12 | 0.06 | 0.6% | 1.0 | 0.08 | `__syncthreads();` |
| wait.sync#21 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#10 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |

### indexer_topk_merge_kernel grid (1, 5, 1) block (512, 1, 1)

16 launches, 80 warps; ranges, warp and last warp from the 5 recording CTAs' 80 warps; span 9.28 us, warp 9.05 us, nsys graphs 3.84 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 6.13 | 67.7% | 5.96 | 55.7% |
| unattributed | 2.25 | 24.9% | 2.46 | 23.0% |
| load | 0.59 | 6.5% | 2.16 | 20.1% |
| store | 0.08 | 0.9% | 0.14 | 1.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync#20*9 (est.: sampled iterations x9) | 0.68 | 7.5% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#24*9 (est.: sampled iterations x9) | 0.67 | 7.4% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#19*9 (est.: sampled iterations x9) | 0.64 | 7.1% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#23*9 (est.: sampled iterations x9) | 0.64 | 7.1% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#14*8 (est.: sampled iterations x8) | 0.60 | 6.6% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync#6*8 (est.: sampled iterations x8) | 0.60 | 6.6% | 1.0 | 0.77 | `__syncthreads();` |
| load.global | 0.58 | 6.4% | 1.0 | 0.74 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 8192);` |
| wait.sync#13*8 (est.: sampled iterations x8) | 0.56 | 6.2% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync#5*8 (est.: sampled iterations x8) | 0.56 | 6.2% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync | 0.14 | 1.6% | 1.0 | 0.22 | `__syncthreads();` |
| wait.sync#17 | 0.08 | 0.9% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#16 | 0.07 | 0.8% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#8 | 0.07 | 0.8% | 1.0 | 0.10 | `__syncthreads();` |
| store.global#2 | 0.07 | 0.8% | 0.9 | 0.10 | `Values[((((int64_t)((int)blockIdx.y)) * (int64_t)512) + ((int64_t)((int)threadIdx.x)))] = ` |
| wait.sync#9 | 0.07 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#3 | 0.07 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#25 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#15 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#2 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#12 | 0.06 | 0.7% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#4 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#18 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#7 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#11 | 0.06 | 0.6% | 1.0 | 0.08 | `__syncthreads();` |
| wait.sync#22 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#10 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#21 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| load.global#2 | 0.01 | 0.1% | 0.1 | 0.10 | `if (((0 <= ((int*)sel_idx)[((int)threadIdx.x)]) && (((int*)sel_idx)[((int)threadIdx.x)] < ` |
| store.global | 0.01 | 0.1% | 0.1 | 0.13 | `Values[((((int64_t)((int)blockIdx.y)) * (int64_t)512) + ((int64_t)((int)threadIdx.x)))] = ` |

### mhc_finalize_kernel grid (6, 1, 1) block (128, 1, 1)

12 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 11.66 us, warp 10.60 us, nsys graphs 9.54 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 5.93 | 55.9% | 5.27 | 44.1% |
| store | 1.57 | 14.8% | 2.54 | 21.3% |
| load | 1.20 | 11.3% | 2.07 | 17.3% |
| compute | 1.00 | 9.4% | 1.25 | 10.4% |
| wait | 0.92 | 8.6% | 0.81 | 6.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync | 0.92 | 8.6% | 1.0 | 1.26 | `__syncthreads();` |
| store.global | 0.65 | 6.1% | 0.2 | 2.66 | `if (((int)threadIdx.x) < 4)` |
| compute.reduce | 0.57 | 5.4% | 0.2 | 2.30 | `row_max[0] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<32>>::run(row_max[0]);` |
| store.global#4*7 (est.: sampled iterations x7) | 0.56 | 5.3% | 0.8 | 0.90 | `if ((((int)threadIdx.x) % 4) == 0)` |
| load.global | 0.44 | 4.1% | 1.0 | 1.44 | `if (((int)threadIdx.x) < 32)` |
| compute.reduce#2*7 (est.: sampled iterations x7) | 0.42 | 4.0% | 0.8 | 0.67 | `amax[0] = max(amax[0], __shfl_xor_sync((uint)4294967295, amax[0], 1, 32));` |
| load.global#3 | 0.37 | 3.4% | 0.8 | 0.54 | `for (int i_split_2 = 0; i_split_2 < 40; ++i_split_2)` |
| load.global#4*7 (est.: sampled iterations x7) | 0.35 | 3.3% | 0.8 | 0.67 | `*(uint4*)(yv + 0) = *(uint4*)(C + ((((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((i` |
| store.global#3*7 (est.: sampled iterations x7) | 0.34 | 3.2% | 0.8 | 0.67 | `*(uint4*)(y + ((((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((int64_t)step) * (int6` |
| load.global#2 | 0.05 | 0.5% | 0.2 | 0.19 | `cm[0] = ((mixes_shared[((((int)threadIdx.x) & 15) + 8)] * scale[2]) + base[((((int)threadI` |
| store.global#2 | 0.01 | 0.1% | 0.2 | 0.06 | `if ((((int)threadIdx.x) >> 4) == 0)` |

### token_ranks_kernel grid (5, 1, 1) block (256, 1, 1)

16 launches, 40 warps; ranges, warp and last warp from the 5 recording CTAs' 40 warps; span 7.82 us, warp 7.73 us, nsys graphs 8.21 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 6.97 | 90.1% | 8.00 | 91.7% |
| unattributed | 0.38 | 4.9% | 0.38 | 4.3% |
| compute | 0.34 | 4.4% | 0.30 | 3.4% |
| store | 0.04 | 0.6% | 0.05 | 0.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 6.97 | 90.1% | 1.0 | 7.23 | `if ((((int64_t)0 <= token_ids[((int64_t)((int)blockIdx.x))]) && (token_ids[((int64_t)((int` |
| compute.reduce | 0.34 | 4.4% | 1.0 | 0.62 | `total[0] = tl::AllReduce<tl::SumOp, 256, 1, 0, tl::NamedBarrier<256>>::run(total[0], (&(wo` |
| store.global | 0.04 | 0.6% | 1.0 | 0.06 | `output[((int64_t)((int)blockIdx.x))] = ((int64_t)total[0]);` |

### gemm_splitk_reduce_cast_kernel grid (24, 2, 1) block (128, 1, 1)

102 launches, 192 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 1.06 us, warp 0.96 us, nsys graphs 1.18 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.43 | 45.2% | 0.68 | 51.1% |
| load | 0.37 | 39.0% | 0.48 | 36.2% |
| compute | 0.08 | 8.3% | 0.09 | 6.4% |
| store | 0.07 | 7.5% | 0.08 | 6.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.37 | 39.0% | 1.0 | 0.42 | `if ((1 <= M))` |
| compute.reduce | 0.08 | 8.3% | 1.0 | 0.10 | `amax[0] = tl::AllReduce<tl::MaxOp, 8, 1, 0, tl::NamedBarrier<128>>::run(amax[0]);` |
| store.global | 0.07 | 7.5% | 1.0 | 0.10 | `if (((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 5)) < M)` |

### norm_forward_and_per_token_cast_kernel_kernel grid (2, 1, 1) block (128, 1, 1)

51 launches, 8 warps; ranges, warp and last warp from the 2 recording CTAs' 8 warps; span 2.69 us, warp 2.69 us, nsys graphs 1.34 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 2.03 | 75.6% | 2.06 | 77.9% |
| load | 0.46 | 17.3% | 0.43 | 16.4% |
| wait | 0.14 | 5.1% | 0.06 | 2.3% |
| compute | 0.11 | 4.2% | 0.11 | 4.2% |
| unattributed | 0.00 | 0.0% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*4 (est.: sampled iterations x4) | 1.54 | 57.1% | 1.0 | 1.54 | `for (int j_2 = 0; j_2 < 2; ++j_2)` |
| store.global*4 (est.: sampled iterations x4) | 0.34 | 12.5% | 1.0 | 0.51 | `for (int j_1 = 0; j_1 < 4; ++j_1)` |
| load.global*4 (est.: sampled iterations x4) | 0.24 | 8.9% | 1.0 | 0.26 | `if (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 4)) < num_tokens)` |
| load.global#3*4 (est.: sampled iterations x4) | 0.19 | 7.1% | 1.0 | 0.26 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 128) + ((((int)threadIdx.x) & 15` |
| wait.sync | 0.14 | 5.1% | 1.0 | 0.29 | `__syncthreads();` |
| compute.reduce | 0.11 | 4.2% | 1.0 | 0.13 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| store.global#5*4 (est.: sampled iterations x4) | 0.08 | 3.0% | 0.6 | 0.13 | `*(uint4*)(out + ((((((int64_t)vi_1) * (int64_t)128) + ((((int64_t)((int)threadIdx.x)) & (i` |
| store.global#2 | 0.04 | 1.6% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| store.global#3 | 0.04 | 1.3% | 0.6 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |
| load.global#2 | 0.03 | 1.2% | 1.0 | 0.03 | `*(float2*)(row_partial + 0) = *(float2*)(sum_shared + ((((int)threadIdx.x) >> 6) * 2)); \| ` |

### per_token_cast_kernel_kernel grid (3, 1, 1) block (64, 1, 1)

102 launches, 6 warps; ranges, warp and last warp from the 3 recording CTAs' 6 warps; span 1.22 us, warp 1.11 us, nsys graphs 1.60 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.46 | 41.8% | 0.46 | 37.4% |
| load | 0.34 | 30.8% | 0.36 | 29.3% |
| store | 0.21 | 18.8% | 0.31 | 25.1% |
| compute | 0.10 | 8.7% | 0.10 | 8.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.26 | 23.1% | 1.0 | 0.26 | `for (int i = 0; i < 2; ++i)` |
| store.global#2 | 0.11 | 10.1% | 1.0 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.10 | 8.7% | 1.0 | 0.13 | `amax_fragment[i_1] = tl::AllReduce<tl::MaxOp, 2, 1, 0, tl::NamedBarrier<64>>::run(amax_fra` |
| load.global#2*2 (est.: sampled iterations x2) | 0.09 | 7.7% | 1.0 | 0.13 | `amax_fragment[i_1] = cutlass::bfloat16_t(__hmax((bfloat16_t(((nv_bfloat162*)(&(amax_fragme` |
| store.global*2 (est.: sampled iterations x2) | 0.05 | 4.8% | 0.5 | 0.13 | `out_sf[((((((int64_t)((int)blockIdx.x)) % (int64_t)15) * (int64_t)32) + (((int64_t)((int)t` |
| store.global#3*2 (est.: sampled iterations x2) | 0.04 | 3.8% | 0.5 | 0.13 | `*(fp8_e4_16_t*)(out + ((((((((int64_t)((int)blockIdx.x)) / (int64_t)3) * (int64_t)6144) + ` |

### indexer_mxfp4_quantize_kernel grid (5, 1, 1) block (128, 1, 1)

32 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 3.78 us, warp 3.69 us, nsys graphs 2.94 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 2.31 | 62.4% | 2.45 | 63.9% |
| store | 1.14 | 30.9% | 1.11 | 29.0% |
| unattributed | 0.25 | 6.6% | 0.27 | 7.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2*16 (est.: sampled iterations x16) | 1.28 | 34.6% | 1.0 | 1.54 | `float v__2 = __fmul_rn(((float)X[(((((int64_t)((int)blockIdx.x)) * (int64_t)4096) + (((int` |
| store.global*16 (est.: sampled iterations x16) | 1.10 | 29.8% | 1.0 | 1.54 | `Pool[(((((slot & (int64_t)127) * (int64_t)64) + ((((int64_t)((int)threadIdx.x)) & (int64_t` |
| load.global | 1.03 | 27.8% | 1.0 | 1.09 | `for (int i = 0; i < 32; ++i)` |
| store.global#2 | 0.04 | 1.1% | 1.0 | 0.06 | `Pool[(((((slot & (int64_t)127) * (int64_t)4) + ((slot >> (int64_t)7) * ((int64_t)pool_stri` |

### rope_kernel grid (10, 1, 1) block (192, 1, 1)

51 launches, 60 warps; ranges, warp and last warp from the 5 recording CTAs' 30 warps; span 1.34 us, warp 1.28 us, nsys graphs 1.50 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.76 | 59.3% | 1.60 | 71.6% |
| load | 0.48 | 37.5% | 0.59 | 26.4% |
| store | 0.04 | 3.2% | 0.04 | 2.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.48 | 37.5% | 1.0 | 0.48 | `if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < tokens)) \| int64_t position ` |
| store.global | 0.04 | 3.2% | 1.0 | 0.06 | `*(uint4*)(x + ((((((int64_t)((int)blockIdx.x)) * (int64_t)12288) + ((((int64_t)((int)threa` |

### indexer_mxfp4_quantize_kernel grid (1, 1, 1) block (128, 1, 1)

34 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 2.98 us, warp 1.00 us, nsys graphs 1.98 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.37 | 37.3% | 0.60 | 18.1% |
| load | 0.36 | 36.1% | 1.51 | 45.6% |
| store | 0.26 | 26.5% | 1.20 | 36.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2*16 (est.: sampled iterations x16) | 0.26 | 25.7% | 0.2 | 1.02 | `float v__2 = __fmul_rn(((float)X[(((((int64_t)((int)blockIdx.x)) * (int64_t)4096) + (((int` |
| store.global*16 (est.: sampled iterations x16) | 0.26 | 25.7% | 0.2 | 1.02 | `Pool[(((((slot & (int64_t)127) * (int64_t)64) + ((((int64_t)((int)threadIdx.x)) & (int64_t` |
| load.global | 0.10 | 10.4% | 0.2 | 0.42 | `for (int i = 0; i < 32; ++i)` |
| store.global#2 | 0.01 | 0.8% | 0.2 | 0.03 | `Pool[(((((slot & (int64_t)127) * (int64_t)4) + ((slot >> (int64_t)7) * ((int64_t)pool_stri` |

### prepare_dflash_inputs_kernel grid (9, 1, 1) block (256, 1, 1)

51 launches, 72 warps; ranges, warp and last warp from the 5 recording CTAs' 40 warps; span 1.98 us, warp 0.73 us, nsys graphs 2.53 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.25 | 34.7% | 1.40 | 68.8% |
| unattributed | 0.25 | 33.8% | 0.36 | 17.6% |
| store | 0.23 | 31.5% | 0.28 | 13.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.24 | 33.0% | 0.2 | 1.34 | `req = idx_mapping[((int)blockIdx.y)];` |
| store.global | 0.18 | 24.8% | 0.8 | 0.26 | `if (((int)blockIdx.y) == (num_reqs - 1))` |
| store.global#2 | 0.05 | 6.7% | 0.2 | 0.29 | `if (((int)blockIdx.y) == (num_reqs - 1))` |
| load.global#2 | 0.01 | 1.6% | 0.2 | 0.06 | `vllm_store_if_b32((&(out_query_start_loc[((int)blockIdx.y)])), (((int)blockIdx.y) * num_qu` |

### gemm_bf16_partials_kernel grid (2, 4, 1) block (256, 1, 1)

34 launches, 64 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 2.02 us, warp 1.38 us, nsys graphs 2.11 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.92 | 66.9% | 2.26 | 75.0% |
| wait | 0.23 | 16.5% | 0.43 | 14.1% |
| compute | 0.11 | 8.3% | 0.23 | 7.7% |
| load | 0.07 | 4.9% | 0.00 | 0.0% |
| store | 0.05 | 3.5% | 0.10 | 3.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*2 (est.: sampled iterations x2) | 0.11 | 8.3% | 0.5 | 0.26 | `{` |
| wait.sync | 0.11 | 7.9% | 1.0 | 0.13 | `__syncthreads();` |
| load.tma*2 (est.: sampled iterations x2) | 0.07 | 4.9% | 0.5 | 0.19 | `if (tl::tl_shuffle_elect<128>())` |
| wait.mbarrier#2*2 (est.: sampled iterations x2) | 0.06 | 4.3% | 0.5 | 0.13 | `mbarrier[kb_1].wait(0);` |
| wait.mbarrier*2 (est.: sampled iterations x2) | 0.06 | 4.3% | 0.5 | 0.13 | `mbarrier[(kb + 3)].wait(1);` |
| store.global | 0.05 | 3.5% | 0.5 | 0.13 | `for (int i_1 = 0; i_1 < 4; ++i_1) \| for (int i_2 = 0; i_2 < 8; ++i_2) \| for (int i_4 = 0; ` |

### sparse_mla_kernel grid (5, 1, 6) block (256, 1, 1)

4 launches, 240 warps; ranges, warp and last warp from the 5 recording CTAs' 40 warps; span 25.12 us, warp 19.66 us, nsys graphs 13.36 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed, with loops' unsampled iterations | 11.91 | 60.6% | 12.37 | 49.1% |
| wait | 3.10 | 15.7% | 5.76 | 22.9% |
| load | 2.53 | 12.9% | 4.78 | 19.0% |
| compute | 1.46 | 7.4% | 1.48 | 5.9% |
| store | 0.66 | 3.4% | 0.81 | 3.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync (sampled iterations) | 3.03 | 15.4% | 11.0 | 4.98 | `__syncthreads();` |
| load.global (sampled iterations) | 2.22 | 11.3% | 2.0 | 7.98 | `for (int c = 0; c < 4; ++c) \| for (int i = 0; i < 8; ++i) \| for (int i_1 = 0; i_1 < 8; ++i` |
| compute.mma (sampled iterations) | 1.22 | 6.2% | 2.0 | 1.28 | `for (int ki = 0; ki < 32; ++ki) \| for (int ki_1 = 0; ki_1 < 2; ++ki_1)` |
| store.global | 0.66 | 3.4% | 1.0 | 0.83 | `for (int i_10 = 0; i_10 < 32; ++i_10) \| for (int i_11 = 0; i_11 < 32; ++i_11) \| for (int i` |
| load.cp_async (sampled iterations) | 0.31 | 1.6% | 2.0 | 0.50 | `for (int k_1 = 0; k_1 < 5; ++k_1) \| for (int k_1 = 0; k_1 < 9; ++k_1) \| for (int k_3 = 0; ` |
| compute.reduce (sampled iterations) | 0.25 | 1.3% | 2.0 | 0.27 | `m_g[i_5] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(m_g[i_5], (&((` |
| wait.cp_async (sampled iterations) | 0.07 | 0.4% | 1.0 | 0.10 | `tl::cp_async_wait<0>();` |

### gemm_bf16_partials_kernel grid (1, 40, 1) block (256, 1, 1)

34 launches, 320 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 2.70 us, warp 1.72 us, nsys graphs 2.85 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.27 | 73.8% | 2.33 | 78.8% |
| wait | 0.23 | 13.2% | 0.30 | 10.2% |
| compute | 0.11 | 6.4% | 0.23 | 7.9% |
| load | 0.07 | 3.9% | 0.00 | 0.0% |
| store | 0.05 | 2.8% | 0.09 | 3.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*2 (est.: sampled iterations x2) | 0.11 | 6.4% | 0.5 | 0.26 | `{` |
| wait.sync | 0.10 | 6.0% | 1.0 | 0.13 | `__syncthreads();` |
| load.tma*2 (est.: sampled iterations x2) | 0.07 | 3.9% | 0.5 | 0.19 | `if (tl::tl_shuffle_elect<128>())` |
| wait.mbarrier#2*2 (est.: sampled iterations x2) | 0.06 | 3.7% | 0.5 | 0.13 | `mbarrier[kb_1].wait(0);` |
| wait.mbarrier*2 (est.: sampled iterations x2) | 0.06 | 3.5% | 0.5 | 0.13 | `mbarrier[(kb + 3)].wait(1);` |
| store.global | 0.05 | 2.8% | 0.5 | 0.13 | `for (int i_1 = 0; i_1 < 4; ++i_1) \| for (int i_2 = 0; i_2 < 8; ++i_2) \| for (int i_4 = 0; ` |

### indexer_score_kernel grid (128, 5, 1) block (128, 1, 1)

16 launches, 2560 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 4.27 us, warp 1.47 us, nsys graphs 3.42 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.82 | 55.8% | 4.30 | 73.2% |
| unattributed, with loops' unsampled iterations | 0.53 | 35.7% | 1.23 | 20.9% |
| wait | 0.08 | 5.6% | 0.17 | 3.0% |
| compute | 0.03 | 2.2% | 0.13 | 2.2% |
| store | 0.01 | 0.7% | 0.04 | 0.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.39 | 26.2% | 1.0 | 0.77 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 4096); \| i` |
| load.global#2 (sampled iterations) | 0.38 | 25.7% | 0.2 | 1.60 | `for (int i_1 = 0; i_1 < 32; ++i_1)` |
| load.global#3 (sampled iterations) | 0.06 | 3.9% | 0.2 | 0.40 | `if (((int)threadIdx.x) < 64)` |
| wait.sync#2 (sampled iterations) | 0.05 | 3.5% | 0.2 | 0.37 | `__syncthreads();` |
| compute.mma (sampled iterations) | 0.03 | 2.2% | 0.2 | 0.13 | `for (int ki = 0; ki < 2; ++ki)` |
| wait.sync (sampled iterations) | 0.01 | 0.9% | 0.2 | 0.06 | `__syncthreads();` |
| wait.sync#3 (sampled iterations) | 0.01 | 0.7% | 0.2 | 0.06 | `__syncthreads();` |
| store.global (sampled iterations) | 0.01 | 0.7% | 0.2 | 0.05 | `if ((((int)threadIdx.x) >> 6) == 0)` |
| wait.sync#4 (sampled iterations) | 0.01 | 0.5% | 0.2 | 0.03 | `__syncthreads();` |

### activation_swiglu_clamp_kernel grid (1, 10, 1) block (128, 1, 1)

51 launches, 40 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 1.79 us, warp 0.95 us, nsys graphs 6.72 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.84 | 88.7% | 1.68 | 89.9% |
| load | 0.07 | 7.6% | 0.14 | 7.4% |
| store | 0.04 | 3.7% | 0.05 | 2.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.07 | 7.6% | 0.8 | 0.10 | `*(uint4*)(gate + 0) = *(uint4*)(X + ((((int64_t)((int)blockIdx.y)) * (int64_t)1536) + (((i` |
| store.global | 0.04 | 3.7% | 0.8 | 0.06 | `*(uint4*)(Y + ((((int64_t)((int)blockIdx.y)) * (int64_t)768) + (((int64_t)((int)threadIdx.` |

### activation_swiglu_clamp_kernel grid (1, 5, 1) block (128, 1, 1)

96 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 0.93 us, warp 0.74 us, nsys graphs 1.82 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.63 | 85.9% | 0.80 | 85.3% |
| load | 0.07 | 9.8% | 0.09 | 10.0% |
| store | 0.03 | 4.3% | 0.04 | 4.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.07 | 9.8% | 0.8 | 0.10 | `*(uint4*)(gate + 0) = *(uint4*)(X + ((((int64_t)((int)blockIdx.y)) * (int64_t)1536) + (((i` |
| store.global | 0.03 | 4.3% | 0.8 | 0.06 | `*(uint4*)(Y + ((((int64_t)((int)blockIdx.y)) * (int64_t)768) + (((int64_t)((int)threadIdx.` |

### finish_argmax_kernel grid (2, 1, 1) block (32, 1, 1)

85 launches, 2 warps; ranges, warp and last warp from the 2 recording CTAs' 2 warps; span 0.74 us, warp 0.74 us, nsys graphs 0.90 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.38 | 52.2% | 0.52 | 60.2% |
| compute | 0.13 | 17.4% | 0.12 | 13.7% |
| load | 0.13 | 17.4% | 0.13 | 14.5% |
| store | 0.10 | 13.0% | 0.10 | 11.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.reduce | 0.13 | 17.4% | 1.0 | 0.13 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| load.global | 0.13 | 17.4% | 1.0 | 0.13 | `for (int k = 0; k < 2; ++k) \| if ((((int)threadIdx.x) < parts))` |
| store.global | 0.10 | 13.0% | 1.0 | 0.10 | `if (((int)threadIdx.x) == 0)` |

### sink_merge_kernel grid (1, 1, 1) block (128, 1, 1)

17 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 4.32 us, warp 4.32 us, nsys graphs 4.38 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.90 | 44.1% | 2.04 | 47.7% |
| unattributed | 1.00 | 23.1% | 1.10 | 25.9% |
| wait | 0.82 | 18.9% | 0.43 | 10.2% |
| compute | 0.55 | 12.8% | 0.54 | 12.6% |
| store | 0.05 | 1.1% | 0.16 | 3.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2*3 (est.: sampled iterations x3) | 1.63 | 37.8% | 1.0 | 1.63 | `for (int step = 0; step < ((width + 127) >> 7); ++step) \| for (int step = 0; step < ((widt` |
| compute.reduce*3 (est.: sampled iterations x3) | 0.55 | 12.8% | 1.0 | 0.58 | `float other = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| wait.sync#3*3 (est.: sampled iterations x3) | 0.50 | 11.7% | 1.0 | 0.67 | `__syncthreads();` |
| load.global | 0.22 | 5.2% | 1.0 | 0.22 | `last = sampled[(max((((int64_t)num_sampled[((int64_t)((int)blockIdx.x))]) - (int64_t)1), (` |
| wait.sync*3 (est.: sampled iterations x3) | 0.17 | 3.9% | 1.0 | 0.19 | `__syncthreads();` |
| wait.sync#2*3 (est.: sampled iterations x3) | 0.14 | 3.3% | 1.0 | 0.19 | `__syncthreads();` |
| load.global#3*3 (est.: sampled iterations x3) | 0.05 | 1.1% | 0.2 | 0.19 | `if ((best[0] == -CUDART_INF_F))` |
| store.global*3 (est.: sampled iterations x3) | 0.05 | 1.1% | 0.2 | 0.19 | `alternates[((((int64_t)((int)blockIdx.x)) * (int64_t)3) + ((int64_t)rank))] = condval_31;` |

### sink_candidates_kernel grid (1, 127, 1) block (128, 1, 1)

17 launches, 508 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 4.32 us, warp 3.54 us, nsys graphs 3.17 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.40 | 39.4% | 1.82 | 43.9% |
| compute | 0.90 | 25.5% | 0.87 | 20.9% |
| load | 0.55 | 15.4% | 0.81 | 19.5% |
| wait | 0.44 | 12.5% | 0.34 | 8.2% |
| store | 0.26 | 7.2% | 0.32 | 7.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.reduce*4 (est.: sampled iterations x4) | 0.65 | 18.4% | 1.0 | 0.77 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| load.global | 0.55 | 15.4% | 1.0 | 0.61 | `local[0] = max((num_sampled[((int64_t)((int)blockIdx.x))] - 1), 0);` |
| store.global*4 (est.: sampled iterations x4) | 0.26 | 7.2% | 1.0 | 0.38 | `if (((int)threadIdx.x) == 0)` |
| compute.reduce#2*4 (est.: sampled iterations x4) | 0.25 | 7.0% | 1.0 | 0.26 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 2, 32);` |
| wait.sync#2*4 (est.: sampled iterations x4) | 0.24 | 6.9% | 1.0 | 0.38 | `__syncthreads();` |
| wait.sync*4 (est.: sampled iterations x4) | 0.20 | 5.6% | 1.0 | 0.26 | `__syncthreads();` |

### moe_combine_kernel grid (5, 10, 1) block (128, 1, 1)

51 launches, 200 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 1.41 us, warp 1.10 us, nsys graphs 1.54 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.64 | 58.4% | 0.91 | 66.4% |
| unattributed | 0.40 | 36.5% | 0.41 | 29.9% |
| store | 0.06 | 5.1% | 0.05 | 3.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.64 | 58.4% | 1.0 | 0.93 | `for (int k = 0; k < 3; ++k) \| for (int k = 0; k < 6; ++k) \| if ((((int)blockIdx.y) < share` |
| store.global | 0.06 | 5.1% | 1.0 | 0.06 | `{` |

### gemm_splitk_reduce_cast_kernel grid (24, 3, 1) block (128, 1, 1)

51 launches, 288 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 1.15 us, warp 1.07 us, nsys graphs 1.25 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.46 | 43.2% | 0.51 | 39.9% |
| unattributed | 0.45 | 41.7% | 0.59 | 46.3% |
| compute | 0.08 | 7.7% | 0.08 | 6.5% |
| store | 0.08 | 7.5% | 0.09 | 7.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.46 | 43.2% | 1.0 | 0.51 | `if ((1 <= M))` |
| compute.reduce | 0.08 | 7.7% | 1.0 | 0.10 | `amax[0] = tl::AllReduce<tl::MaxOp, 8, 1, 0, tl::NamedBarrier<128>>::run(amax[0]);` |
| store.global | 0.08 | 7.5% | 1.0 | 0.10 | `if (((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 5)) < M)` |

### indexer_score_kernel grid (64, 5, 1) block (128, 1, 1)

16 launches, 1280 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 4.13 us, warp 1.92 us, nsys graphs 3.33 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.06 | 55.0% | 2.07 | 50.3% |
| unattributed, with loops' unsampled iterations | 0.67 | 34.6% | 1.39 | 33.7% |
| wait | 0.15 | 7.7% | 0.46 | 11.3% |
| compute | 0.04 | 2.1% | 0.16 | 3.8% |
| store | 0.01 | 0.5% | 0.04 | 1.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.77 | 39.9% | 1.0 | 1.09 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 4096); \| i` |
| load.global#2 (sampled iterations) | 0.26 | 13.6% | 0.2 | 1.17 | `for (int i_1 = 0; i_1 < 32; ++i_1)` |
| wait.sync#3 (sampled iterations) | 0.06 | 3.2% | 0.2 | 0.26 | `__syncthreads();` |
| wait.sync#2 (sampled iterations) | 0.04 | 2.2% | 0.2 | 0.22 | `__syncthreads();` |
| compute.mma (sampled iterations) | 0.04 | 2.1% | 0.2 | 0.16 | `for (int ki = 0; ki < 2; ++ki)` |
| load.global#3 (sampled iterations) | 0.03 | 1.5% | 0.2 | 0.22 | `if (((int)threadIdx.x) < 64)` |
| wait.sync (sampled iterations) | 0.03 | 1.5% | 0.2 | 0.14 | `__syncthreads();` |
| wait.sync#4 (sampled iterations) | 0.02 | 0.8% | 0.2 | 0.06 | `__syncthreads();` |
| store.global (sampled iterations) | 0.01 | 0.5% | 0.2 | 0.06 | `if ((((int)threadIdx.x) >> 6) == 0)` |

### adaptive_offsets_kernel grid (1, 1, 1) block (256, 1, 1)

16 launches, 8 warps; ranges, warp and last warp from the 1 recording CTAs' 8 warps; span 3.78 us, warp 3.75 us, nsys graphs 1.95 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 1.80 | 47.9% | 1.12 | 29.5% |
| unattributed, with loops' unsampled iterations | 1.43 | 38.3% | 1.51 | 39.6% |
| store | 0.30 | 8.1% | 0.54 | 14.1% |
| load | 0.22 | 5.8% | 0.64 | 16.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync (sampled iterations) | 0.43 | 11.4% | 1.0 | 0.48 | `__syncthreads();` |
| wait.sync#18 (sampled iterations) | 0.34 | 9.2% | 1.0 | 0.38 | `__syncthreads();` |
| store.global#5 | 0.13 | 3.4% | 1.0 | 0.16 | `for (int chunk_1 = 0; chunk_1 < ((starts_len + 255) >> 8); ++chunk_1)` |
| load.global#2 (sampled iterations) | 0.13 | 3.4% | 1.0 | 0.48 | `if ((((chunk * 256) + ((int)threadIdx.x)) < num_reqs))` |
| load.global (sampled iterations) | 0.08 | 2.2% | 1.0 | 0.10 | `if (((((chunk * 256) + ((int)threadIdx.x)) < num_reqs) & (drafted != 0)))` |
| store.global#2 (sampled iterations) | 0.08 | 2.2% | 1.0 | 0.10 | `if ((((chunk * 256) + ((int)threadIdx.x)) < num_reqs) & (drafted == 0))` |
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

### per_token_cast_kernel_kernel grid (10, 1, 1) block (64, 1, 1)

51 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 10 warps; span 1.18 us, warp 1.07 us, nsys graphs 1.09 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.44 | 40.7% | 0.41 | 35.2% |
| load | 0.31 | 29.0% | 0.36 | 30.8% |
| store | 0.22 | 20.7% | 0.30 | 25.6% |
| compute | 0.10 | 9.6% | 0.10 | 8.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.23 | 21.3% | 1.0 | 0.29 | `for (int i = 0; i < 2; ++i)` |
| store.global#2 | 0.11 | 10.5% | 1.0 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.10 | 9.6% | 1.0 | 0.13 | `amax_fragment[i_1] = tl::AllReduce<tl::MaxOp, 2, 1, 0, tl::NamedBarrier<64>>::run(amax_fra` |
| load.global#2*2 (est.: sampled iterations x2) | 0.08 | 7.8% | 1.0 | 0.13 | `amax_fragment[i_1] = cutlass::bfloat16_t(__hmax((bfloat16_t(((nv_bfloat162*)(&(amax_fragme` |
| store.global*2 (est.: sampled iterations x2) | 0.06 | 5.4% | 0.6 | 0.13 | `out_sf[((((((int64_t)((int)blockIdx.x)) % (int64_t)15) * (int64_t)32) + (((int64_t)((int)t` |
| store.global#3*2 (est.: sampled iterations x2) | 0.05 | 4.8% | 0.6 | 0.13 | `*(fp8_e4_16_t*)(out + ((((((((int64_t)((int)blockIdx.x)) / (int64_t)3) * (int64_t)6144) + ` |

### norm_forward_and_per_token_cast_kernel_kernel grid (8, 1, 1) block (128, 1, 1)

17 launches, 32 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 2.34 us, warp 2.29 us, nsys graphs 1.76 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 2.17 | 94.7% | 3.30 | 94.8% |
| load | 0.45 | 19.6% | 0.46 | 13.2% |
| compute | 0.13 | 5.7% | 0.13 | 3.6% |
| wait | 0.05 | 2.4% | 0.06 | 1.8% |
| unattributed | 0.00 | 0.0% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*5 (est.: sampled iterations x5) | 1.89 | 82.4% | 1.0 | 2.24 | `for (int j_1 = 0; j_1 < 2; ++j_1)` |
| load.global*5 (est.: sampled iterations x5) | 0.23 | 10.0% | 1.0 | 0.32 | `*(uint4*)(x_local + (vi * 8)) = *(uint4*)(x + ((((((int64_t)vi) * (int64_t)1024) + (((int6` |
| store.global*5 (est.: sampled iterations x5) | 0.23 | 10.0% | 1.0 | 0.32 | `for (int j = 0; j < 4; ++j)` |
| load.global#2*5 (est.: sampled iterations x5) | 0.22 | 9.6% | 1.0 | 0.32 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 1024) + (((int)threadIdx.x) * 8)` |
| compute.reduce | 0.13 | 5.7% | 1.0 | 0.16 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| wait.sync | 0.05 | 2.4% | 1.0 | 0.10 | `__syncthreads();` |
| store.global#2 | 0.04 | 1.7% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| store.global#3 | 0.01 | 0.5% | 0.2 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |

### gemm_splitk_reduce_kernel grid (1, 5, 1) block (128, 1, 1)

64 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 0.83 us, warp 0.83 us, nsys graphs 1.02 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.46 | 55.2% | 0.44 | 48.4% |
| load | 0.32 | 38.8% | 0.42 | 45.8% |
| store | 0.05 | 6.0% | 0.05 | 5.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.32 | 38.8% | 1.0 | 0.34 | `acc[0] = P[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x))` |
| store.global | 0.05 | 6.0% | 1.0 | 0.06 | `C[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x)) * (int64` |

### per_token_cast_kernel_kernel grid (20, 1, 1) block (64, 1, 1)

51 launches, 40 warps; ranges, warp and last warp from the 4 recording CTAs' 8 warps; span 1.09 us, warp 1.04 us, nsys graphs 0.99 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.35 | 33.8% | 0.38 | 34.6% |
| store | 0.30 | 29.2% | 0.31 | 28.3% |
| unattributed | 0.29 | 27.7% | 0.31 | 28.5% |
| compute | 0.10 | 9.2% | 0.09 | 8.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.26 | 25.4% | 1.0 | 0.29 | `for (int i = 0; i < 2; ++i)` |
| store.global#2 | 0.11 | 10.8% | 1.0 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3)` |
| store.global*2 (est.: sampled iterations x2) | 0.10 | 10.0% | 1.0 | 0.13 | `out_sf[((((((int64_t)((int)blockIdx.x)) % (int64_t)15) * (int64_t)32) + (((int64_t)((int)t` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.10 | 9.2% | 1.0 | 0.13 | `amax_fragment[i_1] = tl::AllReduce<tl::MaxOp, 2, 1, 0, tl::NamedBarrier<64>>::run(amax_fra` |
| load.global#2*2 (est.: sampled iterations x2) | 0.09 | 8.5% | 1.0 | 0.13 | `amax_fragment[i_1] = cutlass::bfloat16_t(__hmax((bfloat16_t(((nv_bfloat162*)(&(amax_fragme` |
| store.global#3*2 (est.: sampled iterations x2) | 0.09 | 8.5% | 1.0 | 0.13 | `*(fp8_e4_16_t*)(out + ((((((((int64_t)((int)blockIdx.x)) / (int64_t)3) * (int64_t)6144) + ` |

### sink_cut_kernel grid (1, 1, 1) block (128, 1, 1)

34 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 1.28 us, warp 0.54 us, nsys graphs 2.08 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.26 | 47.8% | 0.30 | 19.4% |
| store | 0.16 | 29.4% | 0.62 | 40.8% |
| load | 0.12 | 22.8% | 0.61 | 39.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.16 | 29.4% | 0.6 | 0.48 | `if (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) < num_cut) \| saved[((((int64_t)((int)` |
| load.global | 0.12 | 22.8% | 0.1 | 0.50 | `for (int b = 0; b < num_reqs; ++b)` |

### verify_kernel grid (1, 1, 1) block (96, 1, 1)

17 launches, 3 warps; ranges, warp and last warp from the 1 recording CTAs' 3 warps; span 2.85 us, warp 2.72 us, nsys graphs 2.30 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.09 | 40.0% | 1.24 | 41.0% |
| unattributed | 0.98 | 36.1% | 0.97 | 32.0% |
| compute | 0.29 | 10.6% | 0.28 | 9.4% |
| wait | 0.20 | 7.5% | 0.24 | 7.9% |
| store | 0.16 | 5.9% | 0.30 | 9.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.02 | 37.6% | 1.0 | 1.02 | `start = cu_num_logits[((int64_t)((int)blockIdx.x))];` |
| compute.reduce | 0.29 | 10.6% | 1.0 | 0.32 | `row_max[0] = vllm_max(row_max[0], __shfl_xor_sync((uint)4294967295, row_max[0], 8, 32));` |
| store.global | 0.16 | 5.9% | 1.0 | 0.29 | `if (((int)threadIdx.x) == 0)` |
| wait.sync#3 | 0.09 | 3.1% | 1.0 | 0.13 | `__syncthreads();` |
| load.global#2 | 0.06 | 2.4% | 1.0 | 0.06 | `target = local_argmax[((((int64_t)row) * argmax_stride) + ((int64_t)best_index[0]))];` |
| wait.sync#2 | 0.06 | 2.4% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync | 0.05 | 2.0% | 1.0 | 0.06 | `__syncthreads();` |

### cache_writer_kernel grid (8, 1, 1) block (128, 1, 1)

51 launches, 32 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 1.02 us, warp 0.94 us, nsys graphs 1.12 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.33 | 35.5% | 0.33 | 33.1% |
| load | 0.29 | 30.4% | 0.30 | 29.4% |
| store | 0.24 | 26.0% | 0.30 | 30.1% |
| compute | 0.08 | 8.1% | 0.08 | 7.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.24 | 26.0% | 1.0 | 0.32 | `if (((int64_t)0 <= slot) & ((slot >> (int64_t)7) < ((int64_t)pages))) \| if (((int64_t)0 <=` |
| load.global | 0.23 | 24.3% | 1.0 | 0.26 | `for (int i = 0; i < 4; ++i)` |
| compute.reduce | 0.08 | 8.1% | 1.0 | 0.10 | `amax = max(amax, __shfl_xor_sync((uint)4294967295, amax, 1, 32));` |
| load.global#2 | 0.06 | 6.2% | 1.0 | 0.06 | `int64_t slot = slots[((int64_t)((int)blockIdx.x))];` |

### metadata_pages_kernel grid (5, 1, 1) block (128, 1, 1)

32 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 1.31 us, warp 0.95 us, nsys graphs 1.50 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.72 | 76.0% | 0.79 | 49.5% |
| load | 0.14 | 15.1% | 0.19 | 11.9% |
| store | 0.08 | 8.9% | 0.62 | 38.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.13 | 13.4% | 1.0 | 0.13 | `if ((((offset + ((int)blockIdx.x)) < src_rows) && (0 <= (offset + ((int)blockIdx.x)))))` |
| store.global | 0.08 | 8.9% | 0.2 | 0.35 | `if (((int)threadIdx.x) < 32)` |
| load.global#2 | 0.02 | 1.7% | 0.2 | 0.06 | `if (((req < table_rows) && (((int)threadIdx.x) < table_cols)))` |

### sink_policy_kernel grid (1, 1, 1) block (32, 1, 1)

17 launches, 1 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 2.50 us, warp 2.50 us, nsys graphs 2.62 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.09 | 43.6% | 1.11 | 37.0% |
| unattributed | 0.83 | 33.3% | 1.32 | 44.0% |
| store | 0.58 | 23.1% | 0.57 | 19.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 | 0.64 | 25.6% | 1.0 | 0.64 | `state[0] = idx_mapping[((int64_t)((int)blockIdx.x))];` |
| store.global#3 | 0.51 | 20.5% | 1.0 | 0.51 | `for (int a = 0; a < 3; ++a) \| recovered_logprobs[((int64_t)((int)blockIdx.x))] = 0x0p+0f/*` |
| load.global | 0.22 | 9.0% | 1.0 | 0.22 | `start[0] = cu_num_logits[((int64_t)((int)blockIdx.x))];` |
| load.global#4 | 0.16 | 6.4% | 1.0 | 0.16 | `for (int r = 0; r < 2; ++r) \| for (int r_2 = 0; r_2 < 8; ++r_2)` |
| load.global#5 | 0.06 | 2.6% | 1.0 | 0.06 | `AtomicAdd((&(counts[0])), draft_zero_flag[0]);` |
| store.global#4 | 0.06 | 2.6% | 1.0 | 0.06 | `events[(((int64_t)((int)blockIdx.x)) * (int64_t)6)] = outcome[0];` |

### sparse_mla_kernel grid (1, 2, 6) block (128, 1, 1)

2 launches, 48 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 23.55 us, warp 23.39 us, nsys graphs 15.17 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed, with loops' unsampled iterations | 6.92 | 29.6% | 5.90 | 25.1% |
| load | 6.24 | 26.7% | 9.89 | 42.0% |
| wait | 5.18 | 22.1% | 2.38 | 10.1% |
| compute | 4.73 | 20.2% | 4.86 | 20.7% |
| store | 0.33 | 1.4% | 0.51 | 2.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 | 4.31 | 18.4% | 1.0 | 5.71 | `for (int i = 0; i < 8; ++i)` |
| wait.sync (sampled iterations) | 2.71 | 11.6% | 1.0 | 3.62 | `__syncthreads();` |
| compute.mma*4 (est.: sampled iterations x4) | 1.76 | 7.5% | 1.0 | 1.79 | `for (int ki = 0; ki < 32; ++ki) \| {` |
| compute.mma#2*4 (est.: sampled iterations x4) | 1.16 | 5.0% | 1.0 | 1.22 | `for (int ki_1 = 0; ki_1 < 2; ++ki_1) \| if (((((int)blockIdx.x) * 2) + (c_1 >> 1)) < 3)` |
| load.global (sampled iterations) | 0.95 | 4.1% | 1.0 | 3.65 | `for (int c = 0; c < 4; ++c) \| for (int i = 0; i < 8; ++i) \| for (int i_1 = 0; i_1 < 8; ++i` |
| compute.reduce#2*8 (est.: sampled iterations x8) | 0.92 | 3.9% | 1.0 | 1.02 | `row_sum[i_7] = tl::AllReduce<tl::SumOp, 128, 32, 0, tl::NamedBarrier<128>>::run(row_sum[i_` |
| compute.reduce*8 (est.: sampled iterations x8) | 0.90 | 3.8% | 1.0 | 1.02 | `m_g[i_4] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(m_g[i_4], (&((` |
| load.cp_async#2*4 (est.: sampled iterations x4) | 0.64 | 2.7% | 1.0 | 0.64 | `if (((((int)blockIdx.x) * 2) + (c_1 >> 1)) < 3) \| if ((c_1 < 3) & (c_1 < 3)) \| if (c < 3)` |
| wait.sync#8*8 (est.: sampled iterations x8) | 0.34 | 1.5% | 1.0 | 0.51 | `__syncthreads();` |
| load.cp_async (sampled iterations) | 0.33 | 1.4% | 1.0 | 0.35 | `for (int k_1 = 0; k_1 < 5; ++k_1) \| for (int k_1 = 0; k_1 < 9; ++k_1) \| for (int k_3 = 0; ` |
| store.global | 0.33 | 1.4% | 1.0 | 0.51 | `for (int i_10 = 0; i_10 < 32; ++i_10) \| for (int i_11 = 0; i_11 < 32; ++i_11) \| for (int i` |
| wait.sync#10*8 (est.: sampled iterations x8) | 0.30 | 1.3% | 1.0 | 0.51 | `__syncthreads();` |
| wait.cp_async*4 (est.: sampled iterations x4) | 0.30 | 1.3% | 1.0 | 0.32 | `tl::cp_async_wait<0>();` |
| wait.sync#7*4 (est.: sampled iterations x4) | 0.26 | 1.1% | 1.0 | 0.32 | `__syncthreads();` |
| wait.sync#11*4 (est.: sampled iterations x4) | 0.25 | 1.1% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#5*4 (est.: sampled iterations x4) | 0.25 | 1.1% | 1.0 | 0.32 | `__syncthreads();` |
| wait.sync#9*4 (est.: sampled iterations x4) | 0.23 | 1.0% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#4*4 (est.: sampled iterations x4) | 0.22 | 1.0% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#6*4 (est.: sampled iterations x4) | 0.21 | 0.9% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#3 | 0.06 | 0.3% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#2 | 0.05 | 0.2% | 1.0 | 0.06 | `__syncthreads();` |

### compute_slot_mappings_kernel grid (1, 11, 1) block (256, 1, 1)

17 launches, 88 warps; ranges, warp and last warp from the 6 recording CTAs' 48 warps; span 2.72 us, warp 1.46 us, nsys graphs 3.04 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.80 | 54.8% | 0.80 | 29.7% |
| unattributed | 0.66 | 45.2% | 1.89 | 70.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.80 | 54.8% | 1.0 | 0.83 | `int first_token = query_start_loc[0];` |

### cache_writer_kernel grid (6, 1, 1) block (128, 1, 1)

8 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 7.42 us, warp 6.81 us, nsys graphs 4.51 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 3.98 | 58.5% | 2.28 | 40.0% |
| store | 2.46 | 36.2% | 2.18 | 38.3% |
| load | 0.29 | 4.3% | 1.16 | 20.4% |
| compute | 0.07 | 1.0% | 0.08 | 1.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 2.46 | 36.2% | 1.0 | 3.49 | `if (((int64_t)0 <= slot) & ((slot >> (int64_t)7) < ((int64_t)pages))) \| if (((int64_t)0 <=` |
| load.global | 0.23 | 3.4% | 1.0 | 0.26 | `for (int i = 0; i < 4; ++i)` |
| compute.reduce | 0.07 | 1.0% | 1.0 | 0.10 | `amax = max(amax, __shfl_xor_sync((uint)4294967295, amax, 1, 32));` |
| load.global#2 | 0.06 | 0.9% | 1.0 | 0.06 | `int64_t slot = slots[((int64_t)((int)blockIdx.x))];` |

### sink_rows_kernel grid (1, 64, 1) block (256, 1, 1)

17 launches, 512 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 1.54 us, warp 0.97 us, nsys graphs 1.68 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.39 | 40.4% | 0.71 | 27.3% |
| load | 0.31 | 32.1% | 0.43 | 16.7% |
| store | 0.27 | 27.5% | 1.46 | 56.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.31 | 31.6% | 1.0 | 0.38 | `row = (max((((int64_t)num_sampled[((int64_t)((int)blockIdx.x))]) - (int64_t)1), (int64_t)0` |
| store.global | 0.26 | 26.5% | 1.0 | 0.48 | `for (int k_1 = 0; k_1 < 8; ++k_1)` |
| store.global#2 | 0.01 | 1.0% | 0.0 | 0.32 | `tokens[((((int64_t)((int)blockIdx.x)) * (int64_t)3) + ((int64_t)((int)threadIdx.x)))] = to` |
| load.global#2 | 0.00 | 0.4% | 0.0 | 0.13 | `token = max(alternates[((((int64_t)((int)blockIdx.x)) * (int64_t)3) + ((int64_t)((int)thre` |

### norm_forward_and_per_token_cast_kernel_kernel grid (10, 1, 1) block (128, 1, 1)

17 launches, 40 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 2.37 us, warp 2.30 us, nsys graphs 1.82 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 2.13 | 92.6% | 2.17 | 92.5% |
| load | 0.43 | 18.8% | 0.43 | 18.5% |
| compute | 0.13 | 5.6% | 0.13 | 5.5% |
| wait | 0.06 | 2.5% | 0.05 | 2.2% |
| unattributed | 0.00 | 0.0% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*5 (est.: sampled iterations x5) | 1.86 | 80.8% | 1.0 | 2.08 | `for (int j_1 = 0; j_1 < 2; ++j_1)` |
| load.global#2*5 (est.: sampled iterations x5) | 0.22 | 9.4% | 1.0 | 0.32 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 1024) + (((int)threadIdx.x) * 8)` |
| load.global*5 (est.: sampled iterations x5) | 0.22 | 9.4% | 1.0 | 0.32 | `*(uint4*)(x_local + (vi * 8)) = *(uint4*)(x + ((((((int64_t)vi) * (int64_t)1024) + (((int6` |
| store.global*5 (est.: sampled iterations x5) | 0.22 | 9.4% | 1.0 | 0.32 | `for (int j = 0; j < 4; ++j)` |
| compute.reduce | 0.13 | 5.6% | 1.0 | 0.16 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| wait.sync | 0.06 | 2.5% | 1.0 | 0.10 | `__syncthreads();` |
| store.global#2 | 0.04 | 1.9% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| store.global#3 | 0.01 | 0.5% | 0.2 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |

### resample_kernel grid (1, 127, 1) block (128, 1, 1)

17 launches, 508 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 1.95 us, warp 1.60 us, nsys graphs 2.24 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.27 | 79.3% | 1.98 | 85.3% |
| unattributed | 0.29 | 17.9% | 0.29 | 12.7% |
| compute | 0.04 | 2.8% | 0.05 | 2.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.27 | 79.3% | 1.0 | 1.60 | `end = cu_num_logits[(((int64_t)((int)blockIdx.x)) + (int64_t)1)];` |
| compute.reduce | 0.04 | 2.8% | 1.0 | 0.06 | `if ((temp != 0x0p+0f/*0.000000e+00*/) \| bonus)` |

### sink_flags_kernel grid (1, 1, 1) block (128, 1, 1)

17 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 2.21 us, warp 0.72 us, nsys graphs 3.14 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.39 | 54.4% | 1.55 | 70.4% |
| unattributed | 0.24 | 33.3% | 0.31 | 14.2% |
| store | 0.09 | 12.2% | 0.34 | 15.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.39 | 54.4% | 0.2 | 1.57 | `token = logits_indices[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)th` |
| store.global | 0.09 | 12.2% | 0.2 | 0.35 | `flags[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = (` |

### dspark_confidence_kernel grid (10, 1, 1) block (128, 1, 1)

17 launches, 40 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 1.86 us, warp 1.64 us, nsys graphs 1.86 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.27 | 77.1% | 1.55 | 76.8% |
| load | 0.17 | 10.6% | 0.21 | 10.6% |
| compute | 0.09 | 5.8% | 0.09 | 4.6% |
| wait | 0.08 | 5.2% | 0.07 | 3.3% |
| store | 0.02 | 1.4% | 0.09 | 4.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.16 | 9.9% | 1.0 | 0.19 | `for (int c = 0; c < 5; ++c)` |
| compute.reduce | 0.09 | 5.8% | 1.0 | 0.10 | `acc[0] = tl::warp_reduce_sum(acc[0]);` |
| wait.sync | 0.08 | 5.2% | 1.0 | 0.19 | `__syncthreads();` |
| store.global | 0.02 | 1.4% | 0.2 | 0.10 | `out[((int64_t)((int)blockIdx.x))] = (0x1p+0f/*1.000000e+00*/ / (0x1p+0f/*1.000000e+00*/ + ` |
| load.global#2 | 0.01 | 0.7% | 0.2 | 0.06 | `total[0] = 0x0p+0f/*0.000000e+00*/;` |

### post_update_kernel grid (1, 1, 1) block (32, 1, 1)

17 launches, 1 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 1.82 us, warp 1.82 us, nsys graphs 2.50 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.64 | 35.1% | 0.66 | 32.8% |
| unattributed | 0.61 | 33.3% | 0.56 | 28.0% |
| store | 0.58 | 31.6% | 0.79 | 39.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 | 0.48 | 26.3% | 1.0 | 0.48 | `int count = num_sampled[((((int64_t)((int)blockIdx.x)) * (int64_t)32) + ((int64_t)((int)th` |
| store.global#2 | 0.42 | 22.8% | 1.0 | 0.42 | `if (0 < count)` |
| store.global | 0.16 | 8.8% | 1.0 | 0.16 | `if (0 < count)` |
| load.global#3 | 0.10 | 5.3% | 1.0 | 0.10 | `if ((0 < count))` |
| load.global | 0.06 | 3.5% | 1.0 | 0.06 | `int64_t req = idx_mapping[((((int64_t)((int)blockIdx.x)) * (int64_t)32) + ((int64_t)((int)` |

### gather_block_tables_kernel grid (11, 1, 1) block (256, 1, 1)

17 launches, 88 warps; ranges, warp and last warp from the 6 recording CTAs' 48 warps; span 1.95 us, warp 1.36 us, nsys graphs 2.06 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.07 | 78.2% | 1.64 | 84.3% |
| unattributed | 0.30 | 21.8% | 0.30 | 15.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.07 | 78.2% | 1.0 | 1.63 | `int g = chunk_groups[((int64_t)((int)blockIdx.x))];` |

### rope_kernel grid (5, 1, 1) block (256, 1, 1)

32 launches, 40 warps; ranges, warp and last warp from the 5 recording CTAs' 40 warps; span 1.06 us, warp 0.94 us, nsys graphs 1.15 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.64 | 67.7% | 0.71 | 69.5% |
| load | 0.26 | 28.0% | 0.27 | 26.3% |
| store | 0.04 | 4.3% | 0.04 | 4.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.26 | 28.0% | 1.0 | 0.29 | `if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < tokens)) \| int64_t position ` |
| store.global | 0.04 | 4.3% | 1.0 | 0.06 | `*(uint4*)(x + ((((((int64_t)((int)blockIdx.x)) * (int64_t)12288) + ((((int64_t)((int)threa` |

### topk_ids_kernel grid (6, 1, 1) block (512, 1, 1)

1 launches, 96 warps; ranges, warp and last warp from the 6 recording CTAs' 96 warps; span 30.69 us, warp 28.10 us, nsys graphs 25.18 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 24.05 | 85.6% | 25.82 | 84.2% |
| unattributed | 1.98 | 7.1% | 1.92 | 6.3% |
| wait | 1.54 | 5.5% | 1.47 | 4.8% |
| store | 0.53 | 1.9% | 1.47 | 4.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 24.05 | 85.6% | 1.0 | 27.10 | `for (int t = 0; t < (vocab >> 14); ++t)` |
| wait.sync | 1.41 | 5.0% | 1.0 | 3.90 | `__syncthreads();` |
| store.global | 0.53 | 1.9% | 1.0 | 6.18 | `if ((((int)threadIdx.x) >> 5) == 0)` |
| wait.sync#2 | 0.08 | 0.3% | 1.0 | 0.19 | `__syncthreads();` |
| wait.sync#3 | 0.05 | 0.2% | 1.0 | 0.06 | `__syncthreads();` |

### logits_rows_kernel grid (1, 1, 1) block (128, 1, 1)

17 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 1.18 us, warp 1.14 us, nsys graphs 2.08 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.51 | 45.1% | 0.54 | 31.4% |
| unattributed | 0.50 | 43.7% | 0.30 | 17.2% |
| store | 0.13 | 11.3% | 0.89 | 51.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.51 | 45.1% | 1.0 | 0.51 | `index = logits_indices[min(((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((in` |
| store.global | 0.13 | 11.3% | 1.0 | 0.38 | `if (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) < num_logits)` |

### insert_kernel grid (1, 1, 1) block (32, 1, 1)

17 launches, 1 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 1.57 us, warp 1.57 us, nsys graphs 1.38 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.70 | 44.9% | 0.77 | 44.6% |
| unattributed | 0.42 | 26.5% | 0.43 | 24.7% |
| store | 0.42 | 26.5% | 0.49 | 28.3% |
| compute | 0.03 | 2.0% | 0.04 | 2.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#3 | 0.51 | 32.7% | 1.0 | 0.51 | `int state = mapping[((int64_t)position)]; \| int64_t state = mapping[((int64_t)position)];` |
| store.global | 0.42 | 26.5% | 1.0 | 0.42 | `counts[((int64_t)((int)blockIdx.x))] = (count_local + 1); \| counts[((int64_t)((int)blockId` |
| load.global | 0.13 | 8.2% | 1.0 | 0.13 | `for (int k = 0; k < 4; ++k)` |
| load.global#2 | 0.06 | 4.1% | 1.0 | 0.06 | `count_local = counts[((int64_t)((int)blockIdx.x))];` |
| compute.reduce | 0.03 | 2.0% | 1.0 | 0.03 | `int count = __shfl_sync((uint)4294967295, count_local, 0, 32);` |

### gemm_splitk_reduce_kernel grid (4, 8, 1) block (128, 1, 1)

51 launches, 128 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 0.54 us, warp 0.53 us, nsys graphs 0.74 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.25 | 48.1% | 0.26 | 47.6% |
| load | 0.23 | 43.9% | 0.24 | 44.6% |
| store | 0.04 | 8.0% | 0.04 | 7.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.23 | 43.9% | 1.0 | 0.26 | `acc[0] = P[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x))` |
| store.global | 0.04 | 8.0% | 1.0 | 0.06 | `C[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x)) * (int64` |

### assign_draft_budget_kernel grid (1, 1, 1) block (256, 1, 1)

16 launches, 8 warps; ranges, warp and last warp from the 1 recording CTAs' 8 warps; span 1.73 us, warp 1.73 us, nsys graphs 1.86 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 0.76 | 44.1% | 0.30 | 17.1% |
| unattributed, with loops' unsampled iterations | 0.55 | 31.6% | 0.64 | 37.0% |
| load | 0.23 | 13.2% | 0.61 | 34.8% |
| store | 0.19 | 11.1% | 0.19 | 11.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync | 0.51 | 29.7% | 1.0 | 0.59 | `__syncthreads();` |
| load.global | 0.23 | 13.2% | 1.0 | 0.70 | `for (int chunk = 0; chunk < ((num_reqs + 255) >> 8); ++chunk)` |
| store.global | 0.19 | 11.1% | 1.0 | 0.19 | `for (int chunk_2 = 0; chunk_2 < ((num_reqs + 255) >> 8); ++chunk_2)` |
| wait.sync#3 | 0.18 | 10.6% | 1.0 | 0.22 | `__syncthreads();` |
| wait.sync#2 (sampled iterations) | 0.06 | 3.7% | 1.0 | 0.06 | `__syncthreads();` |

### context_inputs_kernel grid (5, 8, 1) block (128, 1, 1)

17 launches, 160 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 1.54 us, warp 1.22 us, nsys graphs 1.84 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.80 | 65.5% | 1.17 | 73.3% |
| unattributed | 0.34 | 28.1% | 0.35 | 22.0% |
| store | 0.08 | 6.4% | 0.08 | 4.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.80 | 65.5% | 1.0 | 1.09 | `vllm_dspark_context_b128((&(aux[(((((int64_t)((int)blockIdx.y)) * (int64_t)15360) + (((int` |
| store.global | 0.08 | 6.4% | 1.0 | 0.16 | `if (((((int)blockIdx.x) == 0) & (((int)threadIdx.x) == 0)) & (num_tokens <= ((int)blockIdx` |

### metadata_decode_kernel grid (6, 2, 1) block (128, 1, 1)

4 launches, 48 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 5.60 us, warp 3.38 us, nsys graphs 1.78 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 2.11 | 62.4% | 4.63 | 74.4% |
| unattributed | 1.16 | 34.4% | 1.21 | 19.4% |
| store | 0.11 | 3.2% | 0.38 | 6.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.51 | 44.5% | 0.5 | 3.39 | `if ((((offset + ((int)blockIdx.x)) < src_rows) && (0 <= (offset + ((int)blockIdx.x)))))` |
| load.global#3 | 0.58 | 17.1% | 0.5 | 1.20 | `if ((((offset + ((int)blockIdx.x)) < main_src_rows) && (0 <= (offset + ((int)blockIdx.x)))` |
| store.global#4 | 0.07 | 2.0% | 0.5 | 0.35 | `if (0 < live)` |
| store.global | 0.03 | 0.9% | 0.5 | 0.06 | `swa[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = con` |
| load.global#2 | 0.03 | 0.8% | 0.1 | 0.21 | `if (((0 <= req) & ((int64_t)0 <= pos)))` |
| store.global#2 | 0.01 | 0.2% | 0.1 | 0.06 | `lengths[((int64_t)((int)blockIdx.x))] = ((int)max(info[1], (int64_t)0));` |
| store.global#3 | 0.00 | 0.1% | 0.1 | 0.03 | `top_lengths[((int64_t)((int)blockIdx.x))] = condval_6; \| top_lengths[((int64_t)((int)block` |

### combine_sampled_and_draft_tokens_kernel grid (1, 1, 1) block (32, 1, 1)

17 launches, 1 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 1.41 us, warp 1.41 us, nsys graphs 1.86 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.90 | 63.6% | 0.91 | 64.6% |
| unattributed | 0.42 | 29.5% | 0.41 | 29.0% |
| store | 0.10 | 6.8% | 0.09 | 6.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.90 | 63.6% | 1.0 | 0.90 | `if (((((int)blockIdx.x) * 32) + ((int)threadIdx.x)) < num_reqs) \| req = idx_mapping[((int6` |
| store.global | 0.10 | 6.8% | 1.0 | 0.10 | `if ((prefill < seq) && (1 < num_logits))` |

### mhc_post_kernel grid (50, 1, 1) block (128, 1, 1)

17 launches, 200 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 1.34 us, warp 1.06 us, nsys graphs 1.57 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.89 | 83.7% | 1.10 | 83.9% |
| load | 0.12 | 11.2% | 0.14 | 10.6% |
| store | 0.05 | 5.1% | 0.07 | 5.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.12 | 11.2% | 1.0 | 0.16 | `*(float4*)(pm + 0) = *(float4*)(post + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int` |
| store.global | 0.05 | 5.1% | 1.0 | 0.10 | `for (int n_1 = 0; n_1 < 4; ++n_1)` |

### per_token_cast_kernel_kernel grid (60, 1, 1) block (64, 1, 1)

17 launches, 120 warps; ranges, warp and last warp from the 4 recording CTAs' 8 warps; span 1.18 us, warp 1.13 us, nsys graphs 1.31 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.40 | 35.5% | 0.47 | 35.8% |
| unattributed | 0.34 | 29.8% | 0.46 | 34.9% |
| store | 0.30 | 26.2% | 0.29 | 22.2% |
| compute | 0.10 | 8.5% | 0.09 | 7.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.31 | 27.7% | 1.0 | 0.35 | `for (int i = 0; i < 2; ++i)` |
| store.global#2 | 0.11 | 9.9% | 1.0 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.10 | 8.5% | 1.0 | 0.13 | `amax_fragment[i_1] = tl::AllReduce<tl::MaxOp, 2, 1, 0, tl::NamedBarrier<64>>::run(amax_fra` |
| store.global*2 (est.: sampled iterations x2) | 0.10 | 8.5% | 1.0 | 0.13 | `out_sf[((((((int64_t)((int)blockIdx.x)) % (int64_t)15) * (int64_t)32) + (((int64_t)((int)t` |
| load.global#2*2 (est.: sampled iterations x2) | 0.09 | 7.8% | 1.0 | 0.13 | `amax_fragment[i_1] = cutlass::bfloat16_t(__hmax((bfloat16_t(((nv_bfloat162*)(&(amax_fragme` |
| store.global#3*2 (est.: sampled iterations x2) | 0.09 | 7.8% | 1.0 | 0.13 | `*(fp8_e4_16_t*)(out + ((((((((int64_t)((int)blockIdx.x)) / (int64_t)3) * (int64_t)6144) + ` |

### moe_swiglu_cast_kernel_kernel grid (16, 36, 1) block (96, 1, 1)

6 launches, 1728 warps; ranges, warp and last warp from the 4 recording CTAs' 12 warps; span 3.71 us, warp 1.28 us, nsys graphs 1.86 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.00 | 77.9% | 3.18 | 85.5% |
| load | 0.21 | 16.7% | 0.26 | 7.0% |
| compute | 0.04 | 2.9% | 0.15 | 4.0% |
| store | 0.03 | 2.5% | 0.13 | 3.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.21 | 16.7% | 1.0 | 0.27 | `for (int r = 0; r < 2; ++r)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.04 | 2.9% | 0.2 | 0.19 | `if (0 <= orders[r_2])` |
| store.global*2 (est.: sampled iterations x2) | 0.03 | 2.5% | 0.2 | 0.13 | `if (0 <= orders[r_2])` |

### indexer_topk_segment_kernel grid (1, 6, 1) block (512, 1, 1)

2 launches, 96 warps; ranges, warp and last warp from the 6 recording CTAs' 96 warps; span 10.69 us, warp 10.62 us, nsys graphs 5.57 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed, with loops' unsampled iterations | 5.31 | 50.0% | 5.58 | 52.2% |
| wait | 4.06 | 38.2% | 3.81 | 35.6% |
| load | 1.07 | 10.1% | 1.09 | 10.2% |
| store | 0.18 | 1.7% | 0.21 | 1.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.07 | 10.1% | 1.0 | 1.12 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 4096); \| i` |
| wait.sync (sampled iterations) | 0.99 | 9.3% | 14.5 | 1.20 | `__syncthreads();` |
| wait.sync#20*9 (est.: sampled iterations x9) | 0.35 | 3.3% | 0.5 | 0.43 | `__syncthreads();` |
| wait.sync#6*8 (est.: sampled iterations x8) | 0.35 | 3.3% | 0.5 | 0.38 | `__syncthreads();` |
| wait.sync#24*9 (est.: sampled iterations x9) | 0.35 | 3.3% | 0.5 | 0.43 | `__syncthreads();` |
| wait.sync#19*9 (est.: sampled iterations x9) | 0.32 | 3.0% | 0.5 | 0.43 | `__syncthreads();` |
| wait.sync#13*8 (est.: sampled iterations x8) | 0.31 | 2.9% | 0.5 | 0.38 | `__syncthreads();` |
| wait.sync#23*9 (est.: sampled iterations x9) | 0.30 | 2.8% | 0.5 | 0.43 | `__syncthreads();` |
| wait.sync#14*8 (est.: sampled iterations x8) | 0.29 | 2.7% | 0.5 | 0.38 | `__syncthreads();` |
| wait.sync#5*8 (est.: sampled iterations x8) | 0.28 | 2.6% | 0.5 | 0.38 | `__syncthreads();` |
| store.global | 0.18 | 1.7% | 1.5 | 0.24 | `for (int i_3 = 0; i_3 < 4; ++i_3) \| if (((int)threadIdx.x) < total) \| if (((int)threadIdx.` |
| wait.sync#25 | 0.05 | 0.4% | 0.5 | 0.08 | `__syncthreads();` |
| wait.sync#17 | 0.04 | 0.4% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#16 | 0.04 | 0.4% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#18 | 0.04 | 0.4% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#8 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#7 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#11 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#3 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#2 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#4 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#9 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#22 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#15 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#10 | 0.03 | 0.3% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#21 | 0.03 | 0.3% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#12 | 0.03 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |

### token_counts_kernel grid (1, 1, 1) block (128, 1, 1)

17 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 1.12 us, warp 0.47 us, nsys graphs 1.30 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.27 | 57.6% | 0.35 | 29.4% |
| store | 0.14 | 30.5% | 0.63 | 52.5% |
| load | 0.06 | 11.9% | 0.22 | 18.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.14 | 30.5% | 0.2 | 0.58 | `if (((int)threadIdx.x) == 0) \| sampled[((int64_t)((int)blockIdx.x))] = (((int)(prefill_len` |
| load.global | 0.06 | 11.9% | 0.2 | 0.22 | `int count = sampled[((int64_t)((int)blockIdx.x))];` |

### flatten_sampled_kernel grid (1, 1, 1) block (128, 1, 1)

17 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 1.12 us, warp 0.82 us, nsys graphs 1.28 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.47 | 57.3% | 0.63 | 56.2% |
| unattributed | 0.33 | 39.8% | 0.41 | 36.4% |
| store | 0.02 | 2.9% | 0.08 | 7.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.42 | 50.5% | 1.0 | 0.42 | `for (int r = 0; r < num_reqs; ++r)` |
| load.global#2 | 0.06 | 6.8% | 0.2 | 0.22 | `int i = (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) - cu_num_logits[req]);` |
| store.global | 0.02 | 2.9% | 0.2 | 0.10 | `flat[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = co` |

### prepare_pos_seq_lens_kernel grid (2, 1, 1) block (256, 1, 1)

17 launches, 16 warps; ranges, warp and last warp from the 2 recording CTAs' 16 warps; span 1.09 us, warp 0.75 us, nsys graphs 1.33 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.32 | 42.4% | 0.36 | 32.7% |
| load | 0.30 | 40.3% | 0.60 | 55.1% |
| store | 0.13 | 17.2% | 0.13 | 12.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.30 | 40.3% | 0.5 | 0.61 | `int computed = num_computed_tokens[idx_mapping[((int64_t)((int)blockIdx.x))]];` |
| store.global | 0.07 | 8.8% | 0.5 | 0.16 | `for (int it = 0; it < (((max_reqs + 255) - num_reqs) >> 8); ++it)` |
| store.global#2 | 0.06 | 8.5% | 0.5 | 0.13 | `if (((int)threadIdx.x) == 0)` |

### topk_log_softmax_kernel grid (6, 1, 1) block (1024, 1, 1)

1 launches, 192 warps; ranges, warp and last warp from the 6 recording CTAs' 192 warps; span 16.86 us, warp 16.07 us, nsys graphs 16.32 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 13.08 | 81.4% | 13.95 | 82.9% |
| unattributed, with loops' unsampled iterations | 1.58 | 9.9% | 1.73 | 10.3% |
| wait | 1.01 | 6.3% | 0.74 | 4.4% |
| compute | 0.39 | 2.4% | 0.35 | 2.1% |
| store | 0.00 | 0.0% | 0.06 | 0.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 6.62 | 41.2% | 1.0 | 6.94 | `if (0 < (vocab & 2047)) \| if (0 < (vocab & 8191))` |
| load.global#2 | 6.45 | 40.1% | 1.0 | 7.01 | `for (int step_1 = 0; step_1 < (vocab >> 11); ++step_1) \| for (int step_1 = 0; step_1 < (vo` |
| wait.sync#2 | 0.45 | 2.8% | 1.0 | 0.96 | `__syncthreads();` |
| wait.sync | 0.31 | 1.9% | 1.0 | 1.15 | `__syncthreads();` |
| compute.reduce#3 | 0.17 | 1.1% | 1.0 | 0.22 | `ties[0] = vllm_add(ties[0], __shfl_xor_sync((uint)4294967295, ties[0], 16, 32));` |
| compute.reduce#2 | 0.11 | 0.7% | 1.0 | 0.16 | `se[0] = vllm_add(se[0], __shfl_xor_sync((uint)4294967295, se[0], 16, 32));` |
| compute.reduce | 0.11 | 0.7% | 1.0 | 0.13 | `best[0] = vllm_max(best[0], __shfl_xor_sync((uint)4294967295, best[0], 16, 32));` |
| wait.sync#4 | 0.10 | 0.6% | 1.0 | 0.16 | `__syncthreads();` |
| wait.sync#3 | 0.08 | 0.5% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#5 | 0.08 | 0.5% | 1.0 | 0.10 | `__syncthreads();` |
| load.global#3 (sampled iterations) | 0.01 | 0.1% | 0.0 | 0.38 | `float value_32 = vllm_sub(vllm_shifted(logits[((((int64_t)((int)blockIdx.x)) * logits_stri` |
| store.global (sampled iterations) | 0.00 | 0.0% | 0.0 | 0.06 | `output[(((((int64_t)c) * (int64_t)1024) + (((int64_t)((int)blockIdx.x)) * ((int64_t)topk))` |

### scatter_rows_gather_kernel grid (1, 1, 1) block (128, 1, 1)

17 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 0.93 us, warp 0.83 us, nsys graphs 1.81 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.56 | 67.3% | 0.39 | 42.2% |
| store | 0.15 | 18.3% | 0.42 | 45.5% |
| load | 0.12 | 14.4% | 0.11 | 12.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.15 | 18.3% | 1.0 | 0.42 | `if (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) < (num_reqs * width))` |
| load.global | 0.12 | 14.4% | 1.0 | 0.13 | `index = idx_mapping[min((((((int64_t)0 <= ((int64_t)width)) && ((int64_t)0 <= rmod_1)) \|\| ` |

### ced_copy_rows_kernel grid (50, 1, 1) block (128, 1, 1)

17 launches, 200 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 0.90 us, warp 0.83 us, nsys graphs 1.15 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.57 | 69.0% | 0.62 | 70.2% |
| unattributed | 0.26 | 31.0% | 0.26 | 29.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.57 | 69.0% | 1.0 | 0.61 | `for (int k = 0; k < 4; ++k) \| int64_t index = ced_ldg64((&(row_indices[((((int64_t)blockId` |

### mhc_collapse_kernel grid (50, 1, 1) block (128, 1, 1)

17 launches, 200 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 0.86 us, warp 0.77 us, nsys graphs 1.06 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.62 | 80.2% | 0.71 | 80.6% |
| load | 0.11 | 13.7% | 0.12 | 13.4% |
| store | 0.05 | 6.0% | 0.05 | 6.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.11 | 13.7% | 1.0 | 0.13 | `*(float4*)(mv + 0) = *(float4*)(mix + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int6` |
| store.global | 0.05 | 6.0% | 1.0 | 0.06 | `*(uint4*)(out + ((((int64_t)((int)blockIdx.x)) * (int64_t)1024) + (((int64_t)((int)threadI` |

### metadata_chunk_kernel grid (6, 1, 1) block (128, 1, 1)

2 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 7.62 us, warp 5.60 us, nsys graphs 7.92 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 3.18 | 56.7% | 3.30 | 43.4% |
| load | 2.34 | 41.8% | 4.18 | 54.9% |
| store | 0.08 | 1.5% | 0.13 | 1.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.80 | 32.1% | 1.0 | 2.02 | `if ((((offset + ((int)blockIdx.x)) < src_rows) && (0 <= (offset + ((int)blockIdx.x)))))` |
| load.global#2 | 0.54 | 9.7% | 0.2 | 2.18 | `if (((0 <= req) & ((int64_t)0 <= pos)))` |
| store.global | 0.06 | 1.1% | 1.0 | 0.06 | `swa[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = con` |
| store.global#2 | 0.01 | 0.2% | 0.2 | 0.06 | `lengths[((int64_t)((int)blockIdx.x))] = ((int)max(info[1], (int64_t)0));` |
| store.global#3 | 0.01 | 0.1% | 0.2 | 0.03 | `top_lengths[((int64_t)((int)blockIdx.x))] = condval_12; \| top_lengths[((int64_t)((int)bloc` |

### sparse_mla_kernel_1 grid (24, 6, 1) block (128, 1, 1)

6 launches, 144 CTAs, 576 warps; span 2.13 us, warp 1.64 us, nsys graphs 2.21 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.60 | 97.9% | 1.27 | 53.2% |
| load | 0.02 | 1.4% | 0.94 | 39.4% |
| wait | 0.01 | 0.4% | 0.09 | 3.6% |
| compute | 0.00 | 0.1% | 0.06 | 2.7% |
| store | 0.00 | 0.1% | 0.03 | 1.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.02 | 1.4% | 0.1 | 1.12 | `M = __fmul_rn(Sinks[((int)blockIdx.x)], 0x1.7154764ee6c2fp+0f/*1.442695e+00*/); \| if (((in` |
| wait.sync | 0.01 | 0.4% | 0.1 | 0.42 | `__syncthreads();` |
| compute.reduce | 0.00 | 0.1% | 0.0 | 0.10 | `amax[0] = tl::AllReduce<tl::MaxOp, 8, 1, 0, tl::NamedBarrier<128>>::run(amax[0]);` |
| store.global | 0.00 | 0.1% | 0.0 | 0.06 | `*(uint2*)(Output + (((((int64_t)((int)blockIdx.y)) * (int64_t)12288) + (((int64_t)((int)bl` |
| compute.reduce#3 | 0.00 | 0.0% | 0.0 | 0.00 | `amax[0] = tl::AllReduce<tl::MaxOp, 8, 1, 0, tl::NamedBarrier<128>>::run(amax[0]);` |
| load.global#3 | 0.00 | 0.0% | 0.0 | 0.00 | `M = __fmul_rn(Sinks[((int)blockIdx.x)], 0x1.7154764ee6c2fp+0f/*1.442695e+00*/);` |
| load.global#4 | 0.00 | 0.0% | 0.0 | 0.00 | `if (((int)threadIdx.x) < 32)` |
| store.global#2 | 0.00 | 0.0% | 0.0 | 0.00 | `*(uint2*)(Output + (((((int64_t)((int)blockIdx.y)) * (int64_t)12288) + (((int64_t)((int)bl` |
| wait.sync#12 | 0.00 | 0.0% | 0.0 | 0.00 | `__syncthreads();` |
| wait.sync#13 | 0.00 | 0.0% | 0.0 | 0.00 | `__syncthreads();` |

### ced_requests_kernel grid (1, 1, 1) block (128, 1, 1)

17 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 0.86 us, warp 0.60 us, nsys graphs 1.18 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.31 | 52.0% | 0.32 | 37.2% |
| store | 0.16 | 26.7% | 0.43 | 49.4% |
| load | 0.13 | 21.3% | 0.12 | 13.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.16 | 26.7% | 1.0 | 0.42 | `if (((int)threadIdx.x) < 2) \| if (((int)threadIdx.x) <= 2)` |
| load.global | 0.13 | 21.3% | 1.0 | 0.13 | `uint v_ = tl::load_global_32((&(starts[1]))); \| uint v_ = tl::load_global_32((&(starts[min` |

### mhc_collapse_kernel grid (30, 1, 1) block (128, 1, 1)

3 launches, 120 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 4.99 us, warp 4.98 us, nsys graphs 2.16 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 4.83 | 97.0% | 3.88 | 91.7% |
| load | 0.10 | 2.1% | 0.30 | 7.1% |
| store | 0.05 | 0.9% | 0.05 | 1.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.10 | 2.1% | 1.0 | 0.13 | `*(float4*)(mv + 0) = *(float4*)(mix + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int6` |
| store.global | 0.05 | 0.9% | 1.0 | 0.06 | `*(uint4*)(out + ((((int64_t)((int)blockIdx.x)) * (int64_t)1024) + (((int64_t)((int)threadI` |

### moe_combine_kernel grid (5, 6, 1) block (128, 1, 1)

6 launches, 120 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 2.08 us, warp 1.57 us, nsys graphs 1.28 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.11 | 70.7% | 1.63 | 77.9% |
| unattributed | 0.40 | 25.6% | 0.40 | 19.1% |
| store | 0.06 | 3.7% | 0.06 | 3.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.11 | 70.7% | 1.0 | 1.62 | `for (int k = 0; k < 3; ++k) \| for (int k = 0; k < 6; ++k) \| if ((((int)blockIdx.y) < share` |
| store.global | 0.06 | 3.7% | 1.0 | 0.06 | `{` |

### mhc_collapse_column_kernel grid (25, 1, 1) block (128, 1, 1)

16 launches, 100 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 0.80 us, warp 0.78 us, nsys graphs 1.02 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.62 | 80.0% | 0.62 | 79.8% |
| load | 0.10 | 12.4% | 0.10 | 13.3% |
| store | 0.06 | 7.6% | 0.05 | 6.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.10 | 12.4% | 1.0 | 0.10 | `*(float4*)(mv + 0) = *(float4*)(mix + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int6` |
| store.global | 0.06 | 7.6% | 1.0 | 0.06 | `*(uint4*)(out + ((((int64_t)((int)blockIdx.x)) * (int64_t)1024) + (((int64_t)((int)threadI` |

### mhc_post_kernel grid (30, 1, 1) block (128, 1, 1)

4 launches, 120 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 2.26 us, warp 1.79 us, nsys graphs 1.38 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.63 | 91.2% | 2.24 | 92.4% |
| load | 0.11 | 6.2% | 0.13 | 5.3% |
| store | 0.05 | 2.6% | 0.06 | 2.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.11 | 6.2% | 1.0 | 0.13 | `*(float4*)(pm + 0) = *(float4*)(post + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int` |
| store.global | 0.05 | 2.6% | 1.0 | 0.06 | `for (int n_1 = 0; n_1 < 4; ++n_1)` |

### rope_kernel grid (6, 1, 1) block (192, 1, 1)

6 launches, 36 warps; ranges, warp and last warp from the 6 recording CTAs' 36 warps; span 1.82 us, warp 1.52 us, nsys graphs 1.50 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.98 | 64.5% | 1.54 | 75.4% |
| load | 0.50 | 33.0% | 0.46 | 22.5% |
| store | 0.04 | 2.6% | 0.04 | 2.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.50 | 33.0% | 1.0 | 0.51 | `if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < tokens)) \| int64_t position ` |
| store.global | 0.04 | 2.6% | 1.0 | 0.06 | `*(uint4*)(x + ((((((int64_t)((int)blockIdx.x)) * (int64_t)12288) + ((((int64_t)((int)threa` |

### padding_rows_kernel grid (1, 1, 1) block (256, 1, 1)

33 launches, 8 warps; ranges, warp and last warp from the 1 recording CTAs' 8 warps; span 0.35 us, warp 0.35 us, nsys graphs 0.77 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.26 | 72.7% | 0.26 | 72.2% |
| store | 0.10 | 27.3% | 0.10 | 27.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.10 | 27.3% | 1.0 | 0.10 | `if (((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) < rows)` |

### local_stats_kernel grid (6, 16, 1) block (128, 1, 1)

1 launches, 384 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 11.68 us, warp 11.63 us, nsys graphs 12.42 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 10.62 | 91.3% | 10.69 | 91.5% |
| unattributed | 0.70 | 6.0% | 0.64 | 5.5% |
| compute | 0.17 | 1.5% | 0.19 | 1.6% |
| wait | 0.07 | 0.6% | 0.06 | 0.5% |
| store | 0.07 | 0.6% | 0.10 | 0.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 10.62 | 91.3% | 1.0 | 10.69 | `if (expanded_local_pos[((int64_t)((int)blockIdx.x))] < num_speculative_steps)` |
| compute.reduce | 0.12 | 1.0% | 1.0 | 0.13 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| wait.sync | 0.07 | 0.6% | 1.0 | 0.10 | `__syncthreads();` |
| store.global | 0.07 | 0.6% | 1.0 | 0.13 | `if (((int)threadIdx.x) == 0)` |
| compute.reduce#2 | 0.06 | 0.5% | 1.0 | 0.06 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 2, 32);` |

### scatter_rows_kernel grid (1, 1, 1) block (128, 1, 1)

17 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 0.64 us, warp 0.59 us, nsys graphs 0.99 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.41 | 68.9% | 0.39 | 60.2% |
| load | 0.10 | 16.2% | 0.11 | 17.3% |
| store | 0.09 | 14.9% | 0.14 | 22.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.10 | 16.2% | 1.0 | 0.10 | `index = idx_mapping[min((((((int64_t)0 <= ((int64_t)width)) && ((int64_t)0 <= rmod_1)) \|\| ` |
| store.global | 0.09 | 14.9% | 1.0 | 0.16 | `if (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) < (num_reqs * width))` |

### ced_copy_rows_kernel grid (25, 1, 1) block (128, 1, 1)

16 launches, 100 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 0.67 us, warp 0.59 us, nsys graphs 1.22 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.36 | 60.9% | 0.40 | 61.4% |
| unattributed | 0.23 | 39.1% | 0.25 | 38.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.36 | 60.9% | 1.0 | 0.42 | `for (int k = 0; k < 4; ++k) \| int64_t index = ced_ldg64((&(row_indices[((((int64_t)blockId` |

### indexer_topk_segment_kernel grid (2, 6, 1) block (512, 1, 1)

1 launches, 192 warps; ranges, warp and last warp from the 4 recording CTAs' 64 warps; span 9.86 us, warp 5.18 us, nsys graphs 0.83 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 3.16 | 61.0% | 5.89 | 59.9% |
| unattributed, with loops' unsampled iterations | 1.64 | 31.7% | 3.30 | 33.6% |
| load | 0.34 | 6.6% | 0.51 | 5.2% |
| store | 0.04 | 0.8% | 0.13 | 1.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync#24*9 (est.: sampled iterations x9) | 0.39 | 7.5% | 0.5 | 0.86 | `__syncthreads();` |
| wait.sync#23*9 (est.: sampled iterations x9) | 0.34 | 6.6% | 0.5 | 0.86 | `__syncthreads();` |
| wait.sync#13*8 (est.: sampled iterations x8) | 0.33 | 6.3% | 0.5 | 0.77 | `__syncthreads();` |
| wait.sync#20*9 (est.: sampled iterations x9) | 0.32 | 6.1% | 0.5 | 0.86 | `__syncthreads();` |
| wait.sync#5*8 (est.: sampled iterations x8) | 0.31 | 6.0% | 0.5 | 0.77 | `__syncthreads();` |
| wait.sync#19*9 (est.: sampled iterations x9) | 0.30 | 5.8% | 0.5 | 0.86 | `__syncthreads();` |
| wait.sync#14*8 (est.: sampled iterations x8) | 0.30 | 5.8% | 0.5 | 0.77 | `__syncthreads();` |
| wait.sync#6*8 (est.: sampled iterations x8) | 0.29 | 5.6% | 0.5 | 0.77 | `__syncthreads();` |
| load.global | 0.21 | 4.1% | 1.0 | 0.22 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 4096); \| i` |
| load.global#2 | 0.13 | 2.5% | 0.5 | 0.32 | `for (int i = 0; i < 8; ++i)` |
| wait.sync (sampled iterations) | 0.05 | 1.0% | 0.5 | 0.16 | `__syncthreads();` |
| wait.sync#25 | 0.05 | 1.0% | 0.5 | 0.19 | `__syncthreads();` |
| wait.sync#17 | 0.05 | 0.9% | 0.5 | 0.10 | `__syncthreads();` |
| store.global | 0.04 | 0.8% | 0.5 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3) \| if (((int)threadIdx.x) < total) \| if (((int)threadIdx.` |
| wait.sync#16 | 0.04 | 0.8% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#18 | 0.04 | 0.8% | 0.5 | 0.13 | `__syncthreads();` |
| wait.sync#2 | 0.04 | 0.7% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#11 | 0.03 | 0.6% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#15 | 0.03 | 0.6% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#7 | 0.03 | 0.6% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#8 | 0.03 | 0.6% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#4 | 0.03 | 0.6% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#21 | 0.03 | 0.6% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#9 | 0.03 | 0.6% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#3 | 0.03 | 0.5% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#10 | 0.03 | 0.5% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#22 | 0.03 | 0.5% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#12 | 0.02 | 0.4% | 0.5 | 0.06 | `__syncthreads();` |

### indexer_topk_merge_kernel grid (1, 6, 1) block (512, 1, 1)

1 launches, 96 warps; ranges, warp and last warp from the 6 recording CTAs' 96 warps; span 9.18 us, warp 8.97 us, nsys graphs 3.79 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 6.09 | 67.9% | 5.98 | 65.2% |
| unattributed | 2.22 | 24.7% | 2.34 | 25.4% |
| load | 0.58 | 6.5% | 0.74 | 8.0% |
| store | 0.08 | 0.9% | 0.13 | 1.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync#23*9 (est.: sampled iterations x9) | 0.69 | 7.7% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#24*9 (est.: sampled iterations x9) | 0.68 | 7.6% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#20*9 (est.: sampled iterations x9) | 0.64 | 7.2% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#19*9 (est.: sampled iterations x9) | 0.63 | 7.0% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#5*8 (est.: sampled iterations x8) | 0.61 | 6.8% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync#6*8 (est.: sampled iterations x8) | 0.57 | 6.4% | 1.0 | 0.77 | `__syncthreads();` |
| load.global | 0.57 | 6.3% | 1.0 | 0.70 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 8192);` |
| wait.sync#14*8 (est.: sampled iterations x8) | 0.57 | 6.3% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync#13*8 (est.: sampled iterations x8) | 0.53 | 5.9% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync | 0.16 | 1.8% | 1.0 | 0.22 | `__syncthreads();` |
| wait.sync#17 | 0.08 | 0.9% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#16 | 0.08 | 0.9% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#8 | 0.08 | 0.8% | 1.0 | 0.10 | `__syncthreads();` |
| store.global#2 | 0.07 | 0.8% | 0.9 | 0.10 | `Values[((((int64_t)((int)blockIdx.y)) * (int64_t)512) + ((int64_t)((int)threadIdx.x)))] = ` |
| wait.sync#2 | 0.07 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#3 | 0.07 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#9 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#12 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#25 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#22 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#4 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#15 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#18 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#10 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#11 | 0.06 | 0.6% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#21 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#7 | 0.05 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| load.global#2 | 0.01 | 0.1% | 0.1 | 0.10 | `if (((0 <= ((int*)sel_idx)[((int)threadIdx.x)]) && (((int*)sel_idx)[((int)threadIdx.x)] < ` |
| store.global | 0.01 | 0.1% | 0.1 | 0.10 | `Values[((((int64_t)((int)blockIdx.y)) * (int64_t)512) + ((int64_t)((int)threadIdx.x)))] = ` |

### token_ranks_kernel grid (6, 1, 1) block (256, 1, 1)

1 launches, 48 warps; ranges, warp and last warp from the 6 recording CTAs' 48 warps; span 7.74 us, warp 7.67 us, nsys graphs 7.94 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 6.93 | 90.3% | 7.14 | 92.1% |
| unattributed | 0.36 | 4.7% | 0.35 | 4.5% |
| compute | 0.33 | 4.3% | 0.19 | 2.5% |
| store | 0.05 | 0.7% | 0.06 | 0.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 6.93 | 90.3% | 1.0 | 7.14 | `if ((((int64_t)0 <= token_ids[((int64_t)((int)blockIdx.x))]) && (token_ids[((int64_t)((int` |
| compute.reduce | 0.33 | 4.3% | 1.0 | 0.51 | `total[0] = tl::AllReduce<tl::SumOp, 256, 1, 0, tl::NamedBarrier<256>>::run(total[0], (&(wo` |
| store.global | 0.05 | 0.7% | 1.0 | 0.06 | `output[((int64_t)((int)blockIdx.x))] = ((int64_t)total[0]);` |

### indexer_mxfp4_quantize_kernel grid (6, 1, 1) block (128, 1, 1)

2 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 3.76 us, warp 3.68 us, nsys graphs 2.94 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 2.28 | 62.0% | 2.37 | 63.0% |
| store | 1.16 | 31.6% | 1.31 | 34.9% |
| unattributed | 0.24 | 6.4% | 0.08 | 2.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2*16 (est.: sampled iterations x16) | 1.26 | 34.2% | 1.0 | 1.54 | `float v__2 = __fmul_rn(((float)X[(((((int64_t)((int)blockIdx.x)) * (int64_t)4096) + (((int` |
| store.global*16 (est.: sampled iterations x16) | 1.12 | 30.4% | 1.0 | 1.54 | `Pool[(((((slot & (int64_t)127) * (int64_t)64) + ((((int64_t)((int)threadIdx.x)) & (int64_t` |
| load.global | 1.02 | 27.8% | 1.0 | 1.09 | `for (int i = 0; i < 32; ++i)` |
| store.global#2 | 0.04 | 1.2% | 1.0 | 0.06 | `Pool[(((((slot & (int64_t)127) * (int64_t)4) + ((slot >> (int64_t)7) * ((int64_t)pool_stri` |

### activation_swiglu_clamp_kernel grid (1, 6, 1) block (128, 1, 1)

6 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 0.93 us, warp 0.74 us, nsys graphs 1.82 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.63 | 86.0% | 0.79 | 86.1% |
| load | 0.07 | 9.7% | 0.09 | 9.8% |
| store | 0.03 | 4.3% | 0.04 | 4.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.07 | 9.7% | 0.8 | 0.10 | `*(uint4*)(gate + 0) = *(uint4*)(X + ((((int64_t)((int)blockIdx.y)) * (int64_t)1536) + (((i` |
| store.global | 0.03 | 4.3% | 0.8 | 0.06 | `*(uint4*)(Y + ((((int64_t)((int)blockIdx.y)) * (int64_t)768) + (((int64_t)((int)threadIdx.` |

### indexer_score_kernel grid (128, 6, 1) block (128, 1, 1)

1 launches, 3072 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 4.90 us, warp 2.45 us, nsys graphs 2.27 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.50 | 61.1% | 2.24 | 61.4% |
| unattributed, with loops' unsampled iterations | 0.73 | 29.7% | 1.06 | 28.9% |
| wait | 0.14 | 5.7% | 0.16 | 4.4% |
| compute | 0.06 | 2.5% | 0.13 | 3.5% |
| store | 0.02 | 0.9% | 0.06 | 1.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 (sampled iterations) | 0.98 | 39.9% | 0.5 | 2.21 | `for (int i_1 = 0; i_1 < 32; ++i_1)` |
| load.global | 0.44 | 18.1% | 1.0 | 0.70 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 4096); \| i` |
| load.global#3 (sampled iterations) | 0.08 | 3.1% | 0.5 | 0.26 | `if (((int)threadIdx.x) < 64)` |
| compute.mma (sampled iterations) | 0.06 | 2.5% | 0.5 | 0.13 | `for (int ki = 0; ki < 2; ++ki)` |
| wait.sync#2 (sampled iterations) | 0.06 | 2.5% | 0.5 | 0.22 | `__syncthreads();` |
| wait.sync (sampled iterations) | 0.03 | 1.4% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#3 (sampled iterations) | 0.02 | 1.0% | 0.5 | 0.06 | `__syncthreads();` |
| store.global (sampled iterations) | 0.02 | 0.9% | 0.5 | 0.06 | `if ((((int)threadIdx.x) >> 6) == 0)` |
| wait.sync#4 (sampled iterations) | 0.02 | 0.9% | 0.5 | 0.06 | `__syncthreads();` |

### indexer_score_kernel grid (64, 6, 1) block (128, 1, 1)

1 launches, 1536 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 4.70 us, warp 2.42 us, nsys graphs 1.70 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.18 | 48.8% | 2.05 | 50.0% |
| unattributed, with loops' unsampled iterations | 0.89 | 36.7% | 1.47 | 35.9% |
| wait | 0.24 | 9.8% | 0.42 | 10.2% |
| compute | 0.07 | 2.9% | 0.13 | 3.1% |
| store | 0.04 | 1.8% | 0.03 | 0.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 (sampled iterations) | 0.58 | 23.8% | 0.5 | 1.34 | `for (int i_1 = 0; i_1 < 32; ++i_1)` |
| load.global | 0.55 | 22.6% | 1.0 | 0.86 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 4096); \| i` |
| wait.sync#2 (sampled iterations) | 0.09 | 3.7% | 0.5 | 0.26 | `__syncthreads();` |
| wait.sync#4 (sampled iterations) | 0.09 | 3.6% | 0.5 | 0.26 | `__syncthreads();` |
| compute.mma (sampled iterations) | 0.07 | 2.9% | 0.5 | 0.16 | `for (int ki = 0; ki < 2; ++ki)` |
| load.global#3 (sampled iterations) | 0.06 | 2.5% | 0.5 | 0.29 | `if (((int)threadIdx.x) < 64)` |
| store.global (sampled iterations) | 0.04 | 1.8% | 0.5 | 0.16 | `if ((((int)threadIdx.x) >> 6) == 0)` |
| wait.sync (sampled iterations) | 0.03 | 1.4% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#3 (sampled iterations) | 0.03 | 1.1% | 0.5 | 0.06 | `__syncthreads();` |

### gemm_splitk_reduce_kernel grid (1, 6, 1) block (128, 1, 1)

4 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 0.77 us, warp 0.76 us, nsys graphs 0.94 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.40 | 52.3% | 0.40 | 52.1% |
| load | 0.31 | 41.0% | 0.31 | 40.6% |
| store | 0.05 | 6.7% | 0.06 | 7.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.31 | 41.0% | 1.0 | 0.32 | `acc[0] = P[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x))` |
| store.global | 0.05 | 6.7% | 1.0 | 0.06 | `C[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x)) * (int64` |

### metadata_pages_kernel grid (6, 1, 1) block (128, 1, 1)

2 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 1.31 us, warp 0.95 us, nsys graphs 1.06 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.74 | 78.0% | 0.80 | 61.0% |
| load | 0.13 | 13.3% | 0.19 | 14.6% |
| store | 0.08 | 8.7% | 0.32 | 24.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.11 | 11.1% | 1.0 | 0.13 | `if ((((offset + ((int)blockIdx.x)) < src_rows) && (0 <= (offset + ((int)blockIdx.x)))))` |
| store.global | 0.08 | 8.7% | 0.2 | 0.34 | `if (((int)threadIdx.x) < 32)` |
| load.global#2 | 0.02 | 2.2% | 0.2 | 0.10 | `if (((req < table_rows) && (((int)threadIdx.x) < table_cols)))` |

### norm_forward_and_per_token_cast_kernel_kernel grid (6, 1, 1) block (128, 1, 1)

1 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 2.56 us, warp 2.53 us, nsys graphs 1.89 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 2.29 | 90.7% | 2.14 | 83.8% |
| load | 0.44 | 17.4% | 0.48 | 18.8% |
| compute | 0.13 | 5.2% | 0.13 | 5.0% |
| wait | 0.06 | 2.3% | 0.03 | 1.2% |
| unattributed | 0.00 | 0.0% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*5 (est.: sampled iterations x5) | 2.07 | 81.8% | 1.0 | 2.08 | `for (int j_1 = 0; j_1 < 2; ++j_1)` |
| load.global#2*5 (est.: sampled iterations x5) | 0.24 | 9.5% | 1.0 | 0.32 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 1024) + (((int)threadIdx.x) * 8)` |
| load.global*5 (est.: sampled iterations x5) | 0.20 | 7.9% | 1.0 | 0.32 | `*(uint4*)(x_local + (vi * 8)) = *(uint4*)(x + ((((((int64_t)vi) * (int64_t)1024) + (((int6` |
| store.global*5 (est.: sampled iterations x5) | 0.17 | 6.9% | 1.0 | 0.32 | `for (int j = 0; j < 4; ++j)` |
| compute.reduce | 0.13 | 5.2% | 1.0 | 0.16 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| wait.sync | 0.06 | 2.3% | 1.0 | 0.10 | `__syncthreads();` |
| store.global#2 | 0.04 | 1.7% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| store.global#3 | 0.01 | 0.4% | 0.2 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |

### rope_kernel grid (6, 1, 1) block (256, 1, 1)

2 launches, 48 warps; ranges, warp and last warp from the 6 recording CTAs' 48 warps; span 1.02 us, warp 0.93 us, nsys graphs 1.14 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.64 | 68.8% | 0.72 | 70.3% |
| load | 0.26 | 27.4% | 0.26 | 25.0% |
| store | 0.04 | 3.8% | 0.05 | 4.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.26 | 27.4% | 1.0 | 0.26 | `if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < tokens)) \| int64_t position ` |
| store.global | 0.04 | 3.8% | 1.0 | 0.06 | `*(uint4*)(x + ((((((int64_t)((int)blockIdx.x)) * (int64_t)12288) + ((((int64_t)((int)threa` |

### mhc_collapse_column_kernel grid (30, 1, 1) block (128, 1, 1)

1 launches, 120 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 0.80 us, warp 0.76 us, nsys graphs 0.90 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.62 | 81.8% | 0.58 | 75.0% |
| load | 0.09 | 11.9% | 0.13 | 16.7% |
| store | 0.05 | 6.3% | 0.06 | 8.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.09 | 11.9% | 1.0 | 0.13 | `*(float4*)(mv + 0) = *(float4*)(mix + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int6` |
| store.global | 0.05 | 6.3% | 1.0 | 0.06 | `*(uint4*)(out + ((((int64_t)((int)blockIdx.x)) * (int64_t)1024) + (((int64_t)((int)threadI` |

### ced_copy_rows_kernel grid (30, 1, 1) block (128, 1, 1)

1 launches, 120 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 0.64 us, warp 0.58 us, nsys graphs 1.44 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.35 | 59.9% | 0.38 | 60.0% |
| unattributed | 0.23 | 40.1% | 0.26 | 40.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.35 | 59.9% | 1.0 | 0.38 | `for (int k = 0; k < 4; ++k) \| int64_t index = ced_ldg64((&(row_indices[((((int64_t)blockId` |
=== timeline and tables: exit 0 in 22 s
