trace /home/swank/.cache/spark-lab/compile/iket/full-43l-exp/run2_base/trace/iket/pid_0x36/iket.decoded_results.json: 22429 launches, graphLaunches dict of 14

## Timeline: 8355 launches in 478689.4 us; the GPU is busy 190194.8 us (40%) and idle 288494.6 us in 3169 gaps; gaps over 200 us split it into 257 segments

The 15 of 257 segments with the most busy time:

| segment | launches | wall us | busy us | busy | first kernel |
|---:|---:|---:|---:|---:|---|
| 18 | 198 | 13843.8 | 4933.1 | 36% | gemm_bf16_partials_kernel |
| 82 | 164 | 11576.7 | 4268.5 | 37% | rope_kernel |
| 83 | 219 | 16672.7 | 5915.7 | 35% | rope_kernel |
| 90 | 21 | 5758.0 | 4496.1 | 78% | gemm_splitk_reduce_cast_kernel |
| 108 | 22 | 4885.4 | 4332.0 | 89% | sparse_mla_kernel_1 |
| 117 | 1222 | 17941.3 | 17803.3 | 99% | mhc_project_streams_kernel |
| 130 | 39 | 6428.8 | 4679.8 | 73% | activation_swiglu_clamp_kernel |
| 134 | 1273 | 21742.8 | 20039.5 | 92% | partial_argmax_kernel |
| 136 | 2 | 4111.9 | 4106.6 | 100% | ced_copy_rows_kernel |
| 149 | 7 | 4558.2 | 4088.1 | 90% | moe_grouped_fp8_fp4_gemm_kernel |
| 168 | 1229 | 20005.6 | 19451.5 | 97% | metadata_builder_kernel |
| 192 | 1261 | 19936.6 | 19091.9 | 96% | partial_argmax_kernel |
| 193 | 3 | 4540.2 | 4388.5 | 97% | sink_flags_kernel |
| 213 | 27 | 5441.6 | 4338.0 | 80% | attention_front_kernel |
| 222 | 1257 | 20052.8 | 19139.8 | 95% | scatter_rows_gather_kernel |

| idle us | after | before |
|---:|---|---|
| 73359.8 | scatter_rows_kernel | padding_rows_kernel |
| 8781.5 | norm_forward_and_per_token_cast_kernel_kernel | sink_flags_kernel |
| 8013.5 | norm_forward_and_per_token_cast_kernel_kernel | sink_flags_kernel |
| 7970.0 | norm_forward_and_per_token_cast_kernel_kernel | sink_flags_kernel |
| 7932.6 | norm_forward_and_per_token_cast_kernel_kernel | sink_flags_kernel |
| 7665.5 | norm_forward_and_per_token_cast_kernel_kernel | sink_flags_kernel |
| 1233.8 | scatter_rows_kernel | padding_rows_kernel |
| 803.4 | vocab_packed_projection_kernel | logits_rows_kernel |
| 654.7 | local_stats_kernel | verify_kernel |
| 517.2 | sparse_mla_kernel_1 | gemm_fp8w_bf16_partials_kernel |

## Engine steps

| step | kind | launches | wall us | busy us | busy | most span |
|---:|---|---:|---:|---:|---:|---|
| 0 | prefill | 1283 | 133719.6 | 39085.6 | 29% | moe_grouped_fp8_fp4_gemm_kernel 10840, gemm_mxfp8_kernel 10236, vocab_packed_projection_kernel 8133 |
| 1 | decode | 1415 | 52372.0 | 29159.9 | 56% | l2_prefetch_kernel 8688, vocab_packed_projection_kernel 8199, gemm_mxfp8_kernel 7181 |
| 2 | decode | 1415 | 127225.6 | 30419.6 | 24% | vocab_packed_projection_kernel 8162, moe_grouped_fp8_fp4_gemm_kernel 7773, l2_prefetch_kernel 7554 |
| 3 | decode | 1415 | 54522.5 | 30704.9 | 56% | vocab_packed_projection_kernel 8150, l2_prefetch_kernel 8012, moe_grouped_fp8_fp4_gemm_kernel 7853 |
| 4 | decode | 1415 | 54169.0 | 30542.2 | 56% | vocab_packed_projection_kernel 8407, l2_prefetch_kernel 7875, moe_grouped_fp8_fp4_gemm_kernel 7745 |
| 5 | decode | 1412 | 56303.8 | 30282.6 | 54% | vocab_packed_projection_kernel 8124, l2_prefetch_kernel 7786, moe_grouped_fp8_fp4_gemm_kernel 7728 |

## Overview over 5 decode steps: 7072 launches of 121 kernel shapes, spans summing to 238820.0 us (47764.0 a step)

| kernel | grid | launches a step | span us a step | share | span us a launch | last warp: wait | load | compute | store | unattributed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vocab_packed_projection_kernel | (1, 4040, 1) | 2 | 8208.6 | 17.2% | 4077.82 | 0% | 0% | 0% | 0% | 100% |
| l2_prefetch_kernel | (2, 1, 1) | 119 | 7983.1 | 16.7% | 69.18 | 0% | 0% | 0% | 0% | 100% |
| moe_grouped_fp8_fp4_gemm_kernel | (9, 36, 1) | 40 | 4564.9 | 9.6% | 117.58 | 0% | 0% | 0% | 0% | 100% |
| moe_grouped_fp8_fp4_gemm_kernel | (40, 36, 1) | 40 | 2673.3 | 5.6% | 69.23 | 0% | 0% | 0% | 0% | 100% |
| gemm_mxfp8_kernel | (28, 1, 1) | 43 | 2250.4 | 4.7% | 53.12 | 0% | 0% | 0% | 0% | 100% |
| mhc_project_streams_kernel | (40, 1, 1) | 86 | 2104.2 | 4.4% | 20.18 | 0% | 0% | 0% | 0% | 100% |
| attention_front_kernel | (4, 1, 1) | 43 | 2036.4 | 4.3% | 49.73 | 0% | 0% | 0% | 0% | 100% |
| mhc_finalize_kernel | (6, 1, 1) | 80 | 1868.5 | 3.9% | 22.06 | 0% | 0% | 0% | 0% | 100% |
| gemm_fp8w_bf16_partials_kernel | (32, 4, 1) | 43 | 1629.8 | 3.4% | 38.66 | 0% | 0% | 0% | 0% | 100% |
| gemm_mxfp8_kernel | (64, 1, 1) | 51 | 1429.4 | 3.0% | 24.10 | 0% | 0% | 0% | 0% | 100% |
| gemm_bf16_partials_kernel | (6, 8, 1) | 40 | 1381.9 | 2.9% | 33.78 | 0% | 0% | 0% | 0% | 100% |
| moe_swiglu_cast_kernel_kernel | (8, 36, 1) | 40 | 1338.5 | 2.8% | 35.82 | 0% | 0% | 0% | 0% | 100% |
| gemm_bf16_kernel | (2020, 1, 1) | 5 | 1336.7 | 2.8% | 266.27 | 0% | 0% | 0% | 0% | 100% |
| gemm_mxfp8_kernel | (36, 1, 1) | 43 | 1327.3 | 2.8% | 30.40 | 0% | 0% | 0% | 0% | 100% |
| gemm_splitk_reduce_cast_kernel | (16, 2, 1) | 43 | 1215.6 | 2.5% | 30.43 | 0% | 0% | 0% | 0% | 100% |
| sparse_mla_kernel_1 | (16, 6, 1) | 40 | 1055.2 | 2.2% | 28.00 | 0% | 0% | 0% | 0% | 100% |
| sparse_mla_kernel | (5, 1, 6) | 38 | 927.7 | 1.9% | 25.09 | 0% | 0% | 0% | 0% | 100% |
| gemm_mxfp8_kernel | (40, 1, 1) | 43 | 850.5 | 1.8% | 14.72 | 0% | 0% | 0% | 0% | 100% |
| gemm_mxfp8_kernel | (40, 1, 1) | 43 | 447.2 | 0.9% | 8.67 | 0% | 0% | 0% | 0% | 100% |
| gemm_mxfp8_kernel | (80, 1, 1) | 1 | 360.0 | 0.8% | 359.01 | 0% | 0% | 0% | 0% | 100% |

| wait | kernel | grid | last-warp us a step | a launch | first statement |
|---|---|---|---:|---:|---|

### vocab_packed_projection_kernel grid (1, 4040, 1) block (128, 1, 1)

10 launches, 16160 warps; ranges, warp and last warp from the 4040 recording CTAs' 16160 warps; span 4077.82 us, warp 46.55 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 46.55 | 100.0% | 25.46 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### l2_prefetch_kernel grid (2, 1, 1) block (128, 1, 1)

595 launches, 8 warps; ranges, warp and last warp from the 2 recording CTAs' 8 warps; span 69.18 us, warp 68.32 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 68.32 | 100.0% | 67.08 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### moe_grouped_fp8_fp4_gemm_kernel grid (9, 36, 1) block (128, 1, 1)

200 launches, 1296 warps; ranges, warp and last warp from the 324 recording CTAs' 1296 warps; span 117.58 us, warp 14.90 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 14.90 | 100.0% | 44.71 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### moe_grouped_fp8_fp4_gemm_kernel grid (40, 36, 1) block (128, 1, 1)

200 launches, 5760 warps; ranges, warp and last warp from the 1440 recording CTAs' 5760 warps; span 69.23 us, warp 2.01 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 2.01 | 100.0% | 0.31 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### gemm_mxfp8_kernel grid (28, 1, 1) block (128, 1, 1)

215 launches, 112 warps; ranges, warp and last warp from the 28 recording CTAs' 112 warps; span 53.12 us, warp 48.08 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 48.08 | 100.0% | 52.33 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### mhc_project_streams_kernel grid (40, 1, 1) block (128, 1, 1)

430 launches, 160 warps; ranges, warp and last warp from the 40 recording CTAs' 160 warps; span 20.18 us, warp 15.06 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 15.06 | 100.0% | 19.74 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### attention_front_kernel grid (4, 1, 1) block (128, 1, 1)

215 launches, 16 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 49.73 us, warp 48.34 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 48.34 | 100.0% | 47.36 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### mhc_finalize_kernel grid (6, 1, 1) block (128, 1, 1)

400 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 22.06 us, warp 19.88 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 19.88 | 100.0% | 23.35 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### gemm_fp8w_bf16_partials_kernel grid (32, 4, 1) block (128, 1, 1)

215 launches, 512 warps; ranges, warp and last warp from the 128 recording CTAs' 512 warps; span 38.66 us, warp 29.33 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 29.33 | 100.0% | 37.74 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### gemm_mxfp8_kernel grid (64, 1, 1) block (128, 1, 1)

255 launches, 256 warps; ranges, warp and last warp from the 64 recording CTAs' 256 warps; span 24.10 us, warp 13.28 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 13.28 | 100.0% | 12.71 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### gemm_bf16_partials_kernel grid (6, 8, 1) block (256, 1, 1)

200 launches, 384 warps; ranges, warp and last warp from the 48 recording CTAs' 384 warps; span 33.78 us, warp 8.07 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 8.07 | 100.0% | 6.64 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### moe_swiglu_cast_kernel_kernel grid (8, 36, 1) block (288, 1, 1)

200 launches, 2592 warps; ranges, warp and last warp from the 288 recording CTAs' 2592 warps; span 35.82 us, warp 19.41 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 19.41 | 100.0% | 33.45 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### gemm_bf16_kernel grid (2020, 1, 1) block (256, 1, 1)

25 launches, 16160 warps; ranges, warp and last warp from the 2020 recording CTAs' 16160 warps; span 266.27 us, warp 3.14 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 3.14 | 100.0% | 5.18 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### gemm_mxfp8_kernel grid (36, 1, 1) block (128, 1, 1)

215 launches, 144 warps; ranges, warp and last warp from the 36 recording CTAs' 144 warps; span 30.40 us, warp 28.18 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 28.18 | 100.0% | 30.57 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### gemm_splitk_reduce_cast_kernel grid (16, 2, 1) block (128, 1, 1)

215 launches, 128 warps; ranges, warp and last warp from the 32 recording CTAs' 128 warps; span 30.43 us, warp 30.33 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 30.33 | 100.0% | 28.26 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### sparse_mla_kernel_1 grid (16, 6, 1) block (128, 1, 1)

200 launches, 96 CTAs, 384 warps; span 28.00 us, warp 27.22 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 27.22 | 100.0% | 26.36 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### sparse_mla_kernel grid (5, 1, 6) block (128, 1, 1)

190 launches, 120 warps; ranges, warp and last warp from the 30 recording CTAs' 120 warps; span 25.09 us, warp 18.25 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 18.25 | 100.0% | 24.41 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### gemm_mxfp8_kernel grid (40, 1, 1) block (128, 1, 1)

215 launches, 160 warps; ranges, warp and last warp from the 40 recording CTAs' 160 warps; span 14.72 us, warp 12.37 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 12.37 | 100.0% | 19.77 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### gemm_mxfp8_kernel grid (40, 1, 1) block (64, 1, 1)

215 launches, 80 warps; ranges, warp and last warp from the 40 recording CTAs' 80 warps; span 8.67 us, warp 7.07 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 7.07 | 100.0% | 10.39 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### gemm_mxfp8_kernel grid (80, 1, 1) block (128, 1, 1)

5 launches, 320 warps; ranges, warp and last warp from the 80 recording CTAs' 320 warps; span 359.01 us, warp 350.43 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 350.43 | 100.0% | 360.00 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### rope_kernel grid (6, 1, 1) block (128, 1, 1)

200 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 7.79 us, warp 7.76 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 7.76 | 100.0% | 8.44 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### gemm_bf16_splitk_kernel grid (16, 5, 1) block (128, 1, 1)

20 launches, 320 warps; ranges, warp and last warp from the 80 recording CTAs' 320 warps; span 65.22 us, warp 56.48 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 56.48 | 100.0% | 64.62 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### moe_dispatch_kernel grid (1, 1, 1) block (512, 1, 1)

215 launches, 16 warps; ranges, warp and last warp from the 1 recording CTAs' 16 warps; span 5.47 us, warp 5.26 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 5.26 | 100.0% | 5.61 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### compress_gather_partial_state_kernel grid (16, 1, 1) block (128, 1, 1)

15 launches, 64 warps; ranges, warp and last warp from the 16 recording CTAs' 64 warps; span 72.99 us, warp 65.69 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 65.69 | 100.0% | 71.45 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### router_partials_kernel grid (3, 1, 1) block (64, 1, 1)

215 launches, 6 warps; ranges, warp and last warp from the 3 recording CTAs' 6 warps; span 4.03 us, warp 2.41 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 2.41 | 100.0% | 4.36 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### moe_grouped_fp8_fp4_gemm_kernel grid (9, 15, 1) block (128, 1, 1)

15 launches, 540 warps; ranges, warp and last warp from the 135 recording CTAs' 540 warps; span 43.97 us, warp 8.23 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 8.23 | 100.0% | 47.44 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### compress_kernel grid (2, 1, 1) block (128, 1, 1)

20 launches, 8 warps; ranges, warp and last warp from the 2 recording CTAs' 8 warps; span 22.66 us, warp 16.58 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 16.58 | 100.0% | 32.13 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### activation_swiglu_clamp_kernel grid (1, 6, 1) block (128, 1, 1)

200 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 1.52 us, warp 1.32 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.32 | 100.0% | 3.12 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### gemm_bf16_partials_kernel grid (2, 20, 1) block (256, 1, 1)

15 launches, 320 warps; ranges, warp and last warp from the 40 recording CTAs' 320 warps; span 33.28 us, warp 7.10 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 7.10 | 100.0% | 5.71 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### moe_grouped_fp8_fp4_gemm_kernel grid (40, 15, 1) block (128, 1, 1)

15 launches, 2400 warps; ranges, warp and last warp from the 600 recording CTAs' 2400 warps; span 25.86 us, warp 1.78 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.78 | 100.0% | 0.16 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### moe_combine_kernel grid (5, 6, 1) block (128, 1, 1)

200 launches, 120 warps; ranges, warp and last warp from the 30 recording CTAs' 120 warps; span 2.02 us, warp 1.42 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.42 | 100.0% | 1.99 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### rope_kernel grid (6, 1, 1) block (256, 1, 1)

40 launches, 48 warps; ranges, warp and last warp from the 6 recording CTAs' 48 warps; span 4.98 us, warp 4.91 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 4.91 | 100.0% | 7.60 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### metadata_decode_kernel grid (6, 2, 1) block (128, 1, 1)

75 launches, 48 warps; ranges, warp and last warp from the 12 recording CTAs' 48 warps; span 2.62 us, warp 1.27 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.27 | 100.0% | 3.71 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### indexer_score_kernel grid (128, 6, 1) block (128, 1, 1)

25 launches, 3072 warps; ranges, warp and last warp from the 768 recording CTAs' 3072 warps; span 5.41 us, warp 0.54 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.54 | 100.0% | 6.50 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### gemm_bf16_partials_kernel grid (1, 40, 1) block (256, 1, 1)

40 launches, 320 warps; ranges, warp and last warp from the 40 recording CTAs' 320 warps; span 2.64 us, warp 1.54 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.54 | 100.0% | 4.98 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### indexer_topk_merge_kernel grid (1, 6, 1) block (512, 1, 1)

25 launches, 96 warps; ranges, warp and last warp from the 6 recording CTAs' 96 warps; span 7.39 us, warp 7.05 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 7.05 | 100.0% | 7.30 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### sparse_mla_kernel grid (1, 1, 6) block (128, 1, 1)

10 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 16.14 us, warp 15.68 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 15.68 | 100.0% | 17.86 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### mhc_finalize_kernel grid (5, 1, 1) block (128, 1, 1)

30 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 5.09 us, warp 4.73 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 4.73 | 100.0% | 5.68 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### gemm_mxfp8_partials_kernel grid (16, 5, 1) block (128, 1, 1)

15 launches, 320 warps; ranges, warp and last warp from the 80 recording CTAs' 320 warps; span 11.30 us, warp 10.55 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 10.55 | 100.0% | 11.36 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### per_token_cast_kernel_kernel grid (9, 1, 1) block (64, 1, 1)

215 launches, 18 warps; ranges, warp and last warp from the 9 recording CTAs' 18 warps; span 0.51 us, warp 0.43 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.43 | 100.0% | 0.77 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### indexer_topk_segment_kernel grid (4, 6, 1) block (512, 1, 1)

20 launches, 384 warps; ranges, warp and last warp from the 24 recording CTAs' 384 warps; span 6.05 us, warp 3.02 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 3.02 | 100.0% | 7.54 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### sparse_mla_kernel grid (2, 1, 5) block (128, 1, 1)

15 launches, 40 warps; ranges, warp and last warp from the 10 recording CTAs' 40 warps; span 10.02 us, warp 7.97 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 7.97 | 100.0% | 10.02 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### metadata_builder_kernel grid (2, 1, 1) block (128, 1, 1)

130 launches, 8 warps; ranges, warp and last warp from the 2 recording CTAs' 8 warps; span 1.02 us, warp 0.44 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.44 | 100.0% | 1.15 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### indexer_mxfp4_quantize_kernel grid (6, 1, 1) block (128, 1, 1)

40 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 2.62 us, warp 2.55 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 2.55 | 100.0% | 3.32 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### indexer_topk_segment_kernel grid (1, 6, 1) block (512, 1, 1)

20 launches, 96 warps; ranges, warp and last warp from the 6 recording CTAs' 96 warps; span 4.94 us, warp 4.89 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 4.89 | 100.0% | 6.00 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### metadata_pages_kernel grid (6, 1, 1) block (128, 1, 1)

40 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 1.14 us, warp 0.70 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.70 | 100.0% | 2.87 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### token_ranks_kernel grid (1, 1, 1) block (256, 1, 1)

15 launches, 8 warps; ranges, warp and last warp from the 1 recording CTAs' 8 warps; span 7.23 us, warp 7.23 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 7.23 | 100.0% | 7.27 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### partial_argmax_kernel grid (1, 64, 1) block (128, 1, 1)

25 launches, 256 warps; ranges, warp and last warp from the 64 recording CTAs' 256 warps; span 3.94 us, warp 3.40 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 3.40 | 100.0% | 3.92 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### gemm_bf16_partials_kernel grid (2, 4, 1) block (256, 1, 1)

20 launches, 64 warps; ranges, warp and last warp from the 8 recording CTAs' 64 warps; span 4.13 us, warp 2.05 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 2.05 | 100.0% | 4.21 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### topk_log_softmax_kernel grid (1, 1, 1) block (1024, 1, 1)

5 launches, 32 warps; ranges, warp and last warp from the 1 recording CTAs' 32 warps; span 15.55 us, warp 15.54 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 15.54 | 100.0% | 16.18 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### topk_ids_kernel grid (5, 1, 1) block (512, 1, 1)

4 launches, 80 warps; ranges, warp and last warp from the 5 recording CTAs' 80 warps; span 20.00 us, warp 18.06 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 18.06 | 100.0% | 20.00 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### norm_forward_and_per_token_cast_kernel_kernel grid (1, 1, 1) block (128, 1, 1)

35 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 2.18 us, warp 2.18 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 2.18 | 100.0% | 2.14 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### mhc_collapse_kernel grid (30, 1, 1) block (128, 1, 1)

15 launches, 120 warps; ranges, warp and last warp from the 30 recording CTAs' 120 warps; span 3.58 us, warp 3.57 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 3.57 | 100.0% | 4.42 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### rope_kernel grid (1, 1, 1) block (128, 1, 1)

55 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 0.80 us, warp 0.48 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.48 | 100.0% | 1.19 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### topk_log_softmax_kernel grid (5, 1, 1) block (1024, 1, 1)

4 launches, 160 warps; ranges, warp and last warp from the 5 recording CTAs' 160 warps; span 15.76 us, warp 15.12 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 15.12 | 100.0% | 15.70 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### _ZN2at6native44_GLOBAL__N__b23ef953_11_Indexing_cu_89862edb21indexSelectSmallIndexIN3c108BFloat16EljLi2ELi2ELin2EEEvNS_4cuda6detail10TensorInfoIT_T1_EENS7_IKS8_S9_EENS7_IKT0_S9_EEiiS9_l grid (40, 1, 1) block (128, 1, 1)

10 launches, 40 CTAs, 160 warps; span 6.43 us, warp 6.42 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 6.42 | 100.0% | 6.10 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### mhc_post_kernel grid (30, 1, 1) block (128, 1, 1)

20 launches, 120 warps; ranges, warp and last warp from the 30 recording CTAs' 120 warps; span 3.14 us, warp 1.84 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.84 | 100.0% | 2.81 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### local_stats_kernel grid (5, 16, 1) block (128, 1, 1)

4 launches, 320 warps; ranges, warp and last warp from the 80 recording CTAs' 320 warps; span 14.03 us, warp 13.33 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 13.33 | 100.0% | 13.97 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### indexer_score_kernel grid (64, 6, 1) block (128, 1, 1)

15 launches, 1536 warps; ranges, warp and last warp from the 384 recording CTAs' 1536 warps; span 3.01 us, warp 0.78 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.78 | 100.0% | 3.26 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### gemm_splitk_reduce_kernel grid (1, 6, 1) block (128, 1, 1)

60 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 0.51 us, warp 0.50 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.50 | 100.0% | 0.81 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### indexer_mxfp4_quantize_kernel grid (1, 1, 1) block (128, 1, 1)

20 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 2.16 us, warp 0.72 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.72 | 100.0% | 2.42 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### moe_swiglu_cast_kernel_kernel grid (8, 15, 1) block (288, 1, 1)

15 launches, 1080 warps; ranges, warp and last warp from the 120 recording CTAs' 1080 warps; span 2.53 us, warp 0.47 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.47 | 100.0% | 2.53 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### metadata_chunk_kernel grid (5, 1, 1) block (192, 1, 1)

15 launches, 30 warps; ranges, warp and last warp from the 5 recording CTAs' 30 warps; span 1.95 us, warp 1.31 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.31 | 100.0% | 2.04 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### token_ranks_kernel grid (5, 1, 1) block (256, 1, 1)

4 launches, 40 warps; ranges, warp and last warp from the 5 recording CTAs' 40 warps; span 7.58 us, warp 7.47 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 7.47 | 100.0% | 7.58 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### prepare_dflash_inputs_kernel grid (9, 1, 1) block (256, 1, 1)

15 launches, 72 warps; ranges, warp and last warp from the 9 recording CTAs' 72 warps; span 1.66 us, warp 0.45 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.45 | 100.0% | 1.79 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### _ZN2at6native44_GLOBAL__N__b23ef953_11_Indexing_cu_89862edb21indexSelectSmallIndexIN3c108BFloat16EljLi2ELi2ELin2EEEvNS_4cuda6detail10TensorInfoIT_T1_EENS7_IKS8_S9_EENS7_IKT0_S9_EEiiS9_l grid (2, 1, 1) block (128, 1, 1)

20 launches, 2 CTAs, 8 warps; span 1.25 us, warp 1.25 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.25 | 100.0% | 1.35 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### norm_forward_and_per_token_cast_kernel_kernel grid (5, 1, 1) block (128, 1, 1)

9 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 2.94 us, warp 2.88 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 2.88 | 100.0% | 2.97 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### indexer_topk_segment_kernel grid (2, 6, 1) block (512, 1, 1)

5 launches, 192 warps; ranges, warp and last warp from the 12 recording CTAs' 192 warps; span 4.83 us, warp 2.56 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 2.56 | 100.0% | 4.90 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### topk_ids_kernel grid (6, 1, 1) block (512, 1, 1)

1 launches, 96 warps; ranges, warp and last warp from the 6 recording CTAs' 96 warps; span 24.45 us, warp 21.94 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 21.94 | 100.0% | 24.45 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### moe_combine_kernel grid (5, 5, 1) block (128, 1, 1)

15 launches, 100 warps; ranges, warp and last warp from the 25 recording CTAs' 100 warps; span 1.06 us, warp 0.94 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.94 | 100.0% | 1.54 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### per_token_cast_kernel_kernel grid (5, 1, 1) block (64, 1, 1)

40 launches, 10 warps; ranges, warp and last warp from the 5 recording CTAs' 10 warps; span 0.54 us, warp 0.53 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.53 | 100.0% | 0.57 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### cache_writer_kernel grid (6, 1, 1) block (128, 1, 1)

23 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 0.80 us, warp 0.54 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.54 | 100.0% | 0.90 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### resample_kernel grid (1, 127, 1) block (128, 1, 1)

5 launches, 508 warps; ranges, warp and last warp from the 127 recording CTAs' 508 warps; span 1.92 us, warp 1.50 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.50 | 100.0% | 3.88 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### sink_merge_kernel grid (1, 1, 1) block (128, 1, 1)

5 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 3.81 us, warp 3.81 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 3.81 | 100.0% | 3.79 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### sink_candidates_kernel grid (1, 127, 1) block (128, 1, 1)

5 launches, 508 warps; ranges, warp and last warp from the 127 recording CTAs' 508 warps; span 3.90 us, warp 2.94 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 2.94 | 100.0% | 3.48 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### norm_forward_and_per_token_cast_kernel_kernel grid (6, 1, 1) block (128, 1, 1)

6 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 2.88 us, warp 2.84 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 2.84 | 100.0% | 2.89 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### topk_log_softmax_kernel grid (6, 1, 1) block (1024, 1, 1)

1 launches, 192 warps; ranges, warp and last warp from the 6 recording CTAs' 192 warps; span 15.97 us, warp 15.27 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 15.27 | 100.0% | 15.87 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### sparse_mla_kernel_1 grid (16, 5, 1) block (128, 1, 1)

15 launches, 80 CTAs, 320 warps; span 1.09 us, warp 0.98 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.98 | 100.0% | 1.04 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### metadata_chunk_kernel grid (6, 1, 1) block (128, 1, 1)

10 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 1.15 us, warp 0.73 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.73 | 100.0% | 1.56 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### _ZN2at6native27unrolled_elementwise_kernelIZZZNS0_23direct_copy_kernel_cudaERNS_18TensorIteratorBaseEENKUlvE1_clEvENKUlvE2_clEvEUllE_St5arrayIPcLm2EELi4E23TrivialOffsetCalculatorILi1EjESB_NS0_6memory12LoadWithCastILi1EEENSC_13StoreWithCastILi1EEEEEviT_T0_T2_T3_T4_T5_ grid (1, 1, 1) block (128, 1, 1)

10 launches, 1 CTAs, 4 warps; span 1.02 us, warp 0.78 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.78 | 100.0% | 1.52 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### local_stats_kernel grid (6, 16, 1) block (128, 1, 1)

1 launches, 384 warps; ranges, warp and last warp from the 96 recording CTAs' 384 warps; span 13.22 us, warp 10.77 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 10.77 | 100.0% | 13.15 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### mhc_post_kernel grid (25, 1, 1) block (128, 1, 1)

5 launches, 100 warps; ranges, warp and last warp from the 25 recording CTAs' 100 warps; span 0.83 us, warp 0.66 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.66 | 100.0% | 2.30 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### compute_slot_mappings_kernel grid (1, 17, 1) block (256, 1, 1)

5 launches, 136 warps; ranges, warp and last warp from the 17 recording CTAs' 136 warps; span 2.24 us, warp 1.31 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.31 | 100.0% | 2.25 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### _ZN2at6native24index_elementwise_kernelILi128ELi4EZNS0_16gpu_index_kernelIZNS0_17index_kernel_implINS0_10OpaqueTypeILi4EEEEEvRNS_18TensorIteratorBaseEN3c108ArrayRefIlEESA_EUlPcPKclE_EEvS7_SA_SA_RKT_bEUliE_EEvlT1_ grid (1, 1, 1) block (128, 1, 1)

5 launches, 1 CTAs, 4 warps; span 2.21 us, warp 1.97 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.97 | 100.0% | 2.20 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### finish_argmax_kernel grid (1, 1, 1) block (32, 1, 1)

25 launches, 1 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 0.42 us, warp 0.42 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.42 | 100.0% | 0.44 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### activation_swiglu_clamp_kernel grid (1, 5, 1) block (128, 1, 1)

15 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 0.74 us, warp 0.56 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.56 | 100.0% | 0.72 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### sink_cut_kernel grid (1, 1, 1) block (128, 1, 1)

10 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 1.02 us, warp 0.36 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.36 | 100.0% | 1.03 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### sink_policy_kernel grid (1, 1, 1) block (32, 1, 1)

5 launches, 1 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 1.95 us, warp 1.95 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.95 | 100.0% | 2.00 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### verify_kernel grid (1, 1, 1) block (96, 1, 1)

5 launches, 3 warps; ranges, warp and last warp from the 1 recording CTAs' 3 warps; span 1.92 us, warp 1.75 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.75 | 100.0% | 1.95 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### rope_kernel grid (5, 1, 1) block (128, 1, 1)

15 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 0.58 us, warp 0.55 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.55 | 100.0% | 0.62 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### sink_flags_kernel grid (1, 1, 1) block (128, 1, 1)

5 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 1.76 us, warp 0.54 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.54 | 100.0% | 1.82 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### _ZN2at6native44_GLOBAL__N__b23ef953_11_Indexing_cu_89862edb21indexSelectSmallIndexIN3c108BFloat16EijLi2ELi2ELin2EEEvNS_4cuda6detail10TensorInfoIT_T1_EENS7_IKS8_S9_EENS7_IKT0_S9_EEiiS9_l grid (2, 1, 1) block (128, 1, 1)

5 launches, 2 CTAs, 8 warps; span 1.34 us, warp 1.32 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.32 | 100.0% | 1.68 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### per_token_cast_kernel_kernel grid (15, 1, 1) block (64, 1, 1)

15 launches, 30 warps; ranges, warp and last warp from the 15 recording CTAs' 30 warps; span 0.54 us, warp 0.49 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.49 | 100.0% | 0.54 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### post_update_kernel grid (1, 1, 1) block (32, 1, 1)

5 launches, 1 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 1.57 us, warp 1.57 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.57 | 100.0% | 1.57 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### gather_block_tables_kernel grid (17, 1, 1) block (256, 1, 1)

5 launches, 136 warps; ranges, warp and last warp from the 17 recording CTAs' 136 warps; span 1.54 us, warp 1.10 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.10 | 100.0% | 1.50 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### token_ranks_kernel grid (6, 1, 1) block (256, 1, 1)

1 launches, 48 warps; ranges, warp and last warp from the 6 recording CTAs' 48 warps; span 7.49 us, warp 7.39 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 7.39 | 100.0% | 7.49 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### cache_writer_kernel grid (5, 1, 1) block (128, 1, 1)

12 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 0.61 us, warp 0.43 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.43 | 100.0% | 0.62 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### dspark_confidence_kernel grid (5, 1, 1) block (128, 1, 1)

5 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 1.34 us, warp 1.25 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.25 | 100.0% | 1.36 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### sink_rows_kernel grid (1, 64, 1) block (256, 1, 1)

5 launches, 512 warps; ranges, warp and last warp from the 64 recording CTAs' 512 warps; span 1.22 us, warp 0.77 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.77 | 100.0% | 1.32 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### ced_copy_rows_kernel grid (25, 1, 1) block (128, 1, 1)

9 launches, 100 warps; ranges, warp and last warp from the 25 recording CTAs' 100 warps; span 0.74 us, warp 0.72 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.72 | 100.0% | 0.65 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### combine_sampled_and_draft_tokens_kernel grid (1, 1, 1) block (32, 1, 1)

5 launches, 1 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 1.12 us, warp 1.12 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.12 | 100.0% | 1.15 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### adaptive_offsets_kernel grid (1, 1, 1) block (256, 1, 1)

4 launches, 8 warps; ranges, warp and last warp from the 1 recording CTAs' 8 warps; span 1.41 us, warp 1.38 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.38 | 100.0% | 1.41 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### assign_draft_budget_kernel grid (1, 1, 1) block (256, 1, 1)

4 launches, 8 warps; ranges, warp and last warp from the 1 recording CTAs' 8 warps; span 1.36 us, warp 1.36 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.36 | 100.0% | 1.37 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### split_columns_kernel grid (5, 15, 1) block (128, 1, 1)

4 launches, 300 warps; ranges, warp and last warp from the 75 recording CTAs' 300 warps; span 1.30 us, warp 0.98 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.98 | 100.0% | 1.28 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### logits_rows_kernel grid (1, 1, 1) block (128, 1, 1)

5 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 0.96 us, warp 0.65 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.65 | 100.0% | 0.97 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### _ZN2at6native40_GLOBAL__N__f1f992fe_8_Shape_cu_49f7391c30CatArrayBatchedCopy_vectorizedINS1_10OpaqueTypeILj2EEEjLi2ELi128ELi1ELi16ELi8EEEvPcNS1_25CatArrInputTensorMetadataIT_T0_XT2_EXT3_EEENS1_16TensorSizeStrideIS8_Lj4EEEiS8_ grid (1, 5, 1) block (128, 1, 1)

5 launches, 5 CTAs, 20 warps; span 0.90 us, warp 0.37 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.37 | 100.0% | 0.89 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### flatten_sampled_kernel grid (1, 1, 1) block (128, 1, 1)

5 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 0.86 us, warp 0.60 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.60 | 100.0% | 0.86 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### prepare_pos_seq_lens_kernel grid (2, 1, 1) block (256, 1, 1)

5 launches, 16 warps; ranges, warp and last warp from the 2 recording CTAs' 16 warps; span 0.83 us, warp 0.54 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.54 | 100.0% | 0.83 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### mhc_collapse_column_kernel grid (30, 1, 1) block (128, 1, 1)

5 launches, 120 warps; ranges, warp and last warp from the 30 recording CTAs' 120 warps; span 0.80 us, warp 0.74 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.74 | 100.0% | 0.82 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### gemm_splitk_reduce_kernel grid (4, 5, 1) block (128, 1, 1)

12 launches, 80 warps; ranges, warp and last warp from the 20 recording CTAs' 80 warps; span 0.32 us, warp 0.28 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.28 | 100.0% | 0.30 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### insert_kernel grid (1, 1, 1) block (32, 1, 1)

5 launches, 1 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 0.74 us, warp 0.74 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.74 | 100.0% | 0.74 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### token_counts_kernel grid (1, 1, 1) block (128, 1, 1)

5 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 0.74 us, warp 0.30 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.30 | 100.0% | 0.74 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### per_token_cast_kernel_kernel grid (45, 1, 1) block (64, 1, 1)

5 launches, 90 warps; ranges, warp and last warp from the 45 recording CTAs' 90 warps; span 0.70 us, warp 0.59 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.59 | 100.0% | 0.68 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### scatter_rows_gather_kernel grid (1, 1, 1) block (128, 1, 1)

5 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 0.67 us, warp 0.36 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.36 | 100.0% | 0.67 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### ced_requests_kernel grid (1, 1, 1) block (128, 1, 1)

5 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 0.67 us, warp 0.31 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.31 | 100.0% | 0.67 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### scatter_rows_kernel grid (1, 1, 1) block (128, 1, 1)

5 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 0.64 us, warp 0.35 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.35 | 100.0% | 0.65 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### mhc_collapse_kernel grid (25, 1, 1) block (128, 1, 1)

5 launches, 100 warps; ranges, warp and last warp from the 25 recording CTAs' 100 warps; span 0.51 us, warp 0.45 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.45 | 100.0% | 0.60 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### padding_rows_kernel grid (1, 1, 1) block (256, 1, 1)

9 launches, 8 warps; ranges, warp and last warp from the 1 recording CTAs' 8 warps; span 0.19 us, warp 0.19 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.19 | 100.0% | 0.20 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### split_columns_kernel grid (6, 15, 1) block (128, 1, 1)

1 launches, 360 warps; ranges, warp and last warp from the 90 recording CTAs' 360 warps; span 1.38 us, warp 1.06 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.06 | 100.0% | 1.34 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### ced_copy_rows_kernel grid (30, 1, 1) block (128, 1, 1)

1 launches, 120 warps; ranges, warp and last warp from the 30 recording CTAs' 120 warps; span 1.25 us, warp 0.85 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.85 | 100.0% | 1.25 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|

### gemm_splitk_reduce_kernel grid (4, 6, 1) block (128, 1, 1)

3 launches, 96 warps; ranges, warp and last warp from the 24 recording CTAs' 96 warps; span 0.32 us, warp 0.27 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.27 | 100.0% | 0.31 | 100.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
