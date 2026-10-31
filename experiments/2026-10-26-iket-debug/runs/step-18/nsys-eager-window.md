
## Timeline: 2066 launches in 190214.5 us; the GPU is busy 116024.5 us (61%) and idle 74190.0 us in 1781 gaps; gaps over 200 us split it into 5 segments

| segment | launches | wall us | busy us | busy | first kernel |
|---:|---:|---:|---:|---:|---|
| 0 | 28 | 1240.3 | 44.9 | 4% | fill_rows_kernel |
| 1 | 1210 | 72642.7 | 69345.8 | 95% | mhc_project_streams_kernel |
| 2 | 1 | 5.7 | 5.7 | 100% | padding_rows_kernel |
| 3 | 22 | 781.0 | 43.1 | 6% | expand_idx_mapping_kernel |
| 4 | 805 | 49415.5 | 46585.1 | 94% | mhc_project_streams_kernel |

| idle us | after | before |
|---:|---|---|
| 65284.6 | scatter_rows_gather_kernel | padding_rows_kernel |
| 359.4 | padding_rows_kernel | expand_idx_mapping_kernel |
| 267.3 | indexSelectSmallIndex | mhc_project_streams_kernel |
| 218.0 | vectorized_gather_kernel | mhc_project_streams_kernel |
| 183.1 | mhc_finalize_kernel | gemm_mxfp8_kernel |
| 111.7 | padding_rows_kernel | prepare_prefill_inputs_kernel |
| 107.6 | ced_requests_kernel | metadata_builder_kernel |
| 107.3 | apply_write_kernel | apply_write_kernel |
| 100.6 | compute_slot_mappings_kernel | ced_requests_kernel |
| 100.4 | gemm_mxfp8_kernel | norm_forward_and_per_token_cast_kernel_kernel |

## Kernels by total duration (2066 launches, 0 in graphs)

| kernel | launches | in graphs | total us | share | mean us |
|---|---:|---:|---:|---:|---:|
| vocab_packed_projection_kernel | 10 | 0 | 40079.2 | 32.9% | 4007.92 |
| gemm_mxfp8_kernel | 240 | 0 | 31945.0 | 26.2% | 133.10 |
| moe_grouped_fp8_fp4_gemm_kernel | 90 | 0 | 23504.0 | 19.3% | 261.16 |
| gemm_fp8w_bf16_partials_kernel | 45 | 0 | 9320.1 | 7.6% | 207.11 |
| gemm_bf16_kernel | 25 | 0 | 6755.3 | 5.5% | 270.21 |
| gemm_bf16_partials_kernel | 65 | 0 | 1375.8 | 1.1% | 21.17 |
| mhc_project_streams_kernel | 90 | 0 | 1282.0 | 1.1% | 14.24 |
| sparse_mla_kernel | 45 | 0 | 1146.9 | 0.9% | 25.49 |
| moe_swiglu_cast_kernel_kernel | 45 | 0 | 904.3 | 0.7% | 20.10 |
| moe_dispatch_kernel | 45 | 0 | 649.6 | 0.5% | 14.44 |
| router_partials_kernel | 45 | 0 | 598.9 | 0.5% | 13.31 |
| mhc_finalize_kernel | 90 | 0 | 549.7 | 0.5% | 6.11 |
| gemm_bf16_splitk_kernel | 10 | 0 | 398.2 | 0.3% | 39.82 |
| norm_forward_and_per_token_cast_kernel_kernel | 130 | 0 | 241.0 | 0.2% | 1.85 |
| per_token_cast_kernel_kernel | 120 | 0 | 235.6 | 0.2% | 1.96 |
