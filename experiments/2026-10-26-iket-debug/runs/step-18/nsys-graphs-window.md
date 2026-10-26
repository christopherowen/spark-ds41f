
## Timeline: 2636 launches in 226324.8 us; the GPU is busy 137114.2 us (61%) and idle 89210.6 us in 1690 gaps; gaps over 200 us split it into 5 segments

| segment | launches | wall us | busy us | busy | first kernel |
|---:|---:|---:|---:|---:|---|
| 0 | 29 | 1233.8 | 45.4 | 4% | fill_rows_kernel |
| 1 | 1278 | 71567.5 | 69265.6 | 97% | mhc_project_streams_kernel |
| 2 | 1 | 6.0 | 6.0 | 100% | padding_rows_kernel |
| 3 | 23 | 903.0 | 50.0 | 6% | assign_draft_budget_kernel |
| 4 | 1305 | 68720.6 | 67747.3 | 99% | mhc_project_streams_kernel |

| idle us | after | before |
|---:|---|---|
| 82969.3 | scatter_rows_kernel | padding_rows_kernel |
| 366.6 | indexSelectSmallIndex | mhc_project_streams_kernel |
| 322.0 | padding_rows_kernel | assign_draft_budget_kernel |
| 236.0 | vectorized_gather_kernel | mhc_project_streams_kernel |
| 124.2 | sink_policy_kernel | post_update_kernel |
| 115.3 | compute_slot_mappings_kernel | ced_requests_kernel |
| 114.9 | mhc_finalize_kernel | gemm_mxfp8_kernel |
| 108.7 | fill_rows_kernel | apply_write_kernel |
| 107.0 | padding_rows_kernel | prepare_prefill_inputs_kernel |
| 100.6 | ced_requests_kernel | metadata_builder_kernel |

## Kernels by total duration (2636 launches, 2050 in graphs)

| kernel | launches | in graphs | total us | share | mean us |
|---|---:|---:|---:|---:|---:|
| vocab_packed_projection_kernel | 12 | 6 | 48889.3 | 31.6% | 4074.11 |
| gemm_mxfp8_kernel | 288 | 255 | 34639.0 | 22.4% | 120.27 |
| moe_grouped_fp8_fp4_gemm_kernel | 108 | 96 | 26149.8 | 16.9% | 242.13 |
| l2_prefetch_kernel | 133 | 133 | 12440.8 | 8.0% | 93.54 |
| gemm_fp8w_bf16_partials_kernel | 54 | 48 | 8808.7 | 5.7% | 163.12 |
| gemm_bf16_kernel | 30 | 30 | 8013.3 | 5.2% | 267.11 |
| gemm_bf16_partials_kernel | 78 | 68 | 2874.6 | 1.9% | 36.85 |
| mhc_project_streams_kernel | 108 | 96 | 2047.0 | 1.3% | 18.95 |
| sparse_mla_kernel | 54 | 48 | 1683.4 | 1.1% | 31.17 |
| mhc_finalize_kernel | 108 | 96 | 1138.5 | 0.7% | 10.54 |
| moe_dispatch_kernel | 54 | 48 | 901.2 | 0.6% | 16.69 |
| gemm_bf16_splitk_kernel | 12 | 10 | 740.1 | 0.5% | 61.67 |
| router_partials_kernel | 54 | 48 | 547.2 | 0.4% | 10.13 |
| metadata_chunk_kernel | 30 | 28 | 460.3 | 0.3% | 15.34 |
| moe_swiglu_cast_kernel_kernel | 54 | 48 | 455.0 | 0.3% | 8.43 |
