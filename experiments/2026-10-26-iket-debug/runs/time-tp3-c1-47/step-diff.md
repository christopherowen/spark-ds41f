## A decode step against the grid's step (17 decode steps, 9 grid runs)

GPU busy a step: 15514.5 us against 9728.5 us (+5786.0); kernel time a step: 17679.4 us against 11489.1 us (+6190.3).

| kernel | launches a step | grid | us a step | grid | difference us | exposed us | last warp: wait / load / compute / store / other | longest last-warp wait |
|---|---:|---:|---:|---:|---:|---:|---|---|
| vocab_packed_projection_kernel | 2 | 1 | 8163.0 | 4080.1 | +4083.0 | 8163.0 | 47% / 29% / 10% / 0% / 14% | wait.sync#2*17 |
| moe_grouped_fp8_fp4_gemm_kernel | 18 | 18 | 1748.1 | 109.3 | +1638.8 | 1436.8 | 68% / 16% / 21% / 1% / 0% | wait.mbarrier*40 |
| l2_prefetch_kernel | 25 | 25 | 1847.3 | 1730.1 | +117.2 | 245.1 | 0% / 100% / 0% / 0% / 0% |  |
| gemm_bf16_kernel | 5 | 5 | 1334.2 | 1283.3 | +50.9 | 1334.1 | 89% / 0% / 5% / 1% / 5% | wait.mbarrier#2 |
| mhc_finalize_kernel | 18 | 18 | 199.0 | 158.9 | +40.1 | 14.4 | 7% / 15% / 11% / 23% / 44% | wait.sync |
| token_ranks_kernel | 4 | 0 | 35.0 | 0.0 | +35.0 | 35.0 | 0% / 90% / 4% / 1% / 5% |  |
| indexer_topk_merge_kernel | 7 | 7 | 58.1 | 23.2 | +34.9 | 58.1 | 64% / 11% / 0% / 1% / 24% | wait.sync#20*9 |
| topk_log_softmax_kernel | 2 | 0 | 34.8 | 0.0 | +34.8 | 34.8 | 9% / 79% / 2% / 0% / 10% | wait.sync#2 |
| gemm_mxfp8_kernel | 48 | 48 | 2278.4 | 2246.2 | +32.2 | 1478.4 | 68% / 7% / 5% / 0% / 19% | wait.cp_async*22 |
| moe_swiglu_cast_kernel_kernel | 9 | 9 | 42.6 | 14.5 | +28.1 | 42.5 | 0% / 14% / 4% / 4% / 78% |  |
| router_partials_kernel | 9 | 9 | 48.2 | 20.7 | +27.5 | 0.5 | 0% / 35% / 24% / 3% / 37% |  |
| mhc_project_streams_kernel | 18 | 18 | 320.3 | 295.7 | +24.7 | 10.2 | 28% / 28% / 15% / 3% / 26% | wait.sync*4 |
| topk_ids_kernel | 1 | 0 | 23.3 | 0.0 | +23.3 | 23.3 | 3% / 74% / 0% / 14% / 9% | wait.sync |
| activation_swiglu_clamp_kernel | 9 | 9 | 37.6 | 17.0 | +20.6 | 1.4 | 0% / 7% / 0% / 2% / 91% |  |
| sparse_mla_kernel | 9 | 9 | 197.4 | 179.7 | +17.7 | 32.5 | 17% / 23% / 7% / 4% / 49% | wait.sync |
| local_stats_kernel | 1 | 0 | 13.1 | 0.0 | +13.1 | 13.1 | 1% / 91% / 1% / 1% / 6% | wait.sync |
| indexer_topk_segment_kernel | 3 | 3 | 20.5 | 8.0 | +12.5 | 20.5 | 36% / 10% / 0% / 3% / 51% | wait.sync |
| per_token_cast_kernel_kernel | 24 | 24 | 45.3 | 34.2 | +11.1 | 16.6 | 0% / 25% / 5% / 7% / 63% |  |
| moe_combine_kernel | 9 | 9 | 21.6 | 12.2 | +9.4 | 21.6 | 0% / 81% / 0% / 2% / 16% |  |
| metadata_decode_kernel | 4 | 4 | 27.8 | 19.7 | +8.0 | 4.9 | 0% / 72% / 0% / 10% / 18% |  |
| moe_dispatch_kernel | 9 | 9 | 30.6 | 23.9 | +6.7 | 2.5 | 36% / 2% / 0% / 26% / 36% | wait.sync#7*9 |
| cache_writer_kernel | 14 | 11 | 57.0 | 51.1 | +5.9 | 6.2 | 0% / 11% / 1% / 33% / 55% |  |
| mhc_post_kernel | 5 | 5 | 14.6 | 8.7 | +5.9 | 4.5 | 0% / 4% / 0% / 2% / 94% |  |
| gemm_bf16_partials_kernel | 13 | 13 | 200.8 | 195.6 | +5.2 | 9.2 | 39% / 0% / 23% / 1% / 36% | wait.mbarrier#2*10 |
| sink_merge_kernel | 1 | 0 | 4.4 | 0.0 | +4.4 | 4.4 | 7% / 34% / 9% / 2% / 47% | wait.sync*3 |
| gather_block_tables_kernel | 1 | 0 | 3.9 | 0.0 | +3.9 | 3.9 | 0% / 84% / 0% / 0% / 16% |  |
| metadata_builder_kernel | 17 | 17 | 35.6 | 31.7 | +3.9 | 35.6 | 0% / 71% / 0% / 4% / 25% |  |
| sink_candidates_kernel | 1 | 0 | 3.8 | 0.0 | +3.8 | 3.8 | 8% / 20% / 20% / 7% / 44% | wait.sync*4 |
| sink_cut_kernel | 2 | 0 | 3.7 | 0.0 | +3.7 | 3.7 | 0% / 34% / 0% / 43% / 22% |  |
| mhc_collapse_kernel | 4 | 4 | 17.8 | 14.1 | +3.7 | 1.2 | 0% / 10% / 0% / 2% / 88% |  |
| vectorized_elementwise_kernel | 0 | 87 | 0.0 | 101.6 | -101.6 | 0.0 |  |  |
