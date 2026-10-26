## A decode step against the grid's step (17 decode steps, 9 grid runs)

GPU busy a step: 14297.6 us against 8872.6 us (+5425.0); kernel time a step: 16126.1 us against 10511.1 us (+5615.0).

| kernel | launches a step | grid | us a step | grid | difference us | exposed us | last warp: wait / load / compute / store / other | longest last-warp wait |
|---|---:|---:|---:|---:|---:|---:|---|---|
| vocab_packed_projection_kernel | 2 | 1 | 8123.4 | 4070.3 | +4053.1 | 8123.4 | 65% / 29% / 11% / 0% / 0% | wait.cp_async*17 |
| moe_grouped_fp8_fp4_gemm_kernel | 18 | 18 | 1283.3 | 97.7 | +1185.6 | 1186.9 | 46% / 20% / 29% / 1% / 4% | wait.mbarrier*40 |
| l2_prefetch_kernel | 25 | 25 | 1723.8 | 1628.1 | +95.7 | 244.2 | 0% / 100% / 0% / 0% / 0% |  |
| token_ranks_kernel | 4 | 0 | 39.1 | 0.0 | +39.1 | 39.1 | 0% / 90% / 5% / 1% / 5% |  |
| mhc_project_streams_kernel | 18 | 18 | 311.0 | 276.3 | +34.7 | 13.2 | 18% / 31% / 17% / 3% / 30% | wait.sync*4 |
| topk_log_softmax_kernel | 2 | 0 | 33.1 | 0.0 | +33.1 | 33.1 | 5% / 82% / 2% / 0% / 10% | wait.sync#2 |
| indexer_topk_merge_kernel | 7 | 7 | 56.9 | 24.4 | +32.5 | 56.9 | 64% / 8% / 0% / 2% / 26% | wait.sync#20*9 |
| mhc_finalize_kernel | 18 | 18 | 194.2 | 167.0 | +27.2 | 16.7 | 6% / 16% / 9% / 20% / 50% | wait.sync |
| router_partials_kernel | 9 | 9 | 41.6 | 19.4 | +22.2 | 3.9 | 0% / 34% / 25% / 4% / 37% |  |
| topk_ids_kernel | 1 | 0 | 21.8 | 0.0 | +21.8 | 21.8 | 3% / 73% / 0% / 14% / 9% | wait.sync |
| gemm_bf16_kernel | 5 | 5 | 1311.7 | 1292.9 | +18.8 | 1311.7 | 89% / 0% / 5% / 1% / 5% | wait.mbarrier#2 |
| sparse_mla_kernel | 9 | 9 | 191.2 | 175.0 | +16.3 | 32.9 | 3% / 42% / 4% / 2% / 49% | wait.sync |
| moe_swiglu_cast_kernel_kernel | 9 | 9 | 29.6 | 15.1 | +14.5 | 29.6 | 0% / 20% / 17% / 10% / 54% |  |
| cache_writer_kernel | 14 | 11 | 68.8 | 54.4 | +14.4 | 6.6 | 0% / 9% / 3% / 43% / 45% |  |
| local_stats_kernel | 1 | 0 | 13.6 | 0.0 | +13.6 | 13.6 | 1% / 91% / 1% / 1% / 6% | wait.sync |
| indexer_topk_segment_kernel | 3 | 3 | 20.2 | 8.0 | +12.2 | 20.2 | 36% / 10% / 0% / 3% / 51% | wait.sync |
| moe_combine_kernel | 9 | 9 | 22.7 | 11.6 | +11.1 | 22.7 | 0% / 78% / 0% / 2% / 20% |  |
| mhc_post_kernel | 5 | 5 | 16.8 | 7.4 | +9.4 | 3.3 | 0% / 5% / 0% / 3% / 92% |  |
| moe_dispatch_kernel | 9 | 9 | 34.0 | 25.0 | +9.0 | 7.9 | 37% / 2% / 0% / 26% / 35% | wait.sync#7*9 |
| rope_kernel | 27 | 24 | 49.4 | 41.5 | +7.9 | 11.2 | 0% / 15% / 0% / 2% / 83% |  |
| metadata_decode_kernel | 4 | 4 | 27.3 | 21.2 | +6.1 | 4.3 | 0% / 59% / 0% / 14% / 28% |  |
| gemm_bf16_splitk_kernel | 2 | 2 | 120.3 | 114.3 | +6.0 | 0.9 | 116% / 8% / 4% / 0% / 0% | wait.cp_async*13 |
| compress_gather_partial_state_kernel | 1 | 1 | 11.8 | 6.2 | +5.6 | 0.0 | 0% / 51% / 0% / 17% / 32% |  |
| indexSelectSmallIndex | 7 | 6 | 17.9 | 12.9 | +5.0 | 17.9 |  |  |
| mhc_collapse_kernel | 4 | 4 | 19.5 | 14.6 | +4.9 | 1.0 | 0% / 6% / 0% / 2% / 92% |  |
| sink_merge_kernel | 1 | 0 | 4.6 | 0.0 | +4.6 | 4.6 | 11% / 48% / 13% / 3% / 26% | wait.sync*3 |
| sink_candidates_kernel | 1 | 0 | 4.3 | 0.0 | +4.3 | 4.3 | 9% / 20% / 20% / 7% / 44% | wait.sync#2*4 |
| sink_cut_kernel | 2 | 0 | 3.8 | 0.0 | +3.8 | 3.8 | 0% / 39% / 0% / 38% / 23% |  |
| sink_flags_kernel | 1 | 0 | 3.1 | 0.0 | +3.1 | 3.1 | 0% / 70% / 0% / 15% / 15% |  |
| compute_slot_mappings_kernel | 1 | 0 | 2.9 | 0.0 | +2.9 | 2.9 | 0% / 30% / 0% / 0% / 70% |  |
| vectorized_elementwise_kernel | 0 | 87 | 0.0 | 91.8 | -91.8 | 0.0 |  |  |
