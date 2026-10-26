=== records against the dry run's count
pid_0x3d: recommended_max_ts_cnt_per_warp 80
    62 events in one warp,    112 warps recording, bound 62: indexer_topk_segment_kernel
    60 events in one warp,     16 warps recording, bound 60: gemm_fp8w_bf16_cast_kernel
    60 events in one warp,     40 warps recording, bound 64: sparse_mla_kernel
    58 events in one warp,    112 warps recording, bound 58: indexer_topk_merge_kernel
    54 events in one warp,     16 warps recording, bound 54: gemm_fp8w_bf16_partials_kernel
    52 events in one warp,      8 warps recording, bound 52: adaptive_offsets_kernel
    50 events in one warp,     16 warps recording, bound 50: vocab_packed_projection_kernel
    50 events in one warp,     32 warps recording, bound 50: gemm_bf16_splitk_kernel
    32 events in one warp,     40 warps recording, bound 32: gemm_mxfp8_kernel
    30 events in one warp,     20 warps recording, bound 30: moe_grouped_fp8_fp4_gemm_kernel
    26 events in one warp,     16 warps recording, bound 26: gemm_mxfp8_partials_kernel
    24 events in one warp,     96 warps recording, bound 24: moe_dispatch_kernel
    24 events in one warp,    224 warps recording, bound 24: topk_log_softmax_kernel
    22 events in one warp,     16 warps recording, bound 22: mhc_project_streams_kernel
    22 events in one warp,     16 warps recording, bound 30: local_stats_kernel
    20 events in one warp,     28 warps recording, bound 20: norm_forward_and_per_token_cast_kernel_kernel
    20 events in one warp,     24 warps recording, bound 20: compress_kernel
    18 events in one warp,     16 warps recording, bound 18: indexer_score_kernel
    18 events in one warp,     28 warps recording, bound None: bias_kernel
    16 events in one warp,     14 warps recording, bound 16: router_partials_kernel
    16 events in one warp,      8 warps recording, bound 16: sink_merge_kernel
    14 events in one warp,     28 warps recording, bound 22: mhc_finalize_kernel
    14 events in one warp,     56 warps recording, bound 14: min_p_kernel
    14 events in one warp,      6 warps recording, bound 14: verify_kernel
    14 events in one warp,      2 warps recording, bound 14: insert_kernel
    12 events in one warp,     10 warps recording, bound 12: per_token_cast_kernel_kernel
    12 events in one warp,     20 warps recording, bound 12: sink_candidates_kernel
    12 events in one warp,     16 warps recording, bound 12: sparse_mla_kernel_1
    12 events in one warp,      2 warps recording, bound 18: sink_policy_kernel
    12 events in one warp,     20 warps recording, bound 12: resample_kernel
    10 events in one warp,      4 warps recording, bound 10: ced_requests_kernel
    10 events in one warp,     20 warps recording, bound 10: dspark_confidence_kernel
    10 events in one warp,     20 warps recording, bound 10: gumbel_sample_kernel
    10 events in one warp,     20 warps recording, bound 14: metadata_builder_kernel
    10 events in one warp,     30 warps recording, bound 10: metadata_chunk_kernel
    10 events in one warp,     24 warps recording, bound 14: metadata_decode_kernel
    10 events in one warp,    112 warps recording, bound 10: topk_ids_kernel
    10 events in one warp,      1 warps recording, bound 10: post_update_kernel
    10 events in one warp,      8 warps recording, bound 10: sink_shard_top_kernel
    10 events in one warp,      8 warps recording, bound 10: assign_draft_budget_kernel
     8 events in one warp,      8 warps recording, bound 10: moe_topk_gate_forward_kernel_kernel
     8 events in one warp,     48 warps recording, bound 12: gemm_bf16_kernel
     8 events in one warp,     32 warps recording, bound 12: gemm_bf16_partials_kernel
     8 events in one warp,     16 warps recording, bound 8: partial_argmax_kernel
     8 events in one warp,     28 warps recording, bound 8: cache_writer_kernel
     8 events in one warp,     32 warps recording, bound 12: gemm_bf16_two_kernel
     8 events in one warp,      8 warps recording, bound 8: compress_gather_partial_state_kernel
     8 events in one warp,     28 warps recording, bound 8: indexer_mxfp4_quantize_kernel
     8 events in one warp,     32 warps recording, bound 8: sink_rows_kernel
     6 events in one warp,      8 warps recording, bound 8: ced_indices_kernel
     6 events in one warp,     56 warps recording, bound 6: rope_kernel
     6 events in one warp,     16 warps recording, bound 6: routing_masks_kernel
     6 events in one warp,     36 warps recording, bound 6: moe_swiglu_cast_kernel_kernel
     6 events in one warp,     16 warps recording, bound 6: gemm_splitk_reduce_cast_kernel
     6 events in one warp,      6 warps recording, bound 6: finish_argmax_kernel
     6 events in one warp,      2 warps recording, bound 6: gumbel_finish_kernel
     6 events in one warp,      2 warps recording, bound 6: greedy_rejection_kernel
     6 events in one warp,     28 warps recording, bound 6: metadata_pages_kernel
     6 events in one warp,     48 warps recording, bound 8: prepare_dflash_inputs_kernel
     6 events in one warp,     32 warps recording, bound 8: grammar_bitmask_kernel
     6 events in one warp,     32 warps recording, bound 6: penalties_kernel
     6 events in one warp,     56 warps recording, bound 6: token_ranks_kernel
     6 events in one warp,      4 warps recording, bound 8: flatten_sampled_kernel
     4 events in one warp,     20 warps recording, bound 4: ced_copy_rows_kernel
     4 events in one warp,     21 warps recording, bound 4: activation_swiglu_clamp_kernel
     4 events in one warp,     20 warps recording, bound 4: moe_combine_kernel
     4 events in one warp,     20 warps recording, bound 4: mhc_post_kernel
     4 events in one warp,     20 warps recording, bound 4: mhc_collapse_kernel
     4 events in one warp,     20 warps recording, bound 4: mhc_collapse_column_kernel
     4 events in one warp,      4 warps recording, bound 4: logits_rows_kernel
     4 events in one warp,      8 warps recording, bound 4: token_counts_kernel
     4 events in one warp,      8 warps recording, bound 4: ced_routing_mask_kernel
     4 events in one warp,      8 warps recording, bound 4: ced_token_metadata_kernel
     4 events in one warp,     28 warps recording, bound 4: gemm_splitk_reduce_kernel
     4 events in one warp,      8 warps recording, bound 4: ced_slots_kernel
     4 events in one warp,      4 warps recording, bound 4: fill_rows_kernel
     4 events in one warp,     48 warps recording, bound 4: apply_write_kernel
     4 events in one warp,     20 warps recording, bound 4: clear_rows_kernel
     4 events in one warp,     24 warps recording, bound 6: prepare_pos_seq_lens_kernel
     4 events in one warp,      2 warps recording, bound 4: combine_sampled_and_draft_tokens_kernel
     4 events in one warp,      1 warps recording, bound 4: sink_flags_kernel
     4 events in one warp,      7 warps recording, bound 4: bad_words_kernel
     4 events in one warp,     32 warps recording, bound 4: temperature_kernel
     4 events in one warp,      4 warps recording, bound 4: gather_rows_kernel
     4 events in one warp,      4 warps recording, bound 4: scatter_rows_gather_kernel
     4 events in one warp,      4 warps recording, bound 4: sink_cut_kernel
     4 events in one warp,     20 warps recording, bound 4: context_inputs_kernel
     4 events in one warp,      4 warps recording, bound 4: scatter_rows_kernel
     2 events in one warp,     16 warps recording, bound 2: zero_packs_kernel
     2 events in one warp,     20 warps recording, bound 2: split_columns_kernel
     2 events in one warp,      8 warps recording, bound 2: padding_rows_kernel
     2 events in one warp,      8 warps recording, bound 2: bincount_kernel
     2 events in one warp,     16 warps recording, bound 2: prepare_prefill_inputs_kernel
     2 events in one warp,     48 warps recording, bound 2: gather_block_tables_kernel
     2 events in one warp,     48 warps recording, bound None: compute_slot_mappings_kernel
     2 events in one warp,     32 warps recording, bound 18: topp_stats_kernel
     2 events in one warp,     32 warps recording, bound 44: topp_step_kernel
     2 events in one warp,     32 warps recording, bound 36: topp_mask_kernel
     2 events in one warp,      8 warps recording, bound 2: prompt_ids_kernel
     2 events in one warp,      1 warps recording, bound 2: expand_idx_mapping_kernel
     2 events in one warp,      1 warps recording, bound 2: iket_step_marker_kernel
     2 events in one warp,      1 warps recording, bound 2: iket_grid_marker_kernel
     2 events in one warp,      8 warps recording, bound 2: l2_prefetch_kernel
=== records against the dry run's count: exit 0 in 15 s
=== decode steps, nsys, off, graphs

JIT kernel warmup (390 compile keys):   0%|          | 0/1 [00:00<?, ?kernel/s]
JIT kernel warmup (390 compile keys): 100%|██████████| 1/1 [00:03<00:00,  3.74s/kernel, TileLangKernels (390 keys)]
JIT kernel warmup (390 compile keys): 100%|██████████| 1/1 [00:03<00:00,  3.74s/kernel, TileLangKernels (390 keys)]

JIT kernel warmup (390 compile keys):   0%|          | 0/1 [00:00<?, ?kernel/s]INFO 10-26 14:31:40 [kernel_warmup.py:187] JIT kernel warmup finished in 0.00s.

JIT kernel warmup (390 compile keys): 100%|██████████| 1/1 [00:00<00:00, 767.91kernel/s, TileLangKernels (390 keys)]
2026-10-26 14:31:40,957 - INFO - autotuner.py:2789 - flashinfer.jit: [Autotuner]: Loaded 0 configs from /tmp/tmp6n49slbk/autotune_configs.json
2026-10-26 14:31:40,957 - INFO - autotuner.py:972 - flashinfer.jit: [Autotuner]: Autotuning process starts ...
2026-10-26 14:31:40,984 - INFO - autotuner.py:995 - flashinfer.jit: [Autotuner]: Autotuning process ends
2026-10-26 14:31:40,988 - INFO - autotuner.py:2652 - flashinfer.jit: [Autotuner]: Saved 0 configs to /c/vllm-iket-step-tp4-c1/flashinfer_autotune_cache/0.6.18.post1/121a/a955cc9b5f8421eadd8239a57c43f3b40dd7eea67e221bff2e281e9f0a561895/tmp6t_igl_d/autotune_configs.json (0 new, 0 from previous config)





engine ready in 22 s




{"mode": "off", "boot_s": 22.2, "request_s": [0.239, 0.34], "requests": 1, "tokens": 32, "prompt_tokens": 32, "token_ids": [74521, 32271, 74521, 32271, 74521, 32271, 74521, 32271, 74521, 32271, 74521, 32271, 74521, 32271, 74521, 32271, 74521, 32271, 74521, 32271, 74521, 32271, 74521, 32271, 74521, 32271, 74521, 32271, 74521, 32271, 74521, 32271], "generated": [32], "logprobs": [{"27564": -11.769643783569336, "35102": -11.769641876220703, "73830": -11.769646644592285, "74521": -11.769640922546387, "113611": -11.769644737243652}, {"5780": -11.769625663757324, "32271": -11.769623756408691, "85998": -11.769633293151855, "90689": -11.769634246826172, "116208": -11.769635200500488}, {"10053": -11.769635200500488, "40430": -11.769641876220703, "41562": -11.769638061523438, "74521": -11.769625663757324, "119506": -11.769640922546387}, {"5780": -11.769625663757324, "32271": -11.769623756408691, "85998": -11.769633293151855, "90689": -11.769634246826172, "116208": -11.769635200500488}, {"10053": -11.769635200500488, "40430": -11.769641876220703, "41562": -11.769638061523438, "74521": -11.769625663757324, "119506": -11.769640922546387}, {"5780": -11.769625663757324, "32271": -11.769623756408691, "85998": -11.769633293151855, "90689": -11.769634246826172, "116208": -11.769635200500488}, {"10053": -11.769635200500488, "40430": -11.769641876220703, "41562": -11.769638061523438, "74521": -11.769625663757324, "119506": -11.769640922546387}, {"5780": -11.769625663757324, "32271": -11.769623756408691, "85998": -11.769633293151855, "90689": -11.769634246826172, "116208": -11.769635200500488}, {"10053": -11.769635200500488, "40430": -11.769641876220703, "41562": -11.769638061523438, "74521": -11.769625663757324, "119506": -11.769640922546387}, {"5780": -11.769625663757324, "32271": -11.769623756408691, "85998": -11.769633293151855, "90689": -11.769634246826172, "116208": -11.769635200500488}, {"10053": -11.769635200500488, "40430": -11.769641876220703, "41562": -11.769638061523438, "74521": -11.769625663757324, "119506": -11.769640922546387}, {"5780": -11.769625663757324, "32271": -11.769623756408691, "85998": -11.769633293151855, "90689": -11.769634246826172, "116208": -11.769635200500488}, {"10053": -11.769635200500488, "40430": -11.769641876220703, "41562": -11.769638061523438, "74521": -11.769625663757324, "119506": -11.769640922546387}, {"5780": -11.769625663757324, "32271": -11.769623756408691, "85998": -11.769633293151855, "90689": -11.769634246826172, "116208": -11.769635200500488}, {"10053": -11.769635200500488, "40430": -11.769641876220703, "41562": -11.769638061523438, "74521": -11.769625663757324, "119506": -11.769640922546387}, {"5780": -11.769625663757324, "32271": -11.769623756408691, "85998": -11.769633293151855, "90689": -11.769634246826172, "116208": -11.769635200500488}, {"10053": -11.769635200500488, "40430": -11.769641876220703, "41562": -11.769638061523438, "74521": -11.769625663757324, "119506": -11.769640922546387}, {"5780": -11.769625663757324, "32271": -11.769623756408691, "85998": -11.769633293151855, "90689": -11.769634246826172, "116208": -11.769635200500488}, {"10053": -11.769635200500488, "40430": -11.769641876220703, "41562": -11.769638061523438, "74521": -11.769625663757324, "119506": -11.769640922546387}, {"5780": -11.769625663757324, "32271": -11.769623756408691, "85998": -11.769633293151855, "90689": -11.769634246826172, "116208": -11.769635200500488}, {"10053": -11.769635200500488, "40430": -11.769641876220703, "41562": -11.769638061523438, "74521": -11.769625663757324, "119506": -11.769640922546387}, {"5780": -11.769625663757324, "32271": -11.769623756408691, "85998": -11.769633293151855, "90689": -11.769634246826172, "116208": -11.769635200500488}, {"10053": -11.769635200500488, "40430": -11.769641876220703, "41562": -11.769638061523438, "74521": -11.769625663757324, "119506": -11.769640922546387}, {"5780": -11.769625663757324, "32271": -11.769623756408691, "85998": -11.769633293151855, "90689": -11.769634246826172, "116208": -11.769635200500488}, {"10053": -11.769635200500488, "40430": -11.769641876220703, "41562": -11.769638061523438, "74521": -11.769625663757324, "119506": -11.769640922546387}, {"5780": -11.769625663757324, "32271": -11.769623756408691, "85998": -11.769633293151855, "90689": -11.769634246826172, "116208": -11.769635200500488}, {"10053": -11.769635200500488, "40430": -11.769641876220703, "41562": -11.769638061523438, "74521": -11.769625663757324, "119506": -11.769640922546387}, {"5780": -11.769625663757324, "32271": -11.769623756408691, "85998": -11.769633293151855, "90689": -11.769634246826172, "116208": -11.769635200500488}, {"10053": -11.769635200500488, "40430": -11.769641876220703, "41562": -11.769638061523438, "74521": -11.769625663757324, "119506": -11.769640922546387}, {"5780": -11.769625663757324, "32271": -11.769623756408691, "85998": -11.769633293151855, "90689": -11.769634246826172, "116208": -11.769635200500488}, {"10053": -11.769635200500488, "40430": -11.769641876220703, "41562": -11.769638061523438, "74521": -11.769625663757324, "119506": -11.769640922546387}, {"5780": -11.769625663757324, "32271": -11.769623756408691, "85998": -11.769633293151855, "90689": -11.769634246826172, "116208": -11.769635200500488}], "batched": 512, "chunk": 256, "VLLM_DS41_IKET": null, "VLLM_DS41_IKET_CTAS": 0, "VLLM_DS41_IKET_ITERS": 0, "VLLM_DS41_IKET_SKIP": "", "filled": false, "eager": false, "iket_modules": ["cutlass._mlir.dialects.iket", "cutlass.cute.experimental.iket", "cutlass.cute.experimental.iket.iket"], "grid": {"spec": {"shapes": [{"num_reqs": 1, "context_len": 48, "num_tokens": 6}], "replays": 8, "warmup": 1, "settle": 0}, "runs": 9, "step_ms": [17.033376693725586, 15.341631889343262, 12.460127830505371, 10.096799850463867, 10.262751579284668, 10.01801586151123, 10.233023643493652, 10.072575569152832], "forward_ms": [3.6409919261932373, 2.3305599689483643, 2.2515199184417725, 2.2240960597991943, 2.28387188911438, 2.200223922729492, 2.3060479164123535, 2.2167038917541504], "drafter_ms": [7.790304183959961, 7.243936061859131, 6.645823955535889, 6.7300801277160645, 6.855584144592285, 6.7188801765441895, 6.810431957244873, 6.753024101257324]}, "tp_shape": 4, "text_overrides": {"num_hidden_layers": 6, "compress_ratios": [0, 0, 2, 2, 1, 1, 0, 0, 0], "kv_source_layer_ids": [2, 4], "index_source_layer_ids": [2, 4], "candidate_source_layer_id": 4, "engram_layer_ids": [], "dspark_target_layer_ids": [3, 4, 5], "num_attention_heads": 16, "o_groups": 2, "moe_intermediate_size": 576}, "drafts": 5}
engine shut down: 6.8 GiB freed, 11.5 GiB still allocated
Collecting data...
Generating '/tmp/nsys-report-975a.qdstrm'


Generated:
	/c/iket/step-tp4-c1-20261026T142439Z/nsys-graphs.nsys-rep
	/c/iket/step-tp4-c1-20261026T142439Z/nsys-graphs.sqlite
=== decode steps, nsys, off, graphs: exit 0 in 32 s
=== timeline and tables
trace /c/iket/step-tp4-c1-20261026T142439Z/trace/iket/pid_0x188f/iket.decoded_results.json: 12587 launches, graphLaunches dict of 118

## Timeline: 7322 launches in 737981.0 us; the GPU is busy 244401.0 us (33%) and idle 493579.9 us in 4736 gaps; gaps over 200 us split it into 395 segments

The 15 of 395 segments with the most busy time:

| segment | launches | wall us | busy us | busy | first kernel |
|---:|---:|---:|---:|---:|---|
| 1 | 408 | 33607.4 | 16412.0 | 49% | apply_write_kernel |
| 32 | 116 | 6580.8 | 6500.4 | 99% | mhc_project_streams_kernel |
| 52 | 120 | 6857.7 | 6444.9 | 94% | metadata_builder_kernel |
| 81 | 116 | 6530.0 | 6453.0 | 99% | mhc_project_streams_kernel |
| 96 | 120 | 6885.6 | 6447.9 | 94% | metadata_builder_kernel |
| 133 | 116 | 6540.8 | 6455.3 | 99% | mhc_project_streams_kernel |
| 152 | 120 | 7256.6 | 6846.6 | 94% | metadata_builder_kernel |
| 167 | 120 | 6903.1 | 6432.1 | 93% | metadata_builder_kernel |
| 180 | 118 | 6850.5 | 6435.5 | 94% | metadata_builder_kernel |
| 201 | 118 | 6771.3 | 6436.8 | 95% | metadata_builder_kernel |
| 234 | 116 | 6528.7 | 6432.8 | 99% | mhc_project_streams_kernel |
| 323 | 120 | 6915.6 | 6434.3 | 93% | metadata_builder_kernel |
| 347 | 116 | 6521.2 | 6434.1 | 99% | mhc_project_streams_kernel |
| 359 | 116 | 6509.7 | 6424.2 | 99% | mhc_project_streams_kernel |
| 392 | 120 | 6830.0 | 6456.0 | 95% | metadata_builder_kernel |

| idle us | after | before |
|---:|---|---|
| 97120.5 | resample_kernel | insert_kernel |
| 90179.5 | scatter_rows_kernel | padding_rows_kernel |
| 3431.7 | norm_forward_and_per_token_cast_kernel_kernel | sink_flags_kernel |
| 3090.7 | dspark_confidence_kernel | scatter_rows_gather_kernel |
| 3009.5 | dspark_confidence_kernel | scatter_rows_gather_kernel |
| 2918.7 | dspark_confidence_kernel | scatter_rows_gather_kernel |
| 2913.5 | dspark_confidence_kernel | scatter_rows_gather_kernel |
| 2898.8 | dspark_confidence_kernel | scatter_rows_gather_kernel |
| 2896.2 | dspark_confidence_kernel | scatter_rows_gather_kernel |
| 2894.1 | dspark_confidence_kernel | scatter_rows_gather_kernel |

## Engine steps

| step | kind | launches | wall us | busy us | busy | most span |
|---:|---|---:|---:|---:|---:|---|
| 0 | prefill | 413 | 37987.0 | 16419.4 | 43% | vocab_packed_projection_kernel 8155, gemm_mxfp8_kernel 2659, moe_grouped_fp8_fp4_gemm_kernel 2167 |
| 1 | decode | 432 | 131878.4 | 14333.7 | 11% | vocab_packed_projection_kernel 8141, l2_prefetch_kernel 1701, gemm_mxfp8_kernel 1536 |
| 2 | decode | 432 | 120742.7 | 14228.4 | 12% | vocab_packed_projection_kernel 8046, l2_prefetch_kernel 1945, gemm_mxfp8_kernel 1687 |
| 3 | decode | 432 | 30446.1 | 14314.1 | 47% | vocab_packed_projection_kernel 8098, l2_prefetch_kernel 2107, gemm_mxfp8_kernel 1794 |
| 4 | decode | 432 | 29690.3 | 14228.0 | 48% | vocab_packed_projection_kernel 8055, l2_prefetch_kernel 1505, gemm_mxfp8_kernel 1490 |
| 5 | decode | 432 | 33614.6 | 14290.7 | 43% | vocab_packed_projection_kernel 8095, l2_prefetch_kernel 1507, gemm_mxfp8_kernel 1490 |
| 6 | decode | 432 | 29160.2 | 14608.1 | 50% | vocab_packed_projection_kernel 8305, l2_prefetch_kernel 1522, gemm_mxfp8_kernel 1488 |
| 7 | decode | 432 | 27215.4 | 14279.2 | 52% | vocab_packed_projection_kernel 8117, l2_prefetch_kernel 1520, gemm_mxfp8_kernel 1424 |
| 8 | decode | 432 | 27505.7 | 14179.5 | 52% | vocab_packed_projection_kernel 8030, l2_prefetch_kernel 1507, gemm_mxfp8_kernel 1464 |
| 9 | decode | 432 | 30784.2 | 14178.8 | 46% | vocab_packed_projection_kernel 8024, l2_prefetch_kernel 1509, gemm_mxfp8_kernel 1459 |
| 10 | decode | 432 | 35548.7 | 14190.2 | 40% | vocab_packed_projection_kernel 8028, l2_prefetch_kernel 1503, gemm_mxfp8_kernel 1448 |
| 11 | decode | 432 | 34280.2 | 14183.9 | 41% | vocab_packed_projection_kernel 8051, l2_prefetch_kernel 1513, gemm_mxfp8_kernel 1458 |
| 12 | decode | 432 | 35136.8 | 14171.3 | 40% | vocab_packed_projection_kernel 8032, l2_prefetch_kernel 1510, gemm_mxfp8_kernel 1442 |
| 13 | decode | 432 | 34172.6 | 14230.1 | 42% | vocab_packed_projection_kernel 8056, l2_prefetch_kernel 1504, gemm_mxfp8_kernel 1469 |
| 14 | decode | 432 | 33312.6 | 14200.1 | 43% | vocab_packed_projection_kernel 8042, l2_prefetch_kernel 1521, gemm_mxfp8_kernel 1434 |
| 15 | decode | 432 | 27610.8 | 14184.8 | 51% | vocab_packed_projection_kernel 8027, l2_prefetch_kernel 1522, gemm_mxfp8_kernel 1426 |
| 16 | decode | 429 | 34971.5 | 14180.8 | 41% | vocab_packed_projection_kernel 8025, l2_prefetch_kernel 1508, gemm_mxfp8_kernel 1447 |

## Overview over 16 decode steps: 6909 launches of 142 kernel shapes, spans summing to 252866.4 us (15804.1 a step)

| kernel | grid | launches a step | span us a step | share | span us a launch | last warp: wait | load | compute | store | unattributed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vocab_packed_projection_kernel | (1, 4040, 1) | 2 | 8073.2 | 51.1% | 4017.60 | 59% | 32% | 10% | 0% | 0% |
| l2_prefetch_kernel | (2, 1, 1) | 25 | 1587.7 | 10.0% | 63.42 | 0% | 100% | 0% | 0% | 0% |
| gemm_bf16_kernel | (2020, 1, 1) | 5 | 1328.3 | 8.4% | 263.62 | 91% | 0% | 4% | 1% | 5% |
| moe_grouped_fp8_fp4_gemm_kernel | (9, 30, 1) | 8.625 | 713.9 | 4.5% | 102.69 | 37% | 25% | 36% | 1% | 1% |
| moe_grouped_fp8_fp4_gemm_kernel | (40, 30, 1) | 8.625 | 444.4 | 2.8% | 59.31 | 14% | 20% | 0% | 0% | 66% |
| gemm_mxfp8_kernel | (80, 1, 1) | 1 | 397.0 | 2.5% | 398.42 | 94% | 4% | 5% | 0% | 0% |
| gemm_mxfp8_kernel | (64, 1, 1) | 11 | 347.2 | 2.2% | 23.31 | 29% | 10% | 15% | 0% | 46% |
| gemm_mxfp8_kernel | (28, 1, 1) | 9 | 320.0 | 2.0% | 35.70 | 38% | 11% | 11% | 0% | 40% |
| mhc_project_streams_kernel | (40, 1, 1) | 18 | 286.4 | 1.8% | 15.63 | 17% | 37% | 17% | 4% | 26% |
| gemm_fp8w_bf16_partials_kernel | (32, 4, 1) | 9 | 220.7 | 1.4% | 18.10 | 38% | 18% | 11% | 0% | 33% |
| gemm_mxfp8_kernel | (36, 1, 1) | 9 | 191.8 | 1.2% | 19.47 | 36% | 18% | 17% | 0% | 29% |
| gemm_mxfp8_kernel | (40, 1, 1) | 9 | 169.3 | 1.1% | 17.89 | 51% | 18% | 17% | 0% | 14% |
| gemm_bf16_partials_kernel | (6, 8, 1) | 6 | 153.0 | 1.0% | 24.77 | 35% | 0% | 32% | 2% | 31% |
| mhc_finalize_kernel | (5, 1, 1) | 11.25 | 151.7 | 1.0% | 13.90 | 5% | 12% | 11% | 18% | 54% |
| gemm_bf16_splitk_kernel | (16, 5, 1) | 2 | 113.3 | 0.7% | 57.57 | 90% | 7% | 4% | 0% | 0% |
| sparse_mla_kernel | (5, 1, 5) | 3.75 | 82.8 | 0.5% | 15.86 | 3% | 41% | 4% | 2% | 49% |
| mhc_finalize_kernel | (10, 1, 1) | 6 | 73.9 | 0.5% | 13.04 | 5% | 13% | 10% | 16% | 55% |
| gemm_mxfp8_kernel | (40, 1, 1) | 9 | 72.2 | 0.5% | 7.28 | 51% | 12% | 20% | 1% | 15% |
| sparse_mla_kernel | (2, 1, 10) | 3 | 62.7 | 0.4% | 21.71 | 16% | 23% | 23% | 2% | 36% |
| metadata_chunk_kernel | (10, 1, 1) | 3 | 47.1 | 0.3% | 12.45 | 0% | 83% | 0% | 1% | 16% |

| wait | kernel | grid | last-warp us a step | a launch | first statement |
|---|---|---|---:|---:|---|
| wait.cp_async*118 | gemm_mxfp8_kernel | (80, 1, 1) | 266.9 | 266.92 | `tl::cp_async_wait<2>();` |
| wait.sync#2*118 | gemm_mxfp8_kernel | (80, 1, 1) | 97.2 | 97.23 | `__syncthreads();` |
| wait.cp_async*19 | gemm_mxfp8_kernel | (28, 1, 1) | 72.7 | 8.08 | `tl::cp_async_wait<1>();` |
| wait.cp_async*13 | gemm_bf16_splitk_kernel | (16, 5, 1) | 47.8 | 23.92 | `tl::cp_async_wait<2>();` |
| wait.sync*4 | mhc_project_streams_kernel | (40, 1, 1) | 42.5 | 2.36 | `__syncthreads();` |
| wait.cp_async*14 | gemm_mxfp8_kernel | (40, 1, 1) | 37.0 | 4.12 | `tl::cp_async_wait<2>();` |
| wait.mbarrier*40 | moe_grouped_fp8_fp4_gemm_kernel | (9, 30, 1) | 36.9 | 4.28 | `loaded[(ko % 3)].wait(((ko % 6) / 3));` |
| wait.sync#2*17 | vocab_packed_projection_kernel | (1, 4040, 1) | 34.7 | 17.36 | `__syncthreads();` |
| wait.mbarrier#2 | gemm_bf16_kernel | (2020, 1, 1) | 31.9 | 6.38 | `mbarrier[0].wait(0);` |
| wait.sync#2*19 | gemm_mxfp8_kernel | (28, 1, 1) | 31.2 | 3.47 | `__syncthreads();` |
| wait.sync#2*14 | gemm_mxfp8_kernel | (40, 1, 1) | 30.1 | 3.35 | `__syncthreads();` |
| wait.sync#2*18 | gemm_mxfp8_kernel | (36, 1, 1) | 28.9 | 3.21 | `__syncthreads();` |
| wait.cp_async*18 | gemm_mxfp8_kernel | (36, 1, 1) | 25.5 | 2.83 | `tl::cp_async_wait<2>();` |
| wait.cp_async*40 | moe_grouped_fp8_fp4_gemm_kernel | (9, 30, 1) | 23.0 | 2.67 | `tl::cp_async_wait<2>();` |
| wait.sync#4*40 | moe_grouped_fp8_fp4_gemm_kernel | (9, 30, 1) | 21.0 | 2.43 | `__syncthreads();` |
| wait.sync#5*40 | moe_grouped_fp8_fp4_gemm_kernel | (9, 30, 1) | 20.7 | 2.40 | `__syncthreads();` |
| wait.sync#6*40 | moe_grouped_fp8_fp4_gemm_kernel | (9, 30, 1) | 20.3 | 2.36 | `__syncthreads();` |
| wait.cp_async*8 | gemm_mxfp8_kernel | (40, 1, 1) | 16.6 | 1.85 | `tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| wait.cp_async*14 | gemm_fp8w_bf16_partials_kernel | (32, 4, 1) | 15.3 | 1.70 | `tl::cp_async_wait<3>();` |
| wait.cp_async*17 | vocab_packed_projection_kernel | (1, 4040, 1) | 14.3 | 7.14 | `tl::cp_async_wait<6>();` |

### vocab_packed_projection_kernel grid (1, 4040, 1) block (128, 1, 1)

32 launches, 16160 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 4017.60 us, warp 49.54 us, nsys graphs 3984.08 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 30.48 | 61.5% | 30.29 | 58.9% |
| load | 11.03 | 22.3% | 16.58 | 32.3% |
| compute | 5.03 | 10.2% | 5.03 | 9.8% |
| unattributed | 2.94 | 5.9% | 0.00 | 0.0% |
| store | 0.05 | 0.1% | 0.05 | 0.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.cp_async*17 (est.: sampled iterations x17) | 13.97 | 28.2% | 1.0 | 44.06 | `tl::cp_async_wait<6>();` |
| wait.sync#2*17 (est.: sampled iterations x17) | 11.07 | 22.3% | 1.0 | 32.64 | `__syncthreads();` |
| load.global | 8.54 | 17.2% | 1.0 | 14.88 | `int start = row_offsets[(((int)blockIdx.y) * 16)]; \| int start = row_offsets[(((int)blockI` |
| compute.mma*17 (est.: sampled iterations x17) | 4.34 | 8.8% | 1.0 | 4.35 | `{` |
| load.cp_async#2*17 (est.: sampled iterations x17) | 2.21 | 4.5% | 1.0 | 2.72 | `for (int i_6 = 0; i_6 < 8; ++i_6) \| for (int i_9 = 0; i_9 < 4; ++i_9)` |
| wait.sync#3*17 (est.: sampled iterations x17) | 1.09 | 2.2% | 1.0 | 1.63 | `__syncthreads();` |
| wait.sync*17 (est.: sampled iterations x17) | 1.07 | 2.2% | 1.0 | 1.09 | `__syncthreads();` |
| wait.cp_async#2*17 (est.: sampled iterations x17) | 0.95 | 1.9% | 1.0 | 1.09 | `tl::cp_async_wait<7>();` |
| wait.cp_async#3 | 0.45 | 0.9% | 1.0 | 2.05 | `tl::cp_async_wait<4>();` |
| wait.sync#4 | 0.42 | 0.9% | 1.0 | 1.36 | `__syncthreads();` |
| wait.sync#6 | 0.37 | 0.8% | 1.0 | 1.31 | `__syncthreads();` |
| wait.sync#8 | 0.35 | 0.7% | 1.0 | 1.20 | `__syncthreads();` |
| load.cp_async | 0.28 | 0.6% | 1.0 | 0.32 | `for (int i = 0; i < 4; ++i) \| for (int i = 0; i < 8; ++i) \| for (int i_1 = 0; i_1 < 4; ++i` |
| wait.cp_async#5 | 0.28 | 0.6% | 1.0 | 1.52 | `tl::cp_async_wait<2>();` |
| compute.mma#2 | 0.24 | 0.5% | 1.0 | 0.26 | `{` |
| compute.mma#4 | 0.23 | 0.5% | 1.0 | 0.26 | `{` |
| compute.mma#3 | 0.23 | 0.5% | 1.0 | 0.26 | `{` |
| wait.cp_async#7 | 0.17 | 0.4% | 1.0 | 1.18 | `tl::cp_async_wait<0>();` |
| wait.sync#5 | 0.06 | 0.1% | 1.0 | 0.08 | `__syncthreads();` |
| wait.sync#9 | 0.05 | 0.1% | 1.0 | 0.06 | `__syncthreads();` |
| store.global | 0.05 | 0.1% | 1.0 | 0.06 | `for (int i_13 = 0; i_13 < 2; ++i_13) \| for (int i_14 = 0; i_14 < 4; ++i_14) \| for (int i_9` |
| wait.sync#7 | 0.05 | 0.1% | 1.0 | 0.06 | `__syncthreads();` |
| wait.cp_async#4 | 0.04 | 0.1% | 1.0 | 0.06 | `tl::cp_async_wait<5>();` |
| wait.cp_async#6 | 0.04 | 0.1% | 1.0 | 0.06 | `tl::cp_async_wait<3>();` |
| wait.cp_async#8 | 0.04 | 0.1% | 1.0 | 0.06 | `tl::cp_async_wait<1>();` |

### l2_prefetch_kernel grid (2, 1, 1) block (128, 1, 1)

400 launches, 8 warps; ranges, warp and last warp from the 2 recording CTAs' 8 warps; span 63.42 us, warp 62.07 us, nsys graphs 65.06 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 61.77 | 99.5% | 63.22 | 99.5% |
| unattributed | 0.30 | 0.5% | 0.29 | 0.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 61.77 | 99.5% | 1.0 | 63.12 | `for (int s = 0; s < (nseg2 >> 1); ++s)` |

### gemm_bf16_kernel grid (2020, 1, 1) block (256, 1, 1)

80 launches, 16160 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 263.62 us, warp 3.82 us, nsys graphs 264.03 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 3.24 | 84.9% | 6.47 | 90.5% |
| unattributed | 0.36 | 9.5% | 0.33 | 4.6% |
| compute | 0.14 | 3.7% | 0.29 | 4.0% |
| load | 0.04 | 0.9% | 0.00 | 0.0% |
| store | 0.04 | 0.9% | 0.06 | 0.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.mbarrier#2 | 3.14 | 82.1% | 0.5 | 7.47 | `mbarrier[0].wait(0);` |
| compute.mma | 0.14 | 3.7% | 0.5 | 0.29 | `{` |
| wait.sync | 0.07 | 2.0% | 1.0 | 0.10 | `__syncthreads();` |
| load.tma | 0.04 | 0.9% | 0.5 | 0.13 | `if (tl::tl_shuffle_elect<128>())` |
| store.global | 0.04 | 0.9% | 0.5 | 0.10 | `for (int i_3 = 0; i_3 < 4; ++i_3) \| for (int i_4 = 0; i_4 < 8; ++i_4) \| for (int i_6 = 0; ` |
| wait.mbarrier | 0.03 | 0.8% | 0.5 | 0.06 | `mbarrier[2].wait(1);` |

### moe_grouped_fp8_fp4_gemm_kernel grid (9, 30, 1) block (128, 1, 1)

138 launches, 1080 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 102.69 us, warp 20.79 us, nsys graphs 52.80 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 6.67 | 32.1% | 0.22 | 0.6% |
| wait | 5.87 | 28.2% | 14.28 | 37.3% |
| compute | 5.38 | 25.9% | 13.71 | 35.8% |
| load | 2.76 | 13.3% | 9.70 | 25.3% |
| store | 0.12 | 0.6% | 0.36 | 0.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*40 (est.: sampled iterations x40) | 5.38 | 25.9% | 0.4 | 14.08 | `{` |
| load#2*40 (est.: sampled iterations x40) | 2.37 | 11.4% | 0.4 | 8.96 | `if (ko < 38)` |
| wait.sync#5*40 (est.: sampled iterations x40) | 1.79 | 8.6% | 0.4 | 6.40 | `__syncthreads();` |
| wait.mbarrier*40 (est.: sampled iterations x40) | 1.28 | 6.2% | 0.4 | 3.84 | `loaded[(ko % 3)].wait(((ko % 6) / 3));` |
| wait.sync#4*40 (est.: sampled iterations x40) | 0.96 | 4.6% | 0.4 | 2.56 | `__syncthreads();` |
| wait.sync#6*40 (est.: sampled iterations x40) | 0.90 | 4.3% | 0.4 | 2.56 | `__syncthreads();` |
| wait.cp_async*40 (est.: sampled iterations x40) | 0.83 | 4.0% | 0.4 | 2.56 | `tl::cp_async_wait<2>();` |
| load.global#3 | 0.19 | 0.9% | 0.4 | 0.53 | `if (((expert < 128) && (0 <= expert))) \| if (((expert < 384) && (0 <= expert)))` |
| store.global | 0.12 | 0.6% | 0.4 | 0.35 | `for (int i_2 = 0; i_2 < 16; ++i_2) \| for (int i_3 = 0; i_3 < 16; ++i_3) \| for (int i_4 = 0` |
| load.global | 0.10 | 0.5% | 1.0 | 0.29 | `int expert = TileExpert[((int64_t)((int)blockIdx.y))];` |
| load | 0.07 | 0.3% | 0.4 | 0.26 | `for (int s = 0; s < 2; ++s)` |
| wait.sync | 0.05 | 0.2% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#2 | 0.05 | 0.2% | 0.4 | 0.16 | `__syncthreads();` |
| load.global#2 | 0.04 | 0.2% | 0.4 | 0.19 | `if (((int)threadIdx.x) < 32) \| if (((int)threadIdx.x) < 64)` |
| wait.sync#3 | 0.01 | 0.1% | 0.4 | 0.06 | `__syncthreads();` |

### moe_grouped_fp8_fp4_gemm_kernel grid (40, 30, 1) block (128, 1, 1)

138 launches, 4800 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 59.31 us, warp 4.57 us, nsys graphs 31.89 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.74 | 38.0% | 0.20 | 66.1% |
| load | 0.97 | 21.1% | 0.06 | 20.3% |
| compute | 0.87 | 19.0% | 0.00 | 0.0% |
| wait | 0.85 | 18.7% | 0.04 | 13.6% |
| store | 0.14 | 3.2% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*5 (est.: sampled iterations x5) | 0.87 | 19.0% | 0.5 | 1.76 | `if (ko < 4)` |
| load#2*5 (est.: sampled iterations x5) | 0.44 | 9.6% | 0.5 | 0.96 | `if (ko < 3)` |
| load.global#3 | 0.30 | 6.6% | 0.5 | 0.74 | `if (((expert < 128) && (0 <= expert))) \| if (((expert < 384) && (0 <= expert)))` |
| wait.sync#5*5 (est.: sampled iterations x5) | 0.17 | 3.7% | 0.5 | 0.48 | `__syncthreads();` |
| wait.mbarrier*5 (est.: sampled iterations x5) | 0.16 | 3.5% | 0.5 | 0.48 | `loaded[(ko % 3)].wait((ko / 3));` |
| store.global | 0.14 | 3.2% | 0.5 | 0.38 | `for (int i_2 = 0; i_2 < 16; ++i_2) \| for (int i_3 = 0; i_3 < 16; ++i_3) \| for (int i_4 = 0` |
| wait.sync#4*5 (est.: sampled iterations x5) | 0.14 | 3.1% | 0.5 | 0.32 | `__syncthreads();` |
| wait.sync#6*5 (est.: sampled iterations x5) | 0.14 | 3.1% | 0.5 | 0.32 | `__syncthreads();` |
| wait.cp_async*5 (est.: sampled iterations x5) | 0.12 | 2.6% | 0.5 | 0.32 | `tl::cp_async_wait<2>();` |
| load.global | 0.10 | 2.1% | 1.0 | 0.19 | `int expert = TileExpert[((int64_t)((int)blockIdx.y))];` |
| load | 0.08 | 1.8% | 0.5 | 0.26 | `for (int s = 0; s < 2; ++s)` |
| wait.sync#2 | 0.06 | 1.4% | 0.5 | 0.19 | `__syncthreads();` |
| load.global#2 | 0.05 | 1.0% | 0.5 | 0.22 | `if (((int)threadIdx.x) < 32) \| if (((int)threadIdx.x) < 64)` |
| wait.sync | 0.04 | 0.9% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#3 | 0.02 | 0.4% | 0.5 | 0.06 | `__syncthreads();` |

### gemm_mxfp8_kernel grid (80, 1, 1) block (128, 1, 1)

16 launches, 320 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 398.42 us, warp 395.44 us, nsys graphs 378.30 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 328.02 | 83.0% | 371.94 | 93.8% |
| unattributed | 36.27 | 9.2% | 0.00 | 0.0% |
| compute | 17.83 | 4.5% | 17.95 | 4.5% |
| load | 13.27 | 3.4% | 17.07 | 4.3% |
| store | 0.05 | 0.0% | 0.06 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.cp_async*118 (est.: sampled iterations x118) | 215.82 | 54.6% | 1.0 | 505.98 | `tl::cp_async_wait<2>();` |
| wait.sync#2*118 (est.: sampled iterations x118) | 103.01 | 26.1% | 1.0 | 326.62 | `__syncthreads();` |
| compute.mma*118 (est.: sampled iterations x118) | 17.58 | 4.4% | 1.0 | 18.88 | `{` |
| load.cp_async#2*118 (est.: sampled iterations x118) | 12.98 | 3.3% | 1.0 | 15.10 | `for (int i_5 = 0; i_5 < 2; ++i_5) \| for (int i_5 = 0; i_5 < 4; ++i_5) \| tl::cp_async_gs_co` |
| wait.sync*118 (est.: sampled iterations x118) | 6.84 | 1.7% | 1.0 | 7.55 | `__syncthreads();` |
| wait.cp_async#2 | 0.89 | 0.2% | 1.0 | 2.98 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| wait.cp_async#3 | 0.58 | 0.1% | 1.0 | 2.19 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>();` |
| wait.sync#3 | 0.46 | 0.1% | 1.0 | 1.86 | `__syncthreads();` |
| wait.sync#4 | 0.42 | 0.1% | 1.0 | 1.70 | `__syncthreads();` |
| load.cp_async | 0.29 | 0.1% | 1.0 | 0.50 | `for (int i = 0; i < 2; ++i) \| for (int i_1 = 0; i_1 < 2; ++i_1) \| for (int i_1 = 0; i_1 < ` |
| compute.mma#3 | 0.13 | 0.0% | 1.0 | 0.13 | `{` |
| compute.mma#2 | 0.12 | 0.0% | 1.0 | 0.13 | `{` |
| store.global | 0.05 | 0.0% | 1.0 | 0.06 | `for (int i_10 = 0; i_10 < 16; ++i_10) \| for (int i_10 = 0; i_10 < 8; ++i_10) \| for (int i_` |

### gemm_mxfp8_kernel grid (64, 1, 1) block (128, 1, 1)

176 launches, 256 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 23.31 us, warp 13.52 us, nsys graphs 23.81 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 5.56 | 41.1% | 5.64 | 46.1% |
| wait | 4.83 | 35.7% | 3.50 | 28.6% |
| compute | 2.03 | 15.0% | 1.82 | 14.9% |
| load | 1.04 | 7.7% | 1.20 | 9.8% |
| store | 0.06 | 0.5% | 0.06 | 0.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*8 (est.: sampled iterations x8) | 1.66 | 12.3% | 1.0 | 1.79 | `{` |
| wait.sync#2*8 (est.: sampled iterations x8) | 1.50 | 11.1% | 1.0 | 5.89 | `__syncthreads();` |
| wait.cp_async*8 (est.: sampled iterations x8) | 1.26 | 9.3% | 1.0 | 6.40 | `tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| load.cp_async#2*8 (est.: sampled iterations x8) | 0.82 | 6.0% | 1.0 | 1.02 | `for (int i_3 = 0; i_3 < 2; ++i_3) \| for (int i_5 = 0; i_5 < 2; ++i_5) \| tl::cp_async_gs_co` |
| wait.cp_async#3 | 0.51 | 3.8% | 1.0 | 1.94 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>();` |
| wait.cp_async#2 | 0.42 | 3.1% | 1.0 | 1.54 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| wait.sync#4 | 0.42 | 3.1% | 1.0 | 1.47 | `__syncthreads();` |
| wait.sync*8 (est.: sampled iterations x8) | 0.37 | 2.7% | 1.0 | 0.51 | `__syncthreads();` |
| wait.sync#3 | 0.35 | 2.6% | 1.0 | 1.22 | `__syncthreads();` |
| load.cp_async | 0.23 | 1.7% | 1.0 | 0.26 | `for (int i = 0; i < 2; ++i) \| for (int i_1 = 0; i_1 < 2; ++i_1) \| for (int i_1 = 0; i_1 < ` |
| compute.mma#3 | 0.19 | 1.4% | 1.0 | 0.19 | `{` |
| compute.mma#2 | 0.18 | 1.3% | 1.0 | 0.19 | `{` |
| store.global | 0.06 | 0.5% | 1.0 | 0.06 | `for (int i_10 = 0; i_10 < 16; ++i_10) \| for (int i_10 = 0; i_10 < 8; ++i_10) \| for (int i_` |
| compute.mma#4 | 0.00 | 0.0% | 0.0 | 0.00 | `{` |
| compute.mma*2 (est.: sampled iterations x2) | 0.00 | 0.0% | 0.0 | 0.00 | `{` |
| load.cp_async#2*2 (est.: sampled iterations x2) | 0.00 | 0.0% | 0.0 | 0.00 | `for (int i_7 = 0; i_7 < 2; ++i_7)` |
| wait.cp_async#4 | 0.00 | 0.0% | 0.0 | 0.00 | `tl::cp_async_wait<0>();` |
| wait.cp_async*2 (est.: sampled iterations x2) | 0.00 | 0.0% | 0.0 | 0.00 | `tl::cp_async_wait<3>();` |
| wait.sync#2*2 (est.: sampled iterations x2) | 0.00 | 0.0% | 0.0 | 0.00 | `__syncthreads();` |
| wait.sync#5 | 0.00 | 0.0% | 0.0 | 0.00 | `__syncthreads();` |
| wait.sync*2 (est.: sampled iterations x2) | 0.00 | 0.0% | 0.0 | 0.00 | `__syncthreads();` |

### gemm_mxfp8_kernel grid (28, 1, 1) block (128, 1, 1)

144 launches, 112 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 35.70 us, warp 34.56 us, nsys graphs 41.55 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 19.08 | 55.2% | 14.18 | 40.3% |
| wait | 8.27 | 23.9% | 13.26 | 37.7% |
| compute | 3.94 | 11.4% | 3.92 | 11.1% |
| load | 3.22 | 9.3% | 3.78 | 10.7% |
| store | 0.05 | 0.2% | 0.05 | 0.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*19 (est.: sampled iterations x19) | 3.76 | 10.9% | 1.0 | 4.26 | `{` |
| wait.cp_async*19 (est.: sampled iterations x19) | 3.50 | 10.1% | 1.0 | 14.90 | `tl::cp_async_wait<1>();` |
| wait.sync#2*19 (est.: sampled iterations x19) | 3.34 | 9.7% | 1.0 | 10.94 | `__syncthreads();` |
| load.cp_async#2*19 (est.: sampled iterations x19) | 2.66 | 7.7% | 1.0 | 3.04 | `for (int i_3 = 0; i_3 < 2; ++i_3) \| for (int i_3 = 0; i_3 < 8; ++i_3)` |
| wait.sync*19 (est.: sampled iterations x19) | 1.18 | 3.4% | 1.0 | 1.22 | `__syncthreads();` |
| load.cp_async | 0.56 | 1.6% | 1.0 | 0.70 | `for (int i = 0; i < 2; ++i) \| for (int i_1 = 0; i_1 < 2; ++i_1) \| for (int i_1 = 0; i_1 < ` |
| compute.mma#2 | 0.18 | 0.5% | 1.0 | 0.19 | `{` |
| wait.cp_async#2 | 0.14 | 0.4% | 1.0 | 0.38 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| wait.sync#3 | 0.12 | 0.3% | 1.0 | 0.34 | `__syncthreads();` |
| store.global | 0.05 | 0.2% | 1.0 | 0.06 | `for (int i_10 = 0; i_10 < 16; ++i_10) \| for (int i_10 = 0; i_10 < 8; ++i_10) \| for (int i_` |

### mhc_project_streams_kernel grid (40, 1, 1) block (128, 1, 1)

288 launches, 160 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 15.63 us, warp 13.92 us, nsys graphs 14.70 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 6.77 | 48.6% | 3.94 | 25.7% |
| compute | 2.78 | 20.0% | 2.57 | 16.8% |
| load | 2.77 | 19.9% | 5.73 | 37.4% |
| wait | 1.14 | 8.2% | 2.55 | 16.6% |
| store | 0.46 | 3.3% | 0.54 | 3.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*4 (est.: sampled iterations x4) | 2.65 | 19.0% | 1.0 | 2.69 | `{` |
| load.global#2*4 (est.: sampled iterations x4) | 2.53 | 18.2% | 1.0 | 5.82 | `for (int i_1 = 0; i_1 < 6; ++i_1)` |
| wait.sync*4 (est.: sampled iterations x4) | 0.95 | 6.8% | 1.0 | 3.33 | `__syncthreads();` |
| store.global*4 (est.: sampled iterations x4) | 0.25 | 1.8% | 1.0 | 0.38 | `if (((((int)blockIdx.y) * 16) + (((int)threadIdx.x) >> 3)) < M)` |
| load.global | 0.24 | 1.7% | 1.0 | 0.26 | `for (int k = 0; k < 4; ++k)` |
| wait.sync#2*4 (est.: sampled iterations x4) | 0.18 | 1.3% | 1.0 | 0.26 | `__syncthreads();` |
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

144 launches, 512 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 18.10 us, warp 16.03 us, nsys graphs 30.22 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 7.14 | 44.5% | 9.00 | 38.3% |
| load | 3.46 | 21.6% | 4.18 | 17.8% |
| unattributed | 2.90 | 18.1% | 7.70 | 32.8% |
| compute | 2.48 | 15.5% | 2.55 | 10.8% |
| store | 0.06 | 0.4% | 0.06 | 0.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*14 (est.: sampled iterations x14) | 2.24 | 14.0% | 1.0 | 2.69 | `{` |
| wait.sync#2*14 (est.: sampled iterations x14) | 1.23 | 7.7% | 1.0 | 1.79 | `__syncthreads();` |
| load.global*14 (est.: sampled iterations x14) | 1.09 | 6.8% | 1.0 | 1.79 | `if (((int)threadIdx.x) < 4)` |
| load.cp_async#2*14 (est.: sampled iterations x14) | 1.01 | 6.3% | 1.0 | 1.34 | `for (int i_4 = 0; i_4 < 2; ++i_4) \| for (int i_6 = 0; i_6 < 2; ++i_6)` |
| load.cp_async#3*14 (est.: sampled iterations x14) | 0.98 | 6.1% | 1.0 | 1.34 | `for (int i_10 = 0; i_10 < 4; ++i_10) \| for (int i_10 = 0; i_10 < 6; ++i_10) \| for (int i_1` |
| wait.sync#3*14 (est.: sampled iterations x14) | 0.95 | 5.9% | 1.0 | 1.34 | `__syncthreads();` |
| wait.sync#4*14 (est.: sampled iterations x14) | 0.95 | 5.9% | 1.0 | 1.34 | `__syncthreads();` |
| wait.sync#5*14 (est.: sampled iterations x14) | 0.94 | 5.9% | 1.0 | 1.34 | `__syncthreads();` |
| wait.sync*14 (est.: sampled iterations x14) | 0.90 | 5.6% | 1.0 | 1.34 | `__syncthreads();` |
| wait.cp_async*14 (est.: sampled iterations x14) | 0.84 | 5.2% | 1.0 | 0.90 | `tl::cp_async_wait<3>();` |
| wait.cp_async#2*14 (est.: sampled iterations x14) | 0.81 | 5.1% | 1.0 | 0.90 | `tl::cp_async_wait<3>();` |
| load.cp_async | 0.26 | 1.6% | 1.0 | 0.48 | `for (int i_1 = 0; i_1 < 2; ++i_1)` |
| compute.mma#2 | 0.13 | 0.8% | 1.0 | 0.16 | `{` |
| compute.mma#3 | 0.11 | 0.7% | 1.0 | 0.13 | `{` |
| wait.sync#7 | 0.06 | 0.4% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#10 | 0.06 | 0.4% | 1.0 | 0.10 | `__syncthreads();` |
| load.global#2 | 0.06 | 0.4% | 1.0 | 0.10 | `if (((int)threadIdx.x) < 4)` |
| store.global | 0.06 | 0.4% | 1.0 | 0.06 | `for (int i_13 = 0; i_13 < 8; ++i_13) \| for (int i_19 = 0; i_19 < 16; ++i_19) \| for (int i_` |
| load.global#3 | 0.06 | 0.4% | 1.0 | 0.10 | `if (((int)threadIdx.x) < 4)` |
| wait.cp_async#6 | 0.06 | 0.3% | 1.0 | 0.06 | `tl::cp_async_wait<0>();` |
| wait.sync#8 | 0.05 | 0.3% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#9 | 0.05 | 0.3% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#11 | 0.05 | 0.3% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#6 | 0.05 | 0.3% | 1.0 | 0.06 | `__syncthreads();` |
| wait.cp_async#3 | 0.05 | 0.3% | 1.0 | 0.06 | `tl::cp_async_wait<3>();` |
| wait.cp_async#5 | 0.05 | 0.3% | 1.0 | 0.06 | `tl::cp_async_wait<1>();` |
| wait.cp_async#4 | 0.04 | 0.3% | 1.0 | 0.06 | `tl::cp_async_wait<2>();` |

### gemm_mxfp8_kernel grid (36, 1, 1) block (128, 1, 1)

144 launches, 144 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 19.47 us, warp 18.52 us, nsys graphs 28.85 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 8.77 | 47.3% | 6.01 | 28.9% |
| wait | 3.82 | 20.6% | 7.43 | 35.8% |
| compute | 3.53 | 19.1% | 3.53 | 17.0% |
| load | 2.35 | 12.7% | 3.76 | 18.1% |
| store | 0.05 | 0.3% | 0.05 | 0.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*18 (est.: sampled iterations x18) | 3.24 | 17.5% | 1.0 | 3.46 | `{` |
| load.cp_async#2*18 (est.: sampled iterations x18) | 2.16 | 11.7% | 1.0 | 2.30 | `for (int i_4 = 0; i_4 < 2; ++i_4) \| for (int i_5 = 0; i_5 < 4; ++i_5)` |
| wait.sync#2*18 (est.: sampled iterations x18) | 1.39 | 7.5% | 1.0 | 1.73 | `__syncthreads();` |
| wait.sync*18 (est.: sampled iterations x18) | 1.12 | 6.0% | 1.0 | 1.15 | `__syncthreads();` |
| wait.cp_async*18 (est.: sampled iterations x18) | 1.10 | 5.9% | 1.0 | 1.15 | `tl::cp_async_wait<2>();` |
| load.cp_async | 0.19 | 1.0% | 1.0 | 0.22 | `for (int i = 0; i < 2; ++i) \| for (int i_1 = 0; i_1 < 2; ++i_1) \| for (int i_1 = 0; i_1 < ` |
| compute.mma#2 | 0.15 | 0.8% | 1.0 | 0.16 | `{` |
| compute.mma#3 | 0.14 | 0.8% | 1.0 | 0.16 | `{` |
| wait.cp_async#3 | 0.06 | 0.3% | 1.0 | 0.10 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>();` |
| wait.sync#4 | 0.06 | 0.3% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#3 | 0.06 | 0.3% | 1.0 | 0.06 | `__syncthreads();` |
| store.global | 0.05 | 0.3% | 1.0 | 0.06 | `for (int i_10 = 0; i_10 < 16; ++i_10) \| for (int i_10 = 0; i_10 < 8; ++i_10) \| for (int i_` |
| wait.cp_async#2 | 0.05 | 0.3% | 1.0 | 0.06 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |

### gemm_mxfp8_kernel grid (40, 1, 1) block (128, 1, 1)

144 launches, 160 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 17.89 us, warp 15.35 us, nsys graphs 19.15 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 6.11 | 39.8% | 9.52 | 51.2% |
| unattributed | 3.79 | 24.7% | 2.53 | 13.6% |
| compute | 3.23 | 21.1% | 3.20 | 17.2% |
| load | 2.15 | 14.0% | 3.27 | 17.6% |
| store | 0.06 | 0.4% | 0.06 | 0.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*14 (est.: sampled iterations x14) | 2.88 | 18.8% | 1.0 | 3.14 | `{` |
| wait.sync#2*14 (est.: sampled iterations x14) | 2.21 | 14.4% | 1.0 | 7.17 | `__syncthreads();` |
| wait.cp_async*14 (est.: sampled iterations x14) | 2.20 | 14.3% | 1.0 | 9.18 | `tl::cp_async_wait<2>();` |
| load.cp_async#2*14 (est.: sampled iterations x14) | 1.90 | 12.4% | 1.0 | 2.24 | `for (int i_5 = 0; i_5 < 2; ++i_5) \| tl::cp_async_gs_conditional<16>((&(((fp8_e4_t*)A_share` |
| wait.sync*14 (est.: sampled iterations x14) | 0.90 | 5.8% | 1.0 | 0.90 | `__syncthreads();` |
| wait.sync#3 | 0.25 | 1.6% | 1.0 | 0.83 | `__syncthreads();` |
| load.cp_async | 0.25 | 1.6% | 1.0 | 0.29 | `for (int i = 0; i < 2; ++i) \| for (int i_1 = 0; i_1 < 2; ++i_1) \| for (int i_1 = 0; i_1 < ` |
| wait.cp_async#2 | 0.24 | 1.6% | 1.0 | 1.01 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| compute.mma#3 | 0.18 | 1.2% | 1.0 | 0.19 | `{` |
| compute.mma#2 | 0.17 | 1.1% | 1.0 | 0.19 | `{` |
| wait.cp_async#3 | 0.16 | 1.1% | 1.0 | 0.78 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>();` |
| wait.sync#4 | 0.15 | 1.0% | 1.0 | 0.61 | `__syncthreads();` |
| store.global | 0.06 | 0.4% | 1.0 | 0.10 | `for (int i_10 = 0; i_10 < 16; ++i_10) \| for (int i_10 = 0; i_10 < 8; ++i_10) \| for (int i_` |

### gemm_bf16_partials_kernel grid (6, 8, 1) block (256, 1, 1)

96 launches, 384 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 24.77 us, warp 5.75 us, nsys graphs 28.00 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 3.32 | 57.8% | 1.32 | 30.9% |
| wait | 1.31 | 22.8% | 1.50 | 35.3% |
| compute | 0.69 | 12.0% | 1.36 | 31.9% |
| load | 0.39 | 6.8% | 0.00 | 0.0% |
| store | 0.04 | 0.7% | 0.08 | 1.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*10 (est.: sampled iterations x10) | 0.69 | 12.0% | 0.5 | 1.60 | `{` |
| wait.mbarrier*10 (est.: sampled iterations x10) | 0.66 | 11.4% | 0.5 | 1.60 | `mbarrier[((kb % 3) + 3)].wait((((kb % 6) / 3) ^ 1));` |
| wait.mbarrier#2*10 (est.: sampled iterations x10) | 0.56 | 9.7% | 0.5 | 2.40 | `mbarrier[(kb_1 % 3)].wait(((kb_1 % 6) / 3));` |
| load.tma*10 (est.: sampled iterations x10) | 0.39 | 6.8% | 0.5 | 1.28 | `if (tl::tl_shuffle_elect<128>())` |
| wait.sync | 0.10 | 1.7% | 1.0 | 0.13 | `__syncthreads();` |
| store.global | 0.04 | 0.7% | 0.5 | 0.10 | `for (int i_1 = 0; i_1 < 4; ++i_1) \| for (int i_2 = 0; i_2 < 8; ++i_2) \| for (int i_4 = 0; ` |

### mhc_finalize_kernel grid (5, 1, 1) block (128, 1, 1)

180 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 13.90 us, warp 12.63 us, nsys graphs 10.59 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 8.43 | 66.8% | 7.34 | 54.4% |
| store | 1.51 | 12.0% | 2.38 | 17.7% |
| load | 1.03 | 8.2% | 1.68 | 12.4% |
| compute | 1.00 | 7.9% | 1.47 | 10.9% |
| wait | 0.66 | 5.2% | 0.61 | 4.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync | 0.66 | 5.2% | 1.0 | 0.90 | `__syncthreads();` |
| store.global | 0.61 | 4.9% | 0.2 | 2.46 | `if (((int)threadIdx.x) < 4)` |
| compute.reduce | 0.57 | 4.5% | 0.2 | 2.30 | `row_max[0] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<32>>::run(row_max[0]);` |
| store.global#4*7 (est.: sampled iterations x7) | 0.56 | 4.4% | 0.8 | 0.90 | `if ((((int)threadIdx.x) % 4) == 0)` |
| compute.reduce#2*7 (est.: sampled iterations x7) | 0.43 | 3.4% | 0.8 | 0.67 | `amax[0] = max(amax[0], __shfl_xor_sync((uint)4294967295, amax[0], 1, 32));` |
| load.global#4*7 (est.: sampled iterations x7) | 0.37 | 2.9% | 0.8 | 0.67 | `*(uint4*)(yv + 0) = *(uint4*)(C + ((((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((i` |
| load.global | 0.35 | 2.7% | 1.0 | 1.04 | `if (((int)threadIdx.x) < 32)` |
| store.global#3*7 (est.: sampled iterations x7) | 0.32 | 2.6% | 0.8 | 0.45 | `*(uint4*)(y + ((((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((int64_t)step) * (int6` |
| load.global#3 | 0.27 | 2.1% | 0.8 | 0.38 | `for (int i_split_2 = 0; i_split_2 < 40; ++i_split_2)` |
| load.global#2 | 0.04 | 0.4% | 0.2 | 0.19 | `cm[0] = ((mixes_shared[((((int)threadIdx.x) & 15) + 8)] * scale[2]) + base[((((int)threadI` |
| store.global#2 | 0.01 | 0.1% | 0.2 | 0.06 | `if ((((int)threadIdx.x) >> 4) == 0)` |

### gemm_bf16_splitk_kernel grid (16, 5, 1) block (128, 1, 1)

32 launches, 320 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 57.57 us, warp 52.44 us, nsys graphs 53.97 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 25.78 | 49.2% | 0.00 | 0.0% |
| wait | 22.36 | 42.6% | 48.37 | 90.0% |
| load | 2.22 | 4.2% | 3.63 | 6.8% |
| compute | 1.96 | 3.7% | 2.13 | 4.0% |
| store | 0.13 | 0.2% | 0.15 | 0.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.cp_async*13 (est.: sampled iterations x13) | 6.64 | 12.7% | 0.5 | 20.38 | `tl::cp_async_wait<2>();` |
| wait.cp_async*12 (est.: sampled iterations x12) | 2.64 | 5.0% | 0.5 | 8.26 | `tl::cp_async_wait<3>();` |
| wait.sync*13 (est.: sampled iterations x13) | 2.47 | 4.7% | 0.5 | 5.82 | `__syncthreads();` |
| wait.sync*12 (est.: sampled iterations x12) | 2.33 | 4.4% | 0.5 | 5.95 | `__syncthreads();` |
| wait.cp_async#2 | 2.29 | 4.4% | 1.0 | 4.51 | `tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>(); \| tl::cp_async_wait<3>();` |
| wait.cp_async#4 | 1.65 | 3.1% | 1.0 | 3.98 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>();` |
| load.cp_async | 1.04 | 2.0% | 1.0 | 3.23 | `for (int i_1 = 0; i_1 < 2; ++i_1) \| tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)A_sha` |
| wait.cp_async#3 | 0.99 | 1.9% | 1.0 | 3.23 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| wait.sync#5 | 0.82 | 1.6% | 1.0 | 2.43 | `__syncthreads();` |
| compute.mma*12 (est.: sampled iterations x12) | 0.82 | 1.6% | 0.5 | 0.96 | `{` |
| compute.mma*13 (est.: sampled iterations x13) | 0.79 | 1.5% | 0.5 | 0.83 | `{` |
| wait.sync#3 | 0.79 | 1.5% | 1.0 | 2.32 | `__syncthreads();` |
| wait.sync#4 | 0.55 | 1.0% | 1.0 | 1.86 | `__syncthreads();` |
| load.cp_async#2*12 (est.: sampled iterations x12) | 0.52 | 1.0% | 0.5 | 0.58 | `tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)A_shared)[((((((kb & 3) * 1024) + ((((int` |
| load.cp_async#2*13 (est.: sampled iterations x13) | 0.42 | 0.8% | 0.5 | 0.42 | `for (int i_10 = 0; i_10 < 2; ++i_10) \| for (int i_8 = 0; i_8 < 2; ++i_8) \| tl::cp_async_gs` |
| wait.sync#2*13 (est.: sampled iterations x13) | 0.35 | 0.7% | 0.5 | 0.42 | `__syncthreads();` |
| wait.sync#2*12 (est.: sampled iterations x12) | 0.30 | 0.6% | 0.5 | 0.38 | `__syncthreads();` |
| load.global | 0.25 | 0.5% | 1.0 | 1.38 | `if (((int)threadIdx.x) == 0)` |
| wait.sync#7 | 0.21 | 0.4% | 1.0 | 0.37 | `__syncthreads();` |
| wait.sync#8 | 0.16 | 0.3% | 0.5 | 0.22 | `__syncthreads();` |
| wait.sync#6 | 0.11 | 0.2% | 1.0 | 0.37 | `__syncthreads();` |
| compute.mma#2 | 0.10 | 0.2% | 1.0 | 0.13 | `{` |
| compute.mma#3 | 0.10 | 0.2% | 1.0 | 0.10 | `{` |
| compute.mma#4 | 0.10 | 0.2% | 1.0 | 0.13 | `{` |
| wait.cp_async#5 | 0.07 | 0.1% | 0.5 | 0.30 | `tl::cp_async_wait<0>();` |
| store.global#3 | 0.07 | 0.1% | 1.0 | 0.10 | `if (((int*)last)[0] == 1)` |
| store.global | 0.06 | 0.1% | 1.0 | 0.06 | `for (int i_10 = 0; i_10 < 4; ++i_10) \| for (int i_16 = 0; i_16 < 4; ++i_16) \| for (int i_1` |
| compute.mma#5 | 0.05 | 0.1% | 0.5 | 0.06 | `{` |

### sparse_mla_kernel grid (5, 1, 5) block (128, 1, 1)

60 launches, 100 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 15.86 us, warp 15.17 us, nsys graphs 11.89 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed, with loops' unsampled iterations | 9.95 | 65.6% | 10.78 | 49.4% |
| wait | 2.28 | 15.0% | 0.64 | 2.9% |
| load | 1.54 | 10.1% | 8.93 | 40.9% |
| compute | 0.96 | 6.3% | 0.96 | 4.4% |
| store | 0.44 | 2.9% | 0.52 | 2.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync (sampled iterations) | 2.21 | 14.6% | 11.0 | 3.12 | `__syncthreads();` |
| load.global (sampled iterations) | 1.21 | 8.0% | 2.0 | 3.47 | `for (int c = 0; c < 4; ++c) \| for (int i = 0; i < 8; ++i) \| for (int i_1 = 0; i_1 < 8; ++i` |
| compute.mma (sampled iterations) | 0.74 | 4.9% | 2.0 | 0.77 | `for (int ki = 0; ki < 32; ++ki) \| for (int ki_1 = 0; ki_1 < 2; ++ki_1)` |
| store.global | 0.44 | 2.9% | 1.0 | 0.54 | `for (int i_10 = 0; i_10 < 32; ++i_10) \| for (int i_11 = 0; i_11 < 32; ++i_11) \| for (int i` |
| load.cp_async (sampled iterations) | 0.33 | 2.2% | 2.0 | 0.42 | `for (int k_1 = 0; k_1 < 5; ++k_1) \| for (int k_1 = 0; k_1 < 9; ++k_1) \| for (int k_3 = 0; ` |
| compute.reduce (sampled iterations) | 0.22 | 1.5% | 2.0 | 0.26 | `m_g[i_5] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(m_g[i_5], (&((` |
| wait.cp_async (sampled iterations) | 0.07 | 0.5% | 1.0 | 0.10 | `tl::cp_async_wait<0>();` |

### mhc_finalize_kernel grid (10, 1, 1) block (128, 1, 1)

96 launches, 40 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 13.04 us, warp 11.81 us, nsys graphs 9.47 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 7.69 | 65.2% | 6.80 | 55.3% |
| store | 1.40 | 11.8% | 1.99 | 16.2% |
| load | 1.02 | 8.6% | 1.61 | 13.1% |
| compute | 0.99 | 8.4% | 1.28 | 10.4% |
| wait | 0.71 | 6.0% | 0.61 | 5.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync | 0.71 | 6.0% | 1.0 | 1.02 | `__syncthreads();` |
| compute.reduce | 0.57 | 4.9% | 0.2 | 2.30 | `row_max[0] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<32>>::run(row_max[0]);` |
| store.global#4*7 (est.: sampled iterations x7) | 0.56 | 4.7% | 0.8 | 0.90 | `if ((((int)threadIdx.x) % 4) == 0)` |
| store.global | 0.50 | 4.2% | 0.2 | 2.03 | `if (((int)threadIdx.x) < 4)` |
| compute.reduce#2*7 (est.: sampled iterations x7) | 0.41 | 3.5% | 0.8 | 0.67 | `amax[0] = max(amax[0], __shfl_xor_sync((uint)4294967295, amax[0], 1, 32));` |
| load.global | 0.36 | 3.1% | 1.0 | 1.15 | `if (((int)threadIdx.x) < 32)` |
| load.global#4*7 (est.: sampled iterations x7) | 0.36 | 3.0% | 0.8 | 0.67 | `*(uint4*)(yv + 0) = *(uint4*)(C + ((((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((i` |
| store.global#3*7 (est.: sampled iterations x7) | 0.32 | 2.8% | 0.8 | 0.45 | `*(uint4*)(y + ((((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((int64_t)step) * (int6` |
| load.global#3 | 0.25 | 2.1% | 0.8 | 0.35 | `for (int i_split_2 = 0; i_split_2 < 40; ++i_split_2)` |
| load.global#2 | 0.05 | 0.4% | 0.2 | 0.19 | `cm[0] = ((mixes_shared[((((int)threadIdx.x) & 15) + 8)] * scale[2]) + base[((((int)threadI` |
| store.global#2 | 0.01 | 0.1% | 0.2 | 0.06 | `if ((((int)threadIdx.x) >> 4) == 0)` |

### gemm_mxfp8_kernel grid (40, 1, 1) block (64, 1, 1)

144 launches, 80 warps; ranges, warp and last warp from the 4 recording CTAs' 8 warps; span 7.28 us, warp 6.62 us, nsys graphs 8.77 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 2.19 | 33.0% | 1.18 | 15.4% |
| wait | 2.01 | 30.3% | 3.91 | 51.2% |
| compute | 1.50 | 22.7% | 1.53 | 20.0% |
| load | 0.84 | 12.6% | 0.93 | 12.2% |
| store | 0.09 | 1.3% | 0.09 | 1.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*8 (est.: sampled iterations x8) | 1.34 | 20.3% | 1.0 | 1.54 | `{` |
| wait.cp_async*8 (est.: sampled iterations x8) | 0.83 | 12.6% | 1.0 | 2.05 | `tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| load.cp_async#2*8 (est.: sampled iterations x8) | 0.70 | 10.6% | 1.0 | 0.77 | `for (int i_3 = 0; i_3 < 2; ++i_3) \| for (int i_5 = 0; i_5 < 2; ++i_5) \| tl::cp_async_gs_co` |
| wait.sync#2*8 (est.: sampled iterations x8) | 0.51 | 7.7% | 1.0 | 1.41 | `__syncthreads();` |
| wait.sync*8 (est.: sampled iterations x8) | 0.38 | 5.8% | 1.0 | 0.51 | `__syncthreads();` |
| wait.cp_async#2 | 0.21 | 3.1% | 1.0 | 0.42 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| compute.mma#2 | 0.16 | 2.4% | 1.0 | 0.16 | `{` |
| load.cp_async | 0.13 | 2.0% | 1.0 | 0.16 | `for (int i = 0; i < 2; ++i) \| for (int i_1 = 0; i_1 < 2; ++i_1) \| for (int i_1 = 0; i_1 < ` |
| store.global | 0.09 | 1.3% | 1.0 | 0.10 | `for (int i_10 = 0; i_10 < 16; ++i_10) \| for (int i_10 = 0; i_10 < 8; ++i_10) \| for (int i_` |
| wait.sync#3 | 0.07 | 1.1% | 1.0 | 0.19 | `__syncthreads();` |

### sparse_mla_kernel grid (2, 1, 10) block (128, 1, 1)

48 launches, 80 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 21.71 us, warp 16.05 us, nsys graphs 18.08 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed, with loops' unsampled iterations | 9.03 | 56.3% | 7.51 | 36.0% |
| compute | 2.50 | 15.6% | 4.72 | 22.6% |
| load | 2.10 | 13.1% | 4.77 | 22.9% |
| wait | 1.98 | 12.4% | 3.41 | 16.3% |
| store | 0.45 | 2.8% | 0.48 | 2.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 | 1.14 | 7.1% | 1.0 | 3.52 | `for (int i = 0; i < 8; ++i)` |
| compute.mma*4 (est.: sampled iterations x4) | 0.86 | 5.4% | 0.5 | 1.79 | `for (int ki = 0; ki < 32; ++ki) \| {` |
| compute.mma#2*4 (est.: sampled iterations x4) | 0.74 | 4.6% | 1.0 | 1.28 | `for (int ki_1 = 0; ki_1 < 2; ++ki_1) \| if (((((int)blockIdx.x) * 2) + (c_1 >> 1)) < 3)` |
| load.cp_async#2*4 (est.: sampled iterations x4) | 0.50 | 3.1% | 1.0 | 0.77 | `if (((((int)blockIdx.x) * 2) + (c_1 >> 1)) < 3) \| if ((c_1 < 3) & (c_1 < 3)) \| if (c < 3)` |
| compute.reduce#2*8 (est.: sampled iterations x8) | 0.45 | 2.8% | 0.5 | 1.02 | `row_sum[i_7] = tl::AllReduce<tl::SumOp, 128, 32, 0, tl::NamedBarrier<128>>::run(row_sum[i_` |
| compute.reduce*8 (est.: sampled iterations x8) | 0.45 | 2.8% | 0.5 | 1.02 | `m_g[i_4] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(m_g[i_4], (&((` |
| store.global | 0.45 | 2.8% | 1.0 | 0.50 | `for (int i_10 = 0; i_10 < 32; ++i_10) \| for (int i_11 = 0; i_11 < 32; ++i_11) \| for (int i` |
| load.cp_async (sampled iterations) | 0.33 | 2.0% | 1.0 | 0.35 | `for (int k_1 = 0; k_1 < 5; ++k_1) \| for (int k_1 = 0; k_1 < 9; ++k_1) \| for (int k_3 = 0; ` |
| wait.sync#9*4 (est.: sampled iterations x4) | 0.26 | 1.6% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#7*4 (est.: sampled iterations x4) | 0.25 | 1.6% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#11*4 (est.: sampled iterations x4) | 0.24 | 1.5% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#6*4 (est.: sampled iterations x4) | 0.24 | 1.5% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync (sampled iterations) | 0.18 | 1.1% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#10*8 (est.: sampled iterations x8) | 0.18 | 1.1% | 0.5 | 0.51 | `__syncthreads();` |
| wait.sync#8*8 (est.: sampled iterations x8) | 0.14 | 0.9% | 0.5 | 0.51 | `__syncthreads();` |
| wait.cp_async*4 (est.: sampled iterations x4) | 0.14 | 0.8% | 0.5 | 0.38 | `tl::cp_async_wait<0>();` |
| load.global (sampled iterations) | 0.13 | 0.8% | 1.0 | 0.29 | `for (int c = 0; c < 4; ++c) \| for (int i = 0; i < 8; ++i) \| for (int i_1 = 0; i_1 < 8; ++i` |
| wait.sync#4*4 (est.: sampled iterations x4) | 0.13 | 0.8% | 0.5 | 0.26 | `__syncthreads();` |
| wait.sync#5*4 (est.: sampled iterations x4) | 0.13 | 0.8% | 0.5 | 0.26 | `__syncthreads();` |
| wait.sync#3 | 0.06 | 0.4% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#2 | 0.04 | 0.2% | 1.0 | 0.06 | `__syncthreads();` |

### metadata_chunk_kernel grid (10, 1, 1) block (192, 1, 1)

48 launches, 60 warps; ranges, warp and last warp from the 5 recording CTAs' 30 warps; span 12.45 us, warp 7.68 us, nsys graphs 10.66 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 5.00 | 65.1% | 13.09 | 83.4% |
| unattributed | 2.61 | 34.0% | 2.47 | 15.7% |
| store | 0.07 | 0.9% | 0.13 | 0.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 4.83 | 62.9% | 1.0 | 8.67 | `if ((((offset + ((int)blockIdx.x)) < src_rows) && (0 <= (offset + ((int)blockIdx.x)))))` |
| load.global#2 | 0.17 | 2.2% | 0.2 | 1.68 | `if (((0 <= req) & ((int64_t)0 <= pos)))` |
| store.global | 0.05 | 0.7% | 1.0 | 0.06 | `swa[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = con` |
| store.global#3 | 0.01 | 0.1% | 0.2 | 0.06 | `top_lengths[((int64_t)((int)blockIdx.x))] = condval_12; \| top_lengths[((int64_t)((int)bloc` |
| store.global#2 | 0.01 | 0.1% | 0.2 | 0.06 | `lengths[((int64_t)((int)blockIdx.x))] = ((int)max(info[1], (int64_t)0));` |

### gemm_bf16_partials_kernel grid (2, 20, 1) block (256, 1, 1)

48 launches, 320 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 14.38 us, warp 3.76 us, nsys graphs 16.37 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 2.50 | 66.5% | 1.23 | 55.8% |
| wait | 0.60 | 16.0% | 0.47 | 21.2% |
| load | 0.39 | 10.4% | 0.00 | 0.0% |
| compute | 0.22 | 5.9% | 0.43 | 19.6% |
| store | 0.04 | 1.2% | 0.08 | 3.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.tma*4 (est.: sampled iterations x4) | 0.39 | 10.4% | 0.5 | 1.34 | `if (tl::tl_shuffle_elect<128>())` |
| wait.mbarrier#2*4 (est.: sampled iterations x4) | 0.37 | 9.8% | 0.5 | 1.54 | `mbarrier[(kb_1 % 3)].wait((kb_1 / 3));` |
| compute.mma*4 (est.: sampled iterations x4) | 0.22 | 5.9% | 0.5 | 0.51 | `{` |
| wait.mbarrier*4 (est.: sampled iterations x4) | 0.12 | 3.3% | 0.5 | 0.26 | `mbarrier[((kb % 3) + 3)].wait(((kb / 3) ^ 1));` |
| wait.sync | 0.11 | 2.9% | 1.0 | 0.18 | `__syncthreads();` |
| store.global | 0.04 | 1.2% | 0.5 | 0.13 | `for (int i_1 = 0; i_1 < 4; ++i_1) \| for (int i_2 = 0; i_2 < 8; ++i_2) \| for (int i_4 = 0; ` |

### moe_dispatch_kernel grid (1, 1, 1) block (512, 1, 1)

144 launches, 16 warps; ranges, warp and last warp from the 1 recording CTAs' 16 warps; span 4.48 us, warp 4.26 us, nsys graphs 3.07 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 2.59 | 60.7% | 1.71 | 37.6% |
| unattributed, with loops' unsampled iterations | 1.04 | 24.4% | 1.56 | 34.3% |
| store | 0.55 | 13.0% | 1.17 | 25.8% |
| load | 0.08 | 1.9% | 0.10 | 2.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync#7*9 (est.: sampled iterations x9) | 0.67 | 15.6% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#6*9 (est.: sampled iterations x9) | 0.65 | 15.2% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#8 | 0.52 | 12.1% | 1.0 | 0.77 | `__syncthreads();` |
| store.global (sampled iterations) | 0.48 | 11.2% | 1.0 | 0.93 | `ChunkCount[((((int64_t)c) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = count; \| Chu` |
| wait.sync#2 | 0.38 | 8.8% | 1.0 | 0.42 | `__syncthreads();` |
| wait.sync#3 | 0.13 | 3.1% | 1.0 | 0.32 | `__syncthreads();` |
| load.global (sampled iterations) | 0.08 | 1.9% | 1.0 | 0.10 | `expert = ((int)TopkIdx[((((int64_t)((int)blockIdx.x)) * (int64_t)512) + ((int64_t)((int)th` |
| store.global#2 | 0.08 | 1.8% | 1.0 | 0.29 | `RoutePos[((((int64_t)((int)blockIdx.x)) * (int64_t)512) + ((int64_t)((int)threadIdx.x)))] ` |
| wait.sync | 0.07 | 1.6% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#9 | 0.06 | 1.5% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#4 | 0.06 | 1.4% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#5 | 0.06 | 1.4% | 1.0 | 0.06 | `__syncthreads();` |

### sparse_mla_kernel grid (1, 1, 5) block (128, 1, 1)

30 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 21.49 us, warp 21.43 us, nsys graphs 18.58 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed, with loops' unsampled iterations | 7.72 | 36.0% | 6.26 | 29.2% |
| load | 4.72 | 22.0% | 6.12 | 28.6% |
| compute | 4.72 | 22.0% | 4.69 | 21.9% |
| wait | 3.62 | 16.9% | 3.73 | 17.4% |
| store | 0.65 | 3.0% | 0.63 | 2.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 | 2.78 | 13.0% | 1.0 | 3.10 | `for (int i = 0; i < 8; ++i)` |
| compute.mma*4 (est.: sampled iterations x4) | 1.72 | 8.0% | 1.0 | 1.79 | `for (int ki = 0; ki < 32; ++ki) \| {` |
| compute.mma#2*4 (est.: sampled iterations x4) | 1.16 | 5.4% | 1.0 | 1.22 | `for (int ki_1 = 0; ki_1 < 2; ++ki_1) \| if (((((int)blockIdx.x) * 2) + (c_1 >> 1)) < 3)` |
| wait.sync (sampled iterations) | 1.13 | 5.3% | 1.0 | 1.50 | `__syncthreads();` |
| compute.reduce#2*8 (est.: sampled iterations x8) | 0.92 | 4.3% | 1.0 | 1.02 | `row_sum[i_7] = tl::AllReduce<tl::SumOp, 128, 32, 0, tl::NamedBarrier<128>>::run(row_sum[i_` |
| compute.reduce*8 (est.: sampled iterations x8) | 0.92 | 4.3% | 1.0 | 1.02 | `m_g[i_4] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(m_g[i_4], (&((` |
| load.global (sampled iterations) | 0.92 | 4.3% | 1.0 | 2.11 | `for (int c = 0; c < 4; ++c) \| for (int i = 0; i < 8; ++i) \| for (int i_1 = 0; i_1 < 8; ++i` |
| load.cp_async#2*4 (est.: sampled iterations x4) | 0.66 | 3.1% | 1.0 | 0.77 | `if (((((int)blockIdx.x) * 2) + (c_1 >> 1)) < 3) \| if ((c_1 < 3) & (c_1 < 3)) \| if (c < 3)` |
| store.global | 0.65 | 3.0% | 1.0 | 0.70 | `for (int i_10 = 0; i_10 < 32; ++i_10) \| for (int i_11 = 0; i_11 < 32; ++i_11) \| for (int i` |
| load.cp_async (sampled iterations) | 0.36 | 1.7% | 1.0 | 0.38 | `for (int k_1 = 0; k_1 < 5; ++k_1) \| for (int k_1 = 0; k_1 < 9; ++k_1) \| for (int k_3 = 0; ` |
| wait.sync#8*8 (est.: sampled iterations x8) | 0.31 | 1.5% | 1.0 | 0.51 | `__syncthreads();` |
| wait.sync#10*8 (est.: sampled iterations x8) | 0.30 | 1.4% | 1.0 | 0.51 | `__syncthreads();` |
| wait.cp_async*4 (est.: sampled iterations x4) | 0.26 | 1.2% | 1.0 | 0.38 | `tl::cp_async_wait<0>();` |
| wait.sync#11*4 (est.: sampled iterations x4) | 0.26 | 1.2% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#4*4 (est.: sampled iterations x4) | 0.26 | 1.2% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#5*4 (est.: sampled iterations x4) | 0.26 | 1.2% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#7*4 (est.: sampled iterations x4) | 0.26 | 1.2% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#9*4 (est.: sampled iterations x4) | 0.26 | 1.2% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#6*4 (est.: sampled iterations x4) | 0.24 | 1.1% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#3 | 0.06 | 0.3% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#2 | 0.04 | 0.2% | 1.0 | 0.06 | `__syncthreads();` |

### moe_grouped_fp8_fp4_gemm_kernel grid (9, 36, 1) block (128, 1, 1)

6 launches, 1296 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 102.93 us, warp 17.38 us, nsys graphs 2.82 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 6.89 | 39.7% | 0.20 | 64.9% |
| wait | 5.13 | 29.5% | 0.05 | 15.8% |
| compute | 3.52 | 20.3% | 0.00 | 0.0% |
| load | 1.77 | 10.2% | 0.06 | 19.3% |
| store | 0.07 | 0.4% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*40 (est.: sampled iterations x40) | 3.52 | 20.3% | 0.2 | 14.08 | `{` |
| wait.mbarrier*40 (est.: sampled iterations x40) | 2.08 | 12.0% | 0.2 | 8.32 | `loaded[(ko % 3)].wait(((ko % 6) / 3));` |
| load#2*40 (est.: sampled iterations x40) | 1.48 | 8.5% | 0.2 | 7.68 | `if (ko < 38)` |
| wait.sync#5*40 (est.: sampled iterations x40) | 1.12 | 6.4% | 0.2 | 5.12 | `__syncthreads();` |
| wait.sync#4*40 (est.: sampled iterations x40) | 0.64 | 3.7% | 0.2 | 2.56 | `__syncthreads();` |
| wait.sync#6*40 (est.: sampled iterations x40) | 0.64 | 3.7% | 0.2 | 2.56 | `__syncthreads();` |
| wait.cp_async*40 (est.: sampled iterations x40) | 0.56 | 3.2% | 0.2 | 2.56 | `tl::cp_async_wait<2>();` |
| load.global#3 | 0.11 | 0.6% | 0.2 | 0.45 | `if (((expert < 128) && (0 <= expert))) \| if (((expert < 384) && (0 <= expert)))` |
| load.global | 0.11 | 0.6% | 1.0 | 0.27 | `int expert = TileExpert[((int64_t)((int)blockIdx.y))];` |
| store.global | 0.07 | 0.4% | 0.2 | 0.35 | `for (int i_2 = 0; i_2 < 16; ++i_2) \| for (int i_3 = 0; i_3 < 16; ++i_3) \| for (int i_4 = 0` |
| wait.sync | 0.04 | 0.3% | 1.0 | 0.06 | `__syncthreads();` |
| load | 0.04 | 0.3% | 0.2 | 0.22 | `for (int s = 0; s < 2; ++s)` |
| wait.sync#2 | 0.03 | 0.2% | 0.2 | 0.16 | `__syncthreads();` |
| load.global#2 | 0.02 | 0.1% | 0.2 | 0.19 | `if (((int)threadIdx.x) < 32) \| if (((int)threadIdx.x) < 64)` |
| wait.sync#3 | 0.01 | 0.0% | 0.2 | 0.03 | `__syncthreads();` |

### gemm_mxfp8_partials_kernel grid (16, 5, 1) block (128, 1, 1)

48 launches, 320 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 11.74 us, warp 11.09 us, nsys graphs 12.61 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 5.42 | 48.9% | 5.93 | 49.9% |
| wait | 4.55 | 41.0% | 3.60 | 30.3% |
| compute | 0.61 | 5.5% | 0.63 | 5.3% |
| load | 0.46 | 4.1% | 1.67 | 14.1% |
| store | 0.05 | 0.5% | 0.06 | 0.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.cp_async#3 | 1.73 | 15.6% | 1.0 | 3.38 | `tl::cp_async_wait<0>();` |
| wait.cp_async#2 | 1.19 | 10.7% | 1.0 | 2.18 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>();` |
| wait.sync#3 | 0.48 | 4.3% | 1.0 | 1.42 | `__syncthreads();` |
| wait.sync#2*2 (est.: sampled iterations x2) | 0.48 | 4.3% | 1.0 | 1.73 | `__syncthreads();` |
| wait.sync#4 | 0.33 | 2.9% | 1.0 | 0.90 | `__syncthreads();` |
| compute.mma*2 (est.: sampled iterations x2) | 0.31 | 2.8% | 1.0 | 0.32 | `{` |
| wait.cp_async*2 (est.: sampled iterations x2) | 0.25 | 2.3% | 1.0 | 1.70 | `tl::cp_async_wait<2>();` |
| load.cp_async | 0.23 | 2.1% | 1.0 | 0.32 | `for (int i = 0; i < 2; ++i) \| for (int i_1 = 0; i_1 < 4; ++i_1)` |
| load.cp_async#2*2 (est.: sampled iterations x2) | 0.22 | 2.0% | 1.0 | 0.26 | `for (int i_4 = 0; i_4 < 2; ++i_4)` |
| compute.mma#3 | 0.15 | 1.4% | 1.0 | 0.16 | `{` |
| compute.mma#2 | 0.15 | 1.3% | 1.0 | 0.16 | `{` |
| wait.sync*2 (est.: sampled iterations x2) | 0.09 | 0.8% | 1.0 | 0.13 | `__syncthreads();` |
| store.global | 0.05 | 0.5% | 1.0 | 0.06 | `for (int i_11 = 0; i_11 < 16; ++i_11) \| for (int i_13 = 0; i_13 < 8; ++i_13) \| for (int i_` |

### partial_argmax_kernel grid (2, 64, 1) block (128, 1, 1)

80 launches, 512 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 7.01 us, warp 6.03 us, nsys graphs 6.94 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 5.23 | 86.7% | 5.84 | 84.0% |
| wait | 0.33 | 5.5% | 0.24 | 3.5% |
| load | 0.27 | 4.5% | 0.62 | 8.9% |
| compute | 0.12 | 2.0% | 0.12 | 1.8% |
| store | 0.08 | 1.3% | 0.13 | 1.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync | 0.33 | 5.5% | 1.0 | 1.42 | `__syncthreads();` |
| load.global | 0.27 | 4.5% | 1.0 | 0.42 | `for (int j = 0; j < 16; ++j) \| for (int run = 0; run < 2; ++run)` |
| compute.reduce | 0.12 | 2.0% | 1.0 | 0.13 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| store.global | 0.08 | 1.3% | 1.0 | 0.13 | `if (((int)threadIdx.x) == 0)` |

### cache_writer_kernel grid (5, 1, 1) block (128, 1, 1)

120 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 4.99 us, warp 4.95 us, nsys graphs 5.50 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 2.81 | 56.7% | 2.08 | 47.5% |
| store | 1.78 | 35.9% | 1.83 | 42.0% |
| load | 0.30 | 6.0% | 0.39 | 8.9% |
| compute | 0.07 | 1.4% | 0.07 | 1.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 1.78 | 35.9% | 1.0 | 2.00 | `if (((int64_t)0 <= slot) & ((slot >> (int64_t)7) < ((int64_t)pages))) \| if (((int64_t)0 <=` |
| load.global | 0.24 | 4.8% | 1.0 | 0.26 | `for (int i = 0; i < 4; ++i)` |
| compute.reduce | 0.07 | 1.4% | 1.0 | 0.10 | `amax = max(amax, __shfl_xor_sync((uint)4294967295, amax, 1, 32));` |
| load.global#2 | 0.06 | 1.2% | 1.0 | 0.06 | `int64_t slot = slots[((int64_t)((int)blockIdx.x))];` |

### router_partials_kernel grid (3, 1, 1) block (64, 1, 1)

96 launches, 6 warps; ranges, warp and last warp from the 3 recording CTAs' 6 warps; span 5.12 us, warp 3.10 us, nsys graphs 4.83 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.30 | 42.1% | 2.03 | 37.8% |
| load | 0.98 | 31.5% | 1.78 | 33.1% |
| compute | 0.68 | 22.0% | 1.36 | 25.4% |
| store | 0.14 | 4.4% | 0.20 | 3.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.reduce | 0.68 | 22.0% | 0.5 | 1.38 | `for (int k = 0; k < 3; ++k) \| for (int k = 0; k < 6; ++k)` |
| load.global#4 | 0.50 | 16.0% | 0.5 | 1.06 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < image_len)) \| if ((((((int)bl` |
| load.global | 0.37 | 12.0% | 0.8 | 0.45 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < mask_len))` |
| load.global#2 | 0.05 | 1.7% | 0.8 | 0.06 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < mask_len))` |
| load.global#3 | 0.05 | 1.7% | 0.8 | 0.06 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < mask_len))` |
| store.global | 0.05 | 1.5% | 0.8 | 0.06 | `if (condval == (bool)0) \| if (condval == (bool)1)` |
| store.global#3 | 0.05 | 1.5% | 0.5 | 0.10 | `if ((((int)threadIdx.x) & 31) < 3) \| if ((((int)threadIdx.x) & 31) < 6)` |
| store.global#2 | 0.04 | 1.3% | 0.8 | 0.06 | `if (condval_1 == (bool)0) \| if (condval_1 == (bool)1)` |

### norm_forward_and_per_token_cast_kernel_kernel grid (1, 1, 1) block (128, 1, 1)

176 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 2.75 us, warp 2.75 us, nsys graphs 1.28 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 2.10 | 76.2% | 2.16 | 77.6% |
| load | 0.38 | 14.0% | 0.41 | 14.8% |
| compute | 0.11 | 4.1% | 0.11 | 3.9% |
| wait | 0.10 | 3.6% | 0.05 | 1.6% |
| unattributed | 0.06 | 2.2% | 0.06 | 2.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*4 (est.: sampled iterations x4) | 1.54 | 55.8% | 1.0 | 1.54 | `for (int j_2 = 0; j_2 < 2; ++j_2)` |
| store.global*4 (est.: sampled iterations x4) | 0.35 | 12.8% | 1.0 | 0.38 | `for (int j_1 = 0; j_1 < 4; ++j_1)` |
| load.global*4 (est.: sampled iterations x4) | 0.22 | 8.1% | 1.0 | 0.26 | `if (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 4)) < num_tokens)` |
| load.global#3*4 (est.: sampled iterations x4) | 0.13 | 4.7% | 1.0 | 0.13 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 128) + ((((int)threadIdx.x) & 15` |
| store.global#5*4 (est.: sampled iterations x4) | 0.13 | 4.7% | 0.8 | 0.13 | `*(uint4*)(out + ((((((int64_t)vi_1) * (int64_t)128) + ((((int64_t)((int)threadIdx.x)) & (i` |
| compute.reduce | 0.11 | 4.1% | 1.0 | 0.13 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| wait.sync | 0.10 | 3.6% | 1.0 | 0.29 | `__syncthreads();` |
| store.global#2 | 0.05 | 1.7% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| load.global#2 | 0.03 | 1.2% | 1.0 | 0.03 | `*(float2*)(row_partial + 0) = *(float2*)(sum_shared + ((((int)threadIdx.x) >> 6) * 2)); \| ` |
| store.global#3 | 0.03 | 1.2% | 0.8 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |
| load.global#3*2 (est.: sampled iterations x2) | 0.00 | 0.0% | 0.0 | 0.00 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 64) + ((((int)threadIdx.x) & 7) ` |
| load.global*2 (est.: sampled iterations x2) | 0.00 | 0.0% | 0.0 | 0.00 | `if (((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < num_tokens)` |
| store.global#4*2 (est.: sampled iterations x2) | 0.00 | 0.0% | 0.0 | 0.00 | `for (int j_2 = 0; j_2 < 2; ++j_2)` |
| store.global#5*2 (est.: sampled iterations x2) | 0.00 | 0.0% | 0.0 | 0.00 | `*(uint4*)(out + ((((((int64_t)vi_1) * (int64_t)64) + ((((int64_t)((int)threadIdx.x)) & (in` |
| store.global*2 (est.: sampled iterations x2) | 0.00 | 0.0% | 0.0 | 0.00 | `for (int j_1 = 0; j_1 < 4; ++j_1)` |

### moe_swiglu_cast_kernel_kernel grid (8, 30, 1) block (288, 1, 1)

138 launches, 2160 warps; ranges, warp and last warp from the 4 recording CTAs' 36 warps; span 2.82 us, warp 1.10 us, nsys graphs 2.94 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.65 | 59.1% | 1.37 | 58.2% |
| load | 0.24 | 22.0% | 0.46 | 19.5% |
| compute | 0.11 | 10.2% | 0.31 | 13.0% |
| store | 0.10 | 8.7% | 0.22 | 9.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.24 | 22.0% | 1.0 | 0.35 | `for (int r = 0; r < 4; ++r)` |
| compute.reduce*4 (est.: sampled iterations x4) | 0.11 | 10.2% | 0.5 | 0.38 | `if (0 <= orders[r_2])` |
| store.global*4 (est.: sampled iterations x4) | 0.10 | 8.7% | 0.5 | 0.26 | `if (0 <= orders[r_2])` |

### rope_kernel grid (1, 1, 1) block (128, 1, 1)

256 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 1.31 us, warp 1.13 us, nsys graphs 1.44 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.97 | 85.8% | 1.25 | 79.9% |
| load | 0.14 | 12.8% | 0.28 | 17.6% |
| store | 0.02 | 1.4% | 0.04 | 2.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.10 | 8.5% | 1.0 | 0.10 | `if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < tokens)) \| int64_t position ` |
| load.global#2 | 0.05 | 4.3% | 0.5 | 0.10 | `if (((position < ((int64_t)positions_max)) && ((int64_t)0 <= position)))` |
| store.global | 0.02 | 1.4% | 0.5 | 0.03 | `*(uint4*)(x + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + ((((int64_t)((int)thread` |

### metadata_builder_kernel grid (2, 1, 1) block (128, 1, 1)

272 launches, 8 warps; ranges, warp and last warp from the 2 recording CTAs' 8 warps; span 1.22 us, warp 0.67 us, nsys graphs 1.44 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.30 | 44.5% | 0.34 | 24.6% |
| load | 0.30 | 44.2% | 0.98 | 71.2% |
| store | 0.08 | 11.3% | 0.06 | 4.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#3 | 0.25 | 37.6% | 0.5 | 0.86 | `for (int __1 = 0; __1 < 2; ++__1)` |
| store.global#4 | 0.03 | 4.8% | 0.5 | 0.06 | `positions[(((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))` |
| load.global#2 | 0.03 | 4.2% | 0.1 | 0.22 | `if ((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) < nr))` |
| store.global#3 | 0.03 | 4.2% | 0.5 | 0.06 | `if (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) == 0)` |
| load.global | 0.02 | 2.4% | 0.1 | 0.13 | `if ((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) <= nr))` |
| store.global | 0.01 | 1.8% | 0.1 | 0.10 | `out_starts[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] = condval;` |
| store.global#2 | 0.00 | 0.6% | 0.1 | 0.03 | `request_positions[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] = condval_2;` |

### moe_grouped_fp8_fp4_gemm_kernel grid (40, 36, 1) block (128, 1, 1)

6 launches, 5760 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 62.14 us, warp 2.82 us, nsys graphs 9.39 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.25 | 44.4% | 0.20 | 63.8% |
| load | 0.52 | 18.3% | 0.06 | 20.7% |
| wait | 0.50 | 17.7% | 0.05 | 15.5% |
| compute | 0.48 | 17.0% | 0.00 | 0.0% |
| store | 0.07 | 2.6% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*5 (est.: sampled iterations x5) | 0.48 | 17.0% | 0.2 | 1.92 | `if (ko < 4)` |
| load#2*5 (est.: sampled iterations x5) | 0.23 | 8.1% | 0.2 | 0.96 | `if (ko < 3)` |
| load.global#3 | 0.12 | 4.1% | 0.2 | 0.48 | `if (((expert < 128) && (0 <= expert))) \| if (((expert < 384) && (0 <= expert)))` |
| load.global | 0.10 | 3.4% | 1.0 | 0.19 | `int expert = TileExpert[((int64_t)((int)blockIdx.y))];` |
| wait.mbarrier*5 (est.: sampled iterations x5) | 0.09 | 3.2% | 0.2 | 0.48 | `loaded[(ko % 3)].wait((ko / 3));` |
| wait.sync#5*5 (est.: sampled iterations x5) | 0.09 | 3.2% | 0.2 | 0.48 | `__syncthreads();` |
| wait.sync#4*5 (est.: sampled iterations x5) | 0.08 | 2.8% | 0.2 | 0.32 | `__syncthreads();` |
| wait.sync#6*5 (est.: sampled iterations x5) | 0.08 | 2.8% | 0.2 | 0.32 | `__syncthreads();` |
| store.global | 0.07 | 2.6% | 0.2 | 0.35 | `for (int i_2 = 0; i_2 < 16; ++i_2) \| for (int i_3 = 0; i_3 < 16; ++i_3) \| for (int i_4 = 0` |
| wait.cp_async*5 (est.: sampled iterations x5) | 0.07 | 2.3% | 0.2 | 0.32 | `tl::cp_async_wait<2>();` |
| wait.sync | 0.05 | 1.7% | 1.0 | 0.06 | `__syncthreads();` |
| load | 0.05 | 1.6% | 0.2 | 0.26 | `for (int s = 0; s < 2; ++s)` |
| wait.sync#2 | 0.04 | 1.3% | 0.2 | 0.19 | `__syncthreads();` |
| load.global#2 | 0.03 | 0.9% | 0.2 | 0.22 | `if (((int)threadIdx.x) < 32) \| if (((int)threadIdx.x) < 64)` |
| wait.sync#3 | 0.01 | 0.4% | 0.2 | 0.05 | `__syncthreads();` |

### cache_writer_kernel grid (10, 1, 1) block (128, 1, 1)

48 launches, 40 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 8.42 us, warp 8.12 us, nsys graphs 5.22 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 4.53 | 55.8% | 4.11 | 53.1% |
| unattributed | 3.20 | 39.5% | 3.08 | 39.7% |
| load | 0.31 | 3.8% | 0.44 | 5.6% |
| compute | 0.08 | 0.9% | 0.12 | 1.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 4.53 | 55.8% | 1.0 | 4.72 | `if (((int64_t)0 <= slot) & ((slot >> (int64_t)7) < ((int64_t)pages))) \| if (((int64_t)0 <=` |
| load.global | 0.25 | 3.1% | 1.0 | 0.37 | `for (int i = 0; i < 4; ++i)` |
| compute.reduce | 0.08 | 0.9% | 1.0 | 0.10 | `amax = max(amax, __shfl_xor_sync((uint)4294967295, amax, 1, 32));` |
| load.global#2 | 0.06 | 0.7% | 1.0 | 0.06 | `int64_t slot = slots[((int64_t)((int)blockIdx.x))];` |

### token_ranks_kernel grid (1, 1, 1) block (256, 1, 1)

48 launches, 8 warps; ranges, warp and last warp from the 1 recording CTAs' 8 warps; span 7.62 us, warp 7.61 us, nsys graphs 8.10 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 6.86 | 90.0% | 7.01 | 90.8% |
| compute | 0.37 | 4.8% | 0.30 | 3.9% |
| unattributed | 0.35 | 4.5% | 0.37 | 4.8% |
| store | 0.04 | 0.6% | 0.04 | 0.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 6.86 | 90.0% | 1.0 | 7.02 | `if ((((int64_t)0 <= token_ids[((int64_t)((int)blockIdx.x))]) && (token_ids[((int64_t)((int` |
| compute.reduce | 0.37 | 4.8% | 1.0 | 0.58 | `total[0] = tl::AllReduce<tl::SumOp, 256, 1, 0, tl::NamedBarrier<256>>::run(total[0], (&(wo` |
| store.global | 0.04 | 0.6% | 1.0 | 0.06 | `output[((int64_t)((int)blockIdx.x))] = ((int64_t)total[0]);` |

### metadata_decode_kernel grid (5, 2, 1) block (128, 1, 1)

60 launches, 40 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 2.35 us, warp 1.63 us, nsys graphs 2.51 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.95 | 58.4% | 3.55 | 63.2% |
| unattributed | 0.59 | 36.0% | 1.70 | 30.3% |
| store | 0.09 | 5.6% | 0.36 | 6.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.70 | 43.1% | 0.6 | 1.39 | `if ((((offset + ((int)blockIdx.x)) < src_rows) && (0 <= (offset + ((int)blockIdx.x)))))` |
| load.global#3 | 0.22 | 13.5% | 0.4 | 0.59 | `if ((((offset + ((int)blockIdx.x)) < main_src_rows) && (0 <= (offset + ((int)blockIdx.x)))` |
| store.global#4 | 0.05 | 2.8% | 0.4 | 0.24 | `if (0 < live)` |
| store.global | 0.03 | 2.1% | 0.6 | 0.06 | `swa[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = con` |
| load.global#2 | 0.03 | 1.8% | 0.1 | 0.19 | `if (((0 <= req) & ((int64_t)0 <= pos)))` |
| store.global#2 | 0.01 | 0.4% | 0.1 | 0.06 | `lengths[((int64_t)((int)blockIdx.x))] = ((int)max(info[1], (int64_t)0));` |
| store.global#3 | 0.01 | 0.4% | 0.1 | 0.06 | `top_lengths[((int64_t)((int)blockIdx.x))] = condval_6; \| top_lengths[((int64_t)((int)block` |

### indexer_topk_segment_kernel grid (1, 5, 1) block (512, 1, 1)

30 launches, 80 warps; ranges, warp and last warp from the 5 recording CTAs' 80 warps; span 10.90 us, warp 10.68 us, nsys graphs 5.31 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed, with loops' unsampled iterations | 5.70 | 53.3% | 5.48 | 50.8% |
| wait | 3.65 | 34.2% | 3.88 | 36.0% |
| load | 1.17 | 11.0% | 1.12 | 10.4% |
| store | 0.16 | 1.5% | 0.29 | 2.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.17 | 11.0% | 1.0 | 1.36 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 4096); \| i` |
| wait.sync (sampled iterations) | 1.00 | 9.4% | 14.5 | 1.20 | `__syncthreads();` |
| wait.sync#24*9 (est.: sampled iterations x9) | 0.30 | 2.8% | 0.5 | 0.43 | `__syncthreads();` |
| wait.sync#23*9 (est.: sampled iterations x9) | 0.30 | 2.8% | 0.5 | 0.43 | `__syncthreads();` |
| wait.sync#19*9 (est.: sampled iterations x9) | 0.29 | 2.7% | 0.5 | 0.29 | `__syncthreads();` |
| wait.sync#14*8 (est.: sampled iterations x8) | 0.28 | 2.7% | 0.5 | 0.38 | `__syncthreads();` |
| wait.sync#6*8 (est.: sampled iterations x8) | 0.27 | 2.6% | 0.5 | 0.38 | `__syncthreads();` |
| wait.sync#13*8 (est.: sampled iterations x8) | 0.26 | 2.5% | 0.5 | 0.38 | `__syncthreads();` |
| wait.sync#5*8 (est.: sampled iterations x8) | 0.26 | 2.4% | 0.5 | 0.38 | `__syncthreads();` |
| wait.sync#20*9 (est.: sampled iterations x9) | 0.22 | 2.0% | 0.5 | 0.29 | `__syncthreads();` |
| store.global | 0.16 | 1.5% | 1.5 | 0.22 | `for (int i_3 = 0; i_3 < 4; ++i_3) \| if (((int)threadIdx.x) < total) \| if (((int)threadIdx.` |
| wait.sync#25 | 0.05 | 0.4% | 0.5 | 0.08 | `__syncthreads();` |
| wait.sync#16 | 0.04 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#17 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#8 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#18 | 0.03 | 0.3% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#9 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#11 | 0.03 | 0.3% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#2 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#15 | 0.03 | 0.3% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#3 | 0.03 | 0.3% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#12 | 0.03 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#4 | 0.03 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#22 | 0.03 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#21 | 0.02 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#10 | 0.02 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#7 | 0.02 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |

### topk_ids_kernel grid (5, 1, 1) block (512, 1, 1)

15 launches, 80 warps; ranges, warp and last warp from the 5 recording CTAs' 80 warps; span 20.86 us, warp 18.86 us, nsys graphs 20.40 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 15.34 | 81.3% | 15.29 | 73.4% |
| unattributed | 2.25 | 11.9% | 1.95 | 9.4% |
| wait | 0.84 | 4.5% | 0.71 | 3.4% |
| store | 0.43 | 2.3% | 2.88 | 13.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 15.34 | 81.3% | 1.0 | 16.26 | `for (int t = 0; t < (vocab >> 14); ++t)` |
| wait.sync | 0.71 | 3.7% | 1.0 | 1.41 | `__syncthreads();` |
| store.global | 0.43 | 2.3% | 1.0 | 2.98 | `if ((((int)threadIdx.x) >> 5) == 0)` |
| wait.sync#2 | 0.08 | 0.4% | 1.0 | 0.16 | `__syncthreads();` |
| wait.sync#3 | 0.06 | 0.3% | 1.0 | 0.06 | `__syncthreads();` |

### compress_kernel grid (2, 1, 1) block (128, 1, 1)

32 launches, 8 warps; ranges, warp and last warp from the 2 recording CTAs' 8 warps; span 8.27 us, warp 5.78 us, nsys graphs 14.08 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 2.99 | 51.7% | 5.70 | 58.5% |
| unattributed | 2.28 | 39.5% | 3.06 | 31.4% |
| store | 0.45 | 7.7% | 0.89 | 9.1% |
| compute | 0.06 | 1.0% | 0.10 | 1.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 2.44 | 42.2% | 0.6 | 3.71 | `if ((((0 <= counts[0]) & (0 < counts[1])) & (counts[1] <= 2)) & (((((int)blockIdx.x) * 4) ` |
| load.global#2*4 (est.: sampled iterations x4) | 0.49 | 8.4% | 0.6 | 1.28 | `*(float4*)(v1 + 0) = *(float4*)(values + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048)` |
| store.global*4 (est.: sampled iterations x4) | 0.18 | 3.0% | 0.6 | 0.32 | `*(uint2*)(out + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + ((((int64_t)((int)thre` |
| store.global#3 | 0.10 | 1.7% | 0.3 | 0.16 | `emitted_slots[((((int64_t)((int)blockIdx.x)) * (int64_t)4) + (((int64_t)((int)threadIdx.x)` |
| store.global#2*4 (est.: sampled iterations x4) | 0.07 | 1.2% | 0.3 | 0.13 | `if ((slot >> (int64_t)3) < ((int64_t)blocks))` |
| compute.reduce | 0.06 | 1.0% | 0.6 | 0.10 | `float total = tl::warp_reduce_sum(ss[0]);` |
| load.global#3*4 (est.: sampled iterations x4) | 0.04 | 0.7% | 0.3 | 0.06 | `*(float4*)(w + 0) = *(float4*)(weight + ((c_1 * 128) + ((((int)threadIdx.x) & 31) * 4)));` |
| store.global#3*4 (est.: sampled iterations x4) | 0.04 | 0.7% | 0.3 | 0.06 | `*(uint2*)(out + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + ((((int64_t)((int)thre` |
| store.global#5 | 0.04 | 0.7% | 0.3 | 0.16 | `emitted_slots[((((int64_t)((int)blockIdx.x)) * (int64_t)4) + (((int64_t)((int)threadIdx.x)` |
| load.global#3 | 0.01 | 0.2% | 0.3 | 0.03 | `if (valid)` |
| load.global#4 | 0.01 | 0.2% | 0.3 | 0.02 | `if (valid)` |
| store.global#2 | 0.01 | 0.2% | 0.3 | 0.02 | `emitted[((((int64_t)((int)blockIdx.x)) * (int64_t)4) + (((int64_t)((int)threadIdx.x)) >> (` |
| store.global#4 | 0.01 | 0.2% | 0.3 | 0.02 | `emitted[((((int64_t)((int)blockIdx.x)) * (int64_t)4) + (((int64_t)((int)threadIdx.x)) >> (` |

### norm_forward_and_per_token_cast_kernel_kernel grid (3, 1, 1) block (128, 1, 1)

96 launches, 12 warps; ranges, warp and last warp from the 3 recording CTAs' 12 warps; span 2.78 us, warp 2.57 us, nsys graphs 1.89 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 1.62 | 63.2% | 1.84 | 61.5% |
| unattributed | 0.35 | 13.5% | 0.63 | 21.2% |
| load | 0.34 | 13.4% | 0.34 | 11.5% |
| wait | 0.14 | 5.3% | 0.05 | 1.7% |
| compute | 0.12 | 4.6% | 0.12 | 4.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*3 (est.: sampled iterations x3) | 1.15 | 44.9% | 1.0 | 1.15 | `for (int j_2 = 0; j_2 < 2; ++j_2)` |
| store.global*3 (est.: sampled iterations x3) | 0.29 | 11.2% | 1.0 | 0.38 | `for (int j_1 = 0; j_1 < 4; ++j_1)` |
| load.global*3 (est.: sampled iterations x3) | 0.18 | 6.9% | 1.0 | 0.19 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 6)) < num_tokens) && (((vi * 2) + (` |
| load.global#3*3 (est.: sampled iterations x3) | 0.14 | 5.3% | 1.0 | 0.19 | `if (((vi_1 * 2) + ((((int)threadIdx.x) & 63) >> 5)) < 5)` |
| wait.sync | 0.14 | 5.3% | 1.0 | 0.26 | `__syncthreads();` |
| compute.reduce | 0.12 | 4.6% | 1.0 | 0.13 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| store.global#5*3 (est.: sampled iterations x3) | 0.11 | 4.4% | 0.8 | 0.19 | `*(uint4*)(out + ((((((int64_t)vi_1) * (int64_t)512) + ((((int64_t)((int)threadIdx.x)) & (i` |
| store.global#2 | 0.04 | 1.7% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| load.global#2 | 0.03 | 1.2% | 1.0 | 0.03 | `*(float2*)(row_partial + 0) = *(float2*)(sum_shared + ((((int)threadIdx.x) >> 6) * 2)); \| ` |
| store.global#3 | 0.03 | 1.0% | 0.4 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |

### sparse_mla_kernel_1 grid (16, 5, 1) block (128, 1, 1)

90 launches, 80 CTAs, 320 warps; span 1.92 us, warp 1.42 us, nsys graphs 1.98 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.37 | 96.4% | 1.58 | 51.0% |
| load | 0.03 | 2.4% | 1.27 | 41.3% |
| wait | 0.01 | 0.8% | 0.11 | 3.7% |
| compute | 0.00 | 0.3% | 0.08 | 2.6% |
| store | 0.00 | 0.1% | 0.04 | 1.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.03 | 2.4% | 0.1 | 0.86 | `M = __fmul_rn(Sinks[((int)blockIdx.x)], 0x1.7154764ee6c2fp+0f/*1.442695e+00*/); \| if (((in` |
| wait.sync | 0.01 | 0.8% | 0.1 | 0.32 | `__syncthreads();` |
| compute.reduce | 0.00 | 0.3% | 0.1 | 0.10 | `amax[0] = tl::AllReduce<tl::MaxOp, 8, 1, 0, tl::NamedBarrier<128>>::run(amax[0]);` |
| store.global | 0.00 | 0.1% | 0.1 | 0.06 | `*(uint2*)(Output + (((((int64_t)((int)blockIdx.y)) * (int64_t)8192) + (((int64_t)((int)blo` |
| compute.reduce#3 | 0.00 | 0.0% | 0.0 | 0.00 | `amax[0] = tl::AllReduce<tl::MaxOp, 8, 1, 0, tl::NamedBarrier<128>>::run(amax[0]);` |
| load.global#3 | 0.00 | 0.0% | 0.0 | 0.00 | `M = __fmul_rn(Sinks[((int)blockIdx.x)], 0x1.7154764ee6c2fp+0f/*1.442695e+00*/);` |
| load.global#4 | 0.00 | 0.0% | 0.0 | 0.00 | `if (((int)threadIdx.x) < 32)` |
| store.global#2 | 0.00 | 0.0% | 0.0 | 0.00 | `*(uint2*)(Output + (((((int64_t)((int)blockIdx.y)) * (int64_t)8192) + (((int64_t)((int)blo` |
| wait.sync#12 | 0.00 | 0.0% | 0.0 | 0.00 | `__syncthreads();` |
| wait.sync#13 | 0.00 | 0.0% | 0.0 | 0.00 | `__syncthreads();` |

### topk_log_softmax_kernel grid (1, 1, 1) block (1024, 1, 1)

16 launches, 32 warps; ranges, warp and last warp from the 1 recording CTAs' 32 warps; span 16.67 us, warp 16.11 us, nsys graphs 15.87 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 12.94 | 80.3% | 13.74 | 82.4% |
| unattributed, with loops' unsampled iterations | 1.58 | 9.8% | 1.73 | 10.4% |
| wait | 1.20 | 7.5% | 0.76 | 4.5% |
| compute | 0.39 | 2.4% | 0.39 | 2.3% |
| store | 0.00 | 0.0% | 0.06 | 0.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 6.46 | 40.1% | 1.0 | 6.90 | `if (0 < (vocab & 2047)) \| if (0 < (vocab & 8191))` |
| load.global#2 | 6.46 | 40.1% | 1.0 | 6.85 | `for (int step_1 = 0; step_1 < (vocab >> 11); ++step_1) \| for (int step_1 = 0; step_1 < (vo` |
| wait.sync | 0.51 | 3.2% | 1.0 | 1.33 | `__syncthreads();` |
| wait.sync#2 | 0.44 | 2.7% | 1.0 | 0.83 | `__syncthreads();` |
| compute.reduce#3 | 0.18 | 1.1% | 1.0 | 0.19 | `ties[0] = vllm_add(ties[0], __shfl_xor_sync((uint)4294967295, ties[0], 16, 32));` |
| compute.reduce#2 | 0.11 | 0.7% | 1.0 | 0.16 | `se[0] = vllm_add(se[0], __shfl_xor_sync((uint)4294967295, se[0], 16, 32));` |
| compute.reduce | 0.10 | 0.6% | 1.0 | 0.13 | `best[0] = vllm_max(best[0], __shfl_xor_sync((uint)4294967295, best[0], 16, 32));` |
| wait.sync#4 | 0.10 | 0.6% | 1.0 | 0.13 | `__syncthreads();` |
| wait.sync#5 | 0.08 | 0.5% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#3 | 0.07 | 0.5% | 1.0 | 0.10 | `__syncthreads();` |
| load.global#3 (sampled iterations) | 0.01 | 0.1% | 0.0 | 0.38 | `float value_32 = vllm_sub(vllm_shifted(logits[((((int64_t)((int)blockIdx.x)) * logits_stri` |
| store.global (sampled iterations) | 0.00 | 0.0% | 0.0 | 0.06 | `output[(((((int64_t)c) * (int64_t)1024) + (((int64_t)((int)blockIdx.x)) * ((int64_t)topk))` |

### topk_log_softmax_kernel grid (5, 1, 1) block (1024, 1, 1)

15 launches, 160 warps; ranges, warp and last warp from the 5 recording CTAs' 160 warps; span 16.96 us, warp 16.18 us, nsys graphs 16.74 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 13.17 | 81.4% | 14.04 | 82.9% |
| unattributed, with loops' unsampled iterations | 1.60 | 9.9% | 1.72 | 10.2% |
| wait | 1.01 | 6.2% | 0.72 | 4.2% |
| compute | 0.39 | 2.4% | 0.39 | 2.3% |
| store | 0.00 | 0.0% | 0.07 | 0.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 6.69 | 41.4% | 1.0 | 7.01 | `if (0 < (vocab & 2047)) \| if (0 < (vocab & 8191))` |
| load.global#2 | 6.47 | 40.0% | 1.0 | 7.07 | `for (int step_1 = 0; step_1 < (vocab >> 11); ++step_1) \| for (int step_1 = 0; step_1 < (vo` |
| wait.sync#2 | 0.47 | 2.9% | 1.0 | 0.99 | `__syncthreads();` |
| wait.sync | 0.28 | 1.7% | 1.0 | 0.83 | `__syncthreads();` |
| compute.reduce#3 | 0.17 | 1.1% | 1.0 | 0.22 | `ties[0] = vllm_add(ties[0], __shfl_xor_sync((uint)4294967295, ties[0], 16, 32));` |
| compute.reduce#2 | 0.11 | 0.7% | 1.0 | 0.16 | `se[0] = vllm_add(se[0], __shfl_xor_sync((uint)4294967295, se[0], 16, 32));` |
| compute.reduce | 0.11 | 0.7% | 1.0 | 0.16 | `best[0] = vllm_max(best[0], __shfl_xor_sync((uint)4294967295, best[0], 16, 32));` |
| wait.sync#4 | 0.10 | 0.6% | 1.0 | 0.16 | `__syncthreads();` |
| wait.sync#5 | 0.08 | 0.5% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#3 | 0.08 | 0.5% | 1.0 | 0.10 | `__syncthreads();` |
| load.global#3 (sampled iterations) | 0.01 | 0.1% | 0.0 | 0.38 | `float value_32 = vllm_sub(vllm_shifted(logits[((((int64_t)((int)blockIdx.x)) * logits_stri` |
| store.global (sampled iterations) | 0.00 | 0.0% | 0.0 | 0.10 | `output[(((((int64_t)c) * (int64_t)1024) + (((int64_t)((int)blockIdx.x)) * ((int64_t)topk))` |

### metadata_chunk_kernel grid (5, 1, 1) block (128, 1, 1)

30 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 4.78 us, warp 4.12 us, nsys graphs 4.86 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 2.61 | 63.3% | 2.88 | 42.7% |
| load | 1.44 | 35.0% | 3.71 | 55.0% |
| store | 0.07 | 1.7% | 0.15 | 2.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.35 | 32.7% | 1.0 | 1.65 | `if ((((offset + ((int)blockIdx.x)) < src_rows) && (0 <= (offset + ((int)blockIdx.x)))))` |
| load.global#2 | 0.10 | 2.4% | 0.2 | 0.40 | `if (((0 <= req) & ((int64_t)0 <= pos)))` |
| store.global | 0.05 | 1.2% | 1.0 | 0.06 | `swa[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = con` |
| store.global#3 | 0.01 | 0.2% | 0.2 | 0.06 | `top_lengths[((int64_t)((int)blockIdx.x))] = condval_12; \| top_lengths[((int64_t)((int)bloc` |
| store.global#2 | 0.01 | 0.2% | 0.2 | 0.05 | `lengths[((int64_t)((int)blockIdx.x))] = ((int)max(info[1], (int64_t)0));` |

### mhc_collapse_kernel grid (25, 1, 1) block (128, 1, 1)

45 launches, 100 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 3.74 us, warp 3.68 us, nsys graphs 2.24 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 3.53 | 96.0% | 3.84 | 91.7% |
| load | 0.10 | 2.6% | 0.30 | 7.1% |
| store | 0.05 | 1.3% | 0.05 | 1.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.10 | 2.6% | 1.0 | 0.13 | `*(float4*)(mv + 0) = *(float4*)(mix + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int6` |
| store.global | 0.05 | 1.3% | 1.0 | 0.06 | `*(uint4*)(out + ((((int64_t)((int)blockIdx.x)) * (int64_t)1024) + (((int64_t)((int)threadI` |

### moe_combine_kernel grid (5, 5, 1) block (128, 1, 1)

90 launches, 100 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 2.02 us, warp 1.49 us, nsys graphs 2.18 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.03 | 69.4% | 1.61 | 77.6% |
| unattributed | 0.40 | 26.8% | 0.41 | 19.9% |
| store | 0.06 | 3.8% | 0.05 | 2.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.03 | 69.4% | 1.0 | 1.57 | `for (int k = 0; k < 3; ++k) \| for (int k = 0; k < 6; ++k) \| if ((((int)blockIdx.y) < share` |
| store.global | 0.06 | 3.8% | 1.0 | 0.06 | `{` |

### local_stats_kernel grid (5, 16, 1) block (128, 1, 1)

15 launches, 320 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 12.26 us, warp 11.54 us, nsys graphs 12.48 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 10.54 | 91.3% | 11.13 | 91.3% |
| unattributed | 0.65 | 5.6% | 0.68 | 5.6% |
| compute | 0.17 | 1.5% | 0.17 | 1.4% |
| wait | 0.11 | 1.0% | 0.10 | 0.8% |
| store | 0.07 | 0.6% | 0.10 | 0.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 10.54 | 91.3% | 1.0 | 11.10 | `if (expanded_local_pos[((int64_t)((int)blockIdx.x))] < num_speculative_steps)` |
| compute.reduce | 0.12 | 1.0% | 1.0 | 0.13 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| wait.sync | 0.11 | 1.0% | 1.0 | 0.38 | `__syncthreads();` |
| store.global | 0.07 | 0.6% | 1.0 | 0.13 | `if (((int)threadIdx.x) == 0)` |
| compute.reduce#2 | 0.06 | 0.5% | 1.0 | 0.06 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 2, 32);` |

### norm_forward_and_per_token_cast_kernel_kernel grid (5, 1, 1) block (128, 1, 1)

63 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 2.82 us, warp 2.58 us, nsys graphs 1.92 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 1.62 | 62.6% | 1.91 | 65.8% |
| unattributed | 0.41 | 15.9% | 0.44 | 15.2% |
| load | 0.32 | 12.6% | 0.38 | 13.0% |
| compute | 0.12 | 4.6% | 0.12 | 4.2% |
| wait | 0.11 | 4.3% | 0.05 | 1.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*3 (est.: sampled iterations x3) | 1.12 | 43.5% | 1.0 | 1.15 | `for (int j_2 = 0; j_2 < 2; ++j_2)` |
| store.global*3 (est.: sampled iterations x3) | 0.30 | 11.7% | 1.0 | 0.38 | `for (int j_1 = 0; j_1 < 4; ++j_1)` |
| load.global*3 (est.: sampled iterations x3) | 0.17 | 6.7% | 1.0 | 0.19 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 6)) < num_tokens) && (((vi * 2) + (` |
| compute.reduce | 0.12 | 4.6% | 1.0 | 0.13 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| load.global#3*3 (est.: sampled iterations x3) | 0.12 | 4.6% | 1.0 | 0.19 | `if (((vi_1 * 2) + ((((int)threadIdx.x) & 63) >> 5)) < 5)` |
| store.global#5*3 (est.: sampled iterations x3) | 0.12 | 4.5% | 1.0 | 0.19 | `*(uint4*)(out + ((((((int64_t)vi_1) * (int64_t)512) + ((((int64_t)((int)threadIdx.x)) & (i` |
| wait.sync | 0.11 | 4.3% | 1.0 | 0.19 | `__syncthreads();` |
| store.global#2 | 0.04 | 1.7% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| load.global#2 | 0.03 | 1.2% | 1.0 | 0.03 | `*(float2*)(row_partial + 0) = *(float2*)(sum_shared + ((((int)threadIdx.x) >> 6) * 2)); \| ` |
| store.global#3 | 0.03 | 1.2% | 0.5 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |
| load.global#2*5 (est.: sampled iterations x5) | 0.00 | 0.0% | 0.0 | 0.00 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 1024) + (((int)threadIdx.x) * 8)` |
| load.global*5 (est.: sampled iterations x5) | 0.00 | 0.0% | 0.0 | 0.00 | `*(uint4*)(x_local + (vi * 8)) = *(uint4*)(x + ((((((int64_t)vi) * (int64_t)1024) + (((int6` |
| store.global#4*5 (est.: sampled iterations x5) | 0.00 | 0.0% | 0.0 | 0.00 | `for (int j_1 = 0; j_1 < 2; ++j_1)` |
| store.global*5 (est.: sampled iterations x5) | 0.00 | 0.0% | 0.0 | 0.00 | `for (int j = 0; j < 4; ++j)` |

### mhc_finalize_kernel grid (6, 1, 1) block (128, 1, 1)

12 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 14.27 us, warp 12.91 us, nsys graphs 7.76 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 8.53 | 66.1% | 8.10 | 53.7% |
| store | 1.64 | 12.7% | 2.65 | 17.6% |
| load | 1.08 | 8.3% | 1.95 | 12.9% |
| compute | 0.98 | 7.6% | 1.72 | 11.4% |
| wait | 0.68 | 5.3% | 0.66 | 4.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.71 | 5.5% | 0.2 | 2.86 | `if (((int)threadIdx.x) < 4)` |
| wait.sync | 0.68 | 5.3% | 1.0 | 0.98 | `__syncthreads();` |
| compute.reduce | 0.57 | 4.4% | 0.2 | 2.30 | `row_max[0] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<32>>::run(row_max[0]);` |
| store.global#4*7 (est.: sampled iterations x7) | 0.57 | 4.4% | 0.8 | 0.90 | `if ((((int)threadIdx.x) % 4) == 0)` |
| compute.reduce#2*7 (est.: sampled iterations x7) | 0.41 | 3.2% | 0.8 | 0.67 | `amax[0] = max(amax[0], __shfl_xor_sync((uint)4294967295, amax[0], 1, 32));` |
| load.global | 0.38 | 2.9% | 1.0 | 1.06 | `if (((int)threadIdx.x) < 32)` |
| load.global#4*7 (est.: sampled iterations x7) | 0.37 | 2.9% | 0.8 | 0.67 | `*(uint4*)(yv + 0) = *(uint4*)(C + ((((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((i` |
| store.global#3*7 (est.: sampled iterations x7) | 0.34 | 2.6% | 0.8 | 0.67 | `*(uint4*)(y + ((((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((int64_t)step) * (int6` |
| load.global#3 | 0.28 | 2.2% | 0.8 | 0.38 | `for (int i_split_2 = 0; i_split_2 < 40; ++i_split_2)` |
| load.global#2 | 0.04 | 0.3% | 0.2 | 0.19 | `cm[0] = ((mixes_shared[((((int)threadIdx.x) & 15) + 8)] * scale[2]) + base[((((int)threadI` |
| store.global#2 | 0.01 | 0.1% | 0.2 | 0.06 | `if ((((int)threadIdx.x) >> 4) == 0)` |

### per_token_cast_kernel_kernel grid (9, 1, 1) block (64, 1, 1)

144 launches, 18 warps; ranges, warp and last warp from the 5 recording CTAs' 10 warps; span 1.02 us, warp 0.93 us, nsys graphs 1.06 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.46 | 49.8% | 0.56 | 48.4% |
| load | 0.26 | 27.8% | 0.34 | 29.6% |
| store | 0.11 | 12.0% | 0.16 | 13.4% |
| compute | 0.10 | 10.3% | 0.10 | 8.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.17 | 18.2% | 1.0 | 0.26 | `for (int i = 0; i < 2; ++i)` |
| store.global#2 | 0.11 | 12.0% | 1.0 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.10 | 10.3% | 1.0 | 0.13 | `amax_fragment[i_1] = tl::AllReduce<tl::MaxOp, 2, 1, 0, tl::NamedBarrier<64>>::run(amax_fra` |
| load.global#2*2 (est.: sampled iterations x2) | 0.09 | 9.6% | 1.0 | 0.13 | `amax_fragment[i_1] = cutlass::bfloat16_t(__hmax((bfloat16_t(((nv_bfloat162*)(&(amax_fragme` |

### mhc_post_kernel grid (25, 1, 1) block (128, 1, 1)

60 launches, 100 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 2.50 us, warp 1.62 us, nsys graphs 2.14 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.46 | 90.6% | 2.10 | 91.1% |
| load | 0.10 | 6.4% | 0.11 | 4.6% |
| store | 0.05 | 3.0% | 0.10 | 4.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.10 | 6.4% | 1.0 | 0.13 | `*(float4*)(pm + 0) = *(float4*)(post + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int` |
| store.global | 0.05 | 3.0% | 1.0 | 0.06 | `for (int n_1 = 0; n_1 < 4; ++n_1)` |

### per_token_cast_kernel_kernel grid (5, 1, 1) block (64, 1, 1)

128 launches, 10 warps; ranges, warp and last warp from the 5 recording CTAs' 10 warps; span 1.22 us, warp 1.10 us, nsys graphs 1.06 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.46 | 41.4% | 0.46 | 37.9% |
| load | 0.34 | 31.0% | 0.35 | 29.1% |
| store | 0.20 | 18.6% | 0.30 | 24.8% |
| compute | 0.10 | 9.0% | 0.10 | 8.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.25 | 22.9% | 1.0 | 0.26 | `for (int i = 0; i < 2; ++i)` |
| store.global#2 | 0.12 | 10.4% | 1.0 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.10 | 9.0% | 1.0 | 0.13 | `amax_fragment[i_1] = tl::AllReduce<tl::MaxOp, 2, 1, 0, tl::NamedBarrier<64>>::run(amax_fra` |
| load.global#2*2 (est.: sampled iterations x2) | 0.09 | 8.1% | 1.0 | 0.13 | `amax_fragment[i_1] = cutlass::bfloat16_t(__hmax((bfloat16_t(((nv_bfloat162*)(&(amax_fragme` |
| store.global#3*2 (est.: sampled iterations x2) | 0.04 | 4.1% | 0.5 | 0.13 | `*(fp8_e4_16_t*)(out + ((((((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int64_t)10240) +` |
| store.global*2 (est.: sampled iterations x2) | 0.04 | 4.1% | 0.5 | 0.13 | `out_sf[((((((int64_t)((int)blockIdx.x)) % (int64_t)15) * (int64_t)32) + (((int64_t)((int)t` |

### sparse_mla_kernel_1 grid (16, 10, 1) block (128, 1, 1)

48 launches, 160 CTAs, 640 warps; span 1.81 us, warp 1.41 us, nsys graphs 5.70 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.39 | 98.2% | 1.74 | 54.7% |
| load | 0.02 | 1.1% | 1.18 | 37.2% |
| wait | 0.01 | 0.4% | 0.15 | 4.6% |
| compute | 0.00 | 0.2% | 0.07 | 2.3% |
| store | 0.00 | 0.1% | 0.04 | 1.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#3 | 0.01 | 1.0% | 0.0 | 0.64 | `M = __fmul_rn(Sinks[((int)blockIdx.x)], 0x1.7154764ee6c2fp+0f/*1.442695e+00*/);` |
| wait.sync#13 | 0.00 | 0.3% | 0.0 | 0.22 | `__syncthreads();` |
| load.global#4 | 0.00 | 0.2% | 0.0 | 0.22 | `if (((int)threadIdx.x) < 32)` |
| compute.reduce#3 | 0.00 | 0.2% | 0.0 | 0.10 | `amax[0] = tl::AllReduce<tl::MaxOp, 8, 1, 0, tl::NamedBarrier<128>>::run(amax[0]);` |
| wait.sync#12 | 0.00 | 0.1% | 0.0 | 0.10 | `__syncthreads();` |
| store.global#2 | 0.00 | 0.1% | 0.0 | 0.06 | `*(uint2*)(Output + (((((int64_t)((int)blockIdx.y)) * (int64_t)8192) + (((int64_t)((int)blo` |

### indexer_topk_segment_kernel grid (2, 5, 1) block (512, 1, 1)

15 launches, 160 warps; ranges, warp and last warp from the 5 recording CTAs' 80 warps; span 9.92 us, warp 9.87 us, nsys graphs 4.58 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 6.16 | 62.4% | 6.04 | 60.3% |
| unattributed, with loops' unsampled iterations | 3.14 | 31.9% | 3.22 | 32.1% |
| load | 0.47 | 4.8% | 0.65 | 6.5% |
| store | 0.09 | 0.9% | 0.11 | 1.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync#20*9 (est.: sampled iterations x9) | 0.68 | 6.9% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#24*9 (est.: sampled iterations x9) | 0.67 | 6.8% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#19*9 (est.: sampled iterations x9) | 0.66 | 6.7% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#23*9 (est.: sampled iterations x9) | 0.63 | 6.3% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#6*8 (est.: sampled iterations x8) | 0.61 | 6.2% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync#14*8 (est.: sampled iterations x8) | 0.60 | 6.0% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync#13*8 (est.: sampled iterations x8) | 0.58 | 5.8% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync#5*8 (est.: sampled iterations x8) | 0.58 | 5.8% | 1.0 | 0.77 | `__syncthreads();` |
| load.global#2 | 0.26 | 2.7% | 1.0 | 0.32 | `for (int i = 0; i < 8; ++i)` |
| load.global | 0.21 | 2.1% | 1.0 | 0.22 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 4096); \| i` |
| wait.sync (sampled iterations) | 0.10 | 1.0% | 1.0 | 0.19 | `__syncthreads();` |
| wait.sync#25 | 0.10 | 1.0% | 1.0 | 0.16 | `__syncthreads();` |
| store.global | 0.09 | 0.9% | 1.0 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3) \| if (((int)threadIdx.x) < total) \| if (((int)threadIdx.` |
| wait.sync#17 | 0.08 | 0.8% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#18 | 0.08 | 0.8% | 1.0 | 0.16 | `__syncthreads();` |
| wait.sync#8 | 0.07 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#16 | 0.07 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#3 | 0.07 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#2 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#9 | 0.06 | 0.6% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#7 | 0.06 | 0.6% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#15 | 0.06 | 0.6% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#11 | 0.06 | 0.6% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#4 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#21 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#12 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#10 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#22 | 0.05 | 0.5% | 1.0 | 0.06 | `__syncthreads();` |

### router_partials_kernel grid (5, 1, 1) block (64, 1, 1)

48 launches, 10 warps; ranges, warp and last warp from the 5 recording CTAs' 10 warps; span 2.69 us, warp 1.96 us, nsys graphs 2.37 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.80 | 40.9% | 1.28 | 43.8% |
| unattributed | 0.80 | 40.6% | 1.09 | 37.4% |
| store | 0.19 | 9.6% | 0.21 | 7.1% |
| compute | 0.17 | 8.8% | 0.34 | 11.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.42 | 21.6% | 1.0 | 0.45 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < mask_len))` |
| load.global#4 | 0.25 | 12.8% | 0.5 | 0.54 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < image_len)) \| if ((((((int)bl` |
| compute.reduce | 0.17 | 8.8% | 0.5 | 0.35 | `for (int k = 0; k < 3; ++k) \| for (int k = 0; k < 6; ++k)` |
| store.global | 0.07 | 3.4% | 1.0 | 0.10 | `if (condval == (bool)0) \| if (condval == (bool)1)` |
| load.global#2 | 0.06 | 3.3% | 1.0 | 0.06 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < mask_len))` |
| load.global#3 | 0.06 | 3.3% | 1.0 | 0.06 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < mask_len))` |
| store.global#2 | 0.06 | 3.3% | 1.0 | 0.10 | `if (condval_1 == (bool)0) \| if (condval_1 == (bool)1)` |
| store.global#3 | 0.06 | 2.9% | 0.5 | 0.13 | `if ((((int)threadIdx.x) & 31) < 3) \| if ((((int)threadIdx.x) & 31) < 6)` |

### indexer_topk_merge_kernel grid (1, 5, 1) block (512, 1, 1)

15 launches, 80 warps; ranges, warp and last warp from the 5 recording CTAs' 80 warps; span 9.25 us, warp 8.98 us, nsys graphs 3.84 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 6.12 | 68.1% | 5.82 | 63.1% |
| unattributed | 2.20 | 24.5% | 2.49 | 27.0% |
| load | 0.58 | 6.5% | 0.76 | 8.3% |
| store | 0.08 | 0.9% | 0.15 | 1.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync#24*9 (est.: sampled iterations x9) | 0.67 | 7.5% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#20*9 (est.: sampled iterations x9) | 0.67 | 7.4% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#23*9 (est.: sampled iterations x9) | 0.67 | 7.4% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#19*9 (est.: sampled iterations x9) | 0.64 | 7.1% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#6*8 (est.: sampled iterations x8) | 0.60 | 6.7% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync#14*8 (est.: sampled iterations x8) | 0.60 | 6.6% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync#5*8 (est.: sampled iterations x8) | 0.58 | 6.4% | 1.0 | 0.77 | `__syncthreads();` |
| load.global | 0.57 | 6.4% | 1.0 | 0.70 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 8192);` |
| wait.sync#13*8 (est.: sampled iterations x8) | 0.54 | 6.1% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync | 0.14 | 1.5% | 1.0 | 0.22 | `__syncthreads();` |
| wait.sync#17 | 0.08 | 0.9% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#16 | 0.08 | 0.9% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#8 | 0.07 | 0.8% | 1.0 | 0.10 | `__syncthreads();` |
| store.global#2 | 0.07 | 0.8% | 0.9 | 0.10 | `Values[((((int64_t)((int)blockIdx.y)) * (int64_t)512) + ((int64_t)((int)threadIdx.x)))] = ` |
| wait.sync#25 | 0.07 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#9 | 0.07 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#3 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#2 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#15 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#7 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#12 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#11 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#18 | 0.06 | 0.7% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#10 | 0.06 | 0.7% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#4 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#21 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#22 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| load.global#2 | 0.01 | 0.1% | 0.1 | 0.10 | `if (((0 <= ((int*)sel_idx)[((int)threadIdx.x)]) && (((int*)sel_idx)[((int)threadIdx.x)] < ` |
| store.global | 0.01 | 0.1% | 0.1 | 0.10 | `Values[((((int64_t)((int)blockIdx.y)) * (int64_t)512) + ((int64_t)((int)threadIdx.x)))] = ` |

### rope_kernel grid (5, 1, 1) block (128, 1, 1)

90 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 1.36 us, warp 1.31 us, nsys graphs 1.47 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.76 | 58.0% | 1.02 | 66.6% |
| load | 0.51 | 39.0% | 0.47 | 30.8% |
| store | 0.04 | 3.0% | 0.04 | 2.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.51 | 39.0% | 1.0 | 0.51 | `if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < tokens)) \| int64_t position ` |
| store.global | 0.04 | 3.0% | 1.0 | 0.06 | `*(uint4*)(x + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + ((((int64_t)((int)thread` |

### norm_forward_and_per_token_cast_kernel_kernel grid (2, 1, 1) block (128, 1, 1)

48 launches, 8 warps; ranges, warp and last warp from the 2 recording CTAs' 8 warps; span 2.72 us, warp 2.72 us, nsys graphs 1.34 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 2.05 | 75.2% | 2.23 | 80.9% |
| load | 0.42 | 15.3% | 0.43 | 15.6% |
| wait | 0.14 | 5.0% | 0.05 | 1.9% |
| compute | 0.11 | 4.2% | 0.11 | 3.9% |
| unattributed | 0.01 | 0.3% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*4 (est.: sampled iterations x4) | 1.54 | 56.5% | 1.0 | 1.54 | `for (int j_2 = 0; j_2 < 2; ++j_2)` |
| store.global*4 (est.: sampled iterations x4) | 0.34 | 12.4% | 1.0 | 0.51 | `for (int j_1 = 0; j_1 < 4; ++j_1)` |
| load.global*4 (est.: sampled iterations x4) | 0.24 | 8.8% | 1.0 | 0.26 | `if (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 4)) < num_tokens)` |
| load.global#3*4 (est.: sampled iterations x4) | 0.14 | 5.3% | 1.0 | 0.26 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 128) + ((((int)threadIdx.x) & 15` |
| wait.sync | 0.14 | 5.0% | 1.0 | 0.29 | `__syncthreads();` |
| compute.reduce | 0.11 | 4.2% | 1.0 | 0.13 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| store.global#5*4 (est.: sampled iterations x4) | 0.10 | 3.5% | 0.6 | 0.26 | `*(uint4*)(out + ((((((int64_t)vi_1) * (int64_t)128) + ((((int64_t)((int)threadIdx.x)) & (i` |
| store.global#2 | 0.05 | 1.8% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| load.global#2 | 0.03 | 1.2% | 1.0 | 0.03 | `*(float2*)(row_partial + 0) = *(float2*)(sum_shared + ((((int)threadIdx.x) >> 6) * 2)); \| ` |
| store.global#3 | 0.03 | 1.1% | 0.6 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |

### token_ranks_kernel grid (5, 1, 1) block (256, 1, 1)

15 launches, 40 warps; ranges, warp and last warp from the 5 recording CTAs' 40 warps; span 7.78 us, warp 7.71 us, nsys graphs 8.29 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 6.95 | 90.1% | 7.05 | 90.4% |
| unattributed | 0.37 | 4.8% | 0.38 | 4.8% |
| compute | 0.34 | 4.4% | 0.32 | 4.1% |
| store | 0.05 | 0.6% | 0.05 | 0.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 6.95 | 90.1% | 1.0 | 7.17 | `if ((((int64_t)0 <= token_ids[((int64_t)((int)blockIdx.x))]) && (token_ids[((int64_t)((int` |
| compute.reduce | 0.34 | 4.4% | 1.0 | 0.64 | `total[0] = tl::AllReduce<tl::SumOp, 256, 1, 0, tl::NamedBarrier<256>>::run(total[0], (&(wo` |
| store.global | 0.05 | 0.6% | 1.0 | 0.06 | `output[((int64_t)((int)blockIdx.x))] = ((int64_t)total[0]);` |

### compress_gather_partial_state_kernel grid (2, 1, 1) block (128, 1, 1)

16 launches, 8 warps; ranges, warp and last warp from the 2 recording CTAs' 8 warps; span 7.89 us, warp 7.88 us, nsys graphs 3.97 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 4.00 | 50.8% | 4.17 | 58.4% |
| unattributed | 3.75 | 47.6% | 2.82 | 39.5% |
| store | 0.13 | 1.6% | 0.15 | 2.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 3.87 | 49.2% | 1.0 | 3.87 | `int nt = counts[0];` |
| load.global#2*2 (est.: sampled iterations x2) | 0.13 | 1.6% | 1.0 | 0.13 | `for (int i = 0; i < 4; ++i)` |
| store.global#2 | 0.06 | 0.8% | 1.0 | 0.08 | `if (((int)threadIdx.x) == 0)` |
| store.global*2 (est.: sampled iterations x2) | 0.06 | 0.8% | 1.0 | 0.06 | `if (half == 0)` |

### indexer_mxfp4_quantize_kernel grid (5, 1, 1) block (128, 1, 1)

30 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 3.78 us, warp 3.70 us, nsys graphs 2.91 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 2.31 | 62.4% | 2.41 | 63.9% |
| store | 1.12 | 30.2% | 1.12 | 29.7% |
| unattributed | 0.27 | 7.3% | 0.24 | 6.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2*16 (est.: sampled iterations x16) | 1.28 | 34.6% | 1.0 | 1.54 | `float v__2 = __fmul_rn(((float)X[(((((int64_t)((int)blockIdx.x)) * (int64_t)4096) + (((int` |
| store.global*16 (est.: sampled iterations x16) | 1.08 | 29.1% | 1.0 | 1.54 | `Pool[(((((slot & (int64_t)127) * (int64_t)64) + ((((int64_t)((int)threadIdx.x)) & (int64_t` |
| load.global | 1.03 | 27.8% | 1.0 | 1.09 | `for (int i = 0; i < 32; ++i)` |
| store.global#2 | 0.04 | 1.2% | 1.0 | 0.06 | `Pool[(((((slot & (int64_t)127) * (int64_t)4) + ((slot >> (int64_t)7) * ((int64_t)pool_stri` |

### gemm_splitk_reduce_cast_kernel grid (16, 2, 1) block (128, 1, 1)

96 launches, 128 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 1.02 us, warp 0.96 us, nsys graphs 1.09 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.46 | 47.8% | 0.51 | 48.6% |
| load | 0.35 | 36.5% | 0.39 | 36.6% |
| compute | 0.08 | 8.1% | 0.08 | 7.2% |
| store | 0.07 | 7.5% | 0.08 | 7.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.35 | 36.5% | 1.0 | 0.38 | `if ((1 <= M))` |
| compute.reduce | 0.08 | 8.1% | 1.0 | 0.10 | `amax[0] = tl::AllReduce<tl::MaxOp, 8, 1, 0, tl::NamedBarrier<128>>::run(amax[0]);` |
| store.global | 0.07 | 7.5% | 1.0 | 0.10 | `if (((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 5)) < M)` |

### indexer_mxfp4_quantize_kernel grid (1, 1, 1) block (128, 1, 1)

32 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 3.01 us, warp 1.02 us, nsys graphs 1.98 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.39 | 38.6% | 0.53 | 16.9% |
| load | 0.36 | 35.4% | 1.46 | 46.3% |
| store | 0.26 | 26.0% | 1.16 | 36.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2*16 (est.: sampled iterations x16) | 0.26 | 25.2% | 0.2 | 1.02 | `float v__2 = __fmul_rn(((float)X[(((((int64_t)((int)blockIdx.x)) * (int64_t)4096) + (((int` |
| store.global*16 (est.: sampled iterations x16) | 0.26 | 25.2% | 0.2 | 1.02 | `Pool[(((((slot & (int64_t)127) * (int64_t)64) + ((((int64_t)((int)threadIdx.x)) & (int64_t` |
| load.global | 0.10 | 10.2% | 0.2 | 0.42 | `for (int i = 0; i < 32; ++i)` |
| store.global#2 | 0.01 | 0.8% | 0.2 | 0.03 | `Pool[(((((slot & (int64_t)127) * (int64_t)4) + ((slot >> (int64_t)7) * ((int64_t)pool_stri` |

### prepare_dflash_inputs_kernel grid (9, 1, 1) block (256, 1, 1)

48 launches, 72 warps; ranges, warp and last warp from the 5 recording CTAs' 40 warps; span 2.02 us, warp 0.74 us, nsys graphs 2.50 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.26 | 34.8% | 1.43 | 69.4% |
| unattributed | 0.25 | 33.6% | 0.35 | 17.0% |
| store | 0.23 | 31.7% | 0.28 | 13.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.24 | 33.2% | 0.2 | 1.34 | `req = idx_mapping[((int)blockIdx.y)];` |
| store.global | 0.18 | 25.0% | 0.8 | 0.26 | `if (((int)blockIdx.y) == (num_reqs - 1))` |
| store.global#2 | 0.05 | 6.6% | 0.2 | 0.29 | `if (((int)blockIdx.y) == (num_reqs - 1))` |
| load.global#2 | 0.01 | 1.6% | 0.2 | 0.06 | `vllm_store_if_b32((&(out_query_start_loc[((int)blockIdx.y)])), (((int)blockIdx.y) * num_qu` |

### sparse_mla_kernel grid (5, 1, 6) block (128, 1, 1)

4 launches, 120 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 21.84 us, warp 19.78 us, nsys graphs 11.92 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed, with loops' unsampled iterations | 10.11 | 51.1% | 9.91 | 41.8% |
| wait | 5.26 | 26.6% | 0.65 | 2.7% |
| load | 2.98 | 15.1% | 11.63 | 49.1% |
| compute | 0.97 | 4.9% | 0.95 | 4.0% |
| store | 0.45 | 2.3% | 0.55 | 2.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync (sampled iterations) | 5.19 | 26.2% | 11.0 | 8.45 | `__syncthreads();` |
| load.global (sampled iterations) | 2.65 | 13.4% | 2.0 | 9.38 | `for (int c = 0; c < 4; ++c) \| for (int i = 0; i < 8; ++i) \| for (int i_1 = 0; i_1 < 8; ++i` |
| compute.mma (sampled iterations) | 0.74 | 3.7% | 2.0 | 0.77 | `for (int ki = 0; ki < 32; ++ki) \| for (int ki_1 = 0; ki_1 < 2; ++ki_1)` |
| store.global | 0.45 | 2.3% | 1.0 | 0.56 | `for (int i_10 = 0; i_10 < 32; ++i_10) \| for (int i_11 = 0; i_11 < 32; ++i_11) \| for (int i` |
| load.cp_async (sampled iterations) | 0.33 | 1.7% | 2.0 | 0.40 | `for (int k_1 = 0; k_1 < 5; ++k_1) \| for (int k_1 = 0; k_1 < 9; ++k_1) \| for (int k_3 = 0; ` |
| compute.reduce (sampled iterations) | 0.23 | 1.1% | 2.0 | 0.26 | `m_g[i_5] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(m_g[i_5], (&((` |
| wait.cp_async (sampled iterations) | 0.08 | 0.4% | 1.0 | 0.10 | `tl::cp_async_wait<0>();` |

### activation_swiglu_clamp_kernel grid (1, 5, 1) block (128, 1, 1)

90 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 0.93 us, warp 0.73 us, nsys graphs 1.89 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.63 | 85.6% | 0.89 | 85.6% |
| load | 0.07 | 9.8% | 0.10 | 9.9% |
| store | 0.03 | 4.6% | 0.05 | 4.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.07 | 9.8% | 0.8 | 0.10 | `*(uint4*)(gate + 0) = *(uint4*)(X + ((((int64_t)((int)blockIdx.y)) * (int64_t)1152) + (((i` |
| store.global | 0.03 | 4.6% | 0.8 | 0.06 | `*(uint4*)(Y + ((((int64_t)((int)blockIdx.y)) * (int64_t)576) + (((int64_t)((int)threadIdx.` |

### gemm_bf16_partials_kernel grid (1, 40, 1) block (256, 1, 1)

32 launches, 320 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 2.62 us, warp 1.68 us, nsys graphs 2.78 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.21 | 72.3% | 2.07 | 79.0% |
| wait | 0.24 | 14.3% | 0.23 | 8.9% |
| compute | 0.11 | 6.6% | 0.23 | 8.8% |
| load | 0.07 | 4.1% | 0.00 | 0.0% |
| store | 0.05 | 2.8% | 0.09 | 3.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*2 (est.: sampled iterations x2) | 0.11 | 6.6% | 0.5 | 0.26 | `{` |
| wait.sync | 0.11 | 6.4% | 1.0 | 0.13 | `__syncthreads();` |
| wait.mbarrier#2*2 (est.: sampled iterations x2) | 0.07 | 4.3% | 0.5 | 0.26 | `mbarrier[kb_1].wait(0);` |
| load.tma*2 (est.: sampled iterations x2) | 0.07 | 4.1% | 0.5 | 0.19 | `if (tl::tl_shuffle_elect<128>())` |
| wait.mbarrier*2 (est.: sampled iterations x2) | 0.06 | 3.6% | 0.5 | 0.13 | `mbarrier[(kb + 3)].wait(1);` |
| store.global | 0.05 | 2.8% | 0.5 | 0.13 | `for (int i_1 = 0; i_1 < 4; ++i_1) \| for (int i_2 = 0; i_2 < 8; ++i_2) \| for (int i_4 = 0; ` |

### sink_merge_kernel grid (1, 1, 1) block (128, 1, 1)

16 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 4.26 us, warp 4.26 us, nsys graphs 4.27 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.93 | 45.3% | 2.72 | 60.5% |
| wait | 0.88 | 20.6% | 0.45 | 10.0% |
| unattributed | 0.85 | 20.0% | 0.57 | 12.7% |
| compute | 0.58 | 13.5% | 0.62 | 13.9% |
| store | 0.02 | 0.6% | 0.13 | 2.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2*3 (est.: sampled iterations x3) | 1.66 | 38.9% | 1.0 | 1.73 | `for (int step = 0; step < ((width + 127) >> 7); ++step) \| for (int step = 0; step < ((widt` |
| compute.reduce*3 (est.: sampled iterations x3) | 0.58 | 13.5% | 1.0 | 0.58 | `float other = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| wait.sync#3*3 (est.: sampled iterations x3) | 0.53 | 12.4% | 1.0 | 0.67 | `__syncthreads();` |
| load.global | 0.22 | 5.3% | 1.0 | 0.22 | `last = sampled[(max((((int64_t)num_sampled[((int64_t)((int)blockIdx.x))]) - (int64_t)1), (` |
| wait.sync#2*3 (est.: sampled iterations x3) | 0.18 | 4.2% | 1.0 | 0.19 | `__syncthreads();` |
| wait.sync*3 (est.: sampled iterations x3) | 0.17 | 3.9% | 1.0 | 0.19 | `__syncthreads();` |
| load.global#3*3 (est.: sampled iterations x3) | 0.05 | 1.1% | 0.2 | 0.19 | `if ((best[0] == -CUDART_INF_F))` |
| store.global*3 (est.: sampled iterations x3) | 0.02 | 0.6% | 0.2 | 0.10 | `alternates[((((int64_t)((int)blockIdx.x)) * (int64_t)3) + ((int64_t)rank))] = condval_31;` |

### sink_candidates_kernel grid (1, 127, 1) block (128, 1, 1)

16 launches, 508 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 4.34 us, warp 3.55 us, nsys graphs 3.55 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.41 | 39.7% | 1.70 | 39.9% |
| compute | 0.91 | 25.7% | 0.93 | 21.7% |
| load | 0.54 | 15.1% | 0.89 | 20.8% |
| wait | 0.44 | 12.3% | 0.40 | 9.4% |
| store | 0.26 | 7.2% | 0.35 | 8.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.reduce*4 (est.: sampled iterations x4) | 0.66 | 18.5% | 1.0 | 0.77 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| load.global | 0.54 | 15.1% | 1.0 | 0.61 | `local[0] = max((num_sampled[((int64_t)((int)blockIdx.x))] - 1), 0);` |
| compute.reduce#2*4 (est.: sampled iterations x4) | 0.26 | 7.2% | 1.0 | 0.38 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 2, 32);` |
| store.global*4 (est.: sampled iterations x4) | 0.26 | 7.2% | 1.0 | 0.38 | `if (((int)threadIdx.x) == 0)` |
| wait.sync#2*4 (est.: sampled iterations x4) | 0.23 | 6.6% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync*4 (est.: sampled iterations x4) | 0.20 | 5.7% | 1.0 | 0.26 | `__syncthreads();` |

### gemm_bf16_partials_kernel grid (2, 4, 1) block (256, 1, 1)

32 launches, 64 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 1.98 us, warp 1.37 us, nsys graphs 2.11 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.90 | 66.1% | 1.54 | 71.6% |
| wait | 0.23 | 16.7% | 0.28 | 12.9% |
| compute | 0.12 | 8.4% | 0.24 | 11.2% |
| load | 0.07 | 5.1% | 0.00 | 0.0% |
| store | 0.05 | 3.7% | 0.09 | 4.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*2 (est.: sampled iterations x2) | 0.12 | 8.4% | 0.5 | 0.26 | `{` |
| wait.sync | 0.11 | 7.9% | 1.0 | 0.13 | `__syncthreads();` |
| load.tma*2 (est.: sampled iterations x2) | 0.07 | 5.1% | 0.5 | 0.19 | `if (tl::tl_shuffle_elect<128>())` |
| wait.mbarrier#2*2 (est.: sampled iterations x2) | 0.06 | 4.4% | 0.5 | 0.13 | `mbarrier[kb_1].wait(0);` |
| wait.mbarrier*2 (est.: sampled iterations x2) | 0.06 | 4.4% | 0.5 | 0.13 | `mbarrier[(kb + 3)].wait(1);` |
| store.global | 0.05 | 3.7% | 0.5 | 0.13 | `for (int i_1 = 0; i_1 < 4; ++i_1) \| for (int i_2 = 0; i_2 < 8; ++i_2) \| for (int i_4 = 0; ` |

### moe_combine_kernel grid (5, 10, 1) block (128, 1, 1)

48 launches, 200 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 1.41 us, warp 1.11 us, nsys graphs 1.55 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.64 | 57.9% | 0.93 | 66.5% |
| unattributed | 0.41 | 37.2% | 0.41 | 29.3% |
| store | 0.05 | 4.9% | 0.06 | 4.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.64 | 57.9% | 1.0 | 0.93 | `for (int k = 0; k < 3; ++k) \| for (int k = 0; k < 6; ++k) \| if ((((int)blockIdx.y) < share` |
| store.global | 0.05 | 4.9% | 1.0 | 0.06 | `{` |

### finish_argmax_kernel grid (2, 1, 1) block (32, 1, 1)

80 launches, 2 warps; ranges, warp and last warp from the 2 recording CTAs' 2 warps; span 0.74 us, warp 0.72 us, nsys graphs 0.90 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.37 | 51.1% | 0.46 | 57.4% |
| compute | 0.13 | 17.8% | 0.12 | 14.8% |
| load | 0.13 | 17.8% | 0.13 | 15.6% |
| store | 0.10 | 13.3% | 0.10 | 12.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.reduce | 0.13 | 17.8% | 1.0 | 0.13 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| load.global | 0.13 | 17.8% | 1.0 | 0.13 | `for (int k = 0; k < 2; ++k) \| if ((((int)threadIdx.x) < parts))` |
| store.global | 0.10 | 13.3% | 1.0 | 0.10 | `if (((int)threadIdx.x) == 0)` |

### rope_kernel grid (10, 1, 1) block (128, 1, 1)

48 launches, 40 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 1.28 us, warp 1.24 us, nsys graphs 1.50 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.73 | 58.5% | 0.80 | 60.3% |
| load | 0.48 | 38.4% | 0.48 | 36.6% |
| store | 0.04 | 3.1% | 0.04 | 3.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.48 | 38.4% | 1.0 | 0.48 | `if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < tokens)) \| int64_t position ` |
| store.global | 0.04 | 3.1% | 1.0 | 0.06 | `*(uint4*)(x + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + ((((int64_t)((int)thread` |

### indexer_score_kernel grid (128, 5, 1) block (128, 1, 1)

15 launches, 2560 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 4.16 us, warp 1.43 us, nsys graphs 3.49 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.76 | 53.2% | 2.42 | 60.4% |
| unattributed, with loops' unsampled iterations | 0.54 | 37.6% | 1.24 | 30.9% |
| wait | 0.09 | 6.2% | 0.18 | 4.6% |
| compute | 0.03 | 2.2% | 0.13 | 3.1% |
| store | 0.01 | 0.8% | 0.04 | 1.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 (sampled iterations) | 0.37 | 26.2% | 0.2 | 1.54 | `for (int i_1 = 0; i_1 < 32; ++i_1)` |
| load.global | 0.33 | 23.0% | 1.0 | 0.51 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 4096); \| i` |
| load.global#3 (sampled iterations) | 0.06 | 3.9% | 0.2 | 0.38 | `if (((int)threadIdx.x) < 64)` |
| wait.sync#2 (sampled iterations) | 0.05 | 3.5% | 0.2 | 0.35 | `__syncthreads();` |
| compute.mma (sampled iterations) | 0.03 | 2.2% | 0.2 | 0.13 | `for (int ki = 0; ki < 2; ++ki)` |
| wait.sync (sampled iterations) | 0.01 | 1.0% | 0.2 | 0.06 | `__syncthreads();` |
| store.global (sampled iterations) | 0.01 | 0.8% | 0.2 | 0.06 | `if ((((int)threadIdx.x) >> 6) == 0)` |
| wait.sync#3 (sampled iterations) | 0.01 | 0.8% | 0.2 | 0.06 | `__syncthreads();` |
| wait.sync#4 (sampled iterations) | 0.01 | 0.8% | 0.2 | 0.06 | `__syncthreads();` |

### per_token_cast_kernel_kernel grid (10, 1, 1) block (64, 1, 1)

48 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 10 warps; span 1.20 us, warp 1.08 us, nsys graphs 1.09 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.45 | 41.8% | 0.52 | 41.0% |
| load | 0.31 | 29.1% | 0.36 | 28.6% |
| store | 0.22 | 20.2% | 0.29 | 22.8% |
| compute | 0.10 | 8.9% | 0.10 | 7.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.22 | 20.8% | 1.0 | 0.29 | `for (int i = 0; i < 2; ++i)` |
| store.global#2 | 0.11 | 10.4% | 1.0 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.10 | 8.9% | 1.0 | 0.13 | `amax_fragment[i_1] = tl::AllReduce<tl::MaxOp, 2, 1, 0, tl::NamedBarrier<64>>::run(amax_fra` |
| load.global#2*2 (est.: sampled iterations x2) | 0.09 | 8.3% | 1.0 | 0.13 | `amax_fragment[i_1] = cutlass::bfloat16_t(__hmax((bfloat16_t(((nv_bfloat162*)(&(amax_fragme` |
| store.global*2 (est.: sampled iterations x2) | 0.05 | 5.1% | 0.6 | 0.13 | `out_sf[((((((int64_t)((int)blockIdx.x)) % (int64_t)15) * (int64_t)32) + (((int64_t)((int)t` |
| store.global#3*2 (est.: sampled iterations x2) | 0.05 | 4.8% | 0.6 | 0.13 | `*(fp8_e4_16_t*)(out + ((((((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int64_t)10240) +` |

### indexer_score_kernel grid (64, 5, 1) block (128, 1, 1)

15 launches, 1280 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 4.06 us, warp 1.92 us, nsys graphs 3.46 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.04 | 54.3% | 1.92 | 47.7% |
| unattributed, with loops' unsampled iterations | 0.68 | 35.2% | 1.40 | 34.7% |
| wait | 0.15 | 7.9% | 0.51 | 12.5% |
| compute | 0.04 | 2.1% | 0.15 | 3.8% |
| store | 0.01 | 0.5% | 0.05 | 1.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.75 | 38.9% | 1.0 | 0.99 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 4096); \| i` |
| load.global#2 (sampled iterations) | 0.26 | 13.7% | 0.2 | 1.25 | `for (int i_1 = 0; i_1 < 32; ++i_1)` |
| wait.sync#3 (sampled iterations) | 0.06 | 3.0% | 0.2 | 0.26 | `__syncthreads();` |
| wait.sync#2 (sampled iterations) | 0.06 | 2.9% | 0.2 | 0.29 | `__syncthreads();` |
| compute.mma (sampled iterations) | 0.04 | 2.1% | 0.2 | 0.16 | `for (int ki = 0; ki < 2; ++ki)` |
| load.global#3 (sampled iterations) | 0.03 | 1.7% | 0.2 | 0.29 | `if (((int)threadIdx.x) < 64)` |
| wait.sync (sampled iterations) | 0.02 | 1.1% | 0.2 | 0.10 | `__syncthreads();` |
| wait.sync#4 (sampled iterations) | 0.02 | 0.8% | 0.2 | 0.10 | `__syncthreads();` |
| store.global (sampled iterations) | 0.01 | 0.5% | 0.2 | 0.06 | `if ((((int)threadIdx.x) >> 6) == 0)` |

### adaptive_offsets_kernel grid (1, 1, 1) block (256, 1, 1)

15 launches, 8 warps; ranges, warp and last warp from the 1 recording CTAs' 8 warps; span 3.78 us, warp 3.75 us, nsys graphs 1.95 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 1.80 | 47.9% | 1.10 | 29.0% |
| unattributed, with loops' unsampled iterations | 1.45 | 38.6% | 1.51 | 39.9% |
| store | 0.30 | 8.1% | 0.54 | 14.1% |
| load | 0.20 | 5.3% | 0.64 | 16.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync (sampled iterations) | 0.43 | 11.4% | 1.0 | 0.48 | `__syncthreads();` |
| wait.sync#18 (sampled iterations) | 0.34 | 9.2% | 1.0 | 0.38 | `__syncthreads();` |
| store.global#5 | 0.13 | 3.5% | 1.0 | 0.16 | `for (int chunk_1 = 0; chunk_1 < ((starts_len + 255) >> 8); ++chunk_1)` |
| load.global#2 (sampled iterations) | 0.12 | 3.2% | 1.0 | 0.48 | `if ((((chunk * 256) + ((int)threadIdx.x)) < num_reqs))` |
| store.global#2 (sampled iterations) | 0.08 | 2.1% | 1.0 | 0.10 | `if ((((chunk * 256) + ((int)threadIdx.x)) < num_reqs) & (drafted == 0))` |
| load.global (sampled iterations) | 0.07 | 1.9% | 1.0 | 0.10 | `if (((((chunk * 256) + ((int)threadIdx.x)) < num_reqs) & (drafted != 0)))` |
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

### activation_swiglu_clamp_kernel grid (1, 10, 1) block (128, 1, 1)

48 launches, 40 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 0.93 us, warp 0.72 us, nsys graphs 1.82 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.61 | 85.6% | 0.90 | 82.0% |
| load | 0.07 | 9.7% | 0.14 | 12.7% |
| store | 0.03 | 4.7% | 0.06 | 5.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.07 | 9.7% | 0.8 | 0.10 | `*(uint4*)(gate + 0) = *(uint4*)(X + ((((int64_t)((int)blockIdx.y)) * (int64_t)1152) + (((i` |
| store.global | 0.03 | 4.7% | 0.8 | 0.06 | `*(uint4*)(Y + ((((int64_t)((int)blockIdx.y)) * (int64_t)576) + (((int64_t)((int)threadIdx.` |

### per_token_cast_kernel_kernel grid (20, 1, 1) block (64, 1, 1)

48 launches, 40 warps; ranges, warp and last warp from the 4 recording CTAs' 8 warps; span 1.09 us, warp 1.04 us, nsys graphs 0.99 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.35 | 33.8% | 0.39 | 35.5% |
| store | 0.30 | 28.5% | 0.31 | 28.5% |
| unattributed | 0.30 | 28.5% | 0.29 | 26.8% |
| compute | 0.10 | 9.2% | 0.10 | 9.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.26 | 25.4% | 1.0 | 0.29 | `for (int i = 0; i < 2; ++i)` |
| store.global#2 | 0.11 | 10.8% | 1.0 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.10 | 9.2% | 1.0 | 0.13 | `amax_fragment[i_1] = tl::AllReduce<tl::MaxOp, 2, 1, 0, tl::NamedBarrier<64>>::run(amax_fra` |
| store.global*2 (est.: sampled iterations x2) | 0.10 | 9.2% | 1.0 | 0.13 | `out_sf[((((((int64_t)((int)blockIdx.x)) % (int64_t)15) * (int64_t)32) + (((int64_t)((int)t` |
| load.global#2*2 (est.: sampled iterations x2) | 0.09 | 8.5% | 1.0 | 0.13 | `amax_fragment[i_1] = cutlass::bfloat16_t(__hmax((bfloat16_t(((nv_bfloat162*)(&(amax_fragme` |
| store.global#3*2 (est.: sampled iterations x2) | 0.09 | 8.5% | 1.0 | 0.13 | `*(fp8_e4_16_t*)(out + ((((((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int64_t)10240) +` |

### gemm_splitk_reduce_cast_kernel grid (16, 3, 1) block (128, 1, 1)

48 launches, 192 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 1.06 us, warp 1.01 us, nsys graphs 1.12 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.46 | 45.5% | 0.47 | 44.8% |
| load | 0.40 | 39.4% | 0.42 | 40.1% |
| compute | 0.08 | 7.6% | 0.07 | 6.8% |
| store | 0.08 | 7.6% | 0.09 | 8.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.40 | 39.4% | 1.0 | 0.45 | `if ((1 <= M))` |
| compute.reduce | 0.08 | 7.6% | 1.0 | 0.10 | `amax[0] = tl::AllReduce<tl::MaxOp, 8, 1, 0, tl::NamedBarrier<128>>::run(amax[0]);` |
| store.global | 0.08 | 7.6% | 1.0 | 0.10 | `if (((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 5)) < M)` |

### gemm_splitk_reduce_kernel grid (1, 5, 1) block (128, 1, 1)

60 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 0.83 us, warp 0.82 us, nsys graphs 0.99 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.46 | 55.7% | 0.45 | 54.8% |
| load | 0.32 | 38.5% | 0.32 | 38.6% |
| store | 0.05 | 5.8% | 0.05 | 6.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.32 | 38.5% | 1.0 | 0.34 | `acc[0] = P[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x))` |
| store.global | 0.05 | 5.8% | 1.0 | 0.06 | `C[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x)) * (int64` |

### cache_writer_kernel grid (8, 1, 1) block (128, 1, 1)

48 launches, 32 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 1.02 us, warp 0.94 us, nsys graphs 1.12 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.32 | 34.5% | 0.33 | 32.6% |
| load | 0.29 | 31.1% | 0.29 | 29.1% |
| store | 0.25 | 26.4% | 0.31 | 30.7% |
| compute | 0.07 | 7.9% | 0.08 | 7.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.25 | 26.4% | 1.0 | 0.32 | `if (((int64_t)0 <= slot) & ((slot >> (int64_t)7) < ((int64_t)pages))) \| if (((int64_t)0 <=` |
| load.global | 0.23 | 24.7% | 1.0 | 0.26 | `for (int i = 0; i < 4; ++i)` |
| compute.reduce | 0.07 | 7.9% | 1.0 | 0.10 | `amax = max(amax, __shfl_xor_sync((uint)4294967295, amax, 1, 32));` |
| load.global#2 | 0.06 | 6.4% | 1.0 | 0.06 | `int64_t slot = slots[((int64_t)((int)blockIdx.x))];` |

### verify_kernel grid (1, 1, 1) block (96, 1, 1)

16 launches, 3 warps; ranges, warp and last warp from the 1 recording CTAs' 3 warps; span 2.85 us, warp 2.71 us, nsys graphs 2.30 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.09 | 40.2% | 1.09 | 37.8% |
| unattributed | 0.98 | 36.2% | 0.98 | 33.8% |
| compute | 0.29 | 10.8% | 0.29 | 10.1% |
| wait | 0.19 | 6.9% | 0.23 | 7.8% |
| store | 0.16 | 5.9% | 0.30 | 10.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.02 | 37.8% | 1.0 | 1.02 | `start = cu_num_logits[((int64_t)((int)blockIdx.x))];` |
| compute.reduce | 0.29 | 10.8% | 1.0 | 0.30 | `row_max[0] = vllm_max(row_max[0], __shfl_xor_sync((uint)4294967295, row_max[0], 8, 32));` |
| store.global | 0.16 | 5.9% | 1.0 | 0.29 | `if (((int)threadIdx.x) == 0)` |
| wait.sync#3 | 0.09 | 3.1% | 1.0 | 0.13 | `__syncthreads();` |
| load.global#2 | 0.06 | 2.4% | 1.0 | 0.06 | `target = local_argmax[((((int64_t)row) * argmax_stride) + ((int64_t)best_index[0]))];` |
| wait.sync#2 | 0.06 | 2.2% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync | 0.04 | 1.6% | 1.0 | 0.06 | `__syncthreads();` |

### resample_kernel grid (1, 127, 1) block (128, 1, 1)

16 launches, 508 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 2.99 us, warp 2.24 us, nsys graphs 2.21 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.91 | 85.0% | 2.18 | 77.6% |
| unattributed | 0.29 | 13.0% | 0.58 | 20.8% |
| compute | 0.04 | 2.0% | 0.04 | 1.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.91 | 85.0% | 1.0 | 2.61 | `end = cu_num_logits[(((int64_t)((int)blockIdx.x)) + (int64_t)1)];` |
| compute.reduce | 0.04 | 2.0% | 1.0 | 0.06 | `if ((temp != 0x0p+0f/*0.000000e+00*/) \| bonus)` |

### compute_slot_mappings_kernel grid (1, 11, 1) block (256, 1, 1)

16 launches, 88 warps; ranges, warp and last warp from the 6 recording CTAs' 48 warps; span 2.70 us, warp 1.47 us, nsys graphs 2.85 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.80 | 54.7% | 0.80 | 30.1% |
| unattributed | 0.66 | 45.3% | 1.87 | 69.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.80 | 54.7% | 1.0 | 0.83 | `int first_token = query_start_loc[0];` |

### sink_cut_kernel grid (1, 1, 1) block (128, 1, 1)

32 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 1.28 us, warp 0.54 us, nsys graphs 2.11 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.27 | 50.7% | 0.35 | 26.0% |
| store | 0.14 | 26.9% | 0.49 | 36.9% |
| load | 0.12 | 22.4% | 0.50 | 37.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.14 | 26.9% | 0.6 | 0.48 | `if (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) < num_cut) \| saved[((((int64_t)((int)` |
| load.global | 0.12 | 22.4% | 0.1 | 0.48 | `for (int b = 0; b < num_reqs; ++b)` |

### sink_policy_kernel grid (1, 1, 1) block (32, 1, 1)

16 launches, 1 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 2.53 us, warp 2.53 us, nsys graphs 2.66 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.09 | 43.0% | 1.12 | 42.9% |
| unattributed | 0.86 | 34.2% | 0.90 | 34.4% |
| store | 0.58 | 22.8% | 0.59 | 22.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 | 0.64 | 25.3% | 1.0 | 0.64 | `state[0] = idx_mapping[((int64_t)((int)blockIdx.x))];` |
| store.global#3 | 0.51 | 20.3% | 1.0 | 0.51 | `for (int a = 0; a < 3; ++a) \| recovered_logprobs[((int64_t)((int)blockIdx.x))] = 0x0p+0f/*` |
| load.global | 0.22 | 8.9% | 1.0 | 0.22 | `start[0] = cu_num_logits[((int64_t)((int)blockIdx.x))];` |
| load.global#4 | 0.16 | 6.3% | 1.0 | 0.16 | `for (int r = 0; r < 2; ++r) \| for (int r_2 = 0; r_2 < 8; ++r_2)` |
| load.global#5 | 0.06 | 2.5% | 1.0 | 0.06 | `AtomicAdd((&(counts[0])), draft_zero_flag[0]);` |
| store.global#4 | 0.06 | 2.5% | 1.0 | 0.06 | `events[(((int64_t)((int)blockIdx.x)) * (int64_t)6)] = outcome[0];` |

### norm_forward_and_per_token_cast_kernel_kernel grid (8, 1, 1) block (128, 1, 1)

16 launches, 32 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 2.37 us, warp 2.31 us, nsys graphs 1.79 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 2.21 | 95.5% | 3.30 | 126.9% |
| load | 0.45 | 19.4% | 0.44 | 16.9% |
| compute | 0.13 | 5.6% | 0.12 | 4.8% |
| wait | 0.06 | 2.5% | 0.06 | 2.4% |
| unattributed | 0.00 | 0.0% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*5 (est.: sampled iterations x5) | 1.94 | 83.8% | 1.0 | 2.24 | `for (int j_1 = 0; j_1 < 2; ++j_1)` |
| load.global*5 (est.: sampled iterations x5) | 0.23 | 9.9% | 1.0 | 0.32 | `*(uint4*)(x_local + (vi * 8)) = *(uint4*)(x + ((((((int64_t)vi) * (int64_t)1024) + (((int6` |
| load.global#2*5 (est.: sampled iterations x5) | 0.22 | 9.5% | 1.0 | 0.32 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 1024) + (((int)threadIdx.x) * 8)` |
| store.global*5 (est.: sampled iterations x5) | 0.21 | 9.3% | 1.0 | 0.32 | `for (int j = 0; j < 4; ++j)` |
| compute.reduce | 0.13 | 5.6% | 1.0 | 0.16 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| wait.sync | 0.06 | 2.5% | 1.0 | 0.10 | `__syncthreads();` |
| store.global#2 | 0.04 | 1.8% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| store.global#3 | 0.01 | 0.5% | 0.2 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |

### metadata_pages_kernel grid (5, 1, 1) block (128, 1, 1)

30 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 1.34 us, warp 0.96 us, nsys graphs 1.50 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.73 | 75.8% | 0.80 | 59.4% |
| load | 0.14 | 15.0% | 0.19 | 14.2% |
| store | 0.09 | 9.2% | 0.35 | 26.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.13 | 13.3% | 1.0 | 0.13 | `if ((((offset + ((int)blockIdx.x)) < src_rows) && (0 <= (offset + ((int)blockIdx.x)))))` |
| store.global | 0.09 | 9.2% | 0.2 | 0.35 | `if (((int)threadIdx.x) < 32)` |
| load.global#2 | 0.02 | 1.7% | 0.2 | 0.06 | `if (((req < table_rows) && (((int)threadIdx.x) < table_cols)))` |

### cache_writer_kernel grid (6, 1, 1) block (128, 1, 1)

8 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 5.57 us, warp 5.51 us, nsys graphs 3.07 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 3.20 | 58.0% | 2.22 | 47.2% |
| store | 1.92 | 34.8% | 1.97 | 41.9% |
| load | 0.32 | 5.7% | 0.40 | 8.5% |
| compute | 0.08 | 1.4% | 0.12 | 2.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 1.92 | 34.8% | 1.0 | 1.98 | `if (((int64_t)0 <= slot) & ((slot >> (int64_t)7) < ((int64_t)pages))) \| if (((int64_t)0 <=` |
| load.global | 0.26 | 4.7% | 1.0 | 0.37 | `for (int i = 0; i < 4; ++i)` |
| compute.reduce | 0.08 | 1.4% | 1.0 | 0.10 | `amax = max(amax, __shfl_xor_sync((uint)4294967295, amax, 1, 32));` |
| load.global#2 | 0.06 | 1.1% | 1.0 | 0.06 | `int64_t slot = slots[((int64_t)((int)blockIdx.x))];` |

### norm_forward_and_per_token_cast_kernel_kernel grid (10, 1, 1) block (128, 1, 1)

16 launches, 40 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 2.30 us, warp 2.27 us, nsys graphs 1.82 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 2.09 | 92.3% | 2.10 | 91.1% |
| load | 0.43 | 18.9% | 0.42 | 18.2% |
| compute | 0.13 | 5.8% | 0.13 | 5.6% |
| wait | 0.06 | 2.7% | 0.06 | 2.5% |
| unattributed | 0.00 | 0.0% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*5 (est.: sampled iterations x5) | 1.83 | 80.9% | 1.0 | 2.08 | `for (int j_1 = 0; j_1 < 2; ++j_1)` |
| load.global#2*5 (est.: sampled iterations x5) | 0.22 | 9.5% | 1.0 | 0.32 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 1024) + (((int)threadIdx.x) * 8)` |
| load.global*5 (est.: sampled iterations x5) | 0.21 | 9.4% | 1.0 | 0.32 | `*(uint4*)(x_local + (vi * 8)) = *(uint4*)(x + ((((((int64_t)vi) * (int64_t)1024) + (((int6` |
| store.global*5 (est.: sampled iterations x5) | 0.20 | 9.0% | 1.0 | 0.32 | `for (int j = 0; j < 4; ++j)` |
| compute.reduce | 0.13 | 5.8% | 1.0 | 0.16 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| wait.sync | 0.06 | 2.7% | 1.0 | 0.10 | `__syncthreads();` |
| store.global#2 | 0.04 | 1.9% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| store.global#3 | 0.01 | 0.6% | 0.2 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |

### sparse_mla_kernel grid (1, 1, 6) block (128, 1, 1)

2 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 18.30 us, warp 18.29 us, nsys graphs 14.54 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed, with loops' unsampled iterations | 5.02 | 27.4% | 4.99 | 27.3% |
| compute | 4.61 | 25.2% | 4.61 | 25.2% |
| load | 4.11 | 22.5% | 5.54 | 30.3% |
| wait | 3.87 | 21.1% | 2.45 | 13.4% |
| store | 0.69 | 3.8% | 0.70 | 3.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 | 2.21 | 12.1% | 1.0 | 2.43 | `for (int i = 0; i < 8; ++i)` |
| compute.mma*4 (est.: sampled iterations x4) | 1.72 | 9.4% | 1.0 | 1.73 | `for (int ki = 0; ki < 32; ++ki) \| {` |
| wait.sync (sampled iterations) | 1.35 | 7.4% | 1.0 | 1.95 | `__syncthreads();` |
| compute.mma#2*4 (est.: sampled iterations x4) | 1.20 | 6.6% | 1.0 | 1.22 | `for (int ki_1 = 0; ki_1 < 2; ++ki_1) \| if (((((int)blockIdx.x) * 2) + (c_1 >> 1)) < 3)` |
| compute.reduce#2*8 (est.: sampled iterations x8) | 0.87 | 4.8% | 1.0 | 1.02 | `row_sum[i_7] = tl::AllReduce<tl::SumOp, 128, 32, 0, tl::NamedBarrier<128>>::run(row_sum[i_` |
| load.global (sampled iterations) | 0.86 | 4.7% | 1.0 | 2.34 | `for (int c = 0; c < 4; ++c) \| for (int i = 0; i < 8; ++i) \| for (int i_1 = 0; i_1 < 8; ++i` |
| compute.reduce*8 (est.: sampled iterations x8) | 0.82 | 4.5% | 1.0 | 1.02 | `m_g[i_4] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(m_g[i_4], (&((` |
| store.global | 0.69 | 3.8% | 1.0 | 0.72 | `for (int i_10 = 0; i_10 < 32; ++i_10) \| for (int i_11 = 0; i_11 < 32; ++i_11) \| for (int i` |
| load.cp_async#2*4 (est.: sampled iterations x4) | 0.68 | 3.7% | 1.0 | 0.77 | `if (((((int)blockIdx.x) * 2) + (c_1 >> 1)) < 3) \| if ((c_1 < 3) & (c_1 < 3)) \| if (c < 3)` |
| load.cp_async (sampled iterations) | 0.36 | 1.9% | 1.0 | 0.38 | `for (int k_1 = 0; k_1 < 5; ++k_1) \| for (int k_1 = 0; k_1 < 9; ++k_1) \| for (int k_3 = 0; ` |
| wait.sync#8*8 (est.: sampled iterations x8) | 0.33 | 1.8% | 1.0 | 0.51 | `__syncthreads();` |
| wait.cp_async*4 (est.: sampled iterations x4) | 0.31 | 1.7% | 1.0 | 0.38 | `tl::cp_async_wait<0>();` |
| wait.sync#10*8 (est.: sampled iterations x8) | 0.28 | 1.5% | 1.0 | 0.38 | `__syncthreads();` |
| wait.sync#5*4 (est.: sampled iterations x4) | 0.27 | 1.5% | 1.0 | 0.38 | `__syncthreads();` |
| wait.sync#7*4 (est.: sampled iterations x4) | 0.26 | 1.4% | 1.0 | 0.32 | `__syncthreads();` |
| wait.sync#11*4 (est.: sampled iterations x4) | 0.26 | 1.4% | 1.0 | 0.32 | `__syncthreads();` |
| wait.sync#4*4 (est.: sampled iterations x4) | 0.25 | 1.4% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#9*4 (est.: sampled iterations x4) | 0.24 | 1.3% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#6*4 (est.: sampled iterations x4) | 0.21 | 1.1% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#3 | 0.07 | 0.4% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#2 | 0.05 | 0.3% | 1.0 | 0.05 | `__syncthreads();` |

### sink_flags_kernel grid (1, 1, 1) block (128, 1, 1)

16 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 2.21 us, warp 0.72 us, nsys graphs 3.07 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.39 | 54.1% | 1.56 | 70.1% |
| unattributed | 0.24 | 33.7% | 0.32 | 14.3% |
| store | 0.09 | 12.2% | 0.35 | 15.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.39 | 54.1% | 0.2 | 1.57 | `token = logits_indices[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)th` |
| store.global | 0.09 | 12.2% | 0.2 | 0.35 | `flags[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = (` |

### gather_block_tables_kernel grid (11, 1, 1) block (256, 1, 1)

16 launches, 88 warps; ranges, warp and last warp from the 6 recording CTAs' 48 warps; span 1.92 us, warp 1.35 us, nsys graphs 1.98 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.06 | 78.4% | 1.63 | 84.8% |
| unattributed | 0.29 | 21.6% | 0.29 | 15.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.06 | 78.4% | 1.0 | 1.63 | `int g = chunk_groups[((int64_t)((int)blockIdx.x))];` |

### rope_kernel grid (5, 1, 1) block (256, 1, 1)

30 launches, 40 warps; ranges, warp and last warp from the 5 recording CTAs' 40 warps; span 1.09 us, warp 0.95 us, nsys graphs 1.31 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.64 | 67.7% | 0.71 | 69.1% |
| load | 0.27 | 28.2% | 0.28 | 26.9% |
| store | 0.04 | 4.1% | 0.04 | 4.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.27 | 28.2% | 1.0 | 0.29 | `if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < tokens)) \| int64_t position ` |
| store.global | 0.04 | 4.1% | 1.0 | 0.06 | `*(uint4*)(x + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + ((((int64_t)((int)thread` |

### sink_rows_kernel grid (1, 64, 1) block (256, 1, 1)

16 launches, 512 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 1.70 us, warp 0.96 us, nsys graphs 1.90 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.37 | 38.5% | 0.45 | 24.2% |
| load | 0.31 | 32.6% | 0.66 | 35.6% |
| store | 0.28 | 29.0% | 0.75 | 40.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.31 | 32.1% | 1.0 | 0.38 | `row = (max((((int64_t)num_sampled[((int64_t)((int)blockIdx.x))]) - (int64_t)1), (int64_t)0` |
| store.global | 0.26 | 27.3% | 1.0 | 0.48 | `for (int k_1 = 0; k_1 < 8; ++k_1)` |
| store.global#2 | 0.02 | 1.7% | 0.0 | 0.51 | `tokens[((((int64_t)((int)blockIdx.x)) * (int64_t)3) + ((int64_t)((int)threadIdx.x)))] = to` |
| load.global#2 | 0.00 | 0.5% | 0.0 | 0.14 | `token = max(alternates[((((int64_t)((int)blockIdx.x)) * (int64_t)3) + ((int64_t)((int)thre` |

### post_update_kernel grid (1, 1, 1) block (32, 1, 1)

16 launches, 1 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 1.82 us, warp 1.82 us, nsys graphs 2.14 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.67 | 36.8% | 0.66 | 36.6% |
| unattributed | 0.58 | 31.6% | 0.57 | 31.5% |
| store | 0.58 | 31.6% | 0.58 | 31.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 | 0.48 | 26.3% | 1.0 | 0.48 | `int count = num_sampled[((((int64_t)((int)blockIdx.x)) * (int64_t)32) + ((int64_t)((int)th` |
| store.global#2 | 0.42 | 22.8% | 1.0 | 0.42 | `if (0 < count)` |
| store.global | 0.16 | 8.8% | 1.0 | 0.16 | `if (0 < count)` |
| load.global | 0.10 | 5.3% | 1.0 | 0.10 | `int64_t req = idx_mapping[((((int64_t)((int)blockIdx.x)) * (int64_t)32) + ((int64_t)((int)` |
| load.global#3 | 0.10 | 5.3% | 1.0 | 0.10 | `if ((0 < count))` |

### dspark_confidence_kernel grid (10, 1, 1) block (128, 1, 1)

16 launches, 40 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 1.81 us, warp 1.59 us, nsys graphs 1.95 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.23 | 77.5% | 1.35 | 74.6% |
| load | 0.17 | 10.8% | 0.20 | 11.3% |
| compute | 0.10 | 6.1% | 0.10 | 5.3% |
| wait | 0.07 | 4.2% | 0.08 | 4.2% |
| store | 0.02 | 1.4% | 0.08 | 4.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.16 | 10.1% | 1.0 | 0.16 | `for (int c = 0; c < 5; ++c)` |
| compute.reduce | 0.10 | 6.1% | 1.0 | 0.10 | `acc[0] = tl::warp_reduce_sum(acc[0]);` |
| wait.sync | 0.07 | 4.2% | 1.0 | 0.13 | `__syncthreads();` |
| store.global | 0.02 | 1.4% | 0.2 | 0.10 | `out[((int64_t)((int)blockIdx.x))] = (0x1p+0f/*1.000000e+00*/ / (0x1p+0f/*1.000000e+00*/ + ` |
| load.global#2 | 0.01 | 0.7% | 0.2 | 0.06 | `total[0] = 0x0p+0f/*0.000000e+00*/;` |

### insert_kernel grid (1, 1, 1) block (32, 1, 1)

16 launches, 1 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 1.57 us, warp 1.57 us, nsys graphs 1.38 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.70 | 44.9% | 0.78 | 44.6% |
| unattributed | 0.42 | 26.5% | 0.42 | 23.9% |
| store | 0.42 | 26.5% | 0.51 | 29.1% |
| compute | 0.03 | 2.0% | 0.04 | 2.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#3 | 0.51 | 32.7% | 1.0 | 0.51 | `int state = mapping[((int64_t)position)]; \| int64_t state = mapping[((int64_t)position)];` |
| store.global | 0.42 | 26.5% | 1.0 | 0.42 | `counts[((int64_t)((int)blockIdx.x))] = (count_local + 1); \| counts[((int64_t)((int)blockId` |
| load.global | 0.13 | 8.2% | 1.0 | 0.13 | `for (int k = 0; k < 4; ++k)` |
| load.global#2 | 0.06 | 4.1% | 1.0 | 0.06 | `count_local = counts[((int64_t)((int)blockIdx.x))];` |
| compute.reduce | 0.03 | 2.0% | 1.0 | 0.03 | `int count = __shfl_sync((uint)4294967295, count_local, 0, 32);` |

### topk_ids_kernel grid (6, 1, 1) block (512, 1, 1)

1 launches, 96 warps; ranges, warp and last warp from the 6 recording CTAs' 96 warps; span 27.07 us, warp 24.49 us, nsys graphs 24.64 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 20.86 | 85.2% | 21.09 | 77.9% |
| unattributed | 1.93 | 7.9% | 1.95 | 7.2% |
| wait | 1.35 | 5.5% | 2.46 | 9.1% |
| store | 0.35 | 1.4% | 1.57 | 5.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 20.86 | 85.2% | 1.0 | 23.42 | `for (int t = 0; t < (vocab >> 14); ++t)` |
| wait.sync | 1.21 | 4.9% | 1.0 | 4.10 | `__syncthreads();` |
| store.global | 0.35 | 1.4% | 1.0 | 4.45 | `if ((((int)threadIdx.x) >> 5) == 0)` |
| wait.sync#2 | 0.07 | 0.3% | 1.0 | 0.19 | `__syncthreads();` |
| wait.sync#3 | 0.06 | 0.3% | 1.0 | 0.06 | `__syncthreads();` |

### context_inputs_kernel grid (5, 8, 1) block (128, 1, 1)

16 launches, 160 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 1.57 us, warp 1.27 us, nsys graphs 2.00 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.85 | 66.8% | 1.25 | 75.1% |
| unattributed | 0.34 | 27.1% | 0.35 | 20.8% |
| store | 0.08 | 6.1% | 0.07 | 4.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.85 | 66.8% | 1.0 | 1.15 | `vllm_dspark_context_b128((&(aux[(((((int64_t)((int)blockIdx.y)) * (int64_t)15360) + (((int` |
| store.global | 0.08 | 6.1% | 1.0 | 0.16 | `if (((((int)blockIdx.x) == 0) & (((int)threadIdx.x) == 0)) & (num_tokens <= ((int)blockIdx` |

### gemm_splitk_reduce_kernel grid (4, 8, 1) block (128, 1, 1)

48 launches, 128 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 0.54 us, warp 0.53 us, nsys graphs 0.74 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.26 | 49.2% | 0.26 | 48.8% |
| load | 0.23 | 43.2% | 0.23 | 43.2% |
| store | 0.04 | 7.6% | 0.04 | 8.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.23 | 43.2% | 1.0 | 0.26 | `acc[0] = P[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x))` |
| store.global | 0.04 | 7.6% | 1.0 | 0.06 | `C[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x)) * (int64` |

### assign_draft_budget_kernel grid (1, 1, 1) block (256, 1, 1)

15 launches, 8 warps; ranges, warp and last warp from the 1 recording CTAs' 8 warps; span 1.76 us, warp 1.74 us, nsys graphs 1.86 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 0.78 | 44.6% | 0.48 | 27.5% |
| unattributed, with loops' unsampled iterations | 0.54 | 31.0% | 0.61 | 34.8% |
| load | 0.23 | 13.3% | 0.47 | 26.9% |
| store | 0.19 | 11.0% | 0.19 | 10.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync | 0.54 | 31.0% | 1.0 | 0.61 | `__syncthreads();` |
| load.global | 0.23 | 13.3% | 1.0 | 0.74 | `for (int chunk = 0; chunk < ((num_reqs + 255) >> 8); ++chunk)` |
| store.global | 0.19 | 11.0% | 1.0 | 0.19 | `for (int chunk_2 = 0; chunk_2 < ((num_reqs + 255) >> 8); ++chunk_2)` |
| wait.sync#3 | 0.17 | 9.9% | 1.0 | 0.19 | `__syncthreads();` |
| wait.sync#2 (sampled iterations) | 0.06 | 3.7% | 1.0 | 0.06 | `__syncthreads();` |

### logits_rows_kernel grid (1, 1, 1) block (128, 1, 1)

16 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 1.15 us, warp 1.11 us, nsys graphs 2.05 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.50 | 45.0% | 0.53 | 37.1% |
| unattributed | 0.47 | 42.1% | 0.29 | 20.5% |
| store | 0.14 | 12.9% | 0.60 | 42.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.50 | 45.0% | 1.0 | 0.51 | `index = logits_indices[min(((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((in` |
| store.global | 0.14 | 12.9% | 1.0 | 0.38 | `if (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) < num_logits)` |

### combine_sampled_and_draft_tokens_kernel grid (1, 1, 1) block (32, 1, 1)

16 launches, 1 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 1.39 us, warp 1.39 us, nsys graphs 1.86 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.90 | 64.4% | 0.90 | 64.6% |
| unattributed | 0.40 | 28.7% | 0.41 | 29.1% |
| store | 0.10 | 6.9% | 0.09 | 6.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.90 | 64.4% | 1.0 | 0.90 | `if (((((int)blockIdx.x) * 32) + ((int)threadIdx.x)) < num_reqs) \| req = idx_mapping[((int6` |
| store.global | 0.10 | 6.9% | 1.0 | 0.10 | `if ((prefill < seq) && (1 < num_logits))` |

### metadata_decode_kernel grid (6, 2, 1) block (128, 1, 1)

4 launches, 48 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 5.14 us, warp 4.09 us, nsys graphs 1.79 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 2.13 | 52.1% | 3.54 | 64.0% |
| unattributed | 1.86 | 45.3% | 1.84 | 33.3% |
| store | 0.11 | 2.6% | 0.15 | 2.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.49 | 36.5% | 0.5 | 3.23 | `if ((((offset + ((int)blockIdx.x)) < src_rows) && (0 <= (offset + ((int)blockIdx.x)))))` |
| load.global#3 | 0.61 | 15.0% | 0.5 | 1.26 | `if ((((offset + ((int)blockIdx.x)) < main_src_rows) && (0 <= (offset + ((int)blockIdx.x)))` |
| store.global#4 | 0.06 | 1.5% | 0.5 | 0.29 | `if (0 < live)` |
| store.global | 0.03 | 0.8% | 0.5 | 0.06 | `swa[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = con` |
| load.global#2 | 0.02 | 0.6% | 0.1 | 0.19 | `if (((0 <= req) & ((int64_t)0 <= pos)))` |
| store.global#2 | 0.01 | 0.2% | 0.1 | 0.06 | `lengths[((int64_t)((int)blockIdx.x))] = ((int)max(info[1], (int64_t)0));` |
| store.global#3 | 0.01 | 0.1% | 0.1 | 0.06 | `top_lengths[((int64_t)((int)blockIdx.x))] = condval_6; \| top_lengths[((int64_t)((int)block` |

### per_token_cast_kernel_kernel grid (60, 1, 1) block (64, 1, 1)

16 launches, 120 warps; ranges, warp and last warp from the 4 recording CTAs' 8 warps; span 1.18 us, warp 1.11 us, nsys graphs 1.28 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.39 | 35.4% | 0.42 | 30.9% |
| unattributed | 0.32 | 28.9% | 0.52 | 38.4% |
| store | 0.30 | 26.7% | 0.31 | 23.0% |
| compute | 0.10 | 9.0% | 0.10 | 7.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.30 | 27.4% | 1.0 | 0.35 | `for (int i = 0; i < 2; ++i)` |
| store.global#2 | 0.11 | 10.1% | 1.0 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.10 | 9.0% | 1.0 | 0.13 | `amax_fragment[i_1] = tl::AllReduce<tl::MaxOp, 2, 1, 0, tl::NamedBarrier<64>>::run(amax_fra` |
| store.global*2 (est.: sampled iterations x2) | 0.10 | 8.7% | 1.0 | 0.13 | `out_sf[((((((int64_t)((int)blockIdx.x)) % (int64_t)15) * (int64_t)32) + (((int64_t)((int)t` |
| load.global#2*2 (est.: sampled iterations x2) | 0.09 | 7.9% | 1.0 | 0.13 | `amax_fragment[i_1] = cutlass::bfloat16_t(__hmax((bfloat16_t(((nv_bfloat162*)(&(amax_fragme` |
| store.global#3*2 (est.: sampled iterations x2) | 0.09 | 7.9% | 1.0 | 0.13 | `*(fp8_e4_16_t*)(out + ((((((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int64_t)10240) +` |

### mhc_post_kernel grid (50, 1, 1) block (128, 1, 1)

16 launches, 200 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 1.34 us, warp 1.01 us, nsys graphs 1.60 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.84 | 82.3% | 1.11 | 84.0% |
| load | 0.12 | 12.3% | 0.15 | 11.2% |
| store | 0.05 | 5.4% | 0.06 | 4.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.12 | 12.3% | 1.0 | 0.16 | `*(float4*)(pm + 0) = *(float4*)(post + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int` |
| store.global | 0.05 | 5.4% | 1.0 | 0.10 | `for (int n_1 = 0; n_1 < 4; ++n_1)` |

### indexer_topk_segment_kernel grid (1, 6, 1) block (512, 1, 1)

2 launches, 96 warps; ranges, warp and last warp from the 6 recording CTAs' 96 warps; span 10.74 us, warp 10.67 us, nsys graphs 5.36 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed, with loops' unsampled iterations | 5.39 | 50.5% | 5.49 | 51.2% |
| wait | 4.02 | 37.7% | 3.94 | 36.7% |
| load | 1.09 | 10.2% | 1.07 | 10.0% |
| store | 0.17 | 1.6% | 0.22 | 2.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.09 | 10.2% | 1.0 | 1.14 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 4096); \| i` |
| wait.sync (sampled iterations) | 1.02 | 9.6% | 14.5 | 1.20 | `__syncthreads();` |
| wait.sync#23*9 (est.: sampled iterations x9) | 0.36 | 3.4% | 0.5 | 0.43 | `__syncthreads();` |
| wait.sync#20*9 (est.: sampled iterations x9) | 0.33 | 3.1% | 0.5 | 0.43 | `__syncthreads();` |
| wait.sync#24*9 (est.: sampled iterations x9) | 0.32 | 3.0% | 0.5 | 0.43 | `__syncthreads();` |
| wait.sync#14*8 (est.: sampled iterations x8) | 0.32 | 3.0% | 0.5 | 0.38 | `__syncthreads();` |
| wait.sync#19*9 (est.: sampled iterations x9) | 0.31 | 2.9% | 0.5 | 0.43 | `__syncthreads();` |
| wait.sync#6*8 (est.: sampled iterations x8) | 0.29 | 2.7% | 0.5 | 0.38 | `__syncthreads();` |
| wait.sync#5*8 (est.: sampled iterations x8) | 0.27 | 2.5% | 0.5 | 0.38 | `__syncthreads();` |
| wait.sync#13*8 (est.: sampled iterations x8) | 0.26 | 2.5% | 0.5 | 0.38 | `__syncthreads();` |
| store.global | 0.17 | 1.6% | 1.5 | 0.24 | `for (int i_3 = 0; i_3 < 4; ++i_3) \| if (((int)threadIdx.x) < total) \| if (((int)threadIdx.` |
| wait.sync#25 | 0.05 | 0.5% | 0.5 | 0.08 | `__syncthreads();` |
| wait.sync#17 | 0.04 | 0.4% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#18 | 0.04 | 0.4% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#16 | 0.04 | 0.4% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#8 | 0.04 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#3 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#11 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#4 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#7 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#12 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#9 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#2 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#15 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#22 | 0.03 | 0.3% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#10 | 0.03 | 0.3% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#21 | 0.02 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |

### token_counts_kernel grid (1, 1, 1) block (128, 1, 1)

16 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 1.15 us, warp 0.48 us, nsys graphs 1.50 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.28 | 58.3% | 0.35 | 28.7% |
| store | 0.14 | 30.0% | 0.64 | 52.7% |
| load | 0.06 | 11.7% | 0.23 | 18.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.14 | 30.0% | 0.2 | 0.58 | `if (((int)threadIdx.x) == 0) \| sampled[((int64_t)((int)blockIdx.x))] = (((int)(prefill_len` |
| load.global | 0.06 | 11.7% | 0.2 | 0.22 | `int count = sampled[((int64_t)((int)blockIdx.x))];` |

### flatten_sampled_kernel grid (1, 1, 1) block (128, 1, 1)

16 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 1.12 us, warp 0.81 us, nsys graphs 1.31 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.47 | 58.1% | 0.63 | 56.6% |
| unattributed | 0.32 | 38.9% | 0.39 | 35.3% |
| store | 0.02 | 3.0% | 0.09 | 8.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.42 | 51.2% | 1.0 | 0.42 | `for (int r = 0; r < num_reqs; ++r)` |
| load.global#2 | 0.06 | 6.9% | 0.2 | 0.22 | `int i = (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) - cu_num_logits[req]);` |
| store.global | 0.02 | 3.0% | 0.2 | 0.10 | `flat[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = co` |

### prepare_pos_seq_lens_kernel grid (2, 1, 1) block (256, 1, 1)

16 launches, 16 warps; ranges, warp and last warp from the 2 recording CTAs' 16 warps; span 1.09 us, warp 0.75 us, nsys graphs 1.31 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.31 | 41.5% | 0.35 | 32.1% |
| load | 0.30 | 40.3% | 0.62 | 56.1% |
| store | 0.14 | 18.2% | 0.13 | 11.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.30 | 40.3% | 0.5 | 0.61 | `int computed = num_computed_tokens[idx_mapping[((int64_t)((int)blockIdx.x))]];` |
| store.global | 0.07 | 9.7% | 0.5 | 0.19 | `for (int it = 0; it < (((max_reqs + 255) - num_reqs) >> 8); ++it)` |
| store.global#2 | 0.06 | 8.5% | 0.5 | 0.13 | `if (((int)threadIdx.x) == 0)` |

### sparse_mla_kernel_1 grid (16, 6, 1) block (128, 1, 1)

6 launches, 96 CTAs, 384 warps; span 2.05 us, warp 1.48 us, nsys graphs 1.98 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.43 | 96.7% | 1.80 | 63.5% |
| load | 0.03 | 2.2% | 0.75 | 26.5% |
| wait | 0.01 | 0.7% | 0.18 | 6.2% |
| compute | 0.00 | 0.2% | 0.07 | 2.4% |
| store | 0.00 | 0.1% | 0.04 | 1.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.03 | 2.2% | 0.1 | 0.94 | `M = __fmul_rn(Sinks[((int)blockIdx.x)], 0x1.7154764ee6c2fp+0f/*1.442695e+00*/); \| if (((in` |
| wait.sync | 0.01 | 0.7% | 0.1 | 0.38 | `__syncthreads();` |
| compute.reduce | 0.00 | 0.2% | 0.0 | 0.10 | `amax[0] = tl::AllReduce<tl::MaxOp, 8, 1, 0, tl::NamedBarrier<128>>::run(amax[0]);` |
| store.global | 0.00 | 0.1% | 0.0 | 0.06 | `*(uint2*)(Output + (((((int64_t)((int)blockIdx.y)) * (int64_t)8192) + (((int64_t)((int)blo` |
| compute.reduce#3 | 0.00 | 0.0% | 0.0 | 0.00 | `amax[0] = tl::AllReduce<tl::MaxOp, 8, 1, 0, tl::NamedBarrier<128>>::run(amax[0]);` |
| load.global#3 | 0.00 | 0.0% | 0.0 | 0.00 | `M = __fmul_rn(Sinks[((int)blockIdx.x)], 0x1.7154764ee6c2fp+0f/*1.442695e+00*/);` |
| load.global#4 | 0.00 | 0.0% | 0.0 | 0.00 | `if (((int)threadIdx.x) < 32)` |
| store.global#2 | 0.00 | 0.0% | 0.0 | 0.00 | `*(uint2*)(Output + (((((int64_t)((int)blockIdx.y)) * (int64_t)8192) + (((int64_t)((int)blo` |
| wait.sync#12 | 0.00 | 0.0% | 0.0 | 0.00 | `__syncthreads();` |
| wait.sync#13 | 0.00 | 0.0% | 0.0 | 0.00 | `__syncthreads();` |

### moe_swiglu_cast_kernel_kernel grid (8, 36, 1) block (288, 1, 1)

6 launches, 2592 warps; ranges, warp and last warp from the 4 recording CTAs' 36 warps; span 2.67 us, warp 0.88 us, nsys graphs 1.86 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.54 | 61.4% | 1.01 | 52.9% |
| load | 0.23 | 26.3% | 0.43 | 22.4% |
| compute | 0.06 | 6.5% | 0.26 | 13.4% |
| store | 0.05 | 5.8% | 0.21 | 11.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.23 | 26.3% | 1.0 | 0.34 | `for (int r = 0; r < 4; ++r)` |
| compute.reduce*4 (est.: sampled iterations x4) | 0.06 | 6.5% | 0.2 | 0.26 | `if (0 <= orders[r_2])` |
| store.global*4 (est.: sampled iterations x4) | 0.05 | 5.8% | 0.2 | 0.26 | `if (0 <= orders[r_2])` |

### topk_log_softmax_kernel grid (6, 1, 1) block (1024, 1, 1)

1 launches, 192 warps; ranges, warp and last warp from the 6 recording CTAs' 192 warps; span 16.86 us, warp 16.04 us, nsys graphs 16.03 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 13.13 | 81.9% | 13.89 | 82.5% |
| unattributed, with loops' unsampled iterations | 1.58 | 9.9% | 1.66 | 9.9% |
| wait | 0.93 | 5.8% | 0.80 | 4.8% |
| compute | 0.39 | 2.4% | 0.42 | 2.5% |
| store | 0.00 | 0.0% | 0.06 | 0.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 6.63 | 41.4% | 1.0 | 6.91 | `if (0 < (vocab & 2047)) \| if (0 < (vocab & 8191))` |
| load.global#2 | 6.49 | 40.5% | 1.0 | 7.07 | `for (int step_1 = 0; step_1 < (vocab >> 11); ++step_1) \| for (int step_1 = 0; step_1 < (vo` |
| wait.sync#2 | 0.38 | 2.4% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync | 0.30 | 1.9% | 1.0 | 0.74 | `__syncthreads();` |
| compute.reduce#3 | 0.17 | 1.1% | 1.0 | 0.22 | `ties[0] = vllm_add(ties[0], __shfl_xor_sync((uint)4294967295, ties[0], 16, 32));` |
| compute.reduce#2 | 0.11 | 0.7% | 1.0 | 0.16 | `se[0] = vllm_add(se[0], __shfl_xor_sync((uint)4294967295, se[0], 16, 32));` |
| compute.reduce | 0.11 | 0.7% | 1.0 | 0.16 | `best[0] = vllm_max(best[0], __shfl_xor_sync((uint)4294967295, best[0], 16, 32));` |
| wait.sync#4 | 0.10 | 0.6% | 1.0 | 0.16 | `__syncthreads();` |
| wait.sync#5 | 0.08 | 0.5% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#3 | 0.08 | 0.5% | 1.0 | 0.10 | `__syncthreads();` |
| load.global#3 (sampled iterations) | 0.01 | 0.1% | 0.0 | 0.38 | `float value_32 = vllm_sub(vllm_shifted(logits[((((int64_t)((int)blockIdx.x)) * logits_stri` |
| store.global (sampled iterations) | 0.00 | 0.0% | 0.0 | 0.10 | `output[(((((int64_t)c) * (int64_t)1024) + (((int64_t)((int)blockIdx.x)) * ((int64_t)topk))` |

### metadata_chunk_kernel grid (6, 1, 1) block (128, 1, 1)

2 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 8.08 us, warp 6.32 us, nsys graphs 3.52 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 3.18 | 50.2% | 4.77 | 59.0% |
| unattributed | 3.07 | 48.5% | 3.20 | 39.6% |
| store | 0.08 | 1.3% | 0.11 | 1.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 2.93 | 46.4% | 1.0 | 3.81 | `if ((((offset + ((int)blockIdx.x)) < src_rows) && (0 <= (offset + ((int)blockIdx.x)))))` |
| load.global#2 | 0.24 | 3.9% | 0.2 | 0.98 | `if (((0 <= req) & ((int64_t)0 <= pos)))` |
| store.global | 0.06 | 0.9% | 1.0 | 0.06 | `swa[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = con` |
| store.global#3 | 0.01 | 0.2% | 0.2 | 0.06 | `top_lengths[((int64_t)((int)blockIdx.x))] = condval_12; \| top_lengths[((int64_t)((int)bloc` |
| store.global#2 | 0.01 | 0.1% | 0.2 | 0.03 | `lengths[((int64_t)((int)blockIdx.x))] = ((int)max(info[1], (int64_t)0));` |

### scatter_rows_gather_kernel grid (1, 1, 1) block (128, 1, 1)

16 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 0.93 us, warp 0.85 us, nsys graphs 1.89 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.57 | 67.5% | 0.38 | 41.2% |
| store | 0.15 | 17.9% | 0.43 | 46.1% |
| load | 0.12 | 14.6% | 0.12 | 12.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.15 | 17.9% | 1.0 | 0.42 | `if (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) < (num_reqs * width))` |
| load.global | 0.12 | 14.6% | 1.0 | 0.13 | `index = idx_mapping[min((((((int64_t)0 <= ((int64_t)width)) && ((int64_t)0 <= rmod_1)) \|\| ` |

### mhc_collapse_kernel grid (30, 1, 1) block (128, 1, 1)

3 launches, 120 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 5.79 us, warp 5.72 us, nsys graphs 2.00 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 5.57 | 97.4% | 4.73 | 96.7% |
| load | 0.10 | 1.7% | 0.11 | 2.2% |
| store | 0.05 | 0.9% | 0.05 | 1.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.10 | 1.7% | 1.0 | 0.13 | `*(float4*)(mv + 0) = *(float4*)(mix + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int6` |
| store.global | 0.05 | 0.9% | 1.0 | 0.06 | `*(uint4*)(out + ((((int64_t)((int)blockIdx.x)) * (int64_t)1024) + (((int64_t)((int)threadI` |

### ced_requests_kernel grid (1, 1, 1) block (128, 1, 1)

16 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 0.90 us, warp 0.61 us, nsys graphs 1.18 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.32 | 52.6% | 0.34 | 38.0% |
| store | 0.18 | 30.3% | 0.44 | 49.9% |
| load | 0.10 | 17.1% | 0.11 | 12.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.18 | 30.3% | 1.0 | 0.45 | `if (((int)threadIdx.x) < 2) \| if (((int)threadIdx.x) <= 2)` |
| load.global | 0.10 | 17.1% | 1.0 | 0.11 | `uint v_ = tl::load_global_32((&(starts[1]))); \| uint v_ = tl::load_global_32((&(starts[min` |

### ced_copy_rows_kernel grid (50, 1, 1) block (128, 1, 1)

16 launches, 200 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 0.86 us, warp 0.82 us, nsys graphs 1.15 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.57 | 69.2% | 0.59 | 69.1% |
| unattributed | 0.25 | 30.8% | 0.26 | 30.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.57 | 69.2% | 1.0 | 0.61 | `for (int k = 0; k < 4; ++k) \| int64_t index = ced_ldg64((&(row_indices[((((int64_t)blockId` |

### mhc_collapse_kernel grid (50, 1, 1) block (128, 1, 1)

16 launches, 200 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 0.85 us, warp 0.76 us, nsys graphs 1.06 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.61 | 80.3% | 0.68 | 81.6% |
| load | 0.10 | 13.4% | 0.11 | 12.9% |
| store | 0.05 | 6.3% | 0.05 | 5.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.10 | 13.4% | 1.0 | 0.13 | `*(float4*)(mv + 0) = *(float4*)(mix + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int6` |
| store.global | 0.05 | 6.3% | 1.0 | 0.06 | `*(uint4*)(out + ((((int64_t)((int)blockIdx.x)) * (int64_t)1024) + (((int64_t)((int)threadI` |

### moe_combine_kernel grid (5, 6, 1) block (128, 1, 1)

6 launches, 120 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 2.14 us, warp 1.69 us, nsys graphs 1.15 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.22 | 72.3% | 1.69 | 78.7% |
| unattributed | 0.41 | 24.3% | 0.41 | 18.9% |
| store | 0.06 | 3.4% | 0.05 | 2.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.22 | 72.3% | 1.0 | 1.71 | `for (int k = 0; k < 3; ++k) \| for (int k = 0; k < 6; ++k) \| if ((((int)blockIdx.y) < share` |
| store.global | 0.06 | 3.4% | 1.0 | 0.10 | `{` |

### mhc_post_kernel grid (30, 1, 1) block (128, 1, 1)

4 launches, 120 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 3.22 us, warp 1.78 us, nsys graphs 1.36 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.63 | 91.5% | 1.98 | 93.2% |
| load | 0.10 | 5.8% | 0.10 | 4.9% |
| store | 0.05 | 2.7% | 0.04 | 1.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.10 | 5.8% | 1.0 | 0.13 | `*(float4*)(pm + 0) = *(float4*)(post + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int` |
| store.global | 0.05 | 2.7% | 1.0 | 0.06 | `for (int n_1 = 0; n_1 < 4; ++n_1)` |

### local_stats_kernel grid (6, 16, 1) block (128, 1, 1)

1 launches, 384 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 12.10 us, warp 11.67 us, nsys graphs 12.42 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 10.59 | 90.7% | 11.04 | 91.3% |
| unattributed | 0.71 | 6.1% | 0.70 | 5.8% |
| compute | 0.17 | 1.4% | 0.16 | 1.3% |
| wait | 0.14 | 1.2% | 0.06 | 0.5% |
| store | 0.07 | 0.6% | 0.13 | 1.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 10.59 | 90.7% | 1.0 | 11.04 | `if (expanded_local_pos[((int64_t)((int)blockIdx.x))] < num_speculative_steps)` |
| wait.sync | 0.14 | 1.2% | 1.0 | 0.48 | `__syncthreads();` |
| compute.reduce | 0.11 | 1.0% | 1.0 | 0.13 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| store.global | 0.07 | 0.6% | 1.0 | 0.13 | `if (((int)threadIdx.x) == 0)` |
| compute.reduce#2 | 0.06 | 0.5% | 1.0 | 0.06 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 2, 32);` |

### mhc_collapse_column_kernel grid (25, 1, 1) block (128, 1, 1)

15 launches, 100 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 0.80 us, warp 0.77 us, nsys graphs 0.99 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.62 | 80.2% | 0.62 | 78.5% |
| load | 0.10 | 12.6% | 0.11 | 13.6% |
| store | 0.06 | 7.2% | 0.06 | 7.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.10 | 12.6% | 1.0 | 0.13 | `*(float4*)(mv + 0) = *(float4*)(mix + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int6` |
| store.global | 0.06 | 7.2% | 1.0 | 0.06 | `*(uint4*)(out + ((((int64_t)((int)blockIdx.x)) * (int64_t)1024) + (((int64_t)((int)threadI` |

### padding_rows_kernel grid (1, 1, 1) block (256, 1, 1)

31 launches, 8 warps; ranges, warp and last warp from the 1 recording CTAs' 8 warps; span 0.35 us, warp 0.35 us, nsys graphs 0.77 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.26 | 72.7% | 0.27 | 72.4% |
| store | 0.10 | 27.3% | 0.10 | 27.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.10 | 27.3% | 1.0 | 0.10 | `if (((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) < rows)` |

### scatter_rows_kernel grid (1, 1, 1) block (128, 1, 1)

16 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 0.64 us, warp 0.59 us, nsys graphs 0.99 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.41 | 68.9% | 0.39 | 60.6% |
| load | 0.10 | 16.9% | 0.11 | 17.4% |
| store | 0.08 | 14.2% | 0.14 | 22.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.10 | 16.9% | 1.0 | 0.11 | `index = idx_mapping[min((((((int64_t)0 <= ((int64_t)width)) && ((int64_t)0 <= rmod_1)) \|\| ` |
| store.global | 0.08 | 14.2% | 1.0 | 0.13 | `if (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) < (num_reqs * width))` |

### indexer_topk_segment_kernel grid (2, 6, 1) block (512, 1, 1)

1 launches, 192 warps; ranges, warp and last warp from the 4 recording CTAs' 64 warps; span 9.95 us, warp 5.24 us, nsys graphs 0.83 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 3.14 | 59.9% | 5.89 | 59.4% |
| unattributed, with loops' unsampled iterations | 1.71 | 32.6% | 3.39 | 34.2% |
| load | 0.34 | 6.6% | 0.54 | 5.5% |
| store | 0.05 | 0.9% | 0.10 | 1.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync#19*9 (est.: sampled iterations x9) | 0.37 | 7.1% | 0.5 | 0.86 | `__syncthreads();` |
| wait.sync#20*9 (est.: sampled iterations x9) | 0.37 | 7.1% | 0.5 | 0.86 | `__syncthreads();` |
| wait.sync#24*9 (est.: sampled iterations x9) | 0.32 | 6.1% | 0.5 | 0.86 | `__syncthreads();` |
| wait.sync#23*9 (est.: sampled iterations x9) | 0.31 | 5.8% | 0.5 | 0.86 | `__syncthreads();` |
| wait.sync#13*8 (est.: sampled iterations x8) | 0.30 | 5.8% | 0.5 | 0.77 | `__syncthreads();` |
| wait.sync#14*8 (est.: sampled iterations x8) | 0.30 | 5.8% | 0.5 | 0.77 | `__syncthreads();` |
| wait.sync#6*8 (est.: sampled iterations x8) | 0.30 | 5.7% | 0.5 | 0.77 | `__syncthreads();` |
| wait.sync#5*8 (est.: sampled iterations x8) | 0.28 | 5.3% | 0.5 | 0.77 | `__syncthreads();` |
| load.global | 0.21 | 4.0% | 1.0 | 0.22 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 4096); \| i` |
| load.global#2 | 0.13 | 2.5% | 0.5 | 0.32 | `for (int i = 0; i < 8; ++i)` |
| wait.sync (sampled iterations) | 0.05 | 1.0% | 0.5 | 0.19 | `__syncthreads();` |
| wait.sync#25 | 0.05 | 1.0% | 0.5 | 0.16 | `__syncthreads();` |
| store.global | 0.05 | 0.9% | 0.5 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3) \| if (((int)threadIdx.x) < total) \| if (((int)threadIdx.` |
| wait.sync#18 | 0.04 | 0.8% | 0.5 | 0.13 | `__syncthreads();` |
| wait.sync#16 | 0.04 | 0.7% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#17 | 0.04 | 0.7% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#8 | 0.04 | 0.7% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#3 | 0.03 | 0.7% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#15 | 0.03 | 0.6% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#9 | 0.03 | 0.6% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#4 | 0.03 | 0.6% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#7 | 0.03 | 0.6% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#10 | 0.03 | 0.6% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#11 | 0.03 | 0.6% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#2 | 0.03 | 0.6% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#12 | 0.03 | 0.5% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#21 | 0.03 | 0.5% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#22 | 0.02 | 0.4% | 0.5 | 0.06 | `__syncthreads();` |

### ced_copy_rows_kernel grid (25, 1, 1) block (128, 1, 1)

15 launches, 100 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 0.64 us, warp 0.60 us, nsys graphs 1.17 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.36 | 60.4% | 0.39 | 60.7% |
| unattributed | 0.24 | 39.6% | 0.25 | 39.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.36 | 60.4% | 1.0 | 0.42 | `for (int k = 0; k < 4; ++k) \| int64_t index = ced_ldg64((&(row_indices[((((int64_t)blockId` |

### rope_kernel grid (6, 1, 1) block (128, 1, 1)

6 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 1.44 us, warp 1.38 us, nsys graphs 1.50 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.83 | 60.2% | 1.05 | 65.7% |
| load | 0.51 | 37.1% | 0.51 | 31.7% |
| store | 0.04 | 2.7% | 0.04 | 2.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.51 | 37.1% | 1.0 | 0.53 | `if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < tokens)) \| int64_t position ` |
| store.global | 0.04 | 2.7% | 1.0 | 0.06 | `*(uint4*)(x + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + ((((int64_t)((int)thread` |

### indexer_topk_merge_kernel grid (1, 6, 1) block (512, 1, 1)

1 launches, 96 warps; ranges, warp and last warp from the 6 recording CTAs' 96 warps; span 9.25 us, warp 9.02 us, nsys graphs 3.82 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 6.15 | 68.2% | 6.14 | 66.4% |
| unattributed | 2.22 | 24.6% | 2.24 | 24.2% |
| load | 0.58 | 6.4% | 0.74 | 8.0% |
| store | 0.07 | 0.8% | 0.13 | 1.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync#20*9 (est.: sampled iterations x9) | 0.72 | 7.9% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#24*9 (est.: sampled iterations x9) | 0.67 | 7.4% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#19*9 (est.: sampled iterations x9) | 0.64 | 7.0% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#6*8 (est.: sampled iterations x8) | 0.63 | 6.9% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync#23*9 (est.: sampled iterations x9) | 0.62 | 6.8% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#5*8 (est.: sampled iterations x8) | 0.59 | 6.6% | 1.0 | 0.77 | `__syncthreads();` |
| load.global | 0.57 | 6.3% | 1.0 | 0.70 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 8192);` |
| wait.sync#14*8 (est.: sampled iterations x8) | 0.57 | 6.3% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync#13*8 (est.: sampled iterations x8) | 0.56 | 6.2% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync | 0.15 | 1.7% | 1.0 | 0.22 | `__syncthreads();` |
| wait.sync#17 | 0.08 | 0.9% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#16 | 0.08 | 0.9% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#25 | 0.07 | 0.8% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#3 | 0.07 | 0.8% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#8 | 0.07 | 0.8% | 1.0 | 0.10 | `__syncthreads();` |
| store.global#2 | 0.06 | 0.7% | 0.9 | 0.10 | `Values[((((int64_t)((int)blockIdx.y)) * (int64_t)512) + ((int64_t)((int)threadIdx.x)))] = ` |
| wait.sync#2 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#9 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#7 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#12 | 0.06 | 0.7% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#21 | 0.06 | 0.7% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#4 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#10 | 0.06 | 0.6% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#15 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#18 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#22 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#11 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| load.global#2 | 0.01 | 0.1% | 0.1 | 0.10 | `if (((0 <= ((int*)sel_idx)[((int)threadIdx.x)]) && (((int*)sel_idx)[((int)threadIdx.x)] < ` |
| store.global | 0.01 | 0.1% | 0.1 | 0.13 | `Values[((((int64_t)((int)blockIdx.y)) * (int64_t)512) + ((int64_t)((int)threadIdx.x)))] = ` |

### token_ranks_kernel grid (6, 1, 1) block (256, 1, 1)

1 launches, 48 warps; ranges, warp and last warp from the 6 recording CTAs' 48 warps; span 7.74 us, warp 7.64 us, nsys graphs 8.61 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 6.85 | 89.7% | 6.85 | 88.8% |
| unattributed | 0.39 | 5.1% | 0.35 | 4.6% |
| compute | 0.36 | 4.7% | 0.45 | 5.8% |
| store | 0.04 | 0.6% | 0.06 | 0.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 6.85 | 89.7% | 1.0 | 7.10 | `if ((((int64_t)0 <= token_ids[((int64_t)((int)blockIdx.x))]) && (token_ids[((int64_t)((int` |
| compute.reduce | 0.36 | 4.7% | 1.0 | 0.70 | `total[0] = tl::AllReduce<tl::SumOp, 256, 1, 0, tl::NamedBarrier<256>>::run(total[0], (&(wo` |
| store.global | 0.04 | 0.6% | 1.0 | 0.06 | `output[((int64_t)((int)blockIdx.x))] = ((int64_t)total[0]);` |

### indexer_mxfp4_quantize_kernel grid (6, 1, 1) block (128, 1, 1)

2 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 3.78 us, warp 3.70 us, nsys graphs 2.94 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 2.28 | 61.8% | 2.08 | 55.1% |
| store | 1.10 | 29.7% | 1.34 | 35.6% |
| unattributed | 0.31 | 8.5% | 0.35 | 9.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2*16 (est.: sampled iterations x16) | 1.27 | 34.3% | 1.0 | 1.54 | `float v__2 = __fmul_rn(((float)X[(((((int64_t)((int)blockIdx.x)) * (int64_t)4096) + (((int` |
| store.global*16 (est.: sampled iterations x16) | 1.06 | 28.6% | 1.0 | 1.54 | `Pool[(((((slot & (int64_t)127) * (int64_t)64) + ((((int64_t)((int)threadIdx.x)) & (int64_t` |
| load.global | 1.01 | 27.4% | 1.0 | 1.07 | `for (int i = 0; i < 32; ++i)` |
| store.global#2 | 0.04 | 1.2% | 1.0 | 0.06 | `Pool[(((((slot & (int64_t)127) * (int64_t)4) + ((slot >> (int64_t)7) * ((int64_t)pool_stri` |

### activation_swiglu_clamp_kernel grid (1, 6, 1) block (128, 1, 1)

6 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 0.93 us, warp 0.73 us, nsys graphs 1.86 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.62 | 85.5% | 0.78 | 84.4% |
| load | 0.07 | 9.6% | 0.09 | 9.8% |
| store | 0.04 | 4.9% | 0.05 | 5.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.07 | 9.6% | 0.8 | 0.10 | `*(uint4*)(gate + 0) = *(uint4*)(X + ((((int64_t)((int)blockIdx.y)) * (int64_t)1152) + (((i` |
| store.global | 0.04 | 4.9% | 0.8 | 0.06 | `*(uint4*)(Y + ((((int64_t)((int)blockIdx.y)) * (int64_t)576) + (((int64_t)((int)threadIdx.` |

### indexer_score_kernel grid (128, 6, 1) block (128, 1, 1)

1 launches, 3072 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 4.90 us, warp 2.42 us, nsys graphs 2.24 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.45 | 60.0% | 2.21 | 61.1% |
| unattributed, with loops' unsampled iterations | 0.74 | 30.8% | 1.09 | 30.1% |
| wait | 0.14 | 5.9% | 0.16 | 4.4% |
| compute | 0.06 | 2.6% | 0.13 | 3.5% |
| store | 0.02 | 0.7% | 0.03 | 0.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 (sampled iterations) | 0.72 | 29.6% | 0.5 | 1.98 | `for (int i_1 = 0; i_1 < 32; ++i_1)` |
| load.global | 0.66 | 27.1% | 1.0 | 1.06 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 4096); \| i` |
| load.global#3 (sampled iterations) | 0.08 | 3.3% | 0.5 | 0.26 | `if (((int)threadIdx.x) < 64)` |
| wait.sync#2 (sampled iterations) | 0.08 | 3.2% | 0.5 | 0.26 | `__syncthreads();` |
| compute.mma (sampled iterations) | 0.06 | 2.6% | 0.5 | 0.13 | `for (int ki = 0; ki < 2; ++ki)` |
| wait.sync (sampled iterations) | 0.02 | 1.0% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#4 (sampled iterations) | 0.02 | 1.0% | 0.5 | 0.06 | `__syncthreads();` |
| store.global (sampled iterations) | 0.02 | 0.7% | 0.5 | 0.03 | `if ((((int)threadIdx.x) >> 6) == 0)` |
| wait.sync#3 (sampled iterations) | 0.02 | 0.7% | 0.5 | 0.03 | `__syncthreads();` |

### indexer_score_kernel grid (64, 6, 1) block (128, 1, 1)

1 launches, 1536 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 4.26 us, warp 2.36 us, nsys graphs 1.76 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.24 | 52.4% | 2.08 | 49.2% |
| unattributed, with loops' unsampled iterations | 0.80 | 34.0% | 1.38 | 32.6% |
| wait | 0.21 | 9.0% | 0.51 | 12.1% |
| compute | 0.09 | 3.6% | 0.19 | 4.5% |
| store | 0.02 | 0.9% | 0.06 | 1.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.64 | 26.9% | 1.0 | 0.93 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 4096); \| i` |
| load.global#2 (sampled iterations) | 0.54 | 22.9% | 0.5 | 1.12 | `for (int i_1 = 0; i_1 < 32; ++i_1)` |
| wait.sync#2 (sampled iterations) | 0.09 | 4.0% | 0.5 | 0.26 | `__syncthreads();` |
| compute.mma (sampled iterations) | 0.09 | 3.6% | 0.5 | 0.22 | `for (int ki = 0; ki < 2; ++ki)` |
| load.global#3 (sampled iterations) | 0.06 | 2.6% | 0.5 | 0.29 | `if (((int)threadIdx.x) < 64)` |
| wait.sync#3 (sampled iterations) | 0.05 | 2.3% | 0.5 | 0.19 | `__syncthreads();` |
| wait.sync (sampled iterations) | 0.03 | 1.4% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#4 (sampled iterations) | 0.03 | 1.4% | 0.5 | 0.10 | `__syncthreads();` |
| store.global (sampled iterations) | 0.02 | 0.9% | 0.5 | 0.06 | `if ((((int)threadIdx.x) >> 6) == 0)` |

### gemm_splitk_reduce_kernel grid (1, 6, 1) block (128, 1, 1)

4 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 0.85 us, warp 0.83 us, nsys graphs 0.99 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.46 | 55.9% | 0.44 | 53.9% |
| load | 0.32 | 38.5% | 0.32 | 39.2% |
| store | 0.05 | 5.5% | 0.06 | 6.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.32 | 38.5% | 1.0 | 0.34 | `acc[0] = P[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x))` |
| store.global | 0.05 | 5.5% | 1.0 | 0.06 | `C[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x)) * (int64` |

### metadata_pages_kernel grid (6, 1, 1) block (128, 1, 1)

2 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 1.34 us, warp 0.96 us, nsys graphs 1.06 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.73 | 75.8% | 0.78 | 58.3% |
| load | 0.14 | 14.7% | 0.21 | 15.5% |
| store | 0.09 | 9.5% | 0.35 | 26.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.12 | 12.7% | 1.0 | 0.13 | `if ((((offset + ((int)blockIdx.x)) < src_rows) && (0 <= (offset + ((int)blockIdx.x)))))` |
| store.global | 0.09 | 9.5% | 0.2 | 0.37 | `if (((int)threadIdx.x) < 32)` |
| load.global#2 | 0.02 | 1.9% | 0.2 | 0.10 | `if (((req < table_rows) && (((int)threadIdx.x) < table_cols)))` |

### norm_forward_and_per_token_cast_kernel_kernel grid (6, 1, 1) block (128, 1, 1)

1 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 2.59 us, warp 2.55 us, nsys graphs 1.89 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 2.44 | 96.0% | 2.66 | 102.5% |
| load | 0.42 | 16.5% | 0.48 | 18.5% |
| compute | 0.13 | 5.2% | 0.13 | 4.9% |
| wait | 0.05 | 2.0% | 0.06 | 2.5% |
| unattributed | 0.00 | 0.0% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*5 (est.: sampled iterations x5) | 2.13 | 83.8% | 1.0 | 2.24 | `for (int j_1 = 0; j_1 < 2; ++j_1)` |
| store.global*5 (est.: sampled iterations x5) | 0.26 | 10.2% | 1.0 | 0.32 | `for (int j = 0; j < 4; ++j)` |
| load.global*5 (est.: sampled iterations x5) | 0.23 | 8.9% | 1.0 | 0.32 | `*(uint4*)(x_local + (vi * 8)) = *(uint4*)(x + ((((((int64_t)vi) * (int64_t)1024) + (((int6` |
| load.global#2*5 (est.: sampled iterations x5) | 0.19 | 7.6% | 1.0 | 0.32 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 1024) + (((int)threadIdx.x) * 8)` |
| compute.reduce | 0.13 | 5.2% | 1.0 | 0.16 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| wait.sync | 0.05 | 2.0% | 1.0 | 0.06 | `__syncthreads();` |
| store.global#2 | 0.04 | 1.5% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| store.global#3 | 0.01 | 0.4% | 0.2 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |

### rope_kernel grid (6, 1, 1) block (256, 1, 1)

2 launches, 48 warps; ranges, warp and last warp from the 6 recording CTAs' 48 warps; span 1.01 us, warp 0.92 us, nsys graphs 1.17 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.61 | 66.2% | 0.69 | 68.3% |
| load | 0.27 | 29.4% | 0.27 | 27.0% |
| store | 0.04 | 4.4% | 0.05 | 4.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.27 | 29.4% | 1.0 | 0.27 | `if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < tokens)) \| int64_t position ` |
| store.global | 0.04 | 4.4% | 1.0 | 0.06 | `*(uint4*)(x + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + ((((int64_t)((int)thread` |

### mhc_collapse_column_kernel grid (30, 1, 1) block (128, 1, 1)

1 launches, 120 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 0.80 us, warp 0.77 us, nsys graphs 0.93 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.62 | 80.2% | 0.61 | 79.2% |
| load | 0.09 | 11.7% | 0.10 | 12.5% |
| store | 0.06 | 8.1% | 0.06 | 8.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.09 | 11.7% | 1.0 | 0.13 | `*(float4*)(mv + 0) = *(float4*)(mix + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int6` |
| store.global | 0.06 | 8.1% | 1.0 | 0.06 | `*(uint4*)(out + ((((int64_t)((int)blockIdx.x)) * (int64_t)1024) + (((int64_t)((int)threadI` |

### ced_copy_rows_kernel grid (30, 1, 1) block (128, 1, 1)

1 launches, 120 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 0.67 us, warp 0.61 us, nsys graphs 1.18 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.36 | 58.9% | 0.38 | 57.1% |
| unattributed | 0.25 | 41.1% | 0.29 | 42.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.36 | 58.9% | 1.0 | 0.38 | `for (int k = 0; k < 4; ++k) \| int64_t index = ced_ldg64((&(row_indices[((((int64_t)blockId` |
=== timeline and tables: exit 0 in 22 s
=== nsys timeline

## Timeline: 7952 launches in 339147.9 us; the GPU is busy 253605.3 us (75%) and idle 85542.7 us in 5087 gaps; gaps over 200 us split it into 6 segments

| segment | launches | wall us | busy us | busy | first kernel |
|---:|---:|---:|---:|---:|---|
| 0 | 29 | 1232.0 | 45.5 | 4% | fill_rows_kernel |
| 1 | 18 | 828.4 | 273.8 | 33% | mhc_project_streams_kernel |
| 2 | 1260 | 47451.9 | 43125.3 | 91% | gemm_mxfp8_kernel |
| 3 | 1 | 4.7 | 4.7 | 100% | padding_rows_kernel |
| 4 | 23 | 873.9 | 46.6 | 5% | assign_draft_budget_kernel |
| 5 | 6621 | 215267.5 | 210109.4 | 98% | mhc_project_streams_kernel |

| idle us | after | before |
|---:|---|---|
| 72382.8 | scatter_rows_kernel | padding_rows_kernel |
| 343.9 | indexSelectSmallIndex | mhc_project_streams_kernel |
| 307.5 | padding_rows_kernel | assign_draft_budget_kernel |
| 233.2 | mhc_finalize_kernel | gemm_mxfp8_kernel |
| 222.2 | vectorized_gather_kernel | mhc_project_streams_kernel |
| 134.7 | sink_policy_kernel | post_update_kernel |
| 125.3 | mhc_finalize_kernel | gemm_mxfp8_kernel |
| 105.4 | padding_rows_kernel | prepare_prefill_inputs_kernel |
| 103.6 | ced_requests_kernel | metadata_builder_kernel |
| 101.7 | fill_rows_kernel | apply_write_kernel |

## Kernels by total duration (7952 launches, 6670 in graphs)

| kernel | launches | in graphs | total us | share | mean us |
|---|---:|---:|---:|---:|---:|
| vocab_packed_projection_kernel | 36 | 18 | 144531.1 | 50.7% | 4014.75 |
| l2_prefetch_kernel | 433 | 433 | 29201.3 | 10.2% | 67.44 |
| gemm_mxfp8_kernel | 864 | 831 | 28023.3 | 9.8% | 32.43 |
| gemm_bf16_kernel | 90 | 90 | 24276.9 | 8.5% | 269.74 |
| moe_grouped_fp8_fp4_gemm_kernel | 324 | 312 | 19764.9 | 6.9% | 61.00 |
| mhc_project_streams_kernel | 324 | 312 | 5269.4 | 1.8% | 16.26 |
| gemm_fp8w_bf16_partials_kernel | 162 | 156 | 4391.1 | 1.5% | 27.11 |
| gemm_bf16_partials_kernel | 234 | 224 | 4357.7 | 1.5% | 18.62 |
| mhc_finalize_kernel | 324 | 312 | 3556.5 | 1.2% | 10.98 |
| sparse_mla_kernel | 162 | 156 | 3490.4 | 1.2% | 21.55 |
| gemm_bf16_splitk_kernel | 36 | 34 | 2099.4 | 0.7% | 58.32 |
| cache_writer_kernel | 252 | 241 | 1048.0 | 0.4% | 4.16 |
| rope_kernel | 486 | 465 | 989.5 | 0.3% | 2.04 |
| metadata_chunk_kernel | 90 | 88 | 843.3 | 0.3% | 9.37 |
| norm_forward_and_per_token_cast_kernel_kernel | 468 | 449 | 792.6 | 0.3% | 1.69 |
=== nsys timeline: exit 0 in 0 s
=== decode steps against the grid's step
## A decode step against the grid's step (17 decode steps, 9 grid runs)

GPU busy a step: 14042.2 us against 8916.8 us (+5125.3); kernel time a step: 15871.0 us against 10656.6 us (+5214.4).

| kernel | launches a step | grid | us a step | grid | difference us | exposed us | last warp: wait / load / compute / store / other | longest last-warp wait |
|---|---:|---:|---:|---:|---:|---:|---|---|
| vocab_packed_projection_kernel | 2 | 1 | 8038.3 | 4000.9 | +4037.5 | 8038.3 | 59% / 32% / 10% / 0% / 0% | wait.sync#2*17 |
| moe_grouped_fp8_fp4_gemm_kernel | 18 | 18 | 1071.4 | 111.9 | +959.5 | 1001.0 | 37% / 25% / 36% / 1% / 1% | wait.mbarrier*40 |
| l2_prefetch_kernel | 25 | 25 | 1690.3 | 1629.2 | +61.1 | 221.0 | 0% / 100% / 0% / 0% / 0% |  |
| mhc_project_streams_kernel | 18 | 18 | 290.8 | 240.8 | +50.0 | 9.0 | 17% / 37% / 17% / 4% / 26% | wait.sync*4 |
| gemm_bf16_kernel | 5 | 5 | 1352.7 | 1310.0 | +42.7 | 1352.7 | 91% / 0% / 4% / 1% / 5% | wait.mbarrier#2 |
| mhc_finalize_kernel | 18 | 18 | 200.7 | 159.1 | +41.6 | 13.2 | 5% / 12% / 11% / 18% / 54% | wait.sync |
| token_ranks_kernel | 4 | 0 | 39.4 | 0.0 | +39.4 | 39.4 | 0% / 91% / 4% / 1% / 5% |  |
| topk_log_softmax_kernel | 2 | 0 | 35.4 | 0.0 | +35.4 | 35.4 | 5% / 82% / 2% / 0% / 10% | wait.sync#2 |
| router_partials_kernel | 9 | 9 | 44.1 | 18.0 | +26.1 | 3.5 | 0% / 33% / 25% / 4% / 38% |  |
| topk_ids_kernel | 1 | 0 | 21.6 | 0.0 | +21.6 | 21.6 | 3% / 73% / 0% / 14% / 9% | wait.sync |
| sparse_mla_kernel | 9 | 9 | 189.4 | 174.0 | +15.4 | 31.0 | 3% / 41% / 4% / 2% / 49% | wait.sync |
| moe_swiglu_cast_kernel_kernel | 9 | 9 | 30.7 | 16.5 | +14.2 | 30.7 | 0% / 19% / 13% / 9% / 58% |  |
| local_stats_kernel | 1 | 0 | 12.5 | 0.0 | +12.5 | 12.5 | 1% / 91% / 1% / 1% / 6% | wait.sync |
| metadata_decode_kernel | 4 | 4 | 22.7 | 15.0 | +7.7 | 4.0 | 0% / 63% / 0% / 6% / 30% |  |
| rope_kernel | 27 | 24 | 55.9 | 48.5 | +7.3 | 10.4 | 0% / 18% / 0% / 3% / 80% |  |
| moe_combine_kernel | 9 | 9 | 19.3 | 12.0 | +7.3 | 19.3 | 0% / 78% / 0% / 3% / 20% |  |
| sink_policy_kernel | 1 | 0 | 6.6 | 0.0 | +6.6 | 6.6 | 0% / 43% / 0% / 23% / 34% |  |
| compress_gather_partial_state_kernel | 1 | 1 | 11.5 | 5.1 | +6.4 | 0.0 | 0% / 58% / 0% / 2% / 39% |  |
| indexSelectSmallIndex | 7 | 6 | 28.3 | 22.8 | +5.5 | 28.2 |  |  |
| sink_merge_kernel | 1 | 0 | 5.5 | 0.0 | +5.5 | 5.5 | 10% / 60% / 14% / 3% / 13% | wait.sync*3 |
| context_inputs_kernel | 1 | 0 | 5.5 | 0.0 | +5.5 | 5.5 | 0% / 75% / 0% / 4% / 21% |  |
| moe_dispatch_kernel | 9 | 9 | 31.8 | 26.5 | +5.2 | 10.3 | 38% / 2% / 0% / 26% / 34% | wait.sync#6*9 |
| post_update_kernel | 1 | 0 | 4.2 | 0.0 | +4.2 | 4.2 | 0% / 37% / 0% / 32% / 31% |  |
| indexer_topk_segment_kernel | 3 | 3 | 15.4 | 11.5 | +3.9 | 15.4 | 36% / 10% / 0% / 3% / 51% | wait.sync |
| sink_candidates_kernel | 1 | 0 | 3.6 | 0.0 | +3.6 | 3.6 | 9% / 21% / 22% / 8% / 40% | wait.sync*4 |
| gemm_mxfp8_kernel | 48 | 48 | 1527.6 | 1582.9 | -55.3 | 936.1 | 94% / 4% / 5% / 0% / 0% | wait.cp_async*118 |
| vectorized_elementwise_kernel | 0 | 87 | 0.0 | 113.1 | -113.1 | 0.0 |  |  |
=== decode steps against the grid's step: exit 0 in 0 s
{"output": "/c/iket/step-tp4-c1-20261026T142439Z", "steps": [["decode steps, on, under run-iket", 0], ["records against the dry run's count", 0], ["decode steps, nsys, off, graphs", 0], ["timeline and tables", 0], ["nsys timeline", 0], ["decode steps against the grid's step", 0]]}
