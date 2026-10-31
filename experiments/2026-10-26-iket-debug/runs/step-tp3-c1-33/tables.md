=== records against the dry run's count
pid_0x110: recommended_max_ts_cnt_per_warp 0
pid_0x3a: recommended_max_ts_cnt_per_warp 80
pid_0xab: recommended_max_ts_cnt_per_warp 0
    62 events in one warp,    112 warps recording, bound 62: indexer_topk_segment_kernel
    60 events in one warp,     16 warps recording, bound 60: gemm_fp8w_bf16_cast_kernel
    60 events in one warp,     40 warps recording, bound 64: sparse_mla_kernel
    58 events in one warp,    112 warps recording, bound 58: indexer_topk_merge_kernel
    54 events in one warp,     16 warps recording, bound 54: gemm_fp8w_bf16_partials_kernel
    52 events in one warp,      8 warps recording, bound 52: adaptive_offsets_kernel
    50 events in one warp,     16 warps recording, bound 50: vocab_packed_projection_kernel
    50 events in one warp,     32 warps recording, bound 50: gemm_bf16_splitk_kernel
    32 events in one warp,     40 warps recording, bound 32: gemm_mxfp8_kernel
    30 events in one warp,     16 warps recording, bound 30: moe_grouped_fp8_fp4_gemm_kernel
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
    12 events in one warp,     12 warps recording, bound 12: per_token_cast_kernel_kernel
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
     6 events in one warp,     12 warps recording, bound 6: moe_swiglu_cast_kernel_kernel
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
=== records against the dry run's count: exit 0 in 16 s
=== decode steps, nsys, off, graphs

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:00<?, ?kernel/s]
                                                                               

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:03<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:03<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:03<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:03<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:03<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:03<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:03<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:03<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:03<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:03<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:03<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:03<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:03<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:04<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:04<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:04<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:04<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:04<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:04<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:04<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:04<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:04<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:04<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:04<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:04<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:04<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:04<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:04<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:04<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:04<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:04<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:05<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:05<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:05<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:05<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:05<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:05<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:05<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:05<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:05<?, ?kernel/s, TileLangKernels (389 keys)][14:46:14] /workspace/tilelang/src/transform/thread_storage_sync.cc:1084: Warning: [ThreadSync] Hoisting sync out of an if whose condition is not safe for an in-if sync. This is not a fix: both ends of the conflict are inside the branch, so the hoisted barrier no longer separates them and the race remains. Constraining the condition -- a T.assume on the parameters it reads -- keeps the barrier in place instead. Condition: chunk * 256 + tx < num_reqs * 5

                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:05<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:05<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:05<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:05<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:05<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:05<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:05<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:05<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:05<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:06<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:06<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:06<?, ?kernel/s, TileLangKernels (389 keys)][14:46:15] /workspace/tilelang/src/transform/legalize_vectorized_loop.cc:85: Warning: T.vectorized loop over `k` with extent 4 is lowered as a serial loop because TileLang could not find a valid vectorization plan. Scalar accumulator updates inside the loop are a common cause; move reductions to T.unroll or T.serial if this is intended.

                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:06<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:06<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:06<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:06<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:06<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:06<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:06<?, ?kernel/s, TileLangKernels (389 keys)][14:46:15] /workspace/tilelang/src/transform/legalize_vectorized_loop.cc:85: Warning: T.vectorized loop over `k` with extent 4 is lowered as a serial loop because TileLang could not find a valid vectorization plan. Scalar accumulator updates inside the loop are a common cause; move reductions to T.unroll or T.serial if this is intended.

                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:06<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:06<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:06<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:06<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:06<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:06<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:06<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:06<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:06<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:07<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:07<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:07<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:07<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:07<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:07<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:07<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:07<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:07<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:07<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:07<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:07<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:07<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:07<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:08<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:08<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:08<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:08<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:08<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:08<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:08<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:08<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:08<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:08<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:08<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:08<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:09<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:09<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:09<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:09<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:09<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:09<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:09<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:09<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:09<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:09<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:09<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:09<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:09<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:09<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:10<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:10<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:10<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:10<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:10<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:10<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:10<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:10<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:10<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:10<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:10<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:10<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:11<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:11<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:11<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:11<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:11<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:11<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:11<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:11<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:11<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:11<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:11<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:11<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:11<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:12<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:12<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:12<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:12<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:12<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:12<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:12<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:12<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:12<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:12<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:12<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:13<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:13<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:13<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:13<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:13<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:13<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:13<?, ?kernel/s, TileLangKernels (389 keys)][14:46:22] /workspace/tilelang/src/transform/legalize_vectorized_loop.cc:85: Warning: T.vectorized loop over `k` with extent 4 is lowered as a serial loop because TileLang could not find a valid vectorization plan. Scalar accumulator updates inside the loop are a common cause; move reductions to T.unroll or T.serial if this is intended.

                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:13<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:13<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:13<?, ?kernel/s, TileLangKernels (389 keys)][14:46:23] /workspace/tilelang/src/transform/legalize_vectorized_loop.cc:85: Warning: T.vectorized loop over `k` with extent 4 is lowered as a serial loop because TileLang could not find a valid vectorization plan. Scalar accumulator updates inside the loop are a common cause; move reductions to T.unroll or T.serial if this is intended.

                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:14<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:14<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:14<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:14<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:14<?, ?kernel/s, TileLangKernels (389 keys)][14:46:23] /workspace/tilelang/src/transform/legalize_vectorized_loop.cc:85: Warning: T.vectorized loop over `k` with extent 4 is lowered as a serial loop because TileLang could not find a valid vectorization plan. Scalar accumulator updates inside the loop are a common cause; move reductions to T.unroll or T.serial if this is intended.

                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:14<?, ?kernel/s, TileLangKernels (389 keys)][14:46:23] /workspace/tilelang/src/transform/legalize_vectorized_loop.cc:85: Warning: T.vectorized loop over `k` with extent 4 is lowered as a serial loop because TileLang could not find a valid vectorization plan. Scalar accumulator updates inside the loop are a common cause; move reductions to T.unroll or T.serial if this is intended.

                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:14<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:14<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:14<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:14<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:14<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:14<?, ?kernel/s, TileLangKernels (389 keys)][14:46:24] /workspace/tilelang/src/transform/legalize_vectorized_loop.cc:85: Warning: T.vectorized loop over `k` with extent 4 is lowered as a serial loop because TileLang could not find a valid vectorization plan. Scalar accumulator updates inside the loop are a common cause; move reductions to T.unroll or T.serial if this is intended.

                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:15<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:15<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:15<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:15<?, ?kernel/s, TileLangKernels (389 keys)][14:46:24] /workspace/tilelang/src/transform/legalize_vectorized_loop.cc:85: Warning: T.vectorized loop over `k` with extent 4 is lowered as a serial loop because TileLang could not find a valid vectorization plan. Scalar accumulator updates inside the loop are a common cause; move reductions to T.unroll or T.serial if this is intended.

                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:15<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:15<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:15<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:15<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:15<?, ?kernel/s, TileLangKernels (389 keys)][14:46:24] /workspace/tilelang/src/transform/legalize_vectorized_loop.cc:85: Warning: T.vectorized loop over `k` with extent 4 is lowered as a serial loop because TileLang could not find a valid vectorization plan. Scalar accumulator updates inside the loop are a common cause; move reductions to T.unroll or T.serial if this is intended.

                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:15<?, ?kernel/s, TileLangKernels (389 keys)][14:46:24] /workspace/tilelang/src/transform/legalize_vectorized_loop.cc:85: Warning: T.vectorized loop over `k` with extent 4 is lowered as a serial loop because TileLang could not find a valid vectorization plan. Scalar accumulator updates inside the loop are a common cause; move reductions to T.unroll or T.serial if this is intended.

                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:15<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:16<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:16<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:16<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:16<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:16<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:16<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:16<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:16<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:16<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:16<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:16<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:16<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:16<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:16<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:16<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:17<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:17<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:17<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:17<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:17<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:17<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:17<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:17<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:17<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:17<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:17<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:17<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:17<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:17<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:17<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:17<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:18<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:18<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:18<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:18<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:18<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:18<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:19<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:19<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:19<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:19<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:20<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:20<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:20<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:20<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:20<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:20<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:20<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:20<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:21<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:21<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:21<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:21<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:21<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:21<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:22<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:22<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:22<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:22<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:22<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:22<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:23<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:23<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:23<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:23<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:23<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:23<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:23<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:23<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:23<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:23<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:24<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:24<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:24<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:24<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:24<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:24<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:24<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:24<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:24<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:24<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:24<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:24<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:24<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:24<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:24<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:24<?, ?kernel/s, TileLangKernels (389 keys)][14:46:34] /workspace/tilelang/src/transform/verify_buffer_init.cc:394: Warning: Buffer read before initialization: 1 buffer(s) read before anything writes them
  [1] `best` is read before anything writes it
Initialize the buffer before its first read (for example with `T.clear`, `T.fill`, or a `T.copy` into it). If you believe this is a false positive, disable the check by setting `PassConfigKey.TL_DISABLE_BUFFER_INIT_CHECK` in the pass config.

                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:24<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:24<?, ?kernel/s, TileLangKernels (389 keys)][14:46:34] /workspace/tilelang/src/transform/verify_buffer_init.cc:394: Warning: Buffer read before initialization: 1 buffer(s) read before anything writes them
  [1] `best` is read before anything writes it
Initialize the buffer before its first read (for example with `T.clear`, `T.fill`, or a `T.copy` into it). If you believe this is a false positive, disable the check by setting `PassConfigKey.TL_DISABLE_BUFFER_INIT_CHECK` in the pass config.
  [1] `best` is read before anything writes it
Initialize the buffer before its first read (for example with `T.clear`, `T.fill`, or a `T.copy` into it). If you believe this is a false positive, disable the check by setting `PassConfigKey.TL_DISABLE_BUFFER_INIT_CHECK` in the pass config.
  [1] `best` is read before anything writes it
Initialize the buffer before its first read (for example with `T.clear`, `T.fill`, or a `T.copy` into it). If you believe this is a false positive, disable the check by setting `PassConfigKey.TL_DISABLE_BUFFER_INIT_CHECK` in the pass config.

                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:25<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:25<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:25<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:25<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:25<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:25<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:25<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:25<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:25<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:25<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:26<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:26<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:26<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:26<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:26<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:26<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:26<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:26<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:26<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:26<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:26<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:26<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:26<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:26<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:26<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:26<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:27<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:27<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:27<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:27<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:27<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:27<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:27<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:27<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:27<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:27<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:27<?, ?kernel/s, TileLangKernels (389 keys)]
                                                                                                           

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:28<?, ?kernel/s, TileLangKernels (389 keys)]
JIT kernel warmup (389 compile keys): 100%|██████████| 1/1 [00:28<00:00, 28.70s/kernel, TileLangKernels (389 keys)]
JIT kernel warmup (389 compile keys): 100%|██████████| 1/1 [00:28<00:00, 28.70s/kernel, TileLangKernels (389 keys)]

JIT kernel warmup (389 compile keys):   0%|          | 0/1 [00:00<?, ?kernel/s]INFO 10-26 14:46:56 [kernel_warmup.py:187] JIT kernel warmup finished in 0.00s.

JIT kernel warmup (389 compile keys): 100%|██████████| 1/1 [00:00<00:00, 758.88kernel/s, TileLangKernels (389 keys)]
2026-10-26 14:46:56,675 - INFO - autotuner.py:2789 - flashinfer.jit: [Autotuner]: Loaded 0 configs from /tmp/tmpx83dq22k/autotune_configs.json
2026-10-26 14:46:56,675 - INFO - autotuner.py:972 - flashinfer.jit: [Autotuner]: Autotuning process starts ...
2026-10-26 14:46:56,705 - INFO - autotuner.py:995 - flashinfer.jit: [Autotuner]: Autotuning process ends
2026-10-26 14:46:56,710 - INFO - autotuner.py:2652 - flashinfer.jit: [Autotuner]: Saved 0 configs to /c/vllm-iket-step-tp3-c1/flashinfer_autotune_cache/0.6.18.post1/121a/2afa7bd10e7702b4c184498501d131b375fec6593f566cae4f493a0d17a223b0/tmpxyv5jfq1/autotune_configs.json (0 new, 0 from previous config)




engine ready in 135 s




{"mode": "off", "boot_s": 135.4, "request_s": [0.577, 0.493], "requests": 1, "tokens": 32, "prompt_tokens": 32, "token_ids": [69542, 107252, 33678, 107252, 33678, 107252, 33678, 107252, 33678, 107252, 33678, 107252, 33678, 107252, 33678, 107252, 33678, 107252, 33678, 107252, 33678, 107252, 33678, 107252, 33678, 107252, 33678, 107252, 33678, 107252, 33678, 107252], "generated": [32], "logprobs": [{"69542": -11.769635200500488, "74521": -11.769636154174805, "82194": -11.769640922546387, "98865": -11.769638061523438, "115488": -11.769639015197754}, {"467": -11.76964282989502, "30841": -11.76964282989502, "69652": -11.76964282989502, "107252": -11.769640922546387, "112546": -11.769641876220703}, {"17378": -11.769627571105957, "33678": -11.76962661743164, "61135": -11.76962947845459, "62222": -11.769630432128906, "71862": -11.76962947845459}, {"23033": -11.769637107849121, "26940": -11.769632339477539, "79103": -11.769638061523438, "89117": -11.769637107849121, "107252": -11.769625663757324}, {"17378": -11.769627571105957, "33678": -11.76962661743164, "61135": -11.76962947845459, "62222": -11.769630432128906, "71862": -11.76962947845459}, {"23033": -11.769637107849121, "26940": -11.769632339477539, "79103": -11.769638061523438, "89117": -11.769637107849121, "107252": -11.769625663757324}, {"17378": -11.769627571105957, "33678": -11.76962661743164, "61135": -11.76962947845459, "62222": -11.769630432128906, "71862": -11.76962947845459}, {"23033": -11.769637107849121, "26940": -11.769632339477539, "79103": -11.769638061523438, "89117": -11.769637107849121, "107252": -11.769625663757324}, {"17378": -11.769627571105957, "33678": -11.76962661743164, "61135": -11.76962947845459, "62222": -11.769630432128906, "71862": -11.76962947845459}, {"23033": -11.769637107849121, "26940": -11.769632339477539, "79103": -11.769638061523438, "89117": -11.769637107849121, "107252": -11.769625663757324}, {"17378": -11.769627571105957, "33678": -11.76962661743164, "61135": -11.76962947845459, "62222": -11.769630432128906, "71862": -11.76962947845459}, {"23033": -11.769637107849121, "26940": -11.769632339477539, "79103": -11.769638061523438, "89117": -11.769637107849121, "107252": -11.769625663757324}, {"17378": -11.769627571105957, "33678": -11.76962661743164, "61135": -11.76962947845459, "62222": -11.769630432128906, "71862": -11.76962947845459}, {"23033": -11.769637107849121, "26940": -11.769632339477539, "79103": -11.769638061523438, "89117": -11.769637107849121, "107252": -11.769625663757324}, {"17378": -11.769627571105957, "33678": -11.76962661743164, "61135": -11.76962947845459, "62222": -11.769630432128906, "71862": -11.76962947845459}, {"23033": -11.769637107849121, "26940": -11.769632339477539, "79103": -11.769638061523438, "89117": -11.769637107849121, "107252": -11.769625663757324}, {"17378": -11.769627571105957, "33678": -11.76962661743164, "61135": -11.76962947845459, "62222": -11.769630432128906, "71862": -11.76962947845459}, {"23033": -11.769637107849121, "26940": -11.769632339477539, "79103": -11.769638061523438, "89117": -11.769637107849121, "107252": -11.769625663757324}, {"17378": -11.769627571105957, "33678": -11.76962661743164, "61135": -11.76962947845459, "62222": -11.769630432128906, "71862": -11.76962947845459}, {"23033": -11.769637107849121, "26940": -11.769632339477539, "79103": -11.769638061523438, "89117": -11.769637107849121, "107252": -11.769625663757324}, {"17378": -11.769627571105957, "33678": -11.76962661743164, "61135": -11.76962947845459, "62222": -11.769630432128906, "71862": -11.76962947845459}, {"23033": -11.769637107849121, "26940": -11.769632339477539, "79103": -11.769638061523438, "89117": -11.769637107849121, "107252": -11.769625663757324}, {"17378": -11.769627571105957, "33678": -11.76962661743164, "61135": -11.76962947845459, "62222": -11.769630432128906, "71862": -11.76962947845459}, {"23033": -11.769637107849121, "26940": -11.769632339477539, "79103": -11.769638061523438, "89117": -11.769637107849121, "107252": -11.769625663757324}, {"17378": -11.769627571105957, "33678": -11.76962661743164, "61135": -11.76962947845459, "62222": -11.769630432128906, "71862": -11.76962947845459}, {"23033": -11.769637107849121, "26940": -11.769632339477539, "79103": -11.769638061523438, "89117": -11.769637107849121, "107252": -11.769625663757324}, {"17378": -11.769627571105957, "33678": -11.76962661743164, "61135": -11.76962947845459, "62222": -11.769630432128906, "71862": -11.76962947845459}, {"23033": -11.769637107849121, "26940": -11.769632339477539, "79103": -11.769638061523438, "89117": -11.769637107849121, "107252": -11.769625663757324}, {"17378": -11.769627571105957, "33678": -11.76962661743164, "61135": -11.76962947845459, "62222": -11.769630432128906, "71862": -11.76962947845459}, {"23033": -11.769637107849121, "26940": -11.769632339477539, "79103": -11.769638061523438, "89117": -11.769637107849121, "107252": -11.769625663757324}, {"17378": -11.769627571105957, "33678": -11.76962661743164, "61135": -11.76962947845459, "62222": -11.769630432128906, "71862": -11.76962947845459}, {"23033": -11.769637107849121, "26940": -11.769632339477539, "79103": -11.769638061523438, "89117": -11.769637107849121, "107252": -11.769625663757324}], "batched": 512, "chunk": 256, "VLLM_DS41_IKET": null, "VLLM_DS41_IKET_CTAS": 0, "VLLM_DS41_IKET_ITERS": 0, "VLLM_DS41_IKET_SKIP": "", "filled": false, "eager": false, "iket_modules": ["cutlass._mlir.dialects.iket", "cutlass.cute.experimental.iket", "cutlass.cute.experimental.iket.iket"], "grid": {"spec": {"shapes": [{"num_reqs": 1, "context_len": 48, "num_tokens": 6}], "replays": 8, "warmup": 1, "settle": 0}, "runs": 9, "step_ms": [11.075103759765625, 11.236576080322266, 10.885408401489258, 11.254015922546387, 10.785951614379883, 10.733887672424316, 10.74294376373291, 10.746912002563477], "forward_ms": [2.7174720764160156, 2.7640960216522217, 2.789247989654541, 2.810015916824341, 2.667520046234131, 2.66211199760437, 2.66540789604187, 2.6707839965820312], "drafter_ms": [7.079455852508545, 7.313375949859619, 6.955904006958008, 7.289728164672852, 6.999904155731201, 6.950719833374023, 6.9613118171691895, 6.950079917907715]}, "tp_shape": 3, "text_overrides": {"num_hidden_layers": 6, "compress_ratios": [0, 0, 2, 2, 1, 1, 0, 0, 0], "kv_source_layer_ids": [2, 4], "index_source_layer_ids": [2, 4], "candidate_source_layer_id": 4, "engram_layer_ids": [], "dspark_target_layer_ids": [3, 4, 5], "num_attention_heads": 24, "o_groups": 3, "moe_intermediate_size": 768}, "drafts": 5}
engine shut down: 7.2 GiB freed, 15.3 GiB still allocated
Collecting data...
Generating '/tmp/nsys-report-e64e.qdstrm'


Generated:
	/c/iket/step-tp3-c1-20261026T143800Z/nsys-graphs.nsys-rep
	/c/iket/step-tp3-c1-20261026T143800Z/nsys-graphs.sqlite
=== decode steps, nsys, off, graphs: exit 0 in 147 s
=== timeline and tables
trace /c/iket/step-tp3-c1-20261026T143800Z/trace/iket/pid_0x193c/iket.decoded_results.json: 12587 launches, graphLaunches dict of 118

## Timeline: 7322 launches in 802465.6 us; the GPU is busy 262700.4 us (33%) and idle 539765.2 us in 4491 gaps; gaps over 200 us split it into 520 segments

The 15 of 520 segments with the most busy time:

| segment | launches | wall us | busy us | busy | first kernel |
|---:|---:|---:|---:|---:|---|
| 48 | 156 | 10237.3 | 7320.7 | 72% | token_ranks_kernel |
| 79 | 116 | 6860.8 | 6783.0 | 99% | mhc_project_streams_kernel |
| 107 | 117 | 6848.6 | 6755.1 | 99% | padding_rows_kernel |
| 146 | 118 | 7078.9 | 6736.2 | 95% | metadata_builder_kernel |
| 170 | 117 | 6830.4 | 6736.4 | 99% | padding_rows_kernel |
| 224 | 116 | 6829.2 | 6742.8 | 99% | mhc_project_streams_kernel |
| 247 | 116 | 6899.1 | 6786.1 | 98% | mhc_project_streams_kernel |
| 274 | 117 | 6851.2 | 6754.9 | 99% | padding_rows_kernel |
| 310 | 116 | 6843.2 | 6761.1 | 99% | mhc_project_streams_kernel |
| 371 | 120 | 7298.5 | 6735.9 | 92% | metadata_builder_kernel |
| 402 | 117 | 6889.1 | 6736.1 | 98% | padding_rows_kernel |
| 432 | 120 | 7102.2 | 6750.1 | 95% | metadata_builder_kernel |
| 458 | 120 | 7237.8 | 6749.3 | 93% | metadata_builder_kernel |
| 477 | 120 | 7226.7 | 6738.1 | 93% | metadata_builder_kernel |
| 517 | 116 | 6824.6 | 6744.0 | 99% | mhc_project_streams_kernel |

| idle us | after | before |
|---:|---|---|
| 97364.0 | resample_kernel | insert_kernel |
| 79272.0 | scatter_rows_kernel | padding_rows_kernel |
| 3367.3 | dspark_confidence_kernel | scatter_rows_gather_kernel |
| 3349.5 | dspark_confidence_kernel | scatter_rows_gather_kernel |
| 3284.0 | norm_forward_and_per_token_cast_kernel_kernel | sink_flags_kernel |
| 3201.1 | norm_forward_and_per_token_cast_kernel_kernel | sink_flags_kernel |
| 3027.9 | dspark_confidence_kernel | scatter_rows_gather_kernel |
| 2952.6 | norm_forward_and_per_token_cast_kernel_kernel | sink_flags_kernel |
| 2936.4 | norm_forward_and_per_token_cast_kernel_kernel | sink_flags_kernel |
| 2863.9 | dspark_confidence_kernel | scatter_rows_gather_kernel |

## Engine steps

| step | kind | launches | wall us | busy us | busy | most span |
|---:|---|---:|---:|---:|---:|---|
| 0 | prefill | 413 | 48628.2 | 17333.6 | 36% | vocab_packed_projection_kernel 8072, gemm_mxfp8_kernel 3195, moe_grouped_fp8_fp4_gemm_kernel 2408 |
| 1 | decode | 432 | 133496.5 | 15117.8 | 11% | vocab_packed_projection_kernel 8104, gemm_mxfp8_kernel 2292, l2_prefetch_kernel 1866 |
| 2 | decode | 432 | 113313.4 | 15280.0 | 13% | vocab_packed_projection_kernel 8038, gemm_mxfp8_kernel 2469, l2_prefetch_kernel 2097 |
| 3 | decode | 432 | 38173.7 | 15305.9 | 40% | vocab_packed_projection_kernel 8104, gemm_mxfp8_kernel 2517, l2_prefetch_kernel 2276 |
| 4 | decode | 432 | 33973.2 | 15393.7 | 45% | vocab_packed_projection_kernel 8113, gemm_mxfp8_kernel 2271, l2_prefetch_kernel 1577 |
| 5 | decode | 432 | 32678.6 | 15299.7 | 47% | vocab_packed_projection_kernel 8031, gemm_mxfp8_kernel 2259, l2_prefetch_kernel 1586 |
| 6 | decode | 432 | 32592.4 | 15343.3 | 47% | vocab_packed_projection_kernel 8040, gemm_mxfp8_kernel 2238, l2_prefetch_kernel 1576 |
| 7 | decode | 432 | 35415.8 | 15365.4 | 43% | vocab_packed_projection_kernel 8046, gemm_mxfp8_kernel 2208, l2_prefetch_kernel 1591 |
| 8 | decode | 432 | 34422.7 | 15425.0 | 45% | vocab_packed_projection_kernel 8150, gemm_mxfp8_kernel 2291, l2_prefetch_kernel 1595 |
| 9 | decode | 432 | 36043.7 | 15422.6 | 43% | vocab_packed_projection_kernel 8130, gemm_mxfp8_kernel 2256, l2_prefetch_kernel 1589 |
| 10 | decode | 432 | 35492.6 | 15322.8 | 43% | vocab_packed_projection_kernel 8038, gemm_mxfp8_kernel 2256, l2_prefetch_kernel 1599 |
| 11 | decode | 432 | 35040.9 | 15282.8 | 44% | vocab_packed_projection_kernel 8028, gemm_mxfp8_kernel 2247, l2_prefetch_kernel 1585 |
| 12 | decode | 432 | 37161.7 | 15374.9 | 41% | vocab_packed_projection_kernel 8101, gemm_mxfp8_kernel 2223, l2_prefetch_kernel 1598 |
| 13 | decode | 432 | 38662.0 | 15315.2 | 40% | vocab_packed_projection_kernel 8032, gemm_mxfp8_kernel 2233, moe_grouped_fp8_fp4_gemm_kernel 1574 |
| 14 | decode | 432 | 35613.4 | 15509.7 | 44% | vocab_packed_projection_kernel 8227, gemm_mxfp8_kernel 2229, l2_prefetch_kernel 1574 |
| 15 | decode | 432 | 33919.6 | 15289.0 | 45% | vocab_packed_projection_kernel 8042, gemm_mxfp8_kernel 2233, l2_prefetch_kernel 1593 |
| 16 | decode | 429 | 41460.8 | 15319.0 | 37% | vocab_packed_projection_kernel 8041, gemm_mxfp8_kernel 2216, l2_prefetch_kernel 1586 |

## Overview over 16 decode steps: 6909 launches of 142 kernel shapes, spans summing to 273801.4 us (17112.6 a step)

| kernel | grid | launches a step | span us a step | share | span us a launch | last warp: wait | load | compute | store | unattributed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vocab_packed_projection_kernel | (1, 4040, 1) | 2 | 8079.1 | 47.2% | 4022.50 | 65% | 27% | 10% | 0% | 0% |
| l2_prefetch_kernel | (2, 1, 1) | 25 | 1678.6 | 9.8% | 64.77 | 0% | 100% | 0% | 0% | 0% |
| gemm_bf16_kernel | (2020, 1, 1) | 5 | 1321.3 | 7.7% | 263.44 | 90% | 0% | 4% | 1% | 5% |
| gemm_mxfp8_kernel | (80, 1, 1) | 19 | 1073.3 | 6.3% | 49.60 | 67% | 8% | 6% | 0% | 19% |
| moe_grouped_fp8_fp4_gemm_kernel | (12, 30, 1) | 8.625 | 965.6 | 5.6% | 136.32 | 58% | 17% | 22% | 1% | 3% |
| gemm_mxfp8_kernel | (192, 1, 1) | 9 | 538.1 | 3.1% | 51.34 | 22% | 5% | 6% | 0% | 66% |
| moe_grouped_fp8_fp4_gemm_kernel | (40, 30, 1) | 8.625 | 533.2 | 3.1% | 73.15 | 15% | 18% | 0% | 0% | 67% |
| gemm_mxfp8_kernel | (28, 1, 1) | 9 | 323.0 | 1.9% | 35.76 | 36% | 11% | 11% | 0% | 43% |
| mhc_project_streams_kernel | (40, 1, 1) | 18 | 307.0 | 1.8% | 17.02 | 18% | 32% | 16% | 3% | 31% |
| gemm_fp8w_bf16_partials_kernel | (48, 4, 1) | 9 | 305.3 | 1.8% | 24.27 | 37% | 16% | 10% | 0% | 37% |
| gemm_mxfp8_kernel | (24, 1, 1) | 9 | 294.3 | 1.7% | 30.61 | 32% | 20% | 18% | 0% | 30% |
| gemm_bf16_partials_kernel | (6, 8, 1) | 6 | 142.0 | 0.8% | 23.26 | 32% | 0% | 32% | 2% | 34% |
| mhc_finalize_kernel | (5, 1, 1) | 11.25 | 131.4 | 0.8% | 12.24 | 7% | 16% | 10% | 20% | 47% |
| gemm_bf16_splitk_kernel | (16, 5, 1) | 2 | 108.3 | 0.6% | 54.70 | 88% | 8% | 4% | 0% | 0% |
| sparse_mla_kernel | (5, 1, 5) | 3.75 | 90.1 | 0.5% | 17.79 | 21% | 21% | 7% | 4% | 48% |
| mhc_finalize_kernel | (10, 1, 1) | 6 | 72.7 | 0.4% | 12.78 | 9% | 15% | 8% | 16% | 52% |
| sparse_mla_kernel | (2, 2, 10) | 3 | 54.7 | 0.3% | 17.41 | 18% | 21% | 26% | 2% | 33% |
| gemm_mxfp8_kernel | (64, 1, 1) | 2 | 48.8 | 0.3% | 23.55 | 30% | 12% | 10% | 1% | 47% |
| moe_dispatch_kernel | (1, 1, 1) | 9 | 44.2 | 0.3% | 4.70 | 35% | 2% | 0% | 25% | 37% |
| moe_grouped_fp8_fp4_gemm_kernel | (12, 36, 1) | 0.375 | 42.1 | 0.2% | 112.40 | 18% | 18% | 0% | 0% | 63% |

| wait | kernel | grid | last-warp us a step | a launch | first statement |
|---|---|---|---:|---:|---|
| wait.cp_async*22 | gemm_mxfp8_kernel | (80, 1, 1) | 189.5 | 9.97 | `tl::cp_async_wait<2>();` |
| wait.mbarrier*40 | moe_grouped_fp8_fp4_gemm_kernel | (12, 30, 1) | 179.2 | 20.78 | `loaded[(ko % 3)].wait(((ko % 6) / 3));` |
| wait.cp_async*118 | gemm_mxfp8_kernel | (80, 1, 1) | 170.6 | 8.98 | `tl::cp_async_wait<2>();` |
| wait.sync#2*22 | gemm_mxfp8_kernel | (80, 1, 1) | 136.4 | 7.18 | `__syncthreads();` |
| wait.sync#2*118 | gemm_mxfp8_kernel | (80, 1, 1) | 113.3 | 5.96 | `__syncthreads();` |
| wait.cp_async*19 | gemm_mxfp8_kernel | (28, 1, 1) | 65.3 | 7.25 | `tl::cp_async_wait<1>();` |
| wait.cp_async*40 | moe_grouped_fp8_fp4_gemm_kernel | (12, 30, 1) | 59.4 | 6.88 | `tl::cp_async_wait<2>();` |
| wait.cp_async*13 | gemm_bf16_splitk_kernel | (16, 5, 1) | 48.8 | 24.39 | `tl::cp_async_wait<2>();` |
| wait.sync*4 | mhc_project_streams_kernel | (40, 1, 1) | 48.4 | 2.69 | `__syncthreads();` |
| wait.sync#2*19 | gemm_mxfp8_kernel | (28, 1, 1) | 33.7 | 3.75 | `__syncthreads();` |
| wait.sync#2*38 | gemm_mxfp8_kernel | (24, 1, 1) | 33.4 | 3.71 | `__syncthreads();` |
| wait.cp_async*38 | gemm_mxfp8_kernel | (24, 1, 1) | 32.2 | 3.58 | `tl::cp_async_wait<2>();` |
| wait.mbarrier#2 | gemm_bf16_kernel | (2020, 1, 1) | 29.4 | 5.87 | `mbarrier[0].wait(0);` |
| wait.cp_async*17 | vocab_packed_projection_kernel | (1, 4040, 1) | 26.3 | 13.14 | `tl::cp_async_wait<6>();` |
| wait.sync#2*17 | vocab_packed_projection_kernel | (1, 4040, 1) | 26.3 | 13.14 | `__syncthreads();` |
| wait.cp_async*14 | gemm_fp8w_bf16_partials_kernel | (48, 4, 1) | 25.2 | 2.80 | `tl::cp_async_wait<3>();` |
| wait.sync#5*40 | moe_grouped_fp8_fp4_gemm_kernel | (12, 30, 1) | 22.8 | 2.64 | `__syncthreads();` |
| wait.sync*38 | gemm_mxfp8_kernel | (24, 1, 1) | 21.2 | 2.36 | `__syncthreads();` |
| wait.sync#4*40 | moe_grouped_fp8_fp4_gemm_kernel | (12, 30, 1) | 20.5 | 2.37 | `__syncthreads();` |
| wait.cp_async*4 | gemm_mxfp8_kernel | (80, 1, 1) | 20.2 | 1.07 | `tl::cp_async_wait<2>();` |

### vocab_packed_projection_kernel grid (1, 4040, 1) block (128, 1, 1)

32 launches, 16160 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 4022.50 us, warp 49.71 us, nsys graphs 4010.67 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 31.08 | 62.5% | 31.92 | 65.2% |
| load | 10.62 | 21.4% | 13.18 | 26.9% |
| compute | 5.05 | 10.2% | 5.01 | 10.2% |
| unattributed | 2.91 | 5.9% | 0.00 | 0.0% |
| store | 0.05 | 0.1% | 0.05 | 0.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.cp_async*17 (est.: sampled iterations x17) | 13.94 | 28.0% | 1.0 | 41.34 | `tl::cp_async_wait<6>();` |
| wait.sync#2*17 (est.: sampled iterations x17) | 11.66 | 23.5% | 1.0 | 34.54 | `__syncthreads();` |
| load.global | 8.11 | 16.3% | 1.0 | 14.29 | `int start = row_offsets[(((int)blockIdx.y) * 16)]; \| int start = row_offsets[(((int)blockI` |
| compute.mma*17 (est.: sampled iterations x17) | 4.35 | 8.8% | 1.0 | 4.35 | `{` |
| load.cp_async#2*17 (est.: sampled iterations x17) | 2.23 | 4.5% | 1.0 | 2.72 | `for (int i_6 = 0; i_6 < 8; ++i_6) \| for (int i_9 = 0; i_9 < 4; ++i_9)` |
| wait.sync#3*17 (est.: sampled iterations x17) | 1.09 | 2.2% | 1.0 | 1.09 | `__syncthreads();` |
| wait.sync*17 (est.: sampled iterations x17) | 1.07 | 2.2% | 1.0 | 1.09 | `__syncthreads();` |
| wait.cp_async#2*17 (est.: sampled iterations x17) | 0.95 | 1.9% | 1.0 | 1.09 | `tl::cp_async_wait<7>();` |
| wait.sync#4 | 0.51 | 1.0% | 1.0 | 1.68 | `__syncthreads();` |
| wait.cp_async#3 | 0.50 | 1.0% | 1.0 | 2.03 | `tl::cp_async_wait<4>();` |
| wait.sync#6 | 0.37 | 0.8% | 1.0 | 1.39 | `__syncthreads();` |
| wait.cp_async#5 | 0.33 | 0.7% | 1.0 | 1.70 | `tl::cp_async_wait<2>();` |
| load.cp_async | 0.28 | 0.6% | 1.0 | 0.32 | `for (int i = 0; i < 4; ++i) \| for (int i = 0; i < 8; ++i) \| for (int i_1 = 0; i_1 < 4; ++i` |
| compute.mma#2 | 0.24 | 0.5% | 1.0 | 0.26 | `{` |
| compute.mma#3 | 0.23 | 0.5% | 1.0 | 0.26 | `{` |
| compute.mma#4 | 0.23 | 0.5% | 1.0 | 0.26 | `{` |
| wait.sync#8 | 0.21 | 0.4% | 1.0 | 0.88 | `__syncthreads();` |
| wait.cp_async#7 | 0.17 | 0.3% | 1.0 | 0.93 | `tl::cp_async_wait<0>();` |
| wait.sync#5 | 0.06 | 0.1% | 1.0 | 0.06 | `__syncthreads();` |
| store.global | 0.05 | 0.1% | 1.0 | 0.06 | `for (int i_13 = 0; i_13 < 2; ++i_13) \| for (int i_14 = 0; i_14 < 4; ++i_14) \| for (int i_9` |
| wait.sync#7 | 0.05 | 0.1% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#9 | 0.05 | 0.1% | 1.0 | 0.06 | `__syncthreads();` |
| wait.cp_async#4 | 0.04 | 0.1% | 1.0 | 0.06 | `tl::cp_async_wait<5>();` |
| wait.cp_async#6 | 0.04 | 0.1% | 1.0 | 0.06 | `tl::cp_async_wait<3>();` |
| wait.cp_async#8 | 0.04 | 0.1% | 1.0 | 0.06 | `tl::cp_async_wait<1>();` |

### l2_prefetch_kernel grid (2, 1, 1) block (128, 1, 1)

400 launches, 8 warps; ranges, warp and last warp from the 2 recording CTAs' 8 warps; span 64.77 us, warp 63.94 us, nsys graphs 67.94 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 63.52 | 99.3% | 66.85 | 99.6% |
| unattributed | 0.43 | 0.7% | 0.29 | 0.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 63.52 | 99.3% | 1.0 | 64.51 | `for (int s = 0; s < (nseg2 >> 1); ++s)` |

### gemm_bf16_kernel grid (2020, 1, 1) block (256, 1, 1)

80 launches, 16160 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 263.44 us, warp 3.72 us, nsys graphs 263.97 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 3.15 | 84.7% | 5.96 | 89.8% |
| unattributed | 0.35 | 9.5% | 0.33 | 4.9% |
| compute | 0.14 | 3.8% | 0.29 | 4.3% |
| load | 0.04 | 1.0% | 0.00 | 0.0% |
| store | 0.04 | 0.9% | 0.07 | 1.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.mbarrier#2 | 3.04 | 81.8% | 0.5 | 7.01 | `mbarrier[0].wait(0);` |
| compute.mma | 0.14 | 3.8% | 0.5 | 0.29 | `{` |
| wait.sync | 0.08 | 2.0% | 1.0 | 0.10 | `__syncthreads();` |
| load.tma | 0.04 | 1.0% | 0.5 | 0.13 | `if (tl::tl_shuffle_elect<128>())` |
| store.global | 0.04 | 0.9% | 0.5 | 0.10 | `for (int i_3 = 0; i_3 < 4; ++i_3) \| for (int i_4 = 0; i_4 < 8; ++i_4) \| for (int i_6 = 0; ` |
| wait.mbarrier | 0.03 | 0.8% | 0.5 | 0.06 | `mbarrier[2].wait(1);` |

### gemm_mxfp8_kernel grid (80, 1, 1) block (128, 1, 1)

304 launches, 320 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 49.60 us, warp 45.91 us, nsys graphs 49.60 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 43.49 | 94.7% | 10.45 | 18.9% |
| wait | 1.87 | 4.1% | 37.03 | 67.0% |
| compute | 0.25 | 0.5% | 3.11 | 5.6% |
| load | 0.24 | 0.5% | 4.60 | 8.3% |
| store | 0.06 | 0.1% | 0.06 | 0.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.cp_async#2 | 0.60 | 1.3% | 1.0 | 2.18 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| wait.cp_async#3 | 0.44 | 1.0% | 1.0 | 1.82 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>();` |
| wait.sync#3 | 0.44 | 1.0% | 1.0 | 1.60 | `__syncthreads();` |
| wait.sync#4 | 0.38 | 0.8% | 1.0 | 1.41 | `__syncthreads();` |
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

### moe_grouped_fp8_fp4_gemm_kernel grid (12, 30, 1) block (128, 1, 1)

138 launches, 1440 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 136.32 us, warp 33.11 us, nsys graphs 68.51 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 12.84 | 38.8% | 35.16 | 57.5% |
| unattributed | 9.93 | 30.0% | 1.98 | 3.2% |
| compute | 6.56 | 19.8% | 13.32 | 21.8% |
| load | 3.64 | 11.0% | 10.32 | 16.9% |
| store | 0.14 | 0.4% | 0.35 | 0.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.mbarrier*40 (est.: sampled iterations x40) | 6.88 | 20.8% | 0.5 | 24.32 | `loaded[(ko % 3)].wait(((ko % 6) / 3));` |
| compute.mma*40 (est.: sampled iterations x40) | 6.56 | 19.8% | 0.5 | 14.08 | `{` |
| load#2*40 (est.: sampled iterations x40) | 2.88 | 8.7% | 0.5 | 8.96 | `if (ko < 38)` |
| wait.sync#5*40 (est.: sampled iterations x40) | 2.40 | 7.2% | 0.5 | 6.40 | `__syncthreads();` |
| wait.sync#4*40 (est.: sampled iterations x40) | 1.20 | 3.6% | 0.5 | 2.56 | `__syncthreads();` |
| wait.sync#6*40 (est.: sampled iterations x40) | 1.20 | 3.6% | 0.5 | 2.56 | `__syncthreads();` |
| wait.cp_async*40 (est.: sampled iterations x40) | 1.04 | 3.1% | 0.5 | 2.56 | `tl::cp_async_wait<2>();` |
| load.global#3 | 0.52 | 1.6% | 0.5 | 1.60 | `if (((expert < 128) && (0 <= expert))) \| if (((expert < 384) && (0 <= expert)))` |
| store.global | 0.14 | 0.4% | 0.5 | 0.35 | `for (int i_2 = 0; i_2 < 16; ++i_2) \| for (int i_4 = 0; i_4 < 32; ++i_4)` |
| load.global | 0.11 | 0.3% | 1.0 | 0.29 | `int expert = TileExpert[((int64_t)((int)blockIdx.y))];` |
| load | 0.08 | 0.3% | 0.5 | 0.26 | `for (int s = 0; s < 2; ++s)` |
| wait.sync#2 | 0.06 | 0.2% | 0.5 | 0.16 | `__syncthreads();` |
| load.global#2 | 0.05 | 0.1% | 0.5 | 0.19 | `if (((int)threadIdx.x) < 32) \| if (((int)threadIdx.x) < 64)` |
| wait.sync | 0.04 | 0.1% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#3 | 0.02 | 0.0% | 0.5 | 0.06 | `__syncthreads();` |

### gemm_mxfp8_kernel grid (192, 1, 1) block (128, 1, 1)

144 launches, 768 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 51.34 us, warp 30.74 us, nsys graphs 63.90 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 16.72 | 54.4% | 16.68 | 66.4% |
| wait | 11.24 | 36.6% | 5.63 | 22.4% |
| compute | 1.38 | 4.5% | 1.45 | 5.8% |
| load | 1.34 | 4.4% | 1.32 | 5.2% |
| store | 0.05 | 0.2% | 0.06 | 0.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync#2*8 (est.: sampled iterations x8) | 2.90 | 9.4% | 1.0 | 12.03 | `__syncthreads();` |
| wait.cp_async*8 (est.: sampled iterations x8) | 2.40 | 7.8% | 1.0 | 12.29 | `tl::cp_async_wait<2>();` |
| wait.cp_async#2 | 1.82 | 5.9% | 1.0 | 5.55 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| wait.cp_async#3 | 1.62 | 5.3% | 1.0 | 4.48 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>();` |
| compute.mma*8 (est.: sampled iterations x8) | 1.13 | 3.7% | 1.0 | 1.54 | `{` |
| wait.sync#4 | 1.05 | 3.4% | 1.0 | 3.07 | `__syncthreads();` |
| wait.sync#3 | 1.05 | 3.4% | 1.0 | 3.63 | `__syncthreads();` |
| load.cp_async#2*8 (est.: sampled iterations x8) | 1.02 | 3.3% | 1.0 | 2.56 | `for (int i_5 = 0; i_5 < 2; ++i_5) \| for (int i_5 = 0; i_5 < 4; ++i_5) \| tl::cp_async_gs_co` |
| wait.sync*8 (est.: sampled iterations x8) | 0.42 | 1.4% | 1.0 | 0.51 | `__syncthreads();` |
| load.cp_async | 0.33 | 1.1% | 1.0 | 0.77 | `for (int i_1 = 0; i_1 < 2; ++i_1) \| for (int i_1 = 0; i_1 < 4; ++i_1) \| for (int i_1 = 0; ` |
| compute.mma#3 | 0.13 | 0.4% | 1.0 | 0.16 | `{` |
| compute.mma#2 | 0.12 | 0.4% | 1.0 | 0.13 | `{` |
| store.global | 0.05 | 0.2% | 1.0 | 0.06 | `for (int i_10 = 0; i_10 < 16; ++i_10) \| for (int i_10 = 0; i_10 < 8; ++i_10) \| for (int i_` |

### moe_grouped_fp8_fp4_gemm_kernel grid (40, 30, 1) block (128, 1, 1)

138 launches, 4800 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 73.15 us, warp 5.53 us, nsys graphs 34.53 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.91 | 34.6% | 0.21 | 66.9% |
| wait | 1.50 | 27.2% | 0.05 | 15.3% |
| load | 0.99 | 17.8% | 0.06 | 17.8% |
| compute | 0.98 | 17.8% | 0.00 | 0.0% |
| store | 0.14 | 2.6% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*6 (est.: sampled iterations x6) | 0.98 | 17.8% | 0.5 | 2.11 | `{` |
| wait.mbarrier*6 (est.: sampled iterations x6) | 0.45 | 8.1% | 0.5 | 1.34 | `loaded[(ko % 3)].wait((ko / 3));` |
| wait.sync#5*6 (est.: sampled iterations x6) | 0.41 | 7.4% | 0.5 | 1.54 | `__syncthreads();` |
| load#2*6 (est.: sampled iterations x6) | 0.40 | 7.2% | 0.5 | 1.15 | `if (ko < 4)` |
| load.global#3 | 0.39 | 7.1% | 0.5 | 1.10 | `if (((expert < 128) && (0 <= expert))) \| if (((expert < 384) && (0 <= expert)))` |
| wait.cp_async*6 (est.: sampled iterations x6) | 0.19 | 3.5% | 0.5 | 1.15 | `tl::cp_async_wait<2>();` |
| wait.sync#4*6 (est.: sampled iterations x6) | 0.17 | 3.0% | 0.5 | 0.38 | `__syncthreads();` |
| wait.sync#6*6 (est.: sampled iterations x6) | 0.17 | 3.0% | 0.5 | 0.38 | `__syncthreads();` |
| store.global | 0.14 | 2.6% | 0.5 | 0.38 | `for (int i_2 = 0; i_2 < 16; ++i_2) \| for (int i_4 = 0; i_4 < 32; ++i_4)` |
| load | 0.08 | 1.5% | 0.5 | 0.26 | `for (int s = 0; s < 2; ++s)` |
| load.global | 0.07 | 1.3% | 1.0 | 0.10 | `int expert = TileExpert[((int64_t)((int)blockIdx.y))];` |
| wait.sync#2 | 0.06 | 1.0% | 0.5 | 0.16 | `__syncthreads();` |
| load.global#2 | 0.04 | 0.8% | 0.5 | 0.19 | `if (((int)threadIdx.x) < 32) \| if (((int)threadIdx.x) < 64)` |
| wait.sync | 0.04 | 0.8% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#3 | 0.02 | 0.3% | 0.5 | 0.06 | `__syncthreads();` |

### gemm_mxfp8_kernel grid (28, 1, 1) block (128, 1, 1)

144 launches, 112 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 35.76 us, warp 33.98 us, nsys graphs 41.95 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 19.95 | 58.7% | 15.10 | 42.6% |
| wait | 7.13 | 21.0% | 12.63 | 35.6% |
| compute | 3.94 | 11.6% | 3.93 | 11.1% |
| load | 2.91 | 8.6% | 3.76 | 10.6% |
| store | 0.05 | 0.2% | 0.05 | 0.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*19 (est.: sampled iterations x19) | 3.76 | 11.1% | 1.0 | 4.26 | `{` |
| wait.sync#2*19 (est.: sampled iterations x19) | 3.29 | 9.7% | 1.0 | 12.77 | `__syncthreads();` |
| load.cp_async#2*19 (est.: sampled iterations x19) | 2.66 | 7.8% | 1.0 | 3.04 | `for (int i_3 = 0; i_3 < 2; ++i_3) \| for (int i_3 = 0; i_3 < 8; ++i_3)` |
| wait.cp_async*19 (est.: sampled iterations x19) | 2.43 | 7.2% | 1.0 | 14.90 | `tl::cp_async_wait<1>();` |
| wait.sync*19 (est.: sampled iterations x19) | 1.18 | 3.5% | 1.0 | 1.22 | `__syncthreads();` |
| load.cp_async | 0.25 | 0.7% | 1.0 | 0.26 | `for (int i_1 = 0; i_1 < 2; ++i_1) \| for (int i_1 = 0; i_1 < 4; ++i_1) \| for (int i_1 = 0; ` |
| compute.mma#2 | 0.18 | 0.5% | 1.0 | 0.19 | `{` |
| wait.cp_async#2 | 0.13 | 0.4% | 1.0 | 0.35 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| wait.sync#3 | 0.11 | 0.3% | 1.0 | 0.29 | `__syncthreads();` |
| store.global | 0.05 | 0.2% | 1.0 | 0.06 | `for (int i_10 = 0; i_10 < 16; ++i_10) \| for (int i_10 = 0; i_10 < 8; ++i_10) \| for (int i_` |

### mhc_project_streams_kernel grid (40, 1, 1) block (128, 1, 1)

288 launches, 160 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 17.02 us, warp 14.94 us, nsys graphs 15.62 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 7.65 | 51.2% | 4.98 | 31.0% |
| compute | 2.78 | 18.6% | 2.61 | 16.2% |
| load | 2.72 | 18.2% | 5.12 | 31.8% |
| wait | 1.34 | 9.0% | 2.86 | 17.8% |
| store | 0.46 | 3.1% | 0.52 | 3.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*4 (est.: sampled iterations x4) | 2.64 | 17.7% | 1.0 | 2.69 | `{` |
| load.global#2*4 (est.: sampled iterations x4) | 2.49 | 16.6% | 1.0 | 6.34 | `for (int i_1 = 0; i_1 < 6; ++i_1)` |
| wait.sync*4 (est.: sampled iterations x4) | 1.16 | 7.7% | 1.0 | 3.71 | `__syncthreads();` |
| store.global*4 (est.: sampled iterations x4) | 0.24 | 1.6% | 1.0 | 0.38 | `if (((((int)blockIdx.y) * 16) + (((int)threadIdx.x) >> 3)) < M)` |
| load.global | 0.23 | 1.6% | 1.0 | 0.26 | `for (int k = 0; k < 4; ++k)` |
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

### gemm_fp8w_bf16_partials_kernel grid (48, 4, 1) block (128, 1, 1)

144 launches, 768 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 24.27 us, warp 19.04 us, nsys graphs 40.96 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 7.68 | 40.3% | 11.38 | 37.0% |
| unattributed | 4.79 | 25.2% | 11.29 | 36.7% |
| load | 3.82 | 20.0% | 5.04 | 16.4% |
| compute | 2.68 | 14.1% | 2.97 | 9.7% |
| store | 0.07 | 0.3% | 0.06 | 0.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*14 (est.: sampled iterations x14) | 2.42 | 12.7% | 1.0 | 3.14 | `{` |
| wait.sync#2*14 (est.: sampled iterations x14) | 1.34 | 7.1% | 1.0 | 1.79 | `__syncthreads();` |
| load.global*14 (est.: sampled iterations x14) | 1.12 | 5.9% | 1.0 | 1.79 | `if (((int)threadIdx.x) < 4)` |
| load.cp_async#2*14 (est.: sampled iterations x14) | 1.09 | 5.7% | 1.0 | 1.34 | `for (int i_4 = 0; i_4 < 2; ++i_4) \| for (int i_6 = 0; i_6 < 2; ++i_6)` |
| load.cp_async#3*14 (est.: sampled iterations x14) | 1.06 | 5.6% | 1.0 | 1.34 | `for (int i_10 = 0; i_10 < 4; ++i_10) \| for (int i_10 = 0; i_10 < 6; ++i_10) \| for (int i_1` |
| wait.sync#4*14 (est.: sampled iterations x14) | 1.02 | 5.4% | 1.0 | 1.34 | `__syncthreads();` |
| wait.sync#3*14 (est.: sampled iterations x14) | 1.01 | 5.3% | 1.0 | 1.34 | `__syncthreads();` |
| wait.sync#5*14 (est.: sampled iterations x14) | 1.01 | 5.3% | 1.0 | 1.34 | `__syncthreads();` |
| wait.sync*14 (est.: sampled iterations x14) | 0.98 | 5.1% | 1.0 | 1.34 | `__syncthreads();` |
| wait.cp_async*14 (est.: sampled iterations x14) | 0.90 | 4.7% | 1.0 | 1.34 | `tl::cp_async_wait<3>();` |
| wait.cp_async#2*14 (est.: sampled iterations x14) | 0.87 | 4.6% | 1.0 | 1.34 | `tl::cp_async_wait<3>();` |
| load.cp_async | 0.42 | 2.2% | 1.0 | 0.67 | `for (int i_1 = 0; i_1 < 2; ++i_1)` |
| compute.mma#2 | 0.13 | 0.7% | 1.0 | 0.16 | `{` |
| compute.mma#3 | 0.13 | 0.7% | 1.0 | 0.16 | `{` |
| wait.sync#7 | 0.07 | 0.4% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#10 | 0.07 | 0.4% | 1.0 | 0.10 | `__syncthreads();` |
| store.global | 0.07 | 0.3% | 1.0 | 0.10 | `for (int i_13 = 0; i_13 < 8; ++i_13) \| for (int i_19 = 0; i_19 < 16; ++i_19) \| for (int i_` |
| load.global#2 | 0.06 | 0.3% | 1.0 | 0.10 | `if (((int)threadIdx.x) < 4)` |
| load.global#3 | 0.06 | 0.3% | 1.0 | 0.10 | `if (((int)threadIdx.x) < 4)` |
| wait.cp_async#6 | 0.06 | 0.3% | 1.0 | 0.06 | `tl::cp_async_wait<0>();` |
| wait.sync#8 | 0.05 | 0.3% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#9 | 0.05 | 0.3% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#6 | 0.05 | 0.3% | 1.0 | 0.06 | `__syncthreads();` |
| wait.cp_async#3 | 0.05 | 0.3% | 1.0 | 0.06 | `tl::cp_async_wait<3>();` |
| wait.sync#11 | 0.05 | 0.3% | 1.0 | 0.06 | `__syncthreads();` |
| wait.cp_async#5 | 0.05 | 0.3% | 1.0 | 0.06 | `tl::cp_async_wait<1>();` |
| wait.cp_async#4 | 0.05 | 0.2% | 1.0 | 0.06 | `tl::cp_async_wait<2>();` |

### gemm_mxfp8_kernel grid (24, 1, 1) block (128, 1, 1)

144 launches, 96 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 30.61 us, warp 28.96 us, nsys graphs 39.46 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 10.75 | 37.1% | 9.70 | 29.9% |
| wait | 7.73 | 26.7% | 10.38 | 32.0% |
| compute | 6.02 | 20.8% | 5.95 | 18.3% |
| load | 4.40 | 15.2% | 6.37 | 19.6% |
| store | 0.05 | 0.2% | 0.05 | 0.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*38 (est.: sampled iterations x38) | 5.78 | 19.9% | 1.0 | 6.08 | `{` |
| load.cp_async#2*38 (est.: sampled iterations x38) | 4.14 | 14.3% | 1.0 | 4.86 | `for (int i_5 = 0; i_5 < 2; ++i_5) \| for (int i_5 = 0; i_5 < 4; ++i_5) \| tl::cp_async_gs_co` |
| wait.sync#2*38 (est.: sampled iterations x38) | 2.89 | 10.0% | 1.0 | 3.65 | `__syncthreads();` |
| wait.sync*38 (est.: sampled iterations x38) | 2.36 | 8.1% | 1.0 | 2.43 | `__syncthreads();` |
| wait.cp_async*38 (est.: sampled iterations x38) | 2.28 | 7.9% | 1.0 | 2.43 | `tl::cp_async_wait<2>();` |
| load.cp_async | 0.26 | 0.9% | 1.0 | 0.30 | `for (int i_1 = 0; i_1 < 2; ++i_1) \| for (int i_1 = 0; i_1 < 4; ++i_1) \| for (int i_1 = 0; ` |
| compute.mma#3 | 0.13 | 0.4% | 1.0 | 0.13 | `{` |
| compute.mma#2 | 0.12 | 0.4% | 1.0 | 0.13 | `{` |
| wait.cp_async#3 | 0.06 | 0.2% | 1.0 | 0.06 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>();` |
| store.global | 0.05 | 0.2% | 1.0 | 0.06 | `for (int i_10 = 0; i_10 < 16; ++i_10) \| for (int i_10 = 0; i_10 < 8; ++i_10) \| for (int i_` |
| wait.sync#4 | 0.05 | 0.2% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#3 | 0.05 | 0.2% | 1.0 | 0.06 | `__syncthreads();` |
| wait.cp_async#2 | 0.05 | 0.2% | 1.0 | 0.06 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |

### gemm_bf16_partials_kernel grid (6, 8, 1) block (256, 1, 1)

96 launches, 384 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 23.26 us, warp 8.29 us, nsys graphs 22.34 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 5.92 | 71.4% | 1.45 | 34.2% |
| wait | 1.26 | 15.2% | 1.38 | 32.4% |
| compute | 0.68 | 8.2% | 1.36 | 32.0% |
| load | 0.39 | 4.7% | 0.00 | 0.0% |
| store | 0.04 | 0.5% | 0.07 | 1.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*10 (est.: sampled iterations x10) | 0.68 | 8.2% | 0.5 | 1.60 | `{` |
| wait.mbarrier*10 (est.: sampled iterations x10) | 0.66 | 8.0% | 0.5 | 1.60 | `mbarrier[((kb % 3) + 3)].wait((((kb % 6) / 3) ^ 1));` |
| wait.mbarrier#2*10 (est.: sampled iterations x10) | 0.51 | 6.2% | 0.5 | 1.76 | `mbarrier[(kb_1 % 3)].wait(((kb_1 % 6) / 3));` |
| load.tma*10 (est.: sampled iterations x10) | 0.39 | 4.7% | 0.5 | 1.28 | `if (tl::tl_shuffle_elect<128>())` |
| wait.sync | 0.09 | 1.1% | 1.0 | 0.13 | `__syncthreads();` |
| store.global | 0.04 | 0.5% | 0.5 | 0.10 | `for (int i_1 = 0; i_1 < 4; ++i_1) \| for (int i_2 = 0; i_2 < 8; ++i_2) \| for (int i_4 = 0; ` |

### mhc_finalize_kernel grid (5, 1, 1) block (128, 1, 1)

180 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 12.24 us, warp 11.19 us, nsys graphs 9.46 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 6.85 | 61.2% | 5.49 | 47.0% |
| store | 1.50 | 13.4% | 2.35 | 20.1% |
| load | 1.06 | 9.5% | 1.84 | 15.7% |
| compute | 1.01 | 9.0% | 1.16 | 9.9% |
| wait | 0.76 | 6.8% | 0.84 | 7.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync | 0.76 | 6.8% | 1.0 | 1.02 | `__syncthreads();` |
| store.global | 0.61 | 5.4% | 0.2 | 2.43 | `if (((int)threadIdx.x) < 4)` |
| compute.reduce | 0.57 | 5.1% | 0.2 | 2.30 | `row_max[0] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<32>>::run(row_max[0]);` |
| store.global#4*7 (est.: sampled iterations x7) | 0.56 | 5.0% | 0.8 | 0.90 | `if ((((int)threadIdx.x) % 4) == 0)` |
| compute.reduce#2*7 (est.: sampled iterations x7) | 0.44 | 3.9% | 0.8 | 0.67 | `amax[0] = max(amax[0], __shfl_xor_sync((uint)4294967295, amax[0], 1, 32));` |
| load.global | 0.38 | 3.4% | 1.0 | 1.17 | `if (((int)threadIdx.x) < 32)` |
| load.global#4*7 (est.: sampled iterations x7) | 0.36 | 3.2% | 0.8 | 0.67 | `*(uint4*)(yv + 0) = *(uint4*)(C + ((((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((i` |
| store.global#3*7 (est.: sampled iterations x7) | 0.32 | 2.9% | 0.8 | 0.45 | `*(uint4*)(y + ((((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((int64_t)step) * (int6` |
| load.global#3 | 0.27 | 2.4% | 0.8 | 0.38 | `for (int i_split_2 = 0; i_split_2 < 40; ++i_split_2)` |
| load.global#2 | 0.05 | 0.4% | 0.2 | 0.19 | `cm[0] = ((mixes_shared[((((int)threadIdx.x) & 15) + 8)] * scale[2]) + base[((((int)threadI` |
| store.global#2 | 0.01 | 0.1% | 0.2 | 0.06 | `if ((((int)threadIdx.x) >> 4) == 0)` |

### gemm_bf16_splitk_kernel grid (16, 5, 1) block (128, 1, 1)

32 launches, 320 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 54.70 us, warp 47.61 us, nsys graphs 49.47 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 22.71 | 47.7% | 45.59 | 88.2% |
| unattributed | 20.54 | 43.1% | 0.00 | 0.0% |
| load | 2.27 | 4.8% | 4.10 | 7.9% |
| compute | 1.97 | 4.1% | 2.15 | 4.2% |
| store | 0.13 | 0.3% | 0.15 | 0.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.cp_async*13 (est.: sampled iterations x13) | 10.58 | 22.2% | 0.5 | 22.88 | `tl::cp_async_wait<2>();` |
| wait.cp_async#2 | 1.92 | 4.0% | 1.0 | 4.29 | `tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>(); \| tl::cp_async_wait<3>();` |
| wait.sync*12 (est.: sampled iterations x12) | 1.84 | 3.9% | 0.5 | 5.76 | `__syncthreads();` |
| wait.sync*13 (est.: sampled iterations x13) | 1.75 | 3.7% | 0.5 | 5.41 | `__syncthreads();` |
| wait.cp_async#4 | 1.68 | 3.5% | 1.0 | 4.48 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>();` |
| load.cp_async | 1.09 | 2.3% | 1.0 | 3.55 | `for (int i_1 = 0; i_1 < 2; ++i_1) \| tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)A_sha` |
| wait.cp_async*12 (est.: sampled iterations x12) | 0.96 | 2.0% | 0.5 | 5.95 | `tl::cp_async_wait<3>();` |
| wait.cp_async#3 | 0.93 | 2.0% | 1.0 | 3.23 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| compute.mma*13 (est.: sampled iterations x13) | 0.83 | 1.7% | 0.5 | 0.83 | `{` |
| compute.mma*12 (est.: sampled iterations x12) | 0.78 | 1.6% | 0.5 | 0.96 | `{` |
| wait.sync#5 | 0.69 | 1.4% | 1.0 | 2.13 | `__syncthreads();` |
| wait.sync#3 | 0.60 | 1.3% | 1.0 | 1.94 | `__syncthreads();` |
| wait.sync#4 | 0.55 | 1.2% | 1.0 | 1.63 | `__syncthreads();` |
| load.cp_async#2*12 (est.: sampled iterations x12) | 0.48 | 1.0% | 0.5 | 0.58 | `tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)A_shared)[((((((kb & 3) * 1024) + ((((int` |
| load.cp_async#2*13 (est.: sampled iterations x13) | 0.43 | 0.9% | 0.5 | 0.62 | `for (int i_10 = 0; i_10 < 2; ++i_10) \| for (int i_8 = 0; i_8 < 2; ++i_8) \| tl::cp_async_gs` |
| wait.sync#2*13 (est.: sampled iterations x13) | 0.34 | 0.7% | 0.5 | 0.42 | `__syncthreads();` |
| wait.sync#2*12 (est.: sampled iterations x12) | 0.30 | 0.6% | 0.5 | 0.38 | `__syncthreads();` |
| load.global | 0.27 | 0.6% | 1.0 | 1.65 | `if (((int)threadIdx.x) == 0)` |
| wait.sync#7 | 0.22 | 0.5% | 1.0 | 0.42 | `__syncthreads();` |
| wait.sync#8 | 0.18 | 0.4% | 0.5 | 0.29 | `__syncthreads();` |
| compute.mma#2 | 0.11 | 0.2% | 1.0 | 0.13 | `{` |
| wait.sync#6 | 0.11 | 0.2% | 1.0 | 0.42 | `__syncthreads();` |
| compute.mma#3 | 0.10 | 0.2% | 1.0 | 0.13 | `{` |
| compute.mma#4 | 0.10 | 0.2% | 1.0 | 0.13 | `{` |
| store.global#3 | 0.07 | 0.1% | 1.0 | 0.10 | `if (((int*)last)[0] == 1)` |
| store.global | 0.06 | 0.1% | 1.0 | 0.06 | `for (int i_10 = 0; i_10 < 4; ++i_10) \| for (int i_16 = 0; i_16 < 4; ++i_16) \| for (int i_1` |
| wait.cp_async#5 | 0.06 | 0.1% | 0.5 | 0.35 | `tl::cp_async_wait<0>();` |
| compute.mma#5 | 0.05 | 0.1% | 0.5 | 0.05 | `{` |

### sparse_mla_kernel grid (5, 1, 5) block (256, 1, 1)

60 launches, 200 warps; ranges, warp and last warp from the 5 recording CTAs' 40 warps; span 17.79 us, warp 16.76 us, nsys graphs 13.22 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed, with loops' unsampled iterations | 10.58 | 63.2% | 10.91 | 48.0% |
| wait | 2.59 | 15.5% | 4.79 | 21.1% |
| compute | 1.47 | 8.8% | 1.48 | 6.5% |
| load | 1.45 | 8.6% | 4.75 | 20.9% |
| store | 0.67 | 4.0% | 0.81 | 3.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync (sampled iterations) | 2.52 | 15.0% | 11.0 | 3.34 | `__syncthreads();` |
| compute.mma (sampled iterations) | 1.22 | 7.3% | 2.0 | 1.28 | `for (int ki = 0; ki < 32; ++ki) \| for (int ki_1 = 0; ki_1 < 2; ++ki_1)` |
| load.global (sampled iterations) | 1.15 | 6.8% | 2.0 | 3.73 | `for (int c = 0; c < 4; ++c) \| for (int i = 0; i < 8; ++i) \| for (int i_1 = 0; i_1 < 8; ++i` |
| store.global | 0.67 | 4.0% | 1.0 | 0.83 | `for (int i_10 = 0; i_10 < 32; ++i_10) \| for (int i_11 = 0; i_11 < 32; ++i_11) \| for (int i` |
| load.cp_async (sampled iterations) | 0.30 | 1.8% | 2.0 | 0.42 | `for (int k_1 = 0; k_1 < 5; ++k_1) \| for (int k_1 = 0; k_1 < 9; ++k_1) \| for (int k_3 = 0; ` |
| compute.reduce (sampled iterations) | 0.25 | 1.5% | 2.0 | 0.26 | `m_g[i_5] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(m_g[i_5], (&((` |
| wait.cp_async (sampled iterations) | 0.07 | 0.4% | 1.0 | 0.10 | `tl::cp_async_wait<0>();` |

### mhc_finalize_kernel grid (10, 1, 1) block (128, 1, 1)

96 launches, 40 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 12.78 us, warp 12.04 us, nsys graphs 7.28 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 7.46 | 61.9% | 6.23 | 51.6% |
| store | 1.59 | 13.2% | 1.95 | 16.2% |
| load | 1.10 | 9.2% | 1.80 | 14.9% |
| compute | 1.00 | 8.3% | 0.98 | 8.1% |
| wait | 0.89 | 7.4% | 1.11 | 9.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync | 0.89 | 7.4% | 1.0 | 1.22 | `__syncthreads();` |
| store.global | 0.69 | 5.7% | 0.2 | 2.80 | `if (((int)threadIdx.x) < 4)` |
| compute.reduce | 0.57 | 4.8% | 0.2 | 2.30 | `row_max[0] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<32>>::run(row_max[0]);` |
| store.global#4*7 (est.: sampled iterations x7) | 0.56 | 4.6% | 0.8 | 0.90 | `if ((((int)threadIdx.x) % 4) == 0)` |
| load.global | 0.45 | 3.7% | 1.0 | 1.44 | `if (((int)threadIdx.x) < 32)` |
| compute.reduce#2*7 (est.: sampled iterations x7) | 0.43 | 3.5% | 0.8 | 0.67 | `amax[0] = max(amax[0], __shfl_xor_sync((uint)4294967295, amax[0], 1, 32));` |
| load.global#4*7 (est.: sampled iterations x7) | 0.36 | 3.0% | 0.8 | 0.67 | `*(uint4*)(yv + 0) = *(uint4*)(C + ((((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((i` |
| store.global#3*7 (est.: sampled iterations x7) | 0.32 | 2.7% | 0.8 | 0.67 | `*(uint4*)(y + ((((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((int64_t)step) * (int6` |
| load.global#3 | 0.25 | 2.1% | 0.8 | 0.35 | `for (int i_split_2 = 0; i_split_2 < 40; ++i_split_2)` |
| load.global#2 | 0.05 | 0.4% | 0.2 | 0.19 | `cm[0] = ((mixes_shared[((((int)threadIdx.x) & 15) + 8)] * scale[2]) + base[((((int)threadI` |
| store.global#2 | 0.01 | 0.1% | 0.2 | 0.06 | `if ((((int)threadIdx.x) >> 4) == 0)` |

### sparse_mla_kernel grid (2, 2, 10) block (128, 1, 1)

48 launches, 160 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 17.41 us, warp 16.37 us, nsys graphs 18.74 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed, with loops' unsampled iterations | 6.23 | 38.1% | 6.07 | 33.4% |
| compute | 4.69 | 28.6% | 4.65 | 25.5% |
| wait | 2.69 | 16.4% | 3.22 | 17.7% |
| load | 2.42 | 14.8% | 3.81 | 20.9% |
| store | 0.34 | 2.1% | 0.45 | 2.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*4 (est.: sampled iterations x4) | 1.73 | 10.6% | 1.0 | 1.79 | `for (int ki = 0; ki < 32; ++ki) \| {` |
| load.global#2 | 1.24 | 7.6% | 1.0 | 2.26 | `for (int i = 0; i < 8; ++i)` |
| compute.mma#2*4 (est.: sampled iterations x4) | 1.20 | 7.3% | 1.0 | 1.28 | `for (int ki_1 = 0; ki_1 < 2; ++ki_1) \| if (((((int)blockIdx.x) * 2) + (c_1 >> 1)) < 3)` |
| compute.reduce#2*8 (est.: sampled iterations x8) | 0.90 | 5.5% | 1.0 | 1.02 | `row_sum[i_7] = tl::AllReduce<tl::SumOp, 128, 32, 0, tl::NamedBarrier<128>>::run(row_sum[i_` |
| compute.reduce*8 (est.: sampled iterations x8) | 0.86 | 5.3% | 1.0 | 1.02 | `m_g[i_4] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(m_g[i_4], (&((` |
| load.cp_async#2*4 (est.: sampled iterations x4) | 0.70 | 4.3% | 1.0 | 0.77 | `if (((((int)blockIdx.x) * 2) + (c_1 >> 1)) < 3) \| if ((c_1 < 3) & (c_1 < 3)) \| if (c < 3)` |
| store.global | 0.34 | 2.1% | 1.0 | 0.45 | `for (int i_10 = 0; i_10 < 32; ++i_10) \| for (int i_11 = 0; i_11 < 32; ++i_11) \| for (int i` |
| wait.sync#10*8 (est.: sampled iterations x8) | 0.34 | 2.1% | 1.0 | 0.51 | `__syncthreads();` |
| load.cp_async (sampled iterations) | 0.33 | 2.0% | 1.0 | 0.35 | `for (int k_1 = 0; k_1 < 5; ++k_1) \| for (int k_1 = 0; k_1 < 9; ++k_1) \| for (int k_3 = 0; ` |
| wait.sync#8*8 (est.: sampled iterations x8) | 0.33 | 2.0% | 1.0 | 0.51 | `__syncthreads();` |
| wait.cp_async*4 (est.: sampled iterations x4) | 0.27 | 1.7% | 1.0 | 0.38 | `tl::cp_async_wait<0>();` |
| wait.sync#5*4 (est.: sampled iterations x4) | 0.26 | 1.6% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#4*4 (est.: sampled iterations x4) | 0.25 | 1.5% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#6*4 (est.: sampled iterations x4) | 0.25 | 1.5% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#7*4 (est.: sampled iterations x4) | 0.24 | 1.5% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#9*4 (est.: sampled iterations x4) | 0.24 | 1.5% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#11*4 (est.: sampled iterations x4) | 0.24 | 1.4% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync (sampled iterations) | 0.18 | 1.1% | 1.0 | 0.26 | `__syncthreads();` |
| load.global (sampled iterations) | 0.14 | 0.9% | 1.0 | 0.29 | `for (int c = 0; c < 4; ++c) \| for (int i = 0; i < 8; ++i) \| for (int i_1 = 0; i_1 < 8; ++i` |
| wait.sync#3 | 0.06 | 0.4% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#2 | 0.04 | 0.3% | 1.0 | 0.06 | `__syncthreads();` |

### gemm_mxfp8_kernel grid (64, 1, 1) block (128, 1, 1)

32 launches, 256 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 23.55 us, warp 15.47 us, nsys graphs 23.39 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 10.10 | 65.3% | 2.54 | 30.1% |
| unattributed | 3.81 | 24.6% | 3.96 | 47.0% |
| compute | 0.88 | 5.7% | 0.88 | 10.4% |
| load | 0.63 | 4.1% | 0.98 | 11.7% |
| store | 0.06 | 0.4% | 0.06 | 0.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.cp_async*2 (est.: sampled iterations x2) | 2.98 | 19.2% | 1.0 | 6.78 | `tl::cp_async_wait<3>();` |
| wait.cp_async#3 | 1.57 | 10.2% | 1.0 | 3.38 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>();` |
| wait.cp_async#4 | 1.53 | 9.9% | 1.0 | 3.31 | `tl::cp_async_wait<0>();` |
| wait.sync#2*2 (est.: sampled iterations x2) | 1.24 | 8.0% | 1.0 | 4.06 | `__syncthreads();` |
| wait.cp_async#2 | 1.08 | 7.0% | 1.0 | 2.96 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>(); \| tl::cp_async_wait<2>();` |
| wait.sync#3 | 0.60 | 3.9% | 1.0 | 2.14 | `__syncthreads();` |
| wait.sync#4 | 0.53 | 3.5% | 1.0 | 2.02 | `__syncthreads();` |
| wait.sync#5 | 0.47 | 3.0% | 1.0 | 1.42 | `__syncthreads();` |
| load.cp_async | 0.40 | 2.6% | 1.0 | 0.51 | `for (int i_1 = 0; i_1 < 2; ++i_1) \| for (int i_1 = 0; i_1 < 4; ++i_1) \| for (int i_1 = 0; ` |
| compute.mma*2 (est.: sampled iterations x2) | 0.36 | 2.3% | 1.0 | 0.38 | `{` |
| load.cp_async#2*2 (est.: sampled iterations x2) | 0.23 | 1.5% | 1.0 | 0.38 | `for (int i_7 = 0; i_7 < 2; ++i_7)` |
| compute.mma#2 | 0.17 | 1.1% | 1.0 | 0.19 | `{` |
| compute.mma#3 | 0.17 | 1.1% | 1.0 | 0.19 | `{` |
| compute.mma#4 | 0.17 | 1.1% | 1.0 | 0.19 | `{` |
| wait.sync*2 (est.: sampled iterations x2) | 0.10 | 0.6% | 1.0 | 0.13 | `__syncthreads();` |
| store.global | 0.06 | 0.4% | 1.0 | 0.06 | `for (int i_10 = 0; i_10 < 16; ++i_10) \| for (int i_10 = 0; i_10 < 8; ++i_10) \| for (int i_` |

### moe_dispatch_kernel grid (1, 1, 1) block (512, 1, 1)

144 launches, 16 warps; ranges, warp and last warp from the 1 recording CTAs' 16 warps; span 4.70 us, warp 4.40 us, nsys graphs 2.78 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 2.52 | 57.2% | 1.73 | 35.2% |
| unattributed, with loops' unsampled iterations | 1.20 | 27.3% | 1.84 | 37.4% |
| store | 0.60 | 13.6% | 1.25 | 25.5% |
| load | 0.08 | 1.9% | 0.10 | 2.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync#7*9 (est.: sampled iterations x9) | 0.65 | 14.7% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#6*9 (est.: sampled iterations x9) | 0.61 | 13.9% | 1.0 | 0.86 | `__syncthreads();` |
| store.global (sampled iterations) | 0.52 | 11.8% | 1.0 | 0.93 | `ChunkCount[((((int64_t)c) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = count; \| Chu` |
| wait.sync#8 | 0.49 | 11.1% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync#2 | 0.39 | 8.8% | 1.0 | 0.42 | `__syncthreads();` |
| wait.sync#3 | 0.13 | 3.0% | 1.0 | 0.32 | `__syncthreads();` |
| load.global (sampled iterations) | 0.08 | 1.9% | 1.0 | 0.10 | `expert = ((int)TopkIdx[((((int64_t)((int)blockIdx.x)) * (int64_t)512) + ((int64_t)((int)th` |
| store.global#2 | 0.08 | 1.8% | 1.0 | 0.29 | `RoutePos[((((int64_t)((int)blockIdx.x)) * (int64_t)512) + ((int64_t)((int)threadIdx.x)))] ` |
| wait.sync | 0.07 | 1.5% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#9 | 0.06 | 1.4% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#4 | 0.06 | 1.4% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#5 | 0.06 | 1.4% | 1.0 | 0.06 | `__syncthreads();` |

### moe_grouped_fp8_fp4_gemm_kernel grid (12, 36, 1) block (128, 1, 1)

6 launches, 1728 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 112.40 us, warp 18.06 us, nsys graphs 3.50 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 33.14 | 183.4% | 0.06 | 18.3% |
| compute | 3.32 | 18.4% | 0.00 | 0.0% |
| load | 2.30 | 12.7% | 0.06 | 18.3% |
| store | 0.07 | 0.4% | 0.00 | 0.0% |
| unattributed | 0.00 | 0.0% | 0.20 | 63.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.mbarrier*40 (est.: sampled iterations x40) | 29.92 | 165.6% | 0.2 | 119.68 | `loaded[(ko % 3)].wait(((ko % 6) / 3));` |
| compute.mma*40 (est.: sampled iterations x40) | 3.32 | 18.4% | 0.2 | 14.08 | `{` |
| load#2*40 (est.: sampled iterations x40) | 1.52 | 8.4% | 0.2 | 8.96 | `if (ko < 38)` |
| wait.sync#5*40 (est.: sampled iterations x40) | 1.28 | 7.1% | 0.2 | 6.40 | `__syncthreads();` |
| wait.sync#4*40 (est.: sampled iterations x40) | 0.64 | 3.5% | 0.2 | 2.56 | `__syncthreads();` |
| wait.sync#6*40 (est.: sampled iterations x40) | 0.64 | 3.5% | 0.2 | 2.56 | `__syncthreads();` |
| wait.cp_async*40 (est.: sampled iterations x40) | 0.56 | 3.1% | 0.2 | 2.56 | `tl::cp_async_wait<2>();` |
| load.global | 0.37 | 2.0% | 1.0 | 1.28 | `int expert = TileExpert[((int64_t)((int)blockIdx.y))];` |
| load.global#3 | 0.35 | 1.9% | 0.2 | 1.41 | `if (((expert < 128) && (0 <= expert))) \| if (((expert < 384) && (0 <= expert)))` |
| store.global | 0.07 | 0.4% | 0.2 | 0.35 | `for (int i_2 = 0; i_2 < 16; ++i_2) \| for (int i_4 = 0; i_4 < 32; ++i_4)` |
| wait.sync | 0.05 | 0.3% | 1.0 | 0.06 | `__syncthreads();` |
| load | 0.04 | 0.2% | 0.2 | 0.22 | `for (int s = 0; s < 2; ++s)` |
| wait.sync#2 | 0.03 | 0.2% | 0.2 | 0.16 | `__syncthreads();` |
| load.global#2 | 0.02 | 0.1% | 0.2 | 0.19 | `if (((int)threadIdx.x) < 32) \| if (((int)threadIdx.x) < 64)` |
| wait.sync#3 | 0.01 | 0.1% | 0.2 | 0.06 | `__syncthreads();` |

### sparse_mla_kernel grid (1, 2, 5) block (128, 1, 1)

30 launches, 40 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 22.38 us, warp 22.12 us, nsys graphs 25.25 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed, with loops' unsampled iterations | 8.45 | 38.2% | 7.06 | 31.9% |
| compute | 4.69 | 21.2% | 4.73 | 21.4% |
| load | 4.64 | 21.0% | 6.52 | 29.5% |
| wait | 3.91 | 17.7% | 3.32 | 15.0% |
| store | 0.43 | 1.9% | 0.50 | 2.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 | 2.97 | 13.4% | 1.0 | 3.34 | `for (int i = 0; i < 8; ++i)` |
| compute.mma*4 (est.: sampled iterations x4) | 1.74 | 7.9% | 1.0 | 1.79 | `for (int ki = 0; ki < 32; ++ki) \| {` |
| wait.sync (sampled iterations) | 1.42 | 6.4% | 1.0 | 1.90 | `__syncthreads();` |
| compute.mma#2*4 (est.: sampled iterations x4) | 1.16 | 5.2% | 1.0 | 1.28 | `for (int ki_1 = 0; ki_1 < 2; ++ki_1) \| if (((((int)blockIdx.x) * 2) + (c_1 >> 1)) < 3)` |
| compute.reduce*8 (est.: sampled iterations x8) | 0.92 | 4.2% | 1.0 | 1.02 | `m_g[i_4] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(m_g[i_4], (&((` |
| compute.reduce#2*8 (est.: sampled iterations x8) | 0.87 | 3.9% | 1.0 | 1.02 | `row_sum[i_7] = tl::AllReduce<tl::SumOp, 128, 32, 0, tl::NamedBarrier<128>>::run(row_sum[i_` |
| load.global (sampled iterations) | 0.67 | 3.0% | 1.0 | 1.92 | `for (int c = 0; c < 4; ++c) \| for (int i = 0; i < 8; ++i) \| for (int i_1 = 0; i_1 < 8; ++i` |
| load.cp_async#2*4 (est.: sampled iterations x4) | 0.65 | 3.0% | 1.0 | 0.77 | `if (((((int)blockIdx.x) * 2) + (c_1 >> 1)) < 3) \| if ((c_1 < 3) & (c_1 < 3)) \| if (c < 3)` |
| store.global | 0.43 | 1.9% | 1.0 | 0.51 | `for (int i_10 = 0; i_10 < 32; ++i_10) \| for (int i_11 = 0; i_11 < 32; ++i_11) \| for (int i` |
| load.cp_async (sampled iterations) | 0.35 | 1.6% | 1.0 | 0.38 | `for (int k_1 = 0; k_1 < 5; ++k_1) \| for (int k_1 = 0; k_1 < 9; ++k_1) \| for (int k_3 = 0; ` |
| wait.sync#8*8 (est.: sampled iterations x8) | 0.32 | 1.4% | 1.0 | 0.51 | `__syncthreads();` |
| wait.sync#10*8 (est.: sampled iterations x8) | 0.31 | 1.4% | 1.0 | 0.51 | `__syncthreads();` |
| wait.cp_async*4 (est.: sampled iterations x4) | 0.27 | 1.2% | 1.0 | 0.38 | `tl::cp_async_wait<0>();` |
| wait.sync#5*4 (est.: sampled iterations x4) | 0.26 | 1.2% | 1.0 | 0.38 | `__syncthreads();` |
| wait.sync#4*4 (est.: sampled iterations x4) | 0.26 | 1.2% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#7*4 (est.: sampled iterations x4) | 0.26 | 1.2% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#11*4 (est.: sampled iterations x4) | 0.24 | 1.1% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#9*4 (est.: sampled iterations x4) | 0.24 | 1.1% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#6*4 (est.: sampled iterations x4) | 0.23 | 1.0% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#3 | 0.06 | 0.3% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#2 | 0.04 | 0.2% | 1.0 | 0.06 | `__syncthreads();` |

### metadata_chunk_kernel grid (10, 1, 1) block (192, 1, 1)

48 launches, 60 warps; ranges, warp and last warp from the 5 recording CTAs' 30 warps; span 14.75 us, warp 8.45 us, nsys graphs 11.58 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 5.86 | 69.4% | 10.64 | 79.4% |
| unattributed | 2.52 | 29.8% | 2.64 | 19.7% |
| store | 0.07 | 0.8% | 0.13 | 1.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 5.73 | 67.8% | 1.0 | 9.46 | `if ((((offset + ((int)blockIdx.x)) < src_rows) && (0 <= (offset + ((int)blockIdx.x)))))` |
| load.global#2 | 0.13 | 1.6% | 0.2 | 1.28 | `if (((0 <= req) & ((int64_t)0 <= pos)))` |
| store.global | 0.05 | 0.6% | 1.0 | 0.10 | `swa[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = con` |
| store.global#3 | 0.01 | 0.1% | 0.2 | 0.06 | `top_lengths[((int64_t)((int)blockIdx.x))] = condval_12; \| top_lengths[((int64_t)((int)bloc` |
| store.global#2 | 0.01 | 0.1% | 0.2 | 0.06 | `lengths[((int64_t)((int)blockIdx.x))] = ((int)max(info[1], (int64_t)0));` |

### partial_argmax_kernel grid (2, 64, 1) block (128, 1, 1)

80 launches, 512 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 7.31 us, warp 6.04 us, nsys graphs 7.10 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 5.27 | 87.2% | 6.07 | 82.7% |
| wait | 0.30 | 5.0% | 0.34 | 4.6% |
| load | 0.27 | 4.5% | 0.69 | 9.4% |
| compute | 0.12 | 2.0% | 0.12 | 1.6% |
| store | 0.08 | 1.3% | 0.13 | 1.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync | 0.30 | 5.0% | 1.0 | 1.07 | `__syncthreads();` |
| load.global | 0.27 | 4.5% | 1.0 | 0.42 | `for (int j = 0; j < 16; ++j) \| for (int run = 0; run < 2; ++run)` |
| compute.reduce | 0.12 | 2.0% | 1.0 | 0.13 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| store.global | 0.08 | 1.3% | 1.0 | 0.13 | `if (((int)threadIdx.x) == 0)` |

### gemm_mxfp8_partials_kernel grid (16, 5, 1) block (128, 1, 1)

48 launches, 320 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 11.76 us, warp 11.26 us, nsys graphs 12.58 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 5.31 | 47.2% | 5.84 | 48.9% |
| wait | 4.83 | 42.9% | 3.75 | 31.4% |
| compute | 0.61 | 5.4% | 0.63 | 5.3% |
| load | 0.46 | 4.1% | 1.66 | 13.9% |
| store | 0.05 | 0.5% | 0.05 | 0.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.cp_async#3 | 1.55 | 13.8% | 1.0 | 3.44 | `tl::cp_async_wait<0>();` |
| wait.cp_async#2 | 0.99 | 8.8% | 1.0 | 2.13 | `tl::cp_async_wait<0>(); \| tl::cp_async_wait<1>();` |
| wait.sync#2*2 (est.: sampled iterations x2) | 0.87 | 7.7% | 1.0 | 3.07 | `__syncthreads();` |
| wait.cp_async*2 (est.: sampled iterations x2) | 0.52 | 4.6% | 1.0 | 3.04 | `tl::cp_async_wait<2>();` |
| wait.sync#3 | 0.44 | 3.9% | 1.0 | 1.39 | `__syncthreads();` |
| wait.sync#4 | 0.36 | 3.2% | 1.0 | 1.09 | `__syncthreads();` |
| compute.mma*2 (est.: sampled iterations x2) | 0.31 | 2.7% | 1.0 | 0.32 | `{` |
| load.cp_async | 0.24 | 2.1% | 1.0 | 0.35 | `for (int i = 0; i < 2; ++i) \| for (int i_1 = 0; i_1 < 4; ++i_1)` |
| load.cp_async#2*2 (est.: sampled iterations x2) | 0.22 | 2.0% | 1.0 | 0.26 | `for (int i_4 = 0; i_4 < 2; ++i_4)` |
| compute.mma#2 | 0.15 | 1.3% | 1.0 | 0.16 | `{` |
| compute.mma#3 | 0.15 | 1.3% | 1.0 | 0.16 | `{` |
| wait.sync*2 (est.: sampled iterations x2) | 0.09 | 0.8% | 1.0 | 0.13 | `__syncthreads();` |
| store.global | 0.05 | 0.5% | 1.0 | 0.06 | `for (int i_11 = 0; i_11 < 16; ++i_11) \| for (int i_13 = 0; i_13 < 8; ++i_13) \| for (int i_` |

### gemm_bf16_partials_kernel grid (2, 20, 1) block (256, 1, 1)

48 launches, 320 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 11.22 us, warp 5.81 us, nsys graphs 15.52 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 4.46 | 76.8% | 1.47 | 55.5% |
| wait | 0.65 | 11.2% | 0.66 | 24.7% |
| load | 0.43 | 7.3% | 0.00 | 0.0% |
| compute | 0.22 | 3.9% | 0.45 | 17.0% |
| store | 0.05 | 0.8% | 0.07 | 2.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.mbarrier#2*4 (est.: sampled iterations x4) | 0.43 | 7.4% | 0.5 | 1.47 | `mbarrier[(kb_1 % 3)].wait((kb_1 / 3));` |
| load.tma*4 (est.: sampled iterations x4) | 0.43 | 7.3% | 0.5 | 1.09 | `if (tl::tl_shuffle_elect<128>())` |
| compute.mma*4 (est.: sampled iterations x4) | 0.22 | 3.9% | 0.5 | 0.51 | `{` |
| wait.mbarrier*4 (est.: sampled iterations x4) | 0.12 | 2.1% | 0.5 | 0.26 | `mbarrier[((kb % 3) + 3)].wait(((kb / 3) ^ 1));` |
| wait.sync | 0.10 | 1.7% | 1.0 | 0.13 | `__syncthreads();` |
| store.global | 0.05 | 0.8% | 0.5 | 0.13 | `for (int i_1 = 0; i_1 < 4; ++i_1) \| for (int i_2 = 0; i_2 < 8; ++i_2) \| for (int i_4 = 0; ` |

### router_partials_kernel grid (3, 1, 1) block (64, 1, 1)

96 launches, 6 warps; ranges, warp and last warp from the 3 recording CTAs' 6 warps; span 5.06 us, warp 3.03 us, nsys graphs 4.42 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.17 | 38.5% | 2.06 | 35.7% |
| load | 1.06 | 34.8% | 2.16 | 37.4% |
| compute | 0.68 | 22.3% | 1.35 | 23.4% |
| store | 0.13 | 4.4% | 0.20 | 3.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.reduce | 0.68 | 22.3% | 0.5 | 1.38 | `for (int k = 0; k < 3; ++k) \| for (int k = 0; k < 6; ++k)` |
| load.global#4 | 0.50 | 16.3% | 0.5 | 1.06 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < image_len)) \| if ((((((int)bl` |
| load.global | 0.45 | 14.9% | 0.8 | 0.54 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < mask_len))` |
| load.global#2 | 0.05 | 1.8% | 0.8 | 0.06 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < mask_len))` |
| load.global#3 | 0.05 | 1.8% | 0.8 | 0.06 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < mask_len))` |
| store.global | 0.05 | 1.6% | 0.8 | 0.06 | `if (condval == (bool)0) \| if (condval == (bool)1)` |
| store.global#3 | 0.05 | 1.6% | 0.5 | 0.10 | `if ((((int)threadIdx.x) & 31) < 3) \| if ((((int)threadIdx.x) & 31) < 6)` |
| store.global#2 | 0.04 | 1.2% | 0.8 | 0.06 | `if (condval_1 == (bool)0) \| if (condval_1 == (bool)1)` |

### moe_swiglu_cast_kernel_kernel grid (16, 30, 1) block (96, 1, 1)

138 launches, 1440 warps; ranges, warp and last warp from the 4 recording CTAs' 12 warps; span 3.81 us, warp 1.45 us, nsys graphs 4.29 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.14 | 78.2% | 3.18 | 82.2% |
| load | 0.21 | 14.5% | 0.40 | 10.3% |
| compute | 0.06 | 4.0% | 0.16 | 4.0% |
| store | 0.05 | 3.3% | 0.13 | 3.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.21 | 14.5% | 1.0 | 0.29 | `for (int r = 0; r < 2; ++r)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.06 | 4.0% | 0.5 | 0.19 | `if (0 <= orders[r_2])` |
| store.global*2 (est.: sampled iterations x2) | 0.05 | 3.3% | 0.5 | 0.13 | `if (0 <= orders[r_2])` |

### cache_writer_kernel grid (5, 1, 1) block (128, 1, 1)

120 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 5.06 us, warp 5.03 us, nsys graphs 3.81 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 2.64 | 52.6% | 2.09 | 47.5% |
| store | 2.03 | 40.3% | 1.84 | 41.7% |
| load | 0.29 | 5.7% | 0.39 | 8.9% |
| compute | 0.07 | 1.5% | 0.08 | 1.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 2.03 | 40.3% | 1.0 | 2.11 | `if (((int64_t)0 <= slot) & ((slot >> (int64_t)7) < ((int64_t)pages))) \| if (((int64_t)0 <=` |
| load.global | 0.23 | 4.6% | 1.0 | 0.26 | `for (int i = 0; i < 4; ++i)` |
| compute.reduce | 0.07 | 1.5% | 1.0 | 0.10 | `amax = max(amax, __shfl_xor_sync((uint)4294967295, amax, 1, 32));` |
| load.global#2 | 0.06 | 1.1% | 1.0 | 0.06 | `int64_t slot = slots[((int64_t)((int)blockIdx.x))];` |

### norm_forward_and_per_token_cast_kernel_kernel grid (1, 1, 1) block (128, 1, 1)

176 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 2.75 us, warp 2.75 us, nsys graphs 1.28 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 2.09 | 75.9% | 2.09 | 75.3% |
| load | 0.38 | 14.0% | 0.44 | 15.8% |
| compute | 0.10 | 3.8% | 0.11 | 3.8% |
| wait | 0.10 | 3.5% | 0.05 | 1.7% |
| unattributed | 0.08 | 2.9% | 0.09 | 3.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*4 (est.: sampled iterations x4) | 1.54 | 55.8% | 1.0 | 1.54 | `for (int j_2 = 0; j_2 < 2; ++j_2)` |
| store.global*4 (est.: sampled iterations x4) | 0.35 | 12.8% | 1.0 | 0.38 | `for (int j_1 = 0; j_1 < 4; ++j_1)` |
| load.global*4 (est.: sampled iterations x4) | 0.22 | 8.1% | 1.0 | 0.26 | `if (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 4)) < num_tokens)` |
| load.global#3*4 (est.: sampled iterations x4) | 0.13 | 4.7% | 1.0 | 0.13 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 128) + ((((int)threadIdx.x) & 15` |
| store.global#5*4 (est.: sampled iterations x4) | 0.13 | 4.7% | 0.8 | 0.13 | `*(uint4*)(out + ((((((int64_t)vi_1) * (int64_t)128) + ((((int64_t)((int)threadIdx.x)) & (i` |
| compute.reduce | 0.10 | 3.8% | 1.0 | 0.13 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| wait.sync | 0.10 | 3.5% | 1.0 | 0.26 | `__syncthreads();` |
| store.global#2 | 0.04 | 1.5% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| load.global#2 | 0.03 | 1.2% | 1.0 | 0.03 | `*(float2*)(row_partial + 0) = *(float2*)(sum_shared + ((((int)threadIdx.x) >> 6) * 2)); \| ` |
| store.global#3 | 0.03 | 1.2% | 0.8 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |
| load.global#3*2 (est.: sampled iterations x2) | 0.00 | 0.0% | 0.0 | 0.00 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 64) + ((((int)threadIdx.x) & 7) ` |
| load.global*2 (est.: sampled iterations x2) | 0.00 | 0.0% | 0.0 | 0.00 | `if (((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < num_tokens)` |
| store.global#4*2 (est.: sampled iterations x2) | 0.00 | 0.0% | 0.0 | 0.00 | `for (int j_2 = 0; j_2 < 2; ++j_2)` |
| store.global#5*2 (est.: sampled iterations x2) | 0.00 | 0.0% | 0.0 | 0.00 | `*(uint4*)(out + ((((((int64_t)vi_1) * (int64_t)64) + ((((int64_t)((int)threadIdx.x)) & (in` |
| store.global*2 (est.: sampled iterations x2) | 0.00 | 0.0% | 0.0 | 0.00 | `for (int j_1 = 0; j_1 < 4; ++j_1)` |

### compress_kernel grid (2, 1, 1) block (128, 1, 1)

32 launches, 8 warps; ranges, warp and last warp from the 2 recording CTAs' 8 warps; span 15.57 us, warp 10.44 us, nsys graphs 13.92 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 5.70 | 54.6% | 8.43 | 56.9% |
| unattributed | 4.25 | 40.7% | 5.45 | 36.7% |
| store | 0.43 | 4.1% | 0.86 | 5.8% |
| compute | 0.06 | 0.6% | 0.10 | 0.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 5.15 | 49.3% | 0.6 | 6.99 | `if ((((0 <= counts[0]) & (0 < counts[1])) & (counts[1] <= 2)) & (((((int)blockIdx.x) * 4) ` |
| load.global#2*4 (est.: sampled iterations x4) | 0.50 | 4.8% | 0.6 | 1.28 | `*(float4*)(v1 + 0) = *(float4*)(values + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048)` |
| store.global*4 (est.: sampled iterations x4) | 0.16 | 1.5% | 0.6 | 0.26 | `*(uint2*)(out + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + ((((int64_t)((int)thre` |
| store.global#3 | 0.10 | 0.9% | 0.3 | 0.16 | `emitted_slots[((((int64_t)((int)blockIdx.x)) * (int64_t)4) + (((int64_t)((int)threadIdx.x)` |
| store.global#2*4 (est.: sampled iterations x4) | 0.06 | 0.6% | 0.3 | 0.13 | `if ((slot >> (int64_t)3) < ((int64_t)blocks))` |
| compute.reduce | 0.06 | 0.6% | 0.6 | 0.10 | `float total = tl::warp_reduce_sum(ss[0]);` |
| store.global#5 | 0.04 | 0.4% | 0.3 | 0.16 | `emitted_slots[((((int64_t)((int)blockIdx.x)) * (int64_t)4) + (((int64_t)((int)threadIdx.x)` |
| load.global#3*4 (est.: sampled iterations x4) | 0.04 | 0.4% | 0.3 | 0.06 | `*(float4*)(w + 0) = *(float4*)(weight + ((c_1 * 128) + ((((int)threadIdx.x) & 31) * 4)));` |
| store.global#3*4 (est.: sampled iterations x4) | 0.04 | 0.4% | 0.3 | 0.06 | `*(uint2*)(out + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + ((((int64_t)((int)thre` |
| load.global#3 | 0.01 | 0.1% | 0.3 | 0.03 | `if (valid)` |
| load.global#4 | 0.01 | 0.1% | 0.3 | 0.02 | `if (valid)` |
| store.global#2 | 0.01 | 0.1% | 0.3 | 0.02 | `emitted[((((int64_t)((int)blockIdx.x)) * (int64_t)4) + (((int64_t)((int)threadIdx.x)) >> (` |
| store.global#4 | 0.01 | 0.1% | 0.3 | 0.02 | `emitted[((((int64_t)((int)blockIdx.x)) * (int64_t)4) + (((int64_t)((int)threadIdx.x)) >> (` |

### metadata_decode_kernel grid (5, 2, 1) block (128, 1, 1)

60 launches, 40 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 2.46 us, warp 1.79 us, nsys graphs 2.22 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.03 | 57.3% | 4.56 | 67.2% |
| unattributed | 0.67 | 37.3% | 1.85 | 27.3% |
| store | 0.10 | 5.4% | 0.38 | 5.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.72 | 40.4% | 0.6 | 1.38 | `if ((((offset + ((int)blockIdx.x)) < src_rows) && (0 <= (offset + ((int)blockIdx.x)))))` |
| load.global#3 | 0.27 | 15.2% | 0.4 | 0.72 | `if ((((offset + ((int)blockIdx.x)) < main_src_rows) && (0 <= (offset + ((int)blockIdx.x)))` |
| store.global#4 | 0.05 | 2.7% | 0.4 | 0.24 | `if (0 < live)` |
| store.global | 0.04 | 2.0% | 0.6 | 0.06 | `swa[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = con` |
| load.global#2 | 0.03 | 1.6% | 0.1 | 0.19 | `if (((0 <= req) & ((int64_t)0 <= pos)))` |
| store.global#2 | 0.01 | 0.4% | 0.1 | 0.06 | `lengths[((int64_t)((int)blockIdx.x))] = ((int)max(info[1], (int64_t)0));` |
| store.global#3 | 0.01 | 0.4% | 0.1 | 0.06 | `top_lengths[((int64_t)((int)blockIdx.x))] = condval_6; \| top_lengths[((int64_t)((int)block` |

### token_ranks_kernel grid (1, 1, 1) block (256, 1, 1)

48 launches, 8 warps; ranges, warp and last warp from the 1 recording CTAs' 8 warps; span 7.60 us, warp 7.59 us, nsys graphs 8.16 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 6.86 | 90.5% | 7.16 | 87.8% |
| compute | 0.34 | 4.5% | 0.37 | 4.6% |
| unattributed | 0.34 | 4.5% | 0.58 | 7.1% |
| store | 0.04 | 0.5% | 0.05 | 0.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 6.86 | 90.5% | 1.0 | 7.01 | `if ((((int64_t)0 <= token_ids[((int64_t)((int)blockIdx.x))]) && (token_ids[((int64_t)((int` |
| compute.reduce | 0.34 | 4.5% | 1.0 | 0.51 | `total[0] = tl::AllReduce<tl::SumOp, 256, 1, 0, tl::NamedBarrier<256>>::run(total[0], (&(wo` |
| store.global | 0.04 | 0.5% | 1.0 | 0.06 | `output[((int64_t)((int)blockIdx.x))] = ((int64_t)total[0]);` |

### metadata_builder_kernel grid (2, 1, 1) block (128, 1, 1)

272 launches, 8 warps; ranges, warp and last warp from the 2 recording CTAs' 8 warps; span 1.25 us, warp 0.68 us, nsys graphs 1.41 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.31 | 45.6% | 0.34 | 24.5% |
| load | 0.30 | 43.3% | 0.98 | 71.3% |
| store | 0.08 | 11.1% | 0.06 | 4.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#3 | 0.25 | 36.8% | 0.5 | 0.86 | `for (int __1 = 0; __1 < 2; ++__1)` |
| store.global#4 | 0.03 | 4.7% | 0.5 | 0.06 | `positions[(((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))` |
| load.global#2 | 0.03 | 4.1% | 0.1 | 0.22 | `if ((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) < nr))` |
| store.global#3 | 0.03 | 4.1% | 0.5 | 0.06 | `if (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) == 0)` |
| load.global | 0.02 | 2.3% | 0.1 | 0.13 | `if ((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) <= nr))` |
| store.global | 0.01 | 1.8% | 0.1 | 0.10 | `out_starts[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] = condval;` |
| store.global#2 | 0.00 | 0.6% | 0.1 | 0.03 | `request_positions[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] = condval_2;` |

### rope_kernel grid (1, 1, 1) block (128, 1, 1)

256 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 1.30 us, warp 1.16 us, nsys graphs 1.18 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.99 | 85.8% | 1.09 | 75.1% |
| load | 0.14 | 12.5% | 0.31 | 21.6% |
| store | 0.02 | 1.7% | 0.05 | 3.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.10 | 8.3% | 1.0 | 0.10 | `if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < tokens)) \| int64_t position ` |
| load.global#2 | 0.05 | 4.2% | 0.5 | 0.10 | `if (((position < ((int64_t)positions_max)) && ((int64_t)0 <= position)))` |
| store.global | 0.02 | 1.7% | 0.5 | 0.03 | `*(uint4*)(x + ((((((int64_t)((int)blockIdx.x)) * (int64_t)12288) + ((((int64_t)((int)threa` |

### moe_grouped_fp8_fp4_gemm_kernel grid (40, 36, 1) block (128, 1, 1)

6 launches, 5760 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 61.09 us, warp 3.41 us, nsys graphs 9.89 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 1.51 | 44.3% | 0.05 | 16.7% |
| unattributed | 0.88 | 25.9% | 0.21 | 66.7% |
| compute | 0.48 | 14.1% | 0.00 | 0.0% |
| load | 0.46 | 13.6% | 0.05 | 16.7% |
| store | 0.07 | 2.1% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.mbarrier*6 (est.: sampled iterations x6) | 0.96 | 28.2% | 0.2 | 3.84 | `loaded[(ko % 3)].wait((ko / 3));` |
| compute.mma*6 (est.: sampled iterations x6) | 0.48 | 14.1% | 0.2 | 1.92 | `{` |
| load#2*6 (est.: sampled iterations x6) | 0.22 | 6.3% | 0.2 | 1.15 | `if (ko < 4)` |
| wait.sync#5*6 (est.: sampled iterations x6) | 0.17 | 4.9% | 0.2 | 0.77 | `__syncthreads();` |
| load.global#3 | 0.11 | 3.3% | 0.2 | 0.45 | `if (((expert < 128) && (0 <= expert))) \| if (((expert < 384) && (0 <= expert)))` |
| wait.sync#4*6 (est.: sampled iterations x6) | 0.10 | 2.8% | 0.2 | 0.38 | `__syncthreads();` |
| wait.sync#6*6 (est.: sampled iterations x6) | 0.10 | 2.8% | 0.2 | 0.38 | `__syncthreads();` |
| wait.cp_async*6 (est.: sampled iterations x6) | 0.09 | 2.6% | 0.2 | 0.48 | `tl::cp_async_wait<2>();` |
| store.global | 0.07 | 2.1% | 0.2 | 0.35 | `for (int i_2 = 0; i_2 < 16; ++i_2) \| for (int i_4 = 0; i_4 < 32; ++i_4)` |
| load.global | 0.07 | 2.0% | 1.0 | 0.10 | `int expert = TileExpert[((int64_t)((int)blockIdx.y))];` |
| wait.sync | 0.05 | 1.5% | 1.0 | 0.06 | `__syncthreads();` |
| load | 0.04 | 1.3% | 0.2 | 0.22 | `for (int s = 0; s < 2; ++s)` |
| wait.sync#2 | 0.03 | 1.0% | 0.2 | 0.16 | `__syncthreads();` |
| load.global#2 | 0.02 | 0.7% | 0.2 | 0.18 | `if (((int)threadIdx.x) < 32) \| if (((int)threadIdx.x) < 64)` |
| wait.sync#3 | 0.01 | 0.4% | 0.2 | 0.06 | `__syncthreads();` |

### indexer_topk_segment_kernel grid (1, 5, 1) block (512, 1, 1)

30 launches, 80 warps; ranges, warp and last warp from the 5 recording CTAs' 80 warps; span 10.90 us, warp 10.68 us, nsys graphs 5.42 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed, with loops' unsampled iterations | 5.61 | 52.5% | 5.35 | 49.6% |
| wait | 3.74 | 35.0% | 3.99 | 37.0% |
| load | 1.17 | 10.9% | 1.12 | 10.4% |
| store | 0.17 | 1.6% | 0.32 | 3.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.17 | 10.9% | 1.0 | 1.38 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 4096); \| i` |
| wait.sync (sampled iterations) | 0.99 | 9.3% | 14.5 | 1.18 | `__syncthreads();` |
| wait.sync#24*9 (est.: sampled iterations x9) | 0.31 | 2.9% | 0.5 | 0.43 | `__syncthreads();` |
| wait.sync#20*9 (est.: sampled iterations x9) | 0.30 | 2.8% | 0.5 | 0.43 | `__syncthreads();` |
| wait.sync#23*9 (est.: sampled iterations x9) | 0.29 | 2.7% | 0.5 | 0.43 | `__syncthreads();` |
| wait.sync#19*9 (est.: sampled iterations x9) | 0.29 | 2.7% | 0.5 | 0.43 | `__syncthreads();` |
| wait.sync#14*8 (est.: sampled iterations x8) | 0.28 | 2.6% | 0.5 | 0.38 | `__syncthreads();` |
| wait.sync#6*8 (est.: sampled iterations x8) | 0.28 | 2.6% | 0.5 | 0.38 | `__syncthreads();` |
| wait.sync#5*8 (est.: sampled iterations x8) | 0.27 | 2.5% | 0.5 | 0.38 | `__syncthreads();` |
| wait.sync#13*8 (est.: sampled iterations x8) | 0.26 | 2.4% | 0.5 | 0.26 | `__syncthreads();` |
| store.global | 0.17 | 1.6% | 1.5 | 0.22 | `for (int i_3 = 0; i_3 < 4; ++i_3) \| if (((int)threadIdx.x) < total) \| if (((int)threadIdx.` |
| wait.sync#25 | 0.05 | 0.4% | 0.5 | 0.08 | `__syncthreads();` |
| wait.sync#17 | 0.04 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#16 | 0.04 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#8 | 0.04 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#18 | 0.03 | 0.3% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#3 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#2 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#15 | 0.03 | 0.3% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#9 | 0.03 | 0.3% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#4 | 0.03 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#11 | 0.03 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#7 | 0.03 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#22 | 0.03 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#12 | 0.02 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#10 | 0.02 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#21 | 0.02 | 0.2% | 0.5 | 0.03 | `__syncthreads();` |

### topk_ids_kernel grid (5, 1, 1) block (512, 1, 1)

15 launches, 80 warps; ranges, warp and last warp from the 5 recording CTAs' 80 warps; span 20.74 us, warp 18.64 us, nsys graphs 20.56 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 15.17 | 81.3% | 15.18 | 72.8% |
| unattributed | 2.23 | 12.0% | 1.91 | 9.2% |
| wait | 0.82 | 4.4% | 0.93 | 4.5% |
| store | 0.42 | 2.3% | 2.83 | 13.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 15.17 | 81.3% | 1.0 | 16.13 | `for (int t = 0; t < (vocab >> 14); ++t)` |
| wait.sync | 0.69 | 3.7% | 1.0 | 1.41 | `__syncthreads();` |
| store.global | 0.42 | 2.3% | 1.0 | 3.04 | `if ((((int)threadIdx.x) >> 5) == 0)` |
| wait.sync#2 | 0.08 | 0.4% | 1.0 | 0.16 | `__syncthreads();` |
| wait.sync#3 | 0.06 | 0.3% | 1.0 | 0.06 | `__syncthreads();` |

### norm_forward_and_per_token_cast_kernel_kernel grid (3, 1, 1) block (128, 1, 1)

96 launches, 12 warps; ranges, warp and last warp from the 3 recording CTAs' 12 warps; span 2.75 us, warp 2.55 us, nsys graphs 1.89 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 1.61 | 63.2% | 1.89 | 63.0% |
| load | 0.34 | 13.4% | 0.38 | 12.6% |
| unattributed | 0.34 | 13.4% | 0.57 | 18.9% |
| wait | 0.14 | 5.4% | 0.05 | 1.7% |
| compute | 0.12 | 4.6% | 0.12 | 3.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*3 (est.: sampled iterations x3) | 1.15 | 45.1% | 1.0 | 1.20 | `for (int j_2 = 0; j_2 < 2; ++j_2)` |
| store.global*3 (est.: sampled iterations x3) | 0.29 | 11.3% | 1.0 | 0.38 | `for (int j_1 = 0; j_1 < 4; ++j_1)` |
| load.global*3 (est.: sampled iterations x3) | 0.18 | 6.9% | 1.0 | 0.19 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 6)) < num_tokens) && (((vi * 2) + (` |
| wait.sync | 0.14 | 5.4% | 1.0 | 0.26 | `__syncthreads();` |
| load.global#3*3 (est.: sampled iterations x3) | 0.13 | 5.2% | 1.0 | 0.19 | `if (((vi_1 * 2) + ((((int)threadIdx.x) & 63) >> 5)) < 5)` |
| compute.reduce | 0.12 | 4.6% | 1.0 | 0.13 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| store.global#5*3 (est.: sampled iterations x3) | 0.10 | 4.1% | 0.8 | 0.19 | `*(uint4*)(out + ((((((int64_t)vi_1) * (int64_t)512) + ((((int64_t)((int)threadIdx.x)) & (i` |
| store.global#2 | 0.04 | 1.6% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| load.global#2 | 0.03 | 1.4% | 1.0 | 0.06 | `*(float2*)(row_partial + 0) = *(float2*)(sum_shared + ((((int)threadIdx.x) >> 6) * 2)); \| ` |
| store.global#3 | 0.03 | 1.0% | 0.4 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |

### sparse_mla_kernel_1 grid (24, 5, 1) block (128, 1, 1)

90 launches, 120 CTAs, 480 warps; span 2.02 us, warp 1.53 us, nsys graphs 2.27 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.49 | 97.4% | 1.29 | 42.8% |
| load | 0.03 | 1.8% | 1.51 | 50.1% |
| wait | 0.01 | 0.6% | 0.10 | 3.5% |
| compute | 0.00 | 0.2% | 0.07 | 2.3% |
| store | 0.00 | 0.1% | 0.04 | 1.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.03 | 1.8% | 0.1 | 1.09 | `M = __fmul_rn(Sinks[((int)blockIdx.x)], 0x1.7154764ee6c2fp+0f/*1.442695e+00*/); \| if (((in` |
| wait.sync | 0.01 | 0.6% | 0.1 | 0.43 | `__syncthreads();` |
| compute.reduce | 0.00 | 0.2% | 0.0 | 0.10 | `amax[0] = tl::AllReduce<tl::MaxOp, 8, 1, 0, tl::NamedBarrier<128>>::run(amax[0]);` |
| store.global | 0.00 | 0.1% | 0.0 | 0.06 | `*(uint2*)(Output + (((((int64_t)((int)blockIdx.y)) * (int64_t)12288) + (((int64_t)((int)bl` |
| compute.reduce#3 | 0.00 | 0.0% | 0.0 | 0.00 | `amax[0] = tl::AllReduce<tl::MaxOp, 8, 1, 0, tl::NamedBarrier<128>>::run(amax[0]);` |
| load.global#3 | 0.00 | 0.0% | 0.0 | 0.00 | `M = __fmul_rn(Sinks[((int)blockIdx.x)], 0x1.7154764ee6c2fp+0f/*1.442695e+00*/);` |
| load.global#4 | 0.00 | 0.0% | 0.0 | 0.00 | `if (((int)threadIdx.x) < 32)` |
| store.global#2 | 0.00 | 0.0% | 0.0 | 0.00 | `*(uint2*)(Output + (((((int64_t)((int)blockIdx.y)) * (int64_t)12288) + (((int64_t)((int)bl` |
| wait.sync#12 | 0.00 | 0.0% | 0.0 | 0.00 | `__syncthreads();` |
| wait.sync#13 | 0.00 | 0.0% | 0.0 | 0.00 | `__syncthreads();` |

### topk_log_softmax_kernel grid (1, 1, 1) block (1024, 1, 1)

16 launches, 32 warps; ranges, warp and last warp from the 1 recording CTAs' 32 warps; span 16.56 us, warp 16.00 us, nsys graphs 15.87 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 12.99 | 81.2% | 13.82 | 82.8% |
| unattributed, with loops' unsampled iterations | 1.63 | 10.2% | 1.78 | 10.7% |
| wait | 0.98 | 6.1% | 0.65 | 3.9% |
| compute | 0.39 | 2.4% | 0.38 | 2.3% |
| store | 0.00 | 0.0% | 0.07 | 0.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 6.60 | 41.2% | 1.0 | 6.82 | `if (0 < (vocab & 2047)) \| if (0 < (vocab & 8191))` |
| load.global#2 | 6.38 | 39.9% | 1.0 | 6.82 | `for (int step_1 = 0; step_1 < (vocab >> 11); ++step_1) \| for (int step_1 = 0; step_1 < (vo` |
| wait.sync#2 | 0.45 | 2.8% | 1.0 | 0.80 | `__syncthreads();` |
| wait.sync | 0.28 | 1.7% | 1.0 | 0.69 | `__syncthreads();` |
| compute.reduce#3 | 0.17 | 1.1% | 1.0 | 0.19 | `ties[0] = vllm_add(ties[0], __shfl_xor_sync((uint)4294967295, ties[0], 16, 32));` |
| compute.reduce#2 | 0.11 | 0.7% | 1.0 | 0.16 | `se[0] = vllm_add(se[0], __shfl_xor_sync((uint)4294967295, se[0], 16, 32));` |
| compute.reduce | 0.11 | 0.7% | 1.0 | 0.13 | `best[0] = vllm_max(best[0], __shfl_xor_sync((uint)4294967295, best[0], 16, 32));` |
| wait.sync#4 | 0.10 | 0.6% | 1.0 | 0.16 | `__syncthreads();` |
| wait.sync#3 | 0.08 | 0.5% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#5 | 0.08 | 0.5% | 1.0 | 0.10 | `__syncthreads();` |
| load.global#3 (sampled iterations) | 0.01 | 0.1% | 0.0 | 0.38 | `float value_32 = vllm_sub(vllm_shifted(logits[((((int64_t)((int)blockIdx.x)) * logits_stri` |
| store.global (sampled iterations) | 0.00 | 0.0% | 0.0 | 0.06 | `output[(((((int64_t)c) * (int64_t)1024) + (((int64_t)((int)blockIdx.x)) * ((int64_t)topk))` |

### sparse_mla_kernel_1 grid (24, 10, 1) block (128, 1, 1)

48 launches, 240 CTAs, 960 warps; span 5.36 us, warp 4.82 us, nsys graphs 5.25 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 4.79 | 99.4% | 3.90 | 73.1% |
| load | 0.02 | 0.5% | 1.28 | 24.0% |
| wait | 0.00 | 0.1% | 0.07 | 1.3% |
| compute | 0.00 | 0.0% | 0.05 | 1.0% |
| store | 0.00 | 0.0% | 0.03 | 0.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#3 | 0.02 | 0.4% | 0.0 | 1.47 | `M = __fmul_rn(Sinks[((int)blockIdx.x)], 0x1.7154764ee6c2fp+0f/*1.442695e+00*/);` |
| wait.sync#13 | 0.00 | 0.1% | 0.0 | 0.26 | `__syncthreads();` |
| load.global#4 | 0.00 | 0.0% | 0.0 | 0.26 | `if (((int)threadIdx.x) < 32)` |
| compute.reduce#3 | 0.00 | 0.0% | 0.0 | 0.11 | `amax[0] = tl::AllReduce<tl::MaxOp, 8, 1, 0, tl::NamedBarrier<128>>::run(amax[0]);` |
| wait.sync#12 | 0.00 | 0.0% | 0.0 | 0.10 | `__syncthreads();` |
| store.global#2 | 0.00 | 0.0% | 0.0 | 0.06 | `*(uint2*)(Output + (((((int64_t)((int)blockIdx.y)) * (int64_t)12288) + (((int64_t)((int)bl` |

### metadata_chunk_kernel grid (5, 1, 1) block (128, 1, 1)

30 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 6.05 us, warp 5.13 us, nsys graphs 5.46 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 3.40 | 66.2% | 3.33 | 39.0% |
| load | 1.66 | 32.4% | 5.07 | 59.3% |
| store | 0.07 | 1.4% | 0.14 | 1.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.54 | 30.1% | 1.0 | 2.10 | `if ((((offset + ((int)blockIdx.x)) < src_rows) && (0 <= (offset + ((int)blockIdx.x)))))` |
| load.global#2 | 0.12 | 2.4% | 0.2 | 0.50 | `if (((0 <= req) & ((int64_t)0 <= pos)))` |
| store.global | 0.05 | 1.0% | 1.0 | 0.06 | `swa[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = con` |
| store.global#2 | 0.01 | 0.2% | 0.2 | 0.06 | `lengths[((int64_t)((int)blockIdx.x))] = ((int)max(info[1], (int64_t)0));` |
| store.global#3 | 0.01 | 0.2% | 0.2 | 0.03 | `top_lengths[((int64_t)((int)blockIdx.x))] = condval_12; \| top_lengths[((int64_t)((int)bloc` |

### topk_log_softmax_kernel grid (5, 1, 1) block (1024, 1, 1)

15 launches, 160 warps; ranges, warp and last warp from the 5 recording CTAs' 160 warps; span 16.96 us, warp 16.20 us, nsys graphs 16.62 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 13.16 | 81.2% | 13.92 | 81.7% |
| unattributed, with loops' unsampled iterations | 1.63 | 10.1% | 1.76 | 10.3% |
| wait | 1.02 | 6.3% | 0.91 | 5.3% |
| compute | 0.39 | 2.4% | 0.38 | 2.2% |
| store | 0.00 | 0.0% | 0.07 | 0.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 6.67 | 41.2% | 1.0 | 7.01 | `if (0 < (vocab & 2047)) \| if (0 < (vocab & 8191))` |
| load.global#2 | 6.47 | 40.0% | 1.0 | 7.07 | `for (int step_1 = 0; step_1 < (vocab >> 11); ++step_1) \| for (int step_1 = 0; step_1 < (vo` |
| wait.sync#2 | 0.46 | 2.8% | 1.0 | 1.06 | `__syncthreads();` |
| wait.sync | 0.30 | 1.9% | 1.0 | 0.96 | `__syncthreads();` |
| compute.reduce#3 | 0.17 | 1.1% | 1.0 | 0.22 | `ties[0] = vllm_add(ties[0], __shfl_xor_sync((uint)4294967295, ties[0], 16, 32));` |
| compute.reduce#2 | 0.11 | 0.7% | 1.0 | 0.16 | `se[0] = vllm_add(se[0], __shfl_xor_sync((uint)4294967295, se[0], 16, 32));` |
| compute.reduce | 0.11 | 0.7% | 1.0 | 0.13 | `best[0] = vllm_max(best[0], __shfl_xor_sync((uint)4294967295, best[0], 16, 32));` |
| wait.sync#4 | 0.10 | 0.6% | 1.0 | 0.16 | `__syncthreads();` |
| wait.sync#5 | 0.08 | 0.5% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#3 | 0.08 | 0.5% | 1.0 | 0.10 | `__syncthreads();` |
| load.global#3 (sampled iterations) | 0.01 | 0.1% | 0.0 | 0.38 | `float value_32 = vllm_sub(vllm_shifted(logits[((((int64_t)((int)blockIdx.x)) * logits_stri` |
| store.global (sampled iterations) | 0.00 | 0.0% | 0.0 | 0.06 | `output[(((((int64_t)c) * (int64_t)1024) + (((int64_t)((int)blockIdx.x)) * ((int64_t)topk))` |

### cache_writer_kernel grid (10, 1, 1) block (128, 1, 1)

48 launches, 40 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 5.33 us, warp 5.16 us, nsys graphs 3.71 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 2.64 | 51.3% | 2.23 | 43.9% |
| unattributed | 2.15 | 41.7% | 2.30 | 45.3% |
| load | 0.29 | 5.6% | 0.47 | 9.2% |
| compute | 0.08 | 1.5% | 0.08 | 1.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 2.64 | 51.3% | 1.0 | 2.74 | `if (((int64_t)0 <= slot) & ((slot >> (int64_t)7) < ((int64_t)pages))) \| if (((int64_t)0 <=` |
| load.global | 0.23 | 4.5% | 1.0 | 0.26 | `for (int i = 0; i < 4; ++i)` |
| compute.reduce | 0.08 | 1.5% | 1.0 | 0.10 | `amax = max(amax, __shfl_xor_sync((uint)4294967295, amax, 1, 32));` |
| load.global#2 | 0.06 | 1.1% | 1.0 | 0.06 | `int64_t slot = slots[((int64_t)((int)blockIdx.x))];` |

### moe_combine_kernel grid (5, 5, 1) block (128, 1, 1)

90 launches, 100 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 2.34 us, warp 1.73 us, nsys graphs 2.43 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.29 | 74.8% | 1.88 | 81.0% |
| unattributed | 0.38 | 22.1% | 0.39 | 16.8% |
| store | 0.06 | 3.2% | 0.05 | 2.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.29 | 74.8% | 1.0 | 1.89 | `for (int k = 0; k < 3; ++k) \| for (int k = 0; k < 6; ++k) \| if ((((int)blockIdx.y) < share` |
| store.global | 0.06 | 3.2% | 1.0 | 0.06 | `{` |

### local_stats_kernel grid (5, 16, 1) block (128, 1, 1)

15 launches, 320 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 12.90 us, warp 12.42 us, nsys graphs 12.80 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 11.38 | 91.6% | 12.03 | 91.9% |
| unattributed | 0.70 | 5.7% | 0.70 | 5.3% |
| compute | 0.17 | 1.4% | 0.17 | 1.3% |
| wait | 0.10 | 0.8% | 0.11 | 0.8% |
| store | 0.07 | 0.5% | 0.10 | 0.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 11.38 | 91.6% | 1.0 | 11.84 | `if (expanded_local_pos[((int64_t)((int)blockIdx.x))] < num_speculative_steps)` |
| compute.reduce | 0.12 | 0.9% | 1.0 | 0.13 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| wait.sync | 0.10 | 0.8% | 1.0 | 0.29 | `__syncthreads();` |
| store.global | 0.07 | 0.5% | 1.0 | 0.13 | `if (((int)threadIdx.x) == 0)` |
| compute.reduce#2 | 0.05 | 0.4% | 1.0 | 0.06 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 2, 32);` |

### per_token_cast_kernel_kernel grid (6, 1, 1) block (64, 1, 1)

48 launches, 12 warps; ranges, warp and last warp from the 6 recording CTAs' 12 warps; span 3.70 us, warp 3.35 us, nsys graphs 2.72 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 2.40 | 71.6% | 2.32 | 58.1% |
| load | 0.63 | 18.9% | 1.22 | 30.6% |
| store | 0.22 | 6.4% | 0.31 | 7.6% |
| compute | 0.10 | 3.0% | 0.15 | 3.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.54 | 16.2% | 1.0 | 0.94 | `for (int i = 0; i < 2; ++i)` |
| store.global#2 | 0.12 | 3.6% | 1.0 | 0.16 | `for (int i_3 = 0; i_3 < 4; ++i_3)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.10 | 3.0% | 1.0 | 0.13 | `amax_fragment[i_1] = tl::AllReduce<tl::MaxOp, 2, 1, 0, tl::NamedBarrier<64>>::run(amax_fra` |
| load.global#2*2 (est.: sampled iterations x2) | 0.09 | 2.7% | 1.0 | 0.13 | `amax_fragment[i_1] = cutlass::bfloat16_t(__hmax((bfloat16_t(((nv_bfloat162*)(&(amax_fragme` |
| store.global*2 (est.: sampled iterations x2) | 0.05 | 1.6% | 0.5 | 0.13 | `out_sf[((((((int64_t)((int)blockIdx.x)) % (int64_t)15) * (int64_t)32) + (((int64_t)((int)t` |
| store.global#3*2 (est.: sampled iterations x2) | 0.04 | 1.3% | 0.5 | 0.13 | `*(fp8_e4_16_t*)(out + ((((((((int64_t)((int)blockIdx.x)) / (int64_t)3) * (int64_t)6144) + ` |

### norm_forward_and_per_token_cast_kernel_kernel grid (5, 1, 1) block (128, 1, 1)

63 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 2.82 us, warp 2.57 us, nsys graphs 1.89 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 1.65 | 64.2% | 1.94 | 67.9% |
| unattributed | 0.37 | 14.4% | 0.39 | 13.5% |
| load | 0.32 | 12.5% | 0.36 | 12.5% |
| compute | 0.12 | 4.6% | 0.12 | 4.3% |
| wait | 0.11 | 4.3% | 0.05 | 1.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*3 (est.: sampled iterations x3) | 1.15 | 44.9% | 1.0 | 1.15 | `for (int j_2 = 0; j_2 < 2; ++j_2)` |
| store.global*3 (est.: sampled iterations x3) | 0.30 | 11.6% | 1.0 | 0.38 | `for (int j_1 = 0; j_1 < 4; ++j_1)` |
| load.global*3 (est.: sampled iterations x3) | 0.17 | 6.5% | 1.0 | 0.19 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 6)) < num_tokens) && (((vi * 2) + (` |
| store.global#5*3 (est.: sampled iterations x3) | 0.12 | 4.9% | 1.0 | 0.19 | `*(uint4*)(out + ((((((int64_t)vi_1) * (int64_t)512) + ((((int64_t)((int)threadIdx.x)) & (i` |
| load.global#3*3 (est.: sampled iterations x3) | 0.12 | 4.7% | 1.0 | 0.19 | `if (((vi_1 * 2) + ((((int)threadIdx.x) & 63) >> 5)) < 5)` |
| compute.reduce | 0.12 | 4.6% | 1.0 | 0.13 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| wait.sync | 0.11 | 4.3% | 1.0 | 0.19 | `__syncthreads();` |
| store.global#2 | 0.04 | 1.7% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| load.global#2 | 0.03 | 1.2% | 1.0 | 0.03 | `*(float2*)(row_partial + 0) = *(float2*)(sum_shared + ((((int)threadIdx.x) >> 6) * 2)); \| ` |
| store.global#3 | 0.03 | 1.2% | 0.5 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |
| load.global#2*5 (est.: sampled iterations x5) | 0.00 | 0.0% | 0.0 | 0.00 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 1024) + (((int)threadIdx.x) * 8)` |
| load.global*5 (est.: sampled iterations x5) | 0.00 | 0.0% | 0.0 | 0.00 | `*(uint4*)(x_local + (vi * 8)) = *(uint4*)(x + ((((((int64_t)vi) * (int64_t)1024) + (((int6` |
| store.global#4*5 (est.: sampled iterations x5) | 0.00 | 0.0% | 0.0 | 0.00 | `for (int j_1 = 0; j_1 < 2; ++j_1)` |
| store.global*5 (est.: sampled iterations x5) | 0.00 | 0.0% | 0.0 | 0.00 | `for (int j = 0; j < 4; ++j)` |

### mhc_finalize_kernel grid (6, 1, 1) block (128, 1, 1)

12 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 13.46 us, warp 12.50 us, nsys graphs 6.90 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 7.05 | 56.4% | 6.15 | 44.1% |
| store | 1.97 | 15.8% | 2.98 | 21.3% |
| load | 1.37 | 10.9% | 2.88 | 20.7% |
| wait | 1.14 | 9.1% | 0.65 | 4.7% |
| compute | 0.98 | 7.8% | 1.28 | 9.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync | 1.14 | 9.1% | 1.0 | 1.60 | `__syncthreads();` |
| store.global | 1.06 | 8.5% | 0.2 | 4.29 | `if (((int)threadIdx.x) < 4)` |
| load.global | 0.63 | 5.0% | 1.0 | 2.02 | `if (((int)threadIdx.x) < 32)` |
| compute.reduce | 0.57 | 4.6% | 0.2 | 2.30 | `row_max[0] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<32>>::run(row_max[0]);` |
| store.global#4*7 (est.: sampled iterations x7) | 0.56 | 4.5% | 0.8 | 0.90 | `if ((((int)threadIdx.x) % 4) == 0)` |
| compute.reduce#2*7 (est.: sampled iterations x7) | 0.41 | 3.2% | 0.8 | 0.67 | `amax[0] = max(amax[0], __shfl_xor_sync((uint)4294967295, amax[0], 1, 32));` |
| load.global#4*7 (est.: sampled iterations x7) | 0.37 | 3.0% | 0.8 | 0.67 | `*(uint4*)(yv + 0) = *(uint4*)(C + ((((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((i` |
| store.global#3*7 (est.: sampled iterations x7) | 0.33 | 2.7% | 0.8 | 0.56 | `*(uint4*)(y + ((((((int64_t)((int)blockIdx.x)) * (int64_t)5120) + (((int64_t)step) * (int6` |
| load.global#3 | 0.32 | 2.5% | 0.8 | 0.45 | `for (int i_split_2 = 0; i_split_2 < 40; ++i_split_2)` |
| load.global#2 | 0.05 | 0.4% | 0.2 | 0.19 | `cm[0] = ((mixes_shared[((((int)threadIdx.x) & 15) + 8)] * scale[2]) + base[((((int)threadI` |
| store.global#2 | 0.01 | 0.1% | 0.2 | 0.06 | `if ((((int)threadIdx.x) >> 4) == 0)` |

### router_partials_kernel grid (5, 1, 1) block (64, 1, 1)

48 launches, 10 warps; ranges, warp and last warp from the 5 recording CTAs' 10 warps; span 2.91 us, warp 2.13 us, nsys graphs 2.66 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.89 | 41.7% | 1.43 | 41.9% |
| load | 0.88 | 41.2% | 1.40 | 41.2% |
| store | 0.19 | 8.8% | 0.22 | 6.5% |
| compute | 0.18 | 8.3% | 0.36 | 10.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.48 | 22.7% | 1.0 | 0.51 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < mask_len))` |
| load.global#4 | 0.27 | 12.5% | 0.5 | 0.56 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < image_len)) \| if ((((((int)bl` |
| compute.reduce | 0.18 | 8.3% | 0.5 | 0.38 | `for (int k = 0; k < 3; ++k) \| for (int k = 0; k < 6; ++k)` |
| store.global | 0.07 | 3.2% | 1.0 | 0.10 | `if (condval == (bool)0) \| if (condval == (bool)1)` |
| load.global#2 | 0.06 | 3.0% | 1.0 | 0.10 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < mask_len))` |
| load.global#3 | 0.06 | 3.0% | 1.0 | 0.10 | `if ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) < mask_len))` |
| store.global#2 | 0.06 | 2.9% | 1.0 | 0.10 | `if (condval_1 == (bool)0) \| if (condval_1 == (bool)1)` |
| store.global#3 | 0.06 | 2.7% | 0.5 | 0.13 | `if ((((int)threadIdx.x) & 31) < 3) \| if ((((int)threadIdx.x) & 31) < 6)` |

### per_token_cast_kernel_kernel grid (5, 1, 1) block (64, 1, 1)

128 launches, 10 warps; ranges, warp and last warp from the 5 recording CTAs' 10 warps; span 1.18 us, warp 1.09 us, nsys graphs 1.06 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.45 | 41.2% | 0.45 | 37.1% |
| load | 0.34 | 31.5% | 0.37 | 30.8% |
| store | 0.20 | 18.5% | 0.30 | 24.3% |
| compute | 0.10 | 8.8% | 0.10 | 7.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.25 | 23.2% | 1.0 | 0.26 | `for (int i = 0; i < 2; ++i)` |
| store.global#2 | 0.11 | 10.3% | 1.0 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.10 | 8.8% | 1.0 | 0.13 | `amax_fragment[i_1] = tl::AllReduce<tl::MaxOp, 2, 1, 0, tl::NamedBarrier<64>>::run(amax_fra` |
| load.global#2*2 (est.: sampled iterations x2) | 0.09 | 8.2% | 1.0 | 0.13 | `amax_fragment[i_1] = cutlass::bfloat16_t(__hmax((bfloat16_t(((nv_bfloat162*)(&(amax_fragme` |
| store.global#3*2 (est.: sampled iterations x2) | 0.04 | 4.1% | 0.5 | 0.13 | `*(fp8_e4_16_t*)(out + ((((((((int64_t)((int)blockIdx.x)) / (int64_t)3) * (int64_t)6144) + ` |
| store.global*2 (est.: sampled iterations x2) | 0.04 | 4.1% | 0.5 | 0.13 | `out_sf[((((((int64_t)((int)blockIdx.x)) % (int64_t)15) * (int64_t)32) + (((int64_t)((int)t` |

### mhc_collapse_kernel grid (25, 1, 1) block (128, 1, 1)

45 launches, 100 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 2.98 us, warp 2.98 us, nsys graphs 3.65 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 2.83 | 95.2% | 3.11 | 91.0% |
| load | 0.10 | 3.2% | 0.26 | 7.6% |
| store | 0.05 | 1.6% | 0.05 | 1.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.10 | 3.2% | 1.0 | 0.13 | `*(float4*)(mv + 0) = *(float4*)(mix + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int6` |
| store.global | 0.05 | 1.6% | 1.0 | 0.06 | `*(uint4*)(out + ((((int64_t)((int)blockIdx.x)) * (int64_t)1024) + (((int64_t)((int)threadI` |

### indexer_topk_segment_kernel grid (2, 5, 1) block (512, 1, 1)

15 launches, 160 warps; ranges, warp and last warp from the 5 recording CTAs' 80 warps; span 10.05 us, warp 9.83 us, nsys graphs 4.77 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 6.12 | 62.2% | 5.89 | 58.5% |
| unattributed, with loops' unsampled iterations | 3.16 | 32.1% | 3.32 | 33.0% |
| load | 0.46 | 4.7% | 0.74 | 7.4% |
| store | 0.09 | 0.9% | 0.11 | 1.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync#20*9 (est.: sampled iterations x9) | 0.68 | 6.9% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#24*9 (est.: sampled iterations x9) | 0.64 | 6.5% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#19*9 (est.: sampled iterations x9) | 0.62 | 6.3% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#23*9 (est.: sampled iterations x9) | 0.62 | 6.3% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#5*8 (est.: sampled iterations x8) | 0.60 | 6.1% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync#6*8 (est.: sampled iterations x8) | 0.59 | 6.0% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync#13*8 (est.: sampled iterations x8) | 0.59 | 6.0% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync#14*8 (est.: sampled iterations x8) | 0.57 | 5.8% | 1.0 | 0.77 | `__syncthreads();` |
| load.global#2 | 0.26 | 2.7% | 1.0 | 0.51 | `for (int i = 0; i < 8; ++i)` |
| load.global | 0.20 | 2.1% | 1.0 | 0.22 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 4096); \| i` |
| wait.sync (sampled iterations) | 0.14 | 1.4% | 1.0 | 0.38 | `__syncthreads();` |
| wait.sync#25 | 0.10 | 1.0% | 1.0 | 0.16 | `__syncthreads();` |
| store.global | 0.09 | 0.9% | 1.0 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3) \| if (((int)threadIdx.x) < total) \| if (((int)threadIdx.` |
| wait.sync#17 | 0.08 | 0.8% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#18 | 0.08 | 0.8% | 1.0 | 0.16 | `__syncthreads();` |
| wait.sync#16 | 0.07 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#8 | 0.07 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#9 | 0.07 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#2 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#3 | 0.06 | 0.6% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#11 | 0.06 | 0.6% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#15 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#7 | 0.06 | 0.6% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#4 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#10 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#12 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#22 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#21 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |

### norm_forward_and_per_token_cast_kernel_kernel grid (2, 1, 1) block (128, 1, 1)

48 launches, 8 warps; ranges, warp and last warp from the 2 recording CTAs' 8 warps; span 2.69 us, warp 2.69 us, nsys graphs 1.34 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 2.04 | 76.0% | 2.68 | 90.7% |
| load | 0.46 | 17.3% | 0.46 | 15.7% |
| wait | 0.14 | 5.1% | 0.06 | 1.9% |
| compute | 0.11 | 4.2% | 0.11 | 3.7% |
| unattributed | 0.00 | 0.0% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*4 (est.: sampled iterations x4) | 1.54 | 57.4% | 1.0 | 1.66 | `for (int j_2 = 0; j_2 < 2; ++j_2)` |
| store.global*4 (est.: sampled iterations x4) | 0.34 | 12.5% | 1.0 | 0.51 | `for (int j_1 = 0; j_1 < 4; ++j_1)` |
| load.global*4 (est.: sampled iterations x4) | 0.24 | 8.9% | 1.0 | 0.26 | `if (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 4)) < num_tokens)` |
| load.global#3*4 (est.: sampled iterations x4) | 0.19 | 7.1% | 1.0 | 0.26 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 128) + ((((int)threadIdx.x) & 15` |
| wait.sync | 0.14 | 5.1% | 1.0 | 0.29 | `__syncthreads();` |
| compute.reduce | 0.11 | 4.2% | 1.0 | 0.13 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| store.global#5*4 (est.: sampled iterations x4) | 0.10 | 3.6% | 0.6 | 0.26 | `*(uint4*)(out + ((((((int64_t)vi_1) * (int64_t)128) + ((((int64_t)((int)threadIdx.x)) & (i` |
| store.global#2 | 0.04 | 1.6% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| load.global#2 | 0.03 | 1.2% | 1.0 | 0.03 | `*(float2*)(row_partial + 0) = *(float2*)(sum_shared + ((((int)threadIdx.x) >> 6) * 2)); \| ` |
| store.global#3 | 0.02 | 0.9% | 0.6 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |

### rope_kernel grid (5, 1, 1) block (192, 1, 1)

90 launches, 30 warps; ranges, warp and last warp from the 5 recording CTAs' 30 warps; span 1.34 us, warp 1.29 us, nsys graphs 1.47 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.77 | 59.4% | 1.07 | 68.2% |
| load | 0.48 | 37.4% | 0.45 | 28.7% |
| store | 0.04 | 3.2% | 0.05 | 3.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.48 | 37.4% | 1.0 | 0.51 | `if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < tokens)) \| int64_t position ` |
| store.global | 0.04 | 3.2% | 1.0 | 0.06 | `*(uint4*)(x + ((((((int64_t)((int)blockIdx.x)) * (int64_t)12288) + ((((int64_t)((int)threa` |

### mhc_post_kernel grid (25, 1, 1) block (128, 1, 1)

60 launches, 100 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 2.05 us, warp 1.66 us, nsys graphs 3.02 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.51 | 90.9% | 2.10 | 91.5% |
| load | 0.10 | 6.2% | 0.11 | 4.8% |
| store | 0.05 | 2.9% | 0.08 | 3.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.10 | 6.2% | 1.0 | 0.13 | `*(float4*)(pm + 0) = *(float4*)(post + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int` |
| store.global | 0.05 | 2.9% | 1.0 | 0.06 | `for (int n_1 = 0; n_1 < 4; ++n_1)` |

### indexer_topk_merge_kernel grid (1, 5, 1) block (512, 1, 1)

15 launches, 80 warps; ranges, warp and last warp from the 5 recording CTAs' 80 warps; span 9.25 us, warp 9.02 us, nsys graphs 3.84 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 6.16 | 68.3% | 6.11 | 66.1% |
| unattributed | 2.20 | 24.3% | 2.22 | 24.0% |
| load | 0.59 | 6.5% | 0.77 | 8.4% |
| store | 0.08 | 0.9% | 0.14 | 1.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync#20*9 (est.: sampled iterations x9) | 0.68 | 7.5% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#24*9 (est.: sampled iterations x9) | 0.67 | 7.4% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#19*9 (est.: sampled iterations x9) | 0.67 | 7.4% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#23*9 (est.: sampled iterations x9) | 0.65 | 7.2% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#6*8 (est.: sampled iterations x8) | 0.61 | 6.8% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync#14*8 (est.: sampled iterations x8) | 0.59 | 6.6% | 1.0 | 0.77 | `__syncthreads();` |
| load.global | 0.57 | 6.4% | 1.0 | 0.70 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 8192);` |
| wait.sync#5*8 (est.: sampled iterations x8) | 0.56 | 6.2% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync#13*8 (est.: sampled iterations x8) | 0.56 | 6.2% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync | 0.14 | 1.6% | 1.0 | 0.22 | `__syncthreads();` |
| wait.sync#17 | 0.08 | 0.9% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#16 | 0.08 | 0.8% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#8 | 0.07 | 0.8% | 1.0 | 0.10 | `__syncthreads();` |
| store.global#2 | 0.07 | 0.8% | 0.9 | 0.10 | `Values[((((int64_t)((int)blockIdx.y)) * (int64_t)512) + ((int64_t)((int)threadIdx.x)))] = ` |
| wait.sync#25 | 0.07 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#3 | 0.07 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#2 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#15 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#9 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#11 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#7 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#22 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#12 | 0.06 | 0.7% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#4 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#18 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#21 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#10 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| load.global#2 | 0.01 | 0.1% | 0.1 | 0.10 | `if (((0 <= ((int*)sel_idx)[((int)threadIdx.x)]) && (((int*)sel_idx)[((int)threadIdx.x)] < ` |
| store.global | 0.01 | 0.1% | 0.1 | 0.10 | `Values[((((int64_t)((int)blockIdx.y)) * (int64_t)512) + ((int64_t)((int)threadIdx.x)))] = ` |

### token_ranks_kernel grid (5, 1, 1) block (256, 1, 1)

15 launches, 40 warps; ranges, warp and last warp from the 5 recording CTAs' 40 warps; span 7.71 us, warp 7.68 us, nsys graphs 8.50 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 6.93 | 90.2% | 7.02 | 90.4% |
| unattributed | 0.36 | 4.7% | 0.38 | 4.9% |
| compute | 0.34 | 4.5% | 0.31 | 4.0% |
| store | 0.05 | 0.6% | 0.05 | 0.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 6.93 | 90.2% | 1.0 | 7.14 | `if ((((int64_t)0 <= token_ids[((int64_t)((int)blockIdx.x))]) && (token_ids[((int64_t)((int` |
| compute.reduce | 0.34 | 4.5% | 1.0 | 0.64 | `total[0] = tl::AllReduce<tl::SumOp, 256, 1, 0, tl::NamedBarrier<256>>::run(total[0], (&(wo` |
| store.global | 0.05 | 0.6% | 1.0 | 0.06 | `output[((int64_t)((int)blockIdx.x))] = ((int64_t)total[0]);` |

### per_token_cast_kernel_kernel grid (3, 1, 1) block (64, 1, 1)

96 launches, 6 warps; ranges, warp and last warp from the 3 recording CTAs' 6 warps; span 1.20 us, warp 1.11 us, nsys graphs 1.60 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.46 | 41.8% | 0.42 | 35.9% |
| load | 0.34 | 30.8% | 0.35 | 30.1% |
| store | 0.21 | 18.8% | 0.30 | 25.6% |
| compute | 0.10 | 8.7% | 0.10 | 8.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.26 | 23.1% | 1.0 | 0.26 | `for (int i = 0; i < 2; ++i)` |
| store.global#2 | 0.11 | 10.1% | 1.0 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.10 | 8.7% | 1.0 | 0.13 | `amax_fragment[i_1] = tl::AllReduce<tl::MaxOp, 2, 1, 0, tl::NamedBarrier<64>>::run(amax_fra` |
| load.global#2*2 (est.: sampled iterations x2) | 0.09 | 7.7% | 1.0 | 0.13 | `amax_fragment[i_1] = cutlass::bfloat16_t(__hmax((bfloat16_t(((nv_bfloat162*)(&(amax_fragme` |
| store.global*2 (est.: sampled iterations x2) | 0.05 | 4.8% | 0.5 | 0.13 | `out_sf[((((((int64_t)((int)blockIdx.x)) % (int64_t)15) * (int64_t)32) + (((int64_t)((int)t` |
| store.global#3*2 (est.: sampled iterations x2) | 0.04 | 3.8% | 0.5 | 0.13 | `*(fp8_e4_16_t*)(out + ((((((((int64_t)((int)blockIdx.x)) / (int64_t)3) * (int64_t)6144) + ` |

### indexer_mxfp4_quantize_kernel grid (5, 1, 1) block (128, 1, 1)

30 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 3.74 us, warp 3.69 us, nsys graphs 2.91 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 2.28 | 61.9% | 2.26 | 60.1% |
| store | 1.13 | 30.6% | 1.17 | 31.2% |
| unattributed | 0.28 | 7.5% | 0.33 | 8.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2*16 (est.: sampled iterations x16) | 1.25 | 34.0% | 1.0 | 1.54 | `float v__2 = __fmul_rn(((float)X[(((((int64_t)((int)blockIdx.x)) * (int64_t)4096) + (((int` |
| store.global*16 (est.: sampled iterations x16) | 1.09 | 29.5% | 1.0 | 1.54 | `Pool[(((((slot & (int64_t)127) * (int64_t)64) + ((((int64_t)((int)threadIdx.x)) & (int64_t` |
| load.global | 1.03 | 27.8% | 1.0 | 1.09 | `for (int i = 0; i < 32; ++i)` |
| store.global#2 | 0.04 | 1.1% | 1.0 | 0.06 | `Pool[(((((slot & (int64_t)127) * (int64_t)4) + ((slot >> (int64_t)7) * ((int64_t)pool_stri` |

### gemm_splitk_reduce_cast_kernel grid (24, 2, 1) block (128, 1, 1)

96 launches, 192 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 1.06 us, warp 0.96 us, nsys graphs 1.18 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.43 | 45.3% | 0.45 | 43.9% |
| load | 0.37 | 38.8% | 0.42 | 40.7% |
| compute | 0.08 | 8.4% | 0.07 | 7.2% |
| store | 0.07 | 7.5% | 0.09 | 8.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.37 | 38.8% | 1.0 | 0.42 | `if ((1 <= M))` |
| compute.reduce | 0.08 | 8.4% | 1.0 | 0.10 | `amax[0] = tl::AllReduce<tl::MaxOp, 8, 1, 0, tl::NamedBarrier<128>>::run(amax[0]);` |
| store.global | 0.07 | 7.5% | 1.0 | 0.10 | `if (((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 5)) < M)` |

### activation_swiglu_clamp_kernel grid (1, 10, 1) block (128, 1, 1)

48 launches, 40 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 1.65 us, warp 0.87 us, nsys graphs 5.66 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.76 | 87.6% | 1.65 | 79.9% |
| load | 0.07 | 8.3% | 0.36 | 17.7% |
| store | 0.04 | 4.1% | 0.05 | 2.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.07 | 8.3% | 0.8 | 0.10 | `*(uint4*)(gate + 0) = *(uint4*)(X + ((((int64_t)((int)blockIdx.y)) * (int64_t)1536) + (((i` |
| store.global | 0.04 | 4.1% | 0.8 | 0.06 | `*(uint4*)(Y + ((((int64_t)((int)blockIdx.y)) * (int64_t)768) + (((int64_t)((int)threadIdx.` |

### sparse_mla_kernel grid (5, 1, 6) block (256, 1, 1)

4 launches, 240 warps; ranges, warp and last warp from the 5 recording CTAs' 40 warps; span 23.23 us, warp 19.30 us, nsys graphs 13.28 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed, with loops' unsampled iterations | 10.45 | 54.2% | 10.30 | 41.2% |
| wait | 3.95 | 20.5% | 4.77 | 19.1% |
| load | 2.77 | 14.3% | 7.65 | 30.6% |
| compute | 1.47 | 7.6% | 1.49 | 6.0% |
| store | 0.66 | 3.4% | 0.80 | 3.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync (sampled iterations) | 3.88 | 20.1% | 11.0 | 8.42 | `__syncthreads();` |
| load.global (sampled iterations) | 2.47 | 12.8% | 2.0 | 9.63 | `for (int c = 0; c < 4; ++c) \| for (int i = 0; i < 8; ++i) \| for (int i_1 = 0; i_1 < 8; ++i` |
| compute.mma (sampled iterations) | 1.22 | 6.3% | 2.0 | 1.26 | `for (int ki = 0; ki < 32; ++ki) \| for (int ki_1 = 0; ki_1 < 2; ++ki_1)` |
| store.global | 0.66 | 3.4% | 1.0 | 0.85 | `for (int i_10 = 0; i_10 < 32; ++i_10) \| for (int i_11 = 0; i_11 < 32; ++i_11) \| for (int i` |
| load.cp_async (sampled iterations) | 0.30 | 1.5% | 2.0 | 0.42 | `for (int k_1 = 0; k_1 < 5; ++k_1) \| for (int k_1 = 0; k_1 < 9; ++k_1) \| for (int k_3 = 0; ` |
| compute.reduce (sampled iterations) | 0.25 | 1.3% | 2.0 | 0.26 | `m_g[i_5] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(m_g[i_5], (&((` |
| wait.cp_async (sampled iterations) | 0.07 | 0.4% | 1.0 | 0.10 | `tl::cp_async_wait<0>();` |

### indexer_mxfp4_quantize_kernel grid (1, 1, 1) block (128, 1, 1)

32 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 3.01 us, warp 1.00 us, nsys graphs 1.98 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.38 | 37.6% | 0.56 | 17.9% |
| load | 0.36 | 36.0% | 1.42 | 45.3% |
| store | 0.26 | 26.4% | 1.15 | 36.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2*16 (est.: sampled iterations x16) | 0.26 | 25.6% | 0.2 | 1.02 | `float v__2 = __fmul_rn(((float)X[(((((int64_t)((int)blockIdx.x)) * (int64_t)4096) + (((int` |
| store.global*16 (est.: sampled iterations x16) | 0.26 | 25.6% | 0.2 | 1.02 | `Pool[(((((slot & (int64_t)127) * (int64_t)64) + ((((int64_t)((int)threadIdx.x)) & (int64_t` |
| load.global | 0.10 | 10.4% | 0.2 | 0.42 | `for (int i = 0; i < 32; ++i)` |
| store.global#2 | 0.01 | 0.8% | 0.2 | 0.03 | `Pool[(((((slot & (int64_t)127) * (int64_t)4) + ((slot >> (int64_t)7) * ((int64_t)pool_stri` |

### prepare_dflash_inputs_kernel grid (9, 1, 1) block (256, 1, 1)

48 launches, 72 warps; ranges, warp and last warp from the 5 recording CTAs' 40 warps; span 2.02 us, warp 0.74 us, nsys graphs 2.48 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.26 | 34.8% | 1.42 | 69.5% |
| unattributed | 0.25 | 33.6% | 0.35 | 17.2% |
| store | 0.23 | 31.5% | 0.27 | 13.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.24 | 33.2% | 0.2 | 1.34 | `req = idx_mapping[((int)blockIdx.y)];` |
| store.global | 0.18 | 25.0% | 0.8 | 0.26 | `if (((int)blockIdx.y) == (num_reqs - 1))` |
| store.global#2 | 0.05 | 6.5% | 0.2 | 0.29 | `if (((int)blockIdx.y) == (num_reqs - 1))` |
| load.global#2 | 0.01 | 1.6% | 0.2 | 0.06 | `vllm_store_if_b32((&(out_query_start_loc[((int)blockIdx.y)])), (((int)blockIdx.y) * num_qu` |

### gemm_bf16_partials_kernel grid (1, 40, 1) block (256, 1, 1)

32 launches, 320 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 2.64 us, warp 1.67 us, nsys graphs 3.07 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.21 | 72.5% | 2.06 | 78.4% |
| wait | 0.23 | 13.9% | 0.25 | 9.3% |
| compute | 0.11 | 6.6% | 0.23 | 8.7% |
| load | 0.07 | 4.2% | 0.00 | 0.0% |
| store | 0.05 | 2.9% | 0.09 | 3.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*2 (est.: sampled iterations x2) | 0.11 | 6.6% | 0.5 | 0.26 | `{` |
| wait.sync | 0.11 | 6.4% | 1.0 | 0.13 | `__syncthreads();` |
| load.tma*2 (est.: sampled iterations x2) | 0.07 | 4.2% | 0.5 | 0.19 | `if (tl::tl_shuffle_elect<128>())` |
| wait.mbarrier#2*2 (est.: sampled iterations x2) | 0.06 | 3.7% | 0.5 | 0.13 | `mbarrier[kb_1].wait(0);` |
| wait.mbarrier*2 (est.: sampled iterations x2) | 0.06 | 3.7% | 0.5 | 0.13 | `mbarrier[(kb + 3)].wait(1);` |
| store.global | 0.05 | 2.9% | 0.5 | 0.13 | `for (int i_1 = 0; i_1 < 4; ++i_1) \| for (int i_2 = 0; i_2 < 8; ++i_2) \| for (int i_4 = 0; ` |

### activation_swiglu_clamp_kernel grid (1, 5, 1) block (128, 1, 1)

90 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 0.93 us, warp 0.73 us, nsys graphs 1.82 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.63 | 85.8% | 0.79 | 84.9% |
| load | 0.07 | 9.8% | 0.09 | 10.1% |
| store | 0.03 | 4.4% | 0.05 | 4.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.07 | 9.8% | 0.8 | 0.10 | `*(uint4*)(gate + 0) = *(uint4*)(X + ((((int64_t)((int)blockIdx.y)) * (int64_t)1536) + (((i` |
| store.global | 0.03 | 4.4% | 0.8 | 0.06 | `*(uint4*)(Y + ((((int64_t)((int)blockIdx.y)) * (int64_t)768) + (((int64_t)((int)threadIdx.` |

### sink_merge_kernel grid (1, 1, 1) block (128, 1, 1)

16 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 4.30 us, warp 4.30 us, nsys graphs 4.29 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.92 | 44.5% | 2.08 | 39.4% |
| unattributed | 0.97 | 22.6% | 2.03 | 38.6% |
| wait | 0.82 | 19.0% | 0.46 | 8.8% |
| compute | 0.58 | 13.4% | 0.56 | 10.7% |
| store | 0.02 | 0.6% | 0.13 | 2.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2*3 (est.: sampled iterations x3) | 1.64 | 38.2% | 1.0 | 1.68 | `for (int step = 0; step < ((width + 127) >> 7); ++step) \| for (int step = 0; step < ((widt` |
| compute.reduce*3 (est.: sampled iterations x3) | 0.58 | 13.4% | 1.0 | 0.58 | `float other = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| wait.sync#3*3 (est.: sampled iterations x3) | 0.53 | 12.3% | 1.0 | 0.67 | `__syncthreads();` |
| load.global | 0.22 | 5.2% | 1.0 | 0.22 | `last = sampled[(max((((int64_t)num_sampled[((int64_t)((int)blockIdx.x))]) - (int64_t)1), (` |
| wait.sync#2*3 (est.: sampled iterations x3) | 0.14 | 3.3% | 1.0 | 0.19 | `__syncthreads();` |
| wait.sync*3 (est.: sampled iterations x3) | 0.14 | 3.3% | 1.0 | 0.19 | `__syncthreads();` |
| load.global#3*3 (est.: sampled iterations x3) | 0.05 | 1.1% | 0.2 | 0.19 | `if ((best[0] == -CUDART_INF_F))` |
| store.global*3 (est.: sampled iterations x3) | 0.02 | 0.6% | 0.2 | 0.10 | `alternates[((((int64_t)((int)blockIdx.x)) * (int64_t)3) + ((int64_t)rank))] = condval_31;` |

### gemm_bf16_partials_kernel grid (2, 4, 1) block (256, 1, 1)

32 launches, 64 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 2.00 us, warp 1.38 us, nsys graphs 2.14 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.92 | 66.8% | 1.52 | 68.4% |
| wait | 0.23 | 16.5% | 0.37 | 16.8% |
| compute | 0.11 | 8.3% | 0.23 | 10.5% |
| load | 0.07 | 4.8% | 0.00 | 0.0% |
| store | 0.05 | 3.6% | 0.10 | 4.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.mma*2 (est.: sampled iterations x2) | 0.11 | 8.3% | 0.5 | 0.26 | `{` |
| wait.sync | 0.11 | 8.0% | 1.0 | 0.13 | `__syncthreads();` |
| load.tma*2 (est.: sampled iterations x2) | 0.07 | 4.8% | 0.5 | 0.19 | `if (tl::tl_shuffle_elect<128>())` |
| wait.mbarrier#2*2 (est.: sampled iterations x2) | 0.06 | 4.4% | 0.5 | 0.13 | `mbarrier[kb_1].wait(0);` |
| wait.mbarrier*2 (est.: sampled iterations x2) | 0.06 | 4.1% | 0.5 | 0.13 | `mbarrier[(kb + 3)].wait(1);` |
| store.global | 0.05 | 3.6% | 0.5 | 0.13 | `for (int i_1 = 0; i_1 < 4; ++i_1) \| for (int i_2 = 0; i_2 < 8; ++i_2) \| for (int i_4 = 0; ` |

### sink_candidates_kernel grid (1, 127, 1) block (128, 1, 1)

16 launches, 508 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 4.34 us, warp 3.65 us, nsys graphs 4.26 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.47 | 40.3% | 1.50 | 34.8% |
| compute | 0.91 | 24.9% | 0.94 | 21.9% |
| load | 0.56 | 15.3% | 1.18 | 27.3% |
| wait | 0.46 | 12.5% | 0.38 | 8.7% |
| store | 0.26 | 7.0% | 0.31 | 7.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.reduce*4 (est.: sampled iterations x4) | 0.66 | 18.0% | 1.0 | 0.77 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| load.global | 0.56 | 15.3% | 1.0 | 0.61 | `local[0] = max((num_sampled[((int64_t)((int)blockIdx.x))] - 1), 0);` |
| wait.sync#2*4 (est.: sampled iterations x4) | 0.26 | 7.1% | 1.0 | 0.38 | `__syncthreads();` |
| store.global*4 (est.: sampled iterations x4) | 0.26 | 7.0% | 1.0 | 0.38 | `if (((int)threadIdx.x) == 0)` |
| compute.reduce#2*4 (est.: sampled iterations x4) | 0.25 | 6.9% | 1.0 | 0.32 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 2, 32);` |
| wait.sync*4 (est.: sampled iterations x4) | 0.20 | 5.4% | 1.0 | 0.26 | `__syncthreads();` |

### rope_kernel grid (10, 1, 1) block (192, 1, 1)

48 launches, 60 warps; ranges, warp and last warp from the 5 recording CTAs' 30 warps; span 1.34 us, warp 1.29 us, nsys graphs 1.50 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.77 | 59.8% | 0.88 | 62.6% |
| load | 0.48 | 37.1% | 0.48 | 34.2% |
| store | 0.04 | 3.1% | 0.04 | 3.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.48 | 37.1% | 1.0 | 0.48 | `if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < tokens)) \| int64_t position ` |
| store.global | 0.04 | 3.1% | 1.0 | 0.06 | `*(uint4*)(x + ((((((int64_t)((int)blockIdx.x)) * (int64_t)12288) + ((((int64_t)((int)threa` |

### moe_combine_kernel grid (5, 10, 1) block (128, 1, 1)

48 launches, 200 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 1.41 us, warp 1.10 us, nsys graphs 1.50 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.64 | 58.2% | 0.92 | 66.7% |
| unattributed | 0.40 | 36.8% | 0.40 | 29.3% |
| store | 0.05 | 5.0% | 0.06 | 4.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.64 | 58.2% | 1.0 | 0.93 | `for (int k = 0; k < 3; ++k) \| for (int k = 0; k < 6; ++k) \| if ((((int)blockIdx.y) < share` |
| store.global | 0.05 | 5.0% | 1.0 | 0.06 | `{` |

### indexer_score_kernel grid (128, 5, 1) block (128, 1, 1)

15 launches, 2560 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 4.32 us, warp 1.47 us, nsys graphs 3.55 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.82 | 55.6% | 2.61 | 62.2% |
| unattributed, with loops' unsampled iterations | 0.53 | 35.8% | 1.25 | 29.7% |
| wait | 0.08 | 5.7% | 0.18 | 4.2% |
| compute | 0.03 | 2.2% | 0.13 | 3.0% |
| store | 0.01 | 0.7% | 0.04 | 0.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 (sampled iterations) | 0.40 | 27.2% | 0.2 | 1.63 | `for (int i_1 = 0; i_1 < 32; ++i_1)` |
| load.global | 0.36 | 24.3% | 1.0 | 0.61 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 4096); \| i` |
| load.global#3 (sampled iterations) | 0.06 | 4.1% | 0.2 | 0.42 | `if (((int)threadIdx.x) < 64)` |
| wait.sync#2 (sampled iterations) | 0.05 | 3.5% | 0.2 | 0.38 | `__syncthreads();` |
| compute.mma (sampled iterations) | 0.03 | 2.2% | 0.2 | 0.13 | `for (int ki = 0; ki < 2; ++ki)` |
| wait.sync#3 (sampled iterations) | 0.01 | 0.8% | 0.2 | 0.06 | `__syncthreads();` |
| store.global (sampled iterations) | 0.01 | 0.7% | 0.2 | 0.06 | `if ((((int)threadIdx.x) >> 6) == 0)` |
| wait.sync (sampled iterations) | 0.01 | 0.7% | 0.2 | 0.06 | `__syncthreads();` |
| wait.sync#4 (sampled iterations) | 0.01 | 0.7% | 0.2 | 0.06 | `__syncthreads();` |

### indexer_score_kernel grid (64, 5, 1) block (128, 1, 1)

15 launches, 1280 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 4.26 us, warp 1.93 us, nsys graphs 3.34 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.06 | 55.0% | 2.21 | 51.9% |
| unattributed, with loops' unsampled iterations | 0.67 | 34.9% | 1.36 | 32.0% |
| wait | 0.15 | 7.6% | 0.50 | 11.7% |
| compute | 0.04 | 2.1% | 0.15 | 3.5% |
| store | 0.01 | 0.5% | 0.04 | 1.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.73 | 38.0% | 1.0 | 0.83 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 4096); \| i` |
| load.global#2 (sampled iterations) | 0.30 | 15.4% | 0.2 | 1.22 | `for (int i_1 = 0; i_1 < 32; ++i_1)` |
| wait.sync#3 (sampled iterations) | 0.06 | 3.3% | 0.2 | 0.26 | `__syncthreads();` |
| wait.sync#2 (sampled iterations) | 0.05 | 2.6% | 0.2 | 0.26 | `__syncthreads();` |
| compute.mma (sampled iterations) | 0.04 | 2.1% | 0.2 | 0.16 | `for (int ki = 0; ki < 2; ++ki)` |
| load.global#3 (sampled iterations) | 0.03 | 1.6% | 0.2 | 0.29 | `if (((int)threadIdx.x) < 64)` |
| wait.sync (sampled iterations) | 0.02 | 0.8% | 0.2 | 0.06 | `__syncthreads();` |
| wait.sync#4 (sampled iterations) | 0.02 | 0.8% | 0.2 | 0.10 | `__syncthreads();` |
| store.global (sampled iterations) | 0.01 | 0.5% | 0.2 | 0.06 | `if ((((int)threadIdx.x) >> 6) == 0)` |

### compress_gather_partial_state_kernel grid (2, 1, 1) block (128, 1, 1)

16 launches, 8 warps; ranges, warp and last warp from the 2 recording CTAs' 8 warps; span 4.10 us, warp 4.07 us, nsys graphs 3.04 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 2.32 | 57.0% | 2.51 | 62.8% |
| unattributed | 1.62 | 39.9% | 1.33 | 33.4% |
| store | 0.13 | 3.1% | 0.15 | 3.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 2.19 | 53.8% | 1.0 | 2.19 | `int nt = counts[0];` |
| load.global#2*2 (est.: sampled iterations x2) | 0.13 | 3.1% | 1.0 | 0.13 | `for (int i = 0; i < 4; ++i)` |
| store.global#2 | 0.06 | 1.6% | 1.0 | 0.06 | `if (((int)threadIdx.x) == 0)` |
| store.global*2 (est.: sampled iterations x2) | 0.06 | 1.6% | 1.0 | 0.06 | `if (half == 0)` |

### per_token_cast_kernel_kernel grid (10, 1, 1) block (64, 1, 1)

48 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 10 warps; span 1.18 us, warp 1.07 us, nsys graphs 1.09 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.44 | 41.6% | 0.46 | 37.3% |
| load | 0.31 | 28.7% | 0.36 | 29.6% |
| store | 0.22 | 20.7% | 0.31 | 25.7% |
| compute | 0.10 | 9.0% | 0.09 | 7.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.22 | 20.7% | 1.0 | 0.29 | `for (int i = 0; i < 2; ++i)` |
| store.global#2 | 0.11 | 10.5% | 1.0 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.10 | 9.0% | 1.0 | 0.13 | `amax_fragment[i_1] = tl::AllReduce<tl::MaxOp, 2, 1, 0, tl::NamedBarrier<64>>::run(amax_fra` |
| load.global#2*2 (est.: sampled iterations x2) | 0.09 | 8.1% | 1.0 | 0.13 | `amax_fragment[i_1] = cutlass::bfloat16_t(__hmax((bfloat16_t(((nv_bfloat162*)(&(amax_fragme` |
| store.global*2 (est.: sampled iterations x2) | 0.06 | 5.4% | 0.6 | 0.13 | `out_sf[((((((int64_t)((int)blockIdx.x)) % (int64_t)15) * (int64_t)32) + (((int64_t)((int)t` |
| store.global#3*2 (est.: sampled iterations x2) | 0.05 | 4.8% | 0.6 | 0.13 | `*(fp8_e4_16_t*)(out + ((((((((int64_t)((int)blockIdx.x)) / (int64_t)3) * (int64_t)6144) + ` |

### finish_argmax_kernel grid (2, 1, 1) block (32, 1, 1)

80 launches, 2 warps; ranges, warp and last warp from the 2 recording CTAs' 2 warps; span 0.74 us, warp 0.72 us, nsys graphs 0.90 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.37 | 51.1% | 0.39 | 53.6% |
| compute | 0.13 | 17.8% | 0.12 | 15.8% |
| load | 0.13 | 17.8% | 0.13 | 17.1% |
| store | 0.10 | 13.3% | 0.10 | 13.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| compute.reduce | 0.13 | 17.8% | 1.0 | 0.13 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| load.global | 0.13 | 17.8% | 1.0 | 0.13 | `for (int k = 0; k < 2; ++k) \| if ((((int)threadIdx.x) < parts))` |
| store.global | 0.10 | 13.3% | 1.0 | 0.10 | `if (((int)threadIdx.x) == 0)` |

### gemm_splitk_reduce_cast_kernel grid (24, 3, 1) block (128, 1, 1)

48 launches, 288 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 1.15 us, warp 1.08 us, nsys graphs 1.28 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.46 | 42.6% | 0.49 | 41.4% |
| unattributed | 0.46 | 42.5% | 0.54 | 45.0% |
| compute | 0.08 | 7.6% | 0.08 | 6.9% |
| store | 0.08 | 7.3% | 0.08 | 6.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.46 | 42.6% | 1.0 | 0.51 | `if ((1 <= M))` |
| compute.reduce | 0.08 | 7.6% | 1.0 | 0.10 | `amax[0] = tl::AllReduce<tl::MaxOp, 8, 1, 0, tl::NamedBarrier<128>>::run(amax[0]);` |
| store.global | 0.08 | 7.3% | 1.0 | 0.10 | `if (((((int)blockIdx.y) * 4) + (((int)threadIdx.x) >> 5)) < M)` |

### adaptive_offsets_kernel grid (1, 1, 1) block (256, 1, 1)

15 launches, 8 warps; ranges, warp and last warp from the 1 recording CTAs' 8 warps; span 3.78 us, warp 3.73 us, nsys graphs 1.95 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 1.79 | 48.0% | 1.12 | 29.6% |
| unattributed, with loops' unsampled iterations | 1.41 | 37.9% | 1.51 | 40.0% |
| store | 0.31 | 8.4% | 0.52 | 13.8% |
| load | 0.22 | 5.8% | 0.63 | 16.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync (sampled iterations) | 0.42 | 11.2% | 1.0 | 0.48 | `__syncthreads();` |
| wait.sync#18 (sampled iterations) | 0.35 | 9.3% | 1.0 | 0.42 | `__syncthreads();` |
| load.global#2 (sampled iterations) | 0.13 | 3.4% | 1.0 | 0.48 | `if ((((chunk * 256) + ((int)threadIdx.x)) < num_reqs))` |
| store.global#5 | 0.12 | 3.3% | 1.0 | 0.16 | `for (int chunk_1 = 0; chunk_1 < ((starts_len + 255) >> 8); ++chunk_1)` |
| store.global#2 (sampled iterations) | 0.09 | 2.5% | 1.0 | 0.10 | `if ((((chunk * 256) + ((int)threadIdx.x)) < num_reqs) & (drafted == 0))` |
| load.global (sampled iterations) | 0.08 | 2.1% | 1.0 | 0.10 | `if (((((chunk * 256) + ((int)threadIdx.x)) < num_reqs) & (drafted != 0)))` |
| store.global | 0.07 | 1.9% | 1.0 | 0.10 | `if (((int)threadIdx.x) == 0)` |
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

### per_token_cast_kernel_kernel grid (20, 1, 1) block (64, 1, 1)

48 launches, 40 warps; ranges, warp and last warp from the 4 recording CTAs' 8 warps; span 1.09 us, warp 1.04 us, nsys graphs 0.99 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.36 | 34.2% | 0.38 | 34.9% |
| store | 0.30 | 28.5% | 0.31 | 29.0% |
| unattributed | 0.28 | 27.3% | 0.29 | 27.3% |
| compute | 0.10 | 10.0% | 0.09 | 8.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.27 | 25.8% | 1.0 | 0.29 | `for (int i = 0; i < 2; ++i)` |
| store.global#2 | 0.11 | 10.8% | 1.0 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.10 | 10.0% | 1.0 | 0.13 | `amax_fragment[i_1] = tl::AllReduce<tl::MaxOp, 2, 1, 0, tl::NamedBarrier<64>>::run(amax_fra` |
| store.global*2 (est.: sampled iterations x2) | 0.10 | 9.2% | 1.0 | 0.13 | `out_sf[((((((int64_t)((int)blockIdx.x)) % (int64_t)15) * (int64_t)32) + (((int64_t)((int)t` |
| load.global#2*2 (est.: sampled iterations x2) | 0.09 | 8.5% | 1.0 | 0.13 | `amax_fragment[i_1] = cutlass::bfloat16_t(__hmax((bfloat16_t(((nv_bfloat162*)(&(amax_fragme` |
| store.global#3*2 (est.: sampled iterations x2) | 0.09 | 8.5% | 1.0 | 0.13 | `*(fp8_e4_16_t*)(out + ((((((((int64_t)((int)blockIdx.x)) / (int64_t)3) * (int64_t)6144) + ` |

### cache_writer_kernel grid (8, 1, 1) block (128, 1, 1)

48 launches, 32 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 1.01 us, warp 0.93 us, nsys graphs 1.12 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.32 | 33.9% | 0.33 | 32.4% |
| load | 0.29 | 30.8% | 0.30 | 29.0% |
| store | 0.25 | 27.3% | 0.32 | 31.2% |
| compute | 0.07 | 8.0% | 0.08 | 7.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.25 | 27.3% | 1.0 | 0.32 | `if (((int64_t)0 <= slot) & ((slot >> (int64_t)7) < ((int64_t)pages))) \| if (((int64_t)0 <=` |
| load.global | 0.23 | 24.7% | 1.0 | 0.26 | `for (int i = 0; i < 4; ++i)` |
| compute.reduce | 0.07 | 8.0% | 1.0 | 0.10 | `amax = max(amax, __shfl_xor_sync((uint)4294967295, amax, 1, 32));` |
| load.global#2 | 0.06 | 6.1% | 1.0 | 0.06 | `int64_t slot = slots[((int64_t)((int)blockIdx.x))];` |

### verify_kernel grid (1, 1, 1) block (96, 1, 1)

16 launches, 3 warps; ranges, warp and last warp from the 1 recording CTAs' 3 warps; span 2.85 us, warp 2.71 us, nsys graphs 2.34 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.10 | 40.7% | 1.10 | 36.0% |
| unattributed | 0.98 | 36.2% | 1.14 | 37.4% |
| compute | 0.29 | 10.8% | 0.29 | 9.6% |
| wait | 0.17 | 6.3% | 0.22 | 7.3% |
| store | 0.16 | 5.9% | 0.29 | 9.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.04 | 38.4% | 1.0 | 1.06 | `start = cu_num_logits[((int64_t)((int)blockIdx.x))];` |
| compute.reduce | 0.29 | 10.8% | 1.0 | 0.30 | `row_max[0] = vllm_max(row_max[0], __shfl_xor_sync((uint)4294967295, row_max[0], 8, 32));` |
| store.global | 0.16 | 5.9% | 1.0 | 0.29 | `if (((int)threadIdx.x) == 0)` |
| wait.sync#3 | 0.08 | 3.0% | 1.0 | 0.13 | `__syncthreads();` |
| load.global#2 | 0.06 | 2.4% | 1.0 | 0.06 | `target = local_argmax[((((int64_t)row) * argmax_stride) + ((int64_t)best_index[0]))];` |
| wait.sync#2 | 0.05 | 2.0% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync | 0.04 | 1.4% | 1.0 | 0.05 | `__syncthreads();` |

### gemm_splitk_reduce_kernel grid (1, 5, 1) block (128, 1, 1)

60 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 0.83 us, warp 0.83 us, nsys graphs 1.02 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.46 | 55.8% | 0.44 | 54.3% |
| load | 0.32 | 38.3% | 0.32 | 39.6% |
| store | 0.05 | 5.9% | 0.05 | 6.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.32 | 38.3% | 1.0 | 0.32 | `acc[0] = P[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x))` |
| store.global | 0.05 | 5.9% | 1.0 | 0.06 | `C[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x)) * (int64` |

### sparse_mla_kernel grid (1, 2, 6) block (128, 1, 1)

2 launches, 48 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 23.02 us, warp 22.89 us, nsys graphs 14.40 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed, with loops' unsampled iterations | 7.08 | 30.9% | 6.93 | 30.1% |
| load | 6.14 | 26.8% | 8.11 | 35.3% |
| wait | 4.68 | 20.4% | 2.74 | 11.9% |
| compute | 4.66 | 20.3% | 4.74 | 20.6% |
| store | 0.33 | 1.4% | 0.50 | 2.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 | 4.03 | 17.6% | 1.0 | 4.43 | `for (int i = 0; i < 8; ++i)` |
| wait.sync (sampled iterations) | 2.08 | 9.1% | 1.0 | 2.77 | `__syncthreads();` |
| compute.mma*4 (est.: sampled iterations x4) | 1.72 | 7.5% | 1.0 | 1.79 | `for (int ki = 0; ki < 32; ++ki) \| {` |
| compute.mma#2*4 (est.: sampled iterations x4) | 1.18 | 5.2% | 1.0 | 1.22 | `for (int ki_1 = 0; ki_1 < 2; ++ki_1) \| if (((((int)blockIdx.x) * 2) + (c_1 >> 1)) < 3)` |
| compute.reduce*8 (est.: sampled iterations x8) | 0.90 | 3.9% | 1.0 | 1.02 | `m_g[i_4] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(m_g[i_4], (&((` |
| compute.reduce#2*8 (est.: sampled iterations x8) | 0.86 | 3.7% | 1.0 | 0.90 | `row_sum[i_7] = tl::AllReduce<tl::SumOp, 128, 32, 0, tl::NamedBarrier<128>>::run(row_sum[i_` |
| load.global (sampled iterations) | 0.75 | 3.3% | 1.0 | 2.80 | `for (int c = 0; c < 4; ++c) \| for (int i = 0; i < 8; ++i) \| for (int i_1 = 0; i_1 < 8; ++i` |
| load.cp_async#2*4 (est.: sampled iterations x4) | 0.69 | 3.0% | 1.0 | 0.77 | `if (((((int)blockIdx.x) * 2) + (c_1 >> 1)) < 3) \| if ((c_1 < 3) & (c_1 < 3)) \| if (c < 3)` |
| load.cp_async (sampled iterations) | 0.68 | 3.0% | 1.0 | 0.70 | `for (int k_1 = 0; k_1 < 5; ++k_1) \| for (int k_1 = 0; k_1 < 9; ++k_1) \| for (int k_3 = 0; ` |
| wait.sync#8*8 (est.: sampled iterations x8) | 0.39 | 1.7% | 1.0 | 0.51 | `__syncthreads();` |
| wait.sync#10*8 (est.: sampled iterations x8) | 0.37 | 1.6% | 1.0 | 0.51 | `__syncthreads();` |
| store.global | 0.33 | 1.4% | 1.0 | 0.50 | `for (int i_10 = 0; i_10 < 32; ++i_10) \| for (int i_11 = 0; i_11 < 32; ++i_11) \| for (int i` |
| wait.sync#5*4 (est.: sampled iterations x4) | 0.30 | 1.3% | 1.0 | 0.38 | `__syncthreads();` |
| wait.cp_async*4 (est.: sampled iterations x4) | 0.26 | 1.1% | 1.0 | 0.32 | `tl::cp_async_wait<0>();` |
| wait.sync#6*4 (est.: sampled iterations x4) | 0.26 | 1.1% | 1.0 | 0.32 | `__syncthreads();` |
| wait.sync#11*4 (est.: sampled iterations x4) | 0.26 | 1.1% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#4*4 (est.: sampled iterations x4) | 0.24 | 1.1% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#9*4 (est.: sampled iterations x4) | 0.22 | 1.0% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#7*4 (est.: sampled iterations x4) | 0.21 | 0.9% | 1.0 | 0.32 | `__syncthreads();` |
| wait.sync#3 | 0.06 | 0.3% | 1.0 | 0.08 | `__syncthreads();` |
| wait.sync#2 | 0.04 | 0.2% | 1.0 | 0.05 | `__syncthreads();` |

### sink_policy_kernel grid (1, 1, 1) block (32, 1, 1)

16 launches, 1 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 2.51 us, warp 2.51 us, nsys graphs 2.69 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.09 | 43.3% | 1.11 | 42.1% |
| unattributed | 0.88 | 35.0% | 0.88 | 33.4% |
| store | 0.54 | 21.7% | 0.65 | 24.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 | 0.64 | 25.5% | 1.0 | 0.64 | `state[0] = idx_mapping[((int64_t)((int)blockIdx.x))];` |
| store.global#3 | 0.51 | 20.4% | 1.0 | 0.51 | `for (int a = 0; a < 3; ++a) \| recovered_logprobs[((int64_t)((int)blockIdx.x))] = 0x0p+0f/*` |
| load.global | 0.22 | 8.9% | 1.0 | 0.22 | `start[0] = cu_num_logits[((int64_t)((int)blockIdx.x))];` |
| load.global#4 | 0.16 | 6.4% | 1.0 | 0.16 | `for (int r = 0; r < 2; ++r) \| for (int r_2 = 0; r_2 < 8; ++r_2)` |
| load.global#5 | 0.06 | 2.5% | 1.0 | 0.06 | `AtomicAdd((&(counts[0])), draft_zero_flag[0]);` |
| store.global#4 | 0.03 | 1.3% | 1.0 | 0.03 | `events[(((int64_t)((int)blockIdx.x)) * (int64_t)6)] = outcome[0];` |

### compute_slot_mappings_kernel grid (1, 11, 1) block (256, 1, 1)

16 launches, 88 warps; ranges, warp and last warp from the 6 recording CTAs' 48 warps; span 2.62 us, warp 1.45 us, nsys graphs 2.91 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.81 | 55.7% | 0.81 | 31.3% |
| unattributed | 0.64 | 44.3% | 1.78 | 68.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.81 | 55.7% | 1.0 | 0.83 | `int first_token = query_start_loc[0];` |

### norm_forward_and_per_token_cast_kernel_kernel grid (8, 1, 1) block (128, 1, 1)

16 launches, 32 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 2.35 us, warp 2.31 us, nsys graphs 1.76 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 2.21 | 96.0% | 3.55 | 136.4% |
| load | 0.43 | 18.6% | 0.40 | 15.4% |
| compute | 0.13 | 5.7% | 0.13 | 5.1% |
| wait | 0.06 | 2.4% | 0.06 | 2.2% |
| unattributed | 0.00 | 0.0% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*5 (est.: sampled iterations x5) | 1.94 | 84.3% | 1.0 | 2.08 | `for (int j_1 = 0; j_1 < 2; ++j_1)` |
| load.global#2*5 (est.: sampled iterations x5) | 0.22 | 9.5% | 1.0 | 0.32 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 1024) + (((int)threadIdx.x) * 8)` |
| load.global*5 (est.: sampled iterations x5) | 0.21 | 9.1% | 1.0 | 0.32 | `*(uint4*)(x_local + (vi * 8)) = *(uint4*)(x + ((((((int64_t)vi) * (int64_t)1024) + (((int6` |
| store.global*5 (est.: sampled iterations x5) | 0.21 | 9.1% | 1.0 | 0.32 | `for (int j = 0; j < 4; ++j)` |
| compute.reduce | 0.13 | 5.7% | 1.0 | 0.16 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| wait.sync | 0.06 | 2.4% | 1.0 | 0.10 | `__syncthreads();` |
| store.global#2 | 0.05 | 2.0% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| store.global#3 | 0.01 | 0.6% | 0.2 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |

### sink_cut_kernel grid (1, 1, 1) block (128, 1, 1)

32 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 1.26 us, warp 0.55 us, nsys graphs 2.08 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.28 | 51.1% | 0.30 | 23.6% |
| store | 0.14 | 26.3% | 0.47 | 36.7% |
| load | 0.12 | 22.6% | 0.51 | 39.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.14 | 26.3% | 0.6 | 0.48 | `if (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) < num_cut) \| saved[((((int64_t)((int)` |
| load.global | 0.12 | 22.6% | 0.1 | 0.50 | `for (int b = 0; b < num_reqs; ++b)` |

### resample_kernel grid (1, 127, 1) block (128, 1, 1)

16 launches, 508 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 2.85 us, warp 2.11 us, nsys graphs 2.27 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.77 | 84.2% | 2.20 | 86.8% |
| unattributed | 0.29 | 13.8% | 0.30 | 11.8% |
| compute | 0.04 | 2.1% | 0.04 | 1.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.77 | 84.2% | 1.0 | 2.53 | `end = cu_num_logits[(((int64_t)((int)blockIdx.x)) + (int64_t)1)];` |
| compute.reduce | 0.04 | 2.1% | 1.0 | 0.06 | `if ((temp != 0x0p+0f/*0.000000e+00*/) \| bonus)` |

### metadata_pages_kernel grid (5, 1, 1) block (128, 1, 1)

30 launches, 20 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 1.34 us, warp 0.96 us, nsys graphs 1.50 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.73 | 76.0% | 0.80 | 59.3% |
| load | 0.14 | 15.0% | 0.20 | 14.5% |
| store | 0.09 | 9.0% | 0.35 | 26.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.13 | 13.4% | 1.0 | 0.13 | `if ((((offset + ((int)blockIdx.x)) < src_rows) && (0 <= (offset + ((int)blockIdx.x)))))` |
| store.global | 0.09 | 9.0% | 0.2 | 0.35 | `if (((int)threadIdx.x) < 32)` |
| load.global#2 | 0.02 | 1.7% | 0.2 | 0.06 | `if (((req < table_rows) && (((int)threadIdx.x) < table_cols)))` |

### cache_writer_kernel grid (6, 1, 1) block (128, 1, 1)

8 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 4.98 us, warp 4.95 us, nsys graphs 2.69 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 2.43 | 49.1% | 2.28 | 48.8% |
| unattributed | 2.15 | 43.4% | 2.01 | 43.0% |
| load | 0.29 | 6.0% | 0.30 | 6.4% |
| compute | 0.08 | 1.5% | 0.08 | 1.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 2.43 | 49.1% | 1.0 | 2.43 | `if (((int64_t)0 <= slot) & ((slot >> (int64_t)7) < ((int64_t)pages))) \| if (((int64_t)0 <=` |
| load.global | 0.23 | 4.7% | 1.0 | 0.26 | `for (int i = 0; i < 4; ++i)` |
| compute.reduce | 0.08 | 1.5% | 1.0 | 0.10 | `amax = max(amax, __shfl_xor_sync((uint)4294967295, amax, 1, 32));` |
| load.global#2 | 0.06 | 1.2% | 1.0 | 0.06 | `int64_t slot = slots[((int64_t)((int)blockIdx.x))];` |

### norm_forward_and_per_token_cast_kernel_kernel grid (10, 1, 1) block (128, 1, 1)

16 launches, 40 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 2.30 us, warp 2.28 us, nsys graphs 1.82 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 2.12 | 93.1% | 2.15 | 92.8% |
| load | 0.44 | 19.5% | 0.44 | 19.0% |
| compute | 0.13 | 5.7% | 0.13 | 5.5% |
| wait | 0.06 | 2.5% | 0.05 | 2.3% |
| unattributed | 0.00 | 0.0% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*5 (est.: sampled iterations x5) | 1.86 | 81.5% | 1.0 | 2.08 | `for (int j_1 = 0; j_1 < 2; ++j_1)` |
| load.global#2*5 (est.: sampled iterations x5) | 0.23 | 10.2% | 1.0 | 0.32 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 1024) + (((int)threadIdx.x) * 8)` |
| load.global*5 (est.: sampled iterations x5) | 0.21 | 9.3% | 1.0 | 0.32 | `*(uint4*)(x_local + (vi * 8)) = *(uint4*)(x + ((((((int64_t)vi) * (int64_t)1024) + (((int6` |
| store.global*5 (est.: sampled iterations x5) | 0.21 | 9.1% | 1.0 | 0.32 | `for (int j = 0; j < 4; ++j)` |
| compute.reduce | 0.13 | 5.7% | 1.0 | 0.16 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| wait.sync | 0.06 | 2.5% | 1.0 | 0.10 | `__syncthreads();` |
| store.global#2 | 0.04 | 1.9% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| store.global#3 | 0.01 | 0.6% | 0.2 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |

### sink_flags_kernel grid (1, 1, 1) block (128, 1, 1)

16 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 2.21 us, warp 0.72 us, nsys graphs 3.10 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.38 | 53.3% | 1.55 | 70.0% |
| unattributed | 0.25 | 34.4% | 0.32 | 14.3% |
| store | 0.09 | 12.2% | 0.35 | 15.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.38 | 53.3% | 0.2 | 1.54 | `token = logits_indices[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)th` |
| store.global | 0.09 | 12.2% | 0.2 | 0.35 | `flags[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = (` |

### post_update_kernel grid (1, 1, 1) block (32, 1, 1)

16 launches, 1 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 1.81 us, warp 1.81 us, nsys graphs 2.24 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.69 | 38.1% | 0.68 | 33.9% |
| store | 0.58 | 31.9% | 0.78 | 39.0% |
| unattributed | 0.54 | 30.1% | 0.54 | 27.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 | 0.50 | 27.4% | 1.0 | 0.50 | `int count = num_sampled[((((int64_t)((int)blockIdx.x)) * (int64_t)32) + ((int64_t)((int)th` |
| store.global#2 | 0.42 | 23.0% | 1.0 | 0.42 | `if (0 < count)` |
| store.global | 0.16 | 8.8% | 1.0 | 0.16 | `if (0 < count)` |
| load.global | 0.10 | 5.3% | 1.0 | 0.10 | `int64_t req = idx_mapping[((((int64_t)((int)blockIdx.x)) * (int64_t)32) + ((int64_t)((int)` |
| load.global#3 | 0.10 | 5.3% | 1.0 | 0.10 | `if ((0 < count))` |

### gather_block_tables_kernel grid (11, 1, 1) block (256, 1, 1)

16 launches, 88 warps; ranges, warp and last warp from the 6 recording CTAs' 48 warps; span 1.89 us, warp 1.34 us, nsys graphs 2.11 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.03 | 76.9% | 1.56 | 83.7% |
| unattributed | 0.31 | 23.1% | 0.30 | 16.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.03 | 76.9% | 1.0 | 1.58 | `int g = chunk_groups[((int64_t)((int)blockIdx.x))];` |

### dspark_confidence_kernel grid (10, 1, 1) block (128, 1, 1)

16 launches, 40 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 1.82 us, warp 1.61 us, nsys graphs 1.98 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.25 | 77.4% | 1.35 | 72.7% |
| load | 0.17 | 10.5% | 0.25 | 13.3% |
| compute | 0.09 | 5.9% | 0.09 | 5.1% |
| wait | 0.08 | 4.8% | 0.08 | 4.2% |
| store | 0.02 | 1.4% | 0.09 | 4.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.16 | 9.9% | 1.0 | 0.18 | `for (int c = 0; c < 5; ++c)` |
| compute.reduce | 0.09 | 5.9% | 1.0 | 0.10 | `acc[0] = tl::warp_reduce_sum(acc[0]);` |
| wait.sync | 0.08 | 4.8% | 1.0 | 0.13 | `__syncthreads();` |
| store.global | 0.02 | 1.4% | 0.2 | 0.10 | `out[((int64_t)((int)blockIdx.x))] = (0x1p+0f/*1.000000e+00*/ / (0x1p+0f/*1.000000e+00*/ + ` |
| load.global#2 | 0.01 | 0.6% | 0.2 | 0.06 | `total[0] = 0x0p+0f/*0.000000e+00*/;` |

### rope_kernel grid (5, 1, 1) block (256, 1, 1)

30 launches, 40 warps; ranges, warp and last warp from the 5 recording CTAs' 40 warps; span 0.98 us, warp 0.88 us, nsys graphs 1.30 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.59 | 66.7% | 0.69 | 70.1% |
| load | 0.25 | 28.8% | 0.26 | 25.9% |
| store | 0.04 | 4.5% | 0.04 | 4.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.25 | 28.8% | 1.0 | 0.26 | `if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < tokens)) \| int64_t position ` |
| store.global | 0.04 | 4.5% | 1.0 | 0.06 | `*(uint4*)(x + ((((((int64_t)((int)blockIdx.x)) * (int64_t)12288) + ((((int64_t)((int)threa` |

### metadata_decode_kernel grid (6, 2, 1) block (128, 1, 1)

4 launches, 48 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 6.45 us, warp 4.84 us, nsys graphs 1.73 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 2.32 | 47.9% | 3.99 | 57.1% |
| unattributed | 2.23 | 46.0% | 2.02 | 28.8% |
| store | 0.29 | 6.1% | 0.98 | 14.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.45 | 29.9% | 0.5 | 3.52 | `if ((((offset + ((int)blockIdx.x)) < src_rows) && (0 <= (offset + ((int)blockIdx.x)))))` |
| load.global#3 | 0.85 | 17.5% | 0.5 | 1.73 | `if ((((offset + ((int)blockIdx.x)) < main_src_rows) && (0 <= (offset + ((int)blockIdx.x)))` |
| store.global#4 | 0.26 | 5.3% | 0.5 | 1.82 | `if (0 < live)` |
| store.global | 0.03 | 0.6% | 0.5 | 0.06 | `swa[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = con` |
| load.global#2 | 0.02 | 0.5% | 0.1 | 0.19 | `if (((0 <= req) & ((int64_t)0 <= pos)))` |
| store.global#2 | 0.01 | 0.1% | 0.1 | 0.05 | `lengths[((int64_t)((int)blockIdx.x))] = ((int)max(info[1], (int64_t)0));` |
| store.global#3 | 0.00 | 0.1% | 0.1 | 0.03 | `top_lengths[((int64_t)((int)blockIdx.x))] = condval_6; \| top_lengths[((int64_t)((int)block` |

### insert_kernel grid (1, 1, 1) block (32, 1, 1)

16 launches, 1 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 1.57 us, warp 1.57 us, nsys graphs 1.38 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.70 | 44.9% | 0.78 | 44.9% |
| store | 0.43 | 27.6% | 0.50 | 28.6% |
| unattributed | 0.40 | 25.5% | 0.42 | 24.0% |
| compute | 0.03 | 2.0% | 0.04 | 2.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#3 | 0.51 | 32.7% | 1.0 | 0.51 | `int state = mapping[((int64_t)position)]; \| int64_t state = mapping[((int64_t)position)];` |
| store.global | 0.43 | 27.6% | 1.0 | 0.43 | `counts[((int64_t)((int)blockIdx.x))] = (count_local + 1); \| counts[((int64_t)((int)blockId` |
| load.global | 0.13 | 8.2% | 1.0 | 0.13 | `for (int k = 0; k < 4; ++k)` |
| load.global#2 | 0.06 | 4.1% | 1.0 | 0.06 | `count_local = counts[((int64_t)((int)blockIdx.x))];` |
| compute.reduce | 0.03 | 2.0% | 1.0 | 0.03 | `int count = __shfl_sync((uint)4294967295, count_local, 0, 32);` |

### context_inputs_kernel grid (5, 8, 1) block (128, 1, 1)

16 launches, 160 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 1.54 us, warp 1.24 us, nsys graphs 1.95 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.82 | 65.9% | 1.27 | 75.9% |
| unattributed | 0.34 | 27.7% | 0.34 | 20.2% |
| store | 0.08 | 6.4% | 0.06 | 3.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.82 | 65.9% | 1.0 | 1.12 | `vllm_dspark_context_b128((&(aux[(((((int64_t)((int)blockIdx.y)) * (int64_t)15360) + (((int` |
| store.global | 0.08 | 6.4% | 1.0 | 0.16 | `if (((((int)blockIdx.x) == 0) & (((int)threadIdx.x) == 0)) & (num_tokens <= ((int)blockIdx` |

### sink_rows_kernel grid (1, 64, 1) block (256, 1, 1)

16 launches, 512 warps; ranges, warp and last warp from the 4 recording CTAs' 32 warps; span 1.57 us, warp 0.96 us, nsys graphs 1.87 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.37 | 38.9% | 0.44 | 26.4% |
| load | 0.31 | 32.4% | 0.44 | 26.0% |
| store | 0.28 | 28.8% | 0.80 | 47.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.31 | 32.0% | 1.0 | 0.38 | `row = (max((((int64_t)num_sampled[((int64_t)((int)blockIdx.x))]) - (int64_t)1), (int64_t)0` |
| store.global | 0.26 | 27.6% | 1.0 | 0.48 | `for (int k_1 = 0; k_1 < 8; ++k_1)` |
| store.global#2 | 0.01 | 1.2% | 0.0 | 0.37 | `tokens[((((int64_t)((int)blockIdx.x)) * (int64_t)3) + ((int64_t)((int)threadIdx.x)))] = to` |
| load.global#2 | 0.00 | 0.4% | 0.0 | 0.11 | `token = max(alternates[((((int64_t)((int)blockIdx.x)) * (int64_t)3) + ((int64_t)((int)thre` |

### assign_draft_budget_kernel grid (1, 1, 1) block (256, 1, 1)

15 launches, 8 warps; ranges, warp and last warp from the 1 recording CTAs' 8 warps; span 1.82 us, warp 1.81 us, nsys graphs 1.86 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 0.80 | 44.5% | 0.46 | 25.4% |
| unattributed, with loops' unsampled iterations | 0.57 | 31.6% | 0.63 | 35.0% |
| load | 0.23 | 12.8% | 0.51 | 28.5% |
| store | 0.20 | 11.1% | 0.20 | 11.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync | 0.54 | 29.9% | 1.0 | 0.61 | `__syncthreads();` |
| load.global | 0.23 | 12.8% | 1.0 | 0.74 | `for (int chunk = 0; chunk < ((num_reqs + 255) >> 8); ++chunk)` |
| store.global | 0.20 | 11.1% | 1.0 | 0.22 | `for (int chunk_2 = 0; chunk_2 < ((num_reqs + 255) >> 8); ++chunk_2)` |
| wait.sync#3 | 0.20 | 11.1% | 1.0 | 0.22 | `__syncthreads();` |
| wait.sync#2 (sampled iterations) | 0.06 | 3.5% | 1.0 | 0.06 | `__syncthreads();` |

### gemm_splitk_reduce_kernel grid (4, 8, 1) block (128, 1, 1)

48 launches, 128 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 0.54 us, warp 0.53 us, nsys graphs 0.70 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.25 | 48.0% | 0.26 | 48.1% |
| load | 0.23 | 44.4% | 0.24 | 44.7% |
| store | 0.04 | 7.6% | 0.04 | 7.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.23 | 44.4% | 1.0 | 0.26 | `acc[0] = P[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x))` |
| store.global | 0.04 | 7.6% | 1.0 | 0.06 | `C[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x)) * (int64` |

### topk_ids_kernel grid (6, 1, 1) block (512, 1, 1)

1 launches, 96 warps; ranges, warp and last warp from the 6 recording CTAs' 96 warps; span 26.46 us, warp 23.69 us, nsys graphs 24.90 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 20.19 | 85.2% | 22.05 | 83.3% |
| unattributed | 1.96 | 8.3% | 1.95 | 7.4% |
| wait | 1.26 | 5.3% | 1.02 | 3.9% |
| store | 0.29 | 1.2% | 1.44 | 5.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 20.19 | 85.2% | 1.0 | 22.94 | `for (int t = 0; t < (vocab >> 14); ++t)` |
| wait.sync | 1.12 | 4.7% | 1.0 | 3.23 | `__syncthreads();` |
| store.global | 0.29 | 1.2% | 1.0 | 4.00 | `if ((((int)threadIdx.x) >> 5) == 0)` |
| wait.sync#2 | 0.08 | 0.3% | 1.0 | 0.16 | `__syncthreads();` |
| wait.sync#3 | 0.06 | 0.2% | 1.0 | 0.10 | `__syncthreads();` |

### combine_sampled_and_draft_tokens_kernel grid (1, 1, 1) block (32, 1, 1)

16 launches, 1 warps; ranges, warp and last warp from the 1 recording CTAs' 1 warps; span 1.41 us, warp 1.41 us, nsys graphs 1.86 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.93 | 65.9% | 1.14 | 69.7% |
| unattributed | 0.38 | 27.3% | 0.40 | 24.5% |
| store | 0.10 | 6.8% | 0.09 | 5.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.93 | 65.9% | 1.0 | 0.93 | `if (((((int)blockIdx.x) * 32) + ((int)threadIdx.x)) < num_reqs) \| req = idx_mapping[((int6` |
| store.global | 0.10 | 6.8% | 1.0 | 0.10 | `if ((prefill < seq) && (1 < num_logits))` |

### moe_swiglu_cast_kernel_kernel grid (16, 36, 1) block (96, 1, 1)

6 launches, 1728 warps; ranges, warp and last warp from the 4 recording CTAs' 12 warps; span 3.58 us, warp 1.27 us, nsys graphs 1.89 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.00 | 78.7% | 3.23 | 85.6% |
| load | 0.20 | 15.8% | 0.25 | 6.5% |
| compute | 0.04 | 2.9% | 0.17 | 4.5% |
| store | 0.03 | 2.5% | 0.13 | 3.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.20 | 15.8% | 1.0 | 0.26 | `for (int r = 0; r < 2; ++r)` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.04 | 2.9% | 0.2 | 0.19 | `if (0 <= orders[r_2])` |
| store.global*2 (est.: sampled iterations x2) | 0.03 | 2.5% | 0.2 | 0.13 | `if (0 <= orders[r_2])` |

### per_token_cast_kernel_kernel grid (60, 1, 1) block (64, 1, 1)

16 launches, 120 warps; ranges, warp and last warp from the 4 recording CTAs' 8 warps; span 1.34 us, warp 1.15 us, nsys graphs 1.25 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.42 | 36.4% | 0.55 | 39.5% |
| unattributed | 0.33 | 29.1% | 0.44 | 31.8% |
| store | 0.30 | 26.1% | 0.29 | 21.0% |
| compute | 0.10 | 8.4% | 0.11 | 7.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.33 | 28.7% | 1.0 | 0.46 | `for (int i = 0; i < 2; ++i)` |
| store.global#2 | 0.11 | 9.8% | 1.0 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3)` |
| store.global*2 (est.: sampled iterations x2) | 0.10 | 9.1% | 1.0 | 0.13 | `out_sf[((((((int64_t)((int)blockIdx.x)) % (int64_t)15) * (int64_t)32) + (((int64_t)((int)t` |
| compute.reduce*2 (est.: sampled iterations x2) | 0.10 | 8.4% | 1.0 | 0.13 | `amax_fragment[i_1] = tl::AllReduce<tl::MaxOp, 2, 1, 0, tl::NamedBarrier<64>>::run(amax_fra` |
| load.global#2*2 (est.: sampled iterations x2) | 0.09 | 7.7% | 1.0 | 0.13 | `amax_fragment[i_1] = cutlass::bfloat16_t(__hmax((bfloat16_t(((nv_bfloat162*)(&(amax_fragme` |
| store.global#3*2 (est.: sampled iterations x2) | 0.08 | 7.3% | 1.0 | 0.13 | `*(fp8_e4_16_t*)(out + ((((((((int64_t)((int)blockIdx.x)) / (int64_t)3) * (int64_t)6144) + ` |

### mhc_post_kernel grid (50, 1, 1) block (128, 1, 1)

16 launches, 200 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 1.36 us, warp 1.03 us, nsys graphs 1.60 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.86 | 83.2% | 1.10 | 83.4% |
| load | 0.12 | 11.5% | 0.14 | 10.6% |
| store | 0.05 | 5.3% | 0.08 | 6.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.12 | 11.5% | 1.0 | 0.16 | `*(float4*)(pm + 0) = *(float4*)(post + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int` |
| store.global | 0.05 | 5.3% | 1.0 | 0.10 | `for (int n_1 = 0; n_1 < 4; ++n_1)` |

### prepare_pos_seq_lens_kernel grid (2, 1, 1) block (256, 1, 1)

16 launches, 16 warps; ranges, warp and last warp from the 2 recording CTAs' 16 warps; span 1.12 us, warp 0.76 us, nsys graphs 1.31 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.32 | 42.5% | 0.36 | 27.0% |
| load | 0.30 | 40.3% | 0.85 | 63.1% |
| store | 0.13 | 17.2% | 0.13 | 9.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.30 | 40.3% | 0.5 | 0.61 | `int computed = num_computed_tokens[idx_mapping[((int64_t)((int)blockIdx.x))]];` |
| store.global | 0.07 | 8.7% | 0.5 | 0.16 | `for (int it = 0; it < (((max_reqs + 255) - num_reqs) >> 8); ++it)` |
| store.global#2 | 0.06 | 8.5% | 0.5 | 0.13 | `if (((int)threadIdx.x) == 0)` |

### indexer_topk_segment_kernel grid (1, 6, 1) block (512, 1, 1)

2 launches, 96 warps; ranges, warp and last warp from the 6 recording CTAs' 96 warps; span 10.59 us, warp 10.53 us, nsys graphs 5.28 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed, with loops' unsampled iterations | 5.29 | 50.2% | 5.26 | 49.8% |
| wait | 4.00 | 38.0% | 4.03 | 38.1% |
| load | 1.06 | 10.1% | 1.07 | 10.1% |
| store | 0.17 | 1.7% | 0.21 | 2.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.06 | 10.1% | 1.0 | 1.14 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 4096); \| i` |
| wait.sync (sampled iterations) | 1.01 | 9.6% | 14.5 | 1.28 | `__syncthreads();` |
| wait.sync#19*9 (est.: sampled iterations x9) | 0.35 | 3.3% | 0.5 | 0.43 | `__syncthreads();` |
| wait.sync#20*9 (est.: sampled iterations x9) | 0.34 | 3.2% | 0.5 | 0.43 | `__syncthreads();` |
| wait.sync#23*9 (est.: sampled iterations x9) | 0.32 | 3.1% | 0.5 | 0.43 | `__syncthreads();` |
| wait.sync#24*9 (est.: sampled iterations x9) | 0.32 | 3.1% | 0.5 | 0.43 | `__syncthreads();` |
| wait.sync#13*8 (est.: sampled iterations x8) | 0.29 | 2.8% | 0.5 | 0.38 | `__syncthreads();` |
| wait.sync#14*8 (est.: sampled iterations x8) | 0.29 | 2.8% | 0.5 | 0.38 | `__syncthreads();` |
| wait.sync#5*8 (est.: sampled iterations x8) | 0.28 | 2.6% | 0.5 | 0.38 | `__syncthreads();` |
| wait.sync#6*8 (est.: sampled iterations x8) | 0.28 | 2.6% | 0.5 | 0.38 | `__syncthreads();` |
| store.global | 0.17 | 1.7% | 1.5 | 0.24 | `for (int i_3 = 0; i_3 < 4; ++i_3) \| if (((int)threadIdx.x) < total) \| if (((int)threadIdx.` |
| wait.sync#25 | 0.05 | 0.5% | 0.5 | 0.08 | `__syncthreads();` |
| wait.sync#16 | 0.04 | 0.4% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#18 | 0.04 | 0.4% | 0.5 | 0.08 | `__syncthreads();` |
| wait.sync#8 | 0.04 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#17 | 0.04 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#2 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#9 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#11 | 0.03 | 0.3% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#15 | 0.03 | 0.3% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#3 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#22 | 0.03 | 0.3% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#4 | 0.03 | 0.3% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#7 | 0.03 | 0.3% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#21 | 0.03 | 0.3% | 0.5 | 0.03 | `__syncthreads();` |
| wait.sync#10 | 0.03 | 0.3% | 0.5 | 0.05 | `__syncthreads();` |
| wait.sync#12 | 0.03 | 0.3% | 0.5 | 0.03 | `__syncthreads();` |

### flatten_sampled_kernel grid (1, 1, 1) block (128, 1, 1)

16 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 1.12 us, warp 0.81 us, nsys graphs 1.31 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.47 | 57.9% | 0.63 | 49.1% |
| unattributed | 0.32 | 39.1% | 0.40 | 31.1% |
| store | 0.02 | 3.0% | 0.25 | 19.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.41 | 51.0% | 1.0 | 0.42 | `for (int r = 0; r < num_reqs; ++r)` |
| load.global#2 | 0.06 | 6.9% | 0.2 | 0.22 | `int i = (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) - cu_num_logits[req]);` |
| store.global | 0.02 | 3.0% | 0.2 | 0.10 | `flat[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = co` |

### token_counts_kernel grid (1, 1, 1) block (128, 1, 1)

16 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 1.14 us, warp 0.48 us, nsys graphs 1.31 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.28 | 58.0% | 0.34 | 28.3% |
| store | 0.14 | 30.3% | 0.64 | 53.1% |
| load | 0.06 | 11.8% | 0.22 | 18.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.14 | 30.3% | 0.2 | 0.58 | `if (((int)threadIdx.x) == 0) \| sampled[((int64_t)((int)blockIdx.x))] = (((int)(prefill_len` |
| load.global | 0.06 | 11.8% | 0.2 | 0.22 | `int count = sampled[((int64_t)((int)blockIdx.x))];` |

### logits_rows_kernel grid (1, 1, 1) block (128, 1, 1)

16 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 1.15 us, warp 1.10 us, nsys graphs 2.08 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.51 | 46.4% | 0.51 | 43.7% |
| unattributed | 0.46 | 41.3% | 0.30 | 25.6% |
| store | 0.14 | 12.3% | 0.36 | 30.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.51 | 46.4% | 1.0 | 0.51 | `index = logits_indices[min(((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((in` |
| store.global | 0.14 | 12.3% | 1.0 | 0.35 | `if (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) < num_logits)` |

### sparse_mla_kernel_1 grid (24, 6, 1) block (128, 1, 1)

6 launches, 144 CTAs, 576 warps; span 2.18 us, warp 1.63 us, nsys graphs 2.24 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.60 | 97.9% | 1.53 | 50.0% |
| load | 0.02 | 1.5% | 1.32 | 43.2% |
| wait | 0.01 | 0.5% | 0.10 | 3.3% |
| compute | 0.00 | 0.1% | 0.06 | 2.1% |
| store | 0.00 | 0.1% | 0.04 | 1.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.02 | 1.5% | 0.1 | 1.17 | `M = __fmul_rn(Sinks[((int)blockIdx.x)], 0x1.7154764ee6c2fp+0f/*1.442695e+00*/); \| if (((in` |
| wait.sync | 0.01 | 0.5% | 0.1 | 0.43 | `__syncthreads();` |
| compute.reduce | 0.00 | 0.1% | 0.0 | 0.10 | `amax[0] = tl::AllReduce<tl::MaxOp, 8, 1, 0, tl::NamedBarrier<128>>::run(amax[0]);` |
| store.global | 0.00 | 0.1% | 0.0 | 0.06 | `*(uint2*)(Output + (((((int64_t)((int)blockIdx.y)) * (int64_t)12288) + (((int64_t)((int)bl` |
| compute.reduce#3 | 0.00 | 0.0% | 0.0 | 0.00 | `amax[0] = tl::AllReduce<tl::MaxOp, 8, 1, 0, tl::NamedBarrier<128>>::run(amax[0]);` |
| load.global#3 | 0.00 | 0.0% | 0.0 | 0.00 | `M = __fmul_rn(Sinks[((int)blockIdx.x)], 0x1.7154764ee6c2fp+0f/*1.442695e+00*/);` |
| load.global#4 | 0.00 | 0.0% | 0.0 | 0.00 | `if (((int)threadIdx.x) < 32)` |
| store.global#2 | 0.00 | 0.0% | 0.0 | 0.00 | `*(uint2*)(Output + (((((int64_t)((int)blockIdx.y)) * (int64_t)12288) + (((int64_t)((int)bl` |
| wait.sync#12 | 0.00 | 0.0% | 0.0 | 0.00 | `__syncthreads();` |
| wait.sync#13 | 0.00 | 0.0% | 0.0 | 0.00 | `__syncthreads();` |

### scatter_rows_gather_kernel grid (1, 1, 1) block (128, 1, 1)

16 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 0.93 us, warp 0.83 us, nsys graphs 1.82 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.57 | 68.8% | 0.61 | 53.8% |
| store | 0.15 | 18.3% | 0.41 | 36.4% |
| load | 0.11 | 13.0% | 0.11 | 9.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.15 | 18.3% | 1.0 | 0.42 | `if (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) < (num_reqs * width))` |
| load.global | 0.11 | 13.0% | 1.0 | 0.13 | `index = idx_mapping[min((((((int64_t)0 <= ((int64_t)width)) && ((int64_t)0 <= rmod_1)) \|\| ` |

### topk_log_softmax_kernel grid (6, 1, 1) block (1024, 1, 1)

1 launches, 192 warps; ranges, warp and last warp from the 6 recording CTAs' 192 warps; span 17.31 us, warp 16.24 us, nsys graphs 16.10 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 13.20 | 81.3% | 13.44 | 77.8% |
| unattributed, with loops' unsampled iterations | 1.59 | 9.8% | 1.73 | 10.0% |
| wait | 1.05 | 6.5% | 1.63 | 9.4% |
| compute | 0.40 | 2.4% | 0.42 | 2.4% |
| store | 0.00 | 0.0% | 0.06 | 0.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 6.71 | 41.3% | 1.0 | 7.10 | `if (0 < (vocab & 2047)) \| if (0 < (vocab & 8191))` |
| load.global#2 | 6.48 | 39.9% | 1.0 | 7.26 | `for (int step_1 = 0; step_1 < (vocab >> 11); ++step_1) \| for (int step_1 = 0; step_1 < (vo` |
| wait.sync#2 | 0.52 | 3.2% | 1.0 | 1.34 | `__syncthreads();` |
| wait.sync | 0.28 | 1.7% | 1.0 | 0.93 | `__syncthreads();` |
| compute.reduce#3 | 0.17 | 1.1% | 1.0 | 0.22 | `ties[0] = vllm_add(ties[0], __shfl_xor_sync((uint)4294967295, ties[0], 16, 32));` |
| compute.reduce | 0.11 | 0.7% | 1.0 | 0.16 | `best[0] = vllm_max(best[0], __shfl_xor_sync((uint)4294967295, best[0], 16, 32));` |
| compute.reduce#2 | 0.11 | 0.7% | 1.0 | 0.16 | `se[0] = vllm_add(se[0], __shfl_xor_sync((uint)4294967295, se[0], 16, 32));` |
| wait.sync#4 | 0.10 | 0.6% | 1.0 | 0.16 | `__syncthreads();` |
| wait.sync#5 | 0.08 | 0.5% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#3 | 0.08 | 0.5% | 1.0 | 0.10 | `__syncthreads();` |
| load.global#3 (sampled iterations) | 0.01 | 0.1% | 0.0 | 0.38 | `float value_32 = vllm_sub(vllm_shifted(logits[((((int64_t)((int)blockIdx.x)) * logits_stri` |
| store.global (sampled iterations) | 0.00 | 0.0% | 0.0 | 0.06 | `output[(((((int64_t)c) * (int64_t)1024) + (((int64_t)((int)blockIdx.x)) * ((int64_t)topk))` |

### metadata_chunk_kernel grid (6, 1, 1) block (128, 1, 1)

2 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 8.58 us, warp 7.09 us, nsys graphs 4.13 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 3.99 | 56.3% | 4.06 | 47.4% |
| load | 3.03 | 42.7% | 4.35 | 50.7% |
| store | 0.07 | 1.0% | 0.16 | 1.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 2.84 | 40.0% | 1.0 | 3.62 | `if ((((offset + ((int)blockIdx.x)) < src_rows) && (0 <= (offset + ((int)blockIdx.x)))))` |
| load.global#2 | 0.19 | 2.6% | 0.2 | 0.75 | `if (((0 <= req) & ((int64_t)0 <= pos)))` |
| store.global | 0.05 | 0.7% | 1.0 | 0.06 | `swa[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)((int)threadIdx.x)))] = con` |
| store.global#2 | 0.01 | 0.2% | 0.2 | 0.06 | `lengths[((int64_t)((int)blockIdx.x))] = ((int)max(info[1], (int64_t)0));` |
| store.global#3 | 0.01 | 0.1% | 0.2 | 0.05 | `top_lengths[((int64_t)((int)blockIdx.x))] = condval_12; \| top_lengths[((int64_t)((int)bloc` |

### mhc_collapse_kernel grid (50, 1, 1) block (128, 1, 1)

16 launches, 200 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 0.90 us, warp 0.76 us, nsys graphs 1.09 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.61 | 80.3% | 0.73 | 82.1% |
| load | 0.10 | 13.7% | 0.11 | 12.3% |
| store | 0.05 | 6.1% | 0.05 | 5.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.10 | 13.7% | 1.0 | 0.13 | `*(float4*)(mv + 0) = *(float4*)(mix + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int6` |
| store.global | 0.05 | 6.1% | 1.0 | 0.06 | `*(uint4*)(out + ((((int64_t)((int)blockIdx.x)) * (int64_t)1024) + (((int64_t)((int)threadI` |

### ced_copy_rows_kernel grid (50, 1, 1) block (128, 1, 1)

16 launches, 200 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 0.90 us, warp 0.84 us, nsys graphs 1.22 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.58 | 69.2% | 0.62 | 70.3% |
| unattributed | 0.26 | 30.8% | 0.26 | 29.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.58 | 69.2% | 1.0 | 0.64 | `for (int k = 0; k < 4; ++k) \| int64_t index = ced_ldg64((&(row_indices[((((int64_t)blockId` |

### ced_requests_kernel grid (1, 1, 1) block (128, 1, 1)

16 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 0.86 us, warp 0.60 us, nsys graphs 1.18 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.34 | 56.0% | 0.33 | 38.4% |
| store | 0.17 | 28.0% | 0.42 | 49.1% |
| load | 0.10 | 16.0% | 0.11 | 12.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.17 | 28.0% | 1.0 | 0.42 | `if (((int)threadIdx.x) < 2) \| if (((int)threadIdx.x) <= 2)` |
| load.global | 0.10 | 16.0% | 1.0 | 0.10 | `uint v_ = tl::load_global_32((&(starts[1]))); \| uint v_ = tl::load_global_32((&(starts[min` |

### moe_combine_kernel grid (5, 6, 1) block (128, 1, 1)

6 launches, 120 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 2.21 us, warp 1.74 us, nsys graphs 1.25 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.29 | 74.1% | 1.74 | 79.5% |
| unattributed | 0.39 | 22.6% | 0.39 | 17.8% |
| store | 0.06 | 3.2% | 0.06 | 2.7% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 1.29 | 74.1% | 1.0 | 1.79 | `for (int k = 0; k < 3; ++k) \| for (int k = 0; k < 6; ++k) \| if ((((int)blockIdx.y) < share` |
| store.global | 0.06 | 3.2% | 1.0 | 0.06 | `{` |

### local_stats_kernel grid (6, 16, 1) block (128, 1, 1)

1 launches, 384 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 12.45 us, warp 12.13 us, nsys graphs 12.19 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 11.11 | 91.6% | 11.36 | 91.3% |
| unattributed | 0.71 | 5.9% | 0.74 | 5.9% |
| compute | 0.18 | 1.5% | 0.16 | 1.3% |
| store | 0.07 | 0.6% | 0.13 | 1.0% |
| wait | 0.06 | 0.5% | 0.06 | 0.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 11.11 | 91.6% | 1.0 | 11.39 | `if (expanded_local_pos[((int64_t)((int)blockIdx.x))] < num_speculative_steps)` |
| compute.reduce | 0.12 | 1.0% | 1.0 | 0.13 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 16, 32);` |
| store.global | 0.07 | 0.6% | 1.0 | 0.13 | `if (((int)threadIdx.x) == 0)` |
| wait.sync | 0.06 | 0.5% | 1.0 | 0.06 | `__syncthreads();` |
| compute.reduce#2 | 0.06 | 0.5% | 1.0 | 0.10 | `other[0] = __shfl_xor_sync((uint)4294967295, best[0], 2, 32);` |

### mhc_collapse_column_kernel grid (25, 1, 1) block (128, 1, 1)

15 launches, 100 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 0.80 us, warp 0.78 us, nsys graphs 1.02 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.63 | 80.5% | 0.62 | 78.0% |
| load | 0.10 | 12.6% | 0.12 | 14.5% |
| store | 0.05 | 7.0% | 0.06 | 7.5% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.10 | 12.6% | 1.0 | 0.13 | `*(float4*)(mv + 0) = *(float4*)(mix + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int6` |
| store.global | 0.05 | 7.0% | 1.0 | 0.06 | `*(uint4*)(out + ((((int64_t)((int)blockIdx.x)) * (int64_t)1024) + (((int64_t)((int)threadI` |

### mhc_collapse_kernel grid (30, 1, 1) block (128, 1, 1)

3 launches, 120 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 3.84 us, warp 3.78 us, nsys graphs 2.27 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 3.63 | 96.0% | 3.69 | 94.0% |
| load | 0.11 | 2.9% | 0.18 | 4.6% |
| store | 0.04 | 1.1% | 0.05 | 1.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.11 | 2.9% | 1.0 | 0.13 | `*(float4*)(mv + 0) = *(float4*)(mix + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int6` |
| store.global | 0.04 | 1.1% | 1.0 | 0.06 | `*(uint4*)(out + ((((int64_t)((int)blockIdx.x)) * (int64_t)1024) + (((int64_t)((int)threadI` |

### padding_rows_kernel grid (1, 1, 1) block (256, 1, 1)

31 launches, 8 warps; ranges, warp and last warp from the 1 recording CTAs' 8 warps; span 0.35 us, warp 0.35 us, nsys graphs 0.77 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.26 | 72.7% | 0.26 | 72.0% |
| store | 0.10 | 27.3% | 0.10 | 28.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global | 0.10 | 27.3% | 1.0 | 0.10 | `if (((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) < rows)` |

### scatter_rows_kernel grid (1, 1, 1) block (128, 1, 1)

16 launches, 4 warps; ranges, warp and last warp from the 1 recording CTAs' 4 warps; span 0.67 us, warp 0.60 us, nsys graphs 0.99 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.40 | 66.9% | 0.41 | 61.9% |
| load | 0.12 | 19.2% | 0.11 | 16.8% |
| store | 0.08 | 13.9% | 0.14 | 21.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.12 | 19.2% | 1.0 | 0.13 | `index = idx_mapping[min((((((int64_t)0 <= ((int64_t)width)) && ((int64_t)0 <= rmod_1)) \|\| ` |
| store.global | 0.08 | 13.9% | 1.0 | 0.13 | `if (((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) < (num_reqs * width))` |

### ced_copy_rows_kernel grid (25, 1, 1) block (128, 1, 1)

15 launches, 100 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 0.70 us, warp 0.62 us, nsys graphs 1.06 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.37 | 60.6% | 0.41 | 60.6% |
| unattributed | 0.24 | 39.4% | 0.27 | 39.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.37 | 60.6% | 1.0 | 0.42 | `for (int k = 0; k < 4; ++k) \| int64_t index = ced_ldg64((&(row_indices[((((int64_t)blockId` |

### indexer_topk_segment_kernel grid (2, 6, 1) block (512, 1, 1)

1 launches, 192 warps; ranges, warp and last warp from the 4 recording CTAs' 64 warps; span 9.82 us, warp 5.17 us, nsys graphs 0.83 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 3.07 | 59.4% | 6.59 | 67.1% |
| unattributed, with loops' unsampled iterations | 1.72 | 33.3% | 2.62 | 26.7% |
| load | 0.33 | 6.4% | 0.48 | 4.9% |
| store | 0.05 | 0.9% | 0.13 | 1.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync#20*9 (est.: sampled iterations x9) | 0.35 | 6.7% | 0.5 | 0.86 | `__syncthreads();` |
| wait.sync#19*9 (est.: sampled iterations x9) | 0.33 | 6.4% | 0.5 | 0.86 | `__syncthreads();` |
| wait.sync#24*9 (est.: sampled iterations x9) | 0.33 | 6.4% | 0.5 | 0.86 | `__syncthreads();` |
| wait.sync#13*8 (est.: sampled iterations x8) | 0.32 | 6.2% | 0.5 | 0.77 | `__syncthreads();` |
| wait.sync#23*9 (est.: sampled iterations x9) | 0.30 | 5.8% | 0.5 | 0.86 | `__syncthreads();` |
| wait.sync#6*8 (est.: sampled iterations x8) | 0.29 | 5.6% | 0.5 | 0.77 | `__syncthreads();` |
| wait.sync#14*8 (est.: sampled iterations x8) | 0.28 | 5.5% | 0.5 | 0.77 | `__syncthreads();` |
| wait.sync#5*8 (est.: sampled iterations x8) | 0.27 | 5.3% | 0.5 | 0.77 | `__syncthreads();` |
| load.global | 0.20 | 3.9% | 1.0 | 0.22 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 4096); \| i` |
| load.global#2 | 0.13 | 2.5% | 0.5 | 0.32 | `for (int i = 0; i < 8; ++i)` |
| wait.sync (sampled iterations) | 0.05 | 1.0% | 0.5 | 0.16 | `__syncthreads();` |
| wait.sync#25 | 0.05 | 0.9% | 0.5 | 0.19 | `__syncthreads();` |
| store.global | 0.05 | 0.9% | 0.5 | 0.13 | `for (int i_3 = 0; i_3 < 4; ++i_3) \| if (((int)threadIdx.x) < total) \| if (((int)threadIdx.` |
| wait.sync#17 | 0.04 | 0.8% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#18 | 0.04 | 0.8% | 0.5 | 0.13 | `__syncthreads();` |
| wait.sync#16 | 0.04 | 0.7% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#8 | 0.04 | 0.7% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#11 | 0.03 | 0.7% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#4 | 0.03 | 0.6% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#10 | 0.03 | 0.6% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#2 | 0.03 | 0.6% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#9 | 0.03 | 0.6% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#15 | 0.03 | 0.6% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#3 | 0.03 | 0.6% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#7 | 0.03 | 0.6% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#12 | 0.03 | 0.6% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#22 | 0.03 | 0.6% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync#21 | 0.03 | 0.5% | 0.5 | 0.06 | `__syncthreads();` |

### indexer_topk_merge_kernel grid (1, 6, 1) block (512, 1, 1)

1 launches, 96 warps; ranges, warp and last warp from the 6 recording CTAs' 96 warps; span 9.25 us, warp 8.98 us, nsys graphs 3.82 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| wait | 6.07 | 67.6% | 6.14 | 66.7% |
| unattributed | 2.24 | 25.0% | 2.11 | 22.9% |
| load | 0.59 | 6.6% | 0.83 | 9.0% |
| store | 0.08 | 0.9% | 0.13 | 1.4% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| wait.sync#20*9 (est.: sampled iterations x9) | 0.68 | 7.6% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#23*9 (est.: sampled iterations x9) | 0.68 | 7.5% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#19*9 (est.: sampled iterations x9) | 0.66 | 7.3% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#24*9 (est.: sampled iterations x9) | 0.64 | 7.1% | 1.0 | 0.86 | `__syncthreads();` |
| wait.sync#14*8 (est.: sampled iterations x8) | 0.59 | 6.6% | 1.0 | 0.77 | `__syncthreads();` |
| load.global | 0.58 | 6.5% | 1.0 | 0.77 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 8192);` |
| wait.sync#6*8 (est.: sampled iterations x8) | 0.57 | 6.3% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync#5*8 (est.: sampled iterations x8) | 0.56 | 6.2% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync#13*8 (est.: sampled iterations x8) | 0.53 | 5.9% | 1.0 | 0.77 | `__syncthreads();` |
| wait.sync | 0.15 | 1.6% | 1.0 | 0.26 | `__syncthreads();` |
| wait.sync#17 | 0.08 | 0.9% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#16 | 0.08 | 0.9% | 1.0 | 0.13 | `__syncthreads();` |
| wait.sync#8 | 0.07 | 0.8% | 1.0 | 0.10 | `__syncthreads();` |
| store.global#2 | 0.07 | 0.8% | 0.9 | 0.10 | `Values[((((int64_t)((int)blockIdx.y)) * (int64_t)512) + ((int64_t)((int)threadIdx.x)))] = ` |
| wait.sync#9 | 0.07 | 0.8% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#2 | 0.07 | 0.8% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#3 | 0.07 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#25 | 0.07 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#15 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#12 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#10 | 0.06 | 0.7% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#22 | 0.06 | 0.7% | 1.0 | 0.10 | `__syncthreads();` |
| wait.sync#7 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#11 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#4 | 0.06 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#18 | 0.05 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| wait.sync#21 | 0.05 | 0.6% | 1.0 | 0.06 | `__syncthreads();` |
| load.global#2 | 0.01 | 0.1% | 0.1 | 0.10 | `if (((0 <= ((int*)sel_idx)[((int)threadIdx.x)]) && (((int*)sel_idx)[((int)threadIdx.x)] < ` |
| store.global | 0.01 | 0.1% | 0.1 | 0.10 | `Values[((((int64_t)((int)blockIdx.y)) * (int64_t)512) + ((int64_t)((int)threadIdx.x)))] = ` |

### rope_kernel grid (6, 1, 1) block (192, 1, 1)

6 launches, 36 warps; ranges, warp and last warp from the 6 recording CTAs' 36 warps; span 1.34 us, warp 1.31 us, nsys graphs 1.54 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.79 | 60.3% | 0.89 | 63.1% |
| load | 0.48 | 36.7% | 0.45 | 32.3% |
| store | 0.04 | 3.0% | 0.06 | 4.6% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.48 | 36.7% | 1.0 | 0.51 | `if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < tokens)) \| int64_t position ` |
| store.global | 0.04 | 3.0% | 1.0 | 0.06 | `*(uint4*)(x + ((((((int64_t)((int)blockIdx.x)) * (int64_t)12288) + ((((int64_t)((int)threa` |

### mhc_post_kernel grid (30, 1, 1) block (128, 1, 1)

4 launches, 120 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 1.97 us, warp 1.78 us, nsys graphs 1.36 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 1.63 | 91.5% | 1.82 | 92.7% |
| load | 0.10 | 5.8% | 0.10 | 5.3% |
| store | 0.05 | 2.7% | 0.04 | 2.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.10 | 5.8% | 1.0 | 0.13 | `*(float4*)(pm + 0) = *(float4*)(post + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int` |
| store.global | 0.05 | 2.7% | 1.0 | 0.06 | `for (int n_1 = 0; n_1 < 4; ++n_1)` |

### token_ranks_kernel grid (6, 1, 1) block (256, 1, 1)

1 launches, 48 warps; ranges, warp and last warp from the 6 recording CTAs' 48 warps; span 7.78 us, warp 7.68 us, nsys graphs 7.97 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 6.90 | 89.9% | 7.01 | 90.1% |
| unattributed | 0.37 | 4.8% | 0.45 | 5.8% |
| compute | 0.37 | 4.8% | 0.26 | 3.3% |
| store | 0.04 | 0.5% | 0.06 | 0.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 6.90 | 89.9% | 1.0 | 7.17 | `if ((((int64_t)0 <= token_ids[((int64_t)((int)blockIdx.x))]) && (token_ids[((int64_t)((int` |
| compute.reduce | 0.37 | 4.8% | 1.0 | 0.74 | `total[0] = tl::AllReduce<tl::SumOp, 256, 1, 0, tl::NamedBarrier<256>>::run(total[0], (&(wo` |
| store.global | 0.04 | 0.5% | 1.0 | 0.06 | `output[((int64_t)((int)blockIdx.x))] = ((int64_t)total[0]);` |

### indexer_mxfp4_quantize_kernel grid (6, 1, 1) block (128, 1, 1)

2 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 3.74 us, warp 3.67 us, nsys graphs 2.90 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 2.27 | 61.9% | 2.35 | 62.8% |
| store | 1.10 | 29.9% | 1.09 | 29.1% |
| unattributed | 0.30 | 8.2% | 0.30 | 8.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2*16 (est.: sampled iterations x16) | 1.25 | 34.0% | 1.0 | 1.54 | `float v__2 = __fmul_rn(((float)X[(((((int64_t)((int)blockIdx.x)) * (int64_t)4096) + (((int` |
| store.global*16 (est.: sampled iterations x16) | 1.06 | 28.8% | 1.0 | 1.54 | `Pool[(((((slot & (int64_t)127) * (int64_t)64) + ((((int64_t)((int)threadIdx.x)) & (int64_t` |
| load.global | 1.02 | 27.9% | 1.0 | 1.09 | `for (int i = 0; i < 32; ++i)` |
| store.global#2 | 0.04 | 1.1% | 1.0 | 0.06 | `Pool[(((((slot & (int64_t)127) * (int64_t)4) + ((slot >> (int64_t)7) * ((int64_t)pool_stri` |

### activation_swiglu_clamp_kernel grid (1, 6, 1) block (128, 1, 1)

6 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 0.93 us, warp 0.74 us, nsys graphs 1.82 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.64 | 86.5% | 0.79 | 85.6% |
| load | 0.07 | 9.8% | 0.10 | 10.3% |
| store | 0.03 | 3.7% | 0.04 | 4.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.07 | 9.8% | 0.8 | 0.10 | `*(uint4*)(gate + 0) = *(uint4*)(X + ((((int64_t)((int)blockIdx.y)) * (int64_t)1536) + (((i` |
| store.global | 0.03 | 3.7% | 0.8 | 0.06 | `*(uint4*)(Y + ((((int64_t)((int)blockIdx.y)) * (int64_t)768) + (((int64_t)((int)threadIdx.` |

### indexer_score_kernel grid (128, 6, 1) block (128, 1, 1)

1 launches, 3072 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 4.67 us, warp 2.30 us, nsys graphs 2.40 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.35 | 58.9% | 2.08 | 59.6% |
| unattributed, with loops' unsampled iterations | 0.70 | 30.4% | 1.09 | 31.2% |
| wait | 0.16 | 6.9% | 0.16 | 4.6% |
| compute | 0.06 | 2.4% | 0.10 | 2.8% |
| store | 0.03 | 1.4% | 0.06 | 1.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 (sampled iterations) | 0.71 | 31.0% | 0.5 | 1.60 | `for (int i_1 = 0; i_1 < 32; ++i_1)` |
| load.global | 0.54 | 23.4% | 1.0 | 0.77 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 4096); \| i` |
| load.global#3 (sampled iterations) | 0.10 | 4.5% | 0.5 | 0.42 | `if (((int)threadIdx.x) < 64)` |
| wait.sync#2 (sampled iterations) | 0.09 | 3.8% | 0.5 | 0.38 | `__syncthreads();` |
| compute.mma (sampled iterations) | 0.06 | 2.4% | 0.5 | 0.13 | `for (int ki = 0; ki < 2; ++ki)` |
| store.global (sampled iterations) | 0.03 | 1.4% | 0.5 | 0.06 | `if ((((int)threadIdx.x) >> 6) == 0)` |
| wait.sync#3 (sampled iterations) | 0.03 | 1.1% | 0.5 | 0.06 | `__syncthreads();` |
| wait.sync (sampled iterations) | 0.02 | 1.0% | 0.5 | 0.10 | `__syncthreads();` |
| wait.sync#4 (sampled iterations) | 0.02 | 0.9% | 0.5 | 0.06 | `__syncthreads();` |

### indexer_score_kernel grid (64, 6, 1) block (128, 1, 1)

1 launches, 1536 warps; ranges, warp and last warp from the 4 recording CTAs' 16 warps; span 4.06 us, warp 2.36 us, nsys graphs 1.76 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 1.11 | 46.9% | 1.70 | 41.7% |
| unattributed, with loops' unsampled iterations | 0.92 | 39.1% | 1.66 | 40.9% |
| wait | 0.22 | 9.2% | 0.51 | 12.6% |
| compute | 0.07 | 3.1% | 0.16 | 3.9% |
| store | 0.04 | 1.7% | 0.03 | 0.8% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global#2 (sampled iterations) | 0.54 | 23.1% | 0.5 | 1.28 | `for (int i_1 = 0; i_1 < 32; ++i_1)` |
| load.global | 0.50 | 21.3% | 1.0 | 0.83 | `int extent = min(max(min(Lengths[((int64_t)((int)blockIdx.y))], Active[0]), 0), 4096); \| i` |
| wait.sync#2 (sampled iterations) | 0.09 | 3.8% | 0.5 | 0.26 | `__syncthreads();` |
| compute.mma (sampled iterations) | 0.07 | 3.1% | 0.5 | 0.16 | `for (int ki = 0; ki < 2; ++ki)` |
| wait.sync#4 (sampled iterations) | 0.06 | 2.7% | 0.5 | 0.16 | `__syncthreads();` |
| load.global#3 (sampled iterations) | 0.06 | 2.6% | 0.5 | 0.29 | `if (((int)threadIdx.x) < 64)` |
| store.global (sampled iterations) | 0.04 | 1.7% | 0.5 | 0.19 | `if ((((int)threadIdx.x) >> 6) == 0)` |
| wait.sync (sampled iterations) | 0.04 | 1.6% | 0.5 | 0.13 | `__syncthreads();` |
| wait.sync#3 (sampled iterations) | 0.03 | 1.1% | 0.5 | 0.06 | `__syncthreads();` |

### gemm_splitk_reduce_kernel grid (1, 6, 1) block (128, 1, 1)

4 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 0.78 us, warp 0.77 us, nsys graphs 1.02 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.39 | 51.2% | 0.39 | 50.5% |
| load | 0.32 | 41.8% | 0.34 | 43.3% |
| store | 0.05 | 7.0% | 0.05 | 6.2% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.32 | 41.8% | 1.0 | 0.34 | `acc[0] = P[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x))` |
| store.global | 0.05 | 7.0% | 1.0 | 0.06 | `C[(((((int64_t)((int)blockIdx.y)) * (int64_t)384) + (((int64_t)((int)blockIdx.x)) * (int64` |

### metadata_pages_kernel grid (6, 1, 1) block (128, 1, 1)

2 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 1.36 us, warp 0.97 us, nsys graphs 1.06 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.75 | 76.7% | 0.82 | 60.0% |
| load | 0.14 | 14.8% | 0.19 | 14.1% |
| store | 0.08 | 8.6% | 0.35 | 25.9% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.13 | 13.1% | 1.0 | 0.13 | `if ((((offset + ((int)blockIdx.x)) < src_rows) && (0 <= (offset + ((int)blockIdx.x)))))` |
| store.global | 0.08 | 8.6% | 0.2 | 0.35 | `if (((int)threadIdx.x) < 32)` |
| load.global#2 | 0.02 | 1.6% | 0.2 | 0.06 | `if (((req < table_rows) && (((int)threadIdx.x) < table_cols)))` |

### norm_forward_and_per_token_cast_kernel_kernel grid (6, 1, 1) block (128, 1, 1)

1 launches, 24 warps; ranges, warp and last warp from the 6 recording CTAs' 24 warps; span 2.53 us, warp 2.50 us, nsys graphs 1.89 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| store | 2.33 | 93.4% | 2.34 | 92.4% |
| load | 0.45 | 17.9% | 0.48 | 19.0% |
| compute | 0.13 | 5.1% | 0.13 | 5.1% |
| wait | 0.06 | 2.3% | 0.06 | 2.5% |
| unattributed | 0.00 | 0.0% | 0.00 | 0.0% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| store.global#4*5 (est.: sampled iterations x5) | 2.07 | 82.7% | 1.0 | 2.08 | `for (int j_1 = 0; j_1 < 2; ++j_1)` |
| load.global*5 (est.: sampled iterations x5) | 0.24 | 9.6% | 1.0 | 0.32 | `*(uint4*)(x_local + (vi * 8)) = *(uint4*)(x + ((((((int64_t)vi) * (int64_t)1024) + (((int6` |
| store.global*5 (est.: sampled iterations x5) | 0.21 | 8.5% | 1.0 | 0.32 | `for (int j = 0; j < 4; ++j)` |
| load.global#2*5 (est.: sampled iterations x5) | 0.21 | 8.3% | 1.0 | 0.32 | `*(uint4*)(weight_local + 0) = *(uint4*)(weight + ((vi_1 * 1024) + (((int)threadIdx.x) * 8)` |
| compute.reduce | 0.13 | 5.1% | 1.0 | 0.13 | `warp_reducer[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<128>>::run(warp_redu` |
| wait.sync | 0.06 | 2.3% | 1.0 | 0.10 | `__syncthreads();` |
| store.global#2 | 0.04 | 1.7% | 1.0 | 0.06 | `sum_shared[(((int)threadIdx.x) >> 3)] = warp_fragment[0]; \| sum_shared[(((int)threadIdx.x)` |
| store.global#3 | 0.01 | 0.4% | 0.2 | 0.06 | `rstd[(((((int64_t)((int)blockIdx.x)) * (int64_t)16) + (((int64_t)((int)threadIdx.x)) >> (i` |

### rope_kernel grid (6, 1, 1) block (256, 1, 1)

2 launches, 48 warps; ranges, warp and last warp from the 6 recording CTAs' 48 warps; span 0.99 us, warp 0.90 us, nsys graphs 1.12 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.61 | 67.9% | 0.69 | 70.5% |
| load | 0.25 | 27.7% | 0.26 | 26.2% |
| store | 0.04 | 4.3% | 0.03 | 3.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.25 | 27.7% | 1.0 | 0.26 | `if ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 3)) < tokens)) \| int64_t position ` |
| store.global | 0.04 | 4.3% | 1.0 | 0.06 | `*(uint4*)(x + ((((((int64_t)((int)blockIdx.x)) * (int64_t)12288) + ((((int64_t)((int)threa` |

### mhc_collapse_column_kernel grid (30, 1, 1) block (128, 1, 1)

1 launches, 120 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 0.80 us, warp 0.77 us, nsys graphs 0.93 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| unattributed | 0.61 | 79.3% | 0.58 | 75.0% |
| load | 0.10 | 13.2% | 0.13 | 16.7% |
| store | 0.06 | 7.5% | 0.06 | 8.3% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.10 | 13.2% | 1.0 | 0.13 | `*(float4*)(mv + 0) = *(float4*)(mix + ((((int64_t)((int)blockIdx.x)) / (int64_t)5) * (int6` |
| store.global | 0.06 | 7.5% | 1.0 | 0.06 | `*(uint4*)(out + ((((int64_t)((int)blockIdx.x)) * (int64_t)1024) + (((int64_t)((int)threadI` |

### ced_copy_rows_kernel grid (30, 1, 1) block (128, 1, 1)

1 launches, 120 warps; ranges, warp and last warp from the 5 recording CTAs' 20 warps; span 0.70 us, warp 0.64 us, nsys graphs 1.17 us

| phase | us a warp | share | last warp us | last warp share |
|---|---:|---:|---:|---:|
| load | 0.39 | 60.5% | 0.42 | 61.9% |
| unattributed | 0.25 | 39.5% | 0.26 | 38.1% |

| range | us a warp | share | entries | longest warp us | first statement |
|---|---:|---:|---:|---:|---|
| load.global | 0.39 | 60.5% | 1.0 | 0.42 | `for (int k = 0; k < 4; ++k) \| int64_t index = ced_ldg64((&(row_indices[((((int64_t)blockId` |
=== timeline and tables: exit 0 in 22 s
=== nsys timeline

## Timeline: 7952 launches in 492299.6 us; the GPU is busy 273581.6 us (56%) and idle 218718.0 us in 4796 gaps; gaps over 200 us split it into 9 segments

| segment | launches | wall us | busy us | busy | first kernel |
|---:|---:|---:|---:|---:|---|
| 0 | 29 | 1403.9 | 52.4 | 4% | fill_rows_kernel |
| 1 | 18 | 885.8 | 364.3 | 41% | mhc_project_streams_kernel |
| 2 | 645 | 26609.2 | 23576.8 | 89% | gemm_mxfp8_kernel |
| 3 | 15 | 1276.7 | 118.3 | 9% | insert_kernel |
| 4 | 1 | 2.7 | 2.7 | 100% | post_update_kernel |
| 5 | 599 | 23238.6 | 22089.2 | 95% | prepare_dflash_inputs_kernel |
| 6 | 1 | 5.7 | 5.7 | 100% | padding_rows_kernel |
| 7 | 23 | 893.8 | 49.5 | 6% | assign_draft_budget_kernel |
| 8 | 6621 | 232509.9 | 227322.6 | 98% | mhc_project_streams_kernel |

| idle us | after | before |
|---:|---|---|
| 108231.5 | resample_kernel | insert_kernel |
| 95265.9 | scatter_rows_kernel | padding_rows_kernel |
| 577.6 | sink_policy_kernel | post_update_kernel |
| 369.9 | indexSelectSmallIndex | mhc_project_streams_kernel |
| 314.5 | padding_rows_kernel | assign_draft_budget_kernel |
| 246.9 | mhc_finalize_kernel | gemm_mxfp8_kernel |
| 244.2 | vectorized_gather_kernel | mhc_project_streams_kernel |
| 222.8 | post_update_kernel | prepare_dflash_inputs_kernel |
| 197.0 | padding_rows_kernel | unrolled_elementwise_kernel |
| 184.0 | insert_kernel | sink_candidates_kernel |

## Kernels by total duration (7952 launches, 6670 in graphs)

| kernel | launches | in graphs | total us | share | mean us |
|---|---:|---:|---:|---:|---:|
| vocab_packed_projection_kernel | 36 | 18 | 144513.4 | 46.4% | 4014.26 |
| gemm_mxfp8_kernel | 864 | 831 | 40846.2 | 13.1% | 47.28 |
| l2_prefetch_kernel | 433 | 433 | 31905.3 | 10.3% | 73.68 |
| moe_grouped_fp8_fp4_gemm_kernel | 324 | 312 | 28490.0 | 9.2% | 87.93 |
| gemm_bf16_kernel | 90 | 90 | 24239.8 | 7.8% | 269.33 |
| mhc_project_streams_kernel | 324 | 312 | 5937.9 | 1.9% | 18.33 |
| gemm_fp8w_bf16_partials_kernel | 162 | 156 | 4993.9 | 1.6% | 30.83 |
| gemm_bf16_partials_kernel | 234 | 224 | 3843.4 | 1.2% | 16.42 |
| sparse_mla_kernel | 162 | 156 | 3754.1 | 1.2% | 23.17 |
| mhc_finalize_kernel | 324 | 312 | 3515.6 | 1.1% | 10.85 |
| gemm_bf16_splitk_kernel | 36 | 34 | 2038.2 | 0.7% | 56.62 |
| rope_kernel | 486 | 465 | 958.9 | 0.3% | 1.97 |
| cache_writer_kernel | 252 | 241 | 940.4 | 0.3% | 3.73 |
| metadata_chunk_kernel | 90 | 88 | 889.8 | 0.3% | 9.89 |
| moe_swiglu_cast_kernel_kernel | 162 | 156 | 864.1 | 0.3% | 5.33 |
=== nsys timeline: exit 0 in 0 s
=== decode steps against the grid's step
## A decode step against the grid's step (17 decode steps, 9 grid runs)

GPU busy a step: 15130.7 us against 9680.6 us (+5450.2); kernel time a step: 17283.1 us against 11538.0 us (+5745.1).

| kernel | launches a step | grid | us a step | grid | difference us | exposed us | last warp: wait / load / compute / store / other | longest last-warp wait |
|---|---:|---:|---:|---:|---:|---:|---|---|
| vocab_packed_projection_kernel | 2 | 1 | 8028.4 | 4037.3 | +3991.1 | 8028.4 | 65% / 27% / 10% / 0% / 0% | wait.cp_async*17 |
| moe_grouped_fp8_fp4_gemm_kernel | 18 | 18 | 1556.6 | 125.0 | +1431.5 | 1320.4 | 58% / 17% / 22% / 1% / 3% | wait.mbarrier*40 |
| l2_prefetch_kernel | 25 | 25 | 1825.7 | 1751.9 | +73.8 | 203.2 | 0% / 100% / 0% / 0% / 0% |  |
| gemm_bf16_kernel | 5 | 5 | 1348.8 | 1292.5 | +56.3 | 1348.8 | 90% / 0% / 4% / 1% / 5% | wait.mbarrier#2 |
| mhc_finalize_kernel | 18 | 18 | 200.7 | 158.9 | +41.8 | 12.4 | 7% / 16% / 10% / 20% / 47% | wait.sync |
| token_ranks_kernel | 4 | 0 | 34.9 | 0.0 | +34.9 | 34.9 | 0% / 88% / 5% / 1% / 7% |  |
| moe_swiglu_cast_kernel_kernel | 9 | 9 | 48.4 | 15.1 | +33.3 | 31.1 | 0% / 10% / 4% / 3% / 82% |  |
| mhc_project_streams_kernel | 18 | 18 | 328.1 | 294.9 | +33.2 | 8.8 | 18% / 32% / 16% / 3% / 31% | wait.sync*4 |
| topk_log_softmax_kernel | 2 | 0 | 32.8 | 0.0 | +32.8 | 32.8 | 4% / 83% / 2% / 0% / 11% | wait.sync#2 |
| router_partials_kernel | 9 | 9 | 45.5 | 17.0 | +28.5 | 0.5 | 0% / 37% / 23% / 3% / 36% |  |
| topk_ids_kernel | 1 | 0 | 20.9 | 0.0 | +20.9 | 20.9 | 4% / 73% / 0% / 14% / 9% | wait.sync |
| sparse_mla_kernel | 9 | 9 | 205.6 | 186.1 | +19.5 | 28.9 | 21% / 21% / 7% / 4% / 48% | wait.sync |
| activation_swiglu_clamp_kernel | 9 | 9 | 35.5 | 18.6 | +16.9 | 1.3 | 0% / 18% / 0% / 2% / 80% |  |
| local_stats_kernel | 1 | 0 | 13.2 | 0.0 | +13.2 | 13.2 | 1% / 92% / 1% / 1% / 5% | wait.sync |
| metadata_decode_kernel | 4 | 4 | 29.5 | 16.4 | +13.1 | 4.4 | 0% / 67% / 0% / 6% / 27% |  |
| moe_combine_kernel | 9 | 9 | 22.1 | 11.9 | +10.2 | 22.1 | 0% / 81% / 0% / 2% / 17% |  |
| per_token_cast_kernel_kernel | 24 | 24 | 43.4 | 34.2 | +9.1 | 17.3 | 0% / 31% / 4% / 8% / 58% |  |
| mhc_post_kernel | 5 | 5 | 16.8 | 8.2 | +8.6 | 4.5 | 0% / 5% / 0% / 4% / 92% |  |
| moe_dispatch_kernel | 9 | 9 | 32.5 | 25.1 | +7.4 | 3.3 | 35% / 2% / 0% / 25% / 37% | wait.sync#7*9 |
| compress_gather_partial_state_kernel | 1 | 1 | 10.9 | 4.0 | +6.9 | 0.0 | 0% / 63% / 0% / 4% / 33% |  |
| cache_writer_kernel | 14 | 11 | 54.1 | 48.0 | +6.1 | 5.9 | 0% / 9% / 2% / 42% / 48% |  |
| indexSelectSmallIndex | 7 | 6 | 27.0 | 21.7 | +5.3 | 27.0 |  |  |
| sink_merge_kernel | 1 | 0 | 5.3 | 0.0 | +5.3 | 5.3 | 9% / 39% / 11% / 3% / 39% | wait.sync*3 |
| rope_kernel | 27 | 24 | 54.1 | 48.8 | +5.3 | 9.7 | 0% / 22% / 0% / 3% / 75% |  |
| sink_candidates_kernel | 1 | 0 | 4.9 | 0.0 | +4.9 | 4.9 | 9% / 27% / 22% / 7% / 35% | wait.sync#2*4 |
| vectorized_elementwise_kernel | 0 | 87 | 0.0 | 96.2 | -96.2 | 0.0 |  |  |
=== decode steps against the grid's step: exit 0 in 0 s
{"output": "/c/iket/step-tp3-c1-20261026T143800Z", "steps": [["decode steps, on, under run-iket", 0], ["records against the dry run's count", 0], ["decode steps, nsys, off, graphs", 0], ["timeline and tables", 0], ["nsys timeline", 0], ["decode steps against the grid's step", 0]]}
