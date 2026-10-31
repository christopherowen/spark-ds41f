# 2026-10-31: Move V2 GPU Worker Triton Reference Kernels Out of Runtime Modules

## Goal

In `vllm/v1/worker/gpu/`, 22 worker modules whose runtime paths have already moved to TileLang (`tilelang_inputs`, `tilelang_sampling`, `tilelang_logprob`, `tilelang_topk_ids`, `tilelang_auxiliary`, `tilelang_controls`) still held 38 dead `@triton.jit` reference kernels, dead `triton_kernel_dispatcher_with_warmup` dispatchers (`_flatten_sampled_launcher`), and module-level `vllm.triton_utils` / `torch._inductor.runtime.triton_helpers` imports in their production module globals solely because parity tests (`test_gpu_tilelang_inputs.py`, `test_gpu_tilelang_sampling.py`, `test_gpu_sampling_controls.py`, `test_gpu_verified_logprob_token_ids.py`) and microbenchmarks (`bench_tilelang_runner_inputs.py`, `bench_tilelang_runner_sampling.py`, `bench_tilelang_runner_controls.py`) import those symbols by name.

This patch moves all 38 superseded Triton reference kernels into `vllm/v1/worker/gpu/_triton_refs.py`, removes the dead `_flatten_sampled_launcher` `triton_kernel_dispatcher_with_warmup` dispatcher from `rejection_sampler.py`, strips module-level `triton` / `tl` imports from the 17 fully TileLang-ported worker modules, and provides lazy module `__getattr__` forwarding so parity tests and microbenchmarks only load `_triton_refs` when explicitly requesting a reference kernel.

## Patch

- `0001-v1-worker-purge-dead-triton-refs.patch` (applies cleanly on `rc20` / `rc19-exact`)
  - `vllm/v1/worker/gpu/_triton_refs.py`: Houses the 38 parity-test Triton reference kernels and helpers (`_gather_block_tables_kernel`, `_compute_slot_mappings_kernel`, `prepare_boundary_capture`, `_apply_write_kernel`, `_load_ptr`, `cp_local_slot`, `_prepare_prefill_inputs_kernel`, `_prepare_pos_seq_lens_kernel`, `_combine_sampled_and_draft_tokens_kernel`, `_get_num_sampled_and_rejected_kernel`, `_post_update_kernel`, `_post_update_num_computed_tokens_kernel`, `_expand_idx_mapping_kernel`, `_num_nans_kernel`, `_bad_words_kernel`, `_temperature_kernel`, `_gumbel_sample_kernel`, `_bias_kernel`, `_topk_log_softmax_kernel`, `_ranks_kernel`, `_fill_logprob_token_ids_kernel`, `_min_p_kernel`, `_penalties_kernel`, `_bincount_kernel`, `_prompt_logprobs_token_ids_kernel`, `_sink_candidates_kernel`, `_sink_policy_kernel`, `_load_effective_token`, `_update_committed_marker_cache_kernel`, `_thinking_budget_kernel`, `_assign_draft_token_budget`, `_prepare_dflash_inputs_kernel`, `_first_argmax`, `_partial_argmax`, `_partial_argmax_logits`, `_finish_argmax`, `_finish_argmax_pair`, `_select_first_argmax`, `_greedy_rejection_kernel`, `_flatten_sampled_kernel`, `_insert_resampled_kernel`, `_apply_grammar_bitmask_kernel`).
  - Removes module-level `vllm.triton_utils` imports from 17 worker modules (`block_table`, `buffer_utils`, `input_batch`, `metrics.logits`, `sample.bad_words`, `sample.logit_bias`, `sample.logprob`, `sample.min_p`, `sample.penalties`, `sample.prompt_logprob`, `sample.sink`, `sample.thinking_budget`, `spec_decode.dflash.speculator`, `spec_decode.dspark.greedy`, `spec_decode.greedy_verification`, `spec_decode.rejection_sampler`, `structured_outputs`).
  - Removes stale `monkeypatch.setattr(logprob.triton, ...)` and `monkeypatch.setattr(rejection_sampler, "_flatten_sampled_launcher", ...)` from `tests/v1/worker/test_gpu_verified_logprob_token_ids.py` and adds `test_tilelang_worker_modules_hold_no_live_triton_reference_kernels`.

## Verification on `dgx4` (`spark-ds41f:19f2c20ed4d6-next-20`)

- `pytest --noconftest -v tests/v1/worker/test_gpu_verified_logprob_token_ids.py`: **29 passed**
- `pytest --noconftest -v tests/v1/worker/test_gpu_sampling_controls.py`: **247 passed**
- `pytest --noconftest -v tests/v1/worker/test_gpu_tilelang_inputs.py -k "test_prepare_prefill_inputs or test_gather_block_tables or test_apply_staged_writes or test_greedy_rejection or test_flatten_sampled or test_dflash_prepare_inputs"`: **145 passed**
- `pytest --noconftest -v tests/v1/worker/test_gpu_tilelang_sampling.py -k "test_gumbel_sample_int32_positions or test_temperature or test_sink_candidates_int32_positions or test_select_first_argmax or test_greedy_rejection"`: **115 passed**
