# Fuse Temperature Scaling into TileLang Target Sampling and Block Verification

## Summary

Eliminates the upfront `[num_rows, vocab_size]` FP32 allocation, `bfloat16 -> float32` copy (`torch.empty_like(logits, dtype=torch.float32).copy_(logits)`), and standalone `temperature_kernel` launch in `Sampler.apply_sampling_params` when active requests only apply temperature scaling (`temperature not in (0.0, 1.0)` with no `logit_bias`, `penalties`, `bad_words`, `thinking_budget`, `min_p`, `top_k`, or `top_p`, and `logprobs_mode` not in `PROCESSED_LOGPROBS_MODES`):

1. **Track Non-Temperature In-Place Processors Separately (`Sampler`)**:
   - Adds `Sampler.needs_non_temperature_logits_processing` alongside `Sampler.needs_logits_processing` and `skip_temperature: bool = False` in `Sampler.apply_sampling_params`.
   - Also moves the no-op early return (`rows == 0 or (k is None and p is None)`) before `assert logits.dtype == torch.float32` in `tilelang_topk_topp.apply_top_k_top_p` so uncopied BF16 logits pass through when neither `top_k` nor `top_p` is active.

2. **Target Sampling (`Sampler.sample` + `tilelang_sampling`)**:
   - When only temperature scaling is active, `Sampler.sample` passes `skip_temperature=True` to `apply_sampling_params` and `apply_temperature=True` to `gumbel_sample`, which divides by `temperature` in registers on the fly (`vllm_div_full`).
   - Updates `tilelang_sampling.sampling_calls` to declare both `apply_temperature in (False, True)` variants for cold-boot warmup.

3. **Speculative Block Verification (`RejectionSampler._verify` + `tilelang_rejection`)**:
   - Adds a runtime scalar `apply_temperature: T.int32` parameter to `local_stats_kernel`, `verify_kernel`, and `resample_kernel` in `vllm/v1/worker/gpu/spec_decode/tilelang_rejection.py` (adding zero extra compiled kernel variants).
   - When `apply_temperature != 0` and `temp not in (0.0, 1.0)`, `local_stats_kernel`, `verify_kernel`, and `resample_kernel` scale the upcasted BF16 logits in registers via `vllm_div_full`, saving ~198 MB of global memory traffic and 2 kernel launches per 96-row verification step.

## Verification

Verified on `dgx4` inside `spark-ds41f:19f2c20ed4d6-next-20`:

- `tests/v1/worker/test_gpu_sampler_flags.py`: 16 passed (including `test_temperature_only_sampling_skips_fp32_copy_and_matches_fp32_path` and `test_temperature_only_block_verification_skips_fp32_copy_and_matches`).
- `tests/v1/worker/test_gpu_rejection_sampler_chunking.py`: 5 passed.
- `tests/v1/worker/test_gpu_verified_logprob_token_ids.py`: 28 passed.
- `tests/v1/worker/test_prepare_kernels.py`: 7 passed.
- `tests/v1/spec_decode/test_tilelang_rejection.py`: 427 passed.

## Artifacts

- Patch (against `rc20` `1f18be14470339c1227ed0432b896a8ca09811b5`): [`0001-sample-skip-temperature-fp32-copy.patch`](./0001-sample-skip-temperature-fp32-copy.patch)
