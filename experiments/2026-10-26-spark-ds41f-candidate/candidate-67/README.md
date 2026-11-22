# Candidate `candidate-67`

Date: 2026-11-22.
Base: `candidate-66` (`vLLM 0001..0847`, tree `8ce4b43e546ba0d612f9694f0c3785348326c2da`).

## Delta Over `candidate-66`

1. **vLLM patch `0848`** (`0848-perf-startup-prune-unreachable-TileLang-kernel-decla.patch`):
   - Propagates non-module `_router_indices_dtype` / `_router_max_rows` from `router` to `routed_experts` in `fused_moe/layer.py` and reads it in `fused_moe/tilelang.py`, eliminating `20` dead `int32` `moe_dispatch_v41` kernels (`353 -> 333` model calls).
   - Applies `RUNNER_IDX_DTYPE` (`torch.int64`) and aligned-stride `_runs_for_strides` pruning across `tilelang_logprob.py`, `tilelang_bias.py`, `tilelang_controls.py`, `tilelang_auxiliary.py`, `tilelang_thinking.py`, `tilelang_topk_ids.py`, `tilelang_rejection.py`, `rejection_sampler_utils.py`, and `tilelang_inputs.py`, eliminating `121` unreachable step-kernel specializations (`301 -> 180` step calls; total `654 -> 513`, `-21.6%`) while preserving lazy JIT fallback.
   - Updates the four warmup-declaration unit tests (`test_gpu_tilelang_logprob_softmax.py`, `test_gpu_tilelang_topk_ids.py`, `test_gpu_logprob_ranks.py`, `test_tilelang_rejection.py`) to match the stride-aware runner contract.
2. **vLLM patch `0849`** (`0849-perf-startup-skip-redundant-profile_run-under-pinned.patch`):
   - When `kv_cache_memory_bytes` is pinned, `self._b12x_session is None`, and neither target nor draft model contains an active `@support_torch_compile` submodule (`do_not_compile == False`), reserves persistent workspace lanes via `model_runner._reserve_profile_scratch()` and calls `reclaim_uma_memory(synchronize=False)` instead of running the three throwaway profiling passes (6 dummy vision images + two 6,144-token dummy forwards, `-3.3..3.5 s`).
3. **vLLM patch `0850`** (`0850-perf-startup-warm-index-pack_bf16_weight-and-fuse-bu.patch`):
   - Routes `packed_bf16_histogram`, `packed_bf16_exception_count`, `packed_bf16_pack_rows`, and `packed_bf16_build_screen_weights` through `prepare._prepare_with_warm_index` (`0.01 s` warm load per head).
   - Replaces the 43-chunk x 14-alpha PyTorch eager loop (and 1,200+ `.any()` CUDA-CPU syncs) in `build_screen_weights` with a single-pass shared-memory TileLang kernel (`14.33 ms` steady on `[43136, 5120]`, `831.8x` faster than cold PyTorch, `97/97` `test_deepseek_v41_tilelang_vocab.py` passed).
4. **Live `tp3-triangle` overlay evidence** ([2026-11-22-boot-pruning-and-trim](../../../2026-11-22-boot-pruning-and-trim/README.md), `PR #1308`):
   - First-boot / declared TileLang kernels: **`654 -> 513` (`-141`, `-21.6%`)**; step-kernel cold-index preparation `-45.0%` (`12.11 -> 6.66 s`).
   - Warm boot (`2` clean repeat boots): Worker `load_scratch -> ready` **`51.21 -> 46.17 s` (`-5.04 s`, `-9.8%`)**, inside-container **`67.78 -> 63.00 s` mean (`61.83 s` matched-`pynccl`)** (`-34.1 s` / `-35.1%` vs `candidate-65`), `0/4` greedy diffs, and `4/4` frozen-cache live probes (`logprobs+prompt_logprobs`, sampled `top_p`/`top_k`/penalties, `3,677`-tok prefill in `0.84 s`, `64x64` PNG vision in `0.39-0.40 s`) passed on both boots.

## Build & Verification Status

Built `spark-ds41f:candidate-67` (`sha256:22725ab3a3242b16ea23eb59c5d0ee9031b5c7c6559b7837cf3d7228a8a583d4`, vLLM `0001..0850`, `723` patches, `patch_head = 629d215f7b97d9909eaa31df542a599666fb6188`, `expected_tree = d2fefd95b6bfb59cf3a3d2874b919002b472604d`, `patchset_sha256 = 08f7ffa356629308fcc1b1edf4f9ce622ec7f7c0938e3a8cf1bd0763cc9f3a44`) in `466 s` on `dgx4` (lowest `MemAvailable = 81.6 GiB`).

- **Zero-overlay built-image unit tests on `dgx4` (`--gpus all`)**: **`2,838 passed, 8 skipped, 0 failed`** (`947.92 s`) across all 13 suites (`test_deepseek_v41_renderer_pool.py`, `test_deepseek_v41_postload_startup.py`, `test_deepseek_v41_tilelang_moe.py`, `test_deepseek_v41_tilelang_vocab.py`, `test_gpu_tilelang_logprob_softmax.py`, `test_gpu_tilelang_topk_ids.py`, `test_gpu_logprob_ranks.py`, `test_gpu_logit_bias.py`, `test_gpu_tilelang_inputs.py`, `test_jit_warmup_migration.py`, `test_gpu_verified_logprob_token_ids.py`, `test_tilelang_rejection.py`, `test_adaptive_verification.py`).
- **Distribution**: Identical OCI digest `sha256:22725ab3a3242b16ea23eb59c5d0ee9031b5c7c6559b7837cf3d7228a8a583d4` verified on all four nodes (`dgx1`, `dgx2`, `dgx3`, `dgx4`). Production `spark-ds41f:1120` on `dgx1..dgx3` remained untouched.
