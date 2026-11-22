# Boot Pruning & Trim (`candidate-66` Follow-Up)

Date: 2026-11-22.
Base: `spark-ds41f:candidate-66` (`sha256:bcee79aa3f2a8919415b489535a15b6b9e8a0252e0b9af1000a7b3ae7b4e56d3`, `tp3-triangle`, 16 seats).

## Hypotheses & Isolated Arms

1. **Arm 1 (`tp3-arm1-prune-tilelang.json`) — Redundant First-Boot TileLang Kernel Pruning**:
   - Fixes `fused_moe/tilelang.py` (`layer.runner.router.indices_dtype == torch.int64`) to eliminate `20` dead `int32` `moe_dispatch_v41` kernels (`40 -> 20`).
   - Applies the `RUNNER_IDX_DTYPE` (`int64`) and aligned-stride `_runs_for_strides` pruning (matching `tilelang_sampling.py`) across `tilelang_logprob.py`, `tilelang_bias.py`, `tilelang_controls.py`, `tilelang_auxiliary.py`, `tilelang_thinking.py`, `tilelang_topk_ids.py`, `tilelang_rejection.py`, `rejection_sampler_utils.py`, and `tilelang_inputs.py`, pruning `141` mathematically unreachable specializations (`654 -> 513`, `-21.6%`) while keeping full lazy JIT fallback.
2. **Arm 2 (`tp3-arm2-skip-profile.json`) — Skip Redundant `profile_run` Under Pinned `kv_cache_memory_bytes`**:
   - When `kv_cache_memory_bytes` is pinned and `CompilationMode.NONE` (with no `b12x` session), runs `model_runner._reserve_profile_scratch()` (`~1 ms`) to reserve all workspace lanes before KV allocation, and skips the three throwaway profiling passes (`profile_encoder_cache` on 6 dummy vision images, `_dummy_run(6144)` + `_dummy_sampler_run`, and `_profile_deepseek_v4_attention`).
3. **Arm 3 (`tp3-arm3-fast-vocab.json`) — Warm-Index `pack_bf16_weight` & Fused Single-Pass TileLang `build_screen_weights`**:
   - Routes `packed_bf16_histogram`, `packed_bf16_exception_count`, `packed_bf16_pack_rows`, and `packed_bf16_build_screen_weights` through `prepare._prepare_with_warm_index` so warm boots skip TileLang AST/TIR parsing during `load_model`.
   - Replaces the 43-chunk x 14-alpha PyTorch eager loop (and 1,200+ `.any()` CUDA-CPU syncs) in `build_screen_weights` with a single-pass shared-memory TileLang/CUDA kernel (`14.33 ms` steady on `[43136, 5120]`, `97/97` `test_deepseek_v41_tilelang_vocab.py` passed on `dgx4`).

## Results

Pending isolated live runs on `tp3-triangle` (`dgx1..dgx3`).
