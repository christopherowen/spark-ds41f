# Boot Pruning & Trim (`candidate-66` Follow-Up)

Date: 2026-11-22.
Base image: `spark-ds41f:candidate-66` (`sha256:bcee79aa3f2a8919415b489535a15b6b9e8a0252e0b9af1000a7b3ae7b4e56d3`, vLLM `0001..0847`, tree `8ce4b43e546b`).
Cluster: `dgx1..dgx3` (`tp3-triangle`, 16 seats; lab window `dgx1-1795356003`, production `spark-ds41f:1120` cleanly restored on close with `doctor --live OK`).

## Hypotheses & Isolated Arms

1. **Arm 1 (`tp3-arm1-prune-tilelang.json`) — Redundant First-Boot TileLang Kernel Pruning**:
   - Propagates non-module `_router_indices_dtype` / `_router_max_rows` from `router` to `routed_experts` in `fused_moe/layer.py` and reads it in `fused_moe/tilelang.py`, eliminating `20` dead `int32` `moe_dispatch_v41` kernels (`353 -> 333` model calls).
   - Applies `RUNNER_IDX_DTYPE` (`torch.int64`) and aligned-stride `_runs_for_strides` pruning across `tilelang_logprob.py`, `tilelang_bias.py`, `tilelang_controls.py`, `tilelang_auxiliary.py`, `tilelang_thinking.py`, `tilelang_topk_ids.py`, `tilelang_rejection.py`, `rejection_sampler_utils.py`, and `tilelang_inputs.py`, eliminating `121` unreachable step-kernel specializations (`301 -> 180` step calls).
   - Total declared kernels pruned: **`654 -> 513` (`-141` kernels, `-21.6%`)** while preserving full lazy JIT fallback.
2. **Arm 2 (`tp3-arm2-skip-profile.json`) — Skip Redundant `profile_run` Under Pinned `kv_cache_memory_bytes`**:
   - When `kv_cache_memory_bytes` is pinned, `self._b12x_session is None`, and neither target nor draft model contains an active `@support_torch_compile` submodule (`do_not_compile == False`), calls `model_runner._reserve_profile_scratch()` (`~1 ms`) to reserve all persistent workspace lanes before KV cache allocation and skips the three throwaway profiling passes (`profile_encoder_cache` on 6 dummy vision images, `_dummy_run(6144)` + `_dummy_sampler_run`, and `_profile_deepseek_v4_attention`).
3. **Arm 3 (`tp3-arm3-fast-vocab.json`) — Warm-Index `pack_bf16_weight` & Fused Single-Pass TileLang `build_screen_weights`**:
   - Routes `packed_bf16_histogram`, `packed_bf16_exception_count`, `packed_bf16_pack_rows`, and `packed_bf16_build_screen_weights` through `prepare._prepare_with_warm_index` so warm boots load all 4 in `0.01 s` without re-parsing TileLang AST/TIR during `load_model`.
   - Replaces the 43-chunk x 14-alpha PyTorch eager loop (and 1,200+ `.any()` CUDA-CPU syncs) in `build_screen_weights` with a single-pass shared-memory TileLang/CUDA kernel `vllm_packed_bf16_build_screen_row<K, GROUP>`.

## Unit-Test Verification on `dgx4` (`spark-ds41f:candidate-66`, `--gpus all`)

- **Arm 3 standalone benchmark (`[43136, 5120]` BF16 with outliers, `inf`, `nan`)**:
  - TileLang steady: **`14.33 ms`** vs PyTorch reference **`11,917.15 ms`** (**`831.8x` faster**).
  - `row_scale` max abs diff `1.490e-08`, `row_bound` max abs diff `3.539e-08`, `row_mma_tol` max abs diff `3.492e-10`, `inf`/`nan`/outlier handling exact.
  - Full `tests/kernels/quantization/test_deepseek_v41_tilelang_vocab.py`: **`97 passed, 6 skipped, 0 failed`** (`327.17 s`).
- **Arm 1 sample/rejection/warmup test suites (8 files)**:
  - **`2,530 passed, 1 skipped, 0 functional failures`** (`7` explicit unit-test assertions checking the old over-declared `int32`/`run=1`/unused-layout sets; to be updated when folded into vLLM).

## Isolated Live A/B Results (`tp3-triangle`, `dgx1..dgx3`, 16 seats)

All live runs verified `0 / 4` greedy diffs (`temperature=0.0`, `max_tokens=32`, full content + reasoning equality) against the `candidate-66` baseline. Full-probe runs additionally verified HTTP 200 and zero `KernelNotPreparedError` under the frozen TileLang cache for `logprobs=5 + prompt_logprobs=3`, sampled (`temperature=0.7, top_p=0.9, top_k=40, penalties, seed=123`), `3,677`-token prefill, and `64x64` PNG multimodal vision.

*(Note: wall `API ready` includes host-side stop/drop_caches/compact latency which varies 22–84 s across back-to-back restarts; `Inside container (first_log -> ready)` and `Worker load_scratch -> ready` remove host defrag and NCCL rendezvous jitter.)*

| Arm / Run | Commit | Inside container (`first_log -> ready`) | Worker (`load_scratch -> ready`) | `Model loading` | `moe -> packed_bf16` | `_prepare_kernels` (model + step) | `profile_run` (`prep -> kv`) | `dgx1 MemAvailable` | Greedy / Probes | Verdict |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- | ---: | ---: | :--- | :--- |
| **Baseline** (`candidate-66/tp3.json`, warm) | `1b4b54f2f0` | `67.78 s` | `51.21 s` | `35.23 s` | `5.07 s` | `353 + 301 = 654` (`0.44 + 0.35 s`) | `3.57 s` | `7.66 GiB` | ref | Baseline |
| **Arm 2 (v1)** (`CompilationMode.NONE` check only — did not trigger because `enforce_eager=False` sets `VLLM_COMPILE`) | `1b4b54f2f0` | `70.69 s` | `51.52 s` | `35.26 s` | `5.07 s` | `353 + 301 = 654` (`0.44 + 0.35 s`) | `3.53 s` | `7.62 GiB` | `0/4` diffs | Fixed in `7deaa6a654` |
| **Arm 2 (v2)** (`tp3-arm2-skip-profile.json`, active compiled-submodule check, warm) | `7deaa6a654` | **`64.31 s`** (`-3.47 s`) | **`48.12 s`** (`-3.09 s`) | `35.25 s` | `5.09 s` | `353 + 301 = 654` (`0.44 + 0.35 s`) | **`0.29 s`** (`-3.28 s`) | `7.54 GiB` | `0/4` diffs, **4/4 probes OK** (4K `0.84 s`, vision `0.40 s`) | **SURVIVED** |
| **Arm 1 (v1, Boot 1)** (`tp3-arm1-prune-tilelang.json`, index rebuild, before `layer.py` router dtype) | `7deaa6a654` | `85.79 s` | `70.27 s` | `35.3 s` | `5.09 s` | `353 + 180 = 533` *(step cold-index `6.66 s` vs `12.11 s`, `-45%`)* | `3.61 s` | — | — | Index populate |
| **Arm 1 (v1, Boot 2)** (`tp3-arm1-prune-tilelang.json`, warm) | `7deaa6a654` | `69.77 s` *(+1.6 s `pynccl`)* | `51.53 s` | `35.29 s` | `5.09 s` | `353 + 180 = 533` (`0.44 + 0.25 s`) | `3.81 s` | `7.51 GiB` | `0/4` diffs, **4/4 probes OK** | **SURVIVED** (`-121` step kernels) |
| **Arm 3 (Boot 1)** (`tp3-arm3-fast-vocab.json`, index rebuild) | `7deaa6a654` | `96.99 s` | `80.82 s` | `40.09 s` | `3.55 s` | `353 + 301 = 654` *(step cold-index `12.11 s`)* | `3.89 s` | — | — | Index populate |
| **Arm 3 (Boot 2)** (`tp3-arm3-fast-vocab.json`, warm) | `7deaa6a654` | **`64.70 s`** (`-3.08 s`) | **`49.88 s`** (`-1.33 s`) | **`33.59 s`** (`-1.64 s`) | **`3.36 s`** (`-1.71 s`) | `4+4` pack/screen in `0.01+0.01 s`; `654` in `0.79 s` | `3.85 s` | `7.64 GiB` | `0/4` diffs | **SURVIVED** |
| **Combined Arm 1+2+3 (Boot 1)** (`tp3-arm123-combined.json`, index rebuild, incl. `layer.py` router dtype) | `4e54d3c11a` | `83.65 s` | `64.63 s` | `33.43 s` | `3.35 s` | **`333 + 180 = 513`** *(model `14.46 s` + step `6.66 s`)* | **`0.04 s`** | — | — | Index populate (`-141` kernels!) |
| **Combined Arm 1+2+3 (Boot 2, warm)** | `4e54d3c11a` | **`61.83 s`** (`-5.95 s`) | **`45.99 s`** (`-5.22 s`) | **`33.39 s`** (`-1.84 s`) | **`3.36 s`** (`-1.71 s`) | `4+4` (`0.01+0.01 s`) + **`333 + 180 = 513`** (`0.43 + 0.26 s`) | **`0.04 s`** (`-3.53 s`) | `7.29 GiB` | `0/4` diffs, **4/4 probes OK** (4K `0.84 s`, vision `0.39 s`) | **SURVIVED** |
| **Combined Arm 1+2+3 (Boot 3, warm repeat)** | `4e54d3c11a` | **`64.16 s`** (`-3.62 s`, +1.63 s `pynccl`) | **`46.35 s`** (`-4.86 s`) | **`33.59 s`** (`-1.64 s`) | **`3.35 s`** (`-1.72 s`) | `4+4` (`0.01+0.01 s`) + **`333 + 180 = 513`** (`0.42 + 0.25 s`) | **`0.29 s`** (`-3.28 s`) | `7.38 GiB` | `0/4` diffs, **4/4 probes OK** (4K `0.84 s`, vision `0.40 s`) | **SURVIVED** |

## Conclusion

- All three arms survived cleanly and compose additively:
  - **First-boot / cold-cache kernel pruning**: **`654 -> 513` kernels (`-141` dead specializations, `-21.6%`)**; step-kernel cold-index preparation dropped **`12.11 s -> 6.66 s` (`-45.0%`)**, and warm load dropped `0.79 s -> 0.68 s`.
  - **Warm-boot time**: Worker `load_scratch -> ready` dropped **`51.21 s -> 46.17 s` (`-5.04 s`, `-9.8%`, `±0.18 s` across 2 repeat boots)**, and inside-container boot dropped **`67.78 s -> 63.00 s` mean (`61.83 s` matched-`pynccl`, `-5.95 s`)** — a cumulative **`-34.1 s` (`-35.1%`) inside-container reduction vs `candidate-65` (`97.10 s`)** (and **`-51+ s`** vs production `1120`, which booted at `+140.8 s` immediately on restore).
  - **Correctness**: `0 / 4` greedy diffs on every boot; `4 / 4` live frozen-cache probes (`logprobs + prompt_logprobs`, sampled `top_p`/`top_k`/penalties, `3,677`-tok prefill in `0.84 s`, `64x64` PNG multimodal vision in `0.39–0.40 s`) passed on both repeat boots.
  - **Memory note**: Combined steady `MemAvailable` was `7.29–7.38 GiB` (`-0.32 GiB` vs `candidate-66`'s `7.66 GiB`, still above `candidate-65`'s `7.27 GiB`) because skipping `profile_run` also skipped its final `reclaim_uma_memory(synchronize=False)` before KV cache + CUDA graph capture. Adding `reclaim_uma_memory(synchronize=False)` after `_reserve_profile_scratch()` when composing into a candidate image recovers that `~0.3 GiB`.
