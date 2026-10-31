# Skip Adaptive Verification Warmup in CUDA Graph Memory-Profiling Pass (`#63431` / `#61509`)

## Upstream Source

- Upstream commits:
  - `vllm-project/vllm@4135590ffe` (`[Bugfix][MRV2] Skip adaptive verification cost profiling in the CUDA graph memory-profiling pass (#63431)`)
  - `vllm-project/vllm@fbe133ff30` (`[Bugfix][MRV2] Don't count adaptive verification profiling cache as CUDA graph memory (#61509)`)

## Problem

In `GPUModelRunner.capture_model(profile_only=...)`, `self.adaptive_verification.warm_tail_shapes(...)` and `self.adaptive_verification.load_draft_budget_kernels()` were executed even when `profile_only=True` (the pre-KV-cache memory-profiling capture pass). Running adaptive verification tail-shape dummy runs during `profile_only=True` wastes startup time on a throwaway KV cache and risks perturbing the CUDA graph memory estimate before the real capture pass runs.

## Changes

- **`vllm/v1/worker/gpu/model_runner.py` (`capture_model`)**:
  - Guard adaptive verification `warm_tail_shapes` and `load_draft_budget_kernels` with `if self.adaptive_verification is not None and not profile_only:`.
- **`tests/v1/worker/test_gpu_model_runner_v2.py`**:
  - Add `test_capture_model_excludes_adaptive_verification_warmup_in_profile_only` covering both `profile_only=False` and `profile_only=True`.

## Verification

Verified in `spark-ds41f:19f2c20ed4d6-next-21` on `dgx4`:
- `pytest -q tests/v1/worker/test_gpu_model_runner_v2.py -k capture_model` (`5 passed`).
