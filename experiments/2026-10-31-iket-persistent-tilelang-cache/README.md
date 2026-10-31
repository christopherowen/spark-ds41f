# Experiment: Persistent TileLang Disk Cache for IKET (`VLLM_DS41_IKET_CACHE_DIR`)

## Hypothesis & Objective

Previously, `iket.enable()` unconditionally called `tilelang.env.disable_cache()` and pointed `TILELANG_CACHE_DIR` at an empty `<output>/tilelang-cache` directory so that instrumented cubins would never overwrite the uninstrumented deployment cache (`~/.tilelang/cache`). A side effect of disabling TileLang's disk cache was that every `run-iket` dry-run and measured pass — and every family slice or step harness run — had to recompile all non-skipped TileLang kernels from scratch (~60–90 s per subprocess invocation).

Furthermore, when TileLang loads a compiled kernel from its disk cache (`KernelCache._load_kernel_from_disk`), it skips `JITKernel._compile_and_create_adapter` and therefore does not invoke the `tilelang_callback_cuda_postproc` callback (`_postproc`) that writes `<output>/sources/<kernel>.<hash>.{cu,json}`. A naive cache enablement would leave `<output>/sources/` empty on warm runs, breaking downstream analysis tools (`iket_table.py`, `iket_perfetto.py`).

This experiment adds an optional persistent IKET cache root (`cache_dir` parameter on `iket.enable()`, wired to `VLLM_DS41_IKET_CACHE_DIR` in `vllm/envs.py` and `vllm/v1/worker/gpu_worker.py`):

1. **Content-addressed config digest (`_cache_digest`)**:
   - Computes a 16-hex-char SHA-256 digest over `(hash(iket.py), ctas, iters, skip)`.
   - Any change to `iket.py`'s annotator or to the sampling configuration (`VLLM_DS41_IKET_CTAS`, `VLLM_DS41_IKET_ITERS`, `VLLM_DS41_IKET_SKIP`) maps to a distinct `<cache_dir>/<config_digest>/` bucket.
2. **Isolated persistent TileLang disk cache**:
   - When `cache_dir` (or `VLLM_DS41_IKET_CACHE_DIR`) is set, `iket.enable()` points `TILELANG_CACHE_DIR` at `<cache_dir>/<config_digest>/tilelang-cache` and calls `enable_cache()`, leaving the uninstrumented deployment cache completely untouched.
   - When `cache_dir` is unset, `iket.enable()` preserves its default behavior (`disable_cache()`).
3. **Source & manifest mirroring across warm runs**:
   - On cache misses, `_postproc` writes `<kernel>.<hash>.cu` and `<kernel>.<hash>.json` both to `<output>/sources/` and to `<cache_dir>/<config_digest>/sources/`.
   - On `iket.enable()`, any existing `.cu` and `.json` files in `<cache_dir>/<config_digest>/sources/` are copied into `<output>/sources/` so warm runs that hit TileLang's disk cache without invoking `_postproc` still populate `<output>/sources/` completely.

## Verification on `dgx4`

Tested on `dgx4` (`spark-ds41f:1030`, holding `/tmp/spark-lab-gpu.lock`):

1. **Unit & GPU integration tests (`tests/kernels/test_deepseek_v41_tilelang_iket.py`)**:
   - `pytest --noconftest -v tests/kernels/test_deepseek_v41_tilelang_iket.py`: **15 passed** (including `test_cache_digest_isolates_sampling_settings` and GPU subprocess test `test_persistent_cache_dir_hits_and_mirrors_sources`).
2. **Cold vs. Warm `iket_hot.py` benchmark across 28 kernels (`mhc`, `allreduce`, `moe` with `ctas=1, iters=1`)**:

| Metric | Cold Run (`cache-bench` empty) | Warm Run (`cache-bench` populated) | Delta |
| :--- | ---: | ---: | ---: |
| `mhc` prepare time (2 kernels) | `2.6 s` | `0.6 s` | `-76.9%` |
| `allreduce` prepare time (1 kernel) | `0.7 s` | `0.1 s` | `-85.7%` |
| `moe` prepare time (25 kernels) | `5.9 s` | `2.5 s` | `-57.6%` |
| Total subprocess wall time (incl. 10 timed iters) | `17.16 s` | `9.14 s` | `-46.7%` |
| `nvcc` compilations invoked | `28` | `0` | `-100%` |
| Manifests in `<output>/sources/*.json` | `28` | `28` (100% match) | Identical |
| Output SHA-256 (`mhc` / `allreduce` / `moe`) | `f41f71d3...` / `9b90023b...` / `d6d36c7c...` | `f41f71d3...` / `9b90023b...` / `d6d36c7c...` | Bit-exact |

## Files

- `patch.diff`: Unified diff against `vllm` (`spark-ds41f-1030` baseline) for `vllm/models/deepseek_v4_1/tilelang/iket.py`, `vllm/envs.py`, `vllm/v1/worker/gpu_worker.py`, and `tests/kernels/test_deepseek_v41_tilelang_iket.py`.
