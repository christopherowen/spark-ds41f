# `candidate-77` — Rebase TileLang onto `v0.1.16` (`640fdcae3cd9`)

## Base and variables

- **Base deployment commit**: `99cc9cba4b` (`spark-ds41f-1126`, promoted baseline `2026-11-26-spark-ds41f-1126`).
- **Single causal variable**: advance `tile-ai/tilelang` from `2123e412ce2217b324c50c4ec967213906456ac5` (`0.1.15+snapshot.35f15411...`) to **`v0.1.16`** (`640fdcae3cd93620e5fcf98d36ee0ef7e17ae1f1`, 65 upstream commits including updated TVM/Z3, multi-source cache `#3807`, process-based `par_compile` `#3903`, toolchain-aware kernel cache keys `#4511`, L2 prefetching `#3490`, symmetric clamp `#3605`, and `redux.sync` cross-warp reductions).
- **TileLang patch stack (`experiments/2026-10-26-spark-ds41f-candidate/candidate-77/tilelang/series`, 13 patches)**:
  - **9 patches applied unchanged**: `0001` (SM120 MXF8F6F4 MMA), `0002` (SM120 MXFP4 MMA, PR `#3495`), `0004` (IKET extended ranges/payloads), `0005` (preserve MX scale-byte selectors), `0006` (async waits, PR `#3568`), `0008` (producer stores), `0009` (consumer fences, PR `#3537`), `0010` (shared alignment, PR `#3690`), `0013` (multi-warp TMA producer rendezvous).
  - **1 patch dropped (accepted upstream)**: `0014-kernel-cache-device-compiler.patch` superseded by upstream `#4511` (`423b4d81`).
  - **4 patches rebased onto `v0.1.16`**:
    - `0003-jit-prepare-and-frozen-kernel-cache.patch` (`_refuse_if_frozen` adapted to `#3807`'s `KernelCache.lookup()` split; `prepare()` adapted to `#3903` offsets).
    - `0007-cache-bypass.patch` (PR `#3710`, adapted to `v0.1.16` `tilelang/jit/__init__.py`).
    - `0011-explicit-preparation-with-disabled-cache.patch` (adapted to `v0.1.16` `tilelang/jit/__init__.py`; 22/22 CPU controls pass).
    - `0013-compiler-snapshot-version.patch` (`VERSION` set to `0.1.16+snapshot.571c406f6a0473b658533a84e815c78310e6acf7`, covering all 12 functional patches).
- **vLLM compatibility patches (`experiments/2026-10-26-spark-ds41f-candidate/candidate-77/vllm/series`, 744 patches = `candidate-74` `0001..0869` + `0870..0871`)**:
  - `0870-tilelang-prepare-use-v0116-write-cache-path.patch`: update `vllm/models/deepseek_v4_1/tilelang/prepare.py` (`_prepare_with_warm_index`) to use TileLang `v0.1.16`'s `_get_write_cache_path` (renamed from `_get_cache_path` in `#3807`) and catch load exceptions cleanly when falling back to `tilelang.prepare()`.
  - `0871-requirements-cuda-bump-tilelang-to-0.1.16.patch`: bump `requirements/cuda.txt` from `tilelang==0.1.15` to `tilelang==0.1.16` so image dependency checks accept `0.1.16+snapshot.*`.
- **Runtime configuration**: identical to `releases/spark-ds41f-1126/cluster.json` across all 16 shapes (`DeepSeek-V4.1-Flash-1017`, `DSpark`, `GLP-27` at `alpha=2.0`, `sparknet v0.6.0`, `tilelang`).

## Build and verification

- **Image build (`dgx4`, commit `02fbfe76fd`)**:
  - Tag: **`spark-ds41f:candidate-77`**
  - OCI digest: **`sha256:6cad359a15cbc678407f7def7f2195dbefd238c61c5acbf64b28ccb265b83fdc`** (`454 s`, `lowest MemAvailable 82.1 GiB`, identical digest verified across `dgx1..dgx4`).
  - Locked trees: `VLLM_TREE = 58f9b11080cf8ebeccd80c0ce1d9bb7a6ff4ae58` (`patch_head = e0b964f8e646`), `TILELANG_TREE = 16c9ce75cead3b893572b9432f2c39b662391de8` (`patch_head = 2ed9f9cf5f90`, `version = 0.1.16+snapshot.571c406f6a0473b658533a84e815c78310e6acf7`), `SPARKNET_TREE = 1739407706277142da7a12c95fca3c503c8cb11c`.
  - Build smokes: CUDA 13.5, TileLang `0.1.16+snapshot.571c406f...` RECORD/dependency check, SparkNet ABI 10, GB10 GPU import smoke, and 70 DS4.1 modules without B12X all passed.
- **TileLang `v0.1.16` retained-feature GPU test suite (`dgx4`, `29.75 s`)**: **PASS (`212 passed, 10 skipped, 0 failed`)** across 12 files (`test_tilelang_prepared_disabled_cache.py`, `test_tilelang_without_kernel_cache.py`, `test_tilelang_kernel_cache_integrity.py`, `test_tilelang_cuda_binary_cache.py`, `test_tilelang_language_mxf8f6f4_mma_block_scale.py`, `test_tilelang_language_mxfp4_mma_block_scale.py`, `test_tilelang_tools_cuda_iket.py`, `test_tilelang_transform_Inject_software_pipeline.py`, `test_tilelang_issue_ws_simt_copy_full_producer_extent.py`, `test_tilelang_issue_3529.py`, `test_cuda_shared_memory_alias_codegen.py`, `test_tilelang_transform_smem_swizzle_alignment.py`), exercising the isolated `0.1.16_snapshot_571c406f...` cache namespace.
- **Targeted vLLM DeepSeek-V4.1 TileLang GPU tests (`dgx4`, `444.63 s`)**: **PASS (`946 passed, 0 skipped, 0 failed`)** across `test_deepseek_v4_1_prepare.py`, `test_deepseek_v41_tilelang_iket.py`, `test_deepseek_v41_tilelang_glp.py`, `test_deepseek_v41_tilelang_mhc.py`, `test_prepare_kernels.py`, and `test_tilelang_rejection.py`.
- **Cluster state**: `spark-ds41f:candidate-77` staged across `dgx1..dgx4`; `config/cluster.json` remains live on `spark-ds41f:1126`. Full `174`-file `release-tests` gate and clean-reboot `tp3-triangle` screen pending authorization.
