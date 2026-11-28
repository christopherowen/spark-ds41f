# `2026-11-28-canonical-vllm-rebase` — Canonical `vllm-project/vllm` + `spark_ds41` Package

## Goals and architecture

Replace the 744-patch `sgl-project/vllm` fork stack with:
1. **Standalone `spark_ds41` package** ([`packages/spark_ds41/`](../../packages/spark_ds41/)): all Spark-owned runtime code (`110` files: `46` TileLang DS4.1 kernel modules, `direct_io`/`direct_loader`/`weightless`, `engram_rows`, `expert_slots`, `glp`, GB10 DS4.1 model backend, TileLang sampling/rejection/inputs, `display_carveout`, `startup_rendezvous`, `mem_utils`) maintained as ordinary Git history and installed as a Python wheel with a `vllm.general_plugins` entry point (`DirectModelLoader`).
2. **Official ARM64 CUDA 13.5 canonical vLLM nightly base**:
   - `docker.io/vllm/vllm-openai:cu135-nightly-de7cf7f8cb5dc47d2e65207e0fabbe2c8f0d73b3` (`sha256:9e55ab450cb96488964eff0dbfbe366675e5a93599e54950e408f3209e13005d`)
   - Native **CUDA 13.5.37**, **Python 3.14.10**, **`torch 2.16.0.dev20261116+cu135`**, `transformers 5.24.1`, `nvidia-cutlass-dsl 4.8.0`, `quack-kernels 0.6.5`, `flashinfer-python 0.7.4`, `xgrammar 0.2.9`, `NCCL 2.32.3` (`23203`).
3. **Thin upstream seam patches**:
   - [`nccl/series`](nccl/series): `4` patches cleanly rebased onto **`NVIDIA/nccl v2.32.3-1`** (`12df1a11afad322be5a204a2db890161cbf8131d`, `TREE = 675a2b79cd5e700c27c188f3290fb5926ff2e366`).
   - [`vllm/series`](vllm/series): thin integration seams on canonical **`vllm-project/vllm de7cf7f8cb5dc47d2e65207e0fabbe2c8f0d73b3`** (no `b12x` fork files, no upstream churn).
   - **TileLang `v0.1.16+snapshot`** (`candidate-77` 13-patch series), **`tile_kernels 2.0.0`**, and **`dgx-spark-networking 0.6.0`** built as Python 3.14 / CUDA 13.5 wheels.

## Verified progress (on `dgx4`, GB10 SM121)

- **Wheels on Python 3.14.10 / CUDA 13.5.37**: `tilelang-0.1.16+snapshot.571c406f...`, `tile_kernels-2.0.0`, `dgx_spark_networking-0.6.0`, and `spark_ds41-0.1.0` all build and install cleanly.
- **Native C/`io_uring`/RoCE extensions**: `sparknet` RoCE proxy (ABI 10), `spark_ds41.direct_io._native`, `spark_ds41.engram_rows._native`, and `spark_ds41.expert_slots._native` all compile and link `liburing.so` on Python 3.14.
- **Plugin & hardware dispatch**: `vllm.general_plugins` registers `DirectModelLoader` (`load_format="direct"`), `vllm.models.deepseek_v41` dispatches `DeepseekV41ForCausalLM` and `DSparkDeepseekV4ForCausalLM` to `spark_ds41.model.*` on GB10 `(12, 1)`, and `AttentionBackendEnum.TILELANG` points to `spark_ds41.tilelang.attention.DeepseekV41TileLangBackend`.
- **Live GB10 GPU tests on canonical `vllm 0.34.1` (`de7cf7f8cb5d`) + Python 3.14 + CUDA 13.5**:
  - TileLang `v0.1.16` retained-feature GPU suite: **`190 passed, 10 skipped, 0 GPU failures`** (`19.8 s`).
  - `test_deepseek_v41_tilelang_iket.py`: **`22 / 22 passed`** (including `JitWarmupRegistry.active()` subprocess tests).
  - `test_deepseek_v41_tilelang_mhc.py` & `test_deepseek_v41_tilelang_glp.py`: **100% passed**.
  - `test_tilelang_rejection.py` (TileLang block verification, dead-row cuts, greedy verification, replay): **`244 / 244` TileLang kernel tests passed**.

## Seam patch stack status (`vllm/series`)

- **Done (`0001..0005`, `54 KB` total)**:
  1. `0001-requirements-and-spark-config.patch`: `tilelang==0.1.16`, Spark env vars, `load.py` (`direct`), `CopyDraftsConfig`, `PrefillYield`.
  2. `0002-distributed-sparknet-and-all-to-all.patch`: `ring4_relay`, `all_to_all_rows`, `all_reduce_in_place`, parallel state hooks.
  3. `0003-worker-workspace-and-kv-spec.patch`: preallocated workspace, MoE output buffer, CUDA graph capture resources, `CircularBufferSpec`, `SlidingWindowMLASpec`, `uses_only_written_slots`, `dense_varlen_decode_shapes`.
  4. `0004-deepseek-v41-gb10-dispatch-and-tilelang-backend.patch`: GB10 dispatch in `vllm/models/deepseek_v41/__init__.py`, `AttentionBackendEnum.TILELANG`, `_is_packed_head` / `_is_plain_bf16_head`.
  5. `0005-jit-warmup-and-tilelang-block-rejection.patch`: `JitWarmupRegistry.active()` + TileLang block rejection & `insert_resampled` in `rejection_sampler_utils.py` / `rejection_sampler.py`.
- **In progress (`0006..0009`)**:
  6. `0006`: `SparkWorkerMixin` (`o13_pool`, `roce_guard`, `prepare`) + thin hooks in `gpu_worker.py` & `multiproc_executor.py`.
  7. `0007`: `SparkModelRunnerMixin` (`step_kernels`, `engine_grid`, `sharded_sampling`) + thin hooks in `model_runner.py`, `sample/sampler.py`, `dspark/speculator.py`, `dflash/speculator.py`.
  8. `0008`: `scheduler.py` (`prefill_yield`, `set_prefill_fairness`), `kv_cache_utils.py` (`weightless.BLOCK_HASH_KEY`), `kv_cache_manager.py` (`uses_only_written_slots`).
  9. `0009`: `entrypoints/serve/engine/serving.py` (GLP steered model alias), `sse_keep_alive.py`, and DS4.1 parser/tokenizer deltas.
