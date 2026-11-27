# `2026-11-28-vllm-patch-squash` — Tree-identical vLLM patch squash (`744` -> `20`)

## Base and variable

- **Base deployment commit**: `f5f3be34d8` (`candidate-77`, qualified TileLang `v0.1.16` rebase on `spark-ds41f-1126`).
- **Single structural variable**: consolidate the `744` chronological vLLM patches chained across `60` directories in `candidate-77` (`0001..0871`) into **`20` flat, file-disjoint subsystem patches** in [`vllm/series`](vllm/series) on the same upstream base (`sgl-project/vllm` `19f2c20ed4d64ae22bd009ceacadd85e2a81e354`).
- **Bit-for-bit tree identity**:
  - Original `744`-patch `VLLM_TREE`: **`58f9b11080cf8ebeccd80c0ce1d9bb7a6ff4ae58`** (`patch_head = e0b964f8e6463e4a64ca91a2eaba5f0649739f12`)
  - Squashed `20`-patch `VLLM_TREE`: **`58f9b11080cf8ebeccd80c0ce1d9bb7a6ff4ae58`** (`patch_head = 4e49492098a3c789b042d43b874710793fdb6853`, `patchset_sha256 = f52012e8545086a2bcc289cc0e304a0e1c1c5c202c9b45c25a45d9ecc4cde381`)
  - Because `HEAD^{tree}` is bit-for-bit identical, the built container source tree, runtime code, and TileLang/CUDA kernels are completely unchanged from qualified `candidate-77` (`spark-ds41f:candidate-77`, `sha256:6cad359a15cb...`).
- **Churn removed**: eliminates `~47,000+` lines of add-then-modify/revert/delete self-churn (including created-and-deleted `benchmarks/kernels/benchmark_tilelang_vision.py`, explicit revert `0483`, and multi-round rewrites of the same files across `60` chained series directories).

## Subsystem patch breakdown (`20` file-disjoint patches, `544` files)

| # | Patch | Files | Net diff | Scope |
|---|---|---:|---:|---|
| 01 | `0001-remove-retired-b12x-fork-files.patch` | 10 | `+0 / -5,782` | Retire unused `b12x_*` files from `sgl-project/vllm` base |
| 02 | `0002-config-envs-and-requirements.patch` | 23 | `+1,297 / -127` | `requirements/cuda.txt` (`tilelang==0.1.16`), `envs.py`, `config/`, `utils/`, `sampling_params.py`, `docs/` |
| 03 | `0003-direct-loader-and-weightless-loading.patch` | 17 | `+3,128 / -35` | `direct_loader`, `io_uring` packed loader, `weight_transfer.py`, `weightless.py` |
| 04 | `0004-distributed-sparknet-and-startup-rendezvous.patch` | 8 | `+653 / -47` | SparkNet RoCE/SHM collectives, `startup_rendezvous.py`, multiproc executor |
| 05 | `0005-display-carveout-and-o13-worker-memory.patch` | 6 | `+2,318 / -88` | `display_carveout.py`, order-13 contiguous CUDA pool (`gpu_worker.py`), warmup |
| 06 | `0006-v1-scheduler-kv-cache-and-engine-core.patch` | 26 | `+3,391 / -201` | v1 scheduler prefill yielding, KV cache manager, engine core/client/detokenizer, structured output |
| 07 | `0007-entrypoints-parsers-renderers-and-tokenizers.patch` | 54 | `+4,569 / -402` | OpenAI entrypoints, DeepSeek reasoning/tool parsers, renderers, tokenizers, multimodal |
| 08 | `0008-model-executor-layers-attention-and-sampling.patch` | 26 | `+2,408 / -309` | Quantization layers, TileLang attention registration, sampling ops, sink |
| 09 | `0009-v1-gpu-model-runner-and-cudagraphs.patch` | 42 | `+20,713 / -2,356` | v1 GPU model runner, FULL/piecewise CUDA graphs, input batching, TileLang dispatch |
| 10 | `0010-dspark-and-speculative-decoding.patch` | 19 | `+7,233 / -961` | DSpark/DFlash drafting, adaptive verification, acceptance estimation, TileLang rejection |
| 11 | `0011-deepseek-v41-model-engram-vision-and-registry.patch` | 39 | `+16,486 / -3,150` | DeepSeek-V4.1 model, Engram, vision tower, model registry |
| 12 | `0012-tilelang-dsv41-core-prepare-iket-and-collectives.patch` | 15 | `+12,314 / -0` | TileLang DS4.1 runtime, warm-index `prepare.py`, `iket.py`, `pdl.py`, `collectives.py`, quant/norm/vocab |
| 13 | `0013-tilelang-dsv41-gemm-linear-wo-and-moe.patch` | 7 | `+9,992 / -0` | TileLang DS4.1 `gemm.py`, `linear.py`, `wo.py`, `moe.py`, `router.py`, `ced.py`, `confidence.py` |
| 14 | `0014-tilelang-dsv41-attention-indexer-and-shortlist.patch` | 17 | `+9,856 / -0` | TileLang DS4.1 Sparse MLA, indexer/shortlist/topk, RoPE, KV cache writer, vision, DSpark context |
| 15 | `0015-tilelang-dsv41-mhc-engram-and-glp.patch` | 6 | `+3,999 / -0` | TileLang DS4.1 `mhc.py`, `engram*.py`, `glp*.py` |
| 16 | `0016-benchmarks.patch` | 28 | `+7,988 / -0` | DS4.1 kernel and serving microbenchmarks |
| 17 | `0017-tests-tilelang-and-kernels.patch` | 41 | `+20,247 / -4` | TileLang DS4.1 kernel and quantization tests |
| 18 | `0018-tests-v1-engine-runner-and-spec-decode.patch` | 85 | `+25,505 / -529` | v1 engine, scheduler, runner, and DSpark spec-decode tests |
| 19 | `0019-tests-entrypoints-parsers-and-tokenizers.patch` | 31 | `+5,625 / -38` | Entrypoints, reasoning/tool parsers, and tokenizer tests |
| 20 | `0020-tests-models-loader-and-core.patch` | 44 | `+13,217 / -2,049` | DS4.1 model, direct loader, distributed, and core utility tests |

## Verification

- Fresh `git am` application via `spark_ds41f.build.apply_patch_series` onto `19f2c20ed4d64ae22bd009ceacadd85e2a81e354` on `dgx4` produced exact `HEAD^{tree} = 58f9b11080cf8ebeccd80c0ce1d9bb7a6ff4ae58` (matching `candidate-77`).
- `bin/spark candidate lint experiments/2026-11-28-vllm-patch-squash/upstreams.lock.json`: clean.
- `bin/spark --cluster-config experiments/2026-11-28-vllm-patch-squash/tp3.json build check` and `doctor` verified on `dgx4`.
- Runtime qualification inherited identically from `candidate-77` (identical `VLLM_TREE`, `TILELANG_TREE`, `SPARKNET_TREE`, and OCI image digest `sha256:6cad359a15cb...`).
