# Phase B Plan — Rebase onto Canonical `vllm-project/vllm` `main` + `spark_ds41` Package Extraction

## Snapshot audited

- **Canonical upstream**: `https://github.com/vllm-project/vllm.git` `main` at **`48e1793769aba1b14035a15d8e840038dd757949`** (`2026-11-27`).
- **Our Phase A baseline**: `experiments/2026-11-28-vllm-patch-squash/` (`20` flat subsystem patches, `544` files, `VLLM_TREE = 58f9b11080cf8ebeccd80c0ce1d9bb7a6ff4ae58`, tree-identical to qualified `candidate-77`).

## What canonical `vllm-project/vllm` `main` already provides (vs our `sgl-project/vllm` fork)

| Area | Status on canonical `vllm` `main` (`48e1793`) | Implication for rebase |
|---|---|---|
| Retired `b12x_*` files (`Patch 01`, 10 files, `-5,782` lines) | **Absent** | **Drops 100%** |
| `vllm/models/deepseek_v41/` | **Present**, cleanly isolated by hardware (`__init__.py` dispatches `amd/` vs `nvidia/`, plus `common/engram.py`, `attention.py`, `sparse_mla.py`, `compressor.py`, `quant_config.py`, `decoder_replay_layers.py`) | Our GB10 TileLang backend plugs in cleanly at `deepseek_v41/__init__.py` without modifying `amd/` or `nvidia/` |
| `vllm/v1/worker/gpu/spec_decode/` | **Present**: `dspark/` (`config.py`, `speculator.py`, `utils.py`), `dflash/`, `dflash2/`, `dfly/`, `adaptive_verification.py`, `acceptance_estimator.py` | Major shrinkage of `Patch 10`; only TileLang rejection, copy-drafts, and greedy verification hooks remain |
| DS4.1 parsers, tool parsers, reasoning parsers, tokenizers | **Present**: `vllm/parser/deepseek_v41.py`, `vllm/reasoning/deepseek_v41_engine_reasoning_parser.py`, `vllm/tool_parsers/deepseekv41_engine_tool_parser.py`, `vllm/tokenizers/deepseek_v41*.py`, `vllm/transformers_utils/configs/deepseek_v41.py` | Major shrinkage of `Patch 07` (`54` files) & `Patch 19` (`31` test files; `11` already byte-identical!) |
| Thinking budget (`vllm/v1/worker/gpu/sample/thinking_budget.py`) & Engram config (`vllm/config/engram.py`, `vllm/distributed/engram_pool/`) | **Present** | Shrinks `Patch 02`, `Patch 06`, `Patch 09` |
| TileLang in-tree (`vllm/model_executor/kernels/mhc/tilelang*.py`) | **Present** (30 references; pinned to `tilelang==0.1.12`) | Upstream already recognizes TileLang; bump `requirements/cuda.txt` pin to `0.1.16` |
| **Missing on canonical `main` (100% Spark-owned)** | **Absent**: SM120/SM121/GB10 TileLang DS4.1 backend (`45` files), `direct_io`/`direct_loader`/`weightless` (`11` files), `sparknet` RoCE collectives, `display_carveout` + order-13 contiguous CUDA pool, `engram_rows` disk `io_uring` (`6` files), `expert_slots` (`5` files), `glp` steering (`glp.py` + kernels), TileLang sampling/rejection/inputs (`19` files), `sp_prefill` & `tilelang_vision` | **`241` Spark-owned new files** (`~128k` lines, `~84%` of our net code/tests) have **zero upstream file conflicts** |

## Runtime dependency delta to account for

Canonical `vllm` `main` (`48e1793`) pins newer core packages in `requirements/cuda.txt` and `requirements/common.txt`:
- `torch==2.14.0`, `torchaudio==2.11.1`, `torchvision==0.29.2`
- `transformers >= 5.16.1, < 5.25.0`
- `flashinfer-python==0.7.4`, `xgrammar==0.2.9`, `numba==0.65.0`, `tilelang==0.1.12` (override to `0.1.16+snapshot`)
- `model_runner.py` (`2,683` lines vs our `5,326`) and `gpu_worker.py` (`1,979` lines vs our `2,840`) were refactored upstream.

Because this is a major PyTorch + vLLM runner upgrade, it must be executed and qualified as its own dedicated candidate track (not combined with patch squashing).

## Recommended two-stage execution plan

### Stage B.1 — Extract Spark-owned code into a standalone `spark_ds41` wheel package (can be done BEFORE or DURING canonical rebase)

Move all `~241` Spark-owned files out of `.patch` diffs into a real versioned Python/C extension package (`spark_ds41`, installed via wheel in the Docker image, with normal Git history, PRs, and `pytest`):
- `spark_ds41/tilelang/` (`45` DS4.1 TileLang kernel modules, `+36,161` lines)
- `spark_ds41/direct_io/` (`O_DIRECT`/`io_uring` C+Python checkpoint loader, `weight_digest`, `weightless`)
- `spark_ds41/engram_rows/` (`io_uring` disk Engram table reader)
- `spark_ds41/expert_slots/` (expert cache/hotlist C+Python module)
- `spark_ds41/glp/` (GLP steering)
- `spark_ds41/sampling/` (`tilelang_sampling`, `tilelang_rejection`, `tilelang_resampled`, `tilelang_copy`, `tilelang_inputs`, `tilelang_structured`, `tilelang_token_counts`, `sink`, `copy_drafts`, `greedy_verification`)
- `spark_ds41/worker/` (`display_carveout`, `startup_rendezvous`, `late_admission`)
- `spark_ds41/model/` (GB10 TileLang `DeepseekV41ForCausalLM`, `DSparkDeepseekV4ForCausalLM`, `tilelang_vision`, `sp_prefill`, `ced`)
- `spark_ds41/tests/` & `spark_ds41/benchmarks/`

### Stage B.2 — Thin ~9-patch integration stack on canonical `vllm-project/vllm` `main`

With `spark_ds41` installed as a package, our vLLM patch series against canonical `vllm` `main` shrinks from `744` patches (`~234k` gross lines) -> `20` patches (Phase A, `~187k` net lines) -> **`~9` thin seam patches (`~5–10k` net lines)**:

1. `0001-requirements-and-spark-config.patch`: `requirements/cuda.txt` (`tilelang==0.1.16`), Spark/GLP/direct-loader/O13 env & config knobs.
2. `0002-loader-register-direct-and-weightless.patch`: hook `spark_ds41.direct_io` into vLLM's model loader registry.
3. `0003-distributed-sparknet-and-rendezvous.patch`: hook `sparknet` one-shot collectives and `startup_rendezvous` in `vllm/distributed/` and `multiproc_executor.py`.
4. `0004-worker-o13-pool-and-carveout.patch`: order-13 contiguous CUDA pool & `display_carveout` in `vllm/v1/worker/gpu_worker.py`.
5. `0005-scheduler-prefill-yield-and-pp2.patch`: `2,048`-grid chunk step-down, prefill stall interleaving, and PP2 microbatch balancing in `vllm/v1/core/sched/`.
6. `0006-deepseek-v41-spark-tilelang-dispatch.patch`: 3-line GB10 TileLang dispatch in `vllm/models/deepseek_v41/__init__.py` + TileLang attention/MoE backend registration.
7. `0007-gpu-runner-cudagraph-and-tilelang-sampling.patch`: FULL CUDA graph replay & `spark_ds41.sampling` hooks in `vllm/v1/worker/gpu/model_runner.py`.
8. `0008-dspark-tilelang-rejection-and-copy-drafts.patch`: TileLang rejection & copy-draft hooks in `vllm/v1/worker/gpu/spec_decode/dspark/`.
9. `0009-entrypoints-glp-and-ds41-serving.patch`: GLP steered second served-model alias + any DS4.1 DSML/SSE fixes not yet covered by canonical `vllm/parser/deepseek_v41.py`.

### Stage B.3 — Build, test, and qualify

1. Lock `torch==2.14.0` + updated `requirements/` hashes for ARM64/CUDA 13.5 on `dgx4`.
2. Build image on `dgx4`, run TileLang GPU tests + `spark_ds41` tests + full `release-tests` gate.
3. Qualify on `dgx1..dgx3` with clean-reboot 2-boot `tp3-triangle` screen (`screen-tp3`, `single-stream`, `sampled`) + `--wide` determinism against the latest promoted baseline before promotion.
