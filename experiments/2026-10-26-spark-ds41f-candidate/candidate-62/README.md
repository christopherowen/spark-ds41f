# `candidate-62` — Python-Only Fixes & Host Tooling Rollup (Benchmarks Carried Over from `candidate-61` / `candidate-61-r2`)

## Summary

`candidate-62` layers five pure-Python vLLM fixes (`0823..0827`, zero kernel/CUDA/TileLang/graph/loader/collective changes, `jit_new_files = 0`) onto `candidate-61-r2` (`experiments/2026-11-16-c61-decode/`, vLLM `0001..0822`) alongside the merged host-Python PRs (`#1224`, `#1049`, `#1232`, `#958`).

- **vLLM base revision**: `19f2c20ed4d64ae22bd009ceacadd85e2a81e354`
- **vLLM patch series**: [`vllm/series`](vllm/series) (`700` active patches, ending at `0827`)
- **vLLM patch head**: `8538c3a0a410507129e903de1c60ff69f363642d`
- **vLLM expected tree**: `e61742443dbd6a458e1edb3ce13608770d5ee587`
- **Patchset SHA-256**: `db151286acb510242ecb5814b26a15016a668b8fc15fd9a4a59e14deff7b3748`
- **TileLang / TileKernels / SparkNet / NCCL / Model weights / Serve options**: Identical to `experiments/2026-11-16-c61-decode/` (`upstreams.lock.json` differs only in `vllm`).

## Included Host-Python & Tooling PRs

1. **PR `#1224`**: Local/two-node topology and bench/lab harness updates (`cf1ab00a0`).
2. **PR `#1049`**: Cluster sync and shape check documentation clarifications (`243f5bc41`).
3. **PR `#1232`**: `doctor`: free memory by block size and `--stream-rate` per idle node (`23b24a022`).
4. **PR `#958`**: Verify TP3 6-channel bidirectional Ring NCCL profile and align topology checks (`d99dd9514`).

## Included vLLM Python Patches (`0823..0827`)

| Patch | Issue / PR | Scope | Memory & Kernel Impact | Gate Test Added |
|---|---|---|---|---|
| [`0823`](vllm/0823-fix-dsv41-size-the-structured-output-bitmask-for-cop.patch) | `#1194` / PR `#1227` (`@Zee-B33`) | `StructuredOutputManager.grammar_bitmask` sizes buffer rows by `max(num_speculative_tokens, num_verification_tokens)` (`96 -> 272` rows at 16 seats). | Lazy CPU/pinned `int32` buffer growth (`+2.71 MiB`) allocated **only** when a grammar-constrained request runs; `0 B` at startup, CUDA graph capture, and unconstrained serving. Zero kernel/weight changes. | `tests/v1/structured_output/test_copy_draft_bitmask_rows.py` |
| [`0824`](vllm/0824-fix-dsv41-keep-DSML-closers-inside-tool-arguments-un.patch) | `#876` / PR `#1228` (`@jmeadlock`) | `vllm/parser/deepseek_v41.py` keeps genuine-token (`128825`) DSML closers inside `TOOL_ARGS` parameter values unless structural DSML tokens follow (`VLLM_DSV41_STRUCTURAL_DSML=1`). | Tool parser state machine only (`TOOL_ARGS`); zero kernel, weight, graph, or decode/prefill execution changes. | `tests/parser/test_deepseek_v41_structural_closers.py` |
| [`0825`](vllm/0825-fix-serve-bind-SSE-keep-alive-done-callback-per-wait.patch) | `#1210` | `vllm/entrypoints/serve/utils/sse_keep_alive.py` binds per-wait done callbacks (`watched_task`, `waiter`), guards `while not next_task.done():`, and removes callbacks in `finally:` so stale queued callbacks cannot resolve a later chunk's waiter (`InvalidStateError`). | Async SSE transport wrapper only; zero engine/worker/kernel changes. | `tests/entrypoints/serve/utils/test_sse_keep_alive_race.py` |
| [`0826`](vllm/0826-feat-responses-support-CustomTool-and-OpenResponses-.patch) | `#1206` | Non-Harmony Responses API support for `CustomTool` (`"type": "custom"`), `custom_tool_call` / `custom_tool_call_output` history replay, custom-tool streaming events, tool schema/xgrammar helpers, and OpenResponses input normalization (grammar-format custom tools rejected with `400`). | Responses API frontend & schema helpers only; zero chat-completion, kernel, weight, graph, or collective changes. | `tests/entrypoints/openai/test_responses_custom_tools.py` |
| [`0827`](vllm/0827-fix-dsv41-rename-quantization-ignored-layers-into-th.patch) | PR `#1235` (owner request, 2026-11-19) | The V4.1 VL wrapper (`vllm/models/deepseek_v41/nvidia/vl_model.py`, inherited by the GB10 wrapper) gets a class-level `hf_to_vllm_mapper` with its module renames, so `configure_quant_config` renames a quantization config's `ignored_layers` (`layers.1.engram.wkv` -> `language_model.model.layers.1.engram.wkv`) before any layer is built. | Inert on the served checkpoints: `DeepSeek-V4.1-Flash` and `-Flash-1017` list no ignored layers and store both Engram projections as FP8, so every layer keeps its method. Zero kernel, weight, graph, or collective changes. | `tests/models/test_deepseek_v4_1_quant_ignore.py` |

## Benchmark Carry-Over Contract (`rc12 -> rc13 -> rc14` Precedent)

Per the `rc12 -> rc13 -> rc14` carry-over precedent recorded in [`manifests/baselines/2026-10-30-spark-ds41f-1030.json`](../../../manifests/baselines/2026-10-30-spark-ds41f-1030.json), because `candidate-62` preserves `candidate-61-r2`'s exact:
- TileLang/TileKernels/NCCL/SparkNet bits (`jit_new_files = 0`),
- Model weights, packing, KV budgets, and CUDA graph capture shapes,
- Collective policies and speculative decoding parameters, and
- Unconstrained `/v1/chat/completions` greedy/sampled serving hot paths,

all hardware benchmark suites recorded on `candidate-61` and `candidate-61-r2` (`experiments/2026-11-16-c61-decode/`) stand directly for `candidate-62` across all 12 shape profiles (`tp1-local`, `tp2-direct`, `tp2-switched`, `tp3-triangle`, `tp3-triangle-4k`, `tp3-switched`, `tp4-ring`, `tp4-switched`, `tp5-switched`, `tp6-switched`, `tp7-switched`, `tp8-switched`). Every `candidate-62/<shape>.json` resolves via `tests/test_releases.py:serving()` to the exact same runtime configuration as its parent profile in `experiments/2026-11-16-c61-decode/` apart from container image tag (`spark-ds41f:candidate-62`), `local.spark-ds41f.vllm.tree`, and `upstreams_config`.

Before promoting or shipping an image built from `candidate-62`, execute:
- [`bundles/release-tests`](bundles/release-tests) (in-image gate suite) and [`build.json`](build.json) / [`fix-tp4.json`](fix-tp4.json) (two-boot determinism + startup memory verification).
