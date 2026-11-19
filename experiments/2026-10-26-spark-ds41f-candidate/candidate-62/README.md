# `candidate-62` — Python-Only Fixes & Host Tooling Rollup (Benchmarks Carried Over from `candidate-61` / `candidate-61-r2`)

## Summary

`candidate-62` layers four pure-Python vLLM fixes (`0823..0826`, zero kernel/CUDA/TileLang/graph/loader/collective changes, `jit_new_files = 0`) onto `candidate-61-r2` (`experiments/2026-11-16-c61-decode/`, vLLM `0001..0822`) alongside the merged host-Python PRs (`#1224`, `#1049`, `#1232`, `#958`).

- **vLLM base revision**: `19f2c20ed4d64ae22bd009ceacadd85e2a81e354`
- **vLLM patch series**: [`vllm/series`](vllm/series) (`699` active patches, ending at `0826`)
- **vLLM patch head**: `2a758e3c08b031ad0b09d3625df5acb4515e0a92`
- **vLLM expected tree**: `6aec3edca6da865e7e095ab0b87eafff552310a3`
- **Patchset SHA-256**: `0deda9cdf21001992e6a3bd88c4e1e36609af5708792292969e8cb5491a74795`
- **TileLang / TileKernels / SparkNet / NCCL / Model weights**: Identical to `experiments/2026-11-16-c61-decode/` (`upstreams.lock.json` differs only in `vllm`).
- **32-sequence concurrency on `TP4+` (`tp4`..`tp8-switched`)**: Adopts `--max-num-seqs 32` (`--max-cudagraph-capture-size 192`) across all `TP4..TP8` profiles (`TP1..TP3` unchanged at `4`/`8` seats), backed by `#1230` (`+37%` aggregate decode scaling at `c32`, `+19%` mixed-load tasks/h), single-slot 2 MiB RoCE compatibility (`192 × 10,240 B = 1.875 MiB <= 2.0 MiB`), patch `0823` (`544`-row structured-output bitmask), and `0783`/`0784` (`128`-row decode tiles and `_FULL_DEPTH_VARLEN_DECODE_MAX_REQS = 16`).

## `TP4..TP8` 32-Sequence Configuration (`max_num_seqs = 32`, `max_cudagraph_capture_size = 192`)

| Shape | `--max-num-seqs` | `max_cg` | `--max-num-batched-tokens` | `--long-prefill-token-threshold` | `--kv-cache-memory-bytes` | Full `1M` Contexts | 32-Seq Simultaneous Context |
|---|---:|---:|---:|---:|---:|---:|---:|
| `tp4` (`tp4-ring`) | `32` | `192` | `16,384` | `16,192` | `17,716,740,096` (`16.50 GiB`) | `17.86x` | `579,840` tokens |
| `tp4-switched` | `32` | `192` | `16,384` | `16,192` | `17,716,740,096` (`16.50 GiB`) | `17.86x` | `579,840` tokens |
| `tp5-switched` | `32` | `192` | `32,768` | `32,576` | `25,769,803,776` (`24.00 GiB`) | `25.46x` | `831,744` tokens |
| `tp6-switched` | `32` | `192` | `24,576` | `24,384` | `13,958,643,712` (`13.00 GiB`) | `13.41x` | `432,384` tokens |
| `tp7-switched` | `32` | `192` | `32,768` | `32,576` | `17,179,869,184` (`16.00 GiB`) | `16.36x` | `530,176` tokens |
| `tp8-switched` | `32` | `192` | `32,768` | `32,576` | `27,380,416,512` (`25.50 GiB`) | `27.17x` | `888,320` tokens |

`tp6-switched` (`14.00 -> 13.00 GiB`, `+23.8%` over `1107`'s `10.50 GiB`) and `tp8-switched` (`26.00 -> 25.50 GiB`, `+142.9%` over `1107`'s `10.50 GiB`) reserve extra host headroom for the `192`-row CUDA graph pool (`~0.35 GiB/rank`) to maintain the `>= 5.5 GiB` startup `MemAvailable` floor on the head node (`sp1`, in-memory Engram). All `TP4..TP8` shapes keep the 2-to-192 CUDA graph capture ladder (`capture_sizes(192)`).

## Included Host-Python & Tooling PRs

1. **PR `#1224`**: Local/two-node topology and bench/lab harness updates (`cf1ab00a0`).
2. **PR `#1049`**: Cluster sync and shape check documentation clarifications (`243f5bc41`).
3. **PR `#1232`**: `doctor`: free memory by block size and `--stream-rate` per idle node (`23b24a022`).
4. **PR `#958`**: Verify TP3 6-channel bidirectional Ring NCCL profile and align topology checks (`d99dd9514`).

## Included vLLM Python Patches (`0823..0826`)

| Patch | Issue / PR | Scope | Memory & Kernel Impact | Gate Test Added |
|---|---|---|---|---|
| [`0823`](vllm/0823-fix-dsv41-size-the-structured-output-bitmask-for-cop.patch) | `#1194` / PR `#1227` (`@Zee-B33`) | `StructuredOutputManager.grammar_bitmask` sizes buffer rows by `max(num_speculative_tokens, num_verification_tokens)` (`96 -> 272` rows at 16 seats). | Lazy CPU/pinned `int32` buffer growth (`+2.71 MiB`) allocated **only** when a grammar-constrained request runs; `0 B` at startup, CUDA graph capture, and unconstrained serving. Zero kernel/weight changes. | `tests/v1/structured_output/test_copy_draft_bitmask_rows.py` |
| [`0824`](vllm/0824-fix-dsv41-keep-DSML-closers-inside-tool-arguments-un.patch) | `#876` / PR `#1228` (`@jmeadlock`) | `vllm/parser/deepseek_v41.py` keeps genuine-token (`128825`) DSML closers inside `TOOL_ARGS` parameter values unless structural DSML tokens follow (`VLLM_DSV41_STRUCTURAL_DSML=1`). | Tool parser state machine only (`TOOL_ARGS`); zero kernel, weight, graph, or decode/prefill execution changes. | `tests/parser/test_deepseek_v41_structural_closers.py` |
| [`0825`](vllm/0825-fix-serve-bind-SSE-keep-alive-done-callback-per-wait.patch) | `#1210` | `vllm/entrypoints/serve/utils/sse_keep_alive.py` binds per-wait done callbacks (`watched_task`, `waiter`), guards `while not next_task.done():`, and removes callbacks in `finally:` so stale queued callbacks cannot resolve a later chunk's waiter (`InvalidStateError`). | Async SSE transport wrapper only; zero engine/worker/kernel changes. | `tests/entrypoints/serve/utils/test_sse_keep_alive_race.py` |
| [`0826`](vllm/0826-feat-responses-support-CustomTool-and-OpenResponses-.patch) | `#1206` | Non-Harmony Responses API support for `CustomTool` (`"type": "custom"`), `custom_tool_call` / `custom_tool_call_output` history replay, custom-tool streaming events, tool schema/xgrammar helpers, and OpenResponses input normalization (grammar-format custom tools rejected with `400`). | Responses API frontend & schema helpers only; zero chat-completion, kernel, weight, graph, or collective changes. | `tests/entrypoints/openai/test_responses_custom_tools.py` |

## Benchmark Carry-Over & Qualification Scope

Per the `rc12 -> rc13 -> rc14` carry-over precedent recorded in [`manifests/baselines/2026-10-30-spark-ds41f-1030.json`](../../../manifests/baselines/2026-10-30-spark-ds41f-1030.json), because `candidate-62` preserves `candidate-61-r2`'s exact TileLang/TileKernels/NCCL/SparkNet bits (`jit_new_files = 0`), weights, and unconstrained `/v1/chat/completions` hot paths:

- **`TP1–TP3` (`tp1`, `tp2-direct`, `tp2-switched`, `tp3`, `tp3-switched`, `tp3-triangle-4k`)**: Carry over `candidate-61` and `candidate-61-r2` hardware benchmarks directly (`serving()` configuration is unchanged aside from container image tag, `local.spark-ds41f.vllm.tree`, and `upstreams_config`).
- **`TP4–TP8` (`tp4`, `tp4-switched`, `tp5-switched`, `tp6-switched`, `tp7-switched`, `tp8-switched`)**: Raise `--max-num-seqs` to `32`, `--max-cudagraph-capture-size` to `192`, and shift `--long-prefill-token-threshold` by `-96` (`-96` tokens on multi-chunk prefill boundaries), plus KV headroom adjustments on `tp6-switched` (`13.00 GiB`) and `tp8-switched` (`25.50 GiB`). Once active `candidate-61-r2` waves conclude, record 2-boot 32-seat startup-memory, determinism, `release`/`bench`, and `--suite c32` ([`config/suites/c32.json`](../../../config/suites/c32.json)) receipts across `TP4..TP8`.

Before promoting or shipping an image built from `candidate-62`, execute:
- [`bundles/release-tests`](bundles/release-tests) (in-image gate suite) and [`build.json`](build.json) / [`fix-tp4.json`](fix-tp4.json) (two-boot determinism + startup memory verification).
