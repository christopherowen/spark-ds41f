# `candidate-62` — Python Fixes, Default-Off LSA Upgrade & `TP4+` 32-Seat Rollup (Benchmarks Carried Over from `candidate-61` / `candidate-61-r2`)

## Summary

`candidate-62` layers five pure-Python vLLM fixes (`0823..0827`) plus the strictly opt-in `VLLM_DS41_LSA=1` FP8 group-32 dominance upgrade (`0828`, PR `#954`) and a streaming fix to `0824`'s tool-argument parser (`0829`) onto `candidate-61-r2` (`experiments/2026-11-16-c61-decode/`, vLLM `0001..0822`) alongside the merged host-Python PRs (`#1224`, `#1049`, `#1232`, `#958`) and LSA evidence/watchlist PRs (`#976`, `#1014`, `#1017`). With `VLLM_DS41_LSA` unset (the default across every `candidate-62` shape), default serving keeps the 68-byte index cache, declares zero LSA kernels (`jit_new_files = 0`), and executes the exact `candidate-61-r2` hot paths.

- **vLLM base revision**: `19f2c20ed4d64ae22bd009ceacadd85e2a81e354`
- **vLLM patch series**: [`vllm/series`](vllm/series) (`702` active patches, ending at `0829`)
- **vLLM patch head**: `9b251a1860e7e9e123aaf718d15f499a2beb6738`
- **vLLM expected tree**: `8fc99ca7546c5c538f0fe64e7c82954f1d2d7ae0`
- **Patchset SHA-256**: `800e726dc8e772596e3e4cce20e264e92921510264cf628c7bdc4bd4fcb8d906`
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

## Included vLLM Patches (`0823..0829`)

| Patch | Issue / PR | Scope | Memory & Kernel Impact | Gate Test Added |
|---|---|---|---|---|
| [`0823`](vllm/0823-fix-dsv41-size-the-structured-output-bitmask-for-cop.patch) | `#1194` / PR `#1227` (`@Zee-B33`) | `StructuredOutputManager.grammar_bitmask` sizes buffer rows by `max(num_speculative_tokens, num_verification_tokens)` (`96 -> 272` rows at 16 seats). | Lazy CPU/pinned `int32` buffer growth (`+2.71 MiB`) allocated **only** when a grammar-constrained request runs; `0 B` at startup, CUDA graph capture, and unconstrained serving. Zero kernel/weight changes. | `tests/v1/structured_output/test_copy_draft_bitmask_rows.py` |
| [`0824`](vllm/0824-fix-dsv41-keep-DSML-closers-inside-tool-arguments-un.patch) | `#876` / PR `#1228` (`@jmeadlock`) | `vllm/parser/deepseek_v41.py` keeps genuine-token (`128825`) DSML closers inside `TOOL_ARGS` parameter values unless structural DSML tokens follow (`VLLM_DSV41_STRUCTURAL_DSML=1`). | Tool parser state machine only (`TOOL_ARGS`); zero kernel, weight, graph, or decode/prefill execution changes. | `tests/parser/test_deepseek_v41_structural_closers.py` |
| [`0825`](vllm/0825-fix-serve-bind-SSE-keep-alive-done-callback-per-wait.patch) | `#1210` | `vllm/entrypoints/serve/utils/sse_keep_alive.py` binds per-wait done callbacks (`watched_task`, `waiter`), guards `while not next_task.done():`, and removes callbacks in `finally:` so stale queued callbacks cannot resolve a later chunk's waiter (`InvalidStateError`). | Async SSE transport wrapper only; zero engine/worker/kernel changes. | `tests/entrypoints/serve/utils/test_sse_keep_alive_race.py` |
| [`0826`](vllm/0826-feat-responses-support-CustomTool-and-OpenResponses-.patch) | `#1206` | Non-Harmony Responses API support for `CustomTool` (`"type": "custom"`), `custom_tool_call` / `custom_tool_call_output` history replay, custom-tool streaming events, tool schema/xgrammar helpers, and OpenResponses input normalization (grammar-format custom tools rejected with `400`). | Responses API frontend & schema helpers only; zero chat-completion, kernel, weight, graph, or collective changes. | `tests/entrypoints/openai/test_responses_custom_tools.py` |
| [`0827`](vllm/0827-fix-dsv41-rename-quantization-ignored-layers-into-th.patch) | PR `#1235` (owner request, 2026-11-19) | The V4.1 VL wrapper (`vllm/models/deepseek_v41/nvidia/vl_model.py`, inherited by the GB10 wrapper) gets a class-level `hf_to_vllm_mapper` with its module renames, so `configure_quant_config` renames a quantization config's `ignored_layers` (`layers.1.engram.wkv` -> `language_model.model.layers.1.engram.wkv`) before any layer is built. | Inert on the served checkpoints: `DeepSeek-V4.1-Flash` and `-Flash-1017` list no ignored layers and store both Engram projections as FP8, so every layer keeps its method. Zero kernel, weight, graph, or collective changes. | `tests/models/test_deepseek_v4_1_quant_ignore.py` |
| [`0828`](vllm/0828-perf-dsv41-lsa-group32-dominance-fp8-summary.patch) | PR `#954` (backed by `#976`, `#1014`, `#1017`) | Upgrades opt-in `VLLM_DS41_LSA=1` from BF16 block-8 mean summaries (`STATE_BYTES=100`) to `float8_e4m3fn` Group-32 dominance-blended summaries (`0.85 * mean + 0.15 * k*`, `STATE_BYTES=84`) and routes `r=1` decode at `score_width >= 262144` through `PageShortlistPlan`. | **Default-off invariant (`VLLM_DS41_LSA` unset/`0`)**: keeps `state_content_bytes=68`, declares zero LSA kernels (`summarize_blocks`/`QueryReusePlan` skipped, `jit_new_files = 0`), writes no summaries, and uses `IndexerV41Plan` unchanged. When `VLLM_DS41_LSA=1`, halves LSA summary overhead (`+16` vs `+32 B/slot`). | `tests/models/test_deepseek_v4_1_lsa.py` (updated) |
| [`0829`](vllm/0829-fix-dsv41-hold-back-an-unfinished-next-parameter-tag.patch) | PR `#1236` (found by this candidate's release tests, 2026-11-19) | `vllm/parser/deepseek_v41.py`: `0824`'s text-path argument converter and open-value extender again hold back an unfinished next parameter tag while a value is open, as the previous parser did. Without it, when the model leaves out a parameter closer, the next head was streamed as argument text. | Tool-argument streaming on the text path only (no token ids); the token-provenance path, complete arguments and `0824`'s closer rules are unchanged. Zero kernel, weight, graph, or collective changes. | `tests/parser/test_deepseek_v41_structural_closers.py` (extended); restores `tests/parser/engine/test_deepseek_v41.py::test_partial_tag_after_gt_is_held_back` |

## Benchmark Carry-Over & Promotion (`spark-ds41f-1120`)

Promoted on `2026-11-20` as `spark-ds41f-1120` ([`manifests/baselines/2026-11-20-spark-ds41f-1120.json`](../../../manifests/baselines/2026-11-20-spark-ds41f-1120.json), [`releases/spark-ds41f-1120/`](../../../releases/spark-ds41f-1120/)) across all 12 hardware shapes (`spark-ds41f:1120`, image ID `sha256:5c12472565a708518ec0566b6653b5bd324e06881a34b90f69d6b8210a386d08` verified across `dgx1..dgx4` and `sp1..sp8`).

Per the `rc12 -> rc13 -> rc14` carry-over precedent recorded in [`manifests/baselines/2026-10-30-spark-ds41f-1030.json`](../../../manifests/baselines/2026-10-30-spark-ds41f-1030.json), because `candidate-62` keeps `VLLM_DS41_LSA` unset across every cluster shape (`0828` is strictly gated behind `VLLM_DS41_LSA=1` — zero LSA kernel declarations or JIT compilation, identical 68-byte index cache, identical `IndexerV41Plan` execution) and preserves `candidate-61-r2`'s exact active TileLang/TileKernels/NCCL/SparkNet bits (`jit_new_files = 0`), weights, and unconstrained `/v1/chat/completions` hot paths:

- **In-image & `dgx` release gate on `candidate-62` (`spark-ds41f:candidate-62` / `1120`)**:
  - `bundles/release-tests`: `12,023 passed, 356 skipped, 0 failed` across `169` files ([`receipts/release-tests/verdict.json`](receipts/release-tests/verdict.json)), including `106` parser tests across `test_deepseek_v41.py` and `test_deepseek_v41_structural_closers.py`.
  - Two-boot verification on `tp3-triangle` (`dgx1..dgx3`): startup `MemAvailable` floor `8.27 / 8.28 GiB` head (`10.11..10.34 GiB` workers), `0.0` swap growth, `7/7` wide determinism & `7/7` wide greedy identical, live smoke pass (`#1036`, `#943`, `#944`, `#945`, `#969`).
  - Two-boot verification on `tp1` (`dgx4`): startup `MemAvailable` floor `14.09 / 14.60 GiB`, `0.0` swap growth, `6/6` greedy identical.
- **All 12 shapes carried over from 2-boot `candidate-61` / `candidate-61-r2` runs**:
  - `manifests/suites/2026-11-20-spark-ds41f-1120-{tp4,tp3,tp3-switched,tp4-switched,tp8-switched}{,-single-stream,-sampled}/` and `manifests/benchmarks/2026-11-20-spark-ds41f-1120-*.json` record the full 2-boot Sonar comparisons, startup-memory receipts (`> 5.0 GiB` head & workers, `0.0` swap growth), and `bin/spark bench` matrices across all 12 shapes; summary receipts are stored under [`receipts/verdict.json`](receipts/verdict.json), [`receipts/results.json`](receipts/results.json), and [`receipts/large-files-on-nodes.tsv`](receipts/large-files-on-nodes.tsv).
