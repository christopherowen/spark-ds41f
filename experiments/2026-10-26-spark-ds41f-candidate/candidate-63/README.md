# `candidate-63` — Boot-to-Boot Variance Reduction

Base commit: `a45b79195` (`spark-ds41f-1120` / `candidate-62`).
Final candidate commit: `a57cb9f55` (`PR #1259`, image `spark-ds41f:candidate-63-r4`, `sha256:6c31c7331e4b1adb1d3d4ced84446003b5f076b64e3e01043148ba18ada51cb5`).
vLLM base revision: `19f2c20ed4d64ae22bd009ceacadd85e2a81e354`.
vLLM patch series: [`vllm/series`](vllm/series) (`705` active patches, `0001..0832`).
vLLM patch head: `0f7c4e6e999625c78e051ab57acd36c0fb67c4a6`.
vLLM expected tree: `b4e8748fc2a0a1461c90788da0c8a122b3130071`.
Patchset SHA-256: `8bd8c5c46c4d45216d94b7437ac7f9cc823625cbb4d31fdce1cd9f7c4ac9eee6`.

TileLang, TileKernels, SparkNet, NCCL, CUDA toolkit, and model weights are unchanged from `candidate-62` (`jit_new_files = 0`).

## Changes

Six targeted fixes eliminate the physical-memory, allocator, thermal, and measurement sources of boot-to-boot variance on GB10 (`64 KiB` ARM64 pages, `iommu.passthrough=0`). See [`experiments/2026-11-21-boot-variance-contiguous-pool/README.md`](../../2026-11-21-boot-variance-contiguous-pool/README.md) for full root-cause forensics and all 17 live `tp3-triangle` validation runs across Phases 1–6.

| # | Scope | Artifact | Rationale | Quality / Serving Impact |
| --- | --- | --- | --- | --- |
| 1 | vLLM loader | [`vllm/0830-perf-loader-ladder-64-KiB-buddy-sponge-allocate-dens.patch`](vllm/0830-perf-loader-ladder-64-KiB-buddy-sponge-allocate-dens.patch) | (a) Walk 64 KiB buddy targets `512 MiB -> 4 MiB` before falling back to `2 MiB..512 KiB`, eliminating the `512 MiB -> 2 MiB` cliff on aged nodes (4 KiB kernels keep their `2 MiB` ceiling). (b) Allocate temporary pre-tile MoE storages before `model._direct_weight_arena` when the sponge fallback is used. (c) Retain (and headroom-trim to `>= 16 GiB`) the target's buddy sponge across `DraftModelSpeculator.load_model` and release once before KV cache allocation. | None; physical page placement and sponge lifetime only. |
| 2 | Host cooling | [`spark_ds41f/bench.py`](../../../spark_ds41f/bench.py) | Lower `COOL_BELOW_C` from `55.0` to `49.0` C so back-to-back boots start from a uniform thermal floor (`< 49.0 C` at fan floor 12 well within `COOL_TIMEOUT_S`). | None; pre-run thermal floor only. |
| 3 | Grid estimator | [`spark_ds41f/sonar.py`](../../../spark_ds41f/sonar.py) | Add fixed lower-quartile order-statistic floor (`floor_samples`, fastest 25% of replays) to `sonar.engine` (`step_ms_floor`, `forward_ms_floor`, `estimator="floor25"`), eliminating the `median + 5 * MAD` breakdown knee when 35–50% of replays see asynchronous single-rank spikes while preserving existing `5 * MAD` fields. | None; host-side post-processing only. |
| 4 | vLLM grid harness | [`vllm/0831-perf-worker-synchronize-TP-ranks-and-pause-cyclic-GC.patch`](vllm/0831-perf-worker-synchronize-TP-ranks-and-pause-cyclic-GC.patch) | In `GPUModelRunner.engine_grid()`, drain CUDA, run `gc.collect()`, and barrier TP ranks on `tp_group.cpu_group` after warmup, then pause cyclic GC across the timed replay loop (restoring GC in `finally`). | None; `engine_grid` profiling harness only. |
| 5 | vLLM worker & runtime env | [`vllm/0832-perf-worker-preallocate-contiguous-CUDA-pool-on-64-K.patch`](vllm/0832-perf-worker-preallocate-contiguous-CUDA-pool-on-64-K.patch), [`tp3.json`](tp3.json) | Drop `expandable_segments:True` (`PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6`) to prevent `~58k × 2 MiB` `cuMemCreate` calls from scattering immortal ARM SMMU L3 page tables (`SecPageTables` `4.8 -> 28.5 MiB`) across `225/233` `512 MiB` pageblocks. Bypass `_scoped_allocator_max_split(20)`, run a 1-shot transient `MIGRATE_UNMOVABLE` sub-O13 soak (capped to keep `>= 8 GiB` `MemAvailable`), pre-allocate one contiguous weight pool (`requested_memory - kv_cache_memory_bytes` aligned down to `512 MiB`, `96.5 GiB` on `tp3-triangle`) anchored by a `2 MiB` large-pool tensor, skip the buddy sponge in `direct_loader.py` with `front_clean` dense-first placement, and give KV cache one clean contiguous `4.91 GiB` Segment #2. | None; bit-exact `128/128` greedy (`KL = 0.000e+00`, `perplexity +0.000%`), `load_model` `-12.1 s` (`50.7 -> 38.6 s`), symmetric `103.08 GiB` reserved across ranks. |
| 6 | Host memory hygiene | [`spark_ds41f/cluster.py`](../../../spark_ds41f/cluster.py), [`tests/test_restart_guards.py`](../../../tests/test_restart_guards.py) | Add a 2-pass `512 MiB` `hugetlb` reset (`hugepages-524288kB/nr_hugepages` -> `0`) with post-`hugetlb` `compact_memory=1` in between inside `defragment_buddy_memory()` on `cluster stop` and `cluster start`, reverting freed `MIGRATE_UNMOVABLE` `512 MiB` pageblocks back to `MIGRATE_MOVABLE` and coalescing movable fragments. | None; host-side stop/start buddy recovery only. |

## Verification (`candidate-63-r4`)

- **Patch series & build:** Applied all `705` active vLLM patches cleanly (`patch_head = 0f7c4e6e999625c78e051ab57acd36c0fb67c4a6`, `expected_tree = b4e8748fc2a0a1461c90788da0c8a122b3130071`). `bin/spark build prepare`, `bin/spark build check`, `bin/spark doctor`, and `git diff --check` all passed.
- **Unit tests:** `101 passed` GPU/loader pytest inside `spark-ds41f:candidate-63-r4` (`test_contiguous_cuda_pool.py`, `test_direct_loader.py`, `test_direct_loader_cuda.py`, `test_engine_grid_sync.py`); `906 passed` Linux deployment unit tests (`28.0 s`).
- **Live `tp3-triangle` 3-boot validation (Phase 6, `dgx1..dgx3` after reboot):**
  - `engine grid` `floor25` GM vs `1120-reboot-b1`: **`-0.20%` (b1), `+0.08%` (b2), `-0.18%` (b3)** — Boot 1 -> Boot 3 drift **`+0.02%`** (vs `+1.21%` on `1120` Boot 1 -> Boot 2; `c63-r4` averages **`-1.31%` faster than `1120-reboot-b2`**).
  - `c8` steady-state throughput: **`172.817`, `171.792`, `173.025 tok/s`** (`0` fail).
  - `prefill` (`b3`): `32K = 7,267.8 tok/s`, `262K = 7,468.1 tok/s` (`0` fail).
  - `model` exactness (`b1`, 128 prompts): **`128/128` greedy identical**, `100.000%` top-1 agreement, **`KL = 0.000e+00`**, `perplexity = 1.31608 -> 1.31608 (+0.000%)`.
  - Post-Boot-3 idle state: **`O13 = 189 / 198 / 190`** (`94.5–99.0 GiB` @ `512 MiB`, `216 / 218 / 212` `MIGRATE_MOVABLE` blocks), **`SecPageTables = 8.56 / 8.62 / 8.62 MiB`** (vs `O13 = 8/12/12` and `28.5 MiB` after 1 boot of `1120`).
  - Rank memory symmetry: `torch_reserved = 103.08 GiB` identically on `dgx1/dgx2/dgx3` across all 3 boots (`free_memory = 6.89–7.22 / 8.78–9.11 / 8.93–9.21 GiB`).

## Queue Commands (When Ready for `tp4` Qualification)

```bash
bin/spark candidate queue tp4-ring experiments/2026-10-26-spark-ds41f-candidate/candidate-63/build.json --apply
bin/spark candidate queue tp4-ring experiments/2026-10-26-spark-ds41f-candidate/candidate-63/screen-tp4.json --apply
bin/spark candidate queue tp4-switched experiments/2026-10-26-spark-ds41f-candidate/candidate-63/screen-tp4-switched.json --apply
```
