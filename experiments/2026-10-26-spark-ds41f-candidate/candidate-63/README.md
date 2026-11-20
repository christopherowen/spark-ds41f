# `candidate-63` — Boot-to-Boot Variance Reduction

Base commit: `a45b79195` (`spark-ds41f-1120` / `candidate-62`).
vLLM base revision: `19f2c20ed4d64ae22bd009ceacadd85e2a81e354`.
vLLM patch series: [`vllm/series`](vllm/series) (`704` active patches, `0001..0831`).
vLLM patch head: `95acec32065b868f42d798cbf10312252c56192c`.
vLLM expected tree: `83e9a68470443b359dac4ea7aee652ecb72c5f00`.
Patchset SHA-256: `0138f3fe6386e6152f0c4afb552c833f0ce98926649579d7646c3e3fbc7e48ed`.

TileLang, TileKernels, SparkNet, NCCL, CUDA toolkit, and model weights are unchanged from `candidate-62` (`jit_new_files = 0`).

## Changes

Four targeted fixes address the remaining sources of boot-to-boot variance observed across `candidate-61`/`candidate-62` 2-boot suites, 17 identical learner-off boots (`experiments/2026-11-17-dspark-turn-learning`), physical page maps (`experiments/2026-11-17-c1-floor-now/receipts/engine-pages-ring.txt`), and 4-rank Torch Profiler traces:

| # | Scope | Artifact | Rationale | Quality / Serving Impact |
| --- | --- | --- | --- | --- |
| 1 | vLLM loader | [`vllm/0830-perf-loader-ladder-64-KiB-buddy-sponge-allocate-dens.patch`](vllm/0830-perf-loader-ladder-64-KiB-buddy-sponge-allocate-dens.patch) | (a) Walk 64 KiB buddy targets `512 MiB -> 4 MiB` before falling back to `2 MiB..512 KiB`, eliminating the `512 MiB -> 2 MiB` cliff on aged nodes (4 KiB kernels keep their `2 MiB` ceiling). (b) Allocate temporary pre-tile MoE storages before `model._direct_weight_arena` so `nvidia.ko`'s bottom-up buddy drain places leftover fragments in temporary MoE buffers and the permanent ~11 GiB L2-prefetched dense arena claims top-order contiguous blocks. (c) Retain (and headroom-trim to `>= 16 GiB`) the target's buddy sponge across `DraftModelSpeculator.load_model` via `hold_weight_arena_sponge` / `trim_held_weight_arena_sponges` and release once before KV cache allocation. | None; physical page placement and sponge lifetime only. |
| 2 | Host cooling | [`spark_ds41f/bench.py`](../../../spark_ds41f/bench.py) | Lower `COOL_BELOW_C` from `55.0` to `49.0` C. Back-to-back boots start at `49.5-51.5 C` (`r = +0.641` boot-order drift, `6-9 MHz` boost droop) and previously bypassed `55.0 C`, while every `dgx` and `spaceport` node clears `< 49.0 C` at fan floor 12 well within `COOL_TIMEOUT_S`. | None; pre-run thermal floor only. |
| 3 | Grid estimator | [`spark_ds41f/sonar.py`](../../../spark_ds41f/sonar.py) | Add fixed lower-quartile order-statistic floor (`floor_samples`, fastest 25% of replays) to `sonar.engine` (`step_ms_floor`, `forward_ms_floor`, per-shape `floor_change_pct`, `floor_decode_gm_pct`, and `estimator="floor25"` on `compare` / `boot_ratios` / `spread`). Eliminates the `median + 5 * MAD` breakdown knee when 35-50% of replays experience asynchronous single-rank `+0.45..+1.3 ms` forward spikes (reducing 127-pair A/A RMS from `0.524%` to `0.424%` and turning `c61 1x3 @4K` from `+1.91%` to `+0.42%`) while preserving existing `5 * MAD` fields. | None; host-side post-processing only. |
| 4 | vLLM grid harness | [`vllm/0831-perf-worker-synchronize-TP-ranks-and-pause-cyclic-GC.patch`](vllm/0831-perf-worker-synchronize-TP-ranks-and-pause-cyclic-GC.patch) | In `GPUModelRunner.engine_grid()`, drain CUDA, run `gc.collect()`, and barrier TP ranks on `tp_group.cpu_group` after warmup, then pause cyclic GC across the timed replay loop (restoring GC in `finally`) so asynchronous per-rank GC pauses cannot cause multi-millisecond collective-entry skew. | None; `engine_grid` profiling harness only. |

## Verification (Pre-Build / CPU)

- Applied the full 704-patch vLLM series cleanly to `19f2c20ed4d64ae22bd009ceacadd85e2a81e354` in a fresh checkout on `dgx1` (`git diff --check` clean; `patch_head = 95acec32065b868f42d798cbf10312252c56192c`, `expected_tree = 83e9a68470443b359dac4ea7aee652ecb72c5f00`).
- vLLM CPU unit tests inside `spark-ds41f:candidate-62` against the patched tree (`pytest --noconftest`):
  - `tests/model_executor/model_loader/test_direct_loader.py`: `50 passed` (includes new 64 KiB ladder, hold/trim/release lifecycle, and MoE-before-dense allocation order tests).
  - `tests/v1/worker/test_engine_grid_sync.py`: `2 passed` (TP barrier ordering, GC pause during timed replays, and GC restore on exception).
- Deployment host unit tests: `python3 -m unittest tests/test_bench.py tests/test_doctor.py` (all passed), plus `bin/spark doctor`, `bin/spark build check`, and `git diff --check`.

## Queue Commands (After Active `dgx1` Hold Releases)

```bash
bin/spark candidate queue tp4-ring experiments/2026-10-26-spark-ds41f-candidate/candidate-63/build.json --apply
bin/spark candidate queue tp4-ring experiments/2026-10-26-spark-ds41f-candidate/candidate-63/screen-tp4.json --apply
bin/spark candidate queue tp4-switched experiments/2026-10-26-spark-ds41f-candidate/candidate-63/screen-tp4-switched.json --apply
```
