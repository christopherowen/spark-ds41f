# Experiment: Tracker Reuse, Automatic KV-Cache Pool Deduction, and Recipe-Matched Step Harness (`iket_profile.py`, `iket_step.py`, `step_bundle.py`, `step_diff.py`)

## Hypothesis & Objective

Previously, profiling DS4.1 decode steps under `run-iket` suffered from four operational bottlenecks:

1. **Redundant dry runs & CUDA-graph dry-run hangs (`iket_profile.py`)**:
   - Every invocation of `iket_profile.py` ran a full `run-iket --skip-run` dry run before the measured pass, doubling wall-clock profiling time and exposing deterministic harnesses (`iket_step.py`, `iket_hot.py`) to `run-iket`'s CUDA-graph callback hang (~1 in 5 dry runs).
2. **GB10 unified memory OOMs when `run-iket` allocates its device context buffer (`iket_profile.py`)**:
   - In a dry run (`--skip-run`), `run-iket` allocates only lightweight per-warp event counters. In the measured run, `run-iket` allocates the full context buffer (`Auto-computed context buffer size: <G> GB (<B> bytes)`) on the GPU *on top of* the vLLM engine's weights and KV cache.
   - Without automatically deducting `pool_bytes` (+ a safety margin) from `--kv-bytes` / `--kv-cache-memory-bytes`, a step or serving run sized near memory capacity succeeds in the dry run and OOMs during the measured run or exit decode.
3. **`c8` grid dummy step mismatch in `iket_step.py` and `step_bundle.py`**:
   - At `c8` (`--requests 8`) with CUDA graphs and adaptive verification enabled (`adaptive_verification_row_cost=0.035`), the decode step verifies 2 tokens per request (replaying the 16-row graph from the recipe's capture ladder), whereas `iket_step.py`'s `--grid` dummy step defaulted `--grid-rows` to `1 + drafts = 6` rows per request (`48` rows total) unless `--grid-rows 2` was manually specified.
   - Additionally, in `step_bundle.py`, the IKET run omitted `--prefix-caching` and `--capture-ladder` while the `nsys` base run included them, causing the two runs to capture and replay different CUDA graph ladders.
4. **`nsys` SQLite requirement in `step_diff.py`**:
   - `step_diff.py` could only read `CUPTI_ACTIVITY_KIND_KERNEL` from an `nsys` SQLite export (`*.sqlite`), preventing `run-iket --warp-trace` decoded JSON files (`*.json`) from being diffed against the engine grid's dummy step without `nsys`.

This experiment upgrades `iket_profile.py`, `iket_step.py`, `step_bundle.py`, and `step_diff.py`:

1. **`harness/iket_profile.py`**:
   - **`--use-tracker-dir DIR`**: Reuses an existing `tracker/` directory (`pid_*/tracker.json`) from a prior dry run, skipping the dry run completely.
   - **`--warp-trace`**: Runs a single-pass uninstrumented `run-iket profile --warp-trace` capture without a dry run.
   - **`OUT_DIR/pool.json` & `--deduct-kv-bytes` (`--kv-safety-mib`, default `256` MiB)**: Parses `Auto-computed context buffer size: <G> GB (<B> bytes)` (including comma-separated byte counts), writes `pool.json` (`pool_gb`, `pool_bytes`, `safety_margin_bytes`, `recommended_kv_deduction_bytes`, `max_ts_cnt_per_warp`, `per_process_counts`), and rewrites `--kv-bytes` / `--kv-cache-memory-bytes` in the measured command.
2. **`harness/iket_step.py` & `harness/iket_hot.py`**:
   - **`default_grid_rows`**: Defaults `--grid-rows` to `2` when `requests >= 8` and `not eager` (matching adaptive verification's 16-row graph at `c8`).
   - Wires `--warps`, `--granularity {fine,phase}`, `--payloads`, and `--iket-cache-dir` into `iket.enable(...)` and `result.json`.
   - Calls `torch.distributed.destroy_process_group()` (when initialized) alongside `engine_core.shutdown()`, `gc.collect()`, and `torch.cuda.empty_cache()` before `run-iket`'s `atexit` decode on GB10 unified memory.
3. **`step_bundle.py` & `step_diff.py`**:
   - Aligns CUDA graph capture flags (`--prefix-caching --capture-ladder`) across both the IKET pass and the base pass, defaults `--granularity phase`, `--deduct-kv-bytes`, and `--iket-cache-dir /c/iket-tilelang-cache`, and supports `--base-mode {iket,nsys}`.
   - `step_diff.py` accepts either `*.sqlite` (`nsys`) or `*.json` (`run-iket --warp-trace`) traces via `load_rows()`.

## Verification on `dgx4`

Tested on `dgx4` (`spark-ds41f:1030`, holding `/tmp/spark-lab-gpu.lock`):

1. **Unit tests (`test_iket_profile.py`)**:
   - `pytest --noconftest -v test_iket_profile.py`: **4 passed in 0.02s** (`test_parse_pool_size_and_rewrite_kv_bytes`, `test_multiprocess_tracker_counts_and_use_tracker_dir`, `test_default_grid_rows_and_step_spec`, `test_step_diff_supports_iket_json`).
2. **End-to-end GPU `run-iket` verification (`iket_profile.py` + `iket_hot.py --only mhc`)**:

| Pass | Dry Run Executed | `max_ts_cnt_per_warp` | `pool_bytes` | Wall Time | Delta |
| :--- | :---: | ---: | ---: | ---: | ---: |
| Pass 1 (full dry + measured) | Yes | `32` (`pid_0x35`) | `127,180 B` | `16.47 s` | Baseline |
| Pass 2 (`--use-tracker-dir`) | No (skipped) | `32` (`pid_0x35`) | `127,180 B` | `6.81 s` | `-58.7%` |

## Files

- `harness/iket_profile.py`: `run-iket` wrapper with `--use-tracker-dir`, `--warp-trace`, `pool.json`, and `--deduct-kv-bytes`.
- `harness/iket_step.py`: 6-layer + drafter DS4.1 step harness with adaptive-verification `c8` `--grid-rows` default, full IKET sampling/cache flags, and clean teardown.
- `harness/iket_hot.py`: Standalone hot-kernel harness supporting `--sample-warps`, `--granularity`, `--payloads`, and `--iket-cache-dir`.
- `harness/sass_check.py`: SASS disassembly checker for predicated `PMTRIG`, hoisted `%globaltimer` reads, and unsettled `BAR.SYNC` instructions.
- `step_bundle.py`: Step profiling bundle generator supporting `--base-mode {iket,nsys}` and `--deduct-kv-bytes`.
- `step_diff.py`: Decode step vs. engine grid diff tool supporting both `nsys` SQLite and `run-iket` JSON traces.
- `test_iket_profile.py`: Unit test suite for `iket_profile.py`, `iket_step.py`, `step_bundle.py`, and `step_diff.py`.
