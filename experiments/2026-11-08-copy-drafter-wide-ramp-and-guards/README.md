# CopyDrafter Negative-Slot Guards, Gentler Cap Decay, Level-Anchor Suffix Pruning, and 31/63-Token Lookahead Ramp

- **Date**: 2026-11-08
- **Base commit**: `8e5ea208f3d22bc69df91ace94d6975932c858a8` (`spark-ds41f-1107`, vLLM series `0001..0761`, patch head `2756478b60b9`, tree `ed1bdb7a7684`)
- **Candidate patch**: `vllm/0762-perf-spec-decode-copy-drafter-guards-and-wide-ramp.patch` (patch head `673fedf2bf81`, tree `a411d506fd4d`)
- **Profiles**: `tp3.json`, `tp4.json`
- **Hardware**: `sp5` (`NVIDIA GB10`, container `spark-ds41f:1107`, 2 fresh container boots under `flock /tmp/gpu.lock`)

## Hypothesis & Intended Variables

1. **Negative `slot` bounds guard in `copy_alloc_kernel` and `copy_apply_kernel`**:
   - `copy_match_kernel` guards `if slot >= 0:` before reading `total_len[slot]`, whereas `copy_alloc_kernel` and `copy_apply_kernel` previously indexed `total_len[slot]`, `int_state[slot]`, `float_state[slot]`, and `history[slot]` unconditionally even when `idx_mapping` carried negative padded slots (`slot < 0`). Guarding `slot >= 0` in both kernels eliminates out-of-bounds global memory reads and writes on padded batch slots while zero-filling `proposed_out[req]` and `accepted_out[req]`.
2. **Gentler cap decay on partial acceptance (`initial <= accepted < live`)**:
   - In `copy_alloc_kernel`, when a copy proposal of length `live > initial` was partially accepted (`0 < accepted < live`, e.g., `15/16` accepted when a copied block contains a single edited token), resetting `doublings = 0` dropped the next step all the way back to `initial = 4`, requiring three full-cap steps (`4 -> 8 -> 16`) to recover lookahead. Stepping `doublings` down by 1 (`T.max(doublings - 1, 0)`) whenever `accepted >= initial` (while still resetting `doublings = 0` when `accepted < initial`) preserves ramp momentum across isolated edits inside copied blocks.
3. **Level-anchor suffix pruning and 1-variable tile hit reduction in `copy_match_kernel`**:
   - In `copy_match_kernel`, each candidate tile previously probed every ladder level `i = 0 .. count - 1` sequentially without checking whether the target level anchor token `history[slot, e - levels[i]] == key[levels[i] - 1]` matched first, scanned tiles below `need_len`, and executed `5 * count = 25` warp `shfl_xor` steps on every hit tile even though only `hit[0]` is needed to test whether any level matched in the tile. Probing the level-anchor token first, bounding the tile loop by `warp_tile + 32 >= need_len`, and reducing only `hit[0]` inside the tile loop cuts redundant shared-memory and warp-shuffle work on repetitive/boilerplate histories while remaining 100% bit-exact.
4. **Wide single-stream copy lookahead ramp (`max_drafts = 31` and `63`)**:
   - `copy_apply_kernel`, `assign_draft_budget_kernel`, and `verify_kernel` already support up to `64` draft tokens, but `greedy_rejection_kernel` in `vllm/v1/worker/gpu/sample/tilelang_sampling.py` restricted `width <= 32` (`max_drafts <= 31`). Extending `greedy_rejection_kernel` to support `1 <= width <= 65` via `passes = -(-width // 32)` (keeping exact 1-pass code generation for `width <= 32`) alongside `SPARK_DS41F_COPY_MAX_DRAFTS` in `SpeculativeConfig` and `--lookahead` in `benchmarks/kernels/benchmark_copy_drafts.py` unlocks `max_drafts = 31` (`32` total rows = 1 `decode_32` MMA tile) and `max_drafts = 63` (`64` total rows = 1 `decode_64` MMA tile and 1 `DECODE_CHUNK = 64` indexer wave).

## Exact Commands

```bash
# Boot 1 & Boot 2 unit and scheduler test gates (fresh container per boot):
flock /tmp/gpu.lock docker run --rm --gpus all --ipc=host \
  -v /tmp/vllm-1107-repo/...:/opt/spark-ds41f/candidate/vllm/... \
  spark-ds41f:1107 pytest --noconftest -v \
  /opt/spark-ds41f/candidate/vllm/tests/v1/spec_decode/test_copy_drafts.py \
  /opt/spark-ds41f/candidate/vllm/tests/v1/core/test_copy_drafts_scheduler.py

# Boot 1 & Boot 2 microbenchmark gate (fresh container per boot):
flock /tmp/gpu.lock docker run --rm --gpus all --ipc=host \
  -v /tmp/vllm-1107-repo/...:/opt/spark-ds41f/candidate/vllm/... \
  spark-ds41f:1107 python3 /opt/spark-ds41f/candidate/vllm/benchmarks/kernels/benchmark_copy_drafts.py --gate

# Boot 1 & Boot 2 2-boot screening harness (fresh container per boot):
flock /tmp/gpu.lock docker run --rm --gpus all --ipc=host \
  -v /tmp/copy-drafter-bench:/bench \
  -v /tmp/vllm-1107-repo/...:/opt/spark-ds41f/candidate/vllm/... \
  spark-ds41f:1107 python3 /bench/screen_copy_drafter_wide_ramp.py \
  --boot {1,2} --out /bench/screen-copy-drafter-wide-ramp-boot{1,2}.json
```

## Results Across 2 Fresh Container Boots on GB10 (`sp5`)

### 1. Functional & Regression Gates (`pytest` + `benchmark_copy_drafts.py --gate`)

- **Boot 1 (`receipts/gpu-pytest-boot1.txt`)**: `63 passed in 66.64s` (includes new tests `test_negative_slot_bounds_guard_in_alloc_and_apply`, `test_partial_acceptance_steps_down_cap_doublings_gently`, and `test_wide_lookahead_31_and_63_greedy_verification_and_ramp[31,63]`).
- **Boot 2 (`receipts/gpu-pytest-boot2.txt`)**: `63 passed in 66.43s`.
- **`benchmark_copy_drafts.py --gate` (`receipts/benchmark-copy-drafts-gate-boot{1,2}.jsonl`)**: `"slower": []` across both boots (all 24 copy cells and all 12 greedy/block verification cells pass budget).

| Storage | Requests (`B`) | Context (`total_len`) | Boot 1 (`us`) | Boot 2 (`us`) | Mean (`us`) | Budget (`us`) | Slower |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `uva` | 1 | 4,096 | 30.43 | 30.40 | 30.41 | 330.0 | `False` |
| `uva` | 1 | 65,536 | 28.61 | 26.30 | 27.45 | 330.0 | `False` |
| `uva` | 1 | 262,144 | 38.88 | 40.64 | 39.76 | 330.0 | `False` |
| `uva` | 1 | 1,048,576 | 72.06 | 71.39 | 71.72 | 330.0 | `False` |
| `uva` | 8 | 4,096 | 30.72 | 28.58 | 29.65 | 857.0 | `False` |
| `uva` | 8 | 65,536 | 39.94 | 38.62 | 39.28 | 857.0 | `False` |
| `uva` | 8 | 262,144 | 81.70 | 77.47 | 79.59 | 857.0 | `False` |
| `uva` | 8 | 1,048,576 | 227.33 | 214.72 | 221.03 | 857.0 | `False` |
| `uva` | 16 | 4,096 | 30.82 | 32.74 | 31.78 | 1714.0 | `False` |
| `uva` | 16 | 65,536 | 49.12 | 51.39 | 50.25 | 1714.0 | `False` |
| `uva` | 16 | 262,144 | 106.27 | 102.40 | 104.34 | 1714.0 | `False` |
| `uva` | 16 | 1,048,576 | 366.27 | 356.35 | 361.31 | 1714.0 | `False` |
| `device` | 1 | 4,096 | 24.26 | 24.58 | 24.42 | 330.0 | `False` |
| `device` | 1 | 65,536 | 24.38 | 24.58 | 24.48 | 330.0 | `False` |
| `device` | 1 | 262,144 | 34.91 | 34.82 | 34.86 | 330.0 | `False` |
| `device` | 1 | 1,048,576 | 73.41 | 73.70 | 73.56 | 330.0 | `False` |
| `device` | 8 | 4,096 | 22.53 | 24.26 | 23.40 | 857.0 | `False` |
| `device` | 8 | 65,536 | 36.70 | 34.50 | 35.60 | 857.0 | `False` |
| `device` | 8 | 262,144 | 81.66 | 74.43 | 78.05 | 857.0 | `False` |
| `device` | 8 | 1,048,576 | 269.18 | 265.95 | 267.56 | 857.0 | `False` |
| `device` | 16 | 4,096 | 26.34 | 26.30 | 26.32 | 1714.0 | `False` |
| `device` | 16 | 65,536 | 51.23 | 46.78 | 49.00 | 1714.0 | `False` |
| `device` | 16 | 262,144 | 136.93 | 114.34 | 125.64 | 1714.0 | `False` |
| `device` | 16 | 1,048,576 | 408.16 | 346.85 | 377.50 | 1714.0 | `False` |

### 2. Boilerplate Suffix Match Pruning (`copy_match_kernel`, Base `0761` vs Candidate `0762`)

Workload: histories containing repeated 16-token boilerplate matches (`level = 16` hits every 64 tokens) before an earlier 256-token full suffix match (`level = 256`), exercising the level-anchor probe (`history[slot, e - levels[i]] == key[levels[i] - 1]`), `need_len` tile bound, and 1-variable `hit[0]` warp reduction. Candidate `0762` produces 100% bit-exact `(source_start, matched_level, priority)` outputs across all 18 cells while speeding up 256K suffix scans by **1.18x** at `B=1` (`40.64 -> 34.45 us` UVA, `40.66 -> 34.51 us` device) and **1.07x–1.13x** at `B=8..16`.

| Storage | Requests (`B`) | Context (`total_len`) | Base `0761` Boot 1 / Boot 2 (`us`) | Base Mean (`us`) | Cand `0762` Boot 1 / Boot 2 (`us`) | Cand Mean (`us`) | Speedup | Bit-Exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `uva` | 1 | 16,384 | 22.21 / 22.40 | 22.30 | 22.21 / 22.24 | 22.23 | **1.004x** | `True` |
| `uva` | 1 | 65,536 | 26.34 / 26.66 | 26.50 | 24.26 / 24.42 | 24.34 | **1.089x** | `True` |
| `uva` | 1 | 262,144 | 40.64 / 40.96 | 40.80 | 34.46 / 34.82 | 34.64 | **1.178x** | `True` |
| `uva` | 8 | 16,384 | 26.30 / 28.64 | 27.47 | 24.26 / 24.58 | 24.42 | **1.125x** | `True` |
| `uva` | 8 | 65,536 | 35.04 / 36.96 | 36.00 | 34.82 / 36.54 | 35.68 | **1.009x** | `True` |
| `uva` | 8 | 262,144 | 77.54 / 90.85 | 84.19 | 73.73 / 86.08 | 79.91 | **1.054x** | `True` |
| `uva` | 16 | 16,384 | 30.43 / 30.91 | 30.67 | 26.82 / 28.70 | 27.76 | **1.105x** | `True` |
| `uva` | 16 | 65,536 | 45.06 / 45.44 | 45.25 | 43.01 / 45.06 | 44.03 | **1.028x** | `True` |
| `uva` | 16 | 262,144 | 104.00 / 116.74 | 110.37 | 100.03 / 112.77 | 106.40 | **1.037x** | `True` |
| `device` | 1 | 16,384 | 22.21 / 22.53 | 22.37 | 22.21 / 22.53 | 22.37 | **1.000x** | `True` |
| `device` | 1 | 65,536 | 26.30 / 26.56 | 26.43 | 24.26 / 24.67 | 24.47 | **1.080x** | `True` |
| `device` | 1 | 262,144 | 40.67 / 42.75 | 41.71 | 34.53 / 35.17 | 34.85 | **1.197x** | `True` |
| `device` | 8 | 16,384 | 26.62 / 26.62 | 26.62 | 24.58 / 24.58 | 24.58 | **1.083x** | `True` |
| `device` | 8 | 65,536 | 39.14 / 43.01 | 41.08 | 38.91 / 40.96 | 39.94 | **1.029x** | `True` |
| `device` | 8 | 262,144 | 81.92 / 102.37 | 92.15 | 79.87 / 99.04 | 89.46 | **1.030x** | `True` |
| `device` | 16 | 16,384 | 30.72 / 30.82 | 30.77 | 28.67 / 28.93 | 28.80 | **1.068x** | `True` |
| `device` | 16 | 65,536 | 51.20 / 59.14 | 55.17 | 49.63 / 57.31 | 53.47 | **1.032x** | `True` |
| `device` | 16 | 262,144 | 120.74 / 162.02 | 141.38 | 118.30 / 157.44 | 137.87 | **1.025x** | `True` |

### 3. Wide Lookahead Kernel Latency (`lookahead = 16, 31, 63`)

Across `lookahead = 16` (`17` rows at `B=1`), `31` (`32` rows = 1 `decode_32` MMA tile at `B=1`), and `63` (`64` rows = 1 `decode_64` MMA tile at `B=1`), `copy_match + copy_alloc + copy_apply` adds `< 0.5 us` at `B=1` (`24.61 -> 26.74 -> 24.99 us`) and `greedy_rejection_sample` runs in `4.13–5.82 us` at `lookahead = 16..31` and `4.19–6.22 us` at `lookahead = 63` (2-pass warp scan).

| Lookahead (`max_drafts`) | Requests (`B`) | Wide Verification Rows | `copy_match+alloc+apply` Boot 1 / Boot 2 (`us`) | `copy` Mean (`us`) | `greedy_rejection_sample` Boot 1 / Boot 2 (`us`) | `greedy_verify` Mean (`us`) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 1 | 17 | 24.61 / 25.02 | 24.81 | 5.82 / 4.26 | 5.04 |
| 16 | 8 | 59 | 43.01 / 42.08 | 42.55 | 4.13 / 4.22 | 4.17 |
| 16 | 16 | 107 | 57.34 / 57.31 | 57.33 | 4.19 / 4.22 | 4.21 |
| 31 | 1 | 32 | 26.50 / 26.78 | 26.64 | 5.82 / 4.45 | 5.13 |
| 31 | 8 | 74 | 42.69 / 43.01 | 42.85 | 5.82 / 5.86 | 5.84 |
| 31 | 16 | 122 | 58.02 / 59.07 | 58.55 | 5.82 / 5.82 | 5.82 |
| 63 | 1 | 64 | 26.34 / 26.62 | 26.48 | 5.86 / 6.14 | 6.00 |
| 63 | 8 | 106 | 42.75 / 43.01 | 42.88 | 5.86 / 6.14 | 6.00 |
| 63 | 16 | 154 | 55.46 / 56.06 | 55.76 | 5.82 / 6.14 | 5.98 |

### 4. End-to-End Multi-Step Copy Decode Walks (`2,024` Generated Tokens)

Three 2,048-token copy workloads evaluated end-to-end through `copy_match -> copy_alloc -> copy_apply -> greedy_rejection_sample` with step cost priced by `AdaptiveVerificationManager._step_cost(total_rows)`:
- `verbatim_2048`: 2,048-token verbatim span copied from earlier in context.
- `spliced_2048_edits8`: 2,048-token copied span with 8 single-token edits spliced every ~250 tokens.
- `spliced_2048_edits16`: 2,048-token copied span with 16 single-token edits spliced every ~125 tokens.

| Workload (2,024 Generated Tokens) | Arm | `max_drafts` | Decode Steps | Step Reduction vs Base | Mean Tokens / Step | Total Verified Rows | Relative Step-Cost Units | Tokens / Cost Unit | Throughput Gain vs Base |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `verbatim_2048` | `base_0761_k16` | 16 | 120 | 1.000x (ref) | 16.87 | 2,029 | 159.51 | 12.69 | 1.000x (ref) |
| `verbatim_2048` | `cand_0762_k16` | 16 | 120 | **1.000x** | 16.87 | 2,029 | 159.51 | 12.69 | **1.000x** |
| `verbatim_2048` | `cand_0762_k31` | 31 | 65 | **1.846x** | 31.14 | 2,039 | 96.70 | 20.93 | **1.649x** |
| `verbatim_2048` | `cand_0762_k63` | 63 | 34 | **3.529x** | 59.53 | 2,040 | 61.25 | 33.05 | **2.604x** |
| `spliced_2048_edits8` | `base_0761_k16` | 16 | 191 | 1.000x (ref) | 10.60 | 2,404 | 238.06 | 8.50 | 1.000x (ref) |
| `spliced_2048_edits8` | `cand_0762_k16` | 16 | 189 | **1.011x** | 10.71 | 2,386 | 235.70 | 8.59 | **1.011x** |
| `spliced_2048_edits8` | `cand_0762_k31` | 31 | 145 | **1.317x** | 13.96 | 2,607 | 187.99 | 10.77 | **1.267x** |
| `spliced_2048_edits8` | `cand_0762_k63` | 63 | 130 | **1.469x** | 15.57 | 2,963 | 174.62 | 11.59 | **1.364x** |
| `spliced_2048_edits16` | `base_0761_k16` | 16 | 256 | 1.000x (ref) | 7.91 | 2,733 | 309.69 | 6.54 | 1.000x (ref) |
| `spliced_2048_edits16` | `cand_0762_k16` | 16 | 255 | **1.004x** | 7.94 | 2,724 | 308.51 | 6.56 | **1.003x** |
| `spliced_2048_edits16` | `cand_0762_k31` | 31 | 217 | **1.180x** | 9.33 | 2,896 | 266.99 | 7.58 | **1.159x** |
| `spliced_2048_edits16` | `cand_0762_k63` | 63 | 203 | **1.261x** | 9.97 | 3,030 | 252.45 | 8.02 | **1.226x** |

## Memory Observations

- Zero change to model weights, KV cache layout, or persistent state buffer shapes (`int_state` remains `[max_num_reqs, 6] int32`, `float_state` remains `[max_num_reqs, 2] float32`, `history` remains `[max_num_reqs, max_model_len] int32`).
- Guarding `slot >= 0` in `copy_alloc_kernel` and `copy_apply_kernel` prevents negative padded entries in `idx_mapping` from reading or writing `history[-1]` or `int_state[-1]`.

## Conclusion

1. `vllm/0762-perf-spec-decode-copy-drafter-guards-and-wide-ramp.patch` fixes the latent negative-slot bounds hazard in `copy_alloc_kernel` and `copy_apply_kernel`, preserves ramp momentum across isolated edits via gentler partial-acceptance cap decay (`doublings - 1`), and accelerates boilerplate-heavy suffix scans in `copy_match_kernel` by up to **1.18x** (`40.64 -> 34.45 us` at 256K) with 100% bit-exact outputs.
2. Extending `greedy_rejection_kernel` to `width <= 65` and exposing `SPARK_DS41F_COPY_MAX_DRAFTS` unlocks `max_drafts = 31` (`decode_32` MMA tile) and `max_drafts = 63` (`decode_64` MMA tile):
   - At `max_drafts = 31`, decode steps drop by **1.846x** (`120 -> 65` steps, **1.649x** tokens/cost-unit) on verbatim copies and by **1.317x / 1.180x** (**1.267x / 1.159x** tokens/cost-unit) on 8-edit and 16-edit spliced copies.
   - At `max_drafts = 63`, decode steps drop by **3.529x** (`120 -> 34` steps, **2.604x** tokens/cost-unit) on verbatim copies and by **1.469x / 1.261x** (**1.364x / 1.226x** tokens/cost-unit) on 8-edit and 16-edit spliced copies.
