# Cohort-atomic verification budget, `16..40`-row cost band, and idle-wakeup burst coalescing (`2026-11-08-cohort-sync-verification`)

- **Date**: 2026-11-08
- **Base deployment commit**: `cc20ed18b6ca6521b602f09cb3ba326774166d1c` (Candidate `1107`, `experiments/2026-10-26-spark-ds41f-candidate/1107`)
- **Base vLLM commit**: `320021e3c4dbb202d911ab70e2cd924c75420488` (tree `ed1bdb7a7684da33d403a055521ac0c4dd7ca670`, `0001..0761`)
- **Candidate vLLM commit**: `f0589cbc9f43678a450a949ff05e1fd18ee902e3` (tree `64782be3f24c8b5269f360d4182e66cda67b3f49`, `0001..0762`, `patchset_sha256` `6e593e645428a33cfeefe9a3cd2dfeb14afc5be52dd1e844cdfa0e7ee0b0c131`)
- **Intended causal variable**: Preserve cohort phase synchronization across concurrent decode streams and bound the high-concurrency verification row-cost discount band (`0762-perf-spec-decode-cohort-atomic-budget-and-burst-coalesce.patch`).

## 1. Problem Statement & Switched `1030` vs `1107` Benchmark Evidence (`TP3..TP8`)

Comparing the recorded switched benchmark manifests (`manifests/benchmarks/2026-10-30-spark-ds41f-1030-switched*.json` vs `manifests/benchmarks/2026-11-07-spark-ds41f-1107-switched*.json`) across all topologies (`TP3..TP8`):

| Topology | Metric | `1030` (`tok/s`) | `1107` (`tok/s`) | Delta (`1107` vs `1030`) |
| :---: | :--- | ---: | ---: | ---: |
| **TP3** (`switched3`) | `prose` `c1` / `c8` / **`c16`** | `44.9` / `298.8` / **`356.7`** | `45.3` / `326.8` / **`402.0`** | `+0.9%` / `+9.4%` / **`+12.7%`** |
| **TP3** (`switched3`) | `code` `c1` / `c8` / **`c16`** | `94.2` / `632.3` / **`867.9`** | `91.8` / `662.2` / **`901.7`** | `-2.5%` / `+4.7%` / **`+3.9%`** |
| **TP4** (`switched`) | `prose` `c1` / `c8` / **`c16`** | `51.0` / `346.6` / **`432.5`** | `52.0` / `377.7` / **`460.8`** | `+2.0%` / `+9.0%` / **`+6.5%`** |
| **TP4** (`switched`) | `code` `c1` / `c8` / **`c16`** | `108.9` / `725.5` / **`1052.1`** | `108.0` / `768.2` / **`1058.5`** | `-0.8%` / `+5.9%` / **`+0.6%`** |
| **TP5** (`switched5`) | `prose` `c1` / `c8` / **`c16`** | `56.5` / `384.4` / **`533.5`** | `58.9` / `438.3` / **`578.0`** | `+4.2%` / `+14.0%` / **`+8.3%`** |
| **TP5** (`switched5`) | `code` `c1` / `c8` / **`c16`** | `115.6` / `818.4` / **`1189.1`** | `124.9` / `893.8` / **`1265.2`** | `+8.0%` / `+9.2%` / **`+6.4%`** |
| **TP6** (`switched6`) | `prose` `c1` / `c8` / **`c16`** | `60.3` / `431.9` / **`707.0`** | `63.6` / `473.1` / **`638.8`** | `+5.5%` / `+9.5%` / **`-9.6%`** |
| **TP6** (`switched6`) | `code` `c1` / `c8` / **`c16`** | `125.9` / `875.7` / **`1233.6`** | `135.4` / `971.6` / **`1369.5`** | `+7.5%` / `+11.0%` / **`+11.0%`** |
| **TP7** (`switched7`) | `prose` `c1` / `c8` / **`c16`** | `63.7` / `451.7` / **`774.2`** | `66.1` / `506.9` / **`700.0`** | `+3.8%` / `+12.2%` / **`-9.6%`** |
| **TP7** (`switched7`) | `code` `c1` / `c8` / **`c16`** | `135.0` / `930.8` / **`1314.3`** | `141.4` / `1027.9` / **`1435.5`** | `+4.7%` / `+10.4%` / **`+9.2%`** |
| **TP8** (`switched8`) | `prose` `c1` / `c8` / **`c16`** | `68.1` / `486.0` / **`860.6`** | `68.8` / `531.6` / **`744.7`** | `+1.0%` / `+9.4%` / **`-13.5%`** |
| **TP8** (`switched8`) | `code` `c1` / `c8` / **`c16`** | `138.9` / `997.4` / **`1448.1`** | `144.6` / `1078.7` / **`1496.5`** | `+4.1%` / `+8.2%` / **`+3.3%`** |

### Per-Request TTFT Skew & Inter-Stream Completion Spread at `prose-c16` (`1030` vs `1107`)

Inspecting the raw per-request arrays (`ttft_ms` and `e2e_ms`) in `manifests/benchmarks/*switched{4,5,6,7,8}.json` reveals why `prose-c16` regressed on `TP6..TP8` while `prose-c8` and `code-c16` improved:

| Topology (`prose-c16`) | Build | `ttft_ms` (`min` -> `max`) | `e2e_ms` (`min` -> `max`, spread) | Cohort Phase State | `tok/s` |
| :---: | :---: | :---: | :---: | :--- | ---: |
| **TP4** | `1030` | `204.2 -> 325.1 ms` | `6765.8 -> 7396.3 ms` (`630.5 ms`) | Shattered (`1 + 15` split) | `432.5` |
| **TP4** | `1107` | `202.3 -> 307.9 ms` | `6470.5 -> 6948.7 ms` (`478.2 ms`) | Shattered (`1 + 15` split) | `460.8` |
| **TP5** | `1030` | `187.5 -> 299.1 ms` | `5412.6 -> 5998.0 ms` (`585.4 ms`) | Shattered (`1 + 15` split) | `533.5` |
| **TP5** | `1107` | `181.2 -> 277.3 ms` | `4611.6 -> 5536.0 ms` (`924.4 ms`) | Shattered (`1 + 15` split) | `578.0` |
| **TP6** | `1030` | `253.7 -> 266.8 ms` (`13.1 ms`) | `4520.9 -> 4526.2 ms` (**`5.3 ms`**) | **Phase-locked (`16` in lockstep)** | **`707.0`** |
| **TP6** | `1107` | `172.3 -> 265.1 ms` (`92.8 ms`) | `3992.2 -> 5009.4 ms` (**`1017.2 ms`**) | **Shattered (`1 + 15` -> 16 phases)** | **`638.8` (`-9.6%`)** |
| **TP7** | `1030` | `238.1 -> 252.5 ms` (`14.4 ms`) | `4128.8 -> 4133.2 ms` (**`4.4 ms`**) | **Phase-locked (`16` in lockstep)** | **`774.2`** |
| **TP7** | `1107` | `164.0 -> 247.3 ms` (`83.3 ms`) | `3902.5 -> 4571.6 ms` (**`669.1 ms`**) | **Shattered (`1 + 15` -> 16 phases)** | **`700.0` (`-9.6%`)** |
| **TP8** | `1030` | `223.8 -> 238.3 ms` (`14.5 ms`) | `3714.0 -> 3718.5 ms` (**`4.5 ms`**) | **Phase-locked (`16` in lockstep)** | **`860.6`** |
| **TP8** | `1107` | `155.3 -> 235.1 ms` (`79.8 ms`) | `3707.2 -> 4297.2 ms` (**`590.0 ms`**) | **Shattered (`1 + 15` -> 16 phases)** | **`744.7` (`-13.5%`)** |

## 2. Root Cause Analysis

Three interacting changes between `1030` and `1107` caused `prose-c16` to lose cohort phase-lock on `TP6..TP8` and over-admit low-yield verification rows:

1. **Idle-Wakeup `1 + 15` Split Prefill Wave (`EngineCoreProc._process_input_queue`)**:
   - When `scripts/bench.py` launches 16 concurrent HTTP streams at the start of `prose-c16`, the first request to cross ZMQ wakes the blocked `EngineCoreProc._process_input_queue` thread (`input_queue.get(block=True)`) and immediately starts a 1-request prefill step (`~75–90 ms` for `246` tokens), while the remaining 15 requests arrive `100–400 us` later and wait for step 2.
   - In `1030` on `TP6..TP8`, `0742`'s uniform linear row cost (`row_cost = 0.035` with no knee discount) assigned `draft_budget = 0` on step 2 when stream 0 was at token `247` (`1` decode token) and streams `1..15` were in their `246`-token prefill. Because stream 0 received `0` drafts on step 2 (`247 -> 248`) while streams `1..15` finished prefill + bonus (`246 -> 247`), and on step 3 stream 0 hit a 0-accept step (`248 -> 249`) while streams `1..15` accepted 2 drafts (`247 -> 250`), all 16 streams accidentally converged onto the same token position and stayed phase-locked for the rest of the 200-token decode (`e2e_ms` spread `4.4–5.3 ms`, activating only `8–16` unique experts/layer out of 256).
2. **Fractional Cohort-Splitting Budgets & GPU vs CPU Tie-Breaking (`0754` + `0746`)**:
   - In `1107`, `0754` introduced a two-slope row-cost discount (`HIGH_ROW_COST_KNEE = 16`, `HIGH_ROW_COST_SCALE = 0.55`) across all `rows > 16`.
   - On step 2 (15 streams in prefill + 1 stream in decode), `num_non_draft_tokens.sum() = 3691 >> 16`, so `0754` discounted stream 0's drafts by `0.55x` and gave stream 0 `3` drafts while streams `1..15` were still in prefill, pushing stream 0 further ahead (`247 -> 249/250`).
   - Even worse, on subsequent steps, `0746`'s online confidence calibration combined with 1-ULP BF16 reduction differences between even and odd batch rows caused `assign_draft_budget` on the GPU to break ties by `req_idx` or 1-ULP confidence differences (`draft_budget` not a multiple of cohort size), giving half of the cohort `k` drafts and the other half `k-1` drafts. Within 3–5 steps, the 15-stream cohort shattered into 16 distinct token positions (`42–55` unique experts/layer out of 256 instead of `8–16`, tripling MoE expert weight memory reads per layer).
3. **Unbounded `0.55x` Row-Cost Discount Above `M = 40` (`0754`)**:
   - In `0754`, `effective_rows = min(rows, 16) + 0.55 * max(rows - 16, 0)` applied the `0.55x` discount all the way to `M = 96`.
   - While rows `17..40` share the first 32-row CTA / MXFP8 block (`block_M = 32`), rows above `40` cross into a second 32-row CTA / MXFP8 block (`M = 48` or `64`). On `prose-c16`, where 2nd and 3rd draft survival is marginal (`~0.31–0.34`), the unbounded `0.55x` discount over-admitted `48–64` rows (`2–3` drafts/req) when `32–40` rows (`1–2` drafts/req) maximizes `tok/s`. On `code-c16`, 2nd and 3rd draft survival is high (`0.50–0.67`), so `64` rows remains optimal even with a steeper slope above `M = 40`.

## 3. Three-Part Fix (`0762-perf-spec-decode-cohort-atomic-budget-and-burst-coalesce.patch`)

1. **Idle-Wakeup Burst Coalescing (`EngineCoreProc._coalesce_idle_burst` in `vllm/v1/engine/core.py`)**:
   - When `EngineCoreProc._process_input_queue` wakes from an idle blocking wait (`blocked_from_idle = True`), it polls `input_queue` with a `300 us` initial window (`idle_wakeup_first_poll_s = 0.0003`) to detect a concurrent burst.
   - Only if a second arrival is observed within `300 us` (meaning a multi-stream burst is in flight; single-stream `c1` exits after `300 us`), it continues draining arrivals up to `max_num_seqs` (`16`) with a `500 us` inter-arrival poll (`idle_wakeup_burst_poll_s = 0.0005`).
   - All 16 streams of a `c16` wave enter step 1 prefill together (`16` streams at `pos = 0`) across **all** topologies (`TP3..TP8`), eliminating the `1 + 15` split wave on `TP3/TP4/TP5` as well as `TP6/TP7/TP8`.
2. **Cohort-Atomic Verification Budgeting, Exact CPU->GPU Capacities, and Phase-Locking Catch-Up (`AdaptiveVerificationManager` in `vllm/v1/worker/gpu/spec_decode/adaptive_verification.py`)**:
   - Groups active sampling streams sharing `(num_computed_tokens, prefill_len, eligible_drafts)` whose per-position draft confidence spread is within `COHORT_CONFIDENCE_TOL = 0.025` (absorbing 1-ULP BF16/FP32 noise across batch rows) into an atomic cohort.
   - Evaluates cohort draft levels as atomic units in the utility maximizer so a synchronized cohort never receives a fractional budget that splits its streams, and copies the exact cohort-atomic `capacities_cpu` array directly to `self._batch_draft_capacity[:num_reqs]` (bypassing `tilelang_inputs.assign_draft_budget`'s `req_idx` tie-breaker and `dead_row_args` per-row pruning when cohort-locked).
   - If a minority lead subset (`len(lead_idxs) < len(largest_idxs)`) is `1 <= gap <= step_steps + 1` tokens ahead of the main cohort with the same `prefill_len`, holds `capacities_cpu[lead_idxs] = 0` (1 base token/step) and caps `capacities_cpu[largest_idxs] = min(k_b, min_gap)` so the trailing cohort catches up without overshooting and merges into a single 16-stream cohort within 1–2 steps.
3. **Bounded `16..40`-Row Coalesced Cost Band (`AdaptiveVerificationManager._step_cost`)**:
   - Replaces the unbounded `0.55x` discount above `16` rows with a piecewise cost curve:
     - `rows <= 16`: slope `1.0`
     - `16 < rows <= 40` (`HIGH_ROW_COST_CEILING = 40`): slope `0.55` (`HIGH_ROW_COST_SCALE = 0.55`)
     - `rows > 40`: slope `0.78` (`HIGH_ROW_COST_ABOVE_SCALE = 0.78`)
   - On `prose-c16`, medium-confidence steps prune from `48 -> 32` rows and high-confidence steps from `64 -> 48` rows, while `code-c16` preserves `64` rows (`3` drafts/req).

## 4. Two-Boot GPU Verification & Screening (`sp6`, `NVIDIA GB10` `sm_121a`)

Executed across two independent container boots (`boot1` and `boot2`) on `sp6` (`spark-ds41f:1107`):

- `experiments/2026-11-08-cohort-sync-verification/receipts/gpu-pytest-boot1.txt`: **`77 passed, 1 skipped in 42.74s`**
- `experiments/2026-11-08-cohort-sync-verification/receipts/gpu-pytest-boot2.txt`: **`77 passed, 1 skipped in 42.30s`**
- `experiments/2026-11-08-cohort-sync-verification/receipts/screen-cohort-sync-boot1.jsonl`
- `experiments/2026-11-08-cohort-sync-verification/receipts/screen-cohort-sync-boot2.jsonl`

### 4.1 Functional & Cohort-Budget Screen (`boot1` & `boot2`)

| Screen Case | `boot1` | `boot2` | Result |
| :--- | :---: | :---: | :--- |
| `prose_c16_medium_step` (`unbounded_rows` -> `bounded_rows`) | `48 -> 32` | `48 -> 32` | Saves second 32-row CTA tile on medium-confidence prose |
| `prose_c16_high_step` (`unbounded_rows` -> `bounded_rows`) | `64 -> 48` | `64 -> 48` | Saves fourth 16-row MMA tile on high-confidence prose |
| `code_c16_high_step` (`bounded_rows`) | `64` | `64` | Preserves full 3 drafts/req (`64` rows) on high-confidence code |
| `phase_lock_1_plus_15_to_16` (`steps_to_merge`, `unique_positions`) | `2 steps` (`1` pos at `251`) | `2 steps` (`1` pos at `251`) | Deterministic 2-step phase-lock convergence under 1-ULP noise |
| `idle_wakeup_burst_coalesce_16` (`admitted_count`) | `16` | `16` | All 16 streams admitted into step 1 prefill together |

### 4.2 Two-Boot 61-Layer `NVIDIA GB10` Step-Cost Screen (`Shattered M=56, 46 experts/layer` vs `Phase-Locked M=40, 14 experts/layer`)

| Topology (`inter_tp`) | `boot1` Shattered (ms) | `boot2` Shattered (ms) | 2-Boot Shattered Mean (ms) | `boot1` Synced (ms) | `boot2` Synced (ms) | 2-Boot Synced Mean (ms) | 2-Boot Speedup |
| :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **TP4** (`512`) | `529.924` | `535.199` | `532.562` | `180.673` | `181.832` | **`181.253`** | **`2.94x`** |
| **TP5** (`448`) | `464.942` | `467.318` | `466.130` | `158.939` | `159.388` | **`159.164`** | **`2.93x`** |
| **TP6** (`384`) | `395.805` | `396.259` | `396.032` | `133.956` | `136.036` | **`134.996`** | **`2.93x`** |
| **TP7** (`320`) | `331.303` | `332.768` | `332.036` | `111.669` | `112.956` | **`112.313`** | **`2.96x`** |
| **TP8** (`256`) | `263.951` | `263.322` | `263.637` | `90.762` | `95.592` | **`93.177`** | **`2.83x`** |
