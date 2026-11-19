# Experiment: Multi-Regime MMA Tile Coalescing, 64-Row Wave-Saturation Knee, and Context-Band Pricing in `AdaptiveVerificationManager._step_cost`

## Metadata

- **Date**: 2026-11-08 (updated 2026-11-10)
- **Base deployment commit**: `cc20ed18be1f7cbf5c6c53cdee945f7cc77d5fc5` (`spark-ds41f-1107`)
- **Base vLLM patch head**: `2756478ba1fd70aa4b700e7e467c46960c9888fb` (`0761`, tree `ed1bdb7a7684da33d403a055521ac0c4dd7ca670`)
- **Candidate vLLM patch**: `experiments/2026-11-08-adaptive-verify-graph-context-cost/vllm/0762-perf-spec-decode-adaptive-verify-graph-context-cost.patch`
- **Candidate vLLM patch head**: `e87d6e88999534d69bb827f03fdd4e1d54022a9f` (tree `1c220eb5c0a78d6b8cf5d6706fecb26c2c624c66`, `642` active patches)
- **Hardware**: `sp5` and `sp8` (`NVIDIA GB10`, 48 SMs, `sm_121a`, container `spark-ds41f:1107`)
- **Causal variable**: Replace the unbounded two-segment (`<= 16` at `1.0x`, `> 16` at `0.55x`) step-cost model in `AdaptiveVerificationManager._step_cost` (`vllm/v1/worker/gpu/spec_decode/adaptive_verification.py`) with a deterministic, rank-invariant multi-regime cost model calibrated to the `tp4-ring` 4-boot engine grid (`COALESCED_ROW_COST_KNEE = 32` at `0.28x` for `num_reqs >= 10`, `SATURATED_ROW_COST_KNEE = 64` at `0.28x` with `SECOND_WAVE_STEP_ROWS = 6.714`, `SINGLE_REQ_TILE_STEP_ROWS = 1.5`, and `CONTEXT_ROW_COST_KNEE = 262144` with heterogeneous mixed-context tie-breaking).

---

## Motivation & `tp4-ring` Engine Grid Calibration

In `spark-ds41f-1107` (following patches `0715`, `0729`, and `0742`), `AdaptiveVerificationManager._step_cost` priced verification rows `R` as:

$$\text{cost}(R) = 1.0 + \text{row\_cost} \cdot \left(\min(R, 16) + 0.55 \cdot \max(R - 16, 0)\right)$$

Least-squares calibration against the 4-boot `tp4-ring` engine grid (`a_ms = 15.059`, `per_request_ms = 0.034`, `per_row_le16_ms = 0.241`, `per_row_17_64_ms = 0.270`, `jump_above_64_ms = 3.301`, `per_row_above_64_ms = 0.136`, RMSE `0.322 ms`) in `1107` row-cost units (`row_cost = 0.035`, `unit_ms = 14.049 ms`, `1 row unit = 0.4917 ms`) identifies three distinct physical regimes:

1. **Coalesced `33..64` Row `decode_64` Band at High Concurrency (`num_reqs >= 10`, `c16`)**:
   - At `c16`, the 16 non-draft rows and first 16 draft rows (`R = 1..32`, `decode_32`) already activate the routed MoE expert working set. Additional verification rows in `33 <= R <= 64` (`COALESCED_ROW_COST_KNEE = 32`) pack into the single `decode_64` Tensor Core MMA launch and coalesce onto already-active MoE experts at a marginal cost of **`0.1362 ms / 0.4917 ms = 0.277x ~ 0.28x`** (`COALESCED_ROW_COST_SCALE = 0.28`, `COALESCED_MIN_REQS = 10`).
   - Pricing `33..64` rows at `0.55x` at `c16` under-admitted profitable 2nd/3rd/4th draft slots that already pass the device dead-row threshold (`tau = 0.190`). Keeping `num_reqs < 10` at `0.55x` leaves `c1` and `c8` 100% unchanged.
2. **64-Row Second-Wave Boundary (`R > 64`, `SATURATED_ROW_COST_KNEE = 64`)**:
   - Above `R = 64` on 48 SMs, dense projections cross `DECODE_ROWS = 64` into `prefill_128` (`2,857 -> 5,425 us`) and `IndexerV41Plan` crosses `DECODE_CHUNK = 64` into a second 64-row wave.
   - On the 4-boot `tp4-ring` engine grid, crossing `R = 64` incurs a discrete **`3.301 ms` step jump (`SECOND_WAVE_STEP_ROWS = 6.714` row units)** with a marginal slope of `SATURATED_ROW_COST_SCALE = 0.28`.
   - For a single request (`num_reqs == 1`) with `> 15` copy-draft burst tokens crossing `R = 16 -> 17`, there is no cross-request expert coalescing (`mid_scale = 1.0`, `SINGLE_REQ_TILE_STEP_ROWS = 1.5`).
3. **`tp4-ring` Context Scaling (`CONTEXT_ROW_COST_KNEE = 262144`) & Heterogeneous Cohort Tie-Breaking**:
   - On `tp4-ring`, Sparse-MLA (`top-k = 2048`) and the TP-sharded 8-layer Indexer (`2/8` layers per rank, `64K` index positions per rank at `262K`) keep homogeneous step latency flat from `4K` through `262,144` tokens (`1x3`: `15.956 ms` at 4K vs `15.850 ms` at 262K; `8x24`: `21.488 ms` at 4K vs `21.436 ms` at 262K; `16x48`: `27.868 ms` at 4K vs `27.867 ms` at 64K).
   - Setting `CONTEXT_ROW_COST_KNEE = 262144` (`CONTEXT_REINDEX_SATURATION = 524288`, `CONTEXT_SOURCE_SPAN = 524288`) guarantees zero row trimming on homogeneous `4K..262K` steps (`c1` and `c8` at `32K`, `64K`, `128K` are 100% bit-identical to `1107`), while scaling ultra-deep (`> 262K`) contexts and prioritizing short-context draft slots in heterogeneous mixed-context cohorts (`token_spread > 0`).

---

## Two-Seed 20,000-Step Serving Replay (`receipts/replay-seed{0,1}.json`)

Evaluated using the real `1107` and candidate `adaptive_verification.py` modules over 20,000 steps per concurrency across two independent seeds (`seed=0` on Boot 1, `seed=1` on Boot 2), fitted to served `candidate-47` coverage, survival, and tokens/step counters and priced under:
- **Model A**: 4-boot `tp4-ring` engine grid fit + constant served overhead (`c16`: `97.9 ms`, `c8`: `66.0 ms`, `c1`: `13.1 ms`).
- **Model B**: Served per-batch regression of step time on verified tokens and wasted rows (`40` batches).
- **Model M**: `w * A + (1 - w) * B` calibrated so `row_cost = 0.020` vs `0.035` reproduces the `2026-10-18` serving A/B (`c16`: `+0.79%`, `c8`: `-0.28%`, `c1`: `+0.21%`), bracketed by `[-0.5%, +0.5%]`.

| Concurrency / Regime | Seed | Rows / Step (`1107 -> Cand`) | Verified Tokens / Step (`1107 -> Cand`) | Steps `> 64` Rows (`1107 -> Cand`) | Steps Budget Differs | Model A Throughput | Model M Throughput (`-0.5 / central / +0.5`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`c16`** | `0` | `46.41 -> 52.84` (**`+13.86%`**) | `34.350 -> 35.833` (**`+4.32%`**) | **`8.48% -> 0.16%`** | `99.02%` | **`+3.15%`** | **`+0.14% / +0.91% / +1.68%`** |
| **`c16`** | `1` | `46.50 -> 52.95` (**`+13.89%`**) | `34.341 -> 35.827` (**`+4.33%`**) | **`8.69% -> 0.14%`** | `99.07%` | **`+3.15%`** | **`+0.13% / +0.90% / +1.67%`** |
| **`c8`** | `0` | `26.59 -> 26.59` (`+0.00%`) | `17.870 -> 17.870` (`+0.00%`) | `0.00% -> 0.00%` | `0.00%` | `+0.00%` | `+0.00% / +0.00% / +0.00%` |
| **`c8`** | `1` | `26.45 -> 26.45` (`+0.00%`) | `17.848 -> 17.848` (`+0.00%`) | `0.00% -> 0.00%` | `0.00%` | `+0.00%` | `+0.00% / +0.00% / +0.00%` |
| **`c1` (`150`)** | `0, 1` | `3.73..3.76` (`+0.00%`) | `2.284..2.320` (`+0.00%`) | `0.00% -> 0.00%` | `0.00%` | `+0.00%` | `+0.00% / +0.00% / +0.00%` |
| **`c1-ctx32k`** | `0, 1` | `3.73..3.76` (`+0.00%`) | `2.284..2.320` (`+0.00%`) | `0.00% -> 0.00%` | `0.00%` | `+0.00%` | `+0.00% / +0.00% / +0.00%` |
| **`c1-ctx64k`** | `0, 1` | `3.73..3.76` (`+0.00%`) | `2.284..2.320` (`+0.00%`) | `0.00% -> 0.00%` | `0.00%` | `+0.00%` | `+0.00% / +0.00% / +0.00%` |
| **`c1-ctx128k`** | `0, 1` | `3.73..3.76` (`+0.00%`) | `2.284..2.320` (`+0.00%`) | `0.00% -> 0.00%` | `0.00%` | `+0.00%` | `+0.00% / +0.00% / +0.00%` |

---

## 2-Boot GB10 Kernel & Regime Screen (`receipts/screen-graph-context-cost-boot{1,2}.jsonl`)

| Regime | Rows (`1107 -> Cand`) | Boot 1 Tok/s (`1107 -> Cand`) | Boot 1 Speedup | Boot 2 Tok/s (`1107 -> Cand`) | Boot 2 Speedup | 2-Boot Mean |
|---|---:|---:|---:|---:|---:|---:|
| `c1_copy_burst_plateau_2k` | `20 -> 16` | `535.77 -> 542.28` | **`+1.22%`** | `537.59 -> 543.13` | **`+1.03%`** | **`+1.12%`** |
| `c8_med_accept_{2k,32k,64k,128k}` | `40 -> 40` | identical | `+0.00%` | identical | `+0.00%` | **`+0.00%`** |
| `c8_low_accept_{64k,128k}` | `24 -> 24` | identical | `+0.00%` | identical | `+0.00%` | **`+0.00%`** |
| `c8_high_accept_64k` | `48 -> 48` | `933.94 -> 933.94` | `+0.00%` | `932.48 -> 932.48` | `+0.00%` | **`+0.00%`** |
| `c16_coalesced_decode64_4k` | `48 -> 64` | `891.16 -> 892.29` | **`+0.13%`** | `889.59 -> 890.36` | **`+0.09%`** | **`+0.11%`** |
| `c16_second_wave_saturation_2k` | `80 -> 64` | `1206.26 -> 1351.28` | **`+12.02%`** | `1197.56 -> 1348.38` | **`+12.59%`** | **`+12.31%`** |
| `c16_second_wave_saturation_64k` | `80 -> 64` | `1110.84 -> 1240.03` | **`+11.63%`** | `1103.03 -> 1238.37` | **`+12.27%`** | **`+11.95%`** |
| `c8_ultra_long_1m` | `40 -> 32` | `619.84 -> 653.67` | **`+5.46%`** | `623.08 -> 653.47` | **`+4.88%`** | **`+5.17%`** |
| `c8_heterogeneous_2k_128k` | `36 -> 32` | `653.14 -> 700.87` | **`+7.31%`** | `657.03 -> 700.95` | **`+6.68%`** | **`+7.00%`** |

---

## Verification & Receipts

- **2-Boot GPU Unit Tests**:
  - `receipts/gpu-pytest-boot1.txt`: `45 passed, 1 skipped in 10.28s`
  - `receipts/gpu-pytest-boot2.txt`: `45 passed, 1 skipped in 10.20s`
- **2-Boot GB10 Kernel & Regime Screen**:
  - `receipts/attn-idx-profile-boot1.json`, `receipts/attn-idx-profile-boot2.json`
  - `receipts/screen-graph-context-cost-boot1.jsonl`, `receipts/screen-graph-context-cost-boot2.jsonl`
- **2-Seed 20,000-Step Serving Replay**:
  - `receipts/replay-seed0.json`, `receipts/replay-seed1.json`
