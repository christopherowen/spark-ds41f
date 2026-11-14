# Experiment: DSpark Single-Tile Sunk Draft Cost, Warm-Started Confidence Calibration, and Early Draft Halting (`0762`)

## Metadata
- **Date**: 2026-11-08
- **Base deployment commit**: `8e5ea208f` (`main`, promoted `spark-ds41f-1107`)
- **Base vLLM commit / patch head**: `19f2c20ed4d64ae22bd009ceacadd85e2a81e354` + patches `0001..0761` (`2756478b60b90700b9918b457aeb38657da637b5`, tree `ed1bdb7a7684da33d403a055521ac0c4dd7ca670`)
- **Candidate vLLM patch**: `0762-perf-spec-decode-dspark-sunk-cost-and-early-halt.patch` (`patch_head = 6f7bb9efc8f96975906d50f25d94d30844be3724`, `expected_tree = 3ca9fb8092ac65eb796026cbec1269fe4573b0f0`, `patchset_sha256 = 9436c54f66a72a6716e6965bc18a10c4bb7205c5933b3bc181bde65117791ac1`)
- **Hardware node**: `sp2` (`NVIDIA GB10`, SM 12.1, container `spark-ds41f:1107`)
- **Intended causal variable**: Single-tile (`total_rows <= 16`) sunk draft cost and within-tile marginal row pricing in `AdaptiveVerificationManager`, warm-started per-position log-odds calibration in `OnlineAcceptanceEstimator`, and confidence-gated / batch-bounded early sequential Markov draft halting in `DSparkSpeculator`.

## Problem Statement & Algebra

In `manifests/suites/2026-11-07-spark-ds41f-1107-tp4-single-stream/release-table.txt`, `TP=4` single-stream (`c1`) runs at `82.62 tok/s` (`34.72 steps/s` = `28.80 ms/step`, `2.383 tokens/step`, `3.73 rows/step`), with verification coverage by position dropping to `0: 1.000, 1: 0.858, 2: 0.496, 3: 0.248, 4: 0.130` even though conditional acceptance at positions `2, 3, 4` is `48.3%, 51.1%, 56.4%` (`cumulative survival = [0.667, 0.384, 0.185, 0.095, 0.053]`).

Three interacting mechanisms caused `1107` to discard `2.27` of the `5` drafted tokens at `c1` while still executing all `5` sequential Markov steps at high concurrency (`c8`, `c16`):

1. **Omission of Sunk Drafting Cost and Single-Tile Marginal Slope in `AdaptiveVerificationManager._step_cost`**:
   - A decode step takes $T_{\text{step}}(B) = T_{\text{draft}} + T_{\text{verify}}(B)$, so net step throughput is:
     $$\text{Rate}(B) = \frac{\mathbb{E}[\text{tokens}(B)]}{T_{\text{draft}} + T_{\text{verify}}(B)}$$
   - At `c1`, all `1..6` rows (`1` bonus + `5` draft rows) sit inside the first 16-row Tensor Core MMA tile (`HIGH_ROW_COST_KNEE = 16`), where target verification grows from `15.43 ms` (`1x2`) to `16.39 ms` (`1x6`), i.e. `+0.24 ms/row` (`~0.0147` relative to base verify time, `SINGLE_TILE_ROW_COST_SCALE = 0.42` of `row_cost = 0.035`), while `DSpark` block forward + sequential Markov drafting + step bookkeeping (`~12.75 ms`, `DRAFT_COST_RATIO = 0.55`) has already been paid.
   - Omitting `DRAFT_COST_RATIO = 0.55` from the denominator of `utilities = expected_tokens / _step_cost(rows)` and charging `row_cost = 0.035` per row caused `AdaptiveVerificationManager` to require marginal survival `> 0.075` (and `effective_dead_rows_tau = 0.063`) instead of the true single-tile break-even:
     $$\tau_{\text{single-tile}} = \frac{2 \cdot r_{\text{eff}}}{(1 + c_{\text{draft}}) + 3 \cdot r_{\text{eff}}} \approx 0.0184$$
     discarding already-drafted tokens at positions `2, 3, 4` whose marginal yield (`0.0535..0.1855 tokens / 0.24 ms = 0.22..0.77 tok/ms`) far exceeds the step rate (`0.083 tok/ms`).

2. **Cold-Boot `OnlineAcceptanceEstimator` Uncalibrated Window (`REFIT_INTERVAL = 100`)**:
   - `OnlineAcceptanceEstimator.for_confidence_head` initialized `intercepts = 0` (`_refits = 0`), leaving `calibrated_confidence = False` for the first `100` observed decode steps (`dead_rows_tau = 0.20`), starving positions `2..4` at `c1` and mis-calibrating multi-request budgets during cold start.
   - Warm-starting `reset_for_confidence_head(warm_start=True)` with empirical per-position log-odds intercepts `_WARM_CONFIDENCE_HEAD_INTERCEPTS = (-0.52, -0.64, -0.96, -0.85, -0.74)` and exposing `is_calibrated = True` from step 0 eliminates cold-boot starvation while still adapting online via IRLS.
   - `OnlineAcceptanceEstimator.calibrate_confidence_probs` (both the Triton `_calibrate_confidence_kernel` and CPU fallback) now preserves exact `0.0` confidence for halted/unproposed slots (`raw_prob <= 0.0`) without overwriting persistent `features`/`predictions`.

3. **Batch-Bounded & Confidence-Gated Early Halting in `DSparkSpeculator._sample_sequential` / `_sample_sequential_topk`**:
   - `DSparkSpeculator.max_active_draft_steps` bounds the sequential Markov loop to `adaptive_verification.max_draft_steps_for_batch(num_reqs)`, which caps multi-tile batches at the 16-row Tensor Core MMA tile knee (`c1/c2 -> 5`, `c8 -> 4`, `c16 -> 3`), skipping tail Markov bias GEMMs, shard argmax reductions, and cross-node TP `all_gather` collectives for positions that have `0.0%` verification coverage at `c8` (`pos 4`) and `c16` (`pos 3, 4`).
   - Within the active loop (`i >= 1`), when `early_halt_tau > 0` and all requests in the batch have cumulative calibrated survival below `early_halt_tau`, `DSparkSpeculator` halts before step `i`'s Markov bias projection and TP `all_gather`, zero-filling `draft_tokens[:, halted_steps:]` and `draft_token_confidence_probs[:, halted_steps:]` deterministically across TP ranks.

## Exact Commands

### 2-Boot GPU Unit Test Suite (`sp2`)
```bash
for boot in 1 2; do
  flock /tmp/gpu.lock -c "docker run --rm --gpus all --ipc=host --entrypoint pytest \
    -v /home/chris/projects/spark-ds41f/cache:/cache:rw \
    -v /tmp/vllm-1107/vllm/v1/worker/gpu/spec_decode/acceptance_estimator.py:/opt/spark-ds41f/candidate/vllm/vllm/v1/worker/gpu/spec_decode/acceptance_estimator.py:ro \
    -v /tmp/vllm-1107/vllm/v1/worker/gpu/spec_decode/adaptive_verification.py:/opt/spark-ds41f/candidate/vllm/vllm/v1/worker/gpu/spec_decode/adaptive_verification.py:ro \
    -v /tmp/vllm-1107/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py:/opt/spark-ds41f/candidate/vllm/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py:ro \
    -v /tmp/vllm-1107/vllm/v1/worker/gpu/model_runner.py:/opt/spark-ds41f/candidate/vllm/vllm/v1/worker/gpu/model_runner.py:ro \
    -v /tmp/vllm-1107/tests/v1/spec_decode/test_acceptance_estimator.py:/opt/spark-ds41f/candidate/vllm/tests/v1/spec_decode/test_acceptance_estimator.py:ro \
    -v /tmp/vllm-1107/tests/v1/spec_decode/test_adaptive_verification.py:/opt/spark-ds41f/candidate/vllm/tests/v1/spec_decode/test_adaptive_verification.py:ro \
    -v /tmp/vllm-1107/tests/v1/worker/test_dspark_greedy_markov.py:/opt/spark-ds41f/candidate/vllm/tests/v1/worker/test_dspark_greedy_markov.py:ro \
    -e TILELANG_CACHE_DIR=/cache/spark-ds41f/jit/tilelang \
    spark-ds41f:1107 --noconftest -q \
      /opt/spark-ds41f/candidate/vllm/tests/v1/spec_decode/test_acceptance_estimator.py \
      /opt/spark-ds41f/candidate/vllm/tests/v1/spec_decode/test_adaptive_verification.py \
      /opt/spark-ds41f/candidate/vllm/tests/v1/worker/test_dspark_greedy_markov.py"
done
```
- **Boot 1 (`receipts/gpu-pytest-boot1.txt`)**: `117 passed, 1 skipped in 15.59s`
- **Boot 2 (`receipts/gpu-pytest-boot2.txt`)**: `117 passed, 1 skipped in 15.21s`

### 2-Boot GPU Benchmark (`sp2`)
```bash
for boot in 1 2; do
  flock /tmp/gpu.lock -c "docker run --rm --gpus all --ipc=host --entrypoint python3 \
    -v /home/chris/projects/spark-ds41f/cache:/cache:rw \
    -v /tmp/vllm-1107/vllm/v1/worker/gpu/spec_decode/acceptance_estimator.py:/opt/spark-ds41f/candidate/vllm/vllm/v1/worker/gpu/spec_decode/acceptance_estimator.py:ro \
    -v /tmp/vllm-1107/vllm/v1/worker/gpu/spec_decode/adaptive_verification.py:/opt/spark-ds41f/candidate/vllm/vllm/v1/worker/gpu/spec_decode/adaptive_verification.py:ro \
    -v /tmp/vllm-1107/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py:/opt/spark-ds41f/candidate/vllm/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py:ro \
    -v /tmp/vllm-1107/vllm/v1/worker/gpu/model_runner.py:/opt/spark-ds41f/candidate/vllm/vllm/v1/worker/gpu/model_runner.py:ro \
    -v /tmp:/host_tmp:rw \
    -e TILELANG_CACHE_DIR=/cache/spark-ds41f/jit/tilelang \
    spark-ds41f:1107 /host_tmp/screen_dspark_sunk_cost_and_early_halt.py \
      --boot ${boot} \
      --output /host_tmp/screen-dspark-sunk-cost-and-early-halt-boot${boot}.json"
done
```

## 2-Boot Benchmark Results (`sp2`, `NVIDIA GB10`)

### 1. Sequential DSpark Markov + TileLang Confidence + Triton Calibration Kernel Latency (`us` on `GB10`)

| Topology | Batch | `k=1` (`us`) | `k=2` (`us`) | `k=3` (`us`) | `k=4` (`us`) | `k=5` (`us`) |
|---|---|---:|---:|---:|---:|---:|
| `TP=4` (Boot 1) | `c1` | `87.15` | `139.70` | `188.01` | `237.27` | `279.59` |
| `TP=4` (Boot 1) | `c8` | `88.12` | `138.54` | `200.79` | `251.73` | `303.48` |
| `TP=4` (Boot 1) | `c16` | `92.06` | `149.49` | `210.81` | `280.51` | `346.23` |
| `TP=3` (Boot 1) | `c1` | `83.84` | `139.12` | `191.20` | `241.62` | `306.52` |
| `TP=3` (Boot 1) | `c8` | `105.92` | `204.64` | `279.06` | `376.25` | `444.33` |
| `TP=3` (Boot 1) | `c16` | `114.22` | `241.68` | `344.71` | `467.83` | `562.32` |

### 2. End-to-End Speculative Decode Performance (`1107` Baseline vs. `0762` Candidate, 2,000 Steps / Arm)

| Shape | Conc. | Boot | Base `tok/s` | Cand `tok/s` | $\Delta$ `tok/s` | Cold-100 $\Delta$ `tok/s` | Base $1+\tau$ | Cand $1+\tau$ | Base `ms/step` | Cand `ms/step` | Base `rows/step` | Cand `rows/step` | Base Draft Steps | Cand Draft Steps |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `TP=4` | `c1` | Boot 1 | `78.66` | `83.55` | **`+6.22%`** | **`+9.61%`** | `2.267` | `2.391` | `28.826` | `28.618` | `4.12` | `5.03` | `5.00` | `4.03` |
| `TP=4` | `c1` | Boot 2 | `78.66` | `83.55` | **`+6.22%`** | **`+9.61%`** | `2.267` | `2.391` | `28.826` | `28.619` | `4.12` | `5.03` | `5.00` | `4.03` |
| `TP=4` | `c8` | Boot 1 | `184.00` | `185.82` | **`+0.99%`** | **`+4.25%`** | `2.081` | `2.084` | `86.170` | `85.474` | `24.40` | `23.82` | `5.00` | `3.98` |
| `TP=4` | `c8` | Boot 2 | `184.00` | `185.81` | **`+0.98%`** | **`+4.24%`** | `2.081` | `2.084` | `86.170` | `85.481` | `24.40` | `23.82` | `5.00` | `3.98` |
| `TP=4` | `c16` | Boot 1 | `230.71` | `233.50` | **`+1.21%`** | **`+6.24%`** | `1.980` | `1.976` | `123.568` | `121.882` | `39.18` | `37.02` | `5.00` | `3.00` |
| `TP=4` | `c16` | Boot 2 | `230.71` | `233.51` | **`+1.21%`** | **`+6.24%`** | `1.980` | `1.976` | `123.568` | `121.880` | `39.18` | `37.02` | `5.00` | `3.00` |
| `TP=3` | `c1` | Boot 1 | `60.64` | `64.35` | **`+6.12%`** | **`+12.82%`** | `2.180` | `2.305` | `35.957` | `35.822` | `3.92` | `4.87` | `5.00` | `3.87` |
| `TP=3` | `c1` | Boot 2 | `60.64` | `64.35` | **`+6.12%`** | **`+12.84%`** | `2.180` | `2.305` | `35.957` | `35.819` | `3.92` | `4.87` | `5.00` | `3.87` |
| `TP=3` | `c8` | Boot 1 | `146.66` | `147.93` | **`+0.87%`** | **`+3.52%`** | `2.050` | `2.052` | `104.119` | `103.349` | `23.57` | `22.95` | `5.00` | `3.97` |
| `TP=3` | `c8` | Boot 2 | `146.66` | `147.96` | **`+0.89%`** | **`+3.54%`** | `2.050` | `2.052` | `104.119` | `103.325` | `23.57` | `22.95` | `5.00` | `3.97` |
| `TP=3` | `c16` | Boot 1 | `184.99` | `187.07` | **`+1.12%`** | **`+6.19%`** | `1.951` | `1.949` | `148.741` | `146.890` | `37.95` | `35.98` | `5.00` | `3.00` |
| `TP=3` | `c16` | Boot 2 | `184.99` | `187.11` | **`+1.15%`** | **`+6.21%`** | `1.951` | `1.949` | `148.741` | `146.860` | `37.95` | `35.98` | `5.00` | `3.00` |

### 3. Verified Coverage by Draft Position (`0..4`) & Lossless Verification (`KL = 0`)

| Shape | Conc. | Arm | Pos `0` | Pos `1` | Pos `2` | Pos `3` | Pos `4` | Cold-100 Coverage (`0..4`) | `KL` Div. | TP Rank Agreement |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| `TP=4` | `c1` | `1107` Base | `1.000` | `0.912` | `0.608` | `0.372` | `0.229` | `[1.000, 0.890, 0.300, 0.140, 0.000]` | `0.0` | `100.0%` |
| `TP=4` | `c1` | `0762` Cand | `1.000` | `1.000` | `0.856` | `0.672` | `0.504` | `[1.000, 1.000, 0.890, 0.650, 0.440]` | `0.0` | `100.0%` |
| `TP=4` | `c8` | `1107` Base | `0.999` | `0.658` | `0.269` | `0.164` | `0.112` | `[0.990, 0.436, 0.115, 0.000, 0.000]` | `0.0` | `100.0%` |
| `TP=4` | `c8` | `0762` Cand | `1.000` | `0.673` | `0.278` | `0.175` | `0.000` | `[1.000, 0.664, 0.282, 0.171, 0.000]` | `0.0` | `100.0%` |
| `TP=4` | `c16` | `1107` Base | `0.972` | `0.412` | `0.159` | `0.114` | `0.064` | `[0.931, 0.179, 0.000, 0.000, 0.000]` | `0.0` | `100.0%` |
| `TP=4` | `c16` | `0762` Cand | `0.976` | `0.428` | `0.168` | `0.000` | `0.000` | `[0.974, 0.421, 0.152, 0.000, 0.000]` | `0.0` | `100.0%` |
| `TP=3` | `c1` | `1107` Base | `1.000` | `0.880` | `0.523` | `0.292` | `0.219` | `[1.000, 0.830, 0.320, 0.090, 0.000]` | `0.0` | `100.0%` |
| `TP=3` | `c1` | `0762` Cand | `1.000` | `1.000` | `0.857` | `0.548` | `0.467` | `[1.000, 1.000, 0.850, 0.610, 0.540]` | `0.0` | `100.0%` |
| `TP=3` | `c8` | `1107` Base | `0.999` | `0.632` | `0.261` | `0.156` | `0.116` | `[0.993, 0.416, 0.120, 0.000, 0.000]` | `0.0` | `100.0%` |
| `TP=3` | `c8` | `0762` Cand | `0.999` | `0.647` | `0.269` | `0.166` | `0.000` | `[0.998, 0.631, 0.254, 0.171, 0.000]` | `0.0` | `100.0%` |
| `TP=3` | `c16` | `1107` Base | `0.966` | `0.404` | `0.154` | `0.111` | `0.056` | `[0.922, 0.194, 0.000, 0.000, 0.000]` | `0.0` | `100.0%` |
| `TP=3` | `c16` | `0762` Cand | `0.970` | `0.418` | `0.163` | `0.000` | `0.000` | `[0.978, 0.417, 0.168, 0.000, 0.000]` | `0.0` | `100.0%` |

## Errors & Memory Observations
- **Memory footprint**: Zero additional persistent GPU allocations on the model or drafter paths; `_step_confidence_prob` reuses a `[max_num_reqs]` `float32` view (`<= 1 KiB`).
- **Errors**: None. Both 2-boot GPU `pytest` runs (`117 passed, 1 skipped`) and 2-boot GPU benchmark runs completed with exit code `0`.

## Conclusion
- Including sunk drafting cost (`DRAFT_COST_RATIO = 0.55`) and the within-16-row Tensor Core MMA tile slope (`SINGLE_TILE_ROW_COST_SCALE = 0.42`) in `AdaptiveVerificationManager._step_cost` and `effective_dead_rows_tau`, warm-starting `OnlineAcceptanceEstimator` with `_WARM_CONFIDENCE_HEAD_INTERCEPTS`, and bounding/early-halting `DSparkSpeculator`'s sequential Markov loop increases single-stream `c1` throughput by **`+6.22%`** on `TP=4` (`+9.61%` during the first 100 cold-start steps) and **`+6.12%`** on `TP=3` (`+12.83%` cold-100), while reducing `c8` and `c16` step latency via early sequential Markov halting (`+0.99%` and `+1.21%` net `tok/s` on `TP=4`; `+0.88%` and `+1.14%` on `TP=3`) with `KL = 0` and 100% TP rank determinism.
