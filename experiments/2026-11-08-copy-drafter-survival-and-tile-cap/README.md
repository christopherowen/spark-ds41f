# CopyDrafter cumulative survival yield, single-request 16-row MMA tile cap, and wide-step dead-row masking

- **Date**: `2026-11-08`
- **Base deployment commit**: `cc20ed18be1f7cbf5c6c53cdee945f7cc77d5fc5` (`spark-ds41f-1107`)
- **Base vLLM series**: `experiments/2026-10-26-spark-ds41f-candidate/1107/vllm/series` (`0001..0761`, 641 active patches, tree `ed1bdb7a7684da33d403a055521ac0c4dd7ca670`)
- **Candidate vLLM patch**: `vllm/0762-perf-spec-decode-copy-drafter-survival-and-tile-cap.patch` (`patch_head` `c8238e2ae7d218a62968a9a7ccc644c09850d050`, `expected_tree` `d5872bd2bdc9a6a9a1e9c0df90c94319cf0b9c9a`, `patchset_sha256` `980ea3982c18c368c51ca3213f316af50d244017e9c8ac596fbb37b897e4034e`)
- **Status**: `screened` (2-boot GPU unit/contract gate + 2-boot GPU kernel and layer-pair screen on `sp7` NVIDIA GB10; full multi-node serving qualification pending shared cluster window)

## Causal mechanisms and rationale

### 1. Cumulative DSpark survival yield in `copy_apply` (`vllm/v1/worker/gpu/spec_decode/tilelang_copy.py`)

In `copy_apply_kernel`, CopyDrafter previously gated replacement of DSpark's `block = 5` draft proposal by comparing `copy_yield = offered * estimate` against the raw sum of conditional per-position DSpark probabilities:

$$\text{dspark\_yield}_{\text{parent}} = \sum_{j=0}^{\text{block}-1} p_j$$

Because sequential speculative verification rejects all positions after the first mismatch, the expected number of accepted DSpark draft tokens is the cumulative survival sum (matching `AdaptiveVerificationManager.get_num_tokens`):

$$\text{dspark\_yield}_{\text{survival}} = \sum_{j=0}^{\text{block}-1} \prod_{i=0}^{j} p_i$$

For a typical DSpark conditional confidence profile `[0.70, 0.65, 0.60, 0.55, 0.50]`, the raw sum is `3.00` whereas the true cumulative survival yield is `0.70 + 0.455 + 0.273 + 0.15015 + 0.075075 = 1.653225`. Under the raw sum, short 2-token (`offered = 2`, `copy_yield = 1.90` at `estimate = 0.95`) and 3-token (`offered = 3`, `copy_yield = 2.85`) copy continuations were rejected in favor of lower-yield DSpark proposals. Accumulating `dspark_survival *= dspark_confidences[r, j]` before `dspark_yield += dspark_survival` aligns `copy_apply` with the host budget's cumulative survival metric at zero additional global memory traffic.

### 2. Single-request (`c1`) 16-row Tensor Core MMA tile boundary in `AdaptiveVerificationManager._step_cost` (`vllm/v1/worker/gpu/spec_decode/adaptive_verification.py`)

`AdaptiveVerificationManager._step_cost` previously applied the high-concurrency coalescing discount (`HIGH_ROW_COST_SCALE = 0.55` above `HIGH_ROW_COST_KNEE = 16` rows) regardless of `num_sampling_requests`. While concurrent requests (`num_sampling_requests > 1`) amortize weight loads across requests above 16 rows, a single decode request (`num_sampling_requests <= 1`) proposing `16` copy drafts (`1 + 16 = 17` verification rows) crosses the 16-row SM120/SM121 Tensor Core MMA tile boundary by a single row, spilling into the `24`-row CUDA graph and launching a second 16-row MMA wave across dense and attention projections for 1 marginal draft token.

Applying `SINGLE_REQ_TILE_SPILL_SCALE = 2.0` above `HIGH_ROW_COST_KNEE = 16` when `num_sampling_requests <= 1` (and assigning `scheduled_drafts[0] = draft_budget` directly in `reallocate_drafts` when `num_reqs == 1`):
- At `row_cost = 0.035` and the initial `0.95` copy prior (`c1`), `utility(15 drafts = 16 rows) = 11.20125 / 1.525 = 7.3451 > utility(16 drafts = 17 rows) = 11.64138 / 1.595 = 7.2987`, capping initial single-request copy jumps at `15` drafts (`16` total verification rows = 100% single-MMA-tile occupancy in the `16`-row CUDA graph).
- Once online copy accuracy reaches `>= 0.97` on a sustained copy streak (`utility(17 rows) = 8.1518 > utility(16 rows) = 8.0785`), or whenever `num_sampling_requests >= 2` (`c2+`), all `16` copy drafts per request are admitted.

### 3. Device dead-row padding mask on shrunk wide copy steps (`adaptive_verification.py`, `model_runner.py`)

Because the scheduler lays out step $t+1$ using `alloc[r]` computed before step $t$'s verification completes (`planned_confidences`), a rejected or partially accepted wide copy step leaves `scheduled_drafts[r] = 16` (`step_steps = 16`) on step $t+1$ while CopyDrafter shrinks its proposal to `offered = 5` drafts (or falls back to DSpark's `5` drafts) and fills trailing columns `5..15` with `drafts = -1` and `confidences = 0.0`.

When `SPARK_DS41F_DSPARK_DEAD_ROWS_TAU == 0` (the promoted default), `AdaptiveVerificationManager.dead_row_args` previously returned `None`, so `combine_sampled_and_draft_tokens` left `is_padding = False` on the trailing `-1` placeholder rows (`rows 6..16`). Those placeholder rows were then routed through all 58 MoE layers (`6` experts/row) as live tokens.

Returning `DeadRows(..., log_tau=_ZERO_CONFIDENCE_LOG_TAU)` (`-1e30`) from `dead_row_args` whenever `step_steps > num_draft_steps` (and declaring `dead_rows = True` in `GPUModelRunner.declare_step_kernels` when `self.copy_drafter is not None`) causes `combine_sampled_and_draft_tokens` to cut at the first zero-confidence `-1` placeholder slot (`log(0) = -inf < -1e30`), marking all trailing `-1` rows `is_padding = True` so routed MoE (`VLLM_MOE_SKIP_PADDING=1`) skips them and `get_verified_draft_counts` clamps verified counts to live rows.

## Exact commands

### 1. 2-boot GPU unit and contract suite on `sp7` (`NVIDIA GB10`)

```bash
flock /tmp/gpu.lock docker run --rm --gpus all --entrypoint pytest \
  -v /tmp/vllm-0762-work/vllm/v1/worker/gpu/spec_decode/tilelang_copy.py:/opt/spark-ds41f/candidate/vllm/vllm/v1/worker/gpu/spec_decode/tilelang_copy.py:ro \
  -v /tmp/vllm-0762-work/vllm/v1/worker/gpu/spec_decode/adaptive_verification.py:/opt/spark-ds41f/candidate/vllm/vllm/v1/worker/gpu/spec_decode/adaptive_verification.py:ro \
  -v /tmp/vllm-0762-work/vllm/v1/worker/gpu/model_runner.py:/opt/spark-ds41f/candidate/vllm/vllm/v1/worker/gpu/model_runner.py:ro \
  -v /tmp/vllm-0762-work/tests/v1/spec_decode/test_copy_drafts.py:/opt/spark-ds41f/candidate/vllm/tests/v1/spec_decode/test_copy_drafts.py:ro \
  -v /tmp/vllm-0762-work/tests/v1/spec_decode/test_adaptive_verification.py:/opt/spark-ds41f/candidate/vllm/tests/v1/spec_decode/test_adaptive_verification.py:ro \
  -v /tmp/vllm-0762-work/tests/v1/worker/test_prepare_kernels.py:/opt/spark-ds41f/candidate/vllm/tests/v1/worker/test_prepare_kernels.py:ro \
  -w /opt/spark-ds41f/candidate/vllm \
  spark-ds41f:1107 \
  -v --noconftest \
    tests/v1/spec_decode/test_copy_drafts.py \
    tests/v1/core/test_copy_drafts_scheduler.py \
    tests/v1/spec_decode/test_adaptive_verification.py \
    tests/v1/worker/test_prepare_kernels.py
```

- **Boot 1** ([`receipts/gpu-pytest-boot1.txt`](receipts/gpu-pytest-boot1.txt)): `108 passed, 1 skipped in 59.95s`
- **Boot 2** ([`receipts/gpu-pytest-boot2.txt`](receipts/gpu-pytest-boot2.txt)): `108 passed, 1 skipped in 59.80s`

### 2. 2-boot GPU kernel & layer-pair screening benchmark on `sp7` (`NVIDIA GB10`)

```bash
flock /tmp/gpu.lock docker run --rm --gpus all --entrypoint python3 \
  -v /tmp/vllm-0762-work/vllm/v1/worker/gpu/spec_decode/tilelang_copy.py:/opt/spark-ds41f/candidate/vllm/vllm/v1/worker/gpu/spec_decode/tilelang_copy.py:ro \
  -v /tmp/vllm-0762-work/vllm/v1/worker/gpu/spec_decode/adaptive_verification.py:/opt/spark-ds41f/candidate/vllm/vllm/v1/worker/gpu/spec_decode/adaptive_verification.py:ro \
  -v /tmp/vllm-0762-work/vllm/v1/worker/gpu/model_runner.py:/opt/spark-ds41f/candidate/vllm/vllm/v1/worker/gpu/model_runner.py:ro \
  -v /tmp/vllm-0762-work/screen_copy_drafter_smooth.py:/tmp/screen_copy_drafter_smooth.py:ro \
  -w /opt/spark-ds41f/candidate/vllm \
  spark-ds41f:1107 \
  /tmp/screen_copy_drafter_smooth.py --boot <boot1|boot2>
```

- **Boot 1** ([`receipts/screen-copy-drafter-boot1.jsonl`](receipts/screen-copy-drafter-boot1.jsonl)): parent first, candidate second (`20` timed CUDA-graph replays per case after 5 warmup replays)
- **Boot 2** ([`receipts/screen-copy-drafter-boot2.jsonl`](receipts/screen-copy-drafter-boot2.jsonl)): candidate first, parent second (`20` timed CUDA-graph replays per case after 5 warmup replays)

## Results

### Gate 1: `copy_apply` cumulative survival yield vs raw conditional sum (`dspark_confs = [0.70, 0.65, 0.60, 0.55, 0.50]`)

| Offered copy tokens | Copy expected yield (`0.95` prior) | Parent DSpark raw sum | Candidate DSpark cumulative survival | Parent admits copy? | Candidate admits copy? |
| --- | --- | --- | --- | --- | --- |
| `2` | `1.90` | `3.00` | `1.653225` | `False` | `True` |
| `3` | `2.85` | `3.00` | `1.653225` | `False` | `True` |
| `4` | `3.80` | `3.00` | `1.653225` | `True` | `True` |
| `8` | `7.60` | `3.00` | `1.653225` | `True` | `True` |

`copy_apply` kernel CUDA-graph replay latency (`num_reqs = 8`):
- `warm`: Boot 1 median `2.672 us` (mean `3.272 us`), Boot 2 median `2.640 us` (mean `2.930 us`)
- `128MiB_wash`: Boot 1 median `3.792 us` (mean `3.906 us`), Boot 2 median `3.808 us` (mean `4.352 us`)

### Gate 2: Single-request (`c1`) 16-row tile cap and GPU `combine_sampled_and_draft_tokens` dead-row mask

| Metric | Parent (`1107`) | Candidate (`0762`) |
| --- | --- | --- |
| `c1` initial copy step admitted tokens (`0.95` prior, `row_cost=0.035`) | `17` (`16` drafts) | `16` (`15` drafts) |
| `c1` initial copy step CUDA graph bucket | `24` rows (2 MMA waves) | `16` rows (1 MMA wave) |
| Shrunk wide step (`step_steps=16`, `5` live drafts + `11` `-1` drafts) live routed rows | `17` unmasked (`102` routes/layer) | `6` live + `10` masked (`36` routes/layer) |
| Device `dead_row_cut` & `get_verified_draft_counts` | `None` (`16` verified) | `5` (`5` verified) |

### 2-boot 58-layer routed MoE + dense projection step timing on `sp7` (`NVIDIA GB10`, `n = 40` paired replays across 2 boots)

| Shape | Scenario | Cache mode | Parent 58-layer step (`boot1` / `boot2`) | Candidate 58-layer step (`boot1` / `boot2`) | Absolute delta (95% CI) | Relative delta (95% CI) |
| --- | --- | --- | --- | --- | --- | --- |
| `TP3` (`inter=768`) | `c1_initial_copy_step` (`17@24` vs `16@16`) | `warm` | `162.69 ms` (`162.69` / `162.69`) | `153.96 ms` (`152.65` / `155.27`) | `-8.73 ± 2.11 ms` | **`-5.37% ± 1.30%`** |
| `TP3` (`inter=768`) | `c1_initial_copy_step` (`17@24` vs `16@16`) | `128MiB_wash` | `161.44 ms` (`161.55` / `161.33`) | `153.42 ms` (`152.66` / `154.18`) | `-8.02 ± 0.91 ms` | **`-4.97% ± 0.56%`** |
| `TP3` (`inter=768`) | `c1_post_reject_shrunk_step` (`17@24` vs `6@16`) | `warm` | `161.28 ms` (`162.91` / `159.66`) | `62.09 ms` (`62.20` / `61.98`) | `-99.19 ± 1.37 ms` | **`-61.50% ± 0.85%`** |
| `TP3` (`inter=768`) | `c1_post_reject_shrunk_step` (`17@24` vs `6@16`) | `128MiB_wash` | `161.64 ms` (`161.73` / `161.55`) | `65.23 ms` (`64.91` / `65.55`) | `-96.41 ± 0.69 ms` | **`-59.64% ± 0.42%`** |
| `TP4` (`inter=576`) | `c1_initial_copy_step` (`17@24` vs `16@16`) | `warm` | `122.03 ms` (`121.60` / `122.46`) | `118.79 ms` (`121.29` / `116.28`) | `-3.24 ± 2.33 ms` | **`-2.66% ± 1.91%`** |
| `TP4` (`inter=576`) | `c1_initial_copy_step` (`17@24` vs `16@16`) | `128MiB_wash` | `126.86 ms` (`129.59` / `124.12`) | `118.12 ms` (`120.77` / `115.46`) | `-8.74 ± 1.82 ms` | **`-6.89% ± 1.44%`** |
| `TP4` (`inter=576`) | `c1_post_reject_shrunk_step` (`17@24` vs `6@16`) | `warm` | `121.91 ms` (`123.58` / `120.23`) | `47.23 ms` (`47.66` / `46.80`) | `-74.68 ± 1.46 ms` | **`-61.26% ± 1.20%`** |
| `TP4` (`inter=576`) | `c1_post_reject_shrunk_step` (`17@24` vs `6@16`) | `128MiB_wash` | `122.73 ms` (`123.42` / `122.04`) | `49.73 ms` (`49.99` / `49.47`) | `-73.00 ± 0.25 ms` | **`-59.48% ± 0.20%`** |

## Errors and memory observations

- Zero test failures, zero runtime errors, and zero out-of-memory conditions across both boots of `pytest` and both boots of `screen_copy_drafter_smooth.py` on `sp7`.
- No new persistent GPU buffers are allocated: `copy_apply_kernel` uses one thread-local `T.float32` register (`dspark_survival`), `_step_cost` is host NumPy arithmetic, and `dead_row_args` reuses the existing `_confidence_probs` and `_dead_row_cut` tensors allocated in `AdaptiveVerificationManager.__init__`.

## Conclusion

All three changes pass the 109-test speculative-decoding and kernel-preparation contract suite across 2 boots (`108 passed, 1 skipped` each) and reduce single-request (`c1`) copy verification step cost on `NVIDIA GB10` by `-4.97% ± 0.56%` (TP3) and `-6.89% ± 1.44%` (TP4) on initial copy steps (`16`-row vs `24`-row CUDA graph) and by `-59.64% ± 0.42%` (TP3) and `-59.48% ± 0.20%` (TP4) on post-rejection shrunk wide steps (`6` live rows + `10` dead-row-masked padding rows in `16`-row graph vs `17` unmasked rows in `24`-row graph).
