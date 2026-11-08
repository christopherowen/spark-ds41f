# Zero-broadcast speculative confidence sync and lockstep TP sampling seeds (`2026-11-08-spec-decode-zero-broadcast-sync`)

- **Date**: 2026-11-08
- **Base deployment commit**: `cc20ed18b6ca6521b602f09cb3ba326774166d1c` (Candidate `1107`, `experiments/2026-10-26-spark-ds41f-candidate/1107`)
- **Base vLLM commit**: `320021e3c4dbb202d911ab70e2cd924c75420488` (tree `ed1bdb7a7684da33d403a055521ac0c4dd7ca670`, `0001..0761`)
- **Candidate vLLM commit**: `61354efd180ca3f3152e63882f2fca1c958954a9` (tree `962ad153cc8bbd014239fe481577eb4501c8d8b1`, `0001..0762`, `patchset_sha256` `f3f9016e678d5971517bde650d0ee2ce46f75c84744712fc455ee8c8d5de0807`)
- **Intended causal variable**: Eliminate per-step TP collective broadcasts in `AdaptiveVerificationManager.record_confidences` across all speculative drafters (`DSparkSpeculator`, `DraftModelSpeculator`, `DFlashSpeculator`) and lockstep tensor-parallel sampling RNG seeds (`0762-perf-spec-decode-elide-tp-confidence-broadcast-and-lock-sampling-seeds.patch`, `TODO.md` Decode #7).

## 1. Technical Audit & Motivation

### 1.1 Where the per-step broadcast lives and why it existed (`TODO.md` Decode #7)

In `vllm/v1/worker/gpu/spec_decode/adaptive_verification.py` (`AdaptiveVerificationManager.record_confidences`, lines 242–288), each decode step records draft confidence probabilities into `self._confidence_buffer`, shaped `(1 + planned_confidences, max_num_reqs, num_speculative_steps)` in `float32`:
- `128 B` (`1 x 8 x 4` `float32`) for 4 speculative steps without `planned_confidences`;
- `160 B` (`1 x 8 x 5` `float32`) for 5-step DSpark without `planned_confidences`;
- `320 B` (`2 x 8 x 5` `float32`) for 5-step DSpark with `CopyDrafter` `planned_confidences` (the raw `_planned_probs` row consumed by next-step host-side budget planning and the effective `_confidence_probs` row consumed by same-step device-side dead-row pruning).

Every tensor-parallel rank consumes `self._confidence_buffer` in two places:
1. **Same-step device-side dead-row pruning (`dead_row_args`)**: `DSparkSpeculator.propose()` passes `self._confidence_probs` (`_confidence_buffer[0]`) into `dspark_confidence` (`TileLangConfidenceHead.probability`), which writes both the raw confidence probability and the monotonically non-increasing cumulative survival product $S_k = \prod_{j \le k} p_j$, and updates `dead_row_cut` when all active rows fall below `skip_step_threshold`.
2. **Next-step host-side verification budget planning (`get_num_tokens` / `reallocate_drafts`)**: `record_confidences` copies `self._planned_probs` (`_confidence_buffer[-1]`) to pinned host memory (`self._stale_confidences` on `self._copy_stream`). At the start of step $t+1$, every TP rank runs `get_num_tokens` (`_choose_k` / `assign_draft_budget`) on host to choose `draft_budget` (`choose_k`), `scheduled_drafts`, `capacities`, and `total_tokens` (which selects the CUDA graph batch size and collective shape for target verification).

If any two TP ranks disagree on `_confidence_buffer`, they can select different `draft_budget` (`choose_k`), `total_tokens`, or `dead_row_cut`, causing an immediate collective or CUDA graph shape mismatch across ranks.

### 1.2 Audit of cross-rank invariance and three remaining determinism gaps

Patches `0566` (`CudaCommunicator._broadcast_one_shot` via Sparknet one-shot RoCE `all_gather`) and `0568` (`DSparkSpeculator.tp_invariant_confidence = True` when `CudaCommunicator.rank_order_reduce_scatter` is active) removed the broadcast for greedy `DSparkSpeculator`, because:
- Every DSpark layer ends with `CudaCommunicator` one-shot RoCE `all_reduce`, which sums peer shards in canonical rank order (`rank 0 + rank 1 + ...`), leaving `hidden_states` bit-identical across all TP ranks.
- Under `SPARK_DS41F_DSPARK_VOCAB_PARALLEL=1`, `DSparkSpeculator` all-gathers `(max_val, max_idx)` pairs (`8 B/req/rank`) via one-shot RoCE and runs `select_first_argmax`, picking bit-identical greedy draft tokens on every TP rank.
- `TileLangConfidenceHead` (`confidence_head` + `markov_embed`) is replicated identically on all TP ranks (`head_proj`, `token_proj`, `gate_proj`) and runs a fixed single-CTA (`threads=128`) `T.reduce_sum` over `hidden_size`, plus deterministic single-block Triton online logistic calibration (`OnlineAcceptanceEstimator.calibrate_confidence` / `accumulate` / `refit`).

However, inspecting `vllm/v1/worker/gpu/sample/states.py`, `vllm/v1/worker/gpu/sample/sampler.py`, `vllm/v1/worker/gpu/spec_decode/acceptance_estimator.py`, `vllm/v1/worker/gpu/spec_decode/speculator.py`, and `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py` uncovered **three critical gaps**:

1. **Gap 1 — Unseeded `temperature > 0` requests drew seeds from process-global `np.random` (`vllm/v1/worker/gpu/sample/states.py:51-55`)**:
   - In `SamplingStates.add_request`, when `sampling_params.seed is None`, vLLM generated a request seed via:
     ```python
     seed = np.random.randint(_NP_INT64_MIN, _NP_INT64_MAX)
     ```
   - Because this used the process-global `np.random` state rather than a dedicated lockstep generator owned by `SamplingStates`, any rank-asymmetric library call to `np.random` (during startup, warmup, profiling, or request preprocessing) desynchronized `self.seeds.np[req_idx]` across TP ranks.
   - Once `self.seeds.gpu` differed across TP ranks while `DSparkSpeculator.tp_invariant_confidence = True` elided the rank-0 confidence broadcast:
     - `gumbel_sample` (which hashes `(seed, pos)` via counter-based Philox in TileLang) drafted **different tokens on each TP rank** (`mismatch_draft_steps = 1000/1000`);
     - `dspark_confidence` received different `markov_embed(prev_token)` inputs on each TP rank, producing **different `_confidence_probs` on each TP rank** (`max_conf_abs_diff = 0.299` on TP3, `0.516` on TP4);
     - `rejection_sample` accepted different numbers of drafts on each TP rank (`mismatch_sampled_steps = 1000/1000`), causing `OnlineAcceptanceEstimator` IRLS refits to diverge across ranks (`max_slope_abs_diff = 0.505`, `max_intercept_abs_diff = 2.011`);
     - `AdaptiveVerificationManager.get_num_tokens` selected **mismatched `draft_budget` (`choose_k`) and `total_tokens` across TP ranks on `26.3%` of steps (`263/1000`) on TP3 and `23.9%` of steps (`239/1000`) on TP4**!

2. **Gap 2 — `Sampler.sample` dispatched unseeded `top_k` / `top_p` requests to `flashinfer_sample` (`vllm/v1/worker/gpu/sample/sampler.py:442-450`)**:
   - When `any_explicit_seed` was `False` and `top_k` or `top_p` was active, `Sampler.sample` dispatched to `flashinfer_sample`, which ignores `self.sampling_states.seeds.gpu` and draws from rank-local CUDA RNG state.
   - Under `tp_world_size > 1`, routing `top_k` / `top_p` requests through the existing TileLang `apply_top_k_top_p` + counter-based `gumbel_sample` path (which keys its Philox RNG strictly on `(self.sampling_states.seeds.gpu, pos)`) guarantees lockstep token sampling across all TP ranks regardless of whether `sampling_params.seed` was explicitly set by the caller.

3. **Gap 3 — `DraftModelSpeculator` / `DFlashSpeculator` still ran the per-step confidence broadcast and duplicated `logits` scans in `OnlineAcceptanceEstimator.predict`**:
   - `OnlineAcceptanceEstimator` (`vllm/v1/worker/gpu/spec_decode/acceptance_estimator.py`) computes draft confidence from `[batch, vocab]` logits using fixed-grid deterministic Triton reductions (`_local_max_sumexp_kernel`, `_predict_kernel`, `_accumulate_kernel`, `_refit_kernel`).
   - However, `DraftModelSpeculator` (`speculator.py`) and `DFlashSpeculator` (`dflash/speculator.py`) did not expose `tp_invariant_confidence`, so `AdaptiveVerificationManager.record_confidences` still executed a per-step TP broadcast whenever `DraftModelSpeculator` / `DFlashSpeculator` was active.
   - Furthermore, `OnlineAcceptanceEstimator.predict(logits, out)` allocated two fresh GPU tensors (`local_max`, `local_expsum`) via `torch.empty` on every draft step (`10` allocations per 5-step decode), and `DFlashSpeculator` scanned the local vocabulary shard even when vocab-parallel shard statistics `(shard_max, shard_expsum)` were already computed.

## 2. Implementation (`0762-perf-spec-decode-elide-tp-confidence-broadcast-and-lock-sampling-seeds.patch`)

1. **`vllm/v1/worker/gpu/sample/states.py`**:
   - Added a dedicated deterministic generator `self._seed_rng = np.random.default_rng(0)` on `SamplingStates`.
   - In `SamplingStates.add_request`, unseeded requests (`seed is None`) draw `int(self._seed_rng.integers(_NP_INT64_MIN, _NP_INT64_MAX, dtype=np.int64))`. Since every TP rank executes `add_request` in identical scheduler order, every rank assigns bit-identical `self.seeds.np[req_idx]` and `self.seeds.gpu[req_idx]`, completely isolated from process-global `np.random`.
2. **`vllm/v1/worker/gpu/sample/sampler.py`**:
   - Added `Sampler._tp_lockstep_sampling()`, returning `True` whenever tensor-parallel `world_size > 1`.
   - Disabled the `use_flashinfer` sampling branch when `_tp_lockstep_sampling()` is `True`, so `top_k` / `top_p` sampling under `TP > 1` always uses `apply_top_k_top_p` + counter-based `gumbel_sample(logits, temperature, seeds, pos)`.
3. **`vllm/v1/worker/gpu/spec_decode/acceptance_estimator.py`**:
   - Marked `OnlineAcceptanceEstimator.tp_invariant = True`.
   - Added reusable `_scratch_stats(num_reqs, num_splits)` buffers on `OnlineAcceptanceEstimator` so `predict()` performs zero per-step `torch.empty` allocations.
   - Added `OnlineAcceptanceEstimator.predict_from_stats(local_max, local_expsum, out=...)` so vocab-parallel drafters (`DFlashSpeculator`) can feed pre-reduced per-rank `(max, sumexp)` statistics (`[num_reqs, world_size]`) directly into `_predict_kernel` without re-scanning `[num_reqs, vocab_size]`.
4. **`vllm/v1/worker/gpu/spec_decode/speculator.py`, `dflash/speculator.py`, and `adaptive_verification.py`**:
   - Added `@property def tp_invariant_confidence(self) -> bool` on `DraftModelSpeculator` (inheriting to `DFlashSpeculator`), returning `bool(getattr(self.acceptance_estimator, "tp_invariant", False))`.
   - Updated `AdaptiveVerificationManager.record_confidences` to accept either a boolean attribute or a callable `tp_invariant_confidence()`.

## 3. Exact Commands & Workload Identity

### 3.1 Two-Boot Unit Test Suite on `sp8` (`NVIDIA GB10`, `spark-ds41f:1107`)

```bash
for boot in 1 2; do
  flock /tmp/gpu.lock docker run --rm --gpus all --ipc=host --network=host \
    -v /tmp/zero-bcast-vllm/vllm/v1/worker/gpu/sample/states.py:/opt/spark-ds41f/candidate/vllm/vllm/v1/worker/gpu/sample/states.py:ro \
    -v /tmp/zero-bcast-vllm/vllm/v1/worker/gpu/sample/sampler.py:/opt/spark-ds41f/candidate/vllm/vllm/v1/worker/gpu/sample/sampler.py:ro \
    -v /tmp/zero-bcast-vllm/vllm/v1/worker/gpu/spec_decode/acceptance_estimator.py:/opt/spark-ds41f/candidate/vllm/vllm/v1/worker/gpu/spec_decode/acceptance_estimator.py:ro \
    -v /tmp/zero-bcast-vllm/vllm/v1/worker/gpu/spec_decode/adaptive_verification.py:/opt/spark-ds41f/candidate/vllm/vllm/v1/worker/gpu/spec_decode/adaptive_verification.py:ro \
    -v /tmp/zero-bcast-vllm/vllm/v1/worker/gpu/spec_decode/speculator.py:/opt/spark-ds41f/candidate/vllm/vllm/v1/worker/gpu/spec_decode/speculator.py:ro \
    -v /tmp/zero-bcast-vllm/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py:/opt/spark-ds41f/candidate/vllm/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py:ro \
    -v /tmp/zero-bcast-vllm/tests/v1/spec_decode/test_adaptive_verification.py:/opt/spark-ds41f/candidate/vllm/tests/v1/spec_decode/test_adaptive_verification.py:ro \
    -v /tmp/zero-bcast-vllm/tests/v1/spec_decode/test_acceptance_estimator.py:/opt/spark-ds41f/candidate/vllm/tests/v1/spec_decode/test_acceptance_estimator.py:ro \
    --entrypoint pytest spark-ds41f:1107 \
    --noconftest -v \
    /opt/spark-ds41f/candidate/vllm/tests/v1/spec_decode/test_adaptive_verification.py \
    /opt/spark-ds41f/candidate/vllm/tests/v1/spec_decode/test_acceptance_estimator.py \
    /opt/spark-ds41f/candidate/vllm/tests/v1/spec_decode/test_copy_drafts.py \
    /opt/spark-ds41f/candidate/vllm/tests/v1/worker/test_dspark_greedy_markov.py \
    /opt/spark-ds41f/candidate/vllm/tests/distributed/test_roce_policy_no_fallback.py \
    /opt/spark-ds41f/candidate/vllm/tests/kernels/test_deepseek_v41_tilelang_confidence.py \
    | tee /tmp/zero-bcast-receipts/gpu-pytest-boot${boot}.txt
done
```

### 3.2 Two-Boot Multi-Node TP3 (`sp5,sp6,sp7`) & TP4 (`sp5,sp6,sp7,sp8`) 1,000-Step Verification & Latency Benchmark

```bash
python3 experiments/2026-11-08-spec-decode-zero-broadcast-sync/screen_zero_broadcast_sync.py \
  --out-dir /tmp/zero-bcast-receipts \
  --vllm-dir /tmp/zero-bcast-vllm \
  --image spark-ds41f:1107
```

## 4. Measured Results Across Two Boots (`sp5..sp8`, `NVIDIA GB10`, 200GbE RoCE)

### 4.1 Unit & Integration Test Suites (`boot1` & `boot2`)

- `receipts/gpu-pytest-boot1.txt`: **`188 passed, 1 skipped, 0 failed in 25.00s`**
- `receipts/gpu-pytest-boot2.txt`: **`188 passed, 1 skipped, 0 failed in 23.13s`**
- `receipts/screen-zero-broadcast-sync-boot1.jsonl`
- `receipts/screen-zero-broadcast-sync-boot2.jsonl`

### 4.2 1,000-Step Multi-Node TP3 & TP4 Cross-Rank Determinism (`boot1` & `boot2`)

Each 1,000-step run simulates an 8-request batch (`num_speculative_steps = 5`, `CopyDrafter` `planned_confidences` active on half the batch, online `OnlineAcceptanceEstimator` IRLS refitting every 100 steps = `10` refits/run) across real multi-node `CudaCommunicator` (`_roce_policy = True`, Sparknet one-shot RoCE) on `sp5..sp7` (TP3) and `sp5..sp8` (TP4), with `broadcast_calls = 0` (`tp_invariant_confidence = True`) and a rank-asymmetric `np.random` call injected before request arrival:

| Topology | Mode (`1,000` decode steps, `10` online IRLS refits, `broadcast_calls = 0`) | `seeds_equal` | Draft Token Mismatches | Confidence Mismatches (`max_abs_diff`) | **`draft_budget` (`choose_k`) Mismatches** | Rejection Sampled Mismatches | IRLS Estimator Mismatches (`max_slope` / `max_intercept`) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TP3** (`boot1` & `boot2`) | `control_unseeded_global_rng` (`temperature = 0.8`, process-global `np.random` seeds) | `False` | `1000 / 1000` | `1000 / 1000` (`0.298608`) | **`263 / 1000` (`26.3%`)** | `1000 / 1000` | `901 / 1000` (`0.504819` / `1.197604`) |
| **TP3** (`boot1` & `boot2`) | `candidate_greedy` (`temperature = 0.0`, DSpark + TileLang confidence + CopyDrafter) | **`True`** | **`0 / 1000`** | **`0 / 1000` (`0.0`)** | **`0 / 1000` (`0.0%`)** | **`0 / 1000`** | **`0 / 1000` (`0.0` / `0.0`)** |
| **TP3** (`boot1` & `boot2`) | `candidate_sampled_unseeded` (`temperature = 0.8`, `SamplingStates._seed_rng` lockstep) | **`True`** | **`0 / 1000`** | **`0 / 1000` (`0.0`)** | **`0 / 1000` (`0.0%`)** | **`0 / 1000`** | **`0 / 1000` (`0.0` / `0.0`)** |
| **TP3** (`boot1` & `boot2`) | `candidate_dflash_estimator` (`temperature = 0.8`, `predict_from_stats` + `tp_invariant`) | **`True`** | **`0 / 1000`** | **`0 / 1000` (`0.0`)** | **`0 / 1000` (`0.0%`)** | **`0 / 1000`** | **`0 / 1000` (`0.0` / `0.0`)** |
| **TP4** (`boot1` & `boot2`) | `control_unseeded_global_rng` (`temperature = 0.8`, process-global `np.random` seeds) | `False` | `1000 / 1000` | `1000 / 1000` (`0.515928`) | **`239 / 1000` (`23.9%`)** | `1000 / 1000` | `901 / 1000` (`0.364509` / `2.011410`) |
| **TP4** (`boot1` & `boot2`) | `candidate_greedy` (`temperature = 0.0`, DSpark + TileLang confidence + CopyDrafter) | **`True`** | **`0 / 1000`** | **`0 / 1000` (`0.0`)** | **`0 / 1000` (`0.0%`)** | **`0 / 1000`** | **`0 / 1000` (`0.0` / `0.0`)** |
| **TP4** (`boot1` & `boot2`) | `candidate_sampled_unseeded` (`temperature = 0.8`, `SamplingStates._seed_rng` lockstep) | **`True`** | **`0 / 1000`** | **`0 / 1000` (`0.0`)** | **`0 / 1000` (`0.0%`)** | **`0 / 1000`** | **`0 / 1000` (`0.0` / `0.0`)** |
| **TP4** (`boot1` & `boot2`) | `candidate_dflash_estimator` (`temperature = 0.8`, `predict_from_stats` + `tp_invariant`) | **`True`** | **`0 / 1000`** | **`0 / 1000` (`0.0`)** | **`0 / 1000` (`0.0%`)** | **`0 / 1000`** | **`0 / 1000` (`0.0` / `0.0`)** |

### 4.3 Two-Boot `AdaptiveVerificationManager.record_confidences` Per-Step Latency (`TP3` & `TP4`)

1,000 timed steps per arm per boot across `sp5..sp7` (`TP3`) and `sp5..sp8` (`TP4`), including the per-step device-to-pinned-host copy (`self._planned_probs.copy_to_cpu()` on `self._copy_stream`) and event synchronization:

| Topology | Buffer Shape (`bytes`) | Sync Arm | `boot1` Median / Mean / p99 (µs) | `boot2` Median / Mean / p99 (µs) | 2-Boot Median (`mean ± half-range`, µs) | 2-Boot Mean (`mean ± half-range`, µs) | Per-Step Savings vs `nccl` / vs `roce_oneshot` (Median) |
| :---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| **TP3** | `128B_1x8x4` (`128 B`) | `nccl_torch_dist_broadcast` | `49.25 / 65.24 / 393.71` | `49.18 / 67.04 / 458.51` | `49.22 ± 0.03` | `66.14 ± 0.90` | baseline (`nccl`) |
| **TP3** | `128B_1x8x4` (`128 B`) | `roce_oneshot_broadcast` | `66.35 / 74.76 / 237.02` | `65.12 / 77.63 / 745.78` | `65.74 ± 0.62` | `76.20 ± 1.44` | baseline (`roce_oneshot`) |
| **TP3** | `128B_1x8x4` (`128 B`) | **`zero_broadcast_tp_invariant`** | **`32.42 / 34.49 / 53.99`** | **`32.00 / 33.41 / 46.06`** | **`32.21 ± 0.21`** | **`33.95 ± 0.54`** | **`-17.01 µs (-34.6%)` / `-33.53 µs (-51.0%)`** |
| **TP3** | `160B_1x8x5` (`160 B`) | `nccl_torch_dist_broadcast` | `50.56 / 65.77 / 183.33` | `47.58 / 59.81 / 171.37` | `49.07 ± 1.49` | `62.79 ± 2.98` | baseline (`nccl`) |
| **TP3** | `160B_1x8x5` (`160 B`) | `roce_oneshot_broadcast` | `66.37 / 76.18 / 245.92` | `64.61 / 71.74 / 220.79` | `65.49 ± 0.88` | `73.96 ± 2.22` | baseline (`roce_oneshot`) |
| **TP3** | `160B_1x8x5` (`160 B`) | **`zero_broadcast_tp_invariant`** | **`32.77 / 34.96 / 61.07`** | **`31.87 / 33.44 / 50.86`** | **`32.32 ± 0.45`** | **`34.20 ± 0.76`** | **`-16.75 µs (-34.1%)` / `-33.17 µs (-50.6%)`** |
| **TP3** | `320B_2x8x5_planned` (`320 B`) | `nccl_torch_dist_broadcast` | `57.02 / 77.11 / 513.82` | `55.20 / 77.91 / 262.98` | `56.11 ± 0.91` | `77.51 ± 0.40` | baseline (`nccl`) |
| **TP3** | `320B_2x8x5_planned` (`320 B`) | `roce_oneshot_broadcast` | `72.67 / 82.86 / 261.26` | `71.49 / 81.59 / 594.24` | `72.08 ± 0.59` | `82.22 ± 0.63` | baseline (`roce_oneshot`) |
| **TP3** | `320B_2x8x5_planned` (`320 B`) | **`zero_broadcast_tp_invariant`** | **`39.39 / 42.68 / 192.93`** | **`38.05 / 39.45 / 56.78`** | **`38.72 ± 0.67`** | **`41.06 ± 1.62`** | **`-17.39 µs (-31.0%)` / `-33.36 µs (-46.3%)`** |
| **TP4** | `128B_1x8x4` (`128 B`) | `nccl_torch_dist_broadcast` | `49.76 / 66.66 / 370.37` | `50.85 / 76.43 / 241.83` | `50.30 ± 0.54` | `71.55 ± 4.89` | baseline (`nccl`) |
| **TP4** | `128B_1x8x4` (`128 B`) | `roce_oneshot_broadcast` | `65.95 / 82.57 / 635.18` | `66.37 / 94.68 / 703.48` | `66.16 ± 0.21` | `88.63 ± 6.05` | baseline (`roce_oneshot`) |
| **TP4** | `128B_1x8x4` (`128 B`) | **`zero_broadcast_tp_invariant`** | **`32.61 / 35.87 / 153.50`** | **`32.35 / 35.31 / 142.31`** | **`32.48 ± 0.13`** | **`35.59 ± 0.28`** | **`-17.82 µs (-35.4%)` / `-33.68 µs (-50.9%)`** |
| **TP4** | `160B_1x8x5` (`160 B`) | `nccl_torch_dist_broadcast` | `50.40 / 72.26 / 250.86` | `50.18 / 66.46 / 183.76` | `50.29 ± 0.11` | `69.36 ± 2.90` | baseline (`nccl`) |
| **TP4** | `160B_1x8x5` (`160 B`) | `roce_oneshot_broadcast` | `66.18 / 76.98 / 266.31` | `68.32 / 96.37 / 385.34` | `67.25 ± 1.07` | `86.67 ± 9.70` | baseline (`roce_oneshot`) |
| **TP4** | `160B_1x8x5` (`160 B`) | **`zero_broadcast_tp_invariant`** | **`32.61 / 35.12 / 57.74`** | **`32.35 / 35.00 / 133.87`** | **`32.48 ± 0.13`** | **`35.06 ± 0.06`** | **`-17.81 µs (-35.4%)` / `-34.77 µs (-51.7%)`** |
| **TP4** | `320B_2x8x5_planned` (`320 B`) | `nccl_torch_dist_broadcast` | `56.74 / 74.39 / 389.00` | `55.97 / 80.11 / 448.57` | `56.35 ± 0.38` | `77.25 ± 2.86` | baseline (`nccl`) |
| **TP4** | `320B_2x8x5_planned` (`320 B`) | `roce_oneshot_broadcast` | `72.67 / 89.90 / 854.56` | `74.18 / 88.53 / 292.12` | `73.42 ± 0.75` | `89.22 ± 0.68` | baseline (`roce_oneshot`) |
| **TP4** | `320B_2x8x5_planned` (`320 B`) | **`zero_broadcast_tp_invariant`** | **`39.10 / 42.46 / 162.58`** | **`39.01 / 43.35 / 171.34`** | **`39.06 ± 0.05`** | **`42.91 ± 0.44`** | **`-17.30 µs (-30.7%)` / `-34.37 µs (-46.8%)`** |

### 4.4 Two-Boot `OnlineAcceptanceEstimator.predict` vs `predict_from_stats` Per-Draft-Step Latency (`TP3` & `TP4`)

| Topology | Estimator Arm (`[8, 129280]` `bf16` logits) | `boot1` Median / Mean / p90 (µs) | `boot2` Median / Mean / p90 (µs) | 2-Boot Median (`mean ± half-range`, µs) | 5-Step Draft Savings vs Control |
| :---: | :--- | ---: | ---: | ---: | ---: |
| **TP3** | `estimator_predict_alloc_control` (2 `torch.empty` allocs/step) | `29.89 / 30.38 / 31.81` | `29.89 / 30.19 / 32.03` | `29.89 ± 0.00` | baseline |
| **TP3** | `estimator_predict_scratch_candidate` (reusable `_scratch_stats`) | `29.92 / 29.92 / 30.37` | `29.70 / 30.03 / 30.34` | `29.81 ± 0.11` | `-0.4 µs` + `0` allocs/step (`-10` allocs/decode) |
| **TP3** | **`estimator_predict_from_stats_candidate`** (`predict_from_stats`) | **`18.40 / 18.37 / 18.82`** | **`18.27 / 18.11 / 18.78`** | **`18.34 ± 0.06`** | **`-11.55 µs/draft (-38.6%)`, `-57.8 µs / 5 drafts`** |
| **TP4** | `estimator_predict_alloc_control` (2 `torch.empty` allocs/step) | `29.76 / 30.07 / 31.52` | `29.66 / 29.96 / 31.62` | `29.71 ± 0.05` | baseline |
| **TP4** | `estimator_predict_scratch_candidate` (reusable `_scratch_stats`) | `29.60 / 29.75 / 30.08` | `29.60 / 29.70 / 30.02` | `29.60 ± 0.00` | `-0.6 µs` + `0` allocs/step (`-10` allocs/decode) |
| **TP4** | **`estimator_predict_from_stats_candidate`** (`predict_from_stats`) | **`18.30 / 18.27 / 18.72`** | **`18.30 / 18.28 / 18.69`** | **`18.30 ± 0.00`** | **`-11.41 µs/draft (-38.4%)`, `-57.0 µs / 5 drafts`** |

## 5. Memory Observations & Conclusion

- **Memory**: Replaces 2 transient GPU tensor allocations (`local_max`, `local_expsum`) per draft step in `OnlineAcceptanceEstimator.predict()` with a single reusable `(2, max_num_reqs, num_splits)` `float32` scratch tensor (`2 x 8 x 127 x 4 B = 8,128 B` at `max_num_seqs=8`), eliminating `10` allocator calls per 5-step decode iteration.
- **Cross-rank determinism**: Fixes a latent multi-rank verification budget hang under `temperature > 0` speculative decoding where unseeded requests drew divergent `SamplingStates` seeds from process-global `np.random`, causing `AdaptiveVerificationManager` to pick different `draft_budget` (`choose_k`) and `total_tokens` across TP ranks on `26.3%` of steps (`263/1000`) on TP3 and `23.9%` (`239/1000`) on TP4. With `SamplingStates._seed_rng` and `Sampler._tp_lockstep_sampling()`, all TP ranks produce `0/1000` mismatches (`max_abs_diff = 0.0`) across draft tokens, confidence probabilities, online IRLS estimator parameters, rejection-sampled tokens, and verification budgets across both greedy and `temperature > 0` workloads on TP3 and TP4.
- **Per-step latency**: Eliding the per-step `record_confidences` broadcast saves **`16.8–17.8 µs/step` median (`31.9–36.0 µs/step` mean, `-31%` to `-35%` median / `-45%` to `-50%` mean)** vs `torch.distributed.broadcast` and **`33.2–34.8 µs/step` median (`-46%` to `-52%`)** vs `roce_oneshot_broadcast` across `128 B`, `160 B`, and `320 B` buffers on TP3 and TP4, while `OnlineAcceptanceEstimator.predict_from_stats` saves an additional **`11.4–11.6 µs` per draft step (`-38.5%`, `-57 µs` per 5-step decode)**. Ready for review.
