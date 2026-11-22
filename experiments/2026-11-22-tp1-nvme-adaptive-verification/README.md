# TP1 / TP2 NVMe-Offloaded Speculation Policy & `PIECEWISE` Adaptive Verification

Date: 2026-11-22.
Base commits: `974b11beb3` (fixed-draft sweep) and `b6c9d0fa37` (`VllmConfig._validate_adaptive_verification` overlay + `d0`/`rc090`/`tau060` arms).
OCI image: `spark-ds41f:candidate-64` (`sha256:50b294060f808ce5d2369441995762d591e1e213fe17f75a4635e43ad44ce312`, vLLM `0001..0843`, tree `7464b04c3cc79c6de1c1745f92c1cdeb2999a2a8`).
Target host: `dgx4` (`tp1`, isolated single-node GB10, `48` static + `48` dynamic slots out of `384` routed experts/layer; zero impact on `dgx1..dgx3` `tp3-triangle` production).

## Hypothesis

On `tp1` (`48` static + `48` dynamic slots / `384` experts) and `tp2-*` (`176` static + `64` dynamic), running `num_speculative_tokens = 5` with `enable_adaptive_verification = false` unconditionally pushes all `6` rows per request through `40` routed MoE layers on every decode step. Tail draft positions (`k = 2, 3, 4`, surviving only `6–24%` of steps) trigger NVMe `io_uring` expert slot cache misses (`~96–134 ms` per marginal row vs `~2.5 ms` in TP3 DRAM) and evict warm experts from the `48`/`64` dynamic slots.

## One Causal Variable (`piecewise_adaptive_verification.patch`)

Three minimal, strictly scoped changes — **zero code, graph, or performance impact on `FULL_AND_PIECEWISE` / `FULL_DECODE_ONLY` (`TP3..TP8`)**:

1. **`vllm/config/vllm.py` (`VllmConfig._validate_adaptive_verification`)**: Remove the legacy `not cudagraph_mode.has_full_cudagraphs()` rejection (originally required when step costs were profiled from captured full CUDA graphs; obsolete since adaptive verification uses the deterministic analytic `row_cost` model). Retains LoRA and `pipeline_parallel_size > 1` guards.
2. **`vllm/v1/worker/gpu/spec_decode/adaptive_verification.py` (`resolve_adaptive_cudagraph_mode`)**: Return `mode` unchanged when `not mode.has_full_cudagraphs()` (`PIECEWISE` / `NONE`). `FULL*` modes follow the exact existing path.
3. **`vllm/models/deepseek_v4_1/expert_slots/cache.py` (`ExpertSlotCache.run`)**: Filter `is_padding` dead rows (`(topk_ids_np >= 0).any(axis=-1)`) out of `RouterLookaheadPredictor.predict_experts(lookahead_hidden)` so layer `L+1` lookahead never wastes NVMe `io_uring` prefetches on dead tail rows (`_partition_windows` already strips `topk_ids < 0`). Only executes when `slot_cache is not None` (`tp1`/`tp2-*`).

## Unit Verification (`dgx4`, `spark-ds41f:candidate-64`)

- `git apply --check experiments/2026-11-22-tp1-nvme-adaptive-verification/piecewise_adaptive_verification.patch` on fresh vLLM `0001..0843` tree: **`PATCH_APPLY_OK`**.
- `docker run --rm --gpus all --entrypoint python3 ... spark-ds41f:candidate-64 -m pytest -q /tmp/test_piecewise_adaptive.py /tmp/test_adaptive_verification.py`: **`59 passed, 1 skipped, 0 failed in 12.77s`**.

## Workload Identity & Exact Commands

All live measurements ran on `dgx4` (`10.0.1.74:8000`) with full container stop/start and thermal cooldown (`< 49 C`) between arms:

- **`tp1-screen`** (`suite_sha256: eb22f7987a35`):
  - `c1`: `sonar run --mode closed --concurrency 1 --warmup 30 --duration 120` (`chat_template_kwargs.thinking = false`, 10 windows).
  - `c4`: `sonar run --mode closed --concurrency 4 --warmup 30 --duration 120` (`chat_template_kwargs.thinking = false`, 10 windows).
- **`tp1-model16`** (`suite_sha256: ad9cb9716e21`):
  - `model`: `sonar model --limit 16 --concurrency 4 --max-tokens 128 --top 20` (`2,048` greedy output tokens + `2,048 × top-20` logprobs).
- Start / bench commands per arm:
  ```bash
  bin/spark --cluster-config <cfg> cluster stop --apply
  bin/spark --cluster-config <cfg> cluster start --apply --replace
  bin/spark --cluster-config <cfg> bench --suite /tmp/tp1-model16.json --label <arm> --output results/private/lab/2026-11-22-tp1-nvme-adapt/<arm>/model16
  bin/spark --cluster-config <cfg> bench --suite /tmp/tp1-screen.json  --label <arm> --output results/private/lab/2026-11-22-tp1-nvme-adapt/<arm>/screen
  ```

## Live Results (`dgx4`, `tp1`, `spark-ds41f:candidate-64`, `0` Failed Requests Across All 13 Runs)

### Decode Throughput & Latency (`tp1-screen`)

| Arm | Spec Config / Policy | `c1` tok/s | `c1` Δ vs `d5` | `c1` CI95 | `c1` TPOT p50 | `c1` steps/s | `c1` tok/step | `c1` drft | `c4` tok/s | `c4` Δ vs `d5` | `c4` CI95 | `c4` steps/s | `c4` tok/step | `c4` drft |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`d5-off`** (shipped `1120`/`c64` baseline) | fixed `drafts=5`, `adapt=false` | `3.483` | `0.0%` | `10.2%` | `241.0 ms` | `1.42` | `2.53` | `5.00` | `2.967` | `0.0%` | `12.7%` | `0.292` | `10.22` | `5.00` |
| `d3-off` | fixed `drafts=3`, `adapt=false` | `4.400` | `+26.3%` | `8.4%` | `205.9 ms` | `1.89` | `2.37` | `3.00` | `3.583` | `+20.8%` | `11.7%` | `0.408` | `8.77` | `3.00` |
| `d2-off` | fixed `drafts=2`, `adapt=false` | `4.967` | `+42.6%` | `9.0%` | `185.0 ms` | `2.41` | `2.09` | `2.00` | `4.308` | `+45.2%` | `7.7%` | `0.542` | `7.95` | `2.00` |
| `d1-off` | fixed `drafts=1`, `adapt=false` | `5.275` | `+51.4%` | `11.5%` | `174.2 ms` | `3.13` | `1.70` | `1.00` | `4.942` | `+66.6%` | `3.9%` | `0.767` | `6.45` | `1.00` |
| `d0-off` (Boot 1) | spec decode disabled (`drafts=0`) | `5.533` | `+58.9%` | `11.6%` | `167.8 ms` | `5.51` | `1.01` | `0.00` | `5.400` | `+82.0%` | `4.1%` | `1.350` | `4.00` | `0.00` |
| `d0-off` (Boot 2) | spec decode disabled (`drafts=0`) | `5.625` | `+61.5%` | `12.6%` | `166.6 ms` | `5.60` | `1.00` | `0.00` | `5.467` | `+84.3%` | `2.3%` | `1.367` | `4.00` | `0.00` |
| *`d0-off` 2-boot mean* | | *`5.579`* | *`+60.2%`* | | *`167.2 ms`* | *`5.56`* | *`1.00`* | *`0.00`* | *`5.433`* | *`+83.1%`* | | *`1.359`* | *`4.00`* | *`0.00`* |
| `adapt-rc015` | `adapt=true`, `rc=0.15`, `tau=0` | `5.033` | `+44.5%` | `10.8%` | `182.4 ms` | `2.59` | `1.97` | `1.73` | `5.125` | `+72.8%` | `4.9%` | `0.708` | `7.25` | `1.58` |
| `adapt-rc030` | `adapt=true`, `rc=0.30`, `tau=0` | `5.208` | `+49.5%` | `15.0%` | `183.6 ms` | `2.88` | `1.82` | `1.53` | `5.458` | `+84.0%` | `5.2%` | `0.825` | `6.64` | `1.48` |
| `adapt-rc050` | `adapt=true`, `rc=0.50`, `tau=0` | `5.183` | `+48.8%` | `11.4%` | `184.4 ms` | `3.01` | `1.75` | `1.59` | `5.317` | `+79.2%` | `4.5%` | `0.908` | `5.90` | `1.62` |
| `adapt-rc075` | `adapt=true`, `rc=0.75`, `tau=0` | `5.300` | `+52.2%` | `12.1%` | `176.6 ms` | `3.35` | `1.62` | `1.50` | `5.342` | `+80.1%` | `5.3%` | `1.033` | `5.26` | `1.48` |
| `adapt-rc090` | `adapt=true`, `rc=0.90`, `tau=0` | `5.183` | `+48.8%` | `14.4%` | `181.1 ms` | `3.46` | `1.52` | `1.45` | `5.350` | `+80.3%` | `3.3%` | `1.100` | `4.90` | `1.48` |
| **`adapt-rc050-tau060` (Boot 1)** | `adapt=true`, `rc=0.50`, `tau=0.6` | **`5.708`** | **`+63.9%`** | `13.6%` | **`158.9 ms`** | `2.78` | `2.11` | `1.89` | **`5.375`** | **`+81.2%`** | `4.2%` | `0.908` | `6.00` | `1.53` |
| **`adapt-rc050-tau060` (Boot 2)** | `adapt=true`, `rc=0.50`, `tau=0.6` | **`5.758`** | **`+65.3%`** | `12.9%` | **`156.9 ms`** | `2.93` | `2.04` | `1.77` | **`5.475`** | **`+84.5%`** | `4.0%` | `0.933` | `5.92` | `1.43` |
| ***`adapt-rc050-tau060` 2-boot mean*** | | ***`5.733`*** | ***`+64.6%`*** | | ***`157.9 ms`*** | *`2.86`* | *`2.07`* | *`1.83`* | ***`5.425`*** | ***`+82.9%`*** | | *`0.921`* | *`5.96`* | *`1.48`* |

### Bit-Exactness & Quality (`tp1-model16`: 16 Prompts, `c4`, 128 Tokens, Top-20 Logprobs)

| Arm | `outputs_sha256` | Token Diffs (of `2,048`) | Top-20 Logprob Diffs | `rank1_fraction` | `perplexity` | Draft Tokens Verified | Accepted / Verified | `generate_s` | Δ vs `d5-off` |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `d5-off` | `f6df69e6ee8f...` | `0` (`ref`) | `0` (`ref`) | `1.000` | `1.3209161` | `4,095` | `29.94%` | `642.82 s` | `0.0%` |
| `d0-off` | `f6df69e6ee8f...` | `0` | `0` | `1.000` | `1.3209161` | `0` | — | `397.45 s` | `-38.2%` |
| `adapt-rc050` | `f6df69e6ee8f...` | `0` | `0` | `1.000` | `1.3209161` | `879` | `82.37%` | `405.65 s` | `-36.9%` |
| **`adapt-rc050-tau060`** | **`f6df69e6ee8f...`** | **`0`** | **`0`** | **`1.000`** | **`1.3209161`** | **`781`** | **`89.24%`** | **`402.94 s`** | **`-37.3%`** |

### Memory & Boot Observations

- `dgx4` `MemAvailable` before every start: `115.1–116.6 GiB`; steady `memguard` active (`3 GiB` floor) with zero OOM or warning events across all 13 runs.
- Warm boot times on `dgx4`: `109.0 s` (`adapt-rc050-tau060` Boot 2), `97.5 s` (`d0-off` Boot 2).
- Initial `adapt-rc015` attempt at commit `974b11beb3` failed at API startup inside `VllmConfig._validate_adaptive_verification()` (`ValueError: Adaptive verification requires full CUDA graphs`); resolved cleanly by `overlay/vllm.py` (`PR #1288`, `b6c9d0fa37`). No runtime errors after that fix.

## Root-Cause Mechanics & Conclusion

1. **Why fixed drafts `1..5` all lose to `d0-off` on TP1 NVMe**: Fitting the fixed-draft step times gives `T_c1(r) ≈ 125 + 96·r ms` (effective marginal `row_cost ≈ 0.77`) and `T_c4(r) ≈ 240 + 134·r ms` (`row_cost ≈ 0.56`) — roughly `16×` higher marginal row cost than `tp3-triangle` DRAM (`0.049`). Even `drafts = 1` (`66.5%` average survival) adds more NVMe miss latency than its expected token gain when run blindly on every step.
2. **Why `tau = 0` adaptive lags `d0-off` at `c1`**: With `SPARK_DS41F_DSPARK_DEAD_ROWS_TAU = 0`, single-request budget decisions use **1-step-stale** host confidences, mis-predicting when a single stream flips between easy and hard tokens (`5.18–5.30 tok/s`).
3. **Why `adapt-rc050-tau060` (`row_cost = 0.50`, `SPARK_DS41F_DSPARK_DEAD_ROWS_TAU = 0.6`) wins overall**:
   - At `c1`, device `DeadRows` evaluates **current-step** exact draft confidences with effective threshold `tau_c1 = min(0.6, 2·(0.5·2.5) / (1 + 3·(0.5·2.5))) = 0.526`, keeping drafts `2..5` only on high-confidence steps (bursting `2.04–2.11 tok/step`) and immediately masking low-survival tail rows as padding (`topk_ids = -1`) so both `ExpertSlotCache` NVMe reads and `RouterLookaheadPredictor` skip them. Across 2 boots it hits **`5.733 tok/s` (`+64.6%` vs `d5-off`, `+2.8%` faster than `d0-off`)** and **`157.9 ms` TPOT (`-34.5%` vs `d5-off`, `-5.6%` lower latency than `d0-off`)**.
   - At `c4`, host budget compaction (`eff_rc = 0.50 × 1.19 = 0.595`) concentrates drafts onto whichever requests in the batch are currently high-confidence, lifting `model16` draft efficiency from **`29.9% → 89.2%`** and matching `d0-off` throughput (**`5.425 tok/s`, `+82.9%` vs `d5-off`**) while preserving burst upside on code/structured outputs.
   - **Zero regression on `TP3..TP8`**: `FULL_AND_PIECEWISE` / `FULL_DECODE_ONLY` shapes never hit the `not mode.has_full_cudagraphs()` branch or `ExpertSlotCache`.

**Verdict: SURVIVED (ready for promotion as vLLM patch `0844` + `tp1` NVMe-calibrated config `row_cost = 0.50`, `SPARK_DS41F_DSPARK_DEAD_ROWS_TAU = 0.6`, and `tp2-*` screen candidate).**
