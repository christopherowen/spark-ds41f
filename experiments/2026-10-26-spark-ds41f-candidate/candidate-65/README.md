# Candidate `candidate-65`

Date: 2026-11-22.
Base: `candidate-64` (`vLLM 0001..0843`, tree `7464b04c3cc79c6de1c1745f92c1cdeb2999a2a8`) rebased onto `2026-11-20-spark-ds41f-1120` (`config/cluster.json`).

## Delta Over `candidate-64`

1. **vLLM patch `0844`** (`0844-perf-dspark-allow-PIECEWISE-adaptive-verification-an.patch`, tree `34b01980b09e1f634738f7bc781a588327c26b75`, `717` patches total):
   - Allows `enable_adaptive_verification: true` under `cudagraph_mode: PIECEWISE` and `NONE` (`VllmConfig._validate_adaptive_verification` + `resolve_adaptive_cudagraph_mode`), preserving LoRA / `PP > 1` guards and leaving `FULL_AND_PIECEWISE` / `FULL_DECODE_ONLY` (`TP3..TP8`) completely untouched.
   - Filters dead/padding rows (`topk_ids < 0`) out of `ExpertSlotCache`'s `RouterLookaheadPredictor` so NVMe-offloaded `tp1`/`tp2-*` shapes never prefetch experts for pruned tail drafts.
   - Evidence: 13-run isolated `dgx4` (`tp1`) qualification ([2026-11-22-tp1-nvme-adaptive-verification](../../../2026-11-22-tp1-nvme-adaptive-verification/README.md), `PR #1289`): `c1` `+64.6%` (`3.483 → 5.733 tok/s`, TPOT `241.0 → 157.9 ms`), `c4` `+82.9%` (`2.967 → 5.425 tok/s`), `model16` `2,048/2,048` greedy tokens and top-20 logprobs `100%` bit-exact (`rank1 = 1.0`, `ppl = 1.3209161`, draft efficiency `29.9% → 89.2%`).
2. **`tp1.json` / `tp1-dev.json` NVMe-calibrated speculation policy**:
   - `enable_adaptive_verification: true`, `adaptive_verification_row_cost: 0.5`, `SPARK_DS41F_DSPARK_DEAD_ROWS_TAU: "0.6"`.
3. **`tp3.json` / `tp3-dev.json` / `tp3-switched.json` / `tp3-switched-dev.json` 16-seat capacity**:
   - `max_num_seqs: 8 → 16`, `max_cudagraph_capture_size: 48 → 96`, `long_prefill_token_threshold: 6096 → 6048`, `cudagraph_capture_sizes` extended to `96` (`KV` pool unchanged at `4.91 GiB` / `5.49M` tokens).
   - Evidence: 2-boot `tp3-triangle` qualification ([2026-11-22-tp3-16-seats](../../../2026-11-22-tp3-16-seats/README.md), `PR #1291`): `c16` `+29.5%` (`~169.5 → 219.48 tok/s`), `stall-6` probe TTFT `-51.2%` (`1.757 → 0.858 s`, `-27.3%` vs `1107`'s `1.18 s`), `c8` flat (`169.20` vs `169.47 tok/s`), `1M` prefill `7,058.5 tok/s` (`+0.84%`), `0/128` greedy token and top-20 logprob diffs, `dgx1` steady `MemAvailable = 7.27 GiB` (`+0.75 GiB` CUDA graphs).

*(Note: `VLLM_SHARE_PYNCCL_COMMS=1` was intentionally omitted — it was already measured neutral at `-0.043%` grid and dropped in Round 3.)*

## Build & Verification Status

Built `spark-ds41f:candidate-65` (`sha256:4d6be6fa9e4086f60e2b5bc1beb4be9b1efc94a6ed0ff35c6ff45f5bc35d5c0a`, vLLM tree `34b01980b09e1f634738f7bc781a588327c26b75`) in `481 s` on `dgx4` (lowest `MemAvailable = 81.9 GiB`). Built-image unit tests (`test_adaptive_verification.py` + `test_config.py -k adaptive_verification`) passed with `0` failures; streaming identical digest to `dgx1..dgx3`.
