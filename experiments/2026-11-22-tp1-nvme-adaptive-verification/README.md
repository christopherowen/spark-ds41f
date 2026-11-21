# TP1 / TP2 NVMe-Offloaded Speculation Policy & `PIECEWISE` Adaptive Verification

Date: 2026-11-22.
Base commit: `9ccea03a9` (`spark-ds41f:candidate-64`, `sha256:50b29406...`, vLLM `0001..0843`, tree `7464b04c3cc7`).
Target host: `dgx4` (`tp1`, isolated single-node GB10; zero impact on `dgx1..dgx3` `tp3-triangle` production).

## Hypothesis

On `tp1` (`48` static + `48` dynamic slots out of `384` routed experts/layer) and `tp2-*` (`176` static + `64` dynamic), running `num_speculative_tokens = 5` with `enable_adaptive_verification = false` unconditionally pushes all 6 rows per request through 40 routed MoE layers on every decode step. Tail draft positions (`k = 2, 3, 4`, surviving only `~5–18%` of steps) trigger ~20× slower NVMe `io_uring` expert cache misses and evict warm experts from the 48/64 dynamic slots.

## One Causal Variable (`piecewise_adaptive_verification.patch`)

1. **`vllm/v1/worker/gpu/spec_decode/adaptive_verification.py`**: Allow `resolve_adaptive_cudagraph_mode()` to return `mode` unchanged when `not mode.has_full_cudagraphs()` (`PIECEWISE` / `NONE`), unlocking adaptive verification on expert-offloaded `PIECEWISE` shapes (`tp1`, `tp2-*`) with zero code/graph change on `FULL_AND_PIECEWISE` (`TP3..TP8`).
2. **`vllm/models/deepseek_v4_1/expert_slots/cache.py`**: Filter `is_padding` dead rows (`(topk_ids_np >= 0).any(axis=-1)`) out of `RouterLookaheadPredictor.predict_experts(lookahead_hidden)` so layer `L+1` lookahead never wastes NVMe `io_uring` reads on dead tail rows (`ExpertSlotCache._partition_windows` already strips `topk_ids < 0`). Only active when `slot_cache is not None` (`tp1`/`tp2-*`).

## Unit Verification (`dgx4`, `spark-ds41f:candidate-64`)

- `pytest --noconftest -q /tmp/tp1_adapt/test_piecewise_adaptive.py tests/v1/spec_decode/test_adaptive_verification.py`: **`54 passed, 1 skipped, 0 failed in 12.12s`**.

## Pre-Run Prediction

- Fixed `drafts ∈ {1, 2, 3}` will reduce NVMe misses per step vs `drafts = 5`, trading off peak code/boilerplate acceptance.
- `PIECEWISE` adaptive verification with NVMe-calibrated `row_cost ∈ {0.15, 0.30, 0.50, 0.75}` should outperform both fixed `drafts = 5` and fixed `drafts = 1..3` on `c1` and `c4` by keeping `4–5` verified tokens on confident steps while trimming uncertain tails to `1–2` rows, with 100% bit-exact greedy outputs (`0` diffs vs `tp1-d5-off`).

## Live Screening Status (`dgx4`, `tp1`)

Running isolated `dgx4` sweep across `tp1-d5-off` (baseline), `tp1-d3-off`, `tp1-d2-off`, `tp1-d1-off`, and `tp1-adapt-{rc015,rc030,rc050,rc075}`, measuring `c1`, `c4`, and `model` bit-exactness.
