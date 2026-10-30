# Concurrency-Bounded Sequential Markov Drafting Depth for DSpark

## Motivation

In `DSparkSpeculator._sample_sequential`, every draft step currently projects all `num_reqs * num_speculative_steps` query hidden states through the 129,280-vocabulary draft LM head (`compute_draft_logits_local`), runs all `num_speculative_steps` sequential Markov `markov_bias_local` GEMVs + TP4 `all_gather` collectives + `greedy_markov_shard_pair` / `select_first_argmax` reductions, and evaluates `confidence_head` across all `num_reqs * num_speculative_steps` rows—even at `c8` and `c16` where tail draft positions (`k = 3, 4`) have survival below `dead_rows_tau` (`0.25`) or are pruned by the adaptive verification budget before target verification.

Because `DFlashCudaGraphManager` captures a separate full CUDA graph for each `num_reqs` (`num_tokens_padded = num_reqs * num_query_per_req`), `active_steps = self.active_speculative_steps(num_reqs)` is a capture-time constant for each CUDA graph size.

## Design

- Added `SPARK_DS41F_DSPARK_MAX_STEPS_BY_BATCH` (and `hf_config.dspark_max_steps_by_batch`) parsed via `parse_max_steps_by_batch(spec, num_speculative_steps)` into a monotonically non-increasing `(min_reqs, max_steps)` schedule.
- Added `DSparkSpeculator.active_speculative_steps(num_reqs) -> int` and `_configured_step_counts() -> tuple[int, ...]`:
  - When `active_steps < num_speculative_steps`:
    1. Only the first `active_steps` query positions per request (`num_reqs * active_steps` rows) are gathered into `sample_hidden` and projected through the draft LM head (`compute_draft_logits_local` / `compute_draft_logits` / `compute_local_draft_logits`).
    2. Only `active_steps` sequential Markov steps (`markov_bias_local`, `greedy_markov_shard_pair`, TP `all_gather`, and `select_first_argmax`) are executed.
    3. Only `num_reqs * active_steps` rows are evaluated by `compute_confidence`.
    4. Unverified tail slots `active_steps .. num_speculative_steps - 1` have `draft_tokens[:, active_steps:] = 0` and `draft_token_confidence_probs[:, active_steps:] = 0.0`, so `AdaptiveVerificationManager` (`get_num_tokens` and `DeadRows`) assigns them zero survival and never admits them.
  - Updated `DSparkSpeculator.register_warmups` to declare `greedy_markov_shard_pair` for each active step count in `_configured_step_counts()` so `base_stride = active_steps * head_width` is pre-compiled before the TileLang cache freezes.
  - Default behavior (`SPARK_DS41F_DSPARK_MAX_STEPS_BY_BATCH` unset) is 100% unchanged (`active_steps == num_speculative_steps` for all `num_reqs`).

## Verification

- **DGX4 `scripts/lab.py kernel-local` (`spark-ds41f:1030`):**
  - `tests/v1/worker/test_dspark_greedy_markov.py` + `tests/v1/spec_decode/test_dspark_vocab_parallel.py`: `44 passed in 20.63s` (recorded in `experiments/2026-10-30-dspark-concurrency-draft-depth/results.json`).
- **Host unit tests:**
  - `python3 -m unittest -v tests/test_dspark_concurrency_draft_depth.py`: `2 passed` (verifying schedule parsing, monotonic clamping, work savings in LM head rows, sequential Markov steps, TP4 `all_gather` collectives, and `confidence_head` rows, plus zeroing of unverified tail slots).
  - `python3 bin/spark doctor`: `configuration OK`.
