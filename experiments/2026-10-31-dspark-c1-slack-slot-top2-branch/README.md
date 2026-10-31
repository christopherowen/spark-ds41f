# DSpark `c1` Slack-Slot Top-2 Runner-Up Tracking & Markov Lookahead Branch Promotion

## Summary

In `DSparkSpeculator`, vocabulary-parallel greedy drafting already allocates `self._greedy_shard_pairs` with shape `(max_num_reqs, 4)` (`16` bytes per rank for 128-bit aligned one-shot all-gather), leaving columns `2` and `3` unused.

This experiment packs each vocabulary shard's runner-up `(second_val, second_idx)` into columns `2..3` of `self._greedy_shard_pairs` (`zero` extra all-gather bytes) and extends `select_first_argmax` to reduce the global runner-up across all TP shards, record `(draft_runner_up_tokens, draft_top2_margins)`, and optionally promote the runner-up token in-register when the top-1 vs top-2 logit margin is narrow (`0 <= best - second <= branch_margin`) and the runner-up token yields higher next-step Markov transition confidence (`markov_conf_weight . (table[second] - table[best])`).

## Key Changes

- `vllm/v1/worker/gpu/sample/tilelang_sampling.py`:
  - Added branchless `_first_keep_or_take_top2`, `_first_exchange_top2`, `_warp_first_argmax_top2`, and `_select_pairs_top2` macros preserving `vllm_first_argmax_keep`'s total ordering (NaN first, then larger logit, ties to lower global vocabulary index).
  - Extended `partial_argmax_kernel` (`top2=True`) and `finish_argmax_kernel` (`output="pair_top2"`) to write `[best_v, best_i + vocab_start, second_v, second_i + vocab_start]` into the 4 columns of `pairs`.
  - Extended `select_first_argmax_kernel` (`top2_branch=True`) to reduce global top-1 and global top-2 across shards, write `runner_up_out` and `margin_out`, and in the fused embedding variant (`embed_width > 0`) score `markov_conf_weight . (table[second] - table[best])` across the warp when `0 < branch_margin` and `0 <= best - second <= branch_margin`, promoting `second` in-register before writing `step_out` and `confidence_out`.
- `vllm/v1/worker/gpu/spec_decode/dspark/greedy.py`:
  - Wired `top2` / `top2_branch` through `greedy_markov_shard_pair`, `select_first_argmax`, `scratch_shape`, and warmup declarations.
- `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py`:
  - Added `SPARK_DS41F_DSPARK_TOP2_BRANCH` (`dspark_top2_branch`), `SPARK_DS41F_DSPARK_TOP2_BRANCH_MARGIN` (`dspark_top2_branch_margin`, default `0.0`), and `SPARK_DS41F_DSPARK_TOP2_BRANCH_WEIGHT` (`dspark_top2_branch_weight`, default `1.0`), preallocating `draft_runner_up_tokens`, `draft_top2_margins`, and `_markov_conf_weight` (sliced once from `confidence_head.proj.weight[hidden:]` at model load for zero-allocation CUDA-graph capture).
- `tests/v1/spec_decode/test_dspark_vocab_parallel.py`:
  - Added unit tests across `dtype in {bfloat16, float32}` and `ranks in {1, 2, 3, 4}` verifying exact global top-1 and runner-up reduction (including cross-shard ties and intra-tile runner-ups) and Markov transition confidence promotion.

## Verification & Benchmark (`dgx4`, `spark-ds41f:19f2c20ed4d6-next-20`)

- **Correctness & PDL:**
  - `pytest -q --noconftest -p tests.kernels.deepseek_v41_pdl_sources tests/v1/spec_decode/test_dspark_vocab_parallel.py tests/v1/spec_decode/test_dspark_v41_gathered_markov.py tests/v1/spec_decode/test_tilelang_rejection.py tests/v1/spec_decode/test_vocab_parallel_drafting.py`: **720 passed, 0 PDL violations** (`PDL sources: 74 kernels launched, 0 with PDL, 0 violations`).
- **Microbenchmark (`c1`, `vocab=129280`, `embed_width=256`, per speculative draft step):**

| Configuration | `top1_per_step` (median / p10 / p90) | `top2_track_per_step` (median / p10 / p90) | `top2_branch_per_step` (median / p10 / p90) | Extra All-Gather Bytes |
|---|---|---|---|---|
| **TP3 (`shard_vocab=43,096`)** | `28.384 µs` (`28.096` / `28.832`) | `33.248 µs` (`33.024` / `33.664`) | `32.992 µs` (`32.672` / `33.440`) | `0 B` (`pair_width=4`) |
| **TP4 (`shard_vocab=32,320`)** | `28.480 µs` (`28.160` / `29.056`) | `33.440 µs` (`33.152` / `33.792`) | `33.280 µs` (`32.992` / `33.632`) | `0 B` (`pair_width=4`) |
