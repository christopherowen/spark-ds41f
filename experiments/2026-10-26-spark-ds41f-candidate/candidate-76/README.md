# `candidate-76` — `1126` clean base with DSpark mask-embedding cache and truncated dynamic-depth CUDA graph retention

## Base and variables

- **Base deployment commit**: `5f37f854ec` (`spark-ds41f-1126`, promoted baseline `2026-11-26-spark-ds41f-1126`).
- **Base vLLM series**: `19f2c20ed4d64ae22bd009ceacadd85e2a81e354` + `candidate-74` (`0001..0869`, 742 patches, tree `f3783b0ec61851ba481f6a3a316d0a76d0e171d8`).
- **Candidate vLLM series**: `experiments/2026-10-26-spark-ds41f-candidate/candidate-76/vllm/series` (`0001..0871`, 744 patches, patch head `31e6870970eec4f16648556b4146f11c18427dd6`, tree `50dd2dac2e7c75257ed6c15136042050f5629b12`, patchset SHA-256 `ed89bf949fc0e569e16e1e86a0eeb4950a8c470dc3ee0773102cb0444c2bed4a`).
- **Runtime configuration**: identical to `releases/spark-ds41f-1126/cluster.json` across all 16 shapes (`DeepSeek-V4.1-Flash-1017`, `DSpark`, `GLP-27` at `alpha=2.0`, `sparknet v0.6.0`, `tilelang`).
- **Changes over `spark-ds41f-1126`**:
  1. `0870-dspark-cache-constant-mask-token-embedding-for-query.patch`: precompute and cache the constant `parallel_drafting_token_id` embedding once after draft weight load (`cache_mask_embed`), then embed only the `num_reqs` anchor tokens per decode step (`embed_query_block`) and broadcast the cached vector into mask offsets `1..block_size-1`. Bit-exact against full block embedding; shrinks the draft embedding lookup and TP all-reduce by `5x` on DeepSeek-V4.1-Flash (`6x` when `sample_from_anchor=False`).
  2. `0871-dspark-retain-FULL-CUDA-graph-replay-for-truncated-d.patch`: allow greedy `DSparkSpeculator` (without a logit-reading acceptance estimator) to stay on captured `FULL` CUDA graphs when `0 < active_spec < num_speculative_steps`, masking `sample_idx_mapping[:, active_spec:]` to `-1` before replay so in-graph confidence calibration neither records tail features nor predicts tail slots, and zeroing `draft_token_confidence_probs[:, active_spec:]` after replay. Zero behavior change at fixed `K=5`; removes eager fallback for future dynamic draft-depth schedules.

## Verification status

- **Targeted SM121 GPU/CPU unit tests (`dgx4` against `spark-ds41f:1126` + `0870..0871`)**: `106 passed, 1 skipped` across `test_dspark_mask_embed_cache.py`, `test_dspark_topk.py` (excluding TileLang-unsupported FP64 gumbel case), `test_dspark_vocab_parallel.py`, `test_dspark_v41_gathered_markov.py`, `test_dflash_prepare_inputs.py`, `test_dynamic_sd_cug.py`, `test_adaptive_verification.py`, and `test_deepseek_v4_dspark_context_kv.py`.
- **Image build (`spark-ds41f:candidate-76`)**: pending.
- **Full `release-tests` gate (`dgx4`)**: pending.
- **`tp3-triangle` 2-boot screen (`dgx1..dgx3` vs recorded `spark-ds41f-1126`)**: pending.
