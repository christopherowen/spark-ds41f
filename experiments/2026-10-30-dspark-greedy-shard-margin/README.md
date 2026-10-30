# 2026-10-30 — DSpark Greedy Shard Runner-Up Margin and Exposure-Bias Confidence Calibration

## Motivation

In vocabulary-parallel greedy DSpark drafting (`SPARK_DS41F_DSPARK_VOCAB_PARALLEL=1`), each TP rank reduces its local vocabulary shard (`32,320` columns at TP4) into a 16-byte row (`_greedy_shard_pairs` with shape `[max_num_reqs, 4]` FP32) for in-place one-shot `all_gather`:

- Column `0`: shard-local maximum logit
- Column `1`: global vocabulary index of the shard maximum
- Columns `2..3`: previously unused padding required only for 16-byte alignment

Meanwhile, the DSpark acceptance-confidence head (`TileLangConfidenceHead`) predicts per-position acceptance probability from `sigmoid(w . [h_k | e_k])` using a single position-invariant weight vector across all `K = 5` sequential Markov steps, without observing:

1. **Draft logit peakedness** — the logit margin $\Delta_{r,k} = \max(\ell^{(1)}_{r,k} - \ell^{(2)}_{r,k}, 0)$ between the winning draft token and the runner-up candidate.
2. **Sequential exposure bias** — later draft steps condition on unverified draft tokens rather than ground-truth anchors, causing empirical verified-row yield ($S_k / C_k$) to decay systematically with step index $k$.

## Design

1. **Zero-extra-byte shard runner-up reduction (`finish_argmax_kernel("pair_margin")`):**
   - When `track_runner_up=True` is passed to `greedy_markov_shard_pair` / `greedy_shard_pair`, the final warp reduction across the shard's 2048-token block maxima writes the best non-winning block maximum into column `2` of the existing 16-byte `[rows, 4]` pair buffer.
   - Communication volume across TP ranks remains identically `16 B/rank/req` (`64 B/req` at TP4) — zero extra collective bytes and zero extra kernel launches.
   - Default `track_runner_up=False` leaves columns `2..3` untouched.
2. **Cross-rank argmax + non-negative logit margin (`select_first_argmax(..., margin_out=...)`):**
   - After the one-shot `all_gather`, `select_first_argmax_kernel` reduces the global winning token across ranks and, when `margin_out` is provided, reduces the runner-up across non-winning ranks' maxima (`pairs[row, 4*r + 0]`) and the winning rank's shard runner-up (`pairs[row, 4*r + 2]`), writing $\Delta_{r,k} = \max(\text{best} - \text{runner\_up}, 0)$ alongside the Markov embedding lookup in the same single-warp launch.
3. **Fused margin and step-bias confidence calibration (`dspark_confidence_calibrated`):**
   - When `SPARK_DS41F_DSPARK_MARGIN_WEIGHT` (`dspark_margin_weight`) or `SPARK_DS41F_DSPARK_CONFIDENCE_STEP_BIASES` (`dspark_confidence_step_biases`) is configured, `TileLangConfidenceHead.probability` evaluates:
     $$p_{r,k} = \sigma\bigl(w \cdot [h_{r,k} \mid e_{r,k}] + w_{\text{margin}} \Delta_{r,k} + \beta_k\bigr)$$
     in one TileLang launch. When both are unset (`0.0` / `None`), the existing uncalibrated path runs unchanged.

## Verification

- **GPU kernel verification (`dgx4`, `spark-ds41f:1030`):**
  - `scripts/lab.py kernel-local` passed `196 passed` across `test_deepseek_v41_tilelang_confidence.py`, `test_gpu_tilelang_sampling.py`, `test_dspark_vocab_parallel.py`, and `test_dspark_greedy_markov.py`.
- **Host unit tests:**
  - `pytest tests/test_dspark_greedy_shard_margin.py` verifies exactness across shard boundaries, within-shard blocks, cross-rank ties, and recorded TP4 suite profiles.
