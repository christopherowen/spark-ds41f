# TP3-Triangle 16-Seat Qualification (`max_num_seqs: 8 -> 16`)

Date: 2026-11-22.
Base commit: `55a2e8689b` (`spark-ds41f:candidate-64`, `sha256:50b294060f808ce5d2369441995762d591e1e213fe17f75a4635e43ad44ce312`, vLLM `0001..0843`).
Target cluster: `dgx1..dgx3` (`tp3-triangle`).

## Hypothesis

Raising `tp3-triangle` from `8` to `16` concurrent seats (`max_num_seqs: 8 -> 16`, `max_cudagraph_capture_size: 48 -> 96`, `long_prefill_token_threshold: 6096 -> 6048`, extending `cudagraph_capture_sizes` up to `96` using the `tp4` step-8 ladder) with the KV pool unchanged (`4.91 GiB`, `5.49M` tokens = `16 × 342.8K`):

1. Unlocks sublinear routed-expert union scaling from `c8` to `c16` (`tp3` currently saturates at `~169.5 tok/s` because requests 9–16 queue; `tp4` gained `+31%` from `c8` to `c16`).
2. Eliminates the slot-8 occupancy cliff blamed for `stall-6` probe TTFT (`1.757 s` on `candidate-64` vs `1.18 s` on `1107`).
3. Preserves `c1`, `c8`, and `100%` greedy + top-20 logprob bit-exactness while keeping `dgx1` safely above the `5.0 GiB` startup and `3.0 GiB` steady memory-guard floors (predicted `~+1.5 GiB` graph + state cost, leaving `~7.3 GiB` steady `MemAvailable` on `dgx1`).

## One Causal Variable

Pure serving-config change on `spark-ds41f:candidate-64` (`tp3-16-seats.json`, zero code patches; `tp3-12-seats.json` staged as fallback if `dgx1` startup headroom dips below `5.0 GiB`).

## Comparison Reference (Recorded — Not Re-Measured)

- `candidate-64` (`max_num_seqs = 8`, 2-boot `tp3-triangle` qualification):
  - `model`: `100%` bit-exact (`outputs_sha256` match vs `1120`, `rank1 = 1.0`, `perplexity = 1.3137270`).
  - `c1`: `69.80 / 69.87 tok/s` (`tpot 14.38 ms`).
  - `c8`: `169.23 / 169.70 tok/s` (`mean 169.47 tok/s`).
  - `stall-6` probe p50: `1.757 s`.
  - `1M` prefill: `7,007 / 6,992 tok/s` (`mean 7,000 tok/s`).

## Status

Running 2-boot live qualification (`suite-tp3-16.json`) on `dgx1..dgx3` inside a shared lab window.
