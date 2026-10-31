# Align `DSparkContextCudaGraphs` Capture Ladder with Decode Shapes

## Overview

`DSparkContextCudaGraphs` previously constructed its CUDA graph capture ladder strictly as powers of two (`[1, 2, 4, 8, 16, 32, limit]`), ignoring `vllm_config.compilation_config.cudagraph_capture_sizes` and the target decoder's speculative batch shapes (`1..2 * decode_query_len` and uniform multiples of `decode_query_len`).

Because a speculative decode step with `num_speculative_tokens = 5` (`decode_query_len = 6`) advances in multiples of 6 tokens (or variable token counts `1..12` at low concurrency under adaptive draft budgets), almost every standard decode batch was padded up to the next power of two:

- `1` request (`6` tokens) $\rightarrow$ padded to `8` rows (**+33%**)
- `2` requests (`12` tokens) $\rightarrow$ padded to `16` rows (**+33%**)
- `3` requests (`18` tokens) $\rightarrow$ padded to `32` rows (**+78%**)
- `4` requests (`24` tokens) $\rightarrow$ padded to `32` rows (**+33%**)
- `6` requests on `tp3-triangle` (`36` tokens, `limit = 48`) $\rightarrow$ padded to `48` rows (**+33%**)

Every padded row executes extra work across:
1. `dspark_context.context_inputs` (scrubbing tail rows of `aux`, `positions`, and `slot_mappings`),
2. `main_proj` (`per_token_cast` on `15360`-wide BF16 activations, block-32 FP8 GEMM, and when `SPARK_DS41F_DSPARK_MAIN_PROJ_TP=1`, the TP `all_gather` of `10 KiB/row` over RoCE),
3. `main_norm` (`5120`-wide RMSNorm),
4. All 3 DSpark draft layers' `kv_proj` (`per_token_cast` + split-K GEMM + `splitk_reduce`), `kv_norm`, `rope_`, and `write_cache`.

## Changes

- Added `_context_cudagraph_capacities(vllm_config, limit)` in `vllm/models/deepseek_v4_1/nvidia/dspark.py` to include:
  - Exact single- and two-request decode row counts (`1..min(limit, 2 * decode_query_len)`),
  - Uniform speculative-decode batch multiples (`decode_query_len, 2 * decode_query_len, ...` up to `limit`),
  - Configured `compilation_config.cudagraph_capture_sizes` up to `limit`,
  - Power-of-two fallback capacities up to `limit`.
- Added unit test `test_context_cudagraph_capacities_cover_decode_ladder` in `tests/kernels/test_deepseek_v41_tilelang_dspark_context.py`.

## Verification & GB10 (`dgx4`) Benchmark

- `pytest tests/kernels/test_deepseek_v41_tilelang_dspark_context.py`: **13 passed** on `dgx4` (`spark-ds41f:1030`).
- Single-rank TP4 context graph replay (`dspark_context.context_inputs` + `DSparkContextCudaGraphs._forward` replay, 2000 iterations on `dgx4`, excluding additional RoCE `all_gather` savings):
  - `num_tokens = 3`: `95.77 µs` (`cap=4`) $\rightarrow$ **`92.73 µs`** (`cap=3`, `-3.04 µs`)
  - `num_tokens = 5`: `96.21 µs` (`cap=8`) $\rightarrow$ **`93.61 µs`** (`cap=5`, `-2.60 µs`)
  - `num_tokens = 6`: `97.28 µs` (`cap=8`) $\rightarrow$ **`95.70 µs`** (`cap=6`, `-1.58 µs` + `-20 KiB` RoCE `all_gather`)
  - `num_tokens = 12`: `102.91 µs` (`cap=16`) $\rightarrow$ **`100.92 µs`** (`cap=12`, `-1.99 µs` + `-40 KiB` RoCE `all_gather`)
  - `num_tokens = 18`: `124.44 µs` (`cap=32`) $\rightarrow$ **`118.72 µs`** (`cap=18`, `-5.72 µs` + `-140 KiB` RoCE `all_gather`)
  - `num_tokens = 20`: `124.64 µs` (`cap=32`) $\rightarrow$ **`117.28 µs`** (`cap=20`, `-7.36 µs` + `-120 KiB` RoCE `all_gather`)
  - `num_tokens = 24`: `124.66 µs` (`cap=32`) $\rightarrow$ **`118.95 µs`** (`cap=24`, `-5.71 µs` + `-80 KiB` RoCE `all_gather`)
