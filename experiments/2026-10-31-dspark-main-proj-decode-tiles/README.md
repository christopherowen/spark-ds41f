# Tuned TileLang Decode GEMM Tiles for TP4 and TP3 DSpark `main_proj`

## Summary

With `SPARK_DS41F_DSPARK_MAIN_PROJ_TP=1` (enabled in `config/cluster.json`), `DSparkDeepseekV4Model.main_proj` shards the `[5120, 15360]` main projection across TP ranks as `_PaddedColumnParallelLinear`:
- **TP4 (`tp4-ring`)**: `(N=1280, K=15360)` per rank
- **TP3 (`tp3-triangle`)**: `(N=1728, K=15360)` per rank (padded to a multiple of 64)

`DECODE_FP8_CONFIGS` in `vllm/models/deepseek_v4_1/tilelang/gemm.py` previously contained only `(6400, 6144)` under `# TP4 DSpark main projection`, so both `(1280, 15360)` and `(1728, 15360)` fell back to the generic `block_K=128, num_stages=2` decode heuristic inside `DSparkContextCudaGraphs`.

## Changes

- **`vllm/models/deepseek_v4_1/tilelang/gemm.py`**:
  - Add `(1280, 15360)` and `(1728, 15360)` entries to `DECODE_FP8_CONFIGS` using `block_K=256, num_stages=3, threads=128` across `block_M in (16, 32, 64)`.
- **`tests/kernels/quantization/test_deepseek_v41_block_scale_words.py`**:
  - Include `K=15360` and verify `(15360, 256)` in `test_every_serving_k_tile_loads_the_scale_words_it_consumes`.
- **`tests/kernels/quantization/test_deepseek_v41_tilelang_linear.py`**:
  - Add `(1280, 15360)` and `(1728, 15360)` to `test_mxfp8_gemm` to verify bit-exact agreement between the `block_M in (16, 32, 64)` decode tiles and the prefill GEMM across all row counts.

## Benchmark on `dgx4` (`spark-ds41f:1030`, GB10 SM121)

| Shape `(N, K)` | `block_M` | Previous Default (`block_K=128, stages=2`) | Tuned (`block_K=256, stages=3`) | Speedup |
|---|---:|---:|---:|---:|
| `(1280, 15360)` (TP4) | `16` | `23.98 µs` | **`18.12 µs`** (`block_N=32`) | **1.32x (`-5.86 µs`)** |
| `(1280, 15360)` (TP4) | `32` | `31.25 µs` | **`26.18 µs`** (`block_N=64`) | **1.19x (`-5.07 µs`)** |
| `(1280, 15360)` (TP4) | `64` | `47.85 µs` | **`39.60 µs`** (`block_N=32`) | **1.21x (`-8.25 µs`)** |
| `(1728, 15360)` (TP3) | `16` | `65.58 µs` | **`61.55 µs`** (`block_N=32`) | **1.07x (`-4.03 µs`)** |
| `(1728, 15360)` (TP3) | `32` | `93.41 µs` | **`87.30 µs`** (`block_N=64`) | **1.07x (`-6.11 µs`)** |
| `(1728, 15360)` (TP3) | `64` | `96.70 µs` | **`87.31 µs`** (`block_N=64`) | **1.11x (`-9.39 µs`)** |

## Verification

- `pytest tests/kernels/quantization/test_deepseek_v41_block_scale_words.py tests/kernels/quantization/test_deepseek_v41_tilelang_linear.py`: **`36 passed`** on `dgx4`.
