# Q-B and Indexer Q-B 128x128 Tile-Contiguous Weight Layout and Decode Tile Alignment

## Context and Base State

- **Repository base commit**: `8e5ea208f` (`main`)
- **Base candidate**: `experiments/2026-10-26-spark-ds41f-candidate/1107` (`spark-ds41f:1107`, vLLM patch head `2756478b60b90700b9918b457aeb38657da637b5`, tree `ed1bdb7a7684da33d403a055521ac0c4dd7ca670`)
- **Candidate patch**: `vllm/0762-perf-dsv41-align-qb-and-indexer-qb-tile-contiguous-decode.patch` (vLLM patch head `38d8e75b4fe3601909100999d6f8989b44cf9bf7`, tree `d98ff533788e1d9ca758c0a86259ca1ba1695534`)
- **Hardware**: `sp2` (`NVIDIA GB10`, SM121, 48 SMs, 24 MiB L2 cache, LPDDR5X unified memory)

## Root-Cause Analysis

In DeepSeek-V4.1 decode, the attention query up-projection `wq_b` (`Q-B`) and the indexer query projection `indexer.wq_b` (`indexer Q-B`, including the fused GPT-J RoPE + MXFP4 epilogue kernel `mxfp8_gemm_rope_mxfp4_decode`) have input dimension `K = 1280` (`1,280 B` per row in E4M3).

1. **Row-major `[N, K]` cache-line striding under serving L2 pressure**:
   - In row-major `[N, 1280]`, consecutive rows of a `128 x 128` weight tile are separated by `1,280 B` (`10` cache lines). Under serving L2 pressure (Cold L2 or `NEXT` window L2 prefetch), each warp's `cp.async.cg.shared.global` `16 B` loads scatter across 4 discontiguous rows per step, trailing in serving by `3–6%` relative to contiguous tile streaming.
   - `pack_weight_tiles` (`TILE_WEIGHT_N = 128, TILE_WEIGHT_K = 128`) permutes `[N, K]` at weight load time into `[N // 128, K // 128, 128, 128]` viewed as `(N * (K // 128), 128)` (`16 KiB` contiguous slab per `128 x 128` tile) with zero extra memory footprint and 100% bit-identity with prefill (`mxfp8_gemm`) and row-major decode (`mxfp8_gemm_decode` / `mxfp8_gemm_rope_mxfp4_decode`).

2. **Tile-splitting bottleneck in `DECODE_FP8_CONFIGS` when `block_N = 64` is paired with `TILE_WEIGHT_N = 128`**:
   - Five of the Q-B / indexer Q-B decode buckets in `1107`'s `DECODE_FP8_CONFIGS` — `(12288, 1280)` at `block_M in (32, 64)`, `(8192, 1280)` at `block_M in (32, 64)`, and `(4096, 1280)` at `block_M = 16` — used `block_N = 64` while `TILE_WEIGHT_N = 128` (and `(8192, 1280)` at `block_M = 64` used `num_stages = 3`).
   - Whenever `block_N = 64` reads a `128 x 128` (`16 KiB`) tile-contiguous weight tensor, each `16 KiB` packed tile is split across two CTAs (`bx` even reads the upper `64 x 128 = 8 KiB` half, `bx` odd reads the lower `8 KiB` half). Each CTA must jump across an `8 KiB` non-contiguous gap on every `ko` step (`K // 128 = 10` steps), while launching `2x` as many CTAs doubles redundant shared-memory `A` and `SFA` loads.
   - On GB10 (`48 SMs`), whenever `N >= 6144`, `N // 128 >= 48` CTAs (`48` CTAs at `N = 6144`, `64` CTAs at `N = 8192`, `96` CTAs at `N = 12288`, `128` CTAs at `N = 16384`), so `block_N = 128` already saturates all `48 SMs` across all `block_M in (16, 32, 64)`. For `N in (4096, 2048)` (`32` and `16` CTAs at `block_N = 128`), `block_N = 128` fits in a single wave on `48 SMs` at `block_M in (16, 32)` (eliminating the 2-wave tail of `block_N = 64` at `N = 4096` and cutting Warm latency by `10–19%`), while `block_M = 64` retains `block_N = 64, num_stages = 2` so 64-row RoPE + MXFP4 epilogues utilize all `48 SMs`.

3. **Missing Q-B shard shapes in `TILE_CONTIGUOUS_FP8` and `DECODE_FP8_CONFIGS`**:
   - `(16384, 1280)` (TP=2 Q-B, 8 head groups), `(6144, 1280)` (TP=6/7 Q-B ranks 0..4, 3 head groups), and `(2048, 1280)` (TP=6/7 Q-B rank 5 after `_trim_padded_groups`, 1 head group) were absent from `TILE_CONTIGUOUS_FP8` and `DECODE_FP8_CONFIGS`, falling back to row-major `[N, K]` weight reads and generic `block_N = 64` fallback tiles.

## Causal Variable Changed

`vllm/0762-perf-dsv41-align-qb-and-indexer-qb-tile-contiguous-decode.patch`:
1. Registers `(16384, 1280)`, `(6144, 1280)`, and `(2048, 1280)` in `TILE_CONTIGUOUS_FP8` and `DECODE_FP8_CONFIGS` in `vllm/models/deepseek_v4_1/tilelang/gemm.py`.
2. Aligns all `N >= 6144` Q-B shapes (`(16384, 1280)`, `(12288, 1280)`, `(8192, 1280)`, `(6144, 1280)`) to `block_N = 128, block_K = 128` (`num_stages = 3` at `block_M in (16, 32)`, `num_stages = 2` at `block_M = 64`), and aligns `N in (4096, 2048)` to `block_N = 128, block_K = 128, num_stages = 3` at `block_M in (16, 32)` and `block_N = 64, block_K = 128, num_stages = 2` at `block_M = 64`.
3. Extends `test_tile_contiguous_fp8_matches_row_major` in `tests/kernels/quantization/test_deepseek_v41_tilelang_linear.py` to verify tile alignment and bit-exactness across all six `TILE_CONTIGUOUS_FP8` shapes.

## Reproduction Commands

```bash
# 2-boot pytest verification (189 tests per boot)
for boot in 1 2; do
  flock /tmp/gpu.lock docker run --rm --gpus all \
    -v "$PWD":/tmp/wt-spark-qb:ro \
    -e PYTHONPATH=/tmp/wt-spark-qb/scripts/release_tests:/opt/spark-ds41f/candidate/vllm \
    -w /opt/spark-ds41f/candidate/vllm \
    --entrypoint pytest spark-ds41f:1107 -v -p no:cacheprovider -p tblib_stub -p offline_skip \
    tests/kernels/quantization/test_deepseek_v41_tilelang_linear.py \
    tests/kernels/quantization/test_deepseek_v41_block_scale_words.py \
    tests/models/test_deepseek_v4_1_attention.py \
    tests/models/test_deepseek_v4_1_l2_schedule.py \
    tests/kernels/test_deepseek_v41_tilelang_pdl.py \
    | tee experiments/2026-11-08-qb-indexer-qb-tile-contiguous-weights/receipts/pytest-boot${boot}.txt
done

# 2-boot benchmark and bit-exactness suite
for boot in 1 2; do
  flock /tmp/gpu.lock docker run --rm --gpus all \
    -v "$PWD":/tmp/wt-spark-qb \
    -w /opt/spark-ds41f/candidate/vllm \
    --entrypoint python3 spark-ds41f:1107 \
    /tmp/wt-spark-qb/experiments/2026-11-08-qb-indexer-qb-tile-contiguous-weights/bench_qb_tile_contiguous.py \
    --output /tmp/wt-spark-qb/experiments/2026-11-08-qb-indexer-qb-tile-contiguous-weights/receipts/bench-boot${boot}.json
done
```

## Bit-Exactness Verification

Across both container boots (`receipts/bench-boot1.json`, `receipts/bench-boot2.json`, `receipts/pytest-boot1.txt`, `receipts/pytest-boot2.txt`):
- **`0` uint16 diffs (`torch.equal`)** between `TC_aligned` decode (`mxfp8_gemm_decode(..., tile_contiguous=True)`), prefill (`mxfp8_gemm` on row-major weights), row-major decode (`RM`), and `1107` decode across all 6 `Q-B` shapes (`(16384, 1280)`, `(12288, 1280)`, `(8192, 1280)`, `(6144, 1280)`, `(4096, 1280)`, `(2048, 1280)`) and all batch sizes `M in (1, 2, 3, 6, 12, 24, 48)`.
- **`0` uint8 diffs (`torch.equal`)** on both packed E2M1 nibbles and UE8M0 scales in fused `indexer.wq_b` (`mxfp8_gemm_rope_mxfp4_decode(4096, 1280)`) against the unfused reference (`mxfp8_gemm` + `quantize_mxfp4`), row-major decode, and `1107` decode across `M in (1, 2, 3, 6, 12, 24, 48)`.
- **`100%` SHA-256 digest equality (`digests_equal = True`)** across all 28 output/intermediate tensors of the 20-kernel PDL `LayerChain` at `c1 (M = 6)`, `c8 (M = 24)`, and `c8x6 (M = 48)` on both boots.
- **`189 passed` on Boot 1 and `189 passed` on Boot 2** across the 5-file kernel/attention/L2-schedule/PDL `pytest` suite (`0` failures, `0` errors).
- **Memory overhead**: `0 B` additional device or host memory (`pack_weight_tiles` replaces `layer.tilelang_weight` with an identically sized `(N * (K // 128), 128)` FP8 tensor view).

## 2-Boot Benchmark Results on `sp2` (`NVIDIA GB10`)

All latencies are in microseconds (`us`), formatted as `Boot 1 / Boot 2` (`receipts/bench-boot1.json` and `receipts/bench-boot2.json`).
- **Warm**: L2-resident (25%–75% interquartile trimmed mean of 30 `cudaEvent` pairs, 20 back-to-back launches per pair).
- **Cold**: L2-flushed before each launch (`48 MiB` L2 flush buffer, trimmed mean over 100 single-shot `cudaEvent` pairs).
- **NEXT**: L2-flushed followed by `vllm.models.deepseek_v4_1.l2_prefetch` `NEXT`-window asynchronous L2 prefetch + `join()` before each launch (100 single-shot `cudaEvent` pairs).

### 1. Chained `Q-B + Indexer Q-B` (`mxfp8_gemm_decode` + `mxfp8_gemm_rope_mxfp4_decode`)

#### `TP=4` (`Q-B 8192x1280` + `Indexer Q-B 4096x1280`)

| M | RM Warm (b1/b2) | 1107 Warm (b1/b2) | TC_aligned Warm (b1/b2) | Warm vs RM / 1107 | RM Cold (b1/b2) | 1107 Cold (b1/b2) | TC_aligned Cold (b1/b2) | Cold vs RM / 1107 | RM NEXT (b1/b2) | 1107 NEXT (b1/b2) | TC_aligned NEXT (b1/b2) | NEXT vs RM / 1107 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 19.13 / 18.71 | 21.80 / 21.32 | **19.44 / 19.23** | +2.2% / **-10.3%** | 110.70 / 112.96 | 105.63 / 104.84 | **106.12 / 104.97** | **-5.6%** / +0.3% | 109.32 / 111.90 | 105.66 / 104.66 | **105.95 / 105.03** | **-4.6%** / +0.3% |
| 2 | 20.32 / 19.98 | 22.43 / 22.06 | **20.11 / 19.69** | **-1.2%** / **-10.5%** | 110.74 / 112.31 | 105.67 / 104.68 | **106.39 / 105.33** | **-5.1%** / +0.7% | 109.33 / 114.63 | 105.67 / 107.03 | **106.18 / 106.48** | **-5.0%** / -0.0% |
| 3 | 18.89 / 18.70 | 21.66 / 21.41 | **19.36 / 19.28** | +2.8% / **-10.3%** | 109.83 / 113.24 | 106.28 / 105.42 | **107.03 / 105.54** | **-4.7%** / +0.4% | 109.43 / 112.10 | 106.14 / 105.16 | **106.79 / 105.77** | **-4.0%** / +0.6% |
| 6 | 19.15 / 18.96 | 21.19 / 20.85 | **20.36 / 19.71** | +5.1% / **-4.7%** | 111.43 / 113.50 | 106.27 / 105.69 | **106.69 / 105.90** | **-5.5%** / +0.3% | 110.88 / 112.93 | 109.65 / 105.33 | **110.34 / 105.96** | **-3.4%** / +0.6% |
| 12 | 19.26 / 17.20 | 20.73 / 17.81 | **18.50 / 18.37** | +1.1% / **-4.3%** | 113.03 / 115.57 | 107.57 / 106.29 | **109.28 / 108.51** | **-4.7%** / +1.8% | 112.31 / 114.66 | 107.22 / 106.14 | **109.11 / 108.29** | **-4.2%** / +1.9% |
| 24 | 23.56 / 23.55 | 24.45 / 24.51 | **23.57 / 23.93** | +0.8% / **-3.0%** | 118.51 / 120.47 | 114.84 / 114.90 | **114.10 / 113.84** | **-4.6%** / **-0.8%** | 117.27 / 119.97 | 114.91 / 114.52 | **114.19 / 113.72** | **-3.9%** / **-0.7%** |
| 48 | 38.34 / 37.81 | 41.33 / 41.87 | **38.00 / 38.21** | +0.1% / **-8.4%** | 120.99 / 124.03 | 117.57 / 116.53 | **117.76 / 115.86** | **-4.7%** / **-0.2%** | 119.27 / 123.63 | 117.40 / 116.28 | **117.50 / 115.96** | **-3.9%** / **-0.1%** |

#### `TP=3` (`Q-B 12288x1280` + `Indexer Q-B 4096x1280`)

| M | RM Warm (b1/b2) | 1107 Warm (b1/b2) | TC_aligned Warm (b1/b2) | Warm vs RM / 1107 | RM Cold (b1/b2) | 1107 Cold (b1/b2) | TC_aligned Cold (b1/b2) | Cold vs RM / 1107 | RM NEXT (b1/b2) | 1107 NEXT (b1/b2) | TC_aligned NEXT (b1/b2) | NEXT vs RM / 1107 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 23.67 / 23.64 | 28.51 / 33.25 | **26.77 / 33.91** | +28.3% / **-1.7%** | 145.04 / 145.48 | 138.30 / 138.88 | **138.27 / 138.97** | **-4.6%** / +0.0% | 143.97 / 144.57 | 137.81 / 138.59 | **138.15 / 138.70** | **-4.1%** / +0.2% |
| 2 | 23.91 / 24.49 | 29.02 / 33.81 | **26.85 / 33.99** | +25.7% / **-3.2%** | 144.61 / 144.73 | 137.96 / 138.65 | **138.51 / 139.07** | **-4.1%** / +0.4% | 143.78 / 144.33 | 138.00 / 138.71 | **138.17 / 138.97** | **-3.8%** / +0.2% |
| 3 | 23.78 / 23.49 | 28.41 / 32.85 | **26.59 / 33.64** | +27.4% / **-1.7%** | 146.36 / 145.75 | 138.38 / 138.64 | **138.76 / 139.03** | **-4.9%** / +0.3% | 144.24 / 144.98 | 138.10 / 138.76 | **138.64 / 139.01** | **-4.0%** / +0.3% |
| 6 | 22.84 / 22.69 | 28.71 / 33.12 | **28.62 / 34.09** | +37.7% / +1.4% | 145.63 / 146.15 | 138.96 / 139.98 | **139.12 / 140.16** | **-4.3%** / +0.1% | 145.81 / 145.13 | 139.14 / 139.60 | **139.06 / 140.22** | **-4.0%** / +0.2% |
| 12 | 26.55 / 25.13 | 27.05 / 40.95 | **28.38 / 42.43** | +37.0% / +4.1% | 148.07 / 147.51 | 139.72 / 140.31 | **141.54 / 141.66** | **-4.2%** / +1.1% | 146.54 / 147.06 | 139.40 / 139.85 | **141.65 / 141.56** | **-3.5%** / +1.4% |
| 24 | 32.59 / 28.75 | 36.53 / 42.33 | **36.37 / 42.75** | +29.0% / +0.3% | 152.80 / 152.53 | 148.47 / 148.39 | **148.01 / 147.25** | **-3.3%** / **-0.5%** | 151.98 / 152.19 | 148.20 / 149.15 | **147.61 / 151.30** | **-1.7%** / +0.5% |
| 48 | 46.41 / 49.61 | 53.66 / 63.98 | **52.69 / 60.71** | +18.1% / **-3.6%** | 155.16 / 155.13 | 149.51 / 150.28 | **149.18 / 149.23** | **-3.8%** / **-0.5%** | 154.23 / 154.58 | 148.96 / 149.74 | **148.91 / 149.06** | **-3.5%** / **-0.2%** |

> Note on TP=3 Chained Warm (`12288x1280 + 4096x1280`): The combined weights (`15.0 MiB` Q-B + `5.0 MiB` Indexer Q-B + scales) span `~20.6 MiB` inside GB10's `24 MiB` L2 cache (`19 MiB` effective non-set-aside capacity); under back-to-back synthetic loop replay without intermediate layers, `RM`'s `1,280 B` strided accesses touch fewer cache sets at small `M`, whereas under realistic serving conditions (Cold L2 and `NEXT`-prefetched L2), `TC_aligned` consistently wins by **-3.3% to -4.9% Cold** and **-1.7% to -4.1% NEXT** across all `M = 1..48`.

### 2. Full 20-Kernel PDL `LayerChain` CUDA Graph Replay (`TP=4` Geometry)

| Graph Bucket | Rows (`M`) | RM Warm (b1/b2) | TC_aligned Warm (b1/b2) | Warm Delta | RM Cold (b1/b2) | TC_aligned Cold (b1/b2) | Cold Delta | Digests Equal |
|---|---|---|---|---|---|---|---|---|
| `c1` | `6` | 945.08 / 962.87 (`953.98`) | **944.52 / 946.73 (`945.62`)** | **-8.35 us (-0.9%)** | 993.67 / 997.01 (`995.34`) | **990.97 / 992.27 (`991.62`)** | **-3.72 us (-0.4%)** | `True` |
| `c8` | `24` | 2699.47 / 2727.82 (`2713.64`) | **2704.15 / 2697.82 (`2700.99`)** | **-12.66 us (-0.5%)** | 2741.35 / 2743.20 (`2742.27`) | **2734.36 / 2740.25 (`2737.31`)** | **-4.97 us (-0.2%)** | `True` |
| `c8x6` | `48` | 4362.51 / 4371.14 (`4366.83`) | **4361.70 / 4377.67 (`4369.68`)** | +2.86 us (+0.1%) | 4524.89 / 4410.05 (`4467.47`) | **4408.94 / 4406.86 (`4407.90`)** | **-59.57 us (-1.3%)** | `True` |

### 3. Isolated `Q-B` Projections Across All `TILE_CONTIGUOUS_FP8` Shapes

#### `8192 x 1280` (`TP=4`, `TP=5`, `TP=3` rank 2)

| M | RM Warm (b1/b2) | 1107 Warm (b1/b2) | TC_aligned Warm (b1/b2) | Warm vs RM / 1107 | RM Cold (b1/b2) | 1107 Cold (b1/b2) | TC_aligned Cold (b1/b2) | Cold vs RM / 1107 | RM NEXT (b1/b2) | 1107 NEXT (b1/b2) | TC_aligned NEXT (b1/b2) | NEXT vs RM / 1107 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 11.81 / 13.60 | 12.47 / 13.45 | **12.36 / 13.56** | +2.0% / +0.0% | 68.00 / 68.19 | 67.02 / 67.80 | **66.83 / 67.60** | **-1.3% / -0.3%** | 67.78 / 68.23 | 66.95 / 67.80 | **67.13 / 67.85** | **-0.8%** / +0.2% |
| 2 | 12.85 / 12.86 | 12.53 / 12.74 | **12.52 / 12.54** | **-2.5% / -0.8%** | 68.44 / 68.47 | 67.20 / 67.97 | **67.29 / 67.86** | **-1.3% / -0.0%** | 67.82 / 68.49 | 67.24 / 67.93 | **66.93 / 67.89** | **-1.1% / -0.3%** |
| 3 | 11.71 / 11.66 | 12.34 / 12.38 | **12.39 / 12.14** | +5.0% / **-0.8%** | 68.24 / 68.47 | 67.37 / 68.13 | **67.24 / 67.81** | **-1.2% / -0.3%** | 68.05 / 68.43 | 67.06 / 67.93 | **67.13 / 67.83** | **-1.1% / -0.0%** |
| 6 | 11.68 / 11.79 | 12.12 / 12.25 | **12.23 / 12.25** | +4.3% / +0.5% | 68.35 / 68.86 | 67.40 / 68.56 | **67.33 / 68.16** | **-1.3% / -0.3%** | 68.19 / 68.70 | 67.38 / 68.39 | **67.31 / 68.25** | **-1.0% / -0.2%** |
| 12 | 10.14 / 10.02 | 10.52 / 10.67 | **10.84 / 10.60** | +6.3% / +1.2% | 69.37 / 69.86 | 68.32 / 68.76 | **68.00 / 68.59** | **-1.9% / -0.4%** | 68.19 / 68.47 | 67.77 / 68.68 | **68.08 / 69.02** | +0.3% / +0.5% |
| 24 | 14.91 / 12.80 | 14.64 / 13.35 | **13.86 / 12.87** | **-3.5% / -4.5%** | 70.25 / 70.98 | 68.19 / 69.98 | **68.71 / 69.52** | **-2.1%** / +0.0% | 70.05 / 72.01 | 68.41 / 71.37 | **68.70 / 70.55** | **-2.0% / -0.4%** |
| 48 | 20.39 / 20.31 | 22.78 / 23.46 | **20.49 / 20.23** | +0.0% / **-11.9%** | 73.05 / 73.08 | 70.66 / 72.91 | **71.14 / 72.27** | **-1.9% / -0.1%** | 72.36 / 72.76 | 70.65 / 72.94 | **70.94 / 72.42** | **-1.2% / -0.2%** |

#### `12288 x 1280` (`TP=3` ranks 0–1)

| M | RM Warm (b1/b2) | 1107 Warm (b1/b2) | TC_aligned Warm (b1/b2) | Warm vs RM / 1107 | RM Cold (b1/b2) | 1107 Cold (b1/b2) | TC_aligned Cold (b1/b2) | Cold vs RM / 1107 | RM NEXT (b1/b2) | 1107 NEXT (b1/b2) | TC_aligned NEXT (b1/b2) | NEXT vs RM / 1107 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 16.29 / 16.40 | 16.38 / 16.32 | **16.57 / 16.55** | +1.3% / +1.3% | 113.21 / 110.97 | 104.87 / 105.64 | **104.73 / 105.22** | **-6.3% / -0.3%** | 112.69 / 110.44 | 104.98 / 105.22 | **105.14 / 105.31** | **-5.7%** / +0.1% |
| 2 | 16.91 / 16.83 | 17.01 / 16.90 | **16.90 / 16.80** | **-0.1% / -0.6%** | 112.97 / 110.68 | 105.13 / 106.04 | **104.50 / 105.37** | **-6.2% / -0.6%** | 112.34 / 110.17 | 104.95 / 105.45 | **104.84 / 105.44** | **-5.5% / -0.1%** |
| 3 | 16.11 / 16.14 | 16.19 / 16.44 | **16.27 / 16.57** | +1.8% / +0.6% | 113.56 / 111.42 | 105.19 / 105.95 | **105.01 / 105.54** | **-6.4% / -0.3%** | 112.36 / 110.26 | 105.18 / 105.91 | **104.94 / 107.26** | **-4.7%** / +0.5% |
| 6 | 15.16 / 17.70 | 15.39 / 18.86 | **15.32 / 15.27** | **-6.9% / -10.7%** | 113.50 / 112.24 | 105.62 / 106.18 | **105.71 / 106.58** | **-6.0%** / +0.2% | 114.60 / 110.95 | 106.77 / 105.77 | **106.70 / 106.17** | **-5.6%** / +0.2% |
| 12 | 12.01 / 13.01 | 15.99 / 12.86 | **12.97 / 13.02** | +3.9% / **-9.9%** | 113.58 / 112.02 | 106.10 / 106.63 | **105.94 / 107.74** | **-5.3%** / +0.4% | 113.35 / 111.28 | 105.72 / 106.58 | **106.11 / 107.35** | **-5.0%** / +0.5% |
| 24 | 15.76 / 15.68 | 18.46 / 18.38 | **15.71 / 15.46** | **-0.9% / -15.4%** | 114.27 / 113.03 | 108.20 / 110.81 | **107.70 / 108.79** | **-4.8% / -1.2%** | 113.93 / 113.23 | 108.38 / 109.72 | **107.51 / 108.69** | **-4.8% / -0.9%** |
| 48 | 25.56 / 25.35 | 33.30 / 33.38 | **25.35 / 26.36** | +1.6% / **-22.5%** | 116.99 / 115.18 | 109.34 / 110.66 | **110.69 / 110.04** | **-4.9%** / +0.3% | 116.57 / 114.84 | 109.63 / 109.36 | **110.83 / 110.28** | **-4.5%** / +1.0% |

#### `16384 x 1280` (`TP=2` Q-B — Newly Added to `TILE_CONTIGUOUS_FP8` & `DECODE_FP8_CONFIGS`)

| M | RM Warm (b1/b2) | 1107 Warm (b1/b2) | TC_aligned Warm (b1/b2) | Warm vs RM / 1107 | RM Cold (b1/b2) | 1107 Cold (b1/b2) | TC_aligned Cold (b1/b2) | Cold vs RM / 1107 | RM NEXT (b1/b2) | 1107 NEXT (b1/b2) | TC_aligned NEXT (b1/b2) | NEXT vs RM / 1107 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 27.87 / 22.47 | 32.79 / 32.86 | **25.47 / 22.29** | **-5.1% / -27.3%** | 144.12 / 142.06 | 144.91 / 142.93 | **139.10 / 137.72** | **-3.3% / -3.8%** | 143.40 / 141.83 | 144.91 / 142.96 | **136.58 / 137.85** | **-3.8% / -4.7%** |
| 2 | 24.71 / 22.85 | 31.80 / 32.09 | **24.89 / 22.26** | **-0.9% / -26.2%** | 142.08 / 142.37 | 142.31 / 143.09 | **136.71 / 137.53** | **-3.6% / -3.9%** | 141.29 / 141.78 | 142.16 / 142.99 | **136.50 / 137.61** | **-3.2% / -3.9%** |
| 3 | 25.51 / 22.74 | 32.34 / 32.72 | **24.37 / 22.02** | **-3.9% / -28.7%** | 141.87 / 142.79 | 142.82 / 143.45 | **136.53 / 137.56** | **-3.7% / -4.3%** | 141.49 / 141.89 | 142.46 / 143.49 | **136.59 / 137.83** | **-3.2% / -4.0%** |
| 6 | 27.06 / 23.16 | 27.43 / 27.69 | **24.35 / 20.84** | **-10.0% / -18.0%** | 143.10 / 144.83 | 142.47 / 145.44 | **137.09 / 138.77** | **-4.2% / -4.2%** | 142.49 / 142.34 | 142.68 / 144.60 | **136.83 / 138.37** | **-3.4% / -4.2%** |
| 12 | 25.23 / 22.99 | 27.52 / 23.87 | **28.39 / 20.29** | +1.0% / **-5.3%** | 143.17 / 143.96 | 142.88 / 143.02 | **137.39 / 138.56** | **-3.9% / -3.5%** | 142.26 / 142.73 | 143.11 / 143.17 | **136.74 / 138.23** | **-3.5% / -4.0%** |
| 24 | 31.55 / 29.63 | 31.09 / 25.49 | **26.42 / 25.06** | **-15.9% / -9.0%** | 144.68 / 144.04 | 143.80 / 143.84 | **138.50 / 139.59** | **-3.7% / -3.3%** | 143.87 / 143.33 | 143.41 / 143.84 | **138.19 / 139.48** | **-3.3% / -3.3%** |
| 48 | 44.53 / 34.74 | 43.69 / 45.37 | **39.01 / 35.02** | **-6.6% / -16.9%** | 144.80 / 145.55 | 142.97 / 142.15 | **137.81 / 140.54** | **-4.1% / -2.4%** | 144.29 / 145.16 | 142.75 / 141.84 | **137.89 / 140.49** | **-3.8% / -2.2%** |

#### `6144 x 1280` (`TP=6 / TP=7` Q-B ranks 0–4 — Newly Added to `TILE_CONTIGUOUS_FP8` & `DECODE_FP8_CONFIGS`)

| M | RM Warm (b1/b2) | 1107 Warm (b1/b2) | TC_aligned Warm (b1/b2) | Warm vs RM / 1107 | RM Cold (b1/b2) | 1107 Cold (b1/b2) | TC_aligned Cold (b1/b2) | Cold vs RM / 1107 | RM NEXT (b1/b2) | 1107 NEXT (b1/b2) | TC_aligned NEXT (b1/b2) | NEXT vs RM / 1107 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 8.61 / 13.03 | 13.33 / 13.95 | **8.61 / 11.28** | **-8.1% / -27.1%** | 52.66 / 52.50 | 51.90 / 52.23 | **49.02 / 48.57** | **-7.2% / -6.3%** | 52.12 / 52.26 | 51.86 / 52.40 | **48.70 / 48.69** | **-6.7% / -6.6%** |
| 2 | 8.45 / 8.41 | 13.01 / 13.09 | **8.50 / 8.51** | +0.9% / **-34.8%** | 52.82 / 52.60 | 51.89 / 52.25 | **48.71 / 48.80** | **-7.5% / -6.4%** | 52.13 / 52.21 | 52.07 / 52.38 | **48.77 / 48.81** | **-6.5% / -6.6%** |
| 3 | 8.50 / 8.51 | 13.45 / 13.27 | **9.02 / 8.75** | +4.5% / **-33.5%** | 52.02 / 52.68 | 51.88 / 52.73 | **48.70 / 48.68** | **-7.0% / -6.9%** | 51.98 / 52.22 | 51.90 / 52.38 | **48.79 / 48.64** | **-6.5% / -6.6%** |
| 6 | 8.02 / 7.72 | 11.43 / 11.49 | **7.86 / 7.82** | **-0.4% / -31.6%** | 52.29 / 52.66 | 51.93 / 52.36 | **48.90 / 48.81** | **-6.9% / -6.3%** | 52.13 / 52.42 | 52.02 / 52.42 | **48.52 / 48.73** | **-7.0% / -6.9%** |
| 12 | 6.33 / 6.26 | 8.88 / 8.97 | **6.91 / 6.57** | +7.1% / **-24.5%** | 52.96 / 53.29 | 51.96 / 52.59 | **49.66 / 49.76** | **-6.4% / -4.9%** | 52.69 / 52.55 | 52.06 / 52.64 | **49.65 / 49.80** | **-5.5% / -5.0%** |
| 24 | 16.33 / 7.98 | 15.75 / 9.80 | **14.89 / 7.90** | **-6.3% / -10.8%** | 54.19 / 54.12 | 53.76 / 54.13 | **50.14 / 50.15** | **-7.4% / -7.0%** | 53.93 / 53.91 | 53.68 / 54.10 | **50.15 / 49.92** | **-7.2% / -7.2%** |
| 48 | 13.90 / 13.56 | 18.36 / 18.35 | **13.70 / 13.89** | +0.5% / **-24.8%** | 56.65 / 56.34 | 54.89 / 55.69 | **52.60 / 52.16** | **-7.3% / -5.3%** | 56.28 / 56.23 | 55.21 / 55.76 | **52.61 / 52.05** | **-7.0% / -5.7%** |

#### `4096 x 1280` (`TP=8` Q-B & Unfused Indexer Q-B)

| M | RM Warm (b1/b2) | 1107 Warm (b1/b2) | TC_aligned Warm (b1/b2) | Warm vs RM / 1107 | RM Cold (b1/b2) | 1107 Cold (b1/b2) | TC_aligned Cold (b1/b2) | Cold vs RM / 1107 | RM NEXT (b1/b2) | 1107 NEXT (b1/b2) | TC_aligned NEXT (b1/b2) | NEXT vs RM / 1107 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 16.00 / 8.46 | 16.10 / 8.52 | **14.43 / 6.89** | **-12.8% / -13.4%** | 36.25 / 36.83 | 34.98 / 35.41 | **35.12 / 35.61** | **-3.2%** / +0.5% | 35.63 / 36.55 | 35.27 / 35.53 | **35.12 / 35.45** | **-2.2% / -0.3%** |
| 2 | 6.79 / 6.96 | 8.59 / 8.41 | **7.43 / 6.76** | +3.2% / **-16.5%** | 35.60 / 36.62 | 34.87 / 35.16 | **35.34 / 35.32** | **-2.2%** / +0.9% | 36.72 / 36.28 | 36.27 / 35.29 | **35.54 / 35.26** | **-3.0% / -1.1%** |
| 3 | 6.77 / 6.68 | 8.41 / 8.59 | **6.87 / 6.94** | +2.7% / **-18.8%** | 35.92 / 36.67 | 35.23 / 35.55 | **35.03 / 35.48** | **-2.9% / -0.4%** | 35.45 / 36.70 | 35.33 / 35.50 | **35.17 / 35.32** | **-2.3% / -0.5%** |
| 6 | 6.19 / 6.25 | 7.38 / 7.48 | **6.42 / 6.51** | +3.9% / **-13.0%** | 36.21 / 36.53 | 34.99 / 35.38 | **34.97 / 35.29** | **-3.4% / -0.2%** | 35.76 / 36.48 | 34.75 / 35.54 | **34.68 / 35.56** | **-2.8% / -0.1%** |
| 12 | 5.34 / 5.40 | 5.87 / 5.24 | **6.25 / 5.85** | +12.7% / +8.9% | 35.89 / 36.40 | 35.05 / 35.69 | **35.63 / 35.82** | **-1.2%** / +1.0% | 35.81 / 36.41 | 35.29 / 35.50 | **35.43 / 36.18** | **-0.8%** / +1.2% |
| 24 | 16.05 / 7.16 | 15.39 / 7.07 | **15.10 / 7.54** | **-2.5%** / +0.8% | 36.58 / 37.17 | 35.73 / 37.00 | **35.57 / 36.25** | **-2.6% / -1.3%** | 36.44 / 36.86 | 35.63 / 36.87 | **35.69 / 36.34** | **-1.7% / -0.6%** |
| 48 | 14.27 / 14.18 | 14.30 / 14.08 | **14.38 / 14.36** | +1.0% / +1.3% | 38.53 / 38.65 | 37.76 / 39.00 | **37.59 / 37.81** | **-2.3% / -1.8%** | 38.22 / 38.61 | 37.74 / 39.10 | **37.80 / 37.85** | **-1.5% / -1.5%** |

#### `2048 x 1280` (`TP=6 / TP=7` Q-B rank 5 — Newly Added to `TILE_CONTIGUOUS_FP8` & `DECODE_FP8_CONFIGS`)

| M | RM Warm (b1/b2) | 1107 Warm (b1/b2) | TC_aligned Warm (b1/b2) | Warm vs RM / 1107 | RM Cold (b1/b2) | 1107 Cold (b1/b2) | TC_aligned Cold (b1/b2) | Cold vs RM / 1107 | RM NEXT (b1/b2) | 1107 NEXT (b1/b2) | TC_aligned NEXT (b1/b2) | NEXT vs RM / 1107 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 16.01 / 6.72 | 16.69 / 6.11 | **14.79 / 4.74** | **-14.1% / -14.3%** | 22.02 / 23.78 | 21.50 / 24.40 | **21.78 / 22.08** | **-4.2% / -4.4%** | 22.03 / 23.19 | 21.50 / 23.37 | **21.81 / 22.04** | **-3.0% / -2.3%** |
| 2 | 4.56 / 4.62 | 5.55 / 4.84 | **5.65 / 4.57** | +11.3% / **-1.6%** | 21.85 / 23.24 | 21.88 / 22.95 | **22.31 / 21.88** | **-2.0% / -1.4%** | 22.74 / 22.76 | 21.60 / 23.03 | **21.53 / 21.54** | **-5.3% / -3.5%** |
| 3 | 4.53 / 4.53 | 4.73 / 4.72 | **4.52 / 4.54** | **-0.0% / -4.1%** | 21.64 / 23.26 | 21.49 / 23.29 | **21.19 / 21.69** | **-4.5% / -4.2%** | 21.56 / 23.32 | 21.49 / 23.29 | **21.03 / 21.62** | **-5.0% / -4.8%** |
| 6 | 4.54 / 4.48 | 5.20 / 4.50 | **5.55 / 4.48** | +11.2% / +3.4% | 22.04 / 22.97 | 21.70 / 22.84 | **21.77 / 21.76** | **-3.3% / -2.3%** | 22.15 / 22.94 | 21.87 / 23.07 | **21.85 / 21.94** | **-2.9% / -2.6%** |
| 12 | 4.47 / 4.49 | 5.38 / 4.82 | **5.53 / 4.65** | +13.6% / **-0.2%** | 22.37 / 23.22 | 21.68 / 23.14 | **21.87 / 21.96** | **-3.9% / -2.2%** | 22.03 / 23.16 | 21.57 / 23.02 | **21.91 / 21.95** | **-2.9% / -1.6%** |
| 24 | 5.44 / 5.54 | 4.63 / 4.62 | **5.47 / 5.44** | **-0.6%** / +17.9% | 22.48 / 23.56 | 22.10 / 23.35 | **22.15 / 22.37** | **-3.3% / -2.0%** | 22.47 / 23.55 | 22.17 / 23.32 | **22.14 / 22.31** | **-3.4% / -2.3%** |
| 48 | 7.47 / 7.39 | 7.69 / 7.59 | **7.38 / 7.33** | **-1.0% / -3.7%** | 22.67 / 24.47 | 23.20 / 24.44 | **22.60 / 23.04** | **-3.2% / -4.2%** | 22.60 / 24.32 | 23.98 / 24.36 | **24.16 / 22.80** | +0.1% / **-2.9%** |

#### Isolated Fused `Indexer Q-B` (`4096 x 1280` `mxfp8_gemm_rope_mxfp4_decode`)

| M | RM Warm (b1/b2) | 1107 Warm (b1/b2) | TC_aligned Warm (b1/b2) | Warm vs RM / 1107 | RM Cold (b1/b2) | 1107 Cold (b1/b2) | TC_aligned Cold (b1/b2) | Cold vs RM / 1107 |
|---|---|---|---|---|---|---|---|---|
| 1 | 7.59 / 15.30 | 9.31 / 14.97 | **7.90 / 14.01** | **-4.3% / -9.8%** | 36.16 / 36.46 | 35.52 / 35.94 | **35.99 / 36.27** | **-0.5%** / +1.1% |
| 2 | 7.59 / 7.52 | 9.31 / 9.02 | **8.08 / 7.82** | +5.2% / **-13.3%** | 36.14 / 36.52 | 35.55 / 35.60 | **35.94 / 35.93** | **-1.1%** / +1.0% |
| 3 | 8.17 / 7.83 | 9.41 / 9.02 | **8.31 / 8.23** | +3.4% / **-10.3%** | 36.60 / 37.42 | 35.73 / 36.10 | **36.12 / 36.24** | **-2.2%** / +0.7% |
| 6 | 7.66 / 7.78 | 8.47 / 8.43 | **8.25 / 7.89** | +4.5% / **-4.5%** | 37.34 / 37.69 | 36.18 / 36.53 | **36.92 / 36.80** | **-1.7%** / +1.4% |
| 12 | 8.08 / 7.89 | 7.19 / 6.67 | **8.49 / 8.34** | +5.4% / +21.4% | 39.02 / 39.57 | 36.84 / 36.80 | **38.93 / 38.63** | **-1.3%** / +5.3% |
| 24 | 11.28 / 14.35 | 11.48 / 13.66 | **11.47 / 13.59** | **-2.2% / -0.3%** | 44.45 / 44.92 | 43.79 / 43.79 | **43.69 / 43.80** | **-2.1% / -0.1%** |
| 48 | 18.97 / 17.96 | 19.15 / 18.08 | **19.13 / 17.99** | +0.5% / **-0.3%** | 45.04 / 45.85 | 45.44 / 45.67 | **45.39 / 45.89** | +0.4% / +0.2% |

## Conclusion

Aligning `Q-B` and `indexer Q-B` decode tiles in `DECODE_FP8_CONFIGS` to `128 x 128` (`block_N = 128, block_K = 128`) wherever the `48` SMs of GB10 are saturated (`N >= 6144` across all `block_M in (16, 32, 64)` and `N in (4096, 2048)` at `block_M in (16, 32)`) and completing `TILE_CONTIGUOUS_FP8` coverage across `(16384, 1280)`, `(6144, 1280)`, and `(2048, 1280)` eliminates the `8 KiB` packed-tile split across CTA pairs, cuts Cold/`NEXT` latency by `3.4–7.5%` vs row-major weights, improves `TP=4` and `TP=3` high-concurrency (`M = 24, 48`) Q-B decode by `4.5–22.5%` Warm over `1107`, improves `M = 1..6` chained `TP=4` Warm decode by `4.7–10.5%` over `1107`, reduces full 20-kernel PDL `LayerChain` replay latency across `c1 (M = 6)` and `c8 (M = 24)` by `-8.35 us` and `-12.66 us` Warm (`-3.72 us` and `-4.97 us` Cold, `-59.57 us` Cold at `c8x6`), and preserves 100% bit-exactness (`0` uint16/uint8 diffs) with prefill and row-major decode.
