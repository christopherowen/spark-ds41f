# Disk Engram 2-Bit Clock Replacement with Prefill Probationary Shield & Pinned Two-Plane Scales

- **Date**: 2026-11-08
- **Base commit**: `cc20ed18b` (`spark-ds41f-1107`, vLLM patch `0761` `2756478b60b90700b9918b457aeb38657da637b5`, tree `ed1bdb7a7684da33d403a055521ac0c4dd7ca670`)
- **Candidate patch**: `vllm/0762-perf-dsv41-engram-clock-pro-prefill-shield-and-pinned-scales.patch` (`d6d692e57bf6ac09240e0c351e8702f6f7aefdab`, tree `daec86bfc2457ec54f988124377e0072b8fd3ba5`)
- **Intended causal variable**: Host-DRAM 4-way set-associative Engram row cache replacement policy (`_row_plan.h`, `_reader.c`) and optional pinned scale shard lookup for two-plane tables (`_row_plan.h`, `_reader.c`, `rows.py`).

## Rationale

### 1. Pure FIFO Eviction Under Long-Context Prefill Floods (`0750`)

In `0750-Fuse-multi-CTA-disk-Engram-step-and-wire-packed-cach.patch`, the 4-way set-associative user-space row cache (`65,536` rows = `16,384` sets of 4 ways) in `vllm/models/deepseek_v4_1/engram_rows/_row_plan.h` used an unconditional round-robin pointer (`set->clock = (victim + 1) & 3`) on cache misses:

- `ple_plan()` never updated any per-way reference or frequency state when a cache hit occurred.
- `ple_cache_insert()` unconditionally evicted `way = set->clock` once all 4 ways in a set were populated, regardless of whether the victim way held a hot multi-hit decode n-gram row or a cold one-time prefill n-gram row.
- During a chunked long-context prefill (e.g. `32K–64K` prompt tokens = `8` chunks of `M = 4,096..8,192` tokens × `8` owned n-gram heads per TP3 rank = `262,144..524,288` lookups, or `4x–8x` the 65,536-entry cache capacity), streaming one-time prompt n-grams evicted 75%–100% of the active decode working set. Both interleaved `prefill_yield` decode steps and subsequent decode steps suffered cold NVMe `io_uring` misses on previously warm n-grams.

### 2. Zero-Overhead 2-Bit Clock Replacement + Prefill Probationary Shield (`0762`)

1. **Zero-extra-memory per-way usage counters & 8-byte aligned cache payload**:
   - In `ple_cache_set_t`, `uint8_t clock;` at offset 32 was followed by 7 bytes of implicit struct tail padding after `scales[4][8]`.
   - Placing `uint8_t usage[PLE_CACHE_WAYS]; uint8_t reserved[3];` immediately after `uint8_t clock;` uses that existing 7-byte padding (`sizeof(ple_cache_set_t) == 1096` bytes, zero extra memory) while aligning `weights[4][256]` (offset `40`) and `scales[4][8]` (offset `1064`) to 8-byte boundaries.
2. **Hit promotion (`usage` saturating at `3`)**:
   - Cache hits in `ple_plan()` and repeated in-batch hits in `ple_cache_insert()` increment `set->usage[w]` up to `PLE_CACHE_MAX_USAGE` (`3`).
3. **Chunk-epoch / prefill probationary classification (`M > 64`)**:
   - `ple_plan()` counts `live_col_lookups` (`id >= 0` on owned columns) and sets `reader->miss_is_prefill = !reader->is_prefetch && (live_col_lookups > (size_t)PLE_CACHE_MAX_TOKENS * (size_t)__builtin_popcount(reader->columns))` (`M > 64` active tokens).
   - Because `serve()` plans `step.early` and `reader->late` in separate `ple_plan()` calls, even mixed prefill+decode steps classify the early prompt chunk (`M_prefill > 64`) as probationary (`miss_is_prefill = true`) and the late decode tokens (`M_decode <= 64`) as protected (`miss_is_prefill = false`).
4. **Probationary insertion & protected decode working-set shield**:
   - New misses from prefill chunks (`is_prefill`) insert as probationary (`usage = 0`); new misses from decode / speculative verify (`M <= 64`) or warmup prefetch insert with `usage = 1`.
   - When selecting a victim in `ple_cache_insert()`, empty ways (`ids[w] < 0`) are filled first and probationary/decayed ways (`usage[w] == 0`) second.
   - When all 4 ways hold protected rows (`usage >= 1`), a prefill miss (`is_prefill`) claims at most one single-touch way (`usage == 1`) in that set as a probationary slot (`usage = 0`) without aging the remaining 3 ways and **never** evicts multi-hit decode ways (`usage >= 2`). One-time prefill sweeps cycle through the single probationary slot in each set while leaving the decode working set intact; recurring prompt n-grams that hit while in the probationary slot promote to `usage >= 1` and graduate into the protected working set.

### 3. Un-Packed `scale.bin` Pinning (`two_plane` Mode) vs Production `packed` Mode

- In `two_plane` layout (`weight.bin` at `256 B/row` + `scale.bin` at `8 B/row`), every cold or capacity cache miss previously emitted **two** separate 4 KiB `O_DIRECT` `io_uring` reads (`weight.bin` + `scale.bin`).
- Adding opt-in `pin_scales` (`SPARK_DS41F_ENGRAM_PIN_SCALES=1`) pins each shard's `scale.bin` in host memory (`reader->scale_shards[s]`). When enabled in `two_plane` mode, `ple_plan()` skips `scale.bin` `io_uring` fragments on cache misses (`1` read per miss instead of `2`), copies missed scales directly from `scale_shards[s]` in `ple_copy_pinned_miss_scales()`, and populates `set->scales[victim]` on cache insert.
- **Why `packed` (`row.bin`, `264 B/row`) remains the default production layout**: Full DeepSeek-V4.1-Flash Engram tables hold `384,000,000` rows per layer across 2 layers (`16` shards × `24` columns × `1,000,000` rows). Even at `8 B/row`, full-table `scale.bin` shards total `3.07 GiB` per layer (`6.14 GiB` across both layers, or `1.02 GiB/layer` per TP3 rank if column-compacted). Production `packed` mode interleaves each row's `256 B` weight and `8 B` scale contiguously into `row.bin` (`264 B`), achieving single-read misses with **zero** resident scale-table DRAM overhead. `pin_scales` provides an opt-in path for `two_plane` deployments without `row.bin`.

## Commands & Workload Identity

Executed on `sp7` (`GB10`, kernel `7.0.0-1019-nvidia-64k`, NVMe `/mnt/nvme` `O_DIRECT`) inside `spark-ds41f:1107` under `flock /tmp/gpu.lock` across 2 fresh container boots:

```bash
# Unit test verification (Boot 1 and Boot 2)
flock /tmp/gpu.lock docker run --rm --gpus=all --ipc=host \
  --cap-add=IPC_LOCK --ulimit=memlock=-1:-1 \
  --security-opt=seccomp=unconfined --security-opt=label=disable \
  --entrypoint python3 \
  -v /tmp/engram-epoch-exp/vllm/vllm/models/deepseek_v4_1/engram_rows:/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/engram_rows:ro \
  -v /tmp/engram-epoch-exp/vllm/tests/kernels/test_deepseek_v41_engram_rows.py:/opt/spark-ds41f/candidate/vllm/tests/kernels/test_deepseek_v41_engram_rows.py:ro \
  spark-ds41f:1107 -m pytest --noconftest -v /opt/spark-ds41f/candidate/vllm/tests/kernels/test_deepseek_v41_engram_rows.py

# 2-boot standard + interleaved prefill-flood benchmark (Boot 1 and Boot 2)
flock /tmp/gpu.lock docker run --rm --gpus=all --ipc=host \
  --cap-add=IPC_LOCK --ulimit=memlock=-1:-1 \
  --security-opt=seccomp=unconfined --security-opt=label=disable \
  --entrypoint python3 \
  -v /mnt/nvme:/mnt/nvme \
  -v /tmp/engram-epoch-exp/vllm/vllm/models/deepseek_v4_1/engram_rows:/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/engram_rows:ro \
  -v /tmp/engram-epoch-exp:/work:ro \
  spark-ds41f:1107 /work/bench_prefill_shield.py --bench-dir /mnt/nvme/engram_bench_shield
```

## Results

### 1. Correctness & Unit Tests (`receipts/pytest-boot1.txt`, `receipts/pytest-boot2.txt`)

- **Boot 1**: `52 passed in 22.88s`
- **Boot 2**: `52 passed in 22.75s`
- Added `test_row_cache_clock_pro_prefill_shield_preserves_decode_working_set` (verifying `100%` decode hit rate and bit-exact outputs across a `4 x M=512` one-time prefill flood over a 4-way set-associative cache) and `test_two_plane_pinned_scales_halves_read_calls_and_matches_reference` (verifying `read_calls` reduction from `16 -> 8` on cold miss and `0` on hot hit with bit-exact outputs).

### 2. Interleaved Prefill-Flood (`32K` Tokens = `4x` Cache Capacity) + Decode Working-Set Benchmark (`receipts/bench-cache-boot1.json`, `receipts/bench-cache-boot2.json`)

Workload (`run_prefill_flood_benchmark`, `table_rows = 1,050,000`, `65,536`-row cache, TP3 rank 0 owning `8` columns):
1. Warm up a `4,096`-row decode working set (`512` rows/column) over `48` decode steps (`usage >= 2`).
2. Stream `8` prefill chunks of `M = 4,096` tokens (`32,768` prefill tokens = `262,144` owned column lookups = **`4.0x` total cache capacity**), with `1` interleaved `prefill_yield` decode step after each chunk (`8` interleaved decode steps). Each prefill chunk also includes `16` recurring system-prompt header tokens.
3. Run `32` post-prefill decode steps on the decode working set.

#### A. Hot Decode Working Set (`100%` pre-prefill hit rate)

| Configuration | Batch (`M`) | Pre-Prefill Hit Rate / Mean (us) | Interleaved `prefill_yield` Hit Rate | Interleaved `read_calls/step` | Interleaved Mean / p50 (us) | Post-Prefill (32 steps) Hit Rate | Post-Prefill `read_calls/step` | Post-Prefill Mean / p50 (us) |
|---|---:|---|---:|---:|---|---:|---:|---|
| **`0750` FIFO reference (`packed_cache64k`)** | 8 | `100.0%` | `17.77%` (`91/512`) | `52.6` (`210.5 KiB`) | — | `23.44%` (`480/2048`) | `49.0` (`196.0 KiB`) | — |
| **`0762` Clock-Pro Shield (`packed_cache64k`)** | 8 | `100.0%` / **20.53** (`20.8, 20.3`) | **`100.0%`** (`512/512`) | **`0.0`** (`0.0 KiB`) | **65.81** / **39.46** (`36.7, 42.3`) | **`100.0%`** (`2048/2048`) | **`0.0`** (`0.0 KiB`) | **22.63** / **20.48** (`20.4, 20.6`) |
| **`0762` Clock-Pro Shield (`two_plane_cache64k`)** | 8 | `100.0%` / **20.67** (`20.5, 20.8`) | **`100.0%`** (`512/512`) | **`0.0`** (`0.0 KiB`) | **57.09** / **36.22** (`40.4, 32.1`) | **`100.0%`** (`2048/2048`) | **`0.0`** (`0.0 KiB`) | **22.84** / **20.47** (`20.5, 20.5`) |
| **`0762` Clock-Pro Shield (`two_plane_pin_scales_cache64k`)** | 8 | `100.0%` / **20.53** (`20.6, 20.5`) | **`100.0%`** (`512/512`) | **`0.0`** (`0.0 KiB`) | **32.09** / **31.69** (`32.2, 31.2`) | **`100.0%`** (`2048/2048`) | **`0.0`** (`0.0 KiB`) | **20.24** / **20.21** (`20.4, 20.1`) |
| **`0750` FIFO reference (`packed_cache64k`)** | 16 | `100.0%` | `19.14%` (`196/1024`) | `103.5` (`414.0 KiB`) | — | `41.06%` (`1682/4096`) | `75.4` (`301.8 KiB`) | — |
| **`0762` Clock-Pro Shield (`packed_cache64k`)** | 16 | `100.0%` / **20.85** (`21.1, 20.6`) | **`100.0%`** (`1024/1024`) | **`0.0`** (`0.0 KiB`) | **47.14** / **32.09** (`33.0, 31.2`) | **`100.0%`** (`4096/4096`) | **`0.0`** (`0.0 KiB`) | **20.53** / **20.26** (`20.4, 20.1`) |
| **`0762` Clock-Pro Shield (`two_plane_cache64k`)** | 16 | `100.0%` / **21.11** (`21.0, 21.2`) | **`100.0%`** (`1024/1024`) | **`0.0`** (`0.0 KiB`) | **40.53** / **32.78** (`33.0, 32.6`) | **`100.0%`** (`4096/4096`) | **`0.0`** (`0.0 KiB`) | **20.73** / **20.62** (`20.7, 20.5`) |
| **`0762` Clock-Pro Shield (`two_plane_pin_scales_cache64k`)** | 16 | `100.0%` / **21.14** (`21.4, 20.8`) | **`100.0%`** (`1024/1024`) | **`0.0`** (`0.0 KiB`) | **42.73** / **32.70** (`31.8, 33.6`) | **`100.0%`** (`4096/4096`) | **`0.0`** (`0.0 KiB`) | **20.70** / **20.67** (`20.9, 20.5`) |

#### B. Zipfian-85 Decode Working Set (`85%` hot working set + `15%` cold tail)

| Configuration | Batch (`M`) | Pre-Prefill Hit Rate / Mean (us) | Interleaved `prefill_yield` Hit Rate | Interleaved `read_calls/step` | Post-Prefill (32 steps) Hit Rate | Post-Prefill `read_calls/step` | Post-Prefill Mean (us) |
|---|---:|---|---:|---:|---:|---:|---|
| **`0750` FIFO reference (`packed_cache64k`)** | 8 | `85.45%` | `16.21%` (`83/512`) | `53.6` (`214.5 KiB`) | `16.99%` (`348/2048`) | `53.1` (`212.5 KiB`) | — |
| **`0762` Clock-Pro Shield (`packed_cache64k`)** | 8 | `85.45%` / **66.91** (`67.2, 66.6`) | **`84.38%`** (`432/512`) | **`10.0`** (`40.0 KiB`) | **`85.74%`** (`1756/2048`) | **`9.1`** (`36.5 KiB`) | **67.47** (`67.9, 67.0`) |
| **`0762` Clock-Pro Shield (`two_plane_cache64k`)** | 8 | `85.45%` / **85.94** (`86.5, 85.4`) | **`84.38%`** (`432/512`) | **`20.0`** (`80.0 KiB`) | **`85.74%`** (`1756/2048`) | **`18.2`** (`73.0 KiB`) | **86.38** (`86.1, 86.7`) |
| **`0762` Clock-Pro Shield (`two_plane_pin_scales_cache64k`)** | 8 | `85.45%` / **69.16** (`69.8, 68.5`) | **`84.38%`** (`432/512`) | **`10.0`** (`40.0 KiB`) | **`85.74%`** (`1756/2048`) | **`9.1`** (`36.5 KiB`) | **68.09** (`68.3, 67.8`) |
| **`0750` FIFO reference (`packed_cache64k`)** | 16 | `84.81%` | `14.65%` (`150/1024`) | `109.2` (`437.0 KiB`) | `30.71%` (`1258/4096`) | `88.7` (`354.8 KiB`) | — |
| **`0762` Clock-Pro Shield (`packed_cache64k`)** | 16 | `84.81%` / **88.92** (`88.9, 89.0`) | **`86.04%`** (`881/1024`) | **`17.9`** (`71.5 KiB`) | **`85.62%`** (`3507/4096`) | **`18.4`** (`73.6 KiB`) | **85.91** (`86.5, 85.3`) |
| **`0762` Clock-Pro Shield (`two_plane_cache64k`)** | 16 | `84.81%` / **122.46** (`123.5, 121.5`) | **`86.04%`** (`881/1024`) | **`35.8`** (`143.0 KiB`) | **`85.62%`** (`3507/4096`) | **`36.8`** (`147.2 KiB`) | **124.22** (`123.4, 125.0`) |
| **`0762` Clock-Pro Shield (`two_plane_pin_scales_cache64k`)** | 16 | `84.81%` / **90.51** (`91.0, 90.1`) | **`86.04%`** (`881/1024`) | **`17.9`** (`71.5 KiB`) | **`85.62%`** (`3507/4096`) | **`18.4`** (`73.6 KiB`) | **89.01** (`88.9, 89.1`) |

- **Recurring prompt header graduation during prefill**: Under `0750` FIFO, the `16` recurring system-prompt header tokens per prefill chunk are evicted before the next chunk arrives (`prefill_hit_rate = 0.0022`). Under `0762`, they hit in the probationary slot on chunk 2, promote to `usage >= 1`, and remain cached across all subsequent chunks (`prefill_hit_rate = 0.0033`, `+50%` intra-prefill cache hits).
- **Two-plane pinned scales on Zipfian-85 decode**: `two_plane_pin_scales_cache64k` halves post-prefill `read_calls_per_step` (`18.2 -> 9.1` at `M=8`, `36.8 -> 18.4` at `M=16`) and cuts step latency from `86.38 -> 68.09 us` (**-21.2%**) at `M=8` and `124.22 -> 89.01 us` (**-28.3%**) at `M=16`, matching `packed_cache64k` (`67.47 us` at `M=8`, `85.91 us` at `M=16`).

### 3. Standard `run_cache_and_packed_benchmark` Across 2 Boots (`receipts/bench-cache-boot1.json`, `receipts/bench-cache-boot2.json`)

Mean step latency in microseconds (`Boot 1, Boot 2`) at `table_rows = 200,000`:

| Regime | Batch (`M`) | `two_plane_cache64k` (us) | `two_plane_pin_scales_cache64k` (us) | `packed_cache64k` (us) | `two_plane` vs `pin_scales` `read_calls/step` |
|---|---:|---:|---:|---:|---|
| **Cold Miss** (`0%` hit) | 1 | 27.67 (`27.7, 27.6`) | **25.84** (`26.1, 25.5`) | **25.81** (`25.8, 25.8`) | `16.0 -> 8.0` (**-50%**) |
| **Cold Miss** (`0%` hit) | 4 | 39.20 (`39.6, 38.8`) | **38.12** (`38.2, 38.0`) | **29.74** (`29.9, 29.5`) | `16.0 -> 8.0` (**-50%**) |
| **Cold Miss** (`0%` hit) | 8 | 49.49 (`49.6, 49.4`) | **44.66** (`44.6, 44.7`) | **33.48** (`33.3, 33.6`) | `16.0 -> 8.0` (**-50%**) |
| **Cold Miss** (`0%` hit) | 16 | 74.16 (`75.3, 73.0`) | **64.61** (`64.6, 64.6`) | **47.82** (`48.6, 47.0`) | `16.0 -> 8.0` (**-50%**) |
| **Cold Miss** (`0%` hit) | 32 | 127.63 (`130.7, 124.5`) | **124.16** (`123.7, 124.6`) | **102.79** (`103.5, 102.1`) | `16.0 -> 15.8` |
| **Zipfian Working Set** (`~95%` hit) | 1 | 21.37 (`21.6, 21.1`) | **19.31** (`19.2, 19.4`) | **20.38** (`20.4, 20.4`) | `1.12 -> 0.56` (**-50%**) |
| **Zipfian Working Set** (`~94%` hit) | 4 | 21.74 (`21.8, 21.7`) | **20.65** (`20.6, 20.7`) | **20.75** (`20.8, 20.7`) | `5.88 -> 2.94` (**-50%**) |
| **Zipfian Working Set** (`~94%` hit) | 8 | 24.47 (`24.8, 24.1`) | **22.97** (`22.6, 23.3`) | **22.58** (`22.3, 22.9`) | `9.75 -> 4.81` (**-51%**) |
| **Zipfian Working Set** (`~95%` hit) | 16 | 30.89 (`30.9, 30.9`) | **29.74** (`29.9, 29.6`) | **27.93** (`27.7, 28.2`) | `16.66 -> 9.62` (**-42%**) |
| **Zipfian Working Set** (`~95%` hit) | 48 | 41.36 (`41.4, 41.3`) | **38.80** (`39.0, 38.6`) | **32.78** (`33.0, 32.6`) | `15.72 -> 14.34` |
| **Zipfian Working Set** (`~95%` hit) | 128 | 66.75 (`66.8, 66.7`) | **64.72** (`64.6, 64.9`) | **44.11** (`44.1, 44.1`) | `15.97 -> 15.97` |
| **Hot Cache** (`100%` hit) | 1 | 20.63 (`20.9, 20.3`) | **19.72** (`19.7, 19.8`) | **20.17** (`20.2, 20.1`) | `0.0 -> 0.0` |
| **Hot Cache** (`100%` hit) | 4 | 20.10 (`20.2, 20.0`) | **19.99** (`20.0, 19.9`) | **19.95** (`20.0, 19.9`) | `0.0 -> 0.0` |
| **Hot Cache** (`100%` hit) | 8 | 20.57 (`20.8, 20.3`) | **20.14** (`20.2, 20.1`) | **20.38** (`20.5, 20.3`) | `0.0 -> 0.0` |
| **Hot Cache** (`100%` hit) | 16 | 20.39 (`20.5, 20.3`) | **20.40** (`20.3, 20.5`) | **20.37** (`20.5, 20.2`) | `0.0 -> 0.0` |
| **Hot Cache** (`100%` hit) | 32 | 21.02 (`21.3, 20.7`) | **20.88** (`20.8, 20.9`) | **20.76** (`20.9, 20.6`) | `0.0 -> 0.0` |
| **Hot Cache** (`100%` hit) | 64 | 21.74 (`21.8, 21.6`) | **21.67** (`21.7, 21.6`) | **21.66** (`21.6, 21.7`) | `0.0 -> 0.0` |
| **Hot Cache** (`100%` hit) | 128 | 23.62 (`23.7, 23.5`) | **23.54** (`23.6, 23.4`) | **23.57** (`23.7, 23.5`) | `0.0 -> 0.0` |

## Memory Observations

- **Host row cache (`ple_cache_set_t`)**: `sizeof(ple_cache_set_t)` remains `1,096` bytes (`16,384` sets × `1,096 B` = `17.125 MiB` total for `65,536` rows). Placing `uint8_t usage[4]; uint8_t reserved[3];` after `uint8_t clock;` uses the existing 7-byte struct padding with zero extra allocation and aligns `weights[4][256]` (offset `40`) and `scales[4][8]` (offset `1064`) to 8-byte boundaries.
- **Two-plane scale pinning (`pin_scales`)**: Opt-in (`SPARK_DS41F_ENGRAM_PIN_SCALES=1`, default `0`). Requires `8 B/row` when enabled; default `packed` mode (`264 B` row-interleaved `row.bin`) requires `0 B` of resident scale shards.

## Conclusion

Replacing pure FIFO eviction in the 4-way set-associative disk Engram row cache with a zero-extra-memory 2-bit Clock replacement policy and prefill probationary shield (`M > 64`) preserves **100.0% of a warm decode working set** across a `32K`-token (`4x` cache capacity) prefill flood (compared to **17.8%–19.1% interleaved** and **23.4%–41.1% post-prefill** hit rates under `0750` FIFO), maintains **84.4%–86.0%** hit rate on a Zipfian-85 decode workload during and after the prefill flood (vs **14.6%–30.7%** under `0750` FIFO), and increases intra-prefill recurring header hits by **+50%**. Opt-in `scale.bin` pinning for two-plane tables halves miss `io_uring` read calls (`16 -> 8`) and reduces Zipfian-85 decode latency by **21.2%–28.3%**.
