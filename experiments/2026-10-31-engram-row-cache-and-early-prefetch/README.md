# Bounded User-Space 4-Way Engram Row Cache & Pre-Consumed Early Read Overlap

## Hypothesis

Two bottlenecks remain in disk Engram row service (`vllm/models/deepseek_v4_1/engram_rows/`):

1. **Post-Consumed Early Read Serialization (`read_early`)**:
   - In `_reader.c`, `read_early` waits for the GPU to finish gathering the previous step (`wait_for_gpu(reader, CTRL_CONSUMED_SEQ, previous, ...)`) *before* planning (`ple_plan`) or submitting any `io_uring` `O_DIRECT` reads (`ple_uring_run`) for the current step's host-hashed early IDs (`step.early`), because `ple_scatter` writes directly into the mapped `reader->weights` and `reader->scales` buffers.
   - As a result, NVMe `O_DIRECT` I/O for step $N$ cannot overlap with the GPU's execution prior to step $N-1$'s gather.
   - By reading `step.early` into private cached host staging buffers (`reader->early_weights`, `reader->early_scales`) *before* `wait_for_gpu(reader, CTRL_CONSUMED_SEQ, previous, ...)` and copying the owned slots into mapped `reader->weights` / `reader->scales` only *after* `CTRL_CONSUMED_SEQ` advances, NVMe reads overlap with the previous step's GPU compute while keeping the mapped buffers untouched until the previous gather finishes.
2. **Repeated `O_DIRECT` Reads for Hot N-Gram Rows**:
   - Because `O_DIRECT` bypasses the Linux page cache (protecting unified memory on GB10 from unbounded 4 KiB page-cache growth), repeated 2-gram, 3-gram, and 4-gram row lookups across steps and shared prompt prefixes re-issue `4 KiB` NVMe reads on every step.
   - A bounded, fixed-size user-space 4-way set-associative row cache (`cache_rows`, configured via `SPARK_DS41F_ENGRAM_ROW_CACHE_ROWS`, e.g., `65536` rows = `16,384` sets $\times$ `4` ways $\times$ `264 B` $\approx 17.3\text{ MB}$ per table) can serve hot rows directly from cached host RAM without emitting disk fragments or touching `io_uring`, and without ever reading back from write-combined (`cudaHostAllocWriteCombined`) memory.

## Implementation

Patch: `0001-engram-row-cache-and-early-prefetch.patch` (against `vllm` commit `1f18be14470339c1227ed0432b896a8ca09811b5`).

1. **Pre-Consumed Early Read Overlap (`vllm/models/deepseek_v4_1/engram_rows/_reader.c`)**:
   - Allocated private host staging buffers `early_weights` (`max_lookups * 256`) and `early_scales` (`max_lookups * 8`) on `reader_t`.
   - Updated `read_early` to execute `read_rows_into(reader, step.early, step.count, reader->early_weights, reader->early_scales)` *before* `wait_for_gpu(reader, CTRL_CONSUMED_SEQ, previous, ...)`.
   - Once `CTRL_CONSUMED_SEQ` reaches `previous`, `read_early` copies only the owned known early slots from `early_weights`/`early_scales` into the mapped `reader->weights`/`reader->scales` buffers.
2. **Bounded 4-Way Set-Associative Row Cache (`_row_plan.h`, `_reader.c`, `rows.py`, `common/engram.py`)**:
   - Added `ple_cache_set_t` (`4` ways per set, clock replacement, power-of-two set table) to `ple_plan_t`.
   - On a cache hit in `ple_plan`, the cached `256 B` weight and `8 B` scale are copied directly into the destination buffer (`cache_hits++`) and no disk fragments are emitted.
   - On a cache miss (`cache_misses++`), `ple_plan` reserves a way in the target set for the current `plan_epoch` (invalidating `ids[way] = -1` while the read is in flight), `ple_scatter` populates the way directly from the cached `io_uring` bounce buffer (never reading from write-combined memory), and `read_rows_into` commits `ids[way] = id` only after `ple_uring_run` succeeds.
   - Added non-blocking `rows.prefetch(ids)` (`py_prefetch`) to warm the user-space row cache on the worker thread without advancing or modifying the GPU step buffers or control block.
   - Added `SPARK_DS41F_ENGRAM_ROW_CACHE_ROWS` (`ENGRAM_ROW_CACHE_ROWS`) in `vllm/models/deepseek_v4_1/common/engram.py`.

## Verification & Benchmark Results (`dgx4`, `spark-ds41f:1030`)

### 1. Unit Tests

- `pytest tests/kernels/test_deepseek_v41_engram_rows.py -k "early or cache or prefetch"`: **8 passed** (including `test_early_rows_wait_for_the_previous_gather`, `test_early_rows_complete_disk_reads_before_previous_gather_finishes`, `test_user_space_row_cache_hits_evicts_and_matches_reference`, and `test_background_prefetch_warms_row_cache_without_touching_step_buffers`).

### 2. Zipfian Multi-Step `O_DIRECT` Benchmark (`/cache` on `dgx4`, 262,144-row table, TP4)

Across 9 consecutive steps following a 1-step warmup under a Zipfian ($\alpha = 1.15$) lookup distribution on `/cache`:

| Config | Tokens (`T`) | `cache_rows` | Cache Footprint | Hit Rate | `read_calls` / step | `unique_blocks` / step | `early_read` (µs) | Step Wall (µs) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **TP4** (6 cols/rank) | 16 | 0 | 0 MB | 0.0% | 126.0 | 141.4 | 367.8 | 402.3 |
| **TP4** (6 cols/rank) | 16 | **65,536** | **17.3 MB** | **34.1%** | **110.7 (-12.1%)** | **119.6 (-15.4%)** | **188.2 (-48.8%)** | **195.0 (-51.5%)** |
| **TP4** (6 cols/rank) | 128 | 0 | 0 MB | 0.0% | 526.2 | 741.0 | 901.2 | 934.5 |
| **TP4** (6 cols/rank) | 128 | **65,536** | **17.3 MB** | **50.8%** | **473.6 (-10.0%)** | **620.3 (-16.3%)** | **738.8 (-18.0%)** | **760.0 (-18.7%)** |
| **TP4** (6 cols/rank) | 2048 | 0 | 0 MB | 0.0% | 3,109.9 | 4,934.8 | 5,145.6 | 5,176.1 |
| **TP4** (6 cols/rank) | 2048 | **65,536** | **17.3 MB** | **66.5%** | **2,761.7 (-11.2%)** | **3,973.4 (-19.5%)** | **4,707.4 (-8.5%)** | **4,776.0 (-7.7%)** |
