# Single-Plane 264-Byte Row-Interleaved & Rank-Compacted Disk Engram Rows

## Hypothesis

In two-plane disk Engram storage (`table_memory="disk"`), each lookup in `ple_plan` (`vllm/models/deepseek_v4_1/engram_rows/_row_plan.h`) emits two separate disk fragments across two non-adjacent file extents:

1. `weight[id]` (`256 B` E4M3 values at `weight_offset + id * 256`)
2. `scale[id]` (`8 B` E8M0 block scales at `scale_offset + id * 8`)

Because `O_DIRECT` reads at `4096 B` granularity, reading `8 B` scale entries from a separate plane doubles the fragment count, radix-sort work (`plan_ns`), and nearly doubles random 4 KiB NVMe IOPS (`read_calls` / `unique_blocks`) on decode and small-chunk steps (`16–512` tokens). In addition, under column-sharded TP (`column_shard`, where rank $r$ reads columns $r, r + \text{tp}, \dots$), each rank's owned columns are non-contiguous in the global 24-column table, reducing spatial locality across adjacent column ranges.

Interleaving each row's `256 B` E4M3 weights and `8 B` E8M0 scales into a single `264 B` row record (`[weight: 256B | scale: 8B]`) and compacting only the rank's owned hash columns into a dense per-rank file (`id -> id - col_delta[col]`) should:

- Halve the number of `ple_fragment_t` records emitted and radix-sorted in `ple_plan` (`1` fragment per owned lookup instead of `2`, or `2` only when a `264 B` row straddles a `4096 B` page boundary).
- Eliminate separate `4 KiB` `O_DIRECT` page reads for the `8 B` scale plane, cutting random NVMe `pread` calls and 4 KiB blocks by ~40–50% on small steps and reducing planning and read latency across all batch sizes.
- Stream-pack a rank's compacted file from the checkpoint's weight and scale planes using `O_DIRECT` + `POSIX_FADV_DONTNEED` (`pack_engram_columns`) so cold-cache preparation completes at NVMe sequential bandwidth without polluting the OS page cache.

## Implementation

Patch: `0001-engram-packed-rank-rows.patch` (against `vllm` commit `1f18be14470339c1227ed0432b896a8ca09811b5`).

1. **Planner (`vllm/models/deepseek_v4_1/engram_rows/_row_plan.h`)**:
   - Added `packed` (`264 B` stride) and optional per-column compaction tables (`has_col_map`, `col_start[24]`, `col_end[24]`, `col_delta[24]`) to `ple_plan_t`, along with `ple_add_packed_source`.
   - When `reader->packed` is active, `ple_plan` translates global row `id` for owned column `col = i % 24` to physical row `phys_id = id - reader->col_delta[col]` (after validating `id in [col_start[col], col_end[col])`) and emits a single `264 B` read request at `offset + phys_id * 264`.
2. **Scatter & Zero-Page-Cache Packer (`vllm/models/deepseek_v4_1/engram_rows/_reader.c`)**:
   - Updated `ple_scatter` so that when `reader->plan.packed` is set, any page-split slice `[row_off, row_off + len)` of a `264 B` row copies `[0..256)` into `reader->weights + lookup * 256` and `[256..264)` into `reader->scales + lookup * 8`.
   - Added `pack_columns` (`py_pack_columns`), which reads each owned column range from the weight and scale files in `4 MiB` chunks, interleaves `256 B` weights + `8 B` scales in L2/RAM into `O_DIRECT`-aligned buffers, writes to a temporary file, calls `fsync` + `posix_fadvise(..., POSIX_FADV_DONTNEED)` on all three file descriptors, and atomically `rename`s the output file into place.
   - Exposed `read_calls`, `unique_blocks`, and `read_bytes` in `status(reader)`.
3. **Python Bindings & Model Loader (`rows.py`, `common/engram.py`)**:
   - Added `PACKED_ROW_BYTES = 264`, `column_compaction_map`, `pack_engram_columns`, and `EngramDiskRows.add_packed_source`.
   - Added `SPARK_DS41F_ENGRAM_PACKED_DIR` (`ENGRAM_PACKED_DIR`) in `vllm/models/deepseek_v4_1/common/engram.py`: when set on a column-sharded disk table, `ParallelEngramEmbedding.open_disk_rows()` packs (or reuses) a content-addressed `.packed` file for `(layer_id, tp_size, tp_rank, column_mask)` and registers it via `rows.add_packed_source(...)`.

## Verification & Benchmark Results (`dgx4`, `spark-ds41f:1030`)

### 1. Unit & Model Tests

- `pytest tests/kernels/test_deepseek_v41_engram_rows.py`: **30 passed** (21 existing + 9 new parameterized tests covering page-boundary splits at offsets `0`, `4093`, `4096 - 256`, and `4096 - 260`, plus TP3 and TP4 rank-compacted packing and early-ID verification).
- `pytest tests/models/test_deepseek_v4_1_engram.py -k "test_disk_rows_pack_and_read_rank_compacted_files or test_disk_rows_read_whole_table_sources"`: **4 passed**.

### 2. NVMe `O_DIRECT` Microbenchmark (`/cache` on `dgx4`, 262,144-row table)

Packing throughput (`pack_engram_columns` with `O_DIRECT` + `POSIX_FADV_DONTNEED`):
- TP3 rank shard (`23.07 MB`): **10.09 ms** (**2.29 GB/s**), `/proc/meminfo` `Cached` delta: **+128 KB** (zero page-cache pollution).

Per-step `O_DIRECT` I/O and latency (median across 5 steps after warmup, bit-exact BF16 output verified):

| Config | Tokens (`T`) | Mode | `read_calls` | `unique_blocks` (4 KiB) | `read_bytes` | `plan_ns` (µs) | `read` (µs) |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| **TP4** (6 cols/rank) | 16 | 2-plane | 73 | 85 | 348,160 | 6.7 | 129.0 |
| **TP4** (6 cols/rank) | 16 | **Packed 264B** | **42 (-42.5%)** | **43 (-49.4%)** | **176,128 (-49.4%)** | **3.3 (-50.7%)** | **109.7 (-15.0%)** |
| **TP4** (6 cols/rank) | 128 | 2-plane | 319 | 469 | 1,921,024 | 30.9 | 726.0 |
| **TP4** (6 cols/rank) | 128 | **Packed 264B** | **306 (-4.1%)** | **353 (-24.7%)** | **1,445,888 (-24.7%)** | **26.7 (-13.6%)** | **585.4 (-19.4%)** |
| **TP4** (6 cols/rank) | 2048 | 2-plane | 756 | 3,220 | 13,189,120 | 435.4 | 2,332.9 |
| **TP4** (6 cols/rank) | 2048 | **Packed 264B** | **783 (+3.6%)** | **3,136 (-2.6%)** | **12,845,056 (-2.6%)** | **244.3 (-43.9%)** | **2,209.3 (-5.3%)** |
| **TP4** (6 cols/rank) | 16384 | 2-plane | 271 | 4,235 | 17,346,560 | 4,348.8 | 7,483.9 |
| **TP4** (6 cols/rank) | 16384 | **Packed 264B** | **264 (-2.6%)** | **4,224 (-0.3%)** | **17,301,504 (-0.3%)** | **1,973.9 (-54.6%)** | **6,548.8 (-12.5%)** |
| **TP3** (8 cols/rank) | 16 | 2-plane | 89 | 103 | 421,888 | 6.7 | 109.5 |
| **TP3** (8 cols/rank) | 16 | **Packed 264B** | **53 (-40.4%)** | **55 (-46.6%)** | **225,280 (-46.6%)** | **2.7 (-59.7%)** | **66.8 (-39.0%)** |
| **TP3** (8 cols/rank) | 128 | 2-plane | 403 | 605 | 2,478,080 | 39.8 | 841.6 |
| **TP3** (8 cols/rank) | 128 | **Packed 264B** | **382 (-5.2%)** | **450 (-25.6%)** | **1,843,200 (-25.6%)** | **19.8 (-50.3%)** | **779.5 (-7.4%)** |
| **TP3** (8 cols/rank) | 2048 | 2-plane | 1,039 | 4,255 | 17,428,480 | 596.3 | 2,963.0 |
| **TP3** (8 cols/rank) | 2048 | **Packed 264B** | **1,077 (+3.7%)** | **4,154 (-2.4%)** | **17,014,784 (-2.4%)** | **311.8 (-47.7%)** | **2,736.7 (-7.6%)** |
| **TP3** (8 cols/rank) | 16384 | 2-plane | 360 | 5,641 | 23,105,536 | 9,039.1 | 13,448.1 |
| **TP3** (8 cols/rank) | 16384 | **Packed 264B** | **352 (-2.2%)** | **5,632 (-0.2%)** | **23,068,672 (-0.2%)** | **2,546.0 (-71.8%)** | **7,743.3 (-42.4%)** |
