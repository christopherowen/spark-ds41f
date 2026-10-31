# Evict Redundant VFS Page Cache via `POSIX_FADV_DONTNEED` in Engram Disk and RAM Readers

## Problem

On GB10 (`dgx4`), the Grace CPU and Blackwell GPU share a single `128 GiB` unified LPDDR5X physical memory pool across model weights, KV cache, `cudaHostAllocMapped` page-locked allocations, and the Linux kernel's VFS page cache (`Inactive(file)` measured at `73.3 GiB` on `dgx4`).

DeepSeek-V4.1's Engram tables total `34.4 GiB` across layers 1 and 15. Previously, two Engram file paths retained redundant copies of the Engram checkpoint ranges in the Linux VFS page cache:

1. **Mapped-Host / Device Table Loading (`_read_table_rows` in `vllm/models/deepseek_v4_1/common/engram.py`)**:
   - Reads the rank's Engram shard into `MappedHostAllocation` (`table_memory="ram"`) or device-staged CPU memory (`table_memory="device"`) via `open(source.path, "rb", buffering=0)` and `checkpoint.readinto(...)`.
   - While `buffering=0` disables Python's user-space `io.BufferedReader`, the underlying `read(2)` syscalls still populate the Linux kernel VFS page cache with `100%` of the read range and never call `posix_fadvise(..., POSIX_FADV_DONTNEED)`, leaving a duplicate multi-gigabyte copy in unified LPDDR5X memory alongside the pinned `MappedHostAllocation`.
2. **Disk-Offloaded Engram Reader (`_row_plan.h` and `_reader.c` in `vllm/models/deepseek_v4_1/engram_rows/`)**:
   - Although `ple_add_source` opens the Engram files with `O_DIRECT`, any clean pages already resident in the VFS page cache (from checkpoint download, inspection, or earlier buffered reads) remain in the kernel's `xarray` page cache (`Inactive(file)`), both consuming unified LPDDR5X memory (triggering `kswapd` reclamation and `2 MiB` THP fragmentation) and adding kernel page-cache lookup overhead along the `O_DIRECT` I/O path.

## Solution

Evict VFS page-cache pages for Engram table ranges via `posix_fadvise(..., POSIX_FADV_DONTNEED)` by default (`SPARK_DS41F_ENGRAM_DROP_PAGE_CACHE=1`, opt-out via `SPARK_DS41F_ENGRAM_DROP_PAGE_CACHE=0`):

1. **`_read_table_rows` (`vllm/models/deepseek_v4_1/common/engram.py`)**:
   - Issue `os.posix_fadvise(fd, aligned, done_aligned - aligned, os.POSIX_FADV_DONTNEED)` as `8 MiB` read chunks complete and over the full `[aligned_start, aligned_end)` shard range at completion so buffered table loads leave `0 B` of duplicate VFS page cache behind.
2. **`ple_add_source` (`vllm/models/deepseek_v4_1/engram_rows/_row_plan.h`)**:
   - Issue `posix_fadvise(target_fd, range_start, range_end - range_start, POSIX_FADV_DONTNEED)` when registering each Engram weight and scale source file so pre-existing VFS page-cache pages for the table shard are reclaimed before step 1.
3. **`serve` (`vllm/models/deepseek_v4_1/engram_rows/_reader.c`)**:
   - Issue `drop_plan_page_cache(reader)` **after** `__atomic_store_n(&reader->control[CTRL_ROWS_SEQ], step.sequence, __ATOMIC_RELEASE)` (so the GPU is already unblocked and gathering in parallel) and record `fadvise_bytes` in `rows.stats()`.

## Patch

- [`0001-engram-fadvise-dontneed-cold-reads.patch`](0001-engram-fadvise-dontneed-cold-reads.patch) — formatted against `vllm` commit `1f18be14470339c1227ed0432b896a8ca09811b5`.

## Verification & Benchmark (`dgx4`, GB10 + NVMe `/cache`)

### Correctness

Executed inside `spark-ds41f:1030` on `dgx4`:

```bash
pytest -o addopts="" --noconftest -q tests/kernels/test_deepseek_v41_engram_rows.py -k "test_gather_matches_b12x_lookup_bits or test_fadvise_dontneed"
```

- **`5 passed`** (verifying bit-exact `engram_rows_gather` output and `mincore` page-cache eviction on both `EngramDiskRows` source registration/steps and `_read_table_rows`).

### VFS Page-Cache Residency (`mincore`) & `EngramDiskRows` Step Latency (`dgx4`)

Measured on `dgx4` (`/cache`, `64 MiB` weight shard + `2 MiB` scale shard, `262,144` rows, `30` timed steps of `16 × 24 = 384` random lookups after `5` warmup steps):

| Path | Policy (`SPARK_DS41F_ENGRAM_DROP_PAGE_CACHE`) | Resident VFS Page Cache After Load / Reg (`mincore`) | Step Median (`µs`) | Step `p95` (`µs`) | `fadvise_bytes` |
| --- | --- | ---: | ---: | ---: | ---: |
| `_read_table_rows` (`ram`/`device` loader) | `"0"` (Baseline: retain page cache) | `65,536.0 KiB` (`64.00 MiB`, `100%`) | — | — | — |
| **`_read_table_rows` (`ram`/`device` loader)** | **`"1"` (`POSIX_FADV_DONTNEED`)** | **`0.0 KiB` (`0.00 MiB`, `-100%`)** | — | — | — |
| `EngramDiskRows` (`disk` `io_uring` reader) | `"0"` (Baseline: retain page cache) | `65,536.0 KiB` (`64.00 MiB`, `100%`) | `1,122.4` | `1,186.7` | `0` |
| **`EngramDiskRows` (`disk` `io_uring` reader)** | **`"1"` (`POSIX_FADV_DONTNEED`)** | **`0.0 KiB` (`0.00 MiB`, `-100%`)** | **`929.2` (`-17.2%`)** | **`974.3` (`-17.9%`)** | `90,497,024` |
