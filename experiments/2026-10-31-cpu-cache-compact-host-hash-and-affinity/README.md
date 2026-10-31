# Grace CPU L2 Cache Optimization: Compact `int32` `HostEngramHash.token_map` & Max-L2 Core Affinity

## Problem

On GB10 (`dgx4`), the 20-core Grace CPU has a heterogeneous CPU core and private L2 cache hierarchy across two 10-core clusters:

| Core Group | CPUs | Architecture | Private L2 Cache (`cache/index2/size`) |
| --- | --- | --- | ---: |
| Little cluster | `cpu0..4`, `cpu10..14` (10 cores) | Arm Cortex-A725 | **`512 KiB`** |
| Big cluster | `cpu5..9`, `cpu15..19` (10 cores) | Arm Cortex-X925 | **`2,048 KiB` (`2 MiB`)** |

Two CPU-side Engram components interact directly with this asymmetric L2 hierarchy on every step:

1. **`HostEngramHash.token_map` (`vllm/models/deepseek_v4_1/engram_rows/host_hash.py`)**:
   - Stored `token_map` (`129,280` vocabulary entries) as `np.int64` (`1,034,240 B = 1,010 KiB`), even though every compressed token ID fits in `[-1, 129280)`.
   - At `1,010 KiB`, `token_map` is **`2×` larger than the entire `512 KiB` private L2 cache of a Cortex-A725 core** and consumes half of a Cortex-X925 core's `2,048 KiB` L2 cache.
2. **Unpinned OS Thread Scheduling Across `512 KiB` and `2,048 KiB` L2 Cores**:
   - Neither `HostEngramHash` nor the native `engram-rows` worker thread (`vllm/models/deepseek_v4_1/engram_rows/_reader.c`) constrained its CPU affinity mask, allowing the Linux scheduler to migrate both threads between `512 KiB`-L2 Cortex-A725 cores and `2,048 KiB`-L2 Cortex-X925 cores from step to step.
   - On `dgx4`, this cross-cluster migration produces bimodal `HostEngramHash.step_ids` latency with a `p95 - p05` spread of **`218.7–398.4 µs`** per step.

## Solution

1. **Compact `int32` `token_map` in `HostEngramHash` (`vllm/models/deepseek_v4_1/engram_rows/host_hash.py`)**:
   - Store `self.token_map` as `np.int32` (`517,120 B = 505 KiB`, `2×` smaller than `int64`) whenever all entries fit in `int32`.
   - Perform the 4-lag token compression (`_compress_compact`) and `blocked` pad substitution in `int32` (`4 B` per element instead of `8 B`) before widening to `uint64` once for the wrapping 64-bit multiplier product.
   - Keep `HostEngramHash.compress()` returning `np.int64` for full API compatibility.
2. **Automatic Max-L2 Core Pool Discovery & Affinity (`host_hash.py` and `_reader.c`)**:
   - Inspect `/sys/devices/system/cpu/cpu*/cache/index2/size` across the current thread's allowed CPU mask and select the subset of CPUs with the maximum private L2 cache size (`cpu5..9, cpu15..19` with `2,048 KiB` L2 on GB10).
   - Pin the native `_reader.c` `worker_main` thread and `HostEngramHash` caller thread to the full 10-core `2,048 KiB`-L2 Cortex-X925 pool by default (recording `worker_l2_kib` in `rows.stats()`), with opt-out via `SPARK_DS41F_ENGRAM_CPU_AFFINITY=0` or `SPARK_DS41F_ENGRAM_HOST_HASH_AFFINITY=0`.

## Patch

- [`0001-cpu-cache-compact-host-hash-and-affinity.patch`](0001-cpu-cache-compact-host-hash-and-affinity.patch) — formatted against `vllm` commit `1f18be14470339c1227ed0432b896a8ca09811b5`.

## Verification & Benchmark (`dgx4`, Grace CPU + GB10)

### Correctness

Executed inside `spark-ds41f:1030` on `dgx4`:

```bash
pytest -o addopts="" --noconftest -q tests/kernels/test_deepseek_v41_engram_host_placement.py tests/kernels/test_deepseek_v41_engram_rows.py
```

- **`26 passed`** (including `test_compact_int32_token_map_matches_int64_reference`, `test_max_l2_cpus_selects_big_l2_cores`, and verification that `rows.stats()["worker_l2_kib"] == 2048` on `dgx4`).

### Grace CPU `HostEngramHash.step_ids` Latency & Run-to-Run Variance (`dgx4`)

Measured on `dgx4` across `100` timed steps (after `20` warmup steps) with `768 KiB` of per-step runner working-set churn between calls:

| Token Count | Configuration | Median (`µs`) | `p95` (`µs`) | `p95 - p05` Spread (`µs`) | Speedup vs Mixed `int64` | Spread Reduction vs Mixed |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `128` | A725 (`512K` L2) + `int64` `token_map` (`1,010 KiB`) | `158.7` | `176.3` | `53.8` | `-16.5%` | `-75.4%` |
| `128` | A725 (`512K` L2) + `int32` `token_map` (`505 KiB`) | `141.3` | `159.6` | `38.3` | `-3.7%` | `-82.5%` |
| `128` | Unpinned Mixed A725/X925 + `int64` `token_map` | `136.2` | `301.5` | `218.7` | baseline | baseline |
| **`128`** | **Pinned X925 (`2048K` L2) + `int32` `token_map`** | **`46.5`** | **`47.6`** | **`1.5`** | **`-65.9%` (`2.93×`)** | **`-99.3%`** |
| `512` | A725 (`512K` L2) + `int64` `token_map` (`1,010 KiB`) | `287.5` | `301.7` | `18.7` | `-27.8%` | `-94.7%` |
| `512` | Unpinned Mixed A725/X925 + `int64` `token_map` | `225.0` | `513.9` | `350.2` | baseline | baseline |
| **`512`** | **Pinned X925 (`2048K` L2) + `int32` `token_map`** | **`133.0`** | **`134.6`** | **`2.8`** | **`-40.9%` (`1.69×`)** | **`-99.2%`** |
| `2048` | A725 (`512K` L2) + `int64` `token_map` (`1,010 KiB`) | `892.6` | `903.2` | `25.0` | `-25.5%` | `-93.7%` |
| `2048` | Unpinned Mixed A725/X925 + `int64` `token_map` | `711.4` | `917.0` | `398.4` | baseline | baseline |
| **`2048`** | **Pinned X925 (`2048K` L2) + `int32` `token_map`** | **`484.0`** | **`579.9`** | **`100.7`** | **`-32.0%` (`1.47×`)** | **`-74.7%`** |
