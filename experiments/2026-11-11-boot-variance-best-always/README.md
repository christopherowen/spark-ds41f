# Boot-to-Boot Benchmark Variance: Root-Cause Attribution and Deterministic Fast-State Fix

## Metadata

- **Date**: 2026-11-11
- **Base deployment commit**: `ed033551831` (`main`, after frozen `candidate-54`)
- **Base candidate stack**: frozen `candidate-54` (`15833ed19c5d`, tree `ad2dc11f613f`, 672 vLLM patches including `0799`)
- **Candidate patches**:
  - [`vllm/0800-perf-loader-retain-UMA-buddy-sponge-across-post-load.patch`](vllm/0800-perf-loader-retain-UMA-buddy-sponge-across-post-load.patch)
  - [`vllm/0801-perf-worker-L3-cluster-aware-RoCE-proxy-worker-and-A.patch`](vllm/0801-perf-worker-L3-cluster-aware-RoCE-proxy-worker-and-A.patch)
  - [`sparknet/0001-perf-oneshot-pin-RoCE-proxy-thread-to-dedicated-fast.patch`](sparknet/0001-perf-oneshot-pin-RoCE-proxy-thread-to-dedicated-fast.patch)
  - **vLLM patch head**: `a5235a16d54e1d1942904d2f4097e75891a20650` (674 vLLM patches)
  - **vLLM expected tree**: `578bc4a6a1a5432b6db27de2522887f93ac462b1`
  - **sparknet patch head / tree**: `88366d5f79296e83e2715bf572114d56ae2b2b4a` / `dab8c4577ac1cd87f7abc59a9c44a8e9d146f579`
  - **Patchset SHA-256**: recorded in [`source.json`](source.json) and [`upstreams.lock.json`](upstreams.lock.json)
- **Host, cluster-launch, and benchmark hygiene changes**:
  - `spark_ds41f/cluster.py` (`defragment_buddy_memory`): run `sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory` on stopped nodes before container launch so reclaimable VFS page/slab caches do not pin `4 KiB` pages inside `2 MiB` pageblocks (restoring `65–81 GiB` of `>= 2 MiB` contiguous buddy blocks before weight loading).
  - `scripts/memguard.sh`: cache container PID (`{{.State.Pid}}`) and check `/proc/<pid>` + read `/proc/meminfo` via bash `read` builtins so steady-state polling forks **zero** `docker inspect` / `awk` subprocesses (preventing unpinned `dockerd`/`containerd` wakeups on `cpu 19`) while preserving 100% fail-closed kill behavior.
  - `scripts/memguard-start.sh` & `spark_ds41f/lab.py`: confine `systemd-run` memory-guard units and `Telemetry` (`nvidia-smi` and `THREAD_SAMPLER`) to Cluster-0 efficiency cores (`0-4`).
  - Sonar ([christopherowen/Sonar#1](https://github.com/christopherowen/Sonar/pull/1): `sonar/stats.py`, `sonar/engine.py`, `sonar/prefill.py`): replace the 25%-breakdown `median + 5 * MAD` cliff on large replay grids with symmetric upper-quartile stall trimming (`trim_stall_tail`, `drop_fraction=0.25`) and warm cold prefill once before `repeat=0`.

## Exact Root Causes Found

1. **Physical Memory Buddy Fragmentation & Full-Lifecycle Post-Load Allocation (`nvidia.ko` + `direct_loader.py` + `cluster.py`)**:
   - Tracing `/usr/src/nvidia-580.178.04/nvidia/nv.c` (`nv_alloc_pages`) and `nv-vm.c` (`nv_alloc_system_pages`) alongside `/proc/buddyinfo` and `/proc/pagetypeinfo` on `sp8` revealed:
     - `nvidia.ko` allocates UMA system pages via `__get_free_pages(GFP_KERNEL | __GFP_COMP, order)` (`MIGRATE_UNMOVABLE`), draining **`MIGRATE_UNMOVABLE` `64 KiB..1 MiB` blocks first**, then **`>= 2 MiB` `MIGRATE_UNMOVABLE` blocks**, before stealing `MIGRATE_MOVABLE` blocks.
     - Neither `CMA` (`CmaTotal = 0` on both `4K` and `64K` kernels, and `nvidia.ko` only calls `dma_alloc_coherent` on Xen Dom0) nor `HugeTLB` + `cuMemHostRegister` (`nv_register_user_pages` sets `flags.user = NV_TRUE`, routing GPU reads over SMMU/snoop at `+5.1%` GEMM latency and permanently subtracting from `MemAvailable`) can replace `cuMemCreate`'s native GPU GMMU `2 MiB` page-table mappings.
     - Previously, `direct_loader._allocate_order13_sponge` (a) returned empty on all `4 KiB` kernels and on `64 KiB` kernels whenever `512 MiB` (`Order 13`) blocks reached `0`, (b) budgeted from `cudaMemGetInfo` (`MemFree`) rather than UMA `/proc/meminfo` `MemAvailable`, (c) used a `10 GiB` floor that starved DSpark's sponge after target weights loaded, and (d) released the sponge at the end of `create_model()` *before* `BaseModelLoader.load_model()` ran `process_weights_after_loading()` (where `85%` of weights — `26.2 GiB` in target and `1.9 GiB` in DSpark — are re-allocated as repacked TileLang MoE slabs, tails, and FP8 scale words).
   - **Root-Cause Fix (`0800` + `cluster.py`)**:
     - `cluster.py` (`defragment_buddy_memory`) drops clean VFS caches and compacts buddy memory on stopped nodes before container launch (`25 GiB -> 80 GiB` free in `>= 2 MiB` blocks).
     - `direct_loader.py` generalizes the sponge across `4 KiB` and `64 KiB` kernels (`6 GiB` untouched `MemAvailable` floor + `buddy_cap` guard + unprivileged `MADV_HUGEPAGE` defrag fallback), stops at the exact `MIGRATE_UNMOVABLE` knee (`0.36 s` soak) so zero `>= 2 MiB` blocks are wasted inside the sponge, refreshes `_refresh_views()` (`shard_weight`), and **retains `model._direct_buddy_sponge` across `create_model` -> `load_weights` -> `process_weights_after_loading`** (`BaseModelLoader.load_model` `finally:`) so both Target (`33.5 GiB`) and DSpark (`2.95 GiB`) weights draw `>= 2 MiB` blocks end-to-end.

2. **Asymmetric GB10 CPU L3 Clusters (`8 MiB` Cluster 0 vs `16 MiB` Cluster 1) & Host Daemon Wakeups**:
   - Patch `0801` + `sparknet/0001` sort fast cores by `(sysfs cache/index3/size, perf_score, cpu)` so `sparknet-proxy` (`cpu 19`), `Worker_TP` (`cpu 18`), and `EngineCore` (`cpu 17`) deterministically select the `16 MiB` L3 cluster and exclude sibling efficiency cores (`10-14`) from `api_server_cpus`.
   - Zero-fork `memguard.sh` + `AllowedCPUs=0,1,2,3,4` eliminate steady-state `dockerd`/`containerd` wakeups on `cpu 19`.

## 8-Boot Balanced `ABBAABBA` Verification on `sp8` (`spark-ds41f:candidate-54`, `results.json`)

To separate software/memory boot variance from Spaceport's passive cooling (`fan_curve: absent`), we ran both a **fixed-clock (`nvidia-smi -lgc 2418,2418`)** 8-boot `ABBAABBA` suite and an **unlocked-clock** 8-boot `ABBAABBA` suite (1 soak boot + 8 fresh Docker container boots, 16L DS4.1 + 3L DSpark at TP4 rank shapes, 12 decode shapes x 80 timed replays each):

| Order | Boot | CPU (`pre->post`) | SM Clock (`post`) | Buddy Sponge (`target` / `DSpark`) | `drf_drop25` (ms) | `host_med` (ms) | `step_p10` (ms) | `step_drop25` (ms) |
| ---: | :--- | :---: | ---: | :--- | ---: | ---: | ---: | ---: |
| 1 | `base-1` | `8->6` | `2398 MHz` | disabled (`48 GiB` VFS cache) | `8.8238` | `0.9862` | `15.3245` | `15.4968` |
| 2 | `best-1` | `18->18` | `2398 MHz` | `9.6 GiB` (`0.36 s`) -> `79.7 / 49.5 GiB >= 2M` | **`8.8679`** | **`1.0063`** | **`15.4364`** | **`15.6141`** |
| 3 | `best-2` | `18->18` | `2405 MHz` | `9.3 GiB` (`0.35 s`) -> `81.1 / 50.9 GiB >= 2M` | **`8.8920`** | **`1.0092`** | **`15.4196`** | **`15.6058`** |
| 4 | `base-2` | `19->18` | `2411 MHz` | disabled (`48 GiB` VFS cache) | `8.8364` | `1.0190` | `15.3619` | `15.5653` |
| 5 | `base-3` | `17->16` | `2411 MHz` | disabled (`48 GiB` VFS cache) | `8.8386` | `1.1639` | `15.4846` | `15.6995` |
| 6 | `best-3` | `18->18` | `2411 MHz` | `9.3 GiB` (`0.36 s`) -> `77.9 / 47.7 GiB >= 2M` | **`8.8911`** | **`1.0103`** | **`15.4273`** | **`15.6284`** |
| 7 | `best-4` | `18->18` | `2411 MHz` | `9.3 GiB` (`0.36 s`) -> `79.1 / 48.9 GiB >= 2M` | **`8.8643`** | **`0.9978`** | **`15.4330`** | **`15.6154`** |
| 8 | `base-4` | `17->16` | `2411 MHz` | disabled (`48 GiB` VFS cache) | `8.8934` | `1.1835` | `15.5364` | `15.7529` |

### Summary Across All 8 Fixed-Clock Balanced Boots

| Metric | `base` Mean ± SD (`SD%`, Max-Min Range) | `best` Mean ± SD (`SD%`, Max-Min Range) | Boot-to-Boot SD Reduction |
| :--- | ---: | ---: | ---: |
| **Whole step (`step_drop25_gm_ms`)** | `15.6286 ± 0.1181 ms` (`0.756%`, `256.1 µs` / `1.64%`) | **`15.6159 ± 0.0093 ms`** (`0.060%`, **`22.6 µs` / `0.145%`**) | **`12.6x` (`-92%`)** |
| **Whole step (`step_trim_mad_gm_ms`)** | `15.7488 ± 0.1286 ms` (`0.817%`, `278.5 µs` / `1.77%`) | **`15.7281 ± 0.0116 ms`** (`0.074%`, **`25.1 µs` / `0.160%`**) | **`11.0x` (`-91%`)** |
| **Whole step `p10` (`step_p10_gm_ms`)** | `15.4268 ± 0.1001 ms` (`0.649%`, `211.9 µs` / `1.37%`) | **`15.4291 ± 0.0073 ms`** (`0.048%`, **`16.8 µs` / `0.109%`**) | **`13.5x` (`-93%`)** |
| **DSpark drafter (`drf_drop25_gm_ms`)** | `8.8481 ± 0.0309 ms` (`0.350%`, `69.6 µs` / `0.79%`) | **`8.8788 ± 0.0148 ms`** (`0.166%`, **`27.7 µs` / `0.312%`**) | **`2.11x` (`-53%`)** |
| **Target forward (`fwd_drop25_gm_ms`)** | `5.6739 ± 0.0238 ms` (`0.419%`, `43.4 µs` / `0.77%`) | **`5.7044 ± 0.0197 ms`** (`0.345%`, **`44.4 µs` / `0.78%`**) | **`1.21x` (`-18%`)** |
| **Host / CPU gap (`host_med_mean_ms`)** | `1.0881 ± 0.1000 ms` (`9.19%`, `197.3 µs` / `18.1%`) | **`1.0059 ± 0.0057 ms`** (`0.562%`, **`12.5 µs` / `1.24%`**) | **`16.4x` (`-94%`, `-7.6%` faster)** |
