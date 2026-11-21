# Boot-to-Boot Variance Elimination via Contiguous CUDA Pool Pre-Allocation and 512 MiB Pageblock Reset

- **Date:** 2026-11-21
- **Base baseline:** `a45b79195` (`spark-ds41f:1120` / `candidate-62`)
- **Final candidate commit:** `a57cb9f55` (`PR #1259`, `spark-ds41f:candidate-63-r4`, `sha256:6c31c7331e4b1adb1d3d4ced84446003b5f076b64e3e01043148ba18ada51cb5`)
- **Intermediate PRs:** `#1255` (`baa4ff32e`, probe profile, later removed), `#1257` (`c9f199f0c`, initial `0832` + `hugetlb` reset), `#1258` (`9915bda7b`, `r3` post-load KV placeholder, superseded by `#1259`)
- **Cluster & shape:** `tp3-triangle` (`dgx1`, `dgx2`, `dgx3`), plus standalone kernel/allocator probes on `dgx4` (GB10, Ubuntu `7.0.0-1019-nvidia-64k`, ARM64 64 KiB pages, `iommu.passthrough=0`)
- **Saved raw artifacts:** `dgx1:/home/swank/spark-lab/c63-phase{1-aged,2-reboot,3-noexpseg,4-pool,5-r3,6-r4}/`

## 1. Problem Statement

Even after prior physical-placement fixes (`0800`/`0801`) and a full reboot of all cluster nodes, Boot 2 of `spark-ds41f:1120` consistently slowed down by **`+1.21%`** across the 15-shape `engine grid` (`floor25` GM) and **`-1.22%`** on `c8` steady-state throughput (`172.775 -> 170.667 tok/s`), with every decode shape degrading `+0.99%..+1.47%`. Applying only `0830`/`0831` (`candidate-63` r1) improved aged-node performance (`+1.32% -> +0.25%` vs fresh-reboot Boot 1) but still dropped `+0.79%..+0.97%` immediately after a single post-reboot boot.

## 2. Physical Root-Cause Forensics (Standalone `dgx4` & `tp3-triangle`)

Inspecting `/proc/buddyinfo`, `/proc/pagetypeinfo`, `/proc/meminfo` (`SecPageTables`), bounded `/proc/kpageflags`, and slice-by-slice GPU read bandwidth on `dgx4` isolated four interacting kernel/allocator mechanisms:

1. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` permanently fragments `512 MiB` (`Order-13`) pageblocks on Boot 1 via immortal ARM SMMU L3 page tables:**
   - On GB10 (`64 KiB` pages, `iommu.passthrough=0`), a fresh reboot exposes `233 × 512 MiB` free `Order-13` blocks (`116.5 GiB`, `242 Movable` / `9 Unmovable` blocks) and `SecPageTables = 4.81 MiB` (`4,928 kB`).
   - With `expandable_segments:True`, PyTorch allocates ~`114 GiB` via `~58,000 × 2 MiB` `cuMemCreate` calls. Each `dma_map` allocates ARM SMMU L3 page-table pages from `MIGRATE_UNMOVABLE` that Linux `io-pgtable-arm` **never frees on `dma_unmap`** (not even after process exit or `rmmod nvidia`).
   - After a single Boot 1, `SecPageTables` jumped **`4.81 -> 28.5 MiB`** (`+369` immortal `flags=0x0` kernel pages), scattering 1–2 unmovable pages into `225` of `233` `512 MiB` pageblocks and collapsing `Order-13` free blocks from **`233 -> 8`** (`-96.6%`) until reboot.
   - Controlled `dgx4` allocator comparison (`114 GiB` allocation/free cycle from fresh reboot):
     - `expandable_segments:True` (`cuMemCreate` `2 MiB` chunks): `SecPageTables = 28.5 MiB`, post-free `O13 = 8`.
     - `backend:cudaMallocAsync`: `SecPageTables = 23.2 MiB`, post-free `O13 = 27`.
     - `cudaMalloc` in 85 separate chunks: `SecPageTables = 11.3 MiB`, post-free `O13 = 145`.
     - Single pre-allocated `114 GiB` `cudaMalloc` segment + `512 MiB` `hugetlb` reset: `SecPageTables = 7.0 -> 7.1 MiB`, post-free **`O13 = 227 -> 225`** (`112.5 GiB`) across Boot 1 -> Boot 2.

2. **`Worker._scoped_allocator_max_split(max_split_size_mb=20)` activates automatically when `expandable_segments` is disabled:**
   - Simply removing `expandable_segments:True` in Phase 3 triggered `vllm/v1/worker/gpu_worker.py`'s fallback `max_split_size_mb=20`, forcing ~250 separate `cudaMalloc` calls during weight loading (`SecPageTables = 4.81 -> 18.62 -> 20.81 -> 22.69 MiB`, `O13 = 233 -> 55 -> 35 -> 20`).

3. **Linux does not revert `MIGRATE_UNMOVABLE` `512 MiB` pageblocks to `MIGRATE_MOVABLE` on ordinary free, and `nvidia.ko` drains `Unmovable` sub-O13 first:**
   - When `nvidia.ko` frees `Order-13` pages back to the buddy allocator, Linux leaves their `512 MiB` pageblocks typed `MIGRATE_UNMOVABLE` so subsequent kernel allocations scatter across them.
   - Writing free `O13` to `/sys/kernel/mm/hugepages/hugepages-524288kB/nr_hugepages` and then `0` forces `MIGRATE_UNMOVABLE` `512 MiB` blocks back to `MIGRATE_MOVABLE`. Running `echo 1 > /proc/sys/vm/compact_memory` **after** the first `hugetlb` pass followed by a second `hugetlb` pass improved recovery on fragmented `dgx4` from `O13 = 44 -> 76` (`+73%`, `+16 GiB`).
   - Slice-by-slice 4 GiB GPU read-bandwidth sweeps inside a single `64 GiB` `cudaMalloc` on `dgx4` showed `nvidia.ko` drains `MIGRATE_UNMOVABLE` sub-O13 blocks **first** at offset `0..~8–16 GiB` (**`238.7–250.2 GB/s`**), then `Order-13` `512 MiB` blocks (**`262.8 GB/s`**, `+10.1%` faster), then `MIGRATE_MOVABLE` sub-O13 (`243–248 GB/s`). Adding a 1-shot transient `cudaMalloc` soak sized to free `MIGRATE_UNMOVABLE` sub-O13 (read from `/proc/pagetypeinfo`, accessible inside non-privileged containers) raised every 4 GiB slice from offset `0.0 GiB` to **`262.2–262.6 GB/s`**.

4. **Falsified / Abandoned Dead End — `mmap(MAP_HUGETLB | MAP_HUGE_512MB) + cudaHostRegister`:**
   - Tested on `dgx4`: registered 512 MiB hugetlb pages measured **`164.0 GB/s`** sequential read vs **`241.4 GB/s`** native `cudaMalloc` (and `8×` slower random gather), because `cudaHostRegister` maps over the non-L2-coherent sysmem aperture. Abandoned.

## 3. Solution Implemented (`candidate-63-r4`, `PR #1257` + `PR #1259`)

1. **Drop `expandable_segments:True`** in all `candidate-63` runtime configs (`PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6`).
2. **vLLM patch `0832`** ([`0832-perf-worker-preallocate-contiguous-CUDA-pool-on-64-K.patch`](../2026-10-26-spark-ds41f-candidate/candidate-63/vllm/0832-perf-worker-preallocate-contiguous-CUDA-pool-on-64-K.patch)):
   - Bypass `_scoped_allocator_max_split(20)` when the contiguous pool is active.
   - Before weight loading, parse `/proc/pagetypeinfo` for free `MIGRATE_UNMOVABLE` sub-O13 bytes and perform a 1-shot transient `cudaMalloc` soak (capped to preserve `>= 8 GiB` `MemAvailable`) so `nvidia.ko` drains fragmented `Unmovable` blocks before the permanent weight pool.
   - Allocate one contiguous weight pool sized to `requested_memory - kv_cache_memory_bytes` aligned down to `512 MiB` (**`96.5 GiB`** on `tp3-triangle`), anchor it across `empty_cache()` with a `2 MiB` (`large_pool`) tensor, and immediately release the transient soak.
   - Gate `direct_loader.py` on `set_preallocated_cuda_pool_bytes(pool_bytes)`: skip the buddy sponge (cutting model load time from `~50.7 s` to `38.6 s`) and keep dense-first placement when `front_clean=True` so the ~11 GiB L2-prefetched dense arena lands at offset `0` of the clean `512 MiB`-backed pool.
   - Excluding `kv_cache_memory_bytes` from Segment #1 lets KV cache land in one clean contiguous `4.91 GiB` Segment #2, fixing the `5.0 GiB` tail-fragmentation spill observed on `dgx2` in `r2`/`r3`.
3. **2-pass `512 MiB` `hugetlb` reset with post-`hugetlb` compaction** in [`spark_ds41f/cluster.py`](../../spark_ds41f/cluster.py) (`defragment_buddy_memory`, invoked on `cluster stop` and `cluster start`).
4. **Retained `candidate-63` r1 fixes**: `0830` (64 KiB ladder fallback), `0831` (`engine_grid` TP rank barrier + GC pause), `COOL_BELOW_C = 49.0 C` (`spark_ds41f/bench.py`), and `floor25` grid estimator (`spark_ds41f/sonar.py`).

## 4. Exact Commands and Workloads

All cluster operations used `bin/spark cluster sync/stop/start --apply --cluster-config <profile>` inside a coordinated `dgx1` lab window (`scripts/lab.py`). Measurements ran via `sonar` against `http://10.0.1.71:8000` on `tp3-triangle` (`dgx1..dgx3`):

- **Engine grid (15 decode shapes, 100 replays/shape):** `python3 -m sonar engine http://10.0.1.71:8000 --replays 100 --label <tag> --output <dir>/grid-<tag>.json`
- **Steady-state throughput (`c8`, 10 batches):** `python3 -m sonar bench http://10.0.1.71:8000 --concurrency 8 --batches 10 --label <tag> --output <dir>/c8-<tag>.json`
- **Long-context prefill (`32K` & `262K`, 2 samples each):** `python3 -m sonar prefill http://10.0.1.71:8000 --sizes 32768,262144 --samples 2 --label <tag> --output <dir>/prefill-<tag>.json`
- **Model exactness gate (128 prompts, greedy + top-20 logprobs):** `python3 -m sonar model http://10.0.1.71:8000 --limit 128 --concurrency 8 --max-tokens 256 --top 20 --label c63r4-b1 --output <dir>/model-c63r4-b1.json`

## 5. All Measured Runs on `tp3-triangle` (Phases 1–6)

Every `engine grid` `fl25 GM` and `mad5 GM` below is measured against the fresh-reboot `1120-reboot-b1` reference (`15` decode shapes; negative = faster). All runs had `0` failed requests.

| Phase | Run Tag | Pre-Start `O13` (`dgx1/2/3`) | `SecPT` (`MiB`, post-stop) | `c8` tok/s (±CI95) | `c8` steps/s | `tpot` p50 (ms) | Grid `fl25` GM | Grid `mad5` GM | `1x2` (ms) | `4x8` (ms) | `8x16` (ms) | `8x48` (ms) | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **1 (aged)** | `1120-aged` | `0 / 5 / 7` | — | `169.017 ± 3.61` | `9.683` | `46.27` | `+1.32%` | `+1.34%` | `16.586` | `18.220` | `19.908` | `26.963` | Aged nodes, baseline `1120` |
| **1 (aged)** | `c63-aged` | `0 / 6 / 7` | — | `170.492 ± 3.17` | `9.708` | `46.12` | `+0.25%` | `+0.11%` | `16.364` | `18.010` | `19.644` | `26.937` | `0830`+`0831` only (`expseg=True`) |
| **2 (reboot ABBA)** | `1120-reboot-b1` | `232 / 233 / 233` | — | `172.775 ± 2.94` | `9.842` | `45.51` | **`+0.00%`** | **`+0.00%`** | `16.359` | `17.960` | `19.730` | `26.727` | Fresh reboot ref (`O13` destroyed after boot 1) |
| **2 (reboot ABBA)** | `c63-reboot-b1` | `8 / 12 / 12` | — | `170.242 ± 3.20` | `9.750` | `46.02` | `+0.79%` | `+1.05%` | `16.461` | `18.036` | `19.767` | `27.091` | `expseg=True`; `O13` already gone |
| **2 (reboot ABBA)** | `c63-reboot-b2` | `8 / 12 / 12` | — | `170.667 ± 3.10` | `9.725` | `45.99` | `+0.97%` | `+1.16%` | `16.459` | `18.080` | `19.865` | `27.164` | `expseg=True`; degraded |
| **2 (reboot ABBA)** | `1120-reboot-b2` | `8 / 12 / 12` | — | `170.667 ± 3.04` | `9.717` | `46.01` | **`+1.21%`** | **`+1.21%`** | `16.559` | `18.171` | `19.926` | `27.000` | Every shape `+0.99%..+1.47%` slower |
| **3 (no-expseg)** | `noexp-b1` | `233 / 233 / 233` | `18.25 / 18.50 / 18.62` | `173.383 ± 3.02` | `9.875` | `45.38` | `+1.36%` | `+1.26%` | `16.611` | `18.194` | `19.850` | `27.131` | Unfavorable: exposed `_scoped_allocator_max_split(20)` |
| **3 (no-expseg)** | `noexp-b2` | `55 / 63 / 62` | `20.56 / 20.81 / 20.81` | `172.000 ± 2.69` | `9.800` | `45.74` | `+0.76%` | `+0.73%` | `16.522` | `18.055` | `19.734` | `27.115` | `O13` decayed further each boot |
| **3 (no-expseg)** | `noexp-b3` | `35 / 32 / 27` | `22.69 / 22.12 / 22.56` | `171.583 ± 2.90` | `9.775` | `45.77` | `+0.80%` | `+0.88%` | `16.529` | `18.084` | `19.722` | `27.024` | `O13` down to `20/24/23` post-stop |
| **4 (`c63-r2`)** | `c63r2-b1` | `232 / 233 / 233` | `7.31 / 7.44 / 7.25` | `174.033 ± 2.98` | `9.925` | `45.14` | `-0.40%` | `-0.51%` | `16.270` | `17.931` | `19.594` | `26.754` | `101.5 GiB` pool, no soak; load `38.3 s` |
| **4 (`c63-r2`)** | `c63r2-b2` | `202 / 215 / 205` | `7.81 / 8.12 / 7.88` | `173.233 ± 2.85` | `9.867` | `45.35` | `+0.39%` | `+0.29%` | `16.371` | `18.058` | `19.721` | `26.957` | `+0.79%` drift from unsoaked `Unmovable` front |
| **4 (`c63-r2`)** | `c63r2-b3` | `199 / 203 / 207` | `7.88 / 8.19 / 7.94` | `172.767 ± 2.19` | `9.883` | `45.01` | `+0.06%` | `-0.00%` | `16.372` | `17.935` | `19.624` | `26.912` | Prefill `32K=7285`, `262K=7484 tok/s` (`0` fail) |
| **5 (`c63-r3`)** | `c63r3-b4` | `182 / 205 / 201` | `8.06 / 8.50 / 8.31` | `172.225 ± 2.51` | `9.867` | `45.18` | `+0.08%` | `-0.08%` | `16.326` | `18.007` | `19.684` | `26.876` | Unfavorable mem: `dgx2` `5.46 GiB` free (`106.68 GiB` reserved) |
| **5 (`c63-r3`)** | `c63r3-b5` | `190 / 195 / 195` | `8.12 / 8.56 / 8.31` | `171.792 ± 2.28` | `9.833` | `45.29` | `+0.43%` | `+0.32%` | `16.400` | `18.044` | `19.702` | `27.049` | No reboot after `25 GiB` `docker load`; `dgx2` still `5.48 GiB` free |
| **6 (`c63-r4`, FINAL)** | **`c63r4-b1`** | `233 / 233 / 233` | `7.50 / 7.50 / 7.44` | **`172.817 ± 2.77`** | **`9.858`** | `45.72` | **`-0.20%`** | **`-0.34%`** | **`16.320`** | **`18.013`** | **`19.620`** | **`26.778`** | `96.5 GiB` pool + soak (`0.00 GiB`); load `38.6 s` |
| **6 (`c63-r4`, FINAL)** | **`c63r4-b2`** | `206 / 206 / 205` | `8.06 / 8.12 / 8.06` | **`171.792 ± 2.28`** | **`9.833`** | `45.20` | **`+0.08%`** | **`-0.08%`** | **`16.311`** | **`17.960`** | **`19.708`** | **`26.894`** | Soaked `5.36/5.94/6.47 GiB`, `front_clean=True` |
| **6 (`c63-r4`, FINAL)** | **`c63r4-b3`** | `201 / 195 / 209` | `8.56 / 8.62 / 8.62` | **`173.025 ± 2.29`** | **`9.900`** | `45.29` | **`-0.18%`** | **`-0.35%`** | **`16.252`** | **`17.888`** | **`19.704`** | **`26.941`** | Soaked `8.84/9.42/9.46 GiB`; prefill `32K=7268`, `262K=7468 tok/s` |

### Key Phase 6 (`candidate-63-r4`) Observations

- **Boot-to-boot variance eliminated:** Across three full boots after reboot, `floor25` GM vs `1120-reboot-b1` stayed at **`-0.20%`, `+0.08%`, `-0.18%`** (Boot 1 -> Boot 3 drift **`+0.02%`**, total spread `0.28%`, mean `-0.10%` vs `1120-reboot-b1` and **`-1.31%` faster than `1120-reboot-b2`**). `mad5` GM stayed at **`-0.34%`, `-0.08%`, `-0.35%`**.
- **Order-13 & SMMU preservation:** After 3 full engine boots + `model` + `grid` + `c8` + `262K` prefill, idle nodes retained **`O13 = 189 / 198 / 190`** (`94.5–99.0 GiB` @ `512 MiB`, `216 / 218 / 212` `MIGRATE_MOVABLE` blocks) and **`SecPageTables = 8.56 / 8.62 / 8.62 MiB`** (vs `O13 = 8/12/12` and `28.5 MiB` after a single `1120` boot).
- **Rank-symmetric memory footprint:** In `c63-r3` (`101.5 GiB` pool), `dgx2` fragmented the pool tail during `load_model` and spilled KV cache into a separate segment (`torch_reserved = 106.68 GiB`, `free_memory = 5.46 GiB`). Sizing the pool to `96.5 GiB` (`requested_memory - kv_cache_memory_bytes`) in `c63-r4` made **`torch_reserved = 103.08 GiB` identically across all 3 ranks on all 3 boots** (`free_memory`: `dgx1 = 6.89–7.22 GiB`, **`dgx2 = 8.78–9.11 GiB`** up `+3.65 GiB`, `dgx3 = 8.93–9.21 GiB`).
- **Faster model load:** Skipping the buddy sponge inside the pre-allocated pool cut `load_model` from `~50.7 s` (`1120`) to **`38.6 s`** (`-12.1 s` per boot).
- **Exact model output preservation (`sonar model`, 128 prompts):** `c63r4-b1` (`outputs_sha256 = b795e6d4c813815a475ab1516faf277a90a86b73614c38f4700189484a8c532d`, `perplexity = 1.3160824`, `acceptance_length = 2.303`, `0` failed) matched baseline **`128/128` greedy identical**, `100.00%` identical distributions, `100.000%` top-1 agreement, **`KL = 0.000e+00`**, **`perplexity +0.000%`**.
- **Long-context prefill intact:** `32K` = `7,267.8 tok/s` (`samples: 7265.17, 7270.38`, `0` fail), `262K` = `7,468.1 tok/s` (`samples: 7482.87, 7453.24`, `0` fail).

## 6. Verification

- **vLLM GPU & loader unit tests (in `spark-ds41f:candidate-63-r4` on `dgx4`):** `101 passed` (`test_contiguous_cuda_pool.py`, `test_direct_loader.py`, `test_direct_loader_cuda.py`, `test_engine_grid_sync.py`).
- **Deployment unit tests (Linux on `dgx4`):** `906 passed` (`28.0 s`), including new `tests/test_restart_guards.py` coverage for the 2-pass `hugetlb` reset + post-`hugetlb` compaction.
- **Build & health checks:** `bin/spark doctor`, `bin/spark build prepare`, `bin/spark build check`, and `git diff --check` all passed cleanly. Production `spark-ds41f:1120` was restored at window close (`doctor --live OK`).

## 7. Conclusion

The persistent post-reboot boot-to-boot variance (`+1.21%`) was caused by `expandable_segments:True` scattering immortal ARM SMMU L3 page tables across `512 MiB` pageblocks, compounded by `_scoped_allocator_max_split(20)`, un-reverted `MIGRATE_UNMOVABLE` pageblocks, and `nvidia.ko` draining `Unmovable` sub-O13 blocks at offset `0`. `candidate-63-r4` eliminates the degradation across multi-boot cycles (`+0.02%` Boot 1 -> Boot 3 drift, `-1.31%` faster than `1120` Boot 2), preserves `~190–209` `512 MiB` blocks after 3 boots, cuts load time by `12 s`, equalizes memory headroom across all ranks (`103.08 GiB` reserved), and preserves bit-exact greedy outputs (`128/128`, `KL = 0`).
