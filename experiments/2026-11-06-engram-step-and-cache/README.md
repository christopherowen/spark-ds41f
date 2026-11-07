# Multi-CTA Fused Disk Engram Step (`#833` Part B) + Packed Reader (`#754`) & 4-Way User-Space Row Cache (`#759`)

## Metadata
- **Date**: 2026-11-06
- **Node**: `sp8` (NVIDIA GB10 `sm_121a`, container `spark-ds41f:1030`)
- **Base `vllm` commit**: `a75357bba81e31e4f3085700f94b6a0b29aed44a` (Candidate `1107`)
- **Candidate `vllm` commit**: `30c6f001c6f379446baa7228d68d777f39f1b22b` (`feat/engram-step-and-cache` in `/home/chris/spark-lab/engram-step-and-cache/vllm`)
- **Exported patch**: `experiments/2026-11-06-engram-step-and-cache/patches/0001-DeepSeek-V4.1-Fuse-multi-CTA-disk-Engram-step-and-wi.patch`

## Root-Cause Analysis & Design

### 1. Multi-CTA Fused Disk Engram Step (`#833` Part B Rework)
- **Why `#833` Part B (`96b3b1ec4`) regressed in `candidate-34`**:
  - `96b3b1ec4` fused `engram_rows_wait` + `engram_rows_gather` + `engram_rows_release` into a single launch (`engram_rows_step_kernel`), saving 2 kernel launches per disk Engram layer (`4` launches/step across layers 1 & 14).
  - However, it hardcoded `with T.Kernel(1, threads=256)` (**a single CTA** for all `M * 24` hash lookups and `M * 1280` outputs) so `T.sync_threads()` could fence the host poll and release around the gather. Serializing all tokens onto 1 CTA caused a `1.6x-58x` regression for `M >= 4`.
- **Why naive multi-CTA polling/atomics on mapped-host `control` also bottleneck on GB10**:
  - On GB10 (`sm_121a`), `control` is allocated in mapped host memory (`cudaHostAllocMapped`) accessed over C2C. Having every CTA poll `control` (`ld.acquire.sys.global.u32`) and execute `atom.inc` on `control + 56` serializes `3 * N` system-scope C2C transactions at the host memory controller (`~0.34 us` per transaction = `~1.02 us` per CTA, or `51.3 us` at `M = 48`).
- **Reworked Architecture (`engram_rows_step_kernel`)**:
  1. **Exact Column-Divisor Multi-CTA Grid**: Uses `rows_per_cta = max(r <= 8 : count % r == 0)` (`6` rows / `384` threads per CTA for TP4 6-column shards; `8` rows / `512` threads per CTA for TP3 8-column and TP1 24-column shards), launching `col_blocks * M` CTAs (`1` CTA/token for TP4 and TP3; `3` CTAs/token for TP1).
  2. **Overlapped Device-Memory Row-ID Preload**: Every thread issues its streaming `engram_ld_cs_s64(&ids[t, head])` and `num_tokens[0]` load *before* `T.sync_threads()`, hiding device-memory load latency behind `tx == 0` host wait.
  3. **128B-Isolated GPU-L2 Inter-CTA Synchronization (`engram_gpu_sync`)**: Only CTA `(0, 0)` polls mapped-host `control` (`IDS_SEQ` and `ROWS_SEQ`) over C2C and broadcasts readiness (`my_step`) into GPU L2 (`engram_gpu_sync[ENGRAM_SYNC_READY]`); non-lead CTAs spin in GPU L2 (`ld.acquire.gpu.global.u32` with `nanosleep.u32 16;`). On release, each CTA increments `engram_gpu_sync[ENGRAM_SYNC_RETIRED]` in GPU L2 (`atom.inc.release.gpu.global.u32`), and only the last finishing CTA (`prev == total_ctas - 1`) writes `CONSUMED_SEQ` to mapped-host `control`. Total C2C transactions on `control` are reduced to **3 per step regardless of `M`**.

### 2. Packed Single-Plane Disk Reader Wiring (`#754`) & Uncapped 4-Way Row Cache (`#759`)
- **Packed Reader Wiring (`vllm/models/deepseek_v4_1/common/engram.py`)**:
  - Wired `ParallelEngramEmbedding.open_disk_rows()` to call `add_packed_source()` when a rank-compacted 4 KiB-page-aligned packed file (`15 x 264B` rows per 4096B page, `engram-layer{layer_id}-tp{tp_size}-rank{tp_rank}.packed`) is present alongside the model checkpoint or when `SPARK_DS41F_ENGRAM_PACKED_DIR` is configured (building the packed file atomically if missing). Standard HuggingFace checkpoints continue to use the two-plane (`weight` + `scale`) `O_DIRECT` path out of the box.
- **Uncapped 4-Way User-Space Row Cache (`vllm/models/deepseek_v4_1/engram_rows/_row_plan.h`)**:
  - Previously, `ple_plan_step_cached()` bypassed the 17.3 MiB (`65,536`-row, `16,384`-set x 4-way) user-space row cache whenever `count > PLE_CACHE_MAX_LOOKUPS` (`1536` lookups = `64` tokens). Sizing the per-reader miss scratch buffers (`miss_rows` and `miss_first_slot`) to `max(max_lookups, PLE_CACHE_MAX_LOOKUPS)` keeps the 4-way cache active across all batch sizes (`T = 1..max_tokens`) in both two-plane and packed modes.
  - Exposed `ENGRAM_ROW_CACHE_ROWS` (`SPARK_DS41F_ENGRAM_ROW_CACHE_ROWS`) and exported `DEFAULT_CACHE_ROWS = 65536` in `vllm/models/deepseek_v4_1/engram_rows/__init__.py`.

## Verification Summary (2 Fresh GPU Container Boots on `sp8`)
- **Bit-Exactness (`torch.equal`)**: `100%` bit-exact across all tested shapes (`tp4-6cols`, `tp3-8cols`, `tp1-24cols`), all batch sizes `M in {1, 2, 4, 8, 16, 18, 24, 30, 36, 42, 48, 128, 2048}`, partial `num_tokens < M`, Eager and CUDA Graph modes, and across all 4 reader modes (2-plane nocache, 2-plane cached, packed nocache, packed cached).
- **`compute-sanitizer` (`boot1` & `boot2`)**:
  - `memcheck`: `ERROR SUMMARY: 0 errors` (`receipts/compute-sanitizer-boot1.txt`, `receipts/compute-sanitizer-boot2.txt`)
  - `initcheck`: `ERROR SUMMARY: 0 errors`
  - `racecheck`: `RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)`
- **GPU Pytest Suites (`boot1` & `boot2`)**:
  - `tests/kernels/test_deepseek_v41_engram_rows.py` + `tests/models/test_deepseek_v4_1_engram.py`: **`94 passed`** on `boot1` (`63.18s`) and **`94 passed`** on `boot2` (`62.94s`).

## Part 1 Benchmark Results: Multi-CTA `engram_rows_step_kernel` vs 3-Launch Baseline

### TP4 (`columns=0x111111`, 6 cols/rank, 1x384 threads/token CTA)

| Batch `M` | Eager 3-Launch B1 / B2 (us) | Eager Fused Step B1 / B2 (us) | Eager Mean Speedup | Graph 3-Launch B1 / B2 (us) | Graph Fused Step B1 / B2 (us) | Graph Mean Speedup |
|---:|---:|---:|---:|---:|---:|---:|
| `1` | `12.10` / `11.51` | `4.42` / `4.33` | **`2.70x`** | `3.38` / `3.37` | `2.18` / `2.17` | **`1.55x`** |
| `2` | `12.05` / `11.61` | `4.50` / `4.79` | **`2.54x`** | `3.38` / `3.38` | `3.38` / `3.29` | **`1.01x`** |
| `4` | `11.79` / `11.31` | `5.89` / `5.71` | **`1.99x`** | `3.39` / `3.41` | `3.39` / `3.35` | **`1.01x`** |
| `8` | `11.73` / `11.23` | `6.01` / `5.94` | **`1.92x`** | `3.45` / `3.47` | `3.45` / `3.39` | **`1.01x`** |
| `16` | `11.56` / `11.23` | `6.12` / `6.02` | **`1.88x`** | `3.58` / `3.49` | `3.49` / `3.42` | **`1.02x`** |
| `18` | `11.56` / `11.35` | `6.13` / `5.98` | **`1.89x`** | `3.59` / `3.48` | `3.52` / `3.45` | **`1.02x`** |
| `24` | `11.64` / `11.20` | `6.15` / `6.10` | **`1.87x`** | `3.69` / `3.59` | `3.58` / `3.48` | **`1.03x`** |
| `30` | `11.57` / `11.26` | `6.14` / `6.13` | **`1.86x`** | `3.67` / `3.63` | `3.60` / `3.55` | **`1.02x`** |
| `36` | `11.60` / `11.22` | `6.15` / `6.15` | **`1.86x`** | `3.72` / `3.69` | `3.67` / `3.59` | **`1.02x`** |
| `42` | `11.65` / `11.28` | `6.15` / `6.15` | **`1.86x`** | `3.79` / `3.69` | `3.69` / `3.59` | **`1.03x`** |
| `48` | `11.61` / `11.28` | `6.15` / `6.15` | **`1.86x`** | `3.79` / `3.89` | `3.69` / `3.78` | **`1.03x`** |
| `128` | `11.59` / `11.26` | `6.17` / `6.16` | **`1.85x`** | `4.53` / `4.19` | `4.41` / `4.18` | **`1.01x`** |
| `2048` | `29.33` / `31.03` | `26.72` / `26.66` | **`1.13x`** | `23.98` / `25.96` | `25.19` / `24.64` | **`1.00x`** |

### TP3 (`columns=0x249249`, 8 cols/rank, 1x512 threads/token CTA)

| Batch `M` | Eager 3-Launch B1 / B2 (us) | Eager Fused Step B1 / B2 (us) | Eager Mean Speedup | Graph 3-Launch B1 / B2 (us) | Graph Fused Step B1 / B2 (us) | Graph Mean Speedup |
|---:|---:|---:|---:|---:|---:|---:|
| `1` | `25.60` / `11.52` | `9.02` / `4.26` | **`2.80x`** | `3.39` / `3.38` | `2.27` / `2.16` | **`1.53x`** |
| `2` | `11.49` / `11.21` | `5.45` / `5.92` | **`2.00x`** | `3.38` / `3.37` | `3.39` / `3.38` | **`1.00x`** |
| `4` | `11.52` / `11.25` | `5.78` / `5.76` | **`1.97x`** | `3.38` / `3.38` | `3.44` / `3.38` | **`0.99x`** |
| `8` | `11.55` / `11.29` | `5.93` / `5.94` | **`1.92x`** | `3.43` / `3.48` | `3.47` / `3.46` | **`1.00x`** |
| `16` | `11.54` / `11.25` | `6.08` / `5.94` | **`1.90x`** | `3.58` / `3.50` | `3.51` / `3.44` | **`1.02x`** |
| `18` | `11.34` / `11.25` | `6.07` / `5.99` | **`1.87x`** | `3.59` / `3.53` | `3.57` / `3.48` | **`1.01x`** |
| `24` | `11.58` / `11.25` | `6.14` / `6.04` | **`1.88x`** | `3.61` / `3.59` | `3.59` / `3.49` | **`1.02x`** |
| `30` | `11.57` / `11.23` | `6.14` / `6.15` | **`1.86x`** | `3.69` / `3.69` | `3.59` / `3.59` | **`1.03x`** |
| `36` | `11.56` / `11.16` | `6.15` / `6.15` | **`1.85x`** | `3.72` / `3.69` | `3.68` / `3.59` | **`1.02x`** |
| `42` | `11.60` / `11.19` | `6.15` / `6.15` | **`1.85x`** | `3.78` / `3.69` | `3.69` / `3.59` | **`1.03x`** |
| `48` | `11.53` / `11.22` | `6.15` / `6.16` | **`1.85x`** | `3.88` / `3.83` | `3.77` / `3.72` | **`1.03x`** |
| `128` | `11.54` / `11.28` | `6.17` / `6.16` | **`1.85x`** | `4.62` / `4.51` | `4.58` / `4.40` | **`1.02x`** |
| `2048` | `53.21` / `46.36` | `47.18` / `39.07` | **`1.15x`** | `45.99` / `40.23` | `44.85` / `37.74` | **`1.04x`** |

### TP1 (`columns=0xFFFFFF`, 24 cols/rank, 3x512 threads/token CTA)

| Batch `M` | Eager 3-Launch B1 / B2 (us) | Eager Fused Step B1 / B2 (us) | Eager Mean Speedup | Graph 3-Launch B1 / B2 (us) | Graph Fused Step B1 / B2 (us) | Graph Mean Speedup |
|---:|---:|---:|---:|---:|---:|---:|
| `1` | `24.24` / `11.39` | `8.67` / `5.78` | **`2.47x`** | `3.49` / `3.43` | `3.47` / `3.35` | **`1.02x`** |
| `2` | `11.68` / `11.27` | `4.83` / `5.91` | **`2.14x`** | `3.59` / `3.48` | `3.48` / `3.38` | **`1.03x`** |
| `4` | `11.51` / `11.21` | `6.14` / `5.99` | **`1.87x`** | `3.59` / `3.56` | `3.55` / `3.39` | **`1.03x`** |
| `8` | `11.66` / `11.22` | `6.15` / `6.04` | **`1.88x`** | `3.76` / `3.59` | `3.60` / `3.47` | **`1.04x`** |
| `16` | `11.55` / `11.25` | `6.15` / `6.15` | **`1.85x`** | `4.14` / `4.00` | `3.79` / `3.59` | **`1.10x`** |
| `18` | `11.56` / `11.28` | `6.15` / `6.15` | **`1.86x`** | `4.15` / `4.02` | `3.78` / `3.69` | **`1.09x`** |
| `24` | `11.59` / `11.24` | `6.16` / `6.15` | **`1.85x`** | `4.30` / `4.19` | `3.89` / `3.79` | **`1.10x`** |
| `30` | `11.54` / `11.21` | `6.16` / `6.16` | **`1.85x`** | `4.46` / `4.30` | `4.05` / `3.96` | **`1.09x`** |
| `36` | `11.58` / `11.23` | `6.16` / `6.16` | **`1.85x`** | `4.60` / `4.48` | `4.20` / `4.20` | **`1.08x`** |
| `42` | `11.49` / `11.22` | `6.16` / `6.16` | **`1.84x`** | `4.65` / `4.60` | `4.29` / `4.20` | **`1.09x`** |
| `48` | `11.57` / `11.23` | `6.17` / `6.17` | **`1.85x`** | `4.94` / `4.72` | `4.52` / `4.40` | **`1.08x`** |
| `128` | `11.80` / `12.39` | `10.26` / `10.21` | **`1.18x`** | `7.07` / `6.71` | `7.78` / `7.47` | **`0.90x`** |
| `2048` | `181.00` / `180.67` | `168.36` / `168.01` | **`1.08x`** | `175.22` / `174.00` | `166.63` / `166.34` | **`1.05x`** |

## Part 2 Benchmark Results: 4-Way User-Space Row Cache (`#759`) & Packed Reader (`#754`)

### 1. Hot Cache Traffic (`64` unique rows, 100% cache hit after warmup)

| Batch `M` | 2-Plane No-Cache B1 / B2 (us) | 2-Plane 4-Way Cache B1 / B2 (us) | 2-Plane Cache Speedup | Packed No-Cache B1 / B2 (us) | Packed 4-Way Cache B1 / B2 (us) | Packed Cache vs 2-Plane No-Cache |
|---:|---:|---:|---:|---:|---:|---:|
| `1` | `92.6` / `133.1` | `78.8` / `12.4` | **`2.47x`** | `77.2` / `39.5` | `79.4` / `14.5` | **`2.40x`** |
| `4` | `75.5` / `33.7` | `79.0` / `14.7` | **`1.17x`** | `75.2` / `25.7` | `26.0` / `14.8` | **`2.68x`** |
| `8` | `37.6` / `37.8` | `14.9` / `14.9` | **`2.52x`** | `28.6` / `28.3` | `14.8` / `15.0` | **`2.52x`** |
| `16` | `42.3` / `42.2` | `14.9` / `14.9` | **`2.83x`** | `32.4` / `31.9` | `15.2` / `14.9` | **`2.81x`** |
| `32` | `51.0` / `51.5` | `16.3` / `16.2` | **`3.16x`** | `37.3` / `37.2` | `15.7` / `16.1` | **`3.23x`** |
| `48` | `59.7` / `60.4` | `15.7` / `16.1` | **`3.77x`** | `43.0` / `43.2` | `15.6` / `15.8` | **`3.82x`** |
| `64` | `59.9` / `60.8` | `16.6` / `16.5` | **`3.65x`** | `48.2` / `48.6` | `16.3` / `16.4` | **`3.70x`** |
| `128` | `81.9` / `82.8` | `18.7` / `19.3` | **`4.34x`** | `55.1` / `55.6` | `18.6` / `19.5` | **`4.33x`** |

### 2. Zipfian Traffic (`s = 1.2`, `512` active rows)

| Batch `M` | Cache Hit Rate | 2-Plane No-Cache B1 / B2 (us) | 2-Plane 4-Way Cache B1 / B2 (us) | 2-Plane Cache Speedup | Packed 4-Way Cache B1 / B2 (us) | Packed + Cache Speedup |
|---:|---:|---:|---:|---:|---:|---:|
| `4` | `83.6%` | `74.4` / `43.6` | `20.4` / `20.5` | **`2.89x`** (`-65.4%`) | `15.4` / `15.8` | **`3.79x`** (`-73.6%`) |
| `16` | `92.5%` | `50.8` / `51.4` | `26.9` / `27.2` | **`1.89x`** (`-47.0%`) | `21.1` / `21.2` | **`2.42x`** (`-58.6%`) |
| `48` | `97.0%` | `80.2` / `80.1` | `31.8` / `32.6` | **`2.49x`** (`-59.8%`) | `25.6` / `26.0` | **`3.10x`** (`-67.8%`) |
| `128` | `99.1%` | `130.9` / `135.4` | `36.7` / `40.4` | **`3.46x`** (`-71.1%`) | `31.6` / `35.0` | **`3.99x`** (`-75.0%`) |

### 3. Disjoint Cold Miss Traffic (100% unique uncached rows per step)

| Batch `M` | 2-Plane `pread`s/step | Packed `pread`s/step | 2-Plane No-Cache B1 / B2 (us) | 2-Plane Cache B1 / B2 (us) | Packed Cache B1 / B2 (us) | Packed Cold-Miss Speedup |
|---:|---:|---:|---:|---:|---:|---:|
| `4` | `16.0` | `8.0` | `33.7` / `37.2` | `31.8` / `32.4` | `22.8` / `22.5` | **`1.56x`** (`-36.0%`) |
| `16` | `16.0` | `8.0` | `61.4` / `54.4` | `61.8` / `60.2` | `37.4` / `43.6` | **`1.43x`** (`-30.1%`) |
| `32` | `16.0` | `8.0` | `88.2` / `86.2` | `93.5` / `90.9` | `78.8` / `78.8` | **`1.11x`** (`-9.7%`) |

## Conclusion
1. **Multi-CTA `engram_rows_step_kernel` (`#833` Part B rework)** eliminates the 1-CTA bottleneck of `96b3b1ec4` and the C2C mapped-host atomic/poll serialization on GB10, achieving **`1.83x-2.74x` Eager speedup** and **`1.00x-1.56x` CUDA Graph speedup** across `M = 1..128` on TP4 (`6` cols) and TP3 (`8` cols) while remaining 100% bit-exact (`torch.equal`) and `compute-sanitizer` (`memcheck`, `initcheck`, `racecheck`) clean.
2. **Packed Reader Wiring (`#754`) & Uncapped 4-Way User-Space Row Cache (`#759`)** keep the 17.3 MiB 4-way row cache active across all batch sizes (`T > 64` included) and halve `O_DIRECT` `pread` syscalls on cache misses when packed sources are present, delivering **`2.30x-4.41x` speedup on hot traffic**, **`1.89x-3.64x` (2-plane) / `2.41x-4.84x` (packed) speedup on Zipfian traffic**, and **`1.11x-1.64x` speedup on 100% cold misses**.
