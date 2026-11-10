# 2026-11-08 — Sparse MLA Stage-1 KV Gather and Shared-Memory Staging Pipeline (`0762`)

## Context & Base

- **Repository base commit**: `cc20ed18b` (`main`)
- **Candidate base**: `experiments/2026-10-26-spark-ds41f-candidate/1107`
  - vLLM series `0001..0761` (`641` patches, base patch head `320021e3c4dbb202d911ab70e2cd924c75420488`, tree `ed1bdb7a7684da33d403a055521ac0c4dd7ca670`)
  - TileLang `0.1.8+dea9b489.dsv41.p13` (`tree e2b001828fd4e38e6a2eee61f818d5083cb1335f`)
  - SparkNet `0.1.0` (`tree 8cc28a5cd3e683c1f114ed843453e3ce34f323e0`)
- **Candidate patch**: `experiments/2026-11-08-sparse-mla-kv-pipeline/vllm/0762-perf-dsv41-sparse-mla-stage1-kv-gather-and-barrier-opt.patch`
  - Patch head: `a0de45b3a178e3ef681835da9a3766a1198dd728`
  - Expected vLLM tree: `f2baa167507ebcbeb526cd70b0490591af4d15be`
- **Hardware**: `sp8` (NVIDIA GB10, SM121)

## Root-Cause & CUDA C++ Codegen Analysis

Inspecting the generated CUDA C++ (`kernel.get_kernel_source()`) and PTX resource usage of `sparse_mla_fwd_v41` across `split` (`block_H=16, threads=128`), `decode` (`block_H=32, threads=256` on TP=3 `heads=24`; `block_H=16, threads=128` on TP=4 `heads=16`), and `extend` identified two architectural properties and five Stage-1 bottlenecks:

### Why `staged -> KV_shared` Already Forms a 2-Stage Software Pipeline
In `sparse_mla_fwd_v41`, `dequant_chunk(c)` reads compressed records from `staged` (`[32, 528]` uint8) and writes dequantized BF16 keys/values to `KV_shared` (`[32, 512]` bfloat16). Immediately after `dequant_chunk(c)` and a barrier, `stage_chunk(c + 1)` issues `cp.async.cg.shared.global` (`16 B`) for chunk `c + 1` into `staged` before `chunk_update(c)` runs its two `T.gemm` calls (`Q @ KV^T` and `P @ KV`) and online-softmax updates on `KV_shared`. Thus `staged` (receiving compressed chunk `c + 1`) and `KV_shared` (consumed by MMA on chunk `c`) already act as a 2-stage pipeline without needing a second compressed `staged` buffer (+16.9 KiB shared memory).

### Five Stage-1 Bottlenecks Eliminated in `0762`

1. **Parallel 4-Warp (`G * BI = 128`-Thread) Slot Gather in `decode_body`**:
   - In `1107`, `decode_body` resolved its segment's `128` slots (`G = 4`, `BI = 32`) via a 4-iteration serial loop `for c in T.serial(G):` calling `gather_swa(..., BI=32)` or `gather_idx(..., BI=32)`.
   - In the compiled CUDA C++, this emitted `for (int c = 0; c < 4; ++c) { if (threadIdx.x < 32) { ... } }`: only **1 warp (32 threads)** executed while the remaining 3 or 7 warps waited at `__syncthreads()`, serializing 4 waves of dependent `IdxIndices -> IdxPageTable` global memory loads.
   - Because `swa_blocks = 4` is a multiple of `G = 4`, each segment is purely SWA (`seg * G < swa_blocks`) or purely indexed (`seg * G >= swa_blocks`). Calling `gather_swa(..., G * BI=128)` or `gather_idx(..., G * BI=128)` once per segment resolves all `128` slots in **1 parallel wave across 4 warps (128 threads)** (`4x` memory-level parallelism on `IdxIndices` and `IdxPageTable`, zero serial loop).

2. **Exact-Stride `IDX_RECORD_BYTES` (`288 B`) Shared-Memory View for Indexed `cp.async` Staging**:
   - In `1107`, `staged` had shape `[32, 528]`. Staging `288`-byte `IdxRecords` (`18` 16-byte segments) into `staged[e // 18, e % 18 * 16]` had byte offset `(e // 18) * 528 + (e % 18) * 16`, forcing every `cp.async` in indexed segments (segments `1..4`, 80% of all segments) to evaluate non-power-of-two `(((k * 128 + (tx >> 1)) / 9) * 528) + (((k * 256 + tx) % 18) * 16)`.
   - Rounding SWA staging rows up to a multiple of 6 (`swa_rows = tilelang.cdiv(BI, 6) * 6 = 36`, `36 * 528 = 66 * 288 = 19,008 B`, `+2,112 B` shared memory) and viewing the same shared buffer as `staged_idx = T.view(staged_swa, [66, 288], T.uint8)` collapses `(e // 18) * 288 + (e % 18) * 16` algebraically to **`(k * threads + tx) * 16`** (`(k_7 * 4096) + (threadIdx.x * 16)` in generated CUDA C++, with zero `/ 9`, `% 18`, or `* 528`).

3. **Preloading `copy_slot` Before `ptx_wait_group(0)` in `decode_body` (-1 Barrier per Preloaded Chunk)**:
   - In `1107`, `stage_records` read `copy_slot[k] = slots[...]` after `dequant_chunk` + `T.sync_threads()`. Because `slots` and `staged` share `buf_dyn_shmem`, TileLang's `ThreadSync("shared.dyn")` pass inserted an extra `__syncthreads()` between reading `slots` and issuing `T.ptx_cp_async(staged[...])`.
   - Splitting `stage_chunk` into `load_copy_slots` and `issue_stage_chunk` and calling `load_copy_slots(chunk + 1, ...)` before `T.ptx_wait_group(0)` (while chunk `c`'s `cp.async` completes) keeps `copy_slot` in registers when `issue_stage_chunk` runs after `dequant_chunk`, eliminating that auto-inserted barrier.

4. **Register-Predicated Mask Check (`acc_s[h, i] != -T.infinity(accum_dtype)`) in `chunk_update` (-4 `LDS` and -1 Barrier per Chunk)**:
   - In `chunk_update`, the first loop sets masked entries (`slots[base + i] < 0`) to `-T.infinity(accum_dtype)`. Re-reading `slots[base + i] >= 0` from `buf_dyn_shmem` after `T.reduce_max` wrote to shared reduction workspace caused TileLang to emit 4 shared-memory loads and insert an extra `__syncthreads()` before `exp2f`.
   - Checking `acc_s[h, i] != -T.infinity(accum_dtype)` in registers eliminates both the 4 shared-memory loads and the barrier while preserving IEEE-754 `NaN` propagation (`NaN != -inf` is `true`). Combined with (3), per-chunk barriers in `decode_body` drop from **10 to 8** (**-8 `__syncthreads()` per segment**).

5. **Segment-Granularity Active Loop Bound (`active_segs`) in `extend_body`**:
   - Bounding `extend_body`'s outer segment loop by `active_segs = (swa_blocks // G) + T.min(tilelang.cdiv(T.max(IdxLengths[row], 0), G * BI), segments - (swa_blocks // G))` (stored in `T.alloc_var(T.int32)` before the loop) skips trailing inactive indexed segments on short contexts (**-52% latency at `idx_len = 64`**) while keeping `chunk < chunks` constant-folded to `true` inside the inner `c` loop.

## Two-Boot Benchmark Results on `sp8` (CUDA-Graph Replay, 100% `torch.equal` Bit-Exact)

Full receipts are recorded in `receipts/boot-1.json` and `receipts/boot-2.json`. Both standalone Stage-1 (`Cand_S1`, implemented in `0762`) and its composition with the Stage-2 online-softmax identity (`Cand_S1+S2`) are 100% bit-exact (`torch.equal`) with `1107`.

### `TP3_h24` (`heads = 24`) — Medium Context (`swa = 128, idx = 512`)

| `M` | Route | `1107` Boot 1 / Boot 2 (us) | `Cand_S1` Boot 1 / Boot 2 (us) | `Cand_S1` Mean Delta | `Cand_S1+S2` Boot 1 / Boot 2 (us) | `Cand_S1+S2` Mean Delta | Bit-Exact |
|---:|:---:|---:|---:|---:|---:|---:|:---:|
| 1 | `split` | `10.85 / 10.92` | `10.36 / 10.44` | **`-0.49 us (-4.5%)`** | `10.32 / 10.41` | `-0.52 us (-4.8%)` | `True` |
| 2 | `split` | `11.19 / 11.07` | `10.55 / 10.52` | **`-0.60 us (-5.4%)`** | `10.74 / 10.62` | `-0.45 us (-4.0%)` | `True` |
| 4 | `split` | `11.90 / 12.07` | `11.63 / 11.21` | **`-0.57 us (-4.7%)`** | `11.24 / 11.54` | `-0.59 us (-4.9%)` | `True` |
| 6 | `decode` | `14.83 / 14.90` | `14.01 / 13.94` | **`-0.89 us (-6.0%)`** | `13.97 / 13.61` | `-1.08 us (-7.2%)` | `True` |
| 8 | `decode` | `15.74 / 15.83` | `14.61 / 14.84` | **`-1.06 us (-6.7%)`** | `14.34 / 14.54` | `-1.35 us (-8.5%)` | `True` |
| 12 | `decode` | `27.28 / 27.12` | `25.76 / 25.80` | **`-1.42 us (-5.2%)`** | `25.57 / 25.59` | `-1.62 us (-6.0%)` | `True` |
| 16 | `decode` | `28.57 / 28.55` | `26.80 / 26.95` | **`-1.69 us (-5.9%)`** | `26.78 / 26.97` | `-1.69 us (-5.9%)` | `True` |
| 24 | `decode` | `41.38 / 40.93` | `39.07 / 38.93` | **`-2.16 us (-5.2%)`** | `38.76 / 38.98` | `-2.29 us (-5.6%)` | `True` |
| 32 | `extend` | `47.57 / 46.93` | `47.19 / 46.80` | **`-0.26 us (-0.5%)`** | `46.65 / 47.93` | `+0.04 us (+0.1%)` | `True` |
| 48 | `extend` | `48.02 / 48.00` | `47.25 / 47.52` | **`-0.63 us (-1.3%)`** | `47.09 / 47.46` | `-0.74 us (-1.5%)` | `True` |

### `TP3_h24` (`heads = 24`) — Short Context (`swa = 128, idx = 64`)

| `M` | Route | `1107` Boot 1 / Boot 2 (us) | `Cand_S1` Boot 1 / Boot 2 (us) | `Cand_S1` Mean Delta | `Cand_S1+S2` Boot 1 / Boot 2 (us) | `Cand_S1+S2` Mean Delta | Bit-Exact |
|---:|:---:|---:|---:|---:|---:|---:|:---:|
| 1 | `split` | `10.93 / 10.92` | `10.75 / 10.44` | **`-0.33 us (-3.0%)`** | `10.39 / 10.41` | `-0.53 us (-4.8%)` | `True` |
| 2 | `split` | `10.99 / 11.09` | `10.38 / 10.53` | **`-0.58 us (-5.3%)`** | `10.38 / 10.51` | `-0.60 us (-5.4%)` | `True` |
| 4 | `split` | `11.74 / 11.74` | `11.11 / 11.23` | **`-0.57 us (-4.9%)`** | `11.10 / 11.14` | `-0.62 us (-5.3%)` | `True` |
| 6 | `decode` | `13.92 / 13.88` | `13.08 / 13.13` | **`-0.80 us (-5.7%)`** | `12.77 / 12.82` | `-1.11 us (-8.0%)` | `True` |
| 8 | `decode` | `13.82 / 13.82` | `13.50 / 13.16` | **`-0.49 us (-3.5%)`** | `12.97 / 13.12` | `-0.78 us (-5.6%)` | `True` |
| 12 | `decode` | `14.76 / 14.72` | `13.59 / 13.67` | **`-1.11 us (-7.5%)`** | `13.63 / 13.57` | `-1.14 us (-7.7%)` | `True` |
| 16 | `decode` | `14.92 / 15.09` | `14.10 / 14.44` | **`-0.74 us (-4.9%)`** | `13.93 / 13.93` | `-1.08 us (-7.2%)` | `True` |
| 24 | `decode` | `17.08 / 17.04` | `15.56 / 15.70` | **`-1.43 us (-8.4%)`** | `15.98 / 16.25` | `-0.94 us (-5.5%)` | `True` |
| 32 | `extend` | `47.71 / 46.77` | `22.57 / 22.40` | **`-24.76 us (-52.4%)`** | `22.32 / 22.40` | `-24.88 us (-52.7%)` | `True` |
| 48 | `extend` | `47.31 / 47.49` | `22.96 / 23.09` | **`-24.38 us (-51.4%)`** | `22.75 / 23.03` | `-24.51 us (-51.7%)` | `True` |

### `TP4_h16` (`heads = 16`) — Medium Context (`swa = 128, idx = 512`)

| `M` | Route | `1107` Boot 1 / Boot 2 (us) | `Cand_S1` Boot 1 / Boot 2 (us) | `Cand_S1` Mean Delta | `Cand_S1+S2` Boot 1 / Boot 2 (us) | `Cand_S1+S2` Mean Delta | Bit-Exact |
|---:|:---:|---:|---:|---:|---:|---:|:---:|
| 1 | `split` | `10.81 / 10.76` | `10.35 / 10.26` | **`-0.48 us (-4.5%)`** | `10.36 / 10.14` | `-0.54 us (-5.0%)` | `True` |
| 2 | `split` | `11.09 / 10.87` | `10.42 / 10.27` | **`-0.64 us (-5.8%)`** | `10.42 / 10.59` | `-0.48 us (-4.3%)` | `True` |
| 4 | `split` | `11.31 / 11.12` | `10.69 / 10.84` | **`-0.45 us (-4.0%)`** | `10.67 / 10.91` | `-0.43 us (-3.8%)` | `True` |
| 6 | `split` | `11.75 / 11.77` | `10.99 / 11.07` | **`-0.73 us (-6.2%)`** | `11.12 / 11.26` | `-0.57 us (-4.8%)` | `True` |
| 8 | `split` | `12.19 / 12.44` | `11.77 / 11.73` | **`-0.57 us (-4.6%)`** | `11.83 / 12.02` | `-0.39 us (-3.2%)` | `True` |
| 12 | `decode` | `23.16 / 23.10` | `21.52 / 21.50` | **`-1.62 us (-7.0%)`** | `21.22 / 21.22` | `-1.91 us (-8.3%)` | `True` |
| 16 | `decode` | `24.21 / 24.28` | `22.46 / 22.59` | **`-1.72 us (-7.1%)`** | `22.25 / 22.02` | `-2.11 us (-8.7%)` | `True` |
| 24 | `decode` | `34.48 / 34.45` | `32.51 / 32.27` | **`-2.08 us (-6.0%)`** | `31.92 / 31.98` | `-2.52 us (-7.3%)` | `True` |
| 32 | `extend` | `40.92 / 40.78` | `40.66 / 40.59` | **`-0.23 us (-0.6%)`** | `40.71 / 40.58` | `-0.21 us (-0.5%)` | `True` |
| 48 | `extend` | `41.48 / 41.42` | `41.12 / 41.58` | **`-0.10 us (-0.2%)`** | `41.17 / 41.11` | `-0.31 us (-0.7%)` | `True` |

### `TP4_h16` (`heads = 16`) — Short Context (`swa = 128, idx = 64`)

| `M` | Route | `1107` Boot 1 / Boot 2 (us) | `Cand_S1` Boot 1 / Boot 2 (us) | `Cand_S1` Mean Delta | `Cand_S1+S2` Boot 1 / Boot 2 (us) | `Cand_S1+S2` Mean Delta | Bit-Exact |
|---:|:---:|---:|---:|---:|---:|---:|:---:|
| 1 | `split` | `10.82 / 10.86` | `10.35 / 10.47` | **`-0.43 us (-4.0%)`** | `10.33 / 10.62` | `-0.37 us (-3.4%)` | `True` |
| 2 | `split` | `10.89 / 11.19` | `10.36 / 10.83` | **`-0.45 us (-4.0%)`** | `10.22 / 10.10` | `-0.88 us (-8.0%)` | `True` |
| 4 | `split` | `11.17 / 11.40` | `10.70 / 10.67` | **`-0.60 us (-5.3%)`** | `10.68 / 10.68` | `-0.61 us (-5.4%)` | `True` |
| 6 | `split` | `11.61 / 11.45` | `10.97 / 10.91` | **`-0.59 us (-5.1%)`** | `11.01 / 10.98` | `-0.54 us (-4.6%)` | `True` |
| 8 | `split` | `12.13 / 12.19` | `11.62 / 11.61` | **`-0.55 us (-4.5%)`** | `11.63 / 11.59` | `-0.55 us (-4.5%)` | `True` |
| 12 | `decode` | `12.50 / 12.47` | `11.51 / 11.50` | **`-0.98 us (-7.9%)`** | `11.53 / 11.47` | `-0.99 us (-7.9%)` | `True` |
| 16 | `decode` | `13.06 / 12.98` | `12.01 / 12.00` | **`-1.02 us (-7.8%)`** | `11.98 / 12.07` | `-1.00 us (-7.6%)` | `True` |
| 24 | `decode` | `14.58 / 14.61` | `13.38 / 13.43` | **`-1.19 us (-8.2%)`** | `13.47 / 13.51` | `-1.11 us (-7.6%)` | `True` |
| 32 | `extend` | `40.47 / 40.48` | `19.04 / 19.06` | **`-21.43 us (-52.9%)`** | `19.07 / 19.10` | `-21.39 us (-52.8%)` | `True` |
| 48 | `extend` | `41.00 / 41.01` | `19.65 / 19.60` | **`-21.38 us (-52.1%)`** | `19.62 / 19.46` | `-21.47 us (-52.3%)` | `True` |

## Verification & Unit Tests

Two-boot unit test and PDL verification receipts (`receipts/pytest-boot1.txt`, `receipts/pytest-boot2.txt`):
- `25 passed, 35 deselected` (`PDL sources: 27 kernels launched, 18 with PDL, 0 violations`), including `test_sparse_mla_short_indexed_segments_match_across_routes`.
