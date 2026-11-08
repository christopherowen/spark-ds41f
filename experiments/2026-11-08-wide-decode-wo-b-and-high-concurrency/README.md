# 2026-11-08 — Extend `WO-B` `WIDE_DECODE` to 128 Rows and Guard High-Concurrency (`97..299` Row) Decode Cliffs

## Context & Motivation

On `spark-ds41f:1107` (`cc20ed18be1f7cbf5c6c53cdee945f7cc77d5fc5`, vLLM patch head `320021e3c4dbb202d911ab70e2cd924c75420488`), three decode and collective cliffs appear across rows `97..299`:

1. **`WO-B` `WIDE_DECODE` stops at `96` rows while `CopyDrafter` (`c16`) and `c24..c32` replay `104..128`-row decode graphs**:
   - Even at default `max_num_seqs = 16`, `CopyDrafter` (`K = 5`, `V = 16`) widens `max_decode_rows` to **`107` rows** (`16 * 6 + 11`), capturing and replaying the **`104`-row and `112`-row** target `FULL` decode CUDA graphs (and `104, 112, 120, 128` at `max_num_seqs = 21..32`).
   - Across `97..128` rows, `WO-A` (`_WO_A_TILES` up to `128`), `fused_wqa_wkv` (`(3904, 5120)` up to `128`), `shared_expert` (`(1024, 5120)` up to `128`), and `vocab` (`ROW_TILES` up to `128`) all dispatch `128`-row single-wave decode tiles.
   - However, `WO-B` (`(5120, 3072)` on TP3 ranks 0–1, `(5120, 2048)` on TP4 and TP3 rank 2, and `(5120, 1024)` on TP8) stopped at `WIDE_DECODE = {96: 96}` and `DECODE_FP8_CONFIGS` stopped at `96`. Because `TileLangWOProjection.tilelang_decode_rows = 128` passes `m_tile=128` into `mxfp8_gemm`, `decode_tile_rows(M, 5120, K)` returned `DECODE_ROWS = 64` for `M in 97..128`, launching `2 * 40 = 80` CTAs (`2` M-tiles x `40` N-tiles) in **2 waves on 48 GB10 SMs** and reading the `10–15 MiB` `WO-B` weight matrix twice across all 80 layers.

2. **`sp_prefill.resolve_min_rows` and `all_reduce_rank_order` at `max_num_seqs > 32` (`205..299` rows)**:
   - `SparkNet` One-Shot RoCE AllReduce supports up to `SPARKNET_ROCE_ALLREDUCE_DISPATCH_MAX_BYTES = 2,097,152` (`2 MiB` = **`204` rows** at `5120` BF16).
   - Previously, `resolve_min_rows` set `sp_min_rows = max(max_cudagraph_capture_size, max_decode_rows) + 1`. At `max_num_seqs = 48` (`max_decode_rows = 288` or `299` with CopyDrafter), even when `max_cudagraph_capture_size <= 204` (`192`), `max_decode_rows` pushed `sp_min_rows` to `289..300`, creating a `205..299`-row gap where uncaptured batches exceeded `SparkNet`'s 2 MiB all-reduce limit and fell back to `PyNCCL` instead of entering SP prefill (`reduce_scatter` + `all_gather`, which `SparkNet` supports up to `408` rows on `tp3-triangle` / `tp4-switched`).
   - In addition, `all_reduce_rank_order` (`collectives.py` / `CudaCommunicator.all_reduce`) only checked `roce_comm.should_roce_allreduce(input_)` (`<= 204` rows), even though `all_reduce_rank_order` executes `reduce_scatter` + `all_gather` when `roce_comm.should_reduce_scatter(input_)` and `roce_comm.should_all_gather(rs_out)` hold (`<= 408` rows on `tp3-triangle` / `tp4-switched`).

3. **Unbounded full-depth per-request varlen decode captures in `dense_varlen_decode_shapes` for `max_num_seqs > 16`**:
   - In `vllm/v1/worker/gpu/cudagraph_utils.py`, `dense_varlen_decode_shapes` enumerated exact `(r, r * max_q)` full-depth varlen shapes for every `r in 1..max_num_reqs` up to `max_tokens`.
   - For `r > 16` (`102..192` tokens at `max_q = 6`), the 8-row (`104..128`) and 16-row (`144..192`) padded target `FULL` decode ladder already covers every uniform `(r, 6r)` batch via `pad_for_cudagraph(6r)` (`r <= num_tokens`), so exact `(r, 6r)` shapes for `r = 17..32` redundantly added `27..32` target + DSpark context graphs (`138..142 MiB` pool memory) at `max_num_seqs = 32` and `56..68` graphs (`346..422 MiB` pool memory) at `max_num_seqs = 48`.

---

## Base & Candidate Identity

- **Base repository commit**: `cc20ed18be1f7cbf5c6c53cdee945f7cc77d5fc5` (`spark-ds41f:1107`)
- **Base vLLM commit / tree**: `320021e3c4dbb202d911ab70e2cd924c75420488` / `ed1bdb7a7684da33d403a055521ac0c4dd7ca670`
- **Candidate patch**: `experiments/2026-11-08-wide-decode-wo-b-and-high-concurrency/vllm/0762-perf-dsv41-extend-WO-B-WIDE_DECODE-to-128-rows-and-g.patch`
- **Candidate vLLM commit / tree**: `09afe88dd891bf5cd5d841c51ab5e0970228f1f5` / `5cea942238ee68a4ef7497c22e1d4ebb82aa0611`
- **Candidate `patchset_sha256`**: `d4eec5701d0300d210dcd550aaecb845bf546ff8e49b40cfbe38b3d7df38721b`
- **Hardware**: NVIDIA GB10 (`sp4`, SM121, 48 SMs, container `spark-ds41f:1107` under `flock /tmp/gpu.lock`)

---

## Measured Results (2 Container Boots on `sp4`)

### 1. `WO-B` `mxfp8_gemm` Kernel & Tile Candidates (`M = 64..192`, 2 Boots)

Receipts:
- `receipts/wo_b_boot1.json`
- `receipts/wo_b_boot2.json`

All candidates are **100% bit-identical (`torch.equal == True`, `max_abs_diff = 0.0`)** against both the `1107` baseline and `mxfp8_gemm_prefill` (`m_tile=128`) across all tested shapes and row counts.

Why `decode_128_128x128_s2_t256` (`block_M=128, block_N=128, block_K=128, num_stages=2, threads=256`) wins at `M = 97..128`:
- `N = 5120`: `block_N = 128` gives `5120 / 128 = 40` N-tiles (`1 * 40 = 40` CTAs, **1 single wave on 48 GB10 SMs**).
- By contrast, the `1107` fallback (`decode_64`, `block_M=64, block_N=128`) and any `block_N=64` 128-row tile (`decode_128_64x128_s3_t256`, `decode_128_64x128_s2_t256`) launch `80` CTAs (**2 waves on 48 SMs**).
- Unlike `prefill_128x128`, `decode_128_128x128_s2_t256` compiles through the unspecialized `mxfp8_gemm_decode` pipeline, avoiding TMA warp-specialization consumer barriers on padded `TileLangWOProjection` workspaces (`_q_act[:padded_m]`).
- At `M = 160..192`, `192`-row decode tiles (`73.66 us` on `K=3072`, `50.41 us` on `K=2048`) are slower than `128x128` prefill (`64.90 us` on `K=3072`, `46.55 us` on `K=2048`), so `WIDE_DECODE` and `TileLangWOProjection.tilelang_decode_rows` stop at `128` rows.

#### `(N=5120, K=3072)` — TP3 Ranks 0–1 `WO-B` (2-Boot Mean `[Boot 1 / Boot 2]`, microseconds)

| `M` | `1107` Baseline (`64`-row fallback at `97..128`) | `decode_128` (`64x128, s3`) | `decode_128` (`64x128, s2`) | **`decode_128` (`128x128, s2`) [Selected]** | Speedup vs `1107` |
|---:|---:|---:|---:|---:|---:|
| `64` | `30.88` `[29.93 / 31.83]` | `30.88` | `30.88` | **`30.88`** `[29.93 / 31.83]` | `0.0%` (uses `64`-row tile) |
| `96` | `35.09` `[35.24 / 34.93]` | `35.09` | `35.09` | **`35.09`** `[35.24 / 34.93]` | `0.0%` (uses `96`-row tile) |
| `104` | `45.41` `[46.01 / 44.80]` | `47.55` `[47.25 / 47.84]` | `44.91` `[47.41 / 42.42]` | **`37.75`** `[38.65 / 36.86]` | **`-16.9%` (`-7.65 us`)** |
| `112` | `43.64` `[41.53 / 45.74]` | `51.72` `[49.74 / 53.69]` | `44.81` `[47.62 / 42.01]` | **`37.03`** `[39.24 / 34.82]` | **`-15.1%` (`-6.61 us`)** |
| `120` | `42.35` `[43.83 / 40.87]` | `51.01` `[51.51 / 50.52]` | `42.20` `[43.25 / 41.15]` | **`36.20`** `[37.59 / 34.81]` | **`-14.5%` (`-6.15 us`)** |
| `128` | `40.95` `[40.95 / 40.95]` | `49.26` `[49.91 / 48.61]` | `44.14` `[47.17 / 41.11]` | **`35.31`** `[35.79 / 34.83]` | **`-13.8%` (`-5.64 us`)** |

#### `(N=5120, K=2048)` — TP4 All Ranks & TP3 Rank 2 `WO-B` (2-Boot Mean `[Boot 1 / Boot 2]`, microseconds)

| `M` | `1107` Baseline (`64`-row fallback at `97..128`) | `decode_128` (`64x128, s3`) | `decode_128` (`64x128, s2`) | **`decode_128` (`128x128, s2`) [Selected]** | Speedup vs `1107` |
|---:|---:|---:|---:|---:|---:|
| `64` | `21.92` `[21.65 / 22.19]` | `21.92` | `21.92` | **`21.92`** `[21.65 / 22.19]` | `0.0%` (uses `64`-row tile) |
| `96` | `26.86` `[26.35 / 27.36]` | `26.86` | `26.86` | **`26.86`** `[26.35 / 27.36]` | `0.0%` (uses `96`-row tile) |
| `104` | `34.53` `[33.47 / 35.59]` | `36.75` `[36.04 / 37.45]` | `33.03` `[32.85 / 33.20]` | **`28.32`** `[27.30 / 29.34]` | **`-18.0%` (`-6.21 us`)** |
| `112` | `31.67` `[31.72 / 31.62]` | `35.15` `[34.96 / 35.34]` | `32.51` `[30.72 / 34.30]` | **`26.88`** `[25.85 / 27.91]` | **`-15.1%` (`-4.79 us`)** |
| `120` | `31.88` `[32.61 / 31.14]` | `37.13` `[36.35 / 37.90]` | `30.89` `[30.69 / 31.08]` | **`26.29`** `[27.34 / 25.23]` | **`-17.5%` (`-5.59 us`)** |
| `128` | `32.88` `[31.05 / 34.72]` | `37.75` `[37.39 / 38.10]` | `31.51` `[30.48 / 32.54]` | **`28.51`** `[26.84 / 30.18]` | **`-13.3%` (`-4.38 us`)** |

#### `(N=5120, K=1024)` — TP8 `WO-B` (2-Boot Mean `[Boot 1 / Boot 2]`, microseconds)

| `M` | `1107` Baseline (`64`-row fallback at `97..128`) | `decode_128` (`64x128, s3`) | `decode_128` (`64x128, s2`) | **`decode_128` (`128x128, s2`) [Selected]** | Speedup vs `1107` |
|---:|---:|---:|---:|---:|---:|
| `64` | `13.97` `[13.86 / 14.07]` | `13.97` | `13.97` | **`13.97`** `[13.86 / 14.07]` | `0.0%` (uses `64`-row tile) |
| `96` | `16.41` `[16.05 / 16.77]` | `16.41` | `16.41` | **`16.41`** `[16.05 / 16.77]` | `0.0%` (uses `96`-row tile) |
| `104` | `20.35` `[20.31 / 20.39]` | `22.63` `[21.84 / 23.43]` | `20.51` `[20.08 / 20.94]` | **`18.00`** `[18.15 / 17.85]` | **`-11.5%` (`-2.35 us`)** |
| `112` | `19.48` `[18.81 / 20.16]` | `22.25` `[22.13 / 22.37]` | `19.65` `[19.72 / 19.57]` | **`17.15`** `[17.15 / 17.15]` | **`-12.0%` (`-2.33 us`)** |
| `120` | `20.90` `[21.92 / 19.88]` | `22.84` `[23.41 / 22.28]` | `20.00` `[20.36 / 19.65]` | **`16.64`** `[16.43 / 16.86]` | **`-20.4%` (`-4.26 us`)** |
| `128` | `19.86` `[19.44 / 20.28]` | `22.37` `[22.58 / 22.15]` | `20.89` `[22.02 / 19.76]` | **`17.30`** `[17.38 / 17.22]` | **`-12.9%` (`-2.56 us`)** |

#### End-to-End `TileLangWOProjection` (`WO-A` + Quantize + `WO-B`, 2-Boot Mean `[Boot 1 / Boot 2]`, microseconds)

| Shape | `M` | `1107` Baseline (`us`) | Candidate (`128`-row `WO-B`) (`us`) | Delta (`us` / `%`) | Per-Step 80-Layer Savings |
|---|---:|---:|---:|---:|---:|
| TP3 Ranks 0–1 `(5120, 3072)` | `104` | `97.96` `[96.44 / 99.49]` | **`91.02`** `[89.38 / 92.65]` | **`-6.95 us` (`-7.1%`)** | **`-0.56 ms / step`** |
| TP3 Ranks 0–1 `(5120, 3072)` | `112` | `98.82` `[98.71 / 98.94]` | **`90.35`** `[92.41 / 88.30]` | **`-8.47 us` (`-8.6%`)** | **`-0.68 ms / step`** |
| TP3 Ranks 0–1 `(5120, 3072)` | `120` | `101.66` `[102.31 / 101.00]` | **`94.46`** `[96.45 / 92.47]` | **`-7.20 us` (`-7.1%`)** | **`-0.58 ms / step`** |
| TP3 Ranks 0–1 `(5120, 3072)` | `128` | `100.13` `[99.83 / 100.43]` | **`88.34`** `[88.36 / 88.32]` | **`-11.79 us` (`-11.8%`)** | **`-0.94 ms / step`** |
| TP4 / TP3 Rank 2 `(5120, 2048)` | `104` | `69.92` `[70.39 / 69.44]` | **`64.43`** `[65.06 / 63.80]` | **`-5.49 us` (`-7.8%`)** | **`-0.44 ms / step`** |
| TP4 / TP3 Rank 2 `(5120, 2048)` | `112` | `72.24` `[71.88 / 72.59]` | **`65.41`** `[65.91 / 64.91]` | **`-6.83 us` (`-9.5%`)** | **`-0.55 ms / step`** |
| TP4 / TP3 Rank 2 `(5120, 2048)` | `120` | `70.11` `[70.33 / 69.90]` | **`63.70`** `[63.61 / 63.78]` | **`-6.42 us` (`-9.2%`)** | **`-0.51 ms / step`** |
| TP4 / TP3 Rank 2 `(5120, 2048)` | `128` | `71.09` `[70.48 / 71.71]` | **`64.61`** `[64.07 / 65.16]` | **`-6.48 us` (`-9.1%`)** | **`-0.52 ms / step`** |

---

### 2. High-Concurrency (`max_num_seqs = 8, 16, 32, 48`) CUDA Graph Scaling & Pool Memory (2 Boots on `sp4`)

Receipts:
- `receipts/high_concurrency_boot1.json`
- `receipts/high_concurrency_boot2.json`

Bounding full-depth per-request varlen decode shapes in `dense_varlen_decode_shapes` at `_FULL_DEPTH_VARLEN_DECODE_MAX_REQS = 16`:
- Leaves `max_num_seqs = 8` and `max_num_seqs = 16` **100% identical** (`0` graphs changed).
- Preserves `FULL` decode CUDA graph dispatch for **100% of uniform decode/verify batches (`r = 1..max_num_seqs, q = 1..6`)** via the 8-row (`104..128`) and 16-row (`144..192`) padded ladder (`all_uniform_batches_dispatch_full == True`).

| Profile | `max_num_seqs` | `1107` Total Graphs (`FULL` / `PIECEWISE` / `DSpark`) | Candidate Total Graphs (`FULL` / `PIECEWISE` / `DSpark`) | Graph Delta | `1107` Graph Pool (`MiB`) | Candidate Graph Pool (`MiB`) | Pool Saved (`MiB` / `%`) |
|---|---:|---|---|---:|---:|---:|---:|
| `uniform_dspark_k5` (`q=6`) | `8` | `71` (`24` / `29` / `18`) | `71` (`24` / `29` / `18`) | `0` (`0.0%`) | `64.0` | `64.0` | `0.0 MiB` (`0.0%`) |
| `uniform_dspark_k5` (`q=6`) | `16` | `103` (`40` / `37` / `26`) | `103` (`40` / `37` / `26`) | `0` (`0.0%`) | `122.0` | `122.0` | `0.0 MiB` (`0.0%`) |
| `uniform_dspark_k5` (`q=6`) | `32` | `167` (`72` / `53` / `42`) | **`140`** (`61` / `53` / `26`) | **`-27` (`-16.2%`)** | `388.0` | **`250.0`** | **`-138.0 MiB` (`-35.6%`)** |
| `uniform_dspark_k5` (`q=6`) | `48` | `226` (`99` / `69` / `58`) | **`170`** (`75` / `69` / `26`) | **`-56` (`-24.8%`)** | `746.0` | **`400.0`** | **`-346.0 MiB` (`-46.4%`)** |
| `copy_drafter_k5_v16` (`q=17`) | `8` | `126` (`76` / `31` / `19`) | `126` (`76` / `31` / `19`) | `0` (`0.0%`) | `88.0` | `88.0` | `0.0 MiB` (`0.0%`) |
| `copy_drafter_k5_v16` (`q=17`) | `16` | `162` (`93` / `39` / `30`) | `162` (`93` / `39` / `30`) | `0` (`0.0%`) | `156.0` | `156.0` | `0.0 MiB` (`0.0%`) |
| `copy_drafter_k5_v16` (`q=17`) | `32` | `230` (`121` / `55` / `54`) | **`198`** (`113` / `55` / `30`) | **`-32` (`-13.9%`)** | `462.0` | **`320.0`** | **`-142.0 MiB` (`-30.7%`)** |
| `copy_drafter_k5_v16` (`q=17`) | `48` | `293` (`148` / `71` / `74`) | **`225`** (`127` / `71` / `27`) | **`-68` (`-23.2%`)** | `884.0` | **`462.0`** | **`-422.0 MiB` (`-47.7%`)** |

---

### 3. `sp_prefill.resolve_min_rows` & `all_reduce_rank_order` RoCE Threshold Guard

| `max_num_seqs` | `max_cudagraph_capture_size` | `max_decode_rows` (`q=17`) | `1107` `sp_min_rows` | `1107` Uncaptured `PyNCCL` Gap (`>204` rows) | Candidate `sp_min_rows` | Candidate Uncaptured `PyNCCL` Gap |
|---:|---:|---:|---:|---:|---:|---:|
| `16` | `96` | `107` | `205` | `0` rows | `205` | **`0` rows** |
| `32` | `192` | `203` | `205` | `0` rows | `205` | **`0` rows** |
| `48` (capped graph `<= 204`) | `192` | `299` | `300` | **`95` rows (`205..299`)** | **`205`** | **`0` rows** |
| `48` (uncapped graph `288`) | `288` | `299` | `300` | `95` rows (`205..299`) | **`289` + explicit warning** | **`0` uncaptured rows** (`289..299` enter SP prefill) |

Additionally, `all_reduce_rank_order` (`collectives.py` and `CudaCommunicator.all_reduce`) now checks `roce_comm.should_reduce_scatter(x)` and `roce_comm.should_all_gather(rs_out)` instead of `should_roce_allreduce(x)`, keeping rank-ordered `reduce_scatter` + `all_gather` collectives on `SparkNet` RoCE up to `408` rows (`4 MiB`) on `tp3-triangle` / `tp4-switched`.

---

## Reproducibility Commands

1. **2-Boot `WO-B` Kernel & `TileLangWOProjection` Benchmark on `sp4`**:
   ```bash
   ssh sp4 'flock /tmp/gpu.lock docker run --rm --gpus all --ipc=host \
     -v /tmp:/tmp -v /tmp/tilelang_root_cache:/root/.tilelang/cache \
     --entrypoint python3 spark-ds41f:1107 /tmp/bench_wo_b.py /tmp/wo_b_boot1.json'
   ```
2. **2-Boot High-Concurrency Graph Scaling & `sp_min_rows` Benchmark on `sp4`**:
   ```bash
   ssh sp4 'flock /tmp/gpu.lock docker run --rm --gpus all --ipc=host \
     -v /tmp:/tmp --entrypoint python3 spark-ds41f:1107 \
     /tmp/bench_high_concurrency.py /tmp/high_concurrency_boot1.json'
   ```
3. **2-Boot Verification Unit Test Suite (`225 passed` per boot)**:
   - Receipts: `receipts/pytest_boot1.txt`, `receipts/pytest_boot2.txt`
   - Tests run:
     - `tests/kernels/quantization/test_deepseek_v41_tilelang_linear.py::test_wo_b_wide_decode_128_rows_matches_prefill`
     - `tests/kernels/quantization/test_deepseek_v41_tilelang_linear.py::test_mxfp8_decode_tile_repeats`
     - `tests/kernels/quantization/test_deepseek_v41_block_scale_words.py`
     - `tests/models/test_deepseek_v4_1_sp_prefill.py`
     - `tests/v1/cudagraph/test_cudagraph_manager.py`
