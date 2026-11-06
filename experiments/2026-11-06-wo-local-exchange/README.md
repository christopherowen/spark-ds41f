# Per-layer sequence-parallel `WO` local `MXFP8` exchange (`0731`)

- **Date**: 2026-11-06
- **Base deployment commit**: `0b3e98934f0aa7c7c251b79dae3ac148f869304e` (Candidate `1107`, `experiments/2026-10-26-spark-ds41f-candidate/1107`)
- **Base vLLM commit**: `c47b54ae471a24b830e3515212b8c0bf5ee14604` (tree `13e4781d19bc4a18b0d3d4daad7921c465e3355e`, `0001..0730`)
- **Candidate vLLM commit**: `b622187409c9916868da76407007abd79a7570d7` (tree `979f3985aaba6dfe1a0df37da62aadd715b2e97f`, `0001..0731`, `patchset_sha256` `40ce9593d5c261f03ff2cd48cc3aa4f2b035506c5559240e9676bea2296e5904`)
- **Intended causal variable**: Exchange `WO-A`'s native `MXFP8` `(xq, sf)` output (`float8_e4m3fn` values + `uint8` `UE8M0` scales) across tensor-parallel ranks on sequence-parallel encoder attention layers (`layers 0..19`) and project each rank's `WO-B` shard on the `L = ceil(T / world)` local rows into `parts` (`[world, L, 5120]` BF16), instead of projecting `WO-B` on all `T` rows per rank in slices and exchanging `5120`-column BF16 hidden states.

## 1. Mechanism and Bit-Exactness Proof (Zero Precision Reduction)

In DeepSeek-V4.1-Flash (`hidden_size = 5120`, `num_attention_heads = 64`, `o_lora_rank = 1024`, `o_groups = 8`, `head_dim = 512`):
1. `TileLangWOProjection` (`vllm/models/deepseek_v4_1/tilelang/wo.py`) computes:
   - `WO-A` (`self.a`, `_GroupedBF16`): grouped W8A16 GEMM (`fp8w_bf16_gemm_cast` at prefill, `_WO_A_SHARDS = 4`) whose epilogue rounds `a` to BF16 and applies `act_quant(a, 32, "ue8m0")` directly into `xq` (`[T, groups_r * 1024]` `float8_e4m3fn`) and `sf` (`[T, groups_r * 32]` `uint8` UE8M0 scales). As `wo.py` documents, `a` never reaches memory in BF16; `(xq, sf)` are the native input bytes of `WO-B`.
   - `WO-B` (`self.b`, `_Weight`): block-32 `MXFP8` GEMM (`mxfp8_gemm`, `block_K = 128`, `shards = 1`, `groups = 1`, `padded_rows = False`) projecting `(xq, sf)` with `(W_{B,r}, SF_{B,r})` (`[5120, groups_r * 1024]` E4M3 + `[160, groups_r * 8]` `uint32`) into `[T, 5120]` BF16.
2. In the baseline sequence-parallel forward (`DeepseekV4Attention._o_proj`), `sp_rows.project_reduce_scatter(..., slices=wo_slices(T))` calls `_wo_projection_weights(o[start:stop], out)` on slices of `T` rows (`2 * world` calls when `T >= 2048`: `6` launches on TP3, `8` launches on TP4) and exchanges `[world, L, 5120]` BF16 (`10,240` B/row) across ranks into `parts` (`[world, L, 5120]` BF16).
3. In `0731-sp-wo-local-exchange.patch`:
   - `TileLangWOProjection.declare_kernels()` declares all distinct peer `WO-B` prefill `mxfp8_gemm` widths (`3072` and `2048` on TP3; `2048` on TP4) before cache freeze, and `bind()` gathers the trimmed peer `WO-B` shards (`weight` and `tilelang_weight_sf`) once before memory profiling while reusing `self.b` in place for `peer == rank`.
   - On an SP prefill step (`sp_rows is not None` on encoder layers `0..19`), `TileLangWOProjection.project_sp(o, sp_rows, then=...)`:
     1. Runs `self.a` **once** on all `T` rows into `xq_send.view(padded, cols_r)[:T]` and `sf_send.view(padded, cols_r // 32)[:T]`, zeroing any pad rows `T:padded` via `columns.zero_`.
     2. Exchanges `xq_send` and `sf_send` across ranks on the side stream via `sp_rows.exchange_into` while running this rank's own `WO-B` shard (`self.sp_shards[rank] is self.b`) on the `real` local rows (`L` rows, or fewer on the tail rank when `T` is not a multiple of `world`) on the main stream into `parts[rank, :real]`.
     3. Runs each peer rank's `WO-B` shard (`self.sp_shards[peer]`) on the `real` local rows received from `peer` into `parts[peer, :real]`, zeroing any pad rows `real:L` via `columns.zero_`.
     4. Returns `parts` (`[world, L, 5120]` BF16) when `sp_rows.unreduced` is True (fused into `mhc.post_pre` via `RankOrderParts(parts)`), or `collectives.rank_order_sum(list(parts.unbind(0)))` otherwise.
4. **Why outputs are 100% bit-identical with zero precision change**:
   - No tensor is re-quantized, truncated, or re-ordered. The `(xq, sf)` bytes exchanged across ranks are the exact `float8_e4m3fn` + `uint8` `UE8M0` bytes emitted by `WO-A` on each source rank `s`.
   - For each source rank `s in 0..world-1` and each local row `i in 0..real-1` (global row `t = rank * L + i`), `self.sp_shards[s].tilelang_prefill` executes the exact same `mxfp8_gemm(5120, cols_s, block_K=128, shards=1, groups=1, padded_rows=False)` kernel on `(xq_s[t], W_{B,s}, sf_s[t], SF_{B,s})`, accumulating in FP32 across the same `K = cols_s` dimension in `block_K = 128` steps and rounding once to BF16 into `parts[s, i]`.
   - `parts[0, i] + parts[1, i] + ... + parts[world - 1, i]` is then summed in FP32 in rank order (`0, 1, ..., world - 1`) by `mhc.post_pre` or `collectives.rank_order_sum`.

## 2. Analytical Collective Byte and Memory Footprint Bounds

### 2.1 Per-Row and Per-Layer Collective Payload

| Topology | Rank(s) | Local `o_groups` | `WO-A` `cols_r` | Baseline BF16 (`B/row`) | Candidate `xq + sf` (`B/row`) | Payload Reduction |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| **TP3** (`tp3-triangle`) | Ranks 0, 1 | 3 | 3072 | `10,240` B | `3,072 + 96 = 3,168` B | **-69.0625%** |
| **TP3** (`tp3-triangle`) | Rank 2 (trimmed) | 2 | 2048 | `10,240` B | `2,048 + 64 = 2,112` B | **-79.3750%** |
| **TP3** (`tp3-triangle`) | Cluster average | 8 / 3 | 2730.7 | `10,240` B | `2,816` B | **-72.5000%** |
| **TP4** (`tp4-ring`) | Ranks 0, 1, 2, 3 | 2 | 2048 | `10,240` B | `2,048 + 64 = 2,112` B | **-79.3750%** |

### 2.2 Per-Chunk Wire Traffic (`T = 6144` on TP3, `T = 16384` on TP4)

| Topology & Chunk | Metric | Baseline (Full-Row BF16) | Candidate (`WO-A` `MXFP8`) | Delta |
| :--- | :--- | ---: | ---: | ---: |
| **TP3**, `T = 6144` (`L = 2048`) | Per-layer wire sent (Ranks 0, 1) | `40.000 MiB` (`41,943,040` B) | `12.375 MiB` (`12,976,128` B) | **-27.625 MiB (-69.06%)** |
| **TP3**, `T = 6144` (`L = 2048`) | Per-layer wire sent (Rank 2) | `40.000 MiB` (`41,943,040` B) | `8.250 MiB` (`8,650,752` B) | **-31.750 MiB (-79.38%)** |
| **TP3**, `T = 6144` (`L = 2048`) | Per-layer wire recv (Ranks 0, 1) | `40.000 MiB` (`41,943,040` B) | `10.3125 MiB` (`10,813,440` B) | **-29.6875 MiB (-74.22%)** |
| **TP3**, `T = 6144` (`L = 2048`) | Per-layer wire recv (Rank 2) | `40.000 MiB` (`41,943,040` B) | `12.375 MiB` (`12,976,128` B) | **-27.625 MiB (-69.06%)** |
| **TP3**, `T = 6144` (**20 encoder layers**) | Wire sent / recv (Ranks 0, 1) | `800.00 MiB` / `800.00 MiB` | `247.50 MiB` / `206.25 MiB` | **-552.50 MiB / -593.75 MiB** |
| **TP3**, `T = 6144` (**20 encoder layers**) | Wire sent / recv (Rank 2) | `800.00 MiB` / `800.00 MiB` | `165.00 MiB` / `247.50 MiB` | **-635.00 MiB / -552.50 MiB** |
| **TP4**, `T = 16384` (`L = 4096`) | Per-layer direct peer payload | `120.000 MiB` (`125,829,120` B) | `24.750 MiB` (`25,952,256` B) | **-95.250 MiB (-79.38%)** |
| **TP4**, `T = 16384` (`L = 4096`) | Per-layer `ring4_relay` wire | `160.000 MiB` (`167,772,160` B) | `33.000 MiB` (`34,603,008` B) | **-127.000 MiB (-79.38%)** |
| **TP4**, `T = 16384` (**20 encoder layers**) | `ring4_relay` wire per rank | `3,200.00 MiB` (`3.125 GiB`) | `660.00 MiB` (`0.645 GiB`) | **-2,540.00 MiB (-2.480 GiB, -79.38%)** |

### 2.3 Persistent Weight Memory and Per-Layer Weight Read Traffic

- **Persistent peer `WO-B` shards (`20` encoder layers `0..19` only; CED decoder layers `20..39` and draft layers keep only their local TP shard)**:
  - Each `WO-B` group (`5120 x 1024`) occupies `5,242,880` B weight + `5,120` B scale words = `5,248,000` B (`5.0049 MiB`).
  - **TP3 Ranks 0 & 1** (already own `3` groups; bind `3 + 2 = 5` peer groups = `26,240,000` B = `25.0244 MiB`/layer): **`524,800,000` B (`500.49 MiB` = `0.489 GiB`)** across 20 encoder layers.
  - **TP3 Rank 2** (already owns `2` groups; binds `3 + 3 = 6` peer groups = `31,488,000` B = `30.0293 MiB`/layer): **`629,760,000` B (`600.59 MiB` = `0.587 GiB`)** across 20 encoder layers.
  - **TP4 Ranks 0..3** (each owns `2` groups; binds `6` peer groups = `31,488,000` B = `30.0293 MiB`/layer): **`629,760,000` B (`600.59 MiB` = `0.587 GiB`)** across 20 encoder layers.
- **Elimination of repeated per-slice/per-owner weight streaming**:
  - At `T >= 2048` (`wo_slices(T) == 2`), baseline `sp_rows._project_exchange` calls `_wo_projection_weights` `2 * world` times (`6` calls on TP3, `8` calls on TP4), streaming `WO-A` + `WO-B` weights (`28.34 MiB` on TP3 Ranks 0/1, `18.89 MiB` on TP3 Rank 2 and TP4) `6` or `8` times per layer (`170.0 MiB` on TP3 Ranks 0/1, `151.1 MiB` on TP4).
  - Candidate `project_sp` launches `WO-A` **once** (`12.625 MiB` on 3-group ranks, `8.417 MiB` on 2-group ranks) and each rank's `WO-B` shard **once** (`40.039 MiB` total across all 8 groups), reducing per-layer weight DRAM reads by **69.0%** on TP3 Ranks 0/1 (`170.0 MiB -> 52.66 MiB`) and **67.9%** on TP4 (`151.1 MiB -> 48.46 MiB`).

## 3. Phase 1 CPU & Offline `sm_121a` Compiler Verification (`sp8`)

### 3.1 Isolated CPU Unit Tests (`cpu.sh`)

Command:
```bash
experiments/2026-11-06-wo-local-exchange/cpu.sh /home/chris/spark-lab/wo-local-exchange/vllm
```

Output:
```text
host_mem_available_kib 120525500
.............................................ss......................... [ 57%]
.....................................................                    [100%]
123 passed, 2 skipped, 14 warnings in 12.67s
process_maxrss_kib 1322628
host_mem_available_kib 119901832
```

### 3.2 Offline `sm_121a` Device Compilation (`compile.py`)

Command:
```bash
flock -w 30 /tmp/spark-cpu-analysis.lock docker run --rm --runtime runc \
  --cpus=1 --memory=4g --memory-swap=4g --pids-limit=128 --network=none \
  -e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void \
  -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MAX_JOBS=1 \
  -e PYTHONPATH=/src -v /home/chris/spark-lab/wo-local-exchange/vllm:/src:ro \
  -v /home/chris/spark-lab/wo-local-exchange/repo/experiments/2026-11-06-wo-local-exchange:/check:ro \
  --entrypoint python3 sha256:3795255f485ea5c1a9db54c742921dd9b2917daad0799361e5c6f56a70119713 /check/compile.py
```

Output:
```text
device_compile_passed {'kernel': 'wo_a_prefill_g3', 'source_chars': 24299}
device_compile_passed {'kernel': 'wo_a_prefill_g2', 'source_chars': 24293}
device_compile_passed {'kernel': 'wo_b_prefill_g3_3072', 'source_chars': 8231}
device_compile_passed {'kernel': 'wo_b_prefill_g2_2048', 'source_chars': 8231}
device_compile_passed {'kernel': 'splitk_reduce_cast_g3', 'source_chars': 4892}
device_compile_passed {'kernel': 'splitk_reduce_cast_g2', 'source_chars': 4891}
device_compile_passed {'kernel': 'zero_packs', 'source_chars': 4028}
device_compile_passed {'kernel': 'rank_order_sum_tp3', 'source_chars': 2042}
device_compile_passed {'kernel': 'rank_order_sum_tp4', 'source_chars': 2375}
process_maxrss_kib 1469164
```

## 4. Phase 2 GPU Screening Plan (`screen_wo.py`)

When a GPU window is scheduled by the coordinator:
1. Run `screen_wo.py --boot boot1` and `screen_wo.py --boot boot2` inside the candidate container to record 2-boot native `sm_121a` bit-exactness gates and warm/cold-L2 per-rank `WO` compute timings on TP3 (`T = 6144`) and TP4 (`T = 16384`).
2. Run a 2-boot end-to-end `bin/spark bench` prefill screen on TP3 (`tp3.json`) / TP4 (`tp4.json`) against Candidate `1107`'s recorded reference.
