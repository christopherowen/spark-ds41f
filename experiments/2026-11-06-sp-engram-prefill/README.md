# Sequence-parallel Engram prefill (`2026-11-06-sp-engram-prefill`)

## Summary & Goal (`TODO.md` line 93)

Under prefill sequence parallelism (`sp_rows is not None`), `Engram.forward` in
`vllm/models/deepseek_v4_1/common/engram.py` previously gathered and projected
every token row `T` across all ranks before exchanging the projected `wkv` column
blocks (`hidden_size = 5120`, `hc_mult = 4`, `logical_cols = 5120 * 5 = 25600`,
`engram_k = 6144`):

1. **Collective 1 (`interleaved_rows`)**: After each rank gathers its
   `max_cols = ceil(24 / N)` disk hash columns (`[T, max_cols * 256]` BF16) into
   `staged_rows[:T]`, `interleaved_rows` ran a full `tensor_model_parallel_all_gather`
   across all `T` rows (`144.00 MiB` wire/rank on TP4 at `T = 16384`; `72.00 MiB`
   wire/rank at `T = 8192`, producing `[8192, 6144]` BF16 = `96.00 MiB`; `48.00 MiB`
   wire/rank on TP3 at `T = 6144`) followed by `concat_rank_shards` over all `T`
   rows (`192.00 MiB` read + `192.00 MiB` write = `384.00 MiB` DRAM on TP4 at `T = 16384`).
2. **Full-`T` activation quantization (`pdl.per_token_cast`)**: Every rank
   quantized all `T` interleaved BF16 rows (`[T, 6144]`) to MXFP8 (`192.00 MiB`
   BF16 read + `99.00 MiB` MXFP8 write = `291.00 MiB` DRAM/layer on TP4 at `T = 16384`),
   even though each rank only gates its `real_rows <= L = ceil(T / N)`
   sequence-parallel tokens.
3. **Collective 2 (`sp_rows.exchange_rows(kv)`)**: After projecting `[T, block_cols]`
   through `wkv` (`block_cols = 6400` on TP4 -> `[8192, 25600]` BF16 = `400.00 MiB`
   at 8K tokens and `800.00 MiB` at 16K tokens; `block_cols = 8544` on TP3), every
   rank ran a second collective (`all_to_all_rows` on `[T, block_cols]`:
   `150.00 MiB` direct / `200.00 MiB` `ring4_relay` wire/rank per layer on TP4 at
   `T = 16384`; `66.75 MiB` wire/rank per layer on TP3 at `T = 6144`) and discarded
   `(N - 1) / N` of the rows it had interleaved and quantized.

## Base & Intended Causal Variable

- **Deployment base commit**: `0b3e98934f0aa7c7c251b79dae3ac148f869304e` (Candidate `1107`)
- **vLLM base**: `19f2c20ed4d64ae22bd009ceacadd85e2a81e354` with Candidate `1107`
  series `0001..0730` (patch head `c47b54ae471a24b830e3515212b8c0bf5ee14604`,
  tree `13e4781d19bc4a18b0d3d4daad7921c465e3355e`)
- **Candidate vLLM patch**: `experiments/2026-11-06-sp-engram-prefill/vllm/0731-sp-engram-prefill.patch`
  (patch head `1f457d1f3e5b835781e18437da8cb527ffd97cd5`,
  tree `0c737c5bae35f09a6b9f393165dd99b547beced4`,
  `patchset_sha256` `201f4fac4130cb57c0b82a8bb596b5744ff0ca59b736a2b2451d870f0b2b7414`)
- **Intended variable**: Sequence-parallel Engram prefill row-block column exchange
  (`interleaved_sp_rows`) and token-split projection (`_PaddedColumnParallelLinear.local_sp_product`
  and sliced `SPRows.project_exchange_rows`).

## Technical Design

1. **Row-block hash-column exchange & local interleave (`interleaved_sp_rows`)**:
   - Under `sp_rows` when `embed_tokens.columns is not None`, each rank exchanges
     `staged_rows[:num_tokens]` (`[T, max_cols * 256]` BF16) via
     `sp_rows.exchange_rows(...)` (`all_to_all_rows`) into
     `blocks` (`[N, L, max_cols * 256]` BF16).
   - Each rank receives only its `real = sp_rows.real_rows <= L` local rows from
     all `N` ranks and interleaves only those `real` rows into `[real, 6144]` BF16
     via `concat_rank_shards` (4x fewer rows on TP4, 3x fewer on TP3).
2. **Local token-split `wkv` shard projection (`_PaddedColumnParallelLinear.local_sp_product`)**:
   - DeepSeek-V4.1-Flash has only two Engram layers (`[1, 14]`). During
     `Engram.process_weights_after_loading`, when sequence parallelism is active on
     `_PaddedColumnParallelLinear`, `gather_sp_shards()` all-gathers the peer ranks'
     `(tilelang_weight, tilelang_weight_sf)` column blocks into `wkv.sp_shards`,
     aliasing `sp_shards[tp_rank]` to this rank's own shard in place (0 extra bytes
     for the local shard).
   - On an SP prefill step, `local_sp_product(rows, sp_rows.local_rows)` quantizes
     the `real` (`<= L`) local BF16 rows **once** via `pdl.per_token_cast` and runs
     the exact same `_block32_gemm_weights` kernel across the `N` shards in
     `sp_shards` directly into `kv` (`[N, L, block_cols]` BF16).
   - **Zero collective runs after `wkv`**, and `engram_post_gate_blocked` /
     `engram_gate_blocked` consumes `kv` (`[N, L, block_cols]`) unchanged.
3. **Sliced `SPRows.project_exchange_rows` fallback (`ENGRAM_SLICE_TOKENS = 2048`)**:
   - When `embed_tokens.columns is None` (row-sharded resident/RAM tables) or
     `sp_shards` is not populated, `Engram.forward` uses `sp_rows.project_exchange_rows`
     to project `rows[start:stop]` directly into `output[rank, :real]` for the
     rank's own block (`copy_self=False` on `PyNcclCommunicator.all_to_all_rows`)
     and into a reusable slice scratch buffer for peer blocks (`<= 2048` rows),
     overlapping peer PyNCCL exchanges with local projection, `RankOrderParts.sum()`,
     and pad-row `_mhc_post` / `copy_columns`.
   - When `wkv` is replicated (`ReplicatedLinear`), each rank slices `rows` to its
     `real_rows` before projecting.

## Precision & Bit-Identity Verification

- **No precision reduction anywhere**: All tensor dtypes (`torch.bfloat16`,
  `torch.float32`, `torch.float8_e4m3fn`, `torch.uint8` UE8M0 scale words),
  quantization block sizes (`SF_BLOCK = 32`), accumulation types (`T.float32`),
  and rounding modes are 100% unchanged.
- **Bit-identical invariant (`max_abs_diff = 0.0`, `torch.equal` on `.view(torch.int16)`)**:
  - `concat_rank_shards` operates independently per token row `t`; exchanging row
    blocks `[start : start + real]` before `concat_rank_shards` produces the exact
    same 16-bit words as `concat_rank_shards` on all `T` rows followed by slicing
    `[start : start + real]`.
  - `pdl.per_token_cast` quantizes each 32-element row block independently, and
    `mxfp8_gemm` tiles along `M` (`block_M = 64`) with zero cross-row state.
    For every rank `r`, token `t = r * L + i` is at local row index `i` (`0 <= i < real`),
    so `local_sp_product(rows_local, L)[s, :real]` executes the exact same MXFP8
    cast and MMA instructions on the exact same `(x_fp8[i], x_scales[i], w_s, s_s)`
    operands as rank `s`'s `local_product(rows_full)[r * L : r * L + real]`.

## Analytical Traffic, Compute & Memory Comparison (2 Engram Layers `[1, 14]`)

### TP4 (`tp4-ring`, `T = 16384`, `L = 4096`, `block_cols = 6400`)

| Metric (across both Engram layers `[1, 14]` per 16K chunk) | Candidate 1107 Baseline | Candidate (`0731`) | Delta |
| --- | ---: | ---: | ---: |
| Collectives per chunk (2 layers) | 4 (`2x all_gather` + `2x all_to_all_rows`) | **2 (`2x all_to_all_rows` on `staged_rows`)** | **-2 collectives (-50%)** |
| Collective wire bytes / rank (direct) | `588.00 MiB` (`288 + 300 MiB`) | **`72.00 MiB`** (`2 x 36 MiB`) | **`-516.00 MiB` (`-87.8%`)** |
| Collective wire bytes / rank (`ring4_relay`) | `688.00 MiB` (`288 + 400 MiB`) | **`96.00 MiB`** (`2 x 48 MiB`) | **`-592.00 MiB` (`-86.0%`)** |
| `concat_rank_shards` DRAM read+write / rank | `768.00 MiB` (`32,768` rows) | **`192.00 MiB`** (`8,192` rows) | **`-576.00 MiB` (`-75.0%`)** |
| `per_token_cast` MXFP8 rows quantized / rank | `32,768` rows (`582.00 MiB` DRAM) | **`8,192` rows** (`145.50 MiB` DRAM) | **`-24,576` rows (`-75.0%`, `-436.50 MiB`)** |
| `wkv` GEMM FLOPs / rank (per layer) | `1288.5 GFLOPs` (`16384 x 6400 x 6144`) | **`1288.5 GFLOPs`** (`4 x 4096 x 6400 x 6144`) | `0.0%` (identical FLOPs; `4x` smaller activation footprint fits in L2) |
| Persistent weight memory added (2 layers total) | `0.00 MiB` | **`+225.22 MiB`** (`+0.220 GiB`) | `+225.22 MiB` (`3 x 37.54 MiB` x 2 layers; own shard aliased) |
| Peak transient activation buffers in `Engram.forward` | `384.00 MiB` (`gathered` + `rows`) | **`272.00 MiB`** (`blocks` + `rows_local` + `kv`) | **`-112.00 MiB`** |

*(At `T = 8192` on TP4, collective wire traffic across both Engram layers drops from `294.00 MiB` direct / `344.00 MiB` `ring4_relay` per rank — `1.15 GiB`–`1.34 GiB` cluster-wide — to `36.00 MiB` direct / `48.00 MiB` `ring4_relay` per rank.)*

### TP3 (`tp3-triangle`, `T = 6144`, `L = 2048`, `block_cols = 8544`)

| Metric (across both Engram layers `[1, 14]` per 6K chunk) | Candidate 1107 Baseline | Candidate (`0731`) | Delta |
| --- | ---: | ---: | ---: |
| Collectives per chunk (2 layers) | 4 (`2x all_gather` + `2x all_to_all_rows`) | **2 (`2x all_to_all_rows` on `staged_rows`)** | **-2 collectives (-50%)** |
| Collective wire bytes / rank | `229.50 MiB` (`96.0 + 133.5 MiB`) | **`32.00 MiB`** (`2 x 16 MiB`) | **`-197.50 MiB` (`-86.1%`)** |
| `concat_rank_shards` DRAM read+write / rank | `288.00 MiB` (`12,288` rows) | **`96.00 MiB`** (`4,096` rows) | **`-192.00 MiB` (`-66.7%`)** |
| `per_token_cast` MXFP8 rows quantized / rank | `12,288` rows (`218.25 MiB` DRAM) | **`4,096` rows** (`72.75 MiB` DRAM) | **`-8,192` rows (`-66.7%`, `-145.50 MiB`)** |
| Persistent weight memory added (2 layers total) | `0.00 MiB` | **`+200.45 MiB`** (`+0.196 GiB`) | `+200.45 MiB` (`2 x 50.11 MiB` x 2 layers; own shard aliased) |

## CPU Verification Runs (on `sp8`)

Executed in CPU-only container (`spark-ds41f:1030` with `CUDA_VISIBLE_DEVICES=""`
and `NVIDIA_VISIBLE_DEVICES=void`):

```bash
./experiments/2026-11-06-sp-engram-prefill/cpu.sh /home/chris/spark-lab/sp-engram-prefill/vllm
```

Result: **`208 passed, 15 skipped in 50.35s`** (`0 failed`).

## Phase 2: 2-Boot GPU Qualification & Microbenchmark Screen (`sp8`, `NVIDIA GB10` `sm_121a`)

Executed two fresh container boots (`boot1` and `boot2`) on `sp8` (`NVIDIA GB10`,
compute capability `12.1`, PyTorch `2.13.0+cu130`, image `spark-ds41f:1030`) under
strict `sp8:~/spark-hold.json` (`pool: "sp8-single"`) coordination.

Receipts:
- `experiments/2026-11-06-sp-engram-prefill/receipts/screen-engram-boot1.jsonl`
- `experiments/2026-11-06-sp-engram-prefill/receipts/screen-engram-boot2.jsonl`
- `experiments/2026-11-06-sp-engram-prefill/receipts/gpu-pytest-boot1.txt`
- `experiments/2026-11-06-sp-engram-prefill/receipts/gpu-pytest-boot2.txt`

### 1. GPU Unit Test Suite (`boot1` & `boot2`)

```bash
pytest -p tblib_stub -p no:cacheprovider -q \
  tests/models/test_deepseek_v4_1_sp_prefill.py \
  tests/models/test_deepseek_v4_1_engram.py \
  tests/models/test_deepseek_v41_sp_project_reduce_scatter.py
```

- **`boot1`**: **`428 passed, 14 warnings in 131.31s`** (`0 failed`)
- **`boot2`**: **`428 passed, 14 warnings in 124.13s`** (`0 failed`)

### 2. GPU Bit-Exactness Gate (`screen_engram.py`, `boot1` & `boot2`)

Across both fresh boots, all **28/28** rank-and-shape configurations per boot
(**56/56** total across `boot1` and `boot2`) on real `sm_121a` TileLang kernels
(`concat_rank_shards`, `pdl.per_token_cast`, `mxfp8_gemm`, `RankOrderParts.sum()`,
and `engram_post_gate_blocked`) verified **100% bit-identical (`torch.int16` view
equality)**:

- **TP3 (`world = 3`, ranks `0, 1, 2`)**: `T = 206` (unaligned + signed-zero `-0.0` +
  `NaN`/`+Inf`/`-Inf` + `IMAGE_SENTINEL`), `T = 2048`, `T = 4096`, `T = 6144`
- **TP4 (`world = 4`, ranks `0, 1, 2, 3`)**: `T = 205` (unaligned + signed-zero `-0.0` +
  `NaN`/`+Inf`/`-Inf` + `IMAGE_SENTINEL`), `T = 4096`, `T = 8192`, `T = 16384`
- Every record passed:
  - `interleaved_rows_int16_equal = True`
  - `wkv_local_sp_product_int16_equal = True`
  - `engram_forward_int16_equal = True`
  - `max_abs_diff = 0.0`

### 3. 2-Boot Per-Layer GPU Kernel Timings (`NVIDIA GB10`, CUDAGraph Replay, 15 Repeats/Boot)

Note: The timings below measure **local GPU kernel execution per Engram layer**
(`interleave_concat` = `concat_rank_shards`, `wkv_projection` = `per_token_cast` +
`mxfp8_gemm`, `engram_local_total` = `concat_rank_shards` + `wkv` +
`RankOrderParts.sum()` + `engram_post_gate_blocked`). In multi-node TP3/TP4
execution, Candidate (`0731`) **also eliminates `197.5 MiB` (TP3 6K) to `592.0 MiB`
(TP4 16K `ring4_relay`) of inter-node collective wire traffic across the two
Engram layers**.

| World | Tokens (`T`) | Local Rows (`L`) | Stage | Cache Mode | `boot1` Base (ms) | `boot2` Base (ms) | 2-Boot Base Mean (ms) | `boot1` Cand (ms) | `boot2` Cand (ms) | 2-Boot Cand Mean (ms) | Delta (ms) | Delta (%) | Speedup |
| :---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TP3 | 2048 | 683 | `interleave_concat` | `warm` | 0.2270 | 0.2252 | 0.2261 | 0.0241 | 0.0257 | **0.0249** | **-0.2012** | **-88.99%** | **9.08x** |
| TP3 | 2048 | 683 | `wkv_projection` | `warm` | 1.3636 | 1.3482 | 1.3559 | 1.4596 | 1.4373 | **1.4485** | +0.0925 | +6.83% | 0.94x |
| TP3 | 2048 | 683 | `wkv_projection` | `128MiB_wash` | 1.3954 | 1.3790 | 1.3872 | 1.4835 | 1.4650 | **1.4743** | +0.0871 | +6.28% | 0.94x |
| TP3 | 2048 | 683 | `engram_local_total` | `warm` | 2.0843 | 2.0640 | 2.0741 | 2.0499 | 2.0026 | **2.0263** | **-0.0479** | **-2.31%** | **1.02x** |
| TP3 | 2048 | 683 | `engram_local_total` | `128MiB_wash` | 2.0919 | 2.0749 | 2.0834 | 2.0610 | 2.0241 | **2.0425** | **-0.0409** | **-1.96%** | **1.02x** |
| TP3 | 4096 | 1366 | `interleave_concat` | `warm` | 0.4459 | 0.4431 | 0.4445 | 0.1406 | 0.1342 | **0.1374** | **-0.3071** | **-69.09%** | **3.24x** |
| TP3 | 4096 | 1366 | `wkv_projection` | `warm` | 2.7225 | 2.6969 | 2.7097 | 3.0087 | 2.7628 | **2.8857** | +0.1761 | +6.50% | 0.94x |
| TP3 | 4096 | 1366 | `wkv_projection` | `128MiB_wash` | 2.8027 | 2.7235 | 2.7631 | 2.8884 | 2.8014 | **2.8449** | +0.0818 | +2.96% | 0.97x |
| TP3 | 4096 | 1366 | `engram_local_total` | `warm` | 4.2122 | 4.1812 | 4.1967 | 4.0433 | 3.9164 | **3.9798** | **-0.2168** | **-5.17%** | **1.05x** |
| TP3 | 4096 | 1366 | `engram_local_total` | `128MiB_wash` | 4.2400 | 4.1806 | 4.2103 | 3.9902 | 3.9505 | **3.9703** | **-0.2400** | **-5.70%** | **1.06x** |
| TP3 | 6144 | 2048 | `interleave_concat` | `warm` | 0.6675 | 0.6622 | 0.6648 | 0.2261 | 0.2230 | **0.2245** | **-0.4403** | **-66.23%** | **2.96x** |
| TP3 | 6144 | 2048 | `wkv_projection` | `warm` | 4.0734 | 4.2075 | 4.1404 | 4.1521 | 3.7701 | **3.9611** | **-0.1793** | **-4.33%** | **1.05x** |
| TP3 | 6144 | 2048 | `wkv_projection` | `128MiB_wash` | 4.2554 | 4.0577 | 4.1565 | 4.0148 | 3.8039 | **3.9093** | **-0.2472** | **-5.95%** | **1.06x** |
| TP3 | 6144 | 2048 | `engram_local_total` | `warm` | 6.3592 | 6.5466 | 6.4529 | 5.7820 | 5.5825 | **5.6822** | **-0.7706** | **-11.94%** | **1.14x** |
| TP3 | 6144 | 2048 | `engram_local_total` | `128MiB_wash` | 6.6063 | 6.3296 | 6.4680 | 5.6367 | 5.5695 | **5.6031** | **-0.8649** | **-13.37%** | **1.15x** |
| TP4 | 4096 | 1024 | `interleave_concat` | `warm` | 0.4565 | 0.4520 | 0.4543 | 0.0908 | 0.0756 | **0.0832** | **-0.3711** | **-81.68%** | **5.46x** |
| TP4 | 4096 | 1024 | `wkv_projection` | `warm` | 2.1505 | 2.1216 | 2.1361 | 2.0012 | 1.9752 | **1.9882** | **-0.1479** | **-6.92%** | **1.07x** |
| TP4 | 4096 | 1024 | `wkv_projection` | `128MiB_wash` | 2.1839 | 2.1491 | 2.1665 | 2.0310 | 2.0077 | **2.0194** | **-0.1471** | **-6.79%** | **1.07x** |
| TP4 | 4096 | 1024 | `engram_local_total` | `warm` | 3.4259 | 3.3859 | 3.4059 | 2.9274 | 2.8961 | **2.9118** | **-0.4941** | **-14.51%** | **1.17x** |
| TP4 | 4096 | 1024 | `engram_local_total` | `128MiB_wash` | 3.4622 | 3.5704 | 3.5163 | 2.9703 | 2.9389 | **2.9546** | **-0.5617** | **-15.97%** | **1.19x** |
| TP4 | 8192 | 2048 | `interleave_concat` | `warm` | 0.9007 | 0.9177 | 0.9092 | 0.2290 | 0.2297 | **0.2293** | **-0.6799** | **-74.77%** | **3.96x** |
| TP4 | 8192 | 2048 | `wkv_projection` | `warm` | 4.2515 | 4.4306 | 4.3411 | 4.0959 | 3.7804 | **3.9382** | **-0.4029** | **-9.28%** | **1.10x** |
| TP4 | 8192 | 2048 | `wkv_projection` | `128MiB_wash` | 4.3465 | 4.2728 | 4.3096 | 4.6265 | 3.9227 | **4.2746** | **-0.0351** | **-0.81%** | **1.01x** |
| TP4 | 8192 | 2048 | `engram_local_total` | `warm` | 7.6204 | 6.7484 | 7.1844 | 5.8607 | 5.6593 | **5.7600** | **-1.4244** | **-19.83%** | **1.25x** |
| TP4 | 8192 | 2048 | `engram_local_total` | `128MiB_wash` | 6.9151 | 6.8116 | 6.8634 | 5.7218 | 5.7710 | **5.7464** | **-1.1170** | **-16.27%** | **1.19x** |
| TP4 | 16384 | 4096 | `interleave_concat` | `warm` | 1.7253 | 1.7524 | 1.7389 | 0.4499 | 0.4481 | **0.4490** | **-1.2898** | **-74.18%** | **3.87x** |
| TP4 | 16384 | 4096 | `wkv_projection` | `warm` | 8.9095 | 8.7400 | 8.8247 | 8.1366 | 7.8477 | **7.9921** | **-0.8326** | **-9.43%** | **1.10x** |
| TP4 | 16384 | 4096 | `wkv_projection` | `128MiB_wash` | 8.8604 | 8.5808 | 8.7206 | 7.9999 | 7.8475 | **7.9237** | **-0.7969** | **-9.14%** | **1.10x** |
| TP4 | 16384 | 4096 | `engram_local_total` | `warm` | 14.1973 | 13.9571 | 14.0772 | 11.7929 | 11.4361 | **11.6145** | **-2.4627** | **-17.49%** | **1.21x** |
| TP4 | 16384 | 4096 | `engram_local_total` | `128MiB_wash` | 14.2467 | 14.0985 | 14.1726 | 12.1004 | 12.0381 | **12.0693** | **-2.1033** | **-14.84%** | **1.17x** |

## Conclusion & Verdict

- **Bit-exactness**: **PASS** (`56/56` 2-boot GPU bit-exactness gates with
  `max_abs_diff = 0.0` and `torch.int16` view equality; `428/428` GPU unit tests
  passing on both boots; `208/208` CPU unit tests passing).
- **Local GPU compute per prefill chunk (2 Engram layers)**:
  - On **TP4 (`T = 16384`)**: saves **`4.21–4.93 ms` of local GPU kernel time**
    per 16K chunk across the two Engram layers (`-14.84%` to `-17.49%`), driven by
    a **`3.87x` speedup on `concat_rank_shards`** (`1.7389 ms -> 0.4490 ms`/layer),
    a **`1.10x` speedup on `wkv_projection`** (`8.8247 ms -> 7.9921 ms`/layer from
    4x fewer `per_token_cast` rows and L2-resident `[4096, 6144]` MXFP8 activations),
    and fused `RankOrderParts.sum()` on local rows.
  - On **TP4 (`T = 8192`)**: saves **`2.23–2.85 ms` of local GPU kernel time**
    per 8K chunk across the two Engram layers (`-16.27%` to `-19.83%`, `1.19x–1.25x`).
  - On **TP3 (`T = 6144`)**: saves **`1.54–1.73 ms` of local GPU kernel time**
    per 6K chunk across the two Engram layers (`-11.94%` to `-13.37%`, `1.14x–1.15x`).
- **Collective traffic per prefill chunk (2 Engram layers)**:
  - Eliminates **`516.00 MiB` (direct, `-87.8%`) / `592.00 MiB` (`ring4_relay`, `-86.0%`)
    per rank** at TP4 `T = 16384` (`258.00 MiB` / `296.00 MiB` per rank at `T = 8192`)
    and **`197.50 MiB` (`-86.1%`) per rank** at TP3 `T = 6144`, while removing the
    post-`wkv` collective completely.
- **Ready for multi-node TP3/TP4 prefill stack integration.**
