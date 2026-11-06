# Sequence-parallel Engram prefill (`2026-11-06-sp-engram-prefill`)

## Summary & Goal (`TODO.md` line 93)

Under prefill sequence parallelism (`sp_rows is not None`), `Engram.forward` in
`vllm/models/deepseek_v4_1/common/engram.py` previously gathered and projected
every token row `T` across all ranks before exchanging the projected `wkv` column
blocks:

1. **Collective 1 (`interleaved_rows`)**: After each rank gathers its
   `max_cols = ceil(24 / N)` disk hash columns (`[T, max_cols * 256]` BF16) into
   `staged_rows[:T]`, `interleaved_rows` ran a full `tensor_model_parallel_all_gather`
   across all `T` rows (`144.00 MiB` wire/rank on TP4 at `T = 16384`; `48.00 MiB`
   wire/rank on TP3 at `T = 6144`) followed by `concat_rank_shards` over all `T`
   rows (`192.00 MiB` read + `192.00 MiB` write = `384.00 MiB` DRAM on TP4).
2. **Full-`T` activation quantization (`pdl.per_token_cast`)**: Every rank
   quantized all `T` interleaved BF16 rows (`[T, 6144]`) to MXFP8 (`192.00 MiB`
   BF16 read + `99.00 MiB` MXFP8 write = `291.00 MiB` DRAM/layer on TP4), even
   though each rank only gates its `real_rows <= L = ceil(T / N)` sequence-parallel
   tokens.
3. **Collective 2 (`sp_rows.exchange_rows(kv)`)**: After projecting `[T, C]`
   through `wkv`, every rank ran a second collective (`all_to_all_rows` on
   `[T, C]`: `75.00 MiB` direct / `100.00 MiB` `ring4_relay` wire/rank on TP4;
   `33.50 MiB` wire/rank on TP3) and discarded `(N - 1) / N` of the rows it had
   interleaved and quantized.

## Base & Intended Causal Variable

- **Deployment base commit**: `0b3e98934f0aa7c7c251b79dae3ac148f869304e` (Candidate `1107`)
- **vLLM base**: `19f2c20ed4d64ae22bd009ceacadd85e2a81e354` with Candidate `1107`
  series `0001..0730` (patch head `c47b54ae471a24b830e3515212b8c0bf5ee14604`,
  tree `13e4781d19bc4a18b0d3d4daad7921c465e3355e`)
- **Candidate vLLM patch**: `experiments/2026-11-06-sp-engram-prefill/vllm/0731-sp-engram-prefill.patch`
  (patch head `38beeaab9b76d0cfe0a479146b1f95db9d12c191`,
  tree `b1d44a3875769b312879e4d2e9353a6d92b5e198`,
  `patchset_sha256` `7047f15b3e55c6ec25b6439cd99f382e9f7488ffd6c71433891d482a3a636006`)
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

- **No precision reduction anywhere**: All tensor dtypes (`torch.bfloat16` activations
  and outputs, `torch.float8_e4m3fn` + packed `UE8M0` `uint32` block scales in
  `wkv`, `torch.float32` accumulation and `mHC` mixes) and kernel accumulation
  orders are unchanged.
- `interleaved_sp_rows(staged_rows[:T], sp_rows)` is a pure byte permutation of the
  exact same 256-element BF16 hash columns as `interleaved_rows(staged_rows[:T])[start : start + real]`.
- `pdl.per_token_cast` quantizes each token row independently, and `_mxfp8_gemm`
  accumulates each row in K order independently of `M`. Consequently, `kv[:, :real]`
  and the gated output `out[:real]` are **100% bit-identical (`torch.equal` on `.view(torch.int16)`)**
  to the baseline across TP3 and TP4.
- Decode (`sp_rows is None`) is completely untouched.

## Analytical Traffic, Compute & Memory Comparison (2 Engram Layers `[1, 14]`)

### TP4 (`tp4-ring`, `T = 16384`, `L = 4096`, `block_cols = 3200`)

| Metric (across both Engram layers `[1, 14]` per 16K chunk) | Candidate 1107 Baseline | Candidate (`0731`) | Delta |
| --- | ---: | ---: | ---: |
| Collectives per chunk (2 layers) | 4 (`2x all_gather` + `2x all_to_all_rows`) | **2 (`2x all_to_all_rows` on `staged_rows`)** | **-2 collectives (-50%)** |
| Collective wire bytes / rank (direct) | `438.00 MiB` (`288 + 150 MiB`) | **`72.00 MiB`** (`2 x 36 MiB`) | **`-366.00 MiB` (`-83.6%`)** |
| Collective wire bytes / rank (`ring4_relay`) | `488.00 MiB` (`288 + 200 MiB`) | **`96.00 MiB`** (`2 x 48 MiB`) | **`-392.00 MiB` (`-80.3%`)** |
| `concat_rank_shards` DRAM read+write / rank | `768.00 MiB` (`32,768` rows) | **`192.00 MiB`** (`8,192` rows) | **`-576.00 MiB` (`-75.0%`)** |
| `per_token_cast` MXFP8 rows quantized / rank | `32,768` rows (`582.00 MiB` DRAM) | **`8,192` rows** (`145.50 MiB` DRAM) | **`-24,576` rows (`-75.0%`, `-436.50 MiB`)** |
| `wkv` GEMM FLOPs / rank (per layer) | `644.2 GFLOPs` (`16384 x 3200 x 6144`) | **`644.2 GFLOPs`** (`4 x 4096 x 3200 x 6144`) | `0.0%` (identical FLOPs; `4x` smaller activation footprint fits in 24 MiB L2) |
| Persistent weight memory added (2 layers total) | `0.00 MiB` | **`+112.61 MiB`** (`+0.110 GiB`) | `+112.61 MiB` (`3 x 18.77 MiB` x 2 layers; own shard aliased) |
| Peak transient activation buffers in `Engram.forward` | `384.00 MiB` (`gathered` + `rows`) | **`172.75 MiB`** (`blocks` + `rows_local` + `kv`) | **`-211.25 MiB` (net peak memory `-98.64 MiB`)** |

*(At `T = 8192` on TP4, collective wire traffic across both Engram layers drops from `219.00 MiB` direct / `244.00 MiB` `ring4_relay` per rank — `876 MiB`–`976 MiB` cluster-wide — to `36.00 MiB` direct / `48.00 MiB` `ring4_relay` per rank.)*

### TP3 (`tp3-triangle`, `T = 6144`, `L = 2048`, `block_cols = 4288`)

| Metric (across both Engram layers `[1, 14]` per 6K chunk) | Candidate 1107 Baseline | Candidate (`0731`) | Delta |
| --- | ---: | ---: | ---: |
| Collectives per chunk (2 layers) | 4 (`2x all_gather` + `2x all_to_all_rows`) | **2 (`2x all_to_all_rows` on `staged_rows`)** | **-2 collectives (-50%)** |
| Collective wire bytes / rank | `163.00 MiB` (`96 + 67 MiB`) | **`32.00 MiB`** (`2 x 16 MiB`) | **`-131.00 MiB` (`-80.4%`)** |
| `concat_rank_shards` DRAM read+write / rank | `288.00 MiB` (`12,288` rows) | **`96.00 MiB`** (`4,096` rows) | **`-192.00 MiB` (`-66.7%`)** |
| `per_token_cast` MXFP8 rows quantized / rank | `12,288` rows (`218.25 MiB` DRAM) | **`4,096` rows** (`72.75 MiB` DRAM) | **`-8,192` rows (`-66.7%`, `-145.50 MiB`)** |
| Persistent weight memory added (2 layers total) | `0.00 MiB` | **`+100.60 MiB`** (`+0.098 GiB`) | `+100.60 MiB` (`2 x 25.15 MiB` x 2 layers; own shard aliased) |

## CPU Verification Runs (on `sp8`)

Executed in CPU-only container (`spark-ds41f:1030` with `CUDA_VISIBLE_DEVICES=""`
and `NVIDIA_VISIBLE_DEVICES=void`):

```bash
./experiments/2026-11-06-sp-engram-prefill/cpu.sh /home/chris/spark-lab/sp-engram-prefill/vllm
```

- `tests/models/test_deepseek_v4_1_sp_prefill.py`
- `tests/models/test_deepseek_v41_sp_project_reduce_scatter.py`
- `tests/models/test_deepseek_v4_1_engram.py`

Result: **`208 passed, 15 skipped in 46.8s`** (`0 failed`), covering:
- `test_interleaved_sp_rows_matches_interleaved_rows_slice`: bit-identical equality
  against `interleaved_rows(local)[start : start + real]` across TP3, TP4, and TP5
  (`24 % tp != 0` padded hash columns, padded tail token rows, and empty last rank).
- `test_engram_forward_sp_disk_columns_local_sp_product_bit_exact`: bit-identical
  `Engram.forward` across TP3 and TP4 with and without `previous_output = RankOrderParts`,
  `IMAGE_SENTINEL` tokens, padded tail rows, and empty last rank; verifies local
  shard aliasing (`data_ptr()` equality), single collective per layer (`staged_rows`
  exchange only), and activation cast row count (`real_rows`).
- `test_tilelang_linear_apply_shards_single_cast_and_bit_exact`: single `per_token_cast`
  call and bit-identical output across TP3 and TP4 for `TileLangFP8LinearMethod.apply_shards`.
- `test_project_exchange_rows_matches_full_projection_exchange`: bit-identical
  sliced `project_exchange_rows` across TP3 and TP4 (`copy_self=False` for PyNCCL
  and `b12x_ar_comm` fallback).
- `test_engram_forward_sp_sliced_blocked_and_replicated_bit_exact`: bit-identical
  `Engram.forward` for resident row-sharded blocked `wkv` and replicated `wkv` under SP.

## Prepared 2-Boot GPU Benchmark & Screening Harness

Ready to run once a cluster window is scheduled (no GPU touched during Phase 1):

- Cluster configs:
  - `experiments/2026-11-06-sp-engram-prefill/tp4.json`
  - `experiments/2026-11-06-sp-engram-prefill/tp3.json`
- Screening harness:
  - `experiments/2026-11-06-sp-engram-prefill/screen_engram.py`
