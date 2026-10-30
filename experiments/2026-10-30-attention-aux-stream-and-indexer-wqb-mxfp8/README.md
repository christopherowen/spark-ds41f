# 2026-10-30 — Attention Auxiliary-Stream Overlap & Indexer `wq_b` MXFP8 Cast Handover

## Objective

Implement the remaining non-overlapping decode-graph multi-stream & attention query/key path items from `TODO.md` (`A second stream inside the decode graph`, lines 422–426, and `CUDA graph launch cost`, lines 405–407):

1. **Audit of `TODO.md:405–407` (`CUDA graph launch cost`):**
   - Earlier (`2026-09-24`) synchronous Engram disk reads and piecewise graphs exposed `~0.86 ms` of `cudaGraphLaunch` latency.
   - With asynchronous Engram disk staging (`0109–0110`) and full-model CUDA graphs, critical-path trace analysis (`experiments/2026-10-25-pdl/README.md`) showed that total exposed graph launch boundary time across all 4 graph launches in a decode step is only `21–23 µs` (`0.04–0.08%`).
   - Direct measurement on `dgx4` (`spark-ds41f:1030`) of a 2,196-node `torch.cuda.CUDAGraph` shows host `replay()` takes `3.0 µs` (`1852.7 µs` wall time), whereas splitting into a 72-node lead chunk + 2,124-node main graph increases host `replay()` time to `5.3 µs` (`2.2 + 3.1 µs`). No graph-splitting patch is needed.

2. **Indexer `wq_b` MXFP8 cast handover (`DeepseekV4Indexer.wq_b.tilelang_mxfp8_input = True`):**
   - In `0544`, `attention_front` produced `q_cast = MXFP8Rows(qr, values, scales)` (the normalized `q` latent's MXFP8 cast) and handed it to `self.wq_b` so `self.wq_b` stopped running `per_token_cast(qr)`.
   - However, `self.indexer.wq_b` (which takes the exact same `qr` tensor on all 31 indexed target layers + 1 draft layer) still ran `pdl.per_token_cast(qr)` a second time and allocated workspace from `current_workspace_manager()`.
   - Marking `self.indexer.wq_b.tilelang_mxfp8_input = True` and passing `q_cast` into `_index_query` (casting `qr[first:last]` only when sequence-parallel prefill slices a strict row subset) eliminates 32 redundant `per_token_cast` kernel launches per step and makes the entire query path (`indexer.wq_b` -> `_rope(iq)` -> `_quantize_index_query`) zero-workspace.

3. **Auxiliary CUDA stream overlap inside captured decode graphs (`_index_query` and `_prepare_global_kv`):**
   - Because `indexer.wq_b` -> `_rope(iq)` -> `_quantize_index_query` uses zero workspace when `q_cast` is handed over, `self.indexer.weights_proj(hidden_states)` (`(32, 5120)` BF16 split-K projection) can run concurrently on `aux_stream()` via `maybe_execute_in_parallel` at decode row counts (`rows <= DECODE_ROWS`) without any workspace contention.
   - In `_prepare_global_kv`, once `key = indexer.k_norm(indexer.wk(latent))` has read the unrotated `latent`, the index-key path (`_rope(key, "index_key")` + `_write_index_keys`) and the compressed KV-cache path (`_rope(latent, "latent")` + `_write_v41_cache`) operate on disjoint buffers with zero workspace usage and run concurrently across the main stream and `aux_stream()` at decode row counts (`rows <= DECODE_ROWS`).

## Verification & Benchmarks on `dgx4` (`spark-ds41f:1030`)

### 1. CUDA-Graph Replay Latency (`dgx4`)

| Rows (`M`) | `_index_query` Baseline (µs) | `q_cast` Handover Only (µs) | `q_cast` Handover + `aux_stream` `weights_proj` (µs) | `_prepare_global_kv` Serial Writes (µs) | `_prepare_global_kv` `aux_stream` Parallel Writes (µs) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `1` | `14.80` | `14.38` (`-0.42`) | **`12.24` (`-2.56`, `-17.3%`)** | `4.98` | **`3.11` (`-1.87`, `-37.6%`)** |
| `4` | `15.02` | `14.39` (`-0.63`) | **`12.17` (`-2.85`, `-19.0%`)** | `5.35` | **`3.24` (`-2.11`, `-39.4%`)** |
| `16` | `15.22` | `14.43` (`-0.79`) | **`13.46` (`-1.76`, `-11.6%`)** | `5.90` | **`3.87` (`-2.03`, `-34.4%`)** |
| `48` | `22.58` | `22.37` (`-0.21`) | **`20.71` (`-1.87`, `-8.3%`)** | `6.39` | **`4.29` (`-2.10`, `-32.9%`)** |

Across 32 indexed layers and 10 global KV source layers per step, this saves **`~100.6 µs/step` at `M=1`** and **`~112.3 µs/step` at `M=4`**.

### 2. Correctness & PDL Verification

- **Bit-identity (`torch.equal`):** 100% bit-identical (`torch.equal`) across `rows in {1, 4, 16, 48, 64, 96}` in both eager execution and captured `torch.cuda.CUDAGraph` replay for `(iq_data, iq_scale, weights)` in `_index_query` and `(k_cache, kv_cache, latent)` in `_prepare_global_kv`.
- **PDL Source Audit:** `0 violations`.
