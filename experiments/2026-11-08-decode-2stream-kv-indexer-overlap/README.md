# Decode 2-Stream Attention Key & Indexer-Key Write Overlap (`TODO.md` #5)

- **Date**: 2026-11-08
- **Base deployment commit**: `8e5ea208f3d22bc69df91ace94d6975932c858a8` (`spark-ds41f:1107`, vLLM patch head `2756478b60b90700b9918b457aeb38657da637b5`, tree `ed1bdb7a7684da33d403a055521ac0c4dd7ca670`)
- **Candidate vLLM patch**: `experiments/2026-11-08-decode-2stream-kv-indexer-overlap/vllm/0762-perf-dsv41-overlap-decode-kv-index-writes-with-query.patch` (patch head `b15b07de21202825dbf2fbf4211a19c1bc8ab197`, tree `7cd86b2c256015c0615b1abf46f3772fc04aaf2b`)
- **Node**: `sp5` (`NVIDIA GB10`, 48 SMs, driver `580.178.04`, kernel `6.17.0-1014-nvidia`)
- **Intended causal variable**: Overlap independent decode compressor KV-write / indexer-key write work with the query / index-query / decode-metadata path across `main_stream` and `aux_stream()` inside captured CUDA graphs (`TODO.md` lines 206–209).
- **Verification**:
  - **100% bit-exact (`torch.equal`, `0` uint16 diffs, `max_abs_diff == 0.0`)** across all outputs (`out`, `swa_cache`, `kv_cache`, `idx_k_cache`, `topk_indices`, `candidates`) on all `TP=4` and `TP=3` shapes (`M = 1, 2, 3, 6, 12, 24, 48`) and all three indexed layer families (`r2_kv_index_source`, `r1_candidate_source`, `r1_index_only_reindex`), in both eager execution and captured CUDA graph replay across 2 fresh container boots.
  - **Unit & PDL contract tests**:
    - Boot 1 (`receipts/pytest_boot1.txt`): **127 passed in 69.88s**
    - Boot 2 (`receipts/pytest_boot2.txt`): **127 passed in 48.14s**
    - Files tested: `tests/models/test_deepseek_v4_1_attention.py`, `tests/models/test_deepseek_v4_1_compressor.py`, `tests/models/test_deepseek_v4_1_compressor_prepare.py`, `tests/models/test_deepseek_v4_1_l2_schedule.py`, `tests/kernels/test_deepseek_v41_tilelang_pdl.py`.

## Exact Commands

```bash
# Boot 1 & Boot 2 pytest
flock /tmp/gpu.lock docker run --rm --gpus all --ipc=host \
  -w /opt/spark-ds41f/candidate/vllm \
  -e PYTHONPATH=/b:/opt/spark-ds41f/candidate/vllm \
  -e TILELANG_CACHE_DIR=/tmp/tilelang-cache \
  -v /tmp/tilelang-cache:/tmp/tilelang-cache \
  -v $(pwd)/scripts/release_tests:/b:ro \
  -v $(pwd)/.work/vllm-2stream/vllm/models/deepseek_v4_1/compressor.py:/opt/spark-ds41f/candidate/vllm/vllm/models/deepseek_v4_1/compressor.py:ro \
  -v $(pwd)/.work/vllm-2stream/vllm/models/deepseek_v4_1/attention.py:/opt/spark-ds41f/candidate/vllm/vllm/models/deepseek_v4_1/attention.py:ro \
  -v $(pwd)/.work/vllm-2stream/vllm/models/deepseek_v4_1/tilelang/attention.py:/opt/spark-ds41f/candidate/vllm/vllm/models/deepseek_v4_1/tilelang/attention.py:ro \
  -v $(pwd)/.work/vllm-2stream/tests:/opt/spark-ds41f/candidate/vllm/tests:ro \
  --entrypoint pytest spark-ds41f:1107 -v -p no:cacheprovider -p tblib_stub -p offline_skip \
    tests/models/test_deepseek_v4_1_attention.py \
    tests/models/test_deepseek_v4_1_compressor.py \
    tests/models/test_deepseek_v4_1_compressor_prepare.py \
    tests/models/test_deepseek_v4_1_l2_schedule.py \
    tests/kernels/test_deepseek_v41_tilelang_pdl.py

# Boot 1 & Boot 2 benchmarks
flock /tmp/gpu.lock docker run --rm --gpus all --ipc=host \
  -e TILELANG_CACHE_DIR=/tmp/tilelang-cache \
  -v /tmp/tilelang-cache:/tmp/tilelang-cache \
  -v $(pwd)/experiments/2026-11-08-decode-2stream-kv-indexer-overlap:/workspace \
  -v $(pwd)/.work/vllm-2stream/vllm/models/deepseek_v4_1/compressor.py:/opt/spark-ds41f/candidate/vllm/vllm/models/deepseek_v4_1/compressor.py:ro \
  -v $(pwd)/.work/vllm-2stream/vllm/models/deepseek_v4_1/attention.py:/opt/spark-ds41f/candidate/vllm/vllm/models/deepseek_v4_1/attention.py:ro \
  -v $(pwd)/.work/vllm-2stream/vllm/models/deepseek_v4_1/tilelang/attention.py:/opt/spark-ds41f/candidate/vllm/vllm/models/deepseek_v4_1/tilelang/attention.py:ro \
  --entrypoint python3 spark-ds41f:1107 /workspace/bench_decode_2stream_kv_indexer.py --boot <1|2> --out /workspace/receipts/boot<1|2>.json
```

---

## 1. Bounding Serial Dependency vs. SM Occupancy in the Pre-`indexer_score` Region

In `spark-ds41f:1107`, each DeepSeek-V4.1 step executes 60 attention layers:
- **52 sliding-window-only (SWA) layers**: `wqa_wkv` -> `attention_front` (fused `q_a_norm` + `q_cast` + `kv_a_norm` + `k_rope` + SWA cache write) -> `wq_b` -> `_rope(q)` -> `sparse_mla(idx_cache=None)` -> `o_proj`. Here `kv_a_norm` and the SWA KV write are already fused into `attention_front` alongside `q_a_norm` before `wq_b`.
- **8 indexed (global MLA) layers**:
  - **3 Ratio-2 KV+Index source layers (`Layers 2, 8, 14`)**: `compress_ratio = 2`, owns `compressor` + `indexer.wk` + `indexer.wq_b` + `indexer.weights_proj` + `tl_metadata.pages` + `tl_metadata.decode_metadata`.
  - **1 Ratio-1 Candidate source layer (`Layer 20`)**: `compress_ratio = 1`, `candidate_blocks = True`, owns `compressor` + `indexer.wk` + `indexer.wq_b` + `indexer.weights_proj` + `tl_metadata.pages` + `tl_metadata.decode_metadata`.
  - **4 Ratio-1 Index-only reindex layers (`Layers 24, 28, 32, 36`)**: `compress_ratio = 1`, `candidates = True`, `compressor = None`; reuses Layer 20's compressed KV cache, index-key pool, `index_pages`, `decode_metadata`, and 32 candidate blocks, running `wq_b` -> `_rope(q)` -> `indexer.wq_b` -> `indexer.weights_proj` -> `indexer_score(candidates=True)` -> `indexer_topk` -> `sparse_mla`.

### CTA Occupancy Breakdown on GB10 (48 SMs)

After `wq_b` and `compressor._project_local` (`project_parts` + TP `all_gather`), the pre-`indexer_score` region launches 10 small kernels whose grids occupy only a fraction of GB10's 48 SMs at decode batch sizes `M = 1..48`:

| Kernel | Stream Dependency Input | Grid CTAs (`M = 1..48`) | GB10 SM Occupancy (`48` SMs) |
|---|---|---:|---:|
| `_rope(q)` (`rope_`) | `q` from `wq_b` | `1..48` CTAs (`M` CTAs) | `2.1%..100%` |
| `compressor.gather_columns_into` | `gathered` from `compressor._project_local` | `1..12` CTAs (`ceil(M/4)`) | `2.1%..25.0%` |
| `compressor.gather_partial_state` | `state_pool`, `req_pos` | `1..48` CTAs (`live_reqs`) | `2.1%..100%` |
| `compressor.compress(write_kv=True)` | `values, gates` + `pending_state` | `1..48` CTAs (`live_reqs`) | `2.1%..100%` |
| `indexer.wk` (`bf16_gemm_partials`) | `latent` from `compress` | `4..24` CTAs (`2 * 2 * ceil(M/bm)`) | `8.3%..50.0%` |
| `norm_rope_write_index_keys` | `wk_partials` from `indexer.wk` | `1..6` CTAs (`ceil(M/8)`) | `2.1%..12.5%` |
| `indexer.wq_b` (`mxfp8_gemm_rope_mxfp4_decode`) | `q_cast` from `attention_front` | `16..48` CTAs (`16 * ceil(M/bm_fp8)`) | `33.3%..100%` |
| `indexer.weights_proj` (`bf16_partials` + `splitk_reduce`) | `hidden_states` from `attn_norm` | `5..30` + `1..48` CTAs | `10.4%..62.5%` |
| `tl_metadata.pages` | `req_id_per_token`, `cache_lengths` | `1..48` CTAs (`M` CTAs) | `2.1%..100%` |
| `tl_metadata.decode_metadata` | `positions`, `block_table` | `1..48` CTAs (`M` CTAs) | `2.1%..100%` |

Because `1107` chains all 10 kernels on a single stream with PDL (`pdl.chain()`), at most two adjacent kernels overlap their launch/tail at a time (`T.pdl_sync()` holds each kernel until its immediate stream predecessor finishes), even when adjacent kernels in stream order have **zero data dependency** (e.g., `_index_query` only needs `q_cast` and `hidden_states`, not `compressor` or `indexer.wk`; and `_rope(q)` + `decode_metadata` only need `q` and `block_table`, not `indexer_score` / `indexer_topk`).

### Pre-`indexer_score` Region: Standalone Sum vs. 1-Stream PDL vs. 2-Stream Overlap (2-Boot)

#### `TP = 4` (`heads = 16`, `part_cols = 128`)

| `M` | `q_rope` (us) | `gather_cols` (us) | `gather_state` + `compress` (us) | `wk` + `write_idx_k` (us) | `iwq_b` (us) | `weights_proj` (us) | `pages` + `decode_meta` (us) | Sum Standalone B1/B2 (us) | 1-Stream PDL B1/B2 (us) | 1-Stream Mean (us) | 2-Stream Graph B1/B2 (us) | 2-Stream Mean (us) | $\Delta$ 2s vs 1s (us) | Speedup (%) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.71 | 2.58 | 9.20 | 7.13 | 7.73 | 4.26 | 6.22 | 39.93 / 39.77 | 18.43 / 18.67 | 18.55 | 13.88 / 14.51 | 14.20 | **-4.36** | **-23.5%** |
| 2 | 2.73 | 2.73 | 8.99 | 7.15 | 7.16 | 4.09 | 5.95 | 39.07 / 38.56 | 18.43 / 18.85 | 18.64 | 14.32 / 14.17 | 14.25 | **-4.39** | **-23.6%** |
| 3 | 2.71 | 2.67 | 9.37 | 7.15 | 7.88 | 4.56 | 6.03 | 38.66 / 42.08 | 19.06 / 19.35 | 19.20 | 14.33 / 14.32 | 14.32 | **-4.88** | **-25.4%** |
| 6 | 2.79 | 2.73 | 9.24 | 7.42 | 8.45 | 4.09 | 5.85 | 40.43 / 40.69 | 19.56 / 19.41 | 19.48 | 14.28 / 14.23 | 14.25 | **-5.23** | **-26.8%** |
| 12 | 2.78 | 2.78 | 9.68 | 7.28 | 8.84 | 4.30 | 5.84 | 40.55 / 42.42 | 19.94 / 20.46 | 20.20 | 13.61 / 15.49 | 14.55 | **-5.65** | **-28.0%** |
| 24 | 2.81 | 2.77 | 9.48 | 8.16 | 13.14 | 4.11 | 6.58 | 46.40 / 47.72 | 25.51 / 25.90 | 25.70 | 17.36 / 19.29 | 18.32 | **-7.38** | **-28.7%** |
| 48 | 2.80 | 2.91 | 9.78 | 8.19 | 15.00 | 4.11 | 6.12 | 48.79 / 49.02 | 29.10 / 29.57 | 29.34 | 20.72 / 23.05 | 21.88 | **-7.45** | **-25.4%** |

#### `TP = 3` (`heads = 24`, `part_cols = 192`)

| `M` | `q_rope` (us) | `gather_cols` (us) | `gather_state` + `compress` (us) | `wk` + `write_idx_k` (us) | `iwq_b` (us) | `weights_proj` (us) | `pages` + `decode_meta` (us) | Sum Standalone B1/B2 (us) | 1-Stream PDL B1/B2 (us) | 1-Stream Mean (us) | 2-Stream Graph B1/B2 (us) | 2-Stream Mean (us) | $\Delta$ 2s vs 1s (us) | Speedup (%) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3.06 | 3.05 | 9.29 | 7.50 | 7.79 | 4.09 | 7.12 | 41.90 / 41.96 | 19.02 / 18.83 | 18.92 | 14.25 / 14.27 | 14.26 | **-4.66** | **-24.6%** |
| 2 | 3.07 | 3.07 | 9.22 | 7.54 | 8.43 | 4.09 | 7.18 | 42.58 / 42.62 | 19.19 / 19.28 | 19.23 | 14.27 / 14.26 | 14.27 | **-4.97** | **-25.8%** |
| 3 | 3.08 | 3.06 | 9.21 | 7.54 | 8.18 | 4.09 | 7.12 | 42.49 / 42.11 | 19.11 / 18.72 | 18.91 | 14.29 / 14.29 | 14.29 | **-4.62** | **-24.5%** |
| 6 | 3.07 | 3.06 | 9.23 | 7.56 | 8.84 | 4.09 | 7.13 | 42.87 / 43.09 | 19.74 / 19.58 | 19.66 | 12.41 / 12.33 | 12.37 | **-7.29** | **-37.1%** |
| 12 | 3.08 | 3.08 | 9.73 | 7.92 | 9.59 | 4.09 | 7.32 | 44.45 / 45.20 | 20.81 / 21.07 | 20.94 | 14.35 / 14.35 | 14.35 | **-6.59** | **-31.5%** |
| 24 | 3.27 | 3.08 | 10.84 | 8.18 | 12.59 | 4.10 | 7.13 | 50.13 / 48.27 | 26.55 / 26.05 | 26.30 | 18.30 / 18.40 | 18.35 | **-7.95** | **-30.2%** |
| 48 | 3.40 | 3.25 | 9.73 | 8.20 | 16.00 | 4.09 | 7.38 | 52.45 / 51.67 | 28.68 / 29.61 | 29.14 | 21.85 / 22.62 | 22.23 | **-6.91** | **-23.7%** |

- **Finding**: In the pre-`indexer_score` region alone, 1-stream PDL already hides `8.6–10.8 us` of launch gaps relative to standalone kernel launches, but **serial stream ordering still adds `4.16–8.38 us` (`22.3%–37.1%`) of avoidable latency** because `_index_query`, `_rope(q)`, and `decode_metadata` wait in line behind the 5-kernel compressor + `indexer.wk` chain. Moving the independent branches to `aux_stream()` recovers that `4.36–8.26 us` in the pre-`indexer_score` region, and overlapping Phase 2 (`_rope(q)` + `decode_metadata` alongside `indexer_score` + `indexer_topk`) saves an additional `3–5 us/layer` during `forward_mqa`.

---

## 2. Two-Stream Decode Overlap Design (`vllm 0762`)

1. **`DeepseekCompressor._project_local` / `_finish_project` split (`vllm/models/deepseek_v4_1/compressor.py`)**:
   - Factors `_project(hidden_states)` into `_project_local(hidden_states)` (`project_parts` + `tensor_model_parallel_all_gather`) and `_finish_project(gathered, rows)` (`gather_columns_into`), keeping collective communication on the primary stream before forking `aux_stream()`.
2. **Disjoint Two-Stream Workspace Reservation (`vllm/models/deepseek_v4_1/attention.py`, `tilelang/attention.py`)**:
   - `_decode_overlap_workspace_specs(DECODE_CHUNK)` reserves two disjoint `uint8` slices `(main_scratch, aux_scratch)` in `self._scratch_reservations` sized for each layer's concurrent `main_stream` and `aux_stream()` operations, eliminating any workspace clobbering between streams without extra dynamic allocations.
3. **Two-Phase Schedule on KV+Index Source Layers (`Layers 2, 8, 14, 20`)**:
   - After `wq_b` + `_issue_wo_prefetch(rows, after="q-b")`, `compressor._project_local(hidden_states)` runs on `main_stream`.
   - **Phase 1**: `aux_stream.wait_stream(main_stream)` forks `aux_stream()` to run `_index_query(hidden_states, positions, q_cast=q_cast, scratch=aux_scratch)` (`project_block32_rope_mxfp4_decode(indexer.wq_b)` + `indexer.weights_proj`) concurrently with `main_stream` running `compressor._finish_project` -> `compressor._forward_tilelang` (`gather_partial_state` + `compress(write_kv=True)`) -> `_issue_wo_prefetch(rows, after="compressor")` -> `project_bf16_partials(indexer.wk, latent)` -> `_write_index_keys` -> `tl_metadata.pages`.
   - **Phase 2 (inside `forward_mqa`)**: `main_stream.wait_stream(aux_stream)` joins Phase 1 so `_index_chunk` has `(index_q, index_q_scale, index_weights)` and `index_pages`, then immediately forks `aux_stream()` to run `_rope(q, positions, "q")` + `_shared_decode_metadata(attn_metadata, rows, scratch=aux_scratch)` concurrently with `_index_chunk` (`indexer_score` + `indexer_topk`) on `main_stream`, joining `main_stream.wait_stream(aux_stream)` right before `_attend` (`sparse_mla`).
4. **Two-Stream Schedule on Index-Only Reindex Layers (`Layers 24, 28, 32, 36`)**:
   - After `wq_b` + `_issue_wo_prefetch(rows, after="q-b")`, `aux_stream.wait_stream(main_stream)` forks `aux_stream()` to run `_rope(q, positions, "q")` + `indexer.weights_proj(hidden_states, scratch=aux_scratch)` concurrently with `project_block32_rope_mxfp4_decode(indexer.wq_b, ..., scratch=main_scratch)` on `main_stream`, joining `main_stream.wait_stream(aux_stream)` before `forward_mqa`.

---

## 3. Full Decode Attention Sublayer 2-Boot Benchmarks (`TP = 4` and `TP = 3`)

Measured across 2 fresh container boots (`receipts/boot1.json`, `receipts/boot2.json`) using interleaved paired windows (`7` windows $\times$ `15` warm graph replays per window, plus `7` cold-L2 windows with a 48 MB L2 flush before each 1-layer graph replay and active L2 weight prefetch).

### `TP = 4` (`heads = 16`)

#### Family 1: Ratio-2 KV+Index Source Layers (`Layers 2, 8, 14` — 3 layers/step)

| `M` | Warm Base B1/B2 (us) | Warm Base Mean (us) | Warm Cand B1/B2 (us) | Warm Cand Mean (us) | Warm $\Delta$ (us/layer) | Warm $\Delta$ (3L us) | Warm (%) | Cold+L2pf Base B1/B2 (us) | Cold+L2pf Cand B1/B2 (us) | Cold+L2pf $\Delta$ (us/layer) | Cold+L2pf (%) | Bit-Exact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 66.67 / 62.87 | 64.77 | 57.44 / 52.54 | 54.99 | **-9.78** | **-29.34** | **-15.10%** | 160.45 / 160.48 | 151.55 / 151.65 | **-8.86** | **-5.52%** | `0` diffs |
| 2 | 68.47 / 65.28 | 66.88 | 60.03 / 55.24 | 57.64 | **-9.24** | **-27.72** | **-13.82%** | 160.48 / 160.48 | 153.60 / 153.31 | **-7.03** | **-4.38%** | `0` diffs |
| 3 | 71.26 / 72.13 | 71.69 | 62.73 / 61.89 | 62.31 | **-9.38** | **-28.15** | **-13.09%** | 160.86 / 162.56 | 152.32 / 153.86 | **-8.62** | **-5.33%** | `0` diffs |
| 6 | 81.15 / 72.15 | 76.65 | 72.07 / 61.47 | 66.77 | **-9.88** | **-29.64** | **-12.89%** | 162.85 / 162.85 | 155.01 / 156.06 | **-7.31** | **-4.49%** | `0` diffs |
| 12 | 96.63 / 101.30 | 98.97 | 85.57 / 90.91 | 88.24 | **-10.73** | **-32.18** | **-10.84%** | 178.27 / 179.23 | 168.38 / 169.38 | **-9.87** | **-5.52%** | `0` diffs |
| 24 | 142.34 / 146.16 | 144.25 | 131.58 / 134.82 | 133.20 | **-11.05** | **-33.15** | **-7.66%** | 223.30 / 225.34 | 216.26 / 217.25 | **-7.56** | **-3.37%** | `0` diffs |
| 48 | 181.50 / 176.54 | 179.02 | 173.51 / 168.02 | 170.76 | **-8.25** | **-24.76** | **-4.61%** | 239.65 / 237.60 | 236.58 / 235.49 | **-2.59** | **-1.09%** | `0` diffs |

#### Family 2: Ratio-1 Candidate Source Layer (`Layer 20` — 1 layer/step)

| `M` | Warm Base B1/B2 (us) | Warm Base Mean (us) | Warm Cand B1/B2 (us) | Warm Cand Mean (us) | Warm $\Delta$ (us/layer) | Warm $\Delta$ (1L us) | Warm (%) | Cold+L2pf Base B1/B2 (us) | Cold+L2pf Cand B1/B2 (us) | Cold+L2pf $\Delta$ (us/layer) | Cold+L2pf (%) | Bit-Exact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 66.88 / 66.06 | 66.47 | 58.41 / 56.92 | 57.66 | **-8.80** | **-8.80** | **-13.25%** | 163.52 / 163.39 | 156.38 / 155.55 | **-7.49** | **-4.58%** | `0` diffs |
| 2 | 64.31 / 69.05 | 66.68 | 56.10 / 61.08 | 58.59 | **-8.09** | **-8.09** | **-12.13%** | 164.51 / 163.58 | 157.38 / 157.38 | **-6.67** | **-4.06%** | `0` diffs |
| 3 | 68.56 / 69.72 | 69.14 | 60.49 / 61.40 | 60.95 | **-8.20** | **-8.20** | **-11.85%** | 165.09 / 166.21 | 158.05 / 157.76 | **-7.75** | **-4.68%** | `0` diffs |
| 6 | 75.22 / 79.18 | 77.20 | 65.21 / 71.96 | 68.58 | **-8.62** | **-8.62** | **-11.16%** | 167.10 / 170.30 | 161.09 / 161.44 | **-7.44** | **-4.41%** | `0` diffs |
| 12 | 100.10 / 107.93 | 104.02 | 90.91 / 98.82 | 94.86 | **-9.15** | **-9.15** | **-8.80%** | 181.57 / 184.38 | 174.11 / 176.38 | **-7.73** | **-4.22%** | `0` diffs |
| 24 | 146.01 / 156.41 | 151.21 | 134.74 / 148.19 | 141.47 | **-9.74** | **-9.74** | **-6.44%** | 229.82 / 231.46 | 221.25 / 223.30 | **-8.36** | **-3.63%** | `0` diffs |
| 48 | 186.15 / 192.51 | 189.33 | 177.12 / 184.44 | 180.78 | **-8.55** | **-8.55** | **-4.52%** | 250.88 / 249.92 | 243.78 / 247.14 | **-4.94** | **-1.97%** | `0` diffs |

#### Family 3: Ratio-1 Index-Only Reindex Layers (`Layers 24, 28, 32, 36` — 4 layers/step)

| `M` | Warm Base B1/B2 (us) | Warm Base Mean (us) | Warm Cand B1/B2 (us) | Warm Cand Mean (us) | Warm $\Delta$ (us/layer) | Warm $\Delta$ (4L us) | Warm (%) | Cold+L2pf Base B1/B2 (us) | Cold+L2pf Cand B1/B2 (us) | Cold+L2pf $\Delta$ (us/layer) | Cold+L2pf (%) | Bit-Exact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 46.20 / 44.65 | 45.42 | 43.62 / 41.73 | 42.67 | **-2.75** | **-11.00** | **-6.05%** | 142.05 / 141.86 | 140.26 / 140.00 | **-1.83** | **-1.29%** | `0` diffs |
| 2 | 43.56 / 48.31 | 45.94 | 40.93 / 44.92 | 42.92 | **-3.01** | **-12.04** | **-6.55%** | 143.04 / 143.07 | 142.02 / 141.31 | **-1.39** | **-0.97%** | `0` diffs |
| 3 | 47.89 / 46.56 | 47.23 | 44.67 / 43.43 | 44.05 | **-3.18** | **-12.70** | **-6.72%** | 143.55 / 144.45 | 142.34 / 142.40 | **-1.63** | **-1.13%** | `0` diffs |
| 6 | 52.97 / 59.11 | 56.04 | 49.10 / 56.16 | 52.63 | **-3.41** | **-13.64** | **-6.08%** | 145.95 / 148.54 | 146.56 / 146.50 | **-0.72** | **-0.49%** | `0` diffs |
| 12 | 74.13 / 81.42 | 77.78 | 70.37 / 77.97 | 74.17 | **-3.61** | **-14.42** | **-4.64%** | 162.85 / 164.35 | 160.80 / 161.82 | **-2.29** | **-1.40%** | `0` diffs |
| 24 | 118.88 / 160.71 | 139.80 | 114.18 / 145.82 | 130.00 | **-9.80** | **-39.18** | **-7.01%** | 212.96 / 215.10 | 210.24 / 214.40 | **-1.71** | **-0.80%** | `0` diffs |
| 48 | 154.50 / 164.79 | 159.64 | 150.60 / 160.96 | 155.78 | **-3.86** | **-15.46** | **-2.42%** | 237.15 / 236.58 | 230.43 / 230.46 | **-6.42** | **-2.71%** | `0` diffs |

#### Aggregate 8 Indexed Layers per Decode Step (`3 × R2 + 1 × R1_src + 4 × Reidx`) — `TP = 4`

| `M` | Warm 8L Base B1/B2 (us) | Warm 8L Cand B1/B2 (us) | Warm 8L $\Delta$ B1/B2 (us) | Warm 8L $\Delta$ Mean (us/step) | Warm 8L (%) | Cold+L2pf 8L Base B1/B2 (us) | Cold+L2pf 8L Cand B1/B2 (us) | Cold+L2pf 8L $\Delta$ B1/B2 (us) | Cold+L2pf 8L $\Delta$ Mean (us/step) | Cold+L2pf 8L (%) | Bit-Exact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 451.69 / 433.27 | 405.21 / 381.46 | -46.48 / -51.81 | **-49.14** | **-11.11%** | 1213.07 / 1212.27 | 1172.07 / 1170.50 | -41.00 / -41.77 | **-41.39** | **-3.41%** | `0` diffs |
| 2 | 443.96 / 458.13 | 399.91 / 406.48 | -44.05 / -51.65 | **-47.85** | **-10.61%** | 1218.11 / 1217.30 | 1186.26 / 1182.55 | -31.85 / -34.75 | **-33.30** | **-2.73%** | `0` diffs |
| 3 | 473.90 / 472.35 | 427.36 / 420.79 | -46.54 / -51.56 | **-49.05** | **-10.37%** | 1221.87 / 1231.69 | 1184.37 / 1188.94 | -37.50 / -42.75 | **-40.12** | **-3.27%** | `0` diffs |
| 6 | 530.55 / 532.07 | 477.82 / 481.01 | -52.73 / -51.06 | **-51.89** | **-9.77%** | 1239.45 / 1253.01 | 1212.36 / 1215.62 | -27.09 / -37.39 | **-32.24** | **-2.59%** | `0` diffs |
| 12 | 686.51 / 737.51 | 629.10 / 683.43 | -57.41 / -54.08 | **-55.74** | **-7.83%** | 1367.78 / 1379.47 | 1322.45 / 1331.80 | -45.33 / -47.67 | **-46.50** | **-3.39%** | `0` diffs |
| 24 | 1048.55 / 1237.73 | 986.20 / 1135.93 | -62.35 / -101.80 | **-82.08** | **-7.18%** | 1751.56 / 1767.88 | 1710.99 / 1732.65 | -40.57 / -35.23 | **-37.90** | **-2.15%** | `0` diffs |
| 48 | 1348.65 / 1381.29 | 1300.05 / 1332.34 | -48.60 / -48.95 | **-48.78** | **-3.57%** | 1918.43 / 1909.04 | 1875.24 / 1875.45 | -43.19 / -33.59 | **-38.39** | **-2.01%** | `0` diffs |

---

### `TP = 3` (`heads = 24`)

#### Family 1: Ratio-2 KV+Index Source Layers (`Layers 2, 8, 14` — 3 layers/step)

| `M` | Warm Base B1/B2 (us) | Warm Base Mean (us) | Warm Cand B1/B2 (us) | Warm Cand Mean (us) | Warm $\Delta$ (us/layer) | Warm $\Delta$ (3L us) | Warm (%) | Cold+L2pf Base B1/B2 (us) | Cold+L2pf Cand B1/B2 (us) | Cold+L2pf $\Delta$ (us/layer) | Cold+L2pf (%) | Bit-Exact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 126.49 / 118.14 | 122.31 | 118.81 / 110.78 | 114.80 | **-7.52** | **-22.56** | **-6.15%** | 193.50 / 195.65 | 185.28 / 185.76 | **-9.06** | **-4.65%** | `0` diffs |
| 2 | 133.21 / 124.46 | 128.84 | 124.44 / 115.57 | 120.00 | **-8.83** | **-26.49** | **-6.85%** | 193.89 / 194.88 | 185.41 / 187.46 | **-7.95** | **-4.09%** | `0` diffs |
| 3 | 130.25 / 128.33 | 129.29 | 122.29 / 118.06 | 120.18 | **-9.12** | **-27.35** | **-7.05%** | 195.65 / 195.68 | 186.78 / 191.33 | **-6.61** | **-3.38%** | `0` diffs |
| 6 | 140.49 / 136.60 | 138.55 | 134.78 / 130.04 | 132.41 | **-6.14** | **-18.41** | **-4.43%** | 199.68 / 201.79 | 191.49 / 192.86 | **-8.56** | **-4.26%** | `0` diffs |
| 12 | 165.11 / 164.37 | 164.74 | 155.08 / 156.41 | 155.75 | **-9.00** | **-26.99** | **-5.46%** | 216.42 / 216.19 | 205.89 / 205.95 | **-10.39** | **-4.80%** | `0` diffs |
| 24 | 214.41 / 218.23 | 216.32 | 206.17 / 211.07 | 208.62 | **-7.70** | **-23.10** | **-3.56%** | 269.63 / 269.41 | 262.24 / 263.20 | **-6.80** | **-2.52%** | `0` diffs |
| 48 | 248.47 / 248.33 | 248.40 | 240.19 / 238.06 | 239.12 | **-9.28** | **-27.83** | **-3.73%** | 285.73 / 286.02 | 278.59 / 278.88 | **-7.14** | **-2.50%** | `0` diffs |

#### Family 2: Ratio-1 Candidate Source Layer (`Layer 20` — 1 layer/step)

| `M` | Warm Base B1/B2 (us) | Warm Base Mean (us) | Warm Cand B1/B2 (us) | Warm Cand Mean (us) | Warm $\Delta$ (us/layer) | Warm $\Delta$ (1L us) | Warm (%) | Cold+L2pf Base B1/B2 (us) | Cold+L2pf Cand B1/B2 (us) | Cold+L2pf $\Delta$ (us/layer) | Cold+L2pf (%) | Bit-Exact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 123.76 / 115.38 | 119.57 | 118.69 / 108.84 | 113.77 | **-5.80** | **-5.80** | **-4.85%** | 195.62 / 193.57 | 189.47 / 186.40 | **-6.66** | **-3.42%** | `0` diffs |
| 2 | 126.21 / 117.05 | 121.63 | 120.56 / 109.81 | 115.19 | **-6.44** | **-6.44** | **-5.30%** | 196.61 / 194.59 | 189.76 / 186.66 | **-7.39** | **-3.78%** | `0` diffs |
| 3 | 129.48 / 122.29 | 125.88 | 124.60 / 115.27 | 119.94 | **-5.95** | **-5.95** | **-4.73%** | 197.76 / 195.55 | 190.53 / 188.48 | **-7.15** | **-3.64%** | `0` diffs |
| 6 | 144.98 / 130.88 | 137.93 | 138.02 / 123.67 | 130.84 | **-7.09** | **-7.09** | **-5.14%** | 202.21 / 199.58 | 194.88 / 192.54 | **-7.19** | **-3.58%** | `0` diffs |
| 12 | 166.29 / 159.80 | 163.05 | 159.96 / 154.23 | 157.09 | **-5.95** | **-5.95** | **-3.65%** | 218.46 / 215.39 | 210.98 / 207.17 | **-7.85** | **-3.62%** | `0` diffs |
| 24 | 211.84 / 211.48 | 211.66 | 202.79 / 201.82 | 202.31 | **-9.35** | **-9.35** | **-4.42%** | 272.48 / 268.38 | 264.19 / 260.16 | **-8.25** | **-3.05%** | `0` diffs |
| 48 | 256.90 / 250.20 | 253.55 | 247.55 / 238.44 | 243.00 | **-10.55** | **-10.55** | **-4.16%** | 295.97 / 286.75 | 286.05 / 280.83 | **-7.92** | **-2.72%** | `0` diffs |

#### Family 3: Ratio-1 Index-Only Reindex Layers (`Layers 24, 28, 32, 36` — 4 layers/step)

| `M` | Warm Base B1/B2 (us) | Warm Base Mean (us) | Warm Cand B1/B2 (us) | Warm Cand Mean (us) | Warm $\Delta$ (us/layer) | Warm $\Delta$ (4L us) | Warm (%) | Cold+L2pf Base B1/B2 (us) | Cold+L2pf Cand B1/B2 (us) | Cold+L2pf $\Delta$ (us/layer) | Cold+L2pf (%) | Bit-Exact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 95.43 / 92.04 | 93.74 | 92.41 / 89.07 | 90.74 | **-3.00** | **-11.98** | **-3.20%** | 171.07 / 169.18 | 168.99 / 167.01 | **-2.12** | **-1.25%** | `0` diffs |
| 2 | 102.07 / 92.41 | 97.24 | 99.78 / 89.68 | 94.73 | **-2.51** | **-10.04** | **-2.58%** | 172.38 / 170.02 | 170.34 / 167.36 | **-2.35** | **-1.37%** | `0` diffs |
| 3 | 101.90 / 99.36 | 100.63 | 99.77 / 96.81 | 98.29 | **-2.34** | **-9.36** | **-2.33%** | 173.09 / 170.34 | 170.02 / 169.18 | **-2.11** | **-1.23%** | `0` diffs |
| 6 | 114.92 / 106.29 | 110.61 | 113.26 / 103.78 | 108.52 | **-2.08** | **-8.34** | **-1.89%** | 177.22 / 176.32 | 175.26 / 173.34 | **-2.47** | **-1.40%** | `0` diffs |
| 12 | 161.92 / 137.21 | 149.56 | 150.22 / 134.71 | 142.47 | **-7.10** | **-28.40** | **-4.75%** | 198.56 / 193.92 | 195.78 / 191.62 | **-2.54** | **-1.29%** | `0` diffs |
| 24 | 184.29 / 183.63 | 183.96 | 181.61 / 180.51 | 181.06 | **-2.90** | **-11.60** | **-1.58%** | 253.22 / 249.86 | 247.87 / 245.82 | **-4.70** | **-1.87%** | `0` diffs |
| 48 | 228.32 / 223.02 | 225.67 | 222.23 / 215.54 | 218.88 | **-6.79** | **-27.14** | **-3.01%** | 273.50 / 271.42 | 267.52 / 263.20 | **-7.10** | **-2.61%** | `0` diffs |

#### Aggregate 8 Indexed Layers per Decode Step (`3 × R2 + 1 × R1_src + 4 × Reidx`) — `TP = 3`

| `M` | Warm 8L Base B1/B2 (us) | Warm 8L Cand B1/B2 (us) | Warm 8L $\Delta$ B1/B2 (us) | Warm 8L $\Delta$ Mean (us/step) | Warm 8L (%) | Cold+L2pf 8L Base B1/B2 (us) | Cold+L2pf 8L Cand B1/B2 (us) | Cold+L2pf 8L $\Delta$ B1/B2 (us) | Cold+L2pf 8L $\Delta$ Mean (us/step) | Cold+L2pf 8L (%) | Bit-Exact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 884.95 / 837.96 | 844.76 / 797.46 | -40.19 / -40.50 | **-40.34** | **-4.68%** | 1460.40 / 1457.24 | 1421.27 / 1411.72 | -39.13 / -45.52 | **-42.33** | **-2.90%** | `0` diffs |
| 2 | 934.12 / 860.07 | 893.00 / 815.24 | -41.12 / -44.83 | **-42.97** | **-4.79%** | 1467.80 / 1459.31 | 1427.35 / 1418.48 | -40.45 / -40.83 | **-40.64** | **-2.78%** | `0` diffs |
| 3 | 927.83 / 904.72 | 890.55 / 856.69 | -37.28 / -48.03 | **-42.66** | **-4.66%** | 1477.07 / 1463.95 | 1430.95 / 1439.19 | -46.12 / -24.76 | **-35.44** | **-2.41%** | `0` diffs |
| 6 | 1026.13 / 965.84 | 995.40 / 928.91 | -30.73 / -36.93 | **-33.83** | **-3.40%** | 1510.13 / 1510.23 | 1470.39 / 1464.48 | -39.74 / -45.75 | **-42.75** | **-2.83%** | `0` diffs |
| 12 | 1309.30 / 1201.75 | 1226.08 / 1162.30 | -83.22 / -39.45 | **-61.34** | **-4.89%** | 1661.96 / 1639.64 | 1611.77 / 1591.50 | -50.19 / -48.14 | **-49.16** | **-2.98%** | `0` diffs |
| 24 | 1592.23 / 1600.69 | 1547.74 / 1557.07 | -44.49 / -43.62 | **-44.05** | **-2.76%** | 2094.25 / 2076.05 | 2042.39 / 2033.04 | -51.86 / -43.01 | **-47.44** | **-2.27%** | `0` diffs |
| 48 | 1915.59 / 1887.27 | 1857.04 / 1814.78 | -58.55 / -72.49 | **-65.52** | **-3.45%** | 2247.16 / 2230.49 | 2191.90 / 2170.27 | -55.26 / -60.22 | **-57.74** | **-2.58%** | `0` diffs |

---

## Conclusion

1. **Bound on Serial Dependency vs. SM Occupancy**:
   - In the pre-`indexer_score` decode region on GB10 (48 SMs), individual kernels launch `1–48` CTAs (`2.1%–100%` of SMs, median `< 25%` at `M <= 12`). Even with 1-stream PDL (`baseline_1107`), serial stream ordering between the compressor/`indexer.wk` branch and the query/`_index_query`/`decode_metadata` branch leaves `4.36–8.26 us/layer` (`22.3%–37.1%`) of serial dependency latency in the pre-`indexer_score` region alone.
2. **2-Boot Measured Speedup & Bit-Exactness**:
   - Across 2 fresh container boots on `sp5`, the 2-stream CUDA-graph overlap (`vLLM 0762`) is **100% bit-exact (`0` uint16 diffs, `max_abs_diff == 0.0`)** and reduces latency across every single batch size `M = 1..48` on both `TP = 4` and `TP = 3`:
     - **`TP = 4` (8 indexed layers/step)**:
       - Warm-L2 graph replay: **`-47.85` to `-82.08 us/step` (`-3.57%` to `-11.11%`)** across `M = 1..48` (`-8.26` to `-11.04 us/layer` on Ratio-2 KV+Index layers, `-8.09` to `-9.74 us/layer` on Ratio-1 Candidate Source layers, and `-2.75` to `-9.80 us/layer` on Ratio-1 Reindex layers).
       - Cold-L2 + L2-prefetch graph replay: **`-32.24` to `-46.50 us/step` (`-2.01%` to `-3.41%`)** across `M = 1..48`.
     - **`TP = 3` (8 indexed layers/step)**:
       - Warm-L2 graph replay: **`-33.83` to `-65.52 us/step` (`-2.76%` to `-4.89%`)** across `M = 1..48` (`-6.12` to `-9.28 us/layer` on Ratio-2 KV+Index layers, `-5.80` to `-10.56 us/layer` on Ratio-1 Candidate Source layers, and `-2.08` to `-7.10 us/layer` on Ratio-1 Reindex layers).
       - Cold-L2 + L2-prefetch graph replay: **`-35.44` to `-57.74 us/step` (`-2.27%` to `-2.98%`)** across `M = 1..48`.
