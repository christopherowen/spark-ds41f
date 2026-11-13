# 2026-11-10-indexer-multirow-and-decode-split

- **Base commit**: `cc20ed18be1f7cbf5c6c53cdee945f7cc77d5fc5`
- **Branch**: `perf/indexer-multirow-and-decode-split`
- **Causal variable**: Pair consecutive query rows in dense `indexer_score_v41` so rows of the same request share staged 64-position key+scale tiles in a single `[64, 128] x [64, 128]` block-scaled MMA, and split target-model decode indexer rows across TP ranks (`VLLM_DS41_DECODE_INDEX_SPLIT=1`).

## Rationale

1. **Dense `indexer_score_v41` key-tile reuse across row pairs**:
   - The four dense index layers (`2`, `8`, `14` at ratio 2; `20` at ratio 1) scan the entire active context (`min(Lengths[row], Active[0], width)`). Previously `indexer_score_v41` launched `grid = (ctas, rows)` and each query row loaded its own `64`-position MXFP4 key and UE8M0 scale tiles from `KPool` / `KScale` (~68 B per key position per row, or ~8.9 MB per row at 131K keys and ~68 MB per row at 1M keys), even when consecutive rows in prefill or speculative decode belonged to the same request and read identical pages.
   - Pairing consecutive rows (`grid = (ctas, ceildiv(rows, 2))`, `row0 = row_pair * 2`, `row1 = row0 + 1`) checks once per CTA whether `row0` and `row1` share page-table entries and tile coverage across that CTA's tiles. When `can_share` holds, the CTA loads each `64`-key tile once into shared memory (`K_shared`, `SFK_shared`) and scores both rows' 32 query heads together in a single unbranched `[64, 128] x [64, 128]` `T.mma_gemm_blockscaled` call (`Q_shared` `[64, 128]`, `SFQ_shared` `[64, 1]`), followed by independent per-row BF16 ReLU, head-weight multiply, and increasing-head-order FP32 sum. When `can_share` is false (odd trailing row, cross-request boundary, or partial tail tile with differing causal extents), the CTA falls back to per-row scoring.

2. **TP rank-split decode indexing (`VLLM_DS41_DECODE_INDEX_SPLIT=1`)**:
   - Because the indexer is replicated across TP ranks, decode previously ran `_index_query` and `forward_mqa` top-k selection for all `rows` on every TP rank, whereas sequence-parallel prefill already splits indexer rows across TP ranks via `sp_prefill.SPRows(rows, tp_size, tp_rank)`.
   - In `DeepseekV41TileLangAttention._forward`, when `split is None`, `swa.is_decode`, `not self.is_draft`, and `VLLM_DS41_DECODE_INDEX_SPLIT=1` (default enabled), target-model decode indexing now creates `split = sp_prefill.SPRows(rows, tp_size, tp_rank)` whenever `tp_size > 1 and rows >= tp_size`.
   - Each rank computes `_index_query` and runs `IndexerV41Plan` (`score` + `select` + optional `select_blocks`) only on its local row slice (`ceil(rows / tp_size)` rows), keeps candidate source/consumer buffers (`_candidates`, `_candidate_lens` from layer `20` to layers `26`, `32`, `38`, `44`) rank-local, caches its local decode page-table slice (`_ds41_index_pages`) across layers `2`, `8`, and `14`, and all-gathers the compact `[local_rows, 512]` `int32` `topk_indices_buffer` (4 KB per rank at 2 rows, 24 KB per rank at 12 rows).

## Quality / Bit-Identity

- **Bit-exact**: `BitExact = True` across all tested widths (`16,384`, `65,536`, `131,072`), page sizes (`64`, `128`), and row counts (`1..256`), including odd row counts, mixed multi-request batches, and unaligned tail extents.
- Verified with `compute-sanitizer --tool initcheck` (`0` uninitialized-read errors) and PDL source audit (`0` violations).

## Commands

```bash
docker run --rm --gpus all --ipc=host --entrypoint bash \
  -v /tmp/tilelang-cache:/root/.tilelang/cache \
  -v "$VLLM_SRC/vllm/models/deepseek_v4_1:/opt/spark-ds41f/candidate/vllm/vllm/models/deepseek_v4_1:ro" \
  -v "$VLLM_SRC/tests/kernels/attention/test_deepseek_v41_tilelang.py:/tmp/test_deepseek_v41_tilelang.py:ro" \
  -v "$VLLM_SRC/tests/models/test_deepseek_v4_1_attention.py:/tmp/test_deepseek_v4_1_attention.py:ro" \
  -v "$VLLM_SRC/tests/models/test_deepseek_v4_1_sp_prefill.py:/tmp/test_deepseek_v4_1_sp_prefill.py:ro" \
  -v "$EXP_DIR/bench_indexer.py:/tmp/bench_indexer.py:ro" \
  spark-ds41f:candidate-57 -c "
    pytest -q /tmp/test_deepseek_v41_tilelang.py -k 'indexer or pages'
    pytest -q /tmp/test_deepseek_v4_1_attention.py /tmp/test_deepseek_v4_1_sp_prefill.py
    python3 /tmp/bench_indexer.py
  "
```

## Two-Boot Results (`sp8`, GB10 SM121)

### Unit Tests

- `boot1`: `16 passed, 1 skipped, 1 xfailed` (`test_deepseek_v41_tilelang.py -k "indexer or pages"`) + `248 passed` (`test_deepseek_v4_1_attention.py` and `test_deepseek_v4_1_sp_prefill.py`)
- `boot2`: `16 passed, 1 skipped, 1 xfailed` (`test_deepseek_v41_tilelang.py -k "indexer or pages"`) + `248 passed` (`test_deepseek_v4_1_attention.py` and `test_deepseek_v4_1_sp_prefill.py`)

### Part A: Dense `indexer_score_v41` (1-Row per CTA vs 2-Row Fused MMA Key-Tile Reuse)

| Keys | Page | Rows | ReqSpan | 1-Row `boot1` / `boot2` (us) | 2-Row `boot1` / `boot2` (us) | Speedup `boot1` / `boot2` | Bit-Exact |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 16,384 | 64 | 6 | 6 | 25.52 / 25.50 | 21.61 / 21.82 | 1.18x / 1.17x | True |
| 16,384 | 64 | 12 | 6 | 51.63 / 48.99 | 39.42 / 39.47 | 1.31x / 1.24x | True |
| 16,384 | 64 | 24 | 6 | 94.72 / 96.85 | 74.88 / 74.74 | 1.26x / 1.30x | True |
| 16,384 | 64 | 48 | 6 | 186.78 / 186.69 | 150.69 / 149.06 | 1.24x / 1.25x | True |
| 16,384 | 64 | 256 | 256 | 985.11 / 983.97 | 785.21 / 785.55 | 1.25x / 1.25x | True |
| 65,536 | 64 | 6 | 6 | 97.37 / 95.62 | 73.83 / 76.20 | 1.32x / 1.25x | True |
| 65,536 | 64 | 12 | 6 | 187.70 / 191.56 | 143.07 / 141.14 | 1.31x / 1.36x | True |
| 65,536 | 64 | 24 | 6 | 370.61 / 373.10 | 278.75 / 281.66 | 1.33x / 1.32x | True |
| 65,536 | 64 | 48 | 6 | 745.34 / 746.82 | 565.91 / 563.29 | 1.32x / 1.33x | True |
| 65,536 | 64 | 256 | 256 | 3938.99 / 3914.59 | 2968.50 / 2970.67 | 1.33x / 1.32x | True |
| 65,536 | 128 | 6 | 6 | 102.53 / 100.25 | 72.62 / 77.78 | 1.41x / 1.29x | True |
| 65,536 | 128 | 12 | 6 | 199.40 / 202.52 | 141.47 / 143.46 | 1.41x / 1.41x | True |
| 65,536 | 128 | 24 | 6 | 397.90 / 398.80 | 279.04 / 279.72 | 1.43x / 1.43x | True |
| 65,536 | 128 | 48 | 6 | 799.26 / 803.63 | 558.36 / 558.03 | 1.43x / 1.44x | True |
| 65,536 | 128 | 256 | 256 | 4245.10 / 4205.01 | 2974.07 / 2970.48 | 1.43x / 1.42x | True |
| 131,072 | 128 | 6 | 6 | 196.96 / 198.63 | 143.11 / 145.02 | 1.38x / 1.37x | True |
| 131,072 | 128 | 12 | 6 | 387.66 / 391.72 | 276.92 / 279.18 | 1.40x / 1.40x | True |
| 131,072 | 128 | 24 | 6 | 793.29 / 786.40 | 554.88 / 554.12 | 1.43x / 1.42x | True |
| 131,072 | 128 | 48 | 6 | 1594.80 / 1593.81 | 1117.81 / 1193.28 | 1.43x / 1.34x | True |
| 131,072 | 128 | 256 | 256 | 8095.17 / 8165.74 | 5650.35 / 5654.76 | 1.43x / 1.44x | True |

### Part B: Full 8-Layer Decode Indexer Step (Layers 2, 8, 14, 20, 26, 32, 38, 44)

| Context | Rows | Base Unsplit `boot1` / `boot2` (ms) | TP4 Split+2Row `boot1` / `boot2` (ms) | TP4 Saved `boot1` / `boot2` (ms) | TP3 Split+2Row `boot1` / `boot2` (ms) | TP3 Saved `boot1` / `boot2` (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32,768 | 6 | 0.333 / 0.337 | 0.283 / 0.289 | 0.050 / 0.048 | 0.284 / 0.287 | 0.049 / 0.050 |
| 32,768 | 24 | 1.058 / 1.060 | 0.327 / 0.318 | 0.731 / 0.742 | 0.398 / 0.399 | 0.660 / 0.661 |
| 32,768 | 48 | 2.038 / 2.035 | 0.549 / 0.553 | 1.489 / 1.482 | 0.665 / 0.668 | 1.372 / 1.367 |
| 32,768 | 96 | 4.178 / 4.176 | 0.939 / 0.941 | 3.239 / 3.235 | 1.246 / 1.239 | 2.932 / 2.938 |
| 131,072 | 6 | 0.763 / 0.769 | 0.337 / 0.353 | 0.426 / 0.416 | 0.328 / 0.337 | 0.435 / 0.432 |
| 131,072 | 24 | 2.721 / 2.698 | 0.699 / 0.650 | 2.023 / 2.049 | 0.843 / 0.821 | 1.878 / 1.877 |
| 131,072 | 48 | 5.365 / 5.354 | 1.166 / 1.168 | 4.198 / 4.186 | 1.517 / 1.517 | 3.848 / 3.837 |
| 131,072 | 96 | 11.138 / 11.092 | 2.235 / 2.234 | 8.902 / 8.858 | 3.023 / 3.017 | 8.115 / 8.075 |

## Conclusion

- Pairing consecutive query rows in dense `indexer_score_v41` with CTA-level key-tile sharing and a single `[64, 128] x [64, 128]` block-scaled MMA speeds up dense indexer scoring by **1.17x–1.44x** across 16K–131K keys while remaining bit-for-bit identical.
- Splitting target-model decode indexing across TP ranks (`VLLM_DS41_DECODE_INDEX_SPLIT=1`) combined with 2-row key-tile reuse reduces full 8-layer decode indexer latency at 131K context by **0.42–0.44 ms/step** at 6 rows (1 stream), **1.88–2.05 ms/step** at 24 rows (4 streams), **3.84–4.20 ms/step** at 48 rows (8 streams), and **8.08–8.90 ms/step** at 96 rows (16 streams).
