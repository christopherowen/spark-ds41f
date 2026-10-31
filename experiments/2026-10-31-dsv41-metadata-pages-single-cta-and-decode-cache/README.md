# DeepSeek-V4.1 TileLang Metadata: Single-CTA Group-Skipping `pages_kernel` and Cross-Layer Index Page-Table Cache

## Summary

The 43-layer IKET forward-pass trace (`table_43l.md` in PR #777) revealed that `metadata_pages_kernel` (`vllm/models/deepseek_v4_1/tilelang/metadata.py`) was launched **`42` times per forward pass** with **`448` CTAs per launch** (`rows = 14 × ceildiv(4096, 128) = 448` CTAs, `39.62 µs/launch` under IKET), resulting in `18,816` CTAs across the 43-layer pass for two reasons:

1. **Redundant `ceildiv(width, 128)` CTAs per row in `pages_kernel` (`vllm/models/deepseek_v4_1/tilelang/metadata.py`):**
   - `pages_kernel` launched `T.Kernel(rows, T.ceildiv(width, _BLOCK), threads=_BLOCK)` (`32` CTAs per row at `width = 4096`), where `31` of the `32` CTAs per row exited immediately whenever `live < 128`.
   - Meanwhile, `decode_metadata_kernel` (`part == 1`, lines 324–348 in the same file) already solved this exact problem by launching **1 CTA per row** (`T.Kernel(rows, threads=_BLOCK)`) with a 2-pass group-skipping loop (`groups = (width + _GROUP * _BLOCK - 1) // (_GROUP * _BLOCK)`, `if g * _GROUP * _BLOCK < live:`).
2. **Redundant `tl_metadata.pages` launches across layers sharing `indexer.k_cache` (`vllm/models/deepseek_v4_1/attention.py`):**
   - In DeepSeek-V4.1, `21` non-KV-source reindex layers share `self.indexer.k_cache` (`im`) with their `kv_source_layer_id`, so their `im.req_id_per_token`, `im.cache_lengths`, and `im.block_table` are identical.
   - However, `select_rows` in `forward_mqa` re-ran `tl_metadata.pages` on every single layer instead of caching the populated `index_pages[:count]` in `get_forward_context()` across single-chunk layers sharing `im` (just as `_ds41_decode_metadata` does for main/SWA decode metadata on lines 1260–1278).

## Changes

- **`vllm/models/deepseek_v4_1/tilelang/metadata.py` (`pages_kernel`):**
  - Replace the 2D `T.Kernel(rows, T.ceildiv(width, _BLOCK), threads=_BLOCK)` grid with a 1D `T.Kernel(rows, threads=_BLOCK)` grid (`1` CTA per row, reducing CTA count by `32×` at `width = 4096`) using the same 2-pass group-skipping loop as `decode_metadata_kernel`.
- **`vllm/models/deepseek_v4_1/attention.py` (`select_rows` in `forward_mqa`):**
  - When `split is None and index_rows <= chunk_rows`, cache `(im, index_pages[:count])` in `get_forward_context().__dict__.setdefault("_ds41_index_pages", {})` keyed by `(id(im), row, count, index_width, self._index_page)`.
  - Layers sharing `self.indexer.k_cache` reuse the source layer's persistent `index_pages[:count]` buffer directly, cutting `tl_metadata.pages` launches per forward pass in half (`42 -> 21`).

## Verification & Benchmark (`dgx4`, GB10 SM121, `spark-ds41f:19f2c20ed4d6-next-20`)

### Correctness & PDL Source Check
- `pytest -q -p tests.kernels.deepseek_v41_pdl_sources tests/kernels/test_deepseek_v41_tilelang_metadata.py tests/kernels/test_deepseek_v41_tilelang_pdl.py`:
  - All tests pass with `0` PDL source violations.

### CUDA-Graph Replay Latency (`20` calls/graph, `200` replays)

| Kernel / Path | Config | Baseline (`rc20`) | Patched | Delta |
|---|---|---:|---:|---:|
| `metadata.pages` (`width=4096, page=64`) | `rows = 14, live = 2` | `0.820 µs` (`448` CTAs) | **`0.751 µs` (`14` CTAs)** | **`-8.4%` (`32×` fewer CTAs)** |
| `select_rows` index page-table launches | `43`-layer forward pass (`42` index layers) | `42` launches (`18,816` CTAs) | **`21` launches (`294` CTAs)** | **`-50.0%` launches (`-98.4%` CTAs)** |
