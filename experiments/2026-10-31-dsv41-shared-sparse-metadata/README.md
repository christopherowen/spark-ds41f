# Share V4.1 Single-Chunk Index Page Tables and Cache-Group Attention Metadata

## Summary

- **Single-Chunk Index Page Table Sharing (`_ds41_index_pages` in `vllm/models/deepseek_v4_1/attention.py`)**:
  - Out of 19 indexer layers (`is_index_source`), only 4 are `is_kv_source`; the remaining 15 reuse their KV-source layer's `indexer.k_cache` (`im`).
  - Whenever `index_rows <= chunk_rows` (which holds for all decode steps and all single-chunk prefill steps up to `256` / `1,024` tokens) on a non-split span, the first indexer layer of each KV group populates its persistent `index_pages` buffer via `tl_metadata.pages` and records `(im, index_pages)` in `get_forward_context()._ds41_index_pages`. Downstream indexer layers sharing the same `im` read that populated buffer directly without launching `tl_metadata.pages`, eliminating **15 redundant `tl_metadata.pages` kernel launches per step** with zero additional memory allocation.
- **Short-Extend (`rows <= _decode_capacity`) and Decode Metadata Sharing (`_ds41_decode_metadata`)**:
  - Shares `(swa_indices, swa_lengths, top_lengths[, indexed_page_table])` across layers in the same cache group on both `decode` steps and decode-sized `extend` steps (`rows <= self._decode_capacity`, e.g., TTFT probes and short interactive turns), eliminating **56 redundant `_write_decode_metadata` kernel launches** on short-prompt prefills.
  - Stops borrowing unused `*metadata_specs` from `current_workspace_manager().get_simultaneous(...)` when shared metadata buffers are used.
- **Mixed-Step (`_attend_split`) Cache-Group Metadata Sharing (`_ds41_split_metadata`)**:
  - Shares the active contiguous cache group's per-row metadata across layers in `_attend_split` and reuses the allocated tensors across groups with matching `extend_specs`, reducing `_write_decode_metadata` launches per mixed step from **61** to **5** (saving **56 kernel launches** and **59 group allocations**).

## Verification

- Verified on `dgx4` inside `spark-ds41f:1030`:
  - `tests/models/test_deepseek_v4_1_attention.py`: `7 passed` (including `test_forward_mqa_shares_index_pages_and_group_metadata`).
  - `tests/kernels/test_deepseek_v41_tilelang_metadata.py`: `137 passed`.
