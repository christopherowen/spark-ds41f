# Defer V4.1 `WO` L2 Prefetch Until After Indexing and Order `NEXT` Segments Chronologically

## Summary

- **Defer `WO` L2 Prefetch to Immediately Before `_attend` (`vllm/models/deepseek_v4_1/attention.py`)**:
  - Previously, `l2_prefetch.issue(self._l2pf_wo, rows)` (`20 MB` of `wo_a` and `wo_b` into GB10's `24 MB` L2 cache) was issued right after `self.wq_b(qr)` — before `_prepare_global_kv`, before `_index_query` (`indexer.wq_b`, `indexer.weights_proj`), before `select_rows` (`tl_metadata.pages`, `_index_chunk`), and before `_write_decode_metadata`.
  - Issuing `WO` prefetch before indexing caused the `20 MB` fill to evict the `indexer.wq_b` and `indexer.weights_proj` weights that the previous layer's `NEXT` window had just prefetched into L2, and contended with the latency-bound sparse-MLA preparation and indexing kernels (as noted in `l2_prefetch.py`: *"WO alone is nearly neutral (49.12 ms) because the fill slows the latency-bound sparse-MLA preparation kernels it overlaps"*).
  - Moving `l2_prefetch.issue(self._l2pf_wo, rows)` into `forward_mqa` (and `_attend_split`) immediately before `self._attend` keeps `NEXT`'s prefetched weights hot in L2 through `_prepare_global_kv` and `_index_query`, avoids memory-bus contention during sparse-MLA preparation and indexing, and overlaps the `WO` fill with `self._attend` right before `_o_proj` consumes `wo_a` and `wo_b`.
- **Chronological `NEXT` Window Ordering (`vllm/models/deepseek_v4_1/nvidia/model.py`)**:
  - Adds `q_norm.weight` and `kv_norm.weight` between `wqa_wkv` and `wq_b` (matching `attention_front`), and includes `compressor` (`fused_wkv_wgate`, `norm`) and `indexer` (`wk`, `k_norm`) before `indexer.wq_b` and `indexer.weights_proj` on KV-source layers.

## Verification

- Verified on `dgx4` inside `spark-ds41f:1030`:
  - `tests/models/test_deepseek_v4_1_attention.py`: `7 passed` (including `test_wo_l2_prefetch_issues_after_indexing_and_metadata`).
