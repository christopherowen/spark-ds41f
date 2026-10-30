# 2026-10-30: Fuse Global KV Post-Projection Kernels (`_prepare_global_kv`)

## Summary
- Fuses the 3 post-`wk` index-key operations (`indexer.k_norm` + `rope_("index_key")` + `write_index_keys`) into a single TileLang kernel `mxfp4_norm_rope_quantize_v41` (`norm_rope_write_index_keys`).
- Fuses the 2 post-compressor main-KV operations (`rope_("latent")` + `write_cache(..., cache_kind="indexed")`) into `cache_writer_kernel(..., rope_dim=64, ratio=self.compress_ratio)`.
- Reduces post-projection global KV preparation from **5 kernel launches per KV-source layer to 2 launches** (saving **33 launches per step** across the 11 KV-source layers inside the `@eager_break_during_capture` segment).

## Verification on `dgx4` (`spark-ds41f:1030`)
- `pytest -q -p tblib_stub -p tests.kernels.deepseek_v41_pdl_sources tests/models/test_deepseek_v4_1_attention.py tests/kernels/test_deepseek_v41_tilelang_cache_writer.py tests/kernels/attention/test_deepseek_v41_tilelang.py`: `0` PDL violations (`45 kernels launched, 41 with PDL, 0 violations`), all tests pass.
- Bit-for-bit identical (`torch.equal`) across `rows = 1..4096` and `ratio ∈ {1, 2}` for both the index-K cache pages and the main indexed KV cache pages.
- Latency (`results.json`):
  - Decode (`rows=1..64`, `ratio=1`): `68.0–68.6 µs/layer` → `19.3–19.5 µs/layer` (**3.51×–3.55× faster**, saving **~49 µs/layer = ~535–542 µs/step** across 11 KV-source layers).
  - Decode (`rows=1..64`, `ratio=2`): `69.1–79.1 µs/layer` → `19.4–20.0 µs/layer` (**3.53×–3.95× faster**, saving **~50–59 µs/layer = ~545–650 µs/step** across 11 KV-source layers).
  - Prefill (`rows=2048..4096`): `70.4–75.7 µs/layer` → `20.0–25.3 µs/layer` (**2.98×–3.52× faster**, saving **~50–51 µs/layer = ~551–565 µs/chunk**).
