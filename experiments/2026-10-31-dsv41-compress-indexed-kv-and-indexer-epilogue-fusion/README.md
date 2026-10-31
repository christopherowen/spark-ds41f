# DeepSeek-V4.1: Fused Compressor Indexed KV Write, Indexer `wk` Split-K + `wq_b` RoPE/MXFP4 Epilogues, and Index Metadata Dedup

## Summary
Implements **Part 2 Item 9** of the TileLang kernel-fusion roadmap on `rc21` (`a89da720d`) while carrying forward the best optimizations from **PR #808** (Compressor & Indexer PDL overlap) and **PR #809** (single-CTA group-skipping `pages_kernel` + cross-layer `_ds41_index_pages` deduplication):

1. **Fused `compress_kernel` + `write_cache("indexed")` Epilogue (`vllm/models/deepseek_v4_1/tilelang/compress.py`, `compressor.py`, `attention.py`)**:
   - In `compress_kernel(..., write_kv=True, kv_page_size=..., rope_dim=64)`, each warp processes one token across 4 chunks `c in 0..3` where `col = (c * 32 + lane) * 4`. Four consecutive lanes (`lane & ~3 .. (lane & ~3) + 3`) own the 16 contiguous columns of each 16-value indexed KV-cache group `group = c * 8 + lane // 4`, and in chunk `c == 3`, lanes `16..31` (`col >= 448`) own the trailing 64 RoPE columns.
   - `compress_kernel` stores the pre-RoPE normalized `out[token, col + i]` (consumed by `indexer.wk`), then rotates `y[0..3]` in registers for `c == 3, col >= 448` using `token_positions[token] // ratio`, reduces `amax` across 4 lanes via `shfl_xor(1)` and `shfl_xor(2)`, and packs + stores the 288-byte `indexed` KV-cache record (`dsv41_e2m1x4` + E4M3 scale) in the same launch.
   - Eliminates **21 `cache_writer("indexed")` launches per step**.

2. **Fused `indexer.wk` Split-K Partials Reduction in `mxfp4_norm_rope_quantize_v41` (`vllm/models/deepseek_v4_1/tilelang/indexer.py`, `linear.py`, `attention.py`, `tilelang/attention.py`)**:
   - `indexer.wk` (`[128, 512]` BF16) uses `bf16_shards(128, 512) = 4`. At decode sizes (`rows <= 64`), `project_bf16_partials(indexer.wk, latent)` returns the 3D `[4, rows, 128]` FP32 partials tensor, and `mxfp4_norm_rope_quantize_v41(..., shards=4)` sums `P[0] + P[1] + P[2] + P[3]` in FP32 in shard order and rounds to BF16 in registers before `k_norm + RoPE + paged MXFP4 write`.
   - Eliminates **21 `splitk_reduce` launches per step** and the intermediate `[rows, 128]` BF16 key buffer.

3. **Fused `indexer.wq_b` Decode GEMM + RoPE + MXFP4 Query Quantization (`vllm/models/deepseek_v4_1/tilelang/gemm.py`, `linear.py`, `attention.py`)**:
   - Adds `mxfp8_gemm_rope_mxfp4_decode` (`project_block32_rope_mxfp4`), which stages each decode output tile `C_local` through `C_shared` (`[block_M, block_N]` BF16), applies GPT-J interleaved RoPE to the trailing `rope_dim=64` columns of each 128-column index head, computes the UE8M0 scale (`pow2_ceil(amax * fp32(1/6))`), and quantizes + packs 32 E2M1 codes (`dsv41_e2m1x2_rn`) into `Q` (`[rows, 32, 64]` uint8) and `S` (`[rows, 32, 4]` uint8) in the same launch.
   - Eliminates **42 `mxfp4_quantize_v41` launches per step** and the intermediate `[rows, 4096]` BF16 `iq` allocation.

4. **Compressor & Indexer Pre-Wait Weight/Metadata Preload + Single-CTA `pages_kernel` & Cross-Layer Reuse (Best of PRs #808 & #809)**:
   - Preloads static norm weights (`weight` / `Weight`, `CosSin`) and host-prepared step metadata (`counts`, `starts`, `positions`, `table`, `slots`, `state_slots`, `token_positions`) before `pdl.wait_for_dependents()` in `gather_partial_state_kernel`, `compress_kernel`, and `mxfp4_norm_rope_quantize_v41`.
   - Replaces the multi-CTA `pages_kernel` (`grid=(ceildiv(width, 128), rows)`) with a single-CTA per-row loop (`grid=(rows,)`) that skips unallocated page groups past `used`, and caches `_ds41_index_pages` across layers sharing the same KV-cache group (`21` launches saved per step).
   - **Total launch reduction**: **105 fewer kernel launches per step** (`21 + 21 + 42 + 21`).

## Verification on `dgx4` (`spark-ds41f:19f2c20ed4d6-next-21`)
- `pytest --noconftest -p tests.kernels.deepseek_v41_pdl_sources`:
  - `tests/kernels/test_deepseek_v41_tilelang_compress.py`
  - `tests/kernels/attention/test_deepseek_v41_tilelang.py` (including `compute-sanitizer --tool initcheck`)
  - `tests/kernels/test_deepseek_v41_tilelang_metadata.py`
  - `tests/models/test_deepseek_v4_1_attention.py`
  - `tests/models/test_deepseek_v4_1_compressor.py`
  - `tests/models/test_deepseek_v4_1_compressor_prepare.py`
  - `tests/models/test_deepseek_v4_1_prepare.py`
- **Result**: `245 passed, 1 skipped, 1 xfailed` — `PDL sources: 130 kernels launched, 119 with PDL, 0 violations`.
