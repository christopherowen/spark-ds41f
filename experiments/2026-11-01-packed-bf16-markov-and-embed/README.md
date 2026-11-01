# Exact 12-bit Packed BF16 DSpark Markov W2 Projection & Token Embedding Lookup

## Parent PRs
Builds directly on top of:
- **PR #781** (`exp/dsv41-tilelang-embedding`): Native TileLang `VocabParallelEmbedding` lookup kernel.
- **PR #789** (`exp/dsv41-tilelang-packed-bf16-head-packer`): Native TileLang 12-bit `pack_bf16_weight` packer with in-place support (`in_place=True`).
- **PR #831** (`exp/dsv41-dspark-context-and-markov-greedy-fusion`): Fused DSpark Markov greedy pairs (`project_head_markov_greedy_pairs`) and SWAR `__byte_perm` `vllm_vocab_decode_vec16` unpacking.

## Summary
1. **Exact 12-bit Packed BF16 DSpark Markov Projection (`markov_w2`, `[32320, 256]` at TP4)**:
   - Packs `markov_head.markov_w2.weight` using `pack_bf16_weight` (`lm_head_quantization="packed_bf16"` when `not envs.VLLM_DS41_MARKOV_NVFP4` and `linear_backend == "tilelang"`), cutting per-rank weight bytes by **25%** (`16.55 MB` -> `12.41 MB` per step, saving **`-20.7 MB/step`** across the 5 sequential DSpark draft steps).
   - Adds `packed_vocab_markov_greedy_partials` and `_packed_head_markov_greedy_kernels` in `vllm/models/deepseek_v4_1/tilelang/vocab.py`, fusing SWAR `__byte_perm` 12-bit unpacking, out-of-window exception addition, `FP32`/`BF16` base-logit addition, and per-tile top-1 reduction in a single CTA epilogue without materializing `[M, 32320]` logits to DRAM.
   - Specializes `tile_config(block_M, k)` for narrow `K <= 256` heads (`block_N=32, threads=128, step_groups=1, num_stages=1`), halving per-CTA shared memory so 5 CTAs/SM overlap memory loads with tensor-core MMAs.
   - Supports packed row gathering (`vocab_embedding_packed`) in `apply_markov_bias_gathered` when `markov_w2` is packed.
2. **Exact 12-bit Packed BF16 Token Embedding Table (`embed_tokens.weight`, `[32320, 5120]` at TP4)**:
   - Packs `VocabParallelEmbedding` weights in place (`pack_bf16_weight(data, in_place=True)`) in `TileLangEmbeddingMethod.process_weights_after_loading`, preserving GB10 display carve-out buffer ownership (`confirm_streamed_weights`) while reducing the embedding footprint by 25% (`330.96 MB` -> `248.22 MB` per rank).
   - Adds `packed_vocab_embedding_kernel` (`vllm_embed_decode_vec16` + `vllm_packed_vocab_embedding_row` with binary-searched out-of-window exception restoration) in `vllm/models/deepseek_v4_1/tilelang/embedding.py`, reproducing unpacked BF16 embedding rows bit-for-bit across TP1, TP3, and TP4 (`all_reduce` and `reduce_scatter`).

## Verification & Benchmarks (`dgx4` GB10 SM121)
- **Pytest Suite**: `135 passed` across `test_deepseek_v41_tilelang_embedding.py`, `test_deepseek_v41_tilelang_vocab.py`, `test_dspark_v41_gathered_markov.py`, `test_dspark_greedy_markov.py`, `test_deepseek_v41_tilelang_columns.py`, `test_deepseek_v41_tilelang_dspark_context.py`, and `test_prepare_kernels.py`.
- **5-Step DSpark Markov Greedy Projection (`N=32320, K=256`, CUDA Graph)**:
  - `M= 1`: `BF16` warm `87.77 us` / cold-step0 `153.63 us` -> **`pack12` warm `81.82 us` (`1.07x`) / cold-step0 `129.95 us` (`1.18x`, `-23.68 us`)**
  - `M= 8`: `BF16` warm `105.31 us` / cold-step0 `174.82 us` -> **`pack12` warm `94.54 us` (`1.11x`) / cold-step0 `144.16 us` (`1.21x`, `-30.66 us`)**
  - `M=16`: `BF16` warm `137.89 us` / cold-step0 `200.19 us` -> **`pack12` warm `110.28 us` (`1.25x`) / cold-step0 `159.73 us` (`1.25x`, `-40.46 us`)**
  - `M=32`: `BF16` warm `184.02 us` / cold-step0 `245.56 us` -> **`pack12` warm `161.65 us` (`1.14x`) / cold-step0 `210.75 us` (`1.17x`, `-34.81 us`)**
- **Packed `VocabParallelEmbedding` Lookup (`N=32320, K=5120`)**:
  - `tokens=1..64`: `3.06 - 4.10 us` (`100%` bit-for-bit identical to unpacked BF16).
