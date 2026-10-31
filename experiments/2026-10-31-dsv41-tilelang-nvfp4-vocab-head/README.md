# Native TileLang Online NVFP4 Draft and Markov Vocabulary Head

## Motivation

In `rc21` (`5cfcc5813fc`), when `VLLM_DS41_DRAFT_NVFP4_HEAD=1` or `VLLM_DS41_MARKOV_NVFP4=1` is enabled, `Nvfp4OnlineLinearMethod` (`vllm/model_executor/layers/quantization/online/nvfp4.py`) and `_supports_default_lm_head_quantization()` (`vllm/model_executor/layers/vocab_parallel_embedding.py`) still hard-required `B12xNvFp4LinearKernel` (`b12x`). Furthermore, `b12x`'s NVFP4 linear kernel did not support `vocab_parallel_pairs_from_head` (fused per-tile greedy `(max, shard_index)` reduction without materializing full `[M, N]` logits to DRAM).

## Changes

1. **`vllm/models/deepseek_v4_1/tilelang/vocab.py`**:
   - Adds `nvfp4_vocab_projection` and `nvfp4_vocab_greedy_partials` (`@tilelang.jit`), plus `prepare_nvfp4_head(layer, out_dtype)`.
   - Streams packed E2M1 nibbles (`W[N, K // 2]`) and group-16 FP8 E4M3 block scales (`W_scale[N, K // 16]`) through software-pipelined shared memory, decodes each 16-column group into `B_shared` in BF16 (where the product of a 4-bit E2M1 value and an 8-bit E4M3 scale is mathematically exact in BF16), accumulates `A @ B^T` in FP32, and multiplies the FP32 accumulator by `W_global_scale[0]` once after the K loop.
   - Wires `project_head` and `project_head_greedy_pairs` (and thus `vocab_parallel_pairs_from_head` and `LogitsProcessor._apply_head`) to support prepared NVFP4 heads across all `ROW_TILES = (16, 32, 64, 128)`.
2. **`vllm/model_executor/layers/quantization/online/nvfp4.py` & `vllm/model_executor/layers/vocab_parallel_embedding.py`**:
   - Updates `Nvfp4OnlineLinearMethod` and `_supports_default_lm_head_quantization` to use the native TileLang NVFP4 head kernels when `linear_backend == "tilelang"`, without requiring `B12xNvFp4LinearKernel`.
   - Adds `Nvfp4OnlineLinearMethod.declare_layer_kernels` so `prepare.declare_modules` compiles and binds the NVFP4 head's TileLang kernels during worker startup.
3. **`tests/kernels/quantization/test_deepseek_v41_tilelang_vocab.py`**:
   - Adds `test_nvfp4_head_matches_dequantized_reference_and_is_batch_invariant`, `test_nvfp4_head_greedy_pairs_match_greedy_shard_pair`, and `test_nvfp4_online_linear_method_tilelang_without_b12x`.

## Verification

Verified on `dgx4` (GB10 SM121) inside `spark-ds41f:19f2c20ed4d6-next-21`:
- `pytest --noconftest -v -p tests.kernels.deepseek_v41_pdl_sources -k "nvfp4 or bf16_head or logits_processor" tests/kernels/quantization/test_deepseek_v41_tilelang_vocab.py`: **11 passed, 0 PDL violations**.
