# Native TileLang 12-bit Packed BF16 LM Head Weight Packer

## Motivation

In `rc21` (`5cfcc5813fc`), the runtime vocabulary projection for the 12-bit packed BF16 LM head (`packed_vocab_projection` and `packed_vocab_greedy_partials`) runs in TileLang (`vllm/models/deepseek_v4_1/tilelang/vocab.py`), but `PackedBf16LmHeadMethod.process_weights_after_loading()` and `packed_bf16_supported()` in `vllm/model_executor/layers/quantization/online/packed_bf16.py` still required `b12x.gemm.packed_bf16_vocab_projection.pack` (which runs three Triton kernels in `b12x`) to pack the checkpoint's BF16 LM head at model load time.

## Changes

1. **`vllm/models/deepseek_v4_1/tilelang/vocab.py`**:
   - Adds `PackedBf16Weight`, `pack_bf16_weight(weight, *, in_place=False)` (aliased as `pack`), and `unpack_bf16_weight(packed)` (aliased as `unpack`).
   - Implements the three GPU passes natively with `@tilelang.jit`:
     - `packed_bf16_histogram`: per-tensor 256-bin exponent histogram using shared-memory atomics to select the optimal 15-exponent window (`exponent_base`).
     - `packed_bf16_exception_count`: per-row count of out-of-window nonzero exponents to form `row_offsets`.
     - `packed_bf16_pack_rows`: packs sign-and-mantissa bytes and 4-bit exponent codes across 128-column groups, compacts out-of-window exceptions in exact ascending column order via warp `__ballot_sync` + `__popc`, and verifies bit-exact round-trip on every weight. Supports `in_place=True` using a 1,024-row staging buffer.
2. **`vllm/model_executor/layers/quantization/online/packed_bf16.py`**:
   - Updates `packed_bf16_supported()`, `PackedBf16LmHeadMethod.process_weights_after_loading()`, and `PackedBf16LmHeadMethod.packed_weight()` to use `tilelang/vocab.py` directly when `linear_backend == "tilelang"` (or when `b12x` is absent).
3. **`tests/kernels/quantization/test_deepseek_v41_tilelang_vocab.py`**:
   - Removes the module-level `pytest.importorskip("b12x.gemm.packed_bf16_vocab_projection")` dependency.
   - Adds `test_tilelang_pack_bf16_weight_round_trips_and_matches_b12x` (verifying bit-for-bit identical `values`, `exponent_base`, `row_offsets`, `exception_rows`, and `exception_entries` against `b12x` for both `in_place=False` and `in_place=True`) and `test_packed_bf16_lm_head_method_tilelang_without_b12x`.

## Verification

Verified on `dgx4` (GB10 SM121) inside `spark-ds41f:19f2c20ed4d6-next-21`:
- `pytest --noconftest -v -p tests.kernels.deepseek_v41_pdl_sources tests/kernels/quantization/test_deepseek_v41_tilelang_vocab.py`: **40 passed, 0 PDL violations**.
