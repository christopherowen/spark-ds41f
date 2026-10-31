# Drop Unnecessary `@torch.library.custom_op` Wrappers & Compiler Indirection in DeepSeek-V4.1

## Summary

After `torch.compile` was retired for DeepSeek-V4.1 (`vLLM 0493-0496`), 20 operations across `vllm/models/deepseek_v4_1/` remained wrapped in `@torch.library.custom_op(...)` + `.register_fake` along with `torch.compiler.is_compiling()` checks, integer-key `_LAYERS` `WeakValueDictionary` indirection (to work around `custom_op` schema rules forbidding `nn.Module` arguments), `ForwardContext.no_compile_layers[self.prefix]` lookups, and a dummy 1-byte GPU tensor allocation (`global_kv_ready`) per CED boundary to order `torch.compile` functionalization.

Each `@torch.library.custom_op` call routes arguments through `c10::Dispatcher` C++ boxing/unboxing and schema validation (~`8–13 µs` of pure CPU overhead per call on `GB10`), plus extra `.reshape()` / `.view()` tensor wrapper allocations on 2D activations.

## Changes (`+54 / -340` lines across 19 files)

1. **`vllm/models/deepseek_v4_1/tilelang/linear.py`**:
   - Removed `@torch.library.custom_op` and `.register_fake` from `_block32_linear`, `_bf16_linear`, `_split_linear_one`, and `_split_linear_two`.
   - Removed `_LAYERS: WeakValueDictionary[int, nn.Module]` and `layer.tilelang_key = id(layer)` indirection; pass `layer` directly to `_block32_linear`, `_bf16_linear`, and `_split_bf16`.
   - Collapsed `_split_linear_one` / `_split_linear_two` into direct `_split_bf16` calls in `project_parts`.
   - Skipped redundant `.reshape(-1, k)` and `.view(rows, n)` wrapper allocations when `x.ndim == 2`.
   - Replaced `None if torch.compiler.is_compiling() else current_preallocated_workspace()` with `current_preallocated_workspace()`.
2. **`vllm/models/deepseek_v4_1/tilelang/norm.py`**:
   - Removed `@torch.library.custom_op("vllm::dsv41_tilelang_rmsnorm")` and `.register_fake`.
   - Added a 2D fast path in `TileLangRMSNorm.forward` that calls `pdl.norm_forward` directly without creating `.reshape()` / `.view()` tensor wrappers.
3. **`vllm/models/deepseek_v4_1/tilelang/mhc.py`**:
   - Removed `@torch.library.custom_op` and `.register_fake` from `_mhc_pre`, `_mhc_post`, `_mhc_collapse`, and `_mhc_collapse_column`.
4. **`vllm/models/deepseek_v4_1/tilelang/rope.py`, `embedding.py`, `engram.py`, `engram_hash.py`, `engram_rows.py`, `confidence.py`, `glp.py`, `wo.py`, `ced.py`**:
   - Removed `@torch.library.custom_op` and `.register_fake` from `rope_`, `vocab_embedding`, `engram_gate`, `engram_gate_blocked`, `engram_hash`, `engram_rows_gather`, `dspark_confidence`, `glp_steer`, `gather_rows`, and `scatter_rows`.
   - Replaced `torch.compiler.is_compiling()` checks in `wo.py`, `common/engram.py`, and `nvidia/model.py`.
5. **`vllm/models/deepseek_v4_1/attention.py` & `nvidia/model.py`**:
   - Removed `@torch.library.custom_op("vllm::dsv41_b12x_attention")` and `@torch.library.custom_op("vllm::dsv41_b12x_prepare_global_kv")` and their `.register_fake` registrations.
   - `DeepseekV4Attention.forward` and `prepare_global_kv` now invoke `self._forward` and `self._prepare_global_kv` directly without `get_forward_context().no_compile_layers[self.prefix]` lookups or dummy 1-byte `torch.empty(1, dtype=torch.uint8, device=hidden.device)` `global_kv_ready` allocations.

## `GB10` (`dgx4`) Eager Launch Overhead Benchmark (`6 x 2048` BF16)

| Operation | Before (`custom_op`) | After (Direct Call) | Delta | Speedup |
| :--- | ---: | ---: | ---: | ---: |
| `TileLangRMSNorm.forward` | `28.85 µs` | **`16.33 µs`** | `-12.52 µs` | **1.77x** |
| `_mhc_post` | `16.04 µs` | **`8.10 µs`** | `-7.94 µs` | **1.98x** |
| `rope_` | `15.05 µs` | **`5.55 µs`** | `-9.50 µs` | **2.71x** |

## Verification on `dgx4` (`spark-ds41f:1030`)

- **654 passed, 1 skipped** across all DeepSeek-V4.1 model and kernel test suites:
  - `tests/models/test_deepseek_v4_1_attention.py`
  - `tests/models/test_deepseek_v4_1_ced_model.py`
  - `tests/models/test_deepseek_v4_1_sp_prefill.py`
  - `tests/models/test_deepseek_v4_1_glp.py`
  - `tests/models/test_deepseek_v4_1_engram.py`
  - `tests/kernels/test_deepseek_v41_tilelang_rope.py`
  - `tests/kernels/test_deepseek_v41_tilelang_embedding.py`
  - `tests/kernels/test_deepseek_v41_tilelang_confidence.py`
  - `tests/kernels/test_deepseek_v41_tilelang_glp.py`
  - `tests/kernels/test_deepseek_v41_tilelang_engram.py`
  - `tests/kernels/test_deepseek_v41_tilelang_engram_hash.py`
  - `tests/kernels/test_deepseek_v41_tilelang_ced.py`
  - `tests/kernels/test_deepseek_v41_tilelang_mhc.py`
  - `tests/kernels/test_deepseek_v41_tilelang_compress.py`
  - `tests/kernels/test_deepseek_v41_tilelang_front.py`
  - `tests/kernels/test_deepseek_v41_engram_rows.py`
