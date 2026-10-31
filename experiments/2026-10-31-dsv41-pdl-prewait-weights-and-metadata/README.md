# DeepSeek-V4.1 TileLang PDL: Preload Static Weights, Step Metadata, and Sublayer-Entry State Before `pdl.chain()`

## Summary

The 43-layer IKET forward-pass trace (`table_43l.md` in PR #777) showed `43%`–`62%` `MEM_DEPENDENCY / LONG_SCOREBOARD` stalls in five memory-bound PDL-chained kernels that placed `pdl.chain()` (`cudaGridDependencySynchronize()`) at line 1 of the kernel body before loading static weights, step-level metadata (`positions`, `cos_sin`, `slots`), or sublayer-entry state (`residual`, `post`, `comb`, `mix`):

1. **`attention_front_kernel` (`vllm/models/deepseek_v4_1/tilelang/front.py`, `43` launches, `47.5%` `MEM_DEPENDENCY`):**
   - `q_weight`, `kv_weight`, `positions`, `cos_sin`, and `slots` are all independent of the predecessor GEMM (`fused_wqa_wkv`), which only writes `qa_kv`.
2. **`rope_kernel` (`vllm/models/deepseek_v4_1/tilelang/rope.py`, `43` launches, `43.2%` `MEM_DEPENDENCY`):**
   - `positions[row]` and `cos_sin[pos, ...]` are independent of `wq_b` (`x`), yet were loaded after `pdl.chain()`.
3. **`mxfp4_quantize_v41` & `mxfp4_norm_rope_quantize_v41` (`vllm/models/deepseek_v4_1/tilelang/indexer.py`, `63` launches, `54.8%` `MEM_DEPENDENCY`):**
   - `Slots`, `Weight`, `Positions`, and `CosSin` are independent of the predecessor projection (`wk`), which only writes `X`.
4. **`post_streams` & `post_collapse_streams` (`vllm/models/deepseek_v4_1/tilelang/mhc.py`, `2` launches, `61.7%` `MEM_DEPENDENCY`):**
   - `post`, `comb`, `mix`, and `residual` are produced at sublayer entry (`mhc_finalize`) before the attention/FFN sublayer runs, whereas the immediate predecessor (`wo_b` / `moe_combine`) only writes `x`.

## Changes

- **`vllm/models/deepseek_v4_1/tilelang/front.py` (`attention_front_kernel`):**
  - Preload `q_weight` (`for vi in T.unroll(q["num_thread_vec"])`) on Q CTAs and `kv_weight`, `positions`, `slots`, and `cos_sin` on KV CTAs into register arrays before `pdl.chain()`.
- **`vllm/models/deepseek_v4_1/tilelang/rope.py` (`rope_kernel`):**
  - Preload `pos = positions[row]` and `cs` (`cos_sin[pos, ...]`) into registers before `pdl.chain()`.
- **`vllm/models/deepseek_v4_1/tilelang/indexer.py` (`mxfp4_quantize_v41` and `mxfp4_norm_rope_quantize_v41`):**
  - Preload `Slots[r]` in `mxfp4_quantize_v41` (`paged=True`) and preload `Weight`, `Positions`, `Slots`, and `CosSin` into registers in `mxfp4_norm_rope_quantize_v41` before `pdl.chain()`.
- **`vllm/models/deepseek_v4_1/tilelang/mhc.py` (`post_streams` and `post_collapse_streams`):**
  - Preload `p` (`post`), `c` (`comb`), `m` (`mix`), and all 4 streams `r0..r3` (`residual`) into local registers before `pdl.chain()`, so only `xv` (`x`) waits on the sublayer output.
- **`tests/kernels/deepseek_v41_pdl_sources.py`:**
  - Register the predecessor-independent pointer parameters for `attention_front`, `rope`, `indexer_mxfp4_quantize`, `mhc_post`, and `mhc_post_collapse` in `EARLY_PARAMETERS`.

## Verification & Benchmark (`dgx4`, GB10 SM121, `spark-ds41f:19f2c20ed4d6-next-20`)

### Correctness & PDL Source Check
- `pytest -q -p tests.kernels.deepseek_v41_pdl_sources tests/kernels/test_deepseek_v41_tilelang_pdl.py tests/kernels/test_deepseek_v41_tilelang_front.py tests/kernels/test_deepseek_v41_tilelang_rope.py tests/kernels/test_deepseek_v41_tilelang_mhc.py tests/kernels/attention/test_deepseek_v41_tilelang.py`:
  - All tests pass with `0` PDL source violations.

### CUDA-Graph Replay Latency (`20` calls/graph, `200` replays)

| Kernel | Batch (`M`) | Baseline (`rc20`) | Patched | Delta |
|---|---:|---:|---:|---:|
| `attention_front` | `M = 1` | `2.392 µs` (`p10=2.376 µs`) | **`1.170 µs`** (`p10=1.144 µs`) | **`-51.1%`** |
| `attention_front` | `M = 6` | `2.411 µs` (`p10=2.397 µs`) | **`1.173 µs`** (`p10=1.157 µs`) | **`-51.3%`** |
| `attention_front` | `M = 14` | `2.510 µs` (`p10=2.474 µs`) | **`1.274 µs`** (`p10=1.251 µs`) | **`-49.2%`** |
| `rope_` (`16 × 512`) | `M = 1` | `0.830 µs` (`p10=0.819 µs`) | **`0.667 µs`** (`p10=0.651 µs`) | **`-19.6%`** |
| `rope_` (`16 × 512`) | `M = 6` | `0.856 µs` (`p10=0.818 µs`) | **`0.680 µs`** (`p10=0.666 µs`) | **`-20.6%`** |
| `rope_` (`16 × 512`) | `M = 14` | `0.867 µs` (`p10=0.824 µs`) | **`0.678 µs`** (`p10=0.670 µs`) | **`-21.8%`** |
| `norm_rope_write_index_keys` | `M = 1` | `2.101 µs` (`p10=2.086 µs`) | **`1.264 µs`** (`p10=1.242 µs`) | **`-39.8%`** |
| `norm_rope_write_index_keys` | `M = 6` | `2.104 µs` (`p10=2.053 µs`) | **`1.274 µs`** (`p10=1.266 µs`) | **`-39.4%`** |
| `norm_rope_write_index_keys` | `M = 14` | `2.194 µs` (`p10=2.141 µs`) | **`1.294 µs`** (`p10=1.283 µs`) | **`-41.0%`** |
| `mhc.post_streams` (`H=5120`) | `M = 1` | `1.059 µs` (`p10=1.021 µs`) | **`0.954 µs`** (`p10=0.917 µs`) | **`-9.9%`** |
| `mhc.post_streams` (`H=5120`) | `M = 6` | `1.146 µs` (`p10=1.101 µs`) | **`1.069 µs`** (`p10=1.040 µs`) | **`-6.7%`** |
| `mhc.post_streams` (`H=5120`) | `M = 14` | `1.586 µs` (`p10=1.541 µs`) | **`1.379 µs`** (`p10=1.357 µs`) | **`-13.1%`** |
| `mhc.post_collapse_streams` (`H=5120`) | `M = 1` | `0.973 µs` (`p10=0.958 µs`) | **`0.858 µs`** (`p10=0.819 µs`) | **`-11.8%`** |
| `mhc.post_collapse_streams` (`H=5120`) | `M = 6` | `1.077 µs` (`p10=1.067 µs`) | **`0.925 µs`** (`p10=0.888 µs`) | **`-14.1%`** |
| `mhc.post_collapse_streams` (`H=5120`) | `M = 14` | `1.486 µs` (`p10=1.443 µs`) | **`1.138 µs`** (`p10=1.101 µs`) | **`-23.4%`** |
