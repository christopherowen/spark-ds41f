# Column-Shard Target `fused_wqa_wkv` Across TP (`2026-10-31`)

## Summary

On the 60 target layers of DeepSeek-V4.1, `fused_wqa_wkv` (`[1280, 512]` = `1,792` output rows, `K = 5,120`, `9.18 MB` of block-32 FP8 weights + UE8M0 scales per layer) was previously replicated across all TP ranks (`disable_tp=True`), streaming `367 MB/rank` of duplicated weight bytes from DRAM across the 40 active target layers per decode step (`40.55 us/layer` cold on GB10).

This change column-shards target `fused_wqa_wkv` contiguously across TP (`disable_tp=self.is_draft, gather_output=True`) while keeping the 3 draft layers replicated (where DSpark's `_context_kv_projection` slices the trailing 512 `wkv` rows with `TileLangBlock32Rows` and the 3 draft layers run warm in L2 across 6 sequential draft steps):

1. **`_PaddedMergedColumnParallelLinear` (`vllm/models/deepseek_v4_1/attention.py`):**
   - Stacks `[wq_a (1280), wkv (512)]` into one contiguous `1,792`-row logical matrix, pads the total output rows to a multiple of `tp_size * 32` (`1,792` at TP4 = `448` rows/rank; `1,824` at TP3 = `608` rows/rank with zero weights and `0x7F` unit UE8M0 scales in the 32 trailing pad rows), and gives each TP rank a single contiguous row window.
   - `all_gather(dim=-1)` reconstructs the full `[M, 1792]` output in logical `[wq_a, wkv]` order without interleaving, returning a zero-copy `output[..., :1792]` view at TP3 (`stride = (1824, 1)`).
   - Shrinks `wqa_wkv` in `l2_prefetch`'s `BUDGET_NEXT` (`20 MB`) from `9.18 MB` to `2.29 MB` (TP4) / `3.11 MB` (TP3), allowing `wqa_wkv + wq_b + indexer.wq_b + indexer.weights_proj` (`18.35 MB` at TP4) to fit completely inside `BUDGET_NEXT`, and saves `412 MB/rank` (TP4) / `357 MB/rank` (TP3) of GPU memory across the 60 target layers.
2. **Strided `qa_kv` in `attention_front_kernel` (`vllm/models/deepseek_v4_1/tilelang/front.py`):**
   - Accepts `qa_kv: T.StridedTensor((M, width), (qa_stride, 1), T.bfloat16)` so TP3's sliced `[:, :1792]` view is consumed directly by `attention_front` without a copy.
3. **Decode Tile Configurations (`vllm/models/deepseek_v4_1/tilelang/gemm.py`):**
   - Adds `(448, 5120)` (TP4) and `(608, 5120)` (TP3) to `DECODE_FP8_CONFIGS` (`block_N=32, block_K=256, num_stages=4, threads=128` for `block_M = 16, 32, 64`) and `WIDE_DECODE` (`{96: 32, 128: 64}`).
4. **Sequence-Parallel Prefill (`vllm/models/deepseek_v4_1/attention.py`, `linear.py`, `kernels.py`):**
   - When `sp_rows is not None`, `self.compressor is None`, and `self.fused_wqa_wkv.tp_size > 1`, gathers the already-cast local `MXFP8Rows` (from `mhc.pre`) and local `indexer.weights_proj(hidden_states)` in a single grouped `sp_rows.gather_mxfp8` collective without gathering BF16 `hidden_states` or re-running `cast_mxfp8`.

## Benchmark & Verification on `dgx4` (`spark-ds41f:19f2c20ed4d6-next-20`)

### 40-Layer Cold Streaming Decode Benchmark (`M = 6, block_M = 16, K = 5120`)

| Configuration | Shape per Rank | Cold GEMM (`us/layer`) | Warm GEMM (`us/layer`) | Net vs Baseline (incl. `~11 us` `oneshot` AG) |
| :--- | :--- | ---: | ---: | ---: |
| Replicated Baseline | `(1792, 5120)` | `40.55 us` | `8.27 us` | — |
| **TP4 Sharded** | **`(448, 5120)`** | **`11.17 us`** (`-29.38 us`) | **`4.89 us`** | **`-18.4 us/layer` (`-0.74 ms / 40 layers`)** |
| **TP3 Sharded** | **`(608, 5120)`** | **`14.19 us`** (`-26.36 us`) | **`5.31 us`** | **`-15.4 us/layer` (`-0.62 ms / 40 layers`)** |

### Bit-Exactness & Test Suite

- 100% bit-identical (`torch.equal` on `uint16` bits) across `M = 1, 6, 24, 64, 256` and `block_M = 16, 32, 64` for both TP4 (`4 x 448`) and TP3 (`3 x 608` sliced to `1792`, including strided `attention_front`).
- `pytest -o addopts="" --noconftest tests/models/test_deepseek_v4_1_attention.py tests/models/test_deepseek_v4_1_sp_prefill.py tests/kernels/quantization/test_deepseek_v41_tilelang_linear.py` — all passed on `dgx4`.
