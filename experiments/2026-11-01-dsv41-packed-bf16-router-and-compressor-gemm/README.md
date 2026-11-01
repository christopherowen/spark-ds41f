# Exact 12-Bit Packed BF16 Router, Indexer, and Compressor GEMMs with SWAR `__byte_perm` Unpacking

## Summary
This experiment builds on top of:
- [PR #788](https://github.com/christopherowen/spark-ds41f/pull/788) (`exp/dsv41-bf16-gemm-partials-pdl`)
- [PR #789](https://github.com/christopherowen/spark-ds41f/pull/789) (`exp/dsv41-tilelang-packed-bf16-head-packer`)
- [PR #829](https://github.com/christopherowen/spark-ds41f/pull/829) (`exp/dsv41-moe-fused-route-dispatch-and-prewait`)

It extends exact 12-bit (`pack12`) BF16 weight compression (`1.5 bytes/element`, `-25%` DRAM footprint and read bandwidth) to DeepSeek-V4.1's unquantized BF16 projections:
1. **MoE Router (`ffn.gate`)**: `40 × [384, 5120]` (target layers) + `3 × [128, 5120]` (DSpark draft layers × 5 steps) = `174.1 MB -> 130.6 MB` (`-43.5 MB/step` of DRAM reads).
2. **Indexer Head Weights (`indexer.weights_proj`)**: `40 × [32, 5120]` (`TileLangScaledLinearMethod`, folded exact power-of-two `1/64` scale before packing) = `13.1 MB -> 9.8 MB` (`-3.3 MB/step`).
3. **Compressor Projections (`compressor.fused_wkv_wgate`)**: `3 × [1024, 5120]` + `1 × [512, 5120]` (`TileLangSplitLinearMethod`) = `36.7 MB -> 27.5 MB` (`-9.2 MB/step`).

Across a full target + 5-step DSpark speculative step, this eliminates **~56.0 MB/step** of DRAM weight traffic with **zero accuracy loss** (100% lossless reconstruction).

## Key Design Details
- **SWAR `__byte_perm` Unpacking into Swizzled Shared Memory**:
  - Added `vllm_packed_bf16_decode_vec16<BASE_MINUS_ONE>` in `vllm/models/deepseek_v4_1/tilelang/gemm.py`, which unpacks 8× `uint16` low-byte pairs (`low_a`, `low_b`) and 8× `uint16` packed sign+exponent nibbles (`codes`) in registers using SWAR bitmasks (`0x7070u`, `0x8080u`), `__byte_perm(lo, hi, 0x5140u / 0x7362u)`, and a single `0x00800080u` carry-free halfword offset addition.
  - Unpacked `bfloat16` vectors `out_a` and `out_b` are written through TileLang's `B_shared[j, c + v]` indexing so TileLang's shared-memory bank-conflict swizzle (`SwizzledLayout`) is preserved for `ldmatrix` + `mma.sync.aligned.m16n8k16`.
- **Zero-Overhead Post-Loop CSR Exception Accumulation**:
  - Rare out-of-window BF16 values (`exceptions`) are accumulated **after** each shard's `T.Pipelined` K-loop (`if start < end:` with `column // shard_cols == s`), keeping the inner K-loop 100% branch-free while preserving exact batch-invariance between decode split-K (`packed_bf16_gemm_partials` / `packed_bf16_gemm_splitk`) and prefill (`packed_bf16_gemm` / `packed_bf16_gemm_two`).
- **2 CTAs/SM Single-Wave Split-K Occupancy**:
  - Capping `pipe_stages = min(num_stages, 2 if block_N >= 64 else num_stages, shard_groups)` in `packed_bf16_gemm_splitk` keeps per-CTA shared memory at `48 KB` (`<= 49.5 KB`), allowing 2 CTAs/SM on GB10 (`96` concurrent CTAs across 48 SMs) so all 80 CTAs of `(1024, 5120)` (`16 tiles × 5 shards`) execute in a single wave.

## Benchmarks on `dgx4` (GB10 SM121)

### 1. MoE Router `packed_bf16_gemm_partials` (`[384, 5120]`, `shards=8`)
| Rows (`M`) | Unpacked `bf16_gemm_partials` Warm | `packed_bf16_gemm_partials` Warm | Warm Speedup | Unpacked `bf16_gemm_partials` Cold L2 | `packed_bf16_gemm_partials` Cold L2 | Cold L2 Speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | `4.70 us` | `4.10 us` | **`1.15x`** | `36.56 us` | `26.19 us` | **`1.40x`** |
| 6 | `4.69 us` | `4.12 us` | **`1.14x`** | `36.36 us` | `26.32 us` | **`1.38x`** |
| 16 | `4.70 us` | `4.11 us` | **`1.14x`** | `36.51 us` | `26.31 us` | **`1.39x`** |

### 2. Compressor `packed_bf16_gemm_splitk` (`[1024, 5120]`, `parts=2`, `shards=5`, `blocked=True`)
| Rows (`M`) | Unpacked `bf16_gemm_splitk` Cold L2 (`3x`) | `packed_bf16_gemm_splitk` Cold L2 (`3x`) | Cold L2 Speedup |
|---:|---:|---:|---:|
| 1 | `78.23 us` | `66.42 us` | **`1.18x`** |
| 6 | `77.78 us` | `65.91 us` | **`1.18x`** |
| 16 | `78.57 us` | `65.91 us` | **`1.19x`** |

## Verification
- `pytest tests/kernels/quantization/test_deepseek_v41_tilelang_linear.py tests/kernels/moe/test_deepseek_v41_tilelang_router.py tests/models/test_deepseek_v4_1_compressor.py tests/v1/worker/test_prepare_kernels.py` on `dgx4`: **153 passed, 0 PDL violations**.
