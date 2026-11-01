# Lossless TP Column-Sharding of `compressor.fused_wkv_wgate` with Single All-Gather

## Summary
This experiment builds on top of:
- [PR #767](https://github.com/christopherowen/spark-ds41f/pull/767) (`exp/shard-target-wqa-wkv`)
- [PR #832](https://github.com/christopherowen/spark-ds41f/pull/832) (`exp/dsv41-compress-indexed-kv-and-indexer-epilogue-fusion`)

Previously, `DeepseekCompressor.fused_wkv_wgate` (`3 × [1024, 5120]` BF16 for ratio-2 layers 2, 16, 30 and `1 × [512, 5120]` BF16 for ratio-1 layer 20) was constructed with `disable_tp=True`, replicating `36.7 MB` of BF16 weights on every TP rank. In owner layers, `_l2pf_comp` prefetched the full `10.49 MB` ratio-2 weight during `Q-B`, occupying **43% of GB10's 24 MiB L2 cache**.

This PR enables lossless TP column-sharding of `compressor.fused_wkv_wgate` across TP ranks (`disable_tp = tp_size <= 1 or 512 % (tp_size * 64) != 0`):
- At TP4, each rank holds `1/4` of `wkv` (`[128, 5120]`) and, at ratio 2, `1/4` of `wgate` (`[128, 5120]`), cutting per-rank compressor weight bytes by **75%** (`36.7 MB -> 9.18 MB`, saving **`-27.5 MB/step`** of DRAM weight reads) and shrinking `_l2pf_comp` from **`10.49 MB` to `2.62 MB`** (`-7.86 MB` of L2 pressure during `Q-B`).
- Each rank projects its local shard in a single `TileLangSplitLinearMethod` launch into a contiguous preallocated `local_shards` view (`(compress_ratio, rows, 512 // tp_size)`) with the exact same `shards=5, blocked=True` K-reduction order as TP1, and reconstructs `_values[:rows]` (and `_gates[:rows]` at ratio 2) via a **single** `tensor_model_parallel_all_gather(local, dim=-1)` call (`4 KB` per token at ratio 2, `1 KB` per token at ratio 1), producing **100% bit-for-bit identical** outputs to TP1.

## Benchmarks on `dgx4` (GB10 SM121)

### 1. Ratio-2 Compressor Projection (`[1024, 5120]` TP1 vs `[256, 5120]` TP4, `parts=2, shards=5, blocked=True`)
| Rows (`M`) | TP1 (`1024×5120`) Warm | TP4 (`256×5120`) Warm | Warm Speedup | TP1 (`1024×5120`) Cold L2 | TP4 (`256×5120`) Cold L2 | Cold L2 Speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | `7.95 us` | `4.42 us` | **`1.80x`** | `105.13 us` | `20.81 us` | **`5.05x` (`-84.3 us`)** |
| 6 | `8.70 us` | `4.28 us` | **`2.03x`** | `106.12 us` | `22.07 us` | **`4.81x` (`-84.1 us`)** |
| 16 | `10.58 us` | `7.50 us` | **`1.41x`** | `109.58 us` | `28.68 us` | **`3.82x` (`-80.9 us`)** |

### 2. Ratio-1 Compressor Projection (`[512, 5120]` TP1 vs `[128, 5120]` TP4, `parts=1, shards=5, blocked=True`)
| Rows (`M`) | TP1 (`512×5120`) Warm | TP4 (`128×5120`) Warm | Warm Speedup | TP1 (`512×5120`) Cold L2 | TP4 (`128×5120`) Cold L2 | Cold L2 Speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | `4.95 us` | `4.14 us` | **`1.20x`** | `56.83 us` | `18.17 us` | **`3.13x` (`-38.7 us`)** |
| 6 | `5.57 us` | `4.09 us` | **`1.36x`** | `57.42 us` | `19.16 us` | **`3.00x` (`-38.3 us`)** |
| 16 | `6.89 us` | `4.13 us` | **`1.67x`** | `58.96 us` | `20.72 us` | **`2.85x` (`-38.2 us`)** |

## Verification
- `pytest tests/models/test_deepseek_v4_1_compressor.py tests/models/test_deepseek_v4_1_compressor_prepare.py tests/models/test_deepseek_v4_1_attention.py tests/models/test_deepseek_v4_1_l2_schedule.py tests/kernels/test_deepseek_v41_tilelang_compress.py` on `dgx4`: **59 passed, 0 PDL violations**.
