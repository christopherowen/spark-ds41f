# Run Fused V4.1 `attention_front` Across Prefill & Wide-Decode Row Counts

## Summary

- **`attention_front` Across All Non-Breakable-Graph Row Counts (`vllm/models/deepseek_v4_1/attention.py`)**:
  - `attention_front` (`vllm/models/deepseek_v4_1/tilelang/front.py`) fuses five post-`fused_wqa_wkv` kernels into a single launch with bit-for-bit identical outputs (`q_norm`, `kv_norm`, `per_token_cast` of `qr` into `MXFP8Rows`, `rope_` on `kv`, and `write_cache` into the SWA KV cache).
  - Previously, `DeepseekV4Attention._front` artificially bailed out when `sp_rows is not None` or `rows > DECODE_ROWS` (`64`), forcing wide-decode steps (`rows > 64`), all 41 CED decoder layers during prefill, the 4 compressor layers after `sp_rows.gather(hidden_states)`, and non-SP prefill steps to fall back to the 5 separate kernel launches.
  - Because `_front` is only called on the full-row path of `_forward` (after `sp_rows.gather(hidden_states)` on compressor layers or when `sp_rows is None`), removing `sp_rows is not None or rows > DECODE_ROWS` from `_front` replaces 5 kernel launches per layer with 1 fused launch across all those row counts.

## Benchmark on `dgx4` (`spark-ds41f:1030`)

Median latency of `attention_front` vs. the 5 separate kernels (`q_norm`, `kv_norm`, `per_token_cast`, `rope_`, `write_cache`):

| `rows` | `attention_front` (`us`) | 5 Separate Kernels (`us`) | Speedup |
|---:|---:|---:|---:|
| `6` | `30.30` | `259.92` | `8.58x` |
| `48` | `7.79` | `70.40` | `9.04x` |
| `64` | `7.75` | `70.71` | `9.12x` |
| `96` | `7.79` | `70.20` | `9.01x` |
| `128` | `7.77` | `70.45` | `9.07x` |
| `256` | `7.92` | `70.53` | `8.91x` |
| `512` | `7.77` | `71.00` | `9.13x` |
| `1024` | `11.19` | `70.83` | `6.33x` |
| `2048` | `45.05` | `70.96` | `1.58x` |
| `4096` | `159.43` | `174.40` | `1.09x` |

## Verification

- Verified on `dgx4` inside `spark-ds41f:1030` with `-p tests.kernels.deepseek_v41_pdl_sources`:
  - `tests/kernels/test_deepseek_v41_tilelang_front.py` (including `rows=512`) and `tests/models/test_deepseek_v4_1_attention.py`: `30 passed`, `PDL sources: 8 kernels launched, 8 with PDL, 0 violations`.
