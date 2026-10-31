# DeepSeek-V4.1 TP3 Attention Subgroup Trimming and Decode Tile Configs

## Summary

At `TP=3` (`tp3-triangle`), DeepSeek-V4.1 pads its `8` output groups (`64` attention heads) to `9` output groups (`72` attention heads) so each rank initially allocates `3` local groups (`24` heads). However:

1. **Rank 2's 3rd `o_group` (`heads 16..23`, global heads `64..71`) is 100% zero-padded** in `wq_b`, `wo_a`, and `wo_b` (`allow_tp_padding=True`), contributing exact `+0.0` to `wo_b`'s output while forcing Rank 2 to:
   - Execute `wq_b` on `24` heads (`[12288, 1280]`, `15.73 MB`) instead of `16` real heads (`[8192, 1280]`, `10.49 MB`);
   - Execute `SparseMLA` on `24` heads (`2` head blocks, falling off the 1-wave `"split"` threshold `split_max_rows = 4` onto the 256-thread `"decode"` kernel at `c1` `rows = 6`) instead of `16` real heads (`1` head block, `"split"` up to `9` rows);
   - Execute `wo_a` + `wo_b` on `3` groups (`12.58 + 15.73 = 28.31 MB`, exceeding the `20 MB` `_l2pf_wo` prefetch window by `8.31 MB`) instead of `2` real groups (`8.39 + 10.49 = 18.88 MB`, fitting 100% inside `_l2pf_wo` and L2).
2. **Missing TP3 shapes in `TILE_CONTIGUOUS_FP8`, `DECODE_FP8_CONFIGS`, and `WIDE_DECODE`:**
   - `TILE_CONTIGUOUS_FP8`, `DECODE_FP8_CONFIGS`, and `WIDE_DECODE` in `vllm/models/deepseek_v4_1/tilelang/gemm.py` only listed TP4 shapes (`(8192, 1280)`, `(1024, 4096, 2)`, `(5120, 2048)`, `(1152, 5120)`, `(5120, 576)`), omitting the TP3 shapes on Ranks 0 & 1 (`(12288, 1280)` `Q-B`, `(1024, 4096, 3)` `WO-A`, `(5120, 3072)` `WO-B`, `(1536, 5120)` / `(5120, 768)` shared expert, and `(1728, 15360)` DSpark `main_proj`).

## Changes

- **`vllm/models/deepseek_v4_1/attention.py` (`DeepseekV4Attention._trim_padded_groups` / `setup_wo_projection`):**
  - After weight loading (`setup_wo_projection`, before `declare_modules` / `_build_l2_prefetch`), trim any trailing zero-padded local output groups on the last TP rank (`0 < real_groups < self.n_local_groups`, i.e. `3 -> 2` groups and `24 -> 16` heads on TP3 Rank 2), slicing `wq_b`, `attn_sink`, `wo_a`, and `wo_b` in place.
- **`vllm/models/deepseek_v4_1/tilelang/gemm.py` (`TILE_CONTIGUOUS_FP8`, `DECODE_FP8_CONFIGS`, `WIDE_DECODE`):**
  - Add `(12288, 1280)` to `TILE_CONTIGUOUS_FP8` and `DECODE_FP8_CONFIGS`.
  - Add TP3 decode tile configs for `(1536, 5120)`, `(1024, 4096, 3)`, `(5120, 3072)`, and `(1728, 15360)` to `DECODE_FP8_CONFIGS` and `WIDE_DECODE`.
- **`tests/models/test_deepseek_v4_1_attention.py`:**
  - Add `test_tp3_rank2_trims_zero_padded_output_group`.

## Verification & Benchmarks on `dgx4` (GB10, `spark-ds41f:19f2c20ed4d6-next-20`)

### 1. Microbenchmarks (`M = 6`, single-stream `c1` speculative decode)

| Projection / Kernel | TP3 Baseline (`rc20`) | TP3 Ranks 0 & 1 (Tuned Tiles) | TP3 Rank 2 (Trimmed `3g -> 2g`, `24h -> 16h`) | Bit-Exact (`torch.equal`) |
|---|---:|---:|---:|:---:|
| `Q-B` (`rows=6`, warm) | `12.07 µs` | `10.75–11.80 µs` (`-2.2%` to `-10.9%`) | `9.22 µs` (`-21.9%`) | `True` |
| `Q-B` (`rows=24`, warm) | `15.96 µs` | `14.78 µs` (`-7.4%`) | `11.90 µs` (`-25.4%`) | `True` |
| `WO-A` `(1024, 4096, g=3)` (cold DRAM) | `92.66 µs` | `89.34 µs` (`-3.6%`, `-3.32 µs`/layer) | `27.75 µs` fused `WO-A+B` (`-73.8%` in L2) | `True` |
| `WO-B` `(5120, 3072)` (warm L2 / cold DRAM) | `18.40 µs` / `112.37 µs` | `11.93 µs` (`-35.2%`) / `107.53 µs` (`-4.3%`) | `27.75 µs` fused `WO-A+B` (`-73.8%` in L2) | `True` |
| `SparseMLA` (`idx_w=512, rows=6`) | `14.44 µs` (`decode`) | `14.44 µs` (`decode`) | `12.30 µs` (`split`, `-14.9%`) | `True` |
| `SharedExpert gate_up` `(1536, 5120)` (warm / cold) | `8.72 µs` / `52.40 µs` | `7.12 µs` (`-18.3%`) / `48.98 µs` (`-6.5%`) | `7.12 µs` (`-18.3%`) / `48.98 µs` (`-6.5%`) | `True` |
| `DSpark main_proj` `(1728, 15360)` (warm / cold) | `89.48 µs` / `173.64 µs` | `84.40 µs` (`-5.7%`) / `166.15 µs` (`-4.3%`) | `84.40 µs` (`-5.7%`) / `166.15 µs` (`-4.3%`) | `True` |

### 2. Test Suite & PDL Verification

- `50 passed in 10.66s` across `tests/models/test_deepseek_v4_1_attention.py`, `tests/models/test_deepseek_v4_1_config.py`, and `tests/kernels/quantization/test_deepseek_v41_tilelang_linear.py`.
- `PDL sources: 113 kernels launched, 64 with PDL, 0 violations`.
