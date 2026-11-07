# Cross-Topology Parity: `compressor.fused_wkv_wgate` TP3/TP5..TP7 Sharding, `fused_wqa_wkv` TP5..TP8 Decode Configs, and `_trim_padded_groups` Wholly Padded Ranks (+ Combined `[34]` + `[27]` 12-Bit Compressor Stack)

## Summary

- **Base vLLM commit**: `a75357bba81e31e4f3085700f94b6a0b29aed44a` (`candidate-1107` / `0743` on `sp8`)
- **Stacked commits on `sp8:/home/chris/spark-lab/rework-cross-topology-parity/vllm` (`feat/cross-topology-parity`)**:
  1. `41a3d08d5fd39958a8bde08bc1ae32c5c4860c93` (`0001-perf-dsv41-extend-compressor-sharding-wqa_wkv-decode.patch` — `[34]`):
     - Pads each rank's shard of `compressor.fused_wkv_wgate` (`wkv` `256` + `wgate` `256` = `512` valid cols) to a 64-aligned per-projection width across all TP sizes (`TP3`: `192` cols/proj, `TP4..TP7`: `128` cols/proj, `TP8`: `64` cols/proj) via `_PaddedSplitColumnParallelLinear`, zeroing pad rows before weight packing (`_zero_pad_rows()`) and slicing valid rows after `tensor_model_parallel_all_gather`.
     - Adds tuned `DECODE_FP8_CONFIGS` (`M <= 16: bm=16`, `M <= 32: bm=32`, `M <= 64: bm=64`) and `WIDE_DECODE` (`bn=128, bk=128, stages=2, threads=128, split_k=1`) entries for sharded `fused_wqa_wkv` on `TP5` `(384, 5120)`, `TP6` `(320, 5120)`, `TP7` `(256, 5120)`, and `TP8` `(224, 5120)`.
     - Extends `_trim_padded_groups` in `vllm/models/deepseek_v4_1/attention.py` to handle `real_groups == 0` on wholly padded ranks (`TP5` rank 4, `TP6` ranks 4–5, `TP7` ranks 4–6), replacing `wq_b` (`(2048, 1536)` FP8 GEMM), `wo_a` (`2 x (1024, 512)` BF16 GEMMs), and `wo_b` (`(5120, 2048)` FP8 GEMM) with zero tensors while preserving collective participation.
  2. `b2ae2a554791cb17688a8fb8d6c75bb558732a7e` (`0002-perf-dsv41-pack-compressor-BF16-weights-to-12-bit-se.patch` — `[27]` rebased on top of `[34]`):
     - Packs `compressor.fused_wkv_wgate` BF16 weights into selective 12-bit storage (`pack_bf16_layer` / `TileLangSplitLinearMethod`), calling `layer._zero_pad_rows()` before packing so invalid pad rows on TP3/TP5/TP6/TP7 are zeroed prior to 12-bit exponent-window analysis and packing, and unpacking via `unpack_bf16_layer(self.fused_wkv_wgate)` before `tensor_model_parallel_all_gather` on the SP prefill path.

## 2-Boot Verification & Benchmark Receipts (`sp8`, `spark-ds41f:candidate-38`)

### Pytest Suite (Both Boots)
- `tests/models/test_deepseek_v4_1_compressor_prepare.py`, `tests/models/test_deepseek_v4_1_compressor.py`, `tests/models/test_deepseek_v4_1_attention.py`: **104 passed**
- `tests/kernels/quantization/test_deepseek_v41_tilelang_linear.py -k "test_split_linear or test_packed_bf16_compressor_shapes_exact or test_sharded_wqa_wkv_and_strided_attention_front_match_replicated"`: **19 passed** (including all `part_n in {512, 256, 192, 128, 64}`, `parts in {1, 2}`)

### 1. `compressor.fused_wkv_wgate` Bit-Exactness (`TP1..TP8`, `ratio in {1, 2}`) & TP3 Benchmark
- `torch.equal` bit-identical (`sp_weight_exact=True`, `project_exact=True`) across all `tp_size in {1, 2, 3, 4, 5, 6, 7, 8}` and `compress_ratio in {1, 2}` on both boots.
- **Combined `[34] + [27]` (`pack12` + TP3 `192/rank` vs `512` replicated `pack12`, Cold-L2, 2-boot mean)**:
  - `ratio=1, M=1`: `21.22 µs` -> `12.24 µs` (**1.73x** sharding speedup on top of `pack12`; **2.38x** total vs `29.10 µs` replicated BF16; saves **0.55 ms** / token across 61 layers)
  - `ratio=1, M=16`: `22.12 µs` -> `12.24 µs` (**1.81x**, saves **0.60 ms** / 61L)
  - `ratio=1, M=32`: `23.95 µs` -> `12.78 µs` (**1.87x**, saves **0.68 ms** / 61L)
  - `ratio=1, M=128`: `37.36 µs` -> `16.80 µs` (**2.22x**, saves **1.25 ms** / 61L)
  - `ratio=1, M=256`: `51.45 µs` -> `25.84 µs` (**1.99x**, saves **1.56 ms** / 61L)
  - `ratio=2, M=1`: `40.30 µs` -> `19.72 µs` (**2.04x**, saves **1.26 ms** / 61L)

### 2. Sharded `fused_wqa_wkv` Decode GEMM Configs (`TP5..TP8`, 2-Boot Mean)
- **`TP5 (384, 5120)`**: `M=1`: `8.64 µs -> 5.10 µs` (**1.69x**), `M=16`: `8.55 µs -> 4.98 µs` (**1.72x**), `M=32`: `10.82 µs -> 6.29 µs` (**1.72x**), `M=64`: `16.47 µs -> 9.00 µs` (**1.83x**), `M=96`: `43.42 µs -> 8.44 µs` (**5.15x**), `M=128`: `46.50 µs -> 9.50 µs` (**4.89x**)
- **`TP6 (320, 5120)`**: `M=1`: `10.14 µs -> 5.12 µs` (**1.98x**), `M=16`: `8.58 µs -> 4.99 µs` (**1.72x**), `M=32`: `10.85 µs -> 6.19 µs` (**1.75x**), `M=64`: `17.12 µs -> 9.02 µs` (**1.90x**), `M=96`: `43.99 µs -> 8.60 µs` (**5.12x**), `M=128`: `44.67 µs -> 9.11 µs` (**4.90x**)
- **`TP7 (256, 5120)`**: `M=1`: `8.63 µs -> 5.06 µs` (**1.71x**), `M=16`: `8.33 µs -> 4.96 µs` (**1.68x**), `M=32`: `10.98 µs -> 5.92 µs` (**1.86x**), `M=64`: `15.85 µs -> 8.98 µs` (**1.76x**), `M=96`: `43.19 µs -> 6.21 µs` (**6.96x**), `M=128`: `47.94 µs -> 9.03 µs` (**5.31x**)
- **`TP8 (224, 5120)`**: `M=1`: `8.71 µs -> 5.06 µs` (**1.72x**), `M=16`: `8.56 µs -> 4.93 µs` (**1.74x**), `M=32`: `10.60 µs -> 6.04 µs` (**1.76x**), `M=64`: `16.31 µs -> 8.96 µs` (**1.82x**), `M=96`: `44.06 µs -> 7.75 µs` (**5.69x**), `M=128`: `43.53 µs -> 9.10 µs` (**4.78x**)

### 3. `_trim_padded_groups` (`real_groups == 0` on Wholly Padded Ranks, 2-Boot Mean)
- Bit-exact (`torch.equal`) across decode, non-SP prefill, and SP prefill for `TP3`, `TP5`, `TP6`, and `TP7`.
- Wholly padded rank per-layer `wq_b + wo_a + wo_b` latency reduction:
  - `M=1`: `100.37 µs -> 7.39 µs` (**5.67 ms** saved across 61 layers)
  - `M=16`: `117.31 µs -> 7.13 µs` (**6.72 ms** saved across 61 layers)
  - `M=64`: `160.08 µs -> 7.19 µs` (**9.33 ms** saved across 61 layers)
  - `M=256`: `249.10 µs -> 7.19 µs` (**14.76 ms** saved across 61 layers)
  - `M=2048`: `1131.02 µs -> 102.39 µs` (**62.75 ms** saved across 61 layers)
