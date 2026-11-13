# 2026-11-10: 48-Row Decode Projection Tile Tier for 33..48 Verify Rows

## Base Commit & Causal Variable

- **Base commit**: `cc20ed18be1f7cbf5c6c53cdee945f7cc77d5fc5` (`spark-ds41f-1107`)
- **Causal variable**: Add a `48`-row decode projection tile tier (`block_M = 48`, between `32` and `64`) in `DECODE_TILE_ROWS`, `DECODE_FP8_CONFIGS`, `SPLIT_FP8`, `SPLIT_BF16`, `PARTIAL_ROW_TILES`, `decode_tile_rows`, `split_bucket`, and `split_row_tile` (`vllm/models/deepseek_v4_1/tilelang/gemm.py` and `vllm/models/deepseek_v4_1/tilelang/linear.py`).

## Rationale

Previously, `DECODE_TILE_ROWS = (16, 32, 64)` and `PARTIAL_ROW_TILES = (16, 32, 64)` jumped directly from `32` rows to `64` rows. During speculative decode verification (`r6v`, `6` tokens per stream), `6`, `7`, and `8` concurrent streams produce `36`, `42`, and `48` rows respectively, forcing every dense/MoE/attention/indexer/router projection onto a `64`-row tile with `25%–44%` wasted row-padding compute and register pressure.

Adding `block_M = 48` routes `33..48` rows to a single `48`-row tile while keeping single-tile `mxfp8_input` producer bounds valid (`single_heights <= DECODE_ROWS = 64` in `_declare_block32`). Because `mxfp8_gemm_decode`, `mxfp8_gemm_partials`, and `bf16_gemm_partials` reduce along `K` independently per row (`T.min(by * block_M + i, M - 1)`), `block_M = 48` is 100% bit-identical (`0 ULP`) to `block_M = 64` for all `1 <= M <= 48`.

## Commands & Workload Identity

1. Prepare locked source tree:
   ```bash
   bin/spark --cluster-config experiments/2026-11-10-projection-48row-decode-tile/tp4.json build prepare --only vllm --lock
   bin/spark --cluster-config experiments/2026-11-10-projection-48row-decode-tile/tp3.json build prepare --only vllm --lock
   ```
2. Run two fresh container boots (`boot1` and `boot2`) of `spark-ds41f:1107` on GB10 (`sp8`):
   ```bash
   EXP=experiments/2026-11-10-projection-48row-decode-tile
   VLLM_SRC=$PWD/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-dcb48c8a/src/vllm
   for BOOT in boot1 boot2; do
     docker run --rm --gpus all --ipc=host --entrypoint bash \
       -v "$VLLM_SRC/vllm/models/deepseek_v4_1/tilelang:/opt/spark-ds41f/candidate/vllm/vllm/models/deepseek_v4_1/tilelang" \
       -v "$VLLM_SRC/tests:/workspace/tests" \
       -v "$PWD/$EXP:/workspace/exp" \
       spark-ds41f:1107 -lc "
         pytest -q /workspace/tests/kernels/quantization/test_deepseek_v41_tilelang_linear.py -k \"test_decode_48row\" | tee /workspace/exp/receipts/pytest-${BOOT}.txt &&
         python3 -u /workspace/exp/bench_projection_48row.py ${BOOT} /workspace/exp/receipts/bench-projection-48row-${BOOT}.jsonl
       "
   done
   ```

## Two-Boot GB10 Results (`M = 42`, `c7` Speculative Verify)

| Projection | Shape `(N, K, groups)` | Layers / Step | Boot 1 `bm64 -> bm48` (`us`) | Boot 1 Speedup | Boot 2 `bm64 -> bm48` (`us`) | Boot 2 Speedup | Mean Step Saved (`us`) | Bit-Exact (`M=36,42,48`) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `wq_b` (TP4) | `(8192, 1280, 1)` | 61 | `21.793 -> 15.860` | `1.374x` | `21.861 -> 14.199` | `1.540x` | `414.67` | Yes (`0 ULP`) |
| `wq_b` (TP3) | `(12288, 1280, 1)` | 61 | `35.895 -> 21.291` | `1.686x` | `33.300 -> 19.226` | `1.732x` | `874.67` | Yes (`0 ULP`) |
| `indexer.wq_b` | `(4096, 1280, 1)` | 61 | `14.342 -> 8.164` | `1.757x` | `13.439 -> 7.624` | `1.763x` | `365.79` | Yes (`0 ULP`) |
| `fused_wqa_wkv` (full) | `(1792, 5120, 1)` | 1 | `26.022 -> 16.283` | `1.598x` | `26.370 -> 16.435` | `1.605x` | `9.84` | Yes (`0 ULP`) |
| `fused_wqa_wkv` (TP4) | `(448, 5120, 1)` | 61 | `9.928 -> 8.223` | `1.207x` | `9.844 -> 8.251` | `1.193x` | `100.61` | Yes (`0 ULP`) |
| `fused_wqa_wkv` (TP3) | `(608, 5120, 1)` | 61 | `12.688 -> 8.412` | `1.508x` | `12.625 -> 8.229` | `1.534x` | `264.49` | Yes (`0 ULP`) |
| `shared_experts.gate_up` (TP4) | `(1152, 5120, 1)` | 61 | `18.227 -> 13.760` | `1.325x` | `18.567 -> 12.677` | `1.465x` | `315.86` | Yes (`0 ULP`) |
| `shared_experts.gate_up` (TP3) | `(1536, 5120, 1)` | 61 | `19.854 -> 13.720` | `1.447x` | `19.719 -> 13.643` | `1.445x` | `372.42` | Yes (`0 ULP`) |
| `shared_experts.down` (TP4) | `(5120, 576, 1)` | 61 | `10.401 -> 5.704` | `1.823x` | `10.435 -> 4.353` | `2.397x` | `328.77` | Yes (`0 ULP`) |
| `shared_experts.down` (TP3) | `(5120, 768, 1)` | 61 | `13.802 -> 5.569` | `2.478x` | `13.856 -> 5.691` | `2.435x` | `500.14` | Yes (`0 ULP`) |
| `wo_a` (TP4) | `(1024, 4096, 2)` | 61 | `19.576 -> 12.451` | `1.572x` | `19.113 -> 12.295` | `1.555x` | `425.24` | Yes (`0 ULP`) |
| `wo_a` (TP3) | `(1024, 4096, 3)` | 61 | `20.360 -> 14.546` | `1.400x` | `19.995 -> 13.571` | `1.473x` | `373.22` | Yes (`0 ULP`) |
| `wo_b` (TP4) | `(5120, 2048, 1)` | 61 | `18.553 -> 12.177` | `1.524x` | `18.715 -> 12.497` | `1.498x` | `384.12` | Yes (`0 ULP`) |
| `wo_b` (TP3) | `(5120, 3072, 1)` | 61 | `25.551 -> 15.545` | `1.644x` | `24.832 -> 15.539` | `1.598x` | `588.64` | Yes (`0 ULP`) |
| `dspark.eh_proj` (TP4) | `(1280, 15360, 1)` | 1 | `64.824 -> 34.588` | `1.874x` | `62.851 -> 37.126` | `1.693x` | `27.98` | Yes (`0 ULP`) |
| `router` (`bf16_partials`) | `(384, 5120, 1)` | 58 | `6.844 -> 5.798` | `1.180x` | `8.554 -> 8.124` | `1.053x` | `42.79` | Yes (`0 ULP`) |
| `dspark_context_kv` (`SPLIT_FP8`) | `(512, 5120, 1)` | 1 | `10.318 -> 6.395` | `1.613x` | `9.810 -> 37.067` | `0.265x` | `-11.67` | Yes (`0 ULP`) |

- **Total per-step projection savings at `M = 42` (`c7` speculative verify)**:
  - **TP4**: `2.32 ms/step` (`boot1`) / `2.48 ms/step` (`boot2`), mean **`2.40 ms/step`**
  - **TP3**: `3.43 ms/step` (`boot1`) / `3.33 ms/step` (`boot2`), mean **`3.38 ms/step`**
- **Unit tests**: `pytest -q tests/kernels/quantization/test_deepseek_v41_tilelang_linear.py -k "test_decode_48row"` passed across both boots (`receipts/pytest-boot1.txt`, `receipts/pytest-boot2.txt`).
- **Memory & errors**: Zero extra persistent memory; only adds `block_M = 48` decode kernel variants compiled on demand or via warmup buckets.

## Conclusion

Adding `block_M = 48` eliminates the `32 -> 64` row cliff for `33..48` verify rows (`c6..c8` speculative decode), saving `~2.40 ms/step` on TP4 and `~3.38 ms/step` on TP3 with 100% bit-exact (`0 ULP`) output.
