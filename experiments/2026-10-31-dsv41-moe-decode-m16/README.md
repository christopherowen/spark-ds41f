# MoE: 16-row decode tiles below one tile per expert

## Why

On SM120 / SM121, TileLang's block-scaled MMA (`mma.sync.aligned.block_scale.m16n8k32.row.col.f32.e4m3.e2m1.f32.ue8m0`) has an M-dimension atom height of 16 rows and partitions a `(block_M = 16, block_N = 128)` tile across 4 warps along N as `1 x 4` (`compute_warp_partition(..., is_m_first=False)`).

In DeepSeek-V4.1-Flash (`E = 384, top-6`; drafter `E = 128, top-3`), `MoeExpertsV41Plan` previously used two tile heights:
- `block_M = 32` for `rows * topk < 32 * num_experts` (`rows < 2,048` on the target model), and
- `prefill_block_M = 64` for `rows * topk >= 32 * num_experts` (`rows >= 2,048`).

Across the entire decode regime (`1 <= rows <= 96`, up to 16 streams at 6 verification tokens per stream) and short prefills (`rows < 1,024`, where `rows * topk < 16 * num_experts`), nearly every active expert receives at most 16 routes (`used_tiles` is identical for `block_M = 16` and `block_M = 32` from `rows = 1` through `rows = 256`, and `385` vs `383` at `rows = 512`). With 32-row tiles, rows `16..31` of every active tile are unused padding (`Order[r] == -1`), yet every kernel in the routed-MoE chain still pays for them:
1. **`moe_dispatch_v41`:** the per-expert padding loop (`for i in T.serial(padded - count)`) writes 16 extra `-1` / `0.0` stores to `Order` and `LayoutWeights` for every active expert.
2. **`grouped_fp8_fp4_gemm` (`gate_up` and `down`):** each thread issues `a_copies = 2` predicated 16-byte `cp.async` activation loads per K stage (`1` at `block_M = 16`), allocates twice the `A_shared` and `SFA_shared` shared memory, holds `32` instead of `16` FP32 accumulator registers per thread, and executes twice as many `ldmatrix` and `mma.sync.m16n8k32` instructions on zero-padded rows.
3. **`moe_swiglu_cast`:** launches `block_M // rows_per_cta` CTAs per tile (`8` per tile on TP4, `16` per tile on TP3 at `block_M = 32` vs `4` on TP4, `8` on TP3 at `block_M = 16`), halving the grid size and eliminating CTAs whose rows are all `-1`.
4. **Workspace scratch (`layout_rows = tiles * height`):** once `routes > num_experts` (`rows >= 65`), `layout_tiles(rows, topk, height, num_experts) = ceildiv(routes, height) + num_experts` is dominated by `+ num_experts`, so 16-row tiles cut `layout_rows` (`order`, `layout_weights`, `h`, `hq`, `h_sf`, `y`) nearly in half (at `rows = 96`: `6,720` rows vs `12,864`, saving `86.2 MiB` on TP3 and `79.6 MiB` on TP4).

Between `1,024 <= rows < 2,048` (`16 * num_experts <= rows * topk < 32 * num_experts`), the average expert has filled a 16-row tile (`used_tiles` at `block_M = 16` rises to `557` at `1,024` rows and `763` at `1,536` rows, whereas `block_M = 32` keeps `used_tiles = 384` and `401` and remains faster than both `16` and `64`). Keeping all three tiers (`16 / 32 / 64`) matches each regime to the tile height where the average expert fills one tile.

## Change

[0001-dsv41-moe-decode-m16.patch](0001-dsv41-moe-decode-m16.patch) against vLLM `d5df88d83`:
- Adds `decode_block_M = 16` to `MoeExpertsV41Plan` (`vllm/models/deepseek_v4_1/tilelang/moe.py`).
- `MoeExpertsV41Plan.tile_height(rows)` selects:
  - `decode_block_M` (`16`) when `rows * topk < decode_block_M * num_experts` (`rows < 1,024` for 384 experts, top-6; `rows < 683` for the 128-expert top-3 drafter),
  - `block_M` (`32`) when `decode_block_M * num_experts <= rows * topk < block_M * num_experts` (`1,024 <= rows < 2,048`),
  - `prefill_block_M` (`64`) when `rows * topk >= block_M * num_experts` (`rows >= 2,048`).
- Declares and binds the 16-row decode kernels (`gate_up`, `down`, `swiglu`, `late_gate_up`, `late_down`), the 32-row mid kernels (`mid_gate_up`, `mid_down`, `mid_swiglu`), and the 64-row prefill kernels (`prefill_gate_up`, `prefill_down`, `prefill_swiglu`), plus their `MoeDispatchV41Plan` instances.
- Updates `tests/kernels/moe/test_deepseek_v41_tilelang_moe.py` to test the 3-tier `tile_height` schedule and scratch layout, bit-identity across all three tile heights (`16`, `32`, `64`) and the tiered plan, and `moe_swiglu_cast` against TileKernels at `block_M = 16` and `64`.

## Why the bits do not change

Each layout row belongs to a single route `(token, slot)` in route order. Neither `grouped_fp8_fp4_gemm`, `moe_swiglu_cast`, nor `moe_combine_v41` reduces across layout rows inside a tile: every route's K-loop reduction runs in the exact same order regardless of `block_M`, and `moe_combine_v41` sums each token's routes in slot order `k = 0 .. topk - 1` via `route_pos[row, k]`.

## Verification and dgx4 Benchmark (`spark-ds41f:1030`, GB10)

### 1. Kernel tests and PDL source audit
Ran on `dgx4` under `flock /tmp/spark-lab-gpu.lock`:
```bash
pytest -v -p tests.kernels.deepseek_v41_pdl_sources tests/kernels/moe/test_deepseek_v41_tilelang_moe.py
```
- **Result:** `113 passed in 28.67s`
- **PDL source check:** `PDL sources: 97 kernels launched, 86 with PDL, 0 violations`
- **Bit-exactness:** `torch.equal` on `int16` views across every tested `rows` (`1` through `2,048`) on TP4 (`I = 576`), TP3 (`I = 768`), and drafter (`E = 128, topk = 3, I = 768`) shapes.

### 2. Per-kernel and end-to-end latency on `dgx4` (`block_M = 32` vs `decode_block_M = 16`)

#### TP4 (`E = 384, topk = 6, H = 5120, I = 576`)
| `rows` | `used_tiles` (`32` -> `16`) | `layout_rows` (`32` -> `16`) | Scratch (`32` -> `16`) | `dispatch` (us) | `gate_up` (us) | `swiglu` (us) | `down` (us) | Total `plan.run` (`32` -> `16`, us) | Delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 6 -> 6 | 192 -> 96 | 2.5 -> 1.3 MiB | 11.0 -> 11.1 | 50.5 -> 44.3 | 9.6 -> 9.6 | 22.0 -> 19.4 | 121.0 -> 115.8 | -5.2 us (-4.3%) |
| 2 | 12 -> 12 | 384 -> 192 | 5.0 -> 2.5 MiB | 11.2 -> 11.0 | 158.8 -> 156.8 | 9.5 -> 9.6 | 46.2 -> 38.2 | 242.9 -> 239.8 | -3.1 us (-1.3%) |
| 4 | 22 -> 22 | 768 -> 384 | 10.0 -> 5.0 MiB | 11.4 -> 10.9 | 286.9 -> 284.9 | 9.4 -> 9.4 | 125.1 -> 122.1 | 430.3 -> 426.2 | -4.1 us (-1.0%) |
| 6 | 35 -> 35 | 1,152 -> 576 | 14.9 -> 7.5 MiB | 11.5 -> 10.8 | 451.8 -> 447.8 | 9.4 -> 9.5 | 212.5 -> 212.2 | 674.0 -> 671.0 | -3.0 us (-0.4%) |
| 8 | 47 -> 47 | 1,536 -> 768 | 19.9 -> 10.0 MiB | 11.7 -> 10.9 | 597.3 -> 596.0 | 9.4 -> 9.3 | 290.2 -> 286.9 | 885.8 -> 879.9 | -5.9 us (-0.7%) |
| 12 | 65 -> 65 | 2,304 -> 1,152 | 29.8 -> 14.9 MiB | 13.0 -> 11.0 | 817.4 -> 814.5 | 9.4 -> 9.4 | 398.6 -> 397.4 | 1,225.0 -> 1,218.7 | -6.3 us (-0.5%) |
| 16 | 81 -> 81 | 3,072 -> 1,536 | 39.8 -> 19.9 MiB | 13.6 -> 10.9 | 1,010.6 -> 1,007.9 | 9.4 -> 9.3 | 494.8 -> 493.0 | 1,519.9 -> 1,511.6 | -8.3 us (-0.5%) |
| 24 | 128 -> 128 | 4,608 -> 2,304 | 59.7 -> 29.9 MiB | 14.6 -> 12.6 | 1,591.3 -> 1,590.3 | 11.4 -> 9.3 | 783.9 -> 780.7 | 2,384.1 -> 2,373.7 | -10.4 us (-0.4%) |
| 32 | 151 -> 151 | 6,144 -> 3,072 | 79.5 -> 39.8 MiB | 14.5 -> 13.0 | 1,871.9 -> 1,871.0 | 12.5 -> 9.3 | 923.8 -> 922.5 | 2,815.1 -> 2,806.0 | -9.1 us (-0.3%) |
| 48 | 197 -> 197 | 9,216 -> 4,608 | 119.3 -> 59.7 MiB | 15.6 -> 13.3 | 2,442.2 -> 2,438.4 | 12.7 -> 11.4 | 1,214.1 -> 1,210.2 | 3,679.6 -> 3,664.0 | -15.6 us (-0.4%) |
| 64 | 247 -> 247 | 12,288 -> 6,144 | 159.1 -> 79.6 MiB | 17.7 -> 14.6 | 3,051.5 -> 3,046.9 | 13.6 -> 12.5 | 1,526.8 -> 1,517.7 | 4,604.2 -> 4,581.4 | -22.8 us (-0.5%) |
| 96 | 301 -> 301 | 12,864 -> 6,720 | 166.6 -> 87.1 MiB | 22.8 -> 19.3 | 3,722.2 -> 3,715.8 | 15.6 -> 12.5 | 1,848.4 -> 1,844.1 | 5,619.8 -> 5,590.9 | -28.9 us (-0.5%) |
| 128 | 334 -> 334 | 13,056 -> 6,912 | 169.1 -> 89.6 MiB | 22.9 -> 22.9 | 4,132.9 -> 4,127.0 | 15.6 -> 12.5 | 2,058.4 -> 2,047.1 | 6,238.4 -> 6,211.8 | -26.6 us (-0.4%) |
| 256 | 380 -> 380 | 13,824 -> 7,680 | 179.0 -> 99.6 MiB | 25.9 -> 21.0 | 4,764.7 -> 4,757.8 | 19.7 -> 12.4 | 2,368.9 -> 2,363.6 | 7,190.6 -> 7,168.0 | -22.6 us (-0.3%) |
| 512 | 383 -> 385 | 15,360 -> 9,216 | 199.0 -> 119.6 MiB | 23.8 -> 19.3 | 4,941.3 -> 4,949.0 | 21.8 -> 15.6 | 2,487.4 -> 2,482.2 | 7,495.7 -> 7,482.5 | -13.2 us (-0.2%) |

#### TP3 (`E = 384, topk = 6, H = 5120, I = 768`)
| `rows` | `used_tiles` (`32` -> `16`) | `layout_rows` (`32` -> `16`) | Scratch (`32` -> `16`) | `dispatch` (us) | `gate_up` (us) | `swiglu` (us) | `down` (us) | Total `plan.run` (`32` -> `16`, us) | Delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 6 -> 6 | 192 -> 96 | 2.7 -> 1.4 MiB | 11.2 -> 10.9 | 56.5 -> 52.4 | 9.5 -> 9.6 | 25.9 -> 21.8 | 164.2 -> 164.0 | -0.2 us |
| 2 | 12 -> 12 | 384 -> 192 | 5.4 -> 2.7 MiB | 11.1 -> 10.8 | 216.2 -> 215.2 | 9.5 -> 9.7 | 57.6 -> 50.4 | 319.6 -> 315.6 | -4.0 us (-1.3%) |
| 4 | 22 -> 22 | 768 -> 384 | 10.8 -> 5.4 MiB | 11.2 -> 11.0 | 387.9 -> 385.1 | 9.6 -> 9.5 | 154.9 -> 153.4 | 562.3 -> 560.4 | -1.9 us (-0.3%) |
| 8 | 47 -> 47 | 1,536 -> 768 | 21.6 -> 10.8 MiB | 11.8 -> 10.9 | 806.9 -> 802.8 | 9.4 -> 9.5 | 373.8 -> 373.7 | 1,171.5 -> 1,167.3 | -4.2 us (-0.4%) |
| 12 | 65 -> 65 | 2,304 -> 1,152 | 32.4 -> 16.2 MiB | 13.1 -> 10.9 | 1,095.8 -> 1,093.0 | 9.3 -> 9.4 | 514.9 -> 513.1 | 1,619.3 -> 1,613.1 | -6.2 us (-0.4%) |
| 16 | 81 -> 81 | 3,072 -> 1,536 | 43.2 -> 21.6 MiB | 13.6 -> 10.9 | 1,362.0 -> 1,358.9 | 9.4 -> 9.4 | 643.2 -> 640.1 | 2,023.5 -> 2,007.3 | -16.2 us (-0.8%) |
| 24 | 128 -> 128 | 4,608 -> 2,304 | 64.8 -> 32.4 MiB | 14.6 -> 12.6 | 2,140.4 -> 2,142.2 | 9.5 -> 9.5 | 1,015.0 -> 1,007.9 | 3,174.1 -> 3,165.4 | -8.7 us (-0.3%) |
| 32 | 151 -> 151 | 6,144 -> 3,072 | 86.4 -> 43.2 MiB | 14.5 -> 13.1 | 2,521.1 -> 2,519.0 | 10.4 -> 9.4 | 1,199.4 -> 1,191.2 | 3,740.9 -> 3,727.3 | -13.6 us (-0.4%) |
| 48 | 197 -> 197 | 9,216 -> 4,608 | 129.6 -> 64.9 MiB | 15.6 -> 13.4 | 3,286.0 -> 3,283.7 | 11.3 -> 9.4 | 1,565.9 -> 1,557.6 | 4,881.6 -> 4,863.1 | -18.5 us (-0.4%) |
| 64 | 247 -> 247 | 12,288 -> 6,144 | 172.8 -> 86.5 MiB | 17.7 -> 14.6 | 4,119.8 -> 4,108.3 | 12.5 -> 9.8 | 1,981.4 -> 1,978.4 | 6,135.1 -> 6,122.8 | -12.3 us (-0.2%) |
| 96 | 301 -> 301 | 12,864 -> 6,720 | 181.0 -> 94.6 MiB | 22.8 -> 19.3 | 5,027.9 -> 5,021.7 | 12.5 -> 10.5 | 2,400.3 -> 2,395.9 | 7,472.7 -> 7,462.6 | -10.1 us (-0.1%) |
| 128 | 334 -> 334 | 13,056 -> 6,912 | 183.7 -> 97.3 MiB | 23.8 -> 19.0 | 5,585.9 -> 5,577.7 | 12.5 -> 11.4 | 2,671.6 -> 2,664.2 | 8,303.9 -> 8,286.3 | -17.6 us (-0.2%) |
| 512 | 383 -> 385 | 15,360 -> 9,216 | 216.2 -> 129.9 MiB | 23.8 -> 19.2 | 6,650.9 -> 6,660.1 | 15.6 -> 12.5 | 3,190.8 -> 3,182.6 | 9,899.2 -> 9,889.0 | -10.2 us (-0.1%) |

#### Why 32-row mid tiles are kept for `1,024 <= rows < 2,048`
| `rows` | `used_tiles` (`16 / 32 / 64`) | TP4 `plan.run` (`16 / 32 / 64`, us) | TP3 `plan.run` (`16 / 32 / 64`, us) | Best height |
| ---: | ---: | ---: | ---: | ---: |
| 1,024 | `557 / 384 / 384` | `8,183.8 / 7,998.5 / 8,079.5` | `10,641.6 / 10,483.8 / 10,517.6` | **32** (`block_M`) |
| 1,536 | `763 / 401 / 384` | `8,885.8 / 8,358.9 / 8,554.5` | `11,284.5 / 10,824.7 / 10,979.3` | **32** (`block_M`) |
| 2,048 | `957 / 561 / 384` | `9,522.2 / 9,187.0 / 9,048.3` | `12,184.6 / 11,811.1 / 11,458.7` | **64** (`prefill_block_M`) |
