# MoE: 2D CTA panel swizzle for prefill grouped GEMMs

## Why

`grouped_fp8_fp4_gemm` (`vllm/models/deepseek_v4_1/tilelang/moe.py`) launches a 2D CTA grid:
```python
with T.Kernel(T.ceildiv(N, block_N), tiles, threads=threads) as (bx, t):
```
Without threadblock swizzling, CUDA schedules CTAs in `blockIdx.x` (`bx`, the `N / block_N` weight column block) minor order and `blockIdx.y` (`t`, the M-tile index in `0 .. tiles - 1`) major order:
- All `n_blocks = N // block_N` CTAs of tile `t = 0` (`bx = 0 .. n_blocks - 1`) launch first,
- followed by all `n_blocks` CTAs of tile `t = 1`, and so on.

In prefill calls (`rows >= 2,048` on the 384-expert top-6 target model, `prefill_block_M = 64`), each expert owns multiple consecutive M-tiles `t` in the contiguous dispatch layout (`used_tiles = 554` at `4,096` tokens, `948` at `8,192` tokens, `1,720` at `16,384` tokens; on the 128-expert top-3 drafter, `used_tiles = 359 / 586 / 988`). For the `down` projection (`N = 5,120, block_N = 128`), `n_blocks = 40`. When tile `t` and tile `t + 1` belong to the same expert `e`, `40` CTAs across the entire `N = 5,120` weight matrix of expert `e` execute between `(bx, t)` and `(bx, t + 1)`, evicting expert `e`'s `bx` weight slab (`B[e, bx, :]` and scale codes) from GB10's 24 MiB L2 cache before tile `t + 1` reaches column block `bx`. Similarly, for `gate_up` (`N = 2 * I`, `n_blocks = 9` on TP4 and `12` on TP3), consecutive M-tiles of the same expert are separated by `9` or `12` column-block CTAs.

With `T.use_swizzle(panel_size=3)` (`tl::rasterization2DRow<3>()`):
- The grid is partitioned along `t` (`blockIdx.y`) into panels of `3` consecutive M-tiles.
- Within each panel, CTAs step `t` in the inner loop for a fixed `bx`:
  `(bx = 0, t = 0), (bx = 0, t = 1), (bx = 0, t = 2), (bx = 1, t = 0), ...`
- On odd-numbered panels (`panel_idx & 1`), `bx` traverses in reverse (`n_blocks - 1 .. 0`), so consecutive M-tiles of the same expert that straddle a panel boundary also reuse the same `bx` weight slab in L2 immediately.
- Because the outer panel index still advances monotonically along `t`, all unused padding tiles (`t >= used_tiles`, where `TileExpert[t] == -1` and `T.thread_return()` exits immediately) remain at the tail of the grid.

## Change

[0001-dsv41-moe-prefill-swizzle.patch](0001-dsv41-moe-prefill-swizzle.patch) against vLLM `d5df88d83`:
- Adds `swizzle_panel: int = 0` to `grouped_fp8_fp4_gemm` (`vllm/models/deepseek_v4_1/tilelang/moe.py`), calling `T.use_swizzle(panel_size=swizzle_panel)` inside `with T.Kernel(...)` when `swizzle_panel > 0` (before `pdl.wait()` / `pdl.chain()`).
- Adds `prefill_swizzle_panel = 3` to `MoeExpertsV41Plan`, passing `swizzle_panel=self.prefill_swizzle_panel` to `prefill_gate_up` and `prefill_down` while leaving decode GEMMs (`gate_up`, `down`, `late_gate_up`, `late_down`) unswizzled (`swizzle_panel = 0`).
- Adds `test_prefill_swizzle_gives_the_unswizzled_bits` in `tests/kernels/moe/test_deepseek_v41_tilelang_moe.py` verifying bit-exact equality (`torch.equal` on `int16` views) between `prefill_swizzle_panel = 3` and `prefill_swizzle_panel = 0`.

## Why the bits do not change

Each CTA `(bx, t)` computes its own disjoint output tile `C[t * block_M : (t + 1) * block_M, bx * block_N : (bx + 1) * block_N]` with the exact same K-loop accumulation order. `T.use_swizzle(panel_size=3)` bijectively remaps `(blockIdx.x, blockIdx.y)` across the grid without changing any CTA's coordinates or arithmetic.

## Verification and dgx4 Benchmark (`spark-ds41f:1030`, GB10)

### 1. Kernel tests and PDL source audit
Ran on `dgx4` under `flock /tmp/spark-lab-gpu.lock`:
```bash
pytest -v -p tests.kernels.deepseek_v41_pdl_sources tests/kernels/moe/test_deepseek_v41_tilelang_moe.py
```
- **Result:** `102 passed in 22.48s`
- **PDL source check:** `PDL sources: 69 kernels launched, 60 with PDL, 0 violations` (`tl::rasterization2DRow<3>()` has no block scope and touches no pointer parameter before `cudaGridDependencySynchronize()`).
- **Bit-exactness:** `torch.equal` on `int16` views across all tested `swizzle_panel` values (`0, 2, 3, 4, 6, 8`) and shapes (`TP4`, `TP3`, and `drafter`).

### 2. Per-kernel and end-to-end prefill latency on `dgx4` (`swizzle_panel = 0` vs `3`, `prefill_block_M = 64`)

#### TP4 (`E = 384, topk = 6, H = 5120, I = 576`)
| `rows` | `used_tiles` / `tiles` | `prefill_gate_up` (`0` -> `3`, us) | `prefill_down` (`0` -> `3`, us) | `down` delta | End-to-end `plan.run` (`0` -> `3`, us) | `plan.run` delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 384 / 576 | 4,764.6 -> 4,791.3 | 2,869.1 -> 2,846.6 | -0.8% | 9,012.7 -> 9,017.8 | +0.06% |
| 4,096 | 554 / 768 | 5,575.1 -> 5,560.8 | 3,964.9 -> 3,723.5 | **-6.1%** | 11,244.7 -> 10,958.8 | **-285.9 us (-2.54%)** |
| 8,192 | 948 / 1,152 | 6,976.9 -> 6,881.5 (-1.4%) | 5,853.5 -> 5,331.3 | **-8.9%** | 16,790.6 -> 15,995.5 | **-795.1 us (-4.74%)** |
| 16,384 | 1,720 / 1,920 | 11,612.2 -> 11,596.9 | 9,608.3 -> 8,998.1 | **-6.4%** | 28,300.1 -> 27,646.1 | **-654.0 us (-2.31%)** |

#### TP3 (`E = 384, topk = 6, H = 5120, I = 768`)
| `rows` | `used_tiles` / `tiles` | `prefill_gate_up` (`0` -> `3`, us) | `prefill_down` (`0` -> `3`, us) | `down` delta | End-to-end `plan.run` (`0` -> `3`, us) | `plan.run` delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 384 / 576 | 6,571.4 -> 6,599.7 | 3,875.9 -> 3,773.4 | **-2.6%** | 11,488.3 -> 11,468.9 | -19.4 us (-0.17%) |
| 4,096 | 554 / 768 | 7,364.7 -> 7,065.9 (**-4.1%**) | 4,931.9 -> 4,562.9 | **-7.5%** | 13,921.5 -> 13,876.1 | -45.4 us (-0.33%) |
| 8,192 | 948 / 1,152 | 9,599.1 -> 9,065.5 (**-5.6%**) | 7,104.6 -> 6,231.1 | **-12.3%** | 20,237.5 -> 19,417.2 | **-820.3 us (-4.05%)** |
| 16,384 | 1,720 / 1,920 | 15,564.0 -> 15,507.7 | 11,713.9 -> 10,178.6 | **-13.1%** | 34,616.5 -> 33,460.3 | **-1,156.2 us (-3.34%)** |

#### Drafter TP3 (`E = 128, topk = 3, H = 5120, I = 768`)
| `rows` | `used_tiles` / `tiles` | End-to-end `plan.run` (`0` -> `3`, us) | `plan.run` delta |
| ---: | ---: | ---: | ---: |
| 4,096 | 359 / 320 | 5,682.3 -> 5,394.4 | **-287.9 us (-5.07%)** |
| 8,192 | 586 / 512 | 9,448.5 -> 8,861.8 | **-586.7 us (-6.21%)** |
| 16,384 | 988 / 896 | 18,709.6 -> 17,173.7 | **-1,535.9 us (-8.21%)** |
