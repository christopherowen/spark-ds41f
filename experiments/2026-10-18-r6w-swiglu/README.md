# r6w candidate: the routed SwiGLU over the live tiles

r6w is r6v ([experiments/2026-10-17-r6v](../2026-10-17-r6v/README.md)) with vLLM 0181,
the experiment [2026-10-14-moe-swiglu](../2026-10-14-moe-swiglu/README.md) planned (owner
approved 2026-10-14, PR #233), built on r6v's tree.

## Why

Between the routed experts' gate/up and down GEMMs, TileKernels'
`swiglu_forward_and_per_token_cast` applies DeepSeek's clamp, the SwiGLU and the route
weight, and casts the down projection's input to MXFP8. It takes about 22 µs a layer at
every decode row count, 1 to 96 (bench 2026-10-14, `tk-swiglu`), and nothing overlaps it:
each of its 48 1,024-thread CTAs walks the boundaries of all 384 experts, one expert at a
time, before it reaches a row. The shared expert's SwiGLU takes about 3 µs.

**Ceiling** (the 2026-10-14 experiment, from the 2026-10-13 fixed-row profile, TP4): a kernel
near 3 µs saves 20-27 µs a layer, about **2.4-2.7% of a decode step** at one to sixteen
streams (40 layers), plus the drafter's MoE layer. Prefill is memory-bound at about
230 GB/s in this kernel; it must not lose there.

## Change

vLLM 0181, `moe_swiglu_cast` in `vllm/models/deepseek_v4_1/tilelang/moe.py`, replacing the
TileKernels call:

- **Schedule.** A CTA takes 4 rows of one 32-row tile of the dispatch layout (2 when a row
  has fewer than four warps: TP3's 768 columns), issues every row's loads before any
  row's arithmetic, and the CTAs of tiles past `num_tiles` (`tile_expert < 0`) stop at
  once; padding rows (`order < 0`) are skipped, as TileKernels skips them. A decode step
  pays for its live rows only.
- **Arithmetic, element for element TileKernels'.** Route weights; the clamp by compare and
  select (spark-ds41f's TileKernels patch 0001, so a NaN passes as through `torch.clamp`);
  the fast `__expf`/`__fdividef` product, its certificate and the precise fallback; the
  same vector width and `get_sf_and_inv`. A warp takes the same `32 * vec` consecutive
  columns of one row as a TileKernels 32-thread group, so the certificate's warp vote
  covers the same elements and every byte and scale must equal TileKernels'.

## Gates

1. **Bits** (`bundles/swiglu-tests`, `bundles/swiglu-bench`): every live row's E4M3 bytes
   and UE8M0 scales equal TileKernels' on real dispatch layouts at TP4, TP3 and the
   drafter's shapes, decode captures and prefill chunks, with gates past the fast exp's
   domain and the smallest scales; a NaN gate value reaches its row; graph replays give
   the eager bits. The TileLang MoE tests (FP32 reference, arithmetic, NaN rows) pass.
2. **Time per call** (`bundles/swiglu-bench`): faster at decode; within kbench's 3% floor at
   every prefill size.
3. **Serving** (`screen.json`, the ring, 1017): temperature-0 and greedy outputs identical
   to r6v's (3eab5112 / db25fde3, c9cf9dbe); the `release` suite plus `single-stream`, r6w,
   r6v, r6w again: never slower on any metric, the engine grid and the closed-loop streams
   faster.

## Runs

1. `kernel.json`: the tests on dgx1 and the bench on dgx2, the overlay mounted over the r6v
   image.
2. `screen.json`: the determinism gate and distinct streams on the overlay arm, then the
   release comparison.

The owner decides promotion; an image build follows a promotion decision.

## Results

### Kernel (2026-10-18, dgx2, r6v image with the overlay)

Every size's bytes and scales equal TileKernels' at every shape, and graph replays give
the eager bits; the TileLang MoE tests pass (47, `bundles/swiglu-tests` on dgx1). Time per
call, microseconds, warm / cold, TileKernels then ours
([results/bench-default.txt](results/bench-default.txt)):

| Rows | TP4 TileKernels | TP4 ours | TP3 TileKernels | TP3 ours |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 21.9 / 24.6 | **1.5** / 4.4 | 22.3 / 26.4 | **1.9** / 6.1 |
| 6 | 22.0 / 24.6 | **2.4** / 6.0 | 22.8 / 26.6 | **3.0** / 6.1 |
| 16 | 22.1 / 24.6 | **4.3** / 8.2 | 22.9 / 26.6 | **3.4** / 8.2 |
| 48 | 22.6 / 26.6 | **9.2** / 14.5 | 22.8 / 26.6 | **6.3** / 12.3 |
| 96 | 23.5 / 26.8 | **12.8** / 20.5 | 24.2 / 27.0 | **8.4** / 16.4 |
| 512 | 32.3 / 43.0 | 21.3 / 41.0 | | |
| 8,192 | 661 / 637 | 615 / 592 | | |

Worst ratio (ours / TileKernels) per bucket, warm or cold: TP4 decode 0.77, mid 0.95,
prefill 0.96; TP3 0.61, 0.98, 0.98; the drafter (128 experts, top 3) 0.85, 0.90, 0.93.

How the schedule was chosen, all at identical bits:
- 8 rows a CTA, loads row by row: decode 2-10x faster, but slower cold at TP3's 512 rows
  (1.09) and the drafter's 96 rows (1.19): each row's loads waited for the previous row's
  arithmetic, while TileKernels prefetches its next token into L2
  ([results/bench-rows-sweep-serial.txt](results/bench-rows-sweep-serial.txt), 1-8 rows).
  One row a CTA was worse everywhere (1.15-1.75 cold).
- Every row's loads first: the drafter and TP4 within the floor at 4 and 8 rows; TP3's 512
  rows needed 2 rows a CTA (its rows have 3 warps, TP4's 9)
  ([results/bench-rows-sweep-loads-first.txt](results/bench-rows-sweep-loads-first.txt)).
