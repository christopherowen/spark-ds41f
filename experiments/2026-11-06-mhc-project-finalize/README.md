# 2026-11-06 — mHC Fused `project_streams` + `finalize` (`0731`)

- **Base deployment commit**: `0b3e98934f0aa7c7c251b79dae3ac148f869304e` (Candidate `1107`)
- **Base vLLM commit**: `c47b54ae471a24b830e3515212b8c0bf5ee14604` (tree `13e4781d19bc4a18b0d3d4daad7921c465e3355e`, 107 patches through `0730`)
- **Candidate vLLM commit**: `67d1b007b517f70b88f1e153ec63a11a14a44ab9` (tree `148204769ca280a5e9e7e46c04dc2f661be9ab6a`, 108 patches through `0731-mhc-project-finalize.patch`)
- **Candidate branch**: `feat/mhc-project-finalize` (PR #946)
- **Intended causal variable**: Parallelize Stage 2 of `project_and_finalize_streams` across `min(M, 16)` CTAs in the existing `(splits=40, M_tiles)` grid (eliminating the single-CTA `range(live_m)` serial bottleneck from `0634` and hoisting `weight` + `scale` + `base` loads across all `live_m` rows in parallel), and raise `FINALIZE_FUSE_MAX_ROWS` from `1` to `16` so single-tile speculative decode (`M = 6` at `c=1`) and small decode batches (`M <= 16`) execute `project_streams` + `finalize` in a single kernel launch.

## Precision & Bit-Identity Constraint

Per strict owner constraint:
- **Zero precision reduction anywhere**: `0731-mhc-project-finalize.patch` preserves 100% of the exact FP32/BF16 arithmetic, exact TF32 `3x19` split decomposition (`r6h` load-time split), fixed split-order reduction (`for s_idx in T.serial(splits)`), Sinkhorn-Knopp (`20` iterations), and RMSNorm (`1e-20`) of the standalone `project_streams` + `finalize` two-kernel reference.
- No weights (`fn`), activations (`x`), intermediates (`P, S, C, Q`), or outputs (`x_out, H_post, H_res, y_norm, inv_rms`) are quantized, truncated, or reordered.

## Kernel Design (`0731-mhc-project-finalize.patch`)

1. **Root Cause in `0634-mhc-fuse-project-finalize.patch`**:
   - `0634` added `project_and_finalize_streams` with grid `(splits=40, M_tiles)` (`threads=128`), where CTA `s == 0` waited for all `40` split CTAs of tile `bm` to increment `counters[bm]` and then looped `for r in T.serial(live_m)` on a **single CTA** (`s == 0`).
   - Because each row loads `5,120` BF16 elements (`10 KiB`) of `x` and `y_norm` and runs Sinkhorn + RMSNorm, serializing `live_m` rows on 1 CTA caused latency to scale linearly with `M` (`13.06 µs` at `M = 1` -> `51.51 µs` at `M = 16` on GB10), forcing `0634` to cap `FINALIZE_FUSE_MAX_ROWS = 1`.
2. **Parallel-CTA Stage 2 in `0731-mhc-project-finalize.patch`**:
   - Stage 1 (`project_streams` partial compute into `P_partial, S_partial, C_partial, Q_partial`) is unchanged across all `(splits=40, M_tiles)` CTAs.
   - Each CTA `(s, bm)` executes `T.call_extern("int", "__threadfence")`, then thread 0 performs `T.atomic_add(counters[bm], 1, return_prev=True)`.
   - The last split CTA (`prev == splits - 1`) resets `counters[bm] = 0`, executes `__threadfence()`, and sets `done_flag[bm] = 1` via `st.global.cg.s32`.
   - Every CTA with `s < live_m` (up to `16` CTAs per `bm` tile, since `splits = 40 >= 16`) spins on `ld.global.cg.s32(&done_flag[bm]) != 0` on thread 0 (`__syncthreads()`), after which CTA `s` processes exclusively row `m = bm * block_M + s`:
     - Preloads `scale`, `base`, and all `5,120` BF16 elements of `weight` into registers (`wv[step * VEC + v]`) in parallel across all `live_m` CTAs.
     - Reduces `P_partial[t, m, :]`, `S_partial[t, m, :]`, `C_partial[t, m, :]`, and `Q_partial[t, m, :]` in exact fixed split order `for t in T.serial(splits)`.
     - Computes `H_pre`, `H_post`, and `20`-iteration Sinkhorn `H_res` on thread 0, broadcasts via shared memory, mixes `n = 4` streams into `y_acc`, computes RMSNorm + `x_out` (for `post_pre`), and writes `y_norm`.
   - After all `live_m` CTAs complete row `m`, they release `done_flag[bm]` via a second release counter (`prev_done == live_m - 1` sets `done_flag[bm] = 0`), leaving both `counters[bm] == 0` and `done_flag[bm] == 0` before kernel exit so zero `memset` launch is required between calls.

## Phase 1: CPU & SM121 Compiler Verification (`sp8`)

- **Command**:
  ```bash
  systemd-run --user --scope -p CPUQuota=100% -p MemoryMax=4G \
    docker run --rm --entrypoint python3 \
    -v /home/chris/spark-lab/mhc-project-finalize/vllm:/vllm-workspace/src \
    -v /home/chris/spark-lab/mhc-project-finalize/repo/experiments/2026-11-06-mhc-project-finalize:/exp \
    localhost/vllm-ds41f:candidate-1107-20261105 \
    /exp/bench_mhc_project_finalize.py --mode cpu --out /exp/cpu-verification.json
  ```
- **Results** (`cpu-verification.json`):
  - **CPU Golden Reference Bit-Identity**: **28/28 passed** (100% `int16`/`int32`/`uint8` bit-identical across `pre`, `broadcast`, `post_pre` with `w=1, 3, 4`, `with_aux=False, True`, `tokens=1, 6, 15, 16, 32, 48`, and both `finite` and `nonfinite` `NaN`/`±Inf`/`-0.0` inputs).
  - **SM121 (`sm_121a`) Kernel Compilation & Static Resource Audit (`ptxas -v`)**:
    | Kernel | Mode | `with_aux` | `w` | Regs / Thread | Shared Mem (B) | Stack / Spill |
    |---|---|---|---:|---:|---:|---:|
    | `project_and_finalize_streams` (`0731`) | `pre` | `False` | `1` | 150 | 57,344 | 0 / 0 |
    | `project_and_finalize_streams` (`0731`) | `pre` | `True` | `1` | 148 | 57,344 | 0 / 0 |
    | `project_and_finalize_streams` (`0731`) | `broadcast` | `False` | `1` | 207 | 57,344 | 0 / 0 |
    | `project_and_finalize_streams` (`0731`) | `post_pre` | `False` | `1` | 248 | 57,344 | 0 / 0 |
    | `project_and_finalize_streams` (`0731`) | `post_pre` | `True` | `1` | 250 | 57,344 | 0 / 0 |
    | `project_and_finalize_streams` (`0731`) | `post_pre` | `False` | `3` | 248 | 57,344 | 0 / 0 |
    | `project_and_finalize_streams` (`0731`) | `post_pre` | `True` | `3` | 250 | 57,344 | 0 / 0 |
    | `project_and_finalize_streams` (`0731`) | `post_pre` | `False` | `4` | 248 | 57,344 | 0 / 0 |
    | `project_and_finalize_streams` (`0731`) | `post_pre` | `True` | `4` | 250 | 57,344 | 0 / 0 |
    | `project_streams` (standalone) | `pre` | `False` | `1` | 124 | 57,344 | 0 / 0 |
    | `project_streams` (standalone) | `post_pre` | `True` | `3` | 212 | 57,344 | 0 / 0 |
    | `finalize` (standalone) | `pre` | `False` | `1` | 133 | 40 | 0 / 0 |
    | `finalize` (standalone) | `post_pre` | `True` | `3` | 210 | 200 | 0 / 0 |

## Phase 2: 2-Boot Standalone GPU Verification & Benchmark on `sp8` (NVIDIA GB10)

### 1. Exact Commands Run (Under `sp8:~/spark-hold.json`)

```bash
# Full pytest GPU suite (202 tests)
docker run --rm --gpus all --ipc=host --entrypoint pytest \
  -v /home/chris/spark-lab/mhc-project-finalize/vllm:/vllm-workspace/src \
  localhost/vllm-ds41f:candidate-1107-20261105 \
  /vllm-workspace/src/tests/kernels/test_deepseek_v41_tilelang_mhc.py -q

# Boot 1 fresh container GPU verification + benchmark
docker run --rm --gpus all --ipc=host --entrypoint python3 \
  -v /home/chris/spark-lab/mhc-project-finalize/vllm:/vllm-workspace/src \
  -v /home/chris/spark-lab/mhc-project-finalize/repo/experiments/2026-11-06-mhc-project-finalize:/exp \
  localhost/vllm-ds41f:candidate-1107-20261105 \
  /exp/bench_mhc_project_finalize.py --mode gpu --out /exp/gpu-boot1.json

# Boot 2 fresh container GPU verification + benchmark
docker run --rm --gpus all --ipc=host --entrypoint python3 \
  -v /home/chris/spark-lab/mhc-project-finalize/vllm:/vllm-workspace/src \
  -v /home/chris/spark-lab/mhc-project-finalize/repo/experiments/2026-11-06-mhc-project-finalize:/exp \
  localhost/vllm-ds41f:candidate-1107-20261105 \
  /exp/bench_mhc_project_finalize.py --mode gpu --out /exp/gpu-boot2.json
```

### 2. GPU Bit-Identity Receipts

- **`pytest-mhc-gpu.txt`**: **`202 passed in 165.50s (0:02:45)`** on NVIDIA GB10.
- **`gpu-boot1.json` & `gpu-boot2.json` (`correctness_cases`)**: **`49/49 passed`** on both boots (`pre`, `broadcast`, `post_pre` with `w=1, 3, 4`, `with_aux=False, True`, `tokens=1, 6, 15, 16, 32, 48`, finite & non-finite `NaN`/`±Inf`/`-0.0`), verifying 100% bit-exact `int16`/`int32`/`uint8` tensor equality between two-launch `project_streams + finalize` and single-launch `project_and_finalize_streams`, plus `counters == 0` and `done_flag == 0` after every call.

### 3. 2-Boot GPU Latency Results (`gpu-boot1.json` & `gpu-boot2.json`)

#### A. Decode & Multi-Tile Latency (`M = 1..96`, CUDA Graph µs/call, `warmup=50, iters=200`, `post_pre`, `with_aux=True`, `w=3`)

| Rows `M` | Split (`project + finalize`) Boot 1 / Boot 2 (µs) | `Base0634` Fused (1-CTA) Boot 1 / Boot 2 (µs) | `Cand0731` Fused (parallel-CTA) Boot 1 / Boot 2 (µs) | `Cand0731` vs `Base0634` | `Cand0731` vs `Split` | `Base0634` vs `Split` |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8.77 / 8.71 | 13.06 / 13.86 | 17.28 / 17.22 | +3.79 µs (0.78x) | +8.51 µs | +4.72 µs |
| 2 | 8.89 / 8.82 | 12.60 / 12.51 | 18.24 / 19.32 | +6.23 µs (0.67x) | +9.92 µs | +3.70 µs |
| 4 | 8.96 / 9.19 | 18.13 / 18.08 | 19.40 / 19.41 | +1.30 µs (0.93x) | +10.33 µs | +9.03 µs |
| 6 | 8.93 / 8.84 | 23.34 / 23.31 | 20.48 / 20.36 | -2.91 µs (1.14x) | +11.54 µs | +14.44 µs |
| 8 | 8.89 / 8.86 | 29.28 / 28.73 | 22.82 / 21.61 | -6.79 µs (1.31x) | +13.34 µs | +20.12 µs |
| 12 | 9.52 / 9.46 | 40.20 / 40.64 | 22.81 / 21.54 | -18.25 µs (1.82x) | +12.68 µs | +30.93 µs |
| 15 | 9.70 / 9.63 | 52.41 / 52.07 | 22.05 / 21.70 | -30.37 µs (2.39x) | +12.21 µs | +42.58 µs |
| 16 | 9.77 / 9.63 | 51.33 / 51.51 | 22.09 / 22.05 | -29.34 µs (2.33x) | +12.37 µs | +41.72 µs |
| 17 | 14.11 / 14.20 | 51.59 / 51.67 | 27.61 / 26.76 | -24.44 µs (1.90x) | +13.04 µs | +37.48 µs |
| 24 | 14.42 / 14.35 | 52.38 / 52.66 | 28.16 / 27.21 | -24.84 µs (1.90x) | +13.30 µs | +38.14 µs |
| 32 | 15.42 / 14.75 | 58.26 / 57.00 | 28.52 / 27.72 | -29.51 µs (2.05x) | +13.03 µs | +42.54 µs |
| 36 | 19.99 / 19.46 | — | 42.09 / 42.12 | — | +22.38 µs | — |
| 40 | 19.70 / 19.47 | — | 41.56 / 41.64 | — | +22.02 µs | — |
| 48 | 20.45 / 20.39 | — | 42.85 / 43.10 | — | +22.56 µs | — |
| 64 | 25.31 / 25.31 | — | 49.14 / 50.30 | — | +24.41 µs | — |
| 96 | 33.44 / 33.72 | — | 70.07 / 71.62 | — | +37.27 µs | — |

#### B. Prefill Chunk Breakdown (`M = 2048, 6144, 16384`, Eager µs/call, `post_pre`, `with_aux=True`, `w=3`)

| Rows `M` | `project_only` Boot 1 / Boot 2 (µs) | `finalize_only` Boot 1 / Boot 2 (µs) | `Split` (`project + finalize`) Boot 1 / Boot 2 (µs) | PDL Overlap Saved (`proj + fin - split`) |
|---:|---:|---:|---:|---:|
| 2048 | 1008.2 / 1002.7 | 287.7 / 301.7 | 1274.6 / 1297.2 | +14.3 µs |
| 6144 | 2945.5 / 3157.1 | 881.7 / 907.5 | 3774.6 / 3810.8 | +153.3 µs |
| 16384 | 7671.7 / 7682.0 | 2332.3 / 2344.2 | 9928.9 / 9935.8 | +82.8 µs |

## Diagnosis & Conclusion

1. **Parallel-CTA Stage 2 (`Cand0731`) vs Serial-CTA Stage 2 (`Base0634`)**:
   - Parallelizing Stage 2 across `live_m` CTAs eliminates the `O(live_m)` single-CTA serial bottleneck of `Base0634`, speeding up `project_and_finalize_streams` from **`51.33 / 51.51 µs` to `22.09 / 22.05 µs` at `M = 16` (`2.33x` speedup)** and from **`23.34 / 23.31 µs` to `20.48 / 20.36 µs` at `M = 6` (`c=1` speculative decode)**.
   - At `M = 1..4`, `Cand0731` is `1.3..4.5 µs` slower than `Base0634` (`17.25 µs` vs `13.46 µs` at `M = 1`) because hoisting `wv` (`56` BF16 = `28` regs) and `yv` (`56` BF16 = `28` regs) across the fused kernel increases register allocation in Stage 1 (`post_pre` uses `250` regs/thread vs `212` in standalone `project_streams`) and adds a second release counter (`done_flag`).

2. **Why Two-Launch `project_streams` + `finalize` (`Split`) Beats Single-Launch `project_and_finalize_streams` on GB10 Across Every `M = 1..96` (Including `M = 1`!)**:
   - On GB10 (SM121, **48 SMs**), `project_streams` at `M <= 16` launches `splits = 40` CTAs (`56 KiB` shared memory/CTA = 1 CTA/SM), leaving **8 SMs idle**.
   - In the two-launch `Split` path (`project_streams` + `finalize`), `project_streams` issues `pdl.chain()` and `finalize` places `pdl.chain()` **after** preloading `scale`, `base`, and all `5,120` BF16 elements of `weight` into registers (`wv[step * VEC + v]`). Hardware Programmatic Dependent Launch (PDL) spawns `finalize` onto the **8 idle SMs concurrently with `project_streams`**, preloading `weight`, `scale`, and `base` while `project_streams` is still computing partials!
   - Consequently, `Split` (`8.77 / 8.71 µs` graph at `M = 1`, `8.93 / 8.84 µs` at `M = 6`, `9.77 / 9.63 µs` at `M = 16`) hides nearly all of `finalize`'s weight/scale/base load time behind `project_streams`.
   - By contrast, fusing `finalize` into `project_and_finalize_streams`:
     1. Prevents PDL from preloading `weight` (`10 KiB`), `scale`, and `base` on idle SMs while Stage 1 runs,
     2. Inflates Stage 1 register pressure (`250` regs vs `212` regs in standalone `project_streams`), slowing Stage 1 itself, and
     3. Requires every CTA to execute `membar.gl` (`__threadfence()`) + global `atomicAdd` + `membar.gl` before Stage 2 can read `P, S, C, Q` from global memory.

3. **Prefill (`M = 2048, 6144, 16384`)**:
   - At `M = 6144` (TP3 chunk) and `M = 16384` (TP4 chunk), `project_only` takes `2945.5 / 3157.1 µs` (`6144`) and `7671.7 / 7682.0 µs` (`16384`), while `finalize_only` takes `881.7 / 907.5 µs` (`6144`) and `2332.3 / 2344.2 µs` (`16384`).
   - Hardware PDL overlap between `project_streams` and `finalize` already hides `52..90 µs/call` at prefill (`Split = 3774.6 / 3810.8 µs` at `6144`, `9928.9 / 9935.8 µs` at `16384`), and multi-tile `project_and_finalize_streams` at `M = 96` is `2.1x` slower (`70.85 µs` vs `33.58 µs`) due to register pressure and per-tile global atomic barriers.

4. **Recommendation**:
   - **Do NOT promote `0731-mhc-project-finalize.patch` (`FINALIZE_FUSE_MAX_ROWS = 16`)**: while 100% bit-identical (`202/202` pytest + `49/49` harness cases) and `2.33x` faster than `Base0634` at `M = 16`, it regresses `M = 2..16` against the two-launch `project_streams` + `finalize` path (`20.42 µs` vs `8.89 µs` at `M = 6` `c=1` speculative decode).
   - **Actionable Finding for Candidate `1107`**: Even at `M = 1` (where Candidate `1107` currently uses `Base0634` `project_and_finalize_streams` via `FINALIZE_FUSE_MAX_ROWS = 1`), the two-launch `project_streams` + `finalize` path (`8.77 / 8.71 µs`) is **`4.3..5.1 µs/call` (`345..410 µs` per 40-layer pass) faster** than `Base0634` `project_and_finalize_streams` (`13.06 / 13.86 µs`). Setting `FINALIZE_FUSE_MAX_ROWS = 0` (always using the two-launch PDL path) is strictly faster across all `M >= 1` while keeping 100% bit-identical outputs.
