# 2026-11-06: Parallelized Multi-Token mHC `project_and_finalize_streams` (`M <= 16`)

## Summary & Causal Variable

- **Base deployment commit**: `0b3e98934f0aa7c7c251b79dae3ac148f869304e` (Candidate `1107`, vLLM `0001..0730`, patch head `c47b54ae471a24b830e3515212b8c0bf5ee14604`, tree `13e4781d19bc4a18b0d3d4daad7921c465e3355e`)
- **Candidate vLLM patch**: `vllm/0731-mhc-project-finalize.patch` (vLLM `0001..0731`, patch head `770e2c6b5a331d75c1dbe646a91e80086008d2dc`, tree `08d52d012ce22f6278440f69e0851a4c0f43cd15`, 616 patches)
- **Intended causal variable**: Parallelize Stage 2 (`finalize`) of `project_and_finalize_streams` across the last `live_m` (`<= 16`) arriving split CTAs of each token tile, align Stage 1 and Stage 2 with the `0634` optimizations in standalone `project_streams` and `finalize`, and raise `FINALIZE_FUSE_MAX_ROWS` from `1` to `16` (`BLOCK_M`).

### Root Cause in Candidate 1107

When `0634-dsv41-tilelang-mhc-opt.patch` (`b15bb8f9b`) merged standalone `project_streams` and `finalize` optimizations, `project_and_finalize_streams` was left with `FINALIZE_FUSE_MAX_ROWS = 1` and the unoptimized `0606` body:
1. Stage 1 called `load_split_tf32(Fn, ...)` *after* the stream FMA fold rather than before it, forfeiting latency hiding between global `Fn` loads and register stream arithmetic.
2. Stage 2 had a single split CTA loop serially over all `16` rows in the tile (`for pair in T.serial(block_M // 2)` and `for ti in T.serial(block_M)`), with non-unrolled `T.serial(splits)` and `T.serial(steps)` loops that reloaded `weight` from global memory on every token and step (`5` depth-2 `*112` loops in IKET, `events_per_warp_bound = 124` at `iters=2`).
3. `pdl.chain()` triggered downstream kernels at kernel entry instead of after Stage 1, allowing downstream CTAs to compete for SMs during the 40-CTA TF32 projection.

### Candidate Design (`0731-mhc-project-finalize.patch`)

1. **Stage 1 hoisted `load_split_tf32(Fn, ...)`**: Issues `load_split_tf32` before the stream FMA loop (matching standalone `project_streams`), preserving the exact 3-TF32-part decomposition (`& 0xFFFFE000`) and 3 FP32 tensor-core GEMMs.
2. **Split lock-free `mhc_claim` + `mhc_wait_splits` (`nanosleep.u32 32` backoff)**:
   - After `mhc_fence()` + `__syncthreads()`, Thread 0 calls `mhc_claim(counter, splits, live_m)`.
   - The first `splits - live_m` (`40 - live_m`) split CTAs receive `rank = -1`, execute `pdl.trigger()`, and exit immediately so their SMs are freed.
   - The last `live_m` (`1..16`) split CTAs receive `rank = 0 .. live_m - 1`, each claiming one token (`tok = by * block_M + rank`) to finalize in parallel across `live_m` SMs.
   - When `live_m == 1` (`M = 1`), `mhc_claim` resets `counter = 0` immediately via `st.global.release.gpu.b32` and `mhc_wait_splits` executes `__threadfence(); return;` with zero spin-wait and zero second atomic.
   - When `live_m > 1`, `rank == 0` (`arrived == splits - 1`) already observes all `splits` arrivals and proceeds directly to `__threadfence()`, while `rank > 0` spin-loads `counter` with `ld.global.acquire.gpu.b32` and `nanosleep.u32 32` until `counter >= splits`, executes `__threadfence()`, and the last finalizer (`done == splits + live_m - 1`) resets `counter = 0`.
3. **Late `pdl.trigger()` and pre-wait `scale`/`base`/`weight` preload**:
   - `pdl.wait()` runs at kernel entry and `pdl.trigger()` runs immediately after `mhc_claim`.
   - While Thread 0 executes `mhc_wait_splits`, Warp 0 (`tx < 24`) preloads `scale` and `base` into registers and Warps 1–3 (`tx >= 32`) preload all `5,120` BF16 elements of `weight` into `wv[step * VEC + v]`.
4. **Stage 2 unrolled 1-token `finalize` in fixed split order**:
   - Warp 0 (`tx < 24`) reduces `S[0..splits-1, tok, 0]` and `P[0..splits-1, tok, 0, mix_idx]` with `T.unroll(splits)` in fixed split order `0..39` (lanes `0..15` on `comb`, lanes `16..19` on `pre`, lanes `20..23` on `post`) and runs the 20-step Sinkhorn butterfly.
   - Warps 1–3 (`tx >= 32`) batch-load `C[tok, :]` into `yv[step * VEC + v]` before reducing `Q[0..splits-1, tok]` with `T.unroll(splits)`, then normalize and run `cast_block_mxfp8`.
5. **Single-wave `FINALIZE_FUSE_MAX_ROWS = 16` and pre-allocated `_counters`**:
   - At `M <= 16` (`tiles = 1`), all `40` split CTAs fit in a single wave on GB10's 48 SMs, covering `M = 1` decode, speculative verification (`M = 6, 8, 12, 16`), and draft steps in one kernel launch per sublayer entry (saving 80 kernel launches per 40-layer forward pass).
   - `TileKernelsMHC.declare_kernels` pre-allocates `_counters` up to `tilelang.cdiv(max(FINALIZE_FUSE_MAX_ROWS, max_num_batched_tokens), BLOCK_M)` before CUDA graph capture.

## Precision & Bit-Identity Guarantee

- **Zero precision reduction**: The 3-TF32-part `Fn` split (`load_split_tf32`), FP32 stream FMA sequence, BF16 intermediate rounding of `R_out`, `C`, and `Aux`, fixed `0..splits-1` reduction order of `S`, `P`, and `Q`, FP32 Sinkhorn butterfly, FP32 RMSNorm, and `cast_block_mxfp8` are 100% bit-identical (`torch.equal`) to `project_streams` + `finalize` across all finite and non-finite (`NaN`, `±Inf`, `-0.0`) inputs, regardless of which split CTA finishes last.

## Phase 1: CPU & SM121 Compiler Verification (`sp8`)

Executed on `sp8` (`spark-0016`) under `CPUQuota=100%` (`--cpus=1`) and `MemoryMax=4G` (`--memory=4g --memory-swap=4g`) with `--network=none` and `CUDA_VISIBLE_DEVICES=`:

```bash
docker run --rm --runtime=runc --network=none --cpus=1 --memory=4g --memory-swap=4g \
  --entrypoint python3 \
  -e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void \
  -v /home/chris/spark-lab/mhc-project-finalize/vllm/vllm/models/deepseek_v4_1/tilelang/mhc.py:/opt/spark-ds41f/candidate/vllm/vllm/models/deepseek_v4_1/tilelang/mhc.py:ro \
  -v /home/chris/spark-lab/mhc-project-finalize/vllm/tests/kernels/test_deepseek_v41_tilelang_mhc.py:/opt/spark-ds41f/candidate/vllm/tests/kernels/test_deepseek_v41_tilelang_mhc.py:ro \
  -v /home/chris/spark-lab/mhc-project-finalize/repo/experiments/2026-11-06-mhc-project-finalize:/out \
  spark-ds41f:candidate-38 /out/bench_mhc_project_finalize.py --cpu-verify --output /out/cpu-verification.json
```

### SM121 Compilation (`nvcc -cubin -arch=sm_121a -O3 -Xptxas=-v`), PDL & IKET Bounds

| Mode | `world` | `with_aux` | Registers | Barriers | Register Spills | PDL Violations | IKET Sites (`<= 30`) | IKET `iters=1` Bound (`<= 60`) | IKET `iters=2` Bound (vs `124` base) |
| --- | ---: | :---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `pre` | 1 | `False` | 178 | 1 | `0B store / 0B load` | 0 | 28 | 56 | 84 |
| `broadcast` | 1 | `False` | 122 | 1 | `0B store / 0B load` | 0 | 28 | 56 | 72 |
| `post_pre` | 1 | `False` | 250 | 1 | `0B store / 0B load` | 0 | 29 | 58 | 88 |
| `post_pre` | 1 | `True` | 248 | 1 | `0B store / 0B load` | 0 | 30 | 60 | 90 |
| `post_pre` | 2 | `False` | 250 | 1 | `0B store / 0B load` | 0 | 29 | 58 | 88 |
| `post_pre` | 2 | `True` | 248 | 1 | `0B store / 0B load` | 0 | 30 | 60 | 90 |
| `post_pre` | 3 | `False` | 250 | 1 | `0B store / 0B load` | 0 | 29 | 58 | 88 |
| `post_pre` | 3 | `True` | 248 | 1 | `0B store / 0B load` | 0 | 30 | 60 | 90 |
| `post_pre` | 4 | `False` | 250 | 1 | `0B store / 0B load` | 0 | 29 | 58 | 88 |
| `post_pre` | 4 | `True` | 248 | 1 | `0B store / 0B load` | 0 | 30 | 60 | 90 |

- All 28 CPU bit-exact oracle cases (`H=5120, block_H=128` across `pre`, `broadcast`, `post_pre` `w=3,4`, decode `rows = 1..96`, prefill chunks `rows = 6144, 16384`, and non-finite `NaN`/`±Inf`/`-0.0` inputs under random split CTA arrival permutations) passed (`all_cpu_oracle_bits_identical: true`, `all_counters_reset_zero: true`); see [cpu-verification.json](cpu-verification.json).

## Phase 2: GPU Microbenchmark & 2-Boot Serving Screen Plan

When scheduled for a GPU window:
1. Run `pytest tests/kernels/test_deepseek_v41_tilelang_mhc.py` and `bench_mhc_project_finalize.py --gpu-bench --output gpu-benchmark.json` to verify GPU bit-identity and measure kernel latency across `rows = 1..96` decode and `6144` (TP3) / `16384` (TP4) prefill chunks.
2. Build `spark-ds41f:mhc-project-finalize` and run a 2-boot serving screen against recorded Candidate `1107` / `candidate-40` references.
