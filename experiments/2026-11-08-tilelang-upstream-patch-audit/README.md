# TileLang upstream commit and bugfix PR audit (post-`2123e412ce22`)

## Context and goal

`spark-ds41f:1107` (`manifests/baselines/2026-11-07-spark-ds41f-1107.json`) pins
`tile-ai/tilelang` at `2123e412ce2217b324c50c4ec967213906456ac5` (2026-11-04) with the
13-patch stack in `experiments/2026-10-26-spark-ds41f-candidate/1107/tilelang/series`
(tree `e2b001828fd4e38e6a2eee61f818d5083cb1335f`). Since `2123e412ce22`, upstream
`tile-ai/tilelang` has merged 16 commits on `main` and carries several open bugfix pull
requests touching `T.Parallel` layout inference, loop vectorization, warp specialization
(`ProducerConsumerWarpSpecialized`), TF32 rounding modes, per-launch kernel names,
host-mapped buffers, and index promotion.

This experiment audits all 16 merged upstream commits and 7 open bugfix PRs against the
`spark-ds41f:1107` TileLang kernel suite (214 Python files across `vllm` and
`tile_kernels`, and 33 compiled kernel variants), builds and tests the candidate patch
stack on `sp1` (`NVIDIA GB10`, `sm_121a`) across two independent container boots, and
stages the 6 selected patches (`0014..0019`) in [tilelang/series](tilelang/series).

## Audit of upstream commits and open PRs

### 1. Merged on `tile-ai/tilelang` `main` since `2123e412ce22`

| PR / Commit | Summary | Disposition | Rationale for `spark-ds41f` |
| --- | --- | --- | --- |
| `#3714` (`c99577fd91`) | `[CUDA] Support round-to-nearest-even TF32 casts` | **Stage (`0014`)** | Adds `round="rn"\|"rz"` to `T.cast(..., T.tfloat32)`, lowering `round="rn"` to `tl::cast_rn_tf32` (`cvt.rn.tf32.f32`). Unlocks 2-split `round="rn"` TF32 decomposition in `mhc.project_streams` (see below). |
| `#3759` (`71a7c3114e`) | `[Transform] Preserve vector widths for scalar condition casts` | **Stage (`0015`)** | Prevents scalar casts inside `IfThenElse`/`Select`/`if_then_else` conditions (or scalar `Bind`s used only in conditions) from shrinking vector load/store widths in `VectorSizePlanner` or triggering spurious buffer decoupling in `DecoupleTypeCast`. |
| `#3802` (`fd5305807b`) | `[Feature][CUDA] Add explicit host-mapped buffer annotations` | **Stage (`0016`)** | Adds `T.annotate_host_mapped_buffers(...)` and validates zero-copy host-pinned pointers via `cuPointerGetAttribute` in both `tvm_ffi` and `cython` adapters. |
| `#3816` (`78da734369`) | `[CUDA] Support names for individual kernel launches` | **Stage (`0017`)** | Adds `T.Kernel(..., name="...")` (`launch_name` attribute through `SplitHostDevice`). Complements vLLM patch `0118` (which names `@T.prim_func` functions `<file>_<what>`) when a `@T.prim_func` contains multiple `T.Kernel` launches. |
| `#3685` / `#3663` (`438e2db04f`) | `Fix CUDA ProducerConsumerWS test failures` (`#3663` prelude/branch fix) | **Stage (`0019`, CUDA portion)** | Keeps shared-prelude statements at their original control-flow position instead of hoisting them across sibling statements, and keeps producer-free branches guarded to the consumer thread range. (The Ascend PTO portions of `#3685` are omitted.) |
| `#3868` (`1cf5a1f004`) | `[BugFix][CuTeDSL] Synchronize sync_grid on the per-launch grid workspace` | Skip | CuTeDSL backend only; `spark-ds41f` uses the `tvm_ffi` / `cython` CUDA backends. |
| `#3551` (`a8f3d3bcfe`) | `[Bugfix][CuTeDSL] Preserve signed integer truncation` | Skip | CuTeDSL backend only. |
| `#3752` (`61f77f99dd`), `#3760` (`cdbf7ba470`), `#3805` (`65997cb164`), `#3845` (`5e8b7bc235`) | Ascend backend / CI commits | Skip | Ascend NPU backend only. |
| `#3534` (`f2d5e8cd6b`), `#3772` (`da9bc01092`), `#3784` (`e71c77f0be`), `#3807` (`bbdfac4103`), `#3812` (`93678556ab`) | Metal MPS, CI, skills, FlashAttention weight-stationary example, GemmV1 docs | Skip | Non-CUDA or example/CI/doc-only changes. |

### 2. Open bugfix PRs on `tile-ai/tilelang`

| PR | Summary | Disposition | Rationale for `spark-ds41f` |
| --- | --- | --- | --- |
| `#3873` (Fixes `#3870`) | `[BugFix][Layout] Run the atomics and read-modify-writes of a replicated loop once` | **Stage (`0018`)** | When a `T.Parallel` loop has `extent < threads` and writes a register fragment (`store_fragment_buffers` non-empty), layout inference replicates the loop across thread groups. Previously any shared/global atomic or read-modify-write in the same loop executed on every replica, multiplying updates by the replication factor. `GuardReplicaWrites` restricts shared/global writes to replica 0 and rejects read-before-update races across replicas. |
| `#3869` (Fixes `#3664`) | `[BugFix][WarpSpecialize] Split the partitions outside an if around the pipeline` | **Stage (`0019`)** | When a warp-specialized `T.Pipelined` loop is the sole statement inside a thread-invariant `IfThenElse` whose condition does not read buffers written by the branch, splits producer/consumer partitions outside the `if` so `setmaxnreg` instructions execute uniformly at the partition root. |
| `#3883` (Fixes `#3842`) | `[BugFix][WS] Decline warp specialization on loop-carried post-pipeline updates` | **Stage (`0019`)** | Declines warp specialization when a `T.Pipelined` loop inside an outer serial loop has post-pipeline local buffer writes that feed pre-pipeline producer reads on the next outer iteration without an inter-iteration barrier. |
| `#3580` | `[BugFix][TIR] Synchronize predicates with promoted indices` | **Defer / Do not stage** | Tested on `sp1` against all 33 `spark-ds41f:1107` kernels. Because `PromoteExprToInt64` promotes every integer expression inside any enclosing `IfThenElse` condition whenever a nested `BufferLoad`/`BufferStore` has a 64-bit flattened offset, `#3580` alters 4 existing kernels (`mhc.project_streams_post_pre`, `mhc.project_streams_world3`, `router.moe_topk_gate`, `worker.grammar_bitmask`) by widening bounded 32-bit guards such as `((threadIdx.x & 31) < num_physical_topk)` and `((blockIdx.y * 16 + (threadIdx.x >> 3)) < M)` to 64-bit integer comparisons. Our large-stride kernels (`sample.*`, `spec_decode.*`, `router.py`) already write explicit `T.int64` index expressions where needed, and upstream maintainer review (`KellyFrog`) flagged the 505-line pass complexity. |
| `#3882` (Fixes `#3881`) | `[BugFix][JIT] Raise a cython kernel's dynamic shared memory limit on every device it launches on` | Skip | Multi-GPU in-process device switching only; each DGX Spark node has a single GB10 GPU (`cuda:0`), and `spark-ds41f` uses `tvm_ffi` by default. |
| `#3867` / `#3872` | `[BugFix][JIT] Launch on the inputs' device when it is not the current device` / `Restore the caller's current device after a tvm_ffi call` | Skip | Multi-GPU in-process caller/input device mismatch only; adds per-launch device/stream queries that are unnecessary on single-GPU-per-node `cuda:0` ranks. |

## Kernel suite audit and generated CUDA source comparison

Scanning all 214 TileLang-related Python files in `spark-ds41f:1107` (`/opt/spark-ds41f/candidate/vllm` and `tile_kernels`) identified:

- **Replicated `T.Parallel` loops (`#3873`)**: `block_shortlist.py` (`summarize_blocks`, `for c in T.Parallel(64)` on 128 threads), `query_reuse.py` (`prepare_queries`, `for c in T.Parallel(64)` on 128 threads), and `gemm.py` (`_reference_cast`, `for r in T.Parallel(16)` on 128 threads) use `T.Parallel` loops with `extent < threads`. Existing kernels either keep shared/global stores in a separate loop or perform idempotent writes (`S[m0 + r, k // 32] = ...`), so their generated CUDA code is unchanged; `0018` (`#3873`) prevents silent multi-replica corruption if an atomic or read-modify-write is placed in a replicated fragment loop.
- **Conditional casts in vectorized loops (`#3759`)**: `activation.py` (`swiglu_clamp`), `ced.py` (`copy_rows`), `mhc.py` (`finalize`), `tilelang_auxiliary.py` (`nan_count`), and `tilelang_rejection.py` (`verify`, `resample`) use `T.if_then_else` predicates inside `T.Parallel` / `T.vectorized` loops.
- **Warp-specialized `T.Pipelined` loops under control flow (`#3663`, `#3869`, `#3883`)**: `vllm/model_executor/kernels/mhc/tilelang_kernels.py` (`mhc_pre_big_fuse_tilelang`, `mhc_pre_big_fuse_with_norm_tilelang`, `mhc_pre_big_fuse_broadcast_with_norm_tilelang`) and `tile_kernels/mhc/pre_big_fuse_cuda.py` place `T.Pipelined` loops inside outer `if` blocks.
- **Per-launch kernel names (`#3816`) vs. vLLM `0118`**: vLLM patch `0118` renames `@T.prim_func` definitions from `main` to `<file>_<what>`. `0017` (`#3816`) adds optional `T.Kernel(..., name="...")` support for multi-launch `@T.prim_func` definitions without changing any single-launch symbol names.

### Generated CUDA source parity across 33 compiled kernels

Dumping `kernel.get_kernel_source()` across all 33 compiled `spark-ds41f:1107` kernels ([results.json](results.json) -> `kernel_source_audit_33`):

- **Baseline `1107` (`0001..0013`) vs. Candidate `1108` (`0001..0019`)**: **33 / 33 kernels produce byte-for-byte identical CUDA source (`0` diffs)**.
- **Baseline `1107` vs. Candidate + `#3580`**: 29 / 33 identical, 4 / 33 modified (`mhc.project_streams_post_pre`, `mhc.project_streams_world3`, `router.moe_topk_gate`, `worker.grammar_bitmask`) due to `#3580` widening bounded 32-bit `if` guards to `int64_t`.

## Follow-on evaluation unlocked by `#3714` (`0014`): 2-split `round="rn"` TF32 in `mhc.project_streams`

`TODO.md` records that dropping `F_low` in `mhc.project_streams` under bitwise truncation (`& 0xFFFFE000`, $11 + 11 = 22$ mantissa bits) missed by 1 BF16 ULP in 2 of 18 test cases (`M=33, mode=post_pre` and `M=128, mode=pre`).

With `0014` (`#3714`), `T.cast(f, T.tfloat32, round="rn")` rounds `high` to the nearest 11-bit TF32 value (`cvt.rn.tf32.f32`), so the signed remainder `rest = f - high` has magnitude $\le 2^{-11}|f|$ and `middle = T.cast(rest, T.tfloat32, round="rn")` captures the remaining 12 mantissa bits ($12 + 12 = 24$ bits in 2 splits instead of 3, saving 16 KB shared memory and 1 of 3 TF32 tensor-core GEMMs per stream):

- **Accuracy against FP64 (`flat.double() @ fn.double().T`)**: Across `M in {1, 4, 7, 16, 33, 37, 64, 128, 200, 256, 1024}`, 2-split `round="rn"` has **lower maximum relative error** (`2.08e-6 .. 3.11e-6`, $>9\times$ inside the `3e-5` contract bound) than 3-split truncation (`2.46e-6 .. 3.72e-6`).
- **End-to-end `TileKernelsMHC` parity**: Across all 42 `(H in {5120, 7168}, mode in {pre, post_pre, broadcast}, M in {1, 7, 16, 33, 64, 128, 200})` cases, `y` (BF16), `cast.values` (E4M3), and `cast.scales` (UE8M0) are **42 / 42 bit-exact (`0` BF16 ULP difference)** against 3-split truncation, while FP32 mixes (`post`, `comb`, `pre`) agree within `3.2e-6`. (Because `post`/`comb`/`pre` are returned in FP32, switching `mhc.py` from 3-split to 2-split `round="rn"` is kept as a separate follow-on candidate lever rather than bundled into the compiler patch series.)

## Two-boot verification and microbenchmarks on `sp1` (`NVIDIA GB10`, `sm_121a`)

Commands executed on `sp1` across two independent container boots (`Boot 1` and `Boot 2`):

```bash
docker run --rm --gpus all -v /tmp/work-tilelang-upstream:/work --entrypoint python3 spark-ds41f:1107 -u /work/run_boot_suite.py boot1 base /work/results/boot1-base.json
docker run --rm --gpus all -v /tmp/work-tilelang-upstream:/work --entrypoint python3 spark-ds41f:1107 -u /work/run_boot_suite.py boot1 cand /work/results/boot1-cand.json
docker run --rm --gpus all -v /tmp/work-tilelang-upstream:/work --entrypoint python3 spark-ds41f:1107 -u /work/run_boot_suite.py boot2 base /work/results/boot2-base.json
docker run --rm --gpus all -v /tmp/work-tilelang-upstream:/work --entrypoint python3 spark-ds41f:1107 -u /work/run_boot_suite.py boot2 cand /work/results/boot2-cand.json
```

### Test suite summary

| Suite | Boot 1 | Boot 2 |
| --- | --- | --- |
| 16 TileLang compiler & runtime test suites (`0001..0019`) | `293 passed, 12 skipped, 0 failed` (`112.63s`) | `293 passed, 12 skipped, 0 failed` (`111.91s`) |
| vLLM TileLang kernel regression suite (`activation`, `ced`) | `16 passed, 0 failed` (`53.54s`) | `16 passed, 0 failed` (`53.29s`) |
| Full `test_deepseek_v41_tilelang_mhc.py` (180 cases) | `180 passed, 0 failed` (`269.55s`) | Verified |

### Two-boot microbenchmark summary (`us`, 25 warmup + 200 timed iterations per point)

| Kernel / Workload | Base `1107` (Boot 1 / Boot 2, Mean) | Candidate `1108` 3-split (Boot 1 / Boot 2, Mean) | Candidate `1108` 2-split `round="rn"` `#3714` (Boot 1 / Boot 2, Mean) | 2-split `rn` vs. 3-split |
| --- | ---: | ---: | ---: | ---: |
| `swiglu_clamp` (`M=1, D=3072`) | `3.85 / 3.84` (`3.84`) | `3.88 / 3.88` (`3.88`) | — | — |
| `swiglu_clamp` (`M=16, D=3072`) | `3.71 / 3.73` (`3.72`) | `3.79 / 3.75` (`3.77`) | — | — |
| `swiglu_clamp` (`M=256, D=3072`) | `3.73 / 3.75` (`3.74`) | `3.76 / 3.78` (`3.77`) | — | — |
| `bf16_gemm` (`M=16, N=1024, K=1024`) | `13.46 / 11.02` (`12.24`) | `10.29 / 10.27` (`10.28`) | — | — |
| `bf16_gemm` (`M=256, N=1024, K=1024`) | `16.52 / 16.38` (`16.45`) | `16.34 / 16.35` (`16.35`) | — | — |
| `mhc.project_streams` (`pre, H=5120, M=64`) | `19.87 / 19.66` (`19.77`) | `19.68 / 19.64` (`19.66`) | `12.17 / 12.17` (`12.17`) | **+38.1%** |
| `mhc.project_streams` (`pre, H=5120, M=256`) | `72.33 / 68.09` (`70.21`) | `71.50 / 67.66` (`69.58`) | `40.44 / 42.56` (`41.50`) | **+40.4%** |
| `mhc.project_streams` (`pre, H=5120, M=1024`) | `327.30 / 319.41` (`323.36`) | `319.50 / 321.71` (`320.61`) | `259.16 / 257.68` (`258.42`) | **+19.4%** |
| `mhc.project_streams` (`post_pre, H=5120, M=64`) | `22.78 / 21.50` (`22.14`) | `21.58 / 21.58` (`21.58`) | `14.12 / 14.32` (`14.22`) | **+34.1%** |
| `mhc.project_streams` (`pre, H=7168, M=64`) | `25.03 / 26.52` (`25.77`) | `27.20 / 27.50` (`27.35`) | `16.50 / 16.35` (`16.43`) | **+39.9%** |
| `mhc.project_streams` (`pre, H=7168, M=256`) | `103.33 / 102.16` (`102.75`) | `99.73 / 102.30` (`101.02`) | `62.78 / 66.17` (`64.48`) | **+36.2%** |
| `mhc.project_streams` (`pre, H=7168, M=1024`) | `440.11 / 428.82` (`434.47`) | `432.13 / 433.31` (`432.72`) | `362.67 / 360.18` (`361.43`) | **+16.5%** |
| `mhc.project_streams` (`post_pre, H=7168, M=1`) | `14.47 / 9.91` (`12.19`) | `9.88 / 9.92` (`9.90`) | `5.93 / 5.91` (`5.92`) | **+40.2%** |
| `mhc.project_streams` (`post_pre, H=7168, M=4`) | `14.89 / 9.87` (`12.38`) | `9.86 / 9.83` (`9.85`) | `5.96 / 6.21` (`6.08`) | **+38.2%** |
| `mhc.project_streams` (`post_pre, H=7168, M=16`) | `14.16 / 10.73` (`12.44`) | `10.68 / 10.80` (`10.74`) | `7.58 / 7.53` (`7.56`) | **+29.6%** |
| `mhc.project_streams` (`post_pre, H=7168, M=64`) | `27.40 / 27.20` (`27.30`) | `27.52 / 30.48` (`29.00`) | `19.75 / 19.85` (`19.80`) | **+31.7%** |
| `mhc.project_streams` (`post_pre, H=7168, M=256`) | `190.05 / 183.03` (`186.54`) | `188.95 / 183.76` (`186.36`) | `177.49 / 177.53` (`177.51`) | **+4.7%** |
| `mhc.project_streams` (`post_pre, H=7168, M=1024`) | `715.57 / 716.02` (`715.80`) | `711.76 / 711.11` (`711.43`) | `691.05 / 696.28` (`693.67`) | **+2.5%** |

## Staged series (`tilelang/series`)

- Upstream base: `tile-ai/tilelang@2123e412ce2217b324c50c4ec967213906456ac5`
- Base `1107` patches (`0001..0013`) + staged patches (`0014..0019`):
  - [0014-CUDA-Support-round-to-nearest-even-TF32-casts.patch](tilelang/0014-CUDA-Support-round-to-nearest-even-TF32-casts.patch) (`#3714`)
  - [0015-Transform-Preserve-vector-widths-for-scalar-conditio.patch](tilelang/0015-Transform-Preserve-vector-widths-for-scalar-conditio.patch) (`#3759`)
  - [0016-CUDA-Add-explicit-host-mapped-buffer-annotations.patch](tilelang/0016-CUDA-Add-explicit-host-mapped-buffer-annotations.patch) (`#3802`)
  - [0017-CUDA-Support-names-for-individual-kernel-launches.patch](tilelang/0017-CUDA-Support-names-for-individual-kernel-launches.patch) (`#3816`)
  - [0018-Layout-Run-the-atomics-and-read-modify-writes-of-a-r.patch](tilelang/0018-Layout-Run-the-atomics-and-read-modify-writes-of-a-r.patch) (`#3873`)
  - [0019-WarpSpecialize-Guard-prelude-placement-outer-if-part.patch](tilelang/0019-WarpSpecialize-Guard-prelude-placement-outer-if-part.patch) (`#3663` + `#3869` + `#3883`)
- Resulting git tree: `836b3675cc7ddb393e43b0b3245c45ca98a0d675`
