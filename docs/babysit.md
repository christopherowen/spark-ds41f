# Upstream follow-ups

Issues and pull requests outside this repository that we opened, or that a
decision here depends on. Each row says what we are waiting for and what to do
when it changes. Add a row when we open or start relying on an upstream thread;
remove it once the change has landed here or no longer matters.

Check them all at once:

```sh
for ref in tile-ai/tilelang#3430 tile-ai/tilelang#3099 tile-ai/tilelang#3081 \
           tile-ai/tilelang#3286 tile-ai/tilelang#3287 tile-ai/tilelang#3288 \
           NVIDIA/open-gpu-kernel-modules#1269 NVIDIA/cutlass#3038 \
           eugr/spark-vllm-docker#4 christopherowen/dgx-spark-power-control#1; do
  gh api "repos/${ref%#*}/issues/${ref#*#}" --jq "\"$ref [\(.state)] \(.updated_at[:10]) \(.title)\""
done
```

## Waiting on upstream

| Thread | What | Our stake | Status (2026-10-05) | When it changes |
| --- | --- | --- | --- | --- |
| [tile-ai/tilelang#3287](https://github.com/tile-ai/tilelang/issues/3287), fix [#3288](https://github.com/tile-ai/tilelang/pull/3288) | Warp-specialized pipeline releases a stage before a synchronous (non-TMA, non-cp.async) shared copy is consumed | Ruled out as the cause of our decode-tile race on 2026-10-05: with #3288 our 16-row MXFP8 tiles still fail (1,222 of 3,600 runs, against 1,180 without it) and their generated CUDA is identical | Open, not ours | Nothing to do for us; see the planned issue below |
| [tile-ai/tilelang#3099](https://github.com/tile-ai/tilelang/pull/3099), [#3081](https://github.com/tile-ai/tilelang/pull/3081) | qqq-tao's SM120 MXFP8 (`kind::mxf8f6f4`) and MXFP4 block-scaled MMA, the same features as our `sm120-mxfp8-mxfp4` branch | Decides whether we open our MXFP8/MXFP4 PR or fold it into theirs | Both stale and conflicting with main. [We asked](https://github.com/tile-ai/tilelang/pull/3099#issuecomment-5998824942) on 2026-10-05 which they prefer | On a reply, open [our MXFP8/MXFP4 PR](#planned-pull-requests) or help fold it in. If there is no reply by 2026-10-19, open ours crediting theirs |
| [tile-ai/tilelang#3286](https://github.com/tile-ai/tilelang/pull/3286) | SM120 block-scaled register-A GEMM: scales loaded once per K atom | Its compact-scale path passes scale byte selector 0, which would break MXFP8/MXFP4 | Open. [We noted it](https://github.com/tile-ai/tilelang/pull/3286#issuecomment-5998825359) on 2026-10-05 | If it merges before our MXFP8/MXFP4 PR, that PR must pass the selectors through or limit the path to `scale_vec::4X`; our block_K=128 varying-scale test catches it |
| [NVIDIA/open-gpu-kernel-modules#1269](https://github.com/NVIDIA/open-gpu-kernel-modules/issues/1269) | 64 KiB-page ARM64: DMA submap not 2 MiB-aligned, so large GPU mappings fault (Xid 31) | Blocks R610 on the 64 KiB profiles; production stays on 580.178.04 | Open, not ours; reported to NVIDIA in the developer forum. The qualification carries a one-line RM patch | When a driver release includes the fix, re-qualify R610 (memory saver 0.3.0, `NVreg_EnableSystemMemoryPools=0`) |

## Our pull requests

| Thread | What | Status (2026-10-05) | Next |
| --- | --- | --- | --- |
| [tile-ai/tilelang#3430](https://github.com/tile-ai/tilelang/pull/3430) | `[CUDA] Enable SM120 block-scaled MMA on SM121`: without it every block-scaled MMA compiles to a trap on `sm_121a` | Ready for review since 2026-10-05. On GB10: NVF4 tests 162 passed; SASS for `sm_121a` has 32 `OMMA.SF` with the fix and 32 `BPT.TRAP` without | Answer review comments |
| [NVIDIA/cutlass#3038](https://github.com/NVIDIA/cutlass/pull/3038) | Enable the SM121-gated MXFP4 MoE kernel path | Open since 2026-02-16, mergeable, review required | Answer review comments; rebase if it stops being mergeable |
| [eugr/spark-vllm-docker#4](https://github.com/eugr/spark-vllm-docker/pull/4) | Cache encodings to shorten rebuilds | Open since 2025-12-19, no activity | Close it if it is no longer wanted |
| [christopherowen/dgx-spark-power-control#1](https://github.com/christopherowen/dgx-spark-power-control/pull/1) | Power diagnosis, passive tracing and firmware recovery analysis | Draft | Merge only after the hardware writes have been tested with the owner's approval |

## Planned pull requests

| Pull request | Branch | Waiting for | Before opening |
| --- | --- | --- | --- |
| `[CUDA] Support SM120 MXFP8 and MXFP4 block-scaled MMA in T.gemm_blockscaled` | [christopherowen/tilelang `sm120-mxfp8-mxfp4`](https://github.com/christopherowen/tilelang/commits/sm120-mxfp8-mxfp4) (stacked on #3430) | qqq-tao's answer on #3099 | Its block-scaled suites passed on GB10 on 2026-10-05 (213 passed, 4 skipped). Still to do: re-run with #3286 merged in; credit #3081/#3099 |
| `[Example] DeepSeek-V4.1-Flash kernels for SM120/SM121` | the four example commits on `deepseek-v41-sm120`, squashed | The MXFP8/MXFP4 PR | Reword comments and the README that mention our serving shapes. `pytest examples/deepseek_v41` passed on GB10 on 2026-10-05 (11 passed) |

## Planned issues

| Issue | Evidence | Waiting for | Before filing |
| --- | --- | --- | --- |
| Warp-specialized race with `cp.async` scale staging in short block-scaled tiles | [decode-kernel experiment](../experiments/2026-10-05-tilelang-decode-kernels/README.md): 16-row MXFP8 tiles with warp specialization fail the bit check in up to every run (300/300 at `block_N=32`, 2 stages, 128 threads), never with it disabled. Scales are staged by `cp.async` from only some producer threads, and every producer thread arrives with `cp.async.mbarrier.arrive.noinc` on barriers that also carry the TMA transaction count; failures fall as more producer threads copy scales (none at `block_N=128`). Not fixed by #3288 | A reproducer on upstream main: MXFP8 is not upstream yet, so try NVF4 short tiles in a GB10 window, or file it with the MXFP8 PR | A standalone script, the failure matrix and the generated CUDA |
