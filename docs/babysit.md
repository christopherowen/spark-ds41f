# Upstream follow-ups

Issues and pull requests outside this repository that we opened, or that a
decision here depends on. Each row says what we are waiting for and what to do
when it changes. Add a row when we open or start relying on an upstream thread;
remove it once the change has landed here or no longer matters.

Check them all at once:

```sh
for ref in tile-ai/tilelang#3430 tile-ai/tilelang#3495 Horizon-Tech-doo/tilelang#1 \
           tile-ai/tilelang#3099 tile-ai/tilelang#3081 tile-ai/tilelang#3286 \
           tile-ai/tilelang#3287 tile-ai/tilelang#3288 \
           NVIDIA/open-gpu-kernel-modules#1269 christopherowen/dgx-spark-power-control#1 \
           zephyrproject-rtos/zephyr#121740; do
  gh api "repos/${ref%#*}/issues/${ref#*#}" --jq "\"$ref [\(.state)] \(.updated_at[:10]) \(.title)\""
done
```

## Waiting on upstream

| Thread | What | Our stake | Status | When it changes |
| --- | --- | --- | --- | --- |
| [tile-ai/tilelang#3287](https://github.com/tile-ai/tilelang/issues/3287), fix [#3288](https://github.com/tile-ai/tilelang/pull/3288) | Warp-specialized pipeline releases a stage before a synchronous (non-TMA, non-cp.async) shared copy is consumed | Ruled out as the cause of our decode-tile race on 2026-10-05: with #3288 our 16-row MXFP8 tiles still fail (1,222 of 3,600 runs, against 1,180 without it) and their generated CUDA is identical | Checked 2026-10-12: both still open, with no activity since 2026-09-27 | Nothing to do for us; see the planned issue below |
| [tile-ai/tilelang#3099](https://github.com/tile-ai/tilelang/pull/3099), [#3081](https://github.com/tile-ai/tilelang/pull/3081) | qqq-tao's SM120 MXFP8 and MXFP4 block-scaled MMA | Prior work credited by our smaller rebased implementation | Checked 2026-10-12: both still open and conflicting with main; no replies or new commits after [our 2026-10-05 comment](https://github.com/tile-ai/tilelang/pull/3099#issuecomment-5998824942). Opened [#3495](https://github.com/tile-ai/tilelang/pull/3495) | Coordinate scope if the authors respond; track our submission below |
| [tile-ai/tilelang#3286](https://github.com/tile-ai/tilelang/pull/3286) | SM120 block-scaled register-A GEMM: scales loaded once per K atom | Its compact-scale path passes byte selector 0, which breaks MXFP8/MXFP4 when combined | Checked 2026-10-12: no reply or new commits after [our comment](https://github.com/tile-ai/tilelang/pull/3286#issuecomment-5998825359). Reproduced the selector failure and opened [Horizon-Tech-doo/tilelang#1](https://github.com/Horizon-Tech-doo/tilelang/pull/1) against its head branch | Incorporate the guard before combining it with #3495; re-run the varying-scale runtime tests in a GPU window |
| [NVIDIA/open-gpu-kernel-modules#1269](https://github.com/NVIDIA/open-gpu-kernel-modules/issues/1269) | 64 KiB-page ARM64: DMA submap not 2 MiB-aligned, so large GPU mappings fault (Xid 31) | Blocks R610 on the 64 KiB profiles; production stays on 580.178.04 | Open, not ours; reported to NVIDIA in the developer forum. The qualification carries a one-line RM patch | When a driver release includes the fix, re-qualify R610 (memory saver 0.3.0, `NVreg_EnableSystemMemoryPools=0`) |

## Our pull requests

| Thread | What | Status | Next |
| --- | --- | --- | --- |
| [tile-ai/tilelang#3430](https://github.com/tile-ai/tilelang/pull/3430) | SM121 block-scaled MMA enablement | Merged 2026-10-10 as `5f803828`; our new MX PR is based on main containing it | The [main rebase candidate (#154)](https://github.com/christopherowen/spark-ds41f/pull/154) drops the redundant patch; qualify its image before promotion |
| [tile-ai/tilelang#3495](https://github.com/tile-ai/tilelang/pull/3495) | SM120/SM121 MXFP8 and MXFP4 block-scaled MMA, including FP8 × FP4 | Opened 2026-10-12 on current main. Native compiler rebuilt; new suites 20 passed, 7 GPU tests skipped; all three operand paths compile for `sm_120a` and `sm_121a`; pre-commit passed. Earlier GB10 runtime suites: 213 passed, 4 skipped on 2026-10-05 | Answer review comments; re-run runtime coverage in a GPU window |
| [Horizon-Tech-doo/tilelang#1](https://github.com/Horizon-Tech-doo/tilelang/pull/1) | Limit #3286 compact scale packages to `scale_vec::4X`, preserving per-K-atom MX selectors | Opened 2026-10-12 against `sm120-register-a-perf`. Combined with #3495 and #3286: selectors corrected; five block-scaled suites 144 passed, 74 GPU tests skipped; pre-commit passed | Have the author incorporate it into tile-ai/tilelang#3286 |
| [christopherowen/dgx-spark-power-control#1](https://github.com/christopherowen/dgx-spark-power-control/pull/1) | Power diagnosis, passive tracing and firmware recovery analysis | Draft | Merge only after the hardware writes have been tested with the owner's approval |

## Our issues

| Thread | What | Status (2026-10-09) | Next |
| --- | --- | --- | --- |
| [zephyrproject-rtos/zephyr#121740](https://github.com/zephyrproject-rtos/zephyr/issues/121740) | Microchip XEC eSPI drivers: the virtual-wire send timeout added in 2c36616dd2b (in v3.7.2 as f9ef3f04276) never returns `-ETIMEDOUT`, and can report a timeout for a send that succeeded. Found while rebuilding the DGX Spark EC firmware from source; the Spark's EC (Zephyr v3.7.1) predates the timeout code and is not affected | Filed 2026-10-09 as a Bug (impact: functional limitation) with a suggested fix and an offer of a pull request; no maintainer reply yet | When a maintainer responds, send the fix for both drivers as one pull request with a `Signed-off-by` line; remove the row once it merges |

## Planned pull requests

| Pull request | Branch | Waiting for | Before opening |
| --- | --- | --- | --- |
| `[Example] DeepSeek-V4.1-Flash kernels for SM120/SM121` | the four example commits on `deepseek-v41-sm120`, squashed | [#3495](https://github.com/tile-ai/tilelang/pull/3495) | Reword comments and the README that mention our serving shapes. `pytest examples/deepseek_v41` passed on GB10 on 2026-10-05 (11 passed) |

## Planned issues

| Issue | Evidence | Waiting for | Before filing |
| --- | --- | --- | --- |
| Warp-specialized race with `cp.async` scale staging in short block-scaled tiles | [decode-kernel experiment](../experiments/2026-10-05-tilelang-decode-kernels/README.md): 16-row MXFP8 tiles with warp specialization fail the bit check in up to every run (300/300 at `block_N=32`, 2 stages, 128 threads), never with it disabled. Scales are staged by `cp.async` from only some producer threads, and every producer thread arrives with `cp.async.mbarrier.arrive.noinc` on barriers that also carry the TMA transaction count; failures fall as more producer threads copy scales (none at `block_N=128`). Not fixed by #3288 | A reproducer on upstream main: MXFP8 is not upstream yet, so try NVF4 short tiles in a GB10 window, or file it with the MXFP8 PR | A standalone script, the failure matrix and the generated CUDA |
