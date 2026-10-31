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
| [tile-ai/tilelang#3287](https://github.com/tile-ai/tilelang/issues/3287), fix [#3288](https://github.com/tile-ai/tilelang/pull/3288) | Warp-specialized pipeline releases a stage before a synchronous (non-TMA, non-cp.async) shared copy is consumed | Ruled out as the cause of our decode-tile race on 2026-10-05: with #3288 our 16-row MXFP8 tiles still fail (1,222 of 3,600 runs, against 1,180 without it) and their generated CUDA is identical | Checked 2026-10-18: both still open. Labelled on 2026-10-15; no review or new commits since 2026-09-27 | Nothing to do for us; see the planned issue below |
| [tile-ai/tilelang#3099](https://github.com/tile-ai/tilelang/pull/3099), [#3081](https://github.com/tile-ai/tilelang/pull/3081) | qqq-tao's SM120 MXFP8 and MXFP4 block-scaled MMA | Prior work credited by our smaller rebased implementation | On 2026-10-13 qqq-tao [agreed](https://github.com/tile-ai/tilelang/pull/3099#issuecomment-6140351215) to continue in [#3495](https://github.com/tile-ai/tilelang/pull/3495) and asked for attribution. Since 2026-10-15 our MXFP8 and MXFP4 commits carry `Co-authored-by: Tao QU`, and the #3495 description credits both PRs | When #3495 merges, ask qqq-tao to close both PRs or point them at the merge. The remaining features are listed in #3495 as separate follow-ups |
| [tile-ai/tilelang#3286](https://github.com/tile-ai/tilelang/pull/3286) | SM120 block-scaled register-A GEMM: scales loaded once per K atom | Its compact-scale path passes byte selector 0, which breaks MXFP8/MXFP4 when combined | Checked 2026-10-18: labelled on 2026-10-15, no new commits since 2026-09-27, and no reply to [our comment](https://github.com/tile-ai/tilelang/pull/3286#issuecomment-5998825359) or to [Horizon-Tech-doo/tilelang#1](https://github.com/Horizon-Tech-doo/tilelang/pull/1) | Incorporate the guard before combining it with #3495; re-run the varying-scale runtime tests in a GPU window |
| [NVIDIA/open-gpu-kernel-modules#1269](https://github.com/NVIDIA/open-gpu-kernel-modules/issues/1269) | 64 KiB-page ARM64: DMA submap not 2 MiB-aligned, so large GPU mappings fault (Xid 31) | Blocks R610 on the 64 KiB profiles; production stays on 580.178.04 | Open, not ours; reported to NVIDIA in the developer forum. The qualification carries a one-line RM patch. Checked 2026-10-18: no activity since 2026-07-29 | When a driver release includes the fix, re-qualify R610 (memory saver 0.3.0, `NVreg_EnableSystemMemoryPools=0`) |

## Our pull requests

| Thread | What | Status | Next |
| --- | --- | --- | --- |
| [tile-ai/tilelang#3430](https://github.com/tile-ai/tilelang/pull/3430) | SM121 block-scaled MMA enablement | Merged 2026-10-10 as `5f803828`. It has not landed here: r6u's lock pins `b95ee4ff`, 32 commits before the merge, and still applies `patches/tilelang/0001-sm121-block-scaled-mma.patch` | After #3495 merges, move the pin to main in one step: drop 0001-0003, keep 0004, check that the kernels stay bit-identical, and qualify the image. Then remove this row. The r6h-era candidate [#154](https://github.com/christopherowen/spark-ds41f/pull/154) is superseded |
| [tile-ai/tilelang#3495](https://github.com/tile-ai/tilelang/pull/3495) | SM120/SM121 MXFP8 and MXFP4 block-scaled MMA, including FP8 × FP4 | On 2026-10-13 Yongqi-Zhuo requested changes ("LGTM overall; please add CuTeDSL path support and tests"). Addressed in `89be232d` (2026-10-15, rebased onto main `c55eeb1c`): a CuTeDSL lowering for NVFP4, MXFP4, MXFP8 and FP8 × E2M1, with tests. GB10 runtime: 190 passed, 0 skipped ([posted](https://github.com/tile-ai/tilelang/pull/3495#issuecomment-6201059983) 2026-10-15). Checked 2026-10-18: all eight CI checks pass, mergeable, and the one CodeRabbit thread is resolved. Blocked only by the changes-requested review, which has not been revisited; Yongqi-Zhuo merged #3503 on 2026-10-16 | Wait for the re-review; no ping planned (owner, 2026-10-18). Answer new comments: fix valid findings, decline invalid ones with evidence. After the merge, move our pin (see #3430) and open the example PR |
| [Horizon-Tech-doo/tilelang#1](https://github.com/Horizon-Tech-doo/tilelang/pull/1) | Limit #3286 compact scale packages to `scale_vec::4X`, preserving per-K-atom MX selectors | Opened 2026-10-12 against `sm120-register-a-perf`. Combined with #3495 and #3286: selectors corrected; five block-scaled suites 144 passed, 74 GPU tests skipped; pre-commit passed. Checked 2026-10-18: no response | Have the author incorporate it into tile-ai/tilelang#3286 |
| [christopherowen/dgx-spark-power-control#1](https://github.com/christopherowen/dgx-spark-power-control/pull/1) | Power diagnosis, passive tracing and firmware recovery analysis | Draft, unchanged since 2026-09-29 | Merge only after the hardware writes have been tested with the owner's approval |

## Our issues

| Thread | What | Status | Next |
| --- | --- | --- | --- |
| [zephyrproject-rtos/zephyr#121740](https://github.com/zephyrproject-rtos/zephyr/issues/121740) | Microchip XEC eSPI drivers: the virtual-wire send timeout added in 2c36616dd2b (in v3.7.2 as f9ef3f04276) never returns `-ETIMEDOUT`, and can report a timeout for a send that succeeded. Found while rebuilding the DGX Spark EC firmware from source; the Spark's EC (Zephyr v3.7.1) predates the timeout code and is not affected | Filed 2026-10-09 as a Bug (impact: functional limitation) with a suggested fix and an offer of a pull request; no maintainer reply yet (checked 2026-10-18: only the bot's welcome) | When a maintainer responds, send the fix for both drivers as one pull request with a `Signed-off-by` line; remove the row once it merges |

## Planned pull requests

| Pull request | Branch | Waiting for | Before opening |
| --- | --- | --- | --- |
| `[Example] DeepSeek-V4.1-Flash kernels for SM120/SM121` | the four example commits on `deepseek-v41-sm120`, squashed | [#3495](https://github.com/tile-ai/tilelang/pull/3495) | Reword comments and the README that mention our serving shapes. `pytest examples/deepseek_v41` passed on GB10 on 2026-10-05 (11 passed) |
| `[IKET] Add START_END/MULTI_EVENT ranges, cross-warp scopes, 8-64b payloads, and ExtendedNativeDump` | `feat/iket-extended-ranges-and-payloads` (`e58f4575` in `christopherowen/tilelang`, exported as `patches/tilelang/0004-iket-extended-ranges-and-payloads.patch`) | Ready to open upstream against `tile-ai/tilelang` | Verified on GB10 (`dgx4`): `pytest testing/python/tools/test_tilelang_tools_cuda_iket.py` (39 passed) and end-to-end `iket.cli.main profile` trace verification |

## Planned issues

| Issue | Evidence | Waiting for | Before filing |
| --- | --- | --- | --- |
| Warp-specialized race with `cp.async` scale staging in short block-scaled tiles | [decode-kernel experiment](../experiments/2026-10-05-tilelang-decode-kernels/README.md): 16-row MXFP8 tiles with warp specialization fail the bit check in up to every run (300/300 at `block_N=32`, 2 stages, 128 threads), never with it disabled. Scales are staged by `cp.async` from only some producer threads, and every producer thread arrives with `cp.async.mbarrier.arrive.noinc` on barriers that also carry the TMA transaction count; failures fall as more producer threads copy scales (none at `block_N=128`). Not fixed by #3288 | A reproducer on upstream main: MXFP8 is not upstream yet, so try NVF4 short tiles in a GB10 window, or file it with the MXFP8 PR | A standalone script, the failure matrix and the generated CUDA |
