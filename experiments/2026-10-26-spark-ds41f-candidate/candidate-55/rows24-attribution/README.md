# candidate-55: attributing the 24- and 32-row grid shapes at tp4-ring

Base commit: `cf71810d9`. Image: `spark-ds41f:candidate-55`, fourth build (vLLM tree `8aaae3df`, sparknet tree
`e06f6a06`), with the TP4 recipe (L2 prefetch windows at 15/12/12 MB, collectives above the 2 MiB slot on PyNCCL).

**Conclusion (2026-11-13):** the loss is the L2 prefetch's WO window, not a kernel or a setting of any one PR. With
the prefetch off on both builds candidate-55 is 1.5% to 2.2% faster than candidate-54 at the 24-row shapes, where
with the 15/12/12 MB windows it is 1.5% to 1.9% slower. With the WO window at 10 MB candidate-55's 24-row shapes are
2.4% to 2.7% faster and candidate-54's do not move. 24 and 32 rows are the 32-row decode tile tier. A WO window sized
by row band is the fix; at one size for every row count, 10 MB costs 1.9% at 1x2 and the grid comes out the same, so
the recipe is unchanged.

**Corrected 2026-11-13: the three overlay arms mounted nothing.** `candidate-55-no1134-loader`,
`candidate-55-no989-no982` and `candidate-55-no982` named their files under `container.extra_mounts`, which no command
reads, so they booted the image unchanged. #1134's loader, #989 and #982 are therefore **not** ruled out by the
overlays job below; its three arms are six more boots of `candidate-55`. The profiles are fixed and the arms run again as
[attribute-rows24-overlays-r2-tp4.json](attribute-rows24-overlays-r2-tp4.json). The arms that are environment
settings or another build stand: the core partition, #974's control image, the defragmentation, the prefetch off, the
three windows and candidate-54's. So does the conclusion, which rests on those. The single-GPU loader arm on sp8 was
mounted with `docker -v` and stands.

## What is being explained

With the recipe, candidate-55's screen at `tp4-ring` has the grid 1.55% faster than candidate-54's recorded screen,
but four shapes are slower on both boots ([receipts/tp4-recipe](../receipts/tp4-recipe/README.md)): 16x32 by 1.6% and
2.0%, 8x24 by 1.3% and 0.7%, 4x24 by 0.8% and 2.1%, and 8x24 at 256K context by 1.2% and 2.4%.

- candidate-54's recorded screen has the default windows. With the caps candidate-54 gains at these rows too (2.6%
  at 24 rows at `tp4-switched`), so the loss from candidate-55's own changes is larger than the screen shows.
- The third build's CUDA 13.0.2 control had these rows slower as well, so #974 is at most part of it.
- At `tp4-switched` the arms without #982 and without the proxy pinning left these rows unchanged
  ([grid-attribution](../grid-attribution/README.md)).

## What the earlier arms did not isolate

- **#1134's core partition (vLLM `0801`).** The `noaffinity` arm of the grid attribution set
  `VLLM_SPARK_THREAD_AFFINITY=0`, a variable the image does not read. The partition's switch is
  `VLLM_ROCE_CPU_AFFINITY`. That arm measured candidate-55 unchanged.
- **#989's kernels (sparknet `0002`).** The `no989` arm set `SPARKNET_ROCE_PROXY_CPU=none`, which unpins the RoCE
  proxy (sparknet `0001`, #1134). #989's multi-stage all-gather, reduce-scatter and exchange kernels stayed.
- **#1134's loader (vLLM `0800`)** and **#974** at these rows with the recipe: no arm.

## Arms

One intended variable each, on candidate-55's `tp4-dev.json` as it stands at the base commit. The suite is the engine
grid step alone ([candidate-48/grid-suite.json](../../candidate-48/grid-suite.json)); every arm boots twice, the arms
in order and then again. Two jobs, each with its own two boots of `candidate-55`: the arms that change a setting or
the image ([attribute-rows24-settings-tp4.json](attribute-rows24-settings-tp4.json)), and the arms that mount files
over it ([attribute-rows24-overlays-tp4.json](attribute-rows24-overlays-tp4.json)). A boot that fails ends its job,
and an overlay is the likelier to fail.

| Arm | Change | Isolates |
| --- | --- | --- |
| `candidate-54-l2cap` | candidate-54 with the same caps | the loss itself, both builds capped, in one session |
| `candidate-55` | none | the reference |
| `candidate-55-no1134-loader` | [no1134-loader/overlay](no1134-loader/overlay): the four loader files `0800` changes, as before it | #1134's weight arenas and sponge |
| `candidate-55-noaffinity` | `VLLM_ROCE_CPU_AFFINITY=0` | #1134's core partition |
| `candidate-55-no974` | the control image (CUDA 13.0.2) with the same recipe | #974 |
| `candidate-55-no989-no982` | [no989-no982/overlay](no989-no982/overlay): sparknet's seven runtime files as after `0001` alone | #989 and #982 together |
| `candidate-55-no982` | [grid-attribution/no982/overlay](../grid-attribution/no982/overlay) | #982; with the arm above, #989 |
| `candidate-55-nodefrag` | `host.defragment_before_start: false` | the pre-launch defragmentation |

- **The loader overlay** is a `git revert` of `0800`'s commit in the fourth build's prepared tree on dgx4, taken for
  the four files outside `tests/`. It applies cleanly; the one later change to those files is a docstring of #1096's.
- **The sparknet overlay** is `git archive` of the seven files at the commit of `0001` in the candidate's sparknet
  stack. Its all-reduce kernel is byte-identical to the `no982` overlay's.
- **#1096 has no arm.** Its files cannot be put back without the B12X sources the image no longer has. What it
  changes on the serving path is one kernel's row bound.
- **The two sparknet arms compile their kernels** into `/cache/tilelang/c55-rows24`, a copy of each dgx node's kernel
  cache made before the run, so only the changed collectives compile.
- candidate-54 with the caps is measured because no recorded run of it exists at `tp4-ring`.

## Commands

```bash
bin/spark candidate queue tp4-ring experiments/2026-10-26-spark-ds41f-candidate/candidate-55/rows24-attribution/attribute-rows24-settings-tp4.json experiments/2026-10-26-spark-ds41f-candidate/candidate-55/rows24-attribution/attribute-rows24-overlays-tp4.json --apply
bin/spark candidate verdicts tp4-ring candidate-55-attribute-rows24-settings-tp4
bin/spark candidate verdicts tp4-ring candidate-55-attribute-rows24-overlays-tp4
```

## Results: no single change causes it

Jobs `20261113T051720-attribute-rows24-settings-tp4.json` (2026-11-13 05:17Z to 05:56Z) and
`20261113T052132-attribute-rows24-overlays-tp4.json` (05:56Z to 06:28Z), deployment `3d1a3a76`. All 18 boots reached
ready, no memory guard fired, and no boot compiled a kernel into the default cache
([receipts](../receipts/rows24-attribution/tp4-ring)).

**The loss, both builds capped, in one session** (`candidate-55` against `candidate-54-l2cap`, the mean of both boots
of each; step time):

| Shapes | candidate-55 |
| --- | --- |
| 1x2; 1x3 at four context lengths | 2.6% faster; 3.9% to 5.0% faster |
| 6 to 16 rows | 1.9% to 4.4% faster |
| 4x24, 8x24, 8x24 at 64K, 8x24 at 256K | **1.5%, 1.6%, 1.6%, 1.9% slower** |
| 16x32 | **1.3% slower** |
| 16x48, 8x48 | 1.1% to 1.3% faster |
| 12x72; 16x96 | 0.9% slower; the same |
| Grid, geometric mean | 1.5% faster (the tool: 1.54%, margin 0.34%) |

**It is in the target model's forward.** The grid step splits into the forward, the drafter and the remainder
([components.py](components.py)). At 24 and 32 rows candidate-55's forward is 0.36 to 0.58 ms slower (about 15 us a
layer), its drafter the same within 0.08 ms, and the remainder 0.13 ms faster, as at every shape. Below 24 rows the
forward is 0.19 to 0.54 ms faster.

**Each arm against `candidate-55`, the mean of both boots of each** (negative is faster; the four 24-row shapes as one
geometric mean; `candidate-55`'s own two boots differ by 0.1% to 0.3% on these groups):

| Arm | 24 rows | 16x32 | 48 rows | 1x2 | 1x3 | Grid (the tool, each boot) |
| --- | --- | --- | --- | --- | --- | --- |
| `candidate-54-l2cap` | -1.6% | -1.3% | +1.1% | +2.7% | +4.1% | +1.50%, +1.46% |
| `candidate-55-noaffinity` | -0.2% | +0.5% | +0.4% | -0.4% | +1.3% | +0.04%, -0.15% |
| `candidate-55-no974` | -0.2% | +1.1% | +2.8% | +1.7% | +3.1% | +1.08%, +1.03% |
| `candidate-55-nodefrag` | -0.6% | +0.0% | -0.2% | -0.7% | +0.3% | -0.34%, -0.44% |
| `candidate-55-no982` | -0.1% | +1.5% | -0.3% | +0.1% | -2.2% | +0.18%, -0.22% |
| `candidate-55-no1134-loader` | +0.2% | +0.4% | +0.9% | +0.2% | -1.7% | +0.00%, -0.17% |
| `candidate-55-no989-no982` | -0.3% | +0.6% | +0.1% | -0.1% | -2.6% | -0.40%, +0.01% |

- **No arm returns the 24-row shapes to candidate-54's.** The largest move is the pre-launch defragmentation's, 0.6%
  of the 1.6%, at about twice the repeat boots' spread. #1134's core partition, #1134's loader, #982, #989 with #982,
  and #974 each leave them within 0.3%.
- **#974 is a gain, not the cause.** Without CUDA 13.5 the 24-row shapes are the same, and 1x2, 1x3, 32, 48 and 96
  rows are 1% to 3% slower. So without #974 the loss shows at every shape of 24 rows and more (0.8% to 2.4% against
  candidate-54), and #974 wins it back at 32, 48 and 96 rows.
- **1x3 in the overlays job** is 1.7% to 2.6% faster in all three overlay arms alike: that job's two `candidate-55`
  boots had 1x3 at 14.7 and 15.0 ms against 14.0 and 14.2 ms in the settings job. It reads as the reference's, not
  the arms'.
- **Time to ready:** with the earlier loader 117.5 s and 119.7 s; `candidate-55` 113.3 s to 117.7 s over its four
  boots; candidate-54 with the caps 114.3 s and 111.6 s. #1134's loader does not account for the 4 s the third build
  recorded.

## Single GPU on sp8: the same kernels, launched the same way

One GPU, dummy weights at a TP4 rank's shapes, the caps, 60 replays, the third build's image
([receipts](../receipts/rows24-attribution/sp8-single-gpu)). Forward time in ms, two or three boots an arm:

| Arm | 8x16 | 8x24 | 16x32 | 16x48 | 12x72 | 16x96 |
| --- | --- | --- | --- | --- | --- | --- |
| candidate-54 | 1.759, 1.776 | 1.918, 1.921 | 2.023, 2.034 | 2.265, 2.260 | 2.892, 2.902 | 3.118, 3.115 |
| candidate-55 | 1.782, 1.787, 1.800 | 2.008, 2.003, 2.017 | 2.109, 2.117, 2.127 | 2.336, 2.329, 2.349 | 2.991, 3.002, 3.013 | 3.143, 3.153, 3.164 |
| candidate-55, `VLLM_DUMMY_WEIGHT_ARENA=0` | 1.794, 1.798 | 2.020, 2.008 | 2.113, 2.102 | 2.321, 2.316 | 2.991, 3.008 | 3.135, 3.148 |
| candidate-55, the loader overlay | 1.784, 1.809 | 2.022, 2.030 | 2.121, 2.124 | 2.327, 2.343 | 3.004, 3.014 | 3.137, 3.175 |
| candidate-55, `VLLM_ROCE_CPU_AFFINITY=0` | 1.786, 1.792 | 2.011, 2.013 | 2.110, 2.133 | 2.333, 2.363 | 2.984, 3.018 | 3.159, 3.182 |

- **candidate-55's forward is 3% to 5% slower from 24 rows up on one GPU**, where no collective runs, and neither the
  weight arena, nor the earlier loader, nor the core partition changes it. #974 and #1096 are not separated there:
  sp8 has no control image.
- **Both builds launch the same kernels.** In the CUPTI traces of the grid-attribution's single-GPU boots
  (`/tmp/iket-diag/cupti-c5{4,5}-l2cap` on sp8), every (kernel, grid, block) is launched the same number of times a
  step in both builds at 8x16, 8x24 and 16x32. No tile tier or kernel selection differs; the times do. candidate-55's
  L2 prefetch kernel takes 20% longer from 24 rows up and `gemm_mxfp8_kernel` runs alone for less of its time.
- **Both images configure the prefetcher alike** ([probe_l2.py](../receipts/rows24-attribution/sp8-single-gpu/probe_l2.py)):
  L2 25 MB, persisting set-aside 4.5 MB of 18 MB, fetch granularity 64, no set-aside requested.

## What reading #1096 found

#1096 has no arm, so its patch was read for anything that acts from 24 rows.

- **The backend's name.** It renames the DS4.1 attention backend `"B12X"` to `"DSV41"`. The shared attention layers
  compare that name (`layers/attention/attention.py`, `mla_attention.py`), which its rebase note says nothing does;
  DS4.1's own attention classes do not go through those layers, so the comparisons are not reached.
- **`cache_dtype_str`** `"b12x_dsv41"` to `"dsv41"`: neither the KV page size nor the quantization mode keys on it.
- **The quantization files it moves** serve MXFP4-CSF checkpoints, which this deployment does not load.
- **What does change:** `engram_resident_lookup_kernel`'s row bound, and the startup rounds that no longer run
  without B12X (the preparation stages and the disposable KV pool before the real allocation). Neither names a row
  count.

## The prefetch: off on both builds, and each window smaller

Jobs `20261113T063736-attribute-rows24-prefetch-tp4.json` (06:37Z to 07:26Z, deployment `68254cbd`) and
`20261113T072659-attribute-rows24-wo-control-tp4.json` (07:27Z to 07:43Z). All 16 boots reached ready and no memory
guard fired ([receipts](../receipts/rows24-attribution/tp4-ring)).

Arms: both builds with `VLLM_DS41_L2_PREFETCH=0`; candidate-55 with one window smaller at a time (WO 10 MB of 15, FFN
8 MB of 12, NEXT 8 MB of 12); and both builds with the windows at 10/12/12 MB. The COMP window of the four layers
that own a compressor takes WO's budget, so the WO arm shrinks it too.

**With the prefetch off candidate-55 is not slower at 24 and 32 rows** (`candidate-55-l2off` against
`candidate-54-l2off`, the mean of both boots of each):

| Shapes | Step | Forward |
| --- | --- | --- |
| 1x2; 1x3 | 1.5% faster; 3.0% faster | -0.11 ms; -0.37 ms |
| 8x16 | 1.7% faster | -0.20 ms |
| 4x24, 8x24, 8x24 at 64K, 8x24 at 256K | **2.2%, 1.5%, 1.9%, 2.0% faster** | -0.14 to -0.29 ms |
| 16x32 | **1.3% faster** | -0.11 ms |
| 16x48, 8x48 | 1.1% faster | -0.09 to -0.12 ms |
| 12x72; 16x96 | 0.9% slower; 0.7% slower | +0.40 ms; +0.49 ms |

So the kernels' own time is not slower at these rows. With the windows on, the prefetch takes 2.4 ms off
candidate-54's 24-row step and 1.8 ms off candidate-55's.

**Each window, against `candidate-55` with 15/12/12 MB** (the mean of both boots of each; negative is faster):

| Arm | 24 rows | 16x32 | 48 rows | 12x72 | 1x2 | 1x3 | 6 to 16 rows | Grid (the tool, each boot) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `candidate-55-l2-wo10` | **-2.6%** | **-1.5%** | +1.1% | +1.0% | +1.9% | +1.6% | +1.2% | -0.09%, +0.28% |
| `candidate-55-l2-ffn8` | -0.4% | -0.3% | -0.2% | +0.3% | +0.6% | -0.1% | -0.1% | -0.04%, -0.46% |
| `candidate-55-l2-next8` | +1.5% | +1.7% | +1.3% | -0.0% | +1.7% | +1.7% | +0.1% | +1.14%, +0.84% |
| `candidate-55-l2off` | +8.4% | +7.2% | +6.9% | +4.6% | +21.1% | +19.7% | +15.2% | +12.52%, +12.12% |

- **The WO window is the one.** At 10 MB the four 24-row shapes are 2.4% to 2.7% faster and 16x32 1.5%, more than the
  loss. Every other band is 0.5% to 2% slower, and the grid is the same.
- **FFN at 8 MB changes nothing; NEXT at 8 MB is worse** at every band but 6 to 16 rows and 72 to 96.
- **The prefetch is worth 4.5% to 21% of the step** at `tp4-ring`, most at one request.
- `candidate-55`'s two boots in this job differ by 0.69% on the grid, more than in the other jobs (0.1% to 0.2%).

**The 15 MB WO window costs candidate-55, not candidate-54** (both builds at 10/12/12 MB, two boots each):

| Shapes | candidate-54, 10 MB against 15 MB | candidate-55, 10 MB against 15 MB | candidate-55 against candidate-54, both at 10 MB |
| --- | --- | --- | --- |
| The four 24-row shapes | +0.2% (21.12 against 21.08 ms) | -2.6% | **1.4% faster** (0.7% to 2.2%) |
| 16x32 | +0.5% | -1.5% | +0.2%, the same |
| Grid | +0.8% | +0.1% | 1.9% faster (the tool: 1.97%, margin 0.38%) |

candidate-54's 15 MB figures are its `candidate-54-l2cap` boots of the settings job, 90 minutes earlier in the same
window.

**One GPU agrees for 24 and 32 rows.** On sp8 with the prefetch off (two boots each,
[receipts](../receipts/rows24-attribution/sp8-single-gpu)), candidate-55's forward is 0.9% slower at 8x24 and 1.5%
at 16x32, where with the windows on it is 4.6% and 4.2% slower. At 16x48 and 12x72 it stays 2.8% and 2.5% slower
with the prefetch off: a second, smaller loss in the kernels' own time at wide rows, which `tp4-ring` shows at
72 and 96 rows (0.9% and 0.7%, in the table above). That one is not attributed.

## Why the WO window, and why these rows

- **Where it fills.** WO is issued after the query up-projection and fills while the sparse-MLA preparation kernels
  run; `l2_prefetch.py`'s own description says the fill slows those kernels, and that a window still filling delays
  the one queued behind it.
- **Why 24 and 32 rows.** Decode tiles come in heights 16, 32 and wider (`DECODE_TILE_ROWS` in `tilelang/gemm.py`,
  `_WO_A_TILES` in `tilelang/wo.py`), so 24 and 32 rows are both the 32-row tier: one band, 17 to 32 rows. 20 and 28
  rows are not in the grid and were not measured.
- **Why candidate-55.** Not established by an arm: no single change moves these rows. Its forward is faster at every
  band below 24 rows, so each window has less time before its consumer.

## Where this leaves it

- **Attributed:** the 24- and 32-row loss is the WO prefetch window at 15 MB in the 32-row tile tier. candidate-55's
  kernels are not slower there.
- **Not a recipe change.** One WO size for every row count trades the 24-row shapes (-2.6%) against 1x2 (+1.9%), the
  rows a single stream uses, and the grid is the same. candidate-55's TP4 recipe stays at 15/12/12 MB.
- **The fix is a window sized by row band.** The prefetcher's `issue()` already receives the row count and each
  captured graph bakes the plan its row count selects, so a WO budget for 17 to 32 rows is a Python change with no
  kernel to compile. It is a change of its own, for a later candidate, with its own measurement.
- **Still open:** the 0.7% to 0.9% at 72 and 96 rows with the prefetch off, and the 4 s of startup.
