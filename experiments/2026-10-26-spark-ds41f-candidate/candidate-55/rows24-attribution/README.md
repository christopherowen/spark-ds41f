# candidate-55: attributing the 24- and 32-row grid shapes at tp4-ring

Base commit: `cf71810d9`. Image: `spark-ds41f:candidate-55`, fourth build (vLLM tree `8aaae3df`, sparknet tree
`e06f6a06`), with the TP4 recipe (L2 prefetch windows at 15/12/12 MB, collectives above the 2 MiB slot on PyNCCL).

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

## Where this leaves it

The loss is not one PR's kernel or setting among those with an arm, the dispatch is identical, and it sits in the
forward from 24 rows up. The remaining explanation is the one the caps already pointed at: candidate-55's forward is
faster, and the paced L2 prefetch windows now overlap its wide MoE and projection reads. The caps cut this loss from
1.3% to 3.3% (the fourth build's screen without them) to 0.7% to 2.4%.

**Next run** ([attribute-rows24-prefetch-tp4.json](attribute-rows24-prefetch-tp4.json)): both builds with the prefetch
off (`VLLM_DS41_L2_PREFETCH=0`), and candidate-55 with each window smaller in turn (WO 10 MB, FFN 8 MB, NEXT 8 MB).
If candidate-55 is not slower than candidate-54 at 24 rows with the prefetch off, the loss is the prefetch's phase; if
it still is, it is in the kernels' own time and points at #1096, which then needs a control build with B12X.
