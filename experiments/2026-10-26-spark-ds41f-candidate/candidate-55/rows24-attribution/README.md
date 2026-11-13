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

## Results

Not run yet.
