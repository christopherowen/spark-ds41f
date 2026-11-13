# candidate-55: attributing the 256K prefill regression at tp4-ring

**Conclusion:** #989's routing causes it. With reduce-scatters and exchanges above one slot back on PyNCCL
(`SPARKNET_ROCE_MAX_COLLECTIVE_BYTES=2097152`), candidate-55's prefill at `tp4-ring` returns to candidate-54's at
both lengths. The sliced all-gather and the pre-launch defragmentation change nothing. candidate-55's `tp4-ring`
profile takes the limit; `tp4-switched`, where #989's routing is not slower, keeps #989's default.

Base commit: `077b2345e` (the run's deployment was `78d7cb36`). Image: `spark-ds41f:candidate-55`, fourth build (vLLM tree `8aaae3df`, sparknet tree
`e06f6a06`), against `spark-ds41f:candidate-54` (`d502217e`).

## What is being explained

candidate-55's screens at `tp4-ring` are 3.25% (second build) and 3.37% (third build) slower than candidate-54 at
262,144 tokens of prefill, in the CUDA 13.0.2 control too, and the same within the margin at 32,768. At `tp4-switched`
prefill is unchanged. So the cause is among #1134, #1096, #989, #982 and the pre-launch defragmentation, and it acts
on the ring only.

## Hypothesis

#989 changed which collectives sparknet takes. Before it, a reduce-scatter or exchange above one slot, and an
all-gather above the shard limit, went to PyNCCL. With it, collectives up to 64 MiB
(`SPARKNET_ROCE_MAX_COLLECTIVE_BYTES`) stay on sparknet, staged through the two slots.

- The slot is `max(SPARKNET_ROCE_ALLREDUCE_CAPACITY_BYTES, SPARKNET_ROCE_ALLGATHER_MAX_BYTES)`: 2 MiB in the
  `tp4-ring` recipe, 4 MiB in `tp4-switched`.
- The ring's opposite rank is not cabled, so sparknet relays to it, and the ring's NCCL settings were tuned for these
  sizes (`NCCL_ALGO=Ring`, 4 channels, 4 MiB buffers).
- #989's own mark was measured on sp1..sp4 and sp1..sp3, both switched.

## Arms

One intended variable each, by environment or host setting, on the same image. The suite is the screen's prefill step
alone ([prefill-suite.json](prefill-suite.json)); every arm boots twice.

| Arm | Change | Restores |
| --- | --- | --- |
| `candidate-54` | [candidate-54/tp4-dev.json](../../candidate-54/tp4-dev.json) | the comparison on this suite |
| `candidate-55` | [grid-attribution/tp4-b4-dev.json](../grid-attribution/tp4-b4-dev.json): `tp4-dev.json` as it stood when the run was taken, before the recipe changes entered it | the reference |
| `candidate-55-collective-2m` | `SPARKNET_ROCE_MAX_COLLECTIVE_BYTES=2097152` | reduce-scatters and exchanges above one slot on PyNCCL |
| `candidate-55-unsliced-gather` | `SPARKNET_ROCE_SLICED_ALLGATHER=0` | all-gathers above the shard limit on PyNCCL |
| `candidate-55-nodefrag` | `host.defragment_before_start: false` | launch without the pre-launch defragmentation |

`candidate-55-collective-2m` also sends all-gathers of shards between 512 KiB and 2 MiB to PyNCCL, which
candidate-54 kept on sparknet: the limit applies to the gathered size.

candidate-54 is measured here because it has no recorded run of this suite, and a prefill step on a fresh boot need
not match the screen's, which follows the grid and c16 steps.

## Commands

```bash
bin/spark candidate queue tp4-ring experiments/2026-10-26-spark-ds41f-candidate/candidate-55/prefill-attribution/attribute-prefill-tp4.json --apply
bin/spark candidate verdicts tp4-ring candidate-55-attribute-prefill-tp4
```

## Results

Job `20261112T200638-attribute-prefill-tp4.json`, 2026-11-12 20:36Z to 21:16Z, ten boots, each ready with no memory
guard firing ([table](../receipts/prefill-attribution/tp4-ring/lab/candidate-55-attribute-prefill-tp4-table.txt)).
Prefill in tok/s, the mean of 3 repeats:

| Arm | 32,768, boot 1 | boot 2 | 262,144, boot 1 | boot 2 | 262,144, mean against `candidate-55` |
| --- | --- | --- | --- | --- | --- |
| `candidate-54` | 8,892 | 8,776 | 8,894 | 8,816 | +3.2% |
| `candidate-55` | 8,514 | 8,467 | 8,608 | 8,552 | |
| `candidate-55-collective-2m` | 8,830 | 8,720 | 8,915 | 8,834 | +3.4% |
| `candidate-55-unsliced-gather` | 8,526 | 8,430 | 8,586 | 8,512 | -0.4% |
| `candidate-55-nodefrag` | 8,508 | 8,486 | 8,576 | 8,575 | -0.1% |

- **The regression repeats on this suite:** candidate-55 is 3.21% slower than candidate-54 at 262,144 tokens (margin
  1.14%) and 4.05% at 32,768 (margin 2.55%), both boots of each. The screens gave 3.37% and, within their margin,
  2.31%.
- **The collective limit recovers all of it:** `candidate-55-collective-2m` is 0.2% faster than candidate-54 at
  262,144 tokens and 0.7% slower at 32,768, inside the interval of either.
- **Four of the five arms' second boots are slower than their first** by 0.6% to 0.9% at 262,144 tokens
  (candidate-54 too; `nodefrag` is the same on both), so the session drifted down. The order was the five arms, then
  the five again, so the arms carry the same drift.
- **Not the cause:** the sliced all-gather and the pre-launch defragmentation.
- **Not measured here:** which of the reduce-scatter and the exchange carries the loss, and whether a larger stage
  (`SPARKNET_ROCE_STAGE_BYTES`) or slot on the ring would let sparknet keep them. The limit restores candidate-54's
  routing, which is what the recorded references measure.
