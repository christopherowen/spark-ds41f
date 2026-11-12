# candidate-55: attributing the 256K prefill regression at tp4-ring

Base commit: `6c3e98b56`. Image: `spark-ds41f:candidate-55`, fourth build (vLLM tree `8aaae3df`, sparknet tree
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
| `candidate-55` | [tp4-dev.json](../tp4-dev.json) | the reference |
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

Not run yet.
