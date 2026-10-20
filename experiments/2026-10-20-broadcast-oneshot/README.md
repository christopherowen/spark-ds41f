# The TP group's broadcast on the RoCE policy: DSpark's confidences on the one-shot all-gather

Owner, 2026-10-20: "we should move the broadcast over because it's a relatively small change."

**Status:** vLLM branch `broadcast-oneshot`, one commit on r6x's patch head 7316c7f9f
(d5bfe0203, tree 15a206a6), in ~/projects/vllm-ds41-broadcast-oneshot; the series patch
[vllm/0185-RoCE-policy-the-TP-group-s-broadcast-on-the-policy-s.patch](vllm/0185-RoCE-policy-the-TP-group-s-broadcast-on-the-policy-s.patch)
(provisional number 0185). CPU tests pass on dgx4 ([runs/tests.txt](runs/tests.txt)).
**Gated on the TP3 triangle on 2026-10-20 ([Results](#results-tp3-2026-10-20)): outputs
identical to r6x's, no release metric changed, about 0.1 GiB a rank freed.** The release tests
(`scripts/release_tests.py gate`) and the ring have not run; promotion is the owner's call.

## What production does today

DSpark's adaptive verification broadcasts its confidences from rank 0 every decode step
(`record_confidences`, `tp_group.broadcast(self._confidence_probs, src=0)`; max_num_seqs x drafts
FP32, 320 bytes on TP4), so that CPU graph selection and GPU request compaction read one TP-wide
snapshot. `GroupCoordinator.broadcast` calls `torch.distributed.broadcast` on the group's device
process group directly. It never reaches `CudaCommunicator.broadcast`, where 0142 put the policy
guard, so every decode step leaves the policy:

- the PyTorch NCCL process group's communicator, a second NCCL communicator beside PyNCCL,
  is created lazily on the first step (0142 puts that process group at about 3.4 GB of unified
  memory a rank; this experiment measures it);
- the call runs on that process group's own stream, with its events and its 600 s watchdog,
  not the one-shot runtime's fail-stop: the Poisson crash's stalled ranks sat in this
  broadcast ([2026-10-14-poisson-crash](../2026-10-14-poisson-crash));
- it is an NCCL ring broadcast of 320 bytes, launched eagerly between the step's graphs.

The candidate's own test file proves the bypass on the untouched r6x sources:
`test_group_broadcast_goes_through_the_policy` fails there with "the RoCE policy reached the
PyTorch process group".

## The change (vLLM 0185)

- `GroupCoordinator.broadcast` hands the tensor to a device communicator that owns broadcasts
  (`CudaCommunicator.owns_broadcast`, true under the RoCE policy); other groups keep the
  process group's broadcast. `StatelessGroupCoordinator` already hands CUDA broadcasts to its
  device communicator.
- On the TileLang family the communicator broadcasts through sparknet's one-shot all-gather and
  keeps the source's slice: a byte copy, in the model stream with the step's other one-shot
  collectives, under the same timeout and fail-stop. A tensor the all-gather cannot take raises;
  nothing reaches NCCL. The B12X family keeps PyNCCL.

Outputs cannot change: every rank receives rank 0's bytes, as before.

The arms mount the two changed files ([overlay/](overlay)) over the r6x image, whose tree is the
candidate's base: the change is Python only, so this is the candidate's image.

| File | What it is |
| --- | --- |
| [broadcast-tp4.json](broadcast-tp4.json) | r6x on the ring with the overlay |
| [broadcast-tp4-dev.json](broadcast-tp4-dev.json) | the same with the dev-mode collective RPC, for the release suite's grid |
| [gate-tp4.json](gate-tp4.json) | r6x's determinism gate on the candidate |
| [release-tp4.json](release-tp4.json) | release and single-stream suites: candidate, r6x, candidate again |
| [broadcast-tp3.json](broadcast-tp3.json), [broadcast-tp3-dev.json](broadcast-tp3-dev.json) | the same on the triangle at 1M (r6x-tp3-1m) |
| [gate-tp3.json](gate-tp3.json), [release-tp3.json](release-tp3.json) | the triangle's gate and release comparison (release-tp3 and single-stream suites) |
| [runs/run-tests.sh](runs/run-tests.sh) | the CPU tests in a throwaway container on an idle node |

## Gates, in order

On whichever shape is cabled (the triangle on 2026-10-20); the other shape follows with the same
overlay.

1. **Determinism** ([gate-tp4.json](gate-tp4.json), [gate-tp3.json](gate-tp3.json)):
   temperature-0 and greedy outputs identical to r6x's on that shape, zero token and
   log-probability differences in every scenario (on the ring r6x's gate equalled r6v's:
   3eab5112 / db25fde3, c9cf9dbe / c6bb4f4e).
2. **Memory**: the PyTorch NCCL communicator is gone. Compare the arms' KV cache size lines and
   the suites' recorded memory floors (`conditions.json`) between the candidate and r6x.
3. **Release comparison** ([release-tp4.json](release-tp4.json),
   [release-tp3.json](release-tp3.json)): no metric worse than r6x. The step-time effect is
   expected to be small (well under 1%); the reasons to adopt are the policy hole, the second
   communicator and its memory, and a single failure model.

## Acceptance

Outputs identical to r6x's, release tests and the policy suite pass, no release metric worse than
r6x's. Rollback: the arms mount files over the promoted image, so restoring production is the
window's close.

## Results (TP3, 2026-10-20)

Shared lab queue on the triangle, r6x image with the overlay (vLLM d5bfe0203).

**Gate** ([gate-tp3.json](gate-tp3.json), 10:57-11:07 UTC,
[runs/broadcast-tp3-gate-table.txt](runs/broadcast-tp3-gate-table.txt)):
temperature-0 (62313cb5 / 9742ac2c, log probabilities included) and greedy (4d143977 / 12fd8257)
outputs equal r6x's triangle references, zero token and log-probability differences in every
scenario. Distinct streams: c8 146.3 tok/s, c4 112.2 (r6x's TP3 gate on 2026-10-19: 147.6 and
112.5).

**Memory:** MemAvailable once serving, dgx1/dgx2/dgx3: 6.77 / 8.16 / 8.23 GiB against r6x's gate
6.70 / 8.04 / 8.11 (another day), so the PyTorch NCCL communicator the broadcast created cost
about 0.1 GiB a rank on this path, not 0142's "about 3.4 GB".

**Release comparison** ([release-tp3.json](release-tp3.json), 11:07-13:30 UTC; candidate, r6x,
candidate again; [runs/broadcast-tp3-release-table.txt](runs/broadcast-tp3-release-table.txt),
[runs/broadcast-tp3-release-single-stream-table.txt](runs/broadcast-tp3-release-single-stream-table.txt)):

| Metric | Change | ±95% | Repeat | Verdict |
| --- | ---: | ---: | ---: | --- |
| decode geometric mean (15 shapes) | +0.29% | 0.14% | +0.44% | same |
| c8 output tok/s | -0.11% | 1.12% | -0.05% | same |
| c4 output tok/s | -0.29% | 1.83% | +0.16% | same |
| Poisson output tok/s | +0.07% | 22.16% | -0.05% | same |
| prefill 32K / 131K / 262K / 500K | -0.37 / -0.24 / -0.20 / -0.17% | 0.21-0.95% | +0.32-0.43% | same |
| c1 output tok/s (single-stream) | +0.09% | 7.12% | -0.03% | same |

Model behaviour: greedy responses identical 256/256, teacher-forced distributions identical
over 65,536 positions (KL 0), perplexity 1.31340 both.

**Reading:** as expected, no speed effect. What the change buys is structural: the TP group's
broadcast now obeys the RoCE policy (no PyTorch process group, no second NCCL communicator
with its own stream and 600 s watchdog), runs in the model stream under the one-shot runtime's
timeout and fail-stop, and raises instead of reaching NCCL for a tensor the all-gather cannot
take.
