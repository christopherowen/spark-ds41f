# Copy drafter reach: what it does not take

Date: 2026-11-17. Base commit: `45f67d1b4` (main). Offline replays of dumped turns, and one
observe run for the request tokens the first dumps lack.

## Question

Copy steps are 3% of the steps of the AgentX hour and carry 17% of its tokens
([../2026-11-17-draft-calibration-sizing](../2026-11-17-draft-calibration-sizing/README.md),
"Drafts past the fifth"), and a six-token match was there at a fifth of the block-ended steps
without a copy being taken. Is there throughput in the copies the drafter does not take?

## What the drafter does

`copy_drafts.py`, `tilelang_copy.py` and `CopyDraftsConfig`:

- a request whose context ends with the same 16 tokens (`min_match`; or 32, 64, ... 256) as an
  earlier position of its own prompt and output is offered that position's continuation in place
  of DSpark's drafts: the longest level, then a source with 16 tokens to offer, then the latest;
- a first proposal offers 8 drafts, doubled to 16 after a proposal accepted in full, back to 8
  after a rejection;
- it is taken when `offered * estimate` is at least DSpark's survival yield. The estimate starts
  at 0.95 (19 hits of 20 trials, counts decayed by 0.9), so 8 offered drafts beat any DSpark
  block: the gate almost never refuses;
- in a context past 32K tokens the scan is dense over the newest 16K and the first 16K tokens
  and reads the rest only for 256-token matches;
- the scheduler lays out a request's rows two steps ahead, so the step that starts a copy has
  the rows of a DSpark block.

So the six-token matches were below the minimum match, not refused. Three places a copy can be
lost: a match shorter than 16, a source outside the scan, and the start of a copy run.

## One variable

The drafter's rule in a replay: `min_match` (and the first offer), then the scan. Serving is
unchanged.

## Method

[copy_replay.py](copy_replay.py) reads the dumped turns (every decode step's anchor, DSpark drafts
and committed labels) and replays the rule on each turn's history:

- `stats`: at every step's anchor, the longest match in the history, how many of its next
  tokens the stream then commits, and what the step did;
- `simulate`: the turn walked from its first anchor with a given `min_match`. A step serving
  copied is taken as it was. Elsewhere a match proposes by the rule and commits its accepted
  drafts and the target's token; without one, DSpark's step is the dumped one (from an anchor the
  dump does not have, the rest of the dumped step it falls in). A step costs
  `1 + rho * rows verified`: a copy's offered drafts, a DSpark step's rows from the serving form
  at 0.2.

```sh
python copy_replay.py --turns <turns> --features <dir of features.npz>
```

`features.npz` is `size_calibration.py features` of the draft-calibration experiment over the
same turns.

## Runs

### The AgentX hour of 2026-11-17 (output history only)

The 931 turns of boot `20261117T020420` (candidate-60, tp4-switched, 8 sessions), whose dumps
hold no prompt: the history is the turn's committed output, with holes inside copy steps past
their fifth token. 864 turns with two or more steps, 124,989 steps, 2.80 tokens a step
([receipts/replay-output-only.json](receipts/replay-output-only.json)).

Longest match in the turn's own output at a step's anchor:

| Match | Steps | The step committed | First copied token right | Right of the first 8 | A copy of 8 beats the step |
| --- | ---: | ---: | ---: | ---: | ---: |
| none of 4 or more | 96.5% | 2.73 | | | |
| 4 to 5 | 1.3% | 3.69 | 0.638 | 2.53 | 28% |
| 6 to 7 | 0.6% | 4.49 | 0.798 | 3.98 | 42% |
| 8 to 11 | 0.6% | 4.65 | 0.872 | 4.67 | 53% |
| 12 to 15 | 0.4% | 4.89 | 0.887 | 5.29 | 59% |
| 16 or more | 0.5% | 7.45 | 0.912 | 4.89 (holes cut the count) | |

A shorter minimum match, simulated (throughput against the steps as dumped, %, [95% over turns]
at `rho` 0.11):

| `min_match` | First offer | Copy steps added | Offered, accepted a copy | `rho` 0.09 | 0.11 | 0.23 | 0.5 |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 16 (the replay's floor) | 8 | 0.3% of steps | 8.5, 5.2 | −0.03 | −0.04 [−0.06, −0.03] | −0.09 | −0.14 |
| 12 | 8 | 0.6% | 9.5, 5.3 | −0.11 | −0.14 [−0.17, −0.11] | −0.26 | −0.41 |
| 8 | 8 | 1.0% | 9.6, 5.2 | −0.19 | −0.23 [−0.29, −0.18] | −0.45 | −0.70 |
| 6 | 8 | 1.5% | 9.5, 4.8 | −0.35 | −0.43 [−0.51, −0.35] | −0.75 | −1.13 |
| 4 | 8 | 2.5% | 9.1, 3.7 | −0.90 | −1.03 [−1.17, −0.89] | −1.59 | −2.25 |
| 8 | 4 | 0.6% | 5.2, 3.2 | −0.07 | −0.08 [−0.11, −0.05] | −0.10 | −0.13 |
| 6 | 4 | 0.9% | 5.1, 2.9 | −0.13 | −0.13 [−0.17, −0.10] | −0.17 | −0.22 |
| 4 | 4 | 1.8% | 4.8, 2.2 | −0.39 | −0.41 [−0.47, −0.35] | −0.49 | −0.59 |

The row at 16 is the replay's own error: where a 16-token match is in the output and the copy
would give six tokens or fewer, the dump cannot tell serving's copy from a DSpark step, and the
replay adds the copy again.

The start of a copy run, from the tokens each step committed: 666 runs of copy steps (3,798
steps, 59,817 tokens; 237 of one step, 73 of ten or more). The step before a run committed
exactly six tokens at 81% of them, and at 206 of those 539 DSpark's own five drafts were not all
right: that step was the copy, held to five drafts by the rows laid out two steps earlier. The
first step past it committed 17 tokens at 48% of the runs and 9 at 21%.

### Observe run with the request tokens (tp4-switched)

[observe-tp4-switched.json](observe-tp4-switched.json) on tp4-switched (sp1..sp4): AgentX at 8
sessions for 30 minutes on [tp4-switched-observe-tokens.json](tp4-switched-observe-tokens.json),
the turn-learning observe arm (candidate-60, learner in observe mode: no draft, weight or output
changes) with this folder's copy of `turn_learning.py`
([overlay](overlay/vllm/v1/worker/gpu/spec_decode/dspark/turn_learning.py)), whose dump adds each
finished request's whole token row. The row's length is read on the host, one synchronization a
finished turn on rank 0: a data-collection run, not a speed comparison.

```sh
bin/spark candidate queue tp4-switched experiments/2026-11-17-copy-drafter-reach/observe-tp4-switched.json --apply
```

First attempt (queue entry `20261117T204003`, at `fa29a52cf`): the boot failed on every rank
with `ModuleNotFoundError: ... dspark.turn_learning_shadow`. The turn-learning experiment's
speculator now imports its shadow module in every mode, and its observe profile, which this
arm extends, predates that and does not mount it. The arm now mounts it too. No request was
served and nothing was dumped.

Second attempt: queue entry `20261117T204553` at `bb33a560d`, sp1..sp4, boot `20261117T204820`
ready in 135.9 s, 20:46 to 21:24 UTC ([receipts/observe-run.txt](receipts/observe-run.txt)).
AgentX: valid, 323 requests, 0 errors, 153.8 output tok/s, cache hit 97.6% of 97.9% ideal. 445
turns dumped (2.7 GB, left in `cache/copy-reach` on sp1); every step's anchor token is the token
row's at its position.

### The same replay on whole histories

379 turns with two or more steps, 49,589 steps, 2.84 tokens a step, 2.9% of steps copied; the
median history is 89,007 tokens ([receipts/replay-with-tokens.json](receipts/replay-with-tokens.json)).

Longest match anywhere in the request's prompt and output at a step's anchor:

| Match | Steps | Served as a copy of 7+ | The step committed | First copied token right | Right of the first 8 | A copy of 8 beats the step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none of 4 or more | 86.6% | 0% | 2.27 | | | |
| 4 to 5 | 3.7% | 0.2% | 3.42 | 0.641 | 2.64 | 32% |
| 6 to 7 | 1.6% | 0.4% | 4.10 | 0.793 | 3.78 | 43% |
| 8 to 11 | 2.2% | 0.5% | 4.35 | 0.849 | 4.40 | 49% |
| 12 to 15 | 1.2% | 1.6% | 4.67 | 0.872 | 5.21 | 61% |
| 16 or more, in serving's scan | 4.3% | 66% | 11.93 | 0.950 | 6.79 | |
| 16 or more, outside serving's scan | 0.4% | 0% | 4.53 | 0.927 | 5.86 (9.76 of 16) | 71% |

No step with a match outside the scan was served as a copy, and two thirds of those inside it
were (the rest are a run's first step, held to five drafts, and copies that ran six tokens or
fewer): the replay's scan is serving's.

Simulated, against the replay's own floor (`min_match` 16 with serving's scan, which reads
−0.34% [−0.45, −0.24] against the steps as dumped at `rho` 0.11: copies of six tokens or fewer
that the dump cannot tell from DSpark steps, added again), %:

| Rule | Copy steps added | `rho` 0.09 | 0.11 | 0.23 | 0.5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16, the whole history scanned | +0.3% of steps | +0.06 | +0.04 | −0.05 | −0.15 |
| 12 | +1.2% | −0.05 | −0.12 | −0.42 | −0.77 |
| 8 | +2.7% | −0.40 | −0.53 | −1.16 | −1.89 |
| 6 | +4.0% | −0.82 | −1.03 | −1.97 | −3.05 |
| 4 | +6.9% | −2.21 | −2.57 | −4.14 | −5.93 |
| 6, first offer 4 | +2.6% | −0.14 | −0.12 | −0.06 | +0.02 |
| 4, first offer 4 | +4.9% | −0.69 | −0.71 | −0.78 | −0.86 |

(`8, first offer 4` reads +0.02 to +0.34, but its smaller first offer also applies to the
floor's own added copies, so it is not a gain of the rule.)

## Conclusion

- **There is no throughput in the copies the drafter does not take.** With the prompts in the
  history, every change of the rule is inside ±0.1% or a loss.
- **The scan misses little.** A 16-token match outside the newest and the first 16K tokens is
  there at 0.4% of steps. Scanning the whole history gains +0.04% at the measured row cost.
- **A shorter minimum match loses, more so with the prompt.** Matches of 4 to 15 tokens are at
  8.7% of steps in an 89K-token history, a copy from one runs 2.6 to 5.2 tokens of its first 8,
  and the DSpark steps it would replace commit 3.4 to 4.7. Sixteen is about where a copy starts
  to pay.
- **The start of a copy run is worth at most 0.4%** (the first proposal is held to five drafts
  by the rows laid out two steps earlier), before the cost of the rows.
- **The gate is not a lever.** With its estimate at 0.95 it takes every match of 16 or more.
- **Where the steps are:** 86.6% of steps have no 4-token match anywhere in the request and
  commit 2.27 tokens. That text is new, and only the drafter's own accuracy moves it.

## Limits

- AgentX at 8 sessions on candidate-60; the copy drafter of candidate-61 differs in its guards
  and its cap's decay, not in its match rule.
- The replay follows the dumped steps: from an anchor the dump does not have, a DSpark step is
  the rest of the dumped step it falls in.
- A linear row cost; a copy's sixteen rows with the target's are past one 16-row tile.
- At most 48 of the latest sources of a context are compared, and matches are counted to 64
  tokens.
