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

## Conclusion (so far)

- **A shorter minimum match loses.** Short matches are rare in a turn's own output (2.9% of
  steps between 4 and 15 tokens), a copy from one runs 2.5 to 5.3 tokens of its first 8, and the
  DSpark steps it would replace are already better than average (3.7 to 4.9 tokens against 2.7).
  Eight verified rows for that is a loss at every row cost.
- **The start of a copy run is worth at most 0.4%.** A run loses about 11 of 16 draft slots at
  its first step and 8 at a fifth of its second steps: under 530 steps of 124,989 if every run
  were long enough to use them, before the cost of the rows.
- **The gate is not a lever.** With its estimate at 0.95 it takes every match of 16 or more.
- **Open: sources outside the scan, and copyable text where DSpark misses.** Both need the
  prompt. The observe run below dumps it.

## Observe run with the request tokens

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

## Limits

- AgentX at 8 sessions on candidate-60; the copy drafter of candidate-61 differs in its guards
  and its cap's decay, not in its match rule.
- The replay follows the dumped steps: from an anchor the dump does not have, a DSpark step is
  the rest of the dumped step it falls in.
- A linear row cost; a copy's sixteen rows with the target's are past one 16-row tile.
- At most 48 of the latest sources of a context are compared, and matches are counted to 64
  tokens.
