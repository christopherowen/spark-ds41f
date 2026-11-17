# Draft acceptance prediction: how much is left

Date: 2026-11-17. Base commit: `d0a111fa2` (`c61-decode`). Data: the 931 AgentX turns the
turn-learning observe stage dumped on `candidate-60`
([../2026-11-17-dspark-turn-learning](../2026-11-17-dspark-turn-learning/README.md), boot
`20261117T020420`, 8 sessions, one hour). Offline only: no boot, no serving change, no GPU time on
a node.

## Question

A decode step verifies the leading drafted rows whose predicted survival is above a threshold.
Rows the target then rejects are paid for and return nothing: 1.19 of 2.59 drafted rows a step
on `c61-r13`'s single stream. How much throughput would a better predictor of acceptance recover,
at the drafts serving already makes?

## One variable

The predictor of `P(draft i accepted | drafts before it accepted)`. The drafts, the labels and the
policy (verify while the running product of the predictions is at least `tau`) are the same for
every predictor.

## Method

[size_calibration.py](size_calibration.py), two stages:

1. `features`, one pass over the turns. For each drafted position: the confidence head's logit
   `z` (`mtp.2.confidence_head.proj.weight` over the pre-norm head row and the Markov embedding
   of the token before it, as the draft step computes it), the draft distribution's top-1 to
   top-2 margin and `log p_max` (full-vocabulary Markov logits, FP32), and whether the draft is
   the committed token. The FP32 argmax equals the dumped draft at 99.90% of positions.
2. `study`. The first 465 turns in the order they finished fit each predictor; the other 466
   (73,149 draft steps) score it.
   - `serving`: `sigmoid(w * z + b_pos)`, the online estimator's form (shared slope, an
     intercept a position);
   - `posslope`: a slope a position;
   - `draft`: the margin and `log p_max` only, no confidence head;
   - `z+draft`: both; `mlp`: a 64x64 MLP on the same scalars;
   - `probe`: the head's linear form over its own 5,376 inputs, refitted on these turns as a
     residual over `z`; `probe+draft`: that with the draft features (cross-fitted);
   - `oracle`: the label.
   Policy value: tokens a step are 1 plus the accepted rows among those verified; a step costs
   `1 + rho * rows verified`, with `rho` the marginal cost of a verified row over the step's fixed
   cost. `tau` is chosen on the first half for each predictor and `rho`. Intervals are 95%
   bootstrap over the test turns.

`rho`: `c61`'s single-stream rounds put a verified row at 2.0 to 3.0 ms of a 28.0 ms step with
2.59 rows (`r1`, `r6`, `r11`), which is 0.09 to 0.15; 0.23 is the isolated routed-MoE row cost on
sp8; 0.5 and 0.75 stand for many streams, where a request's share of the fixed cost is small
(an estimate from the 16-stream step, not a measurement).

[followup.py](followup.py) refits with the margin alone. [longer_block.py](longer_block.py) puts
each turn's steps in time order and sizes drafts past the fifth
([below](#drafts-past-the-fifth)).

```sh
python size_calibration.py features --turns <turns> --ckpt <snapshot> --out <out>
python size_calibration.py study --out <out>
python followup.py <out>
python summarize.py <out>/study.json
python longer_block.py features --turns <turns> --ckpt <snapshot> --out <out>
python longer_block.py study --out <out>
python longer_block.py summary --out <out>
```

`<turns>` is `~/spark-lab/agents/dspark-turn-learning/turns` on sp8; `<snapshot>` the
`DeepSeek-V4.1-Flash-1017` snapshot `1bae0e10`. Run on a laptop GPU (PyTorch 2.15, MPS) from a
copy of the dumps, because sp8's GPU was in use: 147 s for `features`, 41 s for
`longer_block.py features`, under a minute for the rest.

## Runs

Receipts: [receipts/study.json](receipts/study.json), [receipts/summary.txt](receipts/summary.txt),
[receipts/followup.json](receipts/followup.json), [receipts/longer.json](receipts/longer.json),
[receipts/longer-summary.txt](receipts/longer-summary.txt).

931 turns, 126,328 draft steps, 623,290 labeled positions.

Acceptance given the drafts before it were accepted, by position: 0.674, 0.657, 0.650, 0.660,
0.672. The raw head's mean probability there: 0.788, 0.724, 0.683, 0.659, 0.653 (the online
intercepts correct this level). Accepted drafts a step (test half): none 33.0%, one 23.3%, two
15.5%, three 9.8%, four 6.2%, all five 12.2%.

Discrimination on the test half:

| Predictor | AUC | Log loss |
| --- | ---: | ---: |
| position only | 0.512 | 0.641 |
| `serving` | 0.823 | 0.468 |
| `posslope` | 0.823 | 0.468 |
| `probe` | 0.838 | 0.453 |
| margin only | 0.847 | 0.442 |
| `z` + margin | 0.863 | 0.422 |
| `draft` | 0.865 | 0.423 |
| `z+draft` | 0.871 | 0.414 |
| `mlp` | 0.871 | 0.413 |
| `probe+draft` | 0.871 | 0.414 |

Throughput against `serving` at its own best `tau`, % [95% interval]:

| Predictor | `rho` 0.09 | 0.11 | 0.15 | 0.23 | 0.5 | 0.75 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `posslope` | −0.02 | +0.01 | −0.01 | −0.00 | −0.01 | +0.00 |
| `probe` | +0.54 | +0.62 | +0.71 | +0.79 | +0.42 | +0.11 |
| margin only | +0.60 | +0.63 | +0.62 | +0.80 | +0.73 | +0.17 |
| `z` + margin | +1.29 [1.20, 1.38] | +1.41 [1.31, 1.50] | +1.61 [1.51, 1.72] | +1.81 [1.68, 1.94] | +1.45 [1.32, 1.57] | +0.60 [0.52, 0.68] |
| `draft` | +1.43 | +1.60 | +1.76 | +1.97 | +1.12 | +0.23 |
| `z+draft` | +1.66 [1.55, 1.75] | +1.80 [1.70, 1.91] | +2.06 [1.94, 2.19] | +2.25 [2.10, 2.38] | +1.51 [1.40, 1.62] | +0.62 [0.54, 0.69] |
| `mlp` | +1.65 | +1.83 | +2.02 | +2.26 | +1.51 | +0.65 |
| `probe+draft` | +1.63 | +1.81 | +2.03 | +2.25 | +1.45 | +0.60 |
| `oracle` | +14.4 | +15.8 | +17.9 | +19.5 | +14.9 | +7.7 |
| verify every drafted row | −9.0 | −11.2 | −15.4 | −22.6 | −39.1 | −48.4 |

At `rho` 0.11 `serving` verifies 2.59 rows a step for 2.52 tokens, 1.07 of the rows rejected;
`z+draft` verifies 2.42 with 0.89 rejected; the oracle verifies 1.69.

After a step whose five drafts were all right (12.7% of steps; 15,755 pairs in time order), the
next step's drafts are right for 3.24 positions on average against 1.72 overall, its first draft
at 0.887 against 0.674, and all five again 42.1% of the time. `followup.py`'s `next_step` block
pairs the ring's rows in file order, not time order (3.23, 0.885, 41.9%); `longer_block.py`'s
numbers replace it.

## Conclusion

- **The lever is about 2%, not 15%.** The oracle's 14 to 20% is not reachable from what the draft
  step knows: a logistic on three scalars, an MLP and the refitted head with the draft features
  all stop at AUC 0.871.
- **What there is comes from the draft distribution itself.** The top-2 margin and `log p_max`
  alone predict acceptance better than the confidence head (AUC 0.865 against 0.823). The head's
  logit with the margin, which the greedy pairs can already return, gives +1.3 to +1.8% at
  single-stream row costs; adding `log p_max` (a sum over the vocabulary the certified screen
  avoids) reaches +1.7 to +2.3%.
- **A slope a position and refitting the head on this traffic do little** (0.0% and +0.5 to
  +0.8%).
- **At many-stream row costs the gain is smaller** (+0.6 to +1.5%): fewer rows are verified, so
  fewer are wasted.
- **The five-draft block is not the tighter limit** (corrected; the first reading of these runs
  said it was). One step in eight has all five drafts right, but under half of those end at the
  block; [drafts past the fifth](#drafts-past-the-fifth) are worth less than this lever.

## Limits

- AgentX turns at 8 sessions, not the single-stream prose bench: the size on prose is not
  measured.
- The drafts come from `candidate-60`'s policy; a draft is dumped only for the positions the step
  kept active (five at 99.8% of steps).
- `serving` is the estimator's form fitted on half the hour, not its online state at each step;
  the stale-confidence effects `c61-r10` removed are not modelled.
- `rho` is a linear cost. A row that opens another decode tile costs more than `rho`.
- Every step is scored as a DSpark step of at most six tokens. Serving verified a copy proposal
  at 3.3% of these steps (17.7% of the tokens); with them at their real yield the gains above
  shrink by about a twentieth.
- No serving speed was measured.

## Drafts past the fifth

### Question

A step whose five drafts are all accepted ends there; the target's own token is the sixth. What
would drafts for positions six and later be worth?

### What the dump holds

The ring keeps a row for every decode step of a request (its last 256), whichever proposal the
step verified, and the rows are in ring order. In anchor order the distance to the next anchor is
the number of tokens the step committed, and the next anchor is the label it lands on at 99.8%
of steps. A step that committed more than six tokens verified a `CopyDrafter` proposal (up to 16
drafts), not DSpark's five. 125,254 steps have a next step:

| The step ended at | Steps | Tokens | Tokens a step |
| --- | ---: | ---: | ---: |
| a miss (the next draft was wrong) | 76.0% | 54.4% | 2.00 |
| the survival cut (right drafts left unverified) | 14.6% | 14.8% | 2.84 |
| the five-draft block (six committed) | 6.1% | 13.1% | 6.00 |
| a copy proposal, more than six committed | 3.0% | 17.1% | 15.75 |
| a copy proposal, six or fewer | 0.3% | 0.6% | 5.51 |

Of the steps whose five DSpark drafts were all right (12.7%), 48.2% ended at the block, 34.0% at
the survival cut and 17.8% were copy steps. The step after a block-ended step (7,603 pairs) ends
on a miss 48.1% of the time, at the block again 34.8%, and is a copy step 6.7%; its first DSpark
draft is right at 0.871. Block-ended steps come in runs: 3,470 of one, 917 of two, 610 longer;
a miss ends 73.2% of the runs. After a miss at the fifth draft the next step's first draft is
right at 0.751 (0.615 after a miss at the first), so a deep miss is in part the depth's.

### Method

[longer_block.py](longer_block.py). `features` measures, after every step whose five drafts were
right, the sources of a later draft that need no new drafter, against the committed stream:

- `row5`: the Markov chain continued on the fifth row's base logits;
- `markov`: the Markov bias alone;
- `local`: the continuation of the most recent earlier occurrence of the last 6, 4 or 3 tokens in
  this turn's committed output (what a copy match appended to DSpark's five would see of the
  output; the prompt is not in the dumps);
- `lookup`: the same, also over the outputs of the turns that finished before it, as a stand-in
  for the prompt (it reads other sessions' turns too).

`study` values a block of `5 + m` drafts at acceptance `q` of each added draft given the ones
before it were accepted, and at the measured sources, on the second half of the turns:

- **ceiling**: no step ends at the block (any length, every added draft right, rows free);
- **merge**: a step is saved only where added drafts carry a step to the end of a later one. A run
  of block-ended steps and the step that ends it are covered again with steps of `6 + e` tokens,
  `e` the accepted added drafts; the later steps end where they did. A run a copy step ends gets
  no credit;
- **stationary**: every accepted added draft is a token the later steps need not produce, at the
  average step yield. It over-credits (the later steps keep their yield) and passes the ceiling
  for a long perfect block.

A step costs `1 + rho * rows verified`. Rows: the serving form at serving's 0.2 survival
threshold on DSpark steps (five on a block-ended step), the accepted drafts on copy steps: 2.89
a step for 2.72 tokens. An added row is verified where five are and the predicted survival to it
is at least a threshold, chosen on the first half for each case and `rho`; never adding a row is
one of the choices.

### Runs

Added drafts after a block-ended step (7,640), right given the ones before were right:

| Source | Sixth | Seventh | Eighth | Ninth | Tenth | Accepted a step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `row5` | 0.485 | 0.281 | 0.211 | 0.178 | 0.256 | 0.66 |
| `markov` | 0.071 | 0.053 | 0.103 | 0.000 | | 0.08 |
| `local` (offered at 32.5%) | 0.248 | 0.832 | 0.844 | 0.880 | 0.876 | 0.91 |
| `lookup` (offered at 68.2%) | 0.457 | 0.765 | 0.801 | 0.844 | 0.842 | 1.52 |

A six-token match is there at 20.3% of block-ended steps in the turn's own output (first right
0.843, 3.34 accepted) and at 42.3% with the earlier turns (0.772, 2.79). `row5` and `lookup`
name the same sixth token at 25.5%, and are right 0.938 there. Position five replayed from the
fourth row is right at 0.413 against the real fifth draft's 0.672 (23,590 steps): a row one
position stale loses a third of its acceptance.

Second half: 72,981 steps, 6.0% block-ended, 13.8% verify five rows (43.5% of those end at the
block). Throughput gain, %:

| Case | Reading | `rho` 0.09 | 0.11 | 0.15 | 0.23 | 0.5 | 0.75 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| any block, perfect drafts, free rows | ceiling | +5.01 | +4.78 [4.30, 5.31] | +4.38 | +3.75 | +2.52 | +1.93 |
| `q` 0.67, one added draft | merge | +0.06 | +0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| | stationary | +0.69 | +0.61 [0.53, 0.70] | +0.47 | +0.30 | +0.06 | 0.00 |
| `q` 0.67, five added | merge | +0.06 | +0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| | stationary | +1.00 | +0.86 [0.74, 0.99] | +0.59 | +0.30 | +0.06 | 0.00 |
| `q` 0.885, five added | merge | +0.30 | +0.15 | +0.01 | 0.00 | 0.00 | 0.00 |
| | stationary | +3.71 | +3.28 [2.86, 3.74] | +2.61 | +1.71 | +0.58 | +0.27 |
| `q` 1, one added | merge | +0.25 | +0.17 | +0.05 | 0.00 | 0.00 | 0.00 |
| | stationary | +1.31 | +1.21 [1.07, 1.38] | +1.06 | +0.82 | +0.45 | +0.31 |
| `q` 1, five added | merge | +0.61 | +0.32 | 0.00 | 0.00 | 0.00 | 0.00 |
| | stationary | +6.39 | +5.93 [5.23, 6.70] | +5.12 | +3.96 | +2.18 | +1.52 |
| `row5`, any length | merge | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| | stationary | +0.40 | +0.34 [0.28, 0.42] | +0.25 | +0.12 | 0.00 | 0.00 |
| `local`, five added | merge | +0.09 | +0.04 | 0.00 | 0.00 | 0.00 | 0.00 |
| | stationary | +1.09 | +0.95 [0.73, 1.20] | +0.76 | +0.48 | +0.12 | +0.05 |
| `lookup`, six-token matches, five added | merge | +0.13 | +0.06 | 0.00 | 0.00 | 0.00 | 0.00 |
| | stationary | +1.49 | +1.30 [1.08, 1.54] | +1.07 | +0.68 | +0.18 | +0.07 |

0.67 is the acceptance at the fifth draft given four accepted; 0.885 is the next step's first
draft after five right drafts, from a fresh anchor. The other lengths and sources are in the
receipts.

### Conclusion

- **Drafts past the fifth are worth between nothing and about 1.5% on this traffic, and at most
  4.8%.** The ceiling is the whole of it: 6% of steps end at the block.
- **Serving already drafts past five where it pays most.** Copy steps are 3% of the steps and
  carry 17% of the tokens, at 15.75 tokens a step.
- **A saved step is rare.** After a block-ended step the next step ends on a miss half the time,
  and added drafts that stop short of that miss only move tokens from the next step to this one.
  Under the merge reading a perfect sixth draft gains at most 0.3% and five perfect added drafts
  at most 0.6% at single-stream row costs, nothing at many-stream costs.
- **The added rows are paid for more often than they are used.** Under half of the steps that
  verify five rows end at the block.
- **The free sources are weak or narrow.** A stale row is right at the sixth position under half
  the time. A copy tail appended to DSpark's five is right when a long match exists (0.84 at the
  sixth, 3.3 tokens, for a six-token match in the turn's own output), but that is a fifth of the
  block-ended steps: at most +1.1% from the turn's output, +1.5% with the earlier turns.
- **A third of the steps whose five drafts were right stopped at the survival cut**, not at the
  block. That is the calibration lever above, which is the larger of the two.

### Limits

- The same turns, with the same limits as above: AgentX at 8 sessions, `candidate-60`'s policy,
  no serving speed measured.
- The two readings bracket the value; neither is a replay. Drafts from a moved anchor are not in
  the dumps, so what the next step does after this one takes its first tokens is assumed.
- The prompt is not in the dumps. `lookup` stands in for it with other turns' outputs, which a
  request's own history may not hold; a match against the real prompt is not measured, and nor
  is why `CopyDrafter` did not take the steps where a six-token match was there.
- A drafter trained on a longer block is not measured; `q` is assumed for it.
- Rows verified by serving are not dumped; they are modelled with the serving form at 0.2.
- A copy proposal of five or fewer accepted drafts cannot be told from a DSpark step unless it
  committed more than DSpark's right drafts allow (0.3% of steps).

## The margin in serving

### One variable

The feature the online estimator calibrates: `logit(p) + 2 * log(1 + margin)` in place of
`logit(p)`, with `p` the confidence head's probability and `margin` the gap between the drafted
token's logit and the runner-up's. For the screen it sat behind a switch
(`SPARK_DS41F_DSPARK_MARGIN_CONFIDENCE=2`, at `29cbe60cb`), so that two arms at one commit
differed in it alone. The switch is gone: see [Integration](#integration).

### Which term

[margin_gain.py](margin_gain.py) on the cached features
([receipts/margin-gain-agentx.json](receipts/margin-gain-agentx.json),
[receipts/margin-gain-coding.json](receipts/margin-gain-coding.json)). One term with one constant
carries what the four margin terms a position did:

| Form (first half fits, second half scores) | AUC | Gain at `rho` 0.09 | 0.11 | 0.15 | 0.23 | 0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `serving` | 0.823 | | | | | |
| `z`, four margin terms a position | 0.863 | +1.29 | +1.41 | +1.61 | +1.81 | +1.45 |
| `z + c log1p(margin)`, `c` fitted (2.29) | 0.863 | +1.24 | +1.40 | +1.57 | +1.76 | +1.41 |
| `z + 2 log1p(margin)` | 0.863 | +1.26 | +1.42 | +1.57 | +1.79 | +1.40 |
| `z + 1 log1p(margin)` | 0.858 | +1.11 | +1.20 | +1.36 | +1.59 | +1.19 |

The same on other traffic: the 228 turns of the observe run's first boot (coding conversations,
held-out prompts and the gates; 16,961 steps, boot `20261117T015531`). Fitted there, the gain is
1.82 and the form gives AUC 0.865 against 0.838, +0.7 to +2.1% (a small sample). The agent
turns' fit applied to those turns unchanged gives AUC 0.886 against 0.859.

### The change

In candidate-61's overlay (as integrated; the screened commit had the same arithmetic behind the
switch, in a second kernel beside the unchanged one):

- [confidence.py](../2026-11-16-c61-decode/overlay/vllm/vllm/models/deepseek_v4_1/tilelang/confidence.py)
  (candidate-60's file, new in the overlay): the fused calibration kernel takes each row's margin
  and calibrates on the sum. The margin is clamped to [0, 64]; a zero margin leaves the head's
  own log-odds.
- `dspark/speculator.py`: the greedy steps always report each draft's runner-up (the top-2
  pairs, declared kernels that the top-2 branch already used; with a zero branch margin no draft
  changes), and the step's margins go to the confidence head. The vocabulary-parallel greedy
  steps measure a margin, which is every draft step of this deployment.
- `nvidia/dspark.py`: `compute_and_calibrate_confidence` passes the margins to a head that takes
  them.
- `acceptance_estimator.py`: the estimator's start for the feature (slope 0.67, intercepts
  −1.41, −1.79, −2.25, −2.25, −2.23): the shipped start moved by the difference between the two
  fits on the agent turns (`warm` in the receipts). The online refit takes over from there.

Every draft is still verified by the target, so no output can change.

### Tests (sp8, candidate-60's image with candidate-61's overlay)

[bundles/margin-tests](bundles/margin-tests), run with `scripts/lab.py kernel-local` from a copy
staged under `~/spark-lab` before each commit was published:

- the kernel (`tests/kernels/test_deepseek_v41_tilelang_confidence.py`): no margin is a zero
  margin, which leaves the head's own log-odds; the feature and the probability against an FP64
  reference; an infinite, negative or NaN margin; rows independent of the batch; only live slots
  recorded; refused layouts;
- `tests/v1/spec_decode/test_margin_confidence.py`: the torch form, the estimator's start, the
  model's dispatch (the margin only to a head that takes it), and the speculator's confidence
  step with a step's margins on the fused and the unfused path;
- the estimator, adaptive-verification and DSpark draft tests of the image.

Result: 190 passed, 1 skipped (needs two CUDA devices)
([receipts/margin-tests](receipts/margin-tests)). At the screened commit, with the switch: 192
passed with it off and 158 with it on. One test is deselected:
`test_dspark_revisited_position_uses_step_seed[True]` fails the same way on main's overlay
without this change (the TileLang sampler refuses FP64 Gumbel draws).

Errors on the way: the first run failed six draft tests, whose stand-in speculators lack the new
attribute; the call site now reads it with a default.

### Screen (tp4-ring)

`screen-margin-tp4.json` at `29cbe60cb`: four boots at one commit, the control
(`tp4-margin-off.json`: candidate-61's dev arm with the new files mounted and the switch unset)
and the margin arm (`tp4-margin.json`) alternating, each with candidate-61's screen workloads
(single stream 150 s, 16 streams 90 s, the greedy determinism record). The first queue attempt
(at `9cf1545f1`) was refused before any boot: the switch was not in candidate-60's qualified
profile ("qualified config drift"), and the arms then named a copy of candidate-60's source
lock of their own, which is how an arm with a variable of its own is declared. The spec, the two
arms and the lock copy were removed with the switch; they are in the history at `40d8ec130`.

### Runs

Queue entry `20261117T182659-screen-margin-tp4` on dgx1..dgx4 at `29cbe60cb`, 18:27 to 19:05 UTC,
four boots back to back. Receipts: [receipts/margin-screen](receipts/margin-screen)
([summary.txt](receipts/margin-screen/summary.txt) from [margin_screen.py](margin_screen.py),
the lab's [table.txt](receipts/margin-screen/table.txt), each arm's servebench results and the
gate's lines).

The margin arm acts: its first boot has the switch in the container's environment and
`confidence.py` mounted, and compiled 35 kernels on each node, `dspark_confidence_margin_calibrate`
and the top-2 selection kernels among them (so that boot's time to ready, 119.1 s, is not a
comparison; the second margin boot compiled none and was ready in 112.4 s, the controls in 118
to 119 s). No failed request in any run. Free memory after each stop was 116.9 to 117.9 GiB on
every node, the same for both arms.

Single stream (150 s after a 30 s warm-up, the same seed and prompts):

| Arm | tok/s | Steps/s | ms a step | Tokens a step | Drafts verified a step | Greedy gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| control | 85.09 | 35.840 | 27.90 | 2.3826 | 2.586 | identical, `0a2dacdd` / `12fd8257` |
| margin | 85.95 | 36.233 | 27.60 | 2.3817 | 2.537 | identical, `0a2dacdd` / `12fd8257` |
| control, second boot | 85.01 | 35.807 | 27.93 | 2.3823 | 2.586 | identical, `0a2dacdd` / `12fd8257` |
| margin, second boot | 86.11 | 36.253 | 27.58 | 2.3840 | 2.539 | identical, `0a2dacdd` / `12fd8257` |

16 streams (90 s after a 30 s warm-up):

| Arm | tok/s | Steps/s | Tokens a step | Rows scheduled a step |
| --- | ---: | ---: | ---: | ---: |
| control | 270.16 | 7.789 | 34.736 | 37.55 |
| margin | 273.39 | 7.900 | 34.629 | 37.40 |
| control, second boot | 270.06 | 7.778 | 34.743 | 37.54 |
| margin, second boot | 273.63 | 7.933 | 34.519 | 37.28 |

Margin against control:

| | Means of two boots | Each boot pair | Paired batches (20) [95%] |
| --- | ---: | --- | ---: |
| Single stream tok/s | 85.05 → 86.03, +1.16% | +1.02%, +1.29% | +1.20% [0.17, 2.22] |
| Single stream steps/s | +1.17% | +1.10%, +1.25% | +1.17% [0.82, 1.52] |
| Single stream tokens a step | +0.01% | −0.04%, +0.07% | +0.04% [−1.11, 1.19] |
| 16 streams tok/s | 270.11 → 273.51, +1.26% | +1.20%, +1.32% | +1.28% [0.62, 1.94] |
| 16 streams steps/s | +1.71% | +1.43%, +2.00% | +1.71% [1.04, 2.39] |
| 16 streams tokens a step | −0.48% | −0.31%, −0.65% | −0.47% [−1.16, 0.23] |

The control is the recorded `c61-r18` again (85.03 and 269.84 on the same code without the new
files mounted): the files and the unset switch change nothing, and two boots of one arm differ by
0.1%.

### Conclusion

- **The margin is a gain on both screen workloads: +1.2% single stream, +1.3% at 16 streams**,
  in both boot pairs, with identical greedy output. The size is what the offline replay gave
  (+1.3 to +1.8% single stream, +0.6 to +1.5% at many-stream costs).
- **It comes from the step, not from more tokens.** Single stream commits the same tokens a
  step in 0.33 ms less; it verifies 1.9% fewer drafted rows. At 16 streams a step is 1.7% faster
  for 0.5% fewer tokens.
- **Not all of the step time is attributed.** 0.05 fewer rows a step at the 2 to 3 ms a row
  measured in `c61` is 0.10 to 0.15 ms of the 0.33 ms. The first draft's acceptance also reads
  higher (0.680 against 0.666) though the drafts are the same, so the mix of step kinds moved
  (the copy drafter's choice reads DSpark's survival too); the screen's counters do not separate
  that.
- Against `candidate-60`'s recorded base (83.23 and 272.92): 86.03 single stream (+3.4%) and
  273.51 at 16 streams (+0.2%, the first `c61` state above the base at 16 streams).

### Limits

- The screen's two workloads on tp4-ring only: no AgentX run (the traffic the lever was sized
  on), no other shape, no sampled requests (their draft steps do not measure a margin and
  calibrate as before), no soak.
- The gain is fixed at 2 and the estimator's start comes from agent traffic; neither was tuned
  on the screen.
- An overlay on candidate-60's image, not a build.

### Integration

The owner's decision (2026-11-17): the margin is the better way, so it is the only way. The
switch, the second kernel and the two arms are removed; the calibration kernel always carries the
margin, the greedy steps always report the runner-up, and candidate-61's dev arm
([tp4-dev.json](../2026-11-16-c61-decode/tp4-dev.json)) mounts `confidence.py` and its tests.
A head without the margin (the image's own `confidence.py`, when an arm does not mount the
overlay's) is called as before and starts from the shipped intercepts.

Two boots of the dev arm at `9649db2dc`
([screen-margin-integrated-tp4.json](screen-margin-integrated-tp4.json), queue entry
`20261117T192746`, dgx1..dgx4, 19:28 to 19:47 UTC; receipts:
[receipts/margin-integrated](receipts/margin-integrated)). The dev arm was candidate-61's round
19 by then, recorded once without the margin (`c61-r19`: 85.28 and 270.31).

| Arm | Single stream tok/s | Steps/s | Tokens a step | Drafts verified a step | 16 streams tok/s | Greedy gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `c61-r19` (recorded, no margin) | 85.28 | 35.913 | 2.3839 | 2.587 | 270.31 | identical |
| dev arm with the margin | 86.07 | 36.280 | 2.3825 | 2.538 | 273.96 | identical, `0a2dacdd` / `12fd8257` |
| the same, second boot | 86.39 | 36.320 | 2.3873 | 2.543 | 274.30 | identical, `0a2dacdd` / `12fd8257` |

The mean of the two boots is 86.23 single stream (+1.1% on round 19) and 274.13 at 16 streams
(+1.4%): the screen's gain again, on the next round's code and without the switch. The first
boot compiled 31 kernels on each node (the calibration kernel changed, and the top-2 selection
kernels are now the only ones); the second compiled none. Ready in 113.5 and 114.6 s, no failed
request, free memory after each stop 116.7 to 118.0 GiB.

Against `candidate-60`'s recorded base (83.23 and 272.92): +3.6% single stream, +0.4% at 16
streams.
