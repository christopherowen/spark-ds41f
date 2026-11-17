# DSpark drafter learning at the end of each turn

PR #1002, second design. Base: `origin/main` `7bec4cd0b`, image `spark-ds41f:candidate-60` (vLLM tree
`afcaac35`); the changed files are mounted over the image from [overlay/](overlay), nothing is built.

The first design ([2026-11-08-online-dspark-adaptation](../2026-11-08-online-dspark-adaptation/README.md))
updated DSpark's Markov W1 and confidence head after every verification, in eager PyTorch on the step's
critical path. It measured −3.8% (coding) and −5.5% (held-out) at one stream: about 0.8 ms a step for the
update itself, and shorter verification budgets from the re-learned confidence path. W1 moved at most 0.5%
of a row's norm during its workload, so the drafts themselves barely changed.

This design keeps the same learner target, DSpark's Markov W1, and moves the learning to the end of each
turn (one request):

- **During the turn**, a TileLang kernel inside the captured draft graph keeps each draft step's `N = 5`
  pre-norm head rows, the anchor's position and the step's drafts in the request's ring (a fixed number of
  draft steps a request, newest kept).
- **When the scheduler reports the request finished**, every rank, in the same step and in sorted order
  and before the slot can be reused, reads the committed tokens at each kept draft position and replays
  the turn with the draft step's own declared kernels (head projection, Markov projection, shard pairs,
  first argmax):
  - with the drafted tokens as the chain's inputs, the replay must give the step's drafts, bit for bit;
  - with the committed tokens as the chain's inputs (teacher forcing), it gives the number of positions
    each block would have had accepted, all five of them, under whatever W1 the chain reads.
- DSpark drafts its block in one parallel pass, so the kept rows do not depend on the drafted tokens;
  only the Markov chain does. A W1 change can therefore be scored exactly on a finished turn before it is
  applied (test, then train).

## Stages

1. **Observe** (this revision, `VLLM_DSPARK_TURN_LEARNING=observe`): capture, end-of-turn replay,
   per-turn counters, and with `VLLM_DSPARK_TURN_LEARNING_DUMP` rank 0 writes each turn's rows and labels
   to disk. No draft and no weight changes. Proves the replay in serving and collects real turns.
2. **Offline study** on the dumped turns (sp8, the checkpoint's own heads): the update rule, its step size,
   and the prequential learning curve, i.e. how much traffic before the adapted W1 beats the loaded one.
3. **Shadow** (`VLLM_DSPARK_TURN_LEARNING=shadow`, [below](#shadow)): the update in TileLang at the end of
   each turn on an FP32 copy of W1 the draft step does not read, from the block maxima the step already
   computes; drafts and outputs unchanged. It measures the learner's own cost in serving.
4. **On**: the shadow table promoted to drafting while it wins over a window, and reset when it loses.

## Kernels

| Kernel | When | What |
| --- | --- | --- |
| `capture_kernel` | in the draft graph, each draft step | a block a draft row: its 5 head rows (8-byte words), drafts and anchor position into the request's ring at `count % capacity`, then `count + 1`; padding rows (slot −1) write nothing |
| `labels_kernel` | end of turn | a thread a kept entry: the anchor and the committed tokens at its 5 positions (−1 past the turn's end) |
| `reset_kernel` | request added, finished or preempted | the request's ring emptied |
| `stats_kernel` | end of turn | one thread folds a chunk's entries into the turn's counters, in order |

The replay reuses the declared draft kernels: `compute_draft_logits_local`, `markov_greedy_pairs_local`
(fused) or `markov_bias_local` + `greedy_markov_shard_pair` (by the draft step's own rule),
`tensor_model_parallel_all_gather`, `select_first_argmax` and `markov_anchor_embed`. It runs in chunks of
at most `max_num_reqs` kept steps, the draft step's own shapes.

Per-turn counters (`TurnStats`): `entries`, `labeled`, `forced_accepted` (all five positions),
`draft_accepted` (the drafts within the step's active drafts), `forced_accepted_active`,
`replay_mismatches`, `drafted`, `kept`. In serving, `replay_mismatches` must be 0 and `draft_accepted`
must equal `forced_accepted_active` on every turn; a turn that breaks either is logged as a warning.

## Kernel tests (sp8, candidate-60 image with the overlay)

[bundles/kernels](bundles/kernels): the four kernels against Python references (wrap-around and padding
rows in the capture, a strided token row and a turn that ends inside a block in the labels, both the
overwrite and the accumulate path of the counters).

```bash
python3 scripts/lab.py kernel-local experiments/2026-11-17-dspark-turn-learning/bundles/kernels --out <dir>
```

Result: 6 passed.

## Observe run

[observe-tp4-switched.json](observe-tp4-switched.json), on `tp4-switched` (sp1..sp4), profile
[tp4-switched-observe.json](tp4-switched-observe.json) (capacity 256 draft steps a request, dumps to
`/cache/turn-learning` on sp1):

1. one boot: the coding conversations and held-out prompts of the first design's workload, then the
   determinism and greedy gates, which must equal candidate-60's hashes (`18d61961` / `d6568649`, text
   `0a2dacdd` / `12fd8257`): observe changes no output;
2. a fresh boot: AgentX's agent-trace replay at 8 sessions for an hour, the turns the offline study reads.

The image is `spark-ds41f:candidate-60-afcaac35`: dgx4's `candidate-60` (tree `afcaac35`, the final
candidate-60 build the profiles name) under a new tag, copied to sp1 for `cluster sync` to distribute.
Spaceport's own `candidate-60` tag holds the earlier build (tree `b7c6c958`, before `fdf5751ee`'s SP-prefill
stream-lifetime fix) and is left as it is. The files the overlay replaces, and the draft-step modules the
learner calls, are identical in both trees (SHA-256 of `model_runner.py`, `dspark/speculator.py`,
`dspark/greedy.py`, `sample/tilelang_sampling.py`, `nvidia/dspark.py`, `tilelang_inputs.py`, `states.py`).
The kernel tests ran on sp8's `candidate-60` (tree `b7c6c958`).

This is a correctness and data-collection run, not a performance comparison: with dumps on, the end of a
turn also copies its rows to the host.

```bash
bin/spark candidate queue tp4-switched experiments/2026-11-17-dspark-turn-learning/observe-tp4-switched.json --apply
```


### Smoke boot (2026-11-17, queue entry `20261117T015246-observe-tp4-switched`)

Boot ready at 01:55Z on `candidate-60-afcaac35` with the overlay; the learner logged its ring (256 draft
steps a request, 200 MiB of rows a rank) and wrote every finished turn to `/cache/turn-learning` on sp1.
Turn lines up to the gates: [receipts/observe-smoke/turn-learning-lines.log](receipts/observe-smoke/turn-learning-lines.log).

| | Value |
| --- | --- |
| Turns replayed (workload, before the gates) | 114 |
| Entries the ring held / labeled positions | 6,878 / 34,045 |
| Replay mismatches in held entries | 0 |
| `draft_accepted == forced_accepted_active` | on every turn |
| End of turn | median 10.6 ms, max 17.3 ms; 0.124 ms a kept draft step (two chains, head re-projection, dump copy) |
| Gates (`bin/spark candidate gate`, candidate-60's hashes) | pass: log-probabilities `18d61961` / `d6568649`, text `0a2dacdd` / `12fd8257`, identical |

One turn (17) reported 5 replay mismatches with 0 labeled entries: the host had counted 3 draft steps
that wrote no ring row, and the counters compared those empty slots. The labels kernel now writes -2 for
an entry the ring does not hold (-1 stays "past the turn's end") and the counters skip such entries; the
kernel test covers it (6 passed on sp8 after the change). The running boot carries the earlier counters;
its dumps are unaffected (the study reads only labels >= 0).

### First offline study (smoke turns only)

[bundles/study](bundles/study) on sp8 over the 147 turns the smoke boot wrote (workload and gates),
loaded W1 and five step sizes of three update rules, each turn scored before it is learned
([receipts/study-smoke/output.jsonl](receipts/study-smoke/output.jsonl)):

- The single-GPU replay agrees with the serving drafts at 99.2% of drafted positions (different
  arithmetic, near ties).
- The committed token is among the draft step's block maxima at 72% of labeled positions.

Teacher-forced accepted tokens a block, learned against loaded:

| Turns | Blocks | Loaded | full, 0.1 | shard maxima + label, 0.1 | block maxima, 0.1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| coding conversations (1-24) | 32 | 1.19 | +42.1% | +39.5% | 0.0% |
| held-out prompts (25-40) | 1,070 | 3.94 | +0.3% | +0.3% | +0.3% |
| determinism and greedy gates (41+) | 9,115 | 1.65 | +34.8% | +32.5% | +16.2% |

This sample does not answer the agent-traffic question. The gates send the same prompts again and
again, so their gain is learning text that repeats; the 16 distinct held-out prompts leave little room
(3.9 of 5 accepted) and barely move; the coding conversations kept only 32 DSpark blocks in 24 turns
(most of their steps were copy-drafted). At step 0.1 the most-updated W1 rows moved 5.2 times their own
norm, so a drift cap is needed. The AgentX hour is the test.

### AgentX hour (same entry, fresh boot, 8 sessions, 3,600 s)

The run is valid (815 requests, 0 errors; TTFT p50 0.69 s / p90 1.93 s; 171.2 output tok/s; prefix cache
96.8% of 97.0% ideal; [receipts/observe-agentx](receipts/observe-agentx)). It is not a speed comparison:
the end of each turn also copied its rows to the host. 931 turns were written to disk.

Turn lines ([receipts/observe-agentx/turn-learning-lines.log](receipts/observe-agentx/turn-learning-lines.log)):
935 turns replayed, 847 with labels, 124,510 held entries, 619,495 labeled positions; end of turn median
16.7 ms (p90 32.2 ms, max 36.0 ms; median 131 kept draft steps). Unlike the smoke boot, 66 labeled turns
reported replay mismatches (980 of 627,528 drafted positions, 0.16%), and `draft_accepted` differed from
`forced_accepted_active` by 63 positions in total. The dumped entries of those turns show the cause: a few
entries a turn whose anchor lies hundreds to thousands of positions before their neighbours (turn 126:
−2,171; 123: −545; 127: −1,039), draft steps over another context than the turn's (one path that drafts
outside decoding is `boundary_checkpoint.replay_draft`, which proposes for a restored prefix). Their rows
do not belong to the committed tokens at their anchors. The capture now also keeps the anchor token the
step drafted from, and the labels kernel holds an entry only when that token is the one the turn committed
there (−2 otherwise, counted as `unheld`); the kernel tests cover it (6 passed on sp8). The study below ran
on these dumps as they are: the foreign entries are 0.16% of positions.

### Offline study on the AgentX turns

843 turns with labels, 125,268 blocks, 623,290 labeled positions, 1.72 accepted a block under the loaded
W1; single-GPU replay agreement 98.4%; the committed token is among the step's block maxima at 72.1% of
positions ([receipts/study-agentx/output.jsonl](receipts/study-agentx/output.jsonl)). Each turn is scored
before it is learned; gain in teacher-forced accepted tokens against the loaded W1:

| Rule | Step 0.01 | Step 0.03 | Step 0.1 |
| --- | ---: | ---: | ---: |
| full vocabulary | +1.38% [+1.08, +1.69] | −0.65% | −15.69% |
| block maxima (no label logit) | +0.77% [+0.55, +1.00] | −0.10% | −9.48% |
| shard maxima + label | −0.32% | −2.78% | −17.66% |
| live4 (to the first miss, live competitors + label) | −2.68% | −7.07% | −18.78% |

Brackets: 95% turn-level bootstrap intervals. By quarter of the run (about 210 turns each), the two
positive rules grow: full +0.89 / +1.16 / +1.42 / +1.83%, block maxima +0.46 / +0.58 / +0.62 / +1.21%;
live4 at 0.01 falls −0.10 / −1.90 / −3.74 / −4.46%. Large steps lose from the first quarter and keep losing
(the most-updated rows move 1.5 to 7 times their norm at 0.1).

Reading: on distinct agent sessions the shared W1 table (one row a previous token, used in every context)
learns only with small steps, slowly, and still improving after an hour; what the smoke boot showed was
repeated prompts. The rules that need the least serving work behave differently: block maxima (which the
draft step already computes) is positive, live4 is negative.

Follow-up runs on the same turns ([receipts/study-agentx-cap](receipts/study-agentx-cap),
[study-agentx-small](receipts/study-agentx-small), [study-agentx-adagrad](receipts/study-agentx-adagrad)):

- **Drift cap** (each row within its loaded norm): step 0.1 still loses (full −13.7%, block −9.1%); the
  small steps are unchanged (their rows stay inside the cap). The loss of large steps is the direction:
  a row serves every context after its token.
- **Smaller SGD steps and a per-row AdaGrad step** (frequent rows step less):

| Rule | Step | Gain [95% CI] | By quarter of the hour |
| --- | --- | ---: | --- |
| full vocabulary | AdaGrad 0.3 | +1.52% [+1.26, +1.81] | +1.14 / +1.27 / +1.74 / +1.80% |
| full vocabulary | SGD 0.005 | +1.39% [+1.11, +1.65] | +0.88 / +0.97 / +1.41 / +2.02% |
| full vocabulary | SGD 0.002 | +0.91% [+0.72, +1.10] | +0.42 / +0.78 / +0.95 / +1.30% |
| block maxima | AdaGrad 0.3 | +0.96% [+0.73, +1.22] | +0.69 / +0.53 / +1.03 / +1.46% |
| block maxima | AdaGrad 0.1 | +0.84% [+0.66, +1.02] | +0.36 / +0.73 / +0.80 / +1.31% |
| block maxima | SGD 0.01 | +0.77% [+0.56, +1.00] | +0.46 / +0.58 / +0.63 / +1.22% |
| block maxima | AdaGrad 1.0 | −0.40% [−0.77, −0.03] | −1.26 / −0.84 / −0.30 / +0.49% |

## Shadow

[turn_learning_shadow.py](overlay/vllm/v1/worker/gpu/spec_decode/dspark/turn_learning_shadow.py), mounted
beside observe's files ([tp4-switched-shadow.json](tp4-switched-shadow.json): capacity 512 draft steps a
request, AdaGrad step 0.3). The learner keeps an FP32 copy of Markov W1 and trains it on every finished
turn; the draft step keeps reading the loaded W1, so no draft and no output changes.

It keeps only what the update reads. At every draft position the greedy step already reduces each rank's
vocabulary shard into 2048-column blocks (or, for the first position at three requests or more, the
fused projection reduces it into 64-column tiles). The learner points those reductions at its own
per-position scratch (the shared scratch is overwritten from one position to the next), which costs
nothing. One kernel, forked onto a side stream after the last position's partials and joined at the end
of the draft step, copies each request's 16 block maxima a position (value and global token), its anchor
and its drafts into the request's ring while the last position's pairs are exchanged. No head row is
kept and no logit is recomputed: 640 bytes a draft step a rank, against observe's 51 KB.

At the end of a turn (every rank, the same step, sorted order, before the slot is reused):

| Kernel | What |
| --- | --- |
| `labels_kernel` | observe's: the committed token at each kept position; entries drafted over another context excluded |
| `vocab_embedding_packed` (candidate-60's) | the candidates' W2 rows, decoded from this rank's packed shard |
| `shadow_local_kernel` | a block an entry: each candidate's teacher-forced logit under the loaded table and under the copy (the step's value with the step input's bias replaced by the committed input's, biases and sum rounded to BF16 as the step rounds them), and a 10-value summary a position |
| all-gather | the summaries: 10 floats a position a rank |
| `shadow_combine_kernel` | a block an entry, alike on every rank: the step's drafts, checked against the ring's; each table's teacher-forced run over the maxima; at the positions up to each block's first miss whose committed token is a candidate, this rank's share of the cross-entropy gradient into the input's W1 row |
| all-reduce | the gradient: 256 floats a position |
| `shadow_update_kernel` | a block a position: each W1 row's summed gradient (position order), its AdaGrad accumulator and step on the copy |
| `shadow_totals_kernel` | the call's and the boot's counters |

Rank 0 logs a line a finish (`ShadowStats`: `held`, `seen`, `trained`, `accepted`, `tf_loaded`,
`tf_learned`, `fixed`, `broken`, `checked`, `draft_mismatches`, the loss and the end-of-turn time);
every 256 finishes the ranks compare a sum of their copies. `draft_mismatches` must stay 0: the captured
maxima must give the step's drafts. Memory a rank: the copy (126 MiB) and its accumulators (0.5 MiB), the
rings (640 bytes × 512 draft steps a request), 10 MiB of decoded W2 rows for 256 entries.

### The rule and the score serving can compute (offline, the AgentX turns)

Before building it, the serving rule (`blockseen`: the block-maxima rule on the positions up to each
block's first miss, where the step's input was the committed token, so its maxima are teacher forcing's)
and two scores computable from the captured maxima alone were run through the offline study
([bundles/study](bundles/study), AdaGrad, 843 turns):

| Learner | Exact (FP32 sums) [95% CI] | Exact (BF16 sums, as serving) | By quarter (exact) | Seen-window score | Maxima teacher-forced score |
| --- | ---: | ---: | --- | ---: | ---: |
| `blockseen` 0.1 | +0.70% [+0.54, +0.86] | +0.76% | +0.24 / +0.66 / +0.75 / +1.01% | −0.33% | +0.23% |
| `blockseen` 0.3 | +0.94% [+0.73, +1.17] | +0.92% | +0.37 / +0.73 / +1.01 / +1.45% | −0.88% | +0.03% |
| full 0.1 | +1.16% [+0.97, +1.35] | +1.21% | +0.83 / +0.93 / +1.15 / +1.59% | −0.45% | +0.16% |
| full 0.3 | +1.52% [+1.27, +1.79] | +1.56% | +1.14 / +1.27 / +1.74 / +1.84% | −1.04% | −0.02% |

Receipts: [study-agentx-seen](receipts/study-agentx-seen) (also `block` and step 1.0, and the seen-window
score in its first form, the learned change added in FP32 to the BF16 sums: −0.36 / −0.88 / −2.51% for
`blockseen`), [study-agentx-seen-rounded](receipts/study-agentx-seen-rounded),
[study-agentx-tfproxy](receipts/study-agentx-tfproxy).

- **The rule holds.** Training only where the step's maxima are teacher forcing's loses nothing against
  the block rule on every position (+0.94% against +0.96%), and the gain survives serving's BF16 rounding
  (+0.92%).
- **The scores do not.** The seen-window score (the copy's run over the positions up to the first miss)
  is negative for every learner; the teacher-forced score over every drafted position, both tables by one
  rule, reads about zero where the exact gain is +0.9 to +1.6%, and ranks the steps the wrong way. Where
  the draft missed, the committed token is often in the draft's own 2048-column block, so it is not a
  candidate: the maxima show what the copy breaks but miss most of what it fixes. Serving cannot measure
  the gain from what the update needs; shadow logs the maxima-limited score only as a health signal.

### Capture cost (graph microbenchmark, sp8)

[bundles/capture-bench-shadow](bundles/capture-bench-shadow): CUDA graphs of a tail stand-in for the last
position's exchange, with the capture forked onto a side stream, inline, or absent; median of 5
interleaved rounds after a clock warmup ([receipts/capture-bench-shadow-3](receipts/capture-bench-shadow-3)).

| Requests | Capture alone (graph) | Inline adds | Forked adds, spin tail (SMs free) | Forked adds, copy tail (all SMs busy) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4.10 µs | +2.06 µs | +0.04 µs | +1.23 µs |
| 4 | 4.24 µs | +2.05 µs | +0.07 µs | +1.78 µs |
| 16 | 4.10 µs | +2.15 µs | +0.22 µs | +1.38 µs |

Forked over a tail that leaves the SMs free, as a collective waiting on the network does, the capture
costs a few hundredths of a microsecond; over a tail that fills every SM it costs about as much as
inline. The first version reduced the fused position's 32 tiles a part serially in one thread (capture
alone 5.8 to 6.6 µs, [capture-bench-shadow-1](receipts/capture-bench-shadow-1), copy tail only);
[capture-bench-shadow-2](receipts/capture-bench-shadow-2) is the warp-parallel version without warmup.
The real tail is the network all-gather; the screen below measures the step itself.

### Kernel tests

[bundles/kernels](bundles/kernels) now also runs
[test_dspark_turn_learning_shadow.py](bundles/kernels/test_dspark_turn_learning_shadow.py): the capture
(parts, the fused position's tiles, undrafted positions, padding rows, wrap-around), the end of turn across
4 and 3 simulated ranks against a Python reference (dyadic data, so the rounded logits match to the bit:
counters, gradient summed over ranks, AdaGrad step), the update's row deduplication. 11 passed
([receipts/kernels-6](receipts/kernels-6); the module's log text and docstring changed afterwards).

### Shadow screen (`tp4-switched`)

[shadow-tp4-switched.json](shadow-tp4-switched.json):

1. the screen suite, two boots, against candidate-60's recorded screen
   (`experiments/2026-11-16-c60-prefill/receipts/final/tp4-switched/suites/c60-screen-tp4-switched/c60`):
   the engine grid's step times, c16 throughput and prefill, the model check between the two boots;
2. one boot: the determinism and greedy gates, which must equal candidate-60's hashes;
3. a fresh boot: AgentX at 8 sessions for three hours (the learner's end-of-turn time, the replica
   check, the loss over time).

```bash
bin/spark candidate queue tp4-switched experiments/2026-11-17-dspark-turn-learning/shadow-tp4-switched.json --apply
```

Queue entry `20261117T095559-shadow-tp4-switched`, commit `33bf74665`. Receipts:
[shadow-screen](receipts/shadow-screen) (both boots' suite files and the verdicts), [shadow-gate](receipts/shadow-gate).

**Screen against candidate-60's recorded screen** (both boots of each; the lab's verdicts):

| Metric | Change | ±95% | Margin | Verdict |
| --- | ---: | ---: | ---: | --- |
| grid: decode geometric mean (20 shapes) | +0.40% | 0.14% | 0.27% | worse |
| c16: decode tok/s, paired | −0.08% | 0.53% | 0.58% | same |
| c16: output tok/s | −0.07% | 0.74% | 0.96% | same |
| prefill 32,768 tok/s | −0.18% | 1.20% | 1.56% | same |
| prefill 262,144 tok/s | −0.32% | 0.57% | 0.74% | same |

c16's drafts are candidate-60's: 1.370 accepted a draft against 1.371, the same tokens a step and survival at
every position. The grid's step split, each shadow boot against the mean of candidate-60's two (median over
the 20 shapes):

| Boot | Step | Target forward | Draft step |
| --- | ---: | ---: | ---: |
| shadow | +65.3 µs (+0.22%) | +29.3 µs (+0.18%) | +13.2 µs (+0.52%) |
| shadow-again | +61.4 µs (+0.29%) | +42.5 µs (+0.28%) | +9.0 µs (+0.35%) |
| candidate-60's own repeat | | −14.5 µs | +0.1 µs |

Most of the step's difference is in the target forward, which shadow does not touch (its kernels and
buffers are candidate-60's; the learner adds 144 MiB a rank at load and nothing to the forward), and the
draft step's +9 to +13 µs is about the microbenchmark's inline capture several times over, not its forked
cost. The reference ran on 2026-11-16; this screen on 2026-11-17. A same-session control separates the two:
[shadow-control-tp4-switched.json](shadow-control-tp4-switched.json), four boots alternating the learner off
([tp4-switched-off.json](tp4-switched-off.json): the same files mounted, nothing allocated or launched,
candidate-60's draft path) and shadow, queued once the entry below finishes.

**Gates** (one boot): determinism and greedy identical, candidate-60's hashes (log probabilities `18d61961` /
`d6568649`, text `0a2dacdd` / `12fd8257`): shadow changes no output.

**End of turn** (first screen boot, 159 finishes): median 0.277 ms, p90 0.35 ms for a finish call, with
its all-gather and all-reduce; the first call of a boot, during startup warmup, took 633 ms. 63,048 drafts
checked against the captured maxima, 0 mismatches. The maxima-limited score read −0.50%, as the offline
check predicts of a score that cannot see most fixes. Warmup requests at startup reach the learner too
(16 turns, 23 trained positions a boot); drafting with a learned table would need them left out.

**AgentX, three hours at 8 sessions:** running.

## Where it stands

After an hour of AgentX traffic, end-of-turn W1 learning raises teacher-forced accepted tokens by about
1.5% with the full-vocabulary rule and about 0.9% with the rule serving can run from the block maxima it
already computes (exact, and under serving's BF16 rounding), both still rising at the end of the hour.
Teacher-forced acceptance is an upper bound of what verification realizes; no serving speed has been
measured with learned drafts.

Shadow mode is built and tested: a capture that costs a few hundredths of a microsecond a step when
forked over a tail that leaves the SMs free, and an end-of-turn score and update in TileLang. What it
cannot do is measure the gain: the block maxima show what a learned table breaks and miss most of what it
fixes. The gain needs exact replays (observe and the offline study) or drafts read from the learned table
(on). Learning resets at every restart unless the learned rows are saved.
