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
cost. The reference ran on 2026-11-16; this screen on 2026-11-17. A same-session control (four boots
alternating the learner off and on) was planned for this version; it never ran, replaced before the queue
took it by the same control of the off-step version below, whose learner-off boots also measure the day's
drift against this reference. This version's files are `overlay/` at commit `33bf74665` (the queue entry's).

**Gates** (one boot): determinism and greedy identical, candidate-60's hashes (log probabilities `18d61961` /
`d6568649`, text `0a2dacdd` / `12fd8257`): shadow changes no output.

**End of turn** (first screen boot, 159 finishes): median 0.277 ms, p90 0.35 ms for a finish call, with
its all-gather and all-reduce; the first call of a boot, during startup warmup, took 633 ms. 63,048 drafts
checked against the captured maxima, 0 mismatches. The maxima-limited score read −0.50%, as the offline
check predicts of a score that cannot see most fixes. Warmup requests at startup reach the learner too
(16 turns, 23 trained positions a boot); drafting with a learned table would need them left out.

**AgentX, three hours at 8 sessions** (a fresh boot, profiling 10:38–13:39 UTC;
[receipts/shadow-agentx](receipts/shadow-agentx)): valid, 1,822 requests, 0 errors, TTFT p50/p90
0.70/1.81 s, 26.6/43.7 tok/s a user, 174.7 output tok/s, prefix hits 96.7% of 97.5% ideal, 1.965 accepted
a draft. It is not a speed comparison: no learner-off AgentX run of this shape is recorded (observe's hour
read 0.69/1.93 s, 25.2/44.6 tok/s a user, 171.2 tok/s).

The learner over the three hours ([turn-learning-lines.log](receipts/shadow-agentx/turn-learning-lines.log)):
1,875 finish calls, 958,471 trained positions; the copies of W1 agreed across the four ranks at all seven
checks; 0 draft mismatches in 1,726,496 checked. The maxima-limited score by hour: −0.19%, +0.22%, +0.52%,
+0.57% (a health signal that cannot see most fixes; it climbs as the copy learns). End of turn: 1.94 s in
all over 10,822 s, median 0.48 ms, p90 2.86 ms, max 5.87 ms; below.

### End of turn under agent traffic, and where its time goes

The AgentX boot (1,875 finish calls, 1,839 of them one turn): the end of turn takes far longer than in the
screen, whose turns are short. Rank 0's events around the call:

| Calls | n | Median | p90 | Max |
| --- | ---: | ---: | ---: | ---: |
| all | 1,875 | 0.48 ms | 2.86 ms | 5.87 ms |
| one turn, 32–128 held steps | 679 | 0.22 ms | 1.97 ms | 4.62 ms |
| one turn, 128–256 | 386 | 0.41 ms | 2.47 ms | 4.36 ms |
| one turn, 256–384 | 179 | 0.71 ms | 2.85 ms | 3.47 ms |
| one turn, 384–512 | 337 | 1.04 ms | 3.34 ms | 4.55 ms |
| one turn, 512 or more | 159 | 1.13 ms | 3.35 ms | 3.99 ms |
| observe's AgentX hour, for scale | 935 | 16.5 ms | 32.1 ms | 43.9 ms |

[bundles/finish-bench-shadow](bundles/finish-bench-shadow) runs the learner's own `finish` on sp8 at serving
shapes (rank 0 of four, a packed W2 shard built as at load, 16 slots, rings filled like finished turns), with
one-GPU stand-ins for the collectives: their network time is not in it. `gpu` is the GPU's time alone (a
spin queued ahead keeps the launches ahead of the GPU), `idle` what rank 0 logs with the GPU idle at the
call, `host` the host's time in `finish`; medians of 15
([receipts/finish-bench-shadow-2](receipts/finish-bench-shadow-2); run 1,
[finish-bench-shadow-1](receipts/finish-bench-shadow-1), within 0.03 ms of it on `idle`):

| Turns × kept steps | Chunks | gpu | idle | host |
| --- | ---: | ---: | ---: | ---: |
| 1 × 64 | 1 | 0.08 ms | 0.14 ms | 0.16 ms |
| 1 × 128 | 1 | 0.14 ms | 0.19 ms | 0.17 ms |
| 1 × 256 | 1 | 0.29 ms | 0.34 ms | 0.21 ms |
| 1 × 384 | 2 | 0.42 ms | 0.48 ms | 0.30 ms |
| 1 × 512 | 2 | 0.55 ms | 0.62 ms | 0.34 ms |
| 2 × 512 | 4 | 1.15 ms | 1.23 ms | 0.59 ms |
| 4 × 512 | 8 | 2.32 ms | 2.42 ms | 1.07 ms |
| 8 × 512 | 16 | 4.58 ms | 4.83 ms | 2.15 ms |

GPU time a launch, one turn of 512 steps (two chunks; events around each launch):

| Launch | Launches | GPU | Host |
| --- | ---: | ---: | ---: |
| `shadow_update_kernel` | 2 | 0.249 ms | 0.027 ms |
| `shadow_local_kernel` | 2 | 0.127 ms | 0.033 ms |
| `vocab_embedding_packed` (decode) | 2 | 0.111 ms | 0.033 ms |
| `shadow_totals_kernel` | 2 | 0.031 ms | 0.026 ms |
| `shadow_combine_kernel` | 2 | 0.025 ms | 0.033 ms |
| host-to-device copies | 6 | 0.023 ms | 0.137 ms |
| `labels_kernel`, `reset_kernel` | 2 | 0.016 ms | 0.031 ms |
| collective stand-ins | 4 | 0.008 ms | 0.062 ms |

What this says:

- The work is GPU-bound and linear in kept steps, about 0.28 ms a 256-step chunk; the host stays ahead.
- The serving median at 384–512 steps (1.04 ms) is the bench's 0.62 ms plus about 0.4 ms, the two chunks'
  network all-gather and all-reduce. The tail to 3–4 ms at every turn size is not work (the bench's range is
  within 0.15 ms): it is waiting, the call's first collective absorbing whatever skew the ranks arrive with.
- `shadow_update_kernel` is 45% of the GPU time and needlessly so: each owner block's threads each loop
  serially over every later position for its row (positions² work). A block-wide search for the few
  matching positions would take it to about 0.01 ms a chunk. Decode and the local kernel are then most of
  it, near memory bandwidth (about 13 MB a chunk).
- Even at that cost, a finished long turn holds the whole batch for a few hundred microseconds plus two
  collectives, and the collectives bring the ranks' skew with them.

### Off the step

[overlay-offstep/](overlay-offstep/vllm/v1/worker/gpu/spec_decode/dspark) (`turn_learning_shadow.py`,
`speculator.py`; the profile [tp4-switched-offstep.json](tp4-switched-offstep.json)). One causal change: where
the end of turn runs. Two kernel fixes ride with it, as the owner decided.

- **On the step's stream, only the staging.** `shadow_stage_kernel` (a block an entry) writes each kept entry's
  labels and inputs (`labels_kernel`'s rule) and a copy of its ring entry into the learner's stage, entry-major
  in call order; then the finished and preempted rings are cleared. One host-to-device copy a call carries the
  slots and each entry's turn and step. The stage has room for two calls at full rings (16,384 entries,
  11.4 MiB); a call's rows come round again only after its end of turn is done, and if they are still being
  read the step waits for it and counts a stage wait.
- **On the learner's stream, the rest.** The decode, the local kernel, the all-gather, the combine, the
  all-reduce, the update and the totals read only the stage, on a CUDA stream of the learner's own that waits
  for the staging. The collectives go through an NCCL communicator of the learner's own over the
  tensor-parallel group (vLLM's `PyNcclCommunicator`, built at load), never the step's, whose collectives run
  meanwhile; every rank issues the learner's in the same order. The all-gather's layout is rank-major, so
  the combine reads it as it lands. The replica check is asynchronous too: the copies' sums are exchanged on
  the learner's stream and compared on rank 0 when they arrive.
- **`shadow_update_kernel`:** a warp ballot a 32 positions finds the row's training positions (one word a
  warp), each thread adds its column of those rows only, in position order as before; the squared norm is
  a shuffle tree a warp. **`shadow_totals_kernel`:** a warp a counter instead of a thread, and it replaces
  the call's totals on a call's first chunk (no separate zeroing).
- Rank 0 logs both times a call: the step's (`on the step`) and the learner stream's (`off it`).

Kernel tests ([receipts/kernels-7](receipts/kernels-7), sp8, candidate-60 image): 14 pass. New or changed:
the stage against `labels_kernel` and the ring (held, not held, past the turn, a wrapped ring, more kept
steps than the ring holds, rows outside the call untouched); the end of turn on stage rows at an offset
with the rank-major exchange, against the reference across four and three simulated ranks; the totals'
replace and add; the update at 40, 700 and 1,280 positions (several ballot tiles, rows repeating up to
about 20 times).

**Bench** ([bundles/finish-bench-offstep](bundles/finish-bench-offstep), sp8; the same shapes, rings and
collective stand-ins as the on-step bench above). For each call: the step's stream (the staging) and the
learner's stream, each as GPU time alone and as rank 0 logs it with the GPU idle at the call; the host's
time in `finish`; and the overlap, a memory-bound stand-in for the next step's forward (64 MiB copies, about
19.7 ms) queued on the step's stream right after `finish`, timed with the learner's work beside it and
without, alternating. Medians of 15, run 2
([receipts/finish-bench-offstep-2](receipts/finish-bench-offstep-2)). Run 1
([finish-bench-offstep-1](receipts/finish-bench-offstep-1)) agrees within 0.01 ms on both streams and
within 35 µs on the overlap at every size but its first: there the stand-in itself took 33.5 ms instead
of 19.5, so the GPU was shared or clocked down for that case, and it is set aside (it is kept in the
receipt).

| Turns × kept steps | Step's stream, gpu / idle | Learner's stream, gpu / idle | Host | Stand-in forward slowed |
| --- | ---: | ---: | ---: | ---: |
| 1 × 64 | 0.016 / 0.040 ms | 0.047 / 0.075 ms | 0.14 ms | +2 µs (−154 to +144) |
| 1 × 128 | 0.018 / 0.038 ms | 0.076 / 0.114 ms | 0.14 ms | −13 µs (−89 to +88) |
| 1 × 256 | 0.018 / 0.040 ms | 0.146 / 0.181 ms | 0.15 ms | +55 µs (0 to +200) |
| 1 × 384 | 0.018 / 0.041 ms | 0.217 / 0.273 ms | 0.21 ms | +78 µs (−421 to +536) |
| 1 × 512 | 0.022 / 0.043 ms | 0.297 / 0.332 ms | 0.21 ms | +164 µs (−175 to +258) |
| 2 × 512 | 0.024 / 0.046 ms | 0.595 / 0.630 ms | 0.33 ms | +429 µs (+396 to +537) |
| 4 × 512 | 0.031 / 0.054 ms | 1.19 / 1.23 ms | 0.56 ms | +933 µs (+854 to +1,456) |
| 8 × 512 | 0.043 / 0.081 ms | 2.44 / 2.46 ms | 1.03 ms | +1,337 µs (+1,281 to +1,488) |

GPU time a launch, one turn of 512 steps (two chunks): staging 0.012 ms and reset 0.003 ms on the step's
stream; on the learner's, local 0.125, decode 0.106, update 0.033 (was 0.249), combine 0.027, totals 0.016
(was 0.031), stand-ins 0.008 ms.

- The step now waits 0.02 to 0.08 ms a call, against 0.14 to 4.8 ms before; no stage waits at any size.
- The learner's work is 0.30 ms a 512-step turn, against 0.55 ms: the ballot search took the update from
  0.249 to 0.033 ms. Decode and the local kernel are now 78% of it.
- The stream does not make that work free. Beside a forward that is nothing but memory traffic, about half
  to three quarters of the learner's GPU time comes back as forward time (+0.16 ms for one 512-step turn,
  +1.3 ms for eight): both are bandwidth-bound, and the learner moves about 13 MB a chunk. A decode step
  also waits on the network and on short kernels, which should hide more of it; the serving run measures
  how much. The lever left is the decode: decoding the candidates' W2 rows inside the local and combine
  kernels instead of writing 10 MiB a chunk and reading it back twice.
- The host's time in `finish` (0.21 ms for a 512-step turn) stays on the host's path; with asynchronous
  scheduling it overlaps the GPU's previous step.

Serving run: [offstep-tp4-switched.json](offstep-tp4-switched.json):

1. the screen suite, four boots alternating the learner off
   ([tp4-switched-offstep-off.json](tp4-switched-offstep-off.json): the same files mounted, nothing
   allocated, launched or connected, candidate-60's draft path) and off-step shadow, in one measure job (the
   lab refuses a single boot an arm against a recorded reference; the off boots are the session's own
   reference, and against candidate-60's recorded screen they measure the day's drift);
2. one boot: the determinism and greedy gates, which must equal candidate-60's hashes;
3. a fresh boot: AgentX at 8 sessions for an hour (the step's and the learner's times, the replica check).

Queue entry `20261117T140325-offstep-tp4-switched`, commit `4ec6cdaef`. Receipts:
[offstep-screen](receipts/offstep-screen) (the four boots' suite files, startup memory and telemetry, the lab's
verdicts, [grid-split.txt](receipts/offstep-screen/grid-split.txt) from
[grid_split.py](receipts/offstep-screen/grid_split.py), the learner's log lines),
[offstep-gate](receipts/offstep-gate), [offstep-agentx](receipts/offstep-agentx).

**Gates** (one boot): determinism and greedy identical, candidate-60's hashes (log probabilities `18d61961` /
`d6568649`, text `0a2dacdd` / `12fd8257`). The screen's model step: greedy responses identical (128 of 128)
and identical distributions between every pair of the four boots.

**Screen, the learner off against off-step in one session** (two boots each, alternating; each value the
boot's own):

| Metric | Off | Off-step | Change |
| --- | ---: | ---: | ---: |
| grid: whole step, geometric mean (20 shapes) | 19.152 / 19.148 ms | 19.338 / 19.338 ms | +0.98% |
| grid: target forward, geometric mean | 14.848 / 14.840 ms | 15.025 / 15.026 ms | +1.22% |
| grid: draft step, geometric mean | 2.816 / 2.822 ms | 2.815 / 2.819 ms | −0.07% |
| c16: output tok/s | 269.18 / 268.97 | 269.95 / 268.59 | +0.07% |
| c16: engine steps/s | 7.880 / 7.863 | 7.897 / 7.877 | +0.20% |
| c16: accepted a draft | 1.371 / 1.371 | 1.376 / 1.370 | |
| prefill 32,768 and 262,144 tok/s | | | same (the lab's verdicts, ±0.6%) |

- **The grid is slower with the learner on, by 0.98%, and it is not noise.** Each arm's own repeat is
  −0.02% (off) and −0.00% (off-step). The lab's verdict line calls the grid "same" only because its repeat
  term, off-1 against offstep-2, is itself an off-against-on pair (+0.97%); its first pair's table
  (offstep-1 to off-1) calls off faster at 11 of 20 shapes and slower at none.
- **It is the target forward, not the draft step.** Median over the shapes: whole step +152 µs, target
  forward +139 µs, draft step −7.5 µs. It grows with the work: +46 µs at 1×2 rows @4K, +0.33 ms at 8×48,
  +0.75 ms at 12×72, +1.10 ms at 16×96, +0.58 ms at 16×48 @64K, +0.27 ms at 1×3 @256K.
- **The learner is idle during the grid.** The grid times the server's own dummy steps, in which no
  request finishes: in both off-step boots no learner call fell inside the grid's 82 s (the last a second
  or three before it, the next as it ended). So the cost comes
  from something the off-step version holds while idle that the on-step version did not. On-step shadow's
  forward read +29 and +42 µs against candidate-60's recorded screen; today's off boots read +35 µs
  against it (the day's drift, +0.21% on the step), today's off-step boots +161 µs (+1.19%). What is new
  and present while idle: the learner's NCCL communicator over the tensor-parallel group (made and
  connected at load, over the same RoCE NIC as sparknet's step collectives), its CUDA stream and the
  11.4 MiB stage. GPU clocks, temperatures and the busy threads (sparknet's proxy, the engine core, the
  worker) read the same in all four boots.
- **c16 shows no cost, but it cannot rule out the grid's.** Output tok/s +0.07%; the lab's paired decode
  tok/s: off against off-step −0.20%, margin 0.42%. Its steps take about 127 ms (7.9 steps/s), against
  23–27 ms for its decode rows on the grid (16×32 to 16×48): most of a c16 step is prefill, so a 1% slower
  decode forward is diluted to a few tenths of a percent there. (An earlier version of this note read c16
  as showing that served steps escape the grid's cost; it does not.)
- The learner in the screen boots: 517 and 527 calls; the step's stream median 0.018 ms a call (p90
  0.022 ms); the learner's 0.36 ms (p90 0.46 ms); no stage waits; copies agreed at all four checks; 0 draft
  mismatches in 497,332. The first call of the first boot took 126 ms on the step and 505 ms off it during
  startup warmup (the kernels' first load); the second boot's first call, 0.054 ms.

**AgentX, an hour at 8 sessions** (a fresh boot, profiling 15:10–16:15 UTC): valid, 814 requests, 0 errors,
TTFT p50/p90 0.66/1.86 s, 25.1/42.1 tok/s a user, 169.5 output tok/s, prefix hits 96.8% of 97.1% ideal, 1.80
accepted a draft. Not a speed comparison: no learner-off AgentX hour ran in this session.

The learner over the hour ([turn-learning-lines.log](receipts/offstep-agentx/turn-learning-lines.log)): 871
calls, 934 turns, 359,043 trained positions; the copies agreed at all three checks; 0 draft mismatches in
675,656 checked; no stage waits.

| End of turn, a call | Median | p90 | p99 | Max | In all |
| --- | ---: | ---: | ---: | ---: | ---: |
| step's stream (off-step, this hour) | 0.019 ms | 0.20 ms | 0.34 ms | 0.36 ms | 42 ms |
| learner's stream (off-step, this hour) | 0.53 ms | 3.3 ms | 5.9 ms | 7.5 ms | 1.23 s |
| on the step (on-step shadow, its three hours) | 0.48 ms | 2.86 ms | | 5.87 ms | 1.94 s |

The learner's stream time includes waiting at its collectives for the other ranks. The step's p90 of
0.20 ms is ten times the screen boots' and the bench's, and its cause is not known. (Not the reset: the
learner's stream already waits for an event recorded after the reset kernel.) The step's interval opens
before `finish` builds the call's host-to-device copy and launches the staging, so where the GPU is idle at
the call it includes that host work, and where other streams' work holds the SMs the staging waits for it;
AgentX's long turns make both larger.

**Attribution** ([comm-attribution-tp4-switched.json](comm-attribution-tp4-switched.json), the suite
[model-grid-suite.json](model-grid-suite.json): the screen's model and grid steps alone): six boots
alternating the learner off, the learner off with its communicator alone
([tp4-switched-offstep-comm.json](tp4-switched-offstep-comm.json): [overlay-comm/](overlay-comm) makes the
communicator as shadow does and runs a chunk's all-gather and all-reduce 16 times at load, then leaves it
idle; nothing else of the learner is allocated) and off-step. If the communicator alone carries the +1%,
the fix is in how the learner exchanges; if not, it is in what else the off-step version holds.

Queue entry `20261117T163029-comm-attribution-tp4-switched`, commit `704e87952`; receipts
[comm-attribution](receipts/comm-attribution) (the six boots' grids, startup memory and telemetry, the lab's
verdicts, [grid-split.txt](receipts/comm-attribution/grid-split.txt) from
[grid_split.py](receipts/comm-attribution/grid_split.py)). Both communicator boots logged the communicator
made and exercised ([comm-1-lines.log](receipts/comm-attribution/comm-1-lines.log)).

| Grid, geometric mean (20 shapes) | Whole step | Target forward | Median a shape: step / forward / draft step |
| --- | ---: | ---: | ---: |
| off → communicator alone | +0.91% | +1.02% | +122 / +95 / +9 µs |
| off → off-step | +1.04% | +1.26% | +149 / +130 / +7 µs |
| communicator alone → off-step | +0.13% | +0.24% | +32 / +41 / −2 µs |
| each arm's repeat (off, communicator, off-step) | +0.14%, −0.01%, +0.15% | +0.22%, +0.12%, +0.21% | |

**The learner's own NCCL communicator carries the grid's cost.** Made, connected and exercised at load,
then idle, it alone slows the grid by 0.91%, shape by shape as the whole off-step learner does (16×96 rows:
+1.13 ms alone, +1.11 ms off-step; 16×48: +0.55 and +0.49 ms; 4×12 and 2×12: −0.05 to −0.07 ms both); the
rest of the learner adds 0.13%, inside the repeats. The communicator holds about 0.2 GiB a node (the
learner another 0.17 GiB); clocks, temperatures and busy threads match the off boots. How an idle
communicator slows the grid's dummy forward, and why c16's served steps do not show it, is not known; the
dummy runs take the same collective policy as served steps (sparknet for eligible all-reduce and all-gather,
pynccl for the rest).

### What slows the grid: the learner's communicator and the learner

[overlay-exchange/speculator.py](overlay-exchange/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py):
`VLLM_DSPARK_TURN_LEARNING_EXCHANGE` picks the communicator shadow's two exchanges a chunk go through;
nothing else changes (`turn_learning_shadow.py` is overlay-offstep's).

- `own`: the recorded design, a communicator of the learner's own on NCCL's default network (RoCE).
- `step` ([tp4-switched-exchange-step.json](tp4-switched-exchange-step.json)): the tensor-parallel group's
  existing pynccl communicator, which carries the step's collectives that sparknet does not (all-reduces over
  2 MiB, all-gathers over 4 MiB, the rest); nothing is made. NCCL runs one communicator's operations in
  issue order whatever their stream, and every rank issues the step's and the learner's in the same order,
  so a step operation on it can wait for a learner exchange in flight.
- `socket` ([tp4-switched-exchange-socket.json](tp4-switched-exchange-socket.json)): a communicator of the
  learner's own made by `ncclCommInitRankConfig` with NCCL 2.30's `ncclConfig_t` and the network set to
  Socket: TCP, no RDMA registrations or queue pairs. `NCCL_NET=IB` in the environment overrides a
  communicator's `netName`, so it is cleared for this init only and restored; NCCL's Socket network starts at
  this init (the step's communicators run IB only), so its interface is set for this init too, to
  `enp1s0f0np0` (the 200 GbE port's netdev; the bootstrap's `enP7s7` is 1 GbE).

One-rank checks on sp8, candidate-60 image ([receipts/socket-check](receipts/socket-check)): with
`NCCL_NET=IB` set, the Socket config made a communicator on "network Socket" and a default one after it
still took IB; made after an IB communicator, as in serving, the Socket network bound `enp1s0f0np0`
(10.30.0.16), and both variables read their profile values afterwards; the subclass's path (vLLM's
unique id, warm-up and collectives) all-reduced. The `step` exchange needs ranks; the job's model step is
its first run.

[exchange-tp4-switched.json](exchange-tp4-switched.json): the model and grid steps, six boots alternating
the learner off, `step` and `socket`. A fix passes here if the grid reads the learner-off boots' speed; c16
and AgentX then measure what its exchanges cost the step.

Queue entry `20261117T173718-exchange-tp4-switched`, commit `29490fd83`; receipts
[exchange](receipts/exchange) (the six boots' grids, startup memory and telemetry, the lab's verdicts,
[grid-split.txt](receipts/exchange/grid-split.txt), each learner boot's log lines). Every learner boot
logged its exchange and ran its model step's 155 to 163 learner calls with no error.

| Grid against the learner off (geometric mean, 20 shapes) | Whole step | Target forward | Boot to boot |
| --- | ---: | ---: | ---: |
| `step` (no communicator made) | +0.58% (+0.79, +0.40) | +0.68% | −0.39% |
| `socket` (its own, TCP) | +1.28% | +1.54% | +0.07% |
| learner off, repeat | +0.03% | +0.05% | |

**Neither removes the cost.** The Socket communicator, with no RDMA queue pairs or registrations, costs as
much as the RoCE one (+1.28% against +0.91% alone and +1.04% with the learner in the attribution job), so
it is not the network path. With no communicator made, `step` still costs +0.58%, its boots apart by 0.39%.
Socket is also slow for the learner: its end of turn took 5.3 ms a call at the median (8.6 ms p90) on its
stream, against 0.35 ms for `step`.

What the four costs follow, roughly, is the memory each version holds from load (the boots' available memory
after startup, against the learner off): `step` about 0.15 GiB (+0.58%), the communicator alone about 0.2 GiB
(+0.91%), `socket` about 0.3 GiB (+1.28%), off-step about 0.35 GiB (+1.04%); on-step shadow, about 0.15 GiB,
read about +0.2% beyond the day's drift. The nodes start without the memory-saver module the doctor expects
(every boot warns of it), so memory taken at load could change where later allocations (the KV cache, the
graphs' pool) land. Untested: the learner off with the same amounts merely allocated at load would tell.

**Memory alone** ([reserve-tp4-switched.json](reserve-tp4-switched.json)): six boots of the model and grid steps
alternating the learner off and the learner off with 150 or 350 MiB merely held from load
([overlay-reserve/](overlay-reserve) allocates and zeroes it where the learner and its communicator allocate
theirs; [tp4-switched-reserve-150.json](tp4-switched-reserve-150.json),
[tp4-switched-reserve-350.json](tp4-switched-reserve-350.json)). If the grid slows with the amount, the cost
is memory held from load, whatever holds it; if not, it is in what the learner and a communicator do.

Queue entry `20261117T183957-reserve-tp4-switched`, commit `3eca953c9`; receipts [reserve](receipts/reserve)
(grids, startup memory, telemetry, verdicts, [grid-split.txt](receipts/reserve/grid-split.txt), each reserve
boot's log line). All four reserve boots logged their amount.

| Grid against the learner off (geometric mean, 20 shapes) | Whole step | Target forward |
| --- | ---: | ---: |
| 150 MiB held from load | −0.17% | −0.15% |
| 350 MiB held from load | +0.01% | +0.07% |
| learner off, repeat | +0.23% | +0.21% |

**Memory held from load costs nothing.** The costs above follow NCCL work done for the learner: the
communicator alone (made, and exercised 16 times on a side stream at load), the learner's exchanges over a
Socket communicator, and the learner's exchanges over the step's own communicator from the learner's stream.
The grid's forward time is GPU time between CUDA events on the step's stream (the steps queue back to back,
one sync at the end), and the cost has a shape: within ±0.15 ms up to 16 rows, then growing with the rows
(the communicator alone: +0.11 to +0.19 ms at 24 rows, +0.40 ms at 32, +0.53 to +0.62 ms at 48, +0.77 ms at
72, +1.13 ms at 96). The step's all-reduces grow the same way (96 rows is a 1.4 MB message, which sparknet's
one-shot RoCE all-reduce carries with its CPU proxy), so the step's own collectives are the first suspect; a
per-kernel profile of the grid with and without the communicator would show it.

**Kernel profile** ([comm-profile-tp4-switched.json](comm-profile-tp4-switched.json)): one boot each of the
learner off ([tp4-switched-offstep-off-prof.json](tp4-switched-offstep-off-prof.json)) and the communicator
alone ([tp4-switched-offstep-comm-prof.json](tp4-switched-offstep-comm-prof.json)), each with the torch
profiler on every rank. [profile_decode.py](profile_decode.py) times the grid's 16×96, 16×48 and 1×3 rows
at 4K (the communicator alone cost +1.13 ms, +0.55 ms and nothing there), then profiles each shape in its
own window; [summarize_kernels.py](summarize_kernels.py) and [analyze_costs.py](analyze_costs.py) are the
indexer-decode and cluster-program experiments' copies. If the step's collectives' kernels carry the
difference, the learner has to stay out of NCCL; if every kernel slows, the cause is on the GPU's side.

Queue entry `20261117T193358-comm-profile-tp4-switched`, commit `6df9d7611`; receipts
[comm-profile](receipts/comm-profile) (each arm's grid times and per-rank kernel summaries, the lab's cost
tables, the timeline scripts and [timeline-16x96-sp1.txt](receipts/comm-profile/timeline-16x96-sp1.txt); the
traces stay under `results/private/determinism/turn-learning-comm-profile-tp4-switched/` on sp1). The comm
boot logged its communicator.

| Grid on these boots, step / forward (ms) | Learner off | Communicator alone | Change a step |
| --- | ---: | ---: | ---: |
| 16×96 @4K | 38.21 / 32.84 | 40.10 / 34.40 | +1.89 / +1.56 |
| 16×48 @4K | 26.74 / 21.15 | 27.32 / 21.70 | +0.59 / +0.55 |
| 1×3 @4K | 14.36 / 10.47 | 14.39 / 10.51 | +0.03 / +0.05 |

- **No kernel got slower; the timeline got longer.** In the 16×96 window every rank's span grew by the same
  49 ms (42 steps, +1.2 ms a step) while the same 73,752 launches took the same kernel time on sp1 (2,657 and
  2,650 ms). The step's all-reduces (`oneshot_allreduce_kernel`) took no longer: 690 and 683 ms on sp1.
- **The step's critical path lengthens after its collectives.** Aligning the step's stream kernel by kernel
  (1,255 a step), the step grows 1,224 µs, mostly as `mhc_project_streams_kernel` (the kernel that follows
  each all-reduce, launched early under programmatic dependent launch) ending later: +888 µs a step over its
  86 runs, then `gemm_mxfp8_kernel` +203 µs and `glp_steer_kernel` +141 µs. The side streams' kernels (the L2
  prefetch, MoE dispatch) run beside the step's stream as much as before (51.4% and 49.5% of stream 161's
  time; 99.9% and 98.6% of stream 160's), so nothing serializes; the step's stream gets its SMs later.
- **What changed between the boots is the streams' creation order.** Every side stream the step uses has an
  id three higher in the comm boot (160, 161, 403, 404 against 163, 164, 406, 407): the communicator's two
  NCCL streams and the side stream of its load-time exercise were created before them. CUDA multiplexes
  streams onto `CUDA_DEVICE_MAX_CONNECTIONS` hardware queues (32 here, already the most) in creation order,
  and the GPU arbitrates between queues, so a shift can change which kernel gets free SMs first. The arms'
  extra streams at load follow their costs: memory alone 0 (nothing), `step` 2 (the learner's capture and
  learn streams, +0.58%), the communicator alone 3 (+0.91%), off-step and `socket` 4 (+1.04%, +1.28%).
  This was the working explanation; the streams-alone job below rules it out.

**Streams alone** ([streams-tp4-switched.json](streams-tp4-switched.json); the model and grid steps, six boots
alternating): the learner off; the learner off with three bare CUDA streams made at load where the
communicator made its three ([tp4-switched-streams-3.json](tp4-switched-streams-3.json)); and the
communicator alone followed by 29 bare streams, 32 in all, the hardware queues' count
([tp4-switched-comm-pad-32.json](tp4-switched-comm-pad-32.json)); [overlay-streams/](overlay-streams) is
overlay-comm's speculator with `VLLM_DSPARK_TURN_LEARNING_EXTRA_STREAMS`. If three bare streams cost what the
communicator did and the communicator padded to 32 costs nothing, the cost is the step's streams' order on
the hardware queues, and the learner can keep its own communicator if what it makes at load is padded or
made after the step's streams.

Queue entry `20261117T195856-streams-tp4-switched`, commit `365d4231e`; receipts [streams](receipts/streams)
(grids, startup memory, telemetry, verdicts, [grid-split.txt](receipts/streams/grid-split.txt), each arm's
log lines: the stream boots logged three bare streams, the padded boots the communicator and 29 streams).

| Grid against the learner off (geometric mean, 20 shapes) | Whole step | Target forward | Boot to boot |
| --- | ---: | ---: | ---: |
| three bare streams at load | +0.07% | +0.08% | −0.13% |
| the communicator alone, padded to 32 streams | +1.23% | +1.48% | −0.12% |
| learner off, repeat | +0.02% | +0.06% | |

**Not the streams.** Three bare streams cost nothing, and the communicator costs as much padded to 32 streams
as alone (+1.23% here, +0.91% in the attribution job): the cost is the NCCL communicator itself, on RoCE or on
Socket, while it sits idle. The learner over the step's existing communicator (+0.58%, no communicator made)
is the one cost without one; its exchanges add connections to that communicator during the model step. What
an idle communicator holds is threads (its proxy's, the network's) and, once exercised, connections; the
step's critical path lengthens after its collectives, whose completion sparknet's CPU proxy drives. Next to
separate: the communicator made, exercised and destroyed (threads and connections gone); and made with its
threads confined to the cores sparknet's proxy and the worker do not use.

**The communicator's threads** ([commthreads-tp4-switched.json](commthreads-tp4-switched.json); the model and
grid steps, six boots alternating): the learner off; the communicator alone, exercised and then destroyed
([tp4-switched-comm-destroyed.json](tp4-switched-comm-destroyed.json)); and the communicator alone made and
exercised from a thread confined to the efficiency cores 0–4 and 10–14 (capacity 718–731 against 997–1024 on
5–9 and 15–19, all four nodes), so the threads NCCL starts inherit them (`NCCL_IGNORE_CPU_AFFINITY=1`), then
left idle ([tp4-switched-comm-little.json](tp4-switched-comm-little.json)).
[overlay-commthreads/](overlay-commthreads) is overlay-comm's speculator with both switches; it logs the
threads the communicator started and their cores, and what remains after a destroy. If confining the threads
removes the cost, it is CPU interference with the step's proxy or worker, and the learner keeps its
communicator made that way; if only a destroy removes it, it is what a live communicator holds.

The first run failed (queue entry `20261117T212544-commthreads-tp4-switched`, commit `81e1f9e7d`; receipts
[commthreads-failed](receipts/commthreads-failed): the four ranks' preserved start logs and the lab's log).
Its learner-off boot ran; the first destroyed boot hung at load and the lab rolled the cluster back after its
30-minute barrier timeout. Ranks 0 and 2 logged the communicator made, exercised and destroyed; ranks 1 and 3
never logged it, held in the exercise's last collective: `ncclCommAbort` is local and immediate, so the ranks
done first aborted while their peers' last transfers were still in flight. The destroy now waits at a barrier
over the group's CPU communicator first. The job runs again whole, as
[commthreads-2-tp4-switched.json](commthreads-2-tp4-switched.json).

The second run (queue entry `20261117T221018-commthreads-2-tp4-switched`, commit `4fb187f1b`; receipts
[commthreads-2](receipts/commthreads-2)) ran a learner-off boot and a destroyed boot, then stopped: the first
confined boot refused to start, as its arm is written to, because no core of 0–4 and 10–14 is the
process's. The serving container's cpuset is the performance cores alone (5–9, 15–19), and every thread in
it is pinned within them; the worker of rank 0 is pinned to core 18, and the three threads the communicator
started inherited core 18 (the destroyed boot's log, [destroyed-1-lines.log](receipts/commthreads-2/destroyed-1-lines.log);
the second pair of lines there is the same container restarted by the lab's rollback).

| Grid, one boot each (geometric mean, 20 shapes) | Whole step | Target forward | 16×96 | 16×48 | 1×3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| learner off → the communicator made, exercised and destroyed | +1.44% | +1.81% | +1.10 ms | +0.64 ms | +0.26 ms |

**Destroying the communicator does not remove the cost** (one boot of each; the repeat did not run). With its
threads gone (none left after the destroy) and its connections closed, the grid is as slow as with it alive:
the cost is something that outlasts the communicator once NCCL has been set up and run, not its threads or
connections, so confining its threads (which the container cannot do here anyway) would not have helped. The
`step` arm fits that too: no communicator made, but the learner ran NCCL operations the step does not (an
FP32 all-reduce and an all-gather over pynccl) on the step's communicator. What stays after a destroy is
process-wide: the NCCL kernels loaded for those operations (with CUDA's lazy module loading) and whatever
NCCL set up at its first use.

**NCCL operations on the step's own communicator** ([stepops-tp4-switched.json](stepops-tp4-switched.json); the
model and grid steps, six boots alternating): the learner off, and no communicator made, with NCCL operations
run once at load on the tensor-parallel group's existing pynccl communicator ([overlay-ncclops/](overlay-ncclops)):
the learner's exchanges, 16 FP32 all-gathers and 16 FP32 all-reduces at a chunk's sizes, operations the step
never runs over pynccl ([tp4-switched-stepops-learner.json](tp4-switched-stepops-learner.json)); or 32 BF16
all-reduces of 4 MiB, the kind the step runs over pynccl for its larger all-reduces
([tp4-switched-stepops-bf16.json](tp4-switched-stepops-bf16.json)). If the learner's operations cost and the BF16
ones do not, running a kind of NCCL operation the step does not is what costs, whatever communicator runs it.

Queue entry `20261117T223644-stepops-tp4-switched`, commit `2d7f711e3`; receipts [stepops](receipts/stepops)
(grids, startup memory, telemetry, verdicts, [grid-split.txt](receipts/stepops/grid-split.txt), the arms' log
lines: each operations boot logged its operations).

| Grid against the learner off (geometric mean, 20 shapes) | Whole step | Target forward | Boot to boot |
| --- | ---: | ---: | ---: |
| the learner's FP32 all-gathers and all-reduces at load, on the step's communicator | −0.26% | −0.34% | −0.02% |
| BF16 all-reduces at load, on the step's communicator | −0.08% | −0.08% | −0.29% |
| learner off, repeat | +0.18% | +0.13% | |

**The operations cost nothing.** Run on the step's existing communicator, the learner's exchanges leave the
grid where it is with the learner off. What has cost about 1% in every job is a second NCCL communicator
made: alive and idle (+0.91%, +1.23%), destroyed after use (+1.44%, one boot), on Socket with the learner
(+1.28%), on RoCE with the learner (+0.98%, +1.04%). The learner over the step's communicator, with no second
one made, read +0.58% (boots +0.79%, +0.40%); the attribution job put the learner without a communicator at
+0.13%. A second communicator's cost outlasts its destroy, so it is something its making leaves behind, not
anything it does while it exists.

**Two builds without the cost** ([builds-tp4-switched.json](builds-tp4-switched.json); the model and grid steps,
six boots alternating), each the whole off-step learner: the learner off; the learner's exchanges over the
step's own communicator, none made ([tp4-switched-exchange-step.json](tp4-switched-exchange-step.json), the
`step` arm measured once above, measured again); and the learner's own communicator made at its first
exchange, after startup has made the KV cache and captured the graphs
([tp4-switched-lazycomm.json](tp4-switched-lazycomm.json), [overlay-lazycomm/](overlay-lazycomm)). The first
end of turn comes after both (a boot's log: model loaded, KV cache made, graphs captured, then requests).

Queue entry `20261117T233008-builds-tp4-switched`, commit `659b3ff8b`; receipts [builds](receipts/builds) (grids,
startup memory, telemetry, verdicts, [grid-split.txt](receipts/builds/grid-split.txt), the learner boots' log
lines: each logged its exchange, 151 to 157 calls in its model step, no error). Within the job, against its two
learner-off boots (19.103 and 19.178 ms, +0.39% apart): the step's communicator +0.49%, the communicator made at
the first exchange +0.51%, the two builds 0.01% apart.

**All of the day's grids** ([pooled.py](receipts/builds/pooled.py), [pooled.txt](receipts/builds/pooled.txt)),
against the pool of the 15 learner-off boots (one profile and image in every job; 19.156 ms, sd 0.15%):

| Arm | Boots | Grid (whole step, geometric mean) |
| --- | ---: | ---: |
| memory held from load (150, 350 MiB) | 4 | +0.04% |
| three bare streams at load | 2 | +0.05% |
| the learner's NCCL operations at load, on the step's communicator | 2 | −0.13% |
| BF16 all-reduces at load, on the step's communicator | 2 | +0.05% |
| a second communicator made at load, idle | 4 | +1.13% |
| a second communicator made at load, destroyed | 1 | +1.23% |
| the learner, its own communicator made at load (RoCE) | 4 | +1.06% |
| the learner, its own communicator made at load (Socket) | 2 | +1.13% |
| **the learner, the step's communicator** | 4 | **+0.43%** |
| **the learner, its own communicator made at its first exchange** | 2 | **+0.43%** |

- **A communicator made at load costs about 0.6 of the learner's 1.06%, and making it at the first exchange
  removes that part.** Why a communicator made at load slows the grid's decode forward, and keeps doing so
  once destroyed, is not known; one made after startup does not.
- **The learner's own part is about 0.43%, whichever communicator it uses.** It is not its memory, its
  streams, or its NCCL operations (each costs nothing alone); what is left is what it runs: the capture
  kernel in the draft graph (the draft step's time is unchanged, +9 and −2 µs), and its end-of-turn work
  in the model step before the grid (its kernels' first loads, its exchanges).

**The capture against the end of turn** ([split-tp4-switched.json](split-tp4-switched.json); the model and grid
steps, six boots alternating): the learner off; the learner with its end of turn reduced to clearing the rings
([tp4-switched-skipend.json](tp4-switched-skipend.json), [overlay-skipend/](overlay-skipend): the draft step's
capture as shadow runs it, nothing staged, scored, exchanged or trained, so no communicator is made); and the
whole learner with its communicator made at the first exchange (tp4-switched-lazycomm.json again). The
learner's kernels are declared for the startup warm-up in both learner arms, so the difference between them is
the end of turn's running.

Queue entry `20261118T002423-split-tp4-switched`, commit `4dea1ea17`; receipts [split](receipts/split) (grids,
startup memory, telemetry, verdicts, [grid-split.txt](receipts/split/grid-split.txt), the pooled table again,
the learner boots' log lines: the capture boots logged the end of turn skipped and made no learner call; the
whole-learner boots made their communicator at the first exchange and ran 148 and 157 calls). Within the job:
the capture alone +0.47% (boots 0.01% apart), the whole learner +0.31% (boots 0.48% apart), against learner-off
boots 0.23% apart. Against the day's 17 learner-off boots (sd 0.15%):

| Arm | Boots | Grid |
| --- | ---: | ---: |
| the learner, end of turn skipped (the capture alone) | 2 | +0.48% (+0.49, +0.48) |
| the whole learner, its communicator made at its first exchange | 4 | +0.37% (+0.41, +0.45, +0.56, +0.08) |
| the whole learner, the step's communicator | 4 | +0.42% |

**The end of turn costs nothing measurable; the learner's 0.4% comes with the capture.** It is the target
forward's time (+0.57% within the job) while the draft step's is unchanged (+6.5 µs), and the capture's
kernel cannot overlap the target forward: it is forked onto the learner's side stream after the last
position's maxima and joined before the draft graph ends. What shadow changes besides the end of turn is all
set up at load or captured into the draft graph: the learner's allocations and streams (each costing
nothing alone), its kernels compiled and loaded for the warm-up, a draft graph with a forked branch, and the
draft step's variants that keep each position's maxima (the first position's fused pairs with its tiles'
maxima, the partials in per-position scratch). On-step shadow, the same capture, read +0.40% against
candidate-60's recorded screen a day apart (+0.2% beyond that day's drift).

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

Off the step, the end of turn holds the step 0.02 ms a call at the median (0.20 ms at p90 under AgentX,
against 0.48 and 2.86 ms on the step) and its outputs are candidate-60's. The grid's decode forward pays for the
learner: about 1.06% with its communicator made at load, 0.43% with it made at the first exchange or with the
step's communicator instead. That 0.4% comes with the per-step capture, not the end of turn, which costs
nothing measurable once its communicator is made after startup.
