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
3. **Shadow**: the update in TileLang at the end of each turn on a shadow W1, scored against the live one
   on each new turn; drafts unchanged.
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

## Observe run (prepared)

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
