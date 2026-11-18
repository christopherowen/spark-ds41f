# Where the DSpark drafter's misses are, and what each way of raising its accuracy is worth

Date: 2026-11-18. Base commit: `9a8a33a24` (main). Offline only: no boot, no build.

## Question

Two thirds of drafts are accepted at every depth, and a step ends on a miss 87% of the time.
What is a point of accuracy worth, where are the misses, and which ways of removing them have
a size worth a serving experiment?

## What was already measured

Read before anything was computed, so this record does not repeat it:

- The drafter is the checkpoint's own (`mtp.*`, three layers on target layers 37 to 39, a
  rank-256 Markov bias); we do not train it. Native BF16 heads are in
  (`2026-10-03-native-drafter-heads`), probabilistic drafting lost (`2026-10-18-dspark-probabilistic`).
- Learning the Markov table at the end of a turn: +1.52% offline on AgentX turns, an upper
  bound, not yet measured in serving (`2026-11-17-dspark-turn-learning`).
- A tree at position 1 with every row verified lost 38% at eight streams
  (`2026-10-26-dspark-acceptance`); candidate-61 found the target's verification kernels take
  only linear chains (`2026-11-16-c61-decode`, rank 4).
- `2026-10-26-dspark-acceptance` left two checks prepared and unrun: GSM8K in the chat format
  against the external 4.27 tokens a step (`probe.json`, about 40 minutes of ring), and a CPU
  parity run of our drafter against the reference. Ours reached 3.56 with every draft verified.

## Method

[miss_anatomy.py](miss_anatomy.py) on the turns of boot `20261117T204820` that carry the
request's tokens (`2026-11-17-copy-drafter-reach`'s observe run: candidate-60, tp4-switched,
AgentX, 8 sessions). A step's five hidden rows do not depend on what it drafts, so with the
checkpoint's head and Markov tensors any scoring of those rows replays exactly: a step's accepted
run is its leading run of right answers under the committed tokens.

```bash
python miss_anatomy.py features --turns <turns> --ckpt <drafter tensors> --out <dir>
python miss_anatomy.py study --out <dir>
```

325 turns with DSpark steps, 49,140 steps ([receipts/anatomy.json](receipts/anatomy.json)). The
replay gives 2.7516 tokens a step and the served drafts 2.7517: the replay is the engine's
drafter. Tokens a step here count every draft as verified.

## Results

**The misses are near misses.** Right at each depth given the earlier ones right: 0.677, 0.668,
0.653, 0.665, 0.672. At the miss that ends a step (42,655 of them) the committed token is the
drafter's

| Rank of the committed token | Share of those misses |
| --- | ---: |
| 2 | 34.2% |
| 3 to 5 | 30.1% |
| 6 to 20 | 21.5% |
| over 20 | 14.2% |

and the drafter's lead over its runner-up is small there: a median of 0.92 logits against 4.79
where it is right. The committed token had appeared earlier in the request at 73.6% of them.

**Ceilings**, tokens a step if the committed token were drafted whenever it ranks in the top k:

| k | Tokens a step | Against 2.752 |
| --- | ---: | ---: |
| 2 | 3.390 | +23.2% |
| 3 | 3.748 | +36.2% |
| 5 | 4.159 | +51.2% |

**The Markov bias carries a token a step.** Left out, the same rows give 1.735.

**The request's own earlier tokens add little.** Drafting, in place of the drafter's top-1, the
token that followed the most recent earlier occurrence of the last k tokens, when the match is
at least 4 tokens and the drafter's top-1 leads that token by under 8 logits (both chosen on
the first half of the turns): 2.757 to 2.793 on the second half, **+1.3%**, switching 12.7% of
positions. Past 6 matched tokens that token is right more often than the drafter (0.893 against
0.881), but such positions are 17% of the live ones and the drafter is already right at most.

**A runner-up leaf**: one more verified row whose input is the runner-up at one depth, which adds
a token when the step's first miss is there and the committed token is the runner-up.

| Depth given the leaf | Rows a step | Tokens added | Throughput at 0.035 a row | at 0.07 |
| --- | ---: | ---: | ---: | ---: |
| the first miss, known in advance (ceiling) | 0.30 | +0.297 | +9.6% | +8.5% |
| the first depth with a lead under 2 logits | 0.90 | +0.175 | +3.1% | 0.0% |
| the first depth with a lead under 1 logit | 0.80 | +0.137 | +2.1% | -0.6% |
| the smallest lead of the first two | 1.00 | +0.146 | +1.7% | -1.6% |

## Conclusion

- **A point of accuracy is worth what was assumed**, about 1% of tokens a step, and the misses
  are ranking errors: two thirds of them have the committed token in the drafter's top five.
- **Nothing cheap reaches them.** Re-ranking with the request's own tokens is +1.3% of tokens a
  step, turn learning +1.5% at most; they draw on overlapping evidence and neither is serving
  throughput.
- **A runner-up leaf is +2 to 3% at one stream with a simple rule, +9.6% at its ceiling**, and
  needs the target's verification to take a branch, which candidate-61 found it cannot. It
  loses at twice the row cost, so it is a one-stream change.
- **The largest open item is not a new idea but an unrun check**: whether our drafter reaches
  the reference's acceptance (3.56 against the external 4.27 on GSM8K). If the format explains
  the gap nothing is lost; if the drafter path differs from the reference the size is the gap.

## Limits

- AgentX turns of one boot at eight sessions; prose and code suites accept differently.
- Tokens a step with every draft verified, not serving throughput; the row cost is the
  one-stream figure of docs/current-state.md.
- The leaf has no continuation after it, and the rows it would displace under the budget are
  not modelled.
- 120 of the 445 dumped turns have no usable DSpark step (short turns and ring entries from
  other requests) and are left out.
