# DSpark: probabilistic drafting

Owner, 2026-10-18: "we should also test probabalistic draft_sample_method".

## Why

DSpark drafts greedily (`draft_sample_method: greedy`): each position proposes the
drafter's argmax, and rejection treats the draft distribution as one-hot. The recipe now
samples at temperature 1 (`--override-generation-config {"temperature":1.0,"top_p":1.0}`,
DeepSeek's 1017 defaults), so most served tokens are sampled. With `probabilistic`, drafts
are sampled from the drafter's distribution and verified against it, which accepts a draft
with probability 1 - TV(target, draft) instead of the target's probability of the
drafter's argmax: a lossless change that can only raise acceptance where the two
distributions agree. The cost is a per-step cache of the draft logits (16 x 5 x the
vocabulary) and the ratio test.

Nothing in the code changes: vLLM's DSpark speculator caches the draft logits when the
method is `probabilistic`, and the block-verification kernels take them
(`HAS_DRAFT_LOGITS`). Temperature-0 outputs must stay identical: at temperature 0 the target
samples its argmax and verification accepts exact matches.

## Runs (`screen.json`, the ring, 1017, r6v image)

1. The determinism gate on the probabilistic arm: temperature-0 and greedy outputs equal to
   r6v's (3eab5112 / db25fde3, c9cf9dbe).
2. The screen suite (model behaviour, engine grid, c16, prefill) and the single-stream
   suite: probabilistic, greedy, probabilistic again. Both arms price drafts from r6v's
   pinned cost curves, so the comparison is the sampling method alone.

Gate: no metric worse; acceptance and sampled throughput better to adopt it.

## First results (screen, single-stream)

Determinism identical (3eab5112 / db25fde3, c9cf9dbe / c6bb4f4e). But servebench's suites
decode greedily (`"temperature": 0` in every request), where probabilistic drafting cannot
accept more and only adds its per-step cost: tokens a step unchanged (c16 30.56 against
about 30.5; c1 2.245 against 2.24-2.26), engine steps a second -3.4% at c16 and -1% at c1,
so throughput fell by the same. `sampled.json` measures the case it targets, at the recipe's
temperature 1 (the new `sampled` suite: c16, c8, c1, fixed seed).

## Results and verdict: not adopted

Verdicts against each run's own repeat (probabilistic, greedy, probabilistic again):

| Suite (traffic) | Metric | Greedy drafting | Probabilistic | Change | Verdict |
| --- | --- | ---: | ---: | ---: | --- |
| sampled (temperature 1) | c16 tok/s | 237.2 | 230.2 / 230.2 | -2.93% | worse |
| sampled | c8 tok/s | 180.7 | 176.1 / 176.3 | -2.55% | worse |
| sampled | c1 tok/s | 68.6 | 68.1 / 68.1 | -0.76% | same |
| screen (greedy) | engine grid, step time | | | +9.15% | worse |
| screen (greedy) | c16 tok/s | 241.8 | 234.2 / 234.4 | -3.17% | worse |
| single-stream (greedy) | c1 tok/s | 70.8 | 69.9 / 70.1 | -1.38% | same |

Where it was meant to pay, sampled traffic, it raised tokens a step by 0.1-0.5% (c16 29.86
and 29.96 against 29.72; c1 2.18 against 2.17) while its per-step cost (caching the draft
logits over the whole vocabulary for every draft row, and the ratio test) made a decode step
about 9% slower. Block verification of greedy drafts already accepts nearly as much under
sampling, and the drafter's distributions are confident enough that sampling from them
changes little. Greedy drafting stays.

Also measured here: on the same build, sampled traffic runs 2-3% slower than greedy traffic
(c16 237.2 against 241.8 tok/s, c1 68.6 against 70.8) because fewer drafts are accepted; the
greedy suites never saw that gap, which the `sampled` suite now records.
