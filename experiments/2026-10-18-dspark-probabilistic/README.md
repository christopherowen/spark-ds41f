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
