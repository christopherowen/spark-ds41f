# Draft acceptance: opportunities found while porting token counts

The kernel port preserves counts exactly. Raising a reported acceptance ratio by
changing bookkeeping would not make more tokens pass the verifier. The useful
objectives are more verified output tokens per step and per second, at unchanged
target distributions and reference arithmetic.

## First: longer suffix proposals on copy-heavy agent output

The [existing same-output copy probe](../2026-10-23-next-release/README.md#draft-acceptance-acceptance)
already bounds this opportunity on the current DS41F stack. DSpark achieves about
4.96 tokens per step when reproducing a file and 5.10 when adding comments. A
simulated suffix drafter with an 8-token match and up to 16 proposed tokens gets
15.9 and 14.9 on those same outputs. Tool-call copies improve from 5.41 to 7.1;
prose stays at 2.24. The estimated workload-weighted gain is 7–17% tokens per step
before paying for wider verification, not a measured throughput improvement.

The source confirms the integration boundary: `DSparkSpeculator` fixes
`num_query_per_req` to its trained block width and samples its Markov-conditioned
positions sequentially. The target runner already compacts per-request admitted
draft counts, but still allocates confidences and captures around the configured
maximum. A suffix proposal therefore needs an explicit hybrid proposal source,
a wider maximum verification shape and compatible per-request metadata; merely
raising DSpark's five-token setting is not this experiment.

A useful next implementation is a deterministic suffix proposal for long, repeated
code spans, with DSpark for requests without a match. Verify every proposal with
the existing lossless target procedure. Gate on the same-output copy benchmark,
unchanged temperature-0 output, distribution-correct temperature-1 verification,
and total time including lookup and wider target rows. Keep a one-stream benefit
separate from sixteen-stream results, where larger row batches cost more.

## Second: measure confidence calibration before changing draft budgets

`DSparkSpeculator.load_draft_model` disables the online acceptance estimator when
the checkpoint has a confidence head. `compute_confidence` returns its raw sigmoid;
`_sample_sequential` publishes those values, and `AdaptiveVerificationManager`
multiplies them into survival probabilities. It ranks admitted prefixes by those
products and chooses the total budget using a fixed row cost (0.035 here).

This identifies a measurement opportunity, not a demonstrated miscalibration.
Check predicted conditional probabilities and prefix survival against actual
verdicts, by draft position, request temperature, copy/prose/tool traffic and
concurrency. Grade only genuinely observed positions: accepted drafts and the
first actual rejection. Drafts omitted by the budget, dead-row cuts or sink
recovery must not become negative training examples. Keep unverified proposals
separate from rejected proposals in the measurements.

If the scores are biased, a small deterministic calibration layer could improve
admission without changing the proposal tokens or relaxing verification. First
replay budget choices on the logged verdicts; then compare accepted tokens per
step and time, including any changed target row count. More admitted rows alone
is not better acceptance. No calibration or budget coefficient is changed here.

## Lower priority: branching, and an already rejected alternative

A draft tree can recover from the first linear mismatch, but needs ancestor-aware
state through sparse MLA, SWA, compression, Engram and CED. Measure target mass
in the drafter's top alternatives before building those paths. This is a larger
project than the linear suffix proposal.

[Probabilistic drafting was already measured](../2026-10-18-dspark-probabilistic/README.md#results-and-verdict-not-adopted):
on sampled traffic it gained only 0.1–0.5% tokens per step while adding full-vocabulary
logit-cache and ratio-test cost. Throughput fell about 2.6–2.9% at 8/16 streams.
It should not be repeated without a concrete way to remove that cost.

These changes all retain lossless target verification. None is implemented or
promoted by the token-count migration.
