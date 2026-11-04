# Exact clustered indexer search

Status: unpromoted experiment. The target is identical selected keys and attention
outputs with a materially lower indexer cost. No quality-changing approximation
is accepted. No production configuration or published baseline is changed.

One variable: replace a dense indexer search with a bounded search over clusters
of 64 consecutive keys in the checkpoint's existing 128-dimensional index space.
There are no trained maps, new weights, reduced precision or changed attention
arithmetic. This is an exact search experiment, not a replacement attention rule.

## Algorithm and rounding argument

1. Build per-coordinate minimum and maximum values of every cluster's **visible**
   MXFP4 keys. Metadata is rebuilt for every snapshot in this first implementation.
2. For each head, choose the interval endpoint with the query coordinate's sign.
   Its dot product bounds that head's real dot product for every member key.
3. Inflate the FP32 dot estimate by `sum(abs(q) * max(abs(lo), abs(hi))) / 256`
   plus `2^-50`, then apply the scorer's BF16 rounding, ReLU, positive head
   weight and BF16 rounding. Negative-weight contributions are bounded by zero.
   Add head bounds in the same FP32 head order and round to BF16.
4. Score keys from the 32 clusters with the highest bounds with the unchanged
   MXFP4 scorer. Their kth selected value is a lower bound on the global kth value.
5. Keep every cluster whose bound is **greater than or equal to** that threshold.
   Compact its members in increasing position order, then run the unchanged
   candidate scorer and deterministic top-k. Equality must survive because a
   skipped earlier position could win a threshold tie.

For `u=2^-24`, a 128-term normal FP32 reduction has absolute error at most
`gamma(127) * A`, where `gamma(n)=n*u/(1-n*u)` and `A=sum(abs(products))`.
Endpoint selection makes the real interval dot no smaller than any member's
real dot. Allowing errors in both reductions requires at most `2*gamma(127)*A`.
The computed absolute envelope is at least `(1-gamma(127))*A`, and the `1/256`
allowance exceeds that requirement by over 250 times, including FP32 rounding
of the allowance and its addition. The smallest positive product in the
supported range is `2^-64`; neither products nor nonzero dot sums underflow.
Products of two decoded FP4 values are exact in FP32. At the largest supported
scales and head weights, dots and contributions cannot overflow BF16 or FP32.
The extra slack also covers zero and cancellation. BF16 round-to-nearest and
FP32 addition are monotone. Dropping negative weighted contributions can only
raise the bound, including when the true ordered head sum is negative.

Only UE8M0 scale bytes 96 through 144 and finite BF16 head weights of magnitude
at most `2^20` are certified here. A cluster or query outside that range gets an
infinite bound and is fully scored. It never drops a key based on an unsupported
encoding. Hardware execution must still establish that the compiler and MXFP4
MMA obey the assumed arithmetic; CPU checks alone do not qualify this proof.

This preserves the final scorer and tie rule, but exact output is a **gate**,
not an assertion inferred from an algorithm description. Every GPU case compares
all selected positions and checks each bound against every visible full score.
The gate includes signed weights, empty/short rows, page tails, noncontiguous
physical pages, allocator padding, unsupported scales, ties, and a mutated cache.

## Scope and risks

The implementation is a standalone snapshot plan. It is not selected by a
serving flag and does not yet own CSA source/reindex/reuse state, incremental
metadata, prefix-cache reuse, speculative rollback, or a shared serving workspace.
A rebuild on every call prevents stale metadata. Its cost and allocations are
included in candidate timings. Coordinate intervals may be too loose for real
keys, in which case little or nothing is pruned and the candidate will lose.
A separable synthetic case demonstrates pruning mechanics only; random inputs
and real model snapshots must be reported separately.

The next useful decision comes from real indexer snapshots: exact selection,
pruned-key fraction and complete candidate cost, including updates and compaction.
Only then is serving integration justified. A large kernel gain alone does not
establish an end-to-end gain.

## Reproduction

The patch applies after the promoted vLLM series at tree
`ea1281f880e7535e3d87725be0156a8e38fc2fb2`, upstream base
`19f2c20ed4d64ae22bd009ceacadd85e2a81e354`. Its status is experimental, not
submitted upstream. `bundle/overlay` is a byte-identical executable copy of the
new module; `check_sources.py` verifies it against the patch.

```sh
bin/spark doctor
bin/spark upstream prepare vllm
python3 experiments/2026-11-04-clustered-indexer/check_sources.py
python3 -m unittest discover -s experiments/2026-11-04-clustered-indexer -p test_reference.py -v
git diff --check
```

Inside an authorized cluster window, on an idle node with the recorded release
image present, use the existing kernel lab. It refuses a serving container and
holds the node's GPU lock. Run twice in fresh containers and retain every result.
The release scorer is run for **correctness only**. The harness times only the
candidate; it does not remeasure released performance.

```sh
scripts/lab.py kernel-local experiments/2026-11-04-clustered-indexer/bundle --out results/private/clustered-indexer-a
scripts/lab.py kernel-local experiments/2026-11-04-clustered-indexer/bundle --out results/private/clustered-indexer-b
```

Full qualification still requires two candidate serving boots on each target
shape, exact attention and greedy digests, teacher-forced distributions, mixed
traffic, and memory/admission gates at the frozen workload limits, compared with
the **recorded** release suites. Only the owner may accept promotion.

## Initial hardware results

`runs/attempt-01` retains the failed TileLang build (an integer was used as a
boolean condition). `runs/attempt-02` is the corrected build's first complete
GB10 pass: six snapshot cases, every selected position identical and no bound
violation. At 16,383 keys the separable case keeps 576; the random case keeps all
16,383. This is a useful correctness result and an unfavorable pruning result,
not evidence of a serving speedup. Expanded tests and repeats are recorded
separately; see `results.json` for the final status.

The expanded `runs/attempt-03` passes all 12 cases, adding normal-scale extremes,
NaN keys/queries/weights, a cache mutation that changes the expected winners,
and 262K-key domains. At 262,143 keys, the separable input retains 576 keys and
the random input retains all of them. Candidate-only median wall times, including
rebuilding and allocations, are 3.72 and 5.87 ms respectively. These are different
inputs, so their ratio is **not a speedup measurement**. The released scorer was
used for correctness only, never timed.

Seven CPU cases pass. The source checks also reverse the capture patch and
recover the exact recorded candidate-36 indexer. Full build inputs are absent;
`build check` reports that the build tree is not prepared. The image was neither
rebuilt nor modified. See `decision.md` for outstanding gates and the window's
closure, and `capture/README.md` for the unexecuted real-input workflow.

`runs/attempt-04` repeats all 12 expanded cases in a fresh container: every
selected-position digest matches `attempt-03`. `memcheck-01` passes all 12 cases
with **0 errors**; `racecheck-01` passes the 10 cases through 16K keys with
**0 hazards, 0 errors and 0 warnings**. Sanitizer timings are excluded from
performance interpretation. `provenance.json` records the exact published
harness commit and command for every run and the sanitizer's version/hash.
The second window closed at 09:07:10 UTC, again with the original profile
already serving. Initcheck and real-input replay remain pending.
