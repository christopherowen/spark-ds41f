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

## Hardware results and decision

GB10 SM121, the exact recorded `spark-ds41f:1030` image. The released scorer is
used for correctness only, never timed. `provenance.json` records each published
harness commit, exact command, and the sanitizer version and binary hash.

- `attempt-01`: failed TileLang build; an integer condition needed an explicit
  boolean comparison. The complete error is retained.
- `attempt-02`: six corrected snapshot cases pass.
- `attempt-03` and `attempt-04`: all 12 expanded cases pass in two fresh
  containers. Cases include normal-scale extremes, NaN keys/queries/weights,
  ties, a cache mutation that changes the winners, and domains through 262K keys.
- `initcheck-01`: functional selection passes, but strict initialization checking
  **fails with 4,608 unwritten reads** in the existing top-k merge. Its bitwise
  condition reads scratch entries from inactive lists before masking them.
  The independent `topk-original-01` reproducer reports 1,536 errors without
  invoking the clustered search. Both failures are retained.
- The independent [top-k fix](../2026-11-04-topk-live-reads/README.md) guards the read with a lazy outer
  condition, preserving live arithmetic and ordering. The independent
  `topk-fixed-01` reproducer and complete `initcheck-02` report **0 errors**.
  Initcheck covers 10 cases through 16K with fresh allocations and cache tails
  left unwritten. Allocator statistics are unavailable in this mode; reported
  raw zero counters do not mean zero GPU memory usage.
- `eager-fixed-01` and `eager-fixed-02`: all 12 cases pass, retaining the archived
  selected-position digests. `graph-01` and `graph-02` do the same in two fresh
  containers, including replay after reduced visibility, an empty active
  domain, and restoration of the original counts.
- `memcheck-02`: all 12 cases, **0 errors**. `racecheck-02`: 10 cases through 16K,
  **0 hazards, 0 errors, 0 warnings**. The earlier `memcheck-01`/`racecheck-01`
  receipts are retained at their original source revisions. Sanitizer timings
  are excluded from performance interpretation.
- Seven CPU arithmetic/selection cases pass. Python 3.12 syntax, patch/overlay
  identity and both local doctors pass. Freshly applying the 558-patch capture
  base reproduces candidate-36's exact source tree.

At 262,143 keys, separable synthetic input retains 576 keys; random input retains
all 262,143. The guarded candidate's median wall times are:

| Execution | Separable, two fresh containers | Random, two fresh containers |
| --- | --- | --- |
| Eager | 3.690 / 3.683 ms | 5.801 / 5.854 ms |
| CUDA graph | 3.657 / 3.667 ms | 5.840 / 5.834 ms |

Complete candidate timing includes metadata rebuilding and compaction. The two
input distributions are different, so their ratio is **not a speedup**. There is
no released-build or end-to-end speed claim. Graph capture barely changes the
long-domain cost. The candidate-only instrumented `graph-profile-01` trace puts
about **85% of GPU kernel time in envelope rebuilding** and 9–13% in query
bounds. Raw traces, hashes and stage totals are in `profiling.json`. These
instrumented durations diagnose the bottleneck; they are not performance gates.

The largest observed PyTorch allocation peak was 47,190,016 bytes; this excludes
the CUDA context and is not total host memory. The latest slot began with about
116 GiB of host MemAvailable and a hottest sampled sensor of 40.2 C. Its fixed
source commit is `6cc82a8956a1a67bb21dd60ef969f30063c45fbe`.

The speed target is not established. First reduce snapshot metadata cost and
collect real DeepSeek operands to decide whether the bounds prune useful work.
Persistent metadata would need explicit cache ownership, append, overwrite,
rollback and reuse rules. Approximate selection is not an accepted shortcut.
See `decision.md`, `results.json` and `capture/README.md`.

The three windows closed with the original candidate-36 TP3 profile already
serving (08:23:17, 09:07:10 and 10:15:21 UTC). The latest dgx4 slot ran from
10:05:43 to 10:15:21 UTC and was returned for the other task's qualification.
No serving start, stop, restart or cluster sync was performed. Foreign requests
were preserved. Real-input capture/replay remains pending; its prepared hook
has not run. Full image build inputs are absent locally; `build check` reports
that the build tree is not prepared. No image was rebuilt or modified.

## Prepared envelope resource change

`vllm/0003-stream-envelope.patch` changes one causal variable: the envelope
reduction's register state. It streams the same visible keys with two endpoints
per coordinate instead of materializing two 64-by-128 fragments. The decoder,
unsupported-scale handling, bounds, scorer, threshold and tie order are unchanged.
This remains a complete snapshot rebuild, with no persistent metadata.

`resource-layout.json` records its base and gates. Hardware results above belong
to the preceding reduction at their published commits and do not qualify this
candidate. The harness now compares both interval endpoints against a frozen
copy of that reduction before timing, then runs all existing exact-selection
checks. `graph-profile` uses a new trace label to preserve earlier evidence.
The first two fresh runs, graph replays and sanitizer checks remain pending.

## Independent top-k fix

The scratch-read fix and its reproducer are owned by
`experiments/2026-11-04-topk-live-reads/` in [PR #878](https://github.com/christopherowen/spark-ds41f/pull/878).
This search series references that patch and tested executable directly.
Historical receipts retain the old paths and exact commits at which they ran.
The attention experiment is stacked on that independent fix for review.
