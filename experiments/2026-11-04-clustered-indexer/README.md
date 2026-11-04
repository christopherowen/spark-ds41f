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

## Hardware evidence

Every native receipt, error, exact command and input hash is retained under
`runs/`; `provenance.json`, `results.json` and `sp8-results.json` index the runs.
The native scorer is correctness-only. Timings below measure unreleased
prototypes, including metadata rebuilding, compaction and candidate selection.
They are not released-build or serving speedups.

The initial dgx4 prototype passed 12 fixtures in two eager and two graph
processes. Its medians at 262,143 keys were about 3.7 ms (separable) and 5.8 ms
(random). Profiling identified envelope rebuilding as 85% of kernel time.
The first compile failure and the top-k initialization failures (4,608 in the
full search and 1,536 in the independent reproducer) remain recorded. The lazy
scratch-read guard fixed those reads without changing live arithmetic; full
initcheck, memcheck and racecheck then passed. Three dgx4 windows closed with
candidate-36 production unchanged, at 08:23:17, 09:07:10 and 10:15:21 UTC.

On sp8, the screened changes have these two fresh-process medians:

| Prototype; 262,143 keys | Separable | Random |
| --- | --- | --- |
| Archived unreleased envelope | 3.591 / 3.583 ms | 5.678 / 5.694 ms |
| Streamed envelope, scalar bounds | 0.769 / 0.791 ms | 1.070 / 1.092 ms |
| Streamed envelope, tensor-core bounds | 0.298 / 0.306 ms | 0.551 / 0.553 ms |

All 20 fixtures preserve selected keys and pass full-score bounds checks and
graph visibility rollback/empty/restore. Streamed and tensor-core variants each
pass initcheck (18 fixtures), memcheck (20), and racecheck (18), with zero errors
or hazards. Exceptional scales, large signed weights, cancellation, subnormals,
NaNs, page tails and unwritten allocations are included. Two of the fixtures
are too large for the bounded init/race screen. Eleven CPU tests pass.
`sp8-profiling.json` records instrumented diagnostics: tensor bounds reduce their
stage cost, leaving snapshot rebuilding dominant. Traces are not timing gates.

The explicit one-owner grouped snapshot also passes two graph processes with
four multi-query fixtures. At 65,536 planned keys and 128 separable query rows,
sharing metadata changes candidate medians from 7.046 / 7.047 ms to
0.767 / 0.767 ms. For random rows they change from 13.383 / 13.466 ms to
3.498 / 3.530 ms. Metadata tensors shrink from 128 MiB to 1 MiB and the
harness allocation peak from 181.3 MiB to 64.8 MiB. These are test allocations,
not serving memory or admission measurements. Six fixtures including the 1/256-row boundaries now pass twice; dedicated
initcheck and memcheck pass all six, and racecheck passes two bounded fixtures. No persistent cache metadata is used.

All native sp8 jobs have returned their host reservation. No serving state,
image tag, production configuration or immutable baseline changed. Complete
image build inputs are absent locally; `build check` reports an unprepared
build tree. Real-input capture/replay and serving qualification remain pending.

## Source-layer handoff gate

`vllm/0006-preserve-source-handoff.patch` prepares a joint source-layer plan.
Source rows publish both top-512 keys and top-2048 blocks of eight positions;
a later reindex query depends on the block IDs, their order and position count.
A threshold from immediate winners alone cannot qualify this interface.

The source candidate scores a larger seed, takes the ordinary block cutoff
excluding the native forced-last block, and keeps all bound ties plus the global
last visible cluster. Retained scores are restored to the native logical grid
before unchanged block selection. A shared lookup and one writer per dense
position avoid scatter races and initialization fills. Small planned domains
use the native source plan directly and are correctness-only, never timed.

`source-handoff.json` records the variable. Its CPU oracle and fresh patch/source
checks pass. Two eight-case source runs, seven-case initcheck and memcheck,
and two-case racecheck all pass. Source bounds currently retain every visible
key even on the separable fixture: preserving the larger block handoff is a
stricter requirement. This unfavorable pruning result is retained. The harness compares all immediate
keys, every handoff block and count, BF16 score bits, following reindex output,
graph visibility changes and late cache mutations. There is no serving dispatch.

## Independent top-k fix

The scratch-read fix and its reproducer are owned by
`experiments/2026-11-04-topk-live-reads/` in [PR #878](https://github.com/christopherowen/spark-ds41f/pull/878).
This search series references that patch and tested executable directly.
Historical receipts retain the old paths and exact commits at which they ran.
The attention experiment is stacked on that independent fix for review.

## Reproduce the isolated sp8 screens

`sp8.json` pins spark-0016's existing image separately from dgx4. Every
`sp8-*` bundle pins that exact ID and mounts the recorded source files. The
published clean detached checkout is
`/home/chris/spark-lab/checkouts/clustered-indexer`.

The runner uses the existing lab bundle construction and verdicts, a native
host reservation and GPU lock, a 16 GiB container cap, a 24 GiB host memory floor,
thermal observations and a heartbeat. It stops only its own named container,
yields between jobs when another task requests the host, and refuses existing
result paths. All completed jobs return the reservation. Exact reproduction
commands are retained in each native `command.json` and `verdict.json`.

```sh
python3 experiments/2026-11-04-clustered-indexer/sp8_run.py sp8-source:source-a sp8-source:source-b sp8-source-initcheck:source-init sp8-source-memcheck:source-mem sp8-source-racecheck:source-race
python3 experiments/2026-11-04-clustered-indexer/sp8_run.py sp8-grouped:group-a sp8-grouped:group-b sp8-grouped-initcheck:group-init sp8-grouped-memcheck:group-mem sp8-grouped-racecheck:group-race
```

## Separate screened variables

- `resource-layout.json` and patch 0003: stream the same visible keys with only
  two endpoints per coordinate instead of two 64-by-128 register fragments.
  Endpoints are bitwise equal to the frozen unreleased prototype.
- `matrix-bounds.json` and patch 0004: BF16 tensor products with FP32 accumulation
  compute `q+ @ upper + q- @ lower` and the absolute norm. Decoded certified FP4
  values are exactly representable in BF16. Exceptional metadata is tagged and
  never enters a product. The split sum has at most 256 terms; `2*gamma(255)`
  still leaves more than 100 times slack inside the existing `A/256` allowance.
  Scalar bounds remain the original API default.
- `grouped-metadata.json` and patch 0005: an explicit API for one cache owner and
  1–256 queries. Fresh metadata through the maximum extent bounds all shorter
  rows. Reduction and safe per-row table expansion are included in timing.
  Arbitrary mixed-owner requests are not accepted. The grouped API is separate
  from source handoff and neither is registered in serving.
- `source-handoff.json` and patch 0006: jointly preserve native immediate keys
  and source-block output; never qualify a source layer using keys alone.

Real model capture and replay are described in `capture/README.md`. Timing a
compact copy can change cache behavior, so the diagnostic now records and
reconstructs native physical page IDs, byte stride, pointer alignment, planned
width and active limit. A synthetic capture/layout initialization gate must pass
before collecting real operands. Raw activations and prompts stay outside Git.

## Prepared source tie cutoff

The source cutoff now also retains the worst seeded ordinary winner's original
block position. On equal native BF16 ordering keys, a cluster entirely after
that position cannot contain a selected ordinary block. Stable compaction can
remove it while always retaining the global mandatory last cluster. Native
ordering keys distinguish signed zeros; short and nonfinite seeds disable this
refinement. Original bound values remain intact for the full-score oracle.

`source-ties.json` and patch 0007 record this single variable. Twelve CPU tests
and fresh patch/source checks pass; repeated source, handoff/reindex and
sanitizer GPU gates remain pending. `sp8-source-control` preserves the preceding
unreleased source plan with exactly the same expanded harness. It does not
benchmark a released build.
