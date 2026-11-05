# Copy preparation stream simplification

Drop the optional copy-preparation overlap machinery after one focused rework
review found no defensible correction with evidence of a serving gain above the
recorded noise floor. C41 is declared a simplification **before any image, GPU
gate or serving measurement**. Its mark is correctness and no material core
performance regression; no speedup is claimed or required for this purpose.
This is not a measured second failure and does not change C39/C40's verdicts.

The owner's requested default enablement, safe decode/capture envelope and
PR878 scratch-read fix remain. The fused match/allocation/application kernels,
integer/float acceptance-state policy and older asynchronous allocation delivery
also remain. There is no new setting, fallback mode or tuning policy.

## Review and withdrawn correction

[C39's complete decision](../2026-11-04-copy-default-screening/decision.md)
found no proven serving-throughput gain over either saved C38 recipe after both
candidate and reference repeat floors. Its engine grid cost 0.86–1.32% more than
the standard C38 recipe; retain that adverse result. The grid invokes target and
DSpark dummy forwards without `CopyDrafter.prepare_alloc` or `propose`, so it
cannot attribute that cost to overlap or establish a benefit from removing it.
[C40's decision](../2026-11-04-topk-live-qualification/decision.md) retains the
scratch-read correction with neutral aggregate serving throughput, not as an
end-to-end performance improvement. C40's 0.30–0.48% smaller grid cost likewise
does not measure overlap.

This focused review initially considered passing the existing preparation-ready
event into `AsyncOutput.attach_copy_draft_alloc`, avoiding its second
`copy_stream.wait_stream(main_stream)` readiness event. That proposal is
**withdrawn before measurement**. Preparation is immediately followed by attach;
DSpark is launched only after attach. The second wait therefore snapshots
essentially the same model-stream point as the first; it does not accidentally
wait for substantial subsequent DSpark work. Removing an event record is not
evidence of a gain above the approximately 0.3% paired serving floors. There is
no recorded per-event/postprocess timing with which to bound such a gain, and
no duration is invented here. No event-reuse patch was accepted or GPU-tested.

The earlier [minimum-match-8 review](../2026-11-04-copy-policy-review/README.md)
is preserved separately. It dropped an unsupported policy hypothesis; it was
not this focused overlap rework. Its adverse fixed-match simulations are not a
serving-throughput baseline and do not justify adding a search/reduction tier.

## One source variable

The standalone [follow-on patch](vllm/0001-dspark-simplify-copy-preparation-stream-ordering.patch)
removes only the new side-stream preparation scheduling: `_prepared`,
`_prepared_waiter`, `_join`, the optional preparation stream arguments and the
runner's two stream keywords. The existing fused `_prepare` runs on the current
model stream, before DSpark. The existing asynchronous allocation copy is still
attached before DSpark, and `propose` follows DSpark on the model stream. FIFO
ordering replaces the removed readiness event and joins.

The production delta is 13 insertions and 41 deletions in two files; the third
file updates/adds regression tests. The GPU test now compares preparation and
proposals on the default current stream versus a deliberately delayed alternate
current stream. The new delayed-copy test exercises actual `AsyncOutput` and
`AsyncOutputFuture`, two in-flight credits and twelve steps across four full
turns of the three-slot allocation ring, with alternating allocation rows.

`tilelang_copy.py`, `async_utils.py`, `CpuGpuBuffer`, `SpeculativeConfig`,
`VllmConfig`, the engine queue, executor delivery and fixed top-k module are
byte-identical to C40. The fused `_prepare` and allocation attach methods are
AST-identical. This is a scheduling simplification, not a kernel split or an
integer/float-state policy change.

## Buffer and event lifetime

The lifetime argument is scoped to the frozen **PP=1, V2 asynchronous** profile;
tensor-parallel rank count is separate. `VllmConfig.max_concurrent_batches`
allows `pp_size + 1`, hence two outputs. The engine enqueues while credit is
available, then pops its oldest future and waits for `future.result()` before
another queue cycle. In the multiprocessing executor, `enqueue_output` calls
`AsyncOutput.get_output()` before publishing the successful response. The RoCE
wrapper first completes its inner output. The uniprocess future also calls
`get_output()` before resolving.

`get_output()` synchronizes the allocation's per-slot completion event before
reading its pinned NumPy view into a host-owned list. A three-slot allocation
buffer is reused only on the fourth prepare; the two-credit bound has already
settled its earlier delivery before then. Re-recording its older per-slot event
cannot relabel an outstanding earlier output under that bound. These older
completion events and the required model-to-copy stream wait are unchanged.
The removed preparation-ready event is no longer recorded or reused at all.

The regression test follows that actual credit rule, rather than testing an
unbounded producer or adding synchronization not present in the runner. It
checks distinct slots/events, delayed D2H data and delivered-list independence
after wrap. Source inspection and host doubles do not prove CUDA execution;
the authored GPU regressions still require the new image's gate. No claim is
made for arbitrary pipeline-parallel configurations, and no broader policy
guard is introduced by this patch.

## Source identity and validation

The authority is C40's published measured evidence `0afedcb0`, its frozen
execution commit `1f134c7aad82bc63c95afed36f488becf2959a41`, and its tracked
series/manifest. C40 vLLM head is
`5f926df6d3126266a9c88c82c06544060008c33c`, tree
`ca1496c6481dc5bc4b4234b743e4ca023a6c7597`, image
`sha256:7ae96e98e361601d8232cd99e6af1c6044b7c282bf6904c989e6ad033c147f82`.
A moving deployment HEAD or disposable cache is not source authority.

The local 574-patch canonical replay from pinned upstream
`19f2c20ed4d64ae22bd009ceacadd85e2a81e354` exactly reproduced that C40 head/tree
before edits on branch `feat/copy-overlap-readiness-review`. The follow-on source
head is `38ade7f947cb09c550c2bce077286e467c7fed8d`, tree
`df6b6bc6841c1126b37be652c0e93c13debb8f42`; patch SHA256 is
`8c81f9944c86641fedc8d1da6a20aeeb67de64f24128a5089d139e40cede621e`.
[Patch receipt](runs/source-review/patch-receipt.json) and
[full follow-on replay receipt](runs/source-review/follow-on-replay.json)
preserve all commands, patch hashes and exact identities. Prepared trees are
disposable; the follow-on patch and future published manifest/series are the
durable inputs.

Five [source-only controls](runs/source-review/test_source_controls.py), Python
3.12 parsing of the three changed source/test files, whitespace checks and both
C40 offline doctors passed. The first source-control normalizer missed an
annotated field assignment; its failed result is
[retained](runs/source-review/source-controls-initial-failure.json), the helper
was corrected and the subsequent controls passed. No production correction was
needed for that test-helper failure. [Validation receipts](runs/source-review/source-validation.json)
distinguish these local checks from pending qualification. All replay/check logs
and read-only inspection errors are retained. No deployment commit/push, SSH,
node change, image build, GPU work or remeasurement occurred in this review.

Before use, the new published profile must pass complete Linux preparation and
build checks, actual-image/source/release gates, every native functional outcome,
both complete performance boots and independent restoration verification under
the normal shared protocol. These have not been independently run here. The
source-only proof does not replace sanitizer, release or TP4 correctness gates.

## Acceptance and drop rule

Use both saved C40 boots as the primary reference, preserving all four pairings,
candidate/reference repeat floors and uncertainty. Show immutable 1030, saved
C36, previous and current candidate with signed over/under and explicit verdicts.
Keep throughput, per-user decode, TTFT/TPOT and recorded ITL distinct; include
cold/mixed prefill, first-chunk waits, engine cost, acceptance, memory/swap,
startup and thermals, including every adverse or missing row. Grid gains alone
cannot establish serving gains, and no compatible targeted throughput baseline
is created by a new correctness observation.

Accept this declared simplification only if all correctness/identity gates pass
and no material core regression survives the paired and repeat floors. Highlight
and reject the simplification if it materially regresses; do not invent another
speed retry or call a failed enhancement cleanup after results. Any future
optional performance patch must first have a defensible causal mechanism, then
earn a relevant serving gain beyond both floors without material adverse core
rows. A measured miss gets at most one justified correction; a second miss is
highlighted and dropped. Default behavior and necessary safety fixes are judged
on their stated correctness/regression criteria. No promotion is authorized by
this review.
