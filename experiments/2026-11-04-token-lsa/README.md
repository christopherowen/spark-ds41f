# Token-equivalent shortlist performance experiment

Base: `0d389a0552db4cfab4d833af6ea6290488c7c1ab`. The owner revised fidelity to
generated token IDs and required correctness/proof work outside the timed path;
see [the acceptance contract](../2026-11-04-clustered-indexer/acceptance.md).

Variable: replace exclusion-proof search with a centroid shortlist. Balanced
semantic buckets contain eight native eight-key blocks. Complete512-key chunks
are indexed; the partial visible chunk and final block are always retained.
Queries score BF16 bucket centroids, shortlist a fixed budget, compact original
blocks in order, and score those keys once with native MXFP4. The source layer
selects its2048 blocks from the shortlist and publishes legal original block IDs.
Selected keys and numerical intermediates may differ.

`ShortlistPlan.run` has no native reference, exclusion bounds, equality checks,
host copies or metadata construction. Validation is a separate process. Index
construction is essential work and is reported separately, with amortization
across queries; no request speedup may omit construction/update cost.

The first phase explicitly uses one immutable snapshot owner. It does not yet
implement serving cache writes, invalidation, admission/workspace integration or
full-model token-ID capture. Captured inputs are the twelve existing native C36
operands. Raw tensors and fitted metadata stay node-local. Native C36 is an
unreleased kernel control and may be measured in separate processes; no released
build is retimed. At least two fresh candidate/control processes are required.

Do not promote a shortlist based on kernel speed or key recall. It must also pass
the generated-token gate on predeclared full-model workloads, plus complete
request performance and memory/admission checks. Historical response-digest
receipts do not satisfy raw token-ID equivalence for this changed-selector path.
