# Token-gated shortlist performance experiment

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

`token_gate.py capture` uses the pinned server's `return_token_ids` response
metadata in a separate, untimed request run. `compare` requires identical raw
IDs and termination under the same checkpoint/tokenizer/workload. Keep captures
node-local. It never retokenizes decoded text. Eight gate regression cases cover
changed IDs with identical text/count, missing IDs and mismatched termination or
workload. This client is never imported by the query module or benchmark.

The first two process cohorts are summarized in [results.md](results.md), with
raw samples, uncertainty and construction accounting in `results.json` and
`runs/`. Captured eight-repeated-query dense calls improve1.92–1.93x; captured
single-query and source calls regress. Replicated512K-key calls improve6.27–6.43x
at one query and12.42–12.59x at eight repeated dense queries (source4.06/7.51x).
These are query component results on the stated shapes, not request speedups.
Two additional validation processes pass the resident-cache centroid oracle
(maximum error0.0), native evaluated-score bits, visibility replay and legal
source handoff. Selected-key overlap is diagnostic, not token qualification.

`shortlist-window-01` retains the initial compile failure and its coordinated
restore. `shortlist-window-02` retains the successful eight-query cohort and both
validation modes. C37 was restored once at20:12:53UTC, independently checked on
all three nodes at20:17:53UTC; all native holds and requests were absent, both
checkout roots clean, and experimental mounts/environment removed.

The next [page-owned follow-up](paged/README.md) moves essential summary work to
cache completion and charges storage to cache admission. It is a separate
unqualified grouping geometry until its own lifecycle and generated-token gates.
