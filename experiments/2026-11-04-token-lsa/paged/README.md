# Cache-write page shortlist

Follow-up variable: replace snapshot-only metadata with summaries owned by each
physical native cache page. A completed page partitions its native eight-key
blocks into balanced64-key buckets. Four cosine split iterations per level run
only when a cache write includes the final slot of that page. Every partial
visible page is scored directly. Rewrites after rejected speculation refresh the
summary on completion; reused pages cannot publish summaries from an old owner.
No request identity is inferred from a query tensor or a shared page table.

The cache specification reserves73 bytes per indexed key instead of68, including
BF16 centers, block membership and alignment. Metadata lives beside the native
records in the same physical allocation, follows prefix-cache reuse, and counts
toward memory admission. The native record layout remains the first68 bytes per
key. No CPU activation copies or correctness checks belong in these kernels.

This is a new grouping geometry, not an equivalence claim for the snapshot
clusterer. Separate tests must cover page completion, partial visibility, mixed
owners, physical-page reuse, graph replay and source/reindex handoff. A full-model
raw-token gate and complete request measurements remain required. No promotion.

The first two fresh lifecycle processes pass all eight cases each:128/256-key
pages,1/2/9 rows, mixed owners, partial-page boundaries, native key-record bytes,
balanced summary references, evaluated-score bit equality, graph replay and
physical-page recycling. Runtime commit`bf8d5eb3`; raw receipts are retained in
`../runs/paged-window-01`. These are kernel and lifecycle results, not a token gate.

`control.json` and `candidate.json` keep the same admitted73-byte cache geometry,
native scratch-read guard and C36 image. The control runs native queries and no
summary maintenance; the candidate pays for both cache-write summaries and
shortlist queries. Reindex and small planned domains stay native. All scratch
is reserved before profiling and every kernel declared before the cache freezes.

`run_serving.py` uses a clean published source commit and the existing exclusive
three-node idle-window protocol. The predeclared order is control-A, shortlist-A,
shortlist-B, control-B. Every arm performs a separate raw-token capture: six short
tasks and synthetic notebook contexts of32768/131072/262144 tokens, greedy,
seed0, up to128 output tokens. Tokenization and private capture occur only on the
node. Any mismatch stops the screen. No benchmark requests ask for token IDs.

Request timing covers131072 tokens at concurrency1/4 and262144 at concurrency1.
Each point charges its first cold request to the measured boot, followed by three
explicit same-prompt cache repeats. Both arms must report actual cache-hit usage,
output counts and response digests; no cold request is relabeled as cached. This
small synthetic screen does not qualify production-agent quality or throughput.

The first serving trial is **rejected** by the raw-token gate. All six short
requests match, while the32768/131072/262144-token notebooks first differ at
zero-based generated-token indices15/9/73. Length128 and termination match but
are insufficient. No quality superiority has been established. The controller
stops before candidate performance measurement or further boots, preserves the
native control measurements, and restores C37 at20:48:50UTC. Independent
three-node restoration/clean-checkout/no-reservation evidence is retained.
See `../runs/paged-serving-01/token-divergence.json`. No request-speedup claim.

The direct candidate-construction follow-up preserves the existing selector's
ordered keys/counts/padding. It visits only selected pages and replaces three
full-context scan/prefix/scatter launches with one page-local compaction launch.
It does not repair the failed selector's token fidelity. Separate candidate-list,
graph, owner-remap and native-lifecycle gates precede any timing or serving use.
