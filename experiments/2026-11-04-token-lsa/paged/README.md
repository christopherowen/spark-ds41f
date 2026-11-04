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
