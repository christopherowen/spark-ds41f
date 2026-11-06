# PR925 source-selection clarification

This addendum applies to evidence head
`86d96454212ad12a343ad45f5e828f9b063561d6`. Original archived analyzers,
results, method hashes, errors and adverse comparisons remain unchanged.

C36 selects the compact narrow/wide expert-scale representation already included
through rc20/0595. The isolated historical-base experiment therefore does not
establish an incremental scale-storage or serving improvement over C36.

Packed Engram reader support also exists in C36's source, but support is not
selection. The pinned source audit shows `open_disk_rows` registering the original
separate value and scale sources through `add_source`; it does not select the
packed row representation. That audit is source evidence, not a new live
file-descriptor attestation. The separate packed Engram experiment remains
unmeasured by this expert-scale cohort.

Accordingly, generic phrases such as “C36 contains both storage ideas” in the
preserved analyzer/reference descriptor must not be read as evidence that both
storage layouts were active. Expert-scale compaction is selected; packed Engram
support is present while separate Engram planes are selected. This note clarifies
that distinction without rewriting the historical reproduction inputs.

The memory/storage result remains the reported framework model-load CUDA
allocation counter: 92.62–92.63 GiB per rank versus the saved TP3 1030 scalar
93.95 GiB, about 1.32–1.33 GiB lower. The unchanged counter implementation does
not turn this into whole-host RSS, a reference allocation confidence interval or
demonstrated extra KV capacity.

Keep the absence of a repeatable serving gain and the unfavorable C8 views.
Keep both candidate/reference repeat margins, every adverse row, the missing
eligible 524K/1M prefill repeats and the actual-token/condition warnings. C36's
different streaming collector and single saved boot allow only independently
compatible engine/prefill context, not an incremental client-throughput or
latency claim. Do not substitute TPOT for unrecorded ITL tails or invent KV/cache
counts.

Archive integrity and offline performance reproduction are separate from model
report revalidation and release acceptance. The three model reports remain
outside the archive and require their own safe node-local verification; this
addendum grants no serving access, measurement rerun or promotion.

## Subsequent integrity verification, 2026-11-06 03:35 UTC

All three private model reports now independently match their published sizes
and SHA256 identities at the original DGX1 paths. The recovery snapshot paths
were absent; the first failed path-only attempt is retained. The second bounded
hash job read the original files without decoding, copying their text, sending
model requests or repeating a measurement. It used one CPU,256MiB memory and
zero swap; actual wait exit0,8MiB peak memory,0 swap. Its matching CPU claim was
removed and both shared CPU locks were independently available afterward.

This closes the private-byte integrity gap, not an independent repetition of
the model comparison or native gates. The archive and saved performance analysis
were independently checked as recorded above. PR925 was merged at
`48c6fa1538a8afd261247b249ee9180baece8340` as evidence only; the runtime, release
and all original limitations are unchanged.
