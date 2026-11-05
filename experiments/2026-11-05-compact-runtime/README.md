# Compact page-owned coarse scoring

Base `6ace6083`. This experiment tests the interaction of a fused block8-mean
MXFP4 encoder, compact page storage and the native blockscaled coarse scorer.
The model, dense/source budgets64/512, ordered candidate assembly, retained
native fine arithmetic and source handoff remain unchanged. The separate
coarse feasibility screen motivates this integration but does not qualify it.
No promoted recipe or immutable baseline changes.

Each physical page reserves 80 bytes/key: the native 68-byte/key prefix, an
8.5-byte/key packed mean suffix and padding. Completed pages produce BF16 block
means and immediately encode them with the native MXFP4 scale/rounding rule in
one kernel. Partial pages are handled by native fine selection. No temporary
mean pool, query expansion, reference pass, grading or host copy enters serving.
All scratch uses the existing bounded workspace; the previous 2 MiB decoded
query workspace disappears. The scorer derives complete-page extents and coarse
counts in its own launch, avoiding a metadata conversion kernel.

Summary quantization can change selected keys. A scorer speedup is insufficient:
all independent answer-quality, request-speed and memory/admission gates remain
mandatory before adoption. Model outputs and operands remain private on nodes.
This is not a claim of identical final answers or general quality superiority.

First qualify all 12 frozen captured batches against an independent decoded
quantized-mean score oracle, exact native-encoder bytes, retained fine-score bits,
ordered candidates and exact covered-domain outputs. Next run strict
uninitialized-suffix sanitizer checks and six mixed-owner/cache/recycling/graph
cases, including page128/256, 257 rows and a planned 1M source width. If these
pass, measure two fresh processes of complete selectors against Native68, plus
required summary construction separately. Original physical page IDs and row
counts1/8/64 remain fixed. Component timing is not request timing. Preserve every
command, error, sample, allocator observation and lease release in `runs/`.

The serving profile is experimental and must not start before those gates pass.
Apply the ordered seven predecessor patches and then `vllm/series` against C36
vLLM `6d4e49d1cdbb6513819e47145e0e52fdab327b6b`; the overlays must equal that
fresh prepared tree. Use only clean published commits for GPU work, hold the
sp8 native lease and GPU lock, and leave any serving group untouched during
these isolated component tests.
