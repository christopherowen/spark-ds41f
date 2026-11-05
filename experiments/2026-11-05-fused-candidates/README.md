# Final selection directly constructs candidates

Base `4210b137`. Same compact80 summaries, native coarse scores, dense/source
budgets and native fine arithmetic. One intended variable: the final exact
coarse selector directly writes the fine candidate list and token count, rather
than emitting an intermediate selected-index/value pair and launching a separate
assembly kernel. Intermediate merge values remain necessary; unused final
coarse values are no longer stored. Existing exact direct source pair merging
is preserved.

The fused writer keeps sorted winners, every visible partial-page block and the
required last block. Covered domains emit the exact full visible block range.
A unique last-valid boundary determines winner count without a repeated binary
search or an atomic reduction. One block barrier separates initial writes from
tail appends. It retains the existing negative-infinity/NaN eligibility and tie
order; no new approximation or grading enters the runtime.

This is an out-of-band prototype. Before any model recipe, require adversarial
candidate/count equality against the preceding selector plus assembler, strict
initialization checks, all12 frozen actual batches, six graph/recycling cases
and two fresh complete-selector timing processes. Include covered/pruned, partial
pages, empty rows, nonfinite scores, ties and unwritten score/work suffixes. Keep
all samples/errors and memory observations, including required mean maintenance.
Raw operands remain private on sp8. No final-answer or request-speed claim is
authorized by a component result.
