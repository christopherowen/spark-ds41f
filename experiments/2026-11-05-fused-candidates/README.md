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

The first screen at `3530fddb` passes all39 adversarial candidate/count fixtures
under both initialization and shared-memory race sanitizers, all12 captured
coarse/fine checks, four strict initialization fixtures and six graph/cache/
recycling cases. Two fresh processes retain source Native68/candidate ratios
1.175x,1.779x and2.064x for rows1/8/64, with conditional joint95 intervals
[1.066,1.306], [1.633,1.929] and[1.943,2.198]. These assume normal process
medians with only two samples; no request-speed or model-quality claim follows.
Dense single-row ratios remain0.893–0.894x and are still slower than native;
larger dense batches retain a speed signal. All samples, required maintenance
and memory observations remain in `runs/screen-01`. The lease releases
at20:50:01 UTC, and an independent reader verifies no holds, containers or
compute processes remain.

The package integration replaces construction of the discarded coarse selector
and separate assembler. It omits unused final selected-index/value admission,
retains necessary intermediate merge buffers, and declares precisely its actual
coarse kernels. The screened kernel body remains unchanged. `vllm/series` adds
one patch after the preceding nine; fresh application from the pinned C36 base
equals both runtime overlays. The new candidate profiles are unpromoted and
must requalify frozen-call binding, the same correctness/sanitizer/lifecycle
fixtures and two fresh timing processes before any model evaluation.

The exact integrated source `3ad74ef5` passes all62 declared calls across eight
configurations after cache freeze, all39 adversarial cases under both
sanitizers, all12 actual batches, four strict initialization fixtures and six
lifecycle cases. Two fresh processes retain source ratios1.174x/1.783x/2.065x
at rows1/8/64; the conditional joint95 intervals are[1.102,1.257],
[1.736,1.832] and[1.905,2.227]. Dense single-query components remain slower.
All samples, maintenance and memory observations are retained in
`runs/runtime-gates-01`; the same limitations apply. The lease releases
at20:56:32 UTC and independent verification finds no reservations, containers or
compute processes. No request-speed or model-quality result is attributed to
this integration, and no promotion follows.
