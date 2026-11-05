# Measured decision: reject C41, retain C40

C41 passes the recorded correctness gates and has neutral aggregate serving
throughput, but fails its predeclared simplification criterion because unresolved
core regressions exceed the existing repeat-aware margins in the comparisons
below. Drop this simplification and retain C40 as the best validated corrected,
default-enabled source. Production remains independently verified C36. Nothing
is promoted, rebuilt or remeasured by this decision.

The purpose was declared before any image/GPU work: remove optional copy
preparation stream/event complexity while preserving correctness and avoiding
material incremental core/serving regressions. A speedup was not required.
C39's optional-overlap enhancement had no proven serving gain; C40's scratch-read
fix was accepted as correctness work with neutral serving performance. Keeping
C40 now does not establish a performance benefit from its overlap.

Both C41 boots completed all six suites and 26 ordered performance steps. All
40 ordered functional outcomes and eight image gates passed, including 8,075
tests, 360 skips, both changed GPU regressions and actual-source initialization
checking with zero errors. All 181 exported file hashes verify locally.

Against both saved C40 boots, all four paired throughput changes are:

| Metric | Signed change | Verdict |
|---|---:|---|
| C1 decode | −0.080% to +0.001% | No proven change |
| C4 decode | −0.066% to −0.025% | No proven change |
| C8 decode | −0.042% to +0.078% | No proven change |
| Engine-grid cost | +0.206% to +0.409% | Worse in 7/8 pair-and-floor views |
| Cold 524K prefill throughput | −0.181% to −0.667% | Worse in 4/8 pair-and-floor views |

The eight views use two candidate boots, two saved reference boots and the
unchanged candidate/reference floor calculations; they are not eight independent
experiments. Grid's smallest pairing is same under the saved-reference floor.
Cold prefill is same against C40's first boot and worse against its repeat.
No new tolerance is introduced after measurement. Aggregate decode neutrality
does not waive the predeclared engine/prefill criterion.

The focused completed-result review found no defensible new correction. Grid
bypasses the changed preparation calls; pure cold-prefill also skips that path
under this harness. Their image-level regression remains unexplained and must
not be attributed to overlap removal or discarded. Eligible real drafting can
expose serial preparation, but no saved phase timing bounds a useful correction.
Restoring concurrency returns to already-measured C40. Another speculative image
or a repeat of either measured image is unjustified. The earlier event-reuse
proposal was withdrawn unimplemented/unmeasured, not a second benchmark failure.

Retain additional adverse observations: minimum available host memory fell from
C40's 6.34 GiB to 5.97 GiB; some mixed-prefill, first-chunk and percentile rows
are worse. C41 C4 TTFT p99 on boot two is 3.60% above 1030, without a recorded
percentile interval. C36 has one compatible saved boot; C1 paired throughput
is 0.31–0.37% below it and some prefill rows remain worse. Every pairing against
both C39 and both saved C38 recipes is retained, without choosing a favorable
reference or treating inherited gains over 1030 as this patch's improvement.

No timed native guard halted. Maximum GPU temperature was 77 C, minimum thermal
headroom 10 C, maximum system temperature 88.8 C and recorded thermal slowdown,
power-cap duration and swap growth were zero. Separately, dgx3 reported seven
software power-cap event-mask samples during the repeat release suite. Preserve
that discrepancy rather than claiming absence of power-cap events or inventing
a cause/duration. The original operator observer SSH timeout is retained; native
all-rank telemetry covered the first release span and no controller/runtime reset
occurred. No native management-loss latch was recorded.

The exact image is
`sha256:38ff08bb1eb2495cb934e008b6f31a482aaa238c08b959aaf72cf2cb5658d29c`,
execution commit `0da35cc9668ff491065446b2a842fe996a73e80f`. Attempt 1 passed
the image gate and safely yielded before serving; its original exit 1 and
`RuntimeError: window changed or closure requested` remain scheduling history.
Attempt 2 reused that immutable gate, ended with actual wait exit 0 at 11:21:04
UTC on 2026-11-05, and restored C36 once. Independent doctor and all-rank
image/profile/container checks passed at 11:21:53 UTC with both processes absent.

See the [round report](runs/2026-11-05-dgx/attempt-2/README.md),
[focused review](runs/2026-11-05-dgx/attempt-2/result-review.json),
[core metrics](runs/2026-11-05-dgx/attempt-2/core-metrics.json),
[all comparisons](runs/2026-11-05-dgx/attempt-2/recipe-comparison.json) and
[telemetry review](runs/2026-11-05-dgx/attempt-2/telemetry-review.json).
The hash-bound root README remains the source-review snapshot; current measured
state is recorded here, in qualification.md/results.json and the round report.

The recorded model corpus matches both saved C38 recipes. All four full model
pairings against each of C39 and C40 have 256/256 identical sequences, 65,536
identical score positions and zero mean/maximum KL. These are scoped recorded
quality checks, not general useful-task or TP4 qualification. Timed exports lack
KV occupancy, raw cache hit/miss counts and ITL-tail distributions; TPOT is not
ITL. The 524K/1M release-reference prompt/identity/conditions limits and stale
outer prefill duration remain explicit. No baseline is rewritten.

Continue only distinct justified work from C40 after reviewing compiler
prerequisites, actual shared scheduling and clean published identity. No
promotion without owner acceptance and outstanding TP4 correctness/release
gates. The recurring schedule remains paused at the owner's request.
