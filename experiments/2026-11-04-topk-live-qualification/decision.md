# Decision

Retain C40 as the validated corrected default-enabled base. This round fixes an
invalid top-k scratch-memory read; neutral aggregate serving throughput is an
acceptable result for that correctness purpose. It is not accepted as an
end-to-end performance enhancement and is not promoted to a release.

The distinction is explicit: C39 delivered requested default enablement, but
its concurrent copy-work speed hypothesis did not establish a serving gain.
Do not retroactively call that hypothesis cleanup. The separate compiler
performance hypothesis must demonstrate a repeatable relevant gain beyond
recorded confidence and repeat floors. Neutrality alone would not satisfy it.

C40's eight image/release/actual-source initialization gates passed: 136 per-file
receipts and 14 XML reports agree on 8,074 passed, 360 skipped, zero failures or
errors. The actual-image top-k check verified native source path/byte hash, exact
tie-ordered output and zero initialization errors, without a source overlay.

Both functional boots passed all 40 ordered outcomes, including both original
runtime collectors. All six performance suites passed all 26 ordered steps.
The local audit verified all 176 native artifact hashes and source/image/profile
identity, distinct boots, stable containers and complete fault-free logs.
All four C39/C40 model-corpus pairings have identical tokens/probabilities and
KL 0; corpus digests match both saved C38 recipes as well. This does not establish
general task quality, full-token copy invariance or outstanding TP4 correctness.

C40 output tokens/s were 65.38/65.37 at one stream, 126.10/126.17 at four and
163.41/163.59 at eight. C8 versus C39 spans -0.24% to +0.16% across all four boot
pairings. There is no proven incremental serving-throughput gain over C39 or
either saved C38 recipe after both candidate and reference repeat floors.
The fixed-row engine grid is 0.30–0.48% cheaper than C39, a small component gain
that must not be presented as a serving-speed improvement. C8's +3.03–3.15%
versus immutable 1030 includes inherited changes.

Keep adverse rows visible. C8 TTFT p99 is 0.4327/0.4332 seconds versus C39's
approximately 0.414–0.417 seconds; no percentile confidence interval was recorded.
Mixed 131K prefill beside three decoders is 1.64% below the single compatible C36
reference in the headline comparison, and cold 524K prefill is 0.46% below it.
The first C40 boot's first-chunk decode wait is 25.42% above 1030. The full
pairings preserve variation and warnings; no claim that every metric improved
or remained neutral is justified. C36 lacks a saved repeat in this protocol.

Timed minimum available memory was 6.34 GiB, maximum GPU 77 C, minimum thermal
headroom 10 C and maximum system 90 C, with zero swap growth, thermal slowdown
or power-cap time. Functional API readiness was 112.7/104.3 seconds; performance
boots were 104.2/101.2 seconds. Timed KV occupancy, raw cache hit/miss counts and
ITL tails are absent. Do not call TPOT ITL or zero observed cache hit rate a
cache-rich workload qualification. Preserve prompt-length/duration warnings.

Attempt 1 safely yielded after its completed image gates and before any serving
phase, preserving actual exit 1 and the scheduling error. Attempt 2 reused that
unchanged image/gate, with no rebuild, gate repeat or measured-point rerun. Its
actual supervisor wait exit was 0 at 20:28:35 UTC, with restoration true. The
independent named live doctor and all-rank Docker image/profile checks verified
C36 at 20:28:46 UTC before the natural shared-capacity notice. Original native
receipts, logs and hashes remain intact. Full generated model text and raw logs
remain private; portable timing/conditions/identity JSON and every adverse row
are published separately.

The minimum-match-8 hypothesis was dropped at review before GPU testing, not
counted as a measured second failure. An optional optimization miss receives
one review and at most one justified corrected attempt; another measured miss
is highlighted and dropped. Required correctness fixes are judged on their
stated correctness and regression criteria. No speculative patch is required
merely to consume an attempt. The separate compiler round must preserve one
causal variable, complete image/quality gates and both metric boots.

The recurring schedule remains paused at the owner's request. Any further work
uses ordinary shared reservations and direct waiting. Outstanding TP4 release
gates and hardware findings remain; no limits, topology or hardware policy is
changed to pass them.
