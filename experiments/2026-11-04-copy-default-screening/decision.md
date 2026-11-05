# Decision

Retain CopyDrafter default enablement as source behavior, with no proven serving
throughput gain over either saved candidate-38 recipe. C39 measured 163.34/163.80
output tokens/s at eight streams, 125.91/126.04 at four and 65.35/65.34 at one.
All reference boot pairings, intervals, A/A floors and unfavorable rows remain
in the evidence. Small C4/C1 losses are not robust across both reference floors;
the engine grid has a small consistent cost against the standard recipe.

Both fresh performance boots completed all three suites with no request, foreign
traffic, memory, swap or thermal failures. Model corpus digests match all saved
C38 boots. The 40 functional script exits include two preserved stale-ID collector
failures; explicit offline revalidation of the original data passed without a
benchmark repeat or native-verdict overwrite. Original runtime failures remain
visible. The current source's independent inactive scratch-read issue (#878)
still prevents release qualification. No promotion is made.

Next isolate the #878 liveness fix as a corrected default-enabled candidate, with
its own clean source, image/gates and repeat core metrics. Evaluate #875's
compiler work as a separate round when its prerequisites allow. Fix the
collector's expected-image input and ensure every quality
extra is audited before treating its parent phase exit as a pass. Do not change
C39 or silently replace its measurements.

The subsequent [copy-policy review](../2026-11-04-copy-policy-review/README.md)
dropped the optional minimum-match-8 hypothesis before GPU testing. Its saved
fixed-m simulations are adverse, not serving timings; the engine-grid path does
not exercise copy overlap, so removing overlap is not a supported fix for its
measured cost. No corrected source patch or speedup is asserted. Retain required
default behavior and the independent correctness fix; C39 is not accepted as a
performance improvement. Preserve all measured failures and adverse rows.

Purpose clarification: C39 is not merely cleanup. It combines the requested
default behavior with a copy-work overlap performance hypothesis. Neutrality is
acceptable for retaining the required default behavior, but the speed hypothesis
fell short. C40 subsequently qualified the separate scratch-read correctness fix
with neutral aggregate serving throughput; that is an acceptable correctness
result, not evidence of a C39 serving-speed win.

Before release, address mixed-prefill first-chunk waits and tail latency, reduce
copy-path engine overhead where useful, and capture missing timed KV occupancy,
cache hit/miss counts and ITL tails in future candidates. The current zero cache
hit rate is a property of these inputs, not a cache-rich workload qualification.

Candidate-36 restoration is independently verified on all ranks; the shared hold
closed normally. Respect the other task's subsequent capture window. Spaceport's
reported access recovery does not close its earlier thermal/no-content findings
or authorize our own new hardware operations. AgentX does not score useful task
completion or replace the outstanding TP4 correctness/release gates.

PR #850 closed as unnecessary; #842 merged. Keep the distinct safeguards, site
evidence, compiler, LSA and scratch-read work in #851, #865, #875, #877 and #878.
