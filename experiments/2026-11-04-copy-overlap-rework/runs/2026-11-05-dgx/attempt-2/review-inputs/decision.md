# Decision before qualification

Drop C39's optional copy-preparation stream/event overlap after focused rework
review. The review first considered reusing the existing readiness event, but
found no defensible serving-gain bound: the redundant readiness event is recorded
before DSpark is launched, so removing it does not remove a substantial DSpark
serialization. No event cost or postprocess trace in the saved evidence supports
an above-floor gain. That proposal is withdrawn before any new image or GPU run;
it is not a measured corrected failure.

C41 is explicitly a cleanup/simplification candidate before measurement. It
prepares copy matches/allocation on the current model stream before DSpark and
removes the optional side-stream readiness event, waiter and join machinery.
Keep default CopyDrafter behavior, the safe budget/capture envelope, fused copy
selection kernels and float-state policy, older asynchronous allocation delivery,
and C40's scratch-read guard. No compiler or hardware variable is included.

The simpler path must pass its own source/image/release/initialization gates,
all functional outcomes and both fresh metric boots. Correctness and no material
incremental performance regression versus BOTH saved C40 boots are required.
Neutral performance is acceptable for this declared simplification; no speedup
is claimed. A material adverse result prevents accepting the simplification and
must be reported, not disguised as cleanup success.

Show 1030, C36, previous C40 and current C41, with explicit over/under and
better/worse/no-proven-change verdicts. Compare both C41 boots with both C40,
both C39, both saved C38 standard/copy boots and immutable release references,
retaining all pairings, repeat floors and adverse rows. C36 has one compatible
saved boot; no repeat is fabricated. No old image or release is remeasured.

The saved standard matrix is a compatible timing reference; the separate copy
probes establish correctness/activity, not a budget-binding copy-rich throughput
baseline. Do not infer performance equivalence for unrecorded workloads. Existing
references also lack actual ITL tails, timed KV occupancy and raw cache hit/miss
counts; TPOT must remain distinct from ITL.

Source review and canonical preparation completed before qualification. Attempt
1 at frozen execution commit `0da35cc9668ff491065446b2a842fe996a73e80f` passed all
eight native image gates, including 8,075 tests and strict actual-source top-k
initialization checking. It then safely yielded for the shared queue before any
serving work. Preserve its scheduling error and actual exit 1; this is not a
measured candidate failure. Independent all-rank C36 restoration checks passed.
The unchanged image passed full local gate-evidence review and fresh shared
capacity/source checks, then resumed as the sole attempt-2 controller at
07:42:15 UTC on 2026-11-05. It uses the original qualified image and execution
commit; no build, gate or serving point has been repeated. No C41 serving
performance verdict is available. See [qualification.md](qualification.md) and
the retained original receipts for execution state.

Shared reservations, clean published same-commit checkouts, named plans, guards
and independently verified C36 restoration remain mandatory. No release
promotion or bypass of TP4 gates is implied. The recurring schedule remains
paused at the owner's request.
