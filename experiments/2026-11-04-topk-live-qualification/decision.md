# Decision

Candidate-40 passed its distinct image/release/strict initialization gate.
All eight steps passed; 136 per-file receipts and 14 XML reports agree on
8,074 passed, 360 skipped and zero failures/errors. The actual-image top-k
checker reports zero initialization errors and exact expected output, with
native source-path/byte-hash verification and no source overlay.

This is correctness/build evidence. Both functional boots and both metric
boots remain pending; no speedup or slowdown is measured for candidate-40.
Preserve candidate-39 and complete this image's serving qualification before
selecting a default-enabled tuning base.

The foreign lossless-memory window released. Our matching request was
fulfilled, then a separate guarded source-staging window verified clean exact
execution commit `1f134c7aad82bc63c95afed36f488becf2959a41` on all four checkouts
and closed without restarting the verified C36 service. The sole qualification
controller started at 15:34:15 UTC and opened window `dgx1-1793806495`. It
yielded safely for the next shared request at 16:22:13 UTC, after passed gates
and before any seeding/functional/timed phase. Preserve the original scheduled
closure error and actual exit 1; the child image/gate exit is 0. This is shared
scheduling history, not a measured candidate failure.

Independent live doctor and all-rank Docker image/profile checks confirm C36
restoration. The actual DGX4 image matches the completed immutable gate, and
local explicit resume review passes. Resume under a new attempt after the
foreign request/window completes, without rebuilding or repeating any gate.
The owner paused recurring checks and requested direct waiting in this chat;
do not restart the recurring schedule.
