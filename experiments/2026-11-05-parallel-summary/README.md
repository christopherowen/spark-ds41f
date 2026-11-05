# Cooperative summary construction

Base `3ad74ef5`. One intended variable: parallelize dimensions while constructing
the same eight-key means and MXFP4 encoding. Preserve sequential accumulation of
the eight keys, BF16 rounding, native scale floor/rounding and E2M1 encoding.
Coalesced loads and cooperative maximum reduction replace serial per-thread
32-dimension loops. No summary shape, score, candidate or quality policy changes.

This is out-of-band. Require full cache-byte equality on all four private frozen
captures, both page sizes, wide public synthetic quantized scales, zero/tie/sign
vectors, skipped/outside slots and recycled pages. Require zero initialization
and shared-memory race errors. If these pass, record two fresh maintenance timing
processes, including single-page, all-page and skipped-slot costs and memory.
The published preceding summary timings are comparison evidence; do not rewrite
them. Keep every error. No request or answer-quality claim follows. Any runtime
integration requires the full package/kernel/lifecycle and admission gates.

The first screen (`0a5570c2`, `runs/screen-01`) fails before GPU execution:
the maintenance-only bundle inherited the timing mounts and omitted the host
sanitizer directory. `execv` cannot find its tool. No equivalence or performance
result is produced. The error, exact command and lease release are retained;
there are no remaining containers. The retry adds the same pinned sanitizer
mount used by the prior qualified bundles and disables the PyTorch allocation
cache for strict checking. The kernel, operands, expected bytes and timers are
unchanged.

The corrected128-thread screen at `396a431f` passes full cache-byte equality
on all four captures, both synthetic page sizes, unwritten skipped pages and
recycling, with zero initialization/race errors. Two fresh processes measure
dense single-page maintenance28.2–30.1 microseconds versus the preceding
approximately52–53; source single-page54.7–55.2 is slightly worse. Full snapshots
measure36.2–37.7 dense and131.9–132.2 source microseconds. Every timing, skip-only
cost and memory observation is retained in `runs/screen-02`. The lease releases
at21:01:44 UTC; an independent reader verifies no reservations or GPU processes.
No runtime adoption follows from a mixed result.

The next screen changes only threads per cooperative page CTA,128 to512.
The arithmetic, operands and expected bytes remain identical. Require the same
sanitizer/equality checks and two new timing processes before considering it.
This tests whether more lanes reduce per-thread serial dimension work; retain
both favorable and unfavorable page/batch results.

The512-thread screen at `727595b8` passes the same actual/synthetic byte checks
with zero initialization and race errors. Two fresh processes measure dense
single-page maintenance10.20–10.30 microseconds and source16.56–16.58, versus the
recorded preceding approximately52–53. Full snapshots measure28.09–29.48 dense
and90.90–91.01 source microseconds. Skip-only costs2.73–4.08 microseconds are
retained, not hidden; all samples and memory are in `runs/screen-03`. These are
maintenance components only. The same lease first qualifies the native reader's
compact-prefix behavior on all12 actual batches, then releases at21:06:21 UTC.
Independent verification finds no reservations or compute processes.

The package integration changes only `summarize_compact`; the cooperative kernel
body equals the512-thread prototype. Candidate fusion and all scoring/selection
arithmetic remain unchanged. Fresh application of the ordered11-patch stack
from C36 equals the overlay (`runs/local/source-check.json`). Runtime gates must
include both summary declarations in the frozen-call preparation, all actual
coarse/fine checks, strict initialization, lifecycle/recycling and two complete
selector timing processes. Model recipe `native-source-tp4.json` is an explicit
interaction of the existing native-source dispatch with the two separately
screened exact optimizations; it has no answer or request qualification yet.

Integrated source `f8b66cd5` passes64 declared calls (including both maintenance
kernels) across eight configurations after cache freeze, all12 actual coarse/
fine checks, zero strict initialization errors and six lifecycle/recycling cases.
Two fresh complete-selector processes retain the preceding component behavior;
required maintenance independently measures10.22–10.79 microseconds per dense
page and16.67–16.88 per source page, with full snapshots28.64–29.23 and91.00–91.58.
All samples and memory remain in `runs/runtime-gates-01`. The lease releases
at21:10:48 UTC with no containers. Final-answer and request qualification remain
separate; no whole-model gain is inferred from these maintenance timings.
