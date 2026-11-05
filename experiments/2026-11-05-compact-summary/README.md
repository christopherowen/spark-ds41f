# Compact coarse scoring feasibility

Base `dd579b47`, C36 image and the existing captured operands. This out-of-band
screen tests the interaction of MXFP4-encoded block8 means and the native MXFP4
score kernel at a 32-key tile. The current serving candidate remains unchanged.
The hypothesis is that removing repeated BF16 query expansion and using compact
mean operands can reduce the measured coarse-scoring bottleneck. Precision may
change selected keys; independently graded final answers remain mandatory.

The probe builds the current BF16 means, quantizes a separate mean cache using
the pinned native encoder, and times coarse scores only. Construction and the
extra cache memory are recorded separately; neither is a demonstrated serving
implementation or an amortized request-speed result. A future integrated layout
would still need native-byte, initialization, ownership/recycling, graph/cache,
memory/admission and final-answer qualification. No model recipe uses this probe.

All four outcome-independent `validation-a` snapshots, original physical page
IDs, source/dense shapes and query row counts1/8/64 are retained. Score bits are
checked against an independent BF16 computation of the decoded quantized means;
precision drift and native-top512 coverage remain diagnostics, not quality proof.
Only CUDA graph scorer replay is in score timers. Raw operands remain private
on sp8; no activations, reference calls or comparisons enter serving. All errors,
samples, allocator observations and lease releases must remain in `runs/`.
