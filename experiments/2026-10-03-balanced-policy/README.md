# Balanced collective policy

Unpromoted follow-up to the bidirectional relay and NCCL ring experiments.
The objective is the lowest measured latency with balanced payload transmission,
using both PCIe roots on each cable for bulk operations. Tiny operations may
have indivisible payloads; report their exact channel and byte distribution.

First isolate buffer size and channel count on the existing image. Keep
RoCEnante all-reduce capacity (2 MiB), all-gather shard capacity (4 MiB), and
sequence-parallel model thresholds fixed. Compare a same-image clockwise NCCL
control, balanced baseline, 256 KiB/4 MiB buffers, eight channels and channel
retention. Select adaptively from the evidence, retaining every run.

Capture per-operation RDMA vport bytes (distinct PCIe functions), physical-port
bytes (shared views, never sum them as unique wire bytes), RoCEnante per-HCA
payload deltas, NCCL plans, exact-data/graph checks, cancellation diagnostics,
RDMA errors, host memory and thermal telemetry. Timing uses the slowest rank
per sample, then the median; counters are read outside timing. Bracket the
selected candidate with controls. Changes to reduction order are not promised
bitwise equivalent to production.

After transport screening, qualify the selected profile with matched serving
work, pinned verification costs and unchanged thermal/memory guards. A global
optimum cannot be established by a finite screen; report the tested choice and
remaining limits. No promotion without the owner's acceptance.

## Planner candidate

An independent source patch adds a second bidirectional channel ordering:
CW/root0, CCW/root1, CCW/root0, CW/root1. Every pair starts with opposite
directions; a complete group of four covers both roots in both directions.
Mode 1 retains the previous half-reversal, mode 0 the upstream rings.
`NCCL_MIN_TRAFFIC_PER_CHANNEL` exposes the existing 32 KiB allocation floor,
with that default unchanged. The fine arm tests 512 bytes plus the previously
screened Ring thread thresholds. Actual low/middle/high channel element counts
are logged with TUNING enabled; production WARN logging emits none.

The paired, fine and paired-fine arms separate ordering from finer allocation.
No collective arithmetic implementation is replaced, but partitioning and ring
order can change floating-point rounding. The existing CUDA graph, exact-data,
non-aligned fallback and cancellation checks remain mandatory.

## Adaptive tiny-message threads

The global 128-thread screen fills all four lanes at 128 bytes/rank but costs
latency at larger shapes. The adaptive source adds a mode-2-only Ring rule:
reduce threads down to 128 before dropping channels when a message is too
small for the current four-or-more-channel budget. Large calls retain the
original thread count, and mode 0/1 and non-Ring algorithms are unchanged.
The adaptive profile combines this with the measured 4 MiB buffer, 512-byte
allocation floor and Ring retention thresholds. Its own image and manifests
preserve the earlier source and every unsuccessful screen for comparison.
