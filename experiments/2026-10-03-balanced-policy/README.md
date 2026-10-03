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
