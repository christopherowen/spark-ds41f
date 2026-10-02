# Four-path opposite-peer forwarding

Base: `0fbd7b7`, with main's thermal-baseline tooling merged before testing.
No serving promotion. The rejected direct-first wave patch is excluded.

Hypothesis: split opposite-peer payloads equally across both physical lanes
through both neighbors. This distributes equal payload work across all directed
links, instead of the old 3:1 maximum/minimum imbalance. Direct-neighbor payloads
retain two paths. A second variable rotates the starting interface and direct /
opposite posting priority, offset by rank and sequence, without a completion
barrier. Test that separately in the same image.

- `cluster-control.json`: two opposite paths, existing submission order.
- `cluster-four.json`: four opposite paths, existing submission order.
- `cluster-rotate.json`: four paths and rotating submission order.
- `cluster.json`: four paths with a public example site for source preparation
  and CI. Actual tests use the ignored four-node site map.

Every arm uses the same source-verified image and 8192-entry hairpin queues.
ABI 8 carries the path geometry and rotating-order setting. Neighbor paths are
still two; unused neighbor completion slots are not polled. Each opposite-path
chunk has its own completion flag, and the GPU waits for all four. Reduction
order and numerical operations are unchanged.

Checks before hardware: exact reciprocal endpoint/forwarding-rule mapping,
equal derived directed-link work, unchanged neighbor paths, malformed geometry
rejection, 26,000 simulated collectives under ASan/UBSan including delayed DMA,
uneven small chunks, sequence wrap and explicit rotation-order assertions.
Repository tests cover partial fabric cleanup and the new topology.

Hardware plan: request an exclusive window; require clean published isolated
checkouts; save entry NIC/fan state; apply queues with confirmed reloads; build
and distribute one image ID. Start every screen with all four nodes below 55 C
using the shared cooling routine, restoring their usual fan control before
measurement. Run exact eager/graph collective checks and the same BF16/FP32
latency matrix for control, four paths and rotation, then repeat in reverse
order. Retain relay as the large-payload control. Capture physical TX/RX bytes,
buffer-overflow and RDMA counters around each timed case. Keep all raw attempts,
including failures. Restore entry queues and verify cleanup before releasing.

Status: local correctness tests pass; hardware testing awaits the window held
by another session. A first attempt to take the hold correctly refused the
existing reservation; no host changes occurred. The first repository test run
found a fixture missing the new `paths` argument; the fixture was corrected and
all 171 tests passed. The public CI profile initially inherited the private
head address; doctor rejected it, and it now uses the example head address.
Fresh source preparation and CI pass, including the four-path sanitizer job.
`local-validation/` preserves the local receipts and coordination request.
No hardware performance result is claimed yet.
