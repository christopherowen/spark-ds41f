# Retain for review

The requested DGX4 build verification and DGX1–3 TP3 triangle measurements are
complete. Both standard and CopyDrafter recipes ran functional and full
performance qualification on a boot and repeat boot. Two additional standard
boots retain the C4 tail-latency confirmation. The exact published image and
source were used throughout; released builds were compared from saved records.

Greedy model outputs and recorded probabilities match both archived 1030 boots,
and every functional/performance step completed successfully. Standard headline
decode is about 3–5% faster than 1030 and level with candidate 37. Copy headline
decode is about 3–5% faster than 1030; direct comparison with standard retains
small costs whose verdicts depend on the reference boot.

Keep candidate 38 unpromoted. C4 p99, the mixed-prefill first-chunk wait, small
copy-recipe costs and the inherited prefill loss against candidate 36 prevent a
blanket no-regression verdict. All adverse rows remain in the native comparisons.
The archived release repeat provides context, not a new p99 confidence floor.
The combined recipes do not isolate the cost of patch 0677. Do not select a
favorable reference boot, replace the frozen reference or discard a run.

This experiment does not establish absence of spontaneous long-session loops,
seeded sampling batch invariance, or qualification of another topology. The
long-context canary under #864 and separate topology work retain their own scope.
No runtime workaround or patch reversal is inferred from these measurements.

Candidate 36 was restored on all three nodes and independently verified. The
shared window closed at 2026-11-04 02:02:46 UTC. No promoted configuration, lock
or immutable baseline changed.
