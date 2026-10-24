# Retain for integration

The branchless TileLang sampled/rejected-count kernel satisfies this port's
correctness, memory-safety and kernel-performance gates on DGX4 in candidate 4's
image. It preserves the inputs to request-state commits and future draft steps.

Keep one block per request, 128 threads, and prepare the four live/dead and
request-index variants. Reject the earlier lane-packed and branching candidates;
their receipts remain alongside the final run. No fallback or relaxed timing
threshold is introduced.

This is a small bookkeeping migration, not an acceptance-policy change or a
measured improvement in model throughput. Integrate it with the other pending
ports, run the complete release gate and serving comparison, and build a pinned
image before proposing production promotion. The default recipe and promoted
source lock are unchanged.

The separately requested acceptance review prioritizes a losslessly verified
suffix/DSpark hybrid on copy-heavy agent output, then confidence-calibration
measurements before any budget changes. Neither is implemented here.
