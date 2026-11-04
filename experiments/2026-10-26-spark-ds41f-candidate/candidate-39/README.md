# Candidate-39: CopyDrafter by default

Candidate-38 plus rebased #842, patch 0688. Copy drafts are derived by default
for greedy adaptive DSpark, without an explicit speculative-config option.
The decode-row and CUDA-graph envelope grow automatically while preserving
prefill chunk boundaries. Match/allocation work overlaps DSpark, with a main
stream wait before proposals are consumed; allocation and proposals retain the
existing exact reference checks.

This explicitly tests the interaction of default enablement and the copy-work
implementation. Compare with both saved candidate-38 standard and copy arms,
not just the released recipe. All prior #842 results and failures are preserved
under `experiments/2026-11-01-dspark-default-copy-drafts`; they do not qualify this
rebased build. The core-metric protocol, hardware constraints and round order
are in `experiments/2026-11-04-copy-default-screening/README.md`.

Status: source preparation. No image, serving or performance qualification yet.
Source identifiers are provisional until the fresh canonical Linux replay
updates the lock and source manifest. No promoted configuration or immutable
baseline is changed by this candidate.
