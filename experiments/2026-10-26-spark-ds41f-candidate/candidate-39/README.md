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

Status: canonical Linux source preparation passed, reproducing vLLM tree
`8e33015b1fa1d34250e686db7d1f8061021d6d5a`, head
`31412339aafdd2c1392c7294975355dbc523c5e1`. The full repository suite passes
626 tests with hashed benchmark-client dependencies. No image, serving or
performance qualification yet. No promoted configuration or immutable
baseline is changed by this candidate.
