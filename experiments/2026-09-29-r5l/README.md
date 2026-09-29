# r5l candidate: indexer split and optional carve-out check

Date: 2026-09-29. Base: promoted r5k.

## Changes

- vLLM 0025 splits the replicated indexer's rows across the TP ranks during
  sequence-parallel prefill (`experiments/2026-09-29-indexer-split`): -5% per
  4,096-token chunk at 64K of context to -15% at 200K, decode unchanged.
- vLLM 0026 adds an opt-in integrity check of the weights held in the display
  carve-out: with `SPARK3_DISPLAY_CARVEOUT_CHECK_SECONDS` above 0 the worker
  checksums them before and after the copy and re-checks them on a side
  stream at that interval; a mismatch stops the engine. The candidate turns
  it on at 300 s for long unattended runs, with a removal review due
  2026-10-29 (`TODO.md`).

## Arms

| Arm | What it is |
|---|---|
| `candidate` | the r5l image (vLLM tree `66293624`), r5k's configuration, `SPARK3_DISPLAY_CARVEOUT_CHECK_SECONDS=300` |
| `candidate-check` | `candidate` with `SPARK3_DS41_INDEXER_SPLIT_CHECK=1` and the carve-out check every 60 s |

## Method

`build.sh` builds the image from the lock and loads it on every node.
`run.sh` boots `candidate-check`, runs the carve-out unit tests inside the
running image, and runs `capture_depth.py` (8K-200K of context) with four
background decode streams while every rank compares its split indexer rows
with the full-row indexer; any differing row, or a node without a passed
carve-out check, ends the run there. It then boots `candidate` and runs the
reference bench, real-text prefill to 200K and the needle check at 180K.
