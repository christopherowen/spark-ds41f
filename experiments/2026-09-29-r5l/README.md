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
| `candidate` | the r5l image (vLLM tree `d1886d36`), r5k's configuration, `SPARK3_DISPLAY_CARVEOUT_CHECK_SECONDS=300` |
| `candidate-check` | `candidate` with `SPARK3_DS41_INDEXER_SPLIT_CHECK=1` and the carve-out check every 60 s |

## Method

`build.sh` builds the image from the lock and loads it on every node.
`run.sh` boots `candidate-check`, runs the carve-out unit tests inside the
running image, and runs `capture_depth.py` (8K-200K of context) with four
background decode streams while every rank compares its split indexer rows
with the full-row indexer; any differing row, or a node without a passed
carve-out check, ends the run there. It then boots `candidate` and runs the
reference bench, real-text prefill to 200K and the needle check at 180K.

## First build

The first r5l image (`sha256:df5f5596…`, vLLM tree `66293624`) passed the
carve-out unit tests inside the running image on dgx3 (8 passed) and ran two
carve-out checks on every node. Its split check compared the split with one
full-row run and still reported differences on every sequence-parallel
forward, all at layers 2, 8 and 14 and all as set differences (for example
1,114 of rank 0's 1,366 rows at layer 2), while layers 20-36 and the
candidate lists matched. Rank 0's rows 0-1,366 had the same inputs and chunk
boundaries in both paths, so the split could not explain them. Those three
layers are the only ones that select from every compressed key with B12X's
tiled radix top-k (a port of SGLang's), which keeps an arbitrary subset of the
positions tied at its 512th score through shared-memory atomics; forwards with
at most 512 keys matched. Tied positions carry equal scores, so either choice
is the same attention.

Patch 0025's check now runs the full-row indexer twice and compares the split
with the first run beside the second run's spread (`check_summary.py`), and
the image was rebuilt (vLLM tree `d1886d36`). The first image's logs are in
`results/private/r5l/first-image/`.

