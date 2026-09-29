# r5l candidate: indexer split and optional carve-out check

Date: 2026-09-29. Base: promoted r5k.

## Changes

- vLLM 0025 splits the replicated indexer's rows across the TP ranks during
  sequence-parallel prefill (`experiments/2026-09-29-indexer-split`): -5% per
  4,096-token chunk at 64K of context to -15% at 200K, decode unchanged.
- vLLM 0026 adds a debug-mode integrity check of the weights held in the
  display carve-out, off in configuration: with
  `SPARK3_DISPLAY_CARVEOUT_CHECK_SECONDS` above 0 the worker checksums them
  before and after the copy and re-checks them on a side stream at that
  interval after graph capture; a mismatch kills the worker. Unset, it adds no
  work.

## Arms

| Arm | What it is |
|---|---|
| `candidate` | the r5l image (vLLM tree `c108cd6d`), r5k's configuration |
| `candidate-check` | `candidate` with `SPARK3_DS41_INDEXER_SPLIT_CHECK=1` and the debug-mode carve-out check every 60 s |

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

## Second build

The second image (`sha256:d7de38d8…`, vLLM tree `d1886d36`) passed the
controlled split check (`check_summary.py`, 473,494 rows per encoder indexer
layer):

| Layer | Split vs full: rows / positions (max) | Full vs full: rows / positions (max) |
|---:|---:|---:|
| 2 | 371,636 / 1,853,864 (35) | 369,620 / 1,822,442 (36) |
| 8 | 135,023 / 232,982 (14) | 126,244 / 219,847 (14) |
| 14 | 88,260 / 129,886 (12) | 81,293 / 120,504 (11) |
| 20-36 | 0 / 0 over 6,912 rows | 0 / 0 |

No unwritten rows and no changed candidate lists. Layers 20-36 are CED
decoder layers, which run sequence-parallel only for short forwards, so they
saw few checks. The carve-out unit tests passed 8/8 on dgx3 (dgx1's container
ran out of CUDA memory for the tests' tables beside rank 0 and the API
server), and every node passed two carve-out checks.

Its reference run against r5k's: real-text prefill 4K +0.5%, 32K +1.9%, 64K
+3.5%, 131K +6.0%, 200K +9.7% (3,756 against 3,424 tok/s); filler prefill
+2.1% to +8.4%; prefix replay 6.52 s cold and 0.244 s warm (6.71 and 0.273);
quality 5/5; needle 3/3 at 152,914 tokens. But the lowest MemAvailable fell
by 0.62-0.77 GiB on every node (dgx1 5.41 against 6.03 GiB), and decode at
eight streams read 2-4% lower. The carve-out checksum was the cause of the
memory: a 64-bit `sum` casts its int32 input first, so each 421 MiB table
briefly took 842 MiB, which the check stream's allocator pool kept (measured
on dgx3: 842 MiB peak for one table, 32 MiB when summed in 16 MiB chunks, same
value and 9 ms either way). Patch 0026 now sums in chunks, and the image was
rebuilt. The second image's results are under `results/private/r5l/second-image/`
and `results/private/bench/r5l-*-second`.

## Third build

The third image (`sha256:d4cfa69a…`, vLLM tree `3b024a40`) summed the
checksum in chunks. Its carve-out tests passed 10/10 on dgx3 and its split
check matched the second's (layer 2: 1,853,906 split positions against
1,818,807 between full-row runs; layers 20-36: none). Its reference run was
stopped during prefill: the owner ruled that the check must not run in
production, only as a debug mode, and that quality work must not add
computation to serving. Patch 0026 now starts its thread once after graph
capture instead of testing a flag before every engine step, stops the engine
by killing the worker, and is off in configuration; the image was rebuilt
(vLLM tree `c108cd6d`). The third image's results are under
`results/private/r5l/third-image/`.

