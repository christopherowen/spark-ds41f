# r6s: Engram disk shards' hash columns interleaved across ranks

r6s is r6r ([experiments/2026-10-16-r6r](../2026-10-16-r6r/README.md)) with vLLM 0154.

## Why

r6r gave each rank a contiguous band of the 24 hash columns. Heads 0-7 hash 2-grams, 8-15
3-grams and 16-23 4-grams, and a longer n-gram repeats less, so TP3's rank 2 read the most
distinct rows. The r6r timeline trace (2026-10-16) showed:

| | distinct n-grams per 4K chunk of source text | full-chunk read, layer 1 (median) |
| --- | ---: | ---: |
| rank 0 (2-grams) | 2,028 (1.00) | 51.6 ms (1.00) |
| rank 1 (3-grams) | 2,739 (1.35) | 67.9 ms (1.32) |
| rank 2 (4-grams) | 3,129 (1.54) | 74.8 ms (1.45) |

The drives, firmware, temperatures and file fragmentation are equivalent. Only a prompt's
first prefill chunk waits for its rows (the step-aware reader hides every later chunk), up
to about 40 ms on rank 2. Each row costs two random reads (its E8M0 scales are a separate
table), and one io_uring thread gets about 800K sorted 4 KiB reads a second from the drive
at any queue depth from 128 up.

## Change

- **Ownership.** Rank r reads every tp-th column from r: 3/3/2 heads per order on TP3 and
  2/2/2 on TP4. The slowest TP3 rank's distinct rows drop by about 13%.
- **Reader and gather.** Both take a 24-bit column mask; lookup i of a step is column
  i % 24.
- **Forward.** Each rank's columns are gathered, then put back in column order with one
  transpose. These are exact copies, so outputs should equal r6r's.
- **Timeline.** The debug timeline (off in serving) records each step's planning time
  (`plan_ns`), to size the reader's planning against its I/O.

## Runs

1. `image.json`: built on dgx2 (dgx4 is in use) with serving stopped, then copied over the
   CX7 links.
2. `tests.json`: the image's reader, gather, Engram model and SP prefill tests, on dgx2 and
   dgx3.
3. `gate.json`: temperature-0 determinism and distinct streams. The outputs must match
   r6r's hashes.
4. `timeline.json`: the r6r timeline workload (cold 4K and 16K prompts, then eight
   streams), for per-rank reads, first-chunk waits and planning time.

Patch head `1fb7552e` (tree `937414d5`), replayed on `19f2c20e` as `build prepare` does.
The same replay reproduces r6r's head and tree.
