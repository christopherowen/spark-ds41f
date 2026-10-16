**Promoted for TP3 on 2026-10-16** as [`2026-10-16-karmic-kraken-r6s`](../../manifests/baselines/2026-10-16-karmic-kraken-r6s.json) (owner: "release r6s with 0155"), on the release below ([suite run](../../manifests/suites/2026-10-16-karmic-kraken-r6s-tp3/)). `tp4-ring` resolves to r6s but stays unqualified until a TP4 release. The r6n recipe this release ran against is frozen in [r6n/cluster.json](r6n/cluster.json) with its lock. Deployed on the triangle at 22:30 UTC from `b3f4e0a` (API ready in 90 s, `doctor --live` matches, KV 2,845,543 tokens as r6n's, its own cost curves `build-75cd5b96c4ec4942`).

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

Patch head `1fb7552e` (tree `937414d5`) for the runs above, replayed on `19f2c20e` as `build prepare` does.
The same replay reproduces r6r's head and tree.

## Release (owner, 2026-10-16: "release r6s with 0155")

The series now ends with 0155, which corrects the whole-table disk test. Patch head
`238b7ea3`, tree `71c6e1df`. The image job removes the earlier r6s tag (tree `937414d5`) and
builds again on dgx2.

- `image.json`, then `tests.json`: the image's Engram tests, 0155 included.
- `release.json`: release-tp3, with r6s, r6n and r6s again.

On passing, r6s is promoted for TP3. TP4 stays unqualified until r6s is benched on the ring.

### Release results (2026-10-16, 20:10-22:14 UTC, TP3)

The rebuilt image (`sha256:2414b9d1`) passed the Engram tests. The suite ran r6s
(20:10-20:50), r6n (20:52-21:32) and r6s again (21:34-22:14); every node began each arm
below the 55 C cooling threshold (48-52 C), with no thermal or power slowdown.

```
verdicts: r6n -> r6s, against the run's own repeat (r6s -> r6s-again)
metric                             change    ±95%   repeat  verdict
grid: decode geometric mean (15)   +0.47%   0.14%   -0.40%  same
c8: output tok/s                   +0.40%   1.04%   -0.18%  same
c4: output tok/s                   -0.19%   1.95%   -0.08%  same
poisson: output tok/s              +0.23%  20.99%   -0.02%  same
prefill: prefill 32,768 tok/s      +5.37%   1.81%   -0.57%  better
prefill: prefill 131,072 tok/s     +5.61%   0.96%   -0.60%  better
prefill: prefill 262,144 tok/s     +5.09%   0.79%   -0.53%  better
prefill: prefill 500,000 tok/s     +4.83%   0.20%   -0.49%  better
```

All 256 greedy responses are identical to r6n's and the teacher-forced distributions are
identical (KL 0), in both r6s arms. The first arm's decode step (+0.47%) is inside the
repeat; the second arm is +0.07% ±0.14 against r6n. The full table is in the
[suite run](../../manifests/suites/2026-10-16-karmic-kraken-r6s-tp3/release-table.txt).

## Results (2026-10-16, 19:06-19:41 UTC, TP3)

- **Image:** built on dgx2 in 14 minutes, then copied over CX7 to dgx1 (124 s) and dgx3 (113 s).
  One image ID on every node.
- **Tests:**
  - The reader and gather tests passed (18, with the column-mask cases), as did fail-stop.
  - One model test failed on both nodes: `test_disk_rows_read_whole_table_sources` still
    expected all 24 columns from a TP4 rank that now owns six, so 75.0% of elements
    differed (the 18 unowned columns).
  - vLLM 0155 corrects it (`bundles/engram-tests-fixed`, mounted over the image's copy). With
    it, all 35 Engram model tests pass on dgx2 and dgx3. 0155 joins the next candidate's
    series.
- **Gate:** temperature 0 is identical in all six scenarios, and the reference and long-text
  hashes (`7748d638`, `e27d573d`) equal r6r's and r6n's. Distinct prompts: eight streams
  141.4 tok/s (±1.4%), four streams 110.0 (±1.2%), level with r6r.
- **Timeline:** r6r's workload (cold 4K and 16K prompts, then eight streams), layer 1:

| | r6r | r6s |
| --- | ---: | ---: |
| full 4K chunk read, rank 0 / 1 / 2 (median) | 51.6 / 67.9 / 74.8 ms | 64.6 / 63.3 / 68.3 ms |
| cluster stall over 24 prefill steps (each step's slowest rank) | 217.6 ms | 181.3 ms |
| largest single stalls | 39.6, 37.4, 34.6, 33.8 ms | 35.0, 30.0, 28.1, 25.4 ms |
| planning per full chunk (`plan_ns`) | not recorded | 2.0-2.3 ms |

- **The ranks now read evenly.** The slowest rank's chunk reads dropped 8.7% (13% was
  estimated) and the cluster's Engram stall 17%: about 5 ms per prompt's first chunk, with
  identical outputs.
- **Planning is about 2 ms a chunk,** so a faster planner is not worth building. The
  remaining lever is one read per row (scales stored beside their rows).
