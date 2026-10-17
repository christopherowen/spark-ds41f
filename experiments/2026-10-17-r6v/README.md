# r6v: r6u with the row all-to-all over a four-node ring

**Promoted for TP4 on 2026-10-17** as [`2026-10-17-karmic-kraken-r6v`](../../manifests/baselines/2026-10-17-karmic-kraken-r6v.json), serving DeepSeek-V4.1-Flash-1017 (owner: TP4 with only the benching promotion and the release announcement need), on the release below ([suite run](../../manifests/suites/2026-10-17-karmic-kraken-r6v-tp4/)). The recipe's one image is r6v on both shapes: the triangle's path is r6u's and keeps r6u's release run as its reference (`r6v-tp3-1017.json`). The r6u recipe this release extends is frozen in [r6u/cluster.json](r6u/cluster.json) with its lock.

### Results (2026-10-17, TP4, 1017)

- Image `sha256:8eb7fd81` (built on dgx2, copied over the ring's links): release tests 1,371 passed, 41 skipped.
- Gate: ready in 96 s with no NCCL warning (r6u's three boots had died in memory profiling); temperature 0 (3eab5112 / db25fde3) and greedy (c9cf9dbe) identical in every scenario; 16, 8 and 4 distinct streams 243.4, 184.2 and 137.0 tok/s.
- Release (r6v 19:20-20:06, r6n on its production ring curves 20:07-20:54 UTC):

```
metric                         r6n-tp4-1017 r6v-tp4-1017   change    ±95%  verdict
decode geometric mean (20)                             -0.05%   0.14%  same
c16 output tok/s                243.497      243.770   +0.11%   0.95%  same
c8 output tok/s                 185.380      185.443   +0.03%   1.13%  same
poisson output tok/s            180.247      180.440   +0.11%  16.87%  same
prefill 32,768 tok/s           5762.330     5982.405   +3.82%   0.64%  better
prefill 131,072 tok/s          5706.062     5946.093   +4.21%   1.08%  better
prefill 262,144 tok/s          5466.914     5676.088   +3.83%   0.56%  better
prefill 524,288 tok/s          4988.260     5170.933   +3.66%   0.19%  better
prefill 1,000,000 tok/s        4268.951     4405.278   +3.19%   0.45%  better
```

  Acceptance length 2.268 against 2.261; no greedy response identical to r6n's (top-1 agreement 94.5%: WO-A's input quantization).
- Single stream (`single-stream-tp4-1017.json`): 71.05 tok/s (±3.8%), TTFT p50 0.103 s, TPOT p50 14.1 ms, 100 requests, none failed.
- RigMark (`rigmark-tp4-1017.json`, alexellis/rigmark 770c3106, `reasoning_effort` low) for the release notes
  (the card and its JSON receipt, unmodified: [rigmark/](rigmark/)):
  15/15 output gates; prose 66.4, code 109.6, structured 140.6 tok/s (medians, reasoning included);
  prefill 6,272 / 6,685 / 6,684 tok/s at 8K / 32K / 64K; capped C1 / C2 / C4 77.7 / 124.5 / 176.9 tok/s.
- MiaAI-Lab's release benchmark (`sparkdash_bench.mjs`: sparkDash 1.8.8's own decode and prefill code,
  github.com/MiaAI-Lab/sparkDash 754f40a7, with their settings: greedy, thinking off, 256 new tokens),
  run from the Mac against the ring, 2026-10-17 21:25 UTC. Aggregate tok/s (per stream):

  | prompt type | c1 | c2 | c4 | c8 | c16 |
  |---|---:|---:|---:|---:|---:|
  | prose | 86.9 | 128.4 (65.7) | 185.7 (49.0) | 259.1 (34.6) | 380.5 (25.3) |
  | code | 148.6 | 197.6 (102.4) | 252.2 (68.7) | 332.0 (46.6) | 469.7 (32.0) |
  | structured | 163.6 | 236.0 (122.7) | 377.2 (97.6) | 577.4 (75.6) | 806.3 (53.6) |
  | json | 129.9 | 202.6 (103.0) | 261.5 (70.9) | 422.6 (58.0) | 676.8 (44.2) |

  Cold prefill, two passes (sparkDash's filler, one repeated token, so Engram reads one hot row):
  4K 5,802 / 5,739, 16K 6,545 / 6,727, 32K 6,809 / 6,866, 64K 6,826 / 6,856, 128K 6,666 / 6,693,
  262K 6,383 / 6,369 tok/s. MiaAI's TP4 line (switched fabric, 2026-09-25, original checkpoint):
  prose 87.7 / 120.4 / 163.6 / 237.6 / 342.7, prefill 5,893-5,936 at 32K-64K. Their caveats apply to
  both: prose c1 is one prompt whose greedy text follows each stack's arithmetic, and the non-prose
  types use different prompts at each concurrency.

r6v is r6u ([experiments/2026-10-17-r6u](../2026-10-17-r6u/README.md)) with vLLM 0180.

## Why

r6u could not boot on the four-node ring (2026-10-17, three attempts): Engram's
sequence-parallel exchange (`all_to_all_rows`, vLLM 0150, since r6p) sends every peer its
block directly, and on a switchless ring ranks two apart share no cable. NCCL's queue pair to
the opposite rank timed out in memory profiling (`ibv_modify_qp ... INIT -> RTR`, 110, local
10.23.1.2 to remote 10.14.1.4). r6s, which brought the exchange, was released on the triangle
only; r6n, from before it, boots on the ring.

## Change

- **vLLM 0180**: at four ranks `PyNcclCommunicator.all_to_all_rows` sends its neighbours their
  blocks and each half of the opposite rank's block through one neighbour, which forwards it
  in a second group, as the rank-order reduce-scatter's exchange does. The four links carry
  equal bytes; blocks keep their real lengths and their bits. Other rank counts keep the direct
  exchange, so TP3 runs r6u's code.

## Runs

1. `build.json`: the image on dgx2, copied over the ring's links, and the release tests.
2. `gate-tp4-1017.json`: temperature-0 and greedy determinism, 16, 8 and 4 distinct streams.
3. `release-tp4-1017.json`: the `release` suite, r6v then r6n (on its production ring curves),
   both on 1017. Owner: only the benching promotion and the release announcement need.
