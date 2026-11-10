# #1008's timing mark at tp4-ring (dgx1..dgx4), 2026-11-10

[../../mark-1008-tp4.json](../../mark-1008-tp4.json), run in dgx1's window (10:15-10:30Z). Its preparation and method are
in [prep-result.json](prep-result.json). The verdict, from
`check_1008_bench.py verdict candidate-55-mark-1008-tp4`, is in [verdict.txt](verdict.txt).

**Why not the PR's own bench.** `bench_roce_core_affinity.py` is a single-process C harness, with no network, peers
or real sparknet, and its numbers do not depend on the image. It runs here only as the PR's record, in the
`kernel` job (`kernel/`), beside the PR's sparknet unittest and pytest. Both containers pass.

**The measurement.** ABBA in one session: candidate-54, candidate-55 without #978, the same again, then
candidate-54. Every arm runs with `SPARKNET_ROCE_TRACE=1`, under the same rank-0 load (16 chat streams plus 2
`/tokenize` loops). Sparknet's per-op trace gives the 116-collective step of the serving ring. The traces (257 MB) stay
on dgx1 ([large-files-on-nodes.tsv](large-files-on-nodes.tsv)).

| Metric | candidate-54 (two boots) | candidate-55 without #978 (two boots) | Reading |
| --- | --- | --- | --- |
| ring step p99 (the declared decider) | 21.30 / 21.97 ms | 21.93 / 19.68 ms | the same (the candidate's repeat spread 2.25 ms) |
| ring step mean | 14.53 / 14.78 ms | 14.90 / 14.46 ms | the same |
| rank-0 step p99 | 17.08 / 18.21 ms | 16.78 / 16.34 ms | **lower beyond the interval**, in all four pairings (-1.8% to -10.3%) |
| rank-0 step mean | 12.06 / 12.49 ms | 12.23 / 11.87 ms | the same |
| worker main thread involuntary switches (sum over ranks) | 195 / 188 | 157 / 150 | about 20% fewer |
| placement partitioned | no / no | yes / yes | as declared |

**Verdict: #1008 passes, as an evidence-backed gain on rank 0** (owner, 2026-11-10: a micro gain supported by evidence
is kept even when the end-to-end effect is below resolution). Its mechanism shows beyond the interval on rank 0's own
step p99, and nothing is worse: the ring mean, the screen and `single-stream` are the same
(`../tp4-ring-no978/`). The ring-level p99, the run's declared decider, reads the same within its spread, so a
serving-level p99 gain is not shown.
