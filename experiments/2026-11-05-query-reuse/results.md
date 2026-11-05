# Completed-answer pilot

Published serving source `e9eab4604cf4e46996d28c0bfba8c9fe75f7d282`, immutable
C36 image `sha256:3795255f485ea5c1a9db54c742921dd9b2917daad0799361e5c6f56a70119713`,
Native68 versus `full.json`. Runtime files are unchanged from `fd5958df`.
One intended variable is the complete block8/query-reuse selector versus native
selection; generation, checkpoint, hardware and client protocol are shared.
All raw responses remain private on the serving controller, outside Git.

| Serial cold pilot | Native A | LSA A | LSA B | Native B |
| --- | ---: | ---: | ---: | ---: |
| Independently correct final answers | 8/12 | 9/12 | 9/12 | 8/12 |
| Natural completions | 12/12 | 12/12 | 12/12 | 12/12 |
| Workload seconds | 633.132 | 559.301 | 557.999 | 631.111 |
| Median request seconds | 28.697 | 37.990 | 37.837 | 28.597 |
| Generated tokens, including reasoning | 39767 | 31352 | 31352 | 39767 |
| Accepted tokens per draft round | 3.065 | 2.611 | 2.611 | 3.065 |

Native/LSA mean workload time is 1.1315x, or 11.62% less elapsed time and 13.15%
more requests per second. The two-boot joint95 ratio interval is [1.0542, 1.2136].
It assumes normally distributed boot means with only two samples per arm; it is
not a distribution-free assurance about other tasks or conditions. Correct-answer
throughput increases 27.30%, with the same conditional interval [1.1869, 1.3663].
The quality sample is 12 independent documents, never 48 independent questions.

All four boot pairings have one win (`67041f08bb02136c067cdb52`) and zero losses;
the exact one-sided paired p-value is 0.5. This does not establish general quality
superiority. Every final answer completes naturally at temperature0, seed0,
high thinking and a 65536-token generation allowance. The hard quality checker
also finds zero repeated-boot answer-decision instability.

The speed improvement is uneven. Median request latency rises about 32.4%.
On the long structured-data question, Native A takes 250.94s and 23999 generated
tokens, while LSA A takes 165.84s and 13510 tokens; both answer correctly. Other
requests become slower. Aggregate prefill time falls about 5%, while accepted
draft tokens decline and per-token generation remains a performance concern.
Shorter reasoning contributes materially to the workload win; this is not a
claim of uniformly faster decode kernels or better latency on every request.

Minimum available memory remains 7.09 GiB on sp5 and at least 9.02 GiB on the
other ranks. Every boot records zero swap growth, zero preemptions, no memory
guard stop and no reported thermal slowdown. Each arm uses the same below55C
start criterion and stock firmware fan policy. LSA A's first compilation boot
requires longer cooling; cooling and startup are recorded separately from
request timing. LSA's indexer layout is 100 bytes/key versus native68, plus at
most 2 MiB of admitted query workspace; no capacity improvement is claimed.

C37 restoration completes at 2026-11-05 16:30:49 UTC and releases every hold.
Independent verification at 17:36:55 UTC confirms the same C37 image running on
all three ranks, clean equal serving checkouts, no holds/requests and no
experimental mounts or environment. The verification is retained alongside the
native run receipts. Promoted configuration and immutable baselines are unchanged.

The next confirmation is preselected independently of outcomes: 45 unseen,
document-disjoint LongBench v2 contexts, with serial cold, four-concurrent cold
and four-concurrent cached phases. It uses the same exact runtime and fresh
matched controls, without tuning after viewing confirmation answers:

```sh
python3 experiments/2026-11-05-answer-quality/run_window.py \
  --out /home/chris/spark-lab/results/query-reuse/heldout-window-01 \
  --cache /home/chris/spark-lab/datasets/longbench-v2/ac35867b9c9436dc81c4c7535fe640ba7cf47877/policy-65536-v1 \
  --split heldout \
  --native-config experiments/2026-11-05-query-reuse/native.json \
  --candidate-config experiments/2026-11-05-query-reuse/full.json
```

This pilot permits further evaluation only. Independent coding, longer-context,
memory/admission and project performance-matrix gates remain outstanding.
