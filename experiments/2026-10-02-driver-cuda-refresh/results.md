# Matched screen results

The four columns follow execution order. Each decode point has three measured
samples after a warmup; each prefill point has two. Full intervals, acceptance,
verification work, per-request lengths and native identities are in the linked
[reports](runs/) and [normalized results](results.json). These are screens, not
a resolved estimate of a small speed difference.

## Decode throughput (aggregate tokens/s)

| Workload | R580 before | R610 corrected | R610 + CUDA exp2 | R580 after |
| --- | ---: | ---: | ---: | ---: |
| prose-c1 | 48.91 | 54.23 | 52.40 | 53.60 |
| prose-c8 | 163.99 | 172.38 | 173.25 | 172.15 |
| code-c1 | 62.89 | 61.15 | 62.30 | 63.08 |
| code-c8 | 190.97 | 196.58 | 195.21 | 190.76 |
| prose-nothink-c1 | 55.75 | 60.92 | 59.89 | 60.13 |
| prose-nothink-c8 | 180.96 | 188.78 | 183.28 | 179.58 |
| code-nothink-c1 | 83.12 | 85.46 | 82.94 | 83.13 |
| code-nothink-c8 | 237.72 | 242.87 | 250.19 | 254.23 |

## Estimated single-stream step time (ms, mean / median)

| Workload | R580 before | R610 corrected | R610 + CUDA exp2 | R580 after |
| --- | ---: | ---: | ---: | ---: |
| prose-c1 | 46.42 / 41.94 | 41.07 / 41.14 | 40.84 / 40.57 | 41.73 / 41.09 |
| code-c1 | 46.57 / 47.02 | 44.57 / 44.77 | 44.95 / 44.97 | 45.35 / 45.67 |
| prose-nothink-c1 | 42.93 / 43.13 | 42.70 / 42.67 | 43.57 / 42.68 | 43.13 / 43.54 |
| code-nothink-c1 | 48.74 / 48.86 | 47.21 / 47.34 | 47.37 / 47.41 | 47.81 / 47.60 |

## Source-text prefill (tokens/s)

Nominal lengths are workload labels; actual sample lengths are listed in the README.

| Nominal length | R580 before | R610 corrected | R610 + CUDA exp2 | R580 after |
| --- | ---: | ---: | ---: | ---: |
| 4096 | 3881.95 | 3839.50 | 3898.43 | 3847.61 |
| 16384 | 3835.70 | 3838.36 | 3828.43 | 3795.23 |
| 65536 | 3855.09 | 3861.26 | 3359.56 | 3827.05 |

## Integrity and memory

All four screens passed the 5/5 LRU structural checks and completed without
failed requests, measured swap growth or thermal slowdown. Minimum available
host memory on dgx1 was 6.80, 6.84, 6.60 and 6.58 GiB, respectively.

Two measurements need explicit qualification:

- The first R580 prose single-stream control includes a 55.98 ms step estimate; its median is 41.94 ms. Comparing only its mean with R610 exaggerates the improvement. The second control averages 41.73 ms.
- CUDA exp2 has a slow first long-prefill sample: 20.83 s for 59,653 tokens (2,863.82 tok/s), versus 14.60 s for 56,289 tokens (3,855.31 tok/s). No post-readiness JIT event or driver Xid was found. Its cause is unresolved; it remains in the report.

The corrected R610 boot had no Xid, kernel BUG/Oops or OOM entry in the checked
kernel journal, including both serving arms. Successful screens do not replace
long-context, multimodal, admission or numerical-equivalence qualification.

See [decision](decision.md).
