# Matched request comparison

Both arms use the same C36 image and exact scratch-read guard. The only runtime variable is clustered search.

Inputs match across all four boots; output digests match: **True**.

| Workload | Control seconds, two boots | LSA seconds, two boots | LSA latency change |
| --- | --- | --- | --- |
| code-c1 | 3.5090 / 3.5084 | 3.5703 / 3.5743 | +1.8% |
| code-c8 | 6.5114 / 6.4995 | 6.7449 / 6.7629 | +3.8% |
| prose-c1 | 4.2765 / 4.2843 | 4.3630 / 4.3671 | +2.0% |
| prose-c4 | 6.3247 / 6.3536 | 6.5142 / 6.5238 | +2.8% |
| prose-c8 | 6.8813 / 6.8789 | 7.1345 / 7.1555 | +3.8% |
| code-c4 | 4.3450 / 4.3521 | 4.4960 / 4.4971 | +3.4% |
| prefill 131072 | 19.3159 / 19.2598 | 25.9646 / 25.8925 | +34.4% |
| prefill 32768 | 4.9684 / 4.9581 | 5.6968 / 5.6723 | +14.5% |

Two fresh boots per arm; Student-t intervals use boot medians, df=1. Normality is unverified at n=2. Ratios use Bonferroni97.5% component intervals for joint95%; no positive ratio interval when a component crosses zero.

Positive latency change means slower. Full intervals and per-boot TTFT are in `comparison.json`.

Original native reports and request receipts retain every sample, memory/thermal observation and error. No release was remeasured or promoted.
