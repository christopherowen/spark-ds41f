# Decision

Retain `selected.json` as the preferred tested balanced TP4 candidate. It
balances both directions and both PCIe roots, improves large collective
latency, and gives a modest repeated prefill gain. It does not establish a
large decode gain, eliminate thermal asymmetry, or prove a global optimum.
Production configuration is unchanged.

## Serving result

Both arms used the identical image and pinned verification-cost table. Each
boot passed live configuration checks and the LRU gate (20/20 across four
boots). The first pair measured concurrency 1/8; the repeat added 2/4. Each
decode point has three samples per boot. Native reports and `results.json`
retain all results, realized acceptance, verified rows and confidence intervals.

| Decode case | First pair TPS change | Repeat TPS change |
| --- | ---: | ---: |
| Prose, 1 stream | +3.6% | -7.8% |
| Prose, 2 streams | — | +1.8% |
| Prose, 4 streams | — | -1.3% |
| Prose, 8 streams | -0.3% | +0.0% |
| Code, 1 stream | +4.7% | +6.9% |
| Code, 2 streams | — | -1.6% |
| Code, 4 streams | — | -1.9% |
| Code, 8 streams | -0.5% | -1.8% |

Every decode TPS difference has an interval containing zero. The eight-stream
code estimate is slightly negative in both pairs; this screen does not rule
out a small regression. Single-stream step times change +2.3%/+3.3% for
prose/code in the first pair and -2.8%/+0.8% in the repeat. Step time removes
the direct effect of acceptance but not variation in verification work: for
example first-pair code verifies 4.063 versus 4.263 draft rows per draft. Do
not interpret pinned cost curves as identical executed work.

| Source prefill | Actual input tokens | First pair TPS change | Repeat TPS change |
| --- | --- | ---: | ---: |
| Nominal 4K | 3,465–3,960 | +8.2% | +1.3% |
| Nominal 32K | 27,068–29,569 | +2.5% | +1.6% |
| Nominal 64K | 56,163–59,008 | +1.5% | +1.9% |

The short-input 8% result does not reproduce: the unchanged control improves
5.9% on repeat. Exclude it from the recommendation. The longer-prefill gain
is modest and repeated. In the four-repetition screen the middle size is
4,613 versus 4,687 tok/s, +1.6% with a reported effect interval of +0.3% to
+2.9%; the largest is 4,656 versus 4,745 tok/s, +1.9% with an interval
approximately 0% to +3.9%. Actual prompts and token counts match within each
pair; the nominal size labels are not exact token lengths.

Prefix-cache reuse passes with approximately 99% hits. All measured requests
complete. Minimum observed host memory is 29.73 GiB, swap growth is zero, and
all eight successful benchmark reports record zero GPU thermal slowdown.

## Thermal and qualification limits

Balanced transmission has not removed the hotter dgx1/dgx2 pattern. The
expanded decode control peaks at 96.8 C on a board sensor, versus 88.6 C for
its balanced counterpart. That favorable difference is not consistent across
other matched sections: the first decode pair is 86.6/87.0 C, and expanded
prefill is 89.1/91.8 C. These runs do not establish a thermal cure or explain
the software cause of the asymmetry. Existing guards remained unchanged.

This is a cooled serving screen, not sustained thermal qualification. Full
configured 524K contexts, long-context admission, multimodal requests and the
end-to-end agent workflow were not requalified for this policy. Floating-point
reduction order can change, and generation remains outside the separate
batch-invariance experiment. Promotion requires the remaining qualification
and owner acceptance under the repository procedure.

## Completion

Sixteen transport launches pass on all four ranks: 900 timed cases per rank,
including exact-data and graph checks, with no tracked RDMA error deltas.
The final same-image pair demonstrates the backend cutoffs and 24.8–25.2%
TX/RX shares across all four interfaces at every timed size. Tiny indivisible
fallbacks remain explicit exceptions to equal splitting.

The failed ABI-loader build was corrected and never served. The first client
attempt failed before sending a request because the LAN address was not
reachable from the Mac; measured runs all used `dgx1`. Both receipts remain
in the archive. Final live doctor passed. The window was released with all
four nodes idle, fan-control services active, clean qualification checkouts,
and shared production checkouts unchanged.
