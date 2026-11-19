Timing qualification correction,2026-11-06: these stock-fan measurements are exploratory. The documented performance fan curve was not verified; recorded below55C starts and no thermal-guard interruption do not satisfy that requirement. Answer/kernel correctness evidence remains separate. All original numbers and receipts are unchanged. See `../2026-11-06-timing-conditions/README.md`.

# Raw-wide coarse component results

Source `4ac35e2505550e8f2bec9e08617320d74eaa403c`; one exact selection scheduling variable relative to the source1024/dense64/native-generation parent. All24 adversarial batches (12 length/cap cases each), six race batches,78 declared calls,12 captured runtime batches, four strict initialization cases and72 graph routes pass. Initialization and race reports contain zero errors/hazards. All comparison/reference work remains outside serving and timers.

| Fixed captured rows | Process A generic/raw | Process B generic/raw | Conditional95 interval |
| --- | ---: | ---: | --- |
| 1 | 2.409x | 2.414x | [2.382, 2.442] |
| 8 | 2.688x | 2.650x | [2.440, 2.920] |
| 64 | 3.391x | 3.409x | [3.282, 3.522] |

Each process geometrically aggregates three fixed source captures. The interval assumes normally distributed log boot ratios with only two process samples (df1); fixed captures/replays are not independent boots. Fine attention, summary construction, decode queries, full requests and model answer quality are not measured here. Coarse scratch is one quarter of the prior route at full width; model admission and peak serving memory are unqualified. Before/after host memory and actual-batch memory observations are retained. Strict no-cache allocator counters are unavailable.

Attempt01 is invalid because before-GPU publication was not established. Its full failure, interruption and release are retained and none of its timings is used. Attempt02 records a fresh fetched publication-ancestry check before GPU work, passes every stage and releases its own reservation/container. Independent release verifies both absent and preserves unrelated containers. No serving service was changed on sp8.

The parent model comparison remains separate. A repeated model request comparison is required before a request-speed or final-answer claim. Complete image-build inputs are unavailable. No production promotion.
