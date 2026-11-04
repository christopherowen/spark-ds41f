# Shortlist kernel screen

snapshot query-only timings; replicated cases are synthetic; no full-model token/request qualification.

Times are per call: one query or a batch of eight repeated queries, as labeled.

| Workload | Snapshot | Control call ms (two processes) | Shortlist call ms (two processes) | Speedup |
| --- | --- | --- | --- | ---: |
| captured | rank-0-dense-0.pt | 0.0280/0.0285 | 0.0350/0.0350 | 0.81x |
| captured | rank-0-dense-1.pt | 0.0286/0.0267 | 0.0350/0.0350 | 0.79x |
| captured | rank-0-dense-2.pt | 0.0286/0.0280 | 0.0350/0.0351 | 0.81x |
| captured | rank-0-source-0.pt | 0.0472/0.0473 | 0.0636/0.0652 | 0.73x |
| replicated512K | rank-0-dense-0.pt | 0.2589/0.2582 | 0.0407/0.0403 | 6.38x |
| replicated512K | rank-0-dense-1.pt | 0.2558/0.2715 | 0.0410/0.0410 | 6.43x |
| replicated512K | rank-0-dense-2.pt | 0.2574/0.2597 | 0.0415/0.0410 | 6.27x |
| replicated512K | rank-0-source-0.pt | 0.2984/0.2880 | 0.0717/0.0726 | 4.06x |
| captured-repeated8queries | rank-0-dense-0.pt | 0.0962/0.0963 | 0.0499/0.0497 | 1.93x |
| captured-repeated8queries | rank-0-dense-1.pt | 0.0954/0.0960 | 0.0498/0.0500 | 1.92x |
| captured-repeated8queries | rank-0-dense-2.pt | 0.0962/0.0964 | 0.0500/0.0505 | 1.92x |
| captured-repeated8queries | rank-0-source-0.pt | 0.1227/0.1231 | 0.1446/0.1436 | 0.85x |
| replicated512K-repeated8queries | rank-0-dense-0.pt | 1.7181/1.7006 | 0.1363/0.1351 | 12.59x |
| replicated512K-repeated8queries | rank-0-dense-1.pt | 1.7022/1.7000 | 0.1364/0.1374 | 12.42x |
| replicated512K-repeated8queries | rank-0-dense-2.pt | 1.6951/1.7106 | 0.1355/0.1364 | 12.52x |
| replicated512K-repeated8queries | rank-0-source-0.pt | 1.7518/1.7535 | 0.2339/0.2330 | 7.51x |

Full confidence intervals, cold/warm construction costs, break-even accounting
and all underlying samples are retained in `results.json` and `runs/`.
Construction is required work outside these query timers, not a free input.
Small captured single-query cases lose to native search. Long replicated
cases establish a query-kernel scaling result, not real long-request speed.
Generated-token equivalence and serving cache maintenance remain pending.
