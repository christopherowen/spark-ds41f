# Shortlist kernel screen

snapshot query-only timings; replicated cases are synthetic; no full-model token/request qualification.

| Workload | Snapshot | Control ms (two processes) | Shortlist ms (two processes) | Query speedup |
| --- | --- | --- | --- | ---: |
| captured | rank-0-dense-0.pt | 0.0280/0.0285 | 0.0350/0.0350 | 0.81x |
| captured | rank-0-dense-1.pt | 0.0286/0.0267 | 0.0350/0.0350 | 0.79x |
| captured | rank-0-dense-2.pt | 0.0286/0.0280 | 0.0350/0.0351 | 0.81x |
| captured | rank-0-source-0.pt | 0.0472/0.0473 | 0.0636/0.0652 | 0.73x |
| replicated512K | rank-0-dense-0.pt | 0.2589/0.2582 | 0.0407/0.0403 | 6.38x |
| replicated512K | rank-0-dense-1.pt | 0.2558/0.2715 | 0.0410/0.0410 | 6.43x |
| replicated512K | rank-0-dense-2.pt | 0.2574/0.2597 | 0.0415/0.0410 | 6.27x |
| replicated512K | rank-0-source-0.pt | 0.2984/0.2880 | 0.0717/0.0726 | 4.06x |

Full confidence intervals, cold/warm construction costs, break-even accounting
and all underlying samples are retained in `results.json` and `runs/`.
Construction is required work outside these query timers, not a free input.
Small captured single-query cases lose to native search. Long replicated
cases establish a query-kernel scaling result, not real long-request speed.
Generated-token equivalence and serving cache maintenance remain pending.
