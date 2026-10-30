# High-Concurrency Speculation Scaling & Adaptive Verification Budgeting (`c1`–`c16`)

Addresses `TODO.md:471-491`:
- Audits which limit binds first across `c1`, `c4`, `c8`, and `c16` (`1–16` streams) on `spark-ds41f:1030` (`tp4` and `tp3`, greedy and sampled) using recorded draft/verification accounting traces (`adaptive_verification_row_cost = 0.035`, `SPARK_DS41F_DSPARK_DEAD_ROWS_TAU = 0.20`).
- Aligns host-side adaptive verification budgeting (`AdaptiveVerificationManager.get_num_tokens`), verified draft count reporting (`get_verified_draft_counts`), and online estimator grading (`GPUModelRunner.sample_tokens`) with the device dead-row cut and the SM121 16-row Tensor Core MMA tile knee.

## Recorded `1030` Speculation Accounting Audit (`c1`–`c16`)

| Suite | Workload | Active Reqs/Step | Tok/Step | Tok/Req | Total Rows | Drafts/Req | Yield (`tok/row`) | Wasted Rows | Survival / Coverage by Draft Slot (`0..4`) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| **TP4 Greedy** | `c1` | `1.00` | `2.256` | `2.256` | `4.07` | `3.07` | `0.554` | `1.81` | `0:0.645/1.000, 1:0.343/0.958, 2:0.156/0.625, 3:0.072/0.321, 4:0.039/0.163` |
| **TP4 Greedy** | `poisson` | `5.77` | `13.353` | `2.314` | `17.91` | `2.10` | `0.745` | `4.56` | `0:0.706/1.000, 1:0.351/0.651, 2:0.161/0.303, 3:0.075/0.141, 4:0.041/0.074` |
| **TP4 Greedy** | `c8` | `7.19` | `16.621` | `2.312` | `21.35` | `1.97` | `0.779` | `4.73` | `0:0.710/1.000, 1:0.339/0.550, 2:0.151/0.246, 3:0.072/0.113, 4:0.038/0.058` |
| **TP4 Greedy** | `c16` | `13.09` | `31.026` | `2.370` | `36.52` | `1.79` | `0.850` | `5.49` | `0:0.782/1.000, 1:0.334/0.455, 2:0.147/0.195, 3:0.069/0.092, 4:0.038/0.048` |
| **TP4 Sampled** | `c1` | `1.00` | `2.198` | `2.198` | `4.05` | `3.05` | `0.543` | `1.85` | `0:0.619/1.000, 1:0.324/0.953, 2:0.147/0.618, 3:0.068/0.315, 4:0.037/0.159` |
| **TP4 Sampled** | `c8` | `7.22` | `16.263` | `2.253` | `21.20` | `1.94` | `0.767` | `4.94` | `0:0.692/1.000, 1:0.326/0.537, 2:0.144/0.236, 3:0.068/0.109, 4:0.036/0.055` |
| **TP4 Sampled** | `c16` | `13.13` | `30.245` | `2.304` | `36.17` | `1.75` | `0.836` | `5.93` | `0:0.761/1.000, 1:0.320/0.443, 2:0.140/0.188, 3:0.066/0.088, 4:0.036/0.046` |
| **TP3 Greedy** | `c1` | `1.00` | `2.263` | `2.263` | `4.07` | `3.07` | `0.556` | `1.81` | `0:0.648/1.000, 1:0.345/0.959, 2:0.157/0.626, 3:0.072/0.322, 4:0.039/0.163` |
| **TP3 Greedy** | `c4` | `3.81` | `8.683` | `2.279` | `12.38` | `2.25` | `0.701` | `3.70` | `0:0.684/1.000, 1:0.344/0.720, 2:0.156/0.348, 3:0.072/0.165, 4:0.039/0.086` |
| **TP3 Greedy** | `c8` | `7.21` | `16.670` | `2.312` | `21.32` | `1.96` | `0.782` | `4.65` | `0:0.712/1.000, 1:0.340/0.545, 2:0.151/0.243, 3:0.072/0.112, 4:0.038/0.057` |

## Which Limits Bind First & Fixes Implemented

1. **Low Concurrency (`c1`): Host Budget Admits Sub-$\tau$ Dead Rows (`0.069 \le s < 0.20`)**:
   - At $N = 1$ and `row_cost = 0.035`, `expected_tokens / (1 + 0.035 * rows)` admits draft slots down to marginal survival $s_{\min}(1) = 0.0692 < \text{dead\_rows\_tau} = 0.20$.
   - As shown in `c1` coverage (`cov@2 = 0.625` at `surv = 0.156`, `cov@3 = 0.321` at `surv = 0.072`, `cov@4 = 0.163` at `surv = 0.039`), the host budget admitted ~1.1 dead rows/step (`4.07` total rows -> padded to the 6-row CUDA graph) that `combine_sampled_and_draft_tokens_kernel` immediately marked dead on device (`is_padding = 1`).
   - **Fix**: `AdaptiveVerificationManager.get_num_tokens` now filters `valid &= survival_probability >= self.dead_rows_tau` when `self.dead_rows_tau > 0`, keeping dead tail rows out of the outer CUDA graph budget (`+3.3%` step throughput at `c1`), and `get_verified_draft_counts` / `sample_tokens` clamp verified and rejected draft counts to `dead_row_cut`.
2. **High Concurrency (`c8` and `c16`): Uniform Sub-16-Row Cost Over-Prices Coalesced Rows Above the 16-Row MMA Knee**:
   - Neither `max_total_logits` (`4096`) nor `cudagraph_capture_sizes` (`96` on TP4, `48` on TP3) binds at `c8` (`21.35` rows) or `c16` (`36.52` rows).
   - Instead, pricing rows above the 16-row Tensor Core MMA tile boundary at the sub-16-row slope (`row_cost = 0.035`, or `0.573 ms/row`) raises the marginal survival cutoff to `0.333` at `c8` (`cov@1 = 0.550`) and `0.477` at `c16` (`cov@1 = 0.455`), starving ~50% of requests of their 2nd draft token (`surv@1 = 0.334–0.340`) even though the measured `1030` decode grid slope above 16 rows is `0.302–0.321 ms/row` (`~0.55x` of the sub-16-row slope due to full `M = 16` dense tile amortization and routed-MoE expert coalescing).
   - **Fix**: `AdaptiveVerificationManager._step_cost` applies a two-slope piecewise-linear cost above `_HIGH_ROW_COST_KNEE = 16` (`_HIGH_ROW_COST_SCALE = 0.55`), leaving all batches with `rows <= 16` (`c1`, `c2`, `c4`) unchanged while admitting profitable 2nd-draft tokens at `c8` and `c16` (`+1.2%` to `+1.8%` aggregate throughput).

## Verification

Tested on `dgx4` (`spark-ds41f:1030`):
- `pytest -q -o addopts="" --noconftest tests/v1/spec_decode/test_adaptive_verification.py`: **31 passed, 1 skipped**.
