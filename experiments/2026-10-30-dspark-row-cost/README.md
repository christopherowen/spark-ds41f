# Piecewise CUDA-Graph / GEMM-Tile / Dead-Row Pricing in `AdaptiveVerificationManager`

## Why 7.2% and 3.9% Tail Slots Are Admitted Under Linear `row_cost`

In `AdaptiveVerificationManager.get_num_tokens`, candidate draft budgets $k \in \{0, \dots, M\}$
previously maximised expected tokens over a flat linear denominator:

$$\text{utility}(k) = \frac{\text{num\_sampling\_requests} + \sum_{j=1}^k s_j}{1 + \text{row\_cost} \cdot (\text{non\_draft\_rows} + k)}$$

This linear rule has three structural blind spots:

1. **Ignoring CUDA-graph capture padding (`[1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 72, 80, 88, 96]`)**:
   - At `c1` (`non_draft_rows = 1`), admitting a 4th draft (`rows = 5`) leaves the 4-row CUDA graph
     and pads up to the **6-row CUDA graph** (there is no 5-row graph in `capture_sizes`). In
     `manifests/suites/2026-10-30-spark-ds41f-1030-tp4-single-stream/c1.json`, `coverage_at_3 = 0.3208`
     and `coverage_at_4 = 0.1625`, so `32.08%` of all `c1` steps jump into the 6-row CUDA graph
     (`15.83%` verify 5 rows inside a 6-row graph), even though position 3 (`survival_at_3 = 0.0720`)
     and position 4 (`survival_at_4 = 0.0386`) yield only `0.072` and `0.039` tokens per step.
   - At batched concurrencies in `2026-10-30-spark-ds41f-1030-tp4`, average verified rows spill just
     past capture and tile boundaries: `poisson` averages `17.91` rows (spilling 1.91 rows past `16`
     into the `20`-row graph), `c8` averages `21.35` rows (spilling 1.35 rows past `20` into the
     `24`-row graph), and `c16` averages `36.52` rows (spilling 4.52 rows past `32` into the
     `40`-row graph).
2. **Ignoring 16-row dense GEMM tile and 64-row LM-head tile boundaries**:
   - In the measured `static-r6v-25x3.json` verification curve, crossing row `16 -> 20`, `32 -> 40`,
     or `48 -> 56` starts a new 16-row dense GEMM tile (`+0.6 ms`, or `1.0` row-equivalent), and
     crossing row `64 -> 72` starts both a new 16-row GEMM tile and a second 64-row LM-head tile
     (`+1.2 ms` total, or `2.0` row-equivalents).
3. **Host/device mismatch with `SPARK_DS41F_DSPARK_DEAD_ROWS_TAU`**:
   - On the GPU (`combine_sampled_and_draft_tokens` and `tilelang_rejection`), any draft whose
     survival falls below `dead_rows_tau` is marked `is_padding = 1` (skipping routed MoE) and
     truncated at `dead_row_cut` (`verify` and `resample` end the block at `cut` by replacing drafts
     from `cut` on with `-1` placeholders).
   - On the CPU (`get_num_tokens`), however, the linear threshold at `c1` is only `4.2%` at
     `row_cost = 0.02` (`7.0%` at `row_cost = 0.035`). Thus the host admits slots with survival
     `0.07..0.20` — bumping `c1` into the 6-row CUDA graph — only for the GPU to truncate them at
     `dead_row_cut`!

### Direct Empirical Proof from `2026-10-30-spark-ds41f-1030-tp4`

Because `c1`, `c8`, and `c16` run on the identical 256-prompt workload (`prompts_sha256 = f5efd3d7a...`),
comparing `(coverage_at_i, survival_at_i)` between `c8` and `c1` directly measures the marginal yield
$\Delta \text{survival}_i / \Delta \text{coverage}_i$ of the extra `+1.11` draft rows/req that `c1`
admits over `c8`:

| Suite | Position | `c16` `(cov, surv)` | `c8` `(cov, surv)` | `c1` `(cov, surv)` | $\Delta\text{cov}$ (`c8 -> c1`) | $\Delta\text{surv}$ (`c8 -> c1`) | Marginal Yield |
| --- | :---: | :---: | :---: | :---: | ---: | ---: | ---: |
| Greedy | pos 1 | `(0.455, 0.3345)` | `(0.550, 0.3386)` | `(0.959, 0.3548)` | `+0.410` | `+0.0163` | **3.97%** |
| Greedy | pos 2 | `(0.195, 0.1466)` | `(0.246, 0.1508)` | `(0.635, 0.1569)` | `+0.389` | `+0.0061` | **1.58%** |
| Greedy | pos 3 | `(0.092, 0.0695)` | `(0.113, 0.0723)` | `(0.321, 0.0720)` | `+0.208` | `-0.0003` | **-0.15%** |
| Greedy | pos 4 | `(0.048, 0.0378)` | `(0.058, 0.0383)` | `(0.163, 0.0386)` | `+0.104` | `+0.0003` | **0.28%** |
| Sampled | pos 1 | `(0.445, 0.3180)` | `(0.542, 0.3234)` | `(0.960, 0.3303)` | `+0.418` | `+0.0069` | **1.66%** |
| Sampled | pos 2 | `(0.184, 0.1333)` | `(0.238, 0.1395)` | `(0.623, 0.1396)` | `+0.386` | `+0.0001` | **0.03%** |
| Sampled | pos 3 | `(0.084, 0.0618)` | `(0.106, 0.0633)` | `(0.308, 0.0640)` | `+0.202` | `+0.0006` | **0.30%** |
| Sampled | pos 4 | `(0.043, 0.0318)` | `(0.052, 0.0327)` | `(0.157, 0.0330)` | `+0.104` | `+0.0003` | **0.30%** |

The extra `1.11` verified rows/req at `c1` over `c8` yield **0.0% to 3.97% accepted tokens per row**
while pushing `32.1%` of `c1` steps into the 6-row CUDA graph.

---

## Changes

1. **Piecewise Step-Cost Lookup (`overlay/vllm/v1/worker/gpu/spec_decode/adaptive_verification.py`)**:
   - `warm_tail_shapes` records `self._capture_sizes = tuple(sorted(capture_sizes))` from
     `cudagraph_manager.captured_token_counts()`.
   - `AdaptiveVerificationManager.step_costs(rows, scores)` replaces `1 + row_cost * rows` with:
     $$\text{cost}(k) = \underbrace{1 + \text{row\_cost} \cdot \Big( 0.5 \cdot P_k + \lfloor (P_k - 1)/16 \rfloor + \lfloor (P_k - 1)/64 \rfloor \Big)}_{\text{step\_cost}[\text{capture\_size}(\text{rows}_k)]} + \underbrace{0.5 \cdot \text{row\_cost} \cdot L_k}_{\text{moe\_row\_cost} \cdot \text{live\_rows}_k}$$
     where $P_k = \text{capture\_size}(R_k)$ is the padded CUDA-graph token count and
     $L_k = \text{non\_draft\_rows} + \sum_{j=1}^k \mathbb{I}(s_j \ge \text{dead\_rows\_tau})$ is the
     live routed-MoE row count.
   - In `get_num_tokens`, draft slots with $s_j < \text{dead\_rows\_tau}$ contribute zero live expected
     tokens (`np.where(scores >= dead_rows_tau, scores, 0.0)`), matching device-side `dead_row_cut`
     truncation so dead rows never push the batch into a larger CUDA graph or GEMM tile.
   - When `_capture_sizes` is empty and `dead_rows_tau <= 0`, `step_costs` reduces identically to
     `1.0 + row_cost * rows`.
2. **Tuned `adaptive_verification_row_cost = 0.045` and `SPARK_DS41F_DSPARK_DEAD_ROWS_TAU = 0.25`**:
   - At `row_cost = 0.045`, the linear `c1` 4th-draft threshold rises from `4.2%` (at `0.02`) to
     `8.7%` (`9.2%` at 5th draft), and the piecewise `4 -> 6` CUDA-graph jump threshold rises to
     `13.0%`. Combined with `dead_rows_tau = 0.25`, `c1` prunes the `7.2%` and `3.9%` tail slots
     before graph selection and stays in the 3- or 4-row capture graph.

---

## Replayed Trace Summary (`row_cost_policy.py` -> `results.json`)

| Workload | Baseline `ver_rows` (`pad`) | Piecewise `0.045 / 0.25` `ver_rows` (`pad`) | Baseline Row Yield | Piecewise Row Yield | Yield Change | Net Decode `tok/s` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `single-stream/c1` | `4.08` (`4.28`) | `2.67` (`2.71`) | `0.350` | `0.639` | **+82.3%** | **+2.93%** |
| `release/poisson` | `15.79` (`17.30`) | `15.24` (`15.60`) | `0.657` | `0.668` | **+1.6%** | **+1.35%** |
| `release/c8` | `18.23` (`19.66`) | `17.11` (`17.19`) | `0.668` | `0.689` | **+3.2%** | **+1.67%** |
| `release/c16` | `32.75` (`35.97`) | `30.93` (`30.93`) | `0.687` | `0.705` | **+2.7%** | **+3.14%** |
| `sampled/c1` | `4.08` (`4.28`) | `2.71` (`2.75`) | `0.349` | `0.621` | **+77.8%** | **+2.84%** |
| `sampled/c8` | `18.39` (`19.88`) | `17.22` (`17.28`) | `0.646` | `0.666` | **+3.2%** | **+1.71%** |
| `sampled/c16` | `32.34` (`35.50`) | `30.78` (`30.78`) | `0.663` | `0.678` | **+2.3%** | **+3.09%** |
