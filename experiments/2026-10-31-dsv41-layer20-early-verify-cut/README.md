# Layer-20 Mid-Network Dead-Row Pruning for Speculative Verification (`dsv41-layer20-early-verify-cut`)

## Summary

Pre-forward dead-row pruning (`SPARK_DS41F_DSPARK_DEAD_ROWS_TAU`) masks speculative draft verification rows whose cumulative drafter confidence $\prod_{m=0}^{j} p_m < \tau_{\text{dead}}$ before Layer 0. However, on single-stream (`c1`, `M = 6`) decode, a draft chain often enters Layer 0 with moderately optimistic prior confidences ($p_m \in [0.55, 0.75]$) even when draft slot $j_0 \in \{1, 2\}$ diverges semantically from the target trajectory. By Layer 20 (`idx == 19`, where `DeepseekV4Model` already materializes the unquantized BF16 `aux_hidden_state` for DSpark), the target model has processed half of its 40 layers (`Layers 0..19`) and the hidden-state trajectory of diverging tail draft slots drops sharply in adjacent-token directional alignment.

Still routing those diverging tail draft rows through `Layers 20..39` forces 20 additional MoE layers to load $8 \times 1.84\text{ MB}$ of unique MXFP4 expert tiles per dead draft row (`~1,460 µs/row` on TP4 and `~1,850 µs/row` on TP3 across 20 layers).

This experiment adds an in-graph, zero-sync Layer-20 mid-network dead-row refinement pass (`mid_layer_dead_rows_kernel`) that:
1. Records each request's first draft row offset `draft_row_base[b]` and prefix log-survival $\log S_{b,j} = \sum_{m=0}^{j} \log p_{b,m}$ inside `combine_sampled_and_draft_tokens_kernel` when `mid_dead_rows=True`.
2. At Layer 20 (`SPARK_DS41F_DSPARK_MID_DEAD_ROWS_LAYER`, default `20`), inspects the already-materialized `[num_tokens, 5120]` BF16 `aux_hidden_state` in a single-CTA-per-request TileLang kernel (`128` threads, deterministic shared-memory tree reduction, `T.pdl_sync()` / `T.pdl_trigger()`), computes adjacent-row cosine similarity $\cos(h_{b,j-1}, h_{b,j})$ across the live prefix $[0, \text{cut}_0)$, and updates the effective log-survival:
   $$\ell^{\text{mid}}_{b,j} = \log S_{b,j} + w_{\cos} \sum_{m=0}^{j} \min\!\bigl(0,\, \cos(h_{b,m-1}, h_{b,m}) - c_{\text{floor}}\bigr)$$
3. Whenever $\ell^{\text{mid}}_{b,j} < \log \tau_{\text{mid}}$, monotonically shrinks `cut_mid < cut_0` and updates in place:
   - `is_padding[base + j] = True` (consumed by `TileLangGateRouter`'s fused decode `route_partials`),
   - `routed[base + j] = False` (consumed by `context._ds41_routing_mask` on the fallback gate path), and
   - `dead_cut[b] = cut_mid` (consumed by `count_tokens` so downstream rejection sampling never commits tokens predicted by mid-network pruned rows).
4. Immediately self-clears `draft_row_base[b] = -1` inside the kernel so dummy/capture passes and subsequent steps with fewer active requests execute as zero-write no-ops inside `FULL` CUDA graphs.

## Exact Target Distribution & Bit-Identity Invariant

- **Causal Attention Invariant:** Layers 20..39 attention continues to run over the fixed CUDA-graph token shape; live prefix rows $r < \text{base} + \text{cut}_{\text{mid}}$ never attend to future rows $r' \ge \text{base} + \text{cut}_{\text{mid}}$ by causal masking.
- **MoE Dispatch Invariant:** Each token row's MoE routing (`route_partials`) and expert combine (`MoeDispatchV41Plan`) operate independently per row. Masking dead tail rows `j >= cut_mid` sets `topk_ids[row] = -1` and `topk_weights[row] = 0.0`, producing **bit-identical** (`torch.equal`) hidden states and logits on all surviving prefix rows $0 \le r \le \text{cut}_{\text{mid}}$.
- **Verification Commit Clamp:** `count_tokens` already clamps committed tokens to `min(num_sampled, dead_cut[b] + 1)`. Because `cut_mid` is updated in `dead_cut[b]` before verification sampling, the first rejected draft position (at or before `cut_mid`) still receives its exact target distribution sample, while no unverified row beyond `cut_mid` can ever be committed.

## Configuration Knobs

- `SPARK_DS41F_DSPARK_MID_DEAD_ROWS_TAU`: Mid-layer effective survival threshold $\tau_{\text{mid}} \in (0, 1)$ (`0` = disabled by default).
- `SPARK_DS41F_DSPARK_MID_DEAD_ROWS_COS_WEIGHT`: Penalty weight $w_{\cos}$ on negative cosine deviation below floor (default `0.5`).
- `SPARK_DS41F_DSPARK_MID_DEAD_ROWS_COS_FLOOR`: Cosine similarity floor $c_{\text{floor}}$ below which directional drop penalizes log-survival (default `0.85`).
- `SPARK_DS41F_DSPARK_MID_DEAD_ROWS_LAYER`: 1-based layer index whose `aux_hidden_state` triggers the mid-network refinement (default `20`).

## Verification & Benchmark Results (`dgx4` / GB10, `rc20`)

### Unit & CUDA-Graph Verification (`pytest --noconftest -p tests.kernels.deepseek_v41_pdl_sources`)
- `tests/v1/spec_decode/test_adaptive_verification.py` + `tests/kernels/moe/test_deepseek_v41_tilelang_router.py`:
  - **`116 passed, 1 skipped`**, **`PDL sources: 33 kernels launched, 21 with PDL, 0 violations`**.
  - Verified bit-identical `topk_ids[:3]` and `topk_weights[:3]` (`torch.equal`) on surviving rows alongside `-1` / `0.0` masking on mid-layer pruned rows `[3..4]` across `FULL` CUDA graph capture and replay.

### GB10 Microbenchmark (`M = 6`, `drafts = 5`, `E = 384`, `top_k = 8`, `K = 5120`, 20 MoE Layers `20..39`)

- **`mid_layer_dead_rows_kernel` CUDA-graph replay overhead (`c1`, `M = 6`, `drafts = 5`):** **`2.322 µs`**

| Topology | Pruned Tail Drafts (`cut_0 - cut_mid`) | Live Rows (`M_live / 6`) | Routed MoE (`µs / layer`) | 20-Layer MoE (`Layers 20..39`) | 20-Layer Gross Saved | Net Step Change (incl. `2.32 µs` kernel) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TP4 (`I = 576`)** | `0` | `6 / 6` | `875.46 µs` | `17,509.1 µs` | `+0.0 µs` | `-2.3 µs` |
| **TP4 (`I = 576`)** | `1` | `5 / 6` | `730.21 µs` | `14,604.1 µs` | `+2,905.0 µs` | **`+2,902.7 µs` (`-16.6%`)** |
| **TP4 (`I = 576`)** | `2` | `4 / 6` | `586.03 µs` | `11,720.5 µs` | `+5,788.6 µs` | **`+5,786.3 µs` (`-33.0%`)** |
| **TP4 (`I = 576`)** | `3` | `3 / 6` | `461.79 µs` | `9,235.8 µs` | `+8,273.3 µs` | **`+8,271.0 µs` (`-47.2%`)** |
| **TP4 (`I = 576`)** | `4` | `2 / 6` | `320.88 µs` | `6,417.5 µs` | `+11,091.6 µs` | **`+11,089.3 µs` (`-63.3%`)** |
| **TP3 (`I = 768`)** | `0` | `6 / 6` | `1,165.31 µs` | `23,306.3 µs` | `+0.0 µs` | `-2.3 µs` |
| **TP3 (`I = 768`)** | `1` | `5 / 6` | `987.92 µs` | `19,758.4 µs` | `+3,547.9 µs` | **`+3,545.5 µs` (`-15.2%`)** |
| **TP3 (`I = 768`)** | `2` | `4 / 6` | `784.25 µs` | `15,684.9 µs` | `+7,621.3 µs` | **`+7,619.0 µs` (`-32.7%`)** |
| **TP3 (`I = 768`)** | `3` | `3 / 6` | `583.01 µs` | `11,660.3 µs` | `+11,646.0 µs` | **`+11,643.7 µs` (`-50.0%`)** |
| **TP3 (`I = 768`)** | `4` | `2 / 6` | `413.89 µs` | `8,277.7 µs` | `+15,028.5 µs` | **`+15,026.2 µs` (`-64.5%`)** |
