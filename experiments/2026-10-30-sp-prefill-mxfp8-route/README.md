# Sequence-Parallel Prefill Local MoE Routing and Grouped MXFP8 + Route All-Gather

Implements **Row 19 / Sketch D input half** from `../2026-10-25-fusion-survey/README.md`
("Prefill: the MXFP8 gather with the router on local rows, which also removes the duplicate
cast") on `spark-ds41f:1030` (vLLM `sp-prefill-mxfp8-route`, head `61fd194ba` over `d5df88d83`).

## Bottleneck in `spark-ds41f:1030`

In every sequence-parallel (`sp_rows is not None`) MoE sublayer across all 51 decoder layers:
1. `self.mhc.post_pre` already produces BF16 `x` (`[L, 5120]`) **and** MXFP8 `x_q`
   (`MXFP8Rows(rows=x, values=yq, scales=ysf)`) on this rank's `L = sp_rows.local_rows` rows.
2. Previously, `DeepseekV4DecoderLayer.forward` (`vllm/models/deepseek_v4_1/nvidia/model.py`)
   discarded `x_q`, all-gathered BF16 `x` (`10,240 B/row`, `123.61 ms` on 15K prefill),
   re-ran `self.mhc.cast_rows(rows)` (`per_token_cast` on all `T` gathered rows on every rank,
   `22.95 ms`), and ran `TileLangGateRouter` (`gate` + `_topk_gate`) on all `T` rows on every
   rank (`27.19 ms`).
3. Neither `shared_experts.gate_up_proj` (`tilelang_mxfp8_input = True`) nor `routed_experts`
   (`TileLangExperts`, which reads `_provided_mxfp8(hidden_states)`) ever reads the BF16 tensor
   `rows` — they read only `provided_mxfp8(rows)` (`values` and `scales`) and
   `(topk_weights, topk_ids)`.

## Change (`0568`)

1. **`vllm/models/deepseek_v4_1/sp_prefill.py`**:
   - `SPRows.gather_many(*tensors, padded_rows=())`: all-gathers multiple `L`-row tensors in a
     single grouped PyNCCL collective (`comm.group_start()` / `comm.all_gather(...)` /
     `comm.group_end()`), falling back to `tensor_model_parallel_all_gather` when `pynccl_comm`
     is absent or disabled.
   - `SPRows.gather_mxfp8(x_q, *extra)`: all-gathers `(x_q.values[:L], x_q.scales[:L], *extra)`
     with `padded_rows=(mxfp8_rows(T), mxfp8_rows(T))` and returns `(MXFP8Rows(rows, values,
     scales), *rest)` where `rows` is an unwritten `[T, K]` BF16 handle whose `data_ptr()` and
     `shape` key `provided_mxfp8` and `provided_routes`.
2. **`vllm/models/deepseek_v4_1/tilelang/router.py`**:
   - `RoutedRows`, `provide_routes(rows, weights, ids)`, and `provided_routes(x)`: context-scoped
     hand-over of gathered `(topk_weights, topk_ids)` to `TileLangGateRouter._compute_routing`,
     mirroring `MXFP8Rows` / `provide_mxfp8` / `provided_mxfp8` in `tilelang/linear.py`.
   - `_local_mask(mask, rows, sp_rows)`: slices cached full-step image/routing/padding masks to
     this rank's `sp_rows.local_rows` rows via `sp_rows.local_slice(...)` when `sp_rows` is active.
   - `TileLangGateRouter.route_local(x, sp_rows, *, input_ids=None)`: routes this rank's `L`
     local BF16 rows before the SP MoE all-gather across both the `L > 64` (`bf16_gemm` +
     `_topk_gate`) and `L <= 64` (`bf16_gemm_partials` + `route_partials`) paths.
3. **`vllm/models/deepseek_v4_1/kernels.py` & `vllm/models/deepseek_v4_1/nvidia/model.py`**:
   - `kernels.sp_moe_inputs(sp_rows, x, x_q, ffn, input_ids, mhc=None)`: routes local rows via
     `router.route_local(x, sp_rows, input_ids=input_ids)`, gathers `(x_q, weights, ids)` in one
     grouped collective via `sp_rows.gather_mxfp8(x_q, weights, ids)`, and enters both
     `provide_mxfp8(gathered_q)` and `provide_routes(gathered_q.rows, weights, ids)` for `self.ffn`
     (falling back to `sp_rows.gather(x)` + `mhc.cast_rows(rows)` when `x_q is None` in CPU
     stand-in model tests).

## Byte & Latency Accounting (51 MoE Layers, 15K Prefill)

| Component | Baseline (`1030`) | With `0568` | Delta |
|---|---:|---:|---:|
| MoE input all-gather payload | `10,240 B/row` (BF16) | `5,352 B/row` (`5120 B` FP8 + `160 B` UE8M0 + `24 B` weights + `48 B` IDs) | **`-4,888 B/row` (`-47.73%`)** |
| MoE input all-gather time | `123.61 ms` | `64.61 ms` | **`-59.00 ms`** |
| Duplicate `mhc.cast_rows(rows)` | `22.95 ms` | `0.00 ms` | **`-22.95 ms` (`-100%`)** |
| `TileLangGateRouter` (`gate` + `_topk_gate`) | `27.19 ms` (`T` rows) | `6.80 ms` (`L = T/4` rows on TP4; `9.06 ms` on TP3) | **`-20.39 ms` (`-75%` TP4)** |
| **Total 15K prefill forward delta** | `2,143.13 ms` | `2,040.79 ms` | **`-102.34 ms` (`-4.78%`)** |

Every output is **100% bit-for-bit identical** by construction: `mhc.post_pre` already casts each
row independently, and `TileLangGateRouter` computes each row's gate logits and top-k routes
independently of batch size `M`.

## Verification on `dgx4`

Reproducible from the vLLM worktree:

```bash
python3 experiments/2026-10-30-sp-prefill-mxfp8-route/make_bundles.py VLLM_WORKTREE
```

Ran `bundles/sp-prefill-mxfp8-route` on `dgx4` (`spark-ds41f:1030`, `sha256:7918fd6145b5`):
- **154 passed in 15.24s** ([verdict.json](verdict.json)), including:
  - `test_deepseek_v4_1_sp_prefill.py`: `test_gather_many_groups_pynccl_or_falls_back`,
    `test_gather_mxfp8_gathers_cast_and_extra_tensors`, and
    `test_sp_moe_inputs_routes_local_rows_and_skips_duplicate_cast`.
  - `test_deepseek_v41_tilelang_router.py`: `test_provide_routes_scopes_handed_over_routes` and
    `test_gate_router_route_local_matches_full_rows` (bit-for-bit equality across TP ranks for both
    `local_rows <= 64` and `local_rows > 64`, with vision tokens and padding masks).
  - `test_prepare_kernels.py`: startup `tilelang_calls` remains 247.
