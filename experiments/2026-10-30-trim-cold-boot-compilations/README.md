# Trim Unreachable Cold-Boot Kernel Compilations

## Motivation

During cold-boot kernel preparation (`Worker._prepare_kernels`), several groups of mathematically unreachable or dead kernel variants were compiled or warmed:

1. **Unreachable `MoeDispatchV41Plan` variants across target (`384, 6`) and drafter (`128, 3`):**
   - `torch.int32` expert IDs (`16` `moe_dispatch_v41` kernels): `TileKernelsRouter` and `TileLangGateRouter` exclusively emit `torch.int64` expert IDs and raise `ValueError` for any other `indices_type`.
   - `weighted=True` (`w=True`) combine/fold kernels (`6` `moe_combine_v41` kernels): `MoeExpertsV41Plan.run` applies route weights inside `moe_swiglu_cast` before the down-projection and never passes `topk_weights` to `dispatch.combine`.
   - `prefill_dispatch.fused` (`block_M = 64, part = "fused"`): `prefill_dispatch` is only selected when `rows * topk >= block_M * num_experts` (`>= 12,288` routes on target), whereas `fused` only executes when `rows * topk <= _CHUNK` (`512`).
2. **Unreachable drafter MoE (`num_experts = 128, topk = 3`) prefill and multi-chunk kernels:**
   - The drafter forwards at most `max_rows = max_num_seqs * query_rows` (`80` rows / `240` routes on TP4, `40` rows / `120` routes on TP3), so it never reaches `prefill_block_M = 64` (`>= 4,096` routes) or multi-chunk dispatch (`> 512` routes). Bounding `MoeExpertsV41Plan` by `router.max_rows` skips `prefill_gate_up`, `prefill_down`, `prefill_swiglu`, `prefill_dispatch`, and multi-chunk `count`/`layout`/`scatter` dispatches on the drafter (`32` MoE kernels eliminated across target + drafter).
3. **Drafter router gate `(128, 5120)` split-K threshold and gate linear kernels:**
   - Added `(128, 5120): 256` to `BF16_SPLIT_ROWS` so `gate.tilelang_split_rows` matches the target router's `256` rows. On TP4 (`max_rows = 80 <= 256`), `router._gates_every_row()` is now `False`, skipping `TileKernels`'s `moe_topk_gate_forward(topk=3, num_experts=128)` warmup and `routing_masks_kernel`, keeping `65..80`-row draft steps (`c14..c16`) on the fused `route_partials` fast path, and skipping unreachable `bf16_gemm` / `splitk_reduce` compilations on the router gates.
4. **Drafter Markov transition head (`markov_w2`) token tiles:**
   - Set `markov_w2.tilelang_max_rows = max_num_seqs` (`<= 16`), trimming unreachable `block_M = 32` and `64` `bf16_gemm` head projections.
5. **Single-expansion `prepare_declared` -> `registry.warmup()` handoff:**
   - `prepare_declared(registry)` records compiled `TILELANG` callables in `TILELANG._prepared_calls` so `registry.warmup()` only expands and compiles newly registered step kernels.

## Verification

- **DGX4 `scripts/lab.py kernel-local` (`spark-ds41f:1030`):**
  - `tests/models/test_deepseek_v4_1_prepare.py`, `tests/v1/worker/test_prepare_kernels.py`, and `tests/kernels/moe/test_deepseek_v41_tilelang_router.py`: `85 passed in 32.84s` (recorded in `experiments/2026-10-30-trim-cold-boot-compilations/results.json`).
- **Host unit tests:**
  - `python3 -m unittest -v tests/test_trim_cold_boot_compilations.py`: `2 passed`.
  - `python3 bin/spark doctor`: `configuration OK`.
