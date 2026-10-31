# Post-Gate Auxiliary-Stream Shared Expert Launch for DeepSeek-V4.1 Decode

## Hypothesis & Mechanism

In `MoERunner._apply_quant_method` (`vllm/model_executor/layers/fused_moe/runner/moe_runner.py`), when `fold_shared` is active with `SharedExpertsOrder.MULTI_STREAM_OVERLAPPED`, `self._shared_experts.maybe_forward_async(shared_experts_input, shared_workspace)` was previously launched on `aux_stream` *before* `self.router.select_experts(...)`.

For DeepSeek-V4.1 (`TileLangGateRouter`, `routes_hidden_states = True`), `select_experts` runs the split-K BF16 router gate projection (`partials_kernel`, `384 x 5120`, `3.93 MB` weight read) followed by `route_partials` (`sqrtsoftplus` + top-k routing). At single-tile decode batch sizes (`M <= 32`), `partials_kernel` launches only `1 * (384 / 64) * 4 = 24` CTAs on the main stream. Recording `_input_ready_event` on the main stream before `partials_kernel` allows the GPU block scheduler to co-schedule `aux_stream`'s `shared_experts.gate_up_proj` (`5.90 MB` FP8 weight read on TP4) onto the remaining SMs simultaneously with `partials_kernel` and the tail of the `FFN` L2 prefetch (`_l2pf_ffn`). This 3-way L2/DRAM bandwidth contention slows down the critical-path router gate GEMM, while leaving the subsequent zero-weight-traffic window (`route_partials` + `moe_dispatch`, `~10 µs`) underutilized.

By deferring `maybe_forward_async` until immediately after `partials_kernel` (via a `_post_gate_hook` consumed inside `TileLangGateRouter._compute_routing` when `rows <= 32`):
1. `partials_kernel` (`self.gate`) executes on the main stream without L2/DRAM competition from `aux_stream`, while `_l2pf_ffn` finishes populating `shared_experts.gate_up_proj` and `down_proj` into L2.
2. `_input_ready_event` is recorded on the main stream immediately after `partials_kernel`, so `aux_stream` begins `shared_experts.gate_up_proj` concurrently with `route_partials` (`~5 µs` zero-memory-traffic window) and `moe_dispatch` (`~5 µs`), and finishes `down_proj` well inside `routed MoE` (`FC1` + `SwiGLU` + `FC2`, `~140 µs`) before `moe_combine` awaits `_output_ready_event`.

## Verification & Benchmark Results (`dgx4`, `spark-ds41f:19f2c20ed4d6-next-20`)

### 1. Bit-Identity & Unit / Kernel Tests
- `pytest -q --noconftest -p tests.kernels.deepseek_v41_pdl_sources tests/models/test_deepseek_v4_1_l2_schedule.py tests/kernels/moe/test_deepseek_v41_tilelang_router.py`:
  - **86 passed**, `PDL sources: 30 kernels launched, 19 with PDL, 0 violations`.
- 40-layer end-to-end chain bit-identity (`torch.equal` against eager and `pertype2` baseline):
  - `"equal": true` (`0` differing tensors across all 40 layers at `rows = 6, 48, 96`).

### 2. 40-Layer Decode Step Latency (`tp=4`, `rows=6`, 6 Rotating Rounds x 10 CUDA Graph Replays)

| Schedule | Median 40-Layer Step (`µs`) | Delta vs `pertype2` (`rc20`) | Per-Round Deltas |
| --- | ---: | ---: | --- |
| `pertype2` (`rc20`: `aux_stream` before `gate`) | `27,531.4 µs` | baseline | `[0.0%, 0.0%, 0.0%, 0.0%, 0.0%, 0.0%]` |
| `ffn_aux_after_topk` (`aux_stream` after `route_partials`) | `28,551.1 µs` | `-0.14%` | `[-1.09%, -0.09%, -0.12%, -1.46%, +0.45%, +0.16%]` |
| **`ffn_aux_after_gate` (`aux_stream` right after `gate`)** | **`27,282.8 µs`** | **`-0.90%` (`-248.6 µs`/step)** | **`[-1.09%, -0.76%, -0.84%, -1.12%, -1.00%, -1.08%]`** |
