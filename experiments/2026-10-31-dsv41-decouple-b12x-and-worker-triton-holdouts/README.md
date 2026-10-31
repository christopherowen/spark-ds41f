# Decouple `b12x` from the TileLang Backend and Port Remaining V2 Worker Triton Holdouts

## Summary

Completes the remaining Part 1 items of the 100% TileLang migration audit on `rc21` (`a89da720d` / tree `5cfcc5813fc`):

1. **V2 Worker Triton Holdouts -> Native `@tilelang.jit`**:
   - **`post_update_num_computed_tokens`** (`vllm/v1/worker/gpu/tilelang_inputs.py` & `vllm/v1/worker/gpu/input_batch.py`): Ports `_post_update_num_computed_tokens_kernel` to `@tilelang.jit` `post_update_num_computed_tokens_kernel` (`post_update_num_computed_tokens`, `post_update_num_computed_tokens_calls`) and declares it in `GPUModelRunner.declare_step_kernels` when `self.is_pooling_model` is active.
   - **`trace_replay`** (`vllm/v1/worker/gpu/sample/tilelang_sampling.py` & `vllm/v1/worker/gpu/sample/trace_replay.py`): Ports `_trace_replay_kernel` to `@tilelang.jit` `trace_replay_kernel` (`apply_trace_tokens`, `trace_replay_calls`) and registers it in `TraceReplayState.register_warmups()` (`Sampler.register_warmups`).
   - **`compact_sampling_mask`** (`vllm/v1/worker/gpu/sample/tilelang_sampling.py` & `vllm/v1/worker/gpu/sample/output.py`): Ports `_compact_sampling_mask_kernel` to `@tilelang.jit` `compact_sampling_mask_kernel` (`compact_sampling_mask`, `compact_sampling_mask_calls`), computing per-row finite-logit bitmasks, compact kept indices up to `max_num_kept` via warp+block prefix scans, and exact support counts, and declares it in `Sampler.register_warmups` when `return_sampling_mask=True`.

2. **`b12x` Decoupling on the TileLang Backend**:
   - **Sequence-Parallel RoCE Capacity (`sp_prefill.py` & `cuda_communicator.py`)**: Adds `roce_ar_comm` and the `all_reduce_capacity_bytes` property on `CudaCommunicator` so `_prefill_sp_capacity_bytes()` queries the communicator's registered capacity without coupling to `b12x_ar_comm`.
   - **`LogitsProcessor` (`logits_processor.py`)**: Defers `vllm.utils.b12x` imports and `get_b12x_bf16_vocab_projection()` / `get_b12x_packed_bf16_vocab_projection()` calls to when `kernel_config.linear_backend == "b12x"`.
   - **`GPUModelRunner.__init__` (`model_runner.py`)**: Skips `from vllm.utils.b12x import is_b12x_a4_prefill_enabled` when `kernel_config.moe_backend == "tilelang"`, setting `self._b12x_a4_prefill_enabled = False`.
   - **`mxfp4_csf.py`**: Moves `from vllm.model_executor.layers.fused_moe.b12x import B12xExperts` from module top-level into `Mxfp4CsfMoEMethod.process_weights_after_loading`.

## Verification

Verified on `dgx4` (`spark-ds41f:19f2c20ed4d6-next-21`, SM121 GB10):
- `tests/v1/worker/test_gpu_trace_replay.py`, `tests/v1/test_outputs.py`, `tests/v1/worker/test_prepare_kernels.py`, `tests/v1/worker/test_gpu_sampling_states_gather.py`, and `tests/v1/worker/test_gpu_tilelang_inputs.py`: **95 passed**.
