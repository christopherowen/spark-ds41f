# Eliminate decode-step redundant H2D transfers, UVA copies, and eager ATen ops (`2026-11-08-async-output-fastpath`)

- **Date**: 2026-11-08
- **Base deployment commit**: `cc20ed18b6ca6521b602f09cb3ba326774166d1c` (Candidate `1107`, `experiments/2026-10-26-spark-ds41f-candidate/1107`)
- **Base vLLM commit**: `320021e3c4dbb202d911ab70e2cd924c75420488` (tree `ed1bdb7a7684da33d403a055521ac0c4dd7ca670`, `0001..0761`)
- **Candidate vLLM commit**: `20c98385008ad79795b3cdeda80e0ba4217df73b` (tree `8c2ddc0a71e7b52cfd47725a469ffb28d7affa93`, `0001..0762`, `patchset_sha256` `bb674f7ab493ccd4f41434fe37677c8fc41a0532c17b8865294cc3e2df386211`)
- **Intended causal variable**: Eliminate redundant per-step host-to-device DMA transfers (`async_tensor_h2d`), unconditional UVA buffer rotations (`apply_staged_writes`), and eager PyTorch ATen kernel launches on `main_stream` before `speculator.propose()` (`0762-perf-worker-eliminate-decode-step-h2d-and-aten-overhead.patch`).

## 1. Audit of `GPUModelRunner` Per-Step Decode Critical Path in `1107`

Auditing `vllm/v1/worker/gpu/model_runner.py`, `vllm/v1/worker/gpu/block_table.py`, `vllm/v1/worker/gpu/spec_decode/speculator.py`, and `vllm/v1/worker/gpu/spec_decode/acceptance_estimator.py` on `spark-ds41f:1107` identified the following:

1. **Event-Scoped `AsyncOutput` Placement**:
   - In `1107` (`vllm/v1/worker/gpu/model_runner.py:3491-3501`), `AsyncOutput` is already constructed immediately after `self.sample(...)` (before `postprocess_sampled` and before `self.speculator.propose`), recording `copy_event` on `main_stream` so the D2H readback of `sampled_token_ids` and `num_sampled_tokens` on `output_copy_stream` overlaps with `postprocess_sampled` and the 5-step DSpark draft loop. Similarly, `CopyDrafter.prepare_alloc` and `CopyDrafter.attach_async_output` (`0744`) already attach the next-step copy-draft allocation D2H readback before `self.speculator.propose(...)`.
2. **Unconditional `self.sampler.apply_staged_writes()` on Every Decode Step (`model_runner.py:1653-1654`)**:
   - In `GPUModelRunner.add_requests()`, `if self.sampler is not None: self.sampler.apply_staged_writes()` was left outside the `if scheduler_output.scheduled_new_reqs:` guard.
   - On every steady-state decode step (`scheduled_new_reqs == []`), this rotated and copied **15 `UvaBackedTensor`s** (`temperature`, `top_p`, `top_k`, `min_p`, `seeds`, `num_allowed_token_ids`, `repetition_penalty`, `frequency_penalty`, `presence_penalty`, `num_logit_bias`, `num_bad_words`, `num_token_ids`, `min_lens`, `num_stop_token_ids`, `restore_when_all_masked`) and checked 7 `StagedWriteTensor.apply_write()` calls even though `sampler.add_request()` is only called inside `for new_req_data in scheduler_output.scheduled_new_reqs:` (and `set_forced_thinking_end` already calls `self.thinking_budget_state.apply_staged_writes()` directly when changed).
3. **Unconditional `self.block_tables.apply_staged_writes()` on Non-Allocating Decode Steps (`model_runner.py:2920`, `block_table.py:152-187`)**:
   - On the vast majority of decode steps where no active request crosses a 256-token KV block boundary (`overwrite=False` and `new_block_ids` empty across all KV groups), `BlockTables.append_block_ids()` staged empty writes and `BlockTables.apply_staged_writes()` unconditionally invoked `fused_writer.apply(...)` and `self.num_blocks.copy_to_uva()`.
4. **Two Redundant `async_tensor_h2d` DMA Transfers on Every Adaptive-Verification Decode Step (`model_runner.py:1833-1863`)**:
   - When `adaptive_verification` is active (always enabled in `config/cluster.json`), `GPUModelRunner.prepare_inputs()` executed:
     - `cu_num_logits = async_tensor_h2d(cu_num_logits_np, device=self.device)` at line 1837 (allocating a pinned host buffer, a GPU tensor, and launching an H2D `cudaMemcpyAsync`), which was immediately overwritten at line 1873 by `adaptive_verification.reallocate_drafts()` returning `self._cu_num_logits[: num_reqs + 1]` without ever reading the H2D-copied tensor.
     - `async_tensor_h2d(query_start_loc_np, out=query_start_loc)` at line 1862 (allocating a pinned host buffer for `max_num_reqs + 1` ints and launching an H2D `cudaMemcpyAsync` into `self.input_buffers.query_start_loc`), which was immediately overwritten on-device at line 1874 by `tilelang_inputs.adaptive_offsets(...)` inside `adaptive_verification.reallocate_drafts()`.
5. **9 Redundant Eager ATen Kernel Launches & 8 Temporary GPU Allocations on `main_stream` Before `speculator.propose()` (`model_runner.py:3478-3675`, `acceptance_estimator.py:30-98, 497-530`, `speculator.py:509-552`)**:
   - In `sample_tokens()` before `self.speculator.propose(...)`:
     - Clamping `num_rejected` by `self._dead_row_cut` was performed twice — once in `model_runner.py:3596-3605` and again in `OnlineAcceptanceEstimator.step()` (`acceptance_estimator.py:506-509`) — launching 5 eager ATen kernels (`sub.Scalar`, `clamp_min`, `sub.Tensor`, `clamp_min`, `minimum`) and 5 temporary GPU allocations on `main_stream`.
     - Masking `CopyDrafter` slots in `model_runner.py:3615-3619` (`torch.where(self.copy_drafter.int_state[input_batch.idx_mapping, 1] > 0, -1, observed_idx_mapping)`) launched 3 eager ATen kernels (`index`, `gt.Scalar`, `where.ScalarSelf`) and 3 temporary GPU allocations on `main_stream`.
     - `self.adaptive_verification.get_verified_draft_counts(input_batch.num_reqs, self._dead_row_cut)` was called a second time at `model_runner.py:3655-3662` before `self.copy_drafter.prepare_alloc(...)`, launching a duplicate `torch.minimum` kernel on `main_stream` even though `num_verified_draft_tokens` was already computed at line 3478.
     - In `DraftModelSpeculator._copy_request_inputs()` (`speculator.py:551`), `self.idx_mapping[num_reqs:].fill_(-1)` was unguarded, dispatching a 0-element `fill_` kernel when `num_reqs == self.max_num_reqs`, and in `model_runner.py:3748`, `self.req_states.draft_tokens[input_batch.idx_mapping, :0]` dispatched an `aten::index` kernel for a 0-width slice.

## 2. Changes (`0762-perf-worker-eliminate-decode-step-h2d-and-aten-overhead.patch`)

1. **`GPUModelRunner.add_requests()` (`vllm/v1/worker/gpu/model_runner.py`)**:
   - Indent `if self.sampler is not None: self.sampler.apply_staged_writes()` under `if scheduler_output.scheduled_new_reqs:`, skipping 15 `UvaBackedTensor.copy_to_uva()` rotations and 7 `StagedWriteTensor.apply_write()` checks on decode steps with no new requests.
2. **`BlockTables` Dirty Tracking (`vllm/v1/worker/gpu/block_table.py`)**:
   - Add `self._dirty = False` to `BlockTables.__init__`. In `append_block_ids()`, skip groups where `not overwrite and not block_ids` and set `self._dirty = True` only when a write is staged. In `apply_staged_writes()`, return early when `not self._dirty` (clearing `self._dirty = False`), eliminating `fused_writer.apply()` and `num_blocks.copy_to_uva()` on non-allocating decode steps.
3. **`GPUModelRunner.prepare_inputs()` (`vllm/v1/worker/gpu/model_runner.py`)**:
   - Guard `cu_num_logits = async_tensor_h2d(cu_num_logits_np, device=self.device)` and `async_tensor_h2d(query_start_loc_np, out=query_start_loc)` with `if adaptive_verification is None:`, and allocate `query_start_loc_np` to `num_reqs_padded + 1` instead of `self.max_num_reqs + 1` when `adaptive_verification is not None`.
4. **Fused `_accumulate_kernel` & Deduped Post-Sample Path (`vllm/v1/worker/gpu/spec_decode/acceptance_estimator.py`, `speculator.py`, `model_runner.py`)**:
   - Fuse optional `dead_row_cut_ptr` (`HAS_DEAD_ROW_CUT`) and `copy_int_state_ptr` (`HAS_COPY_STATE`) directly into `OnlineAcceptanceEstimator._accumulate_kernel` (and `_step_cpu`), passing `dead_row_cut=self._dead_row_cut` and `copy_int_state=self.copy_drafter.int_state` through `DraftModelSpeculator.observe_verification()`.
   - Reuse `verified = num_verified_draft_tokens` in `sample_tokens()` instead of calling `get_verified_draft_counts()` a second time.
   - Guard `self.idx_mapping[num_reqs:].fill_(-1)` with `if num_reqs < self.max_num_reqs:` in `DraftModelSpeculator._copy_request_inputs()`, and replace `self.req_states.draft_tokens[input_batch.idx_mapping, :0]` with the zero-cost view `self.req_states.draft_tokens[: input_batch.num_reqs, :0]`.

## 3. Exact Verification & Benchmark Commands (`sp5`, `NVIDIA GB10`)

### 3.1 Two-Boot Unit Test Suite (`sp5`)

```bash
flock /tmp/gpu.lock docker run --rm --gpus all --ipc=host \
  -w /opt/spark-ds41f/candidate/vllm \
  --entrypoint pytest spark-ds41f:1107 \
  --noconftest -k "not test_profile_run_releases_generic_outputs_before_deepseek_profile and not test_moe_output_buffer_is_reused_and_disjoint_from_workspace_scratch" \
  tests/v1/spec_decode/test_acceptance_estimator.py \
  tests/v1/spec_decode/test_adaptive_verification.py \
  tests/v1/spec_decode/test_copy_drafts.py \
  tests/v1/worker/test_gpu_block_table.py \
  tests/v1/worker/test_gpu_model_runner_v2.py \
  tests/v1/worker/test_gpu_input_batch_v2.py -q
```

- `receipts/gpu-pytest-boot1.txt`: **`219 passed, 1 skipped in 63.04s`**
- `receipts/gpu-pytest-boot2.txt`: **`219 passed, 1 skipped in 63.09s`**

### 3.2 Two-Boot Step-Overhead Microbenchmark (`sp5`)

```bash
flock /tmp/gpu.lock docker run --rm --gpus all --ipc=host \
  -w /opt/spark-ds41f/candidate/vllm \
  --entrypoint python3 spark-ds41f:1107 \
  bench_step_fastpath.py --boot 1 --output receipts/bench-boot1.json
```

Measured across 2 independent container boots (`boot1` and `boot2`, 50 warmup iterations + 300 timed iterations per concurrency level `c1`, `c8`, `c16` on `sp5` `NVIDIA GB10`):

| Concurrency | Stage | Baseline `boot1` / `boot2` (`µs`) | Candidate `boot1` / `boot2` (`µs`) | Mean Delta (`µs`) | Speedup | Bit-Exact |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **`c1`** | `staged_writes` (`add_requests` + `block_tables`) | `28.16` / `27.85` | `2.72` / `2.68` | **`-25.31 µs`** | **`10.37x`** | `true` |
| **`c1`** | `prepare_inputs` (adaptive verification H2D + offsets) | `96.78` / `98.22` | `73.47` / `74.24` | **`-23.65 µs`** | **`1.32x`** | `true` |
| **`c1`** | `post_sample` (`observe_verification` + `verified_counts` + copy inputs) | `108.49` / `108.24` | `28.76` / `28.56` | **`-79.71 µs`** | **`3.78x`** | `true` |
| **`c1`** | **Total step overhead (`mean` / `p95`)** | **`233.43` / `234.31`** (`p95`: `249.28` / `252.34`) | **`104.96` / `105.47`** (`p95`: `109.22` / `109.23`) | **`-128.66 µs`** | **`2.22x`** | `true` |
| **`c1`** | **Pipelined step overhead (`mean` / `p95`)** | **`226.33` / `223.75`** (`p95`: `244.78` / `234.88`) | **`98.85` / `98.65`** (`p95`: `102.14` / `102.42`) | **`-126.29 µs`** | **`2.28x`** | `true` |
| **`c8`** | `staged_writes` (`add_requests` + `block_tables`) | `31.28` / `30.96` | `3.61` / `3.71` | **`-27.46 µs`** | **`8.50x`** | `true` |
| **`c8`** | `prepare_inputs` (adaptive verification H2D + offsets) | `101.69` / `102.14` | `76.86` / `76.99` | **`-24.99 µs`** | **`1.32x`** | `true` |
| **`c8`** | `post_sample` (`observe_verification` + `verified_counts` + copy inputs) | `113.00` / `110.98` | `29.19` / `28.86` | **`-82.97 µs`** | **`3.86x`** | `true` |
| **`c8`** | **Total step overhead (`mean` / `p95`)** | **`245.98` / `244.07`** (`p95`: `264.42` / `261.87`) | **`109.66` / `109.56`** (`p95`: `113.54` / `113.10`) | **`-135.42 µs`** | **`2.24x`** | `true` |
| **`c8`** | **Pipelined step overhead (`mean` / `p95`)** | **`234.05` / `234.65`** (`p95`: `241.81` / `244.50`) | **`103.63` / `103.12`** (`p95`: `106.96` / `106.32`) | **`-130.98 µs`** | **`2.27x`** | `true` |
| **`c16`** | `staged_writes` (`add_requests` + `block_tables`) | `34.21` / `34.47` | `4.76` / `4.88` | **`-29.52 µs`** | **`7.12x`** | `true` |
| **`c16`** | `prepare_inputs` (adaptive verification H2D + offsets) | `104.66` / `107.54` | `78.99` / `81.14` | **`-26.04 µs`** | **`1.33x`** | `true` |
| **`c16`** | `post_sample` (`observe_verification` + `verified_counts` + copy inputs) | `114.45` / `112.47` | `29.69` / `29.05` | **`-84.09 µs`** | **`3.86x`** | `true` |
| **`c16`** | **Total step overhead (`mean` / `p95`)** | **`253.31` / `254.47`** (`p95`: `272.51` / `271.30`) | **`113.44` / `115.07`** (`p95`: `117.09` / `120.51`) | **`-139.64 µs`** | **`2.22x`** | `true` |
| **`c16`** | **Pipelined step overhead (`mean` / `p95`)** | **`240.41` / `245.32`** (`p95`: `248.35` / `260.75`) | **`106.66` / `108.72`** (`p95`: `110.37` / `112.82`) | **`-135.18 µs`** | **`2.26x`** | `true` |

## 4. Memory Observations & Conclusion

- **Memory**: Eliminates 2 pinned host buffer allocations (`cu_num_logits_np`, `query_start_loc_np`) and 9 temporary GPU tensor allocations (`cu_num_logits` H2D + 8 eager ATen temporaries in `sample_tokens`) per decode step; persistent buffer footprints are unchanged.
- **Conclusion**: Eliminating redundant per-step H2D DMA transfers, unconditional UVA buffer rotations, and eager PyTorch ATen kernel launches before `speculator.propose()` reduces per-step `GPUModelRunner` decode orchestration overhead by **`128.7–139.6 µs/step` (`2.22–2.24x` speedup)** across `c1..c16` with 100% bit-exact outputs across both boots.
