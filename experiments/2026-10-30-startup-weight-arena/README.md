# Startup weight arena, Order-13 buddy sponge, and post-warmup cache reclamation

On a 4-rank GB10 cluster (`tp4-ring`, 64 KiB base pages, 512 MiB `MAX_PAGE_ORDER=13` buddy blocks),
boot-to-boot serving variability comes from two startup allocation effects on unified LPDDR5x memory:

1. **Physical contiguity of model weights (`Order-13` / `512 MiB` SMMU iTLB reach):**
   During `DirectModelLoader.create_model`, allocating 3,965 small non-MoE parameters (`48.06 GiB`)
   interleaved with 244 `FusedMoE` expert parameters (`23.62 GiB` in `256 MiB` `w13_weight` and
   `128 MiB` `w2_weight` slabs) lets PyTorch's caching allocator draw from fragmented sub-`512 MiB`
   free lists (`Orders 0..12` in `/proc/buddyinfo`) or split fresh `512 MiB` blocks on small tensors
   first. When expert weights land on scattered 64 KiB pages rather than `512 MiB` contiguous slabs,
   ARM SMMU Stage-1/Stage-2 translation walks add `+7.2%` to `+11.6%` to long-prefill latency and
   `~12%` to memory-bound decode.
2. **Unreleased warmup/capture caching-allocator blocks (`~3.77 GiB`) and ping-pong MoE workspaces (`420 MiB`):**
   - In `GPUModelRunner.capture_model`, `adaptive_verification.warm_tail_shapes` runs eager
     `_dummy_run` passes up to `max_num_batched_tokens` *after* the last `torch.cuda.graph(...)`
     context exits and before `end_free_gpu_memory` is sampled. Without `torch.accelerator.empty_cache()`
     after `warm_tail_shapes`, `~3.77 GiB` of peak-shape temporary allocations remain reserved in the
     default CUDA pool and are misattributed to `cuda_graph_size`.
   - In `Worker._compile_or_warm_up_model_after_preparation`, `warmup_kernels` runs after graph
     capture without a final `torch.accelerator.empty_cache()`, leaving any kernel-warmup temporary
     blocks cached in the allocator instead of returned to Linux `MemAvailable`.
   - In `FusedMoEKernelModularImpl.apply`, `current_workspace_manager().get_simultaneous((M, K), dtype)`
     appends to `self._current_workspaces`, which only resets between forward passes. Across the
     61 MoE layers in a single forward pass, odd and even layers ping-pong between two separate
     `420.0 MiB` buffers (`840.0 MiB` total) even though each MoE layer consumes its output before
     the next MoE layer runs.

## Changes

### vLLM patches (`vllm/0568`–`0570`)

- [`vllm/0568-Direct-IO-Bind-contiguous-weight-arena-and-Order-13-.patch`](vllm/0568-Direct-IO-Bind-contiguous-weight-arena-and-Order-13-.patch)
  (`vllm/model_executor/model_loader/direct_loader.py`):
  - Adds `bind_weight_arena(model, target_device, ...)`: packs persistent non-MoE CUDA parameters
    (excluding `load_view` parameters such as display-carveout / host-staged tables and
    `FusedMoEMethodBase` modules whose `process_weights_after_loading` calls `replace_parameter`)
    into a single 256-byte-aligned contiguous `uint8` arena (`model._direct_weight_arena`),
    preserving intra-storage views and tied weights.
  - Temporarily allocates an Order-13 (`512 MiB`) buddy sponge (`_allocate_order13_sponge`) from
    `/proc/buddyinfo` `Orders 0..12` before re-allocating `FusedMoE` parameters largest-first
    (`256 MiB` `w13_weight`, `128 MiB` `w2_weight`, then scales) so large expert weights are forced
    onto contiguous `Order-13` (`512 MiB`) kernel blocks, then releases the sponge and calls
    `torch.accelerator.empty_cache()` after weight loading and post-processing.
- [`vllm/0569-MoE-Reuse-static-MoE-output-buffer-across-layers-in-.patch`](vllm/0569-MoE-Reuse-static-MoE-output-buffer-across-layers-in-.patch)
  (`vllm/v1/worker/workspace.py`, `vllm/model_executor/layers/fused_moe/modular_kernel.py`):
  - Adds `WorkspaceManager.get_moe_output_buffer(shape, dtype)` backed by a dedicated reusable
    `self._moe_output_bytes` buffer disjoint from `self._current_workspaces`, so all MoE layers in a
    forward step reuse a single `(max_num_batched_tokens, hidden_size)` output buffer instead of
    ping-ponging across two `420.0 MiB` workspace reservations.
- [`vllm/0570-Worker-Flush-allocator-cache-after-CUDA-graph-captur.patch`](vllm/0570-Worker-Flush-allocator-cache-after-CUDA-graph-captur.patch)
  (`vllm/v1/worker/gpu/model_runner.py`, `vllm/v1/worker/gpu_worker.py`):
  - Calls `torch.accelerator.empty_cache()` at the end of `GPUModelRunner.capture_model` (after
    `adaptive_verification.warm_tail_shapes` and before `end_free_gpu_memory`) and at the end of
    `Worker._compile_or_warm_up_model_after_preparation` (after `warmup_kernels`).

### Lab telemetry (`scripts/lab.py`, `docs/lab.md`)

- `scripts/lab.py:buddy_free` and `scripts/lab.py:memory_before_boot` record `free_gib_512mib_blocks`
  and `order13_512mib_blocks` (`>= 512 MiB` blocks in `/proc/buddyinfo`) and log a warning when a
  64 KiB-page node has `< 70.0 GiB` in `512 MiB` buddy blocks before boot.
- `docs/lab.md` documents why `vm.compact_memory=1` splits `Order-13` (`512 MiB`) blocks around
  `Unmovable` slab pageblocks on warm GB10 nodes and should not be run before benchmarks.

## Files

- [`startup-weight-arena-tp4.json`](startup-weight-arena-tp4.json) / [`startup-weight-arena-tp4-dev.json`](startup-weight-arena-tp4-dev.json):
  Candidate 12 (`rc12/rc-tp4.json`) with the 5-file overlay mounted.
- [`bundles/startup-weight-arena`](bundles/startup-weight-arena):
  Kernel-lab unit test bundle (`test_direct_loader.py`, `test_direct_loader_cuda.py`,
  `test_gpu_model_runner_v2.py`, `test_gpu_worker.py`) verified on `dgx4`.
