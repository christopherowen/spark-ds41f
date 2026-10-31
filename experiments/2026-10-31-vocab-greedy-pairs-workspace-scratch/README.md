# Borrow `project_head_greedy_pairs` and `vocab_parallel_pairs` Scratch Buffers from WorkspaceManager

## Problem

Patch `0605` (`perf(dsv41): skip full-vocab logits materialization on greedy verify`) replaced the `248320`-wide BF16 logits buffer with `project_head_greedy_pairs(...)` (`vllm/models/deepseek_v4_1/tilelang/vocab.py`) called from `vocab_parallel_pairs_from_head(...)` (`vllm/v1/worker/gpu/spec_decode/greedy_verification.py`).

However, on every greedy speculative-verification decode step (outside the main backbone CUDA graph), `vocab_parallel_pairs_from_head` -> `project_head_greedy_pairs` allocates **3 fresh GPU tensors via `torch.empty` per step**:
1. `partial_vals = torch.empty((rows, num_tiles), dtype=torch.float32, device=x.device)`
2. `partial_idxs = torch.empty((rows, num_tiles), dtype=torch.int32, device=x.device)`
3. `pairs = torch.empty((rows, 2), dtype=torch.int32, device=x.device)` (which is immediately consumed as temporary `local_pairs` by `all_gather_greedy_pairs` when `tp_size > 1` and then discarded).

Similarly, the fallback `vocab_parallel_pairs(logits, ...)` path allocates a temporary `local_pairs = torch.empty((logits.shape[0], 2), dtype=torch.int32, device=logits.device)` when `tp_size > 1`.

## Change

1. **Workspace-Backed Scratch in `project_head_greedy_pairs` (`vllm/models/deepseek_v4_1/tilelang/vocab.py`)**:
   - Add `_vocab_greedy_buffers(layer, rows, num_tiles, device, *, scratch_pairs)` which borrows `(partial_vals, partial_idxs[, pairs])` in a single `current_workspace_manager().get_simultaneous(...)` call when `WorkspaceManager` is initialized, or reuses persistent 1D flat scratch buffers cached on `layer._tilelang_vocab_greedy_scratch` otherwise.
   - Add `scratch_pairs: bool = False` keyword argument to `project_head_greedy_pairs` so callers that consume the returned `(rows, 2)` tensor immediately before the next workspace borrow (`vocab_parallel_pairs_from_head` when `tp_size > 1`) also avoid allocating `pairs`.
2. **Workspace-Backed `local_pairs` in `vocab_parallel_pairs` (`vllm/v1/worker/gpu/spec_decode/greedy_verification.py`)**:
   - Pass `scratch_pairs=tp_size > 1` from `vocab_parallel_pairs_from_head` to `project_head_greedy_pairs`.
   - Borrow `local_pairs` from `current_workspace_manager()` in `vocab_parallel_pairs` when `tp_size > 1`.

## Verification

- Verified on `dgx4` inside `spark-ds41f:19f2c20ed4d6-next-21`:
  - `pytest --noconftest -q tests/kernels/quantization/test_deepseek_v41_tilelang_vocab.py tests/v1/worker/gpu/test_greedy_verification.py`: **52 passed**.
