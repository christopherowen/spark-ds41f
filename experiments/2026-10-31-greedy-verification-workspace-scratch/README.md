# Borrow `vocab_parallel_pairs` Scratch Buffers from `WorkspaceManager`

## Problem

On every greedy speculative verification step, `vocab_parallel_pairs` in `vllm/v1/worker/gpu/spec_decode/greedy_verification.py` allocated temporary reduction buffers via `torch.empty`:
- `values`: `(rows, cdiv(vocab, 2048))` `float32`
- `indices`: `(rows, cdiv(vocab, 2048))` `int32`
- `pairs`: `(rows, PAIR_WIDTH)` `float32` (which on `tp_size > 1` is also consumed only as the input to `tensor_model_parallel_all_gather` and immediately discarded)

Those 2–3 per-step `torch.empty` allocations happen in the uncaptured verification tail between the target model's CUDA graph replay and the draft model's CUDA graph replay.

## Change

- Added `_greedy_pair_buffers(rows, vocab, tp_size, device)` in `vllm/v1/worker/gpu/spec_decode/greedy_verification.py`:
  - When `WorkspaceManager` is initialized on the same device and is either unlocked or already has sufficient capacity in the active workspace slot, borrows `(values, indices)` on TP1 or `(values, indices, pairs)` on TP>1 via `current_workspace_manager().get_simultaneous(...)`.
  - Falls back to `torch.empty` when `WorkspaceManager` is not initialized (e.g., standalone unit tests).
- Added unit test `test_vocab_parallel_pairs_borrows_workspace_scratch` in `tests/v1/worker/gpu/test_greedy_verification.py`.

## Verification

- Base vLLM commit: `d5df88d83`
- Candidate vLLM commit: `0397ac2ac` (`greedy-verification-workspace-scratch`)
- Verified on `dgx4` (`spark-ds41f:1030`):
  - `pytest tests/v1/worker/gpu/test_greedy_verification.py tests/v1/worker/test_gpu_sink.py`: **61 passed** (`17 passed` in `test_greedy_verification.py`, `44 passed` in `test_gpu_sink.py`).
