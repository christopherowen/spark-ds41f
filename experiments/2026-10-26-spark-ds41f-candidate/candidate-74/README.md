# candidate-74

Release candidate superseding frozen `candidate-73` (`24950b3eca`) with:

1. **vLLM patch `0869`** (`fix(worker): anchor and atomically release contiguous CUDA pool on fallback`, 742 patches total, tree `f3783b0ec618`):
   - Drains stale cached-free CUDA segments (`torch.accelerator.empty_cache()`) before allocating the soaked contiguous pool and widens anchor draining to `1024 x 2 MiB` so `PP>1` last stages (which initialize speculator/sampler in `GPUModelRunner.__init__` before `_preallocate_contiguous_cuda_pool()`) always anchor inside `[pool_ptr, pool_end)` instead of silently anchoring an unrelated segment and letting `garbage_collection_threshold:0.6` evict the pool.
   - Raises `RuntimeError` if an anchor cannot land inside `[pool_ptr, pool_end)`.
   - Wires `Worker._release_contiguous_cuda_pool` as `release_cb` into `direct_loader.set_preallocated_cuda_pool_bytes()` so any fallback in `bind_weight_arena()` atomically evacuates and releases the pre-allocated pool before buddy-sponge allocation (preventing double allocation).
   - Gates both `Keeping` fast paths in `_resize_contiguous_cuda_pool_for_model()` on actual `cached_free_bytes >= model_bytes`.
   - Skips the `draft_and_other` pool reservation on non-last PP ranks (`pp_size > 1 and not is_last_rank`).
2. **`scripts/memguard.sh` transient-read retry**: retries up to 3 attempts (0.05 s spacing) on empty/non-numeric `/proc/meminfo` reads before failing closed.

All other components are unchanged from `candidate-73` (`sparknet v0.6.0` tree `173940770627` with zero local patches, `TileLang`, `TileKernels`, `NCCL 2.30.7`, `GLP-27 alpha=2.0`, `temperature=0.3 / top_p=0.95`).
