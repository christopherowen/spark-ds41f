# Candidate `candidate-66`

Date: 2026-11-22.
Base: `candidate-65` (`vLLM 0001..0844`, tree `34b01980b09e1f634738f7bc781a588327c26b75`).

## Delta Over `candidate-65`

1. **vLLM patch `0845`** (`0845-perf-renderer-enable-DeepseekV4Renderer-thread-pool-.patch`):
   - Wires `maybe_make_thread_pool(self.tokenizer, 0)` (lazy per-worker deepcopy with async background warm of copy #1 so cold-boot cost is `0.00 s`) into `DeepseekV4Renderer` when `renderer_num_workers > 1`.
   - Caches and deduplicates BPE-invariant newline subchunks (`_split_parallel_newline_chunks`) individually in `vllm/tokenizers/deepseek_v41.py::_resolve_cached_pieces` (`100%` bit-exact).
   - Enables `--renderer-num-workers 8` across all `candidate-66` shape profiles (`tp1`..`tp8`).
   - Evidence: [2026-11-22-short-delta-ttft-audit](../../../2026-11-22-short-delta-ttft-audit/README.md) (`PR #1296`): `-11.8%..-27.7%` client TTFT and `-30.0%..-38.5%` wave wall time on growing/uncached-segment waves, `-76%` host overhead under `c4` decode, `-1.1%..-4.1%` solo 20-turn multi-turn TTFT, `108/108` bit-exact, `159/159` unit tests pass.
2. **vLLM patch `0846`** (`0846-perf-startup-fuse-MoE-slab-interleave-tile-bound-UMA.patch`):
   - Fuses `_interleave_gate_up_slabs` directly into `tile_expert_weights` (`moe.py`), eliminating `60.4 GiB` of intermediate allocations and `120.8 GiB` DRAM R+W (`100%` `torch.equal`).
   - Keeps a bounded reusable caching-allocator working set (`< 4 GiB` while `MemAvailable >= 8 GiB`) across identical MoE layers in `process_weights_after_loading` (`utils.py`) with one final reclaim after the loop, eliminating 40 consecutive `gc.collect + empty_cache` kernel page unmap/refault cycles (`-13.6 s`).
   - Defers `Mxfp4MoEMethod.create_weights`'s `90.6 GiB` throwaway `torch.zeros` using 1-byte CUDA `expand(shape)` placeholders during `DirectModelLoader.create_model` (`direct_loader.py` + `mxfp4.py`) so `bind_weight_arena` allocates MoE buffers once under the buddy sponge (`-2.75 s`).
3. **vLLM patch `0847`** (`0847-perf-startup-warm-index-TileLang-step-kernels-and-fo.patch`):
   - Routes `TileLangKernels.compile_many` through `_prepare_with_warm_index` (`prepare.py`) and extends provenance hashing (`v2`) to include all `v1/worker/gpu/**/tilelang*.py` step-kernel sources (`301/301` step kernels fast-loaded in `0.34 s` vs `10.06 s`, `-9.7 s`).
   - Allows single-threaded, zero-CUDA `EngineCoreProc.run_engine_core` (`core.py`) to fork local `Worker_TP0` instead of paying a second `8.6 s` cold spawn (`-4.3 s`, plus `+0.37 GiB` CoW headroom on `dgx1`).
   - Combined cold/warm boot evidence: [2026-11-22-cold-boot-optimization](../../../2026-11-22-cold-boot-optimization/README.md): 2-boot `tp3-triangle` `API ready` reduced from **`+114.4 s → +84.35 s` (`-30.05 s`, `-26.3%`)**, inside-container **`97.10 s → 67.97 s` (`-30.0%`)**, `dgx1` `free_memory` improved `7.70 → 8.02 GiB`, `0` greedy diffs across both boots.

## Build & Verification Status

Pending build of `spark-ds41f:candidate-66` (vLLM `0001..0847`, `720` patches, `patch_head = 400ef80a06811ac53cee64e2262b640852815ddb`, `expected_tree = 8ce4b43e546ba0d612f9694f0c3785348326c2da`, `patchset_sha256 = c0787e6370d1351a19cec49ff99e1f9f51e5ac638a56ad9d06d15b30c9e7a301`) on `dgx4`.
