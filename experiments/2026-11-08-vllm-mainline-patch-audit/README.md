# 2026-11-08 — Canonical `vllm-project/vllm` Recent PR & Commit Audit (`#66226`, `#65866`, `#65627`, `#63401`/`#55828`, `#65092`/`#65168`)

- **Base deployment commit**: `8e5ea208f3d22bc69df91ace94d6975932c858a8` (`main`)
- **Base vLLM patch head (`1107/vllm/series`)**: `2756478b60b90700b9918b457aeb38657da637b5` (tree `ed1bdb7a7684da33d403a055521ac0c4dd7ca670`, 641 patches)
- **Candidate vLLM patch head (`experiments/2026-11-08-vllm-mainline-patch-audit/vllm/series`)**: `f1a8d099431c9969ac4568800711f09cfe7a17df` (tree `9dcb526666a33d7c62b490cc5fe0006479d5bb5d`, `patchset_sha256` `a36b7cbb33cb29570eee4aa502e11fcbbf5dcdce13d73c0a5a18b9981b0248a2`, 643 patches)
- **One intended variable**: Backport verified canonical `vllm-project/vllm` bugfixes from the `v0.33.0rc3`..`2026-11-08` window (`0762` / `#65866`, `0763` / `#65627`) while auditing and rejecting changes that regress or do not fit our GB10 unified-memory and DSpark 1-step parallel speculator architecture (`#66226`, `#63401`, `#55828`, `#65092`, `#65168`).
- **Verification node & container**: `sp8` (`NVIDIA GB10`), `spark-ds41f:1107`

---

## Summary Table

| # | Canonical `vllm-project/vllm` PR / Commit | Upstream Subject | Status in `spark-ds41f:1107` Before Audit | Decision & Local Patch |
|---|---|---|---|---|
| 1 | `#66226` (open PR) | `[Feature][Spec Decode] Add spec config to shorten drafts on prefill-heavy steps` | `0624` (`856b98415b`) already sets `skip_drafts = True` (`K = 0`) when no request in the step can use draft tokens; `0746` (`row_cost=0.035`) prunes low-confidence verification tokens online; `0751` interleaves `K=2` pure decode steps after `0.45 s` prefill stall; DSpark (`method="dspark"`) drafts all $K=5$ tokens in **one** parallel forward pass (`num_speculative_steps = 1`, `num_query_per_req = 6`) | **Do not adopt** — sequential-drafter heuristic does not reduce 1-step parallel DSpark draft latency (~`1.00–1.05 ms` flat across $K \in \{1,3,5\}$) and starves the next pure decode step of draft tokens |
| 2 | `#65866` (`0915f490c4`) | `[Bugfix] Move model inspection RPC out of LLM.__repr__` | `vllm/entrypoints/llm.py:909-920` called `self.llm_engine.collective_rpc("get_model_inspection")` synchronously inside `LLM.__repr__()` | **Adopted** as `0762-fix-entrypoints-move-model-inspection-RPC-out-of-LLM.patch` |
| 3 | `#65627` (`d1879210a4`) | `[Bugfix][Attention] Keep FA sliding window active under batch invariance` | `vllm/v1/attention/backends/flash_attn.py:935` used `common_attn_metadata.max_seq_len` instead of `self.model_config.max_model_len` when `VLLM_BATCH_INVARIANT=1`, dropping the sliding window on short mixed/eager batches while keeping it active in FULL CUDA graphs (note: DS4.1 TileLang SWA in `vllm/models/deepseek_v4_1/tilelang/attention.py` was already invariant with fixed `swa_width=128`) | **Adopted** as `0763-fix-attention-keep-FA-sliding-window-active-under-ba.patch` |
| 4 | `#63401` (`fd0c471b3a`) & `#55828` (open PR) | `[ROCm] Fix inaccurate VRAM reporting on RDNA4 GPUs` / `[Bugfix] Size KV cache from process-local GPU memory deltas` | `1107` uses `0021`/`0026`/`0030` (842.5 MiB DRM carve-out), `0491` (sliding-window KV sizing), `0761` (allocator + `malloc_trim(0)` flushes), and explicit `--kv-cache-memory-bytes` | **Do not adopt** — `#63401` is ROCm discrete-only (`props.is_integrated` returns `None`); `#55828` uses NVML per-process `usedGpuMemory`, which reports `0` bytes for `pin_memory=True` (`cudaHostAlloc`) and CPU tensors on GB10 unified memory despite consuming physical LPDDR5x |
| 5 | `#65092` (`d76fe936a1`) & `#65168` (`25cd126067`) | `[ROCm] Reserve kpool decode indexer logits in workspace during profiling` / `[Perf] Support Mooncake store in vllm preload daemon` | DS4.1 TileLang (`0506`) already reserves both `DECODE_CHUNK` and `INDEX_CHUNK` in `WorkspaceManager` during `reserve_profile_scratch()`, but `vllm/models/glm5next/amd/sparse_indexer.py` still allocated a throwaway `torch.empty` instead of reserving decode logits in `WorkspaceManager` before `lock_workspace()` | **Do not adopt** — `#65092` is ROCm-only (`rocm/sparse_attn_indexer_kpool.py`) and unused on GB10 SM121; `#65168` is unused (`1107` uses frozen per-node JIT caches `/cache/spark-ds41f/jit/*` rather than `vllm preload`/Mooncake) |

---

## Detailed Audit of All 5 Items

### 1. `vllm-project/vllm#66226` — `[Feature][Spec Decode] Add spec config to shorten drafts on prefill-heavy steps`

- **Upstream mechanism**: Adds `DraftLoadThresholdConfig` (`token_threshold`, `num_spec_tokens`) to `SpeculativeConfig`. In `Scheduler.schedule()`, when `total_scheduled_tokens >= token_threshold`, `SchedulerOutput.num_spec_tokens` is reduced from $K$ to `draft_load_threshold_config.num_spec_tokens` (e.g. `0` or `1`). Supported only for `_V1_VARIABLE_NUM_SPEC_TOKENS_METHODS = {"ngram", "eagle", "eagle3", "mtp", "draft_model", "dflash"}`.
- **Comparison against `spark-ds41f:1107`**:
  1. **Pure prefill steps (`0624`)**: Our `1107` stack already carries `0624-perf-v1-spec_decode-skip-speculative-drafting-in-ste.patch` (backport of `856b98415b`). In `vllm/v1/core/sched/scheduler.py`, `skip_drafts = not any_req_uses_drafts` is `True` whenever every scheduled request in the step is either in a non-final prefill chunk (`num_computed_tokens + num_new_tokens < num_prompt_tokens`) or guaranteed to finish (`reach max_tokens`). In `vllm/v1/worker/gpu_model_runner.py`, `num_speculative_tokens = 0 if scheduler_output.skip_drafts else self.speculative_config.num_speculative_tokens`. In `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py:870`, `DFlashSpeculator.propose()` stores context K/V to keep the drafter in sync and immediately returns (`if num_speculative_tokens == 0: return self.draft_tokens[:num_reqs]`), skipping the query pass, Markov sampling, and CopyDrafter.
  2. **Mixed prefill+decode steps (`0746`, `0751`, and DSpark parallel drafting)**:
     - Unlike sequential multi-step drafters (EAGLE/MTP/draft_model where $K=5$ runs 5 sequential forward passes), our DSpark speculator (`method="dspark"`, `num_speculative_steps = 1`, `num_query_per_req = 6`) executes **one parallel forward pass** of $N_{\text{req}} \times (1 + K)$ query rows captured as a FULL CUDA graph. On GB10 (`sp8`), a 1-step parallel draft forward pass takes `0.998–1.067 ms` for 1–8 requests whether $K=1$, $K=3$, or $K=5$ (`item1_pr66226_spec_decode_prefill_load` in `receipts/audit-boot{1,2}.json`).
     - Furthermore, when a prefill chunk runs alongside decode requests (`0751` interleaves `K=2` pure decode steps after `0.45 s` accumulated prefill stall), the step immediately preceding a pure decode step is either a mixed step or a final prefill chunk. Shortening or skipping draft generation on that step would leave the decode request with `0` or `1` draft tokens on the ensuing pure decode step, collapsing speculative acceptance depth from $K=5$ to $0/1$ while saving `0 ms` on the 1-step parallel DSpark query pass.
     - On the verification side, `0746` (`VLLM_DSPARK_CONFIDENCE_CALIBRATION=1`, `row_cost=0.035`) already prunes low-confidence draft tokens per request online before the next target verification step.
- **Conclusion**: Do not adopt `#66226`.

### 2. `vllm-project/vllm#65866` (`0915f490c4`) — `[Bugfix] Move model inspection RPC out of LLM.__repr__`

- **Audit of `1107`**: In `vllm/entrypoints/llm.py:909-920`, `LLM.__repr__()` checked `if self._cached_repr is None:` and synchronously invoked `self.llm_engine.collective_rpc("get_model_inspection")`. Any debugger inspection, exception traceback formatting, test assertion failure, or logging statement calling `repr(llm)` triggered a blocking cross-worker `collective_rpc("get_model_inspection")` across all TP ranks.
- **Fix (`0762-fix-entrypoints-move-model-inspection-RPC-out-of-LLM.patch`)**:
  - Replaces `_cached_repr` with `_cached_model_inspection`.
  - Makes `LLM.__repr__()` a pure, side-effect-free local format (`f"LLM(model={self.model_config.model!r})"`), taking `0.137–0.139 us` with `0` RPC calls across 100 calls.
  - Adds explicit `LLM.get_model_inspection() -> str` (caching `self.llm_engine.collective_rpc("get_model_inspection")[0]`).

### 3. `vllm-project/vllm#65627` (`d1879210a4`) — `[Bugfix][Attention] Keep FA sliding window active under batch invariance`

- **Audit of `1107`**:
  - Our DS4.1 TileLang attention backend (`vllm/models/deepseek_v4_1/tilelang/attention.py`) enforces `self.swa_width = 128` unconditionally inside the kernel and metadata builder regardless of batch `max_seq_len`, so DS4.1 TileLang SWA is already batch-invariant.
  - However, in `vllm/v1/attention/backends/flash_attn.py:935`, `FlashAttentionMetadataBuilder.build()` computed `max_seq_len = common_attn_metadata.max_seq_len` and then cleared `window_size = (-1, -1)` whenever `self.vllm_config.model_config.hf_text_config.sliding_window is not None and common_attn_metadata.max_seq_len <= self.vllm_config.model_config.hf_text_config.sliding_window`. Under `VLLM_BATCH_INVARIANT=1`, FULL CUDA graph capture passes `max_seq_len = max_model_len` (keeping the sliding window active), whereas short eager/mixed batches pass `common_attn_metadata.max_seq_len <= sliding_window` (disabling the sliding window and changing numerical outputs).
- **Fix (`0763-fix-attention-keep-FA-sliding-window-active-under-ba.patch`)**:
  - In `FlashAttentionMetadataBuilder.build()`, sets `max_seq_len = self.model_config.max_model_len` when `envs.VLLM_BATCH_INVARIANT` is enabled and checks `max_seq_len <= sliding_window` against that effective `max_seq_len`.

### 4. `vllm-project/vllm#63401` (`fd0c471b3a`) & `#55828` — Process-Local KV Cache Sizing vs GB10 Unified Memory

- **Audit of `#63401`**: Adds `vllm/utils/rocm_vram.py` for discrete AMD RDNA4 GPUs where `hipMemGetInfo` misreports VRAM, explicitly skipping integrated GPUs (`if props.is_integrated: return None, "integrated GPU"`). No-op on NVIDIA GB10 (`is_integrated = True`).
- **Audit of `#55828` on GB10 (`sp8`)**:
  - `#55828` changes `MemorySnapshot` and `memory_profiling()` to compute `total_consumed = after_process - before_process` using NVML's `nvmlDeviceGetComputeRunningProcesses(handle)[pid].usedGpuMemory`, falling back to device-wide `free_memory` only when `usedGpuMemory` is unavailable (`None` or `<= 0`).
  - On GB10 unified memory (`is_integrated_gpu == True`), `nvmlDeviceGetComputeRunningProcesses` **does not** return `None` or `0`: it returns a positive byte count (`248,758,272` bytes base context) that tracks `cudaMalloc` allocations (`+536,870,912` bytes for a `512 MiB` CUDA tensor, returning to `+0` after `torch.cuda.empty_cache()`), so `#55828` activates its process-local branch on GB10.
  - However, when the process allocates `256 MiB` of pinned host memory (`pin_memory=True` / `cudaHostAlloc`) plus `256 MiB` of CPU tensors during profiling, `nvmlDeviceGetComputeRunningProcesses` reports a delta of **`0` bytes**, even though `psutil.virtual_memory().available` drops by `472–480 MiB` from the shared 119 GiB LPDDR5x unified memory pool (`item4_pr63401_pr55828_gb10_uma_memory` in `receipts/audit-boot{1,2}.json`).
  - Adopting `#55828` on GB10 would therefore blind `memory_profiling()` to pinned/host allocations made during model/worker initialization (such as host-resident Engram tables or RDMA bounce buffers) and over-allocate KV cache into unified memory. Our `1107` stack (`0021`, `0026`, `0030`, `0491`, `0761`, and explicit `--kv-cache-memory-bytes`) remains the right architecture for GB10.

### 5. `vllm-project/vllm#65092` (`d76fe936a1`) & `#65168` (`25cd126067`) — Workspace Reservation During Profiling & Preload Daemon

- **Audit of `#65092`**:
  - In `1107`, DS4.1 TileLang (`0506-DS4.1-TileLang-attention-scratch-reserved-in-its-own.patch` in `vllm/models/deepseek_v4_1/tilelang/attention.py`) already reserves both `DECODE_CHUNK` and `INDEX_CHUNK` indexer scratch in `WorkspaceManager` during `reserve_profile_scratch()` before `lock_workspace()` is called in `gpu_worker.py`.
  - However, `vllm/models/glm5next/amd/sparse_indexer.py` in `1107` still had the exact bug fixed by `#65092`: during prefill-only profiling (`num_decode_tokens == 0`), `sparse_attn_indexer_kpool` only requested the smaller prefill gather buffers from `WorkspaceManager` and allocated a throwaway `torch.empty` for decode logits. After `lock_workspace()` froze the workspace size at the end of `profile_run()`, the first decode step with larger `logits` (`[H, decode_rows, max_pool_len]`) exceeded the locked workspace size and raised `RuntimeError`.
- **Fix (`0764-fix-rocm-reserve-kpool-decode-indexer-logits-in-work.patch`)**:
  - Reserves worst-case decode logits `((H, decode_rows, max_pool_len), torch.float32)` via `workspace_manager.get_simultaneous(...)` during profiling (`is_profiling`).
- **Audit of `#65168`**: Adds Mooncake distributed KV/object store support to `vllm preload`. Not applicable to `1107`, which mounts pre-populated node-local deterministic JIT caches (`/cache/spark-ds41f/jit/{tilelang,triton,vllm,nv-compute}`).

---

## Two-Boot Verification (`Boot 1` & `Boot 2`) on `sp8` (`spark-ds41f:1107`)

### Exact Command

```bash
EXP_DIR=/tmp/work-vllm-mainline/experiments/2026-11-08-vllm-mainline-patch-audit
VLLM_TREE=/tmp/work-vllm-mainline/.work/upstreams/vllm
REL_TESTS=/tmp/work-vllm-mainline/scripts/release_tests

for BOOT in 1 2; do
  docker run --rm --gpus all --ipc=host --entrypoint bash \
    -e HF_HUB_OFFLINE=1 \
    -e PYTHONPATH=/opt/spark-ds41f/candidate/vllm:/opt/release_tests \
    -v "${VLLM_TREE}/vllm/entrypoints/llm.py:/opt/spark-ds41f/candidate/vllm/vllm/entrypoints/llm.py:ro" \
    -v "${VLLM_TREE}/vllm/v1/attention/backends/flash_attn.py:/opt/spark-ds41f/candidate/vllm/vllm/v1/attention/backends/flash_attn.py:ro" \
    -v "${VLLM_TREE}/vllm/models/glm5next/amd/sparse_indexer.py:/opt/spark-ds41f/candidate/vllm/vllm/models/glm5next/amd/sparse_indexer.py:ro" \
    -v "${VLLM_TREE}/tests/v1/worker/test_gpu_worker.py:/opt/spark-ds41f/candidate/vllm/tests/v1/worker/test_gpu_worker.py:ro" \
    -v "${VLLM_TREE}/tests/v1/worker/test_workspace.py:/opt/spark-ds41f/candidate/vllm/tests/v1/worker/test_workspace.py:ro" \
    -v "${REL_TESTS}:/opt/release_tests:ro" \
    -v "${EXP_DIR}:/opt/audit" \
    -w /opt/spark-ds41f/candidate/vllm \
    spark-ds41f:1107 \
    -lc "
      python3 -m pytest -v -p no:cacheprovider -p tblib_stub -p offline_skip \
        tests/v1/worker/test_gpu_worker.py \
        tests/v1/worker/test_workspace.py \
        -k \"llm_repr or batch_invariant or kpool\" \
        | tee /opt/audit/receipts/pytest-boot${BOOT}.txt
      python3 /opt/audit/screen_vllm_mainline_patch_audit.py \
        --boot ${BOOT} \
        --output /opt/audit/receipts/audit-boot${BOOT}.json
    "
done
```

### Results Across `Boot 1` and `Boot 2`

| Metric / Verification Check | Unpatched `1107` | Candidate (`+0762..0763`) Boot 1 | Candidate (`+0762..0763`) Boot 2 | Mean / Repeatability |
|---|---|---|---|---|
| Pytest gate suite (`test_llm_repr_does_not_call_collective_rpc`, `test_flash_attn_batch_invariant_keeps_max_model_len_for_sliding_window`, `test_amd_kpool_sparse_indexer_reserves_decode_logits_in_workspace`) | Fails all 3 | **3 passed in `5.76 s`** | **3 passed in `5.61 s`** | `100%` pass (`5.685 s` mean) |
| Item 1 (`#66226`): `DFlashSpeculator.propose` `K=0` early skip (`0624`) | `True` | `True` | `True` | Verified |
| Item 1 (`#66226`): 1-step parallel draft forward latency (`reqs=4, K=1` vs `reqs=4, K=5`) | `0.99–1.05 ms` | `0.9928 ms` vs `1.0335 ms` | `1.0404 ms` vs `1.0513 ms` | `1.0166 ms` ($K=1$) vs `1.0424 ms` ($K=5$), $\Delta = +0.026\text{ ms}$ |
| Item 1 (`#66226`): 1-step parallel draft forward latency (`reqs=8, K=1` vs `reqs=8, K=5`) | `1.00–1.02 ms` | `0.9968 ms` vs `1.0033 ms` | `1.0217 ms` vs `1.0211 ms` | `1.0093 ms` ($K=1$) vs `1.0122 ms` ($K=5$), $\Delta = +0.003\text{ ms}$ |
| Item 2 (`0762` / `#65866`): `collective_rpc` calls during `100 x repr(llm)` | `1` blocking RPC (`~5,060 us`) | **`0` RPCs (`0.139 us`/call)** | **`0` RPCs (`0.137 us`/call)** | `0` RPCs (`0.138 us`/call mean) |
| Item 2 (`0762` / `#65866`): Explicit `llm.get_model_inspection()` RPC count (2 calls) | N/A | `1` RPC (`5.069 ms` 1st, `0.416 us` 2nd) | `1` RPC (`5.062 ms` 1st, `0.464 us` 2nd) | Cached after first explicit call |
| Item 3 (`0763` / `#65627`): FA effective `max_seq_len` for SWA (`sliding_window=1024, batch_max_seq_len=512`) under `VLLM_BATCH_INVARIANT=1` | `512` (drops sliding window) | **`4096` (`max_model_len`, keeps SWA active)** | **`4096` (`max_model_len`, keeps SWA active)** | Invariant across eager & FULL CG |
| Item 4 (`#55828` on GB10 UMA): NVML `usedGpuMemory` delta for `512 MiB` `cudaMalloc` | `+536,870,912 B` | `+536,870,912 B` | `+536,870,912 B` | Exact (`512.0 MiB`) |
| Item 4 (`#55828` on GB10 UMA): NVML `usedGpuMemory` delta for `256 MiB` pinned (`cudaHostAlloc`) + `256 MiB` CPU tensor | `0 B` (blind to UMA host allocs) | `0 B` (`psutil` drops `480.0 MiB`) | `0 B` (`psutil` drops `472.3 MiB`) | Confirms `#55828` is unsafe on GB10 UMA |
| Item 5 (`0764` / `#65092`): Locked `WorkspaceManager` capacity after prefill profiling (`H=32, decode_rows=96, pool=4096`) | `135,168 B` (`RuntimeError` on decode) | **`50,331,648 B` (`48.0 MiB`, decode succeeds)** | **`50,331,648 B` (`48.0 MiB`, decode succeeds)** | Exact (`48.0 MiB` locked workspace) |

---

## Conclusion

1. **Carry `0762` and `0763`** in `experiments/2026-11-08-vllm-mainline-patch-audit/vllm/series` (`f1a8d099431c9969ac4568800711f09cfe7a17df`):
   - `0762-fix-entrypoints-move-model-inspection-RPC-out-of-LLM.patch` (`#65866`) eliminates unexpected synchronous `collective_rpc("get_model_inspection")` calls from `LLM.__repr__()`.
   - `0763-fix-attention-keep-FA-sliding-window-active-under-ba.patch` (`#65627`) keeps FlashAttention sliding window active on short batches when `VLLM_BATCH_INVARIANT=1`.
2. **Reject `#66226`, `#63401`, `#55828`, `#65092` (ROCm-only), and `#65168`** for our GB10 + DSpark stack based on architectural and empirical measurements on `sp8`.
