# Prefix-Cache Block-Boundary & Eviction Retention (`#902`)

- **Date:** 2026-11-08
- **Issue:** [`#902`](https://github.com/christopherowen/spark-ds41f/issues/902) (`TODO.md` lines 275–278)
- **Base commit:** `8e5ea208` (`2026-11-07-spark-ds41f-1107`, vLLM 641-patch head `2756478b60b90700b9918b457aeb38657da637b5`, tree `ed1bdb7a7684da33d403a055521ac0c4dd7ca670`)
- **Candidate patch:** [`vllm/0762-fix-prefix-cache-retain-decode-frontier-swa-and-orde.patch`](vllm/0762-fix-prefix-cache-retain-decode-frontier-swa-and-orde.patch) (`patch_head: 240f1a871b4221c8ede03c299939ba253c7bc694`, `expected_tree: b8654934ac7c849345e788533ad4ef5a4862748f`)
- **Verification node:** `sp8` (`NVIDIA GB10`, container `spark-ds41f:1107`, 2 fresh container boots serialized under `flock /tmp/spark-lab-gpu.lock`)

---

## 1. Root-Cause Analysis

Multi-turn `64K–100K` agent workloads on DeepSeek-V4.1-Flash combine three hybrid KV cache groups sharing one `BlockPool` (`alignment_tokens = lcm(256, 128) = 256`):
- **Group 0 (`FullAttentionManager`):** MLA compressed KV (`block_size = 256`), dense retention.
- **Group 1 (`SlidingWindowManager`):** SWA local window (`block_size = 128`, `sliding_window = 128`, `sliding_window_contiguous_blocks = 1`), sparse checkpoint retention (`prefix_cache_retention_interval = 512` on `tp3-triangle`, `8192` on `tp4-ring`).
- **Group 2 (`FullAttentionManager`):** KDA / full-attention layers (`block_size = 256`).

Under `HybridKVCacheCoordinator.find_longest_cache_hit`, a prefix hit of length $L$ (a multiple of `256`) requires:
1. FullAttention Group 0 & Group 2 blocks cached across `[0, L)`.
2. SlidingWindow Group 1 block at `[L - 128, L)` cached (so the 128-token local sliding window at the boundary of $L$ is valid without recomputing from `0`).

Three gaps in `vllm/v1/core/` caused prefix-cache hit loss on `64K–100K` multi-turn agent streams across decode boundaries and under shared `BlockPool` eviction pressure:

### Gap 1: Decode-Frontier SWA Checkpoint Loss (`SlidingWindowManager.cache_blocks`)
`SparseRetentionPolicy._retain(end_tokens, request)` only returned `True` at sparse segment multiples (`end_tokens % interval == 0`) and at the prompt endpoint (`end_tokens == request.num_prompt_tokens`). During decode (`num_tokens > request.num_prompt_tokens`), whenever a turn crossed a 256-token boundary $B = 256 k$ (not a multiple of `8192` or `512`) and then decoded past $B + 128$ (`T_end % 256 >= 128`):
- `SlidingWindowManager.cache_blocks` skipped caching the SWA block ending at $B$ (`[B - 128, B)`).
- Once decode reached $B + 128$, `[B - 128, B)` left the 128-token sliding window and `SlidingWindowManager.remove_skipped_blocks` freed it (`reset_hash()` + return to `free_block_queue`) before `free()` ever ran.
- When Turn $k$ finished at $T_{\text{end}}$ with $T_{\text{end}} \bmod 256 \ge 128$, `retain_endpoint(request, T_end)` only saw the still-live SWA block `[B, B + 128)` (whose end token $B + 128$ is an odd multiple of `128`, not aligned to the 256-token MLA boundary $B$).
- On Turn $k+1$, `HybridKVCacheCoordinator.find_longest_cache_hit` could not match $B$ because the SWA block at `[B - 128, B)` had already been freed un-hashed, falling back to the last sparse segment multiple (`1,124,352 / 1,128,008 = 99.6759%` on `tp3-triangle` and `1,122,816 / 1,128,008 = 99.5397%` on `tp4-ring` at `evict_n = 18`).

### Gap 2: Hybrid KV Eviction Under Shared `BlockPool` Pressure (`KVCacheCoordinator.free`)
Under shared `BlockPool` pressure, hybrid MLA + SWA eviction has two competing failure modes if all SWA blocks are treated uniformly:
1. **Unanchored SWA head collapse (`1107` base at `evict_n = 36`):** In `1107`, `SlidingWindowManager.remove_skipped_blocks` freed skipped SWA checkpoints in ascending token order as the sliding window advanced, and when $T_{\text{end}} \bmod 256 \ge 128$, even the turn's own aligned endpoint SWA checkpoint at $\lfloor T_{\text{end}} / 256 \rfloor \times 256$ sat outside the final 128-token window in the aged head of `free_block_queue`. Under `evict_n = 36` on `tp4-ring`, evicting `36` blocks wiped every retained SWA checkpoint of an evicted conversation while `98%` of its FullAttention blocks remained cached, collapsing Turn 4's prefix hit to **`0`** (`1,047,808 / 1,128,008 = 92.8901%`, `min_evicted_turn_hit_rate = 0.0000%`).
2. **Over-pulling all interior SWA blocks behind live MLA blocks:** Conversely, pulling *every* interior SWA block into the active MRU tail leaves zero aged SWA blocks at the head of `free_block_queue`, forcing even light eviction (`evict_n = 18`) to shave live top MLA blocks off active conversations (`oracle` upper bound `99.4489%`, below `1107`'s `99.6759%` / `99.5397%`).

### Gap 3: Un-Cached Block Boundary Crossing on Speculative Decode Stopping Steps
In `Scheduler._update_request_with_output`, when a running request crossed a 256-token block boundary via accepted speculative draft tokens (`num_computed_tokens > num_tokens_before`) and finished on that same step (`stopped = True`), `_update_request_with_output` returned early without calling `self.kv_cache_manager.cache_blocks(request, ...)`. In `AsyncScheduler._update_request_with_output`, `cache_blocks` lacked a `!is_stale` guard and `min(request.num_computed_tokens, request.num_tokens)` clamp.

---

## 2. Fix Design

1. **Decode-Frontier SWA Retention & Intermediate SWA Recycling (`vllm/v1/core/single_type_kv_cache_manager.py`):**
   - In `SlidingWindowManager.cache_blocks`, compute the active decode frontier `resend_lo = max(0, (num_computed_tokens // 256 - 1) * 256)`, `resend_hi = ((num_computed_tokens // 256) + 1) * 256`, and `aligned_ext = ((num_computed_tokens + 255) // 256) * 256`.
   - Retain SWA blocks whose `end_tokens` fall in `[resend_lo, resend_hi]` or equal `aligned_ext` so the 256-token aligned SWA block `[B - 128, B)` remains cached when `remove_skipped_blocks` advances the sliding window to `[B, B + 128)`.
   - Track non-segment decode-frontier SWA blocks per request in `_decode_frontier_blocks[request_id]`. As decode advances to a newer boundary, any previously cached non-segment decode-frontier SWA block with `end_tokens < resend_lo` is immediately recycled (`reset_hash()` and returned to the **front** of `free_block_queue` via `prepend_n`), so intermediate decode SWA blocks do not accumulate or consume aged cache capacity (`4 / 11 / 4 / 36` cached SWA blocks after a `4,096`-token decode for `retention = 0 / 512 / 8192 / None`).
   - Track per-request retained SWA checkpoints (`_req_retained_blocks`) and prefix-chain inheritance (`_block_prefix_chain`) across turns without touching aged interior SWA checkpoints during `add_local_computed_blocks` or `remove_skipped_blocks`.

2. **Two-Tier Tail-to-Root Hybrid Eviction + Anchored Checkpoint-Floor Ladder (`vllm/v1/core/kv_cache_coordinator.py`):**
   - Organize `KVCacheCoordinator.free` and `pop_blocks_for_free` into a two-tier tail-to-root order in `free_block_queue`:
     - **Tier 0 (aged SWA head, right after unhashed scratch blocks):** non-anchored interior/superseded SWA blocks ordered strictly descending by `end_token` (`0` within-tier inversions). Light pool pressure (`evict_n = 18`) evicts expendable interior SWA blocks from Tier 0 without touching any conversation's live top MLA blocks or active endpoint SWA checkpoint.
     - **Tier 1 (active conversation tail-to-root run at MRU tail):** all full-attention blocks paired with the active turn-endpoint SWA checkpoint at `max_full_end` (even when $T_{\text{end}} \bmod 256 \ge 128$) plus a sparse fallback SWA checkpoint ladder (`below[0]` root checkpoint + `below[-1]` highest checkpoint $\le \text{max\_full\_end} - 1024$, plus `4096`-token multiples when `len(below) > 32`), ordered strictly by `(-end_token, is_sparse, group_id, -blk_idx)` (`0` within-tier inversions). Under deeper eviction (`evict_n = 36`), each evicted MLA top block steps down cleanly to the anchored fallback checkpoint floor (`73,728` / `81,920` / `90,368`) with **`0` checkpoint-floor violations** across all $k \le \text{first\_root\_mla\_evict}$.

3. **Speculative Decode Stopping-Step Block Caching (`vllm/v1/core/sched/scheduler.py`, `vllm/v1/core/sched/async_scheduler.py`):**
   - In `Scheduler._update_request_with_output` and `AsyncScheduler._update_request_with_output`, call `self.kv_cache_manager.cache_blocks(request, min(request.num_computed_tokens, request.num_tokens))` whenever `status_before_update == RequestStatus.RUNNING and not is_stale and request.num_output_placeholders == 0 and request.num_computed_tokens > num_tokens_before`.

---

## 3. 2-Boot Verification on `sp8` (`NVIDIA GB10`)

All runs were executed inside fresh `spark-ds41f:1107` containers on `sp8` (`NVIDIA GB10`) serialized with `flock /tmp/spark-lab-gpu.lock`.

### 3.1 Multi-Turn `64K–100K` Concurrent Agent Stress Probe (`4` agents × `8` turns, `64K -> 100K` tokens, `1,128,008` reusable prefix query tokens on Turns 2..8)

| Eviction Pressure | Profile | `prefix_cache_retention_interval` | Metric | `1107` Base | Boot 1 (`0762`) | Boot 2 (`0762`) | Mean ± 95% CI |
|---|---|---:|---|---:|---:|---:|---:|
| `evict_n = 18` | `tp3-triangle` | `512` | Overall reusable prefix hit rate | `99.6759%` (`1,124,352 / 1,128,008`) | **`99.8574%`** (`1,126,400 / 1,128,008`) | **`99.8574%`** (`1,126,400 / 1,128,008`) | **`99.8574% ± 0.0000%`** (`+2,048` tok, exact aligned ceiling) |
| `evict_n = 18` | `tp3-triangle` | `512` | Non-evicted / Evicted turn hit rate (min evicted) | `99.6699%` / `99.6960%` (`99.5126%`) | **`99.8478%`** / **`99.8787%`** (**`99.8307%`**) | **`99.8478%`** / **`99.8787%`** (**`99.8307%`**) | **`99.8787% ± 0.0000%`** |
| `evict_n = 18` | `tp4-ring` | `8192` | Overall reusable prefix hit rate | `99.5397%` (`1,122,816 / 1,128,008`) | **`99.8574%`** (`1,126,400 / 1,128,008`) | **`99.8574%`** (`1,126,400 / 1,128,008`) | **`99.8574% ± 0.0000%`** (`+3,584` tok, exact aligned ceiling) |
| `evict_n = 18` | `tp4-ring` | `8192` | Non-evicted / Evicted turn hit rate (min evicted) | `99.6699%` / `99.1066%` (`97.9219%`) | **`99.8478%`** / **`99.8787%`** (**`99.8307%`**) | **`99.8478%`** / **`99.8787%`** (**`99.8307%`**) | **`99.8787% ± 0.0000%`** |
| `evict_n = 36` | `tp3-triangle` | `512` | Overall reusable prefix hit rate | `99.6532%` (`1,124,096 / 1,128,008`) | **`99.8574%`** (`1,126,400 / 1,128,008`) | **`99.8574%`** (`1,126,400 / 1,128,008`) | **`99.8574% ± 0.0000%`** (`+2,304` tok) |
| `evict_n = 36` | `tp4-ring` | `8192` | Overall reusable prefix hit rate | `92.8901%` (`1,047,808 / 1,128,008`) | **`98.9497%`** (`1,116,160 / 1,128,008`) | **`98.9497%`** (`1,116,160 / 1,128,008`) | **`98.9497% ± 0.0000%`** (`+68,352` tok) |
| `evict_n = 36` | `tp4-ring` | `8192` | Evicted turn hit rate (mean / min) | `70.1149%` / `0.0000%` (Turn 4 collapse) | **`96.0192%`** / **`94.4966%`** (zero collapse) | **`96.0192%`** / **`94.4966%`** (zero collapse) | **`96.0192% ± 0.0000%`** |

### 3.2 Block-Boundary, Speculative Decode, Intermediate SWA Recycling, and Two-Tier LRU Ordering Suite

| Probe / Test Suite | `1107` Base | Boot 1 (`0762`) | Boot 2 (`0762`) |
|---|---:|---:|---:|
| Exact Block-Boundary Matrix (`block_size - 1`, `block_size`, `block_size + 1`, `383..385`, `511..513`, `8191..8193`, spec-decode steps `1, 2, 3, 5, 6`, `retention in {0, 512, 8192, None}`) | `1,134 / 1,404` (`270` failures when `T_end % 256 >= 128`) | **`1,404 / 1,404` (`100%`)** | **`1,404 / 1,404` (`100%`)** |
| Long-Decode (`4,096` tokens) Cached SWA Blocks (`retention=0 / 512 / 8192 / None`) | Missing endpoint when `T_end % 256 >= 128` | **`4 / 11 / 4 / 36` blocks (`100%` Turn 2 hit)** | **`4 / 11 / 4 / 36` blocks (`100%` Turn 2 hit)** |
| Speculative Decode (`1..6` tok/step) Boundary Crossing on Stopping Step (`Scheduler` + `AsyncScheduler`) | Failed on `Scheduler` stopping step | **`60 / 60` (`100%`)** | **`60 / 60` (`100%`)** |
| Two-Tier Shared `BlockPool` Eviction Order (`retention in {0, 512, 8192, None}`) | Root/endpoint SWA evicted ahead of MLA | **`0` Tier-0 inv, `0` Tier-1 inv, `1` tier step, `0` floor collapses** | **`0` Tier-0 inv, `0` Tier-1 inv, `1` tier step, `0` floor collapses** |
| Full 23-File GPU `pytest` Suite (`v1/core`, `prefix_cache`, `worker`, `attention`) | — | **`2,681 passed in 116.41s`** | **`2,681 passed in 116.96s`** |

---

## 4. Exact Reproduction Commands

```bash
PREP_VLLM="/home/chris/projects/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-3a6571aa/src/vllm"
EXP_DIR="experiments/2026-11-08-prefix-cache-retention-probe"

# 1. Run standalone stress probe inside container on sp8 (boot 1 & boot 2)
flock /tmp/spark-lab-gpu.lock docker run --rm --gpus all --ipc=host \
  --entrypoint python3 \
  -v "${PREP_VLLM}/vllm/v1/core:/opt/spark-ds41f/candidate/vllm/vllm/v1/core:ro" \
  -v "/home/chris/projects/spark-ds41f/${EXP_DIR}:/workspace/${EXP_DIR}" \
  -w /workspace \
  spark-ds41f:1107 \
  ${EXP_DIR}/probe_prefix_cache_retention.py \
  --boot 1 \
  --output ${EXP_DIR}/receipts/probe-boot1.json

# 2. Run full 23-file v1 core KV cache + scheduler + boundary attention pytest suite on sp8
flock /tmp/spark-lab-gpu.lock docker run --rm --gpus all --ipc=host \
  --entrypoint pytest \
  -v "${PREP_VLLM}/vllm/v1/core:/opt/spark-ds41f/candidate/vllm/vllm/v1/core:ro" \
  -v "${PREP_VLLM}/tests/v1/core/test_prefix_cache_retention_probe.py:/opt/spark-ds41f/candidate/vllm/tests/v1/core/test_prefix_cache_retention_probe.py:ro" \
  -w /opt/spark-ds41f/candidate/vllm \
  spark-ds41f:1107 \
  --noconftest \
  tests/v1/core/test_prefix_cache_retention_probe.py \
  tests/v1/core/test_prefix_group_validity.py \
  tests/v1/core/test_prefix_caching.py \
  tests/v1/core/test_single_type_kv_cache_manager.py \
  tests/v1/core/test_swa_inflight_window_free.py \
  tests/v1/core/test_sink_draft_zero.py \
  tests/v1/core/test_partial_attention_rewind.py \
  tests/v1/core/test_async_scheduler.py \
  tests/v1/core/test_boundary_admission.py \
  tests/v1/core/test_boundary_admission_aged_cache.py \
  tests/v1/core/test_boundary_admission_continuation.py \
  tests/v1/core/test_boundary_admission_horizon.py \
  tests/v1/core/test_boundary_admission_review.py \
  tests/v1/core/prefix_cache/test_mamba_boundary_capture.py \
  tests/v1/core/prefix_cache/test_mamba_boundary_capture_digest.py \
  tests/v1/core/prefix_cache/test_mamba_eagle_resume_checkpoint.py \
  tests/v1/core/prefix_cache/test_partial_prefix_cache_hits.py \
  tests/v1/core/prefix_cache/test_partial_prefix_cache_primitives.py \
  tests/v1/core/prefix_cache/test_sparse_event_context.py \
  tests/v1/worker/test_boundary_checkpoint_attention.py \
  tests/v1/worker/test_boundary_checkpoint_restore_scalars.py \
  tests/v1/attention/test_b12x_v41_ced_metadata.py \
  tests/kernels/attention/test_deepseek_v41_ced_replay.py
```

## 5. Receipts

- [`receipts/probe-boot1.json`](receipts/probe-boot1.json)
- [`receipts/probe-boot2.json`](receipts/probe-boot2.json)
- [`receipts/gpu-pytest-boot1.txt`](receipts/gpu-pytest-boot1.txt)
- [`receipts/gpu-pytest-boot2.txt`](receipts/gpu-pytest-boot2.txt)
