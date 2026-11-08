# Target `CUDAGraphMode.FULL` Attention/CED Metadata Replay, Preallocated CED Pinned Ring Buffer, DFlash Padding-Row Deduplication, and MoE Post-Gate Shared-Expert Fork (`0762`)

- **Date**: 2026-11-08
- **Base deployment commit**: `cc20ed18be1f7cbf5c6c53cdee945f7cc77d5fc5` (`spark-ds41f-1107`)
- **Base vLLM commit**: `320021e3c4dbb202d911ab70e2cd924c75420488` (tree `ed1bdb7a7684da33d403a055521ac0c4dd7ca670`, `0001..0761`)
- **Candidate vLLM commit**: `fccfedd5a7dd22edb0c398f6b10982ccc0d51897` (tree `5a208cb8f5766b164487dc4502dc0eba8b3242a5`, `0001..0762`, `patchset_sha256` `0cad7524d871b0cae5744b5e13432e9a831145f92f8c369fe84eef00ccf04b13`)
- **Verification node**: `sp6` (`spark-0014`, NVIDIA GB10 `sm_121a`, CUDA 13.0.2, `spark-ds41f:1107`)

---

## 1. Summary of Changes (`0762-perf-cudagraph-target-full-replay-and-post-gate-fork.patch`)

1. **In-Graph Target `CUDAGraphMode.FULL` Attention Metadata & `CEDState` Replay (`cudagraph_utils.py`, `model_runner.py`, `ced.py`, `nvidia/model_state.py`)**:
   - In `spark-ds41f:1107`, patch `0735` wired `DeepseekV41B12xMetadataBuilder.prepare_cudagraph_replay` into `DFlashCudaGraphManager` (3 draft KV-cache groups), but `ModelCudaGraphManager` (23 target KV-cache groups) and `GPUModelRunner.execute_model` still ran `build_slot_mappings_by_layer` (90-layer dict) + `model_state.prepare_attn` eagerly on every `CUDAGraphMode.FULL` decode step (`~440 us/step` of Python + 24 eager kernel launches before `cudaGraphLaunch`), even though `run_fullgraph` and `DFlashSpeculator.propose` ignore the returned `attn_metadata` and `slot_mappings_by_layer` dicts.
   - `ModelCudaGraphManager` now discovers `prepare_cudagraph_replay` across all 23 target KV-cache group metadata builders plus `model_state.prepare_cudagraph_replay` (`DeepseekV41ModelState.prepare_cudagraph_replay` -> `CEDState.prepare_cudagraph_replay` launching `ced_kernels.requests(..., compact=False)`), capturing all 24 metadata refresh kernels directly inside the `CUDAGraphMode.FULL` graph.
   - On `CUDAGraphMode.FULL` steps when `captures_full_attn_metadata` holds, `GPUModelRunner.execute_model` calls `model_state.stage_cudagraph_replay(input_batch)` (`CEDState.stage_floors`) instead of `build_slot_mappings_by_layer` + `model_state.prepare_attn`.
2. **Preallocated 4-Slot Pinned Host Ring Buffer + Floor Deduplication in `CEDState` (`ced.py`)**:
   - Replaces per-step `torch.tensor(floors, dtype=torch.int64, pin_memory=True)` and `torch.tensor(limits, dtype=torch.int32, pin_memory=True)` allocations in `CEDState.stage` / `CEDState.stage_floors` with a preallocated 4-slot pinned host ring buffer (`_floors_cpu_ring`, `_limits_cpu_ring`) and skips the H2D copy when `floors` is unchanged across consecutive decode steps (`_staged_floors` tuple comparison).
3. **MoE Post-Gate Shared-Expert Fork in `DefaultMoERunner._forward_impl` (`moe_runner.py`)**:
   - Patch `0657` added `router._post_gate_hook` in `MoERunner._apply_quant_method` so that for decode batches (`rows <= 32`), `shared_experts` launches on `aux_stream` *after* the 24-CTA decode gate GEMM (`bf16_gemm_splitk`) completes on the main stream, overlapping `shared_experts` with `router_partials` + `moe_dispatch` + routed `w13`/`w2` instead of contending with the gate GEMM.
   - However, `DefaultMoERunner._forward_impl` called `maybe_sync_shared_experts_stream` (when `workspace_size > 0`) or `maybe_forward_async` (when `workspace_size == 0`, which holds whenever `tilelang_mxfp8_input=True`) *before* calling `_apply_quant_method`, bypassing `_post_gate_hook` in real model execution.
   - `DefaultMoERunner._forward_impl` now checks the same `defer_shared_until_post_gate` condition (`rows <= 32`, `router.routes_hidden_states`, `not is_monolithic`, `not shared_experts_overlapping`) and defers the `shared_experts` launch to `_apply_quant_method`'s `_post_gate_hook`.
4. **DFlash Steady-State `padding_rows` Deduplication (`spec_decode/dflash/speculator.py`)**:
   - Tracks `self._padding_live` in `DFlashSpeculator.propose` (and clears it in `reset_attn`) so `tilelang_inputs.padding_rows(self._is_padding, live, ...)` is only launched when `live` (`0` on dummy runs or `num_query_tokens` on active runs) changes across steps.

---

## 2. Two-Boot GPU Screening Results (`sp6`, NVIDIA GB10 `sm_121a`)

Receipts:
- `experiments/2026-11-08-cudagraph-decode-optimizations/receipts/boot-1.json`
- `experiments/2026-11-08-cudagraph-decode-optimizations/receipts/boot-2.json`
- `experiments/2026-11-08-cudagraph-decode-optimizations/receipts/gpu-pytest-boot1.txt` (`161 passed`)
- `experiments/2026-11-08-cudagraph-decode-optimizations/receipts/gpu-pytest-boot2.txt` (`161 passed`)

### 2.1 Target `CUDAGraphMode.FULL` Attention Metadata (`23` Groups) + `CEDState` Replay

| Decode Shape `(reqs, tokens)` | Boot 1 Base (`us`) | Boot 2 Base (`us`) | Base Mean (`us`) | Boot 1 Cand (`us`) | Boot 2 Cand (`us`) | Cand Mean (`us`) | Mean Delta (`us/step`) | Bit-Exact (`torch.equal`) | GPU Mem Delta (`B`) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: |
| `(1, 1)` | `437.54` | `442.32` | `439.93` | `32.81` | `33.23` | `33.02` | **`-406.91` (`-92.5%`)** | `True` | `0` |
| `(1, 6)` | `437.90` | `438.79` | `438.35` | `32.89` | `33.13` | `33.01` | **`-405.34` (`-92.5%`)** | `True` | `0` |
| `(2, 12)` | `440.46` | `439.00` | `439.73` | `34.89` | `35.37` | `35.13` | **`-404.60` (`-92.0%`)** | `True` | `0` |
| `(3, 18)` | `439.77` | `441.26` | `440.52` | `33.74` | `34.41` | `34.08` | **`-406.44` (`-92.3%`)** | `True` | `0` |
| `(4, 16)` | `442.05` | `441.23` | `441.64` | `36.46` | `36.89` | `36.68` | **`-404.97` (`-91.7%`)** | `True` | `0` |
| `(4, 24)` | `441.88` | `438.74` | `440.31` | `36.39` | `36.87` | `36.63` | **`-403.68` (`-91.7%`)** | `True` | `0` |
| `(8, 32)` | `444.84` | `445.81` | `445.33` | `40.08` | `41.21` | `40.65` | **`-404.68` (`-90.9%`)** | `True` | `0` |
| `(8, 48)` | `444.98` | `443.26` | `444.12` | `37.55` | `38.26` | `37.91` | **`-406.22` (`-91.5%`)** | `True` | `0` |
| `(16, 64)` | `450.88` | `451.74` | `451.31` | `40.87` | `41.70` | `41.29` | **`-410.03` (`-90.8%`)** | `True` | `0` |

### 2.2 MoE Shared-Expert Multi-Stream Fork-Join & Post-Gate Deferral (Real TileLang Kernels)

| Topology | `M` (`num_tokens`) | Serial Mean (`us/layer`) | Pre-Gate Fork Mean (`us/layer`) | Post-Gate Fork Mean (`us/layer`) | Post-Gate vs Serial (`us/layer`) | Post-Gate vs Pre-Gate (`us/layer`) | Bit-Exact (`torch.equal`) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| **TP=3** (`I_tp=768`) | `1` | `244.83` | `235.16` | `235.09` | **`-9.74`** | **`-0.07`** | `True` |
| **TP=3** (`I_tp=768`) | `2` | `399.38` | `388.07` | `384.82` | **`-14.56`** | **`-3.25`** | `True` |
| **TP=3** (`I_tp=768`) | `4` | `697.63` | `695.48` | `689.53` | **`-8.10`** | **`-5.95`** | `True` |
| **TP=3** (`I_tp=768`) | `6` | `960.26` | `986.30` | `960.89` | `+0.63` | **`-25.41`** | `True` |
| **TP=3** (`I_tp=768`) | `8` | `1313.85` | `1301.98` | `1299.67` | **`-14.18`** | **`-2.31`** | `True` |
| **TP=3** (`I_tp=768`) | `12` | `1667.36` | `1659.67` | `1663.36` | **`-4.00`** | `+3.69` | `True` |
| **TP=3** (`I_tp=768`) | `16` | `2188.71` | `2170.12` | `2168.12` | **`-20.59`** | **`-2.00`** | `True` |
| **TP=3** (`I_tp=768`) | `20` | `2570.41` | `2523.27` | `2514.49` | **`-55.93`** | **`-8.79`** | `True` |
| **TP=3** (`I_tp=768`) | `24` | `2949.02` | `2936.24` | `2934.04` | **`-14.98`** | **`-2.20`** | `True` |
| **TP=3** (`I_tp=768`) | `32` | `3647.18` | `3627.34` | `3619.14` | **`-28.04`** | **`-8.20`** | `True` |
| **TP=4** (`I_tp=576`) | `1` | `199.45` | `187.64` | `186.88` | **`-12.57`** | **`-0.76`** | `True` |
| **TP=4** (`I_tp=576`) | `2` | `305.25` | `299.37` | `294.27` | **`-10.98`** | **`-5.10`** | `True` |
| **TP=4** (`I_tp=576`) | `4` | `534.92` | `529.37` | `521.98` | **`-12.95`** | **`-7.40`** | `True` |
| **TP=4** (`I_tp=576`) | `6` | `740.32` | `733.85` | `731.45` | **`-8.87`** | **`-2.40`** | `True` |
| **TP=4** (`I_tp=576`) | `8` | `917.45` | `909.34` | `907.08` | **`-10.37`** | **`-2.26`** | `True` |
| **TP=4** (`I_tp=576`) | `12` | `1241.37` | `1232.55` | `1233.24` | **`-8.13`** | `+0.69` | `True` |
| **TP=4** (`I_tp=576`) | `16` | `1726.11` | `1641.52` | `1635.92` | **`-90.20`** | **`-5.61`** | `True` |
| **TP=4** (`I_tp=576`) | `20` | `2005.98` | `1984.43` | `1983.04` | **`-22.94`** | **`-1.39`** | `True` |
| **TP=4** (`I_tp=576`) | `24` | `2318.52` | `2310.04` | `2312.02` | **`-6.50`** | `+1.99` | `True` |
| **TP=4** (`I_tp=576`) | `28` | `2431.14` | `2487.58` | `2412.04` | **`-19.10`** | **`-75.55`** | `True` |
| **TP=4** (`I_tp=576`) | `32` | `2828.09` | `2826.30` | `2814.49` | **`-13.60`** | **`-11.81`** | `True` |

### 2.3 DFlash Steady-State `padding_rows` Deduplication

| `(live_tokens, padded_tokens)` | Boot 1 Base (`us`) | Boot 2 Base (`us`) | Boot 1 Cand (`us`) | Boot 2 Cand (`us`) | Mean Delta (`us/step`) | Bit-Exact (`torch.equal`) |
| :--- | ---: | ---: | ---: | ---: | ---: | :---: |
| `(6, 6)` | `4.20` | `12.55` | `0.02` | `0.04` | **`-8.35`** | `True` |
| `(12, 12)` | `4.16` | `12.74` | `0.02` | `0.04` | **`-8.42`** | `True` |
| `(24, 24)` | `3.90` | `12.31` | `0.01` | `0.04` | **`-8.08`** | `True` |
| `(48, 48)` | `3.99` | `12.06` | `0.01` | `0.04` | **`-8.00`** | `True` |
| `(60, 64)` | `4.25` | `12.39` | `0.01` | `0.04` | **`-8.30`** | `True` |
| `(90, 96)` | `4.14` | `12.21` | `0.01` | `0.04` | **`-8.15`** | `True` |

---

## 3. Measured Dead Ends (Not Promoted)

### 3.1 Lead-Chunk / Geometric Split Decode CUDA Graph

- **Hypothesis**: Splitting the 61-layer target `CUDAGraphMode.FULL` graph into a 1–2 layer lead chunk (`[1, 60]`, `[2, 59]`, `[1, 10, 50]`, `[1, 4, 16, 40]`) could hide host `cudaGraphLaunch` latency if `cudaGraphLaunch` took `400–650 us` on GB10's ARM64 CPU.
- **Measured Result (`boot-1.json`, `boot-2.json`)**: On GB10 with CUDA 13.0.2 (`spark-ds41f:1107`), a monolithic 61-layer `cudaGraphLaunch` takes **`2.41–2.67 us`** of host CPU time. The `~440 us` pre-launch gap before Layer 0 in `1107` was caused by eager `prepare_attn` + `CEDState.stage` (fixed in Section 1.1 above), not `cudaGraphLaunch`. Splitting into 2–4 chunks increases host CPU launch time (`2.52 us -> 4.76–10.83 us`), breaks cross-layer PDL overlap at chunk boundaries, and offers no consistent step-time reduction.

### 3.2 `VLLM_USE_BREAKABLE_CUDAGRAPH=1` on Decode

- **Measured Result**: `VLLM_USE_BREAKABLE_CUDAGRAPH` splits each layer around any `cudagraph_break()` context. On decode (`CUDAGraphMode.FULL`), no break occurs, and splitting into piecewise sub-graphs forfeits cross-layer PDL overlap (`pdl-decode-chain`) and captures 61+ separate graphs per shape. Kept disabled (`"0"`).

---

## 4. Verification Commands

```bash
bin/spark doctor
bin/spark --cluster-config experiments/2026-11-08-cudagraph-decode-optimizations/tp3.json doctor
bin/spark --cluster-config experiments/2026-11-08-cudagraph-decode-optimizations/tp4.json doctor
git diff --check
```
