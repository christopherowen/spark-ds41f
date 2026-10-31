# DeepSeek-V4.1 Native GLP Spec v3 (`hc_fold_pre_engram`) Fold & Attention Steering

## Overview

Adds native TileLang support for **GLP Spec v3 (`hc_fold_pre_engram`)** alongside **GLP Spec v1 (`residual_stream_post_layer`)** on the current DeepSeek-V4.1 candidate vLLM tree, preserving per-request dual serving (`WEIGHTLESS_STEER_MODEL`), fused mHC post/pre kernels (`self.mhc.pre` / `self.mhc.post_pre`), sequence-parallel unreduced `RankOrderParts`, and CED decoder compaction.

## Design

### 1. Fail-Closed Container Reader (`vllm/models/deepseek_v4_1/glp.py`)
- Accepts two hook points with strict `glp.spec_version` pairing:
  - `residual_stream_post_layer` requires `glp.spec_version == 1`.
  - `hc_fold_pre_engram` requires `glp.spec_version == 3`.
- Refuses any hook/version mismatch, `glp.spec_version == 2`, `glp.rank != 1`, or unsupported scaling keys (`glp.dir_scales`, `glp.layer_scales`).
- Verifies `glp.content_sha256` (when present in the GGUF header) against the SHA-256 digest of the raw `direction.<layer>` float32 tensor payloads in ascending integer layer order (`sorted(directions)`).

### 2. Stream-Mean Fold Projection Without De-Fusing `mhc.pre` (`vllm/models/deepseek_v4_1/tilelang/glp.py`)
- In the candidate tree (`vllm/models/deepseek_v4_1/nvidia/model.py`), `self.glp.steer(idx, hidden_states, residual, post_mix, res_mix, steer_rows)` runs immediately after `layer(idx)` returns its FFN write `hidden_states`.
- For marked rows (`rows[t] != 0`), `glp_steer_kernel` folds `hidden_states` (or its unreduced `RankOrderParts` summed in rank order) into `residual` using `post_mix` and `res_mix`, projects the folded streams, writes `residual` back in place, and sets `comb = I` and `post = 0`.
- When `mean_streams=True` (`hc_fold_pre_engram`), `glp_steer_kernel` computes a single dot product of the stream mean `mean_n(hs[n])` with `direction` (`1` warp reduction instead of `hc_mult = 4`) and subtracts `alpha * (mean_n(hs[n]) . d) * d` uniformly from all `hc_mult` stream copies.
- Because `comb = I` and `post = 0` on marked rows, the next layer's `mhc.post` (on Engram layers, before `self.engram`), fused `mhc.pre` (on non-Engram layers), `mhc.post_collapse` (for `aux_hidden_states`), and the post-loop collapse all receive the exact post-fold, pre-Engram projected `residual` (`1 * r + 0 * x = r`) **without de-fusing `mhc.pre`**, while unmarked rows exit before any barrier via `T.thread_return()` and remain bit-identical.

### 3. Companion Attention-Write Projection (`glp_steer_write_kernel` & CED Alignment)
- GLP Spec v3 (`hc_fold_pre_engram`) specifies a mandatory companion projection on covered layers' attention output write (`attn._o_proj` output) before `self.mhc.post_pre` folds it into `residual`.
- `glp_steer_write_kernel(hidden, world)` is a PDL-chained (`pdl.chain()`) TileLang kernel that:
  - Exits immediately (`T.thread_return()`) on unmarked rows (`rows[t] == 0`), leaving all `world` parts bit-identical.
  - Sums `parts[0..world-1, t, :]` in rank order, computes `(x . d)` in FP32, writes `bf16(x - alpha * (x . d) * d)` to `parts[0, t, :]`, and zeroes `parts[1..world-1, t, :]` so `mhc.post_pre` folds the exact projected attention write.
- In `DeepseekV4Model.forward`, `steer_rows` is compacted with `boundary_indices` at the CED boundary (`idx == self.ced_decoder_start`) **before** entering `layer(...)`, ensuring `steer_rows` matches the compacted row count both at `glp.steer_attn` (after `self.attn`) and at `self.glp.steer` (after `self.ffn`).

## Verification & Benchmark (`dgx4`, GB10 SM121)

Verified inside `spark-ds41f:19f2c20ed4d6-next-20` on `dgx4`:
- **Kernel, Model, Dual-Serving, CED, SP-Prefill & PDL Audit Suite:**
  - `pytest --noconftest -p tests.kernels.deepseek_v41_pdl_sources tests/kernels/test_deepseek_v41_tilelang_glp.py tests/models/test_deepseek_v4_1_glp.py tests/test_weightless.py tests/models/test_deepseek_v4_1_ced_model.py tests/models/test_deepseek_v4_1_sp_prefill.py`
  - **`211 passed`**, `PDL sources: 18 kernels launched, 13 with PDL, 0 violations`.
- **Published Artifact Validation (`GLP_TEST_FILE`):**
  - Validated against both published artifacts:
    - Spec v1 `GLP-39` (`glp.deepseek-v41-flash-1017-GLP-39-L1-39-a0.75.gguf`, `sha256: 9c66d47701e0ceb42c33906f2d3789c0f1f16b21e7d97185f20f7f3f09fbf5cf`, `content_sha256: 4c0c2caa8b62c7e22d8da5d2938e58e8563b7c1bfde2ad27918b3f60019d9750`)
    - Spec v3 `GLP-27` (`glp.deepseek-v41-flash-1017-GLP-27-L11-37-a3.5.gguf`, `sha256: 24e97d41a9a4f77adad78ff4cb888cb4e339eb62151dbf37a7fcd3ea30eb442e`, `content_sha256: f0aaba2fd1d5d52deb605495e0b99b8e868d44a4e612c994688cf243a95dfcb4`)
- **CUDA-Graph Kernel Latency (`20` launches/graph, `200` replays on GB10 SM121):**

| Rows (`T`) | Spec v1 Fold (`glp_steer`, `mean_streams=False`) | Spec v3 Fold (`glp_steer`, `mean_streams=True`) | Spec v3 Attn Write (`glp_steer_write`) | Unmarked Row (`rows=0` early exit) |
| ---: | ---: | ---: | ---: | ---: |
| `T = 1` | `3.403 µs` (`p10 3.395 µs`) | **`3.400 µs` (`p10 3.357 µs`)** | **`1.651 µs` (`p10 1.603 µs`)** | **`0.598 µs`** |
| `T = 6` | `3.970 µs` (`p10 3.922 µs`) | **`3.907 µs` (`p10 3.875 µs`)** | **`1.725 µs` (`p10 1.717 µs`)** | **`0.600 µs`** |
| `T = 16` | `3.976 µs` (`p10 3.970 µs`) | **`4.000 µs` (`p10 3.970 µs`)** | **`1.830 µs` (`p10 1.770 µs`)** | **`0.602 µs`** |
