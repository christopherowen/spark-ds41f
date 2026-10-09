# TileLang migration: retire B12X, Triton and CuTe DSL

Goal (owner, 2026-10-09): serve DeepSeek V4.1 Flash on one kernel family,
TileLang (with DeepSeek's TileKernels), moving each kernel that B12X, Triton
or CuTe DSL still runs **one at a time**. Each port must **match, and
preferably beat**, the kernel it replaces. B12X itself is archived upstream
(its code continues in FlashInfer), which makes moving off it overdue.

Base: r6c (`experiments/2026-10-08-lil-rebase`, image
`vllm-ds41f-kkref:19f2c20ed4d6-r6c`, vLLM `19f2c20e` + series). Ports live
on the vLLM branch `tilelang-migration` (from r6c's `125c404e4`), one commit
per port, and are screened as file overlays on the r6c image.

## Gate for every port

1. **Same results.** Bit-identical to the kernel it replaces at every serving
   shape, or, where we already follow DeepSeek's reference semantics, equal to
   that reference.
2. **No slower per call.** Timed under CUDA graphs at the serving shapes, with
   the weights warm and cold in L2 and beside the L2 prefetch where the kernel
   overlaps it.
3. **Serving.** A lean screen against the control in the same window: one
   boot per arm, c1 and c8 distinct streams, three samples, bracketed. The
   port's step time must be within noise of the control or better, with
   quality 5/5.
4. **Then the old kernel goes.** Once a port passes, the replaced kernel and
   its switch are removed; no fallback arm stays behind.

Arms share the control's pinned DSpark cost curves, so the kernel is the only
variable. Ports are measured against the control, never against each other;
promotion bundles passed ports through the usual benchmark gate (TP4 1M with
16 streams; TP3 8 x 512K when the triangle is cabled).

## Inventory under `kernel_backend: tilelang`

From the r6 vLLM tree (2026-10-09; r6c moves upstream code but not these
call sites). Decode shares are the 2026-10-07 TP4 c1 profile (share of GPU
kernel time; streams overlap). The first window adds r6c decode, c8 and
prefill profiles to fill the gaps.

### B12X (no switch; all hard-coded)

| # | Kernel | Phase | Decode share | TileLang today |
| --- | --- | --- | ---: | --- |
| B1 | `wo_projection.run_inv_rope`: inverse RoPE + grouped WO-A + WO-B, block FP8 | decode, prefill, drafter | 4.5% | none fused; our TileLang FP8 GEMM covers the GEMMs |
| B2 | `rotary.rotate`: RoPE on the 64 rope dims (query, index query, SWA KV, index key, latent) | decode, prefill, drafter | 0.7% | `tile_kernels.transform.apply_rotary`, not wired |
| B3 | `mla.write_cache`: FP8 (528 B) and FP4 (288 B) paged cache writes | decode, prefill, drafter | ~0.5% | none (TileLang writes only the index keys) |
| B4 | `mla_compress.run`: gated ratio-1/2 compression with fused norm | decode, prefill | small | none |
| B5 | `bf16_gemv.mm`: the compressor's wkv/wgate projection | decode, prefill | small | `TileLangLinearMethod` exists; `compressor.py` hard-codes B12X |
| B6 | `block_fp8_linear.run`: the drafter's context-KV projection | drafter | small | `TileLangFP8LinearMethod` exists, not selected |
| B7 | `hyperconnection.run_engram_mix`: Engram signed-sqrt gate | decode, prefill | small | `tile_kernels.engram.engram_gate_fwd`, not wired |
| B8 | `weight_scale.scale_index_weights` | decode, prefill | small | none |
| B9 | Engram hash `_compress`/`_hash` (Triton inside B12X) | decode, prefill | small | `tile_kernels` engram hash (different interface) |
| B10 | Engram `lookup_op`: dequantize FP8 disk rows (Triton inside B12X) | decode, prefill | small | none |
| B11 | `packed_bf16_vocab_projection.pack` (Triton inside B12X) | load | — | none |
| B12 | ViT ops (GEMV, norm, rope_qkv, varlen attention, GELU, merge) | image prefill | — | none |

### Triton (vLLM)

DS4.1 model kernels: sparse-MLA metadata `_requests`, `_tokens`, `_chunk` and
`_pages`; the compressor's ratio-2 state ring (`_load_partial_state`,
`_save_partial_states`); Engram `_prepare_metadata`; `_gather_lookback_kernel`;
CED `_copy_rows`, `_decoder_requests`, `_decoder_indices` (when the checkpoint
has a CED boundary). Runner kernels (V2 runner): block tables and slot
mappings, input batch (seven), DSpark preparation and greedy drafting
(`_partial_argmax` and three more), rejection sampling (four), Gumbel
sampling. Together about 1% of a decode step.

### CuTe DSL

| # | Kernel | Phase | Decode share | TileLang today |
| --- | --- | --- | ---: | --- |
| C1 | L2 weight prefetch (`glm5next/nvidia/l2_prefetch.py`) | decode | 22% (overlapped) | **port 1, on the branch** |
| C2 | sparknet one-shot all-reduce / all-gather | decode, prefill (≤ 2 MiB) | 11% | TileLang family in sparknet main (`b61660f`), default `cute` |

### Not kernels, still B12X

The checkpoint loader (`--load-format b12x`), the Engram disk reader (io_uring
`DiskTable` and the native row reader), the preparation framework
(`b12x_prepare.py`, `b12x_startup.py`), module-level `b12x` imports and the
packed-head plumbing. These move into our own module before B12X leaves the
image.

### Outside the three engines

vLLM `_C` ops (vocabulary embedding, the shared expert's clamped SwiGLU, for
which TileKernels has `swiglu_forward`), cuBLAS in the DSpark confidence head,
FlashInfer for non-greedy sampling, and NCCL above 2 MiB. Listed for
completeness; not part of this migration unless the owner extends it.

## Order

1. **C1 L2 prefetch** (largest share; same PTX, so a pure engine swap).
2. **C2 sparknet** (the TileLang family exists and is bit-identical).
3. **Switches to TileLang methods that already exist:** B5 compressor GEMV,
   B6 drafter context-KV projection.
4. **TileKernels already has the kernel:** B7 Engram gate, B2 RoPE, B9 Engram
   hash (each checked against B12X's numerics first).
5. **New TileLang kernels:** B1 WO projection (the largest B12X item), B3
   cache writers, B4 compressor, B8 index weight scale, B10 Engram lookup,
   B11 head packing.
6. **Triton:** the DS4.1 model kernels, then the runner kernels.
7. **Non-kernel B12X:** loader, Engram reader, preparation; then remove B12X
   from the image.
8. **B12 ViT** (image prefill only).

## Status

| Port | Branch commit | Kernel checks | Serving screen | State |
| --- | --- | --- | --- | --- |
| C1 L2 prefetch | `44c44e6e3` | compiles for sm_121a: 22 registers, `UBLKPF.L2` per chunk; GPU tests written | pending (arm `prefetch.json`) | awaiting the first window |
| C2 sparknet | sparknet `b61660f` | bit-identical, register fix (2026-10-05) | pending (arms `sparknet-cute.json`, `sparknet-tilelang.json`) | awaiting the first window |
| B5 compressor projection | `1d3752533` | kernels compile for sm_121a; unit test `test_split_linear`; bundle `compressor-projection` (error vs FP64, batch invariance, warm/cold timing vs B12X, split-K at 72-96 rows) | pending (arm `compressor-projection.json`) | awaiting the first window |
| B6 DSpark context KV | `e45271941` | kernels compile for sm_121a; unit test `test_block32_rows` (the slice equals the fused projection's bits); bundle `context-kv` (same checks, plus a decode-tile sweep at 512 x 5120) | after its tiles are tuned | awaiting the first window |
| B7 Engram gate | `00fd5e6aa` | kernels compile for sm_121a; DeepSeek's arithmetic in one 256-thread block per token and stream, every load (the value too) before the reduction; unit test against FP64 and TileKernels' `engram_gate_fwd`; bundle `engram-gate` (B12X, ours at four block shapes, TileKernels' one-warp kernel) | pending (arm `engram-gate.json`) | awaiting the first window |
| B2 RoPE | `4e4af513b` | kernels compile for sm_121a; TileKernels' arithmetic plus the compressed-position floor and inverse direction, in place on the last 64 columns (B12X copies the whole head); unit test bit-equal to TileKernels' `apply_rotary`; bundle `rope` (all five roles) | pending (arm `rope.json`) | awaiting the first window |
| B9 Engram hash | `004bf569c` | kernels compile for sm_121a; one launch hashes every layer and head straight into the step's rows (B12X: vLLM's metadata copy, then three Triton launches and a copy per layer, eager before each forward); unit test equal to B12X's integer oracle; bundle `engram-hash` (exact equality, eager and graph time per step) | pending (arm `engram-hash.json`) | awaiting the first window |

### 65 to 96 rows

TP4 serves up to 16 streams x 6 tokens, so decode steps reach 96 rows, but the
TileLang decode tiles stop at 64 rows (`DECODE_ROWS`). Rows 65 to 96 take the
prefill path: the BF16 GEMM unsplit (B5: 16 CTAs on 48 SMs) and 128 x 128 FP8 tiles
(B6: 4 CTAs). The B5 and B6 bundles time those sizes against B12X, and B5 also
times split-K there. If TileLang loses, the fix is decode tiles up to 96 rows for
every TileLang projection, which is its own port with its own arm.

### Attention staging memory

The B12X KV RoPE plan carries the attention's staging memory requirement and
materializes it during preparation. B2 keeps that one plan declared, though
unused, under TileLang. The staging memory needs its own reservation before the
last attention helper (B3, B8) leaves B12X.

## Windows

Window 1 runs on the queue runner, so the kernel results can change the arms
before they boot:

1. `w1-kernels.json`: the kernel bundles, one per node (prefetch, B5, B6, B7;
   then B2, B7 again on another node, and B9).
2. Read the verdicts; commit any tile or split change to the vLLM branch, sync the
   overlays, push to main.
3. `w1-arms.json`: sync the node checkouts, profile the control (decode, c8,
   prefill), then the lean screen of every arm, bracketed by the control.

## Files

- `ports.py`: each port's commit on the vLLM branch, its files, kernel bundle,
  environment and mounts. Arms take a port at its own commit, so a later port's
  edit to a shared file (`model.py`, `linear.py`) never reaches an earlier arm.
- `sync_overlay.py`: copies each port's files at its commit into
  `overlay/<port>/`, and regenerates `bundles/<port>/` (the files under flattened
  path names, the tests, `kbench.py` and `candidate.json`) around its bench
  scripts.
- `kbench.py`: the benches' shared CUDA-graph timing (warm, and cold after an
  L2 eviction) and error helpers.
- `bundles/<port>/`: each port's kernel bundle (`candidate.json`, its bench).
- `make_configs.py`: writes `control.json` and one arm per port from the r6c
  TP4 recipe.
- `profile_*.py`, `summarize_kernels.py`, `analyze_costs.py`, `tables_arms.py`:
  the lab runner's profile and table scripts (from
  `experiments/2026-09-29-determinism`).
