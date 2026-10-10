# TileLang migration: retire B12X, Triton and CuTe DSL

Goal (owner, 2026-10-09): serve DeepSeek V4.1 Flash on one kernel family,
TileLang (with DeepSeek's TileKernels), moving each kernel that B12X, Triton
or CuTe DSL still runs **one at a time**. Each port must **match, and
preferably beat**, the kernel it replaces. B12X itself is archived upstream
(its code continues in FlashInfer), which makes moving off it overdue.

Base: r6c (`experiments/2026-10-08-lil-rebase`, image
`vllm-ds41f-kkref:19f2c20ed4d6-r6c`, vLLM `19f2c20e` + series). The ports live
on the vLLM branch `tilelang-migration` (from r6c's `125c404e4`) in two kinds of
commit:

- **Module commits** add kernels, refactors that compile to the same code, and
  imports. They change nothing the model runs; the L2 prefetch family is chosen
  by environment.
- **One switch commit per port** routes one part of the model to its TileLang
  kernel. Each switch applies to the modules on its own, without the others.

Every arm mounts the modules over the r6c image. A port's arm adds its switch
(or sets its environment) and nothing else, and the control mounts the modules
alone, so the switch is the only variable. The first layout pinned each port
to its own commit on a linear branch, which carried earlier ports' edits to
shared files (`model.py`, `attention.py`) into later arms; it was replaced
before any window ran (old branch `tilelang-migration-v1`).

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

The owner's bar since window 1 (2026-10-10): faster than the replaced kernel
everywhere, at every row count, warm and cold.

| Port | Switch commit | Kernels (window 1, 2026-10-09) | Serving screen (window 1) | Next |
| --- | --- | --- | --- | --- |
| C1 L2 prefetch | `c916fed96` (module; environment switch) | **pass**: a weight reads 15-27% faster after the TileLang prefetch than after the CuTe one | level with the control | ready |
| C2 sparknet | sparknet `b61660f` | bit-identical (2026-10-05) | TileLang level with CuTe | ready |
| B5 compressor projection | `03ce20c16` | fail: faster than B12X at 4-64 rows and prefill, 1.7x slower at 72-96 (prefill path), FP32 error 20x B12X's | c8 -4.4% with changed acceptance | window 2: one fused launch for both parts, split-K to 128 rows, shard sweep; error gated against DeepSeek's FP32 reference |
| B6 DSpark context KV | `43e30271d` | fail: numerics equal to B12X, faster at 1-32 rows, slower at 48-64 (1.3x), 72-96 (3.1x) and 512 (1.7x) | not screened | window 2: split-K MXFP8 to 128 rows, split prefill tiles, sweep |
| B7 Engram gate | `80bb28564` | **pass**: 0.39-0.87x B12X at every size, error equal to B12X and TileKernels | not screened | arm |
| B2 RoPE | `bf8c1c72a` | unit tests pass (bit-equal to TileKernels); bench crashed (device without index) | not screened | window 2 |
| B9 Engram hash | `ee4871959` | **pass**: exact; 22 us a step eager instead of 100, 2.4 us instead of 10 under graphs | not screened | arm |
| B1 WO projection | `60e79424a` | unit tests pass; bench crashed (no workspace); tile sweep ran (WO-A small wins) | not screened | window 2 |
| B8 index head weights | `b4abab04f` | not run (window 1's third job stopped on a LAN drop) | not screened | window 2 |

### Window 2 (2026-10-10, 06:18-06:36 UTC, kernels only)

- B2 RoPE: faster nearly everywhere (up to 40x at prefill); a few ~1 us points
  within noise.
- B8: faster at decode; above 64 rows the 32-column projection itself takes
  ~31 us on both sides (narrow BF16 projections leave split-K at 64 rows).
- B6 (split-K): 0.34-0.55x B12X at 1-64 rows, 0.65x at 8192; slower at 72-96
  cold (serving tile; the sweep's 96-row tile wins at 0.83x), 512 (1.14x with
  the best prefill tile) and 2048 (1.02-1.13x). Five shards.
- B5 (fused parts, split-K): FP32 path 2-9x faster from 4 rows; error below
  DeepSeek's FP32 reference at 10 shards. Slower at 1-2 rows (the reduce
  launch) and the BF16 path at 72-96 cold; five shards fastest.
- B1: faster warm at 1-64 rows and at 8192; 3-7% slower cold at small rows,
  1.1x at 72-96, 1.25x at 512, 1.05x at 2048.

Window 3 (`w3-kernels.json`, module `e0da3df05`): one-launch split-K and blocked
accumulation for B5, per-bucket tiles for B6 with split-K timed to 512 rows, WO
decode tiles to 128 rows plus a component breakdown and prefill-tile sweep, and
the production projections at 65-1024 rows (`decode-rows`).

### Window 1 (2026-10-09, 22:02-22:32 UTC)

The arms spec ran before the kernel spec (the queue sorts by name). Arms booted in
80-97 s each. The third kernel job stopped when the cooling check could not read
dgx3 during a LAN-switch drop; the window closed itself and production was back at
22:32:44.

### Narrow projections

512 columns are 8 tiles of 64 on 48 SMs, so B5 and B6 left most SMs idle at decode,
and rows 65-96 fell to prefill tiles (4-16 CTAs). The split-K module (`35b5b1e8f`)
splits K for decode rows up to 128 and adds the shards in order; the prefill GEMM
accumulates the same shards, so a row's bits stay batch-invariant. B5 projects both
parts in one launch. Shard counts change bits, so window 2's sweeps choose one per
shape (`SPLIT_FP8`, `SPLIT_BF16`).

DeepSeek's reference computes the compressor's wkv/wgate as FP32 `Linear` layers on
`x.float()`: an FP32 GEMM of BF16-exact values (the checkpoint stores them BF16).
B12X's GEMV reduces in a tree and is more precise than the reference; B5's gate is
the reference's error.

### Attention staging memory

The B12X KV RoPE plan carries the attention's staging memory requirement and
materializes it during preparation. B2 keeps that one plan declared, though
unused, under TileLang. The staging memory needs its own reservation before the
last attention helper (B3, B8) leaves B12X.

## Windows

Window 1 ran on the queue runner, so the kernel results could change the arms
before they boot:

1. `w1-kernels.json`: the kernel bundles, one per node, in three jobs
   (prefetch, B5, B6, B7; B2, B7 again, B9, B1; B8, and B1, B6 and B5 again on
   other nodes).
2. Read the verdicts; commit any tile or split change to the vLLM branch, sync the
   overlays, push to main.
3. `w1-arms.json`: sync the node checkouts, profile the control (decode, c8,
   prefill), then the lean screen of every arm, bracketed by the control.

Window 2 (`w2-kernels.json`) is kernels only: B5, B6, B1 and B2, then B8 with B6
and B5 again on other nodes. It sweeps shard counts and tiles for the narrow
projections; their chosen configurations go into `SPLIT_FP8` and `SPLIT_BF16`
before the arms run again.

## Files

- `ports.py`: the module tip, each port's switch commit, environment and kernel
  bundle.
- `sync_overlay.py`: builds `overlay/modules/` (the control's files) and
  `overlay/<port>/` (the files that port's switch, applied to the modules alone,
  changes) from commits, and regenerates `bundles/<port>/` (the module files its
  tests and benches import, under flattened path names, the tests, `kbench.py`
  and `candidate.json`) around its bench scripts.
- `make_configs.py`: writes `control.json` (the modules over the r6c TP4 recipe)
  and one arm per port (the modules with its switch files over them).
- `kbench.py`: the benches' shared CUDA-graph timing (warm, and cold after an
  L2 eviction) and error helpers.
- `bundles/<port>/`: each port's kernel bundle (`candidate.json`, its bench).
- `profile_*.py`, `summarize_kernels.py`, `analyze_costs.py`, `tables_arms.py`:
  the lab runner's profile and table scripts (from
  `experiments/2026-09-29-determinism`).
