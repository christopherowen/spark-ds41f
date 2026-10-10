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

Owner, 2026-10-10: correctness comes before speed. Where a component is not
deterministic at temperature 0, prefer the deterministic one; tune for speed
beyond that, meeting or beating the current kernel, until the other engines can
be removed.

1. **Deterministic.** The same bits on every run (graph replays, warm and
   L2-evicted, against an eager call: `kbench.repeatable`) and for every row
   count (a row's output equals that row of the full batch). Atomics may only
   elect which CTA reduces; every sum keeps one order.
2. **Same results.** Bit-identical to the kernel it replaces at every serving
   shape, or, where we follow DeepSeek's reference semantics, no worse than
   that reference.
3. **No slower per call.** Timed under CUDA graphs at the serving shapes, with
   the weights warm and cold in L2 and beside the L2 prefetch where the kernel
   overlaps it.
4. **Serving.** A lean screen against the control in the same window: one
   boot per arm, c1, c8 and c16 distinct streams, three samples, bracketed, and
   the temperature-0 output check (`token_determinism.py`). The port's step
   time must be within noise of the control or better, with quality 5/5, and
   its outputs no less deterministic than the control's.
5. **Then the old kernel goes.** Once a port passes, the replaced kernel and
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
everywhere, at every row count, warm and cold, after determinism (see the
gate). Numbers in parentheses are windows.

| Port | Switch commit | Kernels (latest window) | Serving screen | Next |
| --- | --- | --- | --- | --- |
| C1 L2 prefetch | `c916fed96` (module; environment switch) | **pass** (1): a weight reads 15-27% faster after the TileLang prefetch than after the CuTe one | level with the control (1) | ready |
| C2 sparknet | sparknet `b61660f` | bit-identical (2026-10-05) | TileLang level with CuTe (1) | ready |
| B5 compressor projection | `7b4c48900` | (3) FP32 path faster everywhere, error below DeepSeek's reference; BF16 path 4-15% slower cold at 1-2 and 48-96 rows | c8 -4.4% (1, before split-K) | BF16 cold tiles |
| B6 DSpark context KV | `c4c3e932e` | (3) faster at 1-72 rows and 8192; slower at 96 cold, 512 and 2048 | not screened | prefill tiles and splits past 96 rows |
| B7 Engram gate | `ef0fc64b0` | **pass** (1): 0.39-0.87x B12X at every size, error equal to B12X and TileKernels | not screened | window 4 arm |
| B2 RoPE | `c52f1dca7` | (2) faster nearly everywhere; one-head roles now share blocks between tokens (`e03a76cb9`) | not screened | window 4 arm |
| B9 Engram hash | `ff9b245d9` | **pass** (1): exact; 22 us a step eager instead of 100, 2.4 us instead of 10 under graphs | not screened | window 4 arm |
| B1 WO projection | `f1e696017` | (3) faster warm at 1-48 rows and 8192; slower cold and at 64-2048 | not screened | the activation cast inside the GEMM |
| B8 index head weights | `e047bf5ff` | (2) bits equal; faster at decode, level past 64 rows (D1 splits them) | not screened | window 4 arm |
| C3 collectives | `c1d379282` (module `888faa681`) + dispatch 2 MiB | GPU tests pending (host exchange tests pass) | not screened | window 5 arm |
| Chunk alignment | `e5cc1f4ce` (module `e0c06e715`) + threshold 8096 | scheduler unit tests | not screened | window 5 arm |
| B3 cache writers | `1eaf1605b` (modules `f236ddec2`, `69dd2c27b`) | record tests against a torch replica of B12X's arithmetic | not screened | window 6 kernels and arm |
| D1 decode rows | `6ea478c9e` | (3) same bits; faster at 65-128 rows (router to 256, indexer head weights to 1024) | not screened | window 4 arm (c16) |

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

### Determinism of r6c (source audit, 2026-10-10)

What r6c does to a row's bits when its step changes size:

- **Collectives** (the largest gap). sparknet's one-shot all-reduce adds the
  four ranks in order 0-3 in FP32 and rounds once: the same bits at any batch
  size, rank or restart. But it takes messages up to its 1 MiB dispatch limit
  only: hidden-state all-reduces of 103-204 rows go to NCCL all-reduce, and
  forwards of 205 rows or more switch to prefill sequence parallelism, whose
  reduce-scatter is plain NCCL. NCCL's ring order depends on where a row falls
  in the message and rounds to BF16 after every hop, so a decode row that
  shares a step with a prefill chunk gets different bits than alone. The
  drafter (at most 96 rows) always reduces one-shot.
- **TileKernels and the TileLang kernels** (mHC, router, MoE grouping and
  combine, norms, SwiGLU, activation cast) are row-invariant by construction;
  the switches at 64 rows (router split-K, block-32 decode tiles) claim equal
  bits, which the kernel benches check against the full batch.

- **B12X's compressor projection (B5)** picks SIMT (up to 8 rows), cuBLAS
  (ratio 1, 9-255 rows) or its TMA prefill kernel by row count, with
  different bits; the ratio-2 compressor's paired state (B4) inherits them.
  The other B12X kernels on this path are row-invariant.

Before the ports, window 4 measures the control's outputs at temperature 0:
alone, at c8, staggered, beside a long prefill, from the prefix cache and run
to run (`token_determinism.py`).

Window 4 measured it on the control (r6c with the modules): outputs repeat run
to run and from the prefix cache, but **every prompt's tokens change at c8,
staggered and beside a long prefill** (first changed token 3-71 of 192; the
first token's logprob already differs for most, since eight short prompts
prefill together in one step of about 240 rows, past the sequence-parallel
threshold).

### C3: one arithmetic for every hidden-state reduction

- **Dispatch.** sparknet's one-shot all-reduce takes messages up to its 2 MiB
  capacity (204 rows, the last step before sequence parallelism) instead of
  1 MiB. NCCL was faster there (235 against 314 us at 2 MiB in the balanced
  policy screen), so this costs time on 103-204-row steps; `oneshot-dispatch`
  prices it alone.
- **Reduce-scatter.** Under sequence parallelism the TP reduce-scatter
  exchanges the chunks unreduced (NCCL send/recv, the same bytes) and adds
  them as the one-shot does: FP32, rank order, one rounding (vLLM module
  `tilelang/collectives.py`, a TileLang kernel reading the rank's own chunk in
  place). On ring4 the opposite ranks are not cabled, so each half of the
  opposite chunk is relayed through one neighbour: every link carries M/2 per
  direction against NCCL's 3M/8. The r5o research did the same exchange
  directly on the TP3 triangle (vLLM 0031/0041) for +17 ms at 16K tokens.
- **Chunk alignment.** A prefill chunk takes whatever budget the step's other
  requests leave, so where a long prompt splits (and with it the CED decoder
  windows at a chunk end) depends on other traffic. Under the TileLang family
  chunks end at absolute multiples of the long-prefill threshold, now 8096: the
  8192-token budget less 16 streams of 6 decode rows, so a whole chunk always
  fits beside them and B12X's 8192-row plans still cover every step (the r5o
  research's vLLM 0036, there at 4096).
- `deterministic` combines C3, B5 and chunk alignment (the determinism breaks
  the audit found).

### Window 3 (2026-10-10, 06:54-07:04 UTC, kernels only)

- B5 (one-launch split-K, blocked): the FP32 path is faster everywhere
  (0.10-0.98x B12X) at an error of 1.6e-7, below DeepSeek's FP32 reference
  (1.0e-6). The BF16 path is faster warm but 4-8% slower cold at 1-2 rows and
  5-15% at 48-96.
- B6: faster at 1-72 rows and 8192; slower at 96 cold (1.06x), 512 (1.14x) and
  2048 (1.03x). The sweep stopped on a repeated B12X plan name (fixed).
- B1: faster warm at 1-48 rows; 3-5% slower cold at small rows, 1.1-1.2x at
  64-96, 1.27-1.33x at 512, 1.07x at 2048, 0.73-0.76x at 8192. The activation
  cast reads 32 MB and writes 16 MB at 2048 rows, DRAM-bound (231 us); B12X
  quantizes inside its GEMM. The breakdown's warm parts each kept their data
  in L2, so they summed below the whole call; it now reports cold times too
  and skips tiles that do not launch.
- Production projections at 65-128 rows (`decode-rows`): every decode tile and
  split gives the prefill path's bits. Decode tiles win for the fused Q-A/KV
  (32-row tiles to 96 rows, 64 to 128), the shared experts' gate/up (2.4x
  warm) and down (32-row tiles to 96); split-K wins for the router to 256 rows
  and the indexer head weights to 1024 (31 us down to 5-16 us). Q-B, the
  indexer's Q-B and the DSpark main projection keep their prefill tiles. This
  is port D1, switch `f2f8b602a` (module `a48174738` adds the empty tables).

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

### B3: KV-cache record writers

DeepSeek-V4.1's two record formats, written by one TileLang thread per
quantization group with B12X `write_cache`'s arithmetic step for step:

- `swa` (528 bytes): 16 groups of 32 values as E4M3, then 16 UE8M0 scales; a
  scale is `max(amax, 1e-4) * fp32(1/448)` rounded up to a power of two, and
  the values are scaled by its exact reciprocal.
- `indexed` (288 bytes): 32 groups of 16 values as packed E2M1, then 32 E4M3
  scales; a scale is `max(amax, 6 * 2^-9) / 6` (IEEE), saturating to E4M3, and
  the values are divided by its decoded value before nearest-even E2M1.

The two attention call sites go through one helper (`_write_v41_cache`, a
module commit), so the switch touches only its body, apart from B2's RoPE
lines beside the calls.

### Later: programmatic dependent launch

B12X and TileKernels kernels launch with programmatic dependent launch, so a
kernel's prologue overlaps its predecessor's tail inside CUDA graphs. TileLang
has `T.pdl_trigger` / `T.pdl_sync`; a decode step runs on the order of a
thousand kernels, so the TileLang family should adopt it once the ports are
in (B2's 0.05-0.14 us misses on 1.2 us calls are the size of that overlap).

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

Window 5 (`w5.json`): the C3, B9, B5 (with a decode-tile sweep for its BF16
path), B1 (with B12X's batch invariance reported) and chunk-alignment
bundles, then the lean screen with the temperature-0 check (a long prompt of
about 10,000 tokens, past one chunk), c1/c8 distinct streams and
mixed-traffic latency for the control, `oneshot-dispatch`, B5, C3, chunk
alignment and `deterministic`, then the arms window 4 did not reach (B9, B2,
B8), bracketed by the control.

Window 5 results (2026-10-10, 08:15-09:27 UTC), lean screen bracketed by the
control (control-end within 0.2% of it):

| Arm | Temperature-0 outputs (c8 / staggered / mixed / cached / again) | c1 step | c8 distinct | Prefill 1K / 16K | Mixed long TTFT, p99 gap |
| --- | --- | ---: | ---: | ---: | ---: |
| control | 8 / 8 / 8 / 0 / 0 differ | 31.49 / 37.42 ms | 176.7 | 2719 / 5643 tok/s | 893 ms, 784 ms |
| oneshot-dispatch | 8 / 8 / 9 / 0 / 0 | +0.2% / +0.5% | -1.7% | -0.6% / -0.1% | +1.0%, +0.5% |
| B5 compressor | 8 / **0** / 8 / 0 / 0 | +0.3% / 0.0% | -1.1% | -0.3% / +2.3% | -2.5%, -1.5% |
| C3 collectives | 8 / 8 / 8 / 0 / 0 | +0.4% / +0.1% | -2.4% | -2.1% / -3.8% | +2.8%, +3.4% |
| chunk alignment | 8 / 8 / 9 / 0 / 0 | +0.2% / -0.3% | +1.4% | +0.3% / +0.6% | +0.1%, -1.0% |
| **deterministic** | **0 / 0 / 0 / 0 / 0 (tokens and logprobs)** | +0.1% / -0.1% | +0.5% | -3.0% / -1.7% | +2.7%, +3.5% |
| B9 Engram hash | 8 / 8 / 8 / 0 / 0 | +0.2% / -0.3% | +1.4% | +0.1% / +0.6% | +1.0%, +0.2% |
| B2 RoPE | 8 / 8 / 8 / 0 / 0 (text shifts) | -0.8% / -1.4% | -1.1% | +0.6% / +2.7% | +0.4%, -0.9% |
| B8 index weights | 8 / 8 / 8 / 0 / 0 | -0.1% / -0.3% | +0.5% | -1.0% / -0.1% | -0.2%, +0.5% |

C3, B5 and chunk alignment together make the outputs independent of the company
they keep: each prompt's 192 tokens and their logprobs are bit-identical alone,
among eight, arriving staggered, beside a 10,000-token prefill that crosses a
chunk boundary, from the prefix cache, and run to run. B5 alone fixes the
staggered arrivals (short prompts prefilling beside decodes); c8 and mixed need
the collectives and aligned chunks as well. Decode is level; the price is in
prefill (2-4%), from C3's reduce-scatter exchange, which is next to recover.
The kernels: C3's sum and exchange tests pass, B9 passes with int32 history,
B5's bench confirms B12X's compressor varies with the batch (ratio 1 at 4-96
rows, ratio 2 everywhere) while the port does not, and its tile sweep beats
B12X up to 32 rows and is within 1% at 33-96 (module `747a2256b`). The
chunk-alignment bundle mounted no scheduler and failed (fixed).

Window 4 results (2026-10-10, 07:37-08:06 UTC): kernels pass for B7, B9, B8
and D1; every bench is repeatable (200 graph replays per size, warm and
L2-evicted); B2 misses by 0.05-0.14 us on 1.2 us calls (index query and index
key, warm; cold equal); B5, B6 and B1 as in window 3. Serving: D1 is level or
better (c8 +1.3%, c16 +0.7%) with the control's outputs, accepted drafts and
bits; B7 is level (its DeepSeek arithmetic changes the text, so acceptance and
tok/s move: prose +12%, JSON -2.6%, c8 +1.3%). B9's arm failed its profile
run: the runner keeps lookback tokens as int32 and the kernel read int64
(fixed in module `e859afc53`; the bench now uses int32), which stopped the
run before B2 and B8.

Window 4 (`w4.json`): every kernel bundle, now with graph-replay repeatability
(B5, B9; B6; B1, B8; D1, B2, B7 on dgx1-4), then the lean screen with the
temperature-0 output check and c1/c8/c16 distinct streams for the control, D1,
B7, B9, B2 and B8, bracketed by the control.

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
  L2 eviction), error helpers and graph-replay repeatability check.
- `token_determinism.py`: the temperature-0 output check of a serving arm
  (tokens and logprobs against each prompt served alone).
- `bundles/<port>/`: each port's kernel bundle (`candidate.json`, its bench).
- `profile_*.py`, `summarize_kernels.py`, `analyze_costs.py`, `tables_arms.py`:
  the lab runner's profile and table scripts (from
  `experiments/2026-09-29-determinism`).
