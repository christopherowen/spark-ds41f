# Retiring B12X's WO: WO-A as DeepSeek's reference, in TileLang (port B1 on r6j)

Owner, 2026-10-13: match the reference, quantize only where the reference quantizes,
correctness first and then speed; work until TileLang is well ahead and the non-TileLang
version can be retired.

## Where the reference quantizes

DeepSeek's `inference/model.py` (in the checkpoint) quantizes an activation only inside
`linear()` for FP8 and FP4 weights (`act_quant(x, 32, "ue8m0", float8_e8m0fnu)`, then
`fp8_gemm`). The attention's WO-A is the exception: `wo_a` is declared BF16
(`convert.py` dequantizes it) and applied to the BF16 attention output with an einsum.
WO-B is a `linear()`, so its MXFP8 activations are the reference's. Production (B12X's
fused `wo_projection`) quantizes WO-A's activations too: the last known deviation from
the reference.

[check_cast_reference.py](check_cast_reference.py) compares our activation cast
(TileKernels `per_token_cast`, E4M3, 32-element UE8M0 blocks) with the reference's own
`act_quant` from the checkpoint's `kernel.py`, byte for byte: random data at six
magnitudes, heavy tails, all-zero blocks, blocks below the reference's `1e-4` amax floor,
BF16 subnormals, block maxima on exact powers of two, values at the clamp, and 1-200
rows. Run on dgx4 in the r6j image (2026-10-13): **identical everywhere**, so every FP8
projection already quantizes exactly as the reference, and a cast fused into a GEMM stays
the reference's arithmetic if it keeps these steps.

## The change

vLLM branch `wo-bf16-r6j` (on r6j's `151ea817a`): the `r6d-wo-bf16` work of
[2026-10-11-wo-bf16](../2026-10-11-wo-bf16/README.md) and its sequence-parallel fix,
cherry-picked unchanged, plus its two kernels named in the scheme (`gemm_fp8w_bf16`,
`gemm_fp8w_bf16_partials`):

- WO-A as a grouped W8A16 GEMM: each block-32 FP8 weight tile expanded exactly to BF16
  (E4M3 value times its UE8M0 power of two) and multiplied with the BF16 activations on
  BF16 tensor cores, FP32 accumulation; four K shards, summed in a fixed order whatever
  the path, so a row's bits do not depend on the batch.
- WO-B with the reference's MXFP8 activations; sequence-parallel steps through the
  rank-order projection.
- The attention runs `TileLangWOProjection` under the TileLang family instead of B12X's
  fused kernel.

The output moves toward the reference (new temperature-0 references).

## Windows

- [kernels.json](kernels.json): the WO tests and [bench_wo.py](bench_wo.py) against
  B12X's fused WO (error against FP64 no worse, batch invariance, repeatability, time per
  call warm and cold at every decode and prefill size; fails if TileLang is more than 3%
  slower anywhere). Its own spec, so a speed failure does not stop the serving screen.
- [screen.json](screen.json): r6j + WO ([wo.json](wo.json), the vLLM files mounted over
  the r6j image) / r6j ([r6j.json](r6j.json)) / r6j + WO again: determinism `--wide`
  (each arm identical within itself), quality, the paired sweep, four, eight and sixteen
  different prompts, prefill at 32K and 256K.
- [sweep2.json](sweep2.json): [sweep_wo_a_v2.py](sweep_wo_a_v2.py), WO-A's tiles at four
  shards that the first sweep did not try: 96- and 128-row tiles at 72-128 rows, and
  wider tiles, `block_K` 128, three stages and 256 threads from 256 rows.

## Screen results (2026-10-13, 16:16-16:45 UTC)

| against r6j | wo | wo again |
| --- | ---: | ---: |
| quality | 5/5 | 5/5 |
| temperature 0 | **mixed15 7 of 16 differ** (from the first token's logprob) | **7 of 16** |
| four / eight different prompts | **+2.4% / +2.4%** | **+1.9% / +2.2%** |
| sixteen, paired sweep | level | level |
| prefill 32K / 256K | level | level / -0.3% |

Faster at four and eight prompts, but not deterministic in mixed15. The same seven
requests differ in both arms, each from its first token's logprob: the prompt's prefill
already differs. With `--max-parallel-prefills 1` each mixed15 prompt prefills in its own
forward, beside the other streams' verification rows (well under the 205-token SP
threshold), where alone it is the whole forward. Not the kernels:
[check_wo_invariance.py](check_wo_invariance.py) found every row count 1-300 and every
boundary to 4096, at offsets 0, 7, 100 and 1000, identical to the same rows of one
8192-row call. The attention output is an ordinary allocation, and the TileLang and B12X
WO paths hand the same BF16 partials to the same `project_reduce_scatter` and
`all_reduce`.

[debug.json](debug.json) bisects in serving with temporary switches (vLLM
`wo-determinism-debug` `d0bb54446`, `VLLM_DS41_WO_DEBUG`): a copy of the attention output
instead of rotating it in place, a private scratch instead of the shared workspace, and
one slice instead of two, each alone and all together; determinism only.

## Debug results (2026-10-13, 16:58-17:15 UTC)

| arm | temperature 0 |
| --- | --- |
| copy of the attention output | mixed15 7 of 16 differ |
| private scratch (SP path) | 7 of 16 |
| one slice | 7 of 16 |
| all three | 7 of 16 |

Every arm differs in the same seven requests at the same tokens as the screen, with the same
references (`faa92269c192` / `a41b5636b5bd`): deterministic, and none of the three causes.
The private scratch only covered the sequence-parallel path, and the mixed15 prefills
run on the other one, where the WO still borrows vLLM's shared workspace (WO-A's split-K
partials, up to 4 MiB at 128 rows, and WO-B's activation cast), so the workspace is not
ruled out.

[trace.json](trace.json) and [debug-privateall.json](debug-privateall.json) (vLLM
`wo-row-trace` `03437d1d9`, DEBUG ONLY, mounted as [overlay-trace](overlay-trace)):

- privateall: the WO's scratch private on every path (one buffer per device for the
  target, one for the drafter);
- trace: `VLLM_DS41_ROW_TRACE` hashes every row of each attention layer's input, its
  output, the WO partial and the summed WO result on TP rank 0, in every eager forward
  with a prefill. [compare_row_trace.py](compare_row_trace.py) pairs each prompt's
  prefill alone with its prefill beside other rows and prints the first layer and stage
  whose rows differ.

## WO-A sweep 2 (2026-10-13, 16:47-16:58 UTC)

Four shards, best configuration per row count (warm / cold µs, CUDA graphs):

| rows | today | best | path and tile |
| ---: | ---: | ---: | --- |
| 64 | 19.4 / 53.5 | same | split 64x64x64 |
| 72 | 34.5 (split 64) | **27.9 / 57.1** | split 96x64x64 |
| 96 | 34.6 (in-kernel 32) | **29.5 / 59.9** | split 96x64x64 |
| 128 | 38.9 / 72.7 | **38.2 / 64.3** | split 128x64x64 |
| 256 | 69.1 / 74.2 | **61.1 / 81.9** | in-kernel 96x64x64 |
| 512 | 131.8 / 139.3 | **119.1 / 139.3** | in-kernel 96x64x64 |
| 2048 | 459.5 / 468.4 | 455.3 / 471.4 | in-kernel 96x64x64 |
| 8192 | 1757.7 / 1770.4 | **1675.3 / 1685.7** | in-kernel 96x64x64 |

96-row tiles win from 65 rows on, split at decode and in-kernel past 128 (the 72-96 bucket
moves from in-kernel 32 to split 96). Every `block_M` 128 / `block_K` 128 in-kernel
configuration failed to compile (shared memory). Bits do not depend on the tile. Adopted
after the determinism fix, so the screen judges one change.

## Trace results (2026-10-13, 17:35-17:56 UTC)

- **privateall** (the WO's scratch private on every path): mixed15 still 7 of 16, the
  same requests and tokens. The workspace is not the cause.
- **trace**: mixed15's fifteen prompts prefill in **one forward of 263 rows**, which is
  past the 205-token threshold and so sequence-parallel; each prompt alone (14-22 rows) is
  not. Batches below the threshold are equal to the prompt alone at every stage (eight
  prompts in 144 rows: all 160 records). In the SP forward, seven prompts first differ
  at layer 1 or 2's attention output, where layer 0's was equal.
- **trace with the B12X WO** (`VLLM_DS41_WO_DEBUG=b12x`, mixed15 0 of 16 differ):
  the same SP forward also differs from the prompts alone inside, single rows of layer
  1's attention output, but those differences do not reach the outputs. SP prefill was
  never bit-identical to the prompt alone inside the model; the B12X WO's outputs hid it.
- **trace with the TileLang WO**, every rank: the earliest difference is layer 0's summed
  WO result for one row on rank 1, whose four partials (each rank's `wo`) equal the prompt
  alone. The SP sum (the exchanged parts added in rank order) and the alone sum (the one-shot
  all-reduce) are the same arithmetic, FP32 from rank 0 rounded once, so the parts that
  arrived must differ from those sent, or the trace read them early.

[trace-raw.json](trace-raw.json) keeps layers 0 and 1's WO partials, the received parts
and both sums raw (vLLM `wo-row-trace`), to recompute the sums offline.

## Raw sums (2026-10-13, 18:00-18:05 UTC)

[check_raw_sums.py](check_raw_sums.py) on the raw trace: every received part equals the
partial its sender computed, and every SP sum equals the rank-order sum of its parts. The
collectives are exact. The difference is earlier: on rank 2, layer 0's **attention output**
for one row (row 2 of a 17-row prompt) differs between the 263-row SP forward and the
prompt alone, whose three runs agree with each other; ranks 0, 1 and 3 are equal. The B12X
WO's trace has the same row on rank 2 differing at layer 0. So SP prefill was never
bit-identical in the attention; the B12X WO quantizes its input to FP8, which absorbed a
one-ulp difference, and the TileLang WO keeps the reference's BF16 and passes it on.

[trace-q.json](trace-q.json) records the query path (`qr`, `kv`, `q` after `wq_b`, `q`
after RoPE) in every forward, raw for layers 0-1 in forwards up to 512 rows.

## Query-path trace (2026-10-13, 18:27-18:36 UTC)

On rank 2, the prompt whose layer-0 output differed has the same `qr`, `kv`, `q` and rotated
`q` as alone: **the sparse MLA kernel** returns different bits for the same query and keys.
The plan routes a forward by its rows: for the SWA-only layer 0 at TP4 (one 128-entry
segment, 16 heads, 48 SMs) `split` to 48 rows, `decode` to 144, `extend` past that. Alone
(14-22 rows) is `split`, the eight-prompt batch (144) `decode` and equal, mixed15 (263)
`extend` and different. The kernels are built with `TL_ENABLE_FAST_MATH`.
[check_mla_routes.py](check_mla_routes.py) ([mla-routes.json](mla-routes.json)) compares
the routes on 1024 rows with short and full key lists.

### Route check (2026-10-13, 18:37-18:38 UTC)

**Reproduced without the model.** On SWA-only layers `extend` differs from `decode` and
`split` (which agree): 1-6 elements per 1024 rows at 16 and 24 heads, mostly in rows with at
most 8 keys; indexed layers agree. The routes share their formulas; under fast math the
compiler lowers extend's in-kernel segment fold (vectorized `fma2`, paired `exp2f`) and
reciprocal differently from decode's combine kernel. [bundles/mla-ieee](bundles/mla-ieee)
([mla-ieee.json](mla-ieee.json)) runs the route check and times each route
([time_mla.py](time_mla.py)) with fast math on and off.

### IEEE math (2026-10-13, 18:41-18:44 UTC)

**Not the cause**: with fast math off the routes still differ (16 heads: none in three seeds;
24 heads: 3-4 elements per 1024 rows, all in rows with at most 8 keys), and no route is
measurably faster or slower either way (TP4 SWA-only: 12.4 / 28.7-30.8 / 73-75 us at 1 / 96 /
263 rows). The extend kernel computes something its source does not share with decode.
[bundles/mla-source](bundles/mla-source) prints both kernels' generated source and checks
rows with exactly one, two and three keys.

### The cause (2026-10-13, 18:51 UTC, [bundles/mla-fold](bundles/mla-fold))

Recomputing the fold from decode's per-segment state: extend's output is
`L = fma(l, b, a)` (one rounding) on **every** differing element (30 of 30 at 16 heads, 52 of
52 at 24), decode's is `L = a + round(l * b)`. The source of both is `fma(L, a, l * b)`. On
SWA-only layers extend has one segment, so its fold unrolls with the constant start `L = 1`;
the compiler rewrites `fma(1, a, l * b)` as `a + l * b` and, the vectorized multiply being an
ordinary one, contracts it into `fma(l, b, a)`. Decode's merge is a separate kernel whose
`__fmul_rn` cannot contract. Indexed layers (five segments, no unrolling) never differed;
one-key rows (`l * b` exact) never differed.

The fix (vLLM `mla-route-fold` `2b0f96b27`): every route folds as `fma(l, b, L * a)` and
`fma(acc, b, o * a)`, where the product the constant meets is exact and the fused product
is explicit. [bundles/mla-fix](bundles/mla-fix) ([mla-fix.json](mla-fix.json)) runs the
attention tests (with a new short-row route test), the route check and the timing.

### Fix verified (2026-10-13, 18:54-18:56 UTC)

[bundles/mla-fix](bundles/mla-fix), production fast math: the attention tests pass (14, with
the new short-row route test), every route identical at 16 and 24 heads, SWA-only and
indexed, three seeds each, and no route slower (TP4 SWA-only 12.4 / 28.7 / 73.4 us at 1 /
96 / 263 rows, as before).

## The retirement candidate's screen

[retire.json](retire.json) mounts vLLM `wo-retire` `2b0f96b27` over the r6j image
([overlay-retire](overlay-retire)): the TileLang WO (WO-A on BF16 activations as the
reference, the sweep-2 tiles, WO-B's decode tiles), the TileLang DSpark context-KV
projection (port B6) and the sparse MLA fold fix. [retire-screen.json](retire-screen.json):
retire, r6j, retire again, with the WO screen's measurements.

### Retirement screen (2026-10-13, 18:57-19:28 UTC)

| against r6j | retire | retire again |
| --- | ---: | ---: |
| temperature 0 | **identical, mixed15 0/0** (new references `70b3379cc289` / `708ec5b17482`) | identical |
| quality | 5/5 | 5/5 |
| sixteen different prompts | **+1.8%** | **+1.5%** |
| four / eight | level | level / -0.9% |
| paired sweep | level | level |
| prefill 32K / 256K | level / **-0.8%** | level / **-0.9%** |

Correct, at the reference's precision. Not yet well ahead. [bundles/retire-kernels](bundles/retire-kernels) (19:27-19:30) per call:
- **WO:** faster warm at 1-12 rows (0.77-0.99), slower at 16-96 (1.09-1.37 warm, ~1.05 cold), 512 (1.23) and 2048 (1.05); 0.69 at 8192.
- **B6:** 0.36-0.60 of B12X at 1-64 rows; level at 72-96 cold; 1.09-1.11 at 512-2048; 0.62 at 8192.
- **Why prefill loses:** SP prefill projects WO in 1024-row slices (4 owners x 2 slices of an 8192-token chunk), where TileLang costs ~0.45 us a row against B12X's ~0.41. About 1% of a chunk.
- **Why decode loses:** five kernels (RoPE, WO-A, split reduce, cast, WO-B) against B12X's one.

**Fused cast** (vLLM `wo-fused-cast` `51c752bbd`): WO-A's last kernel rounds to BF16 and applies
the reference's cast itself (the in-kernel GEMM's epilogue, or the split reduce), so WO-B
reads MXFP8 straight from WO-A. This drops the BF16 `a` round trip (~10 MB per 1024-row slice)
and two kernels. The bytes equal `per_token_cast` of the BF16 `a` (new test). [bundles/wo-fused](bundles/wo-fused)
([wo-fused.json](wo-fused.json)) runs the tests, the invariance check and
[bench_wo2.py](bench_wo2.py).

### Fused cast results (2026-10-13, 19:42-19:52 UTC)

Tests: 14 WO (the fused bytes equal `per_token_cast` on both paths), 7 block-32, invariance at
every size and offset. [bench_wo2.py](bench_wo2.py) against B12X's fused WO:

| rows | before (warm / cold) | fused (warm / cold) |
| ---: | ---: | ---: |
| 1-16 | 0.77-1.09 / 1.02-1.06 | **0.69-0.96** / 1.03-1.06 |
| 24-32 | 1.10-1.17 / 1.02-1.04 | 0.83-0.98 / 1.02-1.04 |
| 48-64 | 1.13-1.35 / 1.06-1.07 | 1.12-1.39 / 1.05 |
| 72-96 | 1.22-1.37 / 1.06-1.08 | 1.06-1.20 / 1.07-1.08 |
| 512 | 1.23 / 1.37 | 1.16 / 1.35 |
| 2048 | 1.05 / 1.07 | **1.01 / 1.02** |
| 8192 | 0.69 / 0.68 | **0.67 / 0.66** |

(Two runs differ by up to 15% warm at 64-96 rows.) WO-A's BF16 GEMM runs at a steady ~75
TFLOP/s at every size; matching B12X, whose WO-A is FP8 (half the tensor work), rests on
the rest of the pipeline. [fused.json](fused.json) ([overlay-fused](overlay-fused), vLLM
`51c752bbd`) and [fused-screen.json](fused-screen.json): fused, r6j, fused again.

### Fused screen (2026-10-13, 19:52-20:22 UTC)

| against r6j | fused | fused again |
| --- | ---: | ---: |
| temperature 0 | identical, mixed15 0/0, **the retire arm's references** | identical |
| quality | 5/5 | 5/5 |
| sixteen different prompts | **+1.7%** | **+1.7%** |
| four / eight | level / -0.8% | level / -0.7% |
| paired sweep | level | level |
| prefill 32K / 256K | level / -0.6% | level / -0.8% |

The fused cast serves the same bits as the unfused WO and moves nothing measurable in serving.
The candidate's text differs from r6j's (new references), so distinct-prompt and sweep
numbers mix speed with acceptance (-1.4% +-5.2% accepted). [fused-fixed-profile.json](fused-fixed-profile.json)
compares kernel time per step at fixed rows with adaptive verification off (code16: 96
rows a step; c8: 48; one 16K prefill), fused first, using [analyze_costs.py](analyze_costs.py)
with families for the TileLang kernel names.

### Inverse RoPE in WO-A (vLLM `wo-rope-fused` `f806f5103`)

WO-A rotates each head's last 64 columns back as it loads that K block (`rope_`'s
arithmetic in shared memory), so the separate RoPE kernel and the in-place write of the
attention output go: decode WO is two kernels on the in-kernel path, three on the split
path (B12X: one). [bundles/wo-rope](bundles/wo-rope) ([wo-rope.json](wo-rope.json)): the
tests (the cast bytes equal `per_token_cast` of WO-A on `rope_`'s output, and the input
is not written), invariance, [bench_wo3.py](bench_wo3.py).

### Fixed-row profile (2026-10-13, 20:39-20:46 UTC)

Kernel time per step, adaptive verification off, rank 0, r6j -> fused (154 target steps each;
the step mix differs a little because the text does):

| workload | span ms/step | kernel ms/step | WO (B12X dense + its RoPE -> TileLang WO-A + WO-B + RoPE) |
| --- | ---: | ---: | --- |
| code16 (96 rows) | 45.20 -> 45.30 | 52.47 -> 52.63 | 1.65 -> ~1.70 ms/step (level, -3%) |
| c8 (48 rows) | 29.94 -> 31.67 | 32.77 -> 34.86 | +0.17 ms/step; the rest is MoE routing of a different text (+0.87) |
| prefill 16K (window) | 2808 vs 2799 | **2774 vs 2799** | WO ~5 ms cheaper; NCCL waits 34 ms shorter |

At prefill the candidate's kernels take 25 ms less but the window is 9 ms longer: the time
is in gaps, not GPU work. At decode the WO is level at 96 rows and slower at 48, where the
kernel bench put TileLang at 1.12-1.39x warm.

### Inverse RoPE in WO-A: dropped

- **f806f5103 did not compile.** It rotated `A_shared` in place after its copy, and TileLang's pipeline planner allows one write per stage.
- **2d1c6b2a2 compiles and is correct, but far slower.** It copies to a raw tile and rotates as the tile moves to `A_shared`. The 14 WO tests pass, including the fused bytes against `rope_` + WO-A + `per_token_cast`, and invariance passes. But that elementwise pass runs over every K block: WO-A at 96 rows 29 -> 160 us, at 2048 rows 453 -> 2563 us.
- **Not pursued further.** Rotating only the RoPE block needs the K loop split per head, too much for ~4 us a decode step. r6k keeps the fused cast without it.

### WO-A tiles at 24-128 rows on the fused path (2026-10-13, 23:07-23:19 UTC)

[bundles/wo-a-sweep3](bundles/wo-a-sweep3) ([sweep_wo_a_v3.py](sweep_wo_a_v3.py), image r6k, 73
configs):
- Every working config gives the same bytes. The six with 128-wide N tiles do not build.
- Warm, 48-row split tiles beat today's at 40-48 rows by 2 us (16.4-17.1 against 18.6-18.9
  us). Cold they are level (50.1-51.2 against 50.2-50.4).
- Today's tiles are the fastest at 56-96 rows and 1024.

Serving is the cold case: each layer's WO weights are read once a step. The tiles stay.

### Ceiling of fusing the split-K reduce (from the fixed-row profile)

At c8 (48 rows, rank 0), per call and per target step:

| kernel | us/call | ms/step |
| --- | ---: | ---: |
| `gemm_fp8w_bf16_partials` (WO-A) | 23.5 | 0.322 |
| `gemm_splitk_reduce_cast` | 1.9 | 0.016 |

Folding the reduce into the GEMM (last split CTA reduces) would save that kernel and its
launch gap: about 0.03 ms of a 31.7 ms step, 0.1%. Not built. The 48-row deficit is WO-A
itself, a BF16 MMA on FP8 weights widened in the main loop. A warp-specialized W8A16 GEMM
(producer warps load and widen, consumers multiply) is the lever left there, bounded at
about 0.3% of a step if it matched an FP8 GEMM's weight-read time.
