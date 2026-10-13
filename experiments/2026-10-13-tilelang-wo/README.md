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
