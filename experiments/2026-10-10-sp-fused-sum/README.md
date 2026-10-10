# Add the reduce-scatter's parts inside mHC (r6d + overlap)

Owner, 2026-10-10: after the r6d candidate's benchmark (step 1) and the
reduce-scatter overlap ([2026-10-10-sp-overlap](../2026-10-10-sp-overlap/),
step 2), fuse the rank-order sum into its consumer (step 3).

## Why, and the ceiling

r6d's deterministic reduce-scatter (C3) exchanges every rank's partial of a
rank's rows unreduced, then adds the four parts in FP32 in rank order in a
separate kernel: at an 8192-token step it reads 84 MB and writes 21 MB per
rank (0.55 ms measured, window 6), and the next mHC reads the 21 MB back. With
the overlap, the WO result is also copied once more (`out[lo:hi].copy_`). mHC's
`post_pre`, which every one of these results feeds, already streams the
residual: if it loads the four parts and adds them itself, the sum's write, its
re-read and the copy go (126 MB down to 84 MB of traffic per call).

Bound: about 0.17 ms per call at the memory system's rate, up to about 0.3 ms
if the separate sum runs below it; two calls per layer give 15-26 ms of a
1.4 s 8096-token chunk, **1-2% of long prefill**. Decode never
reduce-scatters (one-shot all-reduce up to 204 rows) and is untouched.

## What changes (vLLM branch `r6d-fused-sum`, on `r6d-overlap`)

- `cdebef6f0` (module):
  - `exchange_into` receives into a caller's buffer, including row ranges of
    a larger one;
  - `RankOrderParts` holds every rank's partial of this rank's rows in one
    `(world, L, hidden)` tensor;
  - TileLang mHC's `post_pre` kernel takes `(world, M, H)` parts and adds them
    as it loads them, in rank order with one rounding, so its outputs have the
    bits of the summed input (`post` and the SP gather call `sum()`);
  - `SPRows` gains an unreduced mode: each rank's block is projected into a
    send buffer, this rank's own straight into its slot of the parts, and
    exchanged, slice by slice on the side stream as in the overlap;
  - the MoE runner takes a model-installed reduce-scatter hook that writes the
    shared + routed sum per row block (elementwise, the same bits);
  - the decoder layer wraps both SP results.
- `2cd839efe` (switch): `sp_prefill.UNREDUCED` on. It applies where the TP
  group's reduce-scatter is the rank-order one (the TileLang family on the
  RoCE policy, `CudaCommunicator.rank_order_reduce_scatter`).

Engram layers, the drafter's aux hidden states, the CED boundary and the final
`post` still sum first (`sum()`), as today.

## Arms ([w1.json](w1.json))

`fused.json` mounts the seven changed files over the r6d image
([make_arms.py](make_arms.py)); the control is the overlap arm. The kernel
bundle runs the mHC tests (the parts' result equal to the sum's, bit for bit,
at 2-4 ranks and 1-300 rows), the exchange tests (sliced exchange into one
buffer, simulated ranks) and the SP layout tests, in the r6d image. The lean
screen adds the temperature-0 checks, distinct-prompt decode and mixed
traffic, bracketed by the control. `token_determinism.py` now also reports
digests of its references, and the tables compare them with the control's: the
long prompt's reference (prefill sequence-parallel) must have the same bits.

Acceptance: references bit-identical to the overlap arm, temperature-0 outputs
identical, decode level, prefill faster than the overlap.

## Status

Window 1 (2026-10-10, 17:41-18:05 UTC, overlap / fused / overlap bracket):

| Measure | overlap | fused | overlap (end) |
| --- | --- | --- | --- |
| prose c1 step | 31.48 ms | 31.55 ms (+0.2%) | 31.57 ms (+0.3%) |
| json c1 step | 36.63 ms | 36.62 ms (-0.0%) | 36.91 ms (+0.8%) |
| single stream, distinct prompts | 87.49 tok/s | -0.0% | +0.1% |
| eight distinct concurrent prompts | 175.85 tok/s | +0.1% | +0.1% |
| cold prefill 1K | 2648 tok/s | 2584 (-2.4%) | 2604 (-1.7%) |
| cold prefill 16K | 5694 tok/s | 5781 (+1.5%) | 5704 (+0.2%) |
| mixed: short TTFT median | 232.0 ms | -1.4% | -3.3% |
| mixed: long TTFT mean | 904.7 ms | +1.8% | +2.2% |
| temperature 0 | identical | identical | identical |
| references against overlap | - | same bits | same bits |

The fused arm's references, the eight prompts alone and the long prompt alone
(prefilled sequence-parallel through the fused path), have the overlap arm's
bits, and every temperature-0 scenario is identical. Decode is level. Prefill at
16K is 1.4% faster than the mean of the two overlap runs, inside the 1-2%
ceiling. At 1K it is 1.6% slower than that mean, near the bracket's own spread
(1.7%), but with a cause: below 2048 tokens the unreduced path projected WO once
per rank's block (four launches, each reading the weights) where the reduced
path projects once. Fix: one projection into the send buffer and a copy of this
rank's block (2.6 MB at 1K); per-block writes stay for the MoE's elementwise add.

The kernel bundle's new SP layout test failed in every case on a latent bug in
`rank_order_sum`'s torch path (FP32 parts: `.float()` returned the first part,
which the sum then added into); serving's BF16 CUDA parts take the kernel. Fixed
with a regression test (module `52f23588b`); the mHC parts-against-sum and
exchange tests passed.
