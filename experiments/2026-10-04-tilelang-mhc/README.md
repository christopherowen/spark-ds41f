# TileLang mHC on TileKernels

Base deployment commit: the [TileLang vocabulary heads](../2026-10-04-tilelang-vocab-heads/README.md)
experiment (vLLM patches 0032–0036).

The TileLang family still ran DS4.1's hyper-connections (mHC) on B12X: about
2.1 ms of each single-stream decode step, the largest B12X compute after the
collectives. [0037](vllm/0037-tilelang-tilekernels-mhc.patch) moves them to
DeepSeek's TileKernels in the TileLang backend.

V4.1's mHC keeps four residual streams. Each sublayer's entry:

1. projects the flattened streams through `fn`, RMS-normalized;
2. splits the 24 mixes into a pre-mix, a post-mix and a 4 × 4 combination
   (softmax, then 20 Sinkhorn rounds);
3. feeds the sublayer the streams collapsed with the *incoming* pre-mix and
   RMS-normalized, returning the new pre-mix for the next sublayer (the lag).

TileKernels' references give the same arithmetic as B12X's, term for term.

- **From TileKernels:** the standalone `post` and the final stream collapse
  run as they are.
- **What TileKernels lacks here:**
  - DeepSeek moved the projection GEMM to DeepGEMM, which does not run on
    SM120/SM121;
  - TileKernels' fused pre step applies the mix it has just computed, not
    the lagged one;
  - in prefill the four streams are 335 MB per sublayer at 8,192 tokens, so
    reading them once matters.

Two TileLang kernels make each entry:

- **`project_streams`, per 128-column hidden block and 16-token tile.** It
  folds the previous sublayer's output into the streams and writes them. It
  accumulates the projection's split partials on TF32 tensor cores
  (DeepGEMM's precision) with the streams' sum of squares. It also collapses
  the streams with the incoming pre-mix in BF16, with the collapse's sum of
  squares. The first layer expands the embedding to four copies; its `fn` is
  the stream blocks summed.
- **`finalize`, per token.** Warp 0 runs TileKernels' fused reduce, RMS, mix
  split and Sinkhorn (`mhc_pre_big_fuse`), emitting the new pre-mix. The
  other warps RMS-normalize the collapse with the sublayer's weight.

The split and the token tile are fixed, so a token's results do not depend on
the batch. Decoder layers hold their mHC as `mhc`, from the kernel family; the
B12X family is unchanged.

[candidate.json](candidate.json) is the vocabulary heads candidate on the
`-tilelang-mhc-v1` image.

## Kernel results

Draft bundles on the vocabulary heads image, under CUDA graphs. One sublayer's
`post_pre` (the previous output folded in, then the entry) against B12X's,
prepared as vLLM's `B12xMHC` prepares it, with `fn` warm in L2:

| Rows | B12X | TileKernels (0037) |
| ---: | ---: | ---: |
| 1 | 6.9 µs | 13.3 µs |
| 6 | 13.6 µs | 13.5 µs |
| 16 | 31.3 µs | 13.9 µs |
| 48 | 85.5 µs | 31.9 µs |
| 64 | 113.2 µs | 40.1 µs |
| 512 | 413 µs | 342 µs |
| 8,192 | 9.91 ms | 5.50 ms |

At one row B12X's single launch is faster; from six rows the two kernels
match or beat it. Eight-stream decode verifies 48 rows, and prefill runs
8,192-token chunks.

The tests pass 14 of 14 against V4.1's FP32 lagged reference:

- pre, post_pre, the first layer's broadcast, post and both collapses;
- the projection partials;
- every result bit-identical across batch sizes.

## Procedure

Build `-tilelang-mhc-v1` and run both bundles: [tests](bundles/tests/candidate.json)
and the [mHC bench](bundles/mhc-bench/candidate.json), which also measures
B12X. Then one window boots the vocabulary heads candidate (the control) and
this arm in turn: decode at one and eight streams, prefill, and the
single-stream decode profile for each.
