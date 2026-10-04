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

## Serving results

One window against the vocabulary heads candidate:

| | Control (B12X mHC) | TileKernels mHC | Change |
| --- | ---: | ---: | --- |
| Step time, prose c1 | 32.67 ms | 33.19 ms | +1.6% [+1.1, +2.1], slower |
| Step time, code c1 | 35.75 ms | 36.58 ms | +2.3% [+1.7, +2.9], slower |
| Decode, prose c8 | 215.4 tok/s | 217.8 tok/s | +1.1% [+0.9, +1.3], faster |
| Decode, code c8 | 248.3 tok/s | 252.5 tok/s | +1.7% [−0.8, +4.2], same |
| Prefill, 32,768 tokens | 4,889 tok/s | 4,907 tok/s | +0.4% [−11.8, +12.5], same |
| Prefill, 262,144 tokens | 4,646 tok/s | 4,656 tok/s | +0.2% [−3.9, +4.3], same |

Quality passed 5 of 5. Not promotable as it stands: single-stream decode is
slower. The single-stream profile, per scheduler step:

- B12X's mHC kernels took 2.10 ms;
- the TileKernels path takes 2.79 ms over the same number of launches:
  - `project_streams` and `finalize`: 2.69 ms;
  - TileKernels' standalone post and collapse: 0.10 ms.

That is about 31 µs per sublayer against B12X's 22 µs. Single-stream
sublayers run a few rows, where B12X's kernels are faster (6.7 µs against
13.3 µs at one row in the bench). Prefill shows none of the bench's gain:
B12X prepares its prefill path for the real workload, which the bench's plan
did not. The next step is the few-row latency of `project_streams`.

## Procedure

`-tilelang-mhc-v1` passes both bundles (97 of 97 tests, 7 bench points each
side). One window booted the vocabulary heads candidate (the control) and this
arm in turn: decode at one and eight streams, prefill, and the single-stream
decode profile for each.
