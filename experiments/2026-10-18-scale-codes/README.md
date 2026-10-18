# Routed-expert scales in 4-bit codes, losslessly

Owner, 2026-10-17: "there was a compression opportunity for some weights that was
lossless. if we implemented it rightly and given that we're memory bandwidth starved,
maybe we would get some performance out of it? at the least, maybe we could get the 3x
spark recipe to 1M context?"

## What the scales are

Every 32 routed-expert weights (E2M1) carry one UE8M0 scale: 15.8 GiB in the checkpoint for
the target's experts, 0.4 GiB for the drafter's, 5.9% of the experts' bytes. Per rank that is
5.4 GiB at TP3 and 4.0 GiB at TP4. The GEMM reads them as packed words, four exponents per
128 K columns per output row.

A sample of the 1017 checkpoint's scales (layers 0, 1, 2, 10, 20, 30, 39 and the drafter,
about 18% of the scale tensors, read on dgx1 2026-10-17): 17 distinct exponents (111-127);
120 and 121 are 95.5% of them; every output row spans at most 8 exponents (90% span 2 or
fewer); the entropy of (row maximum - exponent) is 1.05 bits. A base exponent per row and a
4-bit delta per scale are therefore exact, with no exceptions, at half the bytes; 3 bits would
also be exact in the sample, and 2 bits would need exceptions for 0.0035% of scales.

## Ceilings

- **Speed (decode):** the routed GEMMs are 64% of a 16-K-context step's span at eight
  streams and 43% at sixteen (2026-10-13 profile, TP4). Halving the scale bytes removes 2.9%
  of their weight bytes, so if those GEMMs run at the memory system's limit the bound is
  1.3-1.9% of a decode step. Prefill is compute-bound here: no gain expected.
- **Memory:** 2.7 GiB a rank at TP3 (2.0 at TP4), but only if the full-size scales never land
  in GPU memory: on GB10 memory freed after loading stays charged to the process, so the
  codes must be made while loading (the checkpoint loader, today B12X's).
- **TP3 at 1M:** the KV cache is replicated on every rank (0.81M tokens a GiB on both
  shapes). Today's 3.5 GiB pool already holds 2.7 full 1M contexts; with the scales coded
  it would hold about 4.8, the per-sequence headroom TP4 has at 1M.

## Change (vLLM branch moe-scale-codes, on r6w's 0181)

- `pack_scale_codes(exponents)` (gemm.py): a base exponent per row and four 4-bit deltas per
  word, refusing a row whose exponents span more than 16; `expand_scale_codes` gives back the
  words byte for byte.
- `grouped_fp8_fp4_gemm(..., coded_scales=True)` (moe.py): a stage loads its tile's `block_N`
  codes in 16-byte copies (`tile_codes` lays each tile's codes for one K block contiguously;
  the uncoded words are 4-byte copies with a row stride), and each thread expands its row's
  code into the scale word after the stage lands, before the barrier the MMA already waits
  on. The words, and so the output bits, are the uncoded ones.

## Runs

1. The full scan of every layer's scales (read-only, CPU), to confirm no row exceeds 16.
2. `kernel.json`: `bundles/codes-bench`, the coded GEMM against the uncoded one at TP4 and TP3
   gate/up and down shapes, decode captures and prefill chunks: bits, then time.

Only if the coded GEMM is never slower and faster at decode does the next step (codes made
while loading, then the TP3 1M sizing) follow.
