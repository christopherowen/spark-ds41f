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

## Prior art: B12X's MXFP4-CSF

Upstream B12X (lil) already serves a lossless compressed form of these scales,
MXFP4-CSF (`docs/lossless-scale-compression.md` there): a byte base per row, one selector
bit per scale (base or base + 1), exception records for the rest and a raw copy of any
tile with too many exceptions, 23.4 MiB against 101.2 MiB of native words on a synthetic
E384/H5120/I576 layer (4.3x; our 4-bit codes are 2x). Its "inline" variant
(b64cb9c5, f7beda3b) rebuilds each thread's scale words in registers after the stage
barrier, as `coded_scales` does. Measured on a DS4.1 TP4 MoE layer against expanding the
CSF scales before every call, inline was 7-10% faster up to 1,024 tokens but 2.5-6.3%
slower at 2,048 and 4,096, so plans above 1,536 tokens expand once (61 us a layer) and run
the native kernel. No comparison with plain native words, which never pay an expansion,
is published, nor a serving run. What it predicts here: a decode gain is possible, the
prefill chunks are where the codes can lose, and a denser format (1 bit with exceptions)
is what the 1M-context goal would want.

## Change (vLLM branch moe-scale-codes, on r6w's 0181)

- `pack_scale_codes(exponents)` (gemm.py): a base exponent per row and four 4-bit deltas per
  word, refusing a row whose exponents span more than 16; `expand_scale_codes` gives back the
  words byte for byte.
- `grouped_fp8_fp4_gemm(..., coded_scales=True)` (moe.py): a stage loads its tile's `block_N`
  codes in 16-byte copies (`tile_codes` lays each tile's codes for one K block contiguously;
  the uncoded words are 4-byte copies with a row stride), and each thread expands its row's
  code into the scale word after the stage lands, before the barrier the MMA already waits
  on. The words, and so the output bits, are the uncoded ones.

Serving them (vLLM branch moe-scale-codes-plan, cbe4acb8f): the plan's two GEMMs read the
codes, and the TileLang experts layer keeps the codes in place of the scale words, with each
GEMM's row bases beside its K tails. `overlay/` holds the three changed files over r6v
(with r6w's 0181); `codes-tp4-overlay-1017.json` serves them on the ring.

## Runs

1. The full scan of every layer's scales (read-only, CPU), to confirm no row exceeds 16.
2. `kernel.json`: `bundles/codes-bench`, the coded GEMM against the uncoded one at TP4 and TP3
   gate/up and down shapes, decode captures and prefill chunks: bits, then time.

3. `screen.json`: determinism (the bits must equal r6w's), distinct-stream decode speed,
   then the screen suite (model, engine grid, c16, prefill), codes against r6w and codes
   again.

Codes made while loading (the memory half, then the TP3 1M sizing) follow only a serving
gain.

## Results

### Full scan (2026-10-18, dgx1, the 1017 checkpoint)

Every layer's routed-expert scales (40 layers, 1,152 tensors and 3.7 million rows each) and
the drafter's (384 tensors): in every one, 100% of rows span 8 exponents or fewer; a layer
holds 5-11 distinct exponents (117-127), the drafter 14 (111-124)
([results/scan-1017.txt](results/scan-1017.txt), `scale_dist.py` one layer at a time). The
4-bit codes are exact for the whole checkpoint, with half their range to spare; 3 bits
would be exact too.

### Kernel (2026-10-18, dgx3 bench, dgx4 tests; r6v image with the plan's moe.py)

Tests: 57 passed, among them the experts end to end on coded scales and the coded GEMM's
bits against the uncoded one. Bench: every row count's output bits equal the uncoded
GEMM's. Time per call, coded / uncoded, warm and cold:

| Shape | 1-2 rows | 3-96 rows | 512-2,048 rows | 8,192 rows |
| --- | --- | --- | --- | --- |
| TP4 gate/up (N 1,152, K 5,120) | 1.068 / 0.967, 0.954 / 0.956 | 0.941-0.945 | 0.942-0.948 | 0.999 / 0.997 |
| TP4 down (N 5,120, K 576) | 1.037 / 0.999, 1.069 / 1.004 | 0.955-0.984 | 0.975-0.980 | 1.000 |
| TP3 gate/up (N 1,536, K 5,120) | 0.845 / 0.947, 0.960 / 0.943 | 0.941-0.952 | 0.942-0.954 | 0.978 / 0.977 |
| TP3 down (N 5,120, K 768) | 1.038 / 0.986, 0.996 / 0.979 | 0.970-0.976 | 0.973-0.976 | 1.000 |

From three rows up the coded GEMM is 2-6% faster, prefill chunks included; one and two
rows with the weights in L2 cost up to 2.2 us a call more (the expansion sits between the
stage barrier and the MMA's sync, 40 times a call; cold, the same calls are level or
faster). A one-stream verification step has 3-4 rows: there gate/up saves about 5.5% and
down 3% a call, about 0.6 ms of an 18 ms step over 40 layers; the warm one- and two-row
calls (the drafter's MoE layer, once a draft step) give back a few microseconds a step. The
serving screen decides.

Tried and reverted: expanding every K block's words once per CTA into a shared table that
the MMAs read directly. Its words reached the MMA wrong (bits differed at every size) and
the one- and two-row calls were still 2.5-6% slower warm.
