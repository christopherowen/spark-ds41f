# Kernel contracts: non-finite values, independent references, a calibrated MoE check

Shah and Shrestha ("A Contract-Grade Verifier for LLM-Generated GPU Kernels",
[arXiv 2608.12700](https://arxiv.org/abs/2608.12700)) audited 2,638 generated kernels
that a loose `allclose` had accepted and found 39.5% broken in ways no tolerance
excuses. The most common defect: a kernel returns an ordinary number where the true
answer is a NaN or an infinity, which turns a visible error into silent corruption.

Our port gate is already stricter than that check: bit-identical to the replaced
kernel or within DeepSeek's reference, deterministic, and batch-invariant at every
serving shape. Three gaps remained:

1. **Non-finite values.** Only the collectives and Engram lookup tests checked them.
2. **References that leave with the old kernel.** The compression and Engram hash
   tests imported their oracles from B12X.
3. **Hand-picked tolerances.** The MoE test accepts 3e-2 against a reference that
   leaves the down input unquantized, so the kernel already differs from it by
   MXFP8 rounding.

## vLLM branch `kernel-contracts`

The branch is on r6k's patch head `aa31eb85c`, in `~/projects/vllm-ds41-kernel-contracts`.
Its patches are in [vllm/](vllm/), and the bundles carry the changed files:

| Patch | Change |
|---|---|
| 0001 | `tests/kernels/deepseek_v41_contracts.py` and contract tests for 10 kernel families |
| 0002 | compression and Engram hash oracles ported from the checkpoint's `inference/` code |
| 0003 | MoE calibration: a reference that follows the plan's arithmetic, five injected defects |
| 0004 | shared SwiGLU: compare-and-select clamps keep a NaN |
| 0005 | Engram gate: compare-and-select magnitude floor keeps a NaN |
| 0006 | indexer: the score ReLU keeps a NaN |

**Contracts (0001).** These are tolerance-free; they compare positions and bits:
- NaN and signed-infinity positions exactly as the reference's: shared SwiGLU,
  Engram gate, RoPE, indexer scores, BF16 GEMM (split-K decode rows included),
  sparse MLA.
- A NaN row becomes an all-NaN row, and every other row keeps its bits: block-32 rows
  GEMM, WO projection, mHC pre, routed experts.
- Sparse MLA reads only the records it selects. Every unselected record is `0xFF`
  bytes (NaN payloads, the largest scale), including record 0 (where masked slots
  clamp), the allocator padding and the other layers' pages. Each list also points
  past its length at poisoned records. Every route must keep the clean bits.

**What reading the code found before any run.** Four places turn a NaN into a number
where DeepSeek's reference keeps it:

| Kernel | Code | Reference | Effect |
|---|---|---|---|
| shared SwiGLU (`tilelang/activation.py`) | `min`/`max` clamp | `torch.clamp` (`Expert.forward`) | a NaN gate or up becomes ±10 |
| Engram gate (`tilelang/engram.py`) | `max(\|dot\|, 1e-6)` | `clamp_min` (`model.py` 362) | a NaN key leaves the whole stream finite |
| indexer score (`tilelang/indexer.py`) | `max(dot, 0)` | `relu_` (`model.py` 557) | a NaN head contributes zero |
| routed experts (TileKernels `66258df`, `quant/swiglu_forward_cuda.py` 210-211, 264-265) | `T.min`/`T.max` clamp | `torch.clamp` | a NaN token's experts return finite values |

`min`/`max` lower to `fminf`/`fmaxf`, which return the other operand for a NaN.
Patches 0004-0006 fix the first three. In the code generated for `sm_121a`, only those
expressions change:
- the clamps and the floor become compare-and-select;
- the ReLU keeps `max` for every non-NaN value, so signed zeros keep their bits. Its
  float2 `max2` becomes a two-iteration loop.

The TileKernels clamp needs a TileKernels patch or an upstream change, which is the
owner's decision. Until then its contract is a strict `xfail` naming the cause.

**References (0002).** The tests no longer import B12X. On the CPU the ported oracles
equal B12X's exactly: the latent over random inputs, the Engram geometry for layers 1
and 14, and the hash over 20 random packed batches. The vocabulary test still imports
B12X's packed weight format, which serving uses as well.

**Calibration (0003).** The kernel must be at least four times closer to a reference
that follows its arithmetic than each injected defect:
- a dropped route;
- a wrong expert;
- no route weights;
- a skipped K tile;
- a BF16 combine.

The test prints the error table, the unquantized-down reference's error included.

## Window w1 ([w1.json](w1.json)): kernels only

- `bundles/r6k` on dgx3: the new tests against r6k's own kernels. These are expected
  to fail where the table above says.
- `bundles/fixed` on dgx4: the same tests with 0004-0006. Then
  [bench_nan_fixes.py](bundles/fixed/bench_nan_fixes.py) runs r6k's three modules
  (carried as `r6k_*.py`) beside the fixed ones. On finite inputs it requires r6k's
  bits, compared as integers, and at most 3% slower per call under CUDA graphs, warm
  and cold. The shapes are the decode capture sizes and prefill rows, plus the indexer
  over 64K and 1M positions.

## Results

### w1 (2026-10-14, 08:58-09:05 UTC, run by the TileLang session at the owner's request)

**r6k's kernels (dgx3):** 183 passed, 4 failed, 1 expected failure. The failures are
exactly the predicted gaps:
- `test_indexer_nonfinite_query_scale`;
- the shared SwiGLU's `test_nonfinite_like_the_reference`;
- the Engram gate's `test_nonfinite_like_the_reference[key]` and `[residual]`.

The expected failure is TileKernels' routed-expert clamp.

**With 0004-0006 (dgx4):** 187 passed and the same expected failure.
[bench_nan_fixes.py](bundles/fixed/bench_nan_fixes.py) found the fixed kernels
**bit-identical to r6k's on every finite input at every shape**. Its 3% time budget
flagged several points:

| kernel | flagged | reading |
| --- | --- | --- |
| shared SwiGLU | 2, 6, 12, 48, 72, 80, 8192 rows | Noise. Calls take 1.2-1.5 us, the differences are 0.03-0.14 us, and cold times sit on the timer's steps (4.06 / 5.92 us). 512 rows read 17% faster. 8192 rows: +4.8% warm, +0.8% cold. |
| Engram gate | 4, 6 rows warm; 32 rows cold | Noise. 2-17 us calls, 0.1-0.8 us differences, neighbouring sizes level or faster. |
| indexer score | 8K positions x 2048 rows: +4.6% warm, +3.9% cold | **Real.** TileLang warned that the `max2` loop over pairs became serial ("could not find a valid vectorization plan"). Decode sizes and 64K / 1M positions cost +0.1-0.7%. |

End to end, the indexer cost is roughly +0.17 ms per indexer call at an 8K prefill
chunk, about 0.25% of a chunk; decode is unchanged within noise.

**Verdict:** correct, bit-identical on finite inputs, and the contracts catch exactly
the bugs they target. Selected for the next release (owner, 2026-10-14: "if the work is
good, promote it to be selected for the next release"). One condition first: recover
the indexer's paired max so prefill is not slower. One option is a NaN select that
TileLang can vectorize, or applying the select after the vectorized `max2`.
