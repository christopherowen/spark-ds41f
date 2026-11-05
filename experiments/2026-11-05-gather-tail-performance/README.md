# Gather tail performance

Base: exact candidate-38 source in `base.json`. Apply the original gather series
then this narrowly scoped correction, in `vllm/series`. One intended variable:
local tail layout copy; transport policy and bytes remain unchanged by this
correction. The original gather patch's dim-zero transport request still needs
physical-fabric qualification.

Gather tails copy opaque aligned words instead of one byte per thread. Tiny
tails map a warp to each rank/row, requiring no device division. Larger tails
use exact host-computed reciprocals with one correction. Rank-one concatenations
return views. The packed path and main fused gather kernels remain unchanged.
Explicit wrapper bounds and disjoint-output validation protect opaque payloads.

All 24 main fused-gather cases and all four tail cases pass their strict GPU
timing gates on both boots. The tail check retains the original event ordering
markers and times the actual copy kernel through CUPTI hardware activity; the
earlier unresolved event intervals and failed parser variants remain recorded.
Tail mean improvements range from about 10% to 50%, with every five-standard-
error upper ratio below one. See `tail-results.json`.

The previous focused gather boots pass 147 checks each. Both final composed
boots pass all 1,446 correctness tests. Four single-GPU, 43-layer dummy traces
contain only TileLang compute in the request window, with unchanged token IDs
and logprob bits. Physical fabric qualification remains a separate release gate.

`source-equivalence.json` proves the non-collectives files match the composed
source byte-for-byte; all gather functions and constants in the shared module
match by exact AST. Other rank-order functions differ intentionally in this
independent patch. Raw evidence and all unsuccessful runs are retained under
`experiments/2026-11-05-tilelang-runtime-closure` on the published integration
branch. The driver verifies frozen image/source hashes and records exact commands,
inputs, generated CUDA, profiler traces and memory observations on both boots.
No serving cluster operation is performed. Build-image, serving and real-fabric
qualification remain required before release.
