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

All 24 main fused-gather cases already pass both strict GPU timing gates, and
the previous variant passes 147 gather tests per boot. The final combined source passes all 1,266 correctness tests on both boots.
The two tiny-tail event-based timing gates remain unresolved; their measured
interval contains about 3 us of event-node overhead around an approximately
1 us kernel. Paired hardware-activity measurements are pending and must pass
the same strict zero-margin bound. The previous inconclusive results remain. Historical unresolved
variants remain in the combined experiment. Nothing is promoted by this folder.

`source-equivalence.json` proves the non-collectives files match the composed
source byte-for-byte; all gather functions and constants in the shared module
match by exact AST. Other rank-order functions differ intentionally in this
independent patch. Raw evidence and all unsuccessful runs are retained under
`experiments/2026-11-05-tilelang-runtime-closure` on the published integration
branch. The driver verifies frozen image/source hashes and records exact commands,
inputs, generated CUDA, profiler traces and memory observations on both boots.
No serving cluster operation is performed. Build-image, serving and real-fabric
qualification remain required before release.
