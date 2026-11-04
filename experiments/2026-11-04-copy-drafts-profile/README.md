# CopyDrafts profiling and implementation review

The 43-layer IKET runs did not enable CopyDrafts: no `copy_match`, `copy_alloc`
or `copy_select` launches appear in their captures. This experiment assesses
candidate-38's actual copy implementation on authorized idle sp8, with both
serving clusters left running. Base identities and exact file hashes are in
`base.json`. No runtime or build source is changed.

Separate screens: the pinned independent-reference/state/graph/toy-target test
suite; uninstrumented CUDA-graph timings; streaming IKET phase captures. Each
screen boots twice. History placement (the runner's UVA and device memory),
context length, batch size and input pattern are explicit workload variables.
No released build is measured again. Existing benchmark thresholds are historical
and are not treated as a current performance authority.

Measure the actual three-kernel sequence: match, pre-drafter allocation, select.
The bundled older benchmark measures match and select only. Timing excludes
compilation, has a cold GPU L2 before each sample and is separate from tracing.
Inputs include sparse random matches, a copied tail, period-64 text and constant
text. The last case deliberately checks both worst-case search work and the
short continuation chosen by the latest-match rule.

This qualifies kernels and controlled proposer behavior on one GPU. It does not
establish distributed serving throughput or real-model speculative acceptance.

Results pending.
