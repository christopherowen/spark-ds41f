# Native embedding IDs

Base: exact candidate-38 source in `base.json`. One intended variable: combine
TP1 negative-ID normalization and native-width lookup in TileLang. Apply the one
patch after candidate-38's ordered source series. No promoted configuration,
immutable baseline or serving process changes.

TileLang methods read native int32/int64 flat strides and clamp negative IDs to
row zero during lookup. Plain TP1 tables also use TileLang. Other quantization
methods retain their preprocessing; separate parallel specializations preserve
their existing generated code.

Both boots pass all 127 embedding checks, including 36 new public eager/graph
cases over plain/packed tables, both ID widths and empty/shifted/strided/matrix
IDs. All 32 public-route timing cases pass on both boots. Three uncertain cases
use the single predeclared 2,048-round extension: each primary mean is lower, and
each extension and pooled five-standard-error upper bound is at most one.
No slowdown tolerance is used. Four inspected parallel CUDA programs remain
byte-identical on both boots. See `checks/performance.json`.

Four 43-layer dummy-weight traces each contain 7,662 TileLang request launches,
zero ATen launches and no unknown kernels, with unchanged recorded token IDs and
logprob bits. These are single-GPU TP3/TP4-shaped tests with Engram disabled;
startup, benchmark fixtures and physical networking are separate. See
`checks/full-model.json`.

`source-equivalence.json` proves all three files match the tested combined source.
Fresh independent application and local checks are recorded in `checks/local.json`.
Raw samples, generated CUDA, inventories, commands, errors and memory observations
are retained on sp8 and the published `feat/tilelang-runtime-evidence` branch.
The earlier rejected-stride failures remain in the evidence. Built-image,
real-weight quality/throughput and physical-network checks remain release gates.
