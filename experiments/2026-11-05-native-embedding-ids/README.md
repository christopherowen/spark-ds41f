# Native embedding IDs

Base: exact candidate-38 source in `base.json`. One intended variable: combine
TP1 negative-ID normalization and native-width lookup in TileLang. Apply the one
patch after candidate-38's ordered source series. No promoted configuration,
immutable baseline or serving process changes.

The TP1 wrapper previously clamped IDs and cast them to int64 before lookup.
TileLang methods now read native int32/int64 flat strides and clamp negative
multimodal IDs to row zero in the lookup. Plain TP1 tables also use TileLang.
Other quantization methods retain their existing preprocessing. Separate
non-clamping parallel specializations preserve their existing code.

Both boots pass all 127 embedding checks, including 36 new public eager/graph
cases over plain/packed tables, both ID widths, empty and shifted/strided/matrix
IDs. All four inspected parallel CUDA sources are byte-identical to the old
versions on both boots. Of 32 public-route timing cases, three have lower means
but insufficient precision on one boot each. A single fixed 2,048-round extension
is predeclared; primary, extension and pooled zero-margin criteria must pass on
both boots. This candidate remains in draft pending those results and the final
full-model launch inventory. Historical rejected-stride failures are retained.

`source-equivalence.json` proves all three files match the tested combined
source. Full raw paired samples, CUDA sources, profiler inventories, commands,
input/weight hashes, failures and memory observations are retained in the
published November 5 `tilelang-runtime-closure` experiment. Fresh independent
application and local checks are recorded separately. Built-image, real-weight
and physical-network qualification remain release gates.
