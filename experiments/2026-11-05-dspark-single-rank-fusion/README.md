# Single-rank DSpark fusion

Base: exact candidate-38 source in `base.json`. One intended variable: route
single-rank full-vocabulary drafting through the existing fused shard path.
Apply the independent one-patch series after candidate-38. No promoted recipe,
immutable baseline or serving process changes.

TP1 previously gathered the anchor, ran five separate embedding lookups and
stacked confidence embeddings. The fused path selects and embeds each next
token in its existing selection kernel and writes the final confidence layout.
One rank owns the full vocabulary; its pair gather is a view. It shares the
multi-rank supported TileLang head requirement and fails at warmup for an
incompatible head.

Both boots pass all 53 focused checks, including exact full-vocabulary
fused-versus-materialized comparisons. All six complete five-position Markov
and confidence timing cases pass on both boots, at 1/8/16 requests and both
score widths. The one-request BF16 case passes its single predeclared 2,048-round
extension and pooled bound, with lower primary means on both boots. No slowdown
tolerance is used. Shared weights, logits, hidden states and anchor hashes match.
See `checks/performance.json`.

Both composed boots pass all 1,446 tests and all 33 eager/graph IKET numerical
cases, covering 14 required kernel families. This fusion removes 42 ATen request
launches. With the separate native embedding-ID change, four 43-layer dummy
traces each contain only 7,662 TileLang request launches, with unchanged tokens
and logprob bits. These are single-GPU TP3/TP4-shaped tests with Engram disabled.

`source-equivalence.json` proves all three source files match the tested composed
source byte-for-byte. Fresh independent application and local checks remain
recorded in `checks/local.json`. Raw measurements, generated programs, traces,
commands, errors and memory observations are retained on sp8 and the published
`feat/tilelang-runtime-evidence` branch. A built image, real weights, untraced
production features and physical networking remain separate release gates.
