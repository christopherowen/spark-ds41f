# Single-rank DSpark fusion

Base: exact candidate-38 source recorded in `base.json`. One intended variable:
route the single-rank full vocabulary through the existing fused shard drafting
path. Its pair gather is a view. No promoted recipe, baseline or serving process
changes. Apply the one patch after candidate-38's ordered source series.

TP1 previously gathered the anchor, launched five `nn.Embedding` lookups and
stacked confidence embeddings. The fused path selects and embeds each next token
in its existing selection kernel, with final confidence layout. TP1 shares the
multi-rank supported TileLang head requirement; incompatible heads fail at
warmup rather than selecting this path silently.

Both boots pass all 53 focused checks, including full-vocabulary exact
fused-versus-materialized tests. The 43-layer dummy-weight trace removes 42 ATen
request launches and preserves the recorded token IDs and logprob bits. Its
remaining 24 embedding-normalization launches are addressed separately. The
trace is diagnostic, not a model-throughput benchmark. Paired six-case complete
Markov/confidence timing and the final model inventories remain pending.

`source-equivalence.json` proves all three patched files match the tested
combined source byte-for-byte. Raw commands, source hashes, generated programs,
traces, errors and memory observations are retained in the November 5
`tilelang-runtime-closure` experiment on its published integration branch.
Fresh independent patch application and local checks are recorded separately.
A built image, real weights and physical networking still require qualification.
