# Semantic indexer search

Base: `310f1408c1ace65c462d367ce6cb1fc5640d0c5e`.

This experiment tests the interaction between individual-key semantic grouping
and signed, coupled-head score bounds. It leaves the model, native scores,
selection rules and serving configuration unchanged. Query-independent balanced
cosine partitions replace position groups. Affine upper bounds combine positive
ReLU secants and optimized negative-head supporting planes before taking a
support bound. The diagnostic compares coordinate boxes, low-rank shapes and an
optimistic exact member support. Exact member intervals/support are an oracle
screen and cannot establish a fast runtime implementation.

The input is the existing twelve native C36 snapshots on sp8. Initial screening
uses the four rank0 layer/interface samples; raw tensors remain node-local.
Every native full score is an untimed correctness oracle. Bound dominance is
checked against all native scores. Ties survive. Retention is reported both per
key and after expansion to native eight-key blocks, including the forced final
source block. An ideal global cutoff is deliberately optimistic: it screens
geometry before paying for a practical seed or persistent metadata.

`sp8_run.py sp8-screen:semantic-screen-01` runs a published clean checkout using
the existing exclusive sp8 lease, memory/thermal guards and pinned C36 image.
Repeat in a fresh container as `semantic-screen-02`. Exact commands, errors,
memory observations and all results will be retained under `runs/`.

Follow-up bundles separate the next questions:

- `sp8-ranks`: ranks1/2/4/7 and an explicit residual-complement bound, in
  position/semantic groups8/64. This tests the rank/residual-bound interaction;
  it does not silently replace the first screen's `shape` geometry.
- `sp8-projection`: a query-independent centered PCA map with8/16/32/64/96/128
  dimensions. Explicit reconstruction residuals certify dot intervals, followed
  by native BF16/FP32 rounding stages. Projected coefficients are F64 in this
  diagnostic; the nominal BF16 bytes/key are an optimistic storage budget, not
  evidence that a BF16 map has been qualified.
- `sp8-selection`: all twelve snapshots, groups8/shape7, both groupings. Exact
  native top512 selection, source2048 block IDs/counts, and subsequent reindex
  must match after masking certified nonwinners.

The follow-ups also report practical seed accounting. Dense queries propose1024
keys; source queries propose2048 ordinary eight-key blocks plus the forced final
block. Native oracle scores restricted to those seeds establish the cutoff.
All uncertified keys survive, and already scored seeds are reused. Every mask
passed to the native selector includes those seeds. This accounting excludes
duplicate scoring but is still an untimed diagnostic, with neither metadata
cost nor a production kernel implementation hidden in its claimed results.

The first two fresh containers (`semantic-screen-01/02`, commit663f5e7f) agree
exactly. Shape bounds with semantic groups8 retain15402/11385/8836 of33656 keys
on the three dense samples and33571 of36576 on the source sample. Expanding to
native blocks retains32096/27880/23704/36576 positions respectively. All native
score maxima are bounded and threshold ties survive. Position groups8 retain
16240/11296/8128/35008 positions. Boxes and practical shape groups64 remain
ineffective. These are ideal-cutoff, untimed screens; full-shape factor storage
and query work can exceed the native key scan. No speedup follows from retention
alone. Minimum host memory available111018262528 bytes; maximum thermal62.4C;
no lease or memory guard failures. Initial receipt collection used unsupported
remote brace expansion and copied nothing; corrected wildcard collection fetched
all four files per run. No GPU run was lost or repeated for this client error.

Status: useful pruning observed in a diagnostic, but no kernel speedup or serving
qualification established. No runtime or upstream source changes in this step.
