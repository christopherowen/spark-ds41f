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

Status: diagnostic implementation; no pruning result, kernel speedup or serving
qualification established. No runtime or upstream source changes in this step.
