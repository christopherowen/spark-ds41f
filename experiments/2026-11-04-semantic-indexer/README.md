# Semantic indexer search

The owner has revised future acceptance to generated-token equivalence and moved
correctness/proof work outside the timed path; see
[`../2026-11-04-clustered-indexer/acceptance.md`](../2026-11-04-clustered-indexer/acceptance.md).
The exact-selection bounds below remain historical research, not requirements
for the next performance candidate.

Result: **retain as research; no faster runtime qualified**. The completed
[two-process screen](results.md) preserves native selections on all twelve
captured operands, including source block IDs/counts and subsequent reindex.
With a practical native seed cutoff and seed reuse, semantic shape bounds avoid
41.74–84.61% of key evaluations on the nine dense snapshots, but only0.15% on the
three source snapshots. Those source queries repeat across ranks. This is work
accounting from an untimed oracle, not a TileLang runtime or request-speed result.
Full shapes have substantially more metadata/query work than native packed keys.
The cheap compact maps tested here do not provide useful pruning. See
[the decision](decision.md) before extending this experiment.

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

sp8 yielded before the follow-up batch to a pending native-input qualification.
`runs/sp8-deferred-01` retains that refusal. DGX4 overlap was also unavailable
under the other window's frozen measurement protocol. The owner-authorized idle
sp5–sp7 group can run these same pinned bundles through `run_window.py`: clean
published sources, thirty-second idle gate, leases on all three nodes, reviewed
named-profile synchronization/stop, bounded16GiB kernels on sp5, and one C37
restoration with independent watchdog/live verification. No candidate serving
engine starts. Raw snapshot transfer must be hash-verified and remains node-local.
The deployed experiment checkout advances coherently; original project checkouts
stay untouched. This changes deployment metadata when C37 is restored, so report
the final checkout commit explicitly.

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

The sp5–sp7 window completed all seven jobs and restored C37 on every node at
18:38:25UTC (19:38:25CET). Independent live checks confirm the original C37 image,
no search mounts/environment, unchanged clean original806c5797 checkouts, clean
configured experiment checkouts at3b8397f0, and no reservations. Deployment
metadata advanced coherently to that commit; image/model/settings did not change.
Minimum available host memory114732613632 bytes, maximum thermal59.7C; no GPU
job, memory guard or lease failed. Exact commands and all observations are in
`runs/semantic-window-01`. Thirty local CPU math cases and two poor-proposal
witness cases also pass. Native hardware gates remain separate from CPU checks.

`query-map-cpu-01` retains the initial rank-only holdout, including its misleading
source result. The source query is byte-identical to its fitting inputs and is
excluded from acceptance. `query-map-cpu-02` explicitly refuses that overlap;
its distinct dense queries still retain every key through64 dimensions. Both
CPU analyses ran read-only, with two CPU threads, inside our isolated correctness
container; neither launched CUDA work nor wrote source files. These are idea
screens, not independently sampled model-quality or GPU gates.
