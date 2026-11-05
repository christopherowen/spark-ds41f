# Native operand batches for selector fitting

Base `65c6e62f`, immutable C36 image and source pin
`6d4e49d1cdbb6513819e47145e0e52fdab327b6b`. This is separate from the
completed-answer benchmark. One intended variable for this preparation is
collecting 64 native query rows per layer instead of a single row. Native model
weights, scoring and attention remain unchanged; the live-list scratch guard is
retained. No selector map is trained or qualified yet.

The diagnostic hook requires an explicit case/prompt hash in `arm.json`, skips
startup and graph capture, and records three dense layers plus one source layer
per rank and case. It waits for 64 queries with at least 32,769 visible keys,
retains each query's table/length and copies only written native key and scale
prefixes. Distinct query tables are supported. A 512 MiB host-side union limit
refuses excessive captures. Raw inputs and future fitted weights remain
node-local, outside Git.

Use separate seeded public source prompts for fitting and validation. No
LongBench pilot or held-out questions may supply training activations. Separate
documents, not adjacent query rows, define validation independence. The first
mechanical checks must cover both native page sizes, multiple query tables,
partial/aliased pages, skipped startup, explicit caller identity and overwrite
refusal. Full native replay must validate every captured row before fitting.

This profile is untimed: copies and synchronization perturb execution. Never
mount this capture hook in a timed LSA or native control profile. No capture run,
performance result or quality improvement is claimed. A future collection must
use the authorized, reserved all-node window and restore C37 after its jobs.

Preparation applies the scratch guard and this experiment's ordered patch in a
fresh named C36 tree; all three overlays match byte-for-byte and `diff --check`
passes (`runs/local/source-check.json`). An initial attempt used a repository
without the C36 object; that refusal and the correction are retained. The
CPU-only pinned-container copy checks pass on sp8 for 128/256-key pages, 64 rows,
different query tables, physical aliases, partial pages, zeroed saved tails,
arming/graph/decode skips and overwrite refusal (`runs/copy-check-01`). The
container and reservation are removed. These checks do not replace native GPU
replay, which is still required before any fit.

All-row synthetic native replay then passes on sp8 (`runs/replay-check-02`),
including exact visible BF16 score bits, selected keys and source handoff/counts
for both page sizes and 64 rows. A separate process with the CUDA allocator cache
disabled and strict compute-sanitizer initcheck reports zero errors
(`runs/initcheck-01`). The first replay launch used the wrong working directory
and did not start a GPU container; that error is retained in `runs/replay-check-01`.
Every standalone reservation is released and no containers remain.

`collect.py` prepares two fitting documents and one validation document from
entire, SHA-deduplicated Python standard-library files assigned to disjoint
cohorts before capture. It uses no benchmark questions. The original file hashes,
actual tokenizer counts, exact commit and every response/capture identity are
recorded; prompts, responses and tensors stay private on the nodes. Replicated
ranks are mechanical checks, not independent training examples: use rank 0 for a
fit and keep document-level validation separate. Collection is not yet run.

Future collection command, from the clean published sp5 controller checkout
after the current answer-quality window restores C37 and releases all nodes:

```sh
python3 experiments/2026-11-05-selector-fitting/collect.py \
  --out /home/chris/spark-lab/results/selector-fitting/capture-01
```

The runner performs all-node preflight, a quiet-service guard, exclusive native
reservations, heartbeat/watchdog, published commit sync and named coherent
capture startup. Each explicitly armed document has one untimed generation
request. It disarms every node and restores C37 once before releasing holds.
Every actual captured batch still needs exact native replay before fitting.

Before collection, source captures are explicitly delayed to 65,537 visible
keys; dense captures retain 32,769. Both require 64 eligible rows. This gives the
source validation a domain larger than its 32,768-key aggressive shortlist.
The thresholds are in each arm record and captured metadata, and every request
must cover both domains. Mechanical fixtures retain their smaller test domains;
the collector's full requests remain untimed. The revised series again applies
to a fresh named C36 base with the scratch guard and identical overlays.

The predeclared fitting probe is a 32-dimensional query-weighted clustering
metric. `fit_metric.py` forms `M = mean(w_h² q_h q_hᵀ)` from native query heads in
`fit-a`/`fit-b`, then keeps its leading 32 eigen-directions weighted by square-root
eigenvalue. It reports retained query-direction energy on the untouched
`validation-a` document. This is a clustering proxy, not an exclusion bound or a
quality guarantee. Native BF16 score rounding and final answers still need their
own checks. The planned use projects block features only during summary updates;
it introduces no reference or grading work into queries. Actual key-retention
validation and construction timing must precede any serving candidate. No maps
have been fitted and this probe is not integrated into serving.

The revised capture-threshold checks and metric arithmetic contracts pass in the
CPU-only pinned container (`runs/copy-check-03`): all MXFP4 codes/scales, the
leading 32 directions, and invariance to query-weight sign for the squared
metric. The container and reservation are removed. This is a synthetic contract
check; no model-derived metric has been fitted.

`metric/` prepares the construction-only implementation of the fixed 32D probe.
It projects block means for grouping, then writes the original 128D centroids in
the same 73-byte layout. Query scoring and fine scoring are unchanged. The map is
not trained, loaded or used by any current serving profile. Its separate GPU
fixture must check map-sensitive partitioning, exact original-space centroids,
untouched native records, partial-write no-ops, graph replay and page recycling.
No component speed or model-quality result is claimed.

The first synthetic projected-summary GPU gate passes on sp8
(`metric/runs/metric-summary-check-01`). Switching the 32D map changes grouping
as the independent partition oracle predicts; original-space centroids are
bit-exact for the fixture, native bytes are untouched, partial writes are no-ops,
and graph replay/page recycling pass at both page sizes. No trained map or model
quality is tested. A separate strict initialization process is prepared next.
