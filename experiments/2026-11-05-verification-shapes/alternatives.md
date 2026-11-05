# Capture-design alternatives and remaining bound

This audit challenges the first implementation; it does not select a performance
winner. `compare_alternatives.py` executes the actual source dispatch and context
shape functions with controlled capture policies. `runs/alternatives-final.json`
records byte hashes and verifies every extracted source file against its pinned
commit. The base checkout also contains an unrelated, unused GEMM prototype;
the audit explicitly checks its graph files are byte-identical to candidate-40.
Earlier seven-/eight-policy receipts are retained as well.

The census covers 369 feasible request/row shapes at TP3's 59-row copy envelope
and 1,307 at TP4's 107-row envelope. Each shape has equal weight. These totals
are structural costs, not serving-frequency estimates or timings.

| Policy | TP3 target/context graphs | TP3 summed padded target rows/requests | TP4 target/context graphs | TP4 summed padded target rows/requests |
| --- | ---: | ---: | ---: | ---: |
| Candidate-40 | 65 / 39 | 718 / 753 | 74 / 48 | 3,604 / 7,170 |
| Proposed native exact shapes | 70 / 41 | 578 / 651 | 87 / 56 | 2,816 / 5,534 |
| Native rows in the ordinary ladder | 69 / 41 | 502 / 753 | 84 / 56 | 2,132 / 7,238 |
| Native ordinary rows plus exact shapes | 74 / 41 | 502 / 651 | 97 / 56 | 2,132 / 5,558 |
| Exact dense shapes through c3 | 116 / 53 | 517 / 476 | 133 / 68 | 2,755 / 5,215 |
| Exact dense shapes through c4 | 170 / 59 | 424 / 292 | 195 / 79 | 2,663 / 4,775 |
| Every row in the ordinary ladder | 113 / 59 | 0 / 720 | 172 / 107 | 0 / 6,786 |
| Every feasible request/row pair | 369 / 59 | 0 / 0 | 1,307 / 107 | 0 / 0 |

The native-row-only ladder is a credible competitor: it uses fewer target graphs
and removes more row padding than the proposal, but loses the proposal's exact
request counts for native full-depth batches. The union is another useful
comparison; at TP4 its nearer row descriptor can slightly increase request
padding because dispatch prioritizes rows. This is a real tradeoff, not grounds
to assume one descriptor set dominates all costs.

Increasing dense coverage from two to three or four requests adds many graph
instances. Every-row ladders remove all row padding with fewer instances than
every request/row pair, but still pad requests. The graph counts expose the
tradeoff; they do not establish GPU memory consumption. Actual shared-pool
allocation, capture resources and preparation high-water marks must be measured.

## Mechanism-based bound

Using the exact `decode_tile_rows` function and `WIDE_DECODE` entries, compare
candidate-40 and the proposal for `(N,K)` = `(448,5120)`, `(608,5120)`,
`(5120,576)`, and `(5120,768)`. For every enumerated shape at both capacities,
the rounded number of matrix-multiply tile rows is identical: **zero saved MMA
row tiles** in all eight comparisons. The source's 16/32/64 and 96/128 routing
boundaries are already represented by the capture ladder.

Consequently this graph fix cannot claim faster bulk matrix multiplication in
those four projections. Its possible benefit lies in row-sensitive preparation,
loads/stores, context communication, request metadata and other operations that
still process graph padding. It does not reduce the number of graph replays or
provide a speed bound for the whole model. Measure those residual components and
the complete affected path before treating smaller shapes as a speed improvement.
The context input kernel already scrubs rejected/padded rows on every replay;
additional graph instances must not weaken that contract.

## Choice and unresolved work

The current source PR is a conservative restoration of the existing native
full-depth contract, with warmup and context capture kept consistent. CPU tests
support its coverage, not its superiority on hardware. It is intentionally still
a draft and is not the demonstrated best implementation.

The next hardware comparison should screen the proposal, native-row-only, and
their union first, followed by the every-row ladder if capture memory permits.
Dense c3/c4 and every-pair capture expose upper coverage bounds; screen them only
if measured row/request-sensitive cost justifies their extra graph instances.
Use the same recorded workload identities and distinct-agent traffic. Collect a
joint histogram of request count, live verification rows, target padded rows and
context padded rows; the unweighted census cannot rank real traffic performance.
At c1 and TP3 c8 full depth, the proposal already has the same row counts as the
base, so the repaired intermediate request counts and heterogeneity matter.

Each promising runtime candidate needs the exact image gate, native arithmetic,
batch invariance, initialization/memory checks, and a candidate boot plus repeat
boot with confidence intervals. Compare to saved release/candidate references;
do not remeasure released builds. A new, explicitly identified unreleased
parent control is permitted when the saved candidate-40 runs cannot isolate the
causal change. Preserve all losing alternatives. If no end-to-end gain
survives or the graph memory cost is excessive, leave this proposal unadopted.

All three first-screen policies now have actual source implementations, pinned
profiles, and fresh source replays: 576 patches for the exact-native proposal
and 578 for each variant. Each final implementation passes 70 focused CPU
tests; 18 unrelated distributed-embedding tests are explicitly deselected.
`runs/implemented-alternative-census.json` verifies that their real dispatch
functions match the policy census and retain the same piecewise graph counts
(16 at TP3, 22 at TP4). Non-copy coverage is unchanged. The variants and all
setup failures are retained under `variants/` and `runs/`.

None has GPU timing or memory evidence yet. The owner authorized the coordinated
sync, stop/start/restart, candidate replacement and restoration needed for
hardware testing. The shared window is still held by the frontend qualification
job and the norm-kernel request is ahead of this task; authorization does not
override that queue. A finite search cannot prove a global
optimum; the current evidence also does not yet prove the strongest feasible
hardware implementation.
