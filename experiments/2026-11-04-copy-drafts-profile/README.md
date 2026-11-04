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

## Result

Correctness is strong on the tested implementation; long-context matching and
short-copy selection need performance work. No copy-kernel source was changed.

- Both boots pass all 24 pinned CopyDrafts tests: independent references,
  cross-request isolation, CUDA graphs, reset/reuse, growth/shrinkage, shared
  budgeting, placeholder rejection and a toy target's committed-text invariance.
- Both streaming boots pass all 16 reference-checked workloads and per-warp
  pairing: 1,424,192 records each, covering `copy_match_kernel`,
  `copy_alloc_kernel` and `copy_select_kernel`.
- Both NVIDIA-native boots contain real decoded phase data for all three
  kernels: 32 launches per kernel, 70,176 warp lifetimes and 285,872 ranges per
  boot. All ranges have valid timestamps. The certified 40-slot capacity covers
  bounds of 8, 24 and 34 for match, allocation and select. Native captures use
  4K histories; streaming also covers 1M. Native timestamps are not used for the
  performance figures below.
- Both uninstrumented boots pass 96 timing/reference cases: 1/8/16 requests,
  4K/64K/256K/1M histories, UVA/device placement and four input patterns. Each
  component and the real three-kernel pipeline has 21 cold-L2 graph samples per
  boot. `results.json` retains both boot medians and their 95% t interval (df=1);
  two boots provide limited generalization.
- Two verification boots pass the existing paired 2-us regression gate for
  lookahead-16 versus block-5 on steps without copies, for greedy and stochastic
  block verification at 1/8/16 requests. This is a regression test, not proof of
  zero overhead (one interval's upper end is 2.06 us). Wider stochastic
  verification at one request costs about 88.8 us versus 40.4-41.7 us at six rows;
  these figures exclude target-model forward/head work.

Three-kernel pipeline latency, UVA, 1M tokens **per request**, milliseconds
(range of the two boot medians):

| Input | One request | 16 requests | Copied drafts offered per request |
| --- | ---: | ---: | ---: |
| Random, no match | 0.0816-0.0817 | 0.4689-0.4728 | 0, DSpark fallback |
| One copied tail | 0.0960-0.1043 | 0.4656-0.4779 | 8 |
| Period-64 history | 2.8454-2.8467 | 6.0618-6.0731 | 8 |
| Constant-token history | 2.8259-2.8310 | 6.8267-6.8270 | 1 |

At 4K, the one-request no-match pipeline is 10.0-10.2 us. Matching dominates the
long repetitive cases; allocation and selection individually take roughly
4-6 us. Device placement lowers the one-request, 1M repetitive pipeline to
2.034-2.040 ms but does not remove the underlying search cost.

## Implementation assessment

The implementation has useful properties: no inter-request matching, deterministic
max reductions without atomics, a fixed launch grid, graph replay, preallocated
state and async allocation buffers, and explicit reset on slot reuse. The
candidate's shared budget uses scheduled draft counts and the TP-wide confidence
snapshot rather than a rank's latest local proposal. The local regression tests
exercise that contract; this experiment does not validate an actual multi-rank
execution.

The measured weaknesses are:

1. `copy_match` scans the whole history again at every step. Frequent matching
   prefixes keep its dependent suffix comparisons alive up to 256 tokens. The
   large repetitive penalty is present in both boots and both memory placements;
   changing history placement alone does not solve it.
2. The longest-level/latest-position rule can choose an almost empty
   continuation. In a constant history it always picks `length - 1`, so the
   proposal is capped at one token regardless of the adaptive cap. Older equally
   long matches could supply a full proposal. DSpark has already run and is then
   replaced, so a short copy can displace a stronger proposal. No real-model
   acceptance or throughput regression is asserted from that constructed case.
3. The bundled copy benchmark omits `copy_alloc` and exercises a sparse copied
   tail, so it understates the current pipeline and misses the repetitive
   worst case. These measurements include all three kernels and those cases.

Priorities for the next measured candidates: choose useful continuations rather
than always the nearest end (or retain DSpark when its expected gain is higher),
then reduce repeated full scans with a bounded search or a per-request incremental
index. Keep exact-reference, slot-reuse, graph and output-invariance gates.
Actual distributed serving acceptance and throughput must decide the final
policy; these kernel timings alone cannot establish net generation speedup.

## Receipts and limits

`runs/hardware.json` retains all ten jobs, exact commands, source/image identities,
samples, logs and memory observations. `results.json` holds normalized timings,
native trace counts/checksums and verification results. Raw artifacts remain at
`/home/chris/spark-lab/iket-streaming-20261104/{copy-drafts-v1,copy-verify-v1}`.
`analyze.py` reproduces the timing intervals and native checks from them.
Ruff, compilation checks and `git diff --check` pass. `bin/spark doctor` passes;
the build check reports the promoted build tree is not prepared locally. No new
image was built. Sp8's reservation is released and the test container removed;
both serving leaders return HTTP 200. No serving cluster was restarted and no
candidate was promoted.
