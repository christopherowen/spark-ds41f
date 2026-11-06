# Decision

Retain the focused source change as a draft for serving qualification. Two fresh
Spaceport boots establish a component gain with exact integer metadata, and the
selected placement is faster than the eager, direct-kernel and separate-graph
alternatives tested. The static-consumer lower bound leaves approximately
1.55 us of residual refresh cost at 8 requests/8K; further fusion has limited
headroom in this fixture.

The isolated gain is mostly hidden by preceding GPU work: with about 775 us
queued first, the measured saving is 2.940 us (95% CI 2.578–3.308) on boot 1 and
2.663 us (2.418–2.905) on boot 2. This is 0.376% and 0.340% of that synthetic
sequence. Neither the isolated 97.6–97.7% component reduction nor the overlap
percentage is a serving-throughput prediction. See [micro-results.md](micro-results.md).

Broader GPU regression suites (388 passed per boot across DFlash inputs,
causality, DCP, CED metadata, CUDA graph manager, DSpark context, and DSpark
Markov/sampling) and CUDA graph replay traces (`runs/gates-20261106/`) also pass
across two fresh Spaceport boots, confirming `metadata_builder_kernel` executes
inside the FULL query graph with zero eager rebuilds, zero incremental compiler
specializations, and zero incremental CUDA graph-pool bytes.

Inspection of the full DFlash/DSpark draft path (`DSparkSpeculator._generate_draft`,
`_sample_sequential`, and `DFlashSpeculator.propose`) confirms why the 213–220 us
host metadata saving reduces to 2.66–2.94 us when preceded by 775 us of queued
GPU work and why no additional per-step host loop remains to fold into the
query graph: both the parallel backbone forward across all `N` draft query rows
and the sequential `N`-step Markov sampling loop (`_sample_sequential`) are
already captured inside the single FULL query CUDA graph. When 775 us of GPU
execution is already queued on the stream, the 215 us of asynchronous CPU
enqueue work in `eager_full` finishes ~557 us before the GPU stream drains,
hiding the CPU host overhead and leaving only the 2.66–2.94 us GPU-side kernel
launch delta (`separate_graph` vs `query_graph`). The remaining pre-graph
operations in `DFlashSpeculator.propose` (`prepare_dflash_inputs`,
`_context_preparer.run`, and `tilelang_inputs.padding_rows`) cannot be folded
into the query graph because `prepare_dflash_inputs` feeds the separate
`num_target_tokens`-keyed context KV graph and `padding_rows` depends on the
runtime unpadded `live` token count when `num_reqs` is padded up to a captured
graph descriptor.

Full pinned build preparation and `build check` now pass. No serving run,
source-complete image build, target sampling/rank validation or promotion has
occurred. Keep launch disabled and both PRs draft. Full serving
qualification remains necessary before claiming a user-visible performance gain.
The owner explicitly allowed the isolated Spaceport screen; it respected the
local GPU/CPU reservations, yielded between boots, and released its hold. The
serving cluster was not changed.

## Acceptance and claim boundaries

The owner accepts a demonstrated correctness repair with preserved performance.
No such repair is claimed here: the old eager path passes the metadata contract,
and the new tests establish equivalence of placement rather than a before/after
correctness failure. The current classification is:

- Correctness improvement: not demonstrated. Integer equivalence: passed.
- Component performance: repeatable measured speedup, including the much smaller
  exposed gain in the controlled overlap fixture.
- Serving latency/throughput/memory preservation: unmeasured, not neutral.
- Regression: none observed in the exercised component checks; broader behavior
  remains unresolved. A zero-spanning interval alone cannot establish no loss.

The selected graph placement remains the strongest of the measured valid
alternatives. The residual floor does not justify a new kernel or broader
execution rewrite on its own. Keep optional tuning separate if an independently
evidenced correctness repair is subsequently found.
