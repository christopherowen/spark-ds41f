# Complete the TileLang runtime replacements

Base deployment: `9815f36b72e6caacf8565a181d271e522c8bafc9`, combining
the five open replacement candidates (#885, #886, #887, #893 and #895).
The promoted configuration and published baseline are unchanged.
Exact source/image identities and file hashes are in `base.json` and
`candidate.json`. The ordered original experiment patches remain authoritative;
`vllm/series` adds the sampler correction after their combined source.

The first causal variable is vectorized aligned sampler copies. The previous
scalar TileLang copy was slower than the independent framework copy for padded
FP32 logits at 96 rows. The correction preserves the one-row grid, FP32 owned
output, all payload values, and the native contiguous FP32 DMA path. Narrow and
unaligned inputs keep the existing scalar TileLang copy. Added tests cover the
vectorized partial final block for all three input types.

Subsequent phases requalify the unchanged gather and rank-reduction candidates,
then explicitly test the interaction of all five replacements and inventory
runtime GPU launches. Each code correction is a separate causal variable.

## Reproduce

Publish the deployment commit, create a clean detached checkout on idle sp8,
read the dgx1 and sp8 holds, then run from that checkout:

```
python3 experiments/2026-11-05-tilelang-runtime-closure/run.py \
  --experiments experiments/2026-11-05-tilelang-runtime-closure \
  --run sampler-v1 --family sampler --operations tests pair
```

The runner reserves only the already authorized idle sp8; both serving clusters
remain running. It records exact commands, immutable image identity, two process
boots per operation, host memory and temperature. It uses the shared GPU lock,
20-second reservation heartbeats, a 32 GiB container limit and an 8 GiB active
host-memory guard. The independent reference is not a rerun of a released build.

Paired GPU measurements use fixed seeded inputs and identical output addresses.
Each replay contains eight cold-cache samples timed by CUDA event nodes inside
the graph. Roles are shuffled and mirrored in each of 120 rounds. Raw samples,
the previous candidate, unchanged-reference controls, generated CUDA hashes,
and framework CUDA traces are retained. Acceptance has no slowdown allowance:
each boot must show a five-standard-error upper ratio at or below one, or prove
the same GPU program and bindings. Inconclusive measurements remain unresolved.
This gate covers device execution; host latency and complete model/fabric
qualification are separate requirements. Results and failures will be appended.

## First screen

Vectorization alone passed 88 tests on both boots but failed the performance gate.
The 96-row padded FP32 copy remained 1.65% / 1.76% slower; the 96-row dense
FP16 cast was also unresolved. All raw samples and CUDA traces are retained in
`runs/sampler-v1/`. The second variable changes only block traversal order for
aligned copies so consecutive blocks visit adjacent columns. It keeps the scalar
path above CUDA grid-y's 65,535-row limit and adds both boundary cases.

`run_full.py` runs the combined stack twice each at TP4/TP3-shaped layers on
sp8 with Nsight graph-node tracing. `inventory.py` assigns every kernel to exact
generated TileLang symbols, ATen, NCCL or an explicit unknown bucket; it separates
measured requests and grid replay from startup. This is a single-GPU dummy-weight
check without Engram or physical peers, so it cannot establish multi-node
transport coverage or real-weight model quality.

## Combined correction

The first combined suite passed 1,076 tests and failed one obsolete test that
expected the removed partial-pack ATen path. `0003` replaces that expectation
with exact opaque-bit validation; `runs/traversal-v2/` retains the failure.

`0004` removes the prompt-score chunks' three GPU concatenations. Existing
TileLang selection, score and rank kernels write into final row slices. With
zero requested alternatives the IDs remain a view of the native sampled IDs.
The existing four score-mode test now covers lengths 1, 1,024, 1,027 and 2,049
with strided int32/int64 IDs. The prompt benchmark compares unmodified GPU
algorithms while assigning only allocation/concatenation output destinations to
shared buffers; no compute operation is elided from the reference. At zero
alternatives the candidate ID result aliases the input and needs no GPU copy.

All 586 patches through `0004` apply from the pinned upstream in a fresh source
tree, reproducing `701fd9619a4a2af710207f6469a58ede67ce37fc`. Doctor passes with
the existing ignored site maps. Build check reports missing prepared build
inputs; no new image has been built. Ruff on the new prompt change reports only
three inherited diagnostics (the test docstring line and two TileLang DSL nested
conditions); no new diagnostic is introduced.
