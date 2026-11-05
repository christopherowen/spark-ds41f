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

All 589 patches through `0004` apply from the pinned upstream in a fresh source
tree, reproducing `701fd9619a4a2af710207f6469a58ede67ce37fc`. Doctor passes with
the existing ignored site maps. Build check reports missing prepared build
inputs; no new image has been built. Ruff on the new prompt change reports only
three inherited diagnostics (the test docstring line and two TileLang DSL nested
conditions); no new diagnostic is introduced.

## Further independent corrections

The column-first sampler still missed the strict gate at 96 padded FP32 rows
(mean +0.54% on its first boot). `0005` tests a 4,096-element tile for large FP32
matrices while retaining the 1,024-element tile for casts and small batches.
`0006` copies gather tails in naturally aligned words on a flat grid instead of
one byte per thread and one block per row. Each change affects its own kernel
family and retains the packed reduction/gather programs.

Prompt output alone removed concatenations but several expensive cases were
still statistically unresolved. `0007` tests rank calculation in the existing
selected-score pass: logits mode adds selected stores to the rank kernel;
log-probability mode counts exact integer ranks while finding the maximum. The
normal rank/log-softmax paths reuse their bodies with the extra work disabled;
their generated-program identity must be verified. New bitwise tests compare
against the independent kernels for FP16/BF16/FP32 and both ID widths, including
NaN, infinity and signed zero. No modified program is accepted without tests.

## Fused-path precheck and unchanged-path proof

The fused prompt path passes all 76 focused tests on both boots of source
`73e25e5d5b68f9504760cc63bc8f018402668e5c`. Both native-input identity boots
match all 120 per-case qualified CUDA programs, reference programs, input hashes
and strides exactly. Their output checks also pass. This transfers the existing
GPU qualification of normal score/selection paths; it does not substitute for
performance measurements of the new fused paths or for host/model qualification.
`0008` exercises both new prompt-score modes under startup cache freeze and
formats a host-side sampler binding; it changes no GPU program.

## Fourth combined screen

Both boots pass 1,167 correctness checks. Device measurements still reject the
4,096-element copy tile at 96 padded FP32 rows (+2.47% / +1.66%). The word-copy
tail path also remains unresolved. Both timing boots are retained; the queued
IKET/model jobs were yielded after the current atomic boot for an owner-authorized
native retention check. No timed job was interrupted.

`copy_screen.py` separately compares pitched CUDA DMA with a 128-thread,
two-element scalar TileLang copy; it does not change the runtime until a
backend qualifies. `0009` removes generic device integer division from tail
addressing using exact host-computed reciprocals and returns rank-one gathers as
views. Its 10,070 independent integer-division checks pass; GPU tests are pending.
All 594 patches reproduce source tree `d18eb2a61a5c02b4d7c8747f621a4438889db43e`.

Large evidence files are stored as deterministic gzip with original and stored
SHA-256 hashes in `runs/compressed-evidence.json`; every decompressed payload
was verified byte-for-byte before replacing its text representation.
