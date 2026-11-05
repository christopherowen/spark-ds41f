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

## Selected copy backend and precision plan

The independent backend screen passes eager/graph payload checks on two boots.
At 96 padded FP32 rows pitched DMA is 2.19% / 2.04% faster; 32 rows also pass
the strict gate. The 128-thread scalar alternative remains unresolved. Both
backends are slower on the deliberately odd-width probe, which stays TileLang.
`0010` selects pitched DMA only for wide (at least 65,536 columns), 64-byte-aligned
FP32 matrices of at least 8 MiB, with 64-byte row pitches. Smaller/cast/general
copies retain the 1,024-element TileLang path. The rejected 4,096-element path
and its unused warmup variant are removed. Live-data graph replay and side-stream
tests cover the actual public route; its final qualification remains pending.

Tail timing now uses 64 cold samples per replay for both reference and candidate,
with the previous word-copy kernel measured in the same protocol. This increases
precision without changing the zero-margin gate. Two prompt-logit cases whose
primary means are below one receive one predeclared 480-round extension; both
the extension and pooled five-standard-error bounds must pass. It is not a new
slowdown allowance. Filters use the full family traversal to preserve RNG inputs.

Controllers now honor queued reservations between atomic jobs and can resume a
specific boot with `--boots`. They release sp8 and exit 75 when another check is
queued; they never interrupt the current timed job.

## Final focused screens

The actual public sampler passes all 18 timing cases on both boots and 120
correctness tests per boot. Gather passes 147 tests per boot. Its two 129-row
tails pass; the two 7-row, 105-word tails remain unresolved. `0011` tests a
single 128-thread block for only tails of at most 128 opaque words; wrapper
validation and the explicit output bound make redundant source checks unnecessary.
Larger tails retain their previously measured program. All 596 patches reproduce
source tree `ae0dea96608c81e28ca41f1ebce5a10c4ffdf1a7` in the fresh prepared tree.

`wire_probe.py` inventories the real nonblocking CPU result conversion for both
ID widths, dense and strided IDs, zero/five alternatives and both score modes.
This checks whether an apparently copy-free GPU result adds a packing kernel
when it leaves the GPU. Its result is pending; no assumption about that path is
used to claim complete coverage.

## Wire and tiny-tail follow-up

Both wire boots reproduce the ATen pack for strided, zero-alternative IDs in
both ID widths and score modes. `0012` transfers the bounded physical column
span through PyTorch's existing asynchronous DMA path, then preserves the view
on CPU. No raw host pointer, untracked allocator lifetime or device packing
buffer is introduced. The extra transferred holes are explicit in the complete
host-call/stream-wait benchmark; its performance must qualify independently.
Dense IDs and multi-column results retain their ordinary transfer.

The one-block tiny-tail variant remains unresolved. `0013` tests a warp per
rank/row, eliminating all division in those at-most-128-word tails. Larger tails
retain their already measured program. All 598 patches apply in the fresh
prepared tree and reproduce `446dd388e525ae96c0c90c33350e5802b29ddd7e`.
Both full prompt confidence extensions pass, including their pooled bounds.

The final combined candidate passes 1,266 tests on its first boot. The controller
yielded before the repeat boot for a queued owner-authorized sp8 check; no test
or timing job was interrupted. `0014` moves the same 26 transfer cases into the
existing output suite, following its contribution guide; runtime and GPU source
are unchanged. All 599 patches reproduce `9c4c49ec3bbcec299f3f48eb87012106789a0303`.
The inventory accepts the generated `_kernel` suffix on the harness markers.

## Actual kernel duration

Both final combined boots pass 1,266 tests. Both fixed wire boots contain no ATen
packing kernel in any of 16 public score-to-CPU cases. All 12 changed-path transfer
cases pass both complete host-call/stream-wait gates; means are about 11–16% lower.

The divide-free tiny-tail primary remains unresolved and is retained. Its earlier
CUPTI inventory measured 0.96 us for the candidate and 1.056 us for the independent
ATen reference, while external-event intervals averaged about 4 us for either.
`activity.py` tests whether event-node overhead is obscuring the actual compute
change: same inputs and balanced/mirrored roles, 120 rounds of 64 cold samples,
CUPTI hardware activity start/end timestamps, no event nodes in those graphs.
Every replay, kernel name, stream, sample count and positive duration is validated;
the five-standard-error, zero-margin gate is unchanged. Single inventory samples
are diagnostic, never sufficient qualification. Full serving remains separate.

Transfer-case count correction: the matrix contains 2 ID widths × 4 row counts ×
3 strides + 2 lifetime cases = 26, not 50. Both combined totals remain 1,266.
Patch descriptions and the narrow PR now use the actual 26-case count.

`final-iket-v10` stops in the probe's independent byte-comparison helper: FP32
`.float()` retains a sliced input's non-unit column stride, which cannot be
reinterpreted directly as bytes. The helper now materializes logical values on
both sides before its byte comparison. The raw failure remains on sp8 and will
be archived. This changes no production code. Full-model inventory runs first
in the resumed queue so additional runtime kernels are discovered promptly.

## Full-model findings

The first TP4-shaped 43-layer trace completes. Its short-name inventory leaves
66 request launches unidentified; full demangled symbols identify ATen embeddings,
an anchor gather, a confidence stack, copies/casts and clamps. Startup and grid
fixtures have additional work. The inventory now retains both names and classifies
full symbols, so namespaces cannot disappear in a short-name lookup.

`0015` routes TP1 through the already implemented fused vocabulary-shard drafting
path: one rank owns the full vocabulary and its pair gather is a view. It shares
the multi-rank supported-head warmup requirement. Existing exact fused-versus-
materialized tests now include TP1's full vocabulary. The fresh 600-patch source
tree is `24ee0b8367fd68179fcae6f0452c69484ed759f3`; GPU/full checks are pending.
A diagnostic-only `IKET_CALL_ORIGINS=1` mode records CUDA ATen call stacks and
metadata through TorchDispatchMode, without reading device data. It will identify
the residual casts/clamps rather than assigning their caller by kernel name alone.
Four lint diagnostics outside the source diff are inherited (two unused imports,
one unused fixture variable and one long fixture line); other checks pass.

The first activity attempt stopped before any workload at an outdated operation
assertion in the driver. `activity` is now admitted by that assertion as well as
both argument parsers. The raw failed receipt will be archived.

## Native TP1 embedding IDs

Both single-rank drafting boots pass all 53 focused tests; their full model
trace preserves the dummy request's token IDs and logprob bits. Call origins
pin the remaining request clamps/casts to `VocabParallelEmbedding.forward`,
which normalized IDs even when its TileLang method could read their native type.
`0016` folds negative-ID normalization into the TP1 lookup. It also gives plain
TP1 tables a TileLang lookup, supports flat native strides, and keeps parallel
non-clamping specializations. Other quantization methods retain their existing
preprocessing. Thirty-six new public eager/graph cases cover both table forms,
both ID widths and empty/shifted/strided/matrix IDs. New GPU checks are pending.

`embedding_bench.py` compares the actual public forward calls against the exact
previous method for 32 cases: plain/packed BF16 tables, int32/int64 IDs, 1/8/96/4096
rows and strides 1/2. It retains cold paired samples, inventories, input hashes
and both parallel-specialization CUDA sources, so unchanged-path identity can
be checked without assuming it. The fresh 601-patch source tree matches
`155dbd0d578b2f1fb08729abcc945707f82d94ff`.

`embedding-tests-v17` passes 113 cases then stops at two strided packed-ID
validation failures. The new kernel supports those strides, but an old combined
input/table contiguity guard still rejected them. `0017` relaxes only the native
input side; packed table and output requirements remain strict. No GPU program
changes. Both full test boots and the performance screen are rerun; the first
failure remains part of the evidence.
