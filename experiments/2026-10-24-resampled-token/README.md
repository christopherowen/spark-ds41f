# Resampled-token selection and insertion in TileLang

The final speculative token selector now runs in TileLang. It determines the
resampled or bonus token emitted during generation, increments the emitted count,
and retains the target argmax already stored for a greedy rejection. Acceptance,
random draws and verification arithmetic are unchanged. The original Triton
kernel remains only as the correctness and timing reference.

## Qualification

- **125 related tests pass**, covering rejection sampling, vocabulary-parallel
  rejection, sink recovery and worker preparation. Eight demonstrated baseline
  draft-sampling failures are deselected, as explained below.
- **36 focused cases pass CUDA 13.0 memcheck with zero errors**: DS41F vocabulary
  129,280 and five drafts, TP3/TP4 winners, block boundaries, NaNs, infinities,
  subnormals, signed zero, exact ties, FP32/FP64 partials, int32/int64 mappings,
  padded views, mixed temperatures, graph replays with changed counts, and startup
  declarations for optional verification modes.
- All **24 paired timing gates pass**, on fresh seeds and allocations, after an
  independent 24-case screening pass. No slowdown tolerance was added. These are
  kernel measurements, not a full-model throughput claim.

Default sampling, 127 vocabulary-block winners (microseconds per call):

| Requests | TileLang | Triton |
| ---: | ---: | ---: |
| 1 | 1.575 | 1.592 |
| 8 | 1.582 | 1.606 |
| 16 | 1.583 | 1.606 |
| 96 | 1.591 | 1.634 |

At 8/16 requests the paired mean is about 1.3–1.5% lower; at 96 requests it is
2.7% lower. The 3/4-rank reductions are generally 4–6% lower. One-request cases
have wide intervals. In particular, the mixed-temperature 127-block case has a
median of **1.130 vs 1.064 microseconds**, with paired 95% interval
[-0.0212, 0.1823] microseconds for the difference. That case is
inconclusive under the gate; this evidence does not establish every path as
faster or equivalent.

[results.json](results.json) keeps every final sample, interval, image identity and
input hash. [qualified/](qualified/) contains the fresh timing confirmation and
its 121-test/memcheck receipt. The subsequent [startup-qualified/](startup-qualified/)
bundle qualifies the complete five-file source, with 125 tests and 36 memcheck
cases. The preparation fix changes only registration; its kernel and benchmark
bytes equal those in the timing confirmation. All source deltas from candidate 4
are mounted and match the committed source.

## Implementation and startup

The selected layouts are one thread for up to four rank winners and one warp for
the normal vocabulary-block reduction. Physical row strides are compile keys;
request counts and buffer lengths remain dynamic. NaNs map to negative infinity,
real blocks alone may win, and the first maximum wins ties. FP64 support preserves
the existing reduction input contract; it does not add a sampling option.

The insertion predicate uses integer flags combined with bitwise OR, preserving
Triton's eager evaluation of the temperature and bonus conditions. A C++ logical
OR allowed the compiler to defer the end-offset read on the greedy skip path.
Alternative cache changes and bonus-first short-circuiting regressed and were
rejected.

The rejection-sampler owner declares insertion before its draft-logit/watermark
warmup early return, including the configured rank-winner count. Otherwise these
modes would reach the new TileLang call after cache freeze without a declaration.
Both mapping-index types and normal partial widths are prepared through the same
key function that dispatches the runtime call.

## Reproducibility and release boundary

Base source: `174a746f4577579841e5aa3e94c8a4964b0aa326`, candidate-4 image
`spark-ds41f:19f2c20ed4d6-next-4`, image ID
`sha256:1ae57dac1017258128b46665308dd490f1284fb1dd6926dd2c1a3685ea04b82d`.
One causal variable is the final token insertion implementation; screens below
isolate its layouts, load behavior and preparation.

- Source head: `dd73fcd45328ddf405e17fe2f1f0d303d19bee99`.
- Fresh 223-patch replay: `9a4dec9d28e3adb75b166c28f5e4dd93c1689c53`.
- Exact source/replay tree: `33165a773ff28a2a1b70910f2d7ac54a04ea3f88`.

The [complete series](vllm/series), [source manifest](source.json),
[source lock](upstreams.lock.json) and [launch-disabled profile](candidate-tp4.json)
are build inputs. Configuration checks, Ruff, whitespace checks and fresh replay
pass. [Release classification](release-tests.json) selects 127 files, 70 gates,
with zero classification problems. No new release image was built or promoted;
full-model serving and the complete release gate remain outstanding.

Each bundle records its exact Python commands in `candidate.json`. Timing uses
8 allocations and alternating order, 100 kernels per CUDA graph, a paired 95%
interval and no slowdown tolerance. Each graph call has a distinct count buffer,
reset outside the timed interval, so in-place increments cannot overrun output
slots. The small pytest fixture plugin supplies the serving image's missing
conftest environment without altering test assertions.

## Existing draft-seed discrepancy

The unchanged candidate-4 image reproduces all eight failures in
`test_vocab_parallel_draft_matches_full_vocab`: see
[baseline-control/run/](baseline-control/run/). Four vocabulary-parallel rejection
tests pass. Full-vocabulary drafting mixes the speculative cache column into
`spec_step_seed`; vocabulary-parallel drafting does not. Column zero agrees,
while later columns use different random streams. The insertion kernel is not
called by these failing tests.

This port leaves that sampling behavior unchanged. Only those eight demonstrated
baseline failures are deselected from the related suite. The release
classification retains the entire file as a gate, so full release qualification
remains blocked. No improvement to draft acceptance is claimed.

## Preserved tuning history

| Evidence | Result |
| --- | --- |
| `screen/` | 19 tests pass; generic two-reduction layouts miss timing gates. |
| `integer-screen/` | 20 tests pass; generic integer ordering remains slower. |
| `warp-screen/` | 20 tests pass; explicit warp intrinsics lose on serial per-lane work. |
| `packed-screen/` | Shuffle argument order caused compilation failure; traceback retained. |
| `packed-fixed/` | 20 tests pass; one packed 64-bit reduction remains slower. |
| `hierarchical-screen/` | 20 tests pass; one block barrier narrows the deficit. |
| `qualification/` | 20 memcheck cases pass; 109 related tests pass, 8 baseline failures; timing gates fail. |
| `stride-screen/` | 20 tests pass. Numeric output was overwritten by the control's reused bundle basename; no conclusion uses it. The next screen repeats the layouts with distinct output names. |
| `warp-float-screen/` | Selected simple layouts pass all screened timing gates. |
| `final/` | 32 memcheck cases and 121 related tests pass; default sampling passes, mixed timings fail. |
| `cached-loads/` | 32 memcheck cases and 121 tests pass; explicit cached reads regress. Local offline compilation was unavailable because its Docker socket was absent. |
| `lowering-probe/` | Native Triton PTX and TileLang CUDA sources show the control-flow difference. |
| `bonus-first/` | 32 memcheck cases and 121 tests pass; short-circuit reordering regresses timing. |
| `unrestrict/` | Annotation outside a block failed compilation; traceback retained. |
| `unrestrict-fixed/` | Corrected annotation placement still misses speed gates. |
| `early-end/` | 32 tests pass; forcing an end-offset load regresses ordinary sampling. |
| `eager-predicate/` | 32 tests and both 12-case speed gates pass; selected implementation. |
| `qualified/` | Fresh confirmation: 121 related tests, 32 memcheck cases, all 24 timing gates pass. |
| `startup-qualified/` | Complete source: 125 related tests and 36 memcheck cases pass; measured kernel unchanged. |

An initial metadata screen was not executed because the preceding window's time
cap had elapsed. Its log is retained separately; the two cases then ran in a fresh
window. Native transcripts retain their original bytes, including traceback spacing; only
these generated output files are excluded from whitespace lint by the local
`.gitattributes`. Code and patch whitespace checks remain enabled.

Every executed screen retains its exact source bundle and native receipt,
including errors and unfavorable timings. Window logs record coordinated stops
and restoration of the promoted service.

## Service restoration

The final window closed at **14:39:42 UTC on 24 October 2026**, after the
promoted recipe passed `doctor --live`; its hold was removed. No candidate was
promoted. [The final window log](startup-qualified/window.txt) records restoration.
