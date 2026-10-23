# The next release after spark-ds41f-1022

Owner, 2026-10-23: clear the open issues and PRs, keeping our quality standards: rework a
contribution to meet them, close the ones we will not accept. "We must keep our quality
standards and it would be nice to have a release with some performance boost." The release
continues the TileLang migration: every kernel the deployment launches moves to TileLang, each
port meeting or exceeding the kernel it reproduces; kernels that never launch are not ported
and leave with their engine; fusion work starts once the serving path is TileLang.

Base: spark-ds41f-1022 (image `-1022-4`, vLLM patch head 8fc15c18f, tree e68aa19c), TP4
production on the ring since 2026-10-23 02:10 CET. The triangle keeps 1022's qualification
until it is cabled again.

## Contents (planned)

| Piece | Where it comes from | State |
| --- | --- | --- |
| CED row selection in TileLang (vLLM: the 26 commits of experiments/2026-10-19-ced-kernels) | promised in the r6x and 1022 forum posts | ports bit-identical and 2-31% faster per call at serving shapes; full-model gate identical to r6x (TP3) |
| CED nonfinal row skip | experiments/2026-10-19-ced-skip, -ced-gates | TP3: outputs identical, prefill +10.6-11.1% from 32K to 500K; TP4 unmeasured |
| Block verification: fresh random draws across rounds (backport of vllm-project/vllm#61161) | experiments/2026-10-19-sampling-rounds | kernel gate: 74 sampler and DSpark tests pass; temperature 0 unchanged, sampled text changes by design |
| V4.1 tokenizer: `add_generation_prompt` and `continue_final_message` | the v41-generation-controls branch (2026-10-19) | CPU tests pass |
| The DeepSeek API's top-level `thinking` switch (#447) | this record | CPU tests pass (108 tokenizer, 21 request) |
| The drafter's FP8 linear in TileLang (off B12X) | the TileLang migration | to build; gate: bit-identical, per call at least B12X's |
| The launched-kernel inventory | the TileLang migration | one profiled boot, every request type the deployment serves |
| GLP steering, off unless configured (#401, msuiche) | reworked on vLLM 3e09d60c2 | to rebase; off must be identical; its projection a TileLang kernel |
| RoCE GIDs by address (#367, sparknet #1, sparknet 0.6.0) | experiments/2026-10-18-roce-gid-by-address | to qualify on the ring |
| Live names: spark-ds41f in place of karmic-kraken, kkref and spark3 | release naming (forum post #51) | to do; caches move, not rebuilt |

The vLLM branch is `release-next` (local, as vLLM working branches are): the 1022 series
replayed to 8fc15c18f (tree e68aa19c, exact), then the sampling fix, the CED ports, the skip,
the generation controls (one test-file conflict with 0227, both sets of tests kept) and the
thinking switch.

## Queue (owner's decisions, 2026-10-23)

| Item | Decision |
| --- | --- |
| #461, #364 (ours) | merged |
| #359 (ours) | rework: keep the externally reported notes, drop the comparison with another stack (repository rule) and the stale r6v 4 KiB plan |
| #439 (JordiPosthumus, memory creep at 1M) | accept after reproducing on our cluster; the growth itself is root-caused separately |
| #410 (JordiPosthumus, memguard) | take the warning line and the breach logging; the kill stays on the first sample below the line |
| #450 (mxfp0, a vLLM 0.32 recipe) | the recipe closes: a second stack on another base, no serving gate; its scheduler metrics fix is taken as a patch with its tests if it reproduces on our tree |
| #401 (msuiche, GLP steering) | accepted as a default-off option, reworked to our standards |
| #455, #434, #30 | fixed (recipe flag; capture sizes from the sizing; image-limit docs) |
| #447 | fixed (above) |

## Ring screen ([screen-tp4.json](screen-tp4.json))

[make_screen.py](make_screen.py) writes it from the vLLM branch: every file the branch
changes since 8fc15c18f is a runtime overlay over image `-1022-4`
([overlays/](overlays), the arms mount the 8 sources; no test code). One window on the ring,
production restored once at the end:

1. **GPU tests** ([bundles/gpu-tests](bundles/gpu-tests)), the cluster stopped: the CED TileLang
   kernels and metadata, the skip, the CED model and router, block-verification sampling.
2. **Gate** on [candidate-tp4.json](candidate-tp4.json): temperature-0 outputs identical to 1022's
   ring hashes (3eab5112 / db25fde3, c9cf9dbe / c6bb4f4e), control spellings, sink counters and
   probes on every rank.
3. **Screen**, one session, candidate first: [screen-suite-tp4.json](screen-suite-tp4.json) on
   [candidate-tp4-dev.json](candidate-tp4-dev.json), spark-ds41f-1022, the candidate again.
   1022's trimmed TP4 promotion suite (model at 16 streams, grid, closed loops at 16 and 1, 32K and
   131K prefill), one cold 1M prefill, and the sampled suite's sixteen streams.

This is a screen, not the release comparison: the release image, its release tests and the
comparison follow once the remaining pieces are in.

**First screen (2026-10-23 09:11-09:15 CET): stopped at the candidate's boot.** The GPU tests
passed on dgx4 (341 passed: CED kernels, metadata and skip, CED model and router, block-verification
sampling). The candidate then stopped in memory profiling on every rank: "TileLang kernel
'ced_copy_rows' was not prepared ... copy_rows_kernel('float32', (4,), (1,), True, True)". A
compacted decoder scatters the sink detector's columns ([tokens, hc_mult] float32, 0221) back to
every row; the CED ports, qualified on r6x before the detector existed, declared that layout only
under pipeline parallelism. Kernel preparation stopped the boot by name, as designed; the window
restored production at 09:15 (`doctor --live` OK). 0259 declares the layout when the model keeps
the columns, with a test that the frozen cache serves it and refuses it without the declaration.
Preserved logs: [runs/failed-starts](runs/failed-starts).

The second screen (run `next-release-screen2-tp4`) adds 0259, GLP steering (unset, so the arms
must equal 1022), the GLP tests in the GPU bundle and [controls_probe.py](controls_probe.py) in
the gate.

## Release candidate 1 (image `-next-1`)

[vllm/series](vllm/series): 1022's series (184 patches, the local ones by path into the 1022
record) plus 0228-0260, exported from `release-next`:

| Patches | Change |
| --- | --- |
| 0228 | block verification refreshes its random draws across rounds (vllm-project/vllm#61161 backport) |
| 0229-0254 | CED row selection in TileLang: requests, row map, row copy, their benchmarks and tests |
| 0255 | the CED nonfinal row skip |
| 0256 | the V4.1 tokenizer honors `add_generation_prompt` and `continue_final_message` |
| 0257 | the DeepSeek API's top-level `thinking` switch (#447) |
| 0258 | GLP steering, off unless `WEIGHTLESS_STEER_PATH` is set (#401, Matt Suiche) |
| 0259 | CED declares the sink detector's column scatter (found by the first screen, below) |
| 0260 | the DS4.1 model test stubs carry the GLP steering attribute (found by the second screen's GPU tests) |

A clean `git am` of 0228-0260 on 8fc15c18f (the build's committer and dates) gives patch head
2ed0a422, tree 1232558d, the branch's tree ([source.json](source.json),
[upstreams.lock.json](upstreams.lock.json)). Release tests: `scripts/release_tests.py check`
selects 126 files, 69 gate (four new: the thinking switch, the TileLang CED kernels, GLP, block
verification's draws, classified in config/release-tests.json), 0 problems;
[release-tests-gate-manifest.json](release-tests-gate-manifest.json) and
[bundles/release-tests](bundles/release-tests) (its `head` is the branch's 1fa3028ec, same tree).

[rc1-tp4.json](rc1-tp4.json), one window on the ring: the image built on dgx2 and copied over the
ring's links ([image-rc-dgx2.json](image-rc-dgx2.json)); the release tests on dgx4; the gate on
[rc-tp4.json](rc-tp4.json) (determinism against 1022's ring hashes, control spellings,
[controls_probe.py](controls_probe.py) for 0256 and 0257, the TTFT trace, memory, sink counters
and probes); then the one-session comparison on [screen-suite-tp4.json](screen-suite-tp4.json):
the candidate, spark-ds41f-1022, the candidate again. [rc-tp3.json](rc-tp3.json) is the triangle's
arm for when it is cabled.

GLP steering ships off: its gate here is that the unset path leaves outputs identical. Steering on,
with its projection as a TileLang kernel, is qualified in its own experiment (the migration rule:
a kernel that does not launch is not ported; this one launches only when configured).

## The ring's long-prefill budget with the skip ([budget-tp4.json](budget-tp4.json))

Owner, 2026-10-23 11:45 CET: "measure a 16384 prefill budget on the ring too". On four Sparks the
skip measured prefill +5.6-6.0% (32K-1M) in the second screen, half the triangle's +11%: the ring's
chunks are 8,192 tokens, so a long prompt has half as many nonfinal chunks to trim.
[budget-16384-tp4.json](budget-16384-tp4.json) is candidate 1 at a 16,384-token budget (threshold
16,384 - 16 x 6 = 16,288), everything else as [rc-tp4.json](rc-tp4.json).
[budget-suite-tp4.json](budget-suite-tp4.json) is the TP3 sweep's suite at TP4: cold prefill at 32K,
131K and 524K (three repeats), the stall step (what fifteen decoding streams wait during a cold 131K
prefill), and one cold 1M prefill; the run's memory watch records every node's lowest MemAvailable.
One session on image `-next-1`: 8,192, 16,384, 8,192 again. Queued behind candidate 1 in the same
window. The budget is the owner's choice; the recipe does not change here.
