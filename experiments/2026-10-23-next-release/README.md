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
