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

**Candidate 1 on the ring (2026-10-23 10:47-14:20 CET, run `next-rc1-tp4`): passed.** Release
tests on dgx4: 2,081 passed, 159 skipped, none failed. Gate: temperature-0 outputs identical to
1022's ring hashes (3eab5112 / db25fde3, c9cf9dbe / c6bb4f4e), control spellings and the controls
probe pass, no ordinary sink counts and no collapsed probes on any rank; TTFT trace p50 217.4 ms at
4 streams, 286.3 ms at 8, 122.9 ms isolated. Comparison, verdicts against the run's own repeat:

| Metric | spark-ds41f-1022 | candidate 1 | Change | Verdict |
| --- | --- | --- | --- | --- |
| decode step, grid geometric mean (20 shapes) | | | -1.05% ±0.17% | better |
| 16 streams, output tok/s | 254.05 | 253.64 | -0.16% | same |
| 1 stream, output tok/s | 73.39 | 73.31 | -0.11% | same |
| prefill 32,768, tok/s | 6,009 | 6,364 | +5.90% ±0.49% | better |
| prefill 131,072, tok/s | 5,971 | 6,329 | +5.99% ±0.66% | better |
| prefill 1,000,000 (one cold run), tok/s | 4,417 | 4,670 | +5.7% | (no interval) |
| sampled, 16 streams, output tok/s | | | -0.28% | same |

Greedy responses 64/64 identical, teacher-forced KL 0, acceptance unchanged; TTFT level (16
streams p50 0.404 -> 0.403 s, 1 stream 0.129 -> 0.129 s). Lowest MemAvailable 21.0-23.4 GiB per
node in every arm.

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

**Result (same window, 14:20 CET, run `next-budget-tp4`).** Verdicts 16,384 -> 8,192 against the
8,192 repeat:

| Metric | 8,192 | 16,384 | 16,384 vs 8,192 |
| --- | --- | --- | --- |
| prefill 32,768, tok/s | 6,364 | 6,742 | +5.9% |
| prefill 131,072, tok/s | 6,334 | 6,730 | +6.3% |
| prefill 524,288, tok/s | 5,506 | 5,805 | +5.4% |
| prefill 1,000,000 (one cold run), tok/s; TTFT | 4,673; 214.0 s | 4,900; 204.1 s | +4.8% |
| prefill 131,072 under 15 decoding streams, tok/s | 5,759 | 6,308 | +9.5% |
| those streams' decode step during it, p50 / p99 | 1.33 / 1.42 s | 2.36 / 2.61 s | +78% / +84% |
| their worst wait | 2.69 s | 3.37 s | same (±95%) |
| lowest MemAvailable (dgx1, the tightest node) | 22.0 GiB | 18.7 GiB | -3.3 GiB |

The trade is the chunk: twice the tokens per long-prefill step, so the prompt finishes 5-6% sooner
(9.5% under decode load) while every stream decoding beside it waits about twice as long per step
for that prompt's duration. Memory stays 13.7 GiB above dgx1's 5 GiB guard. With candidate 1,
16,384 would put prefill at +12.2% (32K), +12.7% (131K) and +10.9% (1M) over spark-ds41f-1022
(the 8,192 arms of the two sessions agree to 0.01% at 32K).


## Before promotion: the remaining items

Owner, 2026-10-23: "we won't promote until we address the 'other items' at least" (the live-name
rename, the start-up memory peak, draft acceptance, GLP steering on, and #439's memory growth),
then "do 16 and work through the rest": the ring's long-prefill budget becomes 16,384 in the
release (candidate 2 carries it; its gate and comparison qualify it).

### Memory: #439's growth and the start-up peak ([memory/](memory))

#439 (JordiPosthumus) measured, on two TP3 rings at 1M, about 1.2 GiB lost over three hours of
ordinary use: the API server 1.68 -> 2.44 GiB and the engine core 1.43 -> 1.97 GiB, never
returned; the GPU worker flat. Production here has served nothing since its 14:20 CET restore
(API server 1.69 GiB, engine core 1.49 GiB, worker 4.86 GiB at 16:44), so the growth is
reproduced under load in a window, not waited for.

[memory-tp4.json](memory/memory-tp4.json), one window on the ring, two arms of candidate 1 at
16,384 with the start-up allocator trace (a diagnostic overlay of `gpu_worker.py`, behaviour
unchanged; from experiments/2026-10-21-startup-memory):

1. [soak.py](memory/soak.py) for 75 minutes: four agent sessions resend their whole history
   every turn, growing by a 16-48K-token real-text tool result until they pass 960K tokens and
   start over, beside eight short chat streams. Every turn makes the API server parse, render
   and tokenize up to a million tokens again and the engine core hash and schedule them.
2. The same with glibc malloc's growth sources pinned (`MALLOC_ARENA_MAX=2`, a fixed 1 MiB
   `MALLOC_MMAP_THRESHOLD_`), as a hypothesis test.

[sampler.py](memory/sampler.py) runs on every node through the window (armed by
[watch.sh](memory/watch.sh)): MemAvailable every 0.1 s, every container process's RSS every
0.5 s, and every 30 s the API server's and engine core's resident memory by mapping kind (brk
heap, CPython arenas, glibc thread arenas, large single mappings, files), which separates
allocator high-water from live objects. Each arm's two boots also give the start-up peak's
attribution by phase (allocator trace against the engine log) on four nodes.

**Result (2026-10-23 17:15-19:50 CET, run `next-memory-tp4`).** Both arms served the same load
without an error: 554 agent turns (20 restarts of a session, prompts up to 966,014 tokens, 270M
prompt tokens) and about 310 chat requests. dgx1, where the API server and engine core run:

| Process | Plain: soak start -> 10 min -> 75 min (high) | Malloc pinned |
| --- | --- | --- |
| API server | 1.76 -> 2.84 -> 2.92 GiB (3.19) | 1.67 -> 1.95 -> 2.02 GiB (2.28) |
| engine core | 1.51 -> 1.64 -> 1.67 GiB | 1.44 -> 1.55 -> 1.61 GiB |
| lowest MemAvailable during the soak | 17.10 GiB | 18.11 GiB |

By mapping kind, the plain API server's growth is glibc's per-thread arenas: 79 MiB in 19 arena
mappings at the start, 1,158 MiB in 36 at the end (+1,078 of its +1,159 MiB); the brk heap, CPython's
arenas and file mappings barely move. It is a high-water mark, not a leak: the API server reaches
it within ten minutes and holds it (2.72-2.99 GiB from 20 to 75 minutes). Each request thread that
parses, renders and tokenizes a million-token conversation frees its buffers into its own arena,
and glibc keeps an arena's freed memory; glibc also raises its mmap threshold after a large free,
so later large buffers come from the arenas instead of mmap. With two arenas and a fixed 1 MiB
threshold the API server grows 0.35 GiB, nearly all in the first ten minutes (1.95 GiB at 10
minutes, 2.02 at 75). The engine core's smaller growth (+0.16 GiB) is the same in both arms.
#439's three hours of ordinary use (API server +0.76 GiB, engine core +0.54 GiB) fit this.

The fix is the two settings in the recipe's environment (`MALLOC_ARENA_MAX=2`,
`MALLOC_MMAP_THRESHOLD_=1048576`), for every process of the container; candidate 2 carries them and
its release comparison checks that they cost nothing.

**Start-up on the ring: nothing to cut.** Each arm's boot was traced (allocator totals every
0.1 s against the engine log; [analyze_startup.py](memory/analyze_startup.py)). dgx1 reaches API
readiness with 19.15 and 19.32 GiB available, 14 GiB above the 5 GiB guard; the lows after that
(18.93, 18.69 GiB) come with the soak's first minute, not the boot. torch's reserve only grows
through the boot (72.9 GiB weights, 77.7 with the KV pool, 88.1 after preparation, 91.0 after graph
capture; nothing is released and reallocated), and MemAvailable falls as the page cache gives way
to the GPU's memory. The 0.5-1.2 GiB released late after TP3 boots (2026-10-22, four boots) does
not occur at TP4; attributing it needs the triangle cabled. What TP3's guard was short of in #439 is
steady headroom on the head node, which the arena fix returns (about 1 GiB at the soak's load).

### A 24,576 budget ([budget24-tp4.json](budget24-tp4.json))

Suggested to the owner on 2026-10-23: try 24K. [budget-24576-tp4.json](budget-24576-tp4.json) is
candidate 1 at 24,576 (threshold 24,480); the sweep is the 16,384 one's suite, 24,576 against
16,384 (24,576 first and again), queued behind the memory window in the same window. Expected:
a smaller step than 8K -> 16K (the per-step overhead is already amortized over twice the tokens,
and the CED skip trims fewer nonfinal chunks), about three times 8K's stall step, and about
3 GiB more peak memory.

**Result (2026-10-23 19:50-20:29 CET, run `next-budget24-tp4`).** Verdicts 16,384 -> 24,576
against the 24,576 repeat:

| Metric | 16,384 | 24,576 | Change |
| --- | --- | --- | --- |
| prefill 32,768, tok/s | 6,733 | 6,907 | +2.6% |
| prefill 131,072, tok/s | 6,733 | 6,865 | +2.0% |
| prefill 524,288, tok/s | 5,805 | 5,902 | +1.7% |
| prefill 1,000,000 (one cold run), tok/s | 4,893 | 4,962 / 4,981 | +1.4-1.8% |
| prefill 131,072 under 15 decoding streams, tok/s | 6,311 | 6,531 | +3.5% |
| those streams' decode step during it, p50 / p99 | 2.36 / 2.61 s | 1.63 / 3.74 s | -31% / +43% |
| their worst wait | 4.10 s | 6.04 s | same (interval ±100%) |
| lowest MemAvailable (dgx1) | 18.0 GiB | 15.4 GiB | -2.6 GiB |

The second step is a third of the first (8,192 -> 16,384 gave +5.4-6.3%): 2% more prefill for
steps up to 43% longer at p99 beside a long prompt and 2.6 GiB less headroom. The release stays at
16,384 (owner, 2026-10-23: "do 16").

### Draft acceptance ([acceptance/](acceptance))

The ceiling first (experiments/2026-10-18-dspark-acceptance, candidate 2: copy-aware drafting).
On a local proxy for agent traffic (634 coding-agent sessions, 71,030 assistant turns, 26.4M
output tokens rebuilt in DS4.1's prompt format and tokenized with its tokenizer; only counts
kept), a suffix drafter (propose up to 16 tokens after the latest earlier occurrence of the last
8) would cover 38% of output tokens. Taking its proposal whenever there is one and DSpark's flat
2.25 tokens a step elsewhere gives 3.27 tokens a step (+45%); +28% at a 16-token match, +37%
without DSML tool-call markup. The transcripts hide reasoning (about half the output); if it
copies like prose, the figure is +14-23%. Most of the gain is in tool-call arguments (Bash +49%,
edits +31%); prose gains 6%.

What that leaves open is DSpark itself on those spans: it may already accept most of a copy.
[copy_probe.py](acceptance/copy_probe.py) measures both on the same output: one stream,
temperature 0, four categories (prose; reproduce a source file; the file with a comment above
every `def`; an edit tool call whose old_string is a copied function), DSpark's tokens a step from
the spec-decode counters, and a suffix drafter simulated on the exact token ids the server
returns. Queued as [copy-tp4.json](acceptance/copy-tp4.json) in the same window.

**Result (2026-10-23 20:29-20:37 CET, run `next-copy-tp4`, candidate 1).** Twelve requests a category,
no errors; DSpark's tokens a step from the counters, a suffix drafter simulated on the same token ids:

| Category | Output tokens | DSpark | Suffix, 8-token match, up to 16 | 16-token match | 8-token match, up to 8 |
| --- | --- | --- | --- | --- | --- |
| prose | 4,464 | 2.24 | 2.24 | 2.24 | 2.24 |
| reproduce a file | 18,177 | 4.96 | 15.9 | 16.3 | 8.4 |
| the file with comments added | 18,310 | 5.10 | 14.9 | 14.9 | 8.2 |
| an edit tool call | 2,122 | 5.41 | 7.1 | 7.3 | 6.1 |

DSpark already takes about 5 of its 6 possible tokens a step on copied text: the transcripts
study's flat 2.25 overstated the gap. What a suffix drafter adds is length: 16 tokens instead of
DSpark's block of 5, three times the tokens a step on long copies, a third more on tool calls,
nothing on prose. Weighted by the agent proxy's copy share (38% of output tokens, 19% if hidden
reasoning copies like prose), with DSpark at about 5 on copies and 2.25 elsewhere, that is
roughly +7-17% tokens a step at one stream, before the cost of verifying 17-row blocks. Building
it is a hybrid drafter (per request, a suffix proposal when one matches, DSpark otherwise),
variable-length block verification and graphs for the wider blocks: the next release's work, not
this one's.
