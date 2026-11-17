# Indexer share of a long-context decode step

Date: 2026-11-17. Base commit: `00306d1ef`. Image: `spark-ds41f:candidate-60-afcaac35`
(candidate-60, vLLM tree `afcaac3594ac`). Testbed: `tp4-switched` (sp1..sp4).

## Question

How much of one stream's decode step does the indexer take at 4K, 237K and 910K tokens of
context? The answer bounds what a cheaper selection could gain in decode.

Two recorded results disagree, so neither answers it:

- **The engine grid's fixed decode step does not slow with context.** Candidate-60's
  screens (`../2026-11-16-c60-prefill/receipts/final/`, two boots each) give, for one
  request at 3 rows, 13.80 / 13.75 ms at 4K and 13.90 / 13.88 ms at 1,016K on
  `tp4-switched`, and 18.40 / 18.31 ms at 4K and 18.23 / 18.22 ms at 1,016K on
  `tp3-switched`.
- **The score kernel alone is slow at depth.** On sp8
  (`../2026-11-16-indexer-prefill-exact/receipts/sp8/final-1`), 6 rows cost 444 µs a launch at
  524,261 positions on a ratio-2 layer and 1,254 µs at 1,048,549 positions on layer 20,
  against 6 µs and 10 µs at 4K. A decode step scores every row on every rank
  (`attention.py`, `_sp_front`), so three ratio-2 layers and layer 20 would add about
  2.6 ms to a 6-row step at 1M. That bench gives each row its own page order and ran on
  the spaceport `candidate-60` tag (vLLM tree `b7c6c958ada3`).

## One variable

Context depth, on one boot. Nothing else changes between the windows.

## Method

Two jobs on the same image and testbed.

**Timing** ([depth-tp4-switched.json](depth-tp4-switched.json)): a `measure` job, two boots
of [dev-tp4-switched.json](dev-tp4-switched.json), which is
`../2026-11-16-c60-prefill/tp4-switched-dev.json` with only the image tag changed.
[profile_decode.py](profile_decode.py) `--no-profile` runs, at each depth (size labels
4,096, 131,072, 262,144, 524,288 and 1,000,000 of servebench's cold source-text document,
seeds 0..4):

1. `fill`: the document with one output token, so its prefix is cached;
2. `real`, twice: the same request for 12 + 192 output tokens (temperature 0, thinking
   off, `ignore_eos`). The decode span starts 12 output chunks in; its steps, output
   tokens and verified draft rows are the server's counters over the span;
3. `grid`: the engine grid's fixed step for one request at 3 and 6 rows at that context
   (a 4,096 multiple), 50 replays.

The growth of a step with depth bounds what the indexer's depth-dependent work costs.
[tables_arms.py](tables_arms.py) prints the table.

**Kernel attribution** ([profile-tp4-switched.json](profile-tp4-switched.json)): a `profile`
job on [prof-tp4-switched.json](prof-tp4-switched.json), the same profile with
`--profiler-config` appended (torch profiler on every rank, no Python stacks or shapes).
The script without `--no-profile` adds a profiled repeat of phases 2 and 3 at 4,096,
262,144 and 1,000,000, and [summarize_kernels.py](summarize_kernels.py) sums GPU kernel
time by kernel name for every window of every rank.

```sh
bin/spark candidate queue tp4-switched experiments/2026-11-17-indexer-decode-share/depth-tp4-switched.json --apply
bin/spark candidate queue tp4-switched experiments/2026-11-17-indexer-decode-share/profile-tp4-switched.json --apply
```

## Runs

| Run | Result | Note |
| --- | --- | --- |
| `profile-tp4-switched.json`, 2026-11-17 13:42Z | refused before queueing | The queue's dry run on sp1 at `d0a111fa2`: `qualified config drift for tp4-switched: serve_args differ from qualified profile`. The guard (`spark_ds41f/doctor.py`, `qualified_recipe_problems`) compares an arm's serving arguments with its stack's qualified profile, and a profiler arm differs by `--profiler-config`. Nothing booted; the checkouts of sp1..sp4 moved to `d0a111fa2`. |
