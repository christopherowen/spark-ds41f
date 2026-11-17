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

Context depth, within each boot. Nothing else is set differently between depths; each
depth's document is its own.

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
time by kernel name for every window of every rank. The document's seed is the depth's
index, so the two deeper documents are not the timing job's (266,422 and 1,001,009 prompt
tokens here).

Kernel durations overlap: a kernel launched with programmatic dependent launch starts
before its predecessor ends, and the prefetch kernel runs beside the others. So
[busy_time.py](busy_time.py) reads the raw traces and gives, per kernel family, the summed
durations, the time owned (each instant counted once, for the running kernel that started
first) and the time alone (no kernel of another family running), with the window's busy
and idle time. [tables_kernels.py](tables_kernels.py) prints both. The traces stay on sp1
(`results/private/determinism/<run>/<label>/decode-trace/`, 49 MB a rank); `busy_time.py`
ran on a copy of them.

```sh
bin/spark candidate queue tp4-switched experiments/2026-11-17-indexer-decode-share/depth-tp4-switched.json --apply
bin/spark candidate queue tp4-switched experiments/2026-11-17-indexer-decode-share/profile-tp4-switched.json --apply
bin/spark candidate queue tp4-switched experiments/2026-11-17-indexer-decode-share/profile-again-tp4-switched.json --apply
# per rank, on a copy of results/private/determinism/<run>/<label>/
python3 experiments/2026-11-17-indexer-decode-share/busy_time.py decode-busy-sp1.json decode-trace/sp1
python3 experiments/2026-11-17-indexer-decode-share/tables_kernels.py experiments/2026-11-17-indexer-decode-share/receipts/tp4-switched/profile
```

## Runs

| Run | Result | Note |
| --- | --- | --- |
| `profile-tp4-switched.json`, 2026-11-17 13:42Z | refused before queueing | The queue's dry run on sp1 at `d0a111fa2`: `qualified config drift for tp4-switched: serve_args differ from qualified profile`. The guard (`spark_ds41f/doctor.py`, `qualified_recipe_problems`) compares an arm's serving arguments with its stack's qualified profile, and a profiler arm differs by `--profiler-config`. Nothing booted; the checkouts of sp1..sp4 moved to `d0a111fa2`. |
| `depth-tp4-switched.json`, queue entry `20261117T134542-depth-tp4-switched`, 13:46Z to 13:59Z | passed, both boots exit 0 | Deployment commit `026885029`. Boots ready in 132.3 s and 133.5 s, no kernel compiled. The live container of the first boot was checked: image `spark-ds41f:candidate-60-afcaac35`, this profile, development RPC on, no profiler argument. |

| `profile-tp4-switched.json`, queue entry `20261117T161803-profile-tp4-switched`, 16:18Z to 16:28Z | passed, exit 0 | Deployment commit `753add672`, after `c1e467d0f` let `--profiler-config` past the guard. Boot ready in 131.6 s, no kernel compiled. The live container was checked: image `spark-ds41f:candidate-60-afcaac35`, this profile, development RPC on, `--profiler-config` present. Six windows on each of four ranks. |

Receipts: [receipts/tp4-switched/](receipts/tp4-switched/) has the timing job's two boots
(output, node telemetry, the lab's table, the window log lines), and
[receipts/tp4-switched/profile/](receipts/tp4-switched/profile/) the profile job's output,
telemetry, per-rank kernel summaries, busy-time summaries, table and window log lines. GPU
temperature peaked at 79-82 °C and power at 73-77 W in the timing job, and at 76-80 °C and
72-76 W in the profile job, with no clock event; memory was not sampled.

## Results

### Real decode (one stream, two boots, two repeats each)

Target rows are 1 plus the verified draft rows per step. Each depth has its own document,
so rows and tokens per step differ between depths.

| Prompt tokens | Target rows per step | Output tokens per step | Step, mean (min to max) | Depth term of the fit | Share of the step |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,070 | 4.29 | 3.42 | 28.91 ms (28.27 to 29.34) | 0.02 ms | 0.1% |
| 131,880 | 5.53 | 4.46 | 33.24 ms (32.84 to 33.96) | 0.54 ms | 1.6% |
| 262,534 | 4.25 | 2.85 | 30.07 ms (29.78 to 30.27) | 1.08 ms | 3.6% |
| 522,754 | 4.64 | 3.38 | 32.07 ms (31.84 to 32.58) | 2.15 ms | 6.7% |
| 994,677 | 4.79 | 3.45 | 34.57 ms (34.01 to 35.06) | 4.10 ms | 11.9% |

The fit over the 20 spans is `step = 16.56 + 2.90 * rows + 0.0041 * (prompt tokens / 1000)` ms,
with a root-mean-square error of 0.33 ms. Without the row term the depth slope is 0.0043 and
the error 1.43 ms. Rows are confounded with the document: five documents carry the row
term, and only the 4K and 995K documents show two row counts.

### The engine grid's fixed step (50 replays, each boot)

| Context | 3 rows, boot 1 / boot 2 | 6 rows, boot 1 / boot 2 |
| ---: | ---: | ---: |
| 4,096 | 13.552 / 13.636 ms | 14.454 / 14.580 ms |
| 131,072 | 13.568 / 13.818 ms | 14.554 / 14.678 ms |
| 262,144 | 13.674 / 13.809 ms | 14.570 / 14.733 ms |
| 524,288 | 13.748 / 13.700 ms | 14.614 / 14.596 ms |
| 999,424 | 13.779 / 13.764 ms | 14.684 / 14.750 ms |

From 4K to 999K the fixed step grows 0.18 ms at 3 rows and 0.20 ms at 6 rows (1.3% and 1.4%).

### Cold prefill in the same boots

The `fill` requests reached 9,937 / 9,916 tok/s at 131,880 tokens, 9,901 / 9,829 at 262,534,
9,582 / 9,480 at 522,754 and 9,047 / 8,981 at 994,677.

### Kernel attribution (one boot, four ranks)

Real decode windows, mean of the four ranks (they agree within 1%, except the indexer sum
at 4K, 0.369 to 0.401 ms). A step here is the window's kernel span over its steps in the
trace. Every step launches 1,508 kernels.

| Prompt tokens | Steps | Target rows per step | Step | GPU idle | Indexer, summed | Indexer, owned | Indexer, alone |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4,070 | 49 | 4.56 | 29.59 ms | 0.15 ms | 0.38 ms (1.3%) | 0.00 ms | 0.00 ms |
| 266,422 | 59 | 4.29 | 30.24 ms | 0.12 ms | 1.40 ms (4.6%) | 0.63 ms (2.1%) | 0.44 ms (1.5%) |
| 1,001,009 | 61 | 4.57 | 32.64 ms | 0.11 ms | 3.48 ms (10.7%) | 2.33 ms (7.1%) | 1.82 ms (5.6%) |

The indexer's kernels, per step (launches x mean duration):

| Prompt tokens | Score | Top-k segment | Top-k merge | Quantize |
| ---: | ---: | ---: | ---: | ---: |
| 4,070 | 8 x 26.5 µs = 0.21 ms | 9 x 12.8 µs = 0.12 ms | 4 x 8.5 µs = 0.03 ms | 4 x 4.6 µs = 0.02 ms |
| 266,422 | 8 x 124.4 µs = 1.00 ms | 9 x 33.6 µs = 0.30 ms | 4 x 21.2 µs = 0.09 ms | 4 x 4.8 µs = 0.02 ms |
| 1,001,009 | 8 x 331.1 µs = 2.65 ms | 9 x 75.4 µs = 0.68 ms | 4 x 33.7 µs = 0.14 ms | 4 x 4.5 µs = 0.02 ms |

- The 4K and 1,001K windows have the same rows per step, and the step grows 3.05 ms between
  them. The indexer's summed kernel time grows 3.10 ms and its owned time 2.33 ms; the
  owned time of every other family together grows 0.73 ms (MoE 14.10 to 14.33 ms, prefetch
  5.78 to 5.94, collectives 3.33 to 3.39, the rest 6.25 to 6.53, on sp1).
- At 4K the indexer's 0.38 ms runs entirely under kernels that started earlier, so it owns
  no time of the step.
- The GPU is busy for all but 0.11 to 0.16 ms of a real step (0.4 to 0.5%).
- By the server's step counter the same windows give 29.89, 30.40 and 34.00 ms a step, a
  growth of 4.11 ms from 4K to 1,001K. The 1.06 ms difference from the trace is time
  outside the kernel span: 64 ms more of the stream at 1,001K than at 4K, nearly all after
  the last kernel. It is per stream, not per step, and no kernel accounts for it. The timing
  job's depth term contains it.

The engine grid's fixed step, same boot (124 steps a window, both shapes together):

| Context | Step | GPU idle | Launches per step | Indexer, summed | Score kernel | MoE, owned |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 15.76 ms | 2.53 ms | 1,701 | 0.22 ms | 8 x 17.5 µs | 0.08 ms |
| 262,144 | 15.53 ms | 2.53 ms | 1,701 | 0.22 ms | 8 x 17.6 µs | 0.08 ms |
| 999,424 | 15.81 ms | 2.52 ms | 1,701 | 0.22 ms | 8 x 17.6 µs | 0.08 ms |

The fixed step's score kernel takes 17.6 µs at every context length, where a real step's
takes 331 µs at 1,001K: the fixed step does not score the long context. Its MoE kernels own
0.08 ms against 14.1 ms in a real step (the grouped expert GEMM takes 5.3 µs a launch
against 176 µs), which is the difference between the fixed step and a real one. Why the
fixed step's expert GEMM and scoring do so little is not examined here.

## Conclusion

- A real decode step slows with context: about 4.1 ms more at 995K tokens than at 4K, which
  is 12% of that step, and 1.1 ms (3.6%) at 263K. The slope is linear in depth and does not
  depend on whether rows are in the fit.
- The kernel profile attributes it. Of that growth, about 3.1 ms is per step and it is the
  indexer: the step grows 3.05 ms from 4K to 1,001K at equal rows, and the indexer's kernel
  time grows 3.10 ms (2.33 ms of it owned). The other 1 ms of the timing job's depth term is
  a per-stream tail after the last kernel, which a longer answer would dilute.
- The indexer's share of a real decode step is 0.38 ms (1.3%, all of it hidden under other
  kernels) at 4K, 1.40 ms (4.6%) at 266K and 3.48 ms (10.7%) at 1,001K. Removing all of its
  depth-dependent work would save about 1.0 ms a step (3.4%) at 266K and 3.1 ms (9.5%) at
  1,001K. The score kernel is three quarters of it (2.65 ms at 1,001K) and top-k selection
  the rest (0.81 ms).
- A real step has no idle GPU time to recover (0.1 ms of 30) at 1,508 launches a step.
- The engine grid's fixed step does not show the cost. It grows 0.2 ms where real steps grow
  3 ms, because its score kernel takes the same 17.6 µs at every context length; its 64K,
  256K and 1016K shapes do not measure long-context decode.
- A real one-stream step takes 28.9 ms at 4K, twice the fixed step's 13.6 to 14.6 ms at 3
  to 6 rows. The difference is the MoE expert GEMM, which owns 14.1 ms of a real step and
  0.08 ms of the fixed step.

Not covered: a repeat boot of the kernel profile (queued as
[profile-again-tp4-switched.json](profile-again-tp4-switched.json)), why the fixed step's
expert GEMM and scoring do so little work, the per-stream tail at depth, other content than
Python source, more than one stream, and shapes other than `tp4-switched`.
