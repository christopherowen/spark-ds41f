# DSpark context graphs at the target's decode graph sizes

## Summary

`DSparkContextCudaGraphs` (the drafter's context projection and KV preparation, one FULL
graph per row capacity) captured powers of two plus the limit. A full-depth decode step
(1 + 5 rows a request) therefore padded 6 → 8, 24 → 32, and at TP4 36 → 64 and 66 → 96
rows through `main_proj` (its per-token cast, GEMM and TP all_gather), `main_norm` and the
three draft layers' KV projection and `context_kv_front`. This experiment ports the closed
#739 onto candidate-37: the context graphs capture the row counts of the target's own FULL
decode graphs instead. Those come from the compilation config's capture ladder and the token
totals of `dense_varlen_decode_shapes` (every one- and two-request total, every full-depth
step), up to the context limit.

**Result (one GB10, single-rank harness, four processes per shape):**
- **Faster, but only slightly.** With L2 flushed before each replay, as after a target
  forward, the mean over every row count is -0.76 ± 0.59 µs at TP3 and -1.98 ± 0.71 µs at
  TP4. The largest saving is -6.8 ± 2.4 µs at TP4 c11 (96 → 66 rows). Full-depth c1..cmax
  averages -1.0 ± 1.0 µs (TP3) and -2.4 ± 0.6 µs (TP4). Decode steps are about 25-66 ms
  (spark-ds41f-1030's switched benchmark runs,
  `manifests/benchmarks/2026-10-30-spark-ds41f-1030-switched*.json`), so the saving is at
  most about 0.01% of a step: no end-to-end bench can resolve it.
- **Outputs are unchanged.** The context buffer and all three SWA caches are bit-identical
  between the ladders at every row count 1..limit, in all 8 runs.
- **No new kernels.** The TileLang cache was frozen after warming only the old ladder's
  sizes; every new size captured and replayed without compiling.
- **Costs:** 23 graphs instead of 7 at TP3 and 35 instead of 8 at TP4. Capture takes
  +6 ms (TP3) and +10 ms (TP4). Allocator memory is the same for both ladders; driver
  memory outside the allocator grows by about 5-8 MiB per rank.
- **Not measured:** the TP all_gather that `main_proj` runs over RoCE. It moves 10 KiB per
  row, so the ladder sends 20-300 KiB less per step where the capacity drops (table below).
  That needs a cluster window.

Decision: **retain for more evidence**. The patch is tested and harmless, but the
GPU-side gain is far below what promotion can measure. It is worth carrying only together
with a cluster measurement of the all_gather bytes, or as a low-risk rider in a candidate
that is being qualified anyway. Promotion is the owner's call.

## Base and the one variable

- Repository: `15cba190` (main). Candidate: candidate-37, vLLM
  `local-inference-lab/vllm@19f2c20e` plus
  `experiments/2026-10-26-spark-ds41f-candidate/candidate-37/vllm/series` (patch head
  `5b7cfa37`, tree `af0fb152`). A fresh tree on dgx4 reproduced both
  (`commands/prepare-candidate-37.sh`).
- Image: `spark-ds41f:candidate-37` on dgx4,
  `sha256:c4c59de168e665a9954d16a3543a1dc045d6ac60b722d094c00a49f202a81f47`
  (label `local.spark-ds41f.vllm.patch-head` = `5b7cfa37`). The patched `dspark.py` and
  test file are bind-mounted over the image's tree; this is a Python-only change.
- Variable: the capacities `DSparkContextCudaGraphs` captures. Nothing else differs. Both
  arms run the patched class; the `base` arm swaps `_context_capture_sizes` for
  candidate-37's powers of two (`bench_context_ladder.py: powers_of_two`).

| Shape | Limit | candidate-37 (base) | This patch (ladder) |
|---|---:|---|---|
| tp3-triangle (8 seqs) | 48 | 1, 2, 4, 8, 16, 32, 48 (7) | 1-12, 16, 18, 20, 24, 28, 30, 32, 36, 40, 42, 48 (23) |
| tp4-ring (16 seqs) | 96 | 1, 2, 4, 8, 16, 32, 64, 96 (8) | 1-12, 16, 18, 20, 24, 28, 30, 32, 36, 40, 42, 48, 54, 56, 60, 64, 66, 72, 78, 80, 84, 88, 90, 96 (35) |

## The patch

[`0677-perf-dspark-capture-context-graphs-at-the-target-s-d.patch`](./0677-perf-dspark-capture-context-graphs-at-the-target-s-d.patch)
applies on candidate-37's patch head. With the build's fixed committer
(`commands/apply-series-and-patch.sh`) it gives head `f76ccab0`, tree `1ad072da`, and
`git diff --check` is clean against both the base and its parent. It was made on the
branch `perf/dspark-context-capture-ladder` of a disposable prepared tree on dgx4.

- `vllm/models/deepseek_v4_1/nvidia/dspark.py`: `_context_capture_sizes(vllm_config,
  limit)` returns the token totals of `dense_varlen_decode_shapes(max_num_seqs,
  1 + num_verification_tokens, limit)`, plus `cudagraph_capture_sizes` up to `limit`, plus
  `limit` itself. These are the target model runner's inputs for its FULL decode graphs (at
  the verification width, so copy drafts follow too), so the two cannot drift.
  `DSparkContextCudaGraphs.__init__` uses the function in place of its power-of-two loop.
- At both production configurations the sizes equal #739's. #739 kept powers of two and
  its own copy of the decode shapes, at `1 + num_speculative_tokens` rather than the
  target's width.
- Graph count is bounded by construction: never more than the target's own FULL decode
  graphs below the limit. Graph count and memory are measured below.
- `tests/kernels/test_deepseek_v41_tilelang_dspark_context.py` gains two tests:
  - `test_context_capture_sizes_are_the_target_decode_graph_sizes` (tp3-triangle,
    tp4-ring, a copy-draft width): the sizes equal the token counts of a target
    `CudaGraphManager` planned as the model runner plans it (`FULL_AND_PIECEWISE`, variable
    length decode, the recipe ladder), at the limit and below it.
  - `test_context_graphs_pad_decode_rows_no_further_than_the_target` (tp3, tp4): for every
    (requests, rows) a decode step can have, the real constructor's manager dispatches a
    context graph no wider than the target's FULL graph. Full-depth steps and every one-
    and two-request total dispatch exactly.

Tests in the candidate-37 image on dgx4 (`runs/logs/tests.log`): the context test file
passes 24/24 from the fresh tree, and again with `-p tests.kernels.deepseek_v41_pdl_sources`.
`tests/v1/cudagraph/test_cudagraph_manager.py` passes 26/26. The five new tests fail
against candidate-37's `dspark.py` (checked on a tree with only the test file changed).
The repository's unit suite (615 tests) and `bin/spark --shape … doctor` for all nine
shapes pass.

## Harness

`bench_context_ladder.py` builds one rank's context path at the shape's TP degree from the
real TileLang modules, with random weights:

- `main_proj`: this rank's padded column shard of the [5120, 15360] block-32 FP8 projection
  (TP3 1728 rows, TP4 1280) with its per-token MXFP8 cast. A device copy of the shard `tp`
  times side by side stands in for the all_gather: it has the gathered width but no RoCE.
  TP3 then slices back to 5120 columns.
- `main_norm`: TileLang RMSNorm, emitting main_x's MXFP8 cast.
- Three draft layers: the [512, 5120] KV rows of a fused Q-A/KV weight (split-K, reading
  main_x's cast) and `context_kv_front` into 1024 pages of 128 SWA records.
- The real `DSparkContextCudaGraphs`: its constructor, `capture()` and `run()`. `run()`
  includes the `context_inputs` launch.

In each process both arms are captured in one graph pool, in the order `--first` names.
Each process then:
1. compares the bits at every row count 1..limit;
2. times both arms at every row count, three rounds, alternating which arm goes first.

Two timing modes:
- **cold**: a 256 MiB buffer is zeroed before each replay to evict L2, and each replay is
  timed with its own events (median of 100). This is the serving case: the context step
  follows a target forward that streams the weights.
- **hot**: 300 back-to-back replays, mean. The weights partly stay in L2, so this is an
  upper bound of the compute saving.

The confidence interval is the 95% Student t over the four processes per shape. Each
process gives a paired ladder-minus-base delta (`analyze.py`, `summary.json`). Row counts
whose capacity is the same in both ladders are an A/A check.

Capture memory: each `--memory-copies 20` process captures 20 copies of one ladder in a
fresh process, sharing one pool as serving does. It reports device memory (`mem_get_info`;
host and GPU share it on GB10, so a single capture is lost in noise) and allocator memory
(`memory_reserved`, after `empty_cache`) per copy.

## Commands

On dgx4 (dev node; no hold applies, GPU idle before each run), under `/tmp/c37-ladder-739`:

```sh
bash prep.sh                                            # commands/prepare-candidate-37.sh
bash runimg.sh spark-ds41f:candidate-37 /tmp/c37-ladder-739/vllm -m pytest -q -rfEs \
  -p no:cacheprovider -p tblib_stub -p offline_skip \
  tests/kernels/test_deepseek_v41_tilelang_dspark_context.py   # commands/run-in-image.sh
bash matrix.sh                                          # commands/matrix.sh: 8 replay runs
bash memory.sh                                          # commands/memory.sh: 8 memory runs
bash fresh.sh patch/0677-*.patch                        # commands/apply-series-and-patch.sh
python3 analyze.py runs > summary.json                  # here
```

`runs/` keeps every run's JSON and log. `runs/smoke/smoke-tp3.json` is the first harness
check, with 20/5 iterations and one round. It was run before the capture-memory
measurement settled the allocator, which made its base capture read -82 MiB reserved. It
is not in the summary.

## Results

### Replay time per decode step (µs, ladder minus base, 95% CI over 4 processes)

Average over row counts:

| | TP3 cold | TP3 hot | TP4 cold | TP4 hot |
|---|---:|---:|---:|---:|
| every row count 1..limit | -0.76 ± 0.59 | -3.51 ± 2.47 | -1.98 ± 0.71 | -2.36 ± 4.03 |
| row counts whose capacity changes (32 of 48; 76 of 96) | -1.20 ± 0.61 | -4.98 ± 3.42 | -2.51 ± 0.75 | -3.17 ± 4.61 |
| unchanged capacity (A/A) | +0.12 ± 0.57 | -0.57 ± 0.74 | +0.03 ± 1.21 | +0.72 ± 2.59 |
| full depth, c1..cmax | -1.03 ± 0.98 | -4.11 ± 3.04 | -2.37 ± 0.57 | -3.27 ± 4.72 |
| rows 1-12 (one or two requests, any trim) | -0.30 ± 0.62 | -1.47 ± 1.24 | -0.51 ± 0.49 | -2.68 ± 5.06 |

Full-depth steps (6 rows a request). "Gather" is the all_gather bytes the ladder saves:
10 KiB per row, not measured here.

TP3 (tp3-triangle shape: TP3 shard, limit 48):

| c | rows | capacity | gather | base cold | Δ cold | base hot | Δ hot |
|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | 6 | 8 → 6 | 20 KiB | 234.9 | -0.08 ± 0.78 | 164.8 | -1.32 ± 1.58 |
| 2 | 12 | 16 → 12 | 40 KiB | 237.5 | -0.54 ± 1.45 | 168.0 | -1.98 ± 1.85 |
| 3 | 18 | 32 → 18 | 140 KiB | 245.1 | -3.85 ± 4.23 | 185.6 | -6.92 ± 7.33 |
| 4 | 24 | 32 → 24 | 80 KiB | 243.4 | -1.21 ± 0.58 | 182.3 | -6.51 ± 5.35 |
| 5 | 30 | 32 → 30 | 20 KiB | 244.0 | -0.41 ± 0.86 | 183.2 | -3.94 ± 3.86 |
| 6 | 36 | 48 → 36 | 120 KiB | 251.2 | -0.91 ± 2.70 | 195.1 | -7.44 ± 2.92 |
| 7 | 42 | 48 → 42 | 60 KiB | 252.1 | -1.40 ± 1.34 | 196.6 | -5.04 ± 3.27 |
| 8 | 48 | 48 → 48 | 0 | 252.4 | +0.15 ± 0.47 | 196.3 | +0.25 ± 2.89 |

TP4 (tp4-ring shape: TP4 shard, limit 96):

| c | rows | capacity | gather | base cold | Δ cold | base hot | Δ hot |
|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | 6 | 8 → 6 | 20 KiB | 206.0 | -0.30 ± 0.37 | 105.8 | -1.96 ± 5.58 |
| 2 | 12 | 16 → 12 | 40 KiB | 208.4 | -0.79 ± 0.99 | 113.6 | -2.74 ± 5.13 |
| 3 | 18 | 32 → 18 | 140 KiB | 212.9 | -1.85 ± 0.35 | 129.7 | -9.21 ± 5.85 |
| 4 | 24 | 32 → 24 | 80 KiB | 213.8 | -1.80 ± 2.74 | 133.2 | -5.13 ± 11.31 |
| 5 | 30 | 32 → 30 | 20 KiB | 213.5 | -0.57 ± 0.43 | 132.1 | +0.85 ± 8.19 |
| 6 | 36 | 64 → 36 | 280 KiB | 221.0 | -2.79 ± 1.72 | 154.1 | -8.73 ± 8.90 |
| 7 | 42 | 64 → 42 | 220 KiB | 221.9 | -1.94 ± 3.55 | 155.3 | -4.21 ± 7.39 |
| 8 | 48 | 64 → 48 | 160 KiB | 223.2 | -2.32 ± 1.19 | 156.2 | -0.67 ± 9.06 |
| 9 | 54 | 64 → 54 | 100 KiB | 222.9 | -1.33 ± 1.31 | 158.9 | +0.61 ± 8.49 |
| 10 | 60 | 64 → 60 | 40 KiB | 223.8 | -1.15 ± 1.83 | 158.2 | +4.61 ± 9.88 |
| 11 | 66 | 96 → 66 | 300 KiB | 264.2 | -6.79 ± 2.41 | 224.3 | -10.00 ± 2.43 |
| 12 | 72 | 96 → 72 | 240 KiB | 264.5 | -5.01 ± 2.02 | 227.6 | -10.79 ± 3.90 |
| 13 | 78 | 96 → 78 | 180 KiB | 266.8 | -5.64 ± 3.37 | 226.3 | -3.28 ± 10.71 |
| 14 | 84 | 96 → 84 | 120 KiB | 266.2 | -3.12 ± 0.57 | 229.1 | -4.30 ± 6.97 |
| 15 | 90 | 96 → 90 | 60 KiB | 266.9 | -1.90 ± 1.24 | 228.4 | -1.27 ± 6.58 |
| 16 | 96 | 96 → 96 | 0 | 267.7 | -0.54 ± 0.87 | 227.9 | +3.95 ± 10.71 |

Reading:
- The cold saving tracks the rows removed: about 0.08-0.23 µs per padded row. It is
  largest above 64 rows, where the base pads to 96 (c11-c13 at TP4).
- Hot mode at TP4 is noisy: its A/A row counts swing ±2.6 µs, and single row counts swing
  ±10 µs. Cold mode is tight (A/A within ±1.2 µs) and is the one to read.
- In serving, adaptive verification trims drafts, so many steps sit below 6c. All 1..limit
  row counts are in `summary.json` (`per_rows`).
- The context step adds the RoCE all_gather on top; this harness has only its device copy.

### Capture memory and time

Per copy of the ladder, from 20 copies per process, two processes per arm
(`summary.json`, `tp3-memory`, `tp4-memory`):

| Shape | Arm | Graphs | Allocator MiB | Device MiB beyond allocator | Capture (median) |
|---|---|---:|---:|---:|---:|
| TP3 | base | 7 | 2.0 / 2.0 | 0.69 / 0.69 | 5 ms |
| TP3 | ladder | 23 | 2.0 / 2.0 | 7.13 / 5.17 | 11 ms |
| TP4 | base | 8 | 3.0 / 3.0 | 0.44 / -0.31 | 5 ms |
| TP4 | ladder | 35 | 3.0 / 3.0 | 7.73 / 6.57 | 15 ms |

- The allocator figure is the context `aux` buffer (1.41 MiB at 48 rows, 2.81 MiB at 96,
  in 2 MiB pages), the same for both arms. Neither ladder's graph pool use shows above it.
- The device figure (graph executables and other driver allocations) grows by about
  4.5-6.4 MiB per rank at TP3 and 6.1-8.0 MiB at TP4, roughly 0.2-0.3 MiB per added graph.
- That memory is outside the KV pool (`kv_cache_memory_bytes` is fixed). It comes out of
  the headroom TP3's 5 GiB startup guard on dgx1 watches: about 0.1% of it.
- The first capture in a process takes 5-10 ms longer (warm-up). Without the 20× copies,
  the fresh-process capture-first runs read base 12 ms and ladder 17-21 ms
  (`summary.json`, `capture_first`).

## Quality

- Bits: the context buffer and every SWA cache page are identical between the ladders at
  every row count, in all 8 runs (`bit_mismatch_rows` empty). Padded rows keep PAD slots
  and write nothing, so the only change is how many padded rows each replay computes.
- Kernels: the TileLang cache was frozen (`prepare.freeze`) after an eager pass at the old
  ladder's sizes only. Capturing and replaying the new sizes compiled nothing: the GEMM,
  cast, norm and front kernels take a dynamic row count, and the row tiles are the same.
- The context graph is the production decode path: pure decode steps have no CED plan
  (`plan_ced` returns None when every request keeps its rows), and DS4.1 has no streamed
  context states.

## Not covered

- The RoCE all_gather of `main_proj` at TP3/TP4 (bytes above) and the end-to-end decode
  step, which needs a cluster window. At 25-66 ms a step, the GPU-side saving measured here
  is far below the bench's ±0.5 ms step resolution.
- Real weights: the harness uses random FP8 weights of the real shapes. Timing does not
  depend on the values.
