# IKET full forward pass: spec for a lab window

The aim is one Perfetto file per node of a whole DS4.1 step, with every TileLang
kernel's phases on the step's real timeline, plus a combined file of the four ranks. The
configurations are c1 decode, c8 decode and one prefill chunk.

The converter is `iket_perfetto.py`, and the README's "Perfetto: a decode step lane by
lane" describes its output. This spec says which runs make its inputs. The rule is to
slice only if we must.

## Runs, for each configuration

Every run serves the same deterministic request:
- temperature 0, a fixed prompt, a fixed batch (1 or 8 requests), a fresh prefix cache;
- the recipe's environment and sizing, except as below;
- all four ranks at once (or rank 0 alone first).

| # | run | what it gives | about |
|---|---|---|---|
| 1 | **plain:** IKET off, no profiler | each step's time (the engine's step timer): the reference for the base run's overhead | 5 min |
| 2 | **base:** `run-iket profile --warp-trace --postprocess none`, IKET mode off | every launch's warp lifetimes, ATen, NCCL and sparknet included: the timeline of record. No body patching, no tracker pass, one record slot a warp | 6 min |
| 3 | **lean IKET:** the mode on; `iket_profile.py --timeout 900 --retries 2 --max-pool-gb LIMIT` | every TileLang kernel's phases and ranges. A dry run (tracker) sizes the pool, then the measured run records | 15-18 min |
| 3' | **slices, only if 3 exits 3** (pool over the limit): the same run with `VLLM_DS41_IKET_SKIP` keeping one family of 2-4 | the same, a family a run: (a) attention (mHC, norms, rope, cache, indexer, sparse MLA, WO), (b) MoE and dense GEMMs, (c) head, sampler and inputs | 15 min each |

Per configuration that is about 26-30 minutes on the single-run path, and 30 more if
the run must be sliced. Each dry-run hang adds one timeout (15 minutes); it then
retries. The three configurations take about 1.5 hours, or 2.5 with slicing.

### Settings shared by runs 2, 3 and 3'

Runs 2, 3 and 3' must launch the same kernel sequence, or the stitch refuses:
- **Context: `max_model_len` 8192.** At 1M, the warmup and capture launch the indexer's
  top-k with up to a million warps, and run-iket's pool reserves a record for each (run
  45: 17 GB, then the host ran out in the exit decode). At c1 the 1M context changes the
  step by about 0.1 ms (runs 46 and 47).
- **The recipe's streams** (`max_num_seqs`) and environment, breakable CUDA graphs off.
- **Capture sizes: exactly those the plain run replays** (its graph sizes for the
  measured steps), so every run picks the same graphs and the warmup stays small.
- **`max_num_batched_tokens` 512** for decode configurations, and the chunk's size for
  the prefill one. `kv_cache_memory_bytes` is set, so vLLM skips its profiling run.
- **No engine-grid runs, and no warmup request.** Decode configurations generate 8
  tokens (about 4 decode steps).

### The lean run's settings (mode changes still to make)

These are lean and cheap:
- `VLLM_DS41_IKET_CTAS=1`: one traced CTA a launch;
- `VLLM_DS41_IKET_ITERS=1`: the middle iteration of each loop.

These need mode changes (laptop work, then one GPU to validate):
- **phase-boundary ranges only:** consecutive statements of one phase merge into one
  range. That brings a warp's timestamps from up to 80 down to about 16, and the pool
  with it;
- **1-2 traced warps a CTA:** less distortion. The pool stays the same, because run-iket
  reserves a record for every warp of an instrumented launch, traced or not;
- **payloads** (role, expert, tile; bytes per load phase from tile sizes at codegen
  time).

## How big a lean run is

These figures are from run 46's harness records (TP4 shapes, 6 target layers, the
drafter's 3, an 8K context), scaled to the model's 43 layers.

| harness, TP4 c1, run 46 | warps |
|---|---:|
| all instrumented launches | 24.6M |
| the measured window (prefill, 17 decode steps, 9 grid runs) | 4.4M |
| startup, warmup and capture | 20.2M (82%) |
| per-layer kernels | 89.5% |

- **Lean whole-model c1** (4 decode steps, no grid, a smaller warmup): roughly 40M warps.
  The dry run measures the real number before the measured run.
- **Pool:** about 3 GB at 16 timestamps a warp, about 13 GB at today's 80.
- **The binding limit is run-iket's exit decode,** which builds every warp's record on
  the host at about 2.3 KB a warp (run 45). 40M warps means about 90-100 GB. That only
  fits if the engine releases its memory before exit, as `iket_step.py` does.
  - At c1, expect 1-2 runs.
  - At c8, expect 2-3 runs: 3.6 times c1's warps in the window.
  - For a prefill chunk, expect 3-4 runs, or a chunk of 2,048 tokens.
- **Choosing `--max-pool-gb`:** the decode needs about 7 times the pool at 80 timestamps
  and about 30 times at 16. Set the limit so that this multiple fits the node's memory
  after the engine releases its own.

## Hangs

About one tracker (dry) run in five hangs in run-iket's `cuGraphLaunch` callback, with
CUDA graphs. Measured runs never have.
- `iket_profile.py --timeout 900 --retries 2` restarts a hung dry run, so a hang costs one
  timeout.
- A dry run that hangs three times fails the configuration. Report it and move on.

## Stitching (laptop)

1. Copy each rank's base and lean records to the laptop. The nodes convert nothing.
2. Pair launches by ordinal: a launch's place among the step's launches of one kernel,
   grid and block (`iket_perfetto.rebase`). A sequence that differs from the base run's
   refuses to stitch, except for listed, flagged mismatches.
3. Move each lean launch's CTAs, warps and phases into its base launch's span. Each
   launch keeps both `real_us` and `instrumented_us` and the scale. A launch is flagged
   when its overhead is above 50%.
4. Overlap between kernels (PDL, side streams) comes from the base run only.
5. Write one file per node, plus a combined file with the ranks aligned at the
   collectives.
   - Every all-reduce and all-gather is a rendezvous. Each rank's clock offset and drift
     are fitted from the collective boundaries, and the residual is stated.
   - Each rank's wait at every collective, and the late peer, show directly.
6. Report the base run's own overhead (run 2 against run 1, step by step) in the files'
   metadata and in the README.

## Files

Per configuration and rank: `<node>-<decode-c1|decode-c8|prefill-<chunk>>-<build>-<date>.pftrace`,
and `combined-<config>-<build>-<date>.pftrace`, in
`~/projects/commander/outputs/iket-perfetto/cluster/`.
