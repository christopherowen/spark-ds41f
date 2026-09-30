# Temperature-0 determinism (2026-09-29)

r5m made indexer selections exact, yet identical temperature-0 requests at one
stream still produce different outputs (three prompts, five runs each: 5 of 5
distinct). Every run's first generated token already has a different logprob,
so the prefill itself is not repeatable. Drift before the first changed token
is 0.03-0.27 nats; tokens flip where the top two logits are 0.0-0.5 apart.

## Suspects

- **Routed-MoE combine.** B12X's W4A8 phase 2 adds each route's FC2 output
  into the token row with `red.relaxed.gpu.global.add.noftz.bf16x2`. BF16
  atomics round after every add, and the order of arrival between CTAs
  changes run to run, so the sum does too. B12X has a deterministic mode
  (`B12X_DYNAMIC_DETERMINISTIC_OUTPUT=1`): phase 2 stores one row per
  (token, route) and a fixed-order top-k sum reduces them.
- **Dense split-K turbo** (`B12X_DENSE_SPLITK_TURBO=1`): atomics for M <= 6,
  i.e. decode and the one-row logits.

## Why the deterministic mode could not boot before

Planning dropped the request. The dynamic route-mode heuristic, the tuning
materialization check and both direct-routing capacity checks passed a literal
`False`, and the query kept an unset request as `None` while launches resolved
it from the environment. Small launches were planned for direct routing, and
the launch rejected them ("planned dynamic direct routing is unsupported for
this launch shape"). `0004-moe-deterministic-planning.patch` makes planning
resolve and honour the request; with the variable off, the query and compile
keys are unchanged. B12X preparation planner tests: 157/157, including seven
new ones for DS4.1's TP3 geometry.

## Arms

`make_arms.py` writes both from `config/cluster.json` (r5m), with the three
patched planning modules mounted over the image (`overlay.sh`):

| Arm | Change |
|---|---|
| `det` | `B12X_DYNAMIC_DETERMINISTIC_OUTPUT=1` |
| `detsk` | also `B12X_DENSE_SPLITK_TURBO=0` |

`run.sh` boots each arm and r5m once: `determinism.py` (distinct outputs, first
changed token, first logprob difference), one round of prefill chunk timings,
and a lean decode screen.

## Round 1 (`run.sh`)

Determinism probe, five identical requests per prompt:

| Arm | Distinct outputs | First logprob difference |
|---|---|---|
| r5m | 5/5, 5/5, 5/5 | token 0 in every run |
| det | 5/5, 5/5, 5/5 | token 1 (prose, code), 3 (JSON) |
| detsk | 1/5, 1/5, 1/5 | none, except one code run at token 227 (same token) |

So the routed-MoE combine made prefill irreproducible and split-K turbo made
decode irreproducible; with both in a fixed order, outputs repeat.

Cost against r5m in the same session:

| Arm | Prefill chunk, 8K-200K | Decode step at one stream (prose, JSON) | Eight streams (prose, JSON) |
|---|---|---|---|
| r5m | 1032-1158 ms | 41.6, 48.0 ms | 171, 238 tok/s |
| det | +0.2 to +0.6% | 46.4, 54.8 ms | 145, 210 tok/s |
| detsk | 0.0 to +0.4% | 45.7, 54.4 ms | 143, 212 tok/s |

The decode loss is not the reduction: B12X's decode-regime predicate rejected
deterministic output, so decode fell back to grouped routing without the
shared-input decode kernels. Split-K turbo off added nothing measurable.

## Round 2 (`run2.sh`)

`0005-moe-deterministic-decode.patch` admits deterministic output in the W4A8
decode regime: the front-end already records pair indices, phase 1 recovers
the token for the shared input and phase 2 stores each route once, so
deterministic launches use the same kernels as atomic ones plus the
fixed-order top-k sum. Arms `detfast` (0004+0005, split-K through the FP32
reducer) and `detfast-t` (split-K turbo kept); decode alternates with r5m.

- detfast repeats exactly (5/5 identical, no logprob difference anywhere) and
  its tokens and logprobs equal detsk's bit for bit, in a separate boot: the
  decode-regime kernels with fixed-order routes compute exactly what the
  grouped path does. detfast-t varies from the first decode token, so four-way
  split-K turbo is the decode source.
- Prefill: detfast within +/-0.15% of r5m at every depth.
- Decode did not recover: detfast -9% (prose) to -17% (JSON) at one stream and
  -15% at eight against r5m (two boots each); detfast-t the same.

## Round 3: decode profile (`run3.sh`)

One single-stream JSON request each on r5m and detfast (rank 0):

| Kernel | r5m | detfast |
|---|---|---|
| Fused dynamic MoE (target, 96 CTAs), mean per call | 566 us | 619 us |
| Its p10 / p90 | 334 / 734 us | 402 / 890 us |
| Drafter MoE (90 CTAs), median | 244 us | 383 us |
| Fixed-order top-k sum, per call | - | 2 us |

The reduction is not the cost. With deterministic output the fused kernel
collapses every intermediate slice of an M tile into one task
(`dynamic.py`, `task_slice_chunk = route_gate_tile_cnt`) and accumulates the
slices into the route row by read-modify-write. The atomic path also splits
FC2's reduction over the intermediate dimension: one task per 128-column
slice, six for DS4.1's 768, each adding its partial atomically. At decode
(36 routed rows at one stream) the collapse leaves most of the 96 CTAs idle;
prefill has M tiles to spare, so it pays nothing.

Next: keep the slice split and make it deterministic. Each slice task stores
its FC2 partial into its own row (pair x slices + slice) and the fixed-order
top-k sum reduces routes and slices together, for launches whose partial rows
fit the planned route-output capacity (every decode shape); larger launches
keep the collapsed form.

## Round 4: slice partials (`run4.sh`)

`0006-moe-deterministic-slice-partials.patch`: for W4A8 SiLU launches without
a materialized intermediate (every DS4.1 decode batch: fused M16), the
deterministic kernel keeps the atomic path's one task per N128 intermediate
slice and stores each slice's FC2 partial in its own route-output row
(pair x slices + slice); the fixed-order top-k sum reduces num_topk x slices
rows per token (36 for DS4.1: six routes, six slices). Materialized launches
(split M64 prefill, M=1, fused phase B) and launches whose partials would pass
8192 rows keep one row per pair. B12X planner tests 165/165. `run4.sh` runs
the GPU tests with the cluster stopped, then screens arm `detslice` (0004-0006,
split-K through the FP32 reducer) against r5m.

- Decode mostly recovered. One stream: 47.8 ms per JSON step and 42.4 ms per
  prose step against r5m's 47.4-48.0 and 41.3-41.7. Eight streams, against
  the r5m boot between the two detslice boots: prose 164.6 and 164.0 against
  170.4 tok/s (-3.6%), JSON 225.8 and 225.8 against 244.9 (-7.8%). Prefill
  within +/-0.15%.
- The combine is no longer free. The top-k sum now reads 36 rows per token:
  7.6 us per call against 2.1 us collapsed in the one-stream traces, about
  0.3 ms per step over 40 layers; at 48 verification rows the route-output
  scratch is 16.9 MiB instead of 2.8 MiB. There is no eight-stream profile,
  so the rest of the eight-stream loss is not attributed.
- Still not deterministic: 5/5 distinct on all three prompts with split-K
  turbo off. With the side-stream shared-expert overlap off
  (`VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=0`, arm `detslice-noovl`) it is
  1/5 with identical logprobs, so the remaining source runs beside the
  routed MoE.

## Round 5: bisecting the overlap (`run5.sh`-`run13.sh`)

`debug-moe-checksum-log.diff` adds a debug-only device log
(`SPARK3_MOE_CHECKSUM_DIR`): per-row sums of every MoE layer's input, shared
and routed outputs, and inside the shared-expert MLP of its input, gate_up,
activation and down projection. Two identical requests with dumps between
(`run7.sh`, `analyze_requests.py`) first diverge at layer 3's shared expert
in the first decode step: input, gate_up and activation equal on every row,
the down projection different on every row; the all-reduce carries it to the
other ranks.

Ruled out, each still nondeterministic in serving or exact in isolation:
private scratch for the shared expert (`detslice-priv`); fixed per-layer
intermediates instead of caching-allocator tensors (`detslice-static`);
synchronous Engram (`detslice-noeng`); no L2 weight prefetch
(`detslice-nol2`); the block-FP8 linear alone, beside a matmul and under graph
replay (`gemm_concurrency_check.py`), with poisoned scratch
(`linear_poison_check.py`) and under compute-sanitizer initcheck/racecheck;
the routed MoE beside the shared chain on another stream
(`overlap_repro.py`, 30/30 exact).

## Round 6: what the down projection saw (`detslice-probe`)

`SPARK3_SHARED_RECOMPUTE=1` checksums the down projection's packed weights,
scales and unit alpha, recomputes the GEMM right away into a second buffer,
and re-reads its input and first output (`analyze_down_state.py`,
`analyze_recompute.py`). Weights, scales and alpha never change; input and
output are unchanged afterwards; but the recompute disagrees with the first
result in 6-7% of six-row decode calls on every rank (never at one to five
rows), by row sums up to 26.

The first mismatch per layer is then kept bit-exact (device-side latch): the
input, the scratch after each quantization and both outputs, plus the weights
of a few layers (`analyze_capture*.py`).

- Both scratch snapshots, the quantized input included, are identical.
- The first result is wrong on whole columns of a few 64-column tiles, all
  rows, most often exactly 32 of 64 (one warp's slice), by small amounts;
  the recompute matches a float64 reference from the decoded operands to
  1e-6.
- Exact-operand fits name the wrong operand: the B fragment of a k tile's
  last 32-wide sub-block taken from the same columns four k tiles later
  (sub-block 3 from 19, 7 from 23; residual 0.010-0.030). With four stages
  over six k tiles, k tiles 4 and 5 are the refills of the stages holding
  k tiles 0 and 1.

## Round 7: cause and fix (`run16.sh`, `run17.sh`)

The dense GEMM's MMA warps read each mainloop stage through the generic proxy
(ldmatrix, scale-factor copies) and release it right after issuing the last
k block's copies. The refill that the release permits is a TMA write through
the async proxy, which the release's ordering does not cover. Alone, the
copies always finish first; when another kernel's CTAs share the SM and delay
shared-memory traffic, a copy still in flight reads the refill. compute-
sanitizer's racecheck does not track TMA writes, which is why it stayed clean.

`gemm_race_stress.py` reproduces it in a minute with the cluster stopped: the
serving-shaped down projection on a side stream is wrong in 20 of 12000 calls
at six rows and 1 of 12000 at 48 rows beside the deterministic routed MoE,
never alone or beside a bandwidth-bound copy. Serving hit it far more often
(the shared expert is launched right as the routed MoE fills the SMs).
`0007-gemm-fence-stage-reads-before-tma-refill.patch` fences the async proxy
before each stage release on the TMA load path: 0 of 12000 at both sizes.

This is not specific to the deterministic mode: production r5m runs the same
GEMM beside its routed MoE, so its shared-expert outputs are sometimes wrong
too, invisibly among the atomic-combine noise.

## Round 8: 0007 in serving (`run18.sh`)

Arms `detslice-fence` (0004-0007) and `r5m-fence` (r5m plus 0007), overlap on;
`overlay_fence.sh` builds the fenced GEMM from the r5m tree.

- `detslice-fence` repeats exactly: 1/5 distinct on prose, code and JSON,
  zero logprob drift over 256, 256 and 135 tokens.
- Lean decode screen, one boot each, three samples (the r5m arms vary more:
  their atomic combine changes acceptance run to run):

  | Arm | prose c1 step | JSON c1 step | prose c8 tok/s | JSON c8 tok/s |
  |---|---|---|---|---|
  | r5m (control) | 42.84 ms ±1.4% | 47.92 ms ±3.9% | 163.4 ±3.5% | 236.7 ±7.3% |
  | `r5m-fence` | 41.52 ms ±1.1% | 48.38 ms ±2.6% | 166.0 ±6.6% | 233.7 ±8.4% |
  | `detslice-fence` | 42.71 ms ±0.2% | 47.68 ms ±0.2% | 163.8 ±4.2% | 225.1 ±0.5% |

- 0007 costs nothing measurable: step times move within noise, one each way.
- Deterministic decode now matches r5m at one stream. At eight streams prose
  is level and JSON is 4.9% below this control (7.8% below round 4's), the
  0006 combine cost that remains.

Next: 0007 belongs in the production series on its own (it fixes r5m's
shared-expert outputs, not only determinism). For the deterministic mode,
cut the combine cost before promoting it: a fixed-order in-kernel reduction
of the six slices into one route row (back to six rows per token and the
2.8 MiB scratch), or a masked top-k sum that skips dead routes.

## Round 9: masked top-k sum (`0008`, `run20.sh`)

Per owner direction, 0007 shipped alone as r5n; 0004-0006 stay experimental.
`0008-moe-masked-slice-topk-sum.patch` (on 0006, B12X tree `c28e70fd`): the
top-k sum reads each (token, route) expert id once and skips the slice rows of
routes to no expert, eight columns a thread, bitwise equal to the previous
order; the fused kernel no longer clears those rows. Kernel tests 21/21
(dead-route rows left NaN-poisoned, the real reducer checked bitwise), planner
tests 165/165 (`run22.sh`; run20's 86 failures came from exporting
`B12X_DENSE_SPLITK_TURBO=0`, which changes pinned code-generation snapshots).

`moe_combine_bench.py` at fixed shapes (uniformly random routing, so expert
weight traffic dominates; `combine_table.py`): per call at 48 rows the slice
combine costs +0.5/+1.2/+2.5% over the atomic one at 0/25/50% dead rows, the
masked one +0.4/+0.2/-0.2%; its top-k sum falls from 74 to 53 and 34 us as
rows die, where the unmasked one stays at 70-80 us. Nsight Compute (GB10 has no
`dram__` counters; sysmem fills and misses stand in): memory reads are the same
for all three combines; slice partials add 3-20 MiB of writes per call and the
top-k sum reads 2-17 MiB, almost all L2 misses.

## Round 10: against r5n on one pinned cost table (`run21.sh`)

`make_arms_r5n.py`: `r5n-pin-prof` and `detm-pin-prof` (0004-0006 and 0008,
split-K through the FP32 reducer) share `SPARK3_DSPARK_COST_DIR`; the first r5n
boot wrote `dspark-costs-e9eb8edaf99252b3.json` (SHA-256 `3a45eec6…`), every
later boot reused it. Pinning holds the verification policy, not the realized
work: the deterministic arm decodes other trajectories.

| two boots each (`pin_table.py`) | r5n | deterministic | change |
|---|---|---|---|
| prose, one stream: step | 41.67, 42.21 ms | 42.90, 42.90 ms | +2.3% |
| JSON, one stream: step | 47.85, 48.90 ms | 47.45, 47.41 ms | -1.9% |
| prose, eight streams | 162.8, 162.3 tok/s | 167.8, 168.5 tok/s | +3.5% |
| JSON, eight streams | 240.5, 239.5 tok/s | 219.4, 222.1 tok/s | -8.1% |

Eight-stream JSON verified 3.73 against 3.69 drafts per draft and accepted
2.89 against 2.97, so throughput still mixes trajectory with kernel cost.
The profiled window (`profile_c8.py`: the same eight JSON requests on both,
536 against 527 draft events, 1727 against 1738 verified and 1303 against 1316
accepted drafts) took 10.84 s against 10.61 s (+2.2%). Its traces
(`analyze_steps.py`, `moe_grids.py`): the target MoE kernel takes 1596 us per
call against 1451 (+10%), the top-k sum about 34 us per call more; the rest
of the step is level. That is the measured bottleneck: the fused kernel with
slice partials at real routing, not the reduction. The fixed-shape bench
(round 9) shows only 1.4-2.8% for the same kernel, so the serving penalty
likely comes from real routing (more expert overlap, so partial traffic is a
larger share) and from partial writes evicting L2-prefetched weights; neither
is measured yet.

Repeatability: identical sequential requests repeat exactly with 0008 (1/5 on
prose, code and JSON). Across batch compositions they do not
(`determinism_concurrent.py`): the same prompt beside 0, 2, 4 and 7 other
requests gave 4 distinct outputs of 5 for JSON and prose, first differences
at tokens 28-57 and 8; the two seven-request mixes agreed for JSON. The mode is
deterministic for a fixed batch composition, not batch-invariant.

## Round 11: the fence at kernel level (`run22.sh`)

`gemm_fence_timing.py`, identical inputs, r5m image (shipped) against r5n
(fenced), median kernel time: down projection 6.30/6.37 us at 6 rows and
8.42/8.13 us at 48, gate_up 29.41/29.70 and 12.10/12.32 us, alone and beside
the routed MoE alike (-3.8% to +1.8%, both directions): no consistent cost,
about +/-0.02 ms per step for the shared expert.

Next: before any cross-CTA reduction design, measure what makes the fused
kernel 10% slower in serving: the same comparison with the L2 weight prefetch
off, and the fixed-shape bench with routing captured from serving. Batch
invariance would need composition-independent kernel choices (split-K,
tiles, attention splits); size that before proposing the mode.

## Round 12: where batch composition enters (`run23.sh`-`run25.sh`, r5o)

Arm `detm-r5o-trace`: r5o, the deterministic MoE (0004-0006, 0008), the r5o
screen's pinned cost table, and the batch-trace debug overlay
(`debug-batch-trace.diff`): per-row sums of every MoE layer and the shared
expert for up to 256 rows, plus a per-step log of the scheduled batch (row
offsets after draft reallocation, dead rows, request ids). `trace_mixes.py`
sent a tagged target (JSON and prose) beside five background mixes, three
times each; `analyze_trace.py` compares the target's own rows.

- Identical target schedules gave identical outputs in every pair (JSON: mix 0,
  2 and 3 repeats; prose: mix 0 and several others). Repeats of a mix diverged
  only when their schedules did (for example one verify step padded to 16 rows
  against 12).
- JSON, against the target alone: every mix first differs in the target's own
  prefill (same 19 prompt tokens; 19 rows alone, 28-40 with neighbours), at
  layer 0's routed-expert output. The MoE input and the shared expert's output
  match.
- Prose: the prefill first differs at layer 3's MoE input on one row, so
  upstream of the MoE (attention, indexer or mHC), not yet located.
- The drafter's first layer sometimes differs with identical schedules; it
  did not change verified tokens here but can change later schedules.

Replays: the deterministic routed MoE is batch-invariant (`moe_batch_invariance.py`:
19 fixed rows with fixed routing, identical in batches of 20-48, any position,
neighbours sharing or avoiding their experts); the router gate GEMV and the
block-FP8 shared-expert linears give identical bits at every prepared capacity
from 19 to 48 (`gemv_batch_invariance.py`). The cause is the plan lookup: the
V4.1 GEMV linears (router gate among them) take the plan for the exact row
count and fall back to the maximum-capacity plan (4096 rows) otherwise.
Capture sizes skip 19, so the unpadded 19-row solo prefill ran the 4096-row
plan, while mixed batches padded to 28-40 ran small plans: the gate's logits
for the same 19 rows differ (all rows, up to 2.4e-7), and with them the routing
weights. The block-FP8 linears pick the smallest prepared capacity at or above
the row count instead, and stay invariant.

Next: the same lookup for the GEMV linears (smallest capacity at or above the
rows), retrace to see where the target first differs then, and locate the
attention-side difference.

## Round 13: smallest prepared capacity (`run26`-`run31`, r5o)

Two independently reviewable changes, each kept apart from the debug
instrumentation:

- `vllm-0027-gemv-smallest-capacity.patch` (vLLM, on the r5o series head
  `0a682781`): the V4.1 BF16 GEMV projections (router gate, index head weights
  and key, compressor) take the smallest prepared capacity at or above the row
  count instead of the exact count or else the largest (4096-row) plan.
  Overlay `gemv-lookup`. Tests: a new host test of the lookup for both
  `B12xLinearMethod` and the compressor (fails on the old code); the existing
  GPU tests of both pass (3/3, 2/2). `git diff --check` clean.
- `b12x-0006-moe-smallest-variant.patch` (B12X, on r5o's `bb40849f`): a live
  token count binds to the smallest planned fused-MoE variant instead of the
  prefill capacity. Overlays `moe-variant` (production `_preparation.py`) and
  `det-variant` (the det-masked files). B12X's host-only variant-selection
  tests: 26/26 with the two updated tests (the old code fails both).

Debug only, behind `SPARK3_MOE_CHECKSUM_DIR` and `SPARK3_GATE_CAPTURE_ROWS`:
`debug-attn-trace.diff` (overlay `attn-trace`) adds per-row checksums of the
attention (input, q and kv latents, the query projection, rotated q, the
compressor latent and index key on emitted rows, index head weights, the
selected positions and their order, lengths, output, projected output) and of
the router logits, captures the first router gate call after each reset on
rank 0 (input rows, FP32 logits, served weights), dumps each native linear's
selected configuration per capacity, and fixes the watcher's reset
(`torch.inference_mode().__enter__()` on a temporary released the guard at
once, so every reset had failed and round 12's logs were rings). All arms pin
the r5o screen's cost table. The trace arms' KV caches are 256 and 768 MiB
smaller than r5o's: their first boots stopped 57 and 21 MB short of dgx1's
5 GiB startup memory guard.

### The GEMV lookup, validated on captured inputs (`gemv_lookup_validation.py`)

The attn-trace arm captured, on rank 0, every layer's first gate call after a
reset during the solo JSON (19 rows) and prose (15 rows) prefills, with the
served weights. Serving prepares 35 capacities, not just the 14 graph sizes (1-8,
10, 12, 14, 15, 16, 20, 21, 24, 25, 28, 30, 32, 35, 40, 42, 48, 49, 56, 72, 96,
192, 384 ... 4091, 4096), so 15 was already exact and 19 was not.

- Fidelity: all 86 captured calls (43 gates, both prompts) reproduce serving's
  logits bitwise through the plan the new lookup picks, and every row equals
  itself computed alone. The old lookup would have run the JSON prefill on the
  4096-row plan (max |d| 9.5e-7).
- Router gate and ratio-2 compressor (FP32 out): SIMT at capacities up to 192,
  the TMA prefill kernel from 384. With the new lookup every target row equals
  the row alone at every batch of 1-192 rows, target at the start, middle or
  end; batches of 255 and more (384-row plan) differ. The old lookup differed
  at every unprepared count from 9 rows on.
- Index head weights (32 x 5120) and the ratio-1 compressor (BF16 out): SIMT up
  to 8 rows, then the torch backend (cuBLAS picks its kernel by M). The index
  head weights differ from the row alone at 9-16 and from 128 rows with either
  lookup (max |d| 0.031): the lookup cannot fix a backend that is row-count
  dependent by itself. Index key and drafter gate: invariant everywhere.
- wq_b (block FP8, rank 0's 1280 -> 11264): invariant at every batch and
  position; split-K 1 at every capacity (tiles 16/32/64).
- Kernel time per call (CUDA graph): unchanged except where the old lookup ran
  the big plan (router gate at 19 rows 35.9 -> 28.2 us, ratio-1 compressor
  35.9 -> 15.0 us) and at 33 rows, where the 35-row SIMT plan is slower than
  the prefill kernel (gate 36.0 -> 42.2 us, ratio-2 compressor 35.8 -> 54.5 us).

### Trace 1: the lookup alone (`detm-r5o-lookup-trace`)

JSON and prose targets, five background mixes, three repeats each; the target's
schedule is recorded per step (rows, padded batch, requests).

- Identical target schedules gave identical target outputs and identical
  main-model records in every pair (JSON mix 0 x3, mix 3 r0/r1; prose mix 0 x3,
  mix 2 r0/r2).
- JSON, solo against every mix: the router logits now match on all 19 prefill
  rows, and so do the MoE input and shared output, but layer 0's routed output
  still differs. Prose, solo against every mix: the first difference moves from
  round 12's layer-3 MoE input to layer 0's block-FP8 query projection (wq_b),
  on 4 of the 15 rows.
- Repeats of a mix whose schedules differ first differ at wq_b (16 against 12
  rows), at layer 2's index head weights (16 against 20 rows; the torch
  backend), at layer 0's routed output (24 against 27, 28 against 39 rows), or
  only after the drafter changed the drafts and with them the schedule.
- The drafter's layer 40 differs even at identical schedules (attention
  output with matching attention input); it changes drafts, not verified
  tokens, but then the schedule.

### The MoE through serving's one-plan path (`moe_serving_replay.py`)

Serving plans the routed MoE once (the fixed graph sizes plus the 4096-token
limit as warm counts) and binds every live batch to it; B12X's `variant_for`
took the exact planned variant, else the prefill capacity. Replayed through
that path (vLLM's own call factory, DS4.1 TP3 shapes, synthetic weights,
deterministic mode), with batch sizes grouped by the exact bits of 19 target
rows (every fixed batch repeats bitwise):

| Warm counts | Before (det-masked) | With b12x-0006 (det-variant) |
|---|---|---|
| graph sizes | 20, 24, 28, 32, 40, 48 in one group; every other size 19-512 (the solo call among them) in another | 19-48 in one group; 49-512 (prefill capacity) in another |
| every prepared count | prepared sizes 20-192 in one group; unprepared sizes and 256, 512 in a second; 384 in a third | 19-192 in one group; 256-512 in another |

That is round 12's JSON divergence with the gate fixed: a 19-token prefill
alone ran the prefill-capacity launch, the same rows in a 28-48-row batch
ran a planned variant. With the fix, the first R target rows alone also equal
the solo call's first R rows for R = 2-18; a single row alone takes the M=1
launch and still differs.

### Trace 2: both fixes (`detm-r5o-lookup-variant-trace`)

- JSON, solo against every mix: layer 0 now matches entirely, attention and MoE
  (routed output included). The first difference is layer 1's attention input
  on the same 4 of 19 rows (7, 8, 10, 17) whatever the mix's padding (28, 32,
  40, 48) or the target's offset. Layer 1 is an Engram layer: its hashed rows
  go through a block-FP8 projection (`wkv`) before the mix.
- Prose, solo against every mix: still layer 0's wq_b, on the same 4 of 15
  rows (0, 3, 7, 10) at every padding (24, 32, 48). Serving's dumped
  configurations are identical for 16 and 24-48 rows (tile 32 x 128, split-K
  1), matching the replay, which finds wq_b batch-invariant. So the remaining
  first divergences are block-FP8 linears that behave differently in serving
  than in isolation.
- Mix 1's prose repeats now match entirely (round 12 and trace 1 split at the
  MoE). Other repeats first differ at wq_b (16 against 12 rows), at the index
  head weights (16 against 20), or after the drafter changed the schedule.
- Identical schedules still give identical outputs.

### Performance (one boot per arm, pinned cost table, no debug overlay)

`run30.sh`, `tables_lookup.py`; cost table `dspark-costs-e9eb8edaf99252b3.json`,
SHA-256 `5dc8961a`, loaded at every boot. Decode: tokens per second (95%
interval over six samples), GPU step time, verified and accepted drafts per
draft event.

| Point | r5o | + GEMV lookup | + lookup + MoE variant | det. MoE | det. + both |
|---|---|---|---|---|---|
| prose c1 tps | 50.88 ±4.3% | 51.82 ±4.4% | 51.05 ±5.9% | 51.67 ±0.5% | 48.19 ±0.3% |
| step ms | 42.46 | 42.40 | 42.58 | 43.16 | 42.63 |
| verified / accepted | 3.248 / 1.249 | 3.261 / 1.290 | 3.249 / 1.261 | 3.336 / 1.327 | 3.215 / 1.132 |
| JSON c1 tps | 76.92 ±5.1% | 77.06 ±2.4% | 77.16 ±2.5% | 73.96 ±0.3% | 77.31 ±0.6% |
| step ms | 48.35 | 48.54 | 48.16 | 48.15 | 49.73 |
| verified / accepted | 4.521 / 2.929 | 4.522 / 2.959 | 4.502 / 2.934 | 4.391 / 2.754 | 4.587 / 3.079 |
| prose c8 tps | 160.52 ±1.5% | 160.57 ±2.0% | 162.06 ±1.1% | 164.26 ±0.3% | 153.19 ±0.2% |
| verified / accepted | 2.408 / 1.224 | 2.436 / 1.239 | 2.411 / 1.226 | 2.226 / 1.120 | 2.332 / 1.130 |
| JSON c8 tps | 231.01 ±3.0% | 235.83 ±2.1% | 236.43 ±4.5% | 226.71 ±0.1% | 228.48 ±1.8% |
| verified / accepted | 3.520 / 2.871 | 3.513 / 2.881 | 3.550 / 2.937 | 3.549 / 2.934 | 3.501 / 2.873 |

- Production path: neither change moves decode beyond its interval (lookup
  -0.0 to +2.1%, lookup and variant +0.3 to +2.3%), with verification work
  and step time matched (verified per draft within 1.2%, step within 0.4 ms).
  Graph-sized decode batches were exact matches before and after.
- Deterministic path: prose -6.7% at one and eight streams, JSON +4.5% and
  +0.8%. Accepted drafts move the same way (prose c1 1.327 -> 1.132, JSON c1
  2.754 -> 3.079): the fixes change the arm's numerics, so it writes different
  text with different acceptance; step time moves -0.5 ms (prose c1) and
  +1.6 ms (JSON c1). Its intervals are tight because its repeats are identical,
  which understates content variation.
- Short prompts (`ttft_short.py`, request time of a 7-68-token prompt with one
  generated token, median of seven): the lookup is 1-5 ms faster than r5o
  (17-30 tokens 162 -> 157 ms); with the MoE variant too, 3-7 ms slower than
  r5o, including at lengths the variant change cannot touch. With one boot per
  arm these few-millisecond differences include boot-to-boot variation and are
  not established.

### Probe: the block-FP8 query projection in serving (`run31`)

`detm-r5o-lookup-variant-probe` recomputes the first four layers' query
projection beside the served call (`debug-attn-probe.diff`, batches up to 256
rows; the first boot failed because the 4096-row profile run asked for 4097).
Over 20 traced runs (JSON and prose, five mixes, two repeats), all three ranks:
the served wq_b equals the same call repeated, the same rows with a private
scratch, and the rows moved one position down in a batch one row larger, in
all 15,929 layer-steps (`analyze_probe.py`). In serving, wq_b is repeatable
and independent of the shared workspace, of position and of batch size.

Yet the prose target still first differs at q_proj on rows 0, 3, 7 and 10
between its solo prefill and every mix, with the q latent before it and the
attention input matching. The input must differ: the trace compares per-row
sums, and a sum hides flipped low bits that cancel. Every "first difference"
in this round is therefore an upper bound on where the divergence starts (a
differing sum is a real difference; an equal one is not proof of equality).
The MoE and GEMV mechanisms stand on their replays, not on the sums.

Next: an exact per-row fingerprint (the bit patterns, position-weighted,
modulo a prime) in place of the sums, to find where the prose and Engram
divergences really begin; batch-invariant replacements for the torch-backend
GEMVs (index head weights, ratio-1 compressor); and the drafter's
fixed-schedule difference.


## Round 14: exact fingerprints and batch invariance (`run32`-`run44`, r5o)

Round 13's first differences rested on per-row sums, which fail both ways:
flipped low bits can cancel, and `torch.sum` picks its reduction by tensor
shape, so identical rows sum differently in batches of different sizes (the
"wq_b difference" was this; the recompute probe had found wq_b invariant).
From here every checksum is an exact per-row fingerprint: the row's bit
patterns, position-weighted, modulo a prime below 2^24, in one Triton launch
(`fingerprint_gpu_check.py`: matches the torch reference in 64 of 64 cases,
detects each flipped value, replays identically from a CUDA graph). The
torch-op version multiplied graph nodes and pushed boots below dgx1's
startup guard; the exact trace arms also run with 0.45 GiB of KV cache (just
above the 0.43 GiB one 262K-token request needs).

### Causes, each replayed and fixed

| Cause | Fix | Evidence |
|---|---|---|
| mHC `pre`/`post_pre` took the exact count, else the 4096 plan (TF32 TMA, split K by 4) | `vllm-0028-mhc-smallest-capacity.patch` | `mhc_serving_replay.py`: pre invariant 3-192 rows, post_pre 3-72; before, unprepared counts grouped with the 4096 plan |
| BF16-output GEMVs (index head weights, index key, ratio-1 compressor) on the torch backend: cuBLAS switches kernels by M (index weights at 16/17, index key at 64/72) | `vllm-0029-gemv-batch-invariant-backend.patch`: SIMT below 256 rows under `VLLM_DS41_BATCH_INVARIANT=1` | `gemv_backend_replay.py`: SIMT invariant 19-192; index key default split at 64/72 |
| Block-FP8 Engram projection: 1- to 6-row plans split K four ways | `vllm-0030-block-fp8-no-split-k-batch-invariant.patch`: one slice under the same variable | `engram_wkv_replay.py`: 6-row target alone differed from 7-512 |
| Deterministic MoE: the M=1 materialized launch is not repeatable | `0009-moe-deterministic-single-row.patch` (determinism series) | `moe_position_replay.py`: 256 calls, one group, all repeat |
| Prefill sequence parallelism (from 205 tokens): NCCL's reduce-scatter adds a row's partials in an order set by its owning rank and the message size | `vllm-0031-reduce-scatter-rank-order-batch-invariant.patch`: an all-to-all and the one-shot all-reduce's float32 rank-order sum, under the same variable | wide trace below: first difference at exactly the first row whose owner changed; with 0031, one output over 998-1040-row prefills |

Ruled out by probes: the sparse-attention kernel is row-independent
(`debug-attn-probe2.diff`: each of a step's first rows recomputed alone equals
the batched row, 3,886 of 3,886 layer-steps); its split count is fixed per
plan for DS4.1; the MoE is invariant at every position of 2- to 8- and
28-row batches, with random and shared-expert neighbours.

### Traces

- Trace 3 (GEMV and MoE fixes, exact): layer 0 bit-identical between solo and
  mixed prefills; JSON first differs at layer 1 (mHC residual), prose at layer
  20's compressor latent (the index head weights at layers 2-14 differ but do
  not propagate: a short prompt selects all its compressed entries).
- Trace 4 (+ mHC): JSON prefill bit-identical at every layer; remaining
  differences start in torch-backend GEMVs.
- Trace 5 (+ SIMT): 12 of 12 mixed JSON runs one output, 11 of 12 prose; solo
  differs at small decode steps (Engram split-K).
- Trace 6 (+ block-FP8 without split-K): one output per prompt in all 30 runs.
  `analyze_trace3.py` matches target rows across runs by (position, input
  token), comparing a row only where its whole prefix is the accepted text and
  it was not a DSpark dead row (computed for shape, experts skipped, verified
  again later): every compared row bit-identical in every recorded main-model
  tensor, on all three ranks (7,266 rows per rank, 44 run pairs), and no row
  computed twice in a run differs.

### Performance of the five fixes (`run36`, one boot per arm, pinned cost table `5dc8961a`)

`run36.sh`, same method as round 13 (`tables_lookup.py` with run36's arms; cost
table SHA-256 `5dc8961a` loaded at every boot). "Always-on fixes" is r5o with
0027, 0028 and b12x-0006; the last arm is the deterministic MoE (0004-0009)
with all five fixes and `VLLM_DS41_BATCH_INVARIANT=1`.

| Point | r5o | + always-on fixes | det. MoE | det. + all + BI |
|---|---|---|---|---|
| prose c1 tps | 50.50 ±3.7% | 50.05 ±3.8% | 51.76 ±0.4% | 50.46 ±0.3% |
| step ms | 42.75 | 42.18 | 43.18 | 43.30 |
| verified / accepted | 3.239 / 1.247 | 3.243 / 1.197 | 3.336 / 1.327 | 3.339 / 1.277 |
| JSON c1 tps | 76.94 ±2.8% | 76.66 ±2.1% | 73.83 ±0.4% | 76.92 ±0.5% |
| step ms | 48.61 | 48.30 | 48.18 | 48.28 |
| verified / accepted | 4.540 / 2.956 | 4.521 / 2.926 | 4.391 / 2.754 | 4.462 / 2.938 |
| prose c8 tps | 161.69 ±0.8% | 162.69 ±1.9% | 164.45 ±3.3% | 163.40 ±1.9% |
| verified / accepted | 2.417 / 1.226 | 2.409 / 1.220 | 2.244 / 1.125 | 2.333 / 1.196 |
| JSON c8 tps | 232.19 ±3.3% | 235.25 ±2.2% | 220.48 ±1.8% | 228.53 ±3.8% |
| verified / accepted | 3.519 / 2.873 | 3.499 / 2.868 | 3.549 / 2.934 | 3.522 / 2.902 |

- Always-on fixes against r5o: -0.9% to +1.3%, every point inside its
  interval; step time 0.3-0.6 ms lower at one stream.
- Batch-invariant deterministic arm against r5o: -1.6% to +1.1% (JSON c8 the
  only loss, inside both intervals). Against the deterministic MoE alone its
  step time is 0.1 ms higher at one stream: that is the cost of SIMT GEMVs and
  one-slice Engram projections. Its throughput moves with acceptance (its own
  text), as in round 13.
- Short prompts (7-68 tokens, prefill plus one step, median of seven): r5o
  149-171 ms; always-on fixes 4-7 ms faster at every length; the deterministic
  MoE 2-5 ms slower than r5o; the batch-invariant arm 3-8 ms faster than r5o
  (the exact-or-max lookups had sent unprepared prefill sizes to the 4096-row
  plans). One boot per arm.

### Final validation (`run40`, `detm-r5o-final-trace`: 0027-0030, b12x-0006, 0004-0009, `VLLM_DS41_BATCH_INVARIANT=1`)

Five mixes, three repeats, 128 tokens: JSON 15 of 15 runs one output, prose 15
of 15. A third target, `long` (998 prompt tokens), gave 7 outputs in 15 runs,
and its first token's logprob already differed: its output is a function of
the prefill step's row count alone (998 rows solo; 1009, 1010, 1019, 1021,
1035, 1040 beside background decodes; equal counts gave equal outputs, even in
different mixes). The traces fingerprint at most 64 rows per call, so the
prefill left no records.

### Long prefills: sequence parallelism's reduce-scatter (`run41`)

- `prefill_replay.py` (checkpoint weights, a 998-row target alone and behind or
  ahead of 1-1000 rows): the V4.1 GEMVs, the block-FP8 query and Engram
  projections and all three mHC operations are invariant from 998 to 1062 rows
  (the index head weights on cuBLAS change from 1126 rows). `moe_prefill_replay.py`:
  the deterministic MoE through serving's one plan is invariant from 998 to 1998
  rows. (With every prepared count warm, the 1536-row variant does not even
  repeat; serving does not warm it.)
- A prefill of T tokens from 205 (serving log; where the hidden-state message outgrows
  the fixed-order RoCE all-reduce) runs sequence-parallel (`sp_prefill.py`):
  each rank owns ceil(T/3) rows for the row-wise work, and the WO and MoE
  outputs are reduce-scattered (NCCL) instead of all-reduced. NCCL adds the
  three partials of a chunk in an order that depends on the receiving rank
  (ring rotation) and the message size, so a row that changes owner changes bits.
- `debug-attn-exact3.diff` (overlay `attn-exact3`) sends calls of 65-1088 rows
  to separate wide logs that decode steps never overwrite. With it
  (`detm-r5o-final-wide-trace`, 10 long runs, 4 tokens), on all three ranks:
  repeats with equal T match in every row; against solo, the first differing
  row is exactly the first target row whose owner changed (ceil(T/3) minus the
  target's offset: 326, 325, 319, 308, 305 for T = 1009, 1010, 1019, 1035,
  1040), every earlier row matches in every record, and the first record to
  differ is layer 0's MoE input, the first gathered tensor after WO's
  reduce-scatter; attention output before it matches.
- `vllm-0031-reduce-scatter-rank-order-batch-invariant.patch`: under
  `VLLM_DS41_BATCH_INVARIANT=1` the TP reduce-scatter exchanges the chunks
  (grouped NCCL send/recv on the same communicator, the same bytes) and adds
  them locally in rank order. Its first version (`run42`, overlay
  `gemv-lookup-mhc-bi-fp8-rs`) rounded after each BF16 add; the published one
  (`-rs2`, below) adds as the one-shot all-reduce does. Host tests with three
  simulated ranks: rank-order sums, one row's partials at every position of
  two batch sizes reduce to one bit pattern, and per-add rounding is
  distinguishable (3/3). Base: r5o's vLLM tree (`c108cd6d`); it touches only
  `cuda_communicator.py` and applies with or without 0027-0030. Quality: a
  different rounding of the same sums in batch-invariant mode, none
  otherwise. Not proposed upstream.

### Validation with 0031 (`run42`, `detm-r5o-rs-trace`: the final arm plus 0031, wide logs)

Five mixes, three repeats, 128 tokens:

| Target | Runs | Outputs | Prefill rows seen | Rows compared bitwise per rank (22 run pairs) | Differences |
|---|---|---|---|---|---|
| JSON (19 tokens) | 15 | 1 | 19, 28, 32, 40, 48 | 3,550 | 0 |
| prose (15 tokens) | 15 | 1 | 15, 24, 27, 39, 48 | 3,690 | 0 |
| long (998 tokens) | 15 | 1 | 998, 1009, 1010, 1018, 1019, 1021, 1035, 1038, 1040 | 25,139 | 0 |

The long target's first logprob is -0.143140 in all 15 runs; on all three
ranks every recorded main-model tensor of every compared row (prefill rows
included) matches, and no row computed twice in a run differs. Its text
differs from run40's solo text: 0031 adds in another (fixed) order than NCCL's
ring did for these rows.

Performance (`run42.sh`, `tables_rs.py`: one boot per arm, pinned cost table
`5dc8961a` at every boot, three decode samples, cold prefill of real text with
three repeats). The last two arms differ only by 0031.

| Point | r5o | det. + all fixes + BI | + 0031 |
|---|---|---|---|
| prose c1 tps (step ms) | 50.85 ±6.0% (42.48) | 49.34 ±9.3% (44.29) | 50.43 ±0.6% (43.28) |
| JSON c1 tps (step ms) | 77.95 ±3.0% (48.45) | 77.11 ±0.2% (48.30) | 77.21 ±0.2% (48.25) |
| prose c8 tps | 162.39 ±3.6% | 165.66 ±1.7% | 194.97 ±0.2% |
| JSON c8 tps | 234.72 ±7.8% | 229.33 ±1.1% | 285.11 ±0.7% |
| prefill 1024 tok/s (TTFT s) | 2154 ±16.3% (0.44) | 2102 ±15.5% (0.45) | 2072 ±20.7% (0.46) |
| prefill 4096 | 3861 ±6.3% (0.97) | 3891 ±3.7% (0.96) | 3792 ±4.8% (0.99) |
| prefill 16384 | 3820 ±3.0% (3.86) | 3772 ±5.7% (3.90) | 3735 ±1.4% (3.94) |
| prefill 65536 | 3857 ±0.4% (15.01) | 3828 ±1.7% (15.12) | 3765 ±1.1% (15.37) |

- One stream: the batch-invariant arms are within 1% of r5o with 0031 (-0.8%,
  -0.9%), inside r5o's intervals.
- Prefill: 0031 costs 1.0-2.5% against the same arm without it; the whole
  batch-invariant stack is 1.8-3.8% below r5o (at 65536 tokens -2.4%,
  15.37 against 15.01 s, the only difference outside both intervals).
- Eight streams: the bench sends eight copies of one prompt. With batch
  invariance the copies write the same text (r5o: 24 distinct outputs over 24
  requests; without 0031: 6 per sample; with it: 2), so their rows route to
  the same experts and the MoE does less work: the +20-22% at eight streams
  is that, not a speedup for varied traffic.

### The next divergence: small and large steps (`run42` bench, `run43`)

With 0031, each eight-stream sample still gives two texts, and the split is
by prefill: one stream (time to first token 0.20 s) was prefilled alone in a
small step and writes one text; the seven prefilled together (0.52-0.56 s, one
sequence-parallel step of about 460 rows) write the other, in every sample.
The alone-prefilled stream also differs from the one-stream output, because
its first decode rows ran inside the others' large step. A row computed in a
small step (graph sizes: SIMT GEMVs, lagged mHC plans, the MoE's graph-size
variants, decode attention) does not equal the same row computed in a large
step (the GEMV prefill kernel, TF32 mHC, the MoE's 4096-token variant, extend
attention, the reduce-scatter). The traced targets never crossed: short
prompts and every decode row stayed in small steps, the long prompt's prefill
always ran in large ones.

`c8_trace.py` (`run43`, `detm-r5o-rs-trace`) sends eight copies of an
87-token prompt at once and treats each stream as a run, so
`analyze_trace3.py` compares the streams' rows by position and token. Both
rounds scheduled them alike: stream 0 prefilled alone (87 rows), stream 1
beside stream 0's first decode rows (93 rows), streams 2-7 together with both
streams' decode rows (534 rows, sequence-parallel). Streams 0 and 1 match in
every record of every prompt row (87 against 93 rows); streams 2-7 match each
other. Against stream 0, streams 2-7 first differ at position 0 in layer 0's
WO output after its reduction, with every attention record before it equal
(the query and KV projections, the index, the attention output), and stream
1's first generated row (computed in the 534-row step) differs from stream 0's
(93-row step) at the same record. The one-shot RoCE all-reduce that reduces
steps below 205 rows adds in float32 in rank order and rounds once; 0031's
first version rounded after each BF16 add.

0031's second version (`run44`, `detm-r5o-rs2-trace`) adds the chunks that
way (float32, rank order, one rounding; host tests 3/3, one of which shows
per-add rounding gives other bits). The eight streams again split by prefill
step (87 rows alone against 615 sequence-parallel), but on rank 0 WO's
reduced output now matches between the two steps: the first difference moves
to layer 0's router logits, and the router logits differ in every layer while
the MoE input and the routed output mostly still match (layer 7's routed
output is the first to follow). The router gate runs SIMT up to its 192-row
plan and the TMA prefill kernel from 384 rows (round 13), and the two round
differently. (Ranks 1 and 2 record WO's output after the reduce-scatter only
for their own rows, which the analysis maps from row 0, so their WO
comparisons do not align; rank 0's rows start at 0.)

With the second version the long target stays invariant (`run44`, five
mixes, two repeats): one output in 10 runs over prefill steps of 998-1040 rows,
first logprob -0.152026 in each, and every compared row bit-identical on all
three ranks (13 run pairs, 14,996 rows per rank). The performance above was
measured with the first version; the second adds one float32 copy of the
rank's rows per reduce-scatter and was not measured separately.

### What cross-step invariance still needs

The fixes make a row's bits independent of batch composition as long as the
row is computed in the same kind of step: small steps (up to the graph sizes
and small-plan ranges) or large prefill steps. A row computed in a small step
still differs from the same row computed in a large one, and in live serving
that is routine: a request's decode rows share a step with another request's
long prefill, and identical prompts prefill in steps of different sizes.
Every operation would need one arithmetic for every row count in
batch-invariant mode:

- router gate and ratio-2 compressor (FP32 out): SIMT up to 192 rows, the TMA
  prefill kernel from 384 (the prefill kernel gave one result for a 998-row
  target in batches up to 1998 rows; at 19 rows it took 35.9 us against SIMT's
  28.2 us in round 13);
- the BF16-output GEMVs: SIMT below 256 rows (0029), torch or the prefill
  kernel above;
- mHC: native with lagged prepare up to 72 rows, native 96-192, TF32 TMA from
  384, and four K splits at 4096;
- the MoE: the graph-size variants against the 4096-token variant (19-48 and
  49+ rows differ in the round 13 replay);
- attention: decode rows inside a mixed step run the extend kernels (not yet
  traced).

Each needs a replay across 1-4096 rows and a performance screen, since the
choice trades decode or prefill speed for invariance.

### Scope and limits

- Demonstrated in serving: temperature-0 outputs and every traced main-model
  tensor bit-identical across batch composition for requests of 15-998 prompt
  tokens beside up to seven background requests (decode steps up to 48 rows,
  prefill steps up to 1040 rows, sequence-parallel from 205 tokens), on all
  three ranks, with every fix and `VLLM_DS41_BATCH_INVARIANT=1`, as long as
  each row stays in the same kind of step (small, or large prefill).
- Not invariant: the same row computed in a small step and in a large one
  (eight identical concurrent requests give two texts); see "What cross-step
  invariance still needs".
- Replayed beyond that: the GEMVs, block-FP8 projections and mHC stay
  invariant for a 998-row target up to 1998-row batches (across capacity
  changes), except the index head weights on cuBLAS, which change from 1126
  rows (they matter only once the indexer must choose, above 2048 context
  tokens); the MoE through serving's plan is invariant to 1998 rows.
- Not covered: prompts longer than the 4096-token batch budget (chunked
  prefill, whose chunk boundaries move with the other requests' tokens);
  prefill steps of 193-997 rows (between the small-plan ranges, where SIMT,
  the lagged mHC plans and the MoE's graph-size variants hold, and the replayed
  prefill range); more than eight concurrent requests.
- Switches: 0029, 0030 and 0031 act only under `VLLM_DS41_BATCH_INVARIANT=1`;
  the deterministic MoE (0004-0009) only under `B12X_DYNAMIC_DETERMINISTIC_OUTPUT=1`.
  0027, 0028 and b12x-0006 are unconditional: they change production numerics
  for row counts that are not prepared plans, with no measurable cost.
- The drafter still differs at a fixed schedule; with a batch-invariant target
  that changes speed, not text (greedy block verification accepts the
  target's own argmax), which the traces confirm.
- Nothing here is promoted; promotion needs the owner's acceptance.
