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
