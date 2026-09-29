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

Pending.
