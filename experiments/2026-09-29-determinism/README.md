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

Pending.
