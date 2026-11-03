# Census attribution review

Source review on 2026-11-03 at deployment commit
`b28365c38a74c966032dcf558d8d1bb0a282d4b6`. No GPU execution or runtime change
was performed for this review. These findings concern what the diagnostic can
establish; they do not identify the cause of issue #852.

The reviewed files are under
`experiments/2026-11-03-copy-drafter-shared-budget/overlay/census-async/`:

- [`vllm/v1/worker/gpu/model_runner.py`](../2026-11-03-copy-drafter-shared-budget/overlay/census-async/vllm/v1/worker/gpu/model_runner.py),
  SHA256 `fc3ed7138705ea671ecdac84fbc1bf808fa4a239430dc24659869fb9c454bb14`.
- [`vllm/models/deepseek_v4_1/tilelang/router.py`](../2026-11-03-copy-drafter-shared-budget/overlay/census-async/vllm/models/deepseek_v4_1/tilelang/router.py),
  SHA256 `148c4a4df3b1fab2793e24a74dbe0eb6b304a1d739be9d2bd21967cf71a9d7d5`.

The runner drains the census every 64 calls (lines 80–84, 718–722). Its calls
follow target sampling and drafter proposal (3544, 3788). In the usual two-call
step, the drain therefore occurs after the drafter. The router accumulates
per-router row bits until that drain (1096–1111), without retaining the producing
step or phase. The runner nevertheless labels all returned hits using the
draining call's `drafter` argument (737–769). A target hit can consequently be
printed as a drafter hit. A router name remains useful evidence of where a hit
was detected, but the printed phase is not reliable attribution.

When a drain occurs in the target call, its request, slot, position, input ID,
padding, and acceptance fields come from the current batch and mutable buffers
(runner 726–736, 772–806). They need not describe the earlier batch that set the
bit. The bits also lose first-occurrence order across steps and routers. Neither
the printed "hit step" counter nor the list order identifies the first corrupt
operation.

The diagnostic changes computation after a hit: router line 1098 zeroes gate
partials. Contrary to the "weight-0 experts" comment at 1064, zero logits pass
through `sqrt(softplus(0))` and normalization (979–1022), producing nonzero routing
weights. Later hits may include propagation through this altered computation.

Rank checks have separate limits. The baseline
[`overlay/limit-reads/.../copy_drafts.py`](../2026-11-03-copy-drafter-shared-budget/overlay/limit-reads/vllm/v1/worker/gpu/spec_decode/copy_drafts.py)
hashes only consequential read indices (line 162), not the limits, slots, or
layouts. Matching CRCs do not establish matching rank state. The minimal arm's
[`overlay/minimal-crc/.../model_runner.py`](../2026-11-03-copy-drafter-shared-budget/overlay/minimal-crc/vllm/v1/worker/gpu/model_runner.py)
hashes CPU token counts and eligible-draft arrays (1760–1778), not the resulting
GPU offsets, row mapping, or confidence data. A clean comparison run likewise
does not by itself exclude a proposed failure mechanism.

A diagnostic suitable for attribution would retain the first observed failure's
step, model phase, layer, operation, and contemporaneous row metadata on device
until collection. Locating the producer further requires checking inputs and
outputs around the suspected operation, rather than treating the router's
finite-value assertion as the origin. These are review criteria, not an
implemented or hardware-validated fix.
