# DSpark draft budget by a rule of the inputs

Owner, 2026-10-18: "I don't like the boot timings at all and want it to go away. At worst we
should be using heuristics. I think we should maybe be dropping this feature altogether."
and "when we take on the task to improve acceptance, we should rework this part too"
(experiments/2026-10-18-dspark-acceptance).

## Why

The adaptive budget picks, each step, the number of draft rows that maximises expected
accepted tokens per unit of cost, the cost read from curves each build times at its first
boot (`adaptive_verification_cost_scale` 2.0 on the extra rows). With identical outputs,
equivalent builds then draft differently from one timing sample
(experiments/2026-10-18-r6w-swiglu); 25 replays cut that noise four to five times above
16 rows but not at one to six (experiments/2026-10-18-dspark-profile-replays).

## Change (vLLM branch dspark-heuristic-budget, 1462c99fc, on r6v's tree d11e9836)

Every scheduled draft slot whose survival probability (the running product of the drafts'
confidences up to it, as the budget already ranks them) is at least
`adaptive_verification_threshold` is verified, within the one-chunk logits limit; after 32
steps with none admitted one draft a request is verified, as before, so the confidence
estimator keeps learning. Admitting the slots above a fixed threshold is the cost model with
a fixed price per extra row. The boot profile, the pinned curves and the cost scale are
deleted; the eager shapes past the capture limit still run once at startup, untimed, so
their kernels compile there.

## Runs

1. `tests.json`: the adaptive verification, config, warm-up and B12X preparation tests.
2. `screen.json`: the determinism gate on the 0.2 arm, then the screen and single-stream
   suites: threshold 0.2, r6v (its boot-timed budget), threshold 0.3, adaptive verification
   off (five drafts always verified), threshold 0.2 again.

Gate: no metric worse than r6v. Whichever rule is no worse wins, since it is deterministic
and simpler; if adaptive verification off is no worse, the feature goes.
