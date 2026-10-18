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

## Results: no fixed threshold matches the timed budget; the feature itself is worth 9-19%

Determinism: outputs identical (3eab5112 / db25fde3, c9cf9dbe / c6bb4f4e). Boots without the
profile: 76 s against 87-90 s. Tests: the first run's seven failures were the tests' (a
required argument the old calls lacked, one expectation of the cost model, a warm-up stub
missing `batch_sharder` since an earlier commit), fixed in #394.

Throughput against r6v's timed budget (change, verdict against the run's repeat; the 0.2
repeat matched itself to 0.0-0.4%):

| Arm | screen c16 | single-stream c1 | sampled c16 | sampled c8 | sampled c1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| threshold 0.2 | -0.7% | -4.0% | -1.4% (worse) | -1.1% | -3.1% |
| threshold 0.3 | -0.1% | -7.2% (worse) | -0.4% | -0.3% | -6.4% |
| adaptive verification off | -17% | -8.6% (worse) | -18% | -19% | -9% |

A fixed threshold drafts too little at one stream (rows there are nearly free: 2.0-2.1 tokens
a step against r6v's 2.26) and, at 0.2, too much at sixteen (35.4 tokens a step against 30.4,
steps 15% slower). Turning adaptive verification off (five drafts always verified) costs
9-19%: the budget pays at every load, so the feature stays. What a draft is worth depends on
what a row costs relative to the whole step at that load, which a threshold ignores:
experiments/2026-10-18-dspark-linear-budget (one constant) and
experiments/2026-10-18-dspark-static-costs (the full cost shape from a static table) keep
that, with nothing timed.
