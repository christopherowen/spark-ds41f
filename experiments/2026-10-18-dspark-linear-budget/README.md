# DSpark draft budget by a fixed linear step cost

Follows experiments/2026-10-18-dspark-budget-rule. A fixed survival threshold drafts too
little at one stream (rows there are nearly free; threshold 0.2 verified 2.04 tokens a step
on sampled traffic against r6v's 2.17, -3.1% throughput) and too much at sixteen (35.4
tokens a step against 30.5, steps 15% slower, about -0.7%): how much a draft is worth
depends on what a row costs relative to the whole step, which a threshold ignores.

## Change (vLLM branch dspark-linear-budget, 3d9777f8c, on r6v's tree)

The cost model's own argmax with its boot-timed curves replaced by one recipe constant:
each step's budget maximises expected tokens (the step's sampling requests plus the admitted
drafts' survival probabilities, best first) over `1 + row_cost * rows`, `rows` the step's
non-draft rows plus its admitted draft rows. `adaptive_verification_row_cost` is a
verification row's cost as a fraction of a step's fixed cost: about 0.3 ms a row against
about 15.7 ms of fixed step (target and drafter) in the r6v profiles, so 0.02; the old cost
scale 2.0 on accurate curves is about 0.035. Nothing is timed: the same schedule on every
build and boot, and the boot is about 11 s shorter.

## Runs

1. `tests.json`: adaptive verification, config, warm-up and B12X preparation tests.
2. `screen.json`: the determinism gate, then the screen, single-stream and sampled suites:
   row cost 0.02, r6v (its boot-timed budget), row cost 0.035, row cost 0.02 again.

Gate: no metric worse than r6v; the constant then goes into the recipe and the timed cost
model is deleted.

## Results: both constants level with the timed budget; 0.035 keeps r6v's operating point

Tests (`tests.json`, with the stub fix of #398): pass. Outputs identical in every arm (greedy
responses 128/128, teacher-forced distributions identical). Verdicts against the run's own
repeat (row cost 0.02 matched itself within 0.0-0.4% on every throughput metric):

| Suite | Metric | r6v | Row cost 0.02 | 0.02 again | Row cost 0.035 |
| --- | --- | ---: | ---: | ---: | ---: |
| screen | c16 tok/s | 241.6 | 242.5 (+0.35%) | 241.9 | 240.3 (-0.55%) |
| screen | c16 tokens a step | 30.5 | 33.0 | 32.9 | 30.8 |
| screen | c16 TTFT p50 | 408 ms | 437 ms | 432 ms | 413 ms |
| screen | engine grid (step time) | | -0.49% | -1.77% | -1.14% |
| single-stream | c1 tok/s | 70.9 | 71.2 | 71.3 | 71.1 |
| sampled | c16 tok/s | 237.2 | 237.0 | 236.7 | 236.0 (-0.48%) |
| sampled | c8 tok/s | 179.9 | 180.2 | 180.2 | 180.7 (+0.44%) |
| sampled | c8 TTFT p50 | 284 ms | 312 ms | 313 ms | 301 ms |
| sampled | c1 tok/s | 68.4 | 68.35 | 68.4 | 68.4 |

Every throughput verdict is same; no arm is slower. The grid (a fixed-row step cost the
budget does not touch) drifted over the run (the repeat -1.29%); prefill 256K read -0.43%
in the 0.02 repeat against its own first run, also drift.

Throughput is flat across this range of budgets: row cost 0.02 verifies about 10% more
drafts per draft (1.90 against 1.73) and commits 8-12% more tokens a step, but its steps
run 7-11% slower, so the output rate is the same while each step does more work and first
tokens arrive 6-10% later. Row cost 0.035 reproduces r6v's operating point at sixteen
streams (30.8 tokens a step against 30.5, first tokens level) and drafts somewhat more at
eight. With throughput equal, less verification for the same tokens and level latency
decide: **row cost 0.035 goes into the recipe** (r6x, experiments/2026-10-18-r6x). Whether
the cost model's full shape beats one linear constant at that aggressiveness is
experiments/2026-10-18-dspark-static-costs.
