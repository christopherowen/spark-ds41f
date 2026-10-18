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
