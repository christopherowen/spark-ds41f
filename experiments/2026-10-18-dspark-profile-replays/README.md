# DSpark cost profile: 25 replays of each decode shape

Owner, 2026-10-18: "if changing 5 to 25 only introduces a boot-time cost, and it removes
the 'noise' from measurements we should do it immediately. I'm worried that the variance
is causing us to not do the right thing when we're developing."

## Why

DSpark's adaptive verification prices extra verification rows from cost curves each build
profiles once, at its first boot, and then pins: every graph-captured shape and every
eager shape past the capture limit is timed in alternating sweeps, five of them, and each
shape keeps its median. The budget acts on the slopes between neighbouring shapes, and a
slope is the difference of two medians of five. In the r6w release comparison
(experiments/2026-10-18-r6w-swiglu) r6w's pinned 48-to-96-row slope was 5.9% steeper than
r6v's, while the engine grid (100 replays) put the builds' slopes within 0.3-1.7%; r6w
drafted 3.8% fewer tokens a step at sixteen streams and its throughput read level although
its steps were faster. On r6v's curves it was +0.5%. Equivalent builds drafting differently
makes every release comparison carry noise that is not the code.

## Change (vLLM branch dspark-profile-replays, 6bc1fc400, on r6v's tree)

- The graph-captured shapes (decode and verification, up to 96 rows: 15-70 ms a step) are
  timed in 25 sweeps; the eager tail (up to 8,192 tokens, a second or more a step, priced
  as a whole) keeps 5. That is a few seconds more of a build's first boot instead of the
  two minutes 25 sweeps of everything would take; later boots read the pinned curves.
- The pin key names the replay counts, so curves timed with fewer replays are not reused.

The cost model itself (measured costs at boot, `adaptive_verification_cost_scale`) is to
be reworked with the acceptance work; this change only takes the noise out of its inputs.

## Runs

1. `tests.json`: the adaptive verification tests over the r6v image.
2. `stability.json`: three fresh profiles with 25 replays and three with 5, each into its
   own cost directory; `compare_profiles.py` gives each slope's spread across repeats.
   Gate: the 25-replay spread is well under the 5-replay spread; first-boot cost recorded.
3. `screen.json`: the screen suite, 25 replays against r6v and 25 replays again (the first
   arm profiles, the repeat reads its pinned curves, as production would): no metric worse.
