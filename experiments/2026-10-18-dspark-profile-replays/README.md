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

## Results

**Tests:** 28 passed (the two-GPU TP test skipped in a one-GPU container).

**Repeatability** (2026-10-18, `stability.json`: three fresh profiles of the r6v build with
25 replays and three with 5, each into its own cost directory; `compare_profiles.py`). The
spread of each slope across the three profiles, (max - min) / mean:

| Slope | 5 replays | 25 replays |
| --- | ---: | ---: |
| 1-6 rows | 7.4% | 10.0% |
| 6-16 rows | 2.2% | 2.9% |
| 16-48 rows | 13.4% | 2.9% |
| 48-96 rows | 10.0% | 2.6% |

From 16 to 96 rows, where the budget decides how much to draft at eight and sixteen
streams (and where r6w's 48-to-96 slope misled it), the noise falls four to five times.
The 1-6-row slope does not improve: there the profiles differ from boot to boot (the
one-row cost itself moves about 0.2 ms between boots), which more samples of one boot
cannot average away; at one stream extra rows are cheap and the budget drafts deep either
way. A fit across neighbouring shapes is the tool for that part, with the cost model's
rework.

**First boot:** ready in 113.0 / 125.5 / 125.3 s with 25 replays against 107.8 / 108.7 /
110.8 s with 5, about 5-15 s more on a build's first boot; later boots read the pin.
