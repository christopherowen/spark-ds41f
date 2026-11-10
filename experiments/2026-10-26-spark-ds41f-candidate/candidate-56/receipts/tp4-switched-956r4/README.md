# #956 without #957 part 1 (r4) at tp4-switched (sp1..sp4), 2026-11-10

[../../attribute-956r4-tp4-switched.json](../../attribute-956r4-tp4-switched.json), in sp1's window (10:49-11:28Z), on
the candidate-56 image: `tp4-switched-956r4` (candidate-54 + 0796 r4) against `tp4-switched-no957p1-no956`
(candidate-54's code), ABBA in one session.

| Prefill (tok/s) | candidate-54's code (two boots) | #956 r4 (two boots) | Change | Verdict |
| --- | --- | --- | --- | --- |
| 32K | 8,701 / 8,726 | 8,882 / 8,867 | +2.08% (±6.00%) | the same |
| 64K | 9,063 / 9,004 | 9,371 / 9,313 | +3.40% (margin 2.27%) | better |
| 128K | 9,079 / 9,060 | 9,336 / 9,390 | +2.84% (margin 2.69%) | better |
| 256K | 8,815 / 8,804 | 9,150 / 9,074 | +3.80% (margin 1.48%) | better |

The grid step alone, ABBA against the same control: geometric mean +0.16%, the same (margin 0.32%). Decode is
unchanged.

**Verdict: #956 passes** without #957 part 1, and joins the next merged base as 0796 r4.
