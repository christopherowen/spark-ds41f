# Exact raw-wide selection: repeated answer and request screen

Source `e607aea8ef32fb55fae665ddaf95dc0e0445cbd0`, immutable unreleased C36 fixture, sp1–sp4 TP4. One exact selection scheduling variable relative to the source1024/dense64/native-generation parent. Native controls are that parent comparison's unchanged published same-UTC-day pair. The original reuse guard passed before this job; subsequent cooling audit shows that pair lacks the required performance fan curve. These are exploratory timings, not promotion-quality speed evidence. No benchmark prompt, scorer, expected answer, weight or response-policy change.

| RULER screen26 | Native A | LSA A | LSA B | Native B |
| --- | ---: | ---: | ---: | ---: |
| Correct / natural completions | 26/26 | 26/26 | 26/26 | 26/26 |
| Request wall seconds | 573.903 | 563.911 | 564.474 | 575.807 |
| Median request seconds | 17.471 | 16.777 | 16.757 | 17.767 |
| Tokens including thinking | 38383.000 | 38434.000 | 38434.000 | 38383.000 |
| Server prefill seconds | 268.076 | 260.126 | 259.974 | 269.450 |
| Server decode seconds | 302.011 | 300.031 | 300.444 | 302.377 |

Observed correct-answer throughput changes+1.89% versus Native; conditional joint95 interval[0.9653,1.0772]. Observed parent/raw whole-workload wall ratio is1.0033. Neither stock-fan comparison establishes a qualified request-speed win. The larger isolated coarse-selection gain does not imply the same whole-model gain. Two boot means and fixed repeated tasks do not establish general intelligence superiority.

Both candidate boots preserve all26 independently correct answers, with zero paired wins, losses, incomplete responses or unstable scores. Parent comparison: {"lsa-a": {"cases": 26, "final_text_hash_matches": 26, "usage_matches": 26, "reasoning_length_matches": 26, "scope": "final text hashes and public lengths; not a full reasoning-text/token-ID identity claim"}, "lsa-b": {"cases": 26, "final_text_hash_matches": 26, "usage_matches": 26, "reasoning_length_matches": 26, "scope": "final text hashes and public lengths; not a full reasoning-text/token-ID identity claim"}}. All thinking and unsuccessful time counts. Temperature0, seed0, high thinking and65536 generation tokens remain fixed.

All four receipts pass the old memory/thermal guard but show the fan service absent. Starts are below55C and no thermal slowdown is recorded; that does not satisfy the repository measurement method. The corrected summary explicitly records `timing_qualified:false`. The new client refuses such future measurements and control reuse before requests. Every historical receipt remains unchanged; see `../2026-11-06-timing-conditions/README.md`.

C37 restores at2026-11-06T01:41:24.593401+00:00. Independent four-node verification finds the exact digest, clean equal serving revisions, absent experimental mounts/environment and removed reservations. No raw responses, token IDs or activations leave the node. Full-context, held-out, LongBench, coding and serving/admission qualification remain required for this recipe. No production promotion.

The next cooperative-summary integration has a verified13-patch source stack; its integrated GPU checks and request/answer gates remain pending. No claim from that implementation is folded into this comparison.
