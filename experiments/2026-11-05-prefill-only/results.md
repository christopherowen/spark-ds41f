Timing qualification correction,2026-11-06: these stock-fan measurements are exploratory. The documented performance fan curve was not verified; recorded below55C starts and no thermal-guard interruption do not satisfy that requirement. Answer/kernel correctness evidence remains separate. All original numbers and receipts are unchanged. See `../2026-11-06-timing-conditions/README.md`.

# Native generation with compact dense prefill

Source `f625c58c80619e69ca901af9d888daa931da38a7`, immutable C36 image, sp1–sp4 TP4.
This changes only generation/verification selection to native, retaining compact
dense prefill and native source. Native controls are the published same-day pair
from `native-source-ruler-screen-window-03`; their original bytes, timestamps and
reuse checks are retained. Both candidate boots are fresh.

| RULER screen26 | Native A | LSA A | LSA B | Native B |
| --- | ---: | ---: | ---: | ---: |
| Official correct answers | 26/26 | 26/26 | 26/26 | 26/26 |
| Natural completions | 26/26 | 26/26 | 26/26 | 26/26 |
| Request wall seconds | 575.970 | 997.252 | 998.049 | 574.289 |
| Median request seconds | 17.516 | 16.590 | 16.516 | 17.432 |
| Generated tokens including thinking | 38383 | 91878 | 91878 | 38383 |
| Server prefill seconds | 269.712 | 258.599 | 259.415 | 267.773 |
| Server decode seconds | 302.240 | 734.570 | 734.547 | 302.489 |
| Accepted tokens per draft round | 3.198 | 3.223 | 3.223 | 3.198 |

Every answer remains independently correct on both boots, with no paired wins, losses or unstable scores. Correct-answer throughput changes-42.35% and mean workload time changes73.47%. The conditional joint95 throughput interval is[0.5502,0.6048]; it assumes normal boot means with only two samples per arm. This recipe is a performance failure, despite preserving the26-case answer score. It is not promoted.

Generation length increases from38383 to91878 tokens. Most added request time occurs in two independently correct reasoning responses. No answer cap, thinking limit, grader repair or benchmark-specific serving rule is introduced. Temperature0, seed0, high thinking and65536-token budget remain identical; all thinking and unsuccessful time counts.

All four conditions receipts pass. Memory, thermal and swap observations and every request remain in the comparison directory. C37 restores at2026-11-05T22:59:05.307163+00:00; an independent four-node reader verifies its image, clean equal serving revisions, absent experimental mounts/environment and removed reservations. Promoted runtime and immutable release evidence remain unchanged.

The separate source-prefill interaction is qualified only for graph dispatch: source `ef48cc66` passes72 routes on12 frozen captured batches at1/8/64 rows, including original, empty and restored extents, with zero initialization errors. This is not actual decode-query or model-quality evidence. Allocator peak counters are unavailable under strict no-cache checking. Two remote setup failures are retained; no GPU candidate ran during either. Its model comparison remains separate.
