Timing qualification correction,2026-11-06: these stock-fan measurements are exploratory. The documented performance fan curve was not verified; recorded below55C starts and no thermal-guard interruption do not satisfy that requirement. Answer/kernel correctness evidence remains separate. All original numbers and receipts are unchanged. See `../2026-11-06-timing-conditions/README.md`.

# Four-boot retrieval screen

Source `821b5ec5`, C36 Native68 versus compact80 dense selection with native
source selection, sp1-sp4 TP4. Same frozen26-case RULER screen: all13 tasks at
32768/131072 generator lengths, temperature0, seed0, high thinking,65536 output
tokens, serial cold requests. Native-source routing is independently verified
on every rank during LSA A. No prompt, grader, expected answer or task changes.

| Completed screen | Native A | LSA A | LSA B | Native B |
| --- | ---: | ---: | ---: | ---: |
| Correct / natural completions | 26/26 | 26/26 | 26/26 | 26/26 |
| Official mean coverage | 100% | 100% | 100% | 100% |
| Generation wall seconds | 575.970 | 647.368 | 647.102 | 574.289 |
| Median request seconds | 17.516 | 16.882 | 16.727 | 17.432 |
| Tokens including thinking | 38383 | 49474 | 49474 | 38383 |
| Accepted tokens per draft round | 3.198 | 3.176 | 3.176 | 3.198 |

The repeated no-loss gate passes with zero wins, losses, incomplete responses or
unstable scores. Native source recovers the earlier128K QA score loss on both
candidate boots. This establishes preservation on this pilot, not improved
intelligence or quality on untested inputs. Full52/156 RULER and independent
LongBench/coding qualification remain required.

This recipe loses whole-workload speed: mean wall time rises12.54%, completed-
answer throughput falls11.14%. Native/LSA wall ratio is0.8886 with conditional
joint95 interval[0.8511,0.9265]. These normal-boot-mean assumptions are unverified
with only two samples per arm. Median request time is about3.8% lower, so the
longer requests dominate the total slowdown. No performance win is claimed.

Server counters explain the result: mean prefill totals fall about3.6%; decode
seconds per generated token fall about1.45%, while generated length rises28.9%.
Draft acceptance is recovered (accepted fraction0.7302 versus native0.7223,
accepted tokens per round3.176 versus3.198). The remaining slowdown is primarily
extra reasoning length, not evidence that this variant generates tokens more
slowly. All thinking and unsuccessful time is charged; no request is filtered.

Conditions pass on all four boots: no guard interruption or recorded thermal
slowdown, zero swap growth, maximum GPU76C, minimum thermal headroom12C. Minimum
available memory is 21.37 GiB. This is a four-node C1 observation, not an
admission or serving-matrix pass.

C37 restores at22:11:14 UTC. Independent all-node verification finds the exact
image, clean equal serving revisions, no experimental mounts/environment and
no reservations. Public commands, counters, conditions, quality decisions and
restoration evidence are retained in
`../2026-11-05-quality-matrix/runs/native-source-ruler-screen-window-03`.
Raw responses remain private. The recipe remains unpromoted; prefill-only/native
generation is a separate next experiment.
