Timing qualification correction,2026-11-06: these stock-fan measurements are exploratory. The documented performance fan curve was not verified; recorded below55C starts and no thermal-guard interruption do not satisfy that requirement. Answer/kernel correctness evidence remains separate. All original numbers and receipts are unchanged. See `../2026-11-06-timing-conditions/README.md`.

# Source1024 with native generation: repeated request screen

Source `17b9e81e1716c8266076293c0c78e43b2a99701e`, sp1–sp4 TP4, immutable unreleased C36 fixture. Same frozen RULER26 tasks, temperature0, seed0, high thinking and65536 generation tokens. Source-prefill budget1024 retains65536 fine source keys; dense budget64 and native generation/verification are fixed. This recovers the earlier128K QA loss in both candidate boots. No prompts, grader, expected answers or weights changed.

| RULER screen26 | Native A | LSA A | LSA B | Native B |
| --- | ---: | ---: | ---: | ---: |
| Correct / natural completions | 26/26 | 26/26 | 26/26 | 26/26 |
| Full request wall seconds | 573.903 | 566.516 | 565.587 | 575.807 |
| Median request seconds | 17.471 | 16.837 | 16.616 | 17.767 |
| Generated tokens including thinking | 38383.000 | 38434.000 | 38434.000 | 38383.000 |
| Server prefill seconds | 268.076 | 262.268 | 261.057 | 269.450 |
| Server decode seconds | 302.011 | 300.175 | 300.445 | 302.377 |

Correct-answer throughput changes+1.55%, conditional joint95 interval[0.9541,1.0824]. This is a small observed gain whose interval includes a slowdown, not an established performance win. Two boot means cannot establish distributional assumptions or broad quality superiority. Every answer remains correct on both boots, with no paired gains, losses or unstable scores. Native already reaches the ceiling on this screen. All generation, thinking and unsuccessful time counts.

Conditions pass on all four boots. No guard interruption, recorded thermal slowdown or preemption; complete memory, swap and thermal observations remain in the public receipts. This is C1 on a small pilot, not full-context, admission, concurrency, coding or held-out qualification.

C37 restores at2026-11-06T01:11:04.687534+00:00. An independent four-node reader verifies the exact image, clean equal serving revisions, no experimental mounts/environment and removed reservations. Raw text, token IDs and captured activations remain private on nodes. Public receipts are in `../2026-11-05-quality-matrix/runs/wide-merge-ruler-window-01`. No production promotion.

The exact raw-wide coarse optimization is separately component-qualified in `../2026-11-06-wide-coarse/results.md`. Its request-speed and final-answer comparison is the next causal experiment.
