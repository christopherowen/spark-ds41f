Timing qualification correction,2026-11-06: these stock-fan measurements are exploratory. The documented performance fan curve was not verified; recorded below55C starts and no thermal-guard interruption do not satisfy that requirement. Answer/kernel correctness evidence remains separate. All original numbers and receipts are unchanged. See `../2026-11-06-timing-conditions/README.md`.

# Four-node completed-answer pilot

Source `7b631ce12fbdb183a9f56a7013bd3bea1819fd88`, C36 immutable image
`sha256:3795255f485ea5c1a9db54c742921dd9b2917daad0799361e5c6f56a70119713`,
sp1–sp4 TP4, Native68 versus compact80. Runtime overlays equal `4a455449`.
This cohort is independent of the earlier TP3 BF16-summary pilot; their native
quality and timing numbers must not be pooled.

| Serial cold pilot | Native A | LSA A | LSA B | Native B |
| --- | ---: | ---: | ---: | ---: |
| Official correct answers | 9/12 | 9/12 | 9/12 | 9/12 |
| Natural completions | 12/12 | 12/12 | 12/12 | 12/12 |
| Total request seconds | 441.635 | 411.871 | 413.134 | 441.727 |
| Median request seconds | 20.607 | 23.997 | 24.047 | 20.560 |
| Generated tokens, including thinking | 31780 | 29130 | 29130 | 31780 |
| Server prefill seconds | 168.146 | 158.188 | 159.283 | 168.161 |
| Server decode seconds | 270.739 | 251.023 | 251.197 | 270.858 |
| Accepted tokens per draft round | 2.851 | 2.655 | 2.655 | 2.851 |

The original strict no-loss gate passes, with identical answer decisions on both
boots, zero losses and zero paired wins. This is quality preservation on12
documents, not evidence of improved general intelligence. Temperature0, seed0,
high thinking and65536 generated tokens remain shared. The independently
identified reference concerns are diagnostic only; every official score stays
unchanged. See the [reference audit](../2026-11-05-answer-quality/reference-audit/README.md).

Mean workload time is6.61% lower and correct-answer/request throughput7.07%
higher. Native/LSA workload ratio is1.0707 with conditional joint95 interval
[1.0278,1.1171]; correct-answer throughput interval is[1.0263,1.1154]. These
assume normal boot means with only two samples per arm, not universal workload
guarantees. Median latency is16.7% higher. Generated length falls8.34% and
prefill time about5.6%, while decode seconds per generated token rise about1.2%.
Shorter reasoning materially contributes; this is not uniformly faster decoding.

Conditions pass on every arm: minimum available memory21.4 GiB, zero swap
growth, zero preemptions, no thermal guard stop or recorded slowdown. The maximum
GPU temperature is76C. The compact layout remains80 bytes/key versus native68;
no model-admission improvement is claimed without its own test. Required summary
updates are inside model generation; offline scoring and metrics collection are
outside request timers.

C37 restores at20:24:36 UTC and removes all four holds. An independent reader
verifies the exact image, clean equal serving revisions, no experimental mounts
or environment and no reservations. All commands, responses' public metadata,
conditions and restoration receipts are retained in
`../2026-11-05-quality-matrix/runs/compact-pilot-window-02`. Raw generated text
remains private. Promoted configuration and release evidence are unchanged.

The subsequent direct-merge integration preserves coarse score/selection bits
and has separate component qualification; this pilot does not provide its
request timings. All13-task RULER coverage, held-out45-document confirmation,
coding repeats, serving-matrix and admission checks remain outstanding.
