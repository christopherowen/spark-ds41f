# Completed-answer pilot with the larger allowance

Runtime commit `65c6e62f88c89823e610f5f7e3f8a0c3b644be20`, unreleased C36 image
`sha256:3795255f485ea5c1a9db54c742921dd9b2917daad0799361e5c6f56a70119713`.
Both arms use temperature 0, seed 0, high-effort thinking and a shared allowance
of 65,536 generated tokens. The candidate is `native-source-fast.json`: dense
budget 64, native source/reindex selection and a 256-row workspace. Required
summary maintenance remains inside requests; grading stays outside serving.

All 48 responses stop naturally with completed final content. No response hits
the allowance or request timeout. Only final content is graded against the
pinned LongBench v2 answer key; thinking may differ.

| Fresh boot | Correct / independent documents | Total request wall time |
|---|---:|---:|
| Native A | 8 / 12 | 631.35 s |
| LSA A | 9 / 12 | 654.01 s |
| LSA B | 9 / 12 | 654.54 s |
| Native B | 8 / 12 | 631.42 s |

LSA gains question `67041f08bb02136c067cdb52` and loses none, in every boot
pairing. Accuracy is 75.0% versus 66.7%, an 8.33-point pilot difference. One
discordant document gives an exact one-sided paired p-value of 0.5. Repeated
boots do not increase the quality sample size beyond 12 documents. This is not
statistical evidence of general quality superiority; the 45 frozen, disjoint
held-out documents remain unseen.

Mean workload latency is **3.6% higher** with this LSA setting. Native/LSA wall
time is 0.9650, with a joint 95% ratio interval of [0.9540, 0.9763]. Useful
throughput, correct completed answers per second, is **8.56% higher** in this
pilot, with timing-only ratio interval [1.0731, 1.0982]. These two-boot intervals
assume normal boot means, which is unverified; they do not account for uncertainty
in generalizing the 12-document quality score. A higher useful-throughput point
estimate does not establish lower request latency.

The question previously truncated at 8192 tokens now completes correctly on both
boots of both arms. Native uses 23,999 generated tokens, taking 250.4/250.6 s;
LSA uses 23,838, taking 269.6/269.9 s. The larger allowance removes a real
completion failure. The former lower-budget, different-source-policy cohort is
retained separately and is not a speed comparator for this window.

Each native boot generates 39,767 tokens across the full pilot; each LSA boot
generates 39,867. All generation time counts. These request receipts do not
isolate kernel cost from speculative acceptance and changed reasoning paths.
No explanation of the latency difference is inferred from token counts alone.

Every boot passes the repository memory/thermal controls: no memory-guard event,
swap growth or thermal slowdown; minimum observed available memory is 7.11 GiB.
Tracked C37 is restored at `2026-11-05T10:59:32.411402+00:00`; independent checks
at 11:00:31 UTC confirm its image, all three clean checkouts, no experimental
mounts/environment and no remaining reservations. Promoted configuration and
immutable release evidence remain unchanged.

All commands, timings, conditions, final-choice receipts and restoration evidence
are in [the complete window](runs/pilot-window-02/). The
[comparison JSON](runs/pilot-window-02/comparison.json) retains every boot
pairing and interval. Generated text and token IDs remain node-local.

Conclusion: increasing the allowance was necessary. This operating point has a
repeatable pilot quality lead and higher correct-answer throughput, with slower
total request time. It is unqualified for promotion. The predeclared wider-source
probe tests whether more selective source work can recover latency while keeping
the completed-answer quality lead. Learned-metric preparation remains separate;
no fitted map has run in serving.
