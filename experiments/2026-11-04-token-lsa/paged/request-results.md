# Paired request results, 2026-11-05

Neither selector meets the generated-token acceptance criterion. The aggressive
preset matches 6/9 and the wide preset 8/9 on **both fresh boots**, at serial and
four-concurrent qualification. Aggressive changes all three long notebooks; wide
changes the 262144-token notebook. Both native boots match 9/9; the saved native
serial reference and within-boot concurrent repeat also match 9/9. No changed-token
quality superiority has been established.

Runtime source: `df2af0bc9cf505f7b7ff9bb3d99768c236428acb`, immutable unreleased
C36 image `sha256:3795255f485ea5c1a9db54c742921dd9b2917daad0799361e5c6f56a70119713`.
All arms use the same scratch-read guard and 73-byte admitted cache geometry.
Order: native-A, aggressive-A, wide-A, wide-B, aggressive-B, native-B on sp5–sp7.
The candidate pays for page-summary updates inside requests. Native references,
raw-ID capture and token comparisons run separately. No timed request asks for
raw IDs or runs a comparison.

Each request generates 128 tokens, greedy/seed0, from the fixed notebook corpus.
Cold prompts have fresh cache salts; three repeats reuse each prompt per boot.
Values below average the two per-boot medians. This synthetic screen does not
qualify production-agent traffic or universal token equivalence.

Cached aggregate output tokens/s:

| Context / concurrency | Native | Aggressive 64/512 | Wide 1536/native source |
|---|---:|---:|---:|
| 131072 / 1 | 66.65 | 75.64 | 66.32 |
| 131072 / 4 | 155.91 | 168.78 | 155.90 |
| 262144 / 1 | 69.25 | 75.52 | 70.17 |

Cold batch wall time (first request send through final response), seconds:

| Context / concurrency | Native | Aggressive 64/512 | Wide 1536/native source |
|---|---:|---:|---:|
| 131072 / 1 | 20.946 | 20.009 | 20.958 |
| 131072 / 4 | 81.202 | 78.023 | 81.366 |
| 262144 / 1 | 42.457 | 39.045 | 43.070 |

Cached throughput ratios and uncertainty:

| Preset | Context / concurrency | Candidate / native | Joint95% interval |
|---|---|---:|---|
| aggressive-fast | 131072 / 1 | 1.135 | [1.030, 1.241] |
| aggressive-fast | 131072 / 4 | 1.083 | [0.867, 1.336] |
| aggressive-fast | 262144 / 1 | 1.090 | [0.902, 1.286] |
| dense-1536-fast | 131072 / 1 | 0.995 | [0.898, 1.093] |
| dense-1536-fast | 131072 / 4 | 1.000 | [0.924, 1.089] |
| dense-1536-fast | 262144 / 1 | 1.013 | [0.950, 1.079] |

There are only two independent boots per arm. Intervals use df1 Student-t
component 97.5% intervals and a Bonferroni joint 95% ratio bound; normality is
unverified. Broad/undefined intervals remain visible. Individual cached repeats
are not treated as independent boots. All raw samples, token mismatches, actual
cache-hit counts, output counts/digests, commands, and cold/cached latency ratios
are retained in [the complete receipt directory](../runs/request-diagnostic-01/)
and [the comparison JSON](../runs/request-diagnostic-01/diagnostic-comparison.json).

All six timed boots complete without memory-guard events, swap growth or thermal
slowdown. Lowest measured MemAvailable is 7.02 GiB across the six arms. Fan-curve
services are absent on all three nodes; the repository cooling controls still
bring every node below 55°C before each timed boot. Those warnings and thermal
samples remain in each conditions receipt.

The window restores tracked C37 at `2026-11-05T07:46:26.984127+00:00`. An independent
all-node verification confirms the expected image, clean checkouts, no experiment
mounts/environment, and removed reservations. Raw generated IDs remain node-local.

Conclusion: the aggressive selector has modest diagnostic request gains while
failing generated-token equivalence. The wider selector is effectively flat and
still fails its only long case that prunes keys. Neither is a promotion candidate.
The request-level performance goal with identical output tokens remains unmet.


CPU decoding review of the stored captures clarifies the mismatch: the first
changed aggressive tokens are at one-based positions 16, 10 and 74 for 32K,
128K and 256K contexts. The wider preset shares the 256K divergence. Observed
changes are in reasoning wording. Every long capture stops at the 128-token
length cap before closing its reasoning span; all retain the correct registry
code word in that prefix. These receipts establish generated-token divergence,
not an incorrect final answer. Completed-answer equivalence and relative quality
remain untested. The stored corpus confirms temperature 0 and seed 0 for every
request; both qualification and timed requests preserve those settings. See
[the metadata summary](../runs/request-diagnostic-01/output-difference-summary.json).
Decoded text and raw IDs remain outside Git.
