# C41 measured round

**Reject C41's simplification; retain C40 as the corrected default-enabled base.**
Correctness passes and aggregate serving throughput is neutral. The wider
no-material-core-regression criterion fails. No old image is rebuilt or remeasured,
and production was independently verified restored to C36 before this analysis.

## Requested four-build report

Output tokens/s; slash separates the fresh boot and repeat. Values are window
means, not confidence limits. The raw mean changes versus 1030 below are
context, not evidence that the current patch earned those inherited gains.

| Build | C1 | C4 | C8 | C8 mean vs 1030 | Performance verdict |
|---|---:|---:|---:|---:|---|
| 1030 | 62.63 | 121.21 | 158.60 | reference | Immutable release reference |
| C36 | 65.46 | 125.67 | 163.03 | +2.79% | Higher saved means; one compatible boot |
| Previous C40 | 65.38 / 65.37 | 126.10 / 126.17 | 163.41 / 163.59 | +3.09% | No proven serving gain over C39; correctness fix retained |
| Current C41 | 65.36 / 65.34 | 126.08 / 126.10 | 163.69 / 163.53 | +3.16% | Serving neutral versus C40; worse core costs, reject simplification |

Paired C41 over/under against both C40 boots, preserving all four pairings:
C1 −0.080% to +0.001%, C4 −0.066% to −0.025%, C8 −0.042% to +0.078%.
All are **no proven change** under both candidate and saved-reference repeat
floors. Sampled C16 is −0.479% to +0.409% and no proven change; its raw outputs
are 159.57 / 161.36 tokens/s. No single favorable boot supports an improvement.

Against C36's one saved boot, paired C8 is +0.241% to +0.289%, still no proven
change; C1 is −0.372% to −0.308%, with a worse headline under the existing
candidate floor. C36-specific repeat variation is unknown. Against both C39
boots and both C38 recipes, all pairings, adverse rows and separate reference
floors remain in [recipe-comparison.json](recipe-comparison.json).

## Purpose and result

C39 delivered the owner's default-enable behavior, but its optional overlap
performance claim did not establish a serving gain over either saved C38 recipe.
C40 corrected an independent scratch-read issue, with neutral aggregate serving
performance; it was correctness work. C41 was declared a simplification before
measurement: remove preparation-stream/event machinery, with no speedup required,
but no material incremental core/serving regression allowed.

C41 engine-grid cost is **+0.206% to +0.409%** across all four raw C40 pairings.
Seven of eight pair-and-floor views are worse; the smallest pairing is same
under the saved-reference floor. Cold 524K prefill throughput is **−0.181% to
−0.667%**; four of eight views are worse and four same. These eight views are
not independent experiments. The existing comparison method/tolerances are
unchanged, with boot-two non-paired views separately reproduced in
[review-floor-supplement.json](review-floor-supplement.json).

The [focused review](result-review.json) recommends rejection and no further
speculative correction. Grid bypasses the changed calls; pure cold-prefill
also skips them in this harness. Attribution is unresolved, not grounds to
discard image-level regression. Serial preparation could expose work on eligible
drafting traffic, but no saved phase trace bounds a correction. Restoring
removed overlap returns to already-measured C40. Retaining C40 does not prove
that overlap benefits serving, and does not approve its complexity as a new
performance enhancement.

## Complete qualification and restoration

The exact execution commit was `0da35cc9668ff491065446b2a842fe996a73e80f`.
C41 vLLM head/tree are `38ade7f947cb09c550c2bce077286e467c7fed8d` /
`df6b6bc6841c1126b37be652c0e93c13debb8f42`; actual image is
`sha256:38ff08bb1eb2495cb934e008b6f31a482aaa238c08b959aaf72cf2cb5658d29c`.
It adds one scheduling simplification patch to C40. Default copy, kernels,
integer/float policy, safe envelope, older ring/D2H delivery and PR878 remain.
C41 pins C40's recorded four-image request surface; global defaults are unchanged.

Attempt 1 built/gated the image once: eight exits zero, exact 136-file coverage,
8,075 passed, 360 skipped, 14 XML reports and strict actual-image top-k
initialization checking with zero errors and no overlay. Both changed GPU
regressions passed. It yielded for a foreign request before any serving phase;
the actual exit 1 and closure error remain scheduled-sharing evidence.
Attempt 2 safely reused the exact immutable gate; no gate or serving point was
repeated. It launched 07:42:15 UTC on 2026-11-05. All 40 ordered functional
outcomes and six performance suites/26 ordered steps passed. Both fresh metric
boots completed; no request failures, rejects, foreign requests, length or
approximate-timestamp errors occurred. Model corpus digests match both saved
C38 recipes. All four full model pairings against each of C39 and C40 have
256/256 identical sequences, 65,536 identical score positions and zero KL.

Native performance ended 11:18:52 UTC; actual controller wait exit was zero at
11:21:04. Independent named live doctor and all-rank Docker image/profile,
running state, restart-count and stable-container checks passed at 11:21:53.
Both controller/supervisor processes were absent. Restoration profile is
`experiments/2026-10-26-spark-ds41f-candidate/candidate-36/tp3-dev.json`, image
`sha256:b0522146640a7d852c017ec8c99423224492b7c2c4951eb3a64005ab4555ab11`.
Hold/request/legacy request were null at that snapshot; reread actual state
before any subsequent window. No promotion or outstanding TP4 gate is bypassed.

## Adverse observations and limits

Minimum available host memory was **5.97 GiB**, 0.37 GiB below C40. Maximum
GPU temperature was 77 C, minimum headroom 10 C and maximum system temperature
88.8 C. All six native guards were clear, with zero recorded thermal slowdown,
power-cap duration and swap growth. Separately, seven repeat-release dgx3
samples reported software power-cap event mask 0x4. Preserve both records;
no duration or causal performance explanation is inferred. The event definition
is documented in [NVIDIA's NVML reference](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlClocksEventReasons.html).

The original passive observer SSH timeout and successful bounded read-only
checks are preserved. Native first-release telemetry spans 3,272 seconds with
3,271 samples on each rank, covering the later observer gap; no native
management-loss latch or runtime relaunch occurred. Aggregate samples do not
prove every management probe succeeded. No resets or policy changes were made.
[Telemetry review](telemetry-review.json) retains every rank/suite reason count.

Other adverse rows remain: mixed-prefill/first-chunk waits, tail percentiles,
per-shape grid changes and acceptance/repeat variation. C4 TTFT p99 on C41's
repeat is 3.60% above 1030, without a percentile interval. API-ready times were
108.0 / 106.8 s functional and 104.7 / 103.6 s performance. Selected preemptions
and prefix-cache hit rates were zero, so this is not cache-rich qualification.
No timed KV occupancy, raw cache hit/miss counts or ITL tails were recorded;
TPOT is not ITL. Release 524K/1M prompt lengths differ, the old extended outer
prefill duration is stale, its extension lacks independent identity/conditions,
and suite-entry rank identity is not a per-step layout CRC. Passing the saved
corpus does not establish general useful-task correctness.

## Evidence and reproducibility

[functional-audit.json](functional-audit.json),
[performance-analysis.json](performance-analysis.json),
[core-metrics.json](core-metrics.json) and the full comparisons retain positive
and adverse outcomes. All **181** native artifact hashes verify locally.
[publication-receipt.json](publication-receipt.json) maps **51** byte-identical
native timing/conditions/identity/control JSON publications to their source
hashes. Full raw logs/model text remain private in `final-export/raw` and
`raw.tar.gz`, with every file bound by `final-export/receipt.json`. The original
failed scheduling and operator receipts are preserved; none are overwritten.

The independent review bound the pre-final mutable report state; original bytes
are retained in `review-inputs/` and mapped by
[review-input-snapshots.json](review-input-snapshots.json). The root experiment
README remains its hash-bound source-review snapshot. Analysis commands and
helper hashes are recorded in validation.json. No old image was remeasured.
The recurring schedule stays paused; this round used direct waiting.
