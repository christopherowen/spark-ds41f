# Candidate-38 qualification on TP3 triangle

Base deployment: `fb89f24ad50fc3c79bb7b532e796c4f179d08bca`.
Intended variable: the complete selected candidate-38 image/recipe versus the
recorded spark-ds41f-1030 TP3 release. This qualifies the combined candidate;
it does not attribute any difference to an individual patch. The standard and
CopyDrafter recipes are reported separately, each with a boot and repeat boot.

Measurements are complete. Both recipes preserve the recorded greedy model
outputs and improve headline throughput against 1030. Retain the candidate for
review: C4 tail latency, mixed-prefill latency and small copy-recipe costs prevent
a blanket no-regression claim. [Decision](decision.md) and [normalized results](results.json)
retain these limits. The original candidate-36 service is restored on all three
nodes and the shared window is closed; no release is promoted.

The owner authorized building on dgx4 and testing on dgx1–3 using the normal
regression workflow. The current serving profile to restore is
`experiments/2026-10-26-spark-ds41f-candidate/candidate-36/tp3-dev.json` on the
same TP3 triangle. No topology change is intended.

## Image and inputs

The final candidate was already built on dgx4 at deployment `df50aa94`:
`sha256:213ba4df68d06665add7a26260c2321d55854a75b8a51ae71c72d0b97eac814b`.
Its dgx4 build receipt records 466 seconds, eight compile jobs and 78.1 GiB
minimum available memory. Current `build check` passes for the complete prepared
inputs. Reuse this immutable build and its completed image/release-test gates;
copy it unchanged through `bin/spark build distribute`, then independently check
all three image IDs, source labels, layer lists and Config hashes.

Source inputs and profiles remain those of
[the selected candidate](../2026-10-26-spark-ds41f-candidate/candidate-38/README.md).
No source overlays, effort ceiling or new tuning variable is added.

## Workloads and commands

Publish one commit, open a lab window against the verified serving profile, then
synchronize all three clean node checkouts to that exact published commit using
`bin/spark cluster sync --revision COMMIT --apply` with the restore profile named.
The specs deliberately contain no sync job that could move to a later commit.
Inspect start plans' live and shape lines before the first boot.

Run `functional-default.json` and `functional-copy.json` first. It retains the existing candidate's functional
matrix and CopyDrafter probe, repeats each profile on a fresh boot, and collects
all-rank image/state/log receipts after each boot. Review every actual script
exit, request/digest result and rank log before proceeding; an overall lab exit
is insufficient.

```sh
python3 scripts/lab.py run experiments/2026-11-03-candidate-38-tp3-qualification/functional-default.json --production-config experiments/2026-10-26-spark-ds41f-candidate/candidate-36/tp3-dev.json --keep-open
python3 scripts/lab.py run experiments/2026-11-03-candidate-38-tp3-qualification/functional-copy.json --production-config experiments/2026-10-26-spark-ds41f-candidate/candidate-36/tp3-dev.json --keep-open
python3 scripts/lab.py run experiments/2026-11-03-candidate-38-tp3-qualification/release-default.json --production-config experiments/2026-10-26-spark-ds41f-candidate/candidate-36/tp3-dev.json --keep-open
python3 scripts/lab.py run experiments/2026-11-03-candidate-38-tp3-qualification/release-copy.json --production-config experiments/2026-10-26-spark-ds41f-candidate/candidate-36/tp3-dev.json
```

The performance specs run `release-tp3`, `single-stream` and `sampled` on the
same boot of each arm, then repeat that arm on a fresh boot. Each profile has
its own table and paired verdict; different profiles are not pooled as repeats.
Use the immutable TP3 suite references under
`manifests/suites/2026-10-30-spark-ds41f-1030-tp3{,-single-stream,-sampled}`.
The released image is never remeasured. Retain suite conditions, boot/startup
memory, telemetry, all failures and the final serving restoration checks.

Judge fixed-row engine cost, paired one-/four-/eight-stream throughput, sampled
throughput, prefill through one million tokens, mixed-prefill stalls and first-token
latency against the recorded release and its A/A floor. Inspect confidence intervals
and the candidate's own repeat before calling a regression. A failure or unresolved
regression keeps the candidate unqualified; record it before deciding any follow-up.

## Preflight observations

All three nodes initially have clean checkouts at `3adababa` and the same running
candidate-36 image `sha256:b0522146640a7d852c017ec8c99423224492b7c2c4951eb3a64005ab4555ab11`.
Live doctor matches the restore profile with two existing recommendations:
management ports off wireless and model-weight defragmentation on dgx2/dgx3.
Do not change host settings within the comparison. No hold was present on dgx1
or dgx4 during initial read-only inspection. The site's ignored node map is
preserved and is not replaced with this workstation's same-named file.

## Execution and functional evidence

Executed from published deployment `cb657ab1f5dba4e7cca48a48b0c2fe4e8d4a711f`
on dgx1–3, with clean detached checkouts and the site's existing node map.
The image above was distributed unchanged from dgx4; all three image IDs,
layer lists and Config hashes agree. Window `dgx1-1793730237` opened at
2026-11-03 18:23:57 UTC against the candidate-36 restore profile above.

Both functional phases completed. Every ordered script and collector exited
zero: 38 for the standard recipe and 12 for CopyDrafter, across two fresh boots
of each. All twelve complete rank logs have no recorded CUDA, gate-logit,
assertion or engine-death faults. The copied archives and individual files are
hash-bound by the receipts under [runs/2026-11-03-dgx1-3](runs/2026-11-03-dgx1-3/).
`analyze_functional.py` verifies the actual exits, requests, summaries, memory,
runtime identity and repeat boots; seven corrupted-evidence controls were
rejected even after their file and archive hashes were refreshed.

| Recipe / boot | API ready | Minimum available GiB: dgx1 / dgx2 / dgx3 |
| --- | ---: | ---: |
| Standard, first | 118.3 s | 8.034 / 9.820 / 11.017 |
| Standard, repeat | 101.4 s | 7.617 / 10.953 / 10.901 |
| CopyDrafter, first | 114.3 s | 7.149 / 9.418 / 10.272 |
| CopyDrafter, repeat | 105.7 s | 6.546 / 10.146 / 9.744 |

The standard matrix covers wide greedy token/logprob determinism, the 16K
prompt, distinct streams, API controls, fixed-length TTFT traces, sink probes,
prompt logprobs, images and CED cross-shape repeats. Deterministic summaries
repeat across boots. The CopyDrafter token/logprob, seeded-sampling and image
records agree with the standard recipe on both boots. Its copy/prose/edit/tool
probe completed 48 requests and 43,023 output tokens per boot with zero errors;
the copy category produced about 16.03 tokens per engine step, exercising wide
copy verification. The original harness retains aggregate probe counters rather
than full generated text, so this is not a full-token correctness proof for all
copy-probe outputs.

Seeded sampling remains batch-dependent: the c8 digest differs from isolated
generation, while the isolated repeat agrees. All three rows exactly match
saved candidate-17 standard and CopyDrafter records. The sampled c8 penalties
control also changed between boots, as previously documented. These observations
are retained as limitations; they are not described as batch invariance.
The legacy memory extra additionally read dgx4 without running GPU work there;
its observation is recorded separately from the three TP ranks.

## Performance status

The standard recipe's two performance boots are complete: 26 ordered suite
steps, all exiting zero. Their 2,460 closed-loop/Poisson request-table records
have zero failed, rejected, foreign or approximate-timing requests and all
produce 256 tokens. Prefill has all 15 successful samples per boot; every stall
point reports three successful repeats. The two model runs have 256/256
identical greedy responses and identical recorded top-20 distributions over
65,536 scored positions (KL zero); all selected and top-20 log probabilities
are finite.

The normal tables include the candidate's repeat and the recorded A/A floor.
Their paired rows average the two candidate boots. No headline verdict is worse;
the C4 p99 direction below remains unresolved. Each decision margin includes
the protocol's within-run interval and boot allowance.

| Metric | Change vs recorded 1030 | Decision margin | Verdict |
| --- | ---: | ---: | --- |
| Fixed-row grid step time | −16.64% | 0.52% | better |
| Greedy decode C8 | +3.26% | 0.60% | better |
| Greedy decode C4 | +4.23% | 0.58% | better |
| Greedy decode C1 | +4.93% | 0.32% | better |
| Sampled decode C16 | +3.07% | 1.50% | better |
| Sampled decode C8 | +3.86% | 1.39% | better |
| Sampled decode C1 | +4.95% | 0.31% | better |
| Prefill 32K / 131K / 262K | +4.22% / +4.57% / +4.21% | 1.07% / 0.80% / 0.74% | better |
| Prefill 524K | +3.46% | 0.51% | better, reference caveat below |
| Prefill 1M | −0.11% | 0.71% | same, reference caveat below |

Startup minima (dgx1/2/3) are 6.903/10.369/10.491 GiB and
7.342/10.426/10.407 GiB; readiness is 101.4/103.5 seconds. Runtime minima over
the three suites are 6.56/10.35/10.26 GiB and 6.86/10.14/10.18 GiB. No suite
halted, grew swap, recorded thermal slowdown or failed to restore its fan state.

The [native timing records and tables](runs/2026-11-03-dgx1-3/performance-default/)
are copied byte-for-byte from the completed phase. The analysis verifies its
60 archived files, exact source/image identity, distinct repeat boot, every
step exit, requests, finite scores, memory and conditions. All eight
[semantic corruption controls](runs/2026-11-03-dgx1-3/performance-audit-controls/results.json)
are rejected for the expected reason with refreshed file and archive hashes.
The executed control source is archived beside that receipt. Private model
corpora and raw logs retain their hashes in the source receipt.
One offline model comparison began before its two file transfers completed and
exited on truncated input. Both transfers then completed successfully; a separate
comparison and the complete archive audit passed. Its failed collection-attempt
receipt is retained with the controls; it was not a failed GPU measurement.

The focused confirmation and CopyDrafter's two performance boots are complete.
No recipe, lock or published baseline has been promoted or rewritten.

The standard repeat's C4 TTFT p99 is 322.96 ms against the reference's
310.59 ms (+3.98%, a direction-only flag without an interval); the first boot
is 316.55 ms (+1.92%). Paired median TTFT is about 2.86% faster on both boots,
and mean, p90 and p95 are also faster. The repeat's two slowest requests are
adjacent indices 74/75, present in all runs, so unmatched prompts alone do not
explain the flag. Preserve it while checking repeatability.

The [archived 1030 A/A pair](runs/2026-11-03-dgx1-3/historical-1030-c4-aa/receipt.json)
provides context: its C4 p99 is 310.59/322.91 ms (+3.97%), with the same C4
workload and release image `7918fd61` on all three ranks. The first archived
point's parsed JSON equals the frozen reference exactly; their serialization
differs. Candidate 38's flagged repeat is 322.96 ms. This is one historical
repeat pair, not a calibrated p99 confidence bound. The frozen reference and
the candidate's original direction flag remain unchanged.

The [client compatibility check](runs/2026-11-03-dgx1-3/client-compatibility/client-compatibility-check.json)
records the client-code differences since the reference. Model scoring, engine,
prefill, prompt, statistics and Prometheus modules are unchanged. The stream
reader now also counts usage-only/control-token chunks. Independent server
generation counters corroborate the observed window-throughput changes within
0.039 percentage points across all 14 point/boot comparisons. In each C1
comparison, all 72 paired requests have equal first-token counts and chunk
counts. The sampled seed-default addition is inactive here because the suite
already specifies seed 0. Raw timing directions and the original reference
are retained.

`c4-tail-confirmation.json` adds two fresh boots of the identical standard
recipe after the main standard phase, before CopyDrafter performance. Its
`c4-tail-suite.json` copies the release suite's first four steps unchanged:
model/scoring, grid, C8 and C4. This retains the same model/scoring and grid
conditioning before C8 supplies the preceding steady load. The earlier
published two-step plan was expanded before execution; no run was discarded.
Compare each native step with its
frozen reference and retain both new runs, including any adverse result.
The lab's own two-arm table is an A/A check between the candidate boots,
not a released-build comparison. These workload definitions are published,
then exported byte-for-byte to the named private control path; the runtime
checkouts remain clean at `cb657ab1`, and the runner records the separate
workload commit and file hashes. The additional phase keeps the window open;
the original final CopyDrafter phase still restores candidate 36.

Both focused boots completed all four steps successfully, with the exact
published prefix workload and unchanged runtime source/image. C4 p99 is
314.90/324.10 ms (+1.39%/+4.35% against the frozen reference); the second
confirmation repeats the adverse direction. Paired median TTFT improves
2.87%/3.10%, and paired C4 decode improves 3.93%/4.09%. This retains a
tail-latency concern without establishing a candidate-specific regression:
the archived released repeat is 322.91 ms, and neither p99 comparison supplies
an interval. Do not substitute that repeat as the reference or erase the flag.
The [native confirmation records](runs/2026-11-03-dgx1-3/performance-c4-tail/)
include both runs. Startup minima are 7.067/10.241/10.161 GiB and
7.495/10.149/10.250 GiB; runtime minima are 7.64/10.53/10.53 GiB and
7.95/10.90/10.77 GiB. Neither run halted, grew swap, recorded thermal
slowdown or failed to restore its fan state.

The two original released model files were recovered from the recorded 1030
TP3 runs during the idle gap, without sending inference requests. Their source
sizes, timestamps and hashes were stable during transfer; the archived suite
identity names the released image on all three ranks. The
[historical model comparison](runs/2026-11-03-dgx1-3/historical-model-comparison-standard-tail/)
checks the same prompt and scored-output contexts against each released boot.
All four standard/confirmation model runs match both released records:
256/256 greedy outputs, no finish-reason changes, identical selected-token
probabilities and recorded top-20 distributions over 65,536 positions (KL zero).
Every selected and top-20 log probability is finite. The complete model corpora
remain private; the executed comparison source, summaries and hash receipts are
published. This supplements the immutable main analysis, whose release-reference
limitation refers to the published reference's omitted model text.

The [previous-candidate comparison](runs/2026-11-03-dgx1-3/previous-candidates/)
uses the archived candidate-36 and candidate-37 standard runs from November 1,
including the same three full suite definitions despite their `screen-tp3` run
names. Every recorded step exited zero, all ranks name one image per candidate,
the clients were clean, and conditions show no halt, swap growth or thermal
slowdown. Their client `servebench` sources are identical to the current ones.
Their [recorded profile hashes](runs/2026-11-03-dgx1-3/previous-candidates/source-validation.json)
also match the raw files at the recorded client commits, and every rank's
checkout and image agree with the expected archived candidate identity.
Each prior candidate has only one recorded boot here; these are supplementary
comparisons, not repeat-boot qualification of those candidates or their copy path.

| Metric | Change vs candidate 37 | Change vs candidate 36 |
| --- | ---: | ---: |
| Fixed-row grid step time | −0.01% | −3.57% |
| Greedy paired decode C8 / C4 / C1 | −0.16% / +0.09% / +0.04% | +0.44% / +0.40% / −0.06% |
| Sampled paired decode C16 / C8 / C1 | +0.50% / +0.51% / −0.00% | +0.04% / +0.96% / −0.04% |
| Prefill 32K / 131K / 262K / 524K / 1M, first boot | −0.21% / −0.01% / −0.00% / +0.07% / +0.07% | −0.91% / −0.83% / −0.64% / −0.65% / −0.55% |
| 131K prefill beside 3 streams, first boot | −0.14% | −1.41% |

The normal throughput headlines are all `same` versus candidate 37. Versus
candidate 36, the table flags first-boot prefill at 32K, 131K and 524K, and
131K prefill beside three streams; the repeat has the same direction at these
points. Candidate 37 is already essentially level with candidate 38 on them.
Individual adverse grid/latency rows remain in the complete comparison:
candidate 37's C4 p99 is 297.24 ms, so the main candidate-38 boots are
6.50%/8.65% higher; paired C4 median TTFT is 1.09%/1.87% higher, with the
repeat crossing its within-run interval. The C36 comparison also flags some
individual grid shapes and first-boot short-request latency during prefill.
The aggregate grid or decode headline does not erase those observations.
The prior native identity is used here because the older candidate README and
summary JSON contain copied metadata; those summaries are not deployment proof.

The recorded reference's 524K/1M prefill extension (`c06790ef`) retained the
original shorter suite's outer elapsed time. Its actual stored prefill samples
sum to 1,230.88 seconds rather than that index's 523.6 seconds. Comparisons use
the per-size samples, with the discrepancy recorded in
[reference-duration-note.json](runs/2026-11-03-dgx1-3/reference-duration-note.json).
The largest two sizes also have different actual prompt counts (reference /
candidate first boot: 533,290 / 522,774 and 1,000,466 / 994,697). Keep these
warnings alongside the token-rate comparisons; the reference is unchanged.
The [native extension receipt](runs/2026-11-03-dgx1-3/reference-extension-native.json),
recovered from dgx1, contains exactly the reference's six extended samples and
its original 17:33:45 UTC start. It used only the two large sizes. The generator
seeds documents by size index, so its indices 0/1 select different documents
from the full suite's indices 3/4. These points are token-rate comparisons over
nearby lengths rather than exact prompt matches.
The standalone extension also lacks its own all-rank deployment identity and
conditions block; the earlier suite's metadata does not establish these for the
later measurement. Preserve that provenance limit when using the two points.

Local validation is recorded in [local-validation](runs/2026-11-03-dgx1-3/local-validation/):
the promoted configuration and both candidate dev profiles pass `bin/spark doctor`,
and all 626 repository tests pass without skips using the hash-pinned bench-client
dependencies. The initial command named an absent worktree virtual environment;
the next check passed with five validator-dependent skips. Both attempts are
retained alongside the complete successful check. They did not affect GPU work.
The [integration check](runs/2026-11-03-dgx1-3/local-validation/integration-summary.json)
again passes all 626 tests without skips after bringing current main into the
report branch. Its separate TP4 replay and AgentX stub changes do not alter the
tested candidate, profiles, references or `servebench` sources.

During the CopyDrafter repeat, workstation SSH to dgx1 stalled and a bounded
connection timed out around 00:45 UTC on November 4. ICMP still worked;
dgx2/dgx3 remained reachable, and dgx2 could reach dgx1 through both its LAN
and SSH. The lab heartbeat and serving containers continued, and the delayed
original status eventually returned successfully. Monitoring switched through
dgx2 with the existing dgx1 host key. No node, benchmark input, host setting or
job was changed or restarted. The repeat model/scoring step subsequently
completed successfully. The [access incident receipts](runs/2026-11-03-dgx1-3/workstation-access/)
retain this observation without attributing an unconfirmed transport cause to
the candidate.

## Completed CopyDrafter performance

Across both boots, all 26 ordered suite steps completed successfully. Their 2,452
closed-loop/Poisson request-table records have zero failed, rejected, foreign or
approximate-timing requests and all produce 256 tokens. Both boots retain all
15 successful prefill samples and three successful repeats at each stall point.
The [native records and audit](runs/2026-11-03-dgx1-3/performance-copy/) verify
all 60 archived files against the published runtime, profile and image.
All 18 first-boot preview files match the final archive byte-for-byte.

| Metric | Change vs recorded 1030 | Decision margin | Verdict |
| --- | ---: | ---: | --- |
| Fixed-row grid step time | −16.31% | 0.66% | better |
| Greedy decode C8 / C4 / C1 | +2.98% / +4.01% / +4.67% | 0.71% / 0.88% / 0.31% | better |
| Sampled decode C16 / C8 / C1 | +2.93% / +4.14% / +4.69% | 1.89% / 0.99% / 0.31% | better |
| Prefill 32K / 131K / 262K | +4.62% / +4.85% / +4.50% | 1.06% / 0.61% / 0.63% | better |
| Prefill 524K | +3.77% | 0.37% | better, reference caveat above |
| Prefill 1M | +0.10% | 0.58% | same, reference caveat above |

Readiness is 105.5/104.7 seconds. Startup minima (dgx1/2/3) are
6.883/9.703/9.957 GiB and 6.994/10.163/10.247 GiB; runtime minima across
the three suites are 6.58/10.26/10.20 GiB and 6.91/10.20/10.11 GiB.
Neither boot halted, grew swap, recorded thermal slowdown or failed to restore
its fan state. Both model runs also
[match both archived released boots](runs/2026-11-03-dgx1-3/historical-model-comparison-copy/):
256/256 greedy outputs and identical selected-token probabilities and recorded
top-20 distributions over 65,536 positions, all finite, KL zero.

No headline verdict against 1030 is worse, but three adverse raw rows remain.
C4 p99 is 321.95/324.46 ms (+3.66%/+4.46%). With three streams beside a
131K prefill, the wait before the first chunk is 0.512/0.522 s against
0.438 s; the repeat is flagged worse (+19.14%, interval 10.46%). Standard's
corresponding waits are already 0.501/0.488 s. Copy's worst decode wait during
that prefill is 1.700/1.495 s against 1.103 s, with wide intervals and an
inconclusive verdict. Its p90 decode step is faster on both boots and the short
probes' first-token latency improves; these gains do not erase the adverse rows.

The [direct recipe comparison](runs/2026-11-03-dgx1-3/recipe-comparison/) uses
each standard boot separately as the reference for both copy boots, retaining
each profile's repeat independently. Paired C4 decode changes −0.32%/−0.31%,
greedy C1 −0.37%/−0.13%, and sampled C1 −0.34%/−0.15%, depending on the
standard reference boot. The former reference flags these small losses; the
latter is within its decision margins. The grid table reads +0.39%/+0.80%,
with margins 0.62%/0.61% and same/worse verdicts. Prefill is essentially level;
the 524K direction also depends on the reference boot. This compares complete
recipes, including copy's batch/capture sizing, rather than isolating patch 0677.
Keep both reference choices and all raw rows; do not call the copy recipe free.

## Restoration

Window `dgx1-1793730237` closed at 2026-11-04 02:02:46 UTC after candidate 36
became ready at +100.4 seconds. The
[independent restoration records](runs/2026-11-03-dgx1-3/restoration/)
confirm image `sha256:b0522146640a7d852c017ec8c99423224492b7c2c4951eb3a64005ab4555ab11`
and the original `candidate-36/tp3-dev.json` profile on all three nodes, with
matching Config/layers, healthy fresh containers and zero restart counts.
All three checkouts remain clean and detached at the tested deployment commit.
Complete rank logs are untruncated, hash-verified and free of the checked fault
patterns; their content stays private. Fan states match the initial service,
no lab clock marker remains, and both hold-file names are absent on dgx1.
Independent live doctor passes with the same wireless/weight-fragmentation
recommendations recorded before testing.
