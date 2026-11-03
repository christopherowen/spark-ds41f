# Candidate-38 qualification on TP3 triangle

Base deployment: `fb89f24ad50fc3c79bb7b532e796c4f179d08bca`.
Intended variable: the complete selected candidate-38 image/recipe versus the
recorded spark-ds41f-1030 TP3 release. This qualifies the combined candidate;
it does not attribute any difference to an individual patch. The standard and
CopyDrafter recipes are reported separately, each with a boot and repeat boot.

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

The focused confirmation is running; CopyDrafter's two performance boots remain
pending. No recipe, lock or published baseline has been promoted or rewritten.

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
