# Expert-scale recorded results

The isolated lossless scale-storage change completed two fresh serving boots on
DGX1/2/3. It preserves the recorded model outputs exactly and reduces the reported
model-load CUDA allocation counter by about 1.32–1.33 GiB per rank. It does not
produce a repeatable serving speedup. No release promotion is proposed; the
current C36 stack already includes the scale representation.

The causal variable is physical storage of the existing expert scale codes.
The reference is the immutable 1030 record, never a rerun of that released image.
Execution commit `3c7b455eec27c436f76f08ed28efb30135c86f4f`, window
`dgx1-1793885062`, runtime tree `c3a9bf4be9b672f11c99f0c3098778a901dbae99`,
and full image identity are pinned in `recorded-results.json`. The controller
finished with exit 0 and restored C36 at 2026-11-05 17:15:54 UTC. All three ranks
ran the same image/layers and had a distinct container start on the second boot.
Companion suites retained their primary boot's identity. These are fresh-container
boots; compilation and filesystem caches were not claimed cold.

| Workload | First boot output tok/s | Repeat boot output tok/s |
| --- | ---: | ---: |
| Greedy C8 | 157.82 | 157.31 |
| Greedy C4 | 120.807 | 120.64 |
| Greedy C1 | 62.607 | 62.46 |
| Sampled C16 | 154.33 | 153.68 |
| Sampled C8 | 154.743 | 153.043 |
| Sampled C1 | 61.383 | 61.247 |

These are window output rates, not estimates from request sizes. Detailed 95%
intervals, TTFT, TPOT, engine counters, decode grids, prefill, stalls and native
condition samples are preserved in the analysis and raw archive. TPOT is not an
ITL-tail measurement; unrecorded ITL tails, KV occupancy and raw cache hit/miss
counts remain null. The Poisson rate is 119.067 on both boots with a wide ~17.3%
95% interval; it does not establish a performance benefit.

All four candidate/reference boot pairings are retained. Each is evaluated with
both candidate and saved-reference repeat floors using the existing comparison
method, with no new tolerance. These eight views reuse four pairings and are not
independent experiments. Paired C8 decode throughput changes are -0.7116% and
-0.2984% across the two reference choices; six views show no proven change and
two flag a slowdown. The grid includes favorable and unfavorable flags across
pairings and does not establish a robust win. Smaller/sampled decode and eligible
cold-prefill points have no repeatable win. The 524K/1M reference extension has
no eligible saved repeat, incomplete independent conditions and actual-token
warnings; it supplies no promotion claim. No old observation was refreshed.
The saved C36 client timings use another stream collector and have only one boot.
Compatible engine/prefill context remains in the detailed report, with no
single-component attribution or incremental C36 claim.

Both boots report 92.62 GiB on DGX1/2 and 92.63 GiB on DGX3. The baseline's saved
`/evidence/summary` reports 93.95 GiB per rank. The three files implementing that
framework allocation counter are byte identical between the reference and this
candidate; their hashes are recorded. This is a model-load CUDA allocation
counter, not whole-host RSS or all physical GPU memory. The baseline has no
separate saved per-boot allocation range, so no allocation confidence interval
is invented. The KV pool remains 4.9 GiB; additional KV capacity was not tested.
Startup host MemAvailable minima were 7.41/10.054/10.619 GiB on the first boot
and 7.22/9.424/10.081 GiB on repeat. Full suite samples and all thermal/swap/floor
guards pass; these host observations are not a causal 1.3 GiB host-headroom claim.

Each rank's corrected full gate passed 7,337 tests with 355 skipped, zero failed
or errored across 132 files; the retained component gate passed 118 tests per
rank for the exact same image. Each boot matches all 256 saved generated
sequences and 65,536 returned score vectors exactly. This covers the returned
scores, not an unmeasured full-vocabulary distribution. Quality, prefix reuse,
concurrent long-context admission, token/logprob comparisons across traffic and
greedy/gathered paths, deterministic controls and vision all pass. The ordinary
prose/copy/edit/tool smoke ran 12 requests per category with zero recorded errors.
Exact scale/kernel tests and these output checks jointly support losslessness
within the tested implementation and workloads.

`../runs/scales-recorded-two-boots/archive-manifest.json` pins all 187 original
files by byte length and SHA-256. The archive includes 184 original non-model
files plus their three transfer manifests, preserving successes, adverse rows,
commands, errors, engine logs, rank gates, raw workload records and the queued
handoff acknowledgement/watcher. Archive timestamps/ownership are normalized;
member bytes are verified exactly. The three large model reports remain at the
pinned original and storage snapshot paths, with sizes and hashes in the manifest;
the compact exact-comparison reports are included in the archive. No weight bytes
or duplicate large model report are added to Git. Original first-primary records
under the incompatible collector remain unchanged in `first-primary-results.json`.
The first offline analysis failed on its working directory; its original log is
retained alongside the successful second attempt and does not become a run failure.

The analysis ran only on the storage utility host, inside the recorded two-CPU,
4 GiB limit. `analysis/` preserves the exact analyzer sources. To reproduce on a
remote CPU host, extract the archive outside Git to a private directory, run from
this repository root and pass the extracted DGX1 `projects/spark-ds41f` directory:

```sh
python3 experiments/2026-11-04-lossless-memory/expert-scales/analysis/analyze_component_performance.py \
  --snapshot /PRIVATE/dgx1/projects/spark-ds41f --component expert-scales \
  --receipt results/private/lossless-memory-scales-round4/receipt.json \
  --reference-repeat-root experiments/2026-11-04-lossless-memory/runs/recorded-serving-references/results/private/suites \
  --out /PRIVATE/performance.json
```

For a complete exact-model recheck, restore the three separately pinned model
reports to their listed relative paths. Performance analysis uses the compact
reports and does not substitute model equality for timing evidence. The native
auditor's three gate functions are AST-identical to the execution commit; hashes
are recorded in `audit-methods.json`. The source-level C36 Engram selection audit
corrects the earlier assumption that packed-reader support meant activation:
C36's model registers separate sources. Engram remains its own unbuilt candidate,
and observer controls must pass in an owned window before its GPU work.
