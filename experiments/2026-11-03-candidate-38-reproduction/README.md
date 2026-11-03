# Candidate-38 reproduction

Base: `6d59eaebf4ae569075f88b0466051dfcca053699`; executed deployment:
`806c579763876bc042c8d44e7c10f42e59b6168f` (PR #866).
The owner explicitly authorized the coordinated sync, restarts, both reproduction
phases and restoration of candidate-37. Both reproduction phases completed in
window `spark-0013-1793717573`. Candidate-37 was restored on sp5–sp7 at 15:50 UTC;
independent identity and live-doctor checks passed. Both holds were released and
all investigation runners and bridges exited. The
[execution receipt](runs/2026-11-03-spaceport/execution-receipt.json) binds the
coordination, sync, restoration and preserved setup failures to their raw records.

## Loop result

Both fresh boots completed all ten requests, with exact agreement in generated
token IDs, finish/stop reasons and generated logprobs. All six rank observations
match the exact candidate-38 image; collected worker logs are untruncated.

| Request | Detector off | Default detector | Interpretation |
| --- | --- | --- | --- |
| Seeded active turn, effort 100 | Repeats to 1,024-token cap | Forced reasoning end, tool call, 238 tokens | Containment, not spontaneous recovery |
| Seeded active turn, effort 75 | Repeats to 1,024-token cap | Forced reasoning end, tool call, 257 tokens | Lower effort does not remove seeded sustainment |
| Clean audit, effort 100 | Natural stop, 4,666 tokens | Identical output | No onset observed in this fixture |
| Clean audit, effort 75 | Natural stop, 1,518 tokens | Identical output | No onset observed in this fixture |

These counts repeat exactly on both boots. There are 20 successful requests, of
which eight are clean-onset checks (the same audit fixture, effort/detector pairs,
and repeat boot), not 20 independent onset tests. The short seeded continuations
also sustain repetition; the effort-75 short case reaches its 64-token cap after
detection but before a forced reasoning end lands.

[Analysis](runs/2026-11-03-spaceport/loop-analysis.json) and
[receipt](runs/2026-11-03-spaceport/loop-receipt.json) retain request/output hashes,
matched scheduler interventions and the exact offline analysis command.
The experiment does not establish that longer unseeded sessions cannot loop,
and it does not validate an effort-85 ceiling.

## CopyDrafter result

The original fault did not recur on either fresh candidate-38 boot. Each boot
completed the same sequence and retained the exact min-match-8/dead-row-cutoff-0
configuration throughout:

| Stage, on each boot | Completed workload | Observed failures |
| --- | --- | --- |
| Copy probe | Three requests in each of prose, copy, edit and tool categories | 0 |
| Single-stream | 60-second warm-up plus 300 seconds in ten nonempty windows | 0 |
| Eight clients | Three samples of 8,192 tokens | 0 |
| Sixteen clients | Three samples of 16,384 tokens; eight active-request limit | 0 |

The two concurrency stages generated 147,456 tokens across the two boots.
Both all-rank runtime collectors passed. The final independent
[analysis](runs/2026-11-03-spaceport/copy-analysis.json) and
[receipt](runs/2026-11-03-spaceport/copy-receipt.json) bind these
workloads to their raw receipts, same-boot source identities, complete rank logs
and actual per-script exits; the overall lab exit alone is not sufficient.

The single-stream observations were 60.89 ± 3.12 and 60.92 ± 3.18 tokens/s
(suite-reported 95% confidence intervals over ten windows). These are two runs
of the same diagnostic configuration, not a comparison with a released build.
The harness retained existing site warnings for a 4K kernel, missing memory-saver
configuration and missing fan-control service. No settings were changed to hide
those warnings. Normal release/performance qualification remains separate.

The unchanged image has no per-step rank-layout recorder. The result is
non-reproduction of the fault in this workload, not proof that every possible
layout agrees or that all inference is correct. The exact-source CPU control
below separately demonstrates removal of the known host-budget mismatch.
The original copy probe retains aggregate counts, not generated text or token
sequences; matching totals across boots do not establish exact output agreement.

## Executed method

Both phases used the unchanged final candidate-38 image `213ba4df68d0`, vLLM head
`1f33102547e4da5c64712390604695d1abd577c4`, tree
`c061e5f6931306fa4655ba531428f639c365bf10`. No source overlays or effort clamp.
The tracked base reconciles the installed site profile. Its local node map must
name sp5–sp7, head 10.30.0.13; preserve other workstations' same-named maps.

`lab-loop.json` ran first, then `lab-copy.json`, each on a boot and repeat boot.
The loop matrix tests numeric effort 100/75 with default/off chat detection:
two seeded 64-token continuations, four seeded 1,024-token active-turn requests,
four clean 6,000-token onset probes. Cases and source hashes are self-contained.
Ten requests/boot, 28,224 output-token cap; sequential, greedy, fresh cache salts,
600-second request deadline, 32 MiB response cap. Returned tools are never executed.
The seeded screenshot transcription tests sustainment, not onset; short completion
requests do not support the chat detector-off override.

CopyDrafter: `copy_probe.py --prompts 3` (three requests in each of four categories),
then single-stream on the SAME boot, then 8/16 distinct
streams (three samples, 1,024 tokens). Original trigger settings: min-match 8,
drafts 8–16, dead-row cutoff 0, budget 6155, capture 59. Old source overlays are omitted.
Historical crashes are positive controls; no released baseline is remeasured.

The checked commit was published and sync/start plans' live/shape lines inspected.
After explicit authorization, the lab window, publish and idle guards coordinated
all three clean checkouts; one window covered both phases. Both loop summaries
and runtime receipts were reviewed before starting CopyDrafter. The verified
restoration profile was `config/candidate-37-tp3-switched-dev.local.json`, restored
once at the end. Commands (also checked first with `--dry-run`):

```sh
python3 scripts/lab.py run experiments/2026-11-03-candidate-38-reproduction/lab-loop.json --production-config config/candidate-37-tp3-switched-dev.local.json --keep-open
python3 scripts/lab.py run experiments/2026-11-03-candidate-38-reproduction/lab-copy.json --production-config config/candidate-37-tp3-switched-dev.local.json
```

Loop requests and responses, their token/logprob/usage/stop records, and every
workload's emitted records, errors, rank logs, image identities, memory observations
and artifact hashes are retained privately and bound by the tracked receipts.
Each workload's exit was checked; the lab may continue after failures.
Single-stream and concurrent-stream rates
are observations of this diagnostic configuration, not a release comparison.
A bounded clean result cannot exclude longer loops or all NaNs. The stock image
has no per-step layout recorder, so no crash is not proof every rank agreed.

Host-only results and exact commands: runs/2026-11-03-host-budget/.
check_host_budget.py executes SHA-verified pinned host budget/compaction bodies:
mirrors 16/5/16 produce rows 17/6/17 on candidate 37 and 17/17/17 on candidate 38.
Five cases pass, including absent/forbidden obsolete callbacks. Simulated CPU
ranks and representative finite confidences; no GPU, collective or model forward.

Offline check: python3 experiments/2026-11-03-candidate-38-reproduction/loop_probe.py --check.

Preflight receipts: `runs/2026-11-03-preflight/`. Four doctor checks and both lab
dry-runs passed in an isolated checkout using the actual site map. Start plans
show live candidate-37 and unchanged shape; sync targets sp5, sp6 and sp7.
Initial macOS SSH socket-path failures were retained in private logs and resolved
with `TMPDIR=/tmp`; that preflight correction changed no runtime. Repository suite:
626 tests, 5 skipped. The execution receipt also retains the initial reservation
syntax and sp5-to-dgx1 bridge failures, resolved through the existing Mac SSH paths
before any serving restart. Restoration's live doctor reports the four existing
host recommendations; those settings were not changed in this window.
