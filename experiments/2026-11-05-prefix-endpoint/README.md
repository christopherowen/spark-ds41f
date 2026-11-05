# CED encoder endpoint retention

Status: CPU validation passed; image construction and serving qualification pending.
This experiment is not promoted. Hardware testing is authorized; the candidate profiles select the published experiment branch.
No production promotion or merge is authorized by this record.

One variable changes: the block-granular, non-EAGLE cache coordinator retains the
complete encoder prompt endpoint as well as its resend boundary when the configured
replay suffix exceeds one token. Source PR: [vLLM #1054](https://github.com/local-inference-lab/vllm/pull/1054).
Exact deployment, source, image and patch identities are in `base.json`.

## Why

For a 4,352-token prompt, 256-token global pages, 128-token CED replay and sparse
encoder-window retention, an identical resend needs the boundary at 4,096. A later
129-token extension can reuse the complete encoder endpoint at 4,352. Retaining only
the first boundary can discard the second. The native CPU regression observes the
hit move from 4,096 to 4,352: 256 fewer tokens to recompute for that extension.
This is a cache-manager result, not a measured latency or cold-prefill improvement.

The encoder windows anchor already cached global/compressed/indexer pages. The
decoder and drafter windows remain private and reconstruct their CED suffix. The
compressor's circular working state is still reconstructed by its existing path.
No encoder, tokenizer, kernel, sampling, hash, page ownership or arithmetic changes
are made. EAGLE, fine-grained hash lookup and ordinary one-token replay are unchanged.
Only fully computed pages are reusable; incomplete and cancelled work cannot invent
a completed endpoint. Eviction remains the existing pool's responsibility.

## Evidence

All completed native tests ran on Spaceport-8 using the candidate-38 image as a
dependency environment, with source mounted read-only. The image digest is recorded
in `base.json`. Docker used the ordinary runc runtime, no network or GPU devices,
two CPUs, 4 GiB memory and no swap beyond that limit. Heavy preparation and testing
were moved off the Mac after its interruption; the interrupted local preparation is
not validation evidence and its files were left intact.

| Run | Result | Interpretation |
| --- | --- | --- |
| Fresh pinned series | 572 patches, exact expected head/tree | Reconstructed candidate-38 from its lock |
| `before` | exit 127 | Image has `python3`, not `python`; invocation error retained |
| `before-native` | 14 failed, 10 passed | Initial 24 endpoint cases, original policy |
| `after` | 24 passed | Initial fix and test matrix |
| `after-prefix-suite` | 7 failed, 247 passed | Rejected broad policy: three unrelated retention regressions and four HiSparse failures |
| `before-prefix-suite` | 32 failed, 222 passed | Expanded original-policy matrix: 28 new regressions plus four HiSparse failures |
| `after-prefix-suite-focused` | 4 failed, 250 passed | Restrict policy to multi-token replay; only identical HiSparse failures remain |
| `final-cpu` | 277 passed, 4 deselected | Prefix and single-type cache suites; 50 new cases pass |
| `probe-tests` | 4 passed | Mocked streaming parser, usage, cancellation receipt and prefix identity; no serving requests |
| `harness-tests` | 4 passed, 1 failed, 1 error | Reporting fixture exposed a glob that included conditions files as repeat directories; corrected before delivery |
| `harness-tests-final` | 6 passed | Stream/usage parsing, boot-level intervals and failed-guard rejection |

The four deselected cases are exactly
`hisparse_async_admission_requires_only_import_destinations`; they fail identically
before and after, and their full failures are retained. They are not counted as
passes. The final CPU run reported peak RSS 1,076,328 KiB (about 1.03 GiB), elapsed
6.04 s and `cuda_initialized=false`. These are CPU measurements, not GPU memory.

The 50 new cases cover retention 0/512, six complete/partial page positions,
chunked prefill, a freed or still-live producer, a simultaneous consumer, scratch
recycling, exact resend, changed prefix with repeated suffix, cancellation,
single-page endpoint eviction, private decoder/drafter blocks, and final pool
reclamation. Cancellation tests distinguish computed pages from the intended prompt.

Original commands are preserved in `runs/commands/`; these are execution receipts
with the original remote paths, not a portable installer. The source patch itself
contains the current regression tests. Logs over 12 KiB are gzip-compressed; exit
codes and stderr are retained. The first setup failed because the desired patched
head was absent in the available source mirror; reconstruction from the locked base
resolved it. An empty log from a refused CPU-lock launch contained no test run.
Both final offline doctor checks pass with topology-correct example node maps
whose head address matches the inherited profile; this is not live qualification.
The initial build check reported an unprepared input directory. Full remote input
preparation subsequently reproduced all locked sources and verified the hashed
wheels; the final complete-input build check is recorded separately. Fresh application of the final patch after the validated
572-patch series reproduces the exact candidate head and tree (`fresh-final.json`).

Initial doctor/build-check errors include a stale patch count and inappropriate
offline node fixtures; `*-final` receipts supersede them without deleting them.

The additional alternatives audit, 12,290-case page sweep and native checks up
to 1,000,256 prompt tokens are in [alternatives.md](alternatives.md). The chosen
policy reaches the append/resend reuse bound with one extra encoder block per group;
dense retention matches that hit using many more intermediate blocks.

## Memory bound and limits

For the recorded deployment geometry (seven encoder-window groups and a
227,840-byte pool block), at most seven additional tail blocks stay eligible for
reuse per endpoint per rank: 1,594,880 bytes, about 1.52 MiB. This is an accounting
bound on retained pages, not a measured allocation increase. The pool does not grow;
these evictable pages compete with other cached entries. Production memory pressure,
eviction frequency, full-model output equality and TTFT remain unmeasured.

## Serving gate, not yet run

`lab-tp3.json` and `lab-tp4.json` each select only the candidate, with a boot and
repeat boot and three native multi-turn workloads per boot. Choose only the site's
qualified shape and node map. Do not run a released build again. No historical run
of this exact workload exists here. The newly authorized controlled specification
uses the unreleased parent for that causal comparison, as described in hardware-plan.md.

Hardware testing is authorized. Before execution, honor the head node's hold/request files; publish the deployment commit; build once
on a remote build host, distribute one OCI digest to every rank, and verify its
source labels. Record that digest and use
`bin/spark cluster sync` with the named candidate profile and clean detached node
checkouts. Read its `live:` and `shape:` plan lines. Do not change shape to bypass a
refusal. The authorization comes from the owner, not from this experiment directory.
An existing foreign hold or request still blocks our GPU work.

The specs pass the offline plan check. A branch-only experiment cannot use the
main-only lab runner: follow the coordinated branch workflow in `hardware-plan.md`,
without changing the helper publication guard. After the approved publication/build/qualification steps, on the head:

```sh
scripts/lab.py run experiments/2026-11-05-prefix-endpoint/lab-tp3.json --dry-run
# Only in the authorized window, after inspecting the plan:
# The live main-only runner is used only if this commit is already on main.
# Otherwise execute the reviewed branch workflow in hardware-plan.md.
```

The runner coordinates all ranks, keeps the window through both boots, records
startup memory and per-step telemetry, and restores production once at the end.
The guarded probe reuses the repository memory watcher, with the configured
memguard plus 1 GiB floor, cooling/fan requirements and thermal ceilings on every
node. It stops requests if telemetry goes stale or a guard trips, and records memory,
thermal and peak cache-use summaries. Retain all aborts and engine logs. Never use a one-rank launch or a dirty-source overlay for this gate.

`probe.py` uses the server's native messages/tools encoder to construct a complete
four-turn source-review transcript of at least 65,536 tokens. It positions the first
prompt at a boundary with a partial page, retains the exact transcript and token
hashes, and records all request attempts. Its checks compare 64 greedy output token
IDs and returned logprobs for cold/warm, solo/concurrent, changed-prefix and
post-pressure requests. Three additional native conversations use different review instructions and
separate primed histories, then extend concurrently and compare against their solo
cold outputs. The probe records their aggregate output tokens per wall-second and
each request's output tokens per wall-second; neither is cold-prefill throughput.
It asserts exact resend and first-extension cache counts,
closes an in-flight stream then retries, and requires observed loss of reuse after
pressure. A client close alone does not prove the scheduler cancelled a prefill;
confirm that event and ownership cleanup from server metrics/logs. Increase pressure
and retain the failed attempt if the pool was not actually evicted. Mock tests are
not evidence that the live server protocol or these gates pass.

Report each turn's cached and recomputed token counts, TTFT and end-to-end time,
both boot means and their 95% interval, observed memory minima/peaks and preemptions.
`tables_arms.py` reports TTFT intervals across the two independent boot means using
Student's t with one degree of freedom; three within-boot repetitions are not six
independent boots. A wide interval is inconclusive. Cold-versus-warm differences
describe the reuse path and must not be presented as cold-prefill throughput gains.
Use matching recorded release suites separately for any deployment-wide verdict.
Reject on any output mismatch, invalid prefix reuse, cleanup failure or memory-guard
breach. Promotion still requires the owner's acceptance and a new immutable baseline.

The exact upstream PR source also passes 275 CPU cache-manager tests (four known
HiSparse cases deselected), recorded in `runs/upstream-pr-cpu.txt`. The controlled
serving specification and restored-service workflow are in `hardware-plan.md`.

The window-aware harness and restoration helper pass 16 CPU tests; the current
comparison plan and offline TP3 doctor also pass. These are preparation checks, not
hardware gains. Heartbeats must be written atomically so concurrent request checks
never see a partial hold file.
