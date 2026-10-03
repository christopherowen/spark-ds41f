# Focused TP3 speculative-decoding check

Investigate the historical 2.7% one-stream prose throughput difference before
changing kernels. The native benchmark's `step_ms` is calculated from acceptance
and throughput; it is not independent kernel-timing evidence.

Use the same rebuilt r5o image as TP3 revalidation. One freshly profiled boot
saves its cost curves, then two restarts reuse those exact curves. This is a
same-configuration repeatability check, not an old/new image A/B. The historical
main image is unavailable locally; its three-sample report is context only.

Each boot runs the native quality gate and 12 shuffled samples each of prose,
prose without reasoning, story, and code, all at one stream and 256 tokens.
Each point has one discarded warmup. The original payloads, sampling, request
accounting and memory/thermal guards remain active. An experiment wrapper saves
answer content, hashes covering the streamed output (including reasoning),
character counts and before/after Prometheus counters outside the timed request.
It does not instrument the inference engine. Engine request-duration counters
are separate server observations, not GPU kernel timings.

Maximum fans remain selected during all three arms, starting each benchmark
below 45 C on every thermal zone. The only permitted live-doctor differences
are the three intentionally paused fan controllers. No persistent host setting
changes. Restore the stopped entry state and original fan controls on completion.

Run after publishing and synchronizing one clean revision on all TP3 nodes:

```sh
TMPDIR=/tmp python3 experiments/2026-10-03-prose-speculation/run.py
```

The supervisor runs on dgx1, owns `~/spark3-hold.json`, refreshes it during every
command, and uses coordinated cluster start/stop. All receipts, failures and
reports go under `results/private/prose-speculation/`. Do not interrupt a remote
start by killing only its SSH client. The runner waits for a child to exit before
cleanup. A lost management connection requires manual investigation, not a retry.

Compare per-boot means and within-boot variability, acceptance by position,
verified rows, output hashes/lengths, request decode time and derived step time.
Do not pool requests into independent boot replicates, use acceptance-normalized
throughput as independent proof, or infer pinning's causal benefit without an
unpinned comparison. Reproducible losses need investigation; inconclusive small
differences do not justify speculative kernel changes.

## Results

Executed revision `d87e91e257d69f5bc577540dc15fa8cf295bfa95`; every rank used
image `sha256:aad8a74089ff379f5bc7905e86f9c7e2c053396d0a4027039e869c505ca7621b`.
All three cost-table hashes are
`43f7e3cae9b7a0e8cc2c7d30c6fa7998159e6cff4cc0e04a82f3d4f8f8038756`;
both restart logs confirm reuse. The table was measured on the first boot,
not selected for a favorable benchmark result.

| One-stream prose | Fresh profile | Pinned restart 1 | Pinned restart 2 |
| --- | ---: | ---: | ---: |
| Throughput, tok/s | 51.777 | 51.192 | 51.798 |
| Within-boot 95% interval half-width, tok/s | 1.386 | 1.241 | 1.706 |
| Accepted tokens per draft cycle | 1.243 | 1.222 | 1.261 |
| Verified drafts per cycle | 3.389 | 3.362 | 3.403 |
| Server decode duration per counted step, ms | 41.587 | 41.657 | 41.852 |
| Derived native benchmark step estimate, ms | 41.681 | 41.768 | 41.951 |

The server measure uses request decode duration divided by the observed engine
iteration count minus the single short-prompt prefill. All 36 prose requests
have exactly one success and iteration count equal to draft count plus one.
It includes runtime overhead; it does not isolate GPU compute. The native
`step_ms` instead uses `(1 + accepted_per_draft) / client_decode_tps`.
Neither metric holds routed experts or verification work fixed.

The story case sometimes has iterations without a recorded draft cycle, so
the summarizer retains all its samples and counts but restricts the supplemental
server-step summary to samples satisfying that explicit count identity. Its
throughput results include every measured request.

| Other one-stream workloads, tok/s | Fresh | Restart 1 | Restart 2 |
| --- | ---: | ---: | ---: |
| Prose, reasoning off | 58.653 | 58.240 | 57.734 |
| Story, reasoning off | 45.213 | 44.944 | 44.966 |
| Code, reasoning on | 63.275 | 62.985 | 61.801 |

All 144 measured decode requests completed with 256 tokens, plus 12 discarded
warmups and 15/15 quality passes. Each workload produced 36 distinct output
hashes across the three boots: pinning costs does not make the model deterministic.
No foreign-traffic samples were discarded. Minimum available memory was 6.67 GiB,
swap growth and recorded thermal slowdown were zero, and GPU temperature stayed
at or below 56 C. The supervisor restored the stopped entry state, normal fan
controllers and idle GPUs, and removed its hold at 14:03:08 UTC.

## Interpretation and decision

**Retain the existing speculative settings.** The first and third boot means
are almost identical; the second is 1.13% lower. The pairwise prose intervals
include zero. This establishes useful repeatability under one cost table,
not a sub-percent equivalence guarantee or a causal benefit from pinning.

Historical main recorded 52.804 tok/s, 1.240 accepted drafts and 3.006 verified
drafts, from only three samples. These new means remain 1.9–3.1% lower, but every
comparison interval includes zero. The original high-fan screen's lower acceptance
is not a complete explanation of the new runs: their acceptance is near main,
while they verify about 12–13% more drafts. Verification policy and output
variation are confounders, not evidence of a transport slowdown. Historical main
also used a different image ID; its unpinned startup table was not captured here.
An exact old/new runtime comparison remains unperformed.

The [cost audit](cost-audit.json) identifies a concrete policy detail. This
profile measured rows 3/4 at 19.2896/19.2134 ms. The scheduler's cumulative maximum
makes both 19.2896 ms, giving the third draft zero predicted incremental cost.
Every measured prose cycle in the first two boots verified at least three drafts.
That plateau can arise from real kernel shapes or profiling noise; these runs
do not justify imposing an artificial slope, changing cost scale, reducing draft
depth, or promoting this particular table as a faster default.

For future performance comparisons, save the cost table with image/topology
identity and use the same table for both same-TP arms. Report verification counts
and server timing alongside throughput. TP3 and TP4 require their own tables.
A policy tuning experiment should then compare multiple prose/code workloads
with a contemporaneous control, rather than chase one historical three-sample mean.

The owner authorized merging TP3/TP4 support into main during this run. This
experiment does not qualify the unbuilt collective-contract image or alter the
promoted runtime configuration. No kernel or speculative-policy change is selected.

## Evidence and unsuccessful attempt

[results.json](results.json) is reproduced by `summarize.py`; the native reports,
per-request counter snapshots, answer content/output hashes, serving logs, cost
curves, startup/stop receipts and fan observations are in
[hardware/runs.tar.gz](hardware/runs.tar.gz), with per-file hashes in
[hardware/sha256.json](hardware/sha256.json). Extract the archive and pass its
root plus an output JSON path to `summarize.py` with `TMPDIR=/tmp`.

The initial runner at `56c7b4c` omitted `--replace`. Coordinated startup refused
the existing stopped containers, before any benchmark ran. Cleanup completed,
and its full receipts are retained under `attempt1/` in the archive. The corrected
published runner explicitly replaces stopped containers; all subsequent starts,
benchmarks and stops exited zero. The saved served-source file is read-only audit
evidence, not an alternative source tree or a runtime overlay.
