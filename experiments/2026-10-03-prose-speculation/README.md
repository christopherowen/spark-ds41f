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
full replies and before/after Prometheus counters outside the timed request.
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
