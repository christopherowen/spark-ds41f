# Native evidence

Archive SHA-256: `2777fbd9115e2bbf16a96b024e69e60c0b875d9fcee273584e72f34e526ff4b9`. `runs.tar.gz` contains 472 native
receipts and coordinator notes. `files.sha256.json` hashes every file relative
to the extracted `relay-progress/` directory. Preserve the archive unchanged.

The archive includes all proxy runs, failed attempts, exact per-node commands,
source hashes, trace records, graph timings, numerical checks, RDMA and physical
port counters, memory snapshots, cooling receipts, builds, CPU tests, GPUNetIO
endpoint logs, the direct-mode incident and recovery, and the 10:45 UTC idle
return and hold release. Failed setup attempts that produced no endpoint log
are described explicitly in `failure-notes.json`; these are transcribed
coordinator observations, distinguished from native tool output.

`final-transport-summary.json` covers the two post-recovery controls and four
streaming candidates. `trace-summary.json` contains the early readiness probe,
trace-only repeat and initial controls. `gpunetio-summary.json` covers the pair
runs including failures. The latter's microsecond figures are half round-trip
estimates from a last-byte ping-pong sample, not collective latency or complete
payload verification. Diagnostic output remained enabled in the passing runs.

The malformed merged `trace-a` output remains archived but is excluded from
summaries. Its replacement is `trace-b`. `stream-control-a` was interrupted while
cooling; it never launched GPU containers. The whole-fragment controls used
for streaming comparisons are `stream-control-b` and `window-control-final`.

```sh
mkdir -p /tmp/relay-evidence
tar -xzf experiments/2026-10-03-relay-progress/hardware/runs.tar.gz -C /tmp/relay-evidence
python3 experiments/2026-10-03-relay-progress/summarize.py \
  /tmp/relay-evidence/relay-progress/stream-control-b \
  /tmp/relay-evidence/relay-progress/stream64-a \
  /tmp/relay-evidence/relay-progress/stream32-a \
  /tmp/relay-evidence/relay-progress/window2-a \
  /tmp/relay-evidence/relay-progress/window4-a \
  /tmp/relay-evidence/relay-progress/window-control-final
```

Every custom measured case sent exactly 25% of its payload over each of the
four HCAs. All six post-recovery screens recorded zero RDMA error deltas.
Physical port counters are shared views; do not add duplicate hardware views
as independent bandwidth. The full image build check reports incomplete inputs
because these are source-only experiments, not rebuilt serving images.
