# CopyDrafter defaults and measured candidate rounds

The owner requested PR cleanup, CopyDrafter enabled by default, and successive
candidate measurements in core metrics, using Spaceport to help. Base deployment
commit: `d9b557cb8e87bd31331afc27bec9ed953cd0de8d`. Candidate-38 source and its
recorded two-boot TP3 standard/copy qualification are the starting references.
Released builds and already measured earlier candidates are not remeasured.

## PR review

Closed #850: its recipe-form cleanup changes no serving command and introduces
a known frozen-release equality-test failure. Its patch remains on the PR.
Retain #851's unique startup/image safeguards, #865's unique external-site
receipts, #875's active compiler candidate, and #842 for this default-enablement
work. Reuse #842 rather than opening a duplicate default-enablement PR.

## Round order and measurements

1. Candidate-39 rebases #842 on the exact candidate-38 tree. CopyDrafter becomes
   derived default behavior for greedy adaptive DSpark, with automatic decode-row
   and capture-envelope sizing. This is an explicit interaction test of enabling
   the path and overlapping its bit-identical copy work; compare both recorded
   candidate-38 standard and copy recipes to distinguish those questions.
2. After candidate-39's quality and performance review, a separate round may test
   minimum match 8 against 16, changing only that variable. It needs new source
   identity and receipts, including a mixed copy-dense/tool-like workload that
   binds the token budget. A failed or slower round is retained and rejected.
3. The compiler refresh in #875 stays a separate candidate and quality gate:
   its standalone tests do not establish model-output or serving performance.

Every measured arm needs a boot and repeat boot. Record servebench release,
single-stream and sampled suites with their unchanged workload definitions:
paired output throughput, per-user throughput, TTFT and ITL tails, short-request
wait beside long prefill, cold prefill, engine-step time, accepted/verified drafts,
cache reuse, KV occupancy, errors and output integrity, boot time, minimum host
memory, swap allocation movement and thermals. Judge differences against saved
reference variability and retain every adverse row. If a decision hinges on a
small difference, use the repository's null-arm discipline. No isolated kernel
speedup is an end-to-end serving gain.

## Hardware and restoration

Spaceport access currently fails bounded SSH/TCP checks through the running
operator relay and from DGX1. Its TP4 cooling failure and prior no-content
response also remain unresolved. No reservation, running service, hardware,
network or cooling policy there has been changed. Do not launch there until
management access is isolated and independently verified; a target management
failure requires renewed explicit hardware authorization. Use its capacity only
when those conditions and its reservation allow it.

DGX1–3 are reachable, clean and serving the recorded candidate-36 TP3 dev profile
with identical image `sha256:b0522146640a7d852c017ec8c99423224492b7c2c4951eb3a64005ab4555ab11`.
DGX4 is reachable and has no GPU containers. Before any GPU work, re-read the
shared hold, publish the exact source, coordinate all node checkouts through
`bin/spark cluster sync`, use a named restoration profile, and open the guarded
lab window. Restore candidate-36 once at window end and verify every rank and
live doctor. No topology change or released-build remeasurement is authorized
by these experiment files. The owner's request authorizes the candidate
measurement work; hardware recovery remains separate.

## Current state

Source preparation and PR review only. No new GPU measurement has started.
The fresh full vLLM preparation reproduced candidate-38 tree `c061e5f6` exactly;
#842 applies without conflicts. The Mac partial-clone canonical lock replay
failed on a missing intermediate local GLP blob. Its failure is retained in
`runs/preflight/review.json`; canonical preparation will run in a fresh Linux
checkout using the verified pinned object cache. No GPU result is inferred.
