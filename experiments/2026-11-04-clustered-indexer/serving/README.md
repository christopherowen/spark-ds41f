# Paired Spaceport request measurement

The owner requested an integrated request-speed measurement on a free Spaceport
setup. The idle three-node sp5–sp7 group is selected; sp1–sp4 and the DGX cluster
are outside this window. The exact C37 restore image and live site map have been
reconciled into tracked `restore.json`/`nodes.json`. Original serving checkouts
remain unchanged. Restoration uses the same C37 image, model, command settings,
cache and API port from the new published isolated checkout; its profile label
and checkout identity therefore name this tracked restore arm.

One control/candidate variable: `VLLM_DS41_CLUSTERED_SEARCH=0/1`. Both use the same
C36 image and the same independent scratch-read guard, attention factory and
native reindex route. C36 is unreleased; a fresh paired control does not remeasure
a released build. Released1030 remains an immutable recorded reference only.

The serving adapter uses per-row page tables without inferring request ownership,
rebuilds metadata for every call/replay and reserves all scratch in the existing
workspace before memory profiling. At most eight query rows use that scratch at
once. Flat native selector scratch keeps its exact packed layout at every chunk
boundary. Native output buffers receive selected keys and source block IDs/counts
directly. Reindex and small planned domains use unchanged native plans. All kernel
calls are declared before the cache freezes; no compile is permitted in forward.
This adapter changes no attention or weight arithmetic.

`sp8-serving-01` failed before a completed case because the first scratch view
incorrectly sliced a packed one-dimensional native buffer by row. Its receipt is
retained; the correction uses exact native scratch specifications for each chunk.
The corrected adapter passes two fresh20-case runs (16 mixed-owner synthetic
cases plus four real layer/interface operands), and dedicated initcheck,
memcheck and racecheck across both native key-page sizes. The nine-row gate
exercises a final one-row chunk. A declared-cache freeze gate is separate.

The paired request plan is LSA-A, control-A, control-B, LSA-B, with one coherent
three-node boot per arm. Each run uses `bin/spark bench`, quality, prose/code at
concurrency1/4/8, four fixed decode samples, and three cold source-prefill
samples at32768/131072 tokens. Exact commands, telemetry, request results and
both boots are retained. Generated outputs must match before interpreting speed.
These points establish request latency/throughput for this recipe, not a full
release or general quality qualification. The existing real-input0% pruning
result remains unchanged; the purpose is to measure its actual request cost.

`run_window.py` requires the selected hostname, a clean published checkout,
independent live doctor and30s idle traffic. It reserves all three native hold
files with exclusive creation and one token, refreshes their heartbeats, reads
live/shape plans before applying named-profile coordinated operations, and never
passes `--change-shape`. It yields after the current completed job on a request.
It restores C37 once in `finally`, verifies all image IDs, mount/environment
removal and live doctor, then removes only its own holds. An independently
detached watchdog restores through the same coordinated commands if its
heartbeat is over15minutes old. Failed jobs and restoration failures are retained.

Publish one commit, bootstrap clean detached isolated Git worktrees through Git
on all three nodes, then use the controller's normal coordinated sync. No dirty
source tree is copied. Do not change a checkout while a mounted test uses it.

```sh
python3 experiments/2026-11-04-clustered-indexer/serving/run_window.py --out /home/chris/spark-lab/results/clustered-indexer/serving-window-01
```

No promotion is proposed. Candidate correctness and the complete request-level
performance comparison remain pending until native receipts establish them.

The first integrated pilot is retained as `serving-window-01`. It was requested
to close after its current completed arm when adaptive decode sampling was found
to advance the shared RNG by a variable amount, changing later source-prefill
prompts. Its observations are not discarded or presented as a matched comparison.
The corrected matrix fixes both sample limits at4 and uses `bench_requests.py` to
hash each exact model input, retain the full payload hash separately, and save
native raw per-request elapsed time and output digest. Hashing precedes the native
request timer. Payloads are never changed. Random cache salts remain unique to
prevent reuse and are excluded only from the matching hash. The final report
requires matching request identities across arms before comparing their latency.
