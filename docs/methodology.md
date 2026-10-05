# Methodology

## State model

Every fact belongs to one of four states:

- **Observed:** read from a running node and retained in an immutable manifest.
- **Promoted:** accepted desired state in `config/`.
- **Candidate:** an isolated change under `experiments/`.
- **Historical:** a previous immutable baseline retained for comparison.

Observed state is not automatically promoted. A live hotfix must be captured as an
experiment and deliberately promoted or rolled back.

## Experiment layout

Use `experiments/YYYY-MM-DD-short-name/` with:

- `README.md`: hypothesis, one changed variable, risks, and acceptance criteria;
- `base.json`: exact repository commit, upstream pins, image digest, and baseline;
- `candidate.json`: complete delta from promoted configuration;
- `runs/`: native raw receipts, including unsuccessful runs;
- `results.json`: normalized per-run metrics and counts;
- `decision.md`: promote, reject, retain for more evidence, or inconclusive.

Do not compare a new candidate only with a historical number collected under a
different client, prompt, model load, or cluster state. Save sufficient identity to
repeat both sides.

## Minimum benchmark matrix

The stable suite must cover:

1. single-stream code and prose decode;
2. concurrency 1, 2, 4, and the promoted normal agent-workload concurrency;
3. cold prefill at representative short, 32K, 64K, and long-context sizes;
4. prefix-cache replay where applicable;
5. the four-long-context admission target;
6. an end-to-end agent workload with fixed scope and tool policy;
7. minimum available host memory, swap movement, KV use, OOMs, allocation retries,
   request failures, and output-integrity gates.

`bin/spark bench` implements items 1-5 and 7; the agent workload remains
manual. Wrap an agent run in `bin/spark workload --json <path> -- <command>`
to record the server side of that window without sending requests: requests
per hour, prompt and output lengths, prefix-cache share, draft acceptance by
position, the share of engine steps carrying prefill, latency quantiles, and
peak load. Findings per hour come from the agent's own report.

Every benchmark starts from the same thermal baseline: each node's hottest
thermal zone below 55 °C.
- `bin/spark bench` checks every node before measuring.
- If any node is at or above the threshold, every node is pre-cooled at the
  maximum floor of dgx-spark-fan-control (`dgx-fan-control set-state 12`), with
  its `dgx-fan-control` service paused. They stay there until the last one is
  below 55 °C, so the wait cools all of them.
- Then each node's usual fan control returns: the service's curve, or firmware
  automatic where the service is not running. Measurement then starts.
- The bench report's `cooling` section records each node's start and final
  temperature, whether it was cooled, and for how long.
- `--cool-below` changes the threshold (0 skips the check). `--cool-timeout`
  (default 600 s) bounds the wait; a node that cannot cool aborts the bench.
- `scripts/lab.py` kernel jobs apply the same check to their nodes before
  replaying.

Report TTFT, per-stream and aggregate TPS, total wall time, prompt/decode token
counts, and variability across complete runs. Performance is not accepted at the
expense of model quality or silent request rejection.

Every target-hardware startup must be fail-closed. Pre-arm the configured
startup memory guard before each container launch, verify it remains active
through API readiness, and reject the run on any driver allocation failure or
guard stop. Do not repeat a coordinated launch after management-path loss until
the failure is isolated with bounded tests and the owner explicitly authorizes
new hardware work.

## Promotion

A promotion commit must:

1. update `config/`;
2. add a new immutable baseline manifest;
3. link the accepted experiment and all native receipts;
4. update upstream pins or patch series when source changed;
5. prove all configured ranks use the same content-addressed image;
6. pass `bin/spark doctor --live` after coordinated deployment;
7. add the deployed service's complete `bin/spark bench` report as
   `manifests/benchmarks/<baseline>.json`, the reference later runs compare with.

Rollback is a new coordinated deployment of the previous promoted commit. It is
not an ad hoc reconstruction from shell history.
