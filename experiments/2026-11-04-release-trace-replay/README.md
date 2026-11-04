# Candidate-38 trace replay on Spaceport TP4

Requested 2026-11-04. The owner explicitly approved the full coordinated TP4
window: sync, candidate replacement, four fresh-boot replay points and
restoration of candidate-37. Execution receipts will record the outcome.

## Identity and intended variable

Base deployment: `cb657ab1` on origin/main. Full identities are in `base.json`.
The candidate is the unchanged final candidate-38 image
`sha256:213ba4df68d06665add7a26260c2321d55854a75b8a51ae71c72d0b97eac814b`,
vLLM tree `c061e5f6931306fa4655ba531428f639c365bf10`, on the recipe's
`tp4-switched` shape and the site's sp1–sp4 map. No runtime source, speculative
policy, KV sizing, effort setting or promoted configuration is changed.

Variable: concurrent replayed sessions, 8 and 16. Each point boots freshly and
runs twice: boot-a orders 8 then 16, boot-b orders 16 then 8. Each measurement
lasts 3,600 seconds after the harness warmup. Allow about 5–6 hours including
preparation and four starts. Port 8001 keeps returning production clients out
of the replay. Restore the reconciled candidate-37 profile once at the end.

Workload: the hashed `requirements/agentx-harness.txt` (1.0.7), full
`semianalysis_cc_traces_weka_062126` at revision
`23f152f6f0f9399a85901b89a6458def0ef16729`, seed 42, and checkpoint tokenizer
`deepseek-ai/DeepSeek-V4.1-Flash-1017` at
`1bae0e1088993b9b82de08f59de5be5a34e42170`.

## Preflight observations

- sp1–sp4 are clean at `3adababa` and serve candidate-37. None holds the
  candidate-38 image. The checked start plan reports live candidate-37 and
  **shape unchanged**. No `--change-shape` is needed.
- sp1 has about 24 GiB MemAvailable, compared with the earlier TP3 head's
  roughly 8 GiB. The existing client is limited to 12 GiB without swap and
  avoids serving cores. Require at least 20 GiB available before every client
  launch; if unavailable, defer to a separate idle client host. A 12 GiB cap
  alone does not guarantee a serving host's memory margin.
- sp8 has an independent active profiling reservation and about 14 GiB
  available. Do not run a replay client or prepare its corpus there during
  that reservation. The DGX cluster also has an independent qualification
  hold; this experiment does not use its GPUs or change its service.
- sp1 runs `7.0.0-1024-nvidia`, 4 KiB pages; its fan service is inactive.
  Record each rank's actual host state and fan capability. Do not present
  this site as the promoted 64 KiB host profile or silently hide warnings.
- Candidate-38's recorded image gate passes: 8,069 passed, 360 skipped,
  zero failed. Its focused TP3 loop/copy reproductions passed on two boots;
  broader inference and release/performance qualification remains pending.
- Both new profiles pass local doctor. This checkout has no prepared image
  inputs (`build check` reports this); the existing image is reused. No
  upstream series is modified by this experiment.

## Execution after authorization

Publish this experiment as one commit before any node sync. Require clean
node checkouts and detach all four at that published commit using
`bin/spark --cluster-config <named-profile> cluster sync --apply`. Stage the
existing candidate-38 image and verify its layer list, source labels and image
identity on every rank. Use the normal image-distribution helper. Inspect
the sync/start/stop plans and their live/shape lines before every apply.

The pinned harness was installed and all 393 traces fetched on sp1 between
windows; the corpus is loadable offline. `runs/preflight/client-setup.json`
records the checks. The large tokenization/replay preparation remains for the
authorized window. Exact setup commands:

```sh
python3 scripts/agentx.py install
python3 scripts/agentx.py fetch --tokenizer deepseek-ai/DeepSeek-V4.1-Flash-1017 --tokenizer-revision 1bae0e1088993b9b82de08f59de5be5a34e42170 --hf-hub-cache /home/chris/.cache/huggingface/hub --avoid-cpus 5-9,15-19
```

Take the TP4 reservation only after its live doctor and idle/publish checks.
Keep its heartbeat and restoration watchdog active. Corpus preparation stops
serving and runs within that window. Check every node below 55 C before
every measurement, using the site's supported fan policy. Inspect minimum
host memory and abort before crossing the serving memory guard margin.
`run_window.py` applies these checks to every point, uses the lab's reservation
and watchdog, keeps the client within its existing 12 GiB scope, and cancels
before any rank crosses the 4 GiB benchmark memory floor. It records native
memory/thermal conditions and restores candidate-37 in its final cleanup.
Every start and stop goes through the named profile and its reviewed plan;
the unchanged TP4 shape needs no `--change-shape` option.

The two specs define the workloads. The executor adds the conditions and runs
them in one window, stopping on a failed point for review. On sp1, from the
clean, published commit used on all four ranks:

```sh
python3 experiments/2026-11-04-release-trace-replay/run_window.py
python3 experiments/2026-11-04-release-trace-replay/run_window.py --apply
```

Check each step's exit and the post-run engine state. Preserve every failed
point and its rank logs. The executor restores candidate-37 and closes the
window only after live doctor; its receipt retains restoration failures and
leaves the reservation for the watchdog when restoration fails.

## Assessment to complete

Retain native receipts under `results/private/agentx/c38-spaceport-tp4-boot-{a,b}`
and lab tables under `results/private/lab/`. Report each boot separately and
the repeat spread: submission validity, errors, overflows, cancellations,
TTFT p50/p90/p99, inter-token latency, per-user interactivity, output
throughput, cache hits/misses, preemptions, KV occupancy, draft acceptance,
memory minima, swap movement and temperature. Keep cancellations at the
measurement deadline separate from errors.

Use recorded released suites on sp1 at
`results/private/suites/1030-tp4-switched-{release,single-stream,sampled}/1030`
for the separate release comparison. Do not remeasure the released image or
compare this TP4 replay directly with the earlier TP3 replay as a speedup.

AgentX replays request shapes and does not score useful task completion.
Release readiness still needs the candidate's inference, deterministic/KL,
long-context, mixed-load and release/single-stream/sampled gates on repeat
boots. The remaining loop canary and a long compacted tool session need their
own scored correctness checks; forced reasoning completion is containment.

## First execution: setup failure, preserved

Window `spark-0009-1793751815` (00:23:35–00:29:20 UTC) stopped all four
candidate-37 ranks and rebuilt the corpus in 172.5 s, peaking at 26.9 GiB.
The 16-session local-stub restore check hit the 5.95 GiB mmap cache, then
failed with a connection reset during warmup. AgentX exited 1; no candidate-38
engine or measurement was started. Candidate-37 was restored on all four
ranks, independently passed live doctor (four existing recommendations),
and the hold was removed. Native receipts are in
`runs/2026-11-04-spaceport/attempt-1`; larger/private raw records remain
under the recorded native paths and ignored evidence directory.

A CPU-only verification using the existing cache, the measurement's 12 GiB
cap and non-serving cores reproduced the 16-session stub failure while
candidate-37 remained live. Its first two-session replay passed. The client
setup is being repaired before resuming the originally approved matrix.
Subsequent windows use `run_window.py --attempt N --apply` so the failed
window remains intact. The candidate image and measured workloads do not change.

## Conclusion

No candidate-38 TP4 measurements yet. No release verdict or measured tuning
recommendation can be drawn from this preparation. Priorities are to obtain a
valid repeatable trace replay, finish TP4 qualification, and verify useful
long-session behavior before changing serving policy.
