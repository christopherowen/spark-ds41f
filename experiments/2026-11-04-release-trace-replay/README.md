# Candidate-38 trace replay on Spaceport TP4

Requested 2026-11-04. The owner explicitly approved the full coordinated TP4
window: sync, candidate replacement, four fresh-boot replay points and
restoration of candidate-37. The window ended after one valid 8-session point
and a thermal stop during the 16-session warmup. Candidate-37 is restored on
all four ranks. The planned repeat boots were not run. This is incomplete
qualification, with no performance gain or release verdict established.

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

## Assessment limits

Retain native receipts under `results/private/agentx/c38-spaceport-tp4-boot-{a,b}`
and lab tables under `results/private/lab/`. Report each boot separately and
the repeat spread: submission validity, errors, overflows, cancellations,
TTFT p50/p90/p99, inter-token latency, per-user interactivity, output
throughput, cache hits/misses, preemptions, KV occupancy, draft acceptance,
memory minima, swap movement and temperature. Keep cancellations at the
measurement deadline separate from errors.

For subsequent release qualification, use recorded released suites on sp1 at
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
candidate-37 remained live. Its first two-session replay passed. Increasing only the model-free stub listen backlog from five to 128 then
passed two native verification runs, including the 16-session restores
(4.6 and 4.4 GiB peak under the 12 GiB cap). The repair and unchanged timed
workload are recorded under `runs/2026-11-04-spaceport/client-repair`.
The original four-point matrix resumed as attempt 2 at commit `f2a1d74b`;
its controller log is `~/spark-lab/release-trace-replay-c38/window-attempt-2.log`
and its receipt directory is `results/private/lab/c38-spaceport-tp4-window-attempt-2`.
Subsequent windows use `run_window.py --attempt N --apply` so the failed
window remains intact. The candidate image and measured workloads do not change.

## Second execution: thermal stop, preserved

Attempt 2 used published commit `f2a1d74be3f2209bd8da1899284697f5c5d7a334`
on four clean node checkouts and the unchanged image above. Window
`spark-0009-1793753680` opened at 00:54:40 UTC. The final receipt ended at
02:19:35 UTC on 2026-11-04, after restoration. No failed point was replaced.

| Planned point | Actual result | Native client exit |
| --- | --- | --- |
| boot-a, 8 sessions | Valid one-hour profiling window; 765 completed requests | 0 |
| boot-a, 16 sessions | Thermal guard during warmup; profiling never started | 1; supervising job interrupted with -2 |
| boot-b, 16 sessions | Not run after failed point | — |
| boot-b, 8 sessions | Not run after failed point | — |

The detached controller PID 957627 has ended. Its process wait status was not
persisted by the launcher; the final receipt explicitly records the point
failure and successful restoration. Per-point exits above come from native
receipts and the controller error, not from a successful result inferred from
the existence of export files.

### Valid 8-session snapshot

Profiling began at 01:02:42 UTC after 324 s warmup. The harness reports a valid
submission with zero request errors and zero context overflows. Three other
in-flight credits were cancelled after the profiling deadline and its grace
period; those are separate from the 765 completed per-request records.

| Metric | Single boot-a result |
| --- | --- |
| Aggregate output rate, exported by harness | 157.52 tokens/s |
| User interactivity, p10 / p50 / p90 | 15.87 / 24.92 / 40.85 tokens/s/user |
| Time to first token, p50 / p90 / p99 | 1.136 / 3.093 / 7.870 s |
| Inter-token latency, p50 / p90 / p99 | 40.06 / 62.95 / 83.46 ms |
| Input length, median / maximum | 87,662 / 526,131 tokens |
| Output length, median / maximum | 320 / 9,075 tokens |
| Usage-reported prompt cache read | 96.617% (theoretical reuse 96.976%) |
| Server cache hits / queries, profiling bracket | 84,233,728 / 87,168,053 (96.63%) |
| KV occupancy, mean / peak, profiling bracket | 7.42% / 19.81% |
| Preemptions | 0 |
| Accepted / verified draft tokens per round | 1.6545 / 2.7395 (about 60.4%) |
| Client cgroup peak memory | 4.74 GiB |

The server uses five-second samples bracketing completed request timestamps;
its bracket includes 766 finished requests versus 765 exact client records.
Do not mix whole-run warmup statistics with profiling statistics or replace
the harness's throughput denominator with the server bracket. Whole-run KV
peak was 21.43%. The server's reported prefill rate of 3,935.7 computed
tokens per request-second uses overlapping request times, so it is not a
standalone aggregate prefill benchmark.

The median uncached tail was 1,738 tokens (p90 7,672); 19 requests had more
than half their prompt uncached. A descriptive fit of TTFT to uncached tokens
is 1.112 s plus 0.134 s per 1,000 uncached tokens, R² 0.512. This is an
association within one mixed workload, not a causal prediction or proven
optimization. High reuse, low KV occupancy and zero preemptions provide no
case for increasing KV capacity on this sample. Draft acceptance alone does
not identify the cost of drafting or verification; profile their costs before
proposing a speculative-policy change.

Seven completed outputs (7/765, 0.92%) were shorter than the requested token
budget. All seven match one-to-one with increments of the server's
`finished_reason="repetition"` counter within the corresponding five-second
sample interval; the server reports zero `stop` completions in this bracket.
The harness's generic EOS explanation is therefore not the observed finish
reason. This temporal match is consistent with repetition containment; the request
export has no per-request server finish-reason field. The seven are not
deadline cancellations. Requested lengths can be
reconstructed from the exported mismatch percentage and actual length, and
agree with the saved warning lines. No output text or useful-task score was
captured: these may be desirable containment or false positives on legitimate
repetition. Review long tool/copy behavior with scored canaries before release;
do not disable containment merely to meet a replay token budget.

### 16-session failure and host conditions

All ranks began below 55 C under the unchanged site policy. During the
16-session cold-prefill warmup, sp2 reached the existing thermal guard:
`GPU at 78 C is within 5 C of its thermal slowdown limit`, at 02:17:15 UTC.
The guard's closest reported thermal margin was 5 C. Its sampled sp2 core
maximum was 82 C; auxiliary telemetry sampled 83 C. The core temperature
and driver-reported thermal margin are different readings: this does not
establish a fixed 83 C throttle threshold. Recorded thermal-slowdown and
power-cap counter growth was zero on every rank. This proves insufficient
margin for this test window, not that actual throttling or a decode failure
occurred. The one-hour 16-session measurement did not start.

The warmup export contains **135 successful records and one
`InvalidInferenceResultError`**: no response with actual content, for request
`96772e1b-8ebd-4c56-94b1-04a8fba50b3c`, with a 327,905-token input. Its native
request-end timestamp is **02:16:01.543 UTC**, before the thermal cancellation
at 02:17:15.549. The terminal warmup-failure notice was emitted during shutdown;
that log ordering does not establish that the earlier no-content response was
caused by cancellation. Preserve and investigate it as a separate unresolved
inference/streaming issue. The harness issued 148 warmup credits and sent
cancellation for 12 in-flight credits at shutdown. Server counters record 136
length-finished one-token requests; this differs from client success records
and does not explain the empty response. There is no valid c16 throughput,
latency or draft-acceptance result. Its warmup KV peak was 41.79%, with zero
preemptions; the warmup queue is not a steady-state concurrency comparison.

Continuous guard summaries cover each point's warmup, profiling where present,
and export. Values below are ordered sp1 / sp2 / sp3 / sp4:

| Observation | 8-session point | 16-session warmup |
| --- | --- | --- |
| Initial board-zone temperature, C | 48.3 / 46.9 / 47.8 / 47.9 | 53.7 / 52.0 / 52.8 / 51.9 |
| Guard sampled GPU maximum, C | 76 / 79 / 77 / 77 | 79 / 82 / 80 / 79 |
| Minimum thermal margin, C | 15 / 10 / 12 / 13 | 11 / 5 / 9 / 12 |
| Maximum board zone, C | 94.3 / 90.0 / 86.9 / 83.7 | 93.4 / 88.4 / 85.8 / 91.2 |
| Minimum MemAvailable, GiB | 15.61 / 22.51 / 22.39 / 22.99 | 17.52 / 23.88 / 23.14 / 23.71 |
| Swap allocation growth, GiB | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |

The c16 client peaked at 4.77 GiB; both launches had 24.09 GiB head admission
memory. Swap allocation growth does not measure all possible swap I/O. No
memory-floor failure occurred. All four fan-control services were absent;
the machines identify as NVIDIA DGX Spark, board P4242. Cooling, kernel,
hardware and serving policies were not changed. A supported site cooling
review needs its own authorization before any physical or policy change.

Auxiliary one-second GPU telemetry has uneven c8 coverage (1,865 / 3,792 /
1,865 / 3,310 samples over 3,992 s), and thread snapshots cover only 32–124 s
per rank. It cannot establish a full-run CPU, clock or scheduling bottleneck.
Its sampled c8 GPU maxima are 78 / 81 / 79 / 78 C, illustrating why extrema
from different samplers must retain their provenance.

### Restoration and reproducibility

An independent check at 02:37:35 UTC verified four running candidate-37
containers, the expected image `sha256:e1ee7b1953b1131320746f87fe0cb11a673b53e9702bda63048ea3922768c15a`,
vLLM tree `af0fb1525fc6aa4149b18dee7d8c3ae5120b6f33`, clean source checkouts at
`f2a1d74b`, API port 8000 returning HTTP 200, and no reservation remaining.
The tracked restoration profile's live doctor exited 0 with the four existing
host recommendations (64 KiB kernel, memory-saver, fan-control and idle desktop
services). None was silently applied.

Portable native summaries, numeric request/server exports, failures, phase
events and restoration proof are in `runs/2026-11-04-spaceport/attempt-2`.
`native-files.json` records paths, byte sizes and SHA-256 hashes for 70 native
files, including the large exports and all-rank failure logs retained on sp1.
Full collected logs and archives also remain in the ignored local
`results/private/trace-replay-evidence/attempt-2` directory. The portable
gzip exports contain numeric metrics and identifiers, without model text.
Reproduce the assessment offline from this checkout:

```sh
python3 experiments/2026-11-04-release-trace-replay/analyze.py --check
```

## Conclusion

Hold candidate-38 TP4 release qualification. First qualify sustained site
cooling, especially sp2 during cold prefill, and investigate the preserved
no-content warmup response. Then review the seven repetition-contained outputs
with useful-task correctness checks. A later authorized full matrix must retain
this failed window and obtain both fresh boots at each concurrency; there is
no repeat spread, confidence interval or proven gain from this single valid
point. Do not remeasure the released build.

AgentX does not score useful task completion and does not replace the pending
TP4 inference, deterministic/KL, long-context, mixed-load and release gates.
No serving, cache, speculative or host-policy change is promoted by this run.
