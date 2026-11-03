# AgentX on tp3-switched: SemiAnalysis's agent-trace replay on candidate-37

Owner, 2026-11-02: run AgentX. This is its first window. The harness has only
ever been installed and stub-tested on dgx1 (`experiments/2026-10-25-agentx`).

**Status:** prepared, not run.

- Build: candidate-37 at tp3-switched, the profile sp5–sp7 serve
  (`config/candidate-37-tp3-switched-dev.local.json`), image
  `spark-ds41f:candidate-37` (`sha256:e1ee7b1953b1…`).
  [arm-agentx.json](arm-agentx.json) serves it on port 8001, so no production
  client joins the closed loop. Nothing else changes.
- Workload: agentx-harness 1.0.7 (`requirements/agentx-harness.txt`, hashed),
  dataset `semianalysis_cc_traces_weka_062126` at revision `23f152f6`, seed 42.
- [lab-agentx.json](lab-agentx.json), run `agentx-c37-tp3-switched`: points 4
  and 8 (TP3 serves 8 sequences), each a fresh boot, a 15 s clean check, then
  3,600 s measured. About 2.5 to 3 hours with the in-window prepare.
- The arm sets `host.home` to `/home/chris`. The lab's `agentx` job
  (`agentx_target` in `spark_ds41f/lab.py`) reads the profile without the node map, so
  it misses `read_cluster`'s rewrite of the recipe's `/home/swank` to
  `/home/<ssh_user>`, which `bin/spark` gets. Without it the job looks for the hub cache
  under `/home/swank` on sp5, where it doesn't exist. That's a lab bug, for the lab's
  own fix.

## Setup on sp5, between windows (no hold)

1. `uv`: sp5 has none. dgx1's `~/.local/bin/uv` (uv 0.9.17, aarch64, the one
   the dgx1 install used) is copied to sp5's `~/.local/bin/uv`, and its SHA-256
   is checked against dgx1's.
2. `scripts/agentx.py install`: a Python 3.12 venv at
   `~/spark-lab/agentx/venv-1.0.7`, with the hashed requirements only.
3. `scripts/agentx.py fetch --tokenizer deepseek-ai/DeepSeek-V4.1-Flash-1017
   --tokenizer-revision 1bae0e1088993b9b82de08f59de5be5a34e42170
   --hf-hub-cache ~/.cache/huggingface/hub --avoid-cpus 5-9,15-19`: the
   1.72 GiB traces and the datasets cache.
4. `scripts/lab.py run experiments/2026-11-02-agentx-tp3-switched/lab-agentx.json --dry-run`
   on sp5 reports nothing missing.

The job's first step stops serving and runs `prepare --in-window`. That is the
tokenization pass, about 15 minutes the first time.

## Results (window `spark-0013-1793667996`, run `agentx-c37-tp3-switched`, 2026-11-03)

The first AgentX run of this project. Receipts on sp5 under
`results/private/agentx/agentx-c37-tp3-switched/`; prepare took 4 minutes.

**Both points were killed by the host's memory guard, not by the engine.** On a
GB10 the GPU's memory is the host's: with the 1M-context KV pool loaded, sp5 has
about 8 GiB available at idle, and the lab's `agentx` job runs the harness client
on the head node, sp5, beside rank 0. The client peaked at 3.6 GB (c4) and 4.9 GB
(c8). `spark-memguard` kills the serving container when `MemAvailable` falls under
3 GiB (`journalctl -t spark-memguard`; Docker records `kill` then `die 137`).

- **c4** (booted 03:01:58Z): killed at 03:11:28Z, 9.5 minutes in, at 2.96 GiB
  available (3.85 GiB two seconds earlier). The engine was healthy to the kill: 2 to
  4 requests running, KV 7 to 17%, acceptance 2.8 tokens a step. 44 warm-up
  requests had completed; 7 of 67 profiling requests failed at the kill, over the
  harness's 10% threshold, so it aborted. No measurement.
- **c8** (booted 03:13:45Z): the harness ran its whole profiling phase, 03:21:05 to
  04:21:19Z, and the kill came at 04:21:54Z, after it, at 2.63 GiB available, during
  the client's export (its 4.9 GB peak). The lab marked the point failed because the
  engine was dead at its post-run checks; the measurement is complete and
  `submission_valid`.

**c8, candidate-37 at tp3-switched, 8 sessions, 3,613.8 s** (`agentx.json`,
`harness/profile_export_console.txt`):

| Measure | Value |
|---|---|
| requests | 620 completed, 0 errors, 0 context overflows, 5 cancelled at the deadline; 3 of 10 sessions completed |
| input length (tokens) | p50 86,881; p90 285,578; max 521,452; mean 116,452 |
| output length (tokens) | p50 302; p90 1,629; max 12,723; mean 706 |
| time to first token | p50 1.53 s; p90 16.1 s; p99 49.5 s; mean 5.57 s |
| inter-token latency | p50 47.1 ms; p90 63.2 ms; p99 102.0 ms |
| interactivity (tok/s a user) | p10 15.8; p50 21.1; p90 29.9 |
| output throughput | 120.3 tok/s aggregate; 0.170 requests/s |
| prefix cache | 96.9% hit (theoretical 97.6%); 0 preemptions; KV 16.6% at the 43-minute sample |

The harness's `agentx` scenario reports request throughput, latency and
interactivity; it has no findings count, so findings an hour (the project's goal
metric) is not in this output.

**Next:** the client must leave the serving host. The `agentx` job runs it on the
head node by design; running it from sp8 (the build node, no engine) against sp5's
port 8001, with the metrics sampling unchanged, is the lab change. Until then the
TP3 points sit on the guard's line and a measurement depends on luck.
