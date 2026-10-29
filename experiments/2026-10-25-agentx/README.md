# Agent traffic: SemiAnalysis AgentX on production

Owner's brief, 2026-10-25: measure spark-ds41f under agent-shaped traffic with SemiAnalysis
AgentX and report what limits it:
- the prefix-cache hit rate;
- what a cache miss costs;
- the concurrency at which the KV pool stops holding the sessions;
- how much tuning the prefix-cache retention interval and the TP4 KV pool size could gain.

This is a measurement and report. The production recipe does not change; the owner decides on
any recipe change. It is not a release gate either: a closed-loop replay drifts with the server's
own speed, so two builds never replay quite the same mix.

**Status (2026-10-25, 09:30 CET): the tooling, the arm and the spec are ready, tested on the
laptop and against a stub server on dgx1. No window has run.** Results, the ceilings measured
from them and the proposed arms come after the window.

## What AgentX is

- **The corpus.** AgentX ([methodology](https://inferencex.semianalysis.com/agentx/methodology))
  replays 393 opt-in Claude Code sessions. Each session is a main agent plus up to 10 parallel
  subagents, replayed with the recorded gaps between turns. The text is synthetic; the lengths
  and the prefix reuse (64-token block identities) are the recording's.
- **The load.** `--concurrency N` keeps N session trees alive in a closed loop.
- **The run.** Each lane starts 25-75% into its session. A `max_tokens=1` primer carries its
  whole context, then 10 more warm-up requests per lane run with no idle gaps and one output
  token each. A one-hour measurement window follows.
- **The rules.**
  - `ignore_eos`: every turn produces its recorded output length.
  - A cache-bust marker on each new play of a session, so plays never share a prefix.
  - Idle trees are advanced after 300 s, and the whole replay after 10 s of nothing in flight.
- **The harness.** `agentx-harness==1.0.7`, a fork of NVIDIA AIPerf (its `aiperf` package at
  commit `0f885254`). It stamps `submission_valid`.

InferenceX publishes AgentX results for DeepSeek V4.1 Flash on datacenter GPUs only. A GB10 point
would be the first.

## Base

- **Deployment.** Commit `263d11ed` (origin/main when this branch started).
- **Production**, confirmed with `bin/spark doctor --live` on dgx1 at 08:22 CET: "configuration
  OK; live cluster matches with 5 warnings" (the warnings are weight-shard fragmentation only).
  - spark-ds41f-1022, the recipe's `tp4-ring` shape on dgx1-dgx4;
  - image `vllm-ds41f-kkref:19f2c20ed4d6-1022-4`, vLLM tree `e68aa19c`;
  - checkpoint DeepSeek-V4.1-Flash-1017 `1bae0e10`.
- **The live server's cache** (`vllm:cache_config_info`):
  - 49,483 blocks, 8,580,566 tokens;
  - `kv_cache_memory_bytes` 11,274,289,152 (10.5 GiB a rank);
  - retention interval 8,192, max model length 1,048,576, max sequences 16.

## Step 1 findings (no GPU)

### The harness source, read before it ran on a fleet host

The wheel read is the one installed (SHA-256 `a67dc18b…`; the sdist `f4f1877b…`, both pinned in
[requirements/agentx-harness.txt](../../requirements/agentx-harness.txt)). The parts that decide
what runs and what leaves the host:
- the `agentx` CLI wrapper and the `agentx` scenario (`aiperf/common/scenario/inferencex_agentx_mvp.py`);
- the dataset loaders and the Hugging Face download path;
- the chat endpoint's payload (`max_completion_tokens`, `stream_options.include_usage`,
  `ignore_eos`, reasoning read from `reasoning` or `reasoning_content`);
- tokenizer loading, the warm-up strategy, server-metrics collection, listening sockets;
- every URL, `subprocess`, `eval`/`exec`, `pickle` and `trust_remote_code` in the package.

Findings:
- **Network.**
  - It contacts only the server under test (its API and `/metrics`) and Hugging Face (the
    dataset and tokenizer).
  - Runs here are offline (`HF_HUB_OFFLINE=1`): the dataset is fetched beforehand at a pinned
    revision.
  - No telemetry. The W&B, MLflow and OTel exporters are opt-in and not installed.
  - It listens on no socket (health and API servers off by default; its own processes talk over
    local ZMQ).
- **Remote code.**
  - The dataset loads with `trust_remote_code=False`.
  - One error-display path reloads a tokenizer with `trust_remote_code=True`, only after a
    tokenizer failure. Our tokenizer has no `auto_map`, so nothing would execute.
  - We never pass `--tokenizer-trust-remote-code`.
- **Dataset revision.** The dataset is loaded by name with no revision, so a new upload would
  change the corpus silently. `scripts/agentx.py fetch` downloads `main`, refuses unless it is the
  pinned revision, and everything after it reads offline.
- **`submission_valid` is narrow.** It is false only for a broken scenario rule
  (`--unsafe-override`), more than 1% context overflows, a cancelled run, or too little latency
  coverage. Generic errors do not invalidate a run, so `error_request_count` is reported beside it.

### Where the client runs and how it reaches the server

- **Host.** The client runs on dgx1, the head node: the lab runs there, and it has the Hugging
  Face login.
- **Install.** A venv outside the deployment checkout (`~/spark-lab/agentx/venv-1.0.7`). uv's
  CPython 3.12 provides the headers `crick` needs (it ships only an sdist). Installed only from
  the hashed requirements with `--require-hashes --no-deps`; installed 08:37 CET, between windows.
- **The arm.** It serves on **port 8001**, and the client reaches it at
  `http://10.0.1.71:8001`. The production agent's requests to `:8000` then cannot join a closed
  loop that runs for over an hour; they are refused, as in any window.
- **Cores.** The client runs on the cores the serving container is not pinned to:
  `0-4,10-14`, the GB10's slower cores (serving is pinned to `5-9,15-19`).
- **Memory.** The client runs inside a user scope capped at 12 GiB with no swap. The cap was
  tested on dgx1: a first rebuild attempt was OOM-killed inside its scope at 09:01 CET while
  production served, which kept at least 21 GiB available.
- **Workers.** A fixed `--workers-max 8`, so the client is the same at every point.
- **Keep-alive.** `AIPERF_HTTP_KEEPALIVE_TIMEOUT=4`, below vLLM's 5 s keep-alive.

**The corpus rebuild cannot run beside serving.** A first replay validates all 393 traces and
holds their block ids before tokenizing:
- 337 million ids at about 46 bytes each (measured on 40 traces under a 4 GiB cap), so about
  16 GiB before the rebuild's workers;
- dgx1 has about 22 GiB available while serving, and its guard kills serving below 3 GiB.

So the job rebuilds once per dataset, seed and tokenizer with serving stopped, inside the window,
capped at 64 GiB (`scripts/agentx.py prepare --in-window`). Every later run restores the
rebuilt corpus from disk:
- on dgx1, a restore of 20 traces at four sessions peaked at 1.0 GiB (the rebuild of the same 20
  traces at 5.6 GiB);
- the prepare step repeats the restore at 16 sessions and fails unless it fits the
  measurement's 12 GiB cap.

### Tokenizer

The client's tokenizer is the served checkpoint's own, `deepseek-ai/DeepSeek-V4.1-Flash-1017`
at revision `1bae0e10`, read offline from the hub cache already on the nodes:
- a plain `tokenizer.json` (`PreTrainedTokenizerFast`, 129,280 tokens), no remote code;
- the harness finds a tokenizer offline only by repository and revision, not by path.

The tokenizer only sizes the synthetic filler to the recorded token counts. The server encodes
messages with its own DeepSeek encoder. Reported token counts come from the server's `usage`
(`--use-server-token-count`, a scenario default).

### Dataset: full, not 256k

- **Choice: the full corpus.** `semianalysis_cc_traces_weka_062126`
  (`semianalysisai/cc-traces-weka-062126` at `23f152f6`): 1.72 GiB (`traces.jsonl`
  1,847,151,435 bytes), 98,827 requests, inputs up to 990,016 tokens.
- **The 256k variant** (`8fecd2fc`, 0.53 GiB) drops every request over 256k. That is about half
  of the main-agent turns, exactly the deep-context tail that loads the KV pool.
- **Why full.** Production serves 1,048,576 tokens, the model's own limit, which is what
  AgentX expects.
- **Overflows.** A few requests whose input plus recorded output exceeds 1,048,576 are refused.
  They count toward the scenario's 1% overflow limit, and the table reports them.
- **Disk.** dgx1 has 1.9 TB free.
- **On dgx1.** Downloaded at 08:37-08:38 CET, between windows (the other agent's window opened
  at 08:39). The datasets library's cache (1.8 GB) was built at 09:01 CET, and an offline load was
  checked.

### Speculative decoding: production's real DSpark, acceptance reported apart

AgentX forces a fixed acceptance length, because acceptance on synthetic text means nothing.
This vLLM has `rejection_sample_method: "synthetic"` with `synthetic_acceptance_length`. It runs
with DSpark, but it does not deliver the target:
- **Adaptive verification still decides how many drafts are verified.** It uses DSpark's trained
  confidence head, which knows nothing of the forced rate (`adaptive_verification.py`,
  `dspark/speculator.py`).
- **The dead-row cut at survival 0.2 trims further** (`input_batch.py`).
- **The result.** Tokens per step are `1 + min(forced, admitted)`. The cap varies with the text
  and no setting controls it.
- **What would make it exact.** Turning adaptive verification off, which verifies all five
  drafts every step and is not production. Feeding the forced rates to the admission instead
  would be a code change.

Per the brief, the run uses production's real DSpark (`block` rejection, adaptive verification).
- **Acceptance on synthetic text** is reported separately from the server's counters
  (`vllm:spec_decode_num_accepted_tokens*`, `num_drafts`). The prompts are filler; the output is
  the model's own (thinking on, temperature 1), continued past EOS (`ignore_eos`).
- **The engine's speed apart from acceptance** is servebench's engine grid (the release record).

## The arm

[production-agentx.json](production-agentx.json) is the recipe's `tp4-ring` shape with two edits
to what reaches the client:
- **`--port 8001`**, with the matching `deployment.ready_url`;
- **`--enable-prompt-tokens-details`.** It reports each request's cached prompt tokens as
  `usage.prompt_tokens_details.cached_tokens`. The per-request miss cost needs it. The recipe
  gains this flag in #469 and the next release.

Neither changes what the engine computes. `serve_args` is a copy of the recipe's list, and
[tests/test_agentx.py](../../tests/test_agentx.py) fails if the recipe changes without this arm.

## The lab job

`scripts/lab.py` has a new job kind, `agentx`, documented in [docs/lab.md](../../docs/lab.md).
The driver is [scripts/agentx.py](../../scripts/agentx.py); its tests are in
[tests/test_agentx.py](../../tests/test_agentx.py).

1. **Rebuild, once.** If dgx1 has not rebuilt this dataset and seed for this tokenizer, the job
   stops serving and runs `scripts/agentx.py prepare --in-window`:
   - the rebuild, against a stub server;
   - a restore at 16 sessions that must fit 12 GiB.

   A failure ends the job, and the window restores production.
2. **Per point, in order:**
   - **Boot.** The arm boots afresh: an empty prefix cache, counters from zero.
   - **Clean check.** `bin/spark workload --duration 15`: a request that finishes, runs or waits
     ends the job. This is the brief's workload check, recorded as `clean.json`.
   - **Replay.** `bin/spark workload --json … -- scripts/agentx.py run …`, with the node
     telemetry. It runs the harness and samples `/metrics` every 5 s.
   - **Every step** first checks `~/spark-request.json` and `~/spark3-request.json`. A request
     ends the window after the current step, and production is restored.
3. **A table** over the points (`results/private/lab/<run>-table.txt`).

The spec: [agentx-production.json](agentx-production.json). Run `agentx-1022`; points c8, c16,
then c4; one hour each; seed 42; the full corpus.

Every point writes `results/private/agentx/agentx-1022/1022-cN/`:

| Brief asks for | Where |
|---|---|
| Prefix-cache hits and queries, cached prompt tokens | `agentx.json` `server_profiling.prefix_cache`, `.prompt_tokens` (counter deltas over the profiling window); the series in `server-series.jsonl` |
| TTFT p50/p90, interactivity, output throughput | `agentx.json` `harness.ttft_ms`, `.interactivity_tok_s` (per-user tok/s), `.output_tok_s`; the harness's `profile_export_aiperf.json` |
| Preemptions and KV usage over time | `server-series.jsonl` (5 s), summarized as `server_profiling.preemptions`, `.kv_cache_usage`; also the harness's own server-metrics export |
| Harness errors and `submission_valid` | `agentx.json` `harness.submission_valid`, `.submission_invalid_reasons`, `.errors`, `.context_overflows`, `.error_summary` |
| Workload-clean conditions | `clean.json` (before), `server-workload.json` (the whole run, server side) |
| What a miss costs | `agentx.json` `miss_cost`: time to first token against each request's uncached prompt tokens (least squares), and `server_profiling.prefill.tok_per_request_s` |
| The ideal hit rate | `harness.theoretical_prefix_cache_hit_pct`: the corpus's infinite-cache hit rate for the requests replayed |

## The window this needs

All four nodes; production is down for the whole window. The estimates below come from the
corpus model further down: warm-up is prefill-bound at about 5,000 tok/s.

| Step | Minutes |
|---|---:|
| Rebuild (first window only, serving stopped) | ~15 |
| c8: boot 5, warm-up ~2.2M tokens 8-10, measurement 60, drain 1 | ~75 |
| c16: boot 5, warm-up ~5.1M tokens 17-20, measurement 61 | ~85 |
| c4: boot 5, warm-up ~1.1M tokens 4-5, measurement 61 | ~71 |
| Production restored | ~5 |
| **c8 + c16 + c4** | **~4 h 10 min** |
| c8 + c16 only | ~3 h |
| c8 + c16 + c32 (the owner's choice below; c32 warm-up ~10M tokens, 35-40 min) | ~4 h 45 min |

## Before measuring: bounds from the corpus and the KV arithmetic

These numbers are models, not measurements. They come from:
- the traces themselves: block ids, timestamps and subagents, the `062126` revision above;
- the production tree's KV layout;
- a closed-loop simulation of the replay with this server's speeds (prefill 5,000 tok/s, decode
  60 falling to 35 tok/s at 16 running, at most 16 running).

The measurement replaces them.

**KV per session, production layout (per rank).**
- A pool slab holds 256 tokens of compressed KV (227,840 bytes). The pool has 49,483 slabs,
  12.67M tokens with nothing else in it.
- A retention checkpoint is 7 slabs, the sliding-window state of layers 0-19.
- The earlier estimate that at interval 512 one cached 1M prompt's checkpoints fill 58% of the
  pool assumed 16 slabs a checkpoint. With 7 slabs: 29% of the TP4 pool, and 62% of the TP3 pool
  (the shape that runs 512). At TP4's 8,192: 1.8%.
- One live session tree averages about 1,100 slabs at interval 8,192: its main context plus live
  subagents, over a run as the harness starts it.

**Where the limits fall (model).**

| Sessions | Requests in flight, mean / p90 | Time at 16 running | TTFT p50 / p90 s | Prefill offered (share of 5k tok/s) | Live KV (share of the 1x pool) |
|---:|---|---:|---|---:|---:|
| 8 | 5.6 / 10 | 2% | 1.1 / 17 | 34% | 18% |
| 16 | 17 / 28 | 35% | 17 / 55 | 63% | 39% |
| 24 | 23 / 33 | 78% | 21 / 67 | 67% | 55% |
| 32 | 36 / 47 | 100% | 51 / 104 | 69% | 72% |

- **Decode and the 16-request limit bind first.** At 35 tok/s with 16 running, about N = 20.
- **Live sessions fit the pool to about N = 40** at interval 8,192, and to about 85 with a 2x pool.
- **What overflows sooner is history.**
  - Old prefixes of live sessions and finished plays stay in the LRU cache until evicted.
  - Kept whole, they fill the 1x pool after about 55 minutes at c16 and 39 minutes at c32, and a
    2x pool within the hour from about c24.
  - Of main-agent turns, 3.9% return to an older context after a side call; of subagent turns, 24%
    do. Those turns hit only while that history survives.
  - Matching only the previous request (all history lost) costs 1.6-1.9x the prefill of matching
    anything earlier in the tree.
- **Eviction order makes the loss a cliff.** A prompt's slabs go back to the free queue at the
  end of its request, tail first, and its sliding-window checkpoints earlier still, during
  prefill. LRU therefore evicts a prefix's checkpoints and tail before its head. A hit resumes
  only at a surviving checkpoint, so a partly evicted prefix falls back to about nothing rather
  than degrading gradually.
- **The infinite-cache hit rate** over the corpus is 98.3% (matching anything earlier in the same
  tree). It is 98.2% with checkpoints every 8,192 tokens.

**Retention interval: pre-measurement ceiling.**
- Main-agent turns: 89.6% are pure appends, 6.6% diverge (55% of those are shorter than the
  previous prompt), 3.9% are new contexts.
- A pure append resumes at the previous prompt's end checkpoint whatever the interval.
- **Finer than 8,192.** The interval decides only where a divergent turn resumes. Fine
  checkpoints against 8,192 save 104 tokens a turn: 21 ms, 2.9% of the prefill a turn needs, and
  0.1 point of hit rate.
- **Coarser than 8,192.** The checkpoints are about 18% of a live session's slabs at 8,192, so
  that is the most a larger interval can free. It matters only where live KV binds, about N = 40
  and beyond.
- **Verdict.** The interval's ceiling is small at TP4 in both directions. The measurement checks
  it against the hit rate lost to retention (hits below the theoretical rate with no eviction).

**2x KV pool: pre-measurement ceiling.**
- **The best case.** Everything the 1x pool evicts and a later turn needs again. The model puts
  that at the gap between the two prefill columns:
  - up to about 1,300 tok/s of re-prefill at c16 (26% of prefill capacity), once history
    overflows the pool mid-run;
  - more at c24-c32.
- **What would show it.** The measurement shows it as hits below the theoretical rate,
  preemptions, and requests whose uncached part is most of their prompt (`misses_over_half`).
- **Memory.**
  - A 2x pool is 21 GiB a rank, 10.5 GiB more.
  - dgx1 has about 22 GiB available while serving, about 11.5 GiB after the larger pool.
  - That is above the 3 GiB steady guard, but the 5 GiB startup guard must be shown on a boot.
  - The client's 12 GiB cap would no longer fit beside it. The client would move off dgx1, or
    its cap would come down to what the restore check measured.
  - A size that trips a guard is out; the guards are not changed.

**Not asked, but the model says it binds first: `max_num_seqs` 16.** From c16 on, requests wait
for a slot, not for KV. The owner decides whether that is in scope.

## Analysis plan (after the window)

1. **Per point:**
   - the hit rate against the theoretical rate;
   - the cached prompt share;
   - preemptions;
   - KV use (p50/p90/max) and the waiting queue over time;
   - TTFT and per-user tok/s;
   - errors, overflows and `submission_valid`.
2. **Miss cost:**
   - time to first token against uncached prompt tokens, per request (the slope in seconds per
     thousand tokens);
   - the server's computed prompt tokens per second of prefill time under this load;
   - a full miss at the median context (142k) costs about 142 times the slope.
3. **The KV knee.** The point where the hit rate falls away from the theoretical rate while
   preemptions and `misses_over_half` rise.
4. **Ceilings, from the measured miss rate and cost:**
   - (a) **Retention interval.** The lost hits that a checkpoint at the divergence point would
     have saved, times the miss cost.
   - (b) **2x pool.** The misses on evicted history, times the miss cost, as a share of the
     window's prefill time and TTFT.
5. **Arms, one variable each, proposed in a follow-up PR:**
   - `kv_cache_memory_bytes` 2x, at the point where history first overflows (c16 or c32), if (b)
     is worth it and the guards allow;
   - a retention interval only if (a) is.

## Owner decisions

1. **The window.** About 4 h 10 min for c8 + c16 + c4 (3 h for c8 + c16), with all four nodes
   and production down. It can start only after the other agent's runs on dgx1 (the vision eval and
   run 23) release their windows.
2. **c32 instead of c4.** The model puts nothing binding at c4. At c32 history overflows the
   pool and admission queueing dominates, which is the knee the brief asks for. Cost: about 35 more
   minutes.
3. **The arm's two edits.** The private port 8001 keeps the production agent out. The reporting
   flag `--enable-prompt-tokens-details` (#469) is needed for the per-request miss cost; without
   it, the miss cost comes only from server totals.
4. **Speculative decoding.** Real DSpark (production), acceptance reported apart: the synthetic
   control is distorted by adaptive verification. The alternative is a second, non-production arm
   with adaptive verification off and a forced length, for comparability with InferenceX only.

## Results

Pending the window.
