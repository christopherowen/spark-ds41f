# Core metrics: what we measure, what we should, and what to build first

Owner, 2026-10-26: "formulate new plans to extend core metrics (like tps, ttft, prefill, memory
wastage, stalls, etc)". This record is that plan, for the metrics releases and research are
judged by. It has four parts:

1. an inventory of what we measure today, where, and what each source misses;
2. what the records we already keep say, from prototype parsers in this folder;
3. an extended metric set, each with its definition, how it is measured, its noise floor and
   its users;
4. a ranked build list toward 100 tok/s at one stream and better agent latency.

Base: origin/main 767b4113 (spark-ds41f-1025 serving). Laptop only: no cluster or GPU time
was used, and no production tooling changed. Every number below is derived from committed
records (`manifests/suites/*`, experiment receipts) or from lab traces already on this laptop.

Rules this plan keeps:
- metrics in production cost nothing; tracing is a flag that is off;
- no boot-time cost profiling inside serving;
- no fingerprints as change detectors: records are keyed by git commit and release;
- each idea is bounded before it is built.

## The short answer

**Build these first.** Each changes a decision toward 100 tok/s or agent latency, and each is
cheap.

| # | Metric | Decision it changes | Cost | From data we have? |
| --- | --- | --- | --- | --- |
| 1 | Paired single-stream verdict: per-prompt decode time, paired across builds on identical text | Whether a 1-4% c1 lever is real. Today's c1 interval is ±3.7%; the same build repeats within 0.13%. | S: servebench keeps a compact per-request table; compare pairs by prompt | The A/A evidence, yes. The pairs need the per-request table. |
| 2 | The c1 step budget per release: byte floor, below-rate streaming, DRAM idle by cause, host gaps, collective waits and late peers | Where the 8.7 ms of non-byte time goes, and so which of the fusion, PDL, prefetch and collective levers to build next | S-M: one lab profile job per release (all ranks), parsed by floor.py and trace_waits.py | Yes: request-inventory-5 (rank 0) and the 2026-10-03 four-rank traces |
| 3 | Rows a step: verified, live, dead, wasted; acceptance by position on the single stream; unbiased survival | Tokens a step, the other half of 100 tok/s: 2.74 tokens a step at half the overhead | S for per-position (vLLM counters exist); M for live and dead rows (a counter on the accepted-count copy) | Verified rows, yield and wasted rows, yes (derived). Live and dead rows, no. |
| 4 | Agent first-token latency: prefix-cache hit rate, miss cost, retention, and the TTFT split | Agent latency: 98-99% of live agent prompt tokens are cached, so a miss, not a kernel, sets TTFT | L: the AgentX window (#574); S for the histogram split in servebench | The split, partly (window means). The cache metrics, no. |
| 5 | A routed engine grid: fixed shapes with serving's distinct experts, not identical dummy rows | Whether the text-independent verdict sees the routed-expert stream. Today the grid sees 61% of a c1 step and 26-28% of a c8 or c16 step. | M: a dev-mode routing table for dummy steps, and `servebench engine --routes` | The gap, yes. The fix, no. |

**What the records already show** (section 2 has the tables):
- The release's single-stream interval is about 30 times the same-session repeat. c1 tok/s
  repeats within 0.06% (median of three A/A pairs, worst 0.13%), but its window interval is
  ±3.7%. So 1022 → 1025's +2.1% at one stream read "same", while its step rate read
  "better" (+1.95% ±1.59%).
- Some verdict rows call a build better or worse than itself:
  - Poisson TTFT p99 does in 5 of 5 A/A pairs: it moves 3.8% (median) against the 3%
    tolerance;
  - the grid's geometric mean does in 6 of 7: boot-to-boot drift (0.3-1.1%) against a ±0.15%
    interval. The verdict block's repeat floor covers that, but the per-pair blocks do not.
- **A c1 step verifies 4.09 rows for 2.28 tokens:**
  - row yield 0.56;
  - 1.80 verified rows a step produce no token.
  - At c16 the yield is 0.85.
- **The grid sees part of the serving step.**
  - At one stream (TP4, 1025) the grid's step at the same rows is 18.4 ms against 30.3 ms
    served: a gap of 11.9 ms.
  - About 5.3 ms of the gap is the distinct experts real text reaches (13.8 against the
    dummy's 6 a layer).
  - Of the other ~6.5 ms, #610's one-GPU decode-step diff puts about 1.2 ms on the critical
    path. Most of it, about 1.0 ms, is the target's verification head, which the grid's
    dummy step does not run.
  - The remaining ~5.3 ms is not in a one-GPU step: Engram, collectives with real peers, real
    context. Only an all-rank serving trace can attribute it.
  - At c16 the gap is 91 ms of a 123 ms step.
- **Single-stream TTFT is 4.2 steps on 1025, against 3.2 on r6x.** The extra step is sink
  rejection's held first token (the 1022 record measured it: +41 ms isolated). It is 24% of
  c1's 128 ms.
- **Prefill on the 1025 ring:**
  - TTFT = 0.42 s + 134.8 ms a 1K tokens, rising by 134 ms a 1K per 1M of context;
  - the prefill rate at 1M is 27% below 32K's;
  - on the traced 16K chunk (ports build, before #593), communication runs exposed for 21.5%
    of the chunk.
- **Rank 0's collectives at c1** (1025-era ports build) are 4.8 ms a step, about 3.7 ms of it
  above each kernel's 5th-percentile time. **The four-rank traces** (2026-10-03 build) put
  each rank's wait for the last peer at 1.0-1.7 ms a step. dgx4 arrived last at 39% of the
  collectives and dgx2 at 10%; an even share would be 25%.
- **vLLM's latency histograms give exact window means but no usable percentiles here.** In
  the current tree the TTFT buckets step 0.1 / 0.25 / 0.5 / 0.75 s, and the queue, prefill
  and decode histograms start at 0.3 s.

## 1. What we measure today

| Source | What it measures | Where it lands | Production-safe? | What it misses |
| --- | --- | --- | --- | --- |
| servebench `run` (closed loop, Poisson) | output tok/s by sub-window; engine steps/s and tokens a step from `/metrics`; accepted and verified drafts a step; TTFT, TPOT, per-user tok/s and e2e percentiles; goodput under an SLO; prefix-cache hit rate; preemptions; foreign requests | suite step JSON (`servebench/1`); `manifests/suites/<release>-<shape>[-single-stream\|-sampled]/` for promoted builds | yes (client; reads `/metrics`) | per-request records (only with `--save-requests`); acceptance by position; KV use; the server's TTFT split; anything paired across builds |
| servebench `engine` (the grid) | step time at 20 fixed decode shapes, with the target-forward and drafter split | `grid.json` | dev mode only (`VLLM_SERVER_DEV_MODE`) | real routing: dummy rows are identical tokens, so they read 6 experts a layer at every shape; host work between steps; sampling on real logits |
| servebench `model` | greedy text identity, teacher-forced KL, top-1 agreement, perplexity; acceptance by draft position at c16 | `model.json.gz` (not kept in manifests) | yes | acceptance by position at c1, and its unbiased form (every draft verified) |
| servebench `prefill` | cold real-text TTFT at 32K-1M, 3 repeats | `prefill.json` | yes | chunk fixed cost (one budget per run); the split into compute and communication |
| servebench `stall` | decode gaps while one cold prompt prefills | `servebench-stall/1` | yes | in no built-in suite: it ran only in two experiment suites (2026-10-19), with no JSON committed |
| `scripts/servebench_suite_table.py` | verdicts: change, 95% interval, the run's own repeat; every metric of every pair | `results/private/lab/<run>-table.txt`, `release-table.txt` in manifests | n/a | a noise floor per metric across releases; derived metrics (rows, yield, gap, TTFT in steps) |
| `scripts/release_table.py` | the pre-servebench release profile (bench.json, determinism lines) | text | n/a | retired with the old bench (TODO item 7) |
| `bin/spark bench` watcher → `conditions.json` | per node: lowest MemAvailable, swap growth, highest GPU power and temperature, throttle time (1 s memory, 15 s nvidia-smi) | suite run dir and manifests | bench only | raw samples (only extremes are kept); the startup low; mean power |
| lab telemetry (`scripts/lab.py`, PR #216) | per node at 1 Hz: SM clock, power (min, p10-p90, max, mean), temperature, busy threads | `<out>.telemetry.json` on dgx1 | lab only | not copied into manifests, so tokens a joule is not kept |
| vLLM `/metrics` | servebench reads 13 series (prom.py). Not read: `kv_cache_usage_perc`, the latency histograms (`time_to_first_token`, `request_queue_time`, `request_prefill_time`, `request_decode_time`, `inter_token_latency`), `prompt_tokens_cached`, `request_prefill_kv_computed_tokens`, `spec_decode_*_per_pos` (read only by `model`), `num_requests_waiting_by_reason`, `sink_*`; `kv_block_*` residency only with `--kv-cache-metrics` | Prometheus text | yes | per-request detail; coarse buckets |
| `bin/spark workload` | requests an hour, prompt and output lengths, cached share, acceptance by position, steps carrying prefill, latency quantiles, gauges | `--json` | yes (read-only) | ran on live traffic once (2026-09-28, prose in TODO.md); a decode step can emit up to 96 tokens at TP4, above its 64-token "carrying prefill" threshold |
| memguard (`scripts/memguard.sh`) | MemAvailable every 0.25 s at startup, 2 s serving; kills below 5 / 3 GiB | the node's journal, breaches only | yes | the startup low itself: every boot's margin over 5 GiB is lost unless it breaches |
| vLLM startup KV line, `scripts/kv_capacity.py` | pool blocks and the admission ceiling (11 full 1M contexts on the ring, #586) | serving log; arithmetic | yes | pool use under load |
| torch-profiler traces (lab `profile` job) | every kernel of every rank; parsed by `floor.py` (#598), `step_shares.py` (#614), `schedule_bounds.py` (#615), `ceilings.py` (#583) | `results/private/determinism/<run>/` | lab only; Python stacks inflate host gaps | all-rank alignment (every parser reads rank 0); a standard per-release summary |
| IKET (#610) | in-kernel load, wait, compute and store phases. Decode steps at c1 and c8, TP4 and TP3 rank shapes, ran on dgx3 (runs 42-47). `step_diff.py` diffs a step against the grid's dummy step, kernel by kernel, with each kernel's exposed time. | run-iket and nsys traces (`step_bundle.py`, `time-*` bundles) | debug flag, off | one GPU (six target layers and the drafter): no real peers, Engram or long context |
| sparknet RoCE trace | one-shot all-reduce phases: staging, wait, reduce, per-peer arrival | `/dev/shm/sparknet-trace-*.bin` | opt-in traced kernel | all-gathers; a link to the torch traces |
| `experiments/2026-10-20-ttft-gap/ttft_trace.py` | TTFT split into the rest of the running step, the step decided ahead, the own step, delivery | experiment runs | lab only | not a servebench step, so not in any suite |
| Engram timeline (`SPARK3_ENGRAM_TIMELINE=1`) | per layer and rank: reads, GPU wait | jsonl | debug flag | Engram's share of short-prompt TTFT |
| `experiments/2026-10-06-prefetch-fill-rate/energy.py` | GPU tok/J from 0.5 s power samples (c16 1.39 tok/J then) | one experiment | lab only | everything since |
| AgentX (#574, branch `agentx-traffic`) | agent replay: TTFT, ITL, hit rate, miss cost, KV use | never ran | lab window | everything (no window yet) |

## 2. What the records already show

Four prototype parsers in this folder read only committed records or the lab traces on this
laptop. Their receipts are in [runs/](runs/).

### 2.1 Noise floor of every release metric ([noise_floor.py](noise_floor.py))

Every release table ends with the candidate against its own repeat, a separate boot in the
same session. Thirteen tables have that A/A block (r6n to 1025, TP3 and TP4, the release,
single-stream and sampled suites). Selected rows ([runs/noise-floor.txt](runs/noise-floor.txt)):

| Metric | A/A pairs | Median \|A/A\| | Largest | Printed ±95% (median) | Called better or worse |
| --- | ---: | ---: | ---: | ---: | ---: |
| single-stream c1 output tok/s | 3 | 0.06% | 0.13% | 3.70% | 0 |
| single-stream c1 engine steps/s | 3 | 0.05% | 0.09% | 1.59% | 0 |
| single-stream c1 TTFT p50 | 3 | 0.52% | 0.80% | none (3%) | 0 |
| c16 output tok/s | 4 | 0.23% | 0.53% | 0.92% | 0 |
| c8 output tok/s | 6 | 0.18% | 0.29% | 1.12% | 0 |
| grid decode geometric mean | 7 | 0.34-0.43% | 1.14% | 0.14-0.15% | 6 of 7 |
| grid, single shapes | 125 | 0.48% | | 0.92% | 20 of 125 |
| prefill 32K / 131K / 262K / 500K-524K / 1M tok/s | 3-7 | 0.18-0.49% | 0.76% | 0.28-1.05% | 6 of 29 |
| Poisson TTFT p50 | 5 | 1.90% | 3.95% | none (3%) | 2 of 5 |
| Poisson TTFT p99 | 5 | 3.75% | 6.57% | none (3%) | 5 of 5 |
| sampled c8 TTFT p99 | 3 | 4.57% | 5.28% | none (3%) | 2 of 3 |

What follows from it:
- **Closed-loop windows measure text, not noise.**
  - A greedy, batch-invariant build writes the same text on every run. So a window's
    tokens a step depend on which prompts fell in it, and the window-to-window spread
    (±5.1% on tokens a step at c1) is the prompts'.
  - Between two runs of identical text that spread cancels. That is why the A/A is 30 times
    tighter than the printed interval.
  - Pairing per prompt keeps the cancellation for a comparison between builds, whenever the
    determinism gate holds (identical text). That is priority 1.
- **The grid needs boots, not replays.** Within a boot its interval is ±0.15%; between boots
  it moves 0.3-1.1%. The verdict block already widens it by the repeat. Its per-pair blocks
  should print the boot floor instead of the replay interval.
- **Poisson TTFT p99 is not a verdict row** until it has an interval, for example a
  bootstrap over its ~200 requests, or a longer window.
- **Prefill at 500K and above has three repeats under a normal interval.** It miscalls
  sometimes (2 of 3 at 500K). Four repeats at the large sizes, or the run's repeat floor,
  would fix it.

### 2.2 Throughput, rows and the grid ([core_metrics.py](core_metrics.py))

From the suite records of every promoted release ([runs/core-metrics.txt](runs/core-metrics.txt)).
Single stream, TP4:

| Release | tok/s | step ms | tokens a step | verified rows | row yield | wasted rows | grid at same rows | gap | TTFT | TTFT in steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| r6v | 71.1 | 31.71 | 2.255 | 3.98 | 0.566 | 1.73 | 19.57 | 12.13 | 103 ms | 3.24 |
| r6x | 74.1 | 30.50 | 2.263 | 4.05 | 0.559 | 1.79 | 19.00 | 11.51 | 98 ms | 3.22 |
| 1022 | 74.1 | 30.58 | 2.271 | 4.07 | 0.558 | 1.80 | 18.74 | 11.84 | 129 ms | 4.20 |
| 1025 | 75.3 | 30.31 | 2.284 | 4.09 | 0.559 | 1.80 | 18.37 | 11.94 | 128 ms | 4.22 |

TP3 1025: 60.5 tok/s, 37.44 ms, 2.269 tokens from 4.08 rows, grid 22.73 ms, gap 14.71 ms, TTFT 156 ms (4.16 steps).

Definitions, all from counters the suites already record:
- **decoding requests a step** = tokens a step / (1 + accepted a draft): every decoding
  request drafts every step;
- **verified rows a step** = decoding requests × (1 + verified drafts a draft): the target
  row and its verified drafts;
- **row yield** = tokens a step / verified rows;
- **wasted rows** = verified rows − tokens a step.

Wasted rows are rejected live rows plus dead rows; the counters cannot tell them apart.
#598's model splits 1022's 4.07 rows into 2.89 live and 1.18 dead (0.62 live rows wasted,
about 1.8 ms), but only a counter can measure the split (priority 3).

**The grid-to-serving gap** is the serving step less the grid's step at the same decoding
requests and rows a request, interpolated in both.
- **c1 (TP4, 1025):** 11.9 ms, 39% of the step.
  - The dummy rows read 6 experts a layer; the suite's live rows reach about 13.8. At FC1
    and FC2's marginal 17.2 µs an expert-layer (#598's dgx4 fits), that is about 5.3 ms.
  - The other ~6.5 ms is GPU work: the traced c1 step is 96% covered by kernels (#614's
    `step_shares.json`).
  - **#610 attributes part of it** (draft, runs 46-47 on dgx3). It ran six target layers
    and the drafter at a TP4 rank's shapes, with the recipe's environment, sizing and
    capture ladder. The c1 step is 14.30 ms of GPU busy against the grid's 8.87 (+5.43 ms).
    Scaled to the whole model:
    - **Routing:** the routed GEMMs' +1.19 ms over 9 MoE layers is about 5.7 ms over 43, a
      cross-check of the 5.3 ms above.
    - **The target's verification head:** about 1.0 ms, a rank's quarter of the second
      packed-head launch. The grid's dummy step runs only the drafter's head. This one runs
      eagerly after the target's graph, all on the critical path.
    - **Sampler, sink rejection, verification and inputs:** about 0.07 ms, all exposed.
    - **Slower per-layer kernels:** about 1.0 ms of kernel time over 43 layers, about 0.1 ms
      exposed.
    - **More L2 prefetch:** about 0.5 ms on its side stream, none exposed.
    - **Total:** about 1.2 ms on the critical path, of up to 2.6 ms of kernel time.
  - **The remaining ~5.3 ms is not in a one-GPU step at these shapes:** Engram, collectives
    with real peers, real context. That part needs an all-rank serving trace.
- **c8 and c16:** the grid at 13 requests and 2.8 rows is 31.9 ms against a 123 ms step:
  the grid sees 26% of it. On TP3 c8 it sees 28%. Real text at 36 rows reaches tens of
  experts a layer more than 6.
  - **Grid artifact (#610):** on TP4's 16-stream engine, an 8-request grid step runs the
    drafter's packed head over two row tiles and samples 16 rows, about 1.4 ms that a real
    8-request step does not pay.
  - The interpolation at 13 requests weights the 8-request shapes by 0.37, so about 0.5 ms
    of TP4's 31.9 ms is this artifact. That is small against the 91 ms gap. TP3's grid and
    every c1 grid are clean.
  - The grid verifies 6 rows a request where adaptive verification runs 2-3. Interpolating
    at the served rows covers that.
- **The grid split itself** (1x3 rows at 4K, TP4 1025): 17.98 ms = 14.02 target forward +
  2.54 drafter + 1.42 the rest. The drafter is 14% of the dummy step.

The gap is also a cross-check:
- From r6x to 1025 the grid's c1 shape fell 0.63 ms, but the served c1 step fell only
  0.19 ms, so the gap grew 0.43 ms.
- These are different sessions, so boot drift (up to 1%, about 0.2-0.3 ms) is in it.
- Followed per release, the gap shows when a grid gain does not reach users.

### 2.3 First token

- **TTFT in steps.** Short-prompt TTFT is 4.2 steps at one stream on 1025, against 3.2 on
  r6x.
  - The extra step is sink rejection's held first token: it is emitted after the next step
    verifies it as draft zero. 1022 shipped with it, and its record measured the cost: +41 ms
    for an isolated request.
  - It is the largest single part of c1 TTFT that is neither the prompt's own work nor the
    step already running.
- **The server's split** ([ttft_split.py](ttft_split.py), on the only raw `/metrics` pair
  committed, 2026-10-01; [runs/ttft-split-tool-eval-20261001.txt](runs/ttft-split-tool-eval-20261001.txt)):
  - mean server TTFT 351.5 ms;
  - frontend 28.0 ms (8%), queue 0.1 ms, engine 323.3 ms (92%).
  - The histograms' buckets put the TTFT's p50 and p90 both in 250-500 ms, so only means
    come from counters. That build's buckets are the current tree's.
  - The engine part holds the step running at arrival, the step decided ahead, the prompt's
    chunks, Engram's first-chunk wait and the held step. Counters cannot split it; the
    traced replay (`ttft_trace.py`) can.

### 2.4 Prefill

TTFT fitted as a + b N + c N² over each release's prefill samples (N in tokens):

| Release | a | b: ms a 1K tokens at no context | rise of b per 1M of context | worst residual | 32K over largest |
| --- | ---: | ---: | ---: | ---: | ---: |
| r6x TP4 (to 1M) | 0.37 s | 155.7 | +141.8 | 297 ms | +35.9% |
| 1025 TP4 (to 1M) | 0.42 s | 134.8 | +134.4 | 174 ms | +36.8% |
| r6x TP3 (to 500K) | 0.32 s | 212.1 | +192.1 | 181 ms | +16.4% |
| 1025 TP3 (to 500K) | 0.33 s | 164.1 | +171.9 | 177 ms | +17.1% |

- The marginal cost of a token doubles between no context and 1M on the ring. That is the
  context tax agents pay on long cold prompts.
- A chunk's fixed cost cannot come from one budget, because chunks grow with tokens. #599's
  fits over budgets put it at 139-168 ms (TP4) and 192-203 ms (TP3). At the recipe budgets
  that is 4-6% of a prompt's TTFT on the ring (16,384 tokens a chunk) and 12-13% on the
  triangle (8,192).
- **Exposed communication** ([prefill_shares.py](prefill_shares.py) on the 16K chunk of
  request-inventory-5, ports build, rank 0;
  [runs/prefill-shares-ports-16k-rank0.txt](runs/prefill-shares-ports-16k-rank0.txt)):
  - 2,359 ms window: compute only 74.1%, communication exposed 21.5%, both 3.9%, idle 0.5%;
  - SendRecv took 331-336 ms on every rank, so it is transfer, not waiting for a late peer.
  - That 21.5% bounds what hiding the sequence-parallel exchange can win. #593's ~5% at 15K
    is a part of it.
  - Re-run on a next-candidate trace to see what #593 moved.

### 2.5 Host gaps, collective waits and late peers ([trace_waits.py](trace_waits.py))

Rank 0, request-inventory-5 (ports build, 2026-10-25), from
[runs/trace-waits-ports-rank0.txt](runs/trace-waits-ports-rank0.txt):

| Window | Steps | Step | Host gaps | Collectives | Above each kernel's p5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| c1 greedy | 10 | 34.63 ms | 0.72 ms (2.1%) | 4.81 ms | 3.67 ms |
| c1 sampled | 25 | 29.45 ms | 1.98 ms (6.7%) | 4.64 ms | 3.07 ms |
| mixed c8 | 32 | 58.20 ms | 0.79 ms (1.4%) | 7.20 ms | 3.99 ms |

- **Sampled requests leave the GPU idle 2.7 times as long a step as greedy ones.** That is
  1.3 ms more host time a step, with no record that explains it.
- **Four ranks** (TileLang TP4, 2026-10-03, all ranks traced):
  [runs/trace-waits-tp4-20261003-decode.txt](runs/trace-waits-tp4-20261003-decode.txt) and
  [-c8](runs/trace-waits-tp4-20261003-c8.txt).
  - **Alignment.** The collectives are matched by order. Each node's clock offset is the
    median difference of their end times. The offsets were 4.8 ms (dgx2), 0.36 ms and
    1.15 ms, so raw timestamps would have named the wrong peer; they repeat between the two
    windows.
  - **c1 (39.9 ms steps):** each rank waits 1.0-1.7 ms a step for the last peer. The
    start-to-start skew is 15 µs at p50 and 0.2 ms at p99.
  - **Late arrivals are uneven:** dgx4 is last at 39% of collectives and dgx2 at 10%
    (25% each if even). At c8: 45% and 12%.
  - **The one-rank estimate overstates the aligned wait about twofold** (2.4-2.8 against
    1.0-1.7 ms), so the late-peer metric needs every rank's trace.

### 2.6 Memory and power

From `conditions.json`, over each release suite (the lowest node and its headroom over the
3 GiB serving guard):

| Release | Lowest node | Low | Headroom | Others |
| --- | --- | ---: | ---: | --- |
| 1025 TP4 | dgx1 | 19.87 GiB | 16.87 GiB | 21.95-22.26 |
| 1025 TP3 | dgx1 | 5.99 GiB | 2.99 GiB | 7.47-7.54 |
| r6x TP3 | dgx1 | 4.86 GiB | 1.86 GiB | 6.56-6.58 |

- dgx1 sits 1.5-2.1 GiB below the other nodes on every release (the API server and engine
  core, #599's rank-0 tax).
- The startup low is not recorded. The one committed measurement (2026-10-21, TP3,
  4.5 GiB pool) put dgx1 at 5.34-5.81 GiB against the 5 GiB guard, with a 0.47 GiB spread
  between boots and no code change.
- **Tokens a joule.** Only each node's highest GPU power is kept, so tokens a joule is a
  floor: 0.51 at c1 (TP4 1025) and 0.84 at c16.
  - The one measured value used mean power, 153 W summed over four nodes at c16 (2026-10-06,
    an older build): 1.39 tok/J.
  - So the floor is roughly half the truth. Lab telemetry already has mean power, at 1 Hz.

## 3. The extended metric set

Users: **G** release gate (a verdict row), **N** release notes, **R** research and screens.
Measured: **prod** costs serving nothing (client side or existing counters); **dev** needs
a development arm or flag; **lab** needs a profiling window. "Floor" is the noise a change
must clear, from section 2.1 where measured.

### Throughput

| Metric | Definition | Measured | Floor | Users | Today |
| --- | --- | --- | --- | --- | --- |
| tok/s per stream (c1) | output tokens a second, one conversation at a time, distinct prompts, greedy | prod (single-stream suite) | 0.13% A/A; ±3.7% window interval | G N | yes |
| aggregate tok/s (c8, c16; TP3 c4, c8) | the same, N streams in a closed loop | prod | 0.2-0.5% A/A | G N | yes |
| paired decode-time ratio | geometric mean over prompts of candidate / reference decode time (first to last token), when the greedy text is identical | prod (per-request table) | expect ~0.1-0.3% from the A/A | G | **new** (priority 1) |
| step time | 1 / engine steps a second | prod | 0.05% A/A at c1 | G R | steps/s only |
| tokens a step | output tokens / steps | prod | 0.01% A/A at c1 | G R | yes |
| acceptance by draft position | accepted / verified at positions 1-5, on every run step | prod (per-position counters exist) | follows the text | R N | model step at c16 only |
| unbiased survival | acceptance by position with every draft verified | lab arm (`c1-acceptance.json`, #598) | — | R | never run |
| verified rows, row yield, wasted rows | section 2.2 | prod (derived) | follows tokens a step | R N | **derived now** |
| live and dead rows | rows before and after the on-device survival cut | prod once vLLM counts them | — | R | inferred only |
| byte-floor share | bytes a step at 266.6 GB/s / step | lab (floor.py on a profile) | traced steps ±0.5 ms | R N | one record |
| non-byte budget by cause | below-rate streaming; DRAM idle under small kernels, collectives, attention, nothing | lab | — | R | one record |
| grid-to-serving gap | serving step − grid step at the same requests and rows | prod + dev grid | grid boot drift 0.4% | R | **derived now** |
| grid split | target forward, drafter, the rest, per shape | dev | — | R | in grid.json, unprinted |

### First token

| Metric | Definition | Measured | Floor | Users | Today |
| --- | --- | --- | --- | --- | --- |
| short-prompt TTFT p50 / p99 by concurrency | client time to first token | prod | c1 p50 0.5% A/A; Poisson p99 3.8% (miscalls) | G N | yes |
| TTFT in steps | TTFT / step time | prod (derived) | — | N R | **derived now** |
| server split | frontend, queue, engine: histogram sums and counts per window | prod (`/metrics`) | means exact | R | not scraped |
| engine split | rest of the running step, step decided ahead, prompt chunks, Engram first-chunk wait, held first token | lab (traced replay + Engram timeline flag) | — | R | experiments only |
| per-request TTFT parts | queue, scheduled-to-first-token, first-to-last per request | dev (`--enable-per-request-metrics`, off by default) | — | R | never used |
| cold TTFT by size | prefill step | prod | see prefill | G | yes |
| prefix-hit TTFT | a question's TTFT after a cached document of 64K-1M | prod (`run --context`) | — | N R | ran once (2026-10-14) |
| agent TTFT | TTFT p50 / p90 on the AgentX replay at c4-c16 | lab window | — | N R | never run |
| short request behind a long prefill | TTFT of short requests sent during a 131K cold prefill (TODO.md) | prod (a stall-step probe) | — | R N | not measured |

### Prefill

| Metric | Definition | Measured | Floor | Users | Today |
| --- | --- | --- | --- | --- | --- |
| tok/s by size | prompt tokens / TTFT, 32K-1M cold | prod | 0.2-0.8% A/A | G N | yes |
| TTFT curve | a, b and the context rise from the fit | prod (derived) | residual ≤0.3 s | N R | **derived now** |
| context tax | rate at 32K / rate at the largest size − 1 | prod (derived) | — | N | **derived now** |
| chunk fixed cost | ms a chunk, from TTFT across budgets (`fit_chunks.py`, #599) | lab sweep | — | R | per budget study |
| exposed communication | share of a chunk when only collectives run | lab | — | R | **prototype** |
| family shares | routed MoE, attention, dense GEMMs, communication, mHC | lab | — | R | **prototype** |
| prefill under decode | prefill tok/s with N streams decoding | prod (stall step) | — | R | experiments only |

### Memory

| Metric | Definition | Measured | Floor | Users | Today |
| --- | --- | --- | --- | --- | --- |
| serving low per node | lowest MemAvailable during the suite; headroom over 3 GiB | bench | — | G N | yes |
| startup low per node | lowest MemAvailable from launch to ready; margin over 5 GiB | prod (memguard already samples it at 0.25 s) | 0.47 GiB between boots | G N | breaches only |
| rank-0 tax | dgx1's low less the other nodes' median | prod (derived) | — | R | derived now |
| KV pool use | peak and mean `kv_cache_usage_perc` per window; against `kv_capacity.py`'s admission ceiling | prod | — | N R | not scraped |
| reserved-but-idle workspace | each lane's reserve against its peak use (`VLLM_DEBUG_WORKSPACE`, `lane_check.py`) | lab | — | R | #599 only |
| allocator slack | reserved − allocated and inactive split bytes at the low (the memory-trace overlay) | lab | — | R | #599 tooling, unrun |

### Stalls

| Metric | Definition | Measured | Floor | Users | Today |
| --- | --- | --- | --- | --- | --- |
| decode slowdown during a long prefill | the overlapping steps' p50 / the baseline step p50; the worst wait; decode tokens lost against the baseline rate | prod (stall step) | — | G N | experiments only |
| steps carrying prefill | share of steps with prompt tokens | prod (`workload`) | — | R | threshold misreads TP4 decode |
| host gaps | no-kernel time inside a step | lab | — | R | **prototype** |
| collective waits and late peers | each rank's wait for the last arrival; each node's share of last arrivals | lab (every rank's trace) | — | R | **prototype** |
| prefetch not landed | excess time of consumers whose L2 fill was late or lost (#615) | lab | — | R | #615 record |
| Engram wait | GPU wait on Engram rows per step and per first chunk | debug flag | — | R | experiments only |

### Cache

| Metric | Definition | Measured | Floor | Users | Today |
| --- | --- | --- | --- | --- | --- |
| prefix-cache hit rate | cached prompt tokens / prompt tokens (`prompt_tokens_cached`), per window | prod | — | N R | queries/hits only |
| miss cost | TTFT against uncached tokens, fitted (AgentX `miss_cost`) | lab window | — | R | never run |
| retention | block lifetime, idle before eviction and reuse gap (`--kv-cache-metrics`, 1% sampled); hit rate against pool use | dev | — | R | never run |

### Efficiency

| Metric | Definition | Measured | Floor | Users | Today |
| --- | --- | --- | --- | --- | --- |
| GPU tokens a joule per step | output tokens / (each node's mean GPU power summed × window) | lab telemetry (1 Hz, exists) or NVML's energy counter | — | N R | one experiment |
| tokens a joule floor | the same with each node's highest power | prod (derived) | — | — | **derived now** |

## 4. Priorities, with what each costs

1. **Paired per-prompt verdicts** (servebench, S).
   - `run` keeps a compact per-request table: prompt id, prompt and output tokens, sent,
     first and last token times. That is about 100 rows at c1 and 350 at c16.
   - `compare` pairs requests by prompt id when both runs' token counts match (identical
     text, which the determinism gate checks). It gives the geometric mean of decode-time
     ratios with a t interval over prompts, with Welch over windows as the fallback.
   - Bound: the c1 A/A is 0.06-0.13%. Every 100 tok/s lever in reach is +0.6-4% at c1
     (#598's list), and today's ±3.7% cannot see any of them.
2. **The step budget per release** (lab, S-M).
   - Each release comparison adds one `profile` job on the candidate, every rank: c1 greedy,
     c1 sampled, c8 and a 16K prefill, about 15 minutes.
   - It records floor.py's budget, trace_waits.py's host gaps, collective waits and late
     peers, and prefill_shares.py's exposed communication, as release notes, not a gate.
   - #610's `step_diff.py` already diffs a decode step against the grid's on one GPU. It
     attributes about 1.2 ms of the c1 gap's ~6.5 ms non-routing part, mostly the target's
     verification head.
   - The remaining ~5.3 ms (Engram, real peers, real context) needs that diff on a serving
     trace of every rank. So the release profile job should also trace one grid shape at the
     c1 rows on the same boot, with every rank (#610's README says what a multi-rank trace
     needs).
3. **Rows a step** (servebench S; vLLM M).
   - servebench's `prom` scrapes the per-position counters on every run step.
   - vLLM counts live and dead rows: the cut index rides on the accepted-count copy the
     sampler already makes, so no new sync. It needs the grid's verdict that it costs
     nothing.
   - `c1-acceptance.json` (#598) runs whenever the drafter, sink rejection or the
     verification policy changes.
4. **Agent latency** (lab L; servebench S).
   - The AgentX window gives hit rate, miss cost and retention, with `--kv-cache-metrics`
     on its arm.
   - servebench scrapes the TTFT, queue and prefill histograms' sums and counts, and
     `prompt_tokens_cached`, on every run.
   - The traced TTFT replay becomes a servebench `ttft` step, so the engine split is part of
     a suite and not one experiment.
5. **A routed grid** (vLLM dev M; servebench S).
   - Dummy steps take their routes from a fixed table: the reference forward's routes
     (`2026-10-18-json-count`), or a recorded serving window's routes, stored by release.
   - The grid stays text-independent and deterministic, and reads serving's expert counts.
   - Off unless the dev RPC asks for it.
6. **The stall step in the release suites** (suite S, about 4 minutes a build).
   - A new suite version (release v3, release-tp3 v2) adds `stall` at 131K with 8 streams.
   - It also adds a probe that sends short requests during the prefill and records their
     TTFT.
   - The 1025 notes' "about 1.7 times as long" compares the 8,192 budget's stall step with
     the 4,096 budget's. Against plain decode (0.100 s), the steps during a 131K prefill
     were 8.6-14.4 times as long ([2026-10-19-prefill-budget-tp3](../2026-10-19-prefill-budget-tp3/README.md)).
7. **Memory: the startup low every boot** (memguard S).
   - memguard keeps the minimum it already samples and logs it once when the startup phase
     ends (`startup low N KiB, margin M KiB`). bench copies it into `conditions.json`.
   - Also scrape `kv_cache_usage_perc` peaks per window.
8. **Energy** (lab S). Copy each step's `telemetry.json` into the suite run, so manifests keep
   mean power. Check once on dgx4 whether GB10 reports NVML's total-energy counter, which
   would replace sampling with two reads.

Items 1, 3 (per-position), 4 (histograms), 6, 7 and 8 change no serving code. Item 3's live
and dead counters and item 5's routed grid are vLLM patches. Item 3's counter must pass a
grid A/B.

## 5. Proposed tooling changes (not built here)

- **servebench**
  - `run`: a compact per-request table in every suite step.
  - `prom`: per-position counters, `kv_cache_usage_perc` (window maximum),
    `prompt_tokens_cached`, `request_prefill_kv_computed_tokens`, and the latency
    histograms' sums and counts.
  - `compare`: paired by prompt, with a bootstrap interval for latency percentiles; Poisson
    TTFT p99 out of the verdict rows until then.
  - `engine --routes FILE` (with the vLLM hook).
  - A `ttft` step from `ttft_trace.py`.
  - `stall --probe-short`.
- **Suites:** release v3 and release-tp3 v2 add the stall step. single-stream v2 keeps
  per-request rows. Existing suites and their references stay as they are; a changed suite
  is a new suite.
- **Lab:** a `profile` job in the release protocol; telemetry copied into suite runs and
  promotion manifests; the grid-and-serving diff trace; AgentX when a window is free.
- **Tables:**
  - `servebench_suite_table.py` prints a core-metrics block (core_metrics.py's card for each
    arm).
  - The verdict block adds each metric's A/A floor from noise_floor.py beside the run's own
    repeat.
  - The grid's per-pair blocks use the boot floor.
- **vLLM:** live and dead row counters; the routed dummy-step hook (dev only). Finer
  TTFT and queue buckets are not needed if dev arms use `--enable-per-request-metrics`.
- **`bin/spark`:** memguard's startup low into `conditions.json`; `workload`'s
  carrying-prefill test should count prompt tokens (`vllm:prompt_tokens_total` per step),
  not total tokens above 64. A TP4 decode step can emit up to 96.

## 6. Prototypes and how to run them

All are standard library only, run on the laptop, and write a text table (and JSON with
`--json`):

```sh
python3 -I experiments/2026-10-26-core-metrics-plan/core_metrics.py --json experiments/2026-10-26-core-metrics-plan/runs/core-metrics.json
```

```sh
python3 -I experiments/2026-10-26-core-metrics-plan/noise_floor.py --json experiments/2026-10-26-core-metrics-plan/runs/noise-floor.json
```

```sh
python3 -I experiments/2026-10-26-core-metrics-plan/ttft_split.py experiments/2026-10-01-tool-eval-independent/runs/metrics-before.txt experiments/2026-10-01-tool-eval-independent/runs/metrics-after.txt
```

```sh
python3 -I experiments/2026-10-26-core-metrics-plan/trace_waits.py results/private/tilelang-tp4-performance/profile/traces/tilelang/decode/dgx1/*.gz results/private/tilelang-tp4-performance/profile/traces/tilelang/decode/dgx2/*.gz results/private/tilelang-tp4-performance/profile/traces/tilelang/decode/dgx3/*.gz results/private/tilelang-tp4-performance/profile/traces/tilelang/decode/dgx4/*.gz
```

```sh
python3 -I experiments/2026-10-26-core-metrics-plan/prefill_shares.py results/private/determinism/request-inventory-5/ports/prefill-trace/dgx1/dp0_pp0_tp0_dcp0_ep0_rank0.1792907036192468747.pt.trace.json.gz
```

- `core_metrics.py` groups `manifests/suites/<date>-<baseline>-<shape>[-single-stream|-sampled]`
  by release.
- `noise_floor.py` reads every release table's A/A block.
- `trace_waits.py` takes one rank's trace for rank-local figures, or every rank's trace of
  one window for late peers. It refuses traces that are not one window's ranks.
- The traces are git-ignored lab outputs under the main checkout's `results/private/`.

Receipts in [runs/](runs/):
- `core-metrics.{txt,json}`;
- `noise-floor.{txt,json}`;
- `ttft-split-tool-eval-20261001.{txt,json}`;
- `trace-waits-ports-rank0.{txt,json}` and `trace-waits-ports-mixed-c8-rank0.json`;
- `trace-waits-tp4-20261003-{decode,c8}.{txt,json}`;
- `prefill-shares-ports-16k-rank0.{txt,json}`.

## 7. Caveats

- **The A/A floors are same-session.** Each pair is two boots in one lab session.
  - Across sessions, host changes have moved references (the 2026-10-16 APST tuning), and
    this record measures none of that.
  - Pairs are few: three single-stream, four c16.
- **The grid-to-serving split is an estimate.**
  - The 5.3 ms of routing takes #598's 13.8 experts a layer and its FC1 and FC2 fits. #610's
    one-GPU diff, scaled from 9 MoE layers to 43, gives 5.7 ms.
  - Of the rest, #610 attributes about 1.2 ms of critical path from six layers on one GPU,
    scaled to the model. About 5.3 ms is still a residual.
  - At c8 and c16 the expert counts are not measured at all.
- **Trace figures come from profiled windows.**
  - Python stacks inflate host gaps; the traced c1 step is 34.6 ms against 30.3 ms served.
  - The four-rank traces are a 2026-10-03 build. They demonstrate the method; they say
    nothing about 1025's peers.
- **The prefill trace is one 16K chunk on rank 0, before #593.**
- **The TTFT split's fixture is a 2026-10-01 build at tool-evaluation load.** It shows the
  method, not today's numbers.
- **The tokens-a-joule floor uses GPU power only.** Nothing measures whole-system power.
