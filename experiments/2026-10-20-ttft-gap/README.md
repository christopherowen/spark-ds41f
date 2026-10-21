# Short-prompt time to first token: where it goes

Base: promoted r6x (`2026-10-19-karmic-kraken-r6x`, image `vllm-ds41f-kkref:19f2c20ed4d6-r6x`,
vLLM patch head 7316c7f9), recipe at origin/main 6aed2926, the TP3 triangle at 1M. TP4 figures
come from r6x's ring release (2026-10-19).

Question: why does a short prompt's time to first token (TTFT) take about a quarter to a third
of a second at four closed-loop streams on TP3, and which levers could shorten it?
Constraints: native weights, identical outputs, no cadence throttles (a prefill pauses only
when other work needs the step).

This is a ceiling study. Nothing was built; no configuration changes.

## 1. TTFT is about three serving steps

From the recorded release suites (servebench closed loops, about 20-token prompts, thinking
off, greedy, 256 tokens). The serving step is 1 / engine steps per second over the window.

| Shape | Streams | TTFT p50 | Serving step | TTFT / step |
| --- | ---: | ---: | ---: | ---: |
| TP3 | 1 | 125 ms | 37.6 ms | 3.32 |
| TP3 | 4 | 254 ms | 73.9 ms | 3.44 |
| TP3 | 8 | 359 ms | 108.3 ms | 3.31 |
| TP4 | 1 | 98 ms | 30.5 ms | 3.21 |
| TP4 | 8 | 290 ms | 86.5 ms | 3.35 |
| TP4 | 16 | 400 ms | 122.9 ms | 3.25 |

Sources: `manifests/suites/2026-10-19-karmic-kraken-r6x-tp3{,-single-stream}` and
`…-r6x-tp4{,-single-stream}` (`summary.latency.ttft_s`, `summary.server.steps_per_s`).

The serving step is not the engine grid's step: the grid replays one token on every row, so
its rows share experts (TP3 4x12 rows: 27.6 ms). Real text routes each row to its own experts:
a profiled one-stream serving step spends 23.8 ms in routed MoE and 11.7 ms in dense GEMMs
(`experiments/2026-10-19-prefill-ceiling` traces, `decode` workload, rank 0).

## 2. Mechanism: the step decided ahead

`--async-scheduling` runs the engine core with a two-batch queue
(`vllm/v1/engine/core.py`, `step_with_batch_queue`, `max_concurrent_batches` = 2). As soon as
step N is under way, step N+1 is scheduled, its inputs prepared and its CUDA graphs launched
behind N. The engine core then blocks on N's output. A request that arrives meanwhile waits in
the engine core's input queue until N's output is processed. By then N+1 is fixed, so the
earliest step the request can join is N+2.

The r6x one-stream decode trace shows the lead (`decode` workload, rank 0, launches matched to
kernels by correlation id). Each step is three graph launches: the target (1,711 kernels), the
sampler (24) and the drafter (174, 3.8 ms).
- The worker receives a step's scheduler output at 45.97 ms, 2.7 ms after the previous GPU step
  ended, and launches its graphs by 55.7 ms.
- Those graphs run on the GPU from 88.66 to 137.30 ms.
- So the decision leads the GPU by about 42 ms (one profiled step). Preparing and launching
  takes about 10 ms of host time under the profiler.

Every step that carries prompt rows runs eagerly, including the other streams' decode rows in
that step:
- DS4.1 is not torch-compiled: the serving log says "`torch.compile` is turned on, but the
  model … does not support it".
- `VLLM_USE_BREAKABLE_CUDAGRAPH` is 0.
- So the recipe's FULL_AND_PIECEWISE captures only full graphs of uniform decode batches (24
  of them, plus DSpark's).
- A one-token prompt is decode-shaped and replays one. The r6x one-row prefill trace spends
  28.2 ms in the target and 4.1 ms in the drafter on the GPU, in three graph launches.

## 3. Traced replay (measured, 2026-10-20 20:59-21:06 UTC)

`ttft_trace.py` ran on production (r6x TP3, no hold, the server idle for the
`bin/spark workload --duration 15` check, no foreign requests in any phase). It sends
servebench's default prompts as servebench does: fresh cache salts, exactly 256 tokens,
thinking off, greedy. It keeps every streamed chunk's arrival and reads the server's counters
around each phase:
- twelve requests one at a time on an idle engine, counters read around each;
- four streams for 120 s;
- eight streams for 60 s.

`analyze_trace.py` rebuilds step ends from the streams' chunks (the step count matches the
server's: 1,745 and 703). It splits each request's TTFT into:
- the time until the first step end after it was sent;
- whole steps that do not carry it;
- the step that produces its first token.

Requests in each phase's opening burst are excluded from the closed-loop rows below.
Analysis: `runs/r6x-tp3-servebench-analysis.json`. Raw trace: dgx1
`~/spark-lab/ttft-gap/r6x-tp3-servebench.json`, sha256 5ca4eae8.

```sh
scp experiments/2026-10-20-ttft-gap/ttft_trace.py experiments/2026-10-20-ttft-gap/analyze_trace.py dgx1:spark-lab/ttft-gap/
ssh dgx1 'cd ~/spark-lab/ttft-gap && python3 ttft_trace.py --repo ~/projects/spark-ds41f --output r6x-tp3-servebench.json'
ssh dgx1 'cd ~/spark-lab/ttft-gap && python3 analyze_trace.py r6x-tp3-servebench.json --json r6x-tp3-servebench-analysis.json'
```

**Every request waited exactly one step that did not carry it** (53 of 53 at four streams, 33
of 33 at eight).

Means per request, with the plain step (a step that produces no first token) as the
reference:

| Part | 4 streams | Share | 8 streams | Share |
| --- | ---: | ---: | ---: | ---: |
| rest of the running step | 70.6 ms | 27% | 102.9 ms | 29% |
| the step decided ahead | 62.0 ms | 24% | 97.2 ms | 28% |
| its own step | 125.7 ms | 49% | 151.8 ms | 43% |
| TTFT (client) | 258.4 ms | | 352.4 ms | |
| TTFT p50 | 254.5 ms | | 339.4 ms | |
| plain step p50 | 70.8 ms | | 100.2 ms | |
| prompt tokens (mean) | 20.7 | | 20.2 | |

Notes on the parts:
- **The rest of the running step** is almost a whole step in a closed loop. A stream's next
  request is sent about 3 ms after its previous request's last step ended, so it lands near
  the start of the next step.
- **The step decided ahead** is shorter than a plain step. The finished stream's rows have
  already left it.
- **Its own step** is 55 ms (four streams) and 52 ms (eight) longer than a plain step.
- **The server's view agrees.** At four streams the queue time is 0.0 ms and the prefill time
  186.8 ms (the decided step plus its own step). The wait for the running step is spent before
  the request reaches the scheduler, which is why the queue metric reads zero. Server TTFT is
  255.8 ms.

**Isolated requests** (idle engine, one at a time, servebench prompts of 17-24 tokens):
- client TTFT 97-116 ms;
- the server's first-token latency is 2.3 ms less (delivery);
- queue time 0;
- 2.7 ms between the API's arrival stamp and the engine (chat template, tokenization, IPC);
- the rest is the engine's prefill time, 92-111 ms.

Host and delivery together are about 5 ms (2% at four streams): not a lever.

## 4. The prompt's own step

On an idle engine, prefill time = 68.1 ms + 1.58 ms per prompt token. This is a least-squares
fit over 24 isolated requests of 17-74 tokens: these twelve, plus twelve from a second prompt
set run the same way; residual sd 3.6 ms. The second set has prompts of 28-74 tokens, kept
outside the repository; its trace is dgx1 `~/spark-lab/ttft-gap/r6x-tp3-set2.json`, sha256
009bea68.

The eager path's fixed cost is the 68 ms intercept against the one-row prefill's 32 ms
decode-shaped graph. Some of that difference is prefill-only work a graph would still do: CED
metadata, compressor and indexer prefill, the Engram rows of a prompt nobody hashed a step
ahead.

Beside three or seven decoding streams, the prompt step's extra over a plain step fits about
20 ms + 1.58 ms per prompt token:

| Workload | Prompt tokens (mean) | Own step − plain step | 20 ms + 1.58 ms/token |
| --- | ---: | ---: | ---: |
| 4 streams | 20.7 | 55 ms | 53 ms |
| 8 streams | 20.2 | 52 ms | 52 ms |
| second set, 4 streams, 28-48-token prompts | 34.4 | 72 ms | 74 ms |
| second set, 4 streams, 48-74-token prompts | 55.4 | 107 ms | 108 ms |

**What the extra is: GPU work, not host launches.** Measured with a profile of one four-stream
step that carries a 23-token prompt (section 6):
- the eager host launching is hidden behind the previous step;
- the GPU idles 4.2 ms of that step's 135.9 ms, against about 1.5 ms in a plain step;
- the step's extra over a plain four-stream step is routed MoE (+40.8 ms, expert weights for the
  prompt's rows) and dense GEMMs (+7.0 ms).

The fixed 20 ms of the fit is therefore mostly the dense GEMMs and the MoE's first rows, not
host overhead.

The drafter is small: 3.8-4.1 ms of a step's GPU time. It is on the path, because the first
token returns with the whole step's output.

## Levers by ceiling (TP3, four streams, about 20-token prompts)

1. **Decide the next step later (late admission).** Ceiling: the step decided ahead.
   - Four streams: −62 ms (−24%). Eight streams: −97 ms (−28%).
   - With 34-55-token prompts at four streams: −65 to −67 ms (−20 to −22%).
   - Burst arrivals also gain. The first request of a burst on an idle engine prefills alone,
     and the rest wait that step plus a decided one (the opening bursts above: 321-336 ms).

   Async scheduling exists to hide the host's preparation (about 10 ms profiled, section 2),
   not to fix the batch a step early. Deciding step N+1 when step N's GPU work reaches a fixed
   point would keep the GPU fed and admit every request that arrived before that point, as
   long as the point leaves more than the host's preparation before N ends. No request waits
   for anything but work, so it fits the owner's rule. It does not help one stream: the step
   after a finished request's last token is decided before the finish is known.
2. **Graph the steps that carry short prompts.** Measured ceiling: about 3 ms (section 6).
   - A step carrying a prompt keeps the GPU busy: 4.2 ms idle against about 1.5 ms in a plain
     step.
   - The eager host time is hidden behind the step before.
   - Breakable piecewise graphs (`VLLM_USE_BREAKABLE_CUDAGRAPH=1`) would recover only those idle
     milliseconds. Not worth an arm.
   - The L2 weight prefetch barely runs in eager steps (0.9 ms of prefetch kernels against
     14 ms in graphed steps), and the step's dense GEMMs are 7 ms slower. Running the prefetch
     in eager steps is a separate few-millisecond lever; it needs its own profile to confirm.
3. **TP4.** Every step term shrinks with the step: TP4's eight-stream serving step is 0.80x
   TP3's. Needs the ring cabled; not a code change.
4. **Prompt rows.** 1.58 ms a token: 33 ms (13%) at 20 tokens, 54-87 ms at 34-55 tokens.
   Native-precision MoE weight traffic, so only kernel work changes it. TP4 cuts per-rank bytes
   by a quarter.
5. **Return the first token before the drafter.** At most about 4 ms (2%).
6. **Host and delivery.** About 5 ms (2%), measured.

Not levers:
- cadence or quota throttles (rejected 2026-10-19);
- lossy weights;
- turning async scheduling off in production: level on short prompts at four and eight
  streams (section 5), but unmeasured at one stream and on long prefills;
- the rest of the running step. A request cannot join a step that has started; only shorter
  steps shorten this wait.

## 5. Ceiling A/B: synchronous scheduling (2026-10-21 08:10-08:40 UTC)

`lab/ttft-async-ab-fast.json` in the shared lab queue, one session, no foreign requests:
- **Arms:** `lab/async-off.json` (`--no-async-scheduling --disable-cascade-attn`), then
  `lab/r6x.json`.
- **Per arm:** the traced replay, servebench c4 and c8 (30 s warm-up, 120 s) and
  `token_determinism_text.py --wide`.
- **Records:** the analyses are in `runs/ab-async-off-analysis.json` and
  `runs/ab-r6x-analysis.json`; the raw traces are on dgx1 under `results/private/lab/ttft-gap/`.

| | async off | r6x (async) |
| --- | ---: | ---: |
| traced TTFT p50, 4 streams | 189.9 ms | 256.2 ms |
| traced TTFT p50, 8 streams | 246.6 ms | 349.8 ms |
| requests that waited a step not carrying them | 0 of 86 | 86 of 86 |
| servebench TTFT p50, c4 / c8 | 0.188 / 0.259 s | 0.257 / 0.361 s |
| output tok/s, c4 | 117.1 ± 4.1 | 116.6 ± 4.0 |
| engine steps/s, c4 | 13.37 ± 0.39 | 13.40 ± 0.39 |
| output tok/s, c8 | 153.2 ± 3.0 | 151.6 ± 3.4 |
| engine steps/s, c8 | 9.15 ± 0.28 | 9.12 ± 0.26 |
| plain step p50, 4 / 8 streams | 71.2 / 99.8 ms | 71.2 / 99.9 ms |
| determinism | identical, 62313cb5 / 9742ac2c | identical, 62313cb5 / 9742ac2c |

**The ceiling holds.** Without the decided step, TTFT falls 26% at four streams and 30% at
eight, as section 3 predicted (−62 and −97 ms). The step a request runs in and the rest of the
running step are unchanged.

**What async scheduling buys here is not measurable at four and eight streams.** Step rates
are within 0.3%, inside their intervals. The host work it hides is small next to a 70-110 ms
step.

The A/B did not measure where async could still pay:
- one stream, the shortest step;
- long prefills, where the Engram reader reads a chunk's rows during the step before.

## Next: late admission

Rather than turn async scheduling off, keep it and decide each step late in the step before:
[2026-10-21-late-admission](../2026-10-21-late-admission/README.md) (vLLM 0222,
`VLLM_LATE_ADMISSION_LAYER`). Its screen compares late admission with r6x and with
synchronous scheduling on:
- this traced replay;
- servebench c4 and c8;
- one stream;
- cold 32K and 131K prefills;
- determinism.

## 6. Profile of a step carrying a prompt (2026-10-21 08:41 UTC)

Spec `lab/ttft-mixed4-profile.json`, arm `lab/r6x-prof.json`:
- the arm is r6x with the torch profiler recording ten steps after `/start_profile`;
- its KV pool is 4.5 GiB, because at 4.906 GiB the profiler's startup memory took dgx1 below
  its 5 GiB startup guard (08:09 UTC, 4.99 GiB);
- `profile_mixed4.py` lets three streams decode, starts the profiler and sends a 23-token
  prompt.

Rank 0's trace is on dgx1 at `cache/kkref/profiles/ttft-gap-mixed4/`.
`analyze_profile.py` gives `runs/mixed4-profile-rank0.txt`:

| Step (rank 0) | Host launch | GPU span | GPU busy | GPU idle |
| --- | ---: | ---: | ---: | ---: |
| 3 streams, plain (graph) | 5.8 ms | 63.2 ms | 61.6 ms | 1.6 ms |
| 3 streams + 23-token prompt (eager) | 102.8 ms | 135.9 ms | 131.7 ms | 4.2 ms |
| 4 streams, plain (graph) | 7.4 ms | 81.5 ms | 79.9 ms | 1.5 ms |

Kernel time in the prompt step against the plain four-stream step after it:

| Kernels | Prompt step | 4 streams |
| --- | ---: | ---: |
| routed MoE | 96.8 ms | 56.0 ms |
| dense GEMMs | 19.2 ms | 12.1 ms |
| collectives | 6.4 ms | 5.2 ms |
| mHC, sparse MLA, rest | 8.9 ms | 9.3 ms |
| L2 weight prefetch (side stream) | 0.9 ms | 14.0 ms |

What the profile shows:
- **The host is not the bottleneck.** Async scheduling launched the eager step's kernels while
  the step before ran, so the GPU idles only about 3 ms more than in a plain step. Graphing
  these steps has nothing left to win.
- **The cost is routed MoE.** Its kernel time grows about 2.7 ms per decode row between the
  three- and four-stream plain steps (39.9 to 56.0 ms), and the prompt's rows add experts the
  same way. That is native-precision weight traffic; only kernel work or TP4 changes it.
- **The L2 prefetch barely runs in the eager step,** and the step's dense GEMMs are 7 ms slower.
  A small, separate lever.
