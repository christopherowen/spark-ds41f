# Late admission: decide each step when the step in flight reaches a decoder layer

Owner, 2026-10-21: "we should reduce the ttft", then "can we cut that time down significantly
and push?"

**Status:** vLLM branch `late-admission`, one commit on r6x's patch head 7316c7f9f
(91eaaa59e), in ~/projects/vllm-ds41-late-admission; the series patch
[vllm/0222-Late-admission-decide-each-step-when-the-step-in-fli.patch](vllm/0222-Late-admission-decide-each-step-when-the-step-in-fli.patch)
(provisional number 0222). On dgx4 ([runs/tests.txt](runs/tests.txt)):
- **Tests:** 38 pass, the GPU test included.
- **End-to-end smoke test** (dummy weights, V2 runner, async scheduling, multiprocess
  executor): greedy outputs identical, TTFT −32%, no request waiting a step not carrying it, the
  plain step level.

**Screen passed** (2026-10-21 11:05-11:54 CET; see "Screen" below). Late admission at K = 30:
- cuts TTFT as much as synchronous scheduling does: −26% at four streams, −30% at eight;
- keeps what async scheduling buys: synchronous scheduling loses about 3% of long-prefill
  throughput and 2% of one-stream step rate;
- leaves outputs identical and adds no measurable startup memory.

The release comparison is next, once dgx1's startup peak lets a production-sized pool boot
reliably.

## Why

[2026-10-20-ttft-gap](../2026-10-20-ttft-gap/README.md) traced short requests on production
r6x TP3. Every request waited exactly one step that did not carry it:
- 62 ms of a 258 ms time to first token at four streams;
- 97 ms of 352 ms at eight.

`--async-scheduling` runs the engine core with a two-batch queue. As soon as step N is under
way, step N+1 is decided and launched behind it, and the engine core blocks on N's output. A
request that arrives during N is read only after that output, when N+1 is already fixed, so
it joins N+2.

## The change (vLLM 0222)

`VLLM_LATE_ADMISSION_LAYER=K` moves the decision of N+1 to the moment N's GPU work reaches
decoder layer K:
- **Worker:** a forward pre-hook on layer K records one CUDA event. In a full CUDA graph the
  record is captured as an external event node, so every replay records it as it reaches the
  layer; eager steps (those carrying prompt rows) record it as they pass.
- **Reply:** the V2 runner's `execute_model` returns an asynchronous reply that waits on the
  event. WorkerProc's output thread sends replies in order, so the output rank's reply reaches
  the engine core when the step reaches layer K, before the step's sampled output.
- **Engine core:** before deciding a step while another is in flight, it waits for that
  step's `execute_model` reply, takes the requests that arrived meanwhile (`_admit_arrivals`),
  then decides.

Layers K and later, sampling and drafting still hide the decision and the next step's
preparation. Unchanged:
- an idle engine decides at once;
- a step without a forward replies at once;
- the default, −1, keeps today's behaviour;
- startup refuses the setting without async scheduling or with pipeline parallelism.

Outputs cannot change: no tensor is read or written, only which requests share a step, and
batch invariance already covers that.

## Choosing K

The lead from layer K to the step's end must cover the host's work for the next step:
- the engine core's decision;
- the RPC to every rank;
- the worker's input preparation and graph launch.

The async-scheduling A/B ([2026-10-20-ttft-gap](../2026-10-20-ttft-gap/README.md), section 5)
measured that work as the step time synchronous scheduling adds. At four and eight streams it
is not measurable: steps per second are within 0.3%, inside their intervals.

Two cases may still need lead, and the A/B did not measure either:
- **One stream**, the shortest step: about 38 ms, roughly 0.85 ms per target layer plus about
  4 ms of sampling and drafting.
- **Long prefills.** The Engram reader reads a chunk's rows while the step before runs; a 4K
  chunk's reads take about 65 ms.

K = 30 leaves layers 30-39, sampling and drafting:
- about 13 ms of a one-stream step;
- about a quarter of a prefill chunk's step (about 250 ms).

The dummy-weight smoke test kept its plain step level with the decision at layer 18 of 24,
about 4 ms before the end of a 16.7 ms step.

## Files

| File | What it is |
| --- | --- |
| [overlay/](overlay) | the changed files, mounted over the r6x image (whose vLLM tree is the patch's base) |
| [make_arms.py](make_arms.py) | writes `late-tp3.json` (candidate), `r6x-tp3.json` (reference), `screen-suite.json` and `screen-tp3.json` for a K; `--staged DIR` writes a copy that mounts from DIR, to run before merging |
| [late-tp3.json](late-tp3.json), [r6x-tp3.json](r6x-tp3.json), [async-off-tp3.json](async-off-tp3.json), [screen-suite.json](screen-suite.json), [screen-tp3.json](screen-tp3.json) | K = 30 and a 4.5 GiB KV pool for every arm (`make_arms.py 30 --kv-bytes 4831838208`), as generated |
| [runs/run-tests.sh](runs/run-tests.sh) | CPU and GPU tests in a throwaway container on an idle node |
| [runs/smoke.sh](runs/smoke.sh), [runs/smoke_client.py](runs/smoke_client.py) | the end-to-end smoke test on one idle node |

## First boot (2026-10-21 08:56-08:59 UTC)

A trimmed copy of the screen with production-sized arms did not boot:
- dgx1's startup memory guard stopped the late-admission arm at 4.99 GiB MemAvailable, against
  its 5 GiB floor;
- that was after model load and every graph capture with `VLLM_LATE_ADMISSION_LAYER=30`;
- the mixed-step profile arm of the ttft-gap experiment stopped the same way at 08:09.

Production-sized lab boots on the triangle run within tens of MB of the floor. Every arm of the
screen now uses a 4.5 GiB KV pool, which these measurements do not depend on. The release
comparison needs the production pool, and so a startup peak below the guard.

## Screen (2026-10-21 09:05-09:54 UTC)

Run `late-admission-l30-screen-kv45`, queued by the coordinating session as
`20261021T090217-late-admission-screen-kv45.json`. It is this experiment's `screen-tp3.json`
without the servebench c8 step (the traced replay covers eight streams):
- arms `late-kv45.json` (extends `late-tp3.json`), `r6x-kv45.json` and `async-off-kv45.json`;
- each is the recipe's serve arguments with a 4.5 GiB KV pool, staged under
  `results/private/lab/late-admission/` on dgx1-3;
- no foreign requests in any phase.

Per-arm summaries are in [runs/screen-kv45-summaries.txt](runs/screen-kv45-summaries.txt);
servebench and suite outputs are on dgx1 under `results/private/lab/late-admission/`.

| | late admission, K = 30 | r6x (async) | synchronous |
| --- | ---: | ---: | ---: |
| traced TTFT p50, 4 streams | 190.9 ms | 258.8 ms | 188.3 ms |
| traced TTFT p50, 8 streams | 249.0 ms | 353.9 ms | 247.2 ms |
| requests that waited a step not carrying them | 0 of 86 | 85 of 85 | 0 of 86 |
| plain step p50, 4 / 8 streams | 71.2 / 99.9 ms | 71.3 / 100.7 ms | 71.6 / 99.9 ms |
| c4 output tok/s | 117.0 ± 3.8 | 116.0 ± 4.6 | 117.0 ± 4.1 |
| c4 TTFT p50 (servebench) | 0.191 s | 0.259 s | 0.189 s |
| one stream, output tok/s | 57.2 ± 2.7 | 56.7 ± 3.1 | 55.5 ± 3.3 |
| one stream, engine steps/s | 27.28 ± 0.50 | 27.05 ± 0.57 | 26.52 ± 0.58 |
| one stream, TPOT p50 | 17.5 ms | 17.6 ms | 18.2 ms |
| one stream, TTFT p50 | 124 ms | 126 ms | 102 ms |
| cold prefill 32K | 4442 ± 41 tok/s | 4461 ± 44 tok/s | 4332 ± 60 tok/s |
| cold prefill 128K | 4372 ± 23 tok/s | 4386 ± 36 tok/s | 4258 ± 33 tok/s |
| determinism | identical, 62313cb5 / 9742ac2c | identical, 62313cb5 / 9742ac2c | identical, 62313cb5 / 9742ac2c |
| dgx1 startup low (4.5 GiB pool) | 5.456 GiB | 5.809 GiB | 5.339 GiB |

What the screen shows:
- **TTFT.** Late admission removes the decided-ahead step as completely as synchronous
  scheduling. TTFT is −26% at four streams and −30% at eight against r6x, within 3 ms of
  synchronous.
- **Where async scheduling still pays.** Synchronous scheduling is about 3% slower on long
  prefills, beyond both intervals at 32K and 128K. Its one-stream step rate is 2% lower
  (inside the intervals, the same direction in TPOT). Late admission keeps r6x's prefill rate
  and one-stream speed.
- **One stream's TTFT.** As expected, late admission does not shorten one stream's TTFT
  (124 ms against r6x's 126). The step after a request's last token is decided before the
  finish is known. Synchronous scheduling gets 102 ms there, and pays for it in prefill.
- **Startup memory.** The dgx1 minima before readiness (MemAvailable sampled every 0.25 s,
  read-only) spread 0.47 GiB between r6x and synchronous scheduling. Neither runs any
  late-admission code, and late admission sits between them.
  - The dummy-weight comparison on dgx4 ([runs/smoke-mem.txt](runs/smoke-mem.txt), four
    alternating boots) shows no increase either.
  - The 08:58 refusal at the production pool (12.7 MiB under the guard) is dgx1's boot-to-boot
    spread, which any arm can hit.

## Gates

1. **Screen** (`screen-tp3.json`):
   - **Arms:** late admission at K = 30 first, then r6x, then synchronous scheduling (the A/B's
     `async-off`).
   - **Per arm:**
     - the traced replay (TTFT and its parts at 4 and 8 streams);
     - servebench c4 and c8 (throughput, TPOT);
     - `screen-suite.json`: one stream for 120 s, and cold 32K and 131K prefills, the cases
       where async scheduling could still pay;
     - `token_determinism_text.py --wide` (temperature-0 and greedy outputs identical to
       r6x's).

   Expected:
   - against r6x, the step decided ahead disappears (one lookahead step per request becomes
     none);
   - TTFT −20% or better at four streams;
   - steps per second, one stream and prefill level;
   - against synchronous scheduling, the same TTFT with one stream and prefill no worse.
2. **Release comparison** against r6x on the triangle (release-tp3 + single-stream), only once
   the screen passes, for promotion.
