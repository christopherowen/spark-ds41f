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

The screen (`screen-tp3.json`, K = 30) is staged on dgx1-3 for the lab queue.

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
