# Concurrency stalls: where requests wait while the cluster could serve them, and the fixes

Owner, 2026-10-26: "have we appropriately addressed all stalls in the pipeline, even those caused
by multiple concurrent sessions? we should be using the maximum bandwidth to service requests at
all times."

This record answers that in five parts:
1. an inventory of every place a request waits, from the scheduler code (vLLM `next-candidate`)
   and our records;
2. a discrete-event model of vLLM's step building, calibrated on the 1025 release suites;
3. replays of realistic mixes under production scheduling and under each candidate fix;
4. a vLLM patch for the top fix, tested on dgx2 (CPU), and the lab specs that validate it (not run);
5. the ranked plan.

Base: origin/main da7fef4b (spark-ds41f-1025 serving). Laptop only: no cluster, GPU or node time
was used. Times are CET.

## The short answer

**No.** Single steps are near their memory floor (#598, #614), but under concurrency most of a
user's wait is ordering, not bandwidth. The GPU is busy through nearly all of it, on work that
someone else needed less urgently:

- **A short request that arrives while a long prompt prefills waits for that prompt's whole
  prefill.**
  - DS4.1 forces aligned chunks: each long-prompt chunk is exactly the budget less 96 decode
    rows, and a queued prompt that cannot fit whole waits. The waiting pass stops at it, so
    everything queued behind it waits too.
  - The cost: a chat request behind a 131K prompt waits 12.9 s at p50 and 18.9 s at p99 on the
    ring (16.5 s and 22.5 s on the triangle). Behind a 1M prompt it waits 132 s and 202 s.
  - In those mixes that is 87-97% of the short request's time to first token.
- **Slots (`--max-num-seqs` 16 / 8) bind before bandwidth.**
  - From 16 agent lanes, or with heavy chat, 65-98% of time to first token is waiting for a
    slot.
  - Meanwhile the GPU decodes a 16-request batch whose expert reads a larger batch would share.
- **Every chunk step is a 2.4 s decode step for everyone else** (1.45 s on the triangle).
  Decode rows already ride in every chunk step, but they run eagerly at 1.5x a graphed decode
  step's cost on the ring.
- **The held first token (sink rejection) is one more step.**
  - It is 8-32% of a request's time to first token (26% at the release Poisson load, about a
    third in the soak).
  - When that step is a long prompt's chunk, it is 2.4 s.
- **Waits where the GPU itself idles are small:**
  - collective late peers, 1.0-1.7 ms a step;
  - host gaps, 0.7-2.0 ms;
  - late L2 prefetch, 1.7 ms at c1;
  - Engram's first-chunk wait, 24-40 ms per new prompt.

  Together they are about 13% of a one-stream step and about 3% of a 16-stream step. Each has
  its own session.

**The fixes, ranked** by tail latency and goodput over the simulated mixes (geometric means
over sustainable mixes; overload mixes apart; details in section 4):

| # | Fix | Exact how | Short/agent TTFT p50 / p99 | Output | Cost | Needs |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | **Prefill yield** (vLLM patch, `--prefill-yield short`): a long prompt's chunk gives its step to waiting prompts that finish within one chunk, and to a held first token's step; 10 s yield limit | chunk boundaries unchanged; only the step a chunk runs in moves (like late admission) | ring −45% / −53%; triangle −26% / −48%. Behind a long prompt: −72% to −95% (ring), −83% (triangle) | ±0 (+4% in stall mixes) | long prompts +1-11% TTFT; a newly admitted chat then decodes through the long prompt's chunk steps | determinism gate; A/B (prepared) |
| 2 | **More slots**: ring 16 → 24, triangle 8 → 12; budget raised by their rows, so the threshold and every chunk boundary stay | config; boundaries unchanged | at the slot limit: −25% to −83% / −22% to −61%; release Poisson p99 −87%; nothing below the limit | +3-7% at the limit | ITL p99 +16-46% at those loads (bigger batches); the step past 16 requests is unmeasured | the A/B's c24 step |
| 1+2 | both | — | ring −47% / −73%; at overload −74% / −55% | +4% at overload | — | — |
| 3 | **Triangle prefix cache**: retention interval 512 → 2,048, or the freed 1.3 GiB as pool | config | triangle agents p99 −22% | +2% | divergent turns resume up to 2,047 tokens earlier | determinism gate |
| 4 | **Load-dependent chunks**: 4K (or 8K) chunks while others decode; full ones alone | needs the CED anchor (#612) and a whole-model chunking gate | stall step −64% (4K) / −41% (8K); short TTFT p99 −59% | +5% sustainable, −3% overload | long prompts +36% (4K) / +12% (8K); agent mixes −2 to −4% output | #612, a chunking gate, the owner's ITL/TTFT call |
| 5 | **Graphed decode rows in chunk steps** (alpha 1.5 → 1.0) | kernels unchanged; graph capture | agent TTFT p50 −4% to −17% | +1-4% | engineering (breakable graphs) | design |
| 6 | **Ring KV pool 2x** | config | past 24 agent lanes: p99 −17% | +4% at c24 | 10.5 GiB of the ring's headroom | startup-guard boot |
| 7 | **Per-step GPU idle** (late peers, host gaps, prefetch timing; Engram first chunk) | — | small | +1% per 1.5 ms at c8-c16, up to +5% at c1 | — | sessions #614, #615, #616 |

**Not recommended:**
- **vLLM's own prefill lanes (`--max-parallel-prefills 2`).**
  - They livelocked with aligned chunks. Each lane got half the budget, so two long prompts'
    whole chunks never fit; neither lane was released, and nothing prefilled.
  - Simulated, that was 76 of 76 agent, soak and chat runs. On vLLM itself, the pin passed on
    dgx2: two long prompts scheduled nothing.
  - Fixed in vLLM (branch `prefill-lanes-aligned`, f05748ead, local): an aligned chunk is no
    longer capped at its share.
  - Still not for DS4.1. Round-robin of whole chunks keeps several prompts half-prefilled, each
    holding its 3,810 in-flight fill blocks (7.7% of the ring's pool). At overload that starts a
    cache-miss cascade: output −20% / −42% and p99 x3 / x9 at 16 / 24 agent lanes.
- **Shortest-prefill-first** starves long prompts until their cached prefixes are evicted: at
  overload, −28% output and p99 +373%.
- **Prefix-aware eviction** had no measurable effect at these loads. The ring's pool holds the
  live sessions to about 40 lanes (#574); below that, pool size decides.

**The top fix is built:**
- vLLM branch `prefill-yield` (4316d0fc6 on next-candidate e846e64dc), local and not pushed.
- A spark-ds41f-1025 variant on its image's tree, as the overlay the A/B mounts.
- vLLM tests for a test box.
- Lab specs, ready to queue:
  - [stall-probe-tp4.json](stall-probe-tp4.json), production's short-request probe;
  - [ab-tp4.json](ab-tp4.json) and [ab-tp3.json](ab-tp3.json), production against prefill
    yield and slots, plus retention on the triangle;
  - [agentx-tp4.json](agentx-tp4.json), AgentX on production and both fixes, which needs #574.

## 1. Inventory

Sizes are measured (records cited) or simulated (section 2; [runs/ttft-split.txt](runs/ttft-split.txt)
splits production's time to first token by cause in every mix). "GPU" says whether the GPU
idles or serves other work while the request waits.

| # | Stall | Mechanism (next-candidate code) | Measured | Simulated share of the wait | When | GPU |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | **Behind a long prefill** (one long prefill at a time, head-of-line) | `align_prefill_chunks` (forced for DS4.1, models/config.py:582): a chunk ends at the next multiple of the threshold (budget − 96 on the ring), "No partial chunk" (scheduler.py, running pass and waiting pass); the FCFS waiting pass `break`s at the first request that does not fit | 4 x 64K prompts: first tokens at 16/33/49/65 s (2026-09-27); "about 15 s behind a 64K prompt" (2026-10-01, never measured since) | 87% (131K) and 97% (1M) of a probe's TTFT; 52-58% of chat TTFT beside documents; 9-27% in agent and soak mixes | a prompt over one chunk (16,288 tokens ring, 8,144 triangle) prefills | busy (the long prompt) |
| 2 | **Decode stall**: every step carries a chunk | rows ride (the threshold reserves 16 x 6), but the step is the chunk's | 2.36 / 2.61 s p50 / p99 with 15 streams at 16,384 (2026-10-23); 1.444 s at 8,192 on the triangle (2026-10-19); 8.6-14.4x plain decode | model 2.46 s (ring), 1.47 s (triangle); decoding streams get about 5% of their rate | any multi-chunk prefill | busy |
| 3 | **Slots** | `--max-num-seqs` 16 / 8; the waiting pass stops at `num_running >= max_num_running_reqs` | release Poisson TTFT p99 6.3 s (ring) and 10.2 s (triangle), 1025 suites; #574's model: slots bind first from c16 | 65-98% of TTFT at 16+ agent lanes or heavy chat | concurrency above the slots | busy, on a batch smaller than the queue |
| 4 | **Held first token** | sink rejection holds each step's last token as draft zero of the next step | +1 step: 4.2 against 3.2 steps at c1 (#616), +41 ms isolated (1022) | 26% of short TTFT at Poisson load, 29-32% in the soak, 8-12% for agent turns; a full chunk step (2.4 s) when one prefills | every new request | busy |
| 5 | **Eager decode rows in chunk steps** | mixed steps run eagerly; decode rows outside their graphs | fitted alpha: rows cost 1.56x a graphed decode step on the ring, 0.66x on the triangle (prefill rate under 15 / 7 streams; 2026-10-23, 2026-10-19) | +1-4% output if graphed (sensitivity) | every chunk step with decoders | busy, less efficient |
| 6 | **Chunk fixed cost** | each chunk reads every routed expert (the weight stream) | 139-168 ms (TP4), 192-203 ms (TP3) a chunk (#599) | the price of smaller chunks: a 4K chunk pays it 4 times as often | small appends and small chunks | busy |
| 7 | **Prefix-cache misses and eviction** | LRU of freed blocks; checkpoints (7 slabs) join it when made, during prefill, so they leave first; a hit resumes only at a checkpoint | never measured (AgentX #574 has not run) | ring: none at c8, 49 tok/s recomputed at c16 (3% of prefill), 244 at c24; triangle: 151 tok/s at 4 lanes. At interval 512 checkpoints are 3.5x the KV they cover | many sessions, long histories | busy, wasted |
| 8 | **Preemption** | LIFO in the running pass when blocks run out (`--no-scheduler-reserve-full-isl`) | none recorded | none in any mix: the live set fits both pools | KV exhaustion | busy, wasted |
| 9 | **Admission timing** (async scheduling, late admission) | N+1 is decided when N reaches layer 30; a later arrival joins N+2; at c1 the step after a finish is decided before the finish is known | late admission −26/−30% TTFT at 4/8 streams; c1 TTFT 124 ms against 102 synchronous (2026-10-21) | 3-12% of short TTFT ("other") | every arrival | busy |
| 10 | **Engram waits** | reads start only after the GPU hashes the step's IDs | first chunk 24-40 ms (2026-10-16); chunk 2 22-34 ms (2026-10-22); decode layer-1 stall 0.7% c1 / 1% c16 (2026-10-07) | prefetch at admission: TTFT p50 −2%, output +0-1% | each new prompt; every decode step | **idle**, and every decoder in the step waits |
| 11 | **Collective late peers** | ranks spin in all-reduces for the last arrival | 1.0-1.7 ms a step, dgx4 last at 39% (2026-10-03 four-rank traces, #616) | 1.5 ms less a step: +1% output at c8-c16 | every step | **idle** |
| 12 | **Host gaps** | Python between kernels | 0.72 ms c1 greedy (2.1%), 1.98 ms sampled (6.7%), 0.79 ms mixed c8 (#616, rank 0) | as 11 | every step | **idle** |
| 13 | **Late L2 prefetch** | fills that land late or are evicted | 1.67 ms at c1 (#615) | as 11 | every step | partly idle |
| 14 | **Frontend** | the API server renders and tokenizes the whole history each turn | about 20 ms fixed (fitted to c1 TTFT); per-token cost unmeasured. The triangle's traced 120K prefill had TTFT equal to its chunks' sum, so it is small (2026-10-22) | assumed 1 us a token: 0.3 s of a 300K agent turn (unmeasured; AgentX's miss-cost fit measures it) | every turn | idle only if nothing else runs |
| 15 | **vLLM prefill lanes livelock** (latent; fixed in vLLM f05748ead) | `PrefillInterleaveStep` with aligned chunks: half-budget shares never fit a whole chunk, the waiting pass `break`s without releasing the lane, and the second running pass still divides by both lanes | — (production runs one lane) | every agent, soak and chat mix livelocked | only with `--max-parallel-prefills` > 1 | stuck |

What this means for "maximum bandwidth at all times":
- **Idle GPU** (items 10-13) is about 4 ms of a 30 ms one-stream step and about 3.5 ms of a 120 ms
  sixteen-stream step. Each item has an owner already (#614, #615, #616, the Engram records).
- **Wasted work** (items 7-8) is small on the ring below about 20 agent lanes. It is real on the
  triangle (interval 512) and on the ring past 24 lanes.
- **Less efficient work** (items 3, 5, 6) is the slots, eager decode rows and per-chunk fixed cost:
  a few percent of throughput.
- **The rest** (items 1, 2, 4, 9) is order: who gets the step. It costs no bandwidth and most of the
  latency users see under concurrency.

**Decode rows always ride in prefill steps already.** `_check_aligned_prefill_chunks` refuses
a threshold that leaves no room for every sequence's six rows, and the running pass schedules
every decoding request beside the chunk. That fix is done. What remains is the length of the step
those rows ride in (item 2) and that they run eagerly (item 5).

## 2. The simulator

[stallsim.py](stallsim.py) is a discrete-event model of `Scheduler.schedule()` as production
configures it:
- **Passes and budget.** The running pass runs in admission order, then the waiting pass in FCFS
  order, both against the token budget. Each decoding request schedules 1 + 5 draft rows.
- **Aligned chunks.** A chunk is aligned with no partial chunk. The waiting pass breaks at the
  first request that does not fit.
- **Admission.** Slots, and KV admission for the first chunk only. LIFO preemption happens in the
  running pass, and no admission happens in a step that preempted.
- **Async scheduling with late admission.** Step N+1 is decided at 70% of step N. A request that
  finishes in N still has a row in N+1. vLLM's check that skips a request whose in-flight step
  samples its last token is modelled.
- **Sink rejection.** The first token is emitted at the end of the first decode step.
- **The KV pool.**
  - 256-token blocks, and a request's 46 window and ring blocks (#586).
  - The blocks a prompt in flight adds (3,810 on the ring, 1,905 on the triangle).
  - LRU of freed blocks, freed tail first, with retention checkpoints of 7 slabs. They are made
    at every multiple of the interval and at the prompt end, and join the LRU when made.
  - A finished request's partial tail block stays cacheable (vLLM hashes these groups in 64-token
    units).
  - A hit resumes at the deepest cached checkpoint whose blocks are cached, at most the prompt
    less 128 replay rows.
  - Content identity is a tree of lineages (branches share a prefix), so appends, divergent turns
    and subagents share exactly what they would share.

[tests/test_stallsim.py](../../tests/test_stallsim.py) pins these semantics (17 tests in the
unit suite, two of them that the arms follow the recipe):
- vLLM's `_aligned_prefill_chunk`;
- the short request that waits for the long prompt's last chunk;
- decode rows in every chunk step and the budget bound;
- late admission and the held first token;
- prompt-end hits, replay rows, divergent branches and LRU against prefix eviction;
- LIFO preemption;
- the lane livelock, and vLLM's fix reproducing vLLM's own test sequence;
- that the reordering policies never move a long prompt's chunk boundaries.

The real scheduler confirms the two behaviours the conclusions rest on: the wait behind a long
prompt's aligned chunks, and the lane livelock. Both passed as tests on next-candidate on dgx2
(section 6, [runs/dgx2-tests/summary.txt](runs/dgx2-tests/summary.txt)). The fix's own sequence
test passed there as well.

### Step costs and calibration ([calibrate.py](calibrate.py), [runs/calibration.txt](runs/calibration.txt))

A step costs:
- a plain decode step at n decoding requests (piecewise linear over three fitted knots);
- for prefill rows, the cheaper of:
  - a short prompt's eager extra plus a per-token cost (the 2026-10-20 TTFT trace: +20 ms +
    1.58 ms a token on the triangle);
  - alpha x decode + F + m0 x tokens;
- plus m1 x tokens x context;
- plus Engram's first-chunk wait.

Fitted, each to its own record:

| Parameter | Ring (TP4) | Triangle (TP3) | Source |
| --- | ---: | ---: | --- |
| F, a chunk's fixed cost | 153.5 ms | 197.5 ms | #599 (midpoints) |
| m0, per prompt token | 125.3 us | 139.8 us | cold TTFT 32K-1M, 1025 release suites |
| m1, per token per context token | 132.9 ps | 169.0 ps | same |
| plain decode step at 1 / 8 / 16 requests (TP3 1 / 4 / 8) | 29.8 / 83.8 / 119.1 ms | 36.9 / 72.1 / 105.1 ms | closed loops' steps a second |
| alpha, decode rows in a chunk step | 1.56 | 0.66 | 131K prefill rate beside 15 / 7 streams |
| frontend, fixed | 22 ms | 8 ms | one stream's TTFT |
| tokens a decoding request gains a step | 2.28 / 2.09 / 1.93 | 2.27 / 2.17 / 2.08 | release suites' tokens a step |

Checked, not fitted:

| Check | Measured | Model |
| --- | ---: | ---: |
| cold TTFT, 32K-1M (ring / triangle) | — | within 2.7% / 1.7% at every size |
| ring c8: tok/s, TTFT p50, TPOT p50 | 194.9, 297 ms, 40.4 ms | 192.1 (−1.4%), 283 ms (−4.7%), 40.5 ms |
| ring c16: tok/s, TTFT p50 / p99, TPOT p50 | 251.4, 407 / 488 ms, 63.1 ms | 248.0 (−1.3%), 392 / 478 ms, 62.8 ms |
| triangle c4 / c8 tok/s | 118.2 / 154.7 | 116.4 / 152.6 |
| ring stall step p50 / p99 (15 streams, 131K, 16,384) | 2.36 / 2.61 s | 2.46 / 2.64 s |
| triangle stall step p50 (7 streams, 131K, 8,192; r6x + skip, scaled) | 1.42 s | 1.47 s |
| release Poisson TTFT p50 (ring / triangle) | 280 / 315 ms | 271 / 246 ms |
| release Poisson TTFT p99 (ring / triangle) | 6.3 / 10.2 s (one 300 s window each) | 0.4-5.0 / 1.7-10.4 s over five seeds |

The Poisson tails are queueing at the slot limit and depend on each window's arrivals. The
measured windows fall inside the model's seed range, which is why the evaluation pools 4-8 seeds
and reports paired bootstrap intervals.

### Mixes ([workloads.py](workloads.py), [evaluate.py](evaluate.py))

- **poisson**: the release suite's Poisson step: 0.7 req/s on the ring, 0.42 on the triangle;
  20-token prompts, 256 tokens out.
- **stall** and **stall-1m**: servebench's stall step with probes.
  - A cold 131K (1M) document arrives beside decoding streams: 11 on the ring, 4 on the triangle,
    so the probes find slots.
  - Short probes (500-token prompts, 16 tokens out) arrive 2-14 s (10-160 s) into its prefill.
- **chat-docs**: interactive chat (0.25 req/s on the ring, 0.12 on the triangle).
  - Prompts are mostly short, a quarter 0.5-30K; outputs have a median of 350 tokens.
  - Cold documents of 32K-524K (triangle 32K-262K) arrive every 150 s (240 s).
- **agents-cN**: AgentX-like sessions. They are a model of #574's corpus, which is not on the
  laptop; [agent_stats.py](agent_stats.py), [runs/agent-stats.txt](runs/agent-stats.txt):
  - main agent, subagents (24% return to older context) and side calls;
  - 89.6% appends, 6.6% divergent, 3.9% side contexts;
  - lanes start 25-75% into a session with a primer;
  - median input 106-217K (corpus about 142K);
  - about half of main turns over 256K (corpus about half);
  - unbounded-cache hit 0.975-0.990 (corpus 0.983);
  - at 8 lanes 1,713 uncached tok/s offered (#574's model: about 1,700).
- **agents+chat**: 8 lanes beside 0.12 req/s chat (2 lanes beside 0.06 on the triangle); "heavy":
  0.3 req/s.
- **soak**: the 2026-10-23 soak. Agent sessions resend history growing by 16-48K tool results up
  to 960K, beside 8 (4) chat streams.

Each mix runs 4-8 seeds; windows are 10-60 minutes after warm-up. The matrix (2,568 runs: 107
mix-seeds x 24 policies) takes about 6 minutes on the laptop.

## 3. Results

Production first, then each fix's change. Full tables: [runs/results.txt](runs/results.txt)
(every policy) and [runs/ranking.txt](runs/ranking.txt) (geometric means and every mix's change
with its paired 95% bootstrap interval over seeds). "Users wait on" is the class that feels the
mix's stalls: probes, chat, or main-agent turns.

#### tp4-ring: production

| Mix | Users wait on | TTFT p50 | TTFT p99 | Their ITL p99 | Long prompt TTFT p50 | Output tok/s | Recomputed tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| poisson | chat | 0.27 s | 3.54 s | 0.14 s | - | 176.7 | 0 |
| stall | probe | 12.91 s | 18.91 s | 2.42 s (stall step) | 20.8 s | 135.2 | 0 |
| stall-1m | probe | 132.14 s | 202.14 s | 3.19 s (stall step) | 212.0 s | 32.3 | 0 |
| chat-docs | chat | 0.44 s | 84.18 s | 1.12 s | 40.3 s | 118.8 | 0 |
| agents-c8 | agent | 1.13 s | 8.88 s | 0.96 s | - | 138.7 | 0 |
| agents-c16 | agent | 2.54 s | 30.21 s | 1.54 s | - | 159.8 | 49 |
| agents-c24 | agent | 20.83 s | 63.23 s | 1.77 s | - | 152.0 | 244 |
| agents-c8+chat | chat | 0.80 s | 23.25 s | 1.21 s | - | 168.3 | 1 |
| agents-c8+chat-heavy | chat | 29.99 s | 79.06 s | 0.93 s | - | 204.1 | 0 |
| soak | chat | 0.34 s | 14.46 s | 3.95 s | 13.9 s | 68.1 | 0 |

#### tp4-ring: change against production (user TTFT p50 / p99, output, long-prompt TTFT)

| Mix | prefill yield | slots 24 | yield + slots | chunks 8K beside decodes | chunks 4K beside decodes | graphed decode rows* | held first token off* | pool 2x |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| poisson | +0% / +0% | +0% / −87% | +0% / −87% | +0% / +0% | +0% / +0% | +0% / +0% | −30% / −3% | +0% / +0% |
| stall | −72% / −69%, out +4%, long +11% | +1% / +0% | −72% / −69%, out +4%, long +11% | −71% / −61%, out +3%, long +12% | −81% / −82%, out +6%, long +35% | −4% / −3%, long −2% | −1% / −1% | 0 |
| stall-1m | −95% / −95%, out +4%, long +1% | 0 | −95% / −95%, out +4%, long +1% | −96% / −97%, out +17%, long +9% | −98% / −98%, out +44%, long +26% | −3% / −2%, long −2% | 0 | 0 |
| chat-docs | +1% / −13%, long +1% | −3% / −13%, long −11% | −3% / −50%, long −6% | +9% / +3%, long +13% | +33% / +22%, long +35% | −5% / −3% | −18% / −0% | 0 |
| agents-c8 | −0% / −32% | −0% / +7% | −1% / −31% | −2% / +11% | −0% / −22%, out −2% | −6% / +11%, out +2% | −11% / −14% | 0 |
| agents-c16 | −6% / −8% | −47% / −61%, out +3% | −48% / −78%, out +2% | −7% / −3% | −1% / +2%, out −4% | −17% / −9%, out +4% | −12% / −0% | −5% / −10%, out +1% |
| agents-c24 | +5% / −1% | −83% / −22%, out +5% | −83% / −35%, out +4% | +5% / +1%, out +1% | +8% / −3%, out −3% | −1% / +3%, out +3% | −1% / −1% | +0% / −17%, out +4% |
| agents-c8+chat | −2% / −3% | −25% / −52%, out +1% | −21% / −75%, out +1% | +3% / −7% | +10% / +6%, out −2% | −9% / −14%, out +2% | −21% / −0% | 0 |
| agents-c8+chat-heavy | +2% / +6% | −77% / −49%, out +7% | −82% / −36%, out +7% | +1% / −2% | +6% / +10%, out −2% | −6% / +12%, out +1% | −0% / −0% | 0 |
| soak | +0% / −34%, long −8% | +0% / −1% | +0% / −34%, long −10% | +0% / −36%, out −1%, long +17% | +0% / −40%, out −2%, long +46% | +0% / −3%, out +1% | −30% / −22%, long −13% | 0 |

#### tp3-triangle: production

| Mix | Users wait on | TTFT p50 | TTFT p99 | Their ITL p99 | Long prompt TTFT p50 | Output tok/s | Recomputed tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| poisson | chat | 0.26 s | 7.97 s | 0.14 s | - | 105.2 | 0 |
| stall | probe | 16.46 s | 22.46 s | 1.45 s (stall step) | 24.4 s | 71.2 | 0 |
| chat-docs | chat | 0.32 s | 31.65 s | 0.14 s | 17.6 s | 54.9 | 0 |
| agents-c4 | agent | 1.00 s | 10.42 s | 0.64 s | - | 87.4 | 151 |
| agents-c2+chat | chat | 0.38 s | 8.51 s | 0.45 s | - | 82.6 | 3 |
| soak | chat | 0.30 s | 7.79 s | 2.48 s | 12.0 s | 52.8 | 0 |

#### tp3-triangle: change against production

| Mix | prefill yield | slots 12 | yield + slots | chunks 4K beside decodes | retention 2,048 | pool +1.3 GiB | held first token off* |
| --- | --- | --- | --- | --- | --- | --- | --- |
| poisson | 0 | −1% / −84% | −1% / −84% | 0 | 0 | 0 | −31% / −1% |
| stall | −83% / −82%, out +4%, long +8% | +1% / +1% | −83% / −83%, out +4%, long +8% | −85% / −80%, out +5%, long +17% | 0 | 0 | −1% / −0% |
| chat-docs | −0% / −54%, long +5% | −0% / −16% | −0% / −87% | +1% / −12%, long +7% | 0 | 0 | −20% / −1% |
| agents-c4 | −0% / −45% | −1% / −0% | +0% / −42% | −2% / −27%, out −1% | +2% / −22%, out +2% | +0% / −22%, out +2% | −9% / −9% |
| agents-c2+chat | −0% / −47% | +1% / −11% | +1% / −56%, out −1% | −0% / −26%, out −1% | +1% / −4% | +1% / −2% | −20% / −6% |
| soak | +3% / −19% | +2% / −5% | +3% / −18% | +3% / −40%, out −2%, long +8% | 0 | 0 | −30% / −31%, long −9% |

`*` marks a sensitivity, not a scheduling fix: graphed decode rows need graph work, and sink
rejection stays on (owner, 2026-10-22).

How to read them:
- **Prefill yield** removes the wait behind a long prefill wherever one occurs:
  - stall mixes, soak, chat beside documents on the triangle, agent tails;
  - throughput is level.
- **Its limit is the decode stall.** A chat admitted during a long prefill then decodes through
  that prefill's chunk steps.
  - On the ring beside documents that keeps slots full (chat-docs: −13% at p99 alone, −50% with
    more slots), and their ITL p99 rises (chat-docs 1.1 → 2.2 s).
  - Before, those chats waited out the whole prefill instead.
- **Slots** do nothing below the limit and dominate at it.
  - The gains assume the decode step keeps the slope it has from 8 to 16 requests.
  - With twice that slope (`slots-24 (steeper)*` in the tables), throughput falls 3% at overload
    but TTFT still falls 55% at p50. The A/B's c24 step measures the real slope.
- **Smaller chunks beside decodes** cut the stall step, and with it the decode slowdown, by
  41-64%.
  - They also let short prompts fit beside the long chunk.
  - But every extra chunk pays the fixed weight-stream cost (153 ms) and another eager pass of
    the decode rows.
  - So long prompts finish 12-46% later and agent mixes lose 2-4% of output.
  - "Full chunks when nobody decodes" (`alone-32k+4k` in the tables) adds nothing under
    concurrency: such steps are rare. It helps only a lone long prompt (+2% per budget step past
    16K, 2026-10-23).
- **Cache levers** matter only where the pool is under pressure: the triangle at interval 512,
  and the ring past 24 lanes.

## 4. The fixes

All keep temperature-0 outputs bit-identical across scheduling: batch invariance already covers
which requests share a step (late admission and interleaving passed the determinism gate on that
argument). None is a cadence or quota throttle; each prefill pauses only for work that needs the
step.

### 4.1 Prefill yield (top; built)

**Rule.** A step yields when either of these holds:
- a held first token is due: a request whose last prefill chunk ran and whose first decode step
  has not;
- a waiting request can be admitted and its remaining prefill, after its prefix-cache hit,
  finishes within one aligned chunk.

**In such a step:**
- long prefill chunks, running or queued, wait;
- the waiting pass skips long requests instead of stopping at them. FCFS seniority is kept,
  because skipped requests go to `skipped_waiting`, which is served first next step;
- if nothing short is admitted after all (no blocks, say), the deferred long chunks run in a
  leftover pass;
- a long prompt that has yielded for `--prefill-yield-limit-s` (10 s) stops yielding until its
  next chunk runs.

**What it keeps.** A step with nothing short waiting and no first token due schedules exactly as
today. The long prompt's chunks keep their absolute boundaries. Only the step a chunk runs in
changes.

**The owner's rule.** The prefill pauses only for work that needs the step: a prompt that
completes, or the step that releases a held token. There is no cadence or quota. The limit
bounds yielding; it never forces a pause.

**Exactness.** Boundaries are unchanged, so every row computes what it computes today (batch
invariance). Gate: `token_determinism_text.py --wide` (long and mixed scenarios) in the A/B.

**Cost.**
- One prefix-cache lookup per waiting request per step (at most 16), only while requests wait
  with a free slot.
- Long prompts finish +1-11% later.
- Short requests admitted mid-prefill decode through its chunk steps.

**Code.**
- vLLM branch `prefill-yield`: 4316d0fc6 on next-candidate e846e64dc, local and not pushed
  (worktree ~/projects/vllm-prefill-yield).
  - It changes `vllm/config/scheduler.py` (`prefill_yield`, `prefill_yield_limit_s`; refuses
    lanes, compute sharing and the interval beside it), `vllm/engine/arg_utils.py` and
    `vllm/v1/core/sched/scheduler.py` (+129 lines).
  - It adds `tests/v1/core/test_prefill_yield.py`, eight CPU tests:
    - off is unchanged;
    - a short prompt takes the step;
    - boundaries unchanged;
    - a long head of queue does not block;
    - shorts keep their order;
    - nothing short changes nothing;
    - the held-first-token step;
    - the limit and the option checks.
  - Patch: [vllm/next-candidate/](vllm/next-candidate/).
- The spark-ds41f-1025 variant: branch `prefill-yield-1025` (d29e028de on e0ccc837d, the 1025
  image's tree; alignment there is `ALIGN_PREFILL_CHUNKS`). Patches are in
  [vllm/spark-ds41f-1025/](vllm/spark-ds41f-1025/); the three changed files are in
  [overlay/](overlay/), mounted by the A/B arms.
- **Tested on dgx2** (`kernel-local`, CPU, in the 1025 image;
  [runs/dgx2-tests/summary.txt](runs/dgx2-tests/summary.txt)):
  - the eight tests pass on both trees;
  - the regression set (vLLM's aligned-chunk, scheduler and compute-share scheduler tests) gives
    233 passed and the same 49 offline failures with and without the change, on both trees.
  - The overlay's boot is the A/B's first gate.

### 4.2 More slots

**Change.**
- Ring `--max-num-seqs` 24 with budget 16,432; triangle 12 with budget 8,216.
- The threshold stays 16,288 / 8,144, so chunk boundaries do not move.
- Graphs are captured to 144 / 72 rows ([slots-24-tp4.json](slots-24-tp4.json),
  [slots-12-tp3.json](slots-12-tp3.json)).

**Memory.**
- The KV pool is unchanged; the extra cost is graphs and per-sequence state.
- The ring has 13.7 GiB over its startup guard (2026-10-23). The triangle is tight: dgx1 was
  2.99 GiB over its steady guard in the 1025 suite. Its boot is the gate.

**Unknown.** The decode step past 16 requests: the grid has no shape above 16 x 96. The A/B's
`c24` step measures it.

### 4.3 Triangle prefix cache

**Problem.** At interval 512 each 512 tokens of prefill leave a 7-slab checkpoint, 3.5x the KV
they cover. They join the LRU when made and crowd out prefixes.

**Effect.**
- Interval 2,048 ([retention-2048-tp3.json](retention-2048-tp3.json)) cuts recompute at 4 lanes
  from 151 to about 0 tok/s: p99 −22%, +2% output.
- Spending the 1.3 GiB freed by 0505-0507 on the pool does about the same.
- Interval 8,192 is no better than 2,048 in these mixes.

**Cost.** A divergent turn resumes up to 2,047 tokens earlier; #574 put that at 0.1 point of hit
rate at 8,192.

### 4.4 Load-dependent chunks (designed; needs a gate)

**Design.**
- Thresholds by decoding requests: full chunks with none, 4K (or 8K) with any.
- Chunk ends on an absolute 2,048-token grid.
- The budget stays the recipe's: alone steps are rare under concurrency, so a larger alone budget
  buys nothing here.

**Gates.**
- The CED prompt-end anchor (#612) makes CED independent of chunk ends.
- The whole model's independence of chunk ends is untested: 2026-10-19's `chunking.json` never
  booted.

**Before an A/B.**
- #612 in the candidate;
- a whole-model chunking gate (prompts split at 2K, 4K, 8K and 16K multiples, identical
  digests);
- the owner's choice of 8K (mild: stall step −41%, long +12%) or 4K (stall step −64%, long
  +36%).

### 4.5 The rest

- **Graphed decode rows in chunk steps.** Capture the decode rows of a mixed step in their
  graphs: breakable graphs, or a decode-rows sub-graph beside the eager chunk.
  - Today they cost 1.56x a graphed decode step on the ring.
  - Bound: +1-4% output, −4% to −17% agent TTFT p50.
- **Ring pool 2x** past 24 lanes: AgentX's c24 point decides (#574 already plans the 2x arm).
- **Engram prefetch at admission.** Hash a prompt's first chunk on the host and start its reads
  when it is admitted, or while it waits. Small: TTFT p50 −2%.
- **Per-step idle**: #614 (IKET), #615 (prefetch schedule), #616 (late peers, host gaps).
- **Rejected:**
  - vLLM's prefill lanes. The livelock is fixed (section 5), but round-robin whole chunks start
    a cache-miss cascade at overload: −20% to −42% output;
  - shortest-prefill-first (starvation, then the same cascade);
  - prefix-aware eviction (no effect at these loads).

## 5. Findings for other sessions

- **`--max-parallel-prefills` > 1 livelocks with aligned chunks** (vLLM `PrefillInterleaveStep`).
  - Each selected lane's budget is the step's remaining budget over the lanes not yet served.
  - A long prompt's whole aligned chunk exceeds half the budget. The running pass skips it
    without releasing the lane; the waiting pass `break`s on it ("No partial chunk this step").
  - The second running pass still divides by both lanes.
  - So two long prompts never prefill. The 2026-10-01 lanes experiment (never run) would have hit
    it.
  - Shown on vLLM itself on dgx2 (next-candidate).
  - **Fixed:** vLLM branch `prefill-lanes-aligned`, f05748ead on next-candidate, local; patch in
    [vllm/next-candidate/](vllm/next-candidate/).
    - With aligned chunks a lane's share no longer caps its chunk. It takes the step's budget
      when that holds it, and lanes rotate whole chunks; unaligned chunks keep their shares.
    - `tests/v1/core/test_aligned_prefill_lanes.py`: two long prompts prefill; three prompts
      rotate exactly as the simulator predicts; unaligned shares unchanged. 3 passed on dgx2.
    - The pin that asserted the livelock is gone with it.
  - Lanes remain a bad choice for DS4.1 at overload (section 4.5).
- **The stall probe needs free slots and realistic prompts.**
  - With 15 streams and the document, all 16 ring slots are taken, so probes measure slot
    queueing.
  - Production also lets a prompt shorter than the unused decode rows ride beside a long chunk:
    (16 − decoding streams) x 6 tokens. So servebench's 20-token default prompts sneak in beside
    a long chunk while fewer than about 12 streams decode.
  - The probe step therefore uses 11 streams (4 on the triangle) and 500-token prompts with 16
    tokens out.
- **Sink rejection's held token costs a whole step under concurrency.** That is 2.4 s when the
  next step carries a long chunk. Prefill yield's first-token rule keeps that step short.
- **The triangle's retention interval 512** makes checkpoints 3.5x the KV they cover.

## 6. Cluster validation (prepared, not run)

Specs are checked with the lab's own planner (`plan`, `plan_problems`): no problems. The AgentX
spec uses #574's job kind and is checked once #574 is on main.

| Spec | Arms | What it measures | Time | Decision |
| --- | --- | --- | --- | --- |
| [stall-probe-tp4.json](stall-probe-tp4.json) | production | [stall-suite-tp4.json](stall-suite-tp4.json): probes 2-14 s into a 131K prefill and 10-160 s into a 1M prefill (11 streams), the release stall step (15 streams), c16 and c24 closed loops, release Poisson, cold 32K/131K; the determinism scenarios | about 45 min | the model's baseline: probe TTFT p50 about 13 s (131K) and 130 s (1M); stall step about 2.4 s; c24 queues |
| [ab-tp4.json](ab-tp4.json) | production, prefill yield, slots 24, production again (bracket) | the same suite on each arm, plus `token_determinism_text.py --wide` | about 4 h | prefill yield: identical digests; probe TTFT p50 −60% or better (model −72% / −95%); long prompts no worse than +15%; c16 and Poisson level. Slots: boots past the guards; c24 step under 1.5x c16's; c24 TTFT p50 down |
| [ab-tp3.json](ab-tp3.json) | production, prefill yield, retention 2,048, slots 12 (bracket) | [stall-suite-tp3.json](stall-suite-tp3.json) | about 3.5 h | as above; slots 12 must boot past dgx1's startup guard |
| [agentx-tp4.json](agentx-tp4.json) (needs #574) | production c8/c16/c24, prefill yield c8/c16, slots 24 c16/c24 | AgentX: TTFT, hit rate, miss cost, KV use, preemptions | about 8.5 h | the agent tail (model: −32% p99 at c8 with yield; −47% p50 at c16 with slots) and the frontend's per-token cost (the miss-cost fit) |

**Scheduler tests on dgx2** (2026-10-26 20:42-20:44 CET).
- How they ran: `scripts/lab.py kernel-local`, CPU tests in the 1025 image's container.
- What they ran: [make_bundles.py](make_bundles.py) mounts each vLLM tree's changed package files
  over the image. `facebook/opt-125m` (config and tokenizer only) is a synced input.
- Outputs: [runs/dgx2-tests/](runs/dgx2-tests/).

| Tree | New tests | Regression set |
| --- | --- | --- |
| the image (1025, e0ccc837d) | — | 233 passed, 49 failed |
| prefill yield on 1025 (d29e028de) | 8 passed | 233 passed, the same 49 |
| next-candidate (e846e64dc) | 3 passed: the wait behind a long prompt; the lanes livelock; one lane in turn | 233 passed, the same 49 |
| next-candidate + prefill yield + lanes fix (`concurrency-stalls-next`, 6f4adad09) | 11 passed | 233 passed, the same 49 |

The 49 failures are the same set in every tree ([runs/dgx2-tests/regression-failures.txt](runs/dgx2-tests/regression-failures.txt)).
They are offline cache misses: the encoder-cache connector tests need llava-1.5-7b, and the
half-life CLI test needs Qwen3-0.6B, neither cached on dgx2.

**Before queueing.**
- **Prefill yield:** done. Its tests and the regression set ran on dgx2 (section 4.1), and #620
  put the overlay on main.
- **Slots:** nothing more.
- **AgentX:** #574 rebased (its arm predates 1025).

The ring is the production shape; while a cable is missing, the triangle specs apply.

## 7. Caveats

- **The agent mixes are a model of AgentX.** The lengths, branch and append shares, the
  unbounded-cache hit rate and the offered prefill match #574's published aggregates. The corpus's
  timing and fan-out are guesses. AgentX replaces them.
- **Unmeasured costs are assumptions, with sensitivities run:**
  - the decode step past 16 requests;
  - the frontend's per-token cost (1 us);
  - eager decode rows in short-prompt steps (alpha 1 there).
- **Percentiles at p99 rest on 10-30 requests per mix** after pooling seeds. The ranking uses
  geometric means over mixes, and [runs/ranking.txt](runs/ranking.txt) gives each change's
  paired bootstrap interval.
- **The model has no cross-request interference** inside a step beyond its cost formula, and no
  memory-guard events.
- **The calibration is spark-ds41f-1025's.** The ring's stall-step check uses candidate 1 at
  16,384 (2026-10-23), whose prefill was within 1% of 1025's.

## Files

| File | What |
| --- | --- |
| [stallsim.py](stallsim.py) | the simulator: scheduler, cost model, KV and prefix cache, policies |
| [workloads.py](workloads.py) | chat, documents, the stall probe, AgentX-like sessions, the soak |
| [calibrate.py](calibrate.py), [calibration.json](calibration.json) | the fit and the checks; [runs/calibration.txt](runs/calibration.txt) |
| [evaluate.py](evaluate.py), [rank.py](rank.py), [report_tables.py](report_tables.py) | the matrix (2,568 runs), the ranking, the README tables; [runs/results.json](runs/results.json), [runs/results.txt](runs/results.txt), [runs/ranking.txt](runs/ranking.txt), [runs/readme-tables.md](runs/readme-tables.md) |
| [ttft_split.py](ttft_split.py), [agent_stats.py](agent_stats.py) | production's TTFT by cause; the agent generator against #574; [runs/ttft-split.txt](runs/ttft-split.txt), [runs/agent-stats.txt](runs/agent-stats.txt) |
| [make_arms.py](make_arms.py) and the arms | production, prefill yield, slots, retention (ring and triangle), AgentX ports |
| [stall-suite-tp4.json](stall-suite-tp4.json), [stall-suite-tp3.json](stall-suite-tp3.json), the specs | section 6 |
| [overlay/](overlay/), [vllm/](vllm/) | prefill yield for the 1025 image; the patches |
| [make_bundles.py](make_bundles.py), [runs/dgx2-tests/](runs/dgx2-tests/) | the scheduler test bundles (kernel-local, CPU, the 1025 image) and their outputs |

Also changed: `servebench stall` gains `--probes`, `--probe-tokens` and `--probe-max-tokens`
(short cold requests during the prefill, their TTFT and its verdict rows; off by default, so
existing suites are unchanged), with a test in tests/test_servebench.py.

Reproduce (laptop, standard library):

```sh
python3 -I experiments/2026-10-26-concurrency-stalls/calibrate.py
```

```sh
python3 -I experiments/2026-10-26-concurrency-stalls/evaluate.py --workers 10
```

```sh
python3 -I experiments/2026-10-26-concurrency-stalls/rank.py
```
