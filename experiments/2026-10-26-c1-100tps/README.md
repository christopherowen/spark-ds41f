# One stream at 100 tokens a second: what it would take (TP4, c1)

Owner, 2026-10-25: does next week's work get single-stream decode to 100 tok/s? It does not.
The fusion budget tops out near 86 tok/s. This record bounds the levers that could close the
rest, from traces and data we already have:

- tokens a step;
- GEMM bandwidth efficiency;
- communication.

Rules: native weight precision, correctness and determinism first, no fallback paths, and
bound before building. No serving or cluster time was used. Two measurements ran on dgx4,
the node open to single-node testing: the expert-rate kernel bundle and the CPU weight
statistics (both under [Runs](#runs-on-dgx4-2026-10-25-cet)). The acceptance measurement
needs the TP4 cluster and stays a prepared spec.

Inputs:
- rank 0's torch-profiled traces of
  [request-inventory-5](../2026-10-25-request-inventory/README.md)
  (`results/private/determinism/request-inventory-5/ports`). These are the traces the
  fusion survey (draft #583) used: c1 greedy (11 steps of 34.3 ms), c1 sampled (23 steps)
  and mixed c8.
- the release suite's single stream, spark-ds41f-1022
  (`manifests/suites/2026-10-22-spark-ds41f-1022-tp4/c1.json`): 32.73 steps a second,
  2.271 tokens a step, 3.073 verified drafts a step, 74.3 tok/s.
  spark-ds41f-1025 (promoted 2026-10-25 18:27 CET) runs the same kernels at about
  32.4 x 2.28.
- DeepSeek's reference forward on CPU: the routes of all 40 layers for eight consecutive
  greedy tokens ([json-count](../2026-10-18-json-count/README.md), `runs/ref-woa1.json`).
- the expert-rate bundle's FC1 and FC2 timings at known expert counts, and the
  checkpoint's weight statistics (runs/, below).
- the recipe at `config/cluster.json` and the vLLM tree 26382f5cf, read only.

## The short answer

- **The byte floor is 70% of today's step.** GB10 streams expert weights at 266.6 GB/s at
  the margin (the expert-rate bundle; the specification is 273). At that rate a c1
  step's DRAM reads (5.71 GB a rank) take 21.4 ms against today's 30.55 ms.
- **100 tok/s at today's tokens a step is just inside the floor, and not credible.**
  Removing everything that is not a byte would give 103.9 tok/s at 2.27 tokens a step.
  Reaching 100 would take 83% of the 8.7 ms non-byte time.
  - DRAM idles for 3.0 ms a step: small kernels, launch gaps, collectives and attention.
    That is +10.9%.
  - Weights stream below the attainable rate for 5.7 ms. That is +23%. The parts:
    - the routed experts' launch fixed costs: 1.24 ms;
    - the L2 prefetch and its consumers: most of the rest;
    - FC2: 0.27 ms;
    - the LM heads: 0.20 ms;
    - small streams: 0.55 ms.
  - The all-reduces run 4.7 ms of kernels a step, but DRAM idles under them for only
    0.95 ms. The L2 prefetch streams through the waits, so communication is worth at most
    +3.2% at c1, not 12%.
- **A verified row is expensive at c1.** Each live row reaches 3.5-3.8 new experts a
  layer: about 2.9 ms with its dense share, 9.6% of the step. That price bounds every
  acceptance lever:
  - **trees:** +1.3% if the second candidate were always right, −3.7% at a realistic 50%;
  - **two more drafts:** about +2%;
  - **an oracle verification policy:** +12-19%, but it is not realizable;
  - **an oracle drafter:** 6 tokens a step at 37 ms, about 161 tok/s. It is the one lever
    that clears 100 with room, and it is a model change.
- **The credible set reaches 89-91 tok/s.** Half the overhead removed and Q-A/KV
  column-sharded (exact) gives 88.8 tok/s; with 3% more tokens a step, 91.5. Two thirds of
  the overhead gives 94.1.
  - 100 needs 2.74 tokens a step with half the overhead gone (+21%), or 3.31 at today's
    overhead.
  - The levers that would give it are a stronger drafter, suffix drafting on copy-heavy
    agent text (about +6-15% tok/s there, nothing on the suite), or a stretch set that
    needs an owner decision and a research item (110 tok/s, below).
- **Measured on dgx4:**
  - FC1 costs about 14 µs a launch plus 11.4 µs an expert at six rows: 208 GB/s at 6
    experts, 252 at 18, 256 at 36. The traced layers read 19.9 experts each, and FC1 ran
    at the bench's own speed.
  - mHC `fn` is genuinely FP32, so storing it as BF16 is not lossless.
  - The FP8 codes carry 6.58 of 8 bits and the FP4 nibbles 3.90 of 4.
- **Still to measure:** [c1-acceptance.json](c1-acceptance.json), the drafter's unbiased
  survival by position and the price of six live rows, on the suite's own temperature-0
  text. It needs the TP4 cluster.

## How the floor was computed

[floor.py](floor.py) reads the traces and writes [floor.json](floor.json):

```sh
python3 -I experiments/2026-10-26-c1-100tps/floor.py results/private/determinism/request-inventory-5/ports \
  --expert-rate experiments/2026-10-26-c1-100tps/runs/expert-rate-dgx4-2.txt
```

- **Bytes.** Every weight byte a TP4 rank reads in one step, from the checkpoint's shapes:
  - FP8 block-32 projections with their scales, BF16 router and compressors, FP32 `fn`;
  - the packed 12-bit heads;
  - the routed experts as E2M1 plus coded scales: 3.04 MB for FC1 and 1.53 MB for FC2 per
    expert at TP4.
- **The attainable rate is FC1's marginal rate at six rows.** That is its bytes an expert
  over the fitted time an expert: 266.6 GB/s. Cross-checks:
  - FC2: 265 GB/s;
  - FC1 at three rows: 276 GB/s;
  - a plain `torch.sum` read: 248-254 GB/s at 55-220 MiB;
  - the packed LM head: 240.7 GB/s in the traces.

  A first estimate used the head's rate as the attainable one. Its figures remain in
  floor.json (`by_eta`) and model.json (`first estimate`); the bundle superseded them.
- **Distinct experts.** A grouped GEMM's grid is the bucket's maximum routes (`(9, 36)` at
  six rows), and unused tiles exit early, so no trace records how many experts a layer
  read. Two sources fill the gap:
  - floor.py inverts the bundle's FC1 fit, t = 14.2 µs + 11.41 µs × experts, at each
    window's traced FC1 time. The greedy window read 19.9 experts a layer and the drafter
    9.2.
  - [model.py](model.py) takes N(r), the distinct experts that r consecutive tokens reach,
    from the reference forward's routes.

  The two agree: the reference's per-layer counts correlate 0.80 with the traced
  per-layer FC1 times.
- **DRAM busy and idle.** DRAM is counted busy for the union of every kernel that streams
  weights (GEMMs, expert GEMMs, heads, the mHC projection's `fn` read, the L2 prefetch's
  side stream). The rest of the step is DRAM-idle, split by what runs then.

### Where the bytes are (c1, per rank, at the suite's live rows)

| Family | MB a step | ms at 266.6 GB/s |
| --- | ---: | ---: |
| Routed experts, target (40 layers × 13.8 distinct × 4.57 MB) | 2,520 | 9.45 |
| Attention projections (Q-A/KV, Q-B, WO-A, WO-B; 40 layers) | 1,543 | 5.79 |
| LM heads (target and drafter, 248 MB each) | 496 | 1.86 |
| Shared expert (40 layers) | 354 | 1.33 |
| Drafter (3 layers: attention, shared expert, router, `fn`; routed 3 × 9.2) | 158 + 126 | 1.06 |
| Router gate, BF16 | 157 | 0.59 |
| mHC `fn`, FP32 (80 boundaries) | 157 | 0.59 |
| Engram `wkv` (layers 1, 14) | 79 | 0.30 |
| Indexer Q-B and head weights, compressors, DSpark main projection, Markov head | 118 | 0.44 |
| **Total** | **5,708** | **21.41** |

KV and index-key reads are under 0.1 ms at these prompt lengths. They reach roughly 0.3-0.5
ms a step at the 64-100K contexts of agent traffic.

### Where the traced step goes

| c1 greedy, 34.3 ms traced | ms | share |
| --- | ---: | ---: |
| Byte floor (traced window, 19.9 experts a layer) | 25.63 | 75% |
| Streaming below the attainable rate | 5.69 | 16.6% |
| DRAM idle: other small kernels | 0.96 | 2.8% |
| DRAM idle: collectives | 0.95 | 2.8% |
| DRAM idle: no kernel (launch gaps, host) | 0.70 | 2.1% |
| DRAM idle: attention, indexer, cache writes | 0.40 | 1.2% |

- **Sampled** (27.8 ms, 3-row buckets, 11.3 experts a layer): floor 70%, below-rate
  4.8 ms, idle 3.5 ms.
- **Mixed c8:** floor 76%.

### Achieved rates by family (the GEMM-efficiency lever)

- **FC1** is at the bench's own speed in serving: 241.8 µs a traced layer is 19.9 experts
  at the fit.
- **The routed experts' fixed costs.** Every launch pays a fixed cost: FC1 about 14 µs,
  FC2 9-15 µs. That is 1.24 ms a step over 40 target and 3 drafter layers, the largest
  single item of the below-rate time (+4.2%). It is why FC1 reaches only 208 GB/s at six
  experts against 256 GB/s at 36.

  One continuous FC1 and FC2 stream (fusion survey row 11) is what removes it.
- **FC2** streams at 91.7% of FC1's per-byte rate in the traces. In the bench it reaches
  90-97%, rising with experts. At FC1's rate it would save 0.27 ms (+0.9%).
- **LM heads:** 240.7 GB/s, 90% of the marginal rate, so 0.20 ms (+0.7%).
- **Small streams:** Engram `wkv` 219 GB/s, DSpark main projection 196, compressors 151, and
  mHC `project_streams` 166 for its `fn` read. Together they are 0.55 ms of excess at the
  marginal rate.
- **The prefetched GEMMs** (attention, shared expert and router: 2.05 GB, 7.7 ms of DRAM
  time) are read by two kernels: the L2 prefetch, at about 205 GB/s while it runs alone,
  and their consumers.
  - The consumers show 5.0 ms on the main stream.
  - Their excess over L2 speed is 1.6 ms (the PDL record, #589).
  - Most of the rest of the below-rate time, about 3.4-3.7 ms (the fixed costs and FC2's
    gap overlap), sits here.

### Production, calibrated

[model.py](model.py) writes [model.json](model.json):

```sh
python3 -I experiments/2026-10-26-c1-100tps/model.py
```

It treats a step as the traced non-byte time (8.70 ms) plus the bytes over 266.6 GB/s plus
the dead rows.

- **Routes.** It takes N(r) from the reference routes: 6, 10.3, 14.2, 17.7, 21.0, 24.1, 27.2
  and 29.6 for 1-8 rows, against 6, 11.9, 17.7, 23.4 for independent uniform routing.
- **Dead rows** cost 0.37 ms each, the r6v static table's per-row step on identical rows.
- **Live rows.** Solving the suite's 30.55 ms step gives 2.89 live routed rows and 1.18
  dead rows of the 4.07 verified rows, at 13.8 experts a layer and a floor of 21.4 ms.
- **Cross-check.** The r6v budget screen measured −8.6% for verifying all five drafts
  ([dspark-budget-rule](../2026-10-18-dspark-budget-rule/README.md)). The model gives
  −5.9% if full verification reaches 2.60 tokens a step and −12.9% at 2.41, which puts
  full verification at about 2.5 tokens a step.

## Tokens a step

What a token costs at c1:

- **A row is dear, an accepted token is free.** At fixed rows the step does not depend on
  how many drafts are accepted. But a live row adds 3.5-3.8 experts a layer (2.9 ms
  together with its dense share), and a dead row 0.37 ms. To pay for itself, a live row
  must add 0.22 tokens a step, and a dead row 0.03.
- **Today:** 2.27 tokens from 4.07 verified rows (1022 suite). About 2.9 of those rows are
  live.
  - The positions the policy leaves unverified would be admitted only below their
    marginal survival. Taking that survival at 0.07-0.17 puts full verification at
    T5 = 2.41-2.60 tokens a step; the budget screen points at about 2.5.
  - Survival is unmeasured past position 3 at c1. The 1022 c16 model step gives biased
    rates 0.75 / 0.69 / 0.69 / 0.72 / 0.68. Position-1 acceptance on live traffic is 0.72.

| Lever | Tokens a step | Step | tok/s | vs today |
| --- | ---: | ---: | ---: | ---: |
| Today (suite) | 2.27 | 30.55 | 74.3 | |
| Oracle drafter, 5 drafts | 6.00 | 37.2 | 161.4 | +117% |
| Oracle verification policy (T5 2.60 / 2.41, verified rows = tokens) | 2.60 / 2.41 | 29.3 / 28.8 | 88.6 / 83.5 | +19% / +12% |
| Two more drafts as dead rows (tail survival 0.06 + 0.04) | 2.37 | 31.3 | 75.8 | +2.0% |
| Tree: a second candidate at position 1 (P(reject) 0.25, always right) | 2.52 | 33.5 | 75.3 | +1.3% |
| The same, right half the time | 2.40 | 33.5 | 71.6 | −3.7% |
| Every draft verified (T5 2.60 / 2.41) | 2.60 / 2.41 | 37.2 | 69.9 / 64.7 | −6% / −13% |
| Block against standard rejection at temperature 0 | same | same | same | 0 |

Notes:
- **Oracle drafter.** It is the only lever that clears 100 with margin, and it is out of
  reach. DSpark is the checkpoint's own drafter, trained on a block of 5. A better drafter
  changes no output, but it is a model change and an owner decision, not a serving one.
- **Verification policy.** The oracle bound is a ceiling only. Confidences are calibrated
  on average but do not single out which drafts fail. The budget's sweeps so far were
  level or below today's rule:
  - thresholds 0.2 and 0.3;
  - linear row costs 0.02 and 0.035;
  - the static table;
  - r6x's rule.

  A two-price cost model may recover a point or two. It would charge live rows about 9.6%
  and dead rows about 1.2% of the step; today's single linear 0.035 is about 1.1 ms a row.
  That is the realizable part.
- **More drafts.** DSpark drafts a block of 5. Positions 6-7 need either a second drafter
  pass or untrained positions; their tail survival is at most a few points.
- **Trees.** Branches cost little to draft and a lot to verify, because each extra row
  reaches new experts. At c1 on GB10 a tree pays only if its second candidates are nearly
  always right. Trees belong at higher concurrency, where distinct experts saturate and
  rows get cheaper.
- **Block against standard rejection.** Greedy drafts are one-hot, so block and standard
  rejection commit the same tokens at temperature 0. The recipe's `block` matters only for
  sampled requests.
- **Agent traffic.** The 2026-10-23 copy probe (`2026-10-23-next-release`) measured DSpark
  at about 5 tokens a step on copied text and a suffix drafter at 15-16. With rows priced
  as above, a 17-row copy step takes about 56 ms for 15.9 tokens, against 37 ms for 5
  (N(17) extrapolated at 2.4 experts a row). At the agent proxy's 19-38% copy share that is
  about +6-15% tok/s, not the +7-17% tokens a step quoted before row costs. The suite's
  distinct prompts barely copy. AgentX (#574) has no window yet.

## Communication

- **Totals at c1 greedy.** The one-shot all-reduces and all-gathers run 4.71 ms of kernels
  a step, 13.7% of the traced step. The fusion survey's critical-path share is 12.1% for
  the all-reduces.
- **DRAM idles for only 0.95 ms of that time.** The rest overlaps the L2 prefetch, which
  fills the next windows during the waits.
- **A faster collective therefore saves at most +3.2% at c1.** The prefetch has less time,
  and its bytes move onto the consumer GEMMs.
- **What a collective is made of** (anatomy, 2026-10-07):
  - staging and reduce: 13-15 us;
  - hops: about 1 us;
  - the rest is rank skew, 79% of it compute jitter, with no slow node.
- **sparknet 0.5.0** (wide data phase, one poller; −1.6% TP3 step) is already in
  spark-ds41f-1025.
- **Producer-publish and consumer-post fusions** (fusion survey rows 4-5, sparknet
  `fused-halves`) live inside the same +3.2%. They earn more only where they also make
  the weight stream faster.

## Ranked levers

Ceilings are at c1 on the release suite, from 74.3 tok/s, one lever at a time. They come
from model.json's measured basis. Rows do not add up: the overhead rows contain
communication, fusion, PDL and the fixed costs.

| # | Lever | Ceiling at c1 | Evidence | Effort | Risk |
| --- | --- | --- | --- | --- | --- |
| 1 | A drafter that accepts all 5 (oracle) | 161 tok/s (+117%) | reference routes, row price | XL (drafter training) | outside constraints: a model change, owner decision |
| 2 | Every non-byte millisecond removed (fusion, PDL, prefetch timing, fixed costs, collectives, launch gaps) | 103.9 tok/s (+40%) | floor.py: busy/idle split of the traced step at the measured rate | L-XL, the sum of the fusion survey, PDL, audit and moe-overlap plans | the sum of theirs; most stay bit-identical |
| 2a | Weights stream at the attainable rate | +23% (91.3) | busy − floor 5.69 ms: fixed costs, prefetch and consumers, FC2, heads, small streams | L | low-medium |
| 2b | DRAM-idle time to zero | +10.9% (82.5) | small kernels 0.96, collectives 0.95, gaps 0.70, attention 0.40 ms | L | low |
| 3 | Oracle verification policy (realizable: a point or two) | +12-19% (83-89) | suite: 4.07 verified rows for 2.27 tokens; T5 bracket | S-M for a two-price budget | none to outputs (exact); policy only |
| 4 | Entropy-coded weights (lossless) | +5.5% (78.4) | measured: FP8 codes 6.58/8 bits, FP4 nibbles 3.90/4 (runs/weight_stats-dgx4.json) | XL (decode at line rate inside the GEMMs) | high |
| 5 | Routed experts' launch fixed costs (one continuous FC1+FC2 stream; inside 2a) | +4.2% (77.5) | bundle fit: FC1 14.2 µs, FC2 14.5 µs a launch, 86 launches | L (#583 row 11) | none (same per-tile math); liveness needs a deadline trap |
| 6 | Both heads through a certified exact screen (4-bit copy finds candidates; 12-bit rows rescore them) | +3.8% (77.2) | heads 1.86 ms at the marginal rate | L | owner decision: a reduced copy beside the native head; draft confidences become bounds |
| 7 | Communication alone (sparknet, AR fusions) | ≤ +3.2% (76.7) | collectives 4.71 ms, 0.95 ms with DRAM idle | M-L | low; wire changes |
| 8 | Q-A/KV column-sharded with one all-gather a layer (exact) | +2.1% (75.9) | 367 MB replicated a rank; all-gather 11.2 us | M | low: same per-output sums |
| 9 | Two more drafts as dead rows | ≤ +2.0% | tail survival guess | M (drafter positions 6-7) | DSpark is trained on 5 |
| 10 | Every byte at 273 GB/s (the specification) | +1.7% (75.6) | measured marginal 266.6 GB/s | none in reach | - |
| 11 | FC2 at FC1's per-byte rate (inside 2a) | +0.9% (75.0) | FC2 at 91.7% of FC1 traced, 90-97% bench | S-M (retile probe) | none |
| 12 | LM heads at the marginal rate (inside 2a) | +0.7% (74.8) | 240.7 against 266.6 GB/s | M | none |
| 13 | Suffix drafting (agent traffic only) | suite ~0; agents +6-15% | copy probe with row prices | L (hybrid drafter, variable blocks) | none to outputs |
| 14 | Draft trees at c1 | +1.3% at best, −3.7% likely | row price against second-candidate odds | XL | none to outputs |
| - | mHC `fn` as BF16 | 0 | measured: 42,270,042 of 42,270,720 values are not exactly BF16 | - | would change arithmetic |

Next week's draft PRs (#583-#594, about 3-5% at c1) sit inside rows 2, 5 and 11.

## Combinations

| Set | Tokens a step | Step | tok/s |
| --- | ---: | ---: | ---: |
| Credible: half the overhead (next week's PRs, the fusion survey's next steps, one continuous expert stream), Q-A/KV sharded | 2.27 | 25.6 | 88.8 |
| The same, and 3% more tokens a step (two-price budget, drafter window fix) | 2.34 | 25.6 | 91.5 |
| Two thirds of the overhead, Q-A/KV sharded | 2.27 | 24.1 | 94.1 |
| Stretch: 75% of the overhead, the exact byte cut, the head screen, entropy coding | 2.27 | 20.7 | 109.8 |
| The whole floor and the exact byte cut | 2.27 | 21.2 | 107.0 |
| The whole floor and the oracle verification policy | 2.60 | 20.0 | 130.0 |

**The stretch set needs things outside today's constraints:**
- the owner accepting a screening copy of the heads;
- lossless entropy-coded GEMM operands decoded at line rate, which no one has measured on
  GB10;
- three quarters of the non-byte time removed.

On the survey's critical-path measure, the whole fusion budget is 12.5% of the step;
three quarters of the non-byte time is 21%.

**Tokens a step needed for 100 tok/s,** with today's extra live and dead rows:

| Overhead removed | 0% | 25% | 50% | 75% | 100% |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tokens a step | 3.31 | 3.02 | 2.74 | 2.45 | 2.27 |
| vs 2.27 | +46% | +33% | +21% | +8% | 0 |

Every lever within our constraints adds at most a few percent of tokens a step on the suite.
Only the oracle-policy bound (2.41-2.60) and a stronger drafter reach the range.

## Review of the 2026-10-24 plan ("Toward 100 tokens a second", [tilelang-completion](../2026-10-24-tilelang-completion/README.md))

- **The routed MoE is 49% of the main stream.** Holds. In bytes it is 44% of the floor.
  FC1 streams at its bench speed, 252 GB/s at the traced 20 experts. The MoE expert
  pipeline at six rows is worth its launch fixed costs (+4.2%) and FC2's gap (+0.9%).
- **"A plausible 15-25% faster step gives 86-97 tok/s."** Consistent. 15-25% is a third to
  two thirds of the 8.7 ms non-byte time: 86-94 tok/s. The whole of it gives 103.9.
- **"The step's cost barely depends on how many drafts are accepted, so 3.0 tokens a step
  would be about +34%"** (from `2026-10-18-dspark-acceptance`). That holds only at fixed
  rows. More tokens a step mostly need more live rows, and each costs 9.6% of the step.
  3.0 tokens at today's 0.62 wasted live rows gives 92.9 tok/s (+25%). The +34% would
  require every extra token to come from rows already verified.
- **"Draft trees (ceiling to measure)."** Bounded here without logs: at c1 a tree row costs
  more than any realistic second candidate brings back. Logging drafter distributions for
  trees can wait for a higher-concurrency study.
- **"Suffix drafting on agent traffic, +7-17% at one stream."** That is tokens a step
  before verification. In tok/s it is about +6-15% on agent traffic, and nothing on the
  suite.
- **"Measure each kernel's achieved bandwidth against GB10's roofline at one and sixteen
  streams."** Done here for c1 and c8 from the existing traces (floor.py) with the
  expert-rate bundle.

## Runs on dgx4 (2026-10-25, CET)

dgx4 is uncabled and serves nothing. Every run used its own checkout of this branch,
`~/spark-lab/agents/c1-100tps/spark-ds41f`, and the image `spark-ds41f:19f2c20ed4d6-next-6`
(`sha256:82b071b57485`, spark-ds41f-1025). `~/spark-request.json` was absent before each run.

1. **Expert rate, run 1:** commit 8aeb9c4f, 19:08 CET, **FAIL** (exit 1), receipts
   [runs/expert-rate-dgx4-1.txt](runs/expert-rate-dgx4-1.txt) and
   [.verdict.json](runs/expert-rate-dgx4-1.verdict.json).

   ```sh
   python3 scripts/lab.py kernel-local experiments/2026-10-26-c1-100tps/bundles/expert-rate \
     --out ~/spark-lab/agents/c1-100tps/out/expert-rate
   ```

   - All 14 expert cases completed.
   - The plain-read reference raised a `TypeError`: `torch.sum(b, out=o)` takes `out=`
     only with a `dim`.
   - Fixed in 562301b4 (`dim=0`).
2. **Expert rate, run 2:** commit 562301b4, the same command, 20:03 CET, **pass** (9.8 s),
   receipts [runs/expert-rate-dgx4-2.txt](runs/expert-rate-dgx4-2.txt) and
   [.verdict.json](runs/expert-rate-dgx4-2.verdict.json). The node output is
   `~/spark-lab/agents/c1-100tps/out/expert-rate/`. Run 2's expert cases repeat run 1's
   within 1-2%. [expert_rate_table.py](expert_rate_table.py) tabulates them:

   | Rows | Experts | FC1 µs | FC1 GB/s | FC2 µs | FC2 GB/s |
   | ---: | ---: | ---: | ---: | ---: | ---: |
   | 6 | 6 | 87.4 | 208.8 | 48.6 | 188.2 |
   | 6 | 12 | 151.3 | 241.4 | 83.9 | 218.2 |
   | 6 | 18 | 217.7 | 251.6 | 118.0 | 232.8 |
   | 6 | 24 | 288.0 | 253.6 | 152.1 | 240.8 |
   | 6 | 36 | 427.3 | 256.3 | 221.9 | 247.5 |
   | 3 | 6 | 87.0 | 209.9 | 44.2 | 207.2 |
   | 3 | 18 | 217.4 | 251.9 | 114.4 | 240.0 |

   - **Fits:**
     - FC1 at six rows: 14.2 µs + 11.41 µs an expert (266.6 GB/s marginal);
     - FC1 at three rows: 17.1 + 11.03 (275.8);
     - FC2 at six rows: 14.5 + 5.75 (265.2);
     - FC2 at three rows: 9.2 + 5.84 (261.3).
   - **Plain read:** 221.5, 248.4, 253.7 and 253.4 GB/s at 18, 55, 110 and 220 MiB.
   - **Verdict:** FC1 at six rows matches the traced serving FC1 at 19.9 experts. The
     attainable stream is about 266 GB/s at the margin. The per-launch fixed costs, not
     the stream rate, make small expert counts slow.
3. **Weight statistics:** commit 1d06aab0, 19:06-19:11 CET, completed. The owner approved it
   (relayed by the session that started this one). Receipt: [runs/weight_stats-dgx4.json](runs/weight_stats-dgx4.json).

   ```sh
   nice -n 10 python3 -I experiments/2026-10-26-c1-100tps/weight_stats.py \
     ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170 \
     --out ~/spark-lab/agents/c1-100tps/out/weight_stats.json
   ```

   - **What it read,** in place through the snapshot's paths, writing nothing in the hub:
     - the 48 shards' headers;
     - all 86 FP32 mHC `fn` tensors whole;
     - up to 8 MiB of each FP8 and BF16 tensor (3.6 GB);
     - every 32nd routed expert's packed weights (8.7 GB).
   - **mHC `fn`:** 42,270,042 of 42,270,720 values have low mantissa bits, so it is not
     BF16 (that lever is gone).
   - **Entropy:** E4M3 codes 6.579 of 8 bits; E2M1 nibbles 3.895 of 4; the BF16 sign and
     exponent byte 2.88 of 8.

## Lab specs (prepared, not queued)

1. **[c1-acceptance.json](c1-acceptance.json)** runs a `measure` job with the suite file
   [c1-acceptance-suite.json](c1-acceptance-suite.json), on arms spark-ds41f-1025,
   [verify-all](arm-verify-all.json) and spark-ds41f-1025 again.
   - It runs every suite prompt once at c1 with the per-position speculative counters,
     then the single-stream closed loop.
   - The verify-all arm verifies all five drafts. It is a measurement arm, not a
     candidate, as in the r6v budget screen. Temperature-0 text does not depend on the
     verification policy, so its accepted counts by position are the drafter's unbiased
     survival on the release's own text (T5, rows 3 and 9).
   - Its step is the price of six live rows on that same text (the row price).
   - The arms differ only in the speculative config: `enable_adaptive_verification`
     false, with `adaptive_verification_row_cost` dropped, since the config refuses it
     without adaptive verification.
   - It plans cleanly with `scripts/lab.py run --dry-run`. It needs the TP4 cluster and
     this record on `origin/main` (the lab syncs to it). About 40 minutes.
2. **[expert-rate.json](expert-rate.json)** is the bundle as a cluster `kernel` job (dgx2
   and dgx4), for a repeat on another node. It ran on dgx4 above with `kernel-local`.
3. **The profiled step without Python stacks.** The kernel audit's `audit-profile.json`
   (#585) already profiles candidate 6 without them. floor.py runs on its traces unchanged
   and would replace this record's overhead, which the stacked traces may overstate.

## Caveats

- **The routing sample is small.** It is one prompt and eight tokens, so three windows of
  six rows a layer. The traced windows' expert counts (19.9 at mostly six-row buckets,
  11.3 at three rows) fall where it predicts for 4-5 and 3 live rows.
- **The overhead comes from the traced greedy window.** It assumes the suite's step has
  the same absolute non-byte time. The stacked traces may overstate host gaps.
- **Survival past position 3 at c1 is bracketed, not measured.** c1-acceptance.json
  measures it.
- **The bundle times the expert GEMMs alone on one node.** In serving they share DRAM with
  the shared expert's stream and the L2 prefetch; the traced FC1 matches the bench anyway.
- **`docs/fundamentals.md` mislabels one weight.** It lists "DSpark main projection 6400 ×
  6144, 37.5 MiB". That is Engram's `wkv` (6144 → 5 × 5120, column TP). DSpark's
  `main_proj` is 15,360 → 5,120, 1280 × 15360 FP8 a rank (19.7 MB). The doc is left for a
  follow-up.

## Files

- [floor.py](floor.py), [floor.json](floor.json): bytes, DRAM busy and idle, and the floor
  per window, on the measured basis and the first estimate.
- [model.py](model.py), [model.json](model.json): the calibrated c1 model, every lever,
  the combinations and the 100 tok/s frontier.
- [bundles/expert-rate](bundles/expert-rate), [expert-rate.json](expert-rate.json),
  [expert_rate_table.py](expert_rate_table.py): the FC1/FC2 rate bundle, its spec and its
  table.
- [c1-acceptance.json](c1-acceptance.json), [c1-acceptance-suite.json](c1-acceptance-suite.json),
  [arm-1025.json](arm-1025.json), [arm-verify-all.json](arm-verify-all.json): the
  acceptance and row-price measurement.
- [weight_stats.py](weight_stats.py): the lossless-byte questions.
- [runs/](runs): the dgx4 receipts.
