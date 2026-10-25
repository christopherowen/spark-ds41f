# One stream at 100 tokens a second: what it would take (TP4, c1)

Owner, 2026-10-25: does next week's work get single-stream decode to 100 tok/s? It does not.
The fusion budget tops out near 86 tok/s. This record bounds the levers that could close the
rest, from traces and data we already have:

- tokens a step;
- GEMM bandwidth efficiency;
- communication.

Rules: native weight precision, correctness and determinism first, no fallback paths, and
bound before building. Nothing was built and no GPU time was used. The two measurements the
ceilings still need are prepared as lab specs, not queued.

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
- the recipe at `config/cluster.json` and the vLLM tree 26382f5cf, read only.

## The short answer

- **The byte floor is 77% of today's step.** At the measured 240.7 GB/s, a c1 step's DRAM
  reads (5.67 GB a rank) take 23.5 ms against today's 30.55 ms. Removing everything that
  is not a byte gives at most 94.6 tok/s at today's 2.27 tokens a step (+27%).
- **That cap covers every kernel, fusion, PDL and collective change together.**
  - DRAM idles for 3.0 ms a step: small kernels, launch gaps, collectives and attention.
    That is +10.9%.
  - Weights stream below the attainable rate for another 3.5 ms. That is +13%: the L2
    prefetch and its consumers, FC2, the small GEMMs.
  - The all-reduces run 4.7 ms of kernels a step, but DRAM idles under them for only
    0.95 ms. The L2 prefetch streams through the waits, so communication is worth at most
    +3.2% at c1, not 12%.
- **A verified row is expensive at c1.** Each live row reaches 3.5-3.8 new experts a
  layer: 640-690 MB, about 3.2 ms with its dense share, or 10.6% of the step. That price bounds
  every acceptance lever:
  - **trees:** +0.4% if the second candidate were always right, −4.6% at a realistic 50%;
  - **two more drafts:** about +2%;
  - **an oracle verification policy:** +12-19%, but it is not realizable. The policy sweeps
    so far (thresholds, linear row costs, the static table) landed level with today's rule
    or below it.
  - **an oracle drafter:** 6 tokens a step at 38 ms, about 157 tok/s. It is the one lever
    that clears 100 with room, but it is not in our gift: DSpark's weights are the
    checkpoint's.
- **No credible combination reaches 100 under our constraints on the release suite's text.**
  - Half the overhead removed, plus the two exact byte cuts (Q-A/KV column-sharded, `fn` as
    BF16), gives 86.7 tok/s. With 3% more tokens a step, 89.3.
  - 100 needs 2.9 tokens a step with half the overhead gone (+28%), or 3.35 at today's
    overhead.
  - The levers that would give it are a stronger drafter, suffix drafting on copy-heavy
    agent text (about +6-15% tok/s there, nothing on the suite), or a stretch set that
    includes an owner decision and a research item (104 tok/s, below).
- **Measure next, before any build:**
  - [expert-rate.json](expert-rate.json): FC1 and FC2 against a known number of distinct
    experts. It pins FC1's efficiency, which the model already bounds at 91% or more.
  - [c1-acceptance.json](c1-acceptance.json): the drafter's unbiased survival by position
    and the price of six live rows, on the suite's own temperature-0 text.

## How the floor was computed

[floor.py](floor.py) reads the traces and writes [floor.json](floor.json):

```sh
python3 -I experiments/2026-10-26-c1-100tps/floor.py results/private/determinism/request-inventory-5/ports
```

- **Bytes.** Every weight byte a TP4 rank reads in one step, from the checkpoint's shapes:
  - FP8 block-32 projections with their scales, BF16 router and compressors, FP32 `fn`;
  - the packed 12-bit heads;
  - the routed experts as E2M1 plus coded scales: 3.04 MB for FC1 and 1.53 MB for FC2 per
    expert at TP4.
- **Distinct experts are not in any trace.** A grouped GEMM's grid is the bucket's maximum
  routes (`(9, 36)` at six rows); unused tiles exit early.
  - floor.py infers the count from FC1's time at an assumed share of the attainable rate.
  - [model.py](model.py) takes it from the reference forward's routes instead.
  - The two agree: the reference's per-layer expert counts correlate 0.80 with the traced
    per-layer FC1 times.
- **DRAM busy and idle.** DRAM is counted busy for the union of every kernel that streams
  weights (GEMMs, expert GEMMs, heads, the mHC projection's `fn` read, the L2 prefetch's
  side stream). The rest of the step is DRAM-idle, split by what runs then.
- **Attainable rate: 240.7 GB/s.** This is the packed LM head in these traces, 248 MB in
  1.03 ms. GB10's 273 GB/s specification has never been measured on this stack:
  - `cudaMalloc` sequential read: 235 GB/s;
  - in-place stream: 222 GB/s;
  - one unexplained buffer: 253-265 GB/s (`2026-10-24-tilelang-completion`, placement
    rounds).

### Where the bytes are (c1, per rank, at the suite's live rows)

| Family | MB a step | ms at 240.7 GB/s |
| --- | ---: | ---: |
| Routed experts, target (40 layers × 13.6 distinct × 4.57 MB) | 2,477 | 10.29 |
| Attention projections (Q-A/KV, Q-B, WO-A, WO-B; 40 layers) | 1,543 | 6.41 |
| LM heads (target and drafter, 248 MB each) | 496 | 2.06 |
| Shared expert (40 layers) | 354 | 1.47 |
| Drafter (3 layers: attention, shared expert, router, `fn`; routed 3 × 9.4) | 158 + 129 | 1.19 |
| Router gate, BF16 | 157 | 0.65 |
| mHC `fn`, FP32 (80 boundaries) | 157 | 0.65 |
| Engram `wkv` (layers 1, 14) | 79 | 0.33 |
| Indexer Q-B and head weights, compressors, DSpark main projection, Markov head | 118 | 0.49 |
| **Total** | **5,667** | **23.54** |

KV and index-key reads are under 0.1 ms at these prompt lengths. They reach roughly 0.3-0.5
ms a step at the 64-100K contexts of agent traffic.

### Where the traced step goes

| c1 greedy, 34.3 ms traced | ms | share |
| --- | ---: | ---: |
| Byte floor (traced window, 19.1 experts a layer) | 27.78 | 81% |
| Streaming below the attainable rate | 3.54 | 10.3% |
| DRAM idle: other small kernels | 0.96 | 2.8% |
| DRAM idle: collectives | 0.95 | 2.8% |
| DRAM idle: no kernel (launch gaps, host) | 0.70 | 2.1% |
| DRAM idle: attention, indexer, cache writes | 0.40 | 1.2% |

The sampled window (27.8 ms, 3-row buckets) has the same shape: floor 78%, below-rate 2.7 ms,
idle 3.5 ms. Mixed c8 is at 81%.

### Achieved rates by family (the GEMM-efficiency lever)

- **FC1.** Its rate is unknown alone, because the experts behind it are. The suite's step
  bounds it:
  - with the overhead measured here, FC1 below 91% of the attainable rate leaves fewer
    live rows than the suite's 2.27 tokens a step;
  - the dead-row rule forbids that: a request commits at most one token past its last live
    row (`adaptive_verification.py`).
  - So FC1 runs at 91-100%. expert-rate.json measures it.
- **FC2** streams at 91.7% of FC1's per-byte rate (95% sampled, 88% at c8). At FC1's rate
  it would save 0.3 ms at c1 (+1.0%).
- **LM heads:** 240.7 GB/s. This is the reference rate; nothing is left there.
- **Engram `wkv`:** 219 GB/s (91%).
- **DSpark main projection:** 196 GB/s (81%).
- **Compressors:** 151 GB/s (63%).
- **mHC `project_streams`:** 166 GB/s (69%) for its `fn` read.

  These four are 0.43 ms of excess together.
- **The prefetched GEMMs** (attention, shared expert and router: 2.05 GB, 8.5 ms of DRAM
  time) are read by two kernels: the L2 prefetch, at about 205 GB/s while it runs alone,
  and their consumers.
  - The consumers show 5.0 ms on the main stream.
  - Their excess over L2 speed is 1.6 ms (the PDL record, #589).
  - Most of the 3.5 ms below-rate time sits here.

### Production, calibrated

[model.py](model.py) writes [model.json](model.json):

```sh
python3 -I experiments/2026-10-26-c1-100tps/model.py
```

It treats a step as the overhead above plus the bytes over the rate plus the dead rows.

- **Routes.** It takes N(r), the distinct experts that r consecutive tokens reach, from
  the reference routes: 6, 10.3, 14.2, 17.7, 21.0, 24.1, 27.2 and 29.6 for 1-8 rows,
  against 6, 11.9, 17.7, 23.4 for independent uniform routing.
- **Dead rows** cost 0.37 ms each, the r6v static table's per-row step on identical rows.
- **Live rows.** Solving the suite's 30.55 ms step gives 2.83 live routed rows and 1.25
  dead rows of the 4.07 verified rows, at a floor of 23.5 ms.
- **Cross-check.** The model prices verifying all five drafts at −8% if that reaches 2.60
  tokens a step. The r6v budget screen measured −8.6%
  ([dspark-budget-rule](../2026-10-18-dspark-budget-rule/README.md)).

## Tokens a step

What a token costs at c1:

- **A row is dear, an accepted token is free.** At fixed rows the step does not depend on
  how many drafts are accepted. But a live row adds 3.5-3.8 experts a layer (3.2 ms
  together with its dense share), and a dead row 0.37 ms. To pay for itself, a live row
  must add 0.24 tokens a step, and a dead row 0.03.
- **Today:** 2.27 tokens from 4.07 verified rows (1022 suite). About 2.8 of those rows are
  live.
  - The positions the policy leaves unverified would be admitted only below their
    marginal survival. Taking that survival at 0.07-0.17 puts full verification at
    T5 = 2.41-2.60 tokens a step.
  - Survival is unmeasured past position 3 at c1. The 1022 c16 model step gives biased
    rates 0.75 / 0.69 / 0.69 / 0.72 / 0.68. Position-1 acceptance on live traffic is 0.72.

| Lever | Tokens a step | Step | tok/s | vs today |
| --- | ---: | ---: | ---: | ---: |
| Today (suite) | 2.27 | 30.55 | 74.3 | |
| Oracle drafter, 5 drafts | 6.00 | 38.1 | 157.5 | +112% |
| Oracle verification policy (T5 2.60 / 2.41, verified rows = tokens) | 2.60 / 2.41 | 29.4 / 28.8 | 88.4 / 83.4 | +19% / +12% |
| Two more drafts as dead rows (tail survival 0.06 + 0.04) | 2.37 | 31.3 | 75.8 | +2.0% |
| Tree: a second candidate at position 1 (P(reject) 0.25, always right) | 2.52 | 33.8 | 74.6 | +0.4% |
| The same, right half the time | 2.40 | 33.8 | 70.9 | −4.6% |
| Every draft verified (T5 2.60 / 2.41) | 2.60 / 2.41 | 38.1 | 68.2 / 63.2 | −8% / −15% |
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

  A two-price cost model may recover a point or two. It would charge live rows about 10.6%
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
  as above, a 17-row copy step takes about 59 ms for 15.9 tokens, against 38 ms for 5
  (N(17) extrapolated at 2.4 experts a row). At the agent proxy's 19-38% copy share that is
  about +6-15% tok/s, not the +7-17% tokens a step quoted before row costs. The suite's distinct prompts barely copy. AgentX (#574)
  has no window yet.

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
from model.json at FC1 = 100% of the attainable rate. At the 91% bound the byte rows move by
under a point; the overhead, policy and drafter rows by up to 6 points. Rows do not add up: the overhead rows contain
communication, fusion and PDL.

| # | Lever | Ceiling at c1 | Evidence | Effort | Risk |
| --- | --- | --- | --- | --- | --- |
| 1 | A drafter that accepts all 5 (oracle) | 157 tok/s (+112%) | reference routes, row price | XL (drafter training) | outside constraints: a model change, owner decision |
| 2 | Every non-byte millisecond removed (fusion, PDL, prefetch timing, collectives, launch gaps) | 94.6 tok/s (+27%); about 100 at FC1 91% | floor.py: busy/idle split of the traced step | L-XL, the sum of the fusion survey, PDL, audit and moe-overlap plans | the sum of theirs; most stay bit-identical |
| 2a | Weights stream at the attainable rate (prefetch and consumers, FC2, small GEMMs) | +13% (84.1) | busy − floor 3.54 ms; prefetch at ~205 GB/s; consumers' 1.6 ms excess (#589) | L | low-medium |
| 2b | DRAM-idle time to zero | +10.9% (82.5) | small kernels 0.96, collectives 0.95, gaps 0.70, attention 0.40 ms | L | low |
| 3 | Oracle verification policy (realizable: a point or two) | +12-19% (83-88) | suite: 4.07 verified rows for 2.27 tokens; T5 bracket | S-M for a two-price budget | none to outputs (exact); policy only |
| 4 | Every byte at 273 GB/s | +10% (81.8) | specification; 253-265 GB/s seen once, unexplained | probe S, cause unknown | likely a measurement artefact |
| 5 | Entropy-coded weights (lossless; code entropy unmeasured) | +5-9% (78-81) | illustrative 7.0/8 and 3.8/4 bits; weight_stats.py measures them | XL (decode at line rate inside the GEMMs) | high |
| 6 | Both heads through a certified exact screen (4-bit copy finds candidates; 12-bit rows rescore them) | +4.2% (77.5) | heads 2.06 ms at the attainable rate | L | owner decision: a reduced copy beside the native head; draft confidences become bounds |
| 7 | Communication alone (sparknet, AR fusions) | ≤ +3.2% (76.7) | collectives 4.71 ms, 0.95 ms with DRAM idle | M-L | low; wire changes |
| 8 | Q-A/KV column-sharded with one all-gather a layer (exact) | +2.5% (76.2) | 367 MB replicated a rank; all-gather 11.2 us | M | low: same per-output sums |
| 9 | Two more drafts as dead rows | ≤ +2.0% | tail survival guess | M (drafter positions 6-7) | DSpark is trained on 5 |
| 10 | mHC `fn` read as BF16, if every value is exactly BF16 | +1.1% (75.1) | 157 MB FP32; weight_stats.py checks | S | none if exact |
| 11 | FC2 at FC1's per-byte rate (inside 2a) | +1.0% (75.1) | FC2 at 91.7% of FC1 | S-M (retile probe, #583 row 11) | none |
| 12 | Suffix drafting (agent traffic only) | suite ~0; agents +6-15% | copy probe with row prices | L (hybrid drafter, variable blocks) | none to outputs |
| 13 | Draft trees at c1 | +0.4% at best, −4.6% likely | row price against second-candidate odds | XL | none to outputs |

Next week's draft PRs (#583-#594, about 3-5% at c1) sit inside rows 2 and 11.

## Combinations

| Set | Tokens a step | Step | tok/s |
| --- | ---: | ---: | ---: |
| Credible: half the overhead (next week's PRs and the fusion survey's next steps), Q-A/KV sharded, `fn` as BF16 | 2.27 | 26.2 | 86.7 |
| The same, and 3% more tokens a step (two-price budget, drafter window fix) | 2.34 | 26.2 | 89.3 |
| Stretch: 75% of the overhead, the exact byte cuts, the head screen, entropy coding at 7.0/3.8 | 2.27 | 21.8 | 104.4 |
| The whole floor and the exact byte cuts | 2.27 | 22.9 | 99.1 |
| The whole floor and the oracle verification policy | 2.60 | 21.8 | 119.3 |

The stretch set needs two things outside today's constraints: the owner accepting a screening
copy of the heads, and lossless entropy-coded GEMM operands decoded at line rate, which no one
has measured on GB10. It also needs three quarters of an overhead whose whole fusion budget,
on the survey's critical-path measure, is 12.5%.

**Tokens a step needed for 100 tok/s,** with today's extra live and dead rows:

| Overhead removed | 0% | 25% | 50% | 75% | 100% |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tokens a step | 3.35 | 3.12 | 2.90 | 2.68 | 2.46 |
| vs 2.27 | +48% | +37% | +28% | +18% | +8% |

Every lever within our constraints adds at most a few percent of tokens a step on the suite.
Only the oracle-policy bound (2.41-2.60) and a stronger drafter reach the range.

## Review of the 2026-10-24 plan ("Toward 100 tokens a second", [tilelang-completion](../2026-10-24-tilelang-completion/README.md))

- **The routed MoE is 49% of the main stream.** Holds. In bytes it is 44% of the floor,
  and FC1 already streams at 91-100% of the attainable rate. The MoE expert pipeline at
  six rows is worth FC2's 1% plus its share of the launch gaps.
- **"A plausible 15-25% faster step gives 86-97 tok/s."** The top of that range is the
  whole floor. Kernel work alone caps at +27% (94.6 tok/s), and only if every non-byte
  millisecond goes. A step 15% faster is about half the overhead: 86-87 tok/s, the
  credible set above.
- **"The step's cost barely depends on how many drafts are accepted, so 3.0 tokens a step
  would be about +34%"** (from `2026-10-18-dspark-acceptance`). That holds only at fixed
  rows. More tokens a step mostly need more live rows, and each costs 10.6% of the step.
  3.0 tokens at today's 0.56 wasted live rows gives 92.2 tok/s (+24%). The +34% would
  require every extra token to come from rows already verified.
- **"Draft trees (ceiling to measure)."** Bounded here without logs: at c1 a tree row costs
  more than any second candidate can bring back. Logging drafter distributions for trees
  can wait for a higher-concurrency study.
- **"Suffix drafting on agent traffic, +7-17% at one stream."** That is tokens a step
  before verification. In tok/s it is about +6-15% on agent traffic, and nothing on the
  suite.
- **"Measure each kernel's achieved bandwidth against GB10's roofline at one and sixteen
  streams."** Done here for c1 and c8 from the existing traces (floor.py), except FC1,
  which needs expert-rate.json.

## Lab specs (prepared, not queued)

Both plan cleanly with `scripts/lab.py run --dry-run`. They need this record on `origin/main`,
because the lab syncs the node checkouts to it, so they queue after the PR merges.

1. **[expert-rate.json](expert-rate.json)** runs a `kernel` job on dgx2 and dgx4: bundle
   [bundles/expert-rate](bundles/expert-rate) on spark-ds41f-1025's image.
   - The production MoE plan (coded scales, TP4 rank shapes) runs FC1 and FC2 at 3 and 6
     rows, with routes reaching exactly 6-36 of 384 experts. Weights are cold: a disjoint
     pool for each launch and an L2 flush before every replay.
   - A plain read of the same bytes is the reference.
   - It decides FC1's share of the attainable rate (row 2's 94.6 against 101 tok/s), FC2's
     gap at known counts, and how many experts each traced layer read.
   - About ten minutes, cluster stopped.
2. **[c1-acceptance.json](c1-acceptance.json)** runs a `measure` job with the suite file
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
   - About 40 minutes.
3. **[weight_stats.py](weight_stats.py)** runs on CPU with the standard library only and
   no GPU. It reads the snapshot by offset and sends nothing. It answers row 10 (is every
   mHC `fn` value exactly BF16?) and row 5 (the entropy of the E4M3 codes, E2M1 nibbles and
   BF16 exponents). It can run on dgx4's spare cores beside production, like the CPU
   reference runner, once the owner allows it:

   ```sh
   taskset -c 0-4,10-14 python3 -I experiments/2026-10-26-c1-100tps/weight_stats.py \
     ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170 \
     --out results/private/c1-100tps/weight_stats.json
   ```
4. **The profiled step without Python stacks.** The kernel audit's `audit-profile.json`
   (#585) already profiles candidate 6 without them. floor.py runs on its traces unchanged
   and would replace this record's overhead, which the stacked traces may overstate.

## Caveats

- **The routing sample is small.** It is one prompt and eight tokens, so three windows of
  six rows a layer. The 0.80 per-layer correlation with the traced FC1 times supports
  transferring it; the expert-rate bundle checks it.
- **The overhead comes from the traced greedy window.** It assumes the suite's step has
  the same absolute non-byte time.
- **Survival past position 3 at c1 is bracketed, not measured.** c1-acceptance.json
  measures it.
- **`docs/fundamentals.md` mislabels one weight.** It lists "DSpark main projection 6400 ×
  6144, 37.5 MiB". That is Engram's `wkv` (6144 → 5 × 5120, column TP). DSpark's
  `main_proj` is 15,360 → 5,120, 1280 × 15360 FP8 a rank (19.7 MB). The doc is left for a
  follow-up.

## Files

- [floor.py](floor.py), [floor.json](floor.json): bytes, DRAM busy and idle, and the floor
  per window.
- [model.py](model.py), [model.json](model.json): the calibrated c1 model, every lever,
  the combinations and the 100 tok/s frontier.
- [bundles/expert-rate](bundles/expert-rate), [expert-rate.json](expert-rate.json): the
  FC1/FC2 rate bundle and its spec.
- [c1-acceptance.json](c1-acceptance.json), [c1-acceptance-suite.json](c1-acceptance-suite.json),
  [arm-1025.json](arm-1025.json), [arm-verify-all.json](arm-verify-all.json): the
  acceptance and row-price measurement.
- [weight_stats.py](weight_stats.py): the lossless-byte questions.
