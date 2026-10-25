# Copy drafting beside DSpark: what wide verification costs, which variant pays, and the build

The question: does a copy drafter pay on DS4.1 serving? A suffix or n-gram drafter proposes the
tokens that followed an earlier occurrence of the context's tail. The candidate is a hybrid:
a copy proposal where one matches, DSpark everywhere else.

Two earlier records measured the copy ceiling:
- `2026-10-18-dspark-acceptance`, candidate 2: on the agent proxy, 38% of output tokens can be
  copied (19% if hidden reasoning copies like prose).
- `2026-10-23-next-release`, "Draft acceptance" (the copy probe, run `next-copy-tp4`): DSpark
  already takes about 5 of 6 tokens a step on copied text. A suffix drafter reaches 15.9
  tokens a step reproducing a file and 7.1 on an edit tool call, and nothing on prose. The
  estimate was +7-17% tokens a step on agent traffic, before the cost of verifying wide
  blocks.

This record bounds that cost first, then the net gain of each variant, then designs the variant
that clears the cost.

Rules:
- temperature-0 output bit-identical, speculation lossless;
- native weights; every kernel in TileLang, with no fallback paths;
- bound before building.

Base: origin/main f89a89eb (spark-ds41f-1025 promoted, candidate 6's TP3 specs merged). No
serving or cluster time was used: the cluster is qualifying TP3. Two runs used dgx4: a timing
bundle with `kernel-local`, and DeepSeek's reference forward on its CPU under the lab lock.
Nothing is built.

## The short answer

- **Wide verification is cheap per token at one stream.** One request's block of R copied rows
  costs its distinct experts plus a per-row price. Measured routes first, then the range of
  the two bounds (ordinary-text extrapolation to independent routes):

  | Rows | TP4 step, ms | TP4 ms a token, all accepted | TP3 step, ms |
  | ---: | --- | --- | --- |
  | 6 (DSpark's block) | 38.2 (38.3-45.5) | 6.4 (6.4-7.6) | 48.2 (48.3-57.9) |
  | 17 | 59.4 (62.0-88.7) | 3.5 (3.6-5.2) | 75.7 (79.2-114.8) |
  | 33 | 81.7 (95.7-140.9) | 2.5 (2.9-4.3) | 104.3 (123.0-183.3) |
  | 49 | 98.5 (127.8-181.6) | 2.0 (2.6-3.7) | 126.0 (165.0-236.7) |
  | 65 | 113.3 (160.0-214.6) | 1.7 (2.5-3.3) | 144.8 (207.0-279.8) |

  On copied text DSpark takes 5 tokens for 38 ms, 7.7 ms a token.
- **One request's consecutive rows share experts strongly, copied or not.** DeepSeek's
  reference forward on dgx4's CPU, over four conversations, measured how many distinct experts
  65 consecutive reply rows reach in a layer:

  | Reply | Distinct experts at 65 rows |
  | --- | ---: |
  | prose | 93 |
  | a copied file | 98 |
  | the file with comments added | 99 |
  | an edit tool call | 105 |

  The bounds were 166 (extrapolation) and 246 (independent routes). At 6 rows the reply rows
  reach 22-24 experts, what json-count's eight tokens gave.

  A copied row shares 38-48% of its six experts with its source row. The forced replies are
  the model's own: the reference's greedy token is the forced one at 900 of 900, 904 of 904 and
  162 of 165 reply positions.
- **The expert stream holds its rate at wide blocks** (the wide-verify bundle on dgx4): FC1 costs
  12.2-12.5 µs and FC2 5.8 µs for each extra expert, at 6 to 65 rows, with rows spread evenly
  or piled on three experts. What grows with the width is a per-row cost, about 2 µs a row a
  layer at 65 rows, which the per-row price above includes.
- **Agent traffic gains a little; long copies gain a lot.** At TP4, measured routes (TP3 within
  0.3 points; the bounds' range in brackets):

  | Workload | Exact per request, m = 16, at most 16 drafts | At most 64, growing cap |
  | --- | --- | --- |
  | agent proxy, visible output, c1 | +4.9% (+2.9 to +4.6) | +4.4% |
  | agent proxy, c8 | +2.8% (0.0 to +2.3) | +1.9% |
  | agent proxy, hidden reasoning 1:1 decoded like prose, c1 | +2.2% | +2.0% |
  | reproduce a file, c1 | +115% | +297% |
  | the file with comments added, c1 | +74% | +140% |
  | an edit tool call, c1 | +32% | +47% |

  The earlier +7-17% tokens a step shrinks for three reasons:
  - DSpark already takes 5 tokens a step on copies;
  - a 17-row block reaches twice the experts of a 6-row one;
  - drafts that miss waste rows.
- **At most 64 drafts is a recipe choice to measure, not more code.**
  - **Growing cap:** on agent traffic, 0.5 points below 16 at c1 and 0.9 at c8; on long
    copies, +297% against +115% reproducing a file.
  - **The stale allocation:** the engine lays out a step's rows one step before it knows
    whether the copy continued. That costs the 16-draft cap 0.8 points and the 64-draft cap
    1.6; without it, 64 beats 16 at c1 (+6.0% against +5.7%).
  - **Every proposal verified at its full length:** 64 loses at c8 (−3 to −21%).
- **Minimum match 8 loses at eight streams** (−0.6 to −1.2%, −8% under independent routes):
  too many short, wrong copies. Matches of 16 and 24 are within half a point of each other.
- **A global n-gram table across requests adds about 1.8 points:** exact first, the shared
  table elsewhere (m = 24, at most 16) reaches +6.7% at TP4 c1 and +4.2% at c8. The table alone
  reaches +6.3% and +3.6%. Three caveats:
  - the table is shared across requests, so how fast a reply comes back can reveal what other
    requests contained;
  - the simulated table is exact and never evicts, so it is an upper bound;
  - the proxy is one developer's sessions over the same repositories.

  Walked in session order instead of token time, the table scored +9%: a parent session's later
  turns predicted its subagents' output.
- **Go:** exact per-request copies, minimum match 16, at most 16 drafts, the cap growing from 8,
  beside DSpark. Its build is designed below and needs:
  - a verification lookahead separate from DSpark's 5-draft block;
  - spec-as-decode rows to 17;
  - two more graphs a shape;
  - two TileLang kernels.

  The lookahead is a recipe value, so at most 64 drafts is a lab comparison against 16 on the
  same build (copy probe and agent replay), not a second build. A shared table is a second
  phase, scoped to one tenant, and only if a replay of many unrelated sessions (AgentX, #574)
  keeps its gain.

## 1. What a verified row costs

### The model

[cost.py](cost.py) extends the single-stream model of the 100 tok/s study (draft #598,
`experiments/2026-10-26-c1-100tps`) to wide blocks, TP3 and eight streams:

```
step = overhead + (dense + drafter routed + 40 × U × expert) / 266.6 GB/s
       + (rows + graph padding − the suite's rows) × row price
```

- **U** is the distinct routed experts a target layer reads. One request's r consecutive rows
  reach N(r). Eight independent requests reach `384 × (1 − Π(1 − N_i/384))`.
- **dense and expert** are the bytes a rank reads, from #598's floor.py shapes. At TP3 the
  sharded weights are 4/3 as wide; Q-A/KV, the router, mHC `fn`, the compressors and the
  Markov head stay replicated. A routed expert is 4.57 MB at TP4 and 6.09 MB at TP3, against
  3.06 and 3.82 GB of dense weights a step.
- **The row price** is the engine grid's step per row that reaches no new expert (identical
  dummy tokens), and it holds to 96 rows in the 1025 grid:
  - 0.366 ms at TP4 (the r6v static table, as in #598);
  - 0.435 ms at TP3 (1022 grid, 1x2 to 1x6 rows).

  Every row beyond the release suite's (4.07 at c1, 8 × 2.97 at c8) pays it, routed or not.
  Graph padding pays it too: the capture ladder rounds 17 rows up to 20, 33 to 40, 49 to 56
  and 65 to 72.
- **The overhead** is solved from each shape's release suite. At TP4 c1 that is 9.14 ms
  (#598's 8.70, plus the per-row cost of today's 4.07 rows). The c8 solves use the 1025 TP4
  and 1022 TP3 suites at 85.7 and 109.2 ms a step.
- **N(r)** comes from the reference routes measured here (below) and two bounds:
  - **measured:** each copy-probe category is priced by its own case. The agent proxy is
    priced by the edit tool call, the case nearest its tool-command and edit copies; the
    copied file is a cross-check.
  - **ordinary text:** json-count's eight-token reference routes, extended at 2.4 experts a
    row (#598's extrapolation): 24, 51, 90, 128 and 166 experts at 6, 17, 33, 49 and 65 rows;
  - **independent:** uniform routing, 35, 90, 156, 206 and 246.

  DSpark's prose steps keep the suite's calibration whatever the source.

### Measured: the expert stream at wide blocks ([bundles/wide-verify](bundles/wide-verify))

Wide blocks put more rows on each expert, and a copied span's repeated tokens pile rows onto a
few experts. The question is whether FC1 and FC2 still stream each expert's weights once at the
rate #598 measured at three and six rows.

The bundle uses the expert-rate bundle's method: TP4 rank shapes, the production plan with coded
scales, cold weights and an L2 flush between replays. It times FC1 and FC2 at 6, 17, 33, 49 and
65 rows against the number of distinct experts, in two layouts: rows spread evenly, and every row
on three hot experts plus the rest spread. The run is
[runs/wide-verify-dgx4.txt](runs/wide-verify-dgx4.txt), 33 s on dgx4 at 21:42 CET.

Fits, time = fixed + per expert × distinct experts (µs):

| Rows | FC1 even | FC1 skewed | FC2 even | FC2 skewed |
| ---: | --- | --- | --- | --- |
| 6 | 14.3 + 12.36 | | 15.5 + 5.90 | |
| 17 | 21.8 + 12.26 | 13.9 + 12.48 | 34.7 + 5.81 | 33.8 + 5.79 |
| 33 | 25.2 + 12.25 | 18.4 + 12.42 | 60.6 + 5.81 | 65.0 + 5.81 |
| 49 | 32.1 + 12.24 | 26.7 + 12.31 | 86.2 + 5.82 | 89.2 + 5.82 |
| 65 | 41.3 + 12.30 | 38.4 + 12.28 | 113.8 + 5.81 | 123.1 + 5.81 |

- **The per-expert price does not move** with the width or the skew.
- **The fixed part grows with rows:** FC2 by 1.7 µs a row and FC1 by 0.5 µs. That is about
  0.09 ms a row over 40 layers, part of the 0.37 ms row price.
- FC1's per-expert time here is 8% above #598's run (11.41 µs at six rows, the same bundle
  method). The model keeps #598's 266.6 GB/s, which matched the traced serving FC1. A slower
  stream would price both arms alike.

### The table

Each cell is measured routes (the copied file) / ordinary-text sharing / independent routes:

| Rows (graph size) | Experts a layer | TP4 step, ms | TP4 ms a token if all accepted | TP3 step, ms | TP3 ms a token |
| --- | --- | --- | --- | --- | --- |
| 6 (6) | 24.0 / 24.1 / 34.6 | 38.2 / 38.3 / 45.5 | 6.4 / 6.4 / 7.6 | 48.2 / 48.3 / 57.9 | 8.0 / 8.1 / 9.7 |
| 17 (20) | 47.4 / 51.2 / 90.2 | 59.4 / 62.0 / 88.7 | 3.5 / 3.6 / 5.2 | 75.7 / 79.2 / 114.8 | 4.5 / 4.7 / 6.8 |
| 33 (40) | 69.2 / 89.6 / 155.6 | 81.7 / 95.7 / 140.9 | 2.5 / 2.9 / 4.3 | 104.3 / 123.0 / 183.3 | 3.2 / 3.7 / 5.5 |
| 49 (56) | 85.3 / 128.0 / 206.5 | 98.5 / 127.8 / 181.6 | 2.0 / 2.6 / 3.7 | 126.0 / 165.0 / 236.7 | 2.6 / 3.4 / 4.8 |
| 65 (72) | 98.3 / 166.4 / 246.0 | 113.3 / 160.0 / 214.6 | 1.7 / 2.5 / 3.3 | 144.8 / 207.0 / 279.8 | 2.2 / 3.2 / 4.3 |

TP3's 49 and 65 rows are past its graph ladder (48) today and assume the extended ladder of the
design.

**Break-even.** On copied text DSpark's block takes 5 tokens for 38.3 ms (TP4). For a wide block
to pay against DSpark it must accept, on average (measured routes, then the bounds):

| Block | Tokens needed |
| ---: | --- |
| 17 rows | 7.8 (8.1-11.6) |
| 33 rows | 10.7 (12.5-18.4) |
| 65 rows | 14.8 (20.9-28.0) |

So the width pays on copies longer than about a quarter to a half of it. That is what the
variants decide.

### Measured: do copied rows share experts?

Two different sharings are possible:
- **between the rows of the block,** which sets U;
- **between a copy and its source.** A source's experts were read in an earlier step and are
  not resident at c1 (24 MiB L2), so this one saves nothing.

Before this run, json-count's routes (eight tokens of JSON) were the only reference routes, and
the copy probe kept no token ids.

**The method.** [reference/routes_probe.py](reference/routes_probe.py) runs DeepSeek's reference
forward on CPU with json-count's loader and kernels. It does one teacher-forced prefill per
conversation, the prompt plus the reply. It records all 40 layers' top-6 routes at every
position, and the reference's greedy token at each reply position, which checks how close the
forced reply is to the model's own. The conversations are built by [cases.py](cases.py) from
copy_probe.py's prompts, with 1,000-token documents:

| Case | Prompt + reply tokens | Greedy agrees with the forced reply | Seconds |
| --- | --- | --- | ---: |
| a file reproduced | 922 + 900 | 900 / 900 | 698 |
| prose (this repository's methodology, not the model's own reply) | 17 + 1,001 | 523 / 1,001 | 560 |
| an edit tool call (old_string: the first function) | 961 + 165 | 162 / 165 | 626 |
| the file with comments added | 940 + 904 | 904 / 904 | 711 |

So the copies are what the model would write.
[reference/routes_summary.py](reference/routes_summary.py) reduces the routes (7 MB, kept on
dgx4) to [runs/routes-summary.json](runs/routes-summary.json).

**Distinct experts a layer reached by r consecutive reply rows** (mean over layers and windows):

| Rows | 1 | 2 | 6 | 17 | 33 | 49 | 65 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| a file reproduced | 6 | 10.5 | 24.0 | 47.4 | 69.2 | 85.3 | 98.3 |
| the file with comments added | 6 | 10.6 | 24.4 | 47.9 | 69.7 | 85.6 | 98.6 |
| an edit tool call | 6 | 10.5 | 23.9 | 48.0 | 72.0 | 89.9 | 105.2 |
| prose | 6 | 10.1 | 22.2 | 44.4 | 65.7 | 80.8 | 93.4 |
| ordinary-text bound (json-count + 2.4 a row) | 6 | 10.3 | 24.1 | 51.2 | 89.6 | 128.0 | 166.4 |
| independent routes | 6 | 11.9 | 34.6 | 90.2 | 155.6 | 206.5 | 246.0 |

- **Sharing is a property of one request's consecutive rows, not of code.** Prose shares as
  much as copied code. Every case agrees with json-count's routes to 8 rows; the linear
  extension past them was too steep.
- **The spread between layers is wide.** At 65 rows the copied file's layers reach 69 to 138
  experts (median 99), prose 51 to 153. The model prices the layers' mean.
- **A copied row and its source share 38-48% of their experts** (891 copied rows: 48%; 885
  edit rows: 43%; 38 tool rows: 38%; per layer 13-95%). It saves nothing at c1, but it bounds
  how well a copy's experts could be known ahead of its step.

**The run.** [run_routes.sh](reference/run_routes.sh) ran it in the serving image on dgx4's
cores 0-4 and 10-14:
- nice 10, a 16 GiB cap (1.4-2.7 GiB used), no network, the hub read-only, reads throttled to
  800 MB/s: 848 GiB read in 43 minutes;
- holding the kernel lab's lock (`/tmp/spark-lab-gpu.lock`), so no GPU bundle timed its
  kernels beside it.

## 2. What each variant nets

### Variants and workloads

[variants.py](variants.py) walks token streams step by step. The families:

| Family | Proposal |
| --- | --- |
| **exact** (per request) | the latest earlier occurrence, in the request's own context, of the longest of the levels m, 2m, 4m, ... (at most 256) that ends the context; its continuation, up to what exists |
| **request** (chained) | a table from each m-gram to the token after its latest occurrence, extended one token at a time (consecutive drafts may come from different places), per request |
| **global** (chained) | the same table shared across every request, in the order the tokens were written |
| **both** | exact where it matches, global elsewhere |

Policies:
- `full`: verify every proposal at min(k, its length);
- `grow`: the cap starts at 8 and doubles after each fully accepted copy step, up to k, and
  returns to 8 after a rejection;
- `-nolag`: either policy without the stale allocation.

Every combination of m = 8, 16, 24 and k = 16, 32, 64 is walked.

What the walk charges:
- **the allocation lag.** The engine lays out each step before the previous step's tokens are
  known (asynchronous scheduling). So a step's rows come from the state one step earlier,
  assuming that step is accepted in full. Rows the real proposal does not use are paid as dead
  rows. After a DSpark step the allocation is DSpark's 6 rows, so a copy's first step verifies
  at most 5 drafts. The `-nolag` policies remove the lag and bound what it costs.
- **misses.** A rejected draft's row and every row after it are paid in full.
- **DSpark where no copy is taken:** min(5, run + 1) tokens a step inside an exact copy (the
  copy probe's 4.96-5.41), the release suite's 2.25 elsewhere.

Workloads:
- **The copy probe's categories** ([copy_cases.py](copy_cases.py),
  [runs/copy-cases.json](runs/copy-cases.json)). There are 12 conversations a category, built
  like copy_probe.py's ([cases.py](cases.py)) in the checkpoint's own chat format with the
  checkpoint's tokenizer. The reply is the ideal copy, since the 2026-10-23 run kept no ids.
  The walk reproduces that run's suffix drafter on the reproduced file (15.87 against 15.9
  tokens a step) and DSpark alone on it (4.94 against 4.96). The ideal edit is harsher than the
  model's replies were (10.9 against 14.9). DSpark steps are priced at the probe's measured
  rates (4.96, 5.10, 5.41 tokens a step).
- **The agent proxy** ([runs/agent-proxy.json](runs/agent-proxy.json), counts only), the
  2026-10-18 proxy rebuilt on 2026-10-25:
  - 709 local coding-agent session files, 785 context streams, 84,690 assistant turns;
  - 31.0M output tokens over 84.1M context tokens, in DS4.1's prompt format with DSML tool
    calls;
  - the corpus is cut at 20:00 CET 2026-10-25 (nothing written later is read), so it is fixed;
    the global table's entries become visible at the time their token was written;
  - by kind of output: tool commands 53%, tool-call markup 16%, file edits and writes 14%,
    prose 11%, other tools 7%.

  The transcripts hide reasoning, so a second reading adds hidden reasoning tokens 1:1,
  decoded as prose (the 19% bound of 2026-10-18).

At c8, each step draws eight request-steps from a walk's step distribution (3,000 steps a
point). At c8 DSpark's prose steps take the suite's 2.09 tokens a request.

### The copy probe's categories (measured routes / ordinary-text sharing / independent routes)

Each category is priced by its own measured case.

DSpark alone, tok/s (measured / ordinary-text / independent):

| Category | TP4 c1 | TP4 c8 |
| --- | --- | --- |
| reproduce a file | 130 / 129 / 109 | 273 / 272 / 222 |
| comments added | 132 / 133 / 112 | 278 / 280 / 228 |
| edit tool call | 142 / 141 / 119 | 298 / 297 / 242 |

| Variant | file: TP4 c1 | TP4 c8 | TP3 c1 | edit: TP4 c1 | TP4 c8 | tool call: TP4 c1 | TP4 c8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exact m8 k16 full | +116% / +108% / +73% | +101% / +95% / +97% | +114% / +105% / +70% | +82% / +75% / +50% | +69% / +64% / +63% | +38% / +36% / +24% | +30% / +28% / +25% |
| exact m16 k16 grow | +115% / +106% / +73% | +99% / +93% / +95% | +113% / +104% / +70% | +74% / +69% / +47% | +62% / +57% / +55% | +32% / +31% / +21% | +26% / +24% / +21% |
| exact m24 k16 grow | +114% / +105% / +72% | +98% / +92% / +94% | +112% / +103% / +69% | +71% / +66% / +46% | +60% / +55% / +53% | +30% / +28% / +19% | +24% / +22% / +20% |
| exact m16 k64 grow | +297% / +191% / +158% | +267% / +242% / +312% | +292% / +184% / +152% | +140% / +101% / +82% | +114% / +90% / +108% | +47% / +37% / +29% | +37% / +28% / +31% |
| exact m16 k64 full | +307% / +195% / +163% | +277% / +252% / +326% | +302% / +188% / +157% | +141% / +91% / +75% | +114% / +87% / +111% | +52% / +37% / +30% | +40% / +28% / +34% |
| request m16 k16 grow (chained) | +106% / +98% / +66% | +91% / +85% / +87% | +104% / +96% / +64% | +71% / +66% / +44% | +59% / +55% / +53% | +32% / +31% / +21% | +26% / +24% / +21% |

Every exact variant pays on copies. Width pays in proportion to the copy's length, and more with
the measured routes, under which a 65-row block reaches 98 experts, not 166-246.

The chained table is no better than exact here, and much worse with full verification at 64
(+46% on the file against +195%, ordinary-text sharing). It follows the latest occurrence of a
short n-gram, which in code is often the wrong place: `return self.` recurs everywhere. The exact match disambiguates
with up to 256 tokens of context.

### The agent proxy (measured routes / ordinary-text sharing / independent routes)

Copy rows are priced by the edit tool call's measured routes. With the copied file's routes
instead, the go variant is +5.0% at TP4 c1 and +3.0% at c8.

DSpark alone (modelled), tok/s (measured / ordinary-text / independent):

| | TP4 c1 | TP4 c8 | TP3 c1 | TP3 c8 |
| --- | --- | --- | --- | --- |
| visible output (2.83 tokens a step) | 86.9 / 86.8 / 82.0 | 208.7 / 208.2 / 187.8 | 69.6 / 69.5 / 65.4 | 163.0 / 162.6 / 146.1 |
| with reasoning 1:1 | 80.1 / 80.1 / 78.0 | 201.8 / 201.6 / 190.9 | 64.2 / 64.1 / 62.3 | 157.9 / 157.7 / 149.0 |

Visible output:

| Variant | Tokens a step | Copy share of tokens | Drafts / accepted a copy step | Wasted rows a token | TP4 c1 | TP4 c8 | TP3 c1 | TP3 c8 |
| --- | ---: | ---: | --- | ---: | --- | --- | --- | --- |
| exact m16 k16 grow | 3.06 | 26% | 10.9 / 7.3 | 0.28 | +4.9 / +4.6 / +2.9 | +2.8 / +2.3 / +0.0 | +4.9 / +4.6 / +2.7 | +2.9 / +2.4 / +0.0 |
| exact m16 k16 full | 3.07 | 26% | 12.2 / 7.5 | 0.26 | +4.9 / +4.6 / +2.4 | +3.0 / +2.3 / −0.5 | +4.9 / +4.5 / +2.2 | +3.0 / +2.4 / −0.5 |
| exact m24 k16 grow | 3.02 | 20% | 11.5 / 8.2 | 0.17 | +4.5 / +4.3 / +3.0 | +3.0 / +2.6 / +0.9 | +4.5 / +4.2 / +2.9 | +3.1 / +2.7 / +0.9 |
| exact m24 k16 full | 3.02 | 20% | 12.7 / 8.5 | 0.16 | +4.5 / +4.3 / +2.8 | +3.1 / +2.6 / +0.6 | +4.5 / +4.2 / +2.6 | +3.1 / +2.6 / +0.5 |
| exact m8 k16 grow | 3.14 | 43% | 10.0 / 5.7 | 0.63 | +4.7 / +4.2 / +0.8 | −0.6 / −1.4 / −5.4 | +4.6 / +4.1 / +0.5 | −0.4 / −1.3 / −5.5 |
| exact m8 k16 full | 3.17 | 43% | 11.5 / 5.9 | 0.60 | +4.4 / +3.7 / −0.6 | −1.2 / −2.3 / −7.5 | +4.3 / +3.5 / −1.1 | −1.1 / −2.2 / −7.6 |
| exact m16 k32 grow | 3.08 | 26% | 13.3 / 8.0 | 0.43 | +4.8 / +4.1 / +2.2 | +2.3 / +1.0 / −1.3 | +4.8 / +4.0 / +2.0 | +2.4 / +1.1 / −1.3 |
| exact m24 k32 grow | 3.04 | 20% | 14.4 / 9.2 | 0.29 | +4.5 / +3.9 / +2.5 | +3.1 / +2.1 / +0.3 | +4.4 / +3.8 / +2.4 | +3.2 / +2.1 / +0.3 |
| exact m16 k32 full | 3.12 | 26% | 20.6 / 9.3 | 0.57 | +4.0 / +2.4 / −0.6 | −0.5 / −3.1 / −6.5 | +3.9 / +2.2 / −1.0 | −0.3 / −3.0 / −6.6 |
| exact m16 k64 grow | 3.09 | 26% | 14.4 / 8.2 | 0.56 | +4.4 / +3.4 / +1.6 | +1.9 / +0.1 / −1.8 | +4.4 / +3.3 / +1.5 | +2.1 / +0.3 / −1.7 |
| exact m24 k64 grow | 3.04 | 20% | 15.8 / 9.4 | 0.40 | +4.2 / +3.3 / +2.1 | +1.4 / −0.1 / −1.5 | +4.2 / +3.3 / +2.0 | +1.6 / −0.0 / −1.5 |
| exact m24 k64 full | 3.08 | 20% | 38.6 / 12.4 | 0.77 | +2.3 / −0.9 / −2.5 | −2.8 / −7.8 / −8.9 | +2.2 / −1.1 / −2.9 | −2.5 / −7.8 / −9.0 |
| exact m8 k64 full | 3.27 | 43% | 33.8 / 7.2 | 2.70 | −6.5 / −13.8 / −18.7 | −21.0 / −29.1 / −31.3 | −6.7 / −14.5 / −19.5 | −20.5 / −29.1 / −31.2 |
| request m16 k16 grow | 3.05 | 26% | 10.7 / 7.1 | 0.29 | +4.6 / +4.3 / +2.6 | +2.4 / +1.9 / −0.4 | +4.6 / +4.3 / +2.4 | +2.5 / +2.0 / −0.4 |
| global m24 k16 grow | 3.11 | 25% | 11.1 / 7.5 | 0.25 | +6.3 / +6.0 / +3.9 | +3.6 / +3.0 / +0.3 | +6.2 / +5.9 / +3.6 | +3.6 / +3.1 / +0.3 |
| global m16 k16 grow | 3.16 | 33% | 10.4 / 6.2 | 0.43 | +5.9 / +5.6 / +2.4 | +2.0 / +1.3 / −2.5 | +5.8 / +5.4 / +2.1 | +2.0 / +1.4 / −2.6 |
| global m24 k64 grow | 3.14 | 25% | 14.5 / 8.5 | 0.51 | +5.9 / +5.0 / +2.9 | +3.0 / +1.4 / −0.9 | +5.9 / +4.9 / +2.7 | +3.2 / +1.5 / −0.8 |
| both m24 k16 grow | 3.12 | 25% | 11.3 / 7.8 | 0.23 | +6.7 / +6.4 / +4.3 | +4.2 / +3.7 / +1.0 | +6.6 / +6.3 / +4.1 | +4.2 / +3.7 / +0.9 |
| both m24 k16 full | 3.13 | 25% | 12.9 / 8.1 | 0.21 | +6.6 / +6.3 / +3.7 | +4.2 / +3.6 / +0.3 | +6.5 / +6.1 / +3.4 | +4.3 / +3.6 / +0.2 |
| both m16 k16 grow | 3.17 | 33% | 10.5 / 6.5 | 0.40 | +6.7 / +6.3 / +3.2 | +3.0 / +2.4 / −1.5 | +6.6 / +6.2 / +2.9 | +3.1 / +2.4 / −1.6 |
| both m24 k64 grow | 3.15 | 25% | 15.2 / 9.0 | 0.49 | +6.4 / +5.4 / +3.3 | +3.6 / +1.8 / −0.4 | +6.4 / +5.2 / +3.1 | +3.8 / +1.9 / −0.4 |

Gains are in percent of DSpark alone at the same shape, concurrency and routes. `request` is the
chained table per request; `global` is the chained table shared across requests in token-time
order; `both` is exact where it matches and global elsewhere, both at the same m.

**Without the allocation lag** (the `-nolag` bound):

| Variant | TP4 c1 | TP4 c8 |
| --- | --- | --- |
| exact m16 k16 grow | +5.7 / +5.4 / +3.4 | +4.5 / +3.9 / +1.2 |
| exact m24 k16 grow | +5.1 / +4.9 / +3.6 | +3.8 / +3.4 / +1.5 |
| exact m16 k64 grow | +6.0 / +5.0 / +3.0 | +3.7 / +1.9 / −0.7 |
| both m24 k16 grow | +7.5 / +7.2 / +5.0 | +5.7 / +5.2 / +2.2 |

With the measured routes, the lag is worth 0.8 points at 16 drafts, 1.6 at 64 and 0.8 for
exact-then-global. Without it, 64 drafts beat 16 at c1 (+6.0 against +5.7) but not at c8.
Synchronous layout of copy steps would recover it at the price of the asynchronous scheduler's
overlap on those steps, which this record does not bound.

**With hidden reasoning 1:1 decoded as prose:**

| Variant | TP4 c1 | TP4 c8 |
| --- | --- | --- |
| exact m16 k16 grow | +2.2 / +2.1 / +1.4 | +1.5 / +1.2 / −0.4 |
| exact m24 k16 grow | +2.0 / +1.9 / +1.4 | +1.7 / +1.5 / +0.4 |
| exact m16 k64 grow | +2.0 / +1.5 / +0.8 | +0.7 / −0.1 / −1.5 |
| both m24 k16 grow | +3.0 / +2.8 / +2.0 | +2.2 / +1.9 / +0.2 |

Reasoning is unmeasured in both directions:
- if it copies like prose, it halves the gain;
- if the final answer repeats code drafted in the reasoning, which a reasoning model does, that
  is a copy source this proxy cannot see.

The post-build copy probe with thinking on measures it.

### Verdict per variant

| Variant | Verdict | Why |
| --- | --- | --- |
| exact, m 16, at most 16, grow | **go** | measured routes: +4.9% at TP4 c1, +2.8% at c8, TP3 alike; never below DSpark under either bound (+2.9 / 0.0 at worst); +115% / +74% / +32% on the probe's copies |
| exact, m 24, at most 16 | go (alternative) | 0.4 points behind at c1, 0.2 ahead at c8, fewer wasted rows; settle on the post-build agent replay |
| exact, at most 32 or 64, grow | a recipe comparison after the build | against 16 on agent traffic: −0.1 (32) and −0.5 (64) points at c1, −0.5 and −0.9 at c8; reproducing a file +297% against +115%. Without the allocation lag, 64 beats 16 at c1. The lookahead is a recipe value, so the lab compares them on one build |
| exact, at most 32 or 64, full | no-go | wasted rows: at 64, −6.5% (m 8) to +2.3% (m 24) at c1 and −2.8 to −21% at c8; at 32, 0.2-4.1 points below 16 (growing cap, same m) at c1 and 1.8-8.1 at c8 |
| minimum match 8 | no-go | −0.6 to −1.2% at c8 (to −8% under independent routes): too many short wrong copies |
| request-chained table | no-go | never better than exact (+4.6% against +4.9%), much worse on long copies |
| global table (chained, shared across requests) | second phase, measured first | +1.8 points at c1 and +1.4 at c8 over exact alone, as exact-then-global at m 24; an upper bound (exact, unbounded, one developer's sessions); a cross-request side channel unless scoped to one tenant |

## 3. The design

What follows is the build for the variant that clears the cost: copies proposed per request,
exact, from the request's own history, minimum match 16, at most 16 drafts with the cap growing
from 8 (`L = 16`). The lookahead `L` is a recipe value. The same build at `L = 64` is the lab's
comparison: it costs half a point on agent traffic, gains 2.6 times on file copies, and wins at
c1 without the allocation lag. DSpark stays the drafter everywhere else. Nothing is
built; the vLLM work is a branch over `tilelang-ports` 26382f5cf (the frozen candidate 6 tree),
for the release after spark-ds41f-1025.

### The proposer (`vllm/v1/worker/gpu/spec_decode/dspark/`)

DSpark keeps running every step: its context precompute must cover every accepted row, and its
draft is the fallback whenever no copy matches. After the DSpark draft step, inside the same
`propose()` call, two new TileLang kernels run per decoding request:

1. **`copy_match`** scans the request's token history for the latest earlier occurrence of
   the longest of the levels m, 2m, 4m, ... (at most 256) that ends the context. The history
   is already on the device: `RequestState.all_token_ids` (UVA, `[max_num_seqs, max_model_len]`
   int32), which `post_update` extends every step. The key's last token is draft zero (the
   provisional token, below). It is a stateless scan:
   - grid (requests, history chunks); each CTA compares the windows ending in its chunk with
     the key, longest level first, and `atomicMax`es the end position per level (an
     order-independent reduction, so the result is deterministic);
   - it reads 4 bytes a context token, 0.4 MB at a 100K agent context and 4 MB at 1M:
     about 2 and 15 µs at the stream rate, against a 30-37 ms step (more if the UVA history
     reads below the device rate; commit 4's benchmark measures it);
   - there is no index to keep: no insertion at prefill, nothing to roll back when rows are
     rejected or a sunk draft zero is replaced, and no memory beyond the history the runner
     already holds.
2. **`copy_select`** picks the longest level with a source and offers
   `d = min(cap, position - source, allocated rows - 1)` drafts, the tokens that follow the
   source, gathered from the history into the draft buffer. With a match it replaces DSpark's
   drafts for that request; without one DSpark's five stand. It keeps the request's copy
   state on the device: the cap (8, doubled after every copy step whose drafts were all
   accepted, up to 16 in the base design and 64 with the long-copy extension, back to 8 after a
   rejection), the copy survival estimate (below), and the kind and length of the proposal.
   The runner returns that state with each step's output, the way adaptive verification
   returns its confidences, and the scheduler lays out the next step from it.

Copy drafts are point proposals (one-hot), which the verifier already handles: the recipe's
DSpark drafts are greedy (`draft_sample_method: greedy`), one-hot at every position.

### Variable-length verification

Today one number, `num_speculative_tokens` (5), sizes both the drafter's block and every
verification buffer: about 290 uses in the DS4.1 serving path. The design splits it into
two:

- the drafter's block, still 5: DSpark's query block, its Markov head, its sliding-window
  index width (window + block), its graphs;
- the verification lookahead `L` (16 in the base design, 64 with the extension), which
  sizes everything a verified row touches:
  - **the scheduler:** a copying request's scheduled drafts come from its reported copy
    state instead of the uniform `1 + num_spec_tokens` padding. KV slots are allocated for
    `1 + L` rows (`num_lookahead_tokens`), and sliding-window block retention covers rows
    `L` ahead.
  - **the runner's buffers:** `RequestState.draft_tokens` `[max_num_seqs, L]`; the
    confidence, capacity and offset buffers of adaptive verification; the rejection and
    greedy-verification kernels; the resampled-token kernel; the logits chunk
    (`max_total_logits`); the sink detector's rows; the grammar bitmask rows of
    structured-output requests.
  - **the rings that rejected rows write before acceptance is known:** `CompressorStateCache`
    (`block_size = max(8, next power of two above L + 1)`: 32 at `L = 16`, 128 at 64) and the
    compressor's `CircularBufferSpec` ring (it must exceed the ratio by the lookahead).
  - **the decode classification:** `reorder_batch_threshold = 1 + L`, so every row of a
    decoding request stays on the decode plan and the decode index mode, whatever its width.
    Today a request with more than 6 rows is a prefill and its step a mixed one: eager, the
    extend plan.
  - **the decode capacity** of the layers' workspaces: `max_num_seqs × 6 + (L − 5)` rows.

**Temperature 0 stays bit-identical** if every row's logits are the same bits whatever the
block's width, because greedy verification then commits exactly the target's own greedy tokens.
The decode chain is built that way already:
- sparse MLA's decode, split and extend kernels share one arithmetic ("choosing one by row
  count never changes results");
- mixed steps keep decode rows on the decode plan;
- the router, MoE route-output rows and mHC are row-invariant from the batch-invariance and
  determinism work (r6d onward).

Wider blocks only add widths nobody has tested. They are tested, not assumed: a row-invariance
test per decode-chain kernel at widths 1, 6, 17, 33, 49, 65 and the ladder's top, with
bit-equality against the 1-row result.

**Sampled verification stays lossless.** The recipe's block verification already verifies
one-hot drafts at temperature 1. A copy draft is one more one-hot proposal, made from committed
context only, so it is independent of the verifier's draws. The test is the rejection sampler's
existing distribution test, at `L` rows with fixed point proposals.

### CUDA graphs

Verification steps stay FULL graphs, with variable-length decode, as adaptive verification
runs them now. The ladder (scripts/recipe.py `capture_sizes`) ends at
`max_num_seqs × (1 + drafts) + (L − drafts)` instead of `max_num_seqs × (1 + drafts)`:

| Shape | Ladder top today | L = 16 | L = 64 |
| --- | ---: | ---: | ---: |
| tp4-ring (16 streams) | 96 | 107 → 112 (2 more graphs) | 155 → 160 (8 more) |
| tp3-triangle (8 streams) | 48 | 59 → 64 (2 more) | 107 → 112 (8 more) |

One wide block a step fits; the scheduler admits a second copying request at its full width
only if the step still fits the ladder, and at DSpark's 6 rows otherwise. The graphs' memory
counts against TP3's 5 GiB startup guard and is measured at boot (lab specs below). The
drafter's graphs keep their shapes; its context precompute takes up to `1 + L` rows a
request.

### Sink rejection and draft zero

With sink rejection on (the recipe's default), each step's last sampled token is provisional:
it leads the next step as draft zero. Copy drafts follow it exactly as DSpark's do:

- the copy key's last token is draft zero;
- copy rows are rows 1..d of the request, behind row 0;
- a sunk copy row cuts verification at that row, like a sunk drafter row;
- a sunk draft zero rejects the whole block (its alternate changes the context), wasting the
  block once, and the copy state falls back to a cap of 8.

The scheduler's draft-zero hook counts draft zero as row 0 of the allocation.

### The adaptive budget and dead rows

The budget keeps ranking DSpark rows by survival under today's rule (row cost 0.035). Copy rows
enter the same ranking with survival `s^j` at draft `j`: `s` is the request's copy survival
estimate, the share of its recent copy rows that were accepted, kept on the device. At c1 a
16-draft copy at `s = 0.95` scores `11.6 / (1 + 0.035 × 17) ≈ 7.3` expected tokens a unit of
step against DSpark's `2.27 / (1 + 0.035 × 4.07) ≈ 2.0`, so it is admitted. At eight streams
it competes for rows with the others' drafts, which is the point. The dead-row cut (survival
below 0.2) applies to copy rows too, so a long block's tail past `s^j < 0.2` (past draft 31 at
`s = 0.95`) skips the routed experts.

### Phase 2: a shared table, scoped to one tenant

Only if the replay of many unrelated sessions keeps the +1.8 points. The table sits beside
the exact match, which stays first:
- **the table:** a device table of 24-gram fingerprints (32 bits, so a collision misses rather
  than drafts) to the next token, bounded (8M entries, 64 MB), with the oldest entries
  overwritten;
- **inserts:** each step's committed tokens after verification, and a prefill's uncached
  tokens, so an entry exists only once its token is committed;
- **lookups:** `copy_select` chains through the table, one token at a time, when the exact
  match finds nothing;
- **scope:** one tenant's requests (one API key), never across tenants. A shared table turns
  draft acceptance into a timing channel between users.

### Kernels

Every new kernel is TileLang, declared and prepared at startup like the rest of
`tilelang_inputs`, with no JIT on the serving path and no non-TileLang path:

- new: `copy_match`, `copy_select` (and phase 2's table insert);
- widened to `L`: the rejection, greedy-verification and resampled-token kernels;
  `assign_draft_budget`, `adaptive_offsets`, `gather_hidden_rows` and `post_update`;
  the dead-row marking.

There is no copy-off arm in the recipe: the hybrid is the proposer. A DSpark-only arm exists
only as a lab measurement arm, as verify-all does in #598.

### Patches and tests

A vLLM branch `copy-drafts` over `tilelang-ports` 26382f5cf, one commit each, in this order:

| # | Commit | Tests |
| --- | --- | --- |
| 1 | `[Spec] Verification lookahead apart from the drafter's block`: `speculative_config.copy_drafts` (`min_match`, `max_drafts`); the lookahead sizes the scheduler's KV lookahead, `RequestState.draft_tokens`, the adaptive-verification and rejection buffers, the logits chunk, the sink rows and the grammar bitmask rows | with `copy_drafts` absent, every buffer, ring, threshold and graph size is unchanged (an equality test over the resolved sizes); with `max_drafts` 16 and 64, each is the lookahead's |
| 2 | `[DS4.1] Rings and decode capacity for the lookahead`: `CompressorStateCache`, the compressor `CircularBufferSpec` ring, sliding-window block retention, the layers' decode capacity | ring capacities against the lookahead; a rejected wide block leaves the next step's ring reads intact (compressor state replay test at 17 and 65 rows) |
| 3 | `[DS4.1] Wide verification rows stay on the decode plan`: `reorder_batch_threshold = 1 + L` for the V4.1 metadata builders, the sparse-SWA builder and CED | per-kernel row invariance at widths 1, 6, 17, 33, 49, 65 and the ladder top, bit-equal to the 1-row result: sparse MLA (decode, split, extend), indexer decode mode, compressor, mHC, FP8 GEMMs, routed MoE, router, Engram, the vocabulary heads |
| 4 | `[DS4.1] copy_match and copy_select (TileLang)` | against a numpy reference (variants.py's exact proposal): longest level, latest occurrence, history ends, draft zero in the key, no source past the current position; recorded agent and copy-probe streams; benchmark at 4K, 64K and 1M contexts |
| 5 | `[DS4.1] Wide verification in the rejection, greedy-verification and resampled-token kernels (TileLang)` | greedy: identical commits to the 5-row kernel on every prefix; block verification with one-hot drafts at `L` rows: the distribution test (chi-square against the target distribution) |
| 6 | `[Spec] DSpark proposer with copy drafts`: the copy state on the device, copy confidences into the budget, the dead-row cut on copy rows, the D2H copy-state mirror | the proposer's drafts equal variants.py's `walk` on recorded streams (the same proposals, the same caps); a copy that breaks resets the cap; a sunk draft zero rejects the block and resets the cap |
| 7 | `[Scheduler] Per-request draft allocation from the copy state` | allocation follows the reported state one step late; KV slots for `1 + alloc`; preemption and resume keep the state; a second wide request falls back to 6 rows when the ladder is full |
| 8 | `[DS4.1] Graph ladder to the lookahead` (and spark-ds41f `scripts/recipe.py`: the ladder ends at `max_num_seqs × (1 + drafts) + (L − drafts)`) | the recipe's capture checks; boot memory on both shapes |

The release tests (`docs/release-benchmark.md`) gain one determinism case: temperature-0
text of the hybrid identical to DSpark alone on the copy probe's prompts, the release suite's
prompts and an agent replay, at c1 and c8, and across a restart.

### Lab specs

Prepared now, runnable without a build (both plan cleanly with `scripts/lab.py run --dry-run`
once this record is on `origin/main`):

1. **[wide-verify.json](wide-verify.json):** the bundle as a cluster `kernel` job on dgx2 and
   dgx4, a repeat of the dgx4 run on another node.
2. **[wide-rows.json](wide-rows.json):** [wide_rows_probe.py](wide_rows_probe.py) on
   spark-ds41f-1025 at TP4.
   - A cached 4,096-token prefix plus R new tokens is one step of R rows of one request. Its
     time to first token at R = 1, 6, 17, 33, 49 and 65 is measured for three kinds of tail:
     copied from the prefix, fresh text, and random ids.
   - Copied against random at equal R is the routing's sharing in milliseconds. It checks the
     CPU reference's N(r) (98 experts at 65 rows against random's ~246) on the serving engine,
     at production's arithmetic and a 4K context.
   - It runs on the prefill plan and piecewise graphs, so it prices the experts and the rows,
     not the decode plan's graphs.
   - About 10 minutes.

After the build (written once the branch exists; each needs the cluster):

3. **copy-kernels:** a dgx4 `kernel-local` bundle for commits 3-5's row-invariance and kernel
   tests and the copy kernels' benchmark.
4. **hybrid-gate:** the determinism case above, on both shapes.
5. **hybrid-measure:** arms spark-ds41f-1025, the hybrid at `L = 16`, the hybrid at `L = 64`
   (the same build, only the recipe's lookahead and ladder differ), spark-ds41f-1025 again:
   - the single-stream suite (the suite should not move);
   - the copy probe with thinking off and on (the reasoning copies this record could not
     see);
   - an AgentX replay (#574) at its single-session and parallel-subagent points;
   - TP3 the same.

## Runs

1. **wide-verify on dgx4** (`kernel-local`): commit 6c62d32b, spark-ds41f-1025's image
   `spark-ds41f:19f2c20ed4d6-next-6` (`sha256:82b071b57485`), 21:42 CET, **pass** in 33 s.
   - The checkout is `~/spark-lab/agents/suffix-drafter/spark-ds41f`. Its output is
     `~/spark-lab/agents/suffix-drafter/out/wide-verify/` on dgx4, with receipts
     [runs/wide-verify-dgx4.txt](runs/wide-verify-dgx4.txt) and
     [.verdict.json](runs/wide-verify-dgx4.verdict.json).
   - The node had no `~/spark-request.json`.
   - Other sessions' bundles took the lab lock before and after this one.

   ```sh
   python3 scripts/lab.py kernel-local experiments/2026-10-26-suffix-drafter/bundles/wide-verify \
     --out ~/spark-lab/agents/suffix-drafter/out/wide-verify
   ```
2. **The walks** (on the laptop, Python 3.12 with numpy 2.5 and tokenizers 0.23):
   - `copy_cases.py` over the checkpoint's tokenizer and `encoding/` (a copy of the snapshot's
     files): seconds.

     ```sh
     python3 -I experiments/2026-10-26-suffix-drafter/copy_cases.py SNAPSHOT \
       experiments/2026-10-26-suffix-drafter/runs/copy-cases.json
     ```
   - The agent proxy: about 7 minutes on 14 processes, 5 of them for the global table (peak
     about 9 GB resident). It runs from a private adapter that is not in this repository: it
     reads the transcripts and writes only the step histograms and token counts in
     runs/agent-proxy.json.
   - Every proxy walk, in order:
     1. 55 variants without the DSpark-alone baseline: unusable, rerun.
     2. With the baseline.
     3. With the `-nolag` policies and the `both` family.
     4. The global families alone, in token-time order: they did not merge, because the live
        corpus had grown between walks.
     5. Every family on the corpus cut at 20:00 CET, in token-time order: the run recorded
        here.

     Walks 1-3 ordered the global table by session start, which let a parent session's later
     turns predict its subagents' output (+9% at c1). Exact and chained per-request results
     agree within 0.1 point across walks 2, 3 and 5.
3. **The route probe on dgx4's CPU:** commit 105a3247, 22:19-23:03 CET, exit 0, all four cases.
   The coordinating session approved it at 22:15 CET, with the lab lock held throughout and
   23:30 CET as the last case start. The lock was free when asked; the run took 43 minutes and
   read 848 GiB.
   - The raw routes are `~/spark-lab/agents/suffix-drafter/out/routes.json` on dgx4 (7 MB, not
     committed), with its log and status beside it.
   - The node had no `~/spark-request.json`.

   ```sh
   bash experiments/2026-10-26-suffix-drafter/reference/run_routes.sh \
     ~/spark-lab/agents/suffix-drafter/out 2026-10-25T22:30:00Z
   python3 -I experiments/2026-10-26-suffix-drafter/reference/routes_summary.py routes.json \
     experiments/2026-10-26-suffix-drafter/runs/routes-summary.json
   ```
4. **`cost.py` and `report.py`** (standard library only; the c8 draws are seeded, so a rerun
   writes the same [results.json](results.json)):

   ```sh
   python3 -I experiments/2026-10-26-suffix-drafter/cost.py --samples 3000
   python3 -I experiments/2026-10-26-suffix-drafter/report.py --readme
   ```

## Caveats

- **The measured routes are four conversations** at a 1,000-token document, routed by the
  CPU reference (whose logprobs sit within a few nats of production's; json-count). Agent
  copies of shell commands and markup are priced by the edit tool call's routes. Sharing could
  differ at 64-100K contexts. The bounds stay in every table.
- **The models are models.** The c1 model is #598's, calibrated on the 1022 single stream. Its
  wide-block price adds the grid's per-row step and the bundle's per-expert time, but no
  serving step has run 17-65 rows of one request. wide-rows.json measures that on the cluster,
  on the prefill plan.
- **The c8 draws assume independent requests.** A burst of copying requests (subagents writing
  files together) is worse than the draws for wide caps.
- **DSpark's rate inside copies is the copy probe's.** In the agent proxy it is applied wherever
  the exact m = 8 continuation is right, which may overstate DSpark on short copies and so
  understate the hybrid.
- **The agent proxy is one developer's coding sessions.** A replay of other agents' traffic
  (AgentX, #574) is the post-build check.

## Files

- [variants.py](variants.py): the proposal families, the walk, the policies, the allocation lag.
- [cases.py](cases.py), [copy_cases.py](copy_cases.py), [runs/copy-cases.json](runs/copy-cases.json):
  the copy probe's categories.
- [runs/agent-proxy.json](runs/agent-proxy.json): the agent proxy's step histograms (counts only).
- [cost.py](cost.py), [report.py](report.py), [results.json](results.json): the step model, every
  variant's tok/s, the tables.
- [bundles/wide-verify](bundles/wide-verify), [wide-verify.json](wide-verify.json),
  [runs/wide-verify-dgx4.txt](runs/wide-verify-dgx4.txt): the expert stream at wide blocks.
- [wide_rows_probe.py](wide_rows_probe.py), [wide-rows.json](wide-rows.json),
  [arm-1025.json](arm-1025.json): the cluster measurement of R rows on real text (prepared).
- [reference/routes_probe.py](reference/routes_probe.py), [reference/run_routes.sh](reference/run_routes.sh),
  [reference/routes_summary.py](reference/routes_summary.py),
  [runs/routes-summary.json](runs/routes-summary.json): the expert sets of copied text and prose
  from the reference forward.
