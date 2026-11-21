# Candidate `64` (`spark-ds41f:candidate-64`, vLLM `0001..0843`)

Date: 2026-11-21.
Base: `candidate-63-r5` (`vLLM 0001..0832`, tree `5f16a050bb283d3d964ab831de4cdf6fca74dea7`), built over `spark-ds41f-1120` (`candidate-62`, `vLLM 0001..0829`).

## Included Patches (`0833..0843`)

Every patch stacks cleanly on `candidate-63-r5` (which eliminates boot-to-boot SMMU L3 page-table drift across all 12 shapes via universal auto-sized contiguous CUDA pool pre-allocation, 2-pass `512 MiB` `hugetlb` pageblock reset, and `engine_grid` TP rank sync):

| Patch | Change | Source Experiment / PR | Pre-Consolidation Verification |
| --- | --- | --- | --- |
| `0833` | Wire `mhc` `async_sinkhorn` across model + DSpark | [`2026-11-21-mhc-async-sinkhorn`](../../2026-11-21-mhc-async-sinkhorn/README.md) (`#1264`) | 2-boot `tp3`: `-1.00%` grid (`mad5`), `c1 tpot_p50 -0.34 ms (-2.35%)`, `128/128` bit-exact |
| `0834` | Exact BlockMax-32 two-stage pruning for wide prefill indexers | [`2026-11-21-indexer-blockmax-pruning`](../../2026-11-21-indexer-blockmax-pruning/README.md) (`#1265`) | 2-boot `tp3`: `1M` `7,064` vs `6,918 tok/s` (`+2.11%`), `500K +1.39%`, `262K +0.89%`, `128/128` exact |
| `0835` | Group-aware wide-decode `WO-A` `block_K=256/128` + in-kernel shards | [`2026-11-21-wo-a-wide-block-k`](../../2026-11-21-wo-a-wide-block-k/README.md) (`#1267`) | 2-boot `tp3`: `8x48 @4K -0.410 ms (-1.49%)`, `0` regression on `1x3`, `128/128` bit-exact |
| `0836` | Real-row profiler writable KV isolation | [`2026-11-21-real-rows-kv-isolation`](../../2026-11-21-real-rows-kv-isolation/README.md) (`#1270`) | `6/6` GPU `torch.equal` order-independent parity tests |
| `0837` | Defensive async in-flight prefill stall attribution (`_last_prefill_output_id`) | [`2026-11-21-async-prefill-stall-attribution`](../../2026-11-21-async-prefill-stall-attribution/README.md) (`#1273`) | `15/15` unit tests (`T_decode >= 50 ms` bugfix); 2-boot `tp3` neutral (`128/128` bit-exact) |
| `0838` | Tensor-Core WAY-2 exact full-vocab top-2 in batched DSpark | [`2026-11-21-dspark-exact-top2`](../../2026-11-21-dspark-exact-top2/README.md) (`#1274`) | 2-boot `tp3`: `model` `acceptance_length 2.3035 -> 2.32004 (+0.72%)`, `c8 169.68` vs `169.53` (`tpot_p50 -0.33 ms`), `96/96` tests, `128/128` bit-exact |
| `0839` | Content-addressed warm-cache `prepare_declared()` index | [`2026-11-21-boot-prepare-warm-index`](../../2026-11-21-boot-prepare-warm-index/README.md) (`#1275`) | 2-boot `tp3`: warm API ready `108.5 s` vs `122.3 s` (`-13.8 s / -11.3%`), `33/33` tests, `128/128` bit-exact |
| `0840` | Route single-segment SWA-only layers directly to `extend` | [`2026-11-21-swa-single-segment-extend`](../../2026-11-21-swa-single-segment-extend/README.md) (`#1277`) | `38/38` GPU `uint16` exact tests; `16/16` shapes faster (`-1.06..-4.06 us/L`), drops 2 kernels/shape + `Partial` scratch |
| `0841` | Per-request shard-local vs. control split for mixed sampling batches | [`2026-11-21-per-row-mixed-sampling-split`](../../2026-11-21-per-row-mixed-sampling-split/README.md) (`#1278`) | `5/5` GPU + `TP=3` selective-gather parity tests; `-47.3%` (`8x6`) / `-43.7%` (`16x6`) compute + `4x-16x` less RoCE |
| `0842` | Loop containment & thinking-budget multi-token verify invariance (`#864`) | [`2026-11-21-loop-thinking-verify-invariance`](../../2026-11-21-loop-thinking-verify-invariance/README.md) (`#1279`) | `131/131` loop/thinking/invariance/PDL tests; fixes 3 async/grace/2-state-automaton bugs |
| `0843` | Dedicated `R=3` decode dense `indexer_score_v41` (composed cleanly with `0834` at `_BLOCKMAX_MIN_ROWS = 97`) | [`2026-11-21-decode-indexer-multirow-kpool`](../../2026-11-21-decode-indexer-multirow-kpool/README.md) (`#1276`) | `107/107` tests; Cold-DRAM `8x6 @256K -17.1% (-1.31 ms/8L)`, `8x6 @1M -27.4% (-9.39 ms/8L)`, `16x6 @1M -24.2% (-16.15 ms/8L)`; 2-boot `tp3` `fl25 = -0.009%`, `128/128` bit-exact |

## Built-Image Verification (`dgx4`, zero overlays)

- **OCI Image**: `spark-ds41f:candidate-64`, digest `sha256:50b294060f808ce5d2369441995762d591e1e213fe17f75a4635e43ad44ce312`, label `local.spark-ds41f.vllm.tree = 7464b04c3cc79c6de1c1745f92c1cdeb2999a2a8` (built in `494 s`, lowest `MemAvailable 82.1 GiB`, GPU/TileLang/sparknet/no-B12X import smoke passed; exact digest verified across `dgx1`, `dgx2`, `dgx3`, `dgx4`).
- **Built-image unit & GPU test suites**: **`2,315 unique tests passed, 40 skipped, 0 failed`** natively inside `spark-ds41f:candidate-64` (including all 35 DS4.1 TileLang/vocab/WO/PDL/loader/prepare/parser/scheduler/pool suites, all `15/15` PDL `layer_chain` bit-equal replays, repetition & GPU thinking-budget suites, and all 7 post-`1120` experiment test files [`61 passed, 1 skipped, 0 failed`]).

## Live 2-Boot Qualification (`tp3-triangle`, `dgx1..dgx3`, commit `02b6e64ed`)

Ran inside lab window `dgx1-1795263633` on unrebooted `1120`-aged nodes (post-Boot-2 `O13 = 3 / 13 / 7`, `SecPageTables = 26.3 / 25.9 / 22.7 MiB`, so `0832` contiguous pool stayed cleanly inactive pending reboot; see `2026-11-21-boot-variance-contiguous-pool/README.md` Phase 6 for fresh-reboot `O13 = 189..209` verification). Raw receipts saved to `dgx1:~/projects/spark-ds41f/results/private/lab/2026-11-21-candidate-64-tp3/`. **`0` failed requests across every suite on every boot.**

### Bit-Exactness & Determinism Gate (100% Exact)

- **`model` (`256` prompts, `c8`, `max_tokens=256`, `top=20`)**:
  - Boot 1 (`c64-b1`): `outputs_sha256 = 4ce8fcf84506db624870e8df25b04991b35805cc974ba9d201e60d7727eb22e2`, `rank1_fraction = 1.0`, `perplexity = 1.3137270` (`+0.000%` vs `1120`'s `1.31373`), **`acceptance_length = 2.33805`** (vs `2.332` on `1120`), `failed = 0`.
  - Boot 2 (`c64-b2`): **`256/256` greedy & top-20 logprobs bit-identical** to Boot 1 (`outputs_sha256 = 4ce8fcf84506...`, `perplexity = 1.3137270`, `acceptance_length = 2.33763`, `failed = 0`).
  - Cross-check vs `1120` / `c63-ctrl` (first 128 prompts): **`128/128` greedy tokens bit-identical** and **`128/128` top-20 teacher-forced logprob sequences (`32,768` positions) 100% bit-identical** (`KL = 0.000e+00`).
- **`scripts/sonar_determinism.py --wide` (`c64-b1`)**: **`identical: true, logprobs: true`**, `0` token diffs & `0` logprob diffs across all 6 scenarios (`c8`, `staggered`, `mixed`, `cached`, `again`, `mixed15`).

### Warm Boot Time (`API ready`)

- Boot 1 (`c64-b1`, populates warm prepare index + compiles new kernels): `+134.3 s`
- **Boot 2 (`c64-b2`, warm index)**: **`+107.6 s`**
- **Boot 3 (`c64-b3-warm`, warm index)**: **`+109.8 s`**
- **2-boot warm mean**: **`108.7 s`** vs `122.3 s` `c63` (`-13.6 s`, `-11.1%`) and `~134.6 s` `1120` (**`-25.9 s`, `-19.2%`**).

### Throughput, Latency & Engine Grid (`c64-b1` / `c64-b2` vs `1120` Reference & `c63-ctrl`)

| Metric | `1120` Ref (`tp3`) | `c63-ctrl` (aged, 2-boot) | `c64-b1` | `c64-b2` | `c64` 2-Boot Mean | Delta vs `1120` / `c63` |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| **Grid GM (`mad5`)** | 9 multi-stream shapes | 15 shapes (`±0.20%`) | `b1→b2 = -0.180%` | — | **All 15 cells faster** | **`-1.72%` vs `1120` (9 multi) / `-1.44%` vs `c63` (all 15)** |
| `decode 1x3 @4K` (ms) | `33.243`* | `16.800` | `16.596` | `16.689` | **`16.643`** | `-0.94%` vs `c63` |
| `decode 8x24 @4K` (ms) | `22.579` | `22.512` | `22.237` | `22.204` | **`22.220`** | **`-1.59%` vs `1120` / `-1.29%` vs `c63`** |
| `decode 8x24 @256K` (ms) | `22.682` | `22.522` | `22.172` | `22.114` | **`22.143`** | **`-2.37%` vs `1120` / `-1.68%` vs `c63`** |
| `decode 8x48 @4K` (ms) | `27.411` | `27.556` | `27.245` | `27.145` | **`27.195`** | **`-0.79%` vs `1120` / `-1.31%` vs `c63`** |
| `decode 1x3 @1016K` (ms) | `33.529`* | `16.961` | `16.609` | `16.580` | **`16.595`** | **`-2.16%` vs `c63`** |
| **`c1` (`single-stream`, tok/s)** | — | `69.69` (`tpot 14.44`) | `69.80` (`14.34`) | `69.87` (`14.41`) | **`69.84` (`tpot 14.38`)** | `+0.21%` vs `c63` |
| **`c8` (tok/s, `tpot_p50`)** | `170.68` (`46.56`) | `169.53` (`46.96`) | `169.23` (`46.79`) | `169.70` (`46.85`) | **`169.46` (`46.82`)** | `-0.72%` vs fresh `1120` / flat vs aged `c63` |
| **`c4` (tok/s, `tpot_p50`)** | `131.59` (`30.36`) | — | `131.01` (`30.56`) | `130.98` (`30.29`) | **`130.99` (`30.43`)** | `-0.46%` vs fresh `1120` (within `±1.8%` CI) |
| **`poisson` (tok/s)** | `119.54` (`30.59`) | — | `119.55` (`30.31`) | `119.57` (`30.94`) | **`119.56` (`30.62`)** | `+0.02%` (neutral) |
| **`sampled` (`c16 / c8 / c1`)** | — | — | `166.05 / 165.80 / 68.09` | — | `166.05 / 165.80 / 68.09` | `0` failed |
| **`stall-3` `probe_ttft_p50` (s)** | `0.796` | `0.684` | `0.686` | `0.664` | **`0.675`** | **`-15.2%` vs `1120`** |
| **`stall-6` `probe_ttft_p50` (s)** | `1.761` | `1.568` | `1.627` | `1.653` | **`1.640`** | `-6.9%` vs `1120` (slot-8 cascade open) |
| **Prefill `32K` (tok/s)** | `7,270.0` | `7,259` | `7,182.9` | `7,156.5` | `7,169.7` | `-1.38%` (aged `O13=3/13/7`) |
| **Prefill `131K` (tok/s)** | `7,502.4` | `7,470` | `7,478.4` | `7,443.7` | `7,461.0` | `-0.55%` (aged `O13=3/13/7`) |
| **Prefill `262K` (tok/s)** | `7,445.2` | `7,437` | `7,441.4` | `7,398.9` | `7,420.2` | `-0.34%` (aged `O13=3/13/7`) |
| **Prefill `524K` (tok/s)** | `7,258.6` | `7,276` | `7,293.7` | `7,260.3` | **`7,277.0`** | **`+0.25%` vs `1120`** |
| **Prefill `1M` (tok/s)** | `6,921.6` | `6,918` | `7,007.3` | `6,992.4` | **`6,999.8`** | **`+1.13%` vs `1120` (`+1.18%` vs `c63`)** |

*(\*`1120`'s recorded `1x*` grid shapes hit the pre-`0831` unsynced dummy-shape `33 ms` bug; `0831` fixed it to `~16.6–16.8 ms`.)*

Note: Cold-DRAM deep-context decode (`0843`: `-17.1%` / `-1.31 ms/8L` at `8x6 @256K`, `-27.4%` / `-9.39 ms/8L` at `8x6 @1M`, `-24.2%` / `-16.15 ms/8L` at `16x6 @1M`) and mixed-sampling batches (`0841`: `-47.3%` / `-43.7%` compute + `4x–16x` less RoCE) do not appear in synthetic L2-resident `engine grid` or homogeneous greedy `c8`, and stack directly on top in real production traffic. A clean post-reboot run will also activate `0832`'s contiguous CUDA pool (`O13 = ~190–209`) for the additional `+1.2%..+1.5%` steady-state lift verified in `candidate-63-r4` Phase 6.
