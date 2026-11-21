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

## Pre-Build Composed Verification (`dgx4`)

All 11 composed patches (`0833..0843`) were verified together over the `candidate-63-r5` vLLM tree before building the image:
- **`1,050 passed, 8 skipped, 0 failed`** across TileLang, PDL (`15/15`, `0` violations), vocab, DSpark Markov, attention, CED, SP prefill, repetition/thinking-budget invariance, contiguous CUDA pool, engine grid sync, direct loader (`io_uring` CUDA), and prepare kernel suites.

## Qualification Status (`tp3-triangle`, `dgx1..dgx3`)

Pending OCI image build (`spark-ds41f:candidate-64`), release test bundle on `dgx4`, single-digest sync to `dgx1..dgx3`, and 2-boot `release-tp3` + `single-stream` + `sampled` + `token_determinism.py --wide` qualification.
