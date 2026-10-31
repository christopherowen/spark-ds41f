# TP6 on a switch: rank-order sum and mHC at five and six ranks

## What this changes

Three vLLM commits on top of candidate 10's series (`rc10/vllm/series`: patch head `ebfa73d3`, tree `24313745`, over vLLM `19f2c20e`). They are numbered 0538-0540 on rc10, which collides with rc11's 0538-0540: they need renumbering on rebase (no rc11 patch touches these five files, so a clean apply is expected but not verified).

- **rank-order sum, 5 and 6 sources** (`vllm/models/deepseek_v4_1/tilelang/collectives.py`, +63/-33; tests +59/-9).
  - Today `rank_order_sum_kernel`, its prepare and its dispatch assert `2 <= world <= 4`. At 5-6 ranks `rank_order_sum` falls back to torch: same bits, several passes and an FP32 temporary per chunk.
  - The patch replaces this with one `sum_rows` body unrolled over a Python tuple, with one signature per world size. Source *r* loads in slot *r*. The first source is taken as-is, so signed zeros survive. The rest add in FP32 and round once.
  - `MAX_RANK_ORDER_WORLD = 6` bounds kernel, dispatch and prepare together, so the kernel the forward launches is the one the frozen cache compiled.
  - Each source needs its own `T.Tensor` annotation. If they share one object, TileLang binds a single buffer; the IR comparison caught that on the first try.
- **mHC `post_pre`, 5 and 6 parts** (`tilelang/mhc.py:113`, +7/-2). This is the hard block: with SP in the RoCE policy every layer declares `("post_pre", tp_size)`, and `project_streams` asserted `world <= 4`, so the worker dies at kernel prepare. The body was already generic over `world`: `T.unroll(world)`, 16 `COLS` registers, and no growth in shared memory or registers. Only the bound changes.
- **Tests only** (`tests/models/test_deepseek_v41_sp_project_reduce_scatter.py`). Row exchange and `all_to_all_rows` move from `[3, 4]` to `[3, 4, 5, 6]`. At 6 ranks the exchange is direct, because the relay applies only when `world == 4`.

Rank-limit sweep of the rebuilt rc10 tree (422 patches over `19f2c20e`, reproducing `ebfa73d3`/`24313745`). Nothing else blocks 6:
- `round_up(o_groups, tp)` pads 64 heads to 96, so ranks 4-5 carry only zero-weight padding heads. Per-rank shapes stay those of TP4.
- MoE 2304/6 = 384 = 3×128.
- Engram: 24 % 6 = 0.
- Vocabulary pads to `lcm(64, 6)`.
- The tile tables are tuned for TP4/TP3. Their `.get` defaults are valid, so missing entries cost speed but not correctness.

## Evidence

All measurements: 6 Sparks on one switch, image `spark-ds41f:19f2c20ed4d6-next-10-tp6` (candidate 10 plus these 3 patches), sparknet 0.5.0 `direct`, servebench and the determinism tool from `75bd750c`.

| | Measured |
|---|---|
| Decode, closed loop (60 s warm + 240 s, 10 windows), 16 seats | aggregate: c1 83.7 · c8 234.6 · c16 305.9 tok/s (c1 is also per request) |
| Prefill, real text (lab boot) | 32K 7,391 (±7 %) · 131K 7,648 · 262K 7,383 · 524K 6,845 tok/s; 1M was skipped by servebench (prompt = max_model_len) |
| Determinism, `token_determinism_text.py --wide` | `identical: true` with and without logprobs (text digests `7fde24b3` and `12fd8257`, the same in both runs) |
| Against a TP4 (1025) corpus, `servebench model --limit 64 --top 20` | `servebench compare`: KL mean 0.0239 nats (p99 0.295), top-1 94.65%, acceptance length +0.3% (2.259) |
| tool-eval-bench 3.0.1.dev6, 104 scenarios | final score 90/100 (89 pass, 10 partial, 5 fail) |
| Weights per rank (model loading) | 48.9 GiB (TP4 ≈ 72) |
| Overnight serving (8 h, real traffic) | ~280-293 tok/s hourly mean at 16-25 concurrent requests; no 5xx and no restart; 12 requests were aborted by clients after their own timeouts (11 at ~120 s during the post-cutover burst, 1 at 3,600 s) |

**Not paired.** Your candidate-10 TP4 ring numbers (c1 77.50; c8 24.32 per stream ≈ 195 aggregate) come from another fabric and another session. Read the speed rows as indicative only. TP6 and TP4 cannot match bit for bit, because they reduce different partial sums. The KL gate we used (≤0.025, ≥95% top-1) was taken from your r6h/r6k comparison. Top-1 at **94.65% is just under 95%**: we accepted it on KL plus acceptance, and that is a judgement call.

## Your gates: run and not run

| Gate | Status |
|---|---|
| Determinism gate `--wide`, with and without logprobs | **run**, `identical: true` |
| `servebench model` KL / top-1 against a reference corpus | **run**, against TP4-1025, not a TP6 A/A |
| `servebench engine` grid (20 shapes, 100 replays) | **run**, widest CI 1.31% |
| Closed loops c1/c8/c16, prefill 32K-524K | **run** (lab boot); 1M only on the production boot, with live traffic |
| Host memory | **run**: 1 s sampling on all 6 nodes in both boots; under the production config (32 seats, KV 39.5 GiB/rank) the minimum MemAvailable was 13.7 GiB on the head and 15.4-17.5 GiB on the others, reached during a 1.04M prefill |
| A/A repeat after restart (your "candidate again") | **partial**: the production restart (different seats/KV, same image) reproduced our 7-case T=0 probe hashes exactly; the 64-response greedy A/A was not repeated |
| Paired TP4 control in the same session | **not run** |
| Release tests on the `-tp6` image | **not run**. The image passed only the build checks and import smokes |
| GPU kernel tests for these patches (rank-order sum and mHC bit-exact at 2-6 parts, graph replay) | **no archived receipt**: run once on one node during the build window (350 passed; 12 errors from a missing conftest fixture in an unrelated test under `--noconftest`), but the output was not kept. Treat as not run until we re-run and attach it |
| CED cross-shape probe, distinct streams, controls probe, sink counters | **not run** |
| `probe collectives` at 6 ranks | **cannot run**: `--world-size` accepts 2/3/4 |

Host-only evidence:
- TIR identical (`structural_equal`) to rc10 for world 2/3/4;
- world 7 rejected;
- extended collectives tests: 76 pass, 47 GPU-skipped; row exchange 3-6: 20 pass;
- three mutants killed (wrong first source, reversed order, double rounding);
- `bin/spark build prepare` reproduced head `46f11748` and tree `6ffc6c78` from the series.

## Still unqualified

- sparknet `direct` above 4 ranks is measured only by our serving run. The all-gather uses a second 4-peer group for the first time (`_device.py:461`). The 2 MiB dispatch threshold was tuned at 4 ranks, and each rank now receives 5 full messages instead of 3.
- `topology.py` still caps deployments at 3 or 4 nodes (#662 keeps that). There is no TP6 shape in `config/`, and we do not propose one here.
- Ranks 4-5 carry only padding heads (zero weights). Attention therefore costs the same as at TP4, and the decode win comes from MoE, heads and shared weights.
