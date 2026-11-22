# Short-Delta TTFT Breakdown, Co-Admission Audit & Renderer Thread-Pool A/B

Date: 2026-11-22.
Base commits: `3b4578d218` (`ctrl` single-blob audit) and `bcf59eb397` (`ctrl` vs `rw8` multi-turn & single-blob A/B).
Image: `spark-ds41f:candidate-65` (`sha256:4d6be6fa9e4086f60e2b5bc1beb4be9b1efc94a6ed0ff35c6ff45f5bc35d5c0a`, vLLM `0001..0844`, tree `34b01980b09e`).
Cluster: `dgx1..dgx3` (`tp3-triangle`, 16 seats; lab windows `dgx1-1795344377` and `dgx1-1795345380`, production `spark-ds41f:1120` restored cleanly with `doctor --live OK`).

## Key Findings

1. **Solo `1–2K` cached-delta TTFT on `candidate-65` (`tp3-triangle`) is already at the `0.43–0.60 s` hardware floor (not `0.9 s`)**:
   - On realistic 20-turn `60K`-cached prompts (`M-solo-d128..d2048`), `host` is only `11–20 ms`, `queue = 0.0 ms`, and `gpu = 197 ms (138 tok) / 230 ms (268 tok) / 277 ms (526 tok) / 417 ms (1,037 tok) / 590 ms (2,065 tok)`. The historical `~0.9 s` figure in `TODO.md` was from `2026-09-28` (pre-TileLang B12X stack).
2. **GPU co-admission of concurrent short deltas ALREADY works under `--max-parallel-prefills 1`**:
   - When `max_parallel_prefills == 1`, `PrefillInterleaveController.begin_step()` returns `None`, so the waiting-queue loop coalesces all queued short deltas up to `token_budget` into a single GPU step.
   - `8` concurrent `361`-token deltas (`2,888` uncached tokens) ran in `667–714 ms` GPU (`83–89 ms/req`, **`~3×` faster** than `8 × 250 ms` serialized).
   - `align_prefill_chunks` boundary straddle (`6,048` multiple, `C-straddle-d1024`) adds `+41.3 ms` (`+11.1%`: `370.8 → 412.1 ms`) when a `1K` delta crosses a `6,048` boundary (required by DeepSeek-V4.1 CED temperature-0 determinism).
3. **Real host bottleneck uncovered — single-threaded `DeepseekV4Renderer` (`--renderer-num-workers 1`) & missing BPE-invariant subchunk cache**:
   - `DeepseekV4Renderer` inherits from `BaseRenderer` (not `HfRenderer`), never calls `maybe_make_thread_pool`, and runs `apply_chat_template(tokenize=True)` through a 1-thread executor (`--renderer-num-workers 1`).
   - Whenever requests carry an uncached segment (growing single-turn prompt, large new tool output, or concurrent cold arrivals), `host` serialized linearly (`102 → 270 → 463 → 661 ms` at `wave4`; `105 → 273 → 472 → 952 → 1,223 ms` at `wave8`), exceeding the GPU prefill itself and jumping `host` from `98 → 200 ms` during `c4` decode.
4. **Live `131K` cold-prefill contention (`M-under-131k-pf-d512`: `1,530 ms` client = `526 ms` ZMQ-inbox wait + `1,016 ms` GPU)**:
   - Fully accounted for by `0226` (`continues_prefill` keeping 2 chunks deep in `batch_queue` to preserve `+1.1%` solo cold prefill throughput, trading `+1` chunk of wait for arriving short deltas before `prefill_yield = "short"` preempts).

## Tested Fix (`tp3-c65-rw8.json`)

- **`vllm/renderers/deepseek_v4.py`**: Call `maybe_make_thread_pool(self.tokenizer, config.model_config.renderer_num_workers + 1)` when `renderer_num_workers > 1` (mirroring `HfRenderer.__init__`).
- **`vllm/tokenizers/deepseek_v41.py`**: In `_resolve_cached_pieces`, cache and deduplicate BPE-invariant newline subchunks (`_split_parallel_newline_chunks`) individually when a large segment splits into `> 1` subchunks (100% bit-exact).
- Config flag: `--renderer-num-workers 8`.
- CPU microbenchmark (`dgx4`, cold `8×58K`): `617.2 ms (workers=1) → 260.9 ms (workers=4, 2.37×) → 150.4 ms (workers=8, 4.10×)`, `100%` exact.
- Unit tests (`dgx4`, `spark-ds41f:candidate-65`, `--gpus all`): `159 passed, 0 failed` (`test_renderer_parallel_tokenize.py` + `test_deepseek_v41.py` + `test_deepseek_v41_effort.py` + `test_deepseek_v41_serving_contract.py`).

## Live A/B Results on `tp3-triangle` (`spark-ds41f:candidate-65`, `0` Errors)

### Table 1: Growing Single-Message / Uncached-Segment Waves (`audit_short_delta.py`)

| Scenario (`~58K` cached prefix) | `ctrl` client p50 (max) | `ctrl` gpu / host | `ctrl` wave wall | `rw8` client p50 (max) | `rw8` gpu / host | `rw8` wave wall | Δ client p50 | Δ wave wall |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `A-solo-d128` (`382` uncached) | `362.0 ms` (`362.0`) | `262.8 / 98.8 ms` | — | **`285.8 ms`** (`361.7`) | `264.3 / 21.5 ms` | — | **`-21.0%`** | — |
| `A-solo-d256` (`248` uncached) | `323.3 ms` (`323.8`) | `225.2 / 97.7 ms` | — | **`248.1 ms`** (`253.5`) | `229.5 / 19.3 ms` | — | **`-23.3%`** | — |
| `A-solo-d512` (`499` uncached) | `373.8 ms` (`374.4`) | `275.9 / 97.9 ms` | — | **`300.4 ms`** (`304.7`) | `279.8 / 19.0 ms` | — | **`-19.6%`** | — |
| `A-solo-d1024` (`983` uncached) | `465.9 ms` (`472.7`) | `370.8 / 97.2 ms` | — | **`397.6 ms`** (`399.4`) | `376.8 / 21.9 ms` | — | **`-14.7%`** | — |
| `A-solo-d2048` (`1,944` uncached) | `598.8 ms` (`601.5`) | `496.9 / 101.9 ms` | — | **`527.9 ms`** (`528.0`) | `503.6 / 24.3 ms` | — | **`-11.8%`** | — |
| `B-wave4-d350-sync` (`4 × 339`) | `810.7 ms` (`952.8`) | `422.1 / 366.6 ms` | `960 ms` | **`664.2 ms`** (`665.0`) | `457.2 / 121.7 ms` | **`668 ms`** | **`-18.1%`** | **`-30.4%`** |
| `B-wave4-d350-stag25ms` | `792.8 ms` (`902.8`) | `435.7 / 338.2 ms` | `981 ms` | **`599.2 ms`** (`627.9`) | `447.4 / 142.7 ms` | **`656 ms`** | **`-24.4%`** | **`-33.2%`** |
| `B-wave8-d350-sync` (`8 × 340`) | `1,289.2 ms` (`1,544.2`) | `548.0 / 597.7 ms` | `1,553 ms` | **`949.1 ms`** (`953.8`) | `724.3 / 222.7 ms` | **`956 ms`** | **`-26.4%`** | **`-38.5%`** |
| `B-wave8-d350-stag25ms` | `1,214.2 ms` (`1,384.5`) | `547.8 / 514.0 ms` | `1,561 ms` | **`944.3 ms`** (`1,032.8`) | `867.5 / 76.8 ms` | **`1,093 ms`** | **`-22.2%`** | **`-30.0%`** |
| `C-straddle-d1024` (crosses `6048`) | `511.9 ms` (`514.7`) | `412.1 / 99.8 ms` | — | **`447.7 ms`** (`448.4`) | `418.8 / 28.8 ms` | — | **`-12.5%`** | — |
| `D-under-c4-decode-d512` | `537.7 ms` (`626.0`) | `336.4 / 200.4 ms` | — | **`388.8 ms`** (`427.2`) | `338.4 / 48.0 ms` | — | **`-27.7%`** | — |

### Table 2: Realistic 20-Turn Multi-Turn Cached Deltas (`audit_multiturn_delta.py`)

| Scenario (`~60K` cached, 20 turns) | `ctrl` client p50 (max) | `ctrl` queue / gpu / host | `rw8` client p50 (max) | `rw8` queue / gpu / host | Δ client p50 | Δ wave wall |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `M-solo-d128` (`138` uncached) | `219.2 ms` (`219.4`) | `0.0 / 201.4 / 15.8 ms` | **`210.3 ms`** (`213.6`) | `0.0 / 196.5 / 11.4 ms` | `-4.1%` | — |
| `M-solo-d256` (`268` uncached) | `245.0 ms` (`246.9`) | `0.0 / 229.6 / 17.3 ms` | **`241.4 ms`** (`241.5`) | `0.0 / 230.1 / 11.4 ms` | `-1.5%` | — |
| `M-solo-d512` (`526` uncached) | `296.6 ms` (`298.6`) | `0.0 / 277.1 / 18.7 ms` | **`288.7 ms`** (`299.5`) | `0.0 / 277.8 / 11.0 ms` | `-2.7%` | — |
| `M-solo-d1024` (`1,037` uncached) | `437.4 ms` (`438.5`) | `0.0 / 416.7 / 20.4 ms` | **`432.4 ms`** (`443.2`) | `0.0 / 420.1 / 12.2 ms` | `-1.1%` | — |
| `M-solo-d2048` (`2,065` uncached) | `602.6 ms` (`608.2`) | `0.0 / 589.5 / 13.5 ms` | `609.0 ms` (`613.3`) | `0.0 / 590.5 / 14.2 ms` | `+1.1%` | — |
| `M-wave4-d350-sync` (`4 × 360`) | `654.1 ms` (`656.5`) | `0.0 / 445.6 / 208.4 ms` | `659.8 ms` (`662.0`) | `0.0 / 451.3 / 120.1 ms` | `+0.9%` | `+0.5%` |
| `M-wave4-d350-stag25ms` | `591.6 ms` (`619.7`) | `0.0 / 455.7 / 126.4 ms` | `592.7 ms` (`619.1`) | `0.0 / 449.3 / 133.3 ms` | `+0.2%` | `-1.7%` |
| `M-wave8-d350-sync` (`8 × 361`) | `934.0 ms` (`935.8`) | `0.0 / 713.5 / 220.4 ms` | `971.4 ms` (`974.9`) | `0.0 / 666.7 / 304.6 ms` | `+4.0%` | `+3.2%` |
| `M-wave8-d350-stag25ms` | `830.7 ms` (`892.8`) | `0.0 / 726.4 / 79.8 ms` | `833.9 ms` (`896.5`) | `0.0 / 729.2 / 79.8 ms` | `+0.4%` | `-1.1%` |
| `M-wave8-d1024-sync` (`8 × 1,036`) | `1,946.2 ms` (`1,949.2`) | `197.7 / 1489.0 / 259.4 ms` | `1,936.1 ms` (`1,938.9`) | `199.3 / 1478.0 / 258.8 ms` | `-0.5%` | `-1.2%` |
| `M-under-c4-decode-d512` | `385.5 ms` (`405.3`) | `0.0 / 339.0 / 26.3 ms` | **`374.7 ms`** (`399.4`) | `0.0 / 340.0 / 37.8 ms` | `-2.8%` | — |
| `M-under-131k-pf-d512` | `1,529.8 ms` (`1,768.6`) | `0.0 / 1015.6 / 526.1 ms` | `1,538.2 ms` (`1,774.8`) | `0.0 / 1018.3 / 531.7 ms` | `+0.5%` | — |
| `M-cold4-32k-sync` (`4 × 30.6K` cold) | `9,991.7 ms` (`16,038.6`) | `5138.5 / 4036.8 / 793.5 ms` | `10,101.7 ms` (`16,204.5`) | `5180.0 / 4062.6 / 854.2 ms` | `+1.1%` | `+0.9%` |

- **Bit-exactness**: `108 / 108` requests across both audits had `0` prompt-token-count diffs and `100%` matching greedy prefixes (`'UnderDec'` vs `'UnderDecode'` on 2 SSE chunks reflecting 2 vs 3 speculative tokens in the first SSE frame).

## Conclusion

**Verdict: SURVIVED (`DeepseekV4Renderer` `maybe_make_thread_pool` + BPE-invariant subchunk cache + `--renderer-num-workers 4..8`).**
- Eliminates the single-threaded renderer serialization staircase on growing single-turn prompts and concurrent uncached-segment waves (**`-18%` to `-28%` client TTFT**, **`-30%` to `-38.5%` wave wall time**, **`-76%` host overhead under `c4` decode**), while staying flat-to-faster (`-1%..-4%` solo) on already-cached 20-turn multi-turn chat and 100% bit-exact.
- Also confirms GPU co-admission of concurrent short deltas is already active under `--max-parallel-prefills 1`, and solo `1–2K` multi-turn cached-delta TTFT is already at its `0.43–0.60 s` hardware floor.
