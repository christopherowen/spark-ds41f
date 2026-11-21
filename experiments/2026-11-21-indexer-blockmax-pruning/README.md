# Exact BlockMax-32 Two-Stage Pruning for Wide Prefill Dense and Source Indexers

- **Date**: 2026-11-21
- **Base commit**: `737ef7d8b` (`spark-ds41f:candidate-63-r5`, vLLM tree `5f16a050bb283d3d964ab831de4cdf6fca74dea7`)
- **One intended variable**: Replace wide prefill (`rows >= 64`, `width > 16,384`) dense and source-layer (`candidate_blocks=True`) full-matrix top-$k$ selection in `vllm/models/deepseek_v4_1/tilelang/indexer.py` with exact BlockMax-32 two-stage pruning (`indexer_score_blockmax_v41` + `IndexerTopK(..., extent_div=32)` + vectorized `indexer_topk_gather32_v41` + vectorized `gather_scores` load in `indexer_topk_segment_v41`). Decode (`rows < 64`) and narrow widths (`width <= 16,384`) remain completely untouched.

## Mathematical Exactness Theorem (Zero Approximation)

Partition a row's $N$ visible positions into disjoint 32-position superblocks $b \in \{0, \dots, \lceil N / 32 \rceil - 1\}$ and record each superblock's exact ordered-key maximum:

$$M_b = \max_{i \in [32b,\, \min(32b+32, N))} \mathrm{key16}(s_i)$$

where $\mathrm{key16}$ is the exact order-preserving bijection over all 65,536 BF16 bit patterns (matching `_key16` in `indexer.py`, including `-NaN`, `-inf`, `-0.0`, `+0.0`, and `+NaN`).

- **Dense layers (`group = 1`, `topk = 512`)**: Let $S$ be the exact top-$K$ superblocks under $(M_b \text{ DESC}, b \text{ ASC})$. Any superblock $b' \notin S$ has $K$ distinct superblocks $b \in S$ that either satisfy $M_b > M_{b'}$ or $(M_b = M_{b'} \land b < b')$. Each such superblock $b \in S$ contains at least one token $i_b \in b$ with $(\mathrm{key16}(s_{i_b}), -i_b) > (\mathrm{key16}(s_j), -j)$ for every $j \in b'$. Therefore, at least $K$ distinct tokens strictly beat every token in $b'$, so the exact top-$K$ tokens are **100% contained** in $\bigcup_{b \in S} b$ ($512 \times 32 = 16,384 = \text{\_LONG\_SEGMENT\_ENTRIES}$ tokens, selected in a single 512-thread CTA pass with zero segment merge).
- **Source layer (`Layer 20`, `group = 8`, `CANDIDATE_BLOCKS = 2048`)**: Each 32-token superblock contains exactly $32 / 8 = 4$ aligned 8-token candidate blocks. By the same pigeonhole proof—enforcing $M_{b_{\text{last}}} \ge \texttt{0xFF80}$ on the superblock containing the last visible position—the exact top-2,048 8-token candidate blocks are **100% contained** in the top-2,048 superblocks ($2,048 \times 4 = 8,192 \le 16,384$ candidate blocks, selected in a single 512-thread CTA pass).

## Why Streaming In-Kernel Top-$k$ Fusion Lost vs. BlockMax-32

Before adopting BlockMax-32, five streaming in-kernel top-$k$ prototypes (`V1..V5`: SMEM `keys_buf`, threshold-filtered `cand_buf` + atomic radix, register binary search, and 1-warp-per-row compaction inside `indexer_score`) were timed on `dgx4` with `clock64()` breakdown:

- Baseline `indexer_score` (`4,706 us` at `910K`, `rows=256`) is **compute/MMA/reduction-bound** on GB10; its `233 MB` coalesced BF16 `Scores` writes are hidden behind MMA.
- Any per-tile filtering (`ballot`/`atomicAdd`) or periodic compaction inside the score loop adds directly to the compute bottleneck (`6,429–8,925 us` vs. `5,843 us` baseline).
- By contrast, computing a 32-token warp max via 1-instruction hardware `__reduce_max_sync(0xffffffffu, ok)` (`redux.sync.max.u32`) and flushing `BlockMax` in `CHUNK = 16` contiguous tiles (`64` B coalesced writes from `256` B SMEM) with cross-chunk K prefetch and `T.annotate_min_blocks_per_sm(3)` (`REG:166`, `LOCAL:0`, 3 CTAs/SM) adds negligible overhead while cutting selection by **13.6×** on dense layers (`1,150 us -> 84.4 us`) and **4.7×** on Layer 20 (`2,363 us -> 510 us`).

## Pre-Serving Kernel Verification (`dgx4`, `spark-ds41f:candidate-63-r5`)

### Correctness & PDL Contracts (`OVERLAY VERIFIED`)

- `tests/kernels/test_deepseek_v41_tilelang_pdl.py`: **`15 passed, 0 failed`** (`0` PDL violations).
- `tests/kernels/attention/test_deepseek_v41_tilelang.py` + `tests/models/test_deepseek_v4_1_attention.py` (`-k "indexer or topk"`): **`35 passed, 1 skipped, 0 failed`**.
- 13-case edge-case bit-exactness stress suite (`rows = 63, 64, 65`; `extent = 0, 1, 7, 8, 9, 31, 32, 33, 511, 512, 513, 16384, 16385, 32768`; `active = 0, 1000`; all-zeros ties, constant-positive ties, `-inf`, mixed `NaN`, differing page tables, `page_size = 64, 128`, `width = 1,048,576` 2-segment superblock merge, live `Scores` uint16 parity): **`13/13 passed, 100% torch.equal`**.
- Workspace scratch footprint (`rows = 256`): Dense (`width = 524,288`) shrinks from `281.00 -> 265.00 MiB` (`-16.00 MiB`); Source (`width = 1,048,576`) is `541.25 MiB` vs. `532.00 MiB` (`+9.25 MiB`).

### Standalone Kernel Benchmark vs. `candidate-63-r5` Baseline (`dgx4`, `rows = 256`, `page_size = 256`, 100% `torch.equal`)

#### Ratio-2 Dense Layers (`Layers 2, 8, 14`, `width = 524,288`, `candidate_blocks = False`)

| `extent` (ratio-2) | Context | Base (`us`) | BlockMax-32 (`us`) | Delta (`us`) | Delta (`%`) | Breakdown (`score` / `bmsel` / `gath`) |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `4,096` | `8K` | `132.1` | `98.1` | `-34.0` | `-25.70%` | `73.0 / 12.5 / 19.7` |
| `16,384` | `32K` | `298.2` | `240.8` | `-57.4` | `-19.25%` | `189.5 / 13.7 / 42.2` |
| `32,768` | `64K` | `488.6` | `404.7` | `-83.9` | `-17.17%` | `370.0 / 12.5 / 44.1` |
| `61,000` | `122K` | `845.7` | `715.9` | `-129.9` | `-15.35%` | `628.2 / 15.6 / 44.2` |
| `118,500` | `237K` | `1,566.8` | `1,294.4` | `-272.4` | `-17.39%` | `1,193.2 / 17.9 / 47.3` |
| `235,000` | `470K` | `2,985.0` | `2,462.9` | `-522.1` | `-17.49%` | `2,498.6 / 25.8 / 46.1` |
| `455,000` | `910K` | `5,843.2` | `4,936.9` | `-906.2` | `-15.51%` | `4,834.3 / 38.1 / 46.3` |

#### Ratio-1 Source Layer (`Layer 20`, `width = 1,048,576`, `candidate_blocks = True`)

| `extent` (ratio-1) | Base (`us`) | BlockMax-32 (`us`) | Delta (`us`) | Delta (`%`) | Breakdown (`score` / `bmsel` / `gath` / `fin`) |
| ---: | ---: | ---: | ---: | ---: | --- |
| `32,768` | `522.4` | `507.1` | `-15.3` | `-2.92%` | `347.3 / 25.8 / 44.1 / 45.9` |
| `122,000` | `1,756.2` | `1,629.3` | `-126.8` | `-7.22%` | `1,228.8 / 26.0 / 207.1 / 145.6` |
| `237,000` | `3,165.3` | `2,797.6` | `-367.7` | `-11.62%` | `2,353.2 / 34.0 / 218.1 / 158.9` |
| `470,000` | `6,063.3` | `5,411.8` | `-651.5` | `-10.74%` | `4,959.2 / 48.4 / 239.4 / 167.9` |
| `910,000` | `12,102.9` | `10,095.9` | `-2,007.0` | `-16.58%` | `9,591.1 / 103.7 / 228.8 / 177.4` |

Combined at `910K` (`3 × Dense-r2 + 1 × Source-r1`): **`-4.73 ms` per 256-row sub-chunk** (`-37.8 ms` per 6K TP3 chunk / `-75.6 ms` per 16K TP4 chunk).

## Prediction for Live `tp3-triangle` Serving Screen

- **Outputs**: 100% bit-exact (`model` `outputs_sha256 = b795e6d4...`, `rank1 = 1.0`, `KL = 0`).
- **Decode (`grid`, `c8`, `c1`)**: Strictly neutral (`rows < 64` uses the unchanged decode path; `0` regression).
- **Cold prefill**:
  - `32K`: `+0.0..+0.3%`
  - `262K`: `+0.5..+1.0%`
  - `500K`: `+1.0..+1.6%`
  - `1M`: **`+2.0..+3.0%`** (largest gain at long context where dense + source indexer selection dominates).

## Live Serving Runs (`tp3-triangle`, `dgx1..dgx3`)

All runs cooled `< 49 C` on `spark-ds41f:candidate-63-r5` inside lab window `dgx1-1795263633` (`results/private/lab/2026-11-21-campaign/`). Both boots passed `model` `128/128` greedy bit-exact (`outputs_sha256 = b795e6d4c813815a475ab1516faf277a90a86b73614c38f4700189484a8c532d`, `rank1_fraction = 1.0`, `perplexity = 1.3160824`, `failed = 0`).

| Arm | `grid` `mad5` GM vs `ctrl` | `c8` (tok/s) | `32K` (tok/s) | `131K` (tok/s) | `262K` (tok/s) | `500K` (tok/s) | `1M` (tok/s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ctrl-b1` | `+0.085%` | `169.52 ±1.03%` | `7,280` | `7,480` | `7,455` | `7,291` | `6,932` |
| `ctrl-b2` | `-0.085%` | `169.54 ±0.81%` | `7,238` | `7,459` | `7,419` | `7,262` | `6,903` |
| **`blockmax-b1`** | `+0.241%` | `169.08 ±0.85%` | `7,265` | **`7,499`** | **`7,460`** | **`7,345`** | **`7,038`** |
| **`blockmax-b2`** | `-0.017%` | `169.57 ±0.76%` | `7,321` | **`7,572`** | **`7,545`** | **`7,410`** | **`7,089`** |
| **2-boot delta (`blockmax` vs `ctrl`)** | `+0.112%` (neutral) | `-0.12%` (neutral) | `+0.47%` | **`+0.88%`** | **`+0.89%`** | **`+1.39%`** | **`+2.11%` (`+146 tok/s`)** |

### Conclusion

**2-boot verified win**: breaks `7,000 tok/s` at `1M` cold prefill on `tp3-triangle` (`6,918 -> 7,064 tok/s`, `+2.11%`) while keeping decode neutral and outputs `100%` bit-exact. Ready for `candidate-64`.
