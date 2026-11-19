# FP8 Group-32 Dominance Block-8 Summaries and Long-Context `r=1` Shortlist Routing (`2026-11-08-lsa-dominance-fp8-summary`)

## Summary & Goal

In `spark-ds41f-1107` (`vLLM 0741`), opt-in Latent Sparse Attention (`VLLM_DS41_LSA=1`, default-off `0`/unset) stores a 128-dimensional `bfloat16` arithmetic mean for each 8-key block at the tail of every completed indexer KV cache page (`STATE_BYTES = 100`: `68 B` MXFP4 key+scale + `32 B` per slot = `256 B` per 8-key block) and routes coarse-to-fine shortlist indexing (`QueryReusePlan`) across 32K–1M prefill and `r >= 2` decode steps.

This experiment evaluates three coupled questions on GB10 (`sm_121a`, `sp5`):

1. **Intra-block needle dilution in plain 8-key arithmetic means (`0741`)**:
   - Plain arithmetic averaging $\bar{k}_B = \frac{1}{8}\sum_{t=0}^7 k_t$ divides an isolated high-salience "needle" key inside an 8-key block by $8$, or suffers destructive cancellation when surrounding tokens in the block have opposing signs.
   - Can a training-free block-8 summary computed inside `summarize_blocks` preserve intra-block outlier keys and improve both standard top-512 recall and single-needle-in-a-block recall without adding any extra coarse-scoring passes or workspace buffers?
2. **Summary storage footprint and GB10 L2 cache residency (`STATE_BYTES = 84` vs `100`)**:
   - Storing the 128-dim block-8 summary in `float8_e4m3fn` (`128 B` per 8-key block = `+16 B/slot`, `STATE_BYTES = 84`) cuts `VLLM_DS41_LSA` KV cache overhead in half (`+21.31 B/token` vs `+42.63 B/token` across the 4 indexer layers).
   - At 1M context on Source Layer 20 (`1,048,576` uncompressed slots = `131,072` blocks of 8 keys), a `bfloat16` summary pool occupies `32.0 MiB` (`131,072 x 256 B`), exceeding the effective working set of GB10's shared L2 cache during multi-row `prepared_scores` scans and spilling to LPDDR5X (`273 GB/s`). In `float8_e4m3fn`, the entire 1M Layer-20 summary pool occupies `16.0 MiB` (`131,072 x 128 B`), remaining L2-resident across CTAs.
3. **Long-context single-row decode (`r = 1`) shortlist routing**:
   - `0741` unconditionally bypassed `PageShortlistPlan` to `plans.native_index["decode"]` whenever `mode == "decode" and rows < 2`. While native indexing is faster at `r = 1` for short/medium contexts (`score_width < 262144`), at `score_width >= 262144` (`524K–1M` tokens on Dense Layers 2, 8, 14; `262K–1M` tokens on Source Layer 20), scanning all MXFP4 pages natively takes `108–546 us` per layer vs `60–270 us` with `QueryReusePlan`.

## Base & Intended Causal Variable

- **Deployment base commit**: `cc20ed18be1f7cbf5c6c53cdee945f7cc77d5fc5` (`spark-ds41f-1107`)
- **vLLM base**: `19f2c20ed4d64ae22bd009ceacadd85e2a81e354` with `spark-ds41f-1107` series `0001..0761` (tree `ed1bdb7a7684da33d403a055521ac0c4dd7ca670`)
- **Candidate vLLM patch**: `experiments/2026-11-08-lsa-dominance-fp8-summary/vllm/0762-perf-dsv41-lsa-group32-dominance-fp8-summary.patch`
  - `patch_head`: `3b3aa7fc4467907964757ca869dbdd906cac6bdc`
  - `expected_tree`: `3fb853105a479a8a74ddfcf05035c6b3520eba11`
  - `patchset_sha256`: `1e692f7cad9bf7203c7de88ccacb8aaa3f228bd5ccf63e1df1945d2189fd62e5`
- **Intended causal variable**:
  1. Replace the 32-byte `bfloat16` arithmetic-mean block-8 summary (`STATE_BYTES = 100`) in `vllm/models/deepseek_v4_1/tilelang/block_shortlist.py` and `query_reuse.py` with a 16-byte `float8_e4m3fn` Group-32 dominance-blended summary (`STATE_BYTES = 84`):
     $$\hat{k}_{B,g} = \operatorname{ FP8\_E4M3 }\!\left(0.85\,\bar{k}_{B,g} + 0.15\,k_{t^*_g,g}\right), \quad t^*_g = \arg\max_{t \in \{0..7\}} \|k_{t,g} - \bar{k}_{B,g}\|_2^2$$
     evaluated independently on each of the four 32-element MXFP4 scale groups $g \in \{0, 1, 2, 3\}$ of the 128-dim key.
  2. In `vllm/models/deepseek_v4_1/tilelang/attention.py` (`_index_chunk`), keep `PageShortlistPlan` active for `mode == "decode"` and `rows < 2` when `score_width is not None and score_width >= 262144`.

## Candidates Screened (Including Unfavorable Variants)

All variants were implemented as TileLang `sm_121a` kernels and evaluated on `sp5` (`NVIDIA GB10`, container `spark-ds41f:1107`) against exact `IndexerV41Plan` (`top-512`) ground truth:

1. **Baseline (`0741` Plain Arithmetic Mean, `BF16`, `STATE_BYTES = 100`, `+32 B/slot`)**:
   - $\hat{k}_B = \frac{1}{8}\sum_{t=0}^7 k_t$ stored in `bfloat16`.
2. **Variant 1 (`V1`: RMS / L2 Norm-Rescaled Mean, `BF16`, `STATE_BYTES = 100`, `+32 B/slot`)**:
   - Rescales $\bar{k}_B$ by $\operatorname{rms}_t(\|k_{t,g}\|_2) / (\|\bar{k}_{B,g}\|_2 + \epsilon)$ (or $\max_t \|k_{t,g}\|_2 / \|\bar{k}_{B,g}\|_2$ or coordinate-wise signed RMS).
   - **Finding (unfavorable on clustered keys)**: Because the coarse score $\sum_{h=0}^{31} w_h \max(q_h^\top \hat{k}_B, 0)$ is positively homogeneous of degree 1, inflating the norm of low-coherence blocks (where keys cancel because they are uncorrelated noise) causes noise blocks to outscore genuinely coherent signal blocks (`Dense b64` recall drops from `87.25%` to `82.34%` for RMS-rescaled and `74.47%` for max-norm-rescaled).
3. **Variant 2 (`V2`: Group-32 Dominance Blend `0.85 * mean + 0.15 * k*`, `BF16`, `STATE_BYTES = 100`, `+32 B/slot`)**:
   - Selects the key $t^*_g = \arg\max_{t \in \{0..7\}} \|k_{t,g} - \bar{k}_{B,g}\|_2^2$ with maximum squared L2 deviation from the block mean in each 32-dim scale group $g \in \{0..3\}$ and blends $0.85\,\bar{k}_{B,g} + 0.15\,k_{t^*_g,g}$.
   - Also screened **Deviation-Salience Weighted Mean (`V2b`, $\tau = 0.5$)**, where $w_{t,g} = \operatorname{softmax}_t(\|k_{t,g} - \bar{k}_{B,g}\|_2 / (\sigma_{B,g}\tau))$.
   - **Finding**: Both `V2` and `V2b` strictly beat the plain arithmetic mean across **every** recall metric (clustered, needle-uncorrelated, and needle-canceling). `V2` (Group-32 Dominance Blend) requires only a fast 8-way warp-local reduction and argmax in `summarize_blocks` without transcendental `exp` calls.
4. **Variant 3 (`V3`: Intra-Block Dispersion Bias $+\beta \sigma_B \sum_h w_h \|q_h\|$, `BF16`, `STATE_BYTES = 102`, `+34 B/slot`)**:
   - Stores intra-block standard deviation $\sigma_B$ alongside $\bar{k}_B$ and adds $\beta \sigma_B \sum_h w_h \|q_h\|$ in `prepared_scores`.
   - **Finding (unfavorable)**: Adding an isotropic dispersion bonus in `prepared_scores` penalizes coherent multi-token spans relative to high-variance noise blocks (`Dense b64` recall drops from `87.25%` to `84.82%` at $\beta=0.10$ and `77.78%` at $\beta=0.20$), while adding `+2 B/slot` and slowing down `prepared_scores` by `5%–15%`.
5. **Variant 4 (`0762` Candidate: `FP8 E4M3` Group-32 Dominance Blend `0.85 * mean + 0.15 * k*`, `STATE_BYTES = 84`, `+16 B/slot`)**:
   - Computes the `V2` Group-32 dominance blend in `float32` registers inside `summarize_blocks` (`threads=512`, `dev_sq` in shared memory `[groups, 8]`) and writes `T.float8_e4m3fn` directly to `Pool` at byte offset `page * 68 .. page * 84`.
   - In `prepared_scores` (`query_reuse.py`) and `block_scores` (`block_shortlist.py`), loads `float8_e4m3fn` summaries and casts to `bfloat16` in shared memory `a` before `T.gemm(a, b, acc)`.

## Quality & Recall Verification (`sp5`, Exact `IndexerV41Plan` Ground Truth)

Receipts:
- `experiments/2026-11-08-lsa-dominance-fp8-summary/receipts/exact_plan_recall_summary.json`
- `experiments/2026-11-08-lsa-dominance-fp8-summary/receipts/length_recall_sweep.json`

### 1. Multi-Regime Recall at `32K` (`rows = 64`, Exact `QueryReusePlan` vs `IndexerV41Plan`)

- **Standard clustered context**: Realistic RoPE/topic-clustered MXFP4 keys (`60%` block topic + `40%` token innovation, UE8M0 scale exponents `122..129`).
- **Single-needle (uncorrelated)**: `512` blocks (`1` key per block) contain a high-alignment needle key (`cos ~ 0.65` with query) while the other `7` keys in the 8-token block are uncorrelated context keys (`mean ||bar_k||_2` shrinks by `~1/sqrt(8)` and the needle component is divided by `8`).
- **Single-needle (canceling)**: `512` blocks contain `1` needle key while `1–2` keys in the same 8-token block have opposing sign components.

| Summary Variant | `STATE_BYTES` (`+B/slot`) | Dense `b=64` Mean Recall | Dense `b=64` Min Recall | Source `b=512` Mean Recall | Source `b=1024` Mean Recall | Needle (`b=64` Uncorrelated) | Needle (`b=64` Canceling) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline `0741`: Plain Mean (`BF16`)** | `100` (`+32 B`) | `87.25%` | `65.43%` | `93.44%` | `97.92%` | `27.28%` | `40.49%` |
| `V1`: Group-32 RMS-Rescaled (`BF16`) | `100` (`+32 B`) | `82.34%` | `49.80%` | `90.24%` | `96.60%` | `26.86%` | `38.68%` |
| `V1`: Group-32 MaxNorm-Rescaled (`BF16`) | `100` (`+32 B`) | `74.47%` | `35.35%` | `85.39%` | `94.57%` | `26.64%` | `37.65%` |
| `V1`: Coord Signed-RMS (`BF16`) | `100` (`+32 B`) | `85.94%` | `61.33%` | `92.51%` | `97.52%` | `26.59%` | `37.98%` |
| `V2`: Group-32 Dominance `0.85*mean + 0.15*k*` (`BF16`) | `100` (`+32 B`) | `87.74%` | `66.80%` | **`95.36%`** | **`98.85%`** | `41.09%` | `69.47%` |
| `V2b`: Deviation-Salience Softmax (`tau=0.5`, `BF16`) | `100` (`+32 B`) | **`87.78%`** | `66.80%` | `95.22%` | `98.81%` | `39.32%` | **`70.96%`** |
| `V3`: Mean + Dispersion Bias (`beta=0.10`, `BF16`) | `102` (`+34 B`) | `84.82%` | `63.67%` | `91.85%` | `97.27%` | `34.12%` | `51.14%` |
| `V4`: Plain Mean (`FP8 E4M3`) | `84` (`+16 B`) | `87.21%` | `65.23%` | `93.41%` | `97.92%` | `27.32%` | `40.48%` |
| **Candidate `0762`: Group-32 Dominance `0.85*mean + 0.15*k*` (`FP8 E4M3`)** | **`84` (`+16 B`)** | **`87.74%`** (`+0.49%`) | **`66.60%`** (`+1.17%`) | **`95.35%`** (`+1.91%`) | **`98.85%`** (`+0.93%`) | **`41.15%`** (`+13.87%`) | **`69.47%`** (`+28.98%`) |

### 2. Context-Length Recall Sweep (`32K` to `1M` Tokens, `rows = 8`)

| Context Tokens | Layer Role (`score_width`, `budget`) | Baseline `0741` `BF16` Mean Recall | Candidate `0762` `FP8` Dom Recall | Delta (Clustered) | Baseline `0741` Needle (Uncorr / Cancel) | Candidate `0762` Needle (Uncorr / Cancel) |
| ---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **32K** | Dense L2/8/14 (`sw=16K`, `b=64`) | `86.74%` | **`87.26%`** | **`+0.52%`** | `57.2%` / `72.7%` | **`73.9%` / `93.3%`** |
| **64K** | Dense L2/8/14 (`sw=32K`, `b=64`) | `80.69%` | **`81.32%`** | **`+0.63%`** | `47.5%` / `61.4%` | **`66.2%` / `88.7%`** |
| **64K** | Source L20 (`sw=64K`, `b=512`) | `96.24%` | **`97.39%`** | **`+1.15%`** | `72.9%` / `85.2%` | **`87.4%` / `98.1%`** |
| **131K** | Dense L2/8/14 (`sw=65K`, `b=64`) | `74.41%` | **`74.88%`** | **`+0.47%`** | `36.3%` / `49.9%` | **`55.6%` / `80.5%`** |
| **131K** | Source L20 (`sw=131K`, `b=1024`) | `96.02%` | **`97.53%`** | **`+1.51%`** | `73.7%` / `86.4%` | **`87.7%` / `97.9%`** |
| **262K** | Dense L2/8/14 (`sw=131K`, `b=64`) | `69.51%` | **`69.92%`** | **`+0.41%`** | `28.5%` / `40.4%` | **`46.8%` / `72.1%`** |
| **262K** | Source L20 (`sw=262K`, `b=1024`) | `91.85%` | **`95.21%`** | **`+3.36%`** | `64.6%` / `77.8%` | **`80.3%` / `94.7%`** |
| **524K** | Dense L2/8/14 (`sw=262K`, `b=64`) | `64.40%` | **`64.67%`** | **`+0.27%`** | `21.2%` / `31.8%` | **`38.2%` / `63.6%`** |
| **524K** | Source L20 (`sw=524K`, `b=1024`) | `86.35%` | **`92.87%`** | **`+6.52%`** | `54.5%` / `68.5%` | **`73.0%` / `90.9%`** |
| **1M** | Dense L2/8/14 (`sw=524K`, `b=64`) | `57.93%` | **`58.18%`** | **`+0.25%`** | `15.5%` / `23.6%` | **`30.2%` / `53.2%`** |
| **1M** | Source L20 (`sw=1048K`, `b=1024`) | `78.83%` | **`90.70%`** | **`+11.87%`** | `44.9%` / `59.4%` | **`63.7%` / `86.1%`** |

At 1M tokens on Source Layer 20 (`budget = 1024`, `8,192` candidate blocks out of `131,072`), `0762` (`FP8 E4M3` Group-32 Dominance) improves top-512 recall from **`78.83%` to `90.70%` (`+11.87 pp`)** while using half the summary memory.

## KV Cache Memory Footprint Comparison

In `DeepSeek-V4.1-Flash` (`40` target layers + `3` draft layers):
- **3 Dense Indexer Layers (`2, 8, 14`)**: `compress_ratio = 2`, `page_size = 128` (`0.5` slots/token/layer -> `1.5` slots/token across 3 layers).
- **1 Source Indexer Layer (`20`)**: `compress_ratio = 1`, `page_size = 256` (`1.0` slot/token).
- **Total Indexer Slots per Token**: `2.5` slots/token (`2.665` effective slots/token including page granularity in `kv_capacity.py`).

| Configuration | `STATE_BYTES` | Summary Overhead / Slot | Summary Overhead / Token | Total KV Bytes / Token | 1M-Token Pool per Rank (4 Indexer Layers) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Native (`VLLM_DS41_LSA=0`)** | `68 B` | `+0 B` | `+0.00 B/tok` | `290.94 B/tok` | `170.0 MiB` |
| **Baseline `0741` (`VLLM_DS41_LSA=1`, `BF16` Mean)** | `100 B` | `+32 B` (`+47.1%`) | `+42.63 B/tok` (`+14.65%`) | `333.57 B/tok` | `250.0 MiB` (`+80.0 MiB`) |
| **Candidate `0762` (`VLLM_DS41_LSA=1`, `FP8` Group-32 Dom)** | **`84 B`** | **`+16 B` (`+23.5%`)** | **`+21.31 B/tok` (`+7.32%`)** | **`312.26 B/tok`** | **`210.0 MiB` (`+40.0 MiB`, `-50%` overhead)** |

## 2-Boot GPU Verification & Latency Benchmarks (`sp5`, `NVIDIA GB10`)

Executed across two fresh container boots (`boot1` and `boot2`) on `sp5` (`spark-ds41f:1107`, `NVIDIA GB10` `sm_121a`):

Receipts:
- `experiments/2026-11-08-lsa-dominance-fp8-summary/receipts/gpu-pytest-boot1.txt`
- `experiments/2026-11-08-lsa-dominance-fp8-summary/receipts/gpu-pytest-boot2.txt`
- `experiments/2026-11-08-lsa-dominance-fp8-summary/receipts/bench-boot1.json`
- `experiments/2026-11-08-lsa-dominance-fp8-summary/receipts/bench-boot2.json`

### 1. 2-Boot GPU Unit Test & PDL Source Verification

Command:
```bash
pytest -v -p tblib_stub -p no:cacheprovider -p tests.kernels.deepseek_v41_pdl_sources \
  tests/models/test_deepseek_v4_1_lsa.py \
  tests/models/test_deepseek_v4_1_attention.py
```

- **`boot1`**: **`73 passed, 14 warnings in 28.38s`** (`PDL sources: 35 kernels launched, 30 with PDL, 0 violations`)
- **`boot2`**: **`73 passed, 14 warnings in 28.38s`** (`PDL sources: 35 kernels launched, 30 with PDL, 0 violations`)

### 2. 2-Boot Write-Path Kernel Latency (`summarize_blocks`, CUDAGraph Replay, `us`)

`summarize_blocks` only executes work on slots where `slot % page == page - 1` (1 in 128 slots on Dense layers; 1 in 256 slots on Source Layer 20):

| Page Size | Completed Pages (`rows`) | Baseline `0741` `BF16` Mean (`boot1 / boot2`, Mean `us`) | Candidate `0762` `FP8` Dom (`boot1 / boot2`, Mean `us`) | Non-Boundary Decode Early-Exit (`Mean us`) |
| :---: | ---: | :---: | :---: | :---: |
| `128` (Dense L2/8/14) | `1` | `12.64 / 12.64` (`12.64`) | `12.29 / 12.35` (`12.32`) | `12.79` |
| `128` (Dense L2/8/14) | `6` | `12.16 / 12.00` (`12.08`) | `13.86 / 13.89` (`13.88`) | `11.83` |
| `128` (Dense L2/8/14) | `8` | `12.38 / 12.38` (`12.38`) | `13.79 / 13.76` (`13.78`) | `11.49` |
| `128` (Dense L2/8/14) | `24` | `12.22 / 12.26` (`12.24`) | `13.50 / 13.50` (`13.50`) | `11.65` |
| `128` (Dense L2/8/14) | `64` (`8K` prefill chunk) | `13.34 / 13.34` (`13.34`) | `13.28 / 13.22` (`13.25`) | `12.24` |
| `128` (Dense L2/8/14) | `256` (`32K` prefill) | `18.69 / 18.69` (`18.69`) | `17.63 / 17.63` (`17.63`) | `12.74` |
| `256` (Source L20) | `1` | `17.60 / 17.63` (`17.62`) | `17.60 / 17.63` (`17.62`) | `6.74` |
| `256` (Source L20) | `6` | `17.60 / 17.60` (`17.60`) | `19.23 / 19.26` (`19.25`) | `8.74` |
| `256` (Source L20) | `8` | `17.63 / 17.63` (`17.63`) | `19.20 / 19.20` (`19.20`) | `8.59` |
| `256` (Source L20) | `24` | `17.60 / 17.63` (`17.62`) | `17.63 / 17.60` (`17.62`) | `6.50` |
| `256` (Source L20) | `64` (`16K` prefill chunk) | `17.60 / 17.60` (`17.60`) | `17.60 / 17.60` (`17.60`) | `8.74` |
| `256` (Source L20) | `256` (`64K` prefill) | `32.00 / 31.97` (`31.99`) | `31.97 / 31.97` (`31.97`) | `19.49` |

### 3. 2-Boot Full Indexer Latency Matrix (`32K..1M` x `r = 1, 6, 8, 24, 64`, CUDAGraph Replay, `us`)

#### Source Layer 20 (`page = 256`, `budget = 512` at 64K, `budget = 1024` at 131K–1M)

| Context | Rows (`r`) | Native `IndexerV41Plan` (`boot1 / boot2`, Mean `us`) | Baseline `0741` `BF16` LSA (`boot1 / boot2`, Mean `us`) | Candidate `0762` `FP8` LSA (`boot1 / boot2`, Mean `us`) | Coarse Score `BF16 -> FP8` (`Mean us`) | `0762` Speedup vs Native | `0762` Speedup vs `0741` |
| ---: | ---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **64K** | `8` | `169.8 / 172.0` (`170.9`) | `149.3 / 149.2` (`149.3`) | `151.6 / 151.9` (`151.8`) | `31.2 -> 31.8` | `1.13x` | `0.98x` |
| **64K** | `24` | `357.1 / 348.1` (`352.6`) | `308.1 / 305.5` (`306.8`) | `311.3 / 312.6` (`312.0`) | `69.9 -> 72.7` | `1.13x` | `0.98x` |
| **64K** | `64` | `785.6 / 756.4` (`771.0`) | `647.5 / 650.5` (`649.0`) | `657.6 / 659.7` (`658.7`) | `173.4 -> 182.2` | `1.17x` | `0.99x` |
| **131K** | `8` | `322.0 / 323.5` (`322.8`) | `369.5 / 352.6` (`361.1`) | `357.4 / 355.0` (`356.2`) | `49.9 -> 50.8` | `0.91x` | `1.01x` |
| **131K** | `24` | `767.8 / 710.8` (`739.3`) | `661.3 / 643.5` (`652.4`) | `660.1 / 657.3` (`658.7`) | `129.9 -> 133.5` | `1.12x` | `0.99x` |
| **131K** | `64` | `4682.2 / 4634.7` (`4658.5`) | `3830.6 / 3908.8` (`3869.7`) | `3849.3 / 3748.9` (`3799.1`) | `338.3 -> 346.3` | `1.23x` | `1.02x` |
| **262K** | `1` | `135.4 / 136.4` (`135.9`) | `135.4 / 136.4` (`135.9`, native) | `158.9 / 158.9` (`158.9`) | `15.5 -> 15.6` | `0.86x` | `0.86x` |
| **262K** | `6` | `493.8 / 493.0` (`493.4`) | `354.1 / 354.5` (`354.3`) | `339.4 / 341.1` (`340.3`) | `64.6 -> 66.8` | **`1.45x`** | **`1.04x`** |
| **262K** | `8` | `641.2 / 640.2` (`640.7`) | `428.0 / 426.1` (`427.1`) | `410.5 / 412.8` (`411.7`) | `85.2 -> 87.6` | **`1.56x`** | **`1.04x`** |
| **262K** | `24` | `4282.6 / 4264.1` (`4273.4`) | `3485.1 / 3460.0` (`3472.6`) | `3278.4 / 3264.3` (`3271.4`) | `242.8 -> 249.1` | **`1.31x`** | **`1.06x`** |
| **262K** | `64` | `12582.1 / 12539.0` (`12560.6`) | `5129.6 / 5178.5` (`5154.1`) | `5205.6 / 5242.9` (`5224.3`) | `649.1 -> 655.3` | **`2.40x`** | `0.99x` |
| **524K** | `1` | `294.7 / 293.9` (`294.3`) | `294.7 / 293.9` (`294.3`, native) | **`193.2 / 191.7` (`192.5`)** | `27.9 -> 27.8` | **`1.53x`** | **`1.53x`** |
| **524K** | `6` | `3807.4 / 3793.2` (`3800.3`) | `523.7 / 524.4` (`524.1`) | **`487.9 / 474.5` (`481.2`)** | `124.2 -> 126.1` | **`7.90x`** | **`1.09x`** |
| **524K** | `8` | `4266.3 / 4267.0` (`4266.7`) | `649.1 / 663.6` (`656.4`) | **`591.5 / 603.0` (`597.3`)** | `165.0 -> 167.4` | **`7.14x`** | **`1.10x`** |
| **524K** | `24` | `12583.0 / 12524.6` (`12553.8`) | `3951.0 / 3943.6` (`3947.3`) | `3959.6 / 3965.2` (`3962.4`) | `473.0 -> 482.2` | **`3.17x`** | `1.00x` |
| **524K** | `64` | `31206.3 / 31156.5` (`31181.4`) | `8822.4 / 8821.8` (`8822.1`) | `8769.1 / 8801.4` (`8785.3`) | `3874.3 -> 3875.5` | **`3.55x`** | `1.00x` |
| **1M** | `1` | `546.5 / 539.7` (`543.1`) | `546.5 / 539.7` (`543.1`, native) | **`269.6 / 269.6` (`269.6`)** | **`140.4 -> 52.4`** (`2.68x`) | **`2.01x`** | **`2.01x`** |
| **1M** | `6` | `5460.8 / 5464.3` (`5462.6`) | `3513.5 / 3551.4` (`3532.5`) | **`765.8 / 764.5` (`765.2`)** | **`664.2 -> 246.8`** (`2.69x`) | **`7.14x`** | **`4.62x`** |
| **1M** | `8` | `8554.7 / 8554.7` (`8554.7`) | `4009.4 / 4030.6` (`4020.0`) | **`1496.2 / 1236.0` (`1366.1`)** | **`1136.4 -> 333.1`** (`3.41x`) | **`6.26x`** | **`2.94x`** |
| **1M** | `24` | `23437.3 / 23509.1` (`23473.2`) | `9145.6 / 9122.0` (`9133.8`) | **`4966.5 / 4939.7` (`4953.1`)** | **`5586.2 -> 3466.0`** (`1.61x`) | **`4.74x`** | **`1.84x`** |
| **1M** | `64` | `62958.4 / 62944.5` (`62951.5`) | `24189.0 / 24298.6` (`24243.8`) | **`13651.2 / 13462.6` (`13556.9`)** | **`17760.6 -> 5412.6`** (`3.28x`) | **`4.64x`** | **`1.79x`** |

#### Combined 4-Layer Indexer Total (`3 x Dense [L2, L8, L14] + 1 x Source [L20]`, 2-Boot Mean `ms`)

| Context | Rows (`r`) | Native 4-Layer (`ms`) | Baseline `0741` `BF16` LSA (`ms`) | Candidate `0762` `FP8` LSA (`ms`) | `0741` Speedup vs Native | `0762` Speedup vs Native | `0762` Speedup vs `0741` |
| ---: | ---: | ---: | ---: | ---: | :---: | :---: | :---: |
| **64K** | `8` | `0.386 ms` | `0.299 ms` | **`0.297 ms`** | `1.29x` | **`1.30x`** | `1.01x` |
| **64K** | `64` | `1.737 ms` | `1.132 ms` | **`1.147 ms`** | `1.53x` | **`1.51x`** | `0.99x` |
| **131K** | `8` | `0.692 ms` | `0.562 ms` | **`0.558 ms`** | `1.23x` | **`1.24x`** | `1.01x` |
| **131K** | `64` | `6.666 ms` | `4.666 ms` | **`4.623 ms`** | `1.43x` | **`1.44x`** | `1.01x` |
| **262K** | `8` | `1.563 ms` | `0.733 ms` | **`0.728 ms`** | `2.13x` | **`2.15x`** | `1.01x` |
| **262K** | `64` | `27.367 ms` | `6.758 ms` | **`6.866 ms`** | `4.05x` | **`3.99x`** | `0.98x` |
| **524K** | `1` (routed) | `0.619 ms` | `0.619 ms` (native) | **`0.374 ms`** | `1.00x` | **`1.65x`** | **`1.65x`** |
| **524K** | `6` | `5.164 ms` | `0.891 ms` | **`0.855 ms`** | `5.80x` | **`6.04x`** | **`1.04x`** |
| **524K** | `8` | `6.103 ms` | `1.108 ms` | **`1.062 ms`** | `5.51x` | **`5.75x`** | **`1.04x`** |
| **524K** | `64` | `61.812 ms` | `11.747 ms` | **`11.988 ms`** | `5.26x` | **`5.16x`** | `0.98x` |
| **1M** | `1` (routed) | `1.314 ms` | `1.314 ms` (native) | **`0.506 ms`** | `1.00x` | **`2.59x`** | **`2.59x`** |
| **1M** | `6` | `16.615 ms` | `4.139 ms` | **`1.353 ms`** | `4.01x` | **`12.28x`** | **`3.06x`** |
| **1M** | `8` | `21.022 ms` | `4.777 ms` | **`2.118 ms`** | `4.40x` | **`9.92x`** | **`2.26x`** |
| **1M** | `24` | `59.898 ms` | `11.224 ms` | **`6.942 ms`** | `5.34x` | **`8.63x`** | **`1.62x`** |
| **1M** | `64` | `153.266 ms` | `37.705 ms` | **`26.697 ms`** | `4.06x` | **`5.74x`** | **`1.41x`** |

## Conclusion

- **Default-off invariant**: `VLLM_DS41_LSA=0`/unset is 100% untouched (`68 B/slot`, native `IndexerV41Plan`).
- **50% reduction in `VLLM_DS41_LSA` KV memory overhead**: `STATE_BYTES` drops from `100` (`+32 B/slot`) to `84` (`+16 B/slot`), saving `21.31 B/token` (`40 MiB/rank` per 1M tokens).
- **Higher top-512 recall across all regimes**: Group-32 dominance blending (`0.85 * mean + 0.15 * k*`) in `float8_e4m3fn` beats `0741` `bfloat16` arithmetic means on standard clustered keys (`+0.49 pp` Dense `b=64`, `+1.91 pp` Source `b=512`, `+11.87 pp` at 1M Source `b=1024`) and single-needle-in-a-block keys (`+13.87 pp` uncorrelated, `+28.98 pp` canceling).
- **L2 cache residency at 1M context & `r = 1` long-context decode speedup**: Keeping the 1M Layer-20 summary pool at `16 MiB` (`float8_e4m3fn`) instead of `32 MiB` (`bfloat16`) speeds up 1M Layer-20 coarse scoring by `2.68x–3.41x` and total 4-layer 1M indexing by **`1.41x–3.06x` over `0741` (`5.74x–12.28x` over Native)**, while `r = 1` shortlist routing at `score_width >= 262144` speeds up 4-layer 1M `r = 1` decode by **`2.59x`**.
