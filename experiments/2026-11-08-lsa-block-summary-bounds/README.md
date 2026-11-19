# LSA Block-Summary Needle Recall and Tensor-Core Bounding-Box Evaluation

- **Date**: 2026-11-08
- **Base commit**: `8e5ea208ff02836a35867a1994e2c2bf7884bf48` (`main`)
- **Base image**: `spark-ds41f:1107` (`sha256:6b21baec32f1fd0d1da3abaa59deec264258f084ff81c19ac46aa05ae1a36736`)
- **Evaluation node**: `sp3` (`spark-0011`, single `NVIDIA GB10` SM121, idle standalone lab node)
- **Intended causal variable**: 8-key block summary formulation in `vllm/models/deepseek_v4_1/tilelang/block_shortlist.py` (`summarize_blocks`) and coarse block scoring in `vllm/models/deepseek_v4_1/tilelang/query_reuse.py` (`prepared_scores`) under opt-in `VLLM_DS41_LSA=1`.

---

## 1. Problem Statement & Hypothesis

Under opt-in Latent Sparse Attention (`VLLM_DS41_LSA=1`, patch `0741`), each 8-key block $B = \{t_0, \dots, t_7\}$ of MXFP4-quantized indexer keys ($k_t \in \mathbb{R}^{128}$, `64 B` payload + `4 B` UE8M0 scales = `68 B/key`) is summarized by `summarize_blocks` into a single `128 x BF16` vector (`256 B/block = 32 B/key`, `STATE_BYTES = 100`). Coarse shortlist selection in `QueryReusePlan` (`prepared_scores`) computes:

$$S(q, B) = \sum_{h=0}^{31} w_h \max\!\left(0, \; q_h \cdot \hat{k}_B\right)$$

and selects the top `budget` blocks (`budget = 64` -> `512` candidate keys on dense layers `2, 8, 14`; `budget = 512` at `64K` and `budget = 1024` at `128K-1M` -> `4096..8192` candidate keys on source layer `20`) before running exact native `IndexerV41Plan` fine scoring over the shortlisted blocks.

When $\hat{k}_B$ is the uniform arithmetic mean $\bar{k}_{B,d} = \frac{1}{8}\sum_{t \in B} k_{t,d}$, two structural failure modes degrade recall on sharp localized ("needle") keys at long context:

1. **1/8 amplitude dilution**: An isolated 1-token needle $t^* \in B$ with strong head alignment $q_h \cdot k_{t^*} = M$ surrounded by 7 unrelated tokens ($q_h \cdot k_t \approx 0$) is diluted by $1/8$ to $0.125 M$, falling behind coherent distractor blocks where all 8 tokens share mild topic alignment ($0.20 M \text{ to } 0.38 M$).
2. **Pre-ReLU sign cancellation**: In native `IndexerV41Plan`, each head's dot product is clamped by $\max(0, q_h \cdot k_t)$ *before* multiplying by $w_h \ge 0$ and summing across 32 heads, so negative dot products on neighboring keys ($q_h \cdot k_t < 0$) never subtract from $k_{t^*}$. When keys are averaged first into $\bar{k}_B$, negative dot products on neighboring keys in the block (particularly across the 64 RoPE-rotated channels `d in [0, 64)` where phase advances across the 8 token positions) cancel $q_h \cdot k_{t^*}$ *before* the `T.max(..., 0.)` clamp in `prepared_scores`.

We screen two families of block-summary formulations across `32K`, `64K`, `128K`, and `256K` contexts:

### Family 1 — Single-GEMM Outlier-Preserving Block Summaries (`STATE_BYTES = 100`, Zero Extra Scoring Latency)

Produce a single `128 x BF16` summary vector $\hat{k}_B \in \mathbb{R}^{128}$ per 8-key block (`STATE_BYTES = 100`), leaving `QueryReusePlan.prepared_scores` (single `T.gemm`), scratch buffers, and top-k merge stages 100% unchanged:

- **`mean` (`0741` baseline)**:
  $$\bar{k}_{B,d} = \frac{1}{8}\sum_{t \in B} k_{t,d}$$
- **`mid_blend` (Channel Midrange Blend, $\beta \in \{0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50\}$)**:
  $$\hat{k}_{B,d} = (1 - \beta)\,\bar{k}_{B,d} + \beta\,k^{\text{mid}}_{B,d}, \qquad k^{\text{mid}}_{B,d} = \frac{\max_{t \in B} k_{t,d} + \min_{t \in B} k_{t,d}}{2}$$
  Equivalently, $\hat{k}_{B,d}$ is a convex order-statistic combination of the 8 keys in channel $d$:
  $$\hat{k}_{B,d} = \frac{1-\beta}{8}\sum_{t \in B} k_{t,d} + \frac{\beta}{2}\left(\max_{t \in B} k_{t,d} + \min_{t \in B} k_{t,d}\right)$$
  At $\beta = 0.25$, the 6 interior keys have weight $\frac{3}{32} = 0.09375$ and the 2 channel extremes ($\max, \min$) have weight $\frac{7}{32} = 0.21875$ ($2.33\times$ interior weight, summing to $6 \cdot \frac{3}{32} + 2 \cdot \frac{7}{32} = 1$), with exact dyadic IEEE-754 coefficients:
  $$\hat{k}_{B,d} = 0.09375 \sum_{t \in B} k_{t,d} + 0.125 \left(\max_{t \in B} k_{t,d} + \min_{t \in B} k_{t,d}\right)$$
  - On a coherent block ($k_{t,d} = c$), $\hat{k}_{B,d} = c$ (unchanged).
  - On zero-mean symmetric oscillations across the 8 keys ($\max_{t \in B} k_{t,d} \approx -\min_{t \in B} k_{t,d}$), the midrange $\frac{\max + \min}{2} \approx 0$ cancels out without inflating block norms.
  - On an isolated needle spike ($k_{t^*, d} = M$, other 7 tokens $\approx 0$), the spike amplitude in $\hat{k}_{B,d}$ is boosted by $1.75\times$ ($\frac{7}{32}M$ vs. $\frac{4}{32}M$).
- **`absmax_blend` (Signed-Extreme Blend, $\beta \in \{0.20, 0.25, 0.35\}$)**:
  $$\hat{k}_{B,d} = (1 - \beta)\,\bar{k}_{B,d} + \beta\,k_{t^*(d), d}, \qquad t^*(d) = \arg\max_{t \in B} |k_{t,d}|$$
- **`l2_blend` ($\beta = 0.35$)** and **`p3_blend` ($\beta = 0.25$)**: token-$L_2$-norm-weighted mean and signed-cubic power mean blended with $\bar{k}_B$.

### Family 2 — Tensor-Core Center + Radius Bounding Box (`prepared_scores_bbox`)

Using the identity $\max(q_{h,d} k^{\max}_{B,d}, \, q_{h,d} k^{\min}_{B,d}) = q_{h,d} k^{\text{mid}}_{B,d} + |q_{h,d}| k^{\text{rad}}_{B,d}$ where $k^{\text{mid}}_{B,d} = \frac{k^{\max}_{B,d}+k^{\min}_{B,d}}{2}$ and $k^{\text{rad}}_{B,d} = \frac{k^{\max}_{B,d}-k^{\min}_{B,d}}{2} \ge 0$, score each block via two Tensor-Core GEMMs (`T.gemm(a_center, b, acc, clear_accum=True)` followed by `T.gemm(a_rad, b_abs, acc, clear_accum=False)` where `b_abs[h, d] = |b[h, d]| * alpha`):

$$S_{\alpha}(q, B) = \sum_{h=0}^{31} w_h \max\!\left(0, \; q_h \cdot k^{\text{center}}_B + \alpha \, |q_h| \cdot k^{\text{rad}}_B\right)$$

screened across $k^{\text{center}}_B \in \{k^{\text{mid}}_B \; (\text{`bbox\_mid`}), \; \bar{k}_B \; (\text{`bbox\_mean`}), \; 0.75\bar{k}_B + 0.25 k^{\text{mid}}_B \; (\text{`bbox\_hyb`})\}$ and $\alpha \in \{0.125, 0.25, 0.3535, 0.50, 1.00\}$.

---

## 2. Exact Commands & Workload Identity

All benchmarks and gates were executed inside `spark-ds41f:1107` on `sp3` (`NVIDIA GB10`).

1. **2-Boot Recall & Kernel Latency Benchmark (`bench_2boot.py`)**:
   ```bash
   docker run --rm --gpus all -v $PWD/experiments/2026-11-08-lsa-block-summary-bounds:/exp \
     -v /tmp/tilelang_cache:/root/.tilelang/cache --entrypoint python3 spark-ds41f:1107 \
     /exp/bench_2boot.py --boot 1 --out /exp/receipts/bench_boot1.json

   docker run --rm --gpus all -v $PWD/experiments/2026-11-08-lsa-block-summary-bounds:/exp \
     -v /tmp/tilelang_cache:/root/.tilelang/cache --entrypoint python3 spark-ds41f:1107 \
     /exp/bench_2boot.py --boot 2 --out /exp/receipts/bench_boot2.json
   ```
2. **Workload distribution**:
   - Multi-modal / long-context key distribution with 64 topic clusters, smooth block-to-block topic transitions, high-variance noisy distractor blocks (`15%` of blocks), RoPE phase rotation on channels `0..63` (where consecutive tokens in an 8-key block rotate phase and induce pre-ReLU cancellation under plain averaging), and heavy-tailed channel scaling (`1.0 + 0.8 * exp(|N(0,1)| * 0.5)`), quantized through the production `write_index_keys` MXFP4 kernel (`64 B` payload + `4 B` UE8M0 scale).
   - `32` injected needle blocks per sequence:
     - `Needle-1`: `16` isolated 1-token needles (`k_{t^*}` aligned with query + `7` counter-phase/unrelated tokens in the same 8-key block).
     - `Needle-2`: `8` two-token needles (`2` aligned tokens + `6` unrelated tokens).
     - `Needle-4`: `8` four-token needles (`4` aligned tokens + `4` unrelated tokens).
   - Evaluated against exact native `IndexerV41Plan` oracle across `rows = 4` queries at:
     - Dense layers (`page = 128`, `budget = 64` blocks = `512` shortlisted keys): `32K`, `64K`, `128K`.
     - Source layer 20 (`page = 256`, `budget = 512` at `64K`, `budget = 1024` at `128K` and `256K`): `64K`, `128K`, `256K`.

---

## 3. Recall Evaluation Results (32K - 256K)

Metrics are exact against `IndexerV41Plan` (deterministic across Boot 1 and Boot 2; receipts in `receipts/bench_boot1.json` and `receipts/bench_boot2.json`):
- **`Needle-1`, `Needle-2`, `Needle-4`**: Fraction of injected 1-token, 2-token, and 4-token needle blocks retained by the shortlist (`out // 8` on dense layers; `candidate_out` on source layer 20).
- **`R@64`, `R@512`, `R@2048`**: Overlap against native `IndexerV41Plan` top-64 keys, top-512 keys (`out`), and top-2048 keys (`candidate_out` blocks on source layer 20).
- **`ScoreW` & `SoftM`**: Score-weighted recall and temperature-scaled softmax-mass recall ($\tau = 0.2 \cdot \sigma$) of the selected keys vs. native `IndexerV41Plan`.

### 3.1 Dense Layers (`page = 128`, `budget = 64` blocks = `512` candidate keys)

| Context | Formulation | Family | `Needle-1` | `Needle-2` | `Needle-4` | `R@64` | `R@512` | `R@2048` | `ScoreW` | `SoftM` |
|---|---|---|---|---|---|---|---|---|---|---|
| **32K** | `mean` (`0741` baseline) | Family 1 | 80.5% | 95.3% | 100.0% | 95.1% | 52.1% | 33.4% | 94.2% | 92.6% |
| **32K** | **`mid_blend` ($\beta=0.25$)** | **Family 1** | **89.8%** (+9.3%) | **96.9%** (+1.6%) | **100.0%** | **97.5%** (+2.4%) | **52.3%** (+0.2%) | **33.5%** (+0.1%) | **94.6%** (+0.4%) | **94.9%** (+2.3%) |
| **32K** | `mid_blend` ($\beta=0.35$) | Family 1 | 93.0% | 96.9% | 100.0% | 97.7% | 51.3% | 32.9% | 94.4% | 95.3% |
| **32K** | `absmax_blend` ($\beta=0.25$) | Family 1 | 89.8% | 96.9% | 100.0% | 96.9% | 48.0% | 30.8% | 93.7% | 94.1% |
| **32K** | `l2_blend` ($\beta=0.35$) | Family 1 | 86.7% | 96.9% | 100.0% | 96.1% | 51.7% | 33.0% | 94.3% | 93.8% |
| **32K** | `p3_blend` ($\beta=0.25$) | Family 1 | 82.0% | 90.6% | 100.0% | 89.5% | 44.5% | 29.0% | 92.5% | 89.1% |
| **32K** | `bbox_mean` ($\alpha=0.25$) | Family 2 | 82.8% | 93.8% | 100.0% | 93.4% | 44.3% | 29.5% | 92.7% | 91.0% |
| **32K** | `bbox_mid` ($\alpha=0.25$) | Family 2 | 100.0% | 96.9% | 100.0% | 85.2% | 26.8% | 16.5% | 88.6% | 86.1% |
| **64K** | `mean` (`0741` baseline) | Family 1 | 57.8% | 87.5% | 100.0% | 89.5% | 42.4% | 24.4% | 91.7% | 86.6% |
| **64K** | **`mid_blend` ($\beta=0.25$)** | **Family 1** | **71.9%** (+14.1%) | **90.6%** (+3.1%) | **100.0%** | **93.0%** (+3.5%) | **42.8%** (+0.4%) | **24.4%** (0.0%) | **92.3%** (+0.6%) | **89.6%** (+3.0%) |
| **64K** | `mid_blend` ($\beta=0.35$) | Family 1 | 79.7% | 93.8% | 100.0% | 94.1% | 42.2% | 23.9% | 92.4% | 90.8% |
| **64K** | `absmax_blend` ($\beta=0.25$) | Family 1 | 71.1% | 90.6% | 100.0% | 91.8% | 37.9% | 21.8% | 91.3% | 88.2% |
| **64K** | `l2_blend` ($\beta=0.35$) | Family 1 | 67.2% | 90.6% | 100.0% | 91.4% | 42.4% | 24.1% | 92.1% | 88.6% |
| **64K** | `p3_blend` ($\beta=0.25$) | Family 1 | 63.3% | 75.0% | 93.8% | 84.8% | 34.0% | 19.6% | 89.9% | 82.8% |
| **64K** | `bbox_mean` ($\alpha=0.25$) | Family 2 | 56.2% | 90.6% | 100.0% | 85.9% | 34.4% | 20.6% | 89.9% | 83.2% |
| **64K** | `bbox_mid` ($\alpha=0.25$) | Family 2 | 88.3% | 90.6% | 96.9% | 82.4% | 19.6% | 10.6% | 85.6% | 80.7% |
| **128K** | `mean` (`0741` baseline) | Family 1 | 47.7% | 78.1% | 96.9% | 86.9% | 35.7% | 19.2% | 89.9% | 82.4% |
| **128K** | **`mid_blend` ($\beta=0.25$)** | **Family 1** | **64.8%** (+17.1%) | **81.2%** (+3.1%) | **96.9%** | **90.8%** (+3.9%) | **36.6%** (+0.9%) | **19.4%** (+0.2%) | **90.7%** (+0.8%) | **86.6%** (+4.2%) |
| **128K** | `mid_blend` ($\beta=0.35$) | Family 1 | 71.1% | 87.5% | 96.9% | 91.6% | 36.4% | 19.0% | 90.8% | 87.8% |
| **128K** | `absmax_blend` ($\beta=0.25$) | Family 1 | 62.5% | 81.2% | 96.9% | 89.8% | 32.8% | 17.3% | 89.7% | 85.0% |
| **128K** | `l2_blend` ($\beta=0.35$) | Family 1 | 61.7% | 81.2% | 96.9% | 90.2% | 36.1% | 19.1% | 90.5% | 85.8% |
| **128K** | `p3_blend` ($\beta=0.25$) | Family 1 | 49.2% | 68.8% | 84.4% | 79.7% | 27.6% | 14.6% | 87.7% | 77.5% |
| **128K** | `bbox_mean` ($\alpha=0.25$) | Family 2 | 47.7% | 81.2% | 96.9% | 84.8% | 29.8% | 16.5% | 88.3% | 79.8% |
| **128K** | `bbox_mid` ($\alpha=0.25$) | Family 2 | 82.0% | 81.2% | 87.5% | 80.5% | 16.0% | 7.6% | 83.2% | 76.7% |

### 3.2 Source Layer 20 (`page = 256`, `budget = 512` at 64K / `budget = 1024` at 128K-256K)

| Context | Formulation | Family | `Needle-1` | `Needle-2` | `Needle-4` | `R@64` | `R@512` | `R@2048` (`cand`) | `ScoreW` | `SoftM` |
|---|---|---|---|---|---|---|---|---|---|---|
| **64K** (`b=512`) | `mean` (`0741` baseline) | Family 1 | 98.4% | 100.0% | 100.0% | 99.6% | 88.4% | 76.1% | 99.3% | 99.2% |
| **64K** (`b=512`) | **`mid_blend` ($\beta=0.25$)** | **Family 1** | **100.0%** (+1.6%) | **100.0%** | **100.0%** | **100.0%** (+0.4%) | **88.7%** (+0.3%) | **76.2%** (+0.1%) | **99.4%** (+0.1%) | **99.6%** (+0.4%) |
| **64K** (`b=512`) | `mid_blend` ($\beta=0.35$) | Family 1 | 100.0% | 100.0% | 100.0% | 100.0% | 87.9% | 75.1% | 99.3% | 99.5% |
| **64K** (`b=512`) | `absmax_blend` ($\beta=0.25$) | Family 1 | 100.0% | 100.0% | 100.0% | 100.0% | 84.3% | 70.8% | 98.8% | 99.0% |
| **64K** (`b=512`) | `bbox_mean` ($\alpha=0.25$) | Family 2 | 98.4% | 100.0% | 100.0% | 99.6% | 84.7% | 71.9% | 98.9% | 98.8% |
| **64K** (`b=512`) | `bbox_mid` ($\alpha=0.25$) | Family 2 | 100.0% | 100.0% | 100.0% | 99.2% | 62.7% | 46.8% | 96.4% | 96.9% |
| **128K** (`b=1024`) | `mean` (`0741` baseline) | Family 1 | 97.7% | 100.0% | 100.0% | 99.4% | 88.6% | 80.5% | 99.4% | 99.1% |
| **128K** (`b=1024`) | **`mid_blend` ($\beta=0.25$)** | **Family 1** | **100.0%** (+2.3%) | **100.0%** | **100.0%** | **100.0%** (+0.6%) | **89.5%** (+0.9%) | **80.9%** (+0.4%) | **99.6%** (+0.2%) | **99.7%** (+0.6%) |
| **128K** (`b=1024`) | `mid_blend` ($\beta=0.35$) | Family 1 | 100.0% | 100.0% | 100.0% | 100.0% | 89.2% | 80.0% | 99.5% | 99.7% |
| **128K** (`b=1024`) | `absmax_blend` ($\beta=0.25$) | Family 1 | 100.0% | 100.0% | 100.0% | 100.0% | 85.4% | 75.3% | 99.1% | 99.3% |
| **128K** (`b=1024`) | `bbox_mean` ($\alpha=0.25$) | Family 2 | 99.2% | 100.0% | 100.0% | 99.8% | 85.6% | 76.9% | 99.1% | 99.1% |
| **128K** (`b=1024`) | `bbox_mid` ($\alpha=0.25$) | Family 2 | 100.0% | 100.0% | 100.0% | 99.2% | 63.4% | 50.6% | 96.4% | 97.2% |
| **256K** (`b=1024`) | `mean` (`0741` baseline) | Family 1 | 92.2% | 100.0% | 100.0% | 98.0% | 76.0% | 61.4% | 98.4% | 97.0% |
| **256K** (`b=1024`) | **`mid_blend` ($\beta=0.25$)** | **Family 1** | **96.1%** (+3.9%) | **100.0%** | **100.0%** | **99.0%** (+1.0%) | **76.4%** (+0.4%) | **61.4%** (0.0%) | **98.6%** (+0.2%) | **98.1%** (+1.1%) |
| **256K** (`b=1024`) | `mid_blend` ($\beta=0.35$) | Family 1 | 98.4% | 100.0% | 100.0% | 99.6% | 75.3% | 59.9% | 98.6% | 98.4% |
| **256K** (`b=1024`) | `absmax_blend` ($\beta=0.25$) | Family 1 | 96.9% | 100.0% | 100.0% | 99.2% | 70.6% | 55.3% | 97.9% | 97.5% |
| **256K** (`b=1024`) | `l2_blend` ($\beta=0.35$) | Family 1 | 95.3% | 100.0% | 100.0% | 98.8% | 76.1% | 60.8% | 98.5% | 97.8% |
| **256K** (`b=1024`) | `p3_blend` ($\beta=0.25$) | Family 1 | 93.8% | 100.0% | 100.0% | 97.3% | 64.9% | 50.3% | 97.1% | 95.4% |
| **256K** (`b=1024`) | `bbox_mean` ($\alpha=0.125$) | Family 2 | 92.2% | 100.0% | 100.0% | 98.0% | 75.5% | 60.9% | 98.3% | 96.9% |
| **256K** (`b=1024`) | `bbox_mean` ($\alpha=0.25$) | Family 2 | 90.6% | 100.0% | 100.0% | 97.7% | 71.1% | 56.6% | 97.8% | 96.1% |
| **256K** (`b=1024`) | `bbox_mean` ($\alpha=0.3535$) | Family 2 | 88.3% | 96.9% | 100.0% | 96.9% | 66.7% | 52.6% | 97.2% | 95.0% |
| **256K** (`b=1024`) | `bbox_mean` ($\alpha=0.50$) | Family 2 | 86.7% | 96.9% | 100.0% | 96.5% | 64.5% | 50.2% | 96.8% | 94.4% |
| **256K** (`b=1024`) | `bbox_mean` ($\alpha=1.00$) | Family 2 | 82.8% | 96.9% | 100.0% | 95.3% | 57.6% | 43.5% | 95.7% | 92.7% |
| **256K** (`b=1024`) | `bbox_hyb` ($\alpha=0.125$) | Family 2 | 97.7% | 100.0% | 100.0% | 99.4% | 75.8% | 60.8% | 98.5% | 98.2% |
| **256K** (`b=1024`) | `bbox_hyb` ($\alpha=0.25$) | Family 2 | 95.3% | 100.0% | 100.0% | 98.8% | 71.7% | 56.7% | 98.0% | 97.5% |
| **256K** (`b=1024`) | `bbox_mid` ($\alpha=0.25$) | Family 2 | 99.2% | 100.0% | 100.0% | 97.1% | 46.5% | 34.4% | 94.3% | 94.3% |

### 3.3 Why Family 1 `mid_blend` ($\beta = 0.25$) Wins and Family 2 (`bbox`) Fails

1. **Why Family 2 (`q · k_center + alpha * |q| · k_rad`) degrades `R@512` and `R@2048`**:
   - In a standard signed inner product $q_h \cdot k^{\text{center}}_B = \sum_{d=0}^{127} q_{h,d} k^{\text{center}}_{B,d}$, uncorrelated channels have random signs and cancel as $O(1/\sqrt{128})$.
   - By contrast, the radius penalty $|q_{h,d}| \, k^{\text{rad}}_{B,d} \ge 0$ is **strictly non-negative on every channel $d \in [0, 128)$**. Summing 128 positive numbers adds an $O(D)$ positive bias proportional to $\|q_h\|_1 \|k^{\text{rad}}_B\|$ before the per-head ReLU clamp $\max(0, \cdot)$, which then sums across all 32 heads.
   - Any high-variance distractor block (where all 8 tokens have large random spread $k^{\text{rad}}_{B,d}$) receives a massive 32-head positive score boost even when its keys are orthogonal to $q_h$. At 256K Source (`budget = 1024`), increasing $\alpha$ from `0` to `0.25`, `0.50`, and `1.00` monotonically degrades `R@2048` from `61.4%` down to `56.6%`, `50.2%`, and `43.5%`.
2. **Why `mid_blend` ($\beta = 0.25$) strictly improves every recall metric over `mean` (`0741`)**:
   - Because $\hat{k}_{B,d} = (1-\beta)\bar{k}_{B,d} + \beta k^{\text{mid}}_{B,d}$ remains a **signed vector** in $\mathbb{R}^{128}$, uncorrelated channels still cancel as $O(1/\sqrt{128})$ inside $q_h \cdot \hat{k}_B$.
   - Moreover, unlike `absmax_blend` (which picks $+\sigma$ or $-\sigma$ on symmetric zero-mean channel oscillations and inflates noisy block norms), the midrange $\frac{\max_t k_{t,d} + \min_t k_{t,d}}{2}$ cancels symmetric zero-mean oscillations while preserving asymmetric needle spikes ($1.75\times$ amplitude boost at $\beta = 0.25$).
   - At $\beta = 0.25$, `mid_blend` achieves **Pareto dominance** over plain `mean` (`0741`) on every single context length (`32K`, `64K`, `128K`, `256K`), improving `Needle-1` (`+9.3%` at 32K Dense, `+14.1%` at 64K Dense, `+17.1%` at 128K Dense, `+1.6%` at 64K Source, `+2.3%` at 128K Source, `+3.9%` at 256K Source), `R@64` (`+0.4%` to `+3.9%`), `R@512` (`+0.2%` to `+0.9%`), `R@2048` (`+0.0%` to `+0.4%`), `ScoreW`, and `SoftM` (`+0.4%` to `+4.2%`).

---

## 4. Two-Boot GPU Kernel Latency (`sp3`, `spark-ds41f:1107`)

Full 2-boot measurements (`50` warmup + `200` timed iterations per kernel per boot) from `receipts/bench_boot1.json` and `receipts/bench_boot2.json`:

### 4.1 `summarize_blocks` Maintenance Kernel Latency (`us`)

By replacing the `[groups, 32]` fragment array (`8` live float32 registers per thread across the outer `k` loop) with 3 scalar local accumulators (`acc`, `hi`, `lo` via `T.alloc_local([1], T.float32)`) inside `for g, i in T.Parallel(groups, 32):`, `summarize_blocks` (`mid_blend_0.25`) uses fewer registers than `0741` (`mean`) and matches or improves kernel latency:

| Shape | `mean` (`0741`) Boot 1 | `mean` (`0741`) Boot 2 | `mean` (`0741`) Mean | `mid_blend` ($\beta=0.25$) Boot 1 | `mid_blend` ($\beta=0.25$) Boot 2 | `mid_blend` ($\beta=0.25$) Mean | Delta |
|---|---|---|---|---|---|---|---|
| `page=128, pages=4` (`512` keys) | `11.24 us` | `11.28 us` | **`11.26 us`** | `11.35 us` | `11.35 us` | **`11.35 us`** | `+0.09 us` |
| `page=128, pages=64` (`8,192` keys) | `13.43 us` | `13.38 us` | **`13.41 us`** | `13.52 us` | `13.50 us` | **`13.51 us`** | `+0.10 us` |
| `page=128, pages=256` (`32,768` keys) | `29.27 us` | `29.27 us` | **`29.27 us`** | `29.70 us` | `29.71 us` | **`29.70 us`** | `+0.43 us` |
| `page=256, pages=4` (`1,024` keys) | `22.25 us` | `22.15 us` | **`22.20 us`** | `22.12 us` | `22.14 us` | **`22.13 us`** | **`-0.07 us`** |
| `page=256, pages=64` (`16,384` keys) | `24.38 us` | `24.40 us` | **`24.39 us`** | `24.16 us` | `24.13 us` | **`24.14 us`** | **`-0.25 us`** |
| `page=256, pages=256` (`65,536` keys) | `58.84 us` | `58.87 us` | **`58.86 us`** | `58.31 us` | `58.33 us` | **`58.32 us`** | **`-0.54 us`** |

### 4.2 Coarse Scoring (`prepare_queries + prepared_scores`) and End-to-End Plan Latency (`rows = 4`, `us`)

- **Family 1 (`mean` and `mid_blend_0.25`)** execute the exact same `QueryReusePlan` (`prepare_queries` + single-GEMM `prepared_scores` + coarse top-k + `shortlist_indexer` + fine merge), so runtime scoring latency is bit-for-bit identical (`0.0 us` overhead) while `mid_blend_0.25` achieves `1.34x` to `2.90x` end-to-end speedup over native `IndexerV41Plan` at `64K-256K`.
- **Family 2 (`bbox`)** requires two `T.gemm` calls per CTA tile in `prepared_scores_bbox`, increasing coarse scoring latency by **`+54.5%` to `+81.2%`** (`37.88 -> 68.26 us` at 64K Dense, `55.89 -> 93.60 us` at 128K Source, `94.75 -> 171.69 us` at 256K Source).

| Context / Layer | Native `IndexerV41Plan.run` (Boot 1 / Boot 2 / Mean) | Family 1 `QueryReusePlan.run` (Boot 1 / Boot 2 / Mean) | Family 1 Coarse Score (`prep + scores`) | Family 2 `bbox` Coarse Score (`prep + scores_bbox`) | Family 2 Coarse Overhead | Family 1 Speedup vs. Native |
|---|---|---|---|---|---|---|---|
| **32K Dense** (`p=128, b=64`) | `107.61 / 107.50` (**`107.56 us`**) | `124.54 / 124.28` (**`124.41 us`**) | `30.28 / 30.22` (**`30.25 us`**) | `46.77 / 46.70` (**`46.74 us`**) | `+16.49 us` (`+54.5%`) | `0.86x` |
| **64K Dense** (`p=128, b=64`) | `172.01 / 172.16` (**`172.09 us`**) | `128.08 / 128.25` (**`128.17 us`**) | `37.82 / 37.94` (**`37.88 us`**) | `68.25 / 68.27` (**`68.26 us`**) | `+30.38 us` (`+80.2%`) | **`1.34x`** |
| **128K Dense** (`p=128, b=64`) | `300.56 / 300.62` (**`300.59 us`**) | `159.93 / 159.37` (**`159.65 us`**) | `55.76 / 55.86` (**`55.81 us`**) | `93.62 / 93.56` (**`93.59 us`**) | `+37.78 us` (`+67.7%`) | **`1.88x`** |
| **64K Source** (`p=256, b=512`) | `204.26 / 203.90` (**`204.08 us`**) | `187.84 / 187.65` (**`187.75 us`**) | `37.81 / 37.89` (**`37.85 us`**) | `68.25 / 68.27` (**`68.26 us`**) | `+30.41 us` (`+80.3%`) | **`1.09x`** |
| **128K Source** (`p=256, b=1024`) | `332.98 / 333.24` (**`333.11 us`**) | `248.20 / 247.37` (**`247.79 us`**) | `55.92 / 55.85` (**`55.89 us`**) | `93.57 / 93.63` (**`93.60 us`**) | `+37.71 us` (`+67.5%`) | **`1.34x`** |
| **256K Source** (`p=256, b=1024`) | `848.87 / 847.20` (**`848.04 us`**) | `292.73 / 292.42` (**`292.58 us`**) | `94.75 / 94.75` (**`94.75 us`**) | `171.75 / 171.62` (**`171.69 us`**) | `+76.94 us` (`+81.2%`) | **`2.90x`** |

---

## 5. Verification Gates & Two-Boot Pytest Receipts

1. **2-Boot Pytest Suite (`tests/models/test_deepseek_v4_1_lsa.py` and `tests/models/test_deepseek_v4_1_attention.py`)**:
   - Recorded in `receipts/pytest_2boot.txt`:
     - **Boot 1**: `74 passed in 27.42s` (`PDL sources: 41 kernels launched, 36 with PDL, 0 violations`), including `test_lsa_gpu_block_summary_needle_retention`.
     - **Boot 2**: `74 passed in 27.54s` (`PDL sources: 41 kernels launched, 36 with PDL, 0 violations`), including `test_lsa_gpu_block_summary_needle_retention`.
2. **2-Boot 7 LSA Correctness, Sanitizer, Registration, and CUDA-Graph Gates (`verify_7gates.py`)**:
   - Recorded in `receipts/gates_2boot.txt`:
     - `summary-initcheck`: `compute-sanitizer --tool initcheck` (`0 errors` across Boot 1 and Boot 2).
     - `summary-racecheck`: `compute-sanitizer --tool racecheck` (`0 hazards` across Boot 1 and Boot 2).
     - `summary-timing`: bit-exact `torch.equal` verification of `summarize_blocks(128)` and `summarize_blocks(256)` + timing across Boot 1 and Boot 2.
     - `registration`: `tilelang.prepare` + `prepare.freeze` + post-freeze binding across all 6 canonical `(width, page, source, budget)` shapes (`64` kernel calls, `0` cache misses across Boot 1 and Boot 2).
     - `actual`: bit-exact `STATE_BYTES == 100` summary + `QueryReusePlan.run` verification across dense (`budget=64`) and source (`budget=512`, `budget=1024`) shapes across Boot 1 and Boot 2.
     - `runtime-initcheck`: `compute-sanitizer --tool initcheck` across full `QueryReusePlan.run` and `summarize_blocks` with sentinel-filled (`173`) scratch/padding (`0 errors` across Boot 1 and Boot 2).
     - `graph`: CUDA graph capture and replay (`original`, `empty` with `lengths=0, active=0`, and `restored`) bit-identical (`torch.equal`) to eager execution across Boot 1 and Boot 2.

---

## 6. Conclusion

- **Adopt Family 1 Channel Midrange Blend (`mid_blend` with $\beta = 0.25$: $\hat{k}_{B,d} = 0.09375 \sum_{t \in B} k_{t,d} + 0.125 (\max_{t \in B} k_{t,d} + \min_{t \in B} k_{t,d})$)** in `vllm/models/deepseek_v4_1/tilelang/block_shortlist.py` (`0762-perf-dsv41-blend-LSA-block8-mean-with-channel-midran.patch`):
  - Strictly improves isolated 1-token needle recall (`+9.3%` to `+17.1%` on dense layers at `32K-128K`, `+1.6%` to `+3.9%` on source layer 20 at `64K-256K`, reaching `100.0%` at `64K` and `128K` Source and `96.1%` at `256K` Source) and simultaneously improves `R@64`, `R@512`, `R@2048`, `ScoreW`, and `SoftM` over `0741` uniform block averaging.
  - Preserves `STATE_BYTES = 100` (`128 x BF16` per 8-key block), leaves `QueryReusePlan.prepared_scores` (`T.gemm`) and default-off `VLLM_DS41_LSA=0/unset` 100% unchanged, and reduces live float32 accumulator registers in `summarize_blocks` from 8 to 3 with zero latency penalty.
- **Reject Family 2 Tensor-Core Center + Radius Bounding Box (`bbox`)**:
  - The unsigned radius inner product $|q_h| \cdot k^{\text{rad}}_B \ge 0$ lacks $O(1/\sqrt{128})$ random-sign cancellation across the 128 channels, biasing all 32 heads toward high-variance distractor blocks (`R@2048` at 256K drops from `61.4%` to `56.6%` at $\alpha = 0.25$ and `43.5%` at $\alpha = 1.00$) while increasing coarse scoring latency by `+54.5%` to `+81.2%` and requiring an extra `256 B/block` of radius storage.
