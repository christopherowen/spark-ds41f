# Context-Aware Step-Time Budgeting on an Absolute 2,048-Token Grid (`2026-10-30`)

## Problem

Interleaving graphed decode steps between multi-chunk prefill chunks (`--prefill-decode-interleave-steps K`) fixes mean ITL, p50 ITL, and decode throughput during long prefills, but the prefill chunk step itself still bounds the **maximum single-step stall (p99 / max ITL)**.

Moreover, because prefill attention scales with context length $c$ as:

$$t_{\text{chunk}}(q, c) \approx F + \alpha\, t_{\text{dec}} + m_0\, q + m_1\, q\, c$$

with $F = 153.5\ \text{ms}$, $\alpha = 1.56$, $m_0 \approx 125.3\ \mu\text{s/tok}$, and $m_1 \approx 132.9\ \text{ps/tok}^2$ on TP4 (`calibrate.py`), a fixed $16,288$-token chunk grows from **$2.26\ \text{s}$** at $c = 0$ to **$4.21\ \text{s}$** at $c = 900\text{K}$ (`stall-1m` / `soak`).

## Design

Now that patch `0526` (`ced-prompt-anchor`) in `vllm/models/deepseek_v4_1/ced.py` anchors CED replay to `prefill_len - 128` rather than chunk ends (verified bit-identical across chunk shapes by `ced_cross_shape`), `Scheduler._aligned_prefill_chunk` no longer needs a single static chunk size when decode requests are active:

1. **Solo prefill (`num_runnable_decodes == 0`)**:
   - Uses the full `long_prefill_token_threshold` ($16,288$ on TP4, $8,144$ / $6,096$ on TP3).
   - Solo 32K / 131K / 1M prefill speed is **100% unchanged** (`0.0%` delta).
2. **Mixed prefill + decode (`num_runnable_decodes > 0` and `--prefill-step-target-s > 0`)**:
   - `Scheduler._context_aware_chunk_threshold` selects $T(c) \in \{2048, 4096, 8192, \text{threshold}\}$ on the absolute `--prefill-chunk-grid` ($2,048$-token) grid to keep the step duration within $\tau_{\text{max}}$ (default $1.0\ \text{s}$):
     $$T(c) = \text{quantize}_{2048}\!\left(\frac{\tau_{\text{max}} - F - \alpha\, t_{\text{dec}}}{m_0 + m_1\, c}\right)$$
   - `Scheduler._aligned_prefill_chunk` snaps chunk ends onto the absolute $2,048$-token grid (`((computed + T(c)) // 2048) * 2048`), preserving deterministic absolute boundaries across prefix-cache hits and solo-to-mixed transitions.
   - `Scheduler.schedule()` caps total scheduled prefill tokens in a mixed step to $T(c)$ so multiple prefill chunks do not stack in the same step when $T(c) < \text{long\_prefill\_token\_threshold}$.

## Results (`results.json`)

### 1. Solo Prefill Invariance (`num_runnable_decodes == 0`)

| Shape | Prompt | Threshold | Production TTFT (s) | Context-Grid TTFT (s) | Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| TP4 | 32K | 16,288 | 4.7281 | 4.7281 | **0.0%** |
| TP4 | 131K | 16,288 | 19.1373 | 19.1373 | **0.0%** |
| TP4 | 1M | 16,288 | 215.5533 | 215.5533 | **0.0%** |
| TP3 (6K) | 32K | 6,096 | 5.9334 | 5.9334 | **0.0%** |
| TP3 (6K) | 131K | 6,096 | 24.2997 | 24.2997 | **0.0%** |

### 2. Step-Time Scaling Across Context Depth (TP4, $\tau_{\text{max}} = 1.0\ \text{s}$)

| Context $c$ | Static Chunk | Static Step (s) | $T(c)$ (tok) | Grid Step (s) | Stall Reduction |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 16,288 | 2.2589 | 8,192 | 1.2311 | **-45.5%** |
| 32,768 | 16,288 | 2.3299 | 8,192 | 1.2668 | **-45.6%** |
| 65,536 | 16,288 | 2.4008 | 8,192 | 1.3025 | **-45.8%** |
| 131,072 | 16,288 | 2.5426 | 8,192 | 1.3738 | **-46.0%** |
| 524,288 | 16,288 | 3.3938 | 4,096 | 0.9999 | **-70.5%** |
| 900,000 | 16,288 | 4.2071 | 4,096 | 1.2044 | **-71.4%** |

### 3. Mixed Prefill + Decode Workloads

| Workload | Arm | Max Chunk Step (s) | Stream ITL Mean (s) | Stream ITL p50 (s) | Stream ITL p99 (s) | Decode tok/s | Probe TTFT Mean (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `stall-15` (TP4) | `production` | 2.4971 | 1.4886 | 2.4213 | 2.4971 | 19.17 | 12.3644 |
| `stall-15` (TP4) | `interleave_k2` | 2.4971 | 0.7558 | 0.1147 | 2.4971 | 36.62 | 13.1628 |
| `stall-15` (TP4) | `context_grid` | **1.4259** | 1.1417 | 1.3919 | **1.4259** | 24.15 | 13.4425 |
| `stall-15` (TP4) | `context_grid_interleave_k2` | **1.4259** | **0.5426** | **0.1147** | **1.4259** | **51.61** | 15.2465 |
| `stall-11` (TP4) | `production` | 2.4833 | 1.4717 | 2.3938 | 2.4833 | 17.29 | 6.7555 |
| `stall-11` (TP4) | `interleave_k2` | 2.4902 | 0.6778 | 0.1103 | 2.4902 | 37.73 | 2.1875 |
| `stall-11` (TP4) | `context_grid` | **1.3984** | 1.1253 | 1.3644 | **1.3984** | 20.92 | 8.9665 |
| `stall-11` (TP4) | `context_grid_interleave_k2` | **1.4191** | **0.5050** | **0.1103** | **1.4191** | **52.74** | **1.4768** |
| `stall-7-tp3` | `production` | 1.4398 | 0.8865 | 1.4174 | 1.4398 | 12.91 | 8.0300 |
| `stall-7-tp3` | `context_grid_interleave_k2` | **0.8659** | **0.3478** | **0.0968** | **0.8659** | **42.41** | 9.4703 |
| `stall-3-tp3` | `production` | 1.1315 | 0.7910 | 1.1079 | 1.1315 | 12.26 | 2.9867 |
| `stall-3-tp3` | `context_grid_interleave_k2` | **0.8503** | **0.3287** | **0.0886** | **0.8503** | **32.40** | **0.9096** |
| `stall-1m` (900K) | `production` | 4.1628 | 3.0832 | 3.1880 | 4.1427 | 1.83 | 7.7792 |
| `stall-1m` (900K) | `interleave_k2` | 4.1628 | 1.0714 | 0.0375 | 4.1032 | 3.33 | 2.4045 |
| `stall-1m` (900K) | `context_grid` | **1.6236** | 1.1398 | 1.1047 | 1.6079 | 2.21 | 188.9937 |
| `stall-1m` (900K) | `context_grid_interleave_k2` | **1.6236** | **0.4041** | **0.0298** | **1.5766** | **6.43** | **1.2402** |

### 4. Deep-Context Multi-Session Soak (`soak`, 960K Contexts + Background Chat)

| Arm | Max Chunk Step (s) | Step p99 (s) | Chat ITL p99 (s) | Chat TTFT p99 (s) | Output tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| `production` | 3.6115 | 3.3087 | 3.3256 | 6.3174 | 41.68 |
| `interleave_k2` | 3.5410 | 3.2613 | 3.3063 | 4.9514 | 43.22 |
| `context_grid` | **1.5972** | 1.5675 | **1.5134** | 23.1601 | 38.40 |
| `context_grid_interleave_k2` | **1.5972** | **1.5613** | **1.5455** | **1.7107** | **41.44** |
