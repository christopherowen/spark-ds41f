## 1107-tp3-same-day (2026-11-27T05:34:30Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 80.53 | 92.85 | 0.226 |
| C4 | 207.32 | 62.58 | 0.323 |
| C8 | 312.08 | 47.96 | 0.447 |
| C16 | 274.76 | 42.27 | 2.678 |

### Per-stream tok/s by category

| category | C1 | C4 | C8 | C16 |
|---|---|---|---|---|
| coding | 118.81 | 95.76 | 73.63 | 63.49 |
| json | 99.46 | 64.07 | 44.13 | 41.08 |
| narrative | 63.69 | 36.76 | 25.61 | 23.56 |
| prose | 63.54 | 42.96 | 33.98 | 29.81 |
| math | 112.32 | 69.3 | 54.77 | 49.33 |
| reasoning | 82.71 | 49.53 | 41.74 | 39.12 |
| summary | 65.91 | 48.65 | 33.2 | 26.38 |
| format | 136.33 | 93.61 | 76.6 | 65.36 |
| ceiling_count | 142.66 | 114.0 | 87.75 | 79.78 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 2950 | 0.661 | 4461.6 |
| 8000 | 11592 | 1.864 | 6217.8 |
| 32000 | 46810 | 6.719 | 6967.1 |
| 64000 | 93335 | 13.362 | 6985.0 |
