## ds41f-1126-tp8-base (2026-11-28T00:19:20Z)

Prompt set `v1` (identical across boots), temperature 0, thinking off (none key). Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 155.98 | 181.04 | 0.126 |
| C8 | 473.93 | 74.5 | 0.329 |
| C16 | 638.25 | 50.66 | 0.476 |
| C32 | 842.79 | 32.97 | 0.722 |

### Per-stream tok/s by category

| category | C1 | C8 | C16 | C32 |
|---|---|---|---|---|
| coding | 236.3 | 98.13 | 65.36 | 39.82 |
| json | 193.01 | 76.31 | 55.19 | 36.56 |
| narrative | 116.74 | 46.55 | 31.66 | 21.67 |
| prose | 133.1 | 55.98 | 40.13 | 26.45 |
| math | 214.35 | 86.15 | 56.95 | 37.36 |
| reasoning | 190.14 | 71.48 | 48.52 | 32.96 |
| summary | 138.14 | 54.69 | 36.88 | 24.86 |
| format | 226.56 | 106.75 | 70.62 | 44.05 |
| ceiling_count | 275.65 | 124.36 | 80.15 | 48.45 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 2950 | 0.492 | 6000.9 |
| 8000 | 11592 | 1.182 | 9807.7 |
| 32000 | 46810 | 4.135 | 11320.6 |
