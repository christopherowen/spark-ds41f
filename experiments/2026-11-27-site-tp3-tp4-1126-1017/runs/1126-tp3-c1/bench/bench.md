## 1126-tp3-c1 (2026-11-27T06:42:51Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 86.77 | 101.87 | 0.225 |
| C4 | 195.99 | 59.13 | 0.314 |
| C8 | 310.46 | 48.19 | 0.468 |
| C16 | 430.48 | 34.37 | 0.7 |

### Per-stream tok/s by category

| category | C1 | C4 | C8 | C16 |
|---|---|---|---|---|
| coding | 137.8 | 82.79 | 72.8 | 44.94 |
| json | 97.63 | 55.8 | 48.02 | 36.28 |
| narrative | 65.0 | 36.51 | 27.27 | 19.72 |
| prose | 73.01 | 43.74 | 31.63 | 25.59 |
| math | 111.72 | 70.62 | 56.69 | 41.12 |
| reasoning | 102.09 | 49.74 | 43.54 | 31.75 |
| summary | 84.65 | 48.01 | 31.65 | 25.63 |
| format | 143.09 | 85.81 | 73.9 | 49.96 |
| ceiling_count | 150.31 | 113.52 | 90.58 | 63.83 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 2950 | 0.598 | 4930.3 |
| 8000 | 11592 | 1.688 | 6865.7 |
| 32000 | 46810 | 6.355 | 7366.4 |
| 64000 | 93335 | 11.811 | 7902.6 |
