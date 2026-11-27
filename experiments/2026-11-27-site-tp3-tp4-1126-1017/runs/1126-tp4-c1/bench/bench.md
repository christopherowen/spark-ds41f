## 1126-tp4-c1 (2026-11-27T07:31:24Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 109.53 | 127.81 | 0.169 |
| C4 | 262.2 | 81.09 | 0.312 |
| C8 | 378.43 | 59.95 | 0.418 |
| C16 | 577.27 | 46.09 | 0.577 |

### Per-stream tok/s by category

| category | C1 | C4 | C8 | C16 |
|---|---|---|---|---|
| coding | 170.83 | 114.21 | 86.33 | 65.9 |
| json | 141.5 | 78.24 | 60.76 | 44.2 |
| narrative | 72.24 | 48.0 | 32.93 | 25.21 |
| prose | 94.5 | 58.3 | 44.31 | 34.47 |
| math | 142.45 | 83.49 | 69.99 | 49.93 |
| reasoning | 129.01 | 69.65 | 56.3 | 41.24 |
| summary | 103.23 | 53.98 | 44.76 | 33.4 |
| format | 168.75 | 142.86 | 84.23 | 74.39 |
| ceiling_count | 194.82 | 144.3 | 102.37 | 74.77 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 2950 | 0.542 | 5439.7 |
| 8000 | 11592 | 1.387 | 8356.3 |
| 32000 | 46810 | 4.874 | 9604.0 |
| 64000 | 93335 | 9.595 | 9727.1 |
