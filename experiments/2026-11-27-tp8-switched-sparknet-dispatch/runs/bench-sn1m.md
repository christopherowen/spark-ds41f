## ds41f-1126-tp8-sn1m (2026-11-28T00:25:29Z)

Prompt set `v1` (identical across boots), temperature 0, thinking off (none key). Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 157.22 | 183.69 | 0.127 |
| C8 | 478.91 | 75.03 | 0.331 |
| C16 | 641.46 | 50.83 | 0.474 |
| C32 | 989.0 | 40.26 | 0.731 |

### Per-stream tok/s by category

| category | C1 | C8 | C16 | C32 |
|---|---|---|---|---|
| coding | 235.7 | 98.62 | 63.97 | 55.11 |
| json | 200.34 | 78.08 | 55.98 | 42.98 |
| narrative | 116.42 | 45.75 | 31.54 | 22.53 |
| prose | 131.12 | 60.64 | 39.77 | 27.32 |
| math | 212.36 | 88.12 | 56.47 | 48.03 |
| reasoning | 190.03 | 72.76 | 49.53 | 37.51 |
| summary | 140.76 | 55.2 | 37.61 | 25.46 |
| format | 242.83 | 101.03 | 71.8 | 63.17 |
| ceiling_count | 266.97 | 124.65 | 80.08 | 70.6 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 2950 | 0.434 | 6799.9 |
| 8000 | 11592 | 1.104 | 10500.0 |
| 32000 | 46810 | 4.162 | 11247.3 |
