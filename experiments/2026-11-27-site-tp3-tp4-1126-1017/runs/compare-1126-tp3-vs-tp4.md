## A/B: 1126-tp3-c1 (A) vs 1126-tp4-c1 (B)

### v41bench throughput (8 categories, thinking off)

| C | A agg tok/s | B agg tok/s | Δ | A per-stream | B per-stream | A TTFT s | B TTFT s |
|---|---|---|---|---|---|---|---|
| C1 | 86.77 | 109.53 | +26.2% | 101.87 | 127.81 | 0.225 | 0.169 |
| C4 | 195.99 | 262.2 | +33.8% | 59.13 | 81.09 | 0.314 | 0.312 |
| C8 | 310.46 | 378.43 | +21.9% | 48.19 | 59.95 | 0.468 | 0.418 |
| C16 | 430.48 | 577.27 | +34.1% | 34.37 | 46.09 | 0.7 | 0.577 |

Per-stream tok/s by category, Δ B vs A (C1 / C4 / C8 / C16):

| category | C1 | C4 | C8 | C16 |
|---|---|---|---|---|
| coding | 170.83 (+24.0%) | 114.21 (+38.0%) | 86.33 (+18.6%) | 65.9 (+46.6%) |
| json | 141.5 (+44.9%) | 78.24 (+40.2%) | 60.76 (+26.5%) | 44.2 (+21.8%) |
| narrative | 72.24 (+11.1%) | 48.0 (+31.5%) | 32.93 (+20.8%) | 25.21 (+27.8%) |
| prose | 94.5 (+29.4%) | 58.3 (+33.3%) | 44.31 (+40.1%) | 34.47 (+34.7%) |
| math | 142.45 (+27.5%) | 83.49 (+18.2%) | 69.99 (+23.5%) | 49.93 (+21.4%) |
| reasoning | 129.01 (+26.4%) | 69.65 (+40.0%) | 56.3 (+29.3%) | 41.24 (+29.9%) |
| summary | 103.23 (+21.9%) | 53.98 (+12.4%) | 44.76 (+41.4%) | 33.4 (+30.3%) |
| format | 168.75 (+17.9%) | 142.86 (+66.5%) | 84.23 (+14.0%) | 74.39 (+48.9%) |
| ceiling_count | 194.82 (+29.6%) | 144.3 (+27.1%) | 102.37 (+13.0%) | 74.77 (+17.1%) |

### Cold prefill (bench, unique prefix)

| target | tokens | A tok/s | B tok/s | Δ | A TTFT | B TTFT |
|---|---|---|---|---|---|---|
| 2000 | 2950 | 4930.3 | 5439.7 | +10.3% | 0.598 | 0.542 |
| 8000 | 11592 | 6865.7 | 8356.3 | +21.7% | 1.688 | 1.387 |
| 32000 | 46810 | 7366.4 | 9604.0 | +30.4% | 6.355 | 4.874 |
| 64000 | 93335 | 7902.6 | 9727.1 | +23.1% | 11.811 | 9.595 |

### Salted prefill sweep and needles

| prompt tokens | A tok/s | B tok/s | Δ | B pass |
|---|---|---|---|---|
| 3,447 | 4823.0 | 5979.1 | +24.0% | True |
| 13,361 | 6525.9 | 8423.9 | +29.1% | True |
| 54,509 | 7765.5 | 9457.2 | +21.8% | True |
| 108,712 | 7935.3 | 9838.5 | +24.0% | True |
| 3,474 | 4725.8 | 5935.2 | +25.6% | True |
| 13,469 | 6554.7 | 8499.1 | +29.7% | True |
| 54,783 | 7819.6 | 9627.0 | +23.1% | True |
| 109,057 | 7955.0 | 9612.9 | +20.8% | True |
| 545,639 (needle.json) | 7676.4 | 9439.6 | +23.0% | True |
| 993,179 (needle-1m.json) | 7368.0 | 9061.6 | +23.0% | True |

### Idle test (tok/s after first token; A from the clean rerun when present)

| run | A | B | Δ |
|---|---|---|---|
| A | 153.5 | 196.8 | +28.2% |
| B | 152.7 | 190.0 | +24.4% |
| C | 131.1 | 161.6 | +23.3% |
| D | 153.8 | 192.3 | +25.0% |
| E | 121.4 | 164.2 | +35.3% |

### Stall probe (a short stream beside a 545K prefill)

| run | A TTFT s | B TTFT s | A gap p50 / max ms | B gap p50 / max ms | A >500ms | B >500ms |
|---|---|---|---|---|---|---|
| alone | 0.146 | 0.123 | 37.7 / 39.2 | 29.3 / 45.5 | 0 | 0 |
| beside_needle_10s | 1.203 | 1.162 | 45.1 / 911.5 | 39.6 / 1087.1 | 17 | 17 |
| needle | pass True 7338.5 tok/s | pass True 8981.5 tok/s | | | | |

### Quality

- qeval: A 72/75 (fails code_interval_intersect, code_word_wrap, math_m9); B 70/75 (fails code_interval_intersect, code_word_wrap, json_count, json_escape, math_m9)
- vision/tools: A summary: 7/7; B summary: 7/7
- dsml-clients.jsonl: A 8/8 ok; B 6/8 ok; B not ok: n1_file_of_closers/whole, n1_file_of_closers/stream
- dsml-agents.jsonl: A 8/10 ok; B 8/10 ok; B not ok: multiturn_n2_max/whole, multiturn_n2_max/stream
- fixtures-quoted-cmp.jsonl: A 10/10 ok; B 8/10 ok, same token IDs as 11-06 8/10; B not ok: n1_file_of_closers/, n1_file_of_closers_stream/
- fixtures-thinking-cmp.jsonl: A 8/8 ok; B 8/8 ok, same token IDs as 11-06 6/8; B not ok: none

### Loop probe (legitimate repetitive output; 1107 loop detection is on by default)

| case | effort | run | A complete / finish / tokens | B complete / finish / tokens | B reasoning tokens |
|---|---|---|---|---|---|
| zeros256 | none | whole | True / tool_calls / 840 | False / tool_calls / 2119 | 0 |
| zeros256 | none | stream | True / tool_calls / 840 | False / tool_calls / 2119 | 0 |
| ones16x16 | none | whole | True / tool_calls / 565 | True / tool_calls / 565 | 0 |
| ones16x16 | none | stream | True / tool_calls / 565 | True / tool_calls / 565 | 0 |
| table120 | none | whole | True / tool_calls / 780 | True / tool_calls / 780 | 0 |
| table120 | none | stream | True / tool_calls / 780 | True / tool_calls / 780 | 0 |
| zeros256_answer | none | whole | True / stop / 779 | False / repetition / 2082 | 0 |
| zeros256_answer | none | stream | True / stop / 779 | False / repetition / 2082 | 0 |
| zeros256 | max | whole | True / tool_calls / 953 | False / tool_calls / 2477 | 357 |
| zeros256 | max | stream | True / tool_calls / 953 | False / tool_calls / 2477 | 357 |
| ones16x16 | max | whole | True / tool_calls / 681 | True / tool_calls / 736 | 170 |
| ones16x16 | max | stream | True / tool_calls / 681 | True / tool_calls / 736 | 170 |
| table120 | max | whole | True / tool_calls / 831 | True / tool_calls / 820 | 39 |
| table120 | max | stream | True / tool_calls / 831 | True / tool_calls / 820 | 39 |
| zeros256_answer | max | whole | False / repetition / 2336 | False / repetition / 2378 | 295 |
| zeros256_answer | max | stream | False / repetition / 2336 | False / repetition / 2378 | 295 |

### Lowest MemAvailable in the battery (GiB)

| node | A | B | Δ |
|---|---|---|---|
| node4 | 117.31 | 18.29 | -99.02 |
| node2 | 8.36 | 19.18 | +10.82 |
| node1 | 6.34 | 16.46 | +10.12 |
| node3 | 8.03 | 19.18 | +11.15 |

### RoCE counters across the battery (delta; port_xmit_data is in 4-byte words)

| node | rail | A retrans / oos | B retrans / oos | B xmit GB |
|---|---|---|---|---|
| node1 | roceP2p1s0f1 | 0 / 0 | 0 / 0 | 610.6 |
| node1 | rocep1s0f1 | 0 / 0 | 0 / 0 | 610.9 |
| node3 | roceP2p1s0f1 | 0 / 0 | 0 / 0 | 610.6 |
| node3 | rocep1s0f1 | 0 / 0 | 0 / 0 | 610.9 |
| node4 | roceP2p1s0f1 | 0 / 0 | 0 / 0 | 610.6 |
| node4 | rocep1s0f1 | 0 / 0 | 0 / 0 | 610.9 |
| node2 | roceP2p1s0f1 | 0 / 0 | 0 / 0 | 610.6 |
| node2 | rocep1s0f1 | 0 / 0 | 0 / 0 | 610.9 |
