## A/B: 1107-tp3-same-day (A) vs 1126-tp3-c1 (B)

### v41bench throughput (8 categories, thinking off)

| C | A agg tok/s | B agg tok/s | Δ | A per-stream | B per-stream | A TTFT s | B TTFT s |
|---|---|---|---|---|---|---|---|
| C1 | 80.53 | 86.77 | +7.7% | 92.85 | 101.87 | 0.226 | 0.225 |
| C4 | 207.32 | 195.99 | -5.5% | 62.58 | 59.13 | 0.323 | 0.314 |
| C8 | 312.08 | 310.46 | -0.5% | 47.96 | 48.19 | 0.447 | 0.468 |
| C16 | 274.76 | 430.48 | +56.7% | 42.27 | 34.37 | 2.678 | 0.7 |

Per-stream tok/s by category, Δ B vs A (C1 / C4 / C8 / C16):

| category | C1 | C4 | C8 | C16 |
|---|---|---|---|---|
| coding | 137.8 (+16.0%) | 82.79 (-13.5%) | 72.8 (-1.1%) | 44.94 (-29.2%) |
| json | 97.63 (-1.8%) | 55.8 (-12.9%) | 48.02 (+8.8%) | 36.28 (-11.7%) |
| narrative | 65.0 (+2.1%) | 36.51 (-0.7%) | 27.27 (+6.5%) | 19.72 (-16.3%) |
| prose | 73.01 (+14.9%) | 43.74 (+1.8%) | 31.63 (-6.9%) | 25.59 (-14.2%) |
| math | 111.72 (-0.5%) | 70.62 (+1.9%) | 56.69 (+3.5%) | 41.12 (-16.6%) |
| reasoning | 102.09 (+23.4%) | 49.74 (+0.4%) | 43.54 (+4.3%) | 31.75 (-18.8%) |
| summary | 84.65 (+28.4%) | 48.01 (-1.3%) | 31.65 (-4.7%) | 25.63 (-2.8%) |
| format | 143.09 (+5.0%) | 85.81 (-8.3%) | 73.9 (-3.5%) | 49.96 (-23.6%) |
| ceiling_count | 150.31 (+5.4%) | 113.52 (-0.4%) | 90.58 (+3.2%) | 63.83 (-20.0%) |

### Cold prefill (bench, unique prefix)

| target | tokens | A tok/s | B tok/s | Δ | A TTFT | B TTFT |
|---|---|---|---|---|---|---|
| 2000 | 2950 | 4461.6 | 4930.3 | +10.5% | 0.661 | 0.598 |
| 8000 | 11592 | 6217.8 | 6865.7 | +10.4% | 1.864 | 1.688 |
| 32000 | 46810 | 6967.1 | 7366.4 | +5.7% | 6.719 | 6.355 |
| 64000 | 93335 | 6985.0 | 7902.6 | +13.1% | 13.362 | 11.811 |

### Salted prefill sweep and needles

| prompt tokens | A tok/s | B tok/s | Δ | B pass |
|---|---|---|---|---|
| 3,475 | 4286.5 | 4823.0 | +12.5% | True |
| 13,419 | 6071.6 | 6525.9 | +7.5% | True |
| 54,603 | 7146.6 | 7765.5 | +8.7% | True |
| 109,131 | 7077.1 | 7935.3 | +12.1% | True |
| 3,493 | 4388.2 | 4725.8 | +7.7% | True |
| 13,427 | 6198.6 | 6554.7 | +5.7% | True |
| 54,961 | 7113.7 | 7819.6 | +9.9% | True |
| 108,736 | 7062.3 | 7955.0 | +12.6% | True |
| 545,995 (needle.json) | 5866.5 | 7676.4 | +30.9% | True |
| 990,329 (needle-1m.json) | 4708.5 | 7368.0 | +56.5% | True |

### Idle test (tok/s after first token; A from the clean rerun when present)

| run | A | B | Δ |
|---|---|---|---|
| A | 146.7 | 153.5 | +4.6% |
| B | 148.3 | 152.7 | +3.0% |
| C | 118.2 | 131.1 | +10.9% |
| D | 147.9 | 153.8 | +4.0% |
| E | 125.7 | 121.4 | -3.4% |

### Stall probe (a short stream beside a 545K prefill)

| run | A TTFT s | B TTFT s | A gap p50 / max ms | B gap p50 / max ms | A >500ms | B >500ms |
|---|---|---|---|---|---|---|
| alone | 0.149 | 0.146 | 39.5 / 43.2 | 37.7 / 39.2 | 0 | 0 |
| beside_needle_10s | 1.66 | 1.203 | 41.8 / 1131.5 | 45.1 / 911.5 | 17 | 17 |
| needle | pass True 5566.6 tok/s | pass True 7338.5 tok/s | | | | |

### Quality

- qeval: A 70/75 (fails code_interval_intersect, code_word_wrap, json_count, json_escape, math_m9); B 72/75 (fails code_interval_intersect, code_word_wrap, math_m9)
- vision/tools: A summary: 7/7; B summary: 7/7
- dsml-clients.jsonl: A 8/8 ok; B 8/8 ok; B not ok: none
- dsml-agents.jsonl: A 8/10 ok; B 8/10 ok; B not ok: multiturn_n2_max/whole, multiturn_n2_max/stream
- fixtures-quoted-cmp.jsonl: A 10/10 ok; B 10/10 ok, same token IDs as 11-06 10/10; B not ok: none
- fixtures-thinking-cmp.jsonl: A 8/8 ok; B 8/8 ok, same token IDs as 11-06 6/8; B not ok: none

### Loop probe (legitimate repetitive output; 1107 loop detection is on by default)

| case | effort | run | A complete / finish / tokens | B complete / finish / tokens | B reasoning tokens |
|---|---|---|---|---|---|
| zeros256 | none | whole | False / tool_calls / 2119 | True / tool_calls / 840 | 0 |
| zeros256 | none | stream | False / tool_calls / 2119 | True / tool_calls / 840 | 0 |
| ones16x16 | none | whole | True / tool_calls / 565 | True / tool_calls / 565 | 0 |
| ones16x16 | none | stream | True / tool_calls / 565 | True / tool_calls / 565 | 0 |
| table120 | none | whole | True / tool_calls / 780 | True / tool_calls / 780 | 0 |
| table120 | none | stream | True / tool_calls / 780 | True / tool_calls / 780 | 0 |
| zeros256_answer | none | whole | False / repetition / 2082 | True / stop / 779 | 0 |
| zeros256_answer | none | stream | False / repetition / 2082 | True / stop / 779 | 0 |
| zeros256 | max | whole | True / tool_calls / 1114 | True / tool_calls / 953 | 125 |
| zeros256 | max | stream | True / tool_calls / 1114 | True / tool_calls / 953 | 125 |
| ones16x16 | max | whole | True / tool_calls / 737 | True / tool_calls / 681 | 116 |
| ones16x16 | max | stream | True / tool_calls / 737 | True / tool_calls / 681 | 116 |
| table120 | max | whole | True / tool_calls / 839 | True / tool_calls / 831 | 50 |
| table120 | max | stream | True / tool_calls / 839 | True / tool_calls / 831 | 50 |
| zeros256_answer | max | whole | True / stop / 1095 | False / repetition / 2336 | 253 |
| zeros256_answer | max | stream | True / stop / 1095 | False / repetition / 2336 | 253 |

### Lowest MemAvailable in the battery (GiB)

| node | A | B | Δ |
|---|---|---|---|
| node4 | 117.31 | 117.31 | +0.00 |
| node2 | 9.47 | 8.36 | -1.11 |
| node1 | 6.77 | 6.34 | -0.43 |
| node3 | 9.37 | 8.03 | -1.34 |

### RoCE counters across the battery (delta; port_xmit_data is in 4-byte words)

| node | rail | A retrans / oos | B retrans / oos | B xmit GB |
|---|---|---|---|---|
| node1 | roceP2p1s0f1 | 0 / 0 | 0 / 0 | 517.6 |
| node1 | rocep1s0f1 | 0 / 0 | 0 / 0 | 518.8 |
| node3 | roceP2p1s0f1 | 0 / 0 | 0 / 0 | 498.8 |
| node3 | rocep1s0f1 | 0 / 0 | 0 / 0 | 499.9 |
| node4 | roceP2p1s0f1 | 0 / 0 | 0 / 0 | 0.0 |
| node4 | rocep1s0f1 | 0 / 0 | 0 / 0 | 0.0 |
| node2 | roceP2p1s0f1 | 0 / 0 | 0 / 0 | 517.5 |
| node2 | rocep1s0f1 | 0 / 0 | 0 / 0 | 518.7 |
