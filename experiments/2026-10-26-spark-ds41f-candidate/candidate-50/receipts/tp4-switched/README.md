# candidate-50 at tp4-switched (sp1..sp4), 2026-11-09

These are the lab's outputs from sp1's window (17:04–18:51Z), copied from sp1 `results/private/`.
- The screen's suite runs are in `../references/`.
- #1009 round 1's suite runs are in `../rework/pr-1009-r1/validation/tp4-switched/`.

**Gate**, candidate-50 without #1009: temperature-0 outputs are identical in every scenario, with TP4's hashes 8e46794a/11ec073c.

**Screen**, ABBA against two 1107 boots. The table's first arm is 1107, so its "worse" means 1107 is worse. Read the other way, candidate-50 without #1009 against 1107:

| Metric | Result |
| --- | --- |
| Engine grid, geometric mean | 0.40% faster (±0.20%) |
| c16 paired decode | 1.31% faster (margin 0.53%) |
| c16 output | 264.6 against 260.5 tok/s |
| Prefill at 256K | 4.6% faster: 8,561 against 8,184 tok/s |
| Prefill at 32K | the same |
| Greedy responses | identical, 128/128; KL 0 |

**#1009 round 1** (`tp4-switched-1009r1.json`), at tp4-ring's sizing:
- **Boot:** ready in 126 s, and the gate is identical with TP4's hashes.
- **Screen** against candidate-50 without #1009 (recorded in the same window):
  - prefill at 256K is 3.28% better (±2.93%);
  - grid, c16 decode and prefill at 32K are the same;
  - greedy responses are identical (128/128).
