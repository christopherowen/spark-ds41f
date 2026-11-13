# candidate-50 at tp3-switched (sp5..sp7), 2026-11-09

These are the lab's outputs from sp5's window (17:04–19:35Z), copied from sp5 `results/private/`.
- The screen's and companions' suite runs are in `../references/`.
- #1009 round 1's suite runs are in `../rework/pr-1009-r1/validation/tp3-switched/`.

**Gate**, candidate-50 without #1009: temperature-0 outputs are identical, with TP3's hashes aad35475/e1a6de33.

**Screen** (`screen-tp3`), ABBA against two 1107 boots. The first arm is 1107. Read the other way, against 1107:
- engine grid: the same;
- c8 paired decode: 0.59% faster (margin 0.48%);
- c8 output: the same;
- greedy responses: identical, 128/128.

**#1009 round 1:** the boot and gate are identical, with TP3's hashes. On the screen against candidate-50 without #1009, grid and c8 are the same and greedy is identical.

**Companions** (the references for candidate-53's marks):

| Suite | Result | Reference |
| --- | --- | --- |
| `single-stream` | recorded | `candidate-50-no1009-tp3-switched-screen-single-stream` |
| `sampled` | recorded | `candidate-50-no1009-tp3-switched-screen-sampled` |
| `release-tp3` | halted at its model step | none |

- **The `release-tp3` halt:** the run stopped with `HALT: sp5: MemAvailable 4.00 GiB fell below the 4 GiB bench floor` while scoring 256 prompts, since the head runs the client beside rank 0. Its directory holds only `conditions.json`, so it is not a reference.
- **The stall rows** for #984 are measured with the stall steps alone. They don't need the model step.
