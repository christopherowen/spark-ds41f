# candidate-53 at tp3-switched (sp5..sp7), 2026-11-09/10

These are the lab's outputs from sp5's window, copied from sp5 `results/private/`.

## Boots
- **Full candidate-53:** fails in the warm-up decode step (#960's missing `has_connector`); see `../../README.md`.
- **Without #960 and #998:** `tp3-switched-no960-no998.json`.

## Without #960 and #998, against candidate-50 without #1009

The reference is candidate-50's runs recorded in the same window, earlier.
- **Gate:** temperature-0 outputs are identical, with TP3's hashes aad35475/e1a6de33.
- **Screen:** grid, c8 decode and c8 output are the same, and greedy is identical. The suite's table file was overwritten by the stall job's, so the screen table is rebuilt from the runs: `lab/candidate-53-screen-tp3-switched-screen-rebuilt-table.txt`.
- **`single-stream` and `sampled`:** the same at every concurrency.
- **Stall job** against the recorded run: stall-3 is 2.1% slower, beyond its margin.
- **Prefill 32K:** −1.2%. sp6 hit its thermal guard (82–85 °C) during this step.

## Attribution, same session, ABBA on one image
1. **#984:** `attribute-984-tp3-switched` runs with and without #984.
   - It adds about 77 s to time to ready: 203/199 s with it, 121.5/126.9 s without.
   - It raises the startup MemAvailable floor by only 0.03–0.12 GiB.
   - Stall-3 and prefill are unchanged.
   - Verdict: #984 misses, and is in rework.
2. **#971 and #977:** `attribute-stall3-tp3-switched` runs candidate-50 and candidate-53 without #960, #998 and #984.
   - Stall-3: candidate-50 4,760/4,767 tok/s, candidate-53 4,783/4,722 tok/s.
   - Prefill and time to ready are within the boot spread.
   - So the stall-3 gap against the recorded run was drift between sessions. #971 (round 1) and #977 pass.
