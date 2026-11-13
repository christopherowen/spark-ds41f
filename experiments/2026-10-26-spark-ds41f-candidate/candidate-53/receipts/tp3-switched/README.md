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
   - Prefill is unchanged. Stall-3 is worse with #984: prefill under decode is 2.37% slower (margin 1.27%), and decode p90 during that prefill is 0.77% higher (interval 0.30%). *(Corrected 2026-11-10: this line first said stall-3 was unchanged.)*
   - Verdict: #984 misses, and is in rework.
2. **#971 and #977:** `attribute-stall3-tp3-switched` runs candidate-50 and candidate-53 without #960, #998 and #984.
   - Stall-3: candidate-50 4,760/4,767 tok/s, candidate-53 4,783/4,722 tok/s.
   - Prefill and time to ready are within the boot spread.
   - So without #984, candidate-53 matches candidate-50 at stall-3. The screen's 2.1% stall-3 gap is #984's (item 1), not drift between sessions. #971 (round 1) and #977 pass. *(Corrected 2026-11-10: this line first called the gap drift.)*
3. **#984 round 1:** `attribute-984r1-tp3-switched` runs candidate-53 without #960 and #998, with #984's round 1 and
   without #984, ABBA (`lab/candidate-53-attribute-984r1-tp3-switched-table.txt`, the window's log in
   `lab/candidate-53-attribute-984r1-window.log`).
   - Time to ready: 126.8/127.8 s with round 1, 124.4/124.9 s without.
   - Lowest `MemAvailable` during the boot, with round 1 against without, first and second boots:
     sp5 5.092/5.187 against 5.075/4.933 GiB, sp6 7.424/7.482 against 7.398/7.479, sp7 7.208/7.295 against 7.203/7.266.
   - Prefill and stall-3 are the same against the run's own repeat. Decode p90 during the stall-3 prefill is 0.8% lower
     without round 1 (interval 0.54%).
   - Verdict: the floor is not higher beyond the boot spread, so #984 misses again and is dropped.
