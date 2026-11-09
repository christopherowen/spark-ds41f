# PR #979 rework, round 1: the T≥6144 exchange pipeline only

**Sub-agent verdict:** `corrected-meets-mark`. The corrected patch is `0766-perf-dsv41-pipeline-the-sliced-SP-Engram-row-exchang.patch`, a git format-patch with the original author. It applies on 0762..0765, and 0767 applies after it.

## Correction
- **`project_exchange_rows`:** a single slice (T < 6,144) takes 1107's serial path. Two or more slices take #979's PyNCCL side-stream pipeline.
- **`gather_many` and `gather_mxfp8`:** byte for byte 1107's grouped PyNCCL, on every topology and T. The one-shot RoCE gather and the small-tensor packing are removed.

**How the variant was chosen.** Two variants were declared before the first run (`raw/plan.txt`): corrA kept the one-shot gather on ring4, and corrB dropped it.
- On ring4, #979's gather was slower at T=512 (+17%) and T=1024 (+60%). corrA ran that same gather code and read −31% at T=512, so the readings for identical code disagreed.
- On direct the one-shot gather was never faster.
- corrB was delivered.

## Mark cells: `project_exchange_rows`

Slowest-rank medians, boot 1 / boot 2, one session, two boots per arm:

| Cell | Corrected (µs) | Baseline (µs) | Original #979 (µs) | Change |
| --- | --- | --- | --- | --- |
| ring4 T=6144 | 9914 / 10061 | 12129 / 11772 | 9762 / 9868 | −16.4% |
| ring4 T=8192 | 12630 / 12753 | 15651 / 15403 | 12608 / 12626 | −18.3% |
| direct T=6144 | 8983 / 8836 | 11408 / 10557 | 9068 / 8759 | −18.9% |
| direct T=8192 | 11829 / 11948 | 13999 / 13751 | 11679 / 11900 | −14.3% |

All four cells gain beyond the interval, on the slowest rank and on rank 0.

## Every other cell runs 1107's code
Thirteen cells read slower beyond the interval, and each runs code identical to the baseline's. An A/A shows those readings are noise:
- baseline boots 3–4 against boots 1–2 read "beyond the interval" in 8 of 32 cells;
- across arms, 45 of 113 pairs that run identical code read beyond the interval on the slowest rank.

Two boots per arm cannot resolve the small-T cells. The verdict there rests on the code being identical.

**Bit-exact:** every rank's output digest is equal across all 28 runs (this round and the own mark).

**Tests:** 396 GPU tests pass, and 219 CPU tests pass with 17 skipped, the original #979's counts. The routing tests now assert the corrected order.

## Coordinator's verdict: meets the mark
- The gain #979 is for, `project_exchange_rows` at T≥6144, holds at 14–19% on both served topologies.
- Everywhere it was slower, 1107's code runs again.

Round 1 joins the next candidate for fresh qualification.

**Caveat:** `ENGRAM_SLICE_TOKENS` = 6144 also moves where a pure-PyNCCL profile starts slicing. No served shape runs that way.

The analysis is the rework sub-agent's; the coordinator wrote the verdict.
