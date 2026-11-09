# PR #986 rework, round 1: no defensible correction; dropped from this release

**Sub-agent verdict:** `no-defensible-correction`. No patch. Every figure below is in `result.json` and the files beside it.

**What #986 changes in serving** (verified rows a step, two boots each, candidate-47 against candidate-47 without #986):
- **c16:** 44.61 → 44.22 rows (−0.87%), with tokens a step −0.44%. It trims a little.
- **c8:** the same rows (+0.12%).
- **c1:** identical by construction. There is one request with at most 6 rows, and every #986 term is zero there, so the paired c1 figure is boot noise.

**Why the modeled gains don't appear:**
- **The regimes are absent from the release suite, or rare in it.**
  - Release contexts never exceed 285 tokens, while the context band starts at 8,192.
  - Served c1 verifies at most 6 rows, against the 16 the plateau needs.
  - c8 holds at most 48 rows, so the 64-row knee never fires.
- **Its own screen priced it with an assumed cost model.**
  - At c16, 64% of the modeled saving is a hand-set 440 µs/row term. Another 23% is a dense-GEMM table measured on zero tensors, and that table is not monotone.
  - The context band triggers a trim rather than pricing a context cost.
- **The tp4-ring engine grid shows no cost of that shape.**
  - Step time is flat with context: 1x3 rows read 15.96 ms at 4K and 15.66 ms at 1M.
  - There is no tile step of the claimed size: about 0.27 ms/row up to 64 rows.
  - Above 64 rows, #986 charges about twice what the grid measures.
- **A replay of the real 1107 and #986 modules** (20,000 steps a loop, two seeds) reproduces the served c16 change: rows −1.00%, tokens −0.50%. The budget differs on 5.7% of c8 steps and on no c1 step.

**A correction cannot deliver.** Under three step-time models (grid fit, served regression, and a blend calibrated to the 2026-10-18 row-cost A/B), both #986 and a knee re-calibrated to the grid predict at most:
- 0.16% at c16, against an A/A floor of 0.43%;
- 0.01% at c8, against 0.39%;
- 0 at c1, against 0.35%.

The one regime serving does hit is long agent prompts (64-100K). There the band trims rows 3.7-5.9% to pay a cost the grid doesn't show, a small predicted loss.

## Coordinator's verdict: dropped from this release

The owner asked for #986 to go first, and it did: it was measured on tp4-ring with its own `no986` control, then reworked.

Under the owner's rule, a second rework is for a PR whose first rework shows promise, and this one shows none:
- its only served effect is a trim below the A/A floor;
- its modeled gains rest on cost terms the engine grid contradicts;
- no re-calibration predicts a gain above the floor.

#986 is dropped from this release; its record stays here.

**For a later PR** (not a correction of #986):
- Served steps are about 4.5 times longer than grid steps at c16, so the grid underprices served rows.
- Whether the draft budget should move is a separate experiment on the recipe's row-cost constant, which `TODO.md` lists as tried and not adopted.
- If the context band returns, the test is servebench c1 and c2 at `--context 65536` and `131072`: 300 s windows, two boots each, against a `no<PR>` control on the same image.

The analysis is the rework sub-agent's; the coordinator wrote the verdict.
