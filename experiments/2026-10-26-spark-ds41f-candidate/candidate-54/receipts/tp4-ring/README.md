# candidate-54 at tp4-ring (dgx1..dgx4), 2026-11-10

These are the lab's outputs from the window of 00:32–02:20Z, copied from dgx1 `results/private/`.

| Path | What |
| --- | --- |
| `lab/candidate-54-image/` | The image build on dgx4 (`spark-ds41f:candidate-54`, image ID `e1de71be`, the same 57 layers on every node) and the release-test kernel job: 10,171 passed, 311 skipped, 0 failed (2,912 s) |
| `determinism/candidate-54-tp4/`, `lab/candidate-54-tp4-table.txt` | The gate: temperature-0 outputs identical in every scenario, hashes 18d61961/d6568649 (logprobs) and 0a2dacdd/12fd8257 (text). These are #1030's TP4 hashes, as candidate-52 recorded them at `tp4-switched`, so no other patch changes greedy output. Distinct prompts give 270.6 / 200.2 / 155.8 tok/s at c16 / c8 / c4 |
| `determinism/candidate-54-screen-tp4/`, `lab/candidate-54-screen-tp4-table.txt` | The screen: two candidate-54 boots against 1107's recorded screen. The suite runs are in `../references/` |
| `lab/candidate-54-window.log` | The window's log |

**Screen, candidate-54 against 1107** (verdicts against the run's own repeat):
- engine grid geometric mean 4.78% faster (±0.14%);
- c16 paired decode 1.61% faster (margin 0.88%); c16 output 270.7 against 265.2 tok/s;
- prefill 3.46% faster at 32K (8,793 tok/s) and 7.84% faster at 256K (8,873 tok/s);
- the model step's acceptance length is 2.301 and 2.304, and perplexity on its own greedy text 1.31888, as
  candidate-52 with #1030 recorded. 1107's reference keeps no greedy corpus, and #1030 changes greedy text, so the
  model comparison is skipped.

**Time to ready:** the gate's first boot of the new image took 157.4 s; the screen's boots took 111.9 and 113.7 s,
as candidate-50's did (110–115 s). The first boot compiles the new kernels.

**Live checks did not run here.** The coordinator asked for the window (`~/spark-request.json`) to close it after the
live checks. The runner reads that file between steps, so it stopped the live-checks job after its boot, before any
check ran, and restored production (`doctor --live` OK at 02:20Z). The checks do not depend on topology. They run at
`tp4-switched` instead ([../../live-checks-tp4-switched.json](../../live-checks-tp4-switched.json)), in sp1's open
window.
