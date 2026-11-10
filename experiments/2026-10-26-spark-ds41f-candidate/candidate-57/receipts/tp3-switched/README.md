# candidate-57 at tp3-switched (sp5..sp7), 2026-11-10

These are the lab's outputs from sp5's window, copied from sp5 `results/private/`. `lab/candidate-57-window.log` is
the window log's part from the gate's start to the screen's end. The image was built on sp8 (ID `ba512cb2`), copied
to sp5 over the fabric in 123 s and distributed to sp6 and sp7.

| Path | What |
| --- | --- |
| `determinism/candidate-57-tp3-switched/`, `lab/candidate-57-tp3-switched-table.txt` | The gate: temperature-0 outputs identical in every scenario, hashes 5c22ce41/bce6092e (logprobs): candidate-54's, so #957 parts 2 and 4 change no greedy output |
| `determinism/candidate-57-screen-tp3-switched/`, `lab/candidate-57-screen-tp3-switched-table.txt`, `suites/` | The screen: `screen-tp3`, `single-stream` and `sampled` on two boots, against candidate-54's recorded runs (`../../../candidate-54/receipts/references/`) |

**The bench memory floor cut both boots short.** sp5's MemAvailable fell to 3.90 GiB in the first boot's prefill
step and to 4.00 GiB in the second boot's `c8` step and `sampled` c1 step, under the 4 GiB bench floor, and each halt
ends that suite. So:
- `grid`: both boots; `c8`: the first boot only; prefill: not measured;
- `single-stream`: both boots; `sampled`: both boots at c16 and c8, the first at c1.
candidate-54's TP3 screen hit the same floor (`../../../candidate-54/receipts/tp3-switched/README.md`), so the floor
is sp5's, not this candidate's: the lowest MemAvailable at boot is 4.04-5.31 GiB here, 3.96-5.98 GiB there.

**Against candidate-54** (verdicts against the run's own repeat where both boots ran):
- engine grid geometric mean -0.34%, the same (margin 0.38%). The 4-request shapes are 1.7-1.9% faster (4x8,
  4x12), 8x16 is 0.74% slower, the 1-request shapes the same;
- `c8` paired decode +0.56% (±0.54%), better, first boot; output the same;
- `single-stream` c1 paired decode -0.27%, the same (margin 0.40%);
- `sampled` c16 and c8 the same;
- model step: greedy outputs identical on both boots (`b795e6d4`), perplexity unchanged (1.31608), acceptance
  length 2.307 and 2.301.
