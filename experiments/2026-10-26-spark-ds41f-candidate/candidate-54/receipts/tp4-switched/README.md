# candidate-54 at tp4-switched (sp1..sp4), 2026-11-10

These are the lab's outputs from sp1's window (opened 02:07Z), copied from sp1 `results/private/`.

| Path | What |
| --- | --- |
| `determinism/candidate-54-tp4-switched/`, `lab/candidate-54-tp4-switched-table.txt` | The gate: temperature-0 outputs identical in every scenario, hashes 18d61961/d6568649 (logprobs) and 0a2dacdd/12fd8257 (text), as at `tp4-ring` |
| `determinism/candidate-54-live-checks-tp4-switched/`, `lab/candidate-54-live-checks-tp4-switched-table.txt` | The live checks of #955, #1013 and #972, all passed. They run here because the `tp4-ring` run stopped before them (`../tp4-ring/README.md`) |
| `determinism/candidate-54-screen-tp4-switched/`, `lab/candidate-54-screen-tp4-switched-table.txt` | The screen: two candidate-54 boots against 1107's recorded screen. The suite runs are in `../references/` |
| `suites/candidate-54-screen-tp4-switched-stopped/`, `lab/candidate-54-screen-tp4-switched/` | The first screen attempt: its first boot's suite run, and the saved logs of its second boot, which refused to start (below) |
| `lab/candidate-54-window.log` | The window's log |

**Screen, candidate-54 against 1107** (verdicts against the run's own repeat):
- engine grid geometric mean 4.29% faster (±0.19%);
- c16 output 1.62% faster (264.8 against 260.5 tok/s); c16 paired decode +0.94%, inside its margin (1.05%), so the
  same;
- prefill the same at 32K (+1.6%, ±3.7%) and 7.5% faster at 256K (8,797 tok/s).

**Two process errors, both the coordinator's:**
- **The first screen attempt lost its second boot.** The coordinator moved sp1's checkout to queue the live checks
  while the screen ran. The second boot then found sp2..sp4's checkouts behind sp1's and refused to start
  (02:31:56Z). The first boot's run is kept in `suites/candidate-54-screen-tp4-switched-stopped/` (prefill 8,638 and
  8,750 tok/s); the screen ran again in full.
- **An image copy overlapped the repeat boot's c16 step.** From about 03:00:10 to 03:02:00Z, `docker save` of
  candidate-56 on sp8 streamed into `docker load` on sp1, during about two of the c16 step's six minutes. It was
  stopped before the load finished. The repeat boot's c16 is 0.30% below the first boot's in output and 0.25% in
  paired decode, both inside the interval, so no effect shows; the boot is kept as measured.

**Time to ready:** 186.4 s for the gate's first boot of the image, then 124–132 s.
