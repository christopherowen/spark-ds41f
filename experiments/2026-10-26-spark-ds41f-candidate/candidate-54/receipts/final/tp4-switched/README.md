# candidate-54's rebuilt image at tp4-switched (sp1..sp4), 2026-11-11

The lab's outputs from sp1's window, 13:09-14:18Z, copied from sp1 `results/private/`. The image is
`spark-ds41f:candidate-54`, image ID `d502217e` (tree `ad2dc11f`), built on dgx4 and shipped from dgx1 to sp1, then
fanned out over the fabric; `build distribute` verified the same ID on sp1..sp4.

| Path | What |
| --- | --- |
| `determinism/candidate-54-gate-final-tp4-switched/`, `lab/candidate-54-gate-final-tp4-switched-table.txt` | The gate on one boot ([gate-final-tp4-switched.json](../../../gate-final-tp4-switched.json)) |
| `lab/candidate-54-0799-screen-tp4-switched-table.txt` | The screen ([screen-0799-tp4-switched.json](../../../screen-0799-tp4-switched.json)): two boots against 1107's recorded screen. The suite runs are `../../references/candidate-54-0799-tp4-switched-screen{,-again}/` |
| `determinism/candidate-54-0799-live-checks-tp4-switched/` | The live checks ([live-checks-0799-tp4-switched.json](../../../live-checks-0799-tp4-switched.json)) |
| `boot-differences/candidate-54-c1-decisions-tp4-switched/`, `determinism/candidate-54-c1-decisions-tp4-switched/` | The c1 decision check ([c1-decisions-tp4-switched.json](../../../c1-decisions-tp4-switched.json)): each boot's `c1.json` with every request's per-step token counts |
| `lab/window.log` | The window log's part for the four jobs |
| `lab/candidate-54-gate-final-tp4-switched-records.sha256` | The gate's record files, kept on sp1 |

**Gate:** all four runs identical in every scenario, on the recorded TP4 hashes 18d61961/d6568649 (logprobs) and
0a2dacdd/12fd8257 (text). Distinct prompts give 266.7 / 197.0 / 152.6 tok/s at c16 / c8 / c4.

**Screen, candidate-54 against 1107** (verdicts against the run's own repeat):

| Metric | Change | Margin | Verdict | Earlier build (`e1de71be`) |
| --- | --- | --- | --- | --- |
| engine grid geometric mean | -4.24% (faster) | 0.30% | better | -4.29% |
| c16 paired decode | +1.13% | 1.09% | better | +0.94%, inside its margin |
| c16 output | +1.50% (264.4 / 264.9 against 260.5 tok/s) | 0.95% | better | +1.62% (264.8 tok/s) |
| prefill 32K | +3.43% (8,658 / 8,668 tok/s) | 4.05% | same | the same |
| prefill 256K | +7.56% (8,731 / 8,772 tok/s) | 2.05% | better | +7.5% |

- The earlier build's figures are another session's (1-2% drift between sessions).
- **The 4-request grid shapes are the earlier build's:** 4x8 rows at 4K take 16.74 / 16.69 ms (earlier 16.80 / 16.61),
  4x12 17.89 / 17.98 (17.998 / 17.79), 4x24 21.28 / 21.27 (21.25 / 21.26).
- **Model step:** acceptance length 2.301 and 2.304, perplexity 1.31888, the same greedy output on both boots and at
  `tp4-ring` (`5accd142`).

**Live checks:** #955 (a tool call quoting DSML keeps its arguments; quoted DSML in a reply stays text), #1013
(`/v1/completions` with a JSON-schema grammar under the `deepseek_v41` reasoning parser is schema-valid from the first
token) and #972 (a client disconnect aborts the engine request; keep-alive frames every 15 s) pass.

**c1 decision check:** the four boots made the same accept decision at every step of all 57 requests (every pair of
boots equal, by [c1_decisions.py](../../../../../2026-11-10-tp4-ring-determinism/boot-differences/c1_decisions.py)).
c1 output is 79.75 / 79.63 / 79.70 / 79.70 tok/s, and 1.301-1.302 tokens are accepted a draft. Before `0799`, the
boots of candidate-58's c1 arms on this testbed split wherever the shared expert stayed on the aux stream with the
drafter's query graph replayed ([../../gate-0799/README.md](../../gate-0799/README.md)).

**Time to ready:** 113.1 s for the gate's boot, the image's first here; 129.8 and 128.7 s for the screen's; 127.5 s for
the live checks'; 125.5, 128.8, 125.3 and 122.1 s for the c1 check's.
