# candidate-54's rebuilt image at tp4-ring (dgx1..dgx4), 2026-11-11

The lab's outputs from dgx1's window, 12:14-12:56Z, copied from dgx1 `results/private/`. The image is
`spark-ds41f:candidate-54`, image ID `d502217e`, tree `ad2dc11f` (with `0799`): built on dgx4 at 11:22Z and copied
along the ring, with the same 57 layers on every node.

| Path | What |
| --- | --- |
| `lab/candidate-54-image/`, `lab/build-window.log` | The build job ([build.json](../../../build.json), 11:11-12:12Z): the image build on dgx4 (patch head `15833ed1`, tree `ad2dc11f`), the ring copies, and the release-test kernel job's verdict |
| `determinism/candidate-54-gate-final-tp4/`, `lab/candidate-54-gate-final-tp4-table.txt` | The gate on one boot ([gate-final-tp4.json](../../../gate-final-tp4.json)) |
| `lab/candidate-54-0799-screen-tp4-table.txt` | The screen ([screen-0799-tp4.json](../../../screen-0799-tp4.json)): two boots against 1107's recorded screen. The suite runs are `../../references/candidate-54-0799-tp4-ring-screen{,-again}/` |
| `boot-differences/candidate-54-c1-decisions-tp4/`, `determinism/candidate-54-c1-decisions-tp4/` | The c1 decision check ([c1-decisions-tp4.json](../../../c1-decisions-tp4.json), 13:37-13:58Z): each boot's `c1.json` with every request's per-step token counts |
| `lab/window.log`, `lab/c1-window.log` | The window log's parts for the gate and screen, and for the c1 check |
| `lab/candidate-54-gate-final-tp4-records.sha256` | The gate's record files, kept on dgx1 |

**Release tests** (the build job, on dgx4): 10,171 passed, 311 skipped, 0 failed (2,612 s). The earlier build's
count is the same, and `0799`'s first build failed one of them (`test_routed_moe_pdl_preloads_before_wait`), which
`0799` now updates.

**Gate:** all four runs identical in every scenario, on the recorded TP4 hashes 18d61961/d6568649 (logprobs) and
0a2dacdd/12fd8257 (text). Distinct prompts give 271.1 / 200.7 / 154.2 tok/s at c16 / c8 / c4.

**Screen, candidate-54 against 1107** (verdicts against the run's own repeat):

| Metric | Change | Margin | Verdict | Earlier build (`e1de71be`) |
| --- | --- | --- | --- | --- |
| engine grid geometric mean | -4.94% (faster) | 0.30% | better | -4.78% |
| c16 paired decode | +1.35% | 1.11% | better | +1.61% |
| c16 output | +1.74% (270.2 / 269.6 against 265.2 tok/s) | 0.94% | better | 270.7 tok/s |
| prefill 32K | +2.10% (8,689 / 8,712 tok/s) | 2.49% | same | +3.46% (8,793 tok/s) |
| prefill 256K | +7.35% (8,849 / 8,859 tok/s) | 0.50% | better | +7.84% (8,873 tok/s) |

- The earlier build's figures are another session's (1-2% drift between sessions).
- **The 4-request grid shapes are not slower:** 4x8 rows at 4K take 16.47 / 16.55 ms against the earlier build's
  16.67 / 16.77, and 4x12 and 4x24 are the same. The c4 distinct-stream figure below the earlier one at
  `tp4-switched`'s `0799` gate ([../../gate-0799/README.md](../../gate-0799/README.md)) does not show here.
- **Model step:** acceptance length 2.301 and 2.304, perplexity on its own greedy text 1.31888, the same greedy
  output on both boots (`5accd142`), as the earlier build recorded.

**c1 decision check:** the four boots made the same accept decision at every step of all 58 requests (every pair of
boots equal, by [c1_decisions.py](../../../../../2026-11-10-tp4-ring-determinism/boot-differences/c1_decisions.py)).
c1 output is 80.65 / 80.55 / 80.61 / 80.65 tok/s, and 1.302-1.304 tokens are accepted a draft. This is the testbed
where the gate failed and where candidate-58's c1 step trace found a decision split between boots.

**Time to ready:** 105.3 s for the gate's boot, the image's first, then 113.6 and 111.5 s for the screen's, and
110.6-112.8 s for the c1 check's four.
