# candidate-50 at tp4-ring (dgx1..dgx4), 2026-11-09

These are the lab's outputs from the window of 13:38–16:09Z, copied from dgx1 `results/private/`.

| Path | What |
| --- | --- |
| `lab/candidate-50-image/` | The image build on dgx4, and the release-test kernel job's verdicts (pass, 2,619 s; every node agrees) |
| `failed-starts/20261109T144453Z-*` | The boot of the full candidate-50 release arm (`tp4-dev.json`), which fails in `indexer_score`, as the gate boot did (`../tp4-boot-failure/`) |
| `failed-starts/20261109T145127Z-*` | The `tp4-no1012.json` control, which fails the same way |
| `determinism/candidate-50-tp4-no1009/`, `lab/candidate-50-tp4-no1009-table.txt` | The gate without #1009: temperature-0 outputs identical in every scenario, hashes 8e46794a/11ec073c (logprobs) and 8b549cc4/12fd8257 (text). Distinct prompts give 271.6 / 201.9 / 153.5 tok/s at c16 / c8 / c4 |
| `determinism/candidate-50-screen-tp4/`, `lab/candidate-50-screen-tp4-table.txt` | The screen, ABBA: 1107, candidate-50 without #1009 twice, 1107 again. The suite runs are in `../references/` |
| `determinism/candidate-50-live-checks-tp4/` | The live checks of #955, #1013 and #972, all passed |
| `lab/candidate-50-window.log` | The window's log |

**Reading the screen table.** Its first arm is 1107, so it prints changes from candidate-50 to 1107, and the verdict column describes 1107 relative to the candidate. Read the other way, candidate-50 without #1009 against 1107:
- engine grid geometric mean 0.80% faster (±0.15%);
- c16 paired decode 1.45% faster (margin 0.56%);
- c16 output 269.4 against 265.2 tok/s;
- prefill 2.0% faster at 32K and 6.0% faster at 256K;
- greedy responses identical (128/128), KL 0, perplexity unchanged.
