# candidate-54's rebuilt image at tp3-switched (sp5..sp7), 2026-11-11

The lab's outputs from sp5's window (`spark-0013`), 13:15-15:31Z, copied from sp5 `results/private/`. The image is
`spark-ds41f:candidate-54`, image ID `d502217e` (tree `ad2dc11f`), built on dgx4 and shipped to sp5 over the fabric;
`build distribute` verified the same ID on sp5..sp7. Both jobs ran the TP3 profiles with the L2 prefetch caps
(15/12/12 MB, PR #1126).

| Path | What |
| --- | --- |
| `determinism/candidate-54-gate-caps-tp3-switched/`, `lab/candidate-54-gate-caps-tp3-switched-table.txt` | The gate on two boots ([gate-caps-tp3-switched.json](../../../gate-caps-tp3-switched.json)) |
| `lab/candidate-54-0799-screen-tp3-switched-{screen,single-stream,sampled,stall}-table.txt` | The screen's four suites ([screen-0799-tp3-switched.json](../../../screen-0799-tp3-switched.json)), two boots each. The node keeps one table file for the run, which each suite's table replaces, so these are cut from the window log. The suite runs are `../../references/candidate-54-0799-tp3-switched-{screen,screen-again,screen-sampled,screen-again-sampled,screen-single-stream,screen-again-single-stream,stall,stall-again}/` |
| `lab/window.log` | The window log's part for the two jobs |
| `lab/candidate-54-gate-caps-tp3-switched-records.sha256` | The gate's record files, kept on sp5 |

The engine logs (`results/private/suites/candidate-54-0799-screen-tp3-switched/*.engine-sp{5,6,7}.log`) and the
startup-memory records (`results/private/determinism/candidate-54-0799-screen-tp3-switched/`) stay on sp5.

**Gate:** all eight runs identical in every scenario, on candidate-54's recorded TP3 hashes 5c22ce41/bce6092e
(logprobs) and 26a739a9/12cfee9b (text), the same as `0799`'s gate without the caps
([../../gate-0799/README.md](../../gate-0799/README.md)). Distinct prompts give 153.2 / 152.3 tok/s at c8 and
116.9 / 116.2 at c4 (without the caps: 153.35 and 116.30).

**Screen, candidate-54 against 1107 (`screen-tp3`):**

| Metric | Change | Margin | Verdict | Earlier build (`e1de71be`, no caps) |
| --- | --- | --- | --- | --- |
| engine grid geometric mean | -0.10% (-0.21% / +0.04% by boot) | 0.33% | same | +2.87%, worse (one boot) |
| c8 output | +0.87% (157.8 against 156.4 tok/s) | 1.27% | same, one boot | +1.2%, same (one boot) |
| c8 paired decode | +0.64% | 0.88% | better, one boot | +0.9%, same (one boot) |
| prefill 32K / 256K | not measured | | | not measured |

- **The caps remove the 1-request loss:** 1x2 rows at 4K take 18.53 / 18.52 ms against 1107's 18.71 (earlier build
  without caps: 1-request shapes 5.5-7.2% slower), 1x3 at 4K 18.74 / 18.75 against 18.91, and 1x3 at 1016K 0.6% faster.
- **The 8x24 and 4x24 shapes are slower:** 8x24 rows by 1.5-2.3% at 4K, 1.0-1.4% at 64K and 1.5-1.7% at 256K; 4x24
  at 4K by 0.8-1.4%. 8x48 at 4K is 1.3-1.6% faster. The grid's mean is the same.
- **sp5's 4 GiB bench floor halted two steps,** as it did for the earlier build: the first boot's prefill after one
  32K sample (MemAvailable 3.94 GiB) and the second boot's c8 in its sixth window (3.99 GiB). So c8 is the first
  boot's alone and prefill is not measured at TP3.
- **Model step:** the two boots' greedy responses are identical (128 of 128), acceptance length 2.301 and 2.306,
  perplexity 1.31608 on both, identical teacher-forced distributions.

**Companions against candidate-50** (`candidate-50-no1009`'s recorded runs; verdicts against the run's own repeat):

| Suite | Step | Change | Margin | Verdict | Earlier build |
| --- | --- | --- | --- | --- | --- |
| `single-stream` | c1 paired decode | +2.37% | 2.73% | same | +0.55%, same |
| `single-stream` | c1 output | +1.23% (63.13 / 63.20 against 62.36 tok/s) | 4.97% | same | |
| `sampled` | c16 / c8 / c1 paired decode | -0.29% / +0.30% / -0.17% | 1.97% / 1.76% / 2.35% | same | same |
| `sampled` | c16 / c8 / c1 output | +0.11% / +0.47% / +0.70% | 1.46% / 2.07% / 6.32% | same | same |
| stall | prefill 131,072 under decode | stall-3 +1.78% (4,871 / 4,867 against 4,786 tok/s), stall-6 +0.54%, stall-7 +0.78% | 0.65% / 3.26% / 3.40% | stall-3 better, the others same | stall-3 +0.95%, same |

The stall steps' decode waits are the same (stall-3's p90 1.1-1.2% better).

**Time to ready:** 115.8 s for the gate's first boot, the image's first on sp5..sp7; 125.0 s for its second; 124.6 to
131.5 s for the screen's eight.
