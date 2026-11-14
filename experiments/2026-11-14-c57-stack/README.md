# Candidate-57 Patch Stack Screen

Four vLLM commits cherry-picked onto `candidate-55` Build 9 (`44db6a403`):

```
3793961db perf(dsv41): branchless sparse MLA stage-2 online softmax over inactive segments
0ac8a53bb perf(spec_decode,sample): skip dead verification rows in sampler
2171c005e perf(dsv41): SP ratio-2 compressor projection and CED encoder-only chunks
dc712a66f perf(dsv41): fuse DSpark confidence calibration and keep active num_reqs in dummy runs
```

## Two-boot `screen` vs `candidate-55` Build 9

- **`tp4-ring` (`dgx1..dgx4`)**:
  - `grid` decode geometric mean (20 shapes): **`-1.68%` better** (`±0.12%`, repeat `+0.09%`, margin `0.29%`)
  - `c16` paired decode tok/s: **`-0.17%` same** (`±0.36%`, margin `0.45%`; output `270.843 / 270.750 tok/s`)
  - `prefill 32,768 tok/s`: **`+3.41%` better** (`9,160.002 / 9,146.832 tok/s`, `±0.81%`, margin `0.95%`)
  - `prefill 262,144 tok/s`: **`+3.40%` better** (`9,196.827 / 9,179.900 tok/s`, `±0.91%`, margin `1.00%`)
  - `model` (128 prompts): `128/128` greedy identical (`KL = 0.000e+00`, perplexity `1.31888`)
- **`tp4-switched` (`sp1..sp4`)**:
  - `grid` decode geometric mean (20 shapes): **`-2.42%` better** (`±0.15%`, repeat `-0.00%`, margin `0.27%`)
  - `c16` paired decode tok/s: **`-0.14%` same** (`±0.33%`, margin `0.41%`; output `269.203 / 268.833 tok/s`)
  - `prefill 32,768 tok/s`: **`+2.84%` better** (`8,978.744 / 9,024.136 tok/s`, `±1.88%`, margin `2.16%`)
  - `prefill 262,144 tok/s`: **`+3.82%` better** (`9,147.307 / 9,116.495 tok/s`, `±0.49%`, margin `0.70%`)
  - `model` (128 prompts): `128/128` greedy identical (`KL = 0.000e+00`, perplexity `1.31888`)
- **`tp3-switched` (`sp5..sp7`)**:
  - `grid` decode geometric mean (15 shapes): **`-0.66%` better** (`±0.11%`, repeat `+0.20%`, margin `0.32%`)
  - `c8` paired decode tok/s: **`+0.13%` same** (`±0.52%`, repeat `-0.13%`, margin `0.59%`; output `158.903 / 158.743 tok/s`, `+0.09% ±1.36%`)
  - `prefill 32,768 tok/s`: **`+0.19%` repeat same** (`6,784.972 / 6,797.665 tok/s`, `±0.99%`)
  - `prefill 262,144 tok/s`: **`-0.21%` repeat same** (`6,704.541 / 6,690.368 tok/s`, `±0.60%`)
  - `model` (128 prompts): `128/128` greedy identical (`KL = 0.000e+00`, perplexity `1.31608`)

## Two-boot `gate` and 4-boot `c1-decisions`

- **`tp4-ring` (`dgx1..dgx4`)**:
  - `gate`: all 8 determinism/greedy runs `identical: true` (`0` token/logprob diffs across all 6 scenarios), matching `candidate-55` b9 hashes (`18d619615e...` / `d65686498e...`). Distinct streams: `c16` `271.64 / 271.72 tok/s`, `c8` `200.44 / 201.55 tok/s`, `c4` `157.81 / 155.76 tok/s`, `0` foreign requests.
  - `c1-decisions` (4 boots): `82.93 / 82.76 / 82.76 / 82.87 tok/s` (mean `82.83 tok/s`, `35.75 steps/s`, `0` failed, vs `candidate-55` b9 `82.85 tok/s`).
- **`tp4-switched` (`sp1..sp4`)**:
  - `gate`: all 8 determinism/greedy runs `identical: true` (`0` token/logprob diffs across all 6 scenarios), matching `candidate-55` b9 hashes (`18d619615e...` / `d65686498e...`). Distinct streams: `c16` `269.72 / 268.37 tok/s`, `c8` `198.87 / 199.43 tok/s`, `c4` `154.51 / 154.49 tok/s`, `0` foreign requests.
  - `c1-decisions` (4 boots): `82.77 / 82.81 / 82.83 / 82.74 tok/s` (mean `82.79 tok/s`, `35.70 steps/s`, `0` failed).
- **`tp3-switched` (`sp5..sp7`):**
  - `gate`: all 8 determinism/greedy runs `identical: true` (`0` token/logprob diffs across all 6 scenarios), matching reference hashes (`5c22ce41c1...` / `26a739a99b...`). Distinct streams: `c8` `153.95 / 153.66 tok/s`, `c4` `117.78 / 117.78 tok/s`, `0` foreign requests.

## Kernel release-tests (sp8)

- **157 files (`chunk000`–`chunk015`, `2,861.4 s`)**: **`10,295 passed, 357 skipped, 0 failed`** (receipt: `experiments/2026-11-14-c57-stack/receipts/release-tests/verdict.json`).
