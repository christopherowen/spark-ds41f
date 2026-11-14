# Candidate-57 + Online DSpark Adaptation (#1002)

Combines `c57-stack` (`experiments/2026-11-14-c57-stack`) with `#1002` online DSpark
missed-verified head adaptation (`W2_PACKED` Triton fast-path + FP32 master buffer
`w1_fp32` + `residual_orth_ce` loss mode, without `c56-host`).

## Two-boot `screen` vs `candidate-55` Build 9 (`tp4-switched`, `sp1..sp4`)

- `grid` decode geometric mean (20 shapes): **`-2.08%` better** (`±0.15%`, repeat `-0.00%`, margin `0.28%`, reference repeat `-0.07%`, recorded A/A floor `0.23%`)
- `c16` paired decode tok/s: **`-0.25%` same** (`±0.63%`, repeat `+0.23%`, margin `0.68%`; output `268.087 / 268.913 tok/s`, `-0.31% ±0.79%`)
- `prefill 32,768 tok/s`: **`+2.58%` better** (`8,996.995 / 8,960.091 tok/s`, `±1.44%`, repeat `-0.41%`, margin `1.68%`)
- `prefill 262,144 tok/s`: **`+3.23%` better** (`9,086.150 / 9,073.037 tok/s`, `±0.31%`, repeat `-0.14%`, margin `0.45%`)

## Verification (`tp4-switched`, `sp1..sp4`)

- `workload-adapt-on` / `again`: `coding_acceptance_length = 5.3285` (`161.0675 / 160.5488 tok/s`), `heldout_acceptance_length = 2.3114` (`87.1375 / 86.8331 tok/s`), identical `text_sha256` (`9df935a06b...`), `0` foreign requests.
- `determinism-adapt-on` / `again`: `identical: true` (`logprobs: true`, `192` tokens, `0` token/logprob diffs across `c8`, `staggered`, `mixed`, `cached`, `again`, `mixed15`), matching reference hashes (`18d619615e...` / `d65686498e...`).
- `digests-adapt-on` / `again`: `112` digests across ranks `[0, 1, 2, 3]` (`28` updates, last update `6912`), `identical: true`, `differing_updates: []`, last drift `0.204583 / 0.202008`.
