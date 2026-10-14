# Overnight queue (2026-10-14)

Owner: "queue up multiple hours of benchmarking or other work so we have things to check
when we come back". It runs unattended after the kernel-contracts round 2
([relu2](../2026-10-14-kernel-contracts/relu2.json)). The queue runner closes the window
10 minutes after the last job and restores production (pinned). Every job uses tooling
already run live today.

| order | spec | what | about |
| ---: | --- | --- | ---: |
| 1 | [release-baseline.json](release-baseline.json) | the `release` suite on r6k as deployed (pinned): model behaviour on 256 prompts, engine grid, 16 and 8 streams, prefill 32K-1M. The next candidate's reference; its model corpus is the text later builds score | 50 min |
| 2 | [context.json](context.json) | decode at depth on each stream's cached document (`servebench run --context`, its first live run): 16 streams at 64K, 8 at 256K, 4 at 512K, 2 at 1,000,000 tokens | 40 min |
| 3 | [goodput.json](goodput.json) | Poisson arrivals at 0.4 / 0.6 / 0.8 / 1.0 requests/s; goodput and attainment against 2 s to first token and 100 ms per token | 26 min |
| 4 | [suite-floor.json](suite-floor.json) | the `screen` suite on three boots of one build: the boot-to-boot floor of every suite measure | 45 min |
| 5 | [pin-boots.json](pin-boots.json) | eight more interleaved boots, unpinned (u5-u8) and pinned (p9-p12), one grid each; joins [boot-spread](../2026-10-14-boot-spread/README.md)'s twelve | 30 min |

## Reading the results

- `scripts/lab.py report <run>` on dgx1: each run's table, plus node telemetry per step.
- Release baseline: `results/private/suites/overnight-release-baseline/r6k-pinned/`.
- Context and goodput: `results/private/overnight/{context,goodput}/*.json`; compare
  with `python3 -m servebench compare`.
- Suite floor: `python3 scripts/servebench_suite_table.py overnight-suite-floor s1 s2 s3`.
- Pinning: `python3 -m servebench spread results/private/boot-spread/u{1..8}-grid.json --against results/private/boot-spread/{p,c}{1..12}-grid.json`,
  with the files that exist.
