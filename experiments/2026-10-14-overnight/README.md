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

## First pass (09:25-12:35 UTC) and the second queue

- **kernel-contracts round 2:** every NaN-keeping form passed its contracts and kept
  r6k's bits. The best indexer form, `bits`, was within 1% everywhere except 8K x 2048
  (1.022). No SwiGLU form was within 1%. See the kernel-contracts README.
- **Context sweep (r6k, pinned):** every document stayed cached (hit rate 0.996-1.0).

  | streams x depth | tok/s | time per output token p50 |
  | --- | ---: | ---: |
  | 16 x 64K | 193.9 ±1.1% | 77.8 ms |
  | 8 x 256K | 147.3 ±1.1% | 48.2 ms |
  | 4 x 512K | 102.8 ±1.3% | 29.5 ms |
  | 2 x 1M | 62.4 ±3.2% | 17.8 ms |

- **Goodput:** every request ended "no output". The stream closed without content, which
  a server error event inside an HTTP 200 stream would do. Closed loops on the same server
  worked. servebench now records such events, and the curve reruns.

Second queue, behind pin-boots:

| order | spec | what | about |
| ---: | --- | --- | ---: |
| 6 | [goodput2.json](goodput2.json) | the goodput curve again; a failure now names the server's error | 26 min |
| 7 | [pin-c16.json](pin-c16.json) | sixteen streams on four unpinned and four pinned boots, alternating. Pinning's effect on real throughput (one pair so far: +0.4%) | 64 min |
| 8 | [context-r6h.json](context-r6h.json) | the context sweep on r6h, pinned ([r6h-big.json](r6h-big.json)) against r6k's | 40 min |
