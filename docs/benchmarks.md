# Benchmarks of earlier releases

The current release's reference numbers are in the README's
[Performance](../README.md#performance) section and, per shape, in
[current-state.md](current-state.md). This page keeps the comparisons each earlier
release was promoted on, newest first. They are records: every table names the
releases, the date and the suite run it comes from, and none is measured again.
[Release history](../README.md#release-history) says what each release changed.

## Independent quality evaluations

Community end-to-end evaluations archived with verbatim receipts under
[manifests/evals/](../manifests/evals/README.md):

| Benchmark | Release & Shape | Score | Soak / Throughput | Receipt |
| --- | --- | ---: | --- | --- |
| GPQA Diamond (`inspect_ai`, 4 epochs × 198, `T=1.0`, `reasoning_effort=100`) | `spark-ds41f-1120`, `tp3-triangle-4k` (stock 4 KiB kernel `7.0.0-1025-nvidia`) | **`0.922 ± 0.017`** (`792/792`) | `8h 04m`, `5.80M` tok (`5.47M` reasoning), ~`192–209 tok/s` at 8 streams, `0` errors | [`header.json`](../manifests/evals/2026-11-25-spark-ds41f-1120-tp34k-gpqa-diamond.json) ([forum #210](https://forums.developer.nvidia.com/t/new-deepseek-4-1-flash-recipe/384438/210), [#212](https://forums.developer.nvidia.com/t/new-deepseek-4-1-flash-recipe/384438/212)) |

## Earlier release comparisons

spark-ds41f-1030 against spark-ds41f-1025 on four nodes, both serving
DeepSeek-V4.1-Flash-1017 (2026-10-29), measured with the `release`, `single-stream` and
`sampled` suites in one session (candidate 12, spark-ds41f-1025, candidate 12 again;
candidates 13 and 14 keep its digests, kernels and bytes read;
[results](../experiments/2026-10-26-spark-ds41f-candidate/README.md);
[release notes](../experiments/2026-10-26-spark-ds41f-candidate/release-notes.md);
[suite run](../manifests/suites/2026-10-30-spark-ds41f-1030-tp4/)). Bold changes exceed their
95% interval and the run's own repeat.

| Workload | TP4 spark-ds41f-1025 | TP4 spark-ds41f-1030 |
| --- | ---: | ---: |
| Decode step time, geometric mean of 20 shapes | | **-4.62%** ±0.15 |
| Decode, 16 streams output / paired decode (tok/s) | 251.7 | 257.7 (**+2.40%** ±0.68; paired **+2.05%** ±0.41) |
| Decode, 8 streams output / paired decode (tok/s) | 194.6 | 198.4 (**+1.96%** ±0.87; paired **+2.02%** ±0.51) |
| One conversation at a time, paired decode / output (tok/s) | 75.4 / 75.4 | 78.3 (**+3.61%** ±0.16) / 78.0 (+3.41% ±3.35) |
| Sampled decode (temp 1.0), c16 / c8 / c1 paired change | | **+1.87%** / **+2.77%** / **+2.68%** |
| Poisson arrivals at 0.7 req/s, output (tok/s) | 180.4 | 180.4 (-0.01% ±18.0) |
| Source-text prefill, 32K (tok/s) | 6,744 | 7,313 (**+8.44%** ±1.83) |
| Prefill, 131K | 6,798 | 7,396 (**+8.79%** ±0.24) |
| Prefill, 262K | 6,477 | 7,120 (**+9.93%** ±0.23) |
| Prefill, 524K | 5,870 | 6,538 (**+11.38%** ±0.14) |
| Prefill, 1M | 4,948 | 5,549 (**+12.16%** ±0.30) |
| 131K prefill under 15 decoding streams (tok/s) | 6,447 | 6,978 (**+8.23%** ±0.78) |
| Short request TTFT p50 during 131K prefill, 3 / 11 streams (s) | 12.36 / 12.79 | **3.30** (**-73.3%**) / **3.59** (**-71.9%**) |

All 256 greedy responses are identical to spark-ds41f-1025's (teacher-forced KL 4.2e-8 nats
over 65,536 positions; 0 between the candidate boots).

spark-ds41f-1025 against spark-ds41f-1022 on four nodes (`tp4-ring` at 16,384 batched tokens)
and three nodes (`tp3-triangle` at 8,192 batched tokens), both serving
DeepSeek-V4.1-Flash-1017 (2026-10-25;
[results](../experiments/2026-10-25-spark-ds41f-candidate/README.md);
[release notes](../experiments/2026-10-25-spark-ds41f-candidate/release-notes.md); suite runs for
[TP4](../manifests/suites/2026-10-25-spark-ds41f-1025-tp4/) and
[TP3](../manifests/suites/2026-10-25-spark-ds41f-1025-tp3/)).

| Workload | TP4 1022 | TP4 1025 | TP3 1022 | TP3 1025 |
| --- | ---: | ---: | ---: | ---: |
| Decode step time, geometric mean (20 TP4 / 15 TP3 shapes) | | **-2.38%** ±0.15 | | **-1.65%** ±0.16 |
| Decode, 16 streams (tok/s) | 251.1 | 251.4 (+0.12% ±0.81) | | |
| Decode, 8 streams (tok/s) | 193.2 | 194.9 (+0.88% ±0.79) | 154.4 | 154.7 (+0.22% ±1.34) |
| Decode, 4 streams (tok/s) | | | 117.5 | 118.2 (+0.59% ±1.99) |
| Poisson arrivals (0.7 TP4 / 0.42 TP3 req/s), output (tok/s) | 180.4 | 180.4 (+0.01% ±17.7) | 118.8 | 118.8 (-0.03% ±22.3) |
| One conversation at a time (`single-stream`, tok/s) | 73.8 | 75.3 (+2.11% ±3.82) | 59.4 | 60.5 (+1.85% ±7.06) |
| Source-text prefill, 32K (tok/s) | 5,994 | 6,762 (**+12.80%** ±0.95) | 4,469 | 5,637 (**+26.14%** ±0.74) |
| Prefill, 131K | 5,969 | 6,796 (**+13.85%** ±0.87) | 4,389 | 5,628 (**+28.22%** ±0.67) |
| Prefill, 262K | 5,700 | 6,482 (**+13.73%** ±0.79) | 4,194 | 5,320 (**+26.84%** ±0.49) |
| Prefill, 524K (TP4) / 500K (TP3) | 5,194 | 5,863 (**+12.89%** ±0.34) | 3,837 | 4,815 (**+25.50%** ±0.34) |
| Prefill, 1M | 4,420 | 4,942 (**+11.81%** ±0.49) | 3,226 | 3,952 (**+22.5%**) |

All 256 greedy responses are identical to spark-ds41f-1022's on both topologies, with
teacher-forced KL 0.

spark-ds41f-1022 against r6x on four nodes, both serving DeepSeek-V4.1-Flash-1017
(2026-10-23), measured with the trimmed promotion suite in one session (spark-ds41f-1022,
r6x, spark-ds41f-1022 again; [results](../experiments/2026-10-22-spark-ds41f-1022/README.md#the-ring-2026-10-23);
[suite run](../manifests/suites/2026-10-22-spark-ds41f-1022-tp4/)). Bold changes exceed their 95%
interval and the run's own repeat.

| Workload | TP4 r6x | TP4 spark-ds41f-1022 |
| --- | ---: | ---: |
| Decode step time, geometric mean of 20 shapes | | **-1.7%** ±0.16 |
| Decode, 16 streams (tok/s) | 253.1 | 256.5 (+1.3% ±0.8) |
| One conversation at a time (tok/s) | 74.1 | 74.1 (0.0% ±8.3) |
| Source-text prefill, 32K (tok/s) | 6,034 | 6,030 (-0.1% ±0.5) |
| Prefill, 131K | 5,982 | 5,985 (+0.1% ±0.6) |
| Time to first token p50, 16 streams / one stream (s) | 0.396 / 0.100 | 0.400 / 0.129 |

All 64 greedy responses are identical to r6x's. A request alone waits one decode step
longer for its first token: sink rejection verifies it before it is sent.

spark-ds41f-1022 on three nodes at 1M (2026-10-22): the same comparison on its image
`-1022-2` (one session; [results](../experiments/2026-10-22-spark-ds41f-1022/README.md#promotion-comparison-2026-10-22-1003-1105-cet))
measured the decode step **1.1%** faster (±0.16), eight streams 151.3 to 153.6 tok/s (same), one
stream level and prefill 0.7-0.8% lower; vLLM 0226 then restored prefill (its A/B: 0.6-1.4%
faster on every paired prompt), and the promoted image measured 4,460 / 4,384 tok/s at
32K / 131K against r6x's 4,461 / 4,386. Time to first token at four and eight streams is level
with r6x's, and one stream is one decode step longer. Reference run:
[manifests/suites/2026-10-22-spark-ds41f-1022-tp3](../manifests/suites/2026-10-22-spark-ds41f-1022-tp3/).

r6x against r6v on four nodes, both serving DeepSeek-V4.1-Flash-1017 (2026-10-18),
measured with the `release` suite in one session (r6x, r6v, r6x again;
[results](../experiments/2026-10-18-r6x/README.md);
[suite run](../manifests/suites/2026-10-19-karmic-kraken-r6x-tp4/)).

| Workload | TP4 r6v | TP4 r6x |
| --- | ---: | ---: |
| Source-text prefill, 32K (tok/s) | 5,987 | 5,989 (0.0% ±1.1) |
| Prefill, 128K | 5,945 | 5,947 (0.0% ±0.7) |
| Prefill, 256K | 5,681 | 5,680 (0.0% ±0.2) |
| Prefill, 512K | 5,172 | 5,173 (0.0% ±0.3) |
| Prefill, 1M | 4,406 | 4,407 (0.0% ±0.6) |
| Decode step time, geometric mean of 20 shapes | | **-2.5%** ±0.15 |
| Decode, 16 streams (tok/s) | 241.7 | 250.3 (**+3.6%** ±0.9) |
| Decode, 8 streams | 184.4 | 191.5 (**+3.8%** ±1.1) |
| Poisson arrivals at 0.7 req/s, output (tok/s) | 180.4 | 180.4 (0.0% ±17) |
| One conversation at a time (tok/s, `single-stream`) | 71.0 | 74.1 (+4.5% ±4.5; TTFT p50 0.10 s) |

All 256 greedy responses are identical to r6v's; acceptance length is level (2.27).

r6x at 1M context against r6v at 512K on three nodes, both serving
DeepSeek-V4.1-Flash-1017 (2026-10-19), measured with the `release-tp3` suite in one
session (r6x, r6v, r6x again;
[results](../experiments/2026-10-19-r6x-tp3-1m/README.md);
[suite run](../manifests/suites/2026-10-19-karmic-kraken-r6x-tp3/)). The suite's prefill
stops at 500K; one cold 1,000,000-token prefill was measured on its own.

| Workload | TP3 r6v (512K) | TP3 r6x (1M) |
| --- | ---: | ---: |
| Source-text prefill, 32K (tok/s) | 4,425 | 4,465 (+0.9% ±0.9) |
| Prefill, 128K | 4,357 | 4,390 (**+0.8%** ±0.7) |
| Prefill, 256K | 4,167 | 4,196 (**+0.7%** ±0.3) |
| Prefill, 500K | 3,810 | 3,835 (**+0.7%** ±0.3) |
| Prefill, 1M (one cold run) | | 3,227 (310 s) |
| Decode step time, geometric mean of 14 shapes | | **-1.3%** ±0.14 |
| Decode, 8 streams (tok/s) | 147.1 | 152.7 (**+3.8%** ±1.5) |
| Decode, 4 streams | 111.5 | 116.0 (**+4.0%** ±1.9) |
| Poisson arrivals at 0.42 req/s, output (tok/s) | 117.6 | 118.3 (+0.7% ±22) |
| One conversation at a time (tok/s, `single-stream`) | 57.0 | 59.7 (+4.7% ±6.3) |
| KV pool, startup line (4,096 batched tokens) | 2,845,543 tokens (5.43x 512K; holds 7 at once) | 4,765,898 tokens (4.55x 1M; holds 5 at once) |

All 256 greedy responses are identical to r6v's. During the cold 1M prefill dgx1's
lowest MemAvailable was 5.93 GiB, above its 3 GiB steady guard.

r6v against r6n on four nodes, both serving DeepSeek-V4.1-Flash-1017 (2026-10-17),
measured with the `release` suite in one session (r6v, r6n on its production ring
curves; [results](../experiments/2026-10-17-r6v/README.md);
[suite run](../manifests/suites/2026-10-17-karmic-kraken-r6v-tp4/)).

| Workload | TP4 r6n | TP4 r6v |
| --- | ---: | ---: |
| Source-text prefill, 32K (tok/s) | 5,762 | 5,982 (**+3.8%** ±0.6) |
| Prefill, 128K | 5,706 | 5,946 (**+4.2%** ±1.1) |
| Prefill, 256K | 5,467 | 5,676 (**+3.8%** ±0.6) |
| Prefill, 512K | 4,988 | 5,171 (**+3.7%** ±0.2) |
| Prefill, 1M | 4,269 | 4,405 (**+3.2%** ±0.5) |
| Decode step time, geometric mean of 20 shapes | | -0.05% ±0.14 |
| Decode, 16 streams (tok/s) | 243.5 | 243.8 (+0.1% ±1.0) |
| Decode, 8 streams | 185.4 | 185.4 (0.0% ±1.1) |
| Poisson arrivals at 0.7 req/s, output (tok/s) | 180.2 | 180.4 (+0.1% ±17) |
| One conversation at a time (tok/s, `single-stream`) | | 71.1 (TTFT p50 0.10 s) |

No greedy response is identical to r6n's (WO-A's input quantization, DeepSeek's current
reference); acceptance length is level (2.27).

r6u against r6s on three nodes, both serving DeepSeek-V4.1-Flash-1017 (2026-10-17),
measured with the `release-tp3` suite ([results](../experiments/2026-10-17-r6u/README.md);
[suite run](../manifests/suites/2026-10-17-karmic-kraken-r6u-tp3/)). r6u is measured with
vLLM 0179 mounted over its image through 0178 (the promoted image carries the same
files), in the session after r6s's, so the decode step time is cross-session (the same
build read 1.5% apart in two sessions that day).

| Workload | TP3 r6s | TP3 r6u |
| --- | ---: | ---: |
| Source-text prefill, 32K (tok/s) | 4,414 | 4,405 (-0.2% ±1.2) |
| Prefill, 128K | 4,346 | 4,343 (-0.1% ±0.6) |
| Prefill, 256K | 4,153 | 4,154 (0.0% ±0.5) |
| Prefill, 500K | 3,801 | 3,801 (0.0% ±0.2) |
| Decode step time, geometric mean of 15 shapes | | -1.0% ±0.13 |
| Decode, 8 different prompts at once (tok/s) | 146.6 | 145.6 (-0.7% ±1.3) |
| Decode, 4 different prompts at once | 112.5 | 111.3 (-1.1% ±1.4) |
| Poisson arrivals at 0.42 req/s, output (tok/s) | 117.7 | 117.3 (-0.3% ±21) |

r6u's image through 0178 quantized WO-A's input in the GEMM's column tiles and was
2.8-3.2% slower at prefill (the same afternoon, r6u / r6s / r6u again); 0179 prepares
it once per row and changes no output bit. No greedy response is identical to r6s's
(WO-A's input quantization); r6s's teacher-forced KL on r6u's text is 0.023 nats.

r6s against r6n on three nodes (2026-10-16), measured with the `release-tp3` suite
in one session (r6s, r6n, r6s again; [results](../experiments/2026-10-16-r6s/README.md);
[suite run](../manifests/suites/2026-10-16-karmic-kraken-r6s-tp3/)), each on its own
build's cost curves. Each change is shown with its 95% interval and judged against
the run's own repeat; it counts (**bold**) only beyond both.

| Workload | TP3 r6n | TP3 r6s |
| --- | ---: | ---: |
| Source-text prefill, 32K (tok/s) | 4,199 | 4,425 (**+5.4%** ±1.8) |
| Prefill, 128K | 4,128 | 4,360 (**+5.6%** ±1.0) |
| Prefill, 256K | 3,963 | 4,164 (**+5.1%** ±0.8) |
| Prefill, 500K | 3,636 | 3,812 (**+4.8%** ±0.2) |
| Decode step time, geometric mean of 15 shapes | | +0.5% ±0.14 (repeat -0.4%) |
| Decode, 8 different prompts at once (tok/s) | 145.6 | 146.2 (+0.4% ±1.0) |
| Decode, 4 different prompts at once | 112.5 | 112.3 (-0.2% ±2.0) |
| Poisson arrivals at 0.42 req/s, output (tok/s) | 117.0 | 117.3 (+0.2% ±21) |

Decode is level: the first r6s arm's +0.5% is within the run's own repeat (the second
r6s arm is +0.1% against r6n). All 256 greedy responses are identical to r6n's, and the
teacher-forced distributions are identical (KL 0). These r6n figures are 0.3-4% above
r6n's own release run of 2026-10-15 (below, 8 streams the most); compare within a session.

r6n against r6h on three nodes (2026-10-16), measured with the `release-tp3` suite
in one session (r6n, r6h, r6n again; [results](../experiments/2026-10-15-r6n-tp3/README.md);
[suite run](../manifests/suites/2026-10-15-karmic-kraken-r6n-tp3/)), r6n on its own
build's cost curves. Each change is judged against the run's own repeat.

| Workload | TP3 r6h | TP3 r6n |
| --- | ---: | ---: |
| Decode step time, geometric mean of 15 shapes | | **-5.5%** ±0.14 |
| Decode, 8 different prompts at once (tok/s) | 137.1 | 139.7 (**+1.9%** ±1.3) |
| Decode, 4 different prompts at once | 109.3 | 112.1 (**+2.5%** ±2.2) |
| Poisson arrivals at 0.42 req/s, goodput (req/s) | 0.280 | 0.290 |
| Source-text prefill, 32K (tok/s) | 4,200 | 4,181 (-0.4% ±1.5) |
| Prefill, 128K | 4,128 | 4,114 (-0.3% ±1.2) |
| Prefill, 256K | 3,970 | 3,948 (-0.5% ±1.2) |
| Prefill, 500K | 3,638 | 3,625 (-0.4% ±0.3) |

Every decode shape is shorter (3.2-7.3%); prefill moved within the repeat's own -0.4
to -0.5%, so no size counts as slower. r6n's text differs from r6h's as r6k's did on
TP4 (top-token agreement 95.0%, KL 0.022 nats).

r6n against r6k on four nodes (2026-10-15), measured with the release suite
in one session (r6n, r6k, r6n again; [results](../experiments/2026-10-15-r6n/README.md#release-against-r6k-2026-10-15-1925-utc);
[suite run](../manifests/suites/2026-10-15-karmic-kraken-r6n-tp4/)). Each change
is shown with its 95% interval and judged against the run's own repeat; it
counts (**bold**) only beyond both.

| Workload | TP4 r6k | TP4 r6n |
| --- | ---: | ---: |
| Greedy responses identical, 256 prompts | | 256 of 256 (KL 0) |
| Decode step time, geometric mean of 20 shapes | | **-2.21%** ±0.15 |
| Decode step, 1 stream x 3 rows at 4K (ms) | 19.61 | 19.21 (**-2.1%** ±1.3) |
| Decode step, 1 stream x 3 rows at 256K | 19.75 | 18.88 (**-4.4%** ±1.2) |
| Decode step, 16 streams x 96 rows at 4K | 51.49 | 51.01 (**-0.9%** ±0.7) |
| Decode, 16 different prompts at once (tok/s) | 224.6 | 226.5 (+0.9% ±1.0) |
| Decode, 8 different prompts at once | 180.8 | 181.2 (+0.2% ±0.7) |
| Source-text prefill, 32K (tok/s) | 5,685 | 5,714 (+0.5% ±0.3) |
| Prefill, 128K | 5,642 | 5,669 (+0.5% ±1.2) |
| Prefill, 256K | 5,407 | 5,430 (+0.4% ±0.5) |
| Prefill, 512K | 4,942 | 4,959 (+0.4% ±0.4) |
| Prefill, 1M | 4,237 | 4,248 (+0.3% ±0.7) |

Every one of the 20 decode shapes is shorter. The 32K prefill gain is beyond its
interval but not beyond the repeat's -0.3%, so it counts as level. r6m, the
candidate before r6n, was 10-13% slower at every prefill size because its Engram
gather polled host memory from every block
([engram-rows-v2](../experiments/2026-10-15-engram-rows-v2/README.md)).

r6k against r6h on four nodes (2026-10-13), measured with the
[release comparison](lab.md#release-comparison) in one session, candidate
first ([results](../experiments/2026-10-13-r6k/README.md);
[report](../manifests/benchmarks/2026-10-13-karmic-kraken-r6k-tp4.json)). Each
change is shown with its combined 95% interval and counts (**bold**) only
beyond it.

| Workload | TP4 r6h | TP4 r6k |
| --- | ---: | ---: |
| Temperature 0, 15 streams + a long prompt (requests differing) | 0 of 16 | 0 of 16 |
| Quality gate | 5/5 | 5/5 |
| Decode, 1 stream, 16 prompts paired | | +0.6% ±2.4 |
| Decode, 4 different prompts at once (tok/s) | 133.9 | 134.5 (+0.4% ±0.9) |
| Decode, 8 different prompts at once | 177.2 | 178.3 (**+0.6%** ±0.3) |
| Decode, 16 different prompts at once | 225.8 | 223.9 (-0.8% ±0.9) |
| Source-text prefill, 32K (tok/s) | 5,835 | 5,825 (-0.2% ±1.4) |
| Prefill, 128K | 5,717 | 5,761 (+0.8% ±0.9) |
| Prefill, 256K | 5,490 | 5,519 (+0.5% ±0.6) |
| Prefill, 512K | 5,046 | 5,086 (**+0.8%** ±0.3) |
| Prefill, 1M | 4,378 | 4,400 (**+0.5%** ±0.1) |

r6k is never slower on these measures, and it is faster at eight prompts and on
the longest prefills, while computing WO at the reference's precision. Its
temperature-0 text differs from r6h's: scored on the same tokens, r6h agrees
with r6k on the most likely token at 95.1% of positions, with a KL divergence of
0.021 nats ([experiment](../experiments/2026-10-14-servebench/README.md)). The
bench's single-prompt decode points follow the text and moved 9-14% either way
with draft acceptance, so the table judges different prompts at once.

r6h against r6d (2026-10-12), measured with the
[release comparison](lab.md#release-comparison) in one session per
topology ([results](../experiments/2026-10-12-r6h/README.md); reports for
[TP4](../manifests/benchmarks/2026-10-12-karmic-kraken-r6h-tp4.json) and
[TP3](../manifests/benchmarks/2026-10-12-karmic-kraken-r6h.json)). TP4 ran the
candidate first after a 30-minute cool-down; the TP3 run is trimmed and ran r6d
first. Each change is shown with its combined 95% interval and counts (**bold**)
only beyond it.

| Workload | TP4 r6d | TP4 r6h | TP3 r6d | TP3 r6h |
| --- | ---: | ---: | ---: | ---: |
| Temperature 0, 15 streams + a long prompt (requests differing) | 13 of 16 | **0 of 16** | 1 | **0** |
| Quality gate | 5/5 | 5/5 | 5/5 | 5/5 |
| Decode, 1 stream, 16 prompts paired | | **+2.6%** ±2.5 | | -1.0% ±2.3 |
| Decode, 4 different prompts at once (tok/s) | 134.0 | 136.5 (+1.9% ±2.1) | 107.5 | 107.1 (-0.4% ±1.1) |
| Decode, 8 different prompts at once | 178.5 | 184.7 (**+3.5%** ±1.5) | 137.2 | 134.9 (**-1.6%** ±0.6) |
| Decode, 16 different prompts at once | 224.1 | 226.5 (+1.1% ±1.5) | | |
| Source-text prefill, 32K (tok/s) | 5,781 | 5,870 (**+1.5%** ±0.8) | 4,304 | 4,252 (-1.2% ±1.5) |
| Prefill, 128K | 5,669 | 5,726 (**+1.0%** ±1.0) | | |
| Prefill, 256K | 5,429 | 5,496 (**+1.3%** ±0.4) | 4,123 | 4,060 (**-1.5%** ±0.4) |
| Prefill, 512K (TP3 500K) | 4,998 | 5,074 (**+1.5%** ±0.2) | 3,797 | 3,753 (**-1.2%** ±1.0) |
| Prefill, 1M | 4,346 | 4,391 (**+1.0%** ±0.3) | | |

On four nodes r6h is at or above r6d on every metric: faster on prefill at
every size, at eight concurrent prompts and on the paired sweep, level
elsewhere (the mixed-traffic p95 stream gap read +3.9%, a single run without an
interval). On three nodes it trails at eight prompts and on long prefill, by
about 1.5%: each rank holds a third of the vocabulary and experts, so the
reference-precision arithmetic costs more, and the head fix applies only past
64 rows, which three nodes never reach. The bench's concurrent decode points send
one prompt to every stream and follow the text, which a precision change
alters (code at one stream read +21%, prose -5%); the release comparison
judges throughput on different prompts at once and on the paired sweep.

The r6 acceptance benchmark (2026-10-05, `bin/spark bench` from dgx1:
quality, then prose and code prompts with reasoning on, temperature 0, 256
output tokens, three samples per decode point, and real source text for
prefill, two repeats) against r5p's, one boot each at the recipe's limits.
Each value is the mean ± its 95% confidence interval; a family is ahead only
where the two intervals do not overlap (**bold**).

Three nodes (TP3, eight sequences at 512K):

| Workload | B12X r5p | TileLang r6 |
| --- | ---: | ---: |
| Quality gate (fixed LRU task, 5 repeats) | 5/5 | 5/5 |
| Prose, 1 stream (aggregate tok/s) | 50.4 ± 12.2% | 52.2 ± 0.7% |
| Prose, 2 streams | 75.9 ± 10.6% | 84.4 ± 0.6% |
| Prose, 4 streams | 118.8 ± 12.9% | 125.7 ± 0.5% |
| Prose, 8 streams | 173.3 ± 8.4% | **198.6 ± 0.4%** |
| Code, 1 stream | 60.1 ± 2.0% | **63.0 ± 0.5%** |
| Code, 2 streams | 90.8 ± 5.0% | 97.3 ± 8.5% |
| Code, 4 streams | 136.3 ± 6.3% | 159.2 ± 44.2% |
| Code, 8 streams | 192.3 ± 3.2% | **240.3 ± 1.0%** |
| One-stream prose / code step time | 41.92 / 46.44 ms | **38.21 / 42.43 ms** |
| Source-text prefill, 32K (tok/s) | 3,778 ± 14.9% | 4,306 ± 20.9% |
| Source-text prefill, 256K | 3,632 ± 2.2% | **4,034 ± 0.3%** |
| Source-text prefill, 500K | 3,432 ± 0.6% | **3,743 ± 0.8%** |
| Lowest MemAvailable, dgx1 | 5.93 GiB | 5.99 GiB |

Four nodes (TP4 1M recipe, 16 sequences at 1,048,576 tokens):

| Workload | B12X r5p | TileLang r6 |
| --- | ---: | ---: |
| Prose, 1 / 2 / 4 streams | 62.1 ± 3.8% / 92.7 ± 8.0% / 141.6 ± 7.9% | 62.4 ± 1.2% / 100.2 ± 2.7% / 147.1 ± 10.2% |
| Prose, 8 / 16 streams | 215.7 ± 7.5% / 297.3 ± 1.2% | 222.1 ± 0.3% / **318.4 ± 1.6%** |
| Code, 1 / 2 / 4 streams | 72.5 ± 7.5% / 111.3 ± 5.4% / 168.3 ± 9.6% | 76.0 ± 0.6% / 116.8 ± 18.9% / 174.9 ± 0.6% |
| Code, 8 / 16 streams | 245.7 ± 1.3% / 323.5 ± 1.5% | **264.3 ± 0.6% / 349.8 ± 2.5%** |
| One-stream prose / code step time | 34.00 / 37.72 ms | **31.79 / 34.75 ms** |
| Source-text prefill, 32K / 256K (tok/s) | 5,117 ± 1.7% / 4,873 ± 2.5% | **5,836 ± 4.2% / 5,471 ± 3.1%** |
| Source-text prefill, 500K / 1M | 4,645 ± 1.7% / 4,026 ± 0.9% | **5,072 ± 1.3% / 4,370 ± 0.6%** |
| KV capacity, 10.5 GiB per rank | 11 full 1M contexts at once (the startup line at 8,192 batched tokens: 8,580,566 tokens) | same |
| Lowest MemAvailable | 20.4 GiB (dgx1) | 19.35 GiB (dgx1) |

TileLang is never behind: it is ahead on single-stream step time, at eight
and sixteen streams and on every prefill point except TP3's 32K, and level
elsewhere. B12X's intervals are wider because its text changes from sample to
sample (DSpark acceptance moves with it); TileLang generates the same text
every sample, so its intervals are tight except where two- and four-stream
scheduling interleaves differently between samples. The intervals cover
sample-to-sample spread on one text; across 13 texts at TP3, DSpark
acceptance per text varied about ±5% while the two families' means were level
([TP3 benchmark](../experiments/2026-10-05-tp3-benchmark/README.md)). Step time,
which does not depend on acceptance, is the kernel comparison. The B12X TP4
500K point is from [its own run](../experiments/2026-10-05-tp4-500k/README.md).

The native benchmark reports
([TP3](../manifests/benchmarks/2026-10-05-karmic-kraken-r6.json),
[TP4](../manifests/benchmarks/2026-10-05-karmic-kraken-r6-tp4.json), and r5p's
[TP3](../manifests/benchmarks/2026-10-05-karmic-kraken-r5p.json) and
[TP4](../experiments/2026-10-05-r5p-promotion/runs/tp4-bench.json)) record
intervals, prompts, memory and thermal results. The
[64 KiB deployment record](../experiments/2026-10-02-memory-saver-capacity/README.md)
covers the TP3 profile's long-context admission (four simultaneous 485K-token
contexts) and near-limit retrieval (3/3 at 519,142 tokens), measured on r5o.

The tables below retain the historical **4 KiB r5l** reference (r5o added the
top-k tie rule and TMA stage-release fences). These were measured with
`bin/spark bench` from dgx1: prose and code
prompts, temperature 0, 256 output tokens. With reasoning on (the server
default) every measured token is reasoning text:

| Prompt | Streams | Aggregate tok/s | Per-stream decode tok/s | First token |
|---|---:|---:|---:|---:|
| prose | 1 | 50.3 | 52.2 | 0.20 s |
| prose | 2 | 76.1 | 41.4 | 0.33 s |
| prose | 4 | 111.8 | 31.0 | 0.42 s |
| prose | 8 | 162.0 | 22.4 | 0.53 s |
| code | 1 | 59.7 | 62.8 | 0.23 s |
| code | 2 | 90.4 | 49.6 | 0.35 s |
| code | 4 | 131.4 | 36.6 | 0.45 s |
| code | 8 | 187.4 | 26.6 | 0.55 s |

With reasoning off (the answer itself), aggregate tok/s at 1/2/4/8 streams:

| Prompt | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| prose | 59.1 | 84.8 | 124.9 | 174.5 |
| code | 80.6 | 118.2 | 165.1 | 233.0 |
| JSON | 77.0 | 109.7 | 165.2 | 234.3 |

| Other measurements | |
|---|---|
| Quality gate (fixed LRU task, 5 repeats) | 5/5 |
| Long-context retrieval (phrase at 10%, 50%, 90% depth) | 3/3 at 152,914 tokens |
| Single-stream decode step | about 42 ms on prose and 47 ms on code; accepted drafts per step 1.2 (prose), 1.9 (code), 3.2 (code answers) |
| Cold prefill, repeated filler | 2K 4.4k, 32K 4.9k, 64K 4.9k, 131K 4.7k tok/s |
| Cold prefill, real text (Python source) | 4K 3.9k, 16K 3.8k, 32K 3.8k, 64K 3.8k, 131K 3.8k, 200K 3.8k tok/s |
| Prefix-cache replay, 32K prompt | 6.47 s cold, 0.24 s warm |
| Four concurrent 64K contexts | all admitted without preemption, peak KV use 20%, 13.1 tok/s per stream |
| Four concurrent 180K contexts (r5k) | all admitted without preemption, peak KV use 36%, 11.9 tok/s per stream |
| KV capacity | all eight sequences at 262,144 tokens at once in 2.2 GiB per rank (vLLM's startup line: 1,348,708 tokens, 5.1 full 256K contexts) |
| Host memory headroom | dgx1 at least 6.38 GiB MemAvailable under load (3 GiB guard); startup passes the 5 GiB guard |

The quick default takes three or four samples per decode point, about ±2-12%
at 95% confidence; temperature-0 outputs differ between identical requests
(the indexer's radix top-k keeps an arbitrary subset of positions tied at its
threshold), which moves acceptance from sample to sample. Report:
[decode, prefill, prefix cache, and admission](../manifests/benchmarks/2026-09-29-karmic-kraken-r5l.json).
See [Benchmarking](../README.md#benchmarking) to reproduce them.
