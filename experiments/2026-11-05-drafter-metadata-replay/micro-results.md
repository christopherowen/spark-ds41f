# GPU component measurements

Two independent fresh-container boots passed on Spaceport (NVIDIA GB10), with
two CPUs, 8 GiB, fresh caches and the same immutable image plus three read-only
committed runtime overlays. The eager control and all alternatives used the same
candidate source and unchanged integer kernel. No released build was benchmarked.

Source: `a2b0ea500dc5022e23c4d6fc667254a806c984dc`; tree
`e11945731049bb810c69644d91b0740fb3ca3680`. Deployment at execution:
`b752c7b8a44c19d9f0ed7879bb18eea773599624`. Image:
`sha256:213ba4df68d06665add7a26260c2321d55854a75b8a51ae71c72d0b97eac814b`.
Six loaded runtime source hashes and the benchmark script hash match on both boots.

## Before and after

At 8 requests, five query rows per request and 8K prefix length, wall completion
time per metadata refresh plus integer consumer (microseconds):

| Placement | Boot 1 | Boot 2 |
| --- | ---: | ---: |
| Full eager host builder (control) | 218.154 | 225.644 |
| Direct kernel before query graph | 68.507 | 70.735 |
| Separate metadata graph | 7.899 | 7.572 |
| Metadata inside query graph (candidate) | 5.128 | 5.096 |
| Static consumer only (invalid dynamic implementation) | 3.583 | 3.549 |

| Comparison: control minus candidate | Boot 1 | Boot 2 |
| --- | ---: | ---: |
| Eager saving, us (95% CI) | 213.026 (212.471–213.585) | 220.547 (220.016–221.092) |
| Separate graph saving, us (95% CI) | 2.771 (2.249–3.344) | 2.476 (2.233–2.736) |

The isolated reduction is 97.65% and 97.74%. This is a component result, not
a token-throughput or model-latency improvement. Across all 16 request/context
cases the saving was 209.878–217.937 us on boot 1 and 217.597–227.498 us on
boot 2. The input matrix covered 1, 7 padded to 8, 8 and 16 requests, and
256, 8192, 131072 and 1048512 context tokens; one metadata group throughout.

The graph placement is the fastest valid placement tested. Its remaining cost
above the invalid static-consumer floor is about 1.55 us on both boots at 8/8K.
That bounds the isolated headroom for eliminating the remaining refresh work;
it does not prove a globally optimal implementation.

## Exposed saving with preceding GPU work

A calibrated CUDA wait graph was queued before the metadata operation. This
challenges the isolated result by allowing host work to overlap prior GPU work.
At 8 requests/8K, mean wall completion times and paired 95% intervals were:

| Boot | Preceding GPU work, us | Eager, us | Candidate, us | Saved, us (95% CI) |
| --- | ---: | ---: | ---: | ---: |
| 1 | 10.547 | 221.577 | 14.557 | 207.020 (205.975–208.256) |
| 1 | 80.465 | 225.131 | 84.702 | 140.430 (138.813–142.228) |
| 1 | 775.234 | 782.652 | 779.711 | 2.940 (2.578–3.308) |
| 2 | 10.577 | 231.287 | 17.393 | 213.894 (212.658–215.417) |
| 2 | 81.406 | 234.867 | 86.514 | 148.353 (146.272–151.133) |
| 2 | 775.536 | 783.171 | 780.508 | 2.663 (2.418–2.905) |

With about 775 us queued first, the exposed saving is only 2.940 and 2.663 us
(0.376% and 0.340% of this synthetic sequence). Most of the host saving can be
hidden. These waits are controlled fixtures, not measured model-head workloads.
The isolated 200+ us cannot be added directly to a serving-speed forecast.

## Correctness, repeatability and memory

- All three GPU regression geometries passed on each boot: ratio-1 paged,
  ratio-4 compressed and ratio-1 circular caches. Each log reports 3 passed,
  39 deselected, with no skipped GPU case.
- Each boot passed 192 exact comparisons of every readable integer buffer and
  consumer output across four valid arms, 16 cases and three mutations. Changed
  lengths, padding and slot reuse were covered; outputs were poisoned first.
- Required specializations compiled before the cache was frozen. Capture and
  replay passed with the cache frozen; no late compilation miss occurred.
- Per case: 24 paired trials, 128 iterations, balanced arm order; overlap:
  24 paired trials of 32 steps. Intervals use 10,000 paired-bootstrap draws.
  They describe within-boot trial uncertainty. The independent boot drift is
  shown separately; the two boots are not pooled into a narrow interval.
- Tensor high-water mark was 1,667,584 bytes on each boot. Container peaks were
  1,189,314,560 and 1,189,294,080 bytes (about 1.11 GiB), including compilation.
- Free unified-device memory changed from 86,546,759,680 to 86,281,465,856 bytes
  on boot 1 and 79,634,219,008 to 79,750,156,288 bytes on boot 2. These are whole
  process/host observations, not incremental graph-memory attribution.
- GPU temperature ended at 50 and 49 C. The runner cooled below 55 C before
  either boot. CPU affinity was 0,1; host times must not be extrapolated to a
  serving process with a different CPU allocation. Fan readings were unavailable.
- Both GPU jobs exited zero. Boot 1 yielded to an existing request; boot 2 used
  a new reservation, container and cache. All unsuccessful earlier CPU setup
  attempts remain archived. No serving-cluster operation was performed.

Native checkpoint floating-point arithmetic was not changed by this patch,
but these tests exercise integer preparation only. They do not establish model
quality, target sampling equality, rank agreement, full batch invariance,
streaming/cancellation behavior, serving memory, throughput, TTFT or TPOT.
Both PRs remain draft pending the [remaining qualification](qualification.md).

Every raw paired sample, log, exact container command, identity and telemetry
is retained under `runs/micro-20261106/`. `summary.json` retains all 16 case
summaries per boot; `archives.json` supplies file hashes. Gzip timestamps are
normalized; contents match the native outputs.
