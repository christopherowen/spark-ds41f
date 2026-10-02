# NVMe interrupt coalescing A/B (2026-10-02)

**Result: keep NVIDIA's default (coalescing on).** With coalescing off, a 4 KiB read
at queue depth 1 completes 3.6–4.4× sooner, but serving gains nothing. Decode step
times are unchanged, prefill is 0.7–1.6% slower, and the drive raises 4.5× more
interrupts. The serving setting was restored on all three nodes.

## Question

DGX OS's `nvidia-nvme-options` package installs
`nvidia-nvme-interrupt-coalescing.service`, which sets NVMe feature 0x08 to
`0x107` on every Samsung, Kioxia or Micron drive at each boot. With that value
the drive raises one interrupt per 8 completions or after 100 µs, whichever comes
first. NVIDIA's Grace tuning guide recommends these values to cut interrupt
overhead.

A public post asked whether disabling it helps Engram, whose rows come from disk
in our profile (`table_memory: "disk"`). B12X reads them with `O_DIRECT` through
io_uring in 4 KiB blocks, up to 64 in flight, and keeps nothing between steps.
Each step therefore issues a burst of small reads. With coalescing on, the last
reads of each burst wait for the timer.

## Method

Owner-approved host change, run in a lab window on the live production profile:
r5o, 64 KiB kernel, memory-saver, 524,288-token limit. There were no restarts,
because the feature is a runtime NVMe setting (`nvme set-feature -f 8`, not saved).

`run.sh` runs three arms in order: on (`0x107`), off (`0`), on again. Each arm
does the following on all three nodes:

- set and verify the feature;
- run `latency.py` against the model's largest safetensors blob (101.5 GB, read
  only), measuring 4 KiB `O_DIRECT` random reads at queue depth 1 for 5 s and
  400 bursts of 32 concurrent reads;
- snapshot NVMe interrupt and block-device read counters;
- from dgx1, run the lean bench: prose and code with reasoning on, one and eight
  streams, three samples, and source-text prefill at 4K and 32K with two repeats;
- run novel-text prefill (random pseudo-words, so Engram rows are read from
  disk) at 4K and 32K;
- snapshot the counters again.

`analyze.py` builds `results/summary.txt`. Its last column is off against the mean
of the two on arms.

## Results

Read latency, per-node medians across the three nodes:

| | Coalescing on (both arms) | Coalescing off |
|---|---|---|
| 4 KiB read, queue depth 1, p50 | 199–242 µs | 53.5–56.3 µs |
| 4 KiB read, queue depth 1, p99 | 391–691 µs | 71–159 µs |
| Burst of 32 threaded reads, p50 | 1.32–1.41 ms | 1.21–1.31 ms |

Serving, with off compared against the mean of the two on arms:

| Metric | On (1) | Off | On (3) | Off vs on |
|---|---:|---:|---:|---:|
| Code, 1 stream, step time | 44.95 ms | 45.42 ms | 45.57 ms | +0.4% |
| Prose, 1 stream, step time | 41.40 ms | 41.92 ms | 40.99 ms | +1.8% |
| Code, 8 streams | 194.2 tok/s | 191.1 tok/s | 188.6 tok/s | −0.2% |
| Prose, 8 streams | 172.2 tok/s | 171.2 tok/s | 170.1 tok/s | 0.0% |
| Source-text prefill, 32K | 3,839 tok/s | 3,796 tok/s | 3,839 tok/s | −1.1% |
| Source-text prefill, 4K | 3,382 tok/s | 3,336 tok/s | 3,333 tok/s | −0.7% |
| Novel-text prefill, 32K | 4,458 tok/s | 4,381 tok/s | 4,450 tok/s | −1.6% |
| Novel-text prefill, 4K | 4,453 tok/s | 4,395 tok/s | 4,450 tok/s | −1.3% |

Single-stream tok/s and first-token times moved in both directions with draft
acceptance and a three-sample screen; for example, code TTFT was +19% and prose
TTFT −10%. They do not show a coalescing effect. Full rows are in
`results/summary.txt`.

I/O during each arm's benchmarks (190 s): every node read 19,000–24,000 times per
second, 15–19 GB per arm, mostly Engram rows. Interrupts per read went from
0.11 with coalescing on to 0.50 with it off, which is 2,000–2,700 against
9,500–12,000 interrupts per second.

## Interpretation

Coalescing does delay a lone read. But Engram's reads come in deep io_uring bursts,
so a batch of 8 fills at once, and the asynchronous reader (`SPARK3_ENGRAM_ASYNC=1`)
overlaps the read with the forward pass. Single-stream decode step times did not
improve.

Prefill was slightly slower in all four prefill rows with coalescing off. The two
on arms agree within 0.0–0.2% at 32K, so the 1–1.6% loss is larger than their
spread. That fits the extra interrupt work. The suggestion fits stacks that make
lone or shallow-queue reads, for example through mmap faults. It does not fit ours.

Limits: one off arm between two on arms; three decode samples per point and two
prefill repeats. Decode differences are within noise. The prefill direction is
consistent but small.

## State after the run

`run.sh` restores `0x107` on exit. All three nodes read back
`Current value:0x00000107`. `nvidia-nvme-interrupt-coalescing.service` was never
stopped or disabled. `doctor --live` reports that the configuration and live
cluster match.
