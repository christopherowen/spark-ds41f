# TP3 triangle revalidation

Owner-authorized testing after restoring the dgx1–dgx2–dgx3 triangle.
The generated `tp3` control uses the existing r5o image and transport settings.
The installed image ID is `aad8a74089ff379f5bc7905e86f9c7e2c053396d0a4027039e869c505ca7621b`, already recorded as the base of the TP4 experiments. Its three source-tree labels match r5o, but its ID differs from the historical promoted artifact; the fresh run records this explicitly.

Network changes replace the former dgx1–dgx4 and dgx3–dgx4 addresses with the two 10.13 subnets. Only the affected fabric connections are activated; management networking is preserved. Each original netplan file is saved under `/root/spark3-tp3-revalidation-20261003/`. The ignored `config/nodes-tp3.local.json` records GID slots, subnets and peer interfaces.

Sequence: verify every directed fabric lane, publish/sync the generated control to the isolated qualification checkout, run model-free collective correctness including graph replay, then coordinated serving startup and quality/decode/source-prefill/prefix measurements. Record all failures and distinguish fresh TP3 measurements from the earlier TP4 run. The historical baseline is immutable.

## Result

The generated TP3 profile passes direct-collective correctness, startup/live doctor,
quality (5/5), the eight-point decode screen, and prefix-cache reuse. The long-prefill
screen **fails the thermal guard on dgx2**; it is not a complete serving-qualification
pass. No image or runtime promotion is made.

This tests the existing r5o TP3 profile. It does **not** qualify the unbuilt
explicit-policy successor image or the experimental batch-invariant arithmetic.

### Decode

Aggregate tokens/s, three samples per point after one discarded warm-up per point.
The reference is the immutable 2026-10-02 TP3 result. Every point's Welch interval
includes zero change; this small screen does not prove equivalence to a narrow bound.

| Prompt | Streams | Previous TP3 | Fresh TP3 | Previous TP4 |
| --- | ---: | ---: | ---: | ---: |
| Prose | 1 | 52.8 | 50.9 | 65.4 |
| Prose | 2 | 79.4 | 80.6 | 90.5 |
| Prose | 4 | 117.9 | 117.6 | 144.7 |
| Prose | 8 | 171.2 | 170.4 | 216.6 |
| Code | 1 | 64.8 | 60.0 | 78.5 |
| Code | 2 | 90.7 | 94.7 | 104.7 |
| Code | 4 | 135.6 | 139.7 | 170.6 |
| Code | 8 | 195.8 | 193.9 | 243.5 |

Derived single-stream step times are 41.108 ms prose and 45.822 ms code, versus the old
40.845/45.356 ms. These are acceptance-adjusted estimates, not fixed-work kernel
timings: adaptive verification work also changes. The new single-stream accepted
and verified draft counts are 1.172/3.259 (prose) and 1.874/3.857 (code).

The TP4 column comes from the earlier retained TP4 run, not a contemporaneous
second arm. TP-dependent dimensions, transport policy and verification-cost plans
also differ. This fresh TP3 run is consistent with the earlier conclusion that TP4
improves decode, but does not isolate GPU count or transport policy alone.

### Prefill failure and prefix result

The prefill run used the previous comparison's shuffled source-text workloads at
nominal 1K/32K/64K/256K, two repetitions each. The first completed calls logged
3,790 tok/s at nominal 64K and 3,736 tok/s at nominal 32K. During the following
nominal 256K request, dgx2 reached 83 C with 3 C of reported T.Limit headroom;
the guard aborted at 12:21:46 UTC. No thermal slowdown or power-cap time was
recorded. These two completed-call figures are single observations, not completed
point summaries; the native suite saves its samples only on successful completion.
The failed report and all console output are retained.

All nodes were below 55 C before measurement. dgx2's fan-controller journal shows
state 12 (maximum) from 12:20:23 UTC through the abort; a slow initial fan ramp
alone therefore does not explain it. GPU/board maxima were 82/91.1 C on dgx1,
83/91.5 C on dgx2, and 67/74.5 C on dgx3. This reproduces the thermal limitation
on a direct triangle without any ring4 relay. It rules out the TP4 relay as a
necessary condition, but does not identify the thermal cause. No host tuning,
guard relaxation or repeated long-prefill attempt was used to turn this into a pass.

A separately cooled prefix screen passed all three cold/six warm requests:
6.569 s cold, 0.270 s warm, 99% cache hit rate. Previous TP3: 6.553/0.268 s.
Separate cooling qualifies these bursts, not continuous combined-suite load.

Across all three serving screens, minimum MemAvailable was 6.68 GiB, swap growth
was zero, and no thermal slowdown was recorded. Startup/final serving logs and
kernel journals contain no NCCL warnings, CUDA errors, tracebacks, OOMs or Xids.

### Transport and workload checks

- All 12 directed jumbo-packet lane checks passed after readdressing.
- All three ranks passed 19 collective/layout cases with repeated CUDA graphs,
  plus BF16/FP32 timing screens for all-reduce, all-gather and reduce-scatter.
- All four RoCEnante HCAs carried approximately equal cumulative payload bytes;
  no timed RDMA retransmission or sequence-error deltas were recorded.
- Every measured decode request's input/output lengths match the reference.
- The Python source corpus hashes and 14 benchmark/helper function ASTs match
  the earlier TP4 client. The prefill/prefix wrapper replays skipped RNG draws.
- The three-rank client/checkout revision was `6f71c3c8ffe5c9dd7867a0116f7bcd4a301c55d1`;
  the client was clean, all images matched, and live doctor passed before and
  after serving measurements. TP3 verification costs were profiled at startup,
  preserving the old TP3 recipe; the historical TP4 run used its TP4 cost table.

## Reproduction and evidence

Use the node map recorded in the evidence archive only with the physical triangle.
The generated `control.json` targets the isolated qualification checkout. The
promoted cluster files and immutable baselines remain unchanged.

The recorded serving commands used `--min-samples 3 --max-samples 3 --seed 0`:

```sh
bin/spark3 --cluster-config experiments/2026-10-03-tp3-revalidation/control.json bench \
  --suites quality,decode --min-samples 3 --max-samples 3 --seed 0
python3 experiments/2026-10-03-tp3-tp4-comparison/prefill_after_cooling.py \
  --cluster-config experiments/2026-10-03-tp3-revalidation/control.json bench \
  --suites prefill --min-samples 3 --max-samples 3 --seed 0 \
  --prefill-text source --prefill-sizes 1024,32768,65536,262144 --prefill-repeats 2
# The prefix run uses the same wrapper/options with --suites prefix.
```

`hardware/runs.tar.gz` preserves the collective logs, native reports, aborted
prefill, workload hashes, host journals, startup/stop and live-doctor evidence.
`hardware/sha256.json` hashes the archive and every raw receipt. After extracting:

```sh
TMPDIR=/tmp python3 experiments/2026-10-03-tp3-revalidation/summarize.py \
  RAW_DIRECTORY experiments/2026-10-03-tp3-revalidation/results.json
```

The fleet entered stopped and returns stopped. The normal fan controllers and
the new triangle addressing remain active. The owner must recable before running
the four-node profile again. The shared hold is released after restoration checks.
