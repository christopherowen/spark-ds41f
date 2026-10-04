# TP4 memory tuning screen

Base deployment commit: `3525568` (the [packed BF16 head](../2026-10-03-packed-bf16-head/README.md)
v3 candidate, packed in place, on the [native drafter heads](../2026-10-03-native-drafter-heads/README.md) change).

The TP4 candidate inherited TP3's memory-limited settings. On TP3, dgx1 served
with about 6 GiB above the 3 GiB steady memory guard, so 8,192 batched tokens,
the startup B12X autotune and a replicated drafter were ruled out for memory.
At TP4 every node keeps about 30 GiB available while serving. This screen
spends that memory one setting at a time, to find what it buys before
combining anything.

**Margin:** the guards stop serving below 5 GiB available at startup and 3 GiB
steady, and the bench stops itself 1 GiB above the steady guard. A setting
qualifies only if every node keeps at least 8 GiB available at its worst moment
(startup and benches), leaving room for long-context prefill and prefix-cache
growth in real traffic.

## Arms

Each arm is [base.json](base.json) (the materialized packed-v3 arm, image
`-r5o-roce-contract-packedhead-v3`) with one setting changed, plus its own
DSpark cost directory, because pinned cost curves are keyed by shapes only:

| Arm | Change | What it may buy |
| --- | --- | --- |
| [batched8k](batched8k.json) | `--max-num-batched-tokens` and `--long-prefill-token-threshold` 4,096 → 8,192 | prefill throughput |
| [seqs16](seqs16.json) | `--max-num-seqs` 8 → 16, graph capture sizes up to 96 | throughput above eight streams |
| [kv12g](kv12g.json) | KV cache 3.5 → 12 GiB per rank | prefix reuse and context capacity in real traffic |
| [autotune](autotune.json) | `B12X_AUTOTUNE=1` | B12X kernel selections measured for TP4's shapes |

Not screened: a replicated drafter (draft TP 1) is rejected by the topology
rule that draft TP equals the node count; testing it needs that rule and the
vocabulary-parallel draft paths revisited first. Engram tables (189 GiB) cannot
move to memory; a cache of hot rows would need new code.

## Procedure

One window: the base, then the arms in table order. Each arm: `cluster start`,
`bin/spark3 bench` decode (prose and code at 1 and 8 streams, and 16 for seqs16, three samples)
and prefill (32K, 64K, 256K, two repeats), against the base's reports;
`free -m`, `/proc/meminfo` and the largest processes' resident memory per
node at the end. A loop on every node logs MemAvailable once a
second for the whole window, so each arm's startup and bench minima are
recorded. A failed start or guard trip is recorded and the window moves on.

## Results

Window 2026-10-04 07:14–07:50 UTC (an earlier attempt stopped at its first arm
to fix the bench's 16-stream limit and the runner's logger start). Every arm
booted cleanly and passed quality 5/5; decode and prefill are against the base
arm, three decode samples and two prefill repeats per point.

| Arm | Decode, 1 / 8 streams | Decode, 16 streams | Prefill 32K / 64K / 256K | Lowest MemAvailable, dgx1–dgx4 (GiB) | Verdict |
| --- | --- | --- | --- | --- | --- |
| base | 61.7 / 215.3 prose, 80.5 / 250.2 code | — (eight sequences) | 4,638 / 4,699 / 4,520 tok/s | 30.2 / 30.6 / 31.0 / 30.8 | reference |
| batched8k | level within noise | — | **+10.6% / +8.8% / +8.7%** | 29.2 / 30.5 / 30.4 / 29.7 | keep |
| seqs16 | level within noise | **303.7 prose, 329.1 code** (+41% / +32% over eight streams) | level | 30.2 / 31.2 / 31.3 / 30.7 | keep |
| kv12g | level within noise | — | level | 22.2 / 23.4 / 23.4 / 22.8 | keep: 9,756,345 KV tokens (18.6 full windows) against 2,845,543 |
| autotune | not reached | — | — | 34 → 25.6 GiB on dgx1 within a minute of tuning | stopped: not feasible as one boot |

Minima are per arm over startup, decode and prefill, from the per-second
logs. End-of-arm host memory used: base 95,802 / 94,648 / 94,529 /
95,195 MiB; batched8k about 1 GiB more (GPU working memory for 8,192-token
forwards: the worker's resident CPU memory fell 0.6 GiB); seqs16 about the
same as the base; kv12g 8.2 GiB more, as reserved.

The base arm is also the packed head v3's serving check: it used 200–550 MiB
less host memory per node than any BF16-head boot and about 1 GiB less than
the v1 and v2 packers.

**Autotune:** with `B12X_AUTOTUNE=1` the preparation measured and compiled
candidates for every tunable plan at startup: after three minutes, 34 of 1,227
plans were ready (35,355 measurements, 4,905 compilations), and dgx1's
available memory fell from 34.2 to 25.6 GiB in one minute, heading for the
startup guard. B12X keeps compiled candidates resident and saves selections
only at the end of a stage, so the window stopped the arm; no selection file
was written. A one-boot autotune of everything does not fit; tuning has to be
staged with bounded memory, and the selections pinned (see below).

## Next

- [combined.json](combined.json): batched8k + seqs16 + kv12g together, run
  against the base in one window with decode at 1, 8 and 16 streams and the
  prefill lengths, to check that the settings compose and that every node
  keeps at least 8 GiB.
- Autotune as a `spark3` command: tune in stages with bounded memory, write
  the selections for each TP size to the repository, prove them stable with a
  second run, and serve from them read-only with autotune off.
