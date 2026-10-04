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
