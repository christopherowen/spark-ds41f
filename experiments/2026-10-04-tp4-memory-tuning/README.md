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

## Combined

[combined.json](combined.json) runs batched8k + seqs16 + kv12g together,
against the base in one window (2026-10-04 07:54–08:10 UTC):

| | Base | Combined | Change |
| --- | ---: | ---: | ---: |
| prose, 1 / 8 streams (tok/s) | 62.8 / 217.1 | 63.0 / 213.2 | +0.3% / −1.8% |
| code, 1 / 8 streams | 82.3 / 246.4 | 77.4 / 247.2 | −6.0% / +0.4% (within noise) |
| prose / code, 16 streams | — (eight sequences) | 303.3 / 325.3 | |
| Prefill 32K / 64K / 256K | 4,678 / 4,734 / 4,526 | 5,137 / 5,114 / 4,904 | **+9.8% / +8.0% / +8.3%** |
| KV cache tokens per node | 2,845,543 | 9,756,345 | ×3.4 |
| Lowest MemAvailable over startup, decode and prefill, dgx1–dgx4 (GiB) | 30.6 / 31.7 / 31.8 / 31.0 | 20.4 / 21.6 / 21.6 / 21.0 | |
| Host memory used at the end, dgx1 (MiB) | 95,187 | 105,856 | +10.4 GiB |

Quality 5/5 in both. The settings compose: the combined arm keeps each one's
gain and its memory is their sum, leaving the tightest node 20.4 GiB above
zero, far outside the 8 GiB margin and the 3 GiB steady guard.

## Full context length

DS4.1 Flash supports 1,048,576 tokens (`max_position_embeddings`, YaRN factor
16 over 65,536); every TP4 recipe stopped at 524,288. The owner set the TP4
recipes to the full length and asked for the settings to be tuned for it.
[ctx1m.json](ctx1m.json) is the combined arm with `--max-model-len 1048576`
and its own DSpark cost directory. One window measures what the full length
asks of the rest of the configuration:

- memory over startup, decode and a ~970K-token prefill (each 8,192-token
  chunk attends to everything before it), from the per-second logs;
- decode at 1, 8 and 16 streams and prefill at 32K, 256K and ~970K, against
  the 512K combined arm's reports;
- [long_context.py](long_context.py): ~960K-token real-text prompts with a
  code word planted at 10% and 90% depth, answered greedily and streamed, for
  retrieval at depth, the full-length prefill time and the decode rate with
  the whole context resident.

Window 2026-10-04 08:21–08:45 UTC, against the combined arm's reports:

| | 1M | 512K (combined) |
| --- | ---: | ---: |
| prose, 1 / 8 / 16 streams (tok/s) | 64.1 / 212.3 / 297.1 | 63.0 / 213.2 / 303.3 |
| code, 1 / 8 / 16 streams | 76.9 / 246.6 / 324.1 | 77.4 / 247.2 / 325.3 |
| Prefill 32K / 256K | 5,075 / 4,875 | 5,137 / 4,903 |
| Prefill, 1,000,000 tokens | 4,013 tok/s (249 s) | — |
| Lowest MemAvailable over startup, decode and prefill, dgx1–dgx4 (GiB) | 19.05 / 20.72 / 20.75 / 20.08 | 20.4 / 21.6 / 21.6 / 21.0 |

Decode and prefill are level within noise, and quality passed 5/5. The
~1M-token prefill moves the floor only 0.6 GiB below decode. Retrieval at
depth passed: a code word planted at 10% and at 90% of ~900K-token prompts
(902,794 and 892,797 tokens) was answered correctly. The time to first token
was 224 s and 221 s, and decode ran at 131 tok/s with the whole context
resident.

## KV cache and prompt cache at 1M

With 12 GiB of KV per rank, vLLM reports a maximum concurrency of 9.35 for
1,048,576-token requests (9,806,361 tokens, 56,552 blocks), about 1.28 GiB per
full window. The owner set the rule for sizing: of the 16 sequences, keep room
for about 8 at full length, and tune the prompt cache to stay reasonable.

The prompt (prefix) cache shares the same pool. Each of the 40 layers, and
the drafter's, keeps a 128-token sliding-window cache.
`--prefix-cache-retention-interval` retains a checkpoint of those windows
every interval along a cached prompt, about 16 blocks per checkpoint, so a
later prompt that diverges mid-way can resume there. At 512, checkpoints keep
a quarter of every cached prompt's sliding-window blocks: about 32,800 blocks
for one cached 1M prompt, against about 5,000 for its compressed KV and
56,552 in the pool. Multi-turn continuations do not need periodic
checkpoints: the end of the previous turn is always retained.

Three arms, one per retention interval. Each is the 1M arm with 10.5 GiB of
KV per rank (8.2 full windows). They keep its DSpark cost directory, since
neither setting changes a shape:

| Arm | Retention interval | Checkpoint blocks per cached 1M prompt (estimate) | Most tokens recomputed after a mid-prompt divergence |
| --- | ---: | ---: | ---: |
| [kv8-r512](kv8-r512.json) | 512 | ~32,800 | 512 |
| [kv8-r8k](kv8-r8k.json) | 8,192 | ~2,050 | 8,192 (one prefill chunk) |
| [kv8-r32k](kv8-r32k.json) | 32,768 | ~510 | 32,768 |

Each arm runs lean decode (prose and code at 1 and 8 streams, three samples,
against the 1M arm) and [prefix_cache.py](prefix_cache.py). The probe sends
~200K-token real-text prompts one at a time and reads each request's cached
tokens from the server's counters:

- a cold prompt and its exact repeat;
- the prompt with a new tail from 75%, where the hit is the last retained
  checkpoint;
- a multi-turn continuation;
- ten other prompts, then the first again, which shows whether the cache
  still holds it.

The per-second memory logs give each arm's floors.

### Results

Window 2026-10-04 09:15–09:53 UTC. At 10.5 GiB vLLM reports 8,580,566 KV tokens
(49,483 blocks) and a maximum concurrency of 8.18 full windows. Every arm
decoded level with the 12 GiB 1M arm, with every point inside its interval,
and passed quality 5/5.

| | kv8-r512 | kv8-r8k | kv8-r32k |
| --- | --- | --- | --- |
| Exact repeat of a 196,133-token prompt | 195,840 cached, 0.6 s | 195,840 cached, 0.6 s | 195,840 cached, 0.6 s |
| New tail from ~151K tokens: resumed at | 151,040 (46,781 recomputed, 10.2 s) | 147,456 (50,365, 11.0 s) | 131,072 (66,749, 14.4 s) |
| Follow-up turn | 195,840 cached | 195,840 cached | 195,840 cached |
| The first prompt after ten more (1.9M tokens) | still cached | still cached | still cached |
| Peak KV use of the running 196K prompt | 5.43% | 5.43% | 5.43% |
| Lowest MemAvailable, dgx1–dgx4 (GiB) | 21.02 / 22.13 / 22.38 / 21.53 | 20.97 / 22.23 / 22.30 / 21.66 | 20.90 / 22.29 / 22.33 / 21.35 |
| Host memory used at the end, dgx1 (MiB) | 105,214 | 105,185 | 105,299 |

The interval does not change the running footprint, so it does not change
how many full-length sequences fit. A running request holds no checkpoints;
they become cached blocks when it ends. After a divergence the hit lands on
the last checkpoint before it, as designed. The cost of each interval is
cache space, and ten ~190K prompts did not fill a 10.5 GiB pool, so this
probe could not measure it. The ~1,000 blocks a running request holds
regardless of length are its sliding-window chunk state: 8,192 tokens plus
the window, in every sliding-window group.

### Cache capacity

Three measurement arms keep the 1M settings but shrink the KV pool to 3 GiB,
so the cache fills quickly:
[cap-r512](cap-r512.json), [cap-r8k](cap-r8k.json) and
[cap-r32k](cap-r32k.json). The space a cached token takes does not depend on
the pool's size. `prefix_cache.py --capacity 30 --tokens 100000` loads 30
distinct ~100K-token prompts, then recalls them newest first until one
misses. The number of hits is how many prompts the cache held at that
interval. A recall hit adds no blocks, so recalling never evicts an older
prompt.

Window 2026-10-04 09:54–10:34 UTC:

| Interval | Prompts held, of 30 | Cached tokens held | Blocks per 1,000 cached tokens | Checkpoints' share of cache space |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 8 | ~745K | 15.9 | ~75% |
| 8,192 | 27 | 2,526,188 | 4.7 | ~16% |
| 32,768 | 30 (did not fill) | ≥2,809,801 | ≤4.2 | ~5% (fit) |

A prompt being computed held about 2,300 of the 3 GiB pool's ~14,100 blocks;
the rest held the cache. Hits took 0.4 s and a miss recomputed in about 19 s.
The two saturated points fit a constant 3.9 blocks per 1,000 tokens of
compressed KV plus about 6,100 / interval of checkpoints, which predicts the
32K arm's fill exactly at the edge.

## TP4 recipe at 1M

Interval 8,192, one checkpoint per prefill chunk. It holds 3.4 times the
prompt cache of 512 in the same memory, and a mid-prompt divergence resumes
at most one chunk (≤1.6 s) back. 32,768 would add about 14% more cache and
quadruple the worst-case recompute. The KV pool stays at 10.5 GiB per rank,
8.18 full 1,048,576-token windows by vLLM's own accounting. An idle server
holds about 10M cached prompt tokens, about ten full 1M prompts, which give
way oldest first as long sequences run.

[candidate.json](candidate.json) is the recipe, launch disabled. It runs the
[streamed vocabulary weights](../2026-10-04-streamed-embeddings/README.md)
series plus [0039](vllm/0039-dummy-context-disjoint-blocks.patch) (image
`-r5o-roce-contract-tp4-1m-v1`, not built yet), with:

- `--max-model-len` 1,048,576;
- `--max-num-seqs` 16, with CUDA graphs up to 96 rows;
- `--max-num-batched-tokens` and `--long-prefill-token-threshold` 8,192;
- `--kv-cache-memory-bytes` 11,274,289,152 (10.5 GiB);
- `--prefix-cache-retention-interval` 8,192;
- the 1M arm's DSpark cost directory.

Against the streamed candidate nothing else changes.
[profiles.json](profiles.json) is the tuning catalog with `tp4` pinned to
it. TP3 is unchanged. The catalog's model layout report now describes the
packed target head, and the audit lists this image's vLLM and B12X trees.
Their changes from the audited TP4 trees are the packed head, its B12X
kernel, and the carve-out. None of them pads or partitions anything
differently.

## Fabricated profiling context (vLLM 0039)

DSpark's startup cost profile fabricates 8,192 tokens of context per dummy
request, and `set_dummy_context` gave every KV cache group the same block
ids, request after request from block 0. The 17 groups draw from one pool
and overlay the same memory; the allocator gives a block to one group and
request at a time, and block 0 is the null block. At 16 sequences the
drafter's sliding window read bytes another group had written in its own
format, and its first attention layer returned non-finite rows for one
dummy request. B12X's router routes them without complaint, so this recipe
started and pinned cost curves from that profile; TileKernels' gate refuses
them, which is how the [TileLang 1M](../2026-10-04-tilelang-1m/README.md)
experiment found it. Serving never shares a block.

[0039](vllm/0039-dummy-context-disjoint-blocks.patch) gives each request of
each group its own whole blocks after the null block. The recipe now has its
own [series](vllm/series), [lock](upstreams.lock.json) and
[source manifest](source.json): the streamed-embeddings series plus 0039. It
profiles into a fresh cost directory (`ring4-tp4-1m-20261005`), so the
pinned curves come from valid reads. The model layout audit lists the new
vLLM tree with the recipe's B12X tree. The image is not built or measured
yet.

## Next
- Autotune as a `spark3` command: tune in stages with bounded memory, write
  the selections for each TP size to the repository, prove them stable with a
  second run, and serve from them read-only with autotune off.
