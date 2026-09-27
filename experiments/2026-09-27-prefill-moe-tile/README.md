# Prefill: M32 fused MoE tile and NCCL's Simple protocol

Base: the promoted configuration (`2026-09-27-karmic-kraken-r5e`).

## Question

A 4K-token real-text prefill chunk spends about 330 ms of its ~1.25 s in
routed MoE. At our shape (384 experts, a 768-wide slice per rank, about 64
routed rows per expert) B12X plans the M64 tile, which runs as three
launches: routing, then the materialized FC1 and FC2 phases. B12X's planner
notes that on GB10 the fused M32 persistent kernel beats the split M64/M128
tactics by 16-36% through the DeepSeek V4 TP2 prefill band. That rule is
keyed to V4's 256 experts and 1024-wide slice, so it never selects M32 for
V4.1 at TP3. Does M32 win here too?

The same chunk spends about 142 ms in 45 NCCL all-reduces of 42 MB each,
run as `RING_LL`: `NCCL_PROTO=^LL128` still lets NCCL's tuner pick the
low-latency protocol for large messages, at about 13.5 GB/s. Decode
collectives use the RoCE one-shot kernel, so NCCL carries only these. Does
the Simple protocol move them faster?

## Arms

| Arm | Change from the promoted configuration |
|---|---|
| `control` | none |
| `tile32` | `B12X_DYNAMIC_TILE_MN=32x128` (every dynamic MoE launch, decode included) |
| `simple` | `NCCL_PROTO=Simple` |

## Workload and gates

`sequence.sh`: control, tile32, simple, control again. Each runs the LRU gate,
single-stream decode (prose and code answers, `explain`), and cold prefill
of real text (`--prefill-text source`) at about 4K, 16K, 32K and 64K tokens,
two repeats. If prefill gains but decode loses, the follow-up is a B12X
patch that selects M32 only in the prefill band.

## Results

One session (2026-09-27, 17:10-17:35 UTC). Cold prefill of real text
(`--prefill-text source`, two repeats), tok/s by prompt length (actual
tokens in brackets), and single-stream decode step time:

| Arm | 4K (3854) | 16K (14768) | 32K (28937) | 64K (57122) | prose / code answer / explain step |
|---|---|---|---|---|---|
| control | 3574 | 3440 | 3427 | 3399 | 47.6 / 54.3 / 48.6 ms |
| tile32 | 3593 | 3469 | 3462 | 3444 | 48.1 / 53.9 / 48.6 ms |
| simple | 3570 | 3432 | 3427 | 3384 | 48.8 / 54.0 / 49.8 ms |
| control-b | 3554 | 3420 | 3420 | 3397 | 48.5 / 54.0 / 49.2 ms |

- `tile32` gains a consistent +0.8% to +1.4% on prefill (repeats agree
  within about 0.5%) and leaves decode unchanged. The override still runs
  the materialized two-phase kernels (B12X's `W4A8Materialized*`), so this
  is not the fused persistent M32 kernel that B12X's planner describes; that
  also needs `B12X_DYNAMIC_W4A8_MATERIALIZED=0`, not tested here.
- `simple` changes nothing: the 42 MB prefill all-reduces are bound by the
  RoCE link (a three-rank ring moves about 56 MB per rank in 3.1 ms, about
  145 Gb/s), not by NCCL's protocol choice.
- Neither is adopted. The prefill levers that remain are sequence-parallel
  prefill (the per-row hyper-connection work, about 150 ms per 4K chunk,
  repeated on every rank) and the fused MoE path.
- Quality gate LRU 5/5 in every arm; dgx1 minimum MemAvailable 6.2 GiB.
