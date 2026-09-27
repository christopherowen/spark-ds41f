# Prefill MoE tile: M32 fused kernel against the M64 split path

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

## Arms

| Arm | Change from the promoted configuration |
|---|---|
| `control` | none |
| `tile32` | `B12X_DYNAMIC_TILE_MN=32x128` (every dynamic MoE launch, decode included) |

## Workload and gates

`sequence.sh`: control, tile32, control again. Each runs the LRU gate,
single-stream decode (prose and code answers, `explain`), and cold prefill
of real text (`--prefill-text source`) at about 4K, 16K, 32K and 64K tokens,
two repeats. If prefill gains but decode loses, the follow-up is a B12X
patch that selects M32 only in the prefill band.

## Results

Pending.
