# Decision

Status: **not evaluated (Phase 1 CPU audit, patch, configs, and unit tests complete; awaiting Phase 2 GPU window)**

## Summary of Phase 1 Preparation

1. **Audit 1 (TP3 NCCL Channels for SP Collectives)**:
   - Confirmed from `1107/tp3.json` and rank-0 NCCL init logs (`dgx1-probe.log`) that TP3 currently runs with `NCCL_MAX_NCHANNELS=8`, `NCCL_BUFFSIZE=1048576` (1 MiB, clamping P2P chunk size to 128 KiB), default `Tree`+`Ring`, and `NCCL_SWITCHLESS_BIDIRECTIONAL` unset (`0`), producing 8 unidirectional clockwise rings (`2 -> 0 -> 1`) that leave the counter-clockwise direct peer link idle during `ncclAllGather`.
   - Prepared `tp3-c4-ring.json` (4 Ring channels, 1 MiB buffer) and `candidate.json` (balanced 4-channel Ring + 4 MiB buffer + `NCCL_SWITCHLESS_BIDIRECTIONAL=2`, matching `scripts/topology.py` `oneshot-direct` and `tp4.json`) on `spark-ds41f:1107` (zero image rebuild required).
   - Verified 100% bit-identity: all PyNCCL SP collectives (`comm.all_gather` and `comm.send`/`comm.recv`) perform unreduced byte transfers followed by deterministic local `rank_order_sum`.

2. **Audit 2 (Sparse-MLA Prefill & Multi-Segment Decode Occupancy)**:
   - Confirmed from compiled SM121 cubin/source artifacts (`87.25` KiB dynamic shared memory at `block_H=32`, `69.25` KiB at `block_H=16`, `99` KiB SM121 limit) that `sparse_mla_fwd_v41` is strictly 1 CTA/SM (48 CTAs/wave on GB10) and full-chunk prefill (`rows = 6,144` or `2,048`) already splits sequences token-wise across SMs (`100.0%` and `99.2%` wave occupancy).
   - Identified and fixed two 100% bit-identical occupancy/efficiency gaps in `vllm/0731-perf-dsv41-right-size-sparse-mla-head-block-and-deco.patch` (`tp3-mla-occupancy.json`):
     1. Right-sized `block_H` to `16` and `threads` to `128` when `heads <= 16` (TP3 Rank 2 after `_trim_padded_groups()` and TP4 all ranks), eliminating 50% wasted warps and tensor-core MMAs on zero-padded dummy heads `16..31` while preserving `seg_active`/`row_active` skipping on `decode` (`skip_inactive=True`) and branch-free single-wave `split` (`skip_inactive=False`).
     2. Raised multi-segment `decode_max_rows` on 5-segment indexed layers from `28` to `48` (`sms // head_blocks`) so segment-split `decode` (`5 * rows` CTAs) is used whenever unsplit `extend` (`1 * rows` CTAs) cannot fill 1 wave of 48 SMs, with zero increase in target or drafter workspace reservations.
