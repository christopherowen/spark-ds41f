# Decision

Status: **Phase 2A (`vllm/0731` Sparse-MLA on `sp8`) and Phase 2B (2-boot 3-rank TP3 NCCL screen on `dgx1-3`) complete and qualified (`cand_c6_8m_bidir` + `vllm/0731`)**

## Summary of Phase 1 Preparation & Phase 2A Standalone GPU Results

1. **Audit 1 (TP3 NCCL Channels for SP Collectives)**:
   - Confirmed from `1107/tp3.json` and rank-0 NCCL init logs (`dgx1-probe.log`) that TP3 currently runs with `NCCL_MAX_NCHANNELS=8`, `NCCL_BUFFSIZE=1048576` (1 MiB, clamping P2P chunk size to 128 KiB), default `Tree`+`Ring`, and `NCCL_SWITCHLESS_BIDIRECTIONAL` unset (`0`), producing 8 unidirectional clockwise rings (`2 -> 0 -> 1`) that leave the counter-clockwise direct peer link idle during `ncclAllGather`.
   - Prepared `tp3-c4-ring.json` (4 Ring channels, 1 MiB buffer) and `candidate.json` (balanced 4-channel Ring + 4 MiB buffer + `NCCL_SWITCHLESS_BIDIRECTIONAL=2`, matching `scripts/topology.py` `oneshot-direct` and `tp4.json`) on `spark-ds41f:1107` (zero image rebuild required).
   - Verified 100% bit-identity: all PyNCCL SP collectives (`comm.all_gather` and `comm.send`/`comm.recv`) perform unreduced byte transfers followed by deterministic local `rank_order_sum`.

2. **Audit 2 & Phase 2A Standalone 2-Boot GPU Benchmark (`vllm/0731`, Sparse-MLA Head-Block Right-Sizing on `sp8`)**:
   - Confirmed from compiled SM121 cubin/source artifacts (`87.25` KiB dynamic shared memory at `block_H=32`, `69.25` KiB at `block_H=16`, `99` KiB SM121 limit) that `sparse_mla_fwd_v41` is strictly 1 CTA/SM (`48` CTAs/wave on GB10) and full-chunk prefill (`rows = 6,144` or `2,048`) already splits sequences token-wise across SMs (`100.0%` and `99.2%` wave occupancy).
   - **Ablation (`receipts/screen-mla-waves5-boot{1,2}.jsonl`)**: Tested raising `decode_max_rows` on 5-segment indexed layers from `28` (`DECODE_MAX_WAVES = 3`) to `48` (`5` waves) alongside `block_H = 16, threads = 128`. Across two fresh container boots on GB10 (`sp8`), `decode_max_rows = 48` sped up `rows = 29..48` when only 2 of 5 segments were active (`partial_256`: `1.32x..1.95x`), **but regressed `rows = 29..48` when all 5 segments were active at long context (`full_640`: `+28%..+105%` slower)** because single-wave `extend` (`29..48` CTAs <= 48 SMs) folds all 5 segments in registers (`0.049 ms` warm / `0.061..0.069 ms` cold) without writing `2.36 MiB` `Partial` to global memory or launching the 24-wave merge kernel (`0.056..0.105 ms` warm / `0.095..0.128 ms` cold). Per `AGENTS.md`, preserved the unfavorable ablation receipts and retained `DECODE_MAX_WAVES = 3` (`decode_max_rows = 28`).
   - **Refined `vllm/0731` (`609f622114a9ddac55ef5ab329294a71c5a061d2`, tree `74723c61264412617284b752d87ee1db4348c3de`, `receipts/screen-mla-boot{1,2}.jsonl`)**: Isolates right-sizing `block_H = 16, threads = 128` when `heads <= 16` (TP3 Rank 2 after `_trim_padded_groups()` and all TP4 ranks) plus `skip_inactive: bool = True` so multi-wave `decode` still skips inactive segments/dead rows at `block_H = 16` while single-wave `split` (`skip_inactive=False`) stays branch-free:
     - **GPU `pytest` contracts**: `20 passed` on `boot1` (`11.58s`) and `boot2` (`11.52s`), `0` PDL violations.
     - **Bit-identity**: `86/86` `int16` bit-identity checks across `boot1` and `boot2` passed (`int16_differ == 0` across all shapes and explicit `extend`/`decode`/`split` routes).
     - **16-head indexed prefill (`heads = 16, idx_width = 512, full_640`)**:
       - `rows = 1024`: `1.0759 ms -> 0.9602 ms` (`-10.76%`, `1.121x` wash), `1.0629 ms -> 0.9459 ms` (`-11.01%`, `1.124x` warm)
       - `rows = 2048` (SP prefill rank slice): `2.1076 ms -> 1.8676 ms` (`-11.39%`, `1.129x` wash), `2.0911 ms -> 1.8514 ms` (`-11.46%`, `1.129x` warm)
       - `rows = 4096`: `4.1761 ms -> 3.7005 ms` (`-11.39%`, `1.129x` wash), `4.1677 ms -> 3.6865 ms` (`-11.55%`, `1.131x` warm)
       - `rows = 6144` (full prefill chunk): `6.2228 ms -> 5.5237 ms` (`-11.23%`, `1.127x` wash), `6.2119 ms -> 5.5079 ms` (`-11.33%`, `1.128x` warm, saving `0.70 ms/layer` = `28.0 ms` across 40 indexed layers)
     - **16-head SWA-only prefill (`heads = 16, idx_width = 0, full_640`)**:
       - `rows = 1024..6144`: `-9.38%` to `-10.86%` (`1.104x..1.122x` speedup; at `rows = 6144`, `1.7881 ms -> 1.5939 ms`, saving `0.194 ms/layer` = `7.38 ms` across 38 SWA layers)
       - Combined across 40 indexed + 38 SWA layers: saves **`35.34 ms` per 6,144-token prefill chunk** (`~212 ms` at 32K/36K prefill, `~389 ms` at 64K prefill) on every 16-head rank (TP3 Rank 2 and all TP4 ranks).
     - **16-head multi-wave `decode` (`rows = 16..28`) and single-wave `extend` (`rows = 29..48`)**:
       - `decode` (`rows = 16, 24, 28`): `-1.42%` to `-13.12%` (`1.014x..1.151x` speedup)
       - `extend` (`rows = 29, 32, 40, 48`): `-4.08%` to `-12.53%` (`1.043x..1.143x` speedup, zero regressions)
     - **24-head control (`heads = 24, idx_width = 512`)**:
       - Unchanged kernel (`block_H = 32, threads = 256`), `0.00%` to `+0.06%` delta (`0.999x..1.001x` noise floor).

3. **Phase 2B (2-Boot 3-Rank TP3 NCCL Collective Benchmark on `dgx1-3`, `receipts/screen-nccl-tp3-boot{1,2}.jsonl`)**:
   - Screened 7 NCCL channel/buffer/direction policies across two fresh boots (`boot1` forward order, `boot2` reversed order, 20 timed iterations/point) on the 3-node switchless triangle (`dgx1`, `dgx2`, `dgx3`) across all 8 DS41F TP3 collective shapes at `T in {16, 2048, 4096, 6144}` (`56/56` bit-identity checks passed with `0` diffs):
     - **`cand_c4_1m` (`NCCL_MIN_NCHANNELS=4, NCCL_MAX_NCHANNELS=4, NCCL_BUFFSIZE=1048576`) regressed prefill collectives by `+4.3%..+5.1%` (`+8.4%..+11.2%` on 60 MiB `engram_sp_ag_bf16` and `moe_reduce_scatter_bf16`)** because cutting `coll_channels` from 8 to 4 and `p2p_channels` from 8 to 4 without bidirectional ring balancing throttles RoCE QP concurrency per peer.
     - **`cand_c6_8m_bidir` (`NCCL_ALGO=Ring, NCCL_MIN_NCHANNELS=6, NCCL_MAX_NCHANNELS=6, NCCL_BUFFSIZE=8388608, NCCL_P2P_NET_CHUNKSIZE=262144, NCCL_SWITCHLESS_BIDIRECTIONAL=1, NCCL_MIN_TRAFFIC_PER_CHANNEL=512, NCCL_THREAD_THRESHOLDS="-2 -2 -2 1 1 1"`) won across every token count (`T = 16, 2048, 4096, 6144`)**:
       - Keeps `p2p_channels = 8` (`p2p_chunk = 524,288` B) for `all_to_all_single` / `send` / `recv` while splitting the 6 collective channels evenly into **3 clockwise rings (`Ring 00..02 : 2 -> 0 -> 1`) and 3 counter-clockwise rings (`Ring 03..05 : 1 -> 0 -> 2`)**.
       - **`T = 16` aggregate (8 collectives)**: `0.5093 ms -> 0.4437 ms` (**`-12.9%`, `1.148x` speedup**; `engram_sp_ag_bf16`: `-24.3%`, `mhc_norm_all_reduce_bf16`: `-19.9%`, `moe_reduce_scatter_bf16`: `-18.8%`, `moe_sp_gather_mxfp8`: `-17.8%`).
       - **`T = 2048` aggregate**: `6.2628 ms -> 6.0482 ms` (**`-3.4%`, `1.035x` speedup**; `wo_local_exchange_slice_a2a`: `-5.2%`, `engram_sp_ag_bf16`: `-4.7%`, `rank_order_all_reduce_bf16`: `-4.1%`).
       - **`T = 4096` aggregate**: `12.3353 ms -> 12.1595 ms` (**`-1.4%`, `1.014x` speedup**; `mhc_norm_all_reduce_bf16`: `-6.8%`, `moe_sp_gather_mxfp8`: `-6.2%`, `attn_front_ag_u8`: `-4.7%`, `engram_sp_ag_bf16`: `-4.6%`, `moe_reduce_scatter_bf16`: `-4.5%`).
       - **`T = 6144` aggregate (full 6K prefill chunk)**: `18.2598 ms -> 17.5272 ms` (**`-4.0%`, `1.042x` speedup — faster on all 8 of 8 collectives at `T = 6144`**: `attn_front_ag_u8` `-7.1%`, `mhc_norm_all_reduce_bf16` `-5.6%`, `moe_sp_gather_mxfp8` `-5.0%`, `engram_sp_ag_bf16` `-4.3%`, `rank_order_all_reduce_bf16` `-3.7%`, `mlp_all_to_all_bf16` `-3.2%`, `wo_local_exchange_slice_a2a` `-1.9%`, `moe_reduce_scatter_bf16` `-1.5%`).
   - Updated `candidate.json` and `tp3-combined.json` to `cand_c6_8m_bidir`.
