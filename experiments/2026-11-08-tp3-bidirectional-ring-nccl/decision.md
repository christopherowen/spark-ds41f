# Decision: TP3 6-Channel Bidirectional Ring NCCL Profile (`tp3-triangle` vs. `tp3-switched`)

## Status

- **`tp3-triangle` / `tp3-triangle-4k` (`tp3.json`)**: **Verified and retained** as promoted in `config/cluster.json` and `experiments/2026-10-26-spark-ds41f-candidate/1107/tp3.json`.
- **`tp3-p2p512k.json` (`NCCL_P2P_NET_CHUNKSIZE=524288` -> `1 MiB` effective P2P chunksize on GB10)**: **Rejected** (`+25.9%` slower on `wo_local_exchange_slice_a2a` at `T=6144` and `+3.1%` to `+14.5%` slower on `T >= 256` bundles than `NCCL_P2P_NET_CHUNKSIZE=262144`).
- **`scripts/topology.py` / `tests/test_switched_topology.py` contract alignment**: **Promoted** (`NCCL_P2P_NET_CHUNKSIZE` added to switched ring-control rejection and candidate stripping to match `scripts/recipe.py:SWITCHED_STRIP_RING_KEYS`).

## Summary of Evidence

1. **Switchless 3-node triangle (`dgx1..dgx3`, 2 dual-root ConnectX-7 cables/node)**:
   - Unidirectional 8-channel `1 MiB` rings (`Ring 00..07 : 2 -> 0 -> 1`) use only one direction of each peer cable (TX on cable A, RX on cable B), leaving 50% of the physical wire duplex dark (`12.56 GB/s` busbw at `T=6144`).
   - The 6-channel bidirectional `8 MiB` Ring profile (`Ring 00..02 : 2 -> 0 -> 1`, `Ring 03..05 : 1 -> 0 -> 2`, `NCCL_SWITCHLESS_BIDIRECTIONAL=1`, `NCCL_P2P_NET_CHUNKSIZE=262144` -> effective `512 KiB`) activates full duplex on both cables simultaneously (`21.74 GB/s` busbw) while using 6 instead of 8 GB10 SMs, reducing standalone SP collective time by **9.1% (`T=16`), 22.8% (`T=256`), 25.8% (`T=1024`), and 41.8% (`T=6144`, `1.72x`)** and overlapped GEMM + collective wall time by **11.5% (`T=16`), 16.6% (`T=256`), 20.5% (`T=1024`), and 36.4% (`T=6144`, `1.57x`)**.
2. **Single-cable switched 3-node subset (`sp5, sp6, sp8`, 1 dual-root ConnectX-7 cable/node)**:
   - Because each node connects to the switch via a single physical cable (`rocep1s0f0` + `roceP2p1s0f0`), a unidirectional ring already utilizes 100% of that cable's TX and RX full duplex (`18.74 GB/s` busbw at `T=6144`).
   - While 6 channels reduce small/medium collective latency at `T=256` (`-6.22%` standalone, `-8.65%` overlapped), sharing the single switch port across both peers at `T=6144` favors the switched profile (`14.61 ms` vs `16.26 ms` standalone, `7.40 ms` vs `7.51 ms` overlapped). Keeping `tp3-switched` on `SWITCHED_DIRECT_ENV` and stripping `NCCL_P2P_NET_CHUNKSIZE` in both `scripts/recipe.py` and `scripts/topology.py` preserves optimal behavior on both topologies.
