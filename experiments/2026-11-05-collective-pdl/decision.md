# Decision: Drop Collective PDL Launch-Overlap Candidate (`2026-11-05-collective-pdl`)

## Verdict

- **Candidate `1107` status**: **Drop** (`completed-dropped`).
- **Source gates on `dgx1-dgx3` (`tp3-triangle`)**: **PASSED** for all three variants (`entry` `9c8d89d1c1c95547c1d228600d6d5949224dbc45`, `staged` `f69c7b16c08464a8f5a3540353a96ae7d0b7560c`, and `arrived` `fb2be9f8d23a797cac9cbaae9f9ea9e538246850`) — `106/106` native GPU collective unit tests per rank (including all 5 new `test_pdl_*` cases and both `test_fail_stop_on_timeout_eager` and `test_fail_stop_on_timeout_graph_replay`) plus the `oneshot-direct` TP3 fabric probe on all 3 ranks with zero RDMA or physical port error/drop counter changes.
- **Two-process reversed synthetic launch-chain comparison (`runs/native-benchmarks-20261106/launch-chain-comparison.json`)**: **NO WINNER** (`staged` vs `entry`: mean delta `-0.031 us` / `-0.300%`, `13/63` cases faster in both rounds vs `10/63` slower in both rounds, `0/63` cases with 95% CI below zero; `arrived` vs `entry`: mean delta `-0.343 us` / `+0.039%`, `18/63` faster in both rounds vs `16/63` slower in both rounds).
- **Tooling bug fix retained**: Commit `9c8d89d` in `dgx-spark-networking` PR #5 fixes a real multi-node bug in `sparknet/collectives/failstop.py` where non-root ranks invoked `TCPStore("127.0.0.1", ...)` instead of resolving the coordinator address from `MASTER_ADDR` / `--master-addr`. That standalone qualification-tooling fix is valid independent of the PDL overlap lever.

## Hardware Qualification & Synthetic Screen Summary

1. **Window `dgx1-1793950542` (`runs/native-retry-no-ifname-20261106`)**:
   - Confirmed the `9c8d89d` `TCPStore` rendezvous fix succeeded across `dgx1-dgx3`.
   - Exposed that `/home/swank/projects/spark-ds41f/config/nodes.json` on `dgx1` did not set `"management_interface": "enP7s7"`, leaving `GLOO_SOCKET_IFNAME` unset while `/etc/hosts` mapped `dgx1` to `127.0.0.1`, causing `ProcessGroupGloo` `connectFullMesh` to fail with `Connection refused` on `127.0.0.1`.
2. **Window `dgx1-1793951262` (`runs/native-gates-20261106`)**:
   - Supplied `--map /home/swank/spark-lab/collective-pdl/nodes-tp3.json` with `"management_interface": "enP7s7"` on `dgx1-dgx3` so `sparknet.topology.render.node_environment` exported `GLOO_SOCKET_IFNAME=enP7s7`, `NCCL_SOCKET_IFNAME==enP7s7`, and `TP_SOCKET_IFNAME=enP7s7`.
   - All three variants passed `106/106` native GPU collective tests per rank (`pytest -q -k "not system"`: `95 passed`, `pytest -q -m system -k "not slow"`: `11 passed`) and passed the TP3 `oneshot-direct` fabric probe on all 3 ranks with zero RDMA or physical port counter changes:
     - `entry-gates` (`9c8d89d`): `49.92s`
     - `staged-gates` (`f69c7b1`): `49.04s`
     - `arrived-gates` (`fb2be9f`): `50.80s`
3. **Window `dgx1-1793951855` (`runs/native-benchmarks-20261106`)**:
   - After adding `--` before `/collective/benchmark_launch_chain.py` in `run-source-gates.py` (`bdf9d2d9722dbccf053f804b4b484b80c30e1bca`) so `torch.distributed.run` did not parse `--run` as `--run-path`, executed the two-round reversed process order synthetic launch-chain screen (`entry -> staged -> arrived` in round 1; `arrived -> staged -> entry` in round 2):
     - `entry-bench-1`: `33.02s`, `staged-bench-1`: `33.15s`, `arrived-bench-1`: `32.83s`
     - `arrived-bench-2`: `34.17s`, `staged-bench-2`: `31.79s`, `entry-bench-2`: `33.13s`
   - Across all 63 cases (7 row counts `1, 6, 9, 16, 24, 48, 96` x 3 ops `all_reduce, all_gather, mixed` x 3 background copy loads `0, 4, 32 MiB`):
     - Every variant and round achieved `exact=True`, `long_delay_distinguished=True`, and `nr_throttled=0`.
     - `staged` vs `entry`: `13/63` faster in both rounds, `10/63` slower in both rounds, `40/63` inconsistent across rounds, `0/63` with `ci95_high_us < 0`, overall mean delta `-0.031 us` (`-0.300%`).
     - `arrived` vs `entry`: `18/63` faster in both rounds, `16/63` slower in both rounds, `29/63` inconsistent across rounds, `4/63` with `ci95_high_us < 0` vs `3/63` with `ci95_low_us > 0`, overall mean delta `-0.343 us` (`+0.039%`).

## Why CUDA Programmatic Dependent Launch (`cudaTriggerProgrammaticLaunchCompletion` / `cudaGridDependencySynchronize`) Does Not Improve TP3 Switchless Collectives Here

- On the single-CTA persistent `oneshot-direct` kernels (`all_reduce_bf16_3rank_oneshot_direct_kernel` and `all_gather_bf16_3rank_oneshot_direct_kernel`), the dominant latency between producer completion and consumer readiness is the cross-node system-scope zero-copy RDMA/PCIe polling phase (`ld_volatile_global_u64` / `ld_acquire_sys_u32`), which takes `~10-115 us` depending on payload size.
- Moving `cudaGridDependencySynchronize()` after local staging (`staged`) or after peer arrival (`arrived`) and firing `cudaTriggerProgrammaticLaunchCompletion()` before final consumer writes does not overlap the network wait with useful downstream work because:
  1. The producer kernel has already retired its final output writes before the collective reads its input buffer (`staged`), or the collective still has to read the producer output immediately upon launch before publishing its sequence/stage to peers (`arrived`).
  2. The downstream consumer kernel immediately reads the collective output buffer, so it must still wait on `cudaGridDependencySynchronize()` until the collective writes the reduced/gathered output.
  3. Any sub-microsecond CTA preamble overlap is smaller than run-to-run variance across the switchless ConnectX-7 links (`~0.5-2.0 us`).

## Restoration Receipt

- `C36` (`experiments/2026-10-26-spark-ds41f-candidate/candidate-36/tp3-dev.json`, commit `2c4b23afcd834ba2adf6a15ca434e59fa4f49f74`, image `sha256:b0522146640a7d852c017ec8c99423224492b7c2c4951eb3a64005ab4555ab11`) was restored across `dgx1-dgx3` (`ready +94.7s`), verified with `bin/spark doctor --live` (`OK`), and `dgx1:~/spark-hold.json` was removed at `2026-11-06T08:03:58Z`.
