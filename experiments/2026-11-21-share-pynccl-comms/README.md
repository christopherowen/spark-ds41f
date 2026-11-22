# Share PyNCCL Communicator Between TP and Idle EP Groups (`VLLM_SHARE_PYNCCL_COMMS=1`)

- **Date**: 2026-11-21
- **Base commit**: `0eace90e2` (`spark-ds41f:candidate-63-r5`, vLLM tree `5f16a050bb283d3d964ab831de4cdf6fca74dea7`)
- **One intended variable**: Set `VLLM_SHARE_PYNCCL_COMMS=1` on `tp3-triangle` (`tp3-dev.json`).

## Hypothesis and Prediction

Code audit of `vllm/distributed/parallel_state.py` (`initialize_model_parallel`, lines 2195 and 2335–2365) and `vllm/distributed/device_communicators/cuda_communicator.py` (`_acquire_pynccl`, lines 41–66) shows that on every multi-node MoE deployment (`is_moe = True`, `enable_expert_parallel = False`), vLLM creates **two** `PyNcclCommunicator` instances at load over the exact same rank set `[0, 1, 2]`:

1. `_TP` (`group_name="tp"`, `world_size=3`)
2. `_EP` (`group_name="ep"`, `world_size=3`, completely idle because `enable_expert_parallel = False`)

In `experiments/2026-11-17-dspark-turn-learning` (`TODO.md` Decode #17), creating an extra idle `PyNcclCommunicator` at load slowed the decode grid by **`+1.13%`** across 4 boots (`sd = 0.15%`).

Setting `VLLM_SHARE_PYNCCL_COMMS=1` (already implemented in `cuda_communicator.py`) makes `_EP` reuse `_TP`'s `PyNcclCommunicator` (`Reusing the PyNCCL communicator of ranks (0, 1, 2)`), eliminating the second NCCL communicator at load.

**Prediction**: Decode grid `floor25` geometric mean improves by **`~0.5–1.1%`**, `c1`/`c8` throughput improves commensurately, startup `torch_reserved`/memory drops by NCCL's second-communicator channel buffers, and outputs remain **100% bit-exact** (`128/128` greedy, `KL = 0`).

## Live Serving Runs (`tp3-triangle`, `dgx1..dgx3`)

Measured neutral on Boot 1 in Round 3 (`-0.043%` decode grid, `c1`/`c8` within noise) and dropped — see [2026-11-21-mhc-async-sinkhorn/README.md](../2026-11-21-mhc-async-sinkhorn/README.md) and [TODO.md](../../TODO.md).
