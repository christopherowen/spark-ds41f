# One-Shot RDMA Exchange, Reduce-Scatter, Active-Row All-to-All, and Pipelined All-Gather

Eliminates multi-group PyNCCL send/recv launches and intermediate DRAM relay copies on sequence-parallel prefill and large tensor-parallel collectives by routing them directly through `sparknet.oneshot` (`SparknetOneShotAllReduce` / [christopherowen/dgx-spark-networking#3](https://github.com/christopherowen/dgx-spark-networking/pull/3)):

1. **Proxy Exchange Mode (`sparknet/oneshot/_roce_proxy.c`)**:
   - Encodes `ROCE_MODE_EXCHANGE` (`0x80000000u`) in the high bit of the per-slot byte count (`ctrl_words[4 + slot]`).
   - Sends peer `p`'s chunk from `send + p * nbytes` instead of broadcasting `send + 0`.
   - On `ring4`, posts the opposite rank's two stripe halves (`first` packs via `next`, `packs - first` packs via `prev`) **first** on hop 1 into the neighbours' unused self-receive slots (`recv[neighbour][slot]`), followed by the direct neighbour payloads on the same QPs; `relay_both_directions_mode` forwards each arrived half into `destination`'s `recv[origin][slot]` as soon as its flag lands.
2. **One-Shot TileLang Kernels & Runtime (`sparknet/oneshot/`)**:
   - `oneshot_reducescatter`: stages peer chunks (`p != rank`) into `send + p * nbytes`, rings the exchange doorbell, waits for peer flags, and reduces in fixed rank order (`0, 1, ..., world - 1`) in FP32 reading the local rank's chunk directly from `inp`.
   - `oneshot_exchange`: supports strided/contiguous peer chunk exchange (`exchange` / `exchange_into`) and active-row `all_to_all_rows`.
   - `oneshot_allgather`: supports `dst_stride_packs` for zero-copy strided destination writes when large `dim=0` or `dim=-1` gathers are pipelined in row slices (`sliced=True`).
3. **vLLM Collective & Sequence-Parallel Routing (`vllm.patch`)**:
   - `vllm/models/deepseek_v4_1/tilelang/collectives.py`: routes `exchange_into`, `reduce_scatter_rank_order`, and `all_reduce_rank_order` through `SparknetOneShotAllReduce` (`roce_comm`) when eligible.
   - `vllm/models/deepseek_v4_1/sp_prefill.py`: `_project_exchange` selects `b12x_ar_comm` when present, enabled, and eligible (`should_exchange(send, parts)`).
   - `vllm/distributed/device_communicators/cuda_communicator.py`: passes `roce_comm=self.b12x_ar_comm` to `all_reduce_rank_order` and `reduce_scatter_rank_order` under `_roce_policy`, and routes `all_to_all_rows` through `b12x_ar_comm.all_to_all_rows` when eligible.

## Verification

- `dgx-spark-networking` sanitizer simulator (`simulate.c` under `-fsanitize=address,undefined`) & CPU suite: **31,000 simulated collectives and 85 unit tests passed**.
- `dgx4` (`GB10` / `SM121`, `spark-ds41f:1030`):
  - Verified TileLang compilation and bit-exact GPU execution of single-slot and multi-slice `reduce_scatter`, `exchange` (`copy_self=False/True`), `all_to_all_rows` (exact and ragged row counts), and `all_gather(..., sliced=True)` (`dim=0` and `dim=-1`).
  - `pytest -v tests/distributed/test_roce_policy_no_fallback.py tests/kernels/test_deepseek_v41_tilelang_collectives.py tests/models/test_deepseek_v41_sp_project_reduce_scatter.py`: **191 passed in 15.94s**.
