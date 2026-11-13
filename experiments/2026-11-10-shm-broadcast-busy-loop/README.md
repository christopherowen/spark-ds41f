# Experiment: 2 ms `SpinCondition` Busy-Loop Window in `shm_broadcast`

- **Date**: 2026-11-10
- **Base commit**: `cc20ed18be1f7cbf5c6c53cdee945f7cc77d5fc5` (`spark-ds41f-1107`)
- **Causal variable**: Lower `SpinCondition` default `busy_loop_s` from `1.0 s` to `SHM_SPIN_BUSY_LOOP_S = 0.002` (`2 ms`) in `vllm/distributed/device_communicators/shm_broadcast.py` (`0762-perf-shm-broadcast-2ms-busy-loop.patch`).

## Rationale

`SpinCondition` in `vllm/distributed/device_communicators/shm_broadcast.py` defaulted `busy_loop_s` to `1.0 s`. Because every GPU decode step (`~21 ms`) and prefill chunk (`200–900 ms`) completes in less than `1.0 s`, `current_time <= self.last_read + self.busy_loop_s` remained continuously true for local `MessageQueue` readers (`EngineCore` waiting on `worker_response_mq` during GPU execution and `Worker_TP0` waiting on `rpc_broadcast_mq` between steps). As a result, the waiting reader never parked on `self.poller.poll()`, spinning in a Python `sched_yield()` loop at `~920K` iterations/second and burning 100% of a CPU core throughout every GPU decode and prefill step.

Lowering the default spin window to `SHM_SPIN_BUSY_LOOP_S = 0.002` (`2 ms`) preserves zero-poll spinning for sub-millisecond control RPC bursts and immediate queue handoffs while parking the waiting reader on the ZMQ notify socket (`local_notify_socket`) during the remaining `19 ms` of a `21 ms` decode step or `248 ms` of a `250 ms` prefill chunk.

## Verification & 2-Boot Benchmark Results (`sp8`)

Measured across two fresh `spark-ds41f:1107` container boots (`boot1` and `boot2`) on `sp8` using `bench_shm_busy_loop.py` (`receipts/bench-shm-busy-loop-boot1.jsonl`, `receipts/bench-shm-busy-loop-boot2.jsonl`) and `pytest tests/distributed/test_shm_broadcast.py` (`receipts/pytest-boot1.txt`, `receipts/pytest-boot2.txt`):

| Workload | Metric | Baseline (`busy_loop_s = 1.0`, Boot 1 / Boot 2) | Candidate (`busy_loop_s = 0.002`, Boot 1 / Boot 2) | Delta |
| :--- | :--- | :---: | :---: | :---: |
| **`decode_step_21ms`**<br>*(200 steps @ 21.0 ms)* | Reader CPU time / step (`ms`)<br>Reader CPU utilization (`%`)<br>`sched_yield()` calls / step<br>`poller.poll()` calls / step | `21.042 ms` / `21.056 ms`<br>`100.17%` / `100.23%`<br>`19,772.9` / `19,125.5`<br>`0.0` / `0.0` | **`2.166 ms` / `2.191 ms`**<br>**`10.31%` / `10.43%`**<br>**`1,823.3` / `1,804.4`**<br>**`1.0` / `1.0`** | **`-89.71%` / `-89.59%` CPU**<br>`-89.8 pt` CPU util<br>`-90.8%` spin calls |
| **`prefill_chunk_250ms`**<br>*(20 chunks @ 250.0 ms)* | Reader CPU time / step (`ms`)<br>Reader CPU utilization (`%`)<br>`sched_yield()` calls / step<br>`poller.poll()` calls / step | `250.107 ms` / `250.104 ms`<br>`100.02%` / `100.01%`<br>`231,491.0` / `235,639.0`<br>`0.0` / `0.0` | **`2.205 ms` / `2.191 ms`**<br>**`0.88%` / `0.88%`**<br>**`1,829.0` / `1,846.6`**<br>**`1.0` / `1.0`** | **`-99.12%` / `-99.12%` CPU**<br>`-99.1 pt` CPU util<br>`-99.2%` spin calls |
| **`sub_ms_rpc_burst_0p2ms`**<br>*(500 msgs @ 0.2 ms)* | `poller.poll()` calls / step<br>Wakeup latency `p50` (`µs`) | `0.0` / `0.0`<br>`8.02 µs` / `14.61 µs` | **`0.0` / `0.0`**<br>**`8.58 µs` / `14.54 µs`** | Fast spin path preserved (`0` polls) |

## Conclusion

Lowering `SpinCondition` `busy_loop_s` from `1.0 s` to `0.002 s` cuts waiting reader CPU consumption by **89.6–89.7%** during 21 ms decode steps and **99.1%** during 250 ms prefill chunks while keeping sub-millisecond RPC bursts on the fast spin path.
