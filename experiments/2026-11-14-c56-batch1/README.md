# Candidate-56 Batch 1 Overlay Screen

Base: `candidate-55` Build 9 (`spark-ds41f:candidate-55`, repo commit `d8b6037b1`).
One overlay package combining low-risk decode/host-overhead PRs:

- `#1078`: 42-row TileLang GEMM tile (`vllm/models/deepseek_v4_1/tilelang/gemm.py`, `linear.py`)
- `#1072`: `sched_yield` in `SpinCondition` (`vllm/distributed/device_communicators/shm_broadcast.py`)
- `#1074`: drop unused `req_ids` in `ModelRunnerOutput` / `SchedulerOutput` (`vllm/v1/core/sched/output.py`, `vllm/v1/worker/gpu/buffer_utils.py`)
- `#1065`: suffix-bounded stop-string slice (`vllm/v1/engine/detokenizer.py`, `vllm/v1/request.py`)

## Commands

```bash
python3 scripts/lab.py queue add experiments/2026-11-14-c56-batch1/screen-tp4.json
python3 scripts/lab.py queue add experiments/2026-11-14-c56-batch1/screen-tp4-switched.json
```
