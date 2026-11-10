# NewRequestData Int32 PickleBuffer IPC and Direct Array Staging

## Base Commit & Causal Variable

- **Base commit**: `cc20ed18be1f7cbf5c6c53cdee945f7cc77d5fc5` (`spark-ds41f-1107`)
- **Causal variable**: Deduplicate `NewRequestData.prefill_token_ids` against `prompt_token_ids` for fresh admissions, pack token sequences of length `>= 256` into `Int32TokenIds` (backed by a contiguous 1D `int32` NumPy array, pre-packed in `Request.__init__` on the input-processing thread and serialized across `MessageQueue` via a single protocol-5 `PickleBuffer` with a prefix-length sentinel for `prompt_token_ids`), and add a direct `np.ndarray` staging fast path to `StagedWriteTensor.stage_write` / `apply_write`.

## Background

In stock `1107`, `Request.__init__` initializes `self._all_token_ids = self.prompt_token_ids.copy()`, and `Scheduler.schedule()` passes `req._all_token_ids` as `prefill_token_ids` alongside `req.prompt_token_ids` in `NewRequestData.from_request`. When `EngineCore` broadcasts `SchedulerOutput` over `MessageQueue`, Python `pickle` serializes two distinct `list[int]` objects element-by-element (~10 bytes/token combined) and allocates `2 * N` `PyLongObject` instances on every TP worker process, followed by $O(N)$ Python-list-to-`int32` conversions in `StagedWriteTensor.stage_write` / `apply_write` (`self.all_token_ids`) and `model_state.add_request` (`np.asarray(new_req_data.prefill_token_ids, dtype=np.int32)`).

## Exact Commands

```bash
bin/spark --cluster-config experiments/2026-11-10-new-request-data-zero-copy-tokens/tp4.json build prepare --only vllm --lock
```

Two-boot container verification (`boot1` and `boot2`) on `sp8` inside `spark-ds41f:1107`:

```bash
VLLM_SRC=/home/chris/projects/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-957fce43/src/vllm
docker run --rm --gpus all --entrypoint bash --shm-size=8g --ipc=host \
    -v "$VLLM_SRC/vllm/v1/core/sched/output.py:/opt/spark-ds41f/candidate/vllm/vllm/v1/core/sched/output.py:ro" \
    -v "$VLLM_SRC/vllm/v1/request.py:/opt/spark-ds41f/candidate/vllm/vllm/v1/request.py:ro" \
    -v "$VLLM_SRC/vllm/v1/worker/gpu/buffer_utils.py:/opt/spark-ds41f/candidate/vllm/vllm/v1/worker/gpu/buffer_utils.py:ro" \
    -v "$VLLM_SRC/vllm/utils/cpu_resource_utils.py:/opt/spark-ds41f/candidate/vllm/vllm/utils/cpu_resource_utils.py:ro" \
    -v "$VLLM_SRC/tests/v1/core/test_new_request_data_ipc.py:/tmp/test_new_request_data_ipc.py:ro" \
    -v "$PWD/experiments/2026-11-10-new-request-data-zero-copy-tokens/bench_new_request_ipc.py:/tmp/bench_new_request_ipc.py:ro" \
    spark-ds41f:1107 \
    -c "python3 -m pytest --noconftest -o addopts=\"\" -v /tmp/test_new_request_data_ipc.py && python3 /tmp/bench_new_request_ipc.py"
```

## Recorded Runs (`boot1` and `boot2`)

### Unit Tests (`tests/v1/core/test_new_request_data_ipc.py`)

- **Boot 1** (`receipts/pytest-boot1.txt`): `7 passed in 5.93s`
- **Boot 2** (`receipts/pytest-boot2.txt`): `7 passed in 6.18s`

### 4-Rank `MessageQueue` Broadcast + `StagedWriteTensor` Admission Benchmark (`receipts/bench-new-request-ipc-boot{1,2}.jsonl`)

| Prompt Tokens | Metric | Stock (Boot 1 / Boot 2) | Candidate (Boot 1 / Boot 2) | Speedup (Boot 1 / Boot 2) |
|---:|---|---:|---:|---:|
| `4,096` | Enqueue + Dequeue + `asarray` + `StagedWrite` | `0.4362 / 0.4360 ms` | `0.0515 / 0.0498 ms` | **`8.47x / 8.76x`** |
| `32,768` | Enqueue + Dequeue + `asarray` + `StagedWrite` | `3.4938 / 3.3531 ms` | `0.1889 / 0.1314 ms` | **`18.50x / 25.52x`** |
| `131,072` | Enqueue + Dequeue + `asarray` + `StagedWrite` | `13.2514 / 15.0606 ms` | `0.2279 / 0.2164 ms` | **`58.15x / 69.60x`** |
| `524,288` | Enqueue + Dequeue + `asarray` + `StagedWrite` | `66.4345 / 63.0623 ms` | `0.6535 / 0.6107 ms` | **`101.66x / 103.26x`** |
| `1,048,576` | Enqueue + Dequeue + `asarray` + `StagedWrite` | `132.6109 / 131.9335 ms` | `1.2922 / 2.0268 ms` | **`102.62x / 65.09x`** |

## Conclusion

Deduplicating `prefill_token_ids` against `prompt_token_ids`, packing long token lists into `Int32TokenIds` (`PickleBuffer`), and staging `ndarray` views directly in `StagedWriteTensor` cuts critical-path host admission overhead by **58x–70x at 131K tokens** (`13.3–15.1 ms -> 0.22–0.23 ms`) and **65x–103x at 524K–1M tokens** (`131.9–132.6 ms -> 1.29–2.03 ms` at 1M), with zero arithmetic or bit-identity change.
