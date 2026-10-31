# Lock Pinned Host Buffers (`CpuGpuBuffer`, `MappedHostAllocation`, `EngramDiskRows`) with `mlock(2)`

## Hypothesis & Objective

On unified-memory Grace-Blackwell (`GB10`) nodes, heavy `O_DIRECT` / page-cache activity from disk-backed Engram tables and model shards can contend with host pages backing pinned CPU-GPU transfer buffers (`CpuGpuBuffer`) and mapped host allocations (`MappedHostAllocation`, `EngramDiskRows.ids`):

1. **`mlock(2)` / `munlock(2)` on pinned and mapped host buffers**:
   Add `mlock_memory(addr, nbytes)` and `munlock_memory(addr, nbytes)` in `vllm/v1/utils.py` (raising `RLIMIT_MEMLOCK` to the maximum permitted limit on first use, while preserving tight `< 64 MiB` `RLIMIT_MEMLOCK` budgets for `io_uring` buffer registration). Lock:
   - `CpuGpuBuffer.cpu` when `pin_memory=True`
   - `MappedHostAllocation` (`cudaHostAllocMapped`) buffers in `vllm/models/deepseek_v4_1/engram_rows/rows.py` (with `munlock_memory` in `close()`)
   - `EngramDiskRows.ids` (`pin_memory=True`) in `vllm/models/deepseek_v4_1/engram_rows/rows.py` (with `munlock_memory` in `close()`).

2. **Remove redundant per-step `.pin_memory()` call in `CpuGpuBuffer.copy_to_gpu`**:
   `CpuGpuBuffer.__init__` already allocates `self.cpu` with `pin_memory=pin_memory`. Previously, `CpuGpuBuffer.copy_to_gpu` called `cpu.pin_memory() if PIN_MEMORY else cpu` on every step, dispatching through PyTorch's `is_pinned` / `pin_memory` check on every sliced tensor view (and allocating a temporary pinned copy if `pin_memory=False` was passed). Replacing this with `gpu.copy_(cpu, non_blocking=True)` avoids redundant per-transfer overhead.

## Method & Variants

Patch: [`mlock-pinned-host-buffers.patch`](./mlock-pinned-host-buffers.patch) against `.work/upstreams/vllm` (`d5df88d83`).

Modified files in vLLM:
- `vllm/v1/utils.py`
- `vllm/models/deepseek_v4_1/engram_rows/rows.py`
- `tests/kernels/test_deepseek_v41_engram_rows.py`

## Verification Run

Verified on `dgx4` inside `spark-ds41f:1030`:

```bash
pytest tests/kernels/test_deepseek_v41_engram_rows.py \
  -k "mlock or gather_matches or every_step or timeline_is_off" -v
```

Result: `7 passed`.
