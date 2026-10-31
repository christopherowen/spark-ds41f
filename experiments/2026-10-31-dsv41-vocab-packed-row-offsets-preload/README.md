# DeepSeek-V4.1 TileLang Packed Vocab: Epilogue Register Binding and Predicated Exception Accumulation in `packed_vocab_projection` and `packed_vocab_greedy_partials`

## Summary

The 43-layer IKET forward-pass trace (`table_43l.md` in PR #777) flagged `28.5%` and `28.2%` `MEM_DEPENDENCY / LONG_SCOREBOARD` stalls in the exception epilogues of `vocab_packed_projection_kernel` (line 310) and `vocab_packed_greedy_partials_kernel` (line 422) in `vllm/models/deepseek_v4_1/tilelang/vocab.py`:

```python
first = bx * block_N
start = row_offsets[first]
end = row_offsets[T.min(first + block_N, N)]
for index in T.serial(start, end):
    row = exception_rows[index] - first
    entry = exception_entries[index]
    value = T.Cast(
        T.float32,
        T.reinterpret(T.Cast(T.uint16, entry & 0xFFFF), T.bfloat16),
    )
    column = entry >> 16
    for i, j in T.Parallel(block_M, block_N):
        token = T.min(by * block_M + i, M - 1)
        C_local[i, j] = T.if_then_else(
            j == row,
            C_local[i, j] + value * T.Cast(T.float32, A[token, column]),
            C_local[i, j],
        )
```

Because `start`, `end`, `row`, `entry`, `value`, and `column` were bound as Python AST expressions (`BufferLoad`) rather than `T.alloc_var` scalar registers:
1. `row_offsets[T.min(first + block_N, N)]` was emitted directly into the `for` loop header and re-evaluated on every iteration of `for index`, and `exception_rows[index]` / `exception_entries[index]` were re-expanded inside the `T.Parallel(block_M, block_N)` fragment loop;
2. `T.if_then_else(j == row, C_local[i, j] + value * T.Cast(T.float32, A[token, column]), C_local[i, j])` unconditionally evaluated the global load `A[token, column]` across all `block_N` (`32` or `64`) columns of the fragment even when `j != row` (31 out of 32 columns).

*(Note: hoisting `start` and `end` before `for s in T.Pipelined(...)` was also tested and rejected because holding 2 additional live registers across the 4-stage `T.Pipelined` loop increased register pressure in the main GEMM loop.)*

## Changes

- **`vllm/models/deepseek_v4_1/tilelang/vocab.py` (`packed_vocab_projection` and `packed_vocab_greedy_partials`):**
  - Bind `start = T.alloc_var(T.int32, init=row_offsets[first])` and `end = T.alloc_var(T.int32, init=row_offsets[T.min(first + block_N, N)])` in scalar registers immediately after `for s in T.Pipelined(...)` (leaving the `T.Pipelined` loop's register and shared-memory footprint 100% unchanged) and guard the loop with `if start < end:`.
  - Bind `row`, `entry`, `value`, and `column` with `T.alloc_var` once per exception `index`.
  - Replace `T.if_then_else(j == row, ...)` with `if j == row:` so `A[token, column]` is only loaded for the matching column `j == row`.

## Verification & Benchmark (`dgx4`, GB10 SM121, `spark-ds41f:19f2c20ed4d6-next-20`)

### Correctness & PDL Source Check
- `pytest -q -p tests.kernels.deepseek_v41_pdl_sources tests/kernels/quantization/test_deepseek_v41_tilelang_vocab.py`:
  - All `33` vocabulary head tests pass (`0` PDL violations).

### CUDA-Graph Replay Latency (`N = 32,320, K = 5,120, block_M = 16`, `10` calls/graph, `100` replays)

| Kernel | Exceptions (`E`) | Batch (`M`) | Baseline (`rc20`) | Patched | Delta |
|---|---:|---:|---:|---:|---:|
| `packed_vocab_projection` | `64` | `M = 1` | `1041.91 µs` (`p10=1038.81 µs`) | **`1036.51 µs`** (`p10=1033.09 µs`) | **`-5.41 µs`** |
| `packed_vocab_projection` | `64` | `M = 14` | `1054.60 µs` (`p10=1051.29 µs`) | **`1049.16 µs`** (`p10=1045.88 µs`) | **`-5.44 µs`** |
| `packed_vocab_projection` | `2048` | `M = 1` | `1041.49 µs` (`p10=1037.71 µs`) | **`1034.58 µs`** (`p10=1031.88 µs`) | **`-6.91 µs`** |
| `packed_vocab_greedy_partials` | `64` | `M = 1` | `1041.05 µs` (`p10=1038.29 µs`) | **`1035.53 µs`** (`p10=1032.49 µs`) | **`-5.51 µs`** |
| `packed_vocab_greedy_partials` | `64` | `M = 14` | `1046.77 µs` (`p10=1043.02 µs`) | **`1040.07 µs`** (`p10=1036.92 µs`) | **`-6.70 µs`** |
| `packed_vocab_greedy_partials` | `2048` | `M = 1` | `1040.14 µs` (`p10=1037.28 µs`) | **`1034.12 µs`** (`p10=1030.94 µs`) | **`-6.02 µs`** |
