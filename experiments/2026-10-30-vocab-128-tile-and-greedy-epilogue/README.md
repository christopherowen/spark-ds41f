# 128-Row Vocabulary LM-Head Tile & Fused Greedy Epilogue Argmax

## Summary

1. **128-Row Token Tile (`ROW_TILES = (16, 32, 64, 128)`):**
   - On TP4 and TP3 speculative verification steps with `65–128` verification rows (`c12..c16` with `K=5` speculative drafts), the `64`-row token tile previously launched `2` row tiles (`by in {0, 1}`), reading and unpacking the 248 MB packed BF16 vocabulary weight shard twice (`32,320 × 7,680` bytes on TP4).
   - Adding a `128`-row token tile with `block_N=32, threads=256, step_groups=1, num_stages=4, act_stages=2` stages activations in `2` pipeline buffers (`64 KB`) while keeping `4` weight stages (`24 KB` packed + `8 KB` decoded) in `98,304` bytes (`96 KiB <= 101,376` bytes) of dynamic shared memory, reading and decoding the weight shard once for `65–128` rows.
2. **Fused Greedy Epilogue Argmax (`packed_vocab_greedy_partials` + `finish_vocab_greedy_pairs`):**
   - For greedy verification (`_verify_greedily` when `sink.rejection` is off), the runner previously materialized the full `[M, 32320]` FP32 local logits tensor to DRAM (up to `12.4 MB` at `96` rows) and read it back in `greedy_shard_pair`.
   - `packed_vocab_greedy_partials` reduces each CTA's `[block_M, block_N]` tile in shared memory (`C_shared`, aliased into the pipeline buffers after the K loop) via warp shuffles and writes one `(max, shard_index)` partial per tile, which `finish_vocab_greedy_pairs` reduces into the rank's 16-byte `(max, global_index)` pair row (`vocab_parallel_pairs_from_head`), bit-for-bit identical (`torch.equal`) to `project_head` + `greedy_shard_pair`.

## Verification (`dgx4`, `spark-ds41f:1030`)

- `python3 -m pytest -q -p no:cacheprovider --noconftest -p tests.kernels.deepseek_v41_pdl_sources tests/kernels/quantization/test_deepseek_v41_tilelang_vocab.py tests/v1/worker/gpu/test_greedy_verification.py tests/v1/worker/test_prepare_kernels.py`: **56 passed**, `0` PDL violations (`74 kernels launched, 0 with PDL, 0 violations`).
- Bit-for-bit identical (`torch.equal`) across all `ROW_TILES = (16, 32, 64, 128)`, both `torch.bfloat16` and `torch.float32` head dtypes, `vocab < N` padded shards, non-zero `vocab_start`, and ties.

## Benchmark Results (`dgx4` GB10, `N = 32,320, K = 5,120`, FP32 head)

| Operation | Rows | Before (`µs`) | After (`µs`) | Speedup | Saved (`µs`) |
|---|---:|---:|---:|---:|---:|
| `packed_vocab_projection` (`bm=64` → `bm=128`) | 65 | 1649.0 | 1510.2 | **1.092×** | 138.8 |
| `packed_vocab_projection` (`bm=64` → `bm=128`) | 72 | 1607.5 | 1474.0 | **1.091×** | 133.5 |
| `packed_vocab_projection` (`bm=64` → `bm=128`) | 96 | 1738.7 | 1545.3 | **1.125×** | 193.4 |
| `packed_vocab_projection` (`bm=64` → `bm=128`) | 128 | 1863.3 | 1812.5 | **1.028×** | 50.8 |
| Greedy shard pairs (`project_head` + `greedy_shard_pair` → `project_head_greedy_pairs`) | 6 | 1090.7 | 1082.0 | **1.008×** | 8.7 |
| Greedy shard pairs (`project_head` + `greedy_shard_pair` → `project_head_greedy_pairs`) | 24 | 1147.4 | 1125.9 | **1.019×** | 21.5 |
| Greedy shard pairs (`project_head` + `greedy_shard_pair` → `project_head_greedy_pairs`) | 48 | 1284.3 | 1258.5 | **1.021×** | 25.8 |
| Greedy shard pairs (`bm=64` + `greedy_shard_pair` → `bm=128` `project_head_greedy_pairs`) | 96 | 1818.2 | 1620.9 | **1.122×** | 197.3 |
