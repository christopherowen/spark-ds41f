# TP=2 Zero-Copy io_uring MoE Expert Slot Offloading (#981)

Evaluates `#981` zero-copy `pin_memory` + `io_uring` (`READ_FIXED`) MoE expert slot caching over `candidate-55` Build 9 (`spark-ds41f:candidate-55`) on two-node topologies:

- `tp2-switched-dev.json`: `sp5..sp6` (`oneshot-switched`), `VLLM_DS41_EXPERT_SLOTS=228`, `VLLM_DS41_EXPERT_DYNAMIC_SLOTS=64`, `VLLM_DS41_EXPERT_PREFILL_PROTECT_ROWS=24`.
- `tp2-direct-dev.json`: `dgx1..dgx2` (`oneshot-direct`), `VLLM_DS41_EXPERT_SLOTS=228`, `VLLM_DS41_EXPERT_DYNAMIC_SLOTS=64`, `VLLM_DS41_EXPERT_PREFILL_PROTECT_ROWS=24`.

Both profiles use `PIECEWISE` target cudagraph mode (resolving target `ModelCudaGraphManager` to `CUDAGraphMode.NONE` while executing `io_uring` slot paging per step) and 4-stream 256K KV sizing (`max_model_len=262144`, `max_num_seqs=4`, `max_num_batched_tokens=4096`, `kv_cache_memory_bytes=2362232012`, `max_cudagraph_capture_size=24`). `suite-screen-tp2.json` runs `model` (`c4`), `grid` (`max_requests=4`), `c4` (`closed`), and `prefill` (`32K`, `256K`).

## Measured Results

### `tp2-direct-dev.json` (`dgx1..dgx2`, `59c9a2d8d`, `c4`, `228/64`, `protect_rows=24`)

Run: `results/private/suites/tp2-slots-screen-direct-c4` (`suite-screen-tp2.json`)

- **Model (`c4`)**: `acceptance_length = 2.4663`, `rank1_fraction = 1.0`, `perplexity = 1.3102`, `outputs_sha256 = 78175f5d5dee993a073fcdf179f0a8f03246e21823e9f90b22984addcda8b6ec`, `failed = 0`.
- **Engine grid**:
  - `decode 1x2 rows @4K`: `48.859 ms ±2.02%`
  - `decode 4x8 rows @4K`: `51.748 ms ±2.30%`
  - `decode 1x3 rows @4K`: `50.351 ms ±2.22%`
  - `decode 4x12 rows @4K`: `53.555 ms ±1.56%`
  - `decode 1x6 rows @4K`: `50.820 ms ±2.12%`
  - `decode 2x12 rows @4K`: `51.786 ms ±2.42%`
  - `decode 4x24 rows @4K`: `56.185 ms ±2.73%`
  - `decode 1x3 rows @64K`: `50.146 ms ±2.53%`
  - `decode 1x3 rows @256K`: `49.502 ms ±2.22%`
  - (`9` shapes, `widest_ci95_pct = 2.73%`).
- **Closed loop (`c4`)**: `12.083 tok/s ± 5.23%`, `steps_per_s = 1.203`, `tokens_per_step = 10.07`, `ttft_p50 = 2.826 s`, `tpot_p50 = 322.382 ms`, `failed = 0`.
- **Prefill**: `32768 = 172.68 tok/s ± 23.49%` (samples `161.53`, `165.09`, `191.42 tok/s`); `262144` was skipped at the `2.20 GiB` KV sizing in `59c9a2d8d`.

### Earlier `tp2-switched` (`sp5..sp6`, `c16` pre-`59c9a2d8d`, `228/64`)

- **Closed loop (`c16`)**: `9.370 tok/s ± 5.87%`, `failed = 0`.
- **Prefill**: `32768 = 101.58 tok/s ± 22.67%` (samples `95.88`, `96.59`, `112.28 tok/s`).

### Finalized TP2 Runtime Defaults (`930c6aa03`)

Both `tp2-direct-dev.json` and `tp2-switched-dev.json` finalize:
- `--max-num-seqs 4` (`concurrency = 4`)
- `--kv-cache-memory-bytes 6442450944` (`6.00 GiB`)
- `--prefix-cache-retention-interval 8192`
- `enable_adaptive_verification = false` (target runs eager under `PIECEWISE` for per-step `io_uring` slot paging)
- `VLLM_MAX_IMAGE_PIXELS=16777216` in `overlay/vllm/vllm/envs.py` (mounted read-only)
