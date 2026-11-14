# TP=2 Zero-Copy io_uring MoE Expert Slot Offloading (#981)

Evaluates `#981` zero-copy `pin_memory` + `io_uring` (`READ_FIXED`) MoE expert slot caching over `candidate-55` Build 9 (`spark-ds41f:candidate-55`) on two-node topologies:

- `tp2-switched-dev.json`: `sp5..sp6` (`oneshot-switched`), `VLLM_DS41_EXPERT_SLOTS=228`, `VLLM_DS41_EXPERT_DYNAMIC_SLOTS=64`, `VLLM_DS41_EXPERT_PREFILL_PROTECT_ROWS=24`.
- `tp2-direct-dev.json`: `dgx1..dgx2` (`oneshot-direct`), `VLLM_DS41_EXPERT_SLOTS=228`, `VLLM_DS41_EXPERT_DYNAMIC_SLOTS=64`, `VLLM_DS41_EXPERT_PREFILL_PROTECT_ROWS=24`.

Both profiles use `PIECEWISE` target cudagraph mode (resolving target `ModelCudaGraphManager` to `CUDAGraphMode.NONE` while executing `io_uring` slot paging per step) and 4-stream 256K KV sizing (`max_model_len=262144`, `max_num_seqs=4`, `max_num_batched_tokens=4096`, `kv_cache_memory_bytes=2362232012`, `max_cudagraph_capture_size=24`). `suite-screen-tp2.json` runs `model` (`c4`), `grid` (`max_requests=4`), `c4` (`closed`), and `prefill` (`32K`, `256K`).
