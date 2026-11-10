# 2026-11-10 — Shard-Local Sampled Verification and Resampling (`0762`)

## Base commit and single causal variable

- **Base commit**: `cc20ed18be1f7cbf5c6c53cdee945f7cc77d5fc5` (`spark-ds41f:1107`)
- **Upstream vLLM base**: `19f2c20ed4d6be5dc5eb7323efccba6ad0eb422f`
- **Single causal variable**: Extend vocabulary-shard-local speculative verification and 1-token sampling from all-greedy batches (`temperature == 0.0`) to batches containing sampled streams (`temperature > 0.0`, including the recipe default `temperature == 1.0`) when no non-temperature logits processors or logprobs are active (`vllm/v1/worker/gpu/spec_decode/tilelang_rejection.py`, `vllm/v1/worker/gpu/spec_decode/greedy_verification.py`, `vllm/v1/worker/gpu/model_runner.py`).

## Mechanism

Previously, a single request with `temperature > 0` in a decode batch caused `GPUModelRunner._verifies_greedily` to return `False`, forcing `self.model.compute_logits(sample_hidden_states)` to all-gather the full `[num_logits, 129280]` BF16 vocabulary logits across tensor-parallel ranks (`258.6 KB/row`, or `18.18 MB` total wire traffic across TP4 ranks at 16 streams / 96 rows) before running `verify_and_resample` and `insert_resampled` over all 129,280 columns on every rank.

With patch `0762-perf-shard-local-sampled-logits.patch`:
1. **`local_stats_shard_kernel`**: Each TP rank scans only its local shard (`32,320` columns on TP4, `43,094` on TP3) in `STATS_BLOCK = 8192` tiles (`4` blocks on TP4, `6` on TP3) and emits a 16-byte-aligned row of `4` FP32 values per block (`[block_max, block_sumexp, owns_draft, draft_logit]` for `temp != 0`, or `[block_max, 0, float32(global_argmax), 0]` for `temp == 0`).
2. **One-shot all-gather of block statistics**: `tensor_model_parallel_all_gather(local_stats, dim=-1)` exchanges `64 B/row/rank` on TP4 (`96 B/row/rank` on TP3) instead of `64.6 KB/row/rank` (`86.3 KB/row/rank` on TP3).
3. **`verify_shard_kernel`**: Combines the gathered block maxima, rescaled exponential sums (`vllm_row_others` + `vllm_softplus`), and the draft token's target logit from the owning shard to run block verification (`temp > 0`) and greedy verification (`temp == 0`) in a single launch.
4. **`resample_shard_kernel` + `resample_shard_pair_kernel` + `insert_shard_resampled_kernel`**: Each rank computes Gumbel-max scores over its local shard using global token indices (`vocab_start + col`) for the Philox RNG draw and rejected-draft mask, reduces its shard to a single 16-byte `(best_max, float32(best_id), 0, 0)` pair per request, all-gathers `[num_reqs, 4 * tp_size]` FP32 pairs, and commits the winning token ID and count.

## Exact commands

```bash
# Container boot 1
docker run --rm --gpus all --ipc=host --entrypoint pytest \
  -v "$VLLM_SRC/vllm/v1/worker/gpu:/opt/spark-ds41f/candidate/vllm/vllm/v1/worker/gpu" \
  -v "$VLLM_SRC/tests:/workspace/tests" \
  spark-ds41f:1107 -q \
  /workspace/tests/v1/worker/gpu/test_greedy_verification.py \
  /workspace/tests/v1/worker/test_gpu_verified_logprob_token_ids.py \
  /workspace/tests/v1/worker/test_dead_rows.py \
  | tee experiments/2026-11-10-shard-local-sampled-logits/receipts/boot1-tests.txt

docker run --rm --gpus all --ipc=host --entrypoint python3 \
  -v "$VLLM_SRC/vllm/v1/worker/gpu:/opt/spark-ds41f/candidate/vllm/vllm/v1/worker/gpu" \
  -v "$PWD/experiments/2026-11-10-shard-local-sampled-logits:/workspace/exp" \
  spark-ds41f:1107 /workspace/exp/bench_shard_sampled.py \
  | tee experiments/2026-11-10-shard-local-sampled-logits/receipts/boot1-bench.txt

# Container boot 2 (repeat boot)
docker run --rm --gpus all --ipc=host --entrypoint pytest \
  -v "$VLLM_SRC/vllm/v1/worker/gpu:/opt/spark-ds41f/candidate/vllm/vllm/v1/worker/gpu" \
  -v "$VLLM_SRC/tests:/workspace/tests" \
  spark-ds41f:1107 -q \
  /workspace/tests/v1/worker/gpu/test_greedy_verification.py \
  /workspace/tests/v1/worker/test_gpu_verified_logprob_token_ids.py \
  /workspace/tests/v1/worker/test_dead_rows.py \
  | tee experiments/2026-11-10-shard-local-sampled-logits/receipts/boot2-tests.txt

docker run --rm --gpus all --ipc=host --entrypoint python3 \
  -v "$VLLM_SRC/vllm/v1/worker/gpu:/opt/spark-ds41f/candidate/vllm/vllm/v1/worker/gpu" \
  -v "$PWD/experiments/2026-11-10-shard-local-sampled-logits:/workspace/exp" \
  spark-ds41f:1107 /workspace/exp/bench_shard_sampled.py \
  | tee experiments/2026-11-10-shard-local-sampled-logits/receipts/boot2-bench.txt
```

## Measured results (`boot1` and `boot2` on `sp8`)

- **Correctness**: `57 passed` on `boot1` (`55.93s`) and `57 passed` on `boot2` (`55.11s`) across `TP1`, `TP3`, and `TP4`, including mixed greedy + sampled batches (`temp = 0.0`, `0.7`, `1.0`).

| Shape | Streams (`reqs` / `rows`) | Collective payload (`full` -> `shard`) | Boot 1 step verify (`full` -> `shard`, delta) | Boot 2 step verify (`full` -> `shard`, delta) |
| :--- | :--- | :--- | :--- | :--- |
| **TP4** | `1` (`6` rows) | `1,136.2 KB` -> `1.17 KB` (**970x**) | `226.3 us` -> `91.9 us` (**-134.4 us**) | `393.9 us` -> `92.3 us` (**-301.6 us**) |
| **TP4** | `2` (`12` rows) | `2,272.5 KB` -> `2.34 KB` (**970x**) | `348.2 us` -> `91.9 us` (**-256.3 us**) | `348.5 us` -> `92.6 us` (**-255.8 us**) |
| **TP4** | `4` (`24` rows) | `4,545.0 KB` -> `4.69 KB` (**970x**) | `596.2 us` -> `92.1 us` (**-504.1 us**) | `594.2 us` -> `92.5 us` (**-501.7 us**) |
| **TP4** | `8` (`48` rows) | `9,090.0 KB` -> `9.38 KB` (**970x**) | `1,123.0 us` -> `92.7 us` (**-1,030.3 us**) | `1,121.9 us` -> `93.7 us` (**-1,028.2 us**) |
| **TP4** | `16` (`96` rows) | `18,180.0 KB` -> `18.75 KB` (**970x**) | `2,242.0 us` -> `93.3 us` (**-2,148.6 us**) | `2,247.1 us` -> `95.3 us` (**-2,151.9 us**) |
| **TP3** | `1` (`6` rows) | `1,011.0 KB` -> `1.16 KB` (**874x**) | `396.4 us` -> `82.1 us` (**-314.3 us**) | `208.4 us` -> `83.6 us` (**-124.8 us**) |
| **TP3** | `2` (`12` rows) | `2,022.0 KB` -> `2.31 KB` (**874x**) | `315.4 us` -> `82.5 us` (**-232.9 us**) | `317.3 us` -> `84.5 us` (**-232.8 us**) |
| **TP3** | `4` (`24` rows) | `4,044.0 KB` -> `4.62 KB` (**874x**) | `533.8 us` -> `83.6 us` (**-450.1 us**) | `534.5 us` -> `82.8 us` (**-451.7 us**) |
| **TP3** | `8` (`48` rows) | `8,088.0 KB` -> `9.25 KB` (**874x**) | `1,013.6 us` -> `83.0 us` (**-930.6 us**) | `1,011.7 us` -> `84.3 us` (**-927.5 us**) |
| **TP3** | `16` (`96` rows) | `16,176.0 KB` -> `18.50 KB` (**874x**) | `2,077.7 us` -> `84.0 us` (**-1,993.7 us**) | `2,082.4 us` -> `84.5 us` (**-1,998.0 us**) |

## Conclusion

Eliminating the full-vocabulary logits all-gather on sampled decode steps reduces the verification collective payload by `970x` on TP4 and `874x` on TP3, saving `0.13–2.15 ms` per decode step across `c1..c16`.
