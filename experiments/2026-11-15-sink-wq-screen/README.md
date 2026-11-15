# sink-wq-screen

Arm 2 of the bit-exact screening series on `tp4-switched` (`sp1..sp4`, switched RoCE TP4, context `1,048,576`, `VLLM_DS41_LSA=0`, image `spark-ds41f:candidate-55`, profile `experiments/2026-11-15-sink-wq-screen/tp4-switched-sink-wq-dev.json`).

## Base & Intended Variable

- **Base commit**: `58d9f93339d11cc53a29d5c112b7368d6340c605` (`c58-prefill-10k` Step 3)
- **Intended variable (`8bfb3f0de`)**: Enable certified 8-bit `W_q` Cauchy-Schwarz vocabulary-head screening (`project_head_greedy_pairs` in `tilelang/vocab.py`) during greedy speculative verification (`_verify_greedily` in `model_runner.py`) when `self.sink.rejection` is active:
  - Fold single-column `sink.cut_cols` suppression into the `refine_kernel` pass (`pack12` head).
  - Return per-row `top_k=4` `(values, global_indices)` across TP shards (`row_topk`) for `sink.alternates_of_greedy_top4` on winning bonus rows via `tilelang_sampling.sink_select_top_rows` / `sink.py`, avoiding materializing the full `[rows, shard_vocab]` FP32 logit tensor.
- **Startup fix (`eecd0ea92`)**: Synchronize TP ranks on `get_tp_group().cpu_group` before `tensor_model_parallel_all_gather` inside `DeepseekCompressor.declare_kernels` -> `bind()` (`compressor.py`), avoiding RoCE watchdog expiry (`failed-starts/20261115T161623Z-spark-ds41f/`) when first-time TileLang kernel compilation creates startup skew across ranks.

## Verification

- **Unit tests (on `sp1` inside `spark-ds41f` against overlay)**:
  `pytest tests/v1/worker/test_gpu_sink.py tests/kernels/test_deepseek_v41_tilelang_vocab.py` -> `122 passed, 6 skipped in 26.15s` (`0` failures), verifying exact parity for single/multi-shard `sink_cut_cols`, `top_k=4` alternates, and CUDA graph execution.
- **Config validation (`sp8`)**:
  `./bin/spark --cluster-config experiments/2026-11-15-sink-wq-screen/tp4-switched-sink-wq-dev.json doctor` -> `configuration OK`.
- **Server logs (`sp1`)**: Both Boot 1 and Boot 2 logged `Greedy verification from vocabulary shards: on` and `Using certified W_q screened greedy verification under sink rejection (top_k=4)` with zero tracebacks or fallbacks.

## Benchmark Commands & Workload Identity

```bash
./bin/spark --cluster-config experiments/2026-11-15-sink-wq-screen/tp4-switched-sink-wq-dev.json \
  bench --compare none --suites quality,decode --concurrency 1,8,16 \
  --output experiments/2026-11-15-sink-wq-screen/runs/sink-wq-b1

./bin/spark --cluster-config experiments/2026-11-15-sink-wq-screen/tp4-switched-sink-wq-dev.json \
  bench --compare none --suites quality,decode --concurrency 1,8,16 \
  --output experiments/2026-11-15-sink-wq-screen/runs/sink-wq-b2
```

- `suites = ["quality", "decode"]`, `concurrency = [1, 8, 16]`, `decode_cases = ["prose", "code"]`, `min_samples = 3`, `max_samples = 4`, `precision_pct = 2.0`, `seed = 0`.
- `prompts_sha256 = cc7f29400eff69775f0d9fb016be00924f46a81be7b42dc5c15a4eb9d82da779`, `precision_prompts_sha256 = a8703e4bf1f55b2360c351c335f00249653557d34ceb79d4e2e749570a2c91bb`.

## Results

### Quality

- **Boot 1 (`runs/sink-wq-b1/bench.json`)**: `5/5` pass (content digest `9fa4c641f714ddb1` on sorted JSON list / `8a177d7fb10287c0` per probe list), bit-identical to `1107-switched` and `c58`.
- **Boot 2 (`runs/sink-wq-b2/bench.json`)**: `5/5` pass (content digest `9fa4c641f714ddb1` / `8a177d7fb10287c0`), bit-identical to Boot 1, `1107-switched`, and `c58`.

### Decode (`tok/s`, `step_ms`, `decode_window_tps`)

| Case | Boot 1 | Boot 2 | 2-Boot Mean | `c58` (`20261115T112802Z-source/decode.json`) | `1107-switched` | vs `1107-switched` | vs `c58` |
|---|---|---|---|---|---|---|---|
| `prose-c1` | `79.76` ±1.59% (`26.43 ms`) | `80.05` ±1.64% (`26.35 ms`) | `79.91` (`26.39 ms`, win `84.32`) | `78.92` (`26.87 ms`, win `83.19`) | `73.34` (`27.41 ms`, win `77.08`) | `+9.0%` (`-3.7%` step) | `+1.3%` (`-1.8%` step) |
| `prose-c8` | `438.08` ±12.42% (`35.51 ms`) | `421.96` ±11.06% (`37.52 ms`) | `430.02` (`36.51 ms`, win `465.13`) | `397.13` (`39.51 ms`, win `426.23`) | `396.10` (`39.00 ms`, win `425.44`) | `+8.6%` (`-6.4%` step) | `+8.3%` (`-7.6%` step) |
| `prose-c16` | `631.32` ±20.68% (`48.78 ms`) | `643.43` ±13.88% (`47.59 ms`) | `637.37` (`48.19 ms`, win `685.97`) | `651.72` (`46.72 ms`, win `702.23`) | `587.04` (`51.57 ms`, win `627.17`) | `+8.6%` (`-6.6%` step) | `-2.2%` |
| `code-c1` | `97.81` ±2.34% (`29.83 ms`) | `98.50` ±1.83% (`29.63 ms`) | `98.15` (`29.73 ms`, win `105.29`) | `98.69` (`29.60 ms`, win `105.73`) | `94.96` (`29.85 ms`, win `101.69`) | `+3.4%` (`-0.4%` step) | `-0.5%` |
| `code-c8` | `525.14` ±20.72% (`40.79 ms`) | `522.98` ±20.48% (`41.10 ms`) | `524.06` (`40.94 ms`, win `580.66`) | `549.27` (`38.36 ms`, win `610.66`) | `490.12` (`42.38 ms`, win `539.01`) | `+6.9%` (`-3.4%` step) | `-4.6%` |
| `code-c16` | `793.92` ±7.74% (`49.94 ms`) | `798.50` ±12.99% (`50.73 ms`) | `796.21` (`50.33 ms`, win `877.03`) | `799.10` (`50.26 ms`, win `880.55`) | `677.32` (`59.38 ms`, win `736.68`) | `+17.6%` (`-15.2%` step) | `-0.4%` |

## Errors, Memory & Thermal Observations

- **Boot 1 failed start (`runs/failed-starts/20261115T161623Z-spark-ds41f/`)**: RoCE all-gather timeout on startup before `_startup_rendezvous`; fixed in `eecd0ea92` and TileLang JIT caches synchronized across `sp1..sp4`.
- **Boot 1**: API ready at `+138.7s`; min `MemAvailable` `sp1 21.52`, `sp2 23.76`, `sp3 23.82`, `sp4 23.87` GiB; `swap_growth_gib = 0.0`; max GPU temp `62.0 C`; `thermal_slowdown_ms = 0.0`; `power_cap_ms = 0.0`.
- **Boot 2**: API ready at `+138.9s`; min `MemAvailable` `sp1 21.57`, `sp2 23.81`, `sp3 23.80`, `sp4 23.89` GiB; `swap_growth_gib = 0.0`; max GPU temp `63.0 C`; `thermal_slowdown_ms = 0.0`; `power_cap_ms = 0.0`.

## Conclusion

Certified 8-bit `W_q` screening with folded `sink.cut` and `top_k=4` alternate selection executes cleanly in serving under `self.sink.rejection = True` with 100% bit-exact output (`5/5` quality identical to `1107-switched` and `c58`), eliminating the `[rows, 32320]` FP32 logit materialization on greedy verification steps. Decode throughput improves `+3.4%` to `+17.6%` over `1107-switched` across all 6 concurrency points and is flat within 4-sample CIs vs `c58` (`prose-c1 -1.8%` step time, `prose-c8 -7.6%` step time, higher-concurrency `c8/c16` points within `11-21%` 4-sample CIs).
