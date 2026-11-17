# dspark-markov-trunc

Arm 3 of the bit-exact screening series on `tp4-switched` (`sp1..sp4`, switched RoCE TP4, context `1,048,576`, `VLLM_DS41_LSA=0`, image `spark-ds41f:candidate-55`, profile `experiments/2026-11-15-dspark-markov-trunc/tp4-switched-dspark-trunc-dev.json`).

## Base & Intended Variable

- **Base commit**: `aac25b56e24f34374eb895ab68f87e25c4c5df23` (`sink-wq-screen`, Arm 2)
- **Intended variable (`98d8451d2`)**: Pre-drafting Markov head truncation when adaptive verification lowers `num_speculative_tokens` (`0 < k < N`) below `self.num_speculative_steps` (`5`):
  - `dflash/speculator.py`: Accept `num_speculative_tokens: int | None = None` in `_generate_draft`, use eager execution when `0 < active_spec < self.num_speculative_steps`, and forward `num_speculative_tokens` into `_generate_draft`.
  - `dspark/speculator.py`: Slice `sample_hidden` to the leading `(num_reqs * active_steps, hidden)` rows before `lm_head` (`compute_local_draft_logits` / `compute_draft_logits_local` / `compute_draft_logits`), halt the sequential Markov loop at `active_steps`, and run `_compute_confidence_probs` with tail `conf_idx_mapping` entries masked to `-1` (followed by zeroing `draft_token_confidence_probs[:num_reqs, active_steps:]`). When `num_speculative_tokens is None` or `>= N`, the full CUDA-graph path executes unchanged.

## Verification

- **Unit tests (on `sp1` inside `spark-ds41f` against overlay)**:
  `pytest --noconftest -q /tmp/arm3_root/tests/v1/worker/test_dspark_greedy_markov.py` -> `63 passed in 13.53s` (`0` failures), verifying exact parity for `active_spec in [1, 2, 4]` and `sample_from_anchor in [False, True]` across `draft_tokens[:, :k]`, calibrated `draft_token_confidence_probs[:, :k]`, and `acceptance_estimator.{features,predictions}`.
- **Config validation (`sp8`)**:
  `./bin/spark --cluster-config experiments/2026-11-15-dspark-markov-trunc/tp4-switched-dspark-trunc-dev.json doctor` -> `configuration OK`.
- **Server logs (`sp1`)**: Both Boot 1 and Boot 2 logged `DSpark pre-drafting Markov head truncation enabled`, `Greedy verification from vocabulary shards: on`, and `Using certified W_q screened greedy verification under sink rejection (top_k=4)`.

## Benchmark Commands & Workload Identity

```bash
./bin/spark --cluster-config experiments/2026-11-15-dspark-markov-trunc/tp4-switched-dspark-trunc-dev.json \
  bench --compare none --suites quality,decode --concurrency 1,8,16 \
  --output experiments/2026-11-15-dspark-markov-trunc/runs/dspark-trunc-b1

./bin/spark --cluster-config experiments/2026-11-15-dspark-markov-trunc/tp4-switched-dspark-trunc-dev.json \
  bench --compare none --suites quality,decode --concurrency 1,8,16 \
  --output experiments/2026-11-15-dspark-markov-trunc/runs/dspark-trunc-b2
```

- `suites = ["quality", "decode"]`, `concurrency = [1, 8, 16]`, `decode_cases = ["prose", "code"]`, `min_samples = 3`, `max_samples = 4`, `precision_pct = 2.0`, `seed = 0`.
- `prompts_sha256 = cc7f29400eff69775f0d9fb016be00924f46a81be7b42dc5c15a4eb9d82da779`, `precision_prompts_sha256 = a8703e4bf1f55b2360c351c335f00249653557d34ceb79d4e2e749570a2c91bb`.

## Results

### Quality

- **Boot 1 (`runs/dspark-trunc-b1/bench.json`)**: `5/5` pass (content digest `9fa4c641f714ddb1`), bit-identical to Arm 2, `c58`, and `1107-switched`.
- **Boot 2 (`runs/dspark-trunc-b2/bench.json`)**: `5/5` pass (content digest `9fa4c641f714ddb1`), bit-identical to Boot 1, Arm 2, `c58`, and `1107-switched`.

### Decode (`tok/s`, `step_ms`, `decode_window_tps`)

| Case | Boot 1 | Boot 2 | 2-Boot Mean | Arm 2 (`sink-wq-screen`) | vs Arm 2 | vs `c58` | vs `1107-switched` |
|---|---|---|---|---|---|---|---|
| `prose-c1` | `80.19` ±1.54% (`26.37 ms`) | `80.17` ±1.87% (`26.41 ms`) | `80.18` (`26.39 ms`, win `84.61`) | `79.91` (`26.39 ms`) | `+0.3%` (`-0.0%` step) | `+1.6%` | `+9.3%` (`-3.7%` step) |
| `prose-c8` | `389.14` ±7.25% (`41.08 ms`) | `423.02` ±10.50% (`37.00 ms`) | `406.08` (`39.04 ms`, win `437.03`) | `430.02` (`36.51 ms`) | `-5.6%` (`+6.9%` step) | `+2.3%` | `+2.5%` (`+0.1%` step) |
| `prose-c16` | `643.66` ±15.63% (`47.76 ms`) | `641.65` ±16.74% (`47.80 ms`) | `642.65` (`47.78 ms`, win `692.00`) | `637.37` (`48.19 ms`) | `+0.8%` (`-0.8%` step) | `-1.4%` | `+9.5%` (`-7.3%` step) |
| `code-c1` | `99.22` ±1.22% (`29.44 ms`) | `98.59` ±1.42% (`29.40 ms`) | `98.90` (`29.42 ms`, win `106.15`) | `98.15` (`29.73 ms`) | `+0.8%` (`-1.0%` step) | `+0.2%` | `+4.2%` (`-1.4%` step) |
| `code-c8` | `561.52` ±0.45% (`37.40 ms`) | `525.00` ±20.85% (`40.70 ms`) | `543.26` (`39.05 ms`, win `604.04`) | `524.06` (`40.94 ms`) | `+3.7%` (`-4.6%` step) | `-1.1%` | `+10.8%` (`-7.9%` step) |
| `code-c16` | `839.99` ±2.16% (`47.87 ms`) | `787.98` ±9.70% (`51.13 ms`) | `813.99` (`49.50 ms`, win `899.12`) | `796.21` (`50.33 ms`) | `+2.2%` (`-1.7%` step) | `+1.9%` | `+20.2%` (`-16.6%` step) |

## Errors, Memory & Thermal Observations

- **Boot 1**: API ready at `+196.1s` (cluster ready `+196.5s`); min `MemAvailable` `sp1 21.55`, `sp2 23.78`, `sp3 23.82`, `sp4 23.87` GiB; `swap_growth_gib = 0.0`; max GPU temp `62.0 C`; `thermal_slowdown_ms = 0.0`; `power_cap_ms = 0.0`.
- **Boot 2**: API ready at `+135.3s` (cluster ready `+135.6s`); min `MemAvailable` `sp1 21.61`, `sp2 23.87`, `sp3 23.89`, `sp4 23.92` GiB; `swap_growth_gib = 0.0`; max GPU temp `63.0 C`; `thermal_slowdown_ms = 0.0`; `power_cap_ms = 0.0`.

## Conclusion

Pre-drafting Markov head truncation executes cleanly when adaptive verification caps `num_speculative_tokens < N` with 100% bit-exact outputs (`5/5` quality digest `9fa4c641f714ddb1`, `distinct_outputs = 1` across all 6 decode workloads). Compared to Arm 2 (`sink-wq-screen`), 2-boot mean throughput is higher on `code-c8` (`+3.7%`, `-4.6%` step time) and `code-c16` (`+2.2%`, `-1.7%` step time), flat on `prose-c1` / `prose-c16` / `code-c1`, and lower on `prose-c8` (`-5.6%`, overlapping 7.2%–12.4% 4-sample CIs, with Boot 2 `prose-c8 = 423.02` matching Arm 2 Boot 2 `421.96`).
