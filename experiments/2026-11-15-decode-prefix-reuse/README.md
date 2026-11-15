# decode-prefix-reuse

Arm 4 of the bit-exact screening series on `tp4-switched` (`sp1..sp4`, switched RoCE TP4, context `1,048,576`, `VLLM_DS41_LSA=0`, image `spark-ds41f:candidate-55`, profile `experiments/2026-11-15-decode-prefix-reuse/tp4-switched-prefix-reuse-dev.json`).

## Base & Intended Variable

- **Base commit**: `2187700ef` (`dspark-markov-trunc`, Arm 3)
- **Intended variable (`54aae2f4c`)**: Intra-request `KPool` / `KScale` register-tile prefix reuse and `<1>` / `<6>` / `<4>` row-block specialization in the dense TileLang indexer (`overlay/vllm/vllm/models/deepseek_v4_1/tilelang/indexer.py`):
  - In `__tl_score_rows_dense_fast<int R>`, replaced the rigid all-row `same_k` tile predicate with per-row sequential `loaded_page` tracking (`loaded_page = -2` initially), reusing `A_reg` and `sfk_reg` whenever `full_tile = (R > 1) && single_page && (p0 + 64 <= ext)` and `cur_page >= 0 && cur_page == loaded_page` (partial tail tiles keep `!full_tile`, reloading safely and resetting `loaded_page = -2`); guarded `contrib` overwrites with `if (!need_load) __syncthreads();`.
  - In `indexer_score_v41`, dispatched `rows <= 1` to `__tl_score_rows_dense_fast<1>`, `rows <= 6` to `__tl_score_rows_dense_fast<6>` (`row_blocks = T.ceildiv(rows, 6)`), and `rows > 6` to `__tl_score_rows_dense_fast<4>` (`row_blocks = T.ceildiv(rows, 4)`).
  - Updated `overlay/vllm/tests/kernels/attention/test_deepseek_v41_tilelang.py` (`test_indexer_score_vectorizes_kpool_loads_and_coschedules_dense_rows` and `test_indexer_dense_decode_prefix_reuse_bit_exact`).

## Verification

- **Unit tests (on `sp1` inside `spark-ds41f` against overlay)**:
  `pytest --noconftest -q /tmp/arm4_root/tests/kernels/attention/test_deepseek_v41_tilelang.py -k "indexer"` -> `29 passed, 1 skipped, 56 deselected, 1 xfailed in 12.60s` (`test_indexer_reads_only_visible_rows` skipped without `compute-sanitizer`; `test_indexer_nonfinite_query_scale` xfailed).
- **Standalone `IndexerV41Plan(32768, 256)` microbenchmark (`sp1`, Arm 4 overlay vs unmodified container base)**:
  - `rows=1`: `0.045 ms` vs `0.043 ms`
  - `rows=4`: `0.043 ms` vs `0.042 ms`
  - `rows=6`: `0.042 ms` vs `0.044 ms`
  - `rows=8`: `0.048 ms` vs `0.052 ms`
  - `rows=16`: `0.073 ms` vs `0.086 ms`
  - `rows=32`: `0.121 ms` vs `0.154 ms`
- **Config validation (`sp8`)**:
  `./bin/spark --cluster-config experiments/2026-11-15-decode-prefix-reuse/tp4-switched-prefix-reuse-dev.json doctor` -> `configuration OK`.
- **Server logs (`sp1`)**: Both Boot 1 and Boot 2 logged `DSpark pre-drafting Markov head truncation enabled`, `Greedy verification from vocabulary shards: on`, and `Using certified W_q screened greedy verification under sink rejection (top_k=4)`.

## Benchmark Commands & Workload Identity

```bash
./bin/spark --cluster-config experiments/2026-11-15-decode-prefix-reuse/tp4-switched-prefix-reuse-dev.json \
  bench --compare none --suites quality,decode --concurrency 1,8,16 \
  --output experiments/2026-11-15-decode-prefix-reuse/runs/prefix-reuse-b1

./bin/spark --cluster-config experiments/2026-11-15-decode-prefix-reuse/tp4-switched-prefix-reuse-dev.json \
  bench --compare none --suites prefill --prefill-text source \
  --prefill-sizes 32768,131072,262144 --prefill-repeats 2 \
  --output experiments/2026-11-15-decode-prefix-reuse/runs/prefix-reuse-b1-prefill

./bin/spark --cluster-config experiments/2026-11-15-decode-prefix-reuse/tp4-switched-prefix-reuse-dev.json \
  bench --compare none --suites quality,decode --concurrency 1,8,16 \
  --output experiments/2026-11-15-decode-prefix-reuse/runs/prefix-reuse-b2

./bin/spark --cluster-config experiments/2026-11-15-decode-prefix-reuse/tp4-switched-prefix-reuse-dev.json \
  bench --compare none --suites prefill --prefill-text source \
  --prefill-sizes 32768,131072,262144 --prefill-repeats 2 \
  --output experiments/2026-11-15-decode-prefix-reuse/runs/prefix-reuse-b2-prefill
```

- `prompts_sha256 = cc7f29400eff69775f0d9fb016be00924f46a81be7b42dc5c15a4eb9d82da779`, `precision_prompts_sha256 = a8703e4bf1f55b2360c351c335f00249653557d34ceb79d4e2e749570a2c91bb`, `seed = 0`.
- Decode suite: `concurrency = [1, 8, 16]`, `decode_cases = ["prose", "code"]`, `min_samples = 3`, `max_samples = 4`, `precision_pct = 2.0`.
- Prefill suite: `prefill_text = "source"`, `prefill_sizes = [32768, 131072, 262144]`, `prefill_repeats = 2`.

## Results

### Quality & Determinism

- **Boot 1 (`runs/prefix-reuse-b1/bench.json`)**: `5/5` pass (`ordered_sha256 = sorted_sha256 = 9fa4c641f714ddb1`), `distinct_outputs = 1` on all 6 decode workloads, bit-identical to Arm 3, Arm 2, `c58`, and `1107-switched`.
- **Boot 2 (`runs/prefix-reuse-b2/bench.json`)**: `5/5` pass (`ordered_sha256 = sorted_sha256 = 9fa4c641f714ddb1`), `distinct_outputs = 1` on all 6 decode workloads, bit-identical across boots and baselines.

### Decode (`tok/s`, `step_ms`, `decode_window_tps`)

| Case | Boot 1 | Boot 2 | 2-Boot Mean | Arm 3 (`dspark-markov-trunc`) | vs Arm 3 | vs Arm 2 | vs `c58` | vs `1107-switched` |
|---|---|---|---|---|---|---|---|---|
| `prose-c1` | `80.02` ±0.36% (`26.40 ms`) | `80.03` ±2.06% (`26.39 ms`) | `80.03` (`26.40 ms`, win `84.45`) | `80.18` (`26.39 ms`) | `-0.2%` (`+0.0%` step) | `+0.2%` | `+1.4%` | `+9.1%` (`-3.7%` step) |
| `prose-c8` | `421.83` ±13.21% (`37.14 ms`) | `411.14` ±12.77% (`38.38 ms`) | `416.48` (`37.76 ms`, win `449.25`) | `406.08` (`39.04 ms`) | `+2.6%` (`-3.3%` step) | `-3.1%` | `+4.9%` | `+5.1%` (`-3.2%` step) |
| `prose-c16` | `629.32` ±17.20% (`49.05 ms`) | `643.93` ±17.37% (`47.85 ms`) | `636.62` (`48.45 ms`, win `683.96`) | `642.65` (`47.78 ms`) | `-0.9%` (`+1.4%` step) | `-0.1%` | `-2.3%` | `+8.4%` (`-6.1%` step) |
| `code-c1` | `98.49` ±1.18% (`29.55 ms`) | `98.71` ±0.93% (`29.49 ms`) | `98.60` (`29.52 ms`, win `105.73`) | `98.90` (`29.42 ms`) | `-0.3%` (`+0.3%` step) | `+0.5%` | `-0.1%` | `+3.8%` (`-1.1%` step) |
| `code-c8` | `523.83` ±20.74% (`40.78 ms`) | `527.22` ±21.83% (`40.87 ms`) | `525.52` (`40.83 ms`, win `583.05`) | `543.26` (`39.05 ms`) | `-3.3%` (`+4.5%` step) | `+0.3%` | `-4.3%` | `+7.2%` (`-3.7%` step) |
| `code-c16` | `755.12` ±14.34% (`52.97 ms`) | `798.34` ±12.76% (`51.46 ms`) | `776.73` (`52.21 ms`, win `853.42`) | `813.99` (`49.50 ms`) | `-4.6%` (`+5.5%` step) | `-2.4%` | `-2.8%` | `+14.7%` (`-12.1%` step) |

### Prefill (`source`, `--prefill-repeats 2`)

| Size | Boot 1 | Boot 2 | 2-Boot Mean | `c58` Step 3 2-Boot Mean | vs `c58` Step 3 |
|---|---|---|---|---|---|
| `32768` | `9311.8` ±7.13% (`3.120s`) | `9319.5` ±1.30% (`3.117s`) | `9315.7` (`3.119s`) | `9383.9` (`3.096s`) | `-0.73%` |
| `131072` | `9618.6` ±5.18% (`11.990s`) | `9622.4` ±3.98% (`11.985s`) | `9620.5` (`11.988s`) | `9702.1` (`11.886s`) | `-0.84%` |
| `262144` | `9448.2` ±2.99% (`25.389s`) | `9447.1` ±5.67% (`25.392s`) | `9447.7` (`25.390s`) | `9628.7` (`24.913s`) | `-1.88%` |

## Errors, Memory & Thermal Observations

- **Boot 1 Decode** (API ready +143.6s, cluster ready +144.0s): min `MemAvailable` sp1 21.54, sp2 23.81, sp3 23.82, sp4 23.89 GiB; `swap_growth_gib = 0.0`; max GPU temps sp1 63.0 C, sp2 61.0 C, sp3 60.0 C, sp4 59.0 C; `thermal_slowdown_ms = 0.0`; `power_cap_ms = 0.0`.
- **Boot 1 Prefill** (same Boot 1 launch): min `MemAvailable` sp1 17.09, sp2 19.67, sp3 19.09, sp4 19.52 GiB; `swap_growth_gib = 0.0`; max GPU temps sp1 74.0 C, sp2 74.0 C, sp3 73.0 C, sp4 72.0 C; `thermal_slowdown_ms = 0.0`; `power_cap_ms = 0.0`.
- **Boot 2 Decode** (API ready +138.6s, cluster ready +139.0s): min `MemAvailable` sp1 21.59, sp2 23.83, sp3 23.82, sp4 23.88 GiB; `swap_growth_gib = 0.0`; max GPU temps sp1 65.0 C, sp2 62.0 C, sp3 61.0 C, sp4 60.0 C; `thermal_slowdown_ms = 0.0`; `power_cap_ms = 0.0`.
- **Boot 2 Prefill** (same Boot 2 launch): min `MemAvailable` sp1 16.99, sp2 19.59, sp3 19.23, sp4 19.69 GiB; `swap_growth_gib = 0.0`; max GPU temps sp1 73.0 C, sp2 76.0 C, sp3 74.0 C, sp4 69.0 C; `thermal_slowdown_ms = 0.0`; `power_cap_ms = 0.0`.

## Conclusion

Intra-request `KPool` register-tile prefix reuse and `<1>` / `<6>` / `<4>` row-block specialization preserved 100% bit-exact outputs (`29 passed, 1 skipped, 1 xfailed` on `indexer` unit tests, `5/5` quality digest `9fa4c641f714ddb1`, `distinct_outputs = 1` on all 6 decode cases across both boots) and accelerated the isolated single-request `IndexerV41Plan(32768, 256)` microbenchmark for `rows=6..32` (`0.073 ms` vs `0.086 ms` at `rows=16`, `0.121 ms` vs `0.154 ms` at `rows=32`).

End-to-end 2-boot serving throughput did **not** improve over Arm 3 (`dspark-markov-trunc`) or `c58` Step 3: decode differences stayed within run-to-run CI variance (`prose-c8` `+2.6%` vs Arm 3 / `-3.1%` vs Arm 2, `code-c8` `-3.3%` vs Arm 3 / `+0.3%` vs Arm 2, `code-c16` `-4.6%` vs Arm 3 due to Boot 1 `755.12` while Boot 2 `798.34` matched `c58` `799.10`), and `source` prefill trailed `c58` Step 3 by `-0.73%` (`32K`), `-0.84%` (`131K`), and `-1.88%` (`262K`). Keep Arm 4 recorded here as measured rather than promoting this `indexer.py` change on top of Arm 3 as an always-on default.
