# c58-prefill-10k

Targeted prefill optimizations on `tp4-switched-dev.json` (`sp1..sp4`, 4-node switched RoCE TP4, context 1,048,576, `VLLM_DS41_LSA=0`, image `spark-ds41f:candidate-55`).

## Intended Variables & Sequence

1. **`b31a7c98e`**: Defer next-chunk Engram prefetch until Layer 14 (`receipts/20261115T140556Z-source/`).
2. **Step 2 (`64e721ef1`)**: Event-driven 16-pair `(ready_event, done_event)` ring (`sp_prefill.py`, `kernels.py`) overlapping `gather_many_async` with `wait_xq` (`receipts/20261115T150413Z-source/`).
3. **Step 3 (`67815e5b7`)**: Unconditional `ENGRAM_PACKED_DIR` default (`/cache/spark-ds41f/engram-264-v1`), `EngramDiskRows` `queue_depth = 512`, Chunk-0 Layer-1 prefetch in `DeepseekV41ModelState.add_request`, and callable deferred Layer-14 staging on prefetched hits in `_early_engram_ids` (Boot 1 `receipts/20261115T153720Z-source/`, Boot 2 `receipts/20261115T154402Z-source/`).

## Results (`source` prefill, `--prefill-repeats 2`)

Commands:
```bash
bin/spark --cluster-config experiments/2026-11-14-c58-prefill-10k/tp4-switched-dev.json \
  bench --compare none --suites quality --output <tmp>
bin/spark --cluster-config experiments/2026-11-14-c58-prefill-10k/tp4-switched-dev.json \
  bench --compare none --suites prefill --prefill-text source \
  --prefill-sizes 32768,131072,262144 --prefill-repeats 2 --output <receipt_dir>
```

| Commit / Run | Quality (sha256) | 32,768 tok/s (TTFT) | 131,072 tok/s (TTFT) | 262,144 tok/s (TTFT) |
|---|---|---|---|---|
| `b31a7c98e` (`20261115T140556Z-source`) | 5/5 | 9239.8 (3.145s) | 9696.2 (11.894s) | 9574.3 (25.055s) |
| Step 2 `64e721ef1` (`20261115T150413Z-source`) | 5/5 (`8a177d7fb10287c0`) | 9162.7 ± 2.9% (3.171s) | 9671.1 ± 0.3% (11.924s) | 9556.9 ± 4.1% (25.100s) |
| Step 3 `67815e5b7` Boot 1 (`20261115T153720Z-source`) | 5/5 (`8a177d7fb10287c0`) | 9388.1 ± 2.2% (3.095s) | 9666.1 ± 1.8% (11.931s) | 9632.2 ± 3.5% (24.904s) |
| Step 3 `67815e5b7` Boot 2 (`20261115T154402Z-source`) | 5/5 (`8a177d7fb10287c0`) | 9379.7 ± 3.4% (3.097s) | 9738.0 ± 0.3% (11.842s) | 9625.1 ± 3.8% (24.922s) |
| **Step 3 2-Boot Mean** | **5/5 (`8a177d7fb10287c0`)** | **9383.9 (3.096s)** | **9702.1 (11.887s)** | **9628.7 (24.913s)** |

## Recorded Baseline Context

- **`candidate-55`** (`candidate-55-b9-tp4-switched-screen/prefill.json`, default prompt text, 3 repeats): 32,768 = `8741.6 tok/s`, 262,144 = `8785.3 tok/s`. Initial `c58` switched screen (`receipts/tp4-switched/table.txt`): 32,768 = `8682.8 tok/s` (`-0.67%`, `same`), 262,144 = `8891.3 tok/s` (`+1.21%`, `better`).
- **`1107-switched`** (`manifests/benchmarks/2026-11-07-spark-ds41f-1107-switched.json`): decode suite only (no prefill point in that receipt).

## Memory & Notes

- On Boot 1 (`67815e5b7`), each rank generated packed Engram files in `/cache/spark-ds41f/engram-264-v1/` (`reused=False`, ~13.5 s per layer); on Boot 2, all 4 ranks logged `reused=True written_bytes=0 pack_s=0.000000` (total startup dropped from `+221.6 s` to `+140.5 s`).
- Min `MemAvailable` during Boot 1/Boot 2 `source` prefill: `17.14 GiB` (`sp1`), `19.46-19.50 GiB` (`sp2`), `19.32-19.33 GiB` (`sp3`), `19.59-19.71 GiB` (`sp4`), `swap growth 0.0 GiB`, `thermal slowdown 0.0 ms`.

## Final Candidate-58 (`c58-final`) Qualification & Full Benchmarks

- **Base stack (`1158a12c8` + release-test compatibility guards)**:
  - Profiles: `tp4.json` / `tp4-dev.json` (`dgx1..dgx4`), `tp4-switched.json` / `tp4-switched-dev.json` (`sp1..sp4`), `tp3-switched.json` / `tp3-switched-dev.json` (`sp5..sp7`) (and the remaining `<shape>.json` / `<shape>-dev.json` shape profiles), extending the matching `experiments/2026-10-26-spark-ds41f-candidate/candidate-55/<shape>.json` (`image: spark-ds41f:candidate-55`; `sp8` digest `sha256:c7976f6964156b848cef83009a22d811bb9b74bf23f85ada7e85e814f064d528`, `dgx1` digest `sha256:a8fbd55997cc16b6eb6fa8ffb92be8b05ef739f5eb876e596eccf6b7ce7022db`) and mounting `experiments/2026-11-14-c58-prefill-10k/overlay/vllm/`.
  - Combined optimization stack:
    1. **Cold-prefill lookahead & SP overlap (`c58-prefill-10k` Steps 1–3, `df1ec49fd..67815e5b7`)**: async SP front (`sp_attention_front` + compressor SP local projection + `gather_mxfp8_async` overlapped with `indexer.weights_proj`), SP MoE input local routing (`route_local`) + `gather_many_async(weights, ids)` overlapped with `wait_xq` and shared experts (`16`-pair `(ready_event, done_event)` ring in `sp_prefill.py` and `kernels.py`), unconditional `/cache/spark-ds41f/engram-264-v1` default + `EngramDiskRows` `queue_depth = 512`, Chunk-0 Layer-1 prefetch in `DeepseekV41ModelState.add_request`, and deferred Layer-14 next-chunk Engram prefetch/staging.
    2. **Arm 2 (`sink-wq-screen`, `8bfb3f0de` + `eecd0ea92`)**: certified 8-bit `W_q` Cauchy-Schwarz vocabulary-head screening (`project_head_greedy_pairs` with `sink.cut_cols` suppression and per-row `top_k=4` shard candidates for `sink.alternates_of_greedy_top4`) active under `self.sink.rejection = True`, plus CPU-group TP rank barrier before compressor startup all-gather.
    3. **Arm 3 (`dspark-markov-trunc`, `98d8451d2`)**: DSpark pre-drafting Markov head truncation when adaptive verification caps `num_speculative_tokens` (`0 < k < N`) below `self.num_speculative_steps` (`5`), slicing `sample_hidden` to `(num_reqs * active_steps, hidden)` before `lm_head` and halting the sequential Markov loop at `active_steps`.
    4. **Release-test compatibility guards**: `can_gather_async` / `_DEFAULT_GATHER_MXFP8` / `_DEFAULT_EXCHANGE_INTO` / `model_parallel_is_initialized()` checks in `attention.py`, `compressor.py`, `kernels.py`, and `sp_prefill.py` so unit tests that monkeypatch `SPRows` or `forward_mqa` execute cleanly.

### Release-Tests Qualification (`receipts/release-tests/verdict.json`)

- Full 157-file suite on `sp8` (`spark-0016`, `sha256:c7976f6964156b848cef83009a22d811bb9b74bf23f85ada7e85e814f064d528`):
  - **`10,310 passed, 357 skipped, 0 failed`** (`2,875.6s`, `exit = 0`, `batch_variant = []`, `bits_unequal = []`, `errors = []`).

### Full Cluster Qualification (`receipts/final/`)

#### `tp4-ring` (`dgx1..dgx4`, `receipts/final/tp4-ring/`)

- **`screen-tp4`** (2 boots, `receipts/final/tp4-ring/lab/c58-final-screen-tp4-table.txt`):
  - `grid` (20 shapes): `19.2568 / 19.2344 ms` -> mean **`19.2456 ms`** (`-2.10% ±0.11%` `better` vs `candidate-55` `19.6800 ms`; `-8.67%` vs `1107` `21.0718 ms`).
  - `c16` output: `272.467 / 272.093 tok/s` -> mean **`272.28 tok/s`** (`+0.31% ±0.84%` `same` vs `candidate-55` `271.44`; `-0.21%` vs `1107` `272.86`).
  - `prefill 32,768` (default text): `9,370.51 / 9,236.39` -> mean **`9,303.45 tok/s`** (`+5.10% ±0.71%` `better` vs `candidate-55` `8,851.87`; `+9.35%` vs `1107` `8,507.97`).
  - `prefill 262,144` (default text): `9,559.18 / 9,415.46` -> mean **`9,487.32 tok/s`** (`+6.76% ±0.95%` `better` vs `candidate-55` `8,886.56`; `+14.85%` vs `1107` `8,260.58`).
- **`gate-tp4`** (2 boots, `receipts/final/tp4-ring/lab/c58-final-gate-tp4-table.txt`):
  - All 8 determinism and greedy checks (`determinism`, `greedy`, `determinism2`, `greedy2` across Boot 1 & Boot 2): `identical: true`, `token_diffs: 0`, `logprob_diffs: 0`.
  - Distinct-prompt throughput: `c16-distinct` `273.00 / 274.78` (**`273.89 tok/s`**), `c8-distinct` `203.58 / 203.19` (**`203.39 tok/s`**), `c4-distinct` `158.41 / 157.55` (**`157.98 tok/s`**).
- **`c1-decisions-tp4`** (4 boots, `receipts/final/tp4-ring/lab/c58-final-c1-decisions-tp4-table.txt`):
  - `83.25 / 83.44 / 83.22 / 83.24` -> mean **`83.29 tok/s`** (`35.88 steps/s`, `2.330 tok/step`).
- **`bench` 2-boot `source` prefill** (`tp4-ring-prefill-b1` / `tp4-ring-prefill-b2`, `prompts_sha256 = cc7f29400eff69775f0d9fb016be00924f46a81be7b42dc5c15a4eb9d82da779`):
  - `32,768`: `9,428.91 / 9,387.08` (**`9,408.00 tok/s`**, TTFT `3.081 / 3.095s`, `+10.58%` vs `1107`).
  - `131,072`: `9,781.92 / 9,728.21` (**`9,755.06 tok/s`**, TTFT `11.789 / 11.856s`, `+12.68%` vs `1107`).
  - `262,144`: `9,643.96 / 9,596.78` (**`9,620.37 tok/s`**, TTFT `24.874 / 24.996s`, `+16.46%` vs `1107`).
- **`bench` 2-boot `quality,decode`** (`tp4-ring-full-b1` / `tp4-ring-full-b2`, `5/5` quality pass both boots; workload `cc7f29400eff6977` / `a8703e4bf1f55b23`, `concurrency = [1, 2, 4, 8, 16]`):

| Case | Boot 1 mean (`tok/s`) | Boot 2 mean (`tok/s`) | 2-Boot Mean (`tok/s`) | 2-Boot Window (`tok/s`) | 2-Boot `step_ms` |
|---|---|---|---|---|---|
| `prose-c1` | `79.88` | `79.98` | `79.93` | `84.19` | `26.52 ms` |
| `prose-c2` | `124.34` | `121.27` | `122.80` | `131.29` | `33.55 ms` |
| `prose-c4` | `255.12` | `264.53` | `259.83` | `282.06` | `30.13 ms` |
| `prose-c8` | `420.41` | `446.77` | `433.59` | `469.08` | `36.34 ms` |
| `prose-c16` | `594.01` | `630.71` | `612.36` | `656.19` | `50.44 ms` |
| `code-c1` | `98.63` | `98.51` | `98.57` | `105.56` | `29.25 ms` |
| `code-c2` | `170.05` | `162.57` | `166.31` | `182.25` | `32.81 ms` |
| `code-c4` | `285.54` | `287.00` | `286.27` | `314.75` | `38.36 ms` |
| `code-c8` | `481.11` | `475.92` | `478.51` | `525.79` | `45.31 ms` |
| `code-c16` | `810.65` | `803.14` | `806.89` | `891.22` | `50.23 ms` |

- **Memory & thermal (`tp4-ring`)**: decode min `MemAvailable` `dgx1 23.08–23.43`, `dgx2..dgx4 24.99–25.80 GiB`; prefill min `MemAvailable` `dgx1 19.29–19.30`, `dgx2..dgx4 21.44–22.03 GiB`; `swap_growth_gib = 0.0`, `thermal_slowdown_ms = 0.0`, `power_cap_ms = 0.0`.

#### `tp4-switched` (`sp1..sp4`, `receipts/final/tp4-switched/`)

- **`screen-tp4-switched`** (2 boots, `receipts/final/tp4-switched/lab/c58-final-screen-tp4-switched-table.txt`):
  - `grid` (20 shapes): `19.0817 / 19.0951 ms` -> mean **`19.0884 ms`** (`-2.46% ±0.15%` `better` vs `candidate-55` `19.5046 ms`; `-9.78%` vs `1107` `21.1576 ms`).
  - `c16` output: `268.720 / 268.343 tok/s` -> mean **`268.53 tok/s`** (`-0.29% ±0.84%` `same` vs `candidate-55` `269.32`; `+2.31%` vs `1107` `262.47`).
  - `prefill 32,768` (default text): `9,176.55 / 9,177.15` -> mean **`9,176.85 tok/s`** (`+4.85% ±1.62%` `better` vs `candidate-55` `8,752.64`; `+8.33%` vs `1107` `8,471.55`).
  - `prefill 262,144` (default text): `9,429.71 / 9,390.68` -> mean **`9,410.19 tok/s`** (`+6.98% ±0.93%` `better` vs `candidate-55` `8,795.83`; `+14.63%` vs `1107` `8,209.23`).
- **`gate-tp4-switched`** (2 boots, `receipts/final/tp4-switched/lab/c58-final-gate-tp4-switched-table.txt`):
  - All 8 determinism and greedy checks (`identical: true`, `token_diffs: 0`, `logprob_diffs: 0`).
  - Distinct-prompt throughput: `c16-distinct` `271.23 / 268.34` (**`269.79 tok/s`**), `c8-distinct` `200.32 / 199.86` (**`200.09 tok/s`**), `c4-distinct` `154.91 / 155.97` (**`155.44 tok/s`**).
- **`c1-decisions-tp4-switched`** (4 boots, `receipts/final/tp4-switched/lab/c58-final-c1-decisions-tp4-switched-table.txt`):
  - `82.80 / 82.86 / 82.84 / 82.81` -> mean **`82.83 tok/s`** (`35.72 steps/s`, `2.327 tok/step`).
- **`bench` 2-boot `source` prefill** (`c58-final-prefill-b1` / `b2`):
  - `32,768`: `9,385.90 / 9,423.59` (**`9,404.74 tok/s`**, TTFT `3.096 / 3.083s`, `+11.02%` vs `1107`).
  - `131,072`: `9,742.71 / 9,779.57` (**`9,761.14 tok/s`**, TTFT `11.838 / 11.793s`, `+13.34%` vs `1107`).
  - `262,144`: `9,634.65 / 9,627.49` (**`9,631.07 tok/s`**, TTFT `24.898 / 24.916s`, `+17.32%` vs `1107`).
- **`bench` 2-boot `quality,decode`** (`c58-final-b1` / `b2`, `5/5` quality pass both boots):

| Case | Boot 1 mean (`tok/s`) | Boot 2 mean (`tok/s`) | 2-Boot Mean (`tok/s`) | 2-Boot Window (`tok/s`) | 2-Boot `step_ms` |
|---|---|---|---|---|---|
| `prose-c1` | `79.29` | `79.00` | `79.15` | `83.33` | `26.89 ms` |
| `prose-c2` | `120.12` | `120.10` | `120.11` | `128.28` | `34.25 ms` |
| `prose-c4` | `263.95` | `265.10` | `264.53` | `286.92` | `29.26 ms` |
| `prose-c8` | `425.70` | `401.93` | `413.82` | `446.06` | `38.58 ms` |
| `prose-c16` | `600.11` | `640.13` | `620.12` | `664.71` | `49.81 ms` |
| `code-c1` | `97.81` | `98.39` | `98.10` | `105.06` | `29.58 ms` |
| `code-c2` | `161.21` | `161.13` | `161.17` | `177.28` | `33.63 ms` |
| `code-c4` | `267.91` | `269.71` | `268.81` | `293.38` | `41.58 ms` |
| `code-c8` | `474.21` | `448.57` | `461.39` | `504.03` | `46.94 ms` |
| `code-c16` | `805.03` | `769.76` | `787.40` | `866.39` | `52.01 ms` |

- **Memory & thermal (`tp4-switched`)**: decode min `MemAvailable` `sp1 21.39`, `sp2..sp4 23.75–23.82 GiB`; prefill min `MemAvailable` `sp1 17.10–17.13`, `sp2..sp4 19.33–19.75 GiB`; `swap_growth_gib = 0.0`, `thermal_slowdown_ms = 0.0`, `power_cap_ms = 0.0`.

#### `tp3-switched` (`sp5..sp7`, `receipts/final/tp3-switched/`)

- **`screen-tp3-switched`** (2 boots, `receipts/final/tp3-switched/lab/c58-final-screen-tp3-switched-table.txt`):
  - `grid` (15 shapes): `20.6299 / 20.6452 ms` -> mean **`20.6376 ms`** (`-0.60% ±0.12%` `better` vs `candidate-55` `20.7443 ms`; `-3.13%` vs `1107` `21.3045 ms`).
  - `c8` output: `158.977 / 158.663 tok/s` -> mean **`158.82 tok/s`** (`+0.09% ±1.42%` `same` vs `candidate-55` `158.68`; `+0.97%` vs `1107-tp3-switched` `157.29`).
- **`gate-tp3-switched`** (2 boots, `receipts/final/tp3-switched/lab/c58-final-gate-tp3-switched-table.txt`):
  - All 8 determinism and greedy checks (`identical: true`, `token_diffs: 0`, `logprob_diffs: 0`).
  - Distinct-prompt throughput: `c8-distinct` `155.66 / 155.10` (**`155.38 tok/s`**), `c4-distinct` `118.90 / 119.00` (**`118.95 tok/s`**).
- **`bench` 2-boot `source` prefill** (`c58-final-prefill-b1` / `b2`):
  - `32,768`: `7,019.35 / 7,021.86` (**`7,020.60 tok/s`**, TTFT `4.139 / 4.138s`).
  - `131,072`: `7,219.71 / 7,204.85` (**`7,212.28 tok/s`**, TTFT `15.975 / 16.008s`).
  - `262,144`: `7,106.56 / 7,089.44` (**`7,098.00 tok/s`**, TTFT `33.755 / 33.836s`).
- **`bench` 2-boot `quality,decode`** (`c58-final-b1` / `b2`, `5/5` quality pass both boots):

| Case | Boot 1 mean (`tok/s`) | Boot 2 mean (`tok/s`) | 2-Boot Mean (`tok/s`) | 2-Boot Window (`tok/s`) | 2-Boot `step_ms` |
|---|---|---|---|---|---|
| `prose-c1` | `56.13` | `56.16` | `56.15` | `58.75` | `33.10 ms` |
| `prose-c2` | `98.98` | `103.60` | `101.29` | `108.22` | `35.84 ms` |
| `prose-c4` | `192.40` | `187.87` | `190.14` | `204.43` | `36.44 ms` |
| `prose-c8` | `308.78` | `310.27` | `309.53` | `331.97` | `47.58 ms` |
| `code-c1` | `73.81` | `73.25` | `73.53` | `78.39` | `38.70 ms` |
| `code-c2` | `115.48` | `116.69` | `116.09` | `126.34` | `45.47 ms` |
| `code-c4` | `196.45` | `193.29` | `194.87` | `210.70` | `54.35 ms` |
| `code-c8` | `348.27` | `333.01` | `340.64` | `369.80` | `60.22 ms` |

- **Memory & thermal (`tp3-switched`)**: decode min `MemAvailable` `sp5 6.43–6.48`, `sp6..sp7 8.38–8.58 GiB`; prefill min `MemAvailable` `sp5 4.03–4.14`, `sp6..sp7 6.34–6.54 GiB`; `swap_growth_gib = 0.0`, `thermal_slowdown_ms = 0.0`, `power_cap_ms = 0.0`.
