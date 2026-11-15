# c58-prefill-10k

Targeted prefill optimizations on `tp4-switched-c58-dev.json` (`sp1..sp4`, 4-node switched RoCE TP4, context 1,048,576, `VLLM_DS41_LSA=0`, image `spark-ds41f:candidate-55`).

## Intended Variables & Sequence

1. **`b31a7c98e`**: Defer next-chunk Engram prefetch until Layer 14 (`receipts/20261115T140556Z-source/`).
2. **Step 2 (`64e721ef1`)**: Event-driven 16-pair `(ready_event, done_event)` ring (`sp_prefill.py`, `kernels.py`) overlapping `gather_many_async` with `wait_xq` (`receipts/20261115T150413Z-source/`).
3. **Step 3 (`67815e5b7`)**: Unconditional `ENGRAM_PACKED_DIR` default (`/cache/spark-ds41f/engram-264-v1`), `EngramDiskRows` `queue_depth = 512`, Chunk-0 Layer-1 prefetch in `DeepseekV41ModelState.add_request`, and callable deferred Layer-14 staging on prefetched hits in `_early_engram_ids` (Boot 1 `receipts/20261115T153720Z-source/`, Boot 2 `receipts/20261115T154402Z-source/`).

## Results (`source` prefill, `--prefill-repeats 2`)

Commands:
```bash
bin/spark --cluster-config experiments/2026-11-14-c58-prefill-10k/tp4-switched-c58-dev.json \
  bench --compare none --suites quality --output <tmp>
bin/spark --cluster-config experiments/2026-11-14-c58-prefill-10k/tp4-switched-c58-dev.json \
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
