# `tp8-switched`: 1 MiB one-shot all-reduce dispatch cap and dedicated `fast` proxy core

## Hypothesis

On an 8-node switched DGX Spark cluster (`tp8-switched`, MikroTik CRS804 200G RoCE), `sparknet`'s one-shot RoCE all-reduce performs an all-to-all across 7 peers and crosses over NCCL near **1 MiB** (`1,048,576` bytes), whereas `config/cluster.json` ships `SPARKNET_ROCE_ALLREDUCE_DISPATCH_MAX_BYTES=2097152` (2 MiB) and `SPARKNET_ROCE_PROXY_CPU=big` (floating across all 10 Cortex-X925 cores).

For DeepSeek-V4.1-Flash (`hidden_size=5120`, BF16 = `10,240 bytes/row`):
- `1 MiB / 10,240 = 102.4 rows`
- `2 MiB / 10,240 = 204.8 rows`
- All `C1`, `C8`, and `C16` decode batches with `drafts=5` (`<= 16 * 6 = 96 rows = 960 KiB`) stay under 1 MiB and remain on one-shot RoCE.
- High-acceptance `C32` speculative verify steps (`103..192 rows = 1.01..1.88 MiB`) and `(1 MiB, 2 MiB]` prefill chunks fall inside the `(1 MiB, 2 MiB]` band where 8-rank one-shot all-to-all is slower than NCCL.

Lowering `SPARKNET_ROCE_ALLREDUCE_DISPATCH_MAX_BYTES` to `1048576` and setting `SPARKNET_ROCE_PROXY_CPU=fast` (pinning the polling thread to a single dedicated Cortex-X925 core, matching `sparknet/integration/vllm/roce_all_reduce.py`'s default) should improve `C32` decode and medium cold-prefill throughput on `tp8-switched` without regressing `C1`/`C8`/`C16`.

## Intended delta

Two environment overrides on `shapes["tp8-switched"]` (tested via `prod.local.json` against stock `spark-ds41f:1126` at `890f8654`, no image rebuild):

- `SPARKNET_ROCE_ALLREDUCE_DISPATCH_MAX_BYTES`: `"2097152"` -> `"1048576"`
- `SPARKNET_ROCE_PROXY_CPU`: `"big"` -> `"fast"`

*(Note: both variables were changed together in the live serving A/B. The pre-test 8-Spark all-reduce microbenchmark and the C1–C16 vs C32 per-category separation below isolate the 1 MiB crossover mechanism; a separate factorial boot isolating `PROXY_CPU=fast` alone has not yet been run.)*

## Supporting 8-Spark All-Reduce Microbenchmark

Measured on the same 8-Spark CRS804 200G fabric (CUDA-graph captured BF16 all-reduce, `hidden=6144`, `12 KiB/row`, pure-CUDA `sparknet` one-shot vs 4-channel NCCL):

| Rows (M) | Payload | Stock NCCL | `sparknet` One-Shot | Ratio |
|---:|---:|---:|---:|---:|
| 1 | 12 KiB | 223.5 us | 24.5 us | **9.1x faster** |
| 4 | 48 KiB | 230.8 us | 38.1 us | **6.1x faster** |
| 8 | 96 KiB | 239.0 us | 52.0 us | **4.6x faster** |
| 16 | 192 KiB | — | 91.1 us | — |
| 32 | 384 KiB | 298.1 us | 144.8 us | **2.1x faster** |
| 64 | 768 KiB | 340.9 us | 276.1 us | **1.23x faster** |
| 128 | 1.5 MiB | **408.6 us** | 524.5 us | **0.78x (NCCL wins)** |

The crossover lies between 768 KiB and 1.5 MiB, pointing directly at `1048576` (1 MiB) for 8-rank switched fabrics.

## Quality and safety gates

- Back-to-back drained boots on all 8 Sparks (`spark-ds41f:1126`, identical weights/image/hardware, zero external router traffic).
- SM clocks sampled across all 8 nodes throughout both runs (`runs/clocks-base.txt`, `runs/clocks-sn1m.txt`): `2,171–2,190 MHz` on every rank (no GB10 clock-cap latch).
- Zero request failures; live `/health` and completion sanity check (`temperature=0`, exact response verified); `MemAvailable` >= `12.2 GiB` on every rank after boot.

## Workloads & Results

Client: `v41bench` prompt set `v1` (identical across boots, `temperature=0`, thinking off, 8 decode categories + `ceiling_count` at `C1`/`C8`/`C16`/`C32`, plus unique-prefix cold prefill at 2K/8K/32K target tokens). Raw summaries in `runs/bench-base.md` and `runs/bench-sn1m.md`.

### Aggregate & Per-Stream Throughput (8 categories, `ceiling_count` excluded)

| Concurrency | Base agg / stream (TTFT) | Candidate agg / stream (TTFT) | Delta (agg / stream) |
|---|---:|---:|---:|
| **C1** | 155.98 / 181.04 (0.126 s) | **157.22 / 183.69** (0.127 s) | +0.8% / **+1.5%** |
| **C8** | 473.93 / 74.50 (0.329 s) | **478.91 / 75.03** (0.331 s) | **+1.1%** / +0.7% |
| **C16** | 638.25 / 50.66 (0.476 s) | **641.46 / 50.83** (0.474 s) | **+0.5%** / +0.3% |
| **C32** | 842.79 / 32.97 (0.722 s) | **989.00 / 40.26** (0.731 s) | **+17.3% / +22.1%** |

### C32 Per-Category Breakdown (mechanism confirmation)

| Category | Base agg / stream | Candidate agg / stream | Delta (agg / stream) |
|---|---:|---:|---:|
| `coding` | 1,112.30 / 39.82 | **1,465.52 / 55.11** | **+31.8% / +38.4%** |
| `format` | 988.94 / 44.05 | **1,275.94 / 63.17** | **+29.0% / +43.4%** |
| `math` | 1,027.10 / 37.36 | **1,282.81 / 48.03** | **+24.9% / +28.6%** |
| `json` | 837.87 / 36.56 | **956.71 / 42.98** | **+14.2% / +17.6%** |
| `reasoning` | 934.54 / 32.96 | **1,043.26 / 37.51** | **+11.6% / +13.8%** |
| `prose` | 702.23 / 26.45 | **727.50 / 27.32** | +3.6% / +3.3% |
| `narrative` | 600.00 / 21.67 | **620.80 / 22.53** | +3.5% / +4.0% |
| `summary` | 539.38 / 24.86 | 539.48 / 25.46 | +0.0% / +2.4% |
| `ceiling_count` *(excluded from avg)* | 1,424.07 / 48.45 | **2,009.98 / 70.60** | **+41.1% / +45.7%** |

The gain scales directly with draft acceptance (and thus verify batch size): high-acceptance categories that push `C32` batches into `103..192 rows` (`1.0–1.88 MiB`) gain **+25% to +46%**, while low-acceptance `narrative`/`prose` (`<=102 rows`) stay on one-shot RoCE and gain ~3.5%.

### Cold Prefill (unique prefix)

| Target | Prompt tokens | Base (TTFT / tok/s) | Candidate (TTFT / tok/s) | Delta |
|---:|---:|---:|---:|---:|
| 2,000 | 2,950 | 0.492 s / 6,000.9 | **0.434 s / 6,799.9** | **+13.3%** |
| 8,000 | 11,592 | 1.182 s / 9,807.7 | **1.104 s / 10,500.0** | **+7.1%** |
| 32,000 | 46,810 | 4.135 s / 11,320.6 | 4.162 s / 11,247.3 | -0.6% |

## Limits & Promotion Notes

- Single drained boot per arm (back-to-back, same day, verified clocks on all 8 ranks).
- Measured with `v41bench` prompt set `v1` rather than the full `bin/spark bench` release suite.
- Both `SPARKNET_ROCE_ALLREDUCE_DISPATCH_MAX_BYTES=1048576` and `SPARKNET_ROCE_PROXY_CPU=fast` were enabled together; isolating `PROXY_CPU=fast` alone is left for a follow-up factorial run.
- **No files outside `experiments/2026-11-27-tp8-switched-sparknet-dispatch/` are modified** (keeping `config/cluster.json` frozen to `releases/spark-ds41f-1126/cluster.json`).
- **Wiring into a future candidate:** `scripts/recipe.py` forbids shapes from repeating keys declared in top-level `environment` (`recipe.py:600`). To scope `SPARKNET_ROCE_ALLREDUCE_DISPATCH_MAX_BYTES=1048576` to `tp8-switched` (or wider switched shapes), move `SPARKNET_ROCE_ALLREDUCE_DISPATCH_MAX_BYTES` from top-level `environment` to per-shape `environment` (or promote `SPARKNET_ROCE_PROXY_CPU=fast` in top-level `environment`, matching `sparknet/integration/vllm/roce_all_reduce.py:116`'s own default). Operators on `spark-ds41f-1126` can reproduce immediately via `prod.local.json`:

```json
{
  "environment": {
    "SPARKNET_ROCE_ALLREDUCE_DISPATCH_MAX_BYTES": "1048576",
    "SPARKNET_ROCE_PROXY_CPU": "fast"
  }
}
```
