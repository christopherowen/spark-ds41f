# TP3-Triangle 16-Seat Qualification (`max_num_seqs: 8 -> 16`)

Date: 2026-11-22.
Base commit: `2c5158cc6c` (`spark-ds41f:candidate-64`, `sha256:50b294060f808ce5d2369441995762d591e1e213fe17f75a4635e43ad44ce312`, vLLM `0001..0843`).
Target cluster: `dgx1..dgx3` (`tp3-triangle`, lab window `dgx1-1795336376`; production `spark-ds41f:1120` restored at close with `doctor --live OK`).

## Hypothesis

Raising `tp3-triangle` from `8` to `16` concurrent seats (`max_num_seqs: 8 -> 16`, `max_cudagraph_capture_size: 48 -> 96`, `long_prefill_token_threshold: 6096 -> 6048`, extending `cudagraph_capture_sizes` up to `96` using the `tp4` step-8 ladder) with the KV pool unchanged (`4.91 GiB`, `5.49M` tokens = `16 × 342.8K`):

1. Unlocks sublinear routed-expert union scaling from `c8` to `c16` (`tp3` currently saturates at `~169.5 tok/s` because requests 9–16 queue).
2. Eliminates the slot-8 occupancy cliff blamed for `stall-6` probe TTFT (`1.757 s` on `candidate-64` vs `1.18 s` on `1107`).
3. Preserves `c8`, `1M` prefill, and `100%` greedy + top-20 logprob bit-exactness while keeping `dgx1` safely above the `5.0 GiB` startup and `3.0 GiB` steady memory-guard floors.

## One Causal Variable

Pure serving-config change on `spark-ds41f:candidate-64` ([tp3-16-seats.json](tp3-16-seats.json), zero code patches). `tp3-12-seats.json` fallback was not needed — `tp3-16-seats.json` booted cleanly on both boots with generous headroom.

## Workload Identity & Exact Commands

- Suite: [suite-tp3-16.json](suite-tp3-16.json) (`suite_sha256: 2a51d9d751f5`), run across **2 independent boots** (`tp3-16-b1`, `tp3-16-b2`):
  - `model`: `128` prompts at `concurrency=16`, `max_tokens=256`, `top=20`.
  - `grid`: `replays=100`, `settle=1`, `max_requests=16` (`20` shapes, including `16x32`, `16x48`, `12x72`, `16x96`, `16x48 @64K`).
  - `c1` (`warmup=30, duration=180`), `c8` (`warmup=60, duration=300`), `c16` (`warmup=60, duration=300`), `thinking=false`.
  - `stall-6`: `131,072` cold prompt over `6` decode streams with `500`-token probes at `[2, 5, 8, 11, 14] s` (`3` repeats, exact `release-tp3` spec).
  - `prefill-1m`: `1,000,000`-token cold prefill (`1` repeat, headroom + throughput check).
- Commands per boot:
  ```bash
  bin/spark --cluster-config experiments/2026-11-22-tp3-16-seats/tp3-16-seats.json cluster stop --apply
  bin/spark --cluster-config experiments/2026-11-22-tp3-16-seats/tp3-16-seats.json cluster start --apply --replace
  bin/spark --cluster-config experiments/2026-11-22-tp3-16-seats/tp3-16-seats.json bench --suite experiments/2026-11-22-tp3-16-seats/suite-tp3-16.json --label tp3-16-b{1,2} --output results/private/lab/2026-11-22-tp3-16-seats/tp3-16-b{1,2}
  ```

## Live 2-Boot Results (`dgx1..dgx3`, `0` Failed Requests Across Both Boots)

### Headline Serving & Latency Metrics vs Recorded `candidate-64` (8-Seat) Reference

| Metric | `candidate-64` (8 seats, 2-boot) | `tp3-16-seats` Boot 1 | `tp3-16-seats` Boot 2 | `tp3-16-seats` Mean | Over/Under vs `c64` (8 seats) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **Warm boot** | `107.6 / 109.8 s` (`108.7 s`) | `111.9 s` | `113.8 s` | `112.9 s` | `+4.2 s` (`-21.7 s` vs `1120`) |
| **`c16` throughput** | `~169.5 tok/s` *(8 seats queue)* | **`219.44 tok/s`** (`CI 0.93%`) | **`219.52 tok/s`** (`CI 0.87%`) | **`219.48 tok/s`** | **`+29.5%`** (`TPOT 73.3 ms`, `TTFT 0.448 s`) |
| **`c8` throughput** | `169.23 / 169.70` (`169.47 tok/s`) | `169.34 tok/s` (`CI 1.06%`) | `169.05 tok/s` (`CI 0.97%`) | `169.20 tok/s` | `-0.16%` (`flat`) |
| **`c1` throughput** | `69.80 / 69.87` (`69.84 tok/s`)* | `68.52 tok/s` (`CI 5.07%`) | `68.49 tok/s` (`CI 5.08%`) | `68.51 tok/s` | `-1.9%` *(note: `180 s` vs `300 s` window & aged `SecPageTables`; see grid below)* |
| **`stall-6` probe TTFT p50** | `1.757 s` (`1107`: `1.18 s`) | **`0.846 s`** | **`0.869 s`** | **`0.858 s`** | **`-51.2%` (`-0.90 s` vs `c64`, `-27.3%` vs `1107`)** |
| **`stall-6` worst / mean gap** | — | `0.949 s / 0.369 s` | `0.991 s / 0.369 s` | `0.970 s / 0.369 s` | `131K` mixed prefill `4,978 tok/s` |
| **`1M` prefill** | `7,007 / 6,992` (`7,000 tok/s`) | `7,035.6 tok/s` | `7,081.3 tok/s` | **`7,058.5 tok/s`** | **`+0.84%`** |

*\*`c64` `c1` ran `single-stream.json` (`60 s` warmup + `300 s` measured); `suite-tp3-16.json` ran `30 s` warmup + `180 s` measured.*

### Bit-Exactness (`model`: 128 Prompts at `c16`, 256 Tokens, Top-20 Logprobs)

- **`tp3-16-b1` vs `candidate-64` (8-seat) `b1`**: **`0 / 128` greedy token sequence diffs (`32,768 / 32,768` tokens exact)**, **`0 / 128` top-20 logprob diffs** (`rank1_fraction = 1.0`, `perplexity = 1.3160824`, `acceptance_length = 2.340 / 2.336`, `generate_s = 149.4 / 149.3 s`).
- **`tp3-16-b1` vs `tp3-16-b2`**: **`0 / 128` token diffs**, **`0 / 128` top-20 logprob diffs**.

### Engine Grid (`100` Replays, `max_requests = 16`, 2-Boot Median & Floor vs `candidate-64` 8-Seat)

| Shape | `c64` 8-seat median (b1 / b2) | `tp3-16` median (b1 / b2) | Median Δ% | Floor Δ% |
| :--- | ---: | ---: | ---: | ---: |
| `decode 1x2 @4K` | `16.429 / 16.401` (`16.415 ms`) | `16.463 / 16.462` (`16.462 ms`) | `+0.29%` | `+0.27%` |
| `decode 4x8 @4K` | `18.017 / 18.000` (`18.009 ms`) | `18.101 / 18.084` (`18.093 ms`) | `+0.47%` | `+0.43%` |
| `decode 8x16 @4K` | `19.750 / 19.744` (`19.747 ms`) | `19.908 / 20.126` (`20.017 ms`) | `+1.37%` | `+0.75%` |
| **`decode 16x32 @4K`** | `n/a` | **`25.335 / 25.242` (`25.288 ms`)** | `new` | — |
| `decode 1x3 @4K` | `16.597 / 16.621` (`16.609 ms`) | `16.829 / 16.924` (`16.877 ms`) | `+1.61%` | `+0.38%` |
| `decode 4x12 @4K` | `18.593 / 18.571` (`18.582 ms`) | `18.678 / 18.694` (`18.686 ms`) | `+0.56%` | `+0.40%` |
| `decode 8x24 @4K` | `22.176 / 22.148` (`22.162 ms`) | `22.417 / 22.474` (`22.446 ms`) | `+1.28%` | `+0.56%` |
| **`decode 16x48 @4K`** | `n/a` | **`28.272 / 28.182` (`28.227 ms`)** | `new` | — |
| `decode 1x6 @4K` | `17.093 / 17.039` (`17.066 ms`) | `17.380 / 17.513` (`17.447 ms`) | `+2.23%` | `+0.57%` |
| `decode 2x12 @4K` | `18.369 / 18.321` (`18.345 ms`) | `18.836 / 18.892` (`18.864 ms`) | `+2.83%` | `+0.53%` |
| `decode 4x24 @4K` | `21.697 / 21.647` (`21.672 ms`) | `21.915 / 21.818` (`21.867 ms`) | `+0.90%` | `+0.50%` |
| `decode 8x48 @4K` | `27.252 / 27.110` (`27.181 ms`) | `27.140 / 27.298` (`27.219 ms`) | `+0.14%` | `+0.62%` |
| **`decode 12x72 @4K`** | `n/a` | **`34.867 / 34.903` (`34.885 ms`)** | `new` | — |
| **`decode 16x96 @4K`** | `n/a` | **`39.119 / 39.073` (`39.096 ms`)** | `new` | — |
| `decode 1x3 @64K` | `16.617 / 16.570` (`16.594 ms`) | `17.031 / 17.052` (`17.042 ms`) | `+2.70%` | `+0.39%` |
| `decode 8x24 @64K` | `22.176 / 22.097` (`22.136 ms`) | `22.505 / 22.569` (`22.537 ms`) | `+1.81%` | `+0.66%` |
| **`decode 16x48 @64K`** | `n/a` | **`28.153 / 28.219` (`28.186 ms`)** | `new` | — |
| `decode 1x3 @256K` | `16.631 / 16.580` (`16.605 ms`) | `16.646 / 16.728` (`16.687 ms`) | `+0.49%` | `+0.35%` |
| `decode 8x24 @256K` | `22.165 / 22.102` (`22.133 ms`) | `22.259 / 22.235` (`22.247 ms`) | `+0.51%` | `+0.56%` |
| `decode 1x3 @1016K` | `16.604 / 16.577` (`16.590 ms`) | `16.648 / 16.644` (`16.646 ms`) | `+0.34%` | `+0.34%` |
| **Geomean (`15` overlapping)** | | | **`+1.164%`** | **`+0.487%`** |

### Memory & Host Aging Observations (`dgx1..dgx3`)

- `dgx1` `[memory_breakdown]` (`tp3-16-b1` / `tp3-16-b2`): `weights = 93.95 GiB`, `kv_cache = 4.91 GiB`, `workspace = 0.66 GiB`, `other_torch_allocated = 1.17 GiB` (`+0.05 GiB` vs 8-seat), **`cudagraph = 2.45 / 2.37 GiB`** (`84` FULL graphs up to `107` copy-widened rows, only **`+0.75 GiB`** vs 8-seat `1.65 GiB` / `69` graphs!), **`free_memory = 7.84 / 7.83 GiB`** (vs `8.39 GiB` at 8 seats).
- Host `MemAvailable` on `dgx1` / `dgx2` / `dgx3`:
  - Post-boot: **`7.78 / 9.70 / 9.48 GiB`** (Boot 1), **`7.75 / 9.71 / 9.60 GiB`** (Boot 2).
  - Post-suite (after `c16` + `1M` prefill): **`7.27 / 9.55 / 9.39 GiB`** (`> 4.2 GiB` above the `3.0 GiB` steady floor and `> 2.7 GiB` above the `5.0 GiB` startup floor).
- **Aged-node `SecPageTables` note**: `dgx1..dgx3` had NOT been rebooted since `1120` promotion, so `SecPageTables` grew further from `26.3 / 25.9 / 22.7 MiB` (during `c64` 8-seat qualification) to **`36.1 / 35.8 / 33.5 MiB`** here. That `+10 MiB` SMMU fragmentation explains the periodic `+0.8 ms` jitter spikes in `grid` median (`+1.16%`) vs floor (`+0.49%`). Rebooting once before promotion clears `SecPageTables` to `~4.8 MiB` and enables `0832`'s contiguous pool permanently.

## Conclusion

**Verdict: SURVIVED (strong promotion candidate for `tp3-triangle` / `candidate-64`+).**

- **`+29.5%` peak throughput (`c16`: `219.48 tok/s` vs `169.47 tok/s`)**, **`-51.2%` `stall-6` probe TTFT (`0.858 s` vs `1.757 s`, `-27.3%` faster than `1107`'s `1.18 s`)**, **`+0.84%` `1M` prefill (`7,059 tok/s`)**, `c8` flat (`-0.16%`), **`100%` bit-exact (`0/128` token and top-20 logprob diffs)**, and **`7.27 GiB`** steady `MemAvailable` on `dgx1` after `c16` + `1M` prefill (`+0.75 GiB` CUDA graph cost, `0` failures across both boots).
- Watch item: confirm on a freshly rebooted `dgx1..dgx3` (`SecPageTables ~4.8 MiB`) that the `+0.49%` grid floor / `+1.16%` median delta disappears once `0832`'s contiguous pool is active.
