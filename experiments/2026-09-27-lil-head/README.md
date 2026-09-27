# Rebase onto the Local Inference Lab heads

Base: the promoted configuration on candidate image
`vllm-ds41f-kkref:04c30fa98e79-r5a`: Local Inference Lab
`integration/karmic-kraken-beta` vLLM `04c30fa9` and B12X `e39b437b`, with
vLLM patches 0001 and 0003-0009 (tree `2b59dbc2`) and the switchless
RoCEnante patch (tree `f77d175f`). Compared with the r4a arms of
`experiments/2026-09-26-dspark-policy` (vLLM `01f1b874`, B12X `0f846212`,
patch 0002 on).

## What the rebase changes

- L2 weight prefetch for decode (`VLLM_DS41_L2_PREFETCH`, on by default on
  SM121): side-stream bulk L2 prefetches of the next projections during
  latency-bound windows. Cache hints only; numerics are unchanged. The fill
  budgets (`VLLM_DS41_L2_PREFETCH_{WO,FFN,NEXT}_MB`) were tuned at TP4.
- Sparse-MLA decode metadata built once per KV-cache group instead of per
  layer.
- The base's Engram overlap (`VLLM_DS41_ENGRAM_OVERLAP`, on by default)
  with B12X `run_lookups` replaces patch 0002.
- B12X CuTe compile-cache integrity check (#418).

## Hypotheses

1. The rebased stack is at least as fast as r4a at every decode point, with
   LRU 5/5 and the same memory guards.
2. L2 weight prefetch accounts for a measurable part of any gain
   (`base` against `nol2`, alternating boots).
3. Profiling adaptive verification on distinct tokens (patch 0009) raises
   the modeled cost of extra verification rows toward their real cost, so
   adaptive verification trims weak drafts and throughput rises where
   acceptance is low (prose).
4. Five drafts, the drafter's trained block, beat three once graphs cover
   eight streams (capture to 48 rows). A temperature-0 trace
   (`experiments/2026-09-26-dspark-policy`, `replay.py`) commits about 20%
   more tokens per step at depth 5 than at depth 3 on the same drafts, and
   shows the raw confidences are calibrated within 2 points per position.

## Arms

`make_arms.py` derives both arms from `config/cluster.json`, drops
`--default-chat-template-kwargs` like the r4a arms, and pins cost curves
per arm.

| Arm | Change from `base` |
|---|---|
| `base` | none: base defaults (L2 prefetch and Engram overlap on) |
| `nol2` | `VLLM_DS41_L2_PREFETCH=0` |
| `engramoff` | `VLLM_DS41_ENGRAM_OVERLAP=0`: disk Engram rows read before the launch; own compile cache (the overlap changes the traced graph) |
| `pagedoff` | `VLLM_B12X_PAGED_DECODE=0` |
| `k5` | five drafts, capture sizes to 48; own pinned curves and compile cache |
| `k5real` | `k5` with the distinct-token profile and cost scale 1.0 |
| `realprof` | `SPARK3_DSPARK_PROFILE_TOKENS=random` (patch 0009); own pinned curves |

## Workload and gates

As in `experiments/2026-09-26-dspark-policy`: `sequence.sh` runs base,
nol2, base, nol2, realprof, realprof; `sequence_k5.sh` then runs k5,
k5real, k5, k5real with the LRU gate and the five-case decode matrix at
1/2/4/8 streams. `--allow-mismatch` because `doctor --live` reports only
the nvidia-drm modeset host setting.

## Results

Screening runs (`sequence_lean.sh`, `results/private/bench/lil-s-*`): one
boot per arm, three samples at one and eight streams. Every arm passed LRU
5/5; dgx1's lowest MemAvailable was 5.95-6.58 GiB (r5 base 5.95 GiB, about
0.4 GiB below r4a). Single-stream step times (ms):

| Arm | prose | code | prose-nothink | code-nothink |
|---|---:|---:|---:|---:|
| r4a base (policy b1+b2) | 49.81 | 51.98 | 50.67 | 52.42 |
| r5 `base` | 51.05 | 53.61 | 51.73 | 53.36 |
| r5 `nol2` | 52.07 | 53.46 | 52.25 | 53.69 |
| r5 `realprof` | 51.96 | 53.67 | 52.19 | 53.79 |
| r5 `k5real` | 55.62 | 58.71 | 57.43 | 60.37 |
| r5 `k5` | 53.38 | 58.18 | 55.52 | 59.99 |

- The rebased stack is 1.0-1.6 ms per step slower than r4a (code c8
  -3.5 ±1.7%). L2 prefetch helps it (without it, steps are another 0.5-1 ms
  slower and code c8 is -6.5 ±1.2% against r4a), so the loss lies elsewhere
  in the rebase, most likely the base's Engram overlap replacing patch 0002.
  Not carried forward.
- `realprof` changes nothing: patch 0009 as first written randomized the
  tokens but left dummy rows marked as padding, which the MoE routers skip,
  so its cost curves match the zero-token profile. Patch 0009 now also
  marks the profile rows as real (`experiments/2026-09-27-dspark-depth5`).
- Five drafts: code answers +17.6 ±4.9% (cost scale 1) and +13.7 ±4.5%
  (cost scale 2) at one stream, 3.1-3.3 accepted drafts per step against
  2.24; reasoning and prose within noise (-7 to +11% at wide intervals).
- The disk-Engram path of `base`, `realprof` and the five-draft arms is the
  base's overlap; the determinism arm ran on r4a (`experiments/2026-09-26-
  dspark-policy`).

Isolation runs (`lil-s-nol2b`, `lil-s-baseb`, `lil-s-engramoff`,
`lil-s-pagedoff`), one boot each, pooled with the first pair where
repeated. Single-stream step times (ms):

| Arm | prose | code | prose-nothink | code-nothink |
|---|---:|---:|---:|---:|
| r5 `base` (2 boots) | 51.06 | 53.63 | 51.90 | 53.64 |
| r5 `nol2` (2 boots) | 52.29 | 53.75 | 52.67 | 53.84 |
| r5 `engramoff` | 51.48 | 54.36 | 52.32 | 54.68 |
| r5 `pagedoff` | 51.23 | 53.92 | 52.00 | 53.75 |

- L2 weight prefetch saves 1.2 ms per prose step (-2.4%, significant) and
  0.8 ms per prose-nothink step; code steps and throughput at eight streams
  do not change measurably (code c8 +1.2 ±1.7% pooled).
- The base's Engram overlap saves only 0.4-1.0 ms per step against reading
  the rows before the launch (and 4-6% at eight streams), where patch 0002
  saved about 1.85 ms on r3: most of the rebase's loss. Patch 0002 is
  re-ported in `experiments/2026-09-27-r5-engram-async`.
- The generic B12X paged decode does not touch DeepSeek V4.1 (no change).
