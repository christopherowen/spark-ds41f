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

Pending.
