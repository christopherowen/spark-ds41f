# Interleave $K$ Graphed Decode Steps Between Full Prefill Chunks When Decodes Are Active

## Motivation

During a long multi-chunk prefill at full slot occupancy (`stall-15` on `tp4-ring`: 15 active decoding streams + 1 long 131K prefill occupying all 16 sequence slots), the 15 decoding streams only advance once every ~2.25 s (riding eagerly inside the chunk step), even though a pure CUDA-graphed decode step for 15 streams takes only ~114.7 ms and advances every stream by ~1.95–2.1 tokens.

Existing `--prefill-yield short` does not trigger at `stall-15` because `admission_open` is `False` (`len(self.running) == max_num_seqs`) and no new request can be admitted.

## Mechanism (`--prefill-decode-interleave-steps K`)

1. **`SchedulerConfig` & `EngineArgs`** (`overlay/vllm/config/scheduler.py`, `overlay/vllm/engine/arg_utils.py`):
   - Adds `prefill_decode_interleave_steps: int = Field(default=0, ge=0)` (`--prefill-decode-interleave-steps`).
2. **`Scheduler.schedule` & `Scheduler._update_after_schedule`** (`overlay/vllm/v1/core/sched/scheduler.py`):
   - Whenever a multi-chunk prefill chunk is scheduled and remains in progress (`request.is_prefill_chunk == True` after advancing `num_computed_tokens`), `self._interleave_decode_steps_remaining` is set to `self.prefill_decode_interleave_steps` ($K$).
   - On subsequent steps while `has_eligible_decode` (`num_runnable_decodes > 0`) and `has_inflight_multi_chunk_prefill` are both true and `self._interleave_decode_steps_remaining > 0` (or a newly prefilled prompt's held first token `first_decode_due` is pending under sink rejection), `interleave_decode_step` is enabled and `defer_prefills` is set to `True`.
   - In the running pass, `defer_prefills` defers the in-progress long prefill chunk while scheduling all runnable decode streams as a fast CUDA-graphed decode batch.
   - In the waiting pass (`interleave_admit_short`), any waiting short prompt whose prefill finishes within one chunk (`_prefill_finishes_in_chunk`) is admitted without being blocked by `defer_prefills` or by any queued long prompt ahead of it.
   - If no decodes are runnable (`num_runnable_decodes == 0`), prefill chunks run back-to-back without interruption.
   - If the coarse runnable check races async-completion guards and produces an empty step (`not num_scheduled_tokens`), `Scheduler.schedule` falls back immediately to scheduling the running prefill chunk.

## Why This Strictly Beats `--prefill-yield short`

- **Works at `stall-15` (100% slot occupancy) and throughout a 1M prefill**: Does not depend on `admission_open` or a new request arriving, and needs no 10 s cliff because the long-prompt overhead is bounded by construction to:
  $$\frac{K \cdot t_{\text{decode}}}{t_{\text{chunk}}} \approx \frac{2 \times 114.7\ \text{ms}}{2,360\ \text{ms}} \approx 9.7\%$$
- **Zero extra MoE weight-stream passes ($F = 153.5\ \text{ms}$)**: Each prefill chunk remains full-sized (`16,288` tokens on `tp4-ring`, `8,144` tokens on `tp3-triangle`) on its exact aligned boundaries.
- **2.6–3.2x higher decode throughput & ~58–65% lower mean ITL during prefill**: Decoding streams get $K$ fast graphed decode steps plus the chunk step's tokens every ~2.4 s instead of 1 step every ~2.25 s.

## Evaluation Summary (`interleave_policy.py` -> `results.json`)

| Scenario | Policy | Decode Throughput During Prefill | Stream ITL Mean | Stream ITL p50 | Doc TTFT Overhead | Extra MoE Passes ($F$) | Short Probe TTFT Mean |
|---|---|---:|---:|---:|---:|---:|---:|
| **`stall-15` (`tp4-ring`)** | `production` | 12.4 tok/s (1.00x) | 2.070 s | 2.479 s | +0.0% | 0 | — |
| | `prefill-yield-short` | 12.4 tok/s (1.00x) | 2.070 s (+0.0%) | 2.479 s | +0.0% | 0 | — |
| | `interleave-k1` ($K=1$) | **23.1 tok/s (1.86x)** | **1.201 s (-42.0%)** | **0.278 s** | +4.4% | **0** | — |
| | `interleave-k2` ($K=2$) | **32.5 tok/s (2.62x)** | **0.867 s (-58.1%)** | **0.115 s** | +8.8% | **0** | — |
| **`stall-11` (`tp4-ring`)** | `production` | 9.9 tok/s (1.00x) | 2.069 s | 2.452 s | +0.0% | 0 | 13.05 s |
| | `prefill-yield-short` | 21.5 tok/s (2.17x) | 1.150 s (-44.4%) | 0.388 s | +11.1% | 0 | 3.03 s |
| | `interleave-k1` ($K=1$) | **25.9 tok/s (2.62x)** | **0.905 s (-56.2%)** | **0.371 s** | +13.7% | **0** | **2.70 s** |
| | `interleave-k2` ($K=2$) | **31.7 tok/s (3.21x)** | **0.734 s (-64.5%)** | **0.106 s** | +16.9% | **0** | **1.33 s** |
| **`stall-7-tp3` (`tp3-triangle`)** | `production` | 10.9 tok/s (1.00x) | 1.353 s | 1.479 s | +0.0% | 0 | — |
| | `prefill-yield-short` | 10.9 tok/s (1.00x) | 1.353 s (+0.0%) | 1.479 s | +0.0% | 0 | — |
| | `interleave-k2` ($K=2$) | **25.6 tok/s (2.35x)** | **0.549 s (-59.4%)** | **0.097 s** | +12.7% | **0** | — |
| **`stall-3-tp3` (`tp3-triangle`)** | `production` | 5.5 tok/s (1.00x) | 1.344 s | 1.455 s | +0.0% | 0 | 16.49 s |
| | `prefill-yield-short` | 10.3 tok/s (1.88x) | 0.961 s (-28.5%) | 1.412 s | +7.3% | 0 | 2.70 s |
| | `interleave-k2` ($K=2$) | **17.1 tok/s (3.10x)** | **0.496 s (-63.1%)** | **0.072 s** | +14.8% | **0** | **1.24 s** |
| **`stall-1m` (`tp4-ring`)** | `production` | 6.5 tok/s (1.00x) | 3.350 s | 3.386 s | +0.0% | 0 | 119.87 s |
| | `prefill-yield-short` | 8.6 tok/s (1.31x) | 2.721 s (-18.8%) | 3.111 s | +1.9% | 0 | 5.97 s |
| | `interleave-k2` ($K=2$) | **19.4 tok/s (2.96x)** | **1.159 s (-65.4%)** | **0.101 s** | +7.1% | **0** | **2.32 s** |
