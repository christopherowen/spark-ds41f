# Async In-Flight Prefill Stall Attribution Fix (`TODO.md` Next Release #3)

Date: 2026-11-21. Base commit: `78b4eee10` (`spark-ds41f:candidate-63-r5` / `spark-ds41f-1120` vLLM tree `8fc99ca7546c5c538f0fe64e7c82954f1d2d7ae0`).

## Root Cause of `tp3-triangle` `stall-6` `probe_ttft_p50` Regression (`1.18 s -> 1.76 s`)

In `manifests/suites/2026-11-20-spark-ds41f-1120-tp3/release-table.txt`:
- `stall-3` (3 decode streams + 131K prefill + 500-token probes): `probe_ttft_p50` **improved** `1.073 -> 0.796 s` (Boot 1) and `0.758 s` (Boot 2), while `during_p90_s` dropped `1.034 -> 0.892 s`.
- `stall-6` (6 decode streams + 131K prefill + 500-token probes): `during_p90_s` also dropped `1.057 -> 0.913 s` (`-13.6%`), yet `probe_ttft_p50` **regressed** `1.183 -> 1.761 s` (`+48.9%`, `worse`) on Boot 1 and `1.438 s ± 52.5%` on Boot 2, `probe_ttft_max` rose `1.860 -> 2.442 s`, and `long_share` worsened `0.380 -> 0.393` (`+3.44%`).

### Exact Mechanism (`vllm/v1/core/sched/scheduler.py` + `vllm/v1/engine/core.py`)

Under `--async-scheduling` (`batch_queue_size = 2`), `EngineCore.step_with_batch_queue()` schedules Step `N+1` immediately after **Step `N-1`** completes — while **Step `N`** is still in flight on the GPU.

1. Step `N` schedules Prefill Chunk 1 (`6,096` tokens, `~0.91 s` GPU time, `_last_prefill_step_est_s = 1.35 s`) and records `self._last_prefill_step_at = t0`.
2. `step_with_batch_queue` then pops and blocks on **Step `N-1` (the preceding decode step)**, which completes at `t1 = t0 + T_decode`.
3. Because `self.scheduler.continues_prefill(batch_queue[0][1])` is `True`, `late_admission` skips waiting on Step `N` and immediately invokes `scheduler.schedule()` for Step `N+1` at `t1`.
4. `scheduler.schedule()` computes `wall_dt = t1 - t0 = T_decode` and compares it against `_PREFILL_WALL_CLOCK_MIN_S = 0.05` (`50 ms`):
   - **At 3 streams (`stall-3`)**: `T_decode ≈ 28 ms < 50 ms`, so `wall_dt < 0.05 s` falls back to `_last_prefill_step_est_s = 1.35 s >= 1.0 s`, immediately triggering `interleave_decode_due` (`K = 2` decode steps) after every chunk. Short probes arriving during Chunk 1 are admitted right after Chunk 1 finishes (`probe_ttft_p50 = 0.76–0.80 s`).
   - **At 6 streams (`stall-6`)**: `T_decode ≈ 50–60 ms >= 50 ms` crosses the `_PREFILL_WALL_CLOCK_MIN_S` cliff. The scheduler mistakes the 55 ms prior decode step for the 913 ms prefill chunk's duration (`_prefill_stall_accum_s = 0.055 s < 1.0 s`), leaves `interleave_decode_due = False`, and immediately **pre-queues Chunk 2 (`6,096` tokens, another `0.913 s`) back-to-back behind Chunk 1** in the async batch queue.
   - **Why `1120` regressed harder than `1107`**: On `1107`, Chunk 1 took `1.057 s`, so when Chunk 1 completed, `0.055 + 1.002 = 1.057 s >= 1.0 s` stopped the cascade after 2 chunks (`probe_ttft_p50 = 1.18 s`, `max = 1.86 s`). Because `1120` sped prefill up by 14% (`0.913 s < 1.0 s`), `0.055 + 0.858 = 0.913 s` stays **below `1.0 s`**, allowing even Chunk 3 to be queued back-to-back (`probe_ttft_p50 = 1.76 s`, `max = 2.44 s`), and Boot 2's `T_decode` straddling `50 ms` caused its `±52.5%` bimodal spread.
   - The same bug also affects `tp4-ring` `stall-11` (`T_decode > 50 ms`, `long_share = 0.442` instead of `~0.33`).

## One Causal Variable

Track `self._last_prefill_output_id = id(scheduler_output)` in `Scheduler._update_after_schedule()` when a continuing prefill chunk is scheduled alongside decodes, and clear it in `Scheduler.update_from_output()` when that `scheduler_output` completes. In `Scheduler.schedule()`, when `self.scheduler_config.async_scheduling` is enabled and `self._last_prefill_output_id is not None` (the prefill chunk is still in flight on the GPU), never treat the unrelated `wall_dt` of an earlier step as the prefill chunk's completion time; instead use `max(wall_dt, self._last_prefill_step_est_s)`.

- Zero change when `async_scheduling=False` (synchronous callers) or when the prefill step has already completed (`update_from_output` cleared `_last_prefill_output_id`).
- Zero change to solo cold prefill (`has_scheduled_decode == False`) or pure decode (`has_continuing_prefill_chunk == False`).
- Zero precision, quality, or output changes (100% bit-exact scheduling fix).

## Pre-Run Prediction

- `tp3-triangle` `stall-6`: `probe_ttft_p50` drops from `1.76 s` (Boot 1) / `1.44 s` (Boot 2) to **`~0.80–0.95 s` (`-45%` to `-50%`)**, eliminating the bimodal `±52%` spread; `probe_ttft_max` drops from `2.44 s` to `~1.3–1.6 s`; `long_share` drops from `0.393` toward `~0.34–0.36`.
- `tp3-triangle` `stall-3`: unchanged (`~0.76–0.80 s`).
- Solo cold prefill, `model`, `grid`, `c1`, `c8`: strictly unchanged (0 greedy diffs).

## Unit Verification (`dgx4`, `spark-ds41f:candidate-63-r5`)

Exact command:
```bash
# Unpatched (reproduces the exact 55 ms async double-chunk bug):
docker run --rm --entrypoint "" -v /tmp/p3_stall6:/tmp/p3_stall6 \
  spark-ds41f:candidate-63-r5 \
  bash -c "cd /opt/spark-ds41f/candidate/vllm && pytest --noconftest -q /tmp/p3_stall6/test_async_stall_attribution.py"

# Patched (all 13 existing prefill_yield tests + 2 new async attribution tests):
docker run --rm --entrypoint "" -v /tmp/p3_stall6:/tmp/p3_stall6 \
  -v /tmp/p3_stall6/scheduler.py:/opt/spark-ds41f/candidate/vllm/vllm/v1/core/sched/scheduler.py:ro \
  spark-ds41f:candidate-63-r5 \
  bash -c "cd /opt/spark-ds41f/candidate/vllm && pytest --noconftest -v tests/v1/core/test_prefill_yield.py /tmp/p3_stall6/test_async_stall_attribution.py"
```

Results:
- **Unpatched**: `test_async_inflight_prefill_does_not_mistake_55ms_decode_wait_for_chunk_time` **FAILED**, reproducing the exact double-queued chunk 2 (`{'d0': 1, ..., 'd5': 1, 'long': 1016}`) after a `55 ms` decode wait.
- **Patched**: **`15 passed, 0 failed in 8.45s`** (13/13 existing `test_prefill_yield.py` + 2/2 new async attribution tests).

## Live Screening Status

Pending 2-boot `stall-3` / `stall-6` / `stall-7` screen on `tp3-triangle` (`experiments/2026-11-21-async-prefill-stall-attribution/tp3-dev.json`) once Round 3 completes.
