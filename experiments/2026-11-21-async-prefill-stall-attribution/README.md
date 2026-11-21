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

## Live Serving Runs (`tp3-triangle`, `dgx1..dgx3`)

Tested inside lab window `dgx1-1795263633` on `spark-ds41f:candidate-63-r5` (`experiments/2026-11-21-async-prefill-stall-attribution/tp3-dev.json`) against `ctrl-stall-b1` (`candidate-63/tp3-dev.json`), `0` failed requests on all runs:

- **`model` bit-exactness (`stall-fix-b1`)**: **`128/128` bit-exact** (`outputs_sha256 = b795e6d4c813815a475ab1516faf277a90a86b73614c38f4700189484a8c532d`, `rank1_fraction = 1.0`, `perplexity = 1.3160824`, `failed = 0`).

| Arm | `stall-3` `p90` / `mean` / `probe_p50` (s) | `stall-6` `p90` / `mean` / `long` / `tps` | `stall-6` `probe_ttft_p50` (3 repeats, s) | `stall-6` `probe_max` (s) | `stall-7` `p90` / `long` / `tps` |
| --- | --- | --- | --- | --- | --- |
| `ctrl-stall-b1` | `0.9035 / 0.3546 / 0.684` | `0.9227 / 0.3798 / 0.390 / 4,953` | `1.568` (`1.206 / 1.676 / 1.821`) | `2.129` | `0.9240 / 0.3432 / 5,296` |
| `stall-fix-b1` | `0.8917 / 0.3533 / 0.757` | `0.9105 / 0.3762 / 0.394 / 5,025` | `1.816` (`1.611 / 1.816 / 2.020`) | `2.684` | `0.9111 / 0.3447 / 5,352` |
| `stall-fix-b2` | `0.8988 / 0.3524 / 0.670` | `0.9206 / 0.3814 / 0.395 / 4,980` | `1.699` (`1.187 / 2.118 / 1.790`) | `2.314` | `0.9219 / 0.3463 / 5,314` |
| **2-boot mean** | **`0.8953 / 0.3529 / 0.713`** | **`0.9156 / 0.3788 / 0.395 / 5,003`** | **`1.757`** | **`2.499`** | **`0.9165 / 0.3455 / 5,333`** |

### Conclusion

- **Prediction falsified on `tp3-triangle` `stall-6`:** `stall-6` `probe_ttft_p50` remained bimodal at **`1.757 s`** (vs `1.568 s` `ctrl`, `1.761 s` `1120`). `stall-7`'s exact `long_share = 0.3432` (`22` chunks / `64` steps) proves `K = 2` interleave was already firing on every chunk on `tp3-triangle` (`T_decode ~ 25 ms < 50 ms`). The `1107 -> 1120` `stall-6` regression is instead a **`max_num_seqs = 8` slot-8 occupancy cascade** (6 background streams + 1 long prefill leave 1 slot for probes spaced `3.0 s` apart; `1120`'s higher `adaptive_verification_row_cost = 0.049` trims drafts during prefill-interleaved decode steps, pushing 16-token probe completion past `3.0 s` so Probe `i` holds slot 8 when Probe `i+1` arrives).
- **Retained as a defensive scheduler correctness fix:** The patch is strictly neutral on `tp3-triangle` (`stall-3/6/7` `during_mean_s` and `p90` neutral-to-slightly-faster, `128/128` bit-exact) while eliminating the unit-proven async in-flight stall undercounting bug on high-concurrency / deep-context shapes where `T_decode >= 50 ms` (such as `tp4-ring` `stall-11` and `TP4..TP8` 32-stream profiles). Ready for `candidate-64`; `tp3-triangle` `stall-6` interleaved-draft budget remains an open follow-up.
