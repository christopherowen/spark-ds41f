# Remaining-token (`max_tokens`) decode-row and copy-placeholder clamping (`2026-11-08-remaining-token-draft-cap`)

- **Date**: 2026-11-08
- **Base deployment commit**: `cc20ed18b6ca6521b602f09cb3ba326774166d1c` (Candidate `1107`, `experiments/2026-10-26-spark-ds41f-candidate/1107`)
- **Base vLLM commit**: `320021e3c4dbb202d911ab70e2cd924c75420488` (tree `ed1bdb7a7684da33d403a055521ac0c4dd7ca670`, `0001..0761`)
- **Candidate vLLM commit**: `8eaa8d41d7b02fea721ad7d673abb72a02704390` (tree `ec992c8434aabfc5cc7b63d4f016109004ea723c`, `0001..0762`, `patchset_sha256` `9b73e7748ba02e4c9a9913b685516538c5c717900260f7dc1d58b5ca1c01fff7`)
- **Intended causal variable**: Clamp scheduled speculative decode verify rows and copy-placeholder widening to each request's remaining `max_tokens` budget (`0762-perf-spec-decode-remaining-token-draft-cap.patch`).

## 1. Technical Audit & Motivation

When a bounded-generation request (`max_tokens` set) approaches completion during speculative decoding (`DSpark` `K = 5` plus `CopyDrafter` widened up to `16` draft placeholders), three stages interact at the tail of generation:

1. **Async-scheduling step before completion (`min_computed_tokens < max_model_len`, `rem` output tokens remaining before `max_tokens`)**:
   - Under `--async-scheduling`, `Scheduler.schedule()` previously checked `if min_computed_tokens >= max_tokens_len: num_new_tokens = 0`, which skips scheduling once the minimum guaranteed progress of the in-flight step already reaches `max_tokens`.
   - However, on the step *immediately preceding* that threshold—when `rem = num_prompt_tokens + max_tokens - min_computed_tokens` satisfies `1 <= rem < 1 + len(spec_token_ids)`—the scheduler still scheduled the full `1 + len(spec_token_ids)` tokens (`6` verify rows for DSpark, or up to `17` verify rows for a copy-widened request).
   - Every verify row beyond the remaining token budget allocates KV cache slots, enters `AdaptiveVerificationManager.get_num_tokens()` as eligible draft capacity, and executes target MLA attention, mHC, and routed/shared MoE rows whose sampled tokens `check_stop()` immediately discards (`del request.output_token_ids[max_tokens:]`).
2. **Copy-placeholder widening (`_widen_copy_placeholders`)**:
   - In multi-stream decode (`c4`, `c8`, `c16`), `_widen_copy_placeholders` enforces a batch-wide envelope on extra copy rows (`total_extra <= _copy_max_extra = 11`).
   - A copying request with only `rem = 2` tokens left before `max_tokens` previously widened its placeholder list to `16` (`11` extra rows), consuming the entire batch's `_copy_max_extra` envelope on `15` rows past `max_tokens` and starving other copying requests in the same batch down to `5` drafts.
3. **Adaptive Verification (`AdaptiveVerificationManager`) & DSpark Drafter (`DSparkSpeculator.propose`)**:
   - `AdaptiveVerificationManager.get_num_tokens()` derives `valid_drafts[req_id]` directly from `min(num_tokens - 1, len(draft_tokens), num_speculative_steps)`. Because `SchedulerOutput` is broadcast identically from rank 0 to all tensor-parallel ranks, clamping `scheduled_spec_decode_tokens[req_id]` and `num_scheduled_tokens[req_id]` in `Scheduler.schedule()` automatically caps `valid_drafts[req_id]` identically across all TP ranks with zero cross-rank divergence.
   - When `check_stop()` trims a request at `max_tokens` in `update_from_output`, patch `0624` (`skip_spec_draft_req_ids`) already marks `req_id` in `finished_req_ids`, causing `model_runner._bookkeep_sync` to pass `skip_spec_draft=True` when all requests in the step finish (skipping all 5 DSpark draft steps).

### Sink Rejection Invariant (`VLLM_SINK_REJECTION=1`)

- When `sink_rejection` is **off**, the `max_tokens`-th output token (`output_token_ids[max_tokens - 1]`) is sampled from the target logits at KV position `num_prompt_tokens + max_tokens - 2`. KV position `num_prompt_tokens + max_tokens - 1` only computes logits for the `(max_tokens + 1)`-th output token, which `check_stop` always discards. Thus the exact row cap is:
  $$\text{max\_decode\_new\_tokens} = \text{num\_prompt\_tokens} + \text{max\_tokens} - \text{min\_computed\_tokens} - \text{num\_sampled\_tokens\_per\_step}$$
- When `sink_rejection` is **on**, the sink detector inspects the residual column of the row where a draft token is the *input*. For the `max_tokens`-th output token (which enters the step as a draft token under async scheduling), its input sits at KV position `num_prompt_tokens + max_tokens - 1`. Retaining that one additional row:
  $$\text{max\_decode\_new\_tokens} = \text{num\_prompt\_tokens} + \text{max\_tokens} - \text{min\_computed\_tokens}$$
  ensures the `max_tokens`-th output token still undergoes sink verification (and is cut/recovered if sunk) while trimming all draft rows strictly past `max_tokens`.

## 2. Implementation (`0762-perf-spec-decode-remaining-token-draft-cap.patch`)

In `vllm/v1/core/sched/scheduler.py`:
- Added `_max_decode_new_tokens(self, request: Request) -> int | None`, gated on `self.enable_adaptive_verification` or `self._copy_max_draft_tokens > 0` when `request.max_tokens is not None`.
- In `_widen_copy_placeholders()`, clamped each request's widened draft count `alloc` to `max_new_tokens - 1` before deducting from the batch-wide `_copy_max_extra` envelope.
- In `Scheduler.schedule()`, clamped `num_new_tokens` to `_max_decode_new_tokens(request)` on decode steps and trimmed `curr_req_spec_token_ids` to `num_new_tokens - 1` (or cleared `spec_token_ids` when `num_new_tokens == 0`).

## 3. Exact Commands & Workload Identity

Executed on `sp8` (`NVIDIA GB10` `sm_121a`, container `spark-ds41f:1107`, locked with `flock /tmp/gpu.lock`) across two independent container boots (`boot1` and `boot2`):

```bash
flock /tmp/gpu.lock bash -c '
set -euo pipefail
for boot in boot1 boot2; do
  docker run --rm --gpus all \
    -v /tmp/rem-tokens-test/scheduler.py:/opt/spark-ds41f/candidate/vllm/vllm/v1/core/sched/scheduler.py:ro \
    -v /tmp/vllm-rem-tokens/tests:/tmp/tests:ro \
    -e PYTHONPATH=/opt/spark-ds41f/candidate/vllm:/tmp \
    --entrypoint pytest \
    spark-ds41f:1107 --noconftest -o addopts="" -q \
      /tmp/tests/v1/core/test_copy_drafts_scheduler.py \
      /tmp/tests/v1/core/test_sink_draft_zero.py \
      /tmp/tests/v1/spec_decode/test_adaptive_verification.py \
      /tmp/tests/v1/spec_decode/test_acceptance_estimator.py \
    | tee /tmp/rem-tokens-test/receipts/gpu-pytest-${boot}.txt
  docker run --rm --gpus all \
    -v /tmp/rem-tokens-test/scheduler.py:/opt/spark-ds41f/candidate/vllm/vllm/v1/core/sched/scheduler.py:ro \
    -v /tmp/vllm-rem-tokens/tests:/tmp/tests:ro \
    -v /tmp/rem-tokens-test/screen_remaining_token_cap.py:/tmp/screen_remaining_token_cap.py:ro \
    -e PYTHONPATH=/opt/spark-ds41f/candidate/vllm:/tmp \
    --entrypoint python3 \
    spark-ds41f:1107 /tmp/screen_remaining_token_cap.py \
    | tee /tmp/rem-tokens-test/receipts/screen-remaining-token-cap-${boot}.jsonl
done
'
```

## 4. Two-Boot GPU Verification & Screening Results (`sp8`, `NVIDIA GB10`)

### 4.1 Unit & Integration Test Suites (`boot1` & `boot2`)

- `receipts/gpu-pytest-boot1.txt`: **`136 passed, 1 skipped in 23.70s`**
- `receipts/gpu-pytest-boot2.txt`: **`136 passed, 1 skipped in 24.19s`**
- `receipts/screen-remaining-token-cap-boot1.jsonl`
- `receipts/screen-remaining-token-cap-boot2.jsonl`

### 4.2 Functional, Multi-Stream Envelope Reclaim & Bit-Exactness Screen (`boot1` & `boot2`)

| Screen Case | `boot1` | `boot2` | Result |
| :--- | :---: | :---: | :--- |
| `single_stream_copy_tail_rem_{2,3,4,8,12}` (`unclamped_rows -> clamped_rows`) | `17 -> {2,3,4,8,12}` | `17 -> {2,3,4,8,12}` | Saves `15, 14, 13, 9, 5` wasted target verify rows on copy-widened tail steps |
| `multi_stream_copy_envelope_reclaim_c4` (`r0` drafts, `r1` drafts) | `r0: 16->1`, `r1: 5->16` (`+11`) | `r0: 16->1`, `r1: 5->16` (`+11`) | Reclaims `_copy_max_extra` envelope from finishing stream so active copying stream widens `5 -> 16` |
| `gpu_adaptive_budget_c8_tail_reclaim` (`total_rows`, `wasted_drafts`) | `48 -> 40` rows (`8 -> 0` wasted) | `48 -> 40` rows (`8 -> 0` wasted) | Real GPU `TileLang` `assign_draft_budget` + `adaptive_offsets` drops 8 dead tail rows while keeping all 32 live drafts |
| `sink_draft_zero_bit_exact_sweep` (`20` configs, `max_tokens=1..32`, sync+async) | `all_bit_exact: true` (`730 -> 671` rows) | `all_bit_exact: true` (`730 -> 671` rows) | 100% bit-exact output tokens and sink metrics across all 20 configurations; saves `59` verify rows |

### 4.3 Two-Boot 61-Layer `NVIDIA GB10` (`sm_121a`) Tail Step Latency (`TP3` & `TP4`)

100 timed 61-layer `DeepSeek-V4.1-Flash` per-rank decode verify steps (`q_a + kv_a + wq_b + wo + w_gate_up + w_down`, `BF16` on `NVIDIA GB10`) per boot:

| Topology | Tail Scenario (`Unclamped -> Clamped` rows) | `boot1` Unclamped (ms) | `boot2` Unclamped (ms) | 2-Boot Unclamped Mean (ms) | `boot1` Clamped (ms) | `boot2` Clamped (ms) | 2-Boot Clamped Mean (ms) | 2-Boot Speedup |
| :---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **TP3** | `c1` copy-widened tail step (`M = 17 -> M = 2`) | `213.589` | `213.981` | `213.785` | `81.895` | `82.264` | **`82.080`** | **`2.60x`** (`-131.7 ms`) |
| **TP3** | `c1` DSpark tail step (`M = 6 -> M = 2`) | `147.478` | `148.483` | `147.981` | `82.036` | `82.372` | **`82.204`** | **`1.80x`** (`-65.8 ms`) |
| **TP3** | `c8` mixed-tail batch step (`M = 48 -> M = 36`) | `586.282` | `577.683` | `581.983` | `456.105` | `462.156` | **`459.131`** | **`1.27x`** (`-122.9 ms`) |
| **TP4** | `c1` copy-widened tail step (`M = 17 -> M = 2`) | `155.649` | `157.217` | `156.433` | `67.062` | `67.588` | **`67.325`** | **`2.32x`** (`-89.1 ms`) |
| **TP4** | `c1` DSpark tail step (`M = 6 -> M = 2`) | `115.124` | `115.620` | `115.372` | `67.041` | `66.824` | **`66.933`** | **`1.72x`** (`-48.4 ms`) |
| **TP4** | `c8` mixed-tail batch step (`M = 48 -> M = 36`) | `422.991` | `417.127` | `420.059` | `335.399` | `334.049` | **`334.724`** | **`1.25x`** (`-85.3 ms`) |

## 5. Memory Observations & Conclusion

- **Memory**: Zero additional host or device buffers allocated; reduces transient KV block allocations at the tail of bounded-generation requests by not allocating KV slots for speculative tokens past `max_tokens`.
- **Cross-rank determinism**: Clamping occurs entirely in `Scheduler.schedule()` on rank 0 before `SchedulerOutput` broadcast, so every tensor-parallel rank receives identical `num_scheduled_tokens` and `scheduled_spec_decode_tokens`.
- **Conclusion**: Confirmed lossless win (`1.72x–2.60x` faster tail verify step at `c1`, `1.25x–1.27x` faster mixed-tail verify step at `c8`, and `+11` active copy drafts reclaimed in multi-stream copy batches) with 100% bit-exact output tokens and sink-rejection preservation. Ready for review.
