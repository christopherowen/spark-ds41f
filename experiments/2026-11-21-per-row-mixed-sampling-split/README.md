# Decode #18: Per-Request Mixed-Batch Shard-Local vs. Non-Temperature Control Sampling Split

- **Base commit**: `a5d1f33a5` (`spark-ds41f:candidate-63-r5`)
- **One intended variable**: Partition mixed sampling/verification batches per request in `GPUModelRunner` (`vllm/v1/worker/gpu/model_runner.py`: `_samples_split_from_shards` / `_sample_split_from_shards`) so temperature-only and greedy requests stay on the shard-local raw-BF16 path while only requests with non-temperature controls (`top_p < 1`, `top_k < vocab`, `min_p > 0`, penalties, `logit_bias`, `bad_words`, `thinking_budget`) all-gather full-vocabulary logits, copy to FP32, and run control kernels.

## Diagnosis

In `candidate-63-r5`, `any_non_temperature_logits_processing(idx_mapping_np)` is a batch-wide `any()` check. When even 1 request in an 8- or 16-stream batch sets `top_p=0.95` (or `top_k`/penalty) while the remaining 7 or 15 requests are greedy or temperature-only:
1. `_verifies_from_shards` and `_samples_from_shards` return `False` for the entire batch, forcing `compute_logits` to all-gather all `48` or `96` rows × `129,280` BF16 logits (`12.4–24.8 MB`) across TP ranks over RoCE.
2. `apply_sampling_params` copies all `48` or `96` rows into a `24.8–49.6 MB` FP32 buffer (`165–318 us`).
3. `tilelang_topk_topp.apply_top_k_top_p` sizes `splits = _topp_split_count(rows, num_sm)` to the full batch row count (`48` -> `splits=1` instead of `6` -> `splits=8`), starving the 6 active rows on 6/48 SMs (`221 us` sharp / `844 us` flat vs `48 us` / `120 us` when compacted to 6 rows).

## Change

- [`overlay/model_runner.py`](overlay/model_runner.py) ([`per_row_mixed_sampling_split.patch`](per_row_mixed_sampling_split.patch)):
  - Tracks `self._cu_num_logits_np_exact` in `prepare_inputs` (`False` only when adaptive verification unevenly compacts drafts on-device: `0 < draft_budget < num_drafts_total`).
  - Adds `_samples_split_from_shards` (zero D2H sync; checks `num_reqs >= 2`, `_cu_num_logits_np_exact`, `_sampled_verification_available()`, no grammar/logprobs/sink-rejection, `processor.use_all_gather` for `TP > 1`, and strictly mixed `ctrl_mask.any() and not ctrl_mask.all()`).
  - Adds `_sample_split_from_shards`: runs `compute_logits_local` once, routes `~ctrl_mask` requests through `sampled_rejection_sample_from_shards` / `sample_tokens_from_shards` (zero full-vocab all-gather, zero FP32 copy), calls `processor._gather_logits` ONLY on `ctrl_row_idx` rows, runs `self.rejection_sampler` / `self.sampler` on the compacted `ctrl_batch` (giving `top_p` full SM split parallelism sized to its active rows), and merges outputs back into batch order.

## Verification (`dgx4`, `spark-ds41f:candidate-63-r5`)

### Correctness & Selective-Gather Tests (`5 passed, 1 skipped, 0 failed`)
```bash
docker run --rm --gpus all \
  -v /tmp/p7_sampling/patched/model_runner.py:/opt/spark-ds41f/candidate/vllm/vllm/v1/worker/gpu/model_runner.py:ro \
  -v /tmp/p7_sampling:/work \
  -v /tmp/p5_decode910/stub:/tmp/stub:ro \
  -e PYTHONPATH=/tmp/stub:/opt/spark-ds41f/candidate/vllm \
  --entrypoint pytest spark-ds41f:candidate-63-r5 \
  -v /work/test_mixed_sampling_split.py
```
- `test_split_guards`: verifies all-fast, all-ctrl, mixed, device-compacted adaptive (`_cu_num_logits_np_exact=False`), and grammar guards.
- `test_mixed_split_bit_exact_parity`: verifies **`100% torch.equal`** fixed-seed token, `num_sampled`, and `num_rejected` parity across interleaved batches (`top_p=0.95`, `top_k=40 + repetition_penalty=1.15`, greedy, `temp=0.5..0.8`), variable drafts `[6, 1, 4, 6, 3, 6, 6, 2]`, `use_drafts=False/True`, and `dead_row_cut=True` — matching every request's pure-batch reference bit-for-bit.
- `test_split_tp3_selective_gather_rows_only`: verifies `processor._gather_logits` is called ONLY on the `6` control rows (`gather_calls == [6]`), never on the `18` fast rows.

### End-to-End Mixed-Batch GPU Latency Benchmark (`dgx4`, `vocab = 129,280`, sharp LLM logits, `TP = 1` GPU compute only — excludes `4×–16×` RoCE all-gather reduction on TP3/TP4)

| Batch (`nctrl/nreqs` at `top_p=0.95`, rest `temp=0.6`) | Unsplit (`us`) | Split (`us`) | GPU Compute Delta | RoCE All-Gather Reduction |
|---|---:|---:|---:|---|
| `8x6` (`1/8` ctrl, `6/48` rows) | `883.7` | `465.8` | **`-418.0 us` (`-47.3%`)** | **`8×`** (`12.4 -> 1.55 MB`) |
| `8x6` (`2/8` ctrl, `12/48` rows) | `815.2` | `482.0` | **`-333.2 us` (`-40.9%`)** | **`4×`** (`12.4 -> 3.10 MB`) |
| `16x6` (`1/16` ctrl, `6/96` rows) | `1,236.2` | `696.5` | **`-539.6 us` (`-43.7%`)** | **`16×`** (`24.8 -> 1.55 MB`) |
| `16x6` (`2/16` ctrl, `12/96` rows) | `1,269.9` | `757.8` | **`-512.0 us` (`-40.3%`)** | **`8×`** (`24.8 -> 3.10 MB`) |

Pure-batch `apply_top_k_top_p` isolation on sharp LLM logits (`dgx4`): `8x6` (`1/8` active): `221.1 -> 48.4 us` (`-172.6 us`, `4.6×` faster); `16x6` (`1/16` active): `139.8 -> 48.4 us` (`-91.3 us`). On flat Gaussian logits: `8x6`: `843.6 -> 120.2 us` (`-723.5 us`, `7.0×` faster); `16x6`: `407.3 -> 120.6 us` (`-286.7 us`).

## Conclusion

Clear, 100% bit-exact win for mixed production traffic (`-0.42..-0.54 ms/step` GPU compute + `4×–16×` less RoCE vocab all-gather, `~1.5–2.5 ms/step` total on TP3/TP4 when a batch mixes temperature-only/greedy streams with `top_p`/`top_k`/penalty streams) with zero overhead on homogeneous batches. Ready to roll into `candidate-64`.
