# Dead verification row sampler skip & sparse top-k DSpark Markov sampling (`2026-11-08-sampler-dead-row-skip-and-sparse-topk`)

- **Date**: 2026-11-08
- **Base deployment commit**: `cc20ed18be1f7cbf5c6c53cdee945f7cc77d5fc5` (Promoted `spark-ds41f-1107`, `experiments/2026-10-26-spark-ds41f-candidate/1107`)
- **Base vLLM commit**: `2756478b60b90700b9918b457aeb38657da637b5` (tree `ed1bdb7a7684da33d403a055521ac0c4dd7ca670`, `0001..0761`, `patchset_sha256` `26fac4e56977e6f2dc057821f8fa5d10f1d6159861843e8f725bb8a2a10cef38`)
- **Candidate vLLM commit**: `6b3595eab18670f9349c590a855019080eaebb70` (tree `134aed8230d74b93d40e4f5d191e22e09ab186bd`, `0001..0762`, `patchset_sha256` `9b1ab9f9b0de8ec63b9ed7a7a5f8b48ca64996162c74a1c289b9c4e6d0606090`)
- **Intended causal variable**: Skip unverified dead verification rows (`row - cu_num_logits[r] > dead_row_cut[r]`) inside `Sampler.apply_sampling_params` during `RejectionSampler._verify`, and sample DSpark `dspark_draft_topk` sequential Markov candidates sparsely on `[num_reqs, k]` without full-vocabulary `-inf` fills, scatters, or full-vocabulary sampler scans (`0762-perf-spec-decode-sampler-dead-row-skip-and-sparse-topk.patch`).

## 1. Technical Audit & Motivation

1. **Unverified dead verification rows in `RejectionSampler._verify` (`Sampler.apply_sampling_params`)**:
   - During speculative block verification (`n_spec = 5`, `6` verification rows per request), `dead_row_cut` (`[num_reqs]`, `int32`) caps the last local verification offset (`0 .. dead_row_cut[r]`) that request `r` can commit in the current step (set by remaining `max_tokens`, adaptive verification confidence gating, or CED/local draft cuts).
   - While `rejection_sample` (`tilelang_rejection.py`) already skips rows with `j > dead_row_cut[r]` and the pure-temperature fast path (`in_kernel_temperature`) fused temperature into `rejection_sample`, any request batch with non-temperature sampling parameters (`top_k`, `top_p`, `min_p`, `repetition_penalty`, `frequency_penalty`, `presence_penalty`, `logit_bias`, `bad_words`, or `thinking_budget`) previously ran `Sampler.apply_sampling_params` across **all** `num_tokens` rows without `dead_row_cut`:
     - `tilelang_sampling.copy_logits` allocated and copied all `num_tokens * 129,280` elements to FP32 (`517.12 KB` per dead row);
     - `logit_bias`, `penalties`, `bad_words`, `thinking_budget`, `temperature`, and `min_p` read and mutated every dead row;
     - `tilelang_topk_topp.apply_top_k_top_p` ran 16-pass bitonic top-k and up to 32-iteration binary-search top-p pivot sweeps over all `129,280` columns of every dead row (and could stall up to 32 passes when unverified dead rows had flat/near-uniform logits).
2. **Dense `-inf` fill, scatter, and full-vocabulary scan in `DSparkSpeculator._sample_sequential_topk`**:
   - When `dspark_draft_topk = K` (`K = 128`) is configured on the DSpark drafter, `_sample_sequential_topk` selects `base_values, draft_indices = base_logits.topk(K, dim=-1)` (`[num_reqs, n_spec, K]`), but previously:
     - Filled the entire `[num_reqs, n_spec, 129280]` `base_logits` tensor with `-inf` (`1.29 MB` per request);
     - At each sequential Markov step `i in 0..4`, scattered the `K` corrected candidate logits back into `base_logits[:, i]`;
     - Launched full-vocabulary `argmax(dim=-1)` or 127-block `gumbel_sample` + `gumbel_finish` over all `129,280` columns per step.

## 2. Implementation (`0762-perf-spec-decode-sampler-dead-row-skip-and-sparse-topk.patch`)

1. **`tilelang_inputs.mask_dead_rows` & `Sampler.apply_sampling_params` dead-row skip**:
   - Added `mask_dead_rows_kernel` in `vllm/v1/worker/gpu/tilelang_inputs.py`: in a single 1-block launch over `num_tokens`, binary-searches `cu_num_logits` to find request `r` and local offset `j = row - cu_num_logits[r]`, writing `live_idx_mapping[row] = expanded_idx_mapping[row]` when `j <= dead_row_cut[r]` and `-1` when `j > dead_row_cut[r]`, while simultaneously gathering per-row `(row_top_k, row_top_p)` with neutral `(vocab_size, 1.0)` on dead rows (replacing the two `SamplingStates.get_top_k_top_p` `gather_rows` launches).
   - Guarded `copy_logits_kernel`, `bias_kernel`, `penalties_kernel`, `bad_words_kernel`, `thinking_kernel`, `temperature_kernel`, and `min_p_kernel` so rows with `idx_mapping[row] < 0` exit before reading or writing vocabulary rows, and passed `(row_top_k, row_top_p)` directly into `tilelang_topk_topp.apply_top_k_top_p`.
   - Passed `dead_row_cut` and `cu_num_logits` from `RejectionSampler._verify` into `Sampler.apply_sampling_params` whenever `use_block_verification` is active and `logprobs_mode not in PROCESSED_LOGPROBS_MODES`.
2. **Sparse-support sequential Markov sampling (`apply_markov_bias_topk`, `topk_greedy_sample`, `topk_gumbel_sample`)**:
   - Added `apply_markov_bias_topk` on `DSparkDeepseekV4ForCausalLM`, `DSparkMarkovHead`, and `Qwen3DSparkForCausalLM` returning `values + bias` (`[num_reqs, K]`) directly without scattering into a dense vocabulary buffer.
   - Added `topk_greedy_sample_kernel` and `topk_gumbel_sample_kernel` in `vllm/v1/worker/gpu/sample/tilelang_sampling.py`. Each candidate's Philox-4x32 word is drawn at its global vocabulary index `index = cand_indices[t, c]` via `vllm_randint32(word_seed[0], index)` and reduced with `vllm_first_argmax_keep` on global vocabulary indices (seeding `(-inf, 0)` when `topk < vocab`), matching full-vocabulary `argmax` and `gumbel_sample` on a `-inf`-masked vocabulary tensor bit for bit.
   - When `draft_logits` caching is active (`draft_sample_method == "probabilistic"`), `DSparkSpeculator` initializes `draft_logits` once to `-inf` and tracks `_cached_topk_indices` (`[max_num_reqs, n_spec, K]`); `topk_gumbel_sample_kernel` clears only the previous step's `K` candidate indices to `-inf` before writing the new `K` candidate scores.

## 3. Exact Verification & Benchmark Commands (`sp5`, `NVIDIA GB10`)

```bash
flock /tmp/gpu.lock bash -c '
set -e
for boot in 1 2; do
  docker run --rm --gpus all \
    -v /tmp/sampler-dead-row-work/cand/vllm/v1/worker/gpu/tilelang_inputs.py:/opt/spark-ds41f/candidate/vllm/vllm/v1/worker/gpu/tilelang_inputs.py:ro \
    -v /tmp/sampler-dead-row-work/cand/vllm/v1/worker/gpu/sample/tilelang_controls.py:/opt/spark-ds41f/candidate/vllm/vllm/v1/worker/gpu/sample/tilelang_controls.py:ro \
    -v /tmp/sampler-dead-row-work/cand/vllm/v1/worker/gpu/sample/tilelang_bias.py:/opt/spark-ds41f/candidate/vllm/vllm/v1/worker/gpu/sample/tilelang_bias.py:ro \
    -v /tmp/sampler-dead-row-work/cand/vllm/v1/worker/gpu/sample/tilelang_auxiliary.py:/opt/spark-ds41f/candidate/vllm/vllm/v1/worker/gpu/sample/tilelang_auxiliary.py:ro \
    -v /tmp/sampler-dead-row-work/cand/vllm/v1/worker/gpu/sample/tilelang_thinking.py:/opt/spark-ds41f/candidate/vllm/vllm/v1/worker/gpu/sample/tilelang_thinking.py:ro \
    -v /tmp/sampler-dead-row-work/cand/vllm/v1/worker/gpu/sample/tilelang_sampling.py:/opt/spark-ds41f/candidate/vllm/vllm/v1/worker/gpu/sample/tilelang_sampling.py:ro \
    -v /tmp/sampler-dead-row-work/cand/vllm/v1/worker/gpu/sample/sampler.py:/opt/spark-ds41f/candidate/vllm/vllm/v1/worker/gpu/sample/sampler.py:ro \
    -v /tmp/sampler-dead-row-work/cand/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py:/opt/spark-ds41f/candidate/vllm/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py:ro \
    -v /tmp/sampler-dead-row-work/cand/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py:/opt/spark-ds41f/candidate/vllm/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py:ro \
    -v /tmp/sampler-dead-row-work/cand/vllm/models/deepseek_v4_1/nvidia/dspark.py:/opt/spark-ds41f/candidate/vllm/vllm/models/deepseek_v4_1/nvidia/dspark.py:ro \
    -v /tmp/sampler-dead-row-work/cand/vllm/model_executor/models/qwen3_dspark.py:/opt/spark-ds41f/candidate/vllm/vllm/model_executor/models/qwen3_dspark.py:ro \
    -v /tmp/sampler-dead-row-work/cand/tests/v1/worker/test_dead_rows.py:/opt/spark-ds41f/candidate/vllm/tests/v1/worker/test_dead_rows.py:ro \
    -v /tmp/sampler-dead-row-work/cand/tests/v1/spec_decode/test_dspark_topk.py:/opt/spark-ds41f/candidate/vllm/tests/v1/spec_decode/test_dspark_topk.py:ro \
    -v /tmp/sampler-dead-row-work/cand/tests/v1/spec_decode/test_dspark_v41_gathered_markov.py:/opt/spark-ds41f/candidate/vllm/tests/v1/spec_decode/test_dspark_v41_gathered_markov.py:ro \
    --entrypoint pytest spark-ds41f:1107 --noconftest -q \
    /opt/spark-ds41f/candidate/vllm/tests/v1/worker/test_dead_rows.py \
    /opt/spark-ds41f/candidate/vllm/tests/v1/spec_decode/test_dspark_topk.py \
    /opt/spark-ds41f/candidate/vllm/tests/v1/spec_decode/test_dspark_v41_gathered_markov.py \
    /opt/spark-ds41f/candidate/vllm/tests/v1/worker/test_gpu_sampling_states_gather.py \
    /opt/spark-ds41f/candidate/vllm/tests/kernels/test_deepseek_v41_copy_logits.py \
    /opt/spark-ds41f/candidate/vllm/tests/v1/worker/test_prepare_kernels.py

  docker run --rm --gpus all \
    -v /tmp/sampler-dead-row-work:/work \
    ... \
    --entrypoint python3 spark-ds41f:1107 /work/screen_dead_row_and_sparse_topk.py \
    --boot $boot --out /work/bench-boot${boot}.json
done
'
```

## 4. Two-Boot GPU Verification & Benchmark Results (`sp5`, `NVIDIA GB10`)

### 4.1 Unit & Integration Test Suites (`boot1` & `boot2`)

- Target test suite (`test_dead_rows.py`, `test_dspark_topk.py`, `test_dspark_v41_gathered_markov.py`, `test_gpu_sampling_states_gather.py`, `test_deepseek_v41_copy_logits.py`, `test_prepare_kernels.py`):
  - `receipts/gpu-pytest-boot1.txt`: **`151 passed in 68.61s`**
  - `receipts/gpu-pytest-boot2.txt`: **`151 passed in 68.38s`**
- Full rejection sampler suites (`test_tilelang_rejection.py`, `test_rejection_sampler_utils.py`): **`784 passed`** (`935` total tests passing).

### 4.2 Dead Verification Row Skip in `Sampler.apply_sampling_params` (`V = 129,280`, `n_spec = 5`)

All runs configure `temperature=0.8`, `top_k=64` (`VOCAB_SIZE` in the `top_p`-only rows), `top_p=0.9`, `min_p=0.05`, `repetition_penalty=1.15`, `frequency_penalty=0.2`, `presence_penalty=0.3`, `logit_bias={7: 1.5, 12: -2.0}`, and `bad_words=[[15, 16], [25]]`. Every case verified `bit_exact: true` (`torch.equal` on all live processed rows and `rejection_sample` outputs even when all dead rows in the input logits are filled with `NaN`):

| Case (`B`, `total -> live` rows) | `boot1` Base ($\mu$s) | `boot2` Base ($\mu$s) | 2-Boot Base Mean ($\mu$s) | `boot1` Cand ($\mu$s) | `boot2` Cand ($\mu$s) | 2-Boot Cand Mean ($\mu$s) | 2-Boot Speedup |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `normal_B1_cut1` (`6 -> 2` rows) | `161.02` | `161.02` | `161.02` | `140.54` | `141.73` | **`141.14`** | **`1.141x`** (`-19.9 us`) |
| `normal_B1_cut2` (`6 -> 3` rows) | `161.07` | `161.55` | `161.31` | `166.11` | `165.76` | **`165.94`** | `0.972x` (`+4.6 us`) |
| `normal_B4_cut1` (`24 -> 8` rows) | `438.88` | `486.05` | `462.47` | `290.05` | `290.05` | **`290.05`** | **`1.594x`** (`-172.4 us`) |
| `normal_B4_cut2` (`24 -> 12` rows) | `442.69` | `438.46` | `440.58` | `324.64` | `324.86` | **`324.75`** | **`1.357x`** (`-115.8 us`) |
| `top_p_only_flat_dead_B4_cut1` (`24 -> 8` rows) | `467.07` | `465.89` | `466.48` | `324.93` | `324.86` | **`324.90`** | **`1.436x`** (`-141.6 us`) |
| `normal_B8_cut1` (`48 -> 16` rows) | `940.16` | `941.33` | `940.75` | `353.44` | `353.54` | **`353.49`** | **`2.661x`** (`-587.3 us`) |
| `normal_B8_cut2` (`48 -> 24` rows) | `962.56` | `1015.41` | `988.99` | `443.63` | `450.61` | **`447.12`** | **`2.212x`** (`-541.9 us`) |
| `top_p_only_flat_dead_B8_cut1` (`48 -> 16` rows) | `1271.97` | `1308.91` | `1290.44` | `574.70` | `584.77` | **`579.74`** | **`2.226x`** (`-710.7 us`) |
| `normal_B16_cut1` (`96 -> 32` rows) | `2428.62` | `2508.37` | `2468.50` | `682.91` | `676.53` | **`679.72`** | **`3.632x`** (`-1,788.8 us`) |
| `normal_B16_cut2` (`96 -> 48` rows) | `2669.66` | `2658.50` | `2664.08` | `1205.31` | `1205.26` | **`1205.29`** | **`2.210x`** (`-1,458.8 us`) |
| `top_p_only_flat_dead_B16_cut1` (`96 -> 32` rows) | `3317.90` | `3362.99` | `3340.45` | `975.94` | `983.01` | **`979.48`** | **`3.410x`** (`-2,361.0 us`) |

### 4.3 Sparse-Support Sequential Markov Sampling (`V = 129,280`, `r = 256` 12-bit packed BF16 `markov_w2`, `n_spec = 5`, `K = 128`)

Every case verified `bit_exact: true` on `draft_tokens` (`[B, 5]`) and, in probabilistic mode, on the entire `[B, 5, 129280]` `draft_logits` cache across consecutive decode steps:

| Case (`B`, Mode) | `boot1` Base ($\mu$s) | `boot2` Base ($\mu$s) | 2-Boot Base Mean ($\mu$s) | `boot1` Cand ($\mu$s) | `boot2` Cand ($\mu$s) | 2-Boot Cand Mean ($\mu$s) | 2-Boot Speedup |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `greedy_markov_loop_B1` | `272.56` | `269.14` | `270.85` | `242.43` | `247.09` | **`244.76`** | **`1.107x`** (`-26.1 us`) |
| `greedy_markov_loop_B2` | `286.64` | `290.37` | `288.51` | `268.96` | `266.18` | **`267.57`** | **`1.078x`** (`-20.9 us`) |
| `greedy_markov_loop_B4` | `273.14` | `275.52` | `274.33` | `266.78` | `267.97` | **`267.38`** | **`1.026x`** (`-7.0 us`) |
| `greedy_markov_loop_B8` | `271.62` | `273.66` | `272.64` | `264.88` | `265.63` | **`265.26`** | **`1.028x`** (`-7.4 us`) |
| `greedy_markov_loop_B16` | `428.80` | `430.26` | `429.53` | `267.46` | `267.30` | **`267.38`** | **`1.606x`** (`-162.2 us`) |
| `probabilistic_markov_loop_B1` | `352.45` | `384.26` | `368.36` | `302.14` | `301.89` | **`302.02`** | **`1.220x`** (`-66.3 us`) |
| `probabilistic_markov_loop_B2` | `396.05` | `401.02` | `398.54` | `345.26` | `350.10` | **`347.68`** | **`1.146x`** (`-50.9 us`) |
| `probabilistic_markov_loop_B4` | `397.47` | `403.34` | `400.41` | `343.17` | `347.52` | **`345.35`** | **`1.159x`** (`-55.1 us`) |
| `probabilistic_markov_loop_B8` | `400.51` | `405.76` | `403.14` | `345.09` | `348.64` | **`346.87`** | **`1.162x`** (`-56.3 us`) |
| `probabilistic_markov_loop_B16` | `584.58` | `585.97` | `585.28` | `346.88` | `346.32` | **`346.60`** | **`1.689x`** (`-238.7 us`) |

## 5. Errors Encountered During Screening

- During initial harness setup, `test_sampler_apply_sampling_params_skips_dead_rows_bit_exact` wrote `req_states.num_computed_tokens.cpu` and `sampler.sampling_states.repetition_penalty` directly instead of using `RequestState.add_request` + `Sampler.add_request` + `apply_staged_writes()`, and `test_dspark_revisited_position_uses_step_seed` constructed a bare `DSparkSpeculator.__new__(DSparkSpeculator)` without `use_confidence_head`. Updated `speculator.py` to use `getattr(self, "use_confidence_head", False)` and updated the test/benchmark harness to use `add_request` + `apply_staged_writes()`.

## 6. Memory Observations & Conclusion

- **Memory**: `mask_dead_rows` allocates only per-token 1D row metadata (`[num_tokens]` `int64` / `int32` / `float32`, `< 2 KB` at `96` rows), and `_cached_topk_indices` is allocated only when both `dspark_draft_topk` and `draft_logits` are active (`max_num_reqs * n_spec * K * 8` bytes = `80 KB` at `max_num_reqs = 16, n_spec = 5, K = 128`).
- **Conclusion**: Confirmed bit-exact win across both optimizations (`1.36x–3.63x` faster `Sampler.apply_sampling_params` under `dead_row_cut` at `B = 4..16`, saving up to `2.36 ms`/step on `NVIDIA GB10`, and `1.15x–1.69x` faster probabilistic top-k sequential Markov sampling plus `1.61x` faster greedy top-k Markov sampling at `B = 16`). Ready for review.
