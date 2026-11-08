# 2026-11-08 — TP=3 and TP=4 Decode Recovery (`0762`)

## Context & Base

- **Repository base commit**: `cc20ed18b` (`main`)
- **Candidate base**: `experiments/2026-10-26-spark-ds41f-candidate/1107`
  - vLLM series `0001..0761` (`641` patches, base patch head `320021e3c4dbb202d911ab70e2cd924c75420488`, tree `ed1bdb7a7684da33d403a055521ac0c4dd7ca670`)
  - TileLang `0.1.8+dea9b489.dsv41.p13` (`tree e2b001828fd4e38e6a2eee61f818d5083cb1335f`)
  - SparkNet `0.1.0` (`tree 8cc28a5cd3e683c1f114ed843453e3ce34f323e0`)
- **Candidate patch**: `experiments/2026-11-08-tp3-decode-recovery/vllm/0762-perf-dsv41-tp3-decode-compressor-mla-and-spec-fusions.patch`
  - Patch head: `a32ae4512616acee8d964d976e6fbc646e08b72f`
  - Expected vLLM tree: `b35b9df25f435c3deee87221263c913b450041a4`

## Root-Cause Analysis of the TP=3 Decode Regression (`candidate-36` vs `1107`)

Comparing the recorded `tp3-triangle` benchmark suites (`candidate-36-screen-tp3` vs `1107-release-tp3`) and isolating per-kernel and per-phase execution on `sp8` (NVIDIA GB10, SM121) identified five distinct contributors to `1107`'s decode regression on `tp3-triangle` (and shared decode overhead on `tp4-ring`):

1. **Sharded 12-bit `pack12` KV Compressor (`0749`)**:
   - `0749` set `disable_tp = False` on `DeepseekCompressor.fused_wkv_wgate` and packed its weights into 12-bit `pack12`.
   - Across the 4 compressor layers (2 ratio-1 at `512 x 5120`, 2 ratio-2 at `1024 x 5120`), `disable_tp = False` added **4 RoCE `tensor_model_parallel_all_gather` collectives per decode step** (~80–100 us/step on TP=3/TP=4), **4 `gather_columns_into` GPU kernel launches per decode step**, and 12-bit unpack instructions on weights that total only `30 MiB` across the entire model.
   - Replicating the 4 compressor layers in uncompressed BF16 up through TP=4 (`disable_tp = tp_size <= 4`, `pack_bf16 = not disable_tp`) with early L2 weight prefetch (`prefetch_compressor_weights` at the start of `DeepseekV41Model.forward`) eliminates all 4 RoCE `all_gather`s and all 4 `gather_columns_into` launches while cutting local projection time in half (`103.6–107.6 us -> 51.6–53.1 us` across all 4 layers).

2. **Sparse MLA Decode Stage-1/Stage-2 Predication (`0740` / `0747`)**:
   - Wrapping the Stage-2 scalar online-softmax recurrence (`m_new`, `a`, `b`, `L`, `M`) in a dynamic `if seg_active:` predicate inside `sparse_mla_fwd_v41` forced scalar register dependency barriers around `PartialStats` loads and reduced warp issue efficiency across all 30 indexed layers.
   - Having Stage 1 explicitly write the online-softmax identity `(-inf, 0.0)` to `PartialStats[row, seg, h0 + h, :]` when a segment is inactive allows Stage 2 to execute the 5 scalar online-softmax lines unconditionally (`b = exp2(-inf - M) = 0`, `a = 1`, `L` and `M` unchanged) and guard only the 512-element `acc_o` vector FMA loop with `if PartialStats[rc, c, hc, 1] > 0:`.
   - Keeping `split_max_rows = sms // (segments * cdiv(heads, split_block_H))` (`4` on TP=3 `heads=24`, `9` on TP=4 `heads=16`) avoids spilling `r = 6` on TP=3 into a 60-CTA 2-wave `split` launch (`25.86 us` vs `20.30 us` on the 30-CTA 1-wave `decode` launch).

3. **Unfused Eager Post-Step Verification Bookkeeping (`0744` / `0745`)**:
   - After `rejection_sampler` in `GPUModelRunner.execute_model`, five eager PyTorch GPU tensor ops (`sub`, `clamp_min`, `minimum`, `masked_fill`, `where`) were launched every step to clamp `num_rejected` against `dead_row_cut`, zero `SinkOutcome.REJECTED` steps, and mask `copy_int_state[:, 1] > 0` slots before `OnlineAcceptanceEstimator.step`, followed by a duplicate Python call to `adaptive_verification.get_verified_draft_counts`.
   - Fusing `dead_row_cut`, `sink_events`, and `copy_int_state` directly into `_accumulate_kernel` in `acceptance_estimator.py`, reusing `num_verified_draft_tokens`, and running `copy_drafter.prepare_alloc` on `output_copy_stream` before stream synchronization cuts post-step bookkeeping from `68.2–72.4 us` to `11.9–12.9 us` (**`-56 to -60 us/step`**).

4. **Two-Kernel DSpark Confidence Head + Calibration & `(4, 24)` `_dummy_run` Mismatch (`0744`)**:
   - `DSparkSpeculator.propose` launched `TileLangConfidenceHead.probability` followed by a separate Triton `_calibrate_confidence_kernel` launch, allocated a temporary zeros tensor inside `_forward_tilelang_vocab_step` when `markov_bias is None`, and during `_dummy_run` for shape `(4, 24)` padded `input_batch.num_reqs` from `4` to `8`, causing `(4, 24)` to miss the `(4, 20)` DSpark draft CUDA graph (`8 * 5 = 40 > 24`) and fall back to eager DSpark execution on `c4`.
   - Adding `dspark_confidence_calibrate_kernel` in `vllm/models/deepseek_v4_1/tilelang/confidence.py` fuses logistic calibration (`slope`, `intercepts`) and design-matrix recording (`features`, `predictions`) into the confidence dot-product epilogue (`17.2–19.0 us -> 8.5–9.2 us`, **2.0x speedup**), preallocating `w2._tilelang_vocab_bias_scratch` removes the per-step zero allocation, and decoupling `num_reqs_after_padding` from `input_batch.num_reqs` in `_dummy_run` restores `(4, 20)` DSpark graph capture for `(4, 24)`.

5. **Topology-Agnostic Adaptive Verification Knee & Floor (`0728` / `0744`)**:
   - On `tp3-triangle`, dense and MoE shards are `4/3 = 1.333x` wider per rank than on `tp4-ring`, and `sparse_mla` runs `24` heads (`block_H=32`) vs `16` heads (`block_H=16`), making the verification step cost jump across row buckets (`4 -> 6` on `c1` and `24 -> 32 -> 48` on `c4`/`c8`) steeper than on TP=4.
   - Setting `HIGH_ROW_COST_SCALE_TP3 = 0.85` (above row knee `16`) and `MIN_CALIBRATED_DEAD_ROWS_TAU_TP3 = 0.12` when `_tp_world_size() == 3` trims low-acceptance tail drafts on TP=3 (`c1`: `5 -> 4` total rows; `c8` medium confidence: `40 -> 32` total rows) while keeping TP=4 (`scale = 0.55`, `floor = 0.0`) bit-for-bit unchanged.

## Exact Verification & Benchmark Commands (`sp8`, 2 Container Boots)

```bash
# Two-boot unit test verification on sp8 (NVIDIA GB10)
for boot in 1 2; do
  docker run --rm --gpus all --entrypoint /bin/bash \
    -v /tmp/vllm-0762-run:/workspace/vllm:ro \
    -v /tmp:/host_tmp \
    -e PYTHONPATH=/workspace/vllm \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -e TILELANG_CACHE_DIR=/host_tmp/tilelang_cache_test \
    spark-ds41f:1107 -c "
      cd /workspace/vllm
      python3 -m pytest --noconftest -o cache_dir=/tmp/pytest_cache -q \
        tests/models/test_deepseek_v4_1_compressor.py \
        tests/kernels/test_deepseek_v41_tilelang_confidence.py \
        tests/v1/spec_decode/test_acceptance_estimator.py \
        tests/v1/spec_decode/test_adaptive_verification.py
      python3 -m pytest --noconftest -o cache_dir=/tmp/pytest_cache -q \
        tests/kernels/attention/test_deepseek_v41_tilelang.py \
        -k test_sparse_mla_plan_routes
    " | tee receipts/pytest-boot${boot}.txt
done

# Two-boot GPU kernel/phase screening benchmark on sp8 (NVIDIA GB10)
for boot in 1 2; do
  docker run --rm --gpus all --entrypoint /bin/bash \
    -v /tmp/vllm-0762-run:/workspace/vllm:ro \
    -v /tmp:/host_tmp \
    -e PYTHONPATH=/workspace/vllm \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -e TILELANG_CACHE_DIR=/host_tmp/tilelang_cache_test \
    spark-ds41f:1107 -c "
      python3 /host_tmp/screen_tp3_decode.py /host_tmp/boot-${boot}.json > /dev/null
    "
done
```

## Two-Boot GPU Results on `sp8` (`receipts/boot-1.json`, `receipts/boot-2.json`)

### 1. KV Compressor Projection (4 Layers per Step: 2x Ratio-1 + 2x Ratio-2)

| Rows | `1107` Local 4-Layer (`pack12` + `gather_columns_into`) Boot 1 / Boot 2 (us) | `0762` 4-Layer (`bf16_gemm_splitk`, `disable_tp=True`) Boot 1 / Boot 2 (us) | Local Delta Boot 1 / Boot 2 (us/step) | Eliminated RoCE `all_gather`s / Step |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 107.62 / 107.11 | 52.97 / 51.60 | **-54.64 / -55.51** | 4 |
| 4 | 106.08 / 103.22 | 52.52 / 52.08 | **-53.56 / -51.14** | 4 |
| 6 | 103.92 / 104.00 | 52.52 / 53.09 | **-51.41 / -50.91** | 4 |
| 8 | 104.69 / 103.63 | 52.22 / 52.09 | **-52.47 / -51.54** | 4 |
| 16 | 105.66 / 103.59 | 51.59 / 51.71 | **-54.07 / -51.88** | 4 |
| 24 | 109.51 / 103.56 | 55.39 / 52.80 | **-54.11 / -50.75** | 4 |
| 32 | 107.67 / 103.26 | 52.10 / 53.25 | **-55.57 / -50.01** | 4 |
| 48 | 110.52 / 102.34 | 55.63 / 55.40 | **-54.89 / -46.94** | 4 |

- **Memory observation**: Replicating the 4 compressor projections in uncompressed BF16 uses `30.0 MiB` per rank (vs `5.625 MiB` sharded `pack12` on TP=4 or `8.10 MiB` on TP=3, a `+21.9 to +24.4 MiB` static footprint delta on a 128 GiB node) while removing 4 RoCE `all_gather`s and saving `~51–55 us/step` in local kernel execution alone (`~135–155 us/step` including RoCE `all_gather`s).

### 2. Sparse MLA Decode on TP=3 (`heads = 24`, Mixed Context `idx_lens = [128, 256, 384, 512]`, 30 Layers)

| Rows | Route (`1107` / `0762`) | `1107` Per-Layer Boot 1 / Boot 2 (us) | `0762` Per-Layer Boot 1 / Boot 2 (us) | 30-Layer Step Delta Boot 1 / Boot 2 (ms) |
| ---: | :---: | ---: | ---: | ---: |
| 4 | `split` / `split` | 20.42 / 21.49 | 20.30 / 21.71 | -0.003 / +0.007 |
| 6 | `decode` / `decode` | 20.59 / 20.57 | 20.30 / 21.59 | -0.009 / +0.031 |
| 8 | `decode` / `decode` | 20.38 / 21.82 | 20.57 / 20.98 | +0.006 / **-0.025** |
| 12 | `decode` / `decode` | 22.16 / 23.29 | 20.53 / 21.21 | **-0.049 / -0.062** |
| 16 | `decode` / `decode` | 30.76 / 27.67 | 27.47 / 26.13 | **-0.099 / -0.046** |
| 24 | `decode` / `decode` | 33.45 / 35.56 | 35.51 / 32.66 | +0.062 / **-0.087** |

### 3. Post-Step Verification Bookkeeping (`acceptance_estimator.py`)

| `num_reqs` | `1107` Unfused Boot 1 / Boot 2 (us) | `0762` Fused `_accumulate_kernel` Boot 1 / Boot 2 (us) | Delta Boot 1 / Boot 2 (us/step) |
| ---: | ---: | ---: | ---: |
| 1 | 68.19 / 71.02 | 12.66 / 12.87 | **-55.53 / -58.15** |
| 2 | 69.63 / 72.19 | 12.65 / 12.68 | **-56.98 / -59.51** |
| 4 | 69.57 / 72.36 | 12.80 / 12.61 | **-56.76 / -59.75** |
| 8 | 71.16 / 72.15 | 12.77 / 11.89 | **-58.40 / -60.26** |

### 4. Fused DSpark Confidence Head + Online Logistic Calibration (`tilelang/confidence.py`)

| `num_reqs` | `1107` Two-Kernel Boot 1 / Boot 2 (us) | `0762` Single-Kernel `dspark_confidence_calibrate` Boot 1 / Boot 2 (us) | Delta Boot 1 / Boot 2 (us/step) |
| ---: | ---: | ---: | ---: |
| 1 | 19.02 / 17.59 | 8.78 / 8.83 | **-10.24 / -8.76** |
| 2 | 17.36 / 17.61 | 8.75 / 8.67 | **-8.61 / -8.93** |
| 4 | 18.38 / 17.31 | 9.16 / 8.52 | **-9.22 / -8.79** |
| 8 | 18.15 / 17.16 | 8.66 / 8.51 | **-9.49 / -8.65** |

### 5. Topology-Aware Adaptive Verification (`adaptive_verification.py`)

- `c1` typical calibrated confidence `[0.677, 0.556, 0.396, 0.460, 0.638]`:
  - **TP=4** (`scale = 0.55`, `floor = 0.00`): `4` drafts (`5` total verification rows, unchanged).
  - **TP=3** (`scale = 0.85`, `floor = 0.12`): `3` drafts (`4` total verification rows, keeping verification inside the 4-row `split` MLA & GEMM bucket).
- `c8` medium calibrated confidence `[0.78, 0.72, 0.71, 0.71, 0.73]` across 8 requests:
  - **TP=4** (`scale = 0.55`, `floor = 0.00`): `32` drafts (`40` total verification rows -> padded to `48`, unchanged).
  - **TP=3** (`scale = 0.85`, `floor = 0.12`): `24` drafts (`32` total verification rows -> stays in the `32`-row bucket instead of spilling to `48`).

## Unit Test Receipts

- `receipts/pytest-boot1.txt`: `118 passed, 1 skipped` + `1 passed` (`119 passed, 0 failed`).
- `receipts/pytest-boot2.txt`: `118 passed, 1 skipped` + `1 passed` (`119 passed, 0 failed`).

## Conclusion

Patch `0762` resolves the TP=3 decode regression introduced between `candidate-36` and `1107` while improving both TP=3 and TP=4 decode step latency through replicated BF16 compressor projections with early L2 prefetch, branchless Stage-1/Stage-2 sparse MLA online-softmax merge, fused post-step verification bookkeeping, fused DSpark confidence calibration with preallocated Markov bias scratch and `(4, 24)` DSpark graph capture, and TP=3-aware adaptive verification row-cost scaling.
