# Enable Online Calibration of `confidence_head` (Fix Overconfident Tail Survival)

## Problem

In `DSparkSpeculator.load_draft_model`, enabling `confidence_head` previously disabled `OnlineAcceptanceEstimator`:

```python
self.use_confidence_head = (
    self.enable_adaptive_verification
    and getattr(model.model, "confidence_head", None) is not None
)
if self.use_confidence_head:
    # The acceptance estimator is not needed when a trained confidence head
    # is available.
    self.use_acceptance_estimator = False
```

This creates two compounding errors in `AdaptiveVerificationManager`:

1. **Zero Online Calibration Against Domain & Temperature Shift**: `confidence_head` was trained offline on greedy (`temperature = 0`) distributions where conditional acceptance averages `[0.78, 0.72, 0.71, 0.71, 0.73]`. Under the production serving recipe (`temperature = 1.0`, `top_p = 0.95` with greedy DSpark drafting) and across creative prose vs. code workloads, empirical conditional acceptance drops to `[0.6545, 0.5421, 0.4422, 0.4589, 0.6181]` at `c1` and `[0.5877, 0.5317, 0.4115, 0.3554, 0.5142]` at `c8`.
2. **Compounding Parallel-Drafting Error in Tail Survival**: `AdaptiveVerificationManager._accept_Counts` computes `survival_probability = np.cumprod(stale_confidences, axis=1)`, assuming conditional independence $P(\text{accept } 0..k) = \prod_{j=0}^k p_j$. In DSpark, positions $1..4$ are drafted in parallel from mask tokens with only a 1-step Markov bias (`proj([h_k | e_{k-1}])`), so errors compound faster than $\prod p_{\text{head}, j}$ at positions 2, 3, and 4. Uncalibrated `cumprod(p_head) = [0.7800, 0.5616, 0.3987, 0.2831, 0.2067]` overestimates position-3 survival by **3.93x** (`0.2831` vs `0.0720` at `c1`) to **6.19x** (`0.2831` vs `0.0457` at `c8`) and position-4 survival by **4.64x** (`0.2067` vs `0.0445` at `c1`) to **8.79x** (`0.2067` vs `0.0235` at `c8`).

## Solution

1. **Feed $\text{logit}(p_{\text{head}})$ Into the Existing GPU-Resident Online IRLS Solver**:
   - Added `_calibrate_confidence_kernel` and `OnlineAcceptanceEstimator.calibrate_confidence_probs` in `vllm/v1/worker/gpu/spec_decode/acceptance_estimator.py`.
   - Initialized with identity cold-start parameters $\omega_0 = 1.0$ and $\beta_{0, k} = 0.0$ (`OnlineAcceptanceEstimator.for_confidence_head` / `reset_for_confidence_head`) so before the first refit, $\sigma(\omega_0 \cdot \text{logit}(p_{\text{head}}) + \beta_{0, k}) = p_{\text{head}}$ identically.
   - Inside the captured draft CUDA graph (`DSparkSpeculator._sample_sequential`), `_calibrate_confidence_kernel` transforms raw `confidence_head` probabilities in place via $\hat{p}_{r, k} = \sigma(\omega \cdot \text{logit}(p_{\text{head}}(r, k)) + \beta_k)$ in ~2–3 $\mu$s/step while recording $(\text{logit}(p_{\text{head}}), \hat{p}_{r, k})$ in persistent request-state slots (`features`, `predictions`).
   - Every step, `observe_verification` runs `_accumulate_kernel` on graded `(num_sampled, num_rejected)` outcomes, and every `REFIT_INTERVAL` steps `_refit_kernel` solves the arrowhead Newton-IRLS system in place on GPU to update the shared slope $\omega$ and per-position intercepts $\beta_0, \dots, \beta_4$.
2. **Preserve Vocabulary-Parallel Greedy Reductions (`_vocab_parallel_greedy`)**:
   - Added `_needs_logits_for_acceptance(speculator)` in `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py` (`acceptance_estimator is not None and not use_confidence_head`).
   - Because confidence-head calibration reads $\text{logit}(p_{\text{head}})$ rather than full-vocabulary draft logits, `_vocab_parallel_greedy` and `sample_greedy_markov` remain enabled when `use_confidence_head=True` and `acceptance_estimator` is active.

## Replay Results (`calibration_policy.py` / `results.json`)

| Workload | Empirical Target Survival (`pos 0..4`) | Static `cumprod(p_head)` | Calibrated Survival (`pos 0..4`) | Survival MAE (Static $\to$ Calibrated) | Admitted Depth (Static $\to$ Calibrated / Optimal) |
| --- | --- | --- | --- | ---: | ---: |
| `single-stream/c1` | `[0.6545, 0.3548, 0.1569, 0.0720, 0.0445]` | `[0.7800, 0.5616, 0.3987, 0.2831, 0.2067]` | `[0.6768, 0.3763, 0.1492, 0.0686, 0.0438]` | `0.1895` $\to$ **`0.0111`** (**17.1x** lower) | `5` $\to$ **`3` / `3`** |
| `penalties-sampled/c8` | `[0.5877, 0.3125, 0.1286, 0.0457, 0.0235]` | `[0.7800, 0.5616, 0.3987, 0.2831, 0.2067]` | `[0.6193, 0.3446, 0.1337, 0.0596, 0.0368]` | `0.2264` $\to$ **`0.0192`** (**11.8x** lower) | `4` $\to$ **`2` / `2`** |
| `chat-Mixed/c16` | `[0.7393, 0.5204, 0.3144, 0.2202, 0.1726]` | `[0.7800, 0.5616, 0.3987, 0.2831, 0.2067]` | `[0.7566, 0.5364, 0.3087, 0.2052, 0.1596]` | `0.0526` $\to$ **`0.0134`** (**3.9x** lower) | `2` $\to$ **`2` / `2`** |
