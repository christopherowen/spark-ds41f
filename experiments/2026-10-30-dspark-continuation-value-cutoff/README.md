# Continuation-Value Cutoff and Slot Ranking for DSpark Adaptive Verification

## Motivation

In `AdaptiveVerificationManager`, each proposed draft slot $(r, j)$ is currently scored myopically by its prefix survival probability:

$$S_{r, j} = \prod_{m=0}^{j} p_{r, m}$$

When a draft sequence contains a moderate-confidence bridge token at position $j$ (such as a whitespace, punctuation, or formatting token) followed by high-confidence continuation tokens at positions $j+1, \dots, L_r-1$, myopic scoring prunes position $j$ whenever $S_{r, j}$ falls below the row-cost cutoff or the dead-row threshold $\tau$, discarding the high-yield suffix behind it.

## Formulation

For a request with $L_r$ scheduled draft tokens, the expected number of accepted tokens contributed by the suffix strictly following position $j$ is:

$$V_{r, j+1} = \sum_{m=j+1}^{L_r - 1} S_{r, m}$$

Weighting this continuation yield by $\alpha \ge 0$ (`SPARK_DS41F_DSPARK_CONTINUATION_ALPHA`, default `0`) defines the effective survival score:

$$\tilde{S}_{r, j}(\alpha) = S_{r, j} + \alpha \sum_{m=j+1}^{L_r - 1} S_{r, m} = (1 - \alpha) S_{r, j} + \alpha \sum_{m=j}^{L_r - 1} S_{r, m}$$

### Monotonicity Guarantee

Because $p_{r, m} \in [0, 1]$, prefix survival $S_{r, j}$ is non-negative and monotonically non-increasing in $j$. Since the suffix sum $\sum_{m=j+1}^{L_r - 1} S_{r, m}$ drops a non-negative term $S_{r, j+1}$ at each step $j \to j+1$, it is also monotonically non-increasing in $j$. Therefore:

$$\tilde{S}_{r, 0}(\alpha) \ge \tilde{S}_{r, 1}(\alpha) \ge \dots \ge \tilde{S}_{r, L_r - 1}(\alpha)$$

for every $\alpha \ge 0$. Consequently:
1. Global top-$B$ slot selection in `assign_draft_budget_kernel` and `_assign_draft_token_budget` always admits a contiguous prefix of each request's draft tokens without gaps.
2. On-device dead-row pruning in `combine_sampled_and_draft_tokens_kernel` (`\log \tilde{S}_{r, j}(\alpha) \ge \log \tau`) always defines a contiguous live prefix `0 .. cut - 1`.
3. Terminal draft positions ($j = L_r - 1$) and bridge tokens followed by low-confidence dead tails receive zero or negligible continuation bonus and continue to be pruned.

## Changes

- `vllm/v1/worker/gpu/input_batch.py`:
  - Added optional `continuation_alpha: float = 0.0` field to `DeadRows`.
- `vllm/v1/worker/gpu/spec_decode/adaptive_verification.py`:
  - Added `SPARK_DS41F_DSPARK_CONTINUATION_ALPHA`, `continuation_enabled()`, and `effective_survival_scores()`.
  - Updated `_assign_draft_token_budget`, `AdaptiveVerificationManager.get_num_tokens`, `load_draft_budget_kernels`, `dead_row_args`, and `reallocate_drafts` to apply continuation-value scoring when `continuation_alpha > 0`.
- `vllm/v1/worker/gpu/tilelang_inputs.py`:
  - Updated `assign_draft_budget_kernel` (with `vllm_budget_add` and `vllm_budget_ffma` matching PyTorch FP32 rounding bit-for-bit) and `combine_sampled_and_draft_tokens_kernel` to support `continuation=True` while keeping the default `continuation=False` kernel keys and generated code unchanged.
