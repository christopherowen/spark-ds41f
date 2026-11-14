# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Online verification-driven DSpark drafter adaptation (opt-in, off by default).

Three components learn from the target's verification of DSpark's drafts. The
target's verdict alone decides the emitted tokens, so none of them can change
the text; they change which drafts are proposed and how many rows a step
verifies.

1. ``OnlineConfidenceCalibrator`` (``VLLM_DSPARK_ONLINE_CALIBRATION=1``, or
   part of ``VLLM_DSPARK_ONLINE_ADAPT=1``): per-position log-odds calibration of
   the confidence head, ``c_hat = sigmoid(gamma_i * logit(c) + beta_i)``, updated
   every step in FP64 from the accepted and first-rejected positions. It
   replaces ``OnlineAcceptanceEstimator``'s shared-slope refit as the
   calibration the captured draft graph applies.
2. ``OnlineDSparkHeadAdapter`` (``VLLM_DSPARK_ONLINE_ADAPT=1``): the
   prefix-product total-variation (``E2E-TV``) and confidence ``BCE`` step on
   DSpark's replicated Markov W1 and confidence head, on a three-token support
   built from each verified position (the target's token, the draft, one other
   token) with the drafter's own base and Markov logits there.
   ``DSparkOnlineAdaptation`` wires it to the verification of the previous
   step's drafts.
3. ``SessionTokenMemory`` (``VLLM_DSPARK_SESSION_MEMORY_TOKENS=N``): a bounded
   ring of each request's verified tokens, consulted by ``CopyDrafter`` when the
   request's own history has no copy match. It is per request: a slot's ring is
   cleared when a new request takes the slot, and nothing is read across
   requests.

Tensor parallelism: every input of an update is identical on every rank (the
sampled tokens and counts, the drafter's head hidden, the replicated W1 and
confidence head). The vocabulary-sharded tables (Markov W2 and the LM head) are
read through ``VocabRows``: each rank contributes the rows of the ids its shard
owns and zeros for the rest, and one all-reduce sums them. Each id has one
owner, so the sum is exact and the same bits arrive on every rank. Updates use
fixed-order reductions and no atomics, so ranks stay identical;
``DSparkOnlineAdaptation.state_digest`` hashes the state for checking.

W2 is never updated under tensor parallelism or when it is packed: a sharded
table indexed by global ids would write other rows, and a 12-bit packed table
has no BF16 rows to write.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice
except ImportError:
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    libdevice = None  # type: ignore[assignment]

_TRUE = ("1", "true", "yes", "on")


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in _TRUE


def _env_int(name: str, default: int = 0) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


def online_dspark_adaptation_enabled() -> bool:
    """Part 2 (W1 missed-token adaptation): ``VLLM_DSPARK_ONLINE_ADAPT``."""
    return _env_flag("VLLM_DSPARK_ONLINE_ADAPT")


def online_confidence_calibration_enabled() -> bool:
    """Part 1: ``VLLM_DSPARK_ONLINE_CALIBRATION`` (independent of W1 adapt)."""
    return _env_flag("VLLM_DSPARK_ONLINE_CALIBRATION")


def session_token_memory_capacity() -> int:
    """Part 3's per-request ring, in tokens: ``VLLM_DSPARK_SESSION_MEMORY_TOKENS``.

    Off (0) unless set; ``VLLM_DSPARK_ONLINE_ADAPT`` does not turn it on.
    """
    return max(0, _env_int("VLLM_DSPARK_SESSION_MEMORY_TOKENS"))


def online_update_w2_requested() -> bool:
    """``VLLM_DSPARK_ONLINE_UPDATE_W2``: refused under TP or a packed W2."""
    return _env_flag("VLLM_DSPARK_ONLINE_UPDATE_W2")


def online_digest_interval() -> int:
    """Log the adaptation state's digest every this many updates (0: never)."""
    return max(0, _env_int("VLLM_DSPARK_ONLINE_DIGEST_STEPS"))


def online_max_row_drift() -> float:
    """Bound on a W1 row's displacement from its loaded value, as a fraction of
    that row's norm (``VLLM_DSPARK_ONLINE_MAX_DRIFT``, default 1.0; 0 or less
    turns the bound and the copy of the loaded table off)."""
    return _env_float("VLLM_DSPARK_ONLINE_MAX_DRIFT", 1.0)


@dataclass
class E2ETVAndBCEResult:
    """Structured result of the prefix-product total-variation and BCE objective."""

    total_loss: torch.Tensor
    e2e_tv_loss: torch.Tensor
    bce_loss: torch.Tensor
    step_tv_distance: torch.Tensor
    token_overlap: torch.Tensor
    prefix_survival: torch.Tensor
    excluded_teacher_mass: torch.Tensor


def compute_e2e_tv_and_bce_loss(
    draft_candidate_logits: torch.Tensor,
    target_candidate_probs: torch.Tensor,
    confidence_logits: torch.Tensor | None = None,
    *,
    confidence_probs: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
    bce_weight: float = 0.25,
    eps: float = 1e-6,
) -> E2ETVAndBCEResult:
    """Prefix-product total-variation (``E2E-TV``) and confidence ``BCE`` loss.

    Args:
        draft_candidate_logits: ``[batch, steps, topk]`` drafter logits over the
            candidate support ``S_{b,i}``.
        target_candidate_probs: ``[batch, steps, topk]`` target probabilities
            ``p_{b,i}(v)`` on ``S_{b,i}``; their sum may be below 1, and the
            excluded mass ``1 - sum_v p_{b,i}(v)`` enters the step distance
            ``d = 0.5 * (sum_v |q(v) - p(v)| + 1 - sum_v p(v))`` at each
            ``(b, i)``.
        confidence_logits: Optional ``[batch, steps]`` raw confidence logits.
        confidence_probs: Optional ``[batch, steps]`` confidence probabilities
            (used if ``confidence_logits`` is not provided).
        valid_mask: Optional ``[batch, steps]`` mask of active positions.
        bce_weight: Weight of the confidence BCE against the detached overlap.
        eps: Numerical clamping epsilon.
    """
    if draft_candidate_logits.ndim != 3 or target_candidate_probs.ndim != 3:
        raise ValueError(
            "draft_candidate_logits and target_candidate_probs must have shape "
            "[batch, steps, topk]"
        )
    if draft_candidate_logits.shape != target_candidate_probs.shape:
        raise ValueError(
            f"Shape mismatch: draft_candidate_logits {draft_candidate_logits.shape} "
            f"vs target_candidate_probs {target_candidate_probs.shape}"
        )

    q_probs = F.softmax(draft_candidate_logits.float(), dim=-1)
    p_probs = target_candidate_probs.float().clamp_min(0.0)

    included_teacher_mass = p_probs.sum(dim=-1).clamp(0.0, 1.0)
    excluded_teacher_mass = (1.0 - included_teacher_mass).clamp(0.0, 1.0)

    l1_on_support = torch.abs(q_probs - p_probs).sum(dim=-1)
    step_tv_distance = (0.5 * (l1_on_support + excluded_teacher_mass)).clamp(0.0, 1.0)
    token_overlap = (1.0 - step_tv_distance).clamp(0.0, 1.0)

    if valid_mask is not None:
        mask_f = valid_mask.to(dtype=token_overlap.dtype)
        effective_overlap = torch.where(
            valid_mask.bool(), token_overlap, torch.ones_like(token_overlap)
        )
        prefix_survival = torch.cumprod(effective_overlap, dim=-1)
        denom = mask_f.sum().clamp_min(1.0)
        e2e_tv_loss = 1.0 - (prefix_survival * mask_f).sum() / denom
    else:
        prefix_survival = torch.cumprod(token_overlap, dim=-1)
        e2e_tv_loss = 1.0 - prefix_survival.mean()

    detached_overlap = token_overlap.detach()
    if confidence_logits is not None:
        bce_per_pos = F.binary_cross_entropy_with_logits(
            confidence_logits.float(), detached_overlap, reduction="none"
        )
    elif confidence_probs is not None:
        conf_p = confidence_probs.float().clamp(eps, 1.0 - eps)
        bce_per_pos = -(
            detached_overlap * torch.log(conf_p)
            + (1.0 - detached_overlap) * torch.log(1.0 - conf_p)
        )
    else:
        bce_per_pos = None
    if bce_per_pos is None:
        bce_loss = torch.zeros((), dtype=e2e_tv_loss.dtype, device=e2e_tv_loss.device)
    elif valid_mask is not None:
        mask_f = valid_mask.to(dtype=bce_per_pos.dtype)
        bce_loss = (bce_per_pos * mask_f).sum() / mask_f.sum().clamp_min(1.0)
    else:
        bce_loss = bce_per_pos.mean()

    total_loss = e2e_tv_loss + float(bce_weight) * bce_loss
    return E2ETVAndBCEResult(
        total_loss=total_loss,
        e2e_tv_loss=e2e_tv_loss,
        bce_loss=bce_loss,
        step_tv_distance=step_tv_distance,
        token_overlap=token_overlap,
        prefix_survival=prefix_survival,
        excluded_teacher_mass=excluded_teacher_mass,
    )


def verification_outcomes(
    num_steps: int,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    dead_row_cut: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Each row's observed draft positions and their labels, ``[rows, steps]``.

    A row accepted ``A = num_sampled - 1`` drafts and judged ``A + rejected``
    (the rejected count limited by the dead-row cut, as the estimator's
    accumulation does). Positions ``i < A`` are accepted (label 1) and position
    ``A`` is the first rejection (label 0) when the row judged it.
    """
    acc = (num_sampled.to(torch.int64) - 1).clamp_min(0)
    rej = num_rejected.to(torch.int64)
    if dead_row_cut is not None:
        live = (dead_row_cut[: acc.shape[0]].to(torch.int64) - acc).clamp_min(0)
        rej = torch.minimum(rej, live)
    admitted = acc + rej
    pos = torch.arange(num_steps, dtype=torch.int64, device=acc.device)[None, :]
    observed = (pos <= acc[:, None]) & (pos < admitted[:, None])
    labels = pos < acc[:, None]
    return observed, labels


class OnlineConfidenceCalibrator:
    """Online per-position log-odds calibrator for draft confidences.

    Maintains per-position scale ``gamma_i`` and bias ``beta_i`` in FP64:

        c_hat_{r,i} = sigmoid(gamma_i * logit(c_{r,i}) + beta_i)

    updated by damped exponential-moving Newton steps on the binary
    cross-entropy over the observed positions (``i < A_r`` with label 1, and
    ``i = A_r < B_r`` with label 0). Identical inputs give identical FP64
    results on every TP rank, so no communication is needed. ``gamma32`` and
    ``beta32`` mirror the parameters in FP32 for the captured draft graph.
    """

    def __init__(
        self,
        num_steps: int,
        *,
        lr_bias: float = 0.20,
        lr_scale: float = 0.10,
        ema_decay: float = 0.90,
        damping: float = 0.05,
        min_scale: float = 0.25,
        max_scale: float = 4.0,
        max_abs_bias: float = 6.0,
        max_log_odds: float = 12.0,
        eps: float = 1e-6,
        device: torch.device | str = "cpu",
    ) -> None:
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        self.num_steps = int(num_steps)
        self.lr_bias = float(lr_bias)
        self.lr_scale = float(lr_scale)
        self.ema_decay = float(ema_decay)
        self.damping = float(damping)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.max_abs_bias = float(max_abs_bias)
        self.max_log_odds = float(max_log_odds)
        self.eps = float(eps)
        self.device = torch.device(device)

        f64 = dict(dtype=torch.float64, device=self.device)
        self.beta = torch.zeros(self.num_steps, **f64)
        self.gamma = torch.ones(self.num_steps, **f64)
        self.curv_beta = torch.full((self.num_steps,), 0.25, **f64)
        self.curv_gamma = torch.full((self.num_steps,), 0.25, **f64)
        self.obs_counts = torch.zeros(self.num_steps, **f64)
        f32 = dict(dtype=torch.float32, device=self.device)
        self.gamma32 = torch.ones(self.num_steps, **f32)
        self.beta32 = torch.zeros(self.num_steps, **f32)
        self.step_count: int = 0
        self.raw_bce_ema: float = 0.0
        self.cal_bce_ema: float = 0.0
        self.raw_brier_ema: float = 0.0
        self.cal_brier_ema: float = 0.0

    def _sync_mirrors(self) -> None:
        self.gamma32.copy_(self.gamma)
        self.beta32.copy_(self.beta)

    def reset(self) -> None:
        self.beta.zero_()
        self.gamma.fill_(1.0)
        self.curv_beta.fill_(0.25)
        self.curv_gamma.fill_(0.25)
        self.obs_counts.zero_()
        self._sync_mirrors()
        self.step_count = 0
        self.raw_bce_ema = 0.0
        self.cal_bce_ema = 0.0
        self.raw_brier_ema = 0.0
        self.cal_brier_ema = 0.0

    def state_dict(self) -> dict[str, Any]:
        return {
            "beta": self.beta.clone(),
            "gamma": self.gamma.clone(),
            "curv_beta": self.curv_beta.clone(),
            "curv_gamma": self.curv_gamma.clone(),
            "obs_counts": self.obs_counts.clone(),
            "step_count": self.step_count,
            "raw_bce_ema": self.raw_bce_ema,
            "cal_bce_ema": self.cal_bce_ema,
            "raw_brier_ema": self.raw_brier_ema,
            "cal_brier_ema": self.cal_brier_ema,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for name in ("beta", "gamma", "curv_beta", "curv_gamma", "obs_counts"):
            getattr(self, name).copy_(
                state[name].to(device=self.device, dtype=torch.float64)
            )
        self._sync_mirrors()
        self.step_count = int(state.get("step_count", 0))
        self.raw_bce_ema = float(state.get("raw_bce_ema", 0.0))
        self.cal_bce_ema = float(state.get("cal_bce_ema", 0.0))
        self.raw_brier_ema = float(state.get("raw_brier_ema", 0.0))
        self.cal_brier_ema = float(state.get("cal_brier_ema", 0.0))

    def _to_log_odds(self, confidences: torch.Tensor) -> torch.Tensor:
        clamped = confidences.to(dtype=torch.float64).clamp(self.eps, 1.0 - self.eps)
        return torch.log(clamped / (1.0 - clamped)).clamp(
            -self.max_log_odds, self.max_log_odds
        )

    def calibrate(self, confidences: torch.Tensor) -> torch.Tensor:
        """Return calibrated acceptance probabilities ``c_hat_{r,i}``."""
        if confidences.numel() == 0:
            return confidences.clone()
        steps = confidences.shape[-1]
        if steps > self.num_steps:
            raise ValueError(
                f"confidences has {steps} steps, exceeding calibrator "
                f"num_steps={self.num_steps}"
            )
        z = self._to_log_odds(confidences)
        gamma = self.gamma[:steps].to(device=confidences.device)
        beta = self.beta[:steps].to(device=confidences.device)
        return torch.sigmoid(gamma * z + beta).to(dtype=confidences.dtype)

    def calibrate_inplace(self, confidences: torch.Tensor) -> torch.Tensor:
        """Calibrate ``confidences`` in place and return the tensor."""
        if confidences.numel() == 0:
            return confidences
        confidences.copy_(self.calibrate(confidences))
        return confidences

    def survival_probabilities(
        self, confidences: torch.Tensor, *, calibrate: bool = True
    ) -> torch.Tensor:
        """Prefix survival probabilities ``prod_{j <= i} c_hat_{r,j}``."""
        probs = self.calibrate(confidences) if calibrate else confidences
        return torch.cumprod(probs.float(), dim=-1).to(dtype=confidences.dtype)

    def expected_acceptance_length(
        self, confidences: torch.Tensor, *, calibrate: bool = True
    ) -> torch.Tensor:
        """Expected accepted draft length ``sum_i prod_{j <= i} c_hat_{r,j}``."""
        return self.survival_probabilities(confidences, calibrate=calibrate).sum(dim=-1)

    def _build_observation_mask_and_labels(
        self,
        num_reqs: int,
        steps: int,
        accepted_counts: torch.Tensor,
        verified_counts: torch.Tensor | None,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        acc = accepted_counts[:num_reqs].to(device=device, dtype=torch.int64)
        acc = acc.clamp_min(0)
        if verified_counts is None:
            ver = torch.full((num_reqs,), steps, dtype=torch.int64, device=device)
        else:
            ver = (
                verified_counts[:num_reqs]
                .to(device=device, dtype=torch.int64)
                .clamp(0, steps)
            )
        pos = torch.arange(steps, dtype=torch.int64, device=device).unsqueeze(0)
        observed = (pos <= acc.unsqueeze(1)) & (pos < ver.unsqueeze(1))
        labels = (pos < acc.unsqueeze(1)).to(dtype=torch.float64)
        return observed, labels

    def compute_calibration_metrics(
        self,
        raw_confidences: torch.Tensor,
        accepted_counts: torch.Tensor,
        verified_counts: torch.Tensor | None = None,
        *,
        num_bins: int = 10,
    ) -> dict[str, float]:
        """BCE, Brier score and ECE before and after calibration (host syncs:
        for tests and offline analysis, not the serving loop)."""
        names = ("raw_bce", "cal_bce", "raw_brier", "cal_brier", "raw_ece", "cal_ece")
        zero = dict.fromkeys(names, 0.0)
        if raw_confidences.numel() == 0:
            return zero
        num_reqs, steps = raw_confidences.shape
        device = raw_confidences.device
        observed, labels = self._build_observation_mask_and_labels(
            num_reqs, steps, accepted_counts, verified_counts, device
        )
        if not bool(observed.any()):
            return zero

        raw_p = raw_confidences.to(dtype=torch.float64).clamp(self.eps, 1.0 - self.eps)
        cal_p = (
            self.calibrate(raw_confidences)
            .to(dtype=torch.float64)
            .clamp(self.eps, 1.0 - self.eps)
        )
        obs_raw, obs_cal, obs_lbl = raw_p[observed], cal_p[observed], labels[observed]

        def _bce(p: torch.Tensor) -> float:
            return float(
                (-(obs_lbl * torch.log(p) + (1.0 - obs_lbl) * torch.log(1.0 - p)))
                .mean()
                .item()
            )

        def _ece(probs: torch.Tensor, targets: torch.Tensor) -> float:
            edges = torch.linspace(
                0.0, 1.0, num_bins + 1, dtype=torch.float64, device=probs.device
            )
            total = float(probs.numel())
            ece_val = 0.0
            for b in range(num_bins):
                lo, hi = edges[b], edges[b + 1]
                upper = (probs <= hi) if b == num_bins - 1 else (probs < hi)
                in_bin = (probs >= lo) & upper
                cnt = int(in_bin.sum().item())
                if cnt > 0:
                    gap = probs[in_bin].mean() - targets[in_bin].mean()
                    ece_val += (cnt / total) * abs(float(gap.item()))
            return ece_val

        return {
            "raw_bce": _bce(obs_raw),
            "cal_bce": _bce(obs_cal),
            "raw_brier": float(((obs_raw - obs_lbl) ** 2).mean().item()),
            "cal_brier": float(((obs_cal - obs_lbl) ** 2).mean().item()),
            "raw_ece": _ece(obs_raw, obs_lbl),
            "cal_ece": _ece(obs_cal, obs_lbl),
        }

    @torch.no_grad()
    def observe_log_odds(
        self,
        z: torch.Tensor,
        observed: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        """Update ``(beta_i, gamma_i)`` from raw log-odds ``z`` ([rows, steps]),
        the observed mask and the labels, without a host synchronization."""
        if z.numel() == 0:
            return
        steps = z.shape[-1]
        if steps > self.num_steps:
            raise ValueError(
                f"log-odds have {steps} steps, exceeding calibrator "
                f"num_steps={self.num_steps}"
            )
        z = z.to(device=self.device, dtype=torch.float64).clamp(
            -self.max_log_odds, self.max_log_odds
        )
        obs_mask_f = observed.to(device=self.device, dtype=torch.float64)
        labels = labels.to(device=self.device, dtype=torch.float64)
        gamma = self.gamma[:steps]
        beta = self.beta[:steps]
        c_hat = torch.sigmoid(gamma.unsqueeze(0) * z + beta.unsqueeze(0)).clamp(
            self.eps, 1.0 - self.eps
        )

        counts_per_pos = obs_mask_f.sum(dim=0)
        active_pos = counts_per_pos > 0.0
        safe_counts = counts_per_pos.clamp_min(1.0)

        err = (c_hat - labels) * obs_mask_f
        grad_beta = err.sum(dim=0) / safe_counts
        grad_gamma = (err * z).sum(dim=0) / safe_counts

        fisher_w = c_hat * (1.0 - c_hat) * obs_mask_f
        hess_beta = fisher_w.sum(dim=0) / safe_counts
        hess_gamma = (fisher_w * z * z).sum(dim=0) / safe_counts

        rho = self.ema_decay
        new_curv_beta = torch.where(
            active_pos,
            rho * self.curv_beta[:steps] + (1.0 - rho) * hess_beta,
            self.curv_beta[:steps],
        )
        new_curv_gamma = torch.where(
            active_pos,
            rho * self.curv_gamma[:steps] + (1.0 - rho) * hess_gamma,
            self.curv_gamma[:steps],
        )
        self.curv_beta[:steps].copy_(new_curv_beta)
        self.curv_gamma[:steps].copy_(new_curv_gamma)
        self.obs_counts[:steps].add_(counts_per_pos)

        step_beta = torch.where(
            active_pos,
            self.lr_bias * grad_beta / (new_curv_beta + self.damping),
            torch.zeros_like(grad_beta),
        )
        step_gamma = torch.where(
            active_pos & (self.obs_counts[:steps] >= 2.0),
            self.lr_scale * grad_gamma / (new_curv_gamma + self.damping),
            torch.zeros_like(grad_gamma),
        )
        self.beta[:steps].copy_(
            (beta - step_beta).clamp(-self.max_abs_bias, self.max_abs_bias)
        )
        self.gamma[:steps].copy_(
            (gamma - step_gamma).clamp(self.min_scale, self.max_scale)
        )
        self._sync_mirrors()
        self.step_count += 1

    @torch.no_grad()
    def observe(
        self,
        raw_confidences: torch.Tensor,
        accepted_counts: torch.Tensor,
        verified_counts: torch.Tensor | None = None,
        *,
        return_metrics: bool = True,
    ) -> dict[str, float]:
        """Update from raw confidences ``[rows, steps]``, accepted counts ``A_r``
        and verified counts ``B_r``. With ``return_metrics`` the step's BCE and
        Brier scores are returned (one host synchronization)."""
        if raw_confidences.numel() == 0:
            return {
                "raw_bce": self.raw_bce_ema,
                "cal_bce": self.cal_bce_ema,
                "raw_brier": self.raw_brier_ema,
                "cal_brier": self.cal_brier_ema,
            }
        num_reqs, steps = raw_confidences.shape
        if steps > self.num_steps:
            raise ValueError(
                f"raw_confidences has {steps} steps, exceeding calibrator "
                f"num_steps={self.num_steps}"
            )
        raw_d = raw_confidences.to(device=self.device, dtype=torch.float64).clamp(
            self.eps, 1.0 - self.eps
        )
        observed, labels = self._build_observation_mask_and_labels(
            num_reqs, steps, accepted_counts, verified_counts, self.device
        )
        z = torch.log(raw_d / (1.0 - raw_d))
        self.observe_log_odds(z, observed, labels)
        if not return_metrics:
            return {}

        z = z.clamp(-self.max_log_odds, self.max_log_odds)
        obs_mask_f = observed.to(dtype=torch.float64)
        total_obs = obs_mask_f.sum().clamp_min(1.0)

        def _bce(p: torch.Tensor) -> torch.Tensor:
            nll = -(labels * torch.log(p) + (1.0 - labels) * torch.log(1.0 - p))
            return (nll * obs_mask_f).sum() / total_obs

        c_hat_post = torch.sigmoid(
            self.gamma[:steps].unsqueeze(0) * z + self.beta[:steps].unsqueeze(0)
        ).clamp(self.eps, 1.0 - self.eps)
        raw_bce_step, raw_brier_step, cal_bce_step, cal_brier_step = torch.stack(
            [
                _bce(raw_d),
                (((raw_d - labels) ** 2) * obs_mask_f).sum() / total_obs,
                _bce(c_hat_post),
                (((c_hat_post - labels) ** 2) * obs_mask_f).sum() / total_obs,
            ]
        ).tolist()
        rho = self.ema_decay
        if self.step_count == 1:
            self.raw_bce_ema = raw_bce_step
            self.cal_bce_ema = cal_bce_step
            self.raw_brier_ema = raw_brier_step
            self.cal_brier_ema = cal_brier_step
        else:
            self.raw_bce_ema = rho * self.raw_bce_ema + (1.0 - rho) * raw_bce_step
            self.cal_bce_ema = rho * self.cal_bce_ema + (1.0 - rho) * cal_bce_step
            self.raw_brier_ema = rho * self.raw_brier_ema + (1.0 - rho) * raw_brier_step
            self.cal_brier_ema = rho * self.cal_brier_ema + (1.0 - rho) * cal_brier_step
        return {
            "raw_bce": raw_bce_step,
            "cal_bce": cal_bce_step,
            "raw_brier": raw_brier_step,
            "cal_brier": cal_brier_step,
        }


def _group_first_positions(
    sorted_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """For ids sorted ascending: whether each entry is its id's first, and the
    position of its id's first entry."""
    n = sorted_ids.numel()
    first = torch.ones(n, dtype=torch.bool, device=sorted_ids.device)
    if n > 1:
        first[1:] = sorted_ids[1:] != sorted_ids[:-1]
    positions = torch.arange(n, dtype=torch.int64, device=sorted_ids.device)
    first_pos = torch.cummax(torch.where(first, positions, 0), dim=0).values
    return first, first_pos


def _deterministic_index_add_(
    target: torch.Tensor,
    flat_ids: torch.Tensor,
    flat_deltas: torch.Tensor,
) -> torch.Tensor:
    """``target[ids] += deltas`` with each id's deltas summed in a fixed order
    (stable sort, FP64) and written once per row value, without atomics or a host
    synchronization. Returns the ids sorted (for callers that post-process the
    touched rows)."""
    n = int(flat_ids.numel())
    if n == 0:
        return flat_ids
    sorted_ids, perm = torch.sort(flat_ids.to(torch.int64), stable=True)
    sorted_deltas = flat_deltas[perm].to(dtype=torch.float64)
    first, first_pos = _group_first_positions(sorted_ids)
    same = (sorted_ids.unsqueeze(1) == sorted_ids.unsqueeze(0)) & first.unsqueeze(1)
    # Row i (an id's first entry) sums that id's deltas in sorted order.
    grouped = same.to(torch.float64).unsqueeze(-1) * sorted_deltas.unsqueeze(0)
    grouped = grouped.sum(dim=1)
    new_rows = target[sorted_ids].to(torch.float64) + grouped
    # Every entry of an id writes its first entry's value: equal values, so the
    # duplicates' write order does not matter.
    target.index_copy_(0, sorted_ids, new_rows[first_pos].to(dtype=target.dtype))
    return sorted_ids


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """``x * rsqrt(mean(x^2) + eps) * weight`` in FP32, in ``x``'s dtype: the
    drafter's final norm in torch, for the eager update (the TileLang norm's
    kernels are prepared for the draft step's shapes only)."""
    xf = x.float()
    out = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (out * weight.float()).to(x.dtype)


def _rowwise_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``sum(a * b, -1)`` in FP32 with a fixed reduction (no cuBLAS, no TF32)."""
    return (a.float() * b.float()).sum(dim=-1)


if triton is not None:

    @triton.jit
    def _vocab_rows_local_kernel(
        ids_ptr,
        weight_ptr,
        out_ptr,
        stride_w,
        stride_o,
        n_ids,
        width: tl.constexpr,
        BLOCK_W: tl.constexpr,
        org_start,
        org_end,
        added_start,
        added_end,
        added_base,
    ):
        idx = tl.program_id(0)
        if idx >= n_ids:
            return
        tok = tl.load(ids_ptr + idx).to(tl.int64)
        in_org = (tok >= org_start) & (tok < org_end)
        in_added = (tok >= added_start) & (tok < added_end)
        owned = in_org | in_added
        local_id = tl.where(
            in_org,
            tok - org_start,
            tl.where(in_added, (tok - added_start) + added_base, 0),
        )
        offs_w = tl.arange(0, BLOCK_W)
        mask_w = offs_w < width
        if owned:
            row = tl.load(
                weight_ptr + local_id * stride_w + offs_w, mask=mask_w, other=0
            )
            tl.store(out_ptr + idx * stride_o + offs_w, row, mask=mask_w)
        else:
            zero = tl.zeros([BLOCK_W], dtype=out_ptr.dtype.element_ty)
            tl.store(out_ptr + idx * stride_o + offs_w, zero, mask=mask_w)

    @triton.jit
    def _capture_kernel(
        hidden_dst_ptr,
        hidden_dst_stride,
        hidden_src_ptr,
        hidden_src_stride,
        anchor_dst_ptr,
        anchor_src_ptr,
        slots_dst_ptr,
        slots_src_ptr,
        num_reqs,
        n_hidden_rows,
        max_num_reqs,
        hidden_size: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        idx = tl.program_id(0)
        if idx < n_hidden_rows:
            h_start = 0
            while h_start < hidden_size:
                offs = h_start + tl.arange(0, BLOCK_H)
                m = offs < hidden_size
                v = tl.load(
                    hidden_src_ptr + idx * hidden_src_stride + offs, mask=m, other=0.0
                )
                tl.store(hidden_dst_ptr + idx * hidden_dst_stride + offs, v, mask=m)
                h_start += BLOCK_H
        if idx < max_num_reqs:
            if idx < num_reqs:
                a = tl.load(anchor_src_ptr + idx).to(tl.int64)
                s = tl.load(slots_src_ptr + idx).to(tl.int64)
                tl.store(anchor_dst_ptr + idx, a)
                tl.store(slots_dst_ptr + idx, s)
            else:
                tl.store(slots_dst_ptr + idx, -1)

    @triton.jit
    def _prepare_and_pack_miss_kernel(
        idx_mapping_ptr,
        captured_slots_ptr,
        num_sampled_ptr,
        num_rejected_ptr,
        dead_row_cut_ptr,
        copy_int_state_ptr,
        copy_stride_r,
        sampled_ptr,
        sampled_stride_r,
        drafts_ptr,
        drafts_stride_r,
        anchor_ptr,
        runner_up_ptr,
        runner_stride_r,
        hidden_ptr,
        hidden_stride_r,
        norm_w_ptr,
        lm_ptr,
        lm_stride_r,
        lm_exc_ptr,
        lm_offsets_ptr,
        w2_ptr,
        w2_stride_r,
        miss_mask_ptr,
        cond_ids_ptr,
        packed_ptr,
        packed_stride_r,
        num_reqs,
        n_cands,
        max_num_reqs,
        steps: tl.constexpr,
        width: tl.constexpr,
        vocab_size: tl.constexpr,
        hidden_size: tl.constexpr,
        rank: tl.constexpr,
        norm_eps: tl.constexpr,
        HAS_DEAD_CUT: tl.constexpr,
        HAS_COPY_STATE: tl.constexpr,
        HAS_RUNNER_UP: tl.constexpr,
        LM_PACKED: tl.constexpr,
        LM_EXP_BASE: tl.constexpr,
        LM_SHARDED: tl.constexpr,
        lm_org_start,
        lm_org_end,
        lm_added_start,
        lm_added_end,
        lm_added_base,
        W2_SHARDED: tl.constexpr,
        w2_org_start,
        w2_org_end,
        w2_added_start,
        w2_added_end,
        w2_added_base,
        BLOCK_S: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        idx = tl.program_id(0)
        offs_r = tl.arange(0, BLOCK_R)
        mask_r = offs_r < rank
        zero_r = tl.zeros([BLOCK_R], dtype=tl.float32)

        if idx >= n_cands:
            tl.store(packed_ptr + idx * packed_stride_r + 0, 0.0)
            tl.store(
                packed_ptr + idx * packed_stride_r + 1 + offs_r, zero_r, mask=mask_r
            )
            return

        r = idx // 3
        c = idx % 3

        slot = tl.load(idx_mapping_ptr + r).to(tl.int64)
        slot_in_range = (slot >= 0) & (slot < max_num_reqs)

        draft_row = tl.full((), -1, dtype=tl.int64)
        if slot_in_range:
            s_start = 0
            while s_start < max_num_reqs:
                offs_s = s_start + tl.arange(0, BLOCK_S)
                m_s = offs_s < max_num_reqs
                s_val = tl.load(captured_slots_ptr + offs_s, mask=m_s, other=-1).to(
                    tl.int64
                )
                matched = m_s & (s_val == slot)
                cand_row = tl.max(tl.where(matched, offs_s + 1, 0), axis=0) - 1
                draft_row = tl.where(cand_row >= 0, cand_row.to(tl.int64), draft_row)
                s_start += BLOCK_S

        valid = draft_row >= 0
        if HAS_COPY_STATE and slot_in_range:
            copy_cnt = tl.load(copy_int_state_ptr + slot * copy_stride_r + 1).to(
                tl.int64
            )
            valid = valid & (copy_cnt <= 0)

        acc = tl.load(num_sampled_ptr + r).to(tl.int64) - 1
        acc = tl.where(acc > 0, acc, 0)
        rej = tl.load(num_rejected_ptr + r).to(tl.int64)
        if HAS_DEAD_CUT:
            cut = tl.load(dead_row_cut_ptr + r).to(tl.int64)
            live = cut - acc
            live = tl.where(live > 0, live, 0)
            rej = tl.where(rej < live, rej, live)

        admitted = acc + rej
        in_bounds = (acc < admitted) & (acc < steps) & (acc < width)
        safe_row = tl.where(draft_row >= 0, draft_row, 0)
        safe_pos = tl.where(in_bounds, acc, 0)

        tgt_tok = tl.load(sampled_ptr + r * sampled_stride_r + safe_pos).to(tl.int64)
        drf_tok = tl.load(drafts_ptr + safe_row * drafts_stride_r + safe_pos).to(
            tl.int64
        )
        is_miss = (
            valid
            & (rej > 0)
            & in_bounds
            & (tgt_tok >= 0)
            & (drf_tok >= 0)
            & (tgt_tok != drf_tok)
        )

        if c == 0:
            if is_miss:
                if safe_pos == 0:
                    cond_tok = tl.load(anchor_ptr + safe_row).to(tl.int64)
                else:
                    cond_tok = tl.load(
                        drafts_ptr + safe_row * drafts_stride_r + (safe_pos - 1)
                    ).to(tl.int64)
                cond_tok = tl.where(
                    cond_tok < 0,
                    0,
                    tl.where(cond_tok >= vocab_size, vocab_size - 1, cond_tok),
                )
                tl.store(miss_mask_ptr + r, 1)
                tl.store(cond_ids_ptr + r, cond_tok)
            else:
                tl.store(miss_mask_ptr + r, 0)
                tl.store(cond_ids_ptr + r, 0)

        if not is_miss:
            tl.store(packed_ptr + idx * packed_stride_r + 0, 0.0)
            tl.store(
                packed_ptr + idx * packed_stride_r + 1 + offs_r, zero_r, mask=mask_r
            )
            return

        tgt = tl.where(
            tgt_tok < 0,
            0,
            tl.where(tgt_tok >= vocab_size, vocab_size - 1, tgt_tok),
        )
        drf = tl.where(
            drf_tok < 0,
            0,
            tl.where(drf_tok >= vocab_size, vocab_size - 1, drf_tok),
        )
        alt1 = tl.where(drf != tgt, drf, (tgt + 1) % vocab_size)
        if c == 0:
            tok = tgt
        elif c == 1:
            tok = alt1
        else:
            alt2 = (tgt + 2) % vocab_size
            if HAS_RUNNER_UP:
                rup = tl.load(
                    runner_up_ptr + safe_row * runner_stride_r + safe_pos
                ).to(tl.int64)
                rup_c = tl.where(
                    rup < 0, 0, tl.where(rup >= vocab_size, vocab_size - 1, rup)
                )
                alt2 = tl.where((rup_c != tgt) & (rup_c != alt1), rup_c, alt2)
            alt2 = tl.where(alt2 == alt1, (tgt + 3) % vocab_size, alt2)
            tok = alt2

        if LM_SHARDED:
            lm_in_org = (tok >= lm_org_start) & (tok < lm_org_end)
            lm_in_added = (tok >= lm_added_start) & (tok < lm_added_end)
            lm_owned = lm_in_org | lm_in_added
            lm_local_id = tl.where(
                lm_in_org, tok - lm_org_start, (tok - lm_added_start) + lm_added_base
            )
        else:
            lm_owned = True
            lm_local_id = tok

        acc_dot = 0.0
        if lm_owned:
            h_row = safe_row * steps + safe_pos
            sum_sq = 0.0
            h_start = 0
            while h_start < hidden_size:
                offs_h = h_start + tl.arange(0, BLOCK_H)
                m_h = offs_h < hidden_size
                xv = tl.load(
                    hidden_ptr + h_row * hidden_stride_r + offs_h, mask=m_h, other=0.0
                ).to(tl.float32)
                sum_sq += tl.sum(xv * xv, axis=0)
                h_start += BLOCK_H
            inv_rms = tl.rsqrt(sum_sq * (1.0 / hidden_size) + norm_eps)

            if LM_PACKED:
                exc_start = tl.load(lm_offsets_ptr + lm_local_id)
                exc_end = tl.load(lm_offsets_ptr + lm_local_id + 1)
            else:
                exc_start = 0
                exc_end = 0

            h_start = 0
            while h_start < hidden_size:
                offs_h = h_start + tl.arange(0, BLOCK_H)
                m_h = offs_h < hidden_size
                if LM_PACKED:
                    lb = tl.load(
                        lm_ptr + lm_local_id * lm_stride_r + offs_h,
                        mask=m_h,
                        other=0,
                    ).to(tl.uint32)
                    c_off = hidden_size + ((offs_h >> 7) << 6) + (offs_h & 63)
                    cb = tl.load(
                        lm_ptr + lm_local_id * lm_stride_r + c_off,
                        mask=m_h,
                        other=0,
                    ).to(tl.uint32)
                    nib = (cb >> (((offs_h >> 6) & 1) * 4)) & 0xF
                    exp_bits = tl.where(
                        nib == 0, 0, (nib + (LM_EXP_BASE - 1)) << 7
                    )
                    u16 = ((lb & 0x80) << 8) | (lb & 0x7F) | exp_bits
                    exc_idx = exc_start
                    while exc_idx < exc_end:
                        entry = tl.load(lm_exc_ptr + exc_idx).to(tl.uint32)
                        col = entry >> 16
                        if (col >= h_start) & (col < h_start + BLOCK_H):
                            u16 = tl.where(offs_h == col, entry & 0xFFFF, u16)
                        exc_idx += 1
                    lm_v = (u16 << 16).to(tl.float32, bitcast=True)
                else:
                    lm_v = tl.load(
                        lm_ptr + lm_local_id * lm_stride_r + offs_h,
                        mask=m_h,
                        other=0.0,
                    ).to(tl.float32)

                xv = tl.load(
                    hidden_ptr + h_row * hidden_stride_r + offs_h, mask=m_h, other=0.0
                )
                nw = tl.load(norm_w_ptr + offs_h, mask=m_h, other=0.0).to(tl.float32)
                normed = (
                    ((xv.to(tl.float32) * inv_rms) * nw).to(xv.dtype).to(tl.float32)
                )
                acc_dot += tl.sum(lm_v * normed, axis=0)
                h_start += BLOCK_H

        tl.store(packed_ptr + idx * packed_stride_r + 0, acc_dot)

        if W2_SHARDED:
            w2_in_org = (tok >= w2_org_start) & (tok < w2_org_end)
            w2_in_added = (tok >= w2_added_start) & (tok < w2_added_end)
            if not (w2_in_org | w2_in_added):
                tl.store(
                    packed_ptr + idx * packed_stride_r + 1 + offs_r,
                    zero_r,
                    mask=mask_r,
                )
                return
            w2_local_id = tl.where(
                w2_in_org, tok - w2_org_start, (tok - w2_added_start) + w2_added_base
            )
        else:
            w2_local_id = tok
        w2_row = tl.load(
            w2_ptr + w2_local_id * w2_stride_r + offs_r, mask=mask_r, other=0.0
        ).to(tl.float32)
        tl.store(packed_ptr + idx * packed_stride_r + 1 + offs_r, w2_row, mask=mask_r)

    @triton.jit
    def _miss_update_w1_kernel(
        miss_mask_ptr,
        cond_ids_ptr,
        packed_ptr,
        packed_stride_r,
        w1_ptr,
        w1_stride_r,
        w1_anchor_ptr,
        anchor_stride_r,
        num_reqs,
        logits_scale,
        soft_cap,
        peak_prob,
        rem_prob,
        max_grad_norm,
        weight_decay,
        lr_markov_w1,
        anchor_decay,
        max_row_drift,
        HAS_SOFT_CAP: tl.constexpr,
        HAS_ANCHOR: tl.constexpr,
        rank: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        r = tl.program_id(0)
        if tl.load(miss_mask_ptr + r) == 0:
            return
        cid = tl.load(cond_ids_ptr + r).to(tl.int64)
        is_first = tl.full((), 1, dtype=tl.int32)
        k = 0
        while k < r:
            same = (tl.load(miss_mask_ptr + k) != 0) & (
                tl.load(cond_ids_ptr + k).to(tl.int64) == cid
            )
            is_first = tl.where(same, 0, is_first)
            k += 1
        if is_first == 0:
            return

        offs_r = tl.arange(0, BLOCK_R)
        m_r = offs_r < rank
        cur = tl.load(w1_ptr + cid * w1_stride_r + offs_r, mask=m_r, other=0)
        u = cur.to(tl.float32)
        acc = cur.to(tl.float64)

        k = r
        while k < num_reqs:
            same = (tl.load(miss_mask_ptr + k) != 0) & (
                tl.load(cond_ids_ptr + k).to(tl.int64) == cid
            )
            if same:
                row0 = (k * 3 + 0) * packed_stride_r
                row1 = (k * 3 + 1) * packed_stride_r
                row2 = (k * 3 + 2) * packed_stride_r
                b0 = tl.load(packed_ptr + row0) * logits_scale
                b1 = tl.load(packed_ptr + row1) * logits_scale
                b2 = tl.load(packed_ptr + row2) * logits_scale
                if HAS_SOFT_CAP:
                    b0 = soft_cap * libdevice.tanh(b0 / soft_cap)
                    b1 = soft_cap * libdevice.tanh(b1 / soft_cap)
                    b2 = soft_cap * libdevice.tanh(b2 / soft_cap)

                w2_0 = tl.load(packed_ptr + row0 + 1 + offs_r, mask=m_r, other=0.0)
                w2_1 = tl.load(packed_ptr + row1 + 1 + offs_r, mask=m_r, other=0.0)
                w2_2 = tl.load(packed_ptr + row2 + 1 + offs_r, mask=m_r, other=0.0)

                l0 = b0 + tl.sum(u * w2_0, axis=0)
                l1 = b1 + tl.sum(u * w2_1, axis=0)
                l2 = b2 + tl.sum(u * w2_2, axis=0)
                lmax = tl.maximum(l0, tl.maximum(l1, l2))
                e0 = tl.exp(l0 - lmax)
                e1 = tl.exp(l1 - lmax)
                e2 = tl.exp(l2 - lmax)
                inv_s = 1.0 / (e0 + e1 + e2)
                q0 = e0 * inv_s
                q1 = e1 * inv_s
                q2 = e2 * inv_s

                s0 = tl.where(q0 > peak_prob, 0.5, tl.where(q0 < peak_prob, -0.5, 0.0))
                s1 = tl.where(q1 > rem_prob, 0.5, tl.where(q1 < rem_prob, -0.5, 0.0))
                s2 = tl.where(q2 > rem_prob, 0.5, tl.where(q2 < rem_prob, -0.5, 0.0))
                es = q0 * s0 + q1 * s1 + q2 * s2
                g0 = q0 * (s0 - es)
                g1 = q1 * (s1 - es)
                g2 = q2 * (s2 - es)

                e_sum = (
                    tl.sum(w2_0 * w2_0, axis=0)
                    + tl.sum(w2_1 * w2_1, axis=0)
                    + tl.sum(w2_2 * w2_2, axis=0)
                )
                w2_energy = tl.maximum(e_sum * (1.0 / 3.0), 1e-3)
                grad_u = (g0 * w2_0 + g1 * w2_1 + g2 * w2_2) / w2_energy
                gu_norm = tl.maximum(tl.sqrt(tl.sum(grad_u * grad_u, axis=0)), 1e-8)
                grad_u = grad_u * tl.minimum(1.0, max_grad_norm / gu_norm)
                if weight_decay > 0.0:
                    grad_u = grad_u + weight_decay * u
                acc += (-lr_markov_w1 * grad_u).to(tl.float64)
            k += 1

        new_f = acc.to(cur.dtype).to(tl.float32)
        if HAS_ANCHOR:
            anc = tl.load(
                w1_anchor_ptr + cid * anchor_stride_r + offs_r, mask=m_r, other=0.0
            ).to(tl.float32)
            disp = new_f - anc
            if anchor_decay > 0.0:
                disp = disp * (1.0 - anchor_decay)
            anc_norm = tl.maximum(tl.sqrt(tl.sum(anc * anc, axis=0)), 1e-6)
            disp_norm = tl.maximum(tl.sqrt(tl.sum(disp * disp, axis=0)), 1e-12)
            limit = max_row_drift * anc_norm
            disp = disp * tl.minimum(1.0, limit / disp_norm)
            new_f = anc + disp
        tl.store(w1_ptr + cid * w1_stride_r + offs_r, new_f.to(cur.dtype), mask=m_r)

class VocabRows:
    """This rank's partial rows of a vocabulary table for global token ids.

    A row comes from the rank's shard when the shard owns the id, and is zero
    otherwise; summing the partial rows over the TP ranks gives every full row
    exactly. Handles a plain table (tensor or module ``weight``) and the
    12-bit packed BF16 head layout (``packed_bf16_exponent_base``), whose rows
    the TileLang packed embedding kernel decodes.
    """

    def __init__(self, table: nn.Module | torch.Tensor):
        self.table = table
        self.packed = not isinstance(table, torch.Tensor) and hasattr(
            table, "packed_bf16_exponent_base"
        )
        if isinstance(table, torch.Tensor):
            self.width = int(table.shape[1])
            self.shard = None
        else:
            self.width = int(
                table.packed_bf16_in_features if self.packed else table.weight.shape[1]
            )
            self.shard = getattr(table, "shard_indices", None)

    @property
    def weight(self) -> torch.Tensor:
        return self.table if isinstance(self.table, torch.Tensor) else self.table.weight

    def declare_kernels(self) -> None:
        """Declare the packed decode kernel ``local`` launches (TileLang raises
        on an undeclared call once the worker has prepared its kernels)."""
        if self.packed:
            from vllm.models.deepseek_v4_1.tilelang.embedding import declare_packed

            declare_packed(self.width, int(self.table.packed_bf16_exponent_base))

    @property
    def sharded(self) -> bool:
        """Whether some ids live on other ranks."""
        if self.shard is None:
            return False
        return not getattr(self.table, "disable_tp", False) and int(
            getattr(self.table, "tp_size", 1)
        ) > 1

    def local(self, ids: torch.Tensor) -> torch.Tensor:
        """``[N]`` global ids -> ``[N, width]`` rows owned here, zeros elsewhere."""
        ids = ids.reshape(-1).to(torch.int64).contiguous()
        weight = self.weight
        if self.packed:
            from vllm.models.deepseek_v4_1.tilelang.embedding import (
                vocab_embedding_packed,
            )

            s = self.shard
            out = torch.empty(
                (ids.numel(), self.width), dtype=torch.bfloat16, device=weight.device
            )
            vocab_embedding_packed(
                ids,
                weight,
                self.table.weight_exception_entries,
                self.table.weight_row_offsets,
                self.width,
                self.table.packed_bf16_exponent_base,
                out,
                s.org_vocab_start_index,
                s.org_vocab_end_index,
                s.num_org_vocab_padding,
                s.added_vocab_start_index,
                s.added_vocab_end_index,
            )
            return out
        if self.shard is None:
            return weight[ids]
        s = self.shard
        start, end = s.org_vocab_start_index, s.org_vocab_end_index
        added_start, added_end = s.added_vocab_start_index, s.added_vocab_end_index
        padding = int(getattr(s, "num_org_vocab_padding", 0))
        if triton is not None and ids.is_cuda and weight.is_cuda and ids.numel() > 0:
            n_ids = int(ids.numel())
            w_2d = weight.reshape(weight.shape[0], -1)
            out = torch.empty((n_ids, self.width), dtype=weight.dtype, device=weight.device)
            block_w = triton.next_power_of_2(self.width)
            _vocab_rows_local_kernel[(n_ids,)](
                ids,
                w_2d,
                out,
                w_2d.stride(0),
                out.stride(0),
                n_ids,
                width=self.width,
                BLOCK_W=block_w,
                org_start=start,
                org_end=end,
                added_start=added_start,
                added_end=added_end,
                added_base=(end - start + padding),
            )
            return out
        in_org = (ids >= start) & (ids < end)
        in_added = (ids >= added_start) & (ids < added_end)
        local_ids = torch.where(
            in_org,
            ids - start,
            torch.where(in_added, ids - added_start + (end - start + padding), 0),
        )
        rows = weight[local_ids]
        owned = (in_org | in_added).unsqueeze(-1)
        return torch.where(owned, rows, torch.zeros_like(rows))


class OnlineDSparkHeadAdapter:
    """Online verification-driven adapter for DSpark Markov and confidence heads.

    Updates ``markov_w1`` rows (the Markov embedding of the token before each
    draft), an optional ``hidden_to_markov_proj`` and the ``confidence_head``
    row with the prefix-product total variation (``E2E-TV``) over each
    position's candidate support and the auxiliary confidence ``BCE`` against
    the detached overlap, using closed-form gradients (no autograd graph).

    ``markov_w2`` is read, not written, unless ``update_w2`` asks for it, which
    is refused for a vocabulary-sharded or packed W2. Callers on a sharded W2
    pass the candidates' full rows (``w2_candidate_rows``, from ``VocabRows``
    and an all-reduce).

    With ``max_row_drift > 0`` the loaded W1 is kept and each touched row's
    displacement from it is bounded by ``max_row_drift`` times the loaded row's
    norm (and pulled back by ``anchor_decay`` per update); ``reset`` restores
    the loaded weights.
    """

    def __init__(
        self,
        markov_w1: nn.Embedding | torch.Tensor,
        markov_w2: nn.Module | torch.Tensor,
        confidence_head: nn.Module | torch.Tensor | None = None,
        *,
        hidden_to_markov_proj: nn.Module | torch.Tensor | None = None,
        lr_markov_w1: float = 0.08,
        lr_markov_w2: float = 0.04,
        lr_hidden_to_markov: float | None = None,
        lr_confidence: float = 0.05,
        bce_weight: float = 0.25,
        weight_decay: float = 0.0,
        max_grad_norm: float = 2.0,
        update_w2: bool = False,
        precondition_by_row_norm: bool = True,
        latent_noise_std: float = 0.0,
        noise_seed: int = 1729,
        calibrator: OnlineConfidenceCalibrator | None = None,
        max_row_drift: float = 0.0,
        anchor_decay: float = 0.0,
    ) -> None:
        self.markov_w1 = markov_w1
        self.markov_w2 = markov_w2
        self.confidence_head = confidence_head
        self.hidden_to_markov_proj = hidden_to_markov_proj
        self.lr_markov_w1 = float(lr_markov_w1)
        self.lr_markov_w2 = float(lr_markov_w2)
        self.lr_hidden_to_markov = (
            float(lr_markov_w1)
            if lr_hidden_to_markov is None
            else float(lr_hidden_to_markov)
        )
        self.lr_confidence = float(lr_confidence)
        self.bce_weight = float(bce_weight)
        self.weight_decay = float(weight_decay)
        self.max_grad_norm = float(max_grad_norm)
        self.w2_rows = VocabRows(markov_w2)
        if update_w2 and (self.w2_rows.packed or self.w2_rows.sharded):
            raise ValueError(
                "Online W2 updates are refused for a "
                + ("12-bit packed" if self.w2_rows.packed else "vocabulary-sharded")
                + " Markov W2: a sharded table indexed by global ids writes other "
                "rows, and a packed table has no BF16 rows to write"
            )
        self.update_w2 = bool(update_w2)
        self.precondition_by_row_norm = bool(precondition_by_row_norm)
        self.latent_noise_std = float(latent_noise_std)
        self.noise_seed = int(noise_seed)
        self.calibrator = calibrator
        self.max_row_drift = float(max_row_drift)
        self.anchor_decay = float(anchor_decay)
        self.step_count: int = 0
        self.last_metrics: dict[str, float] = {}
        # Loaded weights: the drift bound's anchor and what ``reset`` restores.
        self.w1_anchor: torch.Tensor | None = (
            self.w1_weight.detach().clone() if self.max_row_drift > 0.0 else None
        )
        conf = self.conf_weight
        self.conf_anchor: torch.Tensor | None = (
            conf.detach().clone() if conf is not None else None
        )

    @property
    def w1_weight(self) -> torch.Tensor:
        if isinstance(self.markov_w1, torch.Tensor):
            return self.markov_w1
        return self.markov_w1.weight

    @property
    def hidden_proj_weight(self) -> torch.Tensor | None:
        if self.hidden_to_markov_proj is None:
            return None
        if isinstance(self.hidden_to_markov_proj, torch.Tensor):
            return self.hidden_to_markov_proj
        return getattr(self.hidden_to_markov_proj, "weight", None)

    @property
    def conf_weight(self) -> torch.Tensor | None:
        if self.confidence_head is None:
            return None
        if isinstance(self.confidence_head, torch.Tensor):
            return self.confidence_head
        proj = getattr(self.confidence_head, "proj", None)
        if proj is not None and hasattr(proj, "weight"):
            return proj.weight
        return getattr(self.confidence_head, "weight", None)

    @torch.no_grad()
    def reset(self) -> None:
        """Restore the loaded W1 and confidence head (W1 needs ``max_row_drift``)."""
        if self.w1_anchor is not None:
            self.w1_weight.copy_(self.w1_anchor)
        conf = self.conf_weight
        if conf is not None and self.conf_anchor is not None:
            conf.copy_(self.conf_anchor)
        if self.calibrator is not None:
            self.calibrator.reset()
        self.step_count = 0

    def _project_hidden_to_markov(
        self, hidden_f: torch.Tensor, markov_rank: int
    ) -> torch.Tensor | None:
        proj_w = self.hidden_proj_weight
        if proj_w is None:
            return None
        steps = int(hidden_f.shape[1]) if hidden_f.ndim >= 2 else 1
        if self.latent_noise_std > 0.0 and steps > 1:
            denom_k = float(max(1, steps - 1))
            pos_f = torch.arange(steps, dtype=torch.float32, device=hidden_f.device)
            sigma_sched = (self.latent_noise_std * (pos_f / denom_k)).view(1, steps, 1)
            hidden_f = hidden_f * (1.0 / (1.0 + 16.0 * sigma_sched.pow(2)))
        pw = proj_w.float()
        if pw.shape == (hidden_f.shape[-1], markov_rank):
            return torch.matmul(hidden_f, pw)
        if pw.shape == (markov_rank, hidden_f.shape[-1]):
            return torch.matmul(hidden_f, pw.t())
        raise ValueError(
            f"hidden_to_markov_proj shape {tuple(pw.shape)} incompatible with "
            f"(hidden_size={hidden_f.shape[-1]}, markov_rank={markov_rank})"
        )

    def apply_position_scaled_latent_noise(
        self,
        latents: torch.Tensor,
        *,
        noise_std: float | None = None,
        generator: torch.Generator | None = None,
        return_sigma_schedule: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """``z_tilde_{b,i} = z_{b,i} + sigma_i * rms(z_{b,i}) * eps``, with
        ``sigma_i = sigma_0 * (i / K)`` and ``K = max(1, W - 1)``."""
        sigma_0 = self.latent_noise_std if noise_std is None else float(noise_std)
        steps = int(latents.shape[1])
        denom_k = float(max(1, steps - 1))
        pos = torch.arange(steps, dtype=torch.float32, device=latents.device)
        sigma_schedule = (sigma_0 * (pos / denom_k)).view(1, steps, 1)
        if sigma_0 <= 0.0 or steps <= 1:
            if return_sigma_schedule:
                return latents, sigma_schedule.view(steps)
            return latents
        if generator is None:
            generator = torch.Generator(device=latents.device)
            generator.manual_seed(
                int(self.noise_seed + self.step_count * 1009 + latents.shape[-1])
            )
        eps = torch.randn(
            latents.shape,
            dtype=torch.float32,
            device=latents.device,
            generator=generator,
        )
        rms = latents.float().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-4)
        perturbed = latents.float() + sigma_schedule * rms * eps
        perturbed = perturbed.to(dtype=latents.dtype)
        if return_sigma_schedule:
            return perturbed, sigma_schedule.view(steps)
        return perturbed

    def _gather_w2_candidate_rows(self, candidate_ids: torch.Tensor) -> torch.Tensor:
        """Full ``[batch, steps, topk, rank]`` W2 rows of the candidates from this
        rank alone: only valid when this rank holds the whole table."""
        if self.w2_rows.sharded:
            raise ValueError(
                "Markov W2 is vocabulary-sharded: pass w2_candidate_rows (this "
                "rank's VocabRows.local summed over the TP group)"
            )
        rows = self.w2_rows.local(candidate_ids.reshape(-1))
        return rows.view(*candidate_ids.shape, -1)

    def compute_candidate_logits_and_confidences(
        self,
        conditioning_token_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
        base_candidate_logits: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        *,
        w2_candidate_rows: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Drafter candidate logits on ``S_{b,i}`` and the confidence logits and
        probabilities at each position."""
        u = self.w1_weight[conditioning_token_ids].float()
        hidden_f = hidden_states.float() if hidden_states is not None else None
        m_query = u
        if hidden_f is not None:
            proj_delta = self._project_hidden_to_markov(hidden_f, int(u.shape[-1]))
            if proj_delta is not None:
                m_query = u + proj_delta
        w2_f = (
            w2_candidate_rows
            if w2_candidate_rows is not None
            else self._gather_w2_candidate_rows(candidate_ids)
        ).float()
        draft_candidate_logits = base_candidate_logits.float() + _rowwise_dot(
            m_query.unsqueeze(2), w2_f
        )
        conf_logits = conf_probs = None
        conf_w = self.conf_weight
        if conf_w is not None and hidden_f is not None:
            feat = torch.cat([hidden_f, m_query], dim=-1)
            conf_logits = _rowwise_dot(feat, conf_w.float().view(-1))
            conf_probs = torch.sigmoid(conf_logits)
            if self.calibrator is not None:
                conf_probs = self.calibrator.calibrate(conf_probs)
        return draft_candidate_logits, conf_logits, conf_probs

    @torch.no_grad()
    def _bound_w1_rows(self, sorted_ids: torch.Tensor) -> None:
        """Pull the touched W1 rows toward the loaded table and bound their
        displacement (every duplicate writes the same value)."""
        if self.w1_anchor is None or sorted_ids.numel() == 0:
            return
        w1 = self.w1_weight
        rows = w1[sorted_ids].float()
        anchor = self.w1_anchor[sorted_ids].float()
        disp = rows - anchor
        if self.anchor_decay > 0.0:
            disp = disp * (1.0 - self.anchor_decay)
        limit = self.max_row_drift * anchor.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        norm = disp.norm(dim=-1, keepdim=True)
        disp = disp * torch.clamp(limit / norm.clamp_min(1e-12), max=1.0)
        w1.index_copy_(0, sorted_ids, (anchor + disp).to(dtype=w1.dtype))

    @torch.no_grad()
    def step_on_candidates(
        self,
        conditioning_token_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
        base_candidate_logits: torch.Tensor,
        target_candidate_probs: torch.Tensor,
        *,
        hidden_states: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        accepted_counts: torch.Tensor | None = None,
        verified_counts: torch.Tensor | None = None,
        w2_candidate_rows: torch.Tensor | None = None,
        return_metrics: bool = True,
    ) -> dict[str, float]:
        """One online ``E2E-TV`` + ``BCE`` step on the DSpark heads, in place.

        ``conditioning_token_ids`` ([B, W]) are the tokens before each draft
        position, ``candidate_ids`` ([B, W, K]) the support, and
        ``base_candidate_logits`` the drafter's base logits there. Without
        ``return_metrics`` it makes no host synchronization.
        """
        bsz, steps, topk = candidate_ids.shape
        if bsz == 0 or steps == 0 or topk == 0:
            return self.last_metrics

        w1 = self.w1_weight
        u = w1[conditioning_token_ids].float()
        markov_rank = int(u.shape[-1])

        denom_k = float(max(1, steps - 1))
        pos_f = torch.arange(steps, dtype=torch.float32, device=candidate_ids.device)
        sigma_schedule = (self.latent_noise_std * (pos_f / denom_k)).view(1, steps, 1)

        hidden_f: torch.Tensor | None = None
        if hidden_states is not None:
            if self.latent_noise_std > 0.0 and steps > 1:
                hidden_f = self.apply_position_scaled_latent_noise(
                    hidden_states.float(), noise_std=0.15 * self.latent_noise_std
                )
            else:
                hidden_f = hidden_states.float()

        proj_delta = (
            self._project_hidden_to_markov(hidden_f, markov_rank)
            if hidden_f is not None
            else None
        )
        m_query = u + proj_delta if proj_delta is not None else u
        if self.latent_noise_std > 0.0 and steps > 1 and proj_delta is None:
            m_for_forward = self.apply_position_scaled_latent_noise(
                m_query, noise_std=0.15 * self.latent_noise_std
            )
        else:
            m_for_forward = m_query

        w2_is_local = w2_candidate_rows is None
        w2_f = (
            self._gather_w2_candidate_rows(candidate_ids)
            if w2_is_local
            else w2_candidate_rows
        ).float()

        markov_logits = _rowwise_dot(m_for_forward.unsqueeze(2), w2_f)
        draft_logits = base_candidate_logits.float() + markov_logits

        conf_w = self.conf_weight
        conf_logits: torch.Tensor | None = None
        feat: torch.Tensor | None = None
        want_conf = (
            conf_w is not None
            and hidden_f is not None
            and (self.lr_confidence > 0.0 or self.bce_weight > 0.0 or self.calibrator is not None)
        )
        if want_conf:
            feat = torch.cat([hidden_f, m_for_forward], dim=-1)
            conf_logits = _rowwise_dot(feat, conf_w.float().view(-1))

        loss_res = compute_e2e_tv_and_bce_loss(
            draft_logits,
            target_candidate_probs,
            confidence_logits=conf_logits,
            valid_mask=valid_mask,
            bce_weight=self.bce_weight,
        )

        # dL/dd_{b,i} = (1 / N) * sum_{j >= i} prod_{m <= j, m != i} alpha_{b,m}.
        alpha = loss_res.token_overlap
        if valid_mask is not None:
            mask_b = valid_mask.bool()
            mask_f = mask_b.to(dtype=alpha.dtype)
            eff_alpha = torch.where(mask_b, alpha, torch.ones_like(alpha))
            denom = mask_f.sum().clamp_min(1.0)
        else:
            mask_f = torch.ones_like(alpha)
            eff_alpha = alpha
            denom = float(bsz * steps)

        idx = torch.arange(steps, device=alpha.device)
        cond_m = (idx.view(1, 1, -1) <= idx.view(1, -1, 1)) & (
            idx.view(1, 1, -1) != idx.view(-1, 1, 1)
        )
        factors = torch.where(
            cond_m.unsqueeze(0),
            eff_alpha[:, None, None, :],
            torch.ones((1, 1, 1, 1), dtype=alpha.dtype, device=alpha.device),
        )
        prod_ij = factors.prod(dim=-1)
        upper_mask = (idx.view(-1, 1) <= idx.view(1, -1)).to(dtype=alpha.dtype)
        omega = (prod_ij * upper_mask.unsqueeze(0) * mask_f.unsqueeze(1)).sum(
            dim=-1
        ) * mask_f

        # dd_{b,i}/dq_{b,i}(v) = 0.5 * sign(q - p), through the softmax.
        q_probs = F.softmax(draft_logits, dim=-1)
        p_probs = target_candidate_probs.float().clamp_min(0.0)
        sign_qp = 0.5 * torch.sign(q_probs - p_probs)
        expected_sign = (q_probs * sign_qp).sum(dim=-1, keepdim=True)
        d_dist_d_logit = q_probs * (sign_qp - expected_sign)

        if self.precondition_by_row_norm:
            eff_omega = torch.maximum(omega, 0.65 * mask_f)
            step_grad_logits = eff_omega.unsqueeze(-1) * d_dist_d_logit
            w2_row_energy = (
                (w2_f * w2_f).sum(dim=-1).mean(dim=-1, keepdim=True).clamp_min(1e-3)
            )
            u_energy = (
                (m_for_forward * m_for_forward)
                .sum(dim=-1, keepdim=True)
                .unsqueeze(-1)
                .clamp_min(0.05)
            )
            grad_m = (step_grad_logits.unsqueeze(-1) * w2_f).sum(dim=2) / w2_row_energy
            grad_w2 = (
                step_grad_logits.unsqueeze(-1) * m_for_forward.unsqueeze(2)
            ) / u_energy
        else:
            grad_logits = (omega / denom).unsqueeze(-1) * d_dist_d_logit
            grad_m = (grad_logits.unsqueeze(-1) * w2_f).sum(dim=2)
            grad_w2 = grad_logits.unsqueeze(-1) * m_for_forward.unsqueeze(2)

        # Only observed positions move anything.
        grad_m = grad_m * mask_f.unsqueeze(-1)
        grad_u = grad_m
        grad_u_norm = grad_u.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        grad_u = grad_u * torch.clamp(self.max_grad_norm / grad_u_norm, max=1.0)
        if self.weight_decay > 0.0:
            grad_u = grad_u + self.weight_decay * u * mask_f.unsqueeze(-1)

        flat_cond_ids = conditioning_token_ids.reshape(-1)
        flat_delta_u = (-self.lr_markov_w1 * grad_u).reshape(-1, grad_u.shape[-1])
        sorted_ids = _deterministic_index_add_(w1, flat_cond_ids, flat_delta_u)
        self._bound_w1_rows(sorted_ids)

        proj_w = self.hidden_proj_weight
        if (
            proj_w is not None
            and hidden_f is not None
            and self.lr_hidden_to_markov > 0.0
        ):
            step_precision = 1.0 / (1.0 + 16.0 * sigma_schedule.pow(2))
            smoothed_hidden = hidden_f * step_precision
            noise_var_weight = (
                3.5 * (sigma_schedule.pow(2) * mask_f.unsqueeze(-1)).mean()
            )
            grad_proj_hr = (smoothed_hidden.unsqueeze(-1) * grad_m.unsqueeze(-2)).sum(
                dim=(0, 1)
            ) / float(max(1, bsz * steps))
            if self.precondition_by_row_norm:
                h_energy = (hidden_f * hidden_f).sum(dim=-1).mean().clamp_min(1e-2)
                grad_proj_hr = grad_proj_hr / h_energy
            pw_f = proj_w.float()
            if pw_f.shape == grad_proj_hr.shape:
                grad_proj = grad_proj_hr + noise_var_weight * pw_f
            else:
                grad_proj = grad_proj_hr.t() + noise_var_weight * pw_f
            grad_proj_norm = grad_proj.norm().clamp_min(1e-8)
            grad_proj = grad_proj * torch.clamp(
                self.max_grad_norm / grad_proj_norm, max=1.0
            )
            proj_w.add_((-self.lr_hidden_to_markov * grad_proj).to(dtype=proj_w.dtype))

        if self.update_w2 and w2_is_local and self.lr_markov_w2 > 0.0:
            grad_w2 = grad_w2 * mask_f[..., None, None]
            grad_w2_norm = grad_w2.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            grad_w2 = grad_w2 * torch.clamp(self.max_grad_norm / grad_w2_norm, max=1.0)
            flat_delta_w2 = (-self.lr_markov_w2 * grad_w2).reshape(
                -1, grad_w2.shape[-1]
            )
            _deterministic_index_add_(
                self.w2_rows.weight, candidate_ids.reshape(-1), flat_delta_w2
            )

        # Confidence head: BCE gradient against the detached overlap.
        if conf_w is not None and conf_logits is not None and feat is not None:
            conf_probs_raw = torch.sigmoid(conf_logits)
            detached_overlap = alpha.detach()
            grad_conf_logit = (
                (conf_probs_raw - detached_overlap) * mask_f / denom
            ) * self.bce_weight
            grad_conf_w = (grad_conf_logit.unsqueeze(-1) * feat).sum(dim=(0, 1))
            if self.precondition_by_row_norm:
                feat_energy = (feat * feat).sum(dim=-1).mean().clamp_min(1.0)
                grad_conf_w = grad_conf_w / feat_energy
            if self.latent_noise_std > 0.0 and hidden_f is not None and steps > 1:
                h_dim = int(hidden_f.shape[-1])
                sigma_sq = sigma_schedule.squeeze(-1).pow(2)
                noise_var_conf = 2.5 * (sigma_sq * mask_f).mean()
                conf_h = conf_w.float().view(-1)[:h_dim]
                grad_conf_w[:h_dim] = grad_conf_w[:h_dim] + noise_var_conf * conf_h
            if self.lr_confidence > 0.0:
                grad_conf_norm = grad_conf_w.norm().clamp_min(1e-8)
                grad_conf_w = grad_conf_w * torch.clamp(
                    self.max_grad_norm / grad_conf_norm, max=1.0
                )
                conf_w.view(-1).add_(
                    (-self.lr_confidence * grad_conf_w).to(dtype=conf_w.dtype)
                )

            if self.calibrator is not None and accepted_counts is not None:
                self.calibrator.observe(
                    conf_probs_raw,
                    accepted_counts,
                    verified_counts=verified_counts,
                    return_metrics=return_metrics,
                )

        self.step_count += 1
        if not return_metrics:
            return self.last_metrics

        vals = torch.stack(
            [
                loss_res.total_loss,
                loss_res.e2e_tv_loss,
                loss_res.bce_loss,
                loss_res.token_overlap.mean(),
                loss_res.prefix_survival.sum(dim=-1).mean(),
                loss_res.excluded_teacher_mass.mean(),
            ]
        ).tolist()
        self.last_metrics = {
            "total_loss": float(vals[0]),
            "e2e_tv_loss": float(vals[1]),
            "bce_loss": float(vals[2]),
            "mean_overlap": float(vals[3]),
            "mean_prefix_survival": float(vals[4]),
            "excluded_teacher_mass": float(vals[5]),
        }
        return self.last_metrics

    @staticmethod
    def verified_support(
        draft_tokens: torch.Tensor,
        verified_target_tokens: torch.Tensor,
        vocab_size: int,
        runner_up_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``[B, W, 3]`` support at each position: the target's token, the draft
        (or the next id when they agree), the runner-up (or another id), all
        distinct."""
        steps = draft_tokens.shape[1]
        tgt = verified_target_tokens[:, :steps].to(torch.int64).clamp(0, vocab_size - 1)
        drf = draft_tokens[:, :steps].to(torch.int64).clamp(0, vocab_size - 1)
        alt1 = torch.where(drf != tgt, drf, (tgt + 1) % vocab_size)
        if runner_up_tokens is not None:
            rup = runner_up_tokens[:, :steps].to(torch.int64).clamp(0, vocab_size - 1)
            distinct = (rup != tgt) & (rup != alt1)
            alt2 = torch.where(distinct, rup, (tgt + 2) % vocab_size)
        else:
            alt2 = (tgt + 2) % vocab_size
        alt2 = torch.where(alt2 == alt1, (tgt + 3) % vocab_size, alt2)
        return torch.stack([tgt, alt1, alt2], dim=-1)

    @staticmethod
    def verified_target_probs(
        shape: tuple[int, int],
        device: torch.device,
        *,
        target_peak_prob: float = 0.92,
        excluded_teacher_mass: float = 0.03,
    ) -> torch.Tensor:
        """The support's target probabilities: greedy verification exposes only
        the target's token, so it gets ``target_peak_prob`` and the other two
        share what remains of ``1 - excluded_teacher_mass``."""
        included = max(0.0, min(1.0, 1.0 - float(excluded_teacher_mass)))
        peak = min(included, float(target_peak_prob))
        rem = max(0.0, included - peak) * 0.5
        probs = torch.empty((*shape, 3), dtype=torch.float32, device=device)
        probs[..., 0] = peak
        probs[..., 1] = rem
        probs[..., 2] = rem
        return probs

    @torch.no_grad()
    def step_from_verified_tokens(
        self,
        conditioning_token_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        verified_target_tokens: torch.Tensor,
        accepted_counts: torch.Tensor,
        *,
        runner_up_tokens: torch.Tensor | None = None,
        base_candidate_logits: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        verified_counts: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        candidate_features: Callable[
            [torch.Tensor], tuple[torch.Tensor, torch.Tensor]
        ]
        | None = None,
        target_peak_prob: float = 0.92,
        excluded_teacher_mass: float = 0.03,
        return_metrics: bool = True,
    ) -> dict[str, float]:
        """Adapt from realized verification on the draft branch.

        The support at each observed position (``i <= A_b`` and ``i < B_b``, or
        ``valid_mask``) is ``verified_support``'s, with
        ``verified_target_probs``. ``candidate_features(candidate_ids)`` returns
        the candidates' full W2 rows and base logits (``DSparkOnlineAdaptation``
        computes them from the vocabulary shards); without it the base logits
        are ``base_candidate_logits`` or zero and W2 is read locally.
        """
        bsz, steps = draft_tokens.shape
        device = draft_tokens.device
        vocab_size = int(self.w1_weight.shape[0])
        if valid_mask is None:
            acc = accepted_counts[:bsz].to(device=device, dtype=torch.int64)
            acc = acc.clamp_min(0)
            if verified_counts is None:
                ver = torch.full((bsz,), steps, dtype=torch.int64, device=device)
            else:
                ver = verified_counts[:bsz].to(device=device, dtype=torch.int64)
                ver = ver.clamp(0, steps)
            pos = torch.arange(steps, dtype=torch.int64, device=device).unsqueeze(0)
            valid_mask = (pos <= acc.unsqueeze(1)) & (pos < ver.unsqueeze(1))

        candidate_ids = self.verified_support(
            draft_tokens, verified_target_tokens, vocab_size, runner_up_tokens
        )
        w2_candidate_rows = None
        if candidate_features is not None:
            w2_candidate_rows, base_candidate_logits = candidate_features(candidate_ids)
        elif base_candidate_logits is None:
            base_candidate_logits = torch.zeros(
                (bsz, steps, 3), dtype=torch.float32, device=device
            )
        target_probs = self.verified_target_probs(
            (bsz, steps),
            device,
            target_peak_prob=target_peak_prob,
            excluded_teacher_mass=excluded_teacher_mass,
        )
        return self.step_on_candidates(
            conditioning_token_ids[:, :steps].to(torch.int64).clamp(0, vocab_size - 1),
            candidate_ids,
            base_candidate_logits,
            target_probs,
            hidden_states=hidden_states,
            valid_mask=valid_mask,
            accepted_counts=accepted_counts,
            verified_counts=verified_counts,
            w2_candidate_rows=w2_candidate_rows,
            return_metrics=return_metrics,
        )


def _bits_digest(tensor: torch.Tensor, chunk: int = 1 << 21) -> torch.Tensor:
    """A position-weighted sum of the tensor's bit patterns (int64, wrapping):
    any changed bit changes it, and integer sums are order-independent."""
    flat = tensor.detach().reshape(-1)
    if flat.dtype in (torch.bfloat16, torch.float16):
        bits = flat.view(torch.int16)
        mask = 0xFFFF
    elif flat.dtype == torch.float32:
        bits = flat.view(torch.int32)
        mask = 0xFFFFFFFF
    elif flat.dtype == torch.float64:
        bits = flat.view(torch.int64)
        mask = -1
    else:
        bits = flat.to(torch.int64)
        mask = -1
    total = torch.zeros((), dtype=torch.int64, device=flat.device)
    for start in range(0, bits.numel(), chunk):
        part = bits[start : start + chunk].to(torch.int64)
        if mask != -1:
            part = part & mask
        index = torch.arange(
            start, start + part.numel(), dtype=torch.int64, device=flat.device
        )
        weight = (index * 2654435761 + 40503) % 2147483629 + 1
        total = total + (part * weight).sum()
    return total


class DSparkOnlineAdaptation:
    """Part 2 wired to a DSpark speculator: the previous step's drafts, graded
    by this step's verification, update the replicated Markov W1 and the
    confidence head.

    ``capture`` runs inside the captured draft step and keeps what the update
    needs that the next step would overwrite or never sees: each request's head
    hidden rows, its anchor token and its request slot, in draft-batch order.
    ``arm`` marks a real (not warmup or dummy) draft. ``observe`` runs once per
    verification, before the next draft, on every rank in the same order, with
    one all-reduce (when ``all_reduce`` is given) whatever the step's outcomes.
    """

    def __init__(
        self,
        *,
        markov_w1: nn.Embedding,
        markov_w2: nn.Module | torch.Tensor,
        lm_head: nn.Module | torch.Tensor,
        norm: Callable[[torch.Tensor], torch.Tensor],
        confidence_head: nn.Module,
        max_num_reqs: int,
        num_steps: int,
        hidden_size: int,
        device: torch.device,
        logits_scale: float = 1.0,
        soft_cap: float | None = None,
        all_reduce: Callable[[torch.Tensor], torch.Tensor] | None = None,
        update_w2: bool = False,
        max_row_drift: float = 1.0,
        digest_interval: int = 0,
        tp_rank: int = 0,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 1e-6,
        lr_markov_w1: float = 0.08,
        lr_confidence: float = 0.0,
        bce_weight: float = 0.0,
        logger: Any = None,
    ) -> None:
        if update_w2 and all_reduce is not None:
            raise ValueError("Online W2 updates are refused under tensor parallelism")
        self.heads = OnlineDSparkHeadAdapter(
            markov_w1,
            markov_w2,
            confidence_head,
            lr_markov_w1=lr_markov_w1,
            lr_confidence=lr_confidence,
            bce_weight=bce_weight,
            update_w2=update_w2,
            max_row_drift=max_row_drift,
        )
        self.w2_rows = VocabRows(markov_w2)
        self.lm_rows = VocabRows(lm_head)
        self.w2_rows.declare_kernels()
        self.lm_rows.declare_kernels()
        self.norm = norm
        self.max_num_reqs = int(max_num_reqs)
        self.num_steps = int(num_steps)
        self.hidden_size = int(hidden_size)
        self.vocab_size = int(self.heads.w1_weight.shape[0])
        self.logits_scale = float(logits_scale)
        self.soft_cap = soft_cap
        self.all_reduce = all_reduce
        self.digest_interval = int(digest_interval)
        self.tp_rank = int(tp_rank)
        self.logger = logger
        self.device = device
        self.hidden = torch.zeros(
            (self.max_num_reqs * self.num_steps, self.hidden_size),
            dtype=torch.bfloat16,
            device=device,
        )
        self.anchor = torch.zeros(self.max_num_reqs, dtype=torch.int64, device=device)
        self.slots = torch.full(
            (self.max_num_reqs,), -1, dtype=torch.int64, device=device
        )
        if norm_weight is None and isinstance(norm, functools.partial):
            kw = norm.keywords or {}
            if norm.func is rms_norm and "weight" in kw:
                norm_weight = kw["weight"]
                norm_eps = float(kw.get("eps", norm_eps))
        self.norm_weight = norm_weight
        self.norm_eps = float(norm_eps)
        self._slot_to_row = torch.full(
            (self.max_num_reqs + 1,), -1, dtype=torch.int64, device=device
        )
        self._rows = torch.arange(self.max_num_reqs, dtype=torch.int64, device=device)
        self._pos = torch.arange(self.num_steps, dtype=torch.int64, device=device)
        rank = int(self.heads.w1_weight.shape[1])
        padded_max = max(8, -(-(self.max_num_reqs * 3) // 8) * 8)
        self._miss_mask = torch.zeros(self.max_num_reqs, dtype=torch.int32, device=device)
        self._cond_ids = torch.zeros(self.max_num_reqs, dtype=torch.int64, device=device)
        self._packed = torch.zeros((padded_max, 1 + rank), dtype=torch.float32, device=device)
        self._dummy_i32 = torch.zeros(2, dtype=torch.int32, device=device)
        self._block_s = min(256, max(16, triton.next_power_of_2(self.max_num_reqs))) if triton is not None else 16
        self._block_r = triton.next_power_of_2(rank) if triton is not None else 256
        self._block_h = min(1024, max(128, triton.next_power_of_2(self.hidden_size))) if triton is not None else 1024
        self._block_cap_h = min(2048, triton.next_power_of_2(self.hidden_size)) if triton is not None else 2048
        lm_s = self.lm_rows.shard
        self._lm_sharded = lm_s is not None
        if lm_s is not None:
            self._lm_os = int(lm_s.org_vocab_start_index)
            self._lm_oe = int(lm_s.org_vocab_end_index)
            self._lm_as = int(lm_s.added_vocab_start_index)
            self._lm_ae = int(lm_s.added_vocab_end_index)
            self._lm_ab = (self._lm_oe - self._lm_os) + int(getattr(lm_s, "num_org_vocab_padding", 0))
        else:
            self._lm_os = self._lm_oe = self._lm_as = self._lm_ae = self._lm_ab = 0
        self._lm_exp_base = int(self.lm_rows.table.packed_bf16_exponent_base) if self.lm_rows.packed else 0
        w2_s = self.w2_rows.shard
        self._w2_sharded = w2_s is not None
        if w2_s is not None:
            self._w2_os = int(w2_s.org_vocab_start_index)
            self._w2_oe = int(w2_s.org_vocab_end_index)
            self._w2_as = int(w2_s.added_vocab_start_index)
            self._w2_ae = int(w2_s.added_vocab_end_index)
            self._w2_ab = (self._w2_oe - self._w2_os) + int(getattr(w2_s, "num_org_vocab_padding", 0))
        else:
            self._w2_os = self._w2_oe = self._w2_as = self._w2_ae = self._w2_ab = 0
        self._armed = False
        self.updates = 0

    # -- draft side (captured) ------------------------------------------------

    def capture(
        self,
        num_reqs: int,
        sample_hidden: torch.Tensor,
        anchor_ids: torch.Tensor,
        slots: torch.Tensor,
    ) -> None:
        """Keep the draft's head hidden rows ([num_reqs * W, H], (request, step)
        order), anchors and request slots. Fixed-shape copies only, so it is
        safe inside a captured graph; rows past ``num_reqs`` are marked empty."""
        n = num_reqs * self.num_steps
        if triton is not None and self.device.type == "cuda":
            _capture_kernel[(max(n, self.max_num_reqs),)](
                self.hidden,
                self.hidden.stride(0),
                sample_hidden,
                sample_hidden.stride(0),
                self.anchor,
                anchor_ids,
                self.slots,
                slots,
                num_reqs,
                n,
                self.max_num_reqs,
                hidden_size=self.hidden_size,
                BLOCK_H=self._block_cap_h,
            )
            return
        self.hidden[:n].copy_(sample_hidden[:n])
        self.anchor[:num_reqs].copy_(anchor_ids[:num_reqs])
        self.slots[:num_reqs].copy_(slots[:num_reqs])
        if num_reqs < self.max_num_reqs:
            self.slots[num_reqs:].fill_(-1)

    def arm(self, drafted: bool) -> None:
        """After a draft step: whether it drafted for real requests."""
        self._armed = bool(drafted)

    # -- verification side (eager) --------------------------------------------

    def _candidate_features(
        self, hidden: torch.Tensor
    ) -> Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        def features(candidate_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            rows, steps, k = candidate_ids.shape
            flat = candidate_ids.reshape(-1)
            w2_local = self.w2_rows.local(flat).float()
            lm_local = self.lm_rows.local(flat)
            normed = self.norm(hidden.reshape(rows * steps, -1).contiguous())
            normed = normed.float().repeat_interleave(k, dim=0)
            base_local = _rowwise_dot(lm_local, normed).unsqueeze(-1)
            packed = torch.cat([base_local, w2_local], dim=-1)
            if self.all_reduce is not None:
                # Whole multiples of eight rows: the rank-order all-reduce then
                # pads nothing itself (its padding is a TileLang fill).
                n = packed.shape[0]
                padded = -(-n // 8) * 8
                if padded != n:
                    packed = F.pad(packed, (0, 0, 0, padded - n))
                packed = self.all_reduce(packed.contiguous())[:n]
            base = packed[:, 0].view(rows, steps, k)
            if self.logits_scale != 1.0:
                base = base * self.logits_scale
            if self.soft_cap is not None:
                base = self.soft_cap * torch.tanh(base / self.soft_cap)
            w2 = packed[:, 1:].reshape(rows, steps, k, -1)
            return w2, base

        return features

    def _can_use_triton_fast_path(self, num_reqs: int) -> bool:
        return (
            triton is not None
            and self.device.type == "cuda"
            and num_reqs <= self.max_num_reqs
            and self.norm_weight is not None
            and not self.w2_rows.packed
            and self.heads.hidden_to_markov_proj is None
            and self.heads.latent_noise_std == 0.0
            and not self.heads.update_w2
            and self.heads.precondition_by_row_norm
            and self.heads.lr_confidence <= 0.0
            and self.heads.calibrator is None
        )

    def _observe_cuda_triton(
        self,
        num_reqs: int,
        idx_mapping: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        sampled_token_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        dead_row_cut: torch.Tensor | None,
        copy_int_state: torch.Tensor | None,
        runner_up_tokens: torch.Tensor | None,
    ) -> None:
        steps = self.num_steps
        width = int(sampled_token_ids.shape[1])
        n_cands = num_reqs * 3
        padded = -(-n_cands // 8) * 8
        miss_mask = self._miss_mask[:num_reqs]
        cond_ids = self._cond_ids[:num_reqs]
        packed = self._packed[:padded]

        lm_packed = self.lm_rows.packed
        lm_w = self.lm_rows.weight
        if lm_packed:
            lm_exc = self.lm_rows.table.weight_exception_entries
            lm_off = self.lm_rows.table.weight_row_offsets
        else:
            lm_exc = self._dummy_i32
            lm_off = self._dummy_i32
            if lm_w.ndim != 2:
                lm_w = lm_w.reshape(lm_w.shape[0], -1)

        w2_w = self.w2_rows.weight
        rank = int(self.heads.w1_weight.shape[1])
        _prepare_and_pack_miss_kernel[(padded,)](
            idx_mapping,
            self.slots,
            num_sampled,
            num_rejected,
            dead_row_cut if dead_row_cut is not None else num_sampled,
            copy_int_state if copy_int_state is not None else num_sampled,
            copy_int_state.stride(0) if copy_int_state is not None else 1,
            sampled_token_ids,
            sampled_token_ids.stride(0),
            draft_tokens,
            draft_tokens.stride(0),
            self.anchor,
            runner_up_tokens if runner_up_tokens is not None else draft_tokens,
            runner_up_tokens.stride(0) if runner_up_tokens is not None else 1,
            self.hidden,
            self.hidden.stride(0),
            self.norm_weight,
            lm_w,
            lm_w.stride(0),
            lm_exc,
            lm_off,
            w2_w,
            w2_w.stride(0),
            miss_mask,
            cond_ids,
            packed,
            packed.stride(0),
            num_reqs,
            n_cands,
            self.max_num_reqs,
            steps=steps,
            width=width,
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            rank=rank,
            norm_eps=self.norm_eps,
            HAS_DEAD_CUT=dead_row_cut is not None,
            HAS_COPY_STATE=copy_int_state is not None,
            HAS_RUNNER_UP=runner_up_tokens is not None,
            LM_PACKED=lm_packed,
            LM_EXP_BASE=self._lm_exp_base,
            LM_SHARDED=self._lm_sharded,
            lm_org_start=self._lm_os,
            lm_org_end=self._lm_oe,
            lm_added_start=self._lm_as,
            lm_added_end=self._lm_ae,
            lm_added_base=self._lm_ab,
            W2_SHARDED=self._w2_sharded,
            w2_org_start=self._w2_os,
            w2_org_end=self._w2_oe,
            w2_added_start=self._w2_as,
            w2_added_end=self._w2_ae,
            w2_added_base=self._w2_ab,
            BLOCK_S=self._block_s,
            BLOCK_H=self._block_h,
            BLOCK_R=self._block_r,
        )

        if self.all_reduce is not None:
            packed = self.all_reduce(packed)

        w1 = self.heads.w1_weight
        anchor = self.heads.w1_anchor
        has_cap = self.soft_cap is not None
        soft_cap_val = float(self.soft_cap) if has_cap else 1.0
        _miss_update_w1_kernel[(num_reqs,)](
            miss_mask,
            cond_ids,
            packed,
            packed.stride(0),
            w1,
            w1.stride(0),
            anchor if anchor is not None else w1,
            anchor.stride(0) if anchor is not None else w1.stride(0),
            num_reqs,
            self.logits_scale,
            soft_cap_val,
            0.92,
            0.025,
            self.heads.max_grad_norm,
            self.heads.weight_decay,
            self.heads.lr_markov_w1,
            self.heads.anchor_decay,
            self.heads.max_row_drift,
            HAS_SOFT_CAP=has_cap,
            HAS_ANCHOR=anchor is not None,
            rank=rank,
            BLOCK_R=self._block_r,
        )

    @torch.no_grad()
    def observe(
        self,
        idx_mapping: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        sampled_token_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        *,
        dead_row_cut: torch.Tensor | None = None,
        copy_int_state: torch.Tensor | None = None,
        runner_up_tokens: torch.Tensor | None = None,
    ) -> None:
        """Grade the armed draft with this step's verification and update.

        Only missed verified tokens (``pos == acc``, ``rej > 0``,
        ``target != draft``) update W1; the confidence head is left untouched.
        """
        if not self._armed:
            return
        self._armed = False
        num_reqs = int(idx_mapping.shape[0])
        if num_reqs == 0:
            return
        if self._can_use_triton_fast_path(num_reqs):
            self._observe_cuda_triton(
                num_reqs,
                idx_mapping,
                num_sampled,
                num_rejected,
                sampled_token_ids,
                draft_tokens,
                dead_row_cut,
                copy_int_state,
                runner_up_tokens,
            )
        else:
            steps = self.num_steps
            dump = self.max_num_reqs
            self._slot_to_row.fill_(-1)
            self._slot_to_row.scatter_(
                0, torch.where(self.slots >= 0, self.slots, dump), self._rows
            )
            slots = idx_mapping[:num_reqs].to(torch.int64)
            safe_slots = slots.clamp(0, dump - 1)
            row = torch.where(
                (slots >= 0) & (slots < dump), self._slot_to_row[safe_slots], -1
            )
            valid = row >= 0
            if copy_int_state is not None:
                valid = valid & (copy_int_state[safe_slots, 1] <= 0)
            observed, labels = verification_outcomes(
                steps, num_sampled[:num_reqs], num_rejected[:num_reqs], dead_row_cut
            )
            width = int(sampled_token_ids.shape[1])
            safe_row = row.clamp_min(0)
            drafts = draft_tokens[safe_row, :steps].to(torch.int64)
            cond = torch.cat([self.anchor[safe_row, None], drafts[:, :-1]], dim=1)
            targets = sampled_token_ids[:num_reqs, : min(width, steps)].to(torch.int64)
            if width < steps:
                targets = F.pad(targets, (0, steps - width), value=-1)
            missed = (
                observed
                & (~labels)
                & valid[:, None]
                & (self._pos[None, :] < width)
                & (targets >= 0)
                & (drafts >= 0)
                & (targets != drafts)
            )
            hidden = self.hidden.view(self.max_num_reqs, steps, -1)[safe_row]
            runner_up = (
                runner_up_tokens[safe_row, :steps]
                if runner_up_tokens is not None
                else None
            )
            self.heads.step_from_verified_tokens(
                cond.clamp(0, self.vocab_size - 1),
                drafts,
                targets,
                num_sampled[:num_reqs] - 1,
                runner_up_tokens=runner_up,
                hidden_states=hidden,
                valid_mask=missed,
                candidate_features=self._candidate_features(hidden),
                return_metrics=False,
            )
        self.updates += 1
        if self.digest_interval and self.updates % self.digest_interval == 0:
            self.log_digest()

    def state_digest(self) -> dict[str, int]:
        """Bit-pattern digests of the adapted state (host synchronization)."""
        conf = self.heads.conf_weight
        values = {"w1": _bits_digest(self.heads.w1_weight)}
        if conf is not None:
            values["conf"] = _bits_digest(conf)
        mask = 0xFFFFFFFFFFFFFFFF
        return {name: int(value.item()) & mask for name, value in values.items()}

    def drift(self, chunk: int = 8192) -> float:
        """Largest W1 row displacement from the loaded table over that row's norm
        (in row chunks, so the temporaries stay a few MiB)."""
        anchor = self.heads.w1_anchor
        if anchor is None:
            return float("nan")
        w1 = self.heads.w1_weight
        worst = torch.zeros((), dtype=torch.float32, device=w1.device)
        for start in range(0, w1.shape[0], chunk):
            a = anchor[start : start + chunk].float()
            disp = (w1[start : start + chunk].float() - a).norm(dim=-1)
            worst = torch.maximum(worst, (disp / a.norm(dim=-1).clamp_min(1e-6)).max())
        return float(worst.item())

    def log_digest(self, extra: dict[str, int] | None = None) -> None:
        if self.logger is None:
            return
        digest = self.state_digest()
        if extra:
            digest.update(extra)
        self.logger.info(
            "DSpark online adaptation digest: rank=%d update=%d %s drift=%.6g",
            self.tp_rank,
            self.updates,
            " ".join(f"{k}={v:016x}" for k, v in sorted(digest.items())),
            self.drift(),
        )

    def reset(self) -> None:
        self.heads.reset()
        self._armed = False

class SessionTokenMemory:
    """Bounded per-request rings of verified tokens on the GPU.

    Slot ``s`` holds the last ``capacity`` verified tokens of the request in
    request slot ``s``; ``reset_slots`` clears a slot when a new request takes
    it, so nothing carries over between requests. ``CopyDrafter`` consults a
    request's ring when the request's own history has no copy match.
    """

    def __init__(
        self,
        capacity: int = 4096,
        *,
        max_slots: int = 1,
        min_match_len: int = 3,
        max_match_len: int = 8,
        base_confidence: float = 0.85,
        confidence_decay: float = 0.92,
        device: torch.device | str = "cpu",
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if max_slots <= 0:
            raise ValueError(f"max_slots must be positive, got {max_slots}")
        self.capacity = int(capacity)
        self.max_slots = int(max_slots)
        self.min_match_len = int(min_match_len)
        self.max_match_len = int(max_match_len)
        self.base_confidence = float(base_confidence)
        self.confidence_decay = float(confidence_decay)
        self.device = torch.device(device)
        self.buffer = torch.full(
            (self.max_slots, self.capacity), -1, dtype=torch.int64, device=self.device
        )
        # Tokens ever written per slot; the ring holds the last ``capacity``.
        self.written = torch.zeros(
            self.max_slots, dtype=torch.int64, device=self.device
        )

    def reset(self) -> None:
        self.buffer.fill_(-1)
        self.written.zero_()

    def reset_slots(self, slots: list[int] | torch.Tensor) -> None:
        """Clear the rings of ``slots`` (a new request took them)."""
        idx = torch.as_tensor(slots, dtype=torch.int64, device=self.device).reshape(-1)
        if idx.numel() == 0:
            return
        self.buffer[idx] = -1
        self.written[idx] = 0

    def size(self, slot: int) -> int:
        return int(min(int(self.written[slot].item()), self.capacity))

    @torch.no_grad()
    def record_verified_tokens(
        self,
        slots: torch.Tensor,
        tokens: torch.Tensor,
        counts: torch.Tensor | None = None,
    ) -> None:
        """Append ``tokens[r, :counts[r]]`` to slot ``slots[r]``'s ring, for each
        row with ``slots[r] >= 0`` (no host synchronization)."""
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
        rows, width = tokens.shape
        if rows == 0 or width == 0:
            return
        slots = slots.reshape(-1)[:rows].to(device=self.device, dtype=torch.int64)
        tokens = tokens.to(device=self.device, dtype=torch.int64)
        if counts is None:
            counts = (tokens >= 0).sum(dim=1)
        counts = counts.reshape(-1)[:rows].to(device=self.device, dtype=torch.int64)
        cols = torch.arange(width, dtype=torch.int64, device=self.device)[None, :]
        live = (slots[:, None] >= 0) & (cols < counts[:, None]) & (tokens >= 0)
        safe_slots = slots.clamp(0, self.max_slots - 1)
        offsets = torch.cumsum(live.to(torch.int64), dim=1) - 1
        position = (self.written[safe_slots][:, None] + offsets) % self.capacity
        # Rows past the live ones write to a dump column that is never read:
        # write into a widened view and drop it.
        flat_index = torch.where(
            live, safe_slots[:, None] * (self.capacity + 1) + position,
            safe_slots[:, None] * (self.capacity + 1) + self.capacity,
        )
        dump = torch.full(
            (self.max_slots, 1), -1, dtype=torch.int64, device=self.device
        )
        widened = torch.cat([self.buffer, dump], dim=1)
        widened.view(-1).scatter_(0, flat_index.reshape(-1), tokens.reshape(-1))
        self.buffer.copy_(widened[:, : self.capacity])
        added = torch.zeros_like(self.written).index_add_(
            0, safe_slots, live.to(torch.int64).sum(dim=1) * (slots >= 0)
        )
        self.written.add_(added)

    def get_chronological_tokens(self, slot: int) -> torch.Tensor:
        """Slot ``slot``'s tokens, oldest first."""
        written = int(self.written[slot].item())
        if written <= self.capacity:
            return self.buffer[slot, :written]
        start = written % self.capacity
        return torch.cat([self.buffer[slot, start:], self.buffer[slot, :start]])

    @torch.no_grad()
    def propose_suffix_continuation(
        self,
        context_tokens: torch.Tensor,
        num_draft_steps: int,
        *,
        slots: torch.Tensor,
        seq_lens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Match each request's context suffix against its own ring.

        Returns ``(drafts [rows, W], match_lengths [rows], confidences [rows, W])``;
        ``match_lengths`` is 0 where no suffix of at least ``min_match_len``
        matched. Host-synchronizing; opt-in only.
        """
        if context_tokens.ndim == 1:
            context_tokens = context_tokens.unsqueeze(0)
        num_reqs, max_len = context_tokens.shape
        device = context_tokens.device
        shape = (num_reqs, num_draft_steps)
        out_tokens = torch.zeros(shape, dtype=torch.int64, device=device)
        match_lengths = torch.zeros((num_reqs,), dtype=torch.int32, device=device)
        confidences = torch.zeros(shape, dtype=torch.float32, device=device)
        if num_draft_steps <= 0:
            return out_tokens, match_lengths, confidences
        step_decay = self.confidence_decay ** torch.arange(
            num_draft_steps, dtype=torch.float32, device=device
        )
        slot_list = slots.reshape(-1)[:num_reqs].tolist()
        for r, slot in enumerate(slot_list):
            if slot < 0:
                continue
            mem = self.get_chronological_tokens(int(slot)).to(device=device)
            mem_len = int(mem.numel())
            if mem_len < self.min_match_len + num_draft_steps:
                continue
            searchable_end = mem_len - num_draft_steps
            cur_len = int(seq_lens[r].item()) if seq_lens is not None else max_len
            if cur_len < self.min_match_len:
                continue
            row = context_tokens[r, :cur_len]
            max_n = min(self.max_match_len, cur_len, searchable_end)
            best_len, best_pos = 0, -1
            for n in range(max_n, self.min_match_len - 1, -1):
                suffix = row[cur_len - n : cur_len]
                windows = mem[: searchable_end + n - 1].unfold(0, n, 1)
                matches = (windows == suffix.unsqueeze(0)).all(dim=-1)
                if bool(matches.any()):
                    best_pos = int(torch.nonzero(matches).view(-1)[-1].item())
                    best_len = n
                    break
            if best_len >= self.min_match_len and best_pos >= 0:
                start = best_pos + best_len
                out_tokens[r] = mem[start : start + num_draft_steps]
                match_lengths[r] = best_len
                len_bonus = min(1.0, best_len / float(self.max_match_len))
                base_c = min(0.98, self.base_confidence + 0.10 * len_bonus)
                confidences[r] = base_c * step_decay
        return out_tokens, match_lengths, confidences


@torch.no_grad()
def verify_greedy_lossless(
    draft_tokens: torch.Tensor,
    target_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference greedy speculative verification (tests only).

    ``target_logits`` are ``[batch, W + 1, vocab]`` on ``[anchor, d_0, ...]``;
    returns ``(emitted [batch, W + 1], num_accepted [batch], num_emitted [batch])``.
    """
    bsz, steps = draft_tokens.shape
    target_argmax = target_logits.argmax(dim=-1)
    matches = draft_tokens == target_argmax[:, :steps]
    num_accepted = torch.cumprod(matches.to(dtype=torch.int64), dim=-1).sum(dim=-1)
    emitted = torch.full(
        (bsz, steps + 1), -1, dtype=draft_tokens.dtype, device=draft_tokens.device
    )
    for b in range(bsz):
        acc = int(num_accepted[b].item())
        if acc > 0:
            emitted[b, :acc] = draft_tokens[b, :acc]
        emitted[b, acc] = target_argmax[b, acc]
    return emitted, num_accepted, num_accepted + 1
