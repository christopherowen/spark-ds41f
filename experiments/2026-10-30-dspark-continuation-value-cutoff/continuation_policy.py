"""Continuation-value cutoff and slot-ranking policy for DSpark adaptive verification."""

from __future__ import annotations

import math
from typing import Sequence


def prefix_survival(confidences: Sequence[float], valid_count: int | None = None) -> list[float]:
    """Compute prefix survival probabilities S_j = prod_{m=0}^j p_m for valid slots."""
    limit = len(confidences) if valid_count is None else min(len(confidences), valid_count)
    out: list[float] = []
    running = 1.0
    for idx, prob in enumerate(confidences):
        if idx >= limit:
            out.append(0.0)
        else:
            running *= float(prob)
            out.append(running)
    return out


def effective_survival_scores(
    confidences: Sequence[float],
    valid_count: int | None = None,
    continuation_alpha: float = 0.0,
) -> list[float]:
    """Score valid draft slots by prefix survival plus weighted suffix yield.

    Each valid draft slot j < valid_count receives:
        S_tilde_j = S_j + alpha * sum_{m=j+1}^{valid_count-1} S_m
                  = (1 - alpha) * S_j + alpha * sum_{m=j}^{valid_count-1} S_m

    Because S_j >= 0 is non-increasing in j, the suffix sum is also non-increasing
    in j, guaranteeing S_tilde_0 >= S_tilde_1 >= ... >= S_tilde_{valid_count-1}.
    """
    limit = len(confidences) if valid_count is None else min(len(confidences), valid_count)
    surv = prefix_survival(confidences, limit)
    if continuation_alpha <= 0.0:
        return surv
    alpha = float(continuation_alpha)
    scores = list(surv)
    suffix = 0.0
    for j in range(len(confidences) - 1, -1, -1):
        if j < limit:
            scores[j] = surv[j] + alpha * suffix
            suffix += surv[j]
        else:
            scores[j] = 0.0
    return scores


def dead_row_cut(
    confidences: Sequence[float],
    valid_count: int,
    tau: float,
    continuation_alpha: float = 0.0,
) -> int:
    """Count live verification draft rows whose effective survival meets tau."""
    if tau <= 0.0:
        return valid_count
    log_tau = math.log(tau)
    scores = effective_survival_scores(confidences, valid_count, continuation_alpha)
    cut = 0
    for j in range(min(len(confidences), valid_count)):
        score = scores[j]
        if score > 0.0 and math.log(score) >= log_tau:
            cut += 1
    return cut


def select_batch_draft_budget(
    batch_confidences: Sequence[Sequence[float]],
    scheduled_drafts: Sequence[int],
    num_non_draft_tokens: Sequence[int],
    row_cost: float,
    continuation_alpha: float = 0.0,
    max_total_logits: int = 1 << 30,
    num_bonus_tokens: int = 1,
) -> tuple[int, list[int]]:
    """Select global draft budget and per-request admitted draft counts."""
    num_reqs = len(batch_confidences)
    flat_Candidates: list[tuple[float, int, int]] = []
    for req_idx, (confs, valid_count) in enumerate(
        zip(batch_confidences, scheduled_drafts, strict=True)
    ):
        scores = effective_survival_scores(confs, valid_count, continuation_alpha)
        for step_idx in range(min(len(confs), valid_count)):
            flat_Candidates.append((scores[step_idx], req_idx, step_idx))

    # Sort descending by score; tie-break by lower (req_idx, step_idx)
    flat_Candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    max_draft_budget = min(
        sum(scheduled_drafts),
        max(0, max_total_logits - num_reqs * num_bonus_tokens),
    )
    ranked = flat_Candidates[:max_draft_budget]

    base_rows = sum(num_non_draft_tokens)
    best_budget = 0
    best_utility = num_reqs / (1.0 + row_cost * base_rows)
    running_score = float(num_reqs)
    for k, (score, _, _) in enumerate(ranked, start=1):
        running_score += score
        utility = running_score / (1.0 + row_cost * (base_rows + k))
        if utility > best_utility:
            best_utility = utility
            best_budget = k

    admitted = [0] * num_reqs
    for _, req_idx, _ in ranked[:best_budget]:
        admitted[req_idx] += 1
    return best_budget, admitted
