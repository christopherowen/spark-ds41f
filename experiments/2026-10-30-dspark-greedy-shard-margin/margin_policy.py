#!/usr/bin/env python3
"""Zero-extra-byte greedy shard runner-up margin and exposure-bias confidence calibration."""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
GREEDY_BLOCK = 2048


def _bf16_round(value: float) -> float:
    """Round FP32 value to BF16 and back to FP32, matching ``(a + b).to(bf16).to(fp32)``."""
    if math.isnan(value) or math.isinf(value):
        return value
    bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    rounding_bias = 0x7FFF + ((bits >> 16) & 1)
    bf16_bits = (bits + rounding_bias) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", bf16_bits))[0]


def _first_keep_or_take(
    best_val: float, best_idx: int, cand_val: float, cand_idx: int
) -> tuple[float, int]:
    """Total order used by TileLang greedy reductions: NaN first, then larger
    value, then smaller global index."""
    best_nan = math.isnan(best_val)
    cand_nan = math.isnan(cand_val)
    if best_nan and cand_nan:
        return (best_val, best_idx) if best_idx <= cand_idx else (cand_val, cand_idx)
    if cand_nan:
        return cand_val, cand_idx
    if best_nan:
        return best_val, best_idx
    if cand_val > best_val or (cand_val == best_val and cand_idx < best_idx):
        return cand_val, cand_idx
    return best_val, best_idx


def reduce_shard_pair_with_runner_up(
    base_logits: Sequence[Sequence[float]],
    markov_bias: Sequence[Sequence[float]],
    vocab_start: int,
    *,
    block_size: int = GREEDY_BLOCK,
    track_runner_up: bool = True,
    bf16_sum: bool = True,
) -> list[list[float]]:
    """Reference for ``greedy_markov_shard_pair`` writing into 16-byte ``[rows, 4]``
    shard rows:
      - col 0: shard maximum logit
      - col 1: global token index of shard maximum
      - col 2: best non-winning block maximum on this shard (when ``track_runner_up``)
      - col 3: untouched sentinel (-3.0)
    """
    if len(base_logits) != len(markov_bias):
        raise ValueError("base_logits and markov_bias must have identical shapes")
    rows = len(base_logits)
    vocab = len(base_logits[0]) if rows else 0
    parts = (vocab + block_size - 1) // block_size
    out: list[list[float]] = []

    for r in range(rows):
        if len(base_logits[r]) != vocab or len(markov_bias[r]) != vocab:
            raise ValueError("base_logits and markov_bias row widths must match")
        summed = [
            _bf16_round(float(a) + float(b)) if bf16_sum else (float(a) + float(b))
            for a, b in zip(base_logits[r], markov_bias[r])
        ]
        part_vals: list[float] = []
        part_idxs: list[int] = []
        for p in range(parts):
            start = p * block_size
            stop = min(vocab, start + block_size)
            bv = -float("inf")
            bi = (1 << 31) - 1
            for offset, val in enumerate(summed[start:stop]):
                bv, bi = _first_keep_or_take(bv, bi, float(val), start + offset)
            part_vals.append(bv)
            part_idxs.append(bi)

        win_val = -float("inf")
        win_idx = (1 << 31) - 1
        for bv, bi in zip(part_vals, part_idxs):
            win_val, win_idx = _first_keep_or_take(win_val, win_idx, bv, bi)

        runner = -3.0
        if track_runner_up:
            runner = -float("inf")
            for bv, bi in zip(part_vals, part_idxs):
                cand = -float("inf") if bi == win_idx else bv
                if not math.isnan(cand) and cand > runner:
                    runner = cand
        out.append([win_val, float(win_idx + vocab_start), runner, -3.0])
    return out


def select_first_argmax_with_margin(
    gathered_pairs: Sequence[Sequence[float]], pair_width: int = 4
) -> tuple[list[int], list[float]]:
    """Reference for ``select_first_argmax(..., pair_width=4, margin_out=...)``.

    Returns ``(selected_indices, logit_margins)`` across all TP ranks' 16-byte
    rows without changing the communication volume (16 bytes/rank/request).
    """
    rows = len(gathered_pairs)
    cols = len(gathered_pairs[0]) if rows else 0
    if cols % pair_width != 0 or pair_width < 4:
        raise ValueError("gathered_pairs must be a multiple of pair_width >= 4")
    ranks = cols // pair_width
    selected: list[int] = []
    margins: list[float] = []

    for r in range(rows):
        row = gathered_pairs[r]
        win_val = -float("inf")
        win_idx = 1 << 30
        for k in range(ranks):
            v = float(row[k * pair_width])
            idx = int(row[k * pair_width + 1])
            win_val, win_idx = _first_keep_or_take(win_val, win_idx, v, idx)

        runner = -float("inf")
        for k in range(ranks):
            v = float(row[k * pair_width])
            idx = int(row[k * pair_width + 1])
            r2 = float(row[k * pair_width + 2])
            cand = r2 if idx == win_idx else v
            if not math.isnan(cand) and cand > runner:
                runner = cand

        selected.append(win_idx)
        if (
            not math.isnan(win_val)
            and win_val > -float("inf")
            and runner > -float("inf")
        ):
            margins.append(max(win_val - runner, 0.0))
        else:
            margins.append(0.0)
    return selected, margins


def calibrate_confidence_probabilities(
    base_logits: Sequence[float],
    margins: Sequence[float] | None = None,
    *,
    margin_weight: float = 0.0,
    step_biases: Sequence[float] | None = None,
) -> list[float]:
    """Compute ``sigmoid(base_logits + margin_weight * margins + step_biases)``."""
    rows = len(base_logits)
    if margins is not None and len(margins) != rows:
        raise ValueError("margins length must match base_logits length")
    n_spec = len(step_biases) if step_biases is not None else 1
    if step_biases is not None and (n_spec <= 0 or rows % n_spec != 0):
        raise ValueError(f"rows {rows} not divisible by n_spec {n_spec}")

    probs: list[float] = []
    for m, z in enumerate(base_logits):
        val = float(z)
        if margins is not None and margin_weight != 0.0:
            val += float(margin_weight) * float(margins[m])
        if step_biases is not None:
            val += float(step_biases[m % n_spec])
        probs.append(1.0 / (1.0 + math.exp(-val)))
    return probs


def empirical_position_profile(manifest_path: Path, n_spec: int = 5) -> dict:
    """Extract per-position coverage, survival, conditional acceptance, verified-row
    yield, and step-level exposure-bias logit offsets from a servebench JSON."""
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    srv = data["summary"]["server"]
    cov = [float(srv[f"coverage_at_{k}"]["mean"]) for k in range(n_spec)]
    surv = [float(srv[f"survival_at_{k}"]["mean"]) for k in range(n_spec)]

    cond_accept: list[float] = []
    verified_yield: list[float] = []
    for k in range(n_spec):
        prev_surv = 1.0 if k == 0 else surv[k - 1]
        cond_accept.append(surv[k] / prev_surv if prev_surv > 0 else 0.0)
        verified_yield.append(surv[k] / cov[k] if cov[k] > 0 else 0.0)

    def _logit(p: float) -> float:
        p_clamped = min(max(p, 1e-4), 1.0 - 1e-4)
        return math.log(p_clamped / (1.0 - p_clamped))

    base_logit = _logit(cond_accept[0])
    step_biases = [round(_logit(p) - base_logit, 4) for p in cond_accept]

    return {
        "manifest": str(manifest_path.relative_to(ROOT)),
        "output_tps": float(data["summary"]["output_tps"]["mean"]),
        "accepted_per_draft": float(srv["accepted_per_draft"]["mean"]),
        "draft_tokens_per_draft": float(srv["draft_tokens_per_draft"]["mean"]),
        "coverage": [round(x, 4) for x in cov],
        "survival": [round(x, 4) for x in surv],
        "conditional_acceptance": [round(x, 4) for x in cond_accept],
        "verified_row_yield": [round(x, 4) for x in verified_yield],
        "recommended_step_biases": step_biases,
        "tp4_allgather_bytes_per_req_per_step": 16 * 4,
        "extra_collective_bytes": 0,
    }


def evaluate_suite_profiles() -> dict:
    suites = {
        "tp4_c16": ROOT / "manifests/suites/2026-10-30-spark-ds41f-1030-tp4/c16.json",
        "tp4_c8": ROOT / "manifests/suites/2026-10-30-spark-ds41f-1030-tp4/c8.json",
        "tp4_sampled_c16": ROOT
        / "manifests/suites/2026-10-30-spark-ds41f-1030-tp4-sampled/c16.json",
        "tp4_sampled_c8": ROOT
        / "manifests/suites/2026-10-30-spark-ds41f-1030-tp4-sampled/c8.json",
        "tp4_sampled_c1": ROOT
        / "manifests/suites/2026-10-30-spark-ds41f-1030-tp4-sampled/c1.json",
    }
    profiles = {name: empirical_position_profile(path) for name, path in suites.items()}
    return {
        "experiment": "2026-10-30-dspark-greedy-shard-margin",
        "profiles": profiles,
    }


if __name__ == "__main__":
    summary = evaluate_suite_profiles()
    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
