"""Concurrency-bounded sequential Markov drafting depth policy for DSpark."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def parse_max_steps_by_batch(
    spec: str | Mapping[int, int] | Sequence[tuple[int, int]] | None,
    num_speculative_steps: int,
) -> tuple[tuple[int, int], ...]:
    """Parse concurrency-bounded speculative step schedule ``(min_reqs, max_steps)``.

    Entries are sorted by ``min_reqs`` ascending, clamped to
    ``[1, num_speculative_steps]``, and made monotonically non-increasing in
    ``max_steps`` as ``min_reqs`` increases.
    """
    if not spec or num_speculative_steps <= 0:
        return ()
    raw_pairs: list[tuple[int, int]] = []
    if isinstance(spec, str):
        for token in spec.split(","):
            token = token.strip()
            if not token:
                continue
            if ":" not in token:
                raise ValueError(
                    f"Invalid max_steps_by_batch entry {token!r}; expected 'batch:steps'"
                )
            batch_s, steps_s = token.split(":", 1)
            raw_pairs.append((int(batch_s.strip()), int(steps_s.strip())))
    elif isinstance(spec, Mapping):
        for batch_k, steps_v in spec.items():
            raw_pairs.append((int(batch_k), int(steps_v)))
    else:
        for batch_k, steps_v in spec:
            raw_pairs.append((int(batch_k), int(steps_v)))

    by_batch: dict[int, int] = {}
    for min_reqs, max_steps in sorted(raw_pairs, key=lambda item: item[0]):
        if min_reqs < 1:
            raise ValueError(f"min_reqs must be >= 1, got {min_reqs}")
        clamped = max(1, min(int(max_steps), int(num_speculative_steps)))
        by_batch[min_reqs] = min(by_batch.get(min_reqs, clamped), clamped)

    normalized: list[tuple[int, int]] = []
    running_cap = int(num_speculative_steps)
    for min_reqs in sorted(by_batch):
        running_cap = min(running_cap, by_batch[min_reqs])
        if running_cap < num_speculative_steps and (
            not normalized or normalized[-1][1] != running_cap
        ):
            normalized.append((min_reqs, running_cap))
    return tuple(normalized)


def active_speculative_steps(
    num_reqs: int,
    num_speculative_steps: int,
    schedule: Sequence[tuple[int, int]] = (),
) -> int:
    """Return active sequential Markov draft steps for batch size ``num_reqs``."""
    if not schedule or num_reqs <= 0:
        return num_speculative_steps
    steps = num_speculative_steps
    for min_reqs, max_steps in schedule:
        if num_reqs >= min_reqs:
            steps = min(steps, max_steps)
        else:
            break
    return max(1, min(steps, num_speculative_steps))


def estimate_draft_work_savings(
    num_reqs: int,
    num_speculative_steps: int,
    schedule: Sequence[tuple[int, int]],
) -> dict[str, int | float]:
    """Quantify saved LM-head rows, confidence-head rows, and Markov collectives."""
    active_steps = active_speculative_steps(num_reqs, num_speculative_steps, schedule)
    full_rows = num_reqs * num_speculative_steps
    active_rows = num_reqs * active_steps
    saved_rows = full_rows - active_rows
    saved_markov_steps = num_speculative_steps - active_steps
    return {
        "num_reqs": num_reqs,
        "num_speculative_steps": num_speculative_steps,
        "active_steps": active_steps,
        "full_rows": full_rows,
        "active_rows": active_rows,
        "saved_rows": saved_rows,
        "saved_markov_steps": saved_markov_steps,
        "row_reduction_fraction": saved_rows / full_rows if full_rows > 0 else 0.0,
    }
