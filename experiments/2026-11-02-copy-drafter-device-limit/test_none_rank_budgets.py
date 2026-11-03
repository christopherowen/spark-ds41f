# SPDX-License-Identifier: Apache-2.0
"""The none arms' claim, on candidate-37's own vLLM: without the host's limit read
(``slot_draft_limits=None`` at the manager, overlay/none), two tensor-parallel ranks
whose limit copies land at different moments budget the same step; with it (the
runner's wiring on candidate-37) they don't.

Two "ranks" are two independent sets of request state, copy drafter and adaptive
verification manager on one GPU, given the same histories, proposals and
confidences. When they budget the next step, rank 0's limit copy has landed and
rank 1's is still in flight (patch 0612 reads it without waiting).

usage (in spark-ds41f:candidate-37, nothing mounted but this file):
  python3 -m pytest -q /x/test_none_rank_budgets.py
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("tilelang")

from vllm.config.speculative import CopyDraftsConfig  # noqa: E402
from vllm.v1.worker.gpu.spec_decode import adaptive_verification as adaptive  # noqa: E402
from vllm.v1.worker.gpu.spec_decode.copy_drafts import CopyDrafter  # noqa: E402
from vllm.v1.worker.gpu.states import RequestState  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)

DEVICE = torch.device("cuda")
MAX_REQS = 16
LOOKAHEAD = 16
BLOCK = 5  # DSpark's drafts
INITIAL = 8
VOCAB = 129280


class _InFlight:
    """A limit copy that has not landed: the host keeps the previous limits."""

    def query(self) -> bool:
        return False

    def synchronize(self) -> None:
        raise AssertionError("the limit read must not block")


def _rank(wiring: str, landed: bool, histories: dict[str, list[int]]):
    """One rank after the step that proposed: a copy for "copy" (its history
    repeats), DSpark's block for "prose". Returns its next step's budget."""
    states = RequestState(
        max_num_reqs=MAX_REQS,
        max_model_len=8192,
        max_num_batched_tokens=4096,
        num_speculative_steps=LOOKAHEAD,
        vocab_size=VOCAB,
        device=DEVICE,
    )
    drafter = CopyDrafter(
        CopyDraftsConfig(min_match=16, max_drafts=LOOKAHEAD, initial_drafts=INITIAL),
        states,
        num_draft_steps=BLOCK,
        device=DEVICE,
    )
    manager = adaptive.AdaptiveVerificationManager(
        states,
        torch.zeros(MAX_REQS + 1, dtype=torch.int32, device=DEVICE),
        num_bonus_tokens=1,
        max_total_logits=4096,
        row_cost=0.035,
        planned_confidences=True,
        num_draft_steps=BLOCK,
        # candidate-37's runner hands the budget the drafter's host limits; the
        # none arms hand it none.
        slot_draft_limits=drafter.sync_slot_draft_limits if wiring == "c37" else None,
    )
    manager.dead_rows_tau = 0.0
    slots = []
    for req_id, tokens in histories.items():
        states.add_request(req_id, len(tokens), tokens, 0, 1 << 20)
        slot = states.req_id_to_index[req_id]
        states.num_computed_tokens_np[slot] = len(tokens)
        drafter.add_request(slot)
        manager.add_request(slot)
        slots.append(slot)
    states.apply_staged_writes()
    drafter.apply_staged_writes()
    manager.apply_staged_writes()
    idx = torch.tensor(slots, dtype=torch.int64, device=DEVICE)
    n = len(slots)
    one = torch.ones(n, dtype=torch.int32, device=DEVICE)
    zero = torch.zeros(n, dtype=torch.int32, device=DEVICE)
    dspark_drafts = torch.zeros((n, BLOCK), dtype=torch.int64, device=DEVICE)
    dspark_conf = torch.full((n, BLOCK), 0.5, device=DEVICE)
    drafter.prepare_alloc(idx, one, zero, None)
    _, confidences, planned = drafter.propose(
        idx, dspark_drafts, dspark_conf, one, zero, None, matched=True
    )
    batch = SimpleNamespace(num_reqs=n, idx_mapping=idx)
    for _ in range(2):  # the budget reads the snapshot of the record before last
        manager.record_confidences(confidences, batch, planned=planned)
    if landed:
        drafter.slot_draft_limits.np[:] = drafter.slot_draft_limits.gpu.cpu().numpy()
    else:
        drafter._limits_event = _InFlight()  # type: ignore[assignment]
    # The scheduler laid this step out from an earlier allocation: 16 rows for the
    # copy, the block for the other.
    num_tokens = manager.get_num_tokens(
        {"copy": 1 + LOOKAHEAD, "prose": 1 + BLOCK},
        {"copy": [-1] * LOOKAHEAD, "prose": [-1] * BLOCK},
    )
    eligible = dict(manager._batch_budget[0])
    cu_num_logits, query_start_loc, budget, _, _ = manager.reallocate_drafts(
        list(histories), idx
    )
    torch.accelerator.synchronize()
    return SimpleNamespace(
        limits=drafter.slot_draft_limits.gpu[idx].tolist(),
        num_tokens=num_tokens,
        eligible=eligible,
        budget=budget,
        query_start_loc=query_start_loc[: n + 1].tolist(),
        cu_num_logits=cu_num_logits.tolist(),
    )


def _histories():
    rng = np.random.default_rng(3)
    return {
        "copy": np.tile(rng.integers(1, VOCAB, 32), 6).tolist()[:96],
        "prose": rng.integers(1, VOCAB, 96).tolist(),
    }


def test_the_host_limit_read_splits_candidate_37s_ranks(monkeypatch):
    """candidate-37's wiring: the rank whose copy landed caps the copy at its 8
    drafts, the other at the stale 16, so the two lay out different steps."""
    monkeypatch.setattr(adaptive, "get_tp_group", lambda: SimpleNamespace(world_size=1))
    landed = _rank("c37", True, _histories())
    in_flight = _rank("c37", False, _histories())
    assert landed.limits == in_flight.limits == [INITIAL, BLOCK]
    assert landed.eligible == {"copy": INITIAL, "prose": BLOCK}
    assert in_flight.eligible == {"copy": LOOKAHEAD, "prose": BLOCK}
    assert landed.num_tokens != in_flight.num_tokens
    assert landed.query_start_loc != in_flight.query_start_loc


def test_without_the_host_limit_read_ranks_budget_alike(monkeypatch):
    """The none arms' wiring: both ranks budget from the scheduler's drafts and the
    confidence snapshot, whatever their limit copies: the same token count, the
    same per-request drafts and the same device layout."""
    monkeypatch.setattr(adaptive, "get_tp_group", lambda: SimpleNamespace(world_size=1))
    landed = _rank("none", True, _histories())
    in_flight = _rank("none", False, _histories())
    assert landed.eligible == in_flight.eligible == {"copy": LOOKAHEAD, "prose": BLOCK}
    for field in ("num_tokens", "budget", "query_start_loc", "cu_num_logits"):
        assert getattr(landed, field) == getattr(in_flight, field), field
