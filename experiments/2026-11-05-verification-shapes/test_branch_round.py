"""CPU checks of the experiment controller's ownership and restoration gates."""

import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

spec = importlib.util.spec_from_file_location(
    "branch_round", Path(__file__).with_name("branch_round.py")
)
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


@pytest.mark.parametrize(
    "hold",
    [
        None,
        {},
        {"holder": "spark-lab: another job", "window": "ours"},
        {"holder": r.HOLDER, "window": "old"},
    ],
)
def test_foreign_or_stale_hold_refuses_mutation(hold):
    with pytest.raises(RuntimeError, match="ownership lost"):
        r.require_owner(hold, {"window": "ours"})


def test_exact_owner_accepted():
    r.require_owner({"holder": r.HOLDER, "window": "ours"}, {"window": "ours"})


@pytest.mark.parametrize(
    "action,output,status",
    [
        ("start", "live: running\nshape:   CHANGES max_model_len\n", 0),
        ("start", "live: running\nshape: unchanged\n--apply would refuse: dirty\n", 0),
        ("start", "live: running\n", 0),
        ("start", "shape: unchanged\n", 0),
        ("stop", "plan only\n", 0),
        ("sync", "plan only\n", 1),
    ],
)
def test_refusal_and_incomplete_plans_stop(action, output, status):
    with pytest.raises(RuntimeError):
        r.check_plan(action, output, status)


@pytest.mark.parametrize(
    "action,output",
    [
        ("start", "live: running\nshape:   unchanged\n"),
        ("start", "live: none\nshape:   new engine\n"),
        ("stop", "live: running\n"),
        ("sync", "three nodes at the published commit\n"),
    ],
)
def test_complete_plans_accepted(action, output):
    r.check_plan(action, output, 0)


def controller(tmp_path):
    state = {
        "window": "ours",
        "restore_checkout": str(tmp_path / "original"),
        "restore_revision": "a" * 40,
        "restore_config": "original.json",
        "candidate_config": "candidate.json",
        "commands": [],
        "sync_started": True,
    }
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(state))
    return r.Round(path)


def test_restore_orders_coordinated_stop_sync_start_then_live_check(
    tmp_path, monkeypatch
):
    round = controller(tmp_path)
    monkeypatch.setattr(round, "own", Mock())
    monkeypatch.setattr(r, "clean_revision", lambda _: "a" * 40)
    calls = []
    monkeypatch.setattr(round, "cli", lambda *a, **kw: calls.append((a, kw)))
    monkeypatch.setattr(round, "run", lambda *a, **kw: calls.append((a, kw)))
    monkeypatch.setattr(
        r.lab.spark,
        "configuration",
        lambda _: (
            {},
            {"nodes": [{"name": "dgx1"}, {"name": "dgx2"}, {"name": "dgx3"}]},
            {},
        ),
    )
    monkeypatch.setattr(
        r.lab.spark, "remote_repository_state", lambda *_: ("a" * 40, "origin", False)
    )
    monkeypatch.setattr(r.lab, "read_hold", lambda: None)
    assert round.restore()
    assert [x[0][2:4] for x in calls[:4]] == [
        ("cluster", "stop"),
        ("cluster", "sync"),
        ("cluster", "start"),
        ("doctor", "--live"),
    ]
    assert calls[1][0][4:] == ("--revision", "a" * 40)
    assert all(x[1]["closing"] for x in calls)
    assert round.state["restored"] is True


def test_restore_never_closes_hold_after_a_failed_sync(tmp_path, monkeypatch):
    round = controller(tmp_path)
    monkeypatch.setattr(round, "own", Mock())
    monkeypatch.setattr(r, "clean_revision", lambda _: "a" * 40)
    calls = []

    def cli(*args, **kwargs):
        calls.append(args[3])
        if args[3] == "sync":
            raise RuntimeError("sync failed")

    monkeypatch.setattr(round, "cli", cli)
    closed = Mock()
    monkeypatch.setattr(round, "run", closed)
    with pytest.raises(RuntimeError, match="sync failed"):
        round.restore()
    assert calls == ["stop", "sync"]
    closed.assert_not_called()
    assert not round.state.get("restored")


def test_lost_window_prevents_restoration_actions(tmp_path, monkeypatch):
    round = controller(tmp_path)
    monkeypatch.setattr(
        r.lab,
        "read_hold",
        lambda: {"holder": "spark-lab: somebody else", "window": "ours"},
    )
    action = Mock()
    monkeypatch.setattr(round, "cli", action)
    with pytest.raises(RuntimeError, match="ownership lost"):
        round.restore()
    action.assert_not_called()


def test_cli_checks_owner_again_between_plan_and_apply(tmp_path, monkeypatch):
    round = controller(tmp_path)
    holds = iter(
        [
            {"holder": r.HOLDER, "window": "ours"},
            {"holder": r.HOLDER, "window": "ours"},
            {"holder": r.HOLDER, "window": "ours"},
            {"holder": r.HOLDER, "window": "ours"},
            {"holder": "spark-lab: different", "window": "new"},
        ]
    )
    monkeypatch.setattr(r.lab, "read_hold", lambda: next(holds))
    monkeypatch.setattr(r.lab, "window_should_close", lambda _: None)
    command = Mock(
        return_value=r.subprocess.CompletedProcess(
            [], 0, "live: running\nshape: unchanged\n"
        )
    )
    monkeypatch.setattr(r.subprocess, "run", command)
    with pytest.raises(RuntimeError, match="ownership lost"):
        round.cli(r.ROOT, "candidate.json", "cluster", "start", apply=True)
    assert command.call_count == 1
    assert "--apply" not in command.call_args.args[0]


def test_incomplete_gate_never_inspects_or_launches_image(monkeypatch):
    action = Mock()
    monkeypatch.setattr(r.subprocess, "check_output", action)
    with pytest.raises(RuntimeError, match="incomplete"):
        r.validate_gate(
            {"passed": True}, {"container": {"expected_labels": {}}}, "a" * 40
        )
    action.assert_not_called()
