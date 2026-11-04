import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pytest

PATH = Path(__file__).parents[1] / "2026-10-31-iket-full-43l-and-runtime-opt/harness/iket_profile.py"
SPEC = importlib.util.spec_from_file_location("profile_under_test", PATH)
profile = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(profile)


def manifest(root, kernel="mhc_project_finalize_kernel", bound=60):
    root.mkdir(exist_ok=True)
    (root / (kernel + ".json")).write_text(json.dumps({
        "kernel": kernel, "sites": [{"range": "compute", "events_per_warp_bound": bound}]}))
    return root


def test_capacity_includes_margin_and_fingerprints(tmp_path):
    source = manifest(tmp_path / "sources")
    result = profile.event_capacity_from_sources([source])
    assert result["maximum"] == 60
    assert result["capacity"] == 64
    assert len(next(iter(result["manifests_sha256"].values()))) == 64


@pytest.mark.parametrize("bound", [None, -1, "60", True])
def test_unknown_or_malformed_bounds_are_not_ignored(tmp_path, bound):
    source = manifest(tmp_path / "sources", bound=bound)
    with pytest.raises(ValueError, match="unknown event bound"):
        profile.event_capacity_from_sources([source])


def test_exact_replay_selection_excludes_other_kernels_but_requires_presence(tmp_path):
    source = manifest(tmp_path / "sources")
    manifest(source, "unbounded", None)
    with pytest.raises(ValueError, match="unknown"):
        profile.event_capacity_from_sources([source])
    assert profile.event_capacity_from_sources([source], ["mhc_project_finalize_kernel"])["capacity"] == 64
    with pytest.raises(ValueError, match="missing event manifests"):
        profile.event_capacity_from_sources([source], ["missing"])


@pytest.mark.parametrize("capacity", ["42", "auto"])
def test_invalid_capacity_never_launches_workload(tmp_path, capacity):
    source = manifest(tmp_path / "sources", bound=60 if capacity == "42" else None)
    with patch.object(profile, "run_iket") as run:
        with pytest.raises(SystemExit) as error:
            profile.main([str(tmp_path / "out"), "--single-pass-max-ts", capacity,
                          "--event-sources", str(source), "--", "workload"])
        assert error.value.code == 2
        run.assert_not_called()


def test_automatic_single_pass_writes_certified_capacity(tmp_path):
    source = manifest(tmp_path / "sources")
    out = tmp_path / "out"
    with patch.object(profile, "run_iket", return_value=(0, "")) as run:
        with pytest.raises(SystemExit) as error:
            profile.main([str(out), "--single-pass-max-ts", "auto", "--event-sources", str(source),
                          "--pool-gb", "0.25", "--", "workload"])
        assert error.value.code == 0
        assert run.call_count == 1
    config = json.loads((out / "stub.config.json").read_text())
    assert config["configs"][0]["maxTsCntPerWarp"] == 64
    assert json.loads((out / "event-capacity.json").read_text())["maximum"] == 60


def test_pool_limit_applies_to_single_pass_before_launch(tmp_path):
    with patch.object(profile, "run_iket") as run:
        with pytest.raises(SystemExit) as error:
            profile.main([str(tmp_path / "out"), "--single-pass-max-ts", "64", "--pool-gb", "2",
                          "--max-pool-gb", "1", "--", "workload"])
        assert error.value.code == 2
        run.assert_not_called()
