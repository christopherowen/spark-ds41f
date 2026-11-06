"""CPU validation of immutable-image and post-build identity checks."""

import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "gpu_gate", Path(__file__).with_name("gpu_gate.py")
)
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)


def sample():
    record = {
        "cpu_build_completed": True,
        "ended": "now",
        "builder_stop_exit": 0,
        "image": "sha256:" + "1" * 64,
    }
    image = {
        "Id": record["image"],
        "RootFS": {"Layers": ["sha256:" + "2" * 64]},
        "Config": {"Labels": {"source": "exact"}, "Env": ["A=1"]},
    }
    return record, image


def test_faithful_copy_accepts_different_docker_store_id():
    record, image = sample()
    copied = copy.deepcopy(image)
    copied["Id"] = "sha256:" + "3" * 64
    g.build_matches(record, image, copied, {"source": "exact"}, [])


@pytest.mark.parametrize("field", ["layer", "config", "label"])
def test_copied_content_drift_rejected(field):
    record, image = sample()
    copied = copy.deepcopy(image)
    if field == "layer":
        copied["RootFS"]["Layers"][0] = "sha256:" + "4" * 64
    elif field == "config":
        copied["Config"]["Env"] = ["A=2"]
    else:
        copied["Config"]["Labels"]["source"] = "wrong"
    with pytest.raises(RuntimeError):
        g.build_matches(record, image, copied, {"source": "exact"}, [])


def test_runtime_input_changes_after_build_are_rejected():
    record, image = sample()
    with pytest.raises(RuntimeError, match="inputs changed"):
        g.build_matches(
            record, image, image, {"source": "exact"}, ["docker/Dockerfile"]
        )


def test_incomplete_build_is_rejected():
    record, image = sample()
    record["cpu_build_completed"] = False
    with pytest.raises(RuntimeError, match="incomplete"):
        g.build_matches(record, image, image, {"source": "exact"}, [])


def test_qualification_only_changes_do_not_force_another_build():
    record, image = sample()
    path = str(g.EXP.relative_to(g.ROOT) / "test_graph_replay.py")
    g.build_matches(record, image, image, {"source": "exact"}, [path])


@pytest.mark.parametrize("damage", ["none", "log", "commit", "missing-check"])
def test_screen_resume_requires_same_identity_and_untouched_logs(tmp_path, damage):
    import hashlib
    import json

    _, image = sample()
    checks = ("gpu-smoke", "graph-replay", "memcheck", "initcheck")
    record = {
        "screen_passed": True,
        "ended": "now",
        "commit": "a" * 40,
        "image_content": g.lab.spark.image_content_identity(image),
        "checks": {name: {"passed": True} for name in checks},
        "steps": [],
    }
    for name in checks:
        data = (name + " passed").encode()
        (tmp_path / (name + ".log")).write_bytes(data)
        record["steps"].append(
            {
                "log": name + ".log",
                "exit": 0,
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    if damage == "log":
        (tmp_path / "memcheck.log").write_text("changed")
    elif damage == "commit":
        record["commit"] = "b" * 40
    elif damage == "missing-check":
        record["checks"].pop("graph-replay")
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(record))
    if damage == "none":
        assert len(g.validated_screen(path, "a" * 40, image)[1]) == 4
    else:
        with pytest.raises(RuntimeError):
            g.validated_screen(path, "a" * 40, image)


def test_boundary_grid_rejects_missing_full_model_path():
    import copy
    import importlib.util

    path = Path(__file__).with_name("boundary_grid.py")
    spec = importlib.util.spec_from_file_location("boundary_grid", path)
    boundary = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(boundary)
    shapes = [{"kind": "decode", "num_reqs": 3, "num_tokens": 18, "context_len": 4096}]
    raw = {
        "workers": 3,
        "replays": 2,
        "shapes": shapes,
        "grid": [
            {
                "shape": {k: v for k, v in shapes[0].items() if k != "kind"},
                "step_ms": [2.0, 2.1],
                "forward_ms": [1.0, 1.1],
                "drafter_ms": [0.5, 0.5],
                "modes": [[True, 18, 3]] * 2,
            }
        ],
    }
    boundary.validate_grid(raw, shapes, 2, 3)
    for modes in ([], [[False, 18, 3]] * 2, [[True, 20, 3]] * 2):
        wrong = copy.deepcopy(raw)
        wrong["grid"][0]["modes"] = modes
        with pytest.raises(RuntimeError, match="full-graph"):
            boundary.validate_grid(wrong, shapes, 2, 3)
    wrong = copy.deepcopy(raw)
    wrong["grid"][0]["drafter_ms"] = []
    with pytest.raises(RuntimeError, match="samples"):
        boundary.validate_grid(wrong, shapes, 2, 3)
