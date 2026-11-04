# SPDX-License-Identifier: Apache-2.0
"""Apply the recorded local phase patch to a disposable image-source copy."""

import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys


def load(out):
    from vllm.models.deepseek_v4_1.tilelang import iket

    here = Path(__file__).resolve().parents[1] / "vllm"
    identity = json.loads((here / "identity.json").read_text())
    original = Path(iket.__file__).read_bytes()
    if hashlib.sha256(original).hexdigest() != identity["base_sha256"]:
        raise RuntimeError("phase patch requires the frozen candidate-38 source")
    root = Path(out) / "phase-overlay"
    target = root / identity["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(original)
    source = (here / "overlay/iket.py").read_bytes()
    if hashlib.sha256(source).hexdigest() != identity["patched_sha256"]:
        raise RuntimeError(
            "phase overlay does not match its recorded patch-series identity"
        )
    target.write_bytes(source)
    name = iket.__name__
    spec = importlib.util.spec_from_file_location(name, target)
    patched = importlib.util.module_from_spec(spec)
    sys.modules[name] = patched
    spec.loader.exec_module(patched)
    parent = importlib.import_module(name.rsplit(".", 1)[0])
    parent.iket = patched
    return patched
