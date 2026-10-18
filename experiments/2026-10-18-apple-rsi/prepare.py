#!/usr/bin/env python3
"""Build the attached copy of Apple's RSI-Inference DS4.1 verifier on this node.

usage: prepare.py SOURCE_TASK_DIR SNAPSHOT DEST

SOURCE_TASK_DIR is tasks/support_deepseek_v41_flash_4xb200_vllm of a local checkout of
apple-aiml-research/ml-rsi-inference-bench (Apple Sample Code License; it stays outside
this repository). SNAPSHOT is the original checkpoint's snapshot (dba1be0a), whose
tokenizer and config files the clients read. DEST (for example ~/spark-lab/apple-rsi)
receives:

- tests/: the task's verifier files, with two changes for a running server:
  - config.json's model id becomes this deployment's served name, and its dataset
    root a directory the runner fills with the evaluator's own GSM8K download;
  - infb_verify/conftest.py's session server is the running server at
    INFB_BASE_URL instead of a fresh launch at the task's B200 recipe;
- tests/hidden-as-examples.json: the held-out fidelity sequences in the public
  checker's examples format;
- model/: the snapshot's tokenizer and config files, resolved (no symlinks).

Nothing here changes how a check reads or judges a reply.
"""
import json
import shutil
import sys
from pathlib import Path

SERVED = "deepseek-v4.1-flash"
CONFTEST = '''"""Attached session server for spark-ds41f: the deployment already serving at
INFB_BASE_URL (written by spark-ds41f experiments/2026-10-18-apple-rsi/prepare.py)."""

from __future__ import annotations

import os

import pytest


class Attached:
    """What the served nodes read from upstream's RemoteOpenAIServer: its URLs, and no
    process of its own (serve.down then asks the health endpoint)."""

    proc = None

    def __init__(self, root: str) -> None:
        self.url_root = root.rstrip("/")

    def url_for(self, *parts: str) -> str:
        return "/".join([self.url_root, *(str(p).strip("/") for p in parts)])


@pytest.fixture(scope="session")
def server():
    yield Attached(os.environ["INFB_BASE_URL"])
'''


def main(source: str, snapshot: str, dest: str) -> None:
    src, out = Path(source), Path(dest).expanduser()
    tests = out / "tests"
    shutil.rmtree(tests, ignore_errors=True)
    shutil.copytree(src / "tests", tests, symlinks=False)
    config = json.loads((tests / "config.json").read_text())
    config["infb"]["model_id"] = SERVED
    config["infb"]["dataset_root"] = "/out/datasets"
    (tests / "config.json").write_text(json.dumps(config, indent=1))
    # The verifier asserts config.json's model id is the grading recipe's MODEL; both name
    # the served model.
    recipe = tests / "eval" / "deepseek_v41_flash" / "recipe.py"
    text = recipe.read_text()
    model_line = next(line for line in text.splitlines() if line.startswith("MODEL = "))
    recipe.write_text(text.replace(model_line, f'MODEL = "{SERVED}"', 1))
    (tests / "infb_verify" / "conftest.py").write_text(CONFTEST)
    hidden = json.loads((tests / "hidden" / "fidelity.json").read_text())
    (tests / "hidden-as-examples.json").write_text(json.dumps(
        {"meta": {"source": "tests/hidden/fidelity.json", "provenance": hidden.get("provenance")},
         "examples": hidden["sequences"]}))
    model = out / "model"
    shutil.rmtree(model, ignore_errors=True)
    model.mkdir(parents=True)
    for name in ("tokenizer.json", "tokenizer_config.json", "config.json", "generation_config.json"):
        path = Path(snapshot) / name
        if path.exists():
            shutil.copyfile(path.resolve(), model / name)
    print(f"prepared {tests} and {model}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
