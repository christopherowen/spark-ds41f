# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The API server's imports leave the process's root logging as it was.

model_hosting_container_standards sets the root logger to ERROR, and raises every
root handler to it, when it is first imported; the server imports it for the
SageMaker routes, so third-party warnings were dropped from ``vllm serve``'s log.
vllm.entrypoints.serve.sagemaker.standards puts the levels back unless
SAGEMAKER_CONTAINER_LOG_LEVEL or LOG_LEVEL asks for the library's behavior.

Each case runs in a fresh interpreter: the side effect happens once, at the first
import in a process.
"""

import json
import logging
import os
import subprocess
import sys
import textwrap

SCRIPT = textwrap.dedent(
    """
    import json
    import logging

    root = logging.getLogger()
    handler = logging.StreamHandler()  # a deployment's root handler
    root.addHandler(handler)
    before = [root.level, handler.level]

    import vllm.entrypoints.cli.serve  # noqa: F401  the server's imports
    import model_hosting_container_standards  # noqa: F401

    seen = []

    class Probe(logging.Handler):
        def emit(self, record):
            seen.append(record.getMessage())

    root.addHandler(Probe())
    logging.getLogger("a_third_party_library").warning("a third-party warning")
    print(json.dumps({"before": before, "after": [root.level, handler.level],
                      "seen": seen}))
    """
)


def _run(**variables: str) -> dict:
    env = {
        name: value
        for name, value in os.environ.items()
        if name not in ("SAGEMAKER_CONTAINER_LOG_LEVEL", "LOG_LEVEL")
    }
    env.update(variables)
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_a_warning_reaches_the_root_logger_after_the_servers_imports():
    report = _run()
    assert report["after"] == report["before"]
    assert report["seen"] == ["a third-party warning"]


def test_an_explicit_sagemaker_log_level_still_applies():
    report = _run(SAGEMAKER_CONTAINER_LOG_LEVEL="ERROR")
    assert report["after"] == [logging.ERROR, logging.ERROR]
    assert report["seen"] == []
