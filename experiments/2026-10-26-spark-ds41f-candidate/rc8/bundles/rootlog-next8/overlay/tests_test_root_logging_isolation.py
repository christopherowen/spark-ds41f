# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A test that raises the root logger's handlers leaves the next test's log capture
working (tests/conftest.py's restore_root_logging_levels).

model_hosting_container_standards sets the root logger and its handlers to ERROR when
it is first imported, during whichever test imports the SageMaker routes first. pytest
reuses its capture handlers for the session, so later tests' caplog dropped warnings
(the next release's release tests: two caplog_vllm tests of test_kv_cache_utils.py
after tests/quantization/test_register_quantization_config.py's dist_init). The tests
run in file order: each "raises" test changes the levels, the test after it captures.
"""

import logging

import pytest

LOGGER = "tests.test_root_logging_isolation"


def test_a_test_raises_the_root_logging_levels():
    root = logging.getLogger()
    root.setLevel(logging.ERROR)
    for handler in root.handlers:
        handler.setLevel(logging.ERROR)


def test_the_next_test_captures_warnings(caplog):
    root = logging.getLogger()
    assert root.level < logging.ERROR
    assert all(handler.level < logging.ERROR for handler in root.handlers)
    logging.getLogger(LOGGER).warning("captured after raised root levels")
    assert "captured after raised root levels" in caplog.text


def test_the_sagemaker_standards_raise_the_root_logging_levels(monkeypatch):
    logging_config = pytest.importorskip(
        "model_hosting_container_standards.logging_config"
    )
    monkeypatch.delenv("SAGEMAKER_CONTAINER_LOG_LEVEL", raising=False)
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    # What its import does: the root logger and every handler below it go to ERROR.
    logging_config.configure_root_logger()
    root = logging.getLogger()
    assert root.level == logging.ERROR
    assert all(handler.level >= logging.ERROR for handler in root.handlers)


def test_the_next_test_captures_warnings_again(caplog):
    logging.getLogger(LOGGER).warning("captured after the sagemaker standards")
    assert "captured after the sagemaker standards" in caplog.text
