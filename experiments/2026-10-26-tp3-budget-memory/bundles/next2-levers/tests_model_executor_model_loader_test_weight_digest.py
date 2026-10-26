# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The weight digest (SPARK_DS41F_WEIGHT_DIGEST): every parameter's bytes in name
order whatever the layout or staging size, logged in the line the lab's loader report
reads, and taken after loading, before post-load processing. Needs no GPU."""

import hashlib
from types import SimpleNamespace

import pytest
import regex as re
import torch

from vllm.model_executor.model_loader import base_loader, weight_digest

# experiments/2026-10-25-final-names/loader/loader_report.py in spark-ds41f.
REPORT = re.compile(r"weight digest (?:rank (\d+) )?([0-9a-f]{64})")


def _model() -> torch.nn.Module:
    model = torch.nn.Module()
    model.b = torch.nn.Linear(24, 40)
    model.a = torch.nn.Module()
    model.a.scale = torch.nn.Parameter(torch.tensor(0.25), requires_grad=False)
    codes = torch.arange(96, dtype=torch.uint8).reshape(8, 12)
    model.a.codes = torch.nn.Parameter(
        codes.view(torch.float8_e4m3fn), requires_grad=False
    )
    model.c = torch.nn.Parameter(
        torch.arange(60, dtype=torch.float32).reshape(6, 10).to(torch.bfloat16),
        requires_grad=False,
    )
    model.mask = torch.nn.Parameter(torch.arange(9) % 2 == 0, requires_grad=False)
    model.shared = model.b.weight  # the same parameter under a later name
    return model


def _expected(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for _, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        flat = parameter.detach().contiguous().reshape(-1).view(torch.uint8)
        digest.update(flat.numpy().tobytes())
    return digest.hexdigest()


@pytest.mark.parametrize("staging_bytes", [1, 7, 64, 1 << 20])
def test_digest_is_every_parameter_in_name_order_at_any_staging_size(staging_bytes):
    model = _model()
    sha256, count, total = weight_digest.weight_digest(model, staging_bytes)
    assert sha256 == _expected(model)
    assert count == 6  # model.shared is model.b.weight
    assert total == sum(p.numel() * p.element_size() for p in model.parameters())


def test_strided_parameters_hash_their_logical_order():
    model = torch.nn.Module()
    backing = torch.arange(12 * 34, dtype=torch.float32).reshape(12, 34)
    model.weight = torch.nn.Parameter(backing.t()[3:30], requires_grad=False)
    assert not model.weight.is_contiguous()
    expected = hashlib.sha256(
        model.weight.detach().contiguous().numpy().tobytes()
    ).hexdigest()
    for staging_bytes in (4, 48, 200, 1 << 16):
        assert weight_digest.weight_digest(model, staging_bytes)[0] == expected


def test_one_changed_byte_changes_the_digest():
    model = _model()
    before = weight_digest.weight_digest(model)[0]
    with torch.no_grad():
        model.c.view(torch.int16)[5, 3] ^= 1
    assert weight_digest.weight_digest(model)[0] != before


def test_meta_parameters_are_refused():
    with torch.device("meta"):
        model = torch.nn.Linear(4, 4)
    with pytest.raises(RuntimeError, match="bias has no storage after loading"):
        weight_digest.weight_digest(model)


def test_the_logged_line_is_the_one_the_loader_report_reads(monkeypatch):
    lines = []
    monkeypatch.setattr(
        weight_digest.logger,
        "info",
        lambda message, *args: lines.append(message % args),
    )
    model = _model()
    weight_digest.log_weight_digest(model)
    assert len(lines) == 1
    found = REPORT.search(lines[0])
    assert found and found.group(1) == "0" and found.group(2) == _expected(model)
    assert " Module: 6 parameters, " in lines[0]


@pytest.mark.parametrize("enabled", [False, True])
def test_load_model_digests_after_loading_and_before_processing(monkeypatch, enabled):
    events = []

    class Loader(base_loader.BaseModelLoader):
        def download_model(self, model_config):
            pass

        def create_model(self, vllm_config, model_config, prefix=""):
            return torch.nn.Linear(4, 4)

        def load_weights(self, model, model_config):
            events.append("load")

    monkeypatch.setattr(weight_digest, "ENABLED", enabled)
    monkeypatch.setattr(
        weight_digest, "log_weight_digest", lambda model: events.append("digest")
    )
    monkeypatch.setattr(
        base_loader,
        "process_weights_after_loading",
        lambda *args: events.append("process"),
    )
    config = SimpleNamespace(
        device_config=SimpleNamespace(device="cpu"),
        load_config=SimpleNamespace(device=None),
    )
    Loader(SimpleNamespace()).load_model(config, SimpleNamespace(dtype=torch.float32))
    assert events == (["load", "digest", "process"] if enabled else ["load", "process"])


def test_only_the_value_1_turns_the_digest_on(monkeypatch):
    import importlib

    try:
        for value, enabled in ((None, False), ("0", False), ("", False), ("1", True)):
            if value is None:
                monkeypatch.delenv("SPARK_DS41F_WEIGHT_DIGEST", raising=False)
            else:
                monkeypatch.setenv("SPARK_DS41F_WEIGHT_DIGEST", value)
            assert importlib.reload(weight_digest).ENABLED is enabled
    finally:
        monkeypatch.undo()
        importlib.reload(weight_digest)
