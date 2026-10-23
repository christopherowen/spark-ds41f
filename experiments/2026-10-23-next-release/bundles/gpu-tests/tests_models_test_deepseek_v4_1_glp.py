# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLP steering: the fail-closed reader, the projection, and the off state."""

import math
import os
import struct

import numpy as np
import pytest
import torch

from vllm import envs
from vllm.models.deepseek_v4_1 import glp

NUM_LAYERS = 8
HIDDEN = 16
STREAMS = 4
LAYERS = (1, 2, 5)
METADATA = {
    "general.base_model.0.name": "test-model",
    "glp.mode": "project",
    "glp.spec_version": 1,
    "glp.rank": 1,
    "glp.hook_point": glp.HOOK_POINT,
    "glp.alpha_default": 0.75,
    "glp.layer_ids_zero_based": "1,2,5",
}
STEER_ENV = (
    "WEIGHTLESS_STEER_PATH",
    "WEIGHTLESS_STEER_ALPHA",
    "WEIGHTLESS_STEER_LAYERS",
)


def _string(text: str) -> bytes:
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def _value(value) -> bytes:
    if isinstance(value, bool):
        return struct.pack("<I?", 7, value)
    if isinstance(value, int):
        return struct.pack("<II", 4, value)
    if isinstance(value, float):
        return struct.pack("<If", 6, value)
    return struct.pack("<I", 8) + _string(value)


def write_gguf(path, metadata, tensors, *, version=3, kind=0, trailing=b""):
    """A GGUF file as llama.cpp's writer lays it out, aligned to 32 bytes."""
    header = b"GGUF" + struct.pack("<IQQ", version, len(tensors), len(metadata))
    for key, value in metadata.items():
        header += _string(key) + _value(value)
    blobs, offset = [], 0
    for name, values in tensors.items():
        values = np.asarray(values, dtype="<f4")
        dims = values.shape[::-1]
        header += _string(name) + struct.pack(f"<I{len(dims)}Q", len(dims), *dims)
        header += struct.pack("<IQ", kind, offset)
        blob = values.tobytes()
        blob += b"\0" * (-len(blob) % 32)
        blobs.append(blob)
        offset += len(blob)
    header += b"\0" * (-len(header) % 32)
    path.write_bytes(header + b"".join(blobs) + trailing)
    return str(path)


def unit(seed: int, width: int = HIDDEN) -> np.ndarray:
    vector = np.random.default_rng(seed).standard_normal(width)
    return (vector / np.linalg.norm(vector)).astype(np.float32)


def glp_file(tmp_path, metadata=None, tensors=None, **kwargs):
    merged = dict(METADATA)
    for key, value in (metadata or {}).items():
        if value is None:
            merged.pop(key, None)
        else:
            merged[key] = value
    if tensors is None:
        tensors = {f"direction.{layer}": unit(layer) for layer in LAYERS}
    return write_gguf(tmp_path / "vector.gguf", merged, tensors, **kwargs)


@pytest.fixture(autouse=True)
def _no_steering_env(monkeypatch):
    for name in STEER_ENV:
        monkeypatch.delenv(name, raising=False)


# The container reader.


def test_reader_returns_the_metadata_and_tensors(tmp_path):
    path = glp_file(tmp_path)
    metadata, tensors = glp.read_gguf(path)
    assert metadata["glp.mode"] == "project"
    assert metadata["glp.spec_version"] == 1
    assert metadata["glp.alpha_default"] == pytest.approx(0.75)
    assert sorted(tensors) == [f"direction.{layer}" for layer in LAYERS]
    for layer in LAYERS:
        np.testing.assert_array_equal(tensors[f"direction.{layer}"], unit(layer))


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"version": 2}, "GGUF version 2"),
        ({"kind": 1}, "not F32"),
        ({"trailing": b"\x7f" * 64}, "unexpected bytes"),
        ({"tensors": {"direction.1": np.zeros((2, HIDDEN))}}, "not 1-D"),
    ],
)
def test_reader_refuses_what_it_cannot_read(tmp_path, kwargs, message):
    with pytest.raises(ValueError, match=message):
        glp.read_gguf(glp_file(tmp_path, **kwargs))


def test_reader_refuses_a_file_that_is_not_gguf(tmp_path):
    path = tmp_path / "vector.gguf"
    path.write_bytes(b"PK\x03\x04" + b"\0" * 60)
    with pytest.raises(ValueError, match="not a GGUF file"):
        glp.read_gguf(str(path))


def test_reader_refuses_a_truncated_file(tmp_path):
    glp_file(tmp_path)
    path = tmp_path / "vector.gguf"
    path.write_bytes(path.read_bytes()[:-40])
    with pytest.raises(ValueError, match="outside the file|truncated"):
        glp.read_gguf(str(path))


# The GLP contract.


def test_directions_land_on_their_own_layer_numbers(tmp_path):
    directions, alpha = glp.read_glp(glp_file(tmp_path), NUM_LAYERS, HIDDEN)
    assert sorted(directions) == list(LAYERS)
    assert alpha == pytest.approx(0.75)
    for layer in LAYERS:
        np.testing.assert_allclose(directions[layer], unit(layer), atol=1e-7)
        assert np.linalg.norm(directions[layer]) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize(
    "metadata, tensors, message",
    [
        ({"glp.mode": None}, None, "absent mode means 'add'"),
        ({"glp.mode": "add"}, None, "never adds"),
        ({"glp.spec_version": None}, None, "spec_version"),
        ({"glp.spec_version": 2}, None, "spec_version"),
        ({"glp.rank": 2}, None, "rank"),
        ({"glp.hook_point": None}, None, "hook_point"),
        ({"glp.hook_point": "ffn_out_pre_residual"}, None, "hook_point"),
        ({"glp.alpha_default": math.nan}, None, "alpha_default"),
        ({"glp.alpha_default": math.inf}, None, "alpha_default"),
        ({"glp.layer_ids_zero_based": "1,2,6"}, None, "disagrees"),
        ({}, {"direction.0": unit(0)}, "direction.0"),
        ({}, {f"direction.{NUM_LAYERS}": unit(9)}, "outside layers"),
        ({}, {"direction.1": unit(1, HIDDEN + 1)}, "width"),
        ({}, {"direction.1": np.full(HIDDEN, np.nan)}, "not finite"),
        ({}, {"direction.1": 2 * unit(1)}, "norm"),
        ({}, {"bias.1": unit(1)}, "not a direction"),
        ({}, {}, "no direction tensors"),
    ],
)
def test_reader_refuses_files_it_cannot_apply_exactly(
    tmp_path, metadata, tensors, message
):
    if tensors is not None:
        metadata = {"glp.layer_ids_zero_based": None, **metadata}
    path = glp_file(tmp_path, metadata=metadata, tensors=tensors)
    with pytest.raises(ValueError, match=message):
        glp.read_glp(path, NUM_LAYERS, HIDDEN)


def test_a_transferred_vector_is_read(tmp_path):
    path = glp_file(tmp_path, metadata={"glp.derived_at": "ffn_out_pre_residual"})
    directions, _ = glp.read_glp(path, NUM_LAYERS, HIDDEN)
    assert sorted(directions) == list(LAYERS)


# Steering on and off.


def test_no_path_means_no_steering_module(monkeypatch):
    monkeypatch.setenv("WEIGHTLESS_STEER_ALPHA", "not a number")
    monkeypatch.setenv("WEIGHTLESS_STEER_LAYERS", "99")
    assert glp.glp_steering(NUM_LAYERS, HIDDEN) is None


def test_steering_stacks_the_file_directions(tmp_path, monkeypatch):
    monkeypatch.setenv("WEIGHTLESS_STEER_PATH", glp_file(tmp_path))
    steering = glp.glp_steering(NUM_LAYERS, HIDDEN)
    assert steering.directions.shape == (NUM_LAYERS, HIDDEN)
    assert steering.directions.dtype == torch.float32
    for layer in range(NUM_LAYERS):
        expected = unit(layer) if layer in LAYERS else np.zeros(HIDDEN)
        np.testing.assert_allclose(steering.directions[layer], expected, atol=1e-7)
    assert steering.alpha.item() == pytest.approx(0.75)
    # Buffers, never weights: the checkpoint loader must not see them.
    assert dict(steering.named_parameters()) == {}
    assert steering.state_dict() == {}


def test_alpha_and_layers_can_be_overridden(tmp_path, monkeypatch):
    monkeypatch.setenv("WEIGHTLESS_STEER_PATH", glp_file(tmp_path))
    monkeypatch.setenv("WEIGHTLESS_STEER_ALPHA", "0.5")
    monkeypatch.setenv("WEIGHTLESS_STEER_LAYERS", "2, 5")
    steering = glp.glp_steering(NUM_LAYERS, HIDDEN)
    assert steering.alpha.item() == pytest.approx(0.5)
    assert torch.count_nonzero(steering.directions.abs().sum(-1)) == 2
    assert steering.directions[1].abs().sum() == 0


@pytest.mark.parametrize(
    "env, message",
    [
        ({"WEIGHTLESS_STEER_ALPHA": "strong"}, "WEIGHTLESS_STEER_ALPHA"),
        ({"WEIGHTLESS_STEER_ALPHA": "inf"}, "WEIGHTLESS_STEER_ALPHA"),
        ({"WEIGHTLESS_STEER_LAYERS": "1,x"}, "comma list"),
        ({"WEIGHTLESS_STEER_LAYERS": "1,3"}, "without a direction"),
        ({"WEIGHTLESS_STEER_LAYERS": ","}, "without a direction"),
    ],
)
def test_malformed_overrides_are_refused(tmp_path, monkeypatch, env, message):
    monkeypatch.setenv("WEIGHTLESS_STEER_PATH", glp_file(tmp_path))
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=message):
        glp.glp_steering(NUM_LAYERS, HIDDEN)


def test_a_missing_file_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("WEIGHTLESS_STEER_PATH", str(tmp_path / "missing.gguf"))
    with pytest.raises(FileNotFoundError):
        glp.glp_steering(NUM_LAYERS, HIDDEN)


def test_no_alpha_anywhere_is_refused(tmp_path, monkeypatch):
    path = glp_file(tmp_path, metadata={"glp.alpha_default": None})
    monkeypatch.setenv("WEIGHTLESS_STEER_PATH", path)
    with pytest.raises(ValueError, match="no glp.alpha_default"):
        glp.glp_steering(NUM_LAYERS, HIDDEN)


def test_the_steering_variables_are_part_of_the_compile_key(monkeypatch):
    unsteered = envs.compile_factors()
    for name in STEER_ENV:
        assert name in unsteered
    monkeypatch.setenv("WEIGHTLESS_STEER_PATH", "/vectors/a.gguf")
    assert envs.compile_factors() != unsteered


# The projection.


def steering(alpha: float, layers=LAYERS) -> glp.GLPSteering:
    directions = torch.zeros(NUM_LAYERS, HIDDEN)
    for layer in layers:
        directions[layer] = torch.from_numpy(unit(layer))
    return glp.GLPSteering(directions, alpha)


def streams(dtype=torch.float32, tokens: int = 7) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.randn(tokens, STREAMS, HIDDEN, generator=generator).to(dtype)


@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.75, 1.0, 2.0])
def test_projection_scales_only_the_direction(alpha):
    d = torch.from_numpy(unit(2))
    h = streams()
    out = steering(alpha)(h, 2)
    torch.testing.assert_close(out @ d, (1 - alpha) * (h @ d), atol=1e-5, rtol=0)
    rest = lambda x: x - (x @ d).unsqueeze(-1) * d  # noqa: E731
    torch.testing.assert_close(rest(out), rest(h), atol=1e-5, rtol=0)


def test_every_stream_is_projected_so_their_mean_is_too():
    d = torch.from_numpy(unit(5))
    h = streams()
    out = steering(1.0)(h, 5)
    torch.testing.assert_close(out @ d, torch.zeros(7, STREAMS), atol=1e-5, rtol=0)
    mean = h.mean(dim=1)
    torch.testing.assert_close(
        out.mean(dim=1), mean - (mean @ d).unsqueeze(-1) * d, atol=1e-5, rtol=0
    )


@pytest.mark.parametrize("alpha", [0.0, 0.75, 1.0, 2.0])
def test_projection_never_adds_to_a_stream(alpha):
    h = streams()
    out = steering(alpha)(h, 1)
    assert (out.norm(dim=-1) <= h.norm(dim=-1) + 1e-5).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_an_unsteered_layer_is_unchanged(dtype):
    h = streams(dtype)
    out = steering(0.75)(h, 3)
    assert out.dtype == dtype
    assert torch.equal(out, h)


def test_alpha_is_read_when_the_projection_runs():
    module = steering(1.0)
    h = streams()
    full = module(h, 1)
    module.alpha.fill_(0.0)
    assert torch.equal(module(h, 1), h)
    assert not torch.equal(full, h)


# The model loop's step.


class _FoldingMHC(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def post(self, x, residual, post_mix, res_mix):
        self.calls += 1
        return post_mix.unsqueeze(-1) * x.unsqueeze(1) + res_mix @ residual


def _layer_outputs():
    generator = torch.Generator().manual_seed(1)
    x = torch.randn(3, HIDDEN, generator=generator)
    residual = torch.randn(3, STREAMS, HIDDEN, generator=generator)
    post_mix = torch.rand(3, STREAMS, generator=generator)
    res_mix = torch.rand(3, STREAMS, STREAMS, generator=generator)
    return x, residual, post_mix, res_mix


def test_unsteered_layer_outputs_pass_through_untouched():
    mhc = _FoldingMHC()
    outputs = _layer_outputs()
    passed = glp.steer_post_layer(None, mhc, 1, *outputs)
    assert all(a is b for a, b in zip(passed, outputs))
    assert mhc.calls == 0


def test_steered_layer_outputs_are_the_projected_fold():
    mhc = _FoldingMHC()
    outputs = _layer_outputs()
    module = steering(1.0)
    hidden_states, residual, post_mix, res_mix = glp.steer_post_layer(
        module, mhc, 5, *outputs
    )
    assert (residual, post_mix, res_mix) == (None, None, None)
    assert mhc.calls == 1
    fold = _FoldingMHC().post(*outputs)
    torch.testing.assert_close(hidden_states, module(fold, 5))
    d = torch.from_numpy(unit(5))
    torch.testing.assert_close(
        hidden_states @ d, torch.zeros(3, STREAMS), atol=1e-5, rtol=0
    )


# A real GLP file, when one is named (it is not redistributed with the tests).

REAL_FILE = os.environ.get("GLP_TEST_FILE")


@pytest.mark.skipif(not REAL_FILE, reason="GLP_TEST_FILE names no file")
def test_a_published_ds41_file_holds_only_unit_directions():
    metadata, tensors = glp.read_gguf(REAL_FILE)
    assert all(key.startswith(("general.", "glp.")) for key in metadata)
    assert all(name.startswith("direction.") for name in tensors)
    directions, alpha = glp.read_glp(REAL_FILE, 40, 5120)
    assert sorted(directions) == list(range(1, 40))
    assert alpha == pytest.approx(0.75)
    for direction in directions.values():
        assert np.linalg.norm(direction) == pytest.approx(1.0, abs=1e-5)
