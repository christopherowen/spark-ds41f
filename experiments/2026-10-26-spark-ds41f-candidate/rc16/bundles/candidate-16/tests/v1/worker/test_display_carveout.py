# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the display carve-out weight placement (display_carveout.py)."""

import ctypes
import os
import pathlib
import time

import pytest
import torch

from vllm.v1.worker import display_carveout

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
needs_drm = pytest.mark.skipif(
    not os.access("/dev/dri/card0", os.R_OK | os.W_OK), reason="no DRM node"
)


class _Table(torch.nn.Module):
    """A vocabulary layer: its weight comes from ``allocate`` when required."""

    def __init__(self, rows: int, width: int, device: str):
        super().__init__()
        data = display_carveout.allocate((rows, width), torch.bfloat16, device)
        if data is None:
            data = torch.empty(rows, width, dtype=torch.bfloat16, device=device)
        data.copy_(
            torch.randn(width, dtype=torch.bfloat16).expand(rows, width)
        )  # "loading"
        self.weight = torch.nn.Parameter(data, requires_grad=False)


def _model(device: str) -> torch.nn.Module:
    model = torch.nn.Module()
    model.embed_tokens = _Table(8192, 4096, device)  # 64 MiB
    model.lm_head = _Table(8192, 4096, device)
    model.small = _Table(16, 64, device)
    model.drafter = torch.nn.Module()
    model.drafter.lm_head = model.lm_head  # aliased, like the DSpark drafter
    return model


@pytest.fixture
def tables(monkeypatch):
    monkeypatch.setattr(display_carveout, "VocabParallelEmbedding", _Table)
    monkeypatch.setattr(display_carveout, "_REQUIRED", False)
    monkeypatch.setattr(display_carveout, "_carveouts", [])
    monkeypatch.setattr(display_carveout, "_integrity", None)


@pytest.fixture
def required(tables, monkeypatch):
    monkeypatch.setattr(display_carveout, "_REQUIRED", True)


def test_checksum_sees_one_flipped_bit():
    weight = torch.randn(64, 48, dtype=torch.bfloat16)
    before = int(display_carveout._checksum(weight))
    for bit in (0, 7, 15):
        flipped = weight.clone()
        flipped.view(torch.int16)[5, 9] ^= 1 << bit
        assert int(display_carveout._checksum(flipped)) != before, bit
    assert int(display_carveout._checksum(weight.clone())) == before


def test_chunked_checksum_matches_the_whole_sum(monkeypatch):
    weight = torch.randn(64, 48, dtype=torch.bfloat16)
    words = weight.reshape(-1).view(torch.int32)
    whole = int(words.sum(dtype=torch.int64))
    monkeypatch.setattr(display_carveout, "_CHECK_CHUNK_WORDS", 100)
    assert int(display_carveout._checksum(weight)) == whole


@needs_cuda
def test_checksum_memory_stays_bounded():
    table = torch.randn(8192, 4096, dtype=torch.bfloat16, device="cuda")  # 64 MiB
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    display_carveout._checksum(table)
    extra = torch.cuda.max_memory_allocated() - base
    assert extra <= 2 * 8 * display_carveout._CHECK_CHUNK_WORDS, extra


def test_integrity_check_stops_the_worker_after_a_change():
    weight = torch.randn(64, 48, dtype=torch.bfloat16)
    stopped: list[str] = []
    check = display_carveout._IntegrityCheck(
        [("lm_head.weight", weight)],
        [int(display_carveout._checksum(weight))],
        torch.device("cpu"),
        interval=0.01,
        on_failure=stopped.append,
    )
    check.check()
    assert check.failure is None and check.passed == 1
    weight[3, 4] = weight[3, 4] + 1
    check.start()
    for _ in range(500):
        if stopped:
            break
        time.sleep(0.01)
    assert stopped and "lm_head.weight" in stopped[0]


def test_start_checks_without_carveout_is_a_no_op(monkeypatch):
    monkeypatch.setattr(display_carveout, "_integrity", None)
    display_carveout.start_checks()


def test_struct_layouts_match_cuda_h():
    assert ctypes.sizeof(display_carveout._ExternalMemoryHandleDesc) == 104
    assert ctypes.sizeof(display_carveout._ExternalMemoryBufferDesc) == 88
    assert display_carveout._ExternalMemoryHandleDesc.size.offset == 24


def test_allocate_is_off_unless_required(monkeypatch):
    monkeypatch.setattr(display_carveout, "_REQUIRED", False)
    monkeypatch.setattr(display_carveout, "_DRM_NODE", "/nonexistent/card0")
    assert display_carveout.allocate((8192, 4096), torch.bfloat16, "cuda") is None


def test_allocate_leaves_small_and_non_cuda_weights_alone(required, monkeypatch):
    # Neither opens the DRM node, which does not exist here.
    monkeypatch.setattr(display_carveout, "_DRM_NODE", "/nonexistent/card0")
    assert display_carveout.allocate((16, 64), torch.bfloat16, "cuda:0") is None
    assert display_carveout.allocate((8192, 4096), torch.bfloat16, "meta") is None
    assert display_carveout._carveouts == []


@needs_cuda
def test_selects_large_tables_once(tables):
    names = [name for name, _ in display_carveout.streamed_weights(_model("cuda"))]
    assert names == ["embed_tokens.weight", "lm_head.weight"]


@needs_cuda
def test_missing_drm_node_refuses(required, monkeypatch):
    monkeypatch.setattr(display_carveout, "_DRM_NODE", "/nonexistent/card0")
    with pytest.raises(RuntimeError, match="display carve-out cannot hold"):
        display_carveout.allocate((8192, 4096), torch.bfloat16, "cuda")


@needs_cuda
@needs_drm
def test_weights_load_in_place(required, monkeypatch):
    monkeypatch.setattr(display_carveout, "_CHECK_SECONDS", 300.0)
    torch.cuda.synchronize()
    free_before = torch.cuda.mem_get_info()[0]
    model = _model("cuda")
    torch.cuda.synchronize()
    # The 128 MiB of tables never touched ordinary memory (the small one did).
    assert free_before - torch.cuda.mem_get_info()[0] < 32 << 20
    assert len(display_carveout._carveouts) == 2
    pointers = {name: p.data_ptr() for name, p in model.named_parameters()}
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    display_carveout.confirm_streamed_weights(model, torch.device("cuda"))
    for name, weight in model.named_parameters():
        assert weight.data_ptr() == pointers[name], name
        assert torch.equal(weight, before[name]), name
    assert model.drafter.lm_head.weight.data_ptr() == model.lm_head.weight.data_ptr()
    # With the check on, its baseline covers both tables and matches.
    check = display_carveout._integrity
    assert [name for name, _ in check.weights] == [
        "embed_tokens.weight",
        "lm_head.weight",
    ]
    check.check()
    assert check.failure is None
    # No DRM file stays open, so the worker never keeps DRM master (which would
    # stop the kernel's framebuffer console from drawing).
    fds = pathlib.Path("/proc/self/fd")
    assert not [fd for fd in fds.iterdir() if os.path.realpath(fd) == "/dev/dri/card0"]


@needs_cuda
def test_weight_outside_the_carveout_refuses(tables):
    # Built with the carve-out not required: ordinary memory.
    model = _model("cuda")
    with pytest.raises(RuntimeError, match="not created in the display carve-out"):
        display_carveout.confirm_streamed_weights(model, torch.device("cuda"))


@needs_cuda
@needs_drm
def test_dropped_layer_refuses(required):
    model = _model("cuda")
    model.drafter.embed_tokens = _Table(8192, 4096, "cuda")
    del model.drafter.embed_tokens  # a placeholder replaced after loading
    with pytest.raises(RuntimeError, match="1 display carve-out buffers"):
        display_carveout.confirm_streamed_weights(model, torch.device("cuda"))


@needs_cuda
@needs_drm
def test_integrity_check_is_off_by_default(required, monkeypatch):
    monkeypatch.setattr(display_carveout, "_CHECK_SECONDS", 0.0)
    display_carveout.confirm_streamed_weights(_model("cuda"), torch.device("cuda"))
    assert display_carveout._integrity is None
    display_carveout.start_checks()
