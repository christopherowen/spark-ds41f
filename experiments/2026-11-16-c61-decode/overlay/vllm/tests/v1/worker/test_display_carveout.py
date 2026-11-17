# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the RM scanout display carve-out placement (display_carveout.py)."""

import ctypes
import os
import pathlib
import subprocess
import sys
import time

import pytest
import torch

from vllm.v1.worker import display_carveout

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
needs_rm = pytest.mark.skipif(
    not os.access("/dev/nvidiactl", os.R_OK | os.W_OK), reason="no NVIDIA RM node"
)


class _Table(torch.nn.Module):
    """A vocabulary layer: its weight comes from allocate when required."""

    def __init__(self, rows: int, width: int, device: str):
        super().__init__()
        data = display_carveout.allocate((rows, width), torch.bfloat16, device)
        if data is None:
            data = torch.empty(rows, width, dtype=torch.bfloat16, device=device)
        data.copy_(
            torch.randn(width, dtype=torch.bfloat16, device=device).expand(rows, width)
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


def _open_fds_for(path: str) -> list[str]:
    fds = pathlib.Path("/proc/self/fd")
    out: list[str] = []
    for fd in fds.iterdir():
        try:
            if os.path.realpath(fd) == path:
                out.append(fd.name)
        except OSError:
            continue
    return out


@pytest.fixture
def tables(monkeypatch):
    carveouts = []
    monkeypatch.setattr(display_carveout, "VocabParallelEmbedding", _Table)
    monkeypatch.setattr(display_carveout, "_REQUIRED", False)
    monkeypatch.setattr(display_carveout, "_carveouts", carveouts)
    monkeypatch.setattr(display_carveout, "_integrity", None)
    yield
    for carveout in list(carveouts):
        carveout.close()


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


def test_struct_layouts_match_rm_and_cuda_abi():
    assert ctypes.sizeof(display_carveout._NvOs21Params) == 32
    assert ctypes.sizeof(display_carveout._NvOs00Params) == 16
    assert ctypes.sizeof(display_carveout._NvOs54Params) == 32
    assert ctypes.sizeof(display_carveout._Nv0000AllocParams) == 120
    assert ctypes.sizeof(display_carveout._Nv0080AllocParams) == 56
    assert ctypes.sizeof(display_carveout._NvMemoryAllocationParams) == 128
    assert ctypes.sizeof(display_carveout._Nv0000CtrlOsUnixExportObjectToFdParams) == 24
    assert ctypes.sizeof(display_carveout._Nv0000GpuIdInfoV2Params) == 32
    assert ctypes.sizeof(display_carveout._Nv0000UuidFromGpuIdParams) == 268
    assert ctypes.sizeof(display_carveout._CUmemLocation) == 8
    assert ctypes.sizeof(display_carveout._CUmemAccessDesc) == 12
    assert display_carveout._NvMemoryAllocationParams.attr.offset == 24
    assert display_carveout._NvMemoryAllocationParams.attr2.offset == 28
    assert display_carveout._NvMemoryAllocationParams.size.offset == 64
    assert display_carveout._NvMemoryAllocationParams.alignment.offset == 72
    assert display_carveout._Nv0000CtrlOsUnixExportObjectToFdParams.fd.offset == 16


def test_allocate_is_off_unless_required(monkeypatch):
    monkeypatch.setattr(display_carveout, "_REQUIRED", False)
    monkeypatch.setattr(display_carveout, "_RM_NODE", "/nonexistent/nvidiactl")
    assert display_carveout.allocate((8192, 4096), torch.bfloat16, "cuda") is None


def test_allocate_leaves_small_and_non_cuda_weights_alone(required, monkeypatch):
    # Neither opens the RM node, which does not exist here.
    monkeypatch.setattr(display_carveout, "_RM_NODE", "/nonexistent/nvidiactl")
    assert display_carveout.allocate((16, 64), torch.bfloat16, "cuda:0") is None
    assert display_carveout.allocate((8192, 4096), torch.bfloat16, "meta") is None
    assert display_carveout._carveouts == []


@needs_cuda
def test_selects_large_tables_once(tables):
    names = [name for name, _ in display_carveout.streamed_weights(_model("cuda"))]
    assert names == ["embed_tokens.weight", "lm_head.weight"]


@needs_cuda
def test_missing_rm_node_refuses(required, monkeypatch):
    monkeypatch.setattr(display_carveout, "_RM_NODE", "/nonexistent/nvidiactl")
    with pytest.raises(RuntimeError, match="display carve-out cannot hold"):
        display_carveout.allocate((8192, 4096), torch.bfloat16, "cuda")


@needs_cuda
@needs_rm
def test_weights_load_in_place_without_drm(required, monkeypatch):
    # Refuse any accidental access to /dev/dri paths.
    real_open = os.open

    def _guarded_open(path, flags, *args, **kwargs):
        assert "/dev/dri" not in os.fspath(path), f"unexpected DRM open: {path}"
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", _guarded_open)
    monkeypatch.setattr(display_carveout, "_CHECK_SECONDS", 300.0)
    torch.cuda.init()
    _ = torch.empty(1, device="cuda")
    torch.cuda.synchronize()
    rm_fds_before = len(_open_fds_for("/dev/nvidiactl"))
    free_before = torch.cuda.mem_get_info()[0]
    model = _model("cuda")
    torch.cuda.synchronize()
    # The 128 MiB of tables never touched ordinary memory (the small one did).
    assert free_before - torch.cuda.mem_get_info()[0] < 32 << 20
    assert len(display_carveout._carveouts) == 2
    # Both RM control and export FDs closed right after VMM import; no DRM FD opened.
    assert len(_open_fds_for("/dev/nvidiactl")) == rm_fds_before
    assert not _open_fds_for("/dev/dri/card0")

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


@needs_cuda
def test_weight_outside_the_carveout_refuses(tables):
    # Built with the carve-out not required: ordinary memory.
    model = _model("cuda")
    with pytest.raises(RuntimeError, match="not created in the display carve-out"):
        display_carveout.confirm_streamed_weights(model, torch.device("cuda"))


@needs_cuda
@needs_rm
def test_dropped_layer_refuses(required):
    model = _model("cuda")
    model.drafter.embed_tokens = _Table(8192, 4096, "cuda")
    del model.drafter.embed_tokens  # a placeholder replaced after loading
    with pytest.raises(RuntimeError, match="1 display carve-out buffers"):
        display_carveout.confirm_streamed_weights(model, torch.device("cuda"))


@needs_cuda
@needs_rm
def test_integrity_check_is_off_by_default(required, monkeypatch):
    monkeypatch.setattr(display_carveout, "_CHECK_SECONDS", 0.0)
    display_carveout.confirm_streamed_weights(_model("cuda"), torch.device("cuda"))
    assert display_carveout._integrity is None
    display_carveout.start_checks()


@needs_cuda
@needs_rm
@pytest.mark.parametrize("cacheable", [True, False])
def test_compact_shrinks_packed_carveout_buffer_in_place_preserving_cacheable(
    required, cacheable
):
    # 96 MiB unpacked BF16 table packed in-place into 64 MiB.
    unpacked = display_carveout.allocate(
        (12288, 4096), torch.bfloat16, "cuda", cacheable=cacheable
    )
    assert unpacked is not None
    assert display_carveout.total_bytes() == 96 << 20
    old_carveout = display_carveout._carveouts[0]
    assert old_carveout.cacheable is cacheable
    packed_view = unpacked.view(torch.uint8).reshape(-1)[: 64 << 20].clone()
    in_place_prefix = unpacked.view(torch.uint8).reshape(-1)[: 64 << 20]
    in_place_prefix.copy_(packed_view)
    del unpacked
    compacted = display_carveout.compact(in_place_prefix)
    del in_place_prefix
    assert display_carveout.total_bytes() == 64 << 20
    assert len(display_carveout._carveouts) == 1
    assert old_carveout._closed
    assert display_carveout._carveouts[0] is not old_carveout
    assert display_carveout._carveouts[0].cacheable is cacheable
    assert torch.equal(compacted, packed_view)


@needs_cuda
@needs_rm
def test_move_respects_max_total_bytes_and_tracks_extra_models(required, monkeypatch):
    monkeypatch.setattr(display_carveout, "_MAX_TOTAL_BYTES", 256 << 20)
    model = _model("cuda")  # 2 x 64 MiB = 128 MiB in carve-out
    speculator = torch.nn.Module()
    speculator.lm_head = model.lm_head
    # 60 MiB stays 60 MiB under 2 MiB VMM granularity (saving 4 MiB over 16 MiB).
    markov_data = display_carveout.allocate((30720, 1024), torch.bfloat16, "cuda")
    assert markov_data is not None
    speculator.markov_w1 = torch.nn.Embedding(
        30720, 1024, _weight=markov_data, _freeze=True
    )
    speculator.markov_w1._display_carveout_required = True
    assert display_carveout.total_bytes() == 188 << 20

    # First 64 MiB decode-only expert slab fits within the 256 MiB cap (188 + 64 = 252 <= 256).
    w13_raw = torch.randint(0, 255, (64 << 20,), dtype=torch.uint8, device="cuda")
    w13_moved = display_carveout.move(w13_raw)
    assert w13_moved.data_ptr() != w13_raw.data_ptr()
    assert torch.equal(w13_moved, w13_raw)
    assert display_carveout.total_bytes() == 252 << 20
    assert display_carveout._carveouts[-1].cacheable is False

    # Second 64 MiB decode-only expert slab exceeds the 256 MiB cap and stays in regular memory.
    w2_raw = torch.randint(0, 255, (64 << 20,), dtype=torch.uint8, device="cuda")
    w2_moved = display_carveout.move(w2_raw)
    assert w2_moved.data_ptr() == w2_raw.data_ptr()
    assert display_carveout.total_bytes() == 252 << 20

    speculator.experts = torch.nn.Module()
    speculator.experts.w13_weight = torch.nn.Parameter(w13_moved, requires_grad=False)
    speculator.experts.w2_weight = torch.nn.Parameter(w2_moved, requires_grad=False)

    names = [name for name, _ in display_carveout.streamed_weights(model, speculator)]
    assert names == [
        "embed_tokens.weight",
        "lm_head.weight",
        "markov_w1.weight",
        "experts.w13_weight",
    ]
    display_carveout.confirm_streamed_weights(model, torch.device("cuda"), speculator)


@needs_cuda
@needs_rm
def test_markov_w1_outside_carveout_refuses(required):
    model = _model("cuda")
    speculator = torch.nn.Module()
    speculator.markov_w1 = torch.nn.Embedding(30720, 1024, device="cuda", dtype=torch.bfloat16)
    speculator.markov_w1._display_carveout_required = True
    with pytest.raises(RuntimeError, match="markov_w1.weight"):
        display_carveout.confirm_streamed_weights(
            model, torch.device("cuda"), speculator
        )


@needs_cuda
@needs_rm
def test_two_mib_granule_console_headroom_and_no_fd_leaks(tables, monkeypatch):
    # Verify optional mode works independently of _REQUIRED and keeps 2 MiB alignment.
    monkeypatch.setattr(display_carveout, "_REQUIRED", False)
    torch.cuda.init()
    _ = torch.empty(1, device="cuda")
    rm_fds_before = len(_open_fds_for("/dev/nvidiactl"))

    t = display_carveout.allocate(
        (61 << 20,), torch.uint8, "cuda", optional=True, cacheable=False
    )
    assert t is not None
    assert display_carveout.total_bytes() == 62 << 20
    assert display_carveout._carveouts[0].cacheable is False
    t.fill_(0xA5)
    torch.cuda.synchronize()
    assert bool((t == 0xA5).all().item())
    del t
    display_carveout._carveouts.pop().close()

    # Repeated alloc/close must leak neither RM handles, FDs, nor scanout capacity.
    for i in range(6):
        c = display_carveout.allocate(
            (64 << 20,), torch.uint8, "cuda", optional=True, cacheable=(i % 2 == 0)
        )
        assert c is not None
        c.fill_(i + 1)
        torch.cuda.synchronize()
        assert int(c[0].item()) == i + 1
        del c
        display_carveout._carveouts.pop().close()
    assert len(_open_fds_for("/dev/nvidiactl")) == rm_fds_before

    # Enforce _MAX_TOTAL_BYTES headroom for console framebuffer (/dev/fb0).
    assert (
        display_carveout.allocate(
            (display_carveout._MAX_TOTAL_BYTES + (2 << 20),),
            torch.uint8,
            "cuda",
            optional=True,
        )
        is None
    )
    monkeypatch.setattr(display_carveout, "_REQUIRED", True)
    with pytest.raises(RuntimeError, match="display carve-out limit"):
        display_carveout.allocate(
            (display_carveout._MAX_TOTAL_BYTES + (2 << 20),),
            torch.uint8,
            "cuda",
        )


@needs_cuda
@needs_rm
def test_cross_process_isolation_and_sigkill_reclamation(required):
    # Child process initializes CUDA first via display_carveout.allocate (cold start)
    # and holds 1024 MiB until killed.
    child_code = """
import os, time, torch
from vllm.v1.worker import display_carveout
display_carveout._REQUIRED = True
t = display_carveout.allocate((1024 << 20,), torch.uint8, 'cuda')
assert t is not None
t.fill_(0x5A)
torch.cuda.synchronize()
print('READY', flush=True)
time.sleep(60)
"""
    proc = subprocess.Popen(
        [sys.executable, "-c", child_code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        ready = proc.stdout.readline().strip()
        assert ready == "READY", proc.stderr.read()
        # Over-allocation beyond remaining scanout reserve fails cleanly.
        with pytest.raises(RuntimeError, match="display carve-out cannot hold"):
            display_carveout.allocate((1200 << 20,), torch.uint8, "cuda")
        # Concurrent non-overlapping allocation in parent succeeds intact.
        t_parent = display_carveout.allocate(
            (128 << 20,), torch.uint8, "cuda", cacheable=False
        )
        assert t_parent is not None
        t_parent.fill_(0x3C)
        torch.cuda.synchronize()
        assert bool((t_parent == 0x3C).all().item())
        del t_parent
        display_carveout._carveouts.pop().close()
    finally:
        proc.kill()
        proc.wait()

    # Reclaimed immediately after child SIGKILL: 1968 MiB fits cleanly.
    t_full = display_carveout.allocate((1968 << 20,), torch.uint8, "cuda")
    assert t_full is not None
    t_full[0] = 7
    t_full[-1] = 9
    torch.cuda.synchronize()
    assert int(t_full[0].item()) == 7 and int(t_full[-1].item()) == 9
    del t_full
    display_carveout._carveouts.pop().close()
