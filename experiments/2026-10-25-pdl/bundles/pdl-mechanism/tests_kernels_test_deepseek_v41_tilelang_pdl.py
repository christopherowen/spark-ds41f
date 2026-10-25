# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Programmatic dependent launch on DeepSeek-V4.1's decode chain (``tilelang/pdl.py``).

PDL changes when a kernel launches, never what it computes. These tests check that
the launches are programmatic in CUDA graphs and that nothing reads early:

- a reader that loads its input before ``griddepcontrol.wait`` sees a slow
  producer's buffer before the producer writes it (the edge is programmatic: the
  reader runs beside the producer), and a reader that waits first never does;
- one decode layer's chain (``deepseek_v41_pdl_chains``), replayed in a CUDA graph
  with every intermediate buffer poisoned with NaN before each replay, gives its
  eager outputs bit for bit on every replay;
- TileKernels compiles with its PDL switch on.

``deepseek_v41_pdl_sources`` (a pytest plugin) checks the compiled sources of every
chain kernel that the DS4.1 kernel tests build: the wait comes before any access to
global memory.
"""

import pytest
import torch

from vllm.platforms import current_platform

if not (
    current_platform.is_cuda() and current_platform.is_device_capability_family(120)
):
    pytest.skip("TileLang V4.1 kernels target SM120/SM121", allow_module_level=True)

tilelang = pytest.importorskip("tilelang")
pytest.importorskip("tile_kernels")

import tilelang.language as T  # noqa: E402

from tests.kernels.deepseek_v41_pdl_chains import ROWS, LayerChain  # noqa: E402
from vllm.models.deepseek_v4_1.tilelang import pdl  # noqa: E402

DEVICE = "cuda"
REPLAYS = 20
THREADS = 128
# FMA steps a producer thread runs before it writes: tens of microseconds on GB10,
# far longer than a launch.
SPIN = 20000
POISON = -1.0


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def slow_copy(n: int):
    """``dst = src`` after a spin, on the PDL chain."""

    @T.prim_func
    def pdl_slow_copy(
        src: T.Tensor((n,), T.float32),
        dst: T.Tensor((n,), T.float32),
    ):
        with T.Kernel(T.ceildiv(n, THREADS), threads=THREADS) as bx:
            pdl.chain()
            i = bx * THREADS + T.get_thread_binding()
            acc = T.alloc_var(T.float32, init=src[i])
            for _ in T.serial(SPIN):
                acc = acc * 0.999999 + 1e-6
            # A comparison the compiler cannot fold keeps the spin.
            dst[i] = T.if_then_else(acc == 12345.5, POISON - 1.0, src[i])

    return pdl_slow_copy


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def reader(n: int, early: bool):
    """``out = dst``; with ``early``, ``dst`` is loaded before the wait."""

    @T.prim_func
    def pdl_reader(
        dst: T.Tensor((n,), T.float32),
        out: T.Tensor((n,), T.float32),
    ):
        with T.Kernel(T.ceildiv(n, THREADS), threads=THREADS) as bx:
            i = bx * THREADS + T.get_thread_binding()
            value = T.alloc_var(T.float32)
            if early:
                value = dst[i]
                T.pdl_sync()
            else:
                pdl.chain()
                value = dst[i]
            out[i] = value

    return pdl_reader


def _graph(fn) -> torch.cuda.CUDAGraph:
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def test_tile_kernels_pdl_on():
    tk = pdl.tile_kernels()
    assert tk.get_pdl()


@pytest.mark.parametrize("early", [True, False])
def test_reader_waits_for_its_producer(early, n=48 * THREADS):
    """In a CUDA graph a reader starts beside its producer: one that loads before
    its wait reads the producer's buffer unwritten; one that waits first reads what
    the producer wrote, on every replay."""
    src = torch.arange(n, dtype=torch.float32, device=DEVICE)
    dst = torch.empty_like(src)
    out = torch.empty_like(src)
    copy, read = slow_copy(n), reader(n, early)

    def chain():
        copy(src, dst)
        read(dst, out)

    graph = _graph(chain)
    early_reads = 0
    for _ in range(REPLAYS):
        dst.fill_(POISON)
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        stale = int((out == POISON).sum())
        if not early:
            assert stale == 0, f"{stale} elements read before the producer wrote them"
            assert torch.equal(out, src)
        early_reads += stale
    if early:
        # Without this the edges are not programmatic and PDL does nothing.
        assert early_reads > 0, "the early reader never ran beside its producer"


@pytest.mark.usefixtures("default_vllm_config")
@pytest.mark.parametrize("step", sorted(ROWS))
def test_layer_chain_replays_bit_equal(step):
    """The layer's chain in a CUDA graph, every intermediate poisoned before each
    replay: every replay gives the eager outputs' bits, and the output is finite."""
    chain = LayerChain(ROWS[step])
    chain.poison()
    chain.run()
    torch.cuda.synchronize()
    want = chain.digests()
    assert torch.isfinite(chain.out).all()
    graph = _graph(chain.run)
    for replay in range(REPLAYS):
        chain.poison()
        graph.replay()
        torch.cuda.synchronize()
        got = chain.digests()
        differ = [name for name in want if got[name] != want[name]]
        assert not differ, f"replay {replay}: {differ} differ from the eager run"
