# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Programmatic dependent launch on DeepSeek-V4.1's decode chain (``tilelang/pdl.py``).

PDL changes when a kernel launches, never what it computes. These tests check that
the launches are programmatic in CUDA graphs and that nothing reads early:

- a reader that loads its input before ``griddepcontrol.wait`` sees a slow
  producer's buffer before the producer writes it (the edge is programmatic: the
  reader runs beside the producer), and a reader that waits first never does, in
  graph replays and in eager launches;
- the captured graph's edges, read back through the driver: a chain kernel's edge
  from the kernel before it is programmatic, and an edge into a kernel launched
  without PDL is not;
- one decode layer's chain (``deepseek_v41_pdl_chains``), replayed in a CUDA graph
  with every intermediate buffer poisoned with NaN before each replay, gives its
  eager outputs bit for bit on every replay;
- TileKernels' cast is chained when called through ``pdl`` and only then.

``deepseek_v41_pdl_sources`` (a pytest plugin) checks the compiled sources of every
chain kernel that the DS4.1 kernel tests build: the wait comes before any access to
global memory.
"""

import ctypes
import functools
import re

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
from tests.kernels.deepseek_v41_pdl_sources import CHAIN  # noqa: E402
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
    """``out = dst``; with ``early``, ``dst`` is loaded before the wait (a kernel the
    source check exempts by name)."""

    @T.prim_func
    def pdl_reader(
        dst: T.Tensor((n,), T.float32),
        out: T.Tensor((n,), T.float32),
    ):
        with T.Kernel(T.ceildiv(n, THREADS), threads=THREADS) as bx:
            pdl.chain()
            i = bx * THREADS + T.get_thread_binding()
            out[i] = dst[i]

    @T.prim_func
    def pdl_early_reader(
        dst: T.Tensor((n,), T.float32),
        out: T.Tensor((n,), T.float32),
    ):
        with T.Kernel(T.ceildiv(n, THREADS), threads=THREADS) as bx:
            i = bx * THREADS + T.get_thread_binding()
            value = T.alloc_var(T.float32)
            value = dst[i]
            T.pdl_sync()
            out[i] = value

    return pdl_early_reader if early else pdl_reader


def _graph(fn, keep: bool = False) -> torch.cuda.CUDAGraph:
    """``fn`` captured after one warm-up call; with ``keep``, the captured graph stays
    readable (``raw_cuda_graph``)."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph(keep_graph=keep)
    with torch.cuda.graph(graph):
        fn()
    return graph


class _EdgeData(ctypes.Structure):  # CUgraphEdgeData
    _fields_ = [
        ("from_port", ctypes.c_ubyte),
        ("to_port", ctypes.c_ubyte),
        ("type", ctypes.c_ubyte),
        ("reserved", ctypes.c_ubyte * 5),
    ]


class _KernelNodeParams(ctypes.Structure):  # CUDA_KERNEL_NODE_PARAMS_v2
    _fields_ = [
        ("func", ctypes.c_void_p),
        ("grid", ctypes.c_uint * 3),
        ("block", ctypes.c_uint * 3),
        ("shared_mem", ctypes.c_uint),
        ("params", ctypes.c_void_p),
        ("extra", ctypes.c_void_p),
        ("kern", ctypes.c_void_p),
        ("ctx", ctypes.c_void_p),
    ]


PROGRAMMATIC = 1  # CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC
KERNEL_NODE = 0  # CU_GRAPH_NODE_TYPE_KERNEL


def _edges(graph: torch.cuda.CUDAGraph) -> list[tuple[str | None, str | None, int]]:
    """``(source kernel, destination kernel, dependency type)`` for every edge of the
    captured graph; a node that is not a kernel is ``None``."""
    cu = ctypes.CDLL("libcuda.so.1")

    def check(result: int, what: str) -> None:
        assert result == 0, f"{what} returned CUresult {result}"

    handle = ctypes.c_void_p(graph.raw_cuda_graph())
    count = ctypes.c_size_t(0)
    check(
        cu.cuGraphGetEdges_v2(handle, None, None, None, ctypes.byref(count)),
        "cuGraphGetEdges_v2",
    )
    sources = (ctypes.c_void_p * count.value)()
    targets = (ctypes.c_void_p * count.value)()
    data = (_EdgeData * count.value)()
    check(
        cu.cuGraphGetEdges_v2(handle, sources, targets, data, ctypes.byref(count)),
        "cuGraphGetEdges_v2",
    )

    def name(node) -> str | None:
        kind = ctypes.c_int()
        check(
            cu.cuGraphNodeGetType(ctypes.c_void_p(node), ctypes.byref(kind)),
            "cuGraphNodeGetType",
        )
        if kind.value != KERNEL_NODE:
            return None
        params = _KernelNodeParams()
        node = ctypes.c_void_p(node)
        result = cu.cuGraphKernelNodeGetParams_v2(node, ctypes.byref(params))
        check(result, "cuGraphKernelNodeGetParams_v2")
        text = ctypes.c_char_p()
        if params.func:
            check(
                cu.cuFuncGetName(ctypes.byref(text), ctypes.c_void_p(params.func)),
                "cuFuncGetName",
            )
        else:
            check(
                cu.cuKernelGetName(ctypes.byref(text), ctypes.c_void_p(params.kern)),
                "cuKernelGetName",
            )
        return text.value.decode()

    return [
        (name(sources[i]), name(targets[i]), data[i].type) for i in range(count.value)
    ]


def _chained(kernel: str) -> bool:
    """Whether a compiled kernel of the layer chain begins with ``pdl.chain()`` or is
    TileKernels' with its PDL switch on."""
    base = re.sub(r"_kernel(_\d+)?$", "", kernel)
    return (
        base in CHAIN
        or base == "gemm_mxfp8"
        or base.startswith(("per_token_cast", "norm_forward"))
    )


def test_tile_kernels_chained_through_pdl_only():
    """TileKernels' cast is chained when the deployment calls it through ``pdl``
    (a programmatic edge from the torch kernel before it) and not when called
    directly; its switch is off again after the call."""
    import tile_kernels

    x = torch.randn((64, 5120), device=DEVICE)

    def cast(through_pdl: bool) -> None:
        y = (x * 2).bfloat16()
        if through_pdl:
            pdl.per_token_cast(y, "e4m3", 32, round_sf=True, use_packed_ue8m0=True)
        else:
            tile_kernels.quant.per_token_cast(
                y, "e4m3", 32, round_sf=True, use_packed_ue8m0=True
            )

    for through_pdl in (True, False):
        graph = _graph(functools.partial(cast, through_pdl), keep=True)
        edges = [
            (source, kind)
            for source, target, kind in _edges(graph)
            if target is not None and target.startswith("per_token_cast")
        ]
        assert len(edges) == 1 and edges[0][0] is not None, edges
        assert (edges[0][1] == PROGRAMMATIC) == through_pdl, (through_pdl, edges)
        assert not tile_kernels.get_pdl()


@pytest.mark.parametrize("launch", ["graph", "eager"])
@pytest.mark.parametrize("early", [True, False])
def test_reader_waits_for_its_producer(early, launch, n=48 * THREADS):
    """A reader starts beside its producer, in a CUDA graph and launched eagerly: one
    that loads before its wait reads the producer's buffer unwritten; one that waits
    first reads what the producer wrote, every time."""
    src = torch.arange(n, dtype=torch.float32, device=DEVICE)
    dst = torch.empty_like(src)
    out = torch.empty_like(src)
    copy, read = slow_copy(n), reader(n, early)

    def chain():
        copy(src, dst)
        read(dst, out)

    graph = _graph(chain) if launch == "graph" else None
    early_reads = 0
    for _ in range(REPLAYS):
        dst.fill_(POISON)
        out.zero_()
        if graph is None:
            chain()
        else:
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


def test_pair_edges():
    """A waiting reader's edge from its producer is programmatic; a torch kernel after
    the same producer, launched without PDL, has an ordinary edge."""
    n = 48 * THREADS
    src = torch.arange(n, dtype=torch.float32, device=DEVICE)
    dst, out = torch.empty_like(src), torch.empty_like(src)
    copy, read = slow_copy(n), reader(n, False)

    def chained():
        copy(src, dst)
        read(dst, out)

    def plain():
        copy(src, dst)
        torch.mul(dst, 2.0, out=out)

    edges = _edges(_graph(chained, keep=True))
    assert [(s, d) for s, d, _ in edges] == [
        ("pdl_slow_copy_kernel", "pdl_reader_kernel")
    ], edges
    assert edges[0][2] == PROGRAMMATIC, edges
    edges = _edges(_graph(plain, keep=True))
    assert len(edges) == 1 and edges[0][0] == "pdl_slow_copy_kernel", edges
    assert edges[0][1] is not None and edges[0][2] != PROGRAMMATIC, edges


@pytest.mark.usefixtures("default_vllm_config")
def test_layer_chain_edges_are_programmatic():
    """In the layer chain's captured graph, every chain kernel whose one predecessor is
    a kernel depends on it through a programmatic edge. Joins (several predecessors)
    are reported, not required."""
    chain = LayerChain(ROWS["c1"])
    edges = _edges(_graph(chain.run, keep=True))
    into = {}
    for source, target, kind in edges:
        into.setdefault(target, []).append((source, kind))
    plain, joins, programmatic = [], [], 0
    for source, target, kind in edges:
        if target is None or source is None or not _chained(target):
            continue
        programmatic += kind == PROGRAMMATIC
        if kind != PROGRAMMATIC:
            (joins if len(into[target]) > 1 else plain).append(f"{source} -> {target}")
    print(f"{len(edges)} edges, {programmatic} programmatic into chain kernels; "
          f"ordinary edges at joins: {joins}")  # fmt: skip
    assert programmatic > 0
    assert not plain, f"chain kernels with an ordinary edge from their kernel: {plain}"
