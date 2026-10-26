# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from types import SimpleNamespace

import pytest
import torch

from tests.v1.attention.utils import dense_kv_cache_views
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    KVCacheLayout,
    SlidingWindowSpec,
)
from vllm.v1.worker import utils as worker_utils
from vllm.v1.worker.utils import (
    AttentionGroup,
    KVBlockZeroer,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "spec",
    [
        SlidingWindowSpec(
            block_size=2,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.uint8,
            sliding_window=4,
        ),
        ChunkedLocalAttentionSpec(
            block_size=2,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.uint8,
            attention_chunk_size=4,
        ),
    ],
    ids=["sliding-window", "chunked-local"],
)
def test_attention_blocks_are_zeroed(spec):
    device = torch.device("cuda")
    storage = torch.ones((4, 1, 2, 2), dtype=torch.uint8, device=device)
    layer_name = "draft.self_attn"
    zeroer = KVBlockZeroer(
        device,
        attn_groups_iter=[AttentionGroup(None, [layer_name], spec, 0)],
        kernel_block_sizes=[2],
        static_forward_context={
            layer_name: SimpleNamespace(kv_cache=storage),
        },
        num_blocks=4,
    )

    zeroer.zero_block_ids([1])
    torch.accelerator.synchronize()

    expected = torch.ones_like(storage)
    expected[1] = 0
    assert torch.equal(storage, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_layers_in_one_group_may_use_different_kernel_pages_per_block():
    """Derive kernel pages per block from each layer's allocation."""
    device = torch.device("cuda")
    num_blocks = 4
    spec = SlidingWindowSpec(
        block_size=2,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.int32,
        sliding_window=2,
    )
    # Two kernel pages per logical block.
    wide = torch.ones((num_blocks * 2, 3), dtype=torch.int32, device=device)
    # One kernel page per logical block, carved out of a larger allocation so
    # an over-strided write lands in the guard region instead of faulting or
    # silently hitting another tensor.
    narrow_backing = torch.ones((num_blocks * 3, 5), dtype=torch.int32, device=device)
    narrow = narrow_backing[:num_blocks]

    zeroer = KVBlockZeroer(
        device,
        attn_groups_iter=[
            AttentionGroup(
                None,
                ["wide", "narrow"],
                spec,
                0,
            )
        ],
        kernel_block_sizes=[1],
        static_forward_context={
            "wide": SimpleNamespace(kv_cache=wide),
            "narrow": SimpleNamespace(kv_cache=narrow),
        },
        num_blocks=num_blocks,
    )

    zeroer.zero_block_ids([num_blocks - 1])
    torch.accelerator.synchronize()

    expected_wide = torch.ones_like(wide)
    expected_wide[2 * (num_blocks - 1) :] = 0
    assert torch.equal(wide, expected_wide)

    expected_narrow = torch.ones_like(narrow_backing)
    expected_narrow[num_blocks - 1] = 0
    assert torch.equal(narrow_backing, expected_narrow)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_block_ids_are_not_overwritten_while_copy_is_in_flight():
    device = torch.device("cuda")
    num_blocks = 4
    page_size_el = 4
    storage = torch.ones((num_blocks, page_size_el), dtype=torch.int32, device=device)

    # Build the minimal zeroer state directly so the test can focus on the
    # in-flight copy behavior without constructing model attention groups.
    zeroer = KVBlockZeroer.__new__(KVBlockZeroer)
    zeroer.device = device
    zeroer._meta = (
        torch.tensor([storage.data_ptr()], dtype=torch.uint64, device=device),
        torch.tensor([page_size_el], dtype=torch.int64, device=device),
        torch.tensor([page_size_el], dtype=torch.int64, device=device),
        page_size_el // page_size_el,  # max_chunks = 1
        page_size_el,  # blk_size
        1,  # n_segs
    )

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        # Keep the first nonblocking H2D copy pending while the host submits the
        # second call. Each call must stage from its own pinned source so the
        # first copy is not corrupted before it runs.
        torch.cuda._sleep(10_000_000)
        zeroer.zero_block_ids([1])
        zeroer.zero_block_ids([2])
    stream.synchronize()

    assert torch.all(storage[0] == 1)
    assert torch.all(storage[1] == 0)
    assert torch.all(storage[2] == 0)
    assert torch.all(storage[3] == 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_non_uniform_page_sizes():
    """Two segments with different page sizes (e.g. MLA + DSA indexer)."""
    device = torch.device("cuda")
    num_blocks = 4
    page_size_a = 10496  # int32 elements
    page_size_b = 2112

    storage_a = torch.ones((num_blocks, page_size_a), dtype=torch.int32, device=device)
    storage_b = torch.ones((num_blocks, page_size_b), dtype=torch.int32, device=device)

    zeroer = KVBlockZeroer.__new__(KVBlockZeroer)
    zeroer.device = device

    seg_page_sizes = [page_size_a, page_size_b]
    max_ps = max(seg_page_sizes)

    blk_size = min(1 << (max_ps - 1).bit_length(), 1024)

    zeroer._meta = (
        torch.tensor(
            [storage_a.data_ptr(), storage_b.data_ptr()],
            dtype=torch.uint64,
            device=device,
        ),
        torch.tensor(seg_page_sizes, dtype=torch.int64, device=device),
        torch.tensor(seg_page_sizes, dtype=torch.int64, device=device),
        (max_ps + blk_size - 1) // blk_size,
        blk_size,
        2,
    )

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        zeroer.zero_block_ids([1, 2])
    stream.synchronize()

    for storage in (storage_a, storage_b):
        assert torch.all(storage[0] == 1)
        assert torch.all(storage[1] == 0)
        assert torch.all(storage[2] == 0)
        assert torch.all(storage[3] == 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_packed_segment_zeros_only_its_last_block_page():
    """A packed KV segment steps by block stride but clears only its page."""
    device = torch.device("cuda")
    num_blocks = 4
    block_stride_el = 12
    page_size_el = 4
    page_offset_el = 3
    backing = torch.ones(
        (num_blocks, block_stride_el), dtype=torch.int32, device=device
    )

    zeroer = KVBlockZeroer.__new__(KVBlockZeroer)
    zeroer.device = device
    zeroer._meta = (
        torch.tensor(
            [backing.data_ptr() + page_offset_el * backing.element_size()],
            dtype=torch.uint64,
            device=device,
        ),
        torch.tensor([block_stride_el], dtype=torch.int64, device=device),
        torch.tensor([page_size_el], dtype=torch.int64, device=device),
        1,
        page_size_el,
        1,
    )

    zeroer.zero_block_ids([num_blocks - 1])
    torch.accelerator.synchronize()

    expected = torch.ones_like(backing)
    expected[-1, page_offset_el : page_offset_el + page_size_el] = 0
    assert torch.equal(backing, expected)


def test_large_dsv4_launch_geometry(monkeypatch):
    """Keep the failing DSV4 shape efficient and within launch limits."""
    device = torch.device("cpu")
    n_blocks, n_segs = 6870, 181
    layer_names = [f"layer.{i}" for i in range(n_segs)]
    page_sizes = [9344 if i % 2 == 0 else 292 for i in range(n_segs)]
    spec = SlidingWindowSpec(
        block_size=1,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.int32,
        sliding_window=1,
    )
    storages = {
        name: torch.ones((1, page_size), dtype=torch.int32)
        for name, page_size in zip(layer_names, page_sizes)
    }
    zeroer = KVBlockZeroer(
        device,
        attn_groups_iter=[
            AttentionGroup(None, [name], spec, group_id)
            for group_id, name in enumerate(layer_names)
        ],
        kernel_block_sizes=[1] * n_segs,
        static_forward_context={
            name: SimpleNamespace(kv_cache=storage)
            for name, storage in storages.items()
        },
        num_blocks=1,
    )

    assert zeroer._meta is not None
    _, _, seg_page_sizes, max_chunks, blk_size, n_segs = zeroer._meta
    assert seg_page_sizes.tolist() == page_sizes
    assert (max_chunks, blk_size, n_segs) == (10, 1024, 181)

    launches = []

    def fake_zero(seg_addrs, strides, pages, block_ids, max_page_size_el):
        assert seg_addrs.dtype == torch.int64  # the address table, viewed as int64
        launches.append((block_ids.numel(), seg_addrs.numel(), max_page_size_el))

    monkeypatch.setattr(worker_utils, "_zero_kv_blocks", fake_zero)
    monkeypatch.setattr(
        worker_utils,
        "async_tensor_h2d",
        lambda values, **kwargs: torch.tensor(values, dtype=torch.int64),
    )

    zeroer.zero_block_ids(list(range(n_blocks)))

    old_max_chunks = max(page_sizes) // 4
    assert math.prod((n_blocks, n_segs, old_max_chunks)) > 2**31 - 1
    # One launch of (blocks, segments, 1024-element chunks) blocks: ten chunks
    # cover the largest page.
    assert launches == [(n_blocks, n_segs, max_chunks * blk_size)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_warmup_compiles_for_all_block_counts():
    """After warmup, no launch should trigger a first-request JIT compile.

    The block count is carried by the launch grid, so changing it must reuse
    the warmup's compiled kernel (one TileLang kernel, no compile-time key).
    """
    device = torch.device("cuda")
    num_blocks = 64
    page_size_el = 4
    storage = torch.ones((num_blocks, page_size_el), dtype=torch.int32, device=device)

    zeroer = KVBlockZeroer.__new__(KVBlockZeroer)
    zeroer.device = device
    zeroer._meta = (
        torch.tensor([storage.data_ptr()], dtype=torch.uint64, device=device),
        torch.tensor([page_size_el], dtype=torch.int64, device=device),
        torch.tensor([page_size_el], dtype=torch.int64, device=device),
        1,  # max_chunks
        page_size_el,  # blk_size
        1,  # n_segs
    )

    from vllm.v1.worker.gpu import tilelang_inputs

    def compiled_variants() -> set:
        factory = id(tilelang_inputs.zero_kv_blocks_kernel)
        return {key for key in tilelang_inputs._KERNELS if key[0] == factory}

    zeroer.warmup(num_blocks)
    torch.accelerator.synchronize()
    warmed = compiled_variants()
    assert warmed

    for n_blocks in (1, 2, 3, 16, 32):
        zeroer.zero_block_ids(list(range(n_blocks)))
    torch.accelerator.synchronize()

    assert compiled_variants() == warmed


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_warmup_respects_available_block_count():
    """An empty KV cache must not be warmed with out-of-range block IDs."""
    device = torch.device("cuda")
    page_size_el = 4
    storage = torch.ones((1, page_size_el), dtype=torch.int32, device=device)

    zeroer = KVBlockZeroer.__new__(KVBlockZeroer)
    zeroer.device = device
    zeroer._meta = (
        torch.tensor([storage.data_ptr()], dtype=torch.uint64, device=device),
        torch.tensor([page_size_el], dtype=torch.int64, device=device),
        torch.tensor([page_size_el], dtype=torch.int64, device=device),
        1,
        page_size_el,
        1,
    )

    zeroer.warmup(0)
    torch.accelerator.synchronize()

    assert torch.all(storage == 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("layout", list(KVCacheLayout))
def test_zeroes_exactly_one_block_per_layer(layout: KVCacheLayout):
    """The zeroer must zero every byte of the target block in every layer and nothing
    outside it — per head-group region under LHBNC, and never past the target block's
    tile under block-major layouts (no out-of-bounds writes, no clobbering)."""
    device = torch.device("cuda")
    num_blocks, num_layers = 4, 2
    spec = FullAttentionSpec(
        block_size=4, num_kv_heads=2, head_size=8, dtype=torch.float32
    )
    raw = torch.empty(
        num_blocks * num_layers * spec.page_size_bytes,
        dtype=torch.int8,
        device=device,
    ).fill_(1)
    views = dense_kv_cache_views(raw, spec, num_blocks, num_layers, layout)
    groups = [
        AttentionGroup(
            backend=None,
            layer_names=[f"layer.{i}" for i in range(num_layers)],
            kv_cache_spec=spec,
            kv_cache_group_id=0,
        )
    ]
    ctx = {f"layer.{i}": SimpleNamespace(kv_cache=views[i]) for i in range(num_layers)}
    zeroer = KVBlockZeroer(
        device,
        attn_groups_iter=iter(groups),
        kernel_block_sizes=[spec.block_size],
        static_forward_context=ctx,
        num_blocks=num_blocks,
    )
    zeroer.zero_block_ids([2])
    torch.accelerator.synchronize()

    for view in views:
        assert (view[2] == 0).all(), layout
        for b in (0, 1, 3):
            assert (view[b].view(torch.int8) == 1).all(), layout
    zero_bytes = int((raw == 0).sum().item())
    assert zero_bytes == num_layers * spec.page_size_bytes, layout


# --- VLLM_KV_CACHE_POISON -----------------------------------------------------


def _cpu_zeroer(monkeypatch, poison: bool, num_blocks: int = 4) -> KVBlockZeroer:
    """A zeroer over two CPU layers, its launches recorded instead of run."""
    if poison:
        monkeypatch.setenv("VLLM_KV_CACHE_POISON", "1")
    else:
        monkeypatch.delenv("VLLM_KV_CACHE_POISON", raising=False)
    spec = SlidingWindowSpec(
        block_size=1, num_kv_heads=1, head_size=1, dtype=torch.int32, sliding_window=1
    )
    storages = {
        name: torch.ones((num_blocks, 8), dtype=torch.int32) for name in ("a", "b")
    }
    zeroer = KVBlockZeroer(
        torch.device("cpu"),
        attn_groups_iter=[AttentionGroup(None, list(storages), spec, 0)],
        kernel_block_sizes=[1],
        static_forward_context={
            name: SimpleNamespace(kv_cache=storage)
            for name, storage in storages.items()
        },
        num_blocks=num_blocks,
    )
    monkeypatch.setattr(
        worker_utils,
        "async_tensor_h2d",
        lambda values, **kwargs: torch.tensor(values, dtype=torch.int64),
    )
    zeroer.launches = []
    monkeypatch.setattr(
        worker_utils,
        "_zero_kv_blocks",
        lambda addrs, strides, pages, ids, max_el: zeroer.launches.append(
            ("zero", ids.tolist(), addrs.numel(), max_el)
        ),
    )
    monkeypatch.setattr(
        worker_utils,
        "_fill_kv_blocks",
        lambda addrs, strides, pages, ids, max_el, word: zeroer.launches.append(
            ("fill", ids.tolist(), addrs.numel(), max_el, word)
        ),
    )
    return zeroer


def test_poison_fills_new_blocks_where_zeroing_would_zero(monkeypatch):
    """The poison check fills the segments and pages zeroing clears, with
    ``KV_POISON_WORD``; off, the zeroing kernel runs as before."""
    zeroer = _cpu_zeroer(monkeypatch, poison=False)
    zeroer.zero_block_ids([1, 3])
    zeroer.zero_block_ids([])
    assert zeroer.launches == [("zero", [1, 3], 2, 8)]

    zeroer = _cpu_zeroer(monkeypatch, poison=True)
    zeroer.zero_block_ids([1, 3])
    zeroer.zero_block_ids([])
    assert zeroer.launches == [("fill", [1, 3], 2, 8, worker_utils.KV_POISON_WORD)]
    assert worker_utils.KV_POISON_WORD & 0xFFFFFFFF == 0xFFFFFFFF


def test_poison_warmup_compiles_the_fill_on_a_zero_null_block(monkeypatch):
    """Warmup compiles the kernel the steps launch; under the poison check it fills
    the null block (no request's) with zeros, as zeroing leaves it."""
    zeroer = _cpu_zeroer(monkeypatch, poison=False)
    zeroer.warmup(4)
    assert zeroer.launches == [("zero", [0], 2, 8)]

    zeroer = _cpu_zeroer(monkeypatch, poison=True)
    zeroer.warmup(4)
    zeroer.warmup(0)
    assert zeroer.launches == [("fill", [0], 2, 8, 0)]


@pytest.mark.parametrize("poison", [False, True])
def test_poison_declares_the_kernel_it_launches(monkeypatch, poison):
    """The frozen kernel cache accepts only declared calls: the poison check
    declares the fill in place of zeroing, and serving declares no fill."""
    if poison:
        monkeypatch.setenv("VLLM_KV_CACHE_POISON", "1")
    else:
        monkeypatch.delenv("VLLM_KV_CACHE_POISON", raising=False)
    declared = []
    fake = SimpleNamespace(
        declare=declared.append,
        zero_kv_blocks_calls=object(),
        fill_kv_blocks_calls=object(),
    )
    import sys

    monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu.tilelang_inputs", fake)
    import vllm.v1.worker.gpu as gpu_package

    monkeypatch.setattr(gpu_package, "tilelang_inputs", fake, raising=False)
    worker_utils.declare_zero_kv_blocks_kernel()
    expected = fake.fill_kv_blocks_calls if poison else fake.zero_kv_blocks_calls
    assert declared == [expected]


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.bfloat16,
        torch.float16,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.float8_e8m0fnu,
    ],
)
def test_poison_bytes_are_nan_in_every_cache_float_format(dtype):
    """All-ones bytes are NaN in each float format a KV page may hold. Every byte
    is the same, so a field at any offset of a record reads this pattern: the
    record's layout cannot turn the poison into a number."""
    word = torch.tensor([worker_utils.KV_POISON_WORD] * 4, dtype=torch.int32)
    assert torch.isnan(word.view(dtype).float()).all()


def test_poison_nibbles_are_the_largest_e2m1_magnitude():
    """E2M1 (FP4) has no NaN or Inf: the all-ones nibble is -6.0, its largest
    magnitude (the poison check relies on the UE8M0 scale beside it)."""
    # E2M1: sign, two exponent bits (bias 1), one mantissa bit.
    sign, exponent, mantissa = 1, 0b11, 1
    assert (-1) ** sign * 2 ** (exponent - 1) * (1 + mantissa / 2) == -6.0
