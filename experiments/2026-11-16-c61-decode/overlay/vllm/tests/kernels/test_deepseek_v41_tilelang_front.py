# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1's attention front in one launch (``tilelang/front.py``) against the
five kernels it replaces, bit for bit: TileKernels' RMSNorm of the q and kv latents,
``per_token_cast`` of the normalized q latent, the kv latent's RoPE and its SWA cache
record."""

import pytest
import torch

from vllm.platforms import current_platform

if not (
    current_platform.is_cuda() and current_platform.is_device_capability_family(120)
):
    pytest.skip("TileLang V4.1 kernels target SM120/SM121", allow_module_level=True)

pytest.importorskip("tilelang")
pytest.importorskip("tile_kernels")

from vllm.models.deepseek_v4_1.tilelang.cache_writer import (  # noqa: E402
    RECORD_BYTES,
    write_cache,
    write_swa_records,
)
from vllm.models.deepseek_v4_1.tilelang.front import (  # noqa: E402
    KV_LATENT,
    Q_LATENT,
    ROPE_DIM,
    attention_front,
    context_kv_front,
    context_kv_front_3layer,
    sp_attention_front,
)
from vllm.models.deepseek_v4_1.tilelang.gemm import scale_words  # noqa: E402
from vllm.models.deepseek_v4_1.tilelang.pdl import (  # noqa: E402
    norm_forward,
    per_token_cast,
)
from vllm.models.deepseek_v4_1.tilelang.rope import rope_  # noqa: E402

DEVICE = "cuda"
EPS = 1e-6
POSITIONS = 4096
PAGES, PAGE_SIZE = 6, 128


def _table():
    inv = 1.0 / (
        10000 ** (torch.arange(0, ROPE_DIM, 2, device=DEVICE).double() / ROPE_DIM)
    )
    angles = torch.arange(POSITIONS, device=DEVICE).double()[:, None] * inv
    return torch.cat((angles.cos(), angles.sin()), dim=-1).float()


def _assert_bytes_equal(actual, expected, what):
    actual, expected = actual.view(torch.uint8), expected.view(torch.uint8)
    assert actual.shape == expected.shape, what
    differ = actual != expected
    if differ.any():
        where = differ.nonzero()[:4].tolist()
        raise AssertionError(
            f"{what}: {int(differ.sum())} of {differ.numel()} bytes differ; first at "
            f"{where}"
        )


@pytest.mark.parametrize("slot_dtype", [torch.int64, torch.int32])
@pytest.mark.parametrize("rows", [1, 2, 3, 6, 8, 9, 16, 24, 48, 64, 65, 512])
def test_front_gives_the_five_kernels_bits(rows, slot_dtype):
    """Every output byte: the BF16 q latent, its E4M3 values and UE8M0 scales, the
    rotated kv latent and the cache pages (records at the slots, skipped slots and
    the rest of the pages untouched). The norm weights put y in every scale regime
    and the latents carry rows of zeros, of one large value and of tiny values."""
    gen = torch.Generator(device=DEVICE).manual_seed(rows)
    qa_kv = (
        torch.randn((rows, Q_LATENT + KV_LATENT), generator=gen, device=DEVICE) * 3
    ).bfloat16()
    qa_kv[0, :64] = 0.0
    if rows > 2:
        qa_kv[1, :] = 0.0
        qa_kv[2, Q_LATENT + 100] = 1e4
    if rows > 3:
        qa_kv[3, :] *= 1e-30
    q_weight = (
        torch.randn(Q_LATENT, generator=gen, device=DEVICE) * 0.3 + 1
    ).bfloat16()
    kv_weight = (
        torch.randn(KV_LATENT, generator=gen, device=DEVICE) * 0.3 + 1
    ).bfloat16()
    q_weight[:32] = 0.0
    q_weight[32:64] *= 4096.0
    positions = torch.randint(0, POSITIONS, (rows,), generator=gen, device=DEVICE)
    slots = torch.randperm(PAGES * PAGE_SIZE, generator=gen, device=DEVICE)[:rows]
    slots[rows // 2] = -1  # a padding row writes no record
    slots = slots.to(slot_dtype)
    table = _table()
    page_bytes = PAGE_SIZE * RECORD_BYTES["swa"]
    pages = torch.empty((PAGES, page_bytes), dtype=torch.uint8, device=DEVICE)
    pages.random_(generator=gen)

    # The five kernels, as the attention forward launches them.
    q_ref = torch.empty((rows, Q_LATENT), dtype=torch.bfloat16, device=DEVICE)
    norm_forward(qa_kv[:, :Q_LATENT], q_weight, EPS, out=q_ref)
    kv_ref = torch.empty((rows, KV_LATENT), dtype=torch.bfloat16, device=DEVICE)
    norm_forward(qa_kv[:, Q_LATENT:], kv_weight, EPS, out=kv_ref)
    values_ref = torch.empty((rows, Q_LATENT), dtype=torch.float8_e4m3fn, device=DEVICE)
    scales_ref = torch.empty(
        (rows, 4 * scale_words(Q_LATENT)), dtype=torch.uint8, device=DEVICE
    )
    per_token_cast(
        q_ref, "e4m3", 32, round_sf=True, use_packed_ue8m0=True,
        out=(values_ref, scales_ref),
    )  # fmt: skip
    rope_(kv_ref, positions, table, 1, False)
    pages_ref = pages.clone()
    write_cache(kv_ref, pages_ref, slots, page_size=PAGE_SIZE, cache_kind="swa")

    qn = torch.full_like(q_ref, float("nan"))
    kv = torch.full_like(kv_ref, float("nan"))
    values = torch.empty_like(values_ref)
    values.view(torch.uint8).fill_(0x7F)
    scales = torch.full_like(scales_ref, 0xFF)
    attention_front(
        qa_kv, q_weight, kv_weight, EPS, positions, table, pages, slots, PAGE_SIZE,
        qn, values, scales, kv,
    )  # fmt: skip
    _assert_bytes_equal(qn, q_ref, "q latent")
    _assert_bytes_equal(values, values_ref, "q E4M3")
    _assert_bytes_equal(scales, scales_ref, "q scales")
    _assert_bytes_equal(kv, kv_ref, "rotated kv latent")
    _assert_bytes_equal(pages, pages_ref, "cache pages")


@pytest.mark.parametrize("slot_dtype", [torch.int64, torch.int32])
@pytest.mark.parametrize("rows", [1, 6, 16, 24, 64, 96, 256])
def test_context_kv_front_gives_the_three_kernels_bits(rows, slot_dtype):
    """``context_kv_front`` matches ``norm_forward`` + ``rope_`` + ``write_cache``
    on ``[rows, 512]`` BF16 context KV rows bit for bit."""
    gen = torch.Generator(device=DEVICE).manual_seed(rows + 100)
    kv_raw = (
        torch.randn((rows, KV_LATENT), generator=gen, device=DEVICE) * 3
    ).bfloat16()
    kv_raw[0, :64] = 0.0
    kv_weight = (
        torch.randn(KV_LATENT, generator=gen, device=DEVICE) * 0.3 + 1
    ).bfloat16()
    positions = torch.randint(0, POSITIONS, (rows,), generator=gen, device=DEVICE)
    slots = torch.randperm(PAGES * PAGE_SIZE, generator=gen, device=DEVICE)[:rows]
    slots[rows // 2] = -1
    slots = slots.to(slot_dtype)
    table = _table()
    page_bytes = PAGE_SIZE * RECORD_BYTES["swa"]
    pages = torch.empty((PAGES, page_bytes), dtype=torch.uint8, device=DEVICE)
    pages.random_(generator=gen)
    pages_ref = pages.clone()

    kv_ref = torch.empty((rows, KV_LATENT), dtype=torch.bfloat16, device=DEVICE)
    norm_forward(kv_raw, kv_weight, EPS, out=kv_ref)
    rope_(kv_ref, positions, table, 1, False)
    write_cache(kv_ref, pages_ref, slots, page_size=PAGE_SIZE, cache_kind="swa")

    context_kv_front(
        kv_raw, kv_weight, EPS, positions, table, pages, slots, PAGE_SIZE
    )
    _assert_bytes_equal(pages, pages_ref, "context kv cache pages")


@pytest.mark.parametrize("slot_dtype", [torch.int64, torch.int32])
@pytest.mark.parametrize("rows", [1, 2, 3, 6, 8, 9, 16, 24, 48, 64, 65, 512])
def test_sp_front_and_write_swa_records_give_the_five_kernels_bits(rows, slot_dtype):
    """``sp_attention_front`` + ``write_swa_records`` match the five-kernel reference
    bit for bit on ``q_values``, ``q_scales``, and ``pages`` (including strided
    ``qa_kv`` views, skipped negative/out-of-range slots, and special FP values)."""
    gen = torch.Generator(device=DEVICE).manual_seed(rows + 200)
    qa_kv_full = (
        torch.randn((rows, Q_LATENT + KV_LATENT + 32), generator=gen, device=DEVICE)
        * 3
    ).bfloat16()
    qa_kv = qa_kv_full[:, : Q_LATENT + KV_LATENT]
    qa_kv[0, :64] = -0.0
    if rows > 2:
        qa_kv[1, :] = 0.0
        qa_kv[2, Q_LATENT + 100] = 1e4
    if rows > 3:
        qa_kv[3, :] *= 1e-30
    q_weight = (
        torch.randn(Q_LATENT, generator=gen, device=DEVICE) * 0.3 + 1
    ).bfloat16()
    kv_weight = (
        torch.randn(KV_LATENT, generator=gen, device=DEVICE) * 0.3 + 1
    ).bfloat16()
    q_weight[:32] = 0.0
    q_weight[32:64] *= 4096.0
    positions = torch.randint(0, POSITIONS, (rows,), generator=gen, device=DEVICE)
    slots = torch.randperm(PAGES * PAGE_SIZE, generator=gen, device=DEVICE)[:rows]
    slots[rows // 2] = -1
    if rows > 4:
        slots[rows - 1] = PAGES * PAGE_SIZE + 5
    slots = slots.to(slot_dtype)
    table = _table()
    page_bytes = PAGE_SIZE * RECORD_BYTES["swa"]
    pages = torch.empty((PAGES, page_bytes), dtype=torch.uint8, device=DEVICE)
    pages.random_(generator=gen)

    q_ref = torch.empty((rows, Q_LATENT), dtype=torch.bfloat16, device=DEVICE)
    norm_forward(qa_kv[:, :Q_LATENT].contiguous(), q_weight, EPS, out=q_ref)
    kv_ref = torch.empty((rows, KV_LATENT), dtype=torch.bfloat16, device=DEVICE)
    norm_forward(qa_kv[:, Q_LATENT:].contiguous(), kv_weight, EPS, out=kv_ref)
    values_ref = torch.empty((rows, Q_LATENT), dtype=torch.float8_e4m3fn, device=DEVICE)
    scales_ref = torch.empty(
        (rows, 4 * scale_words(Q_LATENT)), dtype=torch.uint8, device=DEVICE
    )
    per_token_cast(
        q_ref, "e4m3", 32, round_sf=True, use_packed_ue8m0=True,
        out=(values_ref, scales_ref),
    )  # fmt: skip
    rope_(kv_ref, positions, table, 1, False)
    pages_ref = pages.clone()
    write_cache(kv_ref, pages_ref, slots, page_size=PAGE_SIZE, cache_kind="swa")

    values = torch.empty_like(values_ref)
    values.view(torch.uint8).fill_(0x7F)
    scales = torch.full_like(scales_ref, 0xFF)
    records = torch.full(
        (rows, RECORD_BYTES["swa"]), 0xAA, dtype=torch.uint8, device=DEVICE
    )
    sp_attention_front(
        qa_kv, q_weight, kv_weight, EPS, positions, table, values, scales, records
    )
    write_swa_records(records, pages, slots, page_size=PAGE_SIZE)
    _assert_bytes_equal(values, values_ref, "sp q E4M3")
    _assert_bytes_equal(scales, scales_ref, "sp q scales")
    _assert_bytes_equal(pages, pages_ref, "sp cache pages")


def test_sp_front_and_write_swa_records_empty_and_validation():
    """Zero-row launches are no-ops and malformed record/page inputs raise ValueError."""
    table = _table()
    page_bytes = PAGE_SIZE * RECORD_BYTES["swa"]
    pages = torch.zeros((PAGES, page_bytes), dtype=torch.uint8, device=DEVICE)
    q_weight = torch.ones(Q_LATENT, dtype=torch.bfloat16, device=DEVICE)
    kv_weight = torch.ones(KV_LATENT, dtype=torch.bfloat16, device=DEVICE)
    sp_attention_front(
        torch.empty((0, Q_LATENT + KV_LATENT), dtype=torch.bfloat16, device=DEVICE),
        q_weight,
        kv_weight,
        EPS,
        torch.empty((0,), dtype=torch.int64, device=DEVICE),
        table,
        torch.empty((0, Q_LATENT), dtype=torch.float8_e4m3fn, device=DEVICE),
        torch.empty((0, 4 * scale_words(Q_LATENT)), dtype=torch.uint8, device=DEVICE),
        torch.empty((0, RECORD_BYTES["swa"]), dtype=torch.uint8, device=DEVICE),
    )
    write_swa_records(
        torch.empty((0, RECORD_BYTES["swa"]), dtype=torch.uint8, device=DEVICE),
        pages,
        torch.empty((0,), dtype=torch.int64, device=DEVICE),
        page_size=PAGE_SIZE,
    )
    with pytest.raises(ValueError, match="SWA cache records"):
        write_swa_records(
            torch.zeros((2, 16), dtype=torch.uint8, device=DEVICE),
            pages,
            torch.zeros((2,), dtype=torch.int64, device=DEVICE),
            page_size=PAGE_SIZE,
        )


@pytest.mark.parametrize("slot_dtype", [torch.int64, torch.int32])
@pytest.mark.parametrize("rows", [1, 6, 16, 48])
def test_context_kv_front_accepts_grouped_strided_views(rows, slot_dtype):
    """``context_kv_front`` matches contiguous inputs on ``out.split(512, dim=1)``
    views (stride 1536) emitted by ``TileLangGroupedBlock32Rows``."""
    gen = torch.Generator(device=DEVICE).manual_seed(rows + 777)
    grouped = (
        torch.randn((rows, 3 * KV_LATENT), generator=gen, device=DEVICE) * 3
    ).bfloat16()
    slices = grouped.split(KV_LATENT, dim=1)
    kv_weight = (
        torch.randn(KV_LATENT, generator=gen, device=DEVICE) * 0.3 + 1
    ).bfloat16()
    positions = torch.randint(0, POSITIONS, (rows,), generator=gen, device=DEVICE)
    slots = (
        torch.randperm(PAGES * PAGE_SIZE, generator=gen, device=DEVICE)[:rows]
    ).to(slot_dtype)
    table = _table()
    page_bytes = PAGE_SIZE * RECORD_BYTES["swa"]
    pages_init = torch.empty((PAGES, page_bytes), dtype=torch.uint8, device=DEVICE)
    pages_init.random_(generator=gen)

    for kv_view in slices:
        pages_strided = pages_init.clone()
        pages_contig = pages_init.clone()
        context_kv_front(
            kv_view, kv_weight, EPS, positions, table, pages_strided, slots, PAGE_SIZE
        )
        context_kv_front(
            kv_view.contiguous(),
            kv_weight,
            EPS,
            positions,
            table,
            pages_contig,
            slots,
            PAGE_SIZE,
        )
        _assert_bytes_equal(pages_strided, pages_contig, "strided context kv pages")


@pytest.mark.parametrize("slot_dtype", [torch.int64, torch.int32])
@pytest.mark.parametrize("rows", [1, 6, 16, 48])
def test_context_kv_front_3layer_matches_per_layer_bits(rows, slot_dtype):
    """``context_kv_front_3layer`` matches three serial ``context_kv_front``
    launches bit for bit on grouped strided views and separate layer caches."""
    gen = torch.Generator(device=DEVICE).manual_seed(rows + 999)
    grouped = (
        torch.randn((rows, 3 * KV_LATENT), generator=gen, device=DEVICE) * 3
    ).bfloat16()
    slices = grouped.split(KV_LATENT, dim=1)
    weights = [
        (torch.randn(KV_LATENT, generator=gen, device=DEVICE) * 0.3 + 1).bfloat16()
        for _ in range(3)
    ]
    positions = torch.randint(0, POSITIONS, (rows,), generator=gen, device=DEVICE)
    slots = [
        torch.randperm(PAGES * PAGE_SIZE, generator=gen, device=DEVICE)[:rows].to(
            slot_dtype
        )
        for _ in range(3)
    ]
    slots[1][rows // 2] = -1
    table = _table()
    page_bytes = PAGE_SIZE * RECORD_BYTES["swa"]
    caches_ref = [
        torch.empty((PAGES, page_bytes), dtype=torch.uint8, device=DEVICE)
        for _ in range(3)
    ]
    for c in caches_ref:
        c.random_(generator=gen)
    caches_fused = [c.clone() for c in caches_ref]

    for i in range(3):
        context_kv_front(
            slices[i],
            weights[i],
            EPS,
            positions,
            table,
            caches_ref[i],
            slots[i],
            PAGE_SIZE,
        )
    context_kv_front_3layer(
        slices, weights, EPS, positions, table, caches_fused, slots, PAGE_SIZE
    )
    for ref, got in zip(caches_ref, caches_fused):
        _assert_bytes_equal(got, ref, "fused 3-layer context kv pages")
