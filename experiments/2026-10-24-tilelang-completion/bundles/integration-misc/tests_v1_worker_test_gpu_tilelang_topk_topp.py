# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The TileLang top-k / top-p filter against the Triton kernels it replaces, bit for
bit: ``apply_top_k_top_p`` against ``apply_top_k_top_p_triton`` on copies of the same
logits (the whole padded rows compared, so nothing outside the vocabulary moves), for
top-k, top-p and both, at the batch sizes that take each of Triton's paths (the split
pipeline at every slice count, the monolithic kernel beside it and above 64 rows), on
the DeepSeek V4.1 vocabulary and odd ones, rows aligned or not, with per-row k and p
(k = vocab and p = 1 among them) and logits with ties at the thresholds, infinities
and NaN."""

import pytest
import torch

pytest.importorskip("triton")
pytest.importorskip("tilelang")

from vllm.utils.platform_utils import num_compute_units  # noqa: E402
from vllm.v1.sample.ops.topk_topp_triton import (  # noqa: E402
    _topp_split_count,
    apply_top_k_top_p_triton,
)
from vllm.v1.worker.gpu.sample import tilelang_topk_topp as ttk  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)

DEVICE = torch.device("cuda")
VOCAB = 129280  # DeepSeek V4.1
MODES = ("k", "p", "kp")
# Rows: one request; one request's six with five drafts; the split pipeline's
# smallest slice count on GB10 (48 SMs); above the split pipeline's 64.
ROWS = (1, 6, 16, 96)
# (vocabulary, row layout): contiguous rows of an odd vocabulary are Triton's
# unaligned specialization (one element a thread's run), padded rows the aligned one
# with an odd vocabulary (a run of four on the logits, of one or two on the buffer).
LAYOUTS = (
    (VOCAB, "contiguous"),
    (VOCAB, "padded"),
    (2049, "contiguous"),
    (2049, "padded"),
    (1002, "contiguous"),
)


def _bits(x: torch.Tensor) -> torch.Tensor:
    return x.view(torch.int32) if x.dtype == torch.float32 else x


def _assert_same(ours: torch.Tensor, theirs: torch.Tensor, what: str) -> None:
    a, b = _bits(ours), _bits(theirs)
    if torch.equal(a, b):
        return
    differ = (a != b).nonzero()
    first = differ[:5].tolist()
    raise AssertionError(
        f"{what}: {differ.shape[0]} of {a.numel()} elements differ, first at "
        f"{first}: ours {[ours[tuple(i)].item() for i in first]}, "
        f"Triton's {[theirs[tuple(i)].item() for i in first]}"
    )


def _width(vocab: int, layout: str) -> int:
    """A row's storage: the vocabulary, or padded to a multiple of 16 elements."""
    return vocab if layout == "contiguous" else (vocab + 16 + 15) // 16 * 16


def _logits(rows: int, vocab: int, layout: str, seed: int) -> torch.Tensor:
    """Rows of logits, a pattern each by the row's index: Gaussian; rounded (few
    distinct values, ties everywhere); a peak of 40 equal values over a Gaussian; a
    few large values over a flat tail; -inf, +inf and NaN sprinkled; a grammar mask
    (all -inf but a few); all -inf; NaN everywhere but a few finite values."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    storage = torch.randn(rows, _width(vocab, layout), generator=g, device=DEVICE) * 3
    x = storage[:, :vocab]
    for r in range(rows):
        kind = r % 8
        if kind == 1:
            x[r] = torch.round(x[r] / 3)
        elif kind == 2:
            top = torch.randperm(vocab, generator=g, device=DEVICE)[:40]
            x[r, top] = x[r].max() + 1
        elif kind == 3:
            x[r] = torch.round(x[r] * 4) / 64
            top = torch.randperm(vocab, generator=g, device=DEVICE)[: min(6, vocab)]
            x[r, top] = 12.0 + torch.arange(top.numel(), device=DEVICE) % 3
        elif kind == 4:
            pick = torch.randperm(vocab, generator=g, device=DEVICE)
            x[r, pick[: vocab // 97]] = float("-inf")
            x[r, pick[vocab // 97 : vocab // 97 + 2]] = float("inf")
            x[r, pick[vocab // 97 + 2 : vocab // 97 + 2 + vocab // 307]] = float("nan")
        elif kind == 5:
            keep = torch.randperm(vocab, generator=g, device=DEVICE)[: 3 + r % 50]
            kept = x[r, keep].clone()
            x[r] = float("-inf")
            x[r, keep] = kept
        elif kind == 6:
            x[r] = float("-inf")
        elif kind == 7:
            keep = torch.randperm(vocab, generator=g, device=DEVICE)[:9]
            kept = x[r, keep].clone()
            x[r] = float("nan")
            x[r, keep] = kept
    return x


def _controls(rows: int, vocab: int, mode: str, seed: int):
    """Per-row k (int32) and p (FP32), or None: varied, with k = vocab (no top-k)
    and p = 1 (no top-p) among them, k of 1, k above any outlier count, p tiny and
    p near 1."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    k = torch.randint(1, 300, (rows,), generator=g, device=DEVICE).to(torch.int32)
    k[1::5] = vocab
    k[2::7] = 1
    k[3::9] = max(vocab // 3, 1)
    p = torch.rand(rows, generator=g, device=DEVICE) * 0.9 + 0.05
    p[1::4] = 1.0
    p[2::6] = 0.01
    p[3::8] = 0.999
    return (k if "k" in mode else None), (p if "p" in mode else None)


def _copy(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """A copy of ``x`` with its storage's padding: ``(storage, view)``."""
    storage = torch.full(
        (x.shape[0], max(x.stride(0), x.shape[1])), -7.0, device=DEVICE
    )
    view = storage[:, : x.shape[1]]
    view.copy_(x)
    return storage, view


def _check(logits: torch.Tensor, k, p, what: str) -> None:
    ours_storage, ours = _copy(logits)
    theirs_storage, theirs = _copy(logits)
    ours_out = ttk.apply_top_k_top_p(ours, k, p)
    theirs_out = apply_top_k_top_p_triton(theirs, k, p)
    assert ours_out.data_ptr() == ours.data_ptr()
    assert theirs_out.data_ptr() == theirs.data_ptr()
    _assert_same(ours_storage, theirs_storage, what)


@pytest.mark.parametrize("vocab,layout", LAYOUTS)
@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("mode", MODES)
def test_filter(mode, rows, vocab, layout):
    logits = _logits(rows, vocab, layout, rows * 7 + vocab)
    k, p = _controls(rows, vocab, mode, rows + vocab)
    _check(logits, k, p, f"{mode} rows={rows} vocab={vocab} {layout}")


def _split_rows() -> list[int]:
    """The fewest rows at each of this device's slice counts of the split pipeline."""
    num_sm = num_compute_units(DEVICE.index or 0)
    first: dict[int, int] = {}
    for rows in range(1, ttk.SPLIT_MAX_BATCH + 1):
        first.setdefault(_topp_split_count(rows, num_sm), rows)
    return sorted(first.values())


@pytest.mark.parametrize("mode", ["p", "kp"])
def test_split_slice_counts(mode):
    """Every slice count the split pipeline takes, with and without a top-k."""
    for rows in _split_rows():
        logits = _logits(rows, VOCAB, "contiguous", rows)
        k, p = _controls(rows, VOCAB, mode, rows + 1)
        _check(logits, k, p, f"{mode} rows={rows}")


def _tied_rows(vocab: int) -> torch.Tensor:
    """Rows whose top-k and top-p boundaries fall inside groups of equal values:
    levels of 7, 23, 91 and 400 copies over a low Gaussian tail."""
    g = torch.Generator(device=DEVICE).manual_seed(5)
    rows = 6
    x = torch.randn(rows, vocab, generator=g, device=DEVICE) * 0.25 - 6.0
    for r in range(rows):
        pick = torch.randperm(vocab, generator=g, device=DEVICE)
        start = 0
        for level, copies in enumerate((7, 23, 91, 400)):
            x[r, pick[start : start + copies]] = 4.0 - level * (0.5 + 0.25 * r)
            start += copies
    return x


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("rows", [6, 96])
def test_ties_at_the_thresholds(mode, rows):
    """k and p that cut inside a group of equal values: Triton keeps the first of
    the boundary value's copies by column (counting tile by tile)."""
    x = _tied_rows(VOCAB).repeat((rows + 5) // 6, 1)[:rows].contiguous()
    k = torch.tensor(
        [3, 7, 19, 30, 50, 114, 200, 515, VOCAB], dtype=torch.int32, device=DEVICE
    ).repeat(rows)[:rows]
    p = torch.tensor(
        [0.2, 0.33, 0.5, 0.61, 0.75, 0.9, 0.97, 1.0], device=DEVICE
    ).repeat(rows)[:rows]
    _check(
        x,
        k if "k" in mode else None,
        p if "p" in mode else None,
        f"ties {mode} rows={rows}",
    )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("rows", [1, 6, 96])
def test_nothing_to_filter(mode, rows):
    """k = vocab and p = 1 on every row: the logits stay as they are."""
    logits = _logits(rows, VOCAB, "padded", 3)
    k = torch.full((rows,), VOCAB, dtype=torch.int32, device=DEVICE)
    p = torch.ones(rows, device=DEVICE)
    k, p = (k if "k" in mode else None), (p if "p" in mode else None)
    _check(logits, k, p, f"none {mode} rows={rows}")
    _, view = _copy(logits)
    ttk.apply_top_k_top_p(view, k, p)
    _assert_same(view, logits, "unfiltered logits")


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("rows", [6, 96])
def test_nonfinite_rows(mode, rows):
    """Rows of -inf, of NaN, of +inf, with one finite value, and NaN, +inf and -inf
    mixed into Gaussian rows."""
    g = torch.Generator(device=DEVICE).manual_seed(11)
    x = torch.randn(rows, VOCAB, generator=g, device=DEVICE) * 2
    kinds = 9
    for r in range(rows):
        kind = r % kinds
        if kind == 0:
            x[r] = float("-inf")
        elif kind == 1:
            x[r] = float("nan")
        elif kind == 2:
            x[r] = float("inf")
        elif kind == 3:
            x[r] = float("-inf")
            x[r, r * 37 % VOCAB] = 1.5
        elif kind == 4:
            x[r, ::5] = float("nan")
        elif kind == 5:
            x[r, 1000:1003] = float("inf")
        elif kind == 6:
            x[r, :8192] = float("-inf")
        elif kind == 7:
            x[r, :8192] = float("nan")
            x[r, 0] = float("inf")
        else:
            x[r, -5:] = float("inf")
            x[r, :5] = float("nan")
    k = torch.tensor([1, 5, 50, 2000, VOCAB], dtype=torch.int32, device=DEVICE)
    k = k.repeat(rows)[:rows]
    p = torch.tensor([0.1, 0.5, 0.9, 1.0], device=DEVICE).repeat(rows)[:rows]
    _check(
        x,
        k if "k" in mode else None,
        p if "p" in mode else None,
        f"nonfinite {mode} rows={rows}",
    )
