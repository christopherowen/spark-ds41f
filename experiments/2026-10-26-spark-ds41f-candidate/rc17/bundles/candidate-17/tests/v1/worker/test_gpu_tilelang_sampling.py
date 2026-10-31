# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The TileLang sampling kernels against the Triton kernels they replace, bit for
bit: the FP32 Gumbel noise of every one of the 2**32 Philox words, Triton's
randint streams, and every kernel's outputs (including the elements a launch
leaves alone) on logits with ties, infinities and NaN. The Gumbel finish is
compared with the torch argmax and gather it replaced."""

import itertools

import pytest
import torch

triton = pytest.importorskip("triton")
pytest.importorskip("tilelang")

import triton.language as tl  # noqa: E402
from triton.language.extra import libdevice as tldevice  # noqa: E402

from vllm.platforms import current_platform  # noqa: E402
from vllm.tilelang_utils import T, tilelang  # noqa: E402
from vllm.v1.worker.gpu.sample import gumbel, sink  # noqa: E402
from vllm.v1.worker.gpu.sample import tilelang_sampling as tls  # noqa: E402
from vllm.v1.worker.gpu.spec_decode import greedy_verification  # noqa: E402
from vllm.v1.worker.gpu.spec_decode.dspark import greedy  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)
_SM12X = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not current_platform.is_device_capability_family(120),
    reason="TileLang sampling kernels target SM120/SM121",
)

DEVICE = torch.device("cuda")
VOCAB = 129280  # DeepSeek V4.1
_MIN = tl.constexpr(4.6566127342e-10)


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


# --- The random stream --------------------------------------------------------


@triton.jit
def _triton_noise(words_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """gumbel_noised_scores' FP32 noise of each word: tl.rand's conversion of a
    tl.randint word, the clamp, -log(-log1p(-u))."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    word = tl.load(words_ptr + offs, mask=mask).to(tl.uint32, bitcast=True)
    u = tl.maximum(tl.random.uint_to_uniform_float(word), _MIN)
    tl.store(out_ptr + offs, -tl.log(-tldevice.log1p(-u)), mask=mask)


@tilelang.jit(pass_configs=tls._PASS)
def _tilelang_noise():
    n = T.dynamic("n")

    @T.prim_func
    def noise(words: T.Tensor((n,), T.int32), out: T.Tensor((n,), T.float32)):
        with T.Kernel(T.ceildiv(n, 256), threads=256) as bx:
            T.import_source(tls._HEADER)
            i = bx * 256 + T.get_thread_binding()
            if i < n:
                out[i] = T.call_extern(
                    "float32", "vllm_gumbel_noise", T.reinterpret(words[i], T.uint32)
                )

    return noise


def test_noise_of_every_word():
    """vllm_gumbel_noise equals Triton's noise on all 2**32 words."""
    chunk = 1 << 26
    kernel = _tilelang_noise()
    ours = torch.empty(chunk, dtype=torch.float32, device=DEVICE)
    theirs = torch.empty_like(ours)
    base = torch.arange(chunk, dtype=torch.int64, device=DEVICE)
    for start in range(0, 1 << 32, chunk):
        words = (base + start).to(torch.int32)  # wraps: the word's bits
        kernel(words, ours)
        _triton_noise[(chunk // 1024,)](words, theirs, chunk, BLOCK=1024)
        _assert_same(ours, theirs, f"noise of words {start:#x}..")


@triton.jit
def _triton_randint(seeds_ptr, offsets_ptr, keys_ptr, out64_ptr, out32_ptr):
    i = tl.program_id(0)
    seed = tl.load(seeds_ptr + i)
    word = tl.randint(seed, tl.load(offsets_ptr + i))
    tl.store(out64_ptr + i, word)
    tl.store(out32_ptr + i, tl.randint(word, tl.load(keys_ptr + i)))


@tilelang.jit(pass_configs=tls._PASS)
def _tilelang_randint():
    n = T.dynamic("n")

    @T.prim_func
    def randint(
        seeds: T.Tensor((n,), T.int64),
        offsets: T.Tensor((n,), T.int64),
        keys: T.Tensor((n,), T.int32),
        out64: T.Tensor((n,), T.uint32),
        out32: T.Tensor((n,), T.uint32),
    ):
        with T.Kernel(T.ceildiv(n, 256), threads=256) as bx:
            T.import_source(tls._HEADER)
            i = bx * 256 + T.get_thread_binding()
            if i < n:
                out64[i] = T.call_extern(
                    "uint32", "vllm_randint64", seeds[i], offsets[i]
                )
                out32[i] = T.call_extern("uint32", "vllm_randint32", out64[i], keys[i])

    return randint


def test_randint_streams():
    """tl.randint for int64 seeds and offsets, then for its own uint32 word as
    the seed and an int32 key: edge seeds and offsets, and random ones."""
    g = torch.Generator(device=DEVICE).manual_seed(0)
    n = 1 << 20
    edges = torch.tensor(
        [0, 1, -1, 2**31, -(2**31), 2**32, 2**32 - 1, 2**62, -(2**63), 2**63 - 1],
        dtype=torch.int64,
        device=DEVICE,
    )
    seeds = torch.randint(-(2**63), 2**63 - 1, (n,), generator=g, device=DEVICE)
    offsets = torch.randint(-(2**63), 2**63 - 1, (n,), generator=g, device=DEVICE)
    seeds[: edges.numel() ** 2] = edges.repeat_interleave(edges.numel())
    offsets[: edges.numel() ** 2] = edges.repeat(edges.numel())
    offsets[edges.numel() ** 2 : 2 * edges.numel() ** 2] = torch.arange(
        edges.numel() ** 2, device=DEVICE
    ) + (1 << 30)
    keys = torch.randint(-(2**31), 2**31 - 1, (n,), generator=g, device=DEVICE).to(
        torch.int32
    )
    keys[: 1 << 17] = torch.arange(1 << 17, device=DEVICE, dtype=torch.int32)
    theirs64 = torch.empty(n, dtype=torch.int32, device=DEVICE)
    theirs32 = torch.empty_like(theirs64)
    _triton_randint[(n,)](seeds, offsets, keys, theirs64, theirs32)
    ours64 = torch.empty(n, dtype=torch.uint32, device=DEVICE)
    ours32 = torch.empty_like(ours64)
    _tilelang_randint()(seeds, offsets, keys, ours64, ours32)
    _assert_same(ours64.view(torch.int32), theirs64, "randint(seed, offset)")
    _assert_same(ours32.view(torch.int32), theirs32, "randint(word, key)")


# --- Logits with the cases that decide an argmax ------------------------------


def _logits(rows: int, vocab: int, dtype: torch.dtype, case: str, seed: int):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    x = torch.randn(rows, vocab, generator=g, device=DEVICE) * 4
    if case == "ties":
        # Few distinct values: equal scores everywhere, and equal noise-free maxima.
        x = torch.round(x / 4)
    elif case == "special":
        n = x.numel()
        pick = torch.randperm(n, generator=g, device=DEVICE)
        x.view(-1)[pick[: n // 97]] = float("-inf")
        x.view(-1)[pick[n // 97 : n // 97 + n // 211]] = float("inf")
        x.view(-1)[pick[n // 97 + n // 211 : n // 97 + n // 211 + n // 307]] = float(
            "nan"
        )
        if rows > 1:
            x[0] = float("-inf")
            x[0, vocab // 2] = 3.0
        if rows > 2:
            x[1] = float("nan")
        if rows > 3:
            x[2, :2048] = float("nan")
            x[2, 0] = float("inf")
    return x.to(dtype)


CASES = ("random", "ties", "special")
# Row strides: a multiple of 16 (Triton lays the block out in 16-byte runs) or odd
# (one element a thread); each vocabulary is tested with both.
STRIDES = ("aligned", "odd")


def _stride(vocab: int, stride: str) -> int:
    return (vocab + 64 + 15) // 16 * 16 if stride == "aligned" else vocab + 3


# --- Gumbel sampling ----------------------------------------------------------


def _gumbel_both(
    logits,
    mapping,
    temperature,
    seeds,
    pos,
    apply_temperature,
    is_drafting,
    cache=None,
    col=None,
):
    """Both kernels' per-block outputs and logits caches."""
    tokens, vocab = logits.shape
    blocks = triton.cdiv(vocab, tls.GUMBEL_BLOCK)
    out = []
    for ours in (True, False):
        argmax = torch.full((tokens, blocks), -7, dtype=torch.int64, device=DEVICE)
        best = torch.full((tokens, blocks), -7.0, device=DEVICE)
        cache_copy = None if cache is None else cache.clone()
        if ours:
            key = tls.gumbel_sample_key(
                logits.dtype,
                mapping.dtype,
                apply_temperature,
                is_drafting,
                None if cache is None else cache.dtype,
                None if col is None else col.dtype,
                col is not None and col.dim() > 0,
                tls._triton_run(logits),
                pos.dtype,
            )
            tls._kernel(tls.gumbel_sample_kernel, *key)(
                argmax,
                best,
                logits,
                mapping,
                seeds,
                pos,
                temperature,
                best.unsqueeze(1) if cache is None else cache_copy,
                argmax.view(-1) if col is None else col.view(-1),
            )
        else:
            gumbel._gumbel_sample_kernel[(tokens, blocks)](
                argmax,
                argmax.stride(0),
                best,
                best.stride(0),
                cache_copy,
                0 if cache is None else cache.stride(0),
                0 if cache is None else cache.stride(1),
                col,
                logits,
                logits.stride(0),
                mapping,
                seeds,
                pos,
                temperature,
                vocab,
                BLOCK_SIZE=tls.GUMBEL_BLOCK,
                IS_DRAFTING=is_drafting,
                APPLY_TEMPERATURE=apply_temperature,
                USE_FP64=False,
                PER_TOKEN_COL=col is not None and col.dim() > 0,
            )
        out.append((argmax, best, cache_copy))
    return out


def _sampling_inputs(tokens: int, slots: int, temps: str, seed: int):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    mapping = torch.randperm(slots, generator=g, device=DEVICE)[:tokens] % slots
    if tokens > 2:
        mapping[1] = -1  # a padded row
    values = {
        "zero": [0.0],
        "one": [1.0],
        "mixed": [0.0, 1.0, 0.6, 1.3, 1e-3, 37.0],
    }[temps]
    temperature = torch.tensor(
        [values[i % len(values)] for i in range(slots)], device=DEVICE
    )
    seeds = torch.randint(-(2**63), 2**63 - 1, (slots,), generator=g, device=DEVICE)
    seeds[0] = 0
    pos = torch.randint(0, 1 << 20, (tokens,), generator=g, device=DEVICE)
    pos[0] = (1 << 33) + 5  # a high word in the offset
    return mapping.to(torch.int64), temperature, seeds, pos


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("vocab", [VOCAB, 1000, 2049])
@pytest.mark.parametrize("stride", STRIDES)
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("temps", ["zero", "one", "mixed"])
@pytest.mark.parametrize(
    "apply_temperature,is_drafting", [(False, False), (True, False), (True, True)]
)
def test_gumbel_sample(
    dtype, vocab, stride, case, temps, apply_temperature, is_drafting
):
    tokens, slots = 37, 48
    logits = _logits(tokens, _stride(vocab, stride), dtype, case, vocab)[:, :vocab]
    mapping, temperature, seeds, pos = _sampling_inputs(tokens, slots, temps, vocab)
    (a, b, _), (c, d, _) = _gumbel_both(
        logits, mapping, temperature, seeds, pos, apply_temperature, is_drafting
    )
    _assert_same(a, c, "block argmax")
    _assert_same(b, d, "block maximum")
    ours = tls.gumbel_sample(
        logits, mapping, temperature, seeds, pos, apply_temperature, is_drafting
    )
    # gumbel.gumbel_sample's finish on Triton's per-block outputs.
    theirs = c.gather(dim=-1, index=d.argmax(dim=-1, keepdim=True)).view(-1)
    _assert_same(ours, theirs, "sampled tokens")


def _int32_positions(pos: torch.Tensor) -> torch.Tensor:
    """``pos`` as int32 positions, two of them where the draft salt (2**30) reaches
    and passes 2**31 in 32 bits."""
    pos = (pos % (1 << 30)).to(torch.int32)
    pos[1] = (1 << 30) - 1
    pos[2] = (1 << 31) - 5
    return pos


@pytest.mark.parametrize(
    "apply_temperature,is_drafting", [(False, False), (True, False), (True, True)]
)
def test_gumbel_sample_int32_positions(apply_temperature, is_drafting):
    """Triton takes an int32 position as the Philox offset's only word and salts a
    draft's in 32 bits."""
    tokens, slots, vocab = 37, 48, 5000
    logits = _logits(tokens, vocab, torch.float32, "special", 5)
    mapping, temperature, seeds, pos = _sampling_inputs(tokens, slots, "mixed", 5)
    pos = _int32_positions(pos)
    (a, b, _), (c, d, _) = _gumbel_both(
        logits, mapping, temperature, seeds, pos, apply_temperature, is_drafting
    )
    _assert_same(a, c, "block argmax")
    _assert_same(b, d, "block maximum")
    ours = tls.gumbel_sample(
        logits, mapping, temperature, seeds, pos, apply_temperature, is_drafting
    )
    theirs = c.gather(dim=-1, index=d.argmax(dim=-1, keepdim=True)).view(-1)
    _assert_same(ours, theirs, "sampled tokens")


@pytest.mark.parametrize("col_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("per_token", [False, True])
@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float32])
def test_gumbel_sample_with_logits_cache(col_dtype, per_token, cache_dtype):
    tokens, slots, cols, vocab = 21, 32, 3, 5000
    logits = _logits(tokens, vocab, torch.bfloat16, "special", 3)
    mapping, temperature, seeds, pos = _sampling_inputs(tokens, slots, "mixed", 3)
    cache = torch.full((slots, cols, vocab + 16), -3.0, device=DEVICE).to(cache_dtype)
    if per_token:
        col = torch.arange(tokens, device=DEVICE).to(col_dtype) % cols
    else:
        col = torch.tensor(2, dtype=col_dtype, device=DEVICE)
    (a, b, e), (c, d, f) = _gumbel_both(
        logits, mapping, temperature, seeds, pos, True, True, cache, col
    )
    _assert_same(a, c, "block argmax")
    _assert_same(b, d, "block maximum")
    _assert_same(e.float(), f.float(), "logits cache")


def test_gumbel_rows_do_not_depend_on_the_batch():
    logits = _logits(64, VOCAB, torch.bfloat16, "random", 11)
    mapping, temperature, seeds, pos = _sampling_inputs(64, 64, "mixed", 11)
    whole = tls.gumbel_sample(logits, mapping, temperature, seeds, pos, True, False)
    for i in (0, 5, 63):
        alone = tls.gumbel_sample(
            logits[i : i + 1],
            mapping[i : i + 1],
            temperature,
            seeds,
            pos[i : i + 1],
            True,
            False,
        )
        assert alone.item() == whole[i].item()


# --- Gumbel sampling's finish ---------------------------------------------------


def _finish_inputs(tokens: int, blocks: int, case: str, seed: int):
    """Per-block maxima and argmaxes as the Gumbel kernel leaves them, as views of
    wider rows. The argmaxes are random ids, not increasing with the block, so a tie
    broken by id instead of by block shows. The "special" rows: all -inf, all NaN,
    NaN in the last block only, +inf twice, -0 before +0, NaN twice after +inf, the
    maximum in the last block, one value throughout; then scattered NaN and
    infinities in the rest."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    wide = torch.randn(tokens, blocks + 5, generator=g, device=DEVICE) * 4
    best = wide[:, :blocks]
    if case == "ties":
        best.copy_(torch.round(best / 4))  # equal maxima across blocks
    elif case == "special":
        n = best.numel()
        pick = torch.randperm(n, generator=g, device=DEVICE)
        flat = best.reshape(-1).clone()
        flat[pick[: n // 13]] = float("-inf")
        flat[pick[n // 13 : n // 13 + n // 29]] = float("inf")
        flat[pick[n // 13 + n // 29 : n // 13 + n // 29 + n // 31]] = float("nan")
        best.copy_(flat.view(tokens, blocks))
        last = blocks - 1
        rows = [
            lambda r: r.fill_(float("-inf")),
            lambda r: r.fill_(float("nan")),
            lambda r: (r.fill_(float("inf")), r[last:].fill_(float("nan"))),
            lambda r: (r.fill_(-1.0), r[last // 2 :].fill_(float("inf"))),
            lambda r: (r.fill_(-1.0), r[last // 3 :].fill_(-0.0), r[last:].fill_(0.0)),
            lambda r: (
                r.fill_(1.0),
                r[:1].fill_(float("inf")),
                r[last // 2 :].fill_(float("nan")),
            ),
            lambda r: (r.fill_(2.0), r[last:].fill_(3.0)),
            lambda r: r.fill_(0.5),
        ]
        for i in range(min(tokens, len(rows))):
            rows[i](best[i])
    argmax = torch.randint(
        0, 1 << 40, (tokens, blocks + 3), generator=g, device=DEVICE
    )[:, :blocks]
    return argmax, best


@_SM12X
@pytest.mark.parametrize("vocab", [VOCAB, 1000, 2049, 32768, 32769, 200000])
@pytest.mark.parametrize("tokens", [1, 37, 600])
@pytest.mark.parametrize("case", CASES)
def test_gumbel_finish(vocab, tokens, case):
    """``gumbel_finish`` against the torch argmax and gather it replaced, every
    block count's lanes (one block, a lane's 32 and 33, DeepSeek V4.1's 127, 196)."""
    blocks = triton.cdiv(vocab, tls.GUMBEL_BLOCK)
    argmax, best = _finish_inputs(tokens, blocks, case, vocab + tokens)
    ours = tls.gumbel_finish(argmax, best)
    theirs = argmax.gather(dim=-1, index=best.argmax(dim=-1, keepdim=True)).view(-1)
    assert ours.dtype == torch.int64 and ours.shape == (tokens,)
    _assert_same(ours, theirs, "sampled tokens")


@_SM12X
@pytest.mark.parametrize("vocab", [VOCAB, 1000, 200000])
def test_gumbel_finish_is_declared(vocab, monkeypatch):
    """The finish ``gumbel_sample`` launches is the call its declaration lists for
    the vocabulary (after the kernel cache freezes, an undeclared one raises)."""
    launched = set()
    original = tls._kernel

    def record(factory, *key):
        launched.add((id(factory), key))
        return original(factory, *key)

    monkeypatch.setattr(tls, "_kernel", record)
    logits = _logits(3, vocab, torch.float32, "random", 7)
    mapping, temperature, seeds, pos = _sampling_inputs(3, 4, "mixed", 7)
    tls.gumbel_sample(logits, mapping, temperature, seeds, pos, False, False)
    finish = id(tls.gumbel_finish_kernel)
    declared = {(id(f), args) for f, args, _ in tls.gumbel_finish_calls(vocab)}
    assert {key for key in launched if key[0] == finish} == declared


@pytest.mark.parametrize("vocab", [VOCAB, 100, 8193])
@pytest.mark.parametrize("stride", STRIDES)
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("tokens", [3, 19])  # both block widths
def test_temperature(vocab, stride, case, tokens):
    slots = 24
    logits = _logits(tokens, _stride(vocab, stride), torch.float32, case, 5)[:, :vocab]
    mapping, temperature, _, _ = _sampling_inputs(tokens, slots, "mixed", 5)
    mapping[1] = 4  # Triton reads the temperature without a validity check
    ours, theirs = logits.clone(), logits.clone()
    tls.apply_temperature(ours, mapping, temperature)
    gumbel._temperature_kernel[(tokens, triton.cdiv(vocab, 8192))](
        theirs, theirs.stride(0), mapping, temperature, vocab, BLOCK_SIZE=8192
    )
    _assert_same(ours, theirs, "tempered logits")


# --- The sink detector ---------------------------------------------------------


def _batch(counts, sampled_width, slots, seed):
    """A verification batch: requests with ``counts`` logits rows, their sampled
    tokens and drafts."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    reqs = len(counts)
    cu = torch.tensor(
        [0, *itertools.accumulate(counts)], dtype=torch.int32, device=DEVICE
    )
    rows = int(cu[-1])
    num_sampled = torch.tensor(
        [max(0, min(c, (i * 7) % (c + 2))) for i, c in enumerate(counts)],
        dtype=torch.int32,
        device=DEVICE,
    )
    drafts = torch.randint(-1, 4000, (rows + 4,), generator=g, device=DEVICE).to(
        torch.int32
    )
    sampled = torch.randint(0, 4000, (reqs, sampled_width), generator=g, device=DEVICE)
    # Some rejected drafts: the emitted token differs from the next row's input.
    sampled[::3, 0] = drafts[cu[:-1][::3].long() + 1].long()
    mapping = torch.randperm(slots, generator=g, device=DEVICE)[:reqs].to(torch.int64)
    return cu, num_sampled, drafts, sampled, mapping


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("vocab", [VOCAB, 1000])
@pytest.mark.parametrize("stride", STRIDES)
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("num_candidates", [4, 2])
@pytest.mark.parametrize("mask_rejected_draft", [True, False])
def test_sink_candidates(
    dtype, vocab, stride, case, num_candidates, mask_rejected_draft
):
    _check_sink_candidates(
        dtype, vocab, stride, case, num_candidates, mask_rejected_draft, torch.int64
    )


def test_sink_candidates_int32_positions():
    _check_sink_candidates(
        torch.bfloat16, VOCAB, STRIDES[0], CASES[0], 4, True, torch.int32
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("case", CASES)
def test_sink_candidates_with_dead_rows(dtype, case):
    """With a dead-row cut, verification ended each request's block at its cut, so
    a last draw at or past the cut was the block's bonus, which masked no draft:
    the candidates are those of the same drafts with each request's drafts from its
    cut on as -1 placeholders."""
    counts = [6, 1, 6, 3, 6, 2, 6, 6, 1, 4, 6]
    width, slots = 6, 16
    cu, num_sampled, drafts, sampled, mapping = _batch(counts, width, slots, 3)
    rows = int(cu[-1])
    logits = _logits(rows, _stride(VOCAB, STRIDES[0]), dtype, case, 3)
    _, temperature, seeds, _ = _sampling_inputs(len(counts), slots, "mixed", 3)
    pos = torch.randint(0, 1 << 20, (rows,), device=DEVICE)
    # Cuts from none live to all of a request's drafts, and past them.
    cut = torch.tensor(
        [min(i % 4, c - 1) if i % 5 else 9 for i, c in enumerate(counts)],
        dtype=torch.int32,
        device=DEVICE,
    )
    placeholders = drafts.clone()
    for r, c in enumerate(cut.tolist()):
        start, end = int(cu[r]), int(cu[r + 1])
        placeholders[start + 1 + c : end] = -1
    ours, expected = (
        tls.sink_candidates(
            logits, VOCAB, cu, num_sampled, sampled, d, mapping, temperature,
            seeds, pos, 4, True, dead_cut,
        )
        for d, dead_cut in ((drafts, cut), (placeholders, None))
    )  # fmt: skip
    _assert_same(ours[0], expected[0], "candidate values")
    _assert_same(ours[1], expected[1], "candidate indices")


def _check_sink_candidates(
    dtype, vocab, stride, case, num_candidates, mask_rejected_draft, pos_dtype
):
    counts = [6, 1, 6, 3, 6, 2, 6, 6, 1, 4, 6]
    width, slots = 6, 16
    cu, num_sampled, drafts, sampled, mapping = _batch(counts, width, slots, 2)
    rows = int(cu[-1])
    logits = _logits(rows, _stride(vocab, stride), dtype, case, 2)
    _, temperature, seeds, _ = _sampling_inputs(len(counts), slots, "mixed", 2)
    pos = torch.randint(0, 1 << 20, (rows,), device=DEVICE)
    if pos_dtype == torch.int32:
        pos = _int32_positions(pos)
    ours = tls.sink_candidates(
        logits,
        vocab,
        cu,
        num_sampled,
        sampled,
        drafts,
        mapping,
        temperature,
        seeds,
        pos,
        num_candidates,
        mask_rejected_draft,
    )
    blocks = triton.cdiv(vocab, sink._CANDIDATE_BLOCK_SIZE)
    values = torch.empty_like(ours[0])
    indices = torch.empty_like(ours[1])
    sink._sink_candidates_kernel[(len(counts), blocks)](
        values,
        indices,
        blocks * num_candidates,
        logits,
        logits.stride(0),
        vocab,
        cu,
        num_sampled,
        sampled,
        sampled.stride(0),
        drafts,
        mapping,
        temperature,
        seeds,
        pos,
        BLOCK_SIZE=sink._CANDIDATE_BLOCK_SIZE,
        NUM_CANDIDATES=num_candidates,
        USE_FP64=False,
        MASK_REJECTED_DRAFT=mask_rejected_draft,
    )
    _assert_same(ours[0], values, "candidate values")
    _assert_same(ours[1], indices, "candidate indices")


def _policy_state(reqs, width, slots, alternates, seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    cu, num_sampled, drafts, sampled, mapping = _batch(
        [width if i % 4 else 1 for i in range(reqs)], width, slots, seed
    )
    rows = int(cu[-1])
    na = max(alternates, 1)
    state = {
        "sampled": sampled,
        "num_sampled": num_sampled,
        "num_rejected": torch.zeros(reqs, dtype=torch.int32, device=DEVICE),
        "decoding": (torch.arange(reqs, device=DEVICE) % 5 != 2).to(torch.int8),
        "flags": (torch.rand(rows, generator=g, device=DEVICE) < 0.4).to(torch.int8),
        "values": torch.rand(rows, generator=g, device=DEVICE) * 2e5,
        "new_alternates": torch.randint(
            -1, 4000, (reqs, na), generator=g, device=DEVICE
        ),
        "new_logprobs": torch.randn(reqs, na, generator=g, device=DEVICE),
        "new_ranks": torch.randint(1, 50, (reqs, na), generator=g, device=DEVICE),
        "alternates": torch.randint(-1, 4000, (slots, na), generator=g, device=DEVICE),
        "alt_logprobs": torch.randn(slots, na, generator=g, device=DEVICE),
        "alt_ranks": torch.randint(1, 50, (slots, na), generator=g, device=DEVICE),
        "tries": torch.randint(0, na + 2, (slots,), generator=g, device=DEVICE).to(
            torch.int32
        ),
        "total_len": torch.randint(10, 100, (slots,), generator=g, device=DEVICE).to(
            torch.int32
        ),
        "bin_counts": torch.randint(1, 5, (slots, 4100), generator=g, device=DEVICE).to(
            torch.int32
        ),
        "counts": torch.randint(0, 9, (3,), generator=g, device=DEVICE),
    }
    state["values"][::7] = 0.0
    state["values"][3::11] = state["values"][2::11][: state["values"][3::11].numel()]
    return cu, drafts, mapping, state


@pytest.mark.parametrize(
    "rejection,logprobs", [(True, True), (True, False), (False, False)]
)
@pytest.mark.parametrize("width", [6, 1])
@pytest.mark.parametrize("alternate_major", [False, True])
def test_sink_policy(rejection, logprobs, width, alternate_major):
    """The policy against the Triton kernel. The Triton launch's recovered logprobs
    and ranks were zero-filled by torch; the TileLang kernel writes every request's
    (0 where none was recovered) over garbage. The step's alternate logprobs and
    ranks may be views of alternate-major storage (the sink's ranks are)."""
    reqs, slots, alternates, fields = 23, 40, 3, 6
    cu, drafts, mapping, state = _policy_state(reqs, width, slots, alternates, 9)
    results = []
    for ours in (True, False):
        s = {k: v.clone() for k, v in state.items()}
        events = torch.full((reqs, fields), -9, dtype=torch.int32, device=DEVICE)
        tokens = torch.full((reqs, width + 1), -9, dtype=torch.int64, device=DEVICE)
        num_out = torch.full((reqs,), -9, dtype=torch.int32, device=DEVICE)
        # Only a logprobs launch writes them: garbage for ours, zeros for Triton.
        garbage = ours and logprobs
        rec_lp = torch.full((reqs,), -9.0 if garbage else 0.0, device=DEVICE)
        rec_rk = torch.full(
            (reqs,), -9 if garbage else 0, dtype=torch.int64, device=DEVICE
        )
        new_logprobs, new_ranks = s["new_logprobs"], s["new_ranks"]
        if ours and alternate_major:
            new_logprobs = new_logprobs.t().contiguous().t()
            new_ranks = new_ranks.t().contiguous().t()
            assert not new_ranks.is_contiguous()
        args = (
            s["sampled"],
            s["num_sampled"],
            s["num_rejected"],
            tokens if rejection else s["sampled"],
            num_out if rejection else s["num_sampled"],
            events,
            rec_lp if logprobs else s["alt_logprobs"],
            rec_rk if logprobs else s["alt_ranks"],
            cu,
            mapping,
            s["decoding"],
            s["flags"],
            s["values"],
            drafts,
            s["new_alternates"] if rejection else s["alternates"],
            new_logprobs if logprobs else s["alt_logprobs"],
            new_ranks if logprobs else s["alt_ranks"],
            s["alternates"],
            s["alt_logprobs"],
            s["alt_ranks"],
            s["tries"],
            s["total_len"],
            s["bin_counts"],
        )
        if ours:
            tls.sink_policy(*args, s["counts"], alternates, rejection, logprobs, fields)
        else:
            sink._sink_policy_kernel[(reqs,)](
                *args,
                s["bin_counts"].stride(0),
                s["counts"],
                NUM_ALTERNATES=max(alternates, 1),
                REJECTION=rejection,
                LOGPROBS=logprobs,
                SAMPLED_WIDTH=width,
                PADDED_WIDTH=triton.next_power_of_2(width + 1),
                PADDED_ALTERNATES=triton.next_power_of_2(max(alternates, 1)),
                NUM_EVENT_FIELDS=fields,
            )
        results.append((s, events, tokens, num_out, rec_lp, rec_rk))
    (s1, *o1), (s2, *o2) = results
    for name in s1:
        _assert_same(s1[name], s2[name], name)
    for name, a, b in zip(("events", "tokens", "num_out", "logprobs", "ranks"), o1, o2):
        _assert_same(a, b, name)


# --- Greedy argmax and verification ------------------------------------------


def _triton_markov_pair(base, bias, vocab_start, pairs, values, indices):
    rows, vocab = base.shape
    parts = triton.cdiv(vocab, greedy._BLOCK)
    greedy._partial_argmax[(rows, parts)](
        base,
        bias,
        values,
        indices,
        base.stride(0),
        bias.stride(0),
        vocab,
        parts,
        greedy._BLOCK,
    )
    greedy._finish_argmax_pair[(rows,)](
        values, indices, pairs, vocab_start, parts, triton.next_power_of_2(parts)
    )


def _triton_shard_pair(logits, vocab, vocab_start, pairs, values, indices):
    rows = logits.shape[0]
    parts = triton.cdiv(vocab, greedy._BLOCK)
    greedy._partial_argmax_logits[(rows, parts)](
        logits, values, indices, logits.stride(0), vocab, parts, greedy._BLOCK
    )
    greedy._finish_argmax_pair[(rows,)](
        values, indices, pairs, vocab_start, parts, triton.next_power_of_2(parts)
    )


def _triton_select(pairs, output):
    rows, width = pairs.shape
    ranks = width // 2
    greedy._select_first_argmax[(rows,)](
        pairs, output, output.stride(0), ranks, triton.next_power_of_2(ranks)
    )


def _triton_markov(base, bias, output, values, indices):
    rows, vocab = base.shape
    parts = triton.cdiv(vocab, greedy._BLOCK)
    greedy._partial_argmax[(rows, parts)](
        base,
        bias,
        values,
        indices,
        base.stride(0),
        bias.stride(0),
        vocab,
        parts,
        greedy._BLOCK,
    )
    greedy._finish_argmax[(rows,)](
        values, indices, output, output.stride(0), parts, triton.next_power_of_2(parts)
    )


def _triton_greedy_rejection(target, drafts, cu, steps):
    reqs = cu.shape[0] - 1
    sampled = drafts.new_empty(reqs, steps + 1, dtype=torch.int64)
    num_sampled = sampled.new_empty(reqs, dtype=torch.int32)
    greedy_verification._greedy_rejection_kernel[(reqs,)](
        sampled, sampled.stride(0), num_sampled, target, drafts, cu, num_warps=1
    )
    return sampled, num_sampled


def _scratch(rows, vocab):
    shape = greedy.scratch_shape(rows, vocab)
    return (
        torch.full(shape, -5.0, device=DEVICE),
        torch.full(shape, -5, dtype=torch.int32, device=DEVICE),
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("aligned", [True, False])
@pytest.mark.parametrize("vocab", [32320, VOCAB, 2051])
def test_greedy_markov_shard_pair(dtype, case, aligned, vocab):
    rows = 13
    width = vocab + 8 + (0 if aligned else 1)
    base = _logits(rows, width + 1, dtype, case, 1)[:, (0 if aligned else 1) :][
        :, :vocab
    ]
    bias = _logits(rows, width, dtype, case, 2)[:, :vocab]
    results = []
    for fn in (tls.greedy_markov_shard_pair, _triton_markov_pair):
        pairs = torch.full((rows, 2), -3.0, device=DEVICE)
        values, indices = _scratch(rows, vocab)
        fn(base, bias, 64640, pairs, values, indices)
        results.append((pairs, values, indices))
    for name, a, b in zip(("pairs", "values", "indices"), *results):
        _assert_same(a, b, name)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("case", CASES)
def test_greedy_shard_pair(dtype, case):
    rows, vocab = 29, 32320
    logits = _logits(rows, vocab + 64, dtype, case, 3)
    results = []
    for fn in (tls.greedy_shard_pair, _triton_shard_pair):
        pairs = torch.full((rows, 2), -3.0, device=DEVICE)
        values, indices = _scratch(rows, vocab)
        fn(logits, vocab, 3 * vocab, pairs, values, indices)
        results.append((pairs, values, indices))
    for name, a, b in zip(("pairs", "values", "indices"), *results):
        _assert_same(a, b, name)


@pytest.mark.parametrize("ranks", [1, 3, 4])
@pytest.mark.parametrize("out_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("strided", [False, True])
def test_select_first_argmax(ranks, out_dtype, strided):
    rows = 41
    g = torch.Generator(device=DEVICE).manual_seed(ranks)
    values = torch.round(torch.randn(rows, ranks, generator=g, device=DEVICE))
    values[3] = float("nan")
    values[5, 0] = float("-inf")
    values[7] = float("-inf")
    index = torch.randint(0, 1 << 23, (rows, ranks), generator=g, device=DEVICE).float()
    pairs = torch.stack([values, index], dim=-1).view(rows, 2 * ranks)
    outs = []
    for fn in (tls.select_first_argmax, _triton_select):
        full = torch.full(
            (rows, 3 if strided else 1), -2, dtype=out_dtype, device=DEVICE
        )
        fn(pairs, full[:, 1] if strided else full[:, 0])
        outs.append(full)
    _assert_same(outs[0], outs[1], "selected index")


@pytest.mark.parametrize("case", CASES)
def test_greedy_pairs_in_sixteen_byte_rows(case):
    """The drafter's and the greedy verification's 16-byte pair rows ([rows, 4],
    which the one-shot all-gather writes in place): the shard pair lands in each
    row's first two columns, the others untouched, and the selection over the
    gathered rows equals Triton's over the packed [rows, 2 * ranks] pairs."""
    rows, vocab, ranks = 13, 32320, 4
    base = _logits(rows, vocab, torch.bfloat16, case, 6)
    bias = _logits(rows, vocab, torch.bfloat16, case, 7)
    packed, wide = [], []
    for rank in range(ranks):
        narrow = torch.full((rows, 2), -3.0, device=DEVICE)
        values, indices = _scratch(rows, vocab)
        _triton_markov_pair(
            base.roll(rank, 1), bias, rank * vocab, narrow, values, indices
        )
        rows4 = torch.full((rows, 4), -3.0, device=DEVICE)
        tls.greedy_markov_shard_pair(
            base.roll(rank, 1), bias, rank * vocab, rows4, values, indices
        )
        _assert_same(rows4[:, :2], narrow, "pair")
        assert (rows4[:, 2:] == -3.0).all()
        packed.append(narrow)
        wide.append(rows4)
    outs = []
    for fn, pairs, kwargs in (
        (tls.select_first_argmax, torch.cat(wide, dim=-1), {"pair_width": 4}),
        (_triton_select, torch.cat(packed, dim=-1), {}),
    ):
        out = torch.full((rows,), -2, dtype=torch.int64, device=DEVICE)
        fn(pairs, out, **kwargs)
        outs.append(out)
    _assert_same(outs[0], outs[1], "selected index")


@pytest.mark.parametrize("ranks", [1, 3, 4])
def test_select_first_argmax_embeds_the_next_token(ranks):
    """The drafter's selection with the next step's Markov embedding: the same
    index as without it, and the selected tokens' W1 rows (``nn.Embedding``) in
    the step buffer and in the (request, step) buffer's column; the anchor
    lookup likewise from ``ids[index]``."""
    rows, vocab, width, steps = 13, 1 << 17, 256, 5
    g = torch.Generator(device=DEVICE).manual_seed(ranks + 40)
    values = torch.round(torch.randn(rows, ranks, generator=g, device=DEVICE))
    index = torch.randint(0, vocab, (rows, ranks), generator=g, device=DEVICE)
    pairs = torch.zeros(rows, ranks, 4, device=DEVICE)
    pairs[:, :, 0], pairs[:, :, 1] = values, index.float()
    pairs = pairs.view(rows, 4 * ranks)
    table = torch.randn(vocab, width, generator=g, device=DEVICE).bfloat16()
    plain = torch.full((rows,), -2, dtype=torch.int64, device=DEVICE)
    tls.select_first_argmax(pairs, plain, pair_width=4)
    out = torch.full((rows, steps), -2, dtype=torch.int64, device=DEVICE)
    step = torch.zeros(rows + 3, width, dtype=torch.bfloat16, device=DEVICE)
    confidence = torch.zeros(rows, steps, width, dtype=torch.bfloat16, device=DEVICE)
    tls.select_first_argmax(
        pairs, out[:, 2], pair_width=4, embed=(table, step[:rows], confidence[:, 3])
    )
    _assert_same(out[:, 2], plain, "selected index")
    expected = torch.nn.functional.embedding(plain, table)
    _assert_same(step[:rows], expected, "step rows")
    _assert_same(confidence[:, 3], expected, "confidence rows")
    assert (step[rows:] == 0).all() and (confidence[:, :3] == 0).all()
    ids = torch.randint(0, vocab, (64,), generator=g, device=DEVICE).int()
    anchors = torch.arange(rows, device=DEVICE) * 4
    tls.markov_anchor_embed(ids, anchors, table, step[:rows], confidence[:, 0])
    expected = torch.nn.functional.embedding(ids[anchors].long(), table)
    _assert_same(step[:rows], expected, "anchor step rows")
    _assert_same(confidence[:, 0], expected, "anchor confidence rows")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("case", CASES)
def test_sample_greedy_markov(dtype, case):
    rows, vocab = 17, VOCAB
    base = _logits(rows, vocab, dtype, case, 4)
    bias = _logits(rows, vocab, dtype, case, 5)
    outs = []
    for fn in (tls.sample_greedy_markov, _triton_markov):
        out = torch.full((rows, 4), -2, dtype=torch.int64, device=DEVICE)
        values, indices = _scratch(rows, vocab)
        fn(base, bias, out[:, 2], values, indices)
        outs.append((out, values, indices))
    for name, a, b in zip(("output", "values", "indices"), *outs):
        _assert_same(a, b, name)


@pytest.mark.parametrize("steps", [5, 1])
@pytest.mark.parametrize("drafts_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("ranks", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("pair_width,strided", [(4, False), (4, True), (2, False)])
def test_greedy_rejection(steps, drafts_dtype, ranks, pair_width, strided):
    """Greedy verification from the gathered pairs, one launch, against Triton's
    two: ``_select_first_argmax`` over the packed pairs, then
    ``_greedy_rejection_kernel`` over its targets. The pairs carry ties within and
    across ranks, NaN (alone and in every rank) and -inf rows, junk in the columns
    past each pair, and (strided) a row stride past the pairs; drafts accept, miss,
    hold -1 and end blocks fully accepted."""
    g = torch.Generator(device=DEVICE).manual_seed(steps * 100 + ranks)
    counts = [steps + 1 if i % 3 else 1 for i in range(37)]
    cu = torch.tensor(
        [0, *itertools.accumulate(counts)], dtype=torch.int32, device=DEVICE
    )
    rows = int(cu[-1])
    values = torch.round(torch.randn(rows, ranks, generator=g, device=DEVICE))
    values[2::7, ranks - 1] = float("nan")
    values[5::13] = float("nan")
    values[9::11] = -float("inf")
    index = torch.randint(0, 6, (rows, ranks), generator=g, device=DEVICE).float()
    index[4::17] = torch.randint(
        0, 1 << 24, index[4::17].shape, generator=g, device=DEVICE
    ).float()
    full = torch.full((rows, ranks, pair_width), 7.5, device=DEVICE)
    full[..., 0], full[..., 1] = values, index
    storage = torch.full(
        (rows, ranks * pair_width + (3 if strided else 0)), -2.0, device=DEVICE
    )
    pairs = storage[:, : ranks * pair_width]
    pairs.copy_(full.view(rows, ranks * pair_width))
    packed = full[..., :2].contiguous().view(rows, 2 * ranks)
    target = torch.empty(rows, dtype=torch.int64, device=DEVICE)
    _triton_select(packed, target)
    # A draft verifies the row before it: most are that row's target (accepted),
    # some miss or are -1.
    drafts = torch.empty(rows + 1, dtype=torch.int64, device=DEVICE)
    drafts[0] = 0
    drafts[1:] = target
    drafts[3::5] = (drafts[3::5] + 1) % 6
    drafts[7::9] = -1
    drafts = drafts.to(drafts_dtype)
    ours = tls.greedy_rejection_sample(pairs, drafts, cu, steps, pair_width)
    theirs = _triton_greedy_rejection(target, drafts, cu, steps)
    _assert_same(ours[1], theirs[1], "num_sampled")
    assert (ours[1] > 1).any() and (ours[1] < steps + 1).any()
    for r in range(len(counts)):
        n = int(theirs[1][r])
        _assert_same(ours[0][r, :n], theirs[0][r, :n], f"request {r}'s tokens")


@pytest.mark.parametrize("ranks", [1, 3, 4])
def test_greedy_shard_runner_up_and_margin_selection(ranks):
    """With ``track_runner_up=True``, column 2 of the 16-byte shard pair row
    holds the best non-winning partial maximum of the shard, and
    ``select_first_argmax(..., margin_out=...)`` writes the non-negative logit
    margin ``max(best - runner_up, 0)`` alongside the selected index and Markov
    embeddings without altering columns 0..1 or the selected token."""
    rows, vocab, width, steps = 9, 8192, 256, 5
    g = torch.Generator(device=DEVICE).manual_seed(ranks + 99)
    base = torch.randn(rows, ranks * vocab, generator=g, device=DEVICE).bfloat16()
    bias = torch.randn(rows, ranks * vocab, generator=g, device=DEVICE).bfloat16()
    # Row 1: exact tie between top-1 and top-2 across shards -> margin 0.
    base[1, 10] = 25.0
    bias[1, 10] = 0.0
    if ranks > 1:
        base[1, vocab + 20] = 25.0
        bias[1, vocab + 20] = 0.0
    else:
        # Tie in another 2048-token block of the same shard.
        base[1, 2048 + 20] = 25.0
        bias[1, 2048 + 20] = 0.0

    wide_plain, wide_margin = [], []
    for rank in range(ranks):
        s_base = base[:, rank * vocab : (rank + 1) * vocab]
        s_bias = bias[:, rank * vocab : (rank + 1) * vocab]
        p_plain = torch.full((rows, 4), -3.0, device=DEVICE)
        p_margin = torch.full((rows, 4), -3.0, device=DEVICE)
        v1, i1 = _scratch(rows, vocab)
        v2, i2 = _scratch(rows, vocab)
        tls.greedy_markov_shard_pair(
            s_base, s_bias, rank * vocab, p_plain, v1, i1, track_runner_up=False
        )
        tls.greedy_markov_shard_pair(
            s_base, s_bias, rank * vocab, p_margin, v2, i2, track_runner_up=True
        )
        _assert_same(p_margin[:, :2], p_plain[:, :2], "winner pair")
        assert (p_plain[:, 2:] == -3.0).all()
        assert (p_margin[:, 3] == -3.0).all()
        assert (p_margin[:, 2] <= p_margin[:, 0]).all()
        wide_plain.append(p_plain)
        wide_margin.append(p_margin)

    gathered = torch.cat(wide_margin, dim=-1)
    plain_out = torch.full((rows,), -2, dtype=torch.int64, device=DEVICE)
    tls.select_first_argmax(torch.cat(wide_plain, dim=-1), plain_out, pair_width=4)

    margin_out = torch.full((rows, steps), -9.0, device=DEVICE)
    out = torch.full((rows, steps), -2, dtype=torch.int64, device=DEVICE)
    table = torch.randn(ranks * vocab, width, generator=g, device=DEVICE).bfloat16()
    step = torch.zeros(rows, width, dtype=torch.bfloat16, device=DEVICE)
    confidence = torch.zeros(rows, steps, width, dtype=torch.bfloat16, device=DEVICE)
    tls.select_first_argmax(
        gathered,
        out[:, 1],
        pair_width=4,
        embed=(table, step, confidence[:, 2]),
        margin_out=margin_out[:, 1],
    )
    _assert_same(out[:, 1], plain_out, "selected index with margin")
    assert (margin_out[:, 1] >= 0.0).all()
    assert margin_out[1, 1].item() == 0.0
    expected_embed = torch.nn.functional.embedding(plain_out, table)
    _assert_same(step, expected_embed, "step embed with margin")
    _assert_same(confidence[:, 2], expected_embed, "confidence embed with margin")
