# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``tilelang_logprob.topk_log_softmax`` against an FP64 reference of Barron,
Gnanapragasam and Tschannen's log-softmax ("A Fix for Softmax and Cross-Entropy",
Eq. 5) on the same logits, and against today's arithmetic, the Triton kernel
``logprob._topk_log_softmax_kernel`` launched as production launched it (the
release's TileLang port matched it bit for bit):

- every output within its rounding bound of the reference (the shifted logit's and
  the output's rounding, an ULP of the log-sum-exp, the sum's relative error and,
  for sums in FP32's subnormal range, an absolute subnormal per column), with the
  reference's NaN and infinities;
- never further from the reference than today's output beyond that rounding;
- exact where today's collapses: the paper's z = [9, -9, -9], and margins of the
  top logit over the others' log-sum-exp from 4 to 100 nats around FP32's cliff at
  16.6, the maximum at the row's first, middle and last column (today's sum loses
  the blocks added after the maximum's);
- a row's bits do not depend on its batch or the run.

FP32 and bf16 logits; rows at 16-byte addresses and strides (Triton's 16-byte runs),
at an odd stride and at a base past 16 bytes (one element a thread); the DeepSeek
V4.1 vocabulary, a tensor-parallel shard, odd widths and whole steps of the kernel's
reads; 0 to 2100 requested tokens; rows of infinities, NaN, ties, signed zeros,
extreme and subnormal magnitudes, and exponentials that underflow to subnormals. Then the call sites:
``compute_token_logprobs`` and ``compute_topk_scores`` launch the TileLang kernel
only, a CUDA graph replays it, and the runner declares the calls it launches."""

import math

import pytest
import torch

from vllm.platforms import current_platform

if not (
    current_platform.is_cuda() and current_platform.is_device_capability_family(120)
):
    pytest.skip("TileLang logprob kernels target SM120/SM121", allow_module_level=True)

pytest.importorskip("tilelang")
pytest.importorskip("triton")

from vllm.v1.worker.gpu.sample import logprob, tilelang_logprob  # noqa: E402
from vllm.v1.worker.gpu.sample import tilelang_sampling as tls  # noqa: E402

DEVICE = torch.device("cuda")
VOCAB = 129280  # DeepSeek V4.1
SHARD = 32320  # a quarter of it
PADDED = 131072  # the LM head's padded rows
DTYPES = (torch.float32, torch.bfloat16)
# "aligned": a 16-byte base and a stride of 16 elements, Triton's 16-byte runs;
# "odd_stride" and "shifted_base" (one element past a 16-byte base): one element a
# thread.
LAYOUTS = ("aligned", "odd_stride", "shifted_base")
CASES = ("random", "wide", "ties", "peaked", "special")
# What the padding past each row holds: a read of it would win the maximum.
PAD = 1.0e4
SENTINEL = 12345.0
# The sum of the other columns' exponentials, S, within this of its value: each
# exponential within 1.4e-7 |x - max| + 2^-22 of its own (the shift's and the
# log2(e) product's roundings, log2(e) in FP32, ex2.approx): 1.1e-5 at the 75 nats
# the margins test reaches; then the blocked sum's roundings (the CPU emulation in
# experiments/2026-10-25-softmax-fix measures about 1e-6). Columns whose
# exponentials are subnormal add up to one subnormal each.
S_RELATIVE = 2e-5
SUBNORMAL = 2.0**-149
# Today's sum and this one round independently: the allowance for this one's error
# over today's.
S_ALLOWANCE = 4e-6


def _bits(x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.float32:
        return x.view(torch.int32)
    if x.dtype in (torch.bfloat16, torch.float16):
        return x.view(torch.int16)
    return x


def _assert_same(ours: torch.Tensor, theirs: torch.Tensor, what: str) -> None:
    a, b = _bits(ours), _bits(theirs)
    if torch.equal(a, b):
        return
    differ = (a != b).nonzero()
    first = differ[:5].tolist()
    raise AssertionError(
        f"{what}: {differ.shape[0]} of {a.numel()} elements differ, first at "
        f"{first}: ours {[ours[tuple(i)].item() for i in first]}, "
        f"expected {[theirs[tuple(i)].item() for i in first]}"
    )


# --- Inputs -------------------------------------------------------------------


def _values(rows: int, vocab: int, case: str, seed: int) -> torch.Tensor:
    """FP32 logits ([rows, vocab]). "random": normal, sd 4; "wide": sd 40, most
    exponentials underflow; "ties": few distinct values; "peaked": one 0 a row,
    the rest about -12 (a sum near 1, its log near 0); "special": -inf, +inf and NaN
    scattered, and each row r one of
    (r % 12): all -inf but one; all NaN; all -inf; zeros of both signs; all equal;
    +inf and NaN; magnitudes up to 3e38; subnormal magnitudes; exponentials that
    are subnormal (the maximum 0, the rest in [-103, -87]); all NaN but one; two
    +inf; scattered only."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    x = torch.randn(rows, vocab, generator=g, device=DEVICE) * 4
    if case == "wide":
        x = x * 10
    elif case == "ties":
        x = torch.round(x / 4)
    elif case == "peaked":
        x = x / 4 - 12
        x[:, vocab // 2] = 0.0
    elif case == "special":
        n = x.numel()
        pick = torch.randperm(n, generator=g, device=DEVICE)
        flat = x.view(-1)
        a, b, c = n // 97, n // 97 + n // 211, n // 97 + n // 211 + n // 307
        flat[pick[:a]] = float("-inf")
        flat[pick[a:b]] = float("inf")
        flat[pick[b:c]] = float("nan")
        for r in range(rows):
            row = x[r]
            kind = r % 12
            if kind == 0:
                row.fill_(float("-inf"))
                row[vocab // 2] = 3.0
            elif kind == 1:
                row.fill_(float("nan"))
            elif kind == 2:
                row.fill_(float("-inf"))
            elif kind == 3:
                row.zero_()
                row[1::2] = -0.0
            elif kind == 4:
                row.fill_(5.0)
            elif kind == 5:
                row[0] = float("inf")
                row[vocab - 1] = float("nan")
            elif kind == 6:
                # uniform_'s range width (6e38) would overflow.
                unit = torch.rand(vocab, generator=g, device=DEVICE)
                row.copy_((unit * 2 - 1) * 3.0e38)
            elif kind == 7:
                row.copy_(torch.randn(vocab, generator=g, device=DEVICE) * 1.0e-39)
            elif kind == 8:
                row.uniform_(-103.0, -87.0, generator=g)
                row[vocab // 3] = 0.0
            elif kind == 9:
                row.fill_(float("nan"))
                row[vocab - 1] = -2.0
            elif kind == 10:
                row[0] = float("inf")
                row[vocab // 2] = float("inf")
    return x


def _logits(values: torch.Tensor, dtype: torch.dtype, layout: str) -> torch.Tensor:
    """``values`` in ``dtype`` rows of ``layout``, the padding past each row ``PAD``."""
    rows, vocab = values.shape
    odd = layout == "odd_stride"
    stride = vocab + 3 if odd else (vocab + 64 + 15) // 16 * 16
    flat = torch.full((rows * stride + 1,), PAD, dtype=dtype, device=DEVICE)
    start = 1 if layout == "shifted_base" else 0
    logits = flat[start : start + rows * stride].view(rows, stride)[:, :vocab]
    logits.copy_(values.to(dtype))
    run = 16 // logits.element_size() if layout == "aligned" else 1
    assert tls._triton_run(logits) == run
    return logits


def _ids(values: torch.Tensor, topk: int, seed: int) -> torch.Tensor:
    """Random requested tokens ([rows, topk], int64), the first three the first
    column, the last and the row's maximum."""
    rows, vocab = values.shape
    g = torch.Generator(device=DEVICE).manual_seed(seed + 1)
    ids = torch.randint(vocab, (rows, topk), generator=g, device=DEVICE)
    fixed = [
        torch.zeros(rows, dtype=torch.int64, device=DEVICE),
        torch.full((rows,), vocab - 1, dtype=torch.int64, device=DEVICE),
        values.nan_to_num(nan=float("-inf")).argmax(dim=1),
    ]
    for k, column in enumerate(fixed[:topk]):
        ids[:, k] = column
    return ids


# --- The launches and the reference ---------------------------------------------


def _triton(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Today's arithmetic."""
    rows, vocab = logits.shape
    out = torch.full((rows, ids.shape[1]), SENTINEL, device=DEVICE)
    logprob._topk_log_softmax_kernel[(rows,)](
        out,
        logits,
        logits.stride(0),
        ids,
        ids.shape[1],
        vocab,
        BLOCK_SIZE=1024,
        TOPK_BLOCK_SIZE=logprob._MAX_TOPK_BLOCK,
    )
    return out


def _tilelang(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    out = torch.full((logits.shape[0], ids.shape[1]), SENTINEL, device=DEVICE)
    tilelang_logprob.topk_log_softmax(logits, ids, out)
    return out


def _reference(logits: torch.Tensor, ids: torch.Tensor):
    """Eq. 5 in FP64 on the kernel's FP32 view of ``logits``: with M the row's
    maximum (NaN ignored), S the sum of exp(x - M) over the columns that are not M,
    plus exp(M - M) for each of M's ties but one, the requested tokens' log
    probabilities ``shifted - log1p(S)`` (``shifted`` exactly 0 at M). Returns them,
    their shifted logits, and each row's S and log1p(S) ([rows, 1])."""
    v = logits.float().double()
    m = torch.where(torch.isnan(v), float("-inf"), v).amax(dim=1, keepdim=True)
    tie = v == m
    ties = tie.sum(dim=1, keepdim=True).double()
    terms = torch.where(tie, torch.zeros_like(v), torch.exp(v - m))
    extra = torch.where(ties > 1, (ties - 1) * torch.exp(m - m), torch.zeros_like(m))
    s = terms.sum(dim=1, keepdim=True) + extra
    lse = torch.log1p(s)
    picked = v.gather(1, ids.long())
    shifted = torch.where(picked == m, torch.zeros_like(picked), picked - m)
    return shifted - lse, shifted, s, lse


def _ulp(x: torch.Tensor) -> torch.Tensor:
    """The FP32 ULP at |x| (FP64)."""
    a = x.abs().float()
    return (torch.nextafter(a, torch.full_like(a, float("inf"))) - a).double()


def _assert_accurate(ours, theirs, logits, ids, what: str):
    """``ours`` against the reference: its NaN and infinities, every finite output
    within its rounding bound, and never further from it than today's output
    (``theirs``) beyond the two sums' independent roundings. Returns the reference
    and each output's S (for the callers' own checks)."""
    vocab = logits.shape[1]
    ref, shifted, s, lse = _reference(logits, ids)
    s, lse = s.expand_as(ref), lse.expand_as(ref)
    expect = ref.float()
    nan = torch.isnan(expect)
    if not torch.equal(torch.isnan(ours), nan):
        at = (torch.isnan(ours) != nan).nonzero()[:5].tolist()
        raise AssertionError(f"{what}: NaN where the reference has none or the "
                             f"reverse, at {at}")  # fmt: skip
    inf = torch.isinf(expect)
    _assert_same(ours[inf], expect[inf], f"{what}: infinities")
    finite = ~nan & ~inf
    error = (ours.double() - ref).abs()
    floor = vocab * SUBNORMAL
    bound = (
        0.5 * _ulp(shifted) + _ulp(ref) + _ulp(lse) + S_RELATIVE * s / (1 + s) + floor
    )
    bad = finite & ~(error <= bound)
    if bad.any():
        at = bad.nonzero()[:5].tolist()
        raise AssertionError(
            f"{what}: {int(bad.sum())} outputs outside their bound, first at {at}: "
            f"ours {[ours[tuple(i)].item() for i in at]}, reference "
            f"{[ref[tuple(i)].item() for i in at]}, bound "
            f"{[bound[tuple(i)].item() for i in at]}"
        )
    today = finite & torch.isfinite(theirs)
    allowance = _ulp(ref) + _ulp(lse) + S_ALLOWANCE * s / (1 + s) + floor
    worse = today & ~(error <= (theirs.double() - ref).abs() + allowance)
    if worse.any():
        at = worse.nonzero()[:5].tolist()
        raise AssertionError(
            f"{what}: {int(worse.sum())} outputs further from the reference than "
            f"today's, first at {at}: ours {[ours[tuple(i)].item() for i in at]}, "
            f"today's {[theirs[tuple(i)].item() for i in at]}, reference "
            f"{[ref[tuple(i)].item() for i in at]}"
        )
    return ref, s


def _check(logits: torch.Tensor, ids: torch.Tensor, what: str) -> torch.Tensor:
    before = logits.clone()
    ours = _tilelang(logits, ids)
    theirs = _triton(logits, ids)
    _assert_same(logits, before, f"{what}: logits")
    _assert_accurate(ours, theirs, logits, ids, what)
    return ours


# --- Accuracy -----------------------------------------------------------------


# Besides DeepSeek V4.1's widths and odd ones, whole steps with no guarded one: a
# step is 1024 threads x 2 loads x a run of 1 (2048 columns), 4 FP32 (8192) or 8
# bf16 (16384).
STEPS = (2048, 8192, 16384)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize(
    "vocab",
    [VOCAB, SHARD, 1, 7, 1023, 1025, 8193, 16391, *STEPS],
    ids=lambda v: f"v{v}",
)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_log_softmax_matches_fp64(dtype, layout, vocab, case):
    """A sampled token and twenty logprobs a row."""
    values = _values(12, vocab, case, seed=vocab + len(case))
    logits = _logits(values, dtype, layout)
    _check(logits, _ids(values, 21, vocab), f"{dtype} {layout} {vocab} {case}")


@pytest.mark.parametrize("topk", [1, 2, 5, 20, 128, 129, 1024, 1025, 2100])
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_requested_counts(dtype, layout, topk):
    """Every count, past one 1024-token tile and its 128-token slices included,
    and ids that repeat."""
    values = _values(12, SHARD, "special", seed=topk)
    logits = _logits(values, dtype, layout)
    _check(logits, _ids(values, topk, topk), f"{dtype} {layout} topk {topk}")


@pytest.mark.parametrize("rows", [1, 2, 16, 96, 300])
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_batch_sizes(dtype, rows):
    """Decode and prefill batches at the LM head's padded stride."""
    values = _values(rows, VOCAB, "random", seed=rows)
    flat = torch.full((rows, PADDED), PAD, dtype=dtype, device=DEVICE)
    logits = flat[:, :VOCAB]
    logits.copy_(values.to(dtype))
    _check(logits, _ids(values, 21, rows), f"{dtype} {rows} rows")


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_paper_example(dtype):
    """z = [9, -9, -9]: the top token's log probability is -log1p(2 e^-18) =
    -3.046e-8, its complement 1 - p = 3.046e-8; today's is exactly 0 (so 1 - p
    is 0). The others' -18 - 3e-8 rounds to -18 in FP32 either way."""
    logits = _logits(torch.tensor([[9.0, -9.0, -9.0]], device=DEVICE), dtype, "aligned")
    ids = torch.tensor([[0, 1, 2]], device=DEVICE)
    ours = _tilelang(logits, ids)
    theirs = _triton(logits, ids)
    _assert_accurate(ours, theirs, logits, ids, f"paper example {dtype}")
    others = 2 * math.exp(-18.0)
    expected = -math.log1p(others)
    assert abs(ours[0, 0].item() - expected) <= S_RELATIVE * abs(expected)
    complement = -torch.expm1(ours[0, :1]).item()
    assert abs(complement - others / (1 + others)) <= S_RELATIVE * complement
    assert theirs[0, 0].item() == 0.0
    assert ours[0, 1].item() == ours[0, 2].item() == -18.0


# Margins of the top logit over the others' log-sum-exp, nats: FP32's cliff
# (2^-24, 16.6) inside; past about 75 here the largest of the other exponentials
# turn subnormal, past 87.3 S itself, and past about 103 it underflows (the
# reference's log probability is then below FP32's smallest subnormal too).
MARGINS = (4, 8, 10, 12, 13, 14, 15, 16, 16.3, 16.6, 16.9, 17.5, 18, 20, 25, 30,
           36.7, 40, 50, 60, 70, 80, 87, 90, 95, 100)  # fmt: skip


@pytest.mark.parametrize("column", ["first", "middle", "last"])
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_margins_around_the_cliff(dtype, column):
    """The top token's log probability, -log1p(S) = -S (1 - S / 2 ...), from margin
    4 to 100, the maximum at the row's first, middle or last column: within
    S_RELATIVE of the reference while S is at least a normal FP32 number a column
    (its subnormal terms then add at most 2^-24 of it), and within a subnormal a
    column below that (``_assert_accurate``). Today's is exactly 0 once S is below
    2^-25, wherever the maximum sits; short of that its sum loses each block added
    after the maximum's block below half an ULP of 1, so its error depends on the
    column: never better than ours."""
    g = torch.Generator(device=DEVICE).manual_seed(16)
    others = torch.randn(VOCAB, generator=g, device=DEVICE) * 2
    lse = torch.logsumexp(others.double(), 0).item()
    col = {"first": 0, "middle": VOCAB // 2, "last": VOCAB - 1}[column]
    values = others.repeat(len(MARGINS), 1)
    values[:, col] = torch.tensor([lse + m for m in MARGINS], device=DEVICE)
    logits = _logits(values, dtype, "aligned")
    ids = torch.full((len(MARGINS), 1), col, dtype=torch.int64, device=DEVICE)
    ours = _tilelang(logits, ids)
    theirs = _triton(logits, ids)
    ref, s = _assert_accurate(ours, theirs, logits, ids, f"margins {dtype} {column}")
    assert torch.equal(ids[:, 0], logits.float().argmax(dim=1))
    normal = s[:, 0] >= VOCAB * 2.0**-126
    assert normal.sum() >= 20
    relative = ((ours[:, 0].double() - ref[:, 0]) / ref[:, 0]).abs()
    assert (relative[normal] <= S_RELATIVE).all(), relative[normal].max().item()
    assert (ours[normal, 0] < 0).all()
    tiny = s[:, 0] < 2.0**-25
    assert tiny.any() and (theirs[tiny, 0] == 0).all()
    assert (ours[tiny & normal, 0] < 0).all()


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_rows_do_not_depend_on_the_batch_or_the_run(dtype):
    """Batch invariance and determinism: a row's bits are those it gets alone, and
    the same on every launch."""
    values = _values(96, VOCAB, "peaked", seed=96)
    values[48:] = _values(48, VOCAB, "random", seed=97)
    logits = _logits(values, dtype, "aligned")
    ids = _ids(values, 21, 96)
    whole = _tilelang(logits, ids)
    _assert_same(_tilelang(logits, ids), whole, "second launch")
    for r in (0, 1, 47, 48, 95):
        alone = _tilelang(logits[r : r + 1], ids[r : r + 1])
        _assert_same(alone, whole[r : r + 1], f"row {r} alone")
    _assert_same(_tilelang(logits[40:60], ids[40:60]), whole[40:60], "rows 40-59")


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("layout", ["aligned", "odd_stride"])
def test_masked_columns_are_minus_inf(dtype, layout):
    """The custom-ids path's mask in the same store: the unmasked kernel's bits
    where ``valid``, -inf elsewhere (torch's ``masked_fill`` after it before)."""
    values = _values(37, VOCAB, "special", seed=13)
    logits = _logits(values, dtype, layout)
    ids = _ids(values, 21, 13)
    g = torch.Generator(device=DEVICE).manual_seed(14)
    valid = torch.rand(ids.shape, generator=g, device=DEVICE) < 0.6
    valid[:, 0] = True
    expected = _tilelang(logits, ids).masked_fill(~valid, float("-inf"))
    ours = torch.full_like(expected, SENTINEL)
    tilelang_logprob.topk_log_softmax(logits, ids, ours, valid)
    _assert_same(ours, expected, "masked")


# --- Call sites ------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_compute_token_logprobs_launches_tilelang(dtype, monkeypatch):
    values = _values(5, VOCAB, "random", seed=11)
    logits = _logits(values, dtype, "aligned")
    ids = _ids(values, 21, 11)
    expected = _check(logits, ids, f"{dtype} token logprobs")
    calls = []
    launch = tilelang_logprob.topk_log_softmax

    def spy(*args):
        calls.append(args)
        launch(*args)

    # A launch of the Triton kernel would fail.
    monkeypatch.setattr(logprob, "_topk_log_softmax_kernel", None)
    monkeypatch.setattr(tilelang_logprob, "topk_log_softmax", spy)
    _assert_same(logprob.compute_token_logprobs(logits, ids), expected, "logprobs")
    # int32 ids are widened, as before.
    _assert_same(
        logprob.compute_token_logprobs(logits, ids.int()), expected, "int32 ids"
    )
    assert len(calls) == 2
    # No requested token, or no row: nothing to launch.
    empty = logprob.compute_token_logprobs(logits, ids[:, :0])
    assert empty.shape == (5, 0)
    none = logprob.compute_token_logprobs(logits[:0], ids[:0])
    assert none.shape == (0, 21)


@pytest.mark.parametrize("num_logprobs", [0, 5, 20])
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_compute_topk_scores(dtype, num_logprobs):
    """The fast path's scores: the sampled token, then the top ``num_logprobs``."""
    values = _values(6, VOCAB, "special", seed=num_logprobs)
    logits = _logits(values, dtype, "aligned")
    g = torch.Generator(device=DEVICE).manual_seed(num_logprobs)
    sampled = torch.randint(VOCAB, (6,), generator=g, device=DEVICE)
    result = logprob.compute_topk_scores(logits, num_logprobs, sampled)
    assert result.logprob_token_ids.shape == (6, 1 + num_logprobs)
    ids = result.logprob_token_ids.contiguous()
    _assert_same(
        result.logprobs,
        _check(logits, ids, f"topk scores {num_logprobs}"),
        f"topk scores {num_logprobs}",
    )


def test_cuda_graph_replay():
    """A captured launch replays on new values in the same tensors."""
    values = _values(16, VOCAB, "random", seed=3)
    logits = _logits(values, torch.float32, "aligned")
    ids = _ids(values, 21, 3)
    out = torch.full((16, 21), SENTINEL, device=DEVICE)
    tilelang_logprob.topk_log_softmax(logits, ids, out)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            tilelang_logprob.topk_log_softmax(logits, ids, out)
    torch.cuda.current_stream().wait_stream(side)
    fresh = _values(16, VOCAB, "special", seed=4)
    logits.copy_(fresh)
    ids.copy_(_ids(fresh, 21, 4))
    out.fill_(SENTINEL)
    graph.replay()
    _assert_same(out, _check(logits, ids, "graph replay"), "graph replay")


# --- Preparation --------------------------------------------------------------


def _declared(registry):
    from vllm.models.deepseek_v4_1.tilelang import prepare

    # JIT functions are not hashable: a call is its function's identity and key.
    return {
        (id(function), args) for function, args, _ in prepare.declared_calls(registry)
    }


@pytest.mark.parametrize("logits_mode", [False, True])
def test_runner_declares_the_launched_calls(logits_mode):
    """register_topk_scores_warmup declares the log-softmax of each logits dtype in
    both layouts (the stride is an argument), unless only logits are reported; the
    declared calls compile and serve both layouts."""
    from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry
    from vllm.models.deepseek_v4_1.tilelang import prepare

    pairs = [(torch.bfloat16, PADDED), (torch.float32, VOCAB), (torch.float32, PADDED)]
    registry = JitWarmupRegistry(None)
    with registry.activate():
        logprob.register_topk_scores_warmup(
            logits=pairs, vocab_size=VOCAB, logits_mode=logits_mode
        )
    expected = {
        (
            id(tilelang_logprob.topk_log_softmax_kernel),
            tilelang_logprob.topk_log_softmax_key(dtype, run, masked),
        )
        for dtype in (torch.bfloat16, torch.float32)
        for run in tls._runs(dtype)
        for masked in (False, True)
    }
    declared = _declared(registry)
    softmax = {
        call
        for call in declared
        if call[0] == id(tilelang_logprob.topk_log_softmax_kernel)
    }
    if logits_mode:
        assert not softmax
        return
    assert softmax == expected
    prepare.prepare_declared(registry)
    for dtype in (torch.bfloat16, torch.float32):
        values = _values(3, VOCAB, "random", seed=9)
        for layout in ("aligned", "odd_stride"):
            logits = _logits(values, dtype, layout)
            _check(logits, _ids(values, 21, 9), f"declared {dtype} {layout}")
