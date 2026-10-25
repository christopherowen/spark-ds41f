# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``tilelang_logprob.topk_log_softmax`` against the Triton kernel it replaces,
``logprob._topk_log_softmax_kernel`` launched as production launched it, bit for bit
(outputs compared as FP32 bit patterns): FP32 and bf16 logits; rows at 16-byte
addresses and strides (Triton's 16-byte runs), at an odd stride and at a base past
16 bytes (one element a thread); the DeepSeek V4.1 vocabulary, a tensor-parallel
shard and odd widths; 0 to 2100 requested tokens; rows of infinities, NaN, ties,
signed zeros, extreme and subnormal magnitudes, and exponentials that underflow to
subnormals. Then the call sites: ``compute_token_logprobs`` and
``compute_topk_scores`` launch the TileLang kernel only, a CUDA graph replays it,
and the runner declares the calls it launches."""

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
        f"Triton's {[theirs[tuple(i)].item() for i in first]}"
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


# --- The two launches ----------------------------------------------------------


def _triton(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
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


def _check(logits: torch.Tensor, ids: torch.Tensor, what: str) -> torch.Tensor:
    before = logits.clone()
    ours = _tilelang(logits, ids)
    theirs = _triton(logits, ids)
    _assert_same(ours, theirs, what)
    _assert_same(logits, before, f"{what}: logits")
    return ours


# --- Bits ---------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize(
    "vocab", [VOCAB, SHARD, 1, 7, 1023, 1025, 8193, 16391], ids=lambda v: f"v{v}"
)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_log_softmax_matches_triton(dtype, layout, vocab, case):
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


@pytest.mark.parametrize("vocab", [1024, 8192], ids=lambda v: f"v{v}")
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_layout_decides_the_bits(dtype, vocab):
    """The tests see the layout: on 16-byte rows the one-element layout adds the
    exponentials in another order, and some log probability differs from Triton's
    (one block: the guarded chunk; eight: one chunk of vector loads). Few blocks,
    so a block sum's last bits reach the row's."""
    values = _values(256, vocab, "random", seed=5)
    logits = _logits(values, dtype, "aligned")
    ids = _ids(values, 21, 5)
    theirs = _triton(logits, ids)
    _assert_same(_tilelang(logits, ids), theirs, "vector layout")
    other = torch.full_like(theirs, SENTINEL)
    tls._kernel(
        tilelang_logprob.topk_log_softmax_kernel,
        *tilelang_logprob.topk_log_softmax_key(dtype, 1),
    )(other, logits, ids, ids[:1].view(torch.bool))
    assert not torch.equal(_bits(other), _bits(theirs))


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
    expected = _triton(logits, ids)
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
    _assert_same(
        result.logprobs,
        _triton(logits, result.logprob_token_ids.contiguous()),
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
    _assert_same(out, _triton(logits, ids), "graph replay")


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
