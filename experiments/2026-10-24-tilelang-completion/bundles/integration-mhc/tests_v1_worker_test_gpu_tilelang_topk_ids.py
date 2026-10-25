# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``tilelang_topk_ids`` against ``torch.topk`` on CUDA (PyTorch 2.13), index for
index and in its order: FP32 and bf16 logits; one to 1024 rows (decode batches and a
prompt-logprobs chunk); k from 1 to 32; the DeepSeek V4.1 vocabulary, a quarter
shard and odd widths; rows at 16-byte addresses and strides (vector loads), at
strides past the row, at an odd stride and at a base past 16 bytes (one element at a
time); heavy ties (bf16-rounded logits, a few distinct values, equal rows, tie
groups straddling the k-th place, ascending rows), NaN of any sign and payload,
infinities, zeros of both signs and subnormals. Then the fast path's ``[sampled,
ids]`` rows and the custom-ids path's int32 ids, the call sites (TileLang up to
``TOPK_IDS_MAX``, torch.topk above), a CUDA graph replay and the runner's
declarations.

``_model`` is PyTorch's algorithm as read from its sources (see
``tilelang_topk_ids``): the k largest by radix key and lower column, laid out as
``gatherTopK`` writes them, then ``bitonicSortKVInPlace`` with ``GTOp``.
``test_model_matches_torch`` pins that reading on the tie-heavy rows: if PyTorch
ever orders differently, it fails beside the kernel tests and names the row."""

import math

import pytest
import torch

from vllm.platforms import current_platform

if not (
    current_platform.is_cuda() and current_platform.is_device_capability_family(120)
):
    pytest.skip("TileLang top-k ids target SM120/SM121", allow_module_level=True)

pytest.importorskip("tilelang")

from vllm.sampling_params import SamplingParams  # noqa: E402
from vllm.v1.worker.gpu.sample import logprob  # noqa: E402
from vllm.v1.worker.gpu.sample import tilelang_sampling as tls  # noqa: E402
from vllm.v1.worker.gpu.sample import tilelang_topk_ids as tki  # noqa: E402

DEVICE = torch.device("cuda")
VOCAB = 129280  # DeepSeek V4.1
SHARD = 32320  # a quarter of it
PADDED = 131072
DTYPES = (torch.float32, torch.bfloat16)
# "aligned": 16-byte rows, dense where the width allows; "padded": 16-byte rows with
# a stride 1024 past them; "odd_stride" and "shifted_base" (one element past a
# 16-byte base): one element at a time.
LAYOUTS = ("aligned", "padded", "odd_stride", "shifted_base")
KS = (1, 2, 5, 20, 31, 32)
CASES = (
    "random",
    "rounded",
    "ties",
    "equal",
    "straddle",
    "special",
    "ascending",
)
# Past each row: a NaN outranks everything, so a read past the row would win.
PAD = float("nan")


# --- Inputs -------------------------------------------------------------------


def _values(rows: int, vocab: int, case: str, seed: int) -> torch.Tensor:
    """FP32 values ([rows, vocab]). "random": normal, sd 4; "rounded": the same
    rounded to bf16 (ties among the largest); "ties": integers -3..3; "equal": one
    value a row; "straddle": ``r % 8`` distinct values above a group of 96 equal
    ones at random columns (the k-th place falls in the group), the rest below;
    "special": the kinds of ``_special``; "ascending": rising columns (every
    element enters a warp's list), odd rows falling."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    x = torch.randn(rows, vocab, generator=g, device=DEVICE) * 4
    if case == "rounded":
        x = x.bfloat16().float()
    elif case == "ties":
        x = torch.randint(-3, 4, (rows, vocab), generator=g, device=DEVICE).float()
    elif case == "equal":
        x = (torch.arange(rows, device=DEVICE).float()[:, None] * 0.25 - 1).expand(
            rows, vocab
        )
        x = x.contiguous()
    elif case == "straddle":
        x = x / 4 - 10
        for r in range(rows):
            top = r % 8
            pick = torch.randperm(vocab, generator=g, device=DEVICE)
            above = pick[: min(top, vocab)]
            x[r, above] = 5.0 + torch.arange(above.numel(), device=DEVICE).float()
            x[r, pick[min(top, vocab) : top + 96]] = 2.0
    elif case == "special":
        for r in range(rows):
            _special(x[r], r % 12, g)
    elif case == "ascending":
        x = torch.arange(vocab, device=DEVICE).float().expand(rows, vocab) * 1.0e-3
        x = x.contiguous()
        x[1::2] = x[1::2].flip(-1)
    return x


def _special(row: torch.Tensor, kind: int, g: torch.Generator) -> None:
    """Row kinds: 0 five NaN; 1 forty NaN; 2 all NaN; 3 three +inf and two NaN over
    -inf; 4 all -inf; 5 zeros of both signs; 6 three positives, ten +0 and ten -0
    over negatives (the k-th place among the zeros); 7 subnormals and zeros; 8
    magnitudes up to 3e38 with infinities; 9 NaN of both signs and several payloads
    (``_nan_payloads``); 10 four positives over -0; 11 three finite values over
    -inf."""
    vocab = row.numel()
    pick = torch.randperm(vocab, generator=g, device=DEVICE)

    def at(start: int, count: int) -> torch.Tensor:
        return pick[min(start, vocab) : min(start + count, vocab)]

    if kind == 0:
        row[at(0, 5)] = float("nan")
    elif kind == 1:
        row[at(0, 40)] = float("nan")
    elif kind == 2:
        row.fill_(float("nan"))
    elif kind == 3:
        row.fill_(float("-inf"))
        row[at(0, 3)] = float("inf")
        row[at(3, 2)] = float("nan")
    elif kind == 4:
        row.fill_(float("-inf"))
    elif kind == 5:
        row.zero_()
        row[1::2] = -0.0
    elif kind == 6:
        row.copy_(-row.abs() - 1)
        row[at(0, 3)] = torch.tensor([3.0, 1.0, 2.0], device=DEVICE)[: at(0, 3).numel()]
        row[at(3, 10)] = 0.0
        row[at(13, 10)] = -0.0
    elif kind == 7:
        row.mul_(1.0e-40)
        row[at(0, 20)] = 0.0
        row[at(20, 20)] = -0.0
    elif kind == 8:
        unit = torch.rand(vocab, generator=g, device=DEVICE)
        row.copy_((unit * 2 - 1) * 3.0e38)
        row[at(0, 2)] = float("inf")
        row[at(2, 2)] = float("-inf")
    elif kind == 10:
        row.fill_(-0.0)
        row[at(0, 4)] = torch.tensor([4.0, 1.0, 3.0, 2.0], device=DEVICE)[
            : at(0, 4).numel()
        ]
    elif kind == 11:
        row.fill_(float("-inf"))
        row[at(0, 3)] = torch.tensor([-1.0, 7.0, -1.0], device=DEVICE)[
            : at(0, 3).numel()
        ]


# NaN bit patterns of each width: quiet and signalling, both signs.
_NAN_BITS = {
    4: (0x7FC00000, 0x7F800001, -0x00400000, -0x007FFFFF),
    2: (0x7FC0, 0x7F81, -0x0040, -0x007F),
}


def _nan_payloads(logits: torch.Tensor, case: str) -> None:
    """Rows of "special" kind 9: eight NaNs of each pattern at spread columns."""
    if case != "special":
        return
    rows, vocab = logits.shape
    ints = {4: torch.int32, 2: torch.int16}[logits.element_size()]
    bits = logits.view(ints)
    for r in range(9, rows, 12):
        for i, pattern in enumerate(_NAN_BITS[logits.element_size()]):
            for j in range(8):
                bits[r, (vocab * (4 * j + i)) // 32] = pattern


def _logits(
    values: torch.Tensor, dtype: torch.dtype, layout: str, case: str = ""
) -> torch.Tensor:
    """``values`` in ``dtype`` rows of ``layout``, NaN past each row."""
    rows, vocab = values.shape
    aligned = -(-vocab // 16) * 16
    stride = {
        "aligned": aligned,
        "padded": aligned + 1024,
        "odd_stride": aligned + 1,
        "shifted_base": aligned,
    }[layout]
    flat = torch.full((rows * stride + 1,), PAD, dtype=dtype, device=DEVICE)
    start = 1 if layout == "shifted_base" else 0
    logits = flat[start : start + rows * stride].view(rows, stride)[:, :vocab]
    logits.copy_(values.to(dtype))
    _nan_payloads(logits, case)
    vector = layout in ("aligned", "padded")
    assert tls._triton_run(logits) == (16 // logits.element_size() if vector else 1)
    return logits


def _sampled(rows: int, vocab: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device=DEVICE).manual_seed(seed + 1)
    return torch.randint(vocab, (rows,), generator=g, device=DEVICE)


# --- PyTorch's algorithm --------------------------------------------------------


def _keys(logits: torch.Tensor) -> torch.Tensor:
    """TopKTypeConfig<T>::convert of the values (int64 holding uint32 keys), on the
    host: the FP32 key of the widened value orders as the 16-bit key."""
    x = logits.float().cpu()
    bits = x.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    keys = bits ^ torch.where(bits >= 0x80000000, 0xFFFFFFFF, 0x80000000)
    return torch.where(x.isnan(), torch.full_like(keys, 0xFFFFFFFF), keys)


def _gt(a: float, b: float) -> bool:
    """GTOp<scalar_t, true> (SortingCommon.cuh)."""
    return (math.isnan(a) and not math.isnan(b)) or a > b


def _bitonic(values: list[float], cols: list[int]) -> list[int]:
    """bitonicSortKVInPlace (SortUtils.cuh) as SmallBitonicSort launches it: 32
    slots, two for each of 16 threads, the slots past ``len(cols)`` invalid,
    descending by GTOp."""
    k = len(cols)
    keys = values + [0.0] * (32 - k)
    vals = cols + [0] * (32 - k)
    valid = [True] * k + [False] * (32 - k)

    def swap(t: int, stride: int, direction: bool) -> None:
        a = 2 * t - (t & (stride - 1))
        b = a + stride
        exchange = (_gt(keys[a], keys[b]) and valid[a]) or not valid[b]
        if exchange == direction:
            keys[a], keys[b] = keys[b], keys[a]
            vals[a], vals[b] = vals[b], vals[a]
            valid[a], valid[b] = valid[b], valid[a]

    size = 2
    while size < 32:
        stride = size // 2
        while stride:
            for t in range(16):
                swap(t, stride, (t & (size // 2)) != 0)
            stride //= 2
        size *= 2
    stride = 16
    while stride:
        for t in range(16):
            swap(t, stride, False)
        stride //= 2
    return vals[:k]


def _model(logits: torch.Tensor, k: int) -> torch.Tensor:
    """torch.topk(logits, k).indices (k <= 32) by PyTorch's algorithm, on the host:
    the k largest by key then lower column; gatherTopK's layout (those above the
    k-th key in column order, then those equal to it in column order); the bitonic
    sort unless k is 1."""
    keys = _keys(logits)
    values = logits.float().cpu()
    out = []
    for r in range(keys.shape[0]):
        chosen = torch.sort(keys[r], descending=True, stable=True).indices[:k]
        chosen_keys = keys[r, chosen].tolist()
        chosen = chosen.tolist()
        kth = chosen_keys[-1]
        above = sorted(c for c, key in zip(chosen, chosen_keys) if key != kth)
        equal = [c for c, key in zip(chosen, chosen_keys) if key == kth]
        gathered = above + equal
        if k > 1:
            gathered = _bitonic([values[r, c].item() for c in gathered], gathered)
        out.append(gathered)
    return torch.tensor(out, dtype=torch.int64)


# --- Checks -----------------------------------------------------------------------


def _bits(x: torch.Tensor) -> torch.Tensor:
    return x.view({4: torch.int32, 2: torch.int16}[x.element_size()])


def _assert_ids(
    ours: torch.Tensor, theirs: torch.Tensor, logits: torch.Tensor, what: str
) -> None:
    ours, theirs = ours.cpu(), theirs.cpu()
    assert ours.shape == theirs.shape, f"{what}: {ours.shape} vs {theirs.shape}"
    if torch.equal(ours, theirs):
        return
    rows = (ours != theirs).any(dim=1).nonzero().flatten().tolist()
    r = rows[0]
    last = logits.shape[1] - 1

    def values(ids: torch.Tensor) -> list[float]:
        return logits[r, ids.clamp(0, last).to(logits.device)].float().tolist()

    raise AssertionError(
        f"{what}: {len(rows)} of {ours.shape[0]} rows differ, first row {r}: "
        f"ours {ours[r].tolist()}, torch's {theirs[r].tolist()}, values ours "
        f"{values(ours[r])}, torch's {values(theirs[r])}"
    )


def _check(logits: torch.Tensor, k: int, what: str) -> None:
    """Both paths against torch.topk: the fast path's [sampled, ids] (int64) and the
    custom-ids path's ids (int32); the logits untouched."""
    before = _bits(logits).clone()
    theirs = torch.topk(logits, k, dim=-1).indices
    sampled = _sampled(logits.shape[0], logits.shape[1], k)
    rows = tki.logprob_token_ids(logits, k, sampled)
    assert rows.dtype == torch.int64 and rows.is_contiguous()
    assert torch.equal(rows[:, 0], sampled), f"{what}: sampled column"
    _assert_ids(rows[:, 1:], theirs, logits, f"{what} k={k} int64")
    ids = tki.topk_token_ids(logits, k)
    assert ids.dtype == torch.int32
    _assert_ids(ids, theirs.to(torch.int32), logits, f"{what} k={k} int32")
    assert torch.equal(_bits(logits), before), f"{what}: logits changed"


# --- Indices ---------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize(
    "vocab", [VOCAB, SHARD, 1, 7, 33, 16385, 50001], ids=lambda v: f"v{v}"
)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_ids_match_torch(dtype, layout, vocab, case):
    """Twelve rows (every "special" kind), each k up to the width."""
    values = _values(12, vocab, case, seed=vocab + len(case))
    logits = _logits(values, dtype, layout, case)
    for k in KS:
        if k <= vocab:
            _check(logits, k, f"{dtype} {layout} v{vocab} {case}")


@pytest.mark.parametrize("case", ("straddle", "special", "equal", "ties", "rounded"))
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_every_k(dtype, case):
    """Every k from 1 to 32 where the k-th place falls inside tie groups."""
    values = _values(24, SHARD, case, seed=31 + len(case))
    logits = _logits(values, dtype, "aligned", case)
    for k in range(1, tki.TOPK_IDS_MAX + 1):
        _check(logits, k, f"{dtype} {case}")


@pytest.mark.parametrize("case", ("random", "rounded", "ties"))
@pytest.mark.parametrize("rows", [1, 6, 16, 96, 1024])
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_batch_sizes(dtype, rows, case):
    """Decode batches and a prompt-logprobs chunk (compute_prompt_logprobs_with_
    chunking's 1024 rows) at the LM head's padded stride."""
    values = _values(rows, VOCAB, case, seed=rows)
    flat = torch.full((rows, PADDED), PAD, dtype=dtype, device=DEVICE)
    logits = flat[:, :VOCAB]
    logits.copy_(values.to(dtype))
    for k in (1, 5, 20, 32):
        _check(logits, k, f"{dtype} {rows} rows {case}")


@pytest.mark.parametrize("case", ("straddle", "special", "equal", "ascending"))
@pytest.mark.parametrize("dtype", (*DTYPES, torch.float16), ids=str)
def test_model_matches_torch(dtype, case):
    """PyTorch's algorithm as ``_model`` reads it, against torch.topk itself (the
    kernel's tests stand on this reading), and the kernel against both."""
    values = _values(12, SHARD, case, seed=7 + len(case))
    logits = _logits(values, dtype, "aligned", case)
    for k in KS:
        theirs = torch.topk(logits, k, dim=-1).indices
        _assert_ids(_model(logits, k), theirs, logits, f"model {dtype} {case} k={k}")
        _check(logits, k, f"{dtype} {case}")


def test_ties_leave_in_the_network_order():
    """The order the tests pin is not a stable one: eight equal values at columns
    0..7 come out as the bitonic network leaves them."""
    logits = torch.full((1, 64), -1.0, device=DEVICE)
    logits[0, :8] = 2.0
    expected = [7, 6, 4, 5, 1, 0, 2, 3]
    assert torch.topk(logits, 8, dim=-1).indices[0].tolist() == expected
    assert tki.topk_token_ids(logits, 8)[0].tolist() == expected


# --- Bounds and layouts -------------------------------------------------------------


def test_bounds():
    logits = torch.randn(3, 40, device=DEVICE)
    sampled = torch.zeros(3, dtype=torch.int64, device=DEVICE)
    for k in (0, tki.TOPK_IDS_MAX + 1):
        with pytest.raises(ValueError):
            tki.logprob_token_ids(logits, k, sampled)
        with pytest.raises(ValueError):
            tki.topk_token_ids(logits, k)
    narrow = logits[:, :4]
    with pytest.raises(RuntimeError, match="out of range"):
        tki.topk_token_ids(narrow, 5)
    # No row: nothing to launch.
    assert tki.logprob_token_ids(logits[:0], 5, sampled[:0]).shape == (0, 6)
    assert tki.topk_token_ids(logits[:0], 5).shape == (0, 5)


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_sampled_column(dtype):
    """The fast path's rows are torch.cat((sampled[:, None], indices), 1), int64,
    for int64 and int32 sampled tokens."""
    values = _values(9, VOCAB, "rounded", seed=3)
    logits = _logits(values, dtype, "aligned")
    sampled = _sampled(9, VOCAB, 3)
    expected = torch.cat(
        (sampled.unsqueeze(-1), torch.topk(logits, 20, dim=-1).indices), dim=1
    )
    for tokens in (sampled, sampled.int()):
        rows = tki.logprob_token_ids(logits, 20, tokens)
        assert rows.dtype == expected.dtype
        _assert_ids(rows, expected, logits, f"{dtype} sampled {tokens.dtype}")


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_strided_columns(dtype):
    """Logits whose columns are not adjacent are refused (every logits row the
    samplers pass has a unit column stride; no copy path)."""
    wide = torch.zeros(4, 4099, 2, dtype=dtype, device=DEVICE)
    logits = wide[..., 0]
    assert logits.stride(1) == 2
    with pytest.raises(ValueError, match="unit column stride"):
        tki.topk_token_ids(logits, 20)


# --- Call sites ----------------------------------------------------------------------


def _no_topk(*args, **kwargs):
    raise AssertionError("torch.topk was called below TOPK_IDS_MAX")


@pytest.mark.parametrize("logits_mode", [False, True])
@pytest.mark.parametrize("num_logprobs", [0, 1, 5, 20, 32, 33, 64])
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_compute_topk_scores_fast_path(dtype, num_logprobs, logits_mode, monkeypatch):
    """The fast path's ids: the kernel up to TOPK_IDS_MAX (torch.topk is not
    called); above it the count is refused (--max-logprobs bounds it at startup)."""
    values = _values(6, VOCAB, "special", seed=num_logprobs)
    logits = _logits(values, dtype, "aligned", "special")
    sampled = _sampled(6, VOCAB, num_logprobs)
    if num_logprobs > tki.TOPK_IDS_MAX:
        with pytest.raises(ValueError, match="max-logprobs"):
            logprob.compute_topk_scores(
                logits, num_logprobs, sampled, logits_mode=logits_mode
            )
        return
    expected = sampled.unsqueeze(-1)
    if num_logprobs:
        expected = torch.cat(
            (expected, torch.topk(logits, num_logprobs, dim=-1).indices), dim=1
        )
    monkeypatch.setattr(torch, "topk", _no_topk)
    result = logprob.compute_topk_scores(
        logits, num_logprobs, sampled, logits_mode=logits_mode
    )
    monkeypatch.undo()
    _assert_ids(result.logprob_token_ids, expected, logits, f"fast {num_logprobs}")
    assert result.logprobs.shape == expected.shape


@pytest.mark.parametrize("num_logprobs", [0, 3, 20, 33])
def test_compute_topk_scores_custom_ids(num_logprobs, monkeypatch):
    """With a request's logprob_token_ids, the other rows take the top-k from the
    int32 ids (the kernel up to TOPK_IDS_MAX, above it refused), the request's
    rows its own ids; an unused column is id 0 and -inf."""
    custom = [5, 129279, 2]
    state = logprob.LogprobTokenIdsState(8, DEVICE)
    for req in range(8):
        params = SamplingParams(logprob_token_ids=custom if req == 3 else None)
        state.add_request(req, params)
    state.apply_staged_writes()
    rows = 7
    mapping = torch.tensor([0, 3, 1, 3, 7, 2, 6], dtype=torch.int64, device=DEVICE)
    values = _values(rows, VOCAB, "straddle", seed=num_logprobs)
    logits = _logits(values, torch.float32, "aligned")
    sampled = _sampled(rows, VOCAB, num_logprobs)
    top = torch.topk(logits, max(num_logprobs, 1), dim=-1).indices
    if num_logprobs > tki.TOPK_IDS_MAX:
        with pytest.raises(ValueError, match="max-logprobs"):
            logprob.compute_topk_scores(
                logits,
                num_logprobs,
                sampled,
                logprob_token_ids_state=state,
                expanded_idx_mapping=mapping,
                max_per_req_token_ids=len(custom),
            )
        return
    monkeypatch.setattr(torch, "topk", _no_topk)
    result = logprob.compute_topk_scores(
        logits,
        num_logprobs,
        sampled,
        logprob_token_ids_state=state,
        expanded_idx_mapping=mapping,
        max_per_req_token_ids=len(custom),
    )
    monkeypatch.undo()
    ids = result.logprob_token_ids.cpu()
    width = 1 + max(num_logprobs, len(custom))
    assert ids.shape == (rows, width)
    scores = result.logprobs.cpu()
    for r, req in enumerate(mapping.tolist()):
        assert ids[r, 0].item() == sampled[r].item()
        wanted = custom if req == 3 else top[r, :num_logprobs].tolist()
        assert ids[r, 1 : 1 + len(wanted)].tolist() == wanted, (r, req)
        # The straddle rows are finite: a masked (invalid) column is -inf.
        assert torch.isfinite(scores[r, : 1 + len(wanted)]).all(), (r, req)
        assert (scores[r, 1 + len(wanted) :] == float("-inf")).all(), (r, req)
        assert (ids[r, 1 + len(wanted) :] == 0).all(), (r, req)


def test_cuda_graph_replay():
    """A captured launch replays on new values in the same tensors."""
    values = _values(16, VOCAB, "random", seed=3)
    logits = _logits(values, torch.float32, "aligned")
    sampled = _sampled(16, VOCAB, 3)
    tki.logprob_token_ids(logits, 20, sampled)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = tki.logprob_token_ids(logits, 20, sampled)
    torch.cuda.current_stream().wait_stream(side)
    fresh = _values(16, VOCAB, "straddle", seed=4)
    logits.copy_(fresh)
    sampled.copy_(_sampled(16, VOCAB, 4))
    graph.replay()
    torch.cuda.synchronize()
    expected = torch.cat(
        (sampled.unsqueeze(-1), torch.topk(logits, 20, dim=-1).indices), dim=1
    )
    _assert_ids(out, expected, logits, "graph replay")


# --- Preparation --------------------------------------------------------------


def test_runner_declares_the_launched_calls():
    """register_topk_scores_warmup declares both ids layouts of each logits dtype in
    both load layouts (the stride and k are arguments); the declared calls compile
    and serve every layout."""
    from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry
    from vllm.models.deepseek_v4_1.tilelang import prepare

    pairs = [(torch.bfloat16, PADDED), (torch.float32, VOCAB), (torch.float32, PADDED)]
    registry = JitWarmupRegistry(None)
    with registry.activate():
        logprob.register_topk_scores_warmup(
            logits=pairs, vocab_size=VOCAB, logits_mode=True
        )
    declared = {
        (id(function), args)
        for function, args, _ in prepare.declared_calls(registry)
        if function is tki.topk_ids_kernel
    }
    assert declared == {
        (id(tki.topk_ids_kernel), tki.topk_ids_key(dtype, ids_dtype, run))
        for dtype in (torch.bfloat16, torch.float32)
        for ids_dtype in (torch.int64, torch.int32)
        for run in tls._runs(dtype)
    }
    prepare.prepare_declared(registry)
    for dtype in (torch.bfloat16, torch.float32):
        values = _values(3, VOCAB, "rounded", seed=9)
        for layout in ("aligned", "odd_stride"):
            logits = _logits(values, dtype, layout)
            _check(logits, 20, f"declared {dtype} {layout}")
