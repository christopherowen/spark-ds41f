# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The TileLang rejection sampler (``spec_decode.tilelang_rejection``) against the
Triton kernels it replaces, bit for bit: each kernel's outputs (floats compared as
bit patterns, including the elements a launch leaves alone) and
``rejection_sample``'s tokens and counts against the Triton launches', on batches of
requests with 0 to 5 drafts, -1 placeholders, mixed temperatures and per-request
seeds, and logits with ties, infinities and NaN."""

import itertools

import pytest
import torch

triton = pytest.importorskip("triton")
pytest.importorskip("tilelang")

from vllm.v1.worker.gpu.sample import tilelang_sampling as tls  # noqa: E402
from vllm.v1.worker.gpu.spec_decode import rejection_sampler_utils as rs  # noqa: E402
from vllm.v1.worker.gpu.spec_decode import tilelang_rejection as tlr  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)

DEVICE = torch.device("cuda")
VOCAB = 129280  # DeepSeek V4.1
STEPS = 5  # DSpark's num_speculative_steps
SLOTS = 160


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


# --- Batches ------------------------------------------------------------------


def _counts(rows: int) -> list[int]:
    """Logits rows of the requests covering ``rows`` rows: 1 to STEPS + 1 each (0 to
    5 drafts, adaptive verification's shorter blocks included)."""
    pattern = itertools.cycle([6, 1, 4, 6, 2, 5, 3, 6])
    counts: list[int] = []
    while sum(counts) < rows:
        counts.append(min(next(pattern), rows - sum(counts)))
    return counts


# The runner's index dtypes: int32 drafts and local positions, int64 indices and
# positions; and the other of each.
INDICES = {
    "runner": (torch.int32, torch.int64, torch.int64, torch.int32),
    "other": (torch.int64, torch.int32, torch.int32, torch.int64),
}


def _batch(counts, temps, seed, vocab, indices="runner"):
    """A verification batch: requests of ``counts`` logits rows (one more than their
    drafts) at random states of SLOTS, the request's last token then its drafts as
    the rows' inputs, some trailing and one inner -1 placeholder, per-state
    temperatures (``temps``) and seeds, and positions (one with a high word when they
    are int64)."""
    drafts_dtype, idx_dtype, pos_dtype, local_dtype = INDICES[indices]
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    reqs = len(counts)
    cu = torch.tensor(
        [0, *itertools.accumulate(counts)], dtype=torch.int32, device=DEVICE
    )
    rows = int(cu[-1])
    states = torch.randperm(SLOTS, generator=g, device=DEVICE)[:reqs]
    sizes = torch.tensor(counts, device=DEVICE)
    drafts = torch.randint(0, vocab, (rows,), generator=g, device=DEVICE)
    for r, count in enumerate(counts):
        start = int(cu[r])
        if r % 4 == 1 and count > 2:
            drafts[start + count - 1] = -1
        if r % 4 == 3 and count > 3:
            drafts[start + count - 2 : start + count] = -1
        if r % 9 == 5 and count > 4:
            drafts[start + 2] = -1
    values = {
        "sampled": [1.0, 0.6, 1.3, 1e-3, 37.0],
        "mixed": [0.0, 1.0, 0.6, 0.0, 1.3, 37.0],
    }[temps]
    temperature = torch.tensor(
        [values[i % len(values)] for i in range(SLOTS)], device=DEVICE
    )
    seeds = torch.randint(-(2**63), 2**63 - 1, (SLOTS,), generator=g, device=DEVICE)
    seeds[states[0]] = 0
    pos = torch.randint(0, 1 << 20, (rows,), generator=g, device=DEVICE)
    if pos_dtype == torch.int64:
        pos[0] = (1 << 33) + 5
    return {
        "cu": cu,
        "idx_mapping": states.to(idx_dtype),
        "expanded": states.repeat_interleave(sizes).to(idx_dtype),
        "local": torch.cat([torch.arange(c, device=DEVICE) for c in counts]).to(
            local_dtype
        ),
        "drafts": drafts.to(drafts_dtype),
        "temperature": temperature,
        "seeds": seeds,
        "pos": pos.to(pos_dtype),
    }


CASES = ("peaked", "ties", "special")
# Row strides: a multiple of 16 (Triton lays the block out in 16-byte runs) or odd
# (one element a thread).
STRIDES = ("aligned", "odd")


def _stride(vocab: int, stride: str) -> int:
    return (vocab + 64 + 15) // 16 * 16 if stride == "aligned" else vocab + 3


def _logits(batch, vocab, dtype, case, seed, stride="aligned"):
    """The batch's logits rows, ``vocab`` columns of rows ``_stride`` apart.
    "peaked": random, each row's next input lifted by a margin that varies by row,
    so verification accepts some drafts and rejects others; "ties": few distinct
    values, and rows all equal (one of zeros of both signs); "special": NaN, +inf
    and -inf scattered, a row all -inf but one, a row all NaN, and NaN before
    +inf."""
    rows = batch["cu"][-1].item()
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    x = torch.randn(rows, _stride(vocab, stride), generator=g, device=DEVICE) * 4
    if case == "peaked" and rows > 1:
        nxt = batch["drafts"][1:].long()
        r = torch.arange(rows - 1, device=DEVICE)
        lifted = nxt >= 0
        margin = torch.tensor([0.0, 3.0, 6.0, 9.0, 14.0, 40.0], device=DEVICE)
        x[r[lifted], nxt[lifted]] += margin[r[lifted] % 6]
    elif case == "ties":
        x = torch.round(x / 4)
        x[::5] = 1.0
        if rows > 3:
            x[3] = 0.0
            x[3, 1::2] = -0.0
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
            x[2, : min(vocab, 9000)] = float("nan")
            x[2, 0] = float("inf")
    return x.to(dtype)[:, :vocab]


# --- The Triton launches (rejection_sample's, for this configuration) --------


def _stats_buffers(rows, vocab):
    blocks = triton.cdiv(vocab, tlr.STATS_BLOCK)
    return (
        torch.full((rows, blocks), -7, dtype=torch.int64, device=DEVICE),
        torch.full((rows, blocks), -7.0, device=DEVICE),
        torch.full((rows, blocks), -7.0, device=DEVICE),
    )


def _triton_local_stats(logits, batch):
    rows, vocab = logits.shape
    stats = _stats_buffers(rows, vocab)
    unused = torch.empty_like(stats[1])
    rs._compute_local_logits_stats_kernel[(rows, stats[1].shape[1])](
        stats[0],
        stats[0].stride(0),
        stats[1],
        stats[1].stride(0),
        stats[2],
        stats[2].stride(0),
        unused,
        unused.stride(0),
        unused,
        unused.stride(0),
        logits,
        logits.stride(0),
        None,
        0,
        0,
        batch["expanded"],
        batch["local"],
        batch["temperature"],
        vocab,
        STEPS,
        BLOCK_SIZE=tlr.STATS_BLOCK,
        HAS_DRAFT_LOGITS=False,
    )
    return stats


def _verify_buffers(reqs, rows):
    return (
        torch.full((rows,), -7.0, device=DEVICE),
        torch.full((reqs, STEPS + 1), -7, dtype=torch.int64, device=DEVICE),
        torch.full((reqs,), -7, dtype=torch.int32, device=DEVICE),
        torch.full((reqs,), -7.0, device=DEVICE),
        torch.full((reqs,), -7.0, device=DEVICE),
    )


def _triton_verify(logits, stats, batch):
    """_compute_cumulative_log_p_kernel, then _rejection_kernel: the cumulative log
    ratios, sampled tokens, counts and rejected rows' log-sum-exps."""
    reqs = batch["cu"].shape[0] - 1
    blocks = stats[1].shape[1]
    padded = triton.next_power_of_2(blocks)
    cumulative, sampled, num_sampled, target_lse, draft_lse = _verify_buffers(
        reqs, logits.shape[0]
    )
    unused = torch.empty_like(stats[1])
    rs._compute_cumulative_log_p_kernel[(reqs,)](
        cumulative,
        logits,
        logits.stride(0),
        stats[1],
        stats[1].stride(0),
        stats[2],
        stats[2].stride(0),
        batch["drafts"],
        None,
        0,
        0,
        unused,
        unused.stride(0),
        unused,
        unused.stride(0),
        batch["cu"],
        batch["idx_mapping"],
        batch["temperature"],
        blocks,
        PADDED_VOCAB_NUM_BLOCKS=padded,
        HAS_DRAFT_LOGITS=False,
        num_warps=1,
    )
    rs._rejection_kernel[(reqs,)](
        sampled,
        sampled.stride(0),
        num_sampled,
        target_lse,
        draft_lse,
        logits,
        logits.stride(0),
        stats[0],
        stats[0].stride(0),
        stats[1],
        stats[1].stride(0),
        stats[2],
        stats[2].stride(0),
        batch["drafts"],
        None,
        0,
        0,
        unused,
        unused.stride(0),
        unused,
        unused.stride(0),
        batch["cu"],
        batch["idx_mapping"],
        batch["temperature"],
        batch["seeds"],
        batch["pos"],
        None,
        cumulative,
        None,
        0,
        blocks,
        target_lse,
        target_lse,
        0,
        PADDED_VOCAB_NUM_BLOCKS=padded,
        HAS_DRAFT_LOGITS=False,
        SYNTHETIC_MODE=False,
        USE_BLOCK_VERIFICATION=True,
        VP_DRAFT=False,
        num_warps=1,
    )
    return cumulative, sampled, num_sampled, target_lse, draft_lse


def _resample_buffers(reqs, vocab):
    blocks = triton.cdiv(vocab, tlr.RESAMPLE_BLOCK)
    return (
        torch.full((reqs, blocks), -7, dtype=torch.int64, device=DEVICE),
        torch.full((reqs, blocks), -7.0, device=DEVICE),
    )


def _triton_resample(logits, num_sampled, batch, verified):
    cumulative, _, _, target_lse, draft_lse = verified
    reqs = num_sampled.shape[0]
    vocab = logits.shape[1]
    ids, values = _resample_buffers(reqs, vocab)
    rs._resample_kernel[(reqs, ids.shape[1])](
        ids,
        ids.stride(0),
        values,
        values.stride(0),
        logits,
        logits.stride(0),
        target_lse,
        None,
        0,
        0,
        draft_lse,
        num_sampled,
        batch["cu"],
        batch["expanded"],
        batch["drafts"],
        batch["temperature"],
        batch["seeds"],
        batch["pos"],
        cumulative,
        None,
        0,
        None,
        0,
        0,
        vocab,
        BLOCK_SIZE=tlr.RESAMPLE_BLOCK,
        HAS_DRAFT_LOGITS=False,
        USE_FP64=False,
        USE_BLOCK_VERIFICATION=True,
        CONTEXT_WIDTH=1,
        WATERMARK=False,
    )
    return ids, values


def _triton_rejection_sample(logits, batch):
    """rejection_sample's Triton launches for one-hot drafts and block
    verification, the insertion included."""
    stats = _triton_local_stats(logits, batch)
    verified = _triton_verify(logits, stats, batch)
    _, sampled, num_sampled, _, _ = verified
    ids, values = _triton_resample(logits, num_sampled, batch, verified)
    rs._insert_resampled_kernel[(num_sampled.shape[0],)](
        sampled,
        sampled.stride(0),
        num_sampled,
        ids,
        ids.stride(0),
        values,
        values.stride(0),
        ids.shape[1],
        batch["cu"],
        batch["expanded"],
        batch["temperature"],
        PADDED_RESAMPLE_NUM_BLOCKS=triton.next_power_of_2(ids.shape[1]),
    )
    return sampled, num_sampled


def _rejection_sample(logits, batch):
    return rs.rejection_sample(
        logits,
        None,
        batch["drafts"],
        batch["cu"],
        batch["pos"],
        batch["idx_mapping"],
        batch["expanded"],
        batch["local"],
        batch["temperature"],
        batch["seeds"],
        STEPS,
        use_block_verification=True,
    )


# --- Each kernel --------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("vocab", [VOCAB, 1000, 8193])
@pytest.mark.parametrize("stride", STRIDES)
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("rows", [1, 16, 96])
def test_local_stats(dtype, vocab, stride, case, rows):
    """The block statistics of greedy and sampled rows; bonus rows left alone."""
    batch = _batch(_counts(rows), "mixed", rows, vocab)
    logits = _logits(batch, vocab, dtype, case, rows, stride)
    theirs = _triton_local_stats(logits, batch)
    ours = _stats_buffers(rows, vocab)
    tlr.compute_local_stats(
        *ours,
        logits,
        batch["expanded"],
        batch["local"],
        batch["temperature"],
        STEPS,
    )
    for name, a, b in zip(("argmax", "maximum", "sum of exponentials"), ours, theirs):
        _assert_same(a, b, name)


@pytest.mark.parametrize("indices", INDICES)
def test_local_stats_index_dtypes(indices):
    batch = _batch(_counts(37), "mixed", 3, VOCAB, indices)
    logits = _logits(batch, VOCAB, torch.float32, "special", 3)
    theirs = _triton_local_stats(logits, batch)
    ours = _stats_buffers(logits.shape[0], VOCAB)
    tlr.compute_local_stats(
        *ours,
        logits,
        batch["expanded"],
        batch["local"],
        batch["temperature"],
        STEPS,
    )
    for name, a, b in zip(("argmax", "maximum", "sum of exponentials"), ours, theirs):
        _assert_same(a, b, name)


def _verified(logits, batch):
    """Both implementations' verification outputs from Triton's block statistics."""
    stats = _triton_local_stats(logits, batch)
    theirs = _triton_verify(logits, stats, batch)
    ours = _verify_buffers(batch["cu"].shape[0] - 1, logits.shape[0])
    cumulative, sampled, num_sampled, target_lse, draft_lse = ours
    tlr.verify(
        sampled,
        num_sampled,
        target_lse,
        draft_lse,
        cumulative,
        logits,
        *stats,
        batch["drafts"],
        batch["cu"],
        batch["idx_mapping"],
        batch["temperature"],
        batch["seeds"],
        batch["pos"],
        STEPS,
    )
    return ours, theirs


VERIFIED = ("cumulative log ratio", "sampled", "num_sampled", "target lse", "draft lse")


# Vocabularies of 16 blocks (DeepSeek V4.1's), of 9, 3 and 2 (padded to 16, 4 and 2)
# and of one block, which Triton reduces without a shuffle.
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("vocab", [VOCAB, 70000, 20000, 16000, 2])
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("temps", ["sampled", "mixed"])
@pytest.mark.parametrize("indices", INDICES)
def test_verify(dtype, vocab, case, temps, indices):
    """The cumulative ratio and rejection kernels' outputs; with peaked logits, the
    batch accepts some drafts and rejects others."""
    batch = _batch(_counts(96), temps, 7, vocab, indices)
    logits = _logits(batch, vocab, dtype, case, 7)
    ours, theirs = _verified(logits, batch)
    for name, a, b in zip(VERIFIED, ours, theirs):
        _assert_same(a, b, name)
    if case == "peaked" and vocab > 2:
        drafts = (batch["cu"][1:] - batch["cu"][:-1] - 1).long()
        counts = theirs[2].long()
        assert (counts > 0).any() and (counts < drafts).any()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("vocab", [VOCAB, 1000, 2049])
@pytest.mark.parametrize("stride", STRIDES)
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("temps", ["sampled", "mixed"])
def test_resample(dtype, vocab, stride, case, temps):
    """The residual's block winners: rejected and placeholder drafts, bonus rows,
    greedy rows left alone."""
    batch = _batch(_counts(96), temps, 11, vocab)
    logits = _logits(batch, vocab, dtype, case, 11, stride)
    stats = _triton_local_stats(logits, batch)
    verified = _triton_verify(logits, stats, batch)
    num_sampled = verified[2]
    theirs = _triton_resample(logits, num_sampled, batch, verified)
    ours = _resample_buffers(num_sampled.shape[0], vocab)
    tlr.resample(
        *ours,
        logits,
        num_sampled,
        batch["cu"],
        batch["expanded"],
        batch["drafts"],
        batch["temperature"],
        batch["seeds"],
        batch["pos"],
    )
    _assert_same(ours[0], theirs[0], "block argmax")
    _assert_same(ours[1], theirs[1], "block maximum")


def test_resample_index_dtypes():
    batch = _batch(_counts(96), "mixed", 13, VOCAB, "other")
    logits = _logits(batch, VOCAB, torch.bfloat16, "peaked", 13)
    stats = _triton_local_stats(logits, batch)
    verified = _triton_verify(logits, stats, batch)
    num_sampled = verified[2]
    theirs = _triton_resample(logits, num_sampled, batch, verified)
    ours = _resample_buffers(num_sampled.shape[0], VOCAB)
    tlr.resample(
        *ours,
        logits,
        num_sampled,
        batch["cu"],
        batch["expanded"],
        batch["drafts"],
        batch["temperature"],
        batch["seeds"],
        batch["pos"],
    )
    _assert_same(ours[0], theirs[0], "block argmax")
    _assert_same(ours[1], theirs[1], "block maximum")


# --- rejection_sample ---------------------------------------------------------


def _assert_same_tokens(ours, theirs):
    (sampled, num_sampled), (expected, expected_counts) = ours, theirs
    _assert_same(num_sampled, expected_counts, "num_sampled")
    for r in range(expected_counts.shape[0]):
        n = int(expected_counts[r])
        _assert_same(sampled[r, :n], expected[r, :n], f"request {r}'s tokens")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("vocab", [VOCAB, 1000, 8193])
@pytest.mark.parametrize("stride", STRIDES)
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("rows", [1, 16, 96])
def test_rejection_sample(monkeypatch, dtype, vocab, stride, case, rows):
    """rejection_sample's tokens and counts, through the TileLang kernels, against
    the Triton launches."""
    launched = []

    def tilelang(*args, **kwargs):
        launched.append(True)
        return tlr.verify_and_resample(*args, **kwargs)

    monkeypatch.setattr(rs, "verify_and_resample", tilelang)
    batch = _batch(_counts(rows), "mixed", 17 + rows, vocab)
    logits = _logits(batch, vocab, dtype, case, 17 + rows, stride)
    ours = _rejection_sample(logits, batch)
    assert launched
    _assert_same_tokens(ours, _triton_rejection_sample(logits, batch))


@pytest.mark.parametrize("indices", INDICES)
@pytest.mark.parametrize("temps", ["sampled", "mixed"])
def test_rejection_sample_index_dtypes(indices, temps):
    batch = _batch(_counts(48), temps, 19, VOCAB, indices)
    logits = _logits(batch, VOCAB, torch.float32, "peaked", 19)
    _assert_same_tokens(
        _rejection_sample(logits, batch), _triton_rejection_sample(logits, batch)
    )


def test_rejection_sample_rows_do_not_depend_on_the_batch():
    """A request's tokens are those it gets alone."""
    batch = _batch(_counts(48), "mixed", 23, VOCAB)
    logits = _logits(batch, VOCAB, torch.float32, "peaked", 23)
    whole, counts = _rejection_sample(logits, batch)
    cu = batch["cu"]
    for r in (0, 3, cu.shape[0] - 2):
        start, end = int(cu[r]), int(cu[r + 1])
        alone = {
            "cu": torch.tensor([0, end - start], dtype=torch.int32, device=DEVICE),
            "idx_mapping": batch["idx_mapping"][r : r + 1],
            **{k: batch[k][start:end] for k in ("expanded", "local", "drafts", "pos")},
            "temperature": batch["temperature"],
            "seeds": batch["seeds"],
        }
        sampled, count = _rejection_sample(logits[start:end], alone)
        assert count.item() == counts[r].item()
        assert torch.equal(sampled[0, : count.item()], whole[r, : count.item()])


def test_rejection_sample_without_requests():
    """No launch: TileLang's grids may not be empty."""
    index = torch.empty(0, dtype=torch.int64, device=DEVICE)
    batch = {
        "cu": torch.zeros(1, dtype=torch.int32, device=DEVICE),
        "idx_mapping": index,
        "expanded": index,
        "local": index.int(),
        "drafts": index.int(),
        "temperature": torch.ones(SLOTS, device=DEVICE),
        "seeds": torch.zeros(SLOTS, dtype=torch.int64, device=DEVICE),
        "pos": index,
    }
    logits = torch.empty((0, VOCAB), device=DEVICE)
    sampled, num_sampled = _rejection_sample(logits, batch)
    assert sampled.shape == (0, STEPS + 1) and num_sampled.shape == (0,)


@pytest.mark.parametrize(
    "use_block_verification,synthetic,use_fp64",
    [(False, False, False), (False, True, False), (True, False, True)],
)
def test_other_configurations_launch_triton(
    monkeypatch, use_block_verification, synthetic, use_fp64
):
    def refuse(*args, **kwargs):
        raise AssertionError("launched the TileLang kernels")

    monkeypatch.setattr(rs, "verify_and_resample", refuse)
    batch = _batch(_counts(16), "mixed", 29, 1000)
    logits = _logits(batch, 1000, torch.float32, "peaked", 29)
    rates = torch.full((STEPS,), 0.5, device=DEVICE) if synthetic else None
    sampled, num_sampled = rs.rejection_sample(
        logits,
        None,
        batch["drafts"],
        batch["cu"],
        batch["pos"],
        batch["idx_mapping"],
        batch["expanded"],
        batch["local"],
        batch["temperature"],
        batch["seeds"],
        STEPS,
        rates,
        use_fp64=use_fp64,
        use_block_verification=use_block_verification,
    )
    assert num_sampled.shape == (batch["cu"].shape[0] - 1,)


# --- Preparation --------------------------------------------------------------


def _declared(registry):
    from vllm.models.deepseek_v4_1.tilelang import prepare

    # JIT functions are not hashable: a call is its function's identity and key.
    return {
        (id(function), args) for function, args, _ in prepare.declared_calls(registry)
    }


def test_runner_declares_the_launched_calls():
    """register_rejection_sample_warmups declares the calls the runner's launches
    make (its logits dtypes, both layouts, its index dtypes), and they compile."""
    from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry
    from vllm.models.deepseek_v4_1.tilelang import prepare

    logits = [(torch.bfloat16, VOCAB), (torch.float32, VOCAB)]
    registry = JitWarmupRegistry(None)
    with registry.activate():
        rs.register_rejection_sample_warmups(
            logits=logits,
            vocab_size=VOCAB,
            num_speculative_steps=STEPS,
            synthetic_mode=False,
            use_block_verification=True,
            use_fp64=False,
        )
    drafts, idx, pos, local = INDICES["runner"]
    expected = set()
    for dtype, _ in logits:
        for run in tls._runs(dtype):
            expected.add(
                (
                    id(tlr.local_stats_kernel),
                    tlr.local_stats_key(dtype, idx, local, run),
                )
            )
            expected.add(
                (
                    id(tlr.resample_kernel),
                    tlr.resample_key(dtype, idx, drafts, pos, run),
                )
            )
        expected.add(
            (
                id(tlr.verify_kernel),
                tlr.verify_key(dtype, drafts, idx, pos, VOCAB, STEPS),
            )
        )
    assert expected <= _declared(registry)
    prepare.prepare_declared(registry)


@pytest.mark.parametrize(
    "use_block_verification,synthetic_mode,use_fp64",
    [(False, False, False), (False, True, False), (True, False, True)],
)
def test_other_configurations_declare_no_tilelang_kernel(
    use_block_verification, synthetic_mode, use_fp64
):
    from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry

    registry = JitWarmupRegistry(None)
    with registry.activate():
        rs.register_rejection_sample_warmups(
            logits=[(torch.float32, VOCAB)],
            vocab_size=VOCAB,
            num_speculative_steps=STEPS,
            synthetic_mode=synthetic_mode,
            use_block_verification=use_block_verification,
            use_fp64=use_fp64,
        )
    ours = {id(tlr.local_stats_kernel), id(tlr.verify_kernel), id(tlr.resample_kernel)}
    assert not {function for function, _ in _declared(registry)} & ours
