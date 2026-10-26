# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The TileLang rejection sampler (``spec_decode.tilelang_rejection``) against the
Triton kernels it replaces, and against FP64 references of the probabilities it
computes by Barron, Gnanapragasam and Tschannen's fix ("A Fix for Softmax and
Cross-Entropy", Eq. 5):

- bit for bit where the arithmetic is the Triton kernels': the greedy rows' block
  argmaxes, every block maximum, the sampled tokens, greedy requests' counts, the
  draft log-sum-exps, the elements a launch leaves alone, resampling and the
  insertion of the resampled token;
- within their rounding bounds of FP64 references that follow the kernels' blocks
  (NaN and infinite logits included): the sampled rows' block sums of the other
  columns' exponentials, the cumulative log ratios and the rejected rows'
  log-sum-exps;
- block verification's decisions equal the FP64 reference's from the same uniform
  draws wherever no draw is within 1e-4 of its threshold, and the threshold itself
  within a few ULPs of its FP64 value over confident and unsure logs, where the
  release's formula (1 - m and 1 - p of probabilities) is off by up to 1;
- the paper's z = [9, -9, -9] as a verified draft;
- a request's tokens do not depend on its batch or the run.

Batches of requests with 0 to 5 drafts, -1 placeholders, mixed temperatures and
per-request seeds, and logits with ties, infinities and NaN."""

import itertools
import math

import pytest
import torch

triton = pytest.importorskip("triton")
pytest.importorskip("tilelang")

from vllm.tilelang_utils import T, tilelang  # noqa: E402
from vllm.triton_utils import tl  # noqa: E402
from vllm.v1.worker.gpu.sample import tilelang_sampling as tls  # noqa: E402
from vllm.v1.worker.gpu.sample.gumbel import spec_step_seed, tl_rand32  # noqa: E402
from vllm.v1.worker.gpu.spec_decode import rejection_sampler_utils as rs  # noqa: E402
from vllm.v1.worker.gpu.spec_decode import tilelang_rejection as tlr  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)

DEVICE = torch.device("cuda")
VOCAB = 129280  # DeepSeek V4.1
STEPS = 5  # DSpark's num_speculative_steps
SLOTS = 160
# A sum of the other columns' exponentials within this of its value (the
# exponentials' and the sums' roundings: tests/v1/worker/
# test_gpu_tilelang_logprob_softmax.py), and a subnormal a column for the ones that
# are subnormal.
S_RELATIVE = 2e-5
SUBNORMAL = 2.0**-149
# A draw this close to its threshold is not decided by the reference.
BORDER = 1e-4


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
        f"expected {[theirs[tuple(i)].item() for i in first]}"
    )


def _ulp(x: float) -> float:
    """The FP32 ULP at |x| (finite): FP64's times 2^29, at least a subnormal."""
    return max(math.ulp(abs(x)) * 2.0**29, SUBNORMAL)


def _assert_close(ours: float, ref: float, bound: float, what: str) -> None:
    """Equal NaN and infinities, else within ``bound``."""
    if math.isnan(ref) or math.isinf(ref):
        assert (math.isnan(ours) and math.isnan(ref)) or ours == ref, (what, ours, ref)
        return
    assert abs(ours - ref) <= bound, (what, ours, ref, bound)


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


# --- FP64 references ------------------------------------------------------------


def _block_sums(logits):
    """``local_stats``' sampled-row statistics in FP64: per 8192-column block (the
    columns past the row -inf) the maximum, NaN ignored (NaN for a block of NaN
    only), and the sum of exp(x - maximum) over the block's columns but one holding
    the maximum: the others plus exp(0) for each further tie (NaN for an infinite
    maximum's), 0 unless the maximum is above -inf."""
    rows, vocab = logits.shape
    blocks = -(-vocab // tlr.STATS_BLOCK)
    v = torch.full(
        (rows, blocks * tlr.STATS_BLOCK),
        float("-inf"),
        dtype=torch.float64,
        device=DEVICE,
    )
    v[:, :vocab] = logits.float().double()
    v = v.view(rows, blocks, tlr.STATS_BLOCK)
    nan = torch.isnan(v)
    m = torch.where(nan, float("-inf"), v).amax(dim=2)
    m = torch.where(nan.all(dim=2), float("nan"), m)
    tie = v == m[..., None]
    ties = tie.sum(dim=2).double()
    terms = torch.where(tie, 0.0, torch.exp(v - m[..., None]))
    s = terms.sum(dim=2) + torch.where(ties > 1, (ties - 1) * torch.exp(m - m), 0.0)
    return m, torch.where(m > float("-inf"), s, 0.0)


def _rows(logits):
    """Each row's maximum and S as ``verify`` forms them from the block sums, in
    FP64: the maximum over Triton's padded lanes (NaN ignored; NaN if every lane is
    NaN), S the block sums rescaled to it, each with its left-out column back
    (1 + sum), except the first block holding it; NaN for two or more columns that
    are all -inf. Python lists."""
    vocab = logits.shape[1]
    m, s = _block_sums(logits)
    lanes = tlr._padded_blocks(vocab)
    pad = lanes - m.shape[1]
    m = torch.nn.functional.pad(m, (0, pad), value=float("-inf"))
    s = torch.nn.functional.pad(s, (0, pad), value=0.0)
    nan = torch.isnan(m)
    top = torch.where(nan, float("-inf"), m).amax(dim=1, keepdim=True)
    top = torch.where(nan.all(dim=1, keepdim=True), float("nan"), top)
    holds = m == top
    first = torch.where(holds.any(dim=1), holds.int().argmax(dim=1), lanes)
    lane = torch.arange(lanes, device=DEVICE)
    terms = torch.where(lane == first[:, None], s, torch.exp(m - top) * (1 + s))
    total = terms.sum(dim=1, keepdim=True)
    if vocab > 1:
        total = torch.where(top == float("-inf"), float("nan"), total)
    return top[:, 0].tolist(), total[:, 0].tolist()


def _log_prob(logits, rows, row: int, token: int) -> tuple[float, float]:
    """The FP64 log probability of ``token`` at ``row`` (Eq. 5: the shifted logit,
    exactly 0 at the maximum, minus log1p(S)) and the bound of the kernel's error on
    it (0 where it is not finite)."""
    top, total = rows
    z = logits[row, token].float().item()
    m, s = top[row], total[row]
    shifted = 0.0 if z == m else z - m
    lse = math.log1p(s)
    value = shifted - lse
    if not math.isfinite(value):
        return value, 0.0
    bound = (
        0.5 * _ulp(shifted)
        + _ulp(value)
        + _ulp(lse)
        + S_RELATIVE * s / (1 + s)
        + logits.shape[1] * SUBNORMAL
    )
    return value, bound


def _threshold(log_p: float, log_m: float) -> float:
    """Block verification's threshold p (1 - m) / (1 - p m) in FP64; 1 where the
    denominator is not positive."""
    denom = -math.expm1(log_p + log_m)
    if not denom > 0:
        return 1.0
    return math.exp(log_p) * -math.expm1(log_m) / denom


def _verify_reference(logits, batch, uniforms):
    """Each sampled request's verification in FP64: its cumulative log ratios with
    their bounds, the count block verification accepts from the uniform draws
    (None where a draw is within BORDER of its threshold) and its first rejected
    row (None when every draft is accepted)."""
    rows = _rows(logits)
    cu = batch["cu"].tolist()
    drafts = batch["drafts"].long().tolist()
    temps = batch["temperature"][batch["idx_mapping"].long()].tolist()
    draws = uniforms.tolist()
    out = {}
    for r in range(len(cu) - 1):
        if temps[r] == 0:
            continue
        start = cu[r]
        num_drafts = cu[r + 1] - start - 1
        log_p, bound = 0.0, 0.0
        cumulative = []
        accepted, decided, verifying = 0, True, True
        for i in range(num_drafts):
            draft = drafts[start + i + 1]
            if draft >= 0:
                log_m, error = _log_prob(logits, rows, start + i, draft)
                step = log_p + log_m
                # min.f32: a NaN sum gives 0.
                log_p = 0.0 if math.isnan(step) else min(step, 0.0)
                bound = 0.0 if math.isnan(step) else bound + error + _ulp(log_p)
            cumulative.append((log_p, bound))
            verifying = verifying and draft >= 0
            if verifying:
                nxt = drafts[start + i + 2] if i + 1 < num_drafts else -1
                if nxt >= 0:
                    log_next, _ = _log_prob(logits, rows, start + i + 1, nxt)
                    h = _threshold(log_p, log_next)
                else:
                    h = math.exp(log_p)
                if abs(draws[start + i] - h) <= BORDER:
                    decided = False
                if draws[start + i] <= h:
                    accepted = i + 1
        rejected = start + accepted if accepted < num_drafts else None
        out[r] = (cumulative, accepted if decided else None, rejected)
    return out, rows


@triton.jit
def _uniforms_kernel(out_ptr, cu_ptr, idx_ptr, seeds_ptr, pos_ptr):
    req = tl.program_id(0)
    start = tl.load(cu_ptr + req)
    end = tl.load(cu_ptr + req + 1)
    seed = tl.load(seeds_ptr + tl.load(idx_ptr + req).to(tl.int64))
    for i in range(end - start):
        pos = tl.load(pos_ptr + start + i)
        u = tl_rand32(spec_step_seed(seed, i), pos, includes_zero=False)
        tl.store(out_ptr + start + i, u)


def _uniforms(batch, rows: int) -> torch.Tensor:
    """Each logits row's draw, as ``_rejection_kernel`` draws it: tl_rand32 at
    spec_step_seed(seed, step) and the row's position, the step its place in its
    request."""
    out = torch.full((rows,), float("nan"), device=DEVICE)
    reqs = batch["cu"].shape[0] - 1
    if reqs:
        _uniforms_kernel[(reqs,)](
            out, batch["cu"], batch["idx_mapping"], batch["seeds"], batch["pos"]
        )
    return out


# --- Each kernel --------------------------------------------------------------


def _local_stats(logits, batch):
    ours = _stats_buffers(*logits.shape)
    tlr.compute_local_stats(
        *ours,
        logits,
        batch["expanded"],
        batch["local"],
        batch["temperature"],
        STEPS,
    )
    return ours


def _assert_local_stats(ours, theirs, logits):
    """Argmaxes and maxima bit for bit, the same elements written, and each written
    block sum within its bound of the FP64 reference."""
    _assert_same(ours[0], theirs[0], "argmax")
    _assert_same(ours[1], theirs[1], "maximum")
    written = theirs[2] != -7
    _assert_same(ours[2][~written], theirs[2][~written], "rows left alone")
    _, s = _block_sums(logits)
    ref, got = s[written], ours[2][written].double()
    nan = torch.isnan(ref)
    assert torch.equal(torch.isnan(got), nan), "NaN block sums"
    bound = S_RELATIVE * ref.abs() + tlr.STATS_BLOCK * SUBNORMAL
    error = (got - ref).abs()
    bad = ~nan & ~(error <= bound)
    assert not bad.any(), (
        f"{int(bad.sum())} block sums outside their bound: ours "
        f"{got[bad][:5].tolist()}, reference {ref[bad][:5].tolist()}"
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("vocab", [VOCAB, 1000, 8193])
@pytest.mark.parametrize("stride", STRIDES)
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("rows", [1, 16, 96])
def test_local_stats(dtype, vocab, stride, case, rows):
    """The block statistics of greedy and sampled rows; bonus rows left alone."""
    batch = _batch(_counts(rows), "mixed", rows, vocab)
    logits = _logits(batch, vocab, dtype, case, rows, stride)
    _assert_local_stats(
        _local_stats(logits, batch), _triton_local_stats(logits, batch), logits
    )


@pytest.mark.parametrize("indices", INDICES)
def test_local_stats_index_dtypes(indices):
    batch = _batch(_counts(37), "mixed", 3, VOCAB, indices)
    logits = _logits(batch, VOCAB, torch.float32, "special", 3)
    _assert_local_stats(
        _local_stats(logits, batch), _triton_local_stats(logits, batch), logits
    )


def _verified(logits, batch):
    """The TileLang verification from its own block statistics, and Triton's from
    Triton's."""
    theirs = _triton_verify(logits, _triton_local_stats(logits, batch), batch)
    ours = _verify_buffers(batch["cu"].shape[0] - 1, logits.shape[0])
    cumulative, sampled, num_sampled, target_lse, draft_lse = ours
    tlr.verify(
        sampled,
        num_sampled,
        target_lse,
        draft_lse,
        cumulative,
        logits,
        *_local_stats(logits, batch),
        batch["drafts"],
        batch["cu"],
        batch["idx_mapping"],
        batch["temperature"],
        batch["seeds"],
        batch["pos"],
        STEPS,
    )
    return ours, theirs


def _assert_verified(ours, theirs, logits, batch):
    """Triton's bits for the sampled tokens, the draft log-sum-exps, the elements
    left alone and every greedy request's outputs; each sampled request's cumulative
    log ratios within their bounds of the FP64 reference, its count the reference's
    wherever no draw is within BORDER of its threshold, and the rejected row's
    log-sum-exp within its bound. Returns how many sampled requests were decided."""
    cumulative, sampled, num_sampled, target_lse, draft_lse = ours
    _assert_same(sampled, theirs[1], "sampled tokens")
    _assert_same(draft_lse, theirs[4], "draft lse")
    left = theirs[0] == -7
    _assert_same(cumulative[left], theirs[0][left], "cumulative rows left alone")
    greedy = batch["temperature"][batch["idx_mapping"].long()] == 0
    _assert_same(num_sampled[greedy], theirs[2][greedy], "greedy counts")
    _assert_same(target_lse[greedy], theirs[3][greedy], "greedy lse")
    reference, (top, total) = _verify_reference(
        logits, batch, _uniforms(batch, logits.shape[0])
    )
    cu = batch["cu"].tolist()
    counts = num_sampled.tolist()
    decided = 0
    for r, (steps, accepted, rejected) in reference.items():
        for i, (log_p, bound) in enumerate(steps):
            _assert_close(
                cumulative[cu[r] + i].item(), log_p, bound, f"request {r} step {i}"
            )
        if accepted is None:
            continue
        decided += 1
        assert counts[r] == accepted, (r, counts[r], accepted)
        lse = target_lse[r].item()
        if rejected is None:
            assert lse == 0.0, (r, lse)
            continue
        m, s = top[rejected], total[rejected]
        softplus = math.log1p(s)
        ref = m + softplus
        bound = 0.0
        if math.isfinite(ref):
            bound = (
                _ulp(ref)
                + _ulp(softplus)
                + S_RELATIVE * s / (1 + s)
                + logits.shape[1] * SUBNORMAL
            )
        _assert_close(lse, ref, bound, f"request {r}'s rejected row")
    return decided, len(reference)


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
    decided, sampled = _assert_verified(ours, theirs, logits, batch)
    assert decided >= 0.9 * sampled, (decided, sampled)
    if case == "peaked" and vocab > 2:
        drafts = (batch["cu"][1:] - batch["cu"][:-1] - 1).long()
        counts = ours[2].long()
        assert (counts > 0).any() and (counts < drafts).any()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_paper_example_as_a_draft(dtype):
    """Two drafts of the top token of z = [9, -9, -9]: each step's log probability
    is -log1p(2 e^-18) = -3.046e-8, so the cumulative log ratios are -3.046e-8 and
    -6.092e-8, where the release's were exactly 0 (and its threshold, from
    1 - m = 0 and 1 - p = 0, the not-positive denominator's 1 instead of about
    one half)."""
    batch = {
        "cu": torch.tensor([0, 3], dtype=torch.int32, device=DEVICE),
        "idx_mapping": torch.tensor([4], device=DEVICE),
        "expanded": torch.tensor([4, 4, 4], device=DEVICE),
        "local": torch.tensor([0, 1, 2], dtype=torch.int32, device=DEVICE),
        "drafts": torch.tensor([2, 0, 0], dtype=torch.int32, device=DEVICE),
        "temperature": torch.ones(SLOTS, device=DEVICE),
        "seeds": torch.arange(SLOTS, device=DEVICE),
        "pos": torch.tensor([5, 6, 7], device=DEVICE),
    }
    logits = torch.tensor([[9.0, -9.0, -9.0]] * 3, device=DEVICE).to(dtype)
    ours, theirs = _verified(logits, batch)
    _assert_verified(ours, theirs, logits, batch)
    step = -math.log1p(2 * math.exp(-18.0))
    for i in range(2):
        got = ours[0][i].item()
        assert abs(got - (i + 1) * step) <= S_RELATIVE * (i + 1) * abs(step), got
        assert theirs[0][i].item() == 0.0


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


@tilelang.jit(pass_configs=tls._PASS)
def _threshold_kernel(threads: int):
    n = T.dynamic("n")

    @T.prim_func
    def thresholds(
        log_p: T.Tensor((n,), T.float32),
        log_m: T.Tensor((n,), T.float32),
        out: T.Tensor((n,), T.float32),
    ):
        with T.Kernel(T.ceildiv(n, threads), threads=threads) as block:
            T.import_source(tlr._HEADER)
            i = block * threads + T.get_thread_binding()
            if i < n:
                out[i] = T.call_extern(
                    "float32", "vllm_block_threshold", log_p[i], log_m[i]
                )

    return thresholds


# Logs of probabilities from certain to unlikely: the release's formula broke in the
# confident ones, where 1 - m and 1 - p cancel or are exactly 0.
LOGS = (0.0, -1e-12, -1e-9, -3e-8, -6e-8, -1e-7, -3e-7, -1e-6, -1e-5, -1e-4, -1e-3,
        -0.01, -0.1, -0.7, -3.0, -20.0, float("-inf"))  # fmt: skip


def test_block_threshold():
    """``vllm_block_threshold`` over every pair of LOGS (and a NaN next draft)
    within a few ULPs of its FP64 value (the prefix ratio's exponential within
    1.4e-7 |log p| + 2^-22, two expm1 and a quotient); the release's formula, which
    took 1 - m and p (1 - m) + 1 - p of the probabilities, emulated in FP32, is off
    by up to 1 on the same grid."""
    pairs = [(p, m) for p in LOGS for m in (*LOGS, float("nan"))]
    log_p = torch.tensor([p for p, _ in pairs], device=DEVICE)
    log_m = torch.tensor([m for _, m in pairs], device=DEVICE)
    out = torch.full_like(log_p, -7.0)
    _threshold_kernel(128)(log_p, log_m, out)
    worst_release = 0.0
    for k, (lp, lm) in enumerate(zip(log_p.tolist(), log_m.tolist())):
        ref = _threshold(lp, lm)
        bound = (1.4e-7 * abs(lp) + 1e-6) * ref if math.isfinite(lp) else 0.0
        _assert_close(out[k].item(), ref, bound, f"threshold at {lp}, {lm}")
        if math.isnan(lm):
            continue
        # The release: vllm_block_threshold(exp(log_p), exp(log_m)) in FP32.
        p = torch.tensor(lp).exp().float()
        m = torch.tensor(lm).exp().float()
        rest = 1 - m
        residual = p * rest
        denom = (p.double() * rest.double() + 1).float() - p
        release = (residual / denom).item() if denom > 0 else 1.0
        worst_release = max(worst_release, abs(release - ref))
    assert worst_release >= 0.4, worst_release


# --- rejection_sample ---------------------------------------------------------


def _assert_same_tokens(ours, theirs):
    (sampled, num_sampled), (expected, expected_counts) = ours, theirs
    _assert_same(num_sampled, expected_counts, "num_sampled")
    for r in range(expected_counts.shape[0]):
        n = int(expected_counts[r])
        _assert_same(sampled[r, :n], expected[r, :n], f"request {r}'s tokens")


def _expected_tokens(logits, batch):
    """rejection_sample's tokens and counts from the TileLang verification (which
    ``test_verify`` holds to the FP64 reference), resampled and inserted by the
    Triton kernels, whose arithmetic the TileLang ones are bit for bit."""
    verified, _ = _verified(logits, batch)
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


def _assert_greedy_as_triton(ours, logits, batch):
    """Greedy requests take the Triton launches' tokens and counts."""
    expected, counts = _triton_rejection_sample(logits, batch)
    greedy = (batch["temperature"][batch["idx_mapping"].long()] == 0).tolist()
    for r, is_greedy in enumerate(greedy):
        if is_greedy:
            n = int(counts[r])
            assert int(ours[1][r]) == n
            _assert_same(ours[0][r, :n], expected[r, :n], f"greedy request {r}")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("vocab", [VOCAB, 1000, 8193])
@pytest.mark.parametrize("stride", STRIDES)
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("rows", [1, 16, 96])
def test_rejection_sample(monkeypatch, dtype, vocab, stride, case, rows):
    """rejection_sample's tokens and counts, through the TileLang kernels."""
    launched = []

    def tilelang(*args, **kwargs):
        launched.append(True)
        return tlr.verify_and_resample(*args, **kwargs)

    monkeypatch.setattr(rs, "verify_and_resample", tilelang)
    batch = _batch(_counts(rows), "mixed", 17 + rows, vocab)
    logits = _logits(batch, vocab, dtype, case, 17 + rows, stride)
    ours = _rejection_sample(logits, batch)
    assert launched
    _assert_same_tokens(ours, _expected_tokens(logits, batch))
    _assert_greedy_as_triton(ours, logits, batch)


@pytest.mark.parametrize("indices", INDICES)
@pytest.mark.parametrize("temps", ["sampled", "mixed"])
def test_rejection_sample_index_dtypes(indices, temps):
    batch = _batch(_counts(48), temps, 19, VOCAB, indices)
    logits = _logits(batch, VOCAB, torch.float32, "peaked", 19)
    ours = _rejection_sample(logits, batch)
    _assert_same_tokens(ours, _expected_tokens(logits, batch))
    _assert_greedy_as_triton(ours, logits, batch)


def test_rejection_sample_rows_do_not_depend_on_the_batch():
    """A request's tokens are those it gets alone, and the same on every run."""
    batch = _batch(_counts(48), "mixed", 23, VOCAB)
    logits = _logits(batch, VOCAB, torch.float32, "peaked", 23)
    whole, counts = _rejection_sample(logits, batch)
    again, again_counts = _rejection_sample(logits, batch)
    _assert_same(again_counts, counts, "second run's counts")
    _assert_same_tokens((again, again_counts), (whole, counts))
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
