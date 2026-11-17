# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark's TileLang confidence head against FP64 and the torch head it replaces
(``sigmoid(F.linear(cat([h, e]).float(), w))``: a cuBLAS dot product for a few
rows, a GEMV for more)."""

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

if not (
    current_platform.is_cuda() and current_platform.is_device_capability_family(120)
):
    pytest.skip("TileLang V4.1 kernels target SM120/SM121", allow_module_level=True)

pytest.importorskip("tilelang")

from tests.kernels.deepseek_v41_contracts import (  # noqa: E402
    INF,
    NAN,
    assert_finite_equal,
    assert_other_rows_equal,
    assert_same_nonfinite,
)
from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry  # noqa: E402
from vllm.models.deepseek_v4_1.tilelang import confidence, prepare  # noqa: E402

# DeepSeek-V4.1's drafter: 5120-wide head hidden, 256-wide Markov embeddings.
HIDDEN, RANK = 5120, 256
WIDTH = HIDDEN + RANK
# Roundings along the kernel's longest path, FMA contraction or not (confidence.py).
ROUNDINGS = 23
U = 2.0**-24


def _inputs(rows: int, seed: int = 0):
    """Hidden and Markov rows near unit scale and a weight row that keeps the logits
    near unit scale too, where the probability is most sensitive to them."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    hidden = torch.randn(rows, HIDDEN, generator=generator, device="cuda").bfloat16()
    markov = torch.randn(rows, RANK, generator=generator, device="cuda").bfloat16()
    weight = torch.randn(WIDTH, generator=generator, device="cuda") / math.sqrt(WIDTH)
    return hidden, markov, weight


def _exact_inputs(rows: int, seed: int = 0):
    """Small integers and weights in multiples of 2^-10: every product and partial
    sum is a multiple of 2^-10 below 2^14, exact in FP32 in any order."""
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def ints(low, high, shape):
        return torch.randint(low, high, shape, generator=generator, device="cuda")

    hidden = ints(-4, 5, (rows, HIDDEN)).bfloat16()
    markov = ints(-4, 5, (rows, RANK)).bfloat16()
    weight = ints(-8, 9, (WIDTH,)).float() * 2.0**-10
    return hidden, markov, weight


def _tilelang(hidden, markov, weight) -> torch.Tensor:
    out = torch.empty(hidden.shape[0], dtype=torch.float32, device="cuda")
    confidence.dspark_confidence(hidden, markov, weight, out)
    return out


def _torch_head(hidden, markov, weight) -> torch.Tensor:
    """``DSparkConfidenceHead.probability`` as the deployment ran it."""
    x = torch.cat([hidden, markov], dim=-1).float()
    return torch.sigmoid(F.linear(x, weight.view(1, -1)).squeeze(-1))


def _fp64(hidden, markov, weight):
    """The exact probability and each row's sum of |products|."""
    x = torch.cat([hidden, markov], dim=-1).double()
    w = weight.double()
    return torch.sigmoid(x @ w), x.abs() @ w.abs()


def _bound(p64: torch.Tensor, magnitude: torch.Tensor) -> torch.Tensor:
    """The kernel's error bound: the FP32 sum's roundings through the sigmoid's
    slope, plus the sigmoid's own (expf within 2 ulp, the add and the division)."""
    return p64 * (1 - p64) * ROUNDINGS * U * magnitude * 1.01 + p64 * 2.0**-21


@pytest.mark.parametrize("rows", [1, 5, 10, 80, 96])
def test_an_exact_sum_gives_the_torch_heads_bits(rows):
    """Where FP32 sums exactly in any order, the logits agree, so the probability
    shows the kernel's layout (the concatenation, the weight's halves) and its
    sigmoid: ATen's ``1 / (1 + expf(-z))`` bit for bit."""
    hidden, markov, weight = _exact_inputs(rows, seed=rows)
    got = _tilelang(hidden, markov, weight)
    want = _torch_head(hidden, markov, weight)
    assert got.shape == (rows,) and got.dtype == torch.float32
    assert torch.equal(got, want)


@pytest.mark.parametrize("rows", [1, 5, 10, 80, 96, 1000])
def test_within_its_fp32_bound_of_fp64(rows):
    hidden, markov, weight = _inputs(rows, seed=rows)
    got = _tilelang(hidden, markov, weight).double()
    p64, magnitude = _fp64(hidden, markov, weight)
    error = (got - p64).abs()
    assert (error <= _bound(p64, magnitude)).all(), error.max().item()
    # The torch head computes the same FP32 sum in another order.
    torch_head = _torch_head(hidden, markov, weight).double()
    assert (got - torch_head).abs().max().item() <= 2.0**-18


def test_as_accurate_as_cublas():
    """Over many rows, no less accurate than cuBLAS's FP32 sum: the kernel's eight
    short chains a thread and its trees keep the error near cuBLAS's."""
    hidden, markov, weight = _inputs(4096, seed=11)
    p64, _ = _fp64(hidden, markov, weight)
    ours = (_tilelang(hidden, markov, weight).double() - p64).abs()
    cublas = (_torch_head(hidden, markov, weight).double() - p64).abs()
    assert ours.mean().item() <= 2 * cublas.mean().item() + 2.0**-27, (
        ours.mean().item(),
        cublas.mean().item(),
    )
    assert ours.max().item() <= 2 * cublas.max().item() + 2.0**-24, (
        ours.max().item(),
        cublas.max().item(),
    )


def test_rows_independent_of_batch():
    """cuBLAS sums a row in another order when the step has more rows (dot below,
    GEMV above); the kernel's row does not depend on the other rows."""
    hidden, markov, weight = _inputs(130, seed=3)
    full = _tilelang(hidden, markov, weight)
    for start, stop in ((0, 1), (0, 5), (0, 10), (7, 20), (0, 80), (1, 130)):
        assert torch.equal(
            _tilelang(hidden[start:stop], markov[start:stop], weight),
            full[start:stop],
        ), (start, stop)


def test_repeatable():
    hidden, markov, weight = _inputs(80, seed=4)
    first = _tilelang(hidden, markov, weight)
    for _ in range(3):
        assert torch.equal(_tilelang(hidden, markov, weight), first)


def test_nonfinite_like_the_torch_head():
    """A NaN reaches its row's probability; an infinite product saturates it at 0 or
    1, as through the torch head; other rows keep their bits."""
    hidden, markov, weight = _inputs(8, seed=5)
    weight[3], weight[HIDDEN + 9] = 0.5, -0.5
    hidden[0, 17] = NAN
    hidden[2, 3] = INF  # +inf * 0.5: probability 1
    markov[4, 9] = INF  # +inf * -0.5: probability 0
    markov[5, 30] = NAN
    rows = [0, 2, 4, 5]
    clean = _tilelang(hidden[[1, 3, 6, 7]], markov[[1, 3, 6, 7]], weight)
    got = _tilelang(hidden, markov, weight)
    want = _torch_head(hidden, markov, weight)
    assert_same_nonfinite(got, want)
    assert_finite_equal(got[rows], want[rows])
    assert got[2].item() == 1.0 and got[4].item() == 0.0
    expected = torch.empty_like(got)
    expected[[1, 3, 6, 7]] = clean
    assert_other_rows_equal(got, expected, rows)


def test_no_rows_launch_nothing():
    hidden, markov, weight = _inputs(0)
    assert _tilelang(hidden, markov, weight).shape == (0,)


def test_refuses_other_layouts():
    hidden, markov, weight = _inputs(5)
    with pytest.raises(ValueError):
        _tilelang(hidden.float(), markov, weight)
    with pytest.raises(ValueError):
        _tilelang(hidden, markov, weight.bfloat16())
    with pytest.raises(ValueError):
        _tilelang(hidden, markov[:4], weight)
    with pytest.raises(ValueError):
        _tilelang(hidden[:, ::2], markov, weight[: HIDDEN // 2 + RANK])


def test_the_head_runs_its_weight_row():
    """``TileLangConfidenceHead.probability`` reads its projection's [1, H + R]
    weight; its logit path is not available."""
    from vllm.model_executor.models.qwen3_dspark import DSparkConfidenceHead

    head = confidence.TileLangConfidenceHead.__new__(confidence.TileLangConfidenceHead)
    torch.nn.Module.__init__(head)
    hidden, markov, weight = _inputs(5, seed=6)
    head.proj = SimpleNamespace(weight=weight.view(1, -1))
    head.hidden_size, head.markov_rank = HIDDEN, RANK
    assert head.tp_invariant is True
    assert DSparkConfidenceHead.tp_invariant is False
    want = _tilelang(hidden, markov, weight)
    assert torch.equal(head.probability(hidden, markov), want)
    out = torch.empty(5, dtype=torch.float32, device="cuda")
    returned = head.probability(hidden, markov, out=out)
    assert returned is out
    assert torch.equal(out, want)
    with pytest.raises(NotImplementedError):
        head(hidden, markov)


def test_declared_as_the_head_calls_it():
    registry = JitWarmupRegistry(vllm_config=None)
    with registry.activate():
        confidence.declare(HIDDEN, RANK)
    calls = list(prepare.declared_calls(registry))
    assert calls == [
        (confidence.dspark_confidence_kernel, (HIDDEN, RANK), {}),
        (confidence.dspark_confidence_calibrate_kernel, (HIDDEN, RANK, 5), {}),
    ]


# --- the top-2 margin in the calibrated feature -------------------------------------

STEPS = 5


def _calibrate(hidden, markov, weight, *, margins=None, gain=0.0, slots=None, seed=7):
    """The fused calibration over ``hidden``'s rows: probabilities, and the features
    and predictions it recorded (rows of -7 where it wrote nothing)."""
    rows = hidden.shape[0]
    generator = torch.Generator(device="cuda").manual_seed(seed)
    slope = torch.rand(1, generator=generator, device="cuda") + 0.5
    intercepts = torch.randn(STEPS, generator=generator, device="cuda")
    if slots is None:
        slots = torch.arange(rows, dtype=torch.int32, device="cuda") // STEPS
    num_slots = rows // STEPS + 2
    features = torch.full((num_slots, STEPS), -7.0, device="cuda")
    predictions = torch.full((num_slots, STEPS), -7.0, device="cuda")
    out = torch.empty(rows, dtype=torch.float32, device="cuda")
    confidence.dspark_confidence_calibrate(
        hidden, markov, weight, out, slope, intercepts, slots, features, predictions,
        margins, gain,
    )  # fmt: skip
    return out, features, predictions, slope, intercepts


def _margin_reference(hidden, markov, weight, margins, gain, slope, intercepts):
    """The feature and probability in FP64 from the kernel's own FP32 head
    probability."""
    p = _tilelang(hidden, markov, weight).double().clamp(1e-6, 1 - 1e-6)
    margin = torch.nan_to_num(margins.double(), nan=0.0).clamp(0.0, confidence.MARGIN_CAP)
    feature = (torch.log(p / (1 - p)) + gain * torch.log1p(margin)).clamp(-40.0, 40.0)
    step = torch.arange(hidden.shape[0], device="cuda") % STEPS
    return feature, torch.sigmoid(slope.double() * feature + intercepts.double()[step])


@pytest.mark.parametrize("rows", [5, 10, 80])
def test_zero_margins_give_the_calibration_kernels_bits(rows):
    hidden, markov, weight = _inputs(rows, seed=11)
    plain = _calibrate(hidden, markov, weight)
    zeros = torch.zeros(rows, device="cuda")
    with_margin = _calibrate(hidden, markov, weight, margins=zeros, gain=2.0)
    for a, b in zip(plain[:3], with_margin[:3]):
        assert torch.equal(a, b)
    # No margin tensor, or no gain, is the calibration kernel itself.
    no_gain = _calibrate(hidden, markov, weight, margins=zeros + 3.0, gain=0.0)
    for a, b in zip(plain[:3], no_gain[:3]):
        assert torch.equal(a, b)


@pytest.mark.parametrize("rows", [5, 10, 80])
@pytest.mark.parametrize("gain", [0.5, 2.0])
def test_margin_feature_and_probability(rows, gain):
    hidden, markov, weight = _inputs(rows, seed=12)
    generator = torch.Generator(device="cuda").manual_seed(13)
    margins = torch.rand(rows, generator=generator, device="cuda") * 12.0
    out, features, predictions, slope, intercepts = _calibrate(
        hidden, markov, weight, margins=margins, gain=gain
    )
    feature, prob = _margin_reference(
        hidden, markov, weight, margins, gain, slope, intercepts
    )
    assert torch.allclose(features[: rows // STEPS].reshape(-1).double(), feature, atol=2e-5, rtol=2e-6)
    assert torch.allclose(out.double(), prob, atol=2e-6)
    assert torch.equal(predictions[: rows // STEPS].reshape(-1), out)
    # A larger margin never lowers the probability (the slope is positive).
    more, *_ = _calibrate(hidden, markov, weight, margins=margins + 1.0, gain=gain)
    assert bool((more >= out).all())


def test_margin_is_bounded_and_a_negative_or_nan_margin_is_zero():
    hidden, markov, weight = _inputs(STEPS, seed=14)
    margins = torch.tensor([INF, -3.0, NAN, 1e30, confidence.MARGIN_CAP], device="cuda")
    out, features, _, slope, intercepts = _calibrate(
        hidden, markov, weight, margins=margins, gain=2.0
    )
    feature, prob = _margin_reference(hidden, markov, weight, margins, 2.0, slope, intercepts)
    assert bool(torch.isfinite(out).all()) and bool(torch.isfinite(features[0]).all())
    assert torch.allclose(features[0].double(), feature, atol=2e-5, rtol=2e-6)
    assert torch.allclose(out.double(), prob, atol=2e-6)
    # The negative and the NaN margin add nothing: the head's own log-odds.
    plain = _calibrate(hidden, markov, weight)
    assert features[0, 1] == plain[1][0, 1] and features[0, 2] == plain[1][0, 2]


def test_margin_rows_independent_of_batch_and_repeatable():
    hidden, markov, weight = _inputs(80, seed=15)
    generator = torch.Generator(device="cuda").manual_seed(16)
    margins = torch.rand(80, generator=generator, device="cuda") * 9.0
    whole = _calibrate(hidden, markov, weight, margins=margins, gain=2.0)
    again = _calibrate(hidden, markov, weight, margins=margins, gain=2.0)
    for a, b in zip(whole[:3], again[:3]):
        assert torch.equal(a, b)
    part = _calibrate(
        hidden[:10].contiguous(), markov[:10].contiguous(), weight,
        margins=margins[:10].contiguous(), gain=2.0,
    )  # fmt: skip
    assert torch.equal(part[0], whole[0][:10])


def test_margin_records_only_live_slots():
    hidden, markov, weight = _inputs(10, seed=17)
    margins = torch.full((10,), 2.5, device="cuda")
    slots = torch.tensor([1, 1, 1, -1, -1, 0, 0, 0, 0, 0], dtype=torch.int32, device="cuda")
    out, features, predictions, *_ = _calibrate(
        hidden, markov, weight, margins=margins, gain=2.0, slots=slots
    )
    assert bool((features[1, 3:] == -7.0).all()) and bool((predictions[1, 3:] == -7.0).all())
    assert torch.equal(predictions[1, :3], out[:3])
    assert torch.equal(predictions[0], out[5:])
    assert bool((features[2:] == -7.0).all())
    assert bool(torch.isfinite(out).all())


def test_margin_refuses_other_layouts():
    hidden, markov, weight = _inputs(STEPS, seed=18)
    with pytest.raises(ValueError):
        _calibrate(hidden, markov, weight, margins=torch.zeros(STEPS + 1, device="cuda"), gain=2.0)
    with pytest.raises(ValueError):
        _calibrate(hidden, markov, weight, margins=torch.zeros(STEPS, device="cuda").double(), gain=2.0)
    with pytest.raises(ValueError):
        _calibrate(hidden, markov, weight, margins=torch.zeros(STEPS, 2, device="cuda")[:, 0], gain=2.0)


def test_the_head_passes_its_margins():
    head = confidence.TileLangConfidenceHead.__new__(confidence.TileLangConfidenceHead)
    torch.nn.Module.__init__(head)
    hidden, markov, weight = _inputs(STEPS, seed=19)
    head.proj = SimpleNamespace(weight=weight.view(1, -1))
    head.hidden_size, head.markov_rank = HIDDEN, RANK
    margins = torch.tensor([0.0, 1.0, 2.0, 4.0, 8.0], device="cuda")
    want, features, predictions, slope, intercepts = _calibrate(
        hidden, markov, weight, margins=margins, gain=2.0
    )
    got_features = torch.full_like(features, -7.0)
    got_predictions = torch.full_like(predictions, -7.0)
    slots = torch.zeros(STEPS, dtype=torch.int32, device="cuda")
    got = head.probability_and_calibrate(
        hidden, markov, None, slope, intercepts, slots, got_features, got_predictions,
        margins, 2.0,
    )  # fmt: skip
    assert torch.equal(got, want)
    assert torch.equal(got_features, features) and torch.equal(got_predictions, predictions)


def test_margin_kernel_declared_only_with_a_gain(monkeypatch):
    monkeypatch.setenv(confidence.MARGIN_GAIN_ENV, "2")
    assert confidence.margin_gain() == 2.0
    registry = JitWarmupRegistry(vllm_config=None)
    with registry.activate():
        confidence.declare(HIDDEN, RANK)
    assert list(prepare.declared_calls(registry)) == [
        (confidence.dspark_confidence_kernel, (HIDDEN, RANK), {}),
        (confidence.dspark_confidence_calibrate_kernel, (HIDDEN, RANK, 5), {}),
        (confidence.dspark_confidence_margin_calibrate_kernel, (HIDDEN, RANK, 5), {}),
    ]
    monkeypatch.setenv(confidence.MARGIN_GAIN_ENV, "-1")
    assert confidence.margin_gain() == 0.0
    monkeypatch.delenv(confidence.MARGIN_GAIN_ENV)
    assert confidence.margin_gain() == 0.0
