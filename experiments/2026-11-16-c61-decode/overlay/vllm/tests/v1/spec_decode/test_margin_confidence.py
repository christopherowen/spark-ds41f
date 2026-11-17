# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The draft's top-2 margin in the calibrated confidence feature: the torch form the
paths outside the fused kernel take, the estimator's start for it, the model's dispatch
and the speculator's confidence step."""

import math
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.worker.gpu.spec_decode import acceptance_estimator as ae


def test_add_margin_is_the_log_odds_sum():
    probs = torch.tensor([[0.5, 0.9, 0.2, 0.0, 0.7]])
    margins = torch.tensor([[0.0, 1.0, 3.0, 5.0, float("nan")]])
    want = probs.clone()
    for i, (p, m) in enumerate(zip(probs[0].tolist(), [0.0, 1.0, 3.0, 5.0, 0.0])):
        want[0, i] = (
            0.0 if p == 0.0 else 1 / (1 + math.exp(-(math.log(p / (1 - p)) + 2.0 * math.log1p(m))))
        )
    ae.add_margin_to_confidence_probs(probs, margins.reshape(-1))
    assert torch.allclose(probs, want, atol=1e-6)
    assert probs[0, 3] == 0.0  # a row the drafter left inactive stays inactive


def test_add_margin_bounds_the_margin():
    probs = torch.full((1, 3), 0.5)
    ae.add_margin_to_confidence_probs(probs, torch.tensor([float("inf"), -4.0, 1e30]))
    capped = 1 / (1 + math.exp(-2.0 * math.log1p(ae._MARGIN_CAP)))
    assert torch.allclose(probs, torch.tensor([[capped, 0.5, capped]]), atol=1e-6)


def test_warm_start_for_the_margin_feature():
    est = ae.OnlineAcceptanceEstimator.for_confidence_head(4, 5, torch.device("cpu"))
    est.reset_for_confidence_head(warm_start=True)
    assert est.slope.item() == pytest.approx(ae._INIT_CONFIDENCE_HEAD_SLOPE)
    assert est.intercepts.tolist() == pytest.approx(list(ae._WARM_CONFIDENCE_HEAD_INTERCEPTS))
    est.reset_for_confidence_head(warm_start=True, with_margin=True)
    assert est.slope.item() == pytest.approx(ae._WARM_MARGIN_CONFIDENCE_SLOPE)
    assert est.intercepts.tolist() == pytest.approx(list(ae._WARM_MARGIN_CONFIDENCE_INTERCEPTS))
    assert est.is_calibrated
    # Without a warm start the margin changes nothing.
    est.reset_for_confidence_head(with_margin=True)
    assert est.slope.item() == pytest.approx(ae._INIT_CONFIDENCE_HEAD_SLOPE)
    assert not est.is_calibrated


def test_the_gain_is_the_kernels():
    pytest.importorskip("tilelang")
    from vllm.models.deepseek_v4_1.tilelang import confidence

    assert ae._CONFIDENCE_MARGIN_GAIN == confidence.MARGIN_GAIN == 2.0
    assert ae._MARGIN_CAP == confidence.MARGIN_CAP


class _Head:
    """A confidence head that takes the fused call and records its arguments."""

    def __init__(self, takes_margin):
        self.calls = []
        if takes_margin:
            self.takes_margin = True

    def probability_and_calibrate(self, *args):
        self.calls.append(args)
        return args[2]


@pytest.mark.parametrize("takes_margin", [True, False])
def test_the_model_passes_the_margin_only_to_a_head_that_takes_it(takes_margin):
    from vllm.models.deepseek_v4_1.nvidia.dspark import DSparkDeepseekV4ForCausalLM as Model

    if not torch.cuda.is_available():
        pytest.skip("the fused call is taken for CUDA rows")
    head = _Head(takes_margin)
    model = SimpleNamespace(model=SimpleNamespace(confidence_head=head))
    model.confidence_takes_margin = Model.confidence_takes_margin.fget(model)
    assert model.confidence_takes_margin is takes_margin
    estimator = SimpleNamespace(
        slope=torch.ones(1), intercepts=torch.zeros(5), features=torch.zeros(1, 5),
        predictions=torch.zeros(1, 5), num_speculative_steps=5,
    )  # fmt: skip
    hidden = torch.zeros(5, 8, device="cuda")
    out = torch.zeros(5, device="cuda")
    slots = torch.zeros(5, dtype=torch.int32, device="cuda")
    margins = torch.ones(5, device="cuda")
    call = Model.compute_and_calibrate_confidence
    call(model, hidden, hidden, out, estimator, slots)
    assert len(head.calls[-1]) == 8
    call(model, hidden, hidden, out, estimator, slots, margins=margins)
    if takes_margin:
        assert len(head.calls[-1]) == 9 and head.calls[-1][8] is margins
    else:
        assert len(head.calls[-1]) == 8


@pytest.mark.parametrize("fused", [True, False])
def test_the_speculator_hands_the_steps_margins_to_the_head(fused):
    """``_compute_confidence_probs`` with a step's margins: the fused kernel's result
    over the same rows, and the same feature (to FP32 rounding) on the path that
    calibrates outside it."""
    if not torch.cuda.is_available():
        pytest.skip("the confidence head's kernels are CUDA kernels")
    pytest.importorskip("tilelang")
    from vllm.models.deepseek_v4_1.nvidia.dspark import DSparkDeepseekV4ForCausalLM as Model
    from vllm.models.deepseek_v4_1.tilelang import confidence
    from vllm.v1.worker.gpu.spec_decode.dspark import speculator as sp

    num_reqs, n_spec, hidden_size, rank = 2, 5, 5120, 256
    generator = torch.Generator(device="cuda").manual_seed(3)
    rows = num_reqs * n_spec
    hidden = torch.randn(rows, hidden_size, generator=generator, device="cuda").bfloat16()
    embeds = torch.randn(num_reqs, n_spec, rank, generator=generator, device="cuda").bfloat16()
    weight = torch.randn(hidden_size + rank, generator=generator, device="cuda") / math.sqrt(hidden_size + rank)
    margin_buffer = torch.rand(4, n_spec, generator=generator, device="cuda") * 9.0

    head = confidence.TileLangConfidenceHead.__new__(confidence.TileLangConfidenceHead)
    torch.nn.Module.__init__(head)
    head.proj = SimpleNamespace(weight=weight.view(1, -1))
    head.hidden_size, head.markov_rank = hidden_size, rank
    model = SimpleNamespace(model=SimpleNamespace(confidence_head=head), confidence_takes_margin=True)
    model.compute_confidence = lambda h, e, out=None: head.probability(h, e, out)
    model.compute_and_calibrate_confidence = lambda *a, **k: Model.compute_and_calibrate_confidence(model, *a, **k)

    def run(margins):
        est = ae.OnlineAcceptanceEstimator.for_confidence_head(4, n_spec, torch.device("cuda"))
        est.reset_for_confidence_head(warm_start=True, with_margin=margins is not None)
        # Request r drafts into state slot r + 1; the unfused path takes an int64 map.
        slots = (torch.arange(rows, device="cuda") // n_spec + 1).to(torch.int32 if fused else torch.int64)
        spec = SimpleNamespace(
            num_speculative_steps=n_spec, sample_idx_mapping=slots, acceptance_estimator=est,
            draft_token_confidence_probs=torch.zeros(4, n_spec, device="cuda"), model=model,
        )  # fmt: skip
        sp._compute_confidence_probs(spec, hidden, hidden, None, embeds, [], num_reqs, n_spec, margins=margins)
        return spec.draft_token_confidence_probs[:num_reqs].clone(), est

    def kernel(margins):
        est = ae.OnlineAcceptanceEstimator.for_confidence_head(4, n_spec, torch.device("cuda"))
        est.reset_for_confidence_head(warm_start=True, with_margin=margins is not None)
        out = torch.empty(rows, device="cuda")
        slots = (torch.arange(rows, device="cuda") // n_spec + 1).to(torch.int32)
        confidence.dspark_confidence_calibrate(
            hidden, embeds.flatten(0, 1), weight, out, est.slope, est.intercepts, slots,
            est.features, est.predictions, None if margins is None else margins.reshape(-1),
        )  # fmt: skip
        return out.view(num_reqs, n_spec), est

    for margins in (None, margin_buffer[:num_reqs]):
        got, got_est = run(margins)
        want, want_est = kernel(margins)
        if fused:
            assert torch.equal(got, want)
            assert torch.equal(got_est.features, want_est.features)
            assert torch.equal(got_est.predictions, want_est.predictions)
        else:
            assert torch.allclose(got, want, atol=1e-5)
            assert torch.allclose(got_est.features, want_est.features, atol=1e-4)
        assert bool((got_est.features[0] == 0).all()) and bool((got_est.features[3] == 0).all())
    with_margin, _ = run(margin_buffer[:num_reqs])
    without, _ = run(None)
    assert not torch.equal(with_margin, without)
