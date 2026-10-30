# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 routing through TileKernels' TileLang top-k gate."""

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("tile_kernels")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from vllm.models.deepseek_v4_1.tilelang import router as tilelang_router  # noqa: E402

LO = 129264  # the target's image sentinel token
SCALE = 1.5  # V4.1's routed scaling factor


def _reference(logits, bias, topk, image=None, bias_vl=None):
    """DeepSeek's V4.1 reference gate (inference/model.py), in fp32."""
    scores = F.softplus(logits).sqrt()
    biased = scores + (
        torch.where(image[:, None], bias_vl, bias) if image is not None else bias
    )
    # A stable descending sort puts the lower expert id first on exact ties,
    # which torch.topk leaves unspecified.
    ids = biased.sort(dim=-1, descending=True, stable=True).indices[:, :topk]
    weights = scores.gather(1, ids)
    weights = weights / (weights.sum(-1, keepdim=True) + 1e-20) * SCALE
    return weights, ids


def _router(experts, topk, bias, bias_vl=None, lo=0):
    return tilelang_router.TileKernelsRouter(
        top_k=topk,
        global_num_experts=experts,
        e_score_correction_bias=bias,
        renormalize=True,
        routed_scaling_factor=SCALE,
        scoring_func="sqrtsoftplus",
        bias_vl=bias_vl,
        image_sentinel_lo=lo,
        image_sentinel_count=1,
    )


@pytest.mark.parametrize("experts,topk", [(384, 6), (128, 3)])
@pytest.mark.parametrize("rows", [1, 6, 96, 8192])
@pytest.mark.parametrize("scale", [0.1, 4.0, 40.0])
def test_routes_like_deepseeks_reference(experts, topk, rows, scale):
    generator = torch.Generator(device="cuda").manual_seed(rows * experts)
    logits = torch.randn(rows, experts, device="cuda", generator=generator) * scale
    bias = torch.randn(experts, device="cuda", generator=generator) * 0.1
    weights, ids = _router(experts, topk, bias)._compute_routing(None, logits, None)
    expected_weights, expected_ids = _reference(logits, bias, topk)
    assert ids.dtype == torch.int64 and weights.dtype == torch.float32
    torch.testing.assert_close(ids, expected_ids, rtol=0, atol=0)
    # Only the order of the six-term sum may differ from torch's.
    torch.testing.assert_close(weights, expected_weights, rtol=2e-6, atol=0)


def test_ties_go_to_the_lowest_expert():
    logits = torch.zeros(4, 384, device="cuda")
    bias = torch.zeros(384, device="cuda")
    weights, ids = _router(384, 6, bias)._compute_routing(None, logits, None)
    assert ids.tolist() == [list(range(6))] * 4
    torch.testing.assert_close(weights, torch.full_like(weights, SCALE / 6))


def test_image_tokens_route_with_the_vision_bias():
    generator = torch.Generator(device="cuda").manual_seed(7)
    logits = torch.randn(64, 384, device="cuda", generator=generator)
    bias = torch.randn(384, device="cuda", generator=generator)
    bias_vl = torch.randn(384, device="cuda", generator=generator)
    input_ids = torch.randint(0, 1000, (64,), device="cuda", generator=generator)
    input_ids[::3] = LO
    router = _router(
        384, 6, bias, torch.nn.Parameter(bias_vl, requires_grad=False), lo=LO
    )
    weights, ids = router._compute_routing(None, logits, None, input_ids=input_ids)
    expected_weights, expected_ids = _reference(
        logits, bias, 6, input_ids == LO, bias_vl
    )
    torch.testing.assert_close(ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(weights, expected_weights, rtol=2e-6, atol=0)
    with pytest.raises(ValueError, match="requires input_ids"):
        router._compute_routing(None, logits, None)


def test_image_token_mask_is_computed_once_per_step(monkeypatch):
    context = type("Context", (), {})()
    monkeypatch.setattr(tilelang_router, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(tilelang_router, "get_forward_context", lambda: context)
    input_ids = torch.tensor([1, LO, LO + 1, LO], device="cuda")
    first = tilelang_router._image_token_mask(input_ids, LO, 1)
    assert first.tolist() == [False, True, False, True]
    assert tilelang_router._image_token_mask(input_ids, LO, 1) is first
    # The next step's ids are a new tensor.
    other = tilelang_router._image_token_mask(input_ids.clone(), LO, 1)
    assert other is not first and torch.equal(other, first)


def test_refuses_what_v41_routing_does_not_have():
    bias = torch.zeros(384, device="cuda")
    table = torch.zeros(1000, 6, dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="no hash-routed layers"):
        tilelang_router.TileKernelsRouter(
            top_k=6,
            global_num_experts=384,
            e_score_correction_bias=bias,
            renormalize=True,
            scoring_func="sqrtsoftplus",
            hash_indices_table=table,
        )
    logits = torch.randn(2, 384, device="cuda")
    with pytest.raises(ValueError, match="int64 expert ids"):
        _router(384, 6, bias)._compute_routing(None, logits, torch.int32)


def test_matches_the_triton_router_it_replaces():
    """Same experts as vLLM's dsv4_topk; weights within its approximate math."""
    from vllm.model_executor.layers.fused_moe.router.dsv4_topk import dsv4_topk

    generator = torch.Generator(device="cuda").manual_seed(11)
    logits = torch.randn(512, 384, device="cuda", generator=generator) * 3
    bias = torch.randn(384, device="cuda", generator=generator) * 0.1
    weights, ids = _router(384, 6, bias)._compute_routing(None, logits, None)
    triton_weights, triton_ids = dsv4_topk(logits, bias, torch.int64, SCALE)
    torch.testing.assert_close(ids, triton_ids, rtol=0, atol=0)
    torch.testing.assert_close(weights, triton_weights, rtol=1e-5, atol=0)


class _Gate(torch.nn.Module):
    """A V4.1 router gate on the TileLang BF16 projection, FP32 out."""

    def __init__(self, experts, hidden=5120, seed=0):
        from vllm.models.deepseek_v4_1.tilelang.linear import TileLangLinearMethod

        super().__init__()
        generator = torch.Generator(device="cuda").manual_seed(seed)
        weight = torch.randn(experts, hidden, device="cuda", generator=generator) * 0.02
        self.weight = torch.nn.Parameter(weight.to(torch.bfloat16), requires_grad=False)
        self.out_dtype = torch.float32
        self.quant_method = TileLangLinearMethod()
        self.quant_method.declare_layer_kernels(self)  # standalone: prepares now

    def forward(self, x):
        return self.quant_method.apply(self, x), None


@pytest.fixture
def plain_scratch(monkeypatch):
    from vllm.models.deepseek_v4_1.tilelang import linear

    monkeypatch.setattr(
        linear,
        "_scratch",
        lambda specs, scratch: [
            torch.empty(s, dtype=d, device="cuda") for s, d in specs
        ],
    )


@pytest.mark.parametrize("experts,topk", [(384, 6), (128, 3)])
@pytest.mark.parametrize("rows", [1, 6, 17, 64, 65, 512])
def test_gate_router_matches_the_gate_then_the_router(
    plain_scratch, experts, topk, rows
):
    """Decode rows fuse the gate's split-K reduction into the router, bit for bit."""
    gate = _Gate(experts, seed=experts)
    assert gate.tilelang_shards > 1
    generator = torch.Generator(device="cuda").manual_seed(rows)
    x = torch.randn(rows, 5120, device="cuda", generator=generator).to(torch.bfloat16)
    bias = torch.randn(experts, device="cuda", generator=generator) * 0.1
    kwargs = dict(
        top_k=topk,
        global_num_experts=experts,
        e_score_correction_bias=bias,
        renormalize=True,
        routed_scaling_factor=SCALE,
        scoring_func="sqrtsoftplus",
    )
    fused = tilelang_router.TileLangGateRouter(gate=gate, **kwargs)
    fused.declare_kernels()
    weights, ids = fused._compute_routing(None, x, None)
    logits, _ = gate(x)
    expected = tilelang_router.TileKernelsRouter(**kwargs)._compute_routing(
        None, logits, None
    )
    assert torch.equal(ids, expected[1])
    assert torch.equal(weights.view(torch.int32), expected[0].view(torch.int32))


def test_gate_router_routes_image_tokens(plain_scratch):
    gate = _Gate(384, seed=3)
    generator = torch.Generator(device="cuda").manual_seed(5)
    x = torch.randn(48, 5120, device="cuda", generator=generator).to(torch.bfloat16)
    bias = torch.randn(384, device="cuda", generator=generator) * 0.1
    bias_vl = torch.nn.Parameter(
        torch.randn(384, device="cuda", generator=generator) * 0.1, requires_grad=False
    )
    input_ids = torch.randint(0, 1000, (48,), device="cuda", generator=generator)
    input_ids[1::4] = LO
    kwargs = dict(
        top_k=6,
        global_num_experts=384,
        e_score_correction_bias=bias,
        renormalize=True,
        routed_scaling_factor=SCALE,
        scoring_func="sqrtsoftplus",
        bias_vl=bias_vl,
        image_sentinel_lo=LO,
        image_sentinel_count=1,
    )
    fused = tilelang_router.TileLangGateRouter(gate=gate, **kwargs)
    fused.declare_kernels()
    weights, ids = fused._compute_routing(None, x, None, input_ids=input_ids)
    logits, _ = gate(x)
    expected = tilelang_router.TileKernelsRouter(**kwargs)._compute_routing(
        None, logits, None, input_ids=input_ids
    )
    assert torch.equal(ids, expected[1])
    assert torch.equal(weights.view(torch.int32), expected[0].view(torch.int32))


@pytest.fixture
def padded_step(monkeypatch):
    """A forward step whose last rows pad a CUDA graph's batch."""
    context = type("Context", (), {})()
    context.attn_metadata = None  # ForwardContext always has it; no CED plan here
    monkeypatch.setattr(tilelang_router, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(tilelang_router, "get_forward_context", lambda: context)

    def pad(rows, padding):
        context.is_padding = torch.zeros(rows, dtype=torch.bool, device="cuda")
        context.is_padding[rows - padding :] = True
        return context.is_padding

    return pad


@pytest.mark.parametrize("gate_router", [False, True])
def test_padding_rows_are_not_routed(plain_scratch, padded_step, gate_router):
    rows, padding = 6, 2
    is_padding = padded_step(rows, padding)
    generator = torch.Generator(device="cuda").manual_seed(13)
    x = torch.randn(rows, 5120, device="cuda", generator=generator).to(torch.bfloat16)
    bias = torch.randn(384, device="cuda", generator=generator) * 0.1
    gate = _Gate(384, seed=13)
    kwargs = dict(
        top_k=6,
        global_num_experts=384,
        e_score_correction_bias=bias,
        renormalize=True,
        routed_scaling_factor=SCALE,
        scoring_func="sqrtsoftplus",
    )
    logits, _ = gate(x)
    if gate_router:
        router = tilelang_router.TileLangGateRouter(gate=gate, **kwargs)
        router.declare_kernels()
        weights, ids = router._compute_routing(None, x, None)
    else:
        weights, ids = tilelang_router.TileKernelsRouter(**kwargs)._compute_routing(
            None, logits, None
        )
    assert (ids[is_padding] == -1).all() and (weights[is_padding] == 0).all()
    expected_weights, expected_ids = _reference(logits[~is_padding], bias, 6)
    torch.testing.assert_close(ids[~is_padding], expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(
        weights[~is_padding], expected_weights, rtol=2e-6, atol=0
    )
    if gate_router:
        unfused = tilelang_router.TileKernelsRouter(**kwargs)._compute_routing(
            None, logits, None
        )
        assert torch.equal(ids, unfused[1])
        assert torch.equal(weights.view(torch.int32), unfused[0].view(torch.int32))


@pytest.fixture
def ced_step(monkeypatch):
    """A mixed step whose CED decoder batch is shorter than its CPU plan.

    Five decode requests are scheduled 6 rows each and a prefill 331 rows,
    3,328 of them cached: 361 rows. Adaptive verification shortens the decode
    ranges to 22 rows, so the decoder keeps 22 decode rows and the prefill's
    last 128, and the plan's 158 rows end in eight -1 indices. Rows 1, 7 and
    20 are dead verification rows inside the batch.
    """
    from vllm.models.deepseek_v4_1.ced import CEDState, gather_rows

    state = CEDState(6, 4096, (256, 512, 1024, 2048, 4096), "cuda")
    starts = torch.tensor([0, 4, 8, 12, 16, 22, 353], dtype=torch.int32, device="cuda")
    seq = torch.tensor([4000] * 5 + [3659], dtype=torch.int32, device="cuda")
    state.stage(starts, seq, [6] * 5 + [331], [False] * 6, [0] * 5 + [3328])
    indices = state.get_indices()
    is_padding = torch.zeros(361, dtype=torch.bool, device="cuda")
    is_padding[[1, 7, 20]] = True
    is_padding[353:] = True
    # What DeepseekV41ModelState.prepare_attn puts on the decoder metadata.
    compact = (indices >= 0) & ~gather_rows(is_padding, indices)
    decoder = type("Decoder", (), {"routing_mask": compact})()
    layer = type("Layer", (), {"decoder": decoder})()
    context = type("Context", (), {})()
    context.is_padding = is_padding
    context.attn_metadata = {"layer": layer}
    monkeypatch.setattr(tilelang_router, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(tilelang_router, "get_forward_context", lambda: context)
    return context, indices, compact


def test_ced_decoder_routes_its_gathered_rows(ced_step):
    context, indices, compact = ced_step
    is_padding = context.is_padding
    assert indices.tolist() == list(range(22)) + list(range(225, 353)) + [-1] * 8
    expected = torch.ones(158, dtype=torch.bool, device="cuda")
    expected[[1, 7, 20]] = False
    expected[150:] = False
    assert torch.equal(compact, expected)
    # Truncating the step's mask, as before, keeps the wrong rows.
    assert not torch.equal((~is_padding)[:158], expected)
    assert torch.equal(tilelang_router._routing_mask(158, ced_decoder=True), expected)
    # Encoder and draft routers keep the step's layout, even at the same size.
    assert torch.equal(tilelang_router._routing_mask(361), ~is_padding)
    assert torch.equal(tilelang_router._routing_mask(158), (~is_padding)[:158])


def test_ced_decoder_skips_nan_padding_rows(ced_step):
    _, _, expected = ced_step
    bias = torch.zeros(384, device="cuda")
    router = tilelang_router.TileKernelsRouter(
        top_k=6,
        global_num_experts=384,
        e_score_correction_bias=bias,
        renormalize=True,
        routed_scaling_factor=1.0,
        scoring_func="sqrtsoftplus",
    )
    router._ds41_is_ced_decoder = True
    logits = torch.zeros(158, 384, device="cuda")
    logits[~expected] = float("nan")  # what the gate leaves in padding rows
    weights, ids = router._compute_routing(None, logits, None)
    torch.cuda.synchronize()
    assert (ids[~expected] == -1).all() and (weights[~expected] == 0).all()
    assert (ids[expected] >= 0).all() and torch.isfinite(weights).all()


def test_gate_router_reads_the_padding_and_ced_masks_itself(plain_scratch, ced_step):
    """The fused decode router takes the step's padding mask as it is (a set row
    is skipped) and a CED decoder's routing mask (a clear row is skipped), the
    rows the unfused router routes, bit for bit."""
    context, _, compact = ced_step
    generator = torch.Generator(device="cuda").manual_seed(29)
    bias = torch.randn(384, device="cuda", generator=generator) * 0.1
    kwargs = dict(
        top_k=6,
        global_num_experts=384,
        e_score_correction_bias=bias,
        renormalize=True,
        routed_scaling_factor=SCALE,
        scoring_func="sqrtsoftplus",
    )
    for rows, ced, skipped in ((361, False, context.is_padding), (158, True, ~compact)):
        gate = _Gate(384, seed=rows)
        x = torch.randn(rows, 5120, device="cuda", generator=generator)
        x = x.to(torch.bfloat16)
        router = tilelang_router.TileLangGateRouter(gate=gate, **kwargs)
        router._ds41_is_ced_decoder = ced
        router.declare_kernels()
        weights, ids = router._compute_routing(None, x, None)
        logits, _ = gate(x)
        unfused = tilelang_router.TileKernelsRouter(**kwargs)
        unfused._ds41_is_ced_decoder = ced
        expected = unfused._compute_routing(None, logits, None)
        assert (ids[skipped] == -1).all()
        assert torch.equal(ids, expected[1])
        assert torch.equal(weights.view(torch.int32), expected[0].view(torch.int32))


# --- The forward's routing masks ----------------------------------------------


@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("rows", [1, 6, 255, 256, 257, 16384])
@pytest.mark.parametrize("count", [1, 3])
def test_routing_masks_are_torchs(ids_dtype, rows, count):
    """The image tokens and the rows to route, in one launch or each alone, as
    torch's comparisons and negation gave them: ids at the sentinels' edges and
    across the vocabulary, every padding pattern."""
    generator = torch.Generator(device="cuda").manual_seed(rows * count)
    ids = torch.randint(LO - 3, LO + 5, (rows,), device="cuda", generator=generator)
    ids[::7] = torch.randint(0, 129280, ids[::7].shape, device="cuda")
    ids = ids.to(ids_dtype)
    is_padding = torch.rand(rows, device="cuda", generator=generator) < 0.3
    before = (ids.clone(), is_padding.clone())
    expected_image = (ids >= LO) & (ids < LO + count)
    image, routed = tilelang_router._masks(ids, LO, count, is_padding)
    assert image.dtype == routed.dtype == torch.bool
    assert torch.equal(image, expected_image)
    assert torch.equal(routed, ~is_padding)
    image_alone, none = tilelang_router._masks(ids, LO, count, None)
    assert none is None and torch.equal(image_alone, expected_image)
    none, routed_alone = tilelang_router._masks(None, 0, 0, is_padding)
    assert none is None and torch.equal(routed_alone, ~is_padding)
    assert torch.equal(ids, before[0]) and torch.equal(is_padding, before[1])


def test_a_forward_computes_its_routing_masks_once(padded_step, monkeypatch):
    """The first routing layer of a forward computes the image tokens and the rows
    to route in one launch; later layers, and the per-mask helpers, find them; a
    new tensor of ids (a CED decoder's compact rows) takes one launch of its own."""
    launches = []
    kernel = tilelang_router.routing_masks_kernel

    def counted(*key):
        launches.append(key)
        return kernel(*key)

    monkeypatch.setattr(tilelang_router, "routing_masks_kernel", counted)
    is_padding = padded_step(96, 5)
    ids = torch.full((96,), 7, dtype=torch.int32, device="cuda")
    ids[3::4] = LO
    image, route = tilelang_router._routing_masks(96, ids, LO, 1)
    assert launches == [("int32", True)]
    assert torch.equal(image, ids == LO) and torch.equal(route, ~is_padding)
    again = tilelang_router._routing_masks(96, ids, LO, 1)
    assert again[0] is image and torch.equal(again[1], route)
    assert tilelang_router._image_token_mask(ids, LO, 1) is image
    assert torch.equal(tilelang_router._routing_mask(96), route)
    assert torch.equal(tilelang_router._routing_masks(40, None, LO, 1)[1], route[:40])
    assert len(launches) == 1
    compact = ids[::2].clone()
    image2, route2 = tilelang_router._routing_masks(48, compact, LO, 1)
    assert launches[1:] == [("int32", False)]
    assert torch.equal(image2, compact == LO) and torch.equal(route2, route[:48])


@pytest.mark.parametrize("vision", [False, True])
def test_gate_router_declares_its_masks(
    plain_scratch, padded_step, monkeypatch, vision
):
    """The gate router's prefill rows (past its split rows) compute the masks it
    declares: the image tokens with the rows to route for a vision layer, the rows
    to route for a text layer."""
    gate = _Gate(384, seed=17)
    generator = torch.Generator(device="cuda").manual_seed(17)
    rows = gate.tilelang_split_rows + 9
    padded_step(rows, 3)
    x = torch.randn(rows, 5120, device="cuda", generator=generator).to(torch.bfloat16)
    bias = torch.randn(384, device="cuda", generator=generator) * 0.1
    kwargs = dict(
        top_k=6,
        global_num_experts=384,
        e_score_correction_bias=bias,
        renormalize=True,
        routed_scaling_factor=SCALE,
        scoring_func="sqrtsoftplus",
    )
    input_ids = None
    if vision:
        kwargs.update(
            bias_vl=torch.nn.Parameter(bias.clone(), requires_grad=False),
            image_sentinel_lo=LO,
            image_sentinel_count=1,
        )
        input_ids = torch.randint(0, 1000, (rows,), device="cuda", generator=generator)
    router = tilelang_router.TileLangGateRouter(gate=gate, **kwargs)
    router.declare_kernels()  # standalone: prepares now
    declared = {args for _, args, _ in router._mask_calls()}
    assert declared == (
        {("int32", False), ("int32", True), ("int64", False), ("int64", True)}
        | {(None, True)}
        if vision
        else {(None, True)}
    )
    launches = []
    kernel = tilelang_router.routing_masks_kernel

    def counted(*key):
        launches.append(key)
        return kernel(*key)

    monkeypatch.setattr(tilelang_router, "routing_masks_kernel", counted)
    router._compute_routing(None, x, None, input_ids=input_ids)
    assert launches == [("int64", True) if vision else (None, True)]
    assert set(launches) <= declared


def test_provide_routes_scopes_handed_over_routes():
    rows = torch.empty((8, 5120), dtype=torch.bfloat16, device="cuda")
    other = torch.empty((8, 5120), dtype=torch.bfloat16, device="cuda")
    weights = torch.ones((8, 6), dtype=torch.float32, device="cuda")
    ids = torch.zeros((8, 6), dtype=torch.int64, device="cuda")
    assert tilelang_router.provided_routes(rows) is None
    with tilelang_router.provide_routes(rows, weights, ids):
        assert tilelang_router.provided_routes(rows) == (weights, ids)
        assert tilelang_router.provided_routes(other) is None
    assert tilelang_router.provided_routes(rows) is None


@pytest.mark.parametrize("num_tokens", [150, 205])
@pytest.mark.parametrize("vision", [False, True])
def test_gate_router_route_local_matches_full_rows(
    plain_scratch, padded_step, num_tokens, vision
):
    """Sequence-parallel local routing across ranks matches full-batch routing
    bit-for-bit across both the split-K partials path (local_rows <= 64) and the
    full-gate path (local_rows > 64), including vision tokens and padding masks."""
    from vllm.models.deepseek_v4_1 import sp_prefill

    world_size = 3
    padded_step(num_tokens, 4)
    gate = _Gate(384, seed=num_tokens)
    generator = torch.Generator(device="cuda").manual_seed(num_tokens + int(vision))
    x = torch.randn(num_tokens, 5120, device="cuda", generator=generator).to(
        torch.bfloat16
    )
    bias = torch.randn(384, device="cuda", generator=generator) * 0.1
    kwargs = dict(
        top_k=6,
        global_num_experts=384,
        e_score_correction_bias=bias,
        renormalize=True,
        routed_scaling_factor=SCALE,
        scoring_func="sqrtsoftplus",
    )
    input_ids = None
    if vision:
        kwargs.update(
            bias_vl=torch.nn.Parameter(
                torch.randn(384, device="cuda", generator=generator) * 0.1,
                requires_grad=False,
            ),
            image_sentinel_lo=LO,
            image_sentinel_count=1,
        )
        input_ids = torch.randint(
            0, 1000, (num_tokens,), device="cuda", generator=generator
        )
        input_ids[2::5] = LO
    router = tilelang_router.TileLangGateRouter(gate=gate, **kwargs)
    router.declare_kernels()
    expected_weights, expected_ids = router._compute_routing(
        None, x, None, input_ids=input_ids
    )

    rank_weights = []
    rank_ids = []
    for rank in range(world_size):
        layout = sp_prefill.SPRows(num_tokens, world_size, rank)
        local_x = layout.local_slice(x)
        w, idx = router.route_local(local_x, layout, input_ids=input_ids)
        assert w.shape == (layout.local_rows, 6)
        assert idx.shape == (layout.local_rows, 6)
        assert torch.isfinite(w).all()
        rank_weights.append(w)
        rank_ids.append(idx)

    gathered_weights = torch.cat(rank_weights, dim=0)[:num_tokens]
    gathered_ids = torch.cat(rank_ids, dim=0)[:num_tokens]
    assert torch.equal(gathered_ids, expected_ids)
    assert torch.equal(
        gathered_weights.view(torch.int32), expected_weights.view(torch.int32)
    )

    # When handed over via provide_routes, _compute_routing returns the gathered
    # routes directly without touching the gate or reading the BF16 handle.
    handle = torch.empty_like(x)
    with tilelang_router.provide_routes(handle, gathered_weights, gathered_ids):
        handed_w, handed_ids = router._compute_routing(
            None, handle, None, input_ids=input_ids
        )
    assert handed_w is gathered_weights and handed_ids is gathered_ids
