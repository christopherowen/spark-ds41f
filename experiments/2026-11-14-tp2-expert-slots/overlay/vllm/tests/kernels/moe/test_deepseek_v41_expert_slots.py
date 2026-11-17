# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the zero-copy ``pin_memory`` + ``io_uring`` (``READ_FIXED``) MoE expert slot cache."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v4_1.expert_slots import (
    DEFAULT_PIN_ALIGN,
    MIN_FREE_DISK_BYTES,
    PACK_HEADER_BYTES,
    PAGE_BYTES,
    CountDecayLFU,
    ExpertDiskCache,
    ExpertPackMeta,
    ExpertSlotCache,
    RouterLookaheadPredictor,
    alloc_pinned_aligned,
    cleanup_stale_tmp_files,
    compute_uma_slot_budget,
    pack_layer_experts,
    plan_layer_slot_budgets,
    rank_experts_by_router_bias,
    select_spot_check_experts,
    verify_spot_experts,
    write_layer_expert_file,
)
from vllm.models.deepseek_v4_1.tilelang import linear, prepare
from vllm.models.deepseek_v4_1.tilelang.moe import MoeExpertsV41Plan


def _make_random_weights(
    num_experts: int, hidden: int, intermediate: int, *, seed: int = 17
):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    w13 = torch.randint(
        -128,
        127,
        (num_experts, 2 * intermediate, hidden // 2),
        dtype=torch.int8,
        generator=gen,
    ).cuda()
    s13_base = torch.randint(
        118, 124, (num_experts, 2 * intermediate, 1), dtype=torch.uint8, generator=gen
    )
    s13_delta = torch.randint(
        0,
        7,
        (num_experts, 2 * intermediate, hidden // 32),
        dtype=torch.uint8,
        generator=gen,
    )
    s13 = (s13_base + s13_delta).cuda()

    w2 = torch.randint(
        -128,
        127,
        (num_experts, hidden, intermediate // 2),
        dtype=torch.int8,
        generator=gen,
    ).cuda()
    s2_base = torch.randint(
        118, 124, (num_experts, hidden, 1), dtype=torch.uint8, generator=gen
    )
    s2_delta = torch.randint(
        0,
        7,
        (num_experts, hidden, intermediate // 32),
        dtype=torch.uint8,
        generator=gen,
    )
    s2 = (s2_base + s2_delta).cuda()
    return w13, s13, w2, s2


def test_zero_copy_pinned_allocation_and_alignment():
    host, cuda = alloc_pinned_aligned((1024, 128), torch.uint16)
    assert host.data_ptr() == cuda.data_ptr()
    assert host.data_ptr() % DEFAULT_PIN_ALIGN == 0
    assert cuda.is_cuda

    host.fill_(0x1234)
    doubled = (cuda.to(torch.int32) * 2).cpu()
    assert torch.all(doubled == 0x2468)


def test_count_decay_lfu_replacement_and_cold_prefill_protection():
    lfu = CountDecayLFU(
        num_experts=16,
        num_slots=8,
        static_slots=4,
        half_life_steps=16.0,
        hot_threshold=1.5,
    )
    # Static slots 0..3 are permanently pinned to experts 0..3.
    # Dynamic slots 4..7 hold experts 4..7.
    # Warm experts 4, 5, 6 as decode-hot; leave expert 7 cold.
    for _ in range(4):
        lfu.record_step([0, 4, 5, 6], is_prefill=False)

    # A demand miss for expert 8 must evict slot 7 (lowest decayed score).
    slot8 = lfu.allocate_dynamic_slot(8, protected_experts={0, 4, 5, 6, 8})
    assert slot8 == 7
    assert lfu.expert_to_slot[7] == -1
    assert lfu.expert_to_slot[8] == 7

    # During a cold prefill step, decode-hot slots (4, 5, 6 with score > 1.5)
    # are protected while slot 7 (holding cold expert 8) is chosen first.
    slot9 = lfu.allocate_dynamic_slot(
        9, protected_experts={0, 9}, is_prefill=True
    )
    assert slot9 == 7
    assert lfu.expert_to_slot[4] == 4
    assert lfu.expert_to_slot[5] == 5
    assert lfu.expert_to_slot[6] == 6


@pytest.mark.parametrize("intermediate", [128, 192])
def test_pack_and_role_major_io_uring_read_fixed_bit_exact(
    tmp_path: Path, intermediate: int
):
    num_experts = 16
    num_slots = 12
    static_slots = 6
    topk = 4
    hidden = 256

    w13, s13, w2, s2 = _make_random_weights(
        num_experts, hidden, intermediate, seed=101 + intermediate
    )

    ref_plan = MoeExpertsV41Plan(
        num_experts,
        topk,
        hidden,
        intermediate,
        swiglu_limit=7.0,
        out_dtype=torch.bfloat16,
    )
    ref_plan.declare_kernels()
    ref_weights = ref_plan.prepare_weights(w13, s13, w2, s2)

    slot_plan = MoeExpertsV41Plan(
        num_experts,
        topk,
        hidden,
        intermediate,
        swiglu_limit=7.0,
        out_dtype=torch.bfloat16,
        num_slots=num_slots,
        narrow_codes=False,
    )
    slot_plan.declare_kernels()
    slot_prepared_all = slot_plan.prepare_weights(w13, s13, w2, s2)

    meta = ExpertPackMeta.from_plan(slot_plan)
    assert meta.w13_tiles_slot_bytes % PAGE_BYTES == 0
    assert meta.w2_tiles_slot_bytes % PAGE_BYTES == 0
    assert meta.expert_stride % PAGE_BYTES == 0

    pack_path = pack_layer_experts(
        tmp_path / f"layer_i{intermediate}.pack",
        meta,
        slot_prepared_all[0],
        slot_prepared_all[1],
    )
    cache = ExpertSlotCache(
        meta,
        pack_path,
        num_slots=num_slots,
        static_slots=static_slots,
        prefill_protect_rows=8,
    )
    try:
        rstats = cache.reader_stats()
        assert rstats["buffers_registered"] is True

        # 1. Decode step hitting both static (0..5) and cold (10..15) experts.
        rows = 5
        x_bf16 = torch.randn(rows, hidden, dtype=torch.bfloat16, device="cuda")
        x_q = linear.cast_mxfp8(x_bf16)
        topk_ids = torch.tensor(
            [
                [0, 1, 10, 11],
                [2, 12, 13, 3],
                [4, 5, 14, 15],
                [10, 12, 14, 0],
                [11, 13, 15, 1],
            ],
            dtype=torch.int64,
            device="cuda",
        )
        topk_w = torch.softmax(
            torch.randn(rows, topk, dtype=torch.float32, device="cuda"), dim=-1
        )
        shared = torch.randn(rows, hidden, dtype=torch.bfloat16, device="cuda")

        out_ref = torch.empty(rows, hidden, dtype=torch.bfloat16, device="cuda")
        out_slot = torch.empty(rows, hidden, dtype=torch.bfloat16, device="cuda")

        ref_plan.run(x_q, topk_ids, topk_w, ref_weights, out_ref, shared=shared)
        cache.run(slot_plan, x_q, topk_ids, topk_w, out_slot, shared=shared)
        torch.cuda.synchronize()

        assert torch.equal(out_ref, out_slot)
        assert cache.misses == 4
        misses_after_first = cache.misses

        # 2. Repeat the same step: all 6 cold experts are now resident in dynamic
        # slots (only 4 dynamic slots exist, so 2 will miss and 4 will hit).
        out_slot2 = torch.empty(rows, hidden, dtype=torch.bfloat16, device="cuda")
        cache.run(slot_plan, x_q, topk_ids, topk_w, out_slot2, shared=shared)
        torch.cuda.synchronize()
        assert torch.equal(out_ref, out_slot2)
        assert cache.misses == misses_after_first

        # 3. Cross-layer router lookahead prefetch test.
        gate_w = torch.randn(num_experts, hidden, dtype=torch.bfloat16, device="cuda")
        gate_b = torch.zeros(num_experts, dtype=torch.float32, device="cuda")
        predictor = RouterLookaheadPredictor(gate_w, gate_b, top_k=topk)
        pred_ids = predictor.predict_experts(x_bf16[:2])
        cache.prefetch_async(pred_ids)
        # Immediately run with those predicted experts.
        pred_tensor = torch.tensor(
            [pred_ids[:topk]], dtype=torch.int64, device="cuda"
        )
        w_pred = torch.full((1, topk), 1.0 / topk, dtype=torch.float32, device="cuda")
        x1_q = linear.cast_mxfp8(x_bf16[:1])
        out_ref_la = torch.empty(1, hidden, dtype=torch.bfloat16, device="cuda")
        out_slot_la = torch.empty(1, hidden, dtype=torch.bfloat16, device="cuda")
        ref_plan.run(x1_q, pred_tensor, w_pred, ref_weights, out_ref_la)
        cache.run(slot_plan, x1_q, pred_tensor, w_pred, out_slot_la)
        torch.cuda.synchronize()
        assert torch.equal(out_ref_la, out_slot_la)

        # 4. Multi-window prefill batch (>512 routes, multi-CTA dispatch) touching
        # all 16 experts when only 4 dynamic slots exist.
        p_rows = 140  # 140 * 4 = 560 routes (> _CHUNK=512)
        xp_bf16 = torch.randn(p_rows, hidden, dtype=torch.bfloat16, device="cuda")
        xp_q = linear.cast_mxfp8(xp_bf16)
        p_ids = (
            torch.arange(p_rows * topk, dtype=torch.int64, device="cuda").view(
                p_rows, topk
            )
            % num_experts
        )
        p_w = torch.softmax(
            torch.randn(p_rows, topk, dtype=torch.float32, device="cuda"), dim=-1
        )
        out_ref_p = torch.empty(p_rows, hidden, dtype=torch.bfloat16, device="cuda")
        out_slot_p = torch.empty(p_rows, hidden, dtype=torch.bfloat16, device="cuda")
        ref_plan.run(xp_q, p_ids, p_w, ref_weights, out_ref_p)
        cache.run(slot_plan, xp_q, p_ids, p_w, out_slot_p)
        torch.cuda.synchronize()
        assert torch.equal(out_ref_p, out_slot_p)
        assert cache.prefill_windows > 1
    finally:
        cache.close()


def test_tilelang_experts_layer_integration_with_slot_cache(tmp_path: Path):
    from vllm.model_executor.layers.fused_moe import tilelang as tl_moe
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    num_experts = 12
    num_slots = 8
    static_slots = 5
    topk = 3
    hidden = 256
    intermediate = 128

    w13, s13, w2, s2 = _make_random_weights(
        num_experts, hidden, intermediate, seed=303
    )

    def _build_experts(slot_cfg):
        layer = SimpleNamespace(
            apply_router_weight_on_input=False,
            w13_weight=w13.clone(),
            w13_weight_scale=s13.clone(),
            w2_weight=w2.clone(),
            w2_weight_scale=s2.clone(),
            slot_cache_config=slot_cfg,
        )
        ex = tl_moe.TileLangExperts.__new__(tl_moe.TileLangExperts)
        ex.quant_config = SimpleNamespace(
            gemm1_clamp_limit=None,
            _w1=SimpleNamespace(scale=None),
            _w2=SimpleNamespace(scale=None),
        )
        ex.moe_config = SimpleNamespace(
            experts_per_token=topk,
            swiglu_limit=7.0,
            in_dtype=torch.bfloat16,
        )
        ex.fold_shared = False
        ex.defer_combine = False
        ex._slot_cache = None
        ex._next_lookahead = None
        ex.process_weights_after_loading(layer)
        ex.quant_config.w1_scale = ex.quant_config._w1.scale
        ex.quant_config.w2_scale = ex.quant_config._w2.scale
        ex.declare_kernels()
        return ex, layer

    ref_ex, ref_layer = _build_experts(None)
    slot_ex, slot_layer = _build_experts(
        {
            "pack_path": tmp_path / "tl_experts.pack",
            "num_slots": num_slots,
            "static_slots": static_slots,
        }
    )
    try:
        rows = 4
        x_bf16 = torch.randn(rows, hidden, dtype=torch.bfloat16, device="cuda")
        topk_ids = torch.tensor(
            [[0, 6, 11], [1, 7, 10], [2, 8, 9], [5, 10, 11]],
            dtype=torch.int64,
            device="cuda",
        )
        topk_w = torch.softmax(
            torch.randn(rows, topk, dtype=torch.float32, device="cuda"), dim=-1
        )
        ws_shape = ref_ex.workspace_shapes(
            rows, 2 * intermediate, hidden, topk, num_experts, num_experts, None, MoEActivation.SILU
        )[1]
        ws = torch.empty(ws_shape, dtype=torch.bfloat16, device="cuda")

        out_ref = torch.empty(rows, hidden, dtype=torch.float32, device="cuda")
        out_slot = torch.empty(rows, hidden, dtype=torch.float32, device="cuda")

        with linear.provide_mxfp8(linear.cast_mxfp8(x_bf16)):
            ref_ex.apply(
                out_ref,
                x_bf16,
                ref_layer.w13_weight,
                ref_layer.w2_weight,
                topk_w,
                topk_ids,
                MoEActivation.SILU,
                num_experts,
                None,
                None,
                None,
                None,
                ws,
                None,
                False,
            )
            slot_ex.apply(
                out_slot,
                x_bf16,
                slot_layer.w13_weight,
                slot_layer.w2_weight,
                topk_w,
                topk_ids,
                MoEActivation.SILU,
                num_experts,
                None,
                None,
                None,
                None,
                ws,
                None,
                False,
            )
        torch.cuda.synchronize()
        assert torch.equal(out_ref, out_slot)
    finally:
        if slot_ex._slot_cache is not None:
            slot_ex._slot_cache.close()



def test_unread_lookahead_probation_protects_warm_decode_slots(tmp_path: Path):
    num_experts = 16
    num_slots = 8
    static_slots = 4
    topk = 3
    hidden = 256
    intermediate = 128

    w13, s13, w2, s2 = _make_random_weights(
        num_experts, hidden, intermediate, seed=501
    )
    ref_plan = MoeExpertsV41Plan(
        num_experts, topk, hidden, intermediate, swiglu_limit=7.0, out_dtype=torch.bfloat16
    )
    ref_plan.declare_kernels()
    ref_weights = ref_plan.prepare_weights(w13, s13, w2, s2)

    slot_plan = MoeExpertsV41Plan(
        num_experts,
        topk,
        hidden,
        intermediate,
        swiglu_limit=7.0,
        out_dtype=torch.bfloat16,
        num_slots=num_slots,
        narrow_codes=False,
    )
    slot_plan.declare_kernels()
    slot_prepared_all = slot_plan.prepare_weights(w13, s13, w2, s2)
    meta = ExpertPackMeta.from_plan(slot_plan)
    pack_path = pack_layer_experts(
        tmp_path / "probation.pack", meta, slot_prepared_all[0], slot_prepared_all[1]
    )

    cache = ExpertSlotCache(
        meta, pack_path, num_slots=num_slots, static_slots=static_slots
    )
    try:
        # Warm dynamic experts 4 and 5 across several decode steps; leave 6 and 7 cooler.
        for _ in range(5):
            w_cb = cache.ensure_active([0, 4, 5], is_prefill=False)
            if w_cb is not None:
                w_cb()
        score_4_before = float(cache.lfu.scores[4])
        score_5_before = float(cache.lfu.scores[5])
        assert score_4_before > 2.0 and score_5_before > 2.0

        # Speculatively prefetch cold experts [10, 11]; they must replace the
        # coldest dynamic slots (6 and 7) with probationary=True and score=0.0.
        n_pref = cache.prefetch_async([11, 10])
        assert n_pref == 2
        slot_10 = int(cache.lfu.expert_to_slot[10])
        slot_11 = int(cache.lfu.expert_to_slot[11])
        assert slot_10 in (6, 7) and slot_11 in (6, 7) and slot_10 < slot_11
        assert bool(cache.lfu.probationary[slot_10]) is True
        assert bool(cache.lfu.probationary[slot_11]) is True
        assert float(cache.lfu.scores[10]) == 0.0
        assert float(cache.lfu.scores[11]) == 0.0

        # Router demands [0, 4, 10]: 10 is promoted (lookahead hit), while 11 was
        # mispredicted so its probation is cleared and its score stays 0.0.
        x_bf16 = torch.randn(2, hidden, dtype=torch.bfloat16, device="cuda")
        x_q = linear.cast_mxfp8(x_bf16)
        topk_ids = torch.tensor([[0, 4, 10], [1, 5, 10]], dtype=torch.int64, device="cuda")
        topk_w = torch.softmax(torch.randn(2, topk, dtype=torch.float32, device="cuda"), dim=-1)
        out_ref = torch.empty(2, hidden, dtype=torch.bfloat16, device="cuda")
        out_slot = torch.empty(2, hidden, dtype=torch.bfloat16, device="cuda")

        ref_plan.run(x_q, topk_ids, topk_w, ref_weights, out_ref)
        cache.run(slot_plan, x_q, topk_ids, topk_w, out_slot)
        torch.cuda.synchronize()
        assert torch.equal(out_ref, out_slot)

        assert cache.lookahead_hits == 1
        assert cache.lookahead_evictable == 1
        assert bool(cache.lfu.probationary[slot_10]) is False
        assert bool(cache.lfu.probationary[slot_11]) is False
        assert float(cache.lfu.scores[10]) > 0.0
        assert float(cache.lfu.scores[11]) == 0.0

        # A subsequent demand miss for cold expert 12 must evict the unhit
        # lookahead expert 11 (score=0.0) first, preserving warm experts 4, 5, 10.
        w_cb2 = cache.ensure_active([0, 12], is_prefill=False)
        if w_cb2 is not None:
            w_cb2()
        assert int(cache.lfu.expert_to_slot[12]) == slot_11
        assert int(cache.lfu.expert_to_slot[11]) == -1
        assert int(cache.lfu.expert_to_slot[4]) >= static_slots
        assert int(cache.lfu.expert_to_slot[5]) >= static_slots
        assert int(cache.lfu.expert_to_slot[10]) == slot_10
    finally:
        cache.close()


def test_lba_sorted_miss_scheduling_and_contiguous_slot_coalescing(tmp_path: Path):
    num_experts = 16
    num_slots = 8
    static_slots = 4
    topk = 4
    hidden = 256
    intermediate = 128

    w13, s13, w2, s2 = _make_random_weights(
        num_experts, hidden, intermediate, seed=607
    )
    ref_plan = MoeExpertsV41Plan(
        num_experts, topk, hidden, intermediate, swiglu_limit=7.0, out_dtype=torch.bfloat16
    )
    ref_plan.declare_kernels()
    ref_weights = ref_plan.prepare_weights(w13, s13, w2, s2)

    slot_plan = MoeExpertsV41Plan(
        num_experts,
        topk,
        hidden,
        intermediate,
        swiglu_limit=7.0,
        out_dtype=torch.bfloat16,
        num_slots=num_slots,
        narrow_codes=False,
    )
    slot_plan.declare_kernels()
    slot_prepared_all = slot_plan.prepare_weights(w13, s13, w2, s2)
    meta = ExpertPackMeta.from_plan(slot_plan)
    pack_path = pack_layer_experts(
        tmp_path / "lba_coalesce.pack", meta, slot_prepared_all[0], slot_prepared_all[1]
    )

    cache = ExpertSlotCache(
        meta, pack_path, num_slots=num_slots, static_slots=static_slots
    )
    try:
        # Demand 4 contiguous cold experts in reverse/scrambled order [15, 12, 14, 13].
        x_bf16 = torch.randn(2, hidden, dtype=torch.bfloat16, device="cuda")
        x_q = linear.cast_mxfp8(x_bf16)
        topk_ids = torch.tensor(
            [[15, 12, 14, 13], [0, 13, 15, 12]], dtype=torch.int64, device="cuda"
        )
        topk_w = torch.softmax(torch.randn(2, topk, dtype=torch.float32, device="cuda"), dim=-1)
        out_ref = torch.empty(2, hidden, dtype=torch.bfloat16, device="cuda")
        out_slot = torch.empty(2, hidden, dtype=torch.bfloat16, device="cuda")

        ref_plan.run(x_q, topk_ids, topk_w, ref_weights, out_ref)
        cache.run(slot_plan, x_q, topk_ids, topk_w, out_slot)
        torch.cuda.synchronize()
        assert torch.equal(out_ref, out_slot)

        # Verify LBA-sorted expert-to-slot binding: 12->4, 13->5, 14->6, 15->7.
        assigned_slots = [int(cache.lfu.expert_to_slot[e]) for e in (12, 13, 14, 15)]
        assert assigned_slots == [4, 5, 6, 7]
        rstats = cache.reader_stats()
        assert rstats["lba_ordered_submissions"] >= 1
        assert rstats["contiguous_slot_pairs"] >= 3
    finally:
        cache.close()


def test_router_bias_seeding_and_layer_slot_budget_planner(tmp_path: Path):
    num_experts = 16
    num_slots = 8
    static_slots = 4
    topk = 3
    hidden = 256
    intermediate = 128

    # Construct router bias where tail experts 12..15 have the highest prior bias.
    bias = torch.linspace(-1.0, 1.0, steps=num_experts, dtype=torch.float32)
    bias[12:] += 5.0
    ranked = rank_experts_by_router_bias(bias)
    assert set(ranked[:4]) == {12, 13, 14, 15}
    assert ranked[0] == 15

    # Empirical hotlist_counts boosts expert 7 to the top even with lower bias.
    hotlist = torch.zeros(num_experts, dtype=torch.float32)
    hotlist[7] = 10000.0
    ranked_hot = rank_experts_by_router_bias(bias, hotlist_counts=hotlist)
    assert ranked_hot[0] == 7

    # Verify ExpertSlotCache seeds static_slots from router_bias automatically.
    w13, s13, w2, s2 = _make_random_weights(
        num_experts, hidden, intermediate, seed=709
    )
    slot_plan = MoeExpertsV41Plan(
        num_experts,
        topk,
        hidden,
        intermediate,
        swiglu_limit=7.0,
        out_dtype=torch.bfloat16,
        num_slots=num_slots,
        narrow_codes=False,
    )
    slot_plan.declare_kernels()
    slot_prepared_all = slot_plan.prepare_weights(w13, s13, w2, s2)
    meta = ExpertPackMeta.from_plan(slot_plan)
    pack_path = pack_layer_experts(
        tmp_path / "router_bias.pack", meta, slot_prepared_all[0], slot_prepared_all[1]
    )
    cache = ExpertSlotCache(
        meta,
        pack_path,
        num_slots=num_slots,
        static_slots=static_slots,
        router_bias=bias,
    )
    try:
        # Static slots 0..3 must hold the top-4 biased experts {15, 14, 13, 12},
        # so a decode step routing to [12, 13, 14, 15] incurs zero misses.
        w_cb = cache.ensure_active([12, 13, 14, 15], is_prefill=False)
        assert w_cb is None
        assert cache.misses == 0
        assert cache.hits == 4
    finally:
        cache.close()

    # Verify layer-adaptive slot budget planner across concentrated vs diffuse layers.
    peaked_bias = torch.zeros(384, dtype=torch.float32)
    peaked_bias[:8] = 10.0  # Very low routing entropy (concentrated on 8 experts)
    diffuse_bias = torch.zeros(384, dtype=torch.float32)  # Maximum routing entropy
    moderate_bias = torch.linspace(-0.5, 0.5, steps=384, dtype=torch.float32)

    budgets = plan_layer_slot_budgets(
        [peaked_bias, moderate_bias, diffuse_bias],
        total_slot_budget=3 * 256,
        min_slots_per_layer=96,
        max_slots_per_layer=360,
        dynamic_slots_per_layer=48,
    )
    assert len(budgets) == 3
    assert sum(b[0] for b in budgets) == 3 * 256
    for total_s, static_s in budgets:
        assert 96 <= total_s <= 360
        assert static_s == total_s - 48
    # Diffuse (high-entropy) layer gets strictly more slots than peaked layer.
    assert budgets[2][0] >= budgets[1][0] > budgets[0][0]


def test_compute_uma_slot_budget():
    # 115 GiB available, 8 GiB system, 12 GiB KV, 24 GiB non-routed -> 71 GiB for MoE.
    gib = 1 << 30
    slots_40 = compute_uma_slot_budget(
        num_layers=40,
        expert_bytes=9_134_080,
        non_routed_bytes=24 * gib,
        kv_reserve_bytes=12 * gib,
        system_reserve_bytes=8 * gib,
        available_bytes=115 * gib,
    )
    expected = (71 * gib // 40) // 9_134_080
    assert slots_40 == expected
    assert 64 <= slots_40 <= 384
    # The default is DS4.1 Flash's 40 MoE layers.
    assert (
        compute_uma_slot_budget(
            expert_bytes=9_134_080,
            non_routed_bytes=24 * gib,
            kv_reserve_bytes=12 * gib,
            system_reserve_bytes=8 * gib,
            available_bytes=115 * gib,
        )
        == slots_40
    )

    # Clamp to min_slots when available memory is tight, and max_slots when abundant.
    assert (
        compute_uma_slot_budget(
            num_layers=40,
            expert_bytes=9_134_080,
            available_bytes=10 * gib,
            system_reserve_bytes=8 * gib,
            min_slots=64,
            max_slots=384,
        )
        == 64
    )
    assert (
        compute_uma_slot_budget(
            num_layers=40,
            expert_bytes=9_134_080,
            available_bytes=256 * gib,
            system_reserve_bytes=8 * gib,
            min_slots=64,
            max_slots=384,
        )
        == 384
    )
    # Also verify live /proc/meminfo fallback returns a valid clamped slot count.
    live_slots = compute_uma_slot_budget(expert_bytes=9_134_080)
    assert 64 <= live_slots <= 384


def test_boot1_background_write_and_boot2_cached_start_skips_380_experts(
    tmp_path: Path,
):
    num_experts = 384
    num_slots = 64
    static_slots = 48
    topk = 4
    hidden = 256
    intermediate = 128

    w13, s13, w2, s2 = _make_random_weights(
        num_experts, hidden, intermediate, seed=811
    )
    bias = torch.linspace(-1.0, 2.0, steps=num_experts, dtype=torch.float32)

    ref_plan = MoeExpertsV41Plan(
        num_experts,
        topk,
        hidden,
        intermediate,
        swiglu_limit=7.0,
        out_dtype=torch.bfloat16,
    )
    ref_plan.declare_kernels()
    ref_weights = ref_plan.prepare_weights(w13, s13, w2, s2)

    slot_plan = MoeExpertsV41Plan(
        num_experts,
        topk,
        hidden,
        intermediate,
        swiglu_limit=7.0,
        out_dtype=torch.bfloat16,
        num_slots=num_slots,
        narrow_codes=False,
    )
    slot_plan.declare_kernels()
    meta = ExpertPackMeta.from_plan(slot_plan)
    ckpt_key = "dsv41-layer0-test-sha256"

    # Boot 1: chunked zero-staging build + background O_DIRECT write + loading_done().
    disk_boot1 = ExpertDiskCache(tmp_path, chunk_experts=64)
    cache_boot1 = disk_boot1.build_or_load_layer(
        "layer_00",
        meta,
        slot_plan,
        w13,
        s13,
        w2,
        s2,
        num_slots=num_slots,
        static_slots=static_slots,
        router_bias=bias,
        checkpoint_key=ckpt_key,
        background_write=True,
    )
    try:
        assert cache_boot1.slots_populated_in_ram == num_slots
        assert cache_boot1.slots_filled_from_disk == 0
        stats_boot1 = disk_boot1.loading_done(timeout=30.0)
        assert stats_boot1["background_writes_queued"] == 1
        assert stats_boot1["background_writes_completed"] == 1
        assert stats_boot1["cached_starts"] == 0
        assert stats_boot1["prepared_experts_total"] == 384
        pack_file = tmp_path / "layer_00.bin"
        assert pack_file.is_file()
        assert (
            pack_file.stat().st_size
            == PACK_HEADER_BYTES + num_experts * meta.expert_stride
        )
    finally:
        cache_boot1.close()
        disk_boot1.close()

    # Boot 2: cached start with 4-expert spot check skipping the remaining 380 experts.
    disk_boot2 = ExpertDiskCache(tmp_path, chunk_experts=64)
    cache_boot2 = disk_boot2.build_or_load_layer(
        "layer_00",
        meta,
        slot_plan,
        w13,
        s13,
        w2,
        s2,
        num_slots=num_slots,
        static_slots=static_slots,
        router_bias=bias,
        checkpoint_key=ckpt_key,
        spot_check_seed=20261108,
        static_only_on_cached_start=True,
    )
    try:
        stats_boot2 = disk_boot2.loading_done(timeout=10.0)
        assert stats_boot2["cached_starts"] == 1
        assert stats_boot2["spot_checks_passed"] == 1
        assert stats_boot2["spot_checks_failed"] == 0
        # Only 4 spot-checked experts were prepared; all 380 other experts were skipped.
        assert stats_boot2["prepared_experts_total"] == 4
        assert cache_boot2.slots_filled_from_disk == static_slots
        assert cache_boot2.reader_stats()["header_bytes"] == PACK_HEADER_BYTES

        # Verify bit-identical execution across static hits and cold dynamic misses.
        x_bf16 = torch.randn(6, hidden, dtype=torch.bfloat16, device="cuda")
        x_q = linear.cast_mxfp8(x_bf16)
        topk_ids = torch.tensor(
            [
                [383, 382, 381, 380],  # Top router-bias static slots (hits)
                [0, 1, 15, 200],  # Cold experts streamed on demand via io_uring
                [383, 0, 100, 380],
                [5, 6, 7, 8],
                [382, 381, 15, 200],
                [42, 99, 383, 0],
            ],
            dtype=torch.int64,
            device="cuda",
        )
        topk_w = torch.softmax(
            torch.randn(6, topk, dtype=torch.float32, device="cuda"), dim=-1
        )
        out_ref = torch.empty(6, hidden, dtype=torch.bfloat16, device="cuda")
        out_slot = torch.empty(6, hidden, dtype=torch.bfloat16, device="cuda")
        ref_plan.run(x_q, topk_ids, topk_w, ref_weights, out_ref)
        cache_boot2.run(slot_plan, x_q, topk_ids, topk_w, out_slot)
        torch.cuda.synchronize()
        assert torch.equal(out_ref, out_slot)
    finally:
        cache_boot2.close()
        disk_boot2.close()


def test_four_expert_spot_check_detects_single_byte_corruption_and_rebuilds(
    tmp_path: Path,
):
    num_experts = 16
    num_slots = 8
    static_slots = 4
    topk = 3
    hidden = 256
    intermediate = 128
    seed = 919

    w13, s13, w2, s2 = _make_random_weights(
        num_experts, hidden, intermediate, seed=seed
    )
    ref_plan = MoeExpertsV41Plan(
        num_experts,
        topk,
        hidden,
        intermediate,
        swiglu_limit=7.0,
        out_dtype=torch.bfloat16,
    )
    ref_plan.declare_kernels()
    ref_weights = ref_plan.prepare_weights(w13, s13, w2, s2)

    slot_plan = MoeExpertsV41Plan(
        num_experts,
        topk,
        hidden,
        intermediate,
        swiglu_limit=7.0,
        out_dtype=torch.bfloat16,
        num_slots=num_slots,
        narrow_codes=False,
    )
    slot_plan.declare_kernels()
    meta = ExpertPackMeta.from_plan(slot_plan)
    ckpt_key = "spot-check-test-v1"
    spot_seed = 424242

    spot_experts = select_spot_check_experts(num_experts, seed=spot_seed)
    assert len(spot_experts) == 4
    assert spot_experts[0] == 0
    assert spot_experts[1] == 1
    assert spot_experts[-1] == num_experts - 1
    sampled_expert = spot_experts[2]
    assert 2 <= sampled_expert < num_experts - 1

    disk = ExpertDiskCache(tmp_path, chunk_experts=8)
    cache = disk.build_or_load_layer(
        "layer_spot",
        meta,
        slot_plan,
        w13,
        s13,
        w2,
        s2,
        num_slots=num_slots,
        static_slots=static_slots,
        checkpoint_key=ckpt_key,
        background_write=False,
    )
    cache.close()
    pack_file = tmp_path / "layer_spot.bin"

    # Corrupt 1 byte inside each of the 4 spot-checked experts (0, 1, sampled, E-1)
    # and verify that verify_spot_experts catches every single one and that
    # build_or_load_layer transparently rebuilds the file.
    for target_exp in (0, 1, sampled_expert, num_experts - 1):
        corrupt_offset = PACK_HEADER_BYTES + target_exp * meta.expert_stride + 137
        orig_data = bytearray(pack_file.read_bytes())
        orig_data[corrupt_offset] ^= 0xFF
        pack_file.write_bytes(orig_data)

        ok, reason, checked = verify_spot_experts(
            pack_file,
            meta,
            slot_plan,
            w13,
            s13,
            w2,
            s2,
            expected_checkpoint_key=ckpt_key,
            rng_seed=spot_seed,
        )
        assert not ok
        assert f"expert {target_exp}" in reason
        assert checked == spot_experts

        rebuilt_cache = disk.build_or_load_layer(
            "layer_spot",
            meta,
            slot_plan,
            w13,
            s13,
            w2,
            s2,
            num_slots=num_slots,
            static_slots=static_slots,
            checkpoint_key=ckpt_key,
            spot_check_seed=spot_seed,
        )
        try:
            x_bf16 = torch.randn(4, hidden, dtype=torch.bfloat16, device="cuda")
            x_q = linear.cast_mxfp8(x_bf16)
            topk_ids = torch.tensor(
                [
                    [0, 1, sampled_expert],
                    [num_experts - 1, 0, 2],
                    [1, sampled_expert, num_experts - 1],
                    [3, 7, 11],
                ],
                dtype=torch.int64,
                device="cuda",
            )
            topk_w = torch.softmax(
                torch.randn(4, topk, dtype=torch.float32, device="cuda"), dim=-1
            )
            out_ref = torch.empty(4, hidden, dtype=torch.bfloat16, device="cuda")
            out_slot = torch.empty(4, hidden, dtype=torch.bfloat16, device="cuda")
            ref_plan.run(x_q, topk_ids, topk_w, ref_weights, out_ref)
            rebuilt_cache.run(slot_plan, x_q, topk_ids, topk_w, out_slot)
            torch.cuda.synchronize()
            assert torch.equal(out_ref, out_slot)
        finally:
            rebuilt_cache.close()

    stats = disk.stats()
    assert stats["spot_checks_failed"] == 4
    assert stats["rebuilds"] == 4
    disk.close()


def test_free_disk_space_guard_and_dead_pid_tmp_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import os
    import shutil
    from collections import namedtuple

    # 1. Verify dead-PID .tmp.<pid> cleanup removes stale files while keeping live PID.
    dead_tmp = tmp_path / "layer_00.bin.tmp.99999999"
    live_tmp = tmp_path / f"layer_00.bin.tmp.{os.getpid()}"
    dead_tmp.write_bytes(b"stale")
    live_tmp.write_bytes(b"active")

    cleaned = cleanup_stale_tmp_files(tmp_path)
    assert cleaned == 1
    assert not dead_tmp.exists()
    assert live_tmp.exists()
    live_tmp.unlink()

    # 2. Verify < 8 GiB free disk space guard skips writing without failing startup.
    DiskUsage = namedtuple("DiskUsage", ["total", "used", "free"])
    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda _path: DiskUsage(
            total=100 * (1 << 30),
            used=96 * (1 << 30),
            free=4 * (1 << 30),  # 4 GiB < 8 GiB MIN_FREE_DISK_BYTES
        ),
    )

    num_experts = 8
    num_slots = 8
    static_slots = 8
    topk = 2
    hidden = 256
    intermediate = 128

    w13, s13, w2, s2 = _make_random_weights(
        num_experts, hidden, intermediate, seed=1021
    )
    ref_plan = MoeExpertsV41Plan(
        num_experts,
        topk,
        hidden,
        intermediate,
        swiglu_limit=7.0,
        out_dtype=torch.bfloat16,
    )
    ref_plan.declare_kernels()
    ref_weights = ref_plan.prepare_weights(w13, s13, w2, s2)

    slot_plan = MoeExpertsV41Plan(
        num_experts,
        topk,
        hidden,
        intermediate,
        swiglu_limit=7.0,
        out_dtype=torch.bfloat16,
        num_slots=num_slots,
        narrow_codes=False,
    )
    slot_plan.declare_kernels()
    meta = ExpertPackMeta.from_plan(slot_plan)

    disk = ExpertDiskCache(tmp_path, min_free_bytes=MIN_FREE_DISK_BYTES, chunk_experts=4)
    cache = disk.build_or_load_layer(
        "layer_low_disk",
        meta,
        slot_plan,
        w13,
        s13,
        w2,
        s2,
        num_slots=num_slots,
        static_slots=static_slots,
        background_write=True,
    )
    try:
        stats = disk.loading_done(timeout=10.0)
        assert stats["background_writes_queued"] == 1
        assert stats["background_writes_completed"] == 0
        assert stats["background_writes_skipped_low_disk"] == 1
        assert not (tmp_path / "layer_low_disk.bin").exists()
        assert list(tmp_path.glob("*.tmp.*")) == []

        # Resident slots populated in RAM still execute bit-identically.
        x_bf16 = torch.randn(3, hidden, dtype=torch.bfloat16, device="cuda")
        x_q = linear.cast_mxfp8(x_bf16)
        topk_ids = torch.tensor([[0, 7], [2, 5], [1, 6]], dtype=torch.int64, device="cuda")
        topk_w = torch.softmax(
            torch.randn(3, topk, dtype=torch.float32, device="cuda"), dim=-1
        )
        out_ref = torch.empty(3, hidden, dtype=torch.bfloat16, device="cuda")
        out_slot = torch.empty(3, hidden, dtype=torch.bfloat16, device="cuda")
        ref_plan.run(x_q, topk_ids, topk_w, ref_weights, out_ref)
        cache.run(slot_plan, x_q, topk_ids, topk_w, out_slot)
        torch.cuda.synchronize()
        assert torch.equal(out_ref, out_slot)
    finally:
        cache.close()
        disk.close()


def test_load_checkpoint_router_biases_from_sharded_index(tmp_path: Path):
    import json
    from safetensors.torch import save_file
    from vllm.models.deepseek_v4_1.nvidia.model import _load_checkpoint_router_biases

    b0 = torch.linspace(-1.0, 1.0, 8, dtype=torch.bfloat16)
    b2 = torch.linspace(2.0, -2.0, 8, dtype=torch.bfloat16)
    shard0 = "model-00001-of-00002.safetensors"
    shard1 = "model-00002-of-00002.safetensors"
    save_file({"layers.0.ffn.gate.bias": b0}, str(tmp_path / shard0))
    save_file({"model.layers.2.ffn.gate.bias": b2}, str(tmp_path / shard1))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "layers.0.ffn.gate.bias": shard0,
                    "model.layers.2.ffn.gate.bias": shard1,
                }
            }
        )
    )

    loaded = _load_checkpoint_router_biases(tmp_path, [0, 2, 5], num_experts=8)
    assert len(loaded) == 3
    assert torch.allclose(loaded[0], b0.float())
    assert torch.allclose(loaded[1], b2.float())
    assert torch.equal(loaded[2], torch.zeros(8, dtype=torch.float32))


def test_mxfp4_meta_online_slot_cache_end_to_end(tmp_path: Path, monkeypatch):
    import vllm.envs as envs
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        RoutingMethodType,
    )
    from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod

    num_experts = 16
    num_slots = 8
    static_slots = 5
    topk = 3
    hidden = 256
    intermediate = 128

    w13, s13, w2, s2 = _make_random_weights(
        num_experts, hidden, intermediate, seed=707
    )
    w13_u8 = w13.view(torch.uint8).cpu()
    s13_u8 = s13.cpu()
    w2_u8 = w2.view(torch.uint8).cpu()
    s2_u8 = s2.cpu()

    moe_cfg = FusedMoEConfig(
        num_experts=num_experts,
        experts_per_token=topk,
        hidden_dim=hidden,
        intermediate_size=intermediate,
        num_local_experts=num_experts,
        num_logical_experts=num_experts,
        activation=MoEActivation.SILU,
        device="cuda",
        routing_method=RoutingMethodType.DeepseekV4,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        in_dtype=torch.bfloat16,
        moe_backend="tilelang",
        has_bias=False,
        swiglu_limit=7.0,
    )

    def _copy_loader(
        param: torch.nn.Parameter,
        loaded: torch.Tensor,
        weight_name: str = "",
        shard_id: str = "",
        expert_id: int = 0,
        return_success: bool = False,
    ):
        assert not param.is_meta
        param.data.copy_(loaded.to(device=param.device, dtype=param.dtype))
        return True if return_success else None

    # Draft layer (layer >= 40) must remain non-meta even when VLLM_DS41_EXPERT_SLOTS > 0.
    monkeypatch.setattr(envs, "VLLM_DS41_EXPERT_SLOTS", num_slots)
    draft_layer = torch.nn.Module()
    draft_layer.layer_name = "model.layers.40.ffn.experts.routed_experts"
    draft_qm = Mxfp4MoEMethod(moe_cfg)
    with torch.device("cuda"):
        draft_qm.create_weights(
            draft_layer,
            num_experts=num_experts,
            hidden_size=hidden,
            intermediate_size_per_partition=intermediate,
            params_dtype=torch.bfloat16,
            weight_loader=_copy_loader,
        )
    assert draft_qm.uses_meta_device is False
    assert not draft_layer.w13_weight.is_meta

    # Target layer with slot_cache_config allocates on meta and finalizes online once loaded.
    disk_cache = ExpertDiskCache(tmp_path, chunk_experts=4)
    try:
        slot_layer = torch.nn.Module()
        slot_layer.layer_name = "model.layers.3.ffn.experts.routed_experts"
        slot_layer.apply_router_weight_on_input = False
        slot_layer.activation = MoEActivation.SILU
        slot_layer.moe_config = moe_cfg
        slot_layer.global_num_experts = num_experts
        slot_layer.expert_map = None
        slot_layer._expert_routing_tables = lambda: None
        slot_layer.e_score_correction_bias = None
        slot_layer.slot_cache_config = {
            "pack_path": tmp_path / "layer_003.bin",
            "num_slots": num_slots,
            "static_slots": static_slots,
            "disk_cache": disk_cache,
            "background_write": True,
        }

        slot_qm = Mxfp4MoEMethod(moe_cfg)
        slot_layer.quant_method = slot_qm
        with torch.device("cuda"):
            slot_qm.create_weights(
                slot_layer,
                num_experts=num_experts,
                hidden_size=hidden,
                intermediate_size_per_partition=intermediate,
                params_dtype=torch.bfloat16,
                weight_loader=_copy_loader,
            )
        assert slot_qm.uses_meta_device is True
        assert slot_layer.w13_weight.is_meta
        assert slot_layer.w2_weight.is_meta

        # Stream weights through wrapped online_process_loader.
        slot_layer.w13_weight.weight_loader(slot_layer.w13_weight, w13_u8, "w13.weight", "w1", 0)
        slot_layer.w13_weight_scale.weight_loader(slot_layer.w13_weight_scale, s13_u8, "w13.scale", "w1", 0)
        slot_layer.w2_weight.weight_loader(slot_layer.w2_weight, w2_u8, "w2.weight", "w2", 0)
        slot_layer.w2_weight_scale.weight_loader(slot_layer.w2_weight_scale, s2_u8, "w2.scale", "w2", 0)

        # Online processing already finalized after last parameter arrived.
        assert slot_layer._already_called_process_weights_after_loading is True
        assert not slot_layer.w13_weight.is_meta
        assert slot_qm.moe_kernel.fused_experts._slot_cache is not None
        slot_qm.moe_kernel.fused_experts.declare_kernels()

        stats = disk_cache.loading_done(timeout=10.0)
        assert stats["background_writes_completed"] == 1
        assert (tmp_path / "layer_003.bin").is_file()

        # Reference unoffloaded layer on VLLM_DS41_EXPERT_SLOTS = 0.
        monkeypatch.setattr(envs, "VLLM_DS41_EXPERT_SLOTS", 0)
        ref_layer = torch.nn.Module()
        ref_layer.layer_name = "model.layers.3.ffn.experts.routed_experts"
        ref_layer.apply_router_weight_on_input = False
        ref_layer.activation = MoEActivation.SILU
        ref_layer.moe_config = moe_cfg
        ref_layer.global_num_experts = num_experts
        ref_layer.expert_map = None
        ref_layer._expert_routing_tables = lambda: None
        ref_layer.e_score_correction_bias = None
        ref_qm = Mxfp4MoEMethod(moe_cfg)
        ref_layer.quant_method = ref_qm
        with torch.device("cuda"):
            ref_qm.create_weights(
                ref_layer,
                num_experts=num_experts,
                hidden_size=hidden,
                intermediate_size_per_partition=intermediate,
                params_dtype=torch.bfloat16,
                weight_loader=_copy_loader,
            )
        ref_layer.w13_weight.weight_loader(ref_layer.w13_weight, w13_u8, "w13.weight", "w1", 0)
        ref_layer.w13_weight_scale.weight_loader(ref_layer.w13_weight_scale, s13_u8, "w13.scale", "w1", 0)
        ref_layer.w2_weight.weight_loader(ref_layer.w2_weight, w2_u8, "w2.weight", "w2", 0)
        ref_layer.w2_weight_scale.weight_loader(ref_layer.w2_weight_scale, s2_u8, "w2.scale", "w2", 0)
        ref_qm.process_weights_after_loading(ref_layer)
        ref_qm.moe_kernel.fused_experts.declare_kernels()

        x = torch.randn(4, hidden, dtype=torch.bfloat16, device="cuda")
        topk_ids = torch.tensor(
            [[0, 9, 15], [1, 8, 14], [2, 10, 13], [4, 11, 12]],
            dtype=torch.int64,
            device="cuda",
        )
        topk_w = torch.softmax(
            torch.randn(4, topk, dtype=torch.float32, device="cuda"), dim=-1
        )
        from vllm.v1.worker import workspace as _ws

        ws_inited = _ws._manager is None
        if ws_inited:
            _ws.init_workspace_manager(torch.device("cuda"))
        try:
            with linear.provide_mxfp8(linear.cast_mxfp8(x)):
                out_ref = ref_qm.apply(ref_layer, x, topk_w, topk_ids, None, None)
                out_slot = slot_qm.apply(slot_layer, x, topk_w, topk_ids, None, None)
            torch.cuda.synchronize()
            assert torch.allclose(out_ref, out_slot, rtol=1e-2, atol=1e-2) or torch.equal(out_ref, out_slot)
        finally:
            if ws_inited:
                _ws.reset_workspace_manager()
    finally:
        disk_cache.close()
