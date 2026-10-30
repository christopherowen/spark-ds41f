# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from vllm.config import SpeculativeConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.worker.gpu.attn_utils import AttentionCGSupportInfo
from vllm.v1.worker.gpu.spec_decode import adaptive_verification as adaptive_module
from vllm.v1.worker.gpu.spec_decode.adaptive_verification import (
    AdaptiveVerificationManager,
    maybe_create_adaptive_verification_manager,
    resolve_adaptive_cudagraph_mode,
)
from vllm.v1.worker.gpu.structured_outputs import _build_grammar_mapping


def make_manager(
    confidences: np.ndarray, row_cost: float = 0.02
) -> AdaptiveVerificationManager:
    num_reqs, num_steps = confidences.shape
    manager = AdaptiveVerificationManager.__new__(AdaptiveVerificationManager)
    manager.num_speculative_steps = num_steps
    manager._stale_confidences = [SimpleNamespace(np=confidences)]
    manager._stale_idx = 0
    manager.req_states = SimpleNamespace(
        req_id_to_index={"low": 0, "high": 1},
        num_computed_tokens_np=np.ones(num_reqs, dtype=np.int32),
        prefill_len=SimpleNamespace(np=np.ones(num_reqs, dtype=np.int32)),
    )
    manager._max_total_logits = 1 << 30
    manager.num_bonus_tokens = 1
    manager.row_cost = row_cost
    return manager


@pytest.mark.parametrize(
    ("mode", "piecewise_capture_available", "expected"),
    [
        ("FULL_DECODE_ONLY", True, "FULL_DECODE_ONLY"),
        ("FULL_DECODE_ONLY", False, "FULL_DECODE_ONLY"),
        ("FULL", True, "FULL_AND_PIECEWISE"),
        ("FULL", False, "FULL_DECODE_ONLY"),
        ("FULL_AND_PIECEWISE", True, "FULL_AND_PIECEWISE"),
        ("FULL_AND_PIECEWISE", False, "FULL_DECODE_ONLY"),
    ],
)
def test_resolve_adaptive_cudagraph_mode(mode, piecewise_capture_available, expected):
    assert (
        resolve_adaptive_cudagraph_mode(
            CUDAGraphMode[mode], piecewise_capture_available=piecewise_capture_available
        )
        == CUDAGraphMode[expected]
    )


@pytest.mark.parametrize("world_size,expected_budget", [(1, 0), (2, 1)])
def test_confidence_snapshot_agrees_before_cpu_and_gpu_budgeting(
    monkeypatch, world_size, expected_budget
):
    # One FP32 ULP across the budget's decision changes the graph's token count:
    # at row cost 1 one draft pays exactly when its survival exceeds 0.5.
    local = torch.tensor([[np.nextafter(np.float32(0.5), np.float32(0.0))]])
    leader = torch.tensor([[np.nextafter(np.float32(0.5), np.float32(1.0))]])
    manager = make_manager(local.numpy().copy(), row_cost=1.0)
    manager.req_states.device = torch.device("cpu")
    manager._confidence_probs = torch.empty_like(local)
    manager._pending_resets = []

    class CopyBuffer:
        """Synchronous CPU double for a confidence snapshot transfer buffer."""

        def __init__(self):
            """Allocate independent source and destination snapshot storage."""
            self.np = np.ones((1, 1), dtype=np.float32)
            self.gpu = torch.empty_like(local)

        def copy_to_cpu(self):
            """Copy the synchronized snapshot into the CPU selection buffer."""
            self.np[:] = self.gpu.numpy()

    manager._stale_confidences = [CopyBuffer(), CopyBuffer()]
    manager._copy_events = [
        SimpleNamespace(synchronize=lambda: None, record=lambda: None) for _ in range(2)
    ]
    manager._copy_stream = SimpleNamespace(wait_stream=lambda _stream: None)
    calls = []

    def broadcast(tensor, src):
        assert src == 0
        calls.append(tensor)
        tensor.copy_(leader)

    monkeypatch.setattr(
        adaptive_module,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=world_size, broadcast=broadcast),
    )
    monkeypatch.setattr(adaptive_module, "gpu_sync_allowed", nullcontext)
    monkeypatch.setattr(adaptive_module.torch.cuda, "current_stream", lambda _: None)
    monkeypatch.setattr(adaptive_module, "stream", lambda *_: nullcontext())

    def scatter_rows(idx_mapping, source, state, gather=False):
        """The TileLang row scatter (its own GPU test) as torch, on the host."""
        state[idx_mapping, : source.shape[1]] = source

    monkeypatch.setattr(adaptive_module.tilelang_inputs, "scatter_rows", scatter_rows)
    batch = SimpleNamespace(num_reqs=1, idx_mapping=torch.tensor([0]))

    # Two records expose the completed previous-step snapshot to CPU selection.
    for _ in range(2):
        manager.record_confidences(local, batch)
    expected = leader if world_size > 1 else local
    torch.testing.assert_close(manager._confidence_probs, expected, rtol=0, atol=0)
    np.testing.assert_array_equal(
        manager._stale_confidences[manager._stale_idx].np, expected.numpy()
    )
    assert manager.get_num_tokens({"low": 2}, {"low": [7]}) == 1 + expected_budget
    assert len(calls) == (2 if world_size > 1 else 0)
    assert local.item() < 0.5  # The caller's proposal confidence is not mutated.


@pytest.mark.parametrize(
    "target_support,varlen_decode,allowed",
    [
        (AttentionCGSupport.ALWAYS, False, True),
        (AttentionCGSupport.UNIFORM_BATCH, True, True),
        (AttentionCGSupport.UNIFORM_BATCH, False, False),
        (AttentionCGSupport.NEVER, True, False),
    ],
)
def test_manager_scopes_varlen_check_without_weakening_runner_cg_mode(
    monkeypatch, target_support, varlen_decode, allowed
):
    class Backend:
        @classmethod
        def supports_device_cpu_query_lens_mismatch(cls):
            return True

    class Builder:
        supports_varlen_decode_cudagraph = varlen_decode

        def __init__(self, support):
            self.support = support

        def get_cudagraph_support(self, *_args):
            return self.support

    def group(layer_name, support):
        builder = Builder(support)
        return SimpleNamespace(
            layer_names=[layer_name],
            backend=Backend,
            kv_cache_spec=None,
            get_metadata_builder=lambda _index: builder,
        )

    groups = [
        [
            group("target", target_support),
            group("draft", AttentionCGSupport.UNIFORM_BATCH),
        ]
    ]
    runner_support = AttentionCGSupportInfo(
        AttentionCGSupport.UNIFORM_BATCH, "DraftBackend"
    )
    created = object()
    monkeypatch.setattr(
        adaptive_module,
        "AdaptiveVerificationManager",
        lambda *_args, **_kwargs: created,
    )

    kwargs = dict(
        enable_adaptive_verification=True,
        attn_groups=groups,
        attn_cg_support=runner_support,
        req_states=object(),
        query_start_loc=object(),
        num_bonus_tokens=1,
        max_total_logits=1,
        vllm_config=None,
        target_layer_names={"target"},
    )

    if allowed:
        assert maybe_create_adaptive_verification_manager(**kwargs) is created
    else:
        with pytest.raises(ValueError, match="variable-length decode"):
            maybe_create_adaptive_verification_manager(**kwargs)
    assert runner_support.min_cg_support == AttentionCGSupport.UNIFORM_BATCH


def test_glm53_adaptive_manager_accepts_real_target_backends(monkeypatch):
    from vllm.models.glm5next.nvidia.kda import Glm5NextKDAAttentionBackend
    from vllm.v1.attention.backends.mla.b12x_mla_sparse import (
        B12xGLM5NextMLASparseBackend,
        B12xMLASparseBackend,
    )

    backends = [Glm5NextKDAAttentionBackend, B12xGLM5NextMLASparseBackend]
    groups = []
    for index, backend in enumerate(backends):
        builder = object.__new__(backend.get_builder_cls())
        groups.append(
            SimpleNamespace(
                layer_names=[f"target.{index}"],
                backend=backend,
                kv_cache_spec=None,
                get_metadata_builder=lambda _index, builder=builder: builder,
            )
        )
    created = object()
    monkeypatch.setattr(
        adaptive_module, "AdaptiveVerificationManager", lambda *_a, **_kw: created
    )
    assert (
        maybe_create_adaptive_verification_manager(
            enable_adaptive_verification=True,
            attn_groups=[groups],
            attn_cg_support=AttentionCGSupportInfo(
                AttentionCGSupport.UNIFORM_BATCH, None
            ),
            req_states=object(),
            query_start_loc=object(),
            num_bonus_tokens=1,
            max_total_logits=32,
            vllm_config=SimpleNamespace(
                model_config=SimpleNamespace(
                    hf_text_config=SimpleNamespace(model_type="glm5_next_text")
                )
            ),
            target_layer_names={"target.0", "target.1"},
        )
        is created
    )
    assert not B12xMLASparseBackend.supports_device_cpu_query_lens_mismatch()


def test_compact_batch_preserves_totals_and_bounds():
    # The CPU placeholder layout must keep the batch total equal to the GPU
    # total and every verification row within decode_query_len, or downstream
    # CPU metadata desyncs from the reallocated GPU boundaries.
    manager = make_manager(
        np.array([[0.9, 0.9], [0.9, 0.9], [1.0, 1.0]], dtype=np.float32),
    )
    manager.req_states.req_id_to_index["prefill"] = 2
    manager.req_states.num_computed_tokens_np = np.zeros(3, dtype=np.int32)
    manager.req_states.prefill_len.np = np.array([0, 0, 60], dtype=np.int32)
    num_tokens = manager.get_num_tokens(
        {"low": 3, "high": 3, "prefill": 40},
        {"low": [1, 2], "high": [3, 4]},
    )
    scheduled = np.array([3, 3, 40], dtype=np.int32)
    drafts = np.array([2, 2, 0], dtype=np.int32)
    cu_num_logits_np = np.array([0, 3, 6, 7], dtype=np.int32)
    compacted, _ = manager.compact_batch(drafts, scheduled, cu_num_logits_np)

    assert int(compacted.sum()) == num_tokens
    num_steps = manager.num_speculative_steps
    assert (compacted[:2] <= 1 + num_steps).all()
    assert compacted[2] == 40


def test_budget_caps_at_one_rejection_sampler_chunk():
    # The chunked verification path cannot address the compacted logits
    # layout, so the budget must keep total logits within a single chunk.
    manager = make_manager(
        np.array([[0.9, 0.9], [0.9, 0.9]], dtype=np.float32),
    )
    manager._max_total_logits = 3  # 2 bonus logits + at most 1 draft
    manager.get_num_tokens(
        {"low": 3, "high": 3},
        {"low": [1, 2], "high": [3, 4]},
    )
    _, _, draft_budget = manager._batch_budget
    assert draft_budget <= 1


def test_zero_budget_rebuilds_cpu_cu_num_logits():
    # When one bonus row per request already overflows a verification chunk, the
    # budget clamps to zero but the batch still needs chunking. Every capacity is
    # zeroed on device, so the CPU can name that layout exactly -- and must, since
    # _iter_request_chunks slices the compacted logits with these offsets.
    #
    # The third request is a chunked prefill (no drafts, still mid-prompt). The
    # runner gives *every* request num_bonus_tokens logits rows regardless
    # (num_logits = num_draft_tokens_per_req + num_bonus_tokens), so the rebuilt
    # offsets stay uniform rather than skipping non-verification rows.
    manager = make_manager(
        np.array([[0.9, 0.9], [0.9, 0.9], [1.0, 1.0]], dtype=np.float32),
    )
    manager.req_states.req_id_to_index["prefill"] = 2
    manager.req_states.num_computed_tokens_np = np.zeros(3, dtype=np.int32)
    manager.req_states.prefill_len.np = np.array([0, 0, 60], dtype=np.int32)
    manager._max_total_logits = 2  # < 3 requests * 1 bonus token

    manager.get_num_tokens(
        {"low": 3, "high": 3, "prefill": 40},
        {"low": [1, 2], "high": [3, 4]},
    )
    _, _, draft_budget = manager._batch_budget
    assert draft_budget == 0

    scheduled = np.array([3, 3, 40], dtype=np.int32)
    drafts = np.array([2, 2, 0], dtype=np.int32)
    scheduled_cu_num_logits = np.array([0, 3, 6, 7], dtype=np.int32)
    compacted, cu_num_logits_np = manager.compact_batch(
        drafts, scheduled, scheduled_cu_num_logits
    )

    # One bonus row per request, matching cumsum(capacities + num_bonus_tokens)
    # with every capacity zeroed -- the prefill row included.
    expected = np.arange(4, dtype=np.int32) * manager.num_bonus_tokens
    assert np.array_equal(cu_num_logits_np, expected)
    assert cu_num_logits_np.dtype == scheduled_cu_num_logits.dtype
    # The prefill keeps its scheduled tokens; only drafts are dropped.
    assert np.array_equal(compacted, np.array([1, 1, 40], dtype=np.int32))


def test_zero_budget_keeps_one_grammar_row_per_scheduled_draft():
    # The scheduler sizes the grammar bitmask from the *scheduled* drafts
    # (len(drafts) + 1 rows per request), but a zero budget rewrites
    # cu_num_logits_np to bonus-only. Deriving the bitmask -> logits mapping
    # from those rewritten offsets drops rows and trips the
    # `num_masks == len(mapping)` assert in apply_grammar_bitmask.
    manager = make_manager(
        np.array([[0.9, 0.9], [0.9, 0.9], [1.0, 1.0]], dtype=np.float32),
    )
    manager.req_states.req_id_to_index["prefill"] = 2
    manager.req_states.num_computed_tokens_np = np.zeros(3, dtype=np.int32)
    manager.req_states.prefill_len.np = np.array([0, 0, 60], dtype=np.int32)
    manager._max_total_logits = 2  # < 3 requests * 1 bonus token

    scheduled_spec_decode_tokens = {"low": [1, 2], "high": [3, 4]}
    manager.get_num_tokens(
        {"low": 3, "high": 3, "prefill": 40}, scheduled_spec_decode_tokens
    )
    assert manager._batch_budget[2] == 0

    req_ids = ["low", "high", "prefill"]
    num_draft_tokens_per_req = np.array([2, 2, 0], dtype=np.int32)
    _, cu_num_logits_np = manager.compact_batch(
        num_draft_tokens_per_req,
        np.array([3, 3, 40], dtype=np.int32),
        np.array([0, 3, 6, 7], dtype=np.int32),
    )

    mask_stride = manager.num_speculative_steps + manager.num_bonus_tokens
    mapping = _build_grammar_mapping(
        req_ids,
        req_ids,
        cu_num_logits_np,
        num_draft_tokens_per_req,
        manager.num_bonus_tokens,
        mask_stride,
    )

    num_bitmask_rows = sum(
        len(scheduled_spec_decode_tokens.get(req_id, ())) + 1 for req_id in req_ids
    )
    assert len(mapping) == num_bitmask_rows
    # (request, position) keys, so the kernel can mask rows the compacted
    # device layout no longer has room for.
    assert mapping == [0, 1, 2, 3, 4, 5, 6]


def _run_tp_confidence_consistency(rank, port):
    import torch
    import torch.distributed as dist

    from tests.utils import init_test_distributed_environment
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        get_tp_group,
    )

    device = torch.device("cuda", rank)
    torch.accelerator.set_device_index(rank)
    with set_current_vllm_config(VllmConfig()), torch.no_grad():
        try:
            init_test_distributed_environment(2, 1, rank, str(port), local_rank=rank)
            state = SimpleNamespace(
                device=device,
                max_num_reqs=2,
                num_speculative_steps=5,
                req_id_to_index={"r": 0, "s": 1},
                num_computed_tokens_np=np.array([100, 100], dtype=np.int32),
                prefill_len=SimpleNamespace(np=np.ones(2, dtype=np.int32)),
            )
            manager = AdaptiveVerificationManager(
                state,
                torch.zeros(3, dtype=torch.int32, device=device),
                num_bonus_tokens=1,
                max_total_logits=6,
                # The third draft pays exactly when its survival exceeds 0.24.
                row_cost=0.24 / 1.56,
            )
            manager.add_request(0)
            manager.add_request(1)
            # The runner applies the new requests' confidences after adding them.
            manager.apply_staged_writes()
            batch = SimpleNamespace(
                num_reqs=1,
                idx_mapping=torch.tensor([0], dtype=torch.int32, device=device),
            )
            confidence = torch.tensor(
                [[0.8, 0.6, 0.502 if rank == 0 else 0.498, 0.2, 0.2]],
                dtype=torch.float32,
                device=device,
            )
            # Publish through the real GPU/D2H path twice to consume a landed
            # stale slot. Small rank-local differences straddle a graph budget.
            for _ in range(2):
                manager.record_confidences(confidence, batch)
            tokens = manager.get_num_tokens({"r": 6}, {"r": [1] * 5})
            counts = [None, None]
            dist.all_gather_object(counts, tokens, group=get_tp_group().cpu_group)
            assert counts == [4, 4], counts
            manager.reallocate_drafts(["r"], batch.idx_mapping)
            torch.accelerator.synchronize(device)
            assert manager.query_start_loc[:2].tolist() == [0, 4]

            # A shared total is insufficient: each request's GPU allocation
            # must agree too, even when local confidence rankings are reversed.
            manager._max_total_logits = 3
            batch = SimpleNamespace(
                num_reqs=2,
                idx_mapping=torch.tensor([0, 1], dtype=torch.int32, device=device),
            )
            confidence = torch.tensor(
                [[0.9, 0.1, 0.1, 0.1, 0.1], [0.4, 0.1, 0.1, 0.1, 0.1]]
                if rank == 0
                else [[0.1, 0.1, 0.1, 0.1, 0.1], [0.9, 0.1, 0.1, 0.1, 0.1]],
                dtype=torch.float32,
                device=device,
            )
            for _ in range(2):
                manager.record_confidences(confidence, batch)
            assert (
                manager.get_num_tokens({"r": 6, "s": 6}, {"r": [1] * 5, "s": [1] * 5})
                == 3
            )
            manager.reallocate_drafts(["r", "s"], batch.idx_mapping)
            torch.accelerator.synchronize(device)
            boundaries = [None, None]
            dist.all_gather_object(
                boundaries,
                manager.query_start_loc.tolist(),
                group=get_tp_group().cpu_group,
            )
            assert boundaries == [[0, 2, 3], [0, 2, 3]]
        finally:
            torch.accelerator.synchronize(device)
            destroy_model_parallel()
            destroy_distributed_environment()


@pytest.mark.distributed(num_gpus=2)
def test_tp_confidence_publication_keeps_graph_and_request_budgets_consistent():
    import torch

    from tests.utils import get_open_port

    if not torch.accelerator.is_available() or torch.accelerator.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    torch.multiprocessing.spawn(
        _run_tp_confidence_consistency, args=(get_open_port(),), nprocs=2, join=True
    )


def test_budget_probes_after_steps_without_drafts():
    """An estimator that stopped trusting the drafter must keep being graded.

    Confidences learned only from admitted drafts cannot recover once no draft
    is admitted, so after EXPLORE_AFTER_IDLE_STEPS empty budgets one step
    verifies a draft per request, then budgeting resumes as before.
    """
    manager = make_manager(
        np.array([[1e-9, 1e-9]], dtype=np.float32),
    )
    idle = getattr(adaptive_module, "EXPLORE_AFTER_IDLE_STEPS", 32)
    budgets = []
    for _ in range(2 * idle):
        manager.get_num_tokens({"low": 3}, {"low": [1, 2]})
        budgets.append(manager._batch_budget[2])
    assert budgets == ([0] * (idle - 1) + [1]) * 2


def test_budget_admits_the_drafts_that_pay_for_their_rows():
    # Survival is the running product of a request's confidences: low 0.1, 0.01;
    # high 0.9, 0.81. Two sampling requests, two non-draft rows; at row cost 0.2
    # expected tokens per unit cost peak with the high request's two drafts.
    manager = make_manager(
        np.array([[0.1, 0.1], [0.9, 0.9]], dtype=np.float32), row_cost=0.2
    )

    manager.get_num_tokens(
        {"low": 3, "high": 3},
        {"low": [1, 2], "high": [3, 4]},
    )
    valid_drafts, num_non_draft_tokens, draft_budget = manager._batch_budget

    assert draft_budget == 2
    assert valid_drafts == {"low": 2, "high": 2}
    assert num_non_draft_tokens == {"low": 1, "high": 1}


@pytest.mark.parametrize(("row_cost", "budget"), [(0.02, 2), (5.0, 1), (20.0, 0)])
def test_row_cost_alone_sets_the_budget(row_cost, budget):
    manager = make_manager(np.array([[0.9, 0.9]], dtype=np.float32), row_cost)
    tokens = manager.get_num_tokens({"low": 3}, {"low": [1, 2]})
    assert manager._batch_budget[2] == budget
    assert tokens == 1 + budget


@pytest.mark.parametrize("row_cost", [0.0, -0.5, 1.0, 1.5])
def test_adaptive_verification_row_cost_is_a_fraction(row_cost):
    with pytest.raises(ValidationError):
        SpeculativeConfig(
            method="ngram",
            num_speculative_tokens=1,
            adaptive_verification_row_cost=row_cost,
        )


def test_adaptive_verification_row_cost_requires_adaptive_dspark():
    with pytest.raises(
        ValueError,
        match="requires DSpark adaptive verification",
    ):
        SpeculativeConfig(
            method="ngram",
            num_speculative_tokens=1,
            adaptive_verification_row_cost=0.03,
        )


def test_warm_tail_shapes_runs_each_eager_size_once_untimed():
    manager = AdaptiveVerificationManager.__new__(AdaptiveVerificationManager)
    manager.req_states = SimpleNamespace(max_num_batched_tokens=1024)
    runs = []
    manager.warm_tail_shapes(lambda **kwargs: runs.append(kwargs), [1, 2, 8])
    assert runs == [{"num_tokens": n} for n in (12, 16, 32, 64, 128, 256, 512, 1024)]


def test_seeded_sampled_budget_is_batch_invariant():
    # At row_cost = 0.02, a single request with survival [0.5, 0.25, 0.05, 0.01, 0.002]
    # admits 2 drafts alone (marginal utility of slot 2 is 0.05 < 0.02 * 1.75 / 1.06),
    # whereas in an 8-request pooled batch with seven 0.99-confidence peers, pooling
    # inflates expected_tokens ~ 36 so even slot 1 (0.25 < 0.02 * 36 / 1.8) is dropped.
    # Seeded sampled requests budget per request so their verified block matches alone.
    target_conf = np.array([0.5, 0.5, 0.2, 0.2, 0.2], dtype=np.float32)
    peer_conf = np.array([0.99, 0.99, 0.99, 0.99, 0.99], dtype=np.float32)

    alone = make_manager(target_conf[None, :], row_cost=0.02)
    alone.req_states.req_id_to_index = {"target": 0}
    alone._seeded_sampled = np.array([True])
    alone.get_num_tokens({"target": 6}, {"target": [1, 2, 3, 4, 5]})
    alone_drafts, _, alone_budget = alone._batch_budget
    assert alone_budget == 3
    assert alone_drafts == {"target": 3}

    confs = np.vstack([target_conf[None, :], np.tile(peer_conf, (7, 1))])
    req_ids = ["target", *(f"peer_{i}" for i in range(7))]
    batch_mgr = make_manager(confs, row_cost=0.02)
    batch_mgr.req_states.req_id_to_index = {
        req_id: i for i, req_id in enumerate(req_ids)
    }
    # When unseeded/greedy, pooling across the 8 requests drops target's 3rd draft
    # (and 2nd draft), giving a total batch budget of 36 instead of 3 + 35 = 38.
    batch_mgr.get_num_tokens(
        dict.fromkeys(req_ids, 6),
        {req_id: [1, 2, 3, 4, 5] for req_id in req_ids},
    )
    _, _, pooled_budget = batch_mgr._batch_budget
    assert pooled_budget == 36

    # When seeded sampled, target gets the exact 3-draft budget it gets alone.
    batch_mgr._seeded_sampled = np.ones(8, dtype=bool)
    batch_mgr.get_num_tokens(
        dict.fromkeys(req_ids, 6),
        {req_id: [1, 2, 3, 4, 5] for req_id in req_ids},
    )
    seeded_drafts, _, seeded_budget = batch_mgr._batch_budget
    assert seeded_drafts["target"] == alone_budget
    assert seeded_budget == alone_budget + 7 * 5


def test_seeded_sampled_reallocate_drafts_and_mid_prefill_reset(monkeypatch):
    if not torch.accelerator.is_available():
        pytest.skip("requires CUDA")
    monkeypatch.setattr(
        adaptive_module,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=1),
    )
    device = torch.device("cuda", 0)
    state = SimpleNamespace(
        device=device,
        max_num_reqs=4,
        num_speculative_steps=5,
        req_id_to_index={"seeded": 0, "greedy_hi": 1, "greedy_lo": 2, "prefill": 3},
        num_computed_tokens_np=np.array([10, 10, 10, 0], dtype=np.int32),
        prefill_len=SimpleNamespace(np=np.array([10, 10, 10, 100], dtype=np.int32)),
    )
    manager = AdaptiveVerificationManager(
        state,
        torch.zeros(5, dtype=torch.int32, device=device),
        num_bonus_tokens=1,
        max_total_logits=64,
        row_cost=0.2,
    )
    manager.add_request(0, SimpleNamespace(seed=42, temperature=0.8))
    manager.add_request(1, SimpleNamespace(seed=42, temperature=0.0))
    manager.add_request(2, SimpleNamespace(seed=None, temperature=0.8))
    manager.add_request(3, SimpleNamespace(seed=99, temperature=0.8))
    manager.apply_staged_writes()

    # Request 3 is mid-prefill (40 < 100): its reset stays pending and its
    # confidences remain 1.0 even after two confidence records alongside decodes.
    mid_batch = SimpleNamespace(
        num_reqs=4,
        idx_mapping=torch.tensor([0, 1, 2, 3], dtype=torch.int64, device=device),
        idx_mapping_np=np.array([0, 1, 2, 3], dtype=np.int64),
        num_computed_tokens_np=np.array([10, 10, 10, 0], dtype=np.int32),
        num_scheduled_tokens=np.array([6, 6, 6, 40], dtype=np.int32),
        prefill_len_np=np.array([10, 10, 10, 100], dtype=np.int32),
    )
    step_confs = torch.tensor(
        [
            [0.6, 0.6, 0.1, 0.1, 0.1],
            [0.9, 0.9, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2, 0.2, 0.2],
        ],
        dtype=torch.float32,
        device=device,
    )
    for _ in range(2):
        manager.record_confidences(step_confs, mid_batch)
    assert 3 in manager._pending_resets
    np.testing.assert_array_equal(
        manager._stale_confidences[manager._stale_idx].np[3],
        np.ones(5, dtype=np.float32),
    )

    # Alone, "seeded" (survival [0.6, 0.36, ...], row_cost=0.2) admits 2 drafts.
    alone_tokens = manager.get_num_tokens(
        {"seeded": 6}, {"seeded": [1, 2, 3, 4, 5]}
    )
    assert alone_tokens == 3
    manager.reallocate_drafts(
        ["seeded"], torch.tensor([0], dtype=torch.int64, device=device)
    )
    torch.cuda.synchronize(device)
    assert manager.get_verified_draft_counts(1).tolist() == [2]

    # Mixed with two greedy/unseeded requests whose current GPU confidences make
    # greedy_hi's all 5 slots (0.95^k) outrank seeded's 0.6^k: seeded still keeps
    # its exact 2-draft capacity while the two unpinned requests share their 2-draft
    # budget by survival.
    manager._confidence_probs[:3].copy_(
        torch.tensor(
            [
                [0.25, 0.25, 0.1, 0.1, 0.1],
                [0.95, 0.95, 0.95, 0.95, 0.95],
                [0.1, 0.1, 0.1, 0.1, 0.1],
            ],
            dtype=torch.float32,
            device=device,
        )
    )
    mixed_tokens = manager.get_num_tokens(
        {"seeded": 6, "greedy_hi": 6, "greedy_lo": 6},
        {
            "seeded": [1, 2, 3, 4, 5],
            "greedy_hi": [1, 2, 3, 4, 5],
            "greedy_lo": [1, 2, 3, 4, 5],
        },
    )
    assert mixed_tokens == 3 + 4
    manager.reallocate_drafts(
        ["seeded", "greedy_hi", "greedy_lo"],
        torch.tensor([0, 1, 2], dtype=torch.int64, device=device),
    )
    torch.cuda.synchronize(device)
    assert manager.get_verified_draft_counts(3).tolist() == [2, 2, 0]


def test_seeded_sampled_multi_step_schedule_and_batch_invariance(monkeypatch):
    if not torch.accelerator.is_available():
        pytest.skip("requires CUDA")
    monkeypatch.setattr(
        adaptive_module,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=1),
    )
    device = torch.device("cuda", 0)
    num_reqs = 8
    num_steps = 5
    decode_steps = 5
    rng = np.random.default_rng(12345)
    # Deterministic per-request, per-decode-step confidence trajectories.
    req_trajectories = rng.uniform(
        0.15, 0.98, size=(num_reqs, decode_steps, num_steps)
    ).astype(np.float32)

    def run_alone(req_idx: int) -> list[int]:
        state = SimpleNamespace(
            device=device,
            max_num_reqs=1,
            num_speculative_steps=num_steps,
            req_id_to_index={"r": 0},
            num_computed_tokens_np=np.array([0], dtype=np.int32),
            prefill_len=SimpleNamespace(np=np.array([20], dtype=np.int32)),
        )
        mgr = AdaptiveVerificationManager(
            state,
            torch.zeros(2, dtype=torch.int32, device=device),
            num_bonus_tokens=1,
            max_total_logits=64,
            row_cost=0.04,
        )
        mgr.add_request(0, SimpleNamespace(seed=100 + req_idx, temperature=0.8))
        mgr.apply_staged_writes()

        # Prefill completion step (step 0): proposes drafts for decode step 0.
        mgr.get_num_tokens({"r": 20}, {})
        mgr.reallocate_drafts(["r"], torch.tensor([0], dtype=torch.int64, device=device))
        prefill_batch = SimpleNamespace(
            num_reqs=1,
            idx_mapping=torch.tensor([0], dtype=torch.int64, device=device),
            idx_mapping_np=np.array([0], dtype=np.int64),
            num_computed_tokens_np=np.array([0], dtype=np.int32),
            num_scheduled_tokens=np.array([20], dtype=np.int32),
            prefill_len_np=np.array([20], dtype=np.int32),
        )
        mgr.record_confidences(
            torch.tensor(req_trajectories[req_idx, 0][None, :], device=device),
            prefill_batch,
        )
        state.num_computed_tokens_np[0] = 20

        capacities = []
        for k in range(1, decode_steps):
            mgr.get_num_tokens({"r": 1 + num_steps}, {"r": [1, 2, 3, 4, 5]})
            mgr.reallocate_drafts(
                ["r"], torch.tensor([0], dtype=torch.int64, device=device)
            )
            torch.cuda.synchronize(device)
            capacities.append(int(mgr.get_verified_draft_counts(1)[0].item()))
            decode_batch = SimpleNamespace(
                num_reqs=1,
                idx_mapping=torch.tensor([0], dtype=torch.int64, device=device),
                idx_mapping_np=np.array([0], dtype=np.int64),
                num_computed_tokens_np=np.array([20 + k], dtype=np.int32),
                num_scheduled_tokens=np.array([1 + num_steps], dtype=np.int32),
                prefill_len_np=np.array([20], dtype=np.int32),
            )
            mgr.record_confidences(
                torch.tensor(req_trajectories[req_idx, k][None, :], device=device),
                decode_batch,
            )
        return capacities

    alone_capacities = [run_alone(i) for i in range(num_reqs)]

    # Run all 8 requests under a staggered schedule with chunked prefill and a pause:
    # - r0..r3 finish prefill at engine step 0; r4..r7 run a mid-prefill chunk (10/20)
    #   at engine step 0 and finish prefill at engine step 1.
    # - r0..r1 pause at engine step 2 while r2..r7 decode.
    req_ids = [f"r{i}" for i in range(num_reqs)]
    state = SimpleNamespace(
        device=device,
        max_num_reqs=num_reqs,
        num_speculative_steps=num_steps,
        req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        num_computed_tokens_np=np.zeros(num_reqs, dtype=np.int32),
        prefill_len=SimpleNamespace(np=np.full(num_reqs, 20, dtype=np.int32)),
    )
    mgr = AdaptiveVerificationManager(
        state,
        torch.zeros(num_reqs + 1, dtype=torch.int32, device=device),
        num_bonus_tokens=1,
        max_total_logits=64,
        row_cost=0.04,
    )
    for i in range(num_reqs):
        mgr.add_request(i, SimpleNamespace(seed=100 + i, temperature=0.8))
    mgr.apply_staged_writes()

    req_step = np.zeros(num_reqs, dtype=np.int32)
    staggered_capacities: list[list[int]] = [[] for _ in range(num_reqs)]
    engine_step = 0
    while np.any(req_step < decode_steps):
        active = [
            i
            for i in range(num_reqs)
            if req_step[i] < decode_steps and not (engine_step == 2 and i in (0, 1))
        ]
        num_tokens_per_req = {}
        draft_tokens = {}
        sched_tokens_list = []
        comp_tokens_list = []
        for i in active:
            rid = req_ids[i]
            comp = int(state.num_computed_tokens_np[i])
            comp_tokens_list.append(comp)
            if comp < 20:
                tokens = 10 if (engine_step == 0 and i >= 4) else (20 - comp)
                num_tokens_per_req[rid] = tokens
                sched_tokens_list.append(tokens)
            else:
                num_tokens_per_req[rid] = 1 + num_steps
                draft_tokens[rid] = [1, 2, 3, 4, 5]
                sched_tokens_list.append(1 + num_steps)

        mgr.get_num_tokens(num_tokens_per_req, draft_tokens)
        idx_map = torch.tensor(active, dtype=torch.int64, device=device)
        mgr.reallocate_drafts([req_ids[i] for i in active], idx_map)
        torch.cuda.synchronize(device)
        verified = mgr.get_verified_draft_counts(len(active)).tolist()

        confs = []
        for pos, i in enumerate(active):
            comp = comp_tokens_list[pos]
            sched = sched_tokens_list[pos]
            if comp >= 20:
                staggered_capacities[i].append(int(verified[pos]))
            if comp + sched >= 20:
                k = int(req_step[i])
                confs.append(req_trajectories[i, k])
                req_step[i] += 1
            else:
                # Mid-prefill garbage confidences must be ignored and reset to 1.0.
                confs.append(np.full(num_steps, 0.12, dtype=np.float32))
            state.num_computed_tokens_np[i] = comp + sched

        batch = SimpleNamespace(
            num_reqs=len(active),
            idx_mapping=idx_map,
            idx_mapping_np=np.array(active, dtype=np.int64),
            num_computed_tokens_np=np.array(comp_tokens_list, dtype=np.int32),
            num_scheduled_tokens=np.array(sched_tokens_list, dtype=np.int32),
            prefill_len_np=np.full(len(active), 20, dtype=np.int32),
        )
        mgr.record_confidences(
            torch.tensor(np.stack(confs), dtype=torch.float32, device=device),
            batch,
        )
        engine_step += 1

    assert staggered_capacities == alone_capacities
