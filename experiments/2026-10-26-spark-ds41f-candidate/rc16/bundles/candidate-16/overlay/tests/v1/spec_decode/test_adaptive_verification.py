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
    # Without copy drafts the planned confidences are the current ones.
    manager._confidence_buffer = torch.empty((1, *local.shape))
    manager._confidence_probs = manager._confidence_buffer[0]
    manager._planned_probs = manager._confidence_buffer[-1]
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


def test_planned_confidences_reach_the_budget_and_the_current_the_ranking(
    monkeypatch,
):
    """Copy drafts: the host budget reads a later proposal's planned confidences,
    the device ranking the current ones; one broadcast keeps both TP-wide, and a
    new request's planned row starts at 1 like its current one."""
    steps = 4
    current = torch.tensor([[0.9, 0.9, 0.0, 0.0]])
    planned = torch.tensor([[0.9, 0.9, 0.9, 0.9]])
    manager = make_manager(np.ones((2, steps), dtype=np.float32))
    manager.req_states.device = torch.device("cpu")
    manager._confidence_buffer = torch.full((2, 2, steps), -1.0)
    manager._confidence_probs = manager._confidence_buffer[0]
    manager._planned_probs = manager._confidence_buffer[-1]
    manager._pending_resets = []
    manager._new_rows = []

    class CopyBuffer:
        def __init__(self):
            self.np = np.ones((2, steps), dtype=np.float32)
            self.gpu = torch.empty((2, steps))

        def copy_to_cpu(self):
            self.np[:] = self.gpu.numpy()

    manager._stale_confidences = [CopyBuffer(), CopyBuffer()]
    manager._stale_idx = 0
    manager._copy_events = [
        SimpleNamespace(synchronize=lambda: None, record=lambda: None) for _ in range(2)
    ]
    manager._copy_stream = SimpleNamespace(wait_stream=lambda _stream: None)
    broadcasts = []
    monkeypatch.setattr(
        adaptive_module,
        "get_tp_group",
        lambda: SimpleNamespace(
            world_size=2, broadcast=lambda t, src: broadcasts.append(t)
        ),
    )
    monkeypatch.setattr(adaptive_module, "gpu_sync_allowed", nullcontext)
    monkeypatch.setattr(adaptive_module.torch.cuda, "current_stream", lambda _: None)
    monkeypatch.setattr(adaptive_module, "stream", lambda *_: nullcontext())

    def scatter_rows(idx_mapping, source, state, gather=False):
        state[idx_mapping, : source.shape[1]] = source

    def fill_rows(rows, state, value):
        state[rows] = value

    monkeypatch.setattr(adaptive_module.tilelang_inputs, "scatter_rows", scatter_rows)
    monkeypatch.setattr(adaptive_module.tilelang_inputs, "fill_rows", fill_rows)
    monkeypatch.setattr(
        adaptive_module, "async_tensor_h2d", lambda v, **_: torch.tensor(v)
    )
    batch = SimpleNamespace(num_reqs=1, idx_mapping=torch.tensor([0]))
    for _ in range(2):
        manager.record_confidences(current, batch, planned=planned)
    assert [t is manager._confidence_buffer for t in broadcasts] == [True, True]
    torch.testing.assert_close(manager._confidence_probs[0], current[0])
    np.testing.assert_array_equal(
        manager._stale_confidences[manager._stale_idx].np[0], planned[0].numpy()
    )
    # A request takes slot 1: both of its rows start at 1.
    manager.add_request(1)
    manager.apply_staged_writes()
    assert (manager._confidence_buffer[:, 1] == 1).all()


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


def test_live_rejected_clamps_dead_rows_out_of_estimator_observation():
    # Request 0: capacity=5 (from 1-step stale budget), dead_row_cut=2, and all 2
    # live drafts are accepted (num_sampled=3, num_rejected=3). Without clamping,
    # step 2 (a dead row skipped by MoE) is graded as a false rejection.
    # Request 1: capacity=5, dead_row_cut=2, 1 live draft accepted and 1 live
    # draft rejected (num_sampled=2, num_rejected=4) -> 1 live rejection.
    # Request 2: capacity=5, dead_row_cut=0, 0 live drafts (num_sampled=1,
    # num_rejected=5) -> 0 live rejections.
    dead_row_cut = torch.tensor([2, 2, 0], dtype=torch.int32)
    num_sampled = torch.tensor([3, 2, 1], dtype=torch.int32)
    num_rejected = torch.tensor([3, 4, 5], dtype=torch.int32)
    live_rejected = (dead_row_cut - (num_sampled - 1).clamp_min(0)).clamp_min(0)
    observed_rejected = torch.minimum(num_rejected, live_rejected)
    assert observed_rejected.tolist() == [0, 1, 0]


def test_slot_draft_limits_cap_stale_wide_scheduled_drafts():
    # Step t reported alloc=16 and planned_confidences[0:16]=0.95, so the
    # scheduler laid out 16 draft placeholders for Step t+2. When Step t+1
    # rejects the copy and falls back to DSpark (5 valid drafts, columns 5:16
    # zero-padded), slot_draft_limits[slot]=5 caps Step t+2's eligible drafts to
    # 5 so zero-padded slots are never budgeted or verified.
    manager = make_manager(
        np.full((2, 16), 0.95, dtype=np.float32),
        row_cost=0.02,
    )
    manager.num_draft_steps = 5
    limits = np.array([5, 16], dtype=np.int32)
    manager._slot_draft_limits = lambda: limits

    tokens = manager.get_num_tokens(
        {"low": 17},
        {"low": [-1] * 16},
    )
    valid_drafts, num_non_draft_tokens, draft_budget = manager._batch_budget
    assert valid_drafts == {"low": 5}
    assert num_non_draft_tokens == {"low": 1}
    assert draft_budget == 5
    assert tokens == 6
