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
    ("tp_invariant_confidence", "rank_order_reduce_scatter", "expected_broadcasts"),
    [
        (False, False, 1),
        (False, True, 1),
        (True, False, 1),
        (True, True, 0),
    ],
)
def test_tp_invariant_confidence_elides_broadcast_only_when_rank_ordered(
    monkeypatch,
    tp_invariant_confidence,
    rank_order_reduce_scatter,
    expected_broadcasts,
):
    local = torch.tensor([[0.75]], dtype=torch.float32)
    manager = make_manager(local.numpy().copy(), row_cost=0.1)
    manager.tp_invariant_confidence = tp_invariant_confidence
    manager.req_states.device = torch.device("cpu")
    manager._confidence_buffer = torch.zeros_like(local).unsqueeze(0)
    manager._confidence_probs = manager._confidence_buffer[0]
    manager._planned_probs = manager._confidence_buffer[-1]
    manager._pending_resets = []
    manager._stale_confidences = [
        SimpleNamespace(
            np=np.ones((1, 1), dtype=np.float32),
            gpu=torch.empty_like(local),
            copy_to_cpu=lambda: None,
        )
        for _ in range(2)
    ]
    manager._copy_events = [
        SimpleNamespace(synchronize=lambda: None, record=lambda: None) for _ in range(2)
    ]
    manager._copy_stream = SimpleNamespace(wait_stream=lambda _stream: None)
    calls = []

    def broadcast(tensor, src):
        assert src == 0
        calls.append(tensor)

    monkeypatch.setattr(
        adaptive_module,
        "get_tp_group",
        lambda: SimpleNamespace(
            world_size=2,
            broadcast=broadcast,
            device_communicator=SimpleNamespace(
                rank_order_reduce_scatter=rank_order_reduce_scatter
            ),
        ),
    )
    monkeypatch.setattr(adaptive_module, "gpu_sync_allowed", nullcontext)
    monkeypatch.setattr(adaptive_module.torch.cuda, "current_stream", lambda _: None)
    monkeypatch.setattr(adaptive_module, "stream", lambda *_: nullcontext())
    monkeypatch.setattr(
        adaptive_module.tilelang_inputs,
        "scatter_rows",
        lambda idx_mapping, source, state, gather=False: state.__setitem__(
            (idx_mapping, slice(0, source.shape[1])), source
        ),
    )
    batch = SimpleNamespace(num_reqs=1, idx_mapping=torch.tensor([0]))
    manager.record_confidences(local, batch)
    assert len(calls) == expected_broadcasts
    torch.testing.assert_close(manager._confidence_probs, local, rtol=0, atol=0)



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


def test_warm_tail_shapes_runs_each_eager_size_once_untimed(monkeypatch):
    manager = AdaptiveVerificationManager.__new__(AdaptiveVerificationManager)
    manager.req_states = SimpleNamespace(max_num_batched_tokens=1024)
    runs = []
    flushes = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: flushes.append("sync"))
    monkeypatch.setattr(
        torch.accelerator, "empty_cache", lambda: flushes.append("empty")
    )
    manager.warm_tail_shapes(lambda **kwargs: runs.append(kwargs), [1, 2, 8])
    assert runs == [{"num_tokens": n} for n in (12, 16, 32, 64, 128, 256, 512, 1024)]
    assert flushes == ["sync", "empty"] * 8


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


def test_the_budget_takes_the_schedulers_drafts():
    # Step t reported alloc=16 and planned_confidences[0:16]=0.95, so the
    # scheduler laid out 16 draft placeholders for Step t+2. Whatever Step t+1
    # proposes, the budget takes the scheduler's 16 (the rows past a narrower
    # proposal verify its -1 placeholders, which verification rejects): no rank's
    # own view of the latest proposal enters the step's layout.
    manager = make_manager(
        np.full((2, 16), 0.95, dtype=np.float32),
        row_cost=0.02,
    )
    manager.num_draft_steps = 5

    tokens = manager.get_num_tokens(
        {"low": 17},
        {"low": [-1] * 16},
    )
    valid_drafts, num_non_draft_tokens, draft_budget = manager._batch_budget
    assert valid_drafts == {"low": 16}
    assert num_non_draft_tokens == {"low": 1}
    assert draft_budget == 16
    assert tokens == 17


def test_mid_prefill_chunks_keep_initial_unit_confidence_until_prefill_completes(
    monkeypatch,
):
    steps = 3
    manager = make_manager(np.ones((2, steps), dtype=np.float32), row_cost=0.02)
    manager.req_states.device = torch.device("cpu")
    manager._confidence_buffer = torch.ones((2, 2, steps), dtype=torch.float32)
    manager._confidence_probs = manager._confidence_buffer[0]
    manager._planned_probs = manager._confidence_buffer[-1]
    manager._pending_resets = []
    manager._new_rows = []

    class CopyBuffer:
        def __init__(self):
            self.np = np.ones((2, steps), dtype=np.float32)
            self.gpu = torch.ones((2, steps), dtype=torch.float32)

        def copy_to_cpu(self):
            self.np[:] = self.gpu.numpy()

    manager._stale_confidences = [CopyBuffer(), CopyBuffer()]
    manager._stale_idx = 0
    manager._copy_events = [
        SimpleNamespace(synchronize=lambda: None, record=lambda: None) for _ in range(2)
    ]
    manager._copy_stream = SimpleNamespace(wait_stream=lambda _stream: None)
    monkeypatch.setattr(
        adaptive_module,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=1, broadcast=lambda *_a, **_k: None),
    )
    monkeypatch.setattr(adaptive_module, "gpu_sync_allowed", nullcontext)
    monkeypatch.setattr(adaptive_module.torch.cuda, "current_stream", lambda _: None)
    monkeypatch.setattr(adaptive_module, "stream", lambda *_: nullcontext())
    monkeypatch.setattr(
        adaptive_module.tilelang_inputs,
        "scatter_rows",
        lambda idx, src, state, gather=False: state.__setitem__(
            (idx, slice(None, src.shape[1])), src
        ),
    )
    monkeypatch.setattr(
        adaptive_module.tilelang_inputs,
        "fill_rows",
        lambda rows, state, value: state.__setitem__(rows, value),
    )
    monkeypatch.setattr(
        adaptive_module, "async_tensor_h2d", lambda v, **_: torch.tensor(v)
    )

    manager.add_request(0)
    manager.add_request(1)
    manager.apply_staged_writes()

    # Slot 0 decodes with 0.8 confidence; slot 1 runs two mid-prefill chunks with
    # garbage 0.01 mid-prompt confidences before completing prefill on chunk 3.
    mid_probs = torch.tensor([[0.8, 0.8, 0.8], [0.01, 0.01, 0.01]])
    for computed in (0, 20):
        batch = SimpleNamespace(
            num_reqs=2,
            idx_mapping=torch.tensor([0, 1]),
            idx_mapping_np=np.array([0, 1], dtype=np.intp),
            has_prefill=True,
            num_computed_prefill_tokens_np=np.array([10, computed], dtype=np.int32),
            num_scheduled_tokens=np.array([1, 20], dtype=np.int32),
            prefill_len_np=np.array([10, 60], dtype=np.int32),
        )
        manager.record_confidences(mid_probs, batch, planned=mid_probs)
        assert (manager._confidence_buffer[:, 1] == 1.0).all()
        np.testing.assert_array_equal(
            manager._stale_confidences[manager._stale_idx].np[1],
            np.ones(steps, dtype=np.float32),
        )

    # Chunk 3 completes slot 1's prefill (40 + 20 == 60) and emits its first real
    # draft confidences (0.7). Slot 1's first decode step still sees 1.0 on the
    # host, and the next step sees 0.7.
    final_probs = torch.tensor([[0.8, 0.8, 0.8], [0.7, 0.7, 0.7]])
    final_batch = SimpleNamespace(
        num_reqs=2,
        idx_mapping=torch.tensor([0, 1]),
        idx_mapping_np=np.array([0, 1], dtype=np.intp),
        has_prefill=True,
        num_computed_prefill_tokens_np=np.array([10, 40], dtype=np.int32),
        num_scheduled_tokens=np.array([1, 20], dtype=np.int32),
        prefill_len_np=np.array([10, 60], dtype=np.int32),
    )
    manager.record_confidences(final_probs, final_batch, planned=final_probs)
    torch.testing.assert_close(manager._confidence_probs[1], torch.full((steps,), 0.7))
    np.testing.assert_array_equal(
        manager._stale_confidences[manager._stale_idx].np[1],
        np.ones(steps, dtype=np.float32),
    )

    decode_batch = SimpleNamespace(
        num_reqs=2,
        idx_mapping=torch.tensor([0, 1]),
        idx_mapping_np=np.array([0, 1], dtype=np.intp),
        has_prefill=False,
    )
    manager.record_confidences(final_probs, decode_batch, planned=final_probs)
    np.testing.assert_allclose(
        manager._stale_confidences[manager._stale_idx].np[1],
        np.full(steps, 0.7, dtype=np.float32),
    )



def test_dead_rows_tau_excludes_sub_tau_slots_from_host_budget():
    """Host get_num_tokens must not admit draft slots that dead_rows_tau kills on GPU."""
    confidences = np.array([[0.78, 0.72, 0.71, 0.71, 0.73]], dtype=np.float32)
    manager = make_manager(confidences, row_cost=0.01)
    manager.req_states.req_id_to_index = {"r0": 0}
    manager.dead_rows_tau = 0.30

    total = manager.get_num_tokens({"r0": 6}, {"r0": [10, 11, 12, 13, 14]})
    _, _, draft_budget = manager._batch_budget
    # Survival is [0.78, 0.5616, 0.3987, 0.2831, 0.2067]; only the first 3
    # positions are >= 0.30, so host budget must be 3 rather than 5.
    assert draft_budget == 3
    assert total == 4


def test_calibrated_confidence_matches_dead_rows_tau_to_c1_survival():
    """With calibrated confidence_head survival, c1 admits positions 0..3 on typical
    requests, 0..4 on high-confidence requests, and prunes low-confidence tails."""
    # Typical calibrated c1 conditional probs -> survival [0.677, 0.376, 0.149, 0.0686, 0.0438]
    typical = np.array([[0.677, 0.556, 0.396, 0.460, 0.638]], dtype=np.float32)
    manager = make_manager(typical, row_cost=0.035)
    manager.req_states.req_id_to_index = {"r0": 0}
    manager.dead_rows_tau = 0.20
    manager.calibrated_confidence = True

    tau = manager.effective_dead_rows_tau()
    assert np.isclose(tau, 2.0 * 0.035 / (1.0 + 3.0 * 0.035))
    total = manager.get_num_tokens({"r0": 6}, {"r0": [1, 2, 3, 4, 5]})
    _, _, draft_budget = manager._batch_budget
    assert draft_budget == 4
    assert total == 5

    # High-confidence request -> survival [0.85, 0.68, 0.51, 0.36, 0.25] admits all 5
    high = np.array([[0.85, 0.80, 0.75, 0.70, 0.70]], dtype=np.float32)
    manager_hi = make_manager(high, row_cost=0.035)
    manager_hi.req_states.req_id_to_index = {"r0": 0}
    manager_hi.dead_rows_tau = 0.20
    manager_hi.calibrated_confidence = True
    manager_hi.get_num_tokens({"r0": 6}, {"r0": [1, 2, 3, 4, 5]})
    assert manager_hi._batch_budget[2] == 5

    # Low-confidence branching request -> survival [0.45, 0.18, 0.054, 0.016, 0.005] admits 2
    low = np.array([[0.45, 0.40, 0.30, 0.30, 0.30]], dtype=np.float32)
    manager_lo = make_manager(low, row_cost=0.035)
    manager_lo.req_states.req_id_to_index = {"r0": 0}
    manager_lo.dead_rows_tau = 0.20
    manager_lo.calibrated_confidence = True
    manager_lo.get_num_tokens({"r0": 6}, {"r0": [1, 2, 3, 4, 5]})
    assert manager_lo._batch_budget[2] == 2


def test_get_verified_draft_counts_clamps_to_dead_row_cut_and_clears_when_no_drafts():
    """get_verified_draft_counts must clamp to live rows and dead_row_args must
    return None when a step admitted zero drafts."""
    manager = make_manager(np.ones((2, 5), dtype=np.float32), row_cost=0.035)
    manager._batch_draft_capacity = torch.tensor([5, 4, 3], dtype=torch.int32)
    manager._confidence_probs = torch.ones((3, 5), dtype=torch.float32)
    manager._dead_row_cut = torch.tensor([3, 5, 1], dtype=torch.int32)
    manager.dead_rows_tau = 0.20

    manager._has_admitted_drafts = False
    assert manager.dead_row_args(torch.zeros(12, dtype=torch.bool), 2) is None

    manager._has_admitted_drafts = True
    dead = manager.dead_row_args(torch.zeros(12, dtype=torch.bool), 2)
    assert dead is not None

    verified = manager.get_verified_draft_counts(2, dead.cut)
    assert verified.tolist() == [3, 4]
    assert manager._batch_draft_capacity[:2].tolist() == [3, 4]


def test_high_concurrency_row_cost_knee_admits_coalesced_drafts():
    """Above the 16-row Tensor Core MMA tile knee, coalesced rows face a lower
    marginal cost and admit profitable medium-confidence draft slots."""
    confidences = np.tile(
        np.array([[0.78, 0.72, 0.71, 0.71, 0.73]], dtype=np.float32), (8, 1)
    )
    manager = make_manager(confidences, row_cost=0.035)
    req_ids = [f"r{i}" for i in range(8)]
    manager.req_states.req_id_to_index = {rid: i for i, rid in enumerate(req_ids)}
    manager.dead_rows_tau = 0.20

    num_tokens_per_req = {rid: 6 for rid in req_ids}
    draft_tokens = {rid: [1, 2, 3, 4, 5] for rid in req_ids}

    # Under a flat linear slope (scale=1.0), only 3 drafts per request (24 total)
    # pass the linear cost threshold at c8.
    manager.high_row_cost_scale = 1.0
    manager.get_num_tokens(num_tokens_per_req, draft_tokens)
    _, _, flat_budget = manager._batch_budget
    assert flat_budget == 24

    # With the 16-row Tensor Core tile knee (scale=0.55), positions 0..3 across
    # all 8 requests (32 draft rows, total 40 rows) are admitted.
    manager.high_row_cost_scale = 0.55
    total = manager.get_num_tokens(num_tokens_per_req, draft_tokens)
    _, _, knee_budget = manager._batch_budget
    assert knee_budget == 32
    assert total == 40

    # On TP3 (scale=0.85, calibrated survival floor=0.12), c8 trims to 24 draft
    # rows (total 32 rows). The floor does not bind: position 4 survives 0.2067.
    manager.high_row_cost_scale = 0.85
    manager.min_calibrated_dead_rows_tau = 0.12
    manager.calibrated_confidence = True
    assert np.isclose(manager.effective_dead_rows_tau(), 0.12)
    total_tp3 = manager.get_num_tokens(num_tokens_per_req, draft_tokens)
    _, _, tp3_budget = manager._batch_budget
    assert tp3_budget == 24
    assert total_tp3 == 32

    # TP4's constants (scale=0.55, no floor) keep the calibrated c8 budget.
    manager.high_row_cost_scale = 0.55
    manager.min_calibrated_dead_rows_tau = 0.0
    assert manager.get_num_tokens(num_tokens_per_req, draft_tokens) == 40
    assert manager._batch_budget[2] == 32


@pytest.mark.parametrize(
    ("scale", "floor", "budget"), [(0.55, 0.0, 4), (0.85, 0.12, 3)], ids=["tp4", "tp3"]
)
def test_tp3_survival_floor_prunes_the_typical_c1_tail(scale, floor, budget):
    """The calibrated c1 break-even (0.0633 at row cost 0.035) admits position 3
    (survival 0.0686) on TP4; TP3's floor of 0.12 prunes it (4 -> 3 drafts)."""
    typical = np.array([[0.677, 0.556, 0.396, 0.460, 0.638]], dtype=np.float32)
    manager = make_manager(typical, row_cost=0.035)
    manager.req_states.req_id_to_index = {"r0": 0}
    manager.dead_rows_tau = 0.20
    manager.calibrated_confidence = True
    manager.high_row_cost_scale = scale
    manager.min_calibrated_dead_rows_tau = floor
    assert np.isclose(
        manager.effective_dead_rows_tau(),
        max(floor, 2.0 * 0.035 / (1.0 + 3.0 * 0.035)),
    )
    total = manager.get_num_tokens({"r0": 6}, {"r0": [1, 2, 3, 4, 5]})
    assert manager._batch_budget[2] == budget
    assert total == 1 + budget


def test_tp3_constants_keep_the_single_request_tile_spill_cap():
    """A single request prices rows past the knee with the tile-spill scale on
    every TP size: TP3's multi-request scale does not reopen the 17-row step."""
    manager = make_manager(np.full((1, 16), 0.95, dtype=np.float32), row_cost=0.035)
    manager.req_states.req_id_to_index = {"r0": 0}
    manager.num_draft_steps = 5
    manager.dead_rows_tau = 0.20
    manager.calibrated_confidence = True
    manager.high_row_cost_scale = 0.85
    manager.min_calibrated_dead_rows_tau = 0.12
    assert manager.get_num_tokens({"r0": 17}, {"r0": list(range(16))}) == 16
    assert manager._batch_budget[2] == 15


@pytest.mark.parametrize(
    ("world_size", "scale", "floor"), [(1, 0.55, 0.0), (3, 0.85, 0.12), (4, 0.55, 0.0)]
)
def test_tp_size_selects_the_row_cost_constants(monkeypatch, world_size, scale, floor):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    monkeypatch.setattr(
        adaptive_module,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=world_size),
    )
    req_states = SimpleNamespace(
        num_speculative_steps=5, device=torch.device("cuda"), max_num_reqs=2
    )
    manager = AdaptiveVerificationManager(
        req_states,
        torch.zeros(3, dtype=torch.int32, device="cuda"),
        1,
        max_total_logits=1024,
        row_cost=0.035,
    )
    assert manager.high_row_cost_scale == scale
    assert manager.min_calibrated_dead_rows_tau == floor
    assert manager.single_req_tile_spill_scale == 2.0


def test_sampling_states_unseeded_requests_stay_lockstep_across_tp_ranks():
    """Unseeded temperature > 0 requests must draw identical per-request seeds on
    every TP rank even when rank-local callers perturb the global NumPy RNG."""
    from vllm.sampling_params import SamplingParams
    from vllm.v1.worker.gpu.sample.states import SamplingStates

    np.random.seed(101)
    rank0 = SamplingStates.__new__(SamplingStates)
    rank0.max_num_reqs = 8
    rank0.vocab_size = 1024
    rank0.temperature = SimpleNamespace(np=np.zeros(8, dtype=np.float32))
    rank0.top_p = SimpleNamespace(np=np.ones(8, dtype=np.float32))
    rank0.top_k = SimpleNamespace(np=np.full(8, 1024, dtype=np.int32))
    rank0.min_p = SimpleNamespace(np=np.zeros(8, dtype=np.float32))
    rank0.seeds = SimpleNamespace(np=np.zeros(8, dtype=np.int64))
    rank0.seeds_set = np.zeros(8, dtype=bool)
    rank0.num_logprobs = np.full(8, -1, dtype=np.int32)
    rank0._seed_rng = np.random.default_rng(0)

    # Perturb the global NumPy RNG asymmetrically on rank 1 before and between requests.
    np.random.seed(99999)
    _ = np.random.randint(0, 1000, size=37)
    rank1 = SamplingStates.__new__(SamplingStates)
    rank1.max_num_reqs = 8
    rank1.vocab_size = 1024
    rank1.temperature = SimpleNamespace(np=np.zeros(8, dtype=np.float32))
    rank1.top_p = SimpleNamespace(np=np.ones(8, dtype=np.float32))
    rank1.top_k = SimpleNamespace(np=np.full(8, 1024, dtype=np.int32))
    rank1.min_p = SimpleNamespace(np=np.zeros(8, dtype=np.float32))
    rank1.seeds = SimpleNamespace(np=np.zeros(8, dtype=np.int64))
    rank1.seeds_set = np.zeros(8, dtype=bool)
    rank1.num_logprobs = np.full(8, -1, dtype=np.int32)
    rank1._seed_rng = np.random.default_rng(0)

    for slot in range(8):
        params = SamplingParams(temperature=0.8, top_p=0.95, seed=None)
        rank0.add_request(slot, params)
        _ = np.random.random(13)
        rank1.add_request(slot, params)

    np.testing.assert_array_equal(rank0.seeds.np, rank1.seeds.np)
    assert not rank0.seeds_set.any()
    assert len(set(rank0.seeds.np.tolist())) == 8


def test_sampler_multi_rank_tp_disables_unseeded_flashinfer_rng(monkeypatch):
    """When TP > 1, top-k/top-p sampling must route through the counter-based
    seeded sampler rather than FlashInfer's rank-local device RNG."""
    import vllm.distributed.parallel_state as parallel_state
    from vllm.v1.worker.gpu.sample.sampler import Sampler

    sampler = Sampler.__new__(Sampler)
    sampler.use_flashinfer = True
    sampler.logprobs_mode = "raw_logprobs"
    sampler.needs_logits_processing = np.array([True], dtype=bool)
    sampler.needs_non_temperature_logits_processing = np.array([True], dtype=bool)
    sampler.sampling_states = SimpleNamespace(
        get_top_k_top_p=lambda expanded, idx_np: (torch.tensor([16]), None),
        any_greedy=lambda idx_np: False,
        any_explicit_seed=lambda idx_np: False,
    )
    sampler.apply_sampling_params = lambda logits, *args, **kwargs: logits
    recorded_use_fi = []

    def fake_sample_random(
        processed_logits,
        expanded_idx_mapping,
        idx_mapping_np,
        pos,
        top_k,
        top_p,
        use_flashinfer,
        apply_temperature=False,
    ):
        recorded_use_fi.append(bool(use_flashinfer))
        return torch.zeros(1, dtype=torch.int64), processed_logits

    sampler._sample_random = fake_sample_random
    logits = torch.zeros((1, 32), dtype=torch.float32)
    idx = torch.zeros(1, dtype=torch.int32)
    idx_np = np.array([0], dtype=np.int32)
    pos = torch.zeros(1, dtype=torch.int64)
    input_ids = torch.zeros(1, dtype=torch.int64)

    monkeypatch.setattr(parallel_state, "model_parallel_is_initialized", lambda: True)
    monkeypatch.setattr(
        parallel_state, "get_tensor_model_parallel_world_size", lambda: 1
    )
    sampler.sample(logits, idx, idx, idx_np, pos, input_ids, idx)

    monkeypatch.setattr(
        parallel_state, "get_tensor_model_parallel_world_size", lambda: 4
    )
    sampler.sample(logits, idx, idx, idx_np, pos, input_ids, idx)

    assert recorded_use_fi == [True, False]


def test_draft_model_speculator_tp_invariant_confidence_tracks_estimator():
    """DraftModelSpeculator and DFlashSpeculator report tp_invariant_confidence
    when OnlineAcceptanceEstimator is active."""
    from vllm.v1.worker.gpu.spec_decode.acceptance_estimator import (
        OnlineAcceptanceEstimator,
    )
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
    from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

    assert OnlineAcceptanceEstimator.tp_invariant is True
    spec = DFlashSpeculator.__new__(DFlashSpeculator)
    spec.acceptance_estimator = None
    assert spec.tp_invariant_confidence is False
    assert DraftModelSpeculator.tp_invariant_confidence.fget(spec) is False

    spec.acceptance_estimator = SimpleNamespace(tp_invariant=True)
    assert spec.tp_invariant_confidence is True
    assert DraftModelSpeculator.tp_invariant_confidence.fget(spec) is True

    dflash = DFlashSpeculator.__new__(DFlashSpeculator)
    dflash.acceptance_estimator = SimpleNamespace(tp_invariant=True)
    assert dflash.tp_invariant_confidence is True


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="SamplingStates keeps UVA buffers"
)
def test_unseeded_request_seeds_follow_configured_seed_across_tp_ranks():
    """Unseeded requests' seeds come from a stream keyed by the configured seed
    (ModelConfig.seed): ranks built with the same seed assign the same seeds however
    each perturbs the global NumPy RNG, a different seed gives different seeds, and no
    configured seed is vLLM's default seed 0."""
    from vllm.sampling_params import SamplingParams
    from vllm.v1.worker.gpu.sample.states import SamplingStates

    def unseeded_seeds(seed, perturb):
        np.random.seed(perturb)
        _ = np.random.randint(0, 1000, size=perturb % 41 + 1)
        states = SamplingStates(8, 1024, seed=seed)
        for slot in range(8):
            states.add_request(slot, SamplingParams(temperature=0.8, seed=None))
            _ = np.random.random(perturb % 7 + 1)
        assert not states.seeds_set.any()
        return states.seeds.np.copy()

    rank0 = unseeded_seeds(1234, perturb=1)
    rank1 = unseeded_seeds(1234, perturb=99999)
    np.testing.assert_array_equal(rank0, rank1)
    assert len(set(rank0.tolist())) == 8

    other_seed = unseeded_seeds(4321, perturb=1)
    assert set(rank0.tolist()).isdisjoint(other_seed.tolist())
    np.testing.assert_array_equal(other_seed, unseeded_seeds(4321, perturb=12345))

    np.testing.assert_array_equal(unseeded_seeds(None, 5), unseeded_seeds(0, 7))


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Sampler keeps UVA buffers"
)
def test_sampler_hands_configured_seed_to_sampling_states():
    """The model runner's sampler kwargs carry ModelConfig.seed; the Sampler must
    key its unseeded-request seeds by it."""
    from vllm.sampling_params import SamplingParams
    from vllm.v1.worker.gpu.sample.sampler import Sampler
    from vllm.v1.worker.gpu.states import RequestState

    device = torch.device("cuda")

    def first_unseeded_seed(seed):
        req_states = RequestState(
            max_num_reqs=4,
            max_model_len=64,
            max_num_batched_tokens=16,
            num_speculative_steps=1,
            vocab_size=128,
            device=device,
        )
        sampler = Sampler(
            max_num_reqs=4,
            vocab_size=128,
            device=device,
            req_states=req_states,
            seed=seed,
        )
        sampler.add_request(0, 1, SamplingParams(temperature=0.8, seed=None))
        return int(sampler.sampling_states.seeds.np[0])

    assert first_unseeded_seed(7) == first_unseeded_seed(7)
    assert first_unseeded_seed(7) != first_unseeded_seed(8)


def test_single_request_tile_spill_caps_16th_copy_draft_at_16_rows() -> None:
    # Single-request (c1) initial copy proposal at 0.95 confidence across 16 drafts:
    # capping at 15 drafts (16 rows = 1 MMA tile) beats spilling 1 row into a 2nd MMA tile.
    mgr_c1 = make_manager(np.full((1, 16), 0.95, dtype=np.float32), row_cost=0.035)
    mgr_c1.req_states.req_id_to_index = {"r0": 0}
    mgr_c1.num_draft_steps = 5
    mgr_c1.dead_rows_tau = 0.0
    total_c1 = mgr_c1.get_num_tokens({"r0": 17}, {"r0": list(range(16))})
    assert mgr_c1._batch_budget[2] == 15
    assert total_c1 == 16

    # Once online copy accuracy reaches >= 0.97 on a sustained streak, or when 2 requests
    # share the batch, all 16 copy drafts per request are admitted.
    mgr_c1_hi = make_manager(np.full((1, 16), 0.97, dtype=np.float32), row_cost=0.035)
    mgr_c1_hi.req_states.req_id_to_index = {"r0": 0}
    mgr_c1_hi.num_draft_steps = 5
    mgr_c1_hi.dead_rows_tau = 0.0
    total_c1_hi = mgr_c1_hi.get_num_tokens({"r0": 17}, {"r0": list(range(16))})
    assert mgr_c1_hi._batch_budget[2] == 16
    assert total_c1_hi == 17

    mgr_c2 = make_manager(np.full((2, 16), 0.95, dtype=np.float32), row_cost=0.035)
    mgr_c2.req_states.req_id_to_index = {"r0": 0, "r1": 1}
    mgr_c2.num_draft_steps = 5
    mgr_c2.dead_rows_tau = 0.0
    total_c2 = mgr_c2.get_num_tokens(
        {"r0": 17, "r1": 17},
        {"r0": list(range(16)), "r1": list(range(16))},
    )
    assert mgr_c2._batch_budget[2] == 32
    assert total_c2 == 34
