# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adversarial CPU ownership checks against the installed serving source.

Transport and kernels are controlled doubles; scheduler and ownership methods
are the actual implementation. These are not CUDA or serving-performance tests.
"""

from collections import deque
from concurrent.futures import CancelledError
from contextlib import nullcontext
from types import SimpleNamespace as NS
import gc
import weakref

import numpy as np
import pytest
import torch

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus
from vllm.v1.executor.multiproc_executor import FutureWrapper
from vllm.v1.worker.gpu import pp_utils
from vllm.v1.worker.gpu import async_utils
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.spec_decode import adaptive_verification as av


@pytest.mark.parametrize("cancelled", [False, True])
@pytest.mark.parametrize("old_error", [False, True])
def test_late_rpc_result_is_drained_by_its_original_future(cancelled, old_error):
    queue = deque()
    replies = deque([RuntimeError("old request error") if old_error else "old tokens",
                     "replacement tokens"])
    deliveries = []

    def receive():
        reply = replies.popleft()
        deliveries.append(reply)
        if isinstance(reply, Exception):
            raise reply
        return reply

    old = FutureWrapper(queue, receive)
    if cancelled:
        assert old.cancel()
    replacement = FutureWrapper(queue, receive)
    # Request replacement is already submitted; its result must still drain
    # the prior RPC exactly once, even when nobody wants that prior result.
    assert replacement.result() == "replacement tokens"
    assert len(deliveries) == 2 and not replies and not queue
    if cancelled:
        with pytest.raises(CancelledError):
            old.result()
    elif old_error:
        with pytest.raises(RuntimeError, match="old request error"):
            old.result()
    else:
        assert old.result() == "old tokens"


def test_delayed_verification_output_keeps_its_snapshot_and_releases_tensors(monkeypatch):
    pending = []

    def copy_later(tensor):
        destination = np.empty(tuple(tensor.shape), dtype=tensor.numpy().dtype)
        pending.append((tensor, destination))
        return destination

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self, *_):
            pass

        def synchronize(self):
            for tensor, destination in pending:
                destination[:] = tensor.numpy()
            pending.clear()

    monkeypatch.setattr(async_utils, "async_copy_to_np", copy_later)
    monkeypatch.setattr(async_utils, "stream", lambda *_: nullcontext())
    monkeypatch.setattr(async_utils.torch.cuda, "Event", Event)
    copy_stream = NS(wait_stream=lambda _: None)
    counts = torch.tensor([3, 1], dtype=torch.int32)
    sampled = torch.tensor([[101, 102, 103, 104], [201, 202, -1, -1]])
    sampled_ref = weakref.ref(sampled)
    output = async_utils.AsyncOutput(
        ModelRunnerOutput(req_ids=["old", "live"], req_id_to_index={"old": 0, "live": 1},
                          sampled_token_ids=[], logprobs=None, prompt_logprobs_dict={}),
        SamplerOutput(sampled, None, None, None), torch.tensor([4, 2]),
        None, copy_stream, num_verified_draft_tokens=counts,
    )
    snapshot_ref = weakref.ref(output.num_verified_draft_tokens)
    del sampled
    # A successor verifier reuses its working buffer before the old reply lands.
    counts[:] = torch.tensor([0, 16])
    assert pending and sampled_ref() is not None
    result = output.get_output()
    assert result.req_ids == ["old", "live"]
    assert result.sampled_token_ids == [[101, 102, 103, 104], [201, 202]]
    assert result.num_verified_draft_tokens == [3, 1]
    assert not pending
    del output
    gc.collect()
    assert sampled_ref() is None and snapshot_ref() is None


@pytest.mark.parametrize("steps", [1, 5, 16])
@pytest.mark.parametrize("planned", [False, True])
@pytest.mark.parametrize("reuses", [1, 3])
def test_old_confidence_dma_lands_after_slot_reuse(monkeypatch, steps, planned, reuses):
    manager = av.AdaptiveVerificationManager.__new__(av.AdaptiveVerificationManager)
    manager.req_states = NS(device=torch.device("cpu"))
    manager._confidence_buffer = torch.ones((1 + planned, 2, steps))
    manager._confidence_probs = manager._confidence_buffer[0]
    manager._planned_probs = manager._confidence_buffer[-1]
    manager._pending_resets = []
    manager._new_rows = []
    manager._stale_idx = 0
    manager._copy_stream = NS(wait_stream=lambda _: None)
    delivered = []

    class DelayedCopy:
        def __init__(self):
            self.np = np.ones((2, steps), dtype=np.float32)
            self.gpu = torch.empty((2, steps))
            self.pending = None

        def copy_to_cpu(self):
            assert self.pending is None, "buffer reused before its old copy retired"
            self.pending = self.gpu.numpy().copy()

        def synchronize(self):
            if self.pending is not None:
                self.np[:] = self.pending
                delivered.append(self.pending.copy())
                self.pending = None

        def record(self):
            assert self.pending is not None

    copies = [DelayedCopy(), DelayedCopy()]
    manager._stale_confidences = copies
    manager._copy_events = copies
    monkeypatch.setattr(av, "gpu_sync_allowed", nullcontext)
    monkeypatch.setattr(av, "stream", lambda *_: nullcontext())
    monkeypatch.setattr(av.torch.cuda, "current_stream", lambda *_: None)
    monkeypatch.setattr(av, "get_tp_group", lambda: NS(world_size=1))
    monkeypatch.setattr(av, "async_tensor_h2d", lambda rows, **_: torch.tensor(rows))
    monkeypatch.setattr(av.tilelang_inputs, "fill_rows",
                        lambda rows, state, value: state.__setitem__(rows, value))
    monkeypatch.setattr(av.tilelang_inputs, "scatter_rows",
                        lambda rows, source, state: state.__setitem__(rows, source))
    batch = NS(idx_mapping=torch.tensor([0, 1]), has_prefill=False)
    old = torch.full((2, steps), 0.01)
    old[1] = 0.8
    manager.record_confidences(old, batch, planned=old if planned else None)
    assert not delivered and copies[0].pending is not None

    # Slot 0 is reassigned while its prior owner's transfer is still pending.
    # Repeated reassignment must neither lose the reset nor reset live slot 1.
    for _ in range(reuses):
        manager.add_request(0)
        manager.apply_staged_writes()
    assert (manager._confidence_buffer[:, 0] == 1).all()
    assert (copies[manager._stale_idx].np[0] == 1).all()
    new = torch.full((2, steps), 0.7)
    new[1] = 0.8
    manager.record_confidences(new, batch, planned=new if planned else None)
    assert len(delivered) == 1 and (delivered[0][0] == np.float32(0.01)).all()
    assert copies[0].pending is None
    assert manager._pending_resets == []
    assert (copies[manager._stale_idx].np[0] == 1).all()
    assert (copies[manager._stale_idx].np[1] == np.float32(0.8)).all()
    manager.record_confidences(new, batch, planned=new if planned else None)
    np.testing.assert_array_equal(copies[manager._stale_idx].np, new.numpy())
    for copy in copies:
        copy.synchronize()
    assert all(copy.pending is None for copy in copies)


@pytest.mark.parametrize("freed", [(0,), (0, 1), ()])
@pytest.mark.parametrize("reuses", [1, 3])
def test_pp_completion_filters_reassigned_slots_and_releases_owners(
    monkeypatch, freed, reuses
):
    handler = pp_utils.PPHandler.__new__(pp_utils.PPHandler)
    handler.req_idx_gen_np = np.zeros(2, dtype=np.int32)
    handler.device = torch.device("cpu")
    waits = []
    event = object()
    handler.main_stream = NS(wait_event=lambda e: waits.append(e))
    monkeypatch.setattr(pp_utils, "async_tensor_h2d",
                        lambda data, **_: torch.tensor(data))
    tokens = torch.tensor([[101, 102], [201, 202]])
    drafts = torch.tensor([[111], [211]])
    token_ref, draft_ref = weakref.ref(tokens), weakref.ref(drafts)
    handler.queue = deque([pp_utils.PendingRecv(
        event, tokens, torch.tensor([2, 2]), torch.tensor([0, 0]),
        torch.tensor([0, 1]), np.array([0, 1]), np.array([True, True]),
        handler.req_idx_gen_np.copy(), drafts,
    )])
    del tokens, drafts
    for _ in range(reuses):
        for slot in freed:
            handler.on_req_idx_freed(slot)
    new_owner_drafts = torch.tensor([[900], [901]])
    result = handler.get_prev_sampled_outputs(new_owner_drafts)
    if len(freed) == 2:
        assert result is None and waits == []
    else:
        assert waits == [event]
        assert result["idx_mapping"].tolist() == [-1 if i in freed else i for i in range(2)]
        assert result["sampled_tokens"].tolist() == [[101, 102], [201, 202]]
    assert new_owner_drafts.tolist() == [
        [900 + i if i in freed else 111 + 100 * i] for i in range(2)
    ]
    del result
    gc.collect()
    assert token_ref() is None and draft_ref() is None
    assert list(handler.queue) == [None]


@pytest.mark.parametrize("prompt_tokens", [8, 65])
@pytest.mark.parametrize("spec", [0, 3])
def test_cancelled_work_delivered_after_replacement_cannot_mutate_new_request(
    prompt_tokens, spec
):
    scheduler = create_scheduler(
        async_scheduling=True, use_v2_model_runner=True,
        num_blocks=32, block_size=16, max_num_seqs=1,
        max_num_batched_tokens=32, max_model_len=256,
        num_speculative_tokens=spec or None,
        speculative_method="ngram_gpu" if spec else None,
    )
    old, new = create_requests(
        num_requests=2, num_tokens=prompt_tokens, max_tokens=20,
        ignore_eos=True, req_ids=["same-external-old", "same-external-new"],
    )
    old.external_req_id = new.external_req_id = "same-external"
    scheduler.add_request(old)
    queued = [scheduler.schedule(), scheduler.schedule()]
    assert all(old.request_id in step.num_scheduled_tokens for step in queued)
    scheduler.finish_requests([old.request_id], RequestStatus.FINISHED_ABORTED)
    scheduler.add_request(new)
    new_step = scheduler.schedule()
    assert new.request_id in new_step.num_scheduled_tokens
    before = (new.num_in_flight_tokens, new.num_computed_tokens,
              new.num_output_placeholders, new.copy_draft_alloc,
              tuple(new.output_token_ids))
    free_before = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    for step in queued:
        result = scheduler.update_from_output(step, ModelRunnerOutput(
            req_ids=[old.request_id], req_id_to_index={old.request_id: 0},
            sampled_token_ids=[[777, 778, 779, 780]], logprobs=None,
            prompt_logprobs_dict={}, copy_draft_alloc=[16],
            num_verified_draft_tokens=[3], num_nans_in_logits={old.request_id: 7},
        ))
        assert not any(out.outputs for out in result.values())
        assert (new.num_in_flight_tokens, new.num_computed_tokens,
                new.num_output_placeholders, new.copy_draft_alloc,
                tuple(new.output_token_ids)) == before
        assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == free_before
    scheduler.finish_requests([new.request_id], RequestStatus.FINISHED_ABORTED)
    assert not scheduler.requests and not scheduler.running
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == 31
