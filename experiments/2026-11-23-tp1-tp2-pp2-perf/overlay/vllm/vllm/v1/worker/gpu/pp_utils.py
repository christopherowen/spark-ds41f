# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline Parallelism utils for V2 Model Runner."""

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from vllm.distributed.parallel_state import get_pp_group
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.worker.gpu.input_batch import InputBatch


@dataclass
class PendingRecv:
    """Per-step slot data for a deferred postprocess on the main stream."""

    event: torch.cuda.Event

    sampled_tokens: torch.Tensor  # [num_reqs, max_sample_len]
    num_sampled: torch.Tensor  # [num_reqs]
    num_rejected: torch.Tensor  # [num_reqs]
    idx_mapping: torch.Tensor  # [num_reqs]
    idx_mapping_np: np.ndarray  # [num_reqs]
    # Records which rows need a deferred postprocess (bool).
    need_sampled_mask: np.ndarray  # [num_reqs]
    # Snapshot of slot generation counters at receive time, used to
    # detect requests aborted since then.
    gen_at_receive_np: np.ndarray  # [num_reqs]
    draft_tokens: torch.Tensor | None = None  # [num_reqs, num_speculative_steps]
    confidences: torch.Tensor | None = None  # [planes, num_reqs, num_speculative_steps]
    is_calibrated: torch.Tensor | None = None  # [1] int32


@dataclass
class PendingConf:
    """Last-rank mirror of PendingRecv for lockstep PP adaptive verification."""

    event: torch.cuda.Event
    confidences: torch.Tensor  # [planes, num_reqs, num_speculative_steps]
    is_calibrated: torch.Tensor  # [1] int32
    idx_mapping: torch.Tensor  # [num_reqs]
    idx_mapping_np: np.ndarray  # [num_reqs]
    need_sampled_mask: np.ndarray  # [num_reqs]
    gen_at_receive_np: np.ndarray  # [num_reqs]


def compute_need_sampled_mask(input_batch: InputBatch) -> np.ndarray | None:
    """Return a bool array of shape `[input_batch.num_reqs]` marking requests
    that produce a sampled token this step, and therefore must have that token
    (and the draft block proposed from it) propagated to the earlier PP stages.
    Returns None if no request in the batch produces a sample."""

    old_computed = input_batch.num_computed_tokens_np
    prefill_len = input_batch.prefill_len_np
    # Exclude non-final prefill chunks (they don't produce a sample).
    produces_sample = old_computed + input_batch.num_scheduled_tokens >= prefill_len
    return produces_sample if produces_sample.any() else None


class PPHandler:
    """Runs the PP sampled-token broadcast/recv on a side stream so the
    default stream isn't gated by the matching peer call. Step T's recv is
    consumed at step T+pp_size via `get_prev_sampled_outputs`.

    Uses a dedicated NCCL communicator (sibling of the PP `device_group`)
    for the broadcast so it does not serialize on the wire with the
    inter-stage hidden-state p2p send/recv ops.
    """

    def __init__(
        self, max_num_reqs: int, num_speculative_steps: int, device: torch.device
    ):
        self.is_last_rank = get_pp_group().is_last_rank
        self.last_rank = get_pp_group().last_rank
        self.max_sample_len = num_speculative_steps + 1
        self.num_speculative_steps = num_speculative_steps
        self.device = device
        self.main_stream = torch.cuda.current_stream(device)
        self.broadcast_stream = torch.cuda.Stream(device)

        # On non-last ranks, a FIFO with one entry per in-flight step: the entry
        # pushed by step T's `receive` is consumed pp_size steps later. Pre-seeded
        # with pp_size None placeholders so the first pp_size consumes are no-ops.
        # None means no postprocess is pending for that step (broadcast skipped).
        self.queue: deque[PendingRecv | None] = (
            deque() if self.is_last_rank else deque([None] * get_pp_group().world_size)
        )

        # Per req-index generation counter, incremented every time a request
        # index is freed in RequestStats. Used for invalidating freed req data
        # between PP decodes.
        self.req_idx_gen_np = np.zeros(max_num_reqs, dtype=np.int32)

        # Dedicated subgroup for the sampled-token broadcast.
        self.broadcast_group = get_pp_group().make_sibling_device_group(
            group_desc="pp_broadcast"
        )
        self.aux_hidden_state_relay_keys: tuple[str, ...] = ()
        self.broadcast_confidences: bool = False
        self.confidence_planes: int = 1
        self.local_conf_queue: deque[PendingConf | None] = deque()

    def configure_confidence_broadcast(
        self, enabled: bool, *, planes: int = 1
    ) -> None:
        self.broadcast_confidences = bool(enabled and self.num_speculative_steps > 0)
        self.confidence_planes = max(1, int(planes))
        if self.is_last_rank and self.broadcast_confidences:
            self.local_conf_queue = deque([None] * get_pp_group().world_size)
        else:
            self.local_conf_queue = deque()

    def on_req_idx_freed(self, req_idx: int) -> None:
        self.req_idx_gen_np[req_idx] += 1

    def configure_aux_hidden_state_relay(self, model: torch.nn.Module) -> None:
        from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
            aux_hidden_state_relay_keys,
        )

        self.aux_hidden_state_relay_keys = aux_hidden_state_relay_keys(model)

    def relay_aux_hidden_states(
        self,
        intermediate_tensors: IntermediateTensors | None,
        output_intermediate_tensors: IntermediateTensors,
    ) -> IntermediateTensors:
        if not self.aux_hidden_state_relay_keys:
            return output_intermediate_tensors
        assert intermediate_tensors is not None
        return IntermediateTensors(
            output_intermediate_tensors.tensors
            | {
                key: intermediate_tensors[key]
                for key in self.aux_hidden_state_relay_keys
            }
        )

    def get_prev_sampled_outputs(
        self,
        draft_tokens_to_update: torch.Tensor | None = None,
        *,
        on_confidences: Callable[
            [torch.Tensor, np.ndarray, torch.Tensor, torch.Tensor], None
        ]
        | None = None,
    ) -> dict[str, torch.Tensor] | None:
        """Consume the entry from pp_size steps ago and wait for its recv event,
        then filter out entries whose request was freed since `receive`.
        """
        if not self.queue:
            return None
        slot = self.queue.popleft()
        # Reserve this step's slot; `receive` overwrites it if applicable.
        self.queue.append(None)
        if slot is None:
            return None

        # Skip requests which did not need sampled output and/or those already
        # finished. The post_update kernel skips the -1 entries.
        freed = self.req_idx_gen_np[slot.idx_mapping_np] != slot.gen_at_receive_np
        exclude_mask = freed | ~slot.need_sampled_mask
        idx_mapping = slot.idx_mapping
        if exclude_mask.any():
            if exclude_mask.all():
                # No states require update anymore.
                return None
            # Filter excluded request indices.
            idx_mapping_np = np.where(exclude_mask, -1, slot.idx_mapping_np)
            idx_mapping = async_tensor_h2d(idx_mapping_np, device=self.device)

        self.main_stream.wait_event(slot.event)
        keep_idx_mapping = slot.idx_mapping
        keep_idx_mapping_np = slot.idx_mapping_np
        keep_t = None
        if exclude_mask.any():
            keep = ~exclude_mask
            keep_t = torch.as_tensor(keep, device=self.device)
            keep_idx_mapping_np = slot.idx_mapping_np[keep]
            keep_idx_mapping = async_tensor_h2d(
                keep_idx_mapping_np, device=self.device
            )
        if slot.draft_tokens is not None and draft_tokens_to_update is not None:
            draft_tokens = (
                slot.draft_tokens[keep_t] if keep_t is not None else slot.draft_tokens
            )
            draft_tokens_to_update[keep_idx_mapping] = draft_tokens
        if (
            slot.confidences is not None
            and slot.is_calibrated is not None
            and on_confidences is not None
        ):
            conf_slice = (
                slot.confidences[:, keep_t, :]
                if keep_t is not None
                else slot.confidences
            )
            on_confidences(
                keep_idx_mapping, keep_idx_mapping_np, conf_slice, slot.is_calibrated
            )

        return dict(
            sampled_tokens=slot.sampled_tokens,
            num_sampled=slot.num_sampled,
            num_rejected=slot.num_rejected,
            idx_mapping=idx_mapping,
        )

    def consume_local_confidences(
        self,
        on_confidences: Callable[
            [torch.Tensor, np.ndarray, torch.Tensor, torch.Tensor], None
        ]
        | None,
    ) -> None:
        """Last-rank mirror of `get_prev_sampled_outputs` for lockstep PP confidences."""
        if not self.local_conf_queue:
            return
        slot = self.local_conf_queue.popleft()
        self.local_conf_queue.append(None)
        if slot is None or on_confidences is None:
            return
        freed = self.req_idx_gen_np[slot.idx_mapping_np] != slot.gen_at_receive_np
        exclude_mask = freed | ~slot.need_sampled_mask
        if exclude_mask.all():
            return
        self.main_stream.wait_event(slot.event)
        if exclude_mask.any():
            keep = ~exclude_mask
            keep_t = torch.as_tensor(keep, device=self.device)
            keep_idx_np = slot.idx_mapping_np[keep]
            keep_idx = async_tensor_h2d(keep_idx_np, device=self.device)
            conf_slice = slot.confidences[:, keep_t, :]
        else:
            keep_idx = slot.idx_mapping
            keep_idx_np = slot.idx_mapping_np
            conf_slice = slot.confidences
        on_confidences(keep_idx, keep_idx_np, conf_slice, slot.is_calibrated)

    def broadcast_drafts(
        self,
        draft_tokens: torch.Tensor,
        input_batch: InputBatch,
        *,
        confidences: torch.Tensor | None = None,
        is_calibrated: bool = False,
    ) -> None:
        """Broadcast draft proposals (and optional adaptive verification confidences)
        so all PP ranks share identical tokens and trimmed batch shapes."""
        assert self.is_last_rank
        need_sampled_mask = compute_need_sampled_mask(input_batch)
        if need_sampled_mask is None:
            return
        gen_at_send_np = self.req_idx_gen_np[input_batch.idx_mapping_np]
        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            send = draft_tokens[input_batch.idx_mapping].contiguous()
            # Must record the idx_mapping tensor since it was allocated
            # on the main stream.
            input_batch.idx_mapping.record_stream(self.broadcast_stream)
            torch.distributed.broadcast(
                send, src=self.last_rank, group=self.broadcast_group
            )
            if self.broadcast_confidences:
                if confidences is None:
                    send_conf = torch.ones(
                        (
                            self.confidence_planes,
                            input_batch.num_reqs,
                            self.num_speculative_steps,
                        ),
                        dtype=torch.float32,
                        device=self.device,
                    )
                else:
                    send_conf = confidences.contiguous()
                    confidences.record_stream(self.broadcast_stream)
                send_cal = torch.tensor(
                    [1 if is_calibrated else 0],
                    dtype=torch.int32,
                    device=self.device,
                )
                torch.distributed.broadcast(
                    send_conf, src=self.last_rank, group=self.broadcast_group
                )
                torch.distributed.broadcast(
                    send_cal, src=self.last_rank, group=self.broadcast_group
                )
                ev = self.broadcast_stream.record_event()
                send_conf.record_stream(self.main_stream)
                send_cal.record_stream(self.main_stream)
                if self.local_conf_queue:
                    self.local_conf_queue[-1] = PendingConf(
                        ev,
                        send_conf,
                        send_cal,
                        input_batch.idx_mapping,
                        input_batch.idx_mapping_np,
                        need_sampled_mask,
                        gen_at_send_np,
                    )

    def receive(self, input_batch: InputBatch) -> bool:
        """Returns True iff sampled tokens need to be gathered from *all*
        requests in the batch."""
        assert not self.is_last_rank
        need_sampled_mask = compute_need_sampled_mask(input_batch)
        if need_sampled_mask is None:
            # Leave this step's reserved slot as None.
            return False

        # Snapshot the per-slot generation counter so a later free of any of
        # these RequestStates request indices is detectable at consume time.
        gen_at_receive_np = self.req_idx_gen_np[input_batch.idx_mapping_np]

        num_reqs = input_batch.num_reqs
        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            sampled_tokens = torch.empty(
                num_reqs, self.max_sample_len, dtype=torch.int64, device=self.device
            )
            combined = torch.empty(2, num_reqs, dtype=torch.int32, device=self.device)
            torch.distributed.broadcast(
                sampled_tokens, src=self.last_rank, group=self.broadcast_group
            )
            torch.distributed.broadcast(
                combined, src=self.last_rank, group=self.broadcast_group
            )
            draft_tokens = None
            confidences = None
            is_calibrated = None
            if self.num_speculative_steps > 0:
                draft_tokens = torch.empty(
                    num_reqs,
                    self.num_speculative_steps,
                    dtype=torch.int64,
                    device=self.device,
                )
                torch.distributed.broadcast(
                    draft_tokens, src=self.last_rank, group=self.broadcast_group
                )
                if self.broadcast_confidences:
                    confidences = torch.empty(
                        self.confidence_planes,
                        num_reqs,
                        self.num_speculative_steps,
                        dtype=torch.float32,
                        device=self.device,
                    )
                    is_calibrated = torch.empty(
                        1, dtype=torch.int32, device=self.device
                    )
                    torch.distributed.broadcast(
                        confidences, src=self.last_rank, group=self.broadcast_group
                    )
                    torch.distributed.broadcast(
                        is_calibrated, src=self.last_rank, group=self.broadcast_group
                    )
            event = self.broadcast_stream.record_event()
            num_sampled, num_rejected = combined.unbind(dim=0)
            # Must record_stream since these were allocated on broadcast stream but
            # later used on the main stream.
            sampled_tokens.record_stream(self.main_stream)
            combined.record_stream(self.main_stream)
            if draft_tokens is not None:
                draft_tokens.record_stream(self.main_stream)
            if confidences is not None:
                confidences.record_stream(self.main_stream)
            if is_calibrated is not None:
                is_calibrated.record_stream(self.main_stream)
        self.queue[-1] = PendingRecv(
            event,
            sampled_tokens,
            num_sampled,
            num_rejected,
            input_batch.idx_mapping,
            input_batch.idx_mapping_np,
            need_sampled_mask,
            gen_at_receive_np,
            draft_tokens,
            confidences,
            is_calibrated,
        )
        return bool(need_sampled_mask.all())

    def broadcast(
        self,
        sampled_token_ids: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        input_batch: InputBatch,
    ) -> None:
        assert self.is_last_rank
        if compute_need_sampled_mask(input_batch) is None:
            # No request needs sampled outputs for a subsequent decode step.
            return

        assert sampled_token_ids.dtype == torch.int64

        if current_platform.is_xpu():
            self.main_stream.synchronize()

        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            send_tokens = torch.nn.functional.pad(
                sampled_token_ids,
                (0, self.max_sample_len - sampled_token_ids.shape[-1]),
            )
            torch.distributed.broadcast(
                send_tokens.contiguous(),
                src=self.last_rank,
                group=self.broadcast_group,
            )
            combined = torch.stack((num_sampled, num_rejected), dim=0)
            torch.distributed.broadcast(
                combined, src=self.last_rank, group=self.broadcast_group
            )
            for tensor in (sampled_token_ids, num_sampled, num_rejected):
                tensor.record_stream(self.broadcast_stream)
