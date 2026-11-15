# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pickle

import numpy as np
import pytest
import torch

from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import (
    INT32_TOKEN_IDS_MIN_LEN,
    Int32TokenIds,
    NewRequestData,
)
from vllm.v1.request import Request


def _make_request(prompt_tokens: list[int], req_id: str = "req-1") -> Request:
    return Request(
        request_id=req_id,
        prompt_token_ids=prompt_tokens,
        sampling_params=SamplingParams(max_tokens=16),
        pooling_params=None,
    )


def test_int32_token_ids_sequence_protocol():
    raw = [i * 3 - 7 for i in range(512)]
    packed = Int32TokenIds.from_sequence(raw)
    assert packed is not None
    assert len(packed) == 512
    assert bool(packed) is True
    assert not bool(Int32TokenIds(np.empty(0, dtype=np.int32)))
    assert isinstance(packed[0], int)
    assert packed[0] == raw[0]
    assert packed[-1] == raw[-1]
    sliced = packed[10:50]
    assert isinstance(sliced, Int32TokenIds)
    assert sliced == raw[10:50]
    assert packed == raw
    assert packed.tolist() == raw
    assert list(packed) == raw
    copied = packed.copy()
    assert copied == packed
    assert copied._arr is not packed._arr
    arr_view = np.asarray(packed, dtype=np.int32)
    assert np. shares_memory(arr_view, packed._arr)


def test_new_request_data_from_request_fresh_and_resumed():
    tokens = [i % 100000 for i in range(4096)]
    req = _make_request(tokens)

    # Fresh request: prompt_token_ids and prefill_token_ids share one Int32TokenIds
    data = NewRequestData.from_request(
        req, block_ids=([1, 2, 3],), prefill_token_ids=req._all_token_ids
    )
    assert isinstance(data.prompt_token_ids, Int32TokenIds)
    assert data.prefill_token_ids is data.prompt_token_ids
    assert data.prompt_len == 4096
    assert data.prompt_token_ids == tokens

    # Single PickleBuffer emitted for both fields combined
    buffers = []
    blob = pickle.dumps(
        data, protocol=pickle.HIGHEST_PROTOCOL, buffer_callback=buffers.append
    )
    assert len(buffers) == 1
    raw_views = [b.raw() for b in buffers]
    restored = pickle.loads(blob, buffers=raw_views)
    assert isinstance(restored.prompt_token_ids, Int32TokenIds)
    assert restored.prefill_token_ids is restored.prompt_token_ids
    assert restored.prompt_token_ids == tokens

    # Resumed request with output tokens appended
    req.append_output_token_ids([99991, 99992, 99993])
    resumed = NewRequestData.from_request(
        req, block_ids=([1, 2, 3, 4],), prefill_token_ids=req._all_token_ids
    )
    assert resumed.prompt_len == 4096
    assert len(resumed.prefill_token_ids) == 4099
    assert resumed.prompt_token_ids == tokens
    assert resumed.prefill_token_ids == tokens + [99991, 99992, 99993]

    buffers_resumed = []
    blob_resumed = pickle.dumps(
        resumed,
        protocol=pickle.HIGHEST_PROTOCOL,
        buffer_callback=buffers_resumed.append,
    )
    assert len(buffers_resumed) == 1
    restored_resumed = pickle.loads(
        blob_resumed, buffers=[b.raw() for b in buffers_resumed]
    )
    assert restored_resumed.prompt_len == 4096
    assert restored_resumed.prompt_token_ids == tokens
    assert restored_resumed.prefill_token_ids == tokens + [99991, 99992, 99993]


def test_new_request_data_short_and_overflow_fallback():
    short_tokens = list(range(32))
    assert len(short_tokens) < INT32_TOKEN_IDS_MIN_LEN
    req_short = _make_request(short_tokens, req_id="short")
    data_short = NewRequestData.from_request(
        req_short, block_ids=([1],), prefill_token_ids=req_short._all_token_ids
    )
    assert isinstance(data_short.prompt_token_ids, list)
    assert data_short.prefill_token_ids is data_short.prompt_token_ids

    huge_tokens = [1 << 40] * 512
    req_huge = _make_request(huge_tokens, req_id="huge")
    data_huge = NewRequestData.from_request(
        req_huge, block_ids=([1],), prefill_token_ids=req_huge._all_token_ids
    )
    assert isinstance(data_huge.prompt_token_ids, list)
    restored_huge = pickle.loads(pickle.dumps(data_huge, protocol=5))
    assert restored_huge.prompt_token_ids == huge_tokens


@pytest.mark.parametrize("num_tokens", [4096, 32768, 300000])
def test_message_queue_new_request_data_roundtrip(num_tokens: int):
    tokens = [(i * 17 + 5) % 128000 for i in range(num_tokens)]
    req = _make_request(tokens, req_id=f"mq-{num_tokens}")
    data = NewRequestData.from_request(
        req,
        block_ids=(list(range(num_tokens // 64)),),
        prefill_token_ids=req._all_token_ids,
    )

    mq_writer = MessageQueue(
        n_reader=1,
        n_local_reader=1,
        local_reader_ranks=[0],
        max_chunk_bytes=2 * 1024 * 1024,
        max_chunks=2,
    )
    mq_reader = MessageQueue.create_from_handle(mq_writer.export_handle(), rank=0)
    try:
        mq_writer.enqueue(data, timeout=5.0)
        received = mq_reader.dequeue(timeout=5.0)
        assert isinstance(received.prompt_token_ids, Int32TokenIds)
        assert received.prefill_token_ids is received.prompt_token_ids
        assert received.prompt_len == num_tokens
        assert np.array_equal(
            np.asarray(received.prefill_token_ids, dtype=np.int32),
            np.asarray(tokens, dtype=np.int32),
        )
    finally:
        mq_writer.shutdown()
        mq_reader.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA for UVA")
def test_staged_write_tensor_with_int32_token_ids():
    from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor

    device = torch.device("cuda:0")
    tensor = StagedWriteTensor(
        (4, 8192), dtype=torch.int32, device=device, uva_instead_of_gpu=True
    )
    toks0 = Int32TokenIds(np.arange(100, 4196, dtype=np.int32))
    toks1 = Int32TokenIds(np.arange(5000, 7048, dtype=np.int32))

    tensor.stage_write(0, 0, toks0)
    tensor.stage_write(2, 10, toks1)
    tensor.apply_write()
    torch.cuda.synchronize()

    row0 = tensor.gpu[0, : len(toks0)].cpu().numpy()
    row2 = tensor.gpu[2, 10 : 10 + len(toks1)].cpu().numpy()
    assert np.array_equal(row0, toks0._arr)
    assert np.array_equal(row2, toks1._arr)

    # Also verify mixed array + list + elem staging fallback
    tensor.stage_write(1, 0, toks0[:128])
    tensor.stage_write(1, 128, [77, 88, 99])
    tensor.stage_write_elem(3, 42)
    tensor.apply_write()
    torch.cuda.synchronize()

    assert np.array_equal(tensor.gpu[1, :128].cpu().numpy(), toks0._arr[:128])
    assert tensor.gpu[1, 128:131].cpu().tolist() == [77, 88, 99]
    assert int(tensor.gpu[3, 0].cpu().item()) == 42
