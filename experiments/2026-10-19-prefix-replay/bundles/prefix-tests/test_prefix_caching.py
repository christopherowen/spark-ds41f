# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare the with and without prefix caching."""



import copy


from collections.abc import Callable


from dataclasses import replace


from math import lcm


from types import SimpleNamespace


from unittest.mock import MagicMock


import pytest


import torch


import vllm.v1.core.kv_cache_manager as kv_cache_manager


import vllm.v1.core.kv_cache_utils as kv_cache_utils


from vllm.distributed.kv_events import (
    MEDIUM_CPU,
    MEDIUM_GPU,
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
)


from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.connector import (
    HiSparseConnector,
    HiSparseConnectorScheduler,
    HiSparseConnectorWorkerMetadata,
)


from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector


from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)


from vllm.lora.request import LoRARequest


from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalKwargsItem,
    PlaceholderRange,
)


from vllm.sampling_params import SamplingParams


from vllm.utils.hashing import sha256, sha256_cbor


from vllm.v1.core.block_pool import BlockHashToBlockMap, BlockPool


from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager, Request


from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashWithGroupId,
    KVCacheBlock,
    get_block_hash,
    get_group_id,
    get_request_block_hasher,
    hash_block_tokens,
    init_none_hash,
    make_block_hash_with_group_id,
)


from vllm.v1.core.sched.output import SchedulerOutput


from vllm.v1.core.sched.scheduler import Scheduler


from vllm.v1.hisparse.coordinator import get_hisparse_coordinator


from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    HiSparseHotSpec,
    HiSparseResidentSpec,
    KVCacheConfig,
    KVCacheGroupRole,
    KVCacheGroupSpec,
    KVCacheSpecKind,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
)


from vllm.v1.outputs import KVConnectorOutput


from vllm.v1.request import RequestStatus


pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _auto_init_hash_fn(request):
    hash_fn: Callable
    if "hash_fn" in request.fixturenames:
        hash_fn = request.getfixturevalue("hash_fn")
    else:
        hash_fn = sha256
    init_none_hash(hash_fn)


def make_request(
    request_id: str,
    prompt_token_ids: list[int],
    block_size: int,
    hash_fn: Callable,
    mm_positions: list[PlaceholderRange] | None = None,
    mm_hashes: list[str] | None = None,
    prompt_logprobs: int | None = None,
    cache_salt: str | None = None,
    lora_request: LoRARequest | None = None,
    session_id: str | None = None,
):
    mm_features = []
    if mm_positions is not None:
        for j, position in enumerate(mm_positions):
            identifier = mm_hashes[j] if mm_hashes else f"hash_{j}"
            mm_feature = MultiModalFeatureSpec(
                data=MultiModalKwargsItem.dummy(),
                mm_position=position,
                identifier=identifier,
                modality="image",
            )
            mm_features.append(mm_feature)

    sampling_params = SamplingParams(max_tokens=17, prompt_logprobs=prompt_logprobs)
    sampling_params.update_from_generation_config({}, eos_token_id=100)

    return Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        mm_features=mm_features if mm_features else None,
        sampling_params=sampling_params,
        pooling_params=None,
        lora_request=lora_request,
        cache_salt=cache_salt,
        block_hasher=get_request_block_hasher(block_size, hash_fn),
        session_id=session_id,
    )


def make_kv_cache_manager(kv_cache_config: KVCacheConfig, **kwargs) -> KVCacheManager:
    """Build a ``KVCacheManager``, deriving ``scheduler_block_size`` from the
    config (LCM of group block sizes) unless explicitly provided. This mirrors
    ``resolve_kv_cache_block_sizes`` for the non-context-parallel case used by
    these tests, so callers don't have to pass it at every site."""
    kwargs.setdefault(
        "scheduler_block_size",
        lcm(*(g.kv_cache_spec.block_size for g in kv_cache_config.kv_cache_groups)),
    )
    if "retention_interval" in kwargs:
        kv_cache_config = replace(
            kv_cache_config,
            prefix_cache_retention_interval=kwargs.pop("retention_interval"),
        )
    manager = KVCacheManager(kv_cache_config, **kwargs)
    if kv_cache_config.hisparse_host_num_blocks is not None:
        get_hisparse_coordinator(manager)
    return manager


def make_ced_kv_cache_config(*, ced: bool = True) -> KVCacheConfig:
    global_spec = MLAAttentionSpec(
        block_size=128, num_kv_heads=1, head_size=1, dtype=torch.float32
    )
    encoder_spec = SlidingWindowMLASpec(
        block_size=32,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=128,
    )
    decoder_spec = (
        replace(encoder_spec, prefix_cache_enabled=False, prefill_replay_window=128)
        if ced
        else encoder_spec
    )
    return KVCacheConfig(
        num_blocks=1024,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["global"], global_spec),
            KVCacheGroupSpec(["encoder"], encoder_spec),
            KVCacheGroupSpec(["decoder"], decoder_spec),
            KVCacheGroupSpec(["draft"], replace(decoder_spec, extra_retained_tokens=1)),
        ],
        prefix_cache_retention_interval=0,
    )


def test_ced_private_swa_never_publishes_unwritten_prefix():
    manager = make_kv_cache_manager(
        make_ced_kv_cache_config(), max_model_len=8192, hash_block_size=32
    )
    tokens = list(range(4160))
    producer = make_request("producer", tokens, 32, sha256)
    allocated = manager.allocate_slots(producer, len(tokens))
    assert allocated is not None

    # Allocation alone previously published decoder pages even though CED only
    # writes the final 128 rows. A repeated prompt needs encoder state at 3968,
    # before the decoder's materialized suffix [4032, 4160).
    pool = manager.block_pool
    for block_hash in producer.block_hashes:
        for group_id in (2, 3):
            assert pool.get_cached_block(block_hash, [group_id]) is None
    for group_id in (2, 3):
        assert all(block.block_hash is None for block in allocated.blocks[group_id])

    consumer = make_request("consumer", tokens, 32, sha256)
    hits, hit_tokens, _ = manager.get_computed_blocks(consumer)
    assert hit_tokens == 3968
    assert len(hits.blocks[0]) == 31
    assert [i for i, block in enumerate(hits.blocks[1]) if not block.is_null] == list(
        range(120, 124)
    )
    assert not hits.blocks[2] and not hits.blocks[3]

    producer_private_ids = {
        block.block_id for group_id in (2, 3) for block in allocated.blocks[group_id]
    }
    free_before = pool.get_num_free_blocks()
    fresh = manager.allocate_slots(consumer, len(tokens) - hit_tokens, hit_tokens, hits)
    assert fresh is not None
    assert free_before - pool.get_num_free_blocks() == sum(
        len(blocks) for blocks in fresh.blocks
    )
    for group_id in (2, 3):
        blocks = manager.coordinator.single_type_managers[group_id].req_to_blocks[
            consumer.request_id
        ]
        assert all(not block.is_null for block in blocks[hit_tokens // 32 :])
        assert producer_private_ids.isdisjoint(
            block.block_id for block in blocks if not block.is_null
        )
        assert all(block.block_hash is None for block in blocks if not block.is_null)

    # Same-request short continuation must retain its written decoder tail,
    # even though that tail is never available for cross-request reuse.
    consumer.num_computed_tokens = len(tokens)
    consumer.append_output_token_ids(7)
    previous_tail = [
        manager.coordinator.single_type_managers[gid].req_to_blocks[
            consumer.request_id
        ][-1]
        for gid in (2, 3)
    ]
    assert manager.allocate_slots(consumer, 1) is not None
    for group_id, tail in zip((2, 3), previous_tail):
        blocks = manager.coordinator.single_type_managers[group_id].req_to_blocks[
            consumer.request_id
        ]
        assert blocks[len(tokens) // 32 - 1] is tail
        assert tail.ref_cnt == 1
        assert tail.block_hash is None


@pytest.mark.parametrize(
    ("prompt_length", "expected_hit"), [(96, 0), (128, 0), (256, 128)]
)
def test_ced_prefix_replays_short_prompt_boundary(prompt_length, expected_hit):
    manager = make_kv_cache_manager(
        make_ced_kv_cache_config(), max_model_len=8192, hash_block_size=32
    )
    tokens = list(range(prompt_length))
    producer = make_request("producer", tokens, 32, sha256)
    assert manager.allocate_slots(producer, len(tokens)) is not None
    manager.free(producer)
    consumer = make_request("consumer", tokens, 32, sha256)
    hits, hit_tokens, _ = manager.get_computed_blocks(consumer)
    assert hit_tokens == expected_hit
    assert not hits.blocks[2] and not hits.blocks[3]
    assert (
        manager.allocate_slots(consumer, prompt_length - hit_tokens, hit_tokens, hits)
        is not None
    )

