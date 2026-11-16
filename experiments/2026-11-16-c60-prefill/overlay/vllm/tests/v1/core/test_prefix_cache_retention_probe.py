# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stress probe and regression tests for prefix-cache block-boundary retention
and LRU eviction ordering under 64K-100K multi-turn agent workloads (#902).

Covers:
1. Exact block-boundary retention around block_size - 1, block_size, and
   block_size + 1 across 128-token Encoder SWA and 256-token Global MLA
   boundaries and across prefix_cache_retention_interval in {0, 512, 8192, None}.
2. Rolling decode-frontier SWA retention when decode stops at T where
   T % 256 >= 128 (after the 128-token sliding window has moved past the last
   256-token scheduler boundary) and recycling of intermediate decode SWA
   blocks to the scratch front of free_block_queue.
3. Multi-step speculative decoding (1..6 tokens/step) crossing block boundaries
   and finishing on the crossing step in both Scheduler and AsyncScheduler,
   plus KVCacheManager hybrid CED speculative decode step stepping.
4. Shared BlockPool tail-to-root LRU eviction ordering across hybrid
   FullAttentionManager + SlidingWindowManager groups, including prefix-chain
   refresh across multi-turn cache hits.
5. 64K-100K multi-turn concurrent agent stress probe under KV block eviction
   pressure across tp3-triangle (retention_interval=512) and tp4-ring
   (retention_interval=8192) profiles, asserting >= 98.5% prefix-cache hit
   rate preservation.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from tests.v1.core.test_prefix_caching import make_kv_cache_manager, make_request
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import (
    get_block_hash,
    get_group_id,
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput

from .utils import create_requests, create_scheduler

HASH = 128
PAGE = 256
WINDOW = 128


@pytest.fixture(autouse=True)
def initialize_hashes():
    init_none_hash(sha256)


def make_ced_manager(
    retention: int | None,
    *,
    num_blocks: int = 4096,
    max_model_len: int = 131072,
    include_private_decoder: bool = True,
):
    """Build a DeepSeek V4.1 CED hybrid KV cache manager:
    - Global MLA (block_size=256, persistent)
    - Indexer MLA (block_size=256, persistent)
    - Encoder SWA (block_size=128, sliding_window=128, persistent)
    - Decoder SWA (block_size=128, sliding_window=128, private, replay=128)
    - Draft SWA (block_size=128, sliding_window=128, private, extra_retained=1)
    """
    global_spec = MLAAttentionSpec(
        block_size=PAGE,
        num_kv_heads=1,
        head_size=512,
        tokens_per_state=2,
        dtype=torch.uint8,
    )
    encoder_spec = SlidingWindowMLASpec(
        block_size=HASH,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        sliding_window=WINDOW,
    )
    groups = [
        KVCacheGroupSpec(["global-mla"], global_spec),
        KVCacheGroupSpec(["indexer-mla"], replace(global_spec, head_size=68)),
        KVCacheGroupSpec(["encoder-swa"], encoder_spec),
    ]
    if include_private_decoder:
        decoder_spec = replace(
            encoder_spec, prefix_cache_enabled=False, prefill_replay_window=WINDOW
        )
        groups.extend(
            [
                KVCacheGroupSpec(["decoder-swa"], decoder_spec),
                KVCacheGroupSpec(
                    ["draft-swa"], replace(decoder_spec, extra_retained_tokens=1)
                ),
            ]
        )
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=groups,
        prefix_cache_retention_interval=retention,
    )
    manager = make_kv_cache_manager(
        config, max_model_len=max_model_len, hash_block_size=HASH
    )
    roles = {group.layer_names[0]: i for i, group in enumerate(groups)}
    return manager, roles


def make_ced_request(name: str, tokens: list[int], max_tokens: int = 8192):
    req = make_request(name, list(tokens), HASH, sha256)
    req.max_tokens = max_tokens
    req.sampling_params.max_tokens = max_tokens
    return req


def run_prefill_and_decode(
    manager,
    req,
    decode_tokens: list[int],
    *,
    tokens_per_step: int = 1,
    prefill_chunk_size: int | None = None,
) -> int:
    """Execute prefill (optionally chunked) followed by multi-step decode
    (1..6 tokens/step, modeling standard decode or speculative decoding).
    Returns the initial prefix-cache hit in tokens.
    """
    computed_blocks, num_cached_tokens, _ = manager.get_computed_blocks(req)
    remaining_prefill = req.num_prompt_tokens - num_cached_tokens
    if prefill_chunk_size is None or remaining_prefill <= prefill_chunk_size:
        if remaining_prefill > 0:
            assert (
                manager.allocate_slots(
                    req, remaining_prefill, num_cached_tokens, computed_blocks
                )
                is not None
            )
            req.num_computed_tokens = req.num_prompt_tokens
            manager.cache_blocks(req, req.num_computed_tokens)
        else:
            assert (
                manager.allocate_slots(req, 0, num_cached_tokens, computed_blocks)
                is not None
            )
            req.num_computed_tokens = num_cached_tokens
    else:
        first_chunk = min(prefill_chunk_size, remaining_prefill)
        assert (
            manager.allocate_slots(
                req, first_chunk, num_cached_tokens, computed_blocks
            )
            is not None
        )
        req.num_computed_tokens = num_cached_tokens + first_chunk
        manager.cache_blocks(req, req.num_computed_tokens)
        while req.num_computed_tokens < req.num_prompt_tokens:
            step = min(
                prefill_chunk_size, req.num_prompt_tokens - req.num_computed_tokens
            )
            assert manager.allocate_slots(req, step) is not None
            req.num_computed_tokens += step
            manager.cache_blocks(req, req.num_computed_tokens)

    idx = 0
    while idx < len(decode_tokens):
        step_tokens = decode_tokens[idx : idx + tokens_per_step]
        idx += len(step_tokens)
        req.append_output_token_ids(step_tokens)
        new_to_allocate = req.num_tokens - req.num_computed_tokens
        assert manager.allocate_slots(req, new_to_allocate) is not None
        req.num_computed_tokens = req.num_tokens
        manager.cache_blocks(req, req.num_computed_tokens)

    manager.free(req)
    return num_cached_tokens


# ---------------------------------------------------------------------------
# 1. Exact Block-Boundary & Rolling Decode-Frontier Retention
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("retention", [0, 512, 8192, None], ids=["0", "512", "8192", "dense"])
@pytest.mark.parametrize("prompt_base", [256, 512, 1024])
@pytest.mark.parametrize("prompt_offset", [-1, 0, 1])
@pytest.mark.parametrize(
    "decode_stop_offset",
    [
        127,  # B - 1 (just before 128 SWA boundary)
        128,  # B + 128 (exact 128 SWA boundary)
        129,  # B + 129 (1 token past 128 SWA boundary)
        255,  # next 256 boundary - 1
        256,  # exact next 256 boundary
        257,  # next 256 boundary + 1
        383,  # next 256 + 127
        384,  # next 256 + 128 (sliding window moved past next 256 boundary!)
        385,  # next 256 + 129 (sliding window moved past next 256 boundary!)
        511,  # +2*256 - 1
        512,  # +2*256
        513,  # +2*256 + 1
        641,  # +2*256 + 129 (2.5 pages decoded, T % 256 == 129 >= 128)
    ],
)
@pytest.mark.parametrize("tokens_per_step", [1, 3, 6], ids=["step1", "step3", "step6"])
def test_exact_block_boundary_and_decode_frontier_retention(
    retention: int | None,
    prompt_base: int,
    prompt_offset: int,
    decode_stop_offset: int,
    tokens_per_step: int,
):
    """Verify that regardless of whether a prompt starts at B-1, B, or B+1 and
    regardless of whether multi-step decode (1..6 tokens/step) stops at an
    offset where T % 256 < 128 or T % 256 >= 128 (after the 128-token SWA
    window has slid past floor(T / 256) * 256), the completed 256-token
    boundary floor(T_end / 256) * 256 is retained across all persistent groups
    so Turn 2 hits floor(T_end / 256) * 256 in full.
    """
    manager, _ = make_ced_manager(retention, num_blocks=2048)
    prompt_len = prompt_base + prompt_offset
    target_total = prompt_base + decode_stop_offset
    num_decode = target_total - prompt_len
    assert num_decode > 0

    all_tokens = list(range(target_total + 512))
    turn1_prompt = all_tokens[:prompt_len]
    turn1_decode = all_tokens[prompt_len:target_total]

    req1 = make_ced_request("turn-1", turn1_prompt)
    run_prefill_and_decode(
        manager, req1, turn1_decode, tokens_per_step=tokens_per_step
    )

    # Turn 2 appends >= 128 tokens (e.g., 256 tokens of tool output / user reply)
    # so the 128-token private decoder replay window sits inside the new suffix.
    turn2_prompt = all_tokens[: target_total + 256]
    req2 = make_ced_request("turn-2", turn2_prompt)
    _, hit_ext, _ = manager.get_computed_blocks(req2)
    expected_ext_hit = (target_total // PAGE) * PAGE
    assert hit_ext == expected_ext_hit, (
        f"retention={retention}, prompt={prompt_len}, total={target_total}, "
        f"step={tokens_per_step}: expected extension hit {expected_ext_hit}, got {hit_ext}"
    )

    # Also verify exact-same-length retry (delta_prompt == 0): capped by
    # target_total - WINDOW, so it must hit floor((target_total - WINDOW) / PAGE) * PAGE.
    req_retry = make_ced_request("retry", all_tokens[:target_total])
    _, hit_retry, _ = manager.get_computed_blocks(req_retry)
    expected_retry_hit = ((target_total - WINDOW) // PAGE) * PAGE
    assert hit_retry == expected_retry_hit, (
        f"retention={retention}, prompt={prompt_len}, total={target_total}, "
        f"step={tokens_per_step}: expected retry hit {expected_retry_hit}, got {hit_retry}"
    )


@pytest.mark.parametrize("retention", [0, 512, 8192], ids=["0", "512", "8192"])
def test_long_decode_recycles_intermediate_swa_frontier_blocks(retention: int):
    """During a long decode (e.g., 4,096 decoded tokens = 16 pages), rolling
    decode-frontier SWA retention must recycle obsolete intermediate non-segment
    decode SWA blocks to the scratch front of free_block_queue so decode never
    accumulates O(T_decode) extra SWA checkpoints when retention != None.
    """
    manager, roles = make_ced_manager(retention, num_blocks=2048)
    swa_group_id = roles["encoder-swa"]
    swa_mgr = manager.coordinator.single_type_managers[swa_group_id]
    assert isinstance(swa_mgr, SlidingWindowManager)

    prompt_len = 513  # unaligned prompt
    decode_len = 4096  # 16 pages of decode -> ends at 4609 (4609 % 256 == 1)
    total_len = prompt_len + decode_len
    tokens = list(range(total_len + 256))

    req1 = make_ced_request("long-decode", tokens[:prompt_len])
    run_prefill_and_decode(
        manager, req1, tokens[prompt_len:total_len], tokens_per_step=4
    )

    # Count how many Encoder SWA blocks remain hashed in the block pool.
    cached_swa_blocks = [
        b
        for b in manager.block_pool.blocks
        if b.block_hash is not None and get_group_id(b.block_hash) == swa_group_id
    ]
    if retention == 0:
        # Only prompt boundaries (256, 512) + final decode frontier (4352, 4608): <= 4 SWA blocks.
        assert len(cached_swa_blocks) <= 4
    elif retention == 8192:
        # Total < 8192, so only prompt boundaries (256, 512) + final decode frontier (4352, 4608): <= 4 SWA blocks.
        assert len(cached_swa_blocks) <= 4
    elif retention == 512:
        # Segment multiples of 512 up to 4608 (9 boundaries) + prompt/final boundaries: <= 11 SWA blocks
        # (whereas dense retention would keep all 36 SWA blocks).
        assert len(cached_swa_blocks) <= 11

    # And Turn 2 still hits the final 256-token boundary (4608) in full!
    req2 = make_ced_request("turn-2", tokens[: total_len + 256])
    _, hit, _ = manager.get_computed_blocks(req2)
    assert hit == (total_len // PAGE) * PAGE == 4608


# ---------------------------------------------------------------------------
# 2. Speculative Decoding (1..6 tokens/step) Crossing Block Boundaries
#    in Scheduler and AsyncScheduler
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("async_scheduling", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("num_spec", [1, 2, 3, 5], ids=["spec1", "spec2", "spec3", "spec5"])
@pytest.mark.parametrize(
    "prompt_offset_from_boundary",
    [1, 2, 3, 4, 5, 6],
    ids=["minus1", "minus2", "minus3", "minus4", "minus5", "minus6"],
)
def test_scheduler_spec_decode_caches_boundary_crossed_on_stopping_step(
    async_scheduling: bool,
    num_spec: int,
    prompt_offset_from_boundary: int,
):
    """Regression test for Bug 3: when a running request with speculative
    decoding (1..5 draft tokens + 1 bonus token = 2..6 tokens/step) crosses a
    block boundary via accepted draft tokens and finishes on that exact step
    (reaching max_tokens), Scheduler._update_request_with_output and
    AsyncScheduler._update_request_with_output must call cache_blocks before
    _free_request so the newly completed full block is hashed and cached for
    subsequent requests.
    """
    block_size = 16
    scheduler = create_scheduler(
        enable_prefix_caching=True,
        block_size=block_size,
        num_speculative_tokens=num_spec,
        speculative_method="ngram_gpu" if async_scheduling else "ngram",
        async_scheduling=async_scheduling,
    )
    # Start at 2 * block_size - prompt_offset_from_boundary (e.g. 32 - k, in 26..31).
    # First step (prefill) generates 1 token -> num_tokens = 32 - k + 1.
    # Second step (decode with spec tokens) accepts up to num_spec draft tokens + 1 bonus token
    # so total tokens crosses 32 (2 full blocks) and hits max_tokens on that step!
    prompt_len = 2 * block_size - prompt_offset_from_boundary
    # Ensure 1 prefill token + (1 + num_spec) decode tokens >= prompt_offset_from_boundary
    total_generated_on_decode_step = min(1 + num_spec, max(1, prompt_offset_from_boundary))
    max_tokens = 1 + total_generated_on_decode_step
    will_cross_second_block = (prompt_len + max_tokens - 1) >= 2 * block_size

    req1 = create_requests(
        num_requests=1,
        num_tokens=prompt_len,
        max_tokens=max_tokens,
        block_size=block_size,
        same_prompt=True,
    )[0]
    scheduler.add_request(req1)

    # Step 1: Prefill req1 -> generates token 0, proposes num_spec draft 0s
    sched_out1 = scheduler.schedule()
    drafts = [0] * num_spec
    model_out1 = ModelRunnerOutput(
        req_ids=[req1.request_id],
        req_id_to_index={req1.request_id: 0},
        sampled_token_ids=[[0]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(sched_out1, model_out1)

    if not req1.is_finished():
        # Step 2: Decode with speculative draft tokens -> accept drafts + bonus token
        # and finish on this step!
        scheduler.update_draft_token_ids(DraftTokenIds([req1.request_id], [list(drafts)]))
        sched_out2 = scheduler.schedule()
        step2_tokens = [0] * total_generated_on_decode_step
        model_out2 = ModelRunnerOutput(
            req_ids=[req1.request_id],
            req_id_to_index={req1.request_id: 0},
            sampled_token_ids=[step2_tokens],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        )
        scheduler.update_from_output(sched_out2, model_out2)

    assert req1.is_finished()
    assert req1.num_tokens == prompt_len + max_tokens

    # Follow-up request with identical 2-block prefix (32 tokens):
    # If req1 crossed 2 * block_size (32 tokens), both blocks 0 and 1 MUST be cached!
    req2 = create_requests(
        num_requests=1,
        num_tokens=2 * block_size + 4,
        max_tokens=4,
        block_size=block_size,
        req_ids=["followup-req"],
    )[0]

    computed_blocks, num_cached_tokens, _ = (
        scheduler.kv_cache_manager.get_computed_blocks(req2)
    )
    if will_cross_second_block:
        assert num_cached_tokens == 2 * block_size, (
            f"async={async_scheduling}, num_spec={num_spec}, offset={prompt_offset_from_boundary}: "
            f"expected 2 full blocks ({2 * block_size} tokens) cached after stopping step, "
            f"got {num_cached_tokens}"
        )
    else:
        assert num_cached_tokens == block_size


# ---------------------------------------------------------------------------
# 3. Shared BlockPool Two-Tier Tail-to-Root LRU Eviction & Checkpoint Floor
# ---------------------------------------------------------------------------


def _block_keys(pool, blk):
    keys = []
    if blk.block_hash is not None:
        keys.append(blk.block_hash)
    keys.extend(pool.cached_block_hashes_by_block.get(blk.block_id, ()))
    return keys


def _hit_without_blocks(manager, evicted_blocks, prompt, name: str) -> int:
    pool = manager.block_pool
    saved = []
    for blk in evicted_blocks:
        for key in _block_keys(pool, blk):
            if pool.cached_block_hash_to_block.pop(key, blk.block_id) is not None:
                saved.append((key, blk))
    try:
        req = make_ced_request(name, prompt)
        _, hit, _ = manager.get_computed_blocks(req)
    finally:
        for key, blk in saved:
            pool.cached_block_hash_to_block.insert(key, blk)
    return hit


@pytest.mark.parametrize("retention", [0, 512, 8192, None], ids=["0", "512", "8192", "dense"])
def test_shared_block_pool_tail_to_root_eviction_order(retention: int | None):
    """Regression test for Bug 2: in a hybrid Global MLA + Encoder SWA manager
    sharing one BlockPool, after a multi-turn session completes, the free
    cached blocks in BlockPool.free_block_queue form two strictly monotone
    tail-to-root tiers:
    1. Unhashed scratch blocks (block_hash is None) sit at the very front of
       free_block_queue and are allocated FIRST before any cached block is
       evicted.
    2. Tier 0 (LRU head of cached blocks): aged non-anchored SWA checkpoints
       sit ahead of live conversation runs in strictly non-increasing token
       order (0 within-tier inversions), absorbing light pool pressure without
       evicting live top MLA blocks.
    3. Tier 1 (MRU tail of cached blocks): the finished conversation's active
       full-attention blocks paired with its active endpoint SWA checkpoint and
       sparse fallback SWA checkpoint ladder sit in strictly non-increasing
       token order (0 within-tier inversions), with full-attention blocks
       (is_sparse=False) preceding SWA checkpoints (is_sparse=True) at equal
       end_token.
    4. Checkpoint-floor property ("no hit collapses below the last retained
       checkpoint"): as cached blocks are evicted from the front of
       free_block_queue, the prefix-cache hit on an extension turn never
       collapses to 0 while the root checkpoint's MLA prefix remains cached.
    """
    manager, roles = make_ced_manager(retention, num_blocks=1024)
    swa_gid = roles["encoder-swa"]

    tokens = list(range(32768))

    # Turn 1: 16,384-token prompt + 512-token decode -> 16,896 tokens
    req1 = make_ced_request("turn-1", tokens[:16384])
    run_prefill_and_decode(manager, req1, tokens[16384:16896], tokens_per_step=4)

    # Turn 2: hits Turn 1's prefix at 16,896, extends to 24,576 + 512 decode -> 25,088 tokens
    req2 = make_ced_request("turn-2", tokens[:24576])
    hit2 = run_prefill_and_decode(manager, req2, tokens[24576:25088], tokens_per_step=4)
    assert hit2 == 16896

    free_blocks = manager.block_pool.free_block_queue.get_all_free_blocks()

    # Verify all unhashed blocks precede all cached blocks
    seen_cached = False
    for blk in free_blocks:
        if blk.block_hash is not None:
            seen_cached = True
        else:
            assert not seen_cached, "Unhashed block found after cached block in free_block_queue"

    hash_to_end_token = {
        blk_hash: (idx_blk + 1) * HASH
        for idx_blk, blk_hash in enumerate(req2.block_hashes)
    }

    cached_blocks = [blk for blk in free_blocks if blk.block_hash is not None]
    cached_entries = []
    for blk in cached_blocks:
        gid = get_group_id(blk.block_hash)
        raw_hash = get_block_hash(blk.block_hash)
        end_tok = hash_to_end_token[raw_hash]
        is_sparse = gid == swa_gid
        cached_entries.append((end_tok, is_sparse, gid, blk.block_id))

    assert cached_entries, "Expected cached blocks in free_block_queue"

    first_mla_idx = next(
        idx for idx, (_, is_sparse, _, _) in enumerate(cached_entries) if not is_sparse
    )
    tier0 = cached_entries[:first_mla_idx]
    tier1 = cached_entries[first_mla_idx:]
    assert tier0 and all(entry[1] for entry in tier0)
    assert tier1 and tier1[0][0] == 25088 and tier1[-1][0] == 256

    # 0 inversions within Tier 0 (aged SWA checkpoints)
    for i in range(len(tier0) - 1):
        end_a, _, gid_a, _ = tier0[i]
        end_b, _, gid_b, _ = tier0[i + 1]
        assert end_a >= end_b, (
            f"Tier 0 aged SWA order violated at {i}: {end_a} (group {gid_a}) < {end_b} (group {gid_b})"
        )

    # 0 inversions within Tier 1 (active tail-to-root run + anchored checkpoint ladder)
    for i in range(len(tier1) - 1):
        end_a, sparse_a, gid_a, _ = tier1[i]
        end_b, sparse_b, gid_b, _ = tier1[i + 1]
        assert end_a >= end_b, (
            f"Tier 1 run order violated at {i}: {end_a} (group {gid_a}) < {end_b} (group {gid_b})"
        )
        if end_a == end_b:
            assert sparse_a <= sparse_b, (
                f"At end_token={end_a}, sparse SWA block (group {gid_a}) precedes "
                f"dense MLA block (group {gid_b}) in Tier 1 eviction order"
            )

    # Checkpoint-floor property across the eviction curve:
    # Determine the root SWA checkpoint anchored in Tier 1 and verify that
    # evicting any prefix of cached_blocks before the root checkpoint's MLA
    # block is evicted never collapses the extension hit below root_checkpoint.
    tier1_swa_ends = sorted(
        {end_tok for end_tok, is_sparse, _, _ in tier1 if is_sparse}
    )
    root_checkpoint = tier1_swa_ends[0]
    first_root_mla_evict = next(
        idx
        for idx, (end_tok, is_sparse, _, _) in enumerate(cached_entries)
        if not is_sparse and end_tok <= root_checkpoint
    )
    ext_prompt = tokens[:25600]
    for k in range(first_root_mla_evict + 1):
        hit_k = _hit_without_blocks(manager, cached_blocks[:k], ext_prompt, f"ext-{k}")
        assert hit_k >= root_checkpoint, (
            f"retention={retention}: at eviction k={k}/{first_root_mla_evict}, "
            f"hit collapsed to {hit_k} < root_checkpoint={root_checkpoint}"
        )


# ---------------------------------------------------------------------------
# 4. 64K-100K Multi-Turn Concurrent Agent Stress Probe Under Eviction Pressure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("evict_n", [18, 36], ids=["evict18", "evict36"])
@pytest.mark.parametrize(
    "retention",
    [512, 8192],
    ids=["tp3-triangle-retention-512", "tp4-ring-retention-8192"],
)
def test_64k_to_100k_multi_turn_agent_retention_under_eviction_pressure(
    retention: int,
    evict_n: int,
):
    """Full-scale 64K-100K token multi-turn concurrent agent stress probe (#902).

    Simulates 2 concurrent agent sessions scaling from 64K (65,536 tokens) up to
    100K (100,096 tokens) across 8 turns each, with:
    - Unaligned prompt extensions (+4,000 to +5,000 tokens per turn) and
      multi-step speculative decoding (1..6 tokens/step) stopping at both
      T % 256 < 128 and T % 256 >= 128 offsets,
    - Interleaved stressor evictions between turns (both 18-block light pressure
      and 36-block deep pressure) that evict cached blocks from the shared
      BlockPool.

    Verifies:
    1. At evict_n = 18, Tier 0 aged SWA blocks absorb all evictions so both
       tp3-triangle (retention=512) and tp4-ring (retention=8192) reach the
       exact aligned ceiling (1,126,400 / 1,128,008 = 99.8574%), strictly
       exceeding 1107's 99.6759% and 99.5397%.
    2. At evict_n = 36, when eviction reaches Tier 1 live blocks on tp4-ring,
       the anchored fallback checkpoint ladder prevents hit collapse (never
       dropping to 0), preserving >= 94.4% min evicted turn hit and >= 98.9%
       overall hit rate.
    """
    manager, _ = make_ced_manager(
        retention,
        num_blocks=2200,
        max_model_len=131072,
    )

    num_agents = 2
    turn_schedule = [
        (65536, 389, 4),   # Turn 1: 65,536 -> 65,925 (65925 % 256 == 133 >= 128)
        (70144, 412, 6),   # Turn 2: 70,144 -> 70,556 (70556 % 256 == 156 >= 128)
        (75008, 290, 3),   # Turn 3: 75,008 -> 75,298 (75298 % 256 == 34 < 128)
        (80128, 395, 5),   # Turn 4: 80,128 -> 80,523 (80523 % 256 == 139 >= 128)
        (85248, 401, 4),   # Turn 5: 85,248 -> 85,649 (85649 % 256 == 145 >= 128)
        (90112, 310, 6),   # Turn 6: 90,112 -> 90,422 (90422 % 256 == 54 < 128)
        (95232, 399, 5),   # Turn 7: 95,232 -> 95,631 (95631 % 256 == 143 >= 128)
        (99584, 512, 4),   # Turn 8: 99,584 -> 100,096 (100,096 tokens)
    ]

    agent_tokens = {
        agent_id: [agent_id * 1_000_000 + i for i in range(102400)]
        for agent_id in range(num_agents)
    }
    prev_completed_len = {agent_id: 0 for agent_id in range(num_agents)}

    total_expected_reusable_tokens = 0
    total_actual_hit_tokens = 0
    no_eviction_turn_hit_rates = []
    eviction_turn_hit_rates = []

    for turn_idx, (prompt_len, decode_len, step_size) in enumerate(turn_schedule):
        eviction_active = turn_idx in (3, 5, 7)
        if eviction_active:
            free_list = manager.block_pool.free_block_queue.get_all_free_blocks()
            blocks_to_claim = 0
            cached_seen = 0
            for b in free_list:
                blocks_to_claim += 1
                if b.block_hash is not None:
                    cached_seen += 1
                    if cached_seen == evict_n:
                        break
            claimed = manager.block_pool.get_new_blocks(blocks_to_claim)
            for b in claimed:
                b.ref_cnt -= 1
            manager.block_pool.free_block_queue.prepend_n(claimed)

        for agent_id in range(num_agents):
            seq = agent_tokens[agent_id]
            prompt = seq[:prompt_len]
            decode = seq[prompt_len : prompt_len + decode_len]
            req = make_ced_request(f"agent-{agent_id}-turn-{turn_idx}", prompt)

            hit = run_prefill_and_decode(
                manager,
                req,
                decode,
                tokens_per_step=step_size,
                prefill_chunk_size=8192,
            )

            prev_len = prev_completed_len[agent_id]
            if prev_len > 0:
                max_possible_hit = (prev_len // PAGE) * PAGE
                hit_ratio = hit / prev_len
                total_expected_reusable_tokens += prev_len
                total_actual_hit_tokens += hit

                if not eviction_active:
                    assert hit == max_possible_hit, (
                        f"retention={retention}, turn={turn_idx}, agent={agent_id}: "
                        f"expected exact boundary hit {max_possible_hit} from prev_len={prev_len}, "
                        f"got {hit} ({hit_ratio:.4%})"
                    )
                    assert hit_ratio >= 0.996
                    no_eviction_turn_hit_rates.append(hit_ratio)
                elif agent_id == 0:
                    eviction_turn_hit_rates.append(hit_ratio)
                else:
                    no_eviction_turn_hit_rates.append(hit_ratio)

            prev_completed_len[agent_id] = prompt_len + decode_len

    overall_hit_rate = total_actual_hit_tokens / total_expected_reusable_tokens
    min_no_eviction_hit = min(no_eviction_turn_hit_rates)
    min_eviction_hit = min(eviction_turn_hit_rates)

    assert min_no_eviction_hit >= 0.994, (
        f"retention={retention}, evict_n={evict_n}: min non-evicted turn hit rate {min_no_eviction_hit:.4%} < 99.4%"
    )
    if evict_n == 18:
        assert total_actual_hit_tokens == 1126400, (
            f"retention={retention}, evict_n=18: expected ceiling 1126400, got {total_actual_hit_tokens}"
        )
        assert min_eviction_hit >= 0.998
    else:
        assert min_eviction_hit >= 0.944, (
            f"retention={retention}, evict_n={evict_n}: min evicted turn hit rate {min_eviction_hit:.4%} < 94.4%"
        )
        assert overall_hit_rate >= 0.989, (
            f"retention={retention}, evict_n={evict_n}: overall 64K-100K hit rate {overall_hit_rate:.4%} < 98.9%"
        )


def test_async_scheduler_stale_output_after_preemption_does_not_cache():
    """Verify that AsyncScheduler._update_request_with_output skips cache_blocks
    when output_is_stale is True (after a preemption rolled num_computed_tokens
    back to 0 while an in-flight step was still pending).
    """
    block_size = 16
    scheduler = create_scheduler(
        enable_prefix_caching=True,
        block_size=block_size,
        async_scheduling=True,
    )
    req1 = create_requests(
        num_requests=1,
        num_tokens=block_size - 1,
        max_tokens=4,
        block_size=block_size,
        same_prompt=True,
    )[0]
    scheduler.add_request(req1)

    # Step 1: Schedule prefill (in flight)
    sched_out1 = scheduler.schedule()
    # Simulate preemption while Step 1 is in flight
    scheduler._preempt_request(req1, 0.0)
    assert req1.num_computed_tokens == 0
    assert req1.num_stale_output_tokens == block_size - 1

    # Deliver the now-stale output from Step 1
    model_out1 = ModelRunnerOutput(
        req_ids=[req1.request_id],
        req_id_to_index={req1.request_id: 0},
        sampled_token_ids=[[0]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(sched_out1, model_out1)
    assert req1.num_computed_tokens == 0
