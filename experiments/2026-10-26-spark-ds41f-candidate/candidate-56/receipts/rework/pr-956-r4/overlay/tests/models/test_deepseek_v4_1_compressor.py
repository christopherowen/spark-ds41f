# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Accepted-prefix ownership through the real compressor and metadata."""

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.cpu_test
def test_compressor_cache_rejects_duplicate_prefix_without_replacing_owner():
    from vllm.models.deepseek_v4_1.compressor import CompressorStateCache

    config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context={}),
        speculative_config=None,
    )
    owner = CompressorStateCache(config, "compressor.state_cache")
    with pytest.raises(ValueError):
        CompressorStateCache(config, "compressor.state_cache")
    assert config.compilation_config.static_forward_context[owner.prefix] is owner
    other = CompressorStateCache(config, "other.state_cache")
    assert config.compilation_config.static_forward_context[other.prefix] is other


@pytest.mark.cpu_test
def test_v41_caches_skip_per_allocation_zeroing(monkeypatch):
    """DeepSeek V4.1's byte pages (window, compressed latents, index keys) sit
    beside the compressors' FP32 rings, and every one declares that no unwritten
    slot reaches a result, as DeepSeek's reference reads only written slots: the
    cache is mixed-precision yet needs no per-allocation zeroing (its pool starts
    zeroed). The poison check still fills every new block."""
    from vllm.models.deepseek_v4_1.attention import DeepseekV4Attention, _Cache
    from vllm.models.deepseek_v4_1.compressor import CompressorStateCache
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec

    monkeypatch.delenv("VLLM_KV_CACHE_POISON", raising=False)
    ratios = [0] * 43
    ratios[2], ratios[20] = 2, 1
    config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=256, swa_block_size=None),
        compilation_config=SimpleNamespace(static_forward_context={}),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                num_hidden_layers=40,
                kv_source_layer_ids=[2, 20],
                compress_ratios=ratios,
            )
        ),
        speculative_config=None,
    )
    specs = {}
    for layer in (0, 2, 20, 24, 40):  # target, KV sources, CED decoder, drafter
        cache = _Cache(
            config,
            f"model.layers.{layer}.self_attn.swa_cache",
            kind="swa",
            window=128,
            draft=layer >= 40,
        )
        specs[cache.prefix] = cache.get_kv_cache_spec(config)
    for layer, ratio in ((2, 2), (20, 1)):
        prefix = f"model.layers.{layer}.self_attn"
        specs[prefix] = DeepseekV4Attention.get_kv_cache_spec(
            SimpleNamespace(is_kv_source=True, compress_ratio=ratio), config
        )
        index = _Cache(config, f"{prefix}.indexer.k_cache", kind="index", ratio=ratio)
        specs[index.prefix] = index.get_kv_cache_spec(config)
    ring = CompressorStateCache(
        config, "model.layers.2.self_attn.compressor.state_cache"
    )
    specs[ring.prefix] = ring.get_kv_cache_spec(config)

    assert all(spec.uses_only_written_slots for spec in specs.values())
    cache = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec([name], spec) for name, spec in specs.items()
        ],
    )
    assert cache.has_mixed_precision_kv_cache
    assert not cache.needs_kv_cache_zeroing
    monkeypatch.setenv("VLLM_KV_CACHE_POISON", "1")
    assert cache.needs_kv_cache_zeroing


def test_compressor_ring_writes_follow_the_worker_slot_mapping():
    """Ring metadata emits a write only for rows with a valid worker slot. With
    the runner's per-spec settings, a verify step past the ring capacity writes
    every real row into its request's ring block, and graph padding writes
    nothing."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("DeepSeek-V4.1 metadata kernels require SM12x")

    from vllm.models.deepseek_v4_1.compressor import CompressorStateCache
    from vllm.models.deepseek_v4_1.sparse_mla import DeepseekV41B12xMetadataBuilder
    from vllm.v1.kv_cache_interface import MLAAttentionSpec
    from vllm.v1.worker.gpu.block_table import BlockTables, slot_mapping_mode

    device = torch.device("cuda", torch.cuda.current_device())
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=64, max_num_seqs=4),
        compilation_config=SimpleNamespace(static_forward_context={}),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        speculative_config=SimpleNamespace(
            num_speculative_tokens=7, num_verification_tokens=7, parallel_drafting=False
        ),
    )
    ring = CompressorStateCache(config, "layer.state_cache").get_kv_cache_spec(config)
    main = MLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=288,
        dtype=torch.uint8,
        tokens_per_state=2,
    )
    specs = (main, ring)
    enabled, circular = zip(*(slot_mapping_mode(spec) for spec in specs))
    tables = BlockTables(
        block_sizes=[spec.block_size for spec in specs],
        max_num_reqs=4,
        max_num_batched_tokens=64,
        max_num_blocks_per_group=[2, 1],
        device=device,
        kernel_block_sizes=[spec.block_size for spec in specs],
        slot_mapping_enabled=list(enabled),
        slot_mapping_circular=list(circular),
    )
    tables.append_block_ids(req_index=0, new_block_ids=([1], [5]), overwrite=True)
    tables.append_block_ids(req_index=1, new_block_ids=([2], [6]), overwrite=True)
    tables.apply_staged_writes()

    # A verify step (bonus token and seven drafts) at positions 40-47, a single
    # token at position 3, then two tokens of graph padding.
    positions = list(range(40, 48)) + [3, 0, 0]
    idx_mapping = torch.tensor([0, 1], dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, 8, 9], dtype=torch.int32, device=device)
    slot_mappings = tables.compute_slot_mappings(
        idx_mapping,
        query_start_loc,
        torch.tensor(positions, dtype=torch.int64, device=device),
        num_tokens_padded=len(positions),
    )
    cm = SimpleNamespace(
        query_start_loc=query_start_loc,
        seq_lens=torch.tensor([48, 4], dtype=torch.int32, device=device),
        block_table_tensor=tables.gather_block_tables(idx_mapping, 2)[1],
        slot_mapping=slot_mappings[1],
        num_reqs=2,
        num_actual_tokens=len(positions),
        max_query_len=8,
        max_seq_len=48,
    )
    builder = DeepseekV41B12xMetadataBuilder(ring, ["layer"], config, device)
    state = builder.build(0, cm)
    torch.accelerator.synchronize()

    size = ring.block_size
    expected = [5 * size + p % size for p in range(40, 48)] + [6 * size + 3, -1, -1]
    assert state.slot_mapping[: len(positions)].tolist() == expected


@pytest.fixture
def compressor_runners(monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("DeepSeek-V4.1 compression requires SM12x")
    pytest.importorskip("tilelang")

    import vllm.v1.worker.workspace as workspace
    from vllm.model_executor import parameter
    from vllm.model_executor.layers import linear
    from vllm.models.deepseek_v4_1.compressor import DeepseekCompressor
    from vllm.models.deepseek_v4_1.sparse_mla import (
        DeepseekV41B12xMetadataBuilder,
    )
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    for module in (linear, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 1)

    device = torch.device("cuda", torch.cuda.current_device())
    # The split projection borrows its decode partials from vLLM's workspace.
    monkeypatch.setattr(workspace, "_manager", workspace.WorkspaceManager(device))
    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: 0)
    capacity, requests, width = 96, 3, 16
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=capacity, max_num_seqs=requests
        ),
        compilation_config=SimpleNamespace(static_forward_context={}),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        speculative_config=None,
    )
    torch.manual_seed(411)
    original = torch.randn(3, 64, 128, device=device, dtype=torch.bfloat16)
    replacement = torch.randn_like(original)

    class Runner:
        def __init__(self, name):
            self.module = DeepseekCompressor(config, 2, 128, 512, prefix=name).to(
                device
            )
            self.spec = self.module.state_cache.get_kv_cache_spec(config)
            # Match allocator block-outer views, including a non-dense pool stride.
            self.state_width = self.spec.block_size * 1024
            self.backing = torch.full((64, self.state_width + 16), 123.0, device=device)
            self.pool = self.backing[:, : self.state_width].view(
                64, self.spec.block_size, 1024
            )
            self.module.state_cache.bind_kv_cache(self.pool)
            self.hidden = torch.empty(
                capacity, 128, device=device, dtype=torch.bfloat16
            )
            self.starts = torch.zeros(requests + 1, device=device, dtype=torch.int32)
            self.seq = torch.zeros(requests, device=device, dtype=torch.int32)
            self.input_slots = torch.full(
                (capacity,), -1, device=device, dtype=torch.int64
            )
            self.table = torch.zeros(requests, width, device=device, dtype=torch.int32)
            # Each identity owns a private state ring in its first physical block.
            self.pages = torch.arange(1, 49, device=device, dtype=torch.int32).view(
                3, width
            )
            main_spec = MLAAttentionSpec(
                block_size=128,
                num_kv_heads=1,
                head_size=288,
                dtype=torch.uint8,
                tokens_per_state=2,
            )
            self.builders = [
                DeepseekV41B12xMetadataBuilder(spec, [name], config, device)
                for spec in (main_spec, self.spec)
            ]
            self.graph = None

        def stage(self, identities, positions, lengths, source, pad=False, null=None):
            starts = [0]
            for length in lengths:
                starts.append(starts[-1] + length)
            n = starts[-1]
            # With ``pad`` the last request's rows are graph padding, as the runner
            # pads a captured step: past every request's range, without slots,
            # while its block table row still names its pages.
            live = len(identities) - pad
            self.starts.copy_(
                torch.tensor(
                    starts[: live + 1] + [starts[live]] * (requests - live),
                    device=device,
                    dtype=torch.int32,
                )
            )
            self.seq.zero_()
            self.table.zero_()
            self.hidden.fill_(float("nan"))
            self.input_slots.fill_(-1)
            for row, (identity, pos, length) in enumerate(
                zip(identities, positions, lengths)
            ):
                self.seq[row] = pos + length
                self.table[row].copy_(self.pages[identity])
                self.hidden[starts[row] : starts[row + 1]].copy_(
                    source[identity, pos : pos + length]
                )
                self.input_slots[starts[row] : starts[row + 1]].fill_(1)
            if pad:
                self.input_slots[starts[-2] : n].fill_(-1)
            if null is not None:
                self.table[:, null] = 0
            cm = SimpleNamespace(
                query_start_loc=self.starts,
                query_start_loc_cpu=self.starts[: live + 1].cpu(),
                seq_lens=self.seq,
                block_table_tensor=self.table,
                slot_mapping=self.input_slots,
                num_reqs=live,
                num_actual_tokens=n,
                max_query_len=max(lengths[:live], default=0),
                max_seq_len=max(
                    (pos + length for pos, length in zip(positions, lengths[:live])),
                    default=0,
                ),
            )
            self.main = self.builders[0].build(0, cm)
            state_cm = SimpleNamespace(**vars(cm))
            state_cm.block_table_tensor = self.table[:, :1]
            self.state = self.builders[1].build(0, state_cm)
            return n

        def run(self):
            if self.graph is None:
                self.module(self.hidden, self.main, self.state)
            else:
                self.graph.replay()
            n = self.main.num_actual_tokens
            return self.module._latent[:n].clone(), self.module._slots[:n].clone()

    with torch.no_grad(), workspace.use_workspace_lane(0):
        candidate, reference = Runner("candidate"), Runner("reference")
        candidate.module.fused_wkv_wgate.weight.normal_(0, 0.1)
        candidate.module.norm.weight.uniform_(0.5, 1.5)
        reference.module.load_state_dict(candidate.module.state_dict())
        for runner in (candidate, reference):
            # The worker declares the projection's and the compression's kernels
            # after loading; outside its registry they compile and bind at once.
            projection = runner.module.fused_wkv_wgate
            projection.quant_method.declare_layer_kernels(projection)
            runner.module.declare_kernels()
            runner.stage([0, 1], [0, 0], [10, 10], original)
            runner.run()
        try:
            yield candidate, reference, original, replacement
        finally:
            for runner in (candidate, reference):
                if runner.graph is not None:
                    runner.graph.reset()
                    runner.graph = None


def test_compressor_rejection_reorder_padding_and_page_boundaries(
    compressor_runners,
):
    from vllm.models.deepseek_v4_1.tilelang import metadata as tl_metadata

    candidate, reference, original, replacement = compressor_runners
    device = original.device
    width = candidate.table.shape[1]
    with torch.no_grad():
        # Capture fixed-capacity forward once; counts, tables and ordering change.
        candidate.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(candidate.graph):
            candidate.module(candidate.hidden, candidate.main, candidate.state)

        for boundary in (10, 11, 31, 32):
            for accepted in range(6):
                candidate.backing.fill_(123)
                reference.backing.fill_(123)
                candidate.stage([0, 1], [0, 0], [boundary, boundary], original)
                candidate.run()
                candidate.stage([0, 1], [boundary, boundary], [6, 6], original)
                candidate.run()
                next_pos = boundary + accepted + 1
                # The verifier has touched the whole suffix, but only its accepted
                # prefix belongs to history. The oracle never executes that suffix.
                reference.stage([0, 1], [0, 0], [next_pos, next_pos], original)
                reference.run()
                for runner in (candidate, reference):
                    runner.stage([1, 0], [next_pos, next_pos], [2, 2], replacement)
                actual, slots = candidate.run()
                expected, expected_slots = reference.run()
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                torch.testing.assert_close(slots, expected_slots, atol=0, rtol=0)
                assert (slots >= 0).sum().item() == 2
                assert (candidate.backing[0] == 123).all()
                assert (candidate.backing[:, candidate.state_width :] == 123).all()

        # Graph padding must not be resurrected by a real block table: the padded
        # rows write neither latents nor the ring of the pages its table row names.
        # Serving gives every row a request owns a slot; only rows past the step's
        # requests are padding (the ring-writes test above).
        before = candidate.pool[33:49].clone()
        candidate.stage([0, 1, 2], [20, 20, 20], [2, 2, 2], replacement, pad=True)
        _, slots = candidate.run()
        assert slots[-2:].tolist() == [-1, -1]
        torch.testing.assert_close(candidate.pool[33:49], before, atol=0, rtol=0)
        assert candidate.state.slot_mapping[4:].eq(-1).all()
        assert candidate.state.req_id_per_token[4:].eq(-1).all()

        # An absent request-state block must neither supply carry nor receive
        # partials, even if incoming token slots claim it is writable.
        candidate.stage([0], [5], [1], replacement, null=0)
        _, slots = candidate.run()
        assert slots.tolist() == [-1]
        assert candidate.state.slot_mapping[0].item() == -1
        assert (candidate.backing[0] == 123).all()

        # Sparse reads must also preserve holes rather than attending null page0.
        candidate.stage([0], [8], [1], replacement, null=1)
        sparse = torch.empty(1, 16, device=device, dtype=torch.int32)
        lengths = torch.empty(1, device=device, dtype=torch.int32)
        top_lengths = torch.empty_like(lengths)
        for draft in (False, True):
            tl_metadata.chunk(
                candidate.state.positions,
                candidate.state.req_id_per_token,
                candidate.table,
                sparse,
                lengths,
                top_lengths,
                candidate.state.cache_lengths,
                candidate.state.query_start_loc,
                candidate.state.request_positions,
                0,
                1,
                page=4,
                window=8,
                draft=draft,
            )
            logical = list(range(0 if draft else 1, 9))
            expected = [
                -1 if pos // 4 == 1 else (1 + pos // 4) * 4 + pos % 4 for pos in logical
            ]
            expected += [-1] * (16 - len(expected))
            assert sparse[0].tolist() == expected

        # Compressed/index page tables use the same null-block convention. A row
        # writes only its live pages; here every column is live (``width`` visible
        # entries, one a page).
        pages = torch.empty((1, width), device=device, dtype=torch.int32)
        visible = torch.full_like(candidate.state.req_id_per_token, width)
        tl_metadata.pages(
            candidate.state.req_id_per_token,
            visible,
            candidate.table,
            pages,
            0,
            1,
            width,
            page=1,
        )
        expected_pages = list(range(1, width + 1))
        expected_pages[1] = -1
        assert pages[0].tolist() == expected_pages

        # Empty graph replay must not change any previously accepted partial row.
        before = candidate.backing.clone()
        candidate.stage([], [], [], original)
        candidate.run()
        torch.testing.assert_close(candidate.backing, before, atol=0, rtol=0)


def test_compressor_high_page_offsets(compressor_runners):
    candidate, reference, original, replacement = compressor_runners
    device = original.device
    with torch.no_grad():
        # Recycled state pages can exceed Int32 element offsets even though
        # their page IDs still fit Int32. Warm/capture with that actual pool.
        stride = candidate.state_width + 16
        high_page = 2**31 // stride + 2
        required = (high_page + 1) * stride * 4
        if torch.cuda.mem_get_info()[0] < required + 512 * 1024**2:
            pytest.skip("requires just over 8 GiB for the partial-state offset check")
        high_backing = torch.empty((high_page + 1, stride), device=device)
        high_backing[high_page].fill_(123)
        candidate.graph = None
        candidate.module.state_cache.bind_kv_cache(
            high_backing[:, : candidate.state_width].view(
                high_page + 1, candidate.spec.block_size, 1024
            )
        )
        candidate.pages[0, 0] = high_page
        for runner in (candidate, reference):
            runner.stage([0], [0], [1], original)
            runner.run()
        candidate.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(candidate.graph):
            candidate.module(candidate.hidden, candidate.main, candidate.state)
        for runner in (candidate, reference):
            runner.stage([0], [1], [1], replacement)
        actual, _ = candidate.run()
        expected, _ = reference.run()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert (high_backing[high_page, candidate.state_width :] == 123).all()


@pytest.mark.parametrize("ratio", [1, 2])
def test_compressor_tp_sharded_fused_wkv_wgate_matches_tp1(monkeypatch, ratio):
    """At TP4 each rank holds 1/4 of ``wkv`` (and ``wgate`` at ratio 2), projects
    its local shard with the same 5-shard blocked K reduction, and all-gathers
    ``_values`` (and ``_gates``) bit for bit equal to TP1 across decode and prefill."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("DeepSeek-V4.1 compression requires SM12x")
    pytest.importorskip("tilelang")

    import vllm.models.deepseek_v4_1.compressor as comp_mod
    import vllm.v1.worker.workspace as workspace
    from vllm.model_executor import parameter
    from vllm.model_executor.layers import linear
    from vllm.models.deepseek_v4_1.compressor import DeepseekCompressor

    device = torch.device("cuda", torch.cuda.current_device())
    monkeypatch.setattr(workspace, "_manager", workspace.WorkspaceManager(device))
    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: 0)
    capacity, requests, hidden = 160, 4, 5120

    def make_config():
        return SimpleNamespace(
            scheduler_config=SimpleNamespace(
                max_num_batched_tokens=capacity, max_num_seqs=requests
            ),
            compilation_config=SimpleNamespace(static_forward_context={}),
            parallel_config=SimpleNamespace(decode_context_parallel_size=1),
            speculative_config=None,
        )

    gen = torch.Generator(device=device).manual_seed(77 + ratio)
    wkv = (torch.randn((512, hidden), generator=gen, device=device) * 0.02).bfloat16()
    wgate = (
        (torch.randn((512, hidden), generator=gen, device=device) * 0.02).bfloat16()
        if ratio == 2
        else None
    )

    def build_rank(tp_rank: int, tp_size: int) -> DeepseekCompressor:
        for mod in (linear, parameter):
            monkeypatch.setattr(mod, "get_tensor_model_parallel_rank", lambda: tp_rank)
            monkeypatch.setattr(
                mod, "get_tensor_model_parallel_world_size", lambda: tp_size
            )
        comp = DeepseekCompressor(
            make_config(), ratio, hidden, 512, prefix=f"r{tp_rank}_{tp_size}"
        ).to(device)
        param = comp.fused_wkv_wgate.weight
        param.weight_loader(param, wkv, 0)
        if ratio == 2:
            param.weight_loader(param, wgate, 1)
        valid = comp.fused_wkv_wgate.valid_part_cols
        part = comp.fused_wkv_wgate.part_cols
        if valid < part:
            param.data[valid:part].fill_(99.0)
            if ratio == 2:
                param.data[part + valid :].fill_(99.0)
        comp.fused_wkv_wgate.process_weights_after_loading()
        comp.fused_wkv_wgate.quant_method.declare_layer_kernels(comp.fused_wkv_wgate)
        comp._ensure_capacity_buffers(device, capacity, requests)
        return comp

    with torch.no_grad(), workspace.use_workspace_lane(0):
        from vllm.models.deepseek_v4_1.tilelang.linear import unpack_bf16_layer

        tp1 = build_rank(0, 1)
        assert tp1.fused_wkv_wgate.tp_size == 1
        assert hasattr(tp1.fused_wkv_wgate, "packed_bf16_exponent_base")
        assert tp1.fused_wkv_wgate.weight.shape == (
            512 * ratio,
            hidden + hidden // 2,
        )
        assert unpack_bf16_layer(tp1.fused_wkv_wgate).shape == (512 * ratio, hidden)

        x = torch.randn(
            (capacity, hidden), generator=gen, device=device, dtype=torch.bfloat16
        )
        expected_by_rows = {}
        for rows in (1, 7, 16, 64, 160):
            inp = x[:rows]
            tp1._project(inp)
            expected_by_rows[rows] = (
                tp1._values[:rows].clone(),
                tp1._gates[:rows].clone() if ratio == 2 else None,
            )

        from vllm.models.deepseek_v4_1.compressor import _compressor_part_cols
        from vllm.models.deepseek_v4_1.tilelang.linear import project_parts

        for tp_size in (2, 3, 4, 5, 6, 7, 8):
            part_cols = _compressor_part_cols(tp_size)
            ranks = [build_rank(r, tp_size) for r in range(tp_size)]
            for r, comp in enumerate(ranks):
                assert comp.fused_wkv_wgate.tp_size == tp_size
                assert hasattr(comp.fused_wkv_wgate, "packed_bf16_exponent_base")
                assert comp.fused_wkv_wgate.weight.shape == (
                    part_cols * ratio,
                    hidden + hidden // 2,
                )
                unpacked = unpack_bf16_layer(comp.fused_wkv_wgate)
                assert unpacked.shape == (part_cols * ratio, hidden)
                r0 = r * part_cols
                r1 = min(512, r0 + part_cols)
                valid = max(0, r1 - r0)
                if valid > 0:
                    assert torch.equal(unpacked[:valid], wkv[r0:r1])
                if valid < part_cols:
                    assert not unpacked[valid:part_cols].any()
                if ratio == 2:
                    wgate_part = unpacked[part_cols:]
                    if valid > 0:
                        assert torch.equal(wgate_part[:valid], wgate[r0:r1])
                    if valid < part_cols:
                        assert not wgate_part[valid:part_cols].any()

            for rows in (1, 7, 16, 64, 160):
                inp = x[:rows]
                ref_values, ref_gates = expected_by_rows[rows]

                shards = []
                for comp in ranks:
                    local = comp._local_shards[: ratio * rows].view(
                        ratio, rows, part_cols
                    )
                    project_parts(comp.fused_wkv_wgate, inp, tuple(local), rows)
                    shards.append(local.clone())

                def all_gather_rows(
                    tensor, dim=-1, rows=rows, shards=shards, part_cols=part_cols
                ):
                    assert dim == 0
                    assert tensor.shape == (ratio * rows, part_cols)
                    assert torch.equal(tensor.view(ratio, rows, part_cols), shards[0])
                    return torch.cat(
                        [s.view(ratio * rows, part_cols) for s in shards], dim=0
                    )

                monkeypatch.setattr(
                    comp_mod, "tensor_model_parallel_all_gather", all_gather_rows
                )
                ranks[0]._project(inp)
                assert torch.equal(
                    ranks[0]._values[:rows], ref_values
                ), f"tp={tp_size} rows={rows}"
                if ratio == 2:
                    assert torch.equal(
                        ranks[0]._gates[:rows], ref_gates
                    ), f"tp={tp_size} rows={rows}"
                assert not ranks[0].fused_wkv_wgate.tilelang_counters.any()


@pytest.mark.parametrize("world_size", [3, 4])
def test_sp_global_projection_matches_native_shards_at_prefill_boundaries(world_size):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("DeepSeek-V4.1 compression requires SM12x")
    pytest.importorskip("tilelang")
    from vllm.models.deepseek_v4_1 import sp_prefill
    from vllm.models.deepseek_v4_1.compressor import _compressor_part_cols
    from vllm.models.deepseek_v4_1.tilelang.gemm import bf16_gemm

    gen = torch.Generator(device="cuda").manual_seed(3601)
    weight = (torch.randn(512, 5120, device="cuda", generator=gen) * 0.02).bfloat16()
    inputs = torch.randn(16384, 5120, device="cuda", generator=gen).bfloat16()
    # Preserve the deployed column shard's five blocked K parts in both layouts.
    part_cols = _compressor_part_cols(world_size)
    padded_weight = torch.zeros(
        (world_size * part_cols, 5120), device="cuda", dtype=torch.bfloat16
    )
    padded_weight[:512].copy_(weight)
    base = bf16_gemm(part_cols, 5120, shards=5, blocked=True, out_dtype="bfloat16")
    candidate = bf16_gemm(512, 5120, shards=5, blocked=True, out_dtype="bfloat16")
    for nonfinite in (False, True):
        if nonfinite:
            inputs[0, 0] = float("nan")
            inputs[1, 1] = float("inf")
            inputs[2, 2] = -float("inf")
        for tokens in (205, 206, 207, 385, 6144, 16384):
            x = inputs[:tokens]
            columns = []
            for start in range(0, world_size * part_cols, part_cols):
                out = torch.empty(
                    tokens, part_cols, device="cuda", dtype=torch.bfloat16
                )
                base(x, padded_weight[start : start + part_cols], out)
                columns.append(out)
            expected = torch.cat(columns, dim=-1)[:, :512]
            row_parts = []
            for rank in range(world_size):
                sp = sp_prefill.SPRows(tokens, world_size, rank)
                out = torch.empty(
                    sp.local_rows, 512, device="cuda", dtype=torch.bfloat16
                )
                candidate(sp.local_slice(x), weight, out)
                row_parts.append(out)
            actual = torch.cat(row_parts)[:tokens]
            assert torch.equal(actual.view(torch.int16), expected.view(torch.int16)), (
                world_size,
                tokens,
                nonfinite,
            )


@pytest.mark.parametrize("world_size", [2, 3, 4, 5, 6, 7, 8])
def test_sp_ratio2_projection_matches_full_rows(monkeypatch, world_size):
    """The ratio-2 SP projection, bound as serving binds it, gives the sharded
    projection's values and gates bit for bit: every rank's ``bf16_gemm_two`` on its
    local rows, gathered, equals the ranks' ``project_parts`` of every row plus
    ``gather_columns_into`` (the path without SP) and TP1's ``project_parts``. bind()
    gathers the unpacked shards once and unpads each part into ``[wkv; wgate]``; the
    decode projection keeps the rank's packed shard."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("DeepSeek-V4.1 compression requires SM12x")
    pytest.importorskip("tilelang")

    import vllm.models.deepseek_v4_1.compressor as comp_mod
    import vllm.v1.worker.workspace as workspace
    from vllm.model_executor import parameter
    from vllm.model_executor.layers import linear
    from vllm.models.deepseek_v4_1 import sp_prefill
    from vllm.models.deepseek_v4_1.compressor import (
        DeepseekCompressor,
        _compressor_part_cols,
    )
    from vllm.models.deepseek_v4_1.tilelang.linear import (
        project_parts,
        unpack_bf16_layer,
    )

    device = torch.device("cuda", torch.cuda.current_device())
    monkeypatch.setattr(workspace, "_manager", workspace.WorkspaceManager(device))
    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: 0)
    capacity, requests, hidden = 16384, 4, 5120
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=capacity, max_num_seqs=requests
        ),
        compilation_config=SimpleNamespace(static_forward_context={}),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        speculative_config=None,
    )
    gen = torch.Generator(device=device).manual_seed(9562 + world_size)
    wkv = (torch.randn((512, hidden), generator=gen, device=device) * 0.02).bfloat16()
    wgate = (torch.randn((512, hidden), generator=gen, device=device) * 0.02).bfloat16()

    def build_rank(tp_rank: int, tp_size: int) -> DeepseekCompressor:
        for mod in (linear, parameter):
            monkeypatch.setattr(mod, "get_tensor_model_parallel_rank", lambda: tp_rank)
            monkeypatch.setattr(
                mod, "get_tensor_model_parallel_world_size", lambda: tp_size
            )
        comp = DeepseekCompressor(
            config, 2, hidden, 512, prefix=f"sp2_r{tp_rank}_{tp_size}"
        ).to(device)
        param = comp.fused_wkv_wgate.weight
        param.weight_loader(param, wkv, 0)
        param.weight_loader(param, wgate, 1)
        comp.fused_wkv_wgate.process_weights_after_loading()
        comp.fused_wkv_wgate.quant_method.declare_layer_kernels(comp.fused_wkv_wgate)
        return comp

    part_cols = _compressor_part_cols(world_size)
    with torch.no_grad(), workspace.use_workspace_lane(0):
        tp1 = build_rank(0, 1)
        assert not tp1._sp_enabled
        tp1._ensure_capacity_buffers(device, capacity, requests)
        ranks = [build_rank(r, world_size) for r in range(world_size)]
        for comp in ranks:
            assert comp._sp_enabled and comp.fused_wkv_wgate.tp_size == world_size
            assert comp.fused_wkv_wgate.weight.shape[0] == 2 * part_cols
        shards = [unpack_bf16_layer(comp.fused_wkv_wgate).detach() for comp in ranks]
        gathers = []

        def gather_weights(tensor, dim=-1):
            assert dim == 0
            gathers.append(tuple(tensor.shape))
            return torch.cat(shards, dim=0)

        monkeypatch.setattr(
            comp_mod, "tensor_model_parallel_all_gather", gather_weights
        )
        for comp in ranks:
            comp.declare_kernels()
        assert gathers == [(2 * part_cols, hidden)] * world_size
        for comp in ranks:
            # The decode projection keeps the rank's own (packed) shard.
            assert comp._sp_weight.data_ptr() != comp.fused_wkv_wgate.weight.data_ptr()
            assert comp._sp_weight.shape == (1024, hidden)
            assert torch.equal(comp._sp_weight[:512], wkv)
            assert torch.equal(comp._sp_weight[512:], wgate)

        x = torch.randn(
            (capacity, hidden), generator=gen, device=device, dtype=torch.bfloat16
        )
        for nonfinite in (False, True):
            if nonfinite:
                x[0, 0] = float("nan")
                x[1, 1] = float("inf")
                x[2, 2] = -float("inf")
            for tokens in (129, 205, 206, 207, 257, 385, 4097, 16384):
                inp = x[:tokens]
                tp1._project(inp)
                expected = (
                    tp1._values[:tokens].clone(),
                    tp1._gates[:tokens].clone(),
                )
                # Without SP: every rank projects its column shard of every row; the
                # column all-gather and gather_columns_into rebuild the 512 columns.
                local_shards = []
                for comp in ranks:
                    local = comp._local_shards[: 2 * tokens].view(2, tokens, part_cols)
                    project_parts(comp.fused_wkv_wgate, inp, tuple(local), tokens)
                    local_shards.append(local.clone())

                def gather_columns(tensor, dim=-1, local_shards=local_shards):
                    assert dim == 0
                    return torch.cat(
                        [s.view(-1, part_cols) for s in local_shards], dim=0
                    )

                monkeypatch.setattr(
                    comp_mod, "tensor_model_parallel_all_gather", gather_columns
                )
                ranks[0]._project(inp)
                sharded = (
                    ranks[0]._values[:tokens].clone(),
                    ranks[0]._gates[:tokens].clone(),
                )
                values, gates = [], []
                for rank, comp in enumerate(ranks):
                    sp = sp_prefill.SPRows(tokens, world_size, rank)
                    comp._sp_kernel(
                        sp.local_slice(inp),
                        comp._sp_weight,
                        comp._values[: sp.local_rows],
                        comp._gates[: sp.local_rows],
                    )
                    values.append(comp._values[: sp.local_rows].clone())
                    gates.append(comp._gates[: sp.local_rows].clone())
                actual = (torch.cat(values)[:tokens], torch.cat(gates)[:tokens])
                case = (world_size, tokens, nonfinite)
                for got, want, ref in zip(actual, sharded, expected):
                    got, want, ref = (t.view(torch.int32) for t in (got, want, ref))
                    assert torch.equal(got, want), case
                    assert torch.equal(want, ref), case
