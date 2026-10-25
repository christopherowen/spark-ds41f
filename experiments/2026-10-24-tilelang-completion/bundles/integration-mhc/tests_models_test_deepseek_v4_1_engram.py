# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import json
import pickle
import time
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.preparation import PreparationSession
from b12x.sequence import engram as native
from safetensors.torch import save_file
from torch import nn

from vllm.config.engram import EngramConfig
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader import default_loader, weight_utils
from vllm.model_executor.models.utils import WeightsMapper
from vllm.models.deepseek_v4_1.common import engram as engram_module
from vllm.models.deepseek_v4_1.common.engram import (
    Engram,
    NgramHashState,
    ParallelEngramEmbedding,
)
from vllm.models.deepseek_v4_1.nvidia.dspark import DSparkDeepseekV4ForCausalLM
from vllm.models.deepseek_v4_1.nvidia.model import (
    DeepseekV4Model,
    DeepseekV41LLMForCausalLM,
)
from vllm.models.deepseek_v4_1.nvidia.model_state import DeepseekV41ModelState
from vllm.models.deepseek_v4_1.nvidia.vl_model import (
    DeepseekV41ForCausalLM,
    _make_deepseek_v4_vl_weights_mapper,
)
from vllm.models.deepseek_v41.common.mm_preprocess import image_sentinel_mask
from vllm.utils.b12x import B12xWorkload
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.workspace import collect_cuda_graph_capture_resources


@pytest.fixture
def tiny_engram_checkpoint(tmp_path, dist_init):
    """Real target modules and native plans, omitting unrelated attention/MoE."""
    # Both global row counts leave two padded rows on the final TP4 shard.
    geometry = native.build_geometry(base_table_size=19, compressed_vocab_size=32)
    caps = tuple(
        native.Caps(
            device="cuda",
            max_tokens=8,
            max_seqs=2,
            max_requests=2,
            vocab_size=32,
            layer_id=layer_id,
            tp_size=1,
            tp_rank=0,
        )
        for layer_id in geometry.layer_ids
    )
    config = SimpleNamespace(
        hidden_size=64,
        hc_mult=4,
        rms_norm_eps=1e-6,
        num_attention_heads=1,
    )
    weights = {}
    paths = []
    for index, cap in enumerate(caps):
        prefix = f"layers.{cap.layer_id}.engram."
        rows = torch.arange(geometry.num_embeddings[index])[:, None]
        table = (rows.remainder(7) + index + 1).expand(-1, 256)
        layer_weights = {
            prefix + "embed.weight": table.to(torch.float8_e4m3fn).contiguous(),
            prefix + "embed.scale": torch.arange(124, 132, dtype=torch.uint8)
            .expand(geometry.num_embeddings[index], -1)
            .contiguous()
            .view(torch.float8_e8m0fnu),
            prefix + "q_weight": torch.full((4, 64), 0.5, dtype=torch.bfloat16),
            prefix + "k_weight": torch.full((4, 64), 0.25, dtype=torch.bfloat16),
            prefix + "wkv.weight": torch.full(
                (320, 6144), (index + 1) / 8192, dtype=torch.bfloat16
            ),
        }
        path = tmp_path / f"layer-{index}.safetensors"
        save_file(layer_weights, str(path))
        paths.append(path)
        weights.update(layer_weights)

    def make_model(memory, *, tp_size=1, tp_rank=0):
        session = PreparationSession(device=torch.device("cuda"), autotune=False)
        local_caps = tuple(
            replace(cap, tp_size=tp_size, tp_rank=tp_rank) for cap in caps
        )
        layout = SimpleNamespace(
            caps=local_caps,
            hash_plans=tuple(
                native.plan(caps, token_map=list(range(32)), geometry=geometry)
                for caps in local_caps
            ),
            lookup_plans=tuple(
                None
                if memory == "disk"
                else native.plan(
                    caps,
                    token_map=list(range(32)),
                    geometry=geometry,
                    invocation={
                        "operation": "lookup",
                        "compact_rows": False,
                        "resident_scales": False,
                    },
                )
                for caps in local_caps
            ),
            layer_ids=geometry.layer_ids,
            geometry=geometry,
            token_map=list(range(32)),
            table_memory=memory,
        )
        target = DeepseekV4Model.__new__(DeepseekV4Model)
        nn.Module.__init__(target)
        target.config = config
        target.disk_engram = memory == "disk"
        target.file_backed_engram = memory in ("ram", "disk")
        target.engram_layout = layout
        target.start_layer, target.end_layer = 0, max(layout.layer_ids) + 1
        target.layers = nn.ModuleList(nn.Module() for _ in range(target.end_layer))
        old_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            with torch.device("cuda"):
                for index, layer_id in enumerate(layout.layer_ids):
                    target.layers[layer_id].engram = Engram(
                        config,
                        None,
                        layout,
                        index,
                        False,
                        f"model.layers.{layer_id}.engram",
                    )
        finally:
            torch.set_default_dtype(old_dtype)
        target.engram_hash = NgramHashState(None, layout, None)
        if memory == "disk":
            target.register_engram_hashes()
        target.get_expert_mapping = lambda: []
        language = DeepseekV41LLMForCausalLM.__new__(DeepseekV41LLMForCausalLM)
        nn.Module.__init__(language)
        language.model = target
        language.hf_to_vllm_mapper = WeightsMapper()
        root = DeepseekV41ForCausalLM.__new__(DeepseekV41ForCausalLM)
        nn.Module.__init__(root)
        root.language_model = language
        root.hf_to_vllm_mapper = _make_deepseek_v4_vl_weights_mapper(
            "fp4", "weight_scale_inv"
        )
        return root, target, session

    return make_model, weights, paths


def _engrams(target):
    return tuple(
        target.layers[index].engram for index in target.engram_layout.layer_ids
    )


def _load_checkpoint(root, directory, load_format="safetensors"):
    loader = default_loader.DefaultModelLoader(
        LoadConfig(load_format=load_format, use_tqdm_on_load=False)
    )
    with torch.no_grad():
        loaded = root.load_weights(
            loader.get_all_weights(
                SimpleNamespace(model=str(directory), revision=None), root
            )
        )
        root.process_weights_after_loading()
        return loaded


def _prepare_engram(session, target):
    workload = B12xWorkload(
        stage="state",
        token_counts=(8,),
        fixed_token_counts=(),
        output_dtype=torch.bfloat16,
        max_tokens=8,
        max_seqs=2,
        max_model_len=8,
    )
    providers = (target.engram_hash, *_engrams(target))
    units = [
        unit
        for provider in providers
        for unit in provider.get_b12x_preparation_units(provider, workload)
    ]
    requests = tuple(request for unit in units for request in unit.requests)
    result = session.prepare(requests, autotune=False)
    assert result is not None
    return result


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "load_format", ["safetensors", "fastsafetensors", "instanttensor"]
)
@pytest.mark.parametrize("table_memory", ["disk", "ram"])
def test_engram_descriptor_loading_preserves_ordinary_weights(
    tiny_engram_checkpoint,
    tmp_path,
    monkeypatch,
    load_format,
    table_memory,
):
    """Raw table entries never reach payload readers or placeholder parameters."""
    make_model, weights, _ = tiny_engram_checkpoint
    root, target, session = make_model(table_memory)
    real_open = weight_utils.safe_open

    @contextmanager
    def guarded_open(*args, **kwargs):
        with real_open(*args, **kwargs) as archive:

            class GuardedArchive:
                def __getattr__(self, name):
                    return getattr(archive, name)

                def get_tensor(self, name):
                    assert not root.checkpoint_file_weight_filter(name)
                    return archive.get_tensor(name)

            yield GuardedArchive()

    def forbid_bulk(*args, **kwargs):
        raise AssertionError("mixed Engram file reached bulk payload loading")

    monkeypatch.setattr(weight_utils, "safe_open", guarded_open)
    monkeypatch.setattr(default_loader, f"{load_format}_weights_iterator", forbid_bulk)
    _load_checkpoint(root, tmp_path, load_format)
    _prepare_engram(session, target)
    for layer_id, engram in zip(target.engram_layout.layer_ids, _engrams(target)):
        for name in ("q_weight", "k_weight", "wkv.weight"):
            actual = engram.get_parameter(name)
            torch.testing.assert_close(
                actual.cpu(), weights[f"layers.{layer_id}.engram.{name}"]
            )
        indices = torch.arange(24, device="cuda").expand(4, -1).contiguous()
        count = torch.tensor([3], dtype=torch.int32, device="cuda")
        if table_memory == "disk":
            # The model's hash writes the layer's IDs and live count in place.
            engram.embed_tokens.hashes[:4].copy_(indices)
            engram.embed_tokens.num_tokens.copy_(count)
            engram.stage_rows(4, torch.cuda.current_stream())
            engram.embed_tokens.gather_rows(engram.staged_rows)
        else:
            engram.embed_tokens.lookup_native(indices[:3], engram.staged_rows)
        table = weights[f"layers.{layer_id}.engram.embed.weight"].float()
        expected = table[:24] * torch.pow(2.0, torch.arange(-3, 5)).repeat_interleave(
            32
        )
        torch.testing.assert_close(
            engram.staged_rows[:3].cpu(),
            expected.reshape(1, -1).expand(3, -1).to(torch.bfloat16),
            rtol=0,
            atol=0,
        )
        assert torch.count_nonzero(engram.staged_rows[3:]) == 0
    session.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("tp_rank", [0, 3])
@torch.inference_mode()
def test_ram_engram_checkpoint_tp4_graph_reads_live_host_aliases(
    tiny_engram_checkpoint, tmp_path, tp_rank
):
    """Packed host planes survive model deletion and are read afresh on replay."""
    from cuda.bindings import runtime as cudart

    make_model, weights, paths = tiny_engram_checkpoint
    root, target, session = make_model("ram", tp_size=4, tp_rank=tp_rank)
    _load_checkpoint(root, tmp_path)
    _prepare_engram(session, target)
    embedding = _engrams(target)[0].embed_tokens
    caps = embedding.caps
    shard_rows = embedding.shard_rows
    table_rows = embedding.table_rows
    shard_start = embedding.shard_start
    shard_end = embedding.shard_end
    prefix = f"layers.{caps.layer_id}.engram.embed."
    source_weight = weights[prefix + "weight"]
    source_scale = weights[prefix + "scale"].view(torch.uint8)
    host_weight = embedding.weight_load_view
    host_scale = embedding.weight_scale_load_view
    local_rows = min(shard_rows, table_rows - shard_start)
    for actual, source in (
        (host_weight, source_weight),
        (host_scale, source_scale),
    ):
        assert actual.device.type == "cpu"
        torch.testing.assert_close(
            actual[:local_rows].view(torch.uint8),
            source[shard_start : shard_start + local_rows].view(torch.uint8),
            rtol=0,
            atol=0,
        )
        assert torch.count_nonzero(actual[local_rows:].view(torch.uint8)) == 0
    assert embedding.mapped_host_nbytes == shard_rows * (256 + 8)
    pointers = (
        (embedding.weight.data_ptr(), host_weight.data_ptr()),
        (embedding.weight_scale_inv.data_ptr(), host_scale.data_ptr()),
    )

    def assert_mapped_pointers():
        with torch.cuda.device(caps.device):
            for device_pointer, host_pointer in pointers:
                error, attributes = cudart.cudaPointerGetAttributes(device_pointer)
                assert error == cudart.cudaError_t.cudaSuccess
                assert attributes.type == cudart.cudaMemoryType.cudaMemoryTypeHost
                assert int(attributes.devicePointer) == device_pointer
                assert int(attributes.hostPointer) == host_pointer

    assert_mapped_pointers()
    ids_cpu = torch.tensor(
        [
            [
                shard_start,
                min(shard_end, table_rows) - 1,
                shard_start - 1,
                shard_end,
                table_rows,
                -1,
            ]
            * 4
        ]
        * 4,
        dtype=torch.int64,
    )
    ids = ids_cpu.to(caps.device)
    out = torch.full(
        (caps.max_tokens, 6144),
        float("nan"),
        dtype=torch.bfloat16,
        device=caps.device,
    )
    decoded = source_weight.float() * torch.pow(
        2.0, source_scale.float() - 127
    ).repeat_interleave(32, dim=1)

    def expected_output():
        expected = torch.zeros((caps.max_tokens, 24, 256), dtype=torch.bfloat16)
        owned = (
            (ids_cpu >= shard_start) & (ids_cpu < shard_end) & (ids_cpu < table_rows)
        )
        expected[: len(ids_cpu)][owned] = decoded[ids_cpu[owned]].to(torch.bfloat16)
        return expected.view(caps.max_tokens, 6144)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    resources = []
    with kernel_resolution_guard("RAM Engram graph replay"):
        try:
            with torch.cuda.stream(stream):
                embedding.lookup_native(ids, out)
                stream.synchronize()
                torch.testing.assert_close(out.cpu(), expected_output(), rtol=0, atol=0)
                with (
                    collect_cuda_graph_capture_resources() as resources,
                    session.capture(),
                    torch.cuda.graph(graph, stream=stream),
                ):
                    embedding.lookup_native(ids, out)
                # No checkpoint file or model reference is needed after capture.
                # Only the graph resource collector may retain the mapped owners.
                del embedding, target, root
                for path in paths:
                    path.unlink()
                gc.collect()
                assert_mapped_pointers()
                graph.replay()
                stream.synchronize()
                torch.testing.assert_close(out.cpu(), expected_output(), rtol=0, atol=0)

                # Finish every GPU read before mutating write-combined CPU aliases.
                # Change both planes to detect a stale device copy of either one.
                torch.cuda.synchronize(caps.device)
                host_weight[0].copy_(torch.full((256,), -4.0).to(torch.float8_e4m3fn))
                host_scale[0].view(torch.uint8).fill_(129)
                decoded[shard_start].fill_(-16.0)
                graph.replay()
                stream.synchronize()
                torch.testing.assert_close(out.cpu(), expected_output(), rtol=0, atol=0)

                ids_cpu.fill_(-1)
                ids.copy_(ids_cpu)
                graph.replay()
                stream.synchronize()
                torch.testing.assert_close(out.cpu(), expected_output(), rtol=0, atol=0)
        finally:
            stream.synchronize()
            graph.reset()
            resources.clear()
            session.close()


def _settled_reader_stats(model, timeout_s: float = 30.0) -> list[dict]:
    """Each disk table's reader counters once it has served every queued step. The
    worker updates them just after publishing a step, so a snapshot taken while the
    GPU is still on its way (or the second table is still being read) would race."""
    torch.cuda.synchronize()
    deadline = time.monotonic() + timeout_s
    while True:
        stats = [engram.embed_tokens.disk_rows.stats() for engram in _engrams(model)]
        if all(s["queued"] == 0 for s in stats) or time.monotonic() > deadline:
            return stats
        time.sleep(0.001)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_disk_engram_preparation_refreshes_graph_and_rejects_stale_rows(
    tiny_engram_checkpoint,
    tmp_path,
    monkeypatch,
):
    """Accepted GPU history, images, padding and failures cross the eager boundary."""
    make_model, weights, paths = tiny_engram_checkpoint
    root, target, session = make_model("disk")
    resident_root, resident, resident_session = make_model("device")
    _load_checkpoint(root, tmp_path)
    with torch.no_grad():
        resident_root.load_weights(weights.items())
        for engram in (*_engrams(target), *_engrams(resident)):
            engram.process_weights_after_loading()
    _prepare_engram(session, target)
    _prepare_engram(resident_session, resident)

    state = DeepseekV41ModelState.__new__(DeepseekV41ModelState)
    state.ced_state = None
    state.lookback_token_ids = torch.empty(2, 3, dtype=torch.int32, device="cuda")
    state.disk_engram_models = (target,)
    state._prefill_tokens = {}
    state._host_engram_hash = None
    monkeypatch.setattr(DefaultModelState, "prepare_inputs", lambda *args: {})
    monkeypatch.setattr(DefaultModelState, "prepare_dummy_inputs", lambda *args: {})
    batch = SimpleNamespace(
        input_ids=torch.zeros(4, dtype=torch.int32, device="cuda"),
        query_start_loc=torch.tensor([0, 2, 2], dtype=torch.int32, device="cuda"),
        idx_mapping=torch.tensor([0], dtype=torch.int32, device="cuda"),
        has_prefill=False,
    )
    req_states = SimpleNamespace(
        num_computed_tokens=SimpleNamespace(
            gpu=torch.tensor([3, 4], dtype=torch.int32, device="cuda")
        ),
        all_token_ids=SimpleNamespace(
            gpu=torch.tensor(
                [[5, 9, 13, 17, 19, 23], [4, 129264, 6, 8, 10, 12]],
                dtype=torch.int32,
                device="cuda",
            )
        ),
    )
    hidden = torch.ones(4, 4, 64, dtype=torch.bfloat16, device="cuda")

    def consume():
        mask = ~image_sentinel_mask(batch.input_ids)
        return tuple(
            engram(hidden, target.prepared_engram_hashes[:4, index], mask)
            for index, engram in enumerate(_engrams(target))
        )

    with pytest.raises(RuntimeError, match="not prepared"):
        consume()
    compiled_consume = torch.compile(consume, backend="eager", fullgraph=True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.inference_mode(), torch.cuda.stream(stream):

        def forbid_lookup(*args, **kwargs):
            raise AssertionError("capture preparation/consumer performed table lookup")

        with monkeypatch.context() as capture_guard:
            capture_guard.setattr(native, "run_lookup", forbid_lookup)
            state.prepare_dummy_inputs(2, 4)
            for _ in range(3):
                compiled_consume()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                actual = compiled_consume()
            graph.replay()
            for output in actual:
                torch.testing.assert_close(output, hidden, rtol=0, atol=0)
        for ids, request, accepted, live in (
            ([11, 12, 31, 31], 0, 3, 2),
            ([21, 22, 23, 31], 0, 4, 3),
            ([7, 129264, 9, 31], 1, 3, 3),
            ([25, 31, 31, 31], 0, 1, 1),
        ):
            batch.input_ids.copy_(torch.tensor(ids, dtype=torch.int32, device="cuda"))
            batch.idx_mapping.fill_(request)
            req_states.num_computed_tokens.gpu[request] = accepted
            batch.query_start_loc[1:].fill_(live)
            state.prepare_inputs(batch, req_states)
            # Replay immediately, before reference work can hide a late lookup.
            graph.replay()
            immediate = tuple(output.clone() for output in actual)
            # Build expected lookback independently of the serving gather kernel.
            history = torch.full((2, 3), -1, dtype=torch.int32, device="cuda")
            width = min(accepted, 3)
            history[0, 3 - width :] = req_states.all_token_ids.gpu[
                request, accepted - width : accepted
            ]
            hashes = resident.engram_hash(
                batch.input_ids,
                None,
                batch.query_start_loc,
                image_sentinel_mask(batch.input_ids),
                history,
            )
            for index, engram in enumerate(_engrams(resident)):
                engram.prepare_embeddings(hashes[:, index])
            stats = _settled_reader_stats(target)
            graph.replay()
            assert stats == _settled_reader_stats(target)
            for index, engram in enumerate(_engrams(resident)):
                expected = engram(
                    hidden, hashes[:, index], ~image_sentinel_mask(batch.input_ids)
                )
                torch.testing.assert_close(immediate[index], expected, rtol=0, atol=0)
                torch.testing.assert_close(actual[index], expected, rtol=0, atol=0)
                torch.testing.assert_close(
                    actual[index][live:], hidden[live:], rtol=0, atol=0
                )
            torch.testing.assert_close(
                actual[0][batch.input_ids == 129264], hidden[batch.input_ids == 129264]
            )

        # A smaller host preparation must retire previously populated rows,
        # even though this existing graph still consumes its four-row buffers.
        batch.input_ids.copy_(
            torch.tensor([11, 12, 13, 14], dtype=torch.int32, device="cuda")
        )
        batch.query_start_loc[1:].fill_(4)
        state.prepare_inputs(batch, req_states)
        graph.replay()
        first_rows = [output[:1].clone() for output in actual]
        small_batch = SimpleNamespace(**vars(batch))
        small_batch.input_ids = batch.input_ids[:1]
        batch.query_start_loc[1:].fill_(1)
        state.prepare_inputs(small_batch, req_states)
        graph.replay()
        for index, engram in enumerate(_engrams(target)):
            assert engram.staged_rows[1:].eq(0).all()
            torch.testing.assert_close(
                actual[index][:1], first_rows[index], rtol=0, atol=0
            )
            torch.testing.assert_close(actual[index][1:], hidden[1:], rtol=0, atol=0)

        # A failed read is recorded and never published: the next step's staging
        # names it, and the forward refuses the layer. (Replaying the graph now
        # would wait for rows that never arrive until the GPU's bound stops it.)
        with open(paths[1], "r+b") as file:
            file.truncate(1)
        state.prepare_inputs(batch, req_states)
        failing = _engrams(target)[1].embed_tokens.disk_rows
        deadline = time.monotonic() + 30
        while failing.failure() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert "short Engram row read" in failing.failure()
        with pytest.raises(RuntimeError, match="short Engram row read"):
            state.prepare_inputs(batch, req_states)
        with pytest.raises(RuntimeError, match="not prepared"):
            consume()
    torch.cuda.current_stream().wait_stream(stream)
    session.close()
    resident_session.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_disk_engram_rows_the_host_hashed_equal_the_gpu_rows(
    tiny_engram_checkpoint, tmp_path, monkeypatch
):
    """The host hashes a step's prefill chunks and the reader reads their rows before
    the GPU's hash; a decode row in the same step comes from the GPU's hash. With the
    debug check on, every early ID equals the GPU's, and the rows equal the resident
    table's: a first chunk, a later chunk with history, a mixed step (decode first,
    an image token in its history) and a decode-only step."""
    monkeypatch.setattr(engram_module, "ENGRAM_CHECK_EARLY_IDS", True)
    make_model, weights, paths = tiny_engram_checkpoint
    root, target, session = make_model("disk")
    resident_root, resident, resident_session = make_model("device")
    _load_checkpoint(root, tmp_path)
    with torch.no_grad():
        resident_root.load_weights(weights.items())
        for engram in (*_engrams(target), *_engrams(resident)):
            engram.process_weights_after_loading()
    _prepare_engram(session, target)
    _prepare_engram(resident_session, resident)
    for engram in _engrams(target):
        assert engram.embed_tokens.disk_rows.check_early

    prompt = [5, 9, 13, 17, 19, 23]
    state = DeepseekV41ModelState.__new__(DeepseekV41ModelState)
    state.ced_state = None
    state.lookback_token_ids = torch.empty(2, 3, dtype=torch.int32, device="cuda")
    state.disk_engram_models = (target,)
    state._prefill_tokens = {"a": np.array(prompt, dtype=np.int32)}
    state._host_engram_hash = None
    monkeypatch.setattr(DefaultModelState, "prepare_inputs", lambda *args: {})
    all_token_ids = torch.tensor(
        [prompt, [4, 129264, 6, 8, 10, 12]], dtype=torch.int32, device="cuda"
    )
    slots = {"a": 0, "b": 1}
    hidden = torch.ones(4, 4, 64, dtype=torch.bfloat16, device="cuda")
    # (input ids, [(request, computed tokens, rows, prefilling)], early lookups)
    steps = (
        ([13, 17, 19, 31], [("a", 2, 3, True)], 4 * 24),
        ([23, 31, 31, 31], [("a", 5, 1, True)], 4 * 24),
        ([10, 13, 17, 31], [("b", 4, 1, False), ("a", 2, 2, True)], 3 * 24),
        ([10, 31, 31, 31], [("b", 4, 1, False)], 0),
    )
    with torch.inference_mode():
        for ids, requests, early in steps:
            names = [name for name, *_ in requests]
            computed = torch.zeros(2, dtype=torch.int32, device="cuda")
            for name, done, _, _ in requests:
                computed[slots[name]] = done
            rows = np.array([n for _, _, n, _ in requests], dtype=np.int32)
            starts = np.zeros(len(requests) + 1, dtype=np.int32)
            np.cumsum(rows, out=starts[1:])
            loc = (
                np.concatenate([starts, [starts[-1]]]) if len(requests) == 1 else starts
            )
            prefilling = np.array([p for *_, p in requests], dtype=np.bool_)
            batch = SimpleNamespace(
                input_ids=torch.tensor(ids, dtype=torch.int32, device="cuda"),
                query_start_loc=torch.tensor(loc, dtype=torch.int32, device="cuda"),
                idx_mapping=torch.tensor(
                    [slots[n] for n in names], dtype=torch.int32, device="cuda"
                ),
                req_ids=names,
                query_start_loc_np=loc,
                num_scheduled_tokens=rows,
                num_computed_prefill_tokens_np=np.array(
                    [d for _, d, _, _ in requests], dtype=np.int32
                ),
                is_prefilling_np=prefilling,
                has_prefill=bool(prefilling.any()),
                num_draft_tokens_per_req=None,
            )
            req_states = SimpleNamespace(
                num_computed_tokens=SimpleNamespace(gpu=computed),
                all_token_ids=SimpleNamespace(gpu=all_token_ids),
            )
            before = [s["early_lookups"] for s in _settled_reader_stats(target)]
            state.prepare_inputs(batch, req_states)
            mask = ~image_sentinel_mask(batch.input_ids)
            actual = tuple(
                engram(hidden, target.prepared_engram_hashes[:4, index], mask)
                for index, engram in enumerate(_engrams(target))
            )
            stats = _settled_reader_stats(target)
            for engram in _engrams(target):
                assert engram.embed_tokens.disk_rows.failure() is None
            assert [s["early_lookups"] - b for s, b in zip(stats, before)] == [
                early
            ] * 2

            history = torch.full((2, 3), -1, dtype=torch.int32, device="cuda")
            for row, (name, done, _, _) in enumerate(requests):
                width = min(done, 3)
                history[row, 3 - width :] = all_token_ids[
                    slots[name], done - width : done
                ]
            hashes = resident.engram_hash(
                batch.input_ids,
                None,
                batch.query_start_loc,
                image_sentinel_mask(batch.input_ids),
                history,
            )
            for index, engram in enumerate(_engrams(resident)):
                engram.prepare_embeddings(hashes[:, index])
                expected = engram(hidden, hashes[:, index], mask)
                torch.testing.assert_close(actual[index], expected, rtol=0, atol=0)
    session.close()
    resident_session.close()


def test_dspark_checkpoint_filter_does_not_read_target_engram(tmp_path, monkeypatch):
    target = "layers.1.engram.embed.weight"
    draft = "mtp.0.main_proj.weight"
    target_path, draft_path = (
        tmp_path / "target.safetensors",
        tmp_path / "draft.safetensors",
    )
    save_file({target: torch.zeros(4, 8)}, str(target_path))
    save_file({draft: torch.ones(2, 2), target: torch.zeros(4, 8)}, str(draft_path))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {target: target_path.name, draft: draft_path.name}})
    )
    real_open = weight_utils.safe_open

    @contextmanager
    def guarded_open(path, **kwargs):
        assert str(path) != str(target_path)
        with real_open(path, **kwargs) as archive:

            class GuardedArchive:
                def __getattr__(self, name):
                    return getattr(archive, name)

                def get_tensor(self, name):
                    assert name != target
                    return archive.get_tensor(name)

            yield GuardedArchive()

    monkeypatch.setattr(weight_utils, "safe_open", guarded_open)
    loader = default_loader.DefaultModelLoader(LoadConfig(load_format="safetensors"))
    loaded = dict(
        loader.get_all_weights(
            SimpleNamespace(model=str(tmp_path), revision=None),
            SimpleNamespace(
                checkpoint_weight_name_prefixes=DSparkDeepseekV4ForCausalLM.checkpoint_weight_name_prefixes
            ),
        )
    )
    assert loaded.keys() == {draft}
    torch.testing.assert_close(loaded[draft], torch.ones(2, 2))


@pytest.mark.parametrize("value", [0, 32])
def test_engram_rejects_removed_prefetch_option(value):
    with pytest.raises(ValueError, match="disk_prefetch_max_tokens"):
        EngramConfig(table_memory="disk", disk_prefetch_max_tokens=value)


def test_engram_rejects_removed_resident_scales():
    EngramConfig(table_memory="disk", disk_resident_scales=False)
    with pytest.raises(ValueError, match="disk_resident_scales was removed"):
        EngramConfig(table_memory="disk", disk_resident_scales=True)


@pytest.mark.parametrize("cpu_offload", [False, True])
def test_engram_storage_selection_changes_graph_configuration(cpu_offload):
    hashes = {
        EngramConfig(table_memory=memory, cpu_offload=cpu_offload).compute_hash()
        for memory in ("device", "ram", "disk")
    }
    assert len(hashes) == 3
    with pytest.raises(ValueError):
        EngramConfig(table_memory="invalid")
    disk_hashes = {
        EngramConfig(table_memory="disk", **options).compute_hash()
        for options in (
            {},
            {"projection_tp": True},
        )
    }
    assert len(disk_hashes) == 2


def test_ram_engram_budget_reserves_memory_and_survives_worker_serialization(
    monkeypatch,
):
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    model_config = SimpleNamespace(
        architecture="DeepseekV41ForCausalLM",
        hf_text_config=SimpleNamespace(
            engram_layer_ids=(1, 14), engram_num_embeddings=(17, 31)
        ),
    )
    # TP4 rounds the two global tables to 20 and 32 packed 264-byte rows.
    available = (4 << 30) + (20 + 32) * 264 - 1
    monkeypatch.setattr(weight_utils, "_get_available_ram_bytes", lambda: available)
    config = EngramConfig(table_memory="ram")
    graph_hash = config.compute_hash()
    with pytest.raises(ValueError, match="Insufficient RAM"):
        config.verify_model_config(model_config, tp_size=4)

    available += 1
    config.verify_model_config(model_config, tp_size=4)
    assert config.compute_hash() == graph_hash

    # Workers see memory after their peers allocate. Revalidating the same
    # serialized preflight must not charge all tables against remaining RAM.
    worker_config = pickle.loads(pickle.dumps(config))
    available = 0
    worker_config.verify_model_config(model_config, tp_size=4)
    assert worker_config.compute_hash() == graph_hash
    with pytest.raises(ValueError, match="Insufficient RAM"):
        EngramConfig(table_memory="ram").verify_model_config(model_config, tp_size=4)
    # A different padded footprint is not covered by the original preflight.
    with pytest.raises(ValueError, match="Insufficient RAM"):
        worker_config.verify_model_config(model_config, tp_size=8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_engram_preparation_covers_masked_and_unmasked_graphs(dist_init):
    device = torch.device("cuda", torch.cuda.current_device())
    capacity, hidden, streams = 7, 64, 4
    geometry = native.build_geometry(base_table_size=19, compressed_vocab_size=32)
    caps = native.Caps(
        device=device,
        max_tokens=capacity,
        max_seqs=2,
        max_requests=2,
        vocab_size=32,
        layer_id=1,
        tp_size=1,
        tp_rank=0,
    )
    lookup = native.plan(
        caps,
        token_map=list(range(32)),
        geometry=geometry,
        invocation={"operation": "lookup", "compact_rows": False},
    )
    layout = SimpleNamespace(
        caps=(caps,),
        lookup_plans=(lookup,),
        geometry=geometry,
        table_memory="device",
        projection_tp=False,
    )
    config = SimpleNamespace(hidden_size=hidden, hc_mult=streams, rms_norm_eps=1e-3)
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with torch.device(device):
            module = Engram(config, None, layout, 0, False, "engram_graph_test")
    finally:
        torch.set_default_dtype(old_dtype)
    module.q_weight.fill_(0.5)
    module.k_weight.fill_(0.25)
    module.wkv.weight.normal_(std=0.01)
    module.embed_tokens.weight.fill_(2)
    module.embed_tokens.weight_scale_inv.fill_(127)
    module.process_weights_after_loading()
    workload = B12xWorkload(
        stage="weights",
        token_counts=(1, capacity),
        fixed_token_counts=(1,),
        output_dtype=torch.bfloat16,
        max_tokens=capacity,
        max_seqs=2,
        max_model_len=8,
    )
    requests = tuple(
        request
        for unit in module.get_b12x_preparation_units(module, workload)
        for request in unit.requests
    )
    ids = torch.arange(24, device=device).expand(capacity, -1).contiguous()
    residual = torch.randn(
        (capacity, streams, hidden),
        device=device,
        dtype=torch.bfloat16,
    ).mul_(0.1)
    mask = torch.ones(capacity, dtype=torch.bool, device=device)
    mask[::2] = False

    def reference(rows, token_mask):
        source = residual[:rows].float()
        projected = module.wkv(module.staged_rows[:rows]).float()
        key = projected[:, : streams * hidden].view(rows, streams, hidden)
        value = projected[:, streams * hidden :]
        score = (source * module.norm_weights.view(streams, hidden) * key).sum(-1)
        score *= torch.rsqrt(source.square().mean(-1) + config.rms_norm_eps)
        score *= torch.rsqrt(key.square().mean(-1) + config.rms_norm_eps)
        score *= hidden**-0.5
        gate = torch.sigmoid(torch.copysign(score.abs().clamp_min(1e-6).sqrt(), score))
        if token_mask is not None:
            gate.masked_fill_(~token_mask[:, None], 0)
        return (source + gate[..., None] * value[:, None]).bfloat16()

    with PreparationSession(
        device=device, autotune=False, compile_workers=2
    ) as session:
        session.prepare(requests)
        session.freeze()
        module.prepare_embeddings(ids)
        launch = torch.compile(module.forward, fullgraph=True)
        for rows in (1, capacity):
            for with_mask in (False, True):
                token_mask = mask[:rows] if with_mask else None
                launch(residual[:rows], ids[:rows], token_mask)
                graph = torch.cuda.CUDAGraph()
                try:
                    with session.capture(), torch.cuda.graph(graph):
                        actual = launch(residual[:rows], ids[:rows], token_mask)
                    pointer = actual.data_ptr()
                    allocated = torch.cuda.memory_allocated(device)
                    residual.mul_(-0.5)
                    module.staged_rows.mul_(0.75)
                    mask.logical_not_()
                    actual.fill_(float("nan"))
                    graph.replay()
                    torch.cuda.synchronize(device)
                    assert torch.cuda.memory_allocated(device) == allocated
                    assert actual.data_ptr() == pointer
                    assert (
                        torch.isfinite(actual).all() and torch.count_nonzero(actual) > 0
                    )
                    torch.testing.assert_close(
                        actual,
                        reference(rows, token_mask),
                        rtol=1e-2,
                        atol=1e-2,
                    )
                    if token_mask is not None:
                        torch.testing.assert_close(
                            actual[~token_mask],
                            residual[:rows][~token_mask],
                            rtol=0,
                            atol=0,
                        )
                finally:
                    graph.reset()


@pytest.mark.parametrize("num_layers", [1, 3])
@pytest.mark.parametrize("num_tokens", [1, 13])
def test_engram_hash_first_forward_uses_declared_geometry(
    monkeypatch, num_layers, num_tokens
):
    """Lazy native bindings must not make the first output have zero layers."""
    module = NgramHashState.__new__(NgramHashState)
    nn.Module.__init__(module)
    module.layout = SimpleNamespace(hash_plans=(None,) * num_layers)
    module.bindings = []

    def run_native(ids, mask, starts, history, out):
        if not module.bindings:
            module.bindings = list(range(num_layers))
        for layer in module.bindings:
            out[:, layer].copy_(ids[:, None].expand(-1, 24) + layer)

    monkeypatch.setattr(module, "run_native", run_native)
    ids = torch.arange(num_tokens, dtype=torch.int64)
    starts = torch.tensor([0, num_tokens], dtype=torch.int32)
    dead_mask = torch.zeros(num_tokens, dtype=torch.bool)
    history = torch.full((1, 3), -1, dtype=torch.int64)
    for offset in (0, 7):
        actual = module(ids + offset, ids, starts, dead_mask, history)
        expected = (
            ids[:, None, None] + offset + torch.arange(num_layers)[None, :, None]
        ).expand(-1, -1, 24)
        assert actual.shape == (num_tokens, num_layers, 24)
        assert actual.dtype == torch.int64 and actual.device == ids.device
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_disk_engram_model_allocates_hash_buffer_from_declared_caps(monkeypatch):
    from vllm.models.deepseek_v4_1.nvidia import model as model_module

    caps = SimpleNamespace(max_tokens=13, device=torch.device("cpu"))
    layout = SimpleNamespace(caps=(caps,), layer_ids=(1, 14), table_memory="disk")
    config = SimpleNamespace(
        vocab_size=32,
        hidden_size=64,
        hc_mult=4,
        hc_eps=1e-6,
        rms_norm_eps=1e-6,
        index_topk=8,
        num_hidden_layers=0,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=config,
            dtype=torch.bfloat16,
            # The sink detector's settings follow the served architecture.
            architecture="DeepseekV41ForCausalLM",
        ),
        quant_config=SimpleNamespace(get_name=lambda: "deepseek_v41_fp8"),
        parallel_config=SimpleNamespace(use_ubatching=False, tensor_parallel_size=1),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=13),
        lora_config=None,
        speculative_config=None,
    )
    monkeypatch.setattr(model_module, "_use_sequence_parallel", lambda _: False)
    monkeypatch.setattr(
        model_module,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=False, is_last_rank=False),
    )
    monkeypatch.setattr(model_module.EngramLayout, "from_config", lambda _: layout)
    monkeypatch.setattr(
        model_module,
        "make_layers",
        lambda *args, **kwargs: (0, 0, nn.ModuleList()),
    )
    model = model_module.DeepseekV4Model(vllm_config=vllm_config)
    hashes = model.prepared_engram_hashes
    assert hashes.shape == (13, 2, 24)
    assert hashes.dtype == torch.int64 and hashes.device == caps.device
    # Layer-major: each layer's plane is a contiguous [max_tokens, 24].
    assert hashes.permute(1, 0, 2).is_contiguous()
    assert (hashes == -1).all()
    assert "prepared_engram_hashes" not in model.state_dict()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("tp_rank", [0, 3])
@torch.inference_mode()
def test_disk_rows_read_whole_table_sources(tmp_path, dist_init, tp_rank):
    """Each rank reads its shard (its hash columns) of the global tables from unaligned
    file offsets, and other ranks' columns gather as zeros; disk tables prepare no B12X
    lookup."""
    device = torch.device("cuda", torch.cuda.current_device())
    capacity = 7
    geometry = native.build_geometry(base_table_size=19, compressed_vocab_size=32)
    caps = native.Caps(
        device=device,
        max_tokens=capacity,
        max_seqs=2,
        max_requests=2,
        vocab_size=32,
        layer_id=1,
        tp_size=4,
        tp_rank=tp_rank,
    )
    layout = SimpleNamespace(
        caps=(caps,),
        lookup_plans=(None,),
        geometry=geometry,
        table_memory="disk",
        projection_tp=False,
    )
    config = SimpleNamespace(hidden_size=64, hc_mult=4, rms_norm_eps=1e-3)
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with torch.device(device):
            module = Engram(config, None, layout, 0, False, "engram_disk_test")
    finally:
        torch.set_default_dtype(old_dtype)
    embed = module.embed_tokens
    row_values = torch.arange(embed.table_rows).remainder(7).add(1)
    weight = row_values[:, None].expand(-1, 256).to(torch.float8_e4m3fn).contiguous()
    scales = torch.full((embed.table_rows, 8), 127, dtype=torch.uint8)
    for name, tensor, offset, scale in (
        ("weight.bin", weight, 4093, False),
        ("scale.bin", scales, 123, True),
    ):
        path = tmp_path / name
        with path.open("wb") as stream:
            stream.write(bytes(offset))
            stream.write(tensor.view(torch.uint8).numpy().tobytes())
        embed._disk_sources.append((str(path), offset, scale))
    embed.open_disk_rows()
    workload = B12xWorkload(
        stage="weights",
        token_counts=(1, capacity),
        fixed_token_counts=(1,),
        output_dtype=torch.bfloat16,
        max_tokens=capacity,
        max_seqs=2,
        max_model_len=8,
    )
    units = module.get_b12x_preparation_units(module, workload)
    assert all(unit.name != "EngramLookup" for unit in units)
    ids = torch.arange(24, device=device) + embed.shard_start + 1
    ids = ids.expand(capacity, -1).contiguous()
    count = torch.tensor([capacity], device=device, dtype=torch.int32)
    try:
        output = module.staged_rows
        address = output.data_ptr()
        for shift in (0, 1):
            embed.hashes[:capacity].copy_(ids + shift)
            embed.num_tokens.copy_(count)
            module.stage_rows(capacity, torch.cuda.current_stream())
            embed.gather_rows(output)
            # The rank's own columns, side by side.
            owned = [h for h in range(24) if embed.column_mask >> h & 1]
            assert len(owned) == 24 // caps.tp_size
            reference = weight.float()[ids.cpu()[:, owned] + shift].flatten(1)
            torch.testing.assert_close(
                output.cpu(), reference.bfloat16(), rtol=0, atol=0
            )
            assert output.data_ptr() == address
    finally:
        embed.close()


@pytest.mark.parametrize("tp_size", [1, 2, 3, 4, 6])
def test_disk_shards_interleave_hash_columns(tp_size):
    """When the TP size divides the 24 hash columns, rank r reads every tp-th column
    from r: the ranks' columns tile all 24 once, and each rank holds heads of every
    n-gram order (heads 0-7 2-grams, 8-15 3-grams, 16-23 4-grams) as evenly as the
    split allows, so no rank reads only the longest n-grams' distinct rows."""
    from vllm.models.deepseek_v4_1.common.engram import column_shard

    geometry = native.build_geometry(base_table_size=19, compressed_vocab_size=32)
    for index in range(len(geometry.layer_ids)):
        shards = [column_shard(geometry, index, tp_size, r) for r in range(tp_size)]
        assert sorted(c for shard in shards for c in shard) == list(range(24))
        for rank, shard in enumerate(shards):
            assert shard == tuple(range(rank, 24, tp_size))
            orders = [sum(1 for c in shard if c // 8 == order) for order in range(3)]
            assert max(orders) - min(orders) <= 1, (tp_size, rank, orders)
    assert column_shard(geometry, 0, 5, 0) is None  # 24 columns do not split five ways


@pytest.mark.parametrize("tp_size", [1, 2, 3, 4, 6])
def test_interleaved_rows_restore_column_order(tp_size, monkeypatch):
    """Every rank's columns, all-gathered as 256-wide rows along the last dimension,
    are the column order exactly, with no copy after the collective."""
    import vllm.models.deepseek_v4_1.common.engram as engram_module

    tokens = 5
    full = torch.randn(tokens, 24 * 256).bfloat16()
    locals_ = [
        full.view(tokens, 24 // tp_size, tp_size, 256)[:, :, r, :].reshape(tokens, -1)
        for r in range(tp_size)
    ]
    gathered = []

    def all_gather(local, dim):
        assert local.shape == (tokens * 24 // tp_size, 256) and dim == -1
        gathered.append(torch.cat([part.view(-1, 256) for part in locals_], dim=dim))
        return gathered[-1]

    monkeypatch.setattr(engram_module, "tensor_model_parallel_all_gather", all_gather)
    rows = engram_module.interleaved_rows(locals_[0])
    assert torch.equal(rows.view(torch.int16), full.view(torch.int16))
    assert rows.data_ptr() == gathered[0].data_ptr()
