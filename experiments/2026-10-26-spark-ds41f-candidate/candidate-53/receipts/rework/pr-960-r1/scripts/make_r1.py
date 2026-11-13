#!/usr/bin/env python3
"""make_r1.py TREE: in a vLLM checkout at original #960 (0781 applied on candidate-53's base),
rewrite the working tree to #960 round 1:
1. GPUModelRunner.execute_model's FULL attention-metadata fast path tests for an active KV
   connector as this base defines one (``self.kv_connector is NO_OP_KV_CONNECTOR``, the
   instance get_kv_connector returns without a KV transfer group) instead of calling
   ``KVConnector.has_connector()``, which this base does not define;
2. a regression test drives GPUModelRunner.execute_model through that condition with the
   connector get_kv_connector builds (NO_OP_KV_CONNECTOR, and an ActiveKVConnector).
Exact-text edits; every edit must match once. Other files keep HEAD's content."""
import subprocess
import sys
from pathlib import Path

tree = Path(sys.argv[1])


def show(rev, path):
    return subprocess.run(["git", "-C", str(tree), "show", f"{rev}:{path}"], check=True,
                          capture_output=True, text=True).stdout


def sub(text, old, new, path):
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{path}: expected one match, found {n}: {old[:80]!r}")
    return text.replace(old, new)


# 1. The condition.
p = "vllm/v1/worker/gpu/model_runner.py"
t = show("HEAD", p)
t = sub(
    t,
    "                and not self.kv_connector.has_connector()\n",
    "                # An active KV connector's pre_forward reads the step's metadata.\n"
    "                and self.kv_connector is NO_OP_KV_CONNECTOR\n",
    p,
)
(tree / p).write_text(t)

# 2. The regression test, after the other execute_model test (the boundary logits-only step).
p = "tests/v1/worker/test_gpu_model_runner_v2.py"
t = show("HEAD", p)
t = sub(
    t,
    "import contextlib\nfrom types import MethodType, SimpleNamespace\nfrom weakref import ref\n",
    "import contextlib\nfrom types import MethodType, SimpleNamespace\nfrom unittest.mock import Mock\n"
    "from weakref import ref\n",
    p,
)
t = sub(
    t,
    "import vllm.v1.worker.gpu.model_runner as model_runner_module\n"
    "from vllm.forward_context import MoEPrefillMetadata\n",
    "import vllm.v1.worker.gpu.kv_connector as kv_connector_module\n"
    "import vllm.v1.worker.gpu.model_runner as model_runner_module\n"
    "from vllm.config.compilation import CUDAGraphMode\n"
    "from vllm.forward_context import MoEPrefillMetadata\n",
    p,
)
t = sub(
    t,
    "from vllm.v1.worker.gpu.block_table import BlockTables\n"
    "from vllm.v1.worker.gpu.model_runner import GPUModelRunner\n",
    "from vllm.v1.worker.gpu.block_table import BlockTables\n"
    "from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector\n"
    "from vllm.v1.worker.gpu.model_runner import GPUModelRunner\n",
    p,
)
TEST = '''

class _StopAfterAttention(Exception):
    pass


@pytest.mark.parametrize("has_kv_transfer_group", [False, True])
def test_full_cudagraph_decode_attn_prep_follows_kv_connector(
    monkeypatch: pytest.MonkeyPatch, has_kv_transfer_group: bool
):
    """A runtime FULL decode step whose graph refreshes its attention metadata (the
    warm-up decode step is the first) only stages CED's floors when no KV connector
    is active, and builds the metadata eagerly when one is: the connector's
    pre_forward reads it. The connector is the one get_kv_connector returns."""
    monkeypatch.setattr(
        kv_connector_module, "has_kv_transfer_group", lambda: has_kv_transfer_group
    )
    monkeypatch.setattr(kv_connector_module, "get_kv_transfer_group", Mock)
    connector = model_runner_module.get_kv_connector(SimpleNamespace(), {})
    if has_kv_transfer_group:
        assert isinstance(connector, ActiveKVConnector)
    else:
        assert connector is model_runner_module.NO_OP_KV_CONNECTOR

    events: list[tuple] = []
    input_batch = SimpleNamespace(
        num_reqs=1,
        num_reqs_after_padding=1,
        seq_lens=torch.ones(1, dtype=torch.int32),
        input_ids=torch.zeros(6, dtype=torch.int32),
        positions=torch.zeros(6, dtype=torch.int64),
    )
    block_tables, slot_mappings = object(), object()

    def stop_before_model_inputs(*args):
        raise _StopAfterAttention

    model_state = SimpleNamespace(
        preprocess_state=lambda *args: None,
        stage_full_cudagraph_attn=lambda batch: events.append(("stage", batch)),
        prepare_attn=lambda *args, **kwargs: events.append(
            ("prepare_attn", args, kwargs)
        ),
        prepare_inputs=stop_before_model_inputs,
    )
    scheduler_output = SimpleNamespace(
        boundary_logits_only=False,
        scheduled_new_reqs=[],
        total_num_scheduled_tokens=6,
        num_scheduled_tokens={"decode-request": 6},
    )
    runner = SimpleNamespace(
        **{
            name: lambda *args: None
            for name in (
                "update_pp_decode_requests",
                "finish_requests",
                "free_states",
                "add_requests",
                "update_requests",
            )
        },
        block_tables=SimpleNamespace(apply_staged_writes=lambda: None),
        boundary_checkpoint_state=None,
        kv_connector=connector,
        # DSpark and DFlash build their own draft attention metadata.
        speculator=SimpleNamespace(uses_target_attn_metadata=False),
        model_state=model_state,
        lora_config=None,
        is_encoder_decoder=False,
        dp_size=1,
        dp_rank=0,
        dcp_size=1,
        dcp_rank=0,
        cp_interleave=1,
        pcp_manager=None,
        ubatch_runner=None,
        parallel_config=SimpleNamespace(),
        observability_config=SimpleNamespace(cudagraph_metrics=False),
        uniform_decode_query_len=6,
        cudagraph_manager=SimpleNamespace(captures_full_attn_metadata=True),
        gather_batch_req_state=lambda *args: (SimpleNamespace(num_tokens=6), 6),
        prepare_inputs=lambda *args: input_batch,
        prepare_attn=lambda batch: (block_tables, slot_mappings),
        draft_tokens_handler=SimpleNamespace(set_consumed_draft_tokens=lambda *a: None),
        req_states=SimpleNamespace(
            draft_tokens=None, num_computed_tokens=SimpleNamespace(gpu=None)
        ),
        input_buffers=SimpleNamespace(dcp_local_seq_lens=None),
        kv_cache_config=SimpleNamespace(),
        attn_groups=[[]],
        uses_inputs_embeds=False,
    )
    batch_desc = SimpleNamespace(
        num_tokens=6, cg_mode=CUDAGraphMode.FULL, num_ubatches=1
    )
    monkeypatch.setattr(
        model_runner_module,
        "dispatch_cg_and_sync_dp",
        lambda *args, **kwargs: (batch_desc, None),
    )
    monkeypatch.setattr(
        model_runner_module,
        "build_slot_mappings_by_layer",
        lambda slots, config: events.append(("slot_mappings_by_layer", slots)) or {},
    )

    with pytest.raises(_StopAfterAttention):
        GPUModelRunner.execute_model(runner, scheduler_output)

    if has_kv_transfer_group:
        assert events == [
            ("slot_mappings_by_layer", slot_mappings),
            (
                "prepare_attn",
                (
                    input_batch,
                    CUDAGraphMode.FULL,
                    block_tables,
                    slot_mappings,
                    runner.attn_groups,
                    runner.kv_cache_config,
                ),
                {"for_capture": False},
            ),
        ]
    else:
        assert events == [("stage", input_batch)]
'''
t = sub(
    t,
    "    assert input_batch.positions.item() == 3\n\n\n"
    '@pytest.mark.parametrize("failure_stage", [None, "init", "prepare", "forward"])\n',
    "    assert input_batch.positions.item() == 3\n" + TEST + "\n\n"
    '@pytest.mark.parametrize("failure_stage", [None, "init", "prepare", "forward"])\n',
    p,
)
(tree / p).write_text(t)
print("ok")
