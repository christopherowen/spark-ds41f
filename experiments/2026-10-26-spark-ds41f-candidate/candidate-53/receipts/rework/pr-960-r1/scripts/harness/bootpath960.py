#!/usr/bin/env python3
"""bootpath960.py --label L --out FILE: the TP3 boot's failing segment on one GPU (#960 round 1).

The warm-up decode step (warmup._run_decode_step -> gpu_worker.execute_model ->
GPUModelRunner.execute_model) failed on every rank at execute_model's FULL attention-metadata
condition. This harness runs the image's real GPUModelRunner.execute_model, unmodified, from its
first line through that condition and the branch it selects, up to model-input preparation, for
a step shaped like the warm-up decode step (every request decoding 1 + 5 draft tokens after its
prompt, FULL CUDA graph), with the serving objects the boot used:

- the KV connector get_kv_connector returns in a process with no KV transfer group (the real
  has_kv_transfer_group(), nothing patched): NO_OP_KV_CONNECTOR, as on sp5..sp7;
- an ActiveKVConnector, from the same factory with a transfer group stubbed (Mock backend);
- captures_full_attn_metadata from the real ModelCudaGraphManager.can_capture_full_attn_metadata
  on the real DeepseekV41ModelState and 23 real DeepseekV41B12xMetadataBuilder groups (the value
  capture() sets);
- the DSpark speculator class (its uses_target_attn_metadata);
- the real DeepseekV41ModelState.stage_full_cudagraph_attn with a real CEDState on the GPU, and a
  real InputBatch (InputBatch.make_dummy on real InputBuffers, then filled as a decode step).

The runner object is a namespace holding those (as tests/v1/worker/test_gpu_model_runner_v2.py
builds runners); the request-state updates before the step, prepare_inputs (returns the batch)
and dispatch_cg_and_sync_dp (returns the FULL descriptor) are stubs. model_state.prepare_inputs
stops the step (the next call after the condition's branch). With NO_OP_KV_CONNECTOR the
fast path must run (CED floors staged, no eager metadata, the batch marked staged so
prepare_inputs does not restage it), and a FULL graph holding the step's in-graph CED refresh
must equal an eager CEDState.stage of the same batch. With an ActiveKVConnector the eager
metadata path must run. Exit 0 when every check holds."""
import argparse
import json
import sys
import traceback
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch

import vllm.v1.worker.gpu.kv_connector as kvc
import vllm.v1.worker.gpu.model_runner as mr
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.kv_transfer import has_kv_transfer_group
from vllm.models.deepseek_v4_1.ced import CEDState, replay_rows
from vllm.models.deepseek_v4_1.nvidia.model_state import DeepseekV41ModelState
from vllm.models.deepseek_v4_1.sparse_mla import DeepseekV41B12xMetadataBuilder
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import CircularBufferSpec, FullAttentionSpec
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    ModelCudaGraphManager,
)
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
from vllm.v1.worker.utils import AttentionGroup
from vllm.v1.worker.workspace import current_workspace_manager, init_workspace_manager

p = argparse.ArgumentParser()
p.add_argument("--label", required=True)
p.add_argument("--out", required=True)
a = p.parse_args()
dev = torch.device("cuda:0")
try:
    current_workspace_manager()
except Exception:
    init_workspace_manager(dev)
fh = open(a.out, "w")
failures: list[str] = []


def emit(**r):
    r = {"label": a.label, **r}
    fh.write(json.dumps(r, default=str) + "\n")
    fh.flush()
    print(json.dumps(r, default=str), flush=True)


def check(name, ok, **detail):
    emit(kind="check", check=name, ok=bool(ok), **detail)
    if not ok:
        failures.append(name)


MAX_REQS, MAX_TOKENS = 16, 2048
NUM_REQS, PROMPT_LEN, QUERY_LEN = 3, 64, 6  # 1 + 5 draft tokens, as the recipe's DSpark

# 1. The serving connector, from the real factory with no KV transfer group.
emit(kind="identity", gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
     model_runner=mr.__file__, kv_connector=kvc.__file__)
no_op = kvc.get_kv_connector(SimpleNamespace(), {})
check("serving connector is NO_OP_KV_CONNECTOR (no KV transfer group)",
      (not has_kv_transfer_group()) and no_op is kvc.NO_OP_KV_CONNECTOR,
      type=type(no_op).__name__)
emit(kind="api", KVConnector_has_connector=hasattr(kvc.KVConnector, "has_connector"),
     ActiveKVConnector_has_connector=hasattr(kvc.ActiveKVConnector, "has_connector"),
     DSparkSpeculator_uses_target_attn_metadata=getattr(
         DSparkSpeculator, "uses_target_attn_metadata", "absent"))

# 2. The model state, attention groups and graph manager the boot captured with.
class SC: max_num_batched_tokens = MAX_TOKENS; max_num_seqs = MAX_REQS
class Cfg:
    scheduler_config = SC()
    speculative_config = type("S", (), {"num_speculative_tokens": 5, "num_verification_tokens": 5,
                              "parallel_drafting": False, "use_eagle": lambda self=None: True})()
    cache_config = type("C", (), {"block_size": 64})()
    model_config = type("M", (), {"max_model_len": 4096})()
    parallel_config = type("P", (), {"decode_context_parallel_size": 1})()
    compilation_config = type("Comp", (), {"cudagraph_mode": None})()
attn_groups = []
for i in range(23):
    if i % 3 == 0:
        spec = object.__new__(CircularBufferSpec); tps = 1
    else:
        spec = object.__new__(FullAttentionSpec); tps = 1 if i % 3 == 1 else 2
    object.__setattr__(spec, "block_size", 64); object.__setattr__(spec, "tokens_per_state", tps)
    names = [f"model.layers.{i * 3}.self_attn.attn", f"model.layers.{i * 3 + 1}.self_attn.attn"]
    builder = DeepseekV41B12xMetadataBuilder(spec, names, Cfg(), dev)
    attn_groups.append([AttentionGroup(backend=None, layer_names=names, kv_cache_spec=spec,
                                       kv_cache_group_id=i, metadata_builders=[builder])])

events: list[tuple] = []


class _Stop(Exception):
    pass


def make_model_state():
    ms = object.__new__(DeepseekV41ModelState)
    ms.ced_state = CEDState(MAX_REQS, MAX_TOKENS, (64, 128, 256, 512, 1024, 2048), dev)
    ms.ced_state.warmup(5120, 4, pipeline=False, sink_columns=False)
    ms._ced_prompt_logprobs = set()
    ms._ced_prefix_start = {}
    ms._ced_recovery_prompt_end = {}
    ms._ced_batch = None
    ms.max_cudagraph_query_len = 128  # CED_WINDOW, as __init__ sets it for V4.1
    # Instance stubs for what the step calls outside the condition's fast path.
    ms.preprocess_state = lambda *args: None
    ms.prepare_attn = lambda *args, **kwargs: events.append(("prepare_attn", args[1], kwargs)) or {}

    def prepare_inputs(batch, req_states):
        events.append(("prepare_inputs", batch is ms._ced_batch))
        raise _Stop
    ms.prepare_inputs = prepare_inputs
    return ms


ms = make_model_state()
can = ModelCudaGraphManager.can_capture_full_attn_metadata(
    ms, SimpleNamespace(cp_size=1), attn_groups, None)
check("can_capture_full_attn_metadata(DeepseekV41ModelState, 23 B12x groups)", can is True)
manager = object.__new__(ModelCudaGraphManager)
manager.captures_full_attn_metadata = can
speculator = object.__new__(DSparkSpeculator)

# 3. The warm-up decode step: SchedulerOutput as warmup._run_decode_step builds it.
input_buffers = InputBuffers(MAX_REQS, MAX_TOKENS, dev)


def make_step(ms):
    nt = NUM_REQS * QUERY_LEN
    batch = InputBatch.make_dummy(NUM_REQS, nt, input_buffers, max_query_len=QUERY_LEN)
    seq = [PROMPT_LEN + QUERY_LEN + 7 * r for r in range(NUM_REQS)]
    input_buffers.seq_lens[:NUM_REQS].copy_(torch.tensor(seq, dtype=torch.int32))
    batch.is_padding.fill_(False)
    batch.num_computed_tokens_np[:] = [s - QUERY_LEN for s in seq]
    batch.num_computed_prefill_tokens_np[:] = PROMPT_LEN
    batch.prefill_len_np[:] = PROMPT_LEN
    batch.num_draft_tokens = NUM_REQS * (QUERY_LEN - 1)
    # Request 1 resumed from a prefix-cache hit: its replay floor is its prefix start.
    ms._ced_prefix_start = {batch.req_ids[1]: 64}
    cached = CachedRequestData.make_empty()
    cached.req_ids = list(batch.req_ids)
    cached.num_computed_tokens = [int(c) for c in batch.num_computed_tokens_np]
    cached.num_output_tokens = [1] * NUM_REQS
    cached.new_block_ids = [None] * NUM_REQS
    so = SchedulerOutput.make_empty()
    so.scheduled_cached_reqs = cached
    so.num_scheduled_tokens = {r: QUERY_LEN for r in batch.req_ids}
    so.scheduled_spec_decode_tokens = {r: [0] * (QUERY_LEN - 1) for r in batch.req_ids}
    so.total_num_scheduled_tokens = nt
    so.num_common_prefix_blocks = [0] * len(attn_groups)
    desc = BatchExecutionDescriptor(CUDAGraphMode.FULL, nt, NUM_REQS,
                                    uniform_token_count=QUERY_LEN, max_query_len=QUERY_LEN)
    return batch, so, desc


def make_runner(connector, ms, batch):
    noop = lambda *args: None  # noqa: E731
    return SimpleNamespace(
        update_pp_decode_requests=noop, finish_requests=noop, free_states=noop,
        add_requests=noop, update_requests=noop,
        block_tables=SimpleNamespace(apply_staged_writes=lambda: None),
        boundary_checkpoint_state=None,
        kv_connector=connector,
        speculator=speculator,
        model_state=ms,
        lora_config=None, is_encoder_decoder=False,
        dp_size=1, dp_rank=0, dcp_size=1, dcp_rank=0, cp_interleave=1,
        pcp_manager=None, ubatch_runner=None,
        parallel_config=SimpleNamespace(),
        observability_config=SimpleNamespace(cudagraph_metrics=False),
        uniform_decode_query_len=QUERY_LEN,
        cudagraph_manager=manager,
        gather_batch_req_state=lambda *args: (
            SimpleNamespace(num_tokens=batch.num_tokens, has_prefill=False), QUERY_LEN),
        prepare_inputs=lambda *args: batch,
        prepare_attn=lambda b: ([g[0].metadata_builders[0] for g in attn_groups], object()),
        draft_tokens_handler=SimpleNamespace(set_consumed_draft_tokens=lambda *a: None),
        req_states=SimpleNamespace(draft_tokens=None, num_computed_tokens=SimpleNamespace(gpu=None)),
        input_buffers=input_buffers,
        kv_cache_config=SimpleNamespace(),
        attn_groups=attn_groups,
        uses_inputs_embeds=False,
    )


def run_step(case, connector):
    ms_case = make_model_state()
    batch, so, desc = make_step(ms_case)
    runner = make_runner(connector, ms_case, batch)
    mr.dispatch_cg_and_sync_dp = lambda *args, **kwargs: (desc, None)
    real_build = mr.build_slot_mappings_by_layer
    mr.build_slot_mappings_by_layer = lambda slots, cfg: events.append(("slot_mappings_by_layer",)) or {}
    events.clear()
    outcome, err = "stopped-before-model-inputs", None
    try:
        mr.GPUModelRunner.execute_model(runner, so)
        outcome = "returned"
    except _Stop:
        pass
    except Exception as e:  # the boot's failure mode
        tb = traceback.extract_tb(e.__traceback__)
        last = tb[-1]
        outcome, err = "raised", {"type": type(e).__name__, "message": str(e),
                                  "at": f"{last.filename}:{last.lineno}", "line": last.line}
    finally:
        mr.build_slot_mappings_by_layer = real_build
    torch.cuda.synchronize()
    emit(kind="step", case=case, connector=type(connector).__name__,
         connector_is_no_op=connector is kvc.NO_OP_KV_CONNECTOR, outcome=outcome, error=err,
         events=[e[0] if e[0] != "prepare_inputs" else f"prepare_inputs(batch staged={e[1]})" for e in events],
         ced_plan_is_none=ms_case.ced_state.plan is None,
         staged_floors=ms_case.ced_state._staged_floors[:NUM_REQS]
         if hasattr(ms_case.ced_state, "_staged_floors") else None)
    return outcome, err, ms_case, batch


# 4. NO_OP_KV_CONNECTOR: the serving case.
outcome, err, ms0, batch0 = run_step("no-kv-connector", no_op)
names_seen = [e[0] for e in events]
check("no connector: execute_model passes the condition (no exception)", outcome != "raised", error=err)
check("no connector: fast path (stage_full_cudagraph_attn, no eager metadata)",
      outcome == "stopped-before-model-inputs" and "prepare_attn" not in names_seen
      and "slot_mappings_by_layer" not in names_seen and ms0._ced_batch is batch0)
check("no connector: prepare_inputs sees the batch staged (no eager restage)",
      ("prepare_inputs", True) in events)
floors, _ = replay_rows(batch0.num_scheduled_tokens, batch0.num_computed_prefill_tokens_np,
                        batch0.prefill_len_np, [ms0._ced_prefix_start.get(r, 0) for r in batch0.req_ids],
                        [False] * NUM_REQS)
if outcome != "raised":
    check("no connector: CED floors staged = replay_rows floors",
          ms0.ced_state._staged_floors[:NUM_REQS] == [int(f) for f in floors], floors=floors)
    # The step's FULL graph refreshes CED in-graph (ModelCudaGraphManager.capture records
    # model_state.prepare_cudagraph_replay); it must equal an eager stage of the same batch.
    st = ms0.ced_state
    ms0.prepare_cudagraph_replay(input_buffers, NUM_REQS, batch0.num_tokens)  # kernel warm-up
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        ms0.prepare_cudagraph_replay(input_buffers, NUM_REQS, batch0.num_tokens)
    for name in ("replay_start", "request_positions", "starts"):
        getattr(st, name).fill_(-7)  # the replay must write them
    g.replay()
    eager = CEDState(MAX_REQS, MAX_TOKENS, (64, 128, 256, 512, 1024, 2048), dev)
    keep_full, fl, limits = ms0._ced_rows(batch0)
    eager.stage(batch0.query_start_loc, batch0.seq_lens, batch0.num_scheduled_tokens, keep_full, fl, limits)
    torch.cuda.synchronize()
    eq = {n: torch.equal(getattr(st, n)[:NUM_REQS], getattr(eager, n)[:NUM_REQS])
          for n in ("replay_start", "request_positions", "floors")}
    eq["starts"] = torch.equal(st.starts[:NUM_REQS + 1], eager.starts[:NUM_REQS + 1])
    check("no connector: in-graph CED refresh == eager CEDState.stage", all(eq.values()) and eager.plan is None,
          equal=eq, replay_start=st.replay_start[:NUM_REQS].tolist(),
          request_positions=st.request_positions[:NUM_REQS].tolist())

# 5. ActiveKVConnector: the same factory with a KV transfer group.
real_has, real_get = kvc.has_kv_transfer_group, kvc.get_kv_transfer_group
kvc.has_kv_transfer_group, kvc.get_kv_transfer_group = (lambda: True), Mock
try:
    active = kvc.get_kv_connector(SimpleNamespace(), {})
finally:
    kvc.has_kv_transfer_group, kvc.get_kv_transfer_group = real_has, real_get
check("factory with a transfer group returns ActiveKVConnector", isinstance(active, kvc.ActiveKVConnector))
outcome, err, ms1, batch1 = run_step("active-kv-connector", active)
names_seen = [e[0] for e in events]
check("active connector: execute_model passes the condition (no exception)", outcome != "raised", error=err)
check("active connector: eager metadata path (slot mappings by layer + prepare_attn FULL)",
      outcome == "stopped-before-model-inputs" and names_seen[:2] == ["slot_mappings_by_layer", "prepare_attn"]
      and events[1][1] == CUDAGraphMode.FULL and events[1][2] == {"for_capture": False})

emit(kind="summary", failures=failures, passed=not failures)
fh.close()
sys.exit(1 if failures else 0)
