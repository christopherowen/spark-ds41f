#!/usr/bin/env python3
"""The determinism gate's text requests (token_determinism_text.py --wide, then a
prefix-cache re-run) through the frontend and the engine of the arm's code, with a
deterministic stand-in for the model, recording everything the scheduler decides.

1. Frontend. A real AsyncLLM from the served DS4.1 checkpoint with the recipe's flags
   (--mm-processor-cache-gb 0 and prefix caching: the multimodal fast path is on where
   the arm has it), each gate request rendered by the real OnlineRenderer (chat
   template, thinking off, cache_salt, logprobs) and submitted with AsyncLLM.generate.
   The engine-core client is an in-process fake that msgpack-encodes each
   EngineCoreRequest (the IPC send) and keeps the frames.
2. Engine. A real EngineCore (its __init__, with a stand-in executor that reports
   DS4.1's KV cache specs as ds41_kv_specs.py read them from the worker's own
   get_kv_cache_spec, and the recipe's KV bytes), so its AsyncScheduler and
   KVCacheManager are the real ones. Each request is decoded from the frontend's
   frames, preprocessed (preprocess_add_request: the block hasher) and enters through
   EngineCoreProc._handle_client_request(ADD). The loop is step_with_batch_queue
   (async scheduling's batch queue, late admission at VLLM_LATE_ADMISSION_LAYER=30 as
   the recipe sets it). The stand-in model samples, for each request, a token hashed
   from that request's own token history (equal histories give equal tokens, as greedy
   decoding does) and accepts a hash-chosen number of its scheduled drafts.

Arrivals follow the gate's order and concurrency, counted in engine steps: one at a
time (the next once the previous has finished), all at once (c8), 8 steps apart
(staggered), the long prompt 24 steps after the rest (mixed, mixed15). Salts are fixed
strings so that runs compare. After the gate's scenarios, again_salted re-runs every
prompt with its reference salt (the long prompt then hits the prefix cache).

Recorded: every step's SchedulerOutput (scheduled tokens and drafts per request,
computed tokens and block ids of new requests, new block ids and computed tokens of
running ones, resumptions, preemptions, finishes, common-prefix blocks), and per
request its prefix-cache hit (num_cached_tokens) and output tokens.
Positive controls (REPLAY_CONTROL, candidate-54's code, to show the record catches a
change): "nohit" gives the long prompt's prefix-cache re-run no hit; "late" makes one
request of the gate's last scenario (again-2) arrive one step later.
Usage: gate_replay.py <label> <out.json>
"""
import asyncio
import contextlib
import hashlib
import json
import os
import pickle
import sys
import time
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import patch

import msgspec

sys.path.insert(0, "/opt/exp")
import token_determinism_text as gate  # noqa: E402
from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs  # noqa: E402
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest  # noqa: E402
from vllm.renderers.online_renderer import OnlineRenderer  # noqa: E402
from vllm.usage.usage_lib import UsageContext  # noqa: E402
from vllm.v1.engine import (  # noqa: E402
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreRequest,
    EngineCoreRequestType,
    FinishReason,
)
from vllm.v1.engine.async_llm import AsyncLLM  # noqa: E402
from vllm.v1.engine.core import EngineCore, EngineCoreProc  # noqa: E402
from vllm.v1.engine.core_client import EngineCoreClient  # noqa: E402
from vllm.v1.executor.abstract import Executor  # noqa: E402
from vllm.v1.kv_cache_interface import KVCacheLayout  # noqa: E402
from vllm.v1.outputs import ModelRunnerOutput  # noqa: E402
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder  # noqa: E402

LABEL, OUT = sys.argv[1], sys.argv[2]
CONTROL = os.environ.get("REPLAY_CONTROL", "")
SPECS = os.environ.get("DS41_SPECS", "/c/ds41_specs.pkl")
TOKENS, LONG_TOKENS, LONG_LINES = 192, 64, 700
STAGGER, LONG_DELAY = 8, 24  # engine steps: 100 ms and 0.3 s at ~12 ms a step


def recipe_kwargs():
    return dict(
        model=os.environ["DS41F_SNAPSHOT"], served_model_name=["deepseek-v4.1-flash"],
        tokenizer_mode="deepseek_v41", dtype="bfloat16", block_size=256,
        enable_prefix_caching=True, enable_chunked_prefill=True, max_model_len=1048576,
        max_num_seqs=16, max_num_batched_tokens=16384,
        long_prefill_token_threshold=16384 - 16 * 6, async_scheduling=True,
        prefix_cache_retention_interval=8192, limit_mm_per_prompt={"image": 67},
        mm_processor_cache_gb=0, kv_cache_memory_bytes=11274289152, max_parallel_prefills=1,
        attention_backend="TILELANG", generation_config="vllm",
        engram_config={"table_memory": "disk", "disk_resident_scales": False, "projection_tp": True},
        speculative_config={"method": "dspark", "num_speculative_tokens": 5,
                            "draft_tensor_parallel_size": 1, "attention_backend": "TILELANG",
                            "draft_sample_method": "greedy", "rejection_sample_method": "block",
                            "enable_adaptive_verification": True,
                            "adaptive_verification_row_cost": 0.035},
    )


def sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


# --- the gate's requests, in its order -------------------------------------------------
def gate_plan():
    """Phases of (arrival offset in steps, key, prompt, max tokens, salt). A phase starts
    when every request of the previous one has finished."""
    long_text = gate.long_prompt(LONG_LINES)
    phases = []
    ref_salt = {}
    for i, p in enumerate(gate.PROMPTS):
        ref_salt[i] = f"reference-{i}"
        phases.append(("reference", [(0, f"reference-{i}", p, TOKENS, ref_salt[i])]))
    phases.append(("long_reference", [(0, "long_reference", long_text, LONG_TOKENS, "long-reference")]))
    phases.append(("c8", [(0, f"c8-{i}", p, TOKENS, f"c8-{i}") for i, p in enumerate(gate.PROMPTS)]))
    phases.append(("staggered", [(STAGGER * i, f"staggered-{i}", p, TOKENS, f"staggered-{i}")
                                 for i, p in enumerate(gate.PROMPTS)]))
    phases.append(("mixed", [(0, f"mixed-{i}", p, TOKENS, f"mixed-{i}") for i, p in enumerate(gate.PROMPTS)]
                   + [(LONG_DELAY, "mixed-long", long_text, LONG_TOKENS, "mixed-long")]))
    for i, p in enumerate(gate.MORE_PROMPTS):
        phases.append(("more_reference", [(0, f"more_reference-{i}", p, TOKENS, f"more-{i}")]))
    phases.append(("mixed15", [(0, f"mixed15-{i}", p, TOKENS, f"mixed15-{i}")
                               for i, p in enumerate(gate.PROMPTS + gate.MORE_PROMPTS)]
                   + [(LONG_DELAY, "mixed15-long", long_text, LONG_TOKENS, "mixed15-long")]))
    for i, p in enumerate(gate.PROMPTS):
        phases.append(("cached", [(0, f"cached-{i}-a", p, TOKENS, f"cached-{i}")]))
        phases.append(("cached", [(0, f"cached-{i}-b", p, TOKENS, f"cached-{i}")]))
    for i, p in enumerate(gate.PROMPTS):  # the gate's last scenario: fresh salts
        phases.append(("again", [(0, f"again-{i}", p, TOKENS, f"again-{i}")]))
    # Beyond the gate: the same prompts with their reference salts (prefix-cache re-run).
    for i, p in enumerate(gate.PROMPTS):
        phases.append(("again_salted", [(0, f"again_salted-{i}", p, TOKENS, ref_salt[i])]))
    phases.append(("again_salted", [(0, "again_salted-long", long_text, LONG_TOKENS, "long-reference")]))
    return phases


# --- 1. frontend -----------------------------------------------------------------------
class FakeEngineCore:
    def __init__(self):
        self.resources = SimpleNamespace(engine_dead=False)
        self.engine_ranks_managed = [0]
        self.encoder = MsgpackEncoder()
        self.outputs = asyncio.Queue()
        self.sent: list[EngineCoreRequest] = []

    async def add_request_async(self, req):
        self.encoder.encode(req)  # the IPC serialization add_request_async does
        self.sent.append(req)
        self.outputs.put_nowait(EngineCoreOutputs(
            outputs=[EngineCoreOutput(request_id=req.request_id, new_token_ids=[100],
                                      finish_reason=FinishReason.LENGTH)],
            finished_requests={req.request_id}))

    async def get_output_async(self):
        return await self.outputs.get()

    async def abort_requests_async(self, request_ids):
        pass

    async def get_supported_tasks_async(self):
        return ("generate",)

    def shutdown(self, timeout=None):
        pass


async def frontend(plan):
    args = AsyncEngineArgs(**recipe_kwargs())
    vllm_config = args.create_engine_config(UsageContext.OPENAI_API_SERVER)
    fake = FakeEngineCore()
    with patch.object(EngineCoreClient, "make_async_mp_client", return_value=fake):
        llm = AsyncLLM(vllm_config, executor_class=object, log_stats=False, log_requests=False)
    online = OnlineRenderer(model_config=llm.model_config, renderer=llm.renderer,
                            request_logger=None, chat_template=None,
                            chat_template_content_format="auto",
                            reasoning_parser="deepseek_v41", tool_parser="deepseek_v41",
                            enable_auto_tools=True)
    defaults = llm.model_config.get_diff_sampling_param()
    encoder = MsgpackEncoder()
    requests, digests = {}, {}
    for _, jobs in plan:
        for _, key, prompt, tokens, salt in jobs:
            chat = ChatCompletionRequest(
                model="deepseek-v4.1-flash", messages=[{"role": "user", "content": prompt}],
                max_tokens=tokens, temperature=0, ignore_eos=True, cache_salt=salt,
                chat_template_kwargs={"thinking": False}, logprobs=True)
            _, engine_inputs = await online.render_chat(chat)
            params = chat.to_sampling_params(tokens, defaults)
            async for _ in llm.generate(engine_inputs[0], params, key):
                pass
            sent = fake.sent[-1]
            # Fixed id and arrival time, so arms compare byte for byte.
            norm = msgspec.structs.replace(sent, request_id=key, arrival_time=0.0)
            frames = encoder.encode(norm)
            blob = b"".join(bytes(f) for f in frames)
            requests[key] = [bytes(f) for f in frames]
            digests[key] = {"bytes": hashlib.sha256(blob).hexdigest(),
                            "prompt_tokens": len(sent.prompt_token_ids or ()),
                            "mm_features": len(sent.mm_features or ()),
                            "cache_salt": sent.cache_salt}
    fast_path = bool(getattr(llm, "_mm_prefix_fastpath", False))
    llm.shutdown()
    return requests, digests, fast_path


# --- 2. engine -------------------------------------------------------------------------
def _done(value) -> Future:
    f: Future = Future()
    f.set_result(value)
    return f


def _h(tokens: list[int], salt: bytes) -> int:
    return int.from_bytes(hashlib.blake2b(bytes(str(tokens[-64:]), "ascii") + salt
                                          + len(tokens).to_bytes(4, "little"),
                                          digest_size=8).digest(), "little")


class StandInModel:
    """The V2 runner's side of the scheduler's contract with sink rejection on (the
    recipe's default for DS4.1), as tests/v1/core/test_sink_draft_zero.py's scripted
    worker plays it: the step that finishes a prompt samples the first provisional
    token; a draft-zero step commits the provisional token, accepts some of the
    drafter's tokens and samples the next provisional one. Here the true token at
    each position is a hash of the request's own tokens before it, and the number
    of drafter tokens accepted is a hash of the same history, so a request's outputs
    and step shapes depend only on its own tokens."""

    def __init__(self):
        self.scheduler = None
        self.state: dict[str, dict] = {}
        self.pending = None

    @staticmethod
    def true_token(history: list[int]) -> int:
        return 1000 + _h(history, b"token") % 100000

    def execute(self, so):
        for rid in (so.preempted_req_ids or set()) | so.finished_req_ids:
            self.state.pop(rid, None)
        self.pending = so

    def sample(self):
        so, self.pending = self.pending, None
        draft_zero = so.sink_draft_zero_req_ids or set()
        req_ids = list(so.num_scheduled_tokens)
        sampled = []
        for rid in req_ids:
            request = self.scheduler.requests[rid]
            spec = so.scheduled_spec_decode_tokens.get(rid, [])
            if rid in draft_zero:
                st = self.state[rid]
                st["history"].append(st["provisional"])
                out = [st["provisional"]]
                drafter = max(0, len(spec) - 1)
                accepted = _h(st["history"], b"accept") % (drafter + 1)
                for _ in range(accepted):
                    t = self.true_token(st["history"])
                    st["history"].append(t)
                    out.append(t)
                st["provisional"] = self.true_token(st["history"])
                out.append(st["provisional"])
            elif request.is_prefill_chunk:
                out = []
            else:
                history = list(request.prompt_token_ids) + list(request.output_token_ids)
                st = self.state[rid] = {"history": history,
                                        "provisional": self.true_token(history)}
                out = [st["provisional"]]
            sampled.append(out)
        return ModelRunnerOutput(req_ids=req_ids,
                                 req_id_to_index={r: i for i, r in enumerate(req_ids)},
                                 sampled_token_ids=sampled, logprobs=None,
                                 prompt_logprobs_dict={}, pooler_output=[])


class StandInExecutor(Executor):
    specs = None

    def _init_executor(self) -> None:
        self.model = StandInModel()

    @classmethod
    def supports_async_scheduling(cls) -> bool:
        return True

    @contextlib.contextmanager
    def b12x_warmup_control(self):
        yield

    def _run_b12x_preparation(self, *, stage: str) -> None:
        pass

    def get_kv_cache_specs(self):
        return [dict(self.specs)]

    def determine_available_memory(self):
        return [self.cache_config.kv_cache_memory_bytes]

    def get_supported_kv_cache_layouts(self):
        return [list(KVCacheLayout.__members__)]

    def set_kv_cache_layout(self, layout_name: str) -> None:
        pass

    def initialize_from_config(self, kv_cache_configs) -> None:
        self.kv_cache_configs = kv_cache_configs

    def compile_or_warm_up_model(self) -> None:
        pass

    def collective_rpc(self, method, timeout=None, args=(), kwargs=None, non_block=False, **kw):
        out = [None]
        return _done(out) if non_block else out

    def execute_model(self, scheduler_output, non_block=False):
        self.model.execute(scheduler_output)
        out = None
        if scheduler_output.total_num_scheduled_tokens == 0:
            # Nothing to run (as the worker's EMPTY_MODEL_RUNNER_OUTPUT).
            self.model.pending = None
            out = ModelRunnerOutput(req_ids=[], req_id_to_index={}, sampled_token_ids=[],
                                    logprobs=None, prompt_logprobs_dict={}, pooler_output=[])
        return _done(out) if non_block else out

    def sample_tokens(self, grammar_output, non_block=False):
        out = self.model.sample()
        return _done(out) if non_block else out

    def take_draft_token_ids(self):
        return None

    def check_health(self) -> None:
        pass

    def shutdown(self) -> None:
        pass


class HarnessCore(EngineCore):
    """EngineCore with EngineCoreProc's client-request dispatch (ADD)."""

    _handle_client_request = EngineCoreProc._handle_client_request
    _reject_add_in_shutdown = EngineCoreProc._reject_add_in_shutdown


def summarize(so) -> dict:
    cached = so.scheduled_cached_reqs
    return {
        "new": [[r.req_id, r.num_computed_tokens, [list(g) for g in r.block_ids],
                 len(r.prompt_token_ids or ())] for r in so.scheduled_new_reqs],
        "cached": [[rid, cached.num_computed_tokens[i],
                    None if cached.new_block_ids[i] is None else [list(g) for g in cached.new_block_ids[i]],
                    rid in cached.resumed_req_ids] for i, rid in enumerate(cached.req_ids)],
        "tokens": sorted(so.num_scheduled_tokens.items()),
        "drafts": sorted((k, len(v)) for k, v in so.scheduled_spec_decode_tokens.items()),
        "common_prefix_blocks": list(so.num_common_prefix_blocks),
        "finished": sorted(so.finished_req_ids),
        "preempted": sorted(so.preempted_req_ids or ()),
        "encoder": sorted(so.scheduled_encoder_inputs.items()),
    }


def engine(plan, frames):
    from vllm.v1.engine.core import EngineShutdownState
    with open(SPECS, "rb") as f:
        StandInExecutor.specs = pickle.load(f)
    vllm_config = EngineArgs(**recipe_kwargs()).create_engine_config(UsageContext.OPENAI_API_SERVER)
    core = HarnessCore(vllm_config, StandInExecutor, log_stats=True)
    core.model_executor.model.scheduler = core.scheduler
    core._loop_watchdog = None
    core._add_received = {}
    core._stall_warning_s = 0
    core.shutdown_state = EngineShutdownState.RUNNING
    sched = core.scheduler
    decoder = MsgpackDecoder(EngineCoreRequest)
    trace, outputs = [], {}
    schedule = sched.schedule

    def recorded(*a, **k):
        so = schedule(*a, **k)
        trace.append(summarize(so))
        return so

    sched.schedule = recorded
    if CONTROL == "nohit":
        lookup = sched.kv_cache_manager.get_computed_blocks

        def no_hit(request):
            if request.request_id == "again_salted-long":
                return sched.kv_cache_manager.empty_kv_cache_blocks, 0, 0
            return lookup(request)

        sched.kv_cache_manager.get_computed_blocks = no_hit
    if CONTROL == "late":
        plan = [(name, [(off + 1 if key == "again-2" else off, key, *rest)
                        for off, key, *rest in jobs]) for name, jobs in plan]
    step = 0
    for scenario, jobs in plan:
        start = step
        waiting = sorted(jobs, key=lambda j: j[0])
        active = {j[1] for j in jobs}
        while waiting or active or core.batch_queue:
            while waiting and waiting[0][0] <= step - start:
                _, key, *_ = waiting.pop(0)
                req = decoder.decode(frames[key])
                core._handle_client_request(EngineCoreRequestType.ADD,
                                            core.preprocess_add_request(req))
                trace.append({"add": key, "step": step})
            if sched.has_requests() or core.batch_queue:
                outs, _ = core.step_with_batch_queue()
            else:
                outs = None
            for client_outs in (outs or {}).values():
                for o in client_outs.outputs:
                    rec = outputs.setdefault(o.request_id, {"scenario": scenario, "tokens": [],
                                                            "num_cached_tokens": None})
                    rec["tokens"].extend(o.new_token_ids)
                    if rec["num_cached_tokens"] is None and o.prefill_stats is not None:
                        rec["num_cached_tokens"] = o.prefill_stats.num_cached_tokens
                        rec["prefill"] = [o.prefill_stats.num_prompt_tokens,
                                          o.prefill_stats.num_computed_tokens,
                                          o.prefill_stats.num_local_cached_tokens]
                    if o.finish_reason is not None:
                        rec["finish"] = str(o.finish_reason)
                        active.discard(o.request_id)
            step += 1
            if step - start > 100000:
                raise RuntimeError(f"{scenario} did not finish")
    groups = [[type(g.kv_cache_spec).__name__, g.kv_cache_spec.block_size, len(g.layer_names)]
              for g in sched.kv_cache_config.kv_cache_groups]
    info = {"scheduler": type(sched).__name__, "num_blocks": sched.kv_cache_config.num_blocks,
            "kv_cache_groups": groups, "batch_queue_size": core.batch_queue_size,
            "late_admission": core.late_admission, "num_spec_tokens": sched.num_spec_tokens,
            "block_size": sched.block_size, "sink_rejection": getattr(sched, "sink_rejection", None),
            "hash_block_size": getattr(sched, "hash_block_size", None)}
    return trace, outputs, step, info


def main():
    t0 = time.time()
    plan = gate_plan()
    frames, request_digests, fast_path = asyncio.run(frontend(plan))
    trace, outputs, steps, info = engine(plan, frames)
    per_request = {k: {"scenario": v["scenario"], "num_cached_tokens": v["num_cached_tokens"],
                       "output_tokens": len(v["tokens"]), "finish": v.get("finish"),
                       "prefill": v.get("prefill"),
                       "tokens_sha": sha(v["tokens"])[:16]} for k, v in outputs.items()}
    by_scenario = {}
    for k, v in per_request.items():
        s = by_scenario.setdefault(v["scenario"], {"requests": 0, "cached_tokens": 0})
        s["requests"] += 1
        s["cached_tokens"] += v["num_cached_tokens"] or 0
    # Identical histories give identical outputs: compare each scenario with the reference.
    ref = {k.split("-", 1)[1]: v["tokens_sha"] for k, v in per_request.items()
           if v["scenario"] == "reference"}
    same_as_reference = {}
    for k, v in per_request.items():
        if v["scenario"] in ("c8", "staggered", "mixed", "again", "again_salted") and not k.endswith("long"):
            idx = k.rsplit("-", 1)[1]
            same_as_reference[k] = v["tokens_sha"] == ref.get(idx)
    result = {
        "label": LABEL, "control": CONTROL or None, "seconds": round(time.time() - t0, 1),
        "mm_prefix_fastpath": fast_path,
        "engine": info, "steps": steps, "requests": len(per_request),
        "frontend_requests_sha256": sha(request_digests),
        "frontend_requests": request_digests,
        "trace_sha256": sha(trace), "outputs_sha256": sha(per_request),
        "by_scenario": by_scenario, "per_request": per_request,
        "stand_in_outputs_equal_reference": same_as_reference,
        "trace": trace,
    }
    with open(OUT, "w") as f:
        json.dump(result, f)
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("trace", "per_request", "frontend_requests",
                                   "stand_in_outputs_equal_reference")}, indent=1))


if __name__ == "__main__":
    main()
