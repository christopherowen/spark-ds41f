#!/usr/bin/env python3
"""Build the boot-differences debug overlay from candidate-58's own vLLM files.

usage: make_overlay.py VLLM_TREE [--out DIR] [--check]

VLLM_TREE is a checkout of candidate-58's vLLM tree (the image's
``local.spark-ds41f.vllm.tree`` label). Each overlay file is that tree's file with
debug hooks inserted at fixed anchors (each must occur exactly once); nothing else
changes. Two traces, each off unless its variable is set, so a profile that mounts
the overlay without them serves exactly as the image does:

- SPARK_DS41F_STEP_TRACE_DIR (host side, every step, every rank and the engine
  core): the scheduler's decision and its inputs (scheduled tokens and draft
  placeholders per request, skip_drafts, the copy allocation it read, what it got
  back), the worker's draft budget with its inputs (the stale confidence snapshot,
  tau, the idle counter) and its result, and after each step the drafts, raw and
  planned confidences, accepted and rejected counts, dead-row cut, copy-drafter
  state and the estimator's coefficients and counters. Device values reach the
  host by non-blocking copies behind an event, written by a thread: no sync on
  the step's path, nothing inside a CUDA graph.
- SPARK_DS41F_ROW_TRACE_DIR (device side, the target forward): a 64-bit digest of
  every row, for steps of at most SPARK_DS41F_ROW_TRACE_MAX_ROWS rows (64), of
  each tap in forward order: every all-reduce's input (this rank's partial) and
  output and every all-gather's (kinds 1-3), each compressor's projected values
  and emitted latents (4) and, at ratio 2, the first request's ring carry (5),
  each indexer's top-k selection (6), each attention layer's output before WO (7),
  and each Engram layer's gathered table rows and output (8); plus the step's
  positions, token ids and final hidden states. The digests are kernels on the
  stream that produced the tensor, so CUDA graphs capture and replay them.

- SPARK_DS41F_ROW_TRACE_DRAFTER=1 (with the row trace): the same digests for the
  DSpark drafter's two forwards on every rank, written to their own files
  (rows-dctx-*, rows-dquery-*): the context preparation (the target's auxiliary
  states copied in, the combined context states, every collective) and the query
  forward (its token ids and positions, every collective, every attention output,
  the head hidden state, the confidence head's inputs and output).
- SPARK_DS41F_ROW_TRACE_MOE=all or a decoder-layer range a-b (with the row trace):
  inside each traced MoE layer (fused_moe/runner/moe_runner.py), the router's top-k
  ids and weights (20), the input's MXFP8 cast the routed GEMM reads (24), the
  shared expert's input and output (21), the input and this rank's output, routed
  plus shared (22), and that output at the final reduction's entry (23). The
  drafter's MoE layers record into its traces.
- Drafter arms, one variable each (dflash/speculator.py): SPARK_DS41F_DSPARK_SYNC=1
  synchronizes the device at the drafter's entry, before and after its context
  preparation and after its query forward; SPARK_DS41F_DSPARK_EAGER=context|query|all
  runs the context preparation, the query forward, or both eagerly instead of
  replaying their CUDA graphs.

--check verifies that the overlay files in --out equal what this script builds.
"""
import argparse
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = {
    "cuda_communicator": "vllm/distributed/device_communicators/cuda_communicator.py",
    "model": "vllm/models/deepseek_v4_1/nvidia/model.py",
    "model_runner": "vllm/v1/worker/gpu/model_runner.py",
    "scheduler": "vllm/v1/core/sched/scheduler.py",
    "attention": "vllm/models/deepseek_v4_1/attention.py",
    "compressor": "vllm/models/deepseek_v4_1/compressor.py",
    "engram": "vllm/models/deepseek_v4_1/common/engram.py",
    "dflash": "vllm/v1/worker/gpu/spec_decode/dflash/speculator.py",
    "dspark": "vllm/models/deepseek_v4_1/nvidia/dspark.py",
    "moe_runner": "vllm/model_executor/layers/fused_moe/runner/moe_runner.py",
}

ROW_TRACE = r'''

# ---------------------------------------------------------------------------
# Debug row trace (boot-differences overlay). Off unless SPARK_DS41F_ROW_TRACE_DIR
# is set; unset, nothing below is installed and CudaCommunicator is unchanged.
import os as _rt_os  # noqa: E402

ROW_TRACE_DIR = _rt_os.environ.get("SPARK_DS41F_ROW_TRACE_DIR", "")


class RowTrace:
    """Per-row 64-bit digests of the target forward's collectives on this rank.

    ``begin`` (the target model's forward) copies the step's positions and token
    ids into a device buffer; each all-reduce (kind 1) and all-gather (kind 2: last
    dim, 3: dim 0) in the forward writes the digest of every row of its input and
    output into slot k, k its order in the forward; ``end`` writes the slot count.
    All of it is device work on the forward's stream, so CUDA graph capture records
    it and replays repeat it. ``harvest`` (the runner, after the forward) adds the
    final hidden states' digests and copies the buffer to pinned memory behind an
    event; a thread appends one record per step. A row's digest is the sum of its
    32-bit words times fixed odd weights below 2^16 (no overflow): one changed word
    always changes it.
    """

    MAX_CALLS = 512
    MAX_WORDS = 1 << 15
    SLOTS = 8

    def __init__(self, directory: str, max_rows: int, name: str = "rows"):
        self.directory = directory
        self.name = name
        # Records made while this trace is inactive go to the delegate (the
        # drafter's traces while the drafter runs).
        self.delegate = None
        self.max_rows = int(max_rows)
        self.buf = None
        self.failed = None
        self.active = False
        self.k = 0
        self.n = 0
        self.calls: list = []
        self.layouts: dict = {}

    def _allocate(self, device) -> None:
        import queue
        import threading

        R, C = self.max_rows, self.MAX_CALLS
        self._pos = 2
        self._ids = self._pos + R
        self._ring = self._ids + R
        self._final = self._ring + C * 2 * R
        total = self._final + R
        self.buf = torch.zeros(total, dtype=torch.int64, device=device)
        self.meta = self.buf[0:2]
        self.pos = self.buf[self._pos:self._ids]
        self.ids = self.buf[self._ids:self._ring]
        self.ring = self.buf[self._ring:self._final].view(C, 2, R)
        self.final = self.buf[self._final:total]
        g = torch.Generator(device="cpu").manual_seed(20261110)
        weights = torch.randint(0, 1 << 15, (self.MAX_WORDS,), generator=g, dtype=torch.int64)
        self.weights = (weights * 2 + 1).to(device)
        self.pinned = [torch.empty(total, dtype=torch.int64).pin_memory() for _ in range(self.SLOTS)]
        self.events = [torch.cuda.Event() for _ in range(self.SLOTS)]
        self.free: queue.Queue = queue.Queue()
        for slot in range(self.SLOTS):
            self.free.put(slot)
        self.pending: queue.Queue = queue.Queue()
        threading.Thread(target=self._writer, name="row-trace", daemon=True).start()

    def _digest(self, t, rows: int, out) -> None:
        nbytes = t.numel() * t.element_size()
        if t.dim() == 0 or t.shape[0] != rows or nbytes % (4 * rows) or nbytes // (4 * rows) > self.MAX_WORDS:
            out.fill_(-1)
            return
        words = nbytes // (4 * rows)
        w = t.contiguous().view(-1).view(torch.uint8).view(rows, words * 4).view(torch.int32)
        torch.sum(w.to(torch.int64) * self.weights[:words], dim=1, out=out)

    def begin(self, input_ids, positions) -> None:
        if self.failed is not None:
            return
        n = int(positions.shape[-1])
        if self.buf is None:
            if torch.cuda.is_current_stream_capturing():
                self.failed = "the first traced forward was a CUDA graph capture"
                logger.warning("row trace off: %s", self.failed)
                return
            self._allocate(positions.device)
        self.k, self.n, self.calls = 0, n, []
        self.active = n <= self.max_rows
        self.meta[1:2].fill_(n)
        if self.active:
            self.pos[:n].copy_(positions.reshape(-1)[:n])
            if input_ids is not None:
                self.ids[:n].copy_(input_ids.reshape(-1)[:n])

    def record(self, kind: int, inp, out) -> None:
        if not self.active:
            if self.delegate is not None:
                self.delegate.record(kind, inp, out)
            return
        if self.k >= self.MAX_CALLS:
            return
        k = self.k
        self.k += 1
        rows_in = int(inp.shape[0]) if inp.dim() else 0
        rows_out = int(out.shape[0]) if out.dim() else 0
        self.calls.append((kind, rows_in, rows_out, str(inp.dtype).replace("torch.", ""),
                           int(inp.numel() // max(rows_in, 1))))
        for j, t, rows in ((0, inp, rows_in), (1, out, rows_out)):
            if 0 < rows <= self.max_rows:
                self._digest(t, rows, self.ring[k, j, :rows])

    def end(self) -> None:
        if self.buf is None or self.failed is not None:
            return
        self.meta[0:1].fill_(self.k if self.active else 0)
        self.layouts[self.n] = list(self.calls)
        self.active = False

    def harvest(self, input_batch, hidden_states, info: dict) -> None:
        if self.buf is None or self.failed is not None:
            return
        n = int(input_batch.num_tokens_after_padding)
        if n <= self.max_rows and isinstance(hidden_states, torch.Tensor) and hidden_states.shape[0] >= n:
            self._digest(hidden_states[:n], n, self.final[:n])
        slot = self.free.get()
        self.pinned[slot].copy_(self.buf, non_blocking=True)
        self.events[slot].record()
        if self.name != "rows":
            # A drafter trace: a step without a drafter forward then reads as empty.
            self.meta.zero_()
        self.pending.put((slot, dict(info, padded=n, layout=self.layouts.get(n))))

    def _writer(self) -> None:
        import os
        import pickle
        import socket
        import struct
        import time

        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

        try:
            rank = get_tensor_model_parallel_rank()
        except Exception:  # noqa: BLE001 - outside a TP group
            rank = -1
        os.makedirs(self.directory, exist_ok=True)
        path = os.path.join(self.directory, f"{self.name}-{socket.gethostname()}-tp{rank}-pid{os.getpid()}-"
                                            f"{int(time.time())}.bin")
        R = self.max_rows
        with open(path, "ab") as f:
            while True:
                slot, info = self.pending.get()
                self.events[slot].synchronize()
                b = self.pinned[slot].numpy()
                calls, rows = int(b[0]), int(b[1])
                rec = dict(info, calls=calls, rows=rows)
                if self.name != "rows":  # the drafter's forwards have their own row counts
                    rec["layout"] = self.layouts.get(rows)
                if 0 < rows <= R:
                    # Lists and raw little-endian int64 bytes ([calls, 2, rows]): the
                    # analysis needs no numpy, and a step is about 13 KB at four rows.
                    rec["positions"] = b[self._pos:self._pos + rows].tolist()
                    rec["ids"] = b[self._ids:self._ids + rows].tolist()
                    rec["ring_bytes"] = b[self._ring:self._final].reshape(self.MAX_CALLS, 2, R)[:calls, :, :rows].tobytes()
                    rec["ring_width"] = rows
                    rec["final"] = b[self._final:self._final + rows].tolist()
                self.free.put(slot)
                data = pickle.dumps(rec, protocol=4)
                f.write(struct.pack("<I", len(data)) + data)
                f.flush()


ROW_TRACE = (RowTrace(ROW_TRACE_DIR, int(_rt_os.environ.get("SPARK_DS41F_ROW_TRACE_MAX_ROWS", "64")))
             if ROW_TRACE_DIR else None)
# The drafter's context preparation and query forward, each its own buffer and file.
DRAFT_CTX_TRACE = DRAFT_QUERY_TRACE = None
if ROW_TRACE is not None and _rt_os.environ.get("SPARK_DS41F_ROW_TRACE_DRAFTER") == "1":
    DRAFT_CTX_TRACE = RowTrace(ROW_TRACE_DIR, ROW_TRACE.max_rows, name="rows-dctx")
    DRAFT_QUERY_TRACE = RowTrace(ROW_TRACE_DIR, ROW_TRACE.max_rows, name="rows-dquery")
    ROW_TRACE.delegate = DRAFT_CTX_TRACE
    DRAFT_CTX_TRACE.delegate = DRAFT_QUERY_TRACE

if ROW_TRACE is not None:
    _untraced_all_reduce = CudaCommunicator.all_reduce
    _untraced_all_gather = CudaCommunicator.all_gather

    def _traced_all_reduce(self, input_):
        out = _untraced_all_reduce(self, input_)
        ROW_TRACE.record(1, input_, out)
        return out

    def _traced_all_gather(self, input_, dim=-1):
        out = _untraced_all_gather(self, input_, dim)
        ROW_TRACE.record(3 if dim == 0 else 2, input_, out)
        return out

    CudaCommunicator.all_reduce = _traced_all_reduce
    CudaCommunicator.all_gather = _traced_all_gather
    logger.warning("debug row trace on: digests of every target-forward collective to %s", ROW_TRACE_DIR)
'''

MODEL_IMPORT_ANCHOR = "class DeepseekV41LLMForCausalLM(UpstreamDeepseekV41LLMForCausalLM):"
MODEL_IMPORT = '''from vllm.distributed.device_communicators.cuda_communicator import (  # noqa: E402
    ROW_TRACE as _ROW_TRACE,
)


'''
MODEL_FORWARD_OLD = '''    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            lookback_token_ids=lookback_token_ids,
            ced_indices=ced_indices,
        )
        return hidden_states
'''
MODEL_FORWARD_NEW = '''    ) -> torch.Tensor | IntermediateTensors:
        if _ROW_TRACE is not None:
            _ROW_TRACE.begin(input_ids, positions)
        hidden_states = self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            lookback_token_ids=lookback_token_ids,
            ced_indices=ced_indices,
        )
        if _ROW_TRACE is not None:
            _ROW_TRACE.end()
        return hidden_states
'''

RUNNER_BUDGET_OLD = '''        if self.adaptive_verification is not None and draft_tokens:
            num_toks = self.adaptive_verification.get_num_tokens(
                num_tokens_per_req, draft_tokens
            )
'''
RUNNER_BUDGET_NEW = '''        if self.adaptive_verification is not None and draft_tokens:
            if _STEP_TRACE is not None:
                _STEP_TRACE.before_budget(
                    self.adaptive_verification, num_tokens_per_req, draft_tokens
                )
            num_toks = self.adaptive_verification.get_num_tokens(
                num_tokens_per_req, draft_tokens
            )
            if _STEP_TRACE is not None:
                _STEP_TRACE.after_budget(self.adaptive_verification, num_toks)
'''
RUNNER_HARVEST_ANCHOR = '''        routed_experts = None
        if not dummy_run and (capturer := self.routed_experts_capturer) is not None:
'''
RUNNER_HARVEST = '''        if _ROW_TRACE is not None and not dummy_run:
            _ROW_TRACE.harvest(
                input_batch,
                hidden_states,
                {
                    "t": time.monotonic(),
                    "req_ids": list(input_batch.req_ids),
                    "num_scheduled_tokens": input_batch.num_scheduled_tokens.tolist(),
                    "num_tokens": int(input_batch.num_tokens),
                    "cg_mode": str(batch_desc.cg_mode),
                },
            )
'''
RUNNER_RECORD_ANCHOR = '''        # Post-step KV connector related operations.
        kv_connector_output = self.kv_connector.post_forward(finished_req_ids)
        model_runner_output.kv_connector_output = kv_connector_output
        model_runner_output.ec_connector_output = ec_connector_output

        return async_output
'''
RUNNER_RECORD = '''        if _STEP_TRACE is not None:
            _STEP_TRACE.record(self, input_batch, locals())
        for _draft_trace in _DRAFT_TRACES:
            _draft_trace.harvest(
                input_batch,
                None,
                {
                    "t": time.monotonic(),
                    "req_ids": list(input_batch.req_ids),
                    "num_scheduled_tokens": input_batch.num_scheduled_tokens.tolist(),
                    "num_tokens": int(input_batch.num_tokens),
                    "num_reqs": int(input_batch.num_reqs),
                    "query_per_req": getattr(getattr(self, "speculator", None), "num_query_per_req", None),
                },
            )

'''
STEP_TRACE = r'''

# ---------------------------------------------------------------------------
# Debug step trace (boot-differences overlay). Off unless SPARK_DS41F_STEP_TRACE_DIR
# is set.
_STEP_TRACE_DIR = os.environ.get("SPARK_DS41F_STEP_TRACE_DIR", "")


def _trace_jsonable(value):
    if isinstance(value, np.ndarray):
        return {"dtype": str(value.dtype), "shape": list(value.shape),
                "hex": value.tobytes().hex(), "values": value.tolist()}
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _trace_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_trace_jsonable(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(str(v) for v in value)
    return value


class _StepTrace:
    """Every step's policy inputs and decisions on this worker, on the host side.

    ``before_budget``/``after_budget`` keep the draft budget's inputs (the stale
    confidence snapshot the budget reads, tau, the idle counter, the scheduled draft
    placeholders) and its result. ``record`` (end of ``sample_tokens``) queues
    non-blocking copies of the step's device decisions to pinned memory behind one
    event; a thread waits for the event and appends one JSON line: nothing on the
    step's path waits, and nothing runs inside a CUDA graph.
    """

    def __init__(self, directory: str):
        self.directory = directory
        self.step = 0
        self.budget = None
        self._queue = None

    def _start(self) -> None:
        import queue
        import threading

        self._queue = queue.Queue()
        threading.Thread(target=self._writer, name="step-trace", daemon=True).start()

    def before_budget(self, av, num_tokens_per_req, draft_tokens) -> None:
        slots = [av.req_states.req_id_to_index[r] for r in num_tokens_per_req]
        self.budget = {
            "req_ids": list(num_tokens_per_req),
            "slots": slots,
            "scheduled_tokens": [int(v) for v in num_tokens_per_req.values()],
            "scheduled_drafts": [len(draft_tokens.get(r, ())) for r in num_tokens_per_req],
            "stale_idx": int(av._stale_idx),
            "stale_confidences": av._stale_confidences[av._stale_idx].np[slots].copy(),
            "tau": float(av.effective_dead_rows_tau()),
            "idle_before": int(getattr(av, "_steps_without_drafts", 0)),
            "row_cost": float(av.row_cost),
            "num_computed_np": av.req_states.num_computed_tokens_np[slots].copy(),
            "prefill_len_np": av.req_states.prefill_len.np[slots].copy(),
        }

    def after_budget(self, av, num_toks) -> None:
        if self.budget is not None:
            self.budget["num_tokens"] = int(num_toks)
            self.budget["batch_budget"] = getattr(av, "_batch_budget", None)
            self.budget["idle_after"] = int(getattr(av, "_steps_without_drafts", 0))

    def record(self, runner, input_batch, scope: dict) -> None:
        if self._queue is None:
            self._start()
        n = int(input_batch.num_reqs)
        pinned = {}

        def take(name, t, rows=True):
            if isinstance(t, torch.Tensor) and t.is_cuda:
                t = t[:n] if rows and t.dim() else t
                p = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
                p.copy_(t, non_blocking=True)
                pinned[name] = p

        sampler_output = scope.get("sampler_output")
        take("sampled_token_ids", getattr(sampler_output, "sampled_token_ids", None))
        take("num_sampled", scope.get("num_sampled"))
        take("num_rejected", scope.get("num_rejected"))
        take("num_verified_draft_tokens", scope.get("num_verified_draft_tokens"))
        take("dead_row_cut", getattr(runner, "_dead_row_cut", None))
        take("draft_tokens_next", scope.get("draft_tokens_for_next_step"))
        take("confidences", scope.get("confidences"))
        take("planned_confidences", scope.get("planned_confidences"))
        speculator = getattr(runner, "speculator", None)
        take("raw_confidences", getattr(speculator, "draft_token_confidence_probs", None))
        estimator = getattr(speculator, "acceptance_estimator", None)
        if estimator is not None:
            take("est_slope", estimator.slope, rows=False)
            take("est_intercepts", estimator.intercepts, rows=False)
            take("est_counts", estimator.counts, rows=False)
            take("est_info", estimator.info, rows=False)
            take("est_grad", estimator.grad, rows=False)
        copy_drafter = getattr(runner, "copy_drafter", None)
        if copy_drafter is not None:
            take("copy_int_state", copy_drafter.int_state[input_batch.idx_mapping], rows=False)
            take("copy_float_state", copy_drafter.float_state[input_batch.idx_mapping], rows=False)
            take("copy_chosen", copy_drafter.chosen)
            take("copy_estimates", copy_drafter.estimates)
            take("copy_alloc", copy_drafter.alloc)
        event = torch.cuda.Event()
        event.record()
        av = getattr(runner, "adaptive_verification", None)
        host = {
            "step": self.step,
            "t": time.monotonic(),
            "req_ids": list(input_batch.req_ids),
            "slots": input_batch.idx_mapping_np.tolist(),
            "num_scheduled_tokens": input_batch.num_scheduled_tokens.tolist(),
            "num_draft_tokens_per_req": (None if input_batch.num_draft_tokens_per_req is None
                                         else input_batch.num_draft_tokens_per_req.tolist()),
            "num_tokens": int(input_batch.num_tokens),
            "num_tokens_after_padding": int(input_batch.num_tokens_after_padding),
            "is_prefilling": input_batch.is_prefilling_np.tolist(),
            "num_spec_tokens_to_schedule": scope.get("num_spec_tokens_to_schedule"),
            "finished_req_ids": sorted(scope.get("finished_req_ids") or ()),
            "est_refits": getattr(estimator, "_refits", None),
            "est_steps_since_refit": getattr(estimator, "_steps_since_refit", None),
            "tau": float(av.effective_dead_rows_tau()) if av is not None else None,
            "idle": int(getattr(av, "_steps_without_drafts", 0)) if av is not None else None,
            "budget": self.budget,
        }
        self.budget = None
        self.step += 1
        self._queue.put((event, host, pinned))

    def _writer(self) -> None:
        import json
        import socket

        try:
            rank = get_tensor_model_parallel_rank()
        except Exception:  # noqa: BLE001
            rank = -1
        os.makedirs(self.directory, exist_ok=True)
        path = os.path.join(self.directory, f"steps-{socket.gethostname()}-tp{rank}-pid{os.getpid()}-"
                                            f"{int(time.time())}.jsonl")
        with open(path, "a") as f:
            while True:
                event, host, pinned = self._queue.get()
                event.synchronize()
                host["device"] = {k: _trace_jsonable(v.numpy() if v.dtype != torch.bfloat16
                                                     else v.view(torch.int16).numpy())
                                  for k, v in pinned.items()}
                f.write(json.dumps(_trace_jsonable(host), separators=(",", ":")) + "\n")
                f.flush()


_STEP_TRACE = _StepTrace(_STEP_TRACE_DIR) if _STEP_TRACE_DIR else None
if _STEP_TRACE is not None:
    logger.warning("debug step trace on: every step's policy inputs to %s", _STEP_TRACE_DIR)

from vllm.distributed.device_communicators.cuda_communicator import (  # noqa: E402
    DRAFT_CTX_TRACE as _DRAFT_CTX_TRACE,
    DRAFT_QUERY_TRACE as _DRAFT_QUERY_TRACE,
    ROW_TRACE as _ROW_TRACE,
)

_DRAFT_TRACES = tuple(t for t in (_DRAFT_CTX_TRACE, _DRAFT_QUERY_TRACE) if t is not None)
'''

IMPORT_ROW_TRACE = """

# Debug row trace taps (boot-differences overlay): None unless SPARK_DS41F_ROW_TRACE_DIR.
from vllm.distributed.device_communicators.cuda_communicator import (  # noqa: E402
    ROW_TRACE as _ROW_TRACE,
)
"""

ATTN_TOPK_ANCHOR = '''        if pending_aux is not None:
            torch.cuda.current_stream(q.device).wait_stream(pending_aux)
        if getattr(self, "n_local_heads", 1) == 0:
            return
'''
ATTN_TOPK = '''        if pending_aux is not None:
            torch.cuda.current_stream(q.device).wait_stream(pending_aux)
        if _ROW_TRACE is not None and self.indexer is not None:
            # The step's top-k selection, after both streams joined.
            _ROW_TRACE.record(6, self.topk_indices_buffer[:rows], self.topk_indices_buffer[:rows])
        if getattr(self, "n_local_heads", 1) == 0:
            return
'''
ATTN_OUT_ANCHOR = '''        return self._o_proj(output, sp_rows=sp_rows)
'''
ATTN_OUT = '''        if _ROW_TRACE is not None:
            _ROW_TRACE.record(7, output, output)
        return self._o_proj(output, sp_rows=sp_rows)
'''
COMPRESSOR_TAP = """

# Debug row trace taps (boot-differences overlay): None unless SPARK_DS41F_ROW_TRACE_DIR.
from vllm.distributed.device_communicators.cuda_communicator import (  # noqa: E402
    ROW_TRACE as _ROW_TRACE,
)

if _ROW_TRACE is not None:
    _untraced_forward_tilelang = DeepseekCompressor._forward_tilelang

    def _traced_forward_tilelang(self, rows, *args, **kwargs):
        _untraced_forward_tilelang(self, rows, *args, **kwargs)
        n = int(rows)
        values = kwargs.get("values")
        values = self._values if values is None else values
        _ROW_TRACE.record(4, values[:n], self._latent[:n])
        if self.compress_ratio == 2:
            # The first request's ring carry (its odd first row's even predecessor).
            _ROW_TRACE.record(5, self._pending_values[:1], self._pending_gates[:1])

    DeepseekCompressor._forward_tilelang = _traced_forward_tilelang
"""
MOE_TAP = '\n\n# ---------------------------------------------------------------------------\n# MoE taps (boot-differences overlay): off unless SPARK_DS41F_ROW_TRACE_DIR is set and\n# SPARK_DS41F_ROW_TRACE_MOE is "all" or a decoder-layer range "a-b". For each traced\n# MoE layer, in order: 20 the router\'s top-k ids -> weights, 24 the MoE input\'s MXFP8\n# cast (values -> scales, what the routed GEMM reads), 21 the shared expert\'s input ->\n# output, 22 the MoE input -> this rank\'s output (routed plus shared), 23 that output\n# again at the final reduction\'s entry (before the pre-reduce hook and the all-reduce).\nimport os as _bd_os  # noqa: E402\n\nfrom vllm.distributed.device_communicators.cuda_communicator import (  # noqa: E402\n    ROW_TRACE as _ROW_TRACE,\n)\n\n_BD_MOE = _bd_os.environ.get("SPARK_DS41F_ROW_TRACE_MOE", "")\n\nif _ROW_TRACE is not None and _BD_MOE:\n    if _BD_MOE == "all":\n        _BD_MOE_LAYERS = None\n    else:\n        _lo, _, _hi = _BD_MOE.partition("-")\n        _BD_MOE_LAYERS = range(int(_lo), int(_hi or _lo) + 1)\n    _untraced_apply_quant_method = MoERunner._apply_quant_method\n    _untraced_reduce_final_output = MoERunner._maybe_reduce_final_output\n\n    def _bd_moe_traced(runner) -> bool:\n        traced = runner.__dict__.get("_bd_moe_traced")\n        if traced is None:\n            try:\n                layer = runner.layer_id\n            except Exception:  # noqa: BLE001 - a name without a layer index\n                layer = None\n            traced = _BD_MOE_LAYERS is None or (layer is not None and layer in _BD_MOE_LAYERS)\n            runner.__dict__["_bd_moe_traced"] = traced\n        return traced\n\n    def _bd_trace_router(router) -> None:\n        if router is None or getattr(router, "_bd_select_traced", False):\n            return\n        untraced_select = router.select_experts\n\n        def select_experts(*args, **kwargs):\n            topk_weights, topk_ids = untraced_select(*args, **kwargs)\n            _ROW_TRACE.record(20, topk_ids, topk_weights)\n            return topk_weights, topk_ids\n\n        router.select_experts = select_experts\n        router._bd_select_traced = True\n\n    def _traced_apply_quant_method(self, hidden_states, router_logits, shared_experts_input,\n                                   input_ids=None, shared_experts_overlapping=False):\n        if not _bd_moe_traced(self):\n            return _untraced_apply_quant_method(self, hidden_states, router_logits,\n                                                shared_experts_input, input_ids,\n                                                shared_experts_overlapping)\n        _bd_trace_router(self.router)\n        shared_output, fused_out = _untraced_apply_quant_method(\n            self, hidden_states, router_logits, shared_experts_input, input_ids,\n            shared_experts_overlapping)\n        try:\n            from vllm.model_executor.layers.fused_moe.tilelang import _provided_mxfp8\n\n            q = _provided_mxfp8(hidden_states)\n            _ROW_TRACE.record(24, q.values, q.scales)\n        except Exception:  # noqa: BLE001 - no provided cast (another backend)\n            pass\n        if isinstance(shared_output, torch.Tensor) and isinstance(shared_experts_input, torch.Tensor):\n            _ROW_TRACE.record(21, shared_experts_input, shared_output)\n        if isinstance(fused_out, torch.Tensor):\n            _ROW_TRACE.record(22, hidden_states, fused_out)\n        return shared_output, fused_out\n\n    def _traced_reduce_final_output(self, states, trunc_size, output_is_reduced=None, *,\n                                    in_place=False):\n        if isinstance(states, torch.Tensor) and _bd_moe_traced(self):\n            _ROW_TRACE.record(23, states, states)\n        return _untraced_reduce_final_output(self, states, trunc_size, output_is_reduced,\n                                             in_place=in_place)\n\n    MoERunner._apply_quant_method = _traced_apply_quant_method\n    MoERunner._maybe_reduce_final_output = _traced_reduce_final_output\n'

ENGRAM_TAP = """

# Debug row trace taps (boot-differences overlay): None unless SPARK_DS41F_ROW_TRACE_DIR.
from vllm.distributed.device_communicators.cuda_communicator import (  # noqa: E402
    ROW_TRACE as _ROW_TRACE,
)

if _ROW_TRACE is not None:
    _untraced_engram_forward = Engram.forward

    def _traced_engram_forward(self, hidden_states, *args, **kwargs):
        out = _untraced_engram_forward(self, hidden_states, *args, **kwargs)
        n = int(out.shape[0]) if out.dim() else 0
        staged = getattr(self, "staged_rows", None)
        rows = staged[:n] if isinstance(staged, torch.Tensor) and staged.shape[0] >= n else out
        _ROW_TRACE.record(8, rows, out)
        return out

    Engram.forward = _traced_engram_forward
"""

SCHED_RETURN_OUT_ANCHOR = '''            controller.dispatch(
                scheduler_output.compute_service_class,
                contended=scheduler_output.compute_contention,
            )
        return scheduler_output
'''
SCHED_RETURN_OUT = '''            controller.dispatch(
                scheduler_output.compute_service_class,
                contended=scheduler_output.compute_contention,
            )
        if _SCHED_TRACE is not None:
            _SCHED_TRACE.scheduled(self, scheduler_output, throttle_prefills)
        return scheduler_output
'''
SCHED_UPDATE_ANCHOR = '''            eco.scheduler_stats = stats

        return engine_core_outputs
'''
SCHED_UPDATE = '''            eco.scheduler_stats = stats

        if _SCHED_TRACE is not None:
            _SCHED_TRACE.updated(self, scheduler_output, model_runner_output)
        return engine_core_outputs
'''
SCHED_TRACE = r'''

# ---------------------------------------------------------------------------
# Debug step trace, engine-core side (boot-differences overlay). Off unless
# SPARK_DS41F_STEP_TRACE_DIR is set.
import os as _st_os  # noqa: E402


class _SchedulerTrace:
    """Each schedule() decision with its inputs, and what update_from_output got
    back, one JSON line each, written in the engine core's own thread (a few
    hundred bytes a step)."""

    def __init__(self, directory: str):
        self.directory = directory
        self.file = None
        self.step = 0

    def _write(self, record: dict) -> None:
        import json
        import socket

        if self.file is None:
            _st_os.makedirs(self.directory, exist_ok=True)
            path = _st_os.path.join(self.directory, f"sched-{socket.gethostname()}-pid{_st_os.getpid()}-"
                                                    f"{int(time.time())}.jsonl")
            self.file = open(path, "a")  # noqa: SIM115
        self.file.write(json.dumps(record, separators=(",", ":"), default=str) + "\n")
        self.file.flush()

    def scheduled(self, scheduler, out, throttle_prefills) -> None:
        running = {}
        for request in list(scheduler.running):
            running[request.request_id] = [
                int(request.num_computed_tokens), int(request.num_output_tokens),
                int(getattr(request, "copy_draft_alloc", 0) or 0), len(request.spec_token_ids or ()),
                int(request.num_prompt_tokens),
            ]
        self._write({
            "kind": "schedule", "step": self.step, "t": time.monotonic(),
            "num_scheduled_tokens": dict(out.num_scheduled_tokens),
            "spec_tokens": {k: len(v) for k, v in (out.scheduled_spec_decode_tokens or {}).items()},
            "num_spec_tokens_to_schedule": out.num_spec_tokens_to_schedule,
            "skip_drafts": out.skip_drafts,
            "new": [r.req_id for r in out.scheduled_new_reqs],
            "finished": sorted(out.finished_req_ids),
            "running": running, "waiting": len(scheduler.waiting),
            "throttle_prefills": bool(throttle_prefills),
        })
        self.step += 1

    def updated(self, scheduler, out, model_runner_output) -> None:
        sampled = model_runner_output.sampled_token_ids or []
        index = model_runner_output.req_id_to_index or {}
        self._write({
            "kind": "update", "t": time.monotonic(),
            "num_scheduled_tokens": dict(out.num_scheduled_tokens),
            "generated": {r: len(sampled[i]) for r, i in index.items() if i < len(sampled)},
            "copy_draft_alloc": getattr(model_runner_output, "copy_draft_alloc", None),
            "num_verified_draft_tokens": getattr(model_runner_output, "num_verified_draft_tokens", None),
        })


_SCHED_TRACE = (_SchedulerTrace(_st_os.environ["SPARK_DS41F_STEP_TRACE_DIR"])
                if _st_os.environ.get("SPARK_DS41F_STEP_TRACE_DIR") else None)
'''


DFLASH_ENTRY_ANCHOR = """        num_reqs = input_batch.num_reqs
        num_target_tokens = input_batch.num_tokens
        num_query_tokens = num_reqs * self.num_query_per_req
"""
DFLASH_ENTRY = """        _bd_sync()
        num_reqs = input_batch.num_reqs
        num_target_tokens = input_batch.num_tokens
        num_query_tokens = num_reqs * self.num_query_per_req
"""
DFLASH_CTXGRAPH_ANCHOR = """            and self._context_preparer.can_run(num_target_tokens)
        )
"""
DFLASH_CTXGRAPH = """            and self._context_preparer.can_run(num_target_tokens)
        )
        if _BD_DSPARK_EAGER in ("context", "all"):
            use_context_graph = False
"""
DFLASH_BEFORE_CTX_ANCHOR = """        if use_context_graph:
            assert self._context_preparer is not None
            self._context_preparer.run(aux_hidden_states, num_target_tokens)
"""
DFLASH_BEFORE_CTX = """        _bd_sync()
        if use_context_graph:
            assert self._context_preparer is not None
            self._context_preparer.run(aux_hidden_states, num_target_tokens)
"""
DFLASH_AFTER_CTX_ANCHOR = """        if num_speculative_tokens == 0:
            # No drafts are needed: the context K/V has been stored above to
"""
DFLASH_AFTER_CTX = """        _bd_sync()
        if num_speculative_tokens == 0:
            # No drafts are needed: the context K/V has been stored above to
"""
DFLASH_EAGER_ANCHOR = """            need_eager=is_profile,
"""
DFLASH_EAGER = """            need_eager=is_profile or _BD_DSPARK_EAGER in ("query", "all"),
"""
DFLASH_RETURN_ANCHOR = """        return self.draft_tokens[:num_reqs]


# The Triton reference's BLOCK_SIZE"""
DFLASH_RETURN = """        _bd_sync()
        return self.draft_tokens[:num_reqs]


# The Triton reference's BLOCK_SIZE"""
DFLASH_TAIL = '''

# ---------------------------------------------------------------------------
# Drafter arms (boot-differences overlay), one variable each, off when unset:
# SPARK_DS41F_DSPARK_SYNC=1 synchronizes the device at the drafter's entry, before and
# after its context preparation and after its query forward; SPARK_DS41F_DSPARK_EAGER=
# context|query|all runs those eagerly instead of replaying their CUDA graphs.
import os as _bd_os  # noqa: E402

_BD_DSPARK_SYNC = _bd_os.environ.get("SPARK_DS41F_DSPARK_SYNC") == "1"
_BD_DSPARK_EAGER = _bd_os.environ.get("SPARK_DS41F_DSPARK_EAGER", "")
if _BD_DSPARK_EAGER not in ("", "context", "query", "all"):
    raise ValueError("SPARK_DS41F_DSPARK_EAGER must be context, query or all")
if _BD_DSPARK_SYNC or _BD_DSPARK_EAGER:
    logger.warning("drafter arm: sync=%s eager=%s", _BD_DSPARK_SYNC, _BD_DSPARK_EAGER or "off")


def _bd_sync() -> None:
    if _BD_DSPARK_SYNC and not torch.cuda.is_current_stream_capturing():
        torch.cuda.synchronize()
'''
DSPARK_TAIL = '''

# ---------------------------------------------------------------------------
# Drafter taps (boot-differences overlay): off unless SPARK_DS41F_ROW_TRACE_DIR and
# SPARK_DS41F_ROW_TRACE_DRAFTER=1. Kinds: 10 the target's auxiliary states copied in
# (context graph), 11 the combined context states the KV projections read, 12 the
# query forward's head hidden state, 13 the confidence head's hidden input and its
# output, 14 its Markov embeddings, 15 each draft layer's raw context KV, 16 its
# slots, 17 the confidence rows' request slots, 18 the estimator's slope and
# intercepts; collectives and attention outputs inside these forwards record with
# their own kinds (1-3, 7).
from vllm.distributed.device_communicators.cuda_communicator import (  # noqa: E402
    DRAFT_CTX_TRACE as _DRAFT_CTX_TRACE,
    DRAFT_QUERY_TRACE as _DRAFT_QUERY_TRACE,
)

if _DRAFT_CTX_TRACE is not None and _DRAFT_QUERY_TRACE is not None:
    _untraced_context_forward = DSparkContextCudaGraphs._forward
    _untraced_store_context = DSparkDeepseekV4Model.precompute_and_store_context_kv
    _untraced_draft_forward = DSparkDeepseekV4Model.forward
    _untraced_confidence = DSparkDeepseekV4ForCausalLM.compute_and_calibrate_confidence

    def _traced_context_forward(self, capacity):
        _DRAFT_CTX_TRACE.begin(None, self.positions[:capacity])
        _DRAFT_CTX_TRACE.record(10, self.aux[:capacity], self.aux[:capacity])
        _untraced_context_forward(self, capacity)
        _DRAFT_CTX_TRACE.end()

    def _traced_store_context(self, main_x, context_positions, context_slot_mappings=None):
        began = not _DRAFT_CTX_TRACE.active
        if began:  # the eager context path (the graph path began in _forward)
            _DRAFT_CTX_TRACE.begin(None, context_positions)
        _DRAFT_CTX_TRACE.record(11, main_x, main_x)
        for layer in self.layers:
            _wrap_context_insert(layer.attn)
        _untraced_store_context(self, main_x, context_positions, context_slot_mappings)
        if began:
            _DRAFT_CTX_TRACE.end()

    def _traced_draft_forward(self, input_ids, positions, inputs_embeds=None):
        _DRAFT_QUERY_TRACE.begin(input_ids, positions)
        out = _untraced_draft_forward(self, input_ids, positions, inputs_embeds)
        _DRAFT_QUERY_TRACE.record(12, out, out)
        return out

    def _wrap_context_insert(attn):
        # Each draft layer's context KV (before its norm, RoPE and quantization) and
        # the slots it is written to.
        if getattr(attn, "_bd_traced_insert", False) or not hasattr(attn, "insert_raw_context_kv"):
            return
        untraced = attn.insert_raw_context_kv

        def traced(kv_raw, *args, **kwargs):
            _DRAFT_CTX_TRACE.record(15, kv_raw, kv_raw)
            slots = kwargs.get("slot_mapping", args[-1] if args else None)
            if isinstance(slots, torch.Tensor):
                _DRAFT_CTX_TRACE.record(16, slots, slots)
            return untraced(kv_raw, *args, **kwargs)

        attn.insert_raw_context_kv = traced
        attn._bd_traced_insert = True

    def _traced_confidence(self, head_hidden, markov_embed, out, estimator, idx_mapping):
        probs = _untraced_confidence(self, head_hidden, markov_embed, out, estimator, idx_mapping)
        _DRAFT_QUERY_TRACE.record(13, head_hidden, probs)
        _DRAFT_QUERY_TRACE.record(14, markov_embed, markov_embed)
        if isinstance(idx_mapping, torch.Tensor):
            _DRAFT_QUERY_TRACE.record(17, idx_mapping, idx_mapping)
        slope = getattr(estimator, "slope", None)
        intercepts = getattr(estimator, "intercepts", None)
        if isinstance(slope, torch.Tensor) and isinstance(intercepts, torch.Tensor):
            _DRAFT_QUERY_TRACE.record(18, slope.reshape(1, -1), intercepts.reshape(1, -1))
        _DRAFT_QUERY_TRACE.end()
        return probs

    DSparkContextCudaGraphs._forward = _traced_context_forward
    DSparkDeepseekV4Model.precompute_and_store_context_kv = _traced_store_context
    DSparkDeepseekV4Model.forward = _traced_draft_forward
    DSparkDeepseekV4ForCausalLM.compute_and_calibrate_confidence = _traced_confidence
'''


def replace_once(text: str, old: str, new: str, what: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{what}: anchor found {count} times, expected once")
    return text.replace(old, new)


def build(tree: Path) -> dict[str, str]:
    out = {}
    text = (tree / FILES["cuda_communicator"]).read_text()
    out[FILES["cuda_communicator"]] = text.rstrip("\n") + "\n" + ROW_TRACE

    text = (tree / FILES["model"]).read_text()
    text = replace_once(text, MODEL_IMPORT_ANCHOR, MODEL_IMPORT + MODEL_IMPORT_ANCHOR, "model import")
    text = replace_once(text, MODEL_FORWARD_OLD, MODEL_FORWARD_NEW, "model forward")
    out[FILES["model"]] = text

    text = (tree / FILES["model_runner"]).read_text()
    if "\nimport time\n" not in text or "\nimport numpy as np\n" not in text or "\nimport os\n" not in text:
        raise SystemExit("model_runner: expected module imports of os, time and numpy")
    if "get_tensor_model_parallel_rank" not in text:
        raise SystemExit("model_runner: expected get_tensor_model_parallel_rank in scope")
    text = replace_once(text, RUNNER_BUDGET_OLD, RUNNER_BUDGET_NEW, "runner budget")
    text = replace_once(text, RUNNER_HARVEST_ANCHOR, RUNNER_HARVEST + RUNNER_HARVEST_ANCHOR, "runner harvest")
    text = replace_once(text, RUNNER_RECORD_ANCHOR, RUNNER_RECORD + RUNNER_RECORD_ANCHOR, "runner record")
    out[FILES["model_runner"]] = text.rstrip("\n") + "\n" + STEP_TRACE

    text = (tree / FILES["scheduler"]).read_text()
    text = replace_once(text, SCHED_RETURN_OUT_ANCHOR, SCHED_RETURN_OUT, "scheduler schedule")
    text = replace_once(text, SCHED_UPDATE_ANCHOR, SCHED_UPDATE, "scheduler update")
    out[FILES["scheduler"]] = text.rstrip("\n") + "\n" + SCHED_TRACE

    text = (tree / FILES["attention"]).read_text()
    text = replace_once(text, ATTN_TOPK_ANCHOR, ATTN_TOPK, "attention top-k tap")
    text = replace_once(text, ATTN_OUT_ANCHOR, ATTN_OUT, "attention output tap")
    out[FILES["attention"]] = text.rstrip("\n") + "\n" + IMPORT_ROW_TRACE

    text = (tree / FILES["compressor"]).read_text()
    if "class DeepseekCompressor(" not in text:
        raise SystemExit("compressor: DeepseekCompressor not found")
    out[FILES["compressor"]] = text.rstrip("\n") + "\n" + COMPRESSOR_TAP

    text = (tree / FILES["engram"]).read_text()
    if "\nclass Engram(nn.Module):" not in text or "\nimport torch\n" not in text:
        raise SystemExit("engram: Engram class or torch import not found")
    out[FILES["engram"]] = text.rstrip("\n") + "\n" + ENGRAM_TAP

    text = (tree / FILES["dflash"]).read_text()
    for old, new, what in ((DFLASH_ENTRY_ANCHOR, DFLASH_ENTRY, "drafter entry"),
                           (DFLASH_CTXGRAPH_ANCHOR, DFLASH_CTXGRAPH, "drafter context graph switch"),
                           (DFLASH_BEFORE_CTX_ANCHOR, DFLASH_BEFORE_CTX, "drafter before context"),
                           (DFLASH_AFTER_CTX_ANCHOR, DFLASH_AFTER_CTX, "drafter after context"),
                           (DFLASH_EAGER_ANCHOR, DFLASH_EAGER, "drafter query eager switch"),
                           (DFLASH_RETURN_ANCHOR, DFLASH_RETURN, "drafter return")):
        text = replace_once(text, old, new, what)
    out[FILES["dflash"]] = text.rstrip("\n") + "\n" + DFLASH_TAIL

    text = (tree / FILES["dspark"]).read_text()
    for name in ("class DSparkContextCudaGraphs:", "class DSparkDeepseekV4Model(nn.Module):",
                 "class DSparkDeepseekV4ForCausalLM(nn.Module):", "    def compute_and_calibrate_confidence("):
        if text.count(name) != 1:
            raise SystemExit(f"dspark: {name!r} not found once")
    out[FILES["dspark"]] = text.rstrip("\n") + "\n" + DSPARK_TAIL

    text = (tree / FILES["moe_runner"]).read_text()
    for name in ("class MoERunner(MoERunnerInterface):", "    def _apply_quant_method(",
                 "    def _maybe_reduce_final_output("):
        if text.count(name) != 1:
            raise SystemExit(f"moe_runner: {name!r} not found once")
    out[FILES["moe_runner"]] = text.rstrip("\n") + "\n" + MOE_TAP
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tree", type=Path)
    parser.add_argument("--out", type=Path, default=HERE / "overlay")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for key, rel in FILES.items():
        digest = hashlib.sha256((args.tree / rel).read_bytes()).hexdigest()
        print(f"original {rel} sha256 {digest}")
    built = build(args.tree)
    status = 0
    for rel, text in built.items():
        target = args.out / rel
        digest = hashlib.sha256(text.encode()).hexdigest()
        if args.check:
            same = target.is_file() and target.read_text() == text
            print(f"{'same' if same else 'DIFFERS'} {target} sha256 {digest}")
            status |= not same
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
            print(f"wrote {target} sha256 {digest}")
    return status


if __name__ == "__main__":
    sys.exit(main())
