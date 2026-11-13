#!/usr/bin/env python3
"""One tp3-switched rank's engine startup on one GB10 (rank 0 of 3).

The real vLLM engine startup (EngineCore in-process through the external
launcher, the 1107 recipe's tp3-switched arguments and environment) with
torch.distributed's fake process group standing in for ranks 1 and 2. Every
shape, weight shard, allocation and startup step is rank 0's; collectives
return without exchanging data, so model outputs are not meaningful (nothing
here reads them). What the fake world cannot do is replaced, the same in every
arm:
  - sparknet's one-shot RoCE all-reduce is not built (no fabric), so the TP
    CudaCommunicator takes its PyNCCL/torch.distributed path;
  - PyNCCL is a stand-in with its interface that moves no data (received and
    gathered buffers hold this rank's data or zeros);
  - the shared-memory broadcaster is off, B12X preparation decides on this rank
    alone, and the startup rendezvous with the other ranks returns at once.

Records each startup phase's wall time; the count and host time, by phase, of
gc.collect, trim_host_heap (mi_collect and malloc_trim apart), empty_cache,
synchronize, the UMA pressure release, reclaim_uma_memory and the TileLang
release_host_memory; MemAvailable every 0.25 s from a separate sampler process;
the process's VmRSS and the CUDA allocator's state at readiness.
"""

import argparse
import functools
import gc
import json
import os
import subprocess
import sys
import time

T0 = time.perf_counter()
WALL0 = time.time()

ap = argparse.ArgumentParser()
ap.add_argument("--label", required=True)
ap.add_argument("--out", required=True, help="output directory")
ap.add_argument("--serve-args", required=True, help="JSON list of serve arguments")
ap.add_argument("--settle-s", type=float, default=5.0)
opts = ap.parse_args()
os.makedirs(opts.out, exist_ok=True)

# MemAvailable sampler in its own process (no GIL contention with the engine).
sampler_path = os.path.join(opts.out, f"{opts.label}.memavail.txt")
sampler = subprocess.Popen(
    [
        sys.executable,
        "-c",
        "import time,sys\n"
        "out=open(sys.argv[1],'w',buffering=1)\n"
        "while True:\n"
        "    t=time.time()\n"
        "    with open('/proc/meminfo') as f:\n"
        "        d={l.split(':')[0]:int(l.split()[1]) for l in f}\n"
        "    out.write('%.3f %d %d %d\\n'%(t,d['MemAvailable'],d['MemFree'],d['Cached']))\n"
        "    time.sleep(0.25)\n",
        sampler_path,
    ]
)

# ---------------------------------------------------------------- fake world
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from torch.testing._internal.distributed import fake_pg  # noqa: E402,F401

_orig_new_group = dist.new_group


def _fake_new_group(*args, **kwargs):
    kwargs["backend"] = "fake"
    return _orig_new_group(*args, **kwargs)


dist.new_group = _fake_new_group
dist.distributed_c10d.new_group = _fake_new_group
dist.init_process_group("fake", store=dist.HashStore(), rank=0, world_size=3)

import contextlib  # noqa: E402

import sparknet.integration.vllm as _sparknet_vllm  # noqa: E402


class _NoFabricOneShot:
    """sparknet's one-shot all-reduce without a fabric: reports itself disabled,
    so the TP CudaCommunicator takes its torch.distributed path (fake group)."""

    disabled = True
    backend_name = "none (one rank, fake group)"
    all_reduce_capacity_bytes = int(
        os.environ.get("SPARKNET_ROCE_ALLREDUCE_CAPACITY_BYTES", "2097152"))

    def __init__(self, *args, **kwargs):
        pass

    def should_custom_ar(self, *args, **kwargs):
        return False

    def failure(self):
        return None

    def capture(self, *args, **kwargs):
        return contextlib.nullcontext()


_sparknet_vllm.SparknetOneShotAllReduce = _NoFabricOneShot


class _FakeGroupNccl:
    """PyNCCL's interface on the fake group: no transfer. Received and gathered
    buffers hold this rank's data or zeros, so later kernels see finite values."""

    available = True
    disabled = False

    def __init__(self, group, device):
        self.group = group
        self.world_size = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        self.device = device

    def all_reduce(self, in_tensor, out_tensor=None, op=None, stream=None):
        out = torch.empty_like(in_tensor) if out_tensor is None else out_tensor
        out.copy_(in_tensor)
        return out

    def all_gather(self, output_tensor, input_tensor, stream=None):
        try:
            output_tensor.view(self.world_size, *input_tensor.shape).copy_(
                input_tensor.unsqueeze(0))
        except RuntimeError:
            output_tensor.zero_()

    def reduce_scatter(self, output_tensor, input_tensor, op=None, stream=None):
        try:
            output_tensor.copy_(
                input_tensor.reshape(self.world_size, *output_tensor.shape)[self.rank])
        except RuntimeError:
            output_tensor.zero_()

    def all_gatherv(self, output_tensor, *args, **kwargs):
        output_tensor.zero_()

    def reduce_scatterv(self, output_tensor, *args, **kwargs):
        output_tensor.zero_()

    def recv(self, tensor, src, stream=None):
        tensor.zero_()

    def all_to_all_rows(self, output, input_, rows, stream=None, *, copy_self=True):
        output.zero_()

    def __getattr__(self, name):
        # send, broadcast, reduce, scatter, group_start/end, windows, suspend,
        # resume, batch_isend_irecv, destroy: nothing to do on one rank.
        if name.startswith("__"):
            raise AttributeError(name)
        return lambda *args, **kwargs: None


import vllm.distributed.device_communicators.cuda_communicator as _cuda_comm  # noqa: E402

_cuda_comm._acquire_pynccl = lambda group, device: (_FakeGroupNccl(group, device), None, True)

import vllm.distributed.parallel_state as _ps  # noqa: E402

_gc_init = _ps.GroupCoordinator.__init__


def _gc_init_without_mq(self, *args, **kwargs):
    # The shared-memory broadcaster waits for the other ranks' readers.
    kwargs["use_message_queue_broadcaster"] = False
    return _gc_init(self, *args, **kwargs)


_ps.GroupCoordinator.__init__ = _gc_init_without_mq

import vllm.v1.worker.b12x_startup as _b12x_startup  # noqa: E402

_coord_init = _b12x_startup.B12xPreparationCoordinator.__init__


def _coord_init_local(self, *args, **kwargs):
    # B12X preparation exchanges each round with the other ranks through the
    # store; one rank decides alone.
    kwargs["process_local_only"] = True
    return _coord_init(self, *args, **kwargs)


_b12x_startup.B12xPreparationCoordinator.__init__ = _coord_init_local

import faulthandler  # noqa: E402
import signal  # noqa: E402

faulthandler.register(signal.SIGUSR1, all_threads=True)
faulthandler.dump_traceback_later(3000, exit=True)

# ------------------------------------------------------------ instrumentation
STACK: list[str] = ["engine"]
PHASES: list[dict] = []  # {"name", "start", "end", "depth"}
OPS: dict = {}  # (phase, op) -> {"n", "s", "max", "fired", "fired_s"}
EVENTS: list[dict] = []  # first 400 timed op calls over 50 ms


_PAGE = os.sysconf("SC_PAGE_SIZE")
RSS_OPS = {"mi_collect", "malloc_trim", "trim_host_heap", "tilelang.release_host_memory",
           "reclaim_uma_memory"}


def _rss():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * _PAGE


def _rec(op, dt, fired=None, rss_freed=None):
    key = (STACK[-1], op)
    r = OPS.setdefault(key, {"n": 0, "s": 0.0, "max": 0.0, "fired": 0, "fired_s": 0.0,
                             "rss_freed": 0})
    r["n"] += 1
    r["s"] += dt
    r["max"] = max(r["max"], dt)
    if fired:
        r["fired"] += 1
        r["fired_s"] += dt
    if rss_freed is not None:
        r["rss_freed"] += rss_freed
    if (dt >= 0.05 or rss_freed) and len(EVENTS) < 400:
        EVENTS.append(
            {"t": round(time.perf_counter() - T0, 3), "phase": STACK[-1], "op": op,
             "s": round(dt, 4), "fired": fired,
             "rss_freed_mib": None if rss_freed is None else round(rss_freed / 2**20, 1)}
        )


def timed_op(name, fn):
    if getattr(fn, "_harness_wrapped", False):
        return fn
    track_rss = name in RSS_OPS

    @functools.wraps(fn)
    def w(*a, **k):
        rss0 = _rss() if track_rss else None
        t = time.perf_counter()
        try:
            return (r := fn(*a, **k))
        finally:
            dt = time.perf_counter() - t
            fired = None
            if name == "release_device_memory_under_pressure":
                fired = bool(locals().get("r"))
            _rec(name, dt, fired, None if rss0 is None else rss0 - _rss())

    w._harness_wrapped = True
    return w


def phase(name, fn):
    @functools.wraps(fn)
    def w(*a, **k):
        STACK.append(name)
        rec = {"name": name, "start": time.perf_counter() - T0, "depth": len(STACK) - 1}
        PHASES.append(rec)
        try:
            return fn(*a, **k)
        finally:
            rec["end"] = time.perf_counter() - T0
            STACK.pop()

    return w


gc.collect = timed_op("gc.collect", gc.collect)
torch.accelerator.empty_cache = timed_op("empty_cache", torch.accelerator.empty_cache)
torch.cuda.empty_cache = timed_op("empty_cache", torch.cuda.empty_cache)
torch.accelerator.synchronize = timed_op("synchronize", torch.accelerator.synchronize)


def patch_everywhere(orig, new):
    for mname, mod in list(sys.modules.items()):
        if mod is None or not mname.startswith("vllm"):
            continue
        d = getattr(mod, "__dict__", None)
        if not d:
            continue
        for k, v in list(d.items()):
            if v is orig:
                setattr(mod, k, new)


import vllm.utils.mem_utils as mem_utils  # noqa: E402
import vllm.utils.gc_utils  # noqa: E402,F401
import vllm.model_executor.model_loader.utils as loader_utils  # noqa: E402
import vllm.model_executor.model_loader.base_loader  # noqa: E402,F401
import vllm.model_executor.model_loader.default_loader  # noqa: E402,F401
import vllm.model_executor.model_loader.direct_loader  # noqa: E402,F401
import vllm.v1.worker.gpu_worker as gpu_worker  # noqa: E402
import vllm.v1.worker.gpu.model_runner as v2_runner  # noqa: E402
import vllm.v1.worker.gpu.cudagraph_utils  # noqa: E402,F401
import vllm.models.deepseek_v4_1.tilelang.prepare as tl_prepare  # noqa: E402

ARM_HAS_TRIM = hasattr(mem_utils, "trim_host_heap")
for fname in ("trim_host_heap", "reclaim_uma_memory", "release_device_memory_under_pressure"):
    if hasattr(mem_utils, fname):
        orig = getattr(mem_utils, fname)
        patch_everywhere(orig, timed_op(fname, orig))
for fname, op in (("_get_c10_mi_collect", "mi_collect"), ("_get_malloc_trim", "malloc_trim")):
    if hasattr(mem_utils, fname):
        real = getattr(mem_utils, fname)()
        if real is not None:
            wrapped = timed_op(op, real)
            setattr(mem_utils, fname, lambda w=wrapped: w)
orig = tl_prepare.release_host_memory
tl_prepare.release_host_memory = timed_op("tilelang.release_host_memory", orig)

pwal = loader_utils.process_weights_after_loading
patch_everywhere(pwal, phase("process_weights_after_loading", pwal))

W = gpu_worker.Worker
W._startup_rendezvous = lambda self, phase_name: None
_begin_b12x = W.begin_b12x_preparation


def _begin_b12x_then_settle(self, *args, **kwargs):
    # With one rank deciding alone, a stage with nothing to prepare is done at
    # once; the executor then skips run_b12x_preparation, so finish it here the
    # way that call would (advance until done).
    status = _begin_b12x(self, *args, **kwargs)
    while self._b12x_startup_coordinator is not None and status.get("done"):
        status = self.advance_b12x_preparation()
    return status


W.begin_b12x_preparation = _begin_b12x_then_settle
for meth in ("init_device", "load_model", "determine_available_memory",
             "_prepare_kernels", "initialize_from_config", "compile_or_warm_up_model"):
    setattr(W, meth, phase(meth, getattr(W, meth)))
R = v2_runner.GPUModelRunner
for meth in ("capture_model", "profile_run"):
    if hasattr(R, meth):
        setattr(R, meth, phase("runner." + meth, getattr(R, meth)))
if hasattr(gpu_worker, "warmup_kernels"):
    gpu_worker.warmup_kernels = phase("warmup_kernels", gpu_worker.warmup_kernels)
if hasattr(gpu_worker, "log_startup_memory_breakdown"):
    gpu_worker.log_startup_memory_breakdown = phase(
        "log_startup_memory_breakdown", gpu_worker.log_startup_memory_breakdown)
fgh = vllm.utils.gc_utils.freeze_gc_heap
patch_everywhere(fgh, phase("freeze_gc_heap", fgh))

# ---------------------------------------------------------------- the engine
from vllm.engine.arg_utils import EngineArgs  # noqa: E402
from vllm.utils.argparse_utils import FlexibleArgumentParser  # noqa: E402

serve = json.loads(opts.serve_args)
model = serve[0]
argv = ["--model", model]
skip_with_value = {"--host", "--port", "--tool-call-parser", "--nnodes",
                   "--distributed-executor-backend"}
skip_flag = {"--enable-auto-tool-choice", "--enable-prompt-tokens-details"}
i = 1
dropped = []
while i < len(serve):
    a = serve[i]
    if a in skip_with_value:
        dropped += serve[i:i + 2]
        i += 2
        continue
    if a in skip_flag:
        dropped.append(a)
        i += 1
        continue
    argv.append(a)
    i += 1
argv += ["--distributed-executor-backend", "external_launcher"]
parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
ns, unknown = parser.parse_known_args(argv)
engine_args = EngineArgs.from_cli_args(ns)

from vllm.v1.engine.llm_engine import LLMEngine  # noqa: E402

t_engine = time.perf_counter()
err = None
try:
    engine = LLMEngine.from_engine_args(engine_args)
except BaseException as e:  # record, then re-raise after writing
    err = repr(e)
    import traceback

    traceback.print_exc()
t_ready = time.perf_counter()

status = {}
try:
    with open("/proc/self/status") as f:
        for line in f:
            if line.split(":")[0] in ("VmRSS", "RssAnon", "RssFile", "VmHWM"):
                status[line.split(":")[0]] = int(line.split()[1])
except OSError:
    pass
cuda = {}
if torch.cuda.is_available():
    cuda = {
        "reserved_bytes": torch.cuda.memory_reserved(),
        "allocated_bytes": torch.cuda.memory_allocated(),
        "max_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
time.sleep(opts.settle_s)
sampler.terminate()


def _memavail():
    rows = []
    with open(sampler_path) as f:
        for line in f:
            p = line.split()
            if len(p) == 4:
                rows.append((float(p[0]) - WALL0, int(p[1]), int(p[2]), int(p[3])))
    return rows


rows = _memavail()
ready_s = t_ready - T0
startup = [r for r in rows if r[0] <= ready_s]
after = [r for r in rows if r[0] > ready_s]
kib = 1024 * 1024
mem = {
    "first_available_gib": round(rows[0][1] / kib, 3) if rows else None,
    "min_available_gib_startup": round(min(r[1] for r in startup) / kib, 3) if startup else None,
    "min_available_at_s": round(min(startup, key=lambda r: r[1])[0], 2) if startup else None,
    "available_at_ready_gib": round(startup[-1][1] / kib, 3) if startup else None,
    "available_settled_gib": round(sum(r[1] for r in after) / len(after) / kib, 3) if after else None,
    "samples": len(rows),
}
out = {
    "label": opts.label,
    "arm_has_trim_host_heap": ARM_HAS_TRIM,
    "error": err,
    "dropped_serve_args": dropped,
    "unknown_engine_args": unknown,
    "seconds": {"imports_and_patching": round(t_engine - T0, 2),
                "engine_from_args_to_ready": round(t_ready - t_engine, 2),
                "total_to_ready": round(ready_s, 2)},
    "phases": [
        {"name": p["name"], "depth": p["depth"], "start_s": round(p["start"], 2),
         "seconds": round(p.get("end", t_ready - T0) - p["start"], 3)}
        for p in PHASES
    ],
    "ops_by_phase": [
        {"phase": k[0], "op": k[1], "n": v["n"], "seconds": round(v["s"], 4),
         "max_s": round(v["max"], 4), "fired": v["fired"], "fired_seconds": round(v["fired_s"], 4),
         "rss_freed_mib": round(v["rss_freed"] / 2**20, 1) if k[1] in RSS_OPS else None}
        for k, v in sorted(OPS.items())
    ],
    "slow_op_events": EVENTS,
    "memavailable": mem,
    "proc_status_kib_at_ready": status,
    "cuda_at_ready": cuda,
}
with open(os.path.join(opts.out, f"{opts.label}.json"), "w") as f:
    json.dump(out, f, indent=1)
print("HARNESS_RESULT " + json.dumps({"label": opts.label, "error": err, **out["seconds"],
                                      "mem": mem}))
sys.stdout.flush()
os._exit(0 if err is None else 1)
