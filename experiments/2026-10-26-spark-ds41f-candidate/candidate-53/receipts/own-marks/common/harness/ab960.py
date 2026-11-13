#!/usr/bin/env python3
"""Image A/B for #960 (candidate-53 own marks): the FULL-decode pre-launch step on the path
each image's code takes, same workload as the PR's screen_cudagraph_wins.py section 1
(23 DeepseekV41B12xMetadataBuilder groups + CEDState, shapes (1,1)..(16,64)).

- base image (CEDState has no stage_floors): the eager path the base runner takes on a FULL
  step: build_slot_mappings_by_layer (90-layer dict) + CEDState.stage + 23 builder.build()
  + graph replay. Recorded twice: "eager" as the base code runs it, and "eager_pr" with the
  PR's run_baseline's two extra hand-made pinned tensors (its in-process baseline).
- candidate image (stage_floors exists): stage_floors + replay of a graph that holds
  CEDState.prepare_cudagraph_replay + the 23 builders' prepare_cudagraph_replay; also the
  eager path with the candidate's CEDState.stage (its ring buffer), and a variant whose
  floors change every step (the H2D copy is then not skipped).

Timers, per path and shape:
- wall_us: host+device time per step, 300 back-to-back steps, one sync at the end (the PR's
  method), 7 windows after 100 warm-up steps (well over 100 ms);
- device_us: device time per step with the host hidden: a sleep kernel (3e6 cycles, its length
  recorded) is queued first,
  so every launch of the step is enqueued before the GPU reaches the start event; median of
  101 steps.
- digest: SHA-256 of the metadata buffers after the path (same inputs on both images).
"""
import argparse, hashlib, json, statistics, time
import torch
from vllm.models.deepseek_v4_1.ced import CED_WINDOW, CEDState
from vllm.models.deepseek_v4_1.sparse_mla import DeepseekV41B12xMetadataBuilder
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import CircularBufferSpec, FullAttentionSpec
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
HAS_NEW = hasattr(CEDState, "stage_floors")
max_reqs, max_tokens = 16, 2048

class SC: max_num_batched_tokens = max_tokens; max_num_seqs = max_reqs
class Cfg:
    scheduler_config = SC()
    speculative_config = type("S", (), {"num_speculative_tokens": 5, "num_verification_tokens": 5,
                              "parallel_drafting": False, "use_eagle": lambda self=None: True})()
    cache_config = type("C", (), {"block_size": 64})()
    model_config = type("M", (), {"max_model_len": 4096})()
    parallel_config = type("P", (), {"decode_context_parallel_size": 1})()
    compilation_config = type("Comp", (), {"cudagraph_mode": None})()
cfg = Cfg()
builders = []
for i in range(23):
    if i % 3 == 0:
        spec = object.__new__(CircularBufferSpec); tps = 1
    else:
        spec = object.__new__(FullAttentionSpec); tps = 1 if i % 3 == 1 else 2
    object.__setattr__(spec, "block_size", 64); object.__setattr__(spec, "tokens_per_state", tps)
    builders.append(DeepseekV41B12xMetadataBuilder(spec, [f"layer_{i*3}", f"layer_{i*3+1}"], cfg, dev))
ced = CEDState(max_reqs, max_tokens, (64, 128, 256, 512, 1024, 2048), dev)
ced.warmup(5120, 4, pipeline=False, sink_columns=False)
qsl_g = torch.zeros(max_reqs + 1, dtype=torch.int32, device=dev)
qsl_c = torch.zeros(max_reqs + 1, dtype=torch.int32)
sl_g = torch.zeros(max_reqs, dtype=torch.int32, device=dev)
sl_c = torch.zeros(max_reqs, dtype=torch.int32)
gen = torch.Generator(device="cpu").manual_seed(960)
block_tables = [torch.randint(1, 100, (max_reqs, 64), dtype=torch.int32, generator=gen).to(dev) for _ in range(23)]
slot_mappings = torch.randint(1, 4000, (26, max_tokens), dtype=torch.int64, generator=gen).to(dev)
names = [[f"model.layers.{g*3+k}.self_attn.attn" for k in range(4)] for g in range(23)]
def build_slot_mappings_90(sm):
    d = {}
    for g, ns in enumerate(names):
        for n in ns: d[n] = sm[g]
    return d
dw = torch.randn((512, 512), dtype=torch.bfloat16, device=dev)
dx = torch.randn((64, 512), dtype=torch.bfloat16, device=dev)
do = torch.empty((64, 512), dtype=torch.bfloat16, device=dev)
torch.mm(dx, dw, out=do)
if HAS_NEW:
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        ced.prepare_cudagraph_replay(qsl_g, sl_g, 1)
        for gi, b in enumerate(builders):
            b.prepare_cudagraph_replay(qsl_g[:2], sl_g[:1], block_tables[gi][:1], slot_mappings[gi, :1], 1, 1)
        torch.mm(dx, dw, out=do)
    del g
torch.cuda.synchronize()

def meta_digest(nr, nt):
    h = hashlib.sha256()
    h.update(ced.replay_start[:nr].cpu().numpy().tobytes())
    for b in builders:
        for t in (b.starts, b.request_positions, b.counts, b.positions[:nt], b.reqs[:nt], b.slots[:nt], b.lengths[:nt]):
            h.update(t.cpu().numpy().tobytes())
    return h.hexdigest()[:16]
def zero_meta():
    for b in builders:
        for t in (b.starts, b.request_positions, b.counts, b.positions, b.reqs, b.slots, b.lengths): t.zero_()
    ced.replay_start.zero_()

def wall(fn, windows=7, iters=300, warm=100):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(windows):
        t0 = time.perf_counter()
        for _ in range(iters): fn()
        torch.cuda.synchronize()
        out.append((time.perf_counter() - t0) / iters * 1e6)
    return out
SLEEP_CYCLES = 3_000_000
def sleep_us():
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    v = []
    for _ in range(5):
        s.record(); torch.cuda._sleep(SLEEP_CYCLES); e.record(); e.synchronize(); v.append(s.elapsed_time(e) * 1000.0)
    return statistics.median(v)
def device(fn, n=101, warm=20):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    out, host = [], []
    for _ in range(n):
        torch.cuda._sleep(SLEEP_CYCLES)  # queued first: the step's launches are all enqueued before it ends
        t0 = time.perf_counter(); s.record(); fn(); e.record(); host.append((time.perf_counter() - t0) * 1e6)
        e.synchronize()
        out.append(s.elapsed_time(e) * 1000.0)
    return out, host

shapes = [(1, 1), (1, 6), (2, 12), (3, 18), (4, 16), (4, 24), (8, 32), (8, 48), (16, 64)]
fh = open(a.out, "w")
def emit(**r):
    r = {"label": a.label, "has_stage_floors": HAS_NEW, **r}
    fh.write(json.dumps(r) + "\n"); fh.flush(); print(json.dumps(r), flush=True)
emit(kind="identity", gpu=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda, sleep_kernel_us=round(sleep_us(), 1))
for nr, nt in shapes:
    q = nt // nr
    for r in range(nr + 1): qsl_c[r] = r * q
    qsl_g[: nr + 1].copy_(qsl_c[: nr + 1])
    for r in range(nr): sl_c[r] = 1024 + r * 17
    sl_g[:nr].copy_(sl_c[:nr])
    keep_full = [False] * nr; floors = [0] * nr; limits = [CED_WINDOW] * nr
    g_base = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_base):
        torch.mm(dx, dw, out=do)
    torch.cuda.synchronize()
    def eager(extra_pinned=False):
        _ = build_slot_mappings_90(slot_mappings[:, :nt])
        if extra_pinned:
            _ = torch.tensor(floors, dtype=torch.int64, pin_memory=True)
            _ = torch.tensor(limits, dtype=torch.int32, pin_memory=True)
        ced.stage(qsl_g[: nr + 1], sl_g[:nr], [q] * nr, keep_full, floors, limits)
        qc, qg, sg = qsl_c[: nr + 1], qsl_g[: nr + 1], sl_g[:nr]
        md = {}
        for gi, b in enumerate(builders):
            cm = CommonAttentionMetadata(query_start_loc=qg, query_start_loc_cpu=qc, seq_lens=sg, num_reqs=nr,
                                         num_actual_tokens=nt, max_query_len=q, max_seq_len=4096,
                                         block_table_tensor=block_tables[gi][:nr], slot_mapping=slot_mappings[gi, :nt], causal=True)
            m = b.build(0, cm)
            m.swa_replay_start = ced.replay_start
            for ln in names[gi]: md[ln] = m
        g_base.replay()
    paths = {"eager": lambda: eager(False), "eager_pr": lambda: eager(True)}
    zero_meta(); eager(False); torch.cuda.synchronize(); ref = meta_digest(nr, nt)
    digests = {"eager": ref}
    if HAS_NEW:
        zero_meta()
        g_opt = torch.cuda.CUDAGraph()
        ced.stage_floors(floors)
        with torch.cuda.graph(g_opt):
            ced.prepare_cudagraph_replay(qsl_g, sl_g, nr)
            for gi, b in enumerate(builders):
                b.prepare_cudagraph_replay(qsl_g[: nr + 1], sl_g[:nr], block_tables[gi][:nr], slot_mappings[gi, :nt], nr, nt)
            torch.mm(dx, dw, out=do)
        torch.cuda.synchronize()
        zero_meta()
        def cand():
            ced.stage_floors(floors); g_opt.replay()
        flip = [0]
        def cand_floors_change():
            flip[0] ^= 1
            ced.stage_floors([flip[0]] * nr); g_opt.replay()
        cand(); torch.cuda.synchronize(); digests["graph"] = meta_digest(nr, nt)
        paths["graph"] = cand; paths["graph_floors_change"] = cand_floors_change
    for name, fn in paths.items():
        w = wall(fn); d, hq = device(fn)
        if name == "graph_floors_change":
            ced.stage_floors(floors); torch.cuda.synchronize()
        emit(kind="step", num_reqs=nr, num_tokens=nt, path=name,
             wall_us_windows=[round(x, 2) for x in w], wall_us_median=round(statistics.median(w), 2),
             device_us_median=round(statistics.median(d), 2), device_us_p10=round(sorted(d)[10], 2),
             device_us_p90=round(sorted(d)[90], 2), host_enqueue_us_max=round(max(hq), 1), digest=digests.get(name))
    emit(kind="bit_exact", num_reqs=nr, num_tokens=nt, digests=digests,
         graph_equals_eager=(digests.get("graph") == ref) if HAS_NEW else None)
fh.close()
