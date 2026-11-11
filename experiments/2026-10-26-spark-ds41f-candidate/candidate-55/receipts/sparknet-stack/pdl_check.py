"""Single-GPU PDL check of the sparknet one-shot kernels (world_size 1 and 2, no RoCE).

The collectives run against a fake region in device memory: with world_size=1
the protocol needs no peer; with world_size=2 (rank 0) peer 1's receive slots
and stripe flags are filled before each replay with the sequences the kernel
will wait for. A slow PDL producer writes the collective's input and a PDL
consumer copies its output; every buffer is poisoned before each replay.
"""
import ctypes, importlib.util, json, re, sys, time
import torch
import tilelang
import tilelang.language as T

from sparknet.oneshot import _oneshot_tilelang as AR, _allgather_tilelang as AG
from sparknet.oneshot import _reducescatter_tilelang as RS, _exchange_tilelang as EX
import sparknet
assert sparknet.__file__.startswith("/work/sparknet/"), sparknet.__file__
print("sparknet from", sparknet.__file__, flush=True)

spec = importlib.util.spec_from_file_location(
    "pdl_sources", "/opt/spark-ds41f/candidate/vllm/tests/kernels/deepseek_v41_pdl_sources.py")
pdl_sources = importlib.util.module_from_spec(spec); spec.loader.exec_module(pdl_sources)
pdl_sources.CHAIN |= {"oneshot_allreduce", "oneshot_allgather", "oneshot_reducescatter", "oneshot_exchange"}

DEV = "cuda"
THREADS_T = 128
SPIN = 20000
POISON = 0x7FC17FC1  # a bf16 NaN pair, never produced by the data below
REPLAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
results = {"sources": {}, "edges": {}, "runs": [], "early_reader": {}}

# ---------------------------------------------------------------- helper kernels
@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def slow_copy(n: int):
    @T.prim_func
    def pdl_slow_copy(src: T.Tensor((n,), T.int32), dst: T.Tensor((n,), T.int32)):
        with T.Kernel(T.ceildiv(n, THREADS_T), threads=THREADS_T) as bx:
            T.pdl_sync()
            T.pdl_trigger()
            i = bx * THREADS_T + T.get_thread_binding()
            acc = T.alloc_var(T.float32, init=T.cast(src[i], T.float32))
            for _ in T.serial(SPIN):
                acc = acc * 0.999999 + 1e-6
            dst[i] = T.if_then_else(acc == 12345.5, 7, src[i])
    return pdl_slow_copy

@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def reader(n: int, early: bool):
    @T.prim_func
    def pdl_reader(src: T.Tensor((n,), T.int32), dst: T.Tensor((n,), T.int32)):
        with T.Kernel(T.ceildiv(n, THREADS_T), threads=THREADS_T) as bx:
            T.pdl_sync()
            T.pdl_trigger()
            i = bx * THREADS_T + T.get_thread_binding()
            dst[i] = src[i]
    @T.prim_func
    def pdl_early_reader(src: T.Tensor((n,), T.int32), dst: T.Tensor((n,), T.int32)):
        with T.Kernel(T.ceildiv(n, THREADS_T), threads=THREADS_T) as bx:
            i = bx * THREADS_T + T.get_thread_binding()
            v = T.alloc_var(T.int32)
            v = src[i]
            T.pdl_sync()
            dst[i] = v
    return pdl_early_reader if early else pdl_reader

class _EdgeData(ctypes.Structure):
    _fields_ = [("from_port", ctypes.c_ubyte), ("to_port", ctypes.c_ubyte), ("type", ctypes.c_ubyte),
                ("reserved", ctypes.c_ubyte * 5)]

class _KernelNodeParams(ctypes.Structure):
    _fields_ = [("func", ctypes.c_void_p), ("grid", ctypes.c_uint * 3), ("block", ctypes.c_uint * 3),
                ("shared_mem", ctypes.c_uint), ("params", ctypes.c_void_p), ("extra", ctypes.c_void_p),
                ("kern", ctypes.c_void_p), ("ctx", ctypes.c_void_p)]

def edges(graph):
    cu = ctypes.CDLL("libcuda.so.1")
    h = ctypes.c_void_p(graph.raw_cuda_graph()); cnt = ctypes.c_size_t(0)
    assert cu.cuGraphGetEdges_v2(h, None, None, None, ctypes.byref(cnt)) == 0
    s = (ctypes.c_void_p * cnt.value)(); d = (ctypes.c_void_p * cnt.value)(); data = (_EdgeData * cnt.value)()
    assert cu.cuGraphGetEdges_v2(h, s, d, data, ctypes.byref(cnt)) == 0
    def name(node):
        kind = ctypes.c_int(); assert cu.cuGraphNodeGetType(ctypes.c_void_p(node), ctypes.byref(kind)) == 0
        if kind.value != 0:
            return None
        p = _KernelNodeParams(); assert cu.cuGraphKernelNodeGetParams_v2(ctypes.c_void_p(node), ctypes.byref(p)) == 0
        t = ctypes.c_char_p()
        if p.func:
            assert cu.cuFuncGetName(ctypes.byref(t), ctypes.c_void_p(p.func)) == 0
        else:
            assert cu.cuKernelGetName(ctypes.byref(t), ctypes.c_void_p(p.kern)) == 0
        return t.value.decode()
    return [(name(s[i]), name(d[i]), int(data[i].type)) for i in range(cnt.value)]

def capture(fn):
    side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()
    torch.cuda.current_stream().wait_stream(side); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g):
        fn()
    return g

# ---------------------------------------------------------------- fake region
THREADS, SLOTS, FLAG_STRIDE, HCA, GRID = 512, 2, 64, 1, 8
SLOT_BYTES = 1 << 20
CTRL_OFF, FLAG_OFF, SEND_OFF = 0, 4096, 1 << 20
RECV_OFF = SEND_OFF + SLOTS * SLOT_BYTES
REGION_BYTES = RECV_OFF + 2 * SLOTS * SLOT_BYTES
STAGE_I, TAIL_I, RUNG_I, ARR_I = 1, 2, 3, 4

class Region:
    def __init__(self):
        self.region = torch.zeros(REGION_BYTES, dtype=torch.uint8, device=DEV)
        self.counters = torch.zeros(8, dtype=torch.int32, device=DEV)
        ctrl = self.region[CTRL_OFF:CTRL_OFF + 64].view(torch.int32)
        ctrl[28 // 4] = 2_000_000  # wait timeout 2 s, then the kernel traps
        self.base = int(self.region.data_ptr())
    def epoch(self):
        return int(self.counters[0].item()) & 0xFFFFFFFF
    def flag_word(self, peer, slot, hca=0):
        off = FLAG_OFF + ((peer * SLOTS + slot) * HCA + hca) * FLAG_STRIDE
        return self.region[off:off + 4].view(torch.int32)
    def recv_bytes(self, peer, slot, nbytes):
        off = RECV_OFF + (peer * SLOTS + slot) * SLOT_BYTES
        return self.region[off:off + nbytes]
    def common(self):
        return (RECV_OFF, FLAG_OFF, SEND_OFF, CTRL_OFF, SLOT_BYTES, STAGE_I, TAIL_I, RUNG_I, ARR_I, GRID)

# ---------------------------------------------------------------- sources + audit
def first_statements(src, name, n=6):
    for fname, _, body in pdl_sources.functions(src):
        if fname.startswith(name):
            lines = [l.strip() for l in body.splitlines() if l.strip()]
            return lines[:n]
    return None

def audit(kernel, label, name):
    src = kernel.get_kernel_source()
    v = pdl_sources.violations(src, False)
    body = next(b for f, _, b in pdl_sources.functions(src) if f.startswith(name))
    trig = body.find(pdl_sources.TRIGGER)
    ctx = body[max(0, trig - 160):trig + 45]
    results["sources"][label] = {
        "violations": v, "first_statements": first_statements(src, name),
        "wait_count": body.count(pdl_sources.WAIT), "trigger_count": body.count(pdl_sources.TRIGGER),
        "trigger_context": ctx,
    }
    assert v == [], (label, v)
    return kernel

builds = {}
for world in (1, 2, 4):
    for ring4 in ((False, True) if world == 4 else (False,)):
        hca = 2 if world == 4 else HCA
        tag = f"w{world}{'r4' if ring4 else ''}"
        builds[("ar", world, ring4, False)] = audit(AR._build("bfloat16", world, 0, THREADS, SLOTS, FLAG_STRIDE, hca, ring4, False)(), f"allreduce_{tag}", "oneshot_allreduce")
        builds[("ar", world, ring4, True)] = audit(AR._build("bfloat16", world, 0, THREADS, SLOTS, FLAG_STRIDE, hca, ring4, True)(), f"allreduce_traced_{tag}", "oneshot_allreduce") if world == 4 else None
        builds[("ag", world, ring4)] = audit(AG._build(world, 0, THREADS, SLOTS, FLAG_STRIDE, hca, ring4)(), f"allgather_{tag}", "oneshot_allgather")
        builds[("rs", world, ring4)] = audit(RS._build("bfloat16", world, 0, THREADS, SLOTS, FLAG_STRIDE, hca, ring4)(), f"reducescatter_{tag}", "oneshot_reducescatter")
        builds[("ex", world, ring4)] = audit(EX._build(world, 0, THREADS, SLOTS, FLAG_STRIDE, hca, ring4)(), f"exchange_{tag}", "oneshot_exchange")
print("audit: all kernels 0 violations", flush=True)

# ---------------------------------------------------------------- chains
def bf16_words(t):
    return t.contiguous().view(torch.int32)

def run_case(kind, world, stages, early=False, replays=REPLAYS):
    """producer(src->inp) -> collective(inp->out) -> consumer(out->final), graph replays."""
    torch.manual_seed(1234 + 17 * world + stages)
    R = Region()
    stage_packs = 2048  # 32 KiB per stage and peer chunk
    total_packs = stage_packs * stages
    words_per_pack = 4
    peer_stage_data = []  # per stage: bytes peer 1 delivers (world 2)
    if kind == "ar":
        assert stages == 1
        n_in = total_packs * words_per_pack
        x0 = (torch.rand(n_in * 2, device=DEV) * 4 + 0.5).bfloat16()
        x1 = (torch.rand(n_in * 2, device=DEV) * 4 + 0.5).bfloat16()
        src = bf16_words(x0)
        expected = bf16_words(x0) if world == 1 else bf16_words((x0.float() + x1.float()).bfloat16())
        n_out = n_in
        peer_stage_data = [bf16_words(x1).view(torch.uint8)]
        kern = builds[("ar", world, False, False)]
        def collective(inp, out):
            kern(inp, out, R.counters, R.base, total_packs, total_packs * 16, *R.common(), 0)
    elif kind == "ag":
        # dim-0 gather: shard of total_packs packs, out = [shard r0, shard r1, ...]
        n_in = total_packs * words_per_pack
        src = torch.randint(0, 1 << 30, (n_in,), dtype=torch.int32, device=DEV)
        peer = torch.randint(0, 1 << 30, (n_in,), dtype=torch.int32, device=DEV)
        n_out = n_in * world
        expected = src if world == 1 else torch.cat([src, peer])
        peer_stage_data = [peer[s * stage_packs * 4:(s + 1) * stage_packs * 4].view(torch.uint8) for s in range(stages)]
        kern = builds[("ag", world, False)]
        def collective(inp, out):
            kern(inp, out, R.counters, R.base, total_packs, total_packs * 16, total_packs, total_packs,
                 *R.common(), stages, stage_packs)
    elif kind == "rs":
        # input: world chunks of total_packs packs (bf16); out = sum over ranks of chunk[rank 0]
        chunk_words = total_packs * words_per_pack
        x = (torch.rand(world * chunk_words * 2, device=DEV) * 4 + 0.5).bfloat16()
        p1 = (torch.rand(chunk_words * 2, device=DEV) * 4 + 0.5).bfloat16()  # peer 1's chunk for rank 0
        src = bf16_words(x)
        n_in = world * chunk_words
        n_out = chunk_words
        own = x[:chunk_words * 2]
        expected = bf16_words(own) if world == 1 else bf16_words((own.float() + p1.float()).bfloat16())
        pw = bf16_words(p1)
        peer_stage_data = [pw[s * stage_packs * 4:(s + 1) * stage_packs * 4].view(torch.uint8) for s in range(stages)]
        kern = builds[("rs", world, False)]
        def collective(inp, out):
            kern(inp, out, R.counters, R.base, total_packs, total_packs * 16, total_packs, *R.common(), stages, stage_packs)
    elif kind == "ex":
        chunk_words = total_packs * words_per_pack
        src = torch.randint(0, 1 << 30, (world * chunk_words,), dtype=torch.int32, device=DEV)
        p1 = torch.randint(0, 1 << 30, (chunk_words,), dtype=torch.int32, device=DEV)  # peer 1's chunk for rank 0
        n_in = world * chunk_words
        n_out = world * chunk_words
        expected = src[:chunk_words].clone() if world == 1 else torch.cat([src[:chunk_words], p1])
        peer_stage_data = [p1[s * stage_packs * 4:(s + 1) * stage_packs * 4].view(torch.uint8) for s in range(stages)]
        kern = builds[("ex", world, False)]
        def collective(inp, out):
            kern(inp, out, R.counters, R.base, total_packs, total_packs * 16, total_packs, total_packs,
                 world * total_packs, total_packs, 1, *R.common(), stages, stage_packs)
    inp = torch.empty(n_in, dtype=torch.int32, device=DEV)
    out = torch.empty(n_out, dtype=torch.int32, device=DEV)
    final = torch.empty(n_out, dtype=torch.int32, device=DEV)
    prod, cons = slow_copy(n_in), reader(n_out, early)

    def fill_peer():
        if world == 1:
            return
        e = R.epoch()
        for s in range(stages):
            seq = (e + 1 + s) & 0xFFFFFFFF
            slot = seq & (SLOTS - 1)
            d = peer_stage_data[s]
            R.recv_bytes(1, slot, d.numel()).copy_(d)
            R.flag_word(1, slot).fill_(seq if seq < (1 << 31) else seq - (1 << 32))

    def chain():
        prod(src, inp)
        collective(inp, out)
        cons(out, final)

    def poison():
        inp.fill_(POISON); out.fill_(POISON); final.fill_(POISON)

    # capture() runs one eager warm-up first; fill the peer for it
    fill_peer(); poison()
    torch.cuda.synchronize()
    g = capture_with_peer(chain, fill_peer)
    label = f"{kind}_w{world}_s{stages}{'_early' if early else ''}"
    results["edges"][label] = edges(g)
    if hasattr(g, "instantiate"):
        g.instantiate()
    stale_out = stale_final = mismatches = 0
    epochs = []
    for r in range(replays):
        fill_peer(); poison()
        e0 = R.epoch()
        g.replay(); torch.cuda.synchronize()
        e1 = R.epoch(); epochs.append(e1 - e0)
        stale_out += int((out == POISON).sum()); stale_final += int((final == POISON).sum())
        if not torch.equal(out, expected):
            mismatches += 1
        if not early:
            assert torch.equal(out, expected), (label, r)
            assert torch.equal(final, expected), (label, r)
    rec = {"case": label, "replays": replays, "epoch_advance_per_replay": sorted(set(epochs)),
           "stale_out_elems": stale_out, "stale_final_elems": stale_final, "out_mismatch_replays": mismatches,
           "edges": results["edges"][label]}
    assert set(epochs) == {stages}, (label, epochs)
    print(json.dumps(rec), flush=True)
    results["runs"].append(rec)
    return rec

def capture_with_peer(chain, fill_peer):
    side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        chain()
    torch.cuda.current_stream().wait_stream(side); torch.cuda.synchronize()
    fill_peer(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g):
        chain()
    return g

cases = [("ar", 1, 1), ("ag", 1, 1), ("ag", 1, 4), ("rs", 1, 1), ("rs", 1, 4), ("ex", 1, 1), ("ex", 1, 4),
         ("ar", 2, 1), ("ag", 2, 1), ("ag", 2, 2), ("rs", 2, 1), ("rs", 2, 2), ("ex", 2, 1), ("ex", 2, 2)]
for kind, world, stages in cases:
    run_case(kind, world, stages)
# Control: a consumer that loads before its wait sees stale output, which shows the
# collective's trigger really lets dependents launch before it completes.
for kind, world, stages in (("ar", 2, 1), ("rs", 2, 2), ("rs", 1, 4)):
    rec = run_case(kind, world, stages, early=True)
    results["early_reader"][rec["case"]] = rec["stale_final_elems"]

json.dump(results, open("/out/pdl_check.json", "w"), indent=1, default=str)
print("PDL CHECK OK", flush=True)
