#!/usr/bin/env python3
"""CPU check of the schedule benchmark's bookkeeping, without torch or a GPU.

usage: harness_check.py VLLM_TREE   (a checkout of vLLM l2-prefetch-schedule)

Runs the chain's own ``_Layer.run()`` (tests/kernels/deepseek_v41_l2_schedule_chain.py)
over the forty layers with a recording fake for every launch, so the launches come out
in capture order, then feeds them to the benchmark's own ``roles()``, ``windows()`` and
``summarize()``. Every module the two files import is a stub. Checks, per schedule, how
many fills each window gets and from which anchor, that every fill finds all its
consumers, and that every consumer comes after its fill.
"""

import importlib.util
import sys
import types
from collections import Counter
from contextlib import contextmanager
from unittest import mock

VLLM = sys.argv[1]

for name in [
    "torch", "torch.profiler", "torch.cuda",
    "tests", "tests.kernels", "tests.kernels.deepseek_v41_pdl_chains",
    "vllm", "vllm.config", "vllm.models", "vllm.models.deepseek_v4_1",
    "vllm.models.deepseek_v4_1.l2_prefetch", "vllm.models.glm5next", "vllm.models.glm5next.nvidia",
    "vllm.models.deepseek_v4_1.tilelang",
    "vllm.models.deepseek_v4_1.tilelang.linear",
    "vllm.models.deepseek_v4_1.tilelang.mhc",
    "vllm.models.deepseek_v4_1.tilelang.activation",
    "vllm.models.deepseek_v4_1.tilelang.moe",
    "vllm.models.deepseek_v4_1.tilelang.norm",
    "vllm.models.deepseek_v4_1.tilelang.wo",
]:
    sys.modules[name] = mock.MagicMock(name=name)
pdl = sys.modules["tests.kernels.deepseek_v41_pdl_chains"]
pdl.HIDDEN, pdl.Q_LATENT, pdl.QA_KV, pdl.EXPERTS, pdl.TOPK, pdl.LIMIT = 5120, 1280, (1792, 5120), 384, 6, 10.0
pdl.DEVICE = "cuda"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


chain_mod = load("tests.kernels.deepseek_v41_l2_schedule_chain",
                 f"{VLLM}/tests/kernels/deepseek_v41_l2_schedule_chain.py")
bench = load("bench", f"{VLLM}/benchmarks/kernels/benchmark_deepseek_v41_l2_schedule.py")

launched = []


def k(*names):
    launched.extend(names)


chain_mod.linear._block32_gemm = lambda *a, **kw: k("gemm_mxfp8")
chain_mod.l2_prefetch.issue = lambda plan, rows: plan is not None and k("l2_prefetch")
chain_mod.torch.cuda.current_stream = lambda: types.SimpleNamespace(wait_stream=lambda s: None)


@contextmanager
def stream(s):
    yield


chain_mod.torch.cuda.stream = stream


class Cast:
    def project(self, layer, x, out):
        k("per_token_cast", "gemm_mxfp8")


class Entry:
    def __init__(self):
        self.mx = types.SimpleNamespace(values=None, scales=None)
        self.y = self.streams = self.post = self.comb = self.pre = None

    def __call__(self, *args):
        k("mhc_project_streams", "mhc_finalize")


class BF16:
    def __call__(self, x, out, scratch):
        k("gemm_bf16_partials", "gemm_splitk_reduce")


class WOA:
    def __call__(self, *args):
        k("gemm_fp8w_bf16_partials", "gemm_splitk_reduce_cast")


class Plan:
    pass


def moe_run(plan, x, ids, weights, experts, out, workspace, shared, hooks):
    """The chain's _moe_run: the routed launches, with its hooks between them."""
    k("moe_dispatch")
    hooks.get("dispatch", lambda: None)()
    k("moe_grouped_fp8_fp4_gemm")
    hooks.get("fc1", lambda: None)()
    k("moe_swiglu_cast", "moe_grouped_fp8_fp4_gemm")
    hooks.get("fc2", lambda: None)()
    shared()
    k("moe_combine")


chain_mod._moe_run = moe_run


def fake_layer(i):
    L = chain_mod._Layer.__new__(chain_mod._Layer)
    L.index, L.rows = i, 6
    L.kind = "own" if i in chain_mod.OWN else "idx" if i in chain_mod.IDX else "swa"
    L.engram = i in chain_mod.ENGRAM
    L.attention_entry, L.ffn_entry = Entry(), Entry()
    L.casts = {n: Cast() for n in ("q_b", "down", "index_q_b", "wkv")}
    L.wo = types.SimpleNamespace(a=WOA(), b=None)
    L.gate, L.weights_proj, L.compressor = BF16(), BF16(), BF16()
    L.swiglu = lambda h, act: k("activation_swiglu_clamp")
    L.plan = Plan()
    L.norms = (lambda x, out: k("norm"), lambda x, out: k("norm"))
    L.spins = {key: 1.0 for key in chain_mod.SPINS[6]}
    L.plan_aux = types.SimpleNamespace(wait_stream=lambda s: None)
    for name in ("qa_kv", "q_b", "gate_up", "down", "index_q_b", "wkv", "qa", "qb", "qn",
                 "kvn", "wo_xq", "wo_sf", "wo_out", "logits", "h", "act", "shared_out", "out",
                 "workspace", "ids", "route_weights", "experts", "iq", "iw", "latent",
                 "engram_rows", "kv", "gate_scratch", "wproj_scratch", "comp_scratch"):
        setattr(L, name, mock.MagicMock())
    L.partials = []
    return L


layers = [fake_layer(i) for i in range(40)]
for L, nxt in zip(layers, layers[1:] + [None]):
    L.plans = {"WO": "wo", "FFN": "ffn"}
    if nxt is not None:
        L.plans["NEXT"] = "next"
        if nxt.engram:
            L.plans["ENGRAM"] = "engram"
        else:
            for mb in chain_mod.SPLITS:
                L.plans.update({f"NEXTA{mb}": "nexta", f"NEXTB{mb}": "nextb"})
    for mb in chain_mod.SPLITS:
        L.plans[f"JUNK{mb}"] = "junk"
    if L.kind == "own":
        L.plans.update(IDX="idx", IDXC="idxc", IDXQ="idxq")
fake_chain = types.SimpleNamespace(layers=layers)

for schedule in ("head", "chained", "typed", "pertype", "pertype2", "pertype3", "own", "eng",
                 "pertype2+split:dispatch:2", "pertype2+split:fc1:4", "pertype2+split:fc2:6",
                 "pertype2+junk:fc1:4", "off"):
    launched.clear()
    previous = None
    for L in layers:
        L.run(schedule, (), previous, lambda us: k("spin"))
        previous = L
    ks = [dict(name=n, ts=float(i), dur=0.5, end=i + 0.5, node=i, grid=()) for i, n in enumerate(launched)]
    bench.roles(ks, fake_chain)
    ws = bench.windows(ks, fake_chain, schedule)
    count = Counter(w["window"] for w in ws)
    missing = [(w["window"], w["layer"]) for w in ws
               if len(w["consumers"]) != len(bench.CONSUMERS[w["window"]])]
    anchors = Counter(k["role"] for k in ks if k.get("role", "").startswith("fill@"))
    print(schedule, dict(count), "consumers missing:", missing[:6], dict(anchors))
    assert not missing, (schedule, missing)
    for w in ws:
        # every consumer starts after its fill was issued
        assert all(c["ts"] > w["start"] for c in w["consumers"]), (schedule, w["window"], w["layer"])

traced = {}
for schedule in ("head", "typed"):
    launched.clear()
    previous = None
    for L in layers:
        L.run(schedule, (), previous, lambda us: k("spin"))
        previous = L
    ks = [dict(name=n, ts=float(i), dur=0.5, end=i + 0.5, node=i, grid=()) for i, n in enumerate(launched)]
    bench.roles(ks, fake_chain)
    traced[schedule] = [ks, [dict(x) for x in ks]]
floor = {r: 0.25 for r in ("WO-A", "WO-B", "FFN mHC", "router", "shared gate/up", "shared down",
                           "attention mHC", "fused Q-A/KV", "Q-B", "compressor", "index Q-B",
                           "head weights", "engram")}
win, con, named = bench.summarize(traced, fake_chain, floor)
print(len(win), "window rows;", len(con), "consumer rows;", len(named), "role rows")
print(list(win.items())[:2])
print(list(con.items())[:2])
