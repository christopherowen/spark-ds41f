#!/usr/bin/env python3
"""Image A/B for #971 (candidate-53 own marks), policy-driven: each image's own code decides
the step, then the PR's 58-layer step workload (screen_copy_drafter_smooth.py section 3: one
routed MoE layer + one dense projection GEMM per CUDA-graph replay, x58) times that step.

1. copy_apply gate (the image's kernel): does it admit an offered 2/3/4/8-token copy at the
   0.95 prior against DSpark confidences [0.70, 0.65, 0.60, 0.55, 0.50]?
2. AdaptiveVerificationManager (the image's class, built as the PR's make_manager builds it;
   single_req_tile_spill_scale set from the module constant when the module defines one):
   c1 initial copy step at 0.95 and 0.97, c2 at 0.95; then the post-reject shrunk step
   (5 live drafts, the rest -1 at confidence 0) through the image's dead_row_args and
   combine_sampled_and_draft_tokens. Two dead-row taus: "recipe" = the module's
   _DEAD_ROWS_TAU (from SPARK_DS41F_DSPARK_DEAD_ROWS_TAU, 0.2 in config/cluster.json for every
   shape) and "pr" = 0.0 (the PR's make_manager sets dead_rows_tau = 0.0).
3. Timing: the four step shapes (graph rows, live routed rows) in {(24,17), (16,16), (24,6),
   (16,6)} at TP3 (inter 768) and TP4 (inter 576), warm and 128 MiB wash, on both images
   (the kernels are the same code on both); 50 warm-up replays (>100 ms) then 40 timed.
"""
import argparse, json, statistics, time
from types import SimpleNamespace
import numpy as np
import torch
import vllm  # noqa: F401
from vllm.models.deepseek_v4_1.tilelang import gemm, linear, moe
from vllm.v1.worker.gpu import tilelang_inputs
import vllm.v1.worker.gpu.spec_decode.adaptive_verification as avm
from vllm.v1.worker.gpu.spec_decode.adaptive_verification import AdaptiveVerificationManager
from vllm.v1.worker.gpu.spec_decode.tilelang_copy import copy_alloc, copy_apply

p = argparse.ArgumentParser()
p.add_argument("--label", required=True)
p.add_argument("--out", required=True)
p.add_argument("--warm", type=int, default=50)
p.add_argument("--repeats", type=int, default=40)
a = p.parse_args()
fh = open(a.out, "w")
def emit(**r):
    r = {"label": a.label, **r}
    fh.write(json.dumps(r) + "\n"); fh.flush(); print(json.dumps(r), flush=True)
dev = torch.device("cuda")
flush = torch.zeros(128 * 1024 * 1024 // 4, device=dev, dtype=torch.float32)
SPILL = getattr(avm, "SINGLE_REQ_TILE_SPILL_SCALE", None)
emit(kind="identity", gpu=torch.cuda.get_device_name(), torch=torch.__version__,
     module_DEAD_ROWS_TAU=getattr(avm, "_DEAD_ROWS_TAU", None), module_SINGLE_REQ_TILE_SPILL_SCALE=SPILL,
     module_HIGH_ROW_COST_SCALE=getattr(avm, "HIGH_ROW_COST_SCALE", None))

# 1. copy_apply gate (as the PR's section 1, without its asserts)
block, lookahead, initial, max_reqs, max_len = 5, 16, 8, 8, 512
history = torch.full((max_reqs, max_len), 42, dtype=torch.int32, device=dev)
total_len = torch.zeros(max_reqs, dtype=torch.int32, device=dev)
idx = torch.tensor([0], dtype=torch.int64, device=dev)
zero = torch.zeros(1, dtype=torch.int32, device=dev)
dspark_ids = torch.tensor([[101, 102, 103, 104, 105]], dtype=torch.int64, device=dev)
dspark_confs = torch.tensor([[0.70, 0.65, 0.60, 0.55, 0.50]], dtype=torch.float32, device=dev)
for offered in (2, 3, 4, 8):
    total_len[0] = 100 + offered
    sources = torch.full((1, 32, 5), -1, dtype=torch.int32, device=dev); sources[0, 0, :] = offered
    int_state = torch.zeros((max_reqs, 2), dtype=torch.int32, device=dev)
    float_state = torch.zeros((max_reqs, 2), dtype=torch.float32, device=dev)
    chosen = torch.zeros((1, 3), dtype=torch.int32, device=dev)
    estimates = torch.zeros(1, dtype=torch.float32, device=dev)
    drafts = torch.full((1, lookahead), -1, dtype=torch.int64, device=dev)
    conf = torch.zeros((1, lookahead), dtype=torch.float32, device=dev)
    planned = torch.zeros((1, lookahead), dtype=torch.float32, device=dev)
    alloc = torch.zeros(1, dtype=torch.int32, device=dev)
    copy_alloc(idx, total_len, sources, zero, zero, zero, int_state, float_state, chosen, estimates, alloc,
               min_match=16, lookahead=lookahead, block=block, initial=initial)
    copy_apply(idx, history, chosen, estimates, int_state, dspark_ids, dspark_confs, drafts, conf, planned)
    emit(kind="gate_copy_apply", offered=min(offered, initial),
         admits_copy=int(int_state[0, 1].item()) == min(offered, initial), drafts_head=drafts[0, :min(offered, initial)].tolist())

# 2. policy
def make_manager(confidences, tau, row_cost=0.035):
    num_reqs, num_steps = confidences.shape
    m = AdaptiveVerificationManager.__new__(AdaptiveVerificationManager)
    m.num_speculative_steps = num_steps; m.num_draft_steps = 5; m.step_steps = num_steps
    m._stale_confidences = [SimpleNamespace(np=confidences.copy())]; m._stale_idx = 0
    m.req_states = SimpleNamespace(req_id_to_index={f"r{i}": i for i in range(num_reqs)},
                                   num_computed_tokens_np=np.ones(num_reqs, dtype=np.int32),
                                   prefill_len=SimpleNamespace(np=np.ones(num_reqs, dtype=np.int32)))
    m._max_total_logits = 1 << 30; m.num_bonus_tokens = 1; m.row_cost = row_cost; m.dead_rows_tau = tau
    if SPILL is not None:
        m.single_req_tile_spill_scale = SPILL
    return m
bucket = lambda rows: 16 if rows <= 16 else 24
policy = {}
for tau_name, tau in (("recipe", float(getattr(avm, "_DEAD_ROWS_TAU", 0.0))), ("pr", 0.0)):
    res = {"tau": tau}
    for name, c, nreq in (("c1_095", 0.95, 1), ("c1_097", 0.97, 1), ("c2_095", 0.95, 2)):
        m = make_manager(np.full((nreq, 16), c, dtype=np.float32), tau)
        toks = m.get_num_tokens({f"r{i}": 17 for i in range(nreq)}, {f"r{i}": list(range(16)) for i in range(nreq)})
        res[name] = {"tokens": int(toks), "draft_budget": int(m._batch_budget[2])}
    budget = res["c1_095"]["draft_budget"]; rows = 1 + budget
    m = make_manager(np.full((1, 16), 0.95, dtype=np.float32), tau)
    m.get_num_tokens({"r0": 17}, {"r0": list(range(16))})
    conf_gpu = torch.zeros((1, 16), dtype=torch.float32, device=dev)
    conf_gpu[0, :5] = torch.tensor([0.85, 0.80, 0.75, 0.70, 0.65], device=dev)
    m._confidence_probs = conf_gpu
    m._batch_draft_capacity = torch.tensor([budget], dtype=torch.int32, device=dev)
    m._dead_row_cut = torch.zeros(1, dtype=torch.int32, device=dev)
    m._has_admitted_drafts = True; m.step_steps = 16
    g_rows = bucket(rows)
    is_padding = torch.zeros(g_rows, dtype=torch.bool, device=dev)
    tilelang_inputs.padding_rows(is_padding, rows, g_rows)
    dead = m.dead_row_args(is_padding, 1)
    draft_tokens = torch.full((1, 16), -1, dtype=torch.int64, device=dev)
    draft_tokens[0, :5] = torch.tensor([11, 12, 13, 14, 15], device=dev)
    tilelang_inputs.combine_sampled_and_draft_tokens(
        torch.zeros(24, dtype=torch.int32, device=dev), torch.tensor([0], dtype=torch.int32, device=dev),
        torch.tensor([999], dtype=torch.int64, device=dev), torch.tensor([0, rows], dtype=torch.int32, device=dev),
        torch.tensor([100 + rows], dtype=torch.int32, device=dev), torch.tensor([100], dtype=torch.int32, device=dev),
        draft_tokens, torch.tensor([0, rows], dtype=torch.int32, device=dev),
        torch.zeros(rows, dtype=torch.int64, device=dev), 1, dead_rows=dead)
    torch.cuda.synchronize()
    unmasked = int((~is_padding[:rows]).sum().item())
    res["shrunk_step"] = {"rows": rows, "graph_rows": g_rows, "dead_row_args": dead is not None,
                          "dead_cut": int(dead.cut[0].item()) if dead is not None else None,
                          "log_tau": (dead.log_tau if dead is not None else None), "unmasked_rows": unmasked}
    res["initial_step"] = {"rows": rows, "graph_rows": g_rows, "live_rows": rows}
    policy[tau_name] = res
    emit(kind="policy", tau_name=tau_name, **res)

# 3. timing (the PR's section 3 builder; same shapes on both images)
def timings(call, wash):
    for _ in range(5): call()
    torch.cuda.synchronize()
    b = torch.cuda.Event(enable_timing=True, external=True); e = torch.cuda.Event(enable_timing=True, external=True)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        if wash is not None: wash.add_(1.0)
        b.record(); call(); e.record()
    for _ in range(a.warm): g.replay()
    torch.cuda.synchronize()
    out = []
    for _ in range(a.repeats):
        g.replay(); torch.cuda.synchronize(); out.append(b.elapsed_time(e))
    return out
gen = torch.Generator(device=dev).manual_seed(20261108)
hidden, E, topk, L = 5120, 384, 6, 58
for world, inter in ((3, 768), (4, 576)):
    dense_inter = 18432 // world
    plan = moe.MoeExpertsV41Plan(E, topk, hidden, inter, swiglu_limit=10.0, out_dtype=torch.float32,
                                 idx_dtypes=(torch.int64,), max_rows=96)
    for n, k in linear._launch(plan.kernels()).items(): setattr(plan, n, k)
    for d in plan._dispatches(): d.bind()
    r8 = lambda shape, lo, hi: torch.randint(lo, hi, shape, generator=gen, device=dev, dtype=torch.uint8)
    mw = plan.prepare_weights(r8((E, 2 * inter, hidden // 2), 0, 255), r8((E, 2 * inter, hidden // 32), 124, 128),
                              r8((E, hidden, inter // 2), 0, 255), r8((E, hidden, inter // 32), 124, 128))
    dg = {16: gemm.mxfp8_gemm(dense_inter, hidden, block_M=16, block_N=64, block_K=128, num_stages=2, threads=128,
                              shards=1, groups=1, padded_rows=True, tile_contiguous=False),
          32: gemm.mxfp8_gemm(dense_inter, hidden, block_M=32, block_N=64, block_K=128, num_stages=2, threads=128,
                              shards=1, groups=1, padded_rows=True, tile_contiguous=False)}
    dw = r8((dense_inter, hidden), 0, 120).view(torch.float8_e4m3fn)
    dsf = r8((dense_inter // 32, 4 * gemm.scale_words(hidden)), 125, 128).view(torch.uint32)
    def runner(graph_rows, live):
        xr = SimpleNamespace(values=r8((graph_rows, hidden), 0, 120).view(torch.float8_e4m3fn),
                             scales=r8((graph_rows, 4 * gemm.scale_words(hidden)), 125, 128))
        ti = torch.full((graph_rows, topk), -1, dtype=torch.int64, device=dev)
        for r in range(live):
            ti[r] = torch.tensor([(r * 7 + k * 13) % E for k in range(topk)], dtype=torch.int64, device=dev)
        tw = torch.zeros((graph_rows, topk), dtype=torch.float32, device=dev); tw[:live] = 1.0 / topk
        sh = torch.zeros((graph_rows, hidden), dtype=torch.bfloat16, device=dev)
        mo = torch.empty((graph_rows, hidden), dtype=torch.bfloat16, device=dev)
        ws = torch.empty(moe.workspace_bytes(plan.workspace_specs(graph_rows, torch.int64)), dtype=torch.uint8, device=dev)
        dr = 16 if graph_rows <= 16 else 32
        dx = r8((dr, hidden), 0, 120).view(torch.float8_e4m3fn)
        dxsf = r8((dr, 4 * gemm.scale_words(hidden)), 125, 128).view(torch.uint32)
        do = torch.empty((dr, dense_inter), dtype=torch.bfloat16, device=dev)
        def run():
            plan.run(xr, ti, tw, mw, mo, workspace=ws, shared=sh)
            dg[dr](dx, dw, dxsf, dsf, do)
        return run
    shapes = [(24, 17), (16, 16), (24, 6), (16, 6)]
    if a.label.endswith("2"):
        shapes = list(reversed(shapes))
    fns = {s: runner(*s) for s in shapes}
    for cache_mode, wash in (("warm", None), ("128MiB_wash", flush)):
        for s in shapes:
            t = timings(fns[s], wash)
            emit(kind="timing_step", world=world, inter=inter, cache_mode=cache_mode, graph_rows=s[0], live_routed_rows=s[1],
                 mean_layer_ms=round(statistics.mean(t), 5), median_layer_ms=round(statistics.median(t), 5),
                 stdev_layer_ms=round(statistics.stdev(t), 5), est_58layer_step_ms=round(statistics.mean(t) * L, 3),
                 samples_ms=[round(x, 5) for x in t])
    del mw, dw, dsf, fns
    torch.cuda.empty_cache()
fh.close()
