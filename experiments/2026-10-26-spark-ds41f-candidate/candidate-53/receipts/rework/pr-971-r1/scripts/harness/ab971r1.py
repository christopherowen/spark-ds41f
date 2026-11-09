#!/usr/bin/env python3
"""Image A/B for #971 round 1 (candidate-53 rework), policy-driven. Sections 1-3 are the
candidate-53 own-mark harness ab971.py unchanged (its docstring follows); sections 4-6 are new
and run after the timing, so the timing runs exactly as in ab971.py.

4. assign_draft_budget equivalence (the kept reallocate_drafts change): for one request,
   reallocate_drafts now sets capacities = draft_budget and skips assign_draft_budget; the
   image's assign_draft_budget kernel is run for one request (16 scheduled drafts, budgets
   1..16, three confidence rows) and its admitted count recorded.
5. Dead-row identity grid (the removed part): the image's dead_row_args and
   combine_sampled_and_draft_tokens over tau mode (recipe: the module's _DEAD_ROWS_TAU,
   uncalibrated; recipe_calibrated: calibrated_confidence=True, effective tau
   min(tau, 2 row_cost / (1 + 3 row_cost)); zero: 0.0, the PR's make_manager), step width
   (5 = DSpark step, 16 = wide copy step), drafts admitted or not, and three draft rows
   (shrunk copy: 5 live then -1 at confidence 0; full copy: 16 live at 0.95; DSpark low:
   5 live whose survival crosses 0.2 at the 3rd); each case's DeadRows (None or log_tau),
   cut, is_padding, input ids, logits indices and verified count are recorded.
6. declare_step_kernels' dead_rows flag (the removed model_runner change), with the
   image's dead_rows_enabled (module tau) and with it forced False, copy drafter absent
   and present, built as tests/v1/worker/test_prepare_kernels.py builds the runner.

(ab971.py:) Image A/B for #971 (candidate-53 own marks), policy-driven: each image's own code decides
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

# 4. assign_draft_budget equivalence for one request (kept: reallocate_drafts' num_reqs == 1 path)
conf_rows = {
    "flat_095": [0.95] * 16,
    "shrunk_copy": [0.85, 0.80, 0.75, 0.70, 0.65] + [0.0] * 11,
    "mixed": [0.9, 0.99, 0.5, 1.0, 0.7, 0.3, 0.95, 0.2, 0.8, 0.6, 0.1, 0.4, 0.9, 0.05, 0.5, 0.0],
}
for name, row in conf_rows.items():
    probs = torch.tensor([row], dtype=torch.float32, device=dev)
    admitted = []
    for budget in range(1, 17):
        caps = torch.tensor([16], dtype=torch.int32, device=dev)
        tilelang_inputs.assign_draft_budget(probs, torch.tensor([0], dtype=torch.int64, device=dev), caps, budget, 16)
        torch.cuda.synchronize()
        admitted.append(int(caps[0].item()))
    emit(kind="assign_draft_budget_one_request", confidences=name, budgets=list(range(1, 17)), admitted=admitted,
         equals_budget=admitted == list(range(1, 17)))

# 5. dead-row identity grid (removed: the zero-confidence floor in dead_row_args)
drafts_rows = {
    "shrunk_copy": ([11, 12, 13, 14, 15] + [-1] * 11, [0.85, 0.80, 0.75, 0.70, 0.65] + [0.0] * 11),
    "full_copy": ([21 + i for i in range(16)], [0.95] * 16),
    "dspark_low": ([31, 32, 33, 34, 35] + [-1] * 11, [0.6, 0.4, 0.3, 0.2, 0.1] + [0.0] * 11),
}
module_tau = float(getattr(avm, "_DEAD_ROWS_TAU", 0.0))
for tau_mode in ("recipe", "recipe_calibrated", "zero"):
    for width in (5, 16):
        for admitted_flag in (True, False):
            for dname, (toks, confs) in drafts_rows.items():
                m = make_manager(np.full((1, 16), 0.95, dtype=np.float32), 0.0 if tau_mode == "zero" else module_tau)
                m.calibrated_confidence = tau_mode == "recipe_calibrated"
                m.step_steps = width
                m._has_admitted_drafts = admitted_flag
                ndrafts = min(width, 15)
                rows = 1 + ndrafts
                g_rows = bucket(rows)
                m._confidence_probs = torch.tensor([confs], dtype=torch.float32, device=dev)
                m._batch_draft_capacity = torch.tensor([ndrafts], dtype=torch.int32, device=dev)
                m._dead_row_cut = torch.zeros(1, dtype=torch.int32, device=dev)
                is_padding = torch.zeros(g_rows, dtype=torch.bool, device=dev)
                tilelang_inputs.padding_rows(is_padding, rows, g_rows)
                dead = m.dead_row_args(is_padding, 1)
                draft_tokens = torch.tensor([toks], dtype=torch.int64, device=dev)
                input_ids = torch.zeros(24, dtype=torch.int32, device=dev)
                logits_indices = torch.zeros(rows, dtype=torch.int64, device=dev)
                tilelang_inputs.combine_sampled_and_draft_tokens(
                    input_ids, torch.tensor([0], dtype=torch.int32, device=dev),
                    torch.tensor([999], dtype=torch.int64, device=dev), torch.tensor([0, rows], dtype=torch.int32, device=dev),
                    torch.tensor([100 + rows], dtype=torch.int32, device=dev), torch.tensor([100], dtype=torch.int32, device=dev),
                    draft_tokens[:, :width], torch.tensor([0, rows], dtype=torch.int32, device=dev),
                    logits_indices, 1, dead_rows=dead)
                verified = m.get_verified_draft_counts(1, dead.cut if dead is not None else None)
                torch.cuda.synchronize()
                emit(kind="dead_row_case", tau_mode=tau_mode, effective_tau=round(float(m.effective_dead_rows_tau()), 6),
                     width=width, admitted=admitted_flag, drafts=dname, rows=rows, graph_rows=g_rows,
                     dead_row_args=dead is not None, log_tau=(dead.log_tau if dead is not None else None),
                     cut=(int(dead.cut[0].item()) if dead is not None else None),
                     unmasked_rows=int((~is_padding[:rows]).sum().item()),
                     is_padding=[int(x) for x in is_padding.tolist()], input_ids=input_ids[:rows].tolist(),
                     logits_indices=logits_indices.tolist(), verified=int(verified[0].item()))

# 6. declare_step_kernels' dead_rows flag (removed: copy drafter declares dead rows)
from unittest import mock
import vllm.v1.core.kv_cache_utils as kv_cache_utils
import vllm.v1.worker.gpu.model_runner as runner_module
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler
declared = []
head = VocabParallelEmbedding.__new__(VocabParallelEmbedding); head.num_embeddings_padded = 129408
processor = LogitsProcessor.__new__(LogitsProcessor); processor.org_vocab_size = 129280; processor.head_dtype = torch.bfloat16
language_model = SimpleNamespace(lm_head=head, logits_processor=processor)
def declare(copy_drafter, enabled_override):
    patches = [mock.patch.object(kv_cache_utils, "get_kv_cache_groups", lambda *_: [])]
    for name in ("add_kv_sharing_layers_to_kv_cache_groups", "get_shared_kv_cache_layers", "select_kernel_block_sizes",
                 "declare_input_batch_kernels", "register_input_batch_warmups"):
        patches.append(mock.patch.object(runner_module, name, lambda *_, **__: None))
    patches += [mock.patch.object(runner_module.envs, "VLLM_MOE_SKIP_PADDING", False),
                mock.patch.object(tilelang_inputs, "declare", lambda calls: None),
                mock.patch.object(Sampler, "register_warmups", lambda self, **_: None),
                mock.patch.object(RejectionSampler, "register_warmups", lambda self, **kw: declared.append(kw))]
    if enabled_override is not None:
        patches.append(mock.patch.object(runner_module, "dead_rows_enabled", lambda: enabled_override))
    for pt in patches: pt.start()
    try:
        steps = 5
        runner = SimpleNamespace(
            vllm_config=SimpleNamespace(use_request_boundary_checkpoints=False), get_kv_cache_spec=dict,
            _register_kv_cache_warmups=lambda *_: None,
            speculator=SimpleNamespace(enable_adaptive_verification=True, draft_logits=None),
            adaptive_verification=None, copy_drafter=copy_drafter, batch_sharder=None, num_speculative_steps=steps,
            draft_tokens_handler=SimpleNamespace(track_consumed_drafts=False),
            req_states=SimpleNamespace(next_prefill_tokens=torch.zeros(4, dtype=torch.int32),
                                       draft_tokens=torch.zeros(16, steps, dtype=torch.int32), register_warmups=lambda: None),
            model_state=SimpleNamespace(num_new_sampled_tokens_per_step=1), is_last_pp_rank=True, is_pooling_model=False,
            sampler=Sampler.__new__(Sampler), rejection_sampler=RejectionSampler.__new__(RejectionSampler),
            model_config=SimpleNamespace(get_hidden_size=lambda: 4096), dtype=torch.bfloat16,
            _language_model=lambda: language_model, prompt_logprobs_worker=None, structured_outputs_worker=None,
            sink=None, _greedy_verification_available=lambda: False)
        runner_module.GPUModelRunner.declare_step_kernels(runner)
        return declared.pop()["dead_rows"]
    finally:
        for pt in reversed(patches): pt.stop()
for override in (None, False):
    for cd_name, cd in (("absent", None), ("present", SimpleNamespace(register_warmups=lambda: None))):
        try:
            flag = declare(cd, override)
            emit(kind="declare_dead_rows", dead_rows_enabled=("module" if override is None else "forced_False"),
                 module_dead_rows_enabled=bool(avm.dead_rows_enabled()), copy_drafter=cd_name, dead_rows=bool(flag))
        except Exception as e:  # recorded, not hidden
            emit(kind="declare_dead_rows", dead_rows_enabled=("module" if override is None else "forced_False"),
                 copy_drafter=cd_name, error=f"{type(e).__name__}: {e}")
fh.close()
