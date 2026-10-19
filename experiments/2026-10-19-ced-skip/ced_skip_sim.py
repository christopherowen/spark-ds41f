#!/usr/bin/env python3
"""Fingerprint every CED target dependency under aligned row skipping.

Load the candidate policy and staging methods through AST without importing the
GPU runtime. Compare retained decoder states with the no-skip schedule and
require a fail-stop when a promised successor is shortened.
"""
from __future__ import annotations

import ast
import os
import random
import sys
from bisect import bisect_left
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any

TREE = sys.argv[1]
SOURCES = {
    "vllm/models/deepseek_v4_1/ced.py": {"CED_WINDOW", "_KEEP_ALL", "CEDPlan", "plan_ced", "nonfinal_limits"},
    "vllm/v1/attention/backends/mla/compressor_utils.py": {"_DSPARK_SWA_INDEX_ALIGNMENT", "get_dspark_swa_index_width"},
    "vllm/models/deepseek_v4_1/nvidia/model_state.py": {"_nonfinal_skip_policy", ("DeepseekV41ModelState", "_stage_ced")},
    "vllm/v1/core/sched/scheduler.py": {("Scheduler", "_aligned_prefill_chunk")},
}
UNWRITTEN = "UNWRITTEN"


def load_namespace() -> dict:
    ns = {"bisect_left": bisect_left, "dataclass": dataclass, "replace": replace, "Any": Any, "os": os,
          "envs": SimpleNamespace(VLLM_DS41_KERNEL_BACKEND="tilelang"), "cdiv": lambda a, b: -(-a // b)}
    for path, names in SOURCES.items():
        keep = []
        for node in ast.parse(open(os.path.join(TREE, path)).read()).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
                keep.append(node)
            elif isinstance(node, ast.Assign) and any(getattr(t, "id", None) in names for t in node.targets):
                keep.append(node)
            elif isinstance(node, ast.ClassDef):
                keep += [s for s in node.body if isinstance(s, ast.FunctionDef) and (node.name, s.name) in names]
        found = {getattr(n, "name", None) or n.targets[0].id for n in keep}
        missing = {n if isinstance(n, str) else n[1] for n in names} - found
        assert not missing, f"{path}: missing {missing}"
        for node in keep:
            for fn in [n for n in ast.walk(node) if isinstance(n, ast.FunctionDef)]:
                fn.returns = None
                for a in fn.args.args + fn.args.kwonlyargs:
                    a.annotation = None
        exec(compile(ast.Module(body=keep, type_ignores=[]), path, "exec"), ns)
    return ns


class FakeCEDState:
    """CEDState.stage with the _decoder_requests formula; checks plan_ced's bounds."""

    def __init__(self, ns, capacities):
        self.ns, self.capacities, self.out = ns, capacities, None

    def stage(self, starts, seq, query_lengths, keep_full, prefix_start, limits=None):
        ns, W = self.ns, self.ns["CED_WINDOW"]
        plan = ns["plan_ced"](query_lengths, keep_full, self.capacities, limits)
        unlimited = ns["plan_ced"](query_lengths, keep_full, self.capacities)
        if limits is not None:  # same decoder mode and plans as without the limit
            assert unlimited is not None and plan.max_query_len == unlimited.max_query_len
        self.out = []
        for i, q in enumerate(query_lengths):
            limit = ns["_KEEP_ALL"] if keep_full[i] else W if limits is None else limits[i]
            kept = min(int(q), limit)
            replay = max(prefix_start[i], seq[i] - kept if q > kept else 0)
            self.out.append((kept, replay))
        if plan is not None:
            assert plan.num_tokens == sum(k for k, _ in self.out)


def run_prompt(ns, cfg, T, P, H, skip, written=None, shorten=None):
    """Prefill one request from computed H to prompt end P. Returns (written, errors, floors)."""
    W = ns["CED_WINDOW"]
    policy = ns["_nonfinal_skip_policy"](cfg, aligned_chunks=True) if skip else (0, 0)
    self = SimpleNamespace(ced_state=FakeCEDState(ns, (8, W, 2 * W, 4 * W, 1 << 30)),
                           _ced_prompt_logprobs=set(), _ced_prefix_start={"r": H},
                           _ced_skip_min_rows=policy[0], _ced_chunk_align=policy[1],
                           _ced_mm_requests=set(), _ced_expected_chunk={}, _ced_batch=None)
    sched = SimpleNamespace(scheduler_config=SimpleNamespace(long_prefill_token_threshold=T))
    written = dict(written or {})
    errors, c, skipped = [], H, False
    while c < P:
        n = min(P - c, T)
        n = ns["_aligned_prefill_chunk"](sched, c, n)
        if shorten:
            n = shorten(c, n)
        batch = SimpleNamespace(num_reqs=1, req_ids=["r"], query_start_loc=[0, n], seq_lens=[c + n],
                                num_scheduled_tokens=[n], num_computed_prefill_tokens_np=[c],
                                prefill_len_np=[P], has_prefill=True)
        ns["_stage_ced"](self, batch)
        kept, replay = self.ced_state.out[0]
        skipped |= n > W and kept == 1
        for pos in range(c + n - kept, c + n):
            deps = []
            for r in range(max(pos + 1 - W, replay), pos):
                if r not in written:
                    errors.append((c, n, pos, r))
                deps.append(written.get(r, (UNWRITTEN, r)))
            written[pos] = hash((pos, tuple(deps)))
        c += n
    return written, errors, skipped


def scenario(ns, cfg, T, P, H, shorten=None, P0=None):
    W = ns["CED_WINDOW"]
    # Prefix-cache pages below H come from an earlier request from 0: the same prompt, or
    # (P0) a different prompt sharing the first H tokens.
    P0 = P0 or P
    cache_base = {p: s for p, s in run_prompt(ns, cfg, T, P0, 0, False)[0].items() if p < H}
    cache_skip = {p: s for p, s in run_prompt(ns, cfg, T, P0, 0, True)[0].items() if p < H}
    base, base_err, _ = run_prompt(ns, cfg, T, P, H, False, cache_base)
    got, got_err, floor = run_prompt(ns, cfg, T, P, H, True, cache_skip, shorten)
    # The target's decoder never reads below H (replay clamp), so the caches only matter to the drafter.
    target = [P - 1] + list(range(max(P + 1 - W, H), P))
    target_diff = [p for p in target if base.get(p, UNWRITTEN) != got.get(p, UNWRITTEN)]
    target_unwritten = [p for p in target if p not in got]
    draft = range(max(P - W, 0), P)
    draft_diff = [p for p in draft if base.get(p, UNWRITTEN) != got.get(p, UNWRITTEN)]
    return dict(base_err=base_err, err=got_err, target_diff=target_diff, target_unwritten=target_unwritten,
                draft_diff=draft_diff, skipped=floor > 0,
                draft_unwritten_base=[p for p in draft if p not in base],
                draft_unwritten_skip=[p for p in draft if p not in got])


def config(W, k):
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=None, enable_chunked_prefill=True),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(sliding_window=W)),
        speculative_config=SimpleNamespace(use_dspark=lambda: True, num_speculative_tokens=k,
                                           draft_model_config=SimpleNamespace(hf_config=SimpleNamespace(sliding_window=W))))


def sweep(ns, W, A, k, T, prompts, starts_for, shorten=None, earlier=None):
    ns["CED_WINDOW"], ns["_DSPARK_SWA_INDEX_ALIGNMENT"] = W, A
    cfg = config(W, k)
    cfg.scheduler_config.long_prefill_token_threshold = T
    min_rows, align = ns["_nonfinal_skip_policy"](cfg, aligned_chunks=True)
    tally = dict(cases=0, skipped=0, base_err=0, err=0, target_diff=0, target_unwritten=0, draft_diff=0,
                 draft_unwritten_base=0, draft_unwritten_skip=0, fail_stop=0)
    examples = {}
    for P in prompts:
        for H in starts_for(P):
            try:
                r = scenario(ns, cfg, T, P, H, shorten, earlier(P, H) if earlier else None)
            except RuntimeError as error:
                assert shorten is not None and "CED successor chunk" in str(error), error
                tally["cases"] += 1
                tally["fail_stop"] += 1
                continue
            tally["cases"] += 1
            tally["skipped"] += r["skipped"]
            for key in ("base_err", "err", "target_diff", "target_unwritten", "draft_diff",
                        "draft_unwritten_base", "draft_unwritten_skip"):
                if r[key]:
                    tally[key] += 1
                    examples.setdefault(key, (P, H, r[key][:3]))
    print(f"W={W} align={A} k={k} T={T} policy(min_rows, align)=({min_rows}, {align}) {tally}")
    for key, ex in examples.items():
        print(f"   e.g. {key}: P={ex[0]} H={ex[1]} {ex[2]}")
    return tally


def main():
    ns = load_namespace()
    real_W, real_A = ns["CED_WINDOW"], ns["_DSPARK_SWA_INDEX_ALIGNMENT"]
    print(f"loaded from {TREE}: CED_WINDOW={real_W}, DSpark SWA alignment={real_A}")
    bad = 0
    # 1. Scaled-down, exhaustive: every prompt length and every computed start.
    print("\n[exhaustive, scaled window]")
    for k in (1, 3, 5, 6):
        W, A = 8, 4
        ns["CED_WINDOW"], ns["_DSPARK_SWA_INDEX_ALIGNMENT"] = W, A
        min_rows = ns["_nonfinal_skip_policy"](_with_T(config(W, k), 10**6), aligned_chunks=True)[0]
        for T in sorted({min_rows, min_rows + 1, min_rows + 3, 2 * min_rows, 3 * min_rows + 1}):
            t = sweep(ns, W, A, k, T, range(1, 4 * T + 2 * min_rows), lambda P: range(0, P))
            bad += t["base_err"] + t["err"] + t["target_diff"] + t["target_unwritten"]
    # 2. Real constants and production thresholds, sampled prompt lengths and block-aligned prefix hits.
    print("\n[real constants, sampled]")
    rng = random.Random(0)
    for k, T in ((5, 4048), (5, 6096), (5, 8144), (5, 8096), (6, 24464), (6, 16288), (5, 16288)):
        prompts = sorted({rng.randrange(1, 4 * T) for _ in range(150)} | {T, T + 1, 2 * T, 2 * T - 199, 2 * T + 199, 2 * T + 200, 3 * T + 128})
        t = sweep(ns, real_W, real_A, k, T, prompts,
                  lambda P: sorted({0, P - 1} | {b for b in range(0, P, 256)[:: max(1, P // 256 // 6)]} |
                                   {b for b in (T, 2 * T, T - 256) if 0 <= b < P}))
        bad += t["base_err"] + t["err"] + t["target_diff"] + t["target_unwritten"]
    # 3. Fail-safe probe: a later chunk cut short (what multimodal encoder budgets can do, which the policy excludes).
    print("\n[fault injection: shortened successors must fail before GPU staging]")
    ns["CED_WINDOW"], ns["_DSPARK_SWA_INDEX_ALIGNMENT"] = 8, 4
    cut = random.Random(1)
    fault = sweep(ns, 8, 4, 3, 40, range(1, 220), lambda P: range(0, P, 3),
          shorten=lambda c, n: max(1, cut.randrange(1, n + 1)) if c % 40 == 0 and c > 0 else n)
    assert fault["fail_stop"] > 0, fault
    # 4. Prefix hits on pages an earlier, longer request wrote (its chunk ends fall inside our prompt).
    print("\n[prefix hits on a longer earlier request's pages; drafter-only effects]")
    for k in (3, 5):
        min_rows = ns["_nonfinal_skip_policy"](_with_T(config(8, k), 10**6), aligned_chunks=True)[0]
        T = min_rows + 5
        t = sweep(ns, 8, 4, k, T, range(1, 4 * T), lambda P: range(0, P),
                  earlier=lambda P, H: P + 2 * T + min_rows)
        bad += t["base_err"] + t["err"] + t["target_diff"] + t["target_unwritten"]
    print("\nRESULT:", "PASS (target bit-exact, no unwritten reads)" if bad == 0 else f"FAIL ({bad} bad sweeps)")
    return 1 if bad else 0


def _with_T(cfg, T):
    cfg.scheduler_config.long_prefill_token_threshold = T
    return cfg


if __name__ == "__main__":
    sys.exit(main())
