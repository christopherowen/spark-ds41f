#!/usr/bin/env python3
import hashlib
import inspect
import json
import os
import pathlib
import subprocess
import sys
import time
import types
import numpy as np
import torch
import vllm.v1.worker.gpu.spec_decode.adaptive_verification as av_mod
from vllm.v1.worker.gpu.spec_decode.adaptive_verification import AdaptiveVerificationManager

boot_label = sys.argv[1]
out_dir = pathlib.Path(sys.argv[2])
t0 = time.monotonic()

assert torch.cuda.is_available(), "CUDA must be available"
device = torch.device("cuda:0")
device_name = torch.cuda.get_device_name(device)
free_b, total_b = torch.cuda.mem_get_info(device)

av_path = pathlib.Path(inspect.getsourcefile(av_mod))
av_sha256 = hashlib.sha256(av_path.read_bytes()).hexdigest()

max_num_reqs = 8
num_spec_steps = 3
req_ids = [f"req-{i}" for i in range(4)]

def make_req_states():
    prefill_len = types.SimpleNamespace(np=np.array([16] * max_num_reqs, dtype=np.int32))
    return types.SimpleNamespace(
        num_speculative_steps=num_spec_steps,
        device=device,
        max_num_reqs=max_num_reqs,
        max_num_batched_tokens=4096,
        req_id_to_index={f"req-{i}": i for i in range(max_num_reqs)},
        num_computed_tokens_np=np.array([32] * max_num_reqs, dtype=np.int32),
        prefill_len=prefill_len,
    )

conf_matrix = np.array([
    [0.85, 0.70, 0.55],
    [0.60, 0.40, 0.25],
    [0.30, 0.20, 0.10],
    [0.18, 0.12, 0.08],
], dtype=np.float32)

gpu_decisions = {}
for label, rc in [("c40_0.035", 0.035), ("row020_0.02", 0.02), ("row010_0.01", 0.01)]:
    rs = make_req_states()
    qsl = torch.zeros(max_num_reqs + 1, dtype=torch.int32, device=device)
    mgr = AdaptiveVerificationManager(
        req_states=rs,
        query_start_loc=qsl,
        num_bonus_tokens=1,
        max_total_logits=64,
        row_cost=rc,
    )
    mgr.load_draft_budget_kernels()
    for i in range(4):
        mgr.add_request(i)
    mgr.apply_staged_writes()
    mgr._stale_confidences[mgr._stale_idx].np[:4, :3] = conf_matrix
    mgr._confidence_probs[:4, :3].copy_(torch.from_numpy(conf_matrix).to(device=device))

    num_tokens_per_req = {rid: 4 for rid in req_ids}
    draft_tokens = {rid: [101, 102, 103] for rid in req_ids}
    total_tokens = int(mgr.get_num_tokens(num_tokens_per_req, draft_tokens))
    _, _, draft_budget = mgr._batch_budget
    compacted_sched, compacted_cu = mgr.compact_batch(
        np.array([3, 3, 3, 3], dtype=np.int32),
        np.array([4, 4, 4, 4], dtype=np.int32),
        np.array([0, 4, 8, 12, 16], dtype=np.int32),
    )
    idx_mapping = torch.arange(4, dtype=torch.int64, device=device)
    cu_logits, qsl_out, db_out, exp_idx, exp_pos = mgr.reallocate_drafts(req_ids, idx_mapping)
    torch.cuda.synchronize(device)
    verified_caps = mgr.get_verified_draft_counts(4).cpu().tolist()
    gpu_decisions[label] = {
        "row_cost": rc,
        "total_tokens": total_tokens,
        "draft_budget": int(draft_budget),
        "compacted_scheduled_tokens": compacted_sched.tolist(),
        "device_verified_draft_counts": verified_caps,
        "cu_num_logits": cu_logits.cpu().tolist(),
        "query_start_loc": qsl_out[:5].cpu().tolist(),
        "expanded_idx_mapping": exp_idx.cpu().tolist(),
        "expanded_local_pos": exp_pos.cpu().tolist(),
    }

max_cuda_alloc = int(torch.cuda.max_memory_allocated(device))

env = dict(os.environ)
env["PYTHONPATH"] = "/work/checkout"

test_cmds = [
    [sys.executable, "-m", "unittest", "discover", "-s", "/work/checkout/experiments/2026-11-05-verification-budget-replay", "-p", "test_*.py"],
]
test_results = []
for cmd in test_cmds:
    t_start = time.monotonic()
    proc = subprocess.run(cmd, cwd="/work/checkout", env=env, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Test failed: {cmd}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    test_results.append({
        "cmd": cmd,
        "elapsed_s": round(time.monotonic() - t_start, 4),
        "stderr_tail": proc.stderr.strip().splitlines()[-3:],
    })

audit_out = out_dir / f"{boot_label}-audit.json"
replay_out = out_dir / f"{boot_label}-replay.json"
grid_root = "experiments/2026-11-04-topk-live-qualification/runs/2026-11-04-dgx/attempt-2/performance"

t_audit = time.monotonic()
subprocess.run([
    sys.executable,
    "experiments/2026-11-05-verification-budget-replay/audit.py",
    "--trace", "/work/inputs/k5.jsonl",
    "--source", str(av_path),
    "--grid", "manifests/suites/2026-10-30-spark-ds41f-1030-tp4/grid.json",
    "--grid", f"{grid_root}/candidate-40/release-tp3/grid.json",
    "--grid", f"{grid_root}/candidate-40-repeat/release-tp3/grid.json",
    "--output", str(audit_out),
], cwd="/work/checkout", env=env, check=True)
audit_elapsed = round(time.monotonic() - t_audit, 4)

t_replay = time.monotonic()
subprocess.run([
    sys.executable,
    "experiments/2026-11-05-verification-budget-replay/replay.py",
    "--trace", "/work/inputs/k5.jsonl",
    "--grid", "manifests/suites/2026-10-30-spark-ds41f-1030-tp4/grid.json",
    "--grid", f"{grid_root}/candidate-40/release-tp3/grid.json",
    "--grid", f"{grid_root}/candidate-40-repeat/release-tp3/grid.json",
    "--output", str(replay_out),
], cwd="/work/checkout", env=env, check=True)
replay_elapsed = round(time.monotonic() - t_replay, 4)

audit_json = json.loads(audit_out.read_text())
replay_json = json.loads(replay_out.read_text())

summary = {
    "boot": boot_label,
    "elapsed_s": round(time.monotonic() - t0, 4),
    "cuda": {
        "available": True,
        "device_name": device_name,
        "free_bytes": int(free_b),
        "total_bytes": int(total_b),
        "max_cuda_allocated_bytes": max_cuda_alloc,
    },
    "adaptive_verification": {
        "path": str(av_path),
        "sha256": av_sha256,
        "gpu_decisions": gpu_decisions,
    },
    "tests": test_results,
    "audit_elapsed_s": audit_elapsed,
    "replay_elapsed_s": replay_elapsed,
    "audit_sha256": hashlib.sha256(audit_out.read_bytes()).hexdigest(),
    "replay_sha256": hashlib.sha256(replay_out.read_bytes()).hexdigest(),
    "source_contract": audit_json.get("source_contract"),
    "trace_summary": replay_json.get("trace"),
}
(out_dir / f"{boot_label}-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
