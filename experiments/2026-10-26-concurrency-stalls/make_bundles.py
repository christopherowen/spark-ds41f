#!/usr/bin/env python3
"""Write the scheduler test bundles (kernel-local, CPU tests in the 1025 image's container).

usage: make_bundles.py VLLM_REPO OUT_DIR

Each bundle mounts a vLLM tree over spark-ds41f-1025's image (spark-ds41f:19f2c20ed4d6-next-6,
tree e0ccc837d): every package file that differs from the image's tree (a deleted module
becomes one that fails to import), and the test files it runs.

- sched-1025-base: the image's own tree. The regression set's baseline.
- prefill-yield-1025: prefill yield on the image's tree (branch prefill-yield-1025).
- sched-next-base: next-candidate (e846e64dc), with PIN below: the wait behind a long prompt
  and the prefill lanes' livelock, on the unfixed code.
- sched-next-fixes: next-candidate with prefill yield and the lanes fix (concurrency-stalls-next).

Every bundle then runs the regression set: the image's or the tree's aligned-chunk, scheduler and
compute-share scheduler tests. A fix must not change its failures.
facebook/opt-125m (config and tokenizer only, the scheduler tests' model) is a synced input,
mounted from ~/spark-lab/inputs/hf-opt-125m at /hf.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMAGE = "spark-ds41f:19f2c20ed4d6-next-6"
BASE = "e0ccc837d"
ROOT = "/opt/spark-ds41f/candidate/vllm"
MISSING = '''# Deleted at this ref; importing it fails as importing a missing module does.
raise ModuleNotFoundError(f"No module named {__name__!r}", name=__name__)
'''
REGRESSION = ["tests/v1/core/test_aligned_prefill_chunks.py", "tests/v1/core/test_scheduler.py",
              "tests/v1/core/test_prefill_compute_share_scheduler.py"]
RUN = '''import json, subprocess, sys
codes = []
for step in json.load(open("/b/steps.json")):
    print("=== " + step["name"], flush=True)
    codes.append(subprocess.call([sys.executable, "-m", "pytest", *step["args"]]))
print("exit codes", codes, flush=True)
sys.exit(max(codes))
'''
PYTEST = ["-q", "-p", "no:cacheprovider", "-rfEs"]
# sched-next-base's evidence: next-candidate as it was, before the lanes fix (f05748ead). The
# fix's own tests (test_aligned_prefill_lanes.py) replace it in vLLM.
PIN = '''# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""What aligned prefill chunks do to concurrent requests (spark-ds41f's concurrency-stall study).

Copy to tests/v1/core/ in a vLLM tree (next-candidate) and run on CPU:
    python -m pytest tests/v1/core/test_concurrency_stalls.py
It pins the scheduler behaviour the study's simulator assumes:
- a long prompt's whole aligned chunk takes the step's budget beside the decode rows, so a
  short request that arrives while it prefills waits for its last chunk;
- with two prefill lanes (--max-parallel-prefills 2), each lane's share is half the budget;
  two long prompts' whole chunks never fit, neither lane is released, and nothing prefills.
"""

import functools
import json

import pytest

_OPT_CONFIG = {
    "architectures": ["OPTForCausalLM"],
    "model_type": "opt",
    "hidden_size": 768,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "ffn_dim": 3072,
    "max_position_embeddings": 8192,
    "vocab_size": 50272,
    "word_embed_proj_dim": 768,
    "torch_dtype": "float16",
    "do_layer_norm_before": True,
    "pad_token_id": 1,
    "bos_token_id": 2,
    "eos_token_id": 2,
}

BUDGET = 1024
SEQS = 16
THRESHOLD = BUDGET - SEQS  # one row a decoding request (no drafts), as the recipe derives it


def _scheduler(monkeypatch, tmp_path, lanes=1):
    utils = pytest.importorskip("tests.v1.core.utils")
    # DeepSeek V4.1's configuration sets the flag; this OPT model takes it here.
    monkeypatch.setattr(utils, "SchedulerConfig",
                        functools.partial(utils.SchedulerConfig, align_prefill_chunks=True))
    (tmp_path / "config.json").write_text(json.dumps(_OPT_CONFIG))
    scheduler = utils.create_scheduler(
        model=str(tmp_path),
        max_num_seqs=SEQS,
        max_num_batched_tokens=BUDGET,
        long_prefill_token_threshold=THRESHOLD,
        max_parallel_prefills=lanes,
        max_model_len=8192,
        skip_tokenizer_init=True,
        device="cpu",
    )
    return utils, scheduler


def test_a_short_request_waits_for_the_long_prompts_last_chunk(monkeypatch, tmp_path):
    utils, scheduler = _scheduler(monkeypatch, tmp_path)
    long, = utils.create_requests(num_requests=1, num_tokens=4000, req_ids=["long"])
    scheduler.add_request(long)
    assert scheduler.schedule().num_scheduled_tokens == {"long": THRESHOLD}
    short, = utils.create_requests(num_requests=1, num_tokens=40, req_ids=["short"])
    scheduler.add_request(short)
    steps = []
    for _ in range(3):
        steps.append(dict(scheduler.schedule().num_scheduled_tokens))
    # Two more whole chunks with no room beside them, then the last (partial) chunk and the
    # short request together.
    assert steps[0] == {"long": THRESHOLD}
    assert steps[1] == {"long": THRESHOLD}
    assert steps[2] == {"long": 4000 - 3 * THRESHOLD, "short": 40}  # 976 + 40 <= 1,024


def test_two_lanes_never_schedule_two_long_prompts(monkeypatch, tmp_path):
    utils, scheduler = _scheduler(monkeypatch, tmp_path, lanes=2)
    for request in utils.create_requests(num_requests=2, num_tokens=4000, req_ids=["a", "b"]):
        scheduler.add_request(request)
    for _ in range(3):
        assert scheduler.schedule().total_num_scheduled_tokens == 0


def test_one_lane_schedules_them_in_turn(monkeypatch, tmp_path):
    utils, scheduler = _scheduler(monkeypatch, tmp_path, lanes=1)
    for request in utils.create_requests(num_requests=2, num_tokens=4000, req_ids=["a", "b"]):
        scheduler.add_request(request)
    assert scheduler.schedule().num_scheduled_tokens == {"a": THRESHOLD}
    assert scheduler.schedule().num_scheduled_tokens == {"a": THRESHOLD}
'''
BUNDLES = {
    "sched-1025-base": (BASE, [], ""),
    "prefill-yield-1025": ("prefill-yield-1025", ["tests/v1/core/test_prefill_yield.py"], ""),
    "sched-next-base": ("next-candidate", [], "test_concurrency_stalls.py"),
    "sched-next-fixes": ("concurrency-stalls-next",
                         ["tests/v1/core/test_prefill_yield.py", "tests/v1/core/test_aligned_prefill_lanes.py"], ""),
}


def git(repo: str, *args: str) -> str:
    return subprocess.run(["git", "-C", repo, *args], check=True, text=True, capture_output=True).stdout


def show(repo: str, ref: str, path: str) -> bytes:
    return subprocess.run(["git", "-C", repo, "show", f"{ref}:{path}"], check=True,
                          capture_output=True).stdout


def main() -> int:
    repo, out = sys.argv[1], Path(sys.argv[2])
    for name, (ref, new_tests, pin) in BUNDLES.items():
        head = git(repo, "rev-parse", "--short=9", ref).strip()
        status = [line.split("\t") for line in
                  git(repo, "diff", "--no-renames", "--name-status", BASE, ref, "--", "vllm", "tests").splitlines()]
        bundle = out / name
        if bundle.exists():
            raise SystemExit(f"{bundle} exists: write the bundles to a new directory")
        (bundle / "overlay").mkdir(parents=True)
        mounts = [["run.py", "/b/run.py"], ["steps.json", "/b/steps.json"], ["hf-opt-125m", "/hf"]]
        package = 0
        for kind, path in status:
            if path.startswith("vllm/"):
                target = bundle / "overlay" / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(MISSING.encode() if kind == "D" else show(repo, ref, path))
                package += 1
        tests = list(new_tests)
        for path in REGRESSION + new_tests:
            if any(p == path for _, p in status):
                flat = path.replace("/", "__")
                (bundle / flat).write_bytes(show(repo, ref, path))
                mounts.append([flat, f"{ROOT}/{path}"])
        if pin:
            (bundle / pin).write_text(PIN)
            mounts.append([pin, f"{ROOT}/tests/v1/core/{pin}"])
            tests.append(f"tests/v1/core/{pin}")
        steps = []
        if tests:
            steps.append({"name": "new tests", "args": PYTEST + ["--noconftest", *tests]})
        # vLLM's root conftest imports tblib, which the image lacks.
        steps.append({"name": "regression set", "args": PYTEST + ["--noconftest", *REGRESSION]})
        (bundle / "run.py").write_text(RUN)
        (bundle / "steps.json").write_text(json.dumps(steps, indent=1) + "\n")
        candidate = {
            "description": f"Scheduler tests on vLLM {ref} {head} ({package} package files over the image's "
                           f"{BASE}), CPU, in spark-ds41f-1025's image.",
            "image": IMAGE,
            "env": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_CACHE": "/hf/hub"},
            "workdir": "vllm",
            "argv": ["/b/run.py"],
            "verdict": "exit",
            "mounts": mounts,
        }
        (bundle / "candidate.json").write_text(json.dumps(candidate, indent=1) + "\n")
        print(f"{name}: {ref} {head}, {package} package files, tests {tests or '-'} + regression set")
    return 0


if __name__ == "__main__":
    sys.exit(main())
