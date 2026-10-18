#!/usr/bin/env python3
"""Apple's RSI-Inference DS4.1 checks against the running deployment (a lab extras script).

usage: run_rsi.py URL --label LABEL [--root ~/spark-lab/apple-rsi] [--image IMAGE] [--steps a,b]

Runs in a container of the serving image (its Python has every client dependency), on the
host network, against URL. ROOT holds the attached verifier copy that prepare.py writes.
Each step's full output goes to results/private/apple-rsi/<label>/; one JSON line per step
goes to stdout.

Steps:
- api: the task's API examples (completions, chat, streaming, log-probabilities, repeated
  greedy requests, a second turn, concurrent requests);
- fidelity-public: the four public reference sequences (prose, math, code, structured;
  512 to 20,480 tokens), teacher-forced log-probabilities against the model author's;
- fidelity-hidden: the 32 held-out sequences through the same public checker;
- verifier: the task's own scored nodes that read only a running server: held-out
  fidelity with its per-band bars, text-history fidelity (fresh, after an unrelated
  request, decode against prefill), chat/streaming/concurrent isolation, and GSM8K on the
  held-out 200-question slice against its floor;
- efficiency: the task's serving client, per-user decode at one stream and throughput at
  16, on the public natural-text sample with the held-out seed.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SERVED = "deepseek-v4.1-flash"
EVAL = "/tests/eval/deepseek_v41_flash"
STEPS = {
    "api": f"python3 {EVAL}/tests/api_examples.py --base-url $URL --model {SERVED}",
    "fidelity-public": f"python3 {EVAL}/fidelity/check.py --base-url $URL --model {SERVED} "
                       f"--record /out/fidelity-public.json",
    "fidelity-hidden": f"python3 {EVAL}/fidelity/check.py --base-url $URL --model {SERVED} "
                       f"--examples /tests/hidden-as-examples.json --record /out/fidelity-hidden.json",
    "verifier": (
        # GSM8K's train and test files through the engine evaluator's own download.
        "mkdir -p /out/datasets/gsm8k && python3 - <<'EOF'\n"
        "import sys, shutil\n"
        "sys.path.insert(0, '/opt/spark3/candidate/vllm/tests/evals/gsm8k')\n"
        "import gsm8k_eval as g\n"
        "for split in ('train', 'test'):\n"
        "    path = g.download_and_cache_file(f'{g.VLLM_S3_BUCKET_URL}/ci-datasets/gsm8k/{split}.jsonl')\n"
        "    shutil.copyfile(path, f'/out/datasets/gsm8k/{split}.jsonl')\n"
        "EOF\n"
        "cd /tests && python3 -m pytest -p no:cacheprovider -q -rA -s "
        "--junitxml /out/verifier.junit.xml "
        "infb_verify/test_fidelity.py::test_fidelity_kl "
        "infb_verify/test_text_semantics.py::test_text_history_fidelity "
        "infb_verify/test_text_semantics.py::test_chat_streaming_and_isolation "
        "infb_verify/test_hidden.py::test_gsm8k_hidden"
    ),
    "efficiency": f"python3 {EVAL}/serving/client.py measure --engine vllm --base-url $URL "
                  f"--tokenizer /model/deepseek-ai/DeepSeek-V4.1-Flash/tokenizer.json "
                  f"--texts {EVAL}/serving/sample.jsonl.xz --concurrency 16 --model {SERVED} "
                  f"--seed-file /tests/hidden/efficiency_seed.json --record /out/efficiency.json",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--label", required=True)
    ap.add_argument("--root", default="~/spark-lab/apple-rsi")
    ap.add_argument("--image", default="vllm-ds41f-kkref:19f2c20ed4d6-r6v")
    ap.add_argument("--steps", default=",".join(STEPS))
    ap.add_argument("--port", type=int, help="the arm's port, when it is not the URL's (lab passes :8000)")
    args = ap.parse_args()
    root = Path(os.path.expanduser(args.root))
    out = ROOT / "results" / "private" / "apple-rsi" / args.label
    out.mkdir(parents=True, exist_ok=True)
    url = args.url.rstrip("/")
    if args.port:
        url = f"{url.rsplit(':', 1)[0]}:{args.port}"
    worst = 0
    for step in args.steps.split(","):
        command = ["docker", "run", "--rm", "--network", "host", "--entrypoint", "bash",
                   "-e", f"URL={url}", "-e", f"INFB_BASE_URL={url}", "-e", "TESTS_DIR=/tests",
                   "-e", "APP_DIR=/opt/spark3/candidate/vllm", "-e", "PYTHONPATH=/tests",
                   "-e", "INFB_SEMANTICS_RECORD_DIR=/out/semantics",
                   "-e", "INFB_FIDELITY_RECORD_DIR=/out/fidelity",
                   "-e", "INFB_EFFICIENCY_DIR=/out/efficiency", "-e", "INFB_LAUNCH_LOG_DIR=/out",
                   "-v", f"{root / 'tests'}:/tests:ro",
                   "-v", f"{root / 'model'}:/model/deepseek-ai/DeepSeek-V4.1-Flash:ro",
                   "-v", f"{out}:/out", args.image, "-lc", STEPS[step]]
        started = time.time()
        with open(out / f"{step}.txt", "w") as log:
            code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
        worst = max(worst, code)
        print(json.dumps({"step": step, "label": args.label, "exit": code,
                          "seconds": round(time.time() - started, 1),
                          "output": str((out / f"{step}.txt").relative_to(ROOT))}), flush=True)
    return 0 if worst == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
