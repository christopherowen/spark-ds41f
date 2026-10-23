#!/usr/bin/env python3
"""Write the next release's ring screen from the vLLM release branch.

usage: make_screen.py VLLM_CHECKOUT [BASE]

BASE is the image's vLLM tree (spark-ds41f-1022's patch head, 8fc15c18f). Every file the
branch changes since BASE becomes a runtime overlay over image -1022-4: the arms mount the
non-test sources, and the GPU test bundle mounts sources and tests. Writes overlays/,
bundles/gpu-tests/, the arms, screen-suite-tp4.json and screen-tp4.json.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP = HERE.relative_to(ROOT).as_posix()
TREE = "/opt/spark3/candidate/vllm"
IMAGE = "vllm-ds41f-kkref:19f2c20ed4d6-1022-4"
GPU_TESTS = [
    "tests/kernels/test_deepseek_v41_tilelang_ced.py",
    "tests/kernels/test_deepseek_v41_tilelang_metadata.py",
    "tests/v1/attention/test_b12x_v41_ced_metadata.py",
    "tests/models/test_deepseek_v4_1_ced_model.py",
    "tests/kernels/moe/test_deepseek_v41_tilelang_router.py",
    "tests/v1/spec_decode/test_rejection_sampler_utils.py",
    "tests/v1/spec_decode/test_dspark_topk.py",
]


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n")


def flat(path: str) -> str:
    return path.replace("/", "_")


def main(checkout: str, base: str = "8fc15c18f") -> None:
    repo = Path(checkout)
    changed = subprocess.run(["git", "-C", str(repo), "diff", "--name-only", base, "HEAD"],
                             check=True, capture_output=True, text=True).stdout.split()
    sources = [f for f in changed if f.startswith("vllm/")]
    shutil.rmtree(HERE / "overlays", ignore_errors=True)
    for f in changed:
        (HERE / "overlays").mkdir(exist_ok=True)
        shutil.copyfile(repo / f, HERE / "overlays" / flat(f))

    mounts = [["{home}/projects/spark-ds41f/cache", "/cache", "rw"]] + [
        [f"{{home}}/projects/spark-ds41f/{EXP}/overlays/{flat(f)}", f"{TREE}/{f}", "ro"] for f in sources]
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short=9", "HEAD"],
                          check=True, capture_output=True, text=True).stdout.strip()
    dump(HERE / "candidate-tp4.json", {
        "notes": f"The next release's vLLM branch ({head}) as runtime overlays over spark-ds41f-1022 "
                 "(image -1022-4) at the recipe's tp4-ring: the CED TileLang ports and nonfinal skip, the "
                 "block-verification draw refresh, generation controls and the DeepSeek API thinking "
                 "switch. No test code.",
        "extends": {"config": "config/cluster.json", "shape": "tp4-ring"},
        "set": {"container": {"mounts": mounts}},
    })
    dump(HERE / "candidate-tp4-dev.json", {
        "notes": "candidate-tp4.json with the dev-mode collective RPC, for the suite's engine grid.",
        "extends": {"config": f"{EXP}/candidate-tp4.json"},
        "set": {"environment": {"VLLM_SERVER_DEV_MODE": "1"}},
    })

    bundle = HERE / "bundles" / "gpu-tests"
    shutil.rmtree(bundle, ignore_errors=True)
    bundle.mkdir(parents=True)
    shutil.copyfile(ROOT / "experiments/2026-10-19-ced-kernels/bundles/ced-tests/prepare_fixtures.py",
                    bundle / "prepare_fixtures.py")
    for f in changed:
        shutil.copyfile(repo / f, bundle / flat(f))
    pytest = ["-m", "pytest", "-q", "-rfEs", "-p", "no:cacheprovider", "--noconftest",
              "-p", "prepare_fixtures", "--basetemp", "/c/pytest-next-release", *GPU_TESTS]
    dump(bundle / "candidate.json", {
        "description": f"The next release's GPU tests on image -1022-4 with the branch's files ({head}) "
                       "mounted: CED TileLang kernels, metadata and skip, CED model and router, "
                       "block-verification sampling. Production files stay in the image.",
        "image": IMAGE,
        "env": {"HF_HUB_OFFLINE": "1", "TILELANG_CACHE_DIR": "/c/tilelang-postlaunch",
                "VLLM_DS41_KERNEL_BACKEND": "tilelang", "PYTHONPATH": "/b"},
        "workdir": TREE,
        "argv": ["-c", "import subprocess,sys\nsys.exit(subprocess.run([sys.executable,"
                 + ",".join(repr(a) for a in pytest) + "],timeout=3600).returncode)"],
        "verdict": "exit",
        "mounts": [["prepare_fixtures.py", "/b/prepare_fixtures.py"]]
                  + [[flat(f), f"{TREE}/{f}"] for f in changed],
    })

    trimmed = json.loads((ROOT / "experiments/2026-10-22-spark-ds41f-1022/promotion-suite-tp4.json").read_text())
    sampled = json.loads((ROOT / "servebench/suites/sampled.json").read_text())
    c16 = next(s for s in sampled["steps"] if s["id"] == "c16")
    dump(HERE / "screen-suite-tp4.json", {
        "name": "next-release-screen-tp4", "version": 1,
        "description": "spark-ds41f-1022's trimmed TP4 promotion suite (model at 16 streams, grid up to 16 "
                       "requests, closed loops at 16 and 1, 32K and 131K prefill), one cold 1,000,000-token "
                       "prefill (the long end of the CED skip), and the sampled suite's sixteen streams "
                       "(temperature 1, top_p 1, seed 0; 30 s warm-up, 180 s), where the block-verification "
                       "draw refresh changes the text by design. The grid needs the dev-mode collective RPC.",
        "steps": trimmed["steps"] + [
            {"id": "prefill-1m", "command": "prefill", "args": {"sizes": [1000000], "repeats": 1}},
            {**c16, "id": "sampled-c16", "args": {**c16["args"], "warmup": 30, "duration": 180}},
        ],
    })

    determinism = [["experiments/2026-10-12-r6g/token_determinism_text.py",
                    ["--tokens", "192", "--long-lines", "700", "--wide", *extra], name]
                   for extra, name in (([], "determinism"), (["--no-logprobs"], "greedy"))]
    dump(HERE / "screen-tp4.json", {
        "experiment": EXP,
        "run": "next-release-screen-tp4",
        "jobs": [
            {"kind": "sync"},
            {"kind": "kernel", "bundles": [{"node": "dgx4", "bundle": f"{EXP}/bundles/gpu-tests"}]},
            {"kind": "measure", "bench": False, "bracket": False, "extras": determinism + [
                ["experiments/2026-10-22-spark-ds41f-1022/control_spellings_probe.py", [], "control-spellings"],
                ["experiments/2026-10-22-spark-ds41f-1022/sink_counters.py", [], "sink-ordinary"],
                ["experiments/2026-10-19-sink-rejection-v2/probes.py", ["--nodes", "dgx1,dgx2,dgx3,dgx4"], "probes"],
            ], "arms": [{"config": "candidate-tp4.json", "label": "candidate"}]},
            {"kind": "measure",
             "suite": f"{EXP}/screen-suite-tp4.json",
             "arms": [{"config": "candidate-tp4-dev.json", "label": "candidate"},
                      {"config": "experiments/2026-10-22-spark-ds41f-1022/spark-ds41f-1022-tp4-dev.json",
                       "label": "spark-ds41f-1022"},
                      {"config": "candidate-tp4-dev.json", "label": "candidate-again"}]},
        ],
    })
    print(f"{len(changed)} files ({len(sources)} sources) from {head}")


if __name__ == "__main__":
    main(*sys.argv[1:])
