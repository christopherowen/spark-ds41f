#!/usr/bin/env python3
"""Write this experiment's kernel-lab bundles (dgx4, scripts/lab.py kernel-local).

usage: make_bundles.py VLLM_REPO [--head REF] [--only NAME ...]

Every bundle runs on candidate 6's image (spark-ds41f:19f2c20ed4d6-next-6), whose vLLM tree
is BASE (3bdf9558c: tilelang-ports 26382f5cf with test patches 0489-0490). A vLLM ref is
put in place by mounting every package file it changes from BASE (overlay/vllm/...; a
module it deletes is mounted as one that fails to import, as a missing module does) and
every test file it changes over the image's copy.

- compare (HEAD, B12X installed): checks/pack_compare.py and checks/engram_compare.py,
  vLLM's ports against B12X's code on the 1017 checkpoint, bit for bit.
- checkpoint-absent (HEAD, B12X uninstalled by checks/without_b12x.py): the packer alone
  (its packed bytes and logits hashed, for comparison with compare's), then one TP4 rank's
  load through the direct loader and its weight digest against 2026-10-25's.
- gate-absent (HEAD, B12X uninstalled): the release-test gate of config/release-tests.json
  (scripts/release_tests.py gate and bundle), dgx4's compute-sanitizer mounted.
- sweep-next-candidate and sweep-retire-b12x (NEXT and HEAD): checks/import_sweep.py over
  every vllm module with B12X installed, then again after uninstalling it.

Each bundle is written to bundles/<name>-<vLLM commit>, which must not exist yet: a new
vLLM head gets new directories, and nothing is ever cleared.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import release_tests  # noqa: E402

IMAGE = "spark-ds41f:19f2c20ed4d6-next-6"
ROOT = "/opt/spark-ds41f/candidate/vllm"
BASE = "3bdf9558c"
NEXT = "1bc005688"  # vLLM next-candidate
CHECKPOINT = ("/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/"
              "1bae0e1088993b9b82de08f59de5be5a34e42170")
HUB = ["~/.cache/huggingface/hub", "/hub"]
# The 2026-10-25 direct-loader rank load of TP4 rank 0 (direct and B12X loaders agreed).
DIGEST = "a4893a5ed2f95f53b63ea8783157525c4f9153d8bbb7997345cba1350290c6fd"
RANK_LOAD = REPO_ROOT / "experiments/2026-10-25-final-names/loader/kernel/rank-load/rank_load.py"
SANITIZER = ["/usr/local/cuda-13.0/compute-sanitizer", "/opt/compute-sanitizer"]
WITHOUT_PATH = "/b:/opt/spark-ds41f/candidate/vllm"  # no B12X tree
MISSING = '''# Deleted at this vLLM ref; mounted over the image's copy so that importing it fails
# as importing a missing module does.
raise ModuleNotFoundError(f"No module named {__name__!r}", name=__name__)
'''
RUN = '''#!/usr/bin/env python3
"""Run steps.json's steps in order (each in its own interpreter, with a timeout); print each
step's exit status; exit 1 if any failed."""
import json
import subprocess
import sys

codes = []
for label, argv, timeout in json.load(open("/b/steps.json")):
    print(f"=== {label}", flush=True)
    try:
        code = subprocess.run([sys.executable, *argv], timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        code = "timeout"
    print(f"=== {label}: exit {code}", flush=True)
    codes.append((label, code))
print(json.dumps({"steps": codes}), flush=True)
sys.exit(0 if all(code == 0 for _, code in codes) else 1)
'''


def git(repo: str, *args: str) -> str:
    return subprocess.run(["git", "-C", repo, *args], check=True, capture_output=True,
                          text=True).stdout


def show(repo: str, ref: str, path: str) -> bytes:
    return subprocess.run(["git", "-C", repo, "show", f"{ref}:{path}"], check=True,
                          capture_output=True).stdout


def put_tree(repo: str, ref: str, bundle: Path, candidate: dict) -> tuple[int, int, int]:
    """Mount ``ref``'s changes from BASE into ``bundle``: package files under overlay/,
    deleted modules as MISSING, tests and benchmarks as mounts over the image's."""
    changes = [line.split("\t") for line in
               git(repo, "diff", "--no-renames", "--name-status", BASE, ref).splitlines()]
    assert all(status in ("A", "M", "D") for status, _ in changes), changes
    package = deleted = tests = 0
    for status, path in changes:
        if path.startswith("vllm/"):
            target = bundle / "overlay" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(MISSING.encode() if status == "D" else show(repo, ref, path))
            package += status != "D"
            deleted += status == "D"
        elif status != "D" and path.startswith(("tests/", "benchmarks/")):
            name = path.replace("/", "_")
            (bundle / name).write_bytes(show(repo, ref, path))
            candidate["mounts"].append([name, f"{ROOT}/{path}"])
            tests += 1
    return package, deleted, tests


def fresh(repo: str, ref: str, name: str) -> Path:
    head = git(repo, "rev-parse", "--short=9", ref).strip()
    bundle = HERE / "bundles" / f"{name}-{head}"
    bundle.mkdir(parents=True)  # refuses an existing directory
    return bundle


def step_bundle(repo, ref, name, description, env, steps, scripts, *, argv=None,
                io_uring=False, mounts=()):
    bundle = fresh(repo, ref, name)
    for script in scripts:
        shutil.copyfile(script, bundle / Path(script).name)
    (bundle / "run.py").write_text(RUN)
    (bundle / "steps.json").write_text(json.dumps(steps, indent=1) + "\n")
    candidate = {
        "description": description,
        "image": IMAGE,
        "env": {"HF_HUB_OFFLINE": "1", "TILELANG_CACHE_DIR": "/c/tilelang-retire-b12x", **env},
        "workdir": "vllm",
        "argv": argv or ["/b/run.py"],
        "verdict": "status",
        "mounts": [[n, f"/b/{n}"] for n in
                   ("run.py", "steps.json", *(Path(s).name for s in scripts))] + list(mounts),
    }
    if io_uring:
        candidate["io_uring"] = True
    counts = put_tree(repo, ref, bundle, candidate)
    head = git(repo, "rev-parse", "--short=9", ref).strip()
    candidate["description"] += (f" vLLM {ref} ({head}) over {BASE}: {counts[0]} package files, "
                                 f"{counts[1]} deleted, {counts[2]} test files.")
    (bundle / "candidate.json").write_text(json.dumps(candidate, indent=1) + "\n")
    print(f"{name}: {head}, {counts}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo")
    parser.add_argument("--head", default="retire-b12x")
    parser.add_argument("--only", nargs="*", default=())
    args = parser.parse_args()
    repo, head = args.repo, args.head
    checks = HERE / "checks"
    wanted = lambda name: not args.only or name in args.only  # noqa: E731

    if wanted("compare"):
        step_bundle(
            repo, head, "compare",
            "vLLM's TileLang packer and Engram helpers against B12X's, B12X installed.",
            {}, [["pack: synthetic weights and the 1017 head shards (TP1, TP3, TP4)",
                  ["/b/pack_compare.py", CHECKPOINT], 3600],
                 ["Engram: token map, geometry, caps, resident gather",
                  ["/b/engram_compare.py", CHECKPOINT], 3600]],
            [checks / "pack_compare.py", checks / "engram_compare.py"], mounts=[HUB])
    if wanted("checkpoint-absent"):
        step_bundle(
            repo, head, "checkpoint-absent",
            "B12X uninstalled: the packer on the 1017 head shards (hashes for comparison), "
            "then one TP4 rank's direct load and weight digest against 2026-10-25's.",
            {"PYTHONPATH": WITHOUT_PATH, "XDG_CACHE_HOME": "/c/xdg-retire-b12x",
             "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,garbage_collection_threshold:0.6"},
            [["pack, vLLM's packer only", ["/b/pack_compare.py", CHECKPOINT, "--port-only"], 3600],
             ["TP4 rank 0 direct load digest",
              ["/b/rank_digest.py", "/b/rank_load.py", CHECKPOINT, "/c/retire-b12x-rank-load",
               "--expect", DIGEST], 3600]],
            [checks / "pack_compare.py", checks / "rank_digest.py", checks / "without_b12x.py",
             RANK_LOAD],
            argv=["/b/without_b12x.py", "/b/run.py"], io_uring=True, mounts=[HUB])
    for name, ref in (("sweep-next-candidate", NEXT), ("sweep-retire-b12x", head)):
        if wanted(name):
            step_bundle(
                repo, ref, name,
                "Every vllm module imported with B12X installed, then without it.",
                {"PYTHONPATH": WITHOUT_PATH},
                [["import sweep, B12X installed", ["/b/import_sweep.py", f"{name}-installed"],
                  3600],
                 ["import sweep, B12X uninstalled",
                  ["/b/without_b12x.py", "/b/import_sweep.py", f"{name}-uninstalled"], 3600]],
                [checks / "import_sweep.py", checks / "without_b12x.py"])
    if wanted("gate-absent"):
        bundle = fresh(repo, head, "gate-absent")
        manifest = bundle.parent / f"{bundle.name}.manifest.json"
        release_tests.gate(repo, head, str(manifest))
        release_tests.bundle(str(manifest), IMAGE, str(bundle), "retire-b12x-absent")
        shutil.copyfile(checks / "without_b12x.py", bundle / "without_b12x.py")
        candidate = json.loads((bundle / "candidate.json").read_text())
        candidate["argv"] = ["/b/without_b12x.py", *candidate["argv"]]
        candidate["mounts"] += [["without_b12x.py", "/b/without_b12x.py"], SANITIZER]
        candidate["env"]["TILELANG_CACHE_DIR"] = "/c/tilelang-retire-b12x"
        candidate["env"]["COMPUTE_SANITIZER"] = f"{SANITIZER[1]}/compute-sanitizer"
        counts = put_tree(repo, head, bundle, candidate)
        hash_ = git(repo, "rev-parse", "--short=9", head).strip()
        candidate["description"] = (
            f"Release tests with B12X uninstalled: config/release-tests.json's gate at vLLM "
            f"{head} ({hash_}) over {BASE} ({counts[0]} package files, {counts[1]} deleted, "
            f"{counts[2]} test files); per-test outcomes under /c/release-tests.")
        (bundle / "candidate.json").write_text(json.dumps(candidate, indent=2) + "\n")
        print(f"gate-absent: {hash_}, {counts}")


if __name__ == "__main__":
    main()
