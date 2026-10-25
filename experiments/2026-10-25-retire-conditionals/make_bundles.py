#!/usr/bin/env python3
"""Write the GPU check of the vLLM branch retire-conditionals over candidate 6.

usage: make_bundles.py VLLM_REPO [BASE [HEAD]]

BASE (default 3bdf9558c, candidate 6's vLLM: the series through 0490, the tree image
spark-ds41f:19f2c20ed4d6-next-6 holds) to HEAD (default retire-conditionals) adds,
changes and deletes files. bundles/release-tests runs the release tests with the branch
in place: the gate manifest of config/release-tests.json at HEAD, the image
-next-6, the branch's package files under overlay/vllm (the lab mounts them over the
image's tree), the branch's test and benchmark files mounted over the image's copies,
and every package file the branch deletes replaced by a module that fails to import as
a missing one does, so nothing in the image can still reach the deleted code.
retire-tp4.json is the serving arm: candidate 6 at tp4-ring (rc6/rc-tp4.json) with the
same overlay mounted from the node checkout. retire-check.json is the lab spec.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import release_tests  # noqa: E402

EXPERIMENT = HERE.relative_to(REPO_ROOT).as_posix()
CANDIDATE = "experiments/2026-10-25-spark-ds41f-candidate/rc6"
ROOT = "/opt/spark-ds41f/candidate/vllm"
IMAGE = "spark-ds41f:19f2c20ed4d6-next-6"
LABEL = "retire-conditionals"
DIGESTS = "experiments/2026-10-24-tilelang-completion"
DETERMINISM = "experiments/2026-10-12-r6g/token_determinism_text.py"

MISSING = '''# Deleted by the vLLM branch retire-conditionals; mounted over the image's copy so that
# importing it fails as importing a missing module does.
raise ModuleNotFoundError(f"No module named {__name__!r}", name=__name__)
'''


def git(repo: str, *args: str) -> str:
    return subprocess.run(["git", "-C", repo, *args], check=True, capture_output=True,
                          text=True).stdout


def show(repo: str, head: str, path: str) -> bytes:
    return subprocess.run(["git", "-C", repo, "show", f"{head}:{path}"], check=True,
                          capture_output=True).stdout


def main() -> None:
    repo = sys.argv[1]
    base = sys.argv[2] if len(sys.argv) > 2 else "3bdf9558c"
    head = sys.argv[3] if len(sys.argv) > 3 else "retire-conditionals"
    head_commit = git(repo, "rev-parse", "--short=9", head).strip()
    changes = [line.split("\t") for line in git(repo, "diff", "--name-status", base, head).splitlines()]
    assert all(status in ("A", "M", "D") for status, _ in changes), changes
    package = [(status, path) for status, path in changes if path.startswith("vllm/")]
    others = [path for status, path in changes
              if status != "D" and path.startswith(("tests/", "benchmarks/"))]

    bundle = HERE / "bundles" / "release-tests"
    if bundle.exists():
        shutil.rmtree(bundle)
    manifest = HERE / "release-tests-gate-manifest.json"
    release_tests.gate(repo, head, str(manifest))
    release_tests.bundle(str(manifest), IMAGE, str(bundle), LABEL)
    for status, path in package:
        target = bundle / "overlay" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        if status == "D":
            target.write_text(MISSING)
        else:
            target.write_bytes(show(repo, head, path))
    candidate = json.loads((bundle / "candidate.json").read_text())
    for path in others:
        name = path.replace("/", "_")
        (bundle / name).write_bytes(show(repo, head, path))
        candidate["mounts"].append([name, f"{ROOT}/{path}"])
    deleted = sum(status == "D" for status, _ in package)
    candidate["description"] = (
        f"Release tests ({LABEL}): config/release-tests.json's gate at the vLLM branch "
        f"retire-conditionals ({head_commit}) over image -next-6: {len(package) - deleted} package files, "
        f"{deleted} deleted ones that fail to import, {len(others)} test and benchmark files; "
        "per-test outcomes under /c/release-tests.")
    # Its own TileLang cache, as candidate 6's bundle has (the kernels are the same).
    candidate["env"]["TILELANG_CACHE_DIR"] = f"/c/tilelang-{LABEL}"
    (bundle / "candidate.json").write_text(json.dumps(candidate, indent=2) + "\n")

    overlay = f"{{home}}/projects/spark-ds41f/{EXPERIMENT}/bundles/release-tests/overlay"
    mounts = [["{home}/projects/spark-ds41f/cache", "/cache", "rw"]]
    mounts += [[f"{overlay}/{path}", f"{ROOT}/{path}", "ro"] for _, path in package]
    arm = {
        "notes": (f"Candidate 6 at tp4-ring ({CANDIDATE}/rc-tp4.json) with the vLLM branch "
                  f"retire-conditionals ({head_commit}) mounted over the image's package files, "
                  "its deleted modules failing to import. VLLM_DS41_KERNEL_BACKEND stays set: "
                  "candidate 6's source manifest has the switch, and the branch's vLLM no "
                  "longer reads it. Outputs must equal candidate 6's bit for bit."),
        "extends": {"config": f"{CANDIDATE}/rc-tp4.json"},
        "set": {"container": {"mounts": mounts}},
    }
    (HERE / "retire-tp4.json").write_text(json.dumps(arm, indent=1) + "\n")

    extras = [
        [DETERMINISM, ["--tokens", "192", "--long-lines", "700", "--wide"], "determinism"],
        [DETERMINISM, ["--tokens", "192", "--long-lines", "700", "--wide", "--no-logprobs"],
         "greedy"],
        [f"{DIGESTS}/sampled_digest.py", ["--tokens", "192"], "sampled"],
        [f"{DIGESTS}/controls_digest.py", ["--tokens", "96"], "controls"],
        [f"{DIGESTS}/controls_digest.py", ["--tokens", "96", "--scenarios", "prompt-logprobs"],
         "prompt-logprobs"],
        [f"{DIGESTS}/image_digest.py", ["--tokens", "96"], "images"],
        [f"{DIGESTS}/startup_lines.py", [], "startup"],
    ]
    spec = {
        "experiment": EXPERIMENT,
        "run": "retire-conditionals-check",
        "jobs": [
            {"kind": "sync"},
            {"kind": "kernel",
             "bundles": [{"node": "dgx2", "bundle": f"{EXPERIMENT}/bundles/release-tests"}]},
            {"kind": "measure", "profile": "lean", "bench": False, "bracket": False,
             "extras": extras,
             "arms": [{"config": "retire-tp4.json", "label": "retire"},
                      {"config": f"{CANDIDATE}/rc-tp4.json", "label": "candidate6"}]},
        ],
    }
    (HERE / "retire-check.json").write_text(json.dumps(spec, indent=1) + "\n")
    print(f"{head_commit}: {len(package) - deleted} package files, {deleted} deleted, "
          f"{len(others)} test and benchmark files; serving arm {len(mounts) - 1} mounts")


if __name__ == "__main__":
    main()
