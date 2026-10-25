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
VISION_TEST = "tests/kernels/test_deepseek_v41_tilelang_vision.py"
# Candidate 6's vision test bounds the TileLang tower by B12X's error on the same inputs;
# the branch deletes B12X's tower, so its test needs those errors as numbers. Each
# comparison of the base test also prints both errors (one JSON line, prefix B12X-ERROR).
PRINT = ('    print("B12X-ERROR", json.dumps({{"check": {check}, "ours": [float(x) for x in ours], '
         '"b12x": [float(x) for x in theirs]}}), flush=True)\n')
VISION_PRINTS = [
    ("    b12x_error, b12x_off, _ = theirs\n", PRINT.format(check="name")),
    ("    theirs = (_relative(theirs_out, ref), _max_ulps(theirs_out, ref, floor))\n",
     PRINT.format(check='f"attention at {rows} rows"')),
    ('    message += f"; B12X {theirs[0]:.3e}, {theirs[1]:.3e}"\n',
     PRINT.format(check='f"tower {vit.config.vision_dim} {height}x{width}"')),
]


def vision_error_bundle(repo: str, base: str) -> None:
    """bundles/vision-b12x-error: candidate 6's vision test on its own image (no overlay),
    printing B12X's error beside the TileLang tower's at every bounded comparison."""
    bundle = HERE / "bundles" / "vision-b12x-error"
    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir(parents=True)
    source = show(repo, base, VISION_TEST).decode()
    source = source.replace("import math\n", "import json\nimport math\n", 1)
    for anchor, line in VISION_PRINTS:
        assert source.count(anchor) == 1, anchor
        source = source.replace(anchor, anchor + line)
    name = "test_deepseek_v41_tilelang_vision_b12x_error.py"
    (bundle / name).write_text(source)
    # The release tests' plugins: the image lacks tblib, which the conftest imports.
    plugins = ("tblib_stub.py", "offline_skip.py")
    for plugin in plugins:
        shutil.copyfile(REPO_ROOT / "scripts/release_tests" / plugin, bundle / plugin)
    candidate = {
        "description": (
            "Candidate 6's TileLang vision test on image -next-6 (its own tree, no overlay), "
            "printing B12X's error beside the TileLang tower's for each FP64 comparison: the "
            "numbers retire-conditionals' test bounds the tower by once B12X's tower is gone."),
        "image": IMAGE,
        "env": {"HF_HUB_OFFLINE": "1", "PYTHONPATH": "/b",
                "TILELANG_CACHE_DIR": "/c/tilelang-vision-b12x-error"},
        "workdir": "vllm",
        "argv": ["-m", "pytest", "-q", "-s", "-rfEs", "-p", "no:cacheprovider",
                 "-p", "tblib_stub", "-p", "offline_skip",
                 f"tests/kernels/{name}", "-k", "against_fp64 or against_reference"],
        "verdict": "exit",
        "mounts": [[name, f"{ROOT}/tests/kernels/{name}"],
                   *([plugin, f"/b/{plugin}"] for plugin in plugins)],
    }
    (bundle / "candidate.json").write_text(json.dumps(candidate, indent=2) + "\n")


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
    vision_error_bundle(repo, base)

    overlay = f"{{home}}/projects/spark-ds41f/{EXPERIMENT}/bundles/release-tests/overlay"
    mounts = [["{home}/projects/spark-ds41f/cache", "/cache", "rw"]]
    mounts += [[f"{overlay}/{path}", f"{ROOT}/{path}", "ro"] for _, path in package]
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
    # One serving check per cabling: the ring (tp4) and the triangle (tp3). The ring's
    # also repeats the release-test gate on dgx2 (it passed on dgx4, the same image).
    for shape, tp, suffix, kernel in (("tp4-ring", "tp4", "", True),
                                      ("tp3-triangle", "tp3", "-tp3", False)):
        arm = {
            "notes": (f"Candidate 6 at {shape} ({CANDIDATE}/rc-{tp}.json) with the vLLM branch "
                      f"retire-conditionals ({head_commit}) mounted over the image's package "
                      "files, its deleted modules failing to import. VLLM_DS41_KERNEL_BACKEND "
                      "stays set: candidate 6's source manifest has the switch, and the "
                      "branch's vLLM no longer reads it. Outputs must equal candidate 6's bit "
                      "for bit."),
            "extends": {"config": f"{CANDIDATE}/rc-{tp}.json"},
            "set": {"container": {"mounts": mounts}},
        }
        (HERE / f"retire-{tp}.json").write_text(json.dumps(arm, indent=1) + "\n")
        jobs = [{"kind": "sync"}]
        if kernel:
            jobs.append({"kind": "kernel", "bundles": [
                {"node": "dgx2", "bundle": f"{EXPERIMENT}/bundles/release-tests"}]})
        jobs.append({"kind": "measure", "profile": "lean", "bench": False, "bracket": False,
                     "extras": extras,
                     "arms": [{"config": f"retire-{tp}.json", "label": f"retire{suffix}"},
                              {"config": f"{CANDIDATE}/rc-{tp}.json",
                               "label": f"candidate6{suffix}"}]})
        spec = {"experiment": EXPERIMENT, "run": f"retire-conditionals-check{suffix}",
                "jobs": jobs}
        (HERE / f"retire-check{suffix}.json").write_text(json.dumps(spec, indent=1) + "\n")
    print(f"{head_commit}: {len(package) - deleted} package files, {deleted} deleted, "
          f"{len(others)} test and benchmark files; serving arm {len(mounts) - 1} mounts")


if __name__ == "__main__":
    main()
