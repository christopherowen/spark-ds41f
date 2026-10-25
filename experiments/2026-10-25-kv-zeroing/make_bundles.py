#!/usr/bin/env python3
"""Write this experiment's kernel-lab bundle, serving overlays, arms and lab specs from
the vLLM branch.

usage: make_bundles.py VLLM_REPO

The vLLM branch ``kv-zeroing-poison`` holds three commits on candidate 6's vLLM (BASE,
26382f5cf with the 0489-0490 test patches; package files those of the image
spark-ds41f:19f2c20ed4d6-next-6): POISON (0500, VLLM_KV_CACHE_POISON), FIX (0501,
DSpark draft windows start at CED's replay start) and REMOVAL (0502, DS4.1's caches use
only written slots, so new blocks are not zeroed).

- bundles/poison-tests: the GPU tests of the three commits on candidate 6's image, the
  package files of BASE..REMOVAL mounted over its tree.
- bundles/serving, bundles/serving-zeroing and bundles/serving-unfixed: the package
  files of BASE..REMOVAL, BASE..FIX and BASE..POISON, which the serving arms mount over
  the image's tree.
- Arms (``extends`` candidate 6's own arms; every arm counts NaN logits with
  VLLM_COMPUTE_NANS_IN_LOGITS=1):
  zero-*: 0500-0501, poison off: new blocks zeroed, as in candidate 6 (the reference);
  nofill-*: 0500-0502, poison off: new blocks neither zeroed nor filled (the end
  state); poison-*: nofill with VLLM_KV_CACHE_POISON=1, every new block filled with NaN
  bytes (the gate); *-dev: with the dev-mode RPC for the release suite's engine grid;
  *-pressure: a 262,144-token context and a small KV pool, so 16 (TP3: 8) concurrent
  prompts exhaust it; poison-unfixed-tp4: POISON alone, the positive control.
- Lab specs, one a job (README "Lab spec"): tp4-gate.json, tp4-pressure.json,
  tp4-suite.json, tp4-control.json; tp3-gate.json, tp3-pressure.json, tp3-suite.json.

TP3 arms extend candidate 6's triangle arm (rc6/rc-tp3.json, spark-ds41f PR #582).
Until that PR is on main, candidate 6's files are read from origin/tp3-candidate-6.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import recipe  # noqa: E402

EXPERIMENT = "experiments/2026-10-25-kv-zeroing"
CANDIDATE = "experiments/2026-10-25-spark-ds41f-candidate"
ROOT = "/opt/spark-ds41f/candidate/vllm"
IMAGE = "spark-ds41f:19f2c20ed4d6-next-6"
BRANCH = "kv-zeroing-poison"
BASE, POISON, FIX, REMOVAL = "3bdf9558c", "341928840", "009ad2699", "8d2842981"
TP3_REF = "origin/tp3-candidate-6"  # spark-ds41f PR #582, until it is on main
FIXTURES = REPO / "experiments/2026-10-24-tilelang-completion/bundles/sampling/prepare_fixtures.py"
CACHE_MOUNT = ["{home}/projects/spark-ds41f/cache", "/cache", "rw"]
NODES = "dgx1,dgx2,dgx3,dgx4"

PYTEST = ["-m", "pytest", "-q", "-rfEs", "-p", "no:cacheprovider", "--noconftest",
          "-p", "prepare_fixtures"]
STEPS = [
    ("zeroing, the poison fill and the runner's declared calls", PYTEST + [
        "--basetemp", "/c/pytest-kv-zeroing-fill",
        "tests/v1/worker/test_kv_block_zeroer.py",
        "tests/v1/worker/test_gpu_tilelang_inputs.py"], 3600),
    ("the zeroing rule and the declaration's merges", PYTEST + [
        "--basetemp", "/c/pytest-kv-zeroing-rule",
        "tests/v1/core/test_kv_cache_utils.py", "-k", "zeroing or poison or written_slot"],
     1800),
    ("the ring's recorded blocks and DS4.1's specs", PYTEST + [
        "--basetemp", "/c/pytest-kv-zeroing-specs",
        "tests/v1/core/test_single_type_kv_cache_manager.py",
        "tests/models/test_deepseek_v4_1_compressor.py"], 1800),
    ("window metadata: TileLang against Triton, CED and the draft windows", PYTEST + [
        "--basetemp", "/c/pytest-kv-zeroing-windows",
        "tests/kernels/test_deepseek_v41_tilelang_metadata.py",
        "tests/v1/attention/test_b12x_v41_ced_metadata.py"], 3600),
]

RUN = '''#!/usr/bin/env python3
"""Run this bundle's steps in order; each step's exit status is printed, and the bundle
fails if any step failed. Every step runs whatever came before it."""
import json
import subprocess
import sys

STEPS = json.load(open("/b/steps.json"))
codes = []
for label, argv, timeout in STEPS:
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

# Pool pressure: a 262,144-token context keeps vLLM's one-request startup check below
# these pools (block 227,840 bytes: 4,895 blocks a request at 16,384 batched tokens on
# the ring, 2,975 at 8,192 on the triangle), and every running prompt holds its share.
PRESSURE = {
    "tp4": {"max_model_len": 262144, "kv_cache_memory_bytes": 1342177280},  # 5,890 blocks
    "tp3": {"max_model_len": 262144, "kv_cache_memory_bytes": 805306368},  # 3,534 blocks
}


def git(repo: str, *args: str) -> str:
    return subprocess.run(["git", "-C", repo, *args], check=True, capture_output=True,
                          text=True).stdout


def show(repo: str, ref: str, path: str) -> bytes:
    return subprocess.run(["git", "-C", repo, "show", f"{ref}:{path}"], check=True,
                          capture_output=True).stdout


def read_config(path: str) -> dict:
    """A configuration file of this repository. Until candidate 6's triangle PR is on
    main, candidate 6's files are its (it also sets the triangle's budget in
    cluster.json; the ring's files are unchanged)."""
    if path.startswith(CANDIDATE) and not (REPO / CANDIDATE / "rc6/rc-tp3.json").exists():
        return json.loads(show(str(REPO), TP3_REF, path))
    return json.loads((REPO / path).read_text())


def resolved(path: str, shape: str | None = None) -> dict:
    """bin/spark's read_cluster: an arm over its resolved base, a recipe at its shape."""
    cluster = read_config(path)
    if recipe.is_overlay(cluster):
        base = cluster["extends"]
        return recipe.overlay(cluster, resolved(base["config"], base.get("shape")))
    return recipe.resolve(cluster, shape) if recipe.is_recipe(cluster) else cluster


def changed(repo: str, start: str, end: str) -> list[str]:
    status = [line for line in git(repo, "diff", "--name-status", start, end).split("\n") if line]
    assert all(line.split("\t")[0] in ("A", "M") for line in status), status
    return [line.split("\t")[-1] for line in status]


def write_overlay(repo: str, end: str, name: str) -> list[str]:
    target = HERE / "bundles" / name / "overlay"
    if target.parent.exists():
        shutil.rmtree(target.parent)
    package = [path for path in changed(repo, BASE, end) if path.startswith("vllm/")]
    for path in package:
        out = target / path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(show(repo, end, path))
    return package


def write_tests_bundle(repo: str) -> None:
    bundle = HERE / "bundles" / "poison-tests"
    if bundle.exists():
        shutil.rmtree(bundle)
    (bundle / "overlay").mkdir(parents=True)
    mounts = [["prepare_fixtures.py", "/b/prepare_fixtures.py"],
              ["run.py", "/b/run.py"], ["steps.json", "/b/steps.json"]]
    files = changed(repo, BASE, REMOVAL)
    for path in files:
        if path.startswith("vllm/"):
            target = bundle / "overlay" / path
        else:
            target = bundle / path.replace("/", "_")
            mounts.append([target.name, f"{ROOT}/{path}"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(show(repo, REMOVAL, path))
    shutil.copy(FIXTURES, bundle / "prepare_fixtures.py")
    (bundle / "run.py").write_text(RUN)
    (bundle / "steps.json").write_text(json.dumps(STEPS, indent=1) + "\n")
    candidate = {
        "description": (
            f"VLLM_KV_CACHE_POISON, the DSpark draft-window bound and DS4.1 without "
            f"per-allocation zeroing (vLLM branch {BRANCH} {REMOVAL}, {len(files)} files "
            f"over candidate 6's {BASE}) on candidate 6's image: the GPU tests of the "
            "zeroing and fill kernels, the zeroing rule and DS4.1's specs, the ring's "
            "recorded blocks and the window metadata."),
        "image": IMAGE,
        "env": {
            "HF_HUB_OFFLINE": "1",
            "TILELANG_CACHE_DIR": "/c/tilelang-kv-zeroing",
            "VLLM_DS41_KERNEL_BACKEND": "tilelang",
            "PYTHONPATH": "/b:/opt/spark-ds41f/candidate/vllm:/opt/spark-ds41f/candidate/b12x",
        },
        "workdir": "vllm",
        "argv": ["/b/run.py"],
        "verdict": "exit",
        "mounts": mounts,
    }
    (bundle / "candidate.json").write_text(json.dumps(candidate, indent=1) + "\n")


def mounts(bundle: str, package: list[str]) -> list[list[str]]:
    experiment = "{home}/projects/spark-ds41f/" + EXPERIMENT
    return [CACHE_MOUNT] + [[f"{experiment}/bundles/{bundle}/overlay/{path}", f"{ROOT}/{path}", "ro"]
                            for path in package]


def write_arm(name: str, notes: str, extends: str, sets: dict) -> None:
    arm = {"notes": notes, "extends": {"config": extends}, "set": sets}
    (HERE / f"{name}.json").write_text(json.dumps(arm, indent=1) + "\n")


def pressure_args(base: str, shape: str) -> list[str]:
    args = list(resolved(base)["serve_args"])
    for flag, key in (("--max-model-len", "max_model_len"),
                      ("--kv-cache-memory-bytes", "kv_cache_memory_bytes")):
        args[args.index(flag) + 1] = str(PRESSURE[shape][key])
    return args


def write_arms(final: list[str], zeroing: list[str], unfixed: list[str]) -> None:
    here = lambda name: f"{EXPERIMENT}/{name}.json"  # noqa: E731
    counting = {"VLLM_COMPUTE_NANS_IN_LOGITS": "1"}
    for shape, base, what in (
        ("tp4", f"{CANDIDATE}/rc6/rc-tp4.json", "candidate 6 at tp4-ring"),
        ("tp3", f"{CANDIDATE}/rc6/rc-tp3.json",
         "candidate 6 at tp3-triangle (rc6/rc-tp3.json, PR #582: 8,192 batched tokens)"),
    ):
        write_arm(f"zero-{shape}",
                  f"{what} with vLLM {BRANCH} {FIX} (0500-0501) mounted, the poison off: "
                  "new KV blocks are zeroed as in candidate 6, the reference. Counts NaN "
                  "logits.",
                  base, {"container": {"mounts": mounts("serving-zeroing", zeroing)},
                         "environment": counting})
        write_arm(f"nofill-{shape}",
                  f"{what} with vLLM {BRANCH} {REMOVAL} (0500-0502) mounted, the poison "
                  "off: DS4.1 declares that its caches use only written slots, so new KV "
                  "blocks are neither zeroed nor filled and keep their previous owner's "
                  "bytes (the end state). Counts NaN logits.",
                  base, {"container": {"mounts": mounts("serving", final)},
                         "environment": counting})
        write_arm(f"poison-{shape}",
                  f"nofill-{shape} with VLLM_KV_CACHE_POISON=1: every new KV block, "
                  "compressor rings included, is filled with all-ones bytes (NaN in every "
                  "float format the pages hold) before its first step: the gate.",
                  here(f"nofill-{shape}"), {"environment": {"VLLM_KV_CACHE_POISON": "1"}})
        write_arm(f"poison-{shape}-dev",
                  f"poison-{shape} with the dev-mode collective RPC, for the release suite's "
                  "engine grid.",
                  here(f"poison-{shape}"), {"environment": {"VLLM_SERVER_DEV_MODE": "1"}})
        context, pool = PRESSURE[shape]["max_model_len"], PRESSURE[shape]["kv_cache_memory_bytes"]
        blocks = pool // 227840
        for mode in ("zero", "nofill", "poison"):
            write_arm(f"{mode}-pressure-{shape}",
                      f"{mode}-{shape} with a {context:,}-token context and a {pool:,}-byte KV "
                      f"pool ({blocks:,} blocks of 227,840 bytes), so pool_pressure.py's "
                      "concurrent prompts exhaust it: preemption, eviction of cached prefixes, "
                      "blocks handed between cache groups.",
                      here(f"{mode}-{shape}"), {"serve_args": pressure_args(here(f"{mode}-{shape}"), shape)})
    write_arm("poison-unfixed-tp4",
              f"candidate 6 at tp4-ring with VLLM_KV_CACHE_POISON alone (vLLM {POISON}, 0500 "
              "without 0501), the positive control: DSpark's draft rows of nonfinal prefill "
              "chunks read window slots nobody wrote, now NaN, and the drafter's router "
              "refuses non-finite logits.",
              f"{CANDIDATE}/rc6/rc-tp4.json",
              {"container": {"mounts": mounts("serving-unfixed", unfixed)},
               "environment": {**counting, "VLLM_KV_CACHE_POISON": "1"}})


def gate(shape: str) -> list:
    counters = ["kv_counters.py", ["--nodes", NODES if shape == "tp4" else "dgx1,dgx2,dgx3"],
                "counters"]
    threshold = {"tp4": "16288", "tp3": "8144"}[shape]
    return [
        ["experiments/2026-10-12-r6g/token_determinism_text.py",
         ["--tokens", "192", "--long-lines", "700", "--wide"], "determinism"],
        ["experiments/2026-10-12-r6g/token_determinism_text.py",
         ["--tokens", "192", "--long-lines", "700", "--wide", "--no-logprobs"], "greedy"],
        ["prefix_extension.py", ["--threshold", threshold], "prefix-extension"],
        ["experiments/2026-10-24-tilelang-completion/controls_digest.py", ["--tokens", "96"],
         "controls-digest"],
        ["experiments/2026-10-24-tilelang-completion/sampled_digest.py", ["--tokens", "192"],
         "sampled"],
        counters,
    ]


def write_specs() -> None:
    """One spec a job, each its own run and table; queue them in this order. Only the
    first syncs the node checkouts (origin/main must hold this experiment)."""
    for shape, streams, documents, tokens, max_tokens, suite in (
        ("tp4", "16", "12", "32000", "2048", "release"),
        ("tp3", "8", "6", "48000", "4096", "release-tp3"),
    ):
        nodes = NODES if shape == "tp4" else "dgx1,dgx2,dgx3"
        counters = ["kv_counters.py", ["--nodes", nodes], "counters"]
        threshold = {"tp4": "16288", "tp3": "8144"}[shape]
        gate_jobs = [{"kind": "sync"}]
        if shape == "tp4":
            gate_jobs.append({"kind": "kernel", "bundles": [
                {"node": "dgx2", "bundle": f"{EXPERIMENT}/bundles/poison-tests"}]})
        gate_jobs.append(
            {"kind": "measure", "bench": False, "bracket": False, "extras": gate(shape),
             "arms": [{"config": f"{mode}-{shape}.json", "label": f"{mode}-{shape}"}
                      for mode in ("poison", "nofill", "zero")]})
        specs = {
            "gate": gate_jobs,
            "pressure": [
                {"kind": "measure", "bench": False, "bracket": False, "extras": [
                    ["pool_pressure.py", ["--streams", streams, "--documents", documents,
                                          "--tokens", tokens, "--max-tokens", max_tokens],
                     "pool-pressure"], counters],
                 "arms": [{"config": f"{mode}-pressure-{shape}.json",
                           "label": f"{mode}-pressure-{shape}"}
                          for mode in ("poison", "nofill", "zero")]}],
            "suite": [
                {"kind": "measure", "suite": [suite, "single-stream"],
                 "arms": [{"config": f"poison-{shape}-dev.json", "label": f"poison-{shape}"}]}],
        }
        if shape == "tp4":
            specs["control"] = [
                {"kind": "measure", "bench": False, "bracket": False, "extras": [
                    ["prefix_extension.py", ["--threshold", threshold, "--max-tokens", "16"],
                     "prefix-extension"], counters],
                 "arms": [{"config": "poison-unfixed-tp4.json",
                           "label": "poison-unfixed-tp4"}]}]
        for name, jobs in specs.items():
            run = f"kv-zeroing-{shape}" + ("" if name == "gate" else f"-{name}")
            spec = {"experiment": EXPERIMENT, "run": run, "jobs": jobs}
            (HERE / f"{shape}-{name}.json").write_text(json.dumps(spec, indent=1) + "\n")


def main() -> None:
    repo = sys.argv[1]
    heads = {name: git(repo, "rev-parse", "--short=9", ref).strip()
             for name, ref in (("base", f"{BRANCH}~3"), ("poison", f"{BRANCH}~2"),
                               ("fix", f"{BRANCH}~1"), ("removal", BRANCH))}
    assert heads == {"base": BASE, "poison": POISON, "fix": FIX, "removal": REMOVAL}, heads
    final = write_overlay(repo, REMOVAL, "serving")
    zeroing = write_overlay(repo, FIX, "serving-zeroing")
    unfixed = write_overlay(repo, POISON, "serving-unfixed")
    write_tests_bundle(repo)
    write_arms(final, zeroing, unfixed)
    write_specs()
    print(f"{BRANCH}: {len(final)} package files with 0502, {len(zeroing)} with 0501, "
          f"{len(unfixed)} with 0500 alone; arms and specs written")


if __name__ == "__main__":
    main()
